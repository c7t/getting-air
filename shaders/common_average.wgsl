// THE fine -> parent averaging (restriction) kernel, whole. Its entry file --
// amr_average_pool_parent.wgsl -- is its bindings, its overrides, and two
// @include lines. There is one implementation of the restriction, here.
// plans/2D-backport.md B3-2.
//
// THERE WERE TWO ENTRY FILES until U7-6f, the other being amr_average_f2c.wgsl
// with the dense L0 grid as the parent, and B3-2 records what the two
// near-identical 142/147-line copies actually differed in: the parent's TAU
// (L0's own `state.tau` vs. this level's `levelParams.parentTau`) and WHERE
// the averaged cell is written (a dense `cellIndex()` address at plane stride
// W*H vs. an (lx,ly) inside the parent's own FB*FB tile). Nothing else -- the
// restriction math below was character-for-character the same, which is what
// made merging them safe and what made deleting one of them cheap.
//
// So the split is an ACCESSOR, exactly as plans/2D-backport.md B3 describes
// for 3D's `common_d3_parent_{dense,pool}.wgsl`: each parent kind supplies
//
//   fn parentTau() -> f32
//   fn parentCellForChild(slot: u32, blockID: i32, lcx: u32, lcy: u32) -> u32
//   fn parentStoreWord(cell: u32, wi: u32, word: u32)
//
// and nothing else. The dense pair reads `state` and writes `f_coarse`; the
// pool pair reads `levelParams` and writes `f_parent_pool`. THE ASYMMETRY IS
// REAL AND STAYS (L0 is a dense, ghost-free, cellIndex()-addressed grid --
// plans/AMR-multilevel.md decision 1); the point is that it now costs three
// small functions instead of a second copy of the kernel.
//
// It is a fragment, so per shader-loader.mjs it must not itself @include. It
// depends on `feqD2Q9`/`ex`/`ey` (common_lattice.wgsl) and on the fPack family
// (common_fpack.wgsl); every entry file that includes this one includes both.
// WGSL module-scope resolution is order-independent, so the @include lines --
// and the parent fragment's three functions -- may appear in any order.
//
// Dispatched as (1, 1, MAX_FINE_BLOCKS) with workgroup_size(8,8): a block is
// exactly RB*RB = 8*8 = 64 coarse-equivalent cells, i.e. exactly one
// workgroup, so each thread handles exactly one parent cell's 4 fine
// children -- no separate tile-index dimension (unlike interp/step1, which
// iterate over the larger FBxFB slot buffer).
//
// rho: simple arithmetic mean of the 4 children (exactly mass-conservative).
// Velocity: mass-weighted average (momentum-conservative). Non-equilibrium
// part: inverse-Dupuis-Chopard-rescaled by dcRescaleFineToCoarse below (see
// common_interp.wgsl's `dcRescaleCoarseToFine` for the forward direction).

override RB : u32;

// ── ?dcpre=1 -- the legacy PRE-collision Dupuis-Chopard factor ──────────────
// See common_interp.wgsl's DC_PRE for the derivation; this is the same switch
// on the fine->coarse (restriction) side, and the two forms below are EXACT
// inverses of the two there. Default 0 = post-collision = correct.
override DC_PRE : u32 = 0u;

// Inverse of common_interp.wgsl's dcRescaleCoarseToFine.
//
// The PRE-collision form is (tau_coarse/tau_fine) * (dx_coarse/dx_fine) =
// (tau_coarse/tau_fine) * n with n=2 -- the per-cell velocity gradient
// doubles going to the coarser grid, so fneq is scaled up by n. The
// POST-collision form is what this solver actually needs, because both sides
// of this transfer hold f after collision (the step kernel is a fused
// pull-stream + collide). amr2d.mjs's dcRescaleFineToCoarse is the host
// statement; tools/test-amr2d.js asserts the two directions multiply to 1 to
// within one ulp, which is the property that makes a round trip through the
// interface inert.
fn dcRescaleFineToCoarse(tauCoarse: f32, tauFine: f32) -> f32 {
  if (DC_PRE != 0u) { return 2.0f * tauCoarse / tauFine; }
  return 2.0f * (tauCoarse - 1.0f) / (tauFine - 1.0f);
}
// ── Measurement instrument: ?benchSkip=<group>-noop ──────────────────────────
// Returns before touching any buffer, so the pass is still encoded and
// dispatched at full width but does no work. Skipping the pass ENTIRELY vs.
// running this no-op variant separates the fixed per-pass cost (encode,
// dispatch, pipeline switch, barrier) from the work the pass actually does --
// a split the plain ?benchSkip= groups cannot make, because removing a pass
// removes both at once.
//
// Measured 2026-09-07, desktop RTX 4080, res=8 levels=3 blockage=3.3, via
// tools/bench-amr.js --skip. Share of frame GPU time recovered:
//
//   group    pass removed   dispatched as no-op   -> work
//   ghost    15.5-18.5%     1.7-4.3%                 ~13%
//   interp   17.3%          2.0%                     ~15%
//   avg      15.9%          5.6%                     ~10%
//
// So AMR coupling costs its WORK, not its pass count, and fusing coupling
// passes is not a lever -- the same verdict plans/perf-characterization.md
// reached for the force pass by a different route. Default 0 is byte-identical
// to having no instrument at all (an override constant, folded at pipeline
// creation), matching how ?f16=0 is kept in the tree.
override NOOP : u32 = 0u;

const GHOST = 2u;

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) { averageCell(lid, wgid.z); }

// ?indirect=1 (main-amr.js): launched over the CHILD pool's active-slot list
// (amr_active_list.wgsl), so z indexes the list rather than the pool. `main`
// never reads `activeSlots`, so its layout -- every other page's -- is unchanged.
// `activeSlots` is declared by the includer (amr_average_pool_parent.wgsl).
@compute @workgroup_size(8, 8)
fn mainIndirect(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) { averageCell(lid, activeSlots[wgid.z]); }

fn averageCell(lid: vec3<u32>, slot: u32) {
  if (NOOP != 0u) { return; } // see the NOOP override above
  let lcx = lid.x; let lcy = lid.y; // parent-cell-local coords within the block

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  let FB = RB * 2u + 2u * GHOST;
  let poolPlaneStride = arrayLength(&f_pool) / 9u;

  // The 4 fine children of this parent cell, in slot-local (ghost-inclusive)
  // coords.
  let fx0 = GHOST + 2u * lcx; let fx1 = fx0 + 1u;
  let fy0 = GHOST + 2u * lcy; let fy1 = fy0 + 1u;
  let children = array<u32, 4>(
    slot * (FB * FB) + fy0 * FB + fx0, slot * (FB * FB) + fy0 * FB + fx1,
    slot * (FB * FB) + fy1 * FB + fx0, slot * (FB * FB) + fy1 * FB + fx1
  );

  var rho_sum = 0f;
  var rhou_x_sum = 0f; var rhou_y_sum = 0f;
  var f_children: array<array<f32, 9>, 4>;
  var rho_children: array<f32, 4>;
  var ux_children: array<f32, 4>;
  var uy_children: array<f32, 4>;

  for (var c = 0u; c < 4u; c++) {
    let cell = children[c];
    var rho = 0f; var ux = 0f; var uy = 0f;
    var f: array<f32, 9>;
    for (var i = 0u; i < 9u; i++) {
      f[i] = fUnpack(f_pool[fIdx(i, poolPlaneStride, cell)], i);
      rho += f[i];
      ux  += f[i] * f32(ex[i]);
      uy  += f[i] * f32(ey[i]);
    }
    ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f); // NaN-containment floor
    f_children[c] = f;
    rho_children[c] = rho; ux_children[c] = ux; uy_children[c] = uy;
    rho_sum += rho;
    rhou_x_sum += rho * ux;
    rhou_y_sum += rho * uy;
  }

  let rho_avg = rho_sum * 0.25f;
  let rho_sum_den = max(rho_sum, 1e-6f); // NaN-containment floor
  let ux_avg = rhou_x_sum / rho_sum_den;
  let uy_avg = rhou_y_sum / rho_sum_den;

  // The parent's own tau, which is NOT L0's at every level -- see the
  // accessor note in this file's header, and common_interp.wgsl's
  // dcRescaleCoarseToFine for the same point in the forward direction.
  let tau_coarse = parentTau();
  let tau_fine = 2.0f * tau_coarse - 0.5f;
  let rescale = dcRescaleFineToCoarse(tau_coarse, tau_fine);

  var fneq_avg: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    var s = 0f;
    for (var c = 0u; c < 4u; c++) {
      s += f_children[c][i] - feqD2Q9(rho_children[c], ux_children[c], uy_children[c], i);
    }
    fneq_avg[i] = s * 0.25f;
  }

  let parentCell = parentCellForChild(slot, blockID, lcx, lcy);

  var fo: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    fo[i] = feqD2Q9(rho_avg, ux_avg, uy_avg, i) + rescale * fneq_avg[i];
  }
  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    parentStoreWord(parentCell, wi, fPack(fo[fLo(wi)], fo[fHi(wi)], wi));
  }
}
