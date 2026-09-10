// Seam flux CORRECTION -- the applying half of the M4 interface flux
// correction (plans/3D.md M3/M4). Fragment only; the entry files list every
// include. Runs last in the macro step, after `average`.
//
// common_d3_amr_flux.wgsl accumulated (Fout - Fin)/8, the mass and momentum
// the FINE solver actually moved across the seam over both substeps. This
// pass computes what the COARSE solver thought it moved and corrects the
// difference onto the unrefined coarse cells.
//
// THE COARSE SIDE. The coarse step is globally conservative -- streaming
// permutes populations, BGK preserves the first two moments -- so over one
// macro step the unrefined set changes by exactly
//
//   coarse_out - coarse_in
//     coarse_out = what an unrefined cell PULLED from a covered cell
//                  (sum over i where c - e_i is covered, of f_i(c - e_i))
//     coarse_in  = what a covered cell pulled from THIS cell
//                  (sum over i where c + e_i is covered, of f_i(c))
//
// both read from the coarse field at time t, which is still intact: the step
// wrote the other ping-pong buffer. It SHOULD have changed by (Fout-Fin)/8,
// so the correction is
//
//   delta = fluxAcc + (coarse_in - coarse_out)
//
// and applying it makes the composite exactly conservative in mass and in
// momentum. That the covered cells' own post-step values are then discarded
// by `average` does not enter the accounting: the composite is (unrefined
// coarse) + (fine region), and the covered cells are the restriction's
// shadow of the second.
//
// HOW A MOMENT CORRECTION IS APPLIED to a population set:
//
//   f_i += w_i * (d_rho + 3 e_i . d_mom)
//
// Sum over i gives d_rho exactly (sum w_i = 1, sum w_i e_i = 0) and the
// first moment gives d_mom exactly (sum w_i e_ia e_ib = cs2 delta_ab, and
// 3*cs2 = 1). It adds nothing to the second moment at first order, which is
// right -- the correction is a flux imbalance, not a stress. It is applied
// POST-collision, so the corrected mass does not itself collide this step;
// that is an O(dx^2 dt) difference in the stress it induces and second order
// either way, whereas substituting a population inside the gather would mean
// unfusing the stream+collide kernel.
//
// WHY THIS CAN BE A SEPARATE PASS AT ALL. Earlier drafts had the coarse
// gather read fine-derived populations, which forces the pass order and puts
// a divergent branch and an extra read in the kernel sec 2.4 measured at 77%
// of device peak. Correcting moments afterwards needs neither: the coarse
// step stays refinement-blind and byte-identical, and this pass is a SURFACE
// (O(N^2) against the grid's O(N^3)).
//
// NOT VALID ACROSS A SOLID OR A WALL. The accounting above assumes every
// coarse population goes exactly one place; bounce-back reflects instead, so
// the balance does not close where a seam cell's gather also hits the body.
// Cells within BODY_CLEAR of the surface are therefore SKIPPED -- they keep
// M3's non-conservative behaviour rather than getting a wrong correction --
// and main-3d.js refuses ?reflux=1 outright on a walled scenario. In the
// geometry this is for (a refined shell wrapping a body) the seam sits in
// clean fluid and nothing is skipped; a seam deliberately cutting the solid
// is follow-on work and has no gate today.

@group(0) @binding(0) var<storage, read>       f_in      : array<f32>;   // coarse at t
@group(0) @binding(1) var<storage, read_write> f_out     : array<f32>;   // coarse at t + dt
@group(0) @binding(2) var<storage, read_write> mac       : array<f32>;
@group(0) @binding(3) var<storage, read>       blockSlot : array<i32>;
@group(0) @binding(4) var<storage, read_write> fluxAcc   : array<f32>;
@group(0) @binding(5) var<storage, read>       body      : BodyState3D;

override HAS_BODY : u32 = 0u;
override FORCE_X : f32 = 0.0f;
override FORCE_Y : f32 = 0.0f;
override FORCE_Z : f32 = 0.0f;
// Clearance from the body surface, in coarse cells, below which a seam cell
// is left uncorrected. A coarse gather reaches one cell and the fine cells
// involved are half that, so 2 is comfortable rather than marginal.
override BODY_CLEAR : f32 = 2.0f;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY || gid.z >= NZ) { return; }
  let c = vec3<u32>(gid.x, gid.y, gid.z);
  let cell = coarseCell(c);
  let ncells = NX * NY * NZ;

  let myBlock = vec3<u32>(c.x / RB, c.y / RB, c.z / RB);
  if (blockSlot[blockIdOf(myBlock)] >= 0) { return; }

  // The SAME reachability predicate common_d3_amr_flux.wgsl uses, and that
  // is load-bearing: the flux pass returns early on cells that fail it and
  // therefore leaves fluxAcc stale there, so this pass must never read one.
  // Refinement is static within a macro step (the manager runs at macro-step
  // boundaries), so the two agree by construction.
  let nb = vec3<u32>(nbx(), nby(), nbz());
  var off = vec3<i32>(0, 0, 0);
  off.x = select(select(0, 1, (c.x + 1u) % RB == 0u), -1, c.x % RB == 0u);
  off.y = select(select(0, 1, (c.y + 1u) % RB == 0u), -1, c.y % RB == 0u);
  off.z = select(select(0, 1, (c.z + 1u) % RB == 0u), -1, c.z % RB == 0u);
  var anyRefined = false;
  for (var k = 0u; k < 8u; k++) {
    let s = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u)) * off;
    let b = vec3<u32>(
      wrapu(i32(myBlock.x) + s.x, nb.x),
      wrapu(i32(myBlock.y) + s.y, nb.y),
      wrapu(i32(myBlock.z) + s.z, nb.z));
    if (blockSlot[blockIdOf(b)] >= 0) { anyRefined = true; }
  }
  if (!anyRefined) { return; }

  if (HAS_BODY != 0u) {
    let p = vec3<f32>(f32(c.x), f32(c.y), f32(c.z));
    if (get_phi3(p, body) < BODY_CLEAR) { return; }
  }

  var dm = fluxAcc[4u * cell + 0u];
  var dp = vec3<f32>(fluxAcc[4u * cell + 1u], fluxAcc[4u * cell + 2u], fluxAcc[4u * cell + 3u]);

  for (var i = 0u; i < QN; i++) {
    let e = vec3<i32>(ex[i], ey[i], ez[i]);
    let ef = vec3<f32>(f32(e.x), f32(e.y), f32(e.z));

    // coarse_out: subtract what this cell pulled out of the refined region.
    // i = 0 needs no special case -- e is zero, so both lookups land on this
    // cell's own block, which the test above established is unrefined.
    let s = vec3<u32>(wrapu(i32(c.x) - e.x, NX), wrapu(i32(c.y) - e.y, NY), wrapu(i32(c.z) - e.z, NZ));
    if (blockSlot[blockIdOf(vec3<u32>(s.x / RB, s.y / RB, s.z / RB))] >= 0) {
      let v = f_in[i * ncells + coarseCell(s)];
      dm -= v; dp -= v * ef;
    }

    // coarse_in: add back what the refined region pulled out of this cell.
    let t = vec3<u32>(wrapu(i32(c.x) + e.x, NX), wrapu(i32(c.y) + e.y, NY), wrapu(i32(c.z) + e.z, NZ));
    if (blockSlot[blockIdOf(vec3<u32>(t.x / RB, t.y / RB, t.z / RB))] >= 0) {
      let v = f_in[i * ncells + cell];
      dm += v; dp += v * ef;
    }
  }

  // The APPLIED correction, left where the flux measurement was, for
  // window.__D3.readFluxAcc(). The next macro step's substep-A flux pass
  // overwrites it (ACCUM=0), so this costs a store and no state.
  fluxAcc[4u * cell + 0u] = dm;
  fluxAcc[4u * cell + 1u] = dp.x;
  fluxAcc[4u * cell + 2u] = dp.y;
  fluxAcc[4u * cell + 3u] = dp.z;

  var rho = 0f;
  var m = vec3<f32>(0f);
  for (var i = 0u; i < QN; i++) {
    let ef = vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
    let fo = f_out[i * ncells + cell] + wt[i] * (dm + 3.0f * dot(ef, dp));
    f_out[i * ncells + cell] = fo;
    rho += fo;
    m += fo * ef;
  }

  // Republish the macroscopic field so the renderer and every readback see
  // the corrected cell rather than the coarse step's uncorrected one. Same
  // Guo half-step the solver's own macroscopic output uses, so a forced
  // scenario stays consistent; it folds out at zero force.
  let rhoDen = max(rho, 1e-6f);
  let u = m / rhoDen + vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z) / (2.0f * rhoDen);
  mac[4u * cell + 0u] = rho;
  mac[4u * cell + 1u] = u.x;
  mac[4u * cell + 2u] = u.y;
  mac[4u * cell + 3u] = u.z;
}
