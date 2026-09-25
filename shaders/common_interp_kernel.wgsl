// THE coarse -> fine ghost-cell interpolation kernel, whole. Its entry file --
// amr_interp_pool_parent.wgsl -- is its bindings, its overrides, and its own
// parent fragment. plans/2D-backport.md B3-3.
//
// THERE WERE TWO until U7-6f, the other being amr_interp_dense_parent.wgsl
// with the dense L0 grid as the parent. The accessor split below is what
// survives of that: the root is a pool tile like any other parent now, but it
// is a RINGLESS one (PARENT_GHOST 0), so the fetch still varies.
//
// NOT TO BE CONFUSED WITH common_interp.wgsl, which is the BLEND
// (`CoarseSample`, `dcRescaleCoarseToFine`, `interpCoarseToFine`) and is shared
// more widely than this pair -- the ghost-free work (plans/ghost-free.md)
// needs the STEP kernels to run that same reconstruction inline. This file is
// the kernel around it; that one is the math inside it. Hence the two names.
//
// THE ACCESSOR, same shape as common_average.wgsl's. Each parent kind supplies
//
//   fn parentTau() -> f32
//   fn levelNbx() -> u32
//   fn levelNby() -> u32
//   fn parentOrigin(slot: u32, bx: u32, by: u32) -> vec2<u32>
//   fn sampleParent(slot: u32, bx: u32, by: u32, ix: i32, iy: i32) -> CoarseSample
//
// and nothing else. `parentOrigin`/`sampleParent` are a PAIR and must agree on
// a frame: the dense half works in coarse BUFFER coordinates over the whole
// periodic domain (origin = bx*RB, sample wraps and goes through cellIndex());
// the pool half works in PARENT-LOCAL INTERIOR coordinates inside one tile
// (origin = the quadrant's own 0-or-RB offset, sample is a +PARENT_GHOST shift
// into that tile). Only the two together are meaningful, which is why they are
// one fragment rather than two knobs.
//
// BOTH TAKE (bx, by), and the dense half ignores them. They were added for
// plans/uniform-levels.md U5-1: a ROOT parent has no ring, so a stencil tap
// that leaves the parent tile has to be resolved against the NEIGHBOURING root
// tile, and finding that tile needs the child's own block coordinates. Passing
// them to only one of the pair would split a frame that must not be split --
// the same argument that keeps origin and sample in one file.
//
// Everything else -- the dispatch shape, the GHOST_ONLY/FINE_FINE_ONLY modes,
// and the whole same-level fine-fine + diagonal-corner consultation -- is a
// same-level concern, indifferent to how the parent hop works, and was already
// character-for-character identical in the two files.
//
// Dispatched over (tileX, tileY, slot) -- the Z dimension selects pool slot,
// so cost scales with pool CAPACITY, not domain size (see plans/AMR.md's
// Milestone 4 design note on why this dispatch shape is the one part not worth
// simplifying away).
//
// It is a fragment, so per shader-loader.mjs it must not itself @include. It
// depends on common_lattice.wgsl, common_fpack.wgsl and common_interp.wgsl;
// every entry file that includes this one includes all three.

override RB : u32;  // identical at every level (decision 2) -- refine block size in coarse-equivalent units
// GHOST_ONLY selects between two compiled pipelines from one module:
// GHOST_ONLY=1 (steady-state, every macro-step) only re-interpolates the ghost
// border. GHOST_ONLY=0 (one-time, on activation) fills the WHOLE slot
// including the "real" interior, needed because a freshly-activated slot has
// no prior fine-level state to evolve from.
override GHOST_ONLY : u32;
// When set, this pass does ONLY the fine-fine (edge + diagonal-corner) ghost
// copy below and returns -- it never falls through to coarse interpolation and
// never touches coarse-adjacent ghosts. Used for the between-substep fine-fine
// re-exchange (see main-amr.js): the once-per-macro-step interp fills fine-fine
// seams exactly for the FIRST fine substep only; without a refresh, the SECOND
// substep streams a stale/edge-clamped neighbor halo, injecting a periodic
// error at every fine-fine seam once per macro-step (the block-pitch artifact).
// Coarse-adjacent ghosts intentionally keep their "degrade over 2 substeps"
// behavior (correct multi-rate coupling; coarse is quasi-static), so this mode
// leaves them alone. Default 0u so the existing interp/interpInit pipelines,
// which don't set it, still compile.
//
// LEGACY (?ghostcopy=1) as of 2026-09-08. This MODE builds the between-substep
// fine-fine ghost COPY pass, and the default build no longer encodes it -- the
// fine step reaches into the neighbour tile itself during streaming instead
// (DIRECT_GHOST in shaders/amr_step1.wgsl), which is both cheaper and fresher.
// Kept so the two paths can be A/B'd for speed and physics in one build; see
// plans/perf-characterization.md's "The one lead left".
//
// NOTE this is the mode, not the branch. The same-level consultation inside
// the ordinary GHOST_ONLY pass below still runs in the default build, and must:
// a fine tile's ghost ring is still read by its OWN child's bilinear parent
// sampling (the stencil reaches [-GHOST, ...]), so it still has to hold the
// exact neighbour value there rather than a coarse guess. Removing that branch
// was measured at 1.8-3.9%, not the ~15% an earlier reading of
// plans/perf-characterization.md claimed.
override FINE_FINE_ONLY : u32 = 0u;
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

// Fine ghost-local coordinate -> position in the PARENT's own units, in
// whichever frame parentOrigin/sampleParent agree on (see this file's header).
// Cell-centred refinement: the two children of parent cell c sit at c -/+ 1/4
// of the parent cell, so tile-local fine index j maps to
// origin - 0.25 + 0.5*(j - GHOST). Signed, because ring cells continue the
// same line past both ends. amr2d.mjs's fineToCoarseUnit is the host twin.
fn fineToCoarseUnit(fCoord: u32, origin: u32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { interpCell(gid, gid.z); }

// ?launch=stride (main-amr.js): a DIRECT dispatch of K workgroups in z, each
// walking this pool's active-slot list (amr_active_list.wgsl) at stride K,
// so an empty slot is never launched. K is the host's lagged estimate of the
// count and only sets the parallelism; the loop covers every entry whatever
// it is. The barrier lets the body reuse workgroup memory. `main` never reads
// `activeList`, so its layout -- every other page's -- is unchanged.
@compute @workgroup_size(8, 8)
fn mainStride(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
              @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let n = activeListCount(li);
  for (var z = wid.z; z < n; z += nwg.z) { interpCell(gid, activeList.slots[z]); workgroupBarrier(); }
}

fn interpCell(gid: vec3<u32>, slot: u32) {
  if (NOOP != 0u) { return; } // see the NOOP override above
  let fx = gid.x; let fy = gid.y;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; } // slot not currently assigned

  let isInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;
  if (isInterior && GHOST_ONLY != 0u) { return; }
  // Milestone 4b: init mode only fills genuinely-new slots (see the
  // newlyActivated binding in each entry file) -- an already-active slot
  // reaching this pipeline (only possible if callers dispatch it too broadly)
  // must not be touched.
  if (GHOST_ONLY == 0u && newlyActivated[slot] == 0u) { return; }

  // This level's own logical (bx,by). blockID already encodes by*nbx+bx by
  // construction at every level, so no cached ownBX/ownBY array is needed
  // (a redundant extra buffer M5's first draft allocated and M6 removed --
  // see plans/AMR-multilevel-M5.md's amendment note).
  let nbx = levelNbx();
  let nby = levelNby();
  let bx = u32(blockID) % nbx;
  let by = u32(blockID) / nbx;

  // Milestone 4c: fine-fine ghost consultation. Only in the steady-state
  // pass (GHOST_ONLY=1) -- in the init pass (GHOST_ONLY=0) two edge-adjacent
  // blocks can both be newly-activated in the SAME dispatch, and this pass
  // also fills their interiors (see the isInterior/GHOST_ONLY skip above),
  // so reading a neighbor's interior there would race its own write.
  // Steady-state has no such race: interior cells are only ever written by
  // the PREVIOUS macro-step's step1PL (this pass returns early on interior
  // cells, see above), so a neighbor's interior is always stable data
  // regardless of dispatch order across slots/workgroups.
  //
  // A ghost cell lies along exactly one edge (not a diagonal corner) when
  // exactly one of fx/fy is out of the interior range. Diagonal corners
  // (both out of range) consult the diagonal neighbor block the same way --
  // found to matter in practice, not just a completeness nicety: a corner
  // still on the old coarse-interpolated path, sitting between two edges
  // now made exact by the fine-fine consultation below, goes measurably
  // inconsistent with its (now-precise) edge neighbors. Since the D2Q9
  // stencil streams diagonally too, that inconsistency feeds directly into
  // the interior cell nearest the corner every fine substep and grows
  // there until it diverges -- confirmed directly: a controlled 3x3
  // refined cluster seeded with a smooth analytic (Taylor-Green) field and
  // no card/turbulence involved still blew up within ~2800 macro-steps,
  // bisected to a single corner cell (the one corner of a cluster-corner
  // block that happens to have an active diagonal neighbor) climbing from
  // ~0.55 to over 1 before cascading to NaN.
  let RB2 = RB * 2u;
  let inXRange = fx >= GHOST && fx < GHOST + RB2;
  let inYRange = fy >= GHOST && fy < GHOST + RB2;
  if (GHOST_ONLY != 0u) {
    var neighborSlot = -1;
    var nfx = fx; var nfy = fy;
    if (inYRange && fx < GHOST) {
      neighborSlot = blockSlot[by * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2;
    } else if (inYRange && fx >= GHOST + RB2) {
      neighborSlot = blockSlot[by * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2;
    } else if (inXRange && fy < GHOST) {
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + bx];
      nfy = fy + RB2;
    } else if (inXRange && fy >= GHOST + RB2) {
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + bx];
      nfy = fy - RB2;
    } else if (fx < GHOST && fy < GHOST) {
      // NW diagonal corner.
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2; nfy = fy + RB2;
    } else if (fx >= GHOST + RB2 && fy < GHOST) {
      // NE diagonal corner.
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2; nfy = fy + RB2;
    } else if (fx < GHOST && fy >= GHOST + RB2) {
      // SW diagonal corner.
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2; nfy = fy - RB2;
    } else if (fx >= GHOST + RB2 && fy >= GHOST + RB2) {
      // SE diagonal corner.
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2; nfy = fy - RB2;
    }

    if (neighborSlot >= 0) {
      let poolPlaneStride = arrayLength(&f_pool) / 9u;
      let poolCellBase = slot * (FB * FB) + fy * FB + fx;
      let neighborCellBase = u32(neighborSlot) * (FB * FB) + nfy * FB + nfx;
      // Word-for-word copy: valid in both layouts, and under F16 it moves
      // 5 words per cell instead of 9.
      let nw = fWords();
      for (var wi = 0u; wi < nw; wi++) {
        f_pool[wi * poolPlaneStride + poolCellBase] = f_pool[wi * poolPlaneStride + neighborCellBase];
      }
      return;
    }
  }

  // Fine-fine-only mode: this ghost cell has no active fine neighbor (the copy
  // above didn't fire), so there is nothing to refresh -- leave it exactly as
  // the previous fine substep produced it (coarse-adjacent ghosts keep their
  // intended multi-rate "degrade over 2 substeps" behavior) and do NOT run the
  // coarse interpolation below.
  if (FINE_FINE_ONLY != 0u) { return; }

  // The parent hop, and the only part the two entry files disagree about.
  let origin = parentOrigin(slot, bx, by);
  let px = fineToCoarseUnit(fx, origin.x);
  let py = fineToCoarseUnit(fy, origin.y);

  let x0 = i32(floor(px)); let x1 = x0 + 1;
  let y0 = i32(floor(py)); let y1 = y0 + 1;
  let tx = px - f32(x0);
  let ty = py - f32(y0);

  let s00 = sampleParent(slot, bx, by, x0, y0);
  let s10 = sampleParent(slot, bx, by, x1, y0);
  let s01 = sampleParent(slot, bx, by, x0, y1);
  let s11 = sampleParent(slot, bx, by, x1, y1);

  // Bilinear blend + Dupuis-Chopard rescale: common_interp.wgsl.
  // f_pool is direction-major across the WHOLE pool (matching the coarse
  // f_coarse convention): plane stride = MAX_FINE_BLOCKS*FB*FB, derived via
  // arrayLength instead of a separate override (the buffer's actual size
  // already encodes it).
  let poolPlaneStride = arrayLength(&f_pool) / 9u;
  let poolCellBase = slot * (FB * FB) + fy * FB + fx;
  var fo: array<f32,9> = interpCoarseToFine(s00, s10, s01, s11, tx, ty, parentTau());
  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    f_pool[wi * poolPlaneStride + poolCellBase] = fPack(fo[fLo(wi)], fo[fHi(wi)], wi);
  }
}
