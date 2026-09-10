// Coarse (L0, dense) -> fine (L1 pool) interpolation. plans/3D.md M3.
// Fragment only; the entry files list every include.
//
// TWO PIPELINES FROM ONE MODULE, selected by GHOST_ONLY:
//   1  steady state, every parent step: re-interpolate the GHOST RING only.
//      Interior cells are the fine solver's own evolved state and must not
//      be overwritten with a by-now-stale coarse interpolation.
//   0  activation / reset: fill the WHOLE tile, interior included, because
//      there is no evolved state yet.
//
// WHY THE RING STILL NEEDS FILLING AT ALL, given DIRECT_GHOST. After
// neighbour-addressed streaming, the ring serves exactly ONE case: a tile at
// a coarse/fine INTERFACE has no same-level neighbour in some direction, so
// the gather has nowhere to go and reads the ring instead. Ring cells whose
// owning tile does exist are never read by the step kernel -- see the
// KNOWN LEAD at the bottom of this header.
//
// ACCURACY NOTE, stated rather than buried: the ring is interpolated ONCE
// per parent step, at the parent's time t, and is then used by BOTH fine
// substeps. Substep B therefore sees interface data that is half a coarse
// step stale. That is the same first-order interface treatment the 2D
// solver uses (plans/3D.md sec 2.1: the parent's state does not change
// between the child's A and B, so re-interpolating would produce identical
// values), and GHOST=2 is what lets the ring self-advance far enough for
// substep B's gather to remain well-defined.
//
// KNOWN LEAD, not taken: this pass fills EVERY ring cell, but the step
// kernel only ever reads the ones with no same-level owner. At RB=4 the ring
// is 70% of a tile (1216 of 1728 cells) and each ring cell costs 8 coarse
// cell reads of QN planes, so skipping the owned ones is potentially large.
// It needs the same blockSlot lookup the step kernel already does, so it is
// cheap to add -- deliberately not added here because M3 is a correctness
// milestone and plans/perf-characterization.md's standing lesson is that
// optimizations in this codebase get measured, not assumed.

// The coarse field at the START and END of the parent step. See TIME_BLEND.
@group(0) @binding(0) var<storage, read>       f_coarse_t0 : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_pool      : array<f32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(3) var<storage, read>       f_coarse_t1 : array<f32>;
// Per SLOT: 1 while the slot holds a tile the manager has just handed out
// and nothing has initialized. READ-ONLY here, deliberately: clearing it
// from this kernel is a race. Thread (0,0,0) of a tile would clear the flag
// while its siblings had not yet reached their NEW_ONLY test, and every one
// that read it afterwards would return early -- a tile filled in part and
// left in part as whatever the previous owner had. Threads in a dispatch are
// not ordered, so "one thread does it last" is not a thing that can be said.
// common_d3_manage.wgsl's `clearNew` pass does it afterwards instead.
// Always bound -- a single dummy element when there is no dynamic
// refinement, exactly as common_d3_step.wgsl binds blockSlot -- so this
// layout does not fork. NEW_ONLY folds it out.
@group(0) @binding(4) var<storage, read>       slotNew     : array<u32>;

override TAU_COARSE : f32 = 0.8f;
override GHOST_ONLY : u32 = 1u;

// M4.2b-ii, the FILL pipeline: GHOST_ONLY = 0 and NEW_ONLY = 1 fills only
// the tiles the manager has just allocated, and clears their flag. The whole
// point of a third pipeline over this same module rather than a new shader
// is that the coarse->fine transfer stays in ONE place: a newly-refined tile
// and a ring refresh want the identical interpolation, and a second copy of
// it is how the two would drift.
//
// THE DUPUIS-CHOPARD RESCALE IS CORRECT HERE, and this is worth stating
// because M4.2b-i's own plan note said the opposite. Chen's explode/coalesce
// needs NO rescale, but that is a statement about the INTERFACE accounting:
// there the same state is moved between two bookkeepings of one volume, and
// the a = (n-1)/2n offset absorbs the tau difference. A tile being created
// or destroyed is an ordinary GRID TRANSFER -- two grids with different tau
// must represent the same rho, u and viscous stress, and fneq carries the
// stress, so it scales by (tau_f - 1)/(tau_c - 1)/2. Skipping it here would
// give a new tile the wrong stress at birth.
override NEW_ONLY : u32 = 0u;

// Where in the parent step this ring refresh is for: 0 = the parent's state
// at time t, 0.5 = halfway. DEFAULT 0, AND CURRENTLY ALWAYS 0.
//
// This exists because of a hypothesis that was tested and rejected during
// M3 -- and the rejection has since been RE-OPENED, which is why the knob
// is still here.
//
// The hypothesis: interpolating the ring once per parent step and using it
// for both fine substeps leaves substep B looking at interface data half a
// coarse step stale. A second, time-blended refresh before substep B was
// built and measured and made the seam error slightly WORSE (nearOut
// 3.24e-2 -> 3.93e-2 at t=64), so staleness was recorded as not the cause.
//
// That measurement was taken on top of a sign-flipped interface stress --
// the grid transfers were applying the PRE-collision Dupuis-Chopard factor
// to POST-collision populations (common_d3_pool.wgsl derives the correct
// one; it is -0.25 here where the code used +0.6875). With that fixed the
// seam error fell ~5x, and the staleness experiment has NOT been repeated
// against the corrected interface. Treat it as untested, not as rejected.
//
// The prior for it staying rejected is still decent: the ring SELF-ADVANCES
// through a real fine LBM step between the substeps (GHOST=2 is what pays
// for that), so substep B sees an EVOLVED ring rather than a stale one, and
// re-interpolating would replace a fine-level evolution with a coarse-level
// guess. What remains genuinely open is that the interface is not
// CONSERVATIVE -- with the rescale correct a partially-refined run still
// converges at first order, which is the composite-grid signature and what
// refluxing fixes. plans/3D.md M3 has the measurements and the design
// sketch; benchmarks/d3.json's amr_interface_note has the numbers.
//
// At 0 the second coarse read folds out.
override TIME_BLEND : f32 = 0.0f;

// One coarse cell's macroscopic state and non-equilibrium part.
fn sampleCoarse(c: vec3<i32>) -> CoarseSample3 {
  let w = vec3<u32>(wrapu(c.x, NX), wrapu(c.y, NY), wrapu(c.z, NZ));
  let cell = coarseCell(w);
  let ncells = NX * NY * NZ;
  var out: CoarseSample3;
  var rho = 0f;
  var m = vec3<f32>(0f);
  var f: array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    // Blended in POPULATION space before the moments are taken, so the
    // interpolated state is the equilibrium-plus-fneq decomposition of a
    // genuine intermediate distribution rather than a mix of two different
    // decompositions. At TIME_BLEND = 0 the second read folds out.
    f[i] = mix(f_coarse_t0[i * ncells + cell], f_coarse_t1[i * ncells + cell], TIME_BLEND);
    rho += f[i];
    m += f[i] * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
  }
  let u = m / max(rho, 1e-6f);   // NaN-containment floor
  out.rho = rho;
  out.u = u;
  for (var i = 0u; i < QN; i++) { out.fneq[i] = f[i] - feqD3Q(rho, u.x, u.y, u.z, i); }
  return out;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let FB = poolFB();
  // Dispatch is (FB/4, FB/4, (FB/4) * slots): the slot is folded into z
  // because 3D has no fourth dispatch dimension (plans/3D.md sec 2.4).
  let fz = gid.z % FB;
  let slot = gid.z / FB;
  if (gid.x >= FB || gid.y >= FB) { return; }
  let fi = vec3<u32>(gid.x, gid.y, fz);

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }
  if (GHOST_ONLY != 0u && isInterior3(fi)) { return; }
  if (NEW_ONLY != 0u && slotNew[slot] == 0u) { return; }

  let b = blockXYZ(u32(blockID));
  let origin = b * RB;

  // Fine cell centre in coarse-unit coordinates, then the 8 surrounding
  // coarse cell centres. A fine centre is never at an integer coarse
  // coordinate (it is always origin +- 1/4 + k/2), so floor() is unambiguous
  // and t is never exactly 0 or 1.
  let p = vec3<f32>(
    fineToCoarseUnit3(i32(fi.x), origin.x),
    fineToCoarseUnit3(i32(fi.y), origin.y),
    fineToCoarseUnit3(i32(fi.z), origin.z));
  let c0 = vec3<i32>(i32(floor(p.x)), i32(floor(p.y)), i32(floor(p.z)));
  let t = p - vec3<f32>(f32(c0.x), f32(c0.y), f32(c0.z));

  var s: array<CoarseSample3, 8>;
  for (var k = 0u; k < 2u; k++) {
    for (var j = 0u; j < 2u; j++) {
      for (var i = 0u; i < 2u; i++) {
        s[(k * 2u + j) * 2u + i] = sampleCoarse(c0 + vec3<i32>(i32(i), i32(j), i32(k)));
      }
    }
  }

  let fo = interpCoarseToFine3(s, t, TAU_COARSE);
  let cell = poolCell(slot, fi);
  let poolPlane = arrayLength(&f_pool) / QN;
  for (var i = 0u; i < QN; i++) { f_pool[i * poolPlane + cell] = fo[i]; }

}
