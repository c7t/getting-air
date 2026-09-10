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

override TAU_COARSE : f32 = 0.8f;
override GHOST_ONLY : u32 = 1u;

// Where in the parent step this ring refresh is for: 0 = the parent's state
// at time t, 0.5 = halfway. DEFAULT 0, AND CURRENTLY ALWAYS 0.
//
// This exists because of a hypothesis that was TESTED AND REJECTED, and the
// result is worth keeping. Interpolating the ring once per parent step and
// using it for both fine substeps leaves substep B looking at interface data
// half a coarse step stale, which is a plausible source of the seam error
// this solver does have: on the analytic Beltrami flow with a refined box,
// the coarse cells just outside the refined region grow a relative error of
// 1.35e-3 -> 3.2e-3 -> 6.3e-3 -> 1.0e-2 at t = 1, 2, 4, 8 -- roughly linear
// in t -- while the same run with NO coarse/fine interface (?refine=all)
// holds 1.6e-3 flat.
//
// Adding a second, time-blended refresh before substep B made that error
// slightly WORSE (nearOut 3.24e-2 -> 3.93e-2 at t=64), so staleness is not
// the cause. The remaining candidate is that this interface is not
// CONSERVATIVE: the coarse solver's flux across the seam and the fine
// solver's flux across the same seam are computed independently and do not
// agree, so mass and momentum leak there at a constant rate. Fixing that
// needs a flux correction (refluxing), which is real work and is recorded
// in plans/3D.md rather than guessed at here.
//
// The knob stays because it is the natural thing to re-try once the
// interface is conservative, and at 0 the second coarse read folds out.
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
