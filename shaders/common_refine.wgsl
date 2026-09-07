// Vorticity-magnitude refinement LADDER -- AGAL's Algorithm 3, replacing the
// single binary threshold this project used per level-pair.
// Included via `// @include "common_refine.wgsl"`.
//
// WHY A LADDER
//
// The old form asked one yes/no question per level-pair against a single
// REFINE_THRESH, so how deeply a region got refined was governed mainly by
// its proximity to the body (geometry forcing) rather than by how much was
// happening in it. Measured at N=3: 118 of 132 level-2 blocks were inside
// the geometry halo and the remaining 14 were the 2:1 cascade shell -- level
// 2 never reached the wake at all, however energetic the wake was.
//
// A ladder instead maps vorticity MAGNITUDE to a desired level, so a strong
// shed vortex can earn level 2 on its own merits wherever it happens to be.
// AGAL: `s_W = floor(log2(criterion))`, clamp at N_REFINE_MAX, then walk
// `s_W < N_REFINE_START - N_REFINE_INC*p` to pick a desired level
// (solver_lbm_criterion.cu, Cu_ComputeRefCriteria).
//
// ANCHORED AT THE COARSE END, not the fine end. AGAL indexes its ladder down
// from the finest level, so the threshold for the FIRST level of refinement
// moves whenever the level count changes. Here the base is the level-1
// threshold and each further level costs N_REFINE_INC more octaves:
//
//   desired >= k   <=>   epsPhys >= REFINE_THRESH + N_REFINE_INC*(k-1)
//
// Same uniform-octaves-per-level structure, but adding a level no longer
// silently retunes the coarse end -- which matters because REFINE_THRESH
// here is a MEASURED value (see main-amr.js's own comment above it), and a
// reparameterisation that quietly invalidated it would be a bad trade.
//
// PHYSICAL UNITS. `epsPhys` must be log2 of the PHYSICAL velocity gradient,
// not the raw lattice-cell difference the criterion kernels reduce. A
// level-m cell is 2^-m the size of an L0 cell, so the caller adds m:
//   log2(omega_physical) = log2(omega_lattice) + m
// This is exactly AGAL's `/(2.0*dx_L)`. Without it, deeper levels are
// under-refined by 2^m and a vortex becomes LESS likely to keep refinement
// the moment it gets refined.
//
// DEPENDS ON THE INCLUDER declaring the overrides `REFINE_THRESH`,
// `N_REFINE_INC`, `N_REFINE_MAX` and `MAX_LEVEL` -- the same arrangement
// common_geometry.wgsl uses for `override W`/`override H`.

// Desired level for a block whose physical log2|omega| is epsPhys.
// Returns 0 (no refinement) up to MAX_LEVEL (finest configured).
fn desiredLevelAt(epsPhys: f32, base: f32) -> i32 {
  // Clamp above, as AGAL does: beyond this the answer is "finest" anyway,
  // and it keeps a pathological spike from dominating.
  let w = min(epsPhys, N_REFINE_MAX);
  var L: i32 = 0;
  for (var k: i32 = 1; k <= MAX_LEVEL; k = k + 1) {
    if (w >= base + N_REFINE_INC * f32(k - 1)) { L = k; }
  }
  return L;
}

// Refine-side ladder: the level this block's own flow asks for.
fn desiredLevel(epsPhys: f32) -> i32 {
  return desiredLevelAt(epsPhys, REFINE_THRESH);
}

// Coarsen-side ladder, shifted down by the REFINE/COARSEN gap so a block has
// to fall a full hysteresis band below the level it holds before releasing
// it. Preserves the anti-flicker property the old REFINE_THRESH >
// COARSEN_THRESH pair provided -- without it a block sitting near a rung
// would refine and coarsen on alternate evaluations, and every one of those
// transitions re-interpolates its region from the coarser parent, which is
// visible as the wake "lumpiness" this project has already been bitten by.
fn desiredLevelCoarsen(epsPhys: f32) -> i32 {
  return desiredLevelAt(epsPhys, COARSEN_THRESH);
}
