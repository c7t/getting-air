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
// `N_REFINE_INC`, `N_REFINE_MAX` and `MAX_LEVEL`, the `state` binding, and
// `override W`/`override H` -- the same arrangement common_geometry.wgsl uses.
// Its two includers, amr_manage.wgsl and amr_manage_pool.wgsl, are the only
// files that declare all of those, which is why the geometry predicates below
// live here and not in a more general fragment (see common_criterion.wgsl's
// header for what happens when a fragment's dependencies straddle includers).

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


// ─────────────────────────────────────────────────────────────────────────────
// THE DECISION PREDICATES, ONE COPY (plans/2D-backport.md B3-7)
//
// amr_manage.wgsl and amr_manage_pool.wgsl each carried their own
// isNearBody/inSpongeBand. They were the same tests, differing only in how the
// candidate's centre and half-extent were derived -- which is the one thing
// each manager genuinely knows for itself:
//
//   dense   an L0 block is BLOCK cells of size 1 -> half-extent BLOCK/2
//   pool    a level-m tile's interior is 2*RB cells of size 2^-m
//                                              -> half-extent RB * 2^-m
//
// Those are the SAME FOOTPRINT (that is what "level 1 is footprint-preserving
// 1:1 with L0's blocks" means, plans/AMR-multilevel.md decision 1) written two
// ways, and at level 1 they are the same number -- which is exactly the
// arrangement that lets two copies drift while agreeing on every test anyone
// runs. B3-5 found precisely that: a sibling formula fixed on one page and
// left wrong on the two that move a body.
//
// So the CALLER supplies the centre and the half-extent it knows, and the
// PREDICATE is shared. Same split as B3-3's parentOrigin/sampleParent pair,
// for the same reason.

const EPS_FLOOR = 1e-6f;

// log2 of a raw criterion value, floored and clamped. The criterion kernels
// reduce max|omega| per block; this is the ladder's input before the
// per-level physical correction (see PHYSICAL UNITS above).
fn epsOf(crit: f32) -> f32 {
  return min(1.0f, log2(max(crit, EPS_FLOOR)));
}

// True if ANY POINT of a box of half-extent `halfExtent` L0 units, centred at
// `centre` in L0 BUFFER space, comes within FORCE_REFINE_MARGIN of the card's
// surface -- now, or FORCE_REFINE_LOOKAHEAD macro-steps from now.
// common_geometry.wgsl's nearBodyBox is the test and phiMinPose is the
// two-pose distance; see that file for the Lipschitz bound and for why the
// future pose moves the TEST POINT backward rather than the ellipse forward.
//
// THE BOX, NOT THE CENTRE -- plans/2D-backport.md B4. This used to be one
// get_phi at the centre, which under-reports by the block's own circumradius
// (5.66 L0 cells at BLOCK=8). ?boxrefine=0 restores that exactly, truncation
// and all, so the two live in one build and the difference can be measured
// rather than argued. (It used to restore a truncated WINDOW conversion too;
// the body is in buffer coordinates since B5, so only the truncation is left
// to preserve.)
//
// The box path drops the `(c + W - u32(off_x)) % W` window reduction with it:
// get_phi already takes the NEAREST PERIODIC IMAGE, so the modulo bought
// nothing, and the u32() truncated a fractional offset (and, in the pool
// case, a fractional tile centre) by up to a cell -- error of the same order
// as the margin it is compared against.
fn nearBodyAt(centre: vec2<f32>, halfExtent: f32) -> bool {
  if (HAS_BODY == 0u) { return false; }

  if (BOX_REFINE == 0u) {
    // The u32() round-trip is the LEGACY TRUNCATION, kept deliberately: this
    // path exists to reproduce the pre-B4 predicate exactly, and truncating a
    // fractional tile centre by up to a cell is precisely the error being
    // preserved for comparison. amr2d.mjs's bodyFrameL0Legacy mirrors it.
    let truncated = vec2<f32>(f32(u32(centre.x)), f32(u32(centre.y)));
    return phiMinPose(truncated, FORCE_REFINE_LOOKAHEAD, state) < FORCE_REFINE_MARGIN;
  }

  let p = centre;
  return nearBodyBox(p, halfExtent, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD, state);
}

// True if `centre` lies within SPONGE_EXCLUDE_W (L0 cells) of any window edge,
// i.e. inside/near the ALBC sponge band (amr_step.wgsl's SPONGE_W). A fixed L0
// strip, deliberately NOT scaled per level -- the sponge is a property of the
// window, not of the level being decided. Gated off when SPONGE_EXCLUDE_W <= 0
// (the JS default is 8; ?spongeExclude=0 disables it).
fn inSpongeBandAt(centre: vec2<f32>) -> bool {
  if (SPONGE_EXCLUDE_W <= 0.0f) { return false; }
  let w = bufferToWindowCell(vec2<u32>(u32(centre.x), u32(centre.y)), state);
  let wx = w.x; let wy = w.y;
  let distX = min(f32(wx), f32(W - wx));
  let distY = min(f32(wy), f32(H - wy));
  return min(distX, distY) < SPONGE_EXCLUDE_W;
}
