// The 2D pool manager's decision predicates. Included via
// `// @include "common_refine_predicates.wgsl"`, AFTER common_geometry.wgsl and
// common_refine.wgsl.
//
// Split out of common_refine.wgsl, which the 3D manager also includes: these
// depend on the `state` binding, `override W`/`H`, `HAS_BODY` and `BOX_REFINE`,
// which only amr_manage_pool.wgsl declares.

// ─────────────────────────────────────────────────────────────────────────────
// THE DECISION PREDICATES, ONE COPY (plans/2D-backport.md B3-7)
//
// The dense-parent manager (deleted at U7-6f) and amr_manage_pool.wgsl each
// carried their own isNearBody/inSpongeBand. They were the same tests,
// differing only in how the candidate's centre and half-extent were derived --
// which was the one thing each manager genuinely knew for itself:
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
// i.e. inside/near the ALBC sponge band (amr_step1.wgsl's SPONGE_W). A fixed L0
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
