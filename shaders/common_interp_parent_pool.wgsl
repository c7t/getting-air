// The POOL-parent half of the interpolation accessor
// (plans/2D-backport.md B3-3) -- i.e. every L(m) -> L(m+1) hop where the
// "coarse" data is a SINGLE parent pool slot's own tile. Since
// plans/uniform-levels.md U5-1 that includes m = 0: the ROOT is a pool level
// too, and `PARENT_GHOST` is what tells the two cases apart.
//
// Five functions and nothing else; common_interp_kernel.wgsl is the kernel.
// See its header for the contract, and for why parentOrigin and sampleParent
// are one fragment: they must agree on a frame, and this one is PARENT-LOCAL
// INTERIOR coordinates.
//
// Reads `levelParams`, `f_parent_pool`, `parentSlot` and `quadrant`, all
// declared by the entry file (amr_interp_pool_parent.wgsl) -- a fragment must
// not @include, and WGSL module scope is order-independent.
//
// GENUINELY SIMPLER THAN THE DENSE HALF, for one structural reason: the data
// sampled here is one parent tile's own buffer (interior + its own
// already-ghost-filled border), never a second, different parent tile. A
// dense-parent lookup has to reach arbitrary coarse cells via
// cellIndex()+wrapCoord, because the coarse buffer holds the WHOLE domain.
// Here, because every level>=1 tile uses the identical RB/GHOST/FB shape
// (decision 2, plans/AMR-multilevel.md:9) and a child covers exactly one RB*RB
// quadrant of its parent's own 2*RB*2*RB interior, the full stencil radius
// fineToCoarseUnit can ever require (checked directly: for origin in {0, RB}
// and fx/fy spanning the child's whole FB-wide buffer, the resulting
// parent-local-interior index always lands in [-GHOST, 2*RB-1+GHOST]) maps
// 1:1, via a plain +GHOST offset, onto the parent's own already-valid [0, FB)
// index range -- including the parent's ghost cells. That is exactly why this
// needs no wrap and no second slot lookup.
//
// ── AND AT THE ROOT IT IS NOT TRUE, WHICH IS THE FOURTH SITE OF ONE MISTAKE
//
// The paragraph above justifies having no wrap by "including the parent's
// ghost cells". THE ROOT HAS NONE (amr2d.mjs's ghostDepthAtLevel(0) is 0), so
// parent-local index -1 maps to -1 and 2*RB maps off the end of the tile. It
// is not an edge case either: the stencil reach is [-GHOST, 2*RB-1+GHOST] and
// a child occupies one RB-wide quadrant of its parent, so EVERY child of the
// root needs root cells from OUTSIDE its parent root tile, on two of its four
// sides.
//
// So this is the fourth place a ring depth written as a CONSTANT turned out to
// be an assertion that every level has a ring -- after resolveSource (U4-1a),
// amr_criterion_pool.wgsl's stencil (U4-1), amr_force1.wgsl's gather (U4-2)
// and quadrantOrigin's parent frame (U5-0). The resolution is the same rule
// every time, amr2d.mjs's resolveSource: resolve the out-of-tile index against
// the OWNING same-level tile, on a periodic block grid. Here it is cheaper
// than anywhere else, because the root is ALWAYS FULL -- its block -> slot
// indirection is the identity, so the owning tile's slot is arithmetic and
// needs no second buffer and no new binding.
//
// A quadtree child's parent identity is NOT derivable from anything else
// (unlike this level's own logical (bx,by), which IS derivable from
// slotToBlock[slot] + this level's own NBX -- see the kernel), so
// parentSlot/quadrant are the two genuinely new per-slot fields this side
// needs, written once at allocation time and read-only here. WITH ONE
// EXCEPTION, and it is the root again: a root parent is always full, so its
// slot IS its block index and both fields are derivable from the child's own
// (bx, by). See parentSlotOf/quadrantOf.

// The PARENT's ring depth -- NOT this level's, which is the kernel's own
// `GHOST` const and is 2 at every pool level including level 1. Default 2
// keeps every existing pipeline byte-identical; the root-parent pipeline sets
// 0.
//
// EVERYTHING ROOT-SPECIFIC BELOW IS DERIVED FROM THIS, not from a separate
// flag, and that is deliberate -- the same reasoning amr_criterion_pool.wgsl
// records for its own GHOST override. At PARENT_GHOST == 0 the parent has no
// ring, which means it has no parent, which means it is the root, which means
// it is always full: resolving an out-of-tile index against the owning tile is
// then not one option among several, it is the only correct behaviour. A
// separate flag could be left unset on a ring-free pipeline, and U3 already
// paid for exactly that shape once.
override PARENT_GHOST : u32 = 2u;

// A mid-chain parent has its OWN tau: CardState carries only L0's, but e.g.
// L1 acting as parent to L2 has tau_fine = 2*tau_coarse - 0.5 applied
// recursively (card-params.mjs's tauAtLevel), supplied per child level as
// levelParams.parentTau rather than read from the domain-wide CardState.
// Level 1's is already tauAtLevel(0) -- the same number the dense accessor
// reads as `state.tau` -- so the root hop needs nothing new here.
fn parentTau() -> f32 { return levelParams.parentTau; }

// This level's own logical grid extent (plans/AMR-multilevel-M5.md's
// NBX[m]/NBY[m]) -- runtime uniform values, deliberately NOT `override`
// constants, since (unlike RB/GHOST, which are identical at every level) they
// differ per child level and one compiled pipeline serves all of them
// (decision 2).
fn levelNbx() -> u32 { return levelParams.nbx; }
fn levelNby() -> u32 { return levelParams.nby; }

// The parent's block grid, one quadtree rung coarser than this level's. True
// at every level by construction (main-amr.js's allocation loop doubles NBX
// per level, and the root's own nbx is W/(2*RB), exactly level 1's W/RB
// halved).
fn parentNbx() -> u32 { return levelNbx() >> 1u; }
fn parentNby() -> u32 { return levelNby() >> 1u; }

// This child's quadrant within its parent, and its parent's SLOT.
//
// Stored per slot for a ringed parent, because a SPARSE parent level's
// block -> slot map is data. DERIVED for a root parent, because the root is
// always full: slot == blockID there, so the parent of child block (bx, by) is
// root block (by>>1)*parentNbx + (bx>>1), and the quadrant is the two low bits
// the child's own block index already carries.
fn quadrantOf(slot: u32, bx: u32, by: u32) -> u32 {
  if (PARENT_GHOST == 0u) { return ((by & 1u) << 1u) | (bx & 1u); }
  return quadrant[slot];
}
fn parentSlotOf(slot: u32, bx: u32, by: u32) -> u32 {
  if (PARENT_GHOST == 0u) { return (by >> 1u) * parentNbx() + (bx >> 1u); }
  return u32(parentSlot[slot]);
}

// Origin: this child's own quadrant offset within its parent's 2*RB-wide
// INTERIOR -- 0 or RB per axis, from the quadrant bits alone, with no spatial
// coordinate involved (amr2d.mjs's quadrantOrigin is the host twin, stated in
// the parent's BUFFER frame, i.e. this plus the parent's ring depth). The
// block's own (bx,by) plays no part beyond naming the quadrant; the parent is
// found by slot, not by position.
fn parentOrigin(slot: u32, bx: u32, by: u32) -> vec2<u32> {
  let q = quadrantOf(slot, bx, by);
  return vec2<u32>((q & 1u) * RB, ((q >> 1u) & 1u) * RB);
}

// Parent-local-INTERIOR coordinates (ix,iy) -> flat index into the parent
// tile's own buffer. The +PARENT_GHOST offset is the interior -> buffer frame
// shift, and it is the PARENT's ring depth: this line read the module `GHOST`
// const until U5-1, which was right everywhere it had ever run and wrong at
// the root in both the offset and the tile stride.
fn parentCellIndex(pSlot: u32, ix: i32, iy: i32) -> u32 {
  let side = RB * 2u + 2u * PARENT_GHOST;
  let fxp = u32(ix + i32(PARENT_GHOST));
  let fyp = u32(iy + i32(PARENT_GHOST));
  return pSlot * (side * side) + fyp * side + fxp;
}

// CoarseSample lives in common_interp.wgsl, alongside the blend that consumes
// it -- this supplies only the FETCH.
//
// (bx, by) are this child's own logical block coordinates, which a ringed
// parent does not need and a root parent does: see the ring-free resolution
// below, and the kernel's header for why the accessor takes them at all.
fn sampleParent(slot: u32, bx: u32, by: u32, ix: i32, iy: i32) -> CoarseSample {
  var pSlot = parentSlotOf(slot, bx, by);
  var jx = ix; var jy = iy;
  if (PARENT_GHOST == 0u) {
    // THE RING-FREE PATH. amr2d.mjs's resolveSource, specialised to a level
    // that is always full: an index below 0 belongs to the tile on the low
    // side at +2*RB, one at 2*RB or above to the tile on the high side at
    // -2*RB. ONE HOP ALWAYS SUFFICES -- the stencil reach is GHOST = 2 cells
    // and a tile is 2*RB = 16 of them. The two axes resolve independently, so
    // a corner lands in the DIAGONAL neighbour, which is the same rule applied
    // twice and not a special case.
    //
    // The block grid is periodic, matching every other kernel here AND
    // matching what the dense accessor's wrapCoord(ix, W) does over the whole
    // domain -- which is what makes this hop bit-identical to that one rather
    // than merely equivalent. It inherits that accessor's known WALL_Y gap
    // unchanged (the dense-parent fragment this mirrors said the same before
    // U7-6f deleted it).
    let RB2 = i32(RB * 2u);
    let pnbx = parentNbx(); let pnby = parentNby();
    var pbx = bx >> 1u; var pby = by >> 1u;
    if (jx < 0)         { jx += RB2; pbx = (pbx + pnbx - 1u) % pnbx; }
    else if (jx >= RB2) { jx -= RB2; pbx = (pbx + 1u) % pnbx; }
    if (jy < 0)         { jy += RB2; pby = (pby + pnby - 1u) % pnby; }
    else if (jy >= RB2) { jy -= RB2; pby = (pby + 1u) % pnby; }
    pSlot = pby * pnbx + pbx;
  }
  let cell = parentCellIndex(pSlot, jx, jy);
  let parentPlaneStride = arrayLength(&f_parent_pool) / 9u;

  var f: array<f32, 9>;
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    f[i] = fUnpack(f_parent_pool[fIdx(i, parentPlaneStride, cell)], i);
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f); // NaN-containment floor

  var out: CoarseSample;
  out.rho = rho; out.ux = ux; out.uy = uy;
  for (var i = 0u; i < 9u; i++) {
    out.fneq[i] = f[i] - feqD2Q9(rho, ux, uy, i);
  }
  return out;
}
