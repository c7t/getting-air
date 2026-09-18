// The POOL-parent half of the averaging accessor (plans/2D-backport.md B3-2)
// -- i.e. every L(m) -> L(m-1) hop where the destination is a specific (lx,ly)
// inside the PARENT's own tile rather than a dense cellIndex() address. Since
// plans/uniform-levels.md U5-2 that includes m = 1: the ROOT is a pool level
// too, and `PARENT_GHOST` is what tells the two cases apart.
//
// Three functions and nothing else; common_average.wgsl is the kernel. See
// its header for the contract.
//
// Reads `levelParams`, `parentSlot` and `quadrant` and writes
// `f_parent_pool`, all declared by the entry file
// (amr_average_pool_parent.wgsl) -- a fragment must not @include, and WGSL
// module scope is order-independent. parentSlot/quadrant are the same two
// per-slot fields amr_interp_pool_parent.wgsl already reads for the forward
// (prolongation) direction; this direction needs no new ones.
//
// EASIER THAN THE FORWARD DIRECTION AT THE ROOT, and it is worth saying why,
// because U5-1 was not. Restriction WRITES one parent cell per child cell and
// the destination is always inside the parent's own interior -- there is no
// stencil, so nothing ever reaches past the parent tile and the ring-free root
// needs no neighbour resolution here. All PARENT_GHOST does on this side is
// stop shifting the destination past a ring that is not there, and correct the
// tile stride. The forward direction had to resolve into the neighbouring root
// tile; this one has no out-of-tile index to resolve.

// The PARENT's ring depth -- NOT this level's, which is the kernel's own
// `GHOST` const and is 2 at every pool level including level 1. Default 2
// keeps every existing pipeline byte-identical; the root-parent pipeline sets
// 0. See common_interp_parent_pool.wgsl's own PARENT_GHOST for why everything
// root-specific is DERIVED from it rather than flagged separately.
override PARENT_GHOST : u32 = 2u;

// A mid-chain parent has its OWN tau, not L0's: tau_fine = 2*tau_coarse - 0.5
// applied recursively, which the host supplies per level as
// levelParams.parentTau (card-params.mjs's tauAtLevel(m-1)). Level 1's is
// already tauAtLevel(0) -- the same number the dense accessor reads as
// `state.tau` -- so the root hop needs nothing new here.
fn parentTau() -> f32 { return levelParams.parentTau; }

// This child's quadrant within its parent, and its parent's SLOT. Stored per
// slot for a ringed parent, because a SPARSE parent level's block -> slot map
// is data; DERIVED for a root parent, because the root is always full and its
// slot IS its block index. The forward direction says the same thing in
// common_interp_parent_pool.wgsl, from (bx, by) rather than from blockID --
// the same two lines, reached from whichever of the pair each kernel has.
fn quadrantOfChild(slot: u32, blockID: i32) -> u32 {
  if (PARENT_GHOST == 0u) {
    let bx = u32(blockID) % levelParams.nbx;
    let by = u32(blockID) / levelParams.nbx;
    return ((by & 1u) << 1u) | (bx & 1u);
  }
  return quadrant[slot];
}
fn parentSlotOfChild(slot: u32, blockID: i32) -> u32 {
  if (PARENT_GHOST == 0u) {
    let bx = u32(blockID) % levelParams.nbx;
    let by = u32(blockID) / levelParams.nbx;
    return (by >> 1u) * (levelParams.nbx >> 1u) + (bx >> 1u);
  }
  return u32(parentSlot[slot]);
}

// Destination: this slot's quadrant of its parent's interior. The parent's
// 2*RB interior cells are halved by the quadrant on each axis, so the offset
// is q*RB -- it falls out of the quadrant bits alone, with no spatial
// coordinate involved (amr2d.mjs's quadrantOrigin says the same thing on the
// host). PARENT_GHOST shifts it past the parent tile's own ring, when the
// parent has one.
fn parentCellForChild(slot: u32, blockID: i32, lcx: u32, lcy: u32) -> u32 {
  let side = RB * 2u + 2u * PARENT_GHOST;
  let pSlot = parentSlotOfChild(slot, blockID);
  let q = quadrantOfChild(slot, blockID);
  let qx = q & 1u;
  let qy = (q >> 1u) & 1u;
  let plx = qx * RB + lcx;
  let ply = qy * RB + lcy;
  return pSlot * (side * side) + (ply + PARENT_GHOST) * side + (plx + PARENT_GHOST);
}

fn parentStoreWord(cell: u32, wi: u32, word: u32) {
  let parentPlaneStride = arrayLength(&f_parent_pool) / 9u;
  f_parent_pool[wi * parentPlaneStride + cell] = word;
}
