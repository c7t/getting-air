// The POOL-parent half of the averaging accessor (plans/2D-backport.md B3-2)
// -- i.e. every L(m) -> L(m-1) hop with m>=2, where the destination is a
// specific (lx,ly) inside the PARENT's own FB*FB tile rather than a dense
// cellIndex() address.
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

// A mid-chain parent has its OWN tau, not L0's: tau_fine = 2*tau_coarse - 0.5
// applied recursively, which the host supplies per level as
// levelParams.parentTau (card-params.mjs's tauAtLevel(m-1)).
fn parentTau() -> f32 { return levelParams.parentTau; }

// Destination: this slot's quadrant of its parent's interior. The parent's
// 2*RB interior cells are halved by the quadrant on each axis, so the offset
// is q*RB -- it falls out of the quadrant bits alone, with no spatial
// coordinate involved (amr2d.mjs's quadrantOrigin says the same thing on the
// host). GHOST shifts it past the parent tile's own ring.
fn parentCellForChild(slot: u32, blockID: i32, lcx: u32, lcy: u32) -> u32 {
  let FB = RB * 2u + 2u * GHOST;
  let pSlot = u32(parentSlot[slot]);
  let q = quadrant[slot];
  let qx = q & 1u;
  let qy = (q >> 1u) & 1u;
  let plx = qx * RB + lcx;
  let ply = qy * RB + lcy;
  return pSlot * (FB * FB) + (ply + GHOST) * FB + (plx + GHOST);
}

fn parentStoreWord(cell: u32, wi: u32, word: u32) {
  let parentPlaneStride = arrayLength(&f_parent_pool) / 9u;
  f_parent_pool[wi * parentPlaneStride + cell] = word;
}
