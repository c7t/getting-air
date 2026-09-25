// COALESCE's entry point for a POOL parent (plans/3D.md M5.2b). Replaces the
// dense-grid `main` at the bottom of common_d3_amr_coalesce.wgsl; see
// coalesceAt's own header for why only the dispatch differs.
//
// The parent is a pool, so there is no grid to dispatch over. One thread per
// INTERIOR cell of one parent tile, the slot folded into z exactly as every
// other pool kernel here does it -- dispatch (FB/4, FB/4, (FB/4) * slots).
//
// INTERIOR ONLY. A parent tile's ring cells belong to some OTHER parent tile,
// whose own thread speaks for them. Letting both write would double-count at
// every parent-tile boundary, which is a seam the coarse solver does not
// have and would be invisible to a conservation check -- the two writes land
// in different slots of the same physical cell.
//
// parentSlotToBlock is this level's PARENT slot->block map, which is the
// child level's grandparent bookkeeping and is bound only here.
@group(0) @binding(9) var<storage, read> parentSlotToBlock : array<i32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let FB = poolFB();
  let pz = gid.z % FB;
  let pslot = gid.z / FB;
  if (gid.x >= FB || gid.y >= FB) { return; }
  let loc = vec3<u32>(gid.x, gid.y, pz);
  if (!isInterior3(loc)) { return; }
  if (pslot >= arrayLength(&parentSlotToBlock)) { return; }
  let pbID = parentSlotToBlock[pslot];
  if (pbID < 0) { return; }

  // Parent block -> the global PARENT cell this thread owns. The inverse of
  // common_d3_parent_pool.wgsl's parentIndex, and deliberately written the
  // other way round so the two are not one expression used twice.
  let n = parentNb();
  let pb = vec3<u32>(u32(pbID) % n.x, (u32(pbID) / n.x) % n.y, u32(pbID) / (n.x * n.y));
  coalesceAt(pb * (2u * RB) + loc - vec3<u32>(GHOST));
}
