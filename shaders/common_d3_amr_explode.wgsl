// EXPLODE: coarse -> fine at the interface. plans/3D.md M4.1b.
// Fragment only; the entry files list every include.
//
// Chen, Filippova, Hoch, Molvig, Shock, Teixeira & Zhang (2006), Physica A
// 362(1) 158-167 -- the PowerFLOW grid-refinement algorithm. This replaces
// the trilinear-plus-Dupuis-Chopard `interp` for the STEADY-STATE ring
// refresh. It is not an interpolation and there is no rescale.
//
// WHAT AN EXPLODE IS. An interface coarse cell V is one that is NOT refined
// but has a refined neighbour. For each direction i whose coarse target
// V + e_i is COVERED, V's post-collision population has no coarse
// destination -- the coarse grid does not solve there (M4.1a) -- so instead
// of vanishing it is distributed among the n^D fine cells that cover V, at
// UNCHANGED DENSITY: (n_i)_f = (n_i)_c, mass 1/8 each, totalling exactly
// what V held. Nothing is interpolated and nothing is rescaled.
//
// WHY THE RING IS EXACTLY THE RIGHT PLACE FOR IT. The fine cells covering V
// ARE this tile's ring: GHOST = 2 is one coarse cell deep, so the ring is
// V's n^D subdivision. And n = 2 fine advections carry the exploded mass
// exactly two fine cells = one coarse cell, i.e. out of V and into the
// refined region, which is where it was headed. The ring's existing
// self-advance is not a workaround for this scheme, it IS this scheme: after
// substep A a depth-1 ring cell holds what depth 2 held, and with a uniform
// explosion those are the same value, so the state "reoccurs" for substep B
// exactly as the paper requires.
//
// DIRECTIONS THAT ARE NOT EXPLODED ARE ZEROED, deliberately. They are never
// read: an interior fine cell only ever gathers from a ring cell along a
// direction pointing INTO the tile, and those are precisely the exploded
// ones. The outward directions are refilled from the interior by the
// substeps and then harvested by the coalesce. Zeroing rather than leaving
// them makes that invariant checkable instead of merely true.
//
// CONSERVATION IS STRUCTURAL, not measured and corrected. V's population is
// removed from the coarse dynamics for free -- V + e_i is covered, so the
// coarse step is skipped there and nobody pulls it -- and an exact copy is
// placed in the fine cells. Mass goes exactly one place. That is what our
// reflux pass could not achieve at a convex corner, and the corner never
// arises here because nothing is compared: the coarse cell and its fine
// children are the same volume, and the state is MOVED between the two
// bookkeepings rather than reconciled across them.

@group(0) @binding(0) var<storage, read>       f_coarse    : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_pool      : array<f32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(3) var<storage, read>       blockSlot   : array<i32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let FB = poolFB();
  // Dispatch is (FB/4, FB/4, (FB/4) * slots), the slot folded into z --
  // 3D has no fourth dispatch dimension (plans/3D.md sec 2.4).
  let fz = gid.z % FB;
  let slot = gid.z / FB;
  if (gid.x >= FB || gid.y >= FB) { return; }
  let fi = vec3<u32>(gid.x, gid.y, fz);

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }
  // Ring cells only. The interior is the fine solver's own evolved state.
  if (isInterior3(fi)) { return; }

  let b = blockXYZ(u32(blockID));
  // This ring cell in GLOBAL FINE coordinates, then the coarse cell holding
  // it. Ring cells of a tile whose neighbour is ALSO refined sit inside a
  // covered coarse cell; DIRECT_GHOST means the step kernel never reads
  // those, so there is nothing to explode into them.
  let g = wrapFine3(vec3<i32>(b) * i32(2u * RB) + vec3<i32>(fi) - i32(GHOST));
  let v = coarseOfFine(g);
  if (blockSlot[blockIdOf(blockOfCoarse(v))] >= 0) { return; }

  let ncells = NX * NY * NZ;
  let poolPlane = arrayLength(&f_pool) / QN;
  let cell = poolCell(slot, fi);
  let vc = coarseCell(v);

  for (var i = 0u; i < QN; i++) {
    let t = vec3<u32>(
      wrapu(i32(v.x) + ex[i], NX),
      wrapu(i32(v.y) + ey[i], NY),
      wrapu(i32(v.z) + ez[i], NZ));
    let covered = blockSlot[blockIdOf(blockOfCoarse(t))] >= 0;
    f_pool[i * poolPlane + cell] = select(0f, f_coarse[i * ncells + vc], covered);
  }
}
