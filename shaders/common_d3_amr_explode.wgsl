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
// LINEAR EXPLOSION (M4.1c, `EXPLODE_LINEAR`, ?explin=0 restores uniform).
// A uniform explosion makes the whole coarse cell's worth of a population
// one constant, which is first-order: it is a piecewise-CONSTANT
// reconstruction of a field that varies across the cell. Chen et al. give
// the second-order form
//
//     Ndot_i^f = Ndot_i^c + (r_f - r_c) . F_i
//
// with F_i a gradient of N_i on the coarse grid and (r_f - r_c) the child's
// offset from the parent centre, which is +-1/4 of a coarse cell on each
// axis. Two constraints shape which gradient is admissible here, and both
// are load-bearing:
//
//   1. ONLY AXES WITH TWO REAL COARSE NEIGHBOURS. A central difference
//      needs f at v +- e_a, and a REFINED neighbour has no coarse solution
//      to read -- the coarse grid does not solve there (M4.1a) and its slot
//      holds whatever coalesce last wrote. The test is per-axis and per-
//      cell, so it excludes the interface normal automatically (v + e_i is
//      covered by definition here) and degrades to UNIFORM at a concave
//      corner, where too many neighbours are refined. No geometry needs to
//      be classified.
//
//   2. PROJECTED ORTHOGONAL TO c_i, which is what keeps the scheme's
//      reoccurrence property. The ring self-advance is what lets one
//      explosion serve BOTH fine substeps: after substep A a depth-1 ring
//      cell holds what depth-2 held, so the state "reoccurs" and substep B
//      gathers the same value substep A did. That is exactly the statement
//      that the exploded field is invariant under translation by -c_i, and
//      a linear field is invariant under it iff F_i . c_i = 0. With a
//      component along c_i the two substeps inject DIFFERENT values from
//      one coarse state -- a spurious time variation at the interface,
//      which is an acoustic source, not a truncation error.
//
// CONSERVATION IS UNTOUCHED, by construction rather than by cancellation:
// the eight children are symmetric about the parent centre, so their
// offsets sum to zero on every axis and any linear term sums out. This
// holds for the two-substep injection too, not just the instantaneous
// state -- the substeps draw from the two layers along c_i, whose offsets
// are equal and opposite. So 4.1c cannot move the conservation gates, and
// if it does, it is a bug in this file and not a tolerance to widen.
//
// CONSERVATION IS STRUCTURAL, not measured and corrected. V's population is
// removed from the coarse dynamics for free -- V + e_i is covered, so the
// coarse step is skipped there and nobody pulls it -- and an exact copy is
// placed in the fine cells. Mass goes exactly one place. That is what our
// reflux pass could not achieve at a convex corner, and the corner never
// arises here because nothing is compared: the coarse cell and its fine
// children are the same volume, and the state is MOVED between the two
// bookkeepings rather than reconciled across them.

// M4.1c. 1 = linear explosion, 0 = the M4.1b uniform one. Both paths live in
// one build so they can be A/B'd on the same GPU in the same session
// (?explin=0), which is how every other physics knob here is kept honest.
override EXPLODE_LINEAR : u32 = 1u;

@group(0) @binding(0) var<storage, read>       f_coarse    : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_pool      : array<f32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(3) var<storage, read>       blockSlot   : array<i32>;

fn coveredCoarseX(v: vec3<u32>) -> bool {
  return blockSlot[blockIdOf(blockOfCoarse(v))] >= 0;
}

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

  // The two coarse neighbours on each axis, and whether BOTH are unrefined
  // -- i.e. whether a central difference on that axis reads real coarse
  // solutions. Hoisted out of the direction loop: it depends on v only.
  let vmx = coarseCell(vec3<u32>(wrapu(i32(v.x) - 1, NX), v.y, v.z));
  let vpx = coarseCell(vec3<u32>(wrapu(i32(v.x) + 1, NX), v.y, v.z));
  let vmy = coarseCell(vec3<u32>(v.x, wrapu(i32(v.y) - 1, NY), v.z));
  let vpy = coarseCell(vec3<u32>(v.x, wrapu(i32(v.y) + 1, NY), v.z));
  let vmz = coarseCell(vec3<u32>(v.x, v.y, wrapu(i32(v.z) - 1, NZ)));
  let vpz = coarseCell(vec3<u32>(v.x, v.y, wrapu(i32(v.z) + 1, NZ)));
  let okx = !coveredCoarseX(vec3<u32>(wrapu(i32(v.x) - 1, NX), v.y, v.z))
         && !coveredCoarseX(vec3<u32>(wrapu(i32(v.x) + 1, NX), v.y, v.z));
  let oky = !coveredCoarseX(vec3<u32>(v.x, wrapu(i32(v.y) - 1, NY), v.z))
         && !coveredCoarseX(vec3<u32>(v.x, wrapu(i32(v.y) + 1, NY), v.z));
  let okz = !coveredCoarseX(vec3<u32>(v.x, v.y, wrapu(i32(v.z) - 1, NZ)))
         && !coveredCoarseX(vec3<u32>(v.x, v.y, wrapu(i32(v.z) + 1, NZ)));

  // This child's offset from the parent centre, in COARSE units: the fine
  // cells covering v are 2v and 2v+1 on each axis, so the offset is -1/4 or
  // +1/4. These eight offsets sum to zero, which is the whole conservation
  // argument.
  let d = (vec3<f32>(g - vec3<i32>(v) * 2) - 0.5f) * 0.5f;

  for (var i = 0u; i < QN; i++) {
    let t = vec3<u32>(
      wrapu(i32(v.x) + ex[i], NX),
      wrapu(i32(v.y) + ey[i], NY),
      wrapu(i32(v.z) + ez[i], NZ));
    let ei3 = vec3<i32>(ex[i], ey[i], ez[i]);
    if (blockSlot[blockIdOf(blockOfCoarse(t))] < 0) {
      // IN-ORPHAN (M4.1c). v's COARSE destination v + e_i is unrefined, so
      // this direction is not exploded -- but this particular CHILD's own
      // one-fine-step destination can still be inside the refined region.
      // That happens exactly at a convex edge or corner, where the coarse
      // cell straddles the seam on one axis and the child does not, and it
      // is the mirror image of the coalesce's OUT-orphan: one is a half-step
      // exit no coarse bucket can absorb, this is a half-step entry no
      // coarse cell explodes. Leaving it zero starves the fine region --
      // the interior cell gathers 0 where it should gather v's population.
      //
      // Deliberately UNIFORM even when EXPLODE_LINEAR is on: only a subset
      // of the eight children fire, so a linear term would not sum out over
      // them, and the coalesce has to charge the coarse cell EXACTLY what
      // was injected or conservation goes. A corner case is not the place to
      // trade an exact balance for a second-order term.
      let dst = wrapFine3(g + ei3);
      let intoTile = blockSlot[blockIdOf(blockOfCoarse(coarseOfFine(dst)))] >= 0
                  && all(blockOfFine(dst) == b);
      f_pool[i * poolPlane + cell] = select(0f, f_coarse[i * ncells + vc], intoTile);
      continue;
    }
    let fc = f_coarse[i * ncells + vc];
    if (EXPLODE_LINEAR == 0u) {
      f_pool[i * poolPlane + cell] = fc;
      continue;
    }
    let base = i * ncells;
    var G = vec3<f32>(
      select(0f, 0.5f * (f_coarse[base + vpx] - f_coarse[base + vmx]), okx),
      select(0f, 0.5f * (f_coarse[base + vpy] - f_coarse[base + vmy]), oky),
      select(0f, 0.5f * (f_coarse[base + vpz] - f_coarse[base + vmz]), okz));
    // Project orthogonal to c_i so the exploded state reoccurs across the
    // two substeps. i = 0 cannot reach here -- e_0 is the rest vector, so
    // t == v, and v is unrefined by the guard above -- so |c_i|^2 >= 1.
    let ei = vec3<f32>(ei3);
    G -= ei * (dot(G, ei) / dot(ei, ei));
    f_pool[i * poolPlane + cell] = fc + dot(d, G);
  }
}
