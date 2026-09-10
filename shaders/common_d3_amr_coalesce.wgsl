// COALESCE: fine -> coarse at the interface, plus the macroscopic
// projection under a refined region. plans/3D.md M4.1b. Fragment only.
//
// The other half of Chen et al. (2006)'s explode/coalesce, and the
// replacement for `average`. It is not a restriction and there is no
// rescale.
//
// WHAT A COALESCE IS. Over the two fine substeps, populations leaving the
// refined region advect into the ring -- which is exactly the n^D fine
// subdivision of the first UNREFINED coarse cell outside (GHOST = 2 is one
// coarse cell deep). Summing them recovers the coarse population that cell
// should receive:
//
//     (n_i)_c = (1/n^D) * sum_p (n_i)_f
//
// and the sum is over BOTH ring depths, which is not an approximation but
// the temporal accounting. Trace one outward direction on the low-x face,
// interior starting at index 2:
//
//   substep A   index 1 <- index 2   (interior at t)
//               index 0 <- index 1   (the exploded state)
//   substep B   index 1 <- index 2   (interior at t + dt/2)
//               index 0 <- index 1   (interior at t)
//
// so after substep B the two ring layers hold the outflux of the two
// substeps, one each, in one buffer. Two depths x four lateral cells = the
// eight sub-voxels of one coarse cell, each carrying V_fine = V_coarse/8.
// The 1/8 is a cell-volume ratio, not a weight.
//
// WHY THE RING MUST NOT COLLIDE for this to be right: the paper is explicit
// that coalesce averages the advected-but-UNCOLLIDED states, and that "an
// arithmetic average of N_i's rather than of Ntilde_i's would invalidate the
// correctness of non-equilibrium distributions on the coarse grid".
// common_d3_amr_step1.wgsl collides the interior only, for this reason.
//
// NO DUPUIS-CHOPARD RESCALE, and the thing that replaces it is exact rather
// than approximate. With a = (n-1)/2n and omega_f = 2/(1 + n(2/omega_c - 1))
// -- which at n = 2 is this project's own tau_f = 2 tau_c - 0.5 -- the paper
// establishes the identity 1/(n omega_f) - 1/omega_c + a = 0, and that is
// what makes the coarse non-equilibrium part come out correct in the leading
// order with nothing applied to it. Rohde et al. (2006) independently found
// that ADDING the rescale back to a volumetric scheme degrades it to first
// order. Do not reintroduce it here "to be safe".
//
// WHERE THE RESULT IS WRITTEN, and why it needs no change to the hot kernel.
// The coalesced value is the population the covered cell C would hand to its
// unrefined neighbour C + e_i, so it is written into C's own slot in the
// TIME-t coarse buffer. The coarse step then gathers it by pulling from C
// exactly as it pulls from anything else -- no per-direction test, no extra
// binding, no reorder of the fused stream+collide. For a given direction
// exactly one cell pulls from C, so the (cell, direction) slots this writes
// never collide.
//
// This pass also republishes `mac` under the refined region. `average` used
// to do that as a side effect of writing every covered cell; coalesce writes
// only the interface slots, so the renderer, readSubsampled, readStats and
// readInterfaceDiag would otherwise read a coarse field nothing maintains.

@group(0) @binding(0) var<storage, read>       f_pool    : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_coarse  : array<f32>;   // the TIME-t buffer
@group(0) @binding(2) var<storage, read>       blockSlot : array<i32>;
@group(0) @binding(3) var<storage, read>       mac_pool  : array<f32>;
@group(0) @binding(4) var<storage, read_write> mac       : array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY || gid.z >= NZ) { return; }
  let c = vec3<u32>(gid.x, gid.y, gid.z);

  // Covered cells only: this pass speaks for the coarse cells the coarse
  // solver no longer runs on.
  let b = blockOfCoarse(c);
  let slotI = blockSlot[blockIdOf(b)];
  if (slotI < 0) { return; }
  let slot = u32(slotI);

  let ncells = NX * NY * NZ;
  let cell = coarseCell(c);
  let poolPlane = arrayLength(&f_pool) / QN;
  let macPlane = arrayLength(&mac_pool) / 4u;

  // --- macroscopic projection ---------------------------------------------
  // rho: arithmetic mean of the eight children. u: mass-weighted, so the
  // reported momentum is the children's momentum exactly.
  let f0 = vec3<u32>(GHOST) + 2u * (c - b * RB);
  var rhoSum = 0f;
  var momSum = vec3<f32>(0f);
  for (var k = 0u; k < 8u; k++) {
    let off = vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
    let fc = poolCell(slot, f0 + off);
    let rho = mac_pool[0u * macPlane + fc];
    rhoSum += rho;
    momSum += rho * vec3<f32>(
      mac_pool[1u * macPlane + fc],
      mac_pool[2u * macPlane + fc],
      mac_pool[3u * macPlane + fc]);
  }
  mac[4u * cell + 0u] = rhoSum * 0.125f;
  mac[4u * cell + 1u] = momSum.x / max(rhoSum, 1e-6f);
  mac[4u * cell + 2u] = momSum.y / max(rhoSum, 1e-6f);
  mac[4u * cell + 3u] = momSum.z / max(rhoSum, 1e-6f);

  // --- coalesce ------------------------------------------------------------
  for (var i = 0u; i < QN; i++) {
    let t = vec3<u32>(
      wrapu(i32(c.x) + ex[i], NX),
      wrapu(i32(c.y) + ey[i], NY),
      wrapu(i32(c.z) + ez[i], NZ));
    // Only where a real coarse cell is waiting to pull direction i from
    // here. i = 0 lands on c itself, which is covered, so the rest cell
    // needs no special case.
    if (blockSlot[blockIdOf(blockOfCoarse(t))] >= 0) { continue; }

    // The eight fine cells covering the target coarse cell are this tile's
    // own ring: t is exactly one coarse cell outside this block, because a
    // target inside it would itself be covered.
    var s = 0f;
    for (var k = 0u; k < 8u; k++) {
      let off = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
      let p = vec3<i32>(t) * 2 + off;
      s += f_pool[i * poolPlane + poolCell(slot, localInTile3(p, b))];
    }
    f_coarse[i * ncells + cell] = s * 0.125f;
  }
}
