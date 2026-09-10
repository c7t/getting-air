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

fn coveredCoarse3(v: vec3<u32>) -> bool {
  return blockSlot[blockIdOf(blockOfCoarse(v))] >= 0;
}

// THE ORPHAN PASS: the other half of the coalesce, and the fix for M4.1b's
// convex-edge mass leak. Runs on the UNREFINED cells, which the coalesce
// proper never visits.
//
// A ring cell p holds, after substep B, either its interior source q's
// outflux at t + dt/2 (when q = p - e_i is interior) or at t (when p - e_i
// is itself a ring cell and q = p - 2 e_i is interior). The coalesce
// delivers what now sits in coarse cell coarse(p) by writing it into the
// slot of coarse(p) - e_i, so that the coarse pull carries it to coarse(p).
// That works only if coarse(p) - e_i is COVERED, and:
//
//   - for the t outflux it always is: p - 2 e_i is interior and
//     coarse(p - 2 e_i) == coarse(p) - e_i exactly;
//   - for the t + dt/2 outflux it need not be. That value has moved one
//     FINE cell, so coarse(p) may have advanced on only some of e_i's axes,
//     and coarse(p) - e_i then steps back OUT of the refined region on the
//     others. At a flat face it lands on another covered cell and the
//     accounting is merely smeared laterally; at a CONVEX EDGE it lands on
//     the diagonally-outside cell, which is unrefined, so no coalesce
//     thread claims p and the value is dropped when explode refreshes the
//     ring next macro-step.
//
// Measured before this pass existed: -1.2e-2 of mass per edge coarse cell
// per macro-step, against the f/2 ~ (1/36)/2 = 1.4e-2 the accounting
// predicts -- two grazing directions at an xy edge ((1,-1,0) and (-1,1,0))
// x two fine cells along the edge x one dropped population each, over the
// eight fine cells of a coarse cell. Only the EDGE leaks, and corners add
// no excess over the edges meeting there, which is what the measured
// box:bar ratio of 1.49 against their edge-length ratio of 1.50 said.
//
// SO CLAIM THEM HERE, and deliver them where they physically are rather
// than where a covered cell could have sent them. s is a real unrefined
// fluid cell whose own f_i is already in this buffer and is already pulled
// by s + e_i; the orphan is mass that transited s's volume on its way out
// of the refined region, so it ADDS to that slot. Exactly conservative --
// the value is removed from the fine grid by explode either way -- and
// delivered to the right cell in the right direction, so momentum is right
// too. The alternative, gathering by ORIGIN instead of destination, is also
// exactly conservative and was measured 4.6x WORSE in the field: at a
// convex corner it hands a coarse cell mass that physically sits in a
// different neighbour.
//
// NO DOUBLE COUNTING, structurally: ring cell p in direction i is claimed
// by coarse(p) - e_i, and that cell is either covered (the coalesce proper)
// or not (here). The two are the same test with opposite sign.
fn coalesceOrphans(c: vec3<u32>) {
  // A claimable p has an INTERIOR source one fine cell away, so a covered
  // coarse cell is within one cell of c and a refined block within one
  // block. 27 block-granular loads, against 19 x 8 pool addresses.
  let b = blockOfCoarse(c);
  let nb = vec3<u32>(nbx(), nby(), nbz());
  var near = false;
  for (var d = 0u; d < 27u; d++) {
    let o = vec3<i32>(i32(d % 3u), i32((d / 3u) % 3u), i32(d / 9u)) - 1;
    let nbb = vec3<u32>(
      wrapu(i32(b.x) + o.x, nb.x),
      wrapu(i32(b.y) + o.y, nb.y),
      wrapu(i32(b.z) + o.z, nb.z));
    near = near || (blockSlot[blockIdOf(nbb)] >= 0);
  }
  if (!near) { return; }

  let ncells = NX * NY * NZ;
  let cell = coarseCell(c);
  let poolPlane = arrayLength(&f_pool) / QN;

  // i = 0 is the rest vector: p - e_i = p, which is in c and unrefined, so
  // it can never be claimable. Skipped rather than tested eight times.
  for (var i = 1u; i < QN; i++) {
    let ei = vec3<i32>(ex[i], ey[i], ez[i]);
    let t = vec3<u32>(
      wrapu(i32(c.x) + ei.x, NX),
      wrapu(i32(c.y) + ei.y, NY),
      wrapu(i32(c.z) + ei.z, NZ));
    // A covered target means the coarse grid does not solve at t and this
    // slot is exploded rather than pulled; nothing to hand over.
    if (coveredCoarse3(t)) { continue; }

    var s = 0f;
    for (var k = 0u; k < 8u; k++) {
      let off = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
      let p = wrapFine3(vec3<i32>(t) * 2 + off);
      // Only the t + dt/2 case can be orphaned (see above), so the source
      // is exactly one fine cell back, and the tile that computed p is the
      // one owning that source -- not c's block, which has no tile at all.
      let q = wrapFine3(p - ei);
      let qb = blockOfFine(q);
      let qs = blockSlot[blockIdOf(qb)];
      if (qs < 0) { continue; }
      s += f_pool[i * poolPlane + poolCell(u32(qs), localInTile3(p, qb))];
    }
    if (s != 0f) { f_coarse[i * ncells + cell] += s * 0.125f; }
  }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY || gid.z >= NZ) { return; }
  let c = vec3<u32>(gid.x, gid.y, gid.z);

  // Covered cells only: this pass speaks for the coarse cells the coarse
  // solver no longer runs on.
  let b = blockOfCoarse(c);
  let slotI = blockSlot[blockIdOf(b)];
  if (slotI < 0) { coalesceOrphans(c); return; }
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
