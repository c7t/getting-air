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
// binding 4 is the PARENT's macroscopic array, declared by whichever of
// common_d3_parentmac_{dense,pool}.wgsl this module was assembled with --
// the two have different layouts AND different types, and this body is
// deliberately unable to name either. See those files.

// TIMING ONLY (?orphans=0). Skips the orphan pass, which makes the interface
// WRONG -- it reinstates M4.1b's convex-edge mass leak and its uniform-flow
// inconsistency, both measured and both large. It exists so the pass can be
// priced by differencing two runs of one build, which is the attribution
// method plans/perf-characterization.md insists on: per-pass GPU timestamps
// are not trustworthy at this granularity, so a pass is costed by removing
// it. Never set this to run physics.
override ORPHANS : u32 = 1u;

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
  if (ORPHANS == 0u) { return; }
  // GUARD AT CELL GRANULARITY, NOT BLOCK. Both orphan kinds need a COVERED
  // cell among c's own lattice neighbours, and the proof is the same for
  // both. Write coarse(x + e_i) = coarse(x) + d, where d keeps e_i's
  // component on the axes the child's parity carries across and zeroes the
  // rest -- so d is a sub-vector of e_i, and every sub-vector of a lattice
  // direction is itself a lattice direction (or zero).
  //
  //   IN  needs coarse(q + e_i) covered for a child q of c. That cell is
  //       c + d; d = 0 would mean c itself, which is unrefined here.
  //   OUT needs coarse(p - e_i) covered where coarse(p) = c + e_i. That
  //       cell is c + (e_i - d), the complement, and it is zero only if the
  //       covered cell is c again.
  //
  // So "some lattice neighbour of c is covered" is necessary for either.
  // This costs the same QN loads the old 27-BLOCK test cost and admits ~5x
  // fewer cells to the QN x 8 loop below: the old test let through every
  // cell of every block touching the refined region -- a shell RB cells
  // thick -- where only a shell ONE cell thick can hold an orphan.
  //
  // MEASURED AT WITHIN NOISE, and kept anyway. On a busy desktop
  // tools/bench-d3-interface.js could not separate the two guards: an
  // IDENTICAL build (?orphans=0, which compiles neither) moved 7% between
  // back-to-back runs, which is the same size as the difference being
  // claimed. It is kept because it is strictly less work for the same number
  // of loads, not because a number says so -- and that distinction is
  // recorded rather than dressed up as a measurement.
  var near = false;
  for (var i = 1u; i < QN; i++) {
    near = near || coveredCoarse3(vec3<u32>(
      wrapu(i32(c.x) + ex[i], NX),
      wrapu(i32(c.y) + ey[i], NY),
      wrapu(i32(c.z) + ez[i], NZ)));
  }
  if (!near) { return; }

  // Stride from the binding, not NX*NY*NZ -- identical for a dense parent
  // and correct for a pool one (M5.2b).
  let ncells = arrayLength(&f_coarse) / QN;
  let cell = u32(parentIndex(c));
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
    var nIn = 0f;
    for (var k = 0u; k < 8u; k++) {
      let off = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
      let p = wrapFine3(vec3<i32>(t) * 2 + off);
      // Only the t + dt/2 case can be orphaned (see above), so the source
      // is exactly one fine cell back, and the tile that computed p is the
      // one owning that source -- not c's block, which has no tile at all.
      let q = wrapFine3(p - ei);
      let qb = blockOfFine(q);
      let qs = blockSlot[blockIdOf(qb)];
      if (qs >= 0) {
        s += f_pool[i * poolPlane + poolCell(u32(qs), localInTile3(p, qb))];
      }
      // IN-ORPHAN, the mirror of the above and the other half of M4.1c's
      // edge fix. common_d3_amr_explode.wgsl injects c's population into
      // any of c's OWN children whose one-fine-step destination is inside
      // the refined region, even though c + e_i is unrefined. That mass has
      // left the coarse grid, so it has to be taken OFF this slot -- exactly
      // the amount injected, one eighth of f_i(c) per child, uniform.
      let ch = wrapFine3(vec3<i32>(c) * 2 + off + ei);
      if (blockSlot[blockIdOf(blockOfCoarse(coarseOfFine(ch)))] >= 0) { nIn += 1f; }
    }
    // One write. In a uniform flow the two orphan counts are equal and this
    // is exactly zero, which is the CONSISTENCY the OUT-orphan alone broke:
    // adding half-step exits without removing half-step entries handed the
    // receiving coarse cell more than a uniform state can hold.
    let net = s - nIn * f_coarse[i * ncells + cell];
    if (net != 0f) { f_coarse[i * ncells + cell] += net * 0.125f; }
  }
}

// ONE PARENT CELL'S WORTH OF WORK. The `main` that calls this differs
// between a dense parent and a pool one -- and ONLY in how `c` is obtained,
// which is why the body lives here rather than being forked (M5.2b).
//
//   dense parent   one thread per L0 cell, dispatched over the grid.
//   pool parent    the parent is not a grid, so the dispatch walks parent
//                  TILES and each thread takes one INTERIOR cell of one.
//                  Interior only, because a parent's ring is not a cell the
//                  parent solves -- some other parent tile owns it, and that
//                  tile's own thread is the one that speaks for it.
//
// The orphan pass comes along for free in both: "unrefined parent cells near
// the seam" are exactly the parent cells no child covers, and those are
// visited either way.
fn coalesceAt(c: vec3<u32>) {
  // Covered cells only: this pass speaks for the coarse cells the coarse
  // solver no longer runs on.
  let b = blockOfCoarse(c);
  let slotI = blockSlot[blockIdOf(b)];
  if (slotI < 0) { coalesceOrphans(c); return; }
  let slot = u32(slotI);

  // Stride from the binding, not NX*NY*NZ -- identical for a dense parent
  // and correct for a pool one (M5.2b).
  let ncells = arrayLength(&f_coarse) / QN;
  let cell = u32(parentIndex(c));
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
  // One call, and this body cannot get the layout wrong because it cannot
  // name the array. See common_d3_parentmac_dense.wgsl.
  let inv = 1f / max(rhoSum, 1e-6f);
  parentMacStore(cell, vec4<f32>(rhoSum * 0.125f, momSum.x * inv, momSum.y * inv, momSum.z * inv));

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
