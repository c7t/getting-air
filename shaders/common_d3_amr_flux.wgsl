// Seam flux accumulation -- the measuring half of the M4 interface flux
// correction (plans/3D.md M3/M4). Fragment only; the entry files list every
// include.
//
// STATUS: OPT-IN (?reflux=1), NOT the default, and the reason is the most
// useful thing this file has to say.
//
// It does what it was built to do: global mass and momentum conservation
// across the interface becomes EXACT (measured -- the momentum drift a
// partially-refined run leaks stops dead and stays frozen). On a seam with
// no convex corner (?refine=slab) that also halves the field error, and the
// seam nearly disappears from the bucketed profile.
//
// On a seam WITH a convex corner (?refine=box, and any shell wrapped around
// a body) it is dramatically WORSE than no correction: the measured per-cell
// correction is |d(rho u)| ~ 9e-3 against u ~ 0.04, a 25% perturbation
// injected every step. That is not a bug in the accounting; it is a
// property of the lattice, and it is worth stating exactly.
//
// THE CORNER DOES NOT TILE. Take a coarse cell c diagonally outside a
// convex corner of the refined region, so the covered cells are those with
// x <= c.x-1 AND y <= c.y-1. On the COARSE grid exactly one channel crosses:
// f_(1,1,0) from c-(1,1,0), carrying w * V_coarse. On the FINE grid the only
// source cells that can reach the fine cells covering c are those at
// (2c.x-1, 2c.y-1, *), and D3Q19 has no (1,1,+-1), so only e = (1,1,0)
// crosses -- 2 cells (z) x 2 substeps x w * V_fine = w * V_coarse / 2.
//
// HALF. The fine channels do not tile the coarse one, so demanding that the
// two sides agree cell-by-cell at a corner is demanding that two
// discretizations agree where they provably cannot. At a flat face they DO
// tile exactly (4 cells x 2 substeps x V_fine = V_coarse), which is why
// ?refine=slab works and ?refine=box does not.
//
// This is the assumption classical Berger-Colella refluxing rests on and
// that LBM breaks: finite-volume fluxes live on faces, and fine faces tile a
// coarse face exactly. Lattice populations cross corners, and corner
// channels do not tile. It is a plausible reason the LBM AMR literature --
// AGAL included -- does not reflux.
//
// WHAT THIS EXISTS FOR. The coarse/fine interface is not conservative: the
// coarse solver's flux across the seam and the fine solver's flux across the
// same seam are computed independently and disagree, which pollutes the
// whole domain at FIRST order (measured -- benchmarks/d3.json's
// amr_interface_note). This pass measures what the FINE side actually moved
// across the seam during its two substeps; common_d3_amr_reflux.wgsl then
// measures what the COARSE side thought it moved, and corrects the
// difference onto the unrefined coarse cells.
//
// WHY MOMENTS AND NOT POPULATIONS. The obvious scheme -- replace the
// population the coarse gather pulled from inside the region with a
// fine-derived one -- does not survive contact with the lattice. A fine
// emission in direction e_i does NOT correspond to a coarse displacement
// e_i: a fine cell at the +x,-y corner of a coarse cell, streaming (1,1,0),
// lands one fine cell over in x and back inside in y, i.e. at coarse
// displacement (1,0,0). So fine emissions do not partition into coarse
// direction channels, and lumping a diagonal emission into a face channel
// conserves its mass while silently discarding its transverse momentum.
//
// Measuring MASS and MOMENTUM crossing the seam sidesteps that entirely.
// Both are moments, both are linear in f, and both are exactly what has to
// balance. The correction is then applied to the coarse cell's moments,
// which is a well-defined operation on any lattice (see the reflux pass).
//
// THE ACCOUNTING, stated once so it can be checked rather than trusted.
// Composite mass is (unrefined coarse cells) + (the fine region), the
// covered coarse cells being just the restriction's shadow of the latter.
// The coarse step is globally conservative, so over one macro step the
// unrefined set changes by exactly (coarse_out - coarse_in): what unrefined
// cells pulled OUT of covered cells, minus what covered cells pulled out of
// unrefined ones. It SHOULD change by (Fout - Fin)/8, the fine solver's own
// seam traffic. This pass accumulates (Fout - Fin)/8; the reflux pass
// computes (coarse_out - coarse_in) and applies the difference. The /8 is
// the cell-volume ratio: a fine population carries V_fine = V_coarse/8.
//
// RUN IT TWICE PER MACRO STEP, once per fine substep, against the buffer
// that substep STREAMS -- fPoolA for substep A, fPoolB for substep B.
// ACCUM=0 on the first (write) and 1 on the second (add), which is also
// what zeroes the accumulator without a separate clear pass. Substep B
// overwrites fPoolA, so the A pass cannot be deferred to the end.
//
// It is dispatched over the coarse grid and does real work only on cells
// that are unrefined AND have a refined block among the 8 blocks their 26
// neighbours span -- a SURFACE. The blockSlot lookups for that test are
// hoisted the same way common_d3_amr_step1.wgsl hoists its own.

@group(0) @binding(0) var<storage, read>       f_pool    : array<f32>;
@group(0) @binding(1) var<storage, read>       blockSlot : array<i32>;
@group(0) @binding(2) var<storage, read_write> fluxAcc   : array<f32>;

// 0 = overwrite (first substep, which is also the clear), 1 = accumulate.
override ACCUM : u32 = 0u;

fn fineDim() -> vec3<i32> { return vec3<i32>(i32(2u * NX), i32(2u * NY), i32(2u * NZ)); }

fn wrapFine3(g: vec3<i32>) -> vec3<i32> {
  let n = fineDim();
  return vec3<i32>(
    ((g.x % n.x) + n.x) % n.x,
    ((g.y % n.y) + n.y) % n.y,
    ((g.z % n.z) + n.z) % n.z);
}

fn blockOfFine(g: vec3<i32>) -> vec3<u32> {
  let RB2 = 2u * RB;
  return vec3<u32>(u32(g.x) / RB2, u32(g.y) / RB2, u32(g.z) / RB2);
}

// Tile-local index of GLOBAL FINE cell g inside the tile owning block b --
// which need not be g's own block. g is at most one fine cell outside b's
// interior (it is either interior to b, or a depth-1 ring cell of it), so
// the result is always inside [0, FB). The centred wrap is what makes this
// periodic; it needs at least 3 blocks on the axis, which main-3d.js checks
// before enabling this path.
fn localInTile(g: i32, b: u32, nFine: i32) -> i32 {
  let RB2 = i32(2u * RB);
  var d = g - i32(b) * RB2;
  if (d < -nFine / 2) { d += nFine; }
  if (d >  nFine / 2) { d -= nFine; }
  return d + i32(GHOST);
}

fn slotOfFine(g: vec3<i32>) -> i32 {
  return blockSlot[blockIdOf(blockOfFine(g))];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY || gid.z >= NZ) { return; }
  let c = vec3<u32>(gid.x, gid.y, gid.z);
  let cell = coarseCell(c);

  // A covered cell is the fine region's own shadow; the correction is for
  // the unrefined side only.
  let myBlock = vec3<u32>(c.x / RB, c.y / RB, c.z / RB);
  if (blockSlot[blockIdOf(myBlock)] >= 0) { return; }

  // Is any refined block within reach? c's 26 coarse neighbours span at most
  // 2x2x2 blocks, so this is at most 8 lookups regardless of QN -- the same
  // hoist common_d3_amr_step1.wgsl uses for its neighbour tiles.
  let nb = vec3<u32>(nbx(), nby(), nbz());
  var off = vec3<i32>(0, 0, 0);
  off.x = select(select(0, 1, (c.x + 1u) % RB == 0u), -1, c.x % RB == 0u);
  off.y = select(select(0, 1, (c.y + 1u) % RB == 0u), -1, c.y % RB == 0u);
  off.z = select(select(0, 1, (c.z + 1u) % RB == 0u), -1, c.z % RB == 0u);
  var anyRefined = false;
  for (var k = 0u; k < 8u; k++) {
    let s = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u)) * off;
    let b = vec3<u32>(
      wrapu(i32(myBlock.x) + s.x, nb.x),
      wrapu(i32(myBlock.y) + s.y, nb.y),
      wrapu(i32(myBlock.z) + s.z, nb.z));
    if (blockSlot[blockIdOf(b)] >= 0) { anyRefined = true; }
  }
  if (!anyRefined) { return; }

  let poolPlane = arrayLength(&f_pool) / QN;
  let n = fineDim();
  var dm = 0f;
  var dp = vec3<f32>(0f);

  // The 8 fine cells this coarse cell covers. They are ring cells of every
  // refined tile that reaches them, and a tile's own ring copy is what that
  // tile actually streamed, so the IN term reads the RECEIVING tile's copy
  // rather than any single canonical one -- after substep A the copies have
  // self-advanced independently and genuinely differ.
  for (var oc = 0u; oc < 8u; oc++) {
    let po = vec3<i32>(i32(oc & 1u), i32((oc >> 1u) & 1u), i32((oc >> 2u) & 1u));
    let p = vec3<i32>(c) * 2 + po;

    for (var i = 0u; i < QN; i++) {
      let e = vec3<i32>(ex[i], ey[i], ez[i]);
      let ef = vec3<f32>(f32(e.x), f32(e.y), f32(e.z));

      // OUT: the source is interior to a refined tile, so this population
      // left the fine region and arrived here. i = 0 needs no special case:
      // e is zero, the source is this same cell, and this cell's block is
      // unrefined by the test above.
      let q = wrapFine3(p - e);
      let so = slotOfFine(q);
      if (so >= 0) {
        let bq = blockOfFine(q);
        let lq = vec3<u32>(
          u32(localInTile(q.x, bq.x, n.x)),
          u32(localInTile(q.y, bq.y, n.y)),
          u32(localInTile(q.z, bq.z, n.z)));
        let f = f_pool[i * poolPlane + poolCell(u32(so), lq)];
        dm += f; dp += f * ef;
      }

      // IN: the TARGET is interior to a refined tile, so this population
      // entered the fine region from here. The value that moved is direction
      // i at p in the receiving tile's own frame, where p is a ring cell.
      let t = wrapFine3(p + e);
      let si = slotOfFine(t);
      if (si >= 0) {
        let bt = blockOfFine(t);
        let lp = vec3<u32>(
          u32(localInTile(p.x, bt.x, n.x)),
          u32(localInTile(p.y, bt.y, n.y)),
          u32(localInTile(p.z, bt.z, n.z)));
        let f = f_pool[i * poolPlane + poolCell(u32(si), lp)];
        dm -= f; dp -= f * ef;
      }
    }
  }

  // V_fine / V_coarse.
  dm *= 0.125f; dp *= 0.125f;

  if (ACCUM == 0u) {
    fluxAcc[4u * cell + 0u] = dm;
    fluxAcc[4u * cell + 1u] = dp.x;
    fluxAcc[4u * cell + 2u] = dp.y;
    fluxAcc[4u * cell + 3u] = dp.z;
  } else {
    fluxAcc[4u * cell + 0u] += dm;
    fluxAcc[4u * cell + 1u] += dp.x;
    fluxAcc[4u * cell + 2u] += dp.y;
    fluxAcc[4u * cell + 3u] += dp.z;
  }
}
