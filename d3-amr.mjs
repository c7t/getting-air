// Octree pool geometry for the 3D AMR solver -- plans/3D.md M3.
//
// One fine level (N=2) over the dense L0 grid. The L0 domain is cut into
// blocks of RB^3 coarse cells; a block that is refined gets a SLOT in the
// fine pool holding FB^3 fine cells, where FB = 2*RB + 2*GHOST -- the
// doubled interior plus a materialized ghost ring on every face.
//
// THE RING IS KEPT. plans/3D.md sec 2.1 settles this: ghost-free was built
// and measured in 2D (5-9% at N=3 against a ~10% floor), it imposes a
// corner-balance refinement constraint this codebase never had, and it
// gives up the ring's graceful degradation while refinement converges.
// GHOST=2 is load-bearing rather than slack: each level takes two substeps
// per parent step with no ring refresh between them, and two layers let the
// ring self-advance (substep A's gather on a depth-1 ring cell reaches into
// depth 2, so depth 1 is still valid for substep B).
//
// WHAT THIS MODULE IS FOR. The addressing is the part of an AMR port that
// fails silently: an off-by-one in the fine<->coarse mapping, or a
// neighbour-tile lookup that picks the wrong tile, produces a plausible
// flow with a seam nobody sees until a validation number moves. So the
// mapping lives here, in one place, with tools/test-d3-amr.js checking it
// by an INDEPENDENT route -- global fine coordinates -- rather than by
// re-running the same arithmetic. shaders/common_d3_pool.wgsl mirrors it.
//
// Coordinate systems, named once so the rest can be terse:
//
//   COARSE      integer L0 cell, [0, NX) x [0, NY) x [0, NZ).
//   BLOCK       integer block, [0, NBX) etc., NBX = NX / RB.
//   TILE-LOCAL  integer fine cell within a slot, [0, FB)^3. Interior is
//               [GHOST, GHOST + 2*RB); anything else is ring.
//   GLOBAL FINE integer fine cell over the whole domain, [0, 2*NX) etc.
//               This is the frame in which "which tile owns this cell" is a
//               single division, and it is the independent route the test
//               checks the tile-local arithmetic against.
//   COARSE-UNIT continuous position in L0 cell units -- what the body SDF
//               and the interpolation stencil work in.

export const GHOST = 2;

// Cell-centred refinement: the two fine children of coarse cell `c` sit at
// c - 1/4 and c + 1/4. So tile-local fine index j (interior j = GHOST is the
// first child of the block's first coarse cell) maps to coarse-unit
// position origin - 0.25 + 0.5*(j - GHOST). Ring cells continue the same
// line, which is why the formula takes a signed index.
export function fineToCoarseUnit(j, origin) {
  return origin - 0.25 + 0.5 * (j - GHOST);
}

// Inverse, for a position known to land on a fine-cell centre.
export function coarseUnitToFine(p, origin) {
  return Math.round((p - origin + 0.25) / 0.5) + GHOST;
}

export function makePool({ dims, rb = 4, maxSlots }) {
  const [NX, NY, NZ] = dims;
  for (const [n, name] of [[NX, 'NX'], [NY, 'NY'], [NZ, 'NZ']]) {
    if (n % rb !== 0) throw new Error(`${name}=${n} is not a multiple of RB=${rb}`);
  }
  const NBX = NX / rb, NBY = NY / rb, NBZ = NZ / rb;
  const FB = 2 * rb + 2 * GHOST;
  return {
    dims: [NX, NY, NZ], rb, GHOST, FB,
    nb: [NBX, NBY, NBZ],
    nBlocks: NBX * NBY * NBZ,
    tileCells: FB * FB * FB,
    maxSlots: maxSlots ?? NBX * NBY * NBZ,
    blockId: (bx, by, bz) => (bz * NBY + by) * NBX + bx,
    blockOf: (id) => [id % NBX, Math.floor(id / NBX) % NBY, Math.floor(id / (NBX * NBY))],
    // Tile-local fine cell -> index within the pool.
    cellIndex: (slot, fx, fy, fz) => slot * FB * FB * FB + (fz * FB + fy) * FB + fx,
    isInterior: (j) => j >= GHOST && j < GHOST + 2 * rb,
  };
}

// Assigns slots to the blocks a predicate selects. Deterministic order
// (block id ascending) so a run is reproducible -- unlike the 2D pool's
// atomicSub free list, whose run-to-run slot assignment is exactly why
// CLAUDE.md records AMR Cd as reproducible only to ~1e-3. Static refinement
// has no reason to inherit that.
export function refineWhere(pool, predicate) {
  const blockSlot = new Int32Array(pool.nBlocks).fill(-1);
  const slotToBlock = new Int32Array(pool.maxSlots).fill(-1);
  let n = 0;
  for (let id = 0; id < pool.nBlocks; id++) {
    const [bx, by, bz] = pool.blockOf(id);
    // Predicate takes the block's COARSE-CELL bounds, inclusive-exclusive,
    // plus its centre -- everything a geometric test needs without the
    // caller redoing this arithmetic.
    const lo = [bx * pool.rb, by * pool.rb, bz * pool.rb];
    const hi = [lo[0] + pool.rb, lo[1] + pool.rb, lo[2] + pool.rb];
    const mid = [lo[0] + pool.rb / 2, lo[1] + pool.rb / 2, lo[2] + pool.rb / 2];
    if (!predicate({ bx, by, bz, id, lo, hi, mid })) continue;
    if (n >= pool.maxSlots) throw new Error(`refinement needs more than maxSlots=${pool.maxSlots} tiles`);
    blockSlot[id] = n;
    slotToBlock[n] = id;
    n++;
  }
  return { blockSlot, slotToBlock, activeSlots: n };
}

// Blocks whose coarse-cell box comes within `margin` coarse cells of the
// body surface. The geometry-forced refinement of plans/3D.md sec 1.3, in
// its static form: every leaf near the body must be at the finest level.
export function refineNearBody(pool, sdf, margin) {
  return refineWhere(pool, ({ lo, hi }) => {
    // Distance from the body to the block's box, evaluated at the closest
    // point of the box to the body centre is NOT enough for a general SDF,
    // so sample the box corners and centre and take the minimum |phi|. With
    // RB=4 the box is small against the bodies here and this is exact
    // enough; a body smaller than a block would need a finer test, which
    // the caller would notice as a body that fails to refine.
    let best = Infinity;
    for (let i = 0; i < 8; i++) {
      const p = [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]];
      best = Math.min(best, sdf(p));
    }
    best = Math.min(best, sdf([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]));
    return best <= margin;
  });
}

// --- neighbour resolution -------------------------------------------------
//
// THE piece that fails silently. A fine cell's streaming source may leave
// its own tile's interior; DIRECT_GHOST resolves it against the OWNING
// same-level tile instead of reading a materialized ghost value. This is
// the host statement of that mapping, mirrored by
// shaders/common_d3_pool.wgsl, and tools/test-d3-amr.js checks BOTH against
// the global-fine-coordinate route, which shares no arithmetic with either.
//
// Returns { slot, fx, fy, fz } in the owning tile's local frame, or null if
// no tile owns it (a coarse/fine interface -- the ring's one remaining job,
// see plans/3D.md sec 2.1).
export function resolveSource(pool, blockSlot, blockXYZ, src) {
  const { rb, FB, nb } = pool;
  const RB2 = 2 * rb;
  const out = [0, 0, 0];
  const nbr = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const s = src[a];
    if (s < GHOST) { nbr[a] = -1; out[a] = s + RB2; }
    else if (s >= GHOST + RB2) { nbr[a] = 1; out[a] = s - RB2; }
    else { nbr[a] = 0; out[a] = s; }
  }
  if (nbr[0] === 0 && nbr[1] === 0 && nbr[2] === 0) {
    return { slot: null, own: true, fx: out[0], fy: out[1], fz: out[2] };
  }
  // The block grid is periodic, matching the 2D kernels.
  const b = [0, 1, 2].map(a => (blockXYZ[a] + nbr[a] + nb[a]) % nb[a]);
  const slot = blockSlot[pool.blockId(b[0], b[1], b[2])];
  if (slot < 0) return null;
  return { slot, own: false, fx: out[0], fy: out[1], fz: out[2] };
}

// The independent route: tile-local -> GLOBAL fine coordinate, where
// ownership is one division and there is no ring at all. Used by the test
// to check resolveSource, and by nothing else -- deliberately, so the two
// cannot drift into agreement.
export function toGlobalFine(pool, blockXYZ, local) {
  const RB2 = 2 * pool.rb;
  return [0, 1, 2].map(a => blockXYZ[a] * RB2 + (local[a] - GHOST));
}

export function fromGlobalFine(pool, g) {
  const RB2 = 2 * pool.rb;
  const nFine = [0, 1, 2].map(a => pool.dims[a] * 2);
  const w = [0, 1, 2].map(a => ((g[a] % nFine[a]) + nFine[a]) % nFine[a]);
  const b = [0, 1, 2].map(a => Math.floor(w[a] / RB2));
  const l = [0, 1, 2].map(a => w[a] - b[a] * RB2 + GHOST);
  return { block: b, local: l };
}

// Cells a refined region stores, against the coarse cells it covers -- the
// (FB/RB)^d figure plans/3D.md sec 2.1 tabulates. Reported by the page so
// the memory cost of an RB choice is visible rather than inferred.
export function storageRatio(pool) {
  return Math.pow(pool.FB / pool.rb, 3);
}
