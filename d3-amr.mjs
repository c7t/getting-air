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

// --- structural invariants (plans/3D.md M4.2, risk #2) ---------------------
//
// Ported from the 2D checker (tools/lib/amr-invariants.js and
// main-cylinder-amr.js's debugCheck21Balance) and deliberately written as
// PURE functions over block sets, so tools/test-d3-amr.js can exercise them
// with no server, browser or GPU -- including on inputs that VIOLATE the
// invariant, which is the only way to know a checker checks anything.
//
// WHY THIS LANDS BEFORE THE MANAGER. plans/3D.md sec 7 risk #2: the 2D
// versions of manage/manage_pool carry three separately-documented
// live-verified balance bugs. A checker written after the manager gets
// written to agree with it; a checker written before is an independent
// statement of what the manager must achieve.

// The 4 (of 8) octant children covering one FACE of their parent: the face
// on axis `a` in direction `s` is covered by the children whose octant bit
// on `a` is (s > 0 ? 1 : 0). Recursing with the SAME face picks the correct
// ever-deeper sliver along it, not the tile's maximum depth anywhere.
function faceChildren(a, s) {
  const want = s > 0 ? 1 : 0;
  const out = [];
  for (let k = 0; k < 8; k++) {
    const q = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
    if (q[a] === want) out.push(q);
  }
  return out;
}

const FACES = [[0, -1], [0, 1], [1, -1], [1, 1], [2, -1], [2, 1]];

// 2:1 BALANCE. `levelSets[m]` is a Set of "bx,by,bz" keys naming the blocks
// active at level m (m >= 1); level 0 is the dense grid and is implicit.
// `nbAt(m)` gives that level's block counts per axis, which double each
// level. Every LEAF tile's face neighbour must sit within one level of it.
//
// WITH ONE REFINED LEVEL THIS CANNOT FAIL, and saying so is the point: a
// level-1 leaf's neighbour is either level 1 or level 0, and |1 - 1| and
// |1 - 0| are both <= 1. So a green result here today is evidence the
// machinery runs, NOT evidence that balance holds -- there is nothing yet
// that could break it. It becomes a real gate the moment M4.2 makes
// refinement dynamic or M5 adds a level, which is exactly why it is written
// now, against the shape those will have.
export function check21Balance(levelSets, nbAt, { levels }) {
  const has = (m, x, y, z) => m >= 1 && m < levels && levelSets[m] && levelSets[m].has(`${x},${y},${z}`);
  const hasChild = (m, x, y, z) => has(m + 1, x * 2, y * 2, z * 2);

  // The deepest level actually present at (x,y,z) walking UP from m.
  const ancestorDepth = (m, x, y, z) => {
    let lv = m, a = x, b = y, c = z;
    while (lv >= 1) {
      if (has(lv, a, b, c)) return lv;
      a = Math.floor(a / 2); b = Math.floor(b / 2); c = Math.floor(c / 2);
      lv--;
    }
    return 0;
  };
  // The deepest level present along one FACE of a tile, walking DOWN.
  const faceMaxDepth = (m, x, y, z, a, s) => {
    if (!hasChild(m, x, y, z)) return m;
    let d = m;
    for (const q of faceChildren(a, s)) {
      d = Math.max(d, faceMaxDepth(m + 1, x * 2 + q[0], y * 2 + q[1], z * 2 + q[2], a, s));
    }
    return d;
  };

  const violations = [];
  for (let m = 1; m < levels; m++) {
    if (!levelSets[m]) continue;
    const nb = nbAt(m);
    for (const key of levelSets[m]) {
      const [bx, by, bz] = key.split(',').map(Number);
      if (hasChild(m, bx, by, bz)) continue;   // not a leaf; checked deeper
      for (const [a, s] of FACES) {
        const n = [bx, by, bz];
        n[a] = ((n[a] + s) % nb[a] + nb[a]) % nb[a];   // the block grid is periodic
        const nDepth = has(m, n[0], n[1], n[2])
          ? faceMaxDepth(m, n[0], n[1], n[2], a, -s)   // the neighbour's facing side
          : ancestorDepth(m, n[0], n[1], n[2]);
        if (Math.abs(m - nDepth) > 1) {
          violations.push({ level: m, block: [bx, by, bz], neighbour: n.slice(), nDepth, axis: a, dir: s });
        }
      }
    }
  }
  return { violations, levels, counts: levelSets.map(s => (s ? s.size : 0)) };
}

// GEOMETRY-FORCED REFINEMENT, checked at CELL granularity against the SDF
// directly. This is deliberately NOT refineNearBody's own test: that samples
// a block's eight corners and centre, so it is the thing being checked, and
// re-running it here would assert only that a function equals itself. Its
// own header names the case it can miss -- a body small against a block --
// and a cell-granular scan is precisely the independent route that would
// catch it.
//
// The requirement: no coarse cell whose centre is within `margin` of the
// body may sit in an unrefined block. Same sign convention as
// refineNearBody, so cells INSIDE the body (phi < 0) are included.
export function checkGeometryCoverage(pool, blockSlot, sdf, margin) {
  const [NX, NY, NZ] = pool.dims;
  const violations = [];
  let required = 0;
  for (let z = 0; z < NZ; z++) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        if (sdf([x, y, z]) > margin) continue;
        required++;
        const id = pool.blockId(Math.floor(x / pool.rb), Math.floor(y / pool.rb), Math.floor(z / pool.rb));
        if (blockSlot[id] < 0 && violations.length < 32) violations.push({ cell: [x, y, z], block: id });
      }
    }
  }
  return { violations, required };
}

// 2:1 CASCADE. Turns a WANTED set into the smallest superset of it that is
// 2:1 balanced -- the manager's forced-refinement rule, and the exact
// counterpart of check21Balance above: that one reports violations, this one
// removes them, and tools/test-d3-amr.js runs them against each other rather
// than against a transcription of either.
//
// ONE RULE, AND BOTH HALVES OF THE MILESTONE FALL OUT OF IT:
//
//     present(m, b)  =>  present(m-1, parent(n))  for each face neighbour
//                        n of b at level m, and for b itself.
//
// Read forwards it is "refine forced by a neighbour that wants a deeper
// child"; read backwards -- a block the criterion did NOT want, added back
// -- it is "coarsen blocked by a neighbour that has one". They are not two
// mechanisms. That is why this runs on the WANT set, between `decide` and
// `drain`, instead of being a test bolted onto each of coarsen and refine:
// by the time those passes run the answer is already balanced and they need
// to know nothing about levels.
//
// WHY THAT RULE IS SUFFICIENT, since "add neighbours until it looks right"
// is the shape of the three live balance bugs the 2D manager carries. Take
// any leaf (m, b) and a face (a, s), neighbour n:
//   - n absent at level m: check21Balance walks UP and finds the deepest
//     ancestor. The rule put parent(n) there, so the gap is exactly 1.
//   - n present at level m with a level-(m+2) grandchild g on the SHARED
//     face: g's own face neighbour one step back across the seam is a
//     level-(m+2) block inside b, so the rule demands its level-(m+1)
//     parent, which is a child of b -- and then b is not a leaf, contrary
//     to assumption. So no leaf can face a 2-level jump.
// Deeper slivers are ruled out by applying the same argument at each level,
// which is what the sweep below does.
//
// THAT SECOND CASE ONLY CLOSES BECAUSE REFINEMENT IS OCTET-COMPLETE, and
// the first draft of this function was wrong for exactly that reason. It
// added parent(g') as a single block, which can be any octant of b; the
// checker's hasChild tests octant (0,0,0) alone, still called b a leaf, and
// reported the violation the cascade was supposed to have removed. The
// checker was right and the cascade was wrong -- see `ensure`.
//
// THE SWEEP IS DEEPEST-FIRST AND RUNS ONCE. An addition at level m-1 is
// made before the loop reaches m-1, so its own closure is taken in the same
// pass; there is no fixed-point iteration to get wrong. The test asserts
// idempotence rather than trusting that argument.
//
// VACUOUS AT ?levels=2, PROVABLY: the loop starts at m = levels-1 and stops
// above m = 1, because a level-1 block's parent level is the dense L0 grid,
// which is present everywhere by definition. At levels=2 it therefore does
// not execute at all and this function is the identity -- which is why
// there is no `balance` entry point in shaders/common_d3_manage.wgsl today
// rather than a kernel that provably does nothing. tools/test-d3-amr.js
// asserts the identity on random one-level sets so that claim is checked
// and not merely argued.
export function cascade21(wantSets, nbAt, { levels }) {
  const sets = [null];
  for (let m = 1; m < levels; m++) sets[m] = new Set(wantSets[m] || []);
  const forced = [];
  // REFINEMENT IS OCTET-COMPLETE FROM LEVEL 2 DOWN, and getting this wrong
  // is what the first draft did. A level-m block exists because its level-
  // (m-1) parent SPAWNED IT along with its seven siblings -- the 2D pool
  // manager says it outright ("a parent slot may only spawn a level-(m+1)
  // quad"), because a tile is allocated per block and there is no such
  // thing as a quarter-refined parent. check21Balance's hasChild leans on
  // it too: it tests octant (0,0,0) alone and would read a parent holding
  // only octant (1,0,0) as a LEAF, so a cascade that added single blocks
  // produces a tree the checker then reports as unbalanced -- which is what
  // it did, and the checker was right.
  //
  // Level 1 is the exception and not an inconsistency: its parent is the
  // dense L0 grid, which is present everywhere, so level-1 blocks are
  // refined individually. That is exactly what refineNearBody builds.
  const ensure = (m, b, because) => {
    const parent = [b[0] >> 1, b[1] >> 1, b[2] >> 1];
    for (let k = 0; k < (m >= 2 ? 8 : 1); k++) {
      const c = m >= 2
        ? [parent[0] * 2 + (k & 1), parent[1] * 2 + ((k >> 1) & 1), parent[2] * 2 + ((k >> 2) & 1)]
        : b;
      const key = `${c[0]},${c[1]},${c[2]}`;
      if (sets[m].has(key)) continue;
      sets[m].add(key);
      forced.push({ level: m, block: c, octet: m >= 2 ? parent : null, because });
    }
  };
  for (let m = levels - 1; m >= 2; m--) {
    // COMPLETE THIS LEVEL'S OCTETS FIRST. The criterion is evaluated per
    // block and nothing in it looks sideways, so a want for one child with
    // no siblings is a state the manager can genuinely produce -- and it is
    // not a state the POOL can be in, for the same reason the header gives:
    // the parent spawns the whole octet or none of it. Completing it here
    // rather than rejecting it says what the want MEANS. Octets added at
    // this level by the level below have already come through `ensure`, so
    // this only ever finds the caller's own input.
    for (const key of [...sets[m]]) {
      ensure(m, key.split(',').map(Number), { siblingOf: key });
    }
    const nb = nbAt(m);
    for (const key of sets[m]) {
      const b = key.split(',').map(Number);
      // The tree property first: a block with no parent is not a refinement
      // of anything. The manager can produce one -- the criterion is
      // evaluated per level and nothing in it looks up.
      ensure(m - 1, [b[0] >> 1, b[1] >> 1, b[2] >> 1], { parentOf: b.slice(), level: m });
      for (const [a, s] of FACES) {
        const n = b.slice();
        n[a] = ((n[a] + s) % nb[a] + nb[a]) % nb[a];   // periodic, like the checker
        ensure(m - 1, [n[0] >> 1, n[1] >> 1, n[2] >> 1], { neighbourOf: b.slice(), level: m, axis: a, dir: s });
      }
    }
  }
  return { sets, forced, counts: sets.map((s, m) => (m === 0 ? null : s.size)) };
}

// --- the pool-parent path (plans/3D.md M5.1a) -------------------------------
//
// THE UNIFORM TILE SHAPE, MADE CONCRETE. A tile's interior is 2*RB cells at
// its OWN level, so it spans RB * 2^(1-m) L0 units: a level-m tile is one
// OCTANT of a level-(m-1) tile's footprint at doubled density. Every level
// therefore has the same FB, the same ring, and the same kernels -- the
// level-2 tile is the level-1 tile moved down a rung.
//
// That is what makes this function a one-liner instead of a second
// addressing scheme: LEVEL m'S TILING IS LEVEL 1'S TILING OF A DOMAIN
// 2^(m-1) TIMES LARGER. So `resolveSource`, `toGlobalFine`, `fromGlobalFine`
// and `cellIndex` work at any depth with no change at all, which is exactly
// the claim tools/test-d3-amr.js re-runs at levels 2 and 3 against the
// independent global-coordinate route.
//
// 2D had to CHANGE TOPOLOGY here (plans/AMR-multilevel.md M5, decision 2):
// its L1 was footprint-preserving, so a level-3 tile would have needed 4x
// the linear size of a level-1 tile. 3D never had that problem. Do not port
// the fix for it.
export function poolAtLevel(pool, m, { maxSlots } = {}) {
  if (m < 1) throw new Error(`level ${m} has no pool (level 0 is the dense grid)`);
  const s = 2 ** (m - 1);
  return makePool({ dims: pool.dims.map(d => d * s), rb: pool.rb, maxSlots });
}

// A level-m block's parent at level m-1, and which of the parent's eight
// octants it is. Valid for m >= 2 only: a level-1 block's parent is the
// dense L0 grid, which is not a tile and is addressed spatially instead --
// the two schemes coexist and are not unified, inherited straight from the
// 2D arc.
export function parentOfBlock(b) { return [b[0] >> 1, b[1] >> 1, b[2] >> 1]; }
export function octantOfBlock(b) { return [b[0] & 1, b[1] & 1, b[2] & 1]; }

// WHERE A CHILD SITS INSIDE ITS PARENT'S INTERIOR, from the octant bits
// alone -- no spatial lookup, no parent-chain walk. The parent's interior is
// 2*RB cells on each axis and the octant halves it, so the child covers
// parent-local indices [GHOST + q*RB, GHOST + q*RB + RB) on that axis.
//
// The return value is exactly the `origin` argument fineToCoarseUnit already
// takes, so a child-local index maps into PARENT-LOCAL continuous
// coordinates through the same stencil function the dense-parent path uses.
// One function, two callers. A second copy of that arithmetic specialized to
// pool parents is how the two would drift, and the drift would look like a
// seam rather than like a bug.
export function octantOrigin(rb, q) { return GHOST + q * rb; }

// THE REFINEMENT HIERARCHY (plans/3D.md M5.1b). Per-level blockSlot maps for
// a hierarchy `levels` deep, from ONE criterion.
//
// THE CRITERION IS EVALUATED AT THE FINEST LEVEL ONLY, and every coarser
// level's set is whatever `cascade21` requires around it. That is not a
// simplification, it is the correct statement of what geometry-forced
// refinement asks for: "push the body's surface and its immediate
// surroundings onto the FINEST configured level" (sec 1.3). The intermediate
// shells exist solely as the buffer 2:1 balance demands, so deriving them
// from the closure gives exactly that and nothing more -- and cascade21's
// minimality test is the proof there is no halo of unnecessary tiles.
//
// It also disposes of a worry the 2D arc carried into its M10: "retune
// REFINE_THRESH per level, a level-2 region at the surface with level-1 as
// the buffer shell". For a geometry-forced criterion there is nothing per
// level to tune -- there is one margin, at the finest level, and the shells
// follow. A vorticity-driven criterion will need per-level thresholds; this
// one does not, and that should not be discovered again later.
//
// BIT-IDENTICAL AT levels=2 BY CONSTRUCTION, which is M5.1b's gate: the
// finest level IS level 1, the L0-unit rescale is a no-op, cascade21 is the
// identity (M4.2b-iv), and slots are handed out in ascending block-id order
// exactly as refineWhere does. tools/test-d3-amr.js asserts the equality
// against refineWhere/refineNearBody rather than trusting that reading.
//
// `want` is called with the SAME shape refineWhere's predicate takes and in
// L0 CELL UNITS at every level, so one predicate serves all depths. A
// level-m view's own cells are 2^(m-1) times smaller than an L0 cell (see
// poolAtLevel), and getting that rescale wrong would refine a shell of the
// wrong physical size -- which is why the test drives it at depth 3 and 4,
// where the factor is not 1.
export function refineHierarchy(pool, { levels, want, maxSlots }) {
  if (levels < 2) return { levels: 0, byLevel: [null], sets: [null] };
  const deepest = levels - 1;
  const view = poolAtLevel(pool, deepest);
  const scale = 2 ** (deepest - 1);
  const finest = new Set();
  for (let id = 0; id < view.nBlocks; id++) {
    const [bx, by, bz] = view.blockOf(id);
    const lo = [bx * view.rb, by * view.rb, bz * view.rb];
    const hi = lo.map(c => c + view.rb);
    const mid = lo.map(c => c + view.rb / 2);
    const l0 = (v) => v.map(c => c / scale);
    if (want({ level: deepest, bx, by, bz, id, lo: l0(lo), hi: l0(hi), mid: l0(mid) })) {
      finest.add(`${bx},${by},${bz}`);
    }
  }
  const wants = [null];
  for (let m = 1; m < levels; m++) wants[m] = (m === deepest ? finest : new Set());
  const closed = cascade21(wants, (m) => poolAtLevel(pool, m).nb, { levels });

  const byLevel = [null];
  for (let m = 1; m < levels; m++) {
    const lv = poolAtLevel(pool, m, { maxSlots: maxSlots && maxSlots[m] });
    // Ascending block id, the same deterministic order refineWhere uses --
    // a run has to be reproducible, and the 2D pool's atomicSub free list is
    // exactly why CLAUDE.md records its Cd as reproducible only to ~1e-3.
    const alloc = refineWhere(lv, ({ bx, by, bz }) => closed.sets[m].has(`${bx},${by},${bz}`));
    byLevel[m] = { level: m, pool: lv, ...alloc };
  }
  return { levels, byLevel, sets: closed.sets, forced: closed.forced };
}
