// Quadtree pool geometry for the 2D AMR solver -- plans/2D-backport.md B0.
//
// WHAT THIS MODULE IS FOR, and it is not tidiness. The addressing is the
// part of an AMR solver that fails SILENTLY: an off-by-one in the
// fine<->coarse mapping, a neighbour-tile lookup that picks the wrong tile,
// or a child tile registered at the wrong physical origin all produce a
// plausible flow with a seam nobody sees until a validation number moves.
// This repo has paid for each of those at least once --
// shaders/amr_manage_pool.wgsl:380-403 records a live-verified case where a
// quadrant-1 tile "sat overlapping half of quadrant 0's true territory",
// which left 2:1 balance (an index-only check) and the field-finite check
// perfectly clean while the level-2 force came out wrong-sign and
// wrong-magnitude.
//
// So the mapping lives here, once, as PURE functions over plain arrays, and
// tools/test-amr2d.js checks it by an INDEPENDENT route -- global fine
// coordinates, where ownership is one division with no ring and no offsets
// -- rather than by re-running the same arithmetic. Nothing here touches a
// GPU, a DOM or a server, so it runs in `make check`.
//
// THE 3D FORK GOT THIS FIRST AND IT IS WHY ITS M4/M5 SURVIVED. d3-amr.mjs +
// tools/test-d3-amr.js is the sibling; this file deliberately mirrors its
// shape function for function, so a reader who knows one knows the other and
// a rule that changes in one is visibly missing from the other. Where the
// two genuinely differ (2D has four quadrants and nine neighbour offsets
// against eight and twenty-seven; 2D's pool levels carry a CACHED tile
// origin where 3D derives it) the comment says so.
//
// Coordinate systems, named once so the rest can be terse:
//
//   COARSE      integer cell of THIS LEVEL'S PARENT grid, [0, NX) x [0, NY).
//               At level 1 the parent is the dense L0 grid; below that it is
//               the parent pool level. Everything in this file is written
//               against "the parent grid", which is what makes one set of
//               functions serve every level -- see poolAtLevel.
//   BLOCK       integer block at this level, [0, NBX) x [0, NBY),
//               NBX = NX / RB.
//   TILE-LOCAL  integer fine cell within a slot, [0, FB)^2. Interior is
//               [GHOST, GHOST + 2*RB); anything else is RING.
//   GLOBAL FINE integer fine cell over the whole level, [0, 2*NX) x [0, 2*NY).
//               The frame in which "which tile owns this cell" is a single
//               division. The independent route the tests check against.
//   L0 UNITS    continuous position in dense-grid cell units -- what the body
//               SDF, the sponge and the interpolation stencil work in. Cell
//               CENTRES sit at integers.

export const GHOST = 2;

// The 2D solver's block size, shaders/amr_step.wgsl's `cellIndex` BLOCK and
// main-amr.js's RB. Named here so a test can vary it and a caller need not
// restate it.
export const RB_DEFAULT = 8;

// Cell-centred refinement: the two fine children of coarse cell `c` sit at
// c - dx/2 and c + dx/2, where dx is the CHILD level's cell size in the
// parent's units (always 1/2) or in L0 units (2^-m). Tile-local fine index j
// -- interior j = GHOST is the first child of the block's first parent cell
// -- maps to `origin - dx/2 + dx*(j - GHOST)`. Ring cells continue the same
// line, which is why this takes a signed index.
//
// This is shaders/amr_step1.wgsl:120 (dx = 0.5, origin an integer block
// offset) and shaders/amr_step1_pool.wgsl:151 (dx = levelParams.dxL, origin
// the cached float) as ONE formula. That they are one formula is the whole
// claim plans/2D-backport.md B3 rests on.
export function fineToCoarseUnit(j, origin, dx = 0.5) {
  return origin - 0.5 * dx + dx * (j - GHOST);
}

// Inverse, for a position known to land on a fine-cell centre.
export function coarseUnitToFine(p, origin, dx = 0.5) {
  return Math.round((p - origin + 0.5 * dx) / dx) + GHOST;
}

// Level m's own cell size in L0 units -- main-amr.js:749's cellSizeL0AtLevel.
// (tau at level m is card-params.mjs's tauAtLevel, already shared and already
// tested there. It is not restated here: one rule, one home.)
export function cellSizeL0AtLevel(m) {
  return 2 ** -m;
}

// --- the pool, and why one pool description serves every level -------------

export function makePool({ dims, rb = RB_DEFAULT, maxSlots }) {
  const [NX, NY] = dims;
  for (const [n, name] of [[NX, 'NX'], [NY, 'NY']]) {
    if (!Number.isInteger(n) || n <= 0) throw new Error(`${name}=${n} is not a positive integer`);
    if (n % rb !== 0) throw new Error(`${name}=${n} is not a multiple of RB=${rb}`);
  }
  const NBX = NX / rb, NBY = NY / rb;
  const FB = 2 * rb + 2 * GHOST;
  return {
    dims: [NX, NY], rb, GHOST, FB,
    nb: [NBX, NBY],
    nBlocks: NBX * NBY,
    tileCells: FB * FB,
    maxSlots: maxSlots ?? NBX * NBY,
    blockId: (bx, by) => by * NBX + bx,
    blockOf: (id) => [id % NBX, Math.floor(id / NBX)],
    // Tile-local fine cell -> index within the pool's f array.
    cellIndex: (slot, fx, fy) => slot * FB * FB + fy * FB + fx,
    isInterior: (j) => j >= GHOST && j < GHOST + 2 * rb,
  };
}

// THE TILE SHAPE IS QUADTREE-UNIFORM, AND THIS IS THAT STATED AS CODE.
//
// A level-m tile's interior is 2*RB cells at its OWN level, which spans
// RB * 2^(1-m) L0 units -- so a level-m tile is exactly one QUADRANT of a
// level-(m-1) tile's footprint at doubled density. Same FB, same ring, same
// kernels; the level-2 tile is the level-1 tile moved down a rung. Level m's
// tiling is therefore just makePool on a domain scaled by 2^(m-1), which is
// why this is a one-liner and why resolveSource / toGlobalFine /
// fromGlobalFine / cellIndex need no change at depth at all.
//
// main-amr.js already builds exactly this (`curNBX *= 2` per level, in the
// pool allocation loop), so this is not a proposal -- it is the existing
// allocation's rule, extracted to where it can be tested.
export function poolAtLevel(pool, m, { maxSlots } = {}) {
  if (m < 1) throw new Error(`level ${m} is not a pool level (level 0 is the dense grid)`);
  const s = 2 ** (m - 1);
  return makePool({
    dims: [pool.dims[0] * s, pool.dims[1] * s],
    rb: pool.rb,
    maxSlots: maxSlots ?? pool.maxSlots,
  });
}

// Level m's block counts per axis. The one function every checker and the
// cascade take as `nbAt`.
export function nbAtLevel(pool, m) {
  const s = 2 ** (m - 1);
  return [pool.nb[0] * s, pool.nb[1] * s];
}

// --- the pool-parent path ---------------------------------------------------

export function parentOfBlock(b) { return [b[0] >> 1, b[1] >> 1]; }
export function quadrantOfBlock(b) { return [b[0] & 1, b[1] & 1]; }

// Where a child's interior starts inside its PARENT's tile-local frame. The
// parent's 2*RB interior cells are halved by the quadrant, so the offset is
// GHOST + q*RB on each axis -- it falls out of the quadrant bits alone, with
// no spatial coordinate involved.
export function quadrantOrigin(rb, q) { return GHOST + q * rb; }

// A TILE'S PHYSICAL ORIGIN IN L0 UNITS, TWO WAYS, AND THEY MUST AGREE.
//
// The GPU builds it RECURSIVELY, at allocation time: a child's origin is its
// parent's origin plus the quadrant offset scaled by the PARENT's cell size
// (shaders/amr_manage_pool.wgsl:403, cached into originX/originY because a
// parent-chain walk is not something a per-dispatch kernel should redo).
// That recursion is exactly what got transposed once and cost a wrong
// level-2 force.
//
// The closed form is one multiply. They are independent routes to the same
// number, so tools/test-amr2d.js drives the recursion and checks it against
// this -- which is the check that would have caught it.
export function tileOriginL0(block, m, rb = RB_DEFAULT) {
  const s = rb * cellSizeL0AtLevel(m - 1);
  return [block[0] * s, block[1] * s];
}

// The recursive route, as the GPU computes it: walk up to level 1 collecting
// quadrants, take level 1's own origin from its block index (level 1 caches
// nothing -- shaders/amr_step1.wgsl derives `bx * RB`), then walk back down
// adding each quadrant offset scaled by that hop's PARENT cell size
// (shaders/amr_manage_pool.wgsl's PARENT_CELL_SIZE_L0).
export function tileOriginL0Recursive(block, m, rb = RB_DEFAULT) {
  const quads = [];
  let b = block.slice();
  for (let lv = m; lv >= 2; lv--) {
    quads.unshift(quadrantOfBlock(b));
    b = parentOfBlock(b);
  }
  let origin = [b[0] * rb, b[1] * rb];
  for (let lv = 2; lv <= m; lv++) {
    const q = quads[lv - 2];
    const parentCellSizeL0 = cellSizeL0AtLevel(lv - 1);
    origin = [
      origin[0] + q[0] * rb * parentCellSizeL0,
      origin[1] + q[1] * rb * parentCellSizeL0,
    ];
  }
  return origin;
}

// --- static refinement sets -------------------------------------------------

// Assigns slots to the blocks a predicate selects, in ascending block order.
// DETERMINISTIC, unlike the live solver's atomicSub free list -- whose
// run-to-run slot assignment is exactly why CLAUDE.md records AMR Cd as
// reproducible only to ~1e-3. A host fixture has no reason to inherit that.
export function refineWhere(pool, predicate) {
  const blockSlot = new Int32Array(pool.nBlocks).fill(-1);
  const slotToBlock = new Int32Array(pool.maxSlots).fill(-1);
  let n = 0;
  for (let id = 0; id < pool.nBlocks; id++) {
    const [bx, by] = pool.blockOf(id);
    const lo = [bx * pool.rb, by * pool.rb];
    const hi = [lo[0] + pool.rb, lo[1] + pool.rb];
    const mid = [lo[0] + pool.rb / 2, lo[1] + pool.rb / 2];
    if (!predicate({ bx, by, id, lo, hi, mid })) continue;
    if (n >= pool.maxSlots) throw new Error(`refinement needs more than maxSlots=${pool.maxSlots} tiles`);
    blockSlot[id] = n;
    slotToBlock[n] = id;
    n++;
  }
  return { blockSlot, slotToBlock, activeSlots: n };
}

// --- the body, as the kernels see it ----------------------------------------
//
// THE HOST STATEMENT OF shaders/common_geometry.wgsl's `get_phi`, because a
// coverage checker is only honest if it evaluates the SAME distance the
// kernel refines on.
//
// main-cylinder-amr.js's own checker did NOT: it used the old ALGEBRAIC form
// `(hypot(lx/a, ly/b) - 1) * b`, which is exact only where a == b. That is
// why it worked there and why it could not simply be copied to the pages that
// needed one. On the falling card's default 8:1 ellipse the algebraic form
// UNDER-reports the true distance by up to a/b = 8x along the major axis, so
// a checker using it would flag a ring of tiles the kernel never considered
// near the body -- eight times the margin's worth of them.
//
// Method and every constant are common_geometry.wgsl's, deliberately: 3
// Newton iterations on the closest-point parameter, seeded from the algebraic
// direction, on |lx|,|ly| with the sign restored from the exact algebraic
// test; the circle closed form; and the far-field early-out at SDF_FAR. See
// that file for why each is what it is. The two are checked against a
// brute-force closest point in tools/test-amr2d.js -- an independent route,
// per this module's first rule.
export const SDF_FAR_DEFAULT = 64;

export function ellipsePhi(px, py, state, { W, H, sdfFar = SDF_FAR_DEFAULT }) {
  const { cx, cy, theta, a, b } = state;
  const ca = Math.cos(theta), sa = Math.sin(theta);
  let dx = px - cx, dy = py - cy;
  // NEAREST IMAGE. The domain is periodic, so the body's closest copy is the
  // one that counts -- and this is what lets a caller pass a raw, unwrapped
  // window coordinate.
  dx -= W * Math.round(dx / W);
  dy -= H * Math.round(dy / H);
  const lx = dx * ca + dy * sa;
  const ly = -dx * sa + dy * ca;

  if (Math.abs(a - b) <= 1e-6 * Math.max(a, b)) return Math.hypot(lx, ly) - a;

  const x = Math.abs(lx), y = Math.abs(ly);
  const r = Math.sqrt((x * x) / (a * a) + (y * y) / (b * b));
  const algebraic = (r - 1) * b;
  if (algebraic > sdfFar) return algebraic;
  const inside = r < 1;

  let t = Math.atan2(y * a, x * b);
  for (let i = 0; i < 3; i++) {
    const ct = Math.cos(t), st = Math.sin(t);
    const ex = x - a * ct, ey = y - b * st;
    const F = ex * (-a * st) + ey * (b * ct);
    const Fp = -(a * a * st * st + b * b * ct * ct) + ex * (-a * ct) + ey * (-b * st);
    t -= F / (Math.abs(Fp) < 1e-9 ? -1e-9 : Fp);
    t = Math.min(Math.max(t, 0), 1.5707963);
  }
  const d = Math.hypot(x - a * Math.cos(t), y - b * Math.sin(t));
  return inside ? -d : d;
}

// The distance the REFINEMENT test actually uses: the smaller of the body's
// distance now and FORCE_REFINE_LOOKAHEAD macro-steps from now.
//
// The "future" pose does NOT extrapolate the body's centre forward. The
// moving window keeps (cx, cy) pinned near the domain centre by construction
// -- bulk translation is absorbed into off_x/off_y -- so it is a fixed buffer
// cell's WINDOW position that moves, at -v. Extrapolate the test point
// backward instead of the ellipse forward; theta is the one quantity the
// window does not absorb, so it still runs forward. This mirrors
// shaders/amr_manage.wgsl's isNearBody, which carries the same argument at
// length.
export function bodyPhiL0(px, py, state, dims, lookahead) {
  const now = ellipsePhi(px, py, state, dims);
  if (!(lookahead > 0)) return now;
  const future = ellipsePhi(
    px - state.vx * lookahead, py - state.vy * lookahead,
    { ...state, theta: state.theta + state.omega * lookahead }, dims);
  return Math.min(now, future);
}

// Buffer (L0) coordinates -> window coordinates. Buffer blocks are fixed in
// memory; the body is anchored in WINDOW space, which is what every kernel's
// `(u32(c) + W - u32(state.off_x)) % W` is doing.
//
// THE EXACT FORM IS WHAT THE KERNELS DO NOW. The modulo is dropped because
// ellipsePhi (like get_phi) already takes the nearest image, so reducing into
// [0, W) first changes nothing.
//
// `bufferToWindowLegacy` is the pre-B4 form, kept because ?boxrefine=0 keeps
// the kernel path it mirrors. It applied `u32()` to the centre as well as to
// off_x, so a level-2 tile centre of 34.25 was tested at 34 -- a
// sub-cell-to-one-cell error compared against a margin of a few cells, on
// exactly the borderline blocks a coverage check is about. A checker that
// silently used the exact centre would have disagreed with the kernel there,
// which is why it was mirrored rather than hidden.
export function bufferToWindow(px, py, state) {
  return [px - state.off_x, py - state.off_y];
}

export function bufferToWindowLegacy(px, py, state) {
  return [Math.trunc(px) - Math.trunc(state.off_x), Math.trunc(py) - Math.trunc(state.off_y)];
}

// DOES ANY POINT OF THIS BLOCK COME WITHIN `margin` OF THE BODY?
//
// THE TWO PREDICATES BELOW ANSWER DIFFERENT QUESTIONS, AND THE DIFFERENCE IS
// A LIVE-VERIFIED BUG THIS REPO ALREADY PAID FOR. `nearBodyWant` asks about
// the BLOCK; `nearBodyWantCentre` asks about its CENTRE, which is what
// shaders/amr_manage.wgsl's isNearBody and amr_manage_pool.wgsl's
// isNearBodyAt actually compute today (one get_phi at the block centre
// against FORCE_REFINE_MARGIN). They are not the same test, and the gap is
// the block's own circumradius -- 5.66 L0 cells for an RB=8 level-1 block.
//
// main-cylinder-amr.js:224-236 records what that cost: at the L1->L2 default
// margin, "a tile whose CENTER just missed that margin could still have an
// edge (or, at shallow surface incidence, a corner) touching the body, so
// debugCheckGeometryCoverage's center-only test could pass while a handful of
// body-adjacent L1 tiles permanently lacked their required L2 child", with
// the symptom "L1's own force pass sat at a bit-identical fx~-0.19 for
// 20,000+ steps". It was fixed by ENLARGING THE MARGIN (paramsForChildLevel's
// childLevel===2 special case), not by fixing the test -- so the gap is still
// there, just papered over at one level.
//
// BOTH ARE HERE ON PURPOSE. The checker is only honest if it asks the same
// question the kernel answers, so wiring this module to the live GPU today
// uses the CENTRE predicate and reproduces the existing checker exactly;
// plans/2D-backport.md B4 is where the kernel moves to the box predicate and
// the checker follows. Keeping both means that transition is a one-line
// change with a test that already knows they differ, rather than a silent
// re-baselining of what "covered" means.
//
// `sdf(x, y)` is in L0 units and is the caller's whole body model -- if the
// kernel's test takes the min over the current and extrapolated poses
// (FORCE_REFINE_LOOKAHEAD, and it does), the caller folds that in here rather
// than this module knowing what a card is.

// sqrt(2), ROUNDED UP. A square of half-edge h has circumradius h*sqrt(2),
// and that radius BOUNDS how far the SDF can fall inside the box -- rounding
// it down would make the bound false by a hair. One literal, so a host and a
// kernel cannot disagree about a borderline block.
//
// BOTH CONSTANTS ARE EXPORTED because shaders/common_geometry.wgsl's
// nearBodyBox now types the same two values, and tools/test-amr2d.js parses
// them back out of that file and compares. A generator would be better still
// (lattice-2d.mjs's shape); two numbers did not earn one, but they did earn
// the check that fails when the two drift.
export const SQRT2_UP = 1.4142136;

// The depth of the branch-and-bound. 3 leaves a residual slack of
// RB*sqrt(2)/16 = 0.71 L0 cells at RB=8 -- conservative, so it can
// over-refine by that much and never miss.
export const BLOCK_BB_DEPTH = 3;

// A signed distance is 1-Lipschitz, so ONE evaluation at a box centre
// brackets the minimum over the whole box as phi(c) - R <= min <= phi(c).
// That decides most boxes outright and only the ambiguous shell subdivides.
// Conservative at every depth: it can over-refine, never miss -- which is the
// right direction for a HARD geometric constraint.
export function nearBodyWant(sdf, margin) {
  const near = (c, h, depth) => {
    const R = h * SQRT2_UP;
    const phi = sdf(c[0], c[1]);
    if (phi - R > margin) return false;   // provably clear
    if (phi <= margin) return true;       // the centre itself is inside
    // The deepest level is the only place the answer is a BOUND rather than a
    // decision: having failed to reject, accept. That is where the residual
    // slack lives.
    if (depth === 0) return true;
    const hh = h / 2;
    for (let i = 0; i < 4; i++) {
      const cc = [c[0] + (i & 1 ? hh : -hh), c[1] + (i & 2 ? hh : -hh)];
      if (near(cc, hh, depth - 1)) return true;
    }
    return false;
  };
  return ({ lo, hi }) => near(
    [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2],
    (hi[0] - lo[0]) / 2, BLOCK_BB_DEPTH);
}

// What the kernels compute TODAY: one SDF evaluation at the block centre.
// Kept so a host checker can ask the kernel's own question until B4 changes
// it, and so the test can assert the two genuinely disagree.
export function nearBodyWantCentre(sdf, margin) {
  return ({ mid }) => sdf(mid[0], mid[1]) < margin;
}

export function refineNearBody(pool, sdf, margin) {
  return refineWhere(pool, nearBodyWant(sdf, margin));
}

// --- neighbour-addressed streaming (DIRECT_GHOST) ---------------------------

// Where a gather that leaves a tile's interior actually reads.
//
// shaders/amr_step1.wgsl's DIRECT_GHOST path resolves a source cell that
// falls outside this tile's interior against the OWNING same-level tile
// directly, instead of reading a materialized ghost value. This is the host
// statement of that mapping; tools/test-amr2d.js checks it against the
// global-fine route, which shares no arithmetic with it.
//
// Returns { slot, own, fx, fy } in the owning tile's local frame -- `own`
// when the source never left this tile -- or null when no tile owns it,
// which is a coarse/fine interface and the ring's one remaining job.
export function resolveSource(pool, blockSlot, blockXY, src) {
  const { rb, nb } = pool;
  const RB2 = 2 * rb;
  const out = [0, 0];
  const nbr = [0, 0];
  for (let a = 0; a < 2; a++) {
    const s = src[a];
    if (s < GHOST) { nbr[a] = -1; out[a] = s + RB2; }
    else if (s >= GHOST + RB2) { nbr[a] = 1; out[a] = s - RB2; }
    else { nbr[a] = 0; out[a] = s; }
  }
  if (nbr[0] === 0 && nbr[1] === 0) {
    return { slot: null, own: true, fx: out[0], fy: out[1] };
  }
  // The block grid is periodic, matching the kernels.
  const b = [0, 1].map(a => (blockXY[a] + nbr[a] + nb[a]) % nb[a]);
  const slot = blockSlot[pool.blockId(b[0], b[1])];
  if (slot < 0) return null;
  return { slot, own: false, fx: out[0], fy: out[1] };
}

// The independent route: tile-local -> GLOBAL fine coordinate, where
// ownership is one division and there is no ring at all. Used by the test to
// check resolveSource, and by nothing else -- deliberately, so the two
// cannot drift into agreement.
export function toGlobalFine(pool, blockXY, local) {
  const RB2 = 2 * pool.rb;
  return [0, 1].map(a => blockXY[a] * RB2 + (local[a] - GHOST));
}

export function fromGlobalFine(pool, g) {
  const RB2 = 2 * pool.rb;
  const nFine = [0, 1].map(a => pool.dims[a] * 2);
  const w = [0, 1].map(a => ((g[a] % nFine[a]) + nFine[a]) % nFine[a]);
  const b = [0, 1].map(a => Math.floor(w[a] / RB2));
  const l = [0, 1].map(a => w[a] - b[a] * RB2 + GHOST);
  return { block: b, local: l };
}

// Cells a refined region stores against the coarse cells it covers.
export function storageRatio(pool) {
  return (pool.FB / pool.rb) ** 2;
}

// --- structural invariants --------------------------------------------------
//
// PURE functions over block sets, so tools/test-amr2d.js can exercise them
// with no server, browser or GPU -- INCLUDING on inputs that violate the
// invariant, which is the only way to know a checker checks anything. A
// checker only ever run on valid input is indistinguishable from one that
// returns nothing.
//
// `levelSets[m]` is a Set of "bx,by" keys naming the blocks active at level
// m (m >= 1); level 0 is the dense grid and is implicit, present everywhere.
// `nbAt(m)` gives that level's block counts per axis, which double each
// level.

// The 2 (of 4) quadrant children covering one EDGE of their parent: the edge
// on axis `a` in direction `s` is covered by the children whose quadrant bit
// on `a` is (s > 0 ? 1 : 0). Recursing with the SAME edge picks the correct
// ever-deeper sliver along it, not the tile's maximum depth anywhere -- which
// is the distinction main-amr.js:2946-2960 records having got wrong once, and
// having produced false-positive violations with.
function edgeChildren(a, s) {
  const want = s > 0 ? 1 : 0;
  const out = [];
  for (let k = 0; k < 4; k++) {
    const q = [k & 1, (k >> 1) & 1];
    if (q[a] === want) out.push(q);
  }
  return out;
}

const EDGES = [[0, -1], [0, 1], [1, -1], [1, 1]];

// All 9 block offsets INCLUDING (0,0). The RING needs these, not just the
// four edges -- see checkRingParentCoverage and cascade21.
const NBR9 = (() => {
  const out = [];
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) out.push([dx, dy]);
  return out;
})();

// 2:1 BALANCE. Every LEAF tile's edge neighbour must sit within one level of
// it. Corner (diagonal) neighbours are checked too but reported SEPARATELY
// and NOT folded into `ok` -- corner balance is a requirement of the
// ghost-free path, not of the default one, so failing every default run on it
// would be wrong. main-amr.js:3061-3090 made that call and it stands; callers
// that need it assert `cornerViolations` themselves (see
// tools/lib/amr-invariants.js's requireCornerBalance).
export function check21Balance(levelSets, nbAt, { levels }) {
  const has = (m, x, y) => m >= 1 && m < levels && levelSets[m] && levelSets[m].has(`${x},${y}`);

  // QUAD-COMPLETENESS IS LEANED ON HERE. A parent spawns all four children or
  // none (shaders/amr_manage_pool.wgsl: "a parent slot may only spawn a
  // level-(m+1) quad"), so testing quadrant (0,0) alone decides whether a tile
  // is a leaf. A set that violates quad-completeness reads as a leaf while
  // holding children, and this checker then reports the resulting violation --
  // correctly. cascade21 completes quads for exactly that reason; 3D's first
  // cascade did not, and the checker caught it.
  const hasChild = (m, x, y) => has(m + 1, x * 2, y * 2);

  // The deepest level actually present at (x,y), walking UP from m.
  const ancestorDepth = (m, x, y) => {
    let lv = m, a = x, b = y;
    while (lv >= 1) {
      if (has(lv, a, b)) return lv;
      a = Math.floor(a / 2); b = Math.floor(b / 2);
      lv--;
    }
    return 0;
  };
  // The deepest level present along one EDGE of a tile, walking DOWN.
  const edgeMaxDepth = (m, x, y, a, s) => {
    if (!hasChild(m, x, y)) return m;
    let d = m;
    for (const q of edgeChildren(a, s)) {
      d = Math.max(d, edgeMaxDepth(m + 1, x * 2 + q[0], y * 2 + q[1], a, s));
    }
    return d;
  };

  const violations = [];
  const cornerViolations = [];
  for (let m = 1; m < levels; m++) {
    if (!levelSets[m]) continue;
    const nb = nbAt(m);
    for (const key of levelSets[m]) {
      const [bx, by] = key.split(',').map(Number);
      if (hasChild(m, bx, by)) continue;   // not a leaf; checked one level deeper
      for (const [a, s] of EDGES) {
        const n = [bx, by];
        n[a] = ((n[a] + s) % nb[a] + nb[a]) % nb[a];   // the block grid is periodic
        const nDepth = has(m, n[0], n[1])
          ? edgeMaxDepth(m, n[0], n[1], a, -s)          // the neighbour's facing side
          : ancestorDepth(m, n[0], n[1]);
        if (Math.abs(m - nDepth) > 1) {
          violations.push({ level: m, block: [bx, by], neighbour: n.slice(), nDepth, axis: a, dir: s });
        }
      }
      // Corners. No edgeMaxDepth analogue: a diagonal neighbour touches at a
      // single point, so "deepest tile along a shared edge" is not the right
      // quantity. Ancestor depth is, and it is what the ghost-free stencil
      // actually asks -- does a tile exist at my level there.
      for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        const nx = ((bx + dx) % nb[0] + nb[0]) % nb[0];
        const ny = ((by + dy) % nb[1] + nb[1]) % nb[1];
        const nDepth = has(m, nx, ny) ? m : ancestorDepth(m, nx, ny);
        if (Math.abs(m - nDepth) > 1) {
          cornerViolations.push({ level: m, block: [bx, by], neighbour: [nx, ny], nDepth, corner: [dx, dy] });
        }
      }
    }
  }
  return {
    ok: violations.length === 0,
    cornerOk: cornerViolations.length === 0,
    violations, cornerViolations, levels,
    counts: levelSets.map(s => (s ? s.size : 0)),
  };
}

// RING PARENT COVERAGE -- an invariant 2:1 balance does NOT imply, and one
// that does not exist at all until there is a pool parent (level >= 2).
//
// THE REQUIREMENT. A tile's ring extends one PARENT cell beyond its block on
// every one of the 8 sides, and the interp pass fills a ring cell from the
// parent cell containing it -- so that parent cell has to live in an
// ALLOCATED parent tile. At level 1 this is free and invisible: the parent is
// the dense L0 grid, which exists everywhere. At level >= 2 the parent is a
// pool, and a tile that is not there cannot be read from.
//
// WHY THE EDGES ARE NOT ENOUGH, concretely. Take child block b = (2p, 2p),
// the lower quadrant on both axes. Its diagonal neighbour (2p-1, 2p-1) has
// parent (p-1, p-1). The edge closure supplies parent(2p-1, 2p) = (p-1, p)
// and parent(2p, 2p-1) = (p, p-1) and NOT (p-1, p-1). So the corner ring
// cell's parent tile is absent.
//
// AND THAT CELL IS READ. shaders/amr_step1.wgsl resolves a source against the
// diagonal same-level neighbour tile (`nbrXY`) and falls back to this tile's
// own ring when that tile is absent -- exactly the configuration where the
// corner ring cell matters.
//
// This is NOT the same thing as check21Balance's corner list. That one is
// about SAME-LEVEL corner neighbours and is a ghost-free requirement; this is
// about the PARENT level and is what keeping the ring costs at depth.
export function checkRingParentCoverage(levelSets, nbAt, { levels }) {
  const violations = [];
  let required = 0;
  const has = (m, x, y) => m >= 1 && m < levels && levelSets[m] && levelSets[m].has(`${x},${y}`);
  for (let m = 2; m < levels; m++) {
    if (!levelSets[m]) continue;
    const nb = nbAt(m);
    for (const key of levelSets[m]) {
      const b = key.split(',').map(Number);
      for (const d of NBR9) {
        const n = [0, 1].map(a => ((b[a] + d[a]) % nb[a] + nb[a]) % nb[a]);
        const p = parentOfBlock(n);
        required++;
        if (!has(m - 1, p[0], p[1])) {
          violations.push({ level: m, block: b.slice(), offset: d.slice(), parent: p, parentLevel: m - 1 });
        }
      }
    }
  }
  return { ok: violations.length === 0, violations, required };
}

// GEOMETRY COVERAGE: every block within the margin of the body must be
// refined to the FINEST configured level. plans/2D-backport.md B4 -- the body
// lives entirely on the finest level, and that is a hard requirement rather
// than a policy, because a coarse/fine seam inside the penalization band is a
// bug class this project has already hit more than once.
//
// TAKES THE PREDICATE, does not build it. Which question is being asked --
// the block's or its centre's -- is the caller's to state, because until B4
// the kernel and the requirement differ; see nearBodyWant's header. Build it
// with nearBodyWant(sdf, margin) or nearBodyWantCentre(sdf, margin), both of
// which work in L0 units, as the box this function hands them does.
export function checkGeometryCoverage(pool, levelSets, want, { levels }) {
  const finest = levels - 1;
  if (finest < 1) return { ok: true, violations: [], required: 0 };
  const fine = poolAtLevel(pool, finest);
  const violations = [];
  let required = 0;
  // A finest-level block spans `rb` cells of its PARENT grid, and a parent
  // cell is cellSizeL0AtLevel(finest - 1) L0 units across.
  const cell = cellSizeL0AtLevel(finest - 1);
  for (let id = 0; id < fine.nBlocks; id++) {
    const [bx, by] = fine.blockOf(id);
    const lo = [bx * fine.rb * cell, by * fine.rb * cell];
    const hi = [lo[0] + fine.rb * cell, lo[1] + fine.rb * cell];
    const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
    if (!want({ lo, hi, mid })) continue;
    required++;
    if (!(levelSets[finest] && levelSets[finest].has(`${bx},${by}`))) {
      violations.push({ level: finest, block: [bx, by], lo, hi });
    }
  }
  return { ok: violations.length === 0, violations, required };
}

// --- the 2:1 forcing rule, as ONE closure -----------------------------------
//
// THE RULE:
//
//     present(m, b)  =>  present(m-1, parent(n))   for each neighbour n of b
//                        at level m, and for b itself
//
// Read forwards that is "refine forced by a neighbour that wants a deeper
// child"; read BACKWARDS -- a block the criterion did not want, added back --
// it is "coarsen blocked by a neighbour that has one". THEY ARE NOT TWO
// MECHANISMS, which is why this is one function applied to the WANT set
// between the criterion and the allocator, rather than tests bolted onto
// coarsen and refine.
//
// WHY THAT MATTERS HERE SPECIFICALLY. shaders/amr_manage_pool.wgsl's header
// documents THREE separately live-verified balance bugs produced by the
// per-pass shape: a neighbour-active gate vetoing geometry-forced refinement,
// a grandchild (level m+2) blind spot that produced real
// depth-1-next-to-depth-3 violations, and a criterion-based cascade test that
// had to become an existence-based one because criterion flickers. And a
// FOURTH is still shipped: only the VETO half of the refine cascade was ever
// implemented (main-amr.js:112-131), so a criterion-driven refine can be
// vetoed forever by a neighbour that would only ever have been created BY
// that refine -- measured live, level 2 never extends past the geometry halo
// into the wake, which pins the L1/L2 boundary a few cells off the body so
// every shed vortex crosses it right there.
//
// The closure has none of those failure modes by construction: it is
// transitive (so there is no grandchild special case), it runs on WANTS
// rather than on what exists (so there is no flicker), and the growth half IS
// the veto half.
//
// ALL 9 OFFSETS, NOT THE FOUR EDGES. d = (0,0) is the tree property (a block
// with no parent is not a refinement of anything), the edges are 2:1 balance,
// and the four DIAGONALS are the RING -- see checkRingParentCoverage for the
// concrete cell that goes unreadable without them.
export function cascade21(wantSets, nbAt, { levels }) {
  const sets = [null];
  for (let m = 1; m < levels; m++) sets[m] = new Set(wantSets[m] || []);
  const forced = [];

  // REFINEMENT IS QUAD-COMPLETE FROM LEVEL 2 DOWN. A level-m block exists
  // because its level-(m-1) parent SPAWNED IT along with its three siblings
  // -- a tile is allocated per block and there is no such thing as a
  // quarter-refined parent. check21Balance's hasChild leans on it too, so a
  // cascade that added single blocks produces a tree the checker then reports
  // as unbalanced. Completing the quad says what the want MEANS rather than
  // rejecting it.
  //
  // Level 1 is the exception and not an inconsistency: its parent is the
  // dense L0 grid, present everywhere, so level-1 blocks are refined
  // individually -- which is exactly what main-amr.js's level-1 allocator
  // already does (a per-block free list, not a quad-indexed one).
  const ensure = (m, b, because) => {
    const parent = parentOfBlock(b);
    const n = m >= 2 ? 4 : 1;
    for (let k = 0; k < n; k++) {
      const c = m >= 2 ? [parent[0] * 2 + (k & 1), parent[1] * 2 + ((k >> 1) & 1)] : b;
      const key = `${c[0]},${c[1]}`;
      if (sets[m].has(key)) continue;
      sets[m].add(key);
      forced.push({ level: m, block: c, quad: m >= 2 ? parent : null, because });
    }
  };

  for (let m = levels - 1; m >= 2; m--) {
    // COMPLETE THIS LEVEL'S QUADS FIRST. The criterion is evaluated per block
    // and nothing in it looks sideways, so a want for one child with no
    // siblings is a state the manager can genuinely produce -- and it is not
    // a state the POOL can be in. Quads added at this level by the level
    // below have already come through `ensure`, so this only ever finds the
    // caller's own input.
    for (const key of [...sets[m]]) {
      ensure(m, key.split(',').map(Number), { siblingOf: key });
    }
    const nb = nbAt(m);
    for (const key of sets[m]) {
      const b = key.split(',').map(Number);
      for (const d of NBR9) {
        const n = [0, 1].map(a => ((b[a] + d[a]) % nb[a] + nb[a]) % nb[a]);
        ensure(m - 1, parentOfBlock(n),
          d[0] === 0 && d[1] === 0
            ? { parentOf: b.slice(), level: m }
            : { neighbourOf: b.slice(), level: m, offset: d.slice() });
      }
    }
  }
  return { sets, forced, counts: sets.map((s, m) => (m === 0 ? null : s.size)) };
}
