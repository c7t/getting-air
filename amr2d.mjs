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

import { tauAtLevel } from './card-params.mjs';

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
// This WAS shaders/amr_step1.wgsl and shaders/amr_step1_pool.wgsl stating the
// same formula twice, one with dx = 0.5 and an integer block offset, the other
// with dx = levelParams.dxL and a cached float origin. That they are one
// formula is the claim plans/2D-backport.md B3 rested on, and since B3-1 there
// is one kernel -- shaders/amr_step1.wgsl, every level.
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

// --- the Dupuis-Chopard non-equilibrium rescale ----------------------------
//
// THE HOST STATEMENT OF THE COARSE<->FINE TRANSFER FACTOR, and the reason it
// is not the textbook one (plans/2D-backport.md B1).
//
// Dupuis-Chopard is derived for the PRE-collision non-equilibrium part. With
// refinement ratio n=2, fneq scales as tau * (velocity gradient per lattice
// cell), and the same physical shear spans twice as many fine cells:
//
//   fneq_pre,fine = (1/n) * (tau_f / tau_c) * fneq_pre,coarse          (PRE)
//
// 2D's step kernels are FUSED pull-stream + collide (shaders/amr_step.wgsl's
// own title), so every buffer this solver transfers holds f AFTER collision.
// BGK post-collision is f* = f - (f - feq)/tau, hence
//
//   fneq* = ((tau - 1) / tau) * fneq_pre
//
// at whichever level it is evaluated. Composing decollide -> PRE -> recollide
// cancels both tau's in the middle and leaves
//
//   fneq*_fine = (1/2) * (tau_f - 1) / (tau_c - 1) * fneq*_coarse     (POST)
//
// and its EXACT inverse going the other way. These are not a small
// correction to each other: at tau_c = 0.8 (the channel/TGV default, so
// tau_f = 1.1) PRE gives +0.6875 and POST gives -0.25 -- different magnitude
// AND different sign. Near tau = 0.5, where the card and cylinder pages
// live, they agree to a few percent, which is why the wrong factor shipped
// for as long as it did: the pages carrying a Cd/St number could barely see
// it, and the pages with an analytic answer were not being read.
//
// This was the large half of the 3D fork's M3 seam error.
export function dcRescaleCoarseToFine(tauCoarse, tauFine) {
  return 0.5 * (tauFine - 1) / (tauCoarse - 1);
}

export function dcRescaleFineToCoarse(tauCoarse, tauFine) {
  return 2 * (tauCoarse - 1) / (tauFine - 1);
}

// The legacy PRE-collision pair, kept as a live path (`?dcpre=1`) rather than
// as a deleted branch in the history, so the defect can be RE-MEASURED in one
// build instead of reconstructed from a checkout. Same reason ?ghostcopy=1
// and ?f16=0 are still here.
export function dcRescaleCoarseToFinePre(tauCoarse, tauFine) {
  return 0.5 * tauFine / tauCoarse;
}

export function dcRescaleFineToCoarsePre(tauCoarse, tauFine) {
  return 2 * tauCoarse / tauFine;
}

// TAU = 1 IS A REAL SINGULARITY, NOT A FORMULA ARTIFACT, and it has to be
// refused rather than divided by.
//
// At omega = 1 the post-collision populations ARE the equilibrium: fneq* is
// identically zero at that level, so it carries no stress for a
// post-collision transfer to move. The closed forms above say the same thing
// as 0/0 -- (tau_c - 1) is the coarse->fine denominator and (tau_f - 1) the
// fine->coarse one, so a level sitting at tau = 1 poisons the transfer on
// whichever side it appears.
//
// It is REACHABLE, which is why this is a guard and not a footnote. Level m's
// tau is 2^m (tau_0 - 1/2) + 1/2, so tau_m = 1 exactly when
// tau_0 = 1/2 + 2^-(m+1):
//
//   tau_0 = 1.0     singular at level 0   (a classic BGK choice)
//   tau_0 = 0.75    singular at level 1   (?tau=0.75 on channel/TGV)
//   tau_0 = 0.625   singular at level 2
//   tau_0 = 0.5625  singular at level 3   (inside index-amr.html's OWN
//                                          tau slider range, 0.5005..0.6)
//
// Every level in [0, levels) is a denominator somewhere -- level 0 only as a
// coarse side, the finest only as a fine side, the rest as both -- so the
// whole chain is checked, not its interior.
//
// BAND, and what it costs. Exact equality is the singularity; the
// neighbourhood is merely ill-conditioned, and because the two directions are
// exact inverses one blows up precisely as far as the other collapses. At
// |tau - 1| = TAU_UNITY_BAND one direction still rescales fneq by ~50x. That
// is deliberately left to the user: it is finite, the sign is right, and
// refusing it would refuse configurations that are only unwise. What is
// refused is the part where the answer is not a number.
export const TAU_UNITY_BAND = 0.01;

export function tauChainSingularity(tau0, levels, band = TAU_UNITY_BAND) {
  for (let m = 0; m < levels; m++) {
    const tau = tauAtLevel(tau0, m);
    if (Math.abs(tau - 1) < band) return { level: m, tau, tau0, band };
  }
  return null;
}

// The refusal's own text, here rather than in five pages, so what the user is
// told about a singular chain cannot drift between them -- and so a test can
// assert it names the level it found (a refusal that says only "bad tau" sends
// the reader to the wrong knob on a four-level hierarchy).
export function tauSingularityMessage({ level, tau, tau0, band }) {
  const nearest = 0.5 + 2 ** -(level + 1);
  return `tau = ${tau.toFixed(6)} at level ${level} is within ${band} of 1, where the ` +
    `post-collision coarse<->fine transfer is 0/0 -- at omega = 1 the post-collision ` +
    `populations ARE the equilibrium and carry no stress to transfer. This is a real ` +
    `singularity, not a formula artifact, so it is refused rather than divided by. ` +
    `L0 tau is ${tau0}, and level ${level} is exactly singular at L0 tau = ${nearest} ` +
    `(level m's tau is 2^m (tau_0 - 1/2) + 1/2). Fix: move tau (or Re, or ?levels=) ` +
    `off it. ?dcpre=1 restores the legacy pre-collision factor, which has no ` +
    `singularity here -- but it is the defect plans/2D-backport.md B1 removed, so use ` +
    `it to measure, not to run.`;
}

// --- the pool, and why one pool description serves every level -------------

// `ghost` is the tile's ring depth, and it is a PARAMETER because the ROOT has
// none (U0's ghostDepthAtLevel). The root is exactly this function at ghost 0
// on a half-scaled domain: NBX = W/(2*rb) and a tile whose FB IS its own 2*rb
// interior. Default GHOST keeps every existing caller byte-identical.
export function makePool({ dims, rb = RB_DEFAULT, maxSlots, ghost = GHOST }) {
  const [NX, NY] = dims;
  for (const [n, name] of [[NX, 'NX'], [NY, 'NY']]) {
    if (!Number.isInteger(n) || n <= 0) throw new Error(`${name}=${n} is not a positive integer`);
    if (n % rb !== 0) throw new Error(`${name}=${n} is not a multiple of RB=${rb}`);
  }
  const NBX = NX / rb, NBY = NY / rb;
  const FB = 2 * rb + 2 * ghost;
  return {
    dims: [NX, NY], rb, GHOST: ghost, FB,
    nb: [NBX, NBY],
    nBlocks: NBX * NBY,
    tileCells: FB * FB,
    maxSlots: maxSlots ?? NBX * NBY,
    blockId: (bx, by) => by * NBX + bx,
    blockOf: (id) => [id % NBX, Math.floor(id / NBX)],
    // Tile-local fine cell -> index within the pool's f array.
    cellIndex: (slot, fx, fy) => slot * FB * FB + fy * FB + fx,
    isInterior: (j) => j >= ghost && j < ghost + 2 * rb,
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
    ghost: pool.GHOST,
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

// A SLOT'S QUADRANT IS A FUNCTION OF THE SLOT INDEX, not stored data.
//
// Levels >= 2 allocate in QUADS: a free-list entry is a quad index, and the
// four children go to `quadIdx*4 + quadrant` for quadrant in 0..3. So
// `quadrant == slot % 4` by construction, on the GPU allocator
// (shaders/amr_manage_pool.wgsl's refine) and on the host one
// (main-amr.js's debugActivateBlock) alike -- both compose the slot that way.
//
// The pool nonetheless keeps a per-slot `quadrantBuf` holding exactly this,
// rewritten on every allocation. That is a buffer storing a constant, and
// shaders/amr_manage_pool.wgsl sits at EXACTLY the 16-storage-buffer
// per-stage limit (see CLAUDE.md), so it is also the cheapest binding in the
// project to get back. amr2d-gpu.mjs's checkSlotQuadrantsOnGPU scores the
// live buffer against this rule.
export function quadrantOfSlot(slot) { return slot % 4; }

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


// L0 buffer coordinates -> THE BODY'S frame. Since plans/2D-backport.md B5 the
// body lives in BUFFER coordinates, so this is the identity -- but it is a
// named identity on purpose. The checker that calls it is scoring the kernels,
// and the day the host and the kernels disagree about which frame the body is
// in is the day the coverage gate silently stops measuring anything. That is
// exactly B5-5's bug one layer down: lbm_force.wgsl went on building a WINDOW
// position after state.cx had become a buffer one, and because the two
// coincide whenever off == 0 -- which is every pinned-cylinder config -- no
// gate could see it.
//
// `bodyFrameL0Legacy` mirrors ?boxrefine=0's kernel path, which reaches the
// body through vec2<u32>(...) and therefore TRUNCATES a fractional tile centre
// by up to a cell, compared against a margin of a few, on exactly the
// borderline blocks a coverage check is about. common_refine.wgsl's
// BOX_REFINE == 0u branch carries the same truncation deliberately.
export function bodyFrameL0(px, py, _state) {
  return [px, py];
}

export function bodyFrameL0Legacy(px, py, _state) {
  return [Math.trunc(px), Math.trunc(py)];
}

// ── POOL CAPACITY, FROM MEASURED DEMAND ──────────────────────────────────────
//
// A pool at its cap does not degrade gracefully: slots are granted in blockID
// order, so the free list dries up part-way through a row and the denied
// blocks form horizontal BANDS across the refined region. Worse, when the
// denied blocks are GEOMETRY-forced the body ends up split across levels and
// only the finest computes force, so the run's force is simply wrong -- which
// is what makeRefusalWatch exists to catch.
//
// So capacity is measured, not guessed: run with every cap lifted, record the
// max live tile count per level, and give it headroom. Both pages had a FLAT
// default for every level below the first (512 on the card page, 128 on the
// cylinder), and demand is nowhere near flat -- it roughly doubles with each
// level and jumps again when a level acquires a child.
//
// TWO REGIMES, because a level's job changes when it gains a CHILD. As the
// finest level it hosts its own geometry halo and wake. Once a finer level
// exists beneath it, 2:1 closure additionally forces a parent tile for
// everything refined below, and the demand steps up -- measured, not
// reasoned: the card page's level 2 peaks at 400 as the finest level and
// 492-516 once level 3 exists; the cylinder's goes 96 -> 160, and that body
// is PINNED, so its numbers carry almost no run-to-run scatter.
//
// `peaks` is { finest: {m: peak}, parent: {m: peak} }, per page, in that
// page's own units -- a tumbling card and a pinned cylinder do not have
// comparable demand and must not share a table.
export const POOL_HEADROOM = 1.7;

// Unmeasured levels: extrapolate by this per level and SAY SO. The measured
// ladder grows 261 -> 516 -> 656 -> 904 (ratios 1.98, 1.27, 1.38), so this is
// an upper-ish estimate rather than a fit. It is a starting point for a level
// nobody has run; the refusal watch is still the thing that says it was wrong.
export const POOL_GROWTH = 1.7;

// A POOL'S CAPACITY MUST BE A MULTIPLE OF 4. Slots are allocated a QUAD at a
// time -- one per child quadrant -- so allocLevelPool refuses anything else,
// loudly, at init. Applying the headroom multiplier without rounding produced
// 878 and the page would not boot; that refusal is the reason this is a
// rounding rule here rather than a comment somewhere. Round UP: rounding down
// could put the pool below the measured peak it was derived from, which is the
// exact failure this table exists to prevent.
const quad = (n) => Math.ceil(n / 4) * 4;

export function poolSlotsFor(peaks, m, nLevels) {
  const isFinest = (m === nLevels - 1);
  const table = isFinest ? peaks.finest : peaks.parent;
  const direct = table[m];
  if (direct !== undefined) return quad(direct * POOL_HEADROOM);
  // Fall back to the other regime at the same level before extrapolating --
  // a measured number from the wrong regime beats a guess at the right one.
  const other = (isFinest ? peaks.parent : peaks.finest)[m];
  if (other !== undefined) return quad(other * POOL_HEADROOM);
  // Beyond anything measured: extrapolate from the deepest level that was.
  const known = Object.keys(peaks.finest).concat(Object.keys(peaks.parent)).map(Number);
  const deepest = Math.max(...known);
  const base = Math.max(peaks.finest[deepest] ?? 0, peaks.parent[deepest] ?? 0);
  return quad(base * POOL_HEADROOM * Math.pow(POOL_GROWTH, m - deepest));
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
//
// IT READS THE POOL'S OWN RING DEPTH, not the module constant, because the
// ROOT has no ring (U0's ghostDepthAtLevel) and U4 needs this rule there. At
// the root the `null` return is unreachable by construction -- the root is
// always full, so every source has an owner -- which is what lets a ring-free
// level use a stencil that would otherwise need one.
export function resolveSource(pool, blockSlot, blockXY, src) {
  const { rb, nb } = pool;
  const RB2 = 2 * rb;
  const g = pool.GHOST;   // the POOL's ring depth -- 0 at the root (U4-1)
  const out = [0, 0];
  const nbr = [0, 0];
  for (let a = 0; a < 2; a++) {
    const s = src[a];
    if (s < g) { nbr[a] = -1; out[a] = s + RB2; }
    else if (s >= g + RB2) { nbr[a] = 1; out[a] = s - RB2; }
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
  return [0, 1].map(a => blockXY[a] * RB2 + (local[a] - pool.GHOST));
}

export function fromGlobalFine(pool, g) {
  const RB2 = 2 * pool.rb;
  const nFine = [0, 1].map(a => pool.dims[a] * 2);
  const w = [0, 1].map(a => ((g[a] % nFine[a]) + nFine[a]) % nFine[a]);
  const b = [0, 1].map(a => Math.floor(w[a] / RB2));
  const l = [0, 1].map(a => w[a] - b[a] * RB2 + pool.GHOST);
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
// concrete cell that goes unreadable without them.//
// AND d = (0,0) IS REDUNDANT -- MEASURED, NOT ASSUMED. The three-jobs reading
// above ("tree property, faces, ring") is how this rule is usually written
// down, and it overstates the case: the own-parent offset is IMPLIED by the
// four faces. A quad's members are adjacent and share a parent, so for any
// block at least one x-face neighbour has the same parent -- bx even gives
// (bx+1)>>1 == bx>>1, bx odd gives (bx-1)>>1 == bx>>1, and NBX is a power of
// two so the wrap has no edge case. Dropping (0,0) therefore changes nothing,
// and it is the one mutant of this closure that breaks NOTHING: 0 of 42 host
// checks and 0 of 10 GPU seeds, against 5-9 failures for every other mutant
// tried (face-only, no quad completion, non-periodic wrap, one-axis parent
// shift, transposed parent, quad completion rounding up).
//
// KEEP IT ANYWAY. It costs one ninth of a cheap pass, it makes the rule
// self-evident rather than a consequence, and the redundancy depends on the
// face offsets being present -- so a future change that narrowed the offset
// set would silently take the tree property with it. What is NOT ok is
// deleting it as an optimization on the strength of a green suite, which is
// exactly what a green suite would support here.
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

// ── D0: the deterministic slot handout (plans/uniform-levels.md D0) ─────────
//
// THE DEFECT THESE REPLACE. Both GPU managers hand slots out through a racing
// free list -- `atomicSub(&freeCount, 1)` in shaders/amr_manage.wgsl and
// shaders/amr_manage_pool.wgsl -- so which block gets which slot depends on
// which thread reaches the atomic first. amr_force1.wgsl then atomicAdds one
// TRUNCATED i32 per workgroup, so regrouping the slots regroups the partials
// and they truncate differently. Measured 2026-09-17
// (tools/measure-determinism.js): that is the ONLY live source of run-to-run
// nondeterminism in this solver. With the handout pinned, two runs of one
// build are bit-identical; without it, `?levels=2` differs on every run.
//
// THE RULE. Order the candidates, then serve them in order. Nothing else
// changes -- the pop discipline is still "take the top of the stack, then
// decrement", which is what makes the deterministic and racing paths differ
// only in WHO is served WHEN and not in what a service does.
//
// These return a FREE-LIST INDEX, not a slot or a quad. The free list's
// CONTENTS are state that both paths share; what the rule fixes is which
// index each candidate reads. Keeping the two apart is what lets the GPU be
// scored against this without the test having to model the pool's history.
//
// ORDERING BY ID IS THE STATEMENT; the GPU may reach it by a cheaper route.
// The dense manager dispatches over blocks, so its serial loop is already in
// id order. The pool manager dispatches over parent SLOTS and serves in slot
// order, which is a different permutation -- and legitimately so: a slot
// assignment is part of the state (debugSnapshotLoad restores it), so slot
// order is state-determined too. What it is NOT is id-ordered, so do not
// score the pool manager against grantAssignment's `id` sequence; score it
// against the INVARIANTS below, which are what actually matter.

// Serve `candidates` (integer ids) from a free list of `freeCount` entries.
// Returns, for each granted candidate, the free-list index it must read --
// `freeIndex` counts DOWN from freeCount-1, matching the pop discipline.
// Candidates past the end of the free list are refused, lowest id first, so a
// starved pool starves predictably instead of arbitrarily.
export function grantAssignment({ candidates, freeCount }) {
  const ordered = [...candidates].sort((a, b) => a - b);
  const granted = [];
  const refused = [];
  let count = freeCount;
  for (const id of ordered) {
    if (count <= 0) { refused.push(id); continue; }
    count -= 1;
    granted.push({ id, freeIndex: count });
  }
  return { granted, refused, freeCount: count };
}

// The inverse: return `releases` to the free list. Push order is id order, and
// the j-th writes at freeCount + j -- the same "top of the stack" discipline
// read the other way.
export function releaseAssignment({ releases, freeCount }) {
  const ordered = [...releases].sort((a, b) => a - b);
  const writes = ordered.map((id, j) => ({ id, freeIndex: freeCount + j }));
  return { writes, freeCount: freeCount + ordered.length };
}

// ── U0: the uniform level model (plans/uniform-levels.md) ───────────────────
//
// ONE RULE: the root is a level like any other, special only in that it has no
// parent, is always full, and is never allocated or freed. Everything below is
// that rule made arithmetic, stated here ONCE so the GPU can be scored against
// it rather than against a second spelling of it in WGSL.
//
// Nothing consumes these yet. They exist first, alone, and mutation-checked,
// which is the order B0 used and the reason its addressing survived six
// refactors.

// Every level's tile holds the same number of its OWN cells. Today an L1 tile
// covers RB x RB L0 cells; making the ROOT's tile 2*RB x 2*RB root cells is
// what turns level 1 into a quadrant of a root tile -- exactly the relation
// level 2 already has to level 1 -- and collapses three allocation regimes
// into one.
export function tileCellsAtLevel(_m, rb = RB_DEFAULT) { return 2 * rb; }

// THE ROOT HAS NO RING, and that is not an optimisation. A ring exists to hold
// the parent interface; the root has no parent. It is also always full, so
// DIRECT_GHOST always resolves an out-of-tile source against the owning
// same-level tile and never falls back. A ring there would be storage nothing
// writes and nothing reads -- and skipping it is what keeps root memory at
// exactly today's W*H rather than paying 56% for padding.
export function ghostDepthAtLevel(m) { return m === 0 ? 0 : GHOST; }

export function tileSideAtLevel(m, rb = RB_DEFAULT) {
  return tileCellsAtLevel(m, rb) + 2 * ghostDepthAtLevel(m);
}

// The block grid at level m, in blocks. The root is the domain divided by a
// whole tile; every level below doubles both axes.
//
// `dims` is the domain in ROOT cells, which are today's L0 cells -- the root
// stays full, so its cell size does not move.
export function blockGridAtLevel(dims, m, rb = RB_DEFAULT) {
  const tile = tileCellsAtLevel(m, rb);
  if (dims.W % tile !== 0 || dims.H % tile !== 0) {
    throw new Error(`domain ${dims.W}x${dims.H} does not divide into ${tile}-cell root tiles`);
  }
  const s = 1 << m;
  return [(dims.W / tile) * s, (dims.H / tile) * s];
}

// ── the coarse/fine mailbox ────────────────────────────────────────────────
//
// A tile's ring is GHOST fine cells deep and GHOST is 2, so the ring is
// EXACTLY ONE PARENT CELL deep, and a 2x2 block of ring cells is exactly one
// parent cell. That is the whole of the mailbox geometry, and it is also why
// two fine substeps traverse the ring exactly once: the same 2 buys the inbox
// reach inward and the outbox accumulation outward.

// Tile-local fine coordinate -> parent-local cell index. Returns -1 across the
// low ring and `rb` across the high one, which IS the statement that the ring
// is one parent cell deep -- it is not a clamp or a special case.
export function parentCellOfFineCell(f, m, rb = RB_DEFAULT) {
  const g = ghostDepthAtLevel(m);
  return Math.floor((f - g) / 2);
}

// The inverse: the two fine coordinates covering one parent-local cell.
export function fineCellsOfParentCell(p, m, rb = RB_DEFAULT) {
  const g = ghostDepthAtLevel(m);
  return [g + 2 * p, g + 2 * p + 1];
}

// How far outside the interior a tile-local cell sits: 0 interior, 1 or 2 in
// the ring. Per axis, then the max -- a corner cell is as deep as its deepest
// axis, which is what makes a diagonal step inward reduce it.
export function ringDepth(fx, fy, m, rb = RB_DEFAULT) {
  const g = ghostDepthAtLevel(m);
  const hi = g + tileCellsAtLevel(m, rb);
  const axis = (f) => (f < g ? g - f : (f >= hi ? f - hi + 1 : 0));
  return Math.max(axis(fx), axis(fy));
}

// What ROLE a ring cell's population in direction (ex, ey) plays.
//
// A ring cell is NOT A STATE: its nine numbers do not form a distribution, and
// they split by DIRECTION, not by cell. The population at R in direction i is
// consumed by R + e_i, so the direction decides whether this slot is something
// explode delivers inward (inbox), something the fine step pushes outward for
// coalesce to collect (outbox), or neither.
//
// 'tangential' is HONEST, not a catch-all: whether a direction whose target
// sits at the same ring depth must be written, zeroed or left alone is open
// (plans/uniform-levels.md section 7, question 2). Naming it is the point --
// a function that guessed would hide the question.
export function ringSlotRole(fx, fy, ex, ey, m, rb = RB_DEFAULT) {
  const side = tileSideAtLevel(m, rb);
  const d0 = ringDepth(fx, fy, m, rb);
  if (d0 === 0) return 'interior';
  if (ex === 0 && ey === 0) return 'rest';
  const nx = fx + ex, ny = fy + ey;
  if (nx < 0 || ny < 0 || nx >= side || ny >= side) return 'offtile';
  const d1 = ringDepth(nx, ny, m, rb);
  if (d1 < d0) return 'inbox';
  if (d1 > d0) return 'outbox';
  return 'tangential';
}

// ── the pool's own consistency ─────────────────────────────────────────────
//
// blockSlot and slotToBlock are INVERSES. Nothing has ever asserted it.
//
// It is written down now because D0 found a reason to doubt it: removing the
// link passes -- which rewrite blockSlot from slotToBlock every round, and are
// therefore an idempotent repair -- made a deterministic run diverge, with no
// mechanism anyone could name (plans/uniform-levels.md 1.2e). If the pool
// carries a latent inconsistency, that repair has been hiding it, and this is
// the predicate a GPU checker should score.
export function poolInverseViolations(blockSlot, slotToBlock) {
  const bad = [];
  for (let b = 0; b < blockSlot.length; b++) {
    const s = blockSlot[b];
    if (s < 0) continue;
    if (s >= slotToBlock.length) { bad.push({ kind: 'slot-out-of-range', block: b, slot: s }); continue; }
    if (slotToBlock[s] !== b) bad.push({ kind: 'block-slot-block', block: b, slot: s, back: slotToBlock[s] });
  }
  for (let s = 0; s < slotToBlock.length; s++) {
    const b = slotToBlock[s];
    if (b < 0) continue;
    if (b >= blockSlot.length) { bad.push({ kind: 'block-out-of-range', slot: s, block: b }); continue; }
    if (blockSlot[b] !== s) bad.push({ kind: 'slot-block-slot', slot: s, block: b, back: blockSlot[b] });
  }
  return bad;
}

// ── U0: the coarse/fine claimant rule, DERIVED FOR D2Q9 ─────────────────────
//
// 3D's argument for how the corner behaves rests on D3Q19 having no (1,1,+-1)
// direction, so fine channels tile a coarse FACE exactly and a coarse CORNER
// not at all. D2Q9 DOES carry the (+-1,+-1) diagonals, so that counting says
// nothing here and the rule is re-derived from scratch below.
//
// THE DERIVATION, in one paragraph. A coarse cell Q that is not covered still
// solves, and its update pulls f_i from Q - e_i. If Q - e_i is covered, that
// population has to come from the fine level instead of from the coarse grid.
// Over one macro-step a fine population travels two fine cells, which is
// exactly one parent cell and exactly the ring depth, so the mass that left
// Q - e_i heading in direction i has advected into the ring cells occupying
// Q's OWN volume. Hence:
//
//   f_i(Q) = mean of the 4 ring cells covering Q, in the ring of the tile
//            containing Q - e_i,   for every i where Q - e_i is covered
//
// The mean, not the sum: each ring cell carries mass f_i * (h/2)^2 and Q must
// receive f_i(Q) * h^2, so the factor is 1/4 = (fine volume)/(coarse volume).
// In 3D it would be 1/8. There is no separate corner case: a diagonal i is
// handled by the same sentence, because Q - e_i is a single parent cell for a
// diagonal exactly as it is for an axis direction, and a parent cell lies in
// exactly one block.
//
// WHAT THAT PREDICTS, and it is the thing to falsify: the 2D corner needs no
// flux correction and no orphan pass. Every population leaving the refined
// region is claimed by exactly one coarse cell, and the claim is decided by
// ONE parent cell's coverage rather than by the shape of the boundary. If that
// is wrong, transferLedger below is where it shows up.

// The source cell a coarse cell's direction-i population comes from. Coverage
// of THIS cell -- not the shape of the seam -- is the whole rule.
export function coalesceSource(q, dir) { return [q[0] - dir[0], q[1] - dir[1]]; }

// The fine cell a coarse cell's population is delivered TO, going the other
// way: Q's own population in direction i is consumed by Q + e_i.
export function explodeTarget(q, dir) { return [q[0] + dir[0], q[1] + dir[1]]; }

// A ledger of everything that crosses the seam, for one refined-region shape.
//
// THIS EXISTS TO BE FALSIFIABLE. An audit that merely re-applies the rule it
// is auditing is vacuous -- this project has collected three gates like that.
// So the EXPECTED sets are built from coverage alone (a population crosses the
// seam iff one end is covered and the other is not), the ACTUAL sets are built
// by applying the claimant rule, and the two are compared. A wrong rule --
// claiming at Q + e_i, or deciding by the neighbour's tile instead of the
// source cell's -- produces exits nobody claims and claims nobody exits.
//
// `covered(px, py)` decides coverage; the caller supplies it, so the same
// ledger scores a single tile, an L, a hole, or a random closed set.
// Coordinates are parent cells on a periodic `dims.W` x `dims.H` grid.
export function transferLedger({ dims, covered, dirs }) {
  const wrap = (v, n) => ((v % n) + n) % n;
  const cov = (p) => covered(wrap(p[0], dims.W), wrap(p[1], dims.H));
  const key = (p, i) => `${wrap(p[0], dims.W)},${wrap(p[1], dims.H)}|${i}`;

  const exits = new Set();      // covered -> uncovered: mass leaving the fine region
  const entries = new Set();    // uncovered -> covered: mass entering it
  for (let y = 0; y < dims.H; y++) {
    for (let x = 0; x < dims.W; x++) {
      for (let i = 0; i < dirs.length; i++) {
        const d = dirs[i];
        if (d[0] === 0 && d[1] === 0) continue;
        const from = [x, y], to = [x + d[0], y + d[1]];
        if (cov(from) && !cov(to)) exits.add(key(to, i));       // lands at `to`
        if (!cov(from) && cov(to)) entries.add(key(to, i));     // lands at `to`
      }
    }
  }

  // Now apply the RULE, independently of the sets above.
  const claims = new Map();     // key -> how many coarse cells claim it
  const deliveries = new Map();
  for (let y = 0; y < dims.H; y++) {
    for (let x = 0; x < dims.W; x++) {
      const q = [x, y];
      for (let i = 0; i < dirs.length; i++) {
        const d = dirs[i];
        if (d[0] === 0 && d[1] === 0) continue;
        if (!cov(q) && cov(coalesceSource(q, d))) {
          const k = key(q, i);
          claims.set(k, (claims.get(k) || 0) + 1);
        }
        if (cov(q) && !cov(coalesceSource(q, d))) {
          const k = key(q, i);
          deliveries.set(k, (deliveries.get(k) || 0) + 1);
        }
      }
    }
  }

  const unclaimedExits = [...exits].filter(k => !claims.has(k));
  const phantomClaims = [...claims.keys()].filter(k => !exits.has(k));
  const doubleClaims = [...claims.entries()].filter(([, n]) => n > 1).map(([k]) => k);
  const undelivered = [...entries].filter(k => !deliveries.has(k));
  const phantomDeliveries = [...deliveries.keys()].filter(k => !entries.has(k));
  const doubleDeliveries = [...deliveries.entries()].filter(([, n]) => n > 1).map(([k]) => k);

  return {
    exits: exits.size, entries: entries.size,
    unclaimedExits, phantomClaims, doubleClaims,
    undelivered, phantomDeliveries, doubleDeliveries,
    ok: !unclaimedExits.length && !phantomClaims.length && !doubleClaims.length
      && !undelivered.length && !phantomDeliveries.length && !doubleDeliveries.length,
  };
}

// ── U1: the root pool's shape ──────────────────────────────────────────────
//
// The root is a pool level that is ALWAYS FULL and never allocated or freed,
// so its slot count is not a capacity guess like every other level's -- it is
// exactly its block count, forever. `poolSlotsFor`'s 1.7x headroom has no
// meaning here and must not be applied.
//
// The identity that makes this worth doing is `cells === dims.W * dims.H`,
// EXACTLY: because the root has no ring, tiling it costs no padding, and the
// root pool is the same memory the dense grid it replaces already uses. That
// is asserted in tools/test-amr2d.js rather than argued here.
export function rootPoolSpec({ dims, rb = RB_DEFAULT }) {
  const [nbx, nby] = blockGridAtLevel(dims, 0, rb);
  const side = tileSideAtLevel(0, rb);
  return {
    nbx, nby,
    nblocks: nbx * nby,
    slots: nbx * nby,
    side,
    cellsPerSlot: side * side,
    cells: nbx * nby * side * side,
  };
}

// ── the dense L0 grid's own layout ─────────────────────────────────────────
//
// THE DENSE GRID IS NOT ROW-MAJOR. `shaders/amr_step.wgsl`'s `cellIndex`
// groups it into fixed 8x8 buffer-space blocks laid out block-major, row-major
// WITHIN a block, and `debugSnapshotSave` tags every snapshot `layout:
// 'block8'` precisely so nothing downstream forgets it. Every dense-parent
// accessor in shaders/ restates the same formula
// (common_interp_parent_dense.wgsl, common_avg_parent_dense.wgsl,
// amr_force.wgsl, amr_criterion.wgsl, amr_render.wgsl); `tools/lib/
// field-reconstruct.js`'s `rawIndex` is the host's decoder for it.
//
// It is stated HERE because U2 needed it and did not have it, and the cost of
// that is recorded in plans/uniform-levels.md U2: both of U2's supposedly
// independent routes wrote `gy*W + gx`, so they agreed, the mirror was scored
// CLEAN, and 98.4% of the root pool's cells were nonetheless reading the wrong
// dense cell. Two routes are only independent if they do not share a premise,
// and the premise they shared was never written down anywhere either of them
// could be checked against.
//
// `block` is the dense grid's sub-tile, NOT a level's RB. They are equal today
// (main-amr.js's `RB = BLOCK`) and that coincidence is exactly what makes the
// mistake survivable-looking, so it is a separate parameter here.
export const DENSE_BLOCK = 8;

export function denseCellIndex({ dims, block = DENSE_BLOCK }, gx, gy) {
  if (dims.W % block !== 0 || dims.H % block !== 0) {
    throw new Error(`dense grid ${dims.W}x${dims.H} does not divide into ${block}-cell blocks`);
  }
  if (gx < 0 || gy < 0 || gx >= dims.W || gy >= dims.H) {
    throw new Error(`(${gx},${gy}) is outside a ${dims.W}x${dims.H} domain`);
  }
  const nbx = dims.W / block;
  const bx = Math.floor(gx / block), by = Math.floor(gy / block);
  const lx = gx % block, ly = gy % block;
  return (by * nbx + bx) * (block * block) + ly * block + lx;
}

// The root pool's storage index for a cell at SPATIAL coordinates (gx, gy) --
// the companion to `denseCellIndex`, which answers the same question for the
// dense grid.
//
// U4-4 needs it because `readConservedTotals` is already parameterised by a
// `cellIndex` callback: it walks (x, y) in spatial order and asks where that
// cell lives. Hand it this instead of the dense one and the same reduction
// reads the root pool, in the same order, with no other change -- which is
// what makes the two totals comparable EXACTLY rather than to a tolerance.
//
// `rootCellToDense` goes the other way (pool slot+local -> dense index), and
// tools/test-amr2d.js scores the two as a round trip rather than reading them
// side by side, because U2's lesson is that two routes written together agree.
export function rootCellIndex({ dims, rb = RB_DEFAULT }, gx, gy) {
  const side = tileCellsAtLevel(0, rb);
  if (gx < 0 || gy < 0 || gx >= dims.W || gy >= dims.H) {
    throw new Error(`(${gx},${gy}) is outside a ${dims.W}x${dims.H} domain`);
  }
  const [nbx] = blockGridAtLevel(dims, 0, rb);
  const slot = Math.floor(gy / side) * nbx + Math.floor(gx / side);
  return slot * side * side + (gy % side) * side + (gx % side);
}

// ── U2: the root pool's addressing, as a host route ────────────────────────
//
// Where a root pool cell sits in the dense grid it mirrors. This is the
// INDEPENDENT ROUTE shaders/amr_mirror_root.wgsl is scored against: the shader
// walks pool slots and derives a dense index; this walks the same cell from
// the spec and derives it again from `blockGridAtLevel`.
//
// **AGREEMENT BETWEEN THEM IS NOT SUFFICIENT, AND U2 PROVED IT THE HARD WAY.**
// Both routes originally ended in `gy*W + gx` and therefore agreed exactly,
// over a mirror that was reading the wrong dense cell for 98.4% of the root
// pool. The third route -- the one that makes this checkable -- is
// `denseCellIndex` above, scored in tools/test-amr2d.js against
// `tools/lib/field-reconstruct.js`'s `rawIndex`, which is what decodes real
// GPU snapshots and is therefore validated by data neither of these wrote.
//
// Returns a linear cell index into one direction plane of the dense grid,
// which is what `cellIndex()` means on that side.
export function rootCellToDense({ dims, rb = RB_DEFAULT }, slot, lx, ly) {
  const side = tileCellsAtLevel(0, rb);
  if (lx < 0 || ly < 0 || lx >= side || ly >= side) {
    throw new Error(`(${lx},${ly}) is outside a ${side}x${side} root tile`);
  }
  const [nbx] = blockGridAtLevel(dims, 0, rb);
  const bx = slot % nbx;
  const by = Math.floor(slot / nbx);
  const gx = bx * side + lx;
  const gy = by * side + ly;
  if (gx >= dims.W || gy >= dims.H) {
    throw new Error(`slot ${slot} cell (${lx},${ly}) lands outside the domain`);
  }
  return denseCellIndex({ dims }, gx, gy);
}
