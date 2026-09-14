// The GPU-side half of the 2D AMR invariant checks -- plans/2D-backport.md B0.
//
// WHY THIS IS SEPARATE FROM amr2d.mjs. That module is pure: no device, no
// DOM, no readback, so `make test` can exercise it -- including on inputs
// that violate the invariants, which is the only way to know a checker checks
// anything. This file is the other half, the part that can only run against a
// live device: turn the pool's blockSlot buffers into the block SETS that
// module works on.
//
// WHY IT IS SHARED. Before this, FIVE pages carried their own
// debugCheck21Balance and their own readAllBlockSlots -- main-amr.js,
// main-cylinder-amr.js, main-reentry-amr.js, main-tgv-amr.js and
// main-channel-amr.js. The readback was byte-identical in all five; the
// checker had already drifted (the tgv and channel copies silently dropped
// `counts`, and the comment explaining the corner-balance decision survives
// in only three). That is exactly the shape CLAUDE.md records producing
// 238e48c: one rule, five copies, and a change that lands in some of them.
//
// THE ONE-SUBMIT READ IS LOAD-BEARING AND MOVED HERE VERBATIM. An earlier
// version read each level with its own submit + mapAsync round trip, and
// while the sim is RUNNING the render loop keeps submitting macro-steps
// between them -- so level 1 was read at one instant and level 2 at a later
// one, across one or more refinement rounds. A tile coarsened out of level 1
// after the level-1 read, or refined into level 2 before the level-2 read,
// then reads back as a depth-2-next-to-depth-0 pair that never existed at any
// single instant: a TORN SNAPSHOT reported as a 2:1 violation.
//
// That artifact only ever appeared at levels >= 3, because at levels=2 the
// loop runs once and there is nothing to tear against -- which made it look
// like "the second refinement octave breaks 2:1 balance" when what actually
// changed was the number of readbacks. Live-verified by alternating protocols
// on one page: sampling while live reported violations in 2 of 8 samples,
// while the paused read taken immediately after each of those was clean every
// time. Reading every level in ONE submit makes the check sound in both
// modes.

import {
  check21Balance, nbAtLevel, makePool, cellSizeL0AtLevel,
  nearBodyWant, nearBodyWantCentre, bodyPhiL0, bufferToWindow, bufferToWindowLegacy,
} from './amr2d.mjs';

// Every level's blockSlot, copied in one command encoder and one submit, so
// all levels come from the SAME GPU state. Returns { [level]: Set("bx,by") }.
export async function readAllBlockSlots(device, pools, nLevels) {
  const U = GPUBufferUsage;
  const stages = [];
  const enc = device.createCommandEncoder();
  for (let m = 1; m < nLevels; m++) {
    const pool = pools[m];
    const stage = device.createBuffer({ size: pool.NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
    enc.copyBufferToBuffer(pool.blockSlotBuf, 0, stage, 0, pool.NBLOCKS * 4);
    stages.push({ m, stage, pool });
  }
  device.queue.submit([enc.finish()]);
  await Promise.all(stages.map(s => s.stage.mapAsync(GPUMapMode.READ)));
  const sets = {};
  for (const { m, stage, pool } of stages) {
    const blockSlot = new Int32Array(stage.getMappedRange()).slice();
    stage.unmap();
    stage.destroy();
    const active = new Set();
    for (let blockID = 0; blockID < pool.NBLOCKS; blockID++) {
      if (blockSlot[blockID] !== -1) active.add(`${blockID % pool.NBX},${Math.floor(blockID / pool.NBX)}`);
    }
    sets[m] = active;
  }
  return sets;
}

// The page-facing `debugCheck21Balance`. Reads one coherent multi-level
// snapshot, then hands it to amr2d.mjs's pure checker.
//
// THE RETURN SHAPE IS THE OLD ONE, deliberately: `{ ok, violations, counts,
// cornerOk, cornerViolations }`, with `counts` an object keyed by level and
// every violation carrying the `bx`/`by`/`neighbor`/`edge` names the previous
// in-page implementation used, alongside the module's own
// `block`/`neighbour`/`axis`/`dir`. tools/lib/amr-invariants.js and
// tools/validate-amr-invariants.js only read `.ok`, `.cornerOk` and the array
// lengths, and JSON-print the entries -- but a debug surface with five
// callers is not the place to rename fields as a side effect of sharing the
// code.
//
// CORNER BALANCE IS REPORTED AND DOES NOT GATE `ok`. It is a requirement of
// the ghost-free path (?ghostfree=1, whose bilinear parent stencil reads the
// parent's corner cell directly -- measured: before CORNER_BALANCE, 100% of
// its clamp fallbacks were diagonal), not of the default ring path, where
// interp fills a corner ghost from the parent when the corner tile is absent.
// Failing every default run on it would be wrong; callers that need it assert
// `cornerOk` themselves -- see tools/lib/amr-invariants.js's
// requireCornerBalance.
export async function check21BalanceOnGPU(device, pools, nLevels) {
  const activeSets = await readAllBlockSlots(device, pools, nLevels);
  const levelSets = [null];
  const counts = {};
  for (let m = 1; m < nLevels; m++) {
    levelSets[m] = activeSets[m];
    counts[m] = activeSets[m].size;
  }
  // The block grid doubles per level, which is what nbAtLevel says -- but the
  // pools already know their own extents, so take them from the allocation
  // rather than re-deriving and risking the two disagreeing.
  const nbAt = (m) => [pools[m].NBX, pools[m].NBY];
  const r = check21Balance(levelSets, nbAt, { levels: nLevels });
  const legacy = (v) => ({ ...v, bx: v.block[0], by: v.block[1], neighbor: v.neighbour });
  return {
    ok: r.ok,
    violations: r.violations.map(v => ({ ...legacy(v), myDepth: v.level })),
    counts,
    cornerOk: r.cornerOk,
    cornerViolations: r.cornerViolations.map(legacy),
  };
}

// --- pool indirection readbacks (plans/2D-backport.md B3a) ------------------
//
// FIVE COPIES EACH before this. `readPoolIndirection` differed only in line
// wrapping; `debugListActiveBlocks` was byte-identical.
//
// ALWAYS READS GPU STATE, never a CPU mirror. The mirror goes stale the
// instant automatic refinement mutates the pool without the host seeing it,
// which is every REFINE_EVERY macro-steps on a live page -- and a stale
// answer here looks exactly like a correct one.
//
// Per-call staging buffers rather than a shared pair: these are debug paths
// called at checkpoints, and a shared buffer would need the `serializedOn`
// treatment the 3D fork had to build after a second reader entering the gap
// between a submit and its mapAsync took the whole device down.
export async function readPoolIndirection(device, pools, level = 1) {
  const U = GPUBufferUsage;
  const pool = pools[level];
  const stageBlockSlot = device.createBuffer({ size: pool.NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
  const stageSlotToBlock = device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(pool.blockSlotBuf, 0, stageBlockSlot, 0, pool.NBLOCKS * 4);
  enc.copyBufferToBuffer(pool.slotToBlockBuf, 0, stageSlotToBlock, 0, pool.MAX_FINE_BLOCKS * 4);
  device.queue.submit([enc.finish()]);
  await Promise.all([
    stageBlockSlot.mapAsync(GPUMapMode.READ),
    stageSlotToBlock.mapAsync(GPUMapMode.READ),
  ]);
  const blockSlot = new Int32Array(stageBlockSlot.getMappedRange()).slice();
  const slotToBlock = new Int32Array(stageSlotToBlock.getMappedRange()).slice();
  stageBlockSlot.unmap();
  stageSlotToBlock.unmap();
  stageBlockSlot.destroy();
  stageSlotToBlock.destroy();
  return { blockSlot, slotToBlock };
}

// The active blocks of one level, as {bx, by, slot}.
export async function listActiveBlocks(device, pools, level = 1) {
  const pool = pools[level];
  const { blockSlot } = await readPoolIndirection(device, pools, level);
  const active = [];
  for (let blockID = 0; blockID < pool.NBLOCKS; blockID++) {
    if (blockSlot[blockID] !== -1) {
      active.push({ bx: blockID % pool.NBX, by: Math.floor(blockID / pool.NBX), slot: blockSlot[blockID] });
    }
  }
  return active;
}

// --- the pool allocator (plans/2D-backport.md B3a) --------------------------
//
// FIVE BYTE-IDENTICAL COPIES before this, one per AMR page, INCLUDING every
// one of the four BUGFIX comments below. Each of those records a defect that
// was found once and fixed in five places -- or, the next time, would have
// been found once and fixed in one. `NCELLS1` (= FB*FB, the cells in a pool
// slot) is passed in because each page derives it from its own RB; everything
// else this needs is the device.
//
// ── Milestone 5 (plans/AMR-multilevel.md): level-generic pool allocation.
// Same buffer set as today's flat fine-pool globals, one instance per
// level, sized per plans/AMR-multilevel-M5.md's table. Level 1 is
// footprint-preserving with L0 (today's exact scheme, unchanged shapes --
// its "parent" is the dense L0 grid, addressed by blockID/cellIndex, not
// by anything this function allocates). Levels >=2 are genuine quadtree
// children of a level-(m-1) pool tile and carry two extra fields
// (parentSlot/quadrant) that level 1 has no need for. Buffers for levels
// >=2 are allocated eagerly (so ?levels=3 is a real allocation-only smoke
// test, not a no-op) but not bound into a pipeline until Milestone 6/7
// wires them up.
//
// Milestone 5's first draft also allocated ownBX/ownBY (a cached logical
// position per slot) -- Milestone 6 dropped them: a slot's own (bx,by) is
// always derivable from slotToBlock[slot] + this level's own NBX (one
// mod/div), EXACTLY what amr_interp_dense_parent.wgsl's main() already
// does every dispatch for level 1 today. Caching it would have been a
// second, redundant source of truth for zero performance benefit (the
// "expensive" derivation this would save is a single mod+div the project
// already pays for elsewhere in the same hot path) -- see
// shaders/amr_interp_pool_parent.wgsl's header for where the derivation
// actually happens.
export function allocLevelPool(device, U, m, NBX_m, NBY_m, maxFineBlocks, NCELLS1) {
  const NBLOCKS_m = NBX_m * NBY_m;
  const fSizePool_m = maxFineBlocks * NCELLS1 * 9 * 4;
  const pool = {
    level: m,
    NBX: NBX_m, NBY: NBY_m, NBLOCKS: NBLOCKS_m,
    MAX_FINE_BLOCKS: maxFineBlocks,
    fSizePool: fSizePool_m,
    finePoolF_a: device.createBuffer({ size: fSizePool_m, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    finePoolF_b: device.createBuffer({ size: fSizePool_m, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    // COPY_DST is load-bearing, not boilerplate: debugSnapshotLoad writes
    // this buffer via queue.writeBuffer, which is a validation error --
    // silently discarded -- without it. See velBuf's own note below.
    finePoolVel: device.createBuffer({ size: maxFineBlocks * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    blockSlotBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    slotToBlockBuf: device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    blockCriterionBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),  // COPY_SRC so debugReadBlockCriterion can read it back; without it the
    // copy is a validation error, the whole command buffer is dropped, and
    // the staging buffer reads back as all zeros -- which looks exactly like
    // "the criterion pass never ran" and cost a wrong diagnosis once.
    freeCountBuf: device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    newlyActivatedBuf: device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST }),
  };
  if (m === 1) {
    // Per-block allocation, unchanged from today -- L0 isn't itself
    // decomposed into quads, so there's no "quad" on this boundary.
    pool.freeListBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  } else {
    // Quad-unit allocation (decision 3, plans/AMR-multilevel.md:10):
    // refine/coarsen always grants or releases all 4 children of one
    // parent tile together, so the free list is indexed in quads (stride
    // 4), not individual slots.
    if (maxFineBlocks % 4 !== 0) {
      throw new Error(`level ${m}: MAX_FINE_BLOCKS (${maxFineBlocks}) must be a multiple of 4 (quad allocation)`);
    }
    pool.freeListBuf = device.createBuffer({ size: (maxFineBlocks / 4) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // BUGFIX: same "fix at the source, every level" gap as the blockSlot/
    // slotToBlock -1 init below, but for the free-list/free-count pair --
    // level 1 gets its eager freeListBuf/freeCountBuf write from an
    // explicit caller-side write right after the pools loop, but that was
    // never generalized to levels >=2 either. Left at WebGPU's zero-init
    // default, freeCountBuf reads back 0 ("no free quads"), so refine()
    // always takes the "pool exhausted" branch and NO level>=2 quad can
    // ever be granted until something explicitly calls resetSim() --
    // silently, with no GPU validation error, since this is application
    // logic, not an API misuse. resetSim()/debugSnapshotLoad already write
    // these correctly on their own paths; nothing wrote them at bare
    // allocation time, and nothing calls resetSim() automatically on page
    // load, so a fresh page (or any driver script that steps without
    // calling reset() first) saw permanent level>=2 refinement failure.
    const freeQuads_m = maxFineBlocks / 4;
    device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(freeQuads_m).map((_, i) => i));
    device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([freeQuads_m]));
    // New vs. level 1: a quadtree child needs its own parent lookup --
    // which parent-level slot it was carved from (parentSlot) and which
    // of the 4 quadrants it occupies (quadrant) -- see
    // plans/AMR-multilevel-M5.md §2 and shaders/amr_interp_pool_parent.wgsl.
    // COPY_SRC (not just STORAGE|COPY_DST): Milestone 10's debugSnapshotSave
    // reads these back via copyBufferToBuffer -- without it, that copy is an
    // invalid WebGPU command, which poisons the WHOLE shared command encoder
    // (all commands in an invalid GPUCommandBuffer become no-ops on submit),
    // silently zeroing out every OTHER staging buffer in the same save too.
    pool.parentSlotBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.quadrantBuf   = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // Milestone 7: a level>=2 tile's own physical (L0-buffer-space) origin,
    // cached at quad-activation time -- unlike ownBX/ownBY (correctly
    // dropped, see the amendment above), this is NOT cheaply re-derivable
    // per-dispatch: it requires walking the parent chain (this tile's
    // quadrant offset, scaled by the parent's own cell size in L0 units,
    // plus the parent's own origin, recursively), a cross-BUFFER,
    // cross-LEVEL computation, not a same-buffer mod/div. See
    // shaders/amr_step1_pool.wgsl's header.
    pool.originXBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.originYBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // parentSlot has no meaningful "unset" value read anywhere unless
    // slotToBlock already says active (initialized below) -- 0 is harmless
    // filler, not a correctness requirement, so left at WebGPU's own
    // zero-initialized default.
  }
  // BUGFIX: WebGPU zero-initializes new buffers by default -- 0 is a VALID
  // slot/blockID, not "unassigned" (that's -1, this pool's own convention
  // throughout). Every debug/reset path (resetSim, debugSnapshotLoad) was
  // careful to explicitly (re)write -1 before this milestone, but nothing
  // wrote it at bare ALLOCATION time for levels >=2 -- level 1 got it from
  // an explicit caller-side write (main-amr.js's init(), right after the
  // pools loop), but that was never generalized to every level. Exposed by
  // Milestone 8: with N_LEVELS>=3, a fresh page load (no explicit
  // AMR.reset() call) left level 2's entire pool looking "active, slot 0"
  // from frame 1 -- every slot's own force/step/average pass then ran for
  // real, all racing to write the SAME parent location (parentSlot also
  // defaulted to 0). Fixed at the source (every level, not just level 1)
  // rather than special-cased, so this can't recur if a future level's
  // caller-side init is ever forgotten again.
  device.queue.writeBuffer(pool.blockSlotBuf, 0, new Int32Array(NBLOCKS_m).fill(-1));
  device.queue.writeBuffer(pool.slotToBlockBuf, 0, new Int32Array(maxFineBlocks).fill(-1));
  return pool;
}

// --- conservation (plans/2D-backport.md B0b) --------------------------------
//
// TOTAL MASS AND TOTAL MOMENTUM OF THE WHOLE HYBRID SYSTEM, read off the
// DENSE L0 GRID ALONE. That sounds like it must be wrong and it is exactly
// right, for one reason: `average` runs at the end of every macro-step, for
// every level, and overwrites a refined block's L0 cells with the finer
// level's RESTRICTED moments. The restriction is an arithmetic mean of rho
// and a mass-weighted mean of u, and a coarse cell has four times a fine
// cell's area -- so
//
//     rho_avg * A_coarse  =  (SUM_children rho_c) * A_coarse/4
//
// and likewise for rho*u. The coarse cell's contribution to the sum IS its
// children's, exactly, not approximately. Summing the L0 grid therefore gives
// the hybrid system's own totals with no reconstruction and no double count.
// main-tgv-amr.js:1128-1134 already states the velocity half of this for
// `readField`; this is the same argument carried to the moments.
//
// WHY IT IS WORTH MEASURING AT ALL. On a PERIODIC, FORCE-FREE case -- which
// is what `?scenario=tgv` is: no body, no sponge, no walls -- each level
// ALONE conserves mass and momentum exactly. Streaming permutes populations;
// BGK collision preserves the first two moments by construction. So any drift
// is the coarse/fine interface and nothing else. 2D's interface is interp
// (ring ghosts) plus average (restriction) with NO flux correction, so coarse
// cells at the seam stream from their own coarse neighbours while the fine
// tile streams from its ring -- and nothing in the 2D suite has ever measured
// what that costs.
//
// THE ASYMMETRY IS THE DISCRIMINATOR, and it is what found the equivalent 3D
// bug. `fneq` has no zeroth or first moment, so a wrong Dupuis-Chopard rescale
// cannot perturb mass at all while corrupting the viscous stress, which IS the
// momentum flux: "mass at the readback floor while momentum leaks" says the
// defect is in the non-equilibrium coupling, not in the addressing or the
// flux scheme. 2D has that rescale bug today (plans/2D-backport.md B1), so
// this is the instrument that should see it.
//
// THE SUM IS TAKEN IN f64 ON THE HOST, deliberately. The readback is an exact
// copy of the f32 the solver stores, and JS numbers are doubles, so the
// reduction contributes nothing of its own -- unlike an f32 GPU reduction,
// whose own rounding would be indistinguishable from the drift being measured.
// 262144 cells x 9 directions is ~9 MB and a few ms; this is a diagnostic
// called at checkpoints, not per frame.
//
// `ex`/`ey` come from the caller because every page already has the D2Q9
// basis typed out. That is its own duplication (ten-plus copies, see
// plans/2D-backport.md B9, which derives the lattice once and generates the
// WGSL); passing it in avoids adding an eleventh here.
export async function readConservedTotals(device, opts) {
  const { f, W, H, NCELLS, ex, ey, decode, cellIndex } = opts;
  const U = GPUBufferUsage;
  const bytes = f.size;
  const stage = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(f, 0, stage, 0, bytes);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const fArr = decode(stage.getMappedRange(), NCELLS);
  stage.unmap();
  stage.destroy();

  let mass = 0, momX = 0, momY = 0;
  let rhoMin = Infinity, rhoMax = -Infinity, maxU2 = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const cell = cellIndex(x, y);
      let rho = 0, mx = 0, my = 0;
      for (let i = 0; i < 9; i++) {
        const v = fArr[i * NCELLS + cell];
        rho += v; mx += v * ex[i]; my += v * ey[i];
      }
      // rho - 1 rather than rho: the interesting quantity is the DRIFT, and
      // summing 262144 values all near 1 buries it under the total.
      mass += rho - 1;
      momX += mx; momY += my;
      if (rho < rhoMin) rhoMin = rho;
      if (rho > rhoMax) rhoMax = rho;
      const u2 = (mx * mx + my * my) / (rho * rho);
      if (u2 > maxU2) maxU2 = u2;
    }
  }
  return { mass, momX, momY, rhoMin, rhoMax, maxU: Math.sqrt(maxU2), cells: W * H };
}

// THE ALLOCATION'S OWN EXTENTS AGAINST THE RULE. A cheap assertion the pages
// can make at init: pools[m].NBX/NBY must be what the uniform tile shape says
// they are. If the allocation loop and amr2d.mjs ever disagree, every checker
// above is silently working on a different grid than the solver.
export function assertPoolExtents(pools, nLevels, { W, H, rb }) {
  const base = makePool({ dims: [W, H], rb });
  for (let m = 1; m < nLevels; m++) {
    const want = nbAtLevel(base, m);
    if (pools[m].NBX !== want[0] || pools[m].NBY !== want[1]) {
      throw new Error(`level ${m} block grid is ${pools[m].NBX}x${pools[m].NBY}, `
        + `but the quadtree rule says ${want[0]}x${want[1]}`);
    }
  }
}

// --- geometry-forced refinement, scored against real GPU state --------------
//
// THE HARD CONSTRAINT: every LEAF tile whose footprint comes within that
// level's FORCE_REFINE_MARGIN of the body must already have children. Read
// down the levels it says the body lives entirely on the finest level, which
// is what plans/2D-backport.md B4 then lets three force kernels stop
// defending against.
//
// THIS EXISTED ON ONE PAGE OUT OF THREE THAT HAVE A BODY. main-cylinder-amr.js
// carried the only implementation; main-amr.js (the shipped falling card) and
// main-reentry-amr.js move a body through a refined region with no coverage
// gate at all, and main-tgv-amr.js/main-channel-amr.js carried `{ok: true}`
// STUBS -- which defeated tools/lib/amr-invariants.js's deliberate design of
// PROBING for the function and reporting SKIPPED when it is absent, so a
// missing check could not quietly look greener than a present one. The stubs
// are gone (B3a-4); this is the other half.
//
// THE PREDICATE FOLLOWS THE KERNEL, via `boxRefine`. Since B4-2 the managers
// ask about the whole BLOCK (a Lipschitz branch and bound --
// shaders/common_geometry.wgsl's nearBodyBox, amr2d.mjs's `nearBodyWant`);
// `?boxrefine=0` restores the pre-B4 single sample at the block CENTRE, and
// then so does this, truncated window conversion included. A checker is only
// honest if it asks the question the kernel answers, so the flag selects BOTH
// halves here and the page passes its own BOX_REFINE straight through.
//
// EVERY ORIGIN AND SLOT IS READ BACK FRESH, never taken from a CPU mirror:
// under autoRefine the mirror is stale every REFINE_EVERY macro-steps, and a
// stale answer here looks exactly like a correct one.
export async function checkGeometryCoverageOnGPU(device, pools, opts) {
  const {
    nLevels, W, H, rb, NBX, NBLOCKS, cardState,
    paramsForChildLevel, boxRefine = true,
  } = opts;
  const wantFactory = boxRefine ? nearBodyWant : nearBodyWantCentre;
  const toWindow = boxRefine ? bufferToWindow : bufferToWindowLegacy;
  const state = cardState;
  const violations = [];
  if (nLevels < 2) return { ok: true, violations, checked: 0 };

  // The body model every level shares: window conversion, because the body is
  // window-anchored and the buffer is not.
  //
  // THE LOOKAHEAD IS DELIBERATELY NOT APPLIED, and getting this wrong is what
  // made the check fire the first time it was pointed at a MOVING body.
  // FORCE_REFINE_LOOKAHEAD is not part of the constraint -- it is the
  // kernel's MECHANISM for meeting it. The constraint is "no leaf within the
  // margin of the body, EVER", and the manager only gets to decide every
  // REFINE_EVERY macro-steps, so at decision time t0 it refines everything
  // within the margin over [t0, t0 + LOOKAHEAD] and thereby keeps the
  // now-condition true until t0 + REFINE_EVERY.
  //
  // A checker that also applies the lookahead asks about [t, t + LOOKAHEAD]
  // for a t that is up to REFINE_EVERY past t0 -- i.e. it demands coverage of
  // a window the last decision was never responsible for, and reports the
  // leading edge of a moving body as a violation on a perfectly correct run.
  // main-cylinder-amr.js's original never noticed because its body is PINNED:
  // with v = omega = 0 the future pose IS the current one.
  const sdf = (x, y) => {
    const [wx, wy] = toWindow(x, y, state);
    return bodyPhiL0(wx, wy, state, { W, H }, 0);
  };

  let checked = 0;

  // A parent block's footprint in L0 units. Level 0's dense block is RB
  // COARSE cells; a level-m tile's interior is 2*RB cells of size 2^-m --
  // which for m = 1 is the same 8 L0 units, because level 1 is
  // footprint-preserving 1:1 with L0's blocks.
  const extentL0 = (m) => (m === 0 ? rb : 2 * rb * cellSizeL0AtLevel(m));

  const record = (m, bx, by, lo, hi, slot) =>
    violations.push({ level: m, bx, by, slot, wantsChildLevel: m + 1, lo, hi });

  // L0 -> L1. The dense parent: blockID indexes L0's own coarse block and
  // L1's pool block identically.
  {
    const want = wantFactory(sdf, paramsForChildLevel(1).FORCE_REFINE_MARGIN);
    const { blockSlot } = await readPoolIndirection(device, pools, 1);
    const e = extentL0(0);
    for (let blockID = 0; blockID < NBLOCKS; blockID++) {
      if (blockSlot[blockID] !== -1) continue; // has an L1 child -- not a leaf
      const bx = blockID % NBX, by = Math.floor(blockID / NBX);
      const lo = [bx * rb, by * rb], hi = [lo[0] + e, lo[1] + e];
      checked++;
      if (want({ lo, hi, mid: [lo[0] + e / 2, lo[1] + e / 2] })) record(0, bx, by, lo, hi, -1);
    }
  }

  // L(m) -> L(m+1), m = 1 .. nLevels-2. Level 1 caches no origin at all (it
  // is re-derivable from blockID, and allocLevelPool only allocates
  // originX/YBuf for m >= 2) -- the same PARENT_HAS_CACHED_ORIGIN split
  // shaders/amr_manage_pool.wgsl makes.
  for (let m = 1; m < nLevels - 1; m++) {
    const want = wantFactory(sdf, paramsForChildLevel(m + 1).FORCE_REFINE_MARGIN);
    const pool = pools[m];
    const { slotToBlock } = await readPoolIndirection(device, pools, m);
    const { blockSlot: childBlockSlot } = await readPoolIndirection(device, pools, m + 1);
    const nbxChild = pools[m + 1].NBX;
    const origin = m >= 2 ? await readTileOrigins(device, pool) : null;
    const e = extentL0(m);

    for (let slot = 0; slot < pool.MAX_FINE_BLOCKS; slot++) {
      const blockID = slotToBlock[slot];
      if (blockID < 0) continue; // slot not active
      const bx = blockID % pool.NBX, by = Math.floor(blockID / pool.NBX);
      // Quadrant 0 stands for all four -- refinement is quad-complete from
      // level 2 down, the same all-or-nothing invariant
      // amr_manage_pool.wgsl's hasGrandchild leans on.
      if (childBlockSlot[(by * 2) * nbxChild + (bx * 2)] >= 0) continue; // not a leaf
      const lo = origin ? [origin.x[slot], origin.y[slot]] : [bx * rb, by * rb];
      const hi = [lo[0] + e, lo[1] + e];
      checked++;
      if (want({ lo, hi, mid: [lo[0] + e / 2, lo[1] + e / 2] })) record(m, bx, by, lo, hi, slot);
    }
  }

  return { ok: violations.length === 0, violations, checked };
}

// A level's cached per-slot tile origins, in L0 units. Only levels >= 2 have
// them; level 1's is bx*RB by construction.
async function readTileOrigins(device, pool) {
  const U = GPUBufferUsage;
  const bytes = pool.MAX_FINE_BLOCKS * 4;
  const sx = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
  const sy = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(pool.originXBuf, 0, sx, 0, bytes);
  enc.copyBufferToBuffer(pool.originYBuf, 0, sy, 0, bytes);
  device.queue.submit([enc.finish()]);
  await Promise.all([sx.mapAsync(GPUMapMode.READ), sy.mapAsync(GPUMapMode.READ)]);
  const x = new Float32Array(sx.getMappedRange()).slice();
  const y = new Float32Array(sy.getMappedRange()).slice();
  sx.unmap(); sy.unmap(); sx.destroy(); sy.destroy();
  return { x, y };
}

// --- the rigid body's own state -------------------------------------------
//
// THREE BYTE-IDENTICAL COPIES before this (main-cylinder-amr.js,
// main-tgv-amr.js, main-channel-amr.js) -- and, tellingly, NOT on the two
// pages that actually move a body: main-amr.js and main-reentry-amr.js had
// none, which is half of why neither had a geometry-coverage check either.
//
// THE KEY LIST IS shaders/common_geometry.wgsl's `CardState`, IN ORDER, and
// it is the one thing here that can silently rot: the struct is f32-only and
// tightly packed, so inserting a field in the WGSL and not here re-labels
// every field after it rather than failing. 104 bytes = 26 f32s, asserted
// below against the key list for exactly that reason.
export const CARD_STATE_KEYS = [
  'cx', 'cy', 'theta', 'vx', 'vy', 'omega', 'fx', 'fy', 'tz', 'mass',
  'i_body', 'g_eff', 'a', 'b', 'v_max', 'o_max', 'cx_old', 'cy_old', 'th_old',
  'tau', 'y_total', 'x_total', 'off_x', 'off_y', 'off_x_old', 'off_y_old',
];
export const CARD_STATE_BYTES = CARD_STATE_KEYS.length * 4;

export async function readCardState(device, cardStateBuf) {
  const U = GPUBufferUsage;
  const stage = device.createBuffer({ size: CARD_STATE_BYTES, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(cardStateBuf, 0, stage, 0, CARD_STATE_BYTES);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const d = new Float32Array(stage.getMappedRange());
  const out = {};
  CARD_STATE_KEYS.forEach((k, i) => { out[k] = d[i]; });
  stage.unmap();
  stage.destroy();
  return out;
}
