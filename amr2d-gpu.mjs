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
  nearBodyWant, nearBodyWantCentre, bodyPhiL0, bodyFrameL0, bodyFrameL0Legacy,
  cascade21, quadrantOfSlot, tileOriginL0, poolInverseViolations,
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
    // THE WANT SET at this level, one u32 per BLOCK (not per slot): does a
    // level-m tile want to exist here, before any allocation has happened?
    // plans/2D-backport.md B2 -- the 2:1 closure runs on this between the
    // criterion and coarsen/refine, which is what lets the per-pass balance
    // tests and the fixed-point loop go away. COPY_DST so a test can SEED it
    // (including with sets that violate the invariant), COPY_SRC so the
    // result can be scored against amr2d.mjs's host twin.
    wantBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    // D0's candidate rank, one i32 per block (plans/uniform-levels.md 1.2c).
    // Allocated HERE rather than in each page, which is the whole point of
    // this function: the five AMR pages each build their own manage bind
    // group, and a buffer added five times is a buffer that will eventually be
    // added four times. COPY_SRC so a test can read the ranks back and score
    // them against amr2d.mjs's grantAssignment.
    candRankBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
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
    // WRITTEN ONCE, HERE, because its content never changes: a slot's
    // quadrant is `slot % 4` for the life of the pool (amr2d.mjs's
    // quadrantOfSlot -- both allocators compose the slot as
    // `quadIdx*4 + quadrant`). shaders/amr_manage_pool.wgsl's refine() used
    // to rewrite it on every allocation and needed a whole binding to do it,
    // from a kernel that sits at the 16-buffer per-stage ceiling.
    //
    // Same "fix at the source, for every level" reasoning as the
    // freeList/freeCount write below -- and the same hazard if it is not
    // done here: several other shaders read this buffer, and with nothing
    // writing it per-refine any longer, a zero-initialised or reset copy
    // would be silently wrong for all of them.
    device.queue.writeBuffer(pool.quadrantBuf, 0,
      new Uint32Array(maxFineBlocks).map((_, slot) => quadrantOfSlot(slot)));
    // NO originX/originY BUFFERS (plans/2D-backport.md B3-5). Milestone 7
    // allocated a per-slot cached origin on the argument that it was not
    // cheaply re-derivable -- it required walking the parent chain, a
    // cross-BUFFER, cross-LEVEL computation rather than a same-buffer
    // mod/div. That argument was wrong, and amr2d.mjs had said so since B0:
    // every level's block grid is globally anchored and quadtree-uniform, so
    // tileOriginL0 is `block * RB * 2^-(m-1)` in closed form. Three kernels
    // derive it now (amr_step1.wgsl, amr_force1.wgsl, amr_manage_pool.wgsl's
    // parentOriginL0) and nothing stores it.
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
  // THE ROOT IS ALWAYS FULL, so its indirection is the identity and is written
  // once, here, over the -1 fill above. plans/uniform-levels.md U1.
  //
  // It is a level like any other in every respect this function cares about --
  // same buffers, same addressing -- and differs only in the three things that
  // follow from having no parent: every slot is permanently assigned, nothing
  // is ever granted or released (so the free list is allocated and left
  // untouched rather than seeded), and `newlyActivated` never fires because no
  // slot is ever new.
  //
  // `maxFineBlocks === NBLOCKS_m` is REQUIRED, not defaulted: a root pool that
  // could not hold its own domain would be a silent hole in the grid, and
  // poolSlotsFor's 1.7x headroom has no meaning for a level that never grows.
  if (m === 0) {
    if (maxFineBlocks !== NBLOCKS_m) {
      throw new Error(`root pool must have exactly one slot per block (${maxFineBlocks} slots, ${NBLOCKS_m} blocks)`);
    }
    const identity = new Int32Array(NBLOCKS_m).map((_, i) => i);
    device.queue.writeBuffer(pool.blockSlotBuf, 0, identity);
    device.queue.writeBuffer(pool.slotToBlockBuf, 0, identity);
    pool.isRoot = true;
  }
  return pool;
}

// The root's indirection is the identity, and blockSlot/slotToBlock are
// inverses. Scored on the LIVE buffers, not on the host's intent -- the same
// reason debugCheckSlotQuadrants scores quadrantBuf against quadrantOfSlot
// rather than trusting the write that produced it.
export async function checkRootPoolIdentity(device, pools) {
  const pool = pools[0];
  if (!pool) return { ok: null, skipped: 'no root pool allocated' };
  const { blockSlot, slotToBlock } = await readPoolIndirection(device, pools, 0);
  const notIdentity = [];
  for (let i = 0; i < blockSlot.length; i++) {
    if (blockSlot[i] !== i) notIdentity.push({ kind: 'blockSlot', at: i, got: blockSlot[i] });
  }
  for (let s2 = 0; s2 < slotToBlock.length; s2++) {
    if (slotToBlock[s2] !== s2) notIdentity.push({ kind: 'slotToBlock', at: s2, got: slotToBlock[s2] });
  }
  const inverse = poolInverseViolations(Array.from(blockSlot), Array.from(slotToBlock));
  return {
    ok: notIdentity.length === 0 && inverse.length === 0,
    blocks: blockSlot.length, slots: slotToBlock.length,
    notIdentity: notIdentity.slice(0, 8), inverse: inverse.slice(0, 8),
  };
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
  const toBody = boxRefine ? bodyFrameL0 : bodyFrameL0Legacy;
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
    const [bx2, by2] = toBody(x, y, state);
    return bodyPhiL0(bx2, by2, state, { W, H }, 0);
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

  // L(m) -> L(m+1), m = 1 .. nLevels-2. No level caches an origin since B3-5;
  // tileOriginL0 is the closed form every kernel now derives, so this uses it
  // directly instead of reading a buffer back for levels >= 2.
  for (let m = 1; m < nLevels - 1; m++) {
    const want = wantFactory(sdf, paramsForChildLevel(m + 1).FORCE_REFINE_MARGIN);
    const pool = pools[m];
    const { slotToBlock } = await readPoolIndirection(device, pools, m);
    const { blockSlot: childBlockSlot } = await readPoolIndirection(device, pools, m + 1);
    const nbxChild = pools[m + 1].NBX;
    const e = extentL0(m);

    for (let slot = 0; slot < pool.MAX_FINE_BLOCKS; slot++) {
      const blockID = slotToBlock[slot];
      if (blockID < 0) continue; // slot not active
      const bx = blockID % pool.NBX, by = Math.floor(blockID / pool.NBX);
      // Quadrant 0 stands for all four -- refinement is quad-complete from
      // level 2 down, the same all-or-nothing invariant
      // amr_manage_pool.wgsl's hasGrandchild leans on.
      if (childBlockSlot[(by * 2) * nbxChild + (bx * 2)] >= 0) continue; // not a leaf
      const lo = tileOriginL0([bx, by], m, rb);
      const hi = [lo[0] + e, lo[1] + e];
      checked++;
      if (want({ lo, hi, mid: [lo[0] + e / 2, lo[1] + e / 2] })) record(m, bx, by, lo, hi, slot);
    }
  }

  return { ok: violations.length === 0, violations, checked };
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

// --- geometry-forced refinement REFUSED: latch, don't degrade --------------
//
// THE HAZARD. A refine that cannot get a pool slot is silently abandoned --
// amr_manage.wgsl undoes its atomicSub and the block stays coarse; so does
// amr_manage_pool.wgsl one level down. For a criterion-driven refine that is
// the design working (the pool is a budget and vorticity demand is
// unbounded). For a GEOMETRY-forced one it is the hard constraint being
// refused, which means a coarse/fine seam through the body -- and since B4-3
// only the finest level computes force at all, the refused region contributes
// NOTHING rather than something crude. The run keeps going and looks healthy.
//
// Live reproducer, found by B4-2's probe: index-cylinder-amr.html?levels=4 has
// level 3 at 128/128 with 20 coverage violations, and reads L2 fx=-2.33
// against L3 fx=+2.26 -- large cancelling contributions from a body split
// across two levels.
//
// WHY A TRIP-WIRE AND THEN THE REAL CHECK, rather than a counter in the
// shader. A refusal counter is the direct signal, but the pool manager has no
// diag binding at all today, so adding one is a new binding on two manage
// shaders across FIVE pages' bind groups -- which is precisely the change
// shape that shipped 238e48c. This needs no shader change:
//
//   1. TRIP-WIRE, cheap and always sound in one direction: read every level's
//      freeCount (4 bytes each, one submit). Exhaustion IMPLIES freeCount ==
//      0 at the end of the round, so a zero cannot be missed. It is not
//      sufficient -- a pool exactly consumed reads zero too.
//   2. AUTHORITY: on a zero, run the coverage check, which asks the actual
//      question ("is any leaf within the margin missing its children"). Only
//      that decides.
//
// So benign saturation costs one extra readback and reports nothing, and a
// real refusal is named by the check that defines the constraint.
//
// LIVE LOOP ONLY, deliberately. debugStepSync is not gated by this: the
// validation harness runs debugCheckGeometryCoverage itself, periodically and
// unconditionally (tools/lib/amr-invariants.js), which is a stronger check
// than this trip-wire and already fails loudly. A second implicit latch there
// would change what existing tooling means without adding coverage.
export function makeRefusalWatch({ device, pools, nLevels, checkCoverage, minIntervalMs = 500 }) {
  const U = GPUBufferUsage;
  let inFlight = false;
  let last = -Infinity;
  const watch = { error: null };

  watch.poll = () => {
    if (watch.error || inFlight) return;
    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    if (now - last < minIntervalMs) return;
    last = now;
    inFlight = true;
    (async () => {
      try {
        // One submit for every level, same discipline as readAllBlockSlots:
        // a torn read across levels would be a different topology than any
        // that existed.
        const stages = [];
        const enc = device.createCommandEncoder();
        for (let m = 1; m < nLevels; m++) {
          const stage = device.createBuffer({ size: 4, usage: U.MAP_READ | U.COPY_DST });
          enc.copyBufferToBuffer(pools[m].freeCountBuf, 0, stage, 0, 4);
          stages.push({ m, stage });
        }
        device.queue.submit([enc.finish()]);
        await Promise.all(stages.map(s => s.stage.mapAsync(GPUMapMode.READ)));
        const saturated = [];
        for (const { m, stage } of stages) {
          const free = new Int32Array(stage.getMappedRange())[0];
          stage.unmap();
          stage.destroy();
          if (free <= 0) saturated.push(m);
        }
        if (saturated.length === 0) return;

        const cov = await checkCoverage();
        if (cov.ok) return;   // saturated but still covered -- a budget, not a breach
        const v = cov.violations[0];
        // NAME THE KNOB THAT ACTUALLY REACHES THE SATURATED LEVEL. This said
        // "Raise ?maxFineBlocks=" for every level, and that parameter sizes
        // LEVEL 1 ONLY -- levels >= 2 read ?maxFineBlocks<m>= (main-amr.js and
        // main-cylinder-amr.js both route it that way). So the one piece of
        // advice a user gets on a level-3 exhaustion sent them to a no-op, and
        // raising it looked like the refusal was spurious rather than the knob
        // being wrong. Live-verified 2026-09-15 on
        // index-amr.html?levels=4: ?maxFineBlocks=4096 still refuses at level
        // 3, while ?maxFineBlocks3=4096 runs clean past 84k steps.
        const knobs = saturated.map(m => (m === 1 ? '?maxFineBlocks=' : `?maxFineBlocks${m}=`)).join(' / ');
        watch.error = `geometry-forced refinement REFUSED: level ${saturated.join(',')} pool exhausted `
          + `and ${cov.violations.length} leaf tile(s) within the body's margin have no children `
          + `(first: level ${v.level} block ${v.bx},${v.by}). The body is split across levels and only the `
          + `finest computes force, so this run's force is wrong. Raise ${knobs} (that level's own cap -- `
          + `?maxFineBlocks= sizes level 1 only) or lower ?levels=.`;
      } catch (e) {
        // A readback failure is not evidence of a refusal; say so rather than
        // latching on it, and never take the page down for the watchdog.
        console.warn('[getting-air] refusal watch readback failed:', e);
      } finally {
        inFlight = false;
      }
    })();
  };

  return watch;
}

// --- how far the SHIPPED topology is from closed ---------------------------
//
// THE CLOSURE, RUN ON WHAT ACTUALLY EXISTS. amr2d.mjs's `cascade21` is the
// 2:1 rule as ONE function -- present(m, b) => present(m-1, parent(n)) for
// every neighbour n of b at level m, and for b itself. Run it on the live
// present set and everything it ADDS is a block the rule says must exist and
// does not. `forced` records WHY each one was added, and the four reasons are
// four different defects:
//
//   parentOf      the tree property: a block whose own parent tile is absent.
//   siblingOf     quad incompleteness: a level>=2 block without its three
//                 siblings, which the pool cannot actually represent.
//   neighbourOf   with an EDGE offset: plain 2:1 balance.
//     "  "        with a DIAGONAL offset: the RING. amr_step1.wgsl reads the
//                 corner cell whenever the diagonal same-level neighbour is
//                 absent, and its parent tile has to exist for that read to
//                 land. This is the half debugCheck21Balance reports and
//                 deliberately does not gate.
//
// WHY THIS IS A MEASUREMENT AND NOT YET A GATE. The shipped manager
// implements the rule as per-pass tests inside coarsen and refine, wrapped in
// a fixed-point loop -- and only the VETO half of the refine cascade was ever
// written (main-amr.js:112-131), so a criterion-driven refine can be blocked
// forever by a neighbour that would only ever have been created BY that
// refine. plans/2D-backport.md B2 replaces the whole arrangement with this
// closure applied to the WANT set between decide and coarsen/refine. This
// exists so that change has a before-number instead of an argument.
export async function checkRefinementClosureOnGPU(device, pools, nLevels) {
  const activeSets = await readAllBlockSlots(device, pools, nLevels);
  const levelSets = [null];
  for (let m = 1; m < nLevels; m++) levelSets[m] = activeSets[m];
  const nbAt = (m) => [pools[m].NBX, pools[m].NBY];

  const present = levelSets.map((s, m) => (m === 0 ? null : s.size));
  const r = cascade21(levelSets, nbAt, { levels: nLevels });

  // Classify by the reason cascade21 recorded, which is the whole point of it
  // recording one.
  const byReason = { parentOf: 0, siblingOf: 0, edge: 0, diagonal: 0 };
  const byLevel = {};
  for (const f of r.forced) {
    const b = f.because;
    if (b.siblingOf !== undefined) byReason.siblingOf++;
    else if (b.parentOf !== undefined) byReason.parentOf++;
    else if (b.offset) (b.offset[0] !== 0 && b.offset[1] !== 0 ? byReason.diagonal++ : byReason.edge++);
    byLevel[f.level] = (byLevel[f.level] || 0) + 1;
  }
  return {
    ok: r.forced.length === 0,
    missing: r.forced.length,
    present, byReason, byLevel,
    // Capped: a badly-unbalanced tree can force thousands, and the first
    // handful is what anyone actually reads.
    sample: r.forced.slice(0, 8),
  };
}

// --- the GPU cascade, and scoring it against the host twin -----------------
//
// shaders/amr_cascade.wgsl is the 2:1 closure as two entry points on the want
// arrays. This builds its pipelines and drives it, and -- the part that
// matters -- scores it against amr2d.mjs's `cascade21` on SEEDED want sets.
//
// WHY SEEDED AND NOT LIVE. A closure only ever run on valid input is
// indistinguishable from one that returns its input unchanged (B0's third
// rule). The live criterion produces whatever it produces; a seed can be an
// arbitrary set, including ones that violate 2:1 balance by three levels, sit
// on the periodic seam, or want one child of a quad with no siblings. Those
// are the inputs that tell the two implementations apart.
//
// SHARED, not per-page: five AMR pages would otherwise carry five copies of a
// pipeline set and a bind group, which is the shape CLAUDE.md records
// producing 238e48c.
export function makeCascadePipelines(device, loadedModule, pools, nLevels) {
  const bgl = device.createBindGroupLayout({
    label: 'cascadeBGL',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const byLevel = {};
  // Levels >= 2 only: level 1's parent is the dense L0 grid, which is present
  // everywhere, so nothing cascades OUT of level 1 -- the same reason
  // cascade21's own loop stops at 2.
  for (let m = nLevels - 1; m >= 2; m--) {
    const pool = pools[m];
    const constants = { NBX: pool.NBX, NBY: pool.NBY, QUAD_COMPLETE: 1 };
    byLevel[m] = {
      completeQuads: device.createComputePipeline({
        layout, compute: { module: loadedModule, entryPoint: 'completeQuads', constants },
      }),
      balance: device.createComputePipeline({
        layout, compute: { module: loadedModule, entryPoint: 'balance', constants },
      }),
      bg: device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: pool.wantBuf } },
          { binding: 1, resource: { buffer: pools[m - 1].wantBuf } },
        ],
      }),
      workgroups: Math.ceil((pool.NBX * pool.NBY) / 64),
    };
  }
  return { bgl, byLevel };
}

// ONE SWEEP, DEEPEST FIRST. See the shader's header on why this reaches the
// fixed point without iterating: a want at level m forces wants at level m-1
// only, so the propagation is one-directional down the levels.
export function encodeCascade(enc, cascade, nLevels) {
  for (let m = nLevels - 1; m >= 2; m--) {
    const c = cascade.byLevel[m];
    for (const entry of ['completeQuads', 'balance']) {
      const p = enc.beginComputePass();
      p.setPipeline(c[entry]);
      p.setBindGroup(0, c.bg);
      p.dispatchWorkgroups(c.workgroups);
      p.end();
    }
  }
}

// Every level's want array, as the "bx,by" Sets amr2d.mjs works in, read in
// ONE submit so all levels come from the same GPU state (the torn-snapshot
// discipline readAllBlockSlots' header explains at length).
export async function readWantSets(device, pools, nLevels) {
  const U = GPUBufferUsage;
  const stages = [];
  const enc = device.createCommandEncoder();
  for (let m = 1; m < nLevels; m++) {
    const pool = pools[m];
    const stage = device.createBuffer({ size: pool.NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
    enc.copyBufferToBuffer(pool.wantBuf, 0, stage, 0, pool.NBLOCKS * 4);
    stages.push({ m, stage, pool });
  }
  device.queue.submit([enc.finish()]);
  await Promise.all(stages.map(s => s.stage.mapAsync(GPUMapMode.READ)));
  const sets = [null];
  for (const { m, stage, pool } of stages) {
    const w = new Uint32Array(stage.getMappedRange());
    const set = new Set();
    for (let b = 0; b < pool.NBLOCKS; b++) {
      if (w[b] !== 0) set.add(`${b % pool.NBX},${Math.floor(b / pool.NBX)}`);
    }
    stage.unmap();
    stage.destroy();
    sets[m] = set;
  }
  return sets;
}

export function writeWantSets(device, pools, nLevels, sets) {
  for (let m = 1; m < nLevels; m++) {
    const pool = pools[m];
    const w = new Uint32Array(pool.NBLOCKS);
    for (const key of (sets[m] || [])) {
      const [bx, by] = key.split(',').map(Number);
      w[by * pool.NBX + bx] = 1;
    }
    device.queue.writeBuffer(pool.wantBuf, 0, w);
  }
}

// THE GATE: seed both implementations with the same want set, close it on the
// GPU and on the host, and require the two to agree EXACTLY -- not to a
// tolerance, and not merely in size. Returns the disagreement both ways round,
// because "the GPU forced something the host did not" and "the host forced
// something the GPU did not" are different bugs.
// IT WRITES LIVE STATE, and since B2 wired the want arrays into the manager
// that is worth saying out loud: seeding clobbers whatever the last refinement
// round decided. It self-heals -- the next round clears and rewrites every
// want buffer before reading it, which is exactly why that clear is not
// optional -- but do not read a topology measurement taken between this and
// the next refinement round and expect it to mean anything.
export async function cascadeRoundTrip(device, pools, nLevels, cascade, seedSets) {
  writeWantSets(device, pools, nLevels, seedSets);
  const enc = device.createCommandEncoder();
  encodeCascade(enc, cascade, nLevels);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const gpu = await readWantSets(device, pools, nLevels);

  const nbAt = (m) => [pools[m].NBX, pools[m].NBY];
  const host = cascade21(seedSets, nbAt, { levels: nLevels }).sets;

  const perLevel = [];
  let ok = true;
  for (let m = 1; m < nLevels; m++) {
    const g = gpu[m], h = host[m];
    const gpuOnly = [...g].filter(k => !h.has(k));
    const hostOnly = [...h].filter(k => !g.has(k));
    if (gpuOnly.length || hostOnly.length) ok = false;
    perLevel.push({
      level: m, gpu: g.size, host: h.size,
      gpuOnly: gpuOnly.slice(0, 6), hostOnly: hostOnly.slice(0, 6),
      gpuOnlyCount: gpuOnly.length, hostOnlyCount: hostOnly.length,
    });
  }
  return { ok, perLevel, seeded: seedSets.map((s, m) => (m === 0 ? null : s.size)) };
}

// THE SEED BATTERY, and it is the part that does the work.
//
// A closure only ever run on VALID input is indistinguishable from one that
// returns its input unchanged -- B0's third rule, and the reason
// tools/test-amr2d.js runs its checkers on inputs that violate the invariant.
// The same applies to a GPU closure, with one extra hazard the host twin does
// not have: a data race would show up only where two threads touch the same
// slot, which needs a seed DENSE enough to produce collisions.
//
// So the battery is built from the failure modes, not from typical states:
//
//   deep-single      one block at the finest level and nothing else. Forces a
//                    chain all the way up -- the transitivity the shipped
//                    fixed-point loop needed iterations for.
//   quad-partial     ONE child of a quad, no siblings. The criterion really
//                    can produce this (it is evaluated per block and nothing
//                    in it looks sideways), and it is a state the POOL cannot
//                    represent. 3D's first cascade rejected it instead of
//                    completing it and its checker was right.
//   seam             a block at (0,0), so every parent it forces is across
//                    the periodic wrap. Every earlier geometry fixture in this
//                    project sat in the MIDDLE of the grid, and a mutant that
//                    dropped the wrap broke nothing until a seam fixture was
//                    added.
//   seam-corner      (0,0) AND (NBX-1, NBY-1) -- diagonal neighbours across
//                    both wraps at once, which is the ring case in the one
//                    place the arithmetic can go wrong in both axes.
//   dense-band       a full row of the finest level. Thousands of threads
//                    writing overlapping parents in one dispatch: the seed
//                    that would expose a non-monotone write.
//   already-closed   the output of the host closure fed back in. Must come
//                    back UNCHANGED -- idempotence, which is what makes "one
//                    deepest-first sweep is the fixed point" checkable rather
//                    than asserted.
//   empty            nothing wanted. Must stay empty; a closure that forces
//                    anything from nothing is forcing it from a bug.
export function makeCascadeSeeds(pools, nLevels) {
  const finest = nLevels - 1;
  if (finest < 2) {
    // With one pool level there is nothing to cascade -- level 1's parent is
    // the dense grid. Say so rather than returning a battery that passes
    // vacuously.
    return [{ name: 'levels<3: nothing to cascade', sets: [null, new Set()] }];
  }
  const nb = (m) => [pools[m].NBX, pools[m].NBY];
  const blank = () => { const s = [null]; for (let m = 1; m < nLevels; m++) s[m] = new Set(); return s; };
  const at = (m, ...keys) => { const s = blank(); for (const k of keys) s[m].add(k); return s; };

  const [fx, fy] = nb(finest);
  const seeds = [
    { name: 'empty', sets: blank() },
    { name: 'deep-single', sets: at(finest, `${fx >> 1},${fy >> 1}`) },
    // (7,5) is deliberately odd in both axes: the quad's other three members
    // are (6,4), (7,4), (6,5), so a completion that rounded the wrong way
    // would land on a different quad entirely.
    { name: 'quad-partial', sets: at(finest, '7,5') },
    { name: 'seam', sets: at(finest, '0,0') },
    { name: 'seam-corner', sets: at(finest, '0,0', `${fx - 1},${fy - 1}`) },
  ];

  {
    const s = blank();
    for (let bx = 0; bx < fx; bx++) s[finest].add(`${bx},${fy >> 1}`);
    seeds.push({ name: 'dense-band', sets: s });
  }
  {
    // A mixed set with wants at EVERY level, so the sweep has to compose its
    // own output with the caller's input at each rung rather than only ever
    // seeing one of the two.
    const s = blank();
    for (let m = 1; m < nLevels; m++) {
      const [x, y] = nb(m);
      s[m].add(`${(x >> 1) + m},${(y >> 1) - m}`);
    }
    seeds.push({ name: 'mixed-levels', sets: s });
  }
  // IDEMPOTENCE. Feed the host closure's own output back in; it must come
  // back unchanged. That is what makes "one deepest-first sweep IS the fixed
  // point" a checked property rather than an assertion -- and it is the
  // property the shipped FIXED_POINT_ITERS loop exists because nobody had.
  for (const base of ['deep-single', 'dense-band', 'mixed-levels']) {
    const src = seeds.find(x => x.name === base);
    seeds.push({ name: `already-closed:${base}`, sets: closeSeed(pools, nLevels, src.sets) });
  }
  return seeds;
}

// Idempotence needs the host closure's own output as a seed, which the caller
// cannot build without cascade21 -- so it is derived here rather than asking
// every page to import the pure module too.
export function closeSeed(pools, nLevels, sets) {
  const nbAt = (m) => [pools[m].NBX, pools[m].NBY];
  return cascade21(sets, nbAt, { levels: nLevels }).sets;
}

// --- the stored quadrant against the rule ----------------------------------
//
// amr2d.mjs's `quadrantOfSlot` says a slot's quadrant is `slot % 4`, because
// both allocators compose the slot as `quadIdx*4 + quadrant`. This reads the
// live buffer and checks it -- over ACTIVE slots, which is the only place the
// value has ever been written.
//
// WHY BOTHER, when the argument is two lines of arithmetic: the buffer is
// about to stop being written by shaders/amr_manage_pool.wgsl's refine (it is
// the cheapest binding to recover from that kernel's exactly-16 ceiling), and
// several OTHER shaders still read it. So the claim being relied on is not
// "the arithmetic is right" but "nothing on any path has ever put a different
// value there" -- including debugSnapshotLoad, which writes whatever a
// snapshot recorded. That is an empirical claim about live state, and this is
// how it gets checked rather than asserted.
// THE TILE-ORIGIN GATE WAS HERE, AND IT IS GONE ON PURPOSE (B3-5).
//
// checkTileOriginsOnGPU scored the live originX/originY buffers against
// amr2d.mjs's closed form, and was added precisely to justify taking the
// kernels off those buffers. That done, B3-5 deleted the buffers -- so the
// checker had nothing left to score, and a checker pointed at deleted state
// is exactly how this project collected three vacuous gates (B2-2d's sweep).
// It goes in the same commit as the thing it was checking, deliberately.
//
// The rule it enforced is not unguarded: it is now computed, not stored, in
// one closed form in three kernels, and tools/test-amr2d.js still scores that
// closed form against the recursive route on the host.

export async function checkSlotQuadrantsOnGPU(device, pools, nLevels) {
  const U = GPUBufferUsage;
  const stages = [];
  const enc = device.createCommandEncoder();
  // Levels >= 2 only: level 1 allocates per block, not per quad, and has no
  // quadrantBuf at all.
  for (let m = 2; m < nLevels; m++) {
    const pool = pools[m];
    const bytes = pool.MAX_FINE_BLOCKS * 4;
    const q = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    const s2b = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    enc.copyBufferToBuffer(pool.quadrantBuf, 0, q, 0, bytes);
    enc.copyBufferToBuffer(pool.slotToBlockBuf, 0, s2b, 0, bytes);
    stages.push({ m, q, s2b, pool });
  }
  if (!stages.length) return { ok: true, checked: 0, violations: [] };
  device.queue.submit([enc.finish()]);
  await Promise.all(stages.flatMap(s => [s.q.mapAsync(GPUMapMode.READ), s.s2b.mapAsync(GPUMapMode.READ)]));

  const violations = [];
  let checked = 0;
  for (const { m, q, s2b, pool } of stages) {
    const quad = new Uint32Array(q.getMappedRange());
    const slotToBlock = new Int32Array(s2b.getMappedRange());
    for (let slot = 0; slot < pool.MAX_FINE_BLOCKS; slot++) {
      if (slotToBlock[slot] < 0) continue;   // never allocated -- never written
      checked++;
      const want = quadrantOfSlot(slot);
      if (quad[slot] !== want) {
        if (violations.length < 8) violations.push({ level: m, slot, stored: quad[slot], rule: want });
      }
    }
    q.unmap(); s2b.unmap(); q.destroy(); s2b.destroy();
  }
  return { ok: violations.length === 0, checked, violations };
}
