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
  cascade21, quadrantOfSlot, tileOriginL0, poolInverseViolations, rootPoolSpec,
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
// `quadAlloc` chooses the ALLOCATION UNIT, and it is a parameter rather than a
// function of the level since plans/uniform-levels.md U5-4. Level 1 allocated
// per BLOCK for as long as its parent was the dense grid -- L0 is not itself
// decomposed into quads, so there was no quad on that boundary. Once the root
// is a pool level there is one, and level 1 becomes a quad child like every
// other level. Default `m !== 1` is exactly the old rule, so a page that does
// not ask is unchanged.
export function allocLevelPool(device, U, m, NBX_m, NBY_m, maxFineBlocks, NCELLS1, { quadAlloc } = {}) {
  const quad = quadAlloc ?? (m !== 1);
  const NBLOCKS_m = NBX_m * NBY_m;
  const fSizePool_m = maxFineBlocks * NCELLS1 * 9 * 4;
  const pool = {
    level: m,
    NBX: NBX_m, NBY: NBY_m, NBLOCKS: NBLOCKS_m,
    MAX_FINE_BLOCKS: maxFineBlocks,
    // THIS LEVEL'S OWN cells-per-slot, and it is NOT the same at every level.
    // The root tile has no ghost ring (amr2d.mjs's ghostDepthAtLevel(0) is 0),
    // so it is (2*RB)^2 where every other level's is (2*RB + 2*GHOST)^2. A
    // caller that reaches for the page's module-level NCELLS1 is right for
    // levels >= 1 and wrong for the root -- which is exactly what
    // debugPerturbLevelVel did, writing 400 cells per root block into a buffer
    // holding 256 and having the whole oversized write discarded (U7-6b).
    cellsPerSlot: NCELLS1,
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
  pool.quadAlloc = quad;
  if (!quad) {
    // Per-block allocation: level 1 while its parent is the DENSE grid, which
    // is not itself decomposed into quads, so there is no quad on that
    // boundary. See this function's `quadAlloc` note.
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

// --- THE ROOT POOL IN A SNAPSHOT (plans/uniform-levels.md U7-6c) -----------
//
// SHARED, unlike the rest of debugSnapshotSave/Load. CLAUDE.md's rule is that
// those stay per-page because "they serialise whatever state that page owns",
// and the root pool is the one part of them that is NOT page state: its shape
// comes from `rootPoolSpec` and is the same on every page that has one. Three
// pages carry snapshots, and three copies of a format is three chances to
// carry a different one.
//
// WHAT IS AND IS NOT SAVED. The field (`f`) and the VELOCITY, and nothing
// else. The root's blockSlot/slotToBlock are the IDENTITY, written once by
// allocLevelPool and never touched again -- there is no grant or release at a
// level with no parent -- so they are shape, not state, and `checkRootPoolIdentity`
// already scores them against that rule. The free list is allocated and left
// untouched for the same reason.
//
// THE VELOCITY IS STATE, AND THAT IS THE WHOLE POINT OF THIS RUNG. It looks
// derived -- the step kernel rewrites it from `f` every macro-step -- but
// `dispatchMacroStep` runs the REFINEMENT ROUND FIRST, reading "each level's
// own velocity field as populated by the PREVIOUS macro-step". So the first
// refine round after a load reads velocity the load must have put there. This
// is the same defect D1-a found in `resetSim` on the cylinder page, at a
// second site: a buffer that is rewritten every step is still state if
// something reads it before the first step.
//
// Before this, a load called `seedRootFromDense()`, and `amr_mirror_root.wgsl`
// writes `f_root` ONLY -- so the root's velocity survived a load untouched.
// `tools/lib/render-levels.js` reported it as "restore did not return to
// baseline" the moment it was given a level-0 row (U7-6b).
export function encodeRootCapture(device, U, enc, pools) {
  const root = pools[0];
  if (!root) return null;
  const velBytes = root.MAX_FINE_BLOCKS * root.cellsPerSlot * 2 * 4;
  const st = {
    f: device.createBuffer({ size: root.fSizePool, usage: U.MAP_READ | U.COPY_DST }),
    vel: device.createBuffer({ size: velBytes, usage: U.MAP_READ | U.COPY_DST }),
  };
  // Sized from the pool's own fields, never recomputed from a page's NCELLS1 --
  // the root tile is RINGLESS, so its cellsPerSlot is (2*RB)^2 where every
  // other level's is (2*RB + 2*GHOST)^2. U7-6a records what an over-long copy
  // costs: the encoder is invalidated and EVERY other copy in the same submit
  // is silently dropped, producing an all-zero snapshot that passes any gate
  // made of equalities.
  enc.copyBufferToBuffer(root.finePoolF_a, 0, st.f, 0, root.fSizePool);
  enc.copyBufferToBuffer(root.finePoolVel, 0, st.vel, 0, velBytes);
  return st;
}

// Call after the staging buffers are mapped. Unmaps and destroys its own.
export function readRootCapture(st, pools, { readF, bytesToB64 }) {
  if (!st) return null;
  const root = pools[0];
  const cells = root.MAX_FINE_BLOCKS * root.cellsPerSlot;
  const f = readF(st.f.getMappedRange(), cells);
  const vel = new Float32Array(st.vel.getMappedRange()).slice();
  for (const b of [st.f, st.vel]) { b.unmap(); b.destroy(); }
  const bytes = (a) => bytesToB64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  return {
    level: 0,
    MAX_FINE_BLOCKS: root.MAX_FINE_BLOCKS, NBLOCKS: root.NBLOCKS,
    NBX: root.NBX, NBY: root.NBY, cellsPerSlot: root.cellsPerSlot,
    fB64: bytes(f), velB64: bytes(vel),
  };
}

// Returns true if the root was restored FROM THE SNAPSHOT. False means the
// caller should fall back to seedRootFromDense() -- which is correct only for
// a capture written before this rung, and only for `f`.
export function restoreRootCapture(device, pools, snapRoot, { writeF, b64ToFloat32 }) {
  const root = pools[0];
  if (!snapRoot) return false;
  if (!root) {
    throw new Error('snapshot carries a root pool but this page has none (?rootpool=0) -- '
      + 'refusing rather than dropping level 0 on the floor');
  }
  if (snapRoot.MAX_FINE_BLOCKS !== root.MAX_FINE_BLOCKS
      || snapRoot.NBLOCKS !== root.NBLOCKS
      || snapRoot.cellsPerSlot !== root.cellsPerSlot) {
    throw new Error(`snapshot root (slots=${snapRoot.MAX_FINE_BLOCKS}, blocks=${snapRoot.NBLOCKS}, `
      + `cellsPerSlot=${snapRoot.cellsPerSlot}) does not match this page's `
      + `(slots=${root.MAX_FINE_BLOCKS}, blocks=${root.NBLOCKS}, cellsPerSlot=${root.cellsPerSlot})`);
  }
  const cells = root.MAX_FINE_BLOCKS * root.cellsPerSlot;
  writeF(root.finePoolF_a, b64ToFloat32(snapRoot.fB64, cells * 9), cells);
  const vel = b64ToFloat32(snapRoot.velB64, cells * 2);
  device.queue.writeBuffer(root.finePoolVel, 0, vel.buffer, vel.byteOffset, vel.byteLength);
  return true;
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
// --- THE REFINEMENT ROUND, ONCE (plans/uniform-levels.md U7-3) --------------
//
// Everything one `?refineEvery=` round encodes, in order. Measured 2026-09-18,
// comments and whitespace stripped: BYTE-IDENTICAL across the cylinder,
// reentry, TGV and channel pages -- all four -- with main-amr.js differing only
// by its measurement decoration and by the root having become a parent level
// (U5-4). 344 lines across the five pages of an ordering that is subtle in
// three separate places.
//
// ── THE ORDER, AND WHY EACH PART OF IT IS WHERE IT IS ──────────────────────
//
// CRITERION FIRST, ONCE. A block's own vorticity does not change because a
// neighbour got (de)activated this round, so re-evaluating it per iteration
// would be wasted work. Level 1's comes from the dense criterion (or, once the
// root manages it, from the pool criterion at parent level 0); every deeper
// level's comes from its own parent's pool criterion.
//
// THE WANT BUFFERS ARE CLEARED, NOT OVERWRITTEN. `decide()` at level >= 2 is
// dispatched over PARENT slots and never visits a block whose parent is
// inactive, so a stale want would survive there and resurrect a tile the
// criterion has stopped asking for.
//
// ONE SWEEP, NO FIXED POINT (plans/2D-backport.md B2): decide every level from
// its own reason only, close the want set under the 2:1 rule
// (shaders/amr_cascade.wgsl), coarsen, refine. Once. The closure is
// TRANSITIVE, so after it runs `want[m]` already contains everything
// `want[m+1]` will need a parent for; the loop this replaced iterated because
// its per-pass tests only ever saw one hop at a time.
//
// THE TWO ORDERS STILL MATTER, but for the ALLOCATOR, not for balance.
// Coarsen runs FINEST-FIRST because a child's slots must return to the free
// list before its parent's tile does. Refine runs COARSEST-FIRST because a
// level-(m+1) quad can only be carved from an ACTIVE level-m parent slot.
// Neither is about 2:1 any more.
//
// INIT FILL LAST, so anything refined this round gets its one-time full-slot
// fill before anything else this macro-step reads its pool slot.
//
// `firstParentLevel` is 0 once the root manages level 1, in which case the
// dense manager is not dispatched at all and the pool loops cover every level.
// It disappears at U7-5.
export function makeRefineRound({ nLevels, pools, cascade, encodeCascade, firstParentLevel = 1, passes }) {
  const useDenseManage = firstParentLevel > 0;
  return function encodeRefineRound(enc) {
    for (let m = 1; m < nLevels; m++) {
      // GPU-recorded, not queue.writeBuffer -- a JS-side write would not
      // interleave correctly with commands already recorded into this same
      // not-yet-submitted encoder (plans/AMR.md Milestone 4b).
      enc.clearBuffer(pools[m].newlyActivatedBuf);
    }

    passes.denseCriterion(enc);
    for (let m = firstParentLevel; m < nLevels - 1; m++) passes.poolCriterion(enc, m);

    for (let m = 1; m < nLevels; m++) enc.clearBuffer(pools[m].wantBuf);

    if (useDenseManage) passes.denseDecide(enc);
    for (let m = firstParentLevel; m < nLevels - 1; m++) passes.poolDecide(enc, m);

    encodeCascade(enc, cascade, nLevels);

    for (let m = nLevels - 1; m >= 1; m--) {
      if (m === 1 && useDenseManage) passes.denseCoarsen(enc);
      else passes.poolCoarsen(enc, m);
    }
    for (let m = 1; m < nLevels; m++) {
      if (m === 1 && useDenseManage) passes.denseRefine(enc);
      else passes.poolRefine(enc, m);
    }

    passes.l1InitFill(enc);
    for (let m = 2; m < nLevels; m++) passes.poolInitFill(enc, m);
  };
}

// --- THE MULTI-RATE SCHEDULER, ONCE (plans/uniform-levels.md U7-3) ----------
//
// `S_Advance`: AGAL's own recursive advance order
// (AGAL/src/solver_lbm/solver_lbm_advance.cu), traced precisely rather than
// re-derived from a one-line summary.
//
//   ROOT (level 0, no parent): interpolate INTO level 1 once from L0's CURRENT
//   state, L0's own ONE step, recurse into level 1 ONCE, average level 1 back
//   into L0 once. The root never does a "second substep" -- its own dt IS the
//   reference macro-step unit, so there is nothing to catch up to.
//
//   NON-ROOT (level >= 1, always has an implicit parent -- whoever called it):
//   interpolate INTO level+1 from THIS level's current state, this level's OWN
//   substep A, then -- if level+1 exists -- recurse into it ONCE, average it
//   back, and RE-interpolate into it so its next cycle sees fresh ghosts. Then,
//   under ?ghostcopy=1 only, this level's own same-level fine-fine refresh.
//   Then substep B, and again if level+1 exists, recurse a SECOND time and
//   average again.
//
// Every non-root level therefore does exactly 2 of its own substeps per call
// and drives its child through exactly 2 full cycles, which is what makes level
// L+k run 2^k times more often than L0 -- the correct refinement-ratio-2
// temporal scaling.
//
// "Current buffer" bookkeeping: L0 ping-pongs via the page's persistent `useB`,
// passed in. Every level >= 1 instead starts EVERY call at its own _a buffer
// and ends back at _a (substep A: a->b, substep B: b->a) -- a purely LOCAL,
// per-call invariant needing no persistent state. `cur` tracks it within a call.
//
// ── WHY THIS IS THE RUNG THE LADDER WAS BUILT FOR ──────────────────────────
//
// Measured 2026-09-18: this function was BYTE-IDENTICAL across the cylinder,
// reentry, TGV and channel pages, and main-amr.js's differed only by the root
// pool and its measurement instrumentation. Five copies of the subtlest
// ordering in the project, where a mistake is a physics bug rather than a
// crash -- a pass in the wrong place still runs, still produces a field, and
// still looks like a simulation.
//
// THE SEAM IS ORDER vs. CONTENT. This function owns WHEN each pass is encoded
// and the recursion that gets there. The `passes` object owns WHAT a pass is:
// which pipeline, which bind group, which dispatch size, and whatever
// profiling or ?benchSkip= decoration the page wants around it. The order was
// identical five times over; the content legitimately differs, because
// main-amr.js carries measurement twins the shipped pages do not.
export function makeScheduler({ nLevels, ghostCopy, passes }) {
  function S_Advance(level, enc, useB) {
    const hasChild = (level + 1) < nLevels;

    if (level === 0) {
      if (hasChild) passes.l0InterpIntoL1(enc, useB);
      passes.l0Step(enc, useB);
      if (hasChild) {
        S_Advance(1, enc, useB);
        passes.l1AverageIntoL0(enc, useB);
      }
      return;
    }

    let cur = 'a'; // THIS level's own current buffer, local to this call
    if (hasChild) passes.interpIntoChild(enc, level, cur);
    passes.substep(enc, level, cur);   // reads 'a', writes 'b'
    cur = 'b';
    if (hasChild) {
      S_Advance(level + 1, enc, useB);
      passes.averageFromChild(enc, level, cur); // child's cycle #1 lands in 'b'
      passes.interpIntoChild(enc, level, cur);  // fresh ghosts from the just-updated state
    }
    // Legacy same-level fine-fine refresh (?ghostcopy=1 only). The default path
    // needs no pass here: substep B's own gather reaches into the neighbour
    // tile directly, so it reads the neighbour's POST-average interior rather
    // than a copy taken before that average landed. See the DIRECT_GHOST
    // override in shaders/amr_step1.wgsl.
    if (ghostCopy()) passes.fineFineRefresh(enc, level);
    passes.substep(enc, level, cur);   // reads 'b', writes 'a'
    cur = 'a';
    if (hasChild) {
      S_Advance(level + 1, enc, useB);
      passes.averageFromChild(enc, level, cur); // child's cycle #2 lands in 'a'
    }
  }
  return { S_Advance };
}

// --- THE CRITERION/MANAGE BIND GROUPS, ONCE (plans/uniform-levels.md U7-2) --
//
// One pair per PARENT level, deciding the child level below it. Measured
// 2026-09-18: byte-identical across the cylinder, reentry, TGV and channel
// pages, with main-amr.js differing only by where the loop starts once the
// root became a parent level (U5-4).
//
// `firstParentLevel` is that difference and nothing else -- 0 when the root
// manages level 1, 1 otherwise. It is a CONFIG parameter, not a per-page one,
// and it disappears at U7-5 when `?rootmanage` collapses into the default.
//
// A DEAD LINE WENT WITH THE MOVE. Four of the five copies still computed
// `grandchildPool` for a grandchild cascade that B2-2d deleted -- legal, free
// at runtime, and read as though the loop still weighed it. Deleting a
// mechanism has to include deleting what fed it; amr_manage_pool.wgsl's own
// header records the identical lesson about `refineWants`.
export function makeManageBindGroups(device, layouts, pools, nLevels, { cardStateBuf, diagBuf, firstParentLevel = 1 }) {
  const criterionPoolBGs = {};
  const managePoolBGs = {};
  for (let m = firstParentLevel; m < nLevels - 1; m++) {
    const parentPool = pools[m];
    const childPool = pools[m + 1];
    // ALWAYS the parent pool's own finePoolVel, never a dense velBuf.
    // amr_criterion_pool.wgsl's binding 0 is the PARENT LEVEL's fine pool
    // velocity and it addresses that buffer BY POOL SLOT
    // (slot*(FB*FB) + fy*FB + fx), so a dense cellIndex-addressed buffer is
    // the wrong layout for every slot and, past roughly the first third of
    // them, reads off the end of a buffer less than half the size the pool
    // layout expects.
    //
    // That was a live bug once: the result was a level-2 blockCriterion of
    // essentially ZERO everywhere (measured 2^-39.86 for all 85 active L1
    // parents, against 2^-5.6 from a host reconstruction), so refine() saw
    // maxCrit ~= 0 for every parent and the vorticity criterion could never
    // promote a tile. Level 2 was 100% geometry-forced, which pinned the
    // L1/L2 boundary a few cells off the body so every shed vortex crossed it
    // at the trailing edge -- the reported block artifacts.
    criterionPoolBGs[m] = device.createBindGroup({ layout: layouts.criterionPoolBGL, entries: [
      { binding: 0, resource: { buffer: parentPool.finePoolVel } },
      { binding: 1, resource: { buffer: parentPool.slotToBlockBuf } },
      { binding: 2, resource: { buffer: childPool.blockCriterionBuf } },
      // Bound but never read at GHOST=2 -- a ringed parent needs no
      // neighbour resolution. The ROOT reads it (U4-1).
      { binding: 3, resource: { buffer: parentPool.blockSlotBuf } },
    ]});
    managePoolBGs[m] = device.createBindGroup({ layout: layouts.managePoolBGL, entries: [
      { binding: 0, resource: { buffer: childPool.blockCriterionBuf } },
      { binding: 1, resource: { buffer: childPool.blockSlotBuf } },
      { binding: 2, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 3, resource: { buffer: childPool.freeListBuf } },
      { binding: 4, resource: { buffer: childPool.freeCountBuf } },
      { binding: 5, resource: { buffer: childPool.newlyActivatedBuf } },
      { binding: 6, resource: { buffer: cardStateBuf } },
      { binding: 7, resource: { buffer: childPool.parentSlotBuf } },
      { binding: 8, resource: { buffer: childPool.wantBuf } },
      { binding: 9, resource: { buffer: diagBuf } },
      { binding: 12, resource: { buffer: parentPool.slotToBlockBuf } },
    ]});
  }
  return { criterionPoolBGs, managePoolBGs };
}

// --- THE PER-LEVEL BIND GROUPS, ONCE (plans/uniform-levels.md U7-2) ---------
//
// Every level >= 2's interp / step / average / force bind groups. Measured
// 2026-09-18, comments and whitespace stripped: this loop was BYTE-IDENTICAL
// across main-amr.js, main-cylinder-amr.js and main-reentry-amr.js, and the
// TGV and channel copies were the same loop minus the force block -- 298 lines
// across the five pages, two variants, one a strict subset of the other.
//
// The buffers it names all come from `allocLevelPool`, which was already
// shared; only the wiring was not.
export function makeLevelBindGroups(device, U, layouts, pools, nLevels, { cardStateBuf, forceBuf = null }) {
  for (let c = 2; c < nLevels; c++) {
    const parentPool = pools[c - 1];
    const childPool = pools[c];
    const interpEntries = (parentBuf) => [
      { binding: 0, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 1, resource: { buffer: parentBuf } },
      { binding: 2, resource: { buffer: childPool.finePoolF_a } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.newlyActivatedBuf } },
      { binding: 5, resource: { buffer: childPool.blockSlotBuf } },
      { binding: 6, resource: { buffer: childPool.parentSlotBuf } },
      { binding: 7, resource: { buffer: childPool.quadrantBuf } },
    ];
    childPool.interpPoolParentBG_readA = device.createBindGroup({ layout: layouts.interpPoolParentBGL, entries: interpEntries(parentPool.finePoolF_a) });
    childPool.interpPoolParentBG_readB = device.createBindGroup({ layout: layouts.interpPoolParentBGL, entries: interpEntries(parentPool.finePoolF_b) });
    // Fine-fine-only refresh always operates on THIS level's own _b (the
    // buffer its own substep-1 just wrote) -- binding 1 (f_parent_pool) is
    // unused in FINE_FINE_ONLY mode, bound to parent's _a only to satisfy
    // the shared layout (mirrors dense's interpFFBG_b's f_a-unused note).
    childPool.interpPoolParentFFBG_b = device.createBindGroup({ layout: layouts.interpPoolParentBGL, entries: interpEntries(parentPool.finePoolF_a).map((e, i) => i === 2 ? { binding: 2, resource: { buffer: childPool.finePoolF_b } } : e) });

    childPool.step1BG_ab = device.createBindGroup({ layout: layouts.step1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: childPool.finePoolF_b } },
      { binding: 3, resource: { buffer: childPool.finePoolVel } },
      { binding: 4, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 5, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 6, resource: { buffer: childPool.blockSlotBuf } },
    ]});
    childPool.step1BG_ba = device.createBindGroup({ layout: layouts.step1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_b } },
      { binding: 2, resource: { buffer: childPool.finePoolF_a } },
      { binding: 3, resource: { buffer: childPool.finePoolVel } },
      { binding: 4, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 5, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 6, resource: { buffer: childPool.blockSlotBuf } },
    ]});

    const avgEntries = (parentBuf) => [
      { binding: 0, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: parentBuf } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.parentSlotBuf } },
      { binding: 5, resource: { buffer: childPool.quadrantBuf } },
    ];
    childPool.avgPoolBG_targetA = device.createBindGroup({ layout: layouts.avgPoolBGL, entries: avgEntries(parentPool.finePoolF_a) });
    childPool.avgPoolBG_targetB = device.createBindGroup({ layout: layouts.avgPoolBGL, entries: avgEntries(parentPool.finePoolF_b) });

    // Milestone 8: level c's own force pass.
    //
    // BUILT ONLY WHERE THERE IS A FORCE TO INTEGRATE, and `forceBuf` is the
    // condition rather than a flag: the TGV and channel pages have no body, so
    // no force accumulator, so no force bind group. A page that HAS one always
    // wants this, so there is nothing to decide and no option to get wrong.
    //
    // debugSlotForceBuf is a TEMPORARY diagnostic (level-2 bounce-back sign
    // investigation) -- see amr_force1.wgsl's own debugSlotForce header.
    if (!forceBuf) continue;
    childPool.debugSlotForceBuf = device.createBuffer({ size: childPool.MAX_FINE_BLOCKS * 8, usage: U.STORAGE | U.COPY_SRC });
    childPool.force1BG = device.createBindGroup({ layout: layouts.force1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: forceBuf } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 5, resource: { buffer: childPool.debugSlotForceBuf } },
          // Bound but never read at GHOST=2 -- these levels have a ring.
      { binding: 6, resource: { buffer: childPool.blockSlotBuf } },
    ]});
  }
}

// --- THE COUPLING PIPELINES, ONCE (plans/uniform-levels.md U7-1) ------------
//
// The twelve pipelines every AMR page builds identically: the six interp
// variants (dense parent and pool parent, each in steady-state / init /
// fine-fine-only form), the two averages, the criterion, and manage's three
// entry points. Measured 2026-09-18, comments and whitespace stripped: eleven
// of the twelve were byte-identical across all five pages, and the twelfth
// (`avgPL`) differed only in the NAME of the constants object -- `fineConstants`
// on two pages, `avgConstants` on three, with the same five fields in both.
//
// NOT EVERY PIPELINE, AND THE LINE IS THE SCENARIO. The step, the force, the
// physics integrator and the render fragment stay per page, because their
// override sets are genuinely different things (measured: `step1Constants`
// differs on every single page -- bounce-back and a sponge on the cylinder, a
// card SDF on the dev page, neither on TGV). Those are SCENARIO overrides and
// they are data. WHICH PIPELINES EXIST, and which override selects which mode
// of a kernel, is not.
//
// THE THREE MODES BELONG TO THE KERNEL, NOT TO THE PAGE, which is why this
// takes five scalars rather than ten prebuilt bundles. GHOST_ONLY and
// FINE_FINE_ONLY name states of `common_interp_kernel.wgsl` -- steady-state
// ghost refresh, one-time full-slot fill on activation, and the between-substep
// fine-fine re-exchange. Five pages were each spelling that triple out, so
// five pages could each get it wrong.
//
// Returns the derived constant bundles alongside the pipelines. A page that
// needs a VARIANT -- the ?benchSkip= no-op twins, the SKIP_GHOST ring twin, the
// root-parent twins of U5 -- must build it from these rather than from its own
// second copy of the same literal, which is how a measurement twin drifts from
// the thing it is measuring.
export function makeCouplingPipelines(device, layouts, modules, { W, H, RB, F16, DC_PRE, manage }) {
  const c = {
    interp:         { W, H, RB, GHOST_ONLY: 1, F16, DC_PRE },
    interpInit:     { W, H, RB, GHOST_ONLY: 0, F16, DC_PRE },
    interpFF:       { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1, F16, DC_PRE },
    // No W/H on the pool-parent side: unlike the dense case, a level's own grid
    // extent is a runtime uniform (levelParams), not baked into the pipeline,
    // precisely so ONE compiled pipeline serves every L(m)->L(m+1) pair.
    interpPool:     { RB, GHOST_ONLY: 1, F16, DC_PRE },
    interpPoolInit: { RB, GHOST_ONLY: 0, F16, DC_PRE },
    interpPoolFF:   { RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1, F16, DC_PRE },
    avg:            { W, H, RB, F16, DC_PRE },
    avgPool:        { RB, F16, DC_PRE },
    // amr_criterion.wgsl declares only W/H -- passing an override a shader does
    // not declare is a pipeline-creation error, not a warning.
    criterion:      { W, H },
  };
  const compute = (layout, module, entryPoint, constants) => device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint, constants },
  });
  return {
    constants: c,
    interpPL:               compute(layouts.interpBGL, modules.interpDenseSM, 'main', c.interp),
    interpInitPL:           compute(layouts.interpBGL, modules.interpDenseSM, 'main', c.interpInit),
    interpFFPL:             compute(layouts.interpBGL, modules.interpDenseSM, 'main', c.interpFF),
    interpPoolParentPL:     compute(layouts.interpPoolParentBGL, modules.interpPoolSM, 'main', c.interpPool),
    interpPoolParentInitPL: compute(layouts.interpPoolParentBGL, modules.interpPoolSM, 'main', c.interpPoolInit),
    interpPoolParentFFPL:   compute(layouts.interpPoolParentBGL, modules.interpPoolSM, 'main', c.interpPoolFF),
    avgPL:                  compute(layouts.avgBGL, modules.avgSM, 'main', c.avg),
    avgPoolPL:              compute(layouts.avgPoolBGL, modules.avgPoolSM, 'main', c.avgPool),
    criterionPL:            compute(layouts.criterionBGL, modules.criterionSM, 'main', c.criterion),
    // manage's three entry points share one module and one constants bundle;
    // the bundle is the page's, because thresholds and geometry are scenario.
    manageDecidePL:         compute(layouts.manageBGL, modules.manageSM, 'decide', manage),
    manageCoarsenPL:        compute(layouts.manageBGL, modules.manageSM, 'coarsen', manage),
    manageRefinePL:         compute(layouts.manageBGL, modules.manageSM, 'refine', manage),
  };
}

// --- THE ROOT-POOL FLAGS, ONCE (plans/uniform-levels.md U7-4a) --------------
//
// Four `const`s and about sixty-five lines of design record. THE RECORD IS THE
// REASON THIS IS SHARED: copied into five pages it would be five places to
// update when a flag's meaning moves, and U7-5 moves all four of them at once.
//
// ?rootpool=0 -- keep the DENSE L0 and the per-block level-1 allocator (U1,
//   flipped to on by default at U7-5).
//
//   DEFAULT 1 SINCE U7-5. The root is a pool level like every other, level 1
//   is its quad child, and `amr_manage.wgsl` is not dispatched at all. The
//   flag survives the flip only as the ESCAPE -- `?rootpool=0` restores the
//   dense path for an A/B -- and goes with that path at U7-6.
//
//   It was default 0 through U1..U7-4 for a reason worth keeping in view: with
//   the flag off nothing was allocated at all, so an unused root pool could
//   not cost the phone ~21 MB for a buffer no kernel read. That is no longer
//   hypothetical relief, it is a measured cost -- U7-5a puts the flip at about
//   +2 MB on the shipped configuration, because the pool sizes move with the
//   demand and the demand rose at level 1 only.
//
// ?rootstep=0 -- allocate and mirror the root pool but do NOT step it (U3).
//
//   The CONTROL for U3's gate, and it is not optional. U3 asks "does the fine
//   kernel on the root agree with the dense kernel, bit for bit, from
//   identical input", and the answer is read as a CLEAN comparison after
//   stepping. But a clean comparison is also what a broken COMPARISON
//   produces, so the gate needs a configuration where the same procedure must
//   come back DIRTY. That is this one: mirror, step the dense grid, and leave
//   the root untouched, so the pool is provably stale and the checker has to
//   say so.
//
// ?rootcouple=0 -- allocate, mirror and step the root pool, but leave level 1
//   coupled to the DENSE grid (U5-3).
//
//   Default 1 whenever the root pool exists: with `?rootpool=1`, level 1's
//   ghost ring is interpolated from the root pool and its restriction is
//   written to the root pool. The dense grid keeps its own step and its own
//   restriction, so the two L0 representations stay byte-identical and every
//   remaining dense consumer is untouched.
//
//   This flag exists to separate "the root pool is allocated" from "the solver
//   uses it", the same split ?rootstep= made for U3. `?rootpool=1&rootcouple=0`
//   is U5-2's configuration exactly, which is what makes the coupling A/B-able
//   in one build rather than across two.
//
// ?rootmanage=0 -- couple level 1 to the root but keep managing it with the
//   DENSE manager (U5-4).
//
//   Default 1 whenever the coupling is on: amr_manage_pool.wgsl decides and
//   allocates level 1, with the root as the parent level, and amr_manage.wgsl
//   is not dispatched at all.
//
//   THIS ONE MOVES PUBLISHED NUMBERS AND IT IS MEANT TO. The pool manager
//   allocates in QUADS and decides over the PARENT's footprint, so level 1's
//   refinement granularity goes from one 8-cell block to a 16-cell quad: more
//   tiles, a differently-shaped refined region, and different pool demand.
//   That is a larger move than the plan's "Cd in the 4th digit", which only
//   anticipated slot regrouping -- see U5-4.
//
//   AND THE QUAD IS A CONVENTION, NOT SOMETHING THE ADDRESSING COMPELS. Under
//   a root parent common_interp_parent_pool.wgsl derives a child's parent slot
//   and quadrant from its own BLOCK COORDINATES (the root is always full), not
//   from its slot index, so a level-1 child need not sit in an aligned group
//   of four for the coupling to reach it. What the grouping buys is uniformity
//   with every deeper level -- and one storage binding, since `quadrant ==
//   slot % 4` only holds under it (B2-2b0). See plans/uniform-levels.md, "The
//   quad granularity: what it is, and what actually forces it".
//
// `managed` is derived HERE, once, because it has to agree with the allocator,
// the reset, the cascade and the dispatch, and a condition recomputed in four
// places is how those drift.
//
// THE THREE STAGING FLAGS SURVIVED U7-5, AND THE GATES ARE WHY. The plan had
// this rung "collapse `rootcouple` and `rootmanage` into the default or drop
// them", and measured against what actually reads them that is wrong for two:
// ALL FOUR of tools/validate-root-kernels.js's CONTROL rows are
// `?rootstep=0` / `?rootcouple=0`, and those controls are what make its eleven
// gated rows mean anything. Deleting an instrument to satisfy a plan line is
// how this project collected vacuous gates; they go at U7-6, with the dense
// path they compare against.
//
// `rootmanage` has no tool reading it and could have gone. It stays because
// U7-5a made its question live: quad-vs-per-block level 1 is a CONVENTION
// (see the flag note above), and `?rootmanage=0` is the only handle on
// measuring whether the convention is the better one.
//
// `readRootFlags(urlParams, { staging: false })` is still the shipped pages'
// call: `?rootpool=` selects, the other three are pinned on. A page that never
// A/B's a staging rung has no reason to read them off its URL.
export function readRootFlags(urlParams, { staging = true } = {}) {
  const flag = (name, dflt) =>
    urlParams.has(name) ? (parseInt(urlParams.get(name)) ? 1 : 0) : dflt;
  const pool = flag('rootpool', 1);
  const step   = staging ? flag('rootstep', 1)   : 1;
  const couple = staging ? flag('rootcouple', 1) : 1;
  const manage = staging ? flag('rootmanage', 1) : 1;
  return {
    pool, step, couple, manage,
    stepped: !!(pool && step),
    coupled: !!(pool && couple),
    managed: !!(pool && couple && manage),
  };
}

// The root, as a level. `rootPoolSpec` is the shape; `allocLevelPool` is the
// same allocator every other level uses, which is U1's whole claim in one
// line. No initial field: a buffer nothing reads should not be given a state
// that could be mistaken for one -- `seedRootFromDense` writes it.
export function allocRootPool(device, U, { W, H, RB }) {
  const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
  return allocLevelPool(device, U, 0, spec.nbx, spec.nby, spec.slots, spec.cellsPerSlot);
}

// --- THE ROOT POOL'S SOLVER HALF, ONCE (plans/uniform-levels.md U7-4a) ------
//
// U1..U5 built the root pool on main-amr.js alone, and U7-3's seam is why it
// stayed there: the shared scheduler owns the ORDER and a `passes` object owns
// the CONTENT, and the root's step and the root's restriction ARE content. So
// four pages had no way to encode them however much of the scheduler they
// shared, and U7-4's "everything else is already shared" did not hold.
//
// Measured 2026-09-18, the construction block in main-amr.js was 372 lines and
// it split cleanly:
//
//   SOLVER      ~200   the mirror, the root step, U5-3's live coupling, and
//                      the criterion redirect. Needed by every page. HERE.
//   INSTRUMENT  ~170   the root force pass, the full digest, and U5-1/U5-2's
//                      inert scratch legs. Dev page only.
//
// plus ~280 lines of debugCheckRoot* comparators, which stay on the dev page
// for the reason B3-5 records in the other direction: five copies of a checker
// is how this project collected its vacuous gates.
//
// THE INTERLEAVING WAS THE WORK, not the move. The two halves sat inside one
// `if (ROOT_POOL)` sharing locals -- the `unread` sentinel and `rootAvgPL`,
// both declared in an inert block and both read by the live path -- so the
// extraction had to untangle before it could lift. Both are returned, so the
// dev page's instrument legs bind the same objects rather than building a
// second copy that could drift from the thing they are scoring.
//
// THE INERT CRITERION TWIN STAYED BEHIND, and that is a correction to U7-4's
// own itemisation. Under `managed` the live level-1 criterion is
// `criterionPoolPLs[0]` -- built by the page's existing per-parent-level loop
// once its bound starts at 0 -- so U4-1's `rootCritPL` is never the live
// writer in any configuration. What the solver needs from that stage is the
// REDIRECT: `amr_criterion.wgsl` still gets dispatched by the shared refine
// round, and under quad management it must not land on level 1's real
// criterion buffer. `denseCritBuf` is that target, and it is the only piece of
// U4-1 here.
//
// `beginPass` is the one decoration this takes, defaulting to a plain
// beginComputePass. The dev page passes its profiling wrapper so `root step`
// and `L1->root average` keep their labels; a shipped page wants neither.
export function makeRootPool(device, U, layouts, modules, pools, {
  W, H, RB, F16, DC_PRE, step1Constants, couplingConstants,
  cardStateBuf, denseFBuf, flags,
  beginPass = (enc) => enc.beginComputePass(),
}) {
  const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
  const root = pools[0], l1 = pools[1];

  // U2's mirror: dense L0 -> the root pool, so the pool's addressing can be
  // scored against the live dense buffer before any kernel depends on it --
  // and, since 1.2's fingerprint gate caught an unseeded root driving
  // refinement, the SEEDER as well. One dispatch at init, at reset and after a
  // snapshot load, never per frame. At U7-6 the root is the only L0 and takes
  // `initF()` directly, and this goes with the dense path.
  const mirrorBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]});
  const mirrorRootPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [mirrorBGL] }),
    compute: { module: modules.mirrorRootSM, entryPoint: 'main', constants: { W, H, RB, F16 } },
  });
  const mirrorRootBG = device.createBindGroup({ layout: mirrorBGL, entries: [
    { binding: 0, resource: { buffer: denseFBuf } },
    { binding: 1, resource: { buffer: root.finePoolF_a } },
    { binding: 2, resource: { buffer: root.slotToBlockBuf } },
  ]});
  const seedRootFromDense = () => {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(mirrorRootPL);
    p.setBindGroup(0, mirrorRootBG);
    p.dispatchWorkgroups(spec.side / 8, spec.side / 8, spec.slots);
    p.end();
    device.queue.submit([enc.finish()]);
    return device.queue.onSubmittedWorkDone();
  };

  // U3: the fine step kernel, serving the ROOT.
  //
  // The SAME module as every other level -- amr_step1.wgsl -- with GHOST 0 and
  // OWN_TAU 1. Everything else is the page's own step1 constants, so a
  // scenario override the fine levels get, the root gets too, by construction
  // rather than by a list somebody has to keep in step.
  //
  // DIRECT_GHOST IS PINNED, and it is the one constant the root may not
  // inherit. `step1Constants` carries `DIRECT_GHOST: GHOST_COPY ? 0 : 1`, so
  // `?ghostcopy=1` would hand the root the legacy path -- clamp at the slot's
  // own buffer edge and read a ghost cell a separate fine-fine copy pass
  // filled. The root has GHOST = 0, so there is no ring to clamp into and no
  // copy pass that fills one; it would stream from its own edge cells. The
  // root is also ALWAYS FULL, so the direct path never falls back
  // (amr2d.mjs's ghostDepthAtLevel(0) and rootPoolSpec's `slots === nblocks`)
  // and the legacy path has nothing to offer it.
  //
  // MEASURED, not argued: inherited, `?ghostcopy=1&benchSkip=avg` read
  // 580623/589824 words differing from the dense grid at 512 macro-steps --
  // indistinguishable from not stepping the root at all -- while every other
  // rung was bit-identical. See plans/uniform-levels.md U3.
  //
  // AND IT IS REBUILDABLE, because one page's step constants are not fixed for
  // the session: main-channel-amr.js bakes Re into FORCE_X/WALL_U1 and
  // recreates its step pipelines on every `setRe`. A root left on the pipeline
  // built at init would then be driving a different flow from the dense grid
  // it is supposed to be a second copy of -- silently, since both still step.
  // `rebuildStep` is how that page keeps them one scenario; the bind groups do
  // not depend on the constants, so only the pipeline is rebuilt.
  let rootStepPL = null, rootStepBG_ab = null, rootStepBG_ba = null, rootStepWG = 0;
  const buildRootStepPL = (c) => device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.step1BGL] }),
    compute: { module: modules.step1SM, entryPoint: 'main',
               constants: { ...c, DIRECT_GHOST: 1, GHOST: 0, NO_PARENT: 1, SPONGE_CELL_SNAP: 1 } },
  });
  if (flags.stepped) {
    rootStepPL = buildRootStepPL(step1Constants);
    const rootBG = (fin, fout) => device.createBindGroup({ layout: layouts.step1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: fin } },
      { binding: 2, resource: { buffer: fout } },
      { binding: 3, resource: { buffer: root.finePoolVel } },
      { binding: 4, resource: { buffer: root.slotToBlockBuf } },
      { binding: 5, resource: { buffer: root.levelParamsBuf } },
      { binding: 6, resource: { buffer: root.blockSlotBuf } },
    ]});
    rootStepBG_ab = rootBG(root.finePoolF_a, root.finePoolF_b);
    rootStepBG_ba = rootBG(root.finePoolF_b, root.finePoolF_a);
    rootStepWG = Math.ceil((RB * 2) / 8);
  }

  // U4-1's REDIRECT. Under quad management the pool criterion at parent level
  // 0 is the live writer of level 1's blockCriterion, but the shared refine
  // round still encodes `denseCriterion` -- so amr_criterion.wgsl needs a
  // target that is not the buffer it would otherwise clobber. This is it.
  // Unused (and unread) when the root does not manage level 1.
  const denseCritBuf = device.createBuffer({
    size: l1.NBLOCKS * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
  });

  // The two per-slot fields the root path derives instead of reading. They
  // cannot be left unbound (WGSL module scope has no conditional bindings), so
  // they are bound to ONE buffer of 0xFFFFFFFF, which reads as -1 through the
  // array<i32> binding and as an out-of-range slot through the array<u32> one.
  // A sentinel rather than a zero-filled buffer on purpose: slot 0 and
  // quadrant 0 are both real values, and a binding that returns a plausible
  // number when it should be unreachable is exactly the failure
  // plans/2D-backport.md B6-9c is about.
  const unread = device.createBuffer({
    size: l1.MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST,
  });
  device.queue.writeBuffer(unread, 0, new Uint32Array(l1.MAX_FINE_BLOCKS).fill(0xFFFFFFFF));

  // -- U5-3: THE COUPLING, LIVE -------------------------------------------
  //
  // Level 1's ghost ring is interpolated from the root pool, and level 1's
  // restriction is written to the root pool. The same two modules every
  // L(m)->L(m+1) hop with m>=1 uses, with PARENT_GHOST 0 -- the root has no
  // ring, so the interp's bilinear stencil resolves its out-of-parent fetches
  // against the neighbouring ROOT TILE instead (see
  // shaders/common_interp_parent_pool.wgsl). Restriction needs none of that:
  // it writes one parent cell per child cell, always inside the parent's own
  // interior, so the ring-free root costs that direction only the offset and
  // the stride.
  //
  // THE DENSE GRID KEEPS ITS OWN STEP AND ITS OWN RESTRICTION, and that is the
  // whole staging device. Both L0 representations are stepped (U3) and both
  // receive level 1's restriction, so they stay BYTE-IDENTICAL rather than
  // only until the first `average`. Every remaining dense consumer (the
  // renderer, the criterion, the force, the digest, `debugSnapshotSave`, the
  // whole host and tool tail) therefore needs no change at this stage and
  // cannot be broken by it; they are flipped one at a time afterwards, each
  // against a buffer already proven equal, and the dense writers go at U7-6.
  //
  // ONLY ONE THING IS REPLACED RATHER THAN DUPLICATED: the dense-parent
  // interp. It cannot be run alongside, because both would write the same
  // ghost cells of the same pool -- and U5-1's whole result is that they write
  // the same words, so there would be nothing to gain from a race.
  //
  // Phase: the root pool ping-pongs in lockstep with L0's own `useB` (U3), so
  // "the current buffer" is the same name on both sides. interp reads the
  // current one BEFORE the step; average targets the one the step just wrote.
  const interpPipe = (constants) => device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.interpPoolParentBGL] }),
    compute: { module: modules.interpPoolSM, entryPoint: 'main', constants },
  });
  const rootInterpLivePL = interpPipe({ ...couplingConstants.interpPool, PARENT_GHOST: 0 });
  const rootInterpInitPL = interpPipe({ ...couplingConstants.interpPoolInit, PARENT_GHOST: 0 });
  // FINE_FINE_ONLY never reaches the parent at all (it returns before the
  // parent hop), so this pipeline is behaviourally identical to the dense one.
  // It exists anyway so that ?ghostcopy=1 does not leave a dense-parent
  // pipeline in the live path -- U3 paid once for a root pipeline inheriting a
  // constant that did not belong to it.
  const rootInterpFFPL   = interpPipe({ ...couplingConstants.interpPoolFF, PARENT_GHOST: 0 });
  const rootInterpNoopPL = interpPipe({ ...couplingConstants.interpPool, PARENT_GHOST: 0, NOOP: 1 });

  const liveInterpBG = (parentF, childF) => device.createBindGroup({ layout: layouts.interpPoolParentBGL, entries: [
    { binding: 0, resource: { buffer: l1.levelParamsBuf } },
    { binding: 1, resource: { buffer: parentF } },
    { binding: 2, resource: { buffer: childF } },
    { binding: 3, resource: { buffer: l1.slotToBlockBuf } },
    { binding: 4, resource: { buffer: l1.newlyActivatedBuf } },
    { binding: 5, resource: { buffer: l1.blockSlotBuf } },
    { binding: 6, resource: { buffer: unread } },
    { binding: 7, resource: { buffer: unread } },
  ]});
  const rootInterpLiveBG_readA = liveInterpBG(root.finePoolF_a, l1.finePoolF_a);
  const rootInterpLiveBG_readB = liveInterpBG(root.finePoolF_b, l1.finePoolF_a);
  // The between-substep fine-fine refresh targets level 1's own _b, mirroring
  // the dense interpFFBG_b. Its parent binding is never read (see above).
  const rootInterpLiveFFBG_b = liveInterpBG(root.finePoolF_a, l1.finePoolF_b);

  const rootAvgPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.avgPoolBGL] }),
    compute: { module: modules.avgPoolSM, entryPoint: 'main',
               constants: { RB, F16, DC_PRE, PARENT_GHOST: 0 } },
  });
  const liveAvgBG = (parentF) => device.createBindGroup({ layout: layouts.avgPoolBGL, entries: [
    { binding: 0, resource: { buffer: l1.levelParamsBuf } },
    { binding: 1, resource: { buffer: l1.finePoolF_a } },
    { binding: 2, resource: { buffer: parentF } },
    { binding: 3, resource: { buffer: l1.slotToBlockBuf } },
    { binding: 4, resource: { buffer: unread } },
    { binding: 5, resource: { buffer: unread } },
  ]});
  const rootAvgLiveBG_targetA = liveAvgBG(root.finePoolF_a);
  const rootAvgLiveBG_targetB = liveAvgBG(root.finePoolF_b);

  return {
    spec, seedRootFromDense, denseCritBuf, unread,
    // See buildRootStepPL: for a page whose step constants move with a live
    // parameter, this is how the root stays the same scenario as the dense L0.
    rebuildStep: (nextStep1Constants) => {
      if (rootStepPL) rootStepPL = buildRootStepPL(nextStep1Constants);
    },
    mirrorRootPL, mirrorRootBG, rootAvgPL,
    rootStepPL, rootStepBG_ab, rootStepBG_ba, rootStepWG,
    rootInterpLivePL, rootInterpInitPL, rootInterpFFPL, rootInterpNoopPL,
    rootInterpLiveBG_readA, rootInterpLiveBG_readB, rootInterpLiveFFBG_b,
    rootAvgLiveBG_targetA, rootAvgLiveBG_targetB,

    // U3: the dense L0 step's twin, on the root pool, encoded in parallel with
    // it. Shares L0's own `useB` ping-pong so the two stay in phase -- which
    // is what lets debugCheckRootMirror compare _a against f_a at rest. A
    // no-op under `?rootstep=0`, which is the control that must come back
    // DIRTY.
    encodeRootStep: (enc, useB) => {
      if (!rootStepPL) return;
      const p = beginPass(enc, 'root step');
      p.setPipeline(rootStepPL);
      p.setBindGroup(0, useB ? rootStepBG_ba : rootStepBG_ab);
      p.dispatchWorkgroups(rootStepWG, rootStepWG, root.MAX_FINE_BLOCKS);
      p.end();
    },
    // U5-3: level 1's restriction into the root pool. NOT an alternative to
    // the dense average -- BOTH run, which is what keeps the two L0
    // representations byte-identical while the remaining dense consumers are
    // flipped over one at a time. The caller encodes it inside its own `avg`
    // skip group, so ?benchSkip=avg still isolates the step as the sole writer
    // of either representation -- which validate-root-kernels.js relies on.
    encodeRootAverage: (enc, useB) => {
      if (!flags.coupled) return;
      const p = beginPass(enc, 'L1->root average');
      p.setPipeline(rootAvgPL);
      p.setBindGroup(0, useB ? rootAvgLiveBG_targetA : rootAvgLiveBG_targetB);
      p.dispatchWorkgroups(1, 1, l1.MAX_FINE_BLOCKS);
      p.end();
    },

    // -- WHICH PARENT LEVEL 1 IS COUPLED TO, CHOSEN ONCE -------------------
    //
    // Five dispatch sites want level 1's interp (the macro-step, the
    // between-substep fine-fine refresh, the post-refine init fill,
    // debugActivateBlock, and the benchSkip no-op twin), and a sixth wants its
    // restriction. Selecting the parent at each of them is how a page ends up
    // coupled two different ways depending on the path taken -- the shape
    // CLAUDE.md records for the bind-group mirroring trap, one level down. So
    // the choice is made here, once, and every site reads these names without
    // knowing which parent it got.
    //
    // The caller hands in its own dense-parent bundle and gets one back; with
    // no coupling the bundle it handed in comes straight back out, so the
    // fallback is structural rather than a second ternary per site.
    coupleL1: (dense) => flags.coupled ? {
      interpPL:     rootInterpLivePL,
      interpInitPL: rootInterpInitPL,
      interpFFPL:   rootInterpFFPL,
      interpNoopPL: rootInterpNoopPL,
      interpBG:     (b) => b ? rootInterpLiveBG_readB : rootInterpLiveBG_readA,
      interpInitBG: (b) => b ? rootInterpLiveBG_readB : rootInterpLiveBG_readA,
      interpFFBG:   rootInterpLiveFFBG_b,
    } : dense,
  };
}

// --- THE BIND GROUP LAYOUTS, ONCE (plans/uniform-levels.md U7-0) ------------
//
// Fourteen layouts, and before this they were spelled out inline in FIVE
// PAGES. Measured 2026-09-18 with comments and whitespace stripped: every one
// of the fourteen was BYTE-IDENTICAL across every page that had it -- 698
// lines of them in total, of which ~520 were duplication.
//
// THIS IS THE FUNCTION THAT KILLS THE BINDING-MIRROR TRAP, which is worth more
// than the line count. CLAUDE.md records it producing 238e48c; U4-1 and U4-2
// walked into it again in the other direction and stopped four pages booting
// with `Binding doesn't exist in [BindGroupLayoutInternal "force1BGL"]`; U5-4
// walked it a third time deliberately, with the boot smoke as the net; U6
// found that three of five pages had ALREADY drifted on the render pipeline's
// override. A binding added to a shared shader now has exactly one layout to
// reach.
//
// THE ONE THING TO WATCH: these are identical TODAY. If this function ever
// grows per-page parameters it becomes five copies with extra steps. A page
// that genuinely needs a different layout is a page that should not be sharing
// the SHADER either -- have that conversation rather than adding a flag.
//
// Every page gets all fourteen, including the three that only three pages use
// (frcBGL, phyBGL, force1BGL -- the two bodyless pages have no force pass).
// A GPUBindGroupLayout nobody binds costs nothing, and selecting a subset per
// page would reintroduce exactly the per-page variation this removes.
export function makeAMRLayouts(device) {
  const stepBGL = device.createBindGroupLayout({ label: 'stepBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  // Milestone 8: binding 3 (blockSlot1) is the finest-wins masking check --
  // see amr_force.wgsl's header.
  const frcBGL = device.createBindGroupLayout({ label: 'frcBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const phyBGL = device.createBindGroupLayout({ label: 'phyBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  // Milestone 10: bindings 5/6 (level 2's own vel_pool/blockSlot) are for
  // finest-active-level-wins compositing -- harmless dummies when
  // N_LEVELS<3, see amr_render.wgsl's header.
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    // U6: levels 3 and 4. One velocity/indirection pair per pool level, up to
    // MAX_RENDER_POOL_LEVELS -- see shaders/amr_render.wgsl, which walks them.
    // Bound unconditionally; N_POOL_LEVELS is what stops the walk, because a
    // dummy blockSlot read out of bounds is not safe on every WebGPU stack.
    { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 10, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    // U7-6b: the ROOT pool's indirection, so level 0 can be drawn from the
    // pool rather than the dense grid. Read only when the render pipeline's
    // ROOT_IS_POOL override is set; see renderRootIsPool.
    { binding: 12, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

  // Milestone 4: interp (coarse->fine ghosts), fine step, average (fine->coarse),
  // all pool-aware (an extra read-only slotToBlock/blockSlot binding vs. M2).
  // Binding 4 (newlyActivated) is Milestone 4b: only read by the GHOST_ONLY=0
  // init pipeline, but must still be present in the layout both pipelines share.
  // Milestone 4c: binding 5 (blockSlot) added so a ghost cell can check
  // whether its edge-adjacent neighbor block is also currently refined (see
  // amr_interp_c2f.wgsl's file header on fine-fine ghost consultation).
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 6: L(m)->L(m+1) (m>=1) ghost interpolation, shared by every
  // pool-to-pool level pair (decision 2 -- one pipeline, many levels, only
  // the bind group's buffers/uniform differ). Binding 0 is a small per-
  // child-level uniform (this level's own NBX/NBY + its parent's tau --
  // see shaders/amr_interp_pool_parent.wgsl's LevelParams), not the whole
  // CardState struct the dense layout uses -- a parent mid-chain doesn't
  // have a single domain-wide tau to read off CardState the way L0 does.
  // Bindings 6/7 (parentSlot/quadrant) are the only structurally new
  // per-slot fields vs. interpBGL, both this level's own.
  const interpPoolParentBGL = device.createBindGroupLayout({ label: 'interpPoolParentBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 4b: criterion (per-block vorticity max) and manage (refine/coarsen decision).
  const criterionBGL = device.createBindGroupLayout({ label: 'criterionBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const manageBGL = device.createBindGroupLayout({ label: 'manageBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // Milestone 4c: geometry-forced refinement needs the card's pose/velocity.
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    // bindings 7/8 were level 2's blockCriterion/blockSlot, for the per-pass
    // cascade. Gone with it (B2-2d) -- the closure needs no cross-level read
    // here. Holes, not renumbered.
    // binding 9: ?diag=1 convergence counters. Always bound; never touched at DIAG=0.
    // binding 7: D0's candidate rank -- one of B2-2d's two holes, reclaimed
    // rather than renumbering. Always bound; only written when ?detslots=1.
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // binding 10: level 1's WANT array (B2). Always bound; only read when
    // CASCADE != 0, and only written by the decide() entry point.
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  // Milestone 9: per-quadrant criterion for any level-(m+1) decision,
  // parent=level m -- see amr_criterion_pool.wgsl's header (one pipeline
  // per parent level, not shared, unlike the M6-M8 pool-parent shaders).
  const criterionPoolBGL = device.createBindGroupLayout({ label: 'criterionPoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // binding 3: the PARENT level's blockSlot, for U4-1's ring-free stencil.
    // Always bound, read only when that pipeline's GHOST is 0. Four bindings,
    // against a 16-per-stage ceiling -- see CLAUDE.md before adding a fifth.
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 9: quad allocator + 2:1 balance for any level-(m+1) decision,
  // parent=level m>=1 -- see amr_manage_pool.wgsl's header. 16 bindings
  // (15 original + grandchildBlockSlot, added for the N>=4 2:1-balance
  // cascade fix -- both refine() and coarsen() only ever need EXISTENCE,
  // never level (m+2)'s criterion, so one shared buffer/layout covers
  // both) -- exactly this adapter's real maxStorageBuffersPerShaderStage,
  // not just the WebGPU spec minimum other buffer limits in this file hit.
  const managePoolBGL = device.createBindGroupLayout({ label: 'managePoolBGL', entries: [
    { binding: 0,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 6,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // binding 8 was childQuadrant (it held `slot % 4`); B2 is what that
    // recovery was for -- this is the child level's WANT array.
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // binding 9 is the shared ?diag=1 counter buffer, same numbering and same
    // slot as amr_manage.wgsl's (plans/uniform-levels.md U5-4). Bound on every
    // page: the pool manager's refusal counter is not a level-1 concern, it is
    // every level's, and a binding added to a shared shader has to reach all
    // five layouts (CLAUDE.md's own recorded trap).
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // bindings 9/10 were childOriginX/Y and 13/14 parentOriginX/Y. All four
    // gone (B3-5): the origin is `block * RB * 2^-(m-1)` in closed form, so
    // the kernel derives it -- see amr_manage_pool.wgsl's parentOriginL0.
    // binding 11 was parentBlockSlot, for the neighbour-active veto.
    // Gone with it (B2-2d).
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    // binding 15 was grandchildBlockSlot, for hasGrandchild.
    // Gone with it (B2-2d).
  ]});
  const avgBGL = device.createBindGroupLayout({ label: 'avgBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // THE fine step, one layout for every pool level since B3-1 (there was a
  // second, level-1-only step1BGL here until then). Binding 5 is the
  // per-level uniform -- parentTau, dxL and nbx/nby, all four read by the
  // kernel: dxL and nbx/nby are what place the tile, now that its origin is
  // derived rather than loaded per slot. Shared verbatim with
  // interpPoolParentBGL/avgPoolBGL.
  const step1BGL = device.createBindGroupLayout({ label: 'step1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    // binding 6: blockSlot -- neighbour-addressed streaming (see the
    // DIRECT_GHOST override in shaders/amr_step1.wgsl). Present in the layout
    // even under ?ghostcopy=1, where the shader simply never reads it.
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 7: level>=2 average, writing into a parent POOL tile via
  // parentSlot/quadrant instead of cellIndex(). Since B3-2 that difference
  // IS the whole difference: both average entry files are their bindings
  // plus shaders/common_average.wgsl, with the destination and the parent's
  // tau behind common_avg_parent_{dense,pool}.wgsl.
  const avgPoolBGL = device.createBindGroupLayout({ label: 'avgPoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 8: level 1's own force pass. Binding 4 (childBlockSlot) is
  // level 2's blockSlot when HAS_CHILD=1, or a harmless dummy buffer when
  // HAS_CHILD=0 (N_LEVELS==2) -- see amr_force1.wgsl's header.
  // Milestone 8: level>=2's own force pass, one pipeline shared across every
  // such level (hasChild is a runtime LevelParams field here, not a
  // compile-time override -- see amr_force1.wgsl's header).
  // THE force pass, one layout for every pool level since B3-4 (there was a
  // second, level-1-only force1BGL here until then), and renumbered
  // contiguous now that the origin buffers and the masking's childBlockSlot
  // are both gone -- see shaders/amr_force1.wgsl's binding block.
  const force1BGL = device.createBindGroupLayout({ label: 'force1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // binding 6: THIS level's blockSlot, for U4-2's ring-free gather. Always
    // bound, read only when that pipeline's GHOST is 0. Seven bindings against
    // a 16-per-stage ceiling -- see CLAUDE.md before adding more.
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  return { stepBGL, frcBGL, phyBGL, renBGL, interpBGL, interpPoolParentBGL, avgBGL, avgPoolBGL, criterionBGL, criterionPoolBGL, manageBGL, managePoolBGL, step1BGL, force1BGL };
}

// --- the renderer's per-level wiring (plans/uniform-levels.md U6) -----------
//
// `shaders/amr_render.wgsl` binds one velocity/indirection pair per POOL
// level and walks them finest-first. WGSL cannot index an array of storage
// buffers, so the pairs are separate bindings and this table is the one place
// that says which binding belongs to which level.
//
// IT LIVES HERE BECAUSE FIVE PAGES BUILD THE SAME BIND GROUP. Before U6 they
// each spelled out a fixed two-tier `renBG` inline, and three of the five
// never passed the `HAS_LEVEL2` override at all -- so level 2 was solved and
// never drawn on the cylinder, TGV and channel pages, and level 3 was never
// drawn anywhere. A table copied five times is a table that will be updated
// three times.
export const MAX_RENDER_POOL_LEVELS = 4;
const RENDER_LEVEL_BINDINGS = { 1: [2, 3], 2: [5, 6], 3: [8, 9], 4: [10, 11] };

// A level the configuration does not have is bound to LEVEL 1's buffers, not
// to a dummy. The walk stops at `N_POOL_LEVELS` and never reads them, so the
// contents do not matter -- but a one-element dummy would matter: an
// out-of-range `blockSlot` read is not guaranteed to return the in-bounds -1
// (Dawn clamps to element 0, other stacks need not), and a positive slot there
// would falsely activate a level. That hazard is exactly what the old
// `HAS_LEVEL2` override existed to gate, and the loop bound now carries it.
// U7-6b: IS LEVEL 0 DRAWN FROM THE ROOT POOL? One rule, stated once, because
// TWO SITES MUST AGREE ABOUT IT AND THEY ARE BUILT AT DIFFERENT TIMES -- the
// render PIPELINE takes it as the ROOT_IS_POOL override (constructed early,
// with the other constants) and the BIND GROUP picks binding 0's buffer from
// it (constructed later, once the pools exist). Binding 0 carries a dense L0
// grid or a root pool depending on this, and the two addressings are not
// interchangeable: disagreement is not a validation error, it is a picture
// drawn from the wrong index arithmetic.
//
// Derived from the pools rather than from the page's `ROOT_POOL` flag, so the
// authority is the thing that actually got allocated.
//
// `?rootIsPool=0|1` OVERRIDES IT, AND THAT IS THIS RUNG'S INSTRUMENT. With a
// root pool allocated, the dense L0 grid and the root pool hold the SAME field
// -- U5-3 keeps them byte-identical, which is the whole reason the dense grid
// is still stepped. So the two addressings can be pointed at one state in ONE
// BUILD and required to draw the SAME PICTURE. That is a far better gate for
// U7-6b than comparing a hash across commits, where the page's own
// run-to-run residue is the same size as the thing being measured.
//
// It moves the BUFFER as well as the arithmetic, because switching only one of
// them draws garbage rather than the other representation.
export function renderRootIsPool(pools, override = null) {
  if (override !== null) return override ? 1 : 0;
  return pools[0] ? 1 : 0;
}

export function makeRenderBindGroup(device, layout, pools, { velBuf, cardStateBuf, overlayOpacityBuf, outlineOpacityBuf, rootIsPool }) {
  // THE CALLER PASSES THE SAME VALUE IT GAVE THE PIPELINE. Defaulting it here
  // instead would put the rule in two places and let them disagree, which is
  // the one failure this whole arrangement exists to prevent.
  if (rootIsPool === undefined) throw new Error('makeRenderBindGroup: pass rootIsPool (see renderRootIsPool) -- binding 0 and the ROOT_IS_POOL override must agree');
  const root = rootIsPool ? pools[0] : null;
  if (rootIsPool && !root) throw new Error('makeRenderBindGroup: rootIsPool=1 but no root pool is allocated');
  const entries = [
    { binding: 0, resource: { buffer: root ? root.finePoolVel : velBuf } },
    { binding: 1, resource: { buffer: cardStateBuf } },
    { binding: 4, resource: { buffer: overlayOpacityBuf } },
    { binding: 7, resource: { buffer: outlineOpacityBuf } },
  ];
  for (let m = 1; m <= MAX_RENDER_POOL_LEVELS; m++) {
    const pool = pools[m] || pools[1];
    const [velBinding, slotBinding] = RENDER_LEVEL_BINDINGS[m];
    entries.push({ binding: velBinding, resource: { buffer: pool.finePoolVel } });
    entries.push({ binding: slotBinding, resource: { buffer: pool.blockSlotBuf } });
  }
  // Level 1's indirection stands in when there is no root pool, for the same
  // reason the unused deeper levels get level 1's: ROOT_IS_POOL is what stops
  // it being read, and an out-of-range read is not safe on every stack.
  entries.push({ binding: 12, resource: { buffer: (root || pools[1]).blockSlotBuf } });
  // velBuf is still REQUIRED when rootIsPool is 0 -- the dense grid is the
  // only L0 velocity there is on that path.
  if (!root && !velBuf) throw new Error('makeRenderBindGroup: rootIsPool=0 needs velBuf');
  return device.createBindGroup({ layout, entries });
}

// How many pool levels the renderer will actually walk, and the refusal when a
// configuration asks for more than it can draw.
//
// A CAP THAT SILENTLY DROPS THE FINEST LEVEL IS THE DEFECT U6 EXISTS TO FIX,
// so this refuses rather than rendering a lie. `tools/validate-render-levels.js`
// measured the old behaviour directly: at `?levels=4`, perturbing level 3's
// whole velocity pool left the picture BYTE-IDENTICAL.
export function renderPoolLevels(nLevels) {
  const want = nLevels - 1;
  if (want > MAX_RENDER_POOL_LEVELS) {
    throw new Error(`?levels=${nLevels} needs ${want} pool levels in the renderer, `
      + `which binds ${MAX_RENDER_POOL_LEVELS} (shaders/amr_render.wgsl). Raising it means one more `
      + `binding pair there, in RENDER_LEVEL_BINDINGS, and in every page's renBGL. `
      + `Refused rather than drawn without the finest level -- see plans/uniform-levels.md U6.`);
  }
  return want;
}

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
// `quadCompleteFrom` is the shallowest level whose want set must be closed
// into QUADS, and it is a parameter since plans/uniform-levels.md U5-4. It is
// 2 while level 1 allocates per block, and 1 once level 1 is a quad child of
// the root -- at which point a want for one level-1 block is a want for its
// whole quad, exactly as at every deeper level.
//
// Level 1 never gets a `balance` pass either way: balance writes the PARENT
// level's want set, and the root is always full, so there is nothing there to
// want. That is the one thing about level 1 that stays special after U5, and
// it is a consequence of having no parent rather than of being level 1.
export function makeCascadePipelines(device, loadedModule, pools, nLevels, { quadCompleteFrom = 2 } = {}) {
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
  for (let m = nLevels - 1; m >= Math.min(2, quadCompleteFrom); m--) {
    const pool = pools[m];
    const hasParentLevel = m >= 2;
    const constants = { NBX: pool.NBX, NBY: pool.NBY, QUAD_COMPLETE: 1 };
    byLevel[m] = {
      completeQuads: device.createComputePipeline({
        layout, compute: { module: loadedModule, entryPoint: 'completeQuads', constants },
      }),
      balance: hasParentLevel ? device.createComputePipeline({
        layout, compute: { module: loadedModule, entryPoint: 'balance', constants },
      }) : null,
      // What to encode, in order. Naming it here rather than in encodeCascade
      // keeps "which passes this level has" next to "why it has them".
      entries: hasParentLevel ? ['completeQuads', 'balance'] : ['completeQuads'],
      bg: device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: pool.wantBuf } },
          // Level 1's parent-want binding is the ROOT's, which `completeQuads`
          // never reads and `balance` never runs to write. Bound because the
          // layout requires it, not because it means anything there.
          { binding: 1, resource: { buffer: pools[hasParentLevel ? m - 1 : 0].wantBuf } },
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
  for (let m = nLevels - 1; m >= 1; m--) {
    const c = cascade.byLevel[m];
    if (!c) continue;
    for (const entry of c.entries) {
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
