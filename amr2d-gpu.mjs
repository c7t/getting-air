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

import { check21Balance, nbAtLevel, makePool } from './amr2d.mjs';

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
