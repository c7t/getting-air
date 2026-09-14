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
