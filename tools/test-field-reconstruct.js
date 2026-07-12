#!/usr/bin/env node
// Pure-Node self-test for tools/lib/field-reconstruct.js's
// reconstructAMRToResolution -- the single highest-risk piece of new logic
// in the dense-vs-AMR comparison tool (tools/validate-amr-vs-dense.js): a
// subtle indexing mistake here would silently produce wrong pass/fail
// verdicts against real snapshots. No GPU/Chrome needed -- this hand-builds
// a small, deliberately mixed-depth AMR snapshot (a level-1 block whose
// four level-2 quadrants are [active, INACTIVE, active, INACTIVE]) and
// asserts the reconstructed target-resolution field lands exactly where the
// geometry says it should, at both the finest (level-2, 1:1 injection) and
// standing-in (level-1, one cell fans out to a 2x2 target block) branches.
//
// Run: node tools/test-field-reconstruct.js  (exit 0 = all assertions pass)

const assert = require('assert');
const { reconstructAMRToResolution } = require('./lib/field-reconstruct');

const RB = 8, GHOST = 2, FB = RB * 2 + 2 * GHOST; // 20
const W0 = 8, H0 = 8; // single L0 block (NBX0=NBY0=1)
const NCELLS0 = W0 * H0;

// Fills a pool slot's full FBxFB cell range with a synthetic, uniquely
// identifiable value per cell: ux=base+fx*10+fy, uy=-ux, rho=1+(fx*100+fy)*scale.
// f is NOT a real equilibrium distribution -- direction 0 alone carries rho
// (rhoFromPoolF sums all 9 planes), directions 1-8 are zero. This is a pure
// indexing test, not a physics test.
function fillSlot(velOut, fOut, slot, cellsPerPool, base, rhoScale) {
  for (let fy = 0; fy < FB; fy++) {
    for (let fx = 0; fx < FB; fx++) {
      const cell = slot * FB * FB + fy * FB + fx;
      const ux = base + fx * 10 + fy;
      velOut[cell * 2] = ux;
      velOut[cell * 2 + 1] = -ux;
      const rho = 1 + (fx * 100 + fy) * rhoScale;
      fOut[0 * cellsPerPool + cell] = rho;
      for (let i = 1; i < 9; i++) fOut[i * cellsPerPool + cell] = 0;
    }
  }
}

function b64(arr) { return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64'); }

function buildFixture() {
  // L0: poison value -- must never appear in the output, since the single
  // L0 block is fully covered by an active L1 tile in this fixture.
  const velL0 = new Float32Array(NCELLS0 * 2).fill(-999);
  const fL0 = new Float32Array(NCELLS0 * 9).fill(0);
  for (let c = 0; c < NCELLS0; c++) fL0[0 * NCELLS0 + c] = -999; // rho=-999 too

  // Level 1: one active slot (slot 0) covering the whole (only) L0 block.
  const l1MaxBlocks = 1;
  const l1CellsPerPool = l1MaxBlocks * FB * FB;
  const velL1 = new Float32Array(l1CellsPerPool * 2);
  const fL1 = new Float32Array(l1CellsPerPool * 9);
  fillSlot(velL1, fL1, 0, l1CellsPerPool, 1000, 1e-6);

  // Level 2: two active slots (sA=0 covers L1's quadrant (0,0), sB=1 covers
  // quadrant (0,1)); quadrants (1,0) and (1,1) are INACTIVE (blockSlot=-1),
  // so level 1 stands in for those -- this is the mixed-depth case that
  // matters (see this file's header).
  const l2MaxBlocks = 2;
  const l2CellsPerPool = l2MaxBlocks * FB * FB;
  const velL2 = new Float32Array(l2CellsPerPool * 2);
  const fL2 = new Float32Array(l2CellsPerPool * 9);
  fillSlot(velL2, fL2, 0, l2CellsPerPool, 2000, 2e-6); // sA
  fillSlot(velL2, fL2, 1, l2CellsPerPool, 3000, 3e-6); // sB

  const cardState = new Array(26).fill(0); // off_x=[22]=0, off_y=[23]=0

  return {
    formatVersion: 5, layout: 'block8', W: W0, H: H0, step: 0, cardState,
    fB64: b64(fL0), velB64: b64(velL0),
    params: {}, numLevels: 3,
    pools: [
      null,
      { level: 1, RB, GHOST, FB, MAX_FINE_BLOCKS: l1MaxBlocks, NBLOCKS: 1, NBX: 1, NBY: 1,
        blockSlot: [0], slotToBlock: [0],
        fB64: b64(fL1), velB64: b64(velL1) },
      { level: 2, RB, GHOST, FB, MAX_FINE_BLOCKS: l2MaxBlocks, NBLOCKS: 4, NBX: 2, NBY: 2,
        blockSlot: [0, -1, 1, -1], slotToBlock: [0, 2],
        parentSlot: [0, 0], quadrant: [0, 2], originX: [0, 0], originY: [0, 4],
        fB64: b64(fL2), velB64: b64(velL2) },
    ],
  };
}

function main() {
  const snapshot = buildFixture();
  const targetResLog2 = 5; // W0(8) * 2^(numLevels-1=2) = 32 = 2^5
  const result = reconstructAMRToResolution(snapshot, targetResLog2);
  assert.strictEqual(result.W, 32);
  assert.strictEqual(result.H, 32);

  const at = (x, y) => ({ ux: result.ux[y * 32 + x], uy: result.uy[y * 32 + x], rho: result.rho[y * 32 + x] });
  const valL1 = (fx, fy) => 1000 + fx * 10 + fy;
  const valL2A = (fx, fy) => 2000 + fx * 10 + fy;
  const valL2B = (fx, fy) => 3000 + fx * 10 + fy;
  const rhoL2A = (fx, fy) => 1 + (fx * 100 + fy) * 2e-6;
  const rhoL2B = (fx, fy) => 1 + (fx * 100 + fy) * 3e-6;
  const rhoL1 = (fx, fy) => 1 + (fx * 100 + fy) * 1e-6;
  // 1e-5, not tighter: rho values round-trip through a Float32Array (~1.2e-7
  // relative precision), so exact double-precision equality isn't meaningful
  // here -- ux/uy are small integers, exactly representable in float32, but
  // sharing one epsilon for both keeps the assertions simple.
  const EPS = 1e-5;

  // Region [0,16)x[0,16): finest, from L2 slot sA, 1:1 injection.
  // target(tx,ty) -> L2 fine cell (GHOST+tx, GHOST+ty).
  let p = at(0, 0);
  assert.ok(Math.abs(p.ux - valL2A(2, 2)) < EPS, `(0,0) ux=${p.ux}, expected ${valL2A(2, 2)}`);
  assert.ok(Math.abs(p.rho - rhoL2A(2, 2)) < EPS, `(0,0) rho=${p.rho}, expected ${rhoL2A(2, 2)}`);
  p = at(15, 15);
  assert.ok(Math.abs(p.ux - valL2A(17, 17)) < EPS, `(15,15) ux=${p.ux}, expected ${valL2A(17, 17)}`);

  // Region [16,32)x[0,16): L1 standing in for inactive L2 quadrant (1,0).
  // Each L1 fine cell (dxL=0.5) fans out to a 2x2 target block.
  p = at(16, 0);
  assert.ok(Math.abs(p.ux - valL1(10, 2)) < EPS, `(16,0) ux=${p.ux}, expected ${valL1(10, 2)}`);
  p = at(17, 1);
  assert.ok(Math.abs(p.ux - valL1(10, 2)) < EPS, `(17,1) ux=${p.ux}, expected ${valL1(10, 2)} (same 2x2 block as (16,0))`);
  p = at(18, 0);
  assert.ok(Math.abs(p.ux - valL1(11, 2)) < EPS, `(18,0) ux=${p.ux}, expected ${valL1(11, 2)} (next 2x2 block)`);
  p = at(31, 15);
  assert.ok(Math.abs(p.ux - valL1(17, 9)) < EPS, `(31,15) ux=${p.ux}, expected ${valL1(17, 9)}`);

  // Region [0,16)x[16,32): finest, from L2 slot sB.
  p = at(0, 16);
  assert.ok(Math.abs(p.ux - valL2B(2, 2)) < EPS, `(0,16) ux=${p.ux}, expected ${valL2B(2, 2)}`);
  assert.ok(Math.abs(p.rho - rhoL2B(2, 2)) < EPS, `(0,16) rho=${p.rho}, expected ${rhoL2B(2, 2)}`);

  // Region [16,32)x[16,32): L1 standing in for inactive L2 quadrant (1,1).
  p = at(16, 16);
  assert.ok(Math.abs(p.ux - valL1(10, 10)) < EPS, `(16,16) ux=${p.ux}, expected ${valL1(10, 10)}`);
  p = at(31, 31);
  assert.ok(Math.abs(p.ux - valL1(17, 17)) < EPS, `(31,31) ux=${p.ux}, expected ${valL1(17, 17)}`);
  assert.ok(Math.abs(p.rho - rhoL1(17, 17)) < EPS, `(31,31) rho=${p.rho}, expected ${rhoL1(17, 17)}`);

  // Poison check: L0's -999 must never appear anywhere (L1 fully covers the
  // single L0 block in this fixture).
  for (let i = 0; i < result.ux.length; i++) {
    assert.notStrictEqual(result.ux[i], -999, `L0 poison value leaked into target cell ${i}`);
  }

  // finestCoverageFraction: exactly half the domain (the two L2-covered
  // 16x16 regions) is genuinely at the finest configured level; the other
  // half is L1 standing in for a missing L2 child.
  assert.ok(Math.abs(result.finestCoverageFraction - 0.5) < EPS,
    `finestCoverageFraction=${result.finestCoverageFraction}, expected 0.5`);

  console.log('tools/test-field-reconstruct.js: all assertions passed (per-quadrant recursion, finest injection, standin fan-out, poison-leak, coverage fraction)');
}

main();
