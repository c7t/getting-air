#!/usr/bin/env node
// Level 0 decoded from the ROOT POOL must equal level 0 decoded from the
// DENSE grid. (plans/uniform-levels.md U7-6e)
//
// `reconstructAMRToResolution` used to read L0 from the snapshot's dense
// `fB64`/`velB64` in block8 layout. Since U7-6c the snapshot also carries the
// root pool, and U7-6e makes that the source. Both representations hold the
// same field while the dense grid is still stepped (U5-3), so the two decoders
// must produce byte-identical output -- and this test is that comparison,
// on a hand-built fixture, with no GPU.
//
// THE TWO LAYOUTS ARE GENUINELY DIFFERENT, which is what makes this worth
// testing rather than asserting:
//
//   dense   block8 -- fixed 8x8 buffer-space blocks, block-major, row-major
//           within a block (shaders/amr_step.wgsl's cellIndex)
//   root    tile-major -- slot * (2*RB)^2 + ly*(2*RB) + lx, and the root tile
//           is RINGLESS, so there is no GHOST offset to skip. Every other
//           level's slot is FB = 2*RB + 2*GHOST on a side.
//
// The fixture fills both from ONE logical field, so a decoder that mixed the
// layouts up produces a mismatch rather than a plausible-looking picture.
//
// Run: node tools/test-root-l0-reconstruct.js  (exit 0 = all assertions pass)

const assert = require('assert');
const { reconstructAMRToResolution } = require('./lib/field-reconstruct');

const RB = 8, GHOST = 2, FB = RB * 2 + 2 * GHOST; // 20
const ROOT_SIDE = 2 * RB;                          // 16, ringless
const W0 = 16, H0 = 16;                            // exactly one root tile
const NCELLS0 = W0 * H0;
const BLOCK = 8;                                   // block8's sub-tile

const b64 = (f32) => Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString('base64');

// ONE logical field, addressed in buffer coordinates. Deliberately varies in
// both axes and is not symmetric, so a transposed or block-confused decode
// cannot accidentally agree.
const uxAt = (x, y) => 1 + x * 3 + y * 17;
const uyAt = (x, y) => -(2 + x * 5 + y * 11);
const rhoAt = (x, y) => 1 + (x * 100 + y) * 1e-6;

function denseBlock8Index(cx, cy) {
  const nbx = W0 / BLOCK;
  const bx = Math.floor(cx / BLOCK), by = Math.floor(cy / BLOCK);
  return (by * nbx + bx) * BLOCK * BLOCK + (cy % BLOCK) * BLOCK + (cx % BLOCK);
}

function buildDenseL0() {
  const vel = new Float32Array(NCELLS0 * 2);
  const f = new Float32Array(NCELLS0 * 9);
  for (let y = 0; y < H0; y++) {
    for (let x = 0; x < W0; x++) {
      const c = denseBlock8Index(x, y);
      vel[c * 2] = uxAt(x, y);
      vel[c * 2 + 1] = uyAt(x, y);
      // direction 0 alone carries rho; rhoFromF sums all nine planes
      f[0 * NCELLS0 + c] = rhoAt(x, y);
      for (let i = 1; i < 9; i++) f[i * NCELLS0 + c] = 0;
    }
  }
  return { vel, f };
}

function buildRootPool() {
  const nbx = W0 / ROOT_SIDE, nby = H0 / ROOT_SIDE;
  const slots = nbx * nby;
  const cellsPerSlot = ROOT_SIDE * ROOT_SIDE;
  const cells = slots * cellsPerSlot;
  const vel = new Float32Array(cells * 2);
  const f = new Float32Array(cells * 9);
  for (let by = 0; by < nby; by++) {
    for (let bx = 0; bx < nbx; bx++) {
      const slot = by * nbx + bx;           // the root's indirection is the identity
      for (let ly = 0; ly < ROOT_SIDE; ly++) {
        for (let lx = 0; lx < ROOT_SIDE; lx++) {
          const c = slot * cellsPerSlot + ly * ROOT_SIDE + lx;
          const x = bx * ROOT_SIDE + lx, y = by * ROOT_SIDE + ly;
          vel[c * 2] = uxAt(x, y);
          vel[c * 2 + 1] = uyAt(x, y);
          f[0 * cells + c] = rhoAt(x, y);
          for (let i = 1; i < 9; i++) f[i * cells + c] = 0;
        }
      }
    }
  }
  return {
    level: 0, MAX_FINE_BLOCKS: slots, NBLOCKS: slots, NBX: nbx, NBY: nby,
    cellsPerSlot, fB64: b64(f), velB64: b64(vel),
  };
}

// numLevels = 2: the root plus level 1. Level 1 has a 2x2 block grid and only
// ONE active tile, so most of the domain falls through to the L0 branch --
// which is the branch under test.
function buildFixture({ withRoot }) {
  const dense = buildDenseL0();
  const l1MaxBlocks = 4;
  const l1Cells = l1MaxBlocks * FB * FB;
  const velL1 = new Float32Array(l1Cells * 2);
  const fL1 = new Float32Array(l1Cells * 9);
  for (let fy = 0; fy < FB; fy++) {
    for (let fx = 0; fx < FB; fx++) {
      const c = 0 * FB * FB + fy * FB + fx;
      velL1[c * 2] = 5000 + fx * 10 + fy;
      velL1[c * 2 + 1] = -(5000 + fx * 10 + fy);
      fL1[0 * l1Cells + c] = 2 + fx * 1e-4 + fy * 1e-5;
    }
  }
  const cardState = new Array(26).fill(0);
  const snap = {
    formatVersion: withRoot ? 6 : 5, layout: 'block8', W: W0, H: H0, step: 0, cardState,
    fB64: b64(dense.f), velB64: b64(dense.vel),
    params: {}, numLevels: 2,
    pools: [
      null,
      { level: 1, RB, GHOST, FB, MAX_FINE_BLOCKS: l1MaxBlocks, NBLOCKS: 4, NBX: 2, NBY: 2,
        blockSlot: [0, -1, -1, -1], slotToBlock: [0, -1, -1, -1],
        fB64: b64(fL1), velB64: b64(velL1) },
    ],
  };
  if (withRoot) snap.root = buildRootPool();
  return snap;
}

let failures = 0;
function check(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); }
  catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}

const RES = 5; // W0(16) * 2^(numLevels-1 = 1) = 32 = 2^5

check('root-decoded L0 equals dense-decoded L0, cell for cell', () => {
  const snap = buildFixture({ withRoot: true });
  const viaDense = reconstructAMRToResolution(snap, RES, { l0Source: 'dense' });
  const viaRoot = reconstructAMRToResolution(snap, RES, { l0Source: 'root' });
  for (const key of ['ux', 'uy', 'rho']) {
    assert.deepStrictEqual(Array.from(viaRoot[key]), Array.from(viaDense[key]),
      `${key} differs between the root and dense decodes`);
  }
});

check('the comparison can fail -- a perturbed root does NOT match', () => {
  const snap = buildFixture({ withRoot: true });
  // Corrupt ONE root cell. If the decoders were reading the same bytes, or if
  // the root path were silently falling back to the dense one, this could not
  // change anything -- which is what makes the row above mean something.
  const cells = snap.root.MAX_FINE_BLOCKS * snap.root.cellsPerSlot;
  const vel = new Float32Array(Buffer.from(snap.root.velB64, 'base64').buffer.slice(0));
  vel[2 * (3 * 16 + 5)] += 1;
  snap.root.velB64 = Buffer.from(vel.buffer, vel.byteOffset, vel.byteLength).toString('base64');
  void cells;
  const viaDense = reconstructAMRToResolution(snap, RES, { l0Source: 'dense' });
  const viaRoot = reconstructAMRToResolution(snap, RES, { l0Source: 'root' });
  assert.notDeepStrictEqual(Array.from(viaRoot.ux), Array.from(viaDense.ux),
    'perturbing one root cell changed nothing -- the root path is not being read');
});

check('the root is the DEFAULT source when the snapshot carries one', () => {
  const snap = buildFixture({ withRoot: true });
  const vel = new Float32Array(Buffer.from(snap.root.velB64, 'base64').buffer.slice(0));
  vel[2 * (3 * 16 + 5)] += 1;
  snap.root.velB64 = Buffer.from(vel.buffer, vel.byteOffset, vel.byteLength).toString('base64');
  const auto = reconstructAMRToResolution(snap, RES);
  const viaRoot = reconstructAMRToResolution(snap, RES, { l0Source: 'root' });
  assert.deepStrictEqual(Array.from(auto.ux), Array.from(viaRoot.ux),
    'the default did not take the root path');
});

check('a pre-U7-6c snapshot still decodes, via the dense fallback', () => {
  const snap = buildFixture({ withRoot: false });
  const r = reconstructAMRToResolution(snap, RES);
  assert.strictEqual(r.W, 32);
  assert.ok(Number.isFinite(r.ux[0]));
});

check("asking for 'root' on a snapshot without one REFUSES", () => {
  const snap = buildFixture({ withRoot: false });
  assert.throws(() => reconstructAMRToResolution(snap, RES, { l0Source: 'root' }), /carries no root pool/);
});

check('a root whose geometry disagrees with the domain REFUSES', () => {
  const snap = buildFixture({ withRoot: true });
  snap.root.cellsPerSlot = 400; // FB*FB -- i.e. someone gave it a ringed tile
  assert.throws(() => reconstructAMRToResolution(snap, RES, { l0Source: 'root' }), /cellsPerSlot/);
});

console.log(`\nroot-l0-reconstruct: ${failures ? `${failures} test(s) FAILED` : 'all tests passed'}`);
process.exit(failures ? 1 : 0);
