#!/usr/bin/env node
// The root pool's INITIAL STATE, laid out on the host. (plans/uniform-levels.md U7-6f)
//
// Until U7-6f the root was filled by mirroring the live dense `f` buffer
// through shaders/amr_mirror_root.wgsl. That is the last thing the dense L0
// was needed for, so the mirror's permutation moved to the host
// (amr2d.mjs's `denseL0ToRootF`) and the shader, its pipeline and
// tools/validate-root-mirror.js went with it -- a checker retiring with its
// subject, which is B3-5's rule.
//
// THE PROOF THE MIRROR CARRIED HAS TO SURVIVE THE MOVE, and U2's lesson says
// how not to do it. The mirror shader and amr2d.mjs's `rootCellToDense` were
// written together, both said `gy*W + gx`, they agreed exactly, and 98.4% of
// the root pool was reading the wrong dense cell. Authorship is not
// independence. So this scores the new permutation against a THIRD route that
// neither wrote and that is validated by DATA -- tools/lib/
// field-reconstruct.js's `rootToFlatL0` and `unshiftField`, the decoders that
// read real GPU snapshots:
//
//     dense block8  --denseL0ToRootF-->  root tiles  --rootToFlatL0-->  flat
//     dense block8  --unshiftField---------------------------------->  flat
//
// Those two flat arrays must be equal cell for cell. The fixture's field
// varies in both axes and is not symmetric, so a transposed, block-confused or
// stride-confused permutation cannot agree by accident -- and the mutation
// rows below check exactly that.
//
// Run: node tools/test-root-seed.js  (exit 0 = all assertions pass)

const assert = require('assert');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); pass++; }
  catch (e) { console.log(`  FAIL  ${name}\n        ${e.message}`); process.exitCode = 1; }
}

const b64 = (f32) => Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString('base64');

(async () => {
  const A = await import(path.join(__dirname, '..', 'amr2d.mjs'));
  const { denseL0ToRootF, denseL0ToRootVel, denseCellIndex, rootCellIndex,
          rootCellToDense, rootPoolSpec } = A;
  const { reconstructAMRToResolution } = require('./lib/field-reconstruct');

  const RB = 8, SIDE = 2 * RB;
  // Deliberately NOT square and deliberately more than one tile per axis: a
  // domain of exactly one tile makes the tile walk trivial, and a square one
  // cannot tell an x/y transposition from the identity.
  const W = 64, H = 32, dims = { W, H };
  const NCELLS = W * H;

  // One logical field, addressed in buffer coordinates.
  const uxAt = (x, y) => 1 + x * 3 + y * 17;
  const uyAt = (x, y) => -(2 + x * 5 + y * 11);
  const fAt = (x, y, i) => (i + 1) * 1000 + x + y * 0.5;

  function buildDense() {
    const vel = new Float32Array(NCELLS * 2);
    const f = new Float32Array(NCELLS * 9);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = denseCellIndex({ dims }, x, y);
        vel[c * 2] = uxAt(x, y); vel[c * 2 + 1] = uyAt(x, y);
        for (let i = 0; i < 9; i++) f[i * NCELLS + c] = fAt(x, y, i);
      }
    }
    return { vel, f };
  }

  const dense = buildDense();
  const spec = rootPoolSpec({ dims, rb: RB });

  // A HAND LAYOUT OF THE ROOT'S VELOCITY. It was the only one until
  // `denseL0ToRootVel` existed -- a fixture, because `f` was the subject and
  // velocity only had to be consistent with the dense side -- and it is now
  // the INDEPENDENT ROUTE the real function is scored against. It is written
  // straight off `rootCellIndex` with no reference to `rootFromDenseMap`,
  // which is the composition `denseL0ToRootVel` uses, so the two agreeing is
  // not two spellings of one formula.
  function rootVelFixture() {
    const out = new Float32Array(spec.cells * 2);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = rootCellIndex({ dims, rb: RB }, x, y);
        out[c * 2] = uxAt(x, y); out[c * 2 + 1] = uyAt(x, y);
      }
    }
    return out;
  }

  ok('the root holds exactly W*H cells -- no padding case to reason about', () => {
    assert.strictEqual(spec.cells, NCELLS);
    assert.strictEqual(spec.slots, spec.nblocks);
  });

  ok('every root cell carries the dense value at its own spatial coordinates', () => {
    // The composition U2 says to prefer over an agreement check: not "do the
    // two formulas match" but "does this root cell hold the field at the
    // place it stands for".
    const rf = denseL0ToRootF(dense.f, { dims, rb: RB });
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = rootCellIndex({ dims, rb: RB }, x, y);
        for (let i = 0; i < 9; i++) {
          assert.strictEqual(rf[i * spec.cells + c], fAt(x, y, i), `f[${i}] at (${x},${y})`);
        }
      }
    }
  });

  // --- VELOCITY, the companion denseL0ToRootF did not have at U7-6f --------
  //
  // It was added because two of the five AMR pages gave the root a velocity
  // that was an APPROXIMATION of what its own `f` represented, and the first
  // criterion evaluation after a reset reads it. See plans/uniform-levels.md
  // "U7-6f -- WHAT IT LEFT BEHIND".

  ok('every root cell carries the dense VELOCITY at its own spatial coordinates', () => {
    const rv = denseL0ToRootVel(dense.vel, { dims, rb: RB });
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = rootCellIndex({ dims, rb: RB }, x, y);
        assert.strictEqual(rv[c * 2], uxAt(x, y), `ux at (${x},${y})`);
        assert.strictEqual(rv[c * 2 + 1], uyAt(x, y), `uy at (${x},${y})`);
      }
    }
  });

  ok('denseL0ToRootVel reproduces the hand layout exactly -- and is INTERLEAVED, not plane-major', () => {
    // The layout trap this row exists for: `f` is plane-major (9 planes of
    // NCELLS) and velocity is interleaved [ux, uy] per cell, because that is
    // what finePoolVel and writePoolInitialState's velFill speak. A copy of
    // denseL0ToRootF's inner loop with 9 changed to 2 would pass every
    // spatial-permutation check above and still write ux into the first half
    // of the buffer and uy into the second.
    assert.deepStrictEqual(
      Array.from(denseL0ToRootVel(dense.vel, { dims, rb: RB })),
      Array.from(rootVelFixture()));
  });

  ok('a PLANE-MAJOR velocity layout fails that comparison -- the row is not vacuous', () => {
    const planeMajor = new Float32Array(spec.cells * 2);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = rootCellIndex({ dims, rb: RB }, x, y);
        planeMajor[c] = uxAt(x, y);
        planeMajor[spec.cells + c] = uyAt(x, y);
      }
    }
    assert.notDeepStrictEqual(Array.from(planeMajor), Array.from(rootVelFixture()));
  });

  ok('a wrong-length velocity REFUSES rather than reading past the end', () => {
    assert.throws(() => denseL0ToRootVel(new Float32Array(NCELLS * 9), { dims, rb: RB }), /expected/);
  });

  ok('rootCellToDense round-trips the same permutation, from the other end', () => {
    const rf = denseL0ToRootF(dense.f, { dims, rb: RB });
    for (let slot = 0; slot < spec.slots; slot++) {
      for (const [lx, ly] of [[0, 0], [1, 0], [0, 1], [SIDE - 1, SIDE - 1], [7, 9]]) {
        const d = rootCellToDense({ dims, rb: RB }, slot, lx, ly);
        const c = slot * spec.cellsPerSlot + ly * SIDE + lx;
        assert.strictEqual(rf[0 * spec.cells + c], dense.f[0 * NCELLS + d],
          `slot ${slot} local (${lx},${ly})`);
      }
    }
  });

  // --- THE THIRD ROUTE: the snapshot decoders, validated by real data -------
  //
  // Build an AMR snapshot whose root is the seeded pool and whose dense arrays
  // are the source, and reconstruct it BOTH ways. Level 1 is present (the
  // walk requires it) but holds NO active tile, so every cell falls through to
  // the level-0 branch -- which is the branch under test, with none of the
  // quadtree painting in the way.
  const GHOST = 2, FB = 2 * RB + 2 * GHOST;
  const L1_SLOTS = 4;
  function snapshotWith(rootF, rootVel, denseF, denseVel) {
    return {
      W, H, numLevels: 2, formatVersion: 6, layout: 'block8', step: 0, params: {},
      cardState: new Array(26).fill(0),
      fB64: b64(denseF), velB64: b64(denseVel),
      root: {
        level: 0, RB, GHOST: 0, FB: SIDE,
        NBX: spec.nbx, NBY: spec.nby, NBLOCKS: spec.nblocks,
        MAX_FINE_BLOCKS: spec.slots, cellsPerSlot: spec.cellsPerSlot,
        fB64: b64(rootF), velB64: b64(rootVel),
      },
      pools: [null, {
        level: 1, RB, GHOST, FB,
        MAX_FINE_BLOCKS: L1_SLOTS, NBLOCKS: spec.nbx * 2 * spec.nby * 2,
        NBX: spec.nbx * 2, NBY: spec.nby * 2,
        blockSlot: new Array(spec.nbx * 2 * spec.nby * 2).fill(-1),
        slotToBlock: new Array(L1_SLOTS).fill(-1),
        fB64: b64(new Float32Array(L1_SLOTS * FB * FB * 9)),
        velB64: b64(new Float32Array(L1_SLOTS * FB * FB * 2)),
      }],
    };
  }
  const resLog2 = Math.log2(W * 2);
  const bothWays = (rootF, rootVel) => {
    const snap = snapshotWith(rootF, rootVel, dense.f, dense.vel);
    return [
      reconstructAMRToResolution(snap, resLog2, { l0Source: 'root' }),
      reconstructAMRToResolution(snap, resLog2, { l0Source: 'dense' }),
    ];
  };

  ok('the seeded root decodes to the field the dense grid decodes to', () => {
    // BOTH halves of the seed are the subject here, not just `f`: `ux`/`uy`
    // in this comparison now come from denseL0ToRootVel rather than the hand
    // fixture, so the velocity permutation is gated by the same third route
    // the `f` one is -- decoders validated by real GPU snapshots.
    const rf = denseL0ToRootF(dense.f, { dims, rb: RB });
    const rv = denseL0ToRootVel(dense.vel, { dims, rb: RB });
    const [viaRoot, viaDense] = bothWays(rf, rv);
    for (const field of ['ux', 'uy', 'rho']) {
      assert.deepStrictEqual(Array.from(viaRoot[field]), Array.from(viaDense[field]),
        `${field} differs between the root and dense decodes`);
    }
  });

  // --- MUTATIONS: the rows above must be able to go red --------------------

  ok('a TRANSPOSED root local index fails the decode comparison', () => {
    const map = new Float32Array(spec.cells * 9);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const slot = Math.floor(y / SIDE) * spec.nbx + Math.floor(x / SIDE);
        const c = slot * spec.cellsPerSlot + (x % SIDE) * SIDE + (y % SIDE); // ly/lx swapped
        for (let i = 0; i < 9; i++) map[i * spec.cells + c] = fAt(x, y, i);
      }
    }
    const [viaRoot, viaDense] = bothWays(map, rootVelFixture());
    assert.notDeepStrictEqual(Array.from(viaRoot.rho), Array.from(viaDense.rho),
      'a transposed root decoded the same as the dense grid -- the check is vacuous');
  });

  ok("a ROW-MAJOR root -- U2's original bug -- fails the decode comparison", () => {
    // `gy*W + gx` on the dense side instead of block8: the exact slip that
    // survived two agreeing routes and 98.4% wrong cells.
    const rowMajor = new Float32Array(spec.cells * 9);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = rootCellIndex({ dims, rb: RB }, x, y);
        for (let i = 0; i < 9; i++) rowMajor[i * spec.cells + c] = dense.f[i * NCELLS + (y * W + x)];
      }
    }
    const [viaRoot, viaDense] = bothWays(rowMajor, rootVelFixture());
    assert.notDeepStrictEqual(Array.from(viaRoot.rho), Array.from(viaDense.rho),
      'a row-major root decoded the same as the dense grid -- the check is vacuous');
  });

  ok('a wrong-length input REFUSES rather than reading past the end', () => {
    assert.throws(() => denseL0ToRootF(new Float32Array(NCELLS * 8), { dims, rb: RB }), /expected/);
  });

  ok('a domain that does not divide into whole root tiles REFUSES', () => {
    const bad = { W: 40, H: 32 };
    assert.throws(() => denseL0ToRootF(new Float32Array(40 * 32 * 9), { dims: bad, rb: RB }),
      /does not divide/);
  });

  if (!process.exitCode) console.log(`\nroot-seed: ${pass} test(s) passed`);
  else console.log('\nroot-seed: FAILED');
})();
