#!/usr/bin/env node
// Pure-Node self-test for tools/lib/dense-to-amr.js. No GPU, no Chrome.
//
// The injector's correctness question is "does dense state land in the right
// AMR cells, at the right level, with the right non-equilibrium rescale". All
// three are checkable offline against code that is already itself
// fixture-tested (tools/test-field-reconstruct.js), by ROUND-TRIP:
//
//   dense field -> injectDenseIntoAMRSnapshot -> reconstructAMRToResolution
//                                             -> must reproduce the dense field
//
// with the expected loss, and only the expected loss, wherever the AMR
// hierarchy is coarser than the dense grid: a level-m cell covers 2^(finest-m)
// dense cells per axis, so it can only carry their average, and reconstruct
// then fans that one value back over the same square. Where the AMR IS at the
// finest level the round-trip must be exact -- including f itself, not just
// the macroscopic fields, since the rescale factor is 1 there by construction.
//
// The fixture is deliberately mixed-depth so all three branches are exercised
// in one pass: L0-authoritative blocks, an L1 tile standing in for
// un-refined quadrants, and L2 tiles at the finest level.
//
// Run: node tools/test-dense-to-amr.js  (exit 0 = all assertions pass)

const assert = require('assert');
const { reconstructAMRToResolution, b64ToFloat32 } = require('./lib/field-reconstruct');
const {
  injectDenseIntoAMRSnapshot, fneqRescale, tauAtLevel, feq,
} = require('./lib/dense-to-amr');

const RB = 8, GHOST = 2, FB = RB * 2 + 2 * GHOST; // 20
const W0 = 16, H0 = 16;          // 2x2 L0 blocks, so some stay L0-authoritative
const NUM_LEVELS = 3;
const MULT = 1 << (NUM_LEVELS - 1); // 4
const TARGET_W = W0 * MULT;         // 64
const TAU0 = 0.5045;

const EX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const EY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

let passed = 0;
const failures = [];
function test(name, fn) {
  try { fn(); passed++; console.log(`  ok    ${name}`); }
  catch (e) {
    failures.push(name);
    console.log(`  FAIL  ${name}`);
    console.log(String(e.message).split('\n').map(l => `        ${l}`).join('\n'));
  }
}
function b64(arr) { return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64'); }

// ── Analytic dense field ────────────────────────────────────────────────────
// Smooth and non-separable, so an indexing slip (x/y swap, block-major vs
// row-major, a half-tile offset) cannot alias into a passing result the way
// it can with a constant or an axis-aligned ramp.
function denseRho(x, y) { return 1 + 0.01 * Math.sin(2 * Math.PI * x / TARGET_W) * Math.cos(2 * Math.PI * y / TARGET_W); }
function denseUx(x, y) { return 0.03 * Math.sin(2 * Math.PI * y / TARGET_W) + 0.004 * Math.cos(6 * Math.PI * x / TARGET_W); }
function denseUy(x, y) { return -0.02 * Math.sin(2 * Math.PI * x / TARGET_W) + 0.003 * Math.sin(4 * Math.PI * y / TARGET_W); }

// Non-equilibrium part built from a symmetric TRACELESS tensor contraction,
// the standard fneq form. This is mass- AND momentum-conserving by
// construction (sum_i w_i(e_ie_i - c_s^2 I) = 0 and its first moment vanishes),
// so rhoFromF still recovers rho exactly and the injected velocity stays
// consistent with the stored velBuf -- a fixture with an arbitrary fneq would
// fail the round-trip for reasons that say nothing about the injector.
function denseFneq(x, y, i) {
  const Qxx = 1e-4 * Math.sin(2 * Math.PI * x / TARGET_W) * Math.sin(2 * Math.PI * y / TARGET_W);
  const Qxy = 7e-5 * Math.cos(2 * Math.PI * (x + y) / TARGET_W);
  const Qyy = -Qxx; // traceless
  return WT[i] * 4.5 * (Qxx * (EX[i] * EX[i] - 1 / 3) + 2 * Qxy * EX[i] * EY[i] + Qyy * (EY[i] * EY[i] - 1 / 3));
}

function buildDenseSnapshot() {
  const N = TARGET_W * TARGET_W;
  const f = new Float32Array(N * 9);
  const vel = new Float32Array(N * 2);
  for (let y = 0; y < TARGET_W; y++) {
    for (let x = 0; x < TARGET_W; x++) {
      const c = y * TARGET_W + x;
      const rho = denseRho(x, y), ux = denseUx(x, y), uy = denseUy(x, y);
      vel[c * 2] = ux; vel[c * 2 + 1] = uy;
      for (let i = 0; i < 9; i++) f[i * N + c] = feq(rho, ux, uy, i) + denseFneq(x, y, i);
    }
  }
  const cardState = new Array(26).fill(0);
  cardState[19] = tauAtLevel(TAU0, NUM_LEVELS - 1); // dense tau = AMR's finest-level tau
  return {
    formatVersion: 1, layout: 'flat', W: TARGET_W, H: TARGET_W, step: 12345,
    cardState, fB64: b64(f), velB64: b64(vel), params: {},
  };
}

// ── AMR topology fixture ────────────────────────────────────────────────────
// L1 active on L0 block (0,0) only -> blocks (1,0),(0,1),(1,1) stay
// L0-authoritative. Under that L1 tile, two of the four L2 quadrants are
// active -> the other two are served by L1 standing in.
function buildAMRSnapshot() {
  const NCELLS0 = W0 * H0;
  // Poison: every cell the injector is responsible for must be overwritten.
  const fL0 = new Float32Array(NCELLS0 * 9).fill(-999);
  const velL0 = new Float32Array(NCELLS0 * 2).fill(-999);

  const L1_MAX = 4, L2_MAX = 4; // L2 must be a multiple of 4 (quad allocation)
  const fL1 = new Float32Array(L1_MAX * FB * FB * 9).fill(-999);
  const velL1 = new Float32Array(L1_MAX * FB * FB * 2).fill(-999);
  const fL2 = new Float32Array(L2_MAX * FB * FB * 9).fill(-999);
  const velL2 = new Float32Array(L2_MAX * FB * FB * 2).fill(-999);

  const cardState = new Array(26).fill(0);
  cardState[19] = TAU0;          // L0 tau
  // cardState[22]/[23] (off_x/off_y) left 0: pinned-window (cylinder) case.

  // L1 grid is 2x2 (NBX0 = W0/RB = 2); only block (0,0) active, in slot 0.
  const l1BlockSlot = [0, -1, -1, -1];
  const l1SlotToBlock = [0, -1, -1, -1];

  // L2 grid is 4x4. L1 block (0,0)'s children are (0,0),(1,0),(0,1),(1,1).
  // Activate (0,0)->slot 0 and (1,1)->slot 1; leave (1,0),(0,1) inactive.
  const l2BlockSlot = new Array(16).fill(-1);
  l2BlockSlot[0 * 4 + 0] = 0;
  l2BlockSlot[1 * 4 + 1] = 1;
  const l2SlotToBlock = [0, 5, -1, -1];

  return {
    formatVersion: 5, layout: 'block8', W: W0, H: H0, step: 777, cardState,
    fB64: b64(fL0), velB64: b64(velL0), params: {}, numLevels: NUM_LEVELS,
    pools: [
      null,
      {
        level: 1, RB, GHOST, FB, MAX_FINE_BLOCKS: L1_MAX, NBLOCKS: 4, NBX: 2, NBY: 2,
        blockSlot: l1BlockSlot, slotToBlock: l1SlotToBlock,
        fB64: b64(fL1), velB64: b64(velL1),
      },
      {
        level: 2, RB, GHOST, FB, MAX_FINE_BLOCKS: L2_MAX, NBLOCKS: 16, NBX: 4, NBY: 4,
        blockSlot: l2BlockSlot, slotToBlock: l2SlotToBlock,
        parentSlot: [0, 0, -1, -1], quadrant: [0, 3, -1, -1],
        originX: [0, 4, 0, 0], originY: [0, 4, 0, 0],
        fB64: b64(fL2), velB64: b64(velL2),
      },
    ],
  };
}

// Expected value of a level-m cell: the box-average of its dense footprint,
// with the same conservation choices amr_average_f2c.wgsl makes (arithmetic
// mean density, mass-weighted velocity).
function boxAverage(x0, y0, size) {
  let rhoSum = 0, mux = 0, muy = 0, n = 0;
  for (let dy = 0; dy < size; dy++) {
    for (let dx = 0; dx < size; dx++) {
      const x = (x0 + dx) % TARGET_W, y = (y0 + dy) % TARGET_W;
      const r = denseRho(x, y);
      rhoSum += r; mux += r * denseUx(x, y); muy += r * denseUy(x, y); n++;
    }
  }
  return { rho: rhoSum / n, ux: mux / rhoSum, uy: muy / rhoSum };
}

function main() {
  const denseSnap = buildDenseSnapshot();
  const amrSnap = buildAMRSnapshot();
  const injected = injectDenseIntoAMRSnapshot({ denseSnapshot: denseSnap, amrSnapshot: amrSnap });

  test('injection preserves topology and metadata, replacing only field data', () => {
    assert.strictEqual(injected.numLevels, NUM_LEVELS);
    assert.strictEqual(injected.W, W0);
    assert.deepStrictEqual(injected.pools[1].blockSlot, amrSnap.pools[1].blockSlot);
    assert.deepStrictEqual(injected.pools[2].blockSlot, amrSnap.pools[2].blockSlot);
    assert.deepStrictEqual(injected.pools[2].parentSlot, amrSnap.pools[2].parentSlot);
    assert.notStrictEqual(injected.fB64, amrSnap.fB64, 'L0 field data should have been replaced');
    assert.ok(injected.injectedFrom, 'output should record its provenance');
    assert.strictEqual(injected.injectedFrom.denseStep, 12345);
  });

  test('no poison value survives anywhere the injector is responsible for', () => {
    const N0 = W0 * H0;
    const fL0 = b64ToFloat32(injected.fB64, N0 * 9);
    for (let k = 0; k < fL0.length; k++) {
      assert.ok(fL0[k] !== -999, `L0 f still holds the poison value at index ${k}`);
    }
    // Active slots only: inactive slots are deliberately passed through, and
    // every shader skips them via slotToBlock[slot] < 0.
    for (const m of [1, 2]) {
      const p = injected.pools[m];
      const cells = p.MAX_FINE_BLOCKS * FB * FB;
      const f = b64ToFloat32(p.fB64, cells * 9);
      for (let slot = 0; slot < p.MAX_FINE_BLOCKS; slot++) {
        if (p.slotToBlock[slot] === -1) continue;
        for (let fy = 0; fy < FB; fy++) {
          for (let fx = 0; fx < FB; fx++) {
            const idx = slot * FB * FB + fy * FB + fx;
            assert.ok(f[0 * cells + idx] !== -999,
              `level ${m} slot ${slot} cell (${fx},${fy}) still holds the poison value (ghosts included)`);
          }
        }
      }
    }
  });

  // ── The round-trip ────────────────────────────────────────────────────────
  const back = reconstructAMRToResolution(injected, Math.log2(TARGET_W));

  test('round-trip is EXACT where the AMR is at its finest level', () => {
    // L2 slot 0 covers L1 block (0,0)'s quadrant (0,0): L0 units [0,4)x[0,4),
    // i.e. target cells [0,16)x[0,16). rescale is 1 and the footprint is one
    // dense cell, so this must reproduce the dense field to float precision.
    let checked = 0;
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        const c = y * TARGET_W + x;
        assert.ok(Math.abs(back.rho[c] - denseRho(x, y)) < 1e-6, `rho at (${x},${y}): ${back.rho[c]} vs ${denseRho(x, y)}`);
        assert.ok(Math.abs(back.ux[c] - denseUx(x, y)) < 1e-6, `ux at (${x},${y}): ${back.ux[c]} vs ${denseUx(x, y)}`);
        assert.ok(Math.abs(back.uy[c] - denseUy(x, y)) < 1e-6, `uy at (${x},${y}): ${back.uy[c]} vs ${denseUy(x, y)}`);
        checked++;
      }
    }
    assert.strictEqual(checked, 256);
  });

  test('round-trip is the exact box-average where L1 stands in (2x2 fan-out)', () => {
    // L1 block (0,0) quadrant (1,0) has no L2 child -> L1 is finest there.
    // L0 units [4,8)x[0,4) -> target [16,32)x[0,16), in 2x2 target blocks.
    let checked = 0;
    for (let ty = 0; ty < 16; ty += 2) {
      for (let tx = 16; tx < 32; tx += 2) {
        const exp = boxAverage(tx, ty, 2);
        for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
          const c = (ty + dy) * TARGET_W + (tx + dx);
          assert.ok(Math.abs(back.rho[c] - exp.rho) < 1e-6, `rho at (${tx + dx},${ty + dy}): ${back.rho[c]} vs ${exp.rho}`);
          assert.ok(Math.abs(back.ux[c] - exp.ux) < 1e-6, `ux at (${tx + dx},${ty + dy}): ${back.ux[c]} vs ${exp.ux}`);
          assert.ok(Math.abs(back.uy[c] - exp.uy) < 1e-6, `uy at (${tx + dx},${ty + dy}): ${back.uy[c]} vs ${exp.uy}`);
        }
        checked++;
      }
    }
    assert.strictEqual(checked, 64);
  });

  test('round-trip is the exact box-average where L0 is authoritative (4x4 fan-out)', () => {
    // L0 block (1,0) has no L1 tile -> L0 is finest. L0 units [8,16)x[0,8)
    // -> target [32,64)x[0,32), in 4x4 target blocks.
    let checked = 0;
    for (let ty = 0; ty < 32; ty += 4) {
      for (let tx = 32; tx < 64; tx += 4) {
        const exp = boxAverage(tx, ty, 4);
        for (let dy = 0; dy < 4; dy++) {
          for (let dx = 0; dx < 4; dx++) {
            const c = (ty + dy) * TARGET_W + (tx + dx);
            assert.ok(Math.abs(back.rho[c] - exp.rho) < 1e-6, `rho at (${tx + dx},${ty + dy}): ${back.rho[c]} vs ${exp.rho}`);
            assert.ok(Math.abs(back.ux[c] - exp.ux) < 1e-6, `ux at (${tx + dx},${ty + dy}): ${back.ux[c]} vs ${exp.ux}`);
          }
        }
        checked++;
      }
    }
    assert.strictEqual(checked, 64);
  });

  // ── f itself, not just the macroscopic fields ─────────────────────────────
  // reconstructAMRToResolution only returns rho/ux/uy, so the round-trip
  // above cannot see whether the non-equilibrium part survived. At the finest
  // level the rescale is exactly 1 over a one-cell footprint, so the injected
  // f must equal the dense f cell for cell -- the sharpest available check
  // that fneq is carried rather than silently dropped (injecting f=feq would
  // pass every assertion above and fail this one).
  test('injected f reproduces dense f exactly at the finest level, fneq included', () => {
    const p = injected.pools[2];
    const cells = p.MAX_FINE_BLOCKS * FB * FB;
    const f = b64ToFloat32(p.fB64, cells * 9);
    const N = TARGET_W * TARGET_W;
    const denseF = b64ToFloat32(denseSnap.fB64, N * 9);
    let checked = 0, maxErr = 0, sawNonzeroFneq = false;
    // L2 slot 0's interior maps 1:1 to target cells [0,16)x[0,16).
    for (let ly = 0; ly < 2 * RB; ly++) {
      for (let lx = 0; lx < 2 * RB; lx++) {
        const idx = 0 * FB * FB + (GHOST + ly) * FB + (GHOST + lx);
        const c = ly * TARGET_W + lx;
        for (let i = 0; i < 9; i++) {
          maxErr = Math.max(maxErr, Math.abs(f[i * cells + idx] - denseF[i * N + c]));
        }
        if (Math.abs(denseFneq(lx, ly, 5)) > 1e-9) sawNonzeroFneq = true;
        checked++;
      }
    }
    assert.strictEqual(checked, 256);
    assert.ok(sawNonzeroFneq, 'fixture must actually carry a nonzero fneq or this test proves nothing');
    assert.ok(maxErr < 1e-7, `finest-level f should reproduce dense f exactly; max |diff| = ${maxErr}`);
  });

  test('ghost cells are filled from the dense field, not left stale', () => {
    // L2 slot 0's ghost ring extends one L2 cell per GHOST layer beyond its
    // interior. Spot-check the left ghost column against the dense cell it
    // physically overlaps (target x = -1 wraps to TARGET_W-1).
    const p = injected.pools[2];
    const cells = p.MAX_FINE_BLOCKS * FB * FB;
    const vel = b64ToFloat32(p.velB64, cells * 2);
    for (const ly of [0, 5, 15]) {
      const idx = 0 * FB * FB + (GHOST + ly) * FB + (GHOST - 1);
      const gx = (TARGET_W - 1) % TARGET_W;
      assert.ok(Math.abs(vel[idx * 2] - denseUx(gx, ly)) < 1e-6,
        `ghost ux at ly=${ly}: ${vel[idx * 2]} vs dense ${denseUx(gx, ly)}`);
    }
  });

  // The round-trip above is BLIND to the rescale at coarse levels: fneq sums
  // to zero over the 9 directions, so scaling it cannot change rho, and
  // velocity is read from velBuf rather than from f. A wrong rescale factor
  // would therefore pass every assertion so far and still inject the wrong
  // stress state into every level above the finest. Check it directly.
  test('injected f applies the composed fneq rescale at a coarse level', () => {
    const p = injected.pools[1];
    const cells = p.MAX_FINE_BLOCKS * FB * FB;
    const f = b64ToFloat32(p.fB64, cells * 9);
    const rescale = fneqRescale(TAU0, NUM_LEVELS - 1, 1);
    assert.ok(Math.abs(rescale - 1) > 0.1, `fixture must exercise a non-trivial rescale, got ${rescale}`);

    // L1 slot 0, quadrant (1,0) (no L2 child): local interior cell (lx,ly)
    // with lx in [RB,2RB) maps to the dense 2x2 block at target (2*lx, 2*ly).
    let checked = 0, maxErr = 0;
    for (let ly = 0; ly < 2 * RB; ly += 3) {
      for (let lx = RB; lx < 2 * RB; lx += 3) {
        const idx = 0 * FB * FB + (GHOST + ly) * FB + (GHOST + lx);
        const tx = 2 * lx, ty = 2 * ly;
        const exp = boxAverage(tx, ty, 2);
        // Independent arithmetic mean of fneq over the same 2x2 footprint.
        for (let i = 0; i < 9; i++) {
          let fneqAvg = 0;
          for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) fneqAvg += denseFneq(tx + dx, ty + dy, i);
          fneqAvg /= 4;
          const want = feq(exp.rho, exp.ux, exp.uy, i) + rescale * fneqAvg;
          maxErr = Math.max(maxErr, Math.abs(f[i * cells + idx] - want));
        }
        checked++;
      }
    }
    assert.ok(checked > 0);
    assert.ok(maxErr < 1e-7, `coarse-level f should be feq(avg) + rescale*avg(fneq); max |diff| = ${maxErr}`);
  });

  // ── The rescale itself ────────────────────────────────────────────────────
  test('fneqRescale composes single amr_average_f2c hops exactly', () => {
    for (const tau0 of [0.5005, 0.5045, 0.509, 0.55]) {
      for (let src = 0; src <= 4; src++) {
        for (let dst = 0; dst <= src; dst++) {
          // Compose the shader's own per-hop factor 2*tau_coarse/tau_fine.
          let composed = 1;
          for (let L = src; L > dst; L--) {
            composed *= 2 * tauAtLevel(tau0, L - 1) / tauAtLevel(tau0, L);
          }
          const got = fneqRescale(tau0, src, dst);
          assert.ok(Math.abs(got - composed) < 1e-12 * Math.max(1, composed),
            `tau0=${tau0} ${src}->${dst}: ${got} vs composed ${composed}`);
        }
      }
    }
    assert.strictEqual(fneqRescale(0.509, 2, 2), 1, 'same-level rescale must be exactly 1');
  });

  // ── Failure modes must fail loudly ────────────────────────────────────────
  test('a resolution mismatch is rejected with an actionable message', () => {
    const wrong = { ...buildDenseSnapshot(), W: 32, H: 32 };
    assert.throws(() => injectDenseIntoAMRSnapshot({ denseSnapshot: wrong, amrSnapshot: buildAMRSnapshot() }),
      /finest\s+resolution|res=/);
  });

  test('an orphaned active slot is reported, not silently left stale', () => {
    // Mark an L2 slot active in slotToBlock without any blockSlot entry
    // pointing at it -- the shape a leaked allocation would take.
    const bad = buildAMRSnapshot();
    bad.pools[2].slotToBlock = [0, 5, 9, -1]; // slot 2 claims block 9, unreachable
    assert.throws(() => injectDenseIntoAMRSnapshot({ denseSnapshot: buildDenseSnapshot(), amrSnapshot: bad }),
      /not reachable from the L1 roots/);
  });

  test('a snapshot with an implausible L0 tau is rejected', () => {
    const bad = buildAMRSnapshot();
    bad.cardState[19] = 0.4;
    assert.throws(() => injectDenseIntoAMRSnapshot({ denseSnapshot: buildDenseSnapshot(), amrSnapshot: bad }),
      /BGK floor/);
  });

  console.log();
  if (failures.length) {
    console.log(`dense-to-amr: FAILED (${failures.length} failing, ${passed} passing)`);
    process.exit(1);
  }
  console.log(`dense-to-amr: ${passed} test(s) passed`);
}

main();
