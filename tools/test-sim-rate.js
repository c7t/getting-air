#!/usr/bin/env node
// Pure-Node assertions for sim-rate.mjs. No GPU, no browser, no server --
// run it with `make test` (or `node tools/test-sim-rate.js`).
//
// WHY: the pacer exists to satisfy four specific requirements, and every one
// of them is arithmetic on wall-clock deltas, so none of them needs a GPU to
// check:
//
//   1. the desktop runs at half its old rate,
//   2. the frame rate is not lowered (the pacer never asks a frame to wait),
//   3. a device that could not already keep up is not throttled at all,
//   4. a FASTER machine does the same physics per second rather than more.
//
// (3) is the one worth having a test for. It is not a property of the target
// rate, it is a property of the CEILING -- a slow device asks for more steps
// than the ceiling every frame and gets the ceiling, which is exactly what it
// did before the pacer existed. That is easy to break later by "simplifying"
// the clamp away, and the symptom would be a phone that got slower with no
// error anywhere.
//
// Same CommonJS-script-that-dynamic-imports-an-.mjs shape as
// tools/test-card-params.js.

const assert = require('assert');
const path = require('path');

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok    ${name}`);
  } catch (e) {
    failures.push({ name, e });
    console.log(`  FAIL  ${name}`);
    console.log(String(e.message).split('\n').map(l => `        ${l}`).join('\n'));
  }
}

function close(actual, expected, rel, msg) {
  const tol = expected === 0 ? rel : Math.abs(expected) * rel;
  assert.ok(Math.abs(actual - expected) <= tol,
    `${msg}: got ${actual}, expected ${expected} (|diff|=${Math.abs(actual - expected)} > tol=${tol})`);
}

// Run `seconds` of wall clock at a fixed frame interval, returning the total
// steps dispatched and the per-frame counts.
function simulate(pacer, { fps, seconds, stepsPerTU = 320, startMs = 1000 }) {
  const dt = 1000 / fps;
  const counts = [];
  let t = startMs;
  pacer.stepsForFrame(t, stepsPerTU);           // first call only seeds the clock
  const frames = Math.round(seconds * fps);
  for (let i = 0; i < frames; i++) {
    t += dt;
    counts.push(pacer.stepsForFrame(t, stepsPerTU));
  }
  return { total: counts.reduce((a, b) => a + b, 0), counts, seconds };
}

async function main() {
  const M = await import(path.join(__dirname, '..', 'sim-rate.mjs'));
  const { createSimPacer, parseSimRate, DEFAULT_TU_PER_SEC } = M;

  const STEPS_PER_TU = 320;          // A/U_T at the shipped defaults (16/0.05)
  const CEILING = 64;                // the old STEPS_PER_FRAME
  const mk = (rate = DEFAULT_TU_PER_SEC) =>
    createSimPacer({ maxStepsPerFrame: CEILING, tuPerSec: rate });

  // ── 1. The shipped default really is half the measured desktop rate ───────
  // 1465 steps/s was measured on the desktop AMR page (and independently
  // confirmed by telemetry.log at 1465). Half is 732.5; the default is
  // rounded onto the slider's 0.05 grid, so allow that 0.4%.
  test('default rate is half the measured 1465 steps/s desktop rate', () => {
    close(DEFAULT_TU_PER_SEC * STEPS_PER_TU, 1465 / 2, 0.01, 'target steps/s');
  });

  // ── 2. The rate is the invariant, not the frame count ────────────────────
  test('achieved rate is independent of frame rate', () => {
    const want = DEFAULT_TU_PER_SEC * STEPS_PER_TU;
    for (const fps of [30, 60, 90, 144, 240]) {
      const r = simulate(mk(), { fps, seconds: 10 });
      close(r.total / r.seconds, want, 0.02, `steps/s at ${fps}fps`);
    }
  });

  test('achieved rate scales with A/U_T, so it is resolution-independent', () => {
    for (const spt of [80, 320, 1280]) {
      const r = simulate(mk(), { fps: 60, seconds: 10, stepsPerTU: spt });
      close(r.total / r.seconds, DEFAULT_TU_PER_SEC * spt, 0.02, `steps/s at A/U_T=${spt}`);
    }
  });

  test('setRate changes the achieved rate proportionally', () => {
    for (const rate of [0.5, 1.0, 2.3, 6.0]) {
      const r = simulate(mk(rate), { fps: 60, seconds: 10 });
      close(r.total / r.seconds, rate * STEPS_PER_TU, 0.02, `steps/s at rate=${rate}`);
    }
  });

  // ── 3. THE INVARIANTS THAT BREAK THINGS SILENTLY ─────────────────────────
  // Both pages flip a ping-pong buffer once per step and rely on useB being
  // back to its initial value at a frame boundary (main-amr.js's snapshot
  // code states the invariant). An odd count leaves f_b authoritative, which
  // corrupts snapshot save/load without erroring anywhere.
  test('every step count is even, at every frame rate and rate', () => {
    for (const fps of [15, 30, 60, 144, 240]) {
      for (const rate of [0.25, 2.3, 12]) {
        for (const n of simulate(mk(rate), { fps, seconds: 5 }).counts) {
          assert.ok(n % 2 === 0, `odd step count ${n} at ${fps}fps rate=${rate}`);
        }
      }
    }
  });

  test('step count never exceeds the ceiling', () => {
    for (const fps of [1, 5, 15, 60, 240]) {
      for (const rate of [2.3, 12, 1000]) {   // 1000 = far beyond any device
        for (const n of simulate(mk(rate), { fps, seconds: 5 }).counts) {
          assert.ok(n <= CEILING, `count ${n} exceeds ceiling ${CEILING} at ${fps}fps rate=${rate}`);
        }
      }
    }
  });

  // ── 4. THE MOBILE REQUIREMENT ────────────────────────────────────────────
  // The phone measured 212 steps/s at 3.3 fps -- i.e. it was already only
  // managing 64 steps per frame. Under the pacer it must still get exactly
  // 64 every frame: it asks for far more, and the ceiling gives it what it
  // was already doing. If this ever fails, the phone got SLOWER.
  test('a device below the target is not throttled at all (phone: 3.3fps)', () => {
    const r = simulate(mk(), { fps: 3.3, seconds: 60 });
    for (const n of r.counts) {
      assert.strictEqual(n, CEILING,
        `phone frame got ${n} steps, expected the full ceiling ${CEILING}`);
    }
    // And therefore its throughput is unchanged from the old fixed loop.
    close(r.total / r.seconds, 3.3 * CEILING, 0.02, 'phone steps/s');
  });

  test('the margin holds even if the phone gets 3x faster', () => {
    const r = simulate(mk(), { fps: 10, seconds: 30 });
    for (const n of r.counts) assert.strictEqual(n, CEILING, `got ${n} at 10fps`);
  });

  // The crossover: a device fast enough to reach the target starts being
  // paced. 736 steps/s / 64 per frame = 11.5 fps.
  test('pacing begins only above the crossover frame rate', () => {
    const crossover = (DEFAULT_TU_PER_SEC * STEPS_PER_TU) / CEILING;   // ~11.5 fps
    const below = simulate(mk(), { fps: crossover * 0.8, seconds: 20 });
    assert.ok(below.counts.every(n => n === CEILING),
      'below crossover the device should still run flat out');
    const above = simulate(mk(), { fps: crossover * 3, seconds: 20 });
    assert.ok(above.counts.some(n => n < CEILING),
      'above crossover the pacer should be limiting the step count');
  });

  // ── 5. No banked time, no catch-up bursts ────────────────────────────────
  test('a long stall does not bank time into a burst', () => {
    const p = mk();
    let t = 1000;
    p.stepsForFrame(t, STEPS_PER_TU);
    t += 30000;                                   // 30s frozen (tab backgrounded)
    const after = p.stepsForFrame(t, STEPS_PER_TU);
    assert.ok(after <= CEILING, `burst of ${after} steps after a stall`);
    // and the frames that follow are back to the normal paced count
    for (let i = 0; i < 10; i++) { t += 1000 / 60; }
    const r = simulate(p, { fps: 60, seconds: 5, startMs: t });
    close(r.total / r.seconds, DEFAULT_TU_PER_SEC * STEPS_PER_TU, 0.03, 'steps/s after a stall');
  });

  test('a device that cannot keep up accrues no debt it later repays', () => {
    // 20s at 5fps (well under the target), then the load lifts and it runs at
    // 60fps. The fast phase must be paced normally, not sped up to catch up.
    const p = mk();
    const slow = simulate(p, { fps: 5, seconds: 20 });
    assert.ok(slow.counts.every(n => n === CEILING), 'slow phase should be flat out');
    const fast = simulate(p, { fps: 60, seconds: 10, startMs: 1e6 });
    close(fast.total / fast.seconds, DEFAULT_TU_PER_SEC * STEPS_PER_TU, 0.03,
      'steps/s after recovering from a slow phase');
  });

  test('reset() clears both the clock and the accumulator', () => {
    const p = mk();
    let t = 1000;
    p.stepsForFrame(t, STEPS_PER_TU);
    p.reset();
    // After reset the next call only re-seeds the clock, so it dispatches
    // nothing however long the gap was.
    assert.strictEqual(p.stepsForFrame(t + 60000, STEPS_PER_TU), 0);
  });

  // ── 6. Faster machines cost the same physics, not more ───────────────────
  test('a 4x faster machine does the same steps/s, not 4x', () => {
    const slowBox = simulate(mk(), { fps: 60, seconds: 10 });
    const fastBox = simulate(mk(), { fps: 240, seconds: 10 });
    close(fastBox.total, slowBox.total, 0.03,
      'a 4x faster machine should do the same total work');
  });

  // ── 7. URL parsing ───────────────────────────────────────────────────────
  test('parseSimRate reads ?simRate= and rejects nonsense', () => {
    const P = (q) => parseSimRate(new URLSearchParams(q));
    close(P('simRate=5.5'), 5.5, 1e-12, 'explicit rate');
    assert.strictEqual(P(''), DEFAULT_TU_PER_SEC);
    assert.strictEqual(P('simRate=0'), DEFAULT_TU_PER_SEC);
    assert.strictEqual(P('simRate=-3'), DEFAULT_TU_PER_SEC);
    assert.strictEqual(P('simRate=nonsense'), DEFAULT_TU_PER_SEC);
  });

  test('an odd ceiling is rounded down to keep the parity guarantee', () => {
    const p = createSimPacer({ maxStepsPerFrame: 65, tuPerSec: 1000 });
    let t = 1000; p.stepsForFrame(t, STEPS_PER_TU);
    for (let i = 0; i < 20; i++) {
      t += 1000 / 60;
      const n = p.stepsForFrame(t, STEPS_PER_TU);
      assert.ok(n % 2 === 0 && n <= 64, `got ${n} with an odd ceiling`);
    }
  });

  console.log();
  if (failures.length) {
    console.log(`sim-rate: FAILED (${failures.length} failing, ${passed} passing)`);
    process.exit(1);
  }
  console.log(`sim-rate: ${passed} test(s) passed`);
}

main().catch(e => { console.error(e); process.exit(1); });
