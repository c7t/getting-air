#!/usr/bin/env node
// GPU-free tests for card-total.mjs and its agreement with the wrap rule in
// shaders/physics.wgsl / amr_physics.wgsl.
//
// The point of the whole mechanism is that wrapping the f32 accumulator is
// INVISIBLE two ways at once: invisible to the simulation (off_x/off_y and
// the sub-cell cx/cy are bit-identical with and without it) and invisible to
// the host (the unwrapped total matches what an unwrapped accumulator would
// have held). Both directions are asserted here, against a JS mirror of the
// shader arithmetic evaluated in f32 via Math.fround -- the same
// "independently re-derive it rather than share a helper" convention
// main-cylinder-amr.js's debugCheckGeometryCoverage uses, so a change to one
// side cannot silently drag the other along with it.

const assert = require('assert');
const path = require('path');

let failures = 0;
function ok(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); }
catch (e) { failures++; console.log(`  FAIL  ${name}`); console.log(String(e.message).split('\n').map(l => '        ' + l).join('\n')); }
}

// --- f32 mirror of the shader ------------------------------------------
const f32 = Math.fround;

// shaders/physics.wgsl step 4 + 4b, in f32. Returns the new accumulator and
// the displacement it ACTUALLY applied this step -- which is the f32-rounded
// delta, not `v`. That distinction is the whole contract below: the host
// cannot recover more precision than the shader put in, so what it must
// reproduce is the sum of these deltas, not the exact analytic sum.
function shaderStep(total, v, wrap, { doWrap }) {
  const raw = f32(total + v);
  const delta = raw - total; // exact in f64: a difference of two f32 values
  let t = raw;
  if (doWrap) {
    if (t >= wrap) t = f32(t - wrap);
    else if (t <= -wrap) t = f32(t + wrap);
  }
  return { t, delta };
}
// shaders/physics.wgsl step 5, the two things the accumulator is used for.
function windowState(total, span) {
  const shift = Math.floor(total);
  return {
    off: ((shift % span) + span) % span,
    frac: f32(total - f32(shift)), // -> cx/cy's sub-cell part
  };
}

(async () => {
  const { createTotalUnwrapper, TOTAL_WRAP_SCREENS } =
    await import(path.join('file://', __dirname, '..', 'card-total.mjs'));

  const SPAN = 256;                       // W = H at every page's default
  const WRAP = SPAN * TOTAL_WRAP_SCREENS;

  console.log('== tools/test-card-total.js');

  ok('the wrap constant is a power of two, so the wrap itself is exact', () => {
    assert.ok(Number.isInteger(Math.log2(TOTAL_WRAP_SCREENS)),
      `TOTAL_WRAP_SCREENS=${TOTAL_WRAP_SCREENS} is not a power of two`);
    assert.ok(Number.isInteger(Math.log2(WRAP)), `wrap=${WRAP} is not a power of two`);
  });

  ok('wrapping leaves off_x/off_y and the sub-cell fraction bit-identical', () => {
    // The claim the whole change rests on: a wrap by a whole multiple of the
    // domain changes neither thing the shader reads out of the accumulator.
    for (let k = 0; k < 2000; k++) {
      const base = f32(WRAP - 3 + k * 0.0007);
      const wrapped = f32(base - WRAP);
      const a = windowState(base, SPAN);
      const b = windowState(wrapped, SPAN);
      assert.strictEqual(a.off, b.off, `off differs at ${base}`);
      assert.strictEqual(a.frac, b.frac, `sub-cell fraction differs at ${base}`);
    }
  });

  ok('the wrap subtraction introduces no rounding of its own', () => {
    // Only true because it fires just past the threshold, so both operands
    // are within a factor of two -- assert that, don't assume it.
    for (let k = 0; k < 5000; k++) {
      const t = f32(WRAP + k * 0.01);
      assert.strictEqual(f32(t - WRAP), t - WRAP, `inexact wrap at ${t}`);
    }
    for (let k = 0; k < 5000; k++) {
      const t = f32(-WRAP - k * 0.01);
      assert.strictEqual(f32(t + WRAP), t + WRAP, `inexact negative wrap at ${t}`);
    }
  });

  ok('unwrap reproduces the shader\'s own displacement exactly, across many wraps', () => {
    // The unwrapper's actual contract. It cannot invent precision the f32
    // accumulator never had, so the thing it must reproduce is the sum of the
    // deltas the shader really applied -- exactly, with no drift of its own.
    const u = createTotalUnwrapper(SPAN, SPAN);
    let wrapped = 0, truth = 0, last = 0;
    const VY = 0.04;
    for (let i = 0; i < 400000; i++) {   // many wraps over
      const r = shaderStep(wrapped, VY, WRAP, { doWrap: true });
      wrapped = r.t; truth += r.delta;
      last = u.unwrap(0, wrapped).y;
    }
    assert.ok(Math.abs(wrapped) <= WRAP, `accumulator escaped its bound: ${wrapped}`);
    assert.ok(Math.abs(last - truth) < 1e-6,
      `unwrapped ${last} vs the shader's own total ${truth}`);
    assert.ok(last > 15000, `sanity: expected ~16000 of descent, got ${last}`);
  });

  ok('bounding the accumulator does not make absolute drift worse', () => {
    // The wrap exists to stop precision decaying; check it does not trade
    // that against a worse absolute total. The unbounded accumulator's ULP
    // grows with the value, so it should be the one that drifts more.
    const VY = 0.04, N = 400000, exact = N * VY;
    let wrapped = 0, plain = 0;
    for (let i = 0; i < N; i++) {
      wrapped = shaderStep(wrapped, VY, WRAP, { doWrap: true }).t;
      plain = shaderStep(plain, VY, WRAP, { doWrap: false }).t;
    }
    const u = createTotalUnwrapper(SPAN, SPAN);
    let bounded = 0, w2 = 0;
    for (let i = 0; i < N; i++) { w2 = shaderStep(w2, VY, WRAP, { doWrap: true }).t; bounded = u.unwrap(0, w2).y; }
    const errBounded = Math.abs(bounded - exact);
    const errPlain = Math.abs(plain - exact);
    assert.ok(errBounded <= errPlain,
      `bounded drift ${errBounded} should not exceed unbounded ${errPlain}`);
  });

  ok('the bounded accumulator beats the unbounded one on sub-cell precision', () => {
    // The actual bug: the sub-cell fraction is what the ULP eats first. Run
    // both accumulators over the same displacement and compare the fraction
    // each yields against the exact one.
    const VY = 0.04, N = 3000000;
    let wrapped = 0, plain = 0;
    for (let i = 0; i < N; i++) {
      wrapped = shaderStep(wrapped, VY, WRAP, { doWrap: true }).t;
      plain = shaderStep(plain, VY, WRAP, { doWrap: false }).t;
    }
    const exact = N * VY;
    const errWrapped = Math.abs(windowState(wrapped, SPAN).frac - (exact - Math.floor(exact)));
    const errPlain = Math.abs(windowState(plain, SPAN).frac - (exact - Math.floor(exact)));
    assert.ok(errWrapped < errPlain,
      `bounded fraction err ${errWrapped} should beat unbounded ${errPlain}`);
    // And the unbounded one should be visibly bad by here -- if this ever
    // stops holding, the bug this guards has changed shape.
    assert.ok(errPlain > 0.01, `unbounded error only ${errPlain}; test no longer exercises the bug`);
  });

  ok('unwrap is exact when nothing has wrapped yet', () => {
    const u = createTotalUnwrapper(SPAN, SPAN);
    assert.strictEqual(u.unwrap(1.5, 2.5).y, 2.5);
    assert.strictEqual(u.unwrap(1.5, 3.5).y, 3.5);
    assert.strictEqual(u.unwrap(1.5, 3.5).x, 1.5);
  });

  ok('unwrap follows a card moving upward, across a negative wrap', () => {
    const u = createTotalUnwrapper(SPAN, SPAN);
    let wrapped = 0, truth = 0, last = 0;
    const VY = -0.04;
    for (let i = 0; i < 200000; i++) {
      const r = shaderStep(wrapped, VY, WRAP, { doWrap: true });
      wrapped = r.t; truth += r.delta;
      last = u.unwrap(0, wrapped).y;
    }
    assert.ok(Math.abs(wrapped) <= WRAP, `accumulator escaped its bound: ${wrapped}`);
    assert.ok(Math.abs(last - truth) < 1e-6, `upward total ${last} vs ${truth}`);
    assert.ok(last < -7000, `sanity: expected ~-8000, got ${last}`);
  });

  ok('a NaN passes through without corrupting the accumulator', () => {
    const u = createTotalUnwrapper(SPAN, SPAN);
    u.unwrap(0, 10);
    const bad = u.unwrap(0, NaN);
    assert.ok(Number.isNaN(bad.y), 'NaN should pass through to the caller\'s own check');
    // A recovered readback must not have been rebased against the NaN.
    assert.strictEqual(u.unwrap(0, 11).y, 11);
  });

  ok('reset re-seeds instead of unwrapping across a discontinuity', () => {
    const u = createTotalUnwrapper(SPAN, SPAN);
    u.unwrap(0, WRAP - 1);
    u.reset();
    // resetSim writes the accumulator back to 0; without reset() that would
    // look like a wrap and keep counting.
    assert.strictEqual(u.unwrap(0, 0).y, 0);
    assert.strictEqual(u.unwrap(0, 0.04).y, 0.04);
  });

  ok('x and y unwrap independently', () => {
    const u = createTotalUnwrapper(SPAN, SPAN);
    u.unwrap(WRAP - 0.5, 5);
    const r = u.unwrap(0.5, 5.5); // x wrapped, y did not
    // Pre-wrap x was 0.5 + WRAP, so the true total advanced by exactly 1.0.
    assert.ok(Math.abs(r.x - (WRAP + 0.5)) < 1e-6, `x should keep counting past the wrap, got ${r.x}`);
    assert.ok(Math.abs(r.y - 5.5) < 1e-6, `y should be untouched, got ${r.y}`);
  });

  console.log(failures
    ? `\ncard-total: ${failures} test(s) FAILED`
    : '\ncard-total: all tests passed');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
