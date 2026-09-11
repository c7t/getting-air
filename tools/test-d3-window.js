#!/usr/bin/env node
// GPU-free tests for d3-window.mjs -- the moving window's convention
// (plans/3D.md M8.3).
//
// WHAT MAKES THIS WORTH TESTING, given that the arithmetic is four lines. A
// moving window is the archetype of a feature that looks identical whether
// it is working or switched off: the picture is plausible either way, the
// solve stays finite either way, and the difference only shows up as a drag
// coefficient that converges instead of drifting -- thousands of steps and a
// GPU later. M8.0's standing lesson applies here too: FAILURE IS LATE.
//
// So the properties below are the ones a broken window would violate in the
// first step rather than the ten-thousandth:
//
//   - THE WINDOW HOLDS THE BODY AT THE ANCHOR. windowCoord(body) must equal
//     the anchor plus the body's own sub-cell part, for EVERY position the
//     body can reach including the ones straddling the buffer seam. This is
//     the whole claim; everything else is machinery for it.
//   - WITHOUT A WINDOW, EVERY FUNCTION IS THE IDENTITY. Not "close to", not
//     "in practice" -- exactly, so that every scenario predating M8.3 is
//     bit-identical rather than merely unaffected.
//   - THE OVERRIDE NAMES MATCH THE SHADER'S. An override a module does not
//     declare is a WebGPU validation error; one it declares and nobody sets
//     is a window that never moves. Same contract, and same failure mode, as
//     BODY_FIELDS against BodyState3D.
//
// Run: node tools/test-d3-window.js   (also picked up by `make test`)

const assert = require('assert');
const fs = require('fs');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}
const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b}`);

(async () => {
  const root = path.join(__dirname, '..');
  const W = await import(path.join(root, 'd3-window.mjs'));
  const {
    parseWindowAxes, wrapDelta, wrapDelta3, windowOffset, windowOffset3,
    windowCoord, windowCoord3, wrapPosition, wrapPosition3, wrapDims,
    windowConstants, WINDOW_OVERRIDES,
  } = W;
  const B = await import(path.join(root, 'd3-body.mjs'));
  const S = await import(path.join(root, 'd3-scenarios.mjs'));

  ok('?window= parses the axes it names and REFUSES what it does not know', () => {
    assert.deepStrictEqual(parseWindowAxes('x'), [1, 0, 0]);
    assert.deepStrictEqual(parseWindowAxes('zy'), [0, 1, 1]);
    assert.deepStrictEqual(parseWindowAxes('xyz'), [1, 1, 1]);
    assert.deepStrictEqual(parseWindowAxes('all'), [1, 1, 1]);
    assert.deepStrictEqual(parseWindowAxes('0'), [0, 0, 0]);
    assert.deepStrictEqual(parseWindowAxes('none'), [0, 0, 0]);
    // Absent falls back to the scenario's own choice; present and empty does not.
    assert.deepStrictEqual(parseWindowAxes(null, [1, 0, 0]), [1, 0, 0]);
    assert.deepStrictEqual(parseWindowAxes('0', [1, 0, 0]), [0, 0, 0]);
    // A typo is refused rather than silently ignored -- d3-scenarios.mjs
    // refuses an unknown scenario name for the same reason.
    assert.throws(() => parseWindowAxes('w'), /expected some of xyz/);
    assert.throws(() => parseWindowAxes('xq'), /expected some of xyz/);
  });

  ok('every function is the EXACT identity on an unwindowed axis', () => {
    for (const v of [-7.25, 0, 0.5, 191.75, 1e6]) {
      assert.strictEqual(wrapDelta(v, 0), v, `wrapDelta(${v}, 0)`);
      assert.strictEqual(windowCoord(v, 0, 0), v, `windowCoord(${v})`);
      assert.strictEqual(wrapPosition(v, 0), v, `wrapPosition(${v})`);
      assert.strictEqual(windowOffset(v, 17, 0), 0, `windowOffset(${v})`);
    }
    // And a negative size reads as "off" too, so a caller that computed a
    // size wrongly gets no window rather than a wrap by a negative period.
    assert.strictEqual(wrapDelta(5, -192), 5);
    assert.deepStrictEqual(wrapDims([192, 72, 72], [0, 0, 0]), [0, 0, 0]);
    assert.deepStrictEqual(wrapDims([192, 72, 72], [1, 0, 1]), [192, 0, 72]);
  });

  ok('wrapDelta takes the NEAREST image, on both sides and at the half-period', () => {
    const n = 192;
    close(wrapDelta(3, n), 3, 0, 'inside');
    close(wrapDelta(-3, n), -3, 0, 'inside, negative');
    // 189 cells one way is 3 cells the other, which is the case a body that
    // has wrapped through the seam presents to every cell behind it.
    close(wrapDelta(189, n), -3, 0, 'past the half period');
    close(wrapDelta(-189, n), 3, 0, 'past the half period, negative');
    close(wrapDelta(192, n), 0, 0, 'a whole period is no displacement');
    // Never further than half a period, whatever is asked.
    for (let d = -1000; d <= 1000; d += 7) {
      assert.ok(Math.abs(wrapDelta(d, n)) <= n / 2 + 1e-9, `|wrapDelta(${d})| <= n/2`);
      // And it differs from the original by a whole number of periods.
      close(Math.round((d - wrapDelta(d, n)) / n) * n, d - wrapDelta(d, n), 1e-9, `periodicity at ${d}`);
    }
  });

  // THE CLAIM. Everything above is machinery for this one.
  ok('THE WINDOW HOLDS THE BODY AT THE ANCHOR, at every position it can reach', () => {
    const n = 192, anchor = 168;
    // Walk a body all the way round the buffer twice, at a speed that is not
    // a divisor of anything, so the sub-cell part takes many values and the
    // seam is crossed at several different phases.
    let c = anchor + 0.0;
    for (let k = 0; k < 5000; k++) {
      c = wrapPosition(c + 0.0413, n);
      const off = windowOffset(c, anchor, n);
      const w = windowCoord(c, off, n);
      // The body sits at the anchor cell, offset by its own sub-cell part.
      close(w, anchor + (c - Math.floor(c)), 1e-9, `step ${k} (c=${c.toFixed(3)})`);
      assert.ok(w >= anchor && w < anchor + 1, `step ${k}: window coord must stay in the anchor cell`);
    }
  });

  ok('windowCoord maps the whole buffer onto [0, n) exactly once', () => {
    const n = 64;
    for (const off of [0, 1, 31, 63, -5, 129]) {
      const seen = new Set();
      for (let b = 0; b < n; b++) {
        const w = windowCoord(b, off, n);
        assert.ok(w >= 0 && w < n, `off=${off} b=${b}: ${w} out of range`);
        assert.ok(Number.isInteger(w), `off=${off} b=${b}: an integer buffer cell must map to an integer`);
        assert.ok(!seen.has(w), `off=${off}: window coord ${w} claimed twice`);
        seen.add(w);
      }
      assert.strictEqual(seen.size, n, `off=${off}: every window cell must be claimed`);
    }
  });

  ok('the offset is EXACTLY ZERO at the anchor, so a windowed run starts unwindowed', () => {
    // The property the whole "anchor = the body's initial cell" choice buys:
    // step 0 of a windowed run is bit-identical to step 0 of an unwindowed
    // one, which is what makes a window/no-window A/B a real control.
    const n = 192, anchor = 168;
    for (const frac of [0, 0.5, 0.999]) {
      assert.strictEqual(windowOffset(anchor + frac, anchor, n), 0, `frac=${frac}`);
      for (const b of [0, 37, 191]) assert.strictEqual(windowCoord(b, 0, n), b, `b=${b}`);
    }
  });

  ok('wrapPosition keeps a body inside the buffer, from either direction', () => {
    const n = 192;
    close(wrapPosition(191.5, n), 191.5, 0, 'inside');
    close(wrapPosition(192.25, n), 0.25, 1e-9, 'off the far end');
    close(wrapPosition(-0.25, n), 191.75, 1e-9, 'off the near end');
    close(wrapPosition(-192.5, n), 191.5, 1e-9, 'more than a period back');
    // Fixed point: wrapping twice is wrapping once.
    for (let v = -500; v < 500; v += 3.7) {
      close(wrapPosition(wrapPosition(v, n), n), wrapPosition(v, n), 1e-9, `idempotent at ${v}`);
    }
  });

  ok('the vector forms are the scalar form, per axis, with the axes independent', () => {
    const wrapN = [192, 0, 72];
    assert.deepStrictEqual(wrapDelta3([189, 189, 69], wrapN), [-3, 189, -3]);
    assert.deepStrictEqual(windowOffset3([10.5, 10.5, 10.5], [4, 4, 4], wrapN), [6, 0, 6]);
    assert.deepStrictEqual(wrapPosition3([200, 200, 80], wrapN), [8, 200, 8]);
    // The unwindowed middle axis is untouched in all three, which is what
    // makes `?window=x` on a domain that is narrow in y and z safe.
  });

  // A CHECK ONLY EVER RUN ON CORRECT INPUT IS INDISTINGUISHABLE FROM ONE THAT
  // RETURNS NOTHING -- the same reason tools/test-d3-amr.js scores
  // check21Balance against trees that VIOLATE the invariant. The mutation
  // here is the obvious wrong implementation: a window that pans by the
  // body's position without subtracting the anchor.
  ok('the anchor invariant FAILS for an anchorless offset (the check can fail)', () => {
    const n = 192, anchor = 168;
    const c = anchor + 40.25;
    const wrong = windowCoord(c, Math.floor(c), n);   // offset without the anchor
    const right = windowCoord(c, windowOffset(c, anchor, n), n);
    close(right, anchor + 0.25, 1e-9, 'the real one still holds');
    assert.ok(Math.abs(wrong - (anchor + 0.25)) > 1,
      'an offset that drops the anchor must NOT satisfy the invariant');
  });

  ok('the override names match shaders/common_d3_window.wgsl exactly', () => {
    const src = fs.readFileSync(path.join(root, 'shaders', 'common_d3_window.wgsl'), 'utf8');
    const declared = [...src.matchAll(/^override\s+(WIN_\w+)\s*:/gm)].map(m => m[1]);
    assert.deepStrictEqual(declared.slice().sort(), WINDOW_OVERRIDES.slice().sort(),
      'the shader\'s WIN_* overrides and WINDOW_OVERRIDES must be the same set');
    const c = windowConstants([192, 0, 72], [168, 36, 36]);
    assert.deepStrictEqual(c, { WIN_NX: 192, WIN_NY: 0, WIN_NZ: 72, WIN_AX: 168, WIN_AY: 36, WIN_AZ: 36 });
    // Every declared override gets a value: one left unset is a window that
    // silently never moves on that axis.
    for (const k of declared) assert.ok(k in c, `${k} has no value`);
  });

  ok('stepFreeBody wraps a body into the buffer and keeps its true displacement', () => {
    const { SHAPE, makeBodyState, stepFreeBody } = B;
    const n = 32;
    let s = makeBodyState({ shape: { kind: SHAPE.SPHERE, a: 2 }, x: [30, 5, 5], v: [1, 0, 0] });
    assert.deepStrictEqual(s.d, [0, 0, 0], 'displacement starts at zero');
    for (let k = 0; k < 10; k++) s = stepFreeBody(s, { wrap: [n, 0, 0] });
    // Ten unit steps from x = 30 in a 32-wide buffer: wrapped to 8, travelled 10.
    close(s.x[0], 8, 1e-9, 'position wrapped into the buffer');
    close(s.d[0], 10, 1e-9, 'displacement is the TRUE distance, not the wrapped one');
    // The unwindowed axes are untouched, and so is the unwindowed default.
    let u = makeBodyState({ shape: { kind: SHAPE.SPHERE, a: 2 }, x: [30, 5, 5], v: [1, 0, 0] });
    for (let k = 0; k < 10; k++) u = stepFreeBody(u);
    close(u.x[0], 40, 1e-9, 'no wrap by default');
    close(u.d[0], 10, 1e-9, 'and the displacement agrees with the position');
  });

  ok('the fall scenario windows the moving frames and NOT the pinned control', () => {
    const tow = S.resolveScenario('fall', { tow: 0.04 });
    const str = S.resolveScenario('fall', { stream: 0.04 });
    const free = S.resolveScenario('fall', {});
    assert.deepStrictEqual(tow.window, [1, 0, 0], 'a towed body is followed');
    assert.deepStrictEqual(free.window, [1, 0, 0], 'so is a falling one');
    assert.deepStrictEqual(str.window, [0, 0, 0], 'a pinned streamed body is not');
    // THE GALILEAN PAIR IS THE SAME ARRANGEMENT MIRRORED: 2n of undisturbed
    // fluid between the body and the face it faces, 14n for the wake behind
    // it. That is what makes Cd(tow) and Cd(stream) a comparison of the
    // coupling rather than of two domains (plans/3D.md M8.2b).
    const n = tow.D, L = tow.dims[0];
    close(str.body.x[0], 2 * n, 1e-9, 'streamed: 2n from the inlet');
    close(tow.body.x[0], L - 2 * n, 1e-9, 'towed: 2n from the far face it travels toward');
    close(L - str.body.x[0], tow.body.x[0], 1e-9, 'the two are mirror images');
    // And the sponge is what the window needs to exist at all.
    assert.ok(tow.sponge.width > 0, 'a windowed run needs an absorbing band');
  });

  console.log(`\nd3-window: ${pass} test(s) passed`);
})();
