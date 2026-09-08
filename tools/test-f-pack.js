#!/usr/bin/env node
// GPU-free tests for f-pack.mjs, the host side of the packed-halves `f`
// storage introduced with shaders/common_fpack.wgsl.
//
// The thing actually at risk here is AGREEMENT WITH THE SHADER, not internal
// consistency: the host packs the initial equilibrium and un-packs snapshots,
// the GPU does everything in between, and a disagreement about which half of
// a word holds which plane would not crash -- it would quietly transpose the
// lattice directions and produce a plausible-looking wrong flow. So these
// tests re-implement common_fpack.wgsl's fIdx/fUnpack addressing directly and
// assert the host's packF lands where the shader will look, rather than only
// checking that packF and unpackF agree with each other.
//
// Run: node tools/test-f-pack.js   (also picked up by `make test`)

const assert = require('assert');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}

(async () => {
  const FP = await import('../f-pack.mjs');
  const { f32ToF16Bits, f16BitsToF32, packF, unpackF, fWords } = FP;

  // --- binary16 encoding ---------------------------------------------------
  ok('exact bit patterns for known values', () => {
    assert.strictEqual(f32ToF16Bits(0), 0x0000);
    assert.strictEqual(f32ToF16Bits(-0), 0x8000);
    assert.strictEqual(f32ToF16Bits(1), 0x3c00);
    assert.strictEqual(f32ToF16Bits(-2), 0xc000);
    assert.strictEqual(f32ToF16Bits(0.5), 0x3800);
    assert.strictEqual(f32ToF16Bits(65504), 0x7bff);        // largest finite
    assert.strictEqual(f32ToF16Bits(Infinity), 0x7c00);
  });

  ok('overflow saturates to Inf, underflow to zero', () => {
    assert.strictEqual(f32ToF16Bits(1e5), 0x7c00);
    assert.strictEqual(f32ToF16Bits(-1e5), 0xfc00);
    assert.strictEqual(f32ToF16Bits(1e-9), 0x0000);         // below 2^-25
    assert.strictEqual(f32ToF16Bits(-1e-9), 0x8000);
  });

  ok('subnormals are encoded, not flushed', () => {
    // Smallest positive subnormal is 2^-24; the largest is 1023*2^-24.
    assert.strictEqual(f32ToF16Bits(Math.pow(2, -24)), 0x0001);
    assert.strictEqual(f32ToF16Bits(1023 * Math.pow(2, -24)), 0x03ff);
    // Smallest normal, 2^-14, must NOT come back as a subnormal.
    assert.strictEqual(f32ToF16Bits(Math.pow(2, -14)), 0x0400);
  });

  ok('subnormal ties round to even too', () => {
    // The subnormal branch has its own rounding code, and a mutation that
    // made it round ties away from zero survived every other assertion here.
    // Subnormal quantum is 2^-24; a tie sits half way between two multiples.
    const q = Math.pow(2, -24);
    assert.strictEqual(f32ToF16Bits(2.5 * q), 0x0002);   // tie between 2 and 3 -> even
    assert.strictEqual(f32ToF16Bits(3.5 * q), 0x0004);   // tie between 3 and 4 -> even
    assert.strictEqual(f32ToF16Bits(0.5 * q), 0x0000);   // tie between 0 and 1 -> even
    assert.strictEqual(f32ToF16Bits(1.5 * q), 0x0002);   // tie between 1 and 2 -> even
  });

  ok('rounds to nearest, ties to even', () => {
    // 1 + 2^-11 is exactly half way between 0x3c00 (1.0) and 0x3c01.
    // Tie -> even -> 0x3c00.
    assert.strictEqual(f32ToF16Bits(1 + Math.pow(2, -11)), 0x3c00);
    // 1 + 3*2^-11 is half way between 0x3c01 and 0x3c02; even is 0x3c02.
    assert.strictEqual(f32ToF16Bits(1 + 3 * Math.pow(2, -11)), 0x3c02);
    // Just past the tie rounds up.
    assert.strictEqual(f32ToF16Bits(1 + Math.pow(2, -11) * 1.01), 0x3c01);
  });

  ok('NaN stays NaN', () => {
    const b = f32ToF16Bits(NaN);
    assert.strictEqual((b >>> 10) & 0x1f, 0x1f, 'exponent must be all ones');
    assert.ok((b & 0x3ff) !== 0, 'mantissa must be non-zero (Inf would be a silent corruption)');
    assert.ok(Number.isNaN(f16BitsToF32(b)));
  });

  // Independent oracle where the engine provides one. Inputs are narrowed to
  // f32 first: f32ToF16Bits rounds THROUGH single precision (see its comment),
  // which is the semantics the GPU has, while Math.f16round rounds a double in
  // one step. They disagree by one ulp exactly when the f32 value lands on an
  // f16 tie -- comparing raw doubles would be testing the wrong contract.
  ok('agrees with Math.f16round on f32 inputs', function () {
    if (typeof Math.f16round !== 'function') { console.log('       (skipped: no Math.f16round)'); return; }
    for (let k = 0; k < 50000; k++) {
      const v = Math.fround((Math.random() * 2 - 1) * Math.pow(10, Math.floor(Math.random() * 12) - 8));
      const mine = f16BitsToF32(f32ToF16Bits(v));
      const theirs = Math.f16round(v);
      assert.ok(Object.is(mine, theirs), `${v}: ${mine} vs ${theirs}`);
    }
  });

  ok('rounds through f32, not straight from the double', () => {
    // The one place the two conventions visibly differ; pinned so the choice
    // is a decision on record rather than an accident.
    const v = 0.40246582615253534;                    // f32 lands exactly on an f16 tie
    assert.strictEqual(f16BitsToF32(f32ToF16Bits(v)), 0.40234375);
    if (typeof Math.f16round === 'function') {
      assert.strictEqual(Math.f16round(v), 0.402587890625);
    }
    // ...and for anything already f32 (which is every real caller) they agree.
    assert.strictEqual(f16BitsToF32(f32ToF16Bits(Math.fround(v))), 0.40234375);
  });

  ok('decode inverts encode for every finite binary16', () => {
    for (let b = 0; b < 0x10000; b++) {
      if (((b >>> 10) & 0x1f) === 0x1f) continue;            // Inf/NaN
      const v = f16BitsToF32(b);
      assert.strictEqual(f32ToF16Bits(v), b, `bits ${b.toString(16)} -> ${v}`);
    }
  });

  // --- layout --------------------------------------------------------------
  ok('f16 off is a true no-op (same object, no copy)', () => {
    const f = new Float32Array(9 * 4).map((_, i) => i + 0.5);
    assert.strictEqual(packF(f, 4, false), f);
    assert.strictEqual(unpackF(f, 4, false), f);
    assert.strictEqual(fWords(false), 9);
    assert.strictEqual(fWords(true), 5);
  });

  // Re-implementation of common_fpack.wgsl's addressing. If these two drift,
  // this is where it shows up.
  const shaderIdx = (i, S, c) => (i >> 1) * S + c;
  const shaderUnpack = (w, i) => f16BitsToF32((i & 1) ? (w >>> 16) : (w & 0xffff));

  ok('packF lands where the shader reads (fIdx/fUnpack agreement)', () => {
    const N = 7;
    const f = new Float32Array(9 * N);
    // Distinct, exactly-representable values so a transposition cannot hide.
    for (let i = 0; i < 9; i++) for (let c = 0; c < N; c++) f[i * N + c] = i + c / 8;
    const w = packF(f, N, true);
    assert.strictEqual(w.length, 5 * N);
    for (let i = 0; i < 9; i++) {
      for (let c = 0; c < N; c++) {
        assert.strictEqual(shaderUnpack(w[shaderIdx(i, N, c)], i), f[i * N + c],
          `plane ${i} cell ${c}`);
      }
    }
  });

  ok('word 4 high half is inert padding holding plane 8', () => {
    // common_fpack.wgsl clamps fHi(4) to 8 rather than branching, so the high
    // half of the last word duplicates plane 8. Nothing reads it; asserted so
    // that a future reader finds the convention stated rather than inferred.
    const N = 3;
    const f = new Float32Array(9 * N);
    for (let i = 0; i < 9; i++) for (let c = 0; c < N; c++) f[i * N + c] = i + 1;
    const w = packF(f, N, true);
    for (let c = 0; c < N; c++) {
      assert.strictEqual(f16BitsToF32(w[4 * N + c] >>> 16), 9);
      assert.strictEqual(f16BitsToF32(w[4 * N + c] & 0xffff), 9);
    }
  });

  ok('pack/unpack round-trip equals elementwise f16 rounding', () => {
    const N = 64;
    const f = new Float32Array(9 * N);
    for (let k = 0; k < f.length; k++) f[k] = (Math.random() * 2 - 1) * 0.5;
    const back = unpackF(packF(f, N, true), N, true);
    assert.strictEqual(back.length, f.length);
    for (let k = 0; k < f.length; k++) {
      assert.strictEqual(back[k], f16BitsToF32(f32ToF16Bits(f[k])), `index ${k}`);
    }
  });

  // --- the values this actually has to carry -------------------------------
  ok('D2Q9 equilibrium weights survive within fp16 precision', () => {
    // The initial condition every page writes: rho=1, u=0, so f_i = w_i.
    // These are the values packF sees at startup, and the smallest of them
    // (1/36) is the one closest to the precision floor.
    const WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    const N = 1;
    const f = new Float32Array(9);
    for (let i = 0; i < 9; i++) f[i] = WT[i];
    const back = unpackF(packF(f, N, true), N, true);
    for (let i = 0; i < 9; i++) {
      const rel = Math.abs(back[i] - WT[i]) / WT[i];
      assert.ok(rel < 4.884e-4, `w[${i}] relative error ${rel} exceeds fp16's 2^-11`);
      assert.ok(rel > 0, `w[${i}] round-tripped exactly, which means quantisation did not happen`);
    }
  });

  ok('fneq-scale deviations near the body stay well above the noise floor', () => {
    // plans/perf-characterization.md's accuracy argument in numbers: at peak
    // shear fneq/f ~ 1.3e-2, so quantising f must not eat it. Take f = w_i
    // (1 +- 1.3e-2) and check the recovered deviation is accurate to better
    // than 1%.
    const WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    for (let i = 0; i < 9; i++) {
      const dev = WT[i] * 1.31e-2;
      const got = f16BitsToF32(f32ToF16Bits(WT[i] + dev)) - f16BitsToF32(f32ToF16Bits(WT[i]));
      assert.ok(Math.abs(got - dev) / dev < 0.01,
        `direction ${i}: recovered deviation ${got} vs ${dev}`);
    }
  });

  // --- mode 2: store the deviation from the lattice weight ----------------
  const WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

  ok('mode 2 packs where the shader reads (fIdx/fUnpack agreement)', () => {
    // Same re-implementation of common_fpack.wgsl as above, plus its mode-2
    // `+ wt[i]` on unpack. Getting the weight onto the wrong half or the
    // wrong plane would bias individual lattice directions -- a momentum
    // source, not noise -- so this checks per-plane, not in aggregate.
    const N = 5;
    const f = new Float32Array(9 * N);
    for (let i = 0; i < 9; i++) for (let c = 0; c < N; c++) f[i * N + c] = WT[i] * (1 + (c - 2) * 0.03125);
    const w = packF(f, N, 2);
    const shaderUnpack2 = (word, i) =>
      f16BitsToF32((i & 1) ? (word >>> 16) : (word & 0xffff)) + WT[i];
    for (let i = 0; i < 9; i++) {
      for (let c = 0; c < N; c++) {
        const got = shaderUnpack2(w[shaderIdx(i, N, c)], i);
        const dev = f[i * N + c] - WT[i];
        const tol = Math.abs(dev) * Math.pow(2, -10) + WT[i] * 2e-7; // one fp16 ulp of the
                                                    // DEVIATION, plus the f32 rounding of w_i itself
        assert.ok(Math.abs(got - f[i * N + c]) < tol,
          `plane ${i} cell ${c}: ${got} vs ${f[i * N + c]} (tol ${tol})`);
      }
    }
  });

  ok('mode 2 round-trips through unpackF', () => {
    const N = 32;
    const f = new Float32Array(9 * N);
    for (let i = 0; i < 9; i++) for (let c = 0; c < N; c++) f[i * N + c] = WT[i] * (1 + (Math.random() - 0.5) * 0.25);
    const back = unpackF(packF(f, N, 2), N, 2);
    for (let k = 0; k < f.length; k++) {
      const wi = WT[Math.floor(k / N)];
      const dev = f[k] - wi;
      // Half an ulp of the deviation, plus fp16's ABSOLUTE subnormal floor
      // (2^-24): a deviation that lands near zero is subnormal, where the
      // quantum stops shrinking with the value. Omitting that term made this
      // pass most runs and fail occasionally, which is worse than failing.
      const tol = Math.abs(dev) * Math.pow(2, -10) + wi * 2e-7 + Math.pow(2, -24);
      assert.ok(Math.abs(back[k] - f[k]) < tol, `index ${k}: dev ${dev}, err ${back[k] - f[k]}`);
    }
  });

  ok('mode 2 buys real resolution on the part that carries the flow', () => {
    // The reason mode 1 was not enough, as a number. f_i sits near w_i, so
    // mode 1 spends its mantissa on the weight; mode 2 rescales to the
    // deviation. Measured here as the recovered-deviation error ratio, which
    // should be several-fold for every direction and ~8x for the rest link.
    for (let i = 0; i < 9; i++) {
      const dev = WT[i] * 0.12;                     // ~U0=0.04 scale deviation
      const exact = WT[i] + dev;
      const e1 = Math.abs((f16BitsToF32(f32ToF16Bits(exact)) - f16BitsToF32(f32ToF16Bits(WT[i]))) - dev);
      const e2 = Math.abs(f16BitsToF32(f32ToF16Bits(dev)) - dev);
      assert.ok(e2 <= e1, `direction ${i}: mode 2 error ${e2} not better than mode 1's ${e1}`);
    }
    // The rest link is where the ratio is largest and the claim is sharpest.
    const dev0 = WT[0] * 0.12;
    const ulp1 = WT[0] * Math.pow(2, -11);
    const ulp2 = dev0 * Math.pow(2, -11);
    assert.ok(ulp1 / ulp2 > 7, `expected >7x finer quantum on the deviation, got ${ulp1 / ulp2}`);
  });

  ok('mode 2 leaves f16-off untouched', () => {
    const f = new Float32Array(9 * 3).map((_, i) => i + 0.25);
    assert.strictEqual(packF(f, 3, 0), f);
    assert.strictEqual(unpackF(f, 3, 0), f);
  });

  console.log(`\ntest-f-pack: ${pass} assertion group(s) passed`);
})();
