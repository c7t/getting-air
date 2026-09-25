#!/usr/bin/env node
// GPU-free tests for lattice-2d.mjs and the WGSL it generates --
// plans/2D-backport.md B9. Sibling of tools/test-lattice-3d.js.
//
// TWO HALVES, AND THE SECOND IS THE POINT.
//
//   1. DRIFT. The checked-in shaders/common_lattice.wgsl must be exactly what
//      the generator emits. There is no build step here -- the source IS the
//      artifact -- so the generated file is committed, and a hand-edit or a
//      forgotten regenerate has to fail something.
//
//   2. THE LATTICE ITSELF, from the tables PARSED BACK OUT OF THE WGSL. The
//      moment conditions and a searched-not-assumed `opp` involution,
//      computed on the numbers the shader actually holds. This is the half
//      that fails when the generator and the checked-in file are wrong
//      TOGETHER -- which drift alone can never catch, and which is exactly
//      the situation B9 found: every copy agreed, and every copy was wrong.
//
// Both halves are negative-tested below against a transposed `opp` and a
// mis-shelled weight, because a test that has only ever seen correct input is
// indistinguishable from one that returns nothing.
//
// Run: node tools/test-lattice-2d.js   (also picked up by `make test`)

const assert = require('assert');
const fs = require('fs');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}
const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b} (tol ${tol})`);

// The tables the SHADER holds, read back out of the WGSL rather than imported.
function parseWGSL(src) {
  const arr = (name, re) => {
    const m = src.match(new RegExp(`const\\s+${name}\\s*=\\s*array<[^>]*>\\(([^)]*)\\)`));
    if (!m) throw new Error(`could not find ${name} in the WGSL`);
    const v = m[1].split(',').map(s => parseFloat(s.trim().replace(re, '')));
    if (v.length !== 9 || v.some(x => !isFinite(x))) throw new Error(`${name}: parsed ${v.length} entries: ${v}`);
    return v;
  };
  const cs2 = src.match(/const\s+CS2\s*=\s*([0-9.eE+-]+)f/);
  if (!cs2) throw new Error('could not find CS2 in the WGSL');
  return {
    ex: arr('ex', /$/), ey: arr('ey', /$/),
    wt: arr('wt', /f$/), opp: arr('opp', /u$/),
    cs2: parseFloat(cs2[1]),
  };
}

(async () => {
  const root = path.join(__dirname, '..');
  const L = await import(path.join(root, 'lattice-2d.mjs'));
  const wgslPath = path.join(root, 'shaders', 'common_lattice.wgsl');
  const onDisk = fs.readFileSync(wgslPath, 'utf8');

  // ── 1. drift ─────────────────────────────────────────────────────────────

  ok('the checked-in WGSL is exactly what the generator emits', () => {
    assert.strictEqual(onDisk, L.latticeWGSL(),
      'shaders/common_lattice.wgsl has drifted -- run `node tools/gen-lattice-2d.js`');
  });

  // ── 2. the lattice, from the tables the SHADER holds ─────────────────────

  const S = parseWGSL(onDisk);

  ok('the shader\'s weights conserve VELOCITY exactly -- 2w1 + 8w2 - w0 == 0', () => {
    // THE LOAD-BEARING ASSERTION, and the one that says why the 3D fork's ulp
    // tweak is not ported. Through one BGK collision the velocity is scaled by
    // 1 + omega*(3*Sig2 - S), which for D2Q9 is 1 + omega*(2w1 + 8w2 - w0) --
    // so a nonzero value here compounds into every velocity a gate measures,
    // growing with run length. NO TOLERANCE: the correctly rounded fractions
    // satisfy it to the bit (all three are multiples of 2^-26 and the
    // combination cancels), so anything else is a change of weights.
    const drift = 2 * S.wt[1] + 8 * S.wt[5] - S.wt[0];
    assert.strictEqual(drift, 0,
      `the shader's weights drift velocity by omega*${drift.toExponential(4)} per step`);
    assert.strictEqual(L.velocityDrift(), 0);
  });

  ok('the shader\'s weights are the correctly rounded fractions, and the host agrees', () => {
    // The SHIPPED file held eight-digit decimals, and 0.02777778f is one ulp
    // above fround(1/36) -- four diagonals, so its weights summed to 1+1.49e-8
    // where the correct rounding gives 1+7.45e-9. Measured in the running
    // solver at 1.861e-8 per cell per step against a predicted eps/tau0 of
    // 1.863e-8 (tools/analyze-amr-interface.js).
    //
    // The residual 7.45e-9 mass injection is NOT removable while keeping the
    // velocity exact -- an exhaustive search over |k| <= 4 finds no tweak that
    // zeroes both -- so it is bounded here rather than asserted away.
    for (let i = 0; i < 9; i++) {
      const shell = S.ex[i] ** 2 + S.ey[i] ** 2;
      const exact = [4 / 9, 1 / 9, 1 / 36][shell];
      assert.strictEqual(S.wt[i], Math.fround(exact),
        `direction ${i} is not fround(${exact})`);
    }
    const sum = S.wt.reduce((a, b) => a + b, 0);
    assert.ok(sum - 1 >= 0 && sum - 1 <= 7.4507e-9,
      `mass injection is 1 + ${(sum - 1).toExponential(4)}, worse than the correctly rounded 7.4506e-9`);
    // ...and the host agrees, to the bit. The eleven hand-typed host copies
    // this replaces did NOT: they held exact f64 fractions while the shader
    // held eight-digit f32 decimals.
    assert.deepStrictEqual(S.wt, L.WT, 'the shader and lattice-2d.mjs hold different weights');
    assert.strictEqual(L.weightSum(), sum);
  });

  ok('every weight is a real f32 and the tweak is a SMALL whole number of ulps', () => {
    // Bounded rather than trusted: a "tweak" of many ulps would be a different
    // lattice wearing the right sum.
    for (const w of S.wt) assert.strictEqual(Math.fround(w), w, `${w} is not an f32`);
    for (const p of L.weightProvenance()) {
      assert.ok(Number.isInteger(p.ulps) && Math.abs(p.ulps) <= 4,
        `shell tweak ${p.ulps} ulps is not a small integer`);
      close(p.f32, p.exact, 5 * p.ulp, `tweaked weight vs its exact fraction`);
    }
    // And the multiplicities really are 1/4/4 -- otherwise "sums to 1" is a
    // statement about the wrong nine numbers.
    const counts = {};
    for (const w of S.wt) counts[w] = (counts[w] || 0) + 1;
    assert.deepStrictEqual(Object.values(counts).sort((a, b) => a - b), [1, 4, 4]);
  });

  ok('the shader\'s velocity set is the D2Q9 one, each direction exactly once', () => {
    const seen = new Set();
    for (let i = 0; i < 9; i++) {
      assert.ok(Number.isInteger(S.ex[i]) && Math.abs(S.ex[i]) <= 1, `ex[${i}]`);
      assert.ok(Number.isInteger(S.ey[i]) && Math.abs(S.ey[i]) <= 1, `ey[${i}]`);
      const k = `${S.ex[i]},${S.ey[i]}`;
      assert.ok(!seen.has(k), `direction ${k} appears twice`);
      seen.add(k);
    }
    assert.strictEqual(seen.size, 9, 'the nine directions are not the nine of {-1,0,1}^2');
    // The weight a direction carries is a function of its SHELL alone.
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        const si = S.ex[i] ** 2 + S.ey[i] ** 2, sj = S.ex[j] ** 2 + S.ey[j] ** 2;
        if (si === sj) assert.strictEqual(S.wt[i], S.wt[j], `same shell, different weight: ${i} vs ${j}`);
      }
    }
  });

  ok('opp is an involution and really is the opposite direction -- SEARCHED, not assumed', () => {
    for (let i = 0; i < 9; i++) {
      const o = S.opp[i];
      assert.ok(Number.isInteger(o) && o >= 0 && o < 9, `opp[${i}] = ${o} is not an index`);
      assert.strictEqual(S.opp[o], i, `opp is not an involution at ${i}`);
      // Summed, not compared to -x: Object.is(0, -0) is false and the rest
      // direction would fail a strict compare for no reason.
      assert.strictEqual(S.ex[o] + S.ex[i], 0, `opp[${i}] is not the x-opposite`);
      assert.strictEqual(S.ey[o] + S.ey[i], 0, `opp[${i}] is not the y-opposite`);
    }
    // Independently: search the table for the opposite rather than reading it.
    for (let i = 0; i < 9; i++) {
      const found = [];
      for (let j = 0; j < 9; j++) if (S.ex[j] === -S.ex[i] && S.ey[j] === -S.ey[i]) found.push(j);
      assert.strictEqual(found.length, 1, `direction ${i} has ${found.length} opposites`);
      assert.strictEqual(found[0], S.opp[i], `searched opposite of ${i} is ${found[0]}, table says ${S.opp[i]}`);
    }
    assert.deepStrictEqual(S.opp, L.OPP, 'the shader and lattice-2d.mjs disagree about opp');
  });

  ok('the shader\'s tables satisfy the lattice moment conditions to 4th order', () => {
    // THE HALF THAT CATCHES A WRONG TABLE THE GENERATOR AGREES WITH. Computed
    // on the parsed numbers; nothing here is imported.
    //
    // The tweak deliberately spends a few 1e-9 of moment accuracy to buy an
    // EXACT sum -- a cs2 off by 1e-9 is a pressure off by 1e-9 and does not
    // accumulate, while a conservation law off by 1e-8 compounds every step.
    // So the tolerance here is 1e-8 and the sum above has none at all.
    const TOL = 1e-8;
    const cs2 = 1 / 3;
    const m = (fx) => S.wt.reduce((a, w, i) => a + w * fx(S.ex[i], S.ey[i]), 0);

    close(m(() => 1), 1, TOL, 'sum w');
    close(m((x) => x), 0, TOL, 'sum w e_x');
    close(m((_, y) => y), 0, TOL, 'sum w e_y');
    close(m((x) => x * x), cs2, TOL, 'sum w e_x^2');
    close(m((_, y) => y * y), cs2, TOL, 'sum w e_y^2');
    close(m((x, y) => x * y), 0, TOL, 'sum w e_x e_y');
    close(m((x) => x ** 3), 0, TOL, 'sum w e_x^3');
    close(m((x, y) => x * x * y), 0, TOL, 'sum w e_x^2 e_y');
    // Fourth order: the isotropy D2Q9 does provide, and which the Navier-Stokes
    // limit needs.
    close(m((x) => x ** 4), 3 * cs2 * cs2, TOL, 'sum w e_x^4');
    close(m((x, y) => x * x * y * y), cs2 * cs2, TOL, 'sum w e_x^2 e_y^2');
    close(m((x, y) => x ** 3 * y), 0, TOL, 'sum w e_x^3 e_y');
    // And CS2 as the shader states it is the f32 of 1/3.
    assert.strictEqual(Math.fround(S.cs2), Math.fround(cs2), 'CS2 is not fround(1/3)');
  });

  ok('the equilibrium conserves rho and rho*u on the shader\'s own numbers', () => {
    // The moment conditions above are what MAKE this true; asserting it
    // directly is the statement a solver actually depends on, at velocities
    // this project runs at.
    const feq = (rho, ux, uy, i) => {
      const eu = S.ex[i] * ux + S.ey[i] * uy;
      return S.wt[i] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * (ux * ux + uy * uy));
    };
    for (const [rho, ux, uy] of [[1, 0, 0], [1, 0.04, 0], [1, 0.03, -0.05], [1.002, -0.1, 0.1]]) {
      let m0 = 0, mx = 0, my = 0;
      for (let i = 0; i < 9; i++) {
        const v = feq(rho, ux, uy, i);
        m0 += v; mx += v * S.ex[i]; my += v * S.ey[i];
      }
      // At rest the mass residual is EXACTLY rho*(sum w - 1) -- the known,
      // irreducible 7.45e-9 that cannot be removed without creating a velocity
      // drift (see lattice-2d.mjs). Asserted as that identity rather than as a
      // tolerance, so if it ever changes the test says by how much and why.
      if (ux === 0 && uy === 0) {
        close(m0 - rho, rho * (L.weightSum() - 1), 1e-16,
          'the rest-state mass residual is not rho*(sum w - 1)');
      }
      close(m0, rho, 1e-8 * Math.max(1, ux * ux + uy * uy) + 1e-12, `sum feq at u=(${ux},${uy})`);
      // Momentum EXACTLY at rest, and to the f32 weights' own precision
      // otherwise: the residual is 3*(Sig2 - 1/3)*rho*u, a fixed relative
      // offset of 3*2.48e-9 rather than anything that accumulates -- and the
      // part that WOULD accumulate is the velocity drift asserted above.
      close(mx, rho * ux, 1e-8 * Math.abs(rho * ux) + 1e-12, `sum feq e_x at u=(${ux},${uy})`);
      close(my, rho * uy, 1e-8 * Math.abs(rho * uy) + 1e-12, `sum feq e_y at u=(${ux},${uy})`);
    }
  });

  // ── negative tests: does any of the above actually bite? ─────────────────

  ok('a transposed opp and a mis-shelled weight are both CAUGHT', () => {
    const mutate = (src, from, to) => {
      assert.ok(src.includes(from), `mutation source not found: ${from}`);
      return src.replace(from, to);
    };
    const checkFails = (src, what) => {
      const T = parseWGSL(src);
      let failed = false;
      // opp involution + searched opposite
      for (let i = 0; i < 9 && !failed; i++) {
        if (T.opp[T.opp[i]] !== i || T.ex[T.opp[i]] !== -T.ex[i] || T.ey[T.opp[i]] !== -T.ey[i]) failed = true;
      }
      // moments
      const m = (fx) => T.wt.reduce((a, w, i) => a + w * fx(T.ex[i], T.ey[i]), 0);
      if (Math.abs(m(() => 1) - 1) > 1e-8) failed = true;
      if (Math.abs(m((x) => x * x) - 1 / 3) > 1e-8) failed = true;
      if (Math.abs(m((x, y) => x * x * y * y) - 1 / 9) > 1e-8) failed = true;
      assert.ok(failed, `${what} was NOT caught -- these checks cannot see it`);
    };
    // A transposed opp: swap two entries so the table is still a permutation
    // and still an involution on its own, but no longer the opposite map.
    checkFails(mutate(onDisk, 'array<u32,9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u)',
      'array<u32,9>(0u, 3u, 4u, 1u, 2u, 8u, 7u, 6u, 5u)'), 'a transposed opp');
    // A mis-shelled weight: an axis direction given the diagonal's weight.
    // The sum then misses 1 and the second moment misses 1/3.
    const w1 = L.WT[1].toPrecision(17) + 'f', w5 = L.WT[5].toPrecision(17) + 'f';
    checkFails(mutate(onDisk, `  ${w1},\n  ${w1},\n  ${w1},\n  ${w1},`,
      `  ${w5},\n  ${w1},\n  ${w1},\n  ${w1},`), 'a mis-shelled weight');
    // And the state this file was ACTUALLY in before B9 -- the diagonal weight
    // one ulp high -- must fail the velocity assertion, which is what makes
    // that assertion load-bearing rather than decorative.
    const f32ulp = (x) => 2 ** (Math.floor(Math.log2(Math.abs(x))) - 23);
    const shipped = [Math.fround(4 / 9), Math.fround(1 / 9),
      Math.fround(1 / 36) + f32ulp(Math.fround(1 / 36))];
    const drift = 2 * shipped[1] + 8 * shipped[2] - shipped[0];
    assert.notStrictEqual(drift, 0,
      'the previously shipped weights pass the velocity check -- then it checks nothing');
    close(drift, 1.4901e-8, 1e-11, 'the shipped weights\' velocity drift');
    close(shipped[0] + 4 * shipped[1] + 4 * shipped[2] - 1, 1.4901e-8, 1e-11,
      'the shipped weights\' mass drift');
  });

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
