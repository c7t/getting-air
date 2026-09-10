#!/usr/bin/env node
// GPU-free tests for the D3Q19/D3Q27 lattice tables (lattice-3d.mjs and the
// generated shaders/common_d3q{19,27}_lattice.wgsl).
//
// WHAT IS ACTUALLY AT RISK, and why this is not just "does the generator
// agree with the file". plans/3D.md sec 6 names a transposed `opp` table as
// the highest-consequence, lowest-visibility bug in the 3D port: it does not
// crash and it does not NaN, it silently reflects bounce-back into the wrong
// direction and produces a plausible-looking wrong flow that a Cd/St harness
// will average into a number nobody can tell is wrong. The same is true of a
// mis-shelled weight, which quietly detunes cs2 and therefore the viscosity
// the tau->Re mapping assumes.
//
// So this test has two independent halves:
//
//   (A) DRIFT. The checked-in WGSL must still equal what lattice-3d.mjs
//       emits. There is no build step here, so the generated files are
//       committed and could otherwise be hand-edited apart from their
//       source. This half is exact string equality.
//
//   (B) TRUTH. The tables PARSED BACK OUT of the WGSL must satisfy the
//       lattice moment conditions (sum w = 1, sum w e = 0, sum w e_a e_b =
//       cs2 delta, vanishing third moment, and the fourth-moment isotropy
//       identity), plus the bounce-back involution derived by SEARCHING for
//       each -e in the table rather than by re-running the generator's own
//       closed form. Half (B) is what fails when the generator and the file
//       are wrong together -- which half (A), on its own, would call green.
//
// Run: node tools/test-lattice-3d.js   (also picked up by `make test`)

const assert = require('assert');
const fs = require('fs');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}

// --- WGSL parsing ---------------------------------------------------------
// Deliberately naive and deliberately independent of the emitter: pull the
// literal text of each `const <name> = array<T,N>( ... );` and evaluate the
// entries arithmetically. `wt` entries are fractions like `1.0f/18.0f`, so
// they are divided out at f64 and narrowed with Math.fround -- which is what
// the GPU's f32 const-expression division produces for every value in these
// tables (all far from an f32 tie).
function parseArray(src, name) {
  const m = src.match(new RegExp(`const\\s+${name}\\s*=\\s*array<[^>]+>\\(([\\s\\S]*?)\\)\\s*;`));
  if (!m) throw new Error(`no array named ${name} in shader source`);
  return m[1]
    .split(',')
    .map(s => s.replace(/\/\/.*$/gm, '').trim())
    .filter(s => s.length)
    .map(s => {
      const frac = s.match(/^([0-9.]+)f?\s*\/\s*([0-9.]+)f?$/);
      if (frac) return Math.fround(parseFloat(frac[1]) / parseFloat(frac[2]));
      const n = s.match(/^(-?[0-9.]+)(f|u|i)?$/);
      if (!n) throw new Error(`unparseable entry "${s}" in ${name}`);
      return parseFloat(n[1]);
    });
}

function parseLattice(src) {
  const qm = src.match(/const\s+QN\s*:\s*u32\s*=\s*(\d+)u\s*;/);
  if (!qm) throw new Error('no QN declaration in shader source');
  return {
    Q: parseInt(qm[1], 10),
    ex: parseArray(src, 'ex'),
    ey: parseArray(src, 'ey'),
    ez: parseArray(src, 'ez'),
    wt: parseArray(src, 'wt'),
    opp: parseArray(src, 'opp'),
  };
}

// --- moment conditions ----------------------------------------------------
// A valid isothermal LBM velocity set satisfies these through fourth order;
// they are what make the Chapman-Enskog expansion recover Navier-Stokes with
// nu = cs2*(tau - 1/2). Checked on the parsed tables, so they are a test of
// the lattice itself, not of the code that wrote it down.
function moments(L, CS2) {
  const { Q, ex, ey, ez, wt } = L;
  const e = [ex, ey, ez];
  const d = (a, b) => (a === b ? 1 : 0);
  const m0 = wt.reduce((s, w) => s + w, 0);
  const m1 = [0, 1, 2].map(a => wt.reduce((s, w, i) => s + w * e[a][i], 0));
  const m2 = [];
  const m3 = [];
  const m4 = [];
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    let s = 0; for (let i = 0; i < Q; i++) s += wt[i] * e[a][i] * e[b][i];
    m2.push({ a, b, got: s, want: CS2 * d(a, b) });
    for (let c = 0; c < 3; c++) {
      let s3 = 0; for (let i = 0; i < Q; i++) s3 += wt[i] * e[a][i] * e[b][i] * e[c][i];
      m3.push({ a, b, c, got: s3, want: 0 });
      for (let dd = 0; dd < 3; dd++) {
        let s4 = 0; for (let i = 0; i < Q; i++) s4 += wt[i] * e[a][i] * e[b][i] * e[c][i] * e[dd][i];
        m4.push({ a, b, c, d: dd, got: s4, want: CS2 * CS2 * (d(a, b) * d(c, dd) + d(a, c) * d(b, dd) + d(a, dd) * d(b, c)) });
      }
    }
  }
  return { m0, m1, m2, m3, m4 };
}

// f32 tolerance, deliberately, not f64: the moment checks run on the tables
// PARSED OUT OF THE SHADER, whose weights are the f32 values the GPU will
// actually use (1/18 is not representable), so sum w = 1 only holds to about
// a single-precision epsilon. That is still three orders of magnitude
// tighter than any real table error -- a mis-shelled weight is an O(0.01)
// discrepancy, not an O(1e-7) one.
const TOL = 1e-6;

(async () => {
  const root = path.join(__dirname, '..');
  const LAT = await import(path.join(root, 'lattice-3d.mjs'));
  const { SUPPORTED_Q, CS2, velocities, weights, opposites, shellRanges, latticeWGSL, feq } = LAT;

  const shaderSrc = {};
  for (const Q of SUPPORTED_Q) {
    shaderSrc[Q] = fs.readFileSync(path.join(root, 'shaders', `common_d3q${Q}_lattice.wgsl`), 'utf8');
  }

  // --- (A) drift ----------------------------------------------------------
  for (const Q of SUPPORTED_Q) {
    ok(`D3Q${Q}: checked-in WGSL matches lattice-3d.mjs`, () => {
      assert.strictEqual(shaderSrc[Q], latticeWGSL(Q),
        `shaders/common_d3q${Q}_lattice.wgsl is stale or hand-edited -- run \`node tools/gen-lattice-3d.js\``);
    });
  }

  // --- (B) truth, on the tables parsed out of the shader -------------------
  for (const Q of SUPPORTED_Q) {
    const L = parseLattice(shaderSrc[Q]);

    ok(`D3Q${Q}: shader tables are all length Q`, () => {
      assert.strictEqual(L.Q, Q);
      for (const k of ['ex', 'ey', 'ez', 'wt', 'opp']) assert.strictEqual(L[k].length, Q, `${k} has ${L[k].length} entries`);
    });

    ok(`D3Q${Q}: shader tables equal the host tables`, () => {
      const v = velocities(Q), w = weights(Q), o = opposites(Q);
      for (let i = 0; i < Q; i++) {
        assert.deepStrictEqual([L.ex[i], L.ey[i], L.ez[i]], v[i], `direction ${i}`);
        assert.strictEqual(L.wt[i], Math.fround(w[i]), `weight ${i}`);
        assert.strictEqual(L.opp[i], o[i], `opp ${i}`);
      }
    });

    ok(`D3Q${Q}: velocities are the distinct offsets of shells 0..${Q === 19 ? 2 : 3}`, () => {
      const seen = new Set();
      for (let i = 0; i < Q; i++) {
        const e = [L.ex[i], L.ey[i], L.ez[i]];
        for (const c of e) assert.ok(c === -1 || c === 0 || c === 1, `direction ${i} component out of {-1,0,1}`);
        const n2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        assert.ok(n2 <= (Q === 19 ? 2 : 3), `direction ${i} is in shell ${n2}, outside D3Q${Q}`);
        const k = e.join(',');
        assert.ok(!seen.has(k), `direction ${i} (${k}) is a duplicate`);
        seen.add(k);
      }
    });

    // THE one that matters. opp is re-derived by SEARCHING the parsed table
    // for -e, so it cannot inherit the generator's assumption about
    // pair ordering; a transposed table fails right here.
    ok(`D3Q${Q}: opp[i] is the index of -e_i (searched, not assumed)`, () => {
      for (let i = 0; i < Q; i++) {
        const want = [-L.ex[i], -L.ey[i], -L.ez[i]].join(',');
        let found = -1;
        for (let j = 0; j < Q; j++) if ([L.ex[j], L.ey[j], L.ez[j]].join(',') === want) { found = j; break; }
        assert.notStrictEqual(found, -1, `no opposite present for direction ${i}`);
        assert.strictEqual(L.opp[i], found, `opp[${i}] says ${L.opp[i]}, but -e_${i} is direction ${found}`);
        assert.strictEqual(L.opp[L.opp[i]], i, `opp is not an involution at ${i}`);
      }
      assert.strictEqual(L.opp[0], 0, 'rest direction must be self-opposite');
    });

    ok(`D3Q${Q}: weights depend only on |e|^2`, () => {
      const byShell = new Map();
      for (let i = 0; i < Q; i++) {
        const n2 = L.ex[i] ** 2 + L.ey[i] ** 2 + L.ez[i] ** 2;
        if (!byShell.has(n2)) byShell.set(n2, L.wt[i]);
        assert.strictEqual(L.wt[i], byShell.get(n2), `direction ${i} (shell ${n2}) has a different weight from its shell`);
      }
    });

    ok(`D3Q${Q}: moment conditions (0th..4th) hold to f32 precision`, () => {
      const M = moments(L, CS2);
      assert.ok(Math.abs(M.m0 - 1) < TOL, `sum w = ${M.m0}, want 1`);
      M.m1.forEach((v, a) => assert.ok(Math.abs(v) < TOL, `sum w e_${a} = ${v}, want 0`));
      for (const t of M.m2) assert.ok(Math.abs(t.got - t.want) < TOL, `sum w e_${t.a} e_${t.b} = ${t.got}, want ${t.want}`);
      for (const t of M.m3) assert.ok(Math.abs(t.got - t.want) < TOL, `third moment (${t.a},${t.b},${t.c}) = ${t.got}, want 0`);
      for (const t of M.m4) assert.ok(Math.abs(t.got - t.want) < TOL, `fourth moment (${t.a},${t.b},${t.c},${t.d}) = ${t.got}, want ${t.want}`);
    });

    ok(`D3Q${Q}: shell ranges are contiguous and correctly labelled`, () => {
      const sh = shellRanges(Q);
      const shellOf = (i) => L.ex[i] ** 2 + L.ey[i] ** 2 + L.ez[i] ** 2;
      const check = (name, [s, e], n2) => {
        for (let i = s; i < e; i++) assert.strictEqual(shellOf(i), n2, `direction ${i} is in ${name} range but shell ${shellOf(i)}`);
      };
      check('rest', sh.rest, 0); check('face', sh.face, 1); check('edge', sh.edge, 2);
      if (Q === 27) check('corner', sh.corner, 3);
      assert.deepStrictEqual([sh.rest[0], sh.rest[1]], [0, 1]);
      assert.strictEqual(sh.face[0], sh.rest[1]);
      assert.strictEqual(sh.edge[0], sh.face[1]);
      assert.strictEqual(sh.corner[0], sh.edge[1]);
      assert.strictEqual(sh.corner[1], Q);
    });
  }

  // D3Q19 has no corner directions at all -- this is the concrete
  // simplification plans/3D.md sec 2.3 credits it with (a same-level
  // DIRECT_GHOST gather never needs a diagonal-corner neighbour tile), so
  // it is asserted rather than left as prose.
  ok('D3Q19 has no corner (|e|^2 = 3) directions; D3Q27 has exactly 8', () => {
    assert.strictEqual(shellRanges(19).corner[1] - shellRanges(19).corner[0], 0);
    assert.strictEqual(shellRanges(27).corner[1] - shellRanges(27).corner[0], 8);
  });

  // Index i means the same direction in both sets. Everything that compares
  // a Q19 run against a Q27 run plane-by-plane depends on this, as does the
  // `?q=19|27` build variant being a loop bound rather than a remap.
  ok('D3Q19 is a strict index-prefix of D3Q27', () => {
    const v19 = velocities(19), v27 = velocities(27);
    for (let i = 0; i < 19; i++) assert.deepStrictEqual(v19[i], v27[i], `direction ${i} differs between the sets`);
    const o19 = opposites(19), o27 = opposites(27);
    for (let i = 0; i < 19; i++) assert.strictEqual(o19[i], o27[i], `opp[${i}] differs between the sets`);
  });

  // The host feq must reproduce the moments it is supposed to: an
  // equilibrium at (rho, u) has to have exactly that density and momentum,
  // or every initializer that uses it seeds a field that is not the state it
  // claims. Checked at a few u, all well inside the low-Mach range.
  for (const Q of SUPPORTED_Q) {
    ok(`D3Q${Q}: feq recovers rho and rho*u`, () => {
      for (const [rho, ux, uy, uz] of [[1, 0, 0, 0], [1, 0.05, -0.03, 0.02], [0.97, -0.1, 0.06, 0.01]]) {
        let m0 = 0, mx = 0, my = 0, mz = 0;
        for (let i = 0; i < Q; i++) {
          const f = feq(Q, rho, ux, uy, uz, i);
          const v = velocities(Q)[i];
          m0 += f; mx += f * v[0]; my += f * v[1]; mz += f * v[2];
        }
        assert.ok(Math.abs(m0 - rho) < 1e-12, `sum feq = ${m0}, want rho = ${rho}`);
        assert.ok(Math.abs(mx - rho * ux) < 1e-12, `x momentum ${mx}, want ${rho * ux}`);
        assert.ok(Math.abs(my - rho * uy) < 1e-12, `y momentum ${my}, want ${rho * uy}`);
        assert.ok(Math.abs(mz - rho * uz) < 1e-12, `z momentum ${mz}, want ${rho * uz}`);
      }
    });
  }

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
