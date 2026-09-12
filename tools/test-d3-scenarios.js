#!/usr/bin/env node
// GPU-free tests for d3-scenarios.mjs -- the closed-form reference
// solutions the dense 3D solver (plans/3D.md M1) is validated against.
//
// WHAT IS AT RISK. These functions are the RULER. If the duct series has a
// transposed axis or a dropped factor, or the Beltrami field is not
// actually Beltrami, then the solver is being measured against the wrong
// answer -- and the failure mode is the worst one available: the harness
// reports a number, the number looks plausible, and the solver is wrong in
// exactly the way the ruler is wrong. Worse for beltrami specifically,
// whose initial condition IS its own reference at t = 0, so a shared
// mistake would sail through the t = 0 check.
//
// So nothing here compares a formula against a transcription of the same
// formula. Each reference solution is checked against the PDE it is
// supposed to solve, by finite differences:
//
//   duct     lap(u) = -G/nu in the interior, u = 0 on all four walls, and
//            the b -> infinity limit reduces to plane Poiseuille.
//   beltrami curl(u) = k*u (the Beltrami condition itself), div(u) = 0,
//            and lap(u) = -k^2 u (which is what makes the decay exact).
//   tgv      div(u) = 0 (all that is claimed for it -- it is an initial
//            condition, not a solution).
//
// Run: node tools/test-d3-scenarios.js   (also picked up by `make test`)

const assert = require('assert');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}

// Central differences on a continuously-evaluable vector field.
// h = 0.01 lattice units: small against the field's own O(N) length scale,
// large enough that f64 cancellation in the second difference (~4*eps*|u|/h^2)
// stays ~1e-11, far below every tolerance used here.
const H = 0.01;

function grad(f, x, y, z) {
  const d = (i) => {
    const p = [x, y, z], m = [x, y, z];
    p[i] += H; m[i] -= H;
    return (f(...p) - f(...m)) / (2 * H);
  };
  return [d(0), d(1), d(2)];
}

function laplacian(f, x, y, z) {
  const c = f(x, y, z);
  let s = 0;
  for (let i = 0; i < 3; i++) {
    const p = [x, y, z], m = [x, y, z];
    p[i] += H; m[i] -= H;
    s += (f(...p) - 2 * c + f(...m)) / (H * H);
  }
  return s;
}

// div and curl of a vector field given as u(x,y,z) -> [ux, uy, uz].
function divergence(u, x, y, z) {
  let s = 0;
  for (let i = 0; i < 3; i++) s += grad((...p) => u(...p)[i], x, y, z)[i];
  return s;
}
function curl(u, x, y, z) {
  const g = (i) => grad((...p) => u(...p)[i], x, y, z);
  const [gx, gy, gz] = [g(0), g(1), g(2)];
  return [gz[1] - gy[2], gx[2] - gz[0], gy[0] - gx[1]];
}

const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b} (|diff| ${Math.abs(a - b).toExponential(2)} > ${tol.toExponential(2)})`);

(async () => {
  const root = path.join(__dirname, '..');
  const S = await import(path.join(root, 'd3-scenarios.mjs'));
  const {
    CS2, nuFromTau, tauFromNu, ductCoord, ductVelocityAt, ductPeakCoeff, ductForceForPeak,
    ductProfile, ductSettleTime, beltramiK, beltramiDecayTime, beltramiVelocityAt,
    tgvVelocityAt, SCENARIOS, SCENARIO_NAMES, resolveScenario, schillerNaumann,
  } = S;

  // --- viscosity mapping ---------------------------------------------------
  ok('nu <-> tau round-trips, and nu(tau=0.5) = 0', () => {
    close(nuFromTau(0.5), 0, 1e-15, 'nu at tau=0.5');
    close(nuFromTau(0.8), CS2 * 0.3, 1e-15, 'nu at tau=0.8');
    for (const tau of [0.51, 0.6, 0.8, 1.0, 2.0]) close(tauFromNu(nuFromTau(tau)), tau, 1e-12, `tau round-trip at ${tau}`);
  });

  // --- duct ----------------------------------------------------------------
  ok('duct peak coefficient matches the textbook 0.2947 for a square duct', () => {
    close(ductPeakCoeff(), 0.2947, 5e-5, 'u_max / (G a^2 / nu)');
  });

  ok('duct series vanishes on all four walls', () => {
    const a = 12, G = 1e-4, nu = 0.1;
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      close(ductVelocityAt(a, t * a, a, a, G, nu), 0, 1e-12, `u at y=+a, z=${t}a`);
      close(ductVelocityAt(-a, t * a, a, a, G, nu), 0, 1e-12, `u at y=-a, z=${t}a`);
      close(ductVelocityAt(t * a, a, a, a, G, nu), 0, 1e-12, `u at z=+b, y=${t}a`);
      close(ductVelocityAt(t * a, -a, a, a, G, nu), 0, 1e-12, `u at z=-b, y=${t}a`);
    }
  });

  // THE ONE THAT MATTERS: the series is checked against the equation it
  // solves, by an instrument that shares no code with it.
  //
  // The obvious check -- finite-difference lap(u) and compare to -G/nu --
  // does NOT work, and the way it fails is instructive. The Fourier
  // coefficients of a constant decay only as 1/i, so a truncated series
  // reproduces u to O(i_max^-3) but its LAPLACIAN only to O(i_max^-1):
  // at 24 terms that is a 1.3% error in lap(u) and nothing wrong with the
  // formula. Raising the term count does not rescue it either -- the tail
  // modes have wavelengths far below any usable finite-difference step.
  //
  // So instead: solve lap(u) = -G/nu on the same domain by SOR, from
  // scratch, and check that the discrete solution CONVERGES TO the series
  // at second order. That is the strong form. A wrong closed form would
  // leave the error plateauing at its own bias instead of falling 4x per
  // halving of h, and no tolerance would have to be guessed at.
  function solvePoissonSOR(a, b, M, rhs) {
    const h = 2 * a / M;
    const ny = M + 1, nz = Math.round(2 * b / h) + 1;
    const u = new Float64Array(ny * nz);
    const w = 2 / (1 + Math.sin(Math.PI / Math.max(ny, nz)));   // optimal SOR factor
    const at = (i, j) => u[i * nz + j];
    let iter = 0;
    for (; iter < 200000; iter++) {
      let maxd = 0;
      for (let i = 1; i < ny - 1; i++) {
        for (let j = 1; j < nz - 1; j++) {
          const gs = (at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) + h * h * rhs) / 4;
          const old = u[i * nz + j];
          u[i * nz + j] = old + w * (gs - old);
          const d = Math.abs(u[i * nz + j] - old);
          if (d > maxd) maxd = d;
        }
      }
      if (maxd < 1e-15) break;      // to round-off, so the residual is not the error
    }
    return { ny, nz, h, iter, at };
  }

  function sorVsSeriesL2(a, b, M, G, nu) {
    const s = solvePoissonSOR(a, b, M, G / nu);
    let num = 0, den = 0;
    for (let i = 1; i < s.ny - 1; i++) {
      for (let j = 1; j < s.nz - 1; j++) {
        const exact = ductVelocityAt(-a + i * s.h, -b + j * s.h, a, b, G, nu);
        const d = s.at(i, j) - exact;
        num += d * d; den += exact * exact;
      }
    }
    return Math.sqrt(num / den);
  }

  ok('duct series is what an independent SOR Poisson solve converges to (square, 2nd order)', () => {
    const a = 12, G = 3e-4, nu = 0.1;
    const e = [16, 32, 64].map(M => sorVsSeriesL2(a, a, M, G, nu));
    assert.ok(e[2] < 3e-4, `L2 error at the finest grid is ${e[2].toExponential(2)}, want < 3e-4`);
    for (let i = 0; i + 1 < e.length; i++) {
      const r = e[i] / e[i + 1];
      assert.ok(r > 3.5 && r < 4.5,
        `halving h reduced the error by ${r.toFixed(2)}x, want ~4x (second order). ` +
        `A plateau here means the SOR solution is converging to something OTHER than the series. ` +
        `errors: ${e.map(x => x.toExponential(2)).join(', ')}`);
    }
  });

  // A NON-square duct is where a transposed cos/cosh axis stops being
  // invisible: at a = b the two are interchangeable and the bug hides.
  ok('duct series is what SOR converges to for a NON-square duct too', () => {
    const a = 8, b = 20, G = 3e-4, nu = 0.1;
    const e = [8, 16, 32].map(M => sorVsSeriesL2(a, b, M, G, nu));
    assert.ok(e[2] < 1e-3, `L2 error at the finest grid is ${e[2].toExponential(2)}, want < 1e-3`);
    for (let i = 0; i + 1 < e.length; i++) {
      const r = e[i] / e[i + 1];
      assert.ok(r > 3.4 && r < 4.6, `error ratio ${r.toFixed(2)}x, want ~4x. errors: ${e.map(x => x.toExponential(2)).join(', ')}`);
    }
    for (const t of [0, 0.5, 1]) {
      close(ductVelocityAt(a, t * b, a, b, G, nu), 0, 1e-12, 'u at the y = a wall');
      close(ductVelocityAt(t * a, b, a, b, G, nu), 0, 1e-12, 'u at the z = b wall');
    }
  });

  ok('duct reduces to plane Poiseuille as the second half-width grows', () => {
    const a = 10, b = 400, G = 1e-4, nu = 0.1;
    for (const y of [0, 2.5, -5, 7.5]) {
      const plane = (G / (2 * nu)) * (a * a - y * y);
      close(ductVelocityAt(y, 0, a, b, G, nu), plane, 1e-3 * plane + 1e-12, `u(y=${y}) vs plane Poiseuille`);
    }
  });

  // At N=128 the naive cosh(z)/cosh(b) form overflows f64 in the higher
  // series terms and returns NaN. The exponential-ratio form does not; this
  // is the regression test for that, since a NaN reference would fail the
  // solver rather than itself.
  ok('duct profile stays finite at large N (no cosh overflow)', () => {
    for (const N of [32, 64, 128, 256]) {
      const nu = nuFromTau(0.8);
      const G = ductForceForPeak(0.05, N / 2, nu);
      const p = ductProfile(N, G, nu);
      assert.ok(p.every(Number.isFinite), `N=${N}: profile has non-finite entries`);
      const peak = Math.max(...p);
      // Peak is at the two central cells, half a cell off the true centre,
      // so it lands just under the analytic centreline value.
      close(peak, 0.05, 0.05 * 0.01, `N=${N} peak velocity`);
    }
  });

  ok('ductForceForPeak inverts ductPeakCoeff', () => {
    const nu = nuFromTau(0.8), a = 24;
    const G = ductForceForPeak(0.05, a, nu);
    close(ductVelocityAt(0, 0, a, a, G, nu), 0.05, 1e-12, 'centreline velocity');
  });

  ok('duct cell coordinates are symmetric about the centre (half-cell wall offset)', () => {
    const N = 48;
    close(ductCoord(0, N), -(N / 2 - 0.5), 1e-15, 'first cell');
    close(ductCoord(N - 1, N), N / 2 - 0.5, 1e-15, 'last cell');
    for (let j = 0; j < N; j++) close(ductCoord(j, N), -ductCoord(N - 1 - j, N), 1e-15, `cell ${j} mirror`);
  });

  ok('duct settle time scales as a^2/nu', () => {
    const t1 = ductSettleTime(48, 0.1), t2 = ductSettleTime(96, 0.1);
    close(t2 / t1, 4, 1e-9, 'doubling N quadruples the settle time');
    close(ductSettleTime(48, 0.2) / ductSettleTime(48, 0.1), 0.5, 1e-9, 'halving nu doubles it');
  });

  // --- beltrami ------------------------------------------------------------
  const N = 48, u0 = 0.04, nu = nuFromTau(0.8);
  const k = beltramiK(N);
  const bel = (t) => (x, y, z) => beltramiVelocityAt(x, y, z, N, u0, nu, t);

  ok('beltrami field satisfies curl(u) = k*u (this is what makes it exact)', () => {
    const u = bel(0);
    for (const [x, y, z] of [[0, 0, 0], [3.5, 11, 27], [40, 2, 19], [12.25, 36.5, 5]]) {
      const c = curl(u, x, y, z), v = u(x, y, z);
      for (let i = 0; i < 3; i++) close(c[i], k * v[i], 1e-6 * u0 * k, `curl component ${i} at (${x},${y},${z})`);
    }
  });

  ok('beltrami field is divergence-free', () => {
    const u = bel(0);
    for (const [x, y, z] of [[0, 0, 0], [3.5, 11, 27], [40, 2, 19]]) {
      close(divergence(u, x, y, z), 0, 1e-9, `div(u) at (${x},${y},${z})`);
    }
  });

  ok('beltrami field satisfies lap(u) = -k^2 u (the decay eigenvalue)', () => {
    const u = bel(0);
    for (const [x, y, z] of [[0, 0, 0], [3.5, 11, 27], [40, 2, 19]]) {
      for (let i = 0; i < 3; i++) {
        const lap = laplacian((...p) => u(...p)[i], x, y, z);
        close(lap, -k * k * u(x, y, z)[i], 1e-5 * k * k * u0, `lap(u_${i}) at (${x},${y},${z})`);
      }
    }
  });

  ok('beltrami decays as exp(-nu k^2 t) with its shape exactly preserved', () => {
    const td = beltramiDecayTime(N, nu);
    close(td, 1 / (nu * k * k), 1e-12, 'decay time');
    for (const t of [0.5 * td, 2 * td, 5 * td]) {
      const want = Math.exp(-t / td);
      for (const [x, y, z] of [[7, 13, 41], [0, 0, 0], [23.5, 8, 30]]) {
        const a = beltramiVelocityAt(x, y, z, N, u0, nu, 0);
        const b = beltramiVelocityAt(x, y, z, N, u0, nu, t);
        for (let i = 0; i < 3; i++) {
          if (Math.abs(a[i]) < 1e-9) continue;   // a node of that component
          close(b[i] / a[i], want, 1e-12, `component ${i} decay ratio at t=${t.toFixed(1)}`);
        }
      }
    }
  });

  ok('beltrami is periodic on the N-cell box', () => {
    for (const [x, y, z] of [[0, 0, 0], [5, 17, 31]]) {
      const a = beltramiVelocityAt(x, y, z, N, u0, nu, 0);
      const b = beltramiVelocityAt(x + N, y, z + N, N, u0, nu, 0);
      for (let i = 0; i < 3; i++) close(b[i], a[i], 1e-12, `component ${i} periodicity`);
    }
  });

  // --- tgv -----------------------------------------------------------------
  ok('tgv initial condition is divergence-free', () => {
    const u = (x, y, z) => tgvVelocityAt(x, y, z, 64, 0.04);
    for (const [x, y, z] of [[0, 0, 0], [7.5, 19, 33], [50, 3, 12]]) {
      close(divergence(u, x, y, z), 0, 1e-9, `div(u) at (${x},${y},${z})`);
    }
  });

  ok('tgv has no z-velocity at t=0 but does depend on z', () => {
    const at = (x, y, z) => tgvVelocityAt(x, y, z, 64, 0.04);
    for (const [x, y, z] of [[5, 9, 13], [0, 0, 0], [21, 30, 44]]) close(at(x, y, z)[2], 0, 1e-15, 'w');
    assert.notStrictEqual(at(5, 9, 0)[0], at(5, 9, 16)[0], 'u_x must vary with z');
  });

  // --- scenario table ------------------------------------------------------
  ok('every scenario resolves to explicit dims and seeds a finite field of that size', () => {
    for (const name of SCENARIO_NAMES) {
      const p = resolveScenario(name, { n: 16 });
      assert.ok(p.nu > 0, `${name}: nu must be positive`);
      assert.ok(p.tau > 0.5, `${name}: tau must exceed 1/2 (nu > 0)`);
      assert.strictEqual(p.scenario, name);
      assert.ok(Array.isArray(p.dims) && p.dims.length === 3, `${name}: dims must be a 3-vector`);
      assert.ok(p.dims.every(d => Number.isInteger(d) && d > 0), `${name}: dims must be positive integers, got ${p.dims}`);
      const cells = p.dims[0] * p.dims[1] * p.dims[2];
      const m = SCENARIOS[name].macro(p.dims, p);
      assert.strictEqual(m.length, 4 * cells, `${name}: macro field is the wrong length for dims ${p.dims}`);
      assert.ok(m.every(Number.isFinite), `${name}: macro field has non-finite entries`);
      for (let c = 0; c < cells; c++) {
        assert.ok(m[4 * c] > 0.9 && m[4 * c] < 1.1, `${name}: seeded rho out of range at cell ${c}`);
      }
    }
  });

  ok('duct scenario seeds a quiescent field (its reference is independent of its seed)', () => {
    const p = resolveScenario('duct', { n: 16 });
    const m = SCENARIOS.duct.macro(p.dims, p);
    for (let c = 0; c < 16 ** 3; c++) {
      for (let i = 1; i < 4; i++) close(m[4 * c + i], 0, 0, `velocity component ${i} at cell ${c}`);
      close(m[4 * c], 1, 0, `rho at cell ${c}`);
    }
  });

  ok('duct scenario derives a force that hits its target peak velocity', () => {
    const p = resolveScenario('duct', { n: 32, tau: 0.9, u0: 0.03 });
    assert.deepStrictEqual(p.walls, ['y', 'z']);
    close(p.force[0], p.G, 0, 'force[0] is G');
    close(ductVelocityAt(0, 0, p.a, p.a, p.G, p.nu), 0.03, 1e-12, 'analytic centreline velocity');
  });

  ok('beltrami macro seed matches the analytic field at t=0', () => {
    const n = 16;
    const p = resolveScenario('beltrami', { n });
    const m = SCENARIOS.beltrami.macro(p.dims, p);
    for (const [x, y, z] of [[0, 0, 0], [5, 9, 13], [15, 15, 15]]) {
      const c = (z * n + y) * n + x;
      const want = beltramiVelocityAt(x, y, z, n, p.u0, p.nu, 0);
      for (let i = 0; i < 3; i++) close(m[4 * c + 1 + i], want[i], 1e-7, `component ${i} at (${x},${y},${z})`);
    }
  });

  // --- sphere (M2) ---------------------------------------------------------
  ok('sphere: tau follows from the target Re, and the geometry is consistent', () => {
    for (const [n, re, u0] of [[16, 100, 0.05], [24, 20, 0.04], [12, 200, 0.06]]) {
      const p = resolveScenario('sphere', { n, re, u0 });
      // Re = U D / nu is the DEFINITION, so this must hold by construction.
      close(p.u0 * p.D / p.nu, re, 1e-9 * re, `Re at n=${n}`);
      close(p.tau, tauFromNu(p.nu), 1e-12, 'tau/nu consistency');
      assert.ok(p.tau > 0.5, `tau=${p.tau} must exceed 1/2`);
      close(p.R, n / 2, 0, 'radius');
      close(p.area, Math.PI * (n / 2) ** 2, 1e-9, 'frontal area');
      // Body centred across the span, upstream along x.
      close(p.body.x[1], p.dims[1] / 2, 0, 'body y');
      close(p.body.x[2], p.dims[2] / 2, 0, 'body z');
      assert.ok(p.body.x[0] > p.R * 2, 'body must sit clear of the inlet');
      assert.ok(p.dims[0] - p.body.x[0] > 6 * p.R, 'wake needs room downstream');
      assert.ok(p.blockage < 0.02, `blockage ${(p.blockage * 100).toFixed(2)}% should be small`);
      assert.ok(p.pinned === true, 'the sphere gate measures force on a body that does not move');
    }
  });

  ok('Schiller-Naumann matches the tabulated sphere drag curve over the range it is used in', () => {
    // Standard-drag-curve values. The band is 5%, which is the correlation's
    // own stated accuracy -- not a number chosen to make this pass.
    //
    // Re = 1 is deliberately absent. Schiller-Naumann is quoted for
    // Re up to ~800 and is at its weakest at the Stokes end (it gives 27.6
    // against a tabulated ~26.5 there, a 4% miss that says nothing about
    // this implementation). Nothing in benchmarks/d3-sphere.json goes below
    // Re = 20, so the test covers the range actually relied on.
    for (const [re, cd] of [[10, 4.15], [20, 2.61], [50, 1.57], [100, 1.09], [200, 0.80]]) {
      const got = schillerNaumann(re);
      assert.ok(Math.abs(got - cd) / cd < 0.05, `Cd(Re=${re}) = ${got.toFixed(3)}, want ~${cd}`);
    }
    // Monotone decreasing over the range it is used in.
    for (let re = 1; re < 500; re *= 1.5) {
      assert.ok(schillerNaumann(re * 1.5) < schillerNaumann(re), `Cd must decrease with Re, fails near ${re}`);
    }
  });

  ok('sphere seeds the uniform freestream, so there is no start-up transient to wait out', () => {
    const p = resolveScenario('sphere', { n: 8 });
    const m = SCENARIOS.sphere.macro(p.dims, p);
    const cells = p.dims[0] * p.dims[1] * p.dims[2];
    for (let c = 0; c < cells; c += 97) {
      close(m[4 * c + 1], Math.fround(p.u0), 1e-7, `ux at cell ${c}`);
      close(m[4 * c + 2], 0, 1e-9, `uy at cell ${c}`);
    }
  });

  // --- spin (M2) -----------------------------------------------------------
  ok('spin: a free tumbling body with the fluid force switched off', () => {
    const p = resolveScenario('spin', { n: 32 });
    assert.strictEqual(p.pinned, false);
    assert.strictEqual(p.noFluidForce, true, 'the point of this scenario is to isolate the integrator');
    // Near the INTERMEDIATE principal axis, so the compared trajectory is
    // the tumbling one -- agreeing on a steady spin would be a weaker test.
    const I = p.body.ibody;
    assert.ok(I[0] < I[1] && I[1] < I[2], `expected ordered moments, got ${I.map(v => v.toExponential(2))}`);
    const w = p.body.omega;
    assert.ok(Math.abs(w[1]) > 100 * Math.abs(w[0]), 'spin should be about the intermediate (y) axis');
    // Not axis-aligned: an orientation-handling bug that only shows up off
    // the identity would otherwise hide.
    assert.ok(Math.abs(p.body.q[0] - 1) > 1e-3, 'initial orientation should not be the identity');
    assert.ok(p.body.x.every((c, i) => Math.abs(c - p.dims[i] / 2) < 1e-9), 'body should start centred');
  });

  // --- fall: the Galilean split (plans/3D.md D1) ---------------------------
  //
  // The sweep is only an instrument if sliding the split changes NOTHING
  // except the frame, so these assert the invariant parts stay invariant --
  // the domain, the body's place in it, the reference area and the relative
  // speed -- and that the two ENDS of the sweep still resolve to exactly the
  // pure tow and the pure stream this suite already measures.
  const fallLeg = (tow, stream) => resolveScenario('fall', { n: 12, re: 300, tow, stream });

  ok('fall: the pure tow and the pure stream are unchanged by the split', () => {
    const tow = fallLeg(0.04, 0);
    assert.strictEqual(tow.pinned, false, 'a towed body moves');
    assert.strictEqual(tow.noFluidForce, true, 'a towed body is on rails; the force is recorded, not applied');
    close(tow.body.v[0], 0.04, 1e-12, 'tow body velocity');
    assert.deepStrictEqual(tow.gravity, [0, 0, 0], 'a towed body is not falling');
    assert.deepStrictEqual(tow.window, [1, 0, 0], 'a towed body needs the window to follow it');
    close(tow.sponge.u[0], 0, 1e-12, 'a tow moves through STILL fluid');
    close(tow.body.x[0], tow.dims[0] - 2 * 12, 1e-9, 'a tow sits 2n from the far end, wake room behind');

    const str = fallLeg(0, 0.04);
    assert.strictEqual(str.pinned, true, 'a streamed body is pinned');
    assert.strictEqual(str.noFluidForce, false);
    close(str.body.v[0], 0, 1e-12, 'a pinned body does not move');
    assert.deepStrictEqual(str.window, [0, 0, 0], 'a pinned body has nothing for a window to follow');
    close(str.sponge.u[0], 0.04, 1e-12, 'a streamed body sits in a freestream');
    close(str.body.x[0], 2 * 12, 1e-9, 'a streamed body sits 2n from the inlet');
  });

  ok('fall: a free fall is untouched by either knob existing', () => {
    const f = fallLeg(0, 0);
    assert.strictEqual(f.pinned, false);
    assert.strictEqual(f.noFluidForce, false, 'a free fall is driven BY the fluid force');
    assert.ok(f.gravity[0] > 0, 'a free fall has gravity');
    assert.strictEqual(f.uRel, null, 'a free fall has no prescribed relative speed -- its speed is the answer');
    assert.deepStrictEqual(f.window, [1, 0, 0]);
    close(f.body.x[0], f.dims[0] - 2 * 12, 1e-9, 'a falling body leaves its wake behind it');
  });

  ok('fall: sliding the split changes the frame and NOTHING else', () => {
    const U = 0.04;
    const legs = [0, 0.25, 0.5, 0.75, 1].map(a => fallLeg(a * U, a * U - U));
    const ref = legs[0];
    for (const leg of legs) {
      // The one thing the flow may depend on, held fixed by construction.
      close(leg.uRelSigned, -U, 1e-12, 'relative speed must not move across the sweep');
      close(leg.uRel, U, 1e-12, 'the Cd normalization must not move across the sweep');
      assert.deepStrictEqual(leg.dims, ref.dims, 'same domain');
      close(leg.area, ref.area, 1e-12, 'same reference area');
      close(leg.nu, ref.nu, 1e-12, 'same viscosity');
      close(leg.body.x[0], ref.body.x[0], 1e-9, 'same place in the domain');
      close(leg.sponge.width, ref.sponge.width, 1e-12, 'same sponge');
      // On rails in one of the two ways there are: the pinned end does not
      // integrate at all, every other leg integrates with the fluid force
      // switched off. Either way the trajectory is prescribed, which is what
      // makes the legs comparable.
      assert.ok(leg.pinned || leg.noFluidForce, 'every leg must be on rails');
      // NO LEG MAY EXCEED THE RELATIVE SPEED. A split that let a velocity
      // grow past U would change the Mach number alongside the frame, and
      // the sweep would no longer be one variable.
      assert.ok(Math.max(Math.abs(leg.tow), Math.abs(leg.sponge.u[0])) <= U + 1e-12,
        `leg tow=${leg.tow} stream=${leg.sponge.u[0]} exceeds U=${U}`);
    }
    // The ends ARE the two legs the suite already measures.
    assert.strictEqual(legs[0].pinned, true, 'a = 0 is the pinned leg');
    assert.strictEqual(legs[4].pinned, false, 'a = U is the towed leg');
    close(legs[4].sponge.u[0], 0, 1e-12, 'a = U tows through still fluid');
  });

  ok('fall: a body towed in -x is a tow, not a free fall', () => {
    // `tow > 0` would have read this as "no tow": gravity back on, the force
    // applied, a different experiment reported under the same name.
    const leg = fallLeg(-0.04, 0);
    assert.strictEqual(leg.noFluidForce, true);
    assert.deepStrictEqual(leg.gravity, [0, 0, 0]);
    close(leg.uRelSigned, 0.04, 1e-12);
    close(leg.body.x[0], 2 * 12, 1e-9, 'a body moving -x leaves its wake at +x, so it needs room there');
  });

  // --- card: the blunt-plate measurement leg (plans/3D.md D1) --------------
  //
  // THE ALIGNMENT IS THE WHOLE VALUE OF THE CASE, so it is asserted by
  // COUNTING CELLS rather than by checking a coordinate: the claim is "the
  // discrete body IS the nominal box", and only a count says that.
  const cardLeg = (o) => resolveScenario('card',
    { n: 16, re: 200, u_t: 0.05, span: 1, aspect: 0.125, tilt: 0, ...o });

  // Solid cells along one axis: centres are integers, a box of half-extent h
  // about `centre` owns the cells with |i - centre| < h.
  const solidSpan = (centre, h, extent) => {
    let count = 0;
    for (let i = 0; i < extent; i++) if (Math.abs(i - centre) < h) count++;
    return count;
  };

  ok('card: a prescribed, untilted plate is EXACTLY its nominal box on the lattice', () => {
    for (const leg of [cardLeg({ tow: 0, stream: -0.05 }), cardLeg({ tow: 0.05, stream: 0 })]) {
      // World axes are (thickness, chord, span) = (x, y, z) -- qFlat's
      // permutation -- and the half-extents pair with them in that order.
      const a = leg.n / 2, b = leg.span * leg.n / 2, c = leg.aspect * leg.n / 2;
      const want = [2 * c, 2 * a, 2 * b];
      for (const [i, h] of [c, a, b].entries()) {
        const got = solidSpan(leg.body.x[i], h, leg.dims[i]);
        assert.strictEqual(got, want[i],
          `axis ${'xyz'[i]}: ${got} solid cells against the nominal ${want[i]}`);
      }
      // ...which is what makes `area` the frontal area rather than 12% over it.
      close(solidSpan(leg.body.x[1], a, leg.dims[1]) * solidSpan(leg.body.x[2], b, leg.dims[2]),
        leg.area, 1e-9, 'discrete frontal area must equal the one Cd is normalized by');
    }
  });

  ok('card: a half-integer half-extent snaps the other way', () => {
    // n = 24 makes the semi-thickness 1.5, which wants a cell CENTRE where
    // the integer half-extents want a corner. One rule, two cases, and a
    // version that snapped every axis the same way would be one cell short
    // on this one.
    const leg = resolveScenario('card',
      { n: 24, re: 200, u_t: 0.05, span: 1, aspect: 0.125, tilt: 0, tow: 0, stream: -0.05 });
    close(leg.aspect * leg.n / 2, 1.5, 1e-12, 'this case exists for the half-integer');
    assert.strictEqual(solidSpan(leg.body.x[0], 1.5, leg.dims[0]), 3, 'thickness must be 2c = 3 cells');
    assert.strictEqual(solidSpan(leg.body.x[1], 12, leg.dims[1]), 24, 'chord must be 2a = 24 cells');
  });

  ok('card: a free fall and a tilted leg are NOT snapped', () => {
    // The free-fall card is the published case and every recorded number for
    // it predates this; a tilted plate cannot be lattice-aligned at all.
    const free = resolveScenario('card', { n: 16 });
    assert.ok(free.body.x.every(v => Number.isInteger(v) || Math.abs(v - Math.round(v)) < 1e-9),
      `a free fall keeps its plain centring, got ${free.body.x}`);
    const tilted = cardLeg({ tow: 0, stream: -0.05, tilt: 0.15 });
    assert.ok(Math.abs(tilted.body.x[1] - Math.round(tilted.body.x[1])) < 1e-9,
      'a tilted prescribed leg is not snapped either -- there is nothing to align to');
  });

  ok('card: the prescribed domain is wider than the free-fall one', () => {
    // A bluff body at the free-fall domain's 5% blockage carries a wall
    // correction the size of the effect a literature comparison is after.
    const free = resolveScenario('card', { n: 16 });
    const leg = cardLeg({ tow: 0, stream: -0.05 });
    assert.ok(leg.blockage < free.blockage / 2,
      `prescribed blockage ${leg.blockage} should be far under the free fall's ${free.blockage}`);
    assert.ok(leg.blockage < 0.02, `prescribed blockage ${leg.blockage} should be under 2%`);
    assert.deepStrictEqual(free.dims, [96, 80, 64], 'the FREE-fall domain must not move');
  });

  ok('resolveScenario rejects an unknown name instead of falling back', () => {
    assert.throws(() => resolveScenario('duckt'), /unknown scenario/);
  });

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
