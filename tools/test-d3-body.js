#!/usr/bin/env node
// GPU-free tests for d3-body.mjs -- the 6-DOF rigid body (plans/3D.md M2).
//
// THREE THINGS ARE AT RISK HERE, and each gets its own kind of check.
//
// (A) THE STRUCT LAYOUT. d3-body.mjs packs a flat f32 array that
//     shaders/common_d3_geometry.wgsl reads as a BodyState3D. A field-order
//     mismatch does not crash -- it rotates the body by someone else's
//     quaternion and applies force to the wrong axis. So the WGSL struct is
//     PARSED OUT OF THE SHADER and compared field by field, the same way
//     tools/test-f-pack.js guards the packed-f layout.
//
// (B) THE SIGNED DISTANCES. common_geometry.wgsl's header records at length
//     what an algebraic surrogate cost in 2D (a chi band anisotropic by the
//     body's aspect ratio, 8x at the default card) and plans/3D.md sec 1.3
//     notes it distorts along two axes in 3D. "True signed distance" is a
//     testable claim: |grad phi| = 1 everywhere off the surface. Checked by
//     finite differences, plus a brute-force nearest-point comparison
//     against a densely sampled surface -- which shares no code with the
//     closed forms.
//
// (C) THE INTEGRATOR. It is the reference shaders/d3_physics.wgsl mirrors.
//     Checked against the conservation laws it must obey (|L| exactly, and
//     rotational energy to truncation order at zero torque) and against the
//     qualitative consequence that motivates 6-DOF at all: the
//     intermediate-axis instability. A body spun near its middle principal
//     axis MUST flip. An integrator with a sign error in the gyroscopic
//     coupling conserves |L| just fine and never flips, so the conservation
//     check alone would not catch it.
//
// Run: node tools/test-d3-body.js   (also picked up by `make test`)

const assert = require('assert');
const fs = require('fs');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}
const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b} (|diff| ${Math.abs(a - b).toExponential(2)} > ${tol.toExponential(2)})`);

(async () => {
  const root = path.join(__dirname, '..');
  const B = await import(path.join(root, 'd3-body.mjs'));
  const {
    SHAPE, sdfBody, bodyVolume, principalInertia, qIdentity, qMul, qConj, qNormalize,
    qFromAxisAngle, qRotate, qRotateInv, omegaFromL, rotationalEnergy, stepFreeBody,
    makeBodyState, BODY_FIELDS, packBodyState, unpackBodyState,
  } = B;

  // --- (A) struct layout ---------------------------------------------------
  ok('BODY_FIELDS matches the BodyState3D struct in the shader, field for field', () => {
    const src = fs.readFileSync(path.join(root, 'shaders', 'common_d3_geometry.wgsl'), 'utf8');
    const m = src.match(/struct\s+BodyState3D\s*\{([\s\S]*?)\n\}/);
    assert.ok(m, 'no BodyState3D struct found in common_d3_geometry.wgsl');
    const fields = m[1]
      .replace(/\/\/[^\n]*/g, '')           // strip comments before splitting
      .split(',')
      .map(s => s.trim()).filter(Boolean)
      .map(s => {
        const f = s.match(/^(\w+)\s*:\s*(\w+)$/);
        assert.ok(f, `unparseable struct member "${s}"`);
        assert.strictEqual(f[2], 'f32', `member ${f[1]} is ${f[2]}, but the host packs a Float32Array`);
        return f[1];
      });
    assert.deepStrictEqual(fields, BODY_FIELDS,
      'shader struct order and d3-body.mjs BODY_FIELDS disagree -- the body would be read with fields shifted');
  });

  ok('pack/unpack round-trips and lands each value at its named slot', () => {
    const s = makeBodyState({ shape: { kind: SHAPE.ROUNDBOX, a: 8, b: 3, c: 1, r: 0.5 }, x: [10, 20, 30] });
    const arr = packBodyState(s, { pinned: true, vMax: 0.3, oMax: 0.05 });
    assert.strictEqual(arr.length, BODY_FIELDS.length);
    const u = unpackBodyState(arr);
    // Compared against Math.fround of each value, not the value itself: the
    // buffer is a Float32Array, so 0.3 is not 0.3 on the other side and an
    // exact-equality check would be testing single precision rather than
    // the layout this is about.
    const f32eq = (got, want, what) => close(got, Math.fround(want), 0, what);
    f32eq(u.cx, 10, 'cx'); f32eq(u.cy, 20, 'cy'); f32eq(u.cz, 30, 'cz');
    f32eq(u.qw, 1, 'qw'); f32eq(u.qx, 0, 'qx');
    f32eq(u.a, 8, 'a'); f32eq(u.b, 3, 'b'); f32eq(u.c, 1, 'c'); f32eq(u.r, 0.5, 'r');
    f32eq(u.shape, SHAPE.ROUNDBOX, 'shape'); f32eq(u.pinned, 1, 'pinned');
    f32eq(u.v_max, 0.3, 'v_max'); f32eq(u.o_max, 0.05, 'o_max');
    f32eq(u.fx, 0, 'fx'); f32eq(u.tz, 0, 'tz');
  });

  ok('shader shape constants match d3-body.mjs SHAPE', () => {
    const src = fs.readFileSync(path.join(root, 'shaders', 'common_d3_geometry.wgsl'), 'utf8');
    for (const [name, val] of [['SPHERE', SHAPE.SPHERE], ['SPHEROID', SHAPE.SPHEROID], ['ROUNDBOX', SHAPE.ROUNDBOX]]) {
      const m = src.match(new RegExp(`const\\s+SHAPE_${name}\\s*(?::\\s*u32\\s*)?=\\s*(\\d+)u`));
      assert.ok(m, `no SHAPE_${name} constant in the shader`);
      assert.strictEqual(parseInt(m[1]), val, `SHAPE_${name}`);
    }
  });

  // --- (B) signed distances ------------------------------------------------
  const SHAPES = [
    ['sphere', { kind: SHAPE.SPHERE, a: 7 }],
    ['spheroid oblate', { kind: SHAPE.SPHEROID, a: 9, c: 3 }],
    ['spheroid prolate', { kind: SHAPE.SPHEROID, a: 3, c: 11 }],
    ['roundbox plate', { kind: SHAPE.ROUNDBOX, a: 10, b: 6, c: 1.5, r: 0.4 }],
    ['roundbox cube', { kind: SHAPE.ROUNDBOX, a: 5, b: 5, c: 5, r: 0 }],
  ];

  // A true signed distance has |grad phi| = 1 everywhere it is
  // differentiable. This is THE property an algebraic surrogate fails, and
  // it fails by exactly the anisotropy factor that distorted the 2D chi
  // band -- so it is checked directly rather than inferred.
  for (const [name, shape] of SHAPES) {
    ok(`${name}: |grad phi| = 1 (it is a true distance, not an algebraic surrogate)`, () => {
      const h = 1e-3;
      let worst = 0, worstAt = null;
      for (const p of [[13, 2, 1], [0, 0, 14], [8, 8, 8], [20, -3, 2], [-11, 5, -6], [2, 12, 9]]) {
        const g = [0, 1, 2].map(i => {
          const a = p.slice(), b = p.slice();
          a[i] += h; b[i] -= h;
          return (sdfBody(a, shape) - sdfBody(b, shape)) / (2 * h);
        });
        const mag = Math.hypot(...g);
        if (Math.abs(mag - 1) > worst) { worst = Math.abs(mag - 1); worstAt = p; }
      }
      assert.ok(worst < 2e-3, `|grad phi| deviates by ${worst.toExponential(2)} at ${JSON.stringify(worstAt)}`);
    });

    // Independent instrument: brute-force distance to a dense sampling of
    // the surface. Shares no code with the closed forms above, so a wrong
    // formula cannot agree with it by construction.
    ok(`${name}: matches a brute-force nearest-surface-point search`, () => {
      const pts = surfaceSamples(shape, 240);
      for (const p of [[15, 3, 2], [0, 0, 18], [9, 9, 9], [-13, 4, -5]]) {
        let best = Infinity;
        for (const q of pts) best = Math.min(best, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
        const got = sdfBody(p, shape);
        assert.ok(got > 0, `${JSON.stringify(p)} should be outside`);
        // The brute-force minimum is an OVER-estimate (the true nearest
        // point lies between samples), so it may only exceed the analytic
        // value, never fall below it by more than the sampling pitch.
        assert.ok(got <= best + 1e-9, `analytic ${got} exceeds brute force ${best}`);
        assert.ok(best - got < 0.05 * Math.max(1, got), `analytic ${got} vs brute force ${best} at ${JSON.stringify(p)}`);
      }
    });

    ok(`${name}: sign is negative inside, zero on the surface`, () => {
      assert.ok(sdfBody([0, 0, 0], shape) < 0, 'centre must be inside');
      for (const q of surfaceSamples(shape, 60)) {
        close(sdfBody(q, shape), 0, 2e-3, `surface point ${JSON.stringify(q.map(v => +v.toFixed(2)))}`);
      }
    });
  }

  ok('roundbox interior distance is the distance to the nearest face', () => {
    const shape = { kind: SHAPE.ROUNDBOX, a: 10, b: 6, c: 2, r: 0 };
    close(sdfBody([0, 0, 0], shape), -2, 1e-12, 'centre -> nearest face is the c face');
    close(sdfBody([0, 0, 1], shape), -1, 1e-12, 'one unit from the c face');
    close(sdfBody([9, 0, 0], shape), -1, 1e-12, 'one unit from the a face');
  });

  ok('a spheroid with all axes equal reduces exactly to a sphere', () => {
    for (const p of [[3, 4, 5], [0, 0, 9], [12, 0, 0]]) {
      close(sdfBody(p, { kind: SHAPE.SPHEROID, a: 6, c: 6 }), sdfBody(p, { kind: SHAPE.SPHERE, a: 6 }), 1e-9, `at ${p}`);
    }
  });

  // --- inertia -------------------------------------------------------------
  ok('volumes and principal moments match the textbook forms', () => {
    close(bodyVolume({ kind: SHAPE.SPHERE, a: 3 }), 4 / 3 * Math.PI * 27, 1e-9, 'sphere volume');
    const m = 5;
    assert.deepStrictEqual(principalInertia({ kind: SHAPE.SPHERE, a: 3 }, m).map(v => +v.toFixed(9)),
      [0.4 * m * 9, 0.4 * m * 9, 0.4 * m * 9].map(v => +v.toFixed(9)));
    // Box: I_xx = m(h^2 + d^2)/12 with full extents h = 2b, d = 2c.
    const bx = { kind: SHAPE.ROUNDBOX, a: 2, b: 3, c: 4, r: 0 };
    const I = principalInertia(bx, m);
    close(I[0], m * ((2 * 3) ** 2 + (2 * 4) ** 2) / 12, 1e-9, 'I_xx');
    close(I[1], m * ((2 * 2) ** 2 + (2 * 4) ** 2) / 12, 1e-9, 'I_yy');
    close(I[2], m * ((2 * 2) ** 2 + (2 * 3) ** 2) / 12, 1e-9, 'I_zz');
  });

  ok('a flat plate has a strongly ordered inertia (the reason 6-DOF is interesting)', () => {
    const I = principalInertia({ kind: SHAPE.ROUNDBOX, a: 12, b: 12, c: 0.75, r: 0 }, 1);
    assert.ok(I[2] > I[0] * 1.9, `I_zz ${I[2]} should be ~2x I_xx ${I[0]} for a thin square plate`);
  });

  // --- quaternions ---------------------------------------------------------
  ok('quaternion algebra: identity, conjugate inverse, unit norm preserved', () => {
    const q = qNormalize(qFromAxisAngle([1, 2, 3], 0.7));
    close(Math.hypot(...q), 1, 1e-12, '|q|');
    const p = qMul(q, qConj(q));
    close(p[0], 1, 1e-12, 'q q* real part');
    for (let i = 1; i < 4; i++) close(p[i], 0, 1e-12, `q q* imaginary ${i}`);
    assert.deepStrictEqual(qNormalize([0, 0, 0, 0]), qIdentity());
  });

  ok('qRotate is a rotation: preserves length, inverts, and composes', () => {
    const q = qFromAxisAngle([0.3, -1, 0.5], 1.1);
    const v = [2, -3, 5];
    const r = qRotate(q, v);
    close(Math.hypot(...r), Math.hypot(...v), 1e-12, 'length preserved');
    const back = qRotateInv(q, r);
    for (let i = 0; i < 3; i++) close(back[i], v[i], 1e-12, `inverse component ${i}`);
    // 90 degrees about +z sends +x to +y.
    const qz = qFromAxisAngle([0, 0, 1], Math.PI / 2);
    const e = qRotate(qz, [1, 0, 0]);
    close(e[0], 0, 1e-12, 'x'); close(e[1], 1, 1e-12, 'y'); close(e[2], 0, 1e-12, 'z');
  });

  ok('rotating the body rotates its SDF with it', () => {
    const shape = { kind: SHAPE.ROUNDBOX, a: 10, b: 2, c: 2, r: 0 };
    const q = qFromAxisAngle([0, 0, 1], Math.PI / 2);
    // A point 9 along +x is inside the unrotated long axis; after a 90-degree
    // turn about z, the same is true 9 along +y.
    assert.ok(sdfBody([9, 0, 0], shape) < 0);
    assert.ok(sdfBody(qRotateInv(q, [0, 9, 0]), shape) < 0, 'long axis should now point along +y');
    assert.ok(sdfBody(qRotateInv(q, [9, 0, 0]), shape) > 0, 'and no longer along +x');
  });

  // --- (C) integrator ------------------------------------------------------
  ok('omega and L are consistent: omegaFromL inverts the L built by makeBodyState', () => {
    const s = makeBodyState({
      shape: { kind: SHAPE.ROUNDBOX, a: 4, b: 3, c: 1, r: 0 },
      x: [0, 0, 0], q: qFromAxisAngle([1, 1, 0], 0.4), omega: [0.01, -0.02, 0.03],
    });
    const w = omegaFromL(s.q, s.ibody, s.L);
    for (let i = 0; i < 3; i++) close(w[i], s.omega[i], 1e-12, `omega component ${i}`);
  });

  ok('free body: |L| is conserved EXACTLY and energy to truncation order', () => {
    let s = makeBodyState({
      shape: { kind: SHAPE.ROUNDBOX, a: 6, b: 4, c: 1, r: 0 },
      x: [0, 0, 0], omega: [0.02, 0.01, 0.005],
    });
    const L0 = Math.hypot(...s.L), E0 = rotationalEnergy(s.q, s.ibody, s.L);
    for (let i = 0; i < 20000; i++) s = stepFreeBody(s);
    close(Math.hypot(...s.L), L0, 1e-12 * L0, '|L| after 20000 steps');
    const E = rotationalEnergy(s.q, s.ibody, s.L);
    assert.ok(Math.abs(E - E0) / E0 < 0.02, `rotational energy drifted ${((E - E0) / E0 * 100).toFixed(2)}% over 20000 steps`);
    close(Math.hypot(...s.q), 1, 1e-9, '|q| stays normalized');
  });

  // THE test for the gyroscopic coupling. A rigid body spun about its
  // INTERMEDIATE principal axis is unstable and must periodically flip
  // (the Dzhanibekov / tennis-racket effect). An integrator with a sign
  // error in that coupling still conserves |L| perfectly and simply never
  // flips -- so the conservation check above cannot catch it, and this can.
  ok('free body flips about its intermediate axis (tennis-racket theorem)', () => {
    const shape = { kind: SHAPE.ROUNDBOX, a: 6, b: 3, c: 1, r: 0 };
    const I = principalInertia(shape, 1);
    // With a > b > c the moments order I_xx < I_yy < I_zz, so y is the
    // intermediate axis. Assert that before relying on it.
    assert.ok(I[0] < I[1] && I[1] < I[2], `expected I ordered, got ${I.map(v => v.toFixed(3))}`);
    let s = makeBodyState({ shape, x: [0, 0, 0], omega: [1e-4, 0.02, 1e-4] });
    // Track the body-frame y axis in world coordinates; a flip is that axis
    // reversing, i.e. its dot product with its start going through -1.
    const y0 = qRotate(s.q, [0, 1, 0]);
    let minDot = 1;
    for (let i = 0; i < 400000; i++) {
      s = stepFreeBody(s);
      if (i % 100 === 0) {
        const y = qRotate(s.q, [0, 1, 0]);
        minDot = Math.min(minDot, y[0] * y0[0] + y[1] * y0[1] + y[2] * y0[2]);
      }
    }
    assert.ok(minDot < -0.9, `intermediate axis never flipped (min alignment ${minDot.toFixed(3)}, want < -0.9)`);
  });

  ok('a body spun about a STABLE axis does not flip', () => {
    const shape = { kind: SHAPE.ROUNDBOX, a: 6, b: 3, c: 1, r: 0 };
    for (const [axis, name] of [[[1, 0, 0], 'minor'], [[0, 0, 1], 'major']]) {
      let s = makeBodyState({ shape, x: [0, 0, 0], omega: axis.map(c => c * 0.02).map((c, i) => c + (i === 1 ? 1e-4 : 0)) });
      const a0 = qRotate(s.q, axis);
      let minDot = 1;
      for (let i = 0; i < 400000; i++) {
        s = stepFreeBody(s);
        if (i % 100 === 0) {
          const a = qRotate(s.q, axis);
          minDot = Math.min(minDot, a[0] * a0[0] + a[1] * a0[1] + a[2] * a0[2]);
        }
      }
      assert.ok(minDot > 0.9, `${name}-axis spin should be stable, but alignment fell to ${minDot.toFixed(3)}`);
    }
  });

  ok('linear motion: constant force gives the textbook trajectory', () => {
    let s = makeBodyState({ shape: { kind: SHAPE.SPHERE, a: 2 }, x: [0, 0, 0] });
    const g = [0, -1e-4, 0];
    const n = 1000;
    for (let i = 0; i < n; i++) s = stepFreeBody(s, { gravity: g });
    // Semi-implicit Euler: v_n = n g, x_n = g n(n+1)/2. Exact, not approximate.
    // Relative, not absolute: 1000 f64 additions of -1e-4 accumulate ~2e-15
    // of round-off, which is the arithmetic working correctly.
    close(s.v[1], n * g[1], 1e-12 * Math.abs(n * g[1]), 'velocity');
    close(s.x[1], g[1] * n * (n + 1) / 2, Math.abs(g[1]) * 1e-6, 'position');
  });

  ok('magnitude clamps preserve direction (they contain a blowup, not steer it)', () => {
    let s = makeBodyState({ shape: { kind: SHAPE.SPHERE, a: 2 }, x: [0, 0, 0], v: [0.3, 0.4, 0] });
    s = stepFreeBody(s, { vMax: 0.1 });
    close(Math.hypot(...s.v), 0.1, 1e-12, 'clamped speed');
    close(s.v[0] / s.v[1], 0.3 / 0.4, 1e-12, 'direction unchanged by the clamp');
  });

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();

// Dense sampling of a shape's surface, built from its PARAMETRIC form --
// deliberately a different construction from the implicit distance
// functions under test, so agreement between them is evidence.
function surfaceSamples(shape, n) {
  const out = [];
  const { a, b, c, r } = shape;
  if (shape.kind === 0 || shape.kind === 1) {
    const A = a, C = shape.kind === 0 ? a : c;
    for (let i = 0; i <= n; i++) {
      const th = Math.PI * i / n;
      for (let j = 0; j < n; j++) {
        const ph = 2 * Math.PI * j / n;
        out.push([A * Math.sin(th) * Math.cos(ph), A * Math.sin(th) * Math.sin(ph), C * Math.cos(th)]);
      }
    }
    return out;
  }
  // Rounded box: the surface is the Minkowski sum of the shrunken box's
  // boundary with a sphere of radius r, so sample the shrunken box's faces
  // and push each point out along the box's outward normal.
  const ha = a - r, hb = b - r, hc = c - r;
  const k = Math.max(8, Math.round(n / 6));
  const lin = (h) => Array.from({ length: k + 1 }, (_, i) => -h + 2 * h * i / k);
  // The (u, v) ranges must follow the axis permutation: face `axis` varies
  // along (axis+1)%3 then (axis+2)%3, so the y face varies over z then x,
  // NOT x then z. Getting this wrong put "surface" samples at |z| = 9.6 on
  // a body of half-thickness 1.5 -- which the test caught, and which is a
  // fair warning about how easy the same slip is in the shader.
  for (const [u, v, fix, axis] of [[lin(hb), lin(hc), ha, 0], [lin(hc), lin(ha), hb, 1], [lin(ha), lin(hb), hc, 2]]) {
    for (const uu of u) for (const vv of v) for (const sgn of [1, -1]) {
      const p = [0, 0, 0];
      p[axis] = sgn * fix;
      p[(axis + 1) % 3] = uu;
      p[(axis + 2) % 3] = vv;
      const nrm = [0, 0, 0]; nrm[axis] = sgn;
      out.push([p[0] + r * nrm[0], p[1] + r * nrm[1], p[2] + r * nrm[2]]);
    }
  }
  return out;
}
