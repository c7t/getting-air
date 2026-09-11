#!/usr/bin/env node
// GPU-free tests for d3-criterion.mjs -- the Q-criterion that decides which
// blocks a DETACHED VORTEX needs refined (plans/3D.md M8.4).
//
// WHAT IS ACTUALLY UNDER TEST, and it is not the arithmetic. The choice of Q
// over |omega| is a DESIGN DECISION with a stated reason: in 3D, |omega|
// flags every shear layer -- the boundary layer over the whole body and every
// free shear layer feeding the wake -- while Q is positive only where
// rotation dominates strain, which is what isolates a vortex CORE from the
// shear that produced it. That claim is checkable in closed form, so it is
// checked here rather than asserted in a comment:
//
//   PURE SHEAR      Q = 0 exactly, at every shear rate, while |omega| = gamma
//   SOLID ROTATION  Q = omega^2 > 0, and |omega| = 2*omega
//   UNIFORM FLOW    Q = 0 and |omega| = 0
//
// The first is the whole decision. A criterion that failed it would refine
// the entire boundary layer and still look like it was finding vortices.
//
// The second half of the file scores the finite-difference path against the
// analytic one on a field with curvature, which is the same two-level
// discipline tools/test-d3-scenarios.js uses: the closed-form cases check the
// FORMULA, the differenced case checks the STENCIL, and neither can cover for
// the other.
//
// Run: node tools/test-d3-criterion.js   (also picked up by `make test`)

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
  const C = await import(path.join(root, 'd3-criterion.mjs'));
  const { gradU, qOfGrad, omegaOfGrad, qRef, convectionLead, Q_THRESHOLD } = C;

  // J from an analytic velocity field, by the same centred differences the
  // shader uses, so a test of the stencil is a test of the shader's stencil.
  const gradAt = (u, p, h) => gradU(
    [0, 1, 2].map(j => u(p.map((c, i) => (i === j ? c + h : c)))),
    [0, 1, 2].map(j => u(p.map((c, i) => (i === j ? c - h : c)))), h);

  ok('UNIFORM FLOW: Q = 0 and |omega| = 0', () => {
    const u = () => [0.05, -0.02, 0.01];
    const J = gradAt(u, [1, 2, 3], 1);
    close(qOfGrad(J), 0, 1e-12, 'Q');
    close(omegaOfGrad(J), 0, 1e-12, '|omega|');
  });

  // THE ONE THAT IS THE DECISION.
  ok('PURE SHEAR: Q = 0 EXACTLY while |omega| = gamma -- the reason for Q', () => {
    for (const gamma of [0.001, 0.05, 1, 17.5]) {
      const u = (p) => [gamma * p[1], 0, 0];
      const J = gradAt(u, [0, 0, 0], 1);
      close(qOfGrad(J), 0, 1e-12 * Math.max(1, gamma * gamma),
        `Q at gamma=${gamma} must be identically zero`);
      close(omegaOfGrad(J), gamma, 1e-12 * Math.max(1, gamma),
        `|omega| at gamma=${gamma} must be gamma -- if this were the criterion,`
        + ' every boundary layer would refine');
    }
    // And the two disagree by an unbounded amount, which is the point: no
    // threshold on |omega| can separate shear from rotation.
    const g = 100;
    const J = gradAt((p) => [g * p[1], 0, 0], [0, 0, 0], 1);
    assert.strictEqual(qOfGrad(J) < 1e-9, true, 'Q stays zero as shear grows');
    assert.ok(omegaOfGrad(J) > 99, '|omega| grows without bound');
  });

  ok('SOLID-BODY ROTATION: Q = omega^2 > 0, and it is rotation not strain', () => {
    for (const w of [0.01, 0.5, 3]) {
      const u = (p) => [-w * p[1], w * p[0], 0];
      const J = gradAt(u, [0, 0, 0], 1);
      close(qOfGrad(J), w * w, 1e-10 * Math.max(1, w * w), `Q at omega=${w}`);
      close(omegaOfGrad(J), 2 * w, 1e-10 * Math.max(1, w), `|omega| at omega=${w}`);
    }
  });

  ok('a vortex CORE is flagged and the shear layer beside it is not', () => {
    // A Rankine-style core: solid rotation inside r0, and a pure shear layer
    // outside it with the SAME peak velocity. The two have comparable
    // |omega|; only one is a vortex.
    const w = 0.1;
    const qCore = qOfGrad(gradAt((p) => [-w * p[1], w * p[0], 0], [0, 0, 0], 1));
    const qShear = qOfGrad(gradAt((p) => [2 * w * p[1], 0, 0], [0, 0, 0], 1));
    assert.ok(qCore > 0, 'the core must have Q > 0');
    close(qShear, 0, 1e-12, 'the shear layer beside it must have Q = 0');
    // Same peak velocity, comparable |omega|, opposite verdicts.
    assert.ok(omegaOfGrad(gradAt((p) => [2 * w * p[1], 0, 0], [0, 0, 0], 1)) >= 2 * w,
      'the shear layer has vorticity comparable to the core -- which is what |omega| would see');
  });

  ok('Q is invariant under rotation of the frame (it is a tensor invariant)', () => {
    // Rotate a solid-body vortex about x by 90 degrees: same flow, different
    // axes, same Q. A criterion whose answer depended on the lattice
    // orientation would refine differently depending on which way the wake
    // happened to point -- and benchmarks/d3.json already records this
    // solver picking a 45-degree lattice diagonal when nothing else chose.
    const w = 0.25;
    const zAxis = (p) => [-w * p[1], w * p[0], 0];
    const xAxis = (p) => [0, -w * p[2], w * p[1]];
    close(qOfGrad(gradAt(zAxis, [0, 0, 0], 1)),
          qOfGrad(gradAt(xAxis, [0, 0, 0], 1)), 1e-12, 'Q about z vs about x');
  });

  ok('the finite-difference stencil converges at SECOND order on a curved field', () => {
    // The closed-form cases above are all LINEAR in u, so centred differences
    // are exact on them and they cannot see a stencil error at all. This one
    // has curvature, so the difference between the differenced Q and the
    // analytic Q is the stencil -- and it must fall by 4x per halving.
    // A Taylor-Green cell, which is a real vortex with curvature:
    //   u = ( A sin(kx) cos(ky), -A cos(kx) sin(ky), 0 )
    // whose Q is (A k)^2 [ sin^2(kx) sin^2(ky) - cos^2(kx) cos^2(ky) ].
    //
    // THE FIRST FIELD TRIED HERE WAS u = (sin ky, sin kz, sin kx), AND IT HAS
    // Q IDENTICALLY ZERO -- its J has only the three cyclic off-diagonals, so
    // |Omega|^2 and |S|^2 are the same sum and cancel exactly. The test
    // reported NaN rather than passing, because there was no error to
    // converge; a convergence check on a field whose exact answer is zero
    // measures nothing. Worth keeping as a note: "smooth and non-uniform" is
    // not the same as "has a vortex in it".
    const k = 0.3, A = 1;
    const u = (p) => [A * Math.sin(k * p[0]) * Math.cos(k * p[1]),
                      -A * Math.cos(k * p[0]) * Math.sin(k * p[1]), 0];
    const exact = (p) => (A * k) ** 2 * (
      Math.sin(k * p[0]) ** 2 * Math.sin(k * p[1]) ** 2
      - Math.cos(k * p[0]) ** 2 * Math.cos(k * p[1]) ** 2);
    const p = [0.7, 1.3, 2.1];
    assert.ok(Math.abs(exact(p)) > 1e-3, 'the sample point must actually have a vortex at it');
    let prev = null;
    for (const h of [0.4, 0.2, 0.1]) {
      const err = Math.abs(qOfGrad(gradAt(u, p, h)) - exact(p));
      if (prev !== null) {
        const ratio = prev / err;
        assert.ok(ratio > 3.2 && ratio < 4.8,
          `halving h must cut the error ~4x, got ${ratio.toFixed(2)}`);
      }
      prev = err;
    }
  });

  ok('qRef makes the threshold dimensionless, so one number works at any scale', () => {
    // The same flow at half the velocity scale has a QUARTER the Q, so a bare
    // threshold would mean something different in every scenario. Normalizing
    // by (U/D)^2 is what makes `?qthresh=` portable -- the same move the
    // renderer makes with U_SCALE/V_SCALE.
    const w = 0.1, D = 16, U = 0.05;
    const q1 = qOfGrad(gradAt((p) => [-w * p[1], w * p[0], 0], [0, 0, 0], 1));
    const q2 = qOfGrad(gradAt((p) => [-2 * w * p[1], 2 * w * p[0], 0], [0, 0, 0], 1));
    close(q2 / q1, 4, 1e-9, 'doubling the velocity scale quadruples Q');
    close(q1 / qRef(U, D) / (q2 / qRef(2 * U, D)), 1, 1e-9,
      'and the NORMALIZED value is unchanged, which is what makes it a knob');
    assert.ok(Q_THRESHOLD > 0, 'the default threshold is clear of the Q > 0 noise floor');
  });

  ok('the convection lead is how far the flow carries a vortex between decisions', () => {
    // Zero at manageEvery = 1 (the geometry criterion's own note makes the
    // same observation about its lead term), and linear in both arguments.
    close(convectionLead(16, 0.05), 0.8, 1e-12, '16 steps at u=0.05');
    close(convectionLead(1, 0.05), 0.05, 1e-12, 'one step');
    close(convectionLead(16, 0), 0, 1e-12, 'a still flow carries nothing');
    // Sign-independent: a wake convecting the other way still needs the lead.
    close(convectionLead(16, -0.05), 0.8, 1e-12, 'direction does not matter');
  });

  ok('the shader mirrors the host formula (both norms, not the compact form)', () => {
    const src = fs.readFileSync(path.join(root, 'shaders', 'common_d3_criterion.wgsl'), 'utf8');
    // The compact identity Q = -1/2 J_ij J_ji is one transposition away from
    // silently wrong, and the two files must agree about which form is used.
    assert.ok(/om2 \+= O \* O/.test(src) && /s2 \+= S \* S/.test(src),
      'the shader must accumulate both Frobenius norms separately');
    assert.ok(/0\.5f \* \(om2 - s2\)/.test(src), 'and combine them as 1/2(|Omega|^2 - |S|^2)');
    // The OR is load-bearing: geometry has already written this array.
    assert.ok(/blockWant\[[^\]]*\] = 1u;/.test(src) && !/blockWant\[[^\]]*\] = select/.test(src),
      'the shader must OR into blockWant, never assign it');
  });

  console.log(`\nd3-criterion: ${pass} test(s) passed`);
})();
