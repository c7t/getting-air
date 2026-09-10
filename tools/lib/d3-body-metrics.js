// Shared analysis for the dense 3D solver's BODY cases (plans/3D.md M2),
// used identically by tools/validate-3d.js and tools/validate-all.js.
// Companion to tools/lib/d3-metrics.js, which owns the M1 fluid gates.
//
// Two cases, testing two very different things.
//
// SPHERE -- flow past a pinned sphere, Cd against the Schiller-Naumann
// correlation. This is the 3D counterpart of the 2D cylinder harness, and
// it is scored the same way, with one addition: as well as an absolute
// tolerance it checks that REFINING THE GRID MOVES Cd TOWARD the reference.
// A single tolerance can be satisfied by a body of the wrong size with a
// compensating error; a convergence trend cannot, and it is the check that
// distinguishes "discretization error" from "wrong".
//
// SPIN -- the 6-DOF integrator with the fluid force switched off, compared
// step for step against d3-body.mjs's stepFreeBody(). shaders/d3_physics.wgsl
// is the hardest new code in M2 and the one whose errors are least visible
// (a wrong gyroscopic coupling still conserves |L| perfectly and simply
// tumbles wrongly), so it is checked against the host reference on REAL GPU
// CODE rather than trusted because that reference is well tested. The
// default spin is near the intermediate principal axis, so the compared
// trajectory is the tumbling one -- agreement on a steady spin would be a
// much weaker claim.

const path = require('path');

let _B = null, _S = null;
async function mods() {
  if (!_B) {
    _B = await import(path.join(__dirname, '..', '..', 'd3-body.mjs'));
    _S = await import(path.join(__dirname, '..', '..', 'd3-scenarios.mjs'));
  }
  return { B: _B, S: _S };
}

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}
async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}
function checkTol(label, measured, target, tol) {
  if (tol == null) return { pass: true, label, measured, target: null };
  const pass = measured != null && Math.abs(measured - target) <= tol;
  return { pass, label, measured, target, tol };
}

function caseUrl(baseUrl, scenario, c, extra) {
  const p = new URLSearchParams({ scenario, n: c.n, live: '0' });
  if (c.re != null) p.set('re', c.re);
  if (c.u0 != null) p.set('u0', c.u0);
  if (c.tau != null) p.set('tau', c.tau);
  if (c.q != null) p.set('q', c.q);
  if (c.bounceback) p.set('bounceback', '1');
  return `${baseUrl}/index-3d.html?${p}${extra ? `&${extra}` : ''}`;
}

// --- sphere ----------------------------------------------------------------

async function runSphereCase(Runtime, opts, c, log) {
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  const conv = p.convective;
  const settle = Math.round((c.settle_convective || 30) * conv);
  if (log) {
    log(`D=${p.D} Re=${p.re} ${p.bounceback ? 'bounceback' : 'diffuse'} Q${p.Q}: tau=${p.tau.toFixed(4)} `
      + `nu=${p.nu.toFixed(5)} domain ${p.NX}x${p.NY}x${p.NZ} blockage ${(p.blockage * 100).toFixed(2)}%`);
    log(`  D/U=${conv.toFixed(0)} steps, settling ${c.settle_convective || 30} of them = ${settle} steps`);
  }

  // Two samples, so "has it stopped moving" is measured rather than assumed.
  // Cd was observed to settle to three digits by ~20 D/U at every Re tested,
  // and the default 30 leaves margin.
  await evalOrThrow(Runtime, `window.__D3.debugStepSync(${Math.round(settle * 0.7)})`, ((opts.timeout || 600) + 60) * 1000, 'debugStepSync');
  const early = await evalOrThrow(Runtime, 'window.__D3.readBody()', 60000, 'readBody');
  await evalOrThrow(Runtime, `window.__D3.debugStepSync(${settle - Math.round(settle * 0.7)})`, ((opts.timeout || 600) + 60) * 1000, 'debugStepSync');
  const b = await evalOrThrow(Runtime, 'window.__D3.readBody()', 60000, 'readBody');

  const cd = b.cd;
  const cdRef = p.cdReference;
  const relErr = (cd - cdRef) / cdRef;
  const drift = Math.abs(cd - early.cd) / Math.abs(cd);
  // Cross-flow force should vanish by symmetry for a sphere in axial flow.
  // Normalized by Cd, so it reads as "how far from axisymmetric", and it is
  // a genuinely independent check: a wrong torque or a transposed axis in
  // the force reduction shows up here and not in Cd.
  const lateral = Math.hypot(b.cl, b.cs) / Math.abs(cd);
  if (log) log(`  Cd=${cd.toFixed(4)} vs Schiller-Naumann ${cdRef.toFixed(4)} (${(relErr * 100).toFixed(1)}%)  `
    + `settled to ${(drift * 100).toFixed(3)}%  lateral/Cd=${lateral.toExponential(2)}`);

  return {
    D: p.D, re: p.re, tau: p.tau, Q: p.Q, bounceback: !!p.bounceback,
    steps: settle, cd, cdRef, relErr, drift, lateral,
    cdCheck: checkTol('CdRelErr', relErr, 0, c.cd_tol_rel),
    settledCheck: checkTol('settleDrift', drift, 0, c.settle_tol ?? 0.01),
    lateralCheck: checkTol('lateral/Cd', lateral, 0, c.lateral_tol ?? 0.02),
  };
}

// --- spin ------------------------------------------------------------------

// Angle between two orientations, in radians. |dot| because q and -q are the
// same rotation -- comparing components directly would report a spurious
// 2*pi disagreement the moment the GPU's quaternion crossed a sign.
function quatAngle(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, d));
}

async function runSpinCase(Runtime, opts, c, log) {
  const { B } = await mods();
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  const b0 = await evalOrThrow(Runtime, 'window.__D3.readBody()', 60000, 'readBody');
  const L0 = Math.hypot(b0.Lx, b0.Ly, b0.Lz);
  if (log) log(`shape=${JSON.stringify(p.body.shape)} I=${p.body.ibody.map(v => v.toExponential(2))} |L0|=${L0.toExponential(4)}`);

  // The host reference starts from the page's OWN initial state, read back
  // off the GPU rather than rebuilt -- so a disagreement is in the
  // integrator, not in two different ideas of where the body started.
  let hs = {
    shape: p.body.shape, mass: b0.mass, ibody: [b0.ix, b0.iy, b0.iz],
    x: [b0.cx, b0.cy, b0.cz], q: [b0.qw, b0.qx, b0.qy, b0.qz],
    v: [b0.vx, b0.vy, b0.vz], L: [b0.Lx, b0.Ly, b0.Lz],
    omega: [b0.wx, b0.wy, b0.wz],
  };

  const samples = [];
  let prev = 0;
  for (const t of c.checkpoints) {
    await evalOrThrow(Runtime, `window.__D3.debugStepSync(${t - prev})`, ((opts.timeout || 600) + 60) * 1000, 'debugStepSync');
    for (let i = prev; i < t; i++) hs = B.stepFreeBody(hs, { vMax: 0.2, oMax: 0.5 });
    prev = t;
    const b = await evalOrThrow(Runtime, 'window.__D3.readBody()', 60000, 'readBody');
    const ang = quatAngle([b.qw, b.qx, b.qy, b.qz], hs.q);
    const Ln = Math.hypot(b.Lx, b.Ly, b.Lz);
    const lRel = Math.abs(Ln - L0) / L0;
    const qNorm = Math.abs(Math.hypot(b.qw, b.qx, b.qy, b.qz) - 1);
    samples.push({ t, ang, lRel, qNorm });
    if (log) log(`  step ${t}: GPU-vs-host orientation ${ang.toExponential(3)} rad, |L| drift ${lRel.toExponential(2)}, |q|-1 ${qNorm.toExponential(2)}`);
  }

  // Did the body actually tumble? A trajectory that never left its starting
  // orientation would agree with the host trivially, and prove nothing.
  const totalTurn = quatAngle([b0.qw, b0.qx, b0.qy, b0.qz], hs.q);
  const worst = samples[samples.length - 1];
  return {
    samples, L0, totalTurn,
    angCheck: checkTol('orientationRad', worst.ang, 0, c.orientation_tol),
    lCheck: checkTol('|L|Drift', Math.max(...samples.map(s => s.lRel)), 0, c.l_tol ?? 1e-6),
    qCheck: checkTol('|q|-1', Math.max(...samples.map(s => s.qNorm)), 0, 1e-4),
    // The body must have turned appreciably, or the comparison is vacuous.
    movedCheck: { pass: totalTurn > 1.0, label: 'totalTurnRad', measured: totalTurn, target: '> 1.0' },
  };
}

module.exports = { evalExpr, checkTol, caseUrl, runSphereCase, runSpinCase, quatAngle };
