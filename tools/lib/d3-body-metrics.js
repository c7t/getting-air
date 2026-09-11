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
  // AMR knobs, present only on the M4.1d sphere-with-AMR cases. The body
  // scenarios drive the same page as everything else, so these are the same
  // parameters an interactive run would pass.
  if (c.levels) p.set('levels', c.levels);
  if (c.rb) p.set('rb', c.rb);
  if (c.refine) p.set('refine', c.refine);
  if (c.interface) p.set('interface', c.interface);
  if (c.dynamic) p.set('dynamic', c.dynamic);
  // M8.1. Symmetry breaking for the shedding cases; absent (and therefore 0)
  // everywhere else, so the steady cases stay bit-identical.
  if (c.perturb) p.set('perturb', c.perturb);
  if (c.seed) p.set('seed', c.seed);
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

// --- sphere, SHEDDING (plans/3D.md M8.1) ------------------------------------
//
// WHY THIS IS A SEPARATE RUNNER. runSphereCase asserts that Cd has STOPPED
// MOVING and that the cross-flow force is under 2% of it. Both are correct
// for Re <= 210 -- Johnson & Patel (JFM 378, 1999) place the steady
// axisymmetric regime there -- and both are violated BY CONSTRUCTION once
// the wake sheds. So no existing gate can express "the wake is unsteady and
// its frequency is right", which is the one claim standing between this
// solver and the falling-card target.
//
// THE STROUHAL NUMBER IS THE GATE; Cd IS REPORTED. A frequency is a property
// of the WAKE, and a staircased bounce-back surface at D = 16 perturbs it
// far less than it perturbs the surface integral -- the existing cases
// already measure Cd +12.8% against Schiller-Naumann at Re=20 while the flow
// itself is right. Gating on the quantity the discretization does not
// dominate is the difference between a physics check and a resolution check.
//
// THE WAKE PLANE IS NOT KNOWN IN ADVANCE, and that is the one genuinely 3D
// part. A 2D cylinder sheds in the only plane it has; a sphere's wake picks
// an azimuthal orientation spontaneously (and can drift), so neither Cl nor
// Cs alone is the signal, and their MAGNITUDE is worse than either -- it is
// a rectified sine with twice the frequency and no zero crossings at all.
// The projection below finds the plane from the data (the principal axis of
// the lateral-force covariance) and hands the resulting SCALAR series to
// cylinder-metrics' own estimator, which is tested and wants no second copy.
function projectLateral(history, transientSteps) {
  const win = history.filter(r => r[0] > transientSteps);
  if (win.length < 4) return { rows: [], theta: NaN, anisotropy: NaN };
  const cl = win.map(r => r[4]), cs = win.map(r => r[5]);
  const mL = cl.reduce((a, b) => a + b, 0) / cl.length;
  const mS = cs.reduce((a, b) => a + b, 0) / cs.length;
  let sll = 0, sss = 0, sls = 0;
  for (let i = 0; i < win.length; i++) {
    const a = cl[i] - mL, b = cs[i] - mS;
    sll += a * a; sss += b * b; sls += a * b;
  }
  sll /= win.length; sss /= win.length; sls /= win.length;
  // Principal axis of a symmetric 2x2 covariance, closed form.
  const theta = 0.5 * Math.atan2(2 * sls, sll - sss);
  const ct = Math.cos(theta), st = Math.sin(theta);
  // Eigenvalues, for the anisotropy report below.
  const tr = sll + sss, det = sll * sss - sls * sls;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const e1 = tr / 2 + disc, e2 = tr / 2 - disc;
  // Rebuild rows with the PROJECTED lateral in column 4, so
  // cylinder-metrics' analyze() reads it unchanged.
  const rows = win.map((r, i) => [r[0], r[1], r[2], r[3], (cl[i] - mL) * ct + (cs[i] - mS) * st]);
  // ANISOTROPY IS A CHECK, not decoration: a genuinely planar wake has one
  // dominant axis, so e1/e2 >> 1. A value near 1 means the lateral force is
  // isotropic noise and the "shedding plane" is an artefact of fitting a
  // line to a circle -- in which case the Strouhal number below is fitting
  // noise too, and the tool says so rather than reporting a number.
  return { rows, theta, anisotropy: e2 > 0 ? e1 / e2 : Infinity, e1, e2 };
}

async function runSphereShedCase(Runtime, opts, c, log) {
  const { analyze } = require('./cylinder-metrics');
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  const conv = p.convective;                       // D/U in steps
  const transient = Math.round((c.transient_convective || 60) * conv);
  // Long enough for the requested number of shedding periods at the
  // EXPECTED Strouhal, so a case at a different Re gets a proportionate
  // window rather than a fixed step count.
  const period = conv / c.st;                      // steps per shedding cycle
  const measure = Math.round((c.periods || 12) * period);
  // ~24 samples per period: enough that the zero crossing is interpolated
  // between close neighbours, cheap enough that the readback is noise.
  const every = Math.max(1, Math.round(period / 24));
  if (log) {
    log(`D=${p.D} Re=${p.re} ${p.bounceback ? 'bounceback' : 'diffuse'} Q${p.Q}: tau=${p.tau.toFixed(4)} nu=${p.nu.toFixed(5)}`);
    log(`  D/U=${conv.toFixed(0)} steps, expected period ${period.toFixed(0)} steps`);
    log(`  transient ${transient} then ${measure} steps (${c.periods || 12} periods), sampling every ${every}`);
  }

  await evalOrThrow(Runtime, `window.__D3.debugStepSync(${transient})`, ((opts.timeout || 600) + 120) * 1000, 'debugStepSync');
  const r = await evalOrThrow(Runtime,
    `window.__D3.debugRunAndCollect(${measure}, ${every})`, ((opts.timeout || 600) + 600) * 1000, 'debugRunAndCollect');

  const proj = projectLateral(r.history, 0);       // the transient is already behind us
  const a = analyze(proj.rows, 0, p.D, p.u0);
  const st = a.st;
  const stErr = st == null ? NaN : (st - c.st) / c.st;
  const cdErr = (a.cdMean - c.cd) / c.cd;
  // Peak-to-peak of the projected lateral, as a fraction of Cd: the
  // amplitude of the thing whose frequency is being measured. A steady wake
  // gives ~0 here, which is how a case that did NOT shed is told from one
  // that sheds at the wrong frequency.
  const amp = proj.rows.length
    ? (Math.max(...proj.rows.map(x => x[4])) - Math.min(...proj.rows.map(x => x[4]))) / Math.abs(a.cdMean)
    : 0;
  if (log) {
    log(`  St=${st == null ? 'NONE' : st.toFixed(4)} vs ${c.st} (${st == null ? '-' : (stErr * 100).toFixed(1) + '%'})`
      + `  Cd=${a.cdMean.toFixed(4)} vs ${c.cd} (${(cdErr * 100).toFixed(1)}%)`);
    log(`  ${a.crossings} crossings, ${a.samples} samples, lateral pk-pk/Cd=${amp.toExponential(2)}`
      + `, wake plane ${(proj.theta * 180 / Math.PI).toFixed(0)} deg, anisotropy ${proj.anisotropy.toFixed(1)}`);
  }

  return {
    D: p.D, re: p.re, tau: p.tau, Q: p.Q, bounceback: !!p.bounceback,
    steps: transient + measure, st, cd: a.cdMean, stErr, cdErr, amp,
    crossings: a.crossings, samples: a.samples, theta: proj.theta, anisotropy: proj.anisotropy,
    // THE GATE. St against the DNS benchmark; Cd only reported unless the
    // case asks for it, because at these resolutions Cd is a statement about
    // the staircased surface and St is a statement about the flow.
    stCheck: checkTol('StRelErr', stErr, 0, c.st_tol_rel),
    // IT MUST ACTUALLY SHED. A steady run has no crossings and no amplitude,
    // and would otherwise report St = null and pass nothing -- which reads
    // the same as a missing measurement. This makes "it did not shed" a
    // FAILURE with its own name.
    shedCheck: { pass: amp >= (c.amp_min ?? 0.02) && a.crossings >= 5,
      label: 'shedAmplitude', measured: amp, target: c.amp_min ?? 0.02, tol: null },
    // And the wake must be planar enough for a plane to mean something.
    planarCheck: { pass: proj.anisotropy >= (c.anisotropy_min ?? 3),
      label: 'wakeAnisotropy', measured: proj.anisotropy, target: c.anisotropy_min ?? 3, tol: null },
    ...(c.cd_tol_rel ? { cdCheck: checkTol('CdRelErr', cdErr, 0, c.cd_tol_rel) } : {}),
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

module.exports = { evalExpr, checkTol, caseUrl, runSphereCase, runSphereShedCase, runSpinCase, quatAngle, projectLateral };
