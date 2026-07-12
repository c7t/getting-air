// Shared Taylor-Green-vortex (TGV) analysis, factored out so
// tools/validate-all.js can run the identical field-vs-analytic and
// decay-rate comparison against multiple pages (dense, AMR at various
// levels) without a second, independently-drifting copy -- same pattern as
// tools/lib/channel-metrics.js's own runCase, but against a full 2D
// space-time field (and a fitted decay rate) instead of a 1D steady-state
// profile, since TGV never reaches steady state.
//
// Unlike channel flow's live-settable `re`, every TGV parameter (N, u0,
// tau, and for AMR, levels) is page-load-time on BOTH main-tgv.js and
// main-tgv-amr.js -- there's no window.__CYL.setRe equivalent -- so a case
// here is always driven against a freshly-loaded page (the caller's job;
// see tools/validate-tgv.js / tools/validate-all.js's runTgvPhysics, one
// navigation per case, not grouped).
//
// AMR field comparison deliberately reads window.__CYL.readField() at the
// harness's own base resolution rather than reconstructing a genuinely
// finer grid via tools/lib/field-reconstruct.js: main-tgv-amr.js's
// readField() always returns L0-resolution data (the average/restriction
// pass keeps velBuf holding the finest-available value AT L0 resolution
// even when refined -- see that file's own comment on readField), so dense
// and AMR TGV cases are always compared at the SAME resolution, matching
// main-channel-amr.js's identical choice (chanResFilter capped to the
// dense case's own resolution) rather than validate-amr-vs-dense.js's
// harder different-resolution problem.

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

function checkTol(label, measured, target, tol) {
  if (target == null) return { pass: true, label, measured, target: null };
  const pass = measured != null && Math.abs(measured - target) <= tol;
  return { pass, label, measured, target, tol };
}

// Exact analytic 2D decaying-TGV velocity field at time t (lattice steps,
// dt=1) -- see main-tgv.js's own header for the derivation. kx=ky=2*PI/N
// (one wavelength exactly fills the periodic domain), td=1/(nu*(kx^2+ky^2))
// is the analytic decay time. Density/pressure is intentionally NOT
// compared here -- it's an O(Ma^2) compressibility correction the sim
// reproduces but that adds no diagnostic value over the velocity field
// itself, which is what both the field-error and decay-rate checks
// actually care about (see main-tgv.js's own analyticField for the full
// rho term, used only there for the initial-condition seed).
function analyticVelocity(N, u0, nu, t) {
  const kx = 2 * Math.PI / N, ky = kx;
  const td = 1 / (nu * (kx * kx + ky * ky));
  const decay = Math.exp(-t / td);
  const ux = new Float64Array(N * N), uy = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const c = y * N + x;
      const cx = Math.cos(kx * x), sx = Math.sin(kx * x);
      const cy = Math.cos(ky * y), sy = Math.sin(ky * y);
      ux[c] = -u0 * cx * sy * decay;
      uy[c] = u0 * (kx / ky) * sx * cy * decay;
    }
  }
  return { ux, uy, td };
}

// Joint relative L2 error across both velocity components.
function compareField(sim, analytic) {
  let sumSq = 0, sumA = 0;
  for (let i = 0; i < sim.ux.length; i++) {
    const dux = sim.ux[i] - analytic.ux[i];
    const duy = sim.uy[i] - analytic.uy[i];
    sumSq += dux * dux + duy * duy;
    sumA += analytic.ux[i] * analytic.ux[i] + analytic.uy[i] * analytic.uy[i];
  }
  return Math.sqrt(sumSq / Math.max(sumA, 1e-30));
}

function rmsNorm(ux, uy) {
  let s = 0;
  for (let i = 0; i < ux.length; i++) s += ux[i] * ux[i] + uy[i] * uy[i];
  return Math.sqrt(s / ux.length);
}

// Drives ONE (N,u0,tau[,levels]) case against whatever page `Runtime` is
// currently attached to -- assumes window.__CYL.{reset,getParams,
// debugStepSync,readField} (main-tgv.js's and main-tgv-amr.js's shared
// shape) and that the page was already loaded with the matching N/u0/tau.
//
// caseEntry.checkpoints are multiples of the analytic decay time td (not
// raw step counts), so the same case definition samples equivalent points
// in the decay regardless of N/u0/tau. main-tgv-amr.js's debugStepSync
// rounds UP to its own STEPS_PER_FRAME (64) internally, while main-tgv.js's
// runs the exact requested count -- this always re-derives the analytic
// comparison time from debugStepSync's OWN returned `step`, not the
// requested delta, so both shapes are handled correctly without the caller
// needing to know which one it's driving (a bug hit and fixed live during
// Phase 5's own manual verification of main-tgv-amr.js).
async function runCase(Runtime, opts, caseEntry, log) {
  await evalExpr(Runtime, `window.__CYL.reset()`);
  const params = (await evalExpr(Runtime, `window.__CYL.getParams()`)).result.value;

  if (log) log(`N=${params.N} u0=${params.U0} tau=${params.TAU}: nu=${params.nu.toFixed(4)} td=${params.td.toFixed(1)} steps`);

  const checkpointSteps = caseEntry.checkpoints.map(m => Math.round(m * params.td));
  let prevStep = 0;
  const samples = [];
  for (const target of checkpointSteps) {
    const dn = target - prevStep;
    const r = await evalExpr(Runtime, `window.__CYL.debugStepSync(${dn})`, ((opts.timeout || 120) + 10) * 1000);
    if (r.exceptionDetails) throw new Error(`debugStepSync failed for N=${params.N} u0=${params.U0} tau=${params.TAU}: ${r.exceptionDetails.text}`);
    const actualStep = r.result.value.step;
    prevStep = actualStep;

    const field = (await evalExpr(Runtime, `window.__CYL.readField()`, 30000)).result.value;
    const analytic = analyticVelocity(params.N, params.U0, params.nu, actualStep);
    const l2rel = compareField(field, analytic);
    const rms = rmsNorm(field.ux, field.uy);
    samples.push({ step: actualStep, l2rel, rms });
    if (log) log(`  step ${actualStep}: L2rel=${l2rel.toExponential(3)}`);
  }

  const maxL2rel = Math.max(...samples.map(s => s.l2rel));
  const fieldCheck = checkTol('fieldL2rel', maxL2rel, 0, caseEntry.field_l2_tol);

  const [a, b] = samples.slice(-2);
  const rateMeasured = -Math.log(b.rms / a.rms) / (b.step - a.step);
  const rateAnalytic = 1 / params.td;
  const rateRelErr = Math.abs(rateMeasured - rateAnalytic) / rateAnalytic;
  const rateCheck = checkTol('decayRateRelErr', rateRelErr, 0, caseEntry.decay_rate_tol);

  return {
    N: params.N, u0: params.U0, tau: params.TAU,
    samples, maxL2rel, rateMeasured, rateAnalytic, rateRelErr,
    fieldCheck, rateCheck,
  };
}

module.exports = { evalExpr, checkTol, analyticVelocity, compareField, rmsNorm, runCase };
