// Shared Cd/St physics-analysis logic, factored out of validate-cylinder.js
// so tools/validate-all.js can run the exact same case-by-case analysis
// against multiple pages/configs (dense reference, AMR at various levels)
// without a second, independently-drifting copy of the transient/
// measurement-window sizing or the zero-crossing Strouhal estimator.

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

// Rounds a step count up to a multiple of STEPS_PER_FRAME (64, hardcoded in
// main-cylinder.js/main-cylinder-amr.js) so debugRunAndCollect's per-block
// sampling lands cleanly.
function roundSteps(n) {
  return Math.ceil(n / 64) * 64;
}

// Transient + measurement window sized from the physical parameters, not a
// fixed step count -- a slower freestream or larger cylinder needs more
// steps to reach the same number of flow-through times / shedding cycles.
function computeWindow(caseEntry, D, U0) {
  const transient = roundSteps(40 * D / U0);
  const measurement = caseEntry.st
    ? roundSteps(15 * (D / (caseEntry.st * U0))) // ~15 shedding periods
    : roundSteps(5 * D / U0);                     // steady case: just confirm convergence
  return { transient, measurement, total: transient + measurement };
}

// Cd_mean over the post-transient window, plus a Strouhal estimate from
// Cl(t)'s zero crossings (mean-subtracted, linearly interpolated crossing
// time for sub-sample precision). Returns st=null if fewer than 3 crossings
// are found (steady flow, or window too short).
function analyze(history, transientSteps, D, U0) {
  const win = history.filter(r => r[0] > transientSteps);
  if (win.length < 2) return { cdMean: NaN, st: null, samples: win.length };

  const cdMean = win.reduce((s, r) => s + r[3], 0) / win.length;

  const cl = win.map(r => r[4]);
  const clMean = cl.reduce((a, b) => a + b, 0) / cl.length;
  const centered = cl.map(v => v - clMean);

  const crossings = [];
  for (let i = 1; i < centered.length; i++) {
    if (centered[i - 1] < 0 && centered[i] >= 0) {
      const stepA = win[i - 1][0], stepB = win[i][0];
      const frac = -centered[i - 1] / (centered[i] - centered[i - 1]);
      crossings.push(stepA + frac * (stepB - stepA));
    }
  }
  let st = null;
  if (crossings.length >= 3) {
    const periods = [];
    for (let i = 1; i < crossings.length; i++) periods.push(crossings[i] - crossings[i - 1]);
    const meanPeriod = periods.reduce((a, b) => a + b, 0) / periods.length;
    const fs = 1 / meanPeriod;
    st = fs * D / U0;
  }
  return { cdMean, st, samples: win.length, crossings: crossings.length };
}

function checkTol(label, measured, target, tol) {
  if (target == null) return { pass: true, label, measured, target: null };
  const pass = measured != null && Math.abs(measured - target) <= tol;
  return { pass, label, measured, target, tol };
}

// Drives ONE Re case against whatever page `Runtime` is currently attached
// to -- assumes window.__CYL.{setRe,reset,getParams,debugRunAndCollect} (the
// shape both main-cylinder.js and main-cylinder-amr.js expose identically,
// see main-cylinder-amr.js's own file header). `log`, if given, is called
// with a one-line progress string before the run starts.
async function runCase(Runtime, opts, caseEntry, log) {
  await evalExpr(Runtime, `window.__CYL.setRe(${caseEntry.re})`);
  await evalExpr(Runtime, `window.__CYL.reset()`);
  const params = (await evalExpr(Runtime, `window.__CYL.getParams()`)).result.value;
  const { transient, measurement, total } = computeWindow(caseEntry, params.D, params.U0);

  if (log) log(`Re=${caseEntry.re}: D=${params.D} U0=${params.U0} TAU=${params.TAU.toFixed(5)} -- running ${total} steps (${transient} transient + ${measurement} measurement)`);

  const r = await evalExpr(Runtime, `window.__CYL.debugRunAndCollect(${total})`, ((opts.timeout || 300) + 30) * 1000);
  if (r.exceptionDetails) throw new Error(`debugRunAndCollect failed for Re=${caseEntry.re}: ${r.exceptionDetails.text}`);
  const history = r.result.value.history;

  const { cdMean, st, samples, crossings } = analyze(history, transient, params.D, params.U0);
  const cd = checkTol('Cd', cdMean, caseEntry.cd, caseEntry.cd_tol);
  const stChk = checkTol('St', st, caseEntry.st, caseEntry.st_tol);

  return { re: caseEntry.re, regime: caseEntry.regime, samples, crossings, cd, st: stChk, source: caseEntry.source };
}

module.exports = { evalExpr, roundSteps, computeWindow, analyze, checkTol, runCase };
