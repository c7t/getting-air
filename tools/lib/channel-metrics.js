// Shared Poiseuille/Couette channel-flow analysis, factored out so
// tools/validate-all.js can run the identical steady-state-vs-analytic
// comparison against multiple pages/configs (dense, AMR at various levels)
// without a second, independently-drifting copy -- same pattern as
// tools/lib/cylinder-metrics.js's own runCase, adapted for an exact
// analytic target instead of a literature Cd/St band.
//
// Unlike the cylinder harness's `res` (a URL-only, page-load-time
// parameter that a single validate-all.js config sweeps `re` against),
// channel flow's resolution genuinely IS part of what's being validated
// (convergence order), so a single benchmarks/channel.json case varies
// BOTH `res` and `re`. `runCase` here still assumes the correct page is
// ALREADY loaded (mirrors cylinder-metrics.js's contract exactly) --
// grouping cases by (mode,res) and navigating between groups is the
// caller's job (see tools/validate-channel.js / tools/validate-all.js's
// runChannelPhysics), not this module's.

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

function checkTol(label, measured, target, tol) {
  if (target == null) return { pass: true, label, measured, target: null };
  const pass = measured != null && Math.abs(measured - target) <= tol;
  return { pass, label, measured, target, tol };
}

// Exact analytic steady-state profile -- see main-channel.js's own header
// for the derivation (halfway-bounce-back walls, wall-to-wall height
// exactly H, cell centers at y'=y+0.5, not y).
function analyticProfile(mode, H, nu, FORCE_X, WALL_U1) {
  const u = new Array(H);
  for (let y = 0; y < H; y++) {
    const yp = y + 0.5;
    u[y] = mode === 'poiseuille'
      ? (FORCE_X / (2 * nu)) * yp * (H - yp)
      : WALL_U1 * yp / H;
  }
  return u;
}

function compareProfiles(sim, analytic) {
  let maxErr = 0, sumSq = 0, sumAnalyticSq = 0;
  for (let y = 0; y < sim.length; y++) {
    const err = Math.abs(sim[y] - analytic[y]);
    maxErr = Math.max(maxErr, err);
    sumSq += err * err;
    sumAnalyticSq += analytic[y] * analytic[y];
  }
  const l2rel = Math.sqrt(sumSq / Math.max(sumAnalyticSq, 1e-30));
  return { l2rel, maxErr };
}

// Drives ONE (mode,res,re) case against whatever page `Runtime` is
// currently attached to -- assumes window.__CYL.{setRe,reset,getParams,
// debugRunToSteady} (main-channel.js's and main-channel-amr.js's shared
// shape) and that the page was already loaded with the matching
// mode/res (setRe only adjusts Re live; mode/res are page-load-time).
async function runCase(Runtime, opts, caseEntry, log) {
  await evalExpr(Runtime, `window.__CYL.setRe(${caseEntry.re})`);
  await evalExpr(Runtime, `window.__CYL.reset()`);
  const params = (await evalExpr(Runtime, `window.__CYL.getParams()`)).result.value;

  if (log) log(`${caseEntry.mode} res=${caseEntry.res} Re=${caseEntry.re}: H=${params.H} nu=${params.nu.toFixed(4)} -- running to steady state`);

  const runOpts = { blockSteps: opts.blockSteps || 1024, maxSteps: opts.maxSteps || 200000, tol: opts.convTol || 1e-6 };
  const r = await evalExpr(
    Runtime,
    `window.__CYL.debugRunToSteady(${JSON.stringify(runOpts)})`,
    ((opts.timeout || 300) + 30) * 1000
  );
  if (r.exceptionDetails) throw new Error(`debugRunToSteady failed for ${caseEntry.mode} res=${caseEntry.res} Re=${caseEntry.re}: ${r.exceptionDetails.text}`);
  const { profile, converged, step } = r.result.value;

  const analytic = analyticProfile(params.mode, params.H, params.nu, params.FORCE_X, params.WALL_U1);
  const { l2rel, maxErr } = compareProfiles(profile, analytic);
  const l2 = checkTol('L2rel', l2rel, 0, caseEntry.l2_tol);

  return { mode: caseEntry.mode, res: caseEntry.res, re: caseEntry.re, converged, step, l2rel, maxErr, l2 };
}

module.exports = { evalExpr, checkTol, analyticProfile, compareProfiles, runCase };
