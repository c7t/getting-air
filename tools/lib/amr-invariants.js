// Shared AMR structural-invariant sweep, factored out of
// validate-amr-invariants.js so tools/validate-all.js can run the identical
// checkpointed sweep against multiple pages/configs without a second,
// independently-drifting copy of the loop.

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

function checkFinite(state) {
  const bad = [];
  for (const k of ['fx', 'fy', 'tz', 'vx', 'vy', 'omega']) {
    const v = state[k];
    if (!Number.isFinite(v)) bad.push(k);
  }
  return bad;
}

// Resets the page (deterministic baseline -- see caller-facing header), then
// steps in `checkEvery`-sized batches up to `steps` total, asserting
// <global>.debugCheck21Balance()/debugCheckGeometryCoverage() and a
// field-finite smoke check (debugReadCardState) after every batch. Stops
// early on a field blowup (unrecoverable -- no point spending the rest of
// the budget stepping a NaN'd sim). `onCheckpoint(stepsDone, {bal, cov,
// bad})`, if given, is called after every batch for caller-side logging.
//
// `opts.global` names the page's own debug surface, default `window.__CYL`
// (every scenario harness). The falling-card dev page exposes `window.__AMR`
// instead, and it is the page this project actually SHIPS -- it had never
// been swept at all while this was hardcoded to __CYL, which is exactly how a
// 2:1-balance report against index-amr.html could sit outside every test.
//
// Only debugCheck21Balance is required. debugCheckGeometryCoverage and
// debugReadCardState are probed for and SKIPPED when the page doesn't define
// them (window.__AMR has no coverage scan and no card-state readback), rather
// than throwing -- a missing optional check is reported as skipped, never as
// a pass, so this can't quietly look greener than it is.
async function runInvariantSweep(Runtime, opts) {
  const { steps, checkEvery, timeout = 300, onCheckpoint, global: G = 'window.__CYL',
          requireCornerBalance = false } = opts;

  const has = async (fn) => {
    const r = await evalExpr(Runtime, `typeof ${G}.${fn} === 'function'`);
    return !r.exceptionDetails && r.result.value === true;
  };
  if (!(await has('debugCheck21Balance'))) {
    throw new Error(`${G}.debugCheck21Balance is not available -- wrong page, or it failed to initialize`);
  }
  const hasCoverage = await has('debugCheckGeometryCoverage');
  // Refinement convergence. Only meaningful with ?diag=1: at DIAG=0 every
  // counter stays 0 and `converged` reads TRUE VACUOUSLY, so asserting it
  // without the flag would be a silent false pass -- exactly the failure mode
  // that let a dead ?ghostfree path sit behind green checks. hasDiag gates the
  // assertion, and a skipped check is REPORTED as skipped, never as OK.
  const hasDiag = await has('debugReadDiag');
  const hasCardState = await has('debugReadCardState');

  await evalExpr(Runtime, `${G}.setLive(false)`);
  // Deterministic, reproducible baseline -- debugStepSync's own return value
  // is the ABSOLUTE step count, not progress made this call, and this may
  // attach to a page that's already been stepped; resetting first makes
  // "run N steps" mean the same thing every invocation.
  await evalExpr(Runtime, `${G}.reset()`);

  const balanceViolations = [];
  // Corner (diagonal) 2:1 balance, asserted only when the caller says the run
  // requires it. It is a REQUIREMENT OF ?ghostfree=1, whose bilinear parent
  // stencil reads the parent's corner cell directly, and NOT of the default
  // ring path, where interp fills a corner ghost from the parent when the
  // corner tile is absent. Always COLLECTED, so a default run still reports
  // how much corner imbalance it is carrying -- which is the number that says
  // what corner balance would cost if it were ever made unconditional.
  const cornerViolations = [];
  // A refine round whose 2:1-BALANCE CASCADE was still propagating when the
  // fixed iteration count ran out. Not "still creating tiles" -- a
  // criterion-driven grant in the final iteration is normal operation. See
  // debugReadDiag()'s own comment on why gating on `granted` was wrong.
  const convergenceViolations = [];
  let diagEnabled = false;
  const coverageViolations = [];
  const fieldViolations = [];
  let stepsDone = 0;

  while (stepsDone < steps) {
    const batch = Math.min(checkEvery, steps - stepsDone);
    const r = await evalExpr(Runtime, `${G}.debugStepSync(${batch})`, (timeout + 30) * 1000);
    if (r.exceptionDetails) throw new Error(`debugStepSync failed at step ${stepsDone}: ${r.exceptionDetails.text}`);
    stepsDone = r.result.value.step;

    const bal = await evalExpr(Runtime, `${G}.debugCheck21Balance()`, 30000);
    if (bal.exceptionDetails) throw new Error(`debugCheck21Balance failed at step ${stepsDone}: ${bal.exceptionDetails.text}`);
    if (!bal.result.value.ok) balanceViolations.push({ step: stepsDone, violations: bal.result.value.violations });
    if (bal.result.value.cornerOk === false) {
      cornerViolations.push({ step: stepsDone, count: bal.result.value.cornerViolations.length, violations: bal.result.value.cornerViolations.slice(0, 8) });
    }

    let diag = null;
    if (hasDiag) {
      const d = await evalExpr(Runtime, `${G}.debugReadDiag()`, 30000);
      if (d.exceptionDetails) throw new Error(`debugReadDiag failed at step ${stepsDone}: ${d.exceptionDetails.text}`);
      diag = d.result.value;
      diagEnabled = diagEnabled || !!diag.diagEnabled;
      if (diag.diagEnabled && !diag.converged) {
        convergenceViolations.push({ step: stepsDone, granted: diag.refineGrantedLastIter,
          byCascade: diag.refineByCascadeLastIter, poolExhausted: diag.refinePoolExhausted });
      }
    }

    let cov = null;
    if (hasCoverage) {
      const c = await evalExpr(Runtime, `${G}.debugCheckGeometryCoverage()`, 30000);
      if (c.exceptionDetails) throw new Error(`debugCheckGeometryCoverage failed at step ${stepsDone}: ${c.exceptionDetails.text}`);
      cov = c.result.value;
      if (!cov.ok) coverageViolations.push({ step: stepsDone, violations: cov.violations });
    }

    let bad = [];
    if (hasCardState) {
      const state = await evalExpr(Runtime, `${G}.debugReadCardState()`, 30000);
      if (state.exceptionDetails) throw new Error(`debugReadCardState failed at step ${stepsDone}: ${state.exceptionDetails.text}`);
      bad = checkFinite(state.result.value);
      if (bad.length) fieldViolations.push({ step: stepsDone, fields: bad, state: state.result.value });
    }

    // cov/bad are null (not empty) when this page doesn't expose that check,
    // so a caller renders "n/a" rather than the "OK" an empty result would
    // otherwise read as.
    if (onCheckpoint) onCheckpoint(stepsDone, { diag, bal: bal.result.value, cov, bad: hasCardState ? bad : null });

    if (bad.length) break;
  }

  const cornerFails = requireCornerBalance ? cornerViolations.length > 0 : false;
  // Convergence is a real invariant of the default path (measured
  // converged=true at every checkpoint through 4096 steps), so it is asserted
  // unconditionally -- but only when ?diag=1 actually made the counters live.
  const ok = balanceViolations.length === 0 && coverageViolations.length === 0
    && fieldViolations.length === 0 && !cornerFails && convergenceViolations.length === 0;
  return { ok, stepsDone, balanceViolations, cornerViolations, requireCornerBalance,
    convergenceViolations, convergenceChecked: hasDiag && diagEnabled,
    coverageViolations, fieldViolations, hasCoverage, hasCardState };
}

module.exports = { evalExpr, checkFinite, runInvariantSweep };
