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
// window.__CYL.debugCheck21Balance()/debugCheckGeometryCoverage() and a
// field-finite smoke check (debugReadCardState) after every batch. Stops
// early on a field blowup (unrecoverable -- no point spending the rest of
// the budget stepping a NaN'd sim). `onCheckpoint(stepsDone, {bal, cov,
// bad})`, if given, is called after every batch for caller-side logging.
async function runInvariantSweep(Runtime, opts) {
  const { steps, checkEvery, timeout = 300, onCheckpoint } = opts;

  await evalExpr(Runtime, `window.__CYL.setLive(false)`);
  // Deterministic, reproducible baseline -- debugStepSync's own return value
  // is the ABSOLUTE step count, not progress made this call, and this may
  // attach to a page that's already been stepped; resetting first makes
  // "run N steps" mean the same thing every invocation.
  await evalExpr(Runtime, `window.__CYL.reset()`);

  const balanceViolations = [];
  const coverageViolations = [];
  const fieldViolations = [];
  let stepsDone = 0;

  while (stepsDone < steps) {
    const batch = Math.min(checkEvery, steps - stepsDone);
    const r = await evalExpr(Runtime, `window.__CYL.debugStepSync(${batch})`, (timeout + 30) * 1000);
    if (r.exceptionDetails) throw new Error(`debugStepSync failed at step ${stepsDone}: ${r.exceptionDetails.text}`);
    stepsDone = r.result.value.step;

    const bal = await evalExpr(Runtime, `window.__CYL.debugCheck21Balance()`, 30000);
    if (bal.exceptionDetails) throw new Error(`debugCheck21Balance failed at step ${stepsDone}: ${bal.exceptionDetails.text}`);
    if (!bal.result.value.ok) balanceViolations.push({ step: stepsDone, violations: bal.result.value.violations });

    const cov = await evalExpr(Runtime, `window.__CYL.debugCheckGeometryCoverage()`, 30000);
    if (cov.exceptionDetails) throw new Error(`debugCheckGeometryCoverage failed at step ${stepsDone}: ${cov.exceptionDetails.text}`);
    if (!cov.result.value.ok) coverageViolations.push({ step: stepsDone, violations: cov.result.value.violations });

    const state = await evalExpr(Runtime, `window.__CYL.debugReadCardState()`, 30000);
    if (state.exceptionDetails) throw new Error(`debugReadCardState failed at step ${stepsDone}: ${state.exceptionDetails.text}`);
    const bad = checkFinite(state.result.value);
    if (bad.length) fieldViolations.push({ step: stepsDone, fields: bad, state: state.result.value });

    if (onCheckpoint) onCheckpoint(stepsDone, { bal: bal.result.value, cov: cov.result.value, bad });

    if (bad.length) break;
  }

  const ok = balanceViolations.length === 0 && coverageViolations.length === 0 && fieldViolations.length === 0;
  return { ok, stepsDone, balanceViolations, coverageViolations, fieldViolations };
}

module.exports = { evalExpr, checkFinite, runInvariantSweep };
