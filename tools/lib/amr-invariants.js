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
          // DEFAULT TRUE SINCE B2-2d. Corner 2:1 balance was reported and not
          // gated from B0 until then, for a good reason: the default ring path
          // tolerates a missing diagonal parent (interp fills the corner ghost
          // from the parent when the corner tile is absent), and the shipped
          // per-pass cascade covered only the four FACES -- so the count was
          // reliably nonzero, 13 on index-amr.html, and gating it would have
          // failed every run for a defect that was already written down.
          //
          // The closure covers all nine offsets, so it is now zero at every
          // checkpoint. A check that had to be non-gating because the code
          // could not satisfy it becomes a gate the moment the code can --
          // and leaving it reporting-only after that is how a fixed invariant
          // silently regresses.
          requireCornerBalance = true } = opts;

  const has = async (fn) => {
    const r = await evalExpr(Runtime, `typeof ${G}.${fn} === 'function'`);
    return !r.exceptionDetails && r.result.value === true;
  };
  if (!(await has('debugCheck21Balance'))) {
    throw new Error(`${G}.debugCheck21Balance is not available -- wrong page, or it failed to initialize`);
  }
  const hasCoverage = await has('debugCheckGeometryCoverage');
  // GATED SINCE B2-2d. This runs amr2d.mjs's `cascade21` -- the 2:1 rule as one
  // closure -- on the live PRESENT set and counts what the rule says must
  // exist and does not. It was reported-only for exactly as long as the code
  // could not satisfy it (the per-pass manager left 13 standing on
  // index-amr.html); the closure satisfies it by construction, so it is zero at
  // every checkpoint and there is no reason left not to gate.
  const hasClosure = await has('debugCheckRefinementClosure');
  // A CHECK THAT EXISTS AND NEVER RUNS IS WORSE THAN ONE THAT REPORTS.
  // amr2d.mjs's quadrantOfSlot says a slot's quadrant is `slot % 4`, and
  // B2-2b0 removed a whole binding on the strength of it -- but the scorer it
  // added was only ever driven by a one-off probe, so nothing in the standing
  // suite would notice if a future writer broke the rule (debugSnapshotLoad
  // writes whatever a snapshot recorded). Cheap, exact, and gated.
  const hasQuadrants = await has('debugCheckSlotQuadrants');
  // THE OTHER PER-SLOT BUFFER WHOSE CONTENT IS A FUNCTION OF CHEAPER DATA.
  // A tile's cached origin is `block * RB * 2^-(m-1)`; the pool manager builds
  // it by a parent-chain recursion instead, and that recursion is what got
  // transposed once and cost a wrong level-2 force. amr2d.mjs holds both routes
  // and tools/test-amr2d.js scores them against each other, but only as HOST
  // twins -- this is the same comparison against the buffer the kernels
  // actually read. Gated from the start, for the reason hasQuadrants records.
  const hasOrigins = await has('debugCheckTileOrigins');
  // POOL STARVATION -- refines refused for want of a slot. This used to be
  // "refinement convergence", asking whether the fixed-point loop had stopped
  // creating tiles when its iteration budget ran out; B2 deleted the loop and
  // B2-2d found that what remained of `converged` was an always-true clause
  // AND'd with this one, so the name went and the real check stayed.
  //
  // Still only meaningful with ?diag=1: at DIAG=0 every counter stays 0 and
  // this would read TRUE VACUOUSLY, which is the failure mode that let a dead
  // ?ghostfree path sit behind green checks. hasDiag gates the assertion, and
  // a skipped check is REPORTED as skipped, never as OK.
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
  const starvationViolations = [];
  let diagEnabled = false;
  const closureViolations = [];
  const quadrantViolations = [];
  const originViolations = [];
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
      if (diag.diagEnabled && !diag.poolOk) {
        starvationViolations.push({ step: stepsDone, granted: diag.refineGranted,
          starved: diag.refineStarved });
      }
    }

    let cov = null;
    if (hasCoverage) {
      const c = await evalExpr(Runtime, `${G}.debugCheckGeometryCoverage()`, 30000);
      if (c.exceptionDetails) throw new Error(`debugCheckGeometryCoverage failed at step ${stepsDone}: ${c.exceptionDetails.text}`);
      cov = c.result.value;
      if (!cov.ok) coverageViolations.push({ step: stepsDone, violations: cov.violations });
    }

    let quad = null;
    if (hasQuadrants) {
      const r = await evalExpr(Runtime, `${G}.debugCheckSlotQuadrants()`, 30000);
      if (r.exceptionDetails) throw new Error(`debugCheckSlotQuadrants failed at step ${stepsDone}: ${r.exceptionDetails.text}`);
      quad = r.result.value;
      if (!quad.ok) quadrantViolations.push({ step: stepsDone, violations: quad.violations });
    }

    let origins = null;
    if (hasOrigins) {
      const r = await evalExpr(Runtime, `${G}.debugCheckTileOrigins()`, 30000);
      if (r.exceptionDetails) throw new Error(`debugCheckTileOrigins failed at step ${stepsDone}: ${r.exceptionDetails.text}`);
      origins = r.result.value;
      if (!origins.ok) originViolations.push({ step: stepsDone, violations: origins.violations });
    }

    let closure = null;
    if (hasClosure) {
      const r = await evalExpr(Runtime, `${G}.debugCheckRefinementClosure()`, 30000);
      if (r.exceptionDetails) throw new Error(`debugCheckRefinementClosure failed at step ${stepsDone}: ${r.exceptionDetails.text}`);
      closure = r.result.value;
      if (!closure.ok) closureViolations.push({ step: stepsDone, missing: closure.missing, byReason: closure.byReason });
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
    if (onCheckpoint) onCheckpoint(stepsDone, { diag, bal: bal.result.value, cov, closure, quad, origins, bad: hasCardState ? bad : null });

    if (bad.length) break;
  }

  const cornerFails = requireCornerBalance ? cornerViolations.length > 0 : false;
  // Convergence is a real invariant of the default path (measured
  // converged=true at every checkpoint through 4096 steps), so it is asserted
  // unconditionally -- but only when ?diag=1 actually made the counters live.
  const ok = balanceViolations.length === 0 && coverageViolations.length === 0
    && fieldViolations.length === 0 && !cornerFails && starvationViolations.length === 0
    && closureViolations.length === 0 && quadrantViolations.length === 0
    && originViolations.length === 0;
  return { ok, stepsDone, balanceViolations, cornerViolations, requireCornerBalance,
    starvationViolations, starvationChecked: hasDiag && diagEnabled,
    coverageViolations,
    closureViolations,
    quadrantViolations, originViolations, fieldViolations, hasCoverage, hasCardState };
}

module.exports = { evalExpr, checkFinite, runInvariantSweep };
