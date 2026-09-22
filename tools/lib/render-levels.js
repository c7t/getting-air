// Does every configured refinement level have a path into the PICTURE?
//
// The check itself, shared by `tools/validate-render-levels.js` (which owns the
// browser lifecycle and prints a report) and `tools/validate-all.js` (which
// runs it as one config in the default sweep). Same discipline as
// `tools/lib/cylinder-metrics.js` and `tools/lib/amr-invariants.js`: the leaf
// tool and the sweep call ONE implementation, not two that drift.
//
// See validate-render-levels.js's header for what this measures and why it is
// a capability test rather than a static one.

// Far outside any physical velocity this solver produces: the question is "is
// there a path from this buffer to the picture", and a perturbation small
// enough to be swallowed by the colour map answers a different one.
const PERTURB = 9.0;

async function evalOrThrow(Runtime, expression, timeout = 60000) {
  const r = await Runtime.evaluate({ expression, returnByValue: true, awaitPromise: true, timeout });
  if (r.exceptionDetails) {
    throw new Error(`page threw evaluating \`${expression}\`: ${r.exceptionDetails.text} `
      + (r.exceptionDetails.exception?.description || ''));
  }
  return r.result.value;
}

// The CANVAS only, not the whole viewport: the status line carries a step
// counter and a clock, so a full-page shot would differ between two renders for
// reasons that have nothing to do with the field.
async function shotCanvas(Page, Runtime, crypto) {
  const box = await evalOrThrow(Runtime, `(() => {
    const c = document.querySelector('canvas');
    const r = c.getBoundingClientRect();
    return JSON.stringify({ x: r.x, y: r.y, width: r.width, height: r.height });
  })()`);
  const { x, y, width, height } = JSON.parse(box);
  const { data } = await Page.captureScreenshot({
    format: 'png', clip: { x, y, width, height, scale: 1 },
  });
  return { hash: crypto.createHash('sha256').update(data).digest('hex').slice(0, 16), bytes: data.length };
}

// Assumes the page is ALREADY loaded and its global is ready -- the caller owns
// navigation, because validate-all.js reuses one tab across every config.
//
// Returns { rows, active, baseline, aborted }. `aborted` is set when a
// screenshot cannot score equality on this setup at all, in which case `rows`
// is empty and nothing below it would have meant what it said.
async function runRenderLevels({ Page, Runtime, global, steps = 4096, log = () => {} }) {
  const crypto = require('crypto');
  const rows = [];

  const nLevels = await evalOrThrow(Runtime, `${global}.getNumLevels()`);
  const hasHooks = await evalOrThrow(Runtime,
    `typeof ${global}.debugRenderOnce === 'function' && typeof ${global}.debugPerturbLevelVel === 'function'`);
  if (!hasHooks) {
    return { rows, active: {}, baseline: null, nLevels,
      aborted: 'this page exposes no debugRenderOnce/debugPerturbLevelVel -- the gate cannot run against it' };
  }

  // Settle refinement so the deep levels actually hold tiles. reset() first:
  // the page's rAF loop runs between load and setLive(false), so reading
  // anything before a reset reads an unknown number of steps of drift.
  log(`[setup] reset + ${steps} steps to settle refinement`);
  await evalOrThrow(Runtime, `${global}.reset()`, 120000);
  await evalOrThrow(Runtime, `${global}.debugStepSync(${steps})`, 600000);
  await evalOrThrow(Runtime, `${global}.setLive(false)`);

  // FROM LEVEL 0, since U7-6b. The root is a pool level and the renderer draws
  // it through the same indirection as every other level, so it is scoreable
  // the same way -- and it never was before, which is why this gate could not
  // have caught an L0 render defect at all. The root pool is always FULL, so
  // its count is every block rather than a refinement result.
  const active = {};
  for (let m = 0; m < nLevels; m++) {
    active[m] = await evalOrThrow(Runtime,
      `(async () => { try { return (await ${global}.debugListActiveBlocks(${m})).length; } catch (e) { return -1; } })()`, 120000);
  }
  log(`[setup] active tiles by level: ${Object.entries(active).map(([m, n]) => `L${m}=${n}`).join(' ')}`);

  // Keep the snapshot IN THE PAGE. It is megabytes of typed array and there is
  // no reason to move it through CDP twice per level.
  await evalOrThrow(Runtime, `(async () => { window.__RLGATE = await ${global}.debugSnapshotSave(); return 1; })()`, 300000);

  // ── 1. is a screenshot even reproducible here? ─────────────────────────────
  await evalOrThrow(Runtime, `${global}.debugRenderOnce()`, 60000);
  const base1 = await shotCanvas(Page, Runtime, crypto);
  await evalOrThrow(Runtime, `${global}.debugRenderOnce()`, 60000);
  const base2 = await shotCanvas(Page, Runtime, crypto);
  if (base1.hash !== base2.hash) {
    return { rows, active, baseline: null, nLevels,
      aborted: `two identical renders produced different PNGs (${base1.hash} vs ${base2.hash}). `
        + 'Screenshot equality cannot score anything on this setup; nothing below would mean what it says.' };
  }
  log(`[setup] baseline reproducible: ${base1.hash} (${base1.bytes} B)`);

  // ── 2. one level at a time ─────────────────────────────────────────────────
  //
  // NATURAL ORDER, INCLUDING LEVEL 0. U7-6b had to run level 0 LAST: every row
  // restores the snapshot before perturbing, and the snapshot did not carry
  // the root pool, so a level-0 perturbation survived the restore and poisoned
  // every row after it (one PASS, then "the reference is moving" for the rest).
  // U7-6c put the root in the format, and this ordering going back to the
  // obvious one is that rung's gate: if the restore did not now undo a
  // level-0 perturbation, the rows below it would ABSTAIN again and say so.
  for (let m = 0; m < nLevels; m++) {
    if (active[m] === -1) {
      rows.push({ level: m, verdict: 'ABSTAIN', note: 'no pool at this level on this page/configuration' });
      continue;
    }
    if (active[m] === 0) {
      rows.push({ level: m, verdict: 'ABSTAIN', note: 'no active tiles at this level -- nothing to draw, so nothing to prove' });
      continue;
    }

    await evalOrThrow(Runtime, `${global}.debugSnapshotLoad(window.__RLGATE)`, 300000);
    await evalOrThrow(Runtime, `${global}.debugRenderOnce()`, 60000);
    const restored = await shotCanvas(Page, Runtime, crypto);
    if (restored.hash !== base1.hash) {
      rows.push({ level: m, verdict: 'ABSTAIN', note: `restore did not return to baseline (${restored.hash} != ${base1.hash}); the reference is moving` });
      continue;
    }

    const perturbed = await evalOrThrow(Runtime, `JSON.stringify(${global}.debugPerturbLevelVel(${m}, ${PERTURB}, ${PERTURB}))`);
    await evalOrThrow(Runtime, `${global}.debugRenderOnce()`, 60000);
    const after = await shotCanvas(Page, Runtime, crypto);

    const reached = after.hash !== base1.hash;
    rows.push({
      level: m,
      verdict: reached ? 'PASS' : 'FAIL',
      note: reached
        ? `picture changed (${base1.hash} -> ${after.hash})`
        : `picture UNCHANGED after overwriting ${JSON.parse(perturbed).cells} cells with u=(${PERTURB},${PERTURB}) -- this level has no path into the renderer`,
    });
  }

  // Leave the page live again so a sweep's next config does not inherit a
  // paused one.
  await evalOrThrow(Runtime, `${global}.setLive(true)`);
  return { rows, active, baseline: base1, nLevels, aborted: null };
}

module.exports = { runRenderLevels, PERTURB };
