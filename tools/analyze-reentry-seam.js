#!/usr/bin/env node
// Dense-vs-AMR seam-vorticity diagnostic for the "reentry" scenario
// (index-reentry.html / index-reentry-amr.html) -- the deterministic
// sibling of tools/analyze-seam-vorticity.js's cylinder-based version.
//
// Motivation: the cylinder-based version's Re=100 (shedding) run showed a
// seam-adjacent error elevation, but couldn't rule out vortex-shedding
// PHASE DRIFT between the two independently-run solvers as the real cause
// (a known, already-flagged caveat -- see validate-amr-vs-dense.js's own
// header) -- and the Re=40 (steady) run showed nothing, but steady flow
// never actually advects anything across the interface, so it couldn't
// exercise the hypothesis either. The reentry scenario's prescribed
// kinematics (shaders/physics.wgsl's KINEMATIC override) make the dense and
// AMR runs follow IDENTICAL trajectories -- confirmed bit-for-bit in
// main-reentry.js/main-reentry-amr.js's own CardState (vy/omega/theta/
// y_total match to the last bit at matched step counts) -- while still
// producing real, non-steady vortex shedding off the thin ellipse's sharp
// corners. So any dense-vs-AMR field disagreement here is attributable to
// the solvers themselves, not to trajectory drift.
//
// Same method as analyze-seam-vorticity.js: reconstruct both runs to the
// AMR's finest resolution, derive each target cell's active AMR level
// (tools/lib/field-reconstruct.js's buildLevelMap), bin |omega_amr -
// omega_dense| by distance to the nearest level transition.
//
// Usage:
//   node tools/analyze-reentry-seam.js --res=8 --levels=2 --steps=12000
//   node tools/analyze-reentry-seam.js --res=8 --levels=2 --steps=12000 --vy=0.0185 --omega=0.00059

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal, evalExpr, teardown,
} = require('./lib/browser-lifecycle');
const { loadDenseFields, reconstructAMRToResolution, buildLevelMap } = require('./lib/field-reconstruct');

const REPO_ROOT = path.join(__dirname, '..');

function parseArgs(argv) {
  const opts = {
    baseUrl: 'https://localhost:4444', port: 9333,
    baseRes: 8, levels: 2, steps: 12000,
    vy: 0.0185, omega: 0.00059, a: 32, b: 4, tau: 0.509,
    maxDist: 10,
    vortThresh: null,
    // fullrefine: force every level-1 block active (setAutoRefine(false) +
    // debugActivateBlock everywhere) instead of AMR's own adaptive
    // refine/coarsen -- zero coarse/fine interfaces anywhere, so the
    // distance-to-seam binning below is a null result by construction (every
    // cell is "distance 10+"). The point isn't the binning here -- it's the
    // single aggregate relL2/maxAbsDiff, as a control for whether the
    // adaptive run's seam-adjacent error elevation is actually caused by the
    // coarse/fine interface, or just reflects two independently-implemented
    // solvers disagreeing most wherever gradients are steepest (which is
    // also, by construction of the vorticity-driven refine criterion, where
    // adaptive mode happens to place its seams) -- same distinction
    // validate-amr-vs-dense.js's own --mode=fullrefine draws for the
    // cylinder scenario.
    mode: 'adaptive',
    keepOpen: false,
  };
  for (const arg of argv) {
    if (arg.startsWith('--baseUrl=')) opts.baseUrl = arg.slice(10);
    else if (arg.startsWith('--port=')) opts.port = parseInt(arg.slice(7));
    else if (arg.startsWith('--res=')) opts.baseRes = parseInt(arg.slice(6));
    else if (arg.startsWith('--levels=')) opts.levels = parseInt(arg.slice(9));
    else if (arg.startsWith('--steps=')) opts.steps = parseInt(arg.slice(8));
    else if (arg.startsWith('--vy=')) opts.vy = parseFloat(arg.slice(5));
    else if (arg.startsWith('--omega=')) opts.omega = parseFloat(arg.slice(8));
    else if (arg.startsWith('--a=')) opts.a = parseFloat(arg.slice(4));
    else if (arg.startsWith('--b=')) opts.b = parseFloat(arg.slice(4));
    else if (arg.startsWith('--tau=')) opts.tau = parseFloat(arg.slice(6));
    else if (arg.startsWith('--maxDist=')) opts.maxDist = parseInt(arg.slice(10));
    else if (arg.startsWith('--vortThresh=')) opts.vortThresh = parseFloat(arg.slice(13));
    else if (arg.startsWith('--mode=')) opts.mode = arg.slice(7);
    else if (arg === '--keepOpen') opts.keepOpen = true;
  }
  return opts;
}

function roundSteps(n) { return Math.ceil(n / 64) * 64; }

// BFS multi-source distance-to-nearest-level-transition -- identical to
// analyze-seam-vorticity.js's own (kept as a separate copy rather than a
// shared tools/lib export: both call sites are short enough, and each
// diagnostic script staying self-contained matches this project's own
// "main*.js entry points don't cross-import" convention applied to tools/).
function distanceToSeam(levelMap, W, H, maxDist) {
  const dist = new Int32Array(W * H).fill(-1);
  const queue = new Int32Array(W * H);
  let qh = 0, qt = 0;
  const idx = (x, y) => y * W + x;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const lvl = levelMap[idx(x, y)];
      const xp = (x + 1) % W, xm = (x - 1 + W) % W, yp = (y + 1) % H, ym = (y - 1 + H) % H;
      const isSeam = levelMap[idx(xp, y)] !== lvl || levelMap[idx(xm, y)] !== lvl ||
                     levelMap[idx(x, yp)] !== lvl || levelMap[idx(x, ym)] !== lvl;
      if (isSeam) { dist[idx(x, y)] = 0; queue[qt++] = idx(x, y); }
    }
  }
  while (qh < qt) {
    const c = queue[qh++];
    const d = dist[c];
    if (d >= maxDist) continue;
    const x = c % W, y = Math.floor(c / W);
    const xp = (x + 1) % W, xm = (x - 1 + W) % W, yp = (y + 1) % H, ym = (y - 1 + H) % H;
    for (const n of [idx(xp, y), idx(xm, y), idx(x, yp), idx(x, ym)]) {
      if (dist[n] === -1) { dist[n] = d + 1; queue[qt++] = n; }
    }
  }
  for (let i = 0; i < dist.length; i++) if (dist[i] === -1) dist[i] = maxDist;
  return dist;
}

// main-reentry.js exposes window.__CYL (forked from main-cylinder.js's own
// debug-API shape); main-reentry-amr.js exposes window.__AMR (forked from
// main-amr.js's own shape, per that family's established convention) --
// same two-namespaces split this whole codebase already has between the
// __CYL-shaped pages and main-amr.js itself.
// Mirrors validate-amr-vs-dense.js's own fullyRefine (main-cylinder-amr.js
// target) -- same block-doubles-per-level geometry (every main-*-amr.js
// page shares the same pool-allocation convention), just against
// window.__AMR instead of window.__CYL. Batched into ONE Runtime.evaluate
// (not one CDP round-trip per block): each debugActivateBlock call is a
// real GPU sync, and thousands of individual round-trips from Node would be
// needlessly slow.
async function fullyRefine(Runtime, nLevels, timeoutMs) {
  const expr = `(async () => {
    const dims = window.__AMR.getBlockGridDims();
    window.__AMR.setAutoRefine(false);
    for (let level = 1; level < ${nLevels}; level++) {
      const nbx = dims.NBX * (2 ** (level - 1));
      const nby = dims.NBY * (2 ** (level - 1));
      for (let by = 0; by < nby; by++) {
        for (let bx = 0; bx < nbx; bx++) {
          await window.__AMR.debugActivateBlock(bx, by, level);
        }
      }
    }
    return { ok: true };
  })()`;
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`fullyRefine failed: ${r.exceptionDetails.text}`);
  return r.result.value;
}

async function runLeg(Runtime, Page, url, globalName, steps, timeoutMs, mode, nLevels) {
  await navigateTo(Page, url);
  await waitForGlobal(Runtime, `window.${globalName}`, 15000);
  await evalExpr(Runtime, `window.${globalName}.setLive(false)`);
  await evalExpr(Runtime, `window.${globalName}.reset()`);
  if (mode === 'fullrefine' && globalName === '__AMR') {
    console.log('  activating every block at every level...');
    await fullyRefine(Runtime, nLevels, timeoutMs);
  }
  const r = await evalExpr(Runtime, `window.${globalName}.debugStepSync(${steps})`, timeoutMs);
  if (r.exceptionDetails) throw new Error(`debugStepSync failed: ${r.exceptionDetails.text}`);
  const snap = await evalExpr(Runtime, `window.${globalName}.debugSnapshotSave()`, timeoutMs);
  if (snap.exceptionDetails) throw new Error(`debugSnapshotSave failed: ${snap.exceptionDetails.text}`);
  return snap.result.value;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const steps = roundSteps(opts.steps);
  const targetRes = opts.baseRes + (opts.levels - 1);
  const sharedQS = `a=${opts.a}&b=${opts.b}&tau=${opts.tau}&vy=${opts.vy}&omega=${opts.omega}`;
  console.log(`baseRes=${opts.baseRes} (target=${targetRes}) levels=${opts.levels} steps=${steps} vy=${opts.vy} omega=${opts.omega}`);

  const server = await ensureServer(opts.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(opts.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(opts.port) : await openTab(opts.port, 'about:blank');
  const client = await CDP({ port: opts.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  try {
    const timeoutMs = 300000;
    console.log('running dense reference...');
    const denseUrl = `${opts.baseUrl}/index-reentry.html?res=${targetRes}&${sharedQS}`;
    const denseSnap = await runLeg(Runtime, Page, denseUrl, '__CYL', steps, timeoutMs);
    const dense = loadDenseFields(denseSnap);

    console.log(`running AMR (${opts.mode})...`);
    // fullrefine (levels=2 only, this tool's own scope) needs every level-1
    // block held active at once -- NBX*NBY of them, vs. the page's own
    // default maxFineBlocks=128 (sized for adaptive's typically-sparse
    // coverage).
    const NBX = (1 << opts.baseRes) / 8;
    const maxFineBlocksQS = opts.mode === 'fullrefine' ? `&maxFineBlocks=${NBX * NBX}` : '';
    const amrUrl = `${opts.baseUrl}/index-reentry-amr.html?res=${opts.baseRes}&levels=${opts.levels}&${sharedQS}${maxFineBlocksQS}`;
    const amrSnap = await runLeg(Runtime, Page, amrUrl, '__AMR', steps, timeoutMs, opts.mode, opts.levels);
    const amr = reconstructAMRToResolution(amrSnap, targetRes);
    const levelMap = buildLevelMap(amrSnap, targetRes);

    // Sanity check: the two runs' own CardState trajectories should be
    // (near-)identical -- this is the whole point of KINEMATIC mode. If
    // they've drifted apart, something's wrong with the harness, not with
    // AMR's field accuracy, and the rest of this analysis is meaningless.
    const dc = denseSnap.cardState, ac = amrSnap.cardState;
    console.log(`dense trajectory: step=${denseSnap.step} theta=${dc[2].toFixed(6)} y_total=${dc[20].toFixed(6)}`);
    console.log(`amr   trajectory: step=${amrSnap.step} theta=${ac[2].toFixed(6)} y_total=${ac[20].toFixed(6)}`);
    if (Math.abs(dc[2] - ac[2]) > 1e-4 || Math.abs(dc[20] - ac[20]) > 1e-4) {
      console.warn('WARNING: dense/AMR trajectories disagree by more than 1e-4 -- check vy/omega/a/b/tau match on both URLs before trusting the field diff below.');
    }

    const W = 1 << targetRes;
    const dist = distanceToSeam(levelMap, W, W, opts.maxDist);

    const buckets = Array.from({ length: opts.maxDist + 1 }, () => ({ n: 0, sumAbsErr: 0, maxAbsErr: 0, sumAbsRef: 0 }));
    let anyRefined = false;
    for (let i = 0; i < dist.length; i++) {
      if (levelMap[i] > 0) anyRefined = true;
      const d = Math.min(dist[i], opts.maxDist);
      const err = Math.abs(amr.omega[i] - dense.omega[i]);
      const b = buckets[d];
      b.n++; b.sumAbsErr += err; b.maxAbsErr = Math.max(b.maxAbsErr, err); b.sumAbsRef += Math.abs(dense.omega[i]);
    }
    if (!anyRefined) console.warn('WARNING: levelMap is all-zero -- AMR run never refined anything, this comparison is meaningless.');

    console.log(`\nfinestCoverageFraction=${(amr.finestCoverageFraction * 100).toFixed(1)}%`);
    console.log('\ndistance-to-nearest-level-transition -> |omega_amr - omega_dense|');
    console.log('dist  cells      mean|err|     max|err|      mean|omega_dense|   mean|err|/mean|omega_dense|');
    for (let d = 0; d <= opts.maxDist; d++) {
      const b = buckets[d];
      if (b.n === 0) continue;
      const meanErr = b.sumAbsErr / b.n;
      const meanRef = b.sumAbsRef / b.n;
      const label = d === opts.maxDist ? `${d}+` : `${d}`;
      console.log(
        `${label.padEnd(5)} ${String(b.n).padEnd(10)} ${meanErr.toExponential(3).padEnd(13)} ${b.maxAbsErr.toExponential(3).padEnd(13)} ` +
        `${meanRef.toExponential(3).padEnd(17)} ${meanRef > 0 ? (meanErr / meanRef).toFixed(3) : 'n/a'}`
      );
    }

    // Cross-tab: seam-proximity vs. local-vorticity-magnitude, so "near a
    // seam" and "near a real high-gradient corner-vortex feature" (which
    // the vorticity-driven refine criterion geometrically confounds -- of
    // course seams end up near strong gradients, that's WHERE refinement
    // happens) can be told apart. VORT_THRESH defaults to this run's own
    // near-seam bucket's mean|omega_dense| * 0.5 -- but that only exists
    // when there ARE seams (adaptive mode); pass --vortThresh= explicitly
    // (using the value an adaptive run at the same steps/vy/omega printed)
    // to get a directly comparable fullrefine cross-tab -- otherwise
    // fullrefine has no "near" cells to derive it from and the threshold
    // degenerates to 0.
    let VORT_THRESH = opts.vortThresh;
    if (VORT_THRESH == null) {
      const nearN = buckets.slice(0, Math.min(10, opts.maxDist)).reduce((s, b) => s + b.n, 0);
      const nearSeamMeanRef = buckets.slice(0, Math.min(10, opts.maxDist)).reduce((s, b) => s + b.sumAbsRef, 0) / Math.max(1, nearN);
      VORT_THRESH = nearSeamMeanRef * 0.5;
      if (nearN === 0) console.warn('WARNING: no near-seam cells to derive VORT_THRESH from (fullrefine mode?) -- pass --vortThresh= explicitly for a comparable cross-tab.');
    }
    const cross = { nearHigh: { n: 0, sum: 0 }, nearLow: { n: 0, sum: 0 }, farHigh: { n: 0, sum: 0 }, farLow: { n: 0, sum: 0 } };
    for (let i = 0; i < dist.length; i++) {
      const near = dist[i] < opts.maxDist; // strictly less than the "far" catch-all bucket
      const high = Math.abs(dense.omega[i]) >= VORT_THRESH;
      const key = (near ? 'near' : 'far') + (high ? 'High' : 'Low');
      cross[key].n++; cross[key].sum += Math.abs(amr.omega[i] - dense.omega[i]);
    }
    console.log(`\ncross-tab (VORT_THRESH=${VORT_THRESH.toExponential(3)}, half this run's own near-seam mean|omega_dense|):`);
    console.log('              n          mean|err|');
    for (const [k, v] of Object.entries(cross)) {
      console.log(`  ${k.padEnd(10)} ${String(v.n).padEnd(10)} ${v.n > 0 ? (v.sum / v.n).toExponential(3) : 'n/a'}`);
    }
    console.log('  (nearHigh vs farHigh isolates seam-proximity at MATCHED local vorticity magnitude --');
    console.log('   if nearHigh isn\'t clearly above farHigh, the seam-adjacent elevation seen in the plain');
    console.log('   distance table is likely just "corners have strong gradients", not a seam-specific defect.)');

    let sumSqDiff = 0, sumSqRef = 0, maxAbsDiff = 0;
    for (let i = 0; i < dense.omega.length; i++) {
      const d0 = amr.omega[i] - dense.omega[i];
      sumSqDiff += d0 * d0; sumSqRef += dense.omega[i] * dense.omega[i];
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(d0));
    }
    console.log(`\noverall omega relL2=${Math.sqrt(sumSqDiff / sumSqRef).toExponential(3)}  maxAbsDiff=${maxAbsDiff.toExponential(3)}`);

  } finally {
    await client.close();
    await teardown({ port: opts.port, tabId, chrome, server, keepOpen: opts.keepOpen });
  }
}

main().catch(e => { console.error(e); process.exit(1); });
