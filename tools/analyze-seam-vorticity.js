#!/usr/bin/env node
// Diagnostic for the "visible vorticity shear at quadtree seams" report
// (screenshot: a saturated-red vorticity streak sitting exactly on a
// coarse/fine block boundary, index-cylinder-amr.html). The question this
// answers: is the AMR-vs-dense field disagreement actually CONCENTRATED at
// coarse/fine level transitions (a real seam-coupling defect), or is it
// spread through the domain / concentrated at some other real flow feature
// (in which case the render is just amr_render.wgsl's 80x vorticity gain
// making an ordinary small discretization error look alarming)?
//
// Reuses tools/validate-amr-vs-dense.js's own machinery (deriveAMRParams,
// dense/AMR snapshot + reconstruction) rather than a second copy, then adds
// ONE new axis: tools/lib/field-reconstruct.js's buildLevelMap gives the
// active level (0/1/2/...) of every target cell from the SAME quadtree walk
// reconstructAMRToResolution already does. A cell is a "seam cell" if any of
// its 4 neighbors is at a different level; BFS from every seam cell gives
// every other cell's distance to the nearest seam. Binning |omega_amr -
// omega_dense| by that distance directly tests the hypothesis: if error is
// elevated at distance 0-1 and decays with distance, that's a seam defect;
// if it's flat across distance (or peaks somewhere with no relation to
// level), the seam is not the culprit.
//
// Usage:
//   node tools/analyze-seam-vorticity.js --res=9 --levels=2 --re=100
//   node tools/analyze-seam-vorticity.js --res=9 --levels=2 --re=40 --steps=8000

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal, evalExpr, teardown,
} = require('./lib/browser-lifecycle');
const { loadDenseFields, reconstructAMRToResolution, buildLevelMap } = require('./lib/field-reconstruct');
const { deriveAMRParams, deriveSharedURLParams, buildDenseUrl, buildAMRUrl } = require('./lib/amr-resolution-mapping');

const REPO_ROOT = path.join(__dirname, '..');

function parseArgs(argv) {
  const opts = {
    baseUrl: 'https://localhost:4444', port: 9333,
    res: 9, levels: 2, re: 100,
    blockage: 24, upstream: 8, u0: 0.04, seed: 12345, bounceback: true, perturb: 0,
    steps: null, // null => derive from 40*D/U0 transient, same sizing as validate-amr-vs-dense.js
    maxDist: 10,
    keepOpen: false,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--res=')) opts.res = parseInt(a.slice(6));
    else if (a.startsWith('--levels=')) opts.levels = parseInt(a.slice(9));
    else if (a.startsWith('--re=')) opts.re = parseFloat(a.slice(5));
    else if (a.startsWith('--u0=')) opts.u0 = parseFloat(a.slice(5));
    else if (a === '--diffuse') opts.bounceback = false;
    else if (a.startsWith('--steps=')) opts.steps = parseInt(a.slice(8));
    else if (a.startsWith('--maxDist=')) opts.maxDist = parseInt(a.slice(10));
    else if (a === '--keepOpen') opts.keepOpen = true;
  }
  return opts;
}

function roundSteps(n) { return Math.ceil(n / 64) * 64; }

// BFS multi-source distance-to-nearest-level-transition, capped at maxDist
// (cells farther than that all bucket into the last "maxDist+" row).
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
  // Cells the capped BFS never reached are, by construction, farther than
  // maxDist from any seam -- bucket them with the "maxDist+" row rather than
  // leaving the -1 sentinel (which would otherwise index buckets[-1]).
  for (let i = 0; i < dist.length; i++) if (dist[i] === -1) dist[i] = maxDist;
  return dist;
}

async function runLeg(Runtime, Page, url, steps, snapshotTimeoutMs) {
  await navigateTo(Page, url);
  await waitForGlobal(Runtime, 'window.__CYL', 15000);
  await evalExpr(Runtime, 'window.__CYL.setLive(false)');
  await evalExpr(Runtime, 'window.__CYL.reset()');
  const r = await evalExpr(Runtime, `window.__CYL.debugRunAndCollect(${steps})`, snapshotTimeoutMs);
  if (r.exceptionDetails) throw new Error(`debugRunAndCollect failed: ${r.exceptionDetails.text}`);
  const snap = await evalExpr(Runtime, 'window.__CYL.debugSnapshotSave()', snapshotTimeoutMs);
  if (snap.exceptionDetails) throw new Error(`debugSnapshotSave failed: ${snap.exceptionDetails.text}`);
  return snap.result.value;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const amrParams = deriveAMRParams({
    targetResLog2: opts.res, nLevels: opts.levels, blockage: opts.blockage, u0: opts.u0, re: opts.re,
  });
  const sharedParams = deriveSharedURLParams({
    blockage: opts.blockage, upstream: opts.upstream, re: opts.re, u0: opts.u0, seed: opts.seed,
    perturb: opts.perturb, bounceback: opts.bounceback,
  });
  const D = (1 << opts.res) / opts.blockage;
  const steps = opts.steps || roundSteps(40 * D / opts.u0);
  console.log(`res=${opts.res} levels=${opts.levels} re=${opts.re} bounceback=${opts.bounceback} steps=${steps} (D=${D.toFixed(1)})`);

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
    const denseUrl = buildDenseUrl(opts.baseUrl, { targetResLog2: opts.res, sharedParams });
    const denseSnap = await runLeg(Runtime, Page, denseUrl, steps, timeoutMs);
    const dense = loadDenseFields(denseSnap);

    console.log('running AMR (adaptive, default autoRefine)...');
    const amrUrl = buildAMRUrl(opts.baseUrl, { baseResLog2: amrParams.baseResLog2, nLevels: opts.levels, sharedParams });
    const amrSnap = await runLeg(Runtime, Page, amrUrl, steps, timeoutMs);
    const amr = reconstructAMRToResolution(amrSnap, opts.res);
    const levelMap = buildLevelMap(amrSnap, opts.res);

    const W = 1 << opts.res;
    const dist = distanceToSeam(levelMap, W, W, opts.maxDist);

    // Bin |omega_amr - omega_dense| by distance-to-seam. Also bin |omega_dense|
    // itself (the real local physical signal) at each bucket, so the error can
    // be read as a FRACTION of the real signal there, not just an absolute
    // number that might look small or large out of context.
    const buckets = Array.from({ length: opts.maxDist + 1 }, () => ({ n: 0, sumAbsErr: 0, maxAbsErr: 0, sumAbsRef: 0 }));
    let anyRefined = false;
    for (let i = 0; i < dist.length; i++) {
      if (levelMap[i] > 0) anyRefined = true;
      const d = Math.min(dist[i], opts.maxDist);
      const err = Math.abs(amr.omega[i] - dense.omega[i]);
      const b = buckets[d];
      b.n++; b.sumAbsErr += err; b.maxAbsErr = Math.max(b.maxAbsErr, err); b.sumAbsRef += Math.abs(dense.omega[i]);
    }
    if (!anyRefined) console.warn('WARNING: levelMap is all-zero -- AMR run never refined anything, this comparison is meaningless. Check autoRefine/thresholds.');

    console.log(`\nfinestCoverageFraction=${(amr.finestCoverageFraction * 100).toFixed(1)}%  (fraction of domain at AMR\'s finest configured level)`);
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

    const overallDiffStats = (() => {
      let sumSqDiff = 0, sumSqRef = 0, maxAbsDiff = 0;
      for (let i = 0; i < dense.omega.length; i++) {
        const d0 = amr.omega[i] - dense.omega[i];
        sumSqDiff += d0 * d0; sumSqRef += dense.omega[i] * dense.omega[i];
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(d0));
      }
      return { relL2: Math.sqrt(sumSqDiff / sumSqRef), maxAbsDiff };
    })();
    console.log(`\noverall omega relL2=${overallDiffStats.relL2.toExponential(3)}  maxAbsDiff=${overallDiffStats.maxAbsDiff.toExponential(3)}`);

  } finally {
    await client.close();
    await teardown({ port: opts.port, tabId, chrome, server, keepOpen: opts.keepOpen });
  }
}

main().catch(e => { console.error(e); process.exit(1); });
