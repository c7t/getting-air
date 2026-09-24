#!/usr/bin/env node
// Field scaling (plans/uniform-levels.md sec 8): what does the DOMAIN do to the
// answer, and what does it cost? Reports, does not PASS/FAIL -- there is no
// literature value for how Cd should move with sponge width, and the point is
// the shape of each ladder.
//
// Pinned cylinder on index-cylinder-amr.html, one tab reused across configs
// (browser-lifecycle.js). Per config: Cd mean, Cl rms and St over a window
// sized in D/U0, so every rung gets the same PHYSICAL transient and
// measurement; the shedding ONSET time; a SETTLED check (the window's two
// halves must agree); live tile counts per level sampled through the
// measurement window; and two costs.
//
// THE TRANSIENT IS 150 D/U0, NOT validate-all's 40 (--transientD=). Shedding
// onset on this page is 41-60 D/U0 from reset, so a 40 D/U0 transient opens
// the window on the onset ramp and reads its mean as a Cd shift -- 0.6-3% low
// on the four standing AMR cylinder configs (plans/uniform-levels.md S8-4).
// `settle` in the report is the guard: over 1% means the window moved.
//
//   model   cell-substeps per D/U0 of physical time: (W^2 + sum_m tiles_m *
//           (2RB)^2 * 2^m) per root step, times the root steps in D/U0. The
//           same accounting as tools/lib/amr-cost.js. Deterministic, so it is
//           the number to compare across rungs.
//   wall    ms of debugRunAndCollect per D/U0. Real, but it carries a sync
//           per 64 root steps and whatever else the GPU is doing; read it as a
//           cross-check on the model, not as the measurement.
//
// THE LADDERS (all at fixed Re, U0 and finest-level dx, so the body is resolved
// identically on every rung -- R_finest = 2^(res + levels - 2) / blockage):
//
//   grid    res - 1, levels + 1, blockage fixed. SAME PHYSICAL DOMAIN, same
//           body resolution; only the decomposition moves: the root is 2x
//           coarser per rung. Run twice by default -- sponge pinned in ROOT
//           cells (physically 2x wider per rung, what the page does today) and
//           pinned PHYSICALLY (spongeW halves) -- so the sponge's share of any
//           movement is separated from everything else's.
//   extent  res fixed, levels + 1, blockage x2. The domain doubles per axis at
//           the same root cell count: sec 8.2's "extent exponential in levels,
//           cost linear" claim, measured. Sponge in root cells (the cheap
//           policy), so its physical width doubles with the domain.
//   sponge  one config, spongeW swept in root cells.
//
//   node tools/probe-field-scaling.js --ladder=grid
//   node tools/probe-field-scaling.js --ladder=extent --re=100
//   node tools/probe-field-scaling.js --ladder=sponge --res=9 --levels=2 --widths=1,2,4,8,16
//   node tools/probe-field-scaling.js --ladder=grid --extra=interface=interp
//   node tools/probe-field-scaling.js --baseUrl=https://localhost:4471 --port=9471
//
// Defaults: ?interface=explode (sec 8.3a -- the interp seam is a momentum
// source that grows with seam area, and every rung here adds seam) and
// ?detslots=1 (the cylinder page's racing free list otherwise moves Cd by up
// to 0.018 between configurations with no physics change; see
// main-cylinder-amr.js above POOL_PEAKS).

const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const fs = require('fs');
const path = require('path');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal, teardown, evalExpr,
} = require('./lib/browser-lifecycle');
const { computeWindow, analyze, roundSteps } = require('./lib/cylinder-metrics');

const REPO_ROOT = path.resolve(__dirname, '..');
const BENCH = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'benchmarks/cylinder.json'), 'utf8'));

function parseArgs(argv) {
  const o = {
    port: 9333, baseUrl: 'https://localhost:4444', ladder: 'grid', re: 100,
    res: 9, levels: 2, blockage: 24, rungs: 3, widths: [1, 2, 4, 8, 16], sponge: 4,
    extra: 'interface=explode&detslots=1', addExtra: '', chunks: 8, timeout: 3600,
    keepOpen: false, json: null, periods: 15, transientD: 150,
  };
  for (const a of argv) {
    const [k, v] = a.replace(/^--/, '').split(/=(.*)/s);
    if (k === 'port') o.port = parseInt(v);
    else if (k === 'baseUrl') o.baseUrl = v;
    else if (k === 'ladder') o.ladder = v;
    else if (k === 're') o.re = parseFloat(v);
    else if (k === 'res') o.res = parseInt(v);
    else if (k === 'levels') o.levels = parseInt(v);
    else if (k === 'blockage') o.blockage = parseFloat(v);
    else if (k === 'rungs') o.rungs = parseInt(v);
    else if (k === 'widths') o.widths = v.split(',').map(Number);
    else if (k === 'sponge') o.sponge = parseFloat(v);
    else if (k === 'extra') o.extra = v;          // replaces the defaults
    else if (k === 'addExtra') o.addExtra = v;    // appended to them
    else if (k === 'chunks') o.chunks = parseInt(v);
    else if (k === 'timeout') o.timeout = parseInt(v);
    else if (k === 'keepOpen') o.keepOpen = true;
    else if (k === 'json') o.json = v;
    else if (k === 'periods') o.periods = parseFloat(v);
    else if (k === 'transientD') o.transientD = parseFloat(v);
    else throw new Error(`unknown argument --${k}`);
  }
  return o;
}

function configsFor(o) {
  const c = [];
  if (o.ladder === 'grid') {
    for (const mode of ['root', 'phys']) {
      for (let k = 0; k < o.rungs; k++) {
        const spongeW = mode === 'root' ? o.sponge : o.sponge / 2 ** k;
        c.push({ name: `grid-${mode}-r${o.res - k}-N${o.levels + k}`, res: o.res - k, levels: o.levels + k, blockage: o.blockage, spongeW });
      }
    }
  } else if (o.ladder === 'extent') {
    for (let k = 0; k < o.rungs; k++) {
      c.push({ name: `extent-B${o.blockage * 2 ** k}-N${o.levels + k}`, res: o.res, levels: o.levels + k, blockage: o.blockage * 2 ** k, spongeW: o.sponge });
    }
  } else if (o.ladder === 'sponge') {
    for (const w of o.widths) c.push({ name: `sponge-${w}`, res: o.res, levels: o.levels, blockage: o.blockage, spongeW: w });
  } else if (o.ladder === 'one') {
    c.push({ name: `one-r${o.res}-N${o.levels}-B${o.blockage}-s${o.sponge}`, res: o.res, levels: o.levels, blockage: o.blockage, spongeW: o.sponge });
  } else throw new Error(`--ladder=${o.ladder}: expected grid, extent, sponge or one`);
  return c;
}

function urlFor(o, c) {
  // upstream = blockage/2 ON EVERY RUNG: the page's default (12 D) centres the
  // body only at blockage 24, so an extent ladder that left it would grow the
  // domain downstream only and move the body relative to the sponge.
  const q = [`res=${c.res}`, `levels=${c.levels}`, `blockage=${c.blockage}`, `upstream=${c.blockage / 2}`, `spongeW=${c.spongeW}`, `re=${o.re}`];
  if (o.extra) q.push(o.extra.replace(/^&/, ''));
  if (o.addExtra) q.push(o.addExtra.replace(/^&/, ''));
  return `${o.baseUrl}/index-cylinder-amr.html?${q.join('&')}`;
}

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const rms = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };

async function runConfig(Runtime, o, c, caseEntry) {
  const ev = async (x, t) => {
    const r = await evalExpr(Runtime, x, t || 120000);
    if (r.exceptionDetails) throw new Error(`${c.name}: ${x.slice(0, 60)}: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
    return r.result.value;
  };
  const status = await ev(`document.getElementById('status').textContent`);
  if (/^error:/.test(status)) throw new Error(`${c.name}: page refused/failed: ${status}`);
  await ev(`window.__CYL.setLive(false)`);
  await ev(`window.__CYL.reset()`);
  const p = await ev(`window.__CYL.getParams()`);
  const nLevels = await ev(`window.__CYL.getNumLevels()`);
  const { RB } = await ev(`window.__CYL.getBlockGridDims()`);
  const taus = await ev(`Array.from({length: ${nLevels}}, (_, m) => window.__CYL.tauAtLevel(m))`);
  if (Math.abs(p.spongeW - c.spongeW) > 1e-9) throw new Error(`${c.name}: page reports spongeW=${p.spongeW}, asked for ${c.spongeW} -- stale page?`);

  const w = computeWindow({ st: caseEntry.st }, p.D, p.U0);
  const transient = roundSteps(o.transientD * p.D / p.U0);
  const measurement = caseEntry.st ? roundSteps(o.periods * p.D / (caseEntry.st * p.U0)) : w.measurement;
  const chunk = roundSteps(measurement / o.chunks);

  const t0 = Date.now();
  await ev(`window.__CYL.debugRunAndCollect(${transient}).then(r => r.history.length)`, o.timeout * 1000);
  const tiles = [];   // per sample: [L1, L2, ...]
  let ran = transient;
  while (ran < transient + measurement) {
    await ev(`window.__CYL.debugRunAndCollect(${chunk}).then(r => r.history.length)`, o.timeout * 1000);
    ran += chunk;
    tiles.push(await ev(`Promise.all(Array.from({length: ${nLevels - 1}}, (_, i) => window.__CYL.debugListActiveBlocks(i + 1).then(a => a.length)))`));
  }
  const wallMs = Date.now() - t0;
  const history = await ev(`window.__CYL.getForceHistory()`);
  const cov = await ev(`window.__CYL.debugCheckGeometryCoverage().then(r => r.ok ? 0 : (Array.isArray(r.violations) ? r.violations.length : 'FAIL'))`).catch(e => `err: ${e.message}`);

  const a = analyze(history, transient, p.D, p.U0);
  const win = history.filter(r => r[0] > transient);
  const cl = win.map(r => r[4]);
  const cd = win.map(r => r[3]);
  // SETTLED? The window's two halves must agree. Shedding ONSET time is not
  // invariant across these ladders (it grows from the seeded perturbation at a
  // rate the root's tau and resolution both touch), and a window that opens on
  // the tail of the onset ramp reads its mean as a Cd shift -- measured: at
  // res 9 levels 2, 40 D/U0 of transient left Cd climbing 1.405 -> 1.73
  // through the whole window, a 3% bias that looked like a ladder effect.
  const half = Math.floor(win.length / 2);
  const cdHalves = [mean(cd.slice(0, half)), mean(cd.slice(half))];
  const clHalves = [rms(cl.slice(0, half)), rms(cl.slice(half))];
  // Shedding ONSET, in D/U0 from reset: the first time |Cl|'s running envelope
  // (max over one shedding period) reaches half its level in the window.
  let onsetT = null;
  if (caseEntry.st) {
    const per = Math.max(1, Math.round(p.D / (caseEntry.st * p.U0) / 64));
    const env = history.map((_, i) => Math.max(...history.slice(Math.max(0, i - per), i + 1).map(r => Math.abs(r[4]))));
    const target = 0.5 * Math.max(...env.filter((_, i) => history[i][0] > transient));
    const i0 = env.findIndex(e => e >= target);
    if (i0 >= 0) onsetT = history[i0][0] * p.U0 / p.D;
  }
  const tileMean = Array.from({ length: nLevels - 1 }, (_, i) => mean(tiles.map(t => t[i])));
  const tileMax = Array.from({ length: nLevels - 1 }, (_, i) => Math.max(...tiles.map(t => t[i])));
  const cellsPerRootStep = p.W * p.H + tileMean.reduce((s, n, i) => s + n * (2 * RB) ** 2 * 2 ** (i + 1), 0);
  const rootStepsPerT = p.D / p.U0;          // root steps per D/U0
  const physT = ran / rootStepsPerT;
  return {
    name: c.name, res: c.res, levels: nLevels, blockage: p.blockage, spongeW: p.spongeW,
    spongeD: p.spongeW / p.D, domainD: p.W / p.D, D_root: p.D, D_finest: p.D * 2 ** (nLevels - 1),
    tau0: taus[0], tauFinest: taus[nLevels - 1], steps: ran, physT,
    cdMean: a.cdMean, cdRms: rms(cd), clRms: rms(cl), st: a.st, crossings: a.crossings,
    tileMean, tileMax, coverage: cov, cdHalves, clHalves, onsetT,
    settle: Math.max(Math.abs(cdHalves[1] - cdHalves[0]) / Math.abs(cdHalves[1]), Math.abs(clHalves[1] - clHalves[0]) / Math.max(clHalves[1], 1e-9)),
    modelCost: cellsPerRootStep * rootStepsPerT,
    wallPerT: wallMs / physT, wallMs, finite: Number.isFinite(a.cdMean),
    transient, history,   // [rootStep, fx, fy, Cd, Cl] every 64 root steps -- in the --json only
  };
}

function fmt(v, d = 4) { return v == null || Number.isNaN(v) ? '   --  ' : Number(v).toFixed(d); }

function report(rows) {
  console.log('\n' + '='.repeat(118));
  console.log('config                     domain  spongeD  tau0     Cd      Cd_rms   Cl_rms   St      tiles(mean L1..)        Mcell/T  ms/T');
  console.log('-'.repeat(118));
  for (const r of rows) {
    if (r.error) { console.log(`${r.name.padEnd(26)} ERROR ${r.error}`); continue; }
    console.log(`${r.name.padEnd(26)} ${fmt(r.domainD, 0).padStart(5)}D ${fmt(r.spongeD, 3).padStart(7)}  ${fmt(r.tau0, 4)}  ${fmt(r.cdMean)}  ${fmt(r.cdRms)}  ${fmt(r.clRms)}  ${fmt(r.st)}  `
      + `${r.tileMean.map(n => n.toFixed(0)).join('/').padEnd(22)} ${(r.modelCost / 1e6).toFixed(1).padStart(7)}  ${r.wallPerT.toFixed(1).padStart(6)}`
      + `  onset ${r.onsetT == null ? '--' : r.onsetT.toFixed(0)}  settle ${(100 * r.settle).toFixed(1)}%` + (r.settle > 0.01 ? ' UNSETTLED' : '')
      + (r.coverage ? `  coverage=${r.coverage}` : ''));
  }
  console.log('='.repeat(118));
  console.log('settle = max relative change of Cd mean / Cl rms between the window halves; > 1% means the transient was too short');
  console.log('domain = W/D, spongeD = ramp width in D, Mcell/T = model cell-substeps per D/U0 (millions), ms/T = wall ms per D/U0');
}

(async () => {
  const o = parseArgs(process.argv.slice(2));
  const caseEntry = BENCH.cases.find(c => c.re === o.re) || { re: o.re, st: null };
  const configs = configsFor(o);
  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Page, Runtime, Network } = client;
  await Page.enable(); await Runtime.enable(); await Network.enable();
  // A cached main-cylinder-amr.js would run the previous build and still
  // report a plausible Cd (plans/2D-backport.md B6-8). getParams().spongeW is
  // checked against the request as a second line of defence.
  await Network.setCacheDisabled({ cacheDisabled: true });
  const rows = [];
  try {
    for (const c of configs) {
      const url = urlFor(o, c);
      console.log(`[${c.name}] ${url}`);
      try {
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, 'window.__CYL', 30000);
        const r = await runConfig(Runtime, o, c, caseEntry);
        console.log(`[${c.name}] Cd ${fmt(r.cdMean)} St ${fmt(r.st)} Cl_rms ${fmt(r.clRms)} tiles ${r.tileMean.map(n => n.toFixed(0)).join('/')} `
          + `(${r.steps} root steps = ${r.physT.toFixed(0)} D/U0, ${(r.wallMs / 1000).toFixed(0)} s)`);
        rows.push(r);
      } catch (e) {
        console.log(`[${c.name}] ERROR ${e.message}`);
        rows.push({ name: c.name, error: e.message });
      }
      if (o.json) fs.writeFileSync(o.json, JSON.stringify({ opts: o, rows }, null, 2));
    }
    report(rows);
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }
  console.log('PROBE COMPLETE');
})().catch(e => { console.error(e); process.exit(1); });
