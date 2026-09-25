#!/usr/bin/env node
// HOW CLOSE TO tau = 1/2 DOES THE 3D SOLVER ACTUALLY HOLD TOGETHER?
// plans/3D.md M8. Measured, not predicted.
//
// WHY THIS EXISTS. The 3D fork's target is the 2D project's: a free body
// tumbling and shedding at Re ~ 1100 (card-params.mjs's Pesavento & Wang
// regime). Re is INVARIANT down an AMR hierarchy -- card-params.mjs's
// tauAtLevel says so and tools/test-card-params.js asserts it -- and
// tau grows going FINER, so L0 carries the smallest tau and sets the
// stability limit for the whole run:
//
//     tau0 = 1/2 + 3 u D0 / Re,     D0 = (L0 width) / BLOCKAGE
//
// At the plan's stated target configuration (sec 3.3 Config B, L0 = 128^3)
// that is tau0 = 0.50218 at Re = 1100. Whether that runs is an empirical
// question about THIS solver on THIS lattice, and the 2D project's history
// is the reason not to answer it from a textbook: it ships at tau0 =
// 0.50873 after a great deal of hand-wringing about instability that the
// fluid then ignored.
//
// So: walk tau down, and report where the field actually stops being finite.
//
// WHAT IT REPORTS, and why two numbers rather than one:
//
//   SURVIVED   the field is still finite after the requested steps. This is
//              the number everyone wants and it is the weaker one.
//   DIVERGED   |u| passed a multiple of the case's own velocity scale while
//              still finite. A run can be "stable" and garbage, and on a
//              marginal tau that is the common outcome -- so this is
//              tracked separately and reported even when nothing NaNs.
//   WHERE      debugHotspot's argmax of |u|, annotated with the finest
//              level covering it, its distance to the body surface and its
//              distance to the nearest wall. A limit that is really the
//              bounce-back, or really the coarse/fine seam, is a different
//              finding from BGK giving out in the bulk, and the hotspot's
//              location is what tells them apart. It is sampled at every
//              checkpoint, so its MIGRATION is visible before the blowup.
//
// Owns the whole lifecycle (HTTPS server + a dedicated debug-port Chrome if
// neither is up, one tab reused via Page.navigate), like tools/validate-3d.js.
//
//   node tools/probe-d3-tau.js
//   node tools/probe-d3-tau.js --configs=sphere-D16 --steps=40000
//   node tools/probe-d3-tau.js --taus=0.52,0.51,0.505,0.502,0.501

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

// tau - 1/2, walked down roughly geometrically. 1e-2 is comfortable, 8.73e-3
// is where the 2D card ships, and 2.18e-3 is the plan's Config B at Re=1100.
const DEFAULT_TAUS = [0.52, 0.51, 0.50873, 0.506, 0.504, 0.503, 0.50218, 0.5015, 0.501, 0.5005];

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  steps: 20000, checkEvery: 1000, configs: null, taus: null, extra: '',
  timeout: 900, keepOpen: false, diverge: 4,
};

// One row per flow SHAPE, because "how close to 1/2" has a different answer
// for each and the differences are the finding.
//
// The tau knob differs by scenario and that is deliberate rather than
// awkward: `sphere` derives tau from Re, so sweeping Re IS sweeping tau and
// the page reports which. Sweeping ?tau= on the sphere would silently change
// the Reynolds number instead, which is the one thing this must not do.
const CONFIGS = [
  // The floor: smooth, periodic, force-free, decaying, no walls and no body.
  // Nothing here stresses the collision operator except the operator itself,
  // so a failure at this tau is BGK and nothing else.
  { name: 'beltrami', knob: 'tau',
    url: 'scenario=beltrami&n=48&u0=0.04&q=19&live=0&levels=1' },
  // Small scales made on purpose. TGV starts smooth and cascades energy to
  // the grid scale, which is the honest "does it hold together" test and the
  // closest thing in this suite to a shedding wake.
  { name: 'tgv', knob: 'tau',
    url: 'scenario=tgv&n=64&u0=0.05&q=19&live=0&levels=1' },
  // Walls and a boundary layer.
  { name: 'duct', knob: 'tau',
    url: 'scenario=duct&n=32&q=19&live=0&levels=1' },
  // A body, a wake, and bounce-back. The case the target actually resembles,
  // and the one where the hotspot's distance-to-surface earns its keep.
  { name: 'sphere-D16', knob: 're', tauFromRe: true,
    url: 'scenario=sphere&n=16&u0=0.05&q=19&bounceback=1&live=0&levels=1' },
  // The same body with a refined shell: does the coarse/fine interface give
  // out before the collision operator does? L0 carries the smallest tau, so
  // if the seam is the weak point this is where it shows.
  { name: 'sphere-D16-L2', knob: 're', tauFromRe: true,
    url: 'scenario=sphere&n=16&u0=0.05&q=19&bounceback=1&live=0&levels=2&rb=4&refine=body&interface=explode' },
];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--checkEvery=')) o.checkEvery = parseInt(a.slice(13));
    else if (a.startsWith('--configs=')) o.configs = a.slice(10).split(',');
    else if (a.startsWith('--taus=')) o.taus = a.slice(7).split(',').map(Number);
    else if (a.startsWith('--diverge=')) o.diverge = Number(a.slice(10));
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const e2 = (x) => (Number.isFinite(x) ? Number(x).toExponential(2) : String(x));

// Re that yields a given tau on a case whose tau is derived from Re.
// nu = (tau - 1/2)/3 and Re = u D / nu, with D the body diameter in L0 cells.
const reForTau = (tau, u, D) => (u * D) / ((tau - 0.5) / 3);

async function runOne(Runtime, o, c, tau, log) {
  const G = 'window.__D3';
  const p = await evalOrThrow(Runtime, `${G}.getParams()`, 30000, 'getParams');
  // The page is the authority on what tau it is running, not the URL: on the
  // sphere the knob is Re and the derivation is the page's.
  const tauActual = p.tau ?? p.tauCoarse;
  const uRef = p.u0 || p.uPeak || 0.05;
  const bound = o.diverge * uRef;

  // THE LAST FINITE HOTSPOT IS THE DIAGNOSIS, not the first NaN. Once a
  // BGK run goes it goes globally within a step or two, so "first non-finite
  // cell" degenerates to whatever raster order reaches first -- cell
  // (0,0,0), every time, which says nothing. What says something is where
  // |u| was piling up on the LAST checkpoint that was still finite.
  let survived = 0, diverged = null, blew = null, hot = null, lastFinite = null, maxSeen = 0;
  while (survived < o.steps) {
    const k = Math.min(o.checkEvery, o.steps - survived);
    await evalOrThrow(Runtime, `${G}.debugStepSync(${k})`, (o.timeout + 30) * 1000, 'debugStepSync');
    survived += k;
    const h = await evalOrThrow(Runtime, `${G}.debugHotspot()`, 600000, 'debugHotspot');
    hot = h;
    if (h.hot && h.hot.speed > maxSeen) maxSeen = h.hot.speed;
    if (!h.finite) { blew = { step: survived, ...h, lastFinite }; break; }
    lastFinite = h;
    if (!diverged && h.hot && h.hot.speed > bound) diverged = { step: survived, speed: h.hot.speed };
  }
  // Report the last finite picture when there is one, since that is the one
  // with a location in it.
  if (blew && lastFinite) hot = lastFinite;
  const where = (x) => x && x.hot
    ? `L${x.hot.level}`
      + (x.hot.sdf !== undefined ? ` sdf=${x.hot.sdf.toFixed(1)}` : '')
      + (x.hot.wall !== undefined ? ` wall=${x.hot.wall}` : '')
      + ` at ${x.hot.ijk.join(',')}`
    : '-';
  const verdict = blew ? `NaN@${blew.step}` : diverged ? `diverged@${diverged.step}` : `survived ${survived}`;
  log(`tau=${tauActual.toFixed(5)} (tau-0.5=${e2(tauActual - 0.5)})`
    + `  Re=${p.re ? p.re.toFixed(0) : '-'}  ${verdict}`
    + `  max|u|=${e2(maxSeen)} (${(maxSeen / uRef).toFixed(1)}x u_ref)  hot ${where(hot)}`
    + (blew ? `  [last finite @${blew.lastFinite ? blew.lastFinite.step : '-'},`
        + ` ${blew.nNonFinite} cells NaN at ${blew.step}]` : ''));
  return { tau: tauActual, re: p.re, steps: survived, blew, diverged, maxSeen, uRef, hot };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = CONFIGS.filter(c => !o.configs || o.configs.includes(c.name));
  if (!configs.length) { console.error('no configs selected'); process.exit(2); }
  const taus = o.taus || DEFAULT_TAUS;

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = await attachPageWatch(client, { onError: () => {} });

  const report = [];
  try {
    for (const c of configs) {
      console.log(`\n=== ${c.name}`);
      for (const tau of taus) {
        // The sphere's knob is Re; every other case takes tau directly.
        let q;
        if (c.tauFromRe) {
          const D = 16;                       // n=16 is the body diameter in L0 cells
          q = `re=${reForTau(tau, 0.05, D).toFixed(4)}`;
        } else {
          q = `tau=${tau}`;
        }
        const url = `${o.baseUrl}/index-3d.html?${c.url}&${q}${o.extra ? `&${o.extra}` : ''}`;
        try {
          await navigateTo(Page, url);
          await waitForGlobal(Runtime, 'window.__D3', 60000);
          const res = await runOne(Runtime, o, c, tau, s => console.log('    ' + s));
          report.push({ config: c.name, tauAsked: tau, ...res });
        } catch (err) {
          console.log(`    tau=${tau}: ERROR ${err.message.split('\n')[0]}`);
          report.push({ config: c.name, tauAsked: tau, error: err.message });
        }
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(120));
  console.log(`SUMMARY  ${o.steps} steps per point, |u| divergence bound ${o.diverge}x the case's velocity scale`);
  console.log('='.repeat(120));
  console.log(pad('config', 14) + pad('tau', 10) + pad('tau-0.5', 11) + pad('Re', 9)
    + pad('outcome', 18) + pad('max|u| / u_ref', 16) + 'hotspot');
  console.log('-'.repeat(120));
  const limits = {};
  for (const r of report) {
    if (r.error) { console.log(pad(r.config, 14) + pad(r.tauAsked, 10) + pad('-', 11) + pad('-', 9) + pad('ERROR', 18)); continue; }
    const outcome = r.blew ? `NaN @ ${r.blew.step}` : r.diverged ? `diverged @ ${r.diverged.step}` : 'survived';
    if (!r.blew && !r.diverged) limits[r.config] = Math.min(limits[r.config] ?? Infinity, r.tau);
    const h = r.hot && r.hot.hot;
    console.log(pad(r.config, 14) + pad(r.tau.toFixed(5), 10) + pad(e2(r.tau - 0.5), 11)
      + pad(r.re ? r.re.toFixed(0) : '-', 9) + pad(outcome, 18)
      + pad((r.maxSeen / r.uRef).toFixed(1) + 'x', 16)
      + (h ? `L${h.level}${h.sdf !== undefined ? ` sdf=${h.sdf.toFixed(1)}` : ''}${h.wall !== undefined ? ` wall=${h.wall}` : ''}` : '-'));
  }
  console.log('\nLOWEST tau THAT SURVIVED AND DID NOT DIVERGE, per config:');
  for (const [k, v] of Object.entries(limits)) {
    console.log(`  ${pad(k, 14)} tau = ${v.toFixed(5)}   (tau - 1/2 = ${e2(v - 0.5)})`);
  }
  console.log('\nThis is a MEASUREMENT AT ONE RESOLUTION AND ONE VELOCITY, not a law.');
  console.log('Lattice-Boltzmann stability at fixed tau depends on the local Mach number and');
  console.log('on the steepest gradient present, so a case that survives here can still fail');
  console.log('at higher u, a finer body, or once a wake is actually shedding. Re-measure');
  console.log('rather than extrapolating -- that is what this tool is for.');
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
