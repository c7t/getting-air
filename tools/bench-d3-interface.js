#!/usr/bin/env node
// Frame-scale A/B of the 3D coarse/fine interface couplings (plans/3D.md
// M4.1). Answers one question: does ?interface=explode cost more per
// macro-step than the ?interface=interp it is meant to replace, and if so,
// where.
//
// WHY WALL CLOCK AND NOT PER-PASS TIMESTAMPS. plans/perf-characterization.md
// records the trap that produced a confident wrong conclusion in 2D:
// per-pass GPU timestamps are unusable where the counter granularity
// (65536 ns on the PowerVR part) exceeds the pass duration, so attribution
// has to be done at FRAME scale by differencing configurations that differ
// by one pass. That is exactly what this does -- every row is the same
// topology and the same step count, and the rows are read against each
// other, not against a roofline.
//
// WHY THE TOPOLOGY IS PINNED. Every config runs the SAME ?refine= geometry
// at the same N and RB, so the tile count, the dispatch shapes and the
// memory footprint are identical across rows. Refinement is static
// (M4.2 has not landed), so this needs no equivalent of the 2D bench's
// "freeze refinement" step -- but it is why this tool must be revisited when
// it does.
//
// HOW TO READ THE ROWS.
//
//   dense        no pool at all: the L0 solver's own rate, the floor that
//                says whether a slowdown is the interface or the machine.
//   interp       the M3 coupling. interp + average.
//   explode      M4.1's explode + coalesce, linear explosion.
//   explode-u    the same with ?explin=0. The DIFFERENCE between these two
//                is the linear explosion's gradient: six extra coarse loads
//                per direction per ring cell, which is the one part of M4.1c
//                with an obvious cost. If explode is slow and explode-u is
//                not, that is where it went.
//   explode-noorph  the same with ?orphans=0, which prices the coalesce
//                orphan pass the same way. PHYSICS-WRONG by construction --
//                it reinstates the convex-edge leak -- and here only to be
//                differenced against `explode`.
//
// A cliff would look like explode running several times interp, or the fine
// step's rate collapsing -- the silent-spill signature sec 2.4 warns about.
// A few percent either way is not a cliff; the repeat spread is printed so
// that judgement is made against this machine's actual noise rather than
// against an assumption about it.
//
//   node tools/bench-d3-interface.js
//   node tools/bench-d3-interface.js --n=48 --steps=200 --reps=5
//   node tools/bench-d3-interface.js --refine=body --scenario=sphere

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  n: 48, tau: 0.8, u0: 0.04, q: 19, rb: 4, refine: 'box', boxfrac: 0.5,
  scenario: 'beltrami', steps: 200, warmup: 40, reps: 5, configs: null,
  extra: '', timeout: 600, keepOpen: false,
};

const CONFIGS = [
  { name: 'dense', levels: 1 },
  { name: 'interp', levels: 2, iface: 'interp' },
  { name: 'explode', levels: 2, iface: 'explode' },
  { name: 'explode-u', levels: 2, iface: 'explode', explin: 0 },
  // TIMING ONLY, and physics-wrong: prices the orphan pass by removing it.
  { name: 'explode-noorph', levels: 2, iface: 'explode', orphans: 0 },
];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--n=')) o.n = parseInt(a.slice(4));
    else if (a.startsWith('--tau=')) o.tau = parseFloat(a.slice(6));
    else if (a.startsWith('--u0=')) o.u0 = parseFloat(a.slice(5));
    else if (a.startsWith('--q=')) o.q = parseInt(a.slice(4));
    else if (a.startsWith('--rb=')) o.rb = parseInt(a.slice(5));
    else if (a.startsWith('--refine=')) o.refine = a.slice(9);
    else if (a.startsWith('--boxfrac=')) o.boxfrac = parseFloat(a.slice(10));
    else if (a.startsWith('--scenario=')) o.scenario = a.slice(11);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--warmup=')) o.warmup = parseInt(a.slice(9));
    else if (a.startsWith('--reps=')) o.reps = parseInt(a.slice(7));
    else if (a.startsWith('--configs=')) o.configs = a.slice(10).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

function urlFor(o, c) {
  const p = new URLSearchParams({ scenario: o.scenario, n: o.n, q: o.q, live: '0' });
  if (o.scenario === 'beltrami') { p.set('tau', o.tau); p.set('u0', o.u0); }
  if (c.levels > 1) {
    p.set('levels', c.levels); p.set('rb', o.rb); p.set('refine', o.refine);
    if (['box', 'bar', 'slab'].includes(o.refine)) p.set('boxfrac', o.boxfrac);
    if (c.iface) p.set('interface', c.iface);
    if (c.explin != null) p.set('explin', c.explin);
    if (c.orphans != null) p.set('orphans', c.orphans);
  }
  return `${o.baseUrl}/index-3d.html?${p}${o.extra ? `&${o.extra}` : ''}`;
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const f2 = (x) => (x == null || !isFinite(x) ? '-' : x.toFixed(2));

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = CONFIGS.filter(c => !o.configs || o.configs.includes(c.name));
  if (!configs.length) { console.error('no configs selected'); process.exit(2); }

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = await attachPageWatch(client, {
    onError: (e) => console.error(`    !! [${e.kind}] ${e.text.split('\n')[0]}`),
  });

  const rows = [];
  try {
    for (const c of configs) {
      const url = urlFor(o, c);
      console.log(`\n=== ${c.name} (${url})`);
      try {
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, 'window.__D3', 60000);
        await assertPageHealthy(Runtime, watch, c.name);
        const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
        console.log(`    N=${p.N} Q${p.Q}` + (p.amr ? `  RB=${p.rb} ${p.activeSlots}/${p.blocks} tiles` : '  dense'));
        // Warm up OUTSIDE the timed region: pipeline creation, first-touch
        // allocation and the driver's own clocks all settle here, and none
        // of them are what this measures.
        await evalOrThrow(Runtime, `window.__D3.debugStepSync(${o.warmup})`, (o.timeout + 30) * 1000, 'warmup');
        const ms = [];
        for (let r = 0; r < o.reps; r++) {
          // debugStepSync awaits onSubmittedWorkDone per chunk, so the wall
          // clock around it is GPU time plus one submit per chunk -- not a
          // queue that has run ahead. Same instrument for every row.
          const t = await evalOrThrow(Runtime,
            `(async () => { const t0 = performance.now();`
            + ` await window.__D3.debugStepSync(${o.steps});`
            + ` return performance.now() - t0; })()`,
            (o.timeout + 30) * 1000, 'timed run');
          ms.push(t / o.steps);
        }
        ms.sort((a, b) => a - b);
        // THE MINIMUM, NOT THE MEDIAN, and this is not cherry-picking. The
        // only noise source here is contention -- the user's own browser and
        // compositor sharing the GPU -- and contention can only ADD time. So
        // every rep is the true cost plus a non-negative, uncorrelated
        // amount, and the minimum is the least-contaminated estimator of the
        // true cost, while the median tracks how busy the machine happened to
        // be. Measured on a busy desktop the two disagree by tens of percent
        // and only the minimum is stable across runs. The full sample is
        // printed so that claim stays checkable.
        const best = ms[0];
        const med = ms[Math.floor(ms.length / 2)];
        const spread = (ms[ms.length - 1] - ms[0]) / best;
        console.log(`    ${f2(best)} ms/macro-step (best of ${o.reps}; median ${f2(med)})`
          + `   spread ${(spread * 100).toFixed(1)}%   [${ms.map(f2).join(' ')}]`);
        rows.push({ name: c.name, med: best, median: med, spread, tiles: p.activeSlots || 0, amr: !!p.amr });
      } catch (err) {
        console.error(`    FAILED: ${err.message}`);
        rows.push({ name: c.name, error: err.message });
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(78));
  console.log(`SUMMARY  ${o.scenario} N=${o.n} Q${o.q} refine=${o.refine} rb=${o.rb}, ${o.steps} steps x ${o.reps} reps`);
  console.log('='.repeat(78));
  console.log(pad('config', 16) + padL('ms/step', 11) + padL('median', 10) + padL('spread', 9) + padL('vs interp', 12) + padL('tiles', 9));
  console.log('-'.repeat(78));
  const base = rows.find(r => r.name === 'interp' && !r.error);
  for (const r of rows) {
    if (r.error) { console.log(pad(r.name, 16) + padL('ERROR', 11)); continue; }
    const rel = base ? `${((r.med / base.med - 1) * 100).toFixed(1)}%` : '-';
    console.log(pad(r.name, 16) + padL(f2(r.med), 11) + padL(f2(r.median), 10)
      + padL(`${(r.spread * 100).toFixed(1)}%`, 9) + padL(rel, 12) + padL(r.amr ? r.tiles : '-', 9));
  }
  console.log('\nms/step is the BEST of the reps, not the median -- contention only adds time,');
  console.log('so the minimum is the least-contaminated estimate. The median column shows how');
  console.log('busy the machine was; a large gap between the two means re-run on a quiet one');
  console.log('before believing a small difference.');
  if (rows.some(r => r.error)) process.exit(1);
}

main();
