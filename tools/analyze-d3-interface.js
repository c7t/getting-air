#!/usr/bin/env node
// Diagnostic for plans/3D.md M3's OPEN issue: a partially-refined 3D AMR run
// grows an error at the coarse/fine seam. This is the measurement that
// benchmarks/d3.json's amr_interface_note quotes, made reproducible -- and
// with the one control that tells the two candidate causes apart.
//
// WHAT IT ANSWERS, and why the second half is the point.
//
//   1. The seam profile. Velocity error against the analytic Beltrami
//      solution, binned by signed Chebyshev distance in coarse cells to the
//      interface (negative inside the refined region, positive outside, 0
//      touching it). A seam defect peaks at |d| = 0 and decays; a global
//      accuracy problem is flat. `refine=all` and the dense run have no
//      interface at all, so their whole domain lands in the far bucket and
//      they are the flat controls.
//
//   2. CONSERVATION. Total mass and total momentum over the coarse grid,
//      where a refined block contributes its RESTRICTED fine moments -- an
//      exact arithmetic mean of rho and an exact mass-weighted mean of u, so
//      the sum IS the hybrid system's total. beltrami is periodic with no
//      body and no body force, so each level ALONE conserves both exactly:
//      streaming permutes populations, collision preserves the first two
//      moments. Any drift is the interface and nothing else.
//
//      This is the discriminator. plans/3D.md names refluxing as the fix,
//      and refluxing restores CONSERVATION and nothing else. If mass and
//      momentum hold at the readback floor while the seam error grows, the
//      interface is consistent-but-inaccurate and refluxing would be a
//      large change aimed at the wrong defect. Measure before building it.
//
// Reports; it does not PASS/FAIL. There is no literature value for how fast
// a seam should degrade -- the controls are what the box run is read
// against, which is why they are run in the same invocation rather than
// remembered from another session.
//
// Owns the whole lifecycle (HTTPS dev server + a dedicated debug-port Chrome
// if neither is up, one tab reused via Page.navigate), like
// tools/validate-3d.js -- so it is one command and only one WebGPU context
// is ever alive.
//
//   node tools/analyze-d3-interface.js
//   node tools/analyze-d3-interface.js --n=48 --checkpoints=1,2,4,8,16,32,64
//   node tools/analyze-d3-interface.js --configs=box --boxfrac=0.5
//   node tools/analyze-d3-interface.js --extra=q=27

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

// The default checkpoints are STEPS, not decay times. The seam injection is
// per-step and shows up long before the flow has decayed measurably, and a
// log ladder is what separates "linear in t" from "sqrt(t)" -- which is the
// difference between a constant source and a diffusing one.
const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333, n: 48, tau: 0.8, u0: 0.04, q: 19, rb: 4,
  boxfrac: 0.5, checkpoints: [1, 2, 4, 8, 16, 32, 64, 128, 256], td: null, configs: null, extra: '',
  timeout: 600, keepOpen: false,
};

const CONFIGS = [
  // The refined box in the middle of an exactly-known flow -- the case with
  // the open issue.
  { name: 'box', levels: 2, refine: 'box' },
  // No coarse/fine interface anywhere: the pool, fine solver, interp and
  // average all still run, so this isolates the INTERFACE from everything
  // else the AMR path does.
  // A refined slab spanning y and z: the seam is two FLAT faces, no edge
  // and no corner. The control that separates "the flux correction is
  // wrong" from "the seam has convex corners the lattice cannot tile".
  { name: 'slab', levels: 2, refine: 'slab' },
  { name: 'all', levels: 2, refine: 'all' },
  // No fine level at all: the coarse solver's own discretization error, the
  // floor everything else is measured against.
  { name: 'dense', levels: 1 },
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
    else if (a.startsWith('--boxfrac=')) o.boxfrac = parseFloat(a.slice(10));
    else if (a.startsWith('--checkpoints=')) o.checkpoints = a.slice(14).split(',').map(Number);
    // Checkpoints in DECAY TIMES rather than steps. The step ladder is for
    // reading the per-step injection rate; this is for comparing runs at
    // different N, where the same number of steps is not the same physics.
    else if (a.startsWith('--td=')) o.td = a.slice(5).split(',').map(Number);
    else if (a.startsWith('--configs=')) o.configs = a.slice(10).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

function urlFor(o, c) {
  const p = new URLSearchParams({ scenario: 'beltrami', n: o.n, tau: o.tau, u0: o.u0, q: o.q, live: '0' });
  if (c.levels > 1) { p.set('levels', c.levels); p.set('rb', o.rb); p.set('refine', c.refine); }
  if (c.refine === 'box' || c.refine === 'slab') p.set('boxfrac', o.boxfrac);
  return `${o.baseUrl}/index-3d.html?${p}${o.extra ? `&${o.extra}` : ''}`;
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const e2 = (x) => (x == null || !isFinite(x) ? '   -    ' : x.toExponential(2));
const padL = (s, n) => String(s).padStart(n);

// L2-relative error of one bucket, and the bucket's share of the domain.
function bucketL2(b) { return Math.sqrt(b.err2 / Math.max(b.ref2, 1e-30)); }

async function runConfig(Runtime, o, c) {
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  console.log(`    N=${p.N} tau=${p.tau} Q${p.Q} nu=${p.nu.toFixed(4)} td=${p.td.toFixed(0)}`
    + (p.amr ? `  [RB=${p.rb} ${p.activeSlots}/${p.blocks} tiles refined = ${(100 * p.refinedFraction).toFixed(0)}%]` : '  [dense]'));

  const samples = [];
  let prev = 0;
  const cps = o.td ? o.td.map(m => Math.round(m * p.td)) : o.checkpoints;
  for (const t of [0, ...cps]) {
    if (t > prev) {
      await evalOrThrow(Runtime, `window.__D3.debugStepSync(${t - prev})`, (o.timeout + 30) * 1000, 'debugStepSync');
      prev = t;
    }
    const d = await evalOrThrow(Runtime, `window.__D3.readInterfaceDiag(${t})`, 180000, 'readInterfaceDiag');
    samples.push(d);
  }
  return { params: p, samples };
}

// Total mass/momentum are reported as DRIFT from this run's own t=0 sample,
// because the absolute values carry the readback's f32 quantization and the
// initial condition's own rounding, neither of which is the solver's doing.
function driftTable(res) {
  const z = res.samples[0];
  const rows = res.samples.map(s => ({
    t: s.t,
    mass: s.mass - z.mass,
    mom: s.mom.map((v, a) => v - z.mom[a]),
  }));
  return rows;
}

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

  const results = [];
  try {
    for (const c of configs) {
      const url = urlFor(o, c);
      console.log(`\n=== ${c.name} (${url})`);
      await navigateTo(Page, url);
      await waitForGlobal(Runtime, 'window.__D3', 60000);
      await assertPageHealthy(Runtime, watch, c.name);
      const res = await runConfig(Runtime, o, c);
      await assertPageHealthy(Runtime, watch, c.name);
      results.push({ name: c.name, ...res });
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  // --- 1. conservation -----------------------------------------------------
  console.log('\n\n=== CONSERVATION: drift from t=0, over the whole coarse grid');
  console.log('    sum(rho-1) and sum(rho*u). Both are exactly conserved by either level');
  console.log('    alone on this periodic, force-free scenario, so any drift is the');
  console.log('    interface. Compare against the dense and refine=all controls, which');
  console.log('    have no interface and give the f32 readback floor.\n');
  console.log(`  ${padL('t', 6)}  ${configs.map(c => padL(c.name, 40)).join('')}`);
  console.log(`  ${padL('', 6)}  ${configs.map(() => `${padL('d mass', 10)}${padL('d momX', 10)}${padL('d momY', 10)}${padL('d momZ', 10)}`).join('')}`);
  const tables = results.map(driftTable);
  for (let i = 0; i < tables[0].length; i++) {
    let line = `  ${padL(tables[0][i].t, 6)}  `;
    for (const tb of tables) {
      const r = tb[i];
      line += padL(e2(r.mass), 10) + r.mom.map(v => padL(e2(v), 10)).join('');
    }
    console.log(line);
  }

  // --- 2. seam profile -----------------------------------------------------
  console.log('\n\n=== SEAM PROFILE: velocity L2rel vs the analytic solution, by distance');
  console.log('    to the coarse/fine interface (coarse cells; negative = inside the');
  console.log('    refined region, 0 = touching the seam, +8 = 8 or more away, and the');
  console.log('    whole domain for a run with no interface).\n');
  for (const r of results) {
    console.log(`  --- ${r.name}`);
    const ds = r.samples[r.samples.length - 1].buckets.map(b => b.d);
    console.log(`  ${padL('t', 6)}  ${padL('all', 9)} ${ds.map(d => padL(d > 0 ? `+${d}` : d, 9)).join(' ')}`);
    for (const s of r.samples) {
      if (!s.hasRef) continue;
      const tot = s.buckets.reduce((a, b) => ({ err2: a.err2 + b.err2, ref2: a.ref2 + b.ref2 }), { err2: 0, ref2: 0 });
      const by = new Map(s.buckets.map(b => [b.d, b]));
      console.log(`  ${padL(s.t, 6)}  ${padL(e2(bucketL2(tot)), 9)} `
        + ds.map(d => padL(by.has(d) ? e2(bucketL2(by.get(d))) : '-', 9)).join(' '));
    }
    const n0 = r.samples[0].buckets.find(b => b.d === 0);
    if (n0) console.log(`      (cells per bucket: ${r.samples[0].buckets.map(b => `${b.d}:${b.n}`).join(' ')})`);
    console.log('');
  }
}

main().catch(e => { console.error(e); process.exit(1); });
