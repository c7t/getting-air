#!/usr/bin/env node
// DOES THE PLATE TUMBLE, AND WHEN? plans/3D.md M6.5.
//
// Trajectory only -- no rendering, no ffmpeg. The solver is the entire cost of
// this question and a picture is not what it needs, so this steps the page and
// reads the body, nothing else.
//
// WHY IT EXISTS SEPARATELY FROM tools/render-d3-movie.js, which also reports a
// trajectory: that one answers it ONCE, at the end of a clip, and the end of a
// run is exactly where this question cannot be answered.
//
//   `net` IS A WANDERING QUANTITY UNDER FLUTTER. It is the MAGNITUDE of the
//   vector integral INT omega dt, so a plate that rocks one way and back
//   returns it toward zero -- measured 0.120 at the end of a run whose
//   maximum was 0.306, on the same run. Reading the final value alone
//   therefore samples a quantity that is still moving.
//
//   TUMBLING IS `net` GROWING WITHOUT BOUND, which only the SERIES shows.
//   Flutter is `net` bounded while `arc` (INT |omega| dt, the arc length)
//   climbs linearly -- measured, the bound is 0.05..0.31 revolutions across
//   every configuration tried so far, from release angles differing eightfold.
//
// So this integrates omega on a fine cadence and prints the running pair, and
// its verdict reads the last half of the series rather than its last point.
//
// THE OTHER THING IT IS FOR IS SWEEPING. `--sweep=i_star=0.17,0.34` runs one
// leg per value in one Chrome, which is what turns "flutter is an attractor
// here" into a statement about which parameter moves it.
//
//   node tools/probe-d3-tumble.js
//   node tools/probe-d3-tumble.js --sweep=i_star=0.17,0.34,0.68
//   node tools/probe-d3-tumble.js --sweep=tilt=0.15,1.047 --steps=26000
//   node tools/probe-d3-tumble.js --extra='span=2&re=1100' --sweep=i_star=0.34

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');
const { mergeQuery } = require('./lib/url-query');

const REPO_ROOT = path.join(__dirname, '..');

// THE CONFIGURATION THE SWEEP RIDES ON, and every part of it is a measured
// choice rather than a default (plans/3D.md M6.5):
//   span 2      the 2:1 plate, which glides where the square one falls
//               straight down -- the closest to tumbling of the shapes tried.
//   re 1100     the target. It survives 26000 steps at ?levels=2, which is
//               M8.0's "AMR is the stability mechanism" holding; the dense
//               run at this tau is not something to rely on.
//   tilt 1.047  released 30 degrees off vertical, which produces the swoop
//               IMMEDIATELY (net/arc = 1.00 over the first 2000 steps) and
//               so cuts the onset time a shallow release spends oscillating.
const BASE = 'scenario=card&n=32&live=0&span=2&re=1100&tilt=1.047'
  + '&levels=2&rb=4&refine=body&dynamic=1&manageEvery=8&margin=3';

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  steps: 26000, every: 100, report: 2000, sweep: null, legs: null, extra: '',
  keepOpen: false, quiet: false,
};

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--every=')) o.every = parseInt(a.slice(8));
    else if (a.startsWith('--report=')) o.report = parseInt(a.slice(9));
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--sweep=')) {
      const v = a.slice(8);
      const eq = v.indexOf('=');
      if (eq < 0) { console.error('--sweep= wants name=v1,v2,...'); process.exit(2); }
      o.sweep = { name: v.slice(0, eq), values: v.slice(eq + 1).split(',') };
    } else if (a.startsWith('--legs=')) {
      // One LEG per '|'-separated query fragment. --sweep varies one
      // parameter's value; this varies whatever each leg wants, which is what
      // a comparison between two different mechanisms needs (`bounceback=1`
      // against `levels=3` are not two values of one knob).
      o.legs = a.slice(7).split('|').filter(Boolean);
    } else if (a === '--quiet') o.quiet = true;
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, what, timeoutMs = 900000) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// Least-squares slope of `y` against `x`, for "is net still growing at the end
// of the run". A bounded oscillation gives ~0 over a window covering several
// of its periods; a tumble gives its revolution rate.
function slope(x, y) {
  const n = x.length;
  if (n < 3) return 0;
  const mx = x.reduce((a, v) => a + v, 0) / n, my = y.reduce((a, v) => a + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) ** 2; }
  return den ? num / den : 0;
}

async function runLeg(Runtime, watch, Page, o, url, label) {
  await navigateTo(Page, url);
  await waitForGlobal(Runtime, 'window.__D3', 60000);
  await assertPageHealthy(Runtime, watch, label);
  const p = await ev(Runtime, 'window.__D3.getParams()', 'getParams');
  console.log(`\n=== ${label}`);
  console.log(`    ${p.NX}x${p.NY}x${p.NZ}  Re=${p.re}  tau=${p.tau.toFixed(5)}`
    + `  span=${p.span}  i*=${p.i_star}  rho_b=${p.rho_b.toFixed(2)}  gEff=${p.gEff.toExponential(2)}`);
  if (!o.quiet) console.log('    step     fell  net rev  arc rev  net/arc   |omega|     vx      vy      vz');

  let prev = await ev(Runtime, 'window.__D3.readBody()', 'readBody');
  const O = [0, 0, 0], ts = [], nets = [];
  const x0 = prev.dx;
  let arc = 0, netMax = 0, stopped = null, last = prev, reached = 0, frozen = 0;
  for (let s = o.every; s <= o.steps; s += o.every) {
    await ev(Runtime, `window.__D3.debugStepSync(${o.every})`, 'debugStepSync');
    const b = await ev(Runtime, 'window.__D3.readBody()', 'readBody');
    if (!Number.isFinite(b.cx + b.qw + b.wx + b.vx)) { stopped = `body state non-finite at step ${s}`; break; }
    // A FROZEN BODY INTEGRATES INTO A PERFECT TUMBLE, which is how this
    // instrument lied once and must not again. When a run dies the page stops
    // advancing but debugStepSync keeps returning, so every readBody comes
    // back IDENTICAL -- and a constant non-zero omega integrated forever is
    // `net` growing linearly without bound, which is exactly the signature
    // this tool calls tumbling. Observed 2026-09-12: a bounce-back leg froze
    // at |v| = 0.2 (the body's own v_max clamp, i.e. it had blown up) with
    // `fell` stuck at 119 cells, and was duly reported as 10.03 revolutions
    // with net/arc climbing to 0.94.
    //
    // The tell is that NOTHING changes, position included. A real tumble
    // moves the plate; a dead one does not.
    if (b.dx === prev.dx && b.cx === prev.cx && b.qw === prev.qw && b.wx === prev.wx) {
      frozen += o.every;
      if (frozen >= 5 * o.every) { stopped = `body state FROZEN for ${frozen} steps at ${s} -- the run stopped advancing`; break; }
    } else { frozen = 0; }
    // The velocity clamp is the other end of the same story: a body pinned at
    // its own v_max has already lost, whatever the trajectory does afterwards.
    if (b.v_max && Math.hypot(b.vx, b.vy, b.vz) >= 0.999 * b.v_max) {
      stopped = `|v| hit the body's v_max clamp (${b.v_max}) at step ${s} -- it blew up`;
      break;
    }
    O[0] += 0.5 * (prev.wx + b.wx) * o.every;
    O[1] += 0.5 * (prev.wy + b.wy) * o.every;
    O[2] += 0.5 * (prev.wz + b.wz) * o.every;
    arc += 0.5 * (Math.hypot(prev.wx, prev.wy, prev.wz) + Math.hypot(b.wx, b.wy, b.wz)) * o.every;
    const net = Math.hypot(...O) / (2 * Math.PI);
    ts.push(s); nets.push(net);
    netMax = Math.max(netMax, net);
    prev = b; last = b; reached = s;
    if (!o.quiet && (s % o.report === 0 || s === o.steps)) {
      console.log(`  ${String(s).padStart(6)} ${(b.dx - x0).toFixed(0).padStart(8)}`
        + ` ${net.toFixed(3).padStart(8)} ${(arc / (2 * Math.PI)).toFixed(3).padStart(8)}`
        + ` ${(net / Math.max(arc / (2 * Math.PI), 1e-9)).toFixed(2).padStart(8)}`
        + ` ${Math.hypot(b.wx, b.wy, b.wz).toExponential(2).padStart(9)}`
        + ` ${b.vx.toFixed(3).padStart(6)} ${b.vy.toFixed(3).padStart(6)} ${b.vz.toFixed(3).padStart(6)}`);
    }
  }
  // THE PAGE CAN FAIL WITHOUT THE BODY GOING NON-FINITE. M5.4a latches
  // `error: out of pool slots` into #status and stops advancing -- and
  // debugStepSync keeps returning while it does, so a leg would report a
  // clean trajectory for a run that stopped solving. Checked at the END of
  // every leg and not only at navigation, which is where it cannot yet have
  // happened.
  if (!stopped) {
    try { await assertPageHealthy(Runtime, watch, label); }
    catch (e) { stopped = e.message.split('\n')[0]; }
  }
  const half = Math.floor(ts.length / 2);
  // Revolutions per 1000 steps over the LAST HALF -- the growth rate, which
  // is what separates a tumble from a bounded rock. Zero for flutter whatever
  // its amplitude.
  const growth = 1000 * slope(ts.slice(half), nets.slice(half));
  const netEnd = nets.length ? nets[nets.length - 1] : 0;
  const arcRev = arc / (2 * Math.PI);
  // THE THRESHOLDS ARE THE MEASURED FLUTTER BAND, not taste: across every
  // configuration tried (release angles differing eightfold, four Reynolds
  // numbers, two spans) `net` stayed inside 0.05..0.31 revolutions and its
  // last-half growth was indistinguishable from zero. A plate cannot rock
  // through three quarters of a NET revolution and still be rocking.
  // A LEG THAT DID NOT SURVIVE HAS NO VERDICT. Whatever it accumulated before
  // it died is not evidence about tumbling.
  const verdict = stopped ? 'INVALID -- run did not survive'
    : (netEnd > 0.75 && growth > 0.005) ? 'TUMBLING'
    : netMax > 0.75 ? 'TUMBLING (transient -- net exceeded 3/4 turn)'
      : arcRev > 0.25 ? 'fluttering' : 'falling flat';
  return { label, netEnd, netMax, arcRev, growth, reached, stopped, verdict,
           fell: last.dx - x0, v: [last.vx, last.vy, last.vz],
           iStar: p.i_star, rhoB: p.rho_b };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const legs = o.legs ? o.legs.map(q => ({ label: q, q }))
    : o.sweep
      ? o.sweep.values.map(v => ({ label: `${o.sweep.name}=${v}`, q: `${o.sweep.name}=${v}` }))
      : [{ label: o.extra || 'base', q: '' }];

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
    for (const leg of legs) {
      const url = `${o.baseUrl}/index-3d.html?${mergeQuery(BASE, leg.q, o.extra)}`;
      try {
        rows.push(await runLeg(Runtime, watch, Page, o, url, leg.label));
      } catch (err) {
        console.error(`    FAILED: ${err.message}`);
        rows.push({ label: leg.label, error: err.message });
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(104));
  console.log(`SUMMARY  ${o.steps} steps, omega integrated every ${o.every}`);
  console.log('='.repeat(104));
  console.log('leg'.padEnd(16) + 'i*'.padStart(6) + 'rho_b'.padStart(8) + 'fell'.padStart(8)
    + 'net end'.padStart(9) + 'net max'.padStart(9) + 'arc'.padStart(8)
    + 'growth/1k'.padStart(11) + '  verdict');
  console.log('-'.repeat(104));
  for (const r of rows) {
    if (r.error) { console.log(r.label.padEnd(16) + '  ERROR: ' + r.error.split('\n')[0]); continue; }
    console.log(r.label.padEnd(16) + String(r.iStar).padStart(6) + r.rhoB.toFixed(1).padStart(8)
      + r.fell.toFixed(0).padStart(8) + r.netEnd.toFixed(3).padStart(9)
      + r.netMax.toFixed(3).padStart(9) + r.arcRev.toFixed(2).padStart(8)
      + r.growth.toFixed(4).padStart(11) + '  ' + r.verdict
      + (r.stopped ? `  [${r.stopped}]` : ''));
  }
  console.log('\nnet = |INT omega dt| / 2pi, arc = INT |omega| dt / 2pi. Flutter is net BOUNDED');
  console.log('while arc climbs; tumbling is net growing. Read `net max` and `growth`, not `net end`');
  console.log('-- under flutter net wanders back toward zero and its final value is a sample of a');
  console.log('quantity that is still moving (measured 0.120 against a 0.306 max on one run).');
}

main().catch((e) => { console.error('\n' + (e.stack || e.message)); process.exit(1); });
