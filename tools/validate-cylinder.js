#!/usr/bin/env node
// Physical-observable regression check for the LBM solver(s): drives
// index-cylinder.html (see main-cylinder.js) through a CDP-attached,
// WebGPU-capable Chrome, runs a pinned circular cylinder in uniform
// crossflow at each Reynolds number in benchmarks/cylinder.json, computes
// the time-averaged drag coefficient and (where shedding occurs) the
// Strouhal number, and diffs both against the literature values in that
// file. This is the same "run it, compare to a reference, print PASS/FAIL"
// idea as tools/amr-diff.js's snapshot regression, but at the level of Cd/St
// instead of raw field values -- it validates that the physics is *correct*,
// not just that the AMR build matches a prior run of itself.
//
// Requires a WebGPU-capable Chrome already running with
// --remote-debugging-port and index-cylinder.html loaded (see
// .claude/skills/webgpu-verify).
//
// Usage:
//   node tools/validate-cylinder.js [--url=http://localhost:8000/index-cylinder.html]
//     [--port=9333] [--re=20,40,100,200] [--timeout=300]

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');

function parseArgs(argv) {
  const opts = {
    url: 'http://localhost:8000/index-cylinder.html',
    port: 9333,
    re: null,
    timeout: 300,
  };
  for (const a of argv) {
    if (a.startsWith('--url=')) opts.url = a.slice(6);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--re=')) opts.re = a.slice(5).split(',').map(Number);
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
  }
  return opts;
}

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

async function connect(port) {
  const client = await CDP({ port });
  const { Runtime } = client;
  await Runtime.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  return client;
}

// Rounds a step count up to a multiple of STEPS_PER_FRAME (64, hardcoded in
// main-cylinder.js) so debugRunAndCollect's per-block sampling lands cleanly.
function roundSteps(n) {
  return Math.ceil(n / 64) * 64;
}

// Transient + measurement window sized from the physical parameters, not a
// fixed step count -- a slower freestream or larger cylinder needs more
// steps to reach the same number of flow-through times / shedding cycles.
function computeWindow(caseEntry, D, U0) {
  const transient = roundSteps(40 * D / U0);
  const measurement = caseEntry.st
    ? roundSteps(15 * (D / (caseEntry.st * U0))) // ~15 shedding periods
    : roundSteps(5 * D / U0);                     // steady case: just confirm convergence
  return { transient, measurement, total: transient + measurement };
}

// Cd_mean over the post-transient window, plus a Strouhal estimate from
// Cl(t)'s zero crossings (mean-subtracted, linearly interpolated crossing
// time for sub-sample precision). Returns st=null if fewer than 3 crossings
// are found (steady flow, or window too short).
function analyze(history, transientSteps, D, U0) {
  const win = history.filter(r => r[0] > transientSteps);
  if (win.length < 2) return { cdMean: NaN, st: null, samples: win.length };

  const cdMean = win.reduce((s, r) => s + r[3], 0) / win.length;

  const cl = win.map(r => r[4]);
  const clMean = cl.reduce((a, b) => a + b, 0) / cl.length;
  const centered = cl.map(v => v - clMean);

  const crossings = [];
  for (let i = 1; i < centered.length; i++) {
    if (centered[i - 1] < 0 && centered[i] >= 0) {
      const stepA = win[i - 1][0], stepB = win[i][0];
      const frac = -centered[i - 1] / (centered[i] - centered[i - 1]);
      crossings.push(stepA + frac * (stepB - stepA));
    }
  }
  let st = null;
  if (crossings.length >= 3) {
    const periods = [];
    for (let i = 1; i < crossings.length; i++) periods.push(crossings[i] - crossings[i - 1]);
    const meanPeriod = periods.reduce((a, b) => a + b, 0) / periods.length;
    const fs = 1 / meanPeriod;
    st = fs * D / U0;
  }
  return { cdMean, st, samples: win.length, crossings: crossings.length };
}

function checkTol(label, measured, target, tol) {
  if (target == null) return { pass: true, label, measured, target: null };
  const pass = measured != null && Math.abs(measured - target) <= tol;
  return { pass, label, measured, target, tol };
}

async function runCase(client, opts, caseEntry) {
  const { Runtime } = client;
  await evalExpr(Runtime, `window.__CYL.setRe(${caseEntry.re})`);
  await evalExpr(Runtime, `window.__CYL.reset()`);
  const params = (await evalExpr(Runtime, `window.__CYL.getParams()`)).result.value;
  const { transient, measurement, total } = computeWindow(caseEntry, params.D, params.U0);

  console.log(`Re=${caseEntry.re}: D=${params.D} U0=${params.U0} TAU=${params.TAU.toFixed(5)} -- running ${total} steps (${transient} transient + ${measurement} measurement)`);

  const r = await evalExpr(Runtime, `window.__CYL.debugRunAndCollect(${total})`, (opts.timeout + 30) * 1000);
  const history = r.result.value.history;

  const { cdMean, st, samples, crossings } = analyze(history, transient, params.D, params.U0);
  const cd = checkTol('Cd', cdMean, caseEntry.cd, caseEntry.cd_tol);
  const stChk = checkTol('St', st, caseEntry.st, caseEntry.st_tol);

  return { re: caseEntry.re, regime: caseEntry.regime, samples, crossings, cd, st: stChk, source: caseEntry.source };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const benchPath = path.join(__dirname, '..', 'benchmarks', 'cylinder.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  const cases = opts.re ? bench.cases.filter(c => opts.re.includes(c.re)) : bench.cases;
  if (cases.length === 0) { console.error('No matching cases in benchmarks/cylinder.json'); process.exit(1); }

  const client = await connect(opts.port);
  await new Promise(r => setTimeout(r, 1000));
  await evalExpr(client.Runtime, `window.__CYL.setLive(false)`);

  const results = [];
  for (const c of cases) {
    results.push(await runCase(client, opts, c));
  }
  await client.close();

  // Node's console.log %s doesn't support printf-style width modifiers
  // (%-6s prints literally, it doesn't left-pad) -- pad manually instead.
  // padEnd(n) with a trailing space guarantees a separator even when a
  // field (e.g. the "periodic vortex shedding..." regime string) overflows
  // its target column width.
  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log('\n' + pad('Re', 6) + pad('regime', 24) + pad('Cd (target ± tol)', 22) + pad('St (target ± tol)', 22));
  let allPass = true;
  for (const r of results) {
    const cdStr = `${r.cd.measured.toFixed(3)} (${r.cd.target?.toFixed(3) ?? '-'} ± ${r.cd.tol ?? '-'}) ${r.cd.pass ? 'PASS' : 'FAIL'}`;
    const stStr = r.st.target == null
      ? 'n/a'
      : `${r.st.measured?.toFixed(4) ?? 'NONE'} (${r.st.target.toFixed(3)} ± ${r.st.tol}) ${r.st.pass ? 'PASS' : 'FAIL'}`;
    console.log(pad(r.re, 6) + pad(r.regime, 24) + pad(cdStr, 22) + pad(stStr, 22));
    if (!r.cd.pass || !r.st.pass) allPass = false;
  }
  console.log(allPass ? '\nAll cases within tolerance.' : '\nSome cases OUT OF TOLERANCE -- see above.');
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
