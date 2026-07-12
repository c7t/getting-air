#!/usr/bin/env node
// Physical-observable regression check for the channel-flow solver(s):
// drives index-channel.html (or --amr for index-channel-amr.html) through a
// CDP-attached, WebGPU-capable Chrome, runs every case in
// benchmarks/channel.json, and diffs the steady-state u(y) profile against
// the exact analytic solution. Same "run it, compare to a reference, print
// PASS/FAIL" idea as tools/validate-cylinder.js, but against a closed-form
// target instead of a literature Cd/St band -- see tools/lib/channel-
// metrics.js for the shared analysis, used identically by
// tools/validate-all.js so the two don't drift apart.
//
// Unlike the cylinder harness (where `res` is fixed per page load and only
// `re` varies within benchmarks/cylinder.json), channel flow's resolution
// is itself part of what's being validated, so a case's `res`/`mode` are
// baked into the URL -- this groups cases by (mode,res) and navigates once
// per group, sweeping only `re` within a group via window.__CYL.setRe (no
// page reload needed for that part).
//
// Requires a WebGPU-capable Chrome already running with
// --remote-debugging-port and a channel page loaded (see
// .claude/skills/webgpu-verify).
//
// Usage:
//   node tools/validate-channel.js [--url=http://localhost:8000/index-channel.html]
//     [--port=9333] [--mode=poiseuille,couette] [--res=16,32,64] [--timeout=300]
//   node tools/validate-channel.js --amr --levels=2

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { evalExpr, runCase } = require('./lib/channel-metrics');

function parseArgs(argv) {
  const opts = {
    baseUrl: 'http://localhost:8000',
    port: 9333,
    mode: null,
    res: null,
    timeout: 300,
    amr: false,
    levels: 2,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--mode=')) opts.mode = a.slice(7).split(',');
    else if (a.startsWith('--res=')) opts.res = a.slice(6).split(',').map(Number);
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
    else if (a === '--amr') opts.amr = true;
    else if (a.startsWith('--levels=')) opts.levels = parseInt(a.slice(9));
  }
  return opts;
}

async function connect(port) {
  const client = await CDP({ port });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  return client;
}

async function navigateTo(Page, url) {
  await Page.navigate({ url });
  await Page.loadEventFired();
}

async function waitForCYL(Runtime, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const r = await evalExpr(Runtime, `typeof window.__CYL !== 'undefined'`);
    if (!r.exceptionDetails && r.result.value === true) return;
    await new Promise(res => setTimeout(res, 300));
  }
  throw new Error('window.__CYL never became available');
}

// Groups cases by (mode,res) so each group needs exactly one navigation --
// see this file's own header for why (unlike the cylinder harness, res is
// page-load-time here, not swept live).
function groupCases(cases) {
  const groups = new Map();
  for (const c of cases) {
    const key = `${c.mode}|${c.res}`;
    if (!groups.has(key)) groups.set(key, { mode: c.mode, res: c.res, cases: [] });
    groups.get(key).cases.push(c);
  }
  return Array.from(groups.values());
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const benchPath = path.join(__dirname, '..', 'benchmarks', 'channel.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  let cases = bench.cases;
  if (opts.mode) cases = cases.filter(c => opts.mode.includes(c.mode));
  if (opts.res) cases = cases.filter(c => opts.res.includes(c.res));
  if (cases.length === 0) { console.error('No matching cases in benchmarks/channel.json'); process.exit(1); }

  const client = await connect(opts.port);
  const { Runtime, Page } = client;

  const page = opts.amr ? 'index-channel-amr.html' : 'index-channel.html';
  const results = [];
  for (const group of groupCases(cases)) {
    const levelsParam = opts.amr ? `&levels=${opts.levels}` : '';
    // main-channel.js's `?res=` is H directly; main-channel-amr.js's is
    // log2(H), matching every other AMR page's convention -- see
    // tools/validate-all.js's runChannelPhysics for the identical note.
    const resParam = opts.amr ? Math.log2(group.res) : group.res;
    const url = `${opts.baseUrl}/${page}?mode=${group.mode}&res=${resParam}${levelsParam}`;
    console.log(`\n=== ${group.mode} res=${group.res}${opts.amr ? ` (AMR levels=${opts.levels})` : ''} (${url}) ===`);
    await navigateTo(Page, url);
    await waitForCYL(Runtime, 15000);
    await evalExpr(Runtime, `window.__CYL.setLive(false)`);
    for (const c of group.cases) {
      results.push(await runCase(Runtime, { timeout: opts.timeout }, c, s => console.log('    ' + s)));
    }
  }
  await client.close();

  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log('\n' + pad('mode', 12) + pad('res', 6) + pad('Re', 6) + pad('converged', 11) + pad('L2rel (target ± tol)', 26));
  let allPass = true;
  for (const r of results) {
    const l2Str = `${r.l2rel.toExponential(3)} (0 ± ${r.l2.tol}) ${r.l2.pass ? 'PASS' : 'FAIL'}`;
    console.log(pad(r.mode, 12) + pad(r.res, 6) + pad(r.re, 6) + pad(r.converged, 11) + pad(l2Str, 26));
    if (!r.l2.pass || !r.converged) allPass = false;
  }
  console.log(allPass ? '\nAll cases within tolerance.' : '\nSome cases OUT OF TOLERANCE or did not converge -- see above.');
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
