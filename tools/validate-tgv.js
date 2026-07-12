#!/usr/bin/env node
// Physical-observable regression check for the Taylor-Green-vortex
// solver(s): drives index-tgv.html (dense) or index-tgv-amr.html (AMR,
// when a case sets `levels`) through a CDP-attached, WebGPU-capable
// Chrome, runs every case in benchmarks/tgv.json, and diffs the velocity
// field (plus a fitted decay rate) against the exact analytic decaying-
// vortex solution. Same "run it, compare to a reference, print PASS/FAIL"
// shape as tools/validate-channel.js, but every TGV parameter (N, u0,
// tau, and for AMR, levels) is page-load-time on BOTH pages -- there's no
// live-settable equivalent of channel flow's setRe -- so this navigates
// once per case, not once per group; see tools/lib/tgv-metrics.js for the
// shared analysis, used identically by tools/validate-all.js so the two
// don't drift apart.
//
// Requires a WebGPU-capable Chrome already running with
// --remote-debugging-port and a tgv page loaded (see
// .claude/skills/webgpu-verify).
//
// Usage:
//   node tools/validate-tgv.js [--baseUrl=http://localhost:8000] [--port=9333] [--timeout=120]
//   node tools/validate-tgv.js --names=dense-N32,dense-N64

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { evalExpr, runCase } = require('./lib/tgv-metrics');

function parseArgs(argv) {
  const opts = { baseUrl: 'http://localhost:8000', port: 9333, timeout: 120, names: null };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
    else if (a.startsWith('--names=')) opts.names = a.slice(8).split(',');
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

function urlForCase(baseUrl, c) {
  if (c.levels) {
    // main-tgv-amr.js's ?res= is log2(N), matching every other AMR page's
    // convention -- benchmarks/tgv.json's own `N` field is always the
    // literal domain size, same "convert at the URL-construction boundary,
    // not in the benchmark data" approach as tools/validate-channel.js.
    return `${baseUrl}/index-tgv-amr.html?res=${Math.log2(c.N)}&u0=${c.u0}&tau=${c.tau}&levels=${c.levels}`;
  }
  return `${baseUrl}/index-tgv.html?res=${c.N}&u0=${c.u0}&tau=${c.tau}`;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const benchPath = path.join(__dirname, '..', 'benchmarks', 'tgv.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  let cases = bench.cases;
  if (opts.names) cases = cases.filter(c => opts.names.includes(c.name));
  if (cases.length === 0) { console.error('No matching cases in benchmarks/tgv.json'); process.exit(1); }

  const client = await connect(opts.port);
  const { Runtime, Page } = client;

  const results = [];
  for (const c of cases) {
    const url = urlForCase(opts.baseUrl, c);
    console.log(`\n=== ${c.name} (${url}) ===`);
    await navigateTo(Page, url);
    await waitForCYL(Runtime, 15000);
    await evalExpr(Runtime, `window.__CYL.setLive(false)`);
    results.push({ name: c.name, ...(await runCase(Runtime, { timeout: opts.timeout }, c, s => console.log('    ' + s))) });
  }
  await client.close();

  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log('\n' + pad('case', 16) + pad('N', 6) + pad('fieldL2rel', 14) + pad('rateRelErr', 14) + pad('result', 8));
  let allPass = true;
  for (const r of results) {
    const pass = r.fieldCheck.pass && r.rateCheck.pass;
    console.log(pad(r.name, 16) + pad(r.N, 6) + pad(r.maxL2rel.toExponential(3), 14) + pad(r.rateRelErr.toExponential(3), 14) + pad(pass ? 'PASS' : 'FAIL', 8));
    if (!pass) allPass = false;
  }
  console.log(allPass ? '\nAll cases within tolerance.' : '\nSome cases OUT OF TOLERANCE -- see above.');
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
