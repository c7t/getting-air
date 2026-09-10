#!/usr/bin/env node
// Dense 3D solver validation (plans/3D.md M1): runs index-3d.html's two
// ANALYTIC gates -- square-duct Poiseuille flow and a decaying Beltrami
// (ABC) flow -- plus the 3D Taylor-Green vortex as a report, against
// benchmarks/d3.json.
//
// Owns the whole lifecycle (HTTPS dev server + a dedicated debug-port
// Chrome if neither is up, one tab reused via Page.navigate across every
// case), the same as tools/validate-all.js -- so it is one command, and so
// only one WebGPU context is ever alive at a time.
//
// WHY THESE TWO GATES rather than the "3D Taylor-Green at Re=1600" that
// plans/3D.md M1 originally named: see d3-scenarios.mjs's header. The short
// version is that the Re=1600 TGV is a DNS benchmark scored against
// digitized literature curves at 256^3+, which is a data-and-resolution
// problem, whereas what M1 needs to know is whether the solver realizes the
// viscosity and boundary conditions it claims. The Beltrami flow answers
// that exactly, in 3D, at every point and every time.
//
// Usage:
//   node tools/validate-3d.js
//   node tools/validate-3d.js --cases=duct-N48,bel-N48
//   node tools/validate-3d.js --skip=tgv
//   node tools/validate-3d.js --extra=q=27      # append to every case URL

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');
const { caseUrl, runDuctCase, runBeltramiCase, runTgvReport } = require('./lib/d3-metrics');

const REPO_ROOT = path.join(__dirname, '..');

function parseArgs(argv) {
  const opts = { baseUrl: 'https://localhost:4444', port: 9333, cases: null, skip: [], extra: '', timeout: 600, keepOpen: false };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--cases=')) opts.cases = a.slice(8).split(',');
    else if (a.startsWith('--skip=')) opts.skip = a.slice(7).split(',');
    else if (a.startsWith('--extra=')) opts.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') opts.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return opts;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const e3 = (x) => (x == null || !isFinite(x) ? '-' : x.toExponential(3));

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const bench = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'benchmarks', 'd3.json'), 'utf8'));

  const wanted = (name) => !opts.cases || opts.cases.includes(name);
  const groups = [
    { scenario: 'duct', run: runDuctCase, cases: bench.duct_cases, gate: true },
    { scenario: 'beltrami', run: runBeltramiCase, cases: bench.beltrami_cases, gate: true },
    { scenario: 'tgv', run: runTgvReport, cases: bench.tgv_cases, gate: false },
  ].filter(g => !opts.skip.includes(g.scenario))
   .map(g => ({ ...g, cases: g.cases.filter(c => wanted(c.name)) }))
   .filter(g => g.cases.length);

  if (!groups.length) { console.error('no cases selected'); process.exit(2); }

  const server = await ensureServer(opts.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(opts.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(opts.port) : await openTab(opts.port, 'about:blank');
  const client = await CDP({ port: opts.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  // Catches the failures Runtime.exceptionThrown does NOT see: every page
  // here wraps startup as init().catch(handleErr), so a bad parameter or a
  // shader/bind-group mismatch is a console.error plus an `error:` line in
  // #status and nothing uncaught. See attachPageWatch's own header.
  const watch = await attachPageWatch(client, {
    onError: (e) => console.error(`    !! [${e.kind}] ${e.text.split('\n')[0]}`),
  });

  const report = [];
  let exitCode = 0;
  try {
    for (const g of groups) {
      for (const c of g.cases) {
        const url = caseUrl(opts.baseUrl, g.scenario, c, opts.extra);
        console.log(`\n=== ${c.name} (${url})`);
        try {
          await navigateTo(Page, url);
          await waitForGlobal(Runtime, 'window.__D3', 60000);
          await assertPageHealthy(Runtime, watch, c.name);
          const res = await g.run(Runtime, opts, c, s => console.log('    ' + s));
          await assertPageHealthy(Runtime, watch, c.name);
          report.push({ name: c.name, scenario: g.scenario, gate: g.gate, res });
        } catch (err) {
          console.error(`    FAILED: ${err.message}`);
          report.push({ name: c.name, scenario: g.scenario, gate: g.gate, error: err.message });
          exitCode = 1;
        }
      }
    }
  } finally {
    await client.close();
    await teardown({ port: opts.port, tabId, chrome, server, keepOpen: opts.keepOpen });
  }

  console.log('\n' + '='.repeat(104));
  console.log('SUMMARY');
  console.log('='.repeat(104));
  console.log(pad('case', 18) + pad('kind', 12) + padL('field L2rel', 14) + padL('2nd metric', 20) + padL('3rd metric', 20) + padL('verdict', 10));
  console.log('-'.repeat(104));
  for (const r of report) {
    if (r.error) { console.log(pad(r.name, 18) + pad(r.scenario, 12) + padL('-', 14) + padL('-', 20) + padL('-', 20) + padL('ERROR', 10)); continue; }
    const x = r.res;
    let c2 = '-', c3 = '-', checks = [];
    if (r.scenario === 'duct') {
      c2 = `peak ${e3(x.peakRelErr)}`; c3 = `xspread ${e3(x.xSpreadRel)}`;
      checks = [x.fieldCheck, x.peakCheck, x.xCheck];
    } else if (r.scenario === 'beltrami') {
      c2 = `rate ${e3(x.rateRelErr)}`; c3 = `td ${x.td.toFixed(0)}`;
      checks = [x.fieldCheck, x.rateCheck];
    } else {
      c2 = `eps ${e3(x.samples[x.samples.length - 1].dissipation)}`; c3 = `Re ${x.re.toFixed(0)}`;
      checks = [x.finiteCheck];
    }
    const ok = checks.every(k => k.pass);
    if (!ok && r.gate) exitCode = 1;
    if (!ok && !r.gate) exitCode = 1;   // tgv can still fail on non-finite
    console.log(pad(r.name, 18) + pad(r.scenario + (r.gate ? '' : ' (rep)'), 12)
      + padL(r.scenario === 'tgv' ? '-' : e3(r.scenario === 'duct' ? x.l2rel : x.maxL2rel), 14)
      + padL(c2, 20) + padL(c3, 20) + padL(ok ? 'PASS' : 'FAIL', 10));
  }

  const failed = report.filter(r => r.error || (r.res && [
    ...(r.scenario === 'duct' ? [r.res.fieldCheck, r.res.peakCheck, r.res.xCheck] : []),
    ...(r.scenario === 'beltrami' ? [r.res.fieldCheck, r.res.rateCheck] : []),
    ...(r.scenario === 'tgv' ? [r.res.finiteCheck] : []),
  ].some(k => !k.pass)));
  if (failed.length) {
    console.log('\nDetails for anything not PASS:');
    for (const r of failed) {
      if (r.error) { console.log(`  [${r.name}] ${r.error}`); continue; }
      const checks = r.scenario === 'duct' ? [r.res.fieldCheck, r.res.peakCheck, r.res.xCheck]
        : r.scenario === 'beltrami' ? [r.res.fieldCheck, r.res.rateCheck] : [r.res.finiteCheck];
      for (const k of checks) {
        if (k.pass) continue;
        console.log(`  [${r.name}] ${k.label}: ${e3(k.measured)} against a tolerance of ${e3(k.tol)}`);
      }
    }
  } else {
    console.log('\nAll cases PASS.');
  }
  process.exit(exitCode);
}

main();
