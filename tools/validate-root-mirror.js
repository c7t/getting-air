#!/usr/bin/env node
// U2: does a kernel addressing the ROOT POOL reach the same cells the dense L0
// grid has? (plans/uniform-levels.md U2)
//
// Mirrors the dense grid into the root pool with shaders/amr_mirror_root.wgsl,
// then compares the two buffers word for word against an INDEPENDENT host
// route -- amr2d.mjs's `rootCellToDense`, which derives the dense index from
// `blockGridAtLevel` where the shader derives it from `slotToBlock` and its own
// overrides. Neither consults the other, which is what makes agreement
// evidence rather than a tautology.
//
// THE CLEAN RESULT IS NOT THE GATE. A comparison that never executes, or that
// compares a buffer against itself, reports zero mismatches too. So every
// configuration is run twice:
//
//     mirror, then check          -> must be CLEAN
//     step 64, then check again   -> must be DIRTY
//
// The second is the discrimination: it proves the comparison reads live data
// and that the loop covers cells the solver actually moves. A run where the
// stale check also comes back clean is a broken instrument, not a pass, and is
// reported as a failure.
//
// WHY IT RUNS BEFORE ANY KERNEL USES THE POOL. An addressing or stride error
// found here costs an afternoon. Found at U3 it arrives as a wrong field and
// reads like a physics regression. That ordering is B3-5's -- prove the closed
// form against the live buffer FIRST, then move anything onto it -- and it
// earned its keep on the first run: the mirror divided its plane stride by
// `fWords()` instead of 9, which is correct at the default and puts four of
// every cell's nine planes at the wrong offset under `?f16=1`. Only the f16
// rung saw it, which is why the rungs are not all the default.
//
//     node tools/validate-root-mirror.js
//     node tools/validate-root-mirror.js --baseUrl=https://localhost:4456 --port=9349

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

// Deliberately not all the default: levels vary the pool count, f16 varies the
// packing (and is the rung that caught the stride bug), res varies the domain
// and therefore the root block grid.
const CASES = [
  'levels=2&rootpool=1',
  'levels=3&rootpool=1',
  'levels=4&rootpool=1',
  'levels=2&rootpool=1&f16=1',
  'levels=2&rootpool=1&res=9',
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9333, page: 'index-amr.html', steps: 512, keepOpen: false };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs || 300000);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  const rows = [];
  try {
    for (const q of CASES) {
      await navigateTo(Page, `${o.baseUrl}/${o.page}?${q}`);
      await waitForGlobal(Runtime, 'window.__AMR', 60000);
      // reset() first: the rAF loop runs between load and setLive(false), so
      // anything read before a reset is an unknown number of steps of drift.
      await ev(Runtime, 'window.__AMR.reset()');
      await ev(Runtime, `window.__AMR.debugStepSync(${o.steps})`);

      await ev(Runtime, 'window.__AMR.debugMirrorRoot()');
      const clean = JSON.parse(await ev(Runtime, 'window.__AMR.debugCheckRootMirror().then(r => JSON.stringify(r))'));
      await ev(Runtime, 'window.__AMR.debugStepSync(64)');
      const stale = JSON.parse(await ev(Runtime, 'window.__AMR.debugCheckRootMirror().then(r => JSON.stringify(r))'));

      const ok = clean.ok === true && stale.ok === false && stale.mismatched > 0;
      rows.push({ q, ok, clean, stale });
      console.log(`${ok ? '  ok  ' : ' FAIL '} ${q.padEnd(30)} mirrored ${clean.mismatched}/${clean.checked} bad` +
        `   after 64 steps ${stale.mismatched}/${stale.checked} bad`);
      if (clean.first && clean.first.length) console.log('        first:', JSON.stringify(clean.first[0]));
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  const bad = rows.filter(r => !r.ok);
  const brokenInstrument = rows.filter(r => r.clean.ok === true && r.stale.ok !== false);
  console.log('');
  if (brokenInstrument.length) {
    console.log('  A STALE POOL COMPARED CLEAN. The mirror matched and then 64 steps of the');
    console.log('  solver changed nothing the comparison could see -- that is a comparison');
    console.log('  reading the wrong buffer, not a correct mapping. Do not read the clean');
    console.log('  column above as a result.');
  }
  if (bad.length) {
    console.log(`FAIL: ${bad.map(r => r.q).join(', ')}`);
    process.exit(1);
  }
  console.log('PASS: the root pool addresses the same cells the dense grid does.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
