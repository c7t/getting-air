#!/usr/bin/env node
// How many tiles does each level actually want, with every cap lifted?
// (plans/uniform-levels.md U7-5a)
//
// The persisted form of the ad hoc scan that produced `POOL_PEAKS` on each AMR
// page on 2026-09-15 -- same reasoning tools/measure-refinement.js records for
// its own origin. It reports; it does not PASS/FAIL. There is no right number
// of tiles, only a measured demand and the headroom convention
// (`amr2d.mjs`'s `poolSlotsFor`, 1.7x) applied to it.
//
// WHY IT EXISTS NOW: U5-4 makes level 1 a QUAD child of the root, so a region
// that used to earn one 8-cell block earns a 16-cell quad (see the plan's "The
// quad granularity"). The shipped `POOL_PEAKS` were measured under the
// per-block allocator, and U7-5 flips the quad allocator on by default. Pool
// capacity sized from the wrong allocator is not a small error: CLAUDE.md
// records a flat default producing a `?levels=4` REFUSAL, and a refused
// geometry-forced refine is silent -- the region simply contributes no force.
//
// THE PEAK IS A SAMPLED MAX, and that is a deliberate limitation rather than
// an oversight. Nothing in the page tracks a high-water mark, so this polls
// `debugListActiveBlocks` every `--sample=` steps; a spike entirely between
// two samples is missed. The 1.7x headroom convention exists to absorb exactly
// that, and the numbers this replaces were obtained the same way -- so the
// comparison is like for like, which is what U7-5 needs. Raise `--sample=` (a
// smaller number) if you want a tighter bound and can pay the readbacks.
//
// ── READ THE TWO PAGES DIFFERENTLY, AND THIS IS THE IMPORTANT PART ──────────
//
// ONE ALLOCATOR SINCE U7-6f, so one leg. This ran a `rootpool=0` and a
// `rootpool=1` column; `--rootpool=` is still accepted and now ignored.
//
// THE WARNING THAT COLUMN CARRIED IS WORTH KEEPING. index-amr.html is a
// FALLING CARD and it is chaotic: two runs that differ in refinement are at
// different points in the tumble long before 40k steps, so its two columns
// were NOT a controlled comparison of allocator granularity -- they were two
// different trajectories, each legitimately demanding what it demanded. Same
// trap measure-refinement.js warns about for visual A/B, one level down. Any
// future two-configuration comparison here needs index-cylinder-amr.html,
// which is statistically steady with a pinned body.
//
//   index-amr.html         the numbers that SIZE THE SHIPPED POOLS.
//   index-cylinder-amr.html  statistically steady, pinned body.
//
//     node tools/measure-pool-peaks.js
//     node tools/measure-pool-peaks.js --steps=40000 --levels=3,4,5
//     node tools/measure-pool-peaks.js --page=index-cylinder-amr.html --levels=3,4
//
// VERIFY WHICH TREE THE DEV SERVER IS SERVING first -- `ensureServer` reuses
// whatever already answers on the port, and this repo has twice discarded a
// full sweep that ran green against the wrong checkout.

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = {
    baseUrl: 'https://localhost:4444', port: 9345,
    page: 'index-amr.html', steps: 40000, sample: 256, cap: 2048,
    levels: [3, 4, 5], rootpool: null, extra: '', global: null,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--sample=')) o.sample = parseInt(a.slice(9));
    else if (a.startsWith('--cap=')) o.cap = parseInt(a.slice(6));
    else if (a.startsWith('--levels=')) o.levels = a.slice(9).split(',').map(Number);
    else if (a.startsWith('--rootpool=')) o.rootpool = a.slice(11);
    else if (a.startsWith('--extra=')) o.extra = a.slice(8);
    else if (a.startsWith('--global=')) o.global = a.slice(9);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  // THE DEBUG SURFACE IS NOT THE SAME NAME ON EVERY PAGE, and it does not
  // follow the page name either: index-amr and index-reentry-amr expose
  // `window.__AMR`, while index-cylinder-amr, index-tgv-amr and
  // index-channel-amr expose `window.__CYL` -- the harness pages kept the
  // cylinder harness's original name as they were derived from it. Defaulted
  // from a table rather than guessed from the URL, and overridable, because a
  // wrong guess here reads as "WebGPU init failed" and sends you looking at
  // the page.
  if (!o.global) {
    const AMR_PAGES = ['index-amr.html', 'index-reentry-amr.html'];
    o.global = AMR_PAGES.includes(o.page) ? 'window.__AMR' : 'window.__CYL';
  }
  // Quad allocation refuses a cap that is not a multiple of 4 (allocLevelPool),
  // and since U7-6f that is EVERY level including level 1.
  if (o.cap % 4 !== 0) { console.error('--cap must be a multiple of 4 (quad allocation)'); process.exit(2); }
  return o;
}

async function ev(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// Every level's cap lifted at once. ?maxFineBlocks= sizes LEVEL 1 ONLY and each
// deeper level takes its own ?maxFineBlocks<m>= -- the asymmetry CLAUDE.md
// records as having made a real refusal look spurious.
function capParams(nLevels, cap) {
  const parts = [`maxFineBlocks=${cap}`];
  for (let m = 2; m < nLevels; m++) parts.push(`maxFineBlocks${m}=${cap}`);
  return parts.join('&');
}

async function measureOne(Runtime, Page, o, nLevels, rootpool) {
  const q = [`levels=${nLevels}`, capParams(nLevels, o.cap)];
  if (o.extra) q.push(o.extra.replace(/^&/, ''));
  const url = `${o.baseUrl}/${o.page}?${q.join('&')}`;
  const G = o.global;
  await navigateTo(Page, url);
  await waitForGlobal(Runtime, G, 60000);

  // Assert the configuration took. A cap the page silently ignored, or a flag
  // that did not apply, would read as a clean low peak -- the failure mode this
  // whole measurement exists to prevent.
  const got = await ev(Runtime, `${G}.getNumLevels()`);
  if (got !== nLevels) throw new Error(`asked for levels=${nLevels}, page reports ${got}`);
  const sizes = await ev(Runtime, `JSON.stringify(${G}.getLevelPoolSizes())`);
  const caps = JSON.parse(sizes).map(p => p.MAX_FINE_BLOCKS);
  if (caps.some(c => c !== o.cap)) {
    throw new Error(`cap not applied: levels report MAX_FINE_BLOCKS ${caps.join(',')} against --cap=${o.cap}`);
  }
  void rootpool;
  const rp = await ev(Runtime, `${G}.getRootPool() ? 1 : 0`);
  if (!rp) throw new Error('page reports no root pool -- level 0 has no pool to size');

  await ev(Runtime, `${G}.reset()`, 120000);
  const peaks = new Array(nLevels).fill(0);
  const peakAt = new Array(nLevels).fill(0);
  let done = 0;
  let refused = false;
  while (done < o.steps) {
    const chunk = Math.min(o.sample, o.steps - done);
    await ev(Runtime, `${G}.debugStepSync(${chunk})`, 900000);
    done += chunk;
    const counts = JSON.parse(await ev(Runtime, `(async () => {
      const out = [];
      for (let m = 1; m < ${nLevels}; m++) out.push((await ${G}.debugListActiveBlocks(m)).length);
      return JSON.stringify(out);
    })()`, 300000));
    for (let m = 1; m < nLevels; m++) {
      if (counts[m - 1] > peaks[m]) { peaks[m] = counts[m - 1]; peakAt[m] = done; }
      // A level sitting exactly at its cap is the one reading this measurement
      // cannot interpret: the demand was clipped, so the peak is the cap and
      // not a demand at all.
      if (counts[m - 1] >= o.cap) refused = true;
    }
  }
  return { peaks, peakAt, refused, url };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  // ONE LEG SINCE U7-6f. `?rootpool=` is gone with the dense path, so there is
  // no second allocator to compare against; `--rootpool=` is accepted and
  // ignored rather than removed, so an old invocation reports rather than
  // exiting 2 on an unknown argument.
  const legs = [null];

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  const rows = [];
  try {
    console.log(`  ${o.page}, ${o.steps} steps, cap ${o.cap} on every level, sampled every ${o.sample}\n`);
    for (const rp of legs) {
      for (const n of o.levels) {
        const r = await measureOne(Runtime, Page, o, n, rp);
        rows.push({ rootpool: rp, nLevels: n, ...r });
        const cols = [];
        for (let m = 1; m < n; m++) cols.push(`L${m} ${String(r.peaks[m]).padStart(5)} @${String(r.peakAt[m]).padStart(6)}`);
        console.log(`  levels=${n}  ${cols.join('   ')}${r.refused ? '   *** HIT THE CAP -- clipped, not a demand' : ''}`);
      }
      console.log('');
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  // The POOL_PEAKS shape each page declares: demand keyed by a level's ROLE at
  // that depth (finest vs parent), because it steps up when a level acquires a
  // child -- 2:1 closure forcing a parent tile for everything refined below.
  for (const rp of legs) {
    const mine = rows;
    if (!mine.length) continue;
    const finest = {}, parent = {};
    for (const r of mine) {
      for (let m = 1; m < r.nLevels; m++) {
        const isFinest = m === r.nLevels - 1;
        const bag = isFinest ? finest : parent;
        bag[m] = Math.max(bag[m] || 0, r.peaks[m]);
      }
    }
    void rp;
    console.log('POOL_PEAKS (max over the level counts measured):');
    console.log(`  finest: { ${Object.keys(finest).sort().map(k => `${k}: ${finest[k]}`).join(', ')} },`);
    console.log(`  parent: { ${Object.keys(parent).sort().map(k => `${k}: ${parent[k]}`).join(', ')} },`);
  }

  console.log('\nRead the ABSOLUTE peaks on index-amr.html -- they size the shipped pools.');
  console.log('The two-allocator RATIO this used to report went with ?rootpool= at U7-6f.');
  if (rows.some(r => r.refused)) {
    console.log('\nAt least one level HIT THE CAP: that row reports a clip, not a demand.');
    console.log('Re-run it with a larger --cap= before using any number from it.');
    process.exit(1);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
