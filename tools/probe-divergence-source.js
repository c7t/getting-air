#!/usr/bin/env node
// WHERE does a run-to-run divergence enter -- the WANT SET, or the FORCE?
// (plans/uniform-levels.md D1-a, "One more shot at the residual")
//
// tools/measure-determinism.js answers WHETHER two runs differ, by hashing a
// whole 20 MB snapshot at the end. It cannot say which part moved first, and a
// snapshot diff at step 4096 shows everything downstream of the first
// divergence as well as the divergence itself. This steps in short chunks and
// records only small payloads, so it can checkpoint often:
//
//   the active block list, SORTED   -> the SET of refined blocks, i.e. what the
//                                      criterion and the 2:1 cascade decided
//   the card state                  -> the force accumulators
//
// A want set that never diverges while the force does exonerates the criterion
// and the cascade and points at the force reduction. That is what it found on
// 2026-09-18: 8 runs x 4096 steps of `?levels=3&detslots=1`, want set identical
// at every one of 32 checkpoints, card state splitting 3/5 from step 128.
//
// ── TWO THINGS THIS PROBE GOT WRONG ABOUT ITSELF, BOTH WORTH KEEPING ────────
//
// 1. ITS FIRST CARD CHANNEL WAS VACUOUS. `debugReadCardState` returns an
//    OBJECT; `Array.from` of it is `[]`, so the hash was a constant and the
//    channel reported "never diverges" over 8 runs at 2048 AND 4096 steps.
//    That reads exactly like a clean negative result. The liveness control at
//    the bottom -- distinct values ACROSS checkpoints WITHIN one run -- exists
//    because of it, and prints next to every verdict.
//
// 2. THERE IS NO HANDOUT CHANNEL HERE, AND THERE LOOKED LIKE ONE.
//    `debugListActiveBlocks` returns block IDs already in sorted order, so
//    hashing it unsorted gives the same string as sorting it. An earlier
//    version reported that as a second "handout" channel that "never
//    diverges"; it was a copy of the want-set channel and could not see a slot
//    permutation at all. Reading `blockSlot` itself is what that would take.
//
// It REPORTS; there is no PASS/FAIL. A run in which nothing diverges is not
// evidence of determinism -- see measure-determinism.js's header on which way
// each kind of row cuts.
//
//     node tools/probe-divergence-source.js
//     node tools/probe-divergence-source.js --runs=8 --steps=4096 --chunk=128
//     node tools/probe-divergence-source.js --q='levels=2&detslots=1'
//
// VERIFY WHICH TREE THE DEV SERVER IS SERVING first -- `ensureServer` reuses
// whatever already answers on the port.

const path = require('path');
const crypto = require('crypto');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = {
    baseUrl: 'https://localhost:4444', port: 9346,
    page: 'index-cylinder-amr.html', global: null,
    runs: 8, steps: 4096, chunk: 128, q: 'levels=3&detslots=1',
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--global=')) o.global = a.slice(9);
    else if (a.startsWith('--runs=')) o.runs = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--chunk=')) o.chunk = parseInt(a.slice(8));
    else if (a.startsWith('--q=')) o.q = a.slice(4);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  // Same page table, same reasoning, as measure-determinism.js and
  // measure-pool-peaks.js: the debug surface's name does not follow the page's.
  if (!o.global) {
    const AMR_PAGES = ['index-amr.html', 'index-reentry-amr.html'];
    o.global = AMR_PAGES.includes(o.page) ? 'window.__AMR' : 'window.__CYL';
  }
  return o;
}

async function ev(R, e, t) {
  const r = await evalExpr(R, e, t);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}
const h = (s) => crypto.createHash('sha256').update(s).digest('hex').slice(0, 12);

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const G = o.global;
  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  const runs = [];
  try {
    for (let r = 0; r < o.runs; r++) {
      await navigateTo(Page, `${o.baseUrl}/${o.page}?${o.q}`);
      await waitForGlobal(Runtime, G, 60000);
      // Pause before reset, and check the step count afterwards -- the page
      // boots liveMode=true and an uncounted rAF frame between the reset() and
      // the step() round trips is how this class of measurement goes wrong.
      await ev(Runtime, `${G}.setLive(false)`);
      await ev(Runtime, `${G}.reset()`, 120000);
      const nL = await ev(Runtime, `${G}.getNumLevels()`);
      const marks = [];
      for (let s = 0; s < o.steps; s += o.chunk) {
        await ev(Runtime, `${G}.debugStepSync(${o.chunk})`, 600000);
        const set = await ev(Runtime, `(async()=>{
          const out=[];
          for (let m=1;m<${nL};m++){ const a=await ${G}.debugListActiveBlocks(m); out.push(a.slice().sort((x,y)=>x-y).join(',')); }
          return out.join('|');
        })()`, 120000);
        const card = await ev(Runtime, `(async()=>JSON.stringify(await ${G}.debugReadCardState()))()`, 120000);
        marks.push({ step: s + o.chunk, set: h(set), card: h(card) });
      }
      const got = await ev(Runtime, `${G}.getStep()`);
      if (got !== o.steps) throw new Error(`asked for ${o.steps} steps, page reports ${got} -- it stepped behind the harness`);
      runs.push(marks);
      const L = marks[marks.length - 1];
      console.log(`  run ${r + 1}: ${marks.length} checkpoints, final set=${L.set} card=${L.card}`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  console.log(`\n${o.page}?${o.q} -- first checkpoint at which the runs disagree:`);
  const first = { set: null, card: null };
  for (let i = 0; i < runs[0].length; i++) {
    for (const k of ['set', 'card']) {
      const d = new Set(runs.map(m => m[i][k]));
      if (first[k] === null && d.size > 1) first[k] = { step: runs[0][i].step, n: d.size };
    }
  }
  const say = (k) => first[k] ? `step ${first[k].step}, ${first[k].n} distinct` : 'never -- identical at every checkpoint in every run';
  console.log(`  WANT SET   (which blocks are refined): ${say('set')}`);
  console.log(`  CARD STATE (the force accumulators):   ${say('card')}`);

  // THE CONTROL, and it is not decoration: a channel whose value never moves
  // across steps within a single run is measuring nothing, and its "never
  // diverges" above means nothing either. That is precisely how this probe's
  // first card channel read clean while being constant.
  console.log('\nchannel liveness (distinct values across checkpoints WITHIN run 1):');
  for (const k of ['set', 'card']) {
    const n = new Set(runs[0].map(m => m[k])).size;
    console.log(`  ${k.padEnd(5)} ${n} distinct over ${runs[0].length} checkpoints` +
      `${n === 1 ? '   <-- VACUOUS, its verdict above means nothing' : ''}`);
  }
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
