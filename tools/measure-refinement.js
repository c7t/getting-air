#!/usr/bin/env node
// What the refinement criterion actually SELECTS, and what actually LIMITS it.
//
// Persisted form of the ad hoc scans used to chase "block artifacts in the
// wake" -- same reasoning as tools/validate-amr-invariants.js, which was
// itself the persisted form of a one-off coverage scan. This reports; it does
// not PASS/FAIL. There is no literature value for "how much of a wake should
// be refined"; the numbers are for retuning against, and for catching the
// case where a knob you are turning is not the one that binds.
//
// THREE MEASUREMENTS, because the obvious one alone is misleading:
//
//  1. COVERAGE. Per-L0-block max|omega| reconstructed from a snapshot's own
//     coarse velocity field -- the same quantity amr_criterion.wgsl reduces,
//     with the same central-difference formula -- tabulated against candidate
//     REFINE_THRESH values. This is the table main-amr.js's REFINE_THRESH
//     comment is built from, so a retune can be compared like for like.
//
//  2. CHURN. Per refinement round, how many tiles are created and destroyed,
//     and how many of the creations are a block being RE-created within a few
//     rounds of being released (FLAPPING). Every transition re-interpolates a
//     block-sized patch from the coarser parent, so churn is the mechanism
//     behind block-shaped artifacts -- but churn alone does not prove a
//     problem, because a convecting wake legitimately moves its refined
//     region. Flapping is the part that indicates the hysteresis band is too
//     narrow. Measured on this page at the shipped defaults: L2 turns over
//     ~23% of its tiles per round with ~0% flapping, i.e. the churn is the
//     wake convecting, not the criterion thrashing.
//
//  3. WHAT BINDS. The reason to run this at all. Refinement is gated by more
//     than REFINE_THRESH: the SPONGE_EXCLUDE_W band suppresses vorticity-
//     driven refinement near the window edge, and a pool at its cap silently
//     refuses (not gracefully -- slots are granted in blockID order, so the
//     free list dries up mid-row and denied blocks form horizontal BANDS).
//     Measured on index-amr.html at 26k steps: lowering REFINE_THRESH from -9
//     to -11 with the sponge band ON moved L1 from 86 to only 94 tiles, so
//     the threshold looked nearly inert; with the band OFF the same change
//     went 102 -> 191. The band was the binding constraint and was masking
//     the threshold entirely. --compare runs that A/B for you.
//
// A WARNING ABOUT VISUAL A/B ON THIS PAGE. Do not compare two runs of
// index-amr.html by eye at any long horizon. Changing refinement changes the
// physics, and the falling card is chaotic -- two runs at 26k steps are at
// completely different points in the tumble and their wakes are not
// comparable at all. That is plans/AMR-vs-dense-validation.md's Finding #3 in
// a different costume. Judge a refinement change on the cylinder harness
// (statistically steady, literature Cd/St) or with tools/validate-divergence.js
// (both legs seeded from one state), never on this page's appearance.
//
// Usage:
//   node tools/measure-refinement.js
//   node tools/measure-refinement.js --steps=26000 --rounds=60
//   node tools/measure-refinement.js --url='/index-amr.html?levels=3'
//   node tools/measure-refinement.js --compare      # what-binds A/B

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const BL = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');
const BASE_URL = 'https://localhost:4444';
const PORT = 9333;

function parseArgs(argv) {
  const o = { url: '/index-amr.html', steps: 26000, rounds: 60, every: 16, compare: false, global: 'window.__AMR' };
  for (const a of argv) {
    if (a.startsWith('--url=')) o.url = a.slice(6);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--rounds=')) o.rounds = parseInt(a.slice(9));
    else if (a.startsWith('--every=')) o.every = parseInt(a.slice(8));
    else if (a.startsWith('--global=')) o.global = a.slice(9);
    else if (a === '--compare') o.compare = true;
  }
  return o;
}

// Per-L0-block max|omega| from the snapshot's own coarse velocity field.
// Deliberately re-derives amr_criterion.wgsl's formula rather than sharing
// one, so a change to the shader shows up here as a disagreement instead of
// being silently tracked -- the same convention debugCheckGeometryCoverage
// uses against amr_manage.wgsl.
const BLOCK_OMEGA = (G) => `(async () => {
  const snap = await ${G}.debugSnapshotSave();
  const { W, H } = ${G}.getDims();
  const BLOCK = 8, nbx = W / BLOCK, nby = H / BLOCK;
  const bin = atob(snap.velB64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const vel = new Float32Array(bytes.buffer);
  const cellIndex = (cx, cy) => {
    const bx = (cx / BLOCK) | 0, by = (cy / BLOCK) | 0;
    return (by * nbx + bx) * (BLOCK * BLOCK) + (cy % BLOCK) * BLOCK + (cx % BLOCK);
  };
  const wrap = (v, n) => ((v % n) + n) % n;
  const gx = (cx, cy) => vel[cellIndex(wrap(cx, W), wrap(cy, H)) * 2];
  const gy = (cx, cy) => vel[cellIndex(wrap(cx, W), wrap(cy, H)) * 2 + 1];
  const blockMax = new Float64Array(nbx * nby);
  for (let cy = 0; cy < H; cy++) for (let cx = 0; cx < W; cx++) {
    const om = Math.abs((gy(cx + 1, cy) - gy(cx - 1, cy)) * 0.5 - (gx(cx, cy + 1) - gx(cx, cy - 1)) * 0.5);
    const b = ((cy / BLOCK) | 0) * nbx + ((cx / BLOCK) | 0);
    if (om > blockMax[b]) blockMax[b] = om;
  }
  return { nbx, nby, blockMax: Array.from(blockMax) };
})()`;

const activeCounts = (G) => `(async () => {
  const o = {}; const N = ${G}.getNumLevels();
  for (let m = 1; m < N; m++) o[m] = (await ${G}.debugListActiveBlocks(m)).length;
  return JSON.stringify(o);
})()`;

async function openPage(Page, Runtime, url, G) {
  await BL.navigateTo(Page, BASE_URL + url);
  await BL.waitForGlobal(Runtime, G, 30000);
  await BL.evalExpr(Runtime, `${G}.setLive(false)`);
}

async function reportCoverage(Runtime, opts) {
  const G = opts.global;
  const r = await BL.evalExpr(Runtime, BLOCK_OMEGA(G), 300000);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
  const { nbx, nby, blockMax } = r.result.value;
  const total = nbx * nby;
  const peak = Math.max(...blockMax);
  // "Vorticity-bearing" at 1e-3, the definition main-amr.js's own retune
  // table used, so the two are directly comparable.
  const wake = blockMax.filter(v => v >= 1e-3).length;
  const params = JSON.parse((await BL.evalExpr(Runtime, `JSON.stringify(${G}.getRefineParams())`)).result.value);
  const act = JSON.parse((await BL.evalExpr(Runtime, activeCounts(G), 60000)).result.value);
  const caps = JSON.parse((await BL.evalExpr(Runtime,
    `JSON.stringify(${G}.getLevelPoolSizes().map(p => ({ level: p.level, cap: p.MAX_FINE_BLOCKS })))`)).result.value);

  console.log(`\n  domain ${nbx}x${nby} = ${total} L0 blocks   peak block |omega| ${peak.toExponential(3)} (log2 ${Math.log2(peak).toFixed(2)})`);
  console.log(`  REFINE_THRESH=${params.REFINE_THRESH} COARSEN_THRESH=${params.COARSEN_THRESH}`);
  console.log(`  live tiles: ` + caps.map(c => `L${c.level}=${act[c.level]}/${c.cap}${act[c.level] >= c.cap ? ' SATURATED' : ''}`).join('  '));
  console.log(`  vorticity-bearing blocks (|omega| >= 1e-3): ${wake}`);
  console.log('\n  thresh   |omega| >=    selected   % domain   % of wake');
  for (const t of [-6, -7, -8, -9, -10, -11, -12, -13]) {
    const sel = blockMax.filter(v => v >= 2 ** t).length;
    console.log(`   ${String(t).padStart(4)}     ${(2 ** t).toExponential(2)}      ${String(sel).padStart(5)}     ${(100 * sel / total).toFixed(1).padStart(6)}%    ${(100 * Math.min(sel, wake) / Math.max(1, wake)).toFixed(0).padStart(5)}%`);
  }
  return { act, caps };
}

async function reportChurn(Runtime, opts) {
  const G = opts.global;
  const N = (await BL.evalExpr(Runtime, `${G}.getNumLevels()`)).result.value;
  const snap = async (m) => new Set((await BL.evalExpr(Runtime,
    `(async () => (await ${G}.debugListActiveBlocks(${m})).map(b => b.bx + ',' + b.by))()`, 60000)).result.value);

  const stats = {}, prev = {};
  for (let m = 1; m < N; m++) { stats[m] = { born: 0, died: 0, flap: 0, active: 0, recentlyDied: [] }; prev[m] = await snap(m); }
  for (let r = 0; r < opts.rounds; r++) {
    await BL.evalExpr(Runtime, `${G}.debugStepSync(${opts.every})`, 300000);
    for (let m = 1; m < N; m++) {
      const cur = await snap(m), s = stats[m], diedNow = [];
      for (const k of cur) if (!prev[m].has(k)) { s.born++; if (s.recentlyDied.some(d => d.has(k))) s.flap++; }
      for (const k of prev[m]) if (!cur.has(k)) { s.died++; diedNow.push(k); }
      s.active += cur.size;
      s.recentlyDied.push(new Set(diedNow));
      if (s.recentlyDied.length > 4) s.recentlyDied.shift();
      prev[m] = cur;
    }
  }
  console.log(`\n  churn over ${opts.rounds} rounds of ${opts.every} steps:`);
  console.log('  level  meanActive  born/rd  died/rd  flap/rd   flap% of born   churn% of active');
  for (let m = 1; m < N; m++) {
    const s = stats[m], mean = s.active / opts.rounds;
    const bpr = s.born / opts.rounds, dpr = s.died / opts.rounds;
    console.log(`    ${m}    ${mean.toFixed(1).padStart(9)}  ${bpr.toFixed(2).padStart(7)}  ${dpr.toFixed(2).padStart(7)}` +
      `  ${(s.flap / opts.rounds).toFixed(2).padStart(7)}   ${(100 * s.flap / Math.max(1, s.born)).toFixed(1).padStart(12)}%` +
      `   ${(100 * (bpr + dpr) / Math.max(1e-9, mean)).toFixed(1).padStart(14)}%`);
  }
  console.log('  (high churn with ~0% flap is the wake convecting, not the criterion thrashing)');
}

// The A/B that says which gate is actually binding. Pool caps are raised in
// every arm so a cap cannot be the thing that differs.
const COMPARE_CASES = [
  { name: 'default', q: '' },
  { name: 'thresh -11', q: '&refineThresh=-11&coarsenThresh=-12&refineInc=2&maxFineBlocks=512&maxFineBlocks2=768' },
  { name: 'no sponge band', q: '&spongeExclude=0&maxFineBlocks=512&maxFineBlocks2=768' },
  { name: 'both', q: '&spongeExclude=0&refineThresh=-11&coarsenThresh=-12&refineInc=2&maxFineBlocks=512&maxFineBlocks2=768' },
];

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const server = await BL.ensureServer(BASE_URL, REPO_ROOT);
  const chrome = await BL.ensureChrome(PORT);
  const tabId = await BL.openTab(PORT, 'about:blank');
  const client = await CDP({ port: PORT, target: tabId });
  const { Page, Runtime } = client;
  await Page.enable(); await Runtime.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  try {
    if (opts.compare) {
      const rows = [];
      for (const c of COMPARE_CASES) {
        const url = opts.url + (opts.url.includes('?') ? '' : '?') + c.q.replace(/^&/, opts.url.includes('?') ? '&' : '');
        console.log(`\n=== ${c.name}  ${url}`);
        await openPage(Page, Runtime, url, opts.global);
        await BL.evalExpr(Runtime, `${opts.global}.debugStepSync(${opts.steps})`, 900000);
        const act = JSON.parse((await BL.evalExpr(Runtime, activeCounts(opts.global), 60000)).result.value);
        console.log('  active', JSON.stringify(act));
        rows.push({ name: c.name, act });
      }
      console.log('\n  case                L1     L2');
      for (const r of rows) console.log(`  ${r.name.padEnd(18)} ${String(r.act[1]).padStart(4)}  ${String(r.act[2] ?? '-').padStart(5)}`);
      console.log('\n  If lowering the threshold barely moves L1 but removing the sponge band does,');
      console.log('  the band is the binding gate and the threshold is being masked by it.');
    } else {
      console.log(`=== ${opts.url}  (${opts.steps} steps)`);
      await openPage(Page, Runtime, opts.url, opts.global);
      await BL.evalExpr(Runtime, `${opts.global}.debugStepSync(${opts.steps})`, 900000);
      await reportCoverage(Runtime, opts);
      await reportChurn(Runtime, opts);
    }
  } finally {
    await client.close();
    await BL.teardown({ port: PORT, tabId, chrome, server });
  }
}

main().catch(e => { console.error(e); process.exit(1); });
