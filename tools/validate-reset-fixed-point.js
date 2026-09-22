#!/usr/bin/env node
// Is `reset()` a FIXED POINT -- does it put the page in the same state
// regardless of what the page has already done?
// (plans/uniform-levels.md, "resetSim was not a fixed point")
//
// It should be the most boring property a simulator has, and this project got
// it wrong three times in three days, each time on a different page, each time
// with the same shape:
//
//   D1-a       main-cylinder-amr.js   an unseeded rng, plus velBuf /
//                                     finePoolVel / parentSlotBuf never written
//   U7-6c      the root pool          seedRootFromDense writes f_root only
//   U7-6c-fix  main-amr.js            the same three buffers again
//
// Every one of them is the same false inference: "WebGPU zero-initialises
// buffers, so this one needs no write." TRUE AT ALLOCATION, FALSE AT RESET. A
// fresh buffer is zeros; a buffer being reset holds the last run's history.
//
// `amr2d-gpu.mjs`'s `writePoolInitialState` is the structural answer -- one
// statement of a pool's initial state, called by `allocLevelPool` and by every
// page's `resetSim`, so the two cannot drift. THIS TOOL IS THE OTHER HALF: the
// abstraction covers the pools, and this covers everything, including whatever
// a page adds next to them.
//
// ── THE PROTOCOL ───────────────────────────────────────────────────────────
//
//   A   navigate, pause, reset, snapshot
//   B   step N, reset, snapshot
//   require A == B, bit for bit
//
// B is what a second run in the same page sees. If A != B, the page's second
// run starts somewhere the first never was, and every comparison across a
// reset is measuring page history. That is precisely what made
// `validate-snapshot-roundtrip.js` red for as long as it existed: it runs its
// reference leg from a fresh load and its round-trip leg after a reset.
//
// ── THE CONTROL ────────────────────────────────────────────────────────────
//
// A == B is also what a comparison that cannot see anything produces, so the
// run asserts that stepping actually MOVED the state: the snapshot after N
// steps must differ from the one after the reset. Without it a page that had
// stopped evolving -- or a save whose copies never landed, which has happened
// here and produced an all-zero snapshot -- would pass silently.
//
//     node tools/validate-reset-fixed-point.js
//     node tools/validate-reset-fixed-point.js --steps=512
//     node tools/validate-reset-fixed-point.js --only=cylinder
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

// Every page with a pool hierarchy, at more than one level count, because the
// defects above were all in per-level reset code and a one-level page would
// not have exercised it. The debug surface's name does not follow the page
// name -- same table as measure-determinism.js and measure-pool-peaks.js.
const CONFIGS = [
  { name: 'card levels=2',     page: 'index-amr.html',          q: 'levels=2' },
  { name: 'card levels=3',     page: 'index-amr.html',          q: 'levels=3' },
  { name: 'cylinder levels=2', page: 'index-cylinder-amr.html', q: 'levels=2' },
  { name: 'cylinder levels=3', page: 'index-cylinder-amr.html', q: 'levels=3' },
  { name: 'reentry levels=2',  page: 'index-reentry-amr.html',  q: 'levels=2' },
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9347, steps: 1024, only: null };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--only=')) o.only = a.slice(7);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

const globalFor = (page) =>
  ['index-amr.html', 'index-reentry-amr.html'].includes(page) ? 'window.__AMR' : 'window.__CYL';

async function ev(R, e, t) {
  const r = await evalExpr(R, e, t);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// Key/value pairs of everything in the snapshot, in stable order. `step` is
// excluded: it legitimately counts what the page has done.
function parts(snap) {
  const out = [];
  const walk = (n, k) => {
    if (n === null || n === undefined) return;
    if (typeof n === 'string') { out.push([k, n]); return; }
    if (Array.isArray(n)) { n.forEach((v, i) => walk(v, `${k}[${i}]`)); return; }
    if (typeof n === 'object') { for (const q of Object.keys(n).sort()) walk(n[q], `${k}.${q}`); return; }
    out.push([k, String(n)]);
  };
  const { step, ...rest } = snap;
  walk(rest, '');
  return out;
}
const hash = (p) => crypto.createHash('sha256').update(p.map(([k, v]) => `${k}=${v}`).join('\n')).digest('hex').slice(0, 16);

// Which keys moved, grouped by array so the report is a shape and not 40000
// indices. This is what names the offending buffer.
function diffShape(a, b) {
  const B = new Map(b);
  const g = {};
  for (const [k, v] of a) {
    if (B.get(k) !== v) { const q = k.replace(/\[\d+\]/g, '[]'); g[q] = (g[q] || 0) + 1; }
  }
  return g;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = o.only ? CONFIGS.filter(c => c.name.includes(o.only)) : CONFIGS;
  if (!configs.length) { console.error(`--only=${o.only} matched no configuration`); process.exit(2); }

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
    console.log(`  reset() must be a fixed point -- ${o.steps} steps of history between the two resets\n`);
    for (const cfg of configs) {
      const G = globalFor(cfg.page);
      await navigateTo(Page, `${o.baseUrl}/${cfg.page}?${cfg.q}`);
      await waitForGlobal(Runtime, G, 60000);
      // Pause first: the rAF loop would otherwise step between these calls and
      // the comparison would be measuring frames, not reset.
      await ev(Runtime, `${G}.setLive(false)`);

      await ev(Runtime, `${G}.reset()`, 120000);
      const A = parts(await ev(Runtime, `${G}.debugSnapshotSave()`, 300000));

      await ev(Runtime, `${G}.debugStepSync(${o.steps})`, 900000);
      const stepped = parts(await ev(Runtime, `${G}.debugSnapshotSave()`, 300000));

      await ev(Runtime, `${G}.reset()`, 120000);
      const B = parts(await ev(Runtime, `${G}.debugSnapshotSave()`, 300000));

      const ok = hash(A) === hash(B);
      // THE CONTROL: stepping must have moved something, or A == B proves
      // nothing at all.
      const alive = hash(stepped) !== hash(A);
      rows.push({ cfg, ok, alive });
      console.log(`  ${!alive ? 'DEAD' : ok ? 'ok  ' : 'FAIL'} ${cfg.name.padEnd(20)} A ${hash(A)}  B ${hash(B)}`);
      if (!alive) {
        console.log(`       stepping ${o.steps} steps did not change the snapshot -- the page is not evolving,`);
        console.log('       or the save is not reading it. The equality above means nothing.');
      } else if (!ok) {
        const g = diffShape(A, B);
        for (const [k, n] of Object.entries(g)) console.log(`       ${String(n).padStart(6)}  ${k}`);
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  const bad = rows.filter(r => !r.ok || !r.alive);
  console.log('');
  if (bad.length) {
    console.log(`FAIL: ${bad.length} of ${rows.length} configuration(s) -- reset() does not fully define the state.`);
    console.log('The keys listed are the buffers reset() is not writing. Fix it at the source:');
    console.log('a pool buffer belongs in amr2d-gpu.mjs\'s writePoolInitialState, not in a page\'s');
    console.log('reset, so allocation and reset cannot disagree about it again.');
    process.exit(1);
  }
  console.log(`PASS: reset() is a fixed point on ${rows.length} configuration(s), and stepping moved`);
  console.log('the state in every one of them, so the comparison could have failed.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
