#!/usr/bin/env node
// D0's measurement: is pool slot assignment the ONLY live source of run-to-run
// nondeterminism in this solver? (plans/uniform-levels.md 1.2)
//
// Runs the same build N times per configuration from `reset()`, captures
// `debugSnapshotSave` at a fixed step, and compares the captures for EXACT
// equality. Exact equality is the point: it cannot be forged by a race, it is
// immune to GPU load, and it needs no statistics, no reproducibility floor and
// no same-build repeat protocol on the other side. Every piece of that
// protocol in CLAUDE.md exists because this solver is nondeterministic; this
// tool measures whether it has to be.
//
// THE PREDICTION, and it is scored as one:
//
//   levels=2 detslots=1    IDENTICAL   <- gated. the whole hypothesis.
//   levels=3 detslots=1    IDENTICAL   <- gated.
//   levels=2 detslots=0    DIFFERS     <- gated. THE DISCRIMINATION.
//   levels=3 detslots=0    reported    <- attractors; a match proves nothing.
//
// The control rungs are not optional and the third one is the important one.
// `?levels=3` already reproduces bit-for-bit *within* an attractor, so a green
// detslots=1 pair there is consistent with the flag doing nothing at all. Only
// `?levels=2` -- which today differs on every run -- can tell "the flag worked"
// from "this config happened to land in the same mode twice". If the baseline
// rung comes back IDENTICAL, the instrument is not measuring what it claims
// and no other row means anything.
//
// A FAILURE HERE IS INFORMATION, NOT A BUG TO BE FIXED. If levels=2 still
// differs with detslots=1, there is a second source of nondeterminism, and
// plans/uniform-levels.md's sequencing argument needs revisiting before any of
// D0's engineering is written.
//
//     node tools/measure-determinism.js
//     node tools/measure-determinism.js --steps=8192 --runs=3
//     node tools/measure-determinism.js --baseUrl=https://localhost:4455 --port=9444 --keep=/tmp/d0
//
// VERIFY WHICH TREE THE DEV SERVER IS SERVING first -- `ensureServer` reuses
// whatever already answers on the port, and this repo has twice discarded a
// full sweep that ran green against the wrong checkout.

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

// gate: null means "report only, do not score".
const CONFIGS = [
  { name: 'levels=2 detslots=1', q: 'levels=2&detslots=1', det: 1, gate: 'IDENTICAL' },
  { name: 'levels=3 detslots=1', q: 'levels=3&detslots=1', det: 1, gate: 'IDENTICAL' },
  { name: 'levels=2 detslots=0', q: 'levels=2',            det: 0, gate: 'DIFFERS'   },
  { name: 'levels=3 detslots=0', q: 'levels=3',            det: 0, gate: null        },
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9333, page: 'index-amr.html', steps: 4096, runs: 2, keep: null, only: null };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--runs=')) o.runs = parseInt(a.slice(7));
    else if (a.startsWith('--keep=')) o.keep = a.slice(7);
    else if (a.startsWith('--only=')) o.only = a.slice(7);
    else if (a.startsWith('--extra=')) o.extra = a.slice(8);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// The comparable content of a snapshot: every base64 field payload, in a
// stable order, hashed. Metadata that legitimately varies (timings, counters
// that are not state) is excluded by only taking string fields, which is what
// the field arrays are.
function fingerprint(snap) {
  const parts = [];
  const walk = (node, keyPath) => {
    if (node === null || node === undefined) return;
    if (typeof node === 'string') { parts.push(`${keyPath}=${node}`); return; }
    if (Array.isArray(node)) { node.forEach((v, i) => walk(v, `${keyPath}[${i}]`)); return; }
    if (typeof node === 'object') {
      for (const k of Object.keys(node).sort()) walk(node[k], `${keyPath}.${k}`);
      return;
    }
    parts.push(`${keyPath}=${node}`);
  };
  walk(snap, '');
  const joined = parts.join('\n');
  return { hash: crypto.createHash('sha256').update(joined).digest('hex'), bytes: joined.length, fields: parts.length };
}

async function captureOne(Runtime, Page, o, cfg, runIdx) {
  // --extra= appends to every configuration's URL, so a flag can be swept
  // across the whole table without a second copy of it -- the same shape as
  // validate-all.js's own --extra=.
  //
  // It was in the defaults and in the help text and NOT in parseArgs, so it
  // was rejected as an unknown argument rather than silently ignored. That is
  // the better of the two failures, and only by luck.
  const url = `${o.baseUrl}/${o.page}?${cfg.q}${o.extra ? '&' + o.extra.replace(/^&/, '') : ''}`;
  await navigateTo(Page, url);
  // waitForGlobal THROWS on timeout and returns undefined on success -- do not
  // test its return value. (Testing it inverts the check and reports a boot
  // failure on every healthy page, which cost one wrong diagnosis here.)
  await waitForGlobal(Runtime, 'window.__AMR', 60000);
  // Assert the flag actually took. A URL typo would otherwise read as a clean
  // negative result.
  const det = await ev(Runtime, 'window.__AMR.getDetSlots()');
  if (det !== cfg.det) throw new Error(`${cfg.name}: page reports detslots=${det}, expected ${cfg.det}`);
  const levels = await ev(Runtime, 'window.__AMR.getNumLevels()');

  await ev(Runtime, 'window.__AMR.reset()', 120000);
  // Timed IN THE PAGE, not around the CDP call: the round trip is noise at
  // this step count but there is no reason to measure it. This is what decides
  // whether a deterministic handout can be the DEFAULT or has to stay a
  // validation mode -- the serial loop is O(blocks) in ONE thread, once per
  // refine round, and whether that is affordable is a measurement.
  const timed = await ev(Runtime, `(async () => {
    const t0 = performance.now();
    await window.__AMR.debugStepSync(${o.steps});
    return { ms: performance.now() - t0, step: window.__AMR.getStep() };
  })()`, 900000);
  const stepsDone = timed.step;
  // Active tiles per level. A deterministic handout can land on a DIFFERENT
  // topology than a racing one, and a topology with more tiles costs more per
  // step -- so a timing delta between the two is not necessarily overhead. Do
  // not read the ms column without this one.
  const tiles = [];
  for (let m = 1; m < levels; m++) {
    tiles.push(await ev(Runtime, `window.__AMR.debugListActiveBlocks(${m}).then(a => a.length)`, 120000));
  }
  const snap = await ev(Runtime, 'window.__AMR.debugSnapshotSave()', 300000);
  const fp = fingerprint(snap);
  console.log(`  run ${runIdx + 1}: levels=${levels} steps=${stepsDone} ${timed.ms.toFixed(0).padStart(6)} ms  tiles=[${tiles.join(',')}]  hash=${fp.hash.slice(0, 16)}`);
  if (o.keep) {
    fs.mkdirSync(o.keep, { recursive: true });
    fs.writeFileSync(path.join(o.keep, `${cfg.q.replace(/[^a-z0-9]+/gi, '_')}.run${runIdx + 1}.json`), JSON.stringify(snap));
  }
  return { fp, stepsDone, ms: timed.ms, tiles };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = o.only ? CONFIGS.filter(c => c.q.includes(o.only)) : CONFIGS;
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
    for (const cfg of configs) {
      console.log(`\n[${cfg.name}] ${o.runs} runs x ${o.steps} steps`);
      const caps = [];
      for (let r = 0; r < o.runs; r++) caps.push(await captureOne(Runtime, Page, o, cfg, r));
      const first = caps[0].fp.hash;
      const identical = caps.every(c => c.fp.hash === first);
      const verdict = identical ? 'IDENTICAL' : 'DIFFERS';
      rows.push({ cfg, verdict, hashes: caps.map(c => c.fp.hash.slice(0, 16)), steps: caps[0].stepsDone,
                  msMin: Math.min(...caps.map(c => c.ms)),
                  tiles: caps[0].tiles,
                  tileSum: caps.map(c => c.tiles.reduce((a, b) => a + b, 0)) });
      console.log(`  -> ${verdict}${cfg.gate ? ` (expected ${cfg.gate})` : ' (reported, not gated)'}`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  console.log(`\nD0 determinism, ${o.steps} steps, ${o.runs} runs each\n`);
  for (const r of rows) {
    const mark = r.cfg.gate === null ? '  --  ' : (r.verdict === r.cfg.gate ? '  ok  ' : ' FAIL ');
    console.log(`${mark} ${r.cfg.name.padEnd(22)} ${r.verdict.padEnd(10)} ${r.msMin.toFixed(0).padStart(6)} ms  ${r.hashes.join(' ')}`);
  }

  // The cost of determinism, per level count. MIN of the runs, which is the
  // least contaminated sample -- this is a floor, not an average.
  console.log('\ncost of the deterministic handout (min of runs, same steps):');
  for (const lv of ['levels=2', 'levels=3']) {
    const on = rows.find(r => r.cfg.q === `${lv}&detslots=1`);
    const off = rows.find(r => r.cfg.q === lv);
    if (!on || !off) continue;
    const pct = 100 * (on.msMin - off.msMin) / off.msMin;
    const tOn = on.tileSum[0], tOff = Math.min(...off.tileSum);
    const tPct = 100 * (tOn - tOff) / tOff;
    console.log(`  ${lv.padEnd(10)} ${off.msMin.toFixed(0).padStart(6)} -> ${on.msMin.toFixed(0).padStart(6)} ms   ${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%` +
      `     tiles [${off.tiles.join(',')}] -> [${on.tiles.join(',')}]   ${tPct >= 0 ? '+' : ''}${tPct.toFixed(1)}%`);
  }

  const gated = rows.filter(r => r.cfg.gate !== null);
  const bad = gated.filter(r => r.verdict !== r.cfg.gate);

  console.log('');
  const discrim = rows.find(r => r.cfg.q === 'levels=2');
  if (discrim && discrim.verdict === 'IDENTICAL') {
    console.log('  THE DISCRIMINATION RUNG CAME BACK IDENTICAL. `levels=2` without detslots is');
    console.log('  supposed to differ on every run; if it does not, this instrument is not');
    console.log('  measuring nondeterminism and NO other row above means anything. Check the');
    console.log('  snapshot actually carries the field, and that the runs really restarted.');
    process.exit(1);
  }

  if (bad.length) {
    console.log(`FAIL: ${bad.map(r => r.cfg.name).join(', ')} did not match the prediction.`);
    console.log('That is information, not a bug: it says slot assignment is not the only source');
    console.log('of nondeterminism, and plans/uniform-levels.md 1.1 needs revisiting before any');
    console.log('of D0 is engineered. Do not "fix" it by widening the comparison.');
    process.exit(1);
  }
  console.log('PASS: deterministic handout makes repeated runs bit-identical, and the');
  console.log('baseline still differs. Slot assignment is the source.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
