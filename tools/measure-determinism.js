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
// READ THE BASELINE RUNG AT `--runs=4`, NOT `--runs=2`. Measured 2026-09-18 on
// the DEFAULT configuration: two runs came back identical (`b21dba4e...`
// twice), and four runs of the same build immediately gave three distinct
// hashes. The race lands the same way twice often enough that a 2-run baseline
// is not evidence it has stopped racing. That matters because the next
// paragraph is a case where four runs DID all agree -- the two observations
// look the same at --runs=2 and are not the same thing at all.
//
// AND ON index-amr.html THE levels=2 BASELINE RUNG NO LONGER DISCRIMINATES,
// MEASURED -- IN THE DEFAULT CONFIGURATION. With level 1 managed by
// amr_manage_pool.wgsl (plans/uniform-levels.md U5-4) the racing free list
// produced the SAME assignment in 4 of 4 runs at `levels=2`, so that row comes
// back IDENTICAL and this tool exits nonzero. That was first seen under
// `--extra=rootpool=1`; U7-5 then made rootpool=1 the DEFAULT, so
// `node tools/measure-determinism.js` with no arguments is now that
// configuration and now exits 1 on a healthy build. The gate is deliberately
// NOT relaxed for it: the row is doing its job by saying it can no longer tell
// "the flag worked" from "nothing raced". Read `levels=3 detslots=0` instead,
// which still DIFFERS and is what keeps the instrument honest there.
//
// `--page=index-cylinder-amr.html` HAS NO SUCH PROBLEM, measured 2026-09-18:
// both detslots=0 rungs differ there, `levels=2` with four distinct hashes in
// four runs. If you want a run of this tool whose baseline rung still means
// something, that is the page to point it at.
//
// ── WHAT "IDENTICAL" CAN AND CANNOT ESTABLISH ──────────────────────────────
//
// IT CANNOT ESTABLISH DETERMINISM, AT ANY NUMBER OF RUNS. A nondeterministic
// process can land the same way N times for any N; more runs only shrink the
// probability of missing a rare divergence, and never to zero. The two row
// types here are one-sided, in OPPOSITE directions, and it is worth being
// precise about which way each cuts:
//
//   detslots=1, expected IDENTICAL   REFUTABLE ONLY. One DIFFERS disproves the
//                                    claim outright. No number of IDENTICALs
//                                    proves it.
//   detslots=0, expected DIFFERS     CONFIRMABLE ONLY. One DIFFERS proves
//                                    nondeterminism outright. IDENTICAL proves
//                                    nothing -- see the paragraph above about
//                                    reading that rung at --runs=4.
//
// So this tool is a REGRESSION DETECTOR and a refutation test, not a proof.
// What actually supports "the deterministic handout is deterministic" is the
// ARGUMENT ABOUT THE KERNEL -- amr_manage.wgsl's DET_SLOTS path is one thread,
// in dispatch order, with no atomics in the decision -- and this run's job is
// to fail to contradict it. Treat a green run as "no divergence observed under
// these conditions", and say that rather than "it is deterministic".
//
// AND THESE CONDITIONS ARE THE LOW-POWER ONES, which is the part most worth
// knowing. The repeats run back to back, on an idle GPU, each from a fresh
// navigate -- i.e. as nearly identical as the harness can make them. A race
// whose outcome depends on scheduling is LEAST likely to show up that way. If
// you need more power, vary something that moves timing (contend the GPU,
// interleave other work between runs) rather than only raising --runs.
//
// A FAILURE HERE IS INFORMATION, NOT A BUG TO BE FIXED. If levels=2 still
// differs with detslots=1, there is a second source of nondeterminism, and
// plans/uniform-levels.md's sequencing argument needs revisiting before any of
// D0's engineering is written.
//
//     node tools/measure-determinism.js
//     node tools/measure-determinism.js --steps=8192 --runs=3
//     node tools/measure-determinism.js --page=index-cylinder-amr.html --runs=4
//     node tools/measure-determinism.js --baseUrl=https://localhost:4455 --port=9444 --keep=/tmp/d0
//
// `--page=` carries `--global=` with it from a table (index-amr and
// index-reentry-amr expose `window.__AMR`, the cylinder/TGV/channel harnesses
// expose `window.__CYL`), and it carries the SCORING with it too: only a page
// listed in GATES_BY_PAGE is gated, because the expectations below are
// index-amr.html's measured behaviour and not a law. Everything else reports
// and exits 0.
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
//
// THE GATE COLUMN IS index-amr.html's, AND IT IS NOT A UNIVERSAL PROPERTY.
// Every entry below was MEASURED on the falling-card page; "levels=2 without
// detslots DIFFERS" in particular is a fact about that page's allocator under
// that page's flow, and U5-4 already recorded it ceasing to hold there once
// level 1 became quad-managed. On a page this tool has not been run against,
// the predictions are unknown -- see GATES_BY_PAGE, which reports rather than
// scores until someone has measured the page and written its row down.
const CONFIGS = [
  { name: 'levels=2 detslots=1', q: 'levels=2&detslots=1', det: 1, gate: 'IDENTICAL' },
  { name: 'levels=3 detslots=1', q: 'levels=3&detslots=1', det: 1, gate: 'IDENTICAL' },
  { name: 'levels=2 detslots=0', q: 'levels=2',            det: 0, gate: 'DIFFERS'   },
  { name: 'levels=3 detslots=0', q: 'levels=3',            det: 0, gate: null        },
];

// Which pages have had their rows measured, and therefore may be SCORED.
// A page absent from this table runs report-only: every row prints, nothing
// FAILs, and the exit code is 0 with a loud note saying why. Measure first,
// gate after -- adding a page here without a measurement behind it is exactly
// how this project collected three vacuous gates.
//
// index-cylinder-amr.html: MEASURED 2026-09-18 (D1-a), 4 runs x 4096 steps on
// the shipped default configuration, and it satisfies the WHOLE column above:
//
//   levels=2 detslots=1   IDENTICAL  77797c55ff9a04d7  (4 of 4)
//   levels=3 detslots=1   IDENTICAL  0c7f6ebeb91d10fe  (4 of 4)
//   levels=2 detslots=0   DIFFERS    four distinct hashes in four runs
//   levels=3 detslots=0   DIFFERS    two attractors, 3+1
//
// THIS PAGE DISCRIMINATES BETTER THAN index-amr.html DOES TODAY, which is the
// opposite of what D1 expected and is worth stating plainly. The card page's
// `levels=2 detslots=0` rung came back IDENTICAL over 4 runs on the same build
// -- the vacuity its own header predicted for quad-managed level 1, now that
// U7-5 has made that the default -- so on THAT page only `levels=3` still
// keeps the instrument honest. Here both rungs race, visibly.
//
// THESE HASHES MOVE WHEN THE SNAPSHOT FORMAT MOVES, and that is not a
// regression. `fingerprint` walks the WHOLE snapshot object, so a new key
// changes every hash in the table while changing no physics. U7-6c added
// `root` (formatVersion 6) and did exactly that:
//
//   levels=2 detslots=1   8ddd3ff2f84a697f (v5)  ->  e0b1b22bbe47cbfb (v6)
//   levels=3 detslots=1   3d80fa737af6bf9e (v5)  ->  4705ea2b06226c8b (v6)
//
// Both re-measured 2026-09-22, IDENTICAL over four runs each. Check the
// format version before treating a moved hash here as a physics change.
const GATES_BY_PAGE = {
  'index-amr.html': true,
  'index-cylinder-amr.html': true,
};

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9333, page: 'index-amr.html', steps: 4096, runs: 2, keep: null, only: null, global: null, gate: null };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--runs=')) o.runs = parseInt(a.slice(7));
    else if (a.startsWith('--keep=')) o.keep = a.slice(7);
    else if (a.startsWith('--only=')) o.only = a.slice(7);
    else if (a.startsWith('--extra=')) o.extra = a.slice(8);
    else if (a.startsWith('--global=')) o.global = a.slice(9);
    else if (a === '--gate') o.gate = true;
    else if (a === '--no-gate') o.gate = false;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  // THE DEBUG SURFACE IS NOT THE SAME NAME ON EVERY PAGE, and it does not
  // follow the page name either: index-amr and index-reentry-amr expose
  // `window.__AMR`, while index-cylinder-amr, index-tgv-amr and
  // index-channel-amr expose `window.__CYL` -- the harness pages kept the
  // cylinder harness's original name as they were derived from it. Defaulted
  // from a table rather than guessed from the URL, and overridable, because a
  // wrong guess here reads as "WebGPU init failed" and sends you looking at
  // the page. Same table, same reasoning, as tools/measure-pool-peaks.js.
  if (!o.global) {
    const AMR_PAGES = ['index-amr.html', 'index-reentry-amr.html'];
    o.global = AMR_PAGES.includes(o.page) ? 'window.__AMR' : 'window.__CYL';
  }
  if (o.gate === null) o.gate = !!GATES_BY_PAGE[o.page];
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
  const G = o.global;
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
  await waitForGlobal(Runtime, G, 60000);
  // Assert the flag actually took. A URL typo would otherwise read as a clean
  // negative result.
  const det = await ev(Runtime, `${G}.getDetSlots()`);
  if (det !== cfg.det) throw new Error(`${cfg.name}: page reports detslots=${det}, expected ${cfg.det}`);
  const levels = await ev(Runtime, `${G}.getNumLevels()`);

  // PAUSE THE PAGE BEFORE reset(), and this is load-bearing rather than tidy.
  // Every AMR page boots with `liveMode = true` and an rAF loop that steps
  // STEPS_PER_FRAME at a time, so between the reset() round trip and the
  // debugStepSync() one, an uncontrolled number of live frames can land --
  // stepping the sim, advancing macroStepCounter and running refine rounds
  // that this tool did not ask for and does not count.
  //
  // IT WAS ACTIVE, MEASURED 2026-09-18: index-cylinder-amr.html reported
  // `steps=4160` for a 4096-step request (exactly one 64-step frame of
  // uncounted stepping) while index-amr.html reported 4096. The two pages were
  // NOT being measured the same way, and reading a detslots=1 DIFFERS on the
  // cylinder page as "a second source of nondeterminism in the solver" would
  // have been a conclusion about the harness. The step count in each run line
  // is the guard: it must equal the request.
  await ev(Runtime, `${G}.setLive(false)`);
  await ev(Runtime, `${G}.reset()`, 120000);
  // Timed IN THE PAGE, not around the CDP call: the round trip is noise at
  // this step count but there is no reason to measure it. This is what decides
  // whether a deterministic handout can be the DEFAULT or has to stay a
  // validation mode -- the serial loop is O(blocks) in ONE thread, once per
  // refine round, and whether that is affordable is a measurement.
  const timed = await ev(Runtime, `(async () => {
    const t0 = performance.now();
    await ${G}.debugStepSync(${o.steps});
    return { ms: performance.now() - t0, step: ${G}.getStep() };
  })()`, 900000);
  const stepsDone = timed.step;
  // Active tiles per level. A deterministic handout can land on a DIFFERENT
  // topology than a racing one, and a topology with more tiles costs more per
  // step -- so a timing delta between the two is not necessarily overhead. Do
  // not read the ms column without this one.
  const tiles = [];
  for (let m = 1; m < levels; m++) {
    tiles.push(await ev(Runtime, `${G}.debugListActiveBlocks(${m}).then(a => a.length)`, 120000));
  }
  const snap = await ev(Runtime, `${G}.debugSnapshotSave()`, 300000);
  const fp = fingerprint(snap);
  // See setLive(false) above. An over-count here means live frames stepped the
  // sim behind this tool's back and every hash below is of an uncontrolled
  // state, so it is an error rather than a note.
  if (stepsDone !== o.steps) {
    throw new Error(`${cfg.name}: asked for ${o.steps} steps, page reports ${stepsDone}` +
      ` -- the page stepped behind the harness (live rAF frames?), so the capture is not comparable`);
  }
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
      console.log(`  -> ${verdict}${o.gate && cfg.gate ? ` (expected ${cfg.gate})` : ' (reported, not gated)'}`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  console.log(`\nD0 determinism, ${o.page}, ${o.steps} steps, ${o.runs} runs each\n`);
  for (const r of rows) {
    const mark = (!o.gate || r.cfg.gate === null) ? '  --  ' : (r.verdict === r.cfg.gate ? '  ok  ' : ' FAIL ');
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

  const gated = o.gate ? rows.filter(r => r.cfg.gate !== null) : [];
  const bad = gated.filter(r => r.verdict !== r.cfg.gate);

  console.log('');

  // REPORT-ONLY, because CONFIGS' predictions are index-amr.html's and this is
  // not that page. Printing the rows and exiting 0 is the honest outcome: the
  // alternative is to score a page against another page's measured behaviour,
  // which produces a red cell that means nothing and a green one that means
  // less. Write the page's rows into GATES_BY_PAGE once they are measured.
  if (!o.gate) {
    console.log(`  REPORTED, NOT GATED: ${o.page} has no measured row in GATES_BY_PAGE.`);
    console.log('  The expectations in CONFIGS were measured on index-amr.html and do not');
    console.log('  transfer -- U5-4 records them ceasing to hold on that page itself when the');
    console.log('  allocator changed. Read the verdicts above, record what this page does, then');
    console.log('  add it to GATES_BY_PAGE. `--gate` forces scoring for the measuring run.');
    process.exit(0);
  }

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
  console.log('NOT REFUTED: the deterministic handout produced bit-identical runs, and the');
  console.log('baseline still differs, so the comparison can still see nondeterminism.');
  console.log('');
  console.log('That is the strongest thing this run can say. IDENTICAL over N runs does not');
  console.log('establish determinism at any N -- see this file\'s header. What it rules out is');
  console.log('a REGRESSION large enough to surface in N tries under these conditions.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
