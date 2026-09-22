#!/usr/bin/env node
// Does a snapshot round trip in the MIDDLE of a run change the run?
// (plans/uniform-levels.md U7-6a)
//
// The protocol is one sentence: step 2N from reset and fingerprint; then step
// N, save, load what you just saved, step N more, and fingerprint again. If
// `debugSnapshotSave`/`debugSnapshotLoad` are exact, the two fingerprints are
// BIT-IDENTICAL, because the second leg performed the same 2N steps of
// evolution with a serialise/deserialise in the middle of it.
//
// ── WHY THIS TOOL EXISTS, AND WHY render-levels COULD NOT DO ITS JOB ────────
//
// tools/lib/render-levels.js already round-trips a snapshot: it saves a
// baseline, perturbs a level, restores, and requires the picture to come back
// bit-for-bit. That check is real and it is not this one. It renders
// IMMEDIATELY after the load, so it only ever scores the state the RENDERER
// reads -- the velocity pools and the indirection. It never steps, so it never
// asks the ALLOCATOR anything.
//
// That is exactly the gap U7-6a was written to close. Before it, a load
// rebuilt level 1's free list at BLOCK granularity from `slotToBlock`; under
// `?rootpool=1` level 1 is quad-allocated and that list holds QUAD indices.
// Slot `q` and quad `q` are different things, so the restored list handed out
// overlapping quads on the next refine -- with no thrown error, no NaN, and a
// picture that looked right until refinement moved. render-levels passed
// through all of it, on both pages, because the defect needs a refine round to
// express itself and render-levels never encodes one.
//
// So the fingerprint is taken after N MORE STEPS, which is at least one
// refinement round (?refineEvery= is well under N), and that is the whole
// design. A corrupted free list lands different slots, which moves the
// fingerprint, which fails the row.
//
// ── THE CONTROLS ARE NOT OPTIONAL ──────────────────────────────────────────
//
// A bit-identical comparison is also what a comparison that cannot see
// anything produces, so two controls run alongside every gated row:
//
//   stale     load a snapshot taken at a DIFFERENT step and require the
//             outcome to DIFFER. This is what proves the load is LOAD-BEARING:
//             without it, a `debugSnapshotLoad` that silently did nothing at
//             all would pass every gated row, since the leg would simply have
//             stepped 2N times uninterrupted.
//   refuse    hand a page a snapshot whose level 1 was allocated the OTHER
//             way and require it to THROW. The formats are structurally
//             compatible enough to load and corrupt silently, which is the
//             failure U7-6a's marker exists to make loud -- so "it refused" is
//             a result, not an error.
//
// `?detslots=1` throughout (plans/uniform-levels.md D0): with a racing free
// list the reference leg is not reproducible against itself, and nothing below
// would mean what it says. The tool asserts the flag took rather than trusting
// the URL.
//
//     node tools/validate-snapshot-roundtrip.js
//     node tools/validate-snapshot-roundtrip.js --steps=1024
//     node tools/validate-snapshot-roundtrip.js --baseUrl=https://localhost:4455 --port=9444
//
// VERIFY WHICH TREE THE DEV SERVER IS SERVING first -- `ensureServer` reuses
// whatever already answers on the port, and this repo has twice discarded a
// full sweep that ran green against the wrong checkout.

const path = require('path');
const crypto = require('crypto');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

// `quad` is what level 1's allocator is under each configuration, and it is
// the thing the cross-granularity control pairs rows on.
// BOTH LEGS NAME THE FLAG EXPLICITLY, and that is not redundancy. These rows
// leaned on `?rootpool=` defaulting to 0 until U7-5 flipped it, at which point
// the "rootpool=0" legs were silently running the quad allocator and testing
// the same thing twice. `open()`'s assertion caught it rather than the suite
// going green on four copies of one configuration -- which is the whole reason
// that assertion is there.
const CONFIGS = [
  { name: 'levels=2 rootpool=0', q: 'levels=2&detslots=1&rootpool=0', quad: false },
  { name: 'levels=3 rootpool=0', q: 'levels=3&detslots=1&rootpool=0', quad: false },
  { name: 'levels=2 rootpool=1', q: 'levels=2&detslots=1&rootpool=1', quad: true  },
  { name: 'levels=3 rootpool=1', q: 'levels=3&detslots=1&rootpool=1', quad: true  },
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9337, page: 'index-amr.html', steps: 2048, extra: '' };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--extra=')) o.extra = a.slice(8);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  // useB is NOT in the snapshot -- debugSnapshotLoad forces it to false -- so
  // the save has to happen at a macro-step count where the uninterrupted leg
  // holds that same phase. An odd N would compare two different phases and
  // report a defect that is the harness's.
  if (o.steps % 2 !== 0) { console.error('--steps must be EVEN (useB phase; see the source)'); process.exit(2); }
  return o;
}

async function ev(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// Same shape as measure-determinism.js's: every leaf of the snapshot, in a
// stable key order, hashed. Deliberately includes the indirection arrays and
// not only the field payloads -- an allocator that hands out the wrong slots
// shows up there first.
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
  // `step` legitimately differs between a leg that loaded and one that did
  // not, because the load restores the saved step counter. It is bookkeeping,
  // not state, and including it would fail every row for the one reason the
  // protocol guarantees.
  const { step, ...rest } = snap;
  walk(rest, '');
  const joined = parts.join('\n');
  return { hash: crypto.createHash('sha256').update(joined).digest('hex'), parts };
}

// WHICH FIELDS MOVED, not just "they moved". A round-trip gate that reports
// one bit is a gate you cannot act on: the interesting question is always
// whether the FIELD payloads differ (a physics difference) or only the
// indirection (a slot permutation, which is not one). Collapses each leaf path
// to its array-free prefix -- `.pools[1].blockSlot[37]` becomes
// `.pools[1].blockSlot` -- and counts the leaves that disagree.
function diffFields(a, b) {
  const groups = new Map();
  const n = Math.max(a.parts.length, b.parts.length);
  for (let i = 0; i < n; i++) {
    if (a.parts[i] === b.parts[i]) continue;
    const key = String(a.parts[i] ?? b.parts[i]).split('=')[0].replace(/\[\d+\]/g, '[]');
    groups.set(key, (groups.get(key) || 0) + 1);
  }
  return [...groups.entries()].sort((x, y) => y[1] - x[1]);
}

async function open(Runtime, Page, o, cfg) {
  const url = `${o.baseUrl}/${o.page}?${cfg.q}${o.extra ? '&' + o.extra.replace(/^&/, '') : ''}`;
  await navigateTo(Page, url);
  await waitForGlobal(Runtime, 'window.__AMR', 60000);
  // Assert the flags took. A URL typo would otherwise read as a clean pass on
  // a configuration this tool never visited.
  const det = await ev(Runtime, 'window.__AMR.getDetSlots()');
  if (det !== 1) throw new Error(`${cfg.name}: page reports detslots=${det}, expected 1`);
  const root = await ev(Runtime, 'window.__AMR.getRootPool() ? 1 : 0');
  if (!!root !== cfg.quad) throw new Error(`${cfg.name}: page reports rootPool=${root}, expected ${cfg.quad ? 1 : 0}`);
  // PAUSE BEFORE ANYTHING ELSE, and this is not tidiness -- it is what makes
  // the two legs comparable at all.
  //
  // The page boots `liveMode = true` and its rAF loop steps STEPS_PER_FRAME at
  // a time. `debugStepSync` pauses on entry, so only the FIRST leg after a
  // navigate is exposed -- and that is the reference leg. It absorbs an
  // uncontrolled number of live frames between the navigate, its reset() and
  // its first debugStepSync(); the round-trip leg then runs in the same,
  // already-paused page and absorbs none. The two legs were starting from
  // different amounts of uncounted evolution, and the gated rows have been
  // comparing that difference rather than the round trip.
  //
  // Measured 2026-09-22: with both legs paused, `levels=2 rootpool=1`
  // round-trips blockSlot, slotToBlock, parentSlot, quadrant and freeList
  // EXACTLY -- where the unpaused tool reported 32 parentSlot entries moved.
  // Same defect `tools/measure-determinism.js` carried until D1-a, same fix.
  await ev(Runtime, 'window.__AMR.setLive(false)');
  return url;
}

// Step 2N from reset, uninterrupted. The reference every row is scored against.
//
// IT FINGERPRINTS AT N AS WELL, AND THAT IS A GUARD, NOT A BONUS. Every row
// below is an EQUALITY, so a snapshot that carries nothing -- a page that is
// not evolving, or a save whose copies never landed -- passes all of them
// trivially. Requiring the capture at N to differ from the one at 2N is the
// cheapest statement of "this instrument can see the simulation at all".
//
// It is here because it was needed: a first cut of U7-6a's save copied a
// quad-allocated level's free list at per-slot length, overrunning the source.
// That is a command-encoder validation error, which drops the WHOLE command
// buffer -- so every other copy in the same submit silently produced an
// all-zero staging buffer, and `?levels>=3` returned a snapshot with a zeroed
// dense grid and a zeroed card. Three of the four gated rows went green on it,
// because zeros equal zeros.
async function reference(Runtime, o) {
  await ev(Runtime, 'window.__AMR.reset()', 120000);
  await ev(Runtime, `window.__AMR.debugStepSync(${o.steps})`, 900000);
  const half = fingerprint(await ev(Runtime, 'window.__AMR.debugSnapshotSave()', 300000));
  await ev(Runtime, `window.__AMR.debugStepSync(${o.steps})`, 900000);
  const fp = fingerprint(await ev(Runtime, 'window.__AMR.debugSnapshotSave()', 300000));
  fp.step = await ev(Runtime, 'window.__AMR.getStep()');
  // See open()'s setLive note: an over-count means the page stepped behind
  // this tool and neither leg is what it claims to be.
  if (fp.step !== 2 * o.steps) {
    throw new Error(`reference leg ran ${fp.step} steps, expected ${2 * o.steps} -- the page stepped behind the harness`);
  }
  fp.alive = half.hash !== fp.hash;
  return fp;
}

// Step N, save, load it back, step N. `saveAt` lets the control save at a
// DIFFERENT step from the one it loads at, which is how the comparison is
// shown to be able to fail.
async function roundTrip(Runtime, o, { staleBy = 0 } = {}) {
  await ev(Runtime, 'window.__AMR.reset()', 120000);
  await ev(Runtime, `window.__AMR.debugStepSync(${o.steps})`, 900000);
  // The control steps ON past the save point and re-saves, so the state it
  // restores is a real one this run passed through -- just not the one the
  // reference leg is at. A garbage snapshot would fail for the wrong reason.
  if (staleBy) await ev(Runtime, `window.__AMR.debugStepSync(${staleBy})`, 900000);
  await ev(Runtime, '(async () => { window.__RT = await window.__AMR.debugSnapshotSave(); return 1; })()', 300000);
  await ev(Runtime, 'window.__AMR.debugSnapshotLoad(window.__RT)', 300000);
  await ev(Runtime, `window.__AMR.debugStepSync(${o.steps})`, 900000);
  const fp = fingerprint(await ev(Runtime, 'window.__AMR.debugSnapshotSave()', 300000));
  fp.step = await ev(Runtime, 'window.__AMR.getStep()');
  return fp;
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
  let failed = 0;

  try {
    console.log(`  ${o.steps} steps per leg (${o.steps * 2} total), ${o.page}\n`);
    console.log('  GATED -- a round trip in the middle of a run must not change the run');
    for (const cfg of CONFIGS) {
      await open(Runtime, Page, o, cfg);
      const ref = await reference(Runtime, o);
      if (!ref.alive) {
        failed++;
        console.log(`  DEAD ${cfg.name.padEnd(22)} the snapshot did not change over ${o.steps} steps -- the page is not`);
        console.log('         evolving, or the save is not reading it. Nothing below would mean what it says.');
        rows.push({ cfg, kind: 'roundtrip', ok: false });
        continue;
      }
      const rt = await roundTrip(Runtime, o);
      const ok = ref.hash === rt.hash;
      if (!ok) failed++;
      console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${cfg.name.padEnd(22)} ref ${ref.hash.slice(0, 16)}  roundtrip ${rt.hash.slice(0, 16)}`);
      if (!ok) for (const [k, n] of diffFields(ref, rt)) console.log(`         ${String(n).padStart(8)}  ${k}`);
      rows.push({ cfg, kind: 'roundtrip', ok });
    }

    console.log('\n  CONTROL -- must DIFFER, or the rows above prove nothing');
    for (const cfg of CONFIGS) {
      await open(Runtime, Page, o, cfg);
      const ref = await reference(Runtime, o);
      const stale = await roundTrip(Runtime, o, { staleBy: 64 });
      const ok = ref.hash !== stale.hash;
      if (!ok) failed++;
      console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${cfg.name.padEnd(22)} loading a snapshot 64 steps later ${ok ? 'moved' : 'DID NOT MOVE'} the outcome (ref@${ref.step}, stale@${stale.step})`);
      rows.push({ cfg, kind: 'stale', ok });
    }

    // A STUB IS ENOUGH HERE, AND DELIBERATELY SO. The granularity marker is
    // checked before debugSnapshotLoad writes a single buffer, so a stub
    // carrying the page's own geometry and the OPPOSITE marker reaches it by
    // exactly the path a full capture would -- at a few bytes instead of the
    // several megabytes a real snapshot costs to ship twice over CDP. What is
    // being scored is the marker, not the payload.
    console.log('\n  CONTROL -- a level-1 granularity mismatch must REFUSE, not load');
    for (const cfg of CONFIGS) {
      await open(Runtime, Page, o, cfg);
      const threw = await ev(Runtime, `(async () => {
        const snap = await window.__AMR.debugSnapshotSave();
        snap.pools[1].quadAlloc = ${!cfg.quad};
        if (!${!cfg.quad}) delete snap.pools[1].quadAlloc;
        try { await window.__AMR.debugSnapshotLoad(snap); return ''; }
        catch (e) { return String(e.message || e); }
      })()`, 300000);
      const ok = /QUAD|per-block/.test(threw);
      if (!ok) failed++;
      console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${cfg.name.padEnd(22)} ${threw ? threw.slice(0, 96) : 'LOADED SILENTLY'}`);
      rows.push({ cfg, kind: 'refuse', ok });
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  console.log('');
  if (failed) {
    console.log(`FAIL: ${failed} of ${rows.length} row(s) did not hold.`);
    process.exit(1);
  }
  console.log('PASS: snapshots round-trip exactly at both level-1 granularities, the');
  console.log('comparison can fail, and a granularity mismatch is refused rather than loaded.');
}

main().catch(e => { console.error(e); process.exit(1); });
