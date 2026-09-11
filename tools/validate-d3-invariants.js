#!/usr/bin/env node
// 3D AMR structural invariants (plans/3D.md M4.2, sec 7 risk #2). The 3D
// sibling of tools/validate-amr-invariants.js.
//
// WHY IT EXISTS BEFORE THE THING IT GUARDS. plans/3D.md ranks the pool
// manager and 3D 2:1 balance as risk #2 and records that the 2D versions of
// manage/manage_pool carry three separately-documented, live-verified balance
// bugs -- all of which sat under a green suite, because a checker written
// AFTER a manager gets written to agree with it. This is the independent
// statement of what M4.2's manager must achieve, written while there is
// still nothing to be tempted to agree with.
//
// WHY IT CHECKS PERIODICALLY, not once at the end. A violation that appears
// and heals is still a violation -- some cell was stepped against a
// neighbour two levels away while it lasted -- and an end-state check cannot
// see it. Same reasoning as the 2D tool's own header.
//
// WHAT IT ASSERTS
//
//   2:1 balance   every leaf tile's face neighbour is within one level.
//                 REPORTED AS VACUOUS at ?levels=2, and that is not a
//                 formality: with a single refined level a leaf's neighbour
//                 is level 1 or level 0 and both are legal, so the check
//                 CANNOT fail there. It says so rather than presenting a
//                 green tick, because "the checker ran" and "the invariant
//                 holds" are different claims. The depth-3 configs are what
//                 make it real -- `box3`/`bar3`/`body3` on a tree the HOST
//                 built, and `body3-dynamic`/`drift3` on one the MANAGER
//                 rebuilds every ?manageEvery= steps (M5.5a's cascade
//                 kernel, M5.5b's per-level allocator).
//   geometry      no coarse cell within ?margin= of the body sits in an
//                 unrefined block, checked at CELL granularity against the
//                 SDF -- an independent route from the block-corner
//                 sampling refineNearBody uses to BUILD the set. Skipped,
//                 never passed, where refinement is not geometry-forced.
//   pool          blockSlot and slotToBlock are inverses, and inUse + free
//                 equals the slot budget. Trivial of an uploaded static map;
//                 not trivial once M4.2b's manager writes them, and cross-
//                 checking the two is exactly how the 2D free-list race was
//                 confirmed -- they disagreed for the colliding slot.
//   finite        a NaN/blowup smoke check, so a run that has destroyed
//                 itself is not reported as structurally sound.
//
// Owns the whole lifecycle (HTTPS server + a dedicated debug-port Chrome if
// neither is up, one tab reused via Page.navigate), like tools/validate-3d.js.
//
//   node tools/validate-d3-invariants.js
//   node tools/validate-d3-invariants.js --steps=4000 --checkEvery=250
//   node tools/validate-d3-invariants.js --configs=body

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  steps: 2000, checkEvery: 250, configs: null, extra: '', timeout: 600, keepOpen: false,
};

// One row per refinement geometry that has a distinct structural shape. The
// interface coupling is not swept here on purpose: these are statements about
// the BLOCK TREE, which explode and interp share exactly.
const CONFIGS = [
  { name: 'box', url: 'scenario=beltrami&n=48&tau=0.8&u0=0.04&q=19&live=0&levels=2&rb=4&refine=box&boxfrac=0.5&interface=explode' },
  { name: 'bar', url: 'scenario=beltrami&n=48&tau=0.8&u0=0.04&q=19&live=0&levels=2&rb=4&refine=bar&boxfrac=0.5&interface=explode' },
  // The only geometry-forced one, so the only one where the coverage check
  // is a check rather than a skip.
  { name: 'body', url: 'scenario=sphere&n=16&re=20&u0=0.05&q=19&bounceback=1&live=0&levels=2&rb=4&refine=body&interface=explode' },
  // The same geometry with the M4.2b-i manager writing blockSlot/slotToBlock
  // from a kernel every ?manageEvery= steps. The pool check is the one that
  // earns its keep here: the body is pinned, so the manager decides "no
  // change" and the FIELD is bit-identical either way -- which means the
  // field gates cannot see the allocator at all.
  { name: 'body-dynamic', url: 'scenario=sphere&n=16&re=20&u0=0.05&q=19&bounceback=1&live=0&levels=2&rb=4&refine=body&interface=explode&dynamic=1&manageEvery=4' },
  // THE TWO THAT PROVE THE MANAGER RAN. `body-dynamic` above shows the pool
  // unchanged, which is equally consistent with a manager that never
  // executed -- so these give it a criterion that DISAGREES with the initial
  // set (?manageMargin=) and require the pool to actually move.
  //
  // They are STRUCTURAL only: releasing a slot without restricting it, and
  // handing one out without initializing it, both leave the FIELD wrong
  // until M4.2b-ii. What is being checked is the allocator -- that
  // blockSlot and slotToBlock stay mutual inverses and the budget balances
  // while slots are changing hands, which is precisely the property the 2D
  // free-list race broke.
  { name: 'body-coarsen', expectInUse: 'decrease', skipCoverage: true, steps: 8,
    url: 'scenario=sphere&n=16&re=20&u0=0.05&q=19&bounceback=1&live=0&levels=2&rb=4&refine=body&interface=explode&dynamic=1&manageEvery=1&manageMargin=0.5' },
  // Deliberately asks for more than the headroom allows, so this also
  // exercises the out-of-slots path: the pop must restore the counter and
  // leave the block coarse rather than corrupt the list. Expect it to
  // saturate at the budget with free == 0.
  // NO fieldWrongByDesign: M4.2b-ii initializes a newly-allocated tile from
  // the coarse field, so allocating 80 of them must no longer destroy the
  // run. Before it landed this config blew up and was flagged as expected;
  // the flag coming off is the gate.
  // M4.2b-iii: the shell has to FOLLOW a body that moves. The `drift`
  // scenario gives a free sphere a constant velocity with the fluid force
  // off, so its trajectory is exactly x0 + v t and any coverage failure is
  // the manager rather than the flow. This is the one config whose coverage
  // check is a real gate on dynamic refinement: the initial shell is built
  // around x0, and by the end the body is many blocks away, so coverage can
  // only still hold if the set actually moved.
  //
  // expectBboxMove: inUse alone proves nothing here -- a translating sphere
  // refines as many blocks ahead as it coarsens behind, so the COUNT is
  // constant while every tile changes hands. The refined set's bounding box
  // is what moves.
  // The body travels u0 * steps = 0.02 * 1200 = 24 coarse cells = 6 blocks
  // at RB=4, so the shell's leading edge must move about that far. 4 is a
  // floor with margin, not a prediction: the point is that it moved at all.
  { name: 'drift', expectBboxMove: 4, steps: 1200, startsAtRest: true,
    url: 'scenario=drift&n=24&live=0&levels=2&rb=4&refine=body&margin=2&interface=explode&dynamic=1&manageEvery=4' },
  // THE M5.5b GATE: a moving body at DEPTH, where the allocator has to run
  // at every level and in the order the tree requires. It is the row where
  // three things are real at once and none of them is real without the
  // others:
  //
  //   coverage   at the FINEST level, against a body that has moved -- so
  //              the manager must have refined ahead at level 2, which it
  //              can only do if level 1 was refined around it first
  //              (cascade21's closure, allocated coarsest-first).
  //   2:1        a genuine three-level tree, rebuilt from scratch every
  //              ?manageEvery= steps rather than uploaded once by the host.
  //              This is the first config where the balance check is a gate
  //              on the MANAGER and not on refineHierarchy.
  //   finite     a tile born or absorbed at depth goes through the
  //              pool-parent fill and drain. Get either wrong and the field
  //              blows up, which is exactly what an uninitialized tile did
  //              at depth 2 before M4.2b-ii.
  //
  // Smaller than `drift` (n=12, not 24) because a depth-3 shell around the
  // same sphere is ~270 level-2 tiles at 131 KB each; the claim is about the
  // allocator, and it does not get truer with more cells. The body still
  // travels u0 * steps = 0.02 * 600 = 12 coarse cells, which is 6 blocks at
  // the FINEST level -- where a block spans 2 L0 cells at RB=4.
  { name: 'drift3', expectBboxMove: 4, steps: 600, startsAtRest: true,
    url: 'scenario=drift&n=12&live=0&levels=3&rb=4&refine=body&margin=2&interface=explode&dynamic=1&manageEvery=4' },
  // DEPTH 3 (plans/3D.md M5.3). These are the configs that make the 2:1 and
  // ring-parent checks REAL for the first time -- at ?levels=2 both are
  // vacuous for their own separate reasons, and no amount of dynamic
  // refinement at one level changes that. Static refinement is enough: the
  // claim is that refineHierarchy's tree survives the trip to the GPU and
  // back, which is a different claim from the host-side unit tests.
  { name: 'box3', steps: 200,
    url: 'scenario=beltrami&n=16&tau=0.8&u0=0.04&q=19&live=0&levels=3&rb=4&refine=box&boxfrac=0.5&interface=explode' },
  // The depth-3 row where BOTH new checks are real: geometry coverage at the
  // finest level (M5.4's hard requirement -- the body must live entirely
  // there) and the criterion-and-closure chain against the host. Static:
  // ?dynamic=1 at depth is still refused until M5.5b wires the allocator.
  { name: 'body3', steps: 200,
    url: 'scenario=sphere&n=8&re=20&u0=0.05&q=19&bounceback=1&live=0&levels=3&rb=4&refine=body&interface=explode' },
  // The same depth-3 tree with the MANAGER rebuilding it, on a PINNED body.
  // `drift3` below is what proves the allocator moves tiles; this one is its
  // control -- the criterion agrees with the initial set, so the manager
  // must decide "no change" at every level and the run must stay identical
  // to `body3`. A manager that is subtly wrong at depth tends to fail here
  // first, because "no change" is the one answer that can be checked against
  // a static run.
  { name: 'body3-dynamic', steps: 200,
    url: 'scenario=sphere&n=8&re=20&u0=0.05&q=19&bounceback=1&live=0&levels=3&rb=4&refine=body&interface=explode&dynamic=1&manageEvery=4' },
  { name: 'bar3', steps: 200,
    url: 'scenario=beltrami&n=16&tau=0.8&u0=0.04&q=19&live=0&levels=3&rb=4&refine=bar&boxfrac=0.5&interface=explode' },
  // THE CONFIG THAT PROVES THE HARD FAILURE FIRES. It asks for far more
  // tiles than ?slotHeadroom= allows, so the manager must be refused -- and
  // M5.4a turned that from a silent degradation into a latched stop. It used
  // to assert the opposite (that the pool stayed consistent while quietly
  // under-refining); the pool consistency is still asserted, but the
  // refusal is now the point.
  { name: 'body-refine', expectInUse: 'increase', expectExhausted: true, steps: 8,
    url: 'scenario=sphere&n=16&re=20&u0=0.05&q=19&bounceback=1&live=0&levels=2&rb=4&refine=body&interface=explode&dynamic=1&manageEvery=1&manageMargin=6' },
];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--checkEvery=')) o.checkEvery = parseInt(a.slice(13));
    else if (a.startsWith('--configs=')) o.configs = a.slice(10).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);

async function runConfig(Runtime, o, c, log) {
  const G = 'window.__D3';
  const p = await evalOrThrow(Runtime, `${G}.getParams()`, 20000, 'getParams');
  log(`N=${p.N} Q${p.Q} levels=${p.levels} RB=${p.rb} ${p.activeSlots}/${p.blocks} tiles`);

  const res = { bal: [], ring: [], cov: [], pool: [], finite: true, vacuous: null, covSkipped: null, blewUp: false, fineState: null };
  // A config may cap its own step count: the allocator cases only need a few
  // steps to change hands, and running them long is spending minutes on a
  // field that is deliberately wrong.
  const steps = Math.min(o.steps, c.steps || o.steps);
  let done = 0, firstInUse = null, firstBbox = null;
  while (done <= steps) {
    const bal = await evalOrThrow(Runtime, `${G}.debugCheck21Balance()`, 120000, 'debugCheck21Balance');
    const ring = await evalOrThrow(Runtime, `${G}.debugCheckRingParents()`, 120000, 'debugCheckRingParents');
    const cov = await evalOrThrow(Runtime, `${G}.debugCheckGeometryCoverage()`, 300000, 'debugCheckGeometryCoverage');
    const ps = await evalOrThrow(Runtime, `${G}.debugPoolState()`, 300000, 'debugPoolState');
    const st = await evalOrThrow(Runtime, `${G}.readStats()`, 300000, 'readStats');
    // EVERY POOL LEVEL'S OWN FIELD, not just L0's (M5.6). readStats reads the
    // dense macroscopic array, which a fine level only reaches through
    // coalesce and only at the cells it covers -- so "L0 is finite" is a
    // weaker statement than it looks, and at depth it is two transfers
    // removed from where a newly-filled or newly-drained tile actually
    // lives. This reads each level's own mac pool, over the SLOT BUDGET and
    // honouring slotToBlock, so a tile the manager has just handed out is in
    // the sample the moment it exists.
    const ps2 = await evalOrThrow(Runtime, `${G}.readPoolStats()`, 300000, 'readPoolStats');
    if (ps.ok === false) res.pool.push({ at: done, n: ps.problems.length, first: ps.problems[0] });
    // M5.4a. Running out of slots is a HARD failure: refinement is
    // geometry-forced, so a refused tile means a coarse/fine seam through
    // the body. One config exists to prove the failure FIRES; everywhere
    // else it is a failure.
    if (ps.slotsExhausted) {
      res.exhausted = ps.slotsExhausted;
      if (!c.expectExhausted) {
        res.pool.push({ at: done, n: 1, first: { kind: 'poolExhausted', refusals: ps.slotsExhausted } });
      }
    }
    if (ps.inUse != null) {
      res.poolState = ps;
      // THE FINEST LEVEL'S box, not level 1's. The criterion is evaluated
      // there and every coarser level is derived from it, so the finest
      // level is where "the shell followed the body" is a statement about
      // the manager rather than about the closure. At ?levels=2 it IS level
      // 1 and nothing changes. byLevel is 0-based from level 1.
      const fine = ps.byLevel ? ps.byLevel[ps.byLevel.length - 1] : ps;
      res.fineState = fine;
      if (firstInUse === null) { firstInUse = fine.inUse; firstBbox = fine.bbox; }
    }
    if (bal.skipped) throw new Error(`2:1 balance unavailable: ${bal.skipped}`);
    res.vacuous = bal.vacuous;
    if (cov.skipped) res.covSkipped = cov.skipped;
    if (c.skipCoverage) res.covSkipped = 'not gated (the manager margin is deliberately wrong here)';
    if (!bal.ok) res.bal.push({ at: done, n: bal.nViolations, first: bal.violations[0] });
    res.ringVacuous = ring.vacuous;
    if (ring.ok === false) res.ring.push({ at: done, n: ring.nViolations, first: ring.violations[0] });
    if (cov.ok === false && !c.skipCoverage) res.cov.push({ at: done, n: cov.nViolations, first: cov.violations[0] });
    if (cov.required != null) res.required = cov.required;
    if (ps2 && ps2.byLevel) {
      res.poolLevels = ps2.byLevel;
      const bad = ps2.byLevel.filter(l => !l.finite);
      if (bad.length && !c.fieldWrongByDesign) {
        res.pool.push({ at: done, n: bad.length,
          first: { kind: 'poolLevelNotFinite', level: bad[0].level, cells: bad[0].cells } });
      }
      // A level whose interiors are all-zero after the run has started is
      // not advancing -- the shape an unseeded pool has, and what
      // M5.2b-ii's bug looked like before the blowup reached L0.
      //
      // FROM STEP 0, and that is a claim about reset() rather than a
      // loosened check. A pool's `mac` is DERIVED -- written by the step and
      // coalesce kernels -- and reset() used to seed `f` alone, so every
      // level read rms 0 at step 0 on cases whose L0 energy is plainly
      // nonzero. M6.0 made reset() run the moments kernel over each seeded
      // level, so a zero here now means what it says.
      //
      // Configs that genuinely start from rest say so, because the
      // alternative is a check that cannot fail on the cases it matters
      // most for. A count of zero LIVE CELLS is a different thing and
      // belongs to the pool check, which owns inUse.
      if (!c.startsAtRest || done > 0) {
        const still = ps2.byLevel.filter(l => l.cells > 0 && l.rms === 0).map(l => l.level);
        if (still.length) {
          res.pool.push({ at: done, n: still.length,
            first: { kind: 'poolLevelNotAdvancing', levels: still } });
        }
      }
    }
    if (st.finite === false || !Number.isFinite(st.ke)) {
      res.blewUp = true;
      res.finite = !!c.fieldWrongByDesign;   // expected here, a failure anywhere else
      log(`step ${done}: field blew up`
        + (c.fieldWrongByDesign ? ' -- EXPECTED: this config allocates uninitialized tiles (M4.2b-ii)' : ' -- stopping'));
      break;
    }
    log(`step ${done}: 2:1 ${bal.ok ? 'ok' : `${bal.nViolations} VIOLATIONS`}`
      + `  ring ${ring.ok ? (ring.vacuous ? 'vacuous' : `ok (${ring.required})`) : `${ring.nViolations} VIOLATIONS`}`
      + `  coverage ${cov.skipped ? 'skipped' : (cov.ok ? `ok (${cov.required} cells required)` : `${cov.nViolations} VIOLATIONS`)}`
      + `  pool ${ps.ok ? `ok (${ps.inUse} in use, ${ps.free} free${ps.dynamic ? ', dynamic' : ''})` : `${ps.problems.length} PROBLEMS`}`
      + `  ke=${Number(st.ke).toExponential(3)}`
      + (ps2 && ps2.byLevel
        ? '  rms ' + ps2.byLevel.map(l => `L${l.level} ${Number(l.rms).toExponential(2)}${l.finite ? '' : ' NaN'}`).join(' ')
        : ''));
    if (done === steps) break;
    const k = Math.min(o.checkEvery, steps - done);
    await evalOrThrow(Runtime, `${G}.debugStepSync(${k})`, (o.timeout + 30) * 1000, 'debugStepSync');
    done += k;
  }
  if (c.expectBboxMove && res.fineState && res.fineState.bbox && firstBbox) {
    const moved = Math.abs(res.fineState.bbox.lo[0] - firstBbox.lo[0]);
    const ok = moved >= c.expectBboxMove;
    res.expectation = { want: `finest-level bbox moves >= ${c.expectBboxMove} blocks`, moved, ok };
    if (!ok) {
      res.pool.push({ at: 'end', n: 1,
        first: { kind: 'shellDidNotFollow', level: res.fineState.level, movedBlocks: moved,
                 required: c.expectBboxMove, from: firstBbox.lo, to: res.fineState.bbox.lo } });
    }
    log(`refined shell (L${res.fineState.level}): lo.x ${firstBbox.lo[0]} -> ${res.fineState.bbox.lo[0]}`
      + ` (${moved} blocks) ${ok ? 'ok' : 'DID NOT FOLLOW'}`);
  }
  // M6. THE TREE SAMPLER, scored against d3-amr.mjs's finestLevelAt on real
  // GPU data. Once per config, at the END: the sampler is a function of
  // blockSlot, so on a dynamic config the interesting hierarchy is the one
  // the manager has been rebuilding rather than the one uploaded at boot.
  const ts = await evalOrThrow(Runtime, `${G}.debugCheckTreeSample()`, 300000, 'debugCheckTreeSample');
  if (ts.skipped) res.sampleSkipped = ts.skipped;
  else {
    res.sample = ts;
    if (!ts.ok) {
      res.pool.push({ at: 'end', n: ts.differ + ts.nonFinite,
        first: { kind: 'treeSampleDiffersFromHost', ...ts.first } });
    }
    // A POOL WITH NO REFINED HITS IS A FAILURE, and it is a different one
    // from a mismatch: a sampler that fell back to level 0 everywhere agrees
    // with a host that was handed the same empty level sets, so the counts
    // have to be looked at directly. This is the "?refine=all cannot see an
    // interface bug" lesson in another costume.
    if (ts.sampleLevels > 0 && ts.refinedHits === 0) {
      res.pool.push({ at: 'end', n: 1,
        first: { kind: 'treeSampleNeverLeftL0', points: ts.points, sampleLevels: ts.sampleLevels } });
    }
    log(`tree sample: ${ts.points} points, by level ${ts.byLevel.join('/')}`
      + `  ${ts.ok && (ts.sampleLevels === 0 || ts.refinedHits > 0) ? 'match' : 'DIFFER'}`);
  }

  // Did the manager actually do the thing the config exists to observe?
  // M5.5a. The manager's criterion-and-closure chain, scored against
  // d3-amr.mjs's refineHierarchy -- two independent statements of one rule,
  // run against each other on real GPU data. Once per config rather than per
  // checkpoint: the host side is an SDF sweep over every block at every
  // level, and one run at the END is the interesting one anyway, since a
  // moving body has moved by then.
  const bal2 = await evalOrThrow(Runtime, `${G}.debugRunBalance()`, 300000, 'debugRunBalance');
  if (bal2.skipped) res.wantSkipped = bal2.skipped;
  else {
    res.want = bal2;
    if (!bal2.ok) {
      res.pool.push({ at: 'end', n: 1,
        first: { kind: 'wantSetDiffersFromHost', levels: bal2.levels.filter(d => d.onlyGpu || d.onlyHost) } });
    }
    log('want vs host: ' + bal2.levels.map(d => `L${d.level} ${d.gpu}/${d.host}`).join(' ')
      + (bal2.ok ? '  match' : '  DIFFER'));
  }
  if (c.expectExhausted && !res.exhausted) {
    res.pool.push({ at: 'end', n: 1, first: { kind: 'expectedExhaustionDidNotFire' } });
    log('expected the pool to be exhausted and it was not -- the hard failure did not fire');
  }
  // THE FINEST LEVEL, for the same reason expectBboxMove reads it: that is
  // where the criterion is evaluated, so it is where "the manager acted" is
  // a statement about the criterion firing rather than about the closure
  // propagating something that happened below. Identical at ?levels=2.
  if (c.expectInUse && res.fineState) {
    const now = res.fineState.inUse;
    const moved = now - firstInUse;
    const want = c.expectInUse === 'increase' ? moved > 0 : moved < 0;
    res.expectation = { want: c.expectInUse, from: firstInUse, to: now, ok: want };
    if (!want) {
      res.pool.push({ at: 'end', n: 1,
        first: { kind: 'managerDidNotAct', level: res.fineState.level, expected: c.expectInUse,
                 inUse: `${firstInUse} -> ${now}` } });
    }
    log(`manager ${c.expectInUse} (L${res.fineState.level}): inUse ${firstInUse} -> ${now} ${want ? 'ok' : 'DID NOT ACT'}`);
  }
  return res;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = CONFIGS.filter(c => !o.configs || o.configs.includes(c.name));
  if (!configs.length) { console.error('no configs selected'); process.exit(2); }

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = await attachPageWatch(client, {
    onError: (e) => console.error(`    !! [${e.kind}] ${e.text.split('\n')[0]}`),
  });

  const report = [];
  try {
    for (const c of configs) {
      const url = `${o.baseUrl}/index-3d.html?${c.url}${o.extra ? `&${o.extra}` : ''}`;
      console.log(`\n=== ${c.name} (${url})`);
      try {
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, 'window.__D3', 60000);
        await assertPageHealthy(Runtime, watch, c.name);
        const res = await runConfig(Runtime, o, c, s => console.log('    ' + s));
        // A config that EXPECTS the pool to run out will have put `error:
        // out of pool slots` in #status -- that IS its result, so the health
        // check has to know the difference between the guard firing and the
        // page breaking.
        await assertPageHealthy(Runtime, watch, c.name,
          c.expectExhausted ? /out of pool slots/ : null);
        report.push({ name: c.name, res });
      } catch (err) {
        console.error(`    FAILED: ${err.message}`);
        report.push({ name: c.name, error: err.message });
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(144));
  console.log(`SUMMARY  ${o.steps} steps, checked every ${o.checkEvery}`);
  console.log('='.repeat(144));
  console.log(pad('config', 14) + pad('2:1 balance', 21) + pad('ring parents', 18)
    + pad('geometry coverage', 30) + pad('want vs host', 18) + pad('pool', 24)
    + pad('tree sample', 16) + padL('verdict', 9));
  console.log('-'.repeat(144));
  let exitCode = 0;
  for (const r of report) {
    if (r.error) { console.log(pad(r.name, 14) + pad('-', 21) + pad('-', 18) + pad('-', 30) + pad('-', 18) + pad('-', 24) + pad('-', 16) + padL('ERROR', 9)); exitCode = 1; continue; }
    const x = r.res;
    const balTxt = x.bal.length ? `${x.bal.length} checkpoints BAD` : (x.vacuous ? 'ok (VACUOUS N=2)' : 'ok');
    const covTxt = x.cov.length ? `${x.cov.length} checkpoints BAD`
      : (x.covSkipped ? 'skipped (not geometry-forced)' : `ok (${x.required} cells required)`);
    // Every level, not just level 1: at depth the interesting number is how
    // the tree is distributed, and a single count hides it entirely.
    const perLevel = x.poolState && x.poolState.byLevel
      ? x.poolState.byLevel.map(l => `${l.inUse}+${l.free}`).join(' ')
      : (x.poolState ? `${x.poolState.inUse}+${x.poolState.free}` : null);
    const poolTxt = x.pool.length ? `${x.pool.length} BAD`
      : (perLevel ? `ok ${perLevel}${x.exhausted ? ' EXHAUSTED(ok)' : ''}${x.blewUp ? '*' : ''}` : 'skipped');
    const ringTxt = x.ring.length ? `${x.ring.length} checkpoints BAD` : (x.ringVacuous ? 'ok (VACUOUS N=2)' : 'ok');
    const wantTxt = x.wantSkipped ? 'skipped'
      : (x.want ? (x.want.ok ? `ok ${x.want.levels.map(d => d.gpu).join('/')}` : 'DIFFERS') : '-');
    // M6. In the table rather than only in the per-config log, because this
    // file's own maxim is that "the checker ran" and "the invariant holds"
    // are different claims -- and a summary that cannot say which is which
    // has already given up on the distinction. The by-level histogram is
    // what shows the sampler reached the pool at all.
    const sampTxt = x.sampleSkipped ? 'skipped'
      : (x.sample ? (x.sample.ok ? `ok ${x.sample.byLevel.join('/')}` : 'DIFFERS') : '-');
    const ok = !x.bal.length && !x.ring.length && !x.cov.length && !x.pool.length && x.finite;
    if (!ok) exitCode = 1;
    console.log(pad(r.name, 14) + pad(balTxt, 21) + pad(ringTxt, 18) + pad(covTxt, 30)
      + pad(wantTxt, 18) + pad(poolTxt, 24) + pad(sampTxt, 16) + padL(ok ? 'PASS' : 'FAIL', 9));
  }
  if (report.some(r => r.res && r.res.blewUp && r.res.finite)) {
    console.log('\n* the field blew up, EXPECTEDLY: that config allocates tiles nothing initializes');
    console.log('  (M4.2b-ii). Its structural claims are still gated; only the field is excused.');
  }
  const anyVacuous = report.some(r => r.res && r.res.vacuous);
  if (anyVacuous) {
    console.log('\nVACUOUS marks the ?levels=2 rows, where both checks CANNOT fail: a level-1');
    console.log('leaf\'s neighbour is level 1 or level 0 and both are legal, and level 1\'s parent');
    console.log('is the DENSE L0 grid, so no parent tile can be missing. Dynamic refinement does');
    console.log('not change either -- it moves tiles WITHIN one level. Do not read those rows as');
    console.log('evidence about the invariants, only that the machinery runs.');
    console.log('The ?levels=3 rows are where both are REAL: box3/bar3/body3 on a tree the HOST');
    console.log('built (M5.3), and body3-dynamic/drift3 on one the MANAGER rebuilds every few');
    console.log('steps (M5.5b). drift3 is the only row where the manager must actually ACT.');
    console.log('');
    console.log('RING PARENTS is vacuous for a DIFFERENT reason worth keeping straight: level 1\'s');
    console.log('parent is the DENSE L0 grid, which exists everywhere, so no parent tile can be');
    console.log('missing. At levels>=3 it is a real gate and it is NOT implied by 2:1 balance --');
    console.log('see d3-amr.mjs\'s checkRingParentCoverage, whose test asserts the two disagree.');
  }
  const failed = report.filter(r => r.error || (r.res && (r.res.bal.length || r.res.ring.length || r.res.cov.length || r.res.pool.length || !r.res.finite)));
  if (failed.length) {
    console.log('\nDetails:');
    for (const r of failed) {
      if (r.error) { console.log(`  [${r.name}] ${r.error}`); continue; }
      for (const b of r.res.bal) console.log(`  [${r.name}] 2:1 at step ${b.at}: ${b.n} violations, e.g. ${JSON.stringify(b.first)}`);
      for (const b of r.res.ring) console.log(`  [${r.name}] ring parents at step ${b.at}: ${b.n} violations, e.g. ${JSON.stringify(b.first)}`);
      for (const b of r.res.cov) console.log(`  [${r.name}] coverage at step ${b.at}: ${b.n} violations, e.g. ${JSON.stringify(b.first)}`);
      for (const b of r.res.pool) console.log(`  [${r.name}] pool at step ${b.at}: ${b.n} problems, e.g. ${JSON.stringify(b.first)}`);
      if (!r.res.finite) console.log(`  [${r.name}] field blew up`);
    }
  } else {
    console.log('\nAll configs PASS.');
  }
  process.exit(exitCode);
}

main();
