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
//                 CANNOT fail today. It says so rather than presenting a
//                 green tick, because "the checker ran" and "the invariant
//                 holds" are different claims.
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
  { name: 'body-refine', expectInUse: 'increase', steps: 8,
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

  const res = { bal: [], cov: [], pool: [], finite: true, vacuous: null, covSkipped: null, blewUp: false };
  // A config may cap its own step count: the allocator cases only need a few
  // steps to change hands, and running them long is spending minutes on a
  // field that is deliberately wrong.
  const steps = Math.min(o.steps, c.steps || o.steps);
  let done = 0, firstInUse = null;
  while (done <= steps) {
    const bal = await evalOrThrow(Runtime, `${G}.debugCheck21Balance()`, 120000, 'debugCheck21Balance');
    const cov = await evalOrThrow(Runtime, `${G}.debugCheckGeometryCoverage()`, 300000, 'debugCheckGeometryCoverage');
    const ps = await evalOrThrow(Runtime, `${G}.debugPoolState()`, 300000, 'debugPoolState');
    const st = await evalOrThrow(Runtime, `${G}.readStats()`, 300000, 'readStats');
    if (ps.ok === false) res.pool.push({ at: done, n: ps.problems.length, first: ps.problems[0] });
    if (ps.inUse != null) { res.poolState = ps; if (firstInUse === null) firstInUse = ps.inUse; }
    if (bal.skipped) throw new Error(`2:1 balance unavailable: ${bal.skipped}`);
    res.vacuous = bal.vacuous;
    if (cov.skipped) res.covSkipped = cov.skipped;
    if (c.skipCoverage) res.covSkipped = 'not gated (the manager margin is deliberately wrong here)';
    if (!bal.ok) res.bal.push({ at: done, n: bal.nViolations, first: bal.violations[0] });
    if (cov.ok === false && !c.skipCoverage) res.cov.push({ at: done, n: cov.nViolations, first: cov.violations[0] });
    if (cov.required != null) res.required = cov.required;
    if (st.finite === false || !Number.isFinite(st.ke)) {
      res.blewUp = true;
      res.finite = !!c.fieldWrongByDesign;   // expected here, a failure anywhere else
      log(`step ${done}: field blew up`
        + (c.fieldWrongByDesign ? ' -- EXPECTED: this config allocates uninitialized tiles (M4.2b-ii)' : ' -- stopping'));
      break;
    }
    log(`step ${done}: 2:1 ${bal.ok ? 'ok' : `${bal.nViolations} VIOLATIONS`}`
      + `  coverage ${cov.skipped ? 'skipped' : (cov.ok ? `ok (${cov.required} cells required)` : `${cov.nViolations} VIOLATIONS`)}`
      + `  pool ${ps.ok ? `ok (${ps.inUse} in use, ${ps.free} free${ps.dynamic ? ', dynamic' : ''})` : `${ps.problems.length} PROBLEMS`}`
      + `  ke=${Number(st.ke).toExponential(3)}`);
    if (done === steps) break;
    const k = Math.min(o.checkEvery, steps - done);
    await evalOrThrow(Runtime, `${G}.debugStepSync(${k})`, (o.timeout + 30) * 1000, 'debugStepSync');
    done += k;
  }
  // Did the manager actually do the thing the config exists to observe?
  if (c.expectInUse && res.poolState) {
    const moved = res.poolState.inUse - firstInUse;
    const want = c.expectInUse === 'increase' ? moved > 0 : moved < 0;
    res.expectation = { want: c.expectInUse, from: firstInUse, to: res.poolState.inUse, ok: want };
    if (!want) {
      res.pool.push({ at: 'end', n: 1,
        first: { kind: 'managerDidNotAct', expected: c.expectInUse, inUse: `${firstInUse} -> ${res.poolState.inUse}` } });
    }
    log(`manager ${c.expectInUse}: inUse ${firstInUse} -> ${res.poolState.inUse} ${want ? 'ok' : 'DID NOT ACT'}`);
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
        await assertPageHealthy(Runtime, watch, c.name);
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

  console.log('\n' + '='.repeat(88));
  console.log(`SUMMARY  ${o.steps} steps, checked every ${o.checkEvery}`);
  console.log('='.repeat(88));
  console.log(pad('config', 14) + pad('2:1 balance', 21) + pad('geometry coverage', 30) + pad('pool', 14) + padL('verdict', 9));
  console.log('-'.repeat(88));
  let exitCode = 0;
  for (const r of report) {
    if (r.error) { console.log(pad(r.name, 14) + pad('-', 21) + pad('-', 30) + pad('-', 14) + padL('ERROR', 9)); exitCode = 1; continue; }
    const x = r.res;
    const balTxt = x.bal.length ? `${x.bal.length} checkpoints BAD` : (x.vacuous ? 'ok (VACUOUS N=2)' : 'ok');
    const covTxt = x.cov.length ? `${x.cov.length} checkpoints BAD`
      : (x.covSkipped ? 'skipped (not geometry-forced)' : `ok (${x.required} cells required)`);
    const poolTxt = x.pool.length ? `${x.pool.length} BAD`
      : (x.poolState ? `ok ${x.poolState.inUse}+${x.poolState.free}${x.blewUp ? '*' : ''}` : 'skipped');
    const ok = !x.bal.length && !x.cov.length && !x.pool.length && x.finite;
    if (!ok) exitCode = 1;
    console.log(pad(r.name, 14) + pad(balTxt, 21) + pad(covTxt, 30) + pad(poolTxt, 14) + padL(ok ? 'PASS' : 'FAIL', 9));
  }
  if (report.some(r => r.res && r.res.blewUp && r.res.finite)) {
    console.log('\n* the field blew up, EXPECTEDLY: that config allocates tiles nothing initializes');
    console.log('  (M4.2b-ii). Its structural claims are still gated; only the field is excused.');
  }
  const anyVacuous = report.some(r => r.res && r.res.vacuous);
  if (anyVacuous) {
    console.log('\nVACUOUS means the 2:1 check ran and could not have failed: at ?levels=2 a leaf\'s');
    console.log('neighbour is level 1 or level 0 and both are legal. It becomes a real gate when');
    console.log('M4.2 makes refinement dynamic or M5 adds a level. Do not read it as evidence yet.');
  }
  const failed = report.filter(r => r.error || (r.res && (r.res.bal.length || r.res.cov.length || r.res.pool.length || !r.res.finite)));
  if (failed.length) {
    console.log('\nDetails:');
    for (const r of failed) {
      if (r.error) { console.log(`  [${r.name}] ${r.error}`); continue; }
      for (const b of r.res.bal) console.log(`  [${r.name}] 2:1 at step ${b.at}: ${b.n} violations, e.g. ${JSON.stringify(b.first)}`);
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
