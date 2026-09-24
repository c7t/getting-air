#!/usr/bin/env node
// Top-level "check everything" harness: runs the physics regression
// (tools/lib/cylinder-metrics.js, Cd/St vs. benchmarks/cylinder.json) and
// the structural-invariant regression (tools/lib/amr-invariants.js, 2:1
// balance + geometry-forced-refinement coverage + field-finite sanity)
// across a fixed list of scenarios: the dense reference, and the AMR
// cylinder harness at each of the levels/coupling combinations this
// project's own commit history and plans/AMR-multilevel.md milestones
// track as meaningfully different states (N=2 diffuse, N=2 bounce-back,
// N=3 diffuse, N=3 bounce-back). Also runs a cheap boot smoke check
// (checkBoots -- see defaultConfigs' own comment) against index.html and
// index-amr.html, the two dev pages this suite otherwise never visits
// (they have no window.__CYL, so checkPhysics/checkInvariants don't
// apply) -- added after a shared-shader/JS-bind-group mismatch broke
// index-amr.html in production without failing anything in this suite,
// since nothing here had ever loaded that page. One command, one report,
// instead of hand-launching Chrome and running tools/validate-cylinder.js
// and tools/validate-amr-invariants.js separately per config.
//
// Unlike the two leaf tools (which assume a Chrome + page are already up),
// this one owns the whole lifecycle: starts the HTTPS dev server and a
// dedicated debug-port Chrome if neither is already running, opens one tab
// per scenario, runs whichever checks that scenario calls for, and tears
// down anything it started when done (leaves alone anything that was
// already running, e.g. a Chrome you launched by hand for manual poking).
//
// Usage:
//   node tools/validate-all.js
//   node tools/validate-all.js --configs=amr-N2-bounceback,amr-N3-diffuse
//   node tools/validate-all.js --re=20,40,100,200 --steps=20000
//   node tools/validate-all.js --port=9333 --baseUrl=https://localhost:4444

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { evalExpr: evalExprCyl, runCase } = require('./lib/cylinder-metrics');
const { evalExpr: evalExprChan, runCase: runChanCase } = require('./lib/channel-metrics');
const { evalExpr: evalExprTgv, runCase: runTgvCase } = require('./lib/tgv-metrics');
const { runInvariantSweep } = require('./lib/amr-invariants');
const { runRenderLevels } = require('./lib/render-levels');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

function parseArgs(argv) {
  const opts = {
    baseUrl: 'https://localhost:4444',
    port: 9333,
    re: [100],
    physicsTimeout: 300,
    invariantSteps: 8192,
    invariantCheckEvery: 1024,
    configs: null,
    keepOpen: false,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--re=')) opts.re = a.slice(5).split(',').map(Number);
    else if (a.startsWith('--timeout=')) opts.physicsTimeout = parseInt(a.slice(10));
    else if (a.startsWith('--steps=')) opts.invariantSteps = parseInt(a.slice(8));
    else if (a.startsWith('--checkEvery=')) opts.invariantCheckEvery = parseInt(a.slice(13));
    else if (a.startsWith('--configs=')) opts.configs = a.slice(10).split(',');
    else if (a.startsWith('--extra=')) opts.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a === '--keepOpen') opts.keepOpen = true;
  }
  return opts;
}

// Extra query params from --extra=, as a suffix for a URL that already has a
// '?'. The channel and TGV harnesses build their own URLs per case rather
// than using config.url, so they call this directly; everything else gets it
// appended to config.url in main().
function extraParam(opts) { return opts.extra ? `&${opts.extra}` : ''; }

function defaultConfigs(baseUrl) {
  return [
    // Boot smoke checks: index.html/index-amr.html have no window.__CYL (no
    // Cd/St, no AMR-invariant surface -- those are the cylinder harness's
    // own addition), so checkPhysics/checkInvariants don't apply to them.
    // They're the ONLY consumers of shaders/lbm_*.wgsl (index.html) and one
    // of two consumers of shaders/amr_*.wgsl (index-amr.html, alongside
    // index-cylinder-amr.html) -- but before checkBoots existed, neither
    // page was ever loaded by this harness at all, only the cylinder pages
    // were. That gap is exactly what let a shared-shader/JS-bind-group
    // mismatch break index-amr.html in production undetected (see git log
    // for "Fix index-amr.html: force1PoolBGL was missing the new
    // debugSlotForce binding") -- a WGSL binding count change was made to
    // shaders/amr_force1_pool.wgsl and mirrored into main-cylinder-amr.js's
    // own bind group, but main-amr.js's SEPARATE copy of that same bind
    // group (index-amr.html's only path to that shader) was missed, and
    // nothing in this suite ever visited index-amr.html to notice.
    // checkBoots is the cheap fix: load the page, confirm the compute loop
    // is actually advancing.
    {
      name: 'index-boot',
      url: `${baseUrl}/index.html`,
      checkBoots: true,
    },
    {
      name: 'amr-dev-boot',
      url: `${baseUrl}/index-amr.html`,
      checkBoots: true,
    },
    // The reentry pages are the last consumers of shaders/lbm_*.wgsl and
    // shaders/amr_*.wgsl with no other coverage here -- they have no
    // analytic check of their own (prescribed kinematics, not a validated
    // benchmark), but they DO have their own bind groups over the shared
    // shaders, which is the thing that has actually broken before. A boot
    // smoke is the whole of what's checkable and exactly the gap that let
    // 238e48c ship.
    {
      name: 'reentry-boot',
      url: `${baseUrl}/index-reentry.html`,
      checkBoots: true,
    },
    {
      name: 'reentry-amr-boot',
      url: `${baseUrl}/index-reentry-amr.html`,
      checkBoots: true,
    },
    // THE OTHER THREE AMR PAGES, AT ?levels=3 -- and the level is the point.
    //
    // Four of the five AMR pages default to levels=2, and at levels=2
    // updateLevelParams's `for (c = 2; c < N_LEVELS; c++)` loop runs ZERO
    // times. A whole init path therefore exists that nothing in this suite
    // reached cheaply: the cylinder page's levels=3 configs did reach it, but
    // only via the most expensive runs here, and the two bodyless AMR pages
    // had no boot coverage at all.
    //
    // That is not hypothetical. B3a-1 extracted `tauAtLevel` into
    // card-params.mjs, landed the CALL in all five pages and the IMPORT in
    // one, and the other four threw `ReferenceError: tauAtLevelOf is not
    // defined` at init -- invisible at every page's own default, fatal at
    // levels=3, and `make check` green throughout (it is a PARSE check; the
    // name is perfectly good syntax). Exactly the shape CLAUDE.md records
    // checkBoots being added for after 238e48c, one octave deeper.
    {
      name: 'cylinder-amr-boot-N3',
      url: `${baseUrl}/index-cylinder-amr.html?levels=3`,
      checkBoots: true,
    },
    {
      name: 'tgv-amr-boot-N3',
      url: `${baseUrl}/index-tgv-amr.html?levels=3`,
      checkBoots: true,
    },
    {
      name: 'channel-amr-boot-N3',
      url: `${baseUrl}/index-channel-amr.html?levels=3`,
      checkBoots: true,
    },
    // REFUSALS UNDER TEST (plans/2D-backport.md B4-4). Each deliberately
    // misconfigures a page and requires the refusal to ARRIVE -- see
    // runBootSmoke's expectError note on why a guard that never fires and a
    // guard that cannot fire are indistinguishable from a green suite.
    {
      // A module-scope guard. It was always there and always correct, and
      // until B4-4 it threw where init().catch(handleErr) could not see it,
      // so the page sat at "initializing..." with the reason only in the
      // console (plans/2D-backport.md B0b recorded the symptom).
      name: 'refuse-levels-1',
      url: `${baseUrl}/index-tgv-amr.html?levels=1`,
      checkBoots: true,
      expectError: /levels=1 invalid/i,
    },
    {
      // The refine-ahead requirement: a lookahead below the decision interval
      // leaves part of every refinement round unprotected by construction.
      name: 'refuse-short-lookahead',
      url: `${baseUrl}/index-cylinder-amr.html?levels=2&refineEvery=16&forceRefineLookahead=1`,
      checkBoots: true,
      expectError: /forceRefineLookahead/i,
    },
    {
      // The RUNTIME latch: an under-provisioned pool means geometry-forced
      // refinement is being refused, which since B4-3 means the body's force
      // is simply missing rather than crude. Starved hard enough that the
      // trip-wire and the coverage check must both agree.
      name: 'refuse-pool-exhausted',
      url: `${baseUrl}/index-cylinder-amr.html?levels=3&maxFineBlocks=8`,
      checkBoots: true,
      expectError: /geometry-forced refinement REFUSED/i,
    },
    {
      // tau = 1 makes the post-collision coarse<->fine transfer 0/0
      // (plans/2D-backport.md B1). ?tau=0.75 puts LEVEL 1 exactly on it while
      // L0 itself is perfectly ordinary -- which is the case worth gating,
      // because the value the user typed is not the value that is singular,
      // and a guard that only looked at L0 would pass this.
      name: 'refuse-tau-unity',
      url: `${baseUrl}/index-tgv-amr.html?levels=2&tau=0.75`,
      checkBoots: true,
      expectError: /at level 1 is within .* of 1/i,
    },
    {
      // ...and the same config with the legacy factor selected must BOOT.
      // The singularity belongs to the post-collision form alone, so refusing
      // it under ?dcpre=1 would be refusing a configuration that works. This
      // is the negative half of the gate above: without it, a guard that
      // simply banned tau near 1 outright would look identical.
      name: 'tau-unity-ok-under-dcpre',
      url: `${baseUrl}/index-tgv-amr.html?levels=2&tau=0.75&dcpre=1`,
      checkBoots: true,
    },
    // index-amr.html under the structural-invariant sweep. This is the page
    // the project SHIPS, it defaults to levels=3, and until now the sweep
    // only ever drove window.__CYL -- so the falling-card page's own 2:1
    // balance was never under test on any config, only boot-smoked above.
    // It exposed no body-geometry coverage scan and no card-state readback
    // until plans/2D-backport.md B4 -- so the SHIPPED page, whose body
    // MOVES, was the one running geometry-forced refinement with nothing
    // checking it, while the only implementation in the project sat on the
    // pinned-cylinder page. Both now report OK rather than n/a.
    {
      name: 'amr-dev-invariants',
      url: `${baseUrl}/index-amr.html`,
      global: 'window.__AMR',
      checkInvariants: true,
    },
    // RENDER REACHABILITY -- in the default sweep since plans/uniform-levels.md
    // U6 closed it, and deliberately not before: a known-red gate does not get
    // to turn the whole sweep red. It perturbs one level's velocity pool with a
    // value no flow produces, redraws, and asks whether the picture moved.
    //
    // BOTH PAGES, because the defect it found was not the one it was written
    // for. index-amr.html was missing level 3; index-cylinder-amr.html -- and
    // the TGV and channel pages with it -- never passed a level override to the
    // render fragment at all, so level 2 was solved and never drawn on the page
    // the Cd/St numbers come from.
    {
      name: 'render-levels-card',
      url: `${baseUrl}/index-amr.html?levels=4`,
      global: 'window.__AMR',
      checkRenderLevels: true,
    },
    {
      name: 'render-levels-cylinder',
      url: `${baseUrl}/index-cylinder-amr.html?levels=3`,
      global: 'window.__CYL',
      checkRenderLevels: true,
    },
    {
      name: 'dense-reference',
      url: `${baseUrl}/index-cylinder.html`,
      checkPhysics: true,
      checkInvariants: false,
    },
    {
      name: 'amr-N2-diffuse',
      url: `${baseUrl}/index-cylinder-amr.html?levels=2`,
      checkPhysics: true,
      checkInvariants: true,
    },
    {
      name: 'amr-N2-bounceback',
      url: `${baseUrl}/index-cylinder-amr.html?levels=2&bounceback`,
      checkPhysics: true,
      checkInvariants: true,
    },
    {
      name: 'amr-N3-diffuse',
      url: `${baseUrl}/index-cylinder-amr.html?levels=3`,
      checkPhysics: true,
      checkInvariants: true,
    },
    {
      // Was known-broken (required ?forceBounceback, checkPhysics:false)
      // until the L2 bounce-back registration + coverage-margin fixes --
      // see main-cylinder-amr.js's own comment above N_LEVELS. No longer
      // needs ?forceBounceback (the guard only blocks levels>3 now), and
      // Cd/St now validate like every other config.
      name: 'amr-N3-bounceback',
      url: `${baseUrl}/index-cylinder-amr.html?levels=3&bounceback`,
      checkPhysics: true,
      checkInvariants: true,
    },
    // Analytical-solution benchmarks (exact closed-form target, not a
    // literature band -- see benchmarks/channel.json and
    // tools/lib/channel-metrics.js). checkChannelPhysics configs sweep
    // BOTH res and re from benchmarks/channel.json's cases, so (unlike
    // every config above) they navigate multiple times each -- see
    // runChannelPhysics. `url` here is a display label only, not
    // navigated to directly. AMR configs default to a single mid-sweep
    // resolution (chanResFilter) -- AMR channel flow's own marginal value
    // is "does the AMR step match lbm_step.wgsl," already the same at
    // every resolution, not a resolution study of its own (see
    // main-channel-amr.js's header on autoRefine defaulting off here).
    {
      name: 'channel-poiseuille-dense',
      url: `${baseUrl}/index-channel.html?mode=poiseuille`,
      checkChannelPhysics: true,
      chanMode: 'poiseuille',
      chanPage: 'index-channel.html',
    },
    {
      name: 'channel-couette-dense',
      url: `${baseUrl}/index-channel.html?mode=couette`,
      checkChannelPhysics: true,
      chanMode: 'couette',
      chanPage: 'index-channel.html',
    },
    {
      name: 'channel-poiseuille-amr-N2',
      url: `${baseUrl}/index-channel-amr.html?mode=poiseuille&levels=2`,
      checkChannelPhysics: true,
      chanMode: 'poiseuille',
      chanPage: 'index-channel-amr.html',
      chanLevels: 2,
      chanResFilter: [32],
    },
    {
      name: 'channel-couette-amr-N2',
      url: `${baseUrl}/index-channel-amr.html?mode=couette&levels=2`,
      checkChannelPhysics: true,
      chanMode: 'couette',
      chanPage: 'index-channel-amr.html',
      chanLevels: 2,
      chanResFilter: [32],
    },
    // Taylor-Green vortex: exact closed-form space-time solution, not just
    // a steady-state target -- see benchmarks/tgv.json and
    // tools/lib/tgv-metrics.js. Every case parameter (N, u0, tau, and for
    // AMR, levels) is page-load-time on both index-tgv.html and
    // index-tgv-amr.html (no live-settable Re here), so runTgvPhysics
    // navigates once per case, filtered from the shared benchmark file by
    // whether `levels` is set and to what -- `url` here is a display label
    // only, not navigated to directly (same convention as the channel
    // configs above).
    {
      name: 'tgv-dense',
      url: `${baseUrl}/index-tgv.html`,
      checkTgvPhysics: true,
      tgvFilter: c => !c.levels,
    },
    {
      name: 'tgv-amr-N2',
      url: `${baseUrl}/index-tgv-amr.html?levels=2`,
      checkTgvPhysics: true,
      tgvFilter: c => c.levels === 2,
    },
    {
      name: 'tgv-amr-N3',
      url: `${baseUrl}/index-tgv-amr.html?levels=3`,
      checkTgvPhysics: true,
      tgvFilter: c => c.levels === 3,
    },
  ];
}

// --- process lifecycle: HTTPS dev server + dedicated debug-port Chrome ----
// (ensureServer/ensureChrome/openTab/firstTab/closeTab/navigateTo/waitFor
// now live in tools/lib/browser-lifecycle.js, shared with
// tools/validate-amr-vs-dense.js -- see that file's own header for the "one
// tab reused across the whole run" invariant this relies on.)

// Every scenario harness exposes window.__CYL; index-amr.html (the falling-
// card page this project ships) exposes window.__AMR instead, so a config
// names its own surface via `global`.
async function waitForCYL(Runtime, timeoutMs, global = 'window.__CYL') {
  return waitForGlobal(Runtime, global, timeoutMs);
}

// --- per-config runners -----------------------------------------------

// Boot smoke check (see defaultConfigs' own header on why this exists): a
// pipeline-creation failure inside init() (e.g. a bind-group-layout
// mismatch against a shared shader's binding count) is caught by that
// page's own `init().catch(handleErr)` -- see main.js/main-amr.js -- which
// writes "error: <message>" into #status and does a console.error, NOT an
// uncaught exception. Runtime.exceptionThrown (this file's existing
// listener, registered for every config) would NOT have caught the bug
// this check was added for -- the only real signal is #status itself
// either starting with "error:" or never advancing past its initial
// "initializing..." text, so that's what this polls for instead of relying
// on the exception listener.
// `expectError` inverts this check: the config PASSES only if #status reaches
// an `error:` matching that pattern, and FAILS if the page boots happily.
//
// WHY AN INVERTED CONFIG IS WORTH HAVING. Every refusal added by
// plans/2D-backport.md B4 is code that runs only when something has gone
// wrong, which is exactly the code most likely to be broken without anyone
// noticing -- a guard that never fires and a guard that cannot fire look
// identical from a green suite. These configs deliberately misconfigure a
// page and require the refusal to arrive, so "refuse rather than degrade" is
// itself under test rather than asserted in a comment.
async function runBootSmoke(Runtime, expectError = null) {
  const readStatus = async () => {
    const r = await evalExprCyl(Runtime, `document.getElementById('status') ? document.getElementById('status').textContent : null`);
    return r.exceptionDetails ? null : r.result.value;
  };
  const matchesExpected = (t) => expectError && /^error:/i.test(t) && expectError.test(t);
  const wrongError = (t) => (expectError
    ? `status shows an error, but not the expected one: "${t}" (wanted ${expectError})`
    : `status shows an error: "${t}"`);
  const first = await readStatus();
  if (first == null) return { ok: false, reason: 'no #status element found' };
  if (matchesExpected(first)) return { ok: true, first, second: first };
  if (/^error:/i.test(first)) return { ok: false, reason: wrongError(first) };
  // POLL until the status advances, rather than sampling once after a fixed
  // sleep. The fixed-4s version this replaces was flaky on a COLD run: a
  // freshly-launched Chrome with an empty profile has no pipeline cache, and
  // index-amr.html creates its pipelines from ~16 WGSL modules before it
  // writes its first real status line, which can exceed 4s on the first load
  // of a session while comfortably fitting in it on every subsequent (warm)
  // load. That produced a "page may be stuck" FAIL for a page that was in
  // fact healthy and several thousand steps in moments later -- precisely
  // the sort of false red that trains people to stop believing the suite.
  //
  // Polling also makes the check STRICTLY stronger, not just slower: an
  // `error:` status is caught the moment it appears (the old version could
  // sleep straight through a transient one), and a genuinely stuck page now
  // costs the full BOOT_SMOKE_TIMEOUT_MS instead of being reported after 4s
  // -- the right trade, since the failure path is the rare one.
  const BOOT_SMOKE_TIMEOUT_MS = 30000;
  const POLL_INTERVAL_MS = 250;
  const deadline = Date.now() + BOOT_SMOKE_TIMEOUT_MS;
  let second = first;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
    second = await readStatus();
    if (second == null) return { ok: false, reason: 'no #status element found (second read)' };
    if (matchesExpected(second)) return { ok: true, first, second };
    if (/^error:/i.test(second)) return { ok: false, reason: wrongError(second) };
    // An expectError config is waiting for the refusal, so a status that
    // merely ADVANCED is not success there -- it is the page running on,
    // which is the failure this config exists to catch.
    if (!expectError && second !== first) return { ok: true, first, second };
  }
  return {
    ok: false,
    reason: expectError
      ? `expected a refusal matching ${expectError} within ${BOOT_SMOKE_TIMEOUT_MS}ms, but the page kept running (status "${second}")`
      : `status never advanced past "${first}" in ${BOOT_SMOKE_TIMEOUT_MS}ms -- page may be stuck`,
  };
}

async function runPhysics(Runtime, opts) {
  const benchPath = path.join(REPO_ROOT, 'benchmarks', 'cylinder.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  const cases = bench.cases.filter(c => opts.re.includes(c.re));
  await evalExprCyl(Runtime, `window.__CYL.setLive(false)`);
  const results = [];
  for (const c of cases) {
    const r = await runCase(Runtime, { timeout: opts.physicsTimeout }, c, s => console.log('    ' + s));
    // THE NUMBERS, PRINTED ON PASS TOO -- the same gap 46157cd closed for the
    // analytic gates, still open here. Cd/St were computed and thrown away
    // unless they FAILED, so "did this change move the cylinder?" could only
    // be answered by a config that happened to be red. That is the wrong way
    // round for the several stages of plans/2D-backport.md that deliberately
    // change the refined region and must REPORT the move rather than absorb
    // it -- and CLAUDE.md's own AMR-Cd reproducibility caveat (~1e-3, the
    // atomicSub free list) is unusable without the digits it is about.
    console.log(`      Cd ${r.cd.measured?.toFixed(3)} / ${r.cd.target}±${r.cd.tol} ${r.cd.pass ? 'ok' : 'FAIL'}`
      + `   St ${r.st.measured?.toFixed(4)} / ${r.st.target}±${r.st.tol} ${r.st.pass ? 'ok' : 'FAIL'}`);
    results.push(r);
  }
  const ok = results.every(r => r.cd.pass && r.st.pass);
  return { ok, results };
}

// Channel-flow configs sweep BOTH res and re, and res is page-load-time
// (not live-settable), so this owns its own per-(mode,res)-group
// navigation -- unlike runPhysics/runInvariants above, which assume the
// main loop already navigated once to config.url. See tools/validate-
// channel.js's identical grouping logic (kept as a separate copy there
// since that tool owns its own Chrome connection/lifecycle, not sharing
// this file's Page/tab).
async function runChannelPhysics(Page, Runtime, opts, config) {
  const benchPath = path.join(REPO_ROOT, 'benchmarks', 'channel.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  let cases = bench.cases.filter(c => c.mode === config.chanMode);
  if (config.chanResFilter) cases = cases.filter(c => config.chanResFilter.includes(c.res));

  const groups = new Map();
  for (const c of cases) {
    if (!groups.has(c.res)) groups.set(c.res, []);
    groups.get(c.res).push(c);
  }

  const results = [];
  for (const [res, groupCases] of groups) {
    const levelsParam = config.chanLevels ? `&levels=${config.chanLevels}` : '';
    // main-channel.js's `?res=` is H directly; main-channel-amr.js's is
    // log2(H), matching every other AMR page's convention (its domain is
    // square and power-of-two by construction, unlike the dense harness).
    // benchmarks/channel.json's own `res` field is always H -- convert for
    // AMR pages here rather than making the benchmark data page-shape-aware.
    const resParam = config.chanPage === 'index-channel-amr.html' ? Math.log2(res) : res;
    const url = `${opts.baseUrl}/${config.chanPage}?mode=${config.chanMode}&res=${resParam}${levelsParam}${extraParam(opts)}`;
    await navigateTo(Page, url);
    await waitForGlobal(Runtime, 'window.__CYL', 15000);
    await evalExprChan(Runtime, `window.__CYL.setLive(false)`);
    for (const c of groupCases) {
      const r = await runChanCase(Runtime, { timeout: opts.physicsTimeout }, c, s => console.log('    ' + s));
      // THE MARGIN, PRINTED ON PASS TOO. A gate that prints only a tick
      // cannot answer "did this change help", which is the question every
      // precision or storage change here has to answer -- and CLAUDE.md names
      // these analytic checks, not Cd/St, as the gate for exactly those.
      // Before this the numbers were computed and thrown away unless they
      // failed, so a re-baseline meant driving the pages from a second,
      // parallel harness that did not share this one's case filtering (and
      // got different answers for that reason).
      console.log(`      L2rel ${r.l2rel.toExponential(4)} / ${c.l2_tol.toExponential(1)}`
        + `  maxErr ${r.maxErr.toExponential(3)}  ${r.converged ? `converged at ${r.step}` : `NOT CONVERGED (${r.step})`}`);
      results.push(r);
    }
  }
  const ok = results.every(r => r.l2.pass && r.converged);
  return { ok, results };
}

// TGV configs sweep dense/AMR cases from benchmarks/tgv.json, filtered by
// config.tgvFilter -- unlike runChannelPhysics, there's no grouping to do
// (every TGV parameter is page-load-time, and no two cases in the default
// benchmark share an (N,u0,tau[,levels]) tuple), so this just navigates
// once per matching case. See tools/validate-tgv.js's identical per-case
// navigation (kept as a separate copy there since that tool owns its own
// Chrome connection/lifecycle, not sharing this file's Page/tab).
async function runTgvPhysics(Page, Runtime, opts, config) {
  const benchPath = path.join(REPO_ROOT, 'benchmarks', 'tgv.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  const cases = bench.cases.filter(config.tgvFilter);

  const results = [];
  for (const c of cases) {
    // main-tgv-amr.js's ?res= is log2(N), matching every other AMR page's
    // convention -- see runChannelPhysics's identical note.
    const url = (c.levels
      ? `${opts.baseUrl}/index-tgv-amr.html?res=${Math.log2(c.N)}&u0=${c.u0}&tau=${c.tau}&levels=${c.levels}`
      : `${opts.baseUrl}/index-tgv.html?res=${c.N}&u0=${c.u0}&tau=${c.tau}`) + extraParam(opts);
    await navigateTo(Page, url);
    await waitForGlobal(Runtime, 'window.__CYL', 15000);
    await evalExprTgv(Runtime, `window.__CYL.setLive(false)`);
    const r = await runTgvCase(Runtime, { timeout: opts.physicsTimeout }, c, s => console.log('    ' + s));
    // The margin, on PASS too -- see runChannelPhysics' note.
    const num = (v) => (typeof v === 'number' && isFinite(v) ? v.toExponential(4) : String(v));
    console.log('      ' + Object.entries(r)
      .filter(([k, v]) => typeof v === 'number' && k !== 'step')
      .map(([k, v]) => `${k} ${num(v)}`).join('  '));
    results.push({ name: c.name, ...r });
  }
  const ok = results.every(r => r.fieldCheck.pass && r.rateCheck.pass);
  return { ok, results };
}

async function runInvariants(Runtime, opts, global) {
  return runInvariantSweep(Runtime, {
      // Corner (diagonal) 2:1 balance is a hard requirement of ?ghostfree=1 --
      // its bilinear parent stencil reads the parent's corner cell directly --
      // and is NOT required by the default ring path. Asserted exactly when
      // the run is a ghost-free one. See plans/ghost-free.md.
      // Gated by default since B2-2d -- see tools/lib/amr-invariants.js. The
      // ?ghostfree=1 special case is gone with it: that path NEEDED corner
      // balance (its bilinear parent stencil reads the parent's corner cell
      // directly), which is why it alone used to require it. Now everything
      // does, so there is nothing to special-case.
      requireCornerBalance: true,
    steps: opts.invariantSteps,
    checkEvery: opts.invariantCheckEvery,
    timeout: opts.physicsTimeout,
    global,
    // cov/bad come back null (not empty) on a page that exposes no
    // geometry-coverage scan or card-state readback -- print n/a, never OK.
    onCheckpoint: (stepsDone, { diag, bal, cov, closure, quad, bad }) => {
      const corner = bal.cornerOk === undefined ? ''
        : `, corner ${bal.cornerOk ? 'OK' : `${bal.cornerViolations.length}`}`;
      // `n/a` is printed, never nothing: an invariant that silently is not
      // checked is the failure mode this whole exercise kept running into.
      const conv = !diag ? ', pool n/a (no debugReadDiag)' : (!diag.diagEnabled ? ', pool n/a (?diag=1 not set)'
        : `, pool ${diag.poolOk ? 'OK' : `STARVED (${diag.refineStarved} refine(s) refused)`}`);
      console.log(`    step ${stepsDone}: 2:1-balance ${bal.ok ? 'OK' : `FAIL (${bal.violations.length})`}${corner}, ` +
        `coverage ${cov === null ? 'n/a' : cov.ok ? 'OK' : `FAIL (${cov.violations.length})`}, ` +
        `field ${bad === null ? 'n/a' : bad.length ? `FAIL (${bad.join(',')})` : 'OK'}${conv}` +
        `${closure === null ? '' : `, closure ${closure.ok ? 'OK' : `${closure.missing} missing`}`}` +
        `${quad === null ? '' : `, quadrants ${quad.ok ? 'OK' : `FAIL (${quad.violations.length})`}`}`);
    },
  });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const allConfigs = defaultConfigs(opts.baseUrl);
  // --extra=f16=1 (say) appends to every config's URL, including the ones
  // that build their own URLs per case (channel/tgv), so an override can be
  // swept across the whole suite without a second copy of the table.
  if (opts.extra) {
    for (const c of allConfigs) c.url += (c.url.includes('?') ? '&' : '?') + opts.extra;
    console.log(`(appending "${opts.extra}" to every config URL)`);
  }
  const configs = opts.configs ? allConfigs.filter(c => opts.configs.includes(c.name)) : allConfigs;
  if (configs.length === 0) { console.error('No matching configs (check --configs= names against the default list in this file).'); process.exit(1); }

  const server = await ensureServer(opts.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(opts.port);
  // Chrome takes a moment past the debug port coming up before it's actually
  // ready to serve WebGPU pages -- same settle time the webgpu-verify skill
  // itself waits after launch.
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));

  const report = [];

  // ONE tab for the entire run, reused via Page.navigate between configs --
  // see ensureChrome's own header for why (this replaced an earlier version
  // that opened a new tab per config and left every prior one running
  // concurrently). If Chrome was already running (not ours), open exactly
  // one new tab rather than disturbing whatever the caller already had open;
  // if we launched Chrome ourselves, its one about:blank tab IS that tab.
  const tabId = chrome.started ? await firstTab(opts.port) : await openTab(opts.port, 'about:blank');
  const client = await CDP({ local: true, port: opts.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  // Registered once, not per-config -- re-registering inside the loop would
  // stack one listener per prior config by the end of the run, logging each
  // later exception that many times over.
  let currentConfigName = null;
  Runtime.exceptionThrown(e => console.error(`  [${currentConfigName}] browser exception:`, e.exceptionDetails.text));

  try {
    for (const config of configs) {
      console.log(`\n=== ${config.name} (${config.url}) ===`);
      currentConfigName = config.name;

      let physics = null, invariants = null, boot = null, renderLevels = null;

      if (config.checkChannelPhysics) {
        // Owns its own per-(mode,res) navigation -- see runChannelPhysics's
        // header for why this can't share the single-navigateTo pattern
        // every other config below uses.
        console.log('  -- physics (u(y) vs. analytic) --');
        physics = await runChannelPhysics(Page, Runtime, opts, config);
      } else if (config.checkTgvPhysics) {
        console.log('  -- physics (field vs. analytic TGV solution) --');
        physics = await runTgvPhysics(Page, Runtime, opts, config);
      } else {
        await navigateTo(Page, config.url);
        if (config.checkBoots) {
          console.log('  -- boot smoke (#status advancing, no error) --');
          boot = await runBootSmoke(Runtime, config.expectError || null);
          console.log(`    ${boot.ok ? 'OK' : 'FAIL: ' + boot.reason}`);
        } else {
          await waitForCYL(Runtime, 15000, config.global);
          if (config.checkPhysics) {
            console.log('  -- physics (Cd/St) --');
            physics = await runPhysics(Runtime, opts);
          }
          if (config.checkRenderLevels) {
            console.log('  -- render reachability (every level reaches the picture) --');
            renderLevels = await runRenderLevels({
              Page, Runtime, global: config.global, steps: opts.renderSteps || 4096,
              log: (m) => console.log('  ' + m),
            });
            for (const r of renderLevels.rows) console.log(`    level ${r.level}  ${r.verdict.padEnd(8)} ${r.note}`);
            if (renderLevels.aborted) console.log(`    ABORTED: ${renderLevels.aborted}`);
          }
          if (config.checkInvariants) {
            console.log('  -- structural invariants --');
            // Re-navigate with ?diag=1 so the refinement-convergence counters
            // are LIVE. Without the flag every counter stays 0 and `converged`
            // reads true vacuously, which would be a silent false pass. Done as
            // its own navigation rather than by adding diag=1 to the shared URL
            // so the timed physics run above stays free of the atomics -- and
            // the sweep calls reset() anyway, so it starts from step 0 either
            // way.
            const diagUrl = config.url + (config.url.includes('?') ? '&' : '?') + 'diag=1';
            await navigateTo(Page, diagUrl);
            await waitForCYL(Runtime, 15000, config.global);
            invariants = await runInvariants(Runtime, opts, config.global);
          }
        }
      }

      report.push({ config, physics, invariants, boot, renderLevels });
    }
  } finally {
    await client.close();
    // See tools/lib/browser-lifecycle.js's teardown for the profile-dir
    // cleanup rationale (prior manual webgpu-verify sessions across this
    // project's history had left 6.5GB across 70 uncleaned profile dirs
    // under /tmp/vpm-chrome-profile before this owned the whole lifecycle).
    await teardown({ port: opts.port, tabId, chrome, server, keepOpen: opts.keepOpen });
  }

  // --- final report ---
  console.log('\n' + '='.repeat(72));
  console.log('SUMMARY');
  console.log('='.repeat(72));
  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log(pad('config', 26) + pad('boot', 8) + pad('physics', 12) + pad('invariants', 12) + pad('render', 10));
  let allOk = true;
  for (const { config, physics, invariants, boot, renderLevels } of report) {
    const bootStr = boot ? (boot.ok ? 'PASS' : 'FAIL') : 'n/a';
    const physStr = physics ? (physics.ok ? 'PASS' : 'FAIL') : 'n/a';
    const invStr = invariants ? (invariants.ok ? 'PASS' : 'FAIL') : 'n/a';
    // A run that ABORTED is not a pass and not a fail -- it is a comparison
    // that could not be made, and it must not be reported as either. An
    // ABSTAIN row (a level with no tiles) likewise does not pass; only the
    // absence of FAIL rows over at least one real PASS does.
    let renStr = 'n/a';
    if (renderLevels) {
      const bad = renderLevels.rows.filter(r => r.verdict === 'FAIL').length;
      const good = renderLevels.rows.filter(r => r.verdict === 'PASS').length;
      renStr = renderLevels.aborted ? 'ABORTED' : (bad || !good ? 'FAIL' : 'PASS');
    }
    console.log(pad(config.name, 26) + pad(bootStr, 8) + pad(physStr, 12) + pad(invStr, 12) + pad(renStr, 10));
    if (boot && !boot.ok) allOk = false;
    if (renStr === 'FAIL' || renStr === 'ABORTED') allOk = false;
    if (physics && !physics.ok) allOk = false;
    if (invariants && !invariants.ok) allOk = false;
  }

  console.log('\nDetails for anything not PASS:');
  for (const { config, physics, invariants, boot } of report) {
    if (boot && !boot.ok) {
      console.log(`  [${config.name}] boot: ${boot.reason}`);
    }
    if (physics && !physics.ok) {
      console.log(`  [${config.name}] physics:`);
      for (const r of physics.results) {
        if (config.checkChannelPhysics) {
          if (!r.converged) console.log(`    ${r.mode} res=${r.res} Re=${r.re} did not converge within budget (step=${r.step})`);
          if (!r.l2.pass) console.log(`    ${r.mode} res=${r.res} Re=${r.re} L2rel=${r.l2rel.toExponential(3)} tol=${r.l2.tol} FAIL`);
        } else if (config.checkTgvPhysics) {
          if (!r.fieldCheck.pass) console.log(`    ${r.name} N=${r.N} fieldL2rel=${r.maxL2rel.toExponential(3)} tol=${r.fieldCheck.tol} FAIL`);
          if (!r.rateCheck.pass) console.log(`    ${r.name} N=${r.N} decayRateRelErr=${r.rateRelErr.toExponential(3)} tol=${r.rateCheck.tol} FAIL`);
        } else {
          if (!r.cd.pass) console.log(`    Re=${r.re} Cd=${r.cd.measured?.toFixed(3)} target=${r.cd.target}±${r.cd.tol} FAIL`);
          if (!r.st.pass) console.log(`    Re=${r.re} St=${r.st.measured?.toFixed(4)} target=${r.st.target}±${r.st.tol} FAIL`);
        }
      }
    }
    if (invariants && !invariants.ok) {
      console.log(`  [${config.name}] invariants (over ${invariants.stepsDone} steps):`);
      if (invariants.balanceViolations.length) console.log(`    2:1-balance FAIL @ step ${invariants.balanceViolations[0].step}: ${JSON.stringify(invariants.balanceViolations[0].violations.slice(0, 4))}`);
      if (invariants.starvationViolations && invariants.starvationViolations.length) {
        const v = invariants.starvationViolations[0];
        console.log(`    refinement STARVED @ step ${v.step}: ${v.starved} refine(s) refused for want of a pool slot ` +
          `(${v.granted} granted the same round), ${invariants.starvationViolations.length} checkpoint(s) affected.`);
        console.log(`      A refused refine is abandoned silently. For a criterion-driven one that is the pool working as a budget;`);
        console.log(`      for a GEOMETRY-forced one it means a coarse/fine seam through the body, and since B4-3 only the finest`);
        console.log(`      level computes force, so the refused region contributes NOTHING. Raise ?maxFineBlocks= or lower ?levels=.`);
        console.log(`      (The live rAF loop latches on this itself -- makeRefusalWatch -- but debugStepSync does not go through it,`);
        console.log(`      which is why the harness needs its own signal.)`);
      }
      if (invariants.starvationChecked === false) console.log(`    pool starvation: NOT CHECKED (page exposes no debugReadDiag, or ?diag=1 did not take)`);
      if (invariants.requireCornerBalance && invariants.cornerViolations.length) console.log(`    corner 2:1-balance FAIL @ step ${invariants.cornerViolations[0].step}: ${invariants.cornerViolations[0].count} violation(s)`);
      if (invariants.coverageViolations.length) console.log(`    geometry-coverage FAIL @ step ${invariants.coverageViolations[0].step}: ${JSON.stringify(invariants.coverageViolations[0].violations.slice(0, 4))}`);
      if (invariants.closureViolations.length) console.log(`    2:1-closure FAIL @ step ${invariants.closureViolations[0].step}: ${invariants.closureViolations[0].missing} block(s) the rule requires are absent, ${JSON.stringify(invariants.closureViolations[0].byReason)}`);
      if (invariants.quadrantViolations.length) console.log(`    slot-quadrant rule FAIL @ step ${invariants.quadrantViolations[0].step}: ${JSON.stringify(invariants.quadrantViolations[0].violations.slice(0, 4))}`);
      if (invariants.fieldViolations.length) console.log(`    field blowup @ step ${invariants.fieldViolations[0].step}: ${JSON.stringify(invariants.fieldViolations[0])}`);
    }
  }

  console.log(allOk ? '\nAll configs PASS.' : '\nSome configs FAILED -- see above.');
  process.exit(allOk ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
