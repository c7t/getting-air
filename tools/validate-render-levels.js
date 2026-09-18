#!/usr/bin/env node
// Does every configured refinement level have a path into the PICTURE?
//
// CLOSED BY plans/uniform-levels.md U6, AND IN THE DEFAULT SWEEP SINCE. It
// was written red on purpose and stayed red for one stage: `amr_render.wgsl`
// bound exactly three velocity sources -- the dense L0 `vel`, `vel_pool`
// (level 1) and `vel_pool2` (level 2) -- wired by NUMBER, not by depth, with
// no `vel_pool3`. At `?levels=4` a level-3 tile was refined, solved, stepped
// twice per level-2 substep, force-reduced and invariant-checked, and then
// drawn as its level-2 parent.
//
// IT FOUND A SECOND, LARGER INSTANCE WHEN IT WAS FINALLY POINTED AT ANOTHER
// PAGE. Three of the five AMR pages -- cylinder, TGV and channel -- never
// passed the `HAS_LEVEL2` override to the render fragment at all, so it took
// its declared default of 0 and **level 2 was solved and never drawn on any of
// them**. Measured here on `index-cylinder-amr.html?levels=3` before the fix:
// perturbing level 2's whole velocity pool (65600 cells, u=(9,9)) left the
// picture byte-identical, while level 1 moved. That is the page the Cd/St
// numbers come from.
//
// A FAILURE HERE IS NOW A REGRESSION, not the expected state.
//
// ── WHY IT IS SCORED THIS WAY ───────────────────────────────────────────────
//
// The tempting version of this check is static: parse amr_render.wgsl, count
// the `vel_pool*` bindings, compare against N_LEVELS. That tests the shape of
// the source, not the behaviour, and it would keep passing the day someone
// binds a fourth buffer and forgets to sample it. This project has collected
// three vacuous gates that way.
//
// So the claim is tested as a CAPABILITY, end to end: pause the sim, overwrite
// one level's velocity pool with a value no physical flow could produce,
// redraw, and see whether the picture moves. If it does, that level reaches
// the renderer. If it does not, it does not. Nothing about the renderer's
// internals is assumed, so the check survives U6's rewrite unchanged.
//
// Three things make the negative result trustworthy rather than merely absent:
//
//   1. REPRODUCIBILITY FIRST. Two baseline screenshots with nothing changed in
//      between must be byte-identical. If they are not, screenshots cannot
//      score equality on this setup and the run ABORTS rather than reporting a
//      pass or a fail it has not earned.
//   2. THE RESTORE MUST RETURN TO BASELINE. Before each level's perturbation
//      the snapshot is reloaded and re-rendered, and that frame must equal the
//      baseline. A restore that does not restore means every later comparison
//      is against a moving reference.
//   3. DISCRIMINATION. Levels 1 and 2 must go GREEN while level 3 goes RED. A
//      run where every level reports the same thing has not been shown to
//      measure anything -- the same reason the invariant sweep is sanity-
//      checked against a starved pool, where five gates redden and two do not.
//
// A level with no active tiles ABSTAINS and is reported as such. It never
// counts as a pass: "nothing to draw" and "drawn nowhere" are the same
// screenshot, and conflating them is how this check would rot into a gate that
// passes because refinement stopped happening.
//
//     node tools/validate-render-levels.js                    # levels=4, index-amr.html
//     node tools/validate-render-levels.js --levels=3
//     node tools/validate-render-levels.js --page=index-cylinder-amr.html --global=window.__CYL
//     node tools/validate-render-levels.js --baseUrl=https://localhost:4455 --port=9444
//
// VERIFY WHICH TREE THE DEV SERVER IS SERVING before believing a run -- see
// CLAUDE.md. `ensureServer` reuses whatever already answers on the port.

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const { runRenderLevels } = require('./lib/render-levels');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = {
    baseUrl: 'https://localhost:4444',
    port: 9333,
    page: 'index-amr.html',
    global: 'window.__AMR',
    levels: 4,
    steps: 4096,
    extra: '',
    keepOpen: false,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--global=')) o.global = a.slice(9);
    else if (a.startsWith('--levels=')) o.levels = parseInt(a.slice(9));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--extra=')) o.extra = a.slice(8);
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

// evalOrThrow stays here because main() reads N_LEVELS before handing off; the
// perturbation magnitude and the check itself live in tools/lib/render-levels.js.
async function evalOrThrow(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) {
    throw new Error(`page threw evaluating ${expr.slice(0, 120)}: ${r.exceptionDetails.text} ${r.exceptionDetails.exception && r.exceptionDetails.exception.description || ''}`);
  }
  return r.result.value;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const q = `?levels=${o.levels}${o.extra ? '&' + o.extra.replace(/^&/, '') : ''}`;
  const url = `${o.baseUrl}/${o.page}${q}`;

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  let rows = [];
  let aborted = null;

  try {
    console.log(`[setup] ${url}`);
    await navigateTo(Page, url);
    // waitForGlobal THROWS on timeout and returns undefined on success -- do
    // not test its return value, or every healthy page reads as a boot failure.
    await waitForGlobal(Runtime, o.global, 60000);
    const nLevels = await evalOrThrow(Runtime, `${o.global}.getNumLevels()`);
    if (nLevels !== o.levels) {
      console.log(`[warn] page reports N_LEVELS=${nLevels}, asked for ${o.levels} (the page may be refusing the request)`);
    }
    // THE CHECK ITSELF LIVES IN tools/lib/render-levels.js, shared with
    // validate-all.js's default sweep. This file owns the lifecycle and the
    // report; a second copy of the protocol is exactly what tools/lib exists
    // to prevent.
    const r = await runRenderLevels({ Page, Runtime, global: o.global, steps: o.steps, log: console.log });
    rows = r.rows;
    aborted = r.aborted;
    if (aborted) throw new Error(aborted);
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  // ── report ────────────────────────────────────────────────────────────────
  console.log(`\nrender reachability, ${o.page}?levels=${o.levels}\n`);
  for (const r of rows) console.log(`  level ${r.level}  ${r.verdict.padEnd(8)} ${r.note}`);

  const failed = rows.filter(r => r.verdict === 'FAIL');
  const passed = rows.filter(r => r.verdict === 'PASS');
  const abstained = rows.filter(r => r.verdict === 'ABSTAIN');

  console.log('');
  if (abstained.length) console.log(`  ${abstained.length} level(s) abstained -- they are NOT passes.`);

  // Discrimination: a run where every level says the same thing has not been
  // shown to measure anything. Say so rather than reporting a clean verdict.
  if (!passed.length && failed.length) {
    console.log('  NO level reached the renderer. That is more likely a broken instrument than a');
    console.log('  renderer that draws nothing -- check the canvas is not blank before reading this');
    console.log('  as a result about levels.');
  } else if (passed.length && failed.length) {
    console.log(`  Discrimination OK: ${passed.length} level(s) reached the renderer, ${failed.length} did not.`);
  }

  // WHAT KEEPS AN ALL-PASS RUN FROM BEING VACUOUS, now that the expected
  // result is all-PASS and the pass/fail split can no longer supply the
  // discrimination. Two guards, both already enforced above, and they are the
  // reason this still measures something:
  //
  //   the BASELINE is reproducible -- two untouched renders byte-identical, or
  //     the run aborts. Without it "the picture changed" means nothing.
  //   the RESTORE returns to baseline before every level, or that level
  //     ABSTAINS. Without it each level would be compared against the previous
  //     level's perturbation and would "pass" trivially.
  //
  // Neither is a pass/fail row, so they are restated here rather than left to
  // be inferred from a green summary.

  if (failed.length) {
    console.log(`\nFAIL: level(s) ${failed.map(r => r.level).join(', ')} are solved but never drawn.`);
    console.log('This is a REGRESSION -- U6 closed it. Check N_POOL_LEVELS reaches the render');
    console.log('pipeline on this page, and that renBG binds a pair for every level.');
    process.exit(1);
  }
  console.log('\nPASS: every level with active tiles reaches the renderer.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
