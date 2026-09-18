#!/usr/bin/env node
// Does every configured refinement level have a path into the PICTURE?
//
// THIS GATE FAILS TODAY, ON PURPOSE. `shaders/amr_render.wgsl` binds exactly
// three velocity sources -- the dense L0 `vel`, `vel_pool` (level 1) and
// `vel_pool2` (level 2) -- and `main-amr.js`'s `renBG` wires them to
// `velBuf`, `pools[1]` and `pools[2]` by NUMBER, not by depth. There is no
// `vel_pool3`. So at `?levels=4` a level-3 tile is refined, solved, stepped
// twice per level-2 substep, force-reduced and invariant-checked -- and then
// drawn as though it were its level-2 parent. The finest level in the
// hierarchy, which is the whole reason the hierarchy exists, is invisible.
//
// Closed by plans/uniform-levels.md U6 (the renderer walks levels). Kept OUT
// of tools/validate-all.js's default sweep until then, the same way a
// known-red gate is kept out rather than being allowed to turn the whole
// sweep red -- but it is a gate, it PASS/FAILs, and it must go INTO the sweep
// in the same commit that fixes the renderer.
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
const crypto = require('crypto');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

// Far outside anything this solver produces: the card page runs at |u| of a
// few 1e-2, and the colour map saturates long before 9. A perturbation the
// palette could swallow would answer a different question than the one asked.
const PERTURB = 9.0;

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

async function evalOrThrow(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) {
    throw new Error(`page threw evaluating ${expr.slice(0, 120)}: ${r.exceptionDetails.text} ${r.exceptionDetails.exception && r.exceptionDetails.exception.description || ''}`);
  }
  return r.result.value;
}

// One PNG of the sim canvas alone. The status line and the control panel are
// outside the clip, so a step counter or an FPS readout cannot be mistaken for
// a change in the field.
async function shotCanvas(Page, Runtime) {
  const clip = await evalOrThrow(Runtime, `(() => {
    const el = document.getElementById('c');
    if (!el) throw new Error('no #c canvas on this page');
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  })()`);
  if (!clip.width || !clip.height) throw new Error(`#c has zero area (${clip.width}x${clip.height}) -- is the page laid out?`);
  const { data } = await Page.captureScreenshot({
    format: 'png',
    clip: { x: clip.x, y: clip.y, width: clip.width, height: clip.height, scale: 1 },
    captureBeyondViewport: false,
  });
  return { hash: crypto.createHash('sha256').update(data).digest('hex').slice(0, 16), bytes: data.length };
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

  const rows = [];
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
    const hasHooks = await evalOrThrow(Runtime, `typeof ${o.global}.debugRenderOnce === 'function' && typeof ${o.global}.debugPerturbLevelVel === 'function'`);
    if (!hasHooks) throw new Error('this page exposes no debugRenderOnce/debugPerturbLevelVel -- the gate cannot run against it');

    // Settle refinement so the deep levels actually hold tiles. reset() first:
    // the page's rAF loop runs between load and setLive(false), so reading
    // anything before a reset reads an unknown number of steps of drift.
    console.log(`[setup] reset + ${o.steps} steps to settle refinement`);
    await evalOrThrow(Runtime, `${o.global}.reset()`, 120000);
    await evalOrThrow(Runtime, `${o.global}.debugStepSync(${o.steps})`, 600000);
    await evalOrThrow(Runtime, `${o.global}.setLive(false)`);

    const active = {};
    for (let m = 1; m < nLevels; m++) {
      active[m] = await evalOrThrow(Runtime, `${o.global}.debugListActiveBlocks(${m}).then(a => a.length)`, 120000);
    }
    console.log(`[setup] active tiles by level: ${Object.entries(active).map(([m, n]) => `L${m}=${n}`).join(' ')}`);

    // Keep the snapshot IN THE PAGE. It is megabytes of typed array and there
    // is no reason to move it through CDP twice per level.
    await evalOrThrow(Runtime, `(async () => { window.__RLGATE = await ${o.global}.debugSnapshotSave(); return 1; })()`, 300000);

    // ── 1. is a screenshot even reproducible here? ───────────────────────────
    await evalOrThrow(Runtime, `${o.global}.debugRenderOnce()`, 60000);
    const base1 = await shotCanvas(Page, Runtime);
    await evalOrThrow(Runtime, `${o.global}.debugRenderOnce()`, 60000);
    const base2 = await shotCanvas(Page, Runtime);
    if (base1.hash !== base2.hash) {
      aborted = `two identical renders produced different PNGs (${base1.hash} vs ${base2.hash}). ` +
        `Screenshot equality cannot score anything on this setup; nothing below would mean what it says.`;
      throw new Error(aborted);
    }
    console.log(`[setup] baseline reproducible: ${base1.hash} (${base1.bytes} B)\n`);

    // ── 2. one level at a time ──────────────────────────────────────────────
    for (let m = 1; m < nLevels; m++) {
      if (active[m] === 0) {
        rows.push({ level: m, verdict: 'ABSTAIN', note: 'no active tiles at this level -- nothing to draw, so nothing to prove' });
        continue;
      }

      await evalOrThrow(Runtime, `${o.global}.debugSnapshotLoad(window.__RLGATE)`, 300000);
      await evalOrThrow(Runtime, `${o.global}.debugRenderOnce()`, 60000);
      const restored = await shotCanvas(Page, Runtime);
      if (restored.hash !== base1.hash) {
        rows.push({ level: m, verdict: 'ABSTAIN', note: `restore did not return to baseline (${restored.hash} != ${base1.hash}); the reference is moving` });
        continue;
      }

      const perturbed = await evalOrThrow(Runtime, `JSON.stringify(${o.global}.debugPerturbLevelVel(${m}, ${PERTURB}, ${PERTURB}))`);
      await evalOrThrow(Runtime, `${o.global}.debugRenderOnce()`, 60000);
      const after = await shotCanvas(Page, Runtime);

      const reached = after.hash !== base1.hash;
      rows.push({
        level: m,
        verdict: reached ? 'PASS' : 'FAIL',
        note: reached
          ? `picture changed (${base1.hash} -> ${after.hash})`
          : `picture UNCHANGED after overwriting ${JSON.parse(perturbed).cells} cells with u=(${PERTURB},${PERTURB}) -- this level has no path into the renderer`,
      });
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  // ── report ────────────────────────────────────────────────────────────────
  console.log(`\nrender reachability, ${o.page}${q}\n`);
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

  if (failed.length) {
    console.log(`\nFAIL: level(s) ${failed.map(r => r.level).join(', ')} are solved but never drawn.`);
    console.log('Expected on this branch -- see this file\'s header and plans/uniform-levels.md U6.');
    process.exit(1);
  }
  console.log('\nPASS: every level with active tiles reaches the renderer.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
