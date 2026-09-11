#!/usr/bin/env node
// M0 register/spill spike runner (plans/3D.md sec 2.4, sec 5 M0, risk #1).
//
// Drives index-3d-spike.html in a real WebGPU Chrome and prints one verdict.
// The page does the measuring; this file exists so the answer is one command
// and one report rather than "open a URL and squint at a table" -- and so
// the READING of the table is written down once, in code, instead of being
// re-derived by whoever runs it next.
//
// THE QUESTION. At Q=27 the fused step kernel holds 54 live f32 across two
// `private` arrays indexed by a loop variable. WGSL/Tint keeps such an array
// in registers only if the loop is fully unrolled; if it is not, the array
// spills to scratch memory. That failure is silent -- no error, no wrong
// answer, just a solver running at a fraction of roofline forever after --
// and it is the one outcome that would invalidate the shape of the whole 3D
// design, so plans/3D.md puts it before everything else.
//
// HOW THE TABLE ANSWERS IT. All three kernel modes move exactly the same
// bytes (Q planes in, Q planes out, per cell), so they are directly
// comparable in GB/s and only in GB/s -- GLUPS necessarily falls by 27/19
// between the velocity sets because each cell update carries more bytes,
// and reading that fall as "Q27 is slow" is the mistake this tool is
// written to prevent.
//
//   * `stream` is the access-pattern ceiling: same addresses, no arithmetic.
//   * `collide` has the same register pressure as `full` with perfectly
//     coalesced addresses, so it isolates registers from gather divergence.
//   * Spill signature: Q27 `collide` GB/s well below Q19 `collide` GB/s.
//     Spill traffic is real traffic the roofline cannot see, so it shows up
//     as apparent bandwidth collapsing, not as arithmetic getting slower.
//
// Assumes nothing is running: like tools/validate-all.js (and unlike the
// leaf validators) it owns the HTTPS server and Chrome lifecycle, reusing
// either if already up.
//
// Usage:
//   node tools/spike-d3-registers.js
//   node tools/spike-d3-registers.js --n=96 --reps=7
//   node tools/spike-d3-registers.js --peak=717      # adds a % of peak column
//   node tools/spike-d3-registers.js --wg=8,8,1      # sweep the workgroup shape

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

// Ratio below which a row is called out. Not a physics constant -- a
// judgement about what is worth a human's attention. 0.75 is well outside
// run-to-run noise on this harness (the repeat check below reports the
// actual spread) and well inside the 2-3x a genuine scratch-memory spill
// would produce, so it should neither cry wolf nor miss the thing it is for.
const SPILL_RATIO = 0.75;

function parseArgs(argv) {
  const opts = {
    baseUrl: 'https://localhost:4444', port: 9333,
    n: null, q: null, mode: null, steps: null, reps: null, wg: null, peak: null,
    timeout: 600000, keepOpen: false,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--n=')) opts.n = a.slice(4);
    else if (a.startsWith('--q=')) opts.q = a.slice(4);
    else if (a.startsWith('--mode=')) opts.mode = a.slice(7);
    else if (a.startsWith('--steps=')) opts.steps = a.slice(8);
    else if (a.startsWith('--reps=')) opts.reps = a.slice(7);
    else if (a.startsWith('--wg=')) opts.wg = a.slice(5);
    else if (a.startsWith('--peak=')) opts.peak = a.slice(7);
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') opts.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return opts;
}

function pageUrl(opts) {
  const p = new URLSearchParams();
  for (const k of ['n', 'q', 'mode', 'steps', 'reps', 'wg', 'peak']) if (opts[k] != null) p.set(k, opts[k]);
  const qs = p.toString();
  return `${opts.baseUrl}/index-3d-spike.html${qs ? `?${qs}` : ''}`;
}

const MODE_NAME = ['full', 'stream', 'collide'];
const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);

function printTable(meta, rows, peak) {
  console.log(`\nadapter : ${meta.adapter}`);
  console.log(`grid    : ${meta.n}^3 = ${meta.cells.toLocaleString()} cells` +
    (meta.clampedFrom ? `  (clamped from ${meta.clampedFrom}^3 by the adapter's storage-binding limit)` : ''));
  console.log(`timing  : ${meta.timing}\n`);
  const head = pad('Q', 5) + pad('mode', 9) + padL('GPU ms', 9) + padL('wall ms', 9) +
    padL('GLUPS', 9) + padL('GB/s', 9) + (peak ? padL('% peak', 9) : '') + padL('finite', 8);
  console.log(head);
  console.log('-'.repeat(head.length));
  for (const r of rows) {
    console.log(pad(`Q${r.q}`, 5) + pad(MODE_NAME[r.mode], 9) +
      padL(r.gpuMs == null ? '-' : r.gpuMs.toFixed(3), 9) + padL(r.wallMs.toFixed(3), 9) +
      padL(r.glups.toFixed(3), 9) + padL(r.gbps.toFixed(1), 9) +
      (peak ? padL((100 * r.gbps / peak).toFixed(1), 9) : '') +
      padL(r.finite ? 'ok' : 'NaN!', 8));
  }
}

// The verdict. Written out rather than left to the reader because the whole
// point of M0 is a decision, and a table of six numbers is not a decision.
function verdict(rows, spread) {
  const at = (q, mode) => rows.find(r => r.q === q && r.mode === mode);
  const lines = [];
  let ok = true;

  const nonFinite = rows.filter(r => !r.finite);
  if (nonFinite.length) {
    ok = false;
    lines.push(`FAIL  ${nonFinite.length} row(s) produced non-finite or out-of-range field values -- those timings measure a broken kernel, not a solver.`);
  }

  for (const mode of [0, 2]) {
    const a = at(19, mode), b = at(27, mode);
    if (!a || !b) continue;
    const ratio = b.gbps / a.gbps;
    const label = MODE_NAME[mode];
    if (ratio < SPILL_RATIO) {
      ok = false;
      lines.push(`SPILL?  Q27 ${label} sustains ${b.gbps.toFixed(1)} GB/s against Q19's ${a.gbps.toFixed(1)} (${(100 * ratio).toFixed(0)}%).`);
      lines.push(`        Same bytes per cell-update by construction, so this is bandwidth the roofline cannot see --`);
      lines.push(`        the scratch-memory signature. plans/3D.md sec 2.4: the design would need restructuring around`);
      lines.push(`        storing only \`f\` and writing planes as they are computed.`);
    } else {
      lines.push(`ok      Q27 ${label} holds ${(100 * ratio).toFixed(0)}% of Q19's GB/s -- no spill signature.`);
    }
  }

  const s19 = at(19, 1), s27 = at(27, 1);
  for (const [q, s] of [[19, s19], [27, s27]]) {
    const f = at(q, 0);
    if (!s || !f) continue;
    lines.push(`        Q${q}: full is ${(100 * f.gbps / s.gbps).toFixed(0)}% of the stream-only access-pattern ceiling (${s.gbps.toFixed(1)} GB/s).`);
  }

  if (spread != null) {
    lines.push(`        run-to-run spread on a repeated identical row: ${(100 * spread).toFixed(1)}% -- ratios closer than this mean nothing.`);
  }
  return { ok, lines };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const peak = opts.peak ? parseFloat(opts.peak) : null;
  const server = await ensureServer(opts.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(opts.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));

  const tabId = chrome.started ? await firstTab(opts.port) : await openTab(opts.port, 'about:blank');
  const client = await CDP({ port: opts.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('  browser exception:', e.exceptionDetails.text));

  let exitCode = 0;
  try {
    const url = pageUrl(opts);
    console.log(`[run] ${url}`);
    await navigateTo(Page, url);
    await waitForGlobal(Runtime, 'window.__SPIKE3D', 60000);

    // The page autoruns on load; wait for it rather than kicking off a
    // second run on top of the one already in flight.
    const done = await evalExpr(Runtime, `(async () => {
      const s = window.__SPIKE3D;
      while (!s.done) { await new Promise(r => setTimeout(r, 250)); }
      return { meta: s.meta, results: s.results };
    })()`, opts.timeout);
    if (done.exceptionDetails) throw new Error(`page run failed: ${done.exceptionDetails.text}`);
    const { meta, results } = done.result.value;
    if (!results.length) throw new Error('page reported no measurements');

    printTable(meta, results, peak);

    // A REPEAT of one identical row, so the verdict's ratios can be read
    // against this harness's own noise instead of against an assumption.
    // CLAUDE.md records exactly this discipline for the AMR Cd numbers: a
    // build-vs-build claim needs a same-build repeat.
    let spread = null;
    const first = results[0];
    const rep = await evalExpr(Runtime, `(async () => {
      const s = window.__SPIKE3D;
      s.results.length = 0; s.done = false;
      await s.run();
      return s.results;
    })()`, opts.timeout);
    if (!rep.exceptionDetails && Array.isArray(rep.result.value)) {
      const again = rep.result.value.find(r => r.q === first.q && r.mode === first.mode);
      if (again) {
        spread = Math.abs(again.gbps - first.gbps) / first.gbps;
        console.log(`\nrepeat of Q${first.q} ${MODE_NAME[first.mode]}: ${first.gbps.toFixed(1)} then ${again.gbps.toFixed(1)} GB/s`);
      }
    }

    const v = verdict(results, spread);
    console.log('\n--- verdict ---');
    for (const l of v.lines) console.log(l);
    console.log(`\n${v.ok ? 'PASS' : 'FAIL'}: M0 register/spill spike`);
    exitCode = v.ok ? 0 : 1;
  } catch (e) {
    console.error(`\nFAIL: ${e.message}`);
    exitCode = 1;
  } finally {
    await client.close();
    await teardown({ port: opts.port, tabId, chrome, server, keepOpen: opts.keepOpen });
  }
  process.exit(exitCode);
}

main();
