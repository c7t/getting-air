#!/usr/bin/env node
// U3: does the FINE step kernel, dispatched on the ROOT POOL, reproduce the
// dense L0 kernel EXACTLY? (plans/uniform-levels.md U3)
//
// Mirror the dense grid into the root pool, advance both, and compare word for
// word -- BOTH of the step kernel's outputs, `f` and `vel`. The bar is BIT-IDENTITY, not a tolerance: `amr_step1.wgsl` and
// `amr_step.wgsl` perform the same operations in the same order per cell and
// only the address space moves, so equality is what "the root pool is a
// faithful representation of the dense grid" actually means. A tolerance here
// would pass over an addressing bug worth 2e-3 -- which is exactly what
// happened before U2's mirror was fixed.
//
// `vel` is checked because it is U4's INPUT GATE, not for completeness: the
// criterion differences it, the force reduction integrates over it and the
// digest summarises it, and none of those can be scored on the root while its
// `vel` is unproven -- each would otherwise report a difference that belongs
// to the step. It is 2 interleaved f32 per cell and stays f32 under ?f16=,
// which packs `f` only.
//
// ── ?benchSkip=avg IS LOAD-BEARING, AND IT IS NOT "TURNING OFF THE HARD PART"
//
// Exactly two pipelines bind a dense L0 `f` buffer as WRITABLE: the step
// (`stepBG_ab/ba` binding 2) and the L1->L0 average (`avgBG_targetA/B` binding
// 2). Everything else -- force, interp, criterion, render -- reads. With the
// average skipped the step kernel is the SOLE writer of the dense grid, so
// equality is a complete statement about the two kernels. With it on, the
// difference is whatever `average` writes, and the root pool receives no
// restriction until U4/U5. Both are reported; only the isolated one is gated.
//
// Do NOT reach for `?benchSkip=interp,avg` instead. Skipping interp leaves the
// level-1 pool unfed, the force reduction takes it, and the dense field goes
// 100% non-finite -- at which point this comparison reports ZERO differing
// words and maxAbs 0.0e+0, because identical NaN bit patterns are identical
// words. That is why every row below scores FIELD HEALTH FIRST and why a row
// whose field is not finite is a FAILURE rather than a pass.
//
// ── the control is not optional
//
// `?rootstep=0` mirrors and does not step the root. It must come back DIRTY,
// and in practice it saturates (98.4% of every word). A run where the control
// also reads clean is a broken comparison, not a result -- the same
// discrimination `tools/validate-root-mirror.js` makes for U2, with the
// opposite expectation on the other side.
//
// Be honest about what each control proves. On `f`, ?rootstep=0 leaves a
// MIRRORED field that the dense grid then walks away from -- a genuine stale
// comparison. On `vel` it leaves an UNWRITTEN buffer, since the root step is
// its only writer, so that column is scored against zeros and reads relL2
// exactly 1.0. It still proves the comparison reads live data, which is its
// job, but it is not a stale-field control. The SHIPPED-path row below is:
// there `average` moves the dense `vel` and not the root's, and it reads
// 130941/131072 differing.
//
//     node tools/validate-root-step.js
//     node tools/validate-root-step.js --baseUrl=https://localhost:4471 --port=9371
//     node tools/validate-root-step.js --steps=2048

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

// Deliberately not all the default, for the reason validate-root-mirror.js's
// rungs are not: `levels` varies the pool count, `f16` varies the packing and
// is the rung that caught U2's stride bug, `res` varies the domain and so the
// root block grid, `ghostcopy` varies the streaming path -- and that last one
// is here because it was RED. The root pipeline was inheriting
// `DIRECT_GHOST: 0` from `step1Constants`, which asks a level with no ring to
// read a ghost cell nothing fills. A single-rung version of this gate would
// have been green and shipped it.
const GATED = [
  'levels=2&rootpool=1&rootstep=1&benchSkip=avg',
  'levels=3&rootpool=1&rootstep=1&benchSkip=avg',
  'levels=4&rootpool=1&rootstep=1&benchSkip=avg',
  'levels=2&rootpool=1&rootstep=1&benchSkip=avg&f16=1',
  'levels=2&rootpool=1&rootstep=1&benchSkip=avg&f16=2',
  'levels=2&rootpool=1&rootstep=1&benchSkip=avg&res=9',
  'levels=2&rootpool=1&rootstep=1&benchSkip=avg&ghostcopy=1',
  'levels=2&rootpool=1&rootstep=1&benchSkip=avg&dcpre=1',
];
// Must come back DIRTY. Without these the zeros above are unfalsifiable.
const CONTROLS = [
  'levels=2&rootpool=1&rootstep=0&benchSkip=avg',
  'levels=3&rootpool=1&rootstep=0&benchSkip=avg',
];
// Reported, not gated: the shipped path, where `average` writes L0 and the
// root pool does not yet receive a restriction. The number to watch is that
// stepping the root beats freezing it; exactness is U4/U5's to deliver.
const REPORTED = [
  'levels=2&rootpool=1&rootstep=1',
  'levels=2&rootpool=1&rootstep=0',
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9333, page: 'index-amr.html', steps: 512, keepOpen: false };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, timeoutMs) {
  const r = await evalExpr(Runtime, expr, timeoutMs || 300000);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// Non-finite count over the DENSE velocity field. `velB64` is f32 under every
// packing; `fB64` holds packed halves under ?f16= and reading it as f32 is
// meaningless (it reported max|f| 0.006 on the f16=2 rung, which is not a
// population). Velocity is also the field a NaN in `f` reaches immediately.
const HEALTH = `(async () => {
  const s = await window.__AMR.debugSnapshotSave();
  const b = atob(s.velB64);
  const u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
  const v = new Float32Array(u.buffer);
  let nf = 0, mx = 0;
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (!isFinite(x)) { nf++; continue; }
    if (Math.abs(x) > mx) mx = Math.abs(x);
  }
  return JSON.stringify({ nonFinite: nf, n: v.length, maxU: mx });
})()`;

async function runCase(Runtime, Page, o, q) {
  await navigateTo(Page, `${o.baseUrl}/${o.page}?${q}`);
  await waitForGlobal(Runtime, 'window.__AMR', 60000);
  // setLive(false) FIRST and AGAIN after reset(): the rAF loop runs between
  // load and the first eval, and reset() does not stop it. Without this the
  // dense grid advances between the mirror and the comparison, and "the mirror
  // is clean" reads dirty for a reason that has nothing to do with addressing.
  await ev(Runtime, 'window.__AMR.setLive(false)');
  await ev(Runtime, 'window.__AMR.reset()');
  await ev(Runtime, 'window.__AMR.setLive(false)');
  await ev(Runtime, 'window.__AMR.debugMirrorRoot()');
  const seeded = JSON.parse(await ev(Runtime, 'window.__AMR.debugCheckRootMirror().then(r => JSON.stringify(r))'));
  await ev(Runtime, `window.__AMR.debugStepSync(${o.steps})`);
  const health = JSON.parse(await ev(Runtime, HEALTH));
  const after = JSON.parse(await ev(Runtime, 'window.__AMR.debugCheckRootMirror().then(r => JSON.stringify(r))'));
  const vel = JSON.parse(await ev(Runtime, 'window.__AMR.debugCheckRootVel().then(r => JSON.stringify(r))'));
  return { q, seeded, after, vel, health };
}

function fmt(r) {
  const rel = r.after.relL2 == null ? '  (f16)' : r.after.relL2.toExponential(2);
  const abs = r.after.maxAbs == null ? '  (f16)' : r.after.maxAbs.toExponential(2);
  const h = r.health.nonFinite
    ? `FIELD NOT FINITE ${r.health.nonFinite}/${r.health.n}`
    : `max|u| ${r.health.maxU.toFixed(4)}`;
  return `f ${String(r.after.mismatched).padStart(7)}/${r.after.checked}  ${abs}  ${rel}` +
    `   vel ${String(r.vel.mismatched).padStart(6)}/${r.vel.checked}   seeded ${r.seeded.mismatched}   ${h}`;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page, Network } = client;
  await Runtime.enable();
  await Page.enable();
  await Network.enable();
  // Chrome heuristically caches https.py's header-less responses, so a
  // navigate that only changes the query string can run the PREVIOUS build.
  // A U3 fix appeared inert for one whole measurement because of this.
  await Network.setCacheDisabled({ cacheDisabled: true });
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));

  const fails = [];
  try {
    console.log(`  ${o.steps} macro-steps, ${o.page}\n`);
    console.log('  GATED -- must be bit-identical, on a finite field');
    for (const q of GATED) {
      const r = await runCase(Runtime, Page, o, q);
      const ok = r.after.mismatched === 0 && r.vel.mismatched === 0
        && r.seeded.mismatched === 0 && r.health.nonFinite === 0;
      if (!ok) fails.push(q);
      console.log(`${ok ? '  ok  ' : ' FAIL '} ${q.padEnd(46)} ${fmt(r)}`);
    }
    console.log('\n  CONTROL -- must be DIRTY, or the comparison above proves nothing');
    for (const q of CONTROLS) {
      const r = await runCase(Runtime, Page, o, q);
      const ok = r.after.mismatched > 0 && r.vel.mismatched > 0 && r.health.nonFinite === 0;
      if (!ok) fails.push(`${q} (control came back clean)`);
      console.log(`${ok ? '  ok  ' : ' FAIL '} ${q.padEnd(46)} ${fmt(r)}`);
    }
    console.log('\n  REPORTED -- shipped path; the residual is `average`, which the root does not receive (U4/U5)');
    for (const q of REPORTED) {
      const r = await runCase(Runtime, Page, o, q);
      console.log(`       ${q.padEnd(46)} ${fmt(r)}`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('');
  if (fails.length) {
    console.log(`FAIL: ${fails.join(', ')}`);
    process.exit(1);
  }
  console.log('PASS: the fine step kernel on the root pool reproduces the dense kernel exactly.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
