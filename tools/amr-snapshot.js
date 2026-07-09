#!/usr/bin/env node
// CDP-driven CLI for capturing/restoring AMR-dev-build (main-amr.js) fluid
// state snapshots (see main-amr.js's debugSnapshotSave/debugSnapshotLoad).
// Modeled directly on the vpm branch's tools/vpm-snapshot.js. Exists so a
// specific step count only has to be reached once; subsequent numerical
// comparisons (see tools/amr-diff.js) reload it instantly instead of
// re-running the live sim from scratch. This is the verification backbone
// for plans/AMR.md -- every milestone after M0 diffs an AMR-build snapshot
// against a reference snapshot of the same scenario.
//
// Requires a WebGPU-capable Chrome already running with --remote-debugging-
// port and index-amr.html loaded (see .claude/skills/webgpu-verify). Point
// --url at that page.
//
// Usage:
//   node tools/amr-snapshot.js run-to-step <targetStep> <out.json> [--url=http://localhost:8000/index-amr.html] [--port=9333] [--timeout=600]
//   node tools/amr-snapshot.js save <out.json> [--url=...] [--port=9333]
//   node tools/amr-snapshot.js load <in.json> [--url=...] [--port=9333]

const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const fs = require('fs');

function parseArgs(argv) {
  const positional = [];
  const opts = { url: 'http://localhost:8000/index-amr.html', port: 9333, timeout: 600 };
  for (const a of argv) {
    if (a.startsWith('--url=')) opts.url = a.slice(6);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
    else positional.push(a);
  }
  return { positional, opts };
}

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

async function connect(port) {
  const client = await CDP({ port });
  const { Runtime } = client;
  await Runtime.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  return client;
}

// Step count is monotonic (STEPS_PER_FRAME always advances it, no
// remeshing/pruning process exists in the LBM sim to make it non-monotonic
// the way vpm's particle count was) -- first crossing is final, no retry
// loop needed, unlike vpm-snapshot.js's run-to-n.
async function cmdRunToStep(targetStep, outFile, opts) {
  const client = await connect(opts.port);
  const { Runtime } = client;
  await new Promise(r => setTimeout(r, 2000));

  await evalExpr(Runtime, `window.__AMR.reset()`);
  await evalExpr(Runtime, `window.__AMR.setLive(true)`);

  const deadline = Date.now() + opts.timeout * 1000;
  let lastStep = 0, lastLog = 0;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 500));
    const r = await evalExpr(Runtime, `window.__AMR.getStep()`);
    lastStep = r.result.value;
    if (Date.now() - lastLog > 4000) {
      console.log(`  step=${lastStep} (target step=${targetStep})`);
      lastLog = Date.now();
    }
    if (lastStep >= targetStep) break;
  }
  if (lastStep < targetStep) {
    console.error(`Timed out after ${opts.timeout}s at step=${lastStep}, target was ${targetStep}. Saving anyway.`);
  }
  await evalExpr(Runtime, `window.__AMR.setLive(false)`);
  await new Promise(r => setTimeout(r, 100));
  const r = await evalExpr(Runtime, `window.__AMR.debugSnapshotSave()`, 30000);
  fs.writeFileSync(outFile, JSON.stringify(r.result.value));
  console.log(`Saved snapshot to ${outFile}: step=${r.result.value.step}, ${r.result.value.W}x${r.result.value.H}`);
  await client.close();
}

async function cmdSave(outFile, opts) {
  const client = await connect(opts.port);
  const { Runtime } = client;
  await new Promise(r => setTimeout(r, 1500));
  await evalExpr(Runtime, `window.__AMR.setLive(false)`);
  await new Promise(r => setTimeout(r, 100));
  const r = await evalExpr(Runtime, `window.__AMR.debugSnapshotSave()`, 30000);
  fs.writeFileSync(outFile, JSON.stringify(r.result.value));
  console.log(`Saved snapshot to ${outFile}: step=${r.result.value.step}, ${r.result.value.W}x${r.result.value.H}`);
  await client.close();
}

async function cmdLoad(inFile, opts) {
  const client = await connect(opts.port);
  const { Runtime } = client;
  await new Promise(r => setTimeout(r, 1500));
  const snapshot = JSON.parse(fs.readFileSync(inFile, 'utf8'));
  // Pass the (potentially large) snapshot object into the page via a global,
  // rather than inlining it into the expression string -- avoids any
  // string-escaping/size issues with Runtime.evaluate for a large payload.
  await Runtime.evaluate({ expression: `window.__amrSnapshotToLoad = ${JSON.stringify(snapshot)}`, timeout: 30000 });
  const r = await evalExpr(Runtime, `window.__AMR.debugSnapshotLoad(window.__amrSnapshotToLoad)`, 30000);
  console.log(`Loaded snapshot from ${inFile}: step=${r.result.value.step}`);
  await client.close();
}

(async () => {
  const { positional, opts } = parseArgs(process.argv.slice(2));
  const [cmd, ...rest] = positional;
  if (cmd === 'run-to-step') {
    const [targetStep, outFile] = rest;
    if (!targetStep || !outFile) throw new Error('usage: run-to-step <targetStep> <out.json>');
    await cmdRunToStep(parseInt(targetStep), outFile, opts);
  } else if (cmd === 'save') {
    const [outFile] = rest;
    if (!outFile) throw new Error('usage: save <out.json>');
    await cmdSave(outFile, opts);
  } else if (cmd === 'load') {
    const [inFile] = rest;
    if (!inFile) throw new Error('usage: load <in.json>');
    await cmdLoad(inFile, opts);
  } else {
    console.error('usage: amr-snapshot.js <run-to-step|save|load> ...');
    process.exit(1);
  }
})().catch(e => { console.error('FATAL', e); process.exit(1); });
