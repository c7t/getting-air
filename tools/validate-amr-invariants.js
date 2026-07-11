#!/usr/bin/env node
// Structural-invariant regression check for the AMR cylinder harness: drives
// index-cylinder-amr.html through a CDP-attached, WebGPU-capable Chrome and
// asserts window.__CYL.debugCheck21Balance()/debugCheckGeometryCoverage()
// PERIODICALLY through a run, not just once at the end -- unlike
// tools/validate-cylinder.js (which validates the physics via Cd/St), this
// validates the AMR TOPOLOGY: 2:1 balance between neighboring tiles, and the
// geometry-forced-refinement hard constraint (every near-body leaf tile must
// already be at the finest configured level). Both are meant to hold at
// EVERY macro-step, not just in steady state -- checking only at the end
// would miss a transient violation that self-heals before the run finishes.
//
// This is the persisted form of the ad hoc "coverage_check.js-style scan"
// described (but never committed) in the bounce-back N>=3 investigation --
// see main-cylinder-amr.js's own comment above N_LEVELS, and
// debugCheckGeometryCoverage's header.
//
// Also runs a cheap field-sanity smoke check (debugReadCardState: fx/fy/tz/
// vx/vy/omega all finite) after each batch -- notes.txt records real
// instability blowups at high step counts in this project's history, and
// this is a much cheaper trip-wire for that than a Cd/St run.
//
// The sweep itself (reset, step-in-batches, assert-every-batch) lives in
// tools/lib/amr-invariants.js, shared with tools/validate-all.js.
//
// Requires a WebGPU-capable Chrome already running with
// --remote-debugging-port and index-cylinder-amr.html?levels=N[&bounceback]
// already loaded (see .claude/skills/webgpu-verify) -- same convention as
// tools/validate-cylinder.js, one invocation per levels/bounceback
// combination (this script does not navigate the page itself).
//
// Usage:
//   node tools/validate-amr-invariants.js [--port=9333] [--steps=20000]
//     [--checkEvery=512] [--label=levels3]

const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { evalExpr, runInvariantSweep } = require('./lib/amr-invariants');

function parseArgs(argv) {
  const opts = { port: 9333, steps: 20000, checkEvery: 512, label: null, timeout: 300 };
  for (const a of argv) {
    if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) opts.steps = parseInt(a.slice(8));
    else if (a.startsWith('--checkEvery=')) opts.checkEvery = parseInt(a.slice(13));
    else if (a.startsWith('--label=')) opts.label = a.slice(8);
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
  }
  return opts;
}

async function connect(port) {
  const client = await CDP({ port });
  const { Runtime } = client;
  await Runtime.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  return client;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const client = await connect(opts.port);
  const { Runtime } = client;

  const label = opts.label || (await evalExpr(Runtime, `window.__CYL.getNumLevels()`)).result.value;
  console.log(`[${label}] running ${opts.steps} steps, checking invariants every ${opts.checkEvery}`);

  const result = await runInvariantSweep(Runtime, {
    steps: opts.steps,
    checkEvery: opts.checkEvery,
    timeout: opts.timeout,
    onCheckpoint: (stepsDone, { bal, cov, bad }) => {
      console.log(`  step ${stepsDone}: 2:1-balance ${bal.ok ? 'OK' : `FAIL (${bal.violations.length})`}, ` +
        `coverage ${cov.ok ? 'OK' : `FAIL (${cov.violations.length})`}, ` +
        `field ${bad.length ? `FAIL (${bad.join(',')})` : 'OK'}`);
    },
  });

  await client.close();

  console.log(`\n[${label}] summary over ${result.stepsDone} steps:`);
  console.log(`  2:1-balance violations observed at ${result.balanceViolations.length} checkpoint(s)`);
  console.log(`  geometry-coverage violations observed at ${result.coverageViolations.length} checkpoint(s)`);
  console.log(`  field blowups observed at ${result.fieldViolations.length} checkpoint(s)`);

  if (!result.ok) {
    console.log('\nFirst violations:');
    if (result.balanceViolations.length) console.log('  2:1-balance @ step', result.balanceViolations[0].step, JSON.stringify(result.balanceViolations[0].violations.slice(0, 4)));
    if (result.coverageViolations.length) console.log('  geometry-coverage @ step', result.coverageViolations[0].step, JSON.stringify(result.coverageViolations[0].violations.slice(0, 4)));
    if (result.fieldViolations.length) console.log('  field blowup @ step', result.fieldViolations[0].step, JSON.stringify(result.fieldViolations[0]));
  }
  console.log(result.ok ? '\nAll invariants held throughout the run.' : '\nInvariant violations found -- see above.');
  process.exit(result.ok ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
