#!/usr/bin/env node
// Physical-observable regression check for the LBM solver(s): drives
// index-cylinder.html (see main-cylinder.js) through a CDP-attached,
// WebGPU-capable Chrome, runs a pinned circular cylinder in uniform
// crossflow at each Reynolds number in benchmarks/cylinder.json, computes
// the time-averaged drag coefficient and (where shedding occurs) the
// Strouhal number, and diffs both against the literature values in that
// file. This is the same "run it, compare to a reference, print PASS/FAIL"
// idea as tools/amr-diff.js's snapshot regression, but at the level of Cd/St
// instead of raw field values -- it validates that the physics is *correct*,
// not just that the AMR build matches a prior run of itself.
//
// The actual Cd/St analysis (transient/measurement window sizing, Strouhal
// zero-crossing estimator, tolerance check) lives in tools/lib/cylinder-
// metrics.js, shared with tools/validate-all.js so the top-level harness
// runs the identical analysis against multiple pages/configs rather than a
// second, independently-drifting copy.
//
// Requires a WebGPU-capable Chrome already running with
// --remote-debugging-port and index-cylinder.html loaded (see
// .claude/skills/webgpu-verify).
//
// Usage:
//   node tools/validate-cylinder.js [--url=http://localhost:8000/index-cylinder.html]
//     [--port=9333] [--re=20,40,100,200] [--timeout=300]

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { evalExpr, runCase } = require('./lib/cylinder-metrics');

function parseArgs(argv) {
  const opts = {
    url: 'http://localhost:8000/index-cylinder.html',
    port: 9333,
    re: null,
    timeout: 300,
  };
  for (const a of argv) {
    if (a.startsWith('--url=')) opts.url = a.slice(6);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--re=')) opts.re = a.slice(5).split(',').map(Number);
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
  const benchPath = path.join(__dirname, '..', 'benchmarks', 'cylinder.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  const cases = opts.re ? bench.cases.filter(c => opts.re.includes(c.re)) : bench.cases;
  if (cases.length === 0) { console.error('No matching cases in benchmarks/cylinder.json'); process.exit(1); }

  const client = await connect(opts.port);
  await new Promise(r => setTimeout(r, 1000));
  await evalExpr(client.Runtime, `window.__CYL.setLive(false)`);

  const results = [];
  for (const c of cases) {
    results.push(await runCase(client.Runtime, opts, c, s => console.log(s)));
  }
  await client.close();

  // Node's console.log %s doesn't support printf-style width modifiers
  // (%-6s prints literally, it doesn't left-pad) -- pad manually instead.
  // padEnd(n) with a trailing space guarantees a separator even when a
  // field (e.g. the "periodic vortex shedding..." regime string) overflows
  // its target column width.
  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log('\n' + pad('Re', 6) + pad('regime', 24) + pad('Cd (target ± tol)', 22) + pad('St (target ± tol)', 22));
  let allPass = true;
  for (const r of results) {
    const cdStr = `${r.cd.measured.toFixed(3)} (${r.cd.target?.toFixed(3) ?? '-'} ± ${r.cd.tol ?? '-'}) ${r.cd.pass ? 'PASS' : 'FAIL'}`;
    const stStr = r.st.target == null
      ? 'n/a'
      : `${r.st.measured?.toFixed(4) ?? 'NONE'} (${r.st.target.toFixed(3)} ± ${r.st.tol}) ${r.st.pass ? 'PASS' : 'FAIL'}`;
    console.log(pad(r.re, 6) + pad(r.regime, 24) + pad(cdStr, 22) + pad(stStr, 22));
    if (!r.cd.pass || !r.st.pass) allPass = false;
  }
  console.log(allPass ? '\nAll cases within tolerance.' : '\nSome cases OUT OF TOLERANCE -- see above.');
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
