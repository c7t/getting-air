#!/usr/bin/env node
// Top-level "check everything" harness: runs the physics regression
// (tools/lib/cylinder-metrics.js, Cd/St vs. benchmarks/cylinder.json) and
// the structural-invariant regression (tools/lib/amr-invariants.js, 2:1
// balance + geometry-forced-refinement coverage + field-finite sanity)
// across a fixed list of scenarios: the dense reference, and the AMR
// cylinder harness at each of the levels/coupling combinations this
// project's own commit history and plans/AMR-multilevel.md milestones
// track as meaningfully different states (N=2 diffuse, N=2 bounce-back --
// both currently validated -- and N=3 diffuse / N=3 bounce-back-forced --
// both currently open work, per main-cylinder-amr.js's own comment above
// N_LEVELS). One command, one report, instead of hand-launching Chrome and
// running tools/validate-cylinder.js and tools/validate-amr-invariants.js
// separately per config.
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

const { spawn } = require('child_process');
const https = require('https');
const path = require('path');
const fs = require('fs');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { evalExpr: evalExprCyl, runCase } = require('./lib/cylinder-metrics');
const { evalExpr, runInvariantSweep } = require('./lib/amr-invariants');

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
    else if (a === '--keepOpen') opts.keepOpen = true;
  }
  return opts;
}

function defaultConfigs(baseUrl) {
  return [
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
      // Known-broken per main-cylinder-amr.js's own guard (throws without
      // ?forceBounceback at levels>2) -- included so the harness's report
      // documents the CURRENT extent of the divergence on every run instead
      // of relying on a comment staying accurate. Cd/St analysis assumes a
      // stable oscillation to average over; skip it here in favor of the
      // invariant sweep's own field-finite check, the right instrument for
      // characterizing a divergence rather than a stable regime.
      name: 'amr-N3-bounceback-forced',
      url: `${baseUrl}/index-cylinder-amr.html?levels=3&bounceback&forceBounceback`,
      checkPhysics: false,
      checkInvariants: true,
    },
  ];
}

// --- process lifecycle: HTTPS dev server + dedicated debug-port Chrome ----

function httpsGetOk(url) {
  return new Promise((resolve) => {
    const req = https.get(url, { rejectUnauthorized: false, timeout: 2000 }, (res) => { res.resume(); resolve(res.statusCode < 500); });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function waitFor(fn, timeoutMs, intervalMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await fn()) return true;
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}

async function ensureServer(baseUrl) {
  if (await httpsGetOk(`${baseUrl}/index.html`)) {
    console.log(`[setup] HTTPS dev server already up at ${baseUrl}`);
    return { started: false, proc: null };
  }
  console.log('[setup] starting https.py dev server');
  const proc = spawn('python3', ['https.py'], { cwd: REPO_ROOT, detached: true, stdio: 'ignore' });
  proc.unref();
  const ok = await waitFor(() => httpsGetOk(`${baseUrl}/index.html`), 10000, 300);
  if (!ok) throw new Error(`https.py did not come up at ${baseUrl} within 10s`);
  return { started: true, proc };
}

async function chromeDebugOk(port) {
  try {
    const res = await fetch(`http://localhost:${port}/json/version`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch { return false; }
}

async function ensureChrome(port, baseUrl) {
  if (await chromeDebugOk(port)) {
    console.log(`[setup] Chrome already listening on debug port ${port}`);
    return { started: false, profileDir: null };
  }
  console.log('[setup] launching dedicated WebGPU-capable Chrome');
  const profileRoot = '/tmp/vpm-chrome-profile';
  fs.mkdirSync(profileRoot, { recursive: true });
  const profileDir = fs.mkdtempSync(path.join(profileRoot, 'validate-all-'));
  const proc = spawn('/opt/google/chrome/chrome', [
    `--remote-debugging-port=${port}`,
    '--enable-features=Vulkan,WebGPUService',
    '--enable-unsafe-webgpu',
    '--ignore-certificate-errors',
    '--no-first-run', '--no-default-browser-check',
    `--user-data-dir=${profileDir}`,
    '--window-size=1400,900',
    `${baseUrl}/index-cylinder.html`, // any valid page -- keeps a tab alive so Chrome doesn't quit before configs open their own
  ], { env: { ...process.env, DISPLAY: process.env.DISPLAY || ':0' }, detached: true, stdio: 'ignore' });
  proc.unref();
  const ok = await waitFor(() => chromeDebugOk(port), 10000, 300);
  if (!ok) throw new Error(`Chrome did not come up on debug port ${port} within 10s`);
  return { started: true, profileDir };
}

async function openTab(port, url) {
  const res = await fetch(`http://localhost:${port}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
  const target = await res.json();
  return target.id;
}

async function closeTab(port, id) {
  try { await fetch(`http://localhost:${port}/json/close/${id}`); } catch { /* best-effort */ }
}

async function waitForCYL(Runtime, timeoutMs) {
  const ok = await waitFor(async () => {
    const r = await evalExpr(Runtime, `typeof window.__CYL !== 'undefined'`);
    return !r.exceptionDetails && r.result.value === true;
  }, timeoutMs, 300);
  if (!ok) throw new Error('window.__CYL never became available (page failed to load or WebGPU init failed)');
}

// --- per-config runners -----------------------------------------------

async function runPhysics(Runtime, opts) {
  const benchPath = path.join(REPO_ROOT, 'benchmarks', 'cylinder.json');
  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  const cases = bench.cases.filter(c => opts.re.includes(c.re));
  await evalExprCyl(Runtime, `window.__CYL.setLive(false)`);
  const results = [];
  for (const c of cases) {
    results.push(await runCase(Runtime, { timeout: opts.physicsTimeout }, c, s => console.log('    ' + s)));
  }
  const ok = results.every(r => r.cd.pass && r.st.pass);
  return { ok, results };
}

async function runInvariants(Runtime, opts) {
  return runInvariantSweep(Runtime, {
    steps: opts.invariantSteps,
    checkEvery: opts.invariantCheckEvery,
    timeout: opts.physicsTimeout,
    onCheckpoint: (stepsDone, { bal, cov, bad }) => {
      console.log(`    step ${stepsDone}: 2:1-balance ${bal.ok ? 'OK' : `FAIL (${bal.violations.length})`}, ` +
        `coverage ${cov.ok ? 'OK' : `FAIL (${cov.violations.length})`}, ` +
        `field ${bad.length ? `FAIL (${bad.join(',')})` : 'OK'}`);
    },
  });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const allConfigs = defaultConfigs(opts.baseUrl);
  const configs = opts.configs ? allConfigs.filter(c => opts.configs.includes(c.name)) : allConfigs;
  if (configs.length === 0) { console.error('No matching configs (check --configs= names against the default list in this file).'); process.exit(1); }

  const server = await ensureServer(opts.baseUrl);
  const chrome = await ensureChrome(opts.port, opts.baseUrl);
  // Chrome takes a moment past the debug port coming up before it's actually
  // ready to serve WebGPU pages -- same settle time the webgpu-verify skill
  // itself waits after launch.
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));

  const report = [];
  const openTabs = [];

  try {
    for (const config of configs) {
      console.log(`\n=== ${config.name} (${config.url}) ===`);
      const tabId = await openTab(opts.port, config.url);
      openTabs.push(tabId);
      const client = await CDP({ port: opts.port, target: tabId });
      const { Runtime } = client;
      await Runtime.enable();
      Runtime.exceptionThrown(e => console.error(`  [${config.name}] browser exception:`, e.exceptionDetails.text));

      let physics = null, invariants = null;
      try {
        await waitForCYL(Runtime, 15000);

        if (config.checkPhysics) {
          console.log('  -- physics (Cd/St) --');
          physics = await runPhysics(Runtime, opts);
        }
        if (config.checkInvariants) {
          console.log('  -- structural invariants --');
          invariants = await runInvariants(Runtime, opts);
        }
      } finally {
        await client.close();
      }

      report.push({ config, physics, invariants });
    }
  } finally {
    for (const id of openTabs) await closeTab(opts.port, id);
    if (!opts.keepOpen) {
      if (chrome.started) {
        // Closing every tab we opened leaves Chrome with none left (we
        // seeded it with one extra tab in ensureChrome specifically to
        // avoid it quitting mid-run when a config's own tab closes) --
        // fetch a fresh list and close whatever's left, then it exits on
        // its own once its last window is gone.
        try {
          const res = await fetch(`http://localhost:${opts.port}/json/list`);
          const targets = await res.json();
          for (const t of targets) await closeTab(opts.port, t.id);
        } catch { /* best-effort cleanup */ }
        // Give the browser process a moment to actually exit once its last
        // tab closes before removing the profile dir out from under it --
        // and remove it even on a clean exit: --user-data-dir profiles are
        // never reused across runs (a fresh mkdtemp every launch), so
        // leaving them behind is pure accumulation, not caching anything.
        // Live-measured during this tool's own development: prior manual
        // webgpu-verify sessions across this project's history had left
        // 6.5GB across 70 uncleaned profile dirs under /tmp/vpm-chrome-
        // profile -- this owns the whole Chrome lifecycle, so it's the one
        // place that can safely clean up after itself instead of relying on
        // whoever's driving it to remember the skill's own manual cleanup step.
        await new Promise(r => setTimeout(r, 1000));
        if (chrome.profileDir) { try { fs.rmSync(chrome.profileDir, { recursive: true, force: true }); } catch { /* best-effort cleanup */ } }
      }
      if (server.started && server.proc) { try { process.kill(-server.proc.pid); } catch { /* already gone */ } }
    }
  }

  // --- final report ---
  console.log('\n' + '='.repeat(72));
  console.log('SUMMARY');
  console.log('='.repeat(72));
  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log(pad('config', 26) + pad('physics', 12) + pad('invariants', 12));
  let allOk = true;
  for (const { config, physics, invariants } of report) {
    const physStr = physics ? (physics.ok ? 'PASS' : 'FAIL') : 'n/a';
    const invStr = invariants ? (invariants.ok ? 'PASS' : 'FAIL') : 'n/a';
    console.log(pad(config.name, 26) + pad(physStr, 12) + pad(invStr, 12));
    if (physics && !physics.ok) allOk = false;
    if (invariants && !invariants.ok) allOk = false;
  }

  console.log('\nDetails for anything not PASS:');
  for (const { config, physics, invariants } of report) {
    if (physics && !physics.ok) {
      console.log(`  [${config.name}] physics:`);
      for (const r of physics.results) {
        if (!r.cd.pass) console.log(`    Re=${r.re} Cd=${r.cd.measured?.toFixed(3)} target=${r.cd.target}±${r.cd.tol} FAIL`);
        if (!r.st.pass) console.log(`    Re=${r.re} St=${r.st.measured?.toFixed(4)} target=${r.st.target}±${r.st.tol} FAIL`);
      }
    }
    if (invariants && !invariants.ok) {
      console.log(`  [${config.name}] invariants (over ${invariants.stepsDone} steps):`);
      if (invariants.balanceViolations.length) console.log(`    2:1-balance FAIL @ step ${invariants.balanceViolations[0].step}: ${JSON.stringify(invariants.balanceViolations[0].violations.slice(0, 4))}`);
      if (invariants.coverageViolations.length) console.log(`    geometry-coverage FAIL @ step ${invariants.coverageViolations[0].step}: ${JSON.stringify(invariants.coverageViolations[0].violations.slice(0, 4))}`);
      if (invariants.fieldViolations.length) console.log(`    field blowup @ step ${invariants.fieldViolations[0].step}: ${JSON.stringify(invariants.fieldViolations[0])}`);
    }
  }

  console.log(allOk ? '\nAll configs PASS.' : '\nSome configs FAILED -- see above.');
  process.exit(allOk ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
