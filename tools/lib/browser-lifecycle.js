// Chrome/HTTPS-dev-server lifecycle helpers, factored out of
// tools/validate-all.js (which used to define all of these locally) so
// tools/validate-amr-vs-dense.js can own the same kind of "start the whole
// thing, run everything through one tab, tear down what I started" lifecycle
// without a second, independently-drifting copy. Behavior-preserving
// relocation -- see git history for tools/validate-all.js's own prior
// versions of these functions; no logic changed here, only the module
// boundary.
//
// The hard invariant every caller of these relies on: ONE Chrome tab is
// reused via Page.navigate across an entire run, never more than one
// WebGPU context alive on the GPU at a time (a prior version of
// validate-all.js opened a fresh tab per config and only closed them all at
// the end, which left every earlier config's tab -- and its GPU-resident
// buffers -- running concurrently with whatever was currently under test).

const { spawn } = require('child_process');
const https = require('https');
const path = require('path');
const fs = require('fs');

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

async function ensureServer(baseUrl, repoRoot) {
  if (await httpsGetOk(`${baseUrl}/index.html`)) {
    console.log(`[setup] HTTPS dev server already up at ${baseUrl}`);
    return { started: false, proc: null };
  }
  console.log('[setup] starting https.py dev server');
  const proc = spawn('python3', ['https.py'], { cwd: repoRoot, detached: true, stdio: 'ignore' });
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

async function ensureChrome(port) {
  if (await chromeDebugOk(port)) {
    console.log(`[setup] Chrome already listening on debug port ${port}`);
    return { started: false, profileDir: null, pid: null };
  }
  console.log('[setup] launching dedicated WebGPU-capable Chrome');
  const profileRoot = '/tmp/vpm-chrome-profile';
  fs.mkdirSync(profileRoot, { recursive: true });
  const profileDir = fs.mkdtempSync(path.join(profileRoot, 'validate-all-'));
  // about:blank, not a config's own URL -- callers drive ONE tab for the
  // whole run (Page.navigate between configs, see navigateTo), never more
  // than one WebGPU context alive at once.
  const proc = spawn('/opt/google/chrome/chrome', [
    `--remote-debugging-port=${port}`,
    '--enable-features=Vulkan,WebGPUService',
    '--enable-unsafe-webgpu',
    '--ignore-certificate-errors',
    '--no-first-run', '--no-default-browser-check',
    `--user-data-dir=${profileDir}`,
    '--window-size=1400,900',
    'about:blank',
  ], { env: { ...process.env, DISPLAY: process.env.DISPLAY || ':0' }, detached: true, stdio: 'ignore' });
  proc.unref();
  const ok = await waitFor(() => chromeDebugOk(port), 10000, 300);
  if (!ok) throw new Error(`Chrome did not come up on debug port ${port} within 10s`);
  return { started: true, profileDir, pid: proc.pid };
}

async function openTab(port, url) {
  const res = await fetch(`http://localhost:${port}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
  const target = await res.json();
  return target.id;
}

async function firstTab(port) {
  const res = await fetch(`http://localhost:${port}/json/list`);
  const targets = await res.json();
  const page = targets.find(t => t.type === 'page');
  if (!page) throw new Error('no page target found on debug port ' + port);
  return page.id;
}

async function closeTab(port, id) {
  try { await fetch(`http://localhost:${port}/json/close/${id}`); } catch { /* best-effort */ }
}

// Navigates the SAME tab to a new URL and waits for the load event -- meant
// to be reused between every config so only one page (one WebGPU context)
// is ever alive at a time. Plain Page.navigate to a genuinely different URL
// each call, not the "repeated navigate to the same URL" pattern the
// webgpu-verify skill warns is flaky (that gotcha is about reloading a tab
// whose live sim state you want to preserve; every caller here starts each
// config fresh anyway).
async function navigateTo(Page, url) {
  await Page.navigate({ url });
  await Page.loadEventFired();
}

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

// Generalized form of the "wait for window.__CYL to exist" pattern both
// validate-all.js and validate-amr-vs-dense.js need -- takes the global
// expression to poll (e.g. 'window.__CYL', 'window.__AMR') rather than
// hardcoding one.
async function waitForGlobal(Runtime, globalExpr, timeoutMs) {
  const ok = await waitFor(async () => {
    const r = await evalExpr(Runtime, `typeof ${globalExpr} !== 'undefined'`);
    return !r.exceptionDetails && r.result.value === true;
  }, timeoutMs, 300);
  if (!ok) throw new Error(`${globalExpr} never became available (page failed to load or WebGPU init failed)`);
}

// Cleans up whatever ensureServer/ensureChrome started (leaves alone
// anything that was already running before this process touched it).
async function teardown({ port, tabId, chrome, server, keepOpen }) {
  if (keepOpen) return;
  await closeTab(port, tabId);
  if (chrome.started) {
    if (chrome.pid) { try { process.kill(chrome.pid); } catch { /* already gone */ } }
    await new Promise(r => setTimeout(r, 1000));
    // --user-data-dir profiles are never reused across runs (a fresh
    // mkdtemp every launch), so leaving them behind is pure accumulation --
    // this is the one place that can safely clean them up.
    if (chrome.profileDir) { try { fs.rmSync(chrome.profileDir, { recursive: true, force: true }); } catch { /* best-effort cleanup */ } }
  }
  if (server.started && server.proc) { try { process.kill(-server.proc.pid); } catch { /* already gone */ } }
}

module.exports = {
  httpsGetOk, waitFor, ensureServer, chromeDebugOk, ensureChrome,
  openTab, firstTab, closeTab, navigateTo, evalExpr, waitForGlobal, teardown,
};
