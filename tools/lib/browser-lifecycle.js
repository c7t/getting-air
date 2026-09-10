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

// Every debug Chrome this file launches gets a fresh mkdtemp profile under
// this one root, which is also what makes ours identifiable: a Chrome whose
// --user-data-dir lives here is ours, and any other Chrome (the user's own
// browsing session, say) is not and must never be touched.
const PROFILE_ROOT = '/tmp/vpm-chrome-profile';

// Profile dirs of Chromes that are still running. Reading /proc is Linux-
// specific, which this file already is (it hardcodes /opt/google/chrome).
function liveProfileDirs() {
  const live = new Set();
  let pids = [];
  try { pids = fs.readdirSync('/proc').filter(d => /^\d+$/.test(d)); } catch { return live; }
  for (const pid of pids) {
    let cmd = '';
    try { cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8'); } catch { continue; }
    const m = cmd.split('\0').find(a => a.startsWith(`--user-data-dir=${PROFILE_ROOT}/`));
    if (m) live.add(m.slice('--user-data-dir='.length));
  }
  return live;
}

// Deletes profile dirs no live Chrome is using. Called on every launch, which
// is what stops them accumulating.
//
// WHY THIS IS NEEDED AT ALL. teardown() only cleans up a Chrome THIS process
// started, and ensureChrome deliberately REUSES one that is already listening
// (that is what keeps a single WebGPU context alive across a run). So the
// first run to be killed part-way -- a timeout, a Ctrl-C, a crash before
// teardown -- orphans both its Chrome and its profile dir, and every later run
// then adopts that Chrome without owning it and never cleans up either. The
// dirs are ~150MB each; a session that killed a few runs left 712MB behind
// before this existed. Sweeping at launch is safe precisely because a dir with
// no live owner cannot be in use.
function sweepStaleProfiles() {
  let entries = [];
  try { entries = fs.readdirSync(PROFILE_ROOT); } catch { return 0; }
  const live = liveProfileDirs();
  let removed = 0;
  for (const e of entries) {
    const dir = path.join(PROFILE_ROOT, e);
    if (live.has(dir)) continue;
    try { fs.rmSync(dir, { recursive: true, force: true }); removed++; } catch { /* best-effort */ }
  }
  return removed;
}

// Kills every debug Chrome this file is responsible for (identified by its
// profile dir, never by name) and clears the root. Exposed for `make
// chrome-clean` -- the explicit "I am done, reclaim the GPU" button, since no
// single run can safely decide to kill a Chrome another run may be adopting.
function reapAllChromes() {
  const live = liveProfileDirs();
  let killed = 0;
  let pids = [];
  try { pids = fs.readdirSync('/proc').filter(d => /^\d+$/.test(d)); } catch { /* nothing to do */ }
  for (const pid of pids) {
    let cmd = '';
    try { cmd = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8'); } catch { continue; }
    if (!cmd.split('\0').some(a => a.startsWith(`--user-data-dir=${PROFILE_ROOT}/`))) continue;
    try { process.kill(Number(pid), 'SIGTERM'); killed++; } catch { /* already gone */ }
  }
  return { killed, profiles: live.size };
}

async function ensureChrome(port) {
  if (await chromeDebugOk(port)) {
    console.log(`[setup] Chrome already listening on debug port ${port}`);
    return { started: false, profileDir: null, pid: null };
  }
  console.log('[setup] launching dedicated WebGPU-capable Chrome');
  const profileRoot = PROFILE_ROOT;
  fs.mkdirSync(profileRoot, { recursive: true });
  const swept = sweepStaleProfiles();
  if (swept) console.log(`[setup] removed ${swept} stale Chrome profile dir(s) with no live owner`);
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

// Attaches every channel a page can report a failure on, and collects them
// into one list.
//
// WHY THIS EXISTS. Runtime.exceptionThrown -- which is what every tool here
// listened to -- does NOT see the failures this project actually produces.
// Each page wraps its own startup as `init().catch(handleErr)`, so a bad URL
// parameter, a failed shader fetch or a bind-group mismatch becomes a
// console.error plus an `error: ...` line in #status, and NOTHING is thrown
// uncaught. tools/validate-all.js's boot smoke already knows this and polls
// #status for exactly that reason; every other tool was blind to it, which
// in practice meant a human noticed the red box on the page and pasted it
// back. That is not a tool.
//
// So: exceptions, console.error/assert, browser log entries at error level,
// failed subresource loads, and any 4xx/5xx response. Plus statusError()
// below for the #status channel, which is the one none of the CDP domains
// can see.
//
// Also disables the HTTP cache. https.py sends no cache headers, so Chrome
// is free to reuse a main*.js from its memory cache across a navigation --
// which means a tool can validate the PREVIOUS version of the file it was
// asked about, and report a stale pass or a stale failure. Not a hypothetical.
async function attachPageWatch(client, { label, onError } = {}) {
  const { Runtime, Log, Network } = client;
  const errors = [];
  const push = (kind, text) => {
    const e = { kind, text: String(text).trim(), label };
    errors.push(e);
    if (onError) onError(e);
  };

  Runtime.exceptionThrown((e) => {
    const d = e.exceptionDetails;
    push('exception', (d.exception && d.exception.description) || d.text);
  });
  Runtime.consoleAPICalled((e) => {
    if (e.type !== 'error' && e.type !== 'assert') return;
    push(`console.${e.type}`, e.args.map(a =>
      a.description || (a.value !== undefined ? a.value : a.type)).join(' '));
  });

  if (Log) {
    await Log.enable();
    Log.entryAdded((e) => { if (e.entry.level === 'error') push('log', e.entry.text); });
  }
  if (Network) {
    await Network.enable();
    await Network.setCacheDisabled({ cacheDisabled: true });
    Network.loadingFailed((e) => {
      if (e.canceled) return;
      push('net', `${e.type} failed: ${e.errorText}`);
    });
    Network.responseReceived((e) => {
      if (e.response.status >= 400) push('net', `HTTP ${e.response.status} for ${e.response.url}`);
    });
  }

  return {
    errors,
    // Everything seen since the last drain, then forget it -- so a
    // per-config loop attributes failures to the config that caused them.
    drain() { return errors.splice(0, errors.length); },
    get count() { return errors.length; },
  };
}

// The #status channel, which no CDP domain reports: every page writes
// `error: ...` there on a caught init failure (see error-overlay.mjs, whose
// overlay is deliberately additive to this line precisely so tools can keep
// reading it). Returns the message, or null if the page is healthy.
async function statusError(Runtime) {
  const r = await evalExpr(Runtime, `(() => {
    const el = document.getElementById('status');
    return el ? el.textContent : null;
  })()`);
  if (r.exceptionDetails) return null;
  const text = r.result.value;
  return (typeof text === 'string' && /^error:/i.test(text)) ? text : null;
}

// One call for "did this page come up cleanly": both channels, formatted.
// Throws, because every caller wants to stop rather than continue against a
// page that failed to start.
async function assertPageHealthy(Runtime, watch, what) {
  const st = await statusError(Runtime);
  const seen = watch ? watch.drain() : [];
  const lines = [];
  if (st) lines.push(`  #status: ${st}`);
  for (const e of seen) lines.push(`  [${e.kind}] ${e.text.split('\n')[0]}`);
  if (lines.length) throw new Error(`${what} reported errors:\n${lines.join('\n')}`);
}

// Cleans up whatever ensureServer/ensureChrome started (leaves alone anything
// that was already running before this process touched it -- a run that
// ADOPTED a Chrome must not kill it, since another run may be adopting the
// same one, and the webgpu-verify skill deliberately leaves one open for
// manual driving. `make chrome-clean` is the explicit way to reclaim those).
//
// The stale dirs an adopted-then-orphaned Chrome leaves behind are swept by
// the next ensureChrome; see sweepStaleProfiles.
async function teardown({ port, tabId, chrome, server, keepOpen }) {
  if (keepOpen) return;
  await closeTab(port, tabId);
  if (chrome.started) {
    // Group kill: Chrome is spawned detached, so it leads its own process
    // group and its renderer/GPU children belong to it. Signalling just the
    // parent pid left those children alive holding a GPU context.
    if (chrome.pid) {
      try { process.kill(-chrome.pid, 'SIGTERM'); }
      catch { try { process.kill(chrome.pid, 'SIGTERM'); } catch { /* already gone */ } }
    }
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
  attachPageWatch, statusError, assertPageHealthy,
  PROFILE_ROOT, liveProfileDirs, sweepStaleProfiles, reapAllChromes,
};
