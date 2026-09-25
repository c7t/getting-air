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
  if (process.env.CHROME_WORKSPACE) await moveToWorkspace(proc.pid, process.env.CHROME_WORKSPACE);
  return { started: true, profileDir, pid: proc.pid };
}

// A new window maps on the CURRENT workspace and takes focus (xfwm4's
// focus_new), and nothing on Chrome's command line prevents it. With
// CHROME_WORKSPACE=N set, the launched window is sent to workspace N
// (0-based, wmctrl's numbering) as soon as it maps, so it flashes up rather
// than sitting on top of the user's work. openTab keeps it from being
// activated -- and so from dragging the user over there -- afterwards.
async function moveToWorkspace(pid, ws) {
  const { execFileSync } = require('child_process');
  const run = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8', env: process.env });
  let home, ids = [];
  try { home = run('wmctrl', ['-d']).split('\n').find(l => /^\d+\s+\*/.test(l)).split(/\s+/)[0]; }
  catch { return; }  // no wmctrl / no EWMH window manager: nothing to do
  const windows = () => run('wmctrl', ['-lp']).split('\n').map(l => l.split(/\s+/))
    .filter(f => f[2] === String(pid)).map(f => parseInt(f[0], 16));
  const active = () => parseInt((run('xprop', ['-root', '_NET_ACTIVE_WINDOW']).match(/0x[0-9a-f]+/i) || ['0'])[0], 16);
  // Wait for the window to map AND take its startup focus: Chrome activates
  // its first window a beat after mapping it, and an activation that lands
  // after the move is exactly the workspace switch this is here to prevent.
  await waitFor(() => { try { ids = windows(); return ids.length > 0 && ids.includes(active()); } catch { return false; } }, 5000, 50);
  if (!ids.length) { console.log(`[setup] CHROME_WORKSPACE=${ws}: Chrome's window never appeared in wmctrl -lp; left where it is`); return; }
  for (const id of ids) { try { run('wmctrl', ['-i', '-r', '0x' + id.toString(16), '-t', String(ws)]); } catch { /* best-effort */ } }
  // Belt and braces: if something still pulled the view over, put it back.
  await new Promise(r => setTimeout(r, 300));
  try { if (!run('wmctrl', ['-d']).split('\n').some(l => l.startsWith(home + ' ') && /^\d+\s+\*/.test(l))) run('wmctrl', ['-s', home]); } catch { /* best-effort */ }
  console.log(`[setup] moved Chrome's window to workspace ${ws} (CHROME_WORKSPACE)`);
}

// A tab a finished run leaves behind for the next one (see teardown). An
// about:blank document, so no WebGPU context survives in it.
const PARKED_URL = 'about:blank#harness-idle';

// Returns a tab for the run WITHOUT raising the debug Chrome's window.
//
// WHY NOT JUST /json/new. Creating a foreground tab ACTIVATES its window, and
// under xfwm4 with activate_action=switch (this desktop's setting) activation
// switches the user to whichever workspace the debug Chrome lives on --
// every run, mid-whatever-they-were-doing. Measured under Xvfb+xfwm4 with the
// same xfconf settings: /json/new, Target.activateTarget and Page.bringToFront
// all switch workspace; Page.navigate on an existing tab does not. The
// no-activation ways to CREATE a tab are no use: a background tab is
// document.hidden, so rAF never fires and a chain of 20 setTimeout(0)s took
// 14 s, and createTarget({newWindow, background}) maps a new window on the
// CURRENT workspace and focuses it. So the only quiet tab is one that already
// exists: reuse a VISIBLE idle one (a parked one, or a bare about:blank) by
// navigating it, and fall back to /json/new -- once per Chrome lifetime,
// since teardown parks the tab instead of closing it.
async function openTab(port, url) {
  const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
  let targets = [];
  try { targets = await (await fetch(`http://localhost:${port}/json/list`)).json(); } catch { /* fall through */ }
  for (const t of targets) {
    if (t.type !== 'page' || (t.url !== PARKED_URL && t.url !== 'about:blank')) continue;
    let c = null;
    try {
      c = await CDP({ port, target: t.id });
      const ev = async (e) => (await c.Runtime.evaluate({ expression: e, returnByValue: true })).result.value;
      // A tab that is not its window's selected tab is hidden, and a hidden
      // tab is throttled into uselessness (above). Skip it rather than
      // activate it, which is the very thing being avoided.
      if (await ev('document.visibilityState') !== 'visible') continue;
      // Claim it with a mark unique to this run, then check the mark stuck,
      // so two tools adopting the same Chrome cannot both take one tab.
      const mark = `about:blank#harness-busy-${process.pid}-${Date.now()}`;
      await c.Page.navigate({ url: mark });
      if (await ev('location.href') !== mark) continue;
      if (url !== 'about:blank') await c.Page.navigate({ url });
      console.log(`[setup] reusing idle tab ${t.id.slice(0, 8)} (no window activation)`);
      return t.id;
    } catch { /* try the next one */ }
    finally { if (c) await c.close().catch(() => {}); }
  }
  console.log('[setup] no visible idle tab to reuse -- opening one (activates the Chrome window this once)');
  const res = await fetch(`http://localhost:${port}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
  const target = await res.json();
  return target.id;
}

// Navigates a finished run's tab to PARKED_URL rather than closing it: the
// page (and its GPU buffers) is torn down all the same, but the tab survives
// for openTab to reuse quietly next time. Closing it would force the next
// run back onto /json/new and its window activation; and if it was the
// window's last tab, closing it quits the whole Chrome.
async function parkTab(port, id) {
  const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
  try {
    const c = await CDP({ port, target: id });
    await c.Page.navigate({ url: PARKED_URL });
    // Wait for it to land: mid-navigation /json/list reports the url as "",
    // and a run starting straight after this one would pass the tab over.
    await waitFor(async () => {
      try { return (await c.Runtime.evaluate({ expression: 'location.href', returnByValue: true })).result.value === PARKED_URL; }
      catch { return false; }
    }, 3000, 50);
    await c.close();
  } catch { await closeTab(port, id); }
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

  // Chrome asks every origin for /favicon.ico on its own, and none of these
  // static pages ship one. That 404 is the BROWSER's request, not the page's,
  // so reporting it fails a healthy page -- and it surfaces unpredictably,
  // because Chrome asks once per origin per session, so whichever tool
  // navigates first in a session wears it. It arrives on BOTH channels: a
  // Network.responseReceived and the console's own "Failed to load resource".
  // Nothing else is filtered -- a 404 for a shader or a module is exactly
  // what this watch is for.
  const browserOwned = (url) => /\/favicon\.ico(\?|$)/.test(String(url || ''));

  if (Log) {
    await Log.enable();
    Log.entryAdded((e) => {
      if (e.entry.level !== 'error') return;
      if (browserOwned(e.entry.url)) return;
      push('log', e.entry.text);
    });
  }
  if (Network) {
    await Network.enable();
    await Network.setCacheDisabled({ cacheDisabled: true });
    Network.loadingFailed((e) => {
      if (e.canceled) return;
      push('net', `${e.type} failed: ${e.errorText}`);
    });
    Network.responseReceived((e) => {
      if (e.response.status >= 400 && !browserOwned(e.response.url)) {
        push('net', `HTTP ${e.response.status} for ${e.response.url}`);
      }
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
// `allowStatus` is a regexp for an `error:` line the caller EXPECTS -- a
// config whose whole point is to trip a guard, which otherwise cannot be
// distinguished from a config that broke. It suppresses only the status
// line; uncaught exceptions and console errors still fail, because nothing
// legitimately expects those.
async function assertPageHealthy(Runtime, watch, what, allowStatus) {
  const st = await statusError(Runtime);
  const seen = watch ? watch.drain() : [];
  const lines = [];
  if (st && !(allowStatus && allowStatus.test(st))) lines.push(`  #status: ${st}`);
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
//
// An adopted Chrome's tab is PARKED, not closed -- see parkTab/openTab.
async function teardown({ port, tabId, chrome, server, keepOpen }) {
  if (keepOpen) return;
  if (!chrome.started) { await parkTab(port, tabId); }
  else {
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
  openTab, parkTab, PARKED_URL, firstTab, closeTab, navigateTo, evalExpr, waitForGlobal, teardown,
  attachPageWatch, statusError, assertPageHealthy,
  PROFILE_ROOT, liveProfileDirs, sweepStaleProfiles, reapAllChromes,
};
