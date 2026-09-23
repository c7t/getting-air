// tools/lib/browser-health.js -- is the debug Chrome in a state whose numbers
// can be believed? Checks EVERY page tab, not just the one a tool drove.
//
// WHY THIS EXISTS (2026-09-23). The debug Chrome's GPU process crashed at
// 07:33:59 mid-run, every later `vkCreateInstance` failed, and WebGPU fell back
// SILENTLY to SwiftShader -- Google's CPU rasterizer, `isFallbackAdapter`
// true, 10 storage buffers per stage. Every page that fit in 10 bindings kept
// booting and kept producing plausible numbers; a full afternoon of conservation
// and bisect runs was taken on it before anyone looked. Nothing in any tool
// asked which adapter it was on. The first sign was an old build refusing to
// boot ("needs 16 storage buffers per shader stage, this GPU's max is 10") in a
// tab nobody was reading -- and the user, not a tool, saw it.
//
// The same session had also accumulated TEN tabs (every `--keepOpen` run leaves
// its tab behind), two of them live-stepping their own simulations, which is
// the GPU-contention failure the webgpu-verify skill already warns about.
//
// So "is this result valid" is a property of the WHOLE browser, and every
// check below is a way a result was, or can be, silently invalidated:
//
//   adapter      FAIL  a fallback/software adapter, or none at all
//   status       FAIL  `#status` reads `error: ...` (error-overlay.mjs writes
//                      every init/frame/validation failure there)
//   fatal        FAIL  `#fatal-overlay` present and showing text
//   contention   FAIL  more than one app tab LIVE-stepping at once
//   stuck        WARN  `#status` still at its initial text with no debug
//                      global -- init never finished (a paused, tool-driven
//                      tab also sits at "initializing...", but HAS a global)
//   tabs         WARN  more than one app tab open at all
//   chromeLog    FAIL  (optional) the launch log shows a GPU-process crash or
//                      a Vulkan instance failure -- the CAUSE of a fallback,
//                      which can predate every tab now open
//
// It deliberately does NOT score the field (NaN): that needs a page-specific
// readback, and the webgpu-verify skill's HEALTH snippet / each tool's own
// field check own it.

const fs = require('fs');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');

// Evaluated INSIDE each tab. Returns a plain JSON-able object.
const PROBE = `(async () => {
  const el = (id) => document.getElementById(id);
  const status = el('status') ? el('status').textContent.trim() : null;
  const fo = el('fatal-overlay');
  const fatalShown = !!fo && fo.textContent.trim() !== '' && getComputedStyle(fo).display !== 'none'
    && getComputedStyle(fo).visibility !== 'hidden';
  const globals = ['__CYL', '__AMR', '__LBM', '__D3'].filter(g => typeof window[g] !== 'undefined');
  const G = globals.length ? window[globals[0]] : null;
  let live = null;
  try { if (G && typeof G.isLive === 'function') live = !!G.isLive(); } catch (e) { live = null; }
  let adapter = null;
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) {
        const i = a.info || {};
        adapter = { vendor: i.vendor || '', architecture: i.architecture || '', description: i.description || '',
                    isFallback: !!(a.isFallbackAdapter ?? i.isFallbackAdapter),
                    maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage };
      } else adapter = { none: true };
    } catch (e) { adapter = { error: String(e) }; }
  }
  return { status, fatalShown, fatalText: fatalShown ? fo.textContent.trim().slice(0, 300) : null,
           globals, live, adapter, hasGpu: !!navigator.gpu };
})()`;

const isAppTab = (t) => t.type === 'page' && /^https?:/.test(t.url);
const INITIAL_STATUS = /^(initializing|loading)/i;

// Lines in Chrome's own stderr that mean the GPU process died or Vulkan could
// not start -- the precondition for a silent SwiftShader fallback.
const CHROME_LOG_BAD = [
  /vkCreateInstance failed/i,
  /GPU process (crashed|exited|isn't usable)/i,
  /Exiting GPU process/i,
  /Lost UI shared context/i,
  /SwiftShader/i,
];

async function probeTab(port, t, timeoutMs) {
  const c = await CDP({ port, target: t });
  try {
    const r = await c.Runtime.evaluate({ expression: PROBE, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
    if (r.exceptionDetails) return { probeError: r.exceptionDetails.text };
    let v = r.result.value;
    // A tab with no isLive(): infer liveness from whether #status moves.
    if (v.live === null && v.globals.length && v.status != null) {
      await new Promise(res => setTimeout(res, 500));
      const r2 = await c.Runtime.evaluate({ returnByValue: true,
        expression: `(document.getElementById('status')||{}).textContent` });
      v.live = (r2.result.value || '').trim() !== v.status;
      v.liveInferred = true;
    }
    return v;
  } finally { await c.close(); }
}

// -> { ok, failures: [str], warnings: [str], tabs: [{url, ...}], lastAdapterLine }
async function checkBrowserHealth({ port, chromeLog = null, timeoutMs = 15000 } = {}) {
  const failures = [], warnings = [], tabs = [];
  let list;
  try { list = await CDP.List({ port }); }
  catch (e) { return { ok: false, failures: [`no debug Chrome on port ${port}: ${e.message}`], warnings, tabs }; }

  for (const t of list.filter(isAppTab)) {
    let v;
    try { v = await probeTab(port, t, timeoutMs); }
    catch (e) { v = { probeError: e.message }; }
    const row = { url: t.url, ...v };
    tabs.push(row);
    const where = t.url.replace(/^https?:\/\/[^/]+\//, '');
    if (v.probeError) { failures.push(`[${where}] could not probe tab: ${v.probeError}`); continue; }
    const isApp = v.status !== null || v.globals.length > 0;
    if (!isApp) continue;
    if (!v.hasGpu) failures.push(`[${where}] navigator.gpu is absent -- WebGPU is not available in this tab`);
    else if (!v.adapter || v.adapter.none) failures.push(`[${where}] requestAdapter() returned null`);
    else if (v.adapter.error) failures.push(`[${where}] requestAdapter() threw: ${v.adapter.error}`);
    else if (v.adapter.isFallback || /swiftshader/i.test(v.adapter.architecture + v.adapter.vendor)) {
      failures.push(`[${where}] FALLBACK ADAPTER (${v.adapter.vendor}/${v.adapter.architecture}, `
        + `maxStorageBuffersPerShaderStage ${v.adapter.maxStorageBuffersPerShaderStage}) -- this is the CPU `
        + `rasterizer, not the GPU; usually a GPU-process crash. Relaunch Chrome.`);
    }
    if (v.status && /^error:/i.test(v.status)) failures.push(`[${where}] #status: ${v.status.slice(0, 200)}`);
    if (v.fatalShown) failures.push(`[${where}] fatal overlay: ${v.fatalText.split('\n')[0]}`);
    if (v.status != null && INITIAL_STATUS.test(v.status) && !v.globals.length) {
      warnings.push(`[${where}] still "${v.status}" with no debug global -- init never finished`);
    }
  }

  const apps = tabs.filter(t => !t.probeError && (t.status !== null || (t.globals && t.globals.length)));
  const live = apps.filter(t => t.live);
  if (live.length > 1) {
    failures.push(`${live.length} app tabs are LIVE-stepping at once (GPU contention invalidates timing and `
      + `can reorder work): ${live.map(t => t.url.replace(/^https?:\/\/[^/]+\//, '')).join(' | ')}`);
  }
  if (apps.length > 1) warnings.push(`${apps.length} app tabs open -- close the ones you are not measuring`);

  let lastAdapterLine = null;
  if (chromeLog && fs.existsSync(chromeLog)) {
    const lines = fs.readFileSync(chromeLog, 'utf8').split('\n');
    for (const l of lines) if (/Selected adapter:/.test(l)) lastAdapterLine = l.replace(/^.*Selected adapter:\s*/, '');
    const bad = lines.filter(l => CHROME_LOG_BAD.some(re => re.test(l)));
    if (bad.length) failures.push(`chrome log ${chromeLog}: ${bad.length} GPU-failure line(s), first: ${bad[0].trim().slice(0, 160)}`);
  }
  return { ok: failures.length === 0, failures, warnings, tabs, lastAdapterLine };
}

function formatHealth(h) {
  const out = [];
  for (const t of h.tabs) {
    if (t.probeError) { out.push(`  ?? ${t.url}  probe error: ${t.probeError}`); continue; }
    const a = t.adapter && !t.adapter.none && !t.adapter.error
      ? `${t.adapter.vendor}/${t.adapter.architecture}${t.adapter.isFallback ? ' FALLBACK' : ''} sb=${t.adapter.maxStorageBuffersPerShaderStage}`
      : 'no adapter';
    out.push(`  ${t.live ? 'LIVE ' : '     '}${(t.status || '(no #status)').slice(0, 60).padEnd(62)} ${a}\n         ${t.url}`);
  }
  if (h.lastAdapterLine) out.push(`  chrome log, last selected adapter: ${h.lastAdapterLine}`);
  for (const w of h.warnings) out.push(`  WARN  ${w}`);
  for (const f of h.failures) out.push(`  FAIL  ${f}`);
  out.push(h.ok ? '  browser health: OK' : '  browser health: FAILED -- results from this browser are not trustworthy');
  return out.join('\n');
}

module.exports = { checkBrowserHealth, formatHealth };
