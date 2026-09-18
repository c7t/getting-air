---
name: webgpu-verify
description: Launch this WebGPU app (index.html + main.js/vpm.js) in a real GPU-capable Chrome, drive it via CDP, and capture screenshots AND error state (status line, fatal overlay, uncaught exceptions, NaN fields) to verify changes actually render. Use when asked to run, verify, screenshot or debug this project, to confirm a WebGPU/shader change works, or to find out why a page refused to boot.
---

# Running and screenshotting this app

This is a static WebGPU page (`index.html`, `main.js`, `vpm.js`, `shaders/*.wgsl`) —
no build step, no dev server framework. WebGPU needs a real GPU-capable browser, so
headless/sandboxed Chromium (e.g. `chromium-cli`) is unlikely to have WebGPU support.
Instead, launch a dedicated real Chrome with WebGPU flags and drive it over the
Chrome DevTools Protocol (CDP) via the globally-installed `chrome-remote-interface`
node module.

## 1. Serve the page

WebGPU requires a secure context. `localhost` already counts, but this repo has an
HTTPS dev server set up (`https.py` + `localhost.pem`, self-signed) — use it:

```bash
cd /home/ctalbott/p/getting-air
nohup python3 https.py > /tmp/vpm-https.log 2>&1 &
echo $! > /tmp/vpm-https.pid
# poll until it serves
timeout 10 bash -c 'until curl -sk https://localhost:4444/index.html -o /dev/null; do sleep 0.3; done'
```

Stop it later with `kill $(cat /tmp/vpm-https.pid)`.

## 2. Launch a dedicated WebGPU-capable Chrome

Don't reuse the user's normal Chrome profile/window. Launch a separate instance with
its own profile dir and a remote-debugging port, with WebGPU explicitly enabled:

```bash
mkdir -p /tmp/vpm-chrome-profile
profile_dir=$(mktemp -d -p /tmp/vpm-chrome-profile)
DISPLAY=:0 nohup /opt/google/chrome/chrome \
  --remote-debugging-port=9333 \
  --enable-features=Vulkan,WebGPUService \
  --enable-unsafe-webgpu \
  --ignore-certificate-errors \
  --no-first-run --no-default-browser-check \
  --user-data-dir=$profile_dir \
  --window-size=1400,900 \
  "https://localhost:4444/index.html" > /tmp/vpm-chrome.log 2>&1 &
echo $! > /tmp/vpm-chrome.pid
sleep 3
curl -s http://localhost:9333/json/version   # confirms the debug port is up
```

`DISPLAY=:0` is required — this launches headed (not `--headless`), since headless
Chrome's WebGPU/GPU support is unreliable here and the real GPU process gives a much
more trustworthy signal. The window will briefly appear on the real display; that's
expected, matching how `take_screenshot.sh` (X11 `import`-based) already assumes a
real visible window elsewhere in this repo. This CDP approach is preferred over that
script because it doesn't depend on an already-running Chrome with a debug port, and
`Page.captureScreenshot` avoids X11 window lookup entirely.

## 3. Drive it and capture screenshots via CDP

`chrome-remote-interface` is installed globally, not as a local repo dependency —
require it by absolute path:

```js
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const fs = require('fs');

(async () => {
  const client = await CDP({ port: 9333 }); // attaches to the first/only page target
  const { Page, Runtime, Network } = client;
  await Page.enable();
  await Runtime.enable();
  await Network.enable();
  // NOT optional -- https.py sends no cache headers, so an edited file can
  // silently not reach the page. See section 4.
  await Network.setCacheDisabled({ cacheDisabled: true });

  // Necessary, not sufficient: this sees ONE of the five failure shapes in
  // section 4. Read #status too.
  Runtime.exceptionThrown(e => console.log('[exception]', e.exceptionDetails.text));

  await new Promise(r => setTimeout(r, 3000)); // let WebGPU init + a few frames run

  const r = await Runtime.evaluate({
    expression: `document.getElementById('status').textContent`,
    returnByValue: true,
  });
  console.log('status:', r.result.value);

  const { data } = await Page.captureScreenshot({ format: 'png' });
  fs.writeFileSync('/tmp/screenshot.png', Buffer.from(data, 'base64'));

  await client.close();
})();
```

Run with plain `node script.js` (no npm install needed).

Then use the Read tool on `/tmp/screenshot.png` to actually look at it — a blank/black
canvas is a failure to render, not success.

**The screenshot is the last thing to trust, not the first.** Section 4 is the
part that says what actually went wrong: a black canvas, a legible error box and
a perfectly normal-looking picture over a NaN field are three different
failures, and only one of them is distinguishable by eye.

## 4. Read the ERROR STATE over CDP — the picture is the weakest signal

Once Chrome is up on the debug port, **attach and ask the page what happened**
rather than inferring it from a screenshot. A failure on these pages has at
least five shapes and only one of them reaches `Runtime.exceptionThrown`:

| shape | where it surfaces | does `exceptionThrown` see it? |
|---|---|---|
| module-scope config refusal (`?levels=1`) | `#status`, `#fatal-overlay`, **and** an uncaught module error; `window.__AMR` is never defined | yes |
| anything inside `init()` — adapter, limits, `createBindGroupLayout` | `#status` as `error: ...` + `#fatal-overlay`, via `init().catch(handleErr)` | **no** |
| a per-frame throw | same, via `frame().catch(handleErr)` | **no** |
| a WebGPU validation error | same, via `device.popErrorScope()` | **no** |
| the field goes NaN | **nothing at all** — the page runs, the canvas draws | **no** |

So `Runtime.exceptionThrown` alone misses four of five. Poll the DOM:

```js
const ERROR_STATE = `(() => JSON.stringify({
  status:  document.getElementById('status')?.textContent ?? null,
  fatal:   document.getElementById('fatal-overlay')?.textContent ?? null,
  ready:   !!window.__AMR || !!window.__LBM,
}))()`;
```

`#status` is the one to gate on — `error-overlay.mjs` deliberately writes the
same `error: ...` text there for every path, *because* the DOM overlay is
invisible to a poller, and `tools/validate-all.js`'s boot smoke matches
`/^error:/i` on exactly that string. The overlay is additive, for humans.

**Symptom to recognise: a 60-second `waitForGlobal` timeout is usually a
refusal, not a hang.** `window.__AMR` is assigned at the end of `init()`, so any
of the first four rows above leaves it undefined forever. When a wait times out,
read `#status` before assuming the GPU is wedged — the answer is normally sitting
there in one line.

**And a green run is not a finite field.** `?benchSkip=interp,avg` on
`index-amr.html` drives 589824 of 589824 dense populations non-finite; a
word-equality comparison then reports *zero* differing words and `maxAbs
0.0e+0`, its most emphatic possible pass, because identical NaN bit patterns are
identical words. On screen it is bright single cells scattered over the domain —
which is the only place it shows, and only if someone is looking. Score field
health FIRST, from `debugSnapshotSave().velB64` (f32 under every packing, unlike
`fB64`, which holds packed halves under `?f16=`):

```js
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
```

A row whose field is not finite is a FAILURE, never a pass. See
`tools/validate-root-step.js`, which gates on this.

**Measured 2026-09-17, all three snippets above, on `index-amr.html`:**

```
?levels=1                      status "error: ?levels=1 invalid -- must be >= 2 ..."
                               fatal  same + stack      ready false   exceptionThrown: Uncaught
?levels=2                      status "[AMR-dev] step 760  y=7.9 ... vy=0.0183"
                               fatal  null              ready true    exceptionThrown: nothing
                               health nonFinite 0/131072   max|u| 0.0182
?levels=2&benchSkip=interp,avg status "[AMR-dev] step 752  y=-15.9 ... vy=0.3000"
                               fatal  null              ready true    exceptionThrown: nothing
                               health nonFinite 131072/131072   max|u| 0
```

**Read the third row's status line again.** It is a perfectly ordinary telemetry
line over a field that is 100% NaN. It is not even obviously wrong: `vy=0.3000`
is `v_max` exactly, because `amr_physics.wgsl` clamps the integrated velocity,
and `amr_force1.wgsl`'s `safeFixed` maps NaN to 0 — the rigid body's five
numbers are laundered into something finite and plausible no matter what the
fluid does. CLAUDE.md records the same thing about the invariant sweep's `field`
column. **Nothing on the page tells you.** The health check is the whole signal.

### Disable the HTTP cache on every CDP session

```js
await Network.enable();
await Network.setCacheDisabled({ cacheDisabled: true });
```

**`https.py` sends no cache headers**, so Chrome applies heuristic caching to
`main-amr.js`, `amr2d.mjs` and every `shaders/*.wgsl`. A `Page.navigate` that
only changes the query string then re-runs the **previous build** — the file on
disk is edited, `curl` confirms the server is serving it, and the page is not
using it.

Measured 2026-09-17: a one-line pipeline-constant fix (pinning `DIRECT_GHOST` on
the root-pool step) read as completely inert across a whole measurement — the
before and after numbers were identical to the word — and the fix was about to
be written up as "not the cause". With the cache disabled the same build went
from 580623 differing words to 0.

**If a change appears to do nothing, check this before believing it.** A stale
build and a genuinely inert change are indistinguishable from the numbers alone,
and the cache is the far more common explanation.

### Two smaller ones

- **`--ignore-certificate-errors` is why this Chrome can load the page at all.**
  `https.py`'s cert is self-signed, so pointing the user's normal browser at
  `https://localhost:<port>` gets an interstitial, and CDP against that tab fails
  with `Cannot attach to this target` / `Frame with ID 0 is showing error page`.
  That is the certificate, not the app.
- **`debugStepSync(n)` advances in whole frames of `STEPS_PER_FRAME = 64`** — its
  loop is `for (k = 0; k < n; k += STEPS_PER_FRAME)`, so `debugStepSync(1)` and
  `debugStepSync(2)` return the *identical* field. If two step counts give
  bit-identical results, check this before concluding the solver is insensitive
  to them.

## Gotchas

- **`Runtime.evaluate` calls share one global JS scope.** Declaring `const x = ...` or
  `let x = ...` in one `evaluate` call and again in a later one throws
  `SyntaxError: Identifier 'x' has already been declared`, silently aborting that
  call (check `result.exceptionDetails`, don't assume success). Wrap each snippet in
  an IIFE — `(function(){ var x = ...; ... })()` — or always use `var`.
- **Don't repeatedly `Page.navigate` to the same URL on a long-lived tab**, especially
  while heavy WebGPU compute is running. It's flaky here — sometimes a genuine reload
  happens unexpectedly (visible as sim step-counters resetting), and it isn't obviously
  caused by anything in the driving script. Prefer: launch Chrome once already pointed
  at the target URL (as above), attach with `CDP({ port })` without navigating again,
  and drive everything through `Runtime.evaluate` / `Page.captureScreenshot` from
  there. If you do need a fresh load, close the tab and open a genuinely new one
  (`curl -X PUT http://localhost:9333/json/new?<url>`) rather than re-navigating.
  **Navigating between DIFFERENT urls is fine and is what the Node tools do** --
  `validate-all.js`, `validate-root-mirror.js` and `validate-root-step.js` all
  reuse one tab across every config via `Page.navigate`, precisely so no more
  than one WebGPU context is ever alive. The flaky case is re-navigating to the
  SAME url. Whenever you do navigate, disable the cache first (section 4): two
  configs that differ only in the query string will otherwise share a build.
- **This app runs two independent WebGPU pipelines at once** (`main.js` for the LBM
  sim, `vpm.js` for the vortex-particle sim) — check `console --errors`-equivalent
  (the `Runtime.exceptionThrown` listener above) for either one; a validation error in
  one pipeline won't necessarily stop the other from rendering, which can mask a
  regression if you only eyeball the screenshot.
- Readouts worth checking via `Runtime.evaluate` beyond the screenshot itself:
  `#status` (LBM), `#vpm-status` / `#vpm-r-measured` / `#vpm-r-analytic` /
  `#vpm-r-circ` (VPM) — the VPM panel's measured-vs-analytic numbers are a real
  correctness check, not just telemetry.

- **Kill every previous instance before launching a new one, not just the last
  PID.** `$! > /tmp/vpm-chrome.pid` overwrites the pid file each time, so if you
  launch several times across a session without cleaning up first, earlier
  Chrome processes become orphaned and keep running in the background,
  invisibly competing for the GPU with whichever instance you're currently
  testing against. Symptom: numbers that were rock-solid in an earlier run
  (e.g. an analytic-vs-measured match) suddenly show a large, unexplained
  error — looks exactly like a code regression but is really just GPU
  contention slowing/jittering the simulation. Before trusting a surprising
  result, run `make chrome-clean DRY_RUN=1` to see what is still alive, then
  `make chrome-clean` and rerun on a clean GPU before concluding there's a
  real bug. (Match on the profile dir, never on the process name -- `pkill -f
  chrome` would take the user's own browser with it.)

## Cleanup

If you launched Chrome by hand (section 2), the simplest cleanup is:

```bash
make chrome-clean          # add DRY_RUN=1 to see what it would do first
kill $(cat /tmp/vpm-https.pid) 2>/dev/null
```

`make chrome-clean` kills every debug Chrome whose `--user-data-dir` lives under
`/tmp/vpm-chrome-profile` and removes the dirs. It identifies them from `/proc`
by profile dir, never by process name, so the user's own browser can never be
matched -- unlike a bare `pkill -f chrome`, which can and must not be used here.

The equivalent by hand, if you prefer:

```bash
kill $(cat /tmp/vpm-chrome.pid) $(cat /tmp/vpm-https.pid) 2>/dev/null
pkill -f -- "--user-data-dir=/tmp/vpm-chrome-profile" 2>/dev/null   # orphans too
rm -rf /tmp/vpm-chrome-profile
```

### What the Node tools do about this, and what they deliberately don't

`tools/lib/browser-lifecycle.js` (used by validate-all.js and friends) REUSES a
Chrome that is already listening on the debug port -- that is what keeps exactly
one WebGPU context alive across a whole run instead of one per config. The
consequence is that a run which ADOPTS a Chrome does not own it and will not
kill it on teardown: another run may be adopting the same one, and this skill
deliberately leaves one open for manual driving.

So a run killed part-way (timeout, Ctrl-C, crash before teardown) orphans its
Chrome, and every later run then adopts it without owning it. Its stale PROFILE
DIR is swept automatically by the next launch, so disk no longer grows
unbounded -- one session left 712 MB behind before that existed -- but the
PROCESS is only reclaimed by `make chrome-clean`. Run it when you are done, and
before trusting a surprising performance or accuracy number (see the orphan
warning above).
