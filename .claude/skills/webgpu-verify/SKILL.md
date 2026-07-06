---
name: webgpu-verify
description: Launch this WebGPU app (index.html + main.js/vpm.js) in a real GPU-capable Chrome, drive it via CDP, and capture screenshots to verify changes actually render. Use when asked to run, verify, or screenshot this project, or to confirm a WebGPU/shader change works.
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
DISPLAY=:0 nohup /opt/google/chrome/chrome \
  --remote-debugging-port=9333 \
  --enable-features=Vulkan,WebGPUService \
  --enable-unsafe-webgpu \
  --ignore-certificate-errors \
  --no-first-run --no-default-browser-check \
  --user-data-dir=/tmp/vpm-chrome-profile \
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
  const { Page, Runtime } = client;
  await Page.enable();
  await Runtime.enable();

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
- **This app runs two independent WebGPU pipelines at once** (`main.js` for the LBM
  sim, `vpm.js` for the vortex-particle sim) — check `console --errors`-equivalent
  (the `Runtime.exceptionThrown` listener above) for either one; a validation error in
  one pipeline won't necessarily stop the other from rendering, which can mask a
  regression if you only eyeball the screenshot.
- Readouts worth checking via `Runtime.evaluate` beyond the screenshot itself:
  `#status` (LBM), `#vpm-status` / `#vpm-r-measured` / `#vpm-r-analytic` /
  `#vpm-r-circ` (VPM) — the VPM panel's measured-vs-analytic numbers are a real
  correctness check, not just telemetry.

## Cleanup

```bash
kill $(cat /tmp/vpm-chrome.pid) $(cat /tmp/vpm-https.pid) 2>/dev/null
rm -rf /tmp/vpm-chrome-profile
```
