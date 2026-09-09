// Fatal-error reporting shared by every page's main*.js.
//
// Origin: cwedgwood's PR #11 against c7t/getting-air ("main-amr: surface
// init/render errors legibly"), which proposed this for main-amr.js alone.
// Taken as a shared module instead, for the reason CLAUDE.md's boot-smoke note
// already records: ten near-identical copies of the same few lines is exactly
// the shape that produced the 238e48c failure, where one copy was updated and
// the other nine were not. All ten main*.js had these lines byte-for-byte
// identical, so there was nothing to reconcile.
//
// WHY AN OVERLAY. The #status line is 12px and lives in a corner; a WebGPU
// failure at init would set it and be missed, leaving a black canvas that reads
// as "still loading". The canvas going black is a FAILURE, not success -- the
// same thing the webgpu-verify skill exists to catch -- so a failure should be
// impossible to miss on the page itself.
//
// The overlay is strictly ADDITIVE: every function here still writes the same
// `error: ...` text into #status, because tools/validate-all.js's boot smoke
// polls exactly that (it fails on /^error:/i and on the text never advancing),
// and it cannot see the DOM overlay.

import { revealChrome } from './ui-chrome.mjs';

// Prominent, readable fatal-error box. Idempotent -- repeated calls reuse and
// overwrite the one element rather than stacking boxes for a failure that
// cascades into several handlers.
export function showFatal(msg) {
  let el = document.getElementById('fatal-overlay');
  if (!el) {
    el = document.createElement('div');
    el.id = 'fatal-overlay';
    el.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
      'max-width:80%;max-height:72%;overflow:auto;background:rgba(50,0,0,0.93);color:#fdd;' +
      'border:1px solid #a55;border-radius:5px;padding:14px 18px;font:12px/1.55 monospace;' +
      'white-space:pre-wrap;z-index:20;';
    // Every page has #canvas-container; body is the fallback for a page that
    // fails before its layout exists.
    (document.getElementById('canvas-container') || document.body).appendChild(el);
  }
  el.textContent = msg;
  el.style.display = 'block';
  // A page whose chrome is collapsed (ui-chrome.mjs) would otherwise report
  // the failure into a hidden #status line. The box above is already visible
  // -- it is appended to #canvas-container, which is never part of the
  // chrome -- so this is about the rest of the diagnosis: the status line,
  // the perf readouts, and the controls that let the user try something else.
  // No-op on a page that never collapses its chrome.
  revealChrome();
}

// The body of every page's handleErr(). `String(e)` rather than `e.message`
// because a thrown non-Error (a bare string, a DOMException without .message)
// otherwise renders as "error: undefined", which says nothing.
export function reportFatal(statusEl, e) {
  const msg = (e && e.message) ? e.message : String(e);
  if (statusEl) {
    statusEl.textContent = `error: ${msg}`;
    statusEl.style.color = '#f77';
  }
  console.error('[getting-air] error:', e, (e && e.stack) ? '\n' + e.stack : '');
  showFatal('Error: ' + msg + ((e && e.stack) ? '\n\n' + e.stack : ''));
}

// navigator.gpu missing. The secure-context case is the one worth diagnosing by
// hand: this project is routinely opened from a phone against the dev server's
// LAN address, and http://<LAN-IP>/ fails here with no explanation at all --
// navigator.gpu is simply undefined. Chrome refuses WebGPU on non-localhost
// http even with --unsafely-treat-insecure-origin-as-secure, so the fix is
// https (https.py serves a self-signed cert; accept the warning) or
// http://localhost. That is why https.py exists and why it is not optional.
export function reportNoWebGPU(statusEl) {
  const secure = window.isSecureContext;
  const reason = !secure
    ? `This origin (${location.origin}) is NOT a secure context. navigator.gpu requires https, ` +
      `or http on localhost / 127.0.0.1 / [::1]. A LAN IP over http will NOT work, even with ` +
      `Chrome's --unsafely-treat-insecure-origin-as-secure flag. Fix: serve over https ` +
      `(python3 https.py -- a self-signed cert is fine, accept the browser warning), or open ` +
      `it as http://localhost.`
    : `navigator.gpu is undefined even though this IS a secure context. The browser may not ` +
      `support WebGPU, it may be disabled (chrome://flags/#enable-unsafe-webgpu), or on Linux ` +
      `the GPU/Vulkan stack may be unavailable (see chrome://gpu).`;
  // `error:` prefix so the boot smoke actually catches this. It previously read
  // "WebGPU not available", which does not match /^error:/i, so a page that
  // could not get WebGPU at all still counted as "status advanced" and PASSED.
  if (statusEl) {
    statusEl.textContent = 'error: WebGPU not available';
    statusEl.style.color = '#f77';
  }
  console.error('[getting-air] WebGPU not available:', reason,
    { isSecureContext: secure, origin: location.origin });
  showFatal('WebGPU not available.\n\n' + reason);
}

// requestAdapter() resolved null -- WebGPU exists but no usable adapter.
export function reportNoAdapter(statusEl) {
  const reason = 'navigator.gpu.requestAdapter() returned null -- no compatible GPU adapter. ' +
    'On Linux, Chrome may need a working Vulkan driver, or --enable-unsafe-webgpu; ' +
    'check chrome://gpu for WebGPU status.';
  // `error:` prefix for the same boot-smoke reason as above.
  if (statusEl) {
    statusEl.textContent = 'error: no GPU adapter';
    statusEl.style.color = '#f77';
  }
  console.error('[getting-air]', reason);
  showFatal('No GPU adapter.\n\n' + reason);
}
