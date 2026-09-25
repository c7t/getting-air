#!/usr/bin/env node
// Every CDP connect in tools/ must pass `local: true`. No GPU, no browser --
// run it with `make test` (or `node tools/test-cdp-local.js`).
//
// WHY: without it, chrome-remote-interface fetches the protocol descriptor
// from the browser's `/json/protocol` on every connect, and Chrome for Android
// (153.0.8010.52, the Pixel phone over `adb forward`) crashes its whole browser
// process on that one request -- a SIGTRAP CHECK on the main thread, measured
// 2026-09-24 with a bare `curl`, while `/json/version` and `/json/list` are
// fine. The symptom on the tool side is only `socket hang up`, and the page is
// gone. That is what plans/perf-characterization.md's 2026-09-08 "driving
// debugStepSync killed Chrome outright" was: with `local: true` the same
// debugStepSync runs and Chrome stays up.
//
// `local: true` uses the descriptor bundled with chrome-remote-interface
// instead. Every domain method these tools call is in it; the desktop Chrome
// never needed the remote one.
//
// A lint rather than a shared connect helper because the tools each require
// CDP directly, and a new tool is written by copying an old one -- this is what
// fails when the copied line is from before this fix.

const fs = require('fs');
const path = require('path');

const TOOLS = path.join(__dirname);
const files = [];
(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (/\.(c|m)?js$/.test(e.name) && p !== __filename) files.push(p);
  }
})(TOOLS);

const bad = [];
let connects = 0;
for (const f of files) {
  const lines = fs.readFileSync(f, 'utf8').split('\n');
  lines.forEach((line, i) => {
    // A connect is `CDP(` called with an options object; CDP.List/New/etc.
    // are plain HTTP and never touch /json/protocol.
    if (!/\bCDP\(\s*\{/.test(line)) return;
    connects++;
    if (!/\blocal\s*:\s*true\b/.test(line)) bad.push(`${path.relative(path.join(TOOLS, '..'), f)}:${i + 1}: ${line.trim()}`);
  });
}

// Guard the guard: a regex that stopped matching would pass vacuously.
if (connects < 10) {
  console.error(`FAIL: found only ${connects} CDP connects under tools/ -- the pattern no longer matches how tools connect`);
  process.exit(1);
}
if (bad.length) {
  console.error(`FAIL: ${bad.length} CDP connect(s) without \`local: true\` (crashes Chrome for Android, see header):`);
  for (const b of bad) console.error('  ' + b);
  process.exit(1);
}
console.log(`test-cdp-local: ${connects} CDP connects, all pass local: true`);
