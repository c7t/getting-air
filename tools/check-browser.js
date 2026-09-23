#!/usr/bin/env node
// Is the debug Chrome fit to produce numbers? Checks every page tab: adapter
// (a SwiftShader fallback is a FAIL), #status `error:`, the fatal overlay,
// stuck inits, and more than one tab live-stepping at once. See
// tools/lib/browser-health.js for why each check exists. Exits 1 on any FAIL.
//
//   node tools/check-browser.js --port=9471
//   node tools/check-browser.js --port=9471 --chromeLog=/path/to/chrome.log
//
// The webgpu-verify skill runs this at launch, before trusting any number or
// screenshot, and at the end; browser-lifecycle.js's teardown runs the same
// check at the end of every Node tool.

const { checkBrowserHealth, formatHealth } = require('./lib/browser-health');

const o = { port: 9333, chromeLog: null };
for (const a of process.argv.slice(2)) {
  if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
  else if (a.startsWith('--chromeLog=')) o.chromeLog = a.slice(12);
  else { console.error(`unknown argument ${a}`); process.exit(2); }
}

checkBrowserHealth(o).then(h => {
  console.log(formatHealth(h));
  process.exit(h.ok ? 0 : 1);
}).catch(e => { console.error('check-browser failed:', e.message); process.exit(1); });
