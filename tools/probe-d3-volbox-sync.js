#!/usr/bin/env node
// probe-d3-volbox-sync.js -- M6.4e's gap, measured on the LIVE rAF path,
// which is the only place it exists.
//
// A refined box's placement lives in two buffers: the resample's `originBuf`
// (what gets written into the texture) and the ray params' `boxLo` (where the
// raymarcher reads it out of world space). The pre-M6.4e refresh wrote the
// first for level 1 and then AWAITED level 2's blockSlot readback before
// publishing the second, so every frame drawn in that window showed L1's
// contents at L1's old position.
//
// debugStepSync CANNOT REPRODUCE IT, for `serializedOn`'s reason: the gap is
// between an unawaited refresh and the frame loop, and a tool that drives the
// page synchronously has no frame loop. So this watches a live page and reads
// the counter the page keeps itself.
//   node tools/probe-d3-volbox-sync.js
//   node tools/probe-d3-volbox-sync.js --seconds=60 --legs=0
'use strict';

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const L = require(path.join(__dirname, 'lib', 'browser-lifecycle.js'));
const REPO = path.resolve(__dirname, '..');

const arg = (k, d) => {
  const a = process.argv.find((s) => s.startsWith('--' + k + '='));
  return a ? a.slice(k.length + 3) : d;
};
const SECONDS = parseFloat(arg('seconds', '150'));
const LEGS = arg('legs', '1,0').split(',');
const BASE_URL = arg('baseUrl', 'https://localhost:4444');
const PORT = parseInt(arg('port', '9333'));

if (process.argv.some((a) => a === '--help' || a === '-h')) {
  console.log(`probe-d3-volbox-sync.js -- do the refined volume boxes move ATOMICALLY?
  --seconds=N   wall clock per leg (default ${SECONDS}); the event is rare, so
                this wants a minute or more, not a few frames
  --legs=a,b    1 = today's atomic apply, 0 = the pre-M6.4e stagger (default 1,0)
  --baseUrl=    default ${BASE_URL}
  --port=       Chrome debug port (default ${PORT})

Reports, and the only interesting number is partialFrames, which must be 0.`);
  process.exit(0);
}

const BASE = 'scenario=card&n=32&span=2&re=1100&tilt=1.047&aspect=0.1016'
  + '&levels=3&rb=4&refine=body&dynamic=1&manageEvery=8&margin=3&slotHeadroom=2.5'
  + '&volMargin=4&volBudget=512&view=volume&vol=1&volIso=1.5&volGain=4&volOpacity=0.12'
  + '&azim=90&elev=0&dist=0.55';

async function ev(Rt, e, what) {
  const r = await Rt.evaluate({ expression: e, awaitPromise: true, returnByValue: true, timeout: 900000 });
  if (r.exceptionDetails) throw new Error(what + ': ' + r.exceptionDetails.text);
  return r.result.value;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function leg(volsync) {
  L.reapAllChromes();
  await L.waitFor(async () => !(await L.chromeDebugOk(PORT)), 20000, 250);
  await L.ensureChrome(PORT);
  const tab = await L.firstTab(PORT);
  const c = await CDP({ port: PORT, target: tab.id });
  await Promise.all([c.Page.enable(), c.Runtime.enable()]);
  await c.Emulation.setDeviceMetricsOverride({ width: 1152, height: 720, deviceScaleFactor: 1, mobile: false });
  await L.navigateTo(c.Page, BASE_URL + '/index-3d.html?' + BASE + '&volsync=' + volsync);
  await L.waitForGlobal(c.Runtime, 'window.__D3', 120000);
  const t0 = Date.now();
  while (Date.now() - t0 < SECONDS * 1000) await sleep(2000);
  const st = await ev(c.Runtime, 'window.__D3.debugVolBoxSync()', 'debugVolBoxSync');
  const step = await ev(c.Runtime, 'window.__D3.getStep()', 'getStep');
  console.log('volsync=' + volsync + '  ' + JSON.stringify({ ...st, step }));
  await c.close();
  return { volsync, ...st, step };
}

(async () => {
  await L.ensureServer(BASE_URL, REPO);
  const out = [];
  for (const v of LEGS) out.push(await leg(parseInt(v)));
  L.reapAllChromes();
  console.log('\nleg        updates  partialFrames  maxGapMs  frames  step');
  for (const r of out) {
    console.log(String('volsync=' + r.volsync).padEnd(11)
      + String(r.updates).padEnd(9) + String(r.partialFrames).padEnd(15)
      + String(r.maxGapMs.toFixed(1)).padEnd(10) + String(r.frames).padEnd(8) + r.step);
  }
  console.log('\npartialFrames is the number of rendered frames that showed a refined box'
    + '\nat a position the raymarcher had not been told about yet. It must be 0.');
  process.exit(0);
})().catch((e) => { console.error(e); L.reapAllChromes(); process.exit(1); });
