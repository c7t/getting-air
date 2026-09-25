#!/usr/bin/env node
// bench-d3-raymarch.js -- what a raymarcher change COSTS, and what it CHANGES.
// Priced by REMOVING it, plans/3D.md M6.4f. Default subject: ?volfallback.
//
// The first two attempts at this number measured the DISPLAY. Reading a frame
// rate off the rAF loop gave 59.9 fps on both legs at 1152x720 and again at
// 2560x1440, because the frame costs under 16.6 ms either way; disabling vsync
// separated them but then had the browser's frame pacing in the number too.
// Neither is the shader.
//
// debugBenchRender encodes N draws into one command buffer, submits, and waits
// on the queue -- no presentation, no compositor, no rAF, no readback -- and
// reports the MINIMUM of its batches, which is bench-d3-interface.js's rule for
// this desktop. A same-build repeat (--legs=1,1) gives the noise floor and any
// build-vs-build claim needs one.
//
// It also reports the DIFFERENCE IN THE PICTURE, because a change that is free
// and invisible is not worth having: one offscreen frame each way, differenced.
const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const L = require(path.join(__dirname, 'lib', 'browser-lifecycle.js'));
const REPO = path.resolve(__dirname, '..');

const arg = (k, d) => {
  const a = process.argv.find((s) => s.startsWith('--' + k + '='));
  return a ? a.slice(k.length + 3) : d;
};
const STEPS = parseInt(arg('steps', '3000'));
const SIZE = parseInt(arg('size', '512'));
const LEGS = arg('legs', '1,0').split(',');
const FLAG = arg('flag', 'volfallback');
const BASE_URL = arg('baseUrl', 'https://localhost:4444');
const PORT = parseInt(arg('port', '9333'));

if (process.argv.some((a) => a === '--help' || a === '-h')) {
  console.log(`bench-d3-raymarch.js -- price a raymarcher change by removing it
  --flag=NAME   the URL flag to A/B (default ${FLAG})
  --legs=a,b    its values, first against second (default 1,0).
                --legs=1,1 is the SAME-BUILD REPEAT and is what bounds any
                claim -- this instrument's own floor measured 6.2%.
  --reps=N      draws per command buffer (default 64)
  --batches=N   batches; the MINIMUM is reported (default 25)
  --rw= --rh=   render size (default 1024x1024)
  --steps=N     steps to develop the flow first (default 3000)
  --baseUrl= --port=

Reports, never gates.`);
  process.exit(0);
}
const RW = parseInt(arg('rw', '1024'));
const RH = parseInt(arg('rh', '1024'));
const REPS = parseInt(arg('reps', '48'));
const BATCHES = parseInt(arg('batches', '7'));

const BASE = 'scenario=card&n=32&span=2&re=1100&tilt=1.047&aspect=0.1016&live=0'
  + '&levels=3&rb=4&refine=body&dynamic=1&manageEvery=8&margin=3&slotHeadroom=2.5'
  + '&volMargin=4&volBudget=512&view=volume&vol=1&volIso=1.5&volGain=4&volOpacity=0.12'
  + '&azim=90&elev=0&dist=0.55';

async function ev(Rt, e, what) {
  const r = await Rt.evaluate({ expression: e, awaitPromise: true, returnByValue: true, timeout: 900000 });
  if (r.exceptionDetails) throw new Error(what + ': ' + r.exceptionDetails.text);
  return r.result.value;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function leg(fb) {
  L.reapAllChromes();
  await L.waitFor(async () => !(await L.chromeDebugOk(PORT)), 20000, 250);
  await L.ensureChrome(PORT);
  const tab = await L.firstTab(PORT);
  const c = await CDP({ local: true, port: PORT, target: tab.id });
  await Promise.all([c.Page.enable(), c.Runtime.enable()]);
  await L.navigateTo(c.Page, BASE_URL + '/index-3d.html?' + BASE + '&' + FLAG + '=' + fb);
  await L.waitForGlobal(c.Runtime, 'window.__D3', 120000);
  for (let d = 0; d < STEPS; d += 250) await ev(c.Runtime, 'window.__D3.debugStepSync(250)', 'step');
  // The boxes never followed the body here (live=0), so put them where the
  // body actually is before timing -- otherwise both legs time a stale box and
  // the replicated population is not the one the picture really has.
  await ev(c.Runtime, 'window.__D3.debugRenderFrame({w:64,h:64})', 'warm');

  const bench = await ev(c.Runtime,
    `window.__D3.debugBenchRender({w:${RW},h:${RH},reps:${REPS},batches:${BATCHES}})`, 'bench');
  // THE INSTRUMENT AGAINST THE ONE THAT ALREADY EXISTS. debugRenderFrame is
  // also off the animation thread, so if its readback is small it is the
  // instrument and debugBenchRender should not exist. Timed the same way:
  // repeated calls, minimum taken.
  const rf = [];
  for (let i = 0; i < 12; i++) {
    const t = Date.now();
    await ev(c.Runtime, `window.__D3.debugRenderFrame({w:${RW},h:${RH}})`, 'renderFrame');
    rf.push(Date.now() - t);
  }
  rf.sort((a, b) => a - b);
  const img = await ev(c.Runtime, `window.__D3.debugRenderFrame({w:${SIZE},h:${SIZE}})`, 'render');
  await c.close();
  return { fb, ms: bench.msPerFrame, median: bench.median, spread: bench.spread,
           rfMin: rf[0], rfMed: rf[rf.length >> 1], rgba: img.rgba };
}

(async () => {
  await L.ensureServer(BASE_URL, REPO);
  const out = [];
  for (const v of LEGS) out.push(await leg(parseInt(v)));
  L.reapAllChromes();
  console.log(`\n${RW}x${RH}, ${REPS} draws per batch, min of ${BATCHES} batches`);
  console.log('leg              ms/frame  median   spread   debugRenderFrame min/med');
  for (const r of out) {
    console.log(String(FLAG + '=' + r.fb).padEnd(17)
      + r.ms.toFixed(4).padEnd(10) + r.median.toFixed(4).padEnd(9)
      + (100 * r.spread).toFixed(1).padEnd(9) + '%'
      + '  ' + r.rfMin + ' / ' + r.rfMed + ' ms');
  }
  if (out.length === 2) {
    const d = 100 * (out[0].ms / out[1].ms - 1);
    console.log('\nleg0 / leg1 = ' + (out[0].ms / out[1].ms).toFixed(4)
      + 'x  (' + (d >= 0 ? '+' : '') + d.toFixed(1) + '% on the first leg)');
    console.log('Read against a SAME-BUILD repeat (--legs=1,1); a difference'
      + ' inside that spread is nothing.');
    const a = Buffer.from(out[0].rgba, 'base64'), b = Buffer.from(out[1].rgba, 'base64');
    let diff = 0, worst = 0;
    for (let i = 0; i < a.length; i += 4) {
      let d = 0;
      for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a[i + k] - b[i + k]));
      if (d > 2) diff++;
      worst = Math.max(worst, d);
    }
    console.log('picture: ' + diff + ' of ' + (a.length / 4) + ' pixels differ by >2/255'
      + ' (' + (100 * diff / (a.length / 4)).toFixed(2) + '%), worst ' + worst);
  }
  process.exit(0);
})().catch((e) => { console.error(e); L.reapAllChromes(); process.exit(1); });
