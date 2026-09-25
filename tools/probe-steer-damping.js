#!/usr/bin/env node
// HOW STIFF IS THE BODY/FLUID COUPLING AT THE STEP FREQUENCY?
//
// index-steer.html blows up at large centre-of-mass offsets through a mode
// that flips sign EVERY STEP (plans: the reach-1 runs; the fluid torque went
// +2.1 / -3.4 / +2.1 ...). The hypothesis: the torque the body pays each step
// is, at that frequency, an explicit damper, tau = -D omega, so the update is
//
//     omega_{n+1} = (1 - D/I) omega_n + (everything else)/I
//
// which alternates once D/I > 1 and diverges once D/I > 2. The D the body
// sees is the fluid MIXED across the card's diffuse edge each step -- not the
// interior fluid's inertia, which the step order does not charge
// (tools/probe-steer-inertia.js) -- so about an offset centre of mass it
// should grow like the parallel-axis term, with reach^2.
//
// THE INSTRUMENT is a lock-in: physics_steer.wgsl adds a probe torque
// eps * s_n about C with s_n = +-1 flipping every step. For the model above
// the response at that frequency is
//
//     G = < omega_{n+1} s_n > = eps / (I (2 - D/I))   =>   D/I = 2 - eps/(I G)
//
// A body with no fluid reads D/I = 0. Any real memory in the fluid's response
// makes D the step-frequency value of it, which is the one that decides this
// mode. Windows at +eps and -eps are differenced, which cancels whatever the
// card's own flight correlates with s_n; a kick-off window is printed as the
// control, and two amplitudes check linearity.
//
//   node tools/probe-steer-damping.js
//   node tools/probe-steer-damping.js --configs="comReach=0.5;comReach=0.5&kEps=3"

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9349, settle: 3000, n: 256, skip: 32, eps: 0.2,
    configs: 'comReach=0;comReach=0.25;comReach=0.5;comReach=0.7;comReach=0.8', mode: 'torque' };
  for (const a of argv) {
    const i = a.indexOf('='); const k = a.slice(2, i), v = a.slice(i + 1);
    if (!(k in o) || i < 0) { console.error(`unknown argument: ${a}`); process.exit(2); }
    o[k] = (k === 'baseUrl' || k === 'configs' || k === 'mode') ? v : parseFloat(v);
  }
  return o;
}

async function ev(R, e, t) {
  const r = await evalExpr(R, e, t);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  const L = 'window.__LBM';

  // One window: kick at `eps`, discard `skip` steps of transient, then
  // correlate omega after each step with the sign that step's kick used.
  // mode=force: the broadside force probe at C, read out as the centre of
  // mass's velocity across the chord, and M in place of I throughout.
  const FORCE = o.mode === 'force';
  const window = async (eps) => {
    await ev(Runtime, `${L}.debugSetKick(${eps}, ${FORCE})`);
    await ev(Runtime, `${L}.debugStepSync(${o.skip})`, 60000);
    let sign = (await ev(Runtime, `${L}.debugReadSteer()`)).kickSign > 0.5 ? -1 : 1;
    let g = 0, om2 = 0, dead = false;
    for (let j = 0; j < o.n; j++) {
      await ev(Runtime, `${L}.debugStepSync(1)`, 60000);
      const c = await ev(Runtime, `${L}.debugReadCardState()`);
      if (!c.every(Number.isFinite) || (c[6] === 0 && c[7] === 0)) { dead = true; break; }
      let x = c[5];
      if (FORCE) {
        const st = await ev(Runtime, `${L}.debugReadSteer()`);
        const rx = st.d * Math.cos(c[2]), ry = st.d * Math.sin(c[2]);
        const vcx = c[3] - c[5] * ry + st.dd * Math.cos(c[2]), vcy = c[4] + c[5] * rx + st.dd * Math.sin(c[2]);
        x = -vcx * Math.sin(c[2]) + vcy * Math.cos(c[2]);
      }
      g += x * sign; om2 += x * x;
      sign = -sign;
    }
    return { g: g / o.n, rms: Math.sqrt(om2 / o.n), dead };
  };

  console.log(`  lock-in: settle ${o.settle} steps with the ballast parked, then windows of ${o.n} (after ${o.skip} to settle)`);
  console.log('    config                         I        G0/(eps/2I)   D/I (eps)   D/I (2eps)   D (eps)     omega rms');
  try {
    for (const q of o.configs.split(';')) {
      await navigateTo(Page, `${o.baseUrl}/index-steer.html?comFrame=body&startPaused=1&${q}`);
      await waitForGlobal(Runtime, L, 60000);
      await ev(Runtime, `${L}.setSteer(1)`);
      await ev(Runtime, `${L}.debugStepSync(${o.settle})`, 600000);
      const I = (await ev(Runtime, `${L}.debugReadCardState()`))[FORCE ? 9 : 10];   // M for the force probe
      const w0 = await window(0);
      const wp = await window(o.eps), wm = await window(-o.eps);
      const wp2 = await window(2 * o.eps), wm2 = await window(-2 * o.eps);
      await ev(Runtime, `${L}.debugSetKick(0, ${FORCE})`);
      const dOverI = (gp, gm, e) => 2 - e / (I * ((gp - gm) / 2));
      const d1 = dOverI(wp.g, wm.g, o.eps), d2 = dOverI(wp2.g, wm2.g, 2 * o.eps);
      const dead = [w0, wp, wm, wp2, wm2].some(w => w.dead);
      console.log(`    ${q.padEnd(30)} ${I.toFixed(0).padStart(6)}   ${(w0.g / (o.eps / (2 * I))).toFixed(3).padStart(8)}      `
        + `${d1.toFixed(3).padStart(7)}     ${d2.toFixed(3).padStart(7)}    ${(d1 * I).toFixed(0).padStart(7)}     ${w0.rms.toExponential(2)}${dead ? '   (FLUID DIED -- read nothing here)' : ''}`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }
  console.log('\n  G0 is the kick-off control, in units of a fluid-free body\'s response: it should be ~0.');
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
