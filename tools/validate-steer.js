#!/usr/bin/env node
// index-steer.html's physics against index.html's, on the GPU.
//
// physics_steer.wgsl claims that with the centre of mass on the geometric
// centre (steering input 0) it IS physics.wgsl: every term it adds is a
// product with the offset. The dense page is deterministic (its force
// reduction is integer atomics, so the sum is order-free), so that claim is
// checkable exactly, not to a tolerance: both pages are stepped to the same
// step and their CardState compared value for value at every checkpoint.
//
// Then what the steering is supposed to DO, each leg a fresh page load:
//   alive    input +1 must move the trajectory away from index.html's -- the
//            identity above means nothing if the input is not wired through
//   slew     the ballast's position d must ramp at exactly the slew rate to
//            exactly the reach, then hold (body frame, so the target is fixed)
//   couple   weight at the centre of mass and buoyancy at the geometric
//            centre make a couple: the HEAVY END MUST DROP. Input +1 (mass
//            toward the +x end) must turn the card clockwise on screen
//            (theta up, y is down) relative to input 0, and -1 the other way.
//   world    ?comFrame=world must also move the trajectory, and its d must
//            track reach*cos(theta) -- the world-horizontal shift projected
//            onto the chord.
//   authority  the centre of mass parked at the TIP (?comReach=1) must fly
//            --authoritySteps steps with the step-frequency stabilizer on,
//            and the SAME run with it off (?stab=0) must die: without the
//            control this could not tell a working stabilizer from a run too
//            short to fail. (physics_steer.wgsl 2a, probe-steer-damping.js.)
//   lowfreq  with the stabilizer on and input 0 the card must still be
//            index.html's to 1e-3 after the first checkpoint -- it acts at
//            the step frequency only.
//
//   node tools/validate-steer.js
//   node tools/validate-steer.js --steps=8192 --res=7

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9349, steps: 4096, start: 512, chunk: 256, res: 7, authoritySteps: 8192 };
  for (const a of argv) {
    const [k, v] = a.replace(/^--/, '').split('=');
    if (k in o && v !== undefined) o[k] = k === 'baseUrl' ? v : parseInt(v);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  if (o.chunk % 2 || o.start % 2) { console.error('--chunk and --start must be even (the ping-pong parity)'); process.exit(2); }
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

  // One leg: load, pause, bring to the common start step at input 0, apply
  // the input, and record CardState (plus the ballast) at every checkpoint.
  const leg = async (page, query, input) => {
    await navigateTo(Page, `${o.baseUrl}/${page}?res=${o.res}${query}`);
    await waitForGlobal(Runtime, L, 60000);
    await ev(Runtime, `${L}.setLive(false)`);
    const s0 = await ev(Runtime, `${L}.getStep()`);
    if (s0 > o.start || s0 % 2) throw new Error(`${page}: frame loop ran ${s0} steps before pausing (need <= ${o.start}, even)`);
    if (o.start > s0) await ev(Runtime, `${L}.debugStepSync(${o.start - s0})`, 600000);
    const steered = page.includes('steer');
    if (steered) await ev(Runtime, `${L}.setSteer(${input})`);
    const params = steered ? await ev(Runtime, `${L}.getSteerParams()`) : null;
    const rows = [];
    for (let s = o.start; s < o.steps; s += o.chunk) {
      await ev(Runtime, `${L}.debugStepSync(${o.chunk})`, 600000);
      const card = await ev(Runtime, `${L}.debugReadCardState()`);
      const st = steered ? await ev(Runtime, `${L}.debugReadSteer()`) : null;
      rows.push({ step: await ev(Runtime, `${L}.getStep()`), card, st });
    }
    return { rows, params };
  };

  const results = [];
  const report = (name, ok, msg) => { results.push(ok); console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(8)} ${msg}`); };

  try {
    console.log(`  res=${o.res}, input applied at step ${o.start}, checkpoints every ${o.chunk} to ${o.steps}\n`);
    const ref = await leg('index.html', '', 0);
    // The identity is with the step-frequency stabilizer OFF (?stab=0): that
    // is the page's claim to BE physics.wgsl. With it on, see 'lowfreq'.
    const zero = await leg('index-steer.html', '&comFrame=body&stab=0', 0);
    const zeroStab = await leg('index-steer.html', '&comFrame=body', 0);
    const plus = await leg('index-steer.html', '&comFrame=body', 1);
    const minus = await leg('index-steer.html', '&comFrame=body', -1);
    const world = await leg('index-steer.html', '&comFrame=world', 1);

    // identity: value equality (=== treats -0 and 0 as equal, which is the
    // one thing a product with a zero offset can legitimately change).
    let firstBad = null, compared = 0;
    for (let i = 0; i < ref.rows.length; i++) {
      const a = ref.rows[i].card, b = zero.rows[i].card;
      for (let k = 0; k < a.length; k++) { compared++; if (a[k] !== b[k] && firstBad === null) firstBad = { step: ref.rows[i].step, k, a: a[k], b: b[k] }; }
    }
    const dZero = zero.rows.every(r => r.st.d === 0);
    report('identity', !firstBad && dZero, firstBad
      ? `input 0 departs from index.html at step ${firstBad.step}, word ${firstBad.k}: ${firstBad.a} vs ${firstBad.b}`
      : `input 0 == index.html, ${compared} values over ${ref.rows.length} checkpoints; ballast stayed at 0: ${dZero}`);

    // lowfreq: the stabilizer is a virtual inertia at the STEP frequency
    // only, so on the card's own time scales it should be invisible. Scored
    // at the first checkpoint, before the flutter's own sensitivity to
    // initial conditions amplifies ANY perturbation (a single ULP included).
    {
      const a = ref.rows[0].card, b = zeroStab.rows[0].card;
      const dth = Math.abs(a[2] - b[2]), dv = Math.hypot(a[3] - b[3], a[4] - b[4]) / Math.hypot(a[3], a[4]);
      report('lowfreq', dth < 1e-3 && dv < 1e-3,
        `stabilizer on, input 0, ${o.chunk} steps after release: |dtheta| ${dth.toExponential(2)} rad, |dv|/|v| ${dv.toExponential(2)} (want both < 1e-3)`);
    }

    const last = (l) => l.rows[l.rows.length - 1].card;
    // Travel, from the accumulators, over the steered window only: [20]/[21]
    // are wrapped every 16 domains, which a few thousand steps never reach.
    const travel = (l) => [l.rows[l.rows.length - 1].card[21] - l.rows[0].card[21], l.rows[l.rows.length - 1].card[20] - l.rows[0].card[20]];
    const dist = (a, b) => { const p = travel(a), q = travel(b); return Math.hypot(p[0] - q[0], p[1] - q[1]); };
    const moved = dist(plus, ref);
    report('alive', moved > 0.5, `input +1 ends ${moved.toFixed(2)} cells from index.html's card (theta ${last(ref)[2].toFixed(3)} -> ${last(plus)[2].toFixed(3)})`);

    // slew: d after k steps is min(k * slew, reach), f32-accumulated.
    const { reach, slew } = plus.params;
    let slewErr = 0;
    for (const r of plus.rows) {
      const k = r.step - o.start;
      slewErr = Math.max(slewErr, Math.abs(r.st.d - Math.min(k * slew, reach)));
    }
    report('slew', slewErr < 1e-3 * reach, `d follows min(k*${slew.toFixed(4)}, ${reach.toFixed(3)}) to ${slewErr.toExponential(2)} cells`);

    // couple: the heavy end drops. Compared at the first checkpoint, before
    // the two runs' wakes have diverged enough to make theta a different
    // experiment -- this is the couple's direct effect, not the flight.
    const th = (l, i) => l.rows[i].card[2];
    const dPlus = th(plus, 0) - th(zero, 0), dMinus = th(minus, 0) - th(zero, 0);
    report('couple', dPlus > 0 && dMinus < 0,
      `theta vs input 0 after ${o.chunk} steps: +1 -> ${dPlus >= 0 ? '+' : ''}${dPlus.toExponential(2)}, -1 -> ${dMinus.toExponential(2)} (want +, -)`);

    // world: d == reach * cos(theta) once the ballast has caught up with a
    // target that moves only as fast as the card turns.
    let worldErr = 0;
    for (const r of world.rows.slice(2)) worldErr = Math.max(worldErr, Math.abs(r.st.d - reach * Math.cos(r.card[2])));
    const wMoved = dist(world, ref);
    report('world', wMoved > 0.5 && worldErr < 0.05 * reach,
      `moves the card ${wMoved.toFixed(2)} cells; d tracks reach*cos(theta) to ${(worldErr / reach * 100).toFixed(2)}% of reach`);

    // authority: res 8 (the page's default), where the unstabilized run is
    // measured to die in ~4000 steps from rest.
    const fly = async (stab) => {
      await navigateTo(Page, `${o.baseUrl}/index-steer.html?startPaused=1&comFrame=body&comReach=1&stab=${stab}`);
      await waitForGlobal(Runtime, L, 60000);
      await ev(Runtime, `${L}.setSteer(1)`);
      for (let s = 0; s < o.authoritySteps; s += 512) {
        await ev(Runtime, `${L}.debugStepSync(512)`, 600000);
        const c = await ev(Runtime, `${L}.debugReadCardState()`);
        // Dead: non-finite, or the force reduction reading exactly zero, which
        // is what NaN containment reports for a non-finite fluid.
        if (!c.every(Number.isFinite) || (c[6] === 0 && c[7] === 0)) return s + 512;
      }
      return null;
    };
    const on = await fly(1), off = await fly(0);
    report('authority', on === null && off !== null,
      `tip-parked centre of mass, ${o.authoritySteps} steps: stabilizer on ${on === null ? 'flies' : `DIES at ${on}`}, off ${off === null ? 'ALSO FLIES (control failed to fail)' : `dies at ${off}`}`);
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }
  const bad = results.filter(x => !x).length;
  console.log(bad ? `\nFAIL: ${bad} of ${results.length}` : `\nPASS: all ${results.length} checks`);
  process.exit(bad ? 1 : 0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
