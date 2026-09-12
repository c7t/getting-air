#!/usr/bin/env node
// THE MOVING-BODY COUPLING, MEASURED AGAINST ITSELF. plans/3D.md D1.
//
// WHAT IT IS FOR. D1 is "a free fall never reaches terminal velocity", and
// the reason that was hard to act on is that every number available to
// contradict it came from somewhere else: Schiller-Naumann is a different
// discretization, `?levels=2` is a different grid, a recorded Cd is a
// different session. A drag coefficient that disagrees with any of those has
// a dozen candidate explanations and no way to pick one.
//
// This removes all of them. The body moves at `tow`, the fluid sits at
// `stream`, and only the DIFFERENCE is physical -- a rigid translation of
// both is a change of inertial frame, which the Navier-Stokes equations and
// the lattice-Boltzmann scheme are both invariant under. So hold the
// difference fixed at U and slide the split:
//
//     a = 0     tow = 0,     stream = -U    the body is PINNED. Every
//                                           validated sphere case in this
//                                           suite is this leg.
//     a = U     tow = U,     stream = 0     the body MOVES through still
//                                           fluid. This is the coupling D1
//                                           is about.
//     between   tow = a,     stream = a-U   the same flow, seen from a frame
//                                           moving at a.
//
// EVERY LEG IS THE SAME PHYSICAL PROBLEM IN THE SAME DOMAIN WITH THE SAME
// SPONGE AND THE SAME BLOCKAGE. The reference value is not a number from a
// paper -- it is the other legs. Cd(a) must be constant in a, and the shape
// of the curve says what kind of defect it is:
//
//     flat                  the coupling is Galilean invariant; D1 is
//                           somewhere else.
//     falls linearly in a   a term proportional to the body's own speed is
//                           missing from the momentum exchange.
//     falls with a^2        a term quadratic in it is.
//
// WHY NOT TWO LEGS. tools/probe-d3-window.js already runs the two ENDS as
// `tow` and `stream`, and M8.3 read the gap between them as resolution --
// -29.5% at D = 12 closing to -13.9% at D = 32, both frames extrapolating to
// the same Cd. A two-point comparison cannot separate "a defect that scales
// with the body's speed" from "a coarse body", because the two ends differ in
// every way at once. The INTERIOR of the sweep is what separates them: a
// resolution effect cannot depend on which frame the same grid is described
// in, so any variation across a is the coupling and nothing else.
//
// ALSO REPORTED: THE MOMENTUM BUDGET, which is the mechanism rather than the
// symptom. The sponge makes the domain momentum non-conserved, so the budget
// is reported only as a diagnostic here -- see the `--budget` leg, which runs
// the same body in a sponge-free periodic box (`?scenario=drift`) where the
// fluid's momentum can only change through the body, so
//
//     sum over the run of F_reported   must equal   -(P_end - P_start)
//
// exactly. That one IS a closed statement with no reference value in it, and
// it is what says whether the force kernel is measuring the whole momentum
// transfer or only the part that crosses the bounce-back links.
//
// SAMPLE EVERY STEP. A moving bounce-back body's force oscillates with a
// period of exactly 1/U steps and an amplitude comparable to the drag itself,
// so the obvious sampling rate is exactly the alias -- see
// tools/probe-d3-window.js's own note, which cost a full wrong conclusion.
// Every leg here samples every step and the RMS is printed alongside the
// mean, because on a moving body that RMS is most of the signal.
//
// Owns the whole lifecycle (HTTPS server + a dedicated debug-port Chrome if
// neither is up, one tab reused via Page.navigate), like tools/validate-3d.js.
//
//   node tools/probe-d3-galilean.js
//   node tools/probe-d3-galilean.js --u=0.025 --n=12 --re=160 --td=16
//   node tools/probe-d3-galilean.js --splits=0,1            # just the ends
//   node tools/probe-d3-galilean.js --budget                # the mechanism
//   node tools/probe-d3-galilean.js --scenario=card --n=24 --re=300 \
//        --scenarioExtra=tilt=0,span=1,aspect=0.125 --u=0.03

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  // D = 12 and Re = 160 -> tau = 0.509 at u_t = 0.04, which is where D1's own
  // reproduction lives. `u` is the RELATIVE speed and therefore the Reynolds
  // number: Re = u * n / nu, and nu is fixed by `re` and `u_t` together.
  // `fall` is a sphere and `card` is a plate, and BOTH carry the split. The
  // plate is the sharper instrument -- broadside and axis-aligned it has no
  // staircase error at all, so its Cd is not sitting on top of a +10% offset
  // the way every sphere case in this suite is.
  scenario: 'fall',
  n: 12, re: 160, u: 0.025, td: 16, splits: null, budget: false,
  // Forwarded verbatim to whichever scenario is selected. `tilt=0` is what
  // makes the plate exactly lattice-aligned and is not the card's default,
  // which starts it pitched so that a FREE fall has something to fall over
  // from; a prescribed leg cannot turn, so the pitch would just be a fixed
  // angle of attack nobody asked for.
  scenarioExtra: '',
  // The budget leg runs a DIFFERENT scenario (`drift`), so it takes its own
  // extras: `?tau=` is one of drift's knobs and is not one of `fall`'s, and
  // a single --extra= would be rejected by whichever page did not own it.
  extra: '', budgetExtra: 'tau=0.509', budgetSteps: 0, timeout: 900, keepOpen: false,
};

// The sweep. 0 is the pinned leg and 1 is the pure tow; the interior points
// are what make this an instrument rather than a two-point comparison.
const SPLITS = [0, 0.25, 0.5, 0.75, 1];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--scenario=')) o.scenario = a.slice(11);
    else if (a.startsWith('--scenarioExtra=')) o.scenarioExtra = a.slice(16).replace(/^[?&]/, '');
    else if (a.startsWith('--n=')) o.n = parseInt(a.slice(4));
    else if (a.startsWith('--re=')) o.re = Number(a.slice(5));
    else if (a.startsWith('--u=')) o.u = Number(a.slice(4));
    else if (a.startsWith('--td=')) o.td = Number(a.slice(5));
    else if (a.startsWith('--splits=')) o.splits = a.slice(9).split(',').map(Number);
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--budgetExtra=')) o.budgetExtra = a.slice(14).replace(/^[?&]/, '');
    else if (a.startsWith('--budgetSteps=')) o.budgetSteps = parseInt(a.slice(14));
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--budget') o.budget = true;
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : String(x));
const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));

// --- one leg of the split sweep --------------------------------------------

async function runSplit(Runtime, o, a, log) {
  const G = 'window.__D3';
  const p = await ev(Runtime, `${G}.getParams()`, 30000, 'getParams');
  const D = p.D, U = p.uRel;
  if (!(U > 0)) throw new Error('a split leg needs a nonzero relative speed');
  // Drag points along the RELATIVE flow, which this sweep deliberately keeps
  // in -x on every leg (stream - tow = -U), so the sign is a constant. Taken
  // from the scenario rather than assumed: a leg that resolved to the other
  // sign would show up as a 200% disagreement that looks exactly like the
  // defect being hunted.
  const cdSign = Math.sign(p.uRelSigned) || 1;
  const perTd = Math.max(1, Math.round(D / U));
  const total = Math.round(o.td * perTd);
  log(`tow=${padL(p.tow.toFixed(4), 8)} stream=${padL(p.stream.toFixed(4), 8)}`
    + `  uRel=${(-p.uRelSigned).toFixed(4)}  ${p.pinned ? 'PINNED ' : 'moving '}`
    + ` window ${p.windowAxes.some(Boolean) ? 'on ' : 'off'}`
    + `  ${perTd} steps/D-U, ${total} total`);

  const cds = [];
  let finite = true;
  for (let done = 0; done < total;) {
    const k = Math.min(2 * perTd, total - done);
    const run = await ev(Runtime, `${G}.debugRunAndCollect(${k}, 1)`, (o.timeout + 30) * 1000, 'run');
    done += k;
    for (const h of run.history) cds.push(cdSign * h[3]);
    if (run.history.some(h => !Number.isFinite(h[1]))) { finite = false; break; }
  }
  const st = await ev(Runtime, `${G}.readStats()`, 300000, 'readStats');
  if (!st.finite) finite = false;
  // THE LAST THIRD, not the whole run: every leg starts from a uniform field
  // and needs some convective times to develop a wake, and the pinned leg
  // develops it at a different rate from the towed one (the towed body's own
  // start-up is a step change in the boundary condition). Averaging the
  // transient in would put the difference between two start-ups into a
  // comparison that is supposed to be about the settled state.
  const tail = cds.slice(Math.floor(cds.length * (2 / 3))).filter(Number.isFinite);
  const cd = mean(tail);
  return {
    a, tow: p.tow, stream: p.stream, pinned: p.pinned, cd,
    rms: Math.sqrt(mean(tail.map(c => (c - cd) ** 2))),
    maxSpeed: st.maxSpeed, rhoMin: st.rhoMin, rhoMax: st.rhoMax, finite,
    re: U * D / p.nu, tau: p.tau, blockage: p.blockage,
  };
}

// --- the momentum budget ----------------------------------------------------
//
// `drift` is the one scenario in the suite with a MOVING body, NO sponge and
// NO walls -- fully periodic, fluid force measured and never applied. So the
// fluid's total momentum can change for exactly one reason, and the budget
// closes or it does not:
//
//     P(end) - P(start)  =  - sum over steps of F_reported
//
// The solid interior is included in P, and it has to be: the host cannot
// cheaply ask which cells are inside the body. Under SOLID_EQ every interior
// cell carries exactly (rho = 1, u = u_body), so at a CONSTANT body velocity
// its whole contribution is |S| * u_body and the only part that survives the
// difference is the handful of cells that changed hands -- which is precisely
// the term under suspicion, so it must be in, not out.
async function runBudget(Runtime, o, log) {
  const G = 'window.__D3';
  const p = await ev(Runtime, `${G}.getParams()`, 30000, 'getParams');
  // DEFAULT: TWO CONVECTIVE TIMES, warm and measure. `drift` starts from rest
  // and one chord/U at its defaults is 640 steps, so the 200 an earlier
  // version used was 0.3 of one -- a measurement taken entirely inside the
  // impulsive start, where the drag is several times its settled value and
  // the residual is whatever the transient happens to be doing.
  // SHORT, AND DELIBERATELY SO. `drift` is periodic with NO sponge, which is
  // exactly what makes the budget a closed statement -- and also means the
  // body pumps momentum into a CLOSED BOX forever. Left running it does not
  // settle, it accelerates: measured at 1500 steps the reported impulse was
  // +2.5e4 against a drag of order 0.1 per step, i.e. the run was long gone.
  // So the window is a few tens of steps of body travel, and `maxSpeed` and
  // the density range are REPORTED beside the closure, because a budget that
  // silently reports a diverged field is worse than no budget.
  const steps = o.budgetSteps || 300;
  // WARM UP BEFORE THE FIRST SAMPLE, and this is not settling -- it is a
  // one-time OFFSET that would otherwise be read as the whole defect.
  //
  // The scenario seeds u = 0 EVERYWHERE, including inside the body, and
  // SOLID_EQ then fills the solid interior to u_body over the first step. For
  // a D = 16 sphere that is (4/3) pi 8^3 * u_body = 53.6 of momentum appearing
  // in the very first sample interval, against a whole-run impulse of ~67 --
  // i.e. 80% of the number, none of it physics. Measured without this warmup
  // the budget looked 46-48% short whether the swept-cell term was in or out,
  // which reads exactly like "the fix does nothing" and is in fact "the
  // measurement is dominated by its own initial condition".
  const warm = steps;
  await ev(Runtime, `${G}.debugStepSync(${warm})`, (o.timeout + 30) * 1000, 'warmup');
  if (log) log(`warmed ${warm} steps before the first sample -- the interior fill is not a leak`);
  const before = await ev(Runtime, `${G}.readStats()`, 300000, 'readStats');
  const run = await ev(Runtime, `${G}.debugRunAndCollect(${steps}, 1)`, (o.timeout + 30) * 1000, 'run');
  const after = await ev(Runtime, `${G}.readStats()`, 300000, 'readStats');
  const sumF = [1, 2, 6].map(() => 0);
  // history rows are [step, fx, fy, Cd, Cl, Cs, fz].
  for (const h of run.history) { sumF[0] += h[1]; sumF[1] += h[2]; sumF[2] += h[6]; }
  const dP = [after.px - before.px, after.py - before.py, after.pz - before.pz];
  const dMass = after.mass - before.mass;
  return { steps, sumF, dP, dMass, before, after, p };
}

// --- main -------------------------------------------------------------------

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const splits = o.splits || SPLITS;

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  await attachPageWatch(client, { onError: (e) => console.error(`    !! [${e.kind}] ${e.text.split('\n')[0]}`) });

  const out = [];
  let budget = null;
  try {
    for (const a of splits) {
      // tow = a*U, stream = a*U - U. Both signed, both bounded by U, so the
      // sweep changes the FRAME and not the Mach number.
      const tow = a * o.u, stream = a * o.u - o.u;
      const q = [
        `scenario=${o.scenario}`, `n=${o.n}`, `re=${o.re}`, 'q=19', 'bounceback=1', 'live=0',
        `tow=${tow}`, `stream=${stream}`,
        ...(o.scenarioExtra ? [o.scenarioExtra] : []),
      ].join('&');
      const url = `${o.baseUrl}/index-3d.html?${q}${o.extra ? `&${o.extra}` : ''}`;
      console.log(`\n=== split a = ${a}   (${url})`);
      try {
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, 'window.__D3', 60000);
        out.push(await runSplit(Runtime, o, a, s => console.log('    ' + s)));
      } catch (err) {
        console.log(`    ERROR ${err.message.split('\n')[0]}`);
        out.push({ a, tow, stream, error: err.message });
      }
    }

    if (o.budget) {
      const q = ['scenario=drift', `u0=${o.u}`, 'q=19', 'bounceback=1', 'live=0'].join('&');
      const url = `${o.baseUrl}/index-3d.html?${q}${o.budgetExtra ? `&${o.budgetExtra}` : ''}`;
      console.log(`\n=== momentum budget   (${url})`);
      try {
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, 'window.__D3', 60000);
        budget = await runBudget(Runtime, o, s => console.log('    ' + s));
      } catch (err) {
        console.log(`    ERROR ${err.message.split('\n')[0]}`);
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  const good = out.filter(r => !r.error && r.finite);
  console.log('\n' + '='.repeat(100));
  console.log(`THE GALILEAN SPLIT   ?scenario=${o.scenario} n=${o.n}  relative U=${o.u}`
    + (good.length ? `  Re=${good[0].re.toFixed(0)}  tau=${good[0].tau.toFixed(4)}`
      + `  blockage ${(good[0].blockage * 100).toFixed(2)}%` : ''));
  console.log('='.repeat(100));
  console.log(pad('a', 7) + pad('tow', 10) + pad('stream', 10) + pad('body', 9)
    + pad('Cd', 10) + pad('+- rms', 10) + pad('vs a=0', 10) + pad('max|u|', 10) + 'state');
  console.log('-'.repeat(100));
  const base = out.find(r => r.a === 0 && !r.error && r.finite);
  for (const r of out) {
    if (r.error) { console.log(pad(r.a, 7) + 'ERROR'); continue; }
    const rel = base ? (r.cd - base.cd) / base.cd : NaN;
    console.log(pad(r.a, 7) + pad(r.tow.toFixed(4), 10) + pad(r.stream.toFixed(4), 10)
      + pad(r.pinned ? 'pinned' : 'moving', 9)
      + pad(f4(r.cd), 10) + pad(f3(r.rms), 10)
      + pad(Number.isFinite(rel) ? `${(rel * 100).toFixed(1)}%` : '--', 10)
      + pad(f4(r.maxSpeed), 10) + (r.finite ? 'ok' : 'NON-FINITE'));
  }

  if (base && good.length > 1) {
    const end = good[good.length - 1];
    console.log(`\nCd falls ${((end.cd - base.cd) / base.cd * 100).toFixed(1)}% from the pinned leg to a = ${end.a}.`);
    console.log('EVERY LEG IS THE SAME FLOW IN THE SAME DOMAIN, so this number is not resolution,');
    console.log('not blockage and not the drag correlation -- it is the moving-body coupling. Read the');
    console.log('SHAPE across a, not the endpoint: linear in a means a term proportional to the body');
    console.log('speed is missing from the momentum exchange, quadratic means one that goes as its square.');
    console.log('The RMS column is the staircase, and on a moving body it is most of the signal --');
    console.log('these means are over every step for that reason (probe-d3-window.js\'s alias note).');
  }

  if (budget) {
    const { steps, sumF, dP, dMass, p } = budget;
    console.log('\n' + '='.repeat(100));
    console.log(`THE MOMENTUM BUDGET   ?scenario=drift  ${p.NX}x${p.NY}x${p.NZ}, periodic, NO sponge, ${steps} steps`);
    console.log('='.repeat(100));
    console.log('A closed statement: the fluid momentum can only change through the body, so the');
    console.log('impulse the force kernel reports must be MINUS the momentum the field actually gained.');
    for (const [i, ax] of ['x', 'y', 'z'].entries()) {
      const closure = sumF[i] + dP[i];
      const scale = Math.max(Math.abs(sumF[i]), Math.abs(dP[i]));
      console.log(`  ${ax}:  sum F = ${padL(sumF[i].toExponential(3), 11)}`
        + `   dP = ${padL(dP[i].toExponential(3), 11)}`
        + `   sum F + dP = ${padL(closure.toExponential(3), 11)}`
        + `   ${scale > 0 ? `(${(closure / scale * 100).toFixed(1)}% of the larger)` : ''}`);
    }
    console.log(`  mass drift ${dMass.toExponential(3)} over ${steps} steps`);
    console.log(`  field: max|u| ${budget.before.maxSpeed.toFixed(5)} -> ${budget.after.maxSpeed.toFixed(5)}`
      + `   rho [${budget.after.rhoMin.toFixed(5)}, ${budget.after.rhoMax.toFixed(5)}]`
      + `   ${budget.after.finite ? '' : 'NON-FINITE -- the closure below is meaningless'}`);
    console.log('A nonzero x closure is momentum the fluid gained or lost that the body was never');
    console.log('charged for -- the covered/fresh cells the body sweeps through, which the momentum');
    console.log('exchange over the bounce-back links does not see.');
  }

  // REPORTS, DOES NOT GATE. What "flat enough" means for Cd(a) is the
  // question D1 is asking, and a bound invented before the curve is in hand
  // would be a number fitted to whatever happened to be measured first.
  process.exit(out.some(r => r.error) ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
