#!/usr/bin/env node
// THE MOVING WINDOW, MEASURED. plans/3D.md M8.3.
//
// WHAT IT IS FOR, and why "the run still looks fine" is not it. A moving
// window is the archetype of a change that cannot be seen: the picture is
// plausible whether it works or not, the solve stays finite whether it works
// or not, and the only symptom of a broken one is a drag coefficient that
// drifts instead of converging -- ten thousand steps and a GPU later. So
// this reports the two things that separate those cases, and reports them
// as a TIME SERIES rather than as a final number:
//
//   THE ANCHOR INVARIANT. The body's position IN THE WINDOW must stay in its
//   anchor cell for the whole run, however far it has actually travelled.
//   That is the mechanism stated as a measurement, and it fails immediately
//   and loudly rather than late and subtly. `travelled` alongside it is what
//   makes the statement non-vacuous: holding a body that never moves is free.
//
//   Cd AGAINST CONVECTIVE TIME. The reason the window exists (M8.2b): a
//   towed body travels U per step, so the 30 D/U the validated sphere cases
//   settle for carries it 360 cells through a 192-cell domain, and the towed
//   Cd was still RISING (0.951 at 4 D/U, 1.067 at 8) toward the streamed
//   1.362 when it ran out of room. A window that works shows that series
//   FLATTENING instead of being cut off.
//
// THE GALILEAN PAIR IS RUN IN ONE INVOCATION, because the whole point of it
// is that the two legs share a domain, a blockage and a sponge -- and two
// separate invocations is exactly how those quietly stop being shared. The
// streamed leg is the control and is deliberately NOT windowed: a pinned
// body has nothing to follow, so a window there would be an inert branch in
// the one case that is supposed to be untouched.
//
// REPORTS, DOES NOT PASS/FAIL, on the Cd comparison. What "close enough"
// means for Cd(tow) against Cd(stream) is M8.2c's question and it needs this
// series in hand first. The ANCHOR INVARIANT is gated, because that one has
// an exact answer.
//
// Owns the whole lifecycle (HTTPS server + a dedicated debug-port Chrome if
// neither is up, one tab reused via Page.navigate), like tools/validate-3d.js
// and tools/probe-d3-tau.js.
//
//   node tools/probe-d3-window.js
//   node tools/probe-d3-window.js --u=0.04 --td=30 --n=12
//   node tools/probe-d3-window.js --legs=tow --td=60          # the long one
//   node tools/probe-d3-window.js --legs=tow,tow-nowindow     # the A/B
//   node tools/probe-d3-window.js --legs=tow --sample=25      # reproduce the alias

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  n: 12, re: 100, u: 0.04, td: 30, every: 2, sample: 1, extra: '',
  legs: null, timeout: 900, keepOpen: false,
};

// The three legs, and the third is the control that makes the first two
// legible. `tow-nowindow` is the SAME towed run with the window switched off
// -- i.e. M8.2b's measurement, reproduced in the same build -- so the series
// can be compared against the one that ran out of domain rather than against
// a number recorded in another session. CLAUDE.md's standing rule about
// build-vs-build claims applies here too.
const LEGS = [
  { name: 'tow', frame: 'tow', window: true,
    note: 'the body MOVES through still fluid, in a window that follows it' },
  { name: 'stream', frame: 'stream', window: false,
    note: 'the body is PINNED in a freestream -- the other frame, and the control' },
  { name: 'tow-nowindow', frame: 'tow', window: false,
    note: 'the same tow with ?window=0 -- M8.2b, in this build' },
];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--n=')) o.n = parseInt(a.slice(4));
    else if (a.startsWith('--re=')) o.re = Number(a.slice(5));
    else if (a.startsWith('--u=')) o.u = Number(a.slice(4));
    else if (a.startsWith('--td=')) o.td = Number(a.slice(5));
    else if (a.startsWith('--every=')) o.every = Number(a.slice(8));
    else if (a.startsWith('--sample=')) o.sample = Math.max(1, parseInt(a.slice(9)));
    else if (a.startsWith('--legs=')) o.legs = a.slice(7).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : String(x));

async function runLeg(Runtime, o, leg, log) {
  const G = 'window.__D3';
  const p = await evalOrThrow(Runtime, `${G}.getParams()`, 30000, 'getParams');
  const D = p.D, U = p.uRel;
  // DRAG IS ALONG THE RELATIVE FLOW, AND THE TWO FRAMES DISAGREE ABOUT WHICH
  // WAY THAT POINTS. The page reports Cd from fx, which is right for a
  // streamed body (fluid moving +x, drag +x) and inverted for a towed one
  // (body moving +x, so the fluid moves -x relative to it and the drag it
  // feels is -x). Comparing the raw numbers would report the Galilean pair
  // as disagreeing by 200% and the mistake would look exactly like a
  // coupling bug -- which is the whole thing this pair exists to detect.
  const cdSign = p.tow > 0 ? -1 : 1;
  // Every leg here is one of the two PRESCRIBED frames, so the relative speed
  // is known up front and the convective time is a constant. A free fall has
  // no `uRel` -- its speed is what is being measured -- and belongs to M8.2c
  // rather than here; said rather than left to divide by null.
  if (!(U > 0)) throw new Error('this probe needs a prescribed ?tow= or ?stream= speed');
  const perTd = Math.max(1, Math.round(D / U));           // steps in one D/U
  const chunk = Math.max(1, Math.round(o.every * perTd));
  const total = Math.round(o.td * perTd);
  const windowed = p.windowAxes.some(Boolean);
  log(`D=${D} Re=${p.re.toFixed(0)} U=${U} tau=${p.tau.toFixed(5)} domain ${p.NX}x${p.NY}x${p.NZ}`
    + ` blockage ${(p.blockage * 100).toFixed(2)}%  sponge ${p.sponge.width}`);
  log(`window ${windowed ? `on (${'xyz'.split('').filter((_, i) => p.windowAxes[i]).join('')},`
      + ` anchor ${p.windowAnchor.join(',')})` : 'OFF'}`
    + `   ${perTd} steps per D/U, ${total} total`);

  // SAMPLE EVERY STEP, AND THE REASON IS NOT NOISE -- IT IS ALIASING, AND IT
  // COST A FULL WRONG CONCLUSION.
  //
  // A moving bounce-back body's instantaneous force is dominated by the
  // staircase: the discrete surface changes as the body crosses each cell, so
  // the force oscillates with a period of exactly 1/U STEPS and an amplitude
  // comparable to the drag itself. The obvious sampling rate -- once per cell
  // the body crosses -- is exactly that period, so every sample lands at the
  // SAME PHASE of the staircase and the average is not a time average at all.
  //
  // It does not look like an error. It looks like physics: measured at
  // 1/U = 25 steps this run reported Cd rising to 1.24, holding steady from
  // 24 to 32 D/U, and then COLLAPSING to 0.30 by 64 -- a perfectly smooth,
  // perfectly reproducible curve with an obvious story attached (the wake
  // wrapping round and catching the body). The flow was steady the whole
  // time: domain kinetic energy, enstrophy, max|u| and the density range were
  // constant to four digits from 24 D/U onward. What moved was the sampling
  // PHASE, dragged by the f32 drift in the body's sub-cell position -- 0.03
  // of a cell over 19200 steps, which is enough, because the alias has no
  // other timescale in it.
  //
  // The same recorded samples, decimated every 25, walk 1.11, 0.72, 0.85,
  // 0.79, 1.24, 1.54 across the run; decimated every 7 they sit at 0.98,
  // 0.96, 0.95, 0.95, 0.97, 0.97, matching the every-step mean to 0.5%. So
  // this is sampled dense and averaged, and `--sample=` exists to reproduce
  // the trap rather than to tune the cost.
  //
  // The streamed control has no such oscillation -- its body does not move --
  // which is exactly the asymmetry that would have made the pair look like it
  // disagreed for a reason that was not the coupling.
  const sampleEvery = o.sample;
  const series = [];
  let worstAnchor = 0, finite = true;
  for (let done = 0; done < total;) {
    const k = Math.min(chunk, total - done);
    const run = await evalOrThrow(Runtime, `${G}.debugRunAndCollect(${k}, ${sampleEvery})`,
      (o.timeout + 30) * 1000, 'debugRunAndCollect');
    done += k;
    const cds = run.history.map(h => cdSign * h[3]).filter(Number.isFinite);
    const b = await evalOrThrow(Runtime, `${G}.readBody()`, 120000, 'readBody');
    b.cd = mean(cds);
    b.cdRms = Math.sqrt(mean(cds.map(c => (c - b.cd) ** 2)));
    if (!cds.length || !Number.isFinite(b.cd) || !Number.isFinite(b.cx)) { finite = false; }
    // THE ANCHOR INVARIANT. The window's offset is an INTEGER number of
    // cells, so the body's window position is the anchor cell plus its own
    // sub-cell part: `win - anchor` must lie in [0, 1), always. `escape` is
    // how far outside that it got, and a working window holds it at exactly
    // zero -- not small, zero. Reporting the deviation itself would make a
    // body sitting legitimately at 0.999 of its anchor cell look like a
    // near-failure, which is the wrong reading entirely.
    const inCell = [0, 1, 2].map(i => (p.windowAxes[i] ? b.win[i] - b.winAnchor[i] : 0));
    const escape = Math.max(...inCell.map(d => Math.max(0, -d, d - 1)));
    worstAnchor = Math.max(worstAnchor, escape);
    series.push({ td: done / perTd, cd: b.cd, rms: b.cdRms, travel: b.travel,
                  win: b.win.slice(), escape, cx: b.cx });
    log(`  t=${padL((done / perTd).toFixed(1), 6)} D/U   Cd=${padL(f3(b.cd), 7)}`
      + ` +-${padL(f3(b.cdRms), 6)}   travelled ${padL(b.travel.toFixed(1), 7)}`
      + `   x_buf=${padL(b.cx.toFixed(1), 6)}`
      + `   x_win=${padL(b.win[0].toFixed(2), 7)}   in-cell ${inCell[0].toFixed(3)}`
      + `   escape ${escape.toFixed(3)}`);
    if (!finite) { log('  NON-FINITE -- stopping'); break; }
  }
  return { leg: leg.name, windowed, D, U, perTd, series, worstAnchor, finite,
           cdFinal: series.length ? series[series.length - 1].cd : NaN,
           // The LAST QUARTER's mean, which is what "converged" should be
           // read off rather than a single final sample.
           cdTail: mean(series.slice(Math.floor(series.length * 0.75)).map(s => s.cd)) };
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const legs = LEGS.filter(l => !o.legs || o.legs.includes(l.name));
  if (!legs.length) { console.error('no legs selected'); process.exit(2); }

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = await attachPageWatch(client, { onError: () => {} });

  const out = [];
  try {
    for (const leg of legs) {
      console.log(`\n=== ${leg.name} -- ${leg.note}`);
      const q = [
        'scenario=fall', `n=${o.n}`, `re=${o.re}`, 'q=19', 'bounceback=1', 'live=0',
        `${leg.frame}=${o.u}`,
        // Explicit on BOTH legs, never left to the scenario's default: a
        // control whose configuration is implicit is one nobody can check.
        `window=${leg.window ? 'x' : '0'}`,
      ].join('&');
      const url = `${o.baseUrl}/index-3d.html?${q}${o.extra ? `&${o.extra}` : ''}`;
      console.log(`    ${url}`);
      try {
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, 'window.__D3', 60000);
        out.push(await runLeg(Runtime, o, leg, s => console.log('    ' + s)));
      } catch (err) {
        console.log(`    ERROR ${err.message.split('\n')[0]}`);
        out.push({ leg: leg.name, error: err.message });
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(104));
  console.log(`SUMMARY   sphere D=${o.n} Re=${o.re} U=${o.u}, ${o.td} D/U per leg`);
  console.log('='.repeat(104));
  console.log(pad('leg', 16) + pad('window', 9) + pad('Cd final', 11) + pad('Cd tail', 11)
    + pad('travelled', 12) + pad('escape', 10) + 'verdict');
  console.log('-'.repeat(104));
  let bad = 0;
  for (const r of out) {
    if (r.error) { console.log(pad(r.leg, 16) + 'ERROR'); bad++; continue; }
    const last = r.series[r.series.length - 1] || {};
    // THE GATE: a windowed run must hold the body inside its anchor cell for
    // the whole run, and the bound is EXACTLY ZERO escape from it. The offset
    // is an integer, so there is nothing for a correct window to be
    // approximately right about.
    const held = !r.windowed || r.worstAnchor === 0;
    if (!held || !r.finite) bad++;
    console.log(pad(r.leg, 16) + pad(r.windowed ? 'on' : 'off', 9)
      + pad(f3(r.cdFinal), 11) + pad(f3(r.cdTail), 11)
      + pad((last.travel ?? 0).toFixed(1), 12) + pad(r.worstAnchor.toFixed(3), 10)
      + (!r.finite ? 'NON-FINITE' : held ? 'held' : 'DRIFTED OUT OF THE ANCHOR CELL'));
  }
  const tow = out.find(r => r.leg === 'tow' && !r.error);
  const str = out.find(r => r.leg === 'stream' && !r.error);
  if (tow && str) {
    const rel = (tow.cdTail - str.cdTail) / str.cdTail;
    console.log(`\nTHE GALILEAN PAIR: Cd(tow) = ${f3(tow.cdTail)} against Cd(stream) = ${f3(str.cdTail)}`
      + `  -> ${(rel * 100).toFixed(1)}%`);
    console.log('Reported, not gated. What "agree" means for these two is M8.2c\'s question, and it');
    console.log('needs the series above in hand -- read whether the TOW HAS FLATTENED before reading');
    console.log('the gap, because an unconverged tow is a smaller number for a reason that is not a');
    console.log('defect. The ANCHOR ERROR is the gate here; the Cd column is the evidence for M8.2c.');
  }
  console.log('\nCd on a towed body is not reproducible to better than the wake is: run the SAME');
  console.log('build twice before reading a small build-to-build move, exactly as CLAUDE.md');
  console.log('requires for AMR Cd and for the shedding Strouhal number.');
  process.exit(bad ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
