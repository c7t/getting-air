#!/usr/bin/env node
// probe-d3-volume-crunch.js -- WHERE A ?view=volume RENDER'S SPECKLE COMES
// FROM, measured in the VOLUME rather than in the picture.
//
// A live ?levels=3 card render shows "crunchy" dark stipple along the vortex
// tubes. The hypothesis this tool tests:
//
//   A level's volume is a dense grid over a BOUNDING BOX, but the refined set
//   inside that box is not box-shaped -- `boxRatio` is 1.04-1.14 even on a
//   geometry-forced shell -- and the tree sampler answers "the best level
//   available HERE". So voxels inside the box but outside the SET are filled
//   with coarser data replicated onto a finer grid. Q is a squared velocity
//   GRADIENT, so a replicated-coarse plateau sitting against real fine
//   variation differentiates into isolated speckle.
//
// WHY THE VOLUME AND NOT THE IMAGE. An image statistic cannot separate "the
// volume has speckle" from "the raymarcher undersamples a volume that is
// fine" -- both make crunch, and a picture shows their sum. The volume
// carries the discriminator directly: every voxel has a position,
// `debugSampleTree` reports what level that position actually resolves at,
// and the two populations can be compared against each other. (M6.5a is the
// standing lesson: three image statistics said "no effect" before an
// orthographic measurement said 4.7x.)
//
// THE STATISTIC IS A NORMALIZED LAPLACIAN, and the normalization is the
// point. Raw |lap Q| is largest wherever Q is largest, so an unnormalized
// version measures where the vortices are, not where the speckle is -- it
// would "find" crunch on the brightest tube in a perfectly clean volume.
// Dividing by the local magnitude asks the scale-free question instead: how
// big is this voxel's disagreement with its neighbours RELATIVE to the
// structure it sits in.
//
// Reports, does not PASS/FAIL -- there is no literature value for how much
// stipple is acceptable. Read the RATIO between the two populations.
//
//   node tools/probe-d3-volume-crunch.js
//   node tools/probe-d3-volume-crunch.js --steps=4000 --legs=full,flat
'use strict';

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, chromeDebugOk, reapAllChromes, waitFor,
  firstTab, navigateTo, evalExpr, waitForGlobal,
} = require(path.join(__dirname, 'lib', 'browser-lifecycle.js'));

const REPO = path.resolve(__dirname, '..');

// The configuration the artifact was seen in. `live=0` because this measures a
// STATE and not a frame rate: the crunch is a property of the volume at a
// step, and stepping synchronously makes that step reproducible.
const BASE = 'scenario=card&n=32&live=0&span=2&re=1100&tilt=1.047&aspect=0.1016'
  + '&levels=3&rb=4&refine=body&dynamic=1&manageEvery=8&margin=3&slotHeadroom=2.5'
  + '&volBudget=512&view=volume&vol=1&volIso=1.5&volGain=4&volOpacity=0.12';

// The three predictions the hypothesis makes, one leg each.
//   full    as filmed -- speckle expected, concentrated on the unrefined
//           voxels inside the box
//   static  ?dynamic=0: the refined set stops moving. If the speckle is the
//           set's ragged EDGE it is still there, merely frozen; if it is the
//           manager churning tiles it goes away. This separates "the box is
//           the wrong shape" from "newly-created tiles carry a transient".
//   flat    ?volstack=0: one L0 volume and no refined boxes at all, so there
//           is no box/set mismatch available to have. Speckle must go.
const LEGS = { full: '', static: 'dynamic=0', flat: 'volstack=0' };

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  steps: 2000, legs: ['full', 'static', 'flat'],
};

const HELP = `probe-d3-volume-crunch.js -- where the volume render's speckle comes from
  --steps=N     steps before measuring (default ${DEFAULTS.steps})
  --legs=a,b    ${Object.keys(LEGS).join(',')} (default all)

Each leg gets its OWN Chrome and tears it down again -- see measureLeg for why
that is not the usual one-tab-reused pattern. There is deliberately no
--keepOpen: a leftover Chrome here starves the next leg of GPU memory.`;

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--legs=')) o.legs = a.slice(7).split(',');
    else if (a === '--help' || a === '-h') { console.log(HELP); process.exit(0); }
    else { console.error(`unknown argument ${a}`); process.exit(2); }
  }
  for (const l of o.legs) {
    // `in`, not truthiness: the `full` leg's override is the EMPTY STRING
    // (it is BASE unmodified), and a falsy test rejects the default leg.
    if (!(l in LEGS)) { console.error(`unknown leg "${l}": ${Object.keys(LEGS).join(', ')}`); process.exit(2); }
  }
  return o;
}

// LATER WINS, so a leg can override a BASE parameter. URLSearchParams.get
// returns the FIRST occurrence, so appending is NOT overriding -- that trap
// cost a whole `levels=3` run which came back bit-identical to `levels=2`
// and was caught only because bit-identical is itself suspicious.
function mergeQuery(...parts) {
  const q = new URLSearchParams();
  for (const part of parts) {
    if (!part) continue;
    for (const [k, v] of new URLSearchParams(part)) q.set(k, v);
  }
  return q.toString();
}

// Runs IN THE PAGE. A 192x160 slice is ~30k voxels and each one needs a tree
// sample; shipping them out and the positions back in is two round trips per
// leg for data that reduces to four numbers.
const MEASURE = (steps) => `(async () => {
  const D = window.__D3;
  const out = { step: null, level: null, res: null, boxRatio: null, note: null };

  // The FINEST volume in the stack is where the mismatch bites hardest: it has
  // the most cells per L0 cell, so ONE unrefined L0 cell inside its box is
  // 4x4x4 replicated voxels rather than one.
  const probe0 = await D.debugReadVolume(0, 0, 'scl');
  if (probe0.skipped) return { note: probe0.skipped };
  const stack = probe0.info.stack;
  const which = stack - 1;

  const first = await D.debugReadVolume(0, which, 'scl');
  const [nx, ny, nz] = first.info.res;

  // THE SLICE IS CHOSEN, NOT ASSUMED TO BE THE MIDDLE. The box's centre is
  // only the set's centre for a roughly isotropic refined region; for a flat
  // plate the set is a thin slab inside a fat box, and a mid-box slice can
  // miss it entirely -- which is how the first run of this tool reported
  // "100% of voxels read a coarser level" and no at-level population to
  // compare them against. Pick the slice with the MOST at-level voxels, so
  // the comparison is made where both populations actually exist.
  // The score is voxels that are BOTH at the volume's own level AND carrying
  // structure. At-level alone degenerates on a single-level volume, where
  // every voxel qualifies and the scan happily returns the first slice it
  // tried -- against the domain edge, where there is nothing to measure.
  let best = { z: Math.floor(nz / 2), n: -1 };
  for (let zi = 2; zi < nz; zi += Math.max(1, Math.floor(nz / 12))) {
    const probe = await D.debugReadVolume(zi, which, 'scl');
    const pts = [], qs = [];
    for (let i = 0; i < probe.texels.length; i += 37) {
      if (pts.length >= 4096) break;
      pts.push(probe.texels[i].p); qs.push(probe.texels[i].v[0]);
    }
    const r = await D.debugSampleTree(pts);
    let n = 0;
    for (let i = 0; i < r.length; i++) {
      if (r[i].level >= first.info.level && Math.abs(qs[i]) >= 0.05) n++;
    }
    if (n > best.n) best = { z: zi, n };
  }
  out.sliceZ = best.z; out.sliceScan = best.n;
  const slice = await D.debugReadVolume(best.z, which, 'scl');
  out.step = D.getStep();
  out.level = slice.info.level;
  out.res = slice.info.res;
  out.which = which;
  out.stack = stack;

  // debugPoolState() spreads LEVEL 1 at the top and puts the rest in
  // byLevel, so the finest level's box has to be asked for by index -- the
  // top-level boxRatio is level 1's and would understate the mismatch at the
  // level that actually has one.
  const ps = await D.debugPoolState();
  const pl = ps && ps.byLevel ? ps.byLevel[slice.info.level - 1] : null;
  out.boxRatio = pl && pl.boxRatio != null ? +pl.boxRatio.toFixed(3) : null;
  out.inUse = pl ? pl.inUse : null;

  const t = slice.texels;
  const q = new Float64Array(nx * ny);
  for (let i = 0; i < t.length; i++) q[i] = t[i].v[0];   // channel 0 is Q/qRef

  // The level each voxel's position actually resolves at, batched to the
  // probe's own limit. This is the classifier, and it comes from the SHADER's
  // tree walk rather than from host arithmetic about where the box is -- the
  // question is what the resample actually read, not what it should have.
  const lvl = new Int32Array(nx * ny);
  const B = 4096;
  for (let s = 0; s < t.length; s += B) {
    const pts = [];
    for (let i = s; i < Math.min(s + B, t.length); i++) pts.push(t[i].p);
    const r = await D.debugSampleTree(pts);
    for (let i = 0; i < r.length; i++) lvl[s + i] = r[i].level;
  }

  // Normalized Laplacian -- see the header for why it is normalized. EPS is
  // the fp16 resolution the volume is stored at, so a voxel in genuinely empty
  // far field divides by its own storage noise instead of by zero and is
  // dropped by the magnitude floor below rather than reported as infinite
  // crunch.
  const EPS = 2 ** -14;
  const at = (x, y) => q[y * nx + x];
  const acc = { fine: [0, 0], coarse: [0, 0] };
  let nCoarse = 0, nIn = 0;
  for (let y = 1; y < ny - 1; y++) {
    for (let x = 1; x < nx - 1; x++) {
      const i = y * nx + x;
      const c = at(x, y);
      const n4 = at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1);
      const mag = Math.abs(c) + Math.abs(n4) / 4;
      // Only where there is structure to speckle. Empty far field is
      // uniformly zero and would dilute both populations equally, which
      // flatters the ratio toward 1 -- i.e. it would hide the effect.
      if (mag < 0.05) continue;
      nIn++;
      const lap = Math.abs(4 * c - n4) / (mag + EPS);
      const k = lvl[i] >= out.level ? 'fine' : 'coarse';
      if (k === 'coarse') nCoarse++;
      acc[k][0] += lap; acc[k][1]++;
    }
  }
  out.scored = nIn;
  out.coarseFrac = nIn ? +(nCoarse / nIn).toFixed(4) : null;
  out.crunchFine = acc.fine[1] ? +(acc.fine[0] / acc.fine[1]).toFixed(4) : null;
  out.crunchCoarse = acc.coarse[1] ? +(acc.coarse[0] / acc.coarse[1]).toFixed(4) : null;
  out.nFine = acc.fine[1]; out.nCoarse = acc.coarse[1];
  out.ratio = (out.crunchFine && out.crunchCoarse)
    ? +(out.crunchCoarse / out.crunchFine).toFixed(2) : null;
  return out;
})()`;

// A FRESH CHROME PER LEG, which is NOT this repo's usual pattern and is
// deliberate. Every other tool here reuses one tab across configs so that only
// one WebGPU context is alive at a time -- but `main-3d.js` has no unload
// handler and never calls `device.destroy()`, so a navigation does not
// synchronously free the outgoing page's GPU memory. That is harmless for a
// dense 64^3 config and NOT harmless here: this one is `?levels=3` with
// `slotHeadroom=2.5`, i.e. 4000 + 14400 pool slots plus a 289 MiB volume
// stack, and two of those resident at once is more than the card has. The
// GPU PROCESS dies, every page on it reports "no GPU adapter" until Chrome is
// restarted, and the tool's own error is the useless downstream
// "A valid external Instance reference no longer exists".
// REAP, don't teardown. `teardown` deliberately spares a Chrome it ADOPTED
// rather than started -- right for a suite whose runs may overlap, wrong here,
// where an adopted Chrome is the previous leg's and is holding the very memory
// this leg needs. `reapAllChromes` kills everything under the tools' own
// profile root and nothing else, which is what `make chrome-clean` does and is
// never able to touch the user's own browser.
async function freshChrome(o) {
  reapAllChromes();
  // The port outliving the process is the failure that matters: ensureChrome
  // would see it still answering, ADOPT the corpse, and the leg would run
  // against a Chrome that is on its way down.
  await waitFor(async () => !(await chromeDebugOk(o.port)), 20000, 250);
  await ensureChrome(o.port);
}

async function measureLeg(o, legName) {
  const url = `${o.baseUrl}/index-3d.html?${mergeQuery(BASE, LEGS[legName])}`;
  await freshChrome(o);
  const tab = await firstTab(o.port);
  const client = await CDP({ port: o.port, target: tab.id });
  await Promise.all([client.Page.enable(), client.Runtime.enable()]);
  await navigateTo(client.Page, url);
  await waitForGlobal(client.Runtime, 'window.__D3', 120000);

  const done = async (r) => {
    try { await client.close(); } catch { /* the leg's answer matters more */ }
    reapAllChromes();
    return r;
  };

  const st = await evalExpr(client.Runtime, 'document.getElementById("status").textContent');
  const txt = st.result && st.result.value;
  if (typeof txt === 'string' && txt.startsWith('error:')) return done({ legName, error: txt });

  // STEPPED IN SMALL BATCHES, NOT ONE CALL. `debugStepSync` submits in chunks
  // of 500 steps, and at ?levels=3 with the manager firing every 8 steps that
  // is ~62 topology events -- six passes each -- inside one command buffer.
  // The GPU process was observed dying inside exactly that call on the
  // `dynamic=1` legs while the `dynamic=0` leg walked through it, which is
  // what a watchdog timeout looks like from the outside. Smaller submits also
  // give the page room to report an `error:` we can attribute, instead of one
  // opaque "valid external Instance reference no longer exists".
  const BATCH = 200;
  for (let done_ = 0; done_ < o.steps; done_ += BATCH) {
    const n = Math.min(BATCH, o.steps - done_);
    const r0 = await evalExpr(client.Runtime, `window.__D3.debugStepSync(${n})`, 900000);
    if (r0.exceptionDetails) {
      return done({ legName, error: `at step ~${done_ + n}: ` + (r0.exceptionDetails.exception
        ? r0.exceptionDetails.exception.description : r0.exceptionDetails.text) });
    }
    const s0 = await evalExpr(client.Runtime, 'document.getElementById("status").textContent');
    const t0 = s0.result && s0.result.value;
    if (typeof t0 === 'string' && t0.startsWith('error:')) {
      return done({ legName, error: `at step ~${done_ + n}: ${t0}` });
    }
  }
  const r = await evalExpr(client.Runtime, MEASURE(o.steps), 900000);
  if (r.exceptionDetails) {
    return done({ legName, error: r.exceptionDetails.exception
      ? r.exceptionDetails.exception.description : r.exceptionDetails.text });
  }
  return done({ legName, url, ...r.result.value });
}

(async () => {
  const o = parseArgs(process.argv.slice(2));
  await ensureServer(o.baseUrl, REPO);

  const rows = [];
  for (const leg of o.legs) {
    process.stdout.write(`\n=== ${leg} (${o.steps} steps)\n`);
    const r = await measureLeg(o, leg);
    rows.push(r);
    if (r.error) { console.log(`    ERROR ${r.error.slice(0, 200)}`); continue; }
    if (r.note) { console.log(`    skipped: ${r.note}`); continue; }
    console.log(`    volume ${r.which} of ${r.stack}: level ${r.level}, ${r.res.join('x')}`
      + `   boxRatio ${r.boxRatio}  inUse ${r.inUse}`);
    console.log(`    scored ${r.scored} voxels with structure;`
      + ` ${(100 * r.coarseFrac).toFixed(1)}% of them read a COARSER level than the volume`);
    console.log(`    normalized |lap Q|:  at level ${r.crunchFine}   below level ${r.crunchCoarse}`
      + `   ratio ${r.ratio}`);
  }

  console.log('\n' + '='.repeat(92));
  console.log('SUMMARY');
  console.log('='.repeat(92));
  console.log('leg       level  res              boxRatio  coarse%   crunch@lvl  crunch<lvl  ratio');
  console.log('-'.repeat(92));
  for (const r of rows) {
    if (r.error || r.note) { console.log(`${r.legName.padEnd(9)} ${(r.error || r.note).slice(0, 70)}`); continue; }
    console.log(`${r.legName.padEnd(9)} ${String(r.level).padEnd(6)} ${r.res.join('x').padEnd(16)} `
      + `${String(r.boxRatio).padEnd(9)} ${(100 * r.coarseFrac).toFixed(1).padStart(6)}%  `
      + `${String(r.crunchFine).padStart(10)}  ${String(r.crunchCoarse).padStart(10)}  ${String(r.ratio).padStart(5)}`);
  }
  console.log(`
A ratio >> 1 says the speckle is the BOX/SET MISMATCH: voxels inside the volume's
bounding box that the refined set does not cover read a coarser level, get replicated
onto the fine grid, and differentiate into stipple. A ratio ~ 1 says the crunch is
uniform over the volume and the hypothesis is WRONG -- look at the raymarcher's step
size (?volStep=) or at fp16 quantization instead.

'static' separates the set's ragged EDGE (speckle persists, frozen) from the manager
churning tiles (speckle goes). 'flat' is the control: with no refined boxes there is no
mismatch to have, so a 'flat' leg that still crunches indicts the raymarcher, not the volume.`);

})().catch((e) => { console.error(e); process.exit(1); });
