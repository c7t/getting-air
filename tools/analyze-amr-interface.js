#!/usr/bin/env node
// The 2D coarse/fine interface diagnostic -- plans/2D-backport.md B0b.
// Sibling of tools/analyze-d3-interface.js, which is the 3D version and which
// is what found 3D's own interface bug.
//
// WHAT IT ANSWERS. 2D's coarse/fine interface is interp (ring ghosts) plus
// average (restriction), with NO flux correction: a coarse cell at the seam
// streams from its own coarse neighbours while the fine tile streams from its
// ring, so mass and momentum are not conserved across it -- and until this
// tool, NOTHING in the 2D suite measured either. Cd/St cannot: they are
// time-averaged surface integrals dominated by the near-body region, and
// CLAUDE.md already records them passing a configuration that misses every
// analytic channel tolerance by 5-20x.
//
// WHY TGV. `index-tgv-amr.html` is periodic, has no body, no sponge and no
// walls, so EACH LEVEL ALONE conserves mass and momentum exactly: streaming
// permutes populations and BGK collision preserves the first two moments.
// Any drift is therefore the interface -- or, as it turns out, the lattice
// (see the FLOOR below). The measurement itself is one readback of the dense
// L0 grid; amr2d-gpu.mjs's readConservedTotals explains why that is the WHOLE
// hybrid system's total and not just the coarse level's.
//
// THE TWO CONTROLS RUN IN THE SAME INVOCATION, deliberately, rather than
// being remembered from another session:
//
//   none   nothing refined. The pool, the criterion and the manager all still
//          exist; there is simply no seam. The dense solver's own floor.
//   all    everything refined. Every seam is gone the other way -- interp
//          writes rings nobody reads, and average's restriction is consumed
//          by a coarse step whose result is then discarded. This is NOT a
//          redundant control: it is the one that separates "the interface" from
//          "everything else the AMR path does", and it costs a different
//          amount than `none` for a reason the report prints.
//   half   a real seam. See POOL EXHAUSTION below -- this rung is currently
//          produced by capping the pool, which is honest but not a declared
//          geometry, and plans/2D-backport.md B6 is where it should become one.
//
// THE FLOOR IS NOT ZERO, AND IT IS NOT THE INTERFACE. The f32 D2Q9 weights in
// shaders/common_lattice.wgsl sum to 1 + 1.49e-8, so every collision injects
// `rho * eps / tau` of mass per cell per step (plans/2D-backport.md B9). This
// tool parses those weights out of the shader that actually ran and prints the
// resulting prediction beside the measurement, because on the mass channel it
// is 92% of what you see. Measured 2026-09-14, N=128, tau=0.8, 256 steps:
//
//     rung   d mass    per cell/step   predicted (B9)   ratio
//     none   7.80e-2      1.861e-8        1.863e-8      0.999
//     all    1.136e-1     2.709e-8        2.709e-8      1.000
//
// `all` costs 1.4545x `none` because a refined cell collides TWICE per
// macro-step at tau_1 = 2*tau_0 - 1/2 instead of once at tau_0, and
// 2/1.1 / (1/0.8) = 1.4545. Predicted 1.4545, measured 1.4557. So the mass
// channel currently measures the LATTICE, at 0.1%, on both paths -- and an
// interface mass claim is not available until B9 lands.
//
// THE MOMENTUM CHANNEL IS CLEAN TODAY, and it is the one that matters for
// plans/2D-backport.md B1. The weight excess is isotropic, so it injects mass
// with no net momentum; and `fneq` has no zeroth or first moment, so a wrong
// Dupuis-Chopard rescale -- which 2D has, applying the pre-collision factor to
// post-collision populations -- cannot perturb mass at all while corrupting
// the viscous stress, which IS the momentum flux. The two defects are
// therefore orthogonal and separable by WHICH CHANNEL MOVES:
//
//     mass drifts, momentum holds   ->  the lattice weights (B9)
//     momentum drifts, mass holds   ->  the non-equilibrium coupling (B1)
//
// POOL EXHAUSTION IS NOT A GEOMETRY. The `half` rung refines 128 of 256
// blocks, and that 128 is `maxFineBlocks`, not an answer the criterion gave:
// slots are granted in blockID order, so the free list dries up part-way and
// the denied blocks form a band. It is a real seam and a reproducible one,
// which is why it is usable here, but it is emergent rather than declared and
// its shape is jagged rather than flat. A declared `?refine=` ladder (the 2D
// analogue of 3D's slab/bar/box) is what separates a correction bug from a
// convex corner, and B6 is where that is actually needed.
//
// REPORTS; IT DOES NOT PASS/FAIL. There is no literature value for how fast a
// seam should leak. The controls are what the seam run is read against.
//
//   node tools/analyze-amr-interface.js
//   node tools/analyze-amr-interface.js --res=8 --checkpoints=64,128,256,512
//   node tools/analyze-amr-interface.js --configs=none,half

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');
const { analyticVelocity } = require('./lib/tgv-metrics');

// A page that threw during init still answers debugStepSync with a stale
// global, so a run has to be told the page was healthy rather than assume it.
// The 3D branch has a shared attachPageWatch/assertPageHealthy in
// tools/lib/browser-lifecycle.js; it is 2D-agnostic and worth bringing back,
// but pulling it in here would widen this change, so this is the local
// minimum until then.
function attachPageWatch(client) {
  const errors = [];
  client.Runtime.exceptionThrown(({ exceptionDetails: e }) => {
    errors.push({ kind: 'exception', text: (e.exception && e.exception.description) || e.text });
  });
  client.Runtime.consoleAPICalled(({ type, args }) => {
    if (type !== 'error') return;
    errors.push({ kind: 'console', text: args.map(a => a.value ?? a.description ?? '').join(' ') });
  });
  return errors;
}

async function assertPageHealthy(Runtime, errors, where) {
  const status = await evalOrThrow(Runtime,
    "document.getElementById('status') ? document.getElementById('status').textContent : ''",
    20000, 'status');
  if (/^error:/i.test(status.trim())) throw new Error(`${where}: page reports "${status.trim()}"`);
  if (errors.length) {
    const first = errors[0];
    throw new Error(`${where}: page logged ${errors.length} error(s), first: [${first.kind}] ${first.text.split('\n')[0]}`);
  }
}

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  res: 7, tau: 0.8, u0: 0.04,
  // STEPS, on a log ladder: the injection is per-step and shows up long
  // before the flow decays, and a ladder is what separates "linear in t" from
  // "sqrt(t)" -- a constant source from a diffusing one.
  //
  // THESE ARE REQUESTS, NOT COUNTS. debugStepSync(n) runs in STEPS_PER_FRAME
  // (64) batches and ROUNDS UP -- `for (k = 0; k < n; k += SPF)` with SPF
  // dispatches inside -- so asking for 16 runs 64. Every number below is
  // normalized by the step count the PAGE reports, never by the request. A
  // first version of this tool did use the request, and a ladder of deltas
  // 16/16/32/64/128 actually ran 384 steps while being labelled 256: the mass
  // channel came out at exactly 1.500x its predicted floor on every rung,
  // which reads like a real 50% excess and is entirely the ladder.
  checkpoints: [64, 128, 256, 512],
  // Macro-steps to run BEFORE freezing the topology, so the criterion has
  // settled and the seam stops moving. Everything after this is measured on a
  // fixed set of tiles, which is what makes the drift attributable to the
  // interface rather than to tile churn (2D has no drain pass; a tile freed
  // mid-measurement takes its solution with it).
  settle: 64,
  configs: null, extra: '', timeout: 900, keepOpen: false,
  // The `half` rung's pool cap -- 96, NOT 128, and the difference is the whole
  // instrument (plans/2D-backport.md B6-0, 2026-09-23).
  //
  // Under the quad manager (U5-4 on) a capped pool refines a FLAT full-width
  // band from row 0. At 128 of 256 that band is exactly HALF the domain, so its
  // two seams sit half a Taylor-Green period apart -- where the flow is
  // negated and the seams face opposite ways -- and their leaks CANCEL in the
  // global sum, which is all this tool reads. It measured 7.8e-5 there (2.2x
  // floor) and read as "the seam leak is fixed"; at 96 the same build reads
  // 3.2e-3 (89x) and at 64 6.6e-3 (184x), both linear in t. Nothing was fixed:
  // U5-4 turned the old jagged band into a symmetric flat one. A global
  // conservation sum is blind to any geometry with that symmetry, so do not
  // pick a cap that makes the refined fraction one half.
  halfCap: 96,
};

// thresh -6 is the page's own default and refines nothing on TGV; -11 is past
// the point where every block wants a child, so the cap is what decides.
const CONFIGS = [
  { name: 'none', thresh: -6, maxFineBlocks: 512 },
  { name: 'half', thresh: -11, maxFineBlocks: 96 },  // overridden by --halfCap; see halfCap
  { name: 'all', thresh: -11, maxFineBlocks: 512 },
];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--res=')) o.res = parseInt(a.slice(6));
    else if (a.startsWith('--tau=')) o.tau = parseFloat(a.slice(6));
    else if (a.startsWith('--u0=')) o.u0 = parseFloat(a.slice(5));
    else if (a.startsWith('--settle=')) o.settle = parseInt(a.slice(9));
    else if (a.startsWith('--checkpoints=')) o.checkpoints = a.slice(14).split(',').map(Number);
    else if (a.startsWith('--configs=')) o.configs = a.slice(10).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8);
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a.startsWith('--halfCap=')) o.halfCap = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument ${a}`); process.exit(2); }
  }
  return o;
}

// THE WEIGHT FLOOR, PARSED OUT OF THE SHADER THAT ACTUALLY RAN -- not typed
// here. tools/test-lattice-3d.js does the same thing for D3Q19/D3Q27, and for
// the same reason: a constant restated in the tool is a constant that can
// disagree with the one under test. When B9 lands, this number moves on its
// own and the report follows.
function latticeWeightExcess() {
  const src = fs.readFileSync(path.join(REPO_ROOT, 'shaders', 'common_lattice.wgsl'), 'utf8');
  const m = src.match(/const\s+wt\s*=\s*array<f32,\s*9>\(([^)]*)\)/);
  if (!m) throw new Error('could not find the wt array in shaders/common_lattice.wgsl');
  const w = m[1].split(',').map(s => parseFloat(s.trim().replace(/f$/, '')));
  if (w.length !== 9 || w.some(v => !isFinite(v))) throw new Error(`parsed ${w.length} weights: ${w}`);
  // The f32 values the GPU holds, summed in exact (f64) arithmetic -- which is
  // what the injected mass per collision is proportional to.
  return w.map(Math.fround).reduce((a, b) => a + b, 0) - 1;
}

const e3 = (x) => (x == null || !isFinite(x) ? '    -    ' : x.toExponential(3));
const padL = (s, n) => String(s).padStart(n);

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

function urlFor(o, c) {
  const q = [
    `res=${o.res}`, 'levels=2', `tau=${o.tau}`, `u0=${o.u0}`,
    `refineThresh=${c.thresh}`, `coarsenThresh=${c.thresh - 1}`,
    `maxFineBlocks=${c.maxFineBlocks}`,
  ];
  if (o.extra) q.push(o.extra);
  return `${o.baseUrl}/index-tgv-amr.html?${q.join('&')}`;
}

// THE FIELD CHANNEL (plans/2D-backport.md B6-1). Conservation is necessary
// and NOT sufficient -- 3D's M4.1b was exactly conservative while carrying an
// 8x field defect at the seam -- so every checkpoint also scores the root's
// velocity against the EXACT Taylor-Green solution, whole-domain and bucketed
// by distance to the seam. The seam is taken from the FROZEN level-1 tile set:
// a root cell is covered when its level-1 block holds a tile, and the seam is
// every cell with a 4-neighbour of the other kind. Distance is Chebyshev
// (8-neighbour BFS), periodic. A rung with no seam puts every cell in `far`.
const FIELD_BUCKETS = [[0, 1, 'd0-1'], [2, 4, 'd2-4'], [5, 8, 'd5-8'], [9, Infinity, 'far']];

function seamDistance(N, RB, blocks) {
  const cov = new Uint8Array(N * N);
  const on = new Set(blocks.map(b => `${b.bx},${b.by}`));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (on.has(`${Math.floor(x / RB)},${Math.floor(y / RB)}`)) cov[y * N + x] = 1;
  }
  const d = new Float64Array(N * N).fill(Infinity);
  const q = [];
  const w = (a) => ((a % N) + N) % N;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const c = cov[y * N + x];
    if (c !== cov[y * N + w(x + 1)] || c !== cov[y * N + w(x - 1)]
     || c !== cov[w(y + 1) * N + x] || c !== cov[w(y - 1) * N + x]) { d[y * N + x] = 0; q.push(y * N + x); }
  }
  for (let h = 0; h < q.length; h++) {
    const i = q[h], x = i % N, y = (i - x) / N;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const j = w(y + dy) * N + w(x + dx);
      if (d[j] > d[i] + 1) { d[j] = d[i] + 1; q.push(j); }
    }
  }
  return d;
}

function fieldError(field, analytic, dist) {
  const out = {};
  const acc = FIELD_BUCKETS.map(() => ({ e: 0, a: 0, n: 0 }));
  let E = 0, A = 0;
  for (let i = 0; i < field.ux.length; i++) {
    const dux = field.ux[i] - analytic.ux[i], duy = field.uy[i] - analytic.uy[i];
    const e = dux * dux + duy * duy, a = analytic.ux[i] ** 2 + analytic.uy[i] ** 2;
    E += e; A += a;
    const k = FIELD_BUCKETS.findIndex(([lo, hi]) => dist[i] >= lo && dist[i] <= hi);
    acc[k].e += e; acc[k].a += a; acc[k].n++;
  }
  out.whole = Math.sqrt(E / A);
  FIELD_BUCKETS.forEach(([, , name], k) => { out[name] = acc[k].n ? Math.sqrt(acc[k].e / Math.max(acc[k].a, 1e-30)) : null; });
  // AMPLITUDE vs SHAPE. Taylor-Green is one mode, so project the simulated
  // field onto it: `amp` = <sim, exact>/<exact, exact> - 1 is a decay-rate
  // (or amplitude) error, uniform in space by construction; `shape` is the
  // L2rel of what is left once that is removed -- a seam artifact, a phase
  // error, anything that is not the mode itself. A seam defect lives in
  // `shape` and in the near buckets; a wrong effective viscosity lives in `amp`.
  let sa = 0;
  for (let i = 0; i < field.ux.length; i++) sa += field.ux[i] * analytic.ux[i] + field.uy[i] * analytic.uy[i];
  const g = sa / A;
  let R = 0;
  for (let i = 0; i < field.ux.length; i++) {
    R += (field.ux[i] - g * analytic.ux[i]) ** 2 + (field.uy[i] - g * analytic.uy[i]) ** 2;
  }
  out.amp = g - 1;
  out.shape = Math.sqrt(R / A);
  return out;
}

async function runConfig(Runtime, o, c) {
  const G = 'window.__CYL';
  const T = o.timeout * 1000;
  await evalOrThrow(Runtime, `${G}.setLive(false)`, 20000, 'setLive');
  await evalOrThrow(Runtime, `${G}.reset()`, 60000, 'reset');

  // Settle, then FREEZE. After this the tile set does not change, so nothing
  // in the measurement is tile churn.
  await evalOrThrow(Runtime, `${G}.debugStepSync(${o.settle})`, T, 'settle');
  await evalOrThrow(Runtime, `${G}.setAutoRefine(false)`, 20000, 'setAutoRefine');

  const grid = await evalOrThrow(Runtime, `${G}.getBlockGridDims()`, 20000, 'getBlockGridDims');
  const params = await evalOrThrow(Runtime, `${G}.getParams()`, 20000, 'getParams');
  const blocksBefore = await evalOrThrow(Runtime, `${G}.debugListActiveBlocks(1)`, 60000, 'listActive');
  const activeBefore = blocksBefore.length;
  const dist = seamDistance(params.N, grid.RB, blocksBefore);
  const fieldAt = async (step) => {
    const field = await evalOrThrow(Runtime, `${G}.readField()`, 60000, 'readField');
    return fieldError(field, analyticVelocity(params.N, params.U0, params.nu, step), dist);
  };
  const nBlocks = grid.NBX * grid.NBY;

  // The PAGE's step counter at the freeze, which every sample is measured
  // against -- see the checkpoints comment on why the request is not it.
  const step0 = (await evalOrThrow(Runtime, `${G}.debugStepSync(0)`, T, 'step0')).step;

  const samples = [];
  samples.push({ t: 0, ...await evalOrThrow(Runtime, `${G}.debugConservedTotals()`, 180000, 'totals'),
                 field: await fieldAt(step0) });
  let prev = 0;
  for (const want of o.checkpoints) {
    if (want <= prev) continue;
    const r = await evalOrThrow(Runtime, `${G}.debugStepSync(${want - prev})`, T, 'debugStepSync');
    prev = want;
    samples.push({ t: r.step - step0, ...await evalOrThrow(Runtime, `${G}.debugConservedTotals()`, 180000, 'totals'),
                   field: await fieldAt(r.step) });
  }

  // The freeze has to have held, or every number above is about something
  // else. Checked rather than assumed.
  const activeAfter = (await evalOrThrow(Runtime, `${G}.debugListActiveBlocks(1)`, 60000, 'listActive')).length;
  return { params, grid, nBlocks, activeBefore, activeAfter, samples };
}

function report(o, results, eps) {
  const cells = results[0].samples[0].cells;
  const tau0 = o.tau, tau1 = 2 * tau0 - 0.5;
  const floorNone = eps / tau0;          // one coarse collision per macro-step
  const floorAll = 2 * (eps / tau1);     // two fine substeps, 4 children x 1/4 area

  console.log('\n' + '='.repeat(78));
  console.log('TOPOLOGY (frozen after the settle; `after` must equal `before`)');
  console.log('='.repeat(78));
  for (const r of results) {
    const capped = r.activeBefore === r.grid.MAX_FINE_BLOCKS;
    console.log(`  ${padL(r.name, 6)}  ${padL(r.activeBefore, 4)}/${r.nBlocks} level-1 blocks refined`
      + `  (after: ${r.activeAfter}${r.activeAfter !== r.activeBefore ? '  !! THE FREEZE DID NOT HOLD' : ''})`
      + (capped ? `  !! AT THE POOL CAP (${r.grid.MAX_FINE_BLOCKS}) -- emergent, not a declared geometry` : ''));
  }

  console.log('\n' + '='.repeat(78));
  console.log('CONSERVATION DRIFT from each run\'s own t=0 sample');
  console.log('  TGV is periodic and force-free, so each level alone conserves both');
  console.log('  exactly. Read the MOMENTUM channel against the two controls; the mass');
  console.log('  channel is dominated by the lattice-weight floor printed below.');
  console.log('='.repeat(78));
  const hdr = results.map(r => padL(r.name, 30)).join('');
  console.log(`  ${padL('step', 6)}${hdr}`);
  console.log(`  ${padL('', 6)}${results.map(() => padL('d mass', 10) + padL('d momX', 10) + padL('d momY', 10)).join('')}`);
  const ts = results[0].samples.map(s => s.t);
  for (let i = 0; i < ts.length; i++) {
    let line = `  ${padL(ts[i], 6)}`;
    for (const r of results) {
      const z = r.samples[0], s = r.samples[i];
      line += padL(e3(s.mass - z.mass), 10) + padL(e3(s.momX - z.momX), 10) + padL(e3(s.momY - z.momY), 10);
    }
    console.log(line);
  }

  const last = ts[ts.length - 1];   // the PAGE's own step count, not the request
  console.log('\n' + '='.repeat(78));
  console.log(`MASS, per cell per step, against the lattice-weight floor (B9)`);
  console.log(`  sum of the f32 weights in shaders/common_lattice.wgsl = 1 + ${e3(eps)}`);
  console.log(`  a collision at tau injects rho*eps/tau per cell per step, so`);
  console.log(`  none = eps/tau0 = ${e3(floorNone)},  all = 2*eps/tau1 = ${e3(floorAll)}`);
  console.log(`  (tau0 = ${tau0}, tau1 = 2*tau0 - 1/2 = ${tau1}; a refined cell collides TWICE`);
  console.log('   per macro-step at the finer tau, which is the whole of why `all` costs more)');
  console.log('='.repeat(78));
  console.log(`  ${padL('rung', 6)}${padL('refined', 9)}${padL('d mass', 11)}${padL('per cell/step', 15)}${padL('predicted', 12)}${padL('ratio', 8)}`);
  for (const r of results) {
    const z = r.samples[0], s = r.samples[r.samples.length - 1];
    const d = s.mass - z.mass;
    const per = d / (last * cells);
    const frac = r.activeBefore / r.nBlocks;
    const pred = (1 - frac) * floorNone + frac * floorAll;
    console.log(`  ${padL(r.name, 6)}${padL((100 * frac).toFixed(0) + '%', 9)}${padL(e3(d), 11)}`
      + `${padL(e3(per), 15)}${padL(e3(pred), 12)}${padL((per / pred).toFixed(3), 8)}`);
  }
  console.log('\n  A ratio of 1.000 means the mass channel is the LATTICE and nothing else.');
  console.log('  An excess on a rung with a seam, and only there, is the interface.');

  console.log('\n' + '='.repeat(78));
  console.log('MOMENTUM -- the channel the weights do not touch');
  console.log('='.repeat(78));
  const mom = (r, i) => {
    const z = r.samples[0], s = r.samples[i];
    return Math.max(Math.abs(s.momX - z.momX), Math.abs(s.momY - z.momY));
  };
  const ctl = results.filter(r => r.name === 'none' || r.name === 'all')
    .map(r => mom(r, r.samples.length - 1));
  const floor = ctl.length ? Math.max(...ctl) : null;

  // HOW IT GROWS IS THE DISCRIMINATOR, not how big it is. A least-squares fit
  // of log|d mom| against log(step) over the checkpoints: slope ~1 is a
  // CONSTANT per-step source at the seam, slope ~0.5 is a quantity that
  // diffuses away as fast as it arrives, slope ~0 is a one-off transient from
  // the freeze. Reported, not judged -- but a seam term and a settling
  // transient look nothing alike here and that is the point.
  const slope = (r) => {
    const pts = [];
    for (let i = 1; i < r.samples.length; i++) {
      const m = mom(r, i);
      if (m > 0 && r.samples[i].t > 0) pts.push([Math.log(r.samples[i].t), Math.log(m)]);
    }
    if (pts.length < 2) return null;
    const n = pts.length;
    const mx = pts.reduce((a, p) => a + p[0], 0) / n;
    const my = pts.reduce((a, p) => a + p[1], 0) / n;
    const num = pts.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0);
    const den = pts.reduce((a, p) => a + (p[0] - mx) ** 2, 0);
    return den === 0 ? null : num / den;
  };

  if (floor) console.log(`  control floor (max of the no-interface rungs): ${e3(floor)}`);
  for (const r of results) {
    const m = mom(r, r.samples.length - 1);
    const rel = floor ? ` = ${(m / floor).toFixed(2)}x floor` : '';
    const g = slope(r);
    const gs = g == null ? '' : `   growth ~ step^${g.toFixed(2)}`;
    console.log(`  ${padL(r.name, 6)}  max|d mom| ${e3(m)}${padL(rel, 16)}${gs}`);
  }
  console.log('\n  fneq has no zeroth or first moment, so the Dupuis-Chopard rescale (B1)');
  console.log('  cannot move mass and CAN move momentum. Momentum drifting while mass sits');
  console.log('  at its lattice floor is that defect; the reverse is the weights.');

  console.log('\n' + '='.repeat(78));
  console.log('FIELD ERROR vs the exact Taylor-Green solution (root velocity, L2rel)');
  console.log('  bucketed by Chebyshev distance to the seam, in root cells. Conservation');
  console.log('  does not imply consistency: read the seam buckets against `far` and');
  console.log('  against the no-interface rungs.');
  console.log('='.repeat(78));
  const names = ['whole', ...FIELD_BUCKETS.map(b => b[2]), 'amp', 'shape'];
  console.log(`  ${padL('rung', 6)}${padL('step', 7)}${names.map(n => padL(n, 11)).join('')}`);
  for (const r of results) {
    for (const smp of r.samples) {
      if (!smp.field) continue;
      console.log(`  ${padL(r.name, 6)}${padL(smp.t, 7)}${names.map(n => padL(e3(smp.field[n]), 11)).join('')}`);
    }
  }

  console.log('\n' + '='.repeat(78));
  console.log('FIELD HEALTH (a drift number from a diverging run means nothing)');
  console.log('='.repeat(78));
  for (const r of results) {
    const s = r.samples[r.samples.length - 1];
    console.log(`  ${padL(r.name, 6)}  rho in [${s.rhoMin.toFixed(6)}, ${s.rhoMax.toFixed(6)}]  max|u| ${s.maxU.toFixed(6)}`
      + `  (N=${r.params.N} tau=${r.params.TAU} u0=${r.params.U0})`);
  }
  console.log('\nReports only. There is no literature value for how fast a seam should leak;');
  console.log('the controls in this same run are what the seam rung is read against.\n');
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = CONFIGS.filter(c => !o.configs || o.configs.includes(c.name))
    .map(c => (c.name === 'half' ? { ...c, maxFineBlocks: o.halfCap } : c));
  if (!configs.length) { console.error('no configs selected'); process.exit(2); }
  const eps = latticeWeightExcess();

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = attachPageWatch(client);

  const results = [];
  try {
    for (const c of configs) {
      const url = urlFor(o, c);
      console.log(`\n=== ${c.name} (${url})`);
      await navigateTo(Page, url);
      await waitForGlobal(Runtime, 'window.__CYL', 60000);
      watch.length = 0;
      await assertPageHealthy(Runtime, watch, c.name);
      const res = await runConfig(Runtime, o, c);
      await assertPageHealthy(Runtime, watch, c.name);
      console.log(`    ${res.activeBefore}/${res.nBlocks} refined, frozen; `
        + `${o.checkpoints[o.checkpoints.length - 1]} steps measured`);
      results.push({ name: c.name, ...res });
    }
    report(o, results, eps);
  } finally {
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }
}

main().catch(e => { console.error('FAILED:', e.message); process.exit(1); });
