// Shared analysis for the dense 3D solver's validation cases (plans/3D.md
// M1), factored out so tools/validate-3d.js and tools/validate-all.js run
// the identical comparison rather than two copies that drift -- the same
// arrangement as tools/lib/tgv-metrics.js and tools/lib/channel-metrics.js.
//
// The reference solutions themselves are NOT here: they live in
// d3-scenarios.mjs, which the PAGE also imports to build its initial
// conditions. One definition of "what this flow is", used by both sides.
// d3-scenarios.mjs is an ES module and these tools are CommonJS, hence the
// dynamic import (same pattern as tools/test-f-pack.js).
//
// A note on how each gate resists being fooled, because a validation
// harness that shares a module with the thing it validates deserves the
// question:
//
//   duct      starts from REST. The initial condition carries no
//             information about the answer at all, so a wrong series
//             solution cannot be cancelled by a matching wrong seed. The
//             series is separately checked against the PDE it solves by
//             tools/test-d3-scenarios.js (an independent SOR solve,
//             verified to converge to it at second order).
//   beltrami  IS seeded from its own reference, so the t=0 comparison is
//             vacuous by construction -- which is why nothing is scored at
//             t=0. What is scored is the field several decay times later
//             and the fitted decay rate, neither of which the seed
//             determines. tools/test-d3-scenarios.js separately checks that
//             the field really is Beltrami (curl(u) = k*u by finite
//             differences), which is the property that makes the analytic
//             decay exact in the first place.

const path = require('path');

let _S = null;
async function scenarios() {
  if (!_S) _S = await import(path.join(__dirname, '..', '..', 'd3-scenarios.mjs'));
  return _S;
}

function evalExpr(Runtime, expr, timeoutMs) {
  return Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs || 60000 });
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

function checkTol(label, measured, target, tol) {
  if (tol == null) return { pass: true, label, measured, target: null };
  const pass = measured != null && Math.abs(measured - target) <= tol;
  return { pass, label, measured, target, tol };
}

// URL for a case. Every case drives the SAME page -- index-3d.html with a
// ?scenario= -- which is the whole point of main-3d.js not being forked per
// scenario: the gates exercise the code path the interactive page uses.
function caseUrl(baseUrl, scenario, c, extra) {
  const p = new URLSearchParams({ scenario, n: c.n, tau: c.tau, u0: c.u0, q: c.q || 19, live: '0' });
  return `${baseUrl}/index-3d.html?${p}${extra ? `&${extra}` : ''}`;
}

// --- duct ------------------------------------------------------------------

async function runDuctCase(Runtime, opts, c, log) {
  const S = await scenarios();
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  const steps = Math.round((c.settle_multiple || 12) * p.settle);
  if (log) log(`N=${p.N} tau=${p.tau} Q${p.Q}: nu=${p.nu.toFixed(4)} G=${p.G.toExponential(3)} settle=${p.settle.toFixed(0)} -> ${steps} steps`);

  await evalOrThrow(Runtime, `window.__D3.debugStepSync(${steps})`, ((opts.timeout || 300) + 30) * 1000, 'debugStepSync');
  const r = await evalOrThrow(Runtime, 'window.__D3.readDuctProfile()', 120000, 'readDuctProfile');

  const exact = S.ductProfile(p.N, p.G, p.nu);
  let num = 0, den = 0, peakSim = -Infinity, peakExact = -Infinity;
  for (let i = 0; i < p.N * p.N; i++) {
    const d = r.profile[i] - exact[i];
    num += d * d; den += exact[i] * exact[i];
    if (r.profile[i] > peakSim) peakSim = r.profile[i];
    if (exact[i] > peakExact) peakExact = exact[i];
  }
  const l2rel = Math.sqrt(num / Math.max(den, 1e-30));
  const peakRelErr = (peakSim - peakExact) / peakExact;
  // Relative to the peak, so it is a fraction of the flow rather than an
  // absolute velocity that would have to be re-judged per case.
  const xSpreadRel = r.maxXSpread / peakExact;
  if (log) log(`  L2rel=${l2rel.toExponential(3)}  peakRelErr=${peakRelErr.toExponential(3)}  xSpread=${xSpreadRel.toExponential(2)}`);

  return {
    N: p.N, tau: p.tau, Q: p.Q, steps, l2rel, peakSim, peakExact, peakRelErr, xSpreadRel,
    fieldCheck: checkTol('fieldL2rel', l2rel, 0, c.field_l2_tol),
    peakCheck: checkTol('peakRelErr', peakRelErr, 0, c.peak_tol),
    xCheck: checkTol('xSpreadRel', xSpreadRel, 0, c.x_spread_tol),
  };
}

// --- beltrami --------------------------------------------------------------

// L2 of (sim - exact) over the strided sample, and the sample's RMS speed.
// Both are computed over the SAME points, so the decay rate fitted from the
// RMS is unaffected by the subsampling.
async function beltramiSample(Runtime, S, p, t) {
  const f = await evalOrThrow(Runtime, 'window.__D3.readSubsampled(24)', 120000, 'readSubsampled');
  let num = 0, den = 0, sq = 0, i = 0;
  for (let z = 0; z < p.N; z += f.stride) {
    for (let y = 0; y < p.N; y += f.stride) {
      for (let x = 0; x < p.N; x += f.stride) {
        const e = S.beltramiVelocityAt(x, y, z, p.N, p.u0, p.nu, t);
        const dx = f.ux[i] - e[0], dy = f.uy[i] - e[1], dz = f.uz[i] - e[2];
        num += dx * dx + dy * dy + dz * dz;
        den += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        sq += f.ux[i] * f.ux[i] + f.uy[i] * f.uy[i] + f.uz[i] * f.uz[i];
        i++;
      }
    }
  }
  return { l2rel: Math.sqrt(num / Math.max(den, 1e-30)), rms: Math.sqrt(sq / i), points: i, stride: f.stride };
}

async function runBeltramiCase(Runtime, opts, c, log) {
  const S = await scenarios();
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  if (log) log(`N=${p.N} tau=${p.tau} u0=${p.u0} Q${p.Q}: nu=${p.nu.toFixed(4)} td=${p.td.toFixed(0)} steps`);

  // NOTHING is scored at t=0: the seed IS the reference there, so a t=0
  // check would pass no matter what the solver does. Checkpoints start at
  // half a decay time.
  const targets = c.checkpoints.map(m => Math.round(m * p.td));
  const samples = [];
  let prev = 0;
  for (const t of targets) {
    await evalOrThrow(Runtime, `window.__D3.debugStepSync(${t - prev})`, ((opts.timeout || 300) + 30) * 1000, 'debugStepSync');
    prev = t;
    const s = await beltramiSample(Runtime, S, p, t);
    samples.push({ t, ...s });
    if (log) log(`  t=${t} (${(t / p.td).toFixed(1)} td): L2rel=${s.l2rel.toExponential(3)}`);
  }

  const maxL2rel = Math.max(...samples.map(s => s.l2rel));
  const [a, b] = samples.slice(-2);
  const rateMeasured = -Math.log(b.rms / a.rms) / (b.t - a.t);
  const rateAnalytic = 1 / p.td;
  const rateRelErr = (rateMeasured - rateAnalytic) / rateAnalytic;
  if (log) log(`  decay rate ${rateMeasured.toExponential(4)} vs ${rateAnalytic.toExponential(4)} (rel ${rateRelErr.toExponential(3)})`);

  return {
    N: p.N, tau: p.tau, u0: p.u0, Q: p.Q, td: p.td, samples, maxL2rel,
    rateMeasured, rateAnalytic, rateRelErr,
    fieldCheck: checkTol('fieldL2rel', maxL2rel, 0, c.field_l2_tol),
    rateCheck: checkTol('decayRateRelErr', rateRelErr, 0, c.decay_rate_tol),
  };
}

// --- tgv (reports, does not gate) -----------------------------------------

async function runTgvReport(Runtime, opts, c, log) {
  const p = await evalOrThrow(Runtime, 'window.__D3.getParams()', 20000, 'getParams');
  if (log) log(`N=${p.N} tau=${p.tau} u0=${p.u0} Q${p.Q}: Re=${p.re.toFixed(0)} (L = N/2pi)`);
  const samples = [];
  let prev = 0;
  for (const t of c.checkpoints) {
    await evalOrThrow(Runtime, `window.__D3.debugStepSync(${t - prev})`, ((opts.timeout || 300) + 30) * 1000, 'debugStepSync');
    prev = t;
    const s = await evalOrThrow(Runtime, 'window.__D3.readStats()', 120000, 'readStats');
    samples.push(s);
    if (log) log(`  t=${t}: ke=${s.ke.toExponential(4)} enstrophy=${s.enstrophy.toExponential(4)} eps=${s.dissipation.toExponential(4)}`);
  }
  // The ONLY thing that can fail here: a field that stopped being finite.
  // Everything else is a report -- see benchmarks/d3.json on why.
  const finite = samples.every(s => s.finite);
  return { N: p.N, tau: p.tau, Q: p.Q, re: p.re, samples, finite, finiteCheck: { pass: finite, label: 'finite', measured: finite } };
}

module.exports = { evalExpr, checkTol, caseUrl, runDuctCase, runBeltramiCase, runTgvReport };
