#!/usr/bin/env node
// HOW MUCH INERTIA DOES THE FALLING CARD REALLY HAVE?
//
// The diffuse (chi) coupling leaves fluid INSIDE the card, penalized to move
// with it: lbm_step.wgsl adds F = rho chi (u_body - u*) to the fluid and
// lbm_force.wgsl hands the body exactly -F, so the interior fluid's momentum
// is real and the dynamical body is card + entrained fluid. card-params.mjs
// puts only the card's own M = rho_b pi a b into Newton's law. The claim to
// test: the simulated inertia is M + rho_f pi a b, not M.
//
// THE EXPERIMENT: release the card from rest EDGE-ON (?theta0=pi/2), falling
// along its own chord. By symmetry there is no torque and no lateral force,
// and the OUTSIDE fluid's added mass for an ellipse moving along its major
// axis is only rho_f pi b^2 -- 1/8 of the interior's at aspect 1/8 -- so a
// large extra inertia can only be the interior. Newton:
//
//     a = W / (M + X),   W = M g_eff (net weight; buoyancy is in g_eff)
//     X = M (g_eff / a - 1)                    the extra mass, measured
//
//   configured    X ~ pi b^2             (outside added mass only)
//   entrained     X ~ pi a b + pi b^2    (plus the interior fluid)
//
// Reported as X / (pi a b). Two I* legs, because M changes with I* and X,
// if it is fluid, must not.
//
// EARLY-TIME FRICTION: the long faces grow a Stokes layer, so the drag goes
// as sqrt(t) and the velocity deficit as t^1.5. v(t) = alpha t - beta t^1.5
// is fitted over a window, and alpha is the drag-free acceleration. The raw
// windowed accelerations are printed too, so the fit can be checked by eye.
// Step 1 is a control with a known answer: the interior fluid starts at rest
// and so does the body, so the penalty force is zero and a = g_eff exactly.
//
//   node tools/probe-steer-inertia.js
//   node tools/probe-steer-inertia.js --istar=0.17,0.4,0.8 --res=9 --steps=320

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9349, istar: '0.17,0.4', res: 8, steps: 200, fitFrom: 10, kEps: 1.5, coupling: 'shipped', theta0: Math.PI / 2 };
  for (const a of argv) {
    const [k, v] = a.replace(/^--/, '').split('=');
    if (!(k in o) || v === undefined) { console.error(`unknown argument: ${a}`); process.exit(2); }
    o[k] = (k === 'baseUrl' || k === 'istar' || k === 'coupling') ? v : parseFloat(v);
  }
  return o;
}

async function ev(R, e, t) {
  const r = await evalExpr(R, e, t);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
}

// The ellipse's true signed distance, as common_geometry.wgsl's get_phi
// computes it (3 Newton steps on the closest-point parameter), and the chi
// blend the step and force kernels apply. Cell centres are at INTEGER buffer
// coordinates (lbm_step.wgsl: p = (bx, by)).
function phiEllipse(dx, dy, th, a, b) {
  const ca = Math.cos(th), sa = Math.sin(th);
  const lx = dx * ca + dy * sa, ly = -dx * sa + dy * ca;
  const x = Math.abs(lx), y = Math.abs(ly);
  const inside = (x * x) / (a * a) + (y * y) / (b * b) < 1;
  let t = Math.atan2(y * a, x * b);
  for (let i = 0; i < 3; i++) {
    const ct = Math.cos(t), st = Math.sin(t), ex = x - a * ct, ey = y - b * st;
    const F = ex * (-a * st) + ey * (b * ct);
    const Fp = -(a * a * st * st + b * b * ct * ct) + ex * (-a * ct) + ey * (-b * st);
    t = Math.min(Math.PI / 2, Math.max(0, t - F / (Math.abs(Fp) < 1e-9 ? -1e-9 : Fp)));
  }
  const d = Math.hypot(x - a * Math.cos(t), y - b * Math.sin(t));
  return inside ? -d : d;
}
const chiOf = (phi, eps) => 0.5 * (1 - Math.tanh(Math.max(-20, Math.min(20, phi / eps))));

// Least squares for v = alpha t - beta t^1.5 (no intercept: v(0) = 0).
function fitDrag(ts, vs) {
  let s11 = 0, s12 = 0, s22 = 0, r1 = 0, r2 = 0;
  for (let i = 0; i < ts.length; i++) {
    const p = ts[i], q = -Math.pow(ts[i], 1.5);
    s11 += p * p; s12 += p * q; s22 += q * q; r1 += p * vs[i]; r2 += q * vs[i];
  }
  const det = s11 * s22 - s12 * s12;
  const alpha = (r1 * s22 - r2 * s12) / det, beta = (s11 * r2 - s12 * r1) / det;
  let res = 0;
  for (let i = 0; i < ts.length; i++) res = Math.max(res, Math.abs(vs[i] - (alpha * ts[i] - beta * Math.pow(ts[i], 1.5))));
  return { alpha, beta, maxResid: res };
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

  try {
    for (const istar of o.istar.split(',').map(Number)) {
      await navigateTo(Page, `${o.baseUrl}/index-steer.html?res=${o.res}&istar=${istar}&theta0=${o.theta0}&startPaused=1&coupling=${o.coupling}`);
      await waitForGlobal(Runtime, L, 60000);
      if ((await ev(Runtime, `${L}.getStep()`)) !== 0) throw new Error('page stepped before the probe took over -- ?startPaused=1 not honoured');
      const { A, B } = await ev(Runtime, `${L}.getCardParams()`);
      const c0 = await ev(Runtime, `${L}.debugReadCardState()`);
      const M = c0[9], g = c0[11], V = Math.PI * A * B;
      const rhoB = M / V;
      // v[k] = velocity after k steps. vy is the fall (y down); vx and omega
      // must stay at zero by symmetry, and are reported as the check on that.
      const v = [0];
      let asym = 0;
      const { W, H } = await ev(Runtime, `${L}.getDims()`);
      const CHECK = new Set([2, 5, 20, 80, o.steps]);
      const field = [];
      for (let k = 1; k <= o.steps; k++) {
        await ev(Runtime, `${L}.debugStepSync(1)`, 60000);
        const c = await ev(Runtime, `${L}.debugReadCardState()`);
        v.push(c[4]);
        if (CHECK.has(k)) {
          // THE DIRECT READOUT. Is the fluid under the card moving with it,
          // and where has the momentum the card lost gone? rho ~ 1 to
          // O(Ma^2), so momentum is read as velocity.
          const vel = await ev(Runtime, `${L}.debugReadVelocity()`, 60000);
          let inN = 0, inU = 0, chiM = 0, chiP = 0, totP = 0;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const uy = vel[(y * W + x) * 2 + 1];
            totP += uy;
            let dx = x - c[0], dy = y - c[1];
            dx -= W * Math.round(dx / W); dy -= H * Math.round(dy / H);
            if (Math.abs(dx) > A + 8 || Math.abs(dy) > A + 8) continue;
            const phi = phiEllipse(dx, dy, c[2], A, B);
            if (phi < 0) { inN++; inU += uy; }
            const chi = chiOf(phi, o.kEps);
            chiM += chi; chiP += chi * uy;
          }
          // Impulse the body has handed the fluid: W t - M v.
          const J = M * g * k - M * c[4];
          field.push({ k, vb: c[4], inRatio: inU / inN / c[4], inN, chiM, chiP, totP, J });
        }
        asym = Math.max(asym, Math.abs(c[3]) / Math.max(1e-12, Math.abs(c[4])), Math.abs(c[5]) * A / Math.max(1e-12, Math.abs(c[4])));
      }
      const X = (a) => M * (g / a - 1);
      console.log(`\n  [${o.coupling}] I* ${istar}  res ${o.res}: a=${A} b=${B}  rho_b ${rhoB.toFixed(3)}  M ${M.toFixed(1)}  pi*a*b ${V.toFixed(1)}  g_eff ${g.toExponential(4)}`);
      console.log(`  symmetry: max |vx|/|vy| and |omega a|/|vy| over the run ${asym.toExponential(2)}`);
      console.log(`  prediction  configured X/(pi a b) = ${(B / A).toFixed(3)}   entrained X/(pi a b) = ${(1 + B / A).toFixed(3)}`);
      console.log('    window        a/g_eff    X/(pi a b)');
      for (const [k1, k2] of [[0, 1], [1, 2], [2, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, 160], [160, o.steps]]) {
        if (k2 > o.steps || k1 >= k2) continue;
        const a = (v[k2] - v[k1]) / (k2 - k1);
        console.log(`    ${String(k1).padStart(4)}-${String(k2).padEnd(4)}    ${(a / g).toFixed(4)}     ${(X(a) / V).toFixed(3)}`
          + (k1 === 1 ? `     <- per-step damping D = M (1 - a/g) = ${(M * (1 - a / g)).toFixed(1)}` : ''));
      }
      console.log('  THE FIELD: does the fluid under the card move with it?');
      console.log('    step   interior u/v_card   sum(chi)/(pi a b)   entrained sum(chi u)/(pi a b v)   fluid P / (W t - M v)   share of fluid P under chi');
      for (const f of field) {
        console.log(`    ${String(f.k).padStart(4)}       ${f.inRatio.toFixed(3)}              ${(f.chiM / V).toFixed(3)}                  ${(f.chiP / (V * f.vb)).toFixed(3)}                        ${(f.totP / f.J).toFixed(3)}                  ${(f.chiP / f.totP).toFixed(3)}`);
      }
      const ts = [], vs = [];
      for (let k = o.fitFrom; k <= o.steps; k++) { ts.push(k); vs.push(v[k]); }
      const f = fitDrag(ts, vs);
      console.log(`  fit v = alpha t - beta t^1.5 over steps ${o.fitFrom}-${o.steps}: alpha/g_eff ${(f.alpha / g).toFixed(4)}  `
        + `=> X/(pi a b) = ${(X(f.alpha) / V).toFixed(3)}   (max residual ${(f.maxResid / v[o.steps] * 100).toFixed(2)}% of final v)`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
