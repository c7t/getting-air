#!/usr/bin/env node
// THE FALLING CARD'S AIRFOIL, MEASURED: a lift/drag polar for the pinned
// ellipse (index-cylinder.html ?aspect= ?alpha=), per solid coupling.
//
// A PINNED body in uniform inflow, not a towed one: that removes every
// moving-body coupling question (plans/3D.md D1-D4) from the measurement, and
// it lets BOUNCE-BACK take part -- 2D has no moving bounce-back body at all,
// but the pinned cylinder has always run it (?bounceback).
//
// Defaults reproduce the falling card's own numerics: chord 32 cells (res 10,
// blockage 32, which is also 32 chords of domain), aspect 1/8, Re 1100 on the
// chord at u0 = 0.05 -- tau = 0.50436, index.html's value at res 8. So the
// diffuse kEps=1.5 row is the polar of the card as the shipped page resolves
// it, and the other rows say how far that is from converged:
//
//   diffuse kEps 1.5 / 0.75 / 0.375   the band ladder (first order, measured
//                                     on the cylinder -- CLAUDE.md)
//   bounceback                        the sharp body the band converges to
//
// Coefficients are per CHORD: Cd along the inflow, Cl across it, positive for
// positive alpha (main-cylinder.js states the sign). Averaged over a window
// after a transient, both in convective times (chord / u0); Cl's standard
// deviation over the window is printed as its unsteadiness.
//
//   node tools/probe-ellipse-polar.js
//   node tools/probe-ellipse-polar.js --alphas=0,10,20 --configs="bb:bounceback=1"
//   node tools/probe-ellipse-polar.js --re=100 --out=/tmp/polar.json

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const o = {
    baseUrl: 'https://localhost:4444', port: 9349,
    alphas: '0,5,10,15,20,30,45,60,90',
    configs: 'diffuse-1.5:kEps=1.5;diffuse-0.75:kEps=0.75;diffuse-0.375:kEps=0.375;bounceback:bounceback=1',
    res: 10, blockage: 32, re: 1100, u0: 0.05, aspect: 0.125, transient: 30, window: 30, out: '',
  };
  for (const a of argv) {
    const i = a.indexOf('='); const k = a.slice(2, i), v = a.slice(i + 1);
    if (!(k in o) || i < 0) { console.error(`unknown argument: ${a}`); process.exit(2); }
    o[k] = typeof o[k] === 'number' ? parseFloat(v) : v;
  }
  return o;
}

async function ev(R, e, t) {
  const r = await evalExpr(R, e, t);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
}

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const alphas = o.alphas.split(',').map(Number);
  const configs = o.configs.split(';').map(c => { const [name, q] = c.split(':'); return { name, q: q || '' }; });
  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  const G = 'window.__CYL';
  const results = [];

  try {
    for (const cfg of configs) {
      for (const alpha of alphas) {
        const url = `${o.baseUrl}/index-cylinder.html?res=${o.res}&blockage=${o.blockage}&re=${o.re}&u0=${o.u0}`
          + `&aspect=${o.aspect}&alpha=${alpha}${cfg.q ? '&' + cfg.q : ''}`;
        await navigateTo(Page, url);
        await waitForGlobal(Runtime, G, 60000);
        await ev(Runtime, `${G}.setLive(false)`);
        await ev(Runtime, `${G}.reset()`);
        const p = await ev(Runtime, `${G}.getParams()`);
        const tConv = p.D / p.U0;
        const nT = Math.ceil(o.transient * tConv / 64) * 64, nW = Math.ceil(o.window * tConv / 64) * 64;
        const t0 = Date.now();
        const { history } = await ev(Runtime, `${G}.debugRunAndCollect(${nT + nW})`, 3600000);
        const win = history.filter(r => r[0] > nT);
        const finite = history.every(r => r.every(Number.isFinite));
        // A NaN fluid reads fx = fy = 0 exactly (the force reduction's
        // containment), which would average to a perfectly plausible 0.
        const dead = !finite || win.slice(-4).every(r => r[1] === 0 && r[2] === 0);
        const cd = mean(win.map(r => r[3])), cl = mean(win.map(r => r[4])), clsd = sd(win.map(r => r[4]));
        const row = { config: cfg.name, alpha, cd, cl, ld: cl / cd, clsd, dead, tau: p.TAU, chord: p.D, secs: (Date.now() - t0) / 1000 };
        results.push(row);
        console.log(`  ${cfg.name.padEnd(14)} alpha ${String(alpha).padStart(4)}  Cd ${cd.toFixed(4)}  Cl ${cl.toFixed(4)}  L/D ${(cl / cd).toFixed(3).padStart(7)}`
          + `  Cl sd ${clsd.toFixed(4)}  (tau ${p.TAU.toFixed(5)}, chord ${p.D.toFixed(1)}, ${row.secs.toFixed(0)} s)${dead ? '  FLUID DIED -- read nothing' : ''}`);
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  console.log('\n  L/D by alpha:');
  console.log('    alpha  ' + configs.map(c => c.name.padStart(14)).join(''));
  for (const a of alphas) {
    console.log(`    ${String(a).padStart(5)}  ` + configs.map(c => {
      const r = results.find(x => x.config === c.name && x.alpha === a);
      return (r.dead ? 'died' : r.ld.toFixed(3)).padStart(14);
    }).join(''));
  }
  for (const c of configs) {
    const rows = results.filter(r => r.config === c.name && !r.dead);
    if (!rows.length) continue;
    const best = rows.reduce((b, r) => (r.ld > b.ld ? r : b));
    console.log(`  ${c.name}: max L/D ${best.ld.toFixed(3)} at alpha ${best.alpha} (Cl ${best.cl.toFixed(3)}, Cd ${best.cd.toFixed(3)})`);
  }
  if (o.out) { fs.writeFileSync(o.out, JSON.stringify({ args: o, results }, null, 2)); console.log(`\n  wrote ${o.out}`); }
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
