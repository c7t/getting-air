#!/usr/bin/env node
// Fit plans/performance-snapshot.md's ballpark cost model for an index-amr.html
// root step, per device, and cross-validate it. No GPU, no browser.
//
//   root step (ms) = a * passes + b * stepCells + c * coupledTiles
//
//   passes        compute passes per root step (passesPerRootStep below --
//                 the scheduler's structure, checked against
//                 tools/count-passes.js: 23.8 counted vs 24 here at the page
//                 defaults, 56.3 vs 56 at explode res5 L4)
//   stepCells     cells stepped per root step, each tile counted at its FULL
//                 FB^2 = 20x20 (the ring is stepped too), plus the root grid
//   coupledTiles  one per child tile per parent substep (explode + coalesce +
//                 average, or interp + average)
//
// Fitted by NON-NEGATIVE least squares (exact for three columns: every subset
// solved, the best all-non-negative one kept), then a leave-one-CONFIG-out
// pass -- the number that says how well it PREDICTS a configuration it was not
// fitted on, which is the only use it has.
//
//   node tools/fit-cost-model.js
//   node tools/fit-cost-model.js benchmarks/perf-snapshot-2026-09-24.json

const fs = require('fs');
const path = require('path');

const FILE = process.argv[2] || path.join(__dirname, '..', 'benchmarks', 'perf-snapshot-2026-09-24.json');
const FB2 = 400;

function passesPerRootStep(L, itf) {
  const sum = (a, b) => { let s = 0; for (let m = a; m <= b; m++) s += 2 ** m; return s; };
  const steps = sum(1, L - 1);                       // every pool level's substeps
  const body = 2 * 2 ** (L - 1);                     // force + body update, per finest substep
  const perParent = itf === 'explode' ? 3 : 2;       // explode+coalesce+average | interp+average
  const coupling = perParent * sum(1, L - 2) + perParent;   // pool parents, plus the root's
  return steps + body + coupling + (L - 1) + 1 + 1;  // + active lists + root step + ~1 refine-round pass
}
const features = (r) => ({
  passes: passesPerRootStep(r.levels, r.interface),
  stepCells: r.rootW * r.rootW + r.tiles.reduce((s, n, i) => s + n * 2 ** (i + 1) * FB2, 0),
  coupledTiles: r.tiles.reduce((s, n, i) => s + n * 2 ** i, 0),
});
const COLS = ['passes', 'stepCells', 'coupledTiles'];

// Least squares on the chosen columns, by normal equations (<= 3 unknowns).
function lsq(X, y, cols) {
  const k = cols.length, A = Array.from({ length: k }, () => new Array(k + 1).fill(0));
  for (let r = 0; r < X.length; r++) for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) A[i][j] += X[r][cols[i]] * X[r][cols[j]];
    A[i][k] += X[r][cols[i]] * y[r];
  }
  for (let i = 0; i < k; i++) {                       // Gauss-Jordan
    let p = i; for (let r = i + 1; r < k; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]];
    if (Math.abs(A[i][i]) < 1e-300) return null;
    for (let r = 0; r < k; r++) if (r !== i) { const f = A[r][i] / A[i][i]; for (let c = i; c <= k; c++) A[r][c] -= f * A[i][c]; }
  }
  const coef = new Array(X[0].length).fill(0);
  cols.forEach((c, i) => { coef[c] = A[i][k] / A[i][i]; });
  return coef;
}
function nnls(X, y) {
  let best = null;
  for (let mask = 1; mask < 1 << X[0].length; mask++) {
    const cols = [...Array(X[0].length).keys()].filter(i => mask & (1 << i));
    const c = lsq(X, y, cols);
    if (!c || c.some(v => v < 0)) continue;
    const sse = X.reduce((s, x, r) => s + (x.reduce((a, v, i) => a + v * c[i], 0) - y[r]) ** 2, 0);
    if (!best || sse < best.sse) best = { c, sse };
  }
  return best.c;
}
const predict = (x, c) => x.reduce((a, v, i) => a + v * c[i], 0);
const rms = (e) => Math.sqrt(e.reduce((s, v) => s + v * v, 0) / e.length);

const data = JSON.parse(fs.readFileSync(FILE, 'utf8'));
for (const dev of Object.keys(data.devices)) {
  const rows = data.amr.filter(r => r.ms[dev] != null);
  const X = rows.map(r => { const f = features(r); return COLS.map(k => f[k]); });
  const y = rows.map(r => r.ms[dev]);
  const c = nnls(X, y);
  const err = rows.map((r, i) => (predict(X[i], c) - y[i]) / y[i] * 100);
  const loo = [];
  for (const cfg of [...new Set(rows.map(r => r.config))]) {
    const tr = rows.map((r, i) => i).filter(i => rows[i].config !== cfg);
    const cc = nnls(tr.map(i => X[i]), tr.map(i => y[i]));
    rows.forEach((r, i) => { if (r.config === cfg) loo.push((predict(X[i], cc) - y[i]) / y[i] * 100); });
  }
  console.log(`\n${dev.toUpperCase()} (${data.devices[dev]})`);
  console.log(`  a = ${(c[0] * 1e3).toFixed(2)} us per pass    b = ${(c[1] * 1e6).toFixed(2)} ns per step cell    c = ${(c[2] * 1e3).toFixed(2)} us per coupled tile`);
  console.log(`  fit: rms ${rms(err).toFixed(1)}%  max ${Math.max(...err.map(Math.abs)).toFixed(1)}%    leave-one-config-out: rms ${rms(loo).toFixed(1)}%  max ${Math.max(...loo.map(Math.abs)).toFixed(1)}%`);
  rows.forEach((r, i) => {
    const f = features(r);
    console.log(`  ${r.config.padEnd(44)} t=${String(r.t).padStart(2)}  passes ${String(f.passes).padStart(2)}  cells ${String(f.stepCells).padStart(7)}  coupled ${String(f.coupledTiles).padStart(4)}`
      + `   measured ${y[i].toFixed(3)}  model ${predict(X[i], c).toFixed(3)}  ${err[i] >= 0 ? '+' : ''}${err[i].toFixed(1)}%`);
  });
  for (const f of data.flat) {
    if (f.ms[dev] == null) continue;
    const perCell = (f.ms[dev] - 3 * c[0]) / (f.W * f.W) * 1e6;
    console.log(`  flat ${f.W}^2: ${f.ms[dev]} ms/step (3 passes) -> ${perCell.toFixed(2)} ns per cell after this device's pass cost`);
  }
}
