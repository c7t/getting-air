#!/usr/bin/env node
// Controlled GPU benchmark for the AMR solver.
//
// WHY THIS EXISTS, and why the obvious approach does not work
//
// The natural way to compare two configurations is to load each, let it run,
// and read the overlay's GPU-ms. That is unusable below about 20%: the card's
// trajectory diverges between runs, so the two configurations are not doing
// the same amount of work. Measured directly -- the SAME config, separate
// page loads, gave 10.08, 11.21, 12.06 and 12.61 ms. Any "improvement"
// smaller than that spread read off two such runs is noise, and I published
// one before catching it.
//
// This instead: identical warm-up, then FREEZE refinement (setAutoRefine
// false) so the block topology cannot drift, then time a fixed number of
// macro-steps of pure GPU work with the frame loop stopped. Repeatability
// within one page load is ~0.5%.
//
// A CAVEAT THAT REMAINS, and it is not small. Refinement allocation is
// nondeterministic: blocks race on `atomicSub(&freeCount, 1)` in
// amr_manage.wgsl, so which blocks win slots differs run to run even with
// identical code and inputs. Two page loads therefore freeze at different
// active-block counts (measured 148 vs 162 for the same config), and that
// is a real workload difference, not measurement error. So:
//
//   - `activeL1` is printed with every result. Compare it FIRST. If it
//     differs materially between configurations, the timings are not
//     comparable and no amount of repetition inside a run will fix that.
//   - For a cross-config A/B, run each several times and compare
//     distributions, or use ?bench=1's in-page pass-skip sweep instead,
//     which varies only the dispatch list within a single frozen topology.
//
// Also reports honest cell-updates/s: L0 cells plus, per level, active
// blocks x FB^2 x 2^m substeps. The overlay's own MLUPS counts L0 cells
// only and is not comparable across level counts.
//
// Usage:
//   node tools/bench-amr.js --warm=20000 --steps=20000 --reps=5 \
//        'res=8&levels=3&blockage=3.3' 'res=8&levels=3&blockage=3.3&sdfFar=1e9'

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const REPO_ROOT = path.join(__dirname, '..');
const BL = require('./lib/browser-lifecycle');

const BASE_URL = 'https://localhost:4444';
const PORT = 9333;

function parseArgs(argv) {
  const o = { warm: 20000, steps: 20000, reps: 5, page: 'index-amr.html', configs: [], keepOpen: false };
  for (const a of argv) {
    if (a.startsWith('--warm=')) o.warm = Number(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = Number(a.slice(8));
    else if (a.startsWith('--reps=')) o.reps = Number(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    else if (a === '--keepOpen') o.keepOpen = true;
    else if (a === '--help' || a === '-h') { printHelp(); process.exit(0); }
    else if (a.startsWith('--')) { console.error(`unknown flag ${a}`); process.exit(2); }
    else o.configs.push(a);
  }
  if (!o.configs.length) { printHelp(); process.exit(2); }
  return o;
}

function printHelp() {
  console.log(require('fs').readFileSync(__filename, 'utf8').split('\n')
    .filter(l => l.startsWith('//')).map(l => l.replace(/^\/\/ ?/, '')).join('\n'));
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const server = await BL.ensureServer(BASE_URL, REPO_ROOT);
  const chrome = await BL.ensureChrome(PORT);
  const tabId = await BL.openTab(PORT, 'about:blank');
  const client = await CDP({ port: PORT, target: tabId });
  const { Page, Runtime } = client;
  await Page.enable(); await Runtime.enable();
  const ev = async (expr, t) => {
    const r = await BL.evalExpr(Runtime, expr, t || 900000);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
    return r.result.value;
  };

  const rows = [];
  try {
    for (const cfg of opts.configs) {
      await BL.navigateTo(Page, `${BASE_URL}/${opts.page}?${cfg}`);
      await BL.waitForGlobal(Runtime, 'window.__AMR', 180000);
      await ev('window.__AMR.setLive(false)');
      await ev(`(async()=>{ await window.__AMR.debugStepSync(${opts.warm}); })()`);
      await ev('(async()=>{ await window.__AMR.setAutoRefine(false); })()');

      const nLevels = await ev('window.__AMR.getNumLevels()');
      const dims = await ev('window.__AMR.getBlockGridDims()');
      const W = (await ev('window.__AMR.getDims()')).W;
      const active = {};
      for (let m = 1; m < nLevels; m++) {
        active[m] = await ev(`(async()=>{ return (await window.__AMR.debugListActiveBlocks(${m})).length; })()`);
      }

      const ts = [];
      for (let r = 0; r < opts.reps; r++) {
        ts.push(await ev(`(async()=>{ const t0 = performance.now();
          await window.__AMR.debugStepSync(${opts.steps});
          return performance.now() - t0; })()`));
      }
      ts.sort((a, b) => a - b);
      const med = ts[Math.floor(ts.length / 2)];

      // Honest work metric: L0 plus every level's own substeps.
      let cells = W * W;
      for (let m = 1; m < nLevels; m++) cells += (active[m] || 0) * dims.FB * dims.FB * (2 ** m);
      const mcups = (cells * opts.steps) / (med * 1e3);

      rows.push({ cfg, med, spread: (ts[ts.length - 1] - ts[0]) / med, active, mcups });
      console.log(`?${cfg}`);
      console.log(`   ${opts.steps} macro-steps: median ${med.toFixed(0)} ms  (spread ${((ts[ts.length - 1] - ts[0]) / med * 100).toFixed(1)}%)`);
      console.log(`   activeByLevel=${JSON.stringify(active)}   ${mcups.toFixed(1)} Mcell-updates/s (all levels)`);
    }

    if (rows.length > 1) {
      console.log('\ncomparison (read activeByLevel first -- differing topology means the');
      console.log('timings are measuring different workloads, see this file\'s header):');
      const base = rows[0];
      for (const r of rows.slice(1)) {
        const dt = (r.med - base.med) / base.med * 100;
        const da = Object.keys(base.active).map(m => (r.active[m] - base.active[m])).join(',');
        console.log(`   ${r.cfg}\n      time ${dt >= 0 ? '+' : ''}${dt.toFixed(1)}% vs first   activeL1..n delta [${da}]`);
      }
    }
  } finally {
    await client.close();
    await BL.teardown({ port: PORT, tabId, chrome, server, keepOpen: opts.keepOpen });
  }
}

main().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
