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
//     distributions, or use --skip below.
//
// DO NOT PUT ?benchSkip= IN A CONFIG STRING. It is applied from page load, so
// the WARM-UP runs with the modified physics: skipping the force pass stops
// the card moving, the flow goes somewhere else entirely, and refinement
// settles on a different topology. Measured: warming up with force skipped
// gave 73 active L1 blocks against the baseline's 123, and the resulting
// "-15.9%" was two different simulations, not a saving. This script now
// refuses such a config string.
//
// --skip is the correct instrument: one page load, one warm-up with real
// physics, refinement frozen, and then the skip set changed between timed
// runs via setBenchSkip so the ONLY thing that varies is the dispatch list.
//
// Usage:
//   node tools/bench-amr.js --steps=8000 --skip=none,force,phy,force+phy \
//        'res=8&levels=3&blockage=3.3'
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
  const o = { warm: 20000, steps: 20000, reps: 5, page: 'index-amr.html', global: 'window.__AMR', configs: [], keepOpen: false, skip: null };
  for (const a of argv) {
    if (a.startsWith('--warm=')) o.warm = Number(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = Number(a.slice(8));
    else if (a.startsWith('--reps=')) o.reps = Number(a.slice(7));
    else if (a.startsWith('--page=')) o.page = a.slice(7);
    // --global= names the page's own debug surface, mirroring
    // tools/lib/amr-invariants.js's `global` option. index-amr.html exposes
    // window.__AMR; the cylinder/channel/tgv harnesses expose window.__CYL.
    // This is what makes a PINNED-BODY benchmark possible: index-amr.html is
    // the falling card, so any change that perturbs the flow moves the body
    // and with it the geometry-forced refinement, and the two arms stop
    // measuring the same topology. index-cylinder-amr.html holds still.
    else if (a.startsWith('--global=')) o.global = a.slice(9);
    else if (a.startsWith('--skip=')) o.skip = a.slice(7).split(',').filter(Boolean);
    else if (a === '--keepOpen') o.keepOpen = true;
    else if (a === '--help' || a === '-h') { printHelp(); process.exit(0); }
    else if (a.startsWith('--')) { console.error(`unknown flag ${a}`); process.exit(2); }
    else o.configs.push(a);
  }
  if (!o.configs.length) { printHelp(); process.exit(2); }
  for (const c of o.configs) {
    if (/benchSkip/.test(c)) {
      console.error(`refusing config "${c}": ?benchSkip= in a config string applies from page load, so\n` +
                    `the warm-up runs with altered physics and the runs are not comparable. Use --skip=`);
      process.exit(2);
    }
  }
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

  const G = opts.global;
  // --skip drives the in-page pass-group switch, which only index-amr.html
  // implements. Fail loudly rather than throw a bare "not a function" from
  // inside a timed run.
  if (opts.skip && G !== 'window.__AMR') {
    console.error(`--skip needs setBenchSkip, which only window.__AMR provides (got --global=${G})`);
    process.exit(2);
  }

  const rows = [];
  try {
    for (const cfg of opts.configs) {
      await BL.navigateTo(Page, `${BASE_URL}/${opts.page}?${cfg}`);
      await BL.waitForGlobal(Runtime, G, 180000);
      await ev(`${G}.setLive(false)`);
      await ev(`(async()=>{ await ${G}.debugStepSync(${opts.warm}); })()`);
      await ev(`(async()=>{ await ${G}.setAutoRefine(false); })()`);

      const nLevels = await ev(`${G}.getNumLevels()`);
      const dims = await ev(`${G}.getBlockGridDims()`);
      const W = (await ev(`${G}.getDims()`)).W;
      const active = {};
      for (let m = 1; m < nLevels; m++) {
        active[m] = await ev(`(async()=>{ return (await ${G}.debugListActiveBlocks(${m})).length; })()`);
      }

      const timeOnce = async () => ev(`(async()=>{ const t0 = performance.now();
          await ${G}.debugStepSync(${opts.steps});
          return performance.now() - t0; })()`);

      // --skip: vary ONLY the dispatch list, inside this one frozen topology.
      if (opts.skip) {
        // ROUND-ROBIN, not config-at-a-time, and the order rotates each round.
        // Measuring all reps of one configuration before moving to the next
        // confounds the configuration with TIME, and this GPU drifts: the
        // first arrangement of this sweep reported interp, avg and ghost as
        // NEGATIVE savings (-6%, -12%, -9%) with 33-73% spread on the early
        // rows collapsing to 4-10% on the late ones. Skipping a pass cannot
        // make a frame slower, so that was clock/thermal ramp, not signal.
        // Interleaving spreads any monotonic drift evenly across every
        // configuration instead of loading it onto whichever ran first.
        const apply = async (cfg) => {
          const groups = cfg === 'none' ? [] : cfg.split('+');
          await ev(`JSON.stringify(${G}.setBenchSkip(${JSON.stringify(groups)}))`);
        };
        // Discard a whole settling ROUND, not one run. One was not enough: a
        // sweep whose later rows were clean at 5-11% spread still produced
        // 44-85% on its first rows, with `avg` again coming out NEGATIVE.
        // Touching every configuration once before any of them counts gets
        // each one's first-touch cost out of the measured set.
        for (const cfg of opts.skip) {
          await apply(cfg === 'none' ? 'none' : cfg);
          await timeOnce();
        }

        const samples = new Map(opts.skip.map(c => [c, []]));
        for (let r = 0; r < opts.reps; r++) {
          const order = opts.skip.map((_, i) => opts.skip[(i + r) % opts.skip.length]);
          for (const cfg of order) {
            await apply(cfg);
            samples.get(cfg).push(await timeOnce());
          }
        }
        const base = opts.skip.map(cfg => {
          const xs = samples.get(cfg).slice().sort((a, b) => a - b);
          const med = xs[Math.floor(xs.length / 2)];
          return { cfg, med, spread: (xs[xs.length - 1] - xs[0]) / med };
        });
        await ev(`JSON.stringify(${G}.setBenchSkip([]))`);
        const b0 = base.find(r => r.cfg === 'none') || base[0];
        console.log(`?${cfg}`);
        console.log(`   activeByLevel=${JSON.stringify(active)}  (frozen; identical for every row below)`);
        console.log(`   ${'skipped'.padEnd(20)}${'median ms'.padStart(12)}${'delta'.padStart(10)}${'share'.padStart(9)}${'spread'.padStart(9)}`);
        for (const r of base) {
          const d = b0.med - r.med;
          console.log(`   ${r.cfg.padEnd(20)}${r.med.toFixed(0).padStart(12)}` +
                      `${d.toFixed(0).padStart(10)}${(d / b0.med * 100).toFixed(1).padStart(8)}%` +
                      `${(r.spread * 100).toFixed(1).padStart(8)}%`);
        }
        continue;
      }

      const ts = [];
      for (let r = 0; r < opts.reps; r++) ts.push(await timeOnce());
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
