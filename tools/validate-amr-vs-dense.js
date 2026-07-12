#!/usr/bin/env node
// Dense-vs-AMR full-field correctness tool: runs the dense reference
// (index-cylinder.html) at a HIGH target resolution (e.g. 1024^2) and the
// AMR cylinder harness (index-cylinder-amr.html) refined down to that SAME
// physical resolution at the cylinder surface (coarser in the bulk), then
// directly diffs the two solvers' velocity/density/vorticity FIELDS -- not
// just integrated Cd/St, which is all tools/validate-cylinder.js's
// literature-comparison check does. Two independently-plausible solvers
// isn't the same claim as "AMR reproduces what the dense solver computes at
// matched resolution" -- this tool is the sharper test.
//
// Standalone and opt-in, NOT part of tools/validate-all.js's routine sweep
// -- a 1024^2 dense run is far more expensive than that suite's default
// res=9 configs.
//
// The resolution-scaling law (baseResLog2 = targetResLog2 - (nLevels-1),
// plus the [7,11] clamp / validated-levels cap / base-tau stability margin)
// lives in tools/lib/amr-resolution-mapping.js. The AMR-quadtree-pool ->
// uniform-target-grid reconstruction lives in tools/lib/field-reconstruct.js
// (see tools/test-field-reconstruct.js for its own pure-Node fixture test --
// run that first if this tool's output ever looks suspicious, to rule out
// the reconstruction code itself before suspecting AMR). Cost accounting
// lives in tools/lib/amr-cost.js.
//
// Tolerances below are EXPLICITLY STARTING POINTS, not validated numbers --
// there's no literature ground truth for dense-vs-AMR field relL2 the way
// benchmarks/cylinder.json has for Cd/St. Record real measured relL2 values
// here once this tool has actually been run, the same way
// main-cylinder-amr.js's own N_LEVELS comment cites concrete measured
// numbers instead of leaving claims abstract. (Not yet recorded -- no run
// has been completed at the time this tool was written.)
//
// KNOWN ISSUE, discovered while building this tool (2026-07-11): the
// DIFFUSE (default, non-bounceback) coupling method is currently failing
// its own Cd/St-vs-literature check on this machine, at Re=100, for BOTH
// solvers -- `node tools/validate-all.js` (unmodified main, no changes from
// this tool) currently reports dense-reference Cd=1.908 (target 1.35±0.15),
// amr-N2-diffuse Cd=0.650, amr-N3-diffuse Cd=0.085 -- degrading further with
// each added level, which reads like the diffuse coupling's error compounds
// under refinement rather than being a fixed offset. Only bounceback
// currently passes (amr-N2-bounceback/amr-N3-bounceback both PASS; dense
// index-cylinder.html?bounceback also independently verified PASS: Cd=1.327,
// St=0.161 at Re=100). Root cause not yet investigated (could be a real
// regression or a GPU/driver-specific numerical issue on this particular
// machine) -- tracked separately from this tool's own scope. Until that's
// resolved, THIS TOOL DEFAULTS TO BOUNCEBACK COUPLING (opposite of the raw
// pages' own default) so it's actually usable; a diffuse-mode run today
// would fail for reasons that have nothing to do with dense-vs-AMR
// resolution-matching, and would be indistinguishable from a real finding
// without knowing this. Pass --diffuse to reproduce/investigate the
// regression, or once it's fixed.
//
// REAL MEASURED NUMBERS (2026-07-11, first --mode=fullrefine run, bounceback,
// res=9 nLevels=2 re=20, i.e. dense@512 vs AMR base=256 fully force-refined --
// see this file's own "fullrefine" description above for why this is the
// cleanest self-consistency case): Cd matched to 0.5% (dense=2.180,
// amr=2.191) and ux/rho matched tight (relL2 9.7e-3 / 5.1e-5) -- strong
// corroboration the resolution-matching and reconstruction machinery is
// fundamentally sound. uy/omega did NOT meet --tolFullrefine (relL2 0.22 /
// 0.70) -- traced this by hand (not a placeholder guess): both solvers' own
// peak |uy| locations and magnitudes agree to within 1 cell and <2%, and the
// worst individual-cell disagreements are a small, localized cluster (~0.13%
// of cells) right near the peak-shear-layer region, not spread across the
// domain and not at tile/domain-edge boundaries (ruled out a reconstruction
// indexing bug by checking both). Re=20 is a steady, top/bottom-symmetric
// flow (no perturbation, see --perturb=0 default) -- uy's own physical
// magnitude is tiny (~0.02, vs ux's ~0.04) almost everywhere except right at
// that shear layer, so relL2 (which normalizes by the reference field's own
// norm) is a harsh metric here: a handful of cells where two INDEPENDENTLY-
// implemented WGSL kernels (lbm_step.wgsl vs amr_step.wgsl/amr_step1.wgsl)
// disagree by a small absolute amount, right where gradients are sharpest,
// can dominate the norm of an otherwise-tiny field. Cd/ux/rho are the
// reliable corroborating signals for steady, near-symmetric cases (Re=20,
// 40); uy/omega comparisons are likely more meaningful at shedding Re
// (100, 200) where uy's own magnitude is large domain-wide, not just at one
// shear layer -- NOT YET VERIFIED, follow up before trusting uy/omega
// tolerances at any Re. tolFullrefine/tolOmega below are left at their
// original starting-point values (not loosened to force a pass) so this
// tool keeps reporting the FAIL it's designed to surface -- that's a real,
// open finding, not a tuning target.
//
// Two run modes, both on by default (--mode=both):
//   adaptive   -- AMR's own refine/coarsen decisions, exactly as a real user
//                 would run it. Exercises tools/lib/field-reconstruct.js's
//                 L0-fallback branch (away from the body) most.
//   fullrefine -- every block at every level force-activated
//                 (setAutoRefine(false) + debugActivateBlock everywhere),
//                 zero coarse/fine interface anywhere. Exercises only the
//                 fine-injection branch, and should land very close to
//                 discretization-equivalence with the dense reference --
//                 a FAIL here implicates this tool's OWN reconstruction
//                 code before it implicates AMR's accuracy. A FAIL in
//                 adaptive mode with fullrefine PASSING is a genuine
//                 AMR-accuracy finding, not a tooling bug.
//
// Usage:
//   node tools/validate-amr-vs-dense.js --res=10 --levels=2,3 --re=20,40
//   node tools/validate-amr-vs-dense.js --res=8 --levels=2 --re=20 --mode=fullrefine
//   node tools/validate-amr-vs-dense.js --res=10 --levels=3 --re=200 --allowUnvalidatedLevels
//   node tools/validate-amr-vs-dense.js --res=10 --levels=3 --dryRun

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal, evalExpr, teardown,
} = require('./lib/browser-lifecycle');
const { computeWindow, analyze } = require('./lib/cylinder-metrics');
const { loadDenseFields, reconstructAMRToResolution, diffStats } = require('./lib/field-reconstruct');
const { deriveAMRParams, deriveSharedURLParams, buildDenseUrl, buildAMRUrl } = require('./lib/amr-resolution-mapping');
const { computeCostSavings } = require('./lib/amr-cost');

const REPO_ROOT = path.join(__dirname, '..');
const RB = 8; // level-invariant architectural constant, see amr-resolution-mapping.js's header
const FB = RB * 2 + 2 * 2; // FB=20 (GHOST=2), matches main-cylinder-amr.js

function parseArgs(argv) {
  const opts = {
    baseUrl: 'https://localhost:4444',
    port: 9333,
    res: [10],
    levels: [2, 3],
    re: [20, 40],
    // bounceback defaults to true here (opposite of the raw pages' own
    // default) -- see this file's own header, "KNOWN ISSUE" -- diffuse
    // coupling is currently failing its own Cd/St check independent of
    // this tool. Pass --diffuse to override.
    blockage: 24, upstream: 8, u0: 0.04, seed: 12345, bounceback: true, perturb: 0,
    mode: 'both',
    maxFineBlocksByLevel: {},
    tol: 0.05, tolOmega: 0.15, tolFullrefine: 0.01,
    allowUnvalidatedLevels: false, allowMarginalTau: false,
    timeout: 1800,
    saveSnapshots: null,
    dryRun: false, keepOpen: false,
  };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) opts.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) opts.port = parseInt(a.slice(7));
    else if (a.startsWith('--res=')) opts.res = a.slice(6).split(',').map(Number);
    else if (a.startsWith('--levels=')) opts.levels = a.slice(9).split(',').map(Number);
    else if (a.startsWith('--re=')) opts.re = a.slice(5).split(',').map(Number);
    else if (a.startsWith('--blockage=')) opts.blockage = parseFloat(a.slice(11));
    else if (a.startsWith('--upstream=')) opts.upstream = parseFloat(a.slice(11));
    else if (a.startsWith('--u0=')) opts.u0 = parseFloat(a.slice(5));
    else if (a.startsWith('--seed=')) opts.seed = parseInt(a.slice(7));
    else if (a === '--bounceback') opts.bounceback = true;
    else if (a === '--diffuse') opts.bounceback = false;
    else if (a.startsWith('--perturb=')) opts.perturb = parseFloat(a.slice(10));
    else if (a.startsWith('--mode=')) opts.mode = a.slice(7);
    else if (/^--maxFineBlocks(\d*)=/.test(a)) {
      const m = a.match(/^--maxFineBlocks(\d*)=(.*)$/);
      const level = m[1] ? parseInt(m[1]) : 1;
      opts.maxFineBlocksByLevel[level] = parseInt(m[2]);
    }
    else if (a.startsWith('--tolOmega=')) opts.tolOmega = parseFloat(a.slice(11));
    else if (a.startsWith('--tolFullrefine=')) opts.tolFullrefine = parseFloat(a.slice(16));
    else if (a.startsWith('--tol=')) opts.tol = parseFloat(a.slice(6));
    else if (a === '--allowUnvalidatedLevels') opts.allowUnvalidatedLevels = true;
    else if (a === '--allowMarginalTau') opts.allowMarginalTau = true;
    else if (a.startsWith('--timeout=')) opts.timeout = parseInt(a.slice(10));
    else if (a.startsWith('--saveSnapshots=')) opts.saveSnapshots = a.slice(16);
    else if (a === '--dryRun') opts.dryRun = true;
    else if (a === '--keepOpen') opts.keepOpen = true;
  }
  return opts;
}

function modesFor(opts) {
  if (opts.mode === 'both') return ['adaptive', 'fullrefine'];
  if (opts.mode === 'adaptive' || opts.mode === 'fullrefine') return [opts.mode];
  throw new Error(`--mode=${opts.mode} invalid -- must be adaptive, fullrefine, or both`);
}

function lookupCaseEntry(re, bench) {
  const found = bench.cases.find(c => c.re === re);
  if (found) return found;
  console.warn(`  [warn] Re=${re} not in benchmarks/cylinder.json -- assuming steady (no shedding) for step-count/window sizing; ` +
    `if this Re actually sheds, pass --perturb=0 anyway (RNG doesn't correlate across resolutions, see this tool's own header) and expect a wider window than ideal.`);
  return { re, st: null, regime: 'unknown (not in benchmarks/cylinder.json)' };
}

// Exact block counts needed to fully refine a base grid of W_base x H_base
// at every level up to nLevels-1 -- level m's own block grid is
// (W_base/RB * 2^(m-1)) x (H_base/RB * 2^(m-1)) (doubles per axis per
// level, see main-cylinder-amr.js's pools[] construction), so its total
// block count is NBX_base*NBY_base*4^(m-1).
function fullRefineBlockCounts(W_base, nLevels) {
  const NBX_base = W_base / RB, NBY_base = W_base / RB;
  const counts = {};
  for (let m = 1; m < nLevels; m++) counts[m] = NBX_base * NBY_base * (4 ** (m - 1));
  return { NBX_base, NBY_base, counts };
}

function estimateFullrefineBytes(counts) {
  // Per pool slot: f_a + f_b (9 components each) + vel (2 components), all
  // f32 -- matches allocLevelPool's own buffer set (ignoring the smaller
  // bookkeeping buffers, which are O(NBLOCKS) not O(cells) and negligible).
  const bytesPerSlot = FB * FB * (9 * 4 * 2 + 2 * 4);
  let total = 0;
  for (const m of Object.keys(counts)) total += counts[m] * bytesPerSlot;
  return total;
}

// Batched in-page activation -- ONE Runtime.evaluate looping every block at
// every level, not one CDP round-trip per block (each debugActivateBlock
// call is a real GPU sync; thousands of individual round-trips from Node
// would be needlessly slow). Levels must activate outer-to-inner (level 1
// fully before level 2 starts) since debugActivateBlock throws if a
// level>=2 block's parent isn't active yet -- see main-cylinder-amr.js's
// own debugActivateBlock.
async function fullyRefine(Runtime, nLevels, timeoutMs) {
  const expr = `(async () => {
    const dims = window.__CYL.getBlockGridDims();
    window.__CYL.setAutoRefine(false);
    for (let level = 1; level < ${nLevels}; level++) {
      const nbx = dims.NBX * (2 ** (level - 1));
      const nby = dims.NBY * (2 ** (level - 1));
      for (let by = 0; by < nby; by++) {
        for (let bx = 0; bx < nbx; bx++) {
          await window.__CYL.debugActivateBlock(bx, by, level);
        }
      }
    }
    return { ok: true };
  })()`;
  const r = await evalExpr(Runtime, expr, timeoutMs);
  if (r.exceptionDetails) throw new Error(`fullyRefine failed: ${r.exceptionDetails.text}`);
  return r.result.value;
}

async function runDenseLeg(Runtime, opts, targetResLog2, caseEntry) {
  await evalExpr(Runtime, 'window.__CYL.setLive(false)');
  await evalExpr(Runtime, 'window.__CYL.reset()');
  const params = (await evalExpr(Runtime, 'window.__CYL.getParams()')).result.value;
  const { transient, total } = computeWindow(caseEntry, params.D, params.U0);
  console.log(`  [dense] res=${targetResLog2} Re=${caseEntry.re}: D=${params.D} U0=${params.U0} TAU=${params.TAU.toFixed(5)} -- running ${total} steps`);
  const r = await evalExpr(Runtime, `window.__CYL.debugRunAndCollect(${total})`, (opts.timeout + 30) * 1000);
  if (r.exceptionDetails) throw new Error(`dense debugRunAndCollect failed: ${r.exceptionDetails.text}`);
  const history = r.result.value.history;
  const snap = await evalExpr(Runtime, 'window.__CYL.debugSnapshotSave()', 60000);
  if (snap.exceptionDetails) throw new Error(`dense debugSnapshotSave failed: ${snap.exceptionDetails.text}`);
  return { snapshot: snap.result.value, history, transient, params };
}

async function runAMRLeg(Runtime, opts, amrParams, caseEntry, mode) {
  await evalExpr(Runtime, 'window.__CYL.setLive(false)');
  await evalExpr(Runtime, 'window.__CYL.reset()');
  if (mode === 'fullrefine') {
    console.log(`  [amr:${mode}] activating every block at every level (nLevels=${amrParams.nLevels})...`);
    await fullyRefine(Runtime, amrParams.nLevels, (opts.timeout + 30) * 1000);
  }
  const params = (await evalExpr(Runtime, 'window.__CYL.getParams()')).result.value;
  const { transient, total } = computeWindow(caseEntry, params.D, params.U0);
  console.log(`  [amr:${mode}] base=${amrParams.baseResLog2} nLevels=${amrParams.nLevels} Re=${caseEntry.re}: D=${params.D} TAU=${params.TAU.toFixed(5)} -- running ${total} steps`);
  const r = await evalExpr(Runtime, `window.__CYL.debugRunAndCollect(${total})`, (opts.timeout + 30) * 1000);
  if (r.exceptionDetails) throw new Error(`amr debugRunAndCollect failed: ${r.exceptionDetails.text}`);
  const history = r.result.value.history;
  const snap = await evalExpr(Runtime, 'window.__CYL.debugSnapshotSave()', 90000);
  if (snap.exceptionDetails) throw new Error(`amr debugSnapshotSave failed: ${snap.exceptionDetails.text}`);
  const poolSizes = (await evalExpr(Runtime, 'window.__CYL.getLevelPoolSizes()', 30000)).result.value;
  const activeCountsByLevel = {};
  for (const p of poolSizes) {
    const active = (await evalExpr(Runtime, `window.__CYL.debugListActiveBlocks(${p.level})`, 30000)).result.value;
    activeCountsByLevel[p.level] = active.length;
  }
  return { snapshot: snap.result.value, history, transient, params, activeCountsByLevel };
}

function saveSnapshotIfRequested(opts, name, snapshot) {
  if (!opts.saveSnapshots) return;
  fs.mkdirSync(opts.saveSnapshots, { recursive: true });
  const p = path.join(opts.saveSnapshots, `${name}.json`);
  fs.writeFileSync(p, JSON.stringify(snapshot));
  console.log(`    saved snapshot -> ${p}`);
}

function checkField(label, stats, tol) {
  return { label, ...stats, tol, pass: stats.relL2 <= tol };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const bench = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'benchmarks', 'cylinder.json'), 'utf8'));
  const modes = modesFor(opts);

  // --- build + validate the sweep up front (no GPU touched yet) ---
  const sweep = []; // [{targetRes, re, nLevels, mode, amrParams|null, error|null}]
  for (const targetRes of opts.res) {
    for (const re of opts.re) {
      for (const nLevels of opts.levels) {
        let amrParams;
        try {
          amrParams = deriveAMRParams({
            targetResLog2: targetRes, nLevels, blockage: opts.blockage, u0: opts.u0, re,
            allowUnvalidatedLevels: opts.allowUnvalidatedLevels, allowMarginalTau: opts.allowMarginalTau,
          });
        } catch (e) {
          for (const mode of modes) sweep.push({ targetRes, re, nLevels, mode, amrParams: null, error: e.message });
          continue;
        }
        if (amrParams.tauWarning) console.warn(`  [warn] ${amrParams.tauWarning}`);
        for (const mode of modes) sweep.push({ targetRes, re, nLevels, mode, amrParams, error: null });
      }
    }
  }

  console.log(`\n${sweep.length} config(s) in sweep (${sweep.filter(s => s.error).length} invalid, skipped):`);
  for (const s of sweep) {
    if (s.error) { console.log(`  SKIP  res=${s.targetRes} nLevels=${s.nLevels} re=${s.re} mode=${s.mode}: ${s.error}`); continue; }
    const { W_base, tauBase } = s.amrParams;
    console.log(`  run   res=${s.targetRes} nLevels=${s.nLevels} (base=${s.amrParams.baseResLog2}, W_base=${W_base}) re=${s.re} mode=${s.mode} tauBase=${tauBase.toFixed(4)}`);
  }

  if (opts.dryRun) {
    console.log('\n--dryRun: no Chrome/GPU touched. Fullrefine memory estimates:');
    for (const s of sweep) {
      if (s.error || s.mode !== 'fullrefine') continue;
      const { counts } = fullRefineBlockCounts(s.amrParams.W_base, s.nLevels);
      const bytes = estimateFullrefineBytes(counts);
      console.log(`  res=${s.targetRes} nLevels=${s.nLevels} re=${s.re}: ~${(bytes / 1e6).toFixed(0)}MB fine-pool memory (${JSON.stringify(counts)} blocks/level)`);
    }
    process.exit(0);
  }

  // --- run it for real ---
  const server = await ensureServer(opts.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(opts.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(opts.port) : await openTab(opts.port, 'about:blank');
  const client = await CDP({ port: opts.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  let currentLabel = null;
  Runtime.exceptionThrown(e => console.error(`  [${currentLabel}] browser exception:`, e.exceptionDetails.text));

  const report = [];
  // Dense reference doesn't depend on nLevels -- run/snapshot it once per
  // (targetRes, re) and reuse across the whole --levels= sweep, per this
  // tool's own header (a naive per-triple loop would multiply the
  // expensive leg by |levels|x|modes| for nothing).
  const denseCache = new Map(); // key `${targetRes}:${re}` -> {snapshot, fields, history, transient, params}

  try {
    for (const targetRes of opts.res) {
      for (const re of opts.re) {
        const caseEntry = lookupCaseEntry(re, bench);
        const denseKey = `${targetRes}:${re}`;
        if (!denseCache.has(denseKey)) {
          currentLabel = `dense res=${targetRes} re=${re}`;
          console.log(`\n=== ${currentLabel} ===`);
          const sharedParams = deriveSharedURLParams({
            blockage: opts.blockage, upstream: opts.upstream, re, u0: opts.u0, seed: opts.seed,
            perturb: opts.perturb, bounceback: opts.bounceback,
          });
          const url = buildDenseUrl(opts.baseUrl, { targetResLog2: targetRes, sharedParams });
          await navigateTo(Page, url);
          await waitForGlobal(Runtime, 'window.__CYL', 15000);
          const leg = await runDenseLeg(Runtime, opts, targetRes, caseEntry);
          saveSnapshotIfRequested(opts, `dense-res${targetRes}-re${re}`, leg.snapshot);
          const fields = loadDenseFields(leg.snapshot);
          const cd = analyze(leg.history, leg.transient, leg.params.D, leg.params.U0);
          denseCache.set(denseKey, { ...leg, fields, cd });
        }
        const dense = denseCache.get(denseKey);

        for (const s of sweep) {
          if (s.targetRes !== targetRes || s.re !== re || s.error) continue;
          const { nLevels, mode, amrParams } = s;
          currentLabel = `amr res=${targetRes} nLevels=${nLevels} re=${re} mode=${mode}`;
          console.log(`\n=== ${currentLabel} ===`);

          let maxFineBlocksByLevel = { ...opts.maxFineBlocksByLevel };
          if (mode === 'fullrefine') {
            const { counts } = fullRefineBlockCounts(amrParams.W_base, nLevels);
            maxFineBlocksByLevel = { ...counts, ...opts.maxFineBlocksByLevel }; // explicit CLI overrides win
          }
          const sharedParams = deriveSharedURLParams({
            blockage: opts.blockage, upstream: opts.upstream, re, u0: opts.u0, seed: opts.seed,
            perturb: opts.perturb, bounceback: opts.bounceback,
          });
          const forceBounceback = opts.bounceback && nLevels > 3;
          const url = buildAMRUrl(opts.baseUrl, { baseResLog2: amrParams.baseResLog2, nLevels, sharedParams, maxFineBlocksByLevel, forceBounceback });

          let row;
          try {
            await navigateTo(Page, url);
            await waitForGlobal(Runtime, 'window.__CYL', 15000);
            const leg = await runAMRLeg(Runtime, opts, amrParams, caseEntry, mode);
            saveSnapshotIfRequested(opts, `amr-res${targetRes}-n${nLevels}-re${re}-${mode}`, leg.snapshot);

            const recon = reconstructAMRToResolution(leg.snapshot, targetRes);
            const fieldResults = {
              ux: checkField('ux', diffStats(dense.fields.ux, recon.ux), opts.tol),
              uy: checkField('uy', diffStats(dense.fields.uy, recon.uy), opts.tol),
              rho: checkField('rho', diffStats(dense.fields.rho, recon.rho), opts.tol),
              omega: checkField('omega', diffStats(dense.fields.omega, recon.omega), opts.tolOmega),
            };
            if (mode === 'fullrefine') {
              for (const k of Object.keys(fieldResults)) {
                fieldResults[k].tol = opts.tolFullrefine;
                fieldResults[k].pass = fieldResults[k].relL2 <= opts.tolFullrefine;
              }
            }
            const fieldsPass = Object.values(fieldResults).every(f => f.pass);

            const cdAmr = analyze(leg.history, leg.transient, leg.params.D, leg.params.U0);
            const cost = computeCostSavings({
              W0: amrParams.W_base, H0: amrParams.W_base, RB, nLevels,
              W_target: amrParams.W_target, H_target: amrParams.W_target,
              activeCountsByLevel: leg.activeCountsByLevel,
            });

            row = {
              targetRes, re, nLevels, mode, baseResLog2: amrParams.baseResLog2, tauBase: amrParams.tauBase,
              fieldResults, fieldsPass, finestCoverageFraction: recon.finestCoverageFraction,
              cdDense: dense.cd.cdMean, cdAmr: cdAmr.cdMean,
              cdRelDiff: dense.cd.cdMean ? Math.abs(cdAmr.cdMean - dense.cd.cdMean) / Math.abs(dense.cd.cdMean) : null,
              stDense: dense.cd.st, stAmr: cdAmr.st,
              cost, activeCountsByLevel: leg.activeCountsByLevel,
              error: null,
            };
          } catch (e) {
            row = { targetRes, re, nLevels, mode, error: e.message };
          }
          report.push(row);
        }
      }
    }
  } finally {
    await client.close();
    await teardown({ port: opts.port, tabId, chrome, server, keepOpen: opts.keepOpen });
  }

  // --- final report ---
  console.log('\n' + '='.repeat(100));
  console.log('SUMMARY');
  console.log('='.repeat(100));
  const pad = (s, n) => (String(s) + ' '.repeat(n)).slice(0, Math.max(String(s).length, n)) + ' ';
  console.log(pad('config', 34) + pad('fields', 8) + pad('relL2(ux,uy,rho,omega)', 40) + pad('cost%', 8) + pad('Cd(dense/amr)', 18));
  let allOk = sweep.filter(s => s.error).length === 0; // invalid configs count as failures of the requested sweep
  for (const r of report) {
    const label = `res=${r.targetRes} n=${r.nLevels} re=${r.re} ${r.mode}`;
    if (r.error) {
      console.log(pad(label, 34) + 'ERROR: ' + r.error);
      allOk = false;
      continue;
    }
    const fr = r.fieldResults;
    const relStr = `${fr.ux.relL2.toExponential(2)},${fr.uy.relL2.toExponential(2)},${fr.rho.relL2.toExponential(2)},${fr.omega.relL2.toExponential(2)}`;
    const cdStr = `${r.cdDense.toFixed(3)}/${r.cdAmr.toFixed(3)}`;
    console.log(pad(label, 34) + pad(r.fieldsPass ? 'PASS' : 'FAIL', 8) + pad(relStr, 40) + pad(r.cost.workDoneVsDensePercent.toFixed(1), 8) + cdStr);
    if (!r.fieldsPass) allOk = false;
  }

  console.log('\nDetails for anything not PASS:');
  for (const s of sweep) {
    if (s.error) console.log(`  [SKIPPED] res=${s.targetRes} n=${s.nLevels} re=${s.re} ${s.mode}: ${s.error}`);
  }
  for (const r of report) {
    if (r.error) { console.log(`  [ERROR] res=${r.targetRes} n=${r.nLevels} re=${r.re} ${r.mode}: ${r.error}`); continue; }
    if (!r.fieldsPass) {
      const failing = Object.values(r.fieldResults).filter(f => !f.pass).map(f => `${f.label}(relL2=${f.relL2.toExponential(3)} > tol=${f.tol})`);
      console.log(`  [FAIL] res=${r.targetRes} n=${r.nLevels} re=${r.re} ${r.mode}: ${failing.join(', ')} -- finestCoverage=${(r.finestCoverageFraction * 100).toFixed(1)}%`);
    }
  }

  console.log(allOk ? '\nAll configs PASS.' : '\nSome configs FAILED or were SKIPPED -- see above.');
  process.exit(allOk ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
