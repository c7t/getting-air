#!/usr/bin/env node
// Short-horizon divergence check: run the dense reference and AMR forward
// from THE SAME initial state, and measure how, where, and how fast they
// come apart.
//
// WHY THIS AND NOT tools/validate-amr-vs-dense.js
//
// That tool compares one instantaneous snapshot of each solver after running
// them independently. plans/AMR-vs-dense-validation.md's Finding #3 records
// why that stops meaning anything once the flow sheds: at Re=100 both
// solvers were individually correct (every Cd inside the literature band)
// yet `uy` relL2 came out at 1.63, because two correct periodic solutions
// sampled at uncorrelated phase disagree enormously. The number was
// measuring phase, not correctness.
//
// Seeding both legs from one state removes phase from the comparison. What
// is left is divergence, and its SHAPE is the diagnostic:
//
//   step 0 already nonzero      -> injection/reconstruction, not dynamics
//   clustered at tile edges     -> ghost/seam bug          (see `edge` column)
//   peaks at a half-tile offset -> tile registration       (see the mod histogram)
//   concentrated in refined
//     interiors, not their edges-> per-level tau
//   smooth, diffuse, from zero  -> ordinary truncation error, i.e. correct
//
// This is the automated form of the by-hand analysis in that same doc
// ("per-column max-abs-diff and a histogram of anomaly locations mod tile
// size"), which previously had to be redone by hand every time.
//
// THE BASELINE MATTERS. AMR is a different discretization; it will diverge
// from dense even when perfectly correct. A single divergence number is
// therefore uninterpretable on its own. Run --mode=both (the default): the
// `fullrefine` leg has no coarse/fine interface anywhere, so its divergence
// is the noise floor, and the GAP between `adaptive` and `fullrefine` is
// specifically the interface error you actually care about.
//
// Usage:
//   node tools/validate-divergence.js
//   node tools/validate-divergence.js --res=9 --levels=2 --re=20
//   node tools/validate-divergence.js --mode=adaptive --checkpoints=6 --horizon=400
//   node tools/validate-divergence.js --diffuse --saveSnapshots=/tmp/div
//
// Defaults to --bounceback, like tools/validate-amr-vs-dense.js: bounce-back
// is sharp at exactly R, so it carries no effective-radius offset. --diffuse
// is a supported, no-longer-broken path (see main-cylinder-amr.js's comment
// above N_LEVELS) but its own interface width adds a real physical
// difference on top of the discretization one, which is not what this tool
// is trying to isolate.

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');

const REPO_ROOT = path.join(__dirname, '..');
const BL = require('./lib/browser-lifecycle');
const { deriveAMRParams, deriveSharedURLParams, buildDenseUrl, buildAMRUrl } = require('./lib/amr-resolution-mapping');
const { loadDenseFields, reconstructAMRToResolution, buildLevelMap, diffStats } = require('./lib/field-reconstruct');
const { injectDenseIntoAMRSnapshot } = require('./lib/dense-to-amr');

const BASE_URL = 'https://localhost:4444';
const PORT = 9333;

function parseArgs(argv) {
  const o = {
    res: 9, levels: [2], re: [100], mode: 'both',
    horizon: 256,      // AMR L0 macro-steps to advance in total after seeding
    checkpoints: 4,    // number of comparison points across that horizon
    settle: null,      // AMR macro-steps before snapshotting the topology (null = auto)
    transient: null,   // dense steps before seeding (null = auto)
    bounceback: true,
    timeout: 600,
    saveSnapshots: null,
    keepOpen: false,
    allowUnvalidatedLevels: false,
    allowMarginalTau: false,
  };
  for (const a of argv) {
    if (a.startsWith('--res=')) o.res = Number(a.slice(6));
    else if (a.startsWith('--levels=')) o.levels = a.slice(9).split(',').map(Number);
    else if (a.startsWith('--re=')) o.re = a.slice(5).split(',').map(Number);
    else if (a.startsWith('--mode=')) o.mode = a.slice(7);
    else if (a.startsWith('--horizon=')) o.horizon = Number(a.slice(10));
    else if (a.startsWith('--checkpoints=')) o.checkpoints = Number(a.slice(14));
    else if (a.startsWith('--settle=')) o.settle = Number(a.slice(9));
    else if (a.startsWith('--transient=')) o.transient = Number(a.slice(12));
    else if (a.startsWith('--timeout=')) o.timeout = Number(a.slice(10));
    else if (a.startsWith('--saveSnapshots=')) o.saveSnapshots = a.slice(16);
    else if (a === '--diffuse') o.bounceback = false;
    else if (a === '--bounceback') o.bounceback = true;
    else if (a === '--keepOpen') o.keepOpen = true;
    else if (a === '--allowUnvalidatedLevels') o.allowUnvalidatedLevels = true;
    else if (a === '--allowMarginalTau') o.allowMarginalTau = true;
    else if (a === '--help' || a === '-h') { printHelp(); process.exit(0); }
    else { console.error(`unknown argument: ${a} (try --help)`); process.exit(2); }
  }
  if (!['both', 'adaptive', 'fullrefine'].includes(o.mode)) {
    console.error(`--mode must be one of both|adaptive|fullrefine, got '${o.mode}'`); process.exit(2);
  }
  if (o.checkpoints < 1) { console.error('--checkpoints must be >= 1'); process.exit(2); }
  return o;
}

function printHelp() {
  console.log(fs.readFileSync(__filename, 'utf8').split('\n')
    .filter(l => l.startsWith('//')).map(l => l.replace(/^\/\/ ?/, '')).join('\n'));
}

async function ev(Runtime, expr, timeoutMs) {
  const r = await BL.evalExpr(Runtime, expr, timeoutMs || 120000);
  if (r.exceptionDetails) {
    throw new Error(r.exceptionDetails.text + ' :: ' +
      (r.exceptionDetails.exception && r.exceptionDetails.exception.description || ''));
  }
  return r.result.value;
}

// Reuses validate-amr-vs-dense.js's batched approach: ONE in-page loop, not
// one CDP round-trip per block (each debugActivateBlock is a real GPU sync).
// Levels must activate outer-to-inner -- debugActivateBlock throws if a
// level>=2 block's parent isn't active yet.
async function fullyRefine(Runtime, nLevels, timeoutMs) {
  return ev(Runtime, `(async () => {
    const dims = window.__CYL.getBlockGridDims();
    await window.__CYL.setAutoRefine(false);
    for (let level = 1; level < ${nLevels}; level++) {
      const nbx = dims.NBX * (2 ** (level - 1));
      const nby = dims.NBY * (2 ** (level - 1));
      for (let by = 0; by < nby; by++) {
        for (let bx = 0; bx < nbx; bx++) await window.__CYL.debugActivateBlock(bx, by, level);
      }
    }
    return true;
  })()`, timeoutMs);
}

// Full refinement needs every block at every level resident simultaneously:
// level m's own grid is NBX0*2^(m-1) x NBY0*2^(m-1), so it needs
// NBLOCKS0 * 4^(m-1) slots. The pages' defaults (128) are sized for sparse,
// body-following refinement and will throw "pool exhausted" well before a
// full sweep completes. Deterministic, so size it here rather than making
// the caller discover the number from a crash (which is what
// validate-amr-vs-dense.js's manual --maxFineBlocks= currently requires).
const BLOCK = 8; // matches shaders/amr_step.wgsl's cellIndex
function fullRefinePoolSizes(baseResLog2, nLevels) {
  const nbx0 = (1 << baseResLog2) / BLOCK;
  const nblocks0 = nbx0 * nbx0;
  const byLevel = {};
  let bytes = 0;
  const GHOST = 2, RB = BLOCK, FB = RB * 2 + 2 * GHOST;
  const perSlot = FB * FB * (9 * 4 * 2 + 2 * 4); // f_a + f_b + vel
  for (let m = 1; m < nLevels; m++) {
    const need = nblocks0 * (4 ** (m - 1));
    byLevel[m] = need;
    bytes += need * perSlot;
  }
  return { byLevel, bytes };
}

// An injected snapshot is multi-megabyte JSON (a 512^2 dense f alone is
// ~12MB of base64). Embedding that in one Runtime.evaluate expression is
// fragile, so transfer it in chunks and reassemble in the page. Chunking the
// STRING rather than the object keeps this independent of the snapshot's
// shape -- it stays correct if the format gains fields.
async function loadSnapshotIntoPage(Runtime, snapshot, timeoutMs) {
  const json = JSON.stringify(snapshot);
  const CHUNK = 4 * 1024 * 1024;
  await ev(Runtime, 'window.__divChunks = []; true', timeoutMs);
  for (let i = 0; i < json.length; i += CHUNK) {
    const piece = json.slice(i, i + CHUNK);
    // JSON.stringify of the chunk gives a correctly-escaped JS string
    // literal -- the payload is base64 and JSON punctuation, so hand-quoting
    // it would be a quoting bug waiting to happen.
    await ev(Runtime, `window.__divChunks.push(${JSON.stringify(piece)}); true`, timeoutMs);
  }
  await ev(Runtime, `(async () => {
    const snap = JSON.parse(window.__divChunks.join(''));
    window.__divChunks = null;
    await window.__CYL.debugSnapshotLoad(snap);
    return true;
  })()`, timeoutMs);
}

// Where does the disagreement live? Three cheap, independent cuts, all
// computed on the same |diff| field so they can be read against each other.
function localizeError(denseF, amrF, snapshot, targetResLog2, RB) {
  // targetResLog2 is used by buildLevelMap below.
  const W = denseF.W;
  const N = W * W;
  const diff = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const dx = denseF.ux[i] - amrF.ux[i], dy = denseF.uy[i] - amrF.uy[i];
    diff[i] = Math.hypot(dx, dy);
  }
  let total = 0, maxD = 0;
  for (let i = 0; i < N; i++) { total += diff[i]; if (diff[i] > maxD) maxD = diff[i]; }

  // 1. Error binned by column position modulo the finest level's tile
  //    footprint. A level-m tile's interior is 2*RB cells of size
  //    2^(finest-m) target cells each, so at the FINEST level a tile spans
  //    exactly 2*RB target cells -- that is the period any seam artifact
  //    repeats on. A seam/ghost bug piles error at offset 0 (the tile
  //    border); the registration bug class piles it at period/2.
  const period = 2 * RB;
  const modHist = new Float64Array(period);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) modHist[x % period] += diff[y * W + x];
  }

  // 2. Error by which level actually owns each cell -- a per-level tau or
  //    per-level force problem shows up as one level carrying the error.
  let byLevel = null;
  try {
    // buildLevelMap returns the per-target-cell level array directly.
    const levels = buildLevelMap(snapshot, targetResLog2);
    const sums = {}, counts = {};
    for (let i = 0; i < N; i++) {
      const L = levels[i];
      sums[L] = (sums[L] || 0) + diff[i];
      counts[L] = (counts[L] || 0) + 1;
    }
    byLevel = Object.keys(sums).sort().map(L => ({
      level: Number(L), meanErr: sums[L] / counts[L], cells: counts[L],
      shareOfTotal: total > 0 ? sums[L] / total : 0,
    }));
  } catch { /* buildLevelMap shape varies; localization is best-effort */ }

  // 3. Concentration: what fraction of the total error lives in the worst 1%
  //    of cells. Near 0.01 means diffuse (discretization); near 1 means a
  //    small localized cluster (a bug with an address).
  const sorted = Array.from(diff).sort((a, b) => b - a);
  const top1pct = Math.max(1, Math.floor(N * 0.01));
  let topSum = 0;
  for (let i = 0; i < top1pct; i++) topSum += sorted[i];

  const edgeSum = modHist[0] + modHist[period - 1];
  const halfIdx = Math.floor(period / 2);
  const halfSum = modHist[halfIdx] + modHist[(halfIdx + period - 1) % period];
  const meanPerCol = total / period;

  return {
    meanErr: total / N, maxErr: maxD,
    concentration: total > 0 ? topSum / total : 0,
    edgeRatio: meanPerCol > 0 ? edgeSum / (2 * meanPerCol) : 0,
    halfTileRatio: meanPerCol > 0 ? halfSum / (2 * meanPerCol) : 0,
    period, byLevel,
  };
}

// Navigate the ONE reused tab to a page and hand back its own parameters --
// one WebGPU context alive at a time, the same discipline validate-all.js
// and validate-amr-vs-dense.js follow.
async function openPage(Page, Runtime, url) {
  await BL.navigateTo(Page, url);
  await BL.waitForGlobal(Runtime, 'window.__CYL', 120000);
  await ev(Runtime, 'window.__CYL.setLive(false)');
  return ev(Runtime, 'window.__CYL.getParams()');
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const server = await BL.ensureServer(BASE_URL, REPO_ROOT);
  const chrome = await BL.ensureChrome(PORT);
  const tabId = await BL.openTab(PORT, 'about:blank');
  const client = await CDP({ port: PORT, target: tabId });
  const { Page, Runtime } = client;
  await Page.enable(); await Runtime.enable();

  const modes = opts.mode === 'both' ? ['fullrefine', 'adaptive'] : [opts.mode];
  const results = [];
  const T = opts.timeout * 1000;

  try {
    for (const nLevels of opts.levels) {
      const amrParams = deriveAMRParams({
        targetResLog2: opts.res, nLevels, re: opts.re[0],
        allowUnvalidatedLevels: opts.allowUnvalidatedLevels,
        allowMarginalTau: opts.allowMarginalTau,
      });
      const refineRatio = 1 << (nLevels - 1); // dense steps per AMR L0 macro-step

      for (const re of opts.re) {
        const shared = deriveSharedURLParams({ re, bounceback: opts.bounceback });
        const denseUrl = buildDenseUrl(BASE_URL, { targetResLog2: opts.res, sharedParams: shared });
        // fullrefine needs every slot resident at once; adaptive uses the
        // pages' own sparse defaults. Built per-mode below.
        const fr = fullRefinePoolSizes(amrParams.baseResLog2, nLevels);
        const amrUrlFor = (mode) => buildAMRUrl(BASE_URL, {
          baseResLog2: amrParams.baseResLog2, nLevels, sharedParams: shared,
          maxFineBlocksByLevel: mode === 'fullrefine' ? fr.byLevel : {},
          forceBounceback: opts.bounceback && opts.allowUnvalidatedLevels,
        });

        // Checkpoint schedule, in AMR L0 macro-steps. Dense advances
        // refineRatio times as many of its own steps per checkpoint: under
        // acoustic scaling a dense run at 2^(nLevels-1) the resolution needs
        // that many more native steps to cover the same physical time.
        const perCk = Math.max(1, Math.round(opts.horizon / opts.checkpoints));
        const transient = opts.transient != null ? opts.transient : 40 * refineRatio * 16;
        const settle = opts.settle != null ? opts.settle : 2048;

        console.log(`\n${'='.repeat(72)}`);
        console.log(`res=${opts.res} (dense ${1 << opts.res}^2)  levels=${nLevels} (AMR base ${1 << amrParams.baseResLog2}^2)  Re=${re}  ${opts.bounceback ? 'bounceback' : 'diffuse'}`);
        console.log(`horizon=${opts.horizon} AMR macro-steps in ${opts.checkpoints} checkpoints of ${perCk} (dense: ${perCk * refineRatio} steps each)`);
        console.log('='.repeat(72));

        // ── Dense leg: develop the flow, seed, then walk the checkpoints ──
        console.log(`\n[dense] ${denseUrl}`);
        const dParams = await openPage(Page, Runtime, denseUrl);
        console.log(`  D=${dParams.D.toFixed(2)} U0=${dParams.U0} TAU=${dParams.TAU.toFixed(5)} -- transient ${transient} steps`);
        await ev(Runtime, `(async()=>{ await window.__CYL.setRe(${re}); })()`, T);
        await ev(Runtime, `(async()=>{ await window.__CYL.debugRunAndCollect(${transient}); })()`, T);
        const denseSnaps = [await ev(Runtime, 'window.__CYL.debugSnapshotSave()', T)];
        for (let k = 1; k <= opts.checkpoints; k++) {
          await ev(Runtime, `(async()=>{ await window.__CYL.debugRunAndCollect(${perCk * refineRatio}); })()`, T);
          denseSnaps.push(await ev(Runtime, 'window.__CYL.debugSnapshotSave()', T));
          process.stdout.write(`\r  checkpoint ${k}/${opts.checkpoints} captured`);
        }
        console.log('');

        for (const mode of modes) {
          const amrUrl = amrUrlFor(mode);
          console.log(`\n[amr:${mode}] ${amrUrl}`);
          if (mode === 'fullrefine') {
            console.log(`  fullrefine pools: ${Object.entries(fr.byLevel).map(([m, n]) => `L${m}=${n}`).join(' ')} ` +
              `(~${(fr.bytes / 1024 / 1024).toFixed(0)} MiB of pool buffers)`);
          }
          const aParams = await openPage(Page, Runtime, amrUrl);
          await ev(Runtime, `(async()=>{ await window.__CYL.setRe(${re}); })()`, T);
          if (mode === 'fullrefine') {
            console.log(`  activating every block at every level (nLevels=${nLevels})...`);
            await fullyRefine(Runtime, nLevels, T);
          } else {
            // Let the vorticity criterion build a REAL refinement structure
            // before we borrow its topology -- injecting into a just-reset
            // page would give a body halo and nothing else, which is not the
            // configuration adaptive mode is supposed to be testing.
            console.log(`  settling refinement structure (${settle} macro-steps)...`);
            await ev(Runtime, `(async()=>{ await window.__CYL.debugRunAndCollect(${settle}); })()`, T);
          }
          console.log(`  D=${aParams.D.toFixed(2)} TAU=${aParams.TAU.toFixed(5)}`);

          const template = await ev(Runtime, 'window.__CYL.debugSnapshotSave()', T);
          const injected = injectDenseIntoAMRSnapshot({ denseSnapshot: denseSnaps[0], amrSnapshot: template });
          console.log('  injecting dense state into the AMR hierarchy...');
          await loadSnapshotIntoPage(Runtime, injected, T);
          // Freeze the topology across the horizon. Refine/coarsen during the
          // comparison would change WHICH level owns a region mid-run, so a
          // divergence step could be re-partitioning rather than physics --
          // exactly the confound this tool exists to avoid.
          await ev(Runtime, '(async()=>{ await window.__CYL.setAutoRefine(false); })()', T);

          const amrSnaps = [await ev(Runtime, 'window.__CYL.debugSnapshotSave()', T)];
          for (let k = 1; k <= opts.checkpoints; k++) {
            await ev(Runtime, `(async()=>{ await window.__CYL.debugRunAndCollect(${perCk}); })()`, T);
            amrSnaps.push(await ev(Runtime, 'window.__CYL.debugSnapshotSave()', T));
            process.stdout.write(`\r  checkpoint ${k}/${opts.checkpoints} captured`);
          }
          console.log('');

          if (opts.saveSnapshots) {
            const dir = path.join(opts.saveSnapshots, `res${opts.res}-L${nLevels}-re${re}-${mode}`);
            fs.mkdirSync(dir, { recursive: true });
            for (let k = 0; k <= opts.checkpoints; k++) {
              fs.writeFileSync(path.join(dir, `dense-${k}.json`), JSON.stringify(denseSnaps[k]));
              fs.writeFileSync(path.join(dir, `amr-${k}.json`), JSON.stringify(amrSnaps[k]));
            }
            // The Node-side injected snapshot, before it ever crosses into
            // the page -- diffing this against amr-0.json (the same state
            // read back out) separates an injector bug from a page
            // load/save round-trip bug, which is otherwise a guess.
            fs.writeFileSync(path.join(dir, 'injected.json'), JSON.stringify(injected));
            fs.writeFileSync(path.join(dir, 'template.json'), JSON.stringify(template));
            console.log(`  snapshots -> ${dir}`);
          }

          // ── Compare ────────────────────────────────────────────────────
          const RB = template.pools[1].RB;
          const rows = [];
          for (let k = 0; k <= opts.checkpoints; k++) {
            const dF = loadDenseFields(denseSnaps[k]);
            const aF = reconstructAMRToResolution(amrSnaps[k], opts.res);
            const ux = diffStats(dF.ux, aF.ux, 'ux');
            const uy = diffStats(dF.uy, aF.uy, 'uy');
            const rho = diffStats(dF.rho, aF.rho, 'rho');
            const loc = localizeError(dF, aF, amrSnaps[k], opts.res, RB);
            rows.push({ step: k * perCk, ux, uy, rho, loc });
          }

          console.log(`\n  divergence from a shared initial state (${mode}):`);
          console.log('    AMRstep   relL2(ux)  relL2(uy)  relL2(rho)   meanErr    conc(top1%)  edge   half');
          for (const r of rows) {
            console.log(`    ${String(r.step).padStart(7)}   ` +
              `${r.ux.relL2.toExponential(2).padStart(9)}  ${r.uy.relL2.toExponential(2).padStart(9)}  ` +
              `${r.rho.relL2.toExponential(2).padStart(10)}   ${r.loc.meanErr.toExponential(2).padStart(9)}  ` +
              `${r.loc.concentration.toFixed(3).padStart(10)}  ${r.loc.edgeRatio.toFixed(2).padStart(5)}  ${r.loc.halfTileRatio.toFixed(2).padStart(5)}`);
          }
          const seed = rows[0];
          console.log(`\n    seeding error (step 0): relL2(ux)=${seed.ux.relL2.toExponential(2)} -- this is injection+reconstruction only,`);
          console.log('      not dynamics. It should be at the level the AMR hierarchy can represent');
          console.log('      (exact where refined to the finest level, box-average error elsewhere).');
          if (seed.loc.byLevel) {
            console.log('    error share by owning level at step 0:',
              seed.loc.byLevel.map(b => `L${b.level}=${(b.shareOfTotal * 100).toFixed(1)}%`).join('  '));
          }
          results.push({ nLevels, re, mode, rows });
        }
      }
    }

    // ── Summary ──────────────────────────────────────────────────────────
    console.log(`\n${'='.repeat(72)}\nSUMMARY\n${'='.repeat(72)}`);
    console.log('levels  Re    mode        seed relL2(ux)  final relL2(ux)  growth');
    for (const r of results) {
      const s0 = r.rows[0].ux.relL2, fN = r.rows[r.rows.length - 1].ux.relL2;
      // A zero seed is the GOOD outcome for fullrefine (1:1 injection is
      // bit-exact), not a divide-by-zero to paper over with a floor.
      const growth = s0 === 0 ? 'n/a (exact seed)' : `${(fN / s0).toFixed(1)}x`;
      console.log(`${String(r.nLevels).padStart(6)}  ${String(r.re).padStart(4)}  ${r.mode.padEnd(11)} ` +
        `${s0.toExponential(2).padStart(13)}  ${fN.toExponential(2).padStart(15)}  ${growth}`);
    }
    // A fullrefine seeding error is diagnostic on its own: at fullrefine the
    // finest level maps 1:1 onto the dense grid, so injection is an exact
    // copy and step 0 must be bit-identical. Anything else is an injector or
    // reconstruction fault, NOT a solver one -- and it invalidates every
    // divergence number below it, so say so loudly rather than letting it be
    // read as physics.
    for (const r of results) {
      if (r.mode === 'fullrefine' && r.rows[0].ux.relL2 > 1e-9) {
        console.log(`\n  WARNING levels=${r.nLevels} Re=${r.re}: fullrefine seeded with relL2(ux)=` +
          `${r.rows[0].ux.relL2.toExponential(2)}, but a 1:1 injection must be exact. Suspect the ` +
          `injection/reconstruction path (tools/test-dense-to-amr.js, and check that the page's ` +
          `velocity buffers carry COPY_DST), not the solver.`);
      }
    }
    // The gap between the two modes is the actual quantity of interest --
    // fullrefine has no coarse/fine interface, so whatever it diverges by is
    // the discretization noise floor, and adaptive's excess over it is
    // interface error. Reporting one without the other invites reading
    // ordinary truncation error as a bug.
    const byKey = {};
    for (const r of results) (byKey[`${r.nLevels}/${r.re}`] ||= {})[r.mode] = r;
    for (const [key, m] of Object.entries(byKey)) {
      if (m.adaptive && m.fullrefine) {
        const a = m.adaptive.rows[m.adaptive.rows.length - 1].ux.relL2;
        const f = m.fullrefine.rows[m.fullrefine.rows.length - 1].ux.relL2;
        console.log(`\nlevels/Re ${key}: adaptive final relL2 ${a.toExponential(2)} vs fullrefine (noise floor) ` +
          `${f.toExponential(2)} -- interface excess ${(a / Math.max(f, 1e-30)).toFixed(2)}x`);
      }
    }
    console.log('\nThis tool reports; it does not PASS/FAIL. There is no literature value for');
    console.log('"how fast should two discretizations diverge" -- read the shape (see this');
    console.log("file's header) and compare adaptive against its own fullrefine baseline.");
  } finally {
    await client.close();
    await BL.teardown({ port: PORT, tabId, chrome, server, keepOpen: opts.keepOpen });
  }
}

main().catch(e => { console.error('\nERROR:', e.message); process.exit(1); });
