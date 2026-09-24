#!/usr/bin/env node
// Does AMR WIN in 2D? index-amr.html against index.html in a same-physics,
// same-domain contest, scored in SIMULATED TIME PER WALL-CLOCK SECOND.
//
// THE PAIRING. An AMR run `res=R&levels=L` resolves the body at its finest
// level, 2^(L-1) times the root grid, so the flat page that solves the same
// problem at the same body resolution is `res=R+L-1`. Everything else is held
// equal and CHECKED, not assumed:
//   - card parameters: both pages take the shared ?blockage/aspect/istar/re/ut
//     (card-params.mjs), so the body is the same size relative to the domain;
//   - tau: the flat page's TAU must equal the AMR page's tauAtLevel(L-1) --
//     the relaxation time of the level the body lives on;
//   - the sponge: index-amr.html's ?spongeW= is in ROOT cells and
//     index.html's in its own, so the flat run gets spongeW x 2^(L-1), the
//     same physical band. (Both default to 4, which at L=4 is an 8x
//     difference -- the reason this is a parameter here and not a default.)
// A pair whose tau or chord do not match is REFUSED, not reported.
//
// THE SCORE. a/u_t per second, Pesavento & Wang's time unit (sim-rate.mjs):
// one a/u_t is A/U_T steps of the grid A is measured on, so an AMR root step
// is worth 2^(L-1) flat steps. Raw steps/s would compare different amounts of
// physics.
//
// WHY SEVERAL CHECKPOINTS FOR AMR AND ONE FOR FLAT. The flat page's cost is a
// function of its grid alone. AMR's is a function of how much of the domain
// the developing wake has refined, so it is timed at several points of one
// fall (--at=, in a/u_t after release). `?detslots=1` makes the AMR fall
// reproducible, so two devices see the same topology at the same checkpoint.
//
// THERMALS. The phone slows 24-57% as it heats (plans/perf-characterization.md),
// and a pair's two legs run minutes apart. So the flat page is timed BEFORE
// and AFTER the AMR leg and the faster of the two is the one AMR is scored
// against -- any drift is charged to AMR, never credited to it. Both readings
// are printed; a large gap between them says how much drift there was.
//
// NOT MEASURED HERE: accuracy. Equal body resolution is the premise; what AMR
// gives up is a coarser far field, and whether that changes the card's
// trajectory is a separate question (the pages' field-vs-dense tools).
//
//   node tools/bench-amr-vs-dense.js                       # desktop, own Chrome
//   adb forward tcp:9229 localabstract:chrome_devtools_remote
//   node tools/bench-amr-vs-dense.js --remote --port=9229 --baseUrl=https://era:4471
//   node tools/bench-amr-vs-dense.js --pairs='res=5&levels=4&interface=explode&spongeW=2'

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const BL = require('./lib/browser-lifecycle');
const REPO_ROOT = path.join(__dirname, '..');

const DEFAULT_PAIRS = [
  'res=5&levels=4&interface=explode&spongeW=2',   // the phone configuration this started from
  'res=6&levels=3&spongeW=4',
  'res=7&levels=2&spongeW=4',
  'res=8&levels=3&spongeW=4',                     // index-amr.html's defaults
  'res=8&levels=3&interface=explode&spongeW=4',
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9355, remote: false, pairs: DEFAULT_PAIRS,
              at: [2, 10, 40], window: 1, flatSteps: 1024, reps: 3 };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = Number(a.slice(7));
    else if (a === '--remote') o.remote = true;
    else if (a.startsWith('--pairs=')) o.pairs = a.slice(8).split(';').filter(Boolean);
    else if (a.startsWith('--at=')) o.at = a.slice(5).split(',').map(Number);
    else if (a.startsWith('--window=')) o.window = Number(a.slice(9));
    else if (a.startsWith('--flatSteps=')) o.flatSteps = Number(a.slice(12));
    else if (a.startsWith('--reps=')) o.reps = Number(a.slice(7));
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

const roundUp = (n, k) => Math.max(k, Math.ceil(n / k) * k);

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const { deriveCardParams, parseCardParams } = await import('../card-params.mjs');

  let server = null, chrome = null, tabId;
  if (o.remote) {
    const tabs = await CDP.List({ port: o.port });
    const t = tabs.find(x => x.type === 'page' && x.url.startsWith(o.baseUrl));
    if (!t) throw new Error(`--remote: no tab on ${o.baseUrl} at port ${o.port} -- open a page on the device first, Chrome foregrounded`);
    tabId = t.id;
  } else {
    server = await BL.ensureServer(o.baseUrl, REPO_ROOT);
    chrome = await BL.ensureChrome(o.port);
    tabId = await BL.openTab(o.port, 'about:blank');
  }
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Page, Runtime } = client;
  await Page.enable(); await Runtime.enable();
  const ev = async (e) => {
    const r = await BL.evalExpr(Runtime, e, 1800000);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  // Time `n` steps `reps` times; the minimum, as bench-d3-interface.js does
  // (a noisy device only ever adds time).
  const timeSteps = (G, n, reps) => ev(`(async()=>{ const ts=[]; for (let i=0;i<${reps};i++){ const t0=performance.now();
      await ${G}.debugStepSync(${n}); ts.push(performance.now()-t0); } return ts; })()`);

  const rows = [];
  try {
    for (const q of o.pairs) {
      const qp = new URLSearchParams(q);
      const R = parseInt(qp.get('res')), L = parseInt(qp.get('levels'));
      const sw = qp.has('spongeW') ? parseFloat(qp.get('spongeW')) : 4;
      const card = parseCardParams(qp);
      const Wroot = 1 << R, Wflat = 1 << (R + L - 1);
      const Aroot = deriveCardParams({ W: Wroot, ...card }).A, Aflat = deriveCardParams({ W: Wflat, ...card }).A;
      const rootPerTu = Aroot / card.U_T, flatPerTu = Aflat / card.U_T;
      // Carry the card parameters through to the flat page, whichever were given.
      const cardQ = ['blockage', 'aspect', 'istar', 're', 'ut'].filter(k => qp.has(k)).map(k => `&${k}=${qp.get(k)}`).join('');
      const flatQ = `res=${R + L - 1}&spongeW=${sw * 2 ** (L - 1)}${cardQ}`;
      console.log(`\n=== AMR index-amr.html?${q}\n    flat index.html?${flatQ}`);

      // ── flat ── (timed again after the AMR leg; see THERMALS above)
      const timeFlat = async () => {
        await BL.navigateTo(Page, `${o.baseUrl}/index.html?${flatQ}`);
        await BL.waitForGlobal(Runtime, 'window.__LBM', 120000);
        await ev('window.__LBM.setLive(false)');
        const fp = await ev('window.__LBM.getCardParams()');
        await timeSteps('window.__LBM', 256, 1);   // pipelines warm, clocks up
        const ts = await timeSteps('window.__LBM', o.flatSteps, o.reps);
        return { fp, msPerStep: Math.min(...ts) / o.flatSteps };
      };
      let flat;
      try {
        const f = await timeFlat();
        flat = { ...f, before: f.msPerStep };
        console.log(`    flat  ${Wflat}^2  A=${f.fp.A}  tau=${f.fp.TAU.toFixed(6)}   before ${f.msPerStep.toFixed(3)} ms/step`);
      } catch (e) {
        console.log(`    flat  FAILED: ${e.message.split('\n')[0]}`);
      }

      // ── AMR ──
      await BL.navigateTo(Page, `${o.baseUrl}/index-amr.html?${q}&detslots=1`);
      await BL.waitForGlobal(Runtime, 'window.__AMR', 120000);
      await ev('window.__AMR.setLive(false)');
      await ev('window.__AMR.reset()');
      const tauFinest = await ev(`window.__AMR.tauAtLevel(${L - 1})`);
      if (flat) {
        const dTau = Math.abs(tauFinest - flat.fp.TAU), dA = Math.abs(Aroot * 2 ** (L - 1) - flat.fp.A);
        if (dTau > 1e-9 || dA > 1e-9) {
          console.log(`    REFUSED: not the same physics -- AMR finest tau ${tauFinest} vs flat ${flat.fp.TAU}, chord in finest cells ${Aroot * 2 ** (L - 1)} vs ${flat.fp.A}`);
          continue;
        }
      }
      const win = roundUp(o.window * rootPerTu, 64);
      const amrRows = [];
      let done = 0;
      for (const at of o.at) {
        const target = roundUp(at * rootPerTu, 64);
        if (target > done) { await ev(`window.__AMR.debugStepSync(${target - done})`); done = target; }
        const ts = await timeSteps('window.__AMR', win, o.reps);
        done += win * o.reps;
        const msPerRoot = Math.min(...ts) / win;
        const tuPerSec = 1000 / (msPerRoot * rootPerTu);
        const act = {};
        for (let m = 1; m < L; m++) act[m] = (await ev(`window.__AMR.debugListActiveBlocks(${m})`)).length;
        amrRows.push({ at, msPerRoot, tuPerSec, act });
      }
      if (flat) {
        const after = (await timeFlat()).msPerStep;
        flat.msPerStep = Math.min(flat.before, after);
        flat.tuPerSec = 1000 / (flat.msPerStep * flatPerTu);
        console.log(`    flat  after ${after.toFixed(3)} ms/step  -> scored at ${flat.msPerStep.toFixed(3)} ms/step`
          + `   ${(Wflat * Wflat / (flat.msPerStep * 1e3)).toFixed(0)} MLUPS   ${flat.tuPerSec.toFixed(2)} a/u_t per s`);
      }
      for (const r of amrRows) {
        const speedup = flat ? r.tuPerSec / flat.tuPerSec : null;
        console.log(`    AMR   t=${String(r.at).padStart(3)} a/u_t   ${r.msPerRoot.toFixed(3)} ms/root step   ${r.tuPerSec.toFixed(2)} a/u_t per s`
          + `   ${speedup ? (speedup >= 1 ? `${speedup.toFixed(2)}x FASTER` : `${(1 / speedup).toFixed(2)}x SLOWER`) + ' than flat' : ''}   tiles ${JSON.stringify(r.act)}`);
        rows.push({ q, at: r.at, speedup });
      }
    }
  } finally {
    await client.close();
    if (!o.remote) await BL.teardown({ port: o.port, tabId, chrome, server });
  }
}

main().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
