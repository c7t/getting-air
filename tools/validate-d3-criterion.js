#!/usr/bin/env node
// THE WAKE-REFINEMENT CRITERION, GATED. plans/3D.md M8.4.
//
// WHY NOT STROUHAL, WHICH IS THE OBVIOUS GATE. St under the criterion
// scatters 12% seed-to-seed at the benchmark's 12-period window -- measured,
// against the ~6% benchmarks/d3.json records for the DENSE case -- and the
// effect being looked for is about 10%. Every single-sample St comparison in
// M8.4 (dense 0.1225, refine=body 0.1307, the hand-placed wake box 0.1363)
// therefore sits inside its own error bar. Separating them needs roughly 4x
// the window, and a gate that cannot see the thing it gates is worse than no
// gate: it converts noise into a verdict.
//
// THE TWO CHECKS HERE ARE SPATIAL COUNTS AT AN INSTANT, so they do not pay
// the 1/sqrt(periods) penalty a frequency does, and they test the CRITERION
// rather than the flow it produces:
//
//   COVERAGE      every cell whose Q exceeds the threshold sits in a refined
//                 block. The field analogue of checkGeometryCoverage, and it
//                 must hold AT EVERY STEP, not at the ones the manager runs
//                 on -- the standard common_d3_manage.wgsl's geometry
//                 criterion already holds itself to. Violations are split by
//                 CAUSE: a violating cell adjacent to the refined set is
//                 convection (the structure moved off the edge); an isolated
//                 one is creation (Q rose where nothing was refined), which
//                 no lead can cover and only a shorter interval reduces.
//
//   FALSE NEGATIVES   wherever a level exists the field is in memory at TWO
//                 resolutions, so the criterion can be evaluated on the
//                 parent's field (what it actually sees) and the child's
//                 (what is actually there). Child-says-refine while
//                 parent-does-not is a miss the criterion WOULD have made had
//                 the block not already been refined. It is the only direct
//                 measure of the thing that makes under-refinement dangerous:
//                 it is SELF-CONCEALING. Fail to refine, the vortex
//                 dissipates, Q drops, and the criterion correctly reports
//                 nothing there.
//
// BOTH ARE SAMPLED ACROSS A MANAGEMENT INTERVAL AND SCORED ON THE WORST
// CASE, not the mean. "It held on average" is not what either claim says.
//
// WHAT THEY CANNOT SAY, kept attached to the numbers: the false-negative rate
// is over blocks where refinement ALREADY EXISTS. It is silent about regions
// nothing refined -- which is where a missed vortex would live. It answers
// "is the criterion's view adequate where it can be checked", not "did we
// miss one out there". Only a resolved reference answers the second.
//
// Owns the whole lifecycle, like tools/validate-3d.js and probe-d3-window.js.
//
//   node tools/validate-d3-criterion.js
//   node tools/validate-d3-criterion.js --configs=lead0
//   node tools/validate-d3-criterion.js --manageEvery=256   # make it fail

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  develop: 30,          // convective times before anything is sampled
  samples: 9,           // checkpoints across one management interval
  configs: null, manageEvery: null, extra: '',
  timeout: 900, keepOpen: false,
};

// THE BOUNDS, and they are MEASURED rather than chosen -- see the plan's M8.4
// entry for the runs they come from. Deliberately loose against the measured
// values, because a bound tight enough to fail on the flow's own variation is
// a bound that will be widened later by whoever it inconveniences.
//
//   coverage      measured 0.00-0.05% of required cells, worst case
//   false neg     measured 1.2-8.6% over the field-only population
const BOUNDS = { coverFrac: 0.01, fieldFalseNeg: 0.20 };

const CONFIGS = [
  // The reference configuration: the criterion driving a shed wake, at the
  // threshold whose default was measured against the noise floor.
  { name: 'wake-Re300', qthresh: 0.15, qlead: 0, manageEvery: 16,
    note: 'the criterion on a shed wake, zero lead' },
  // THE SAME RUN WITH THE CONVECTION LEAD ON. Whether the lead buys anything
  // has never actually been measured: the first attempt used ?qlead=0, which
  // numParam silently turned back into the default (zero is falsy), so the
  // control leg and the test leg were the same run. This is that experiment,
  // for real. NOTE the false-negative half is NOT comparable here -- the
  // criterion's per-block value is a max over the block DILATED by the lead
  // while the child's is not -- so only coverage is read from this row.
  { name: 'wake-lead', qthresh: 0.15, qlead: null, manageEvery: 16,
    coverOnly: true, note: 'the same, with the default convection lead' },
];

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--develop=')) o.develop = Number(a.slice(10));
    else if (a.startsWith('--samples=')) o.samples = parseInt(a.slice(10));
    else if (a.startsWith('--manageEvery=')) o.manageEvery = parseInt(a.slice(14));
    else if (a.startsWith('--configs=')) o.configs = a.slice(10).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function evalOrThrow(Runtime, expr, timeoutMs, what) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const pct = (x) => `${(x * 100).toFixed(2)}%`;

async function runConfig(Runtime, Page, o, c, log) {
  const G = 'window.__D3';
  const every = o.manageEvery ?? c.manageEvery;
  const q = [
    'scenario=sphere', 'n=16', 'live=0', 're=300', 'u0=0.05', 'q=19', 'bounceback=1',
    'perturb=0.02', 'seed=12345', 'levels=2', 'rb=4', 'refine=body', 'interface=explode',
    'dynamic=1', `manageEvery=${every}`, `qthresh=${c.qthresh}`, 'slots=2000',
    ...(c.qlead === null ? [] : [`qlead=${c.qlead}`]),
  ].join('&');
  const url = `${o.baseUrl}/index-3d.html?${q}${o.extra ? `&${o.extra}` : ''}`;
  await navigateTo(Page, url);
  try { await waitForGlobal(Runtime, `${G}`, 60000); }
  catch (e) {
    // The page writes its refusals into #status and never exposes __D3, so a
    // bare timeout hides the reason.
    const t = await evalOrThrow(Runtime, `document.getElementById('status').textContent`, 10000, 'status')
      .catch(() => null);
    throw new Error(t || e.message);
  }
  const p = await evalOrThrow(Runtime, `${G}.getParams()`, 30000, 'getParams');
  log(`D=${p.D} Re=${p.re} ${p.NX}x${p.NY}x${p.NZ}  qthresh=${c.qthresh}`
    + `  manageEvery=${every}  ${c.note}`);
  await evalOrThrow(Runtime, `${G}.debugStepSync(${Math.round(o.develop * p.convective)})`,
    (o.timeout + 120) * 1000, 'develop');

  // ACROSS one management interval, so the degradation BETWEEN decisions is
  // visible rather than averaged away. The worst of these is the verdict.
  const rows = [];
  const stride = Math.max(1, Math.round(every / Math.max(1, o.samples - 1)));
  for (let i = 0; i < o.samples; i++) {
    if (i) await evalOrThrow(Runtime, `${G}.debugStepSync(${stride})`, 600000, 'step');
    const cov = await evalOrThrow(Runtime, `${G}.debugCheckQCoverage()`, 600000, 'coverage');
    const miss = c.coverOnly ? null
      : await evalOrThrow(Runtime, `${G}.debugParentChildMiss(1)`, 600000, 'miss');
    rows.push({ at: i * stride, cov, miss });
  }
  const u = await evalOrThrow(Runtime, `${G}.debugPoolUsage()`, 60000, 'usage');
  const st = await evalOrThrow(Runtime, `${G}.readStats()`, 600000, 'stats');
  return { rows, usage: u, finite: st.finite, params: p, every, coverOnly: !!c.coverOnly };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = CONFIGS.filter(c => !o.configs || o.configs.includes(c.name));
  if (!configs.length) { console.error('no configs selected'); process.exit(2); }

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  await attachPageWatch(client, { onError: () => {} });

  const out = [];
  try {
    for (const c of configs) {
      console.log(`\n=== ${c.name}`);
      try {
        const r = await runConfig(Runtime, Page, o, c, s => console.log('    ' + s));
        for (const row of r.rows) {
          const m = row.miss;
          console.log(`    +${padL(row.at, 3)}  coverage ${padL(row.cov.violations, 4)}/${padL(row.cov.required, 6)}`
            + ` (${padL(pct(row.cov.frac), 7)})  conv=${padL(row.cov.convection, 4)} creat=${padL(row.cov.creation, 3)}`
            + (m ? `   falseNeg ${padL(m.fieldMissed, 3)}/${padL(m.fieldChecked, 4)}`
                 + ` (${padL(pct(m.fieldFalseNegRate), 6)})  gap ${m.fieldWorstGapOctaves.toFixed(2)} oct` : ''));
        }
        out.push({ name: c.name, ...r });
      } catch (e) {
        console.log(`    ERROR ${e.message.split('\n')[0]}`);
        out.push({ name: c.name, error: e.message });
      }
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(112));
  console.log(`SUMMARY   worst case over ${o.samples} samples spanning one management interval`);
  console.log('='.repeat(112));
  console.log(pad('config', 14) + pad('coverage worst', 16) + pad('conv/creat', 12)
    + pad('falseNeg worst', 16) + pad('gap (oct)', 11) + pad('peak slots', 12) + 'verdict');
  console.log('-'.repeat(112));
  let bad = 0;
  for (const r of out) {
    if (r.error) { console.log(pad(r.name, 14) + 'ERROR'); bad++; continue; }
    const wCov = Math.max(...r.rows.map(x => x.cov.frac));
    const conv = Math.max(...r.rows.map(x => x.cov.convection));
    const creat = Math.max(...r.rows.map(x => x.cov.creation));
    const wMiss = r.coverOnly ? null : Math.max(...r.rows.map(x => x.miss.fieldFalseNegRate));
    const wGap = r.coverOnly ? null : Math.max(...r.rows.map(x => x.miss.fieldWorstGapOctaves));
    const peak = Math.max(...r.usage.levels.map(l => l.peak));
    const fails = [];
    if (!(wCov <= BOUNDS.coverFrac)) fails.push(`coverage ${pct(wCov)} > ${pct(BOUNDS.coverFrac)}`);
    if (wMiss !== null && !(wMiss <= BOUNDS.fieldFalseNeg)) fails.push(`falseNeg ${pct(wMiss)} > ${pct(BOUNDS.fieldFalseNeg)}`);
    if (r.usage.exhausted) fails.push(`pool exhausted (${r.usage.exhausted})`);
    if (!r.finite) fails.push('non-finite');
    if (fails.length) bad++;
    console.log(pad(r.name, 14) + pad(pct(wCov), 16) + pad(`${conv}/${creat}`, 12)
      + pad(wMiss === null ? '-' : pct(wMiss), 16) + pad(wGap === null ? '-' : wGap.toFixed(2), 11)
      + pad(peak, 12) + (fails.length ? `FAIL: ${fails.join('; ')}` : 'PASS'));
  }
  console.log(`\nBounds: coverage <= ${pct(BOUNDS.coverFrac)} of required cells,`
    + ` field false-negative rate <= ${pct(BOUNDS.fieldFalseNeg)}.`);
  console.log('Both are MEASURED bounds held loosely -- see plans/3D.md M8.4. The false-negative');
  console.log('rate is over blocks where refinement ALREADY EXISTS and is silent about regions');
  console.log('nothing refined, which is where a missed vortex would live.');
  console.log('\nTo see the coverage check FAIL, lengthen the decision interval so structures');
  console.log('convect out of the refined set between decisions:  --manageEvery=256');
  process.exit(bad ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
