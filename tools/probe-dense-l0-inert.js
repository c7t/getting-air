#!/usr/bin/env node
// U7-6d: does turning the DENSE L0 off change the simulation?
//
//     node tools/probe-dense-l0-inert.js
//     node tools/probe-dense-l0-inert.js --steps=4096
//
// Under ?rootpool=1 the root pool is the authority for the renderer (U7-6b),
// the snapshot (U7-6c), level 1's ghosts and level 1's criterion. If that is
// true then ?densel0=0 -- which stops encoding the dense step, the dense
// restriction and the dense criterion -- must leave everything EXCEPT the
// dense buffers themselves bit-identical.
//
// The dense buffers are expected to differ and are excluded by name: `.fB64`
// and `.velB64` at the top level are the dense grid. `.root.*` and `.pools[]`
// are NOT excluded -- those are the claim.
const crypto = require('crypto');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const { ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown } = require('./lib/browser-lifecycle');
const REPO_ROOT = path.resolve(__dirname, '..');
let baseUrl = 'https://localhost:4444', port = 9348, N = 2048;
for (const a of process.argv.slice(2)) {
  if (a.startsWith('--baseUrl=')) baseUrl = a.slice(10);
  else if (a.startsWith('--port=')) port = parseInt(a.slice(7));
  else if (a.startsWith('--steps=')) N = parseInt(a.slice(8));
  else { console.error(`unknown argument: ${a}`); process.exit(2); }
}

const CASES = [
  { name: 'card levels=2',     page: 'index-amr.html',          q: 'levels=2&detslots=1', g: 'window.__AMR' },
  { name: 'card levels=3',     page: 'index-amr.html',          q: 'levels=3&detslots=1', g: 'window.__AMR' },
  { name: 'cylinder levels=2', page: 'index-cylinder-amr.html', q: 'levels=2&detslots=1', g: 'window.__CYL' },
  { name: 'cylinder levels=3', page: 'index-cylinder-amr.html', q: 'levels=3&detslots=1', g: 'window.__CYL' },
];

async function ev(R, e, t) { const r = await evalExpr(R, e, t); if (r.exceptionDetails) throw new Error(r.exceptionDetails.text); return r.result.value; }

// Everything except the DENSE arrays. `step` excluded as bookkeeping.
const DENSE_KEYS = new Set(['.fB64', '.velB64']);
function parts(snap) {
  const out = [];
  const walk = (n, k) => {
    if (n === null || n === undefined) return;
    if (DENSE_KEYS.has(k)) return;
    if (typeof n === 'string') { out.push([k, n]); return; }
    if (Array.isArray(n)) { n.forEach((v, i) => walk(v, `${k}[${i}]`)); return; }
    if (typeof n === 'object') { for (const q of Object.keys(n).sort()) walk(n[q], `${k}.${q}`); return; }
    out.push([k, String(n)]);
  };
  const { step, ...rest } = snap;
  walk(rest, '');
  return out;
}
const hash = (p) => crypto.createHash('sha256').update(p.map(([k, v]) => `${k}=${v}`).join('\n')).digest('hex').slice(0, 16);

(async () => {
  const server = await ensureServer(baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(port);
  const tabId = await openTab(port, 'about:blank');
  const client = await CDP({ port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable(); await Page.enable();
  Runtime.exceptionThrown(e => console.error('[exc]', e.exceptionDetails.text));
  let bad = 0;
  try {
    console.log(`  ?densel0=0 vs =1, ${N} steps, non-dense state only\n`);
    for (const c of CASES) {
      const got = {};
      for (const d of [1, 0]) {
        await navigateTo(Page, `${baseUrl}/${c.page}?${c.q}&densel0=${d}`);
        await waitForGlobal(Runtime, c.g, 60000);
        await ev(Runtime, `${c.g}.setLive(false)`);
        await ev(Runtime, `${c.g}.reset()`, 120000);
        // ASSERT THE FLAG TOOK. A URL typo would otherwise read as a clean
        // "the dense L0 is inert" pass -- the exact vacuity this project keeps
        // catching after the fact.
        const live = await ev(Runtime, `${c.g}.getDenseL0()`);
        if (live !== d) throw new Error(`${c.name}: asked densel0=${d}, page reports ${live}`);
        await ev(Runtime, `${c.g}.debugStepSync(${N})`, 900000);
        got[d] = parts(await ev(Runtime, `${c.g}.debugSnapshotSave()`, 300000));
        // ALIVE CONTROL, on the densel0=0 leg: 64 more steps must MOVE the
        // state this comparison looks at. Every row here is an EQUALITY, so a
        // filter that excluded everything meaningful would pass them all.
        if (d === 0) {
          await ev(Runtime, `${c.g}.debugStepSync(64)`, 900000);
          const moved = parts(await ev(Runtime, `${c.g}.debugSnapshotSave()`, 300000));
          if (hash(moved) === hash(got[0])) {
            throw new Error(`${c.name}: 64 more steps did not change the compared state -- `
              + 'this probe cannot see the simulation and every row above is meaningless');
          }
        }
      }
      const ok = hash(got[1]) === hash(got[0]);
      if (!ok) bad++;
      console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${c.name.padEnd(20)} densel0=1 ${hash(got[1])}  densel0=0 ${hash(got[0])}`);
      if (!ok) {
        const M = new Map(got[0]); const g = {};
        for (const [k, v] of got[1]) if (M.get(k) !== v) { const q = k.replace(/\[\d+\]/g, '[]'); g[q] = (g[q] || 0) + 1; }
        for (const [k, n] of Object.entries(g)) console.log(`         ${String(n).padStart(6)}  ${k}`);
      }
    }
  } finally { await client.close(); await teardown({ port, tabId, chrome, server }); }
  console.log(bad ? `\nFAIL: ${bad} case(s) -- the dense L0 is still feeding the solver.`
                  : '\nPASS: the dense L0 contributes nothing the root pool does not.');
  process.exit(bad ? 1 : 0);
})().catch(e => { console.error(e.message); process.exit(1); });
