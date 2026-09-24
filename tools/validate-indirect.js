#!/usr/bin/env node
// ?indirect=1 must be BIT-IDENTICAL to ?indirect=0 -- a gate, not a tolerance.
//
// WHY IT CAN BE EXACT. Indirect dispatch changes only WHICH z launches a tile
// (z indexes shaders/amr_active_list.wgsl's compacted list instead of the
// pool). Every per-slot kernel's work is a function of its slot alone, and the
// one cross-tile reduction -- the force pass's per-workgroup partials -- is
// summed with INTEGER atomics, which are order-independent. So with the slot
// HANDOUT deterministic (`?detslots=1`; the default atomicSub free list races,
// see CLAUDE.md on attractors), the two runs must agree to the bit. Anything
// less is a list that is stale, short, or pointing at the wrong pool.
//
// ── THE PROTOCOL, per configuration ─────────────────────────────────────────
//
//   direct    ?indirect=0, reset, N steps, snapshot
//   direct'   the same again                       -- THE CONTROL
//   indirect  ?indirect=1, reset, N steps (in chunks, with the GPU list scored
//             against amr2d.mjs's activeSlotList after each), snapshot
//
//   require   direct == direct'     else the comparison below cannot mean anything
//             direct == indirect    the gate
//             stepping moved the state, and the finest level holds tiles --
//             or the equality was measured on a run with nothing to launch
//
//     node tools/validate-indirect.js
//     node tools/validate-indirect.js --steps=2048 --only=explode
//
// VERIFY WHICH TREE THE DEV SERVER IS SERVING first -- `ensureServer` reuses
// whatever already answers on the port.

const path = require('path');
const crypto = require('crypto');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, navigateTo, evalExpr, waitForGlobal, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.resolve(__dirname, '..');

// The phone's configuration first -- the one the measurement was taken on --
// then both interfaces at the shipped depth, so the interp path's
// interp/average twins are exercised as well as explode/coalesce.
const CONFIGS = [
  { name: 'explode res5 L4', q: 'interface=explode&res=5&levels=4&spongeW=2' },
  { name: 'explode L3',      q: 'interface=explode&levels=3' },
  { name: 'interp L3',       q: 'interface=interp&levels=3' },
  { name: 'interp L2',       q: 'interface=interp&levels=2' },
];

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9348, steps: 1024, chunk: 128, only: null };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--chunk=')) o.chunk = parseInt(a.slice(8));
    else if (a.startsWith('--only=')) o.only = a.slice(7);
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  if (o.steps % o.chunk !== 0) { console.error('--steps must be a multiple of --chunk'); process.exit(2); }
  return o;
}

async function ev(R, e, t) {
  const r = await evalExpr(R, e, t);
  if (r.exceptionDetails) throw new Error(`page threw: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
}

// Everything in the snapshot but `step`, in stable key order.
function parts(snap) {
  const out = [];
  const walk = (n, k) => {
    if (n === null || n === undefined) return;
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
function diffShape(a, b) {
  const B = new Map(b), g = {};
  for (const [k, v] of a) if (B.get(k) !== v) { const q = k.replace(/\[\d+\]/g, '[]'); g[q] = (g[q] || 0) + 1; }
  return g;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const configs = o.only ? CONFIGS.filter(c => c.name.includes(o.only)) : CONFIGS;
  if (!configs.length) { console.error(`--only=${o.only} matched no configuration`); process.exit(2); }

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  const tabId = await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  Runtime.exceptionThrown(e => console.error('[browser exception]', e.exceptionDetails.text));
  const G = 'window.__AMR';

  const leg = async (cfg, indirect) => {
    await navigateTo(Page, `${o.baseUrl}/index-amr.html?${cfg.q}&detslots=1&indirect=${indirect ? 1 : 0}`);
    await waitForGlobal(Runtime, G, 60000);
    await ev(Runtime, `${G}.setLive(false)`);
    if ((await ev(Runtime, `${G}.isIndirect()`)) !== indirect) throw new Error(`page did not take ?indirect=${indirect ? 1 : 0}`);
    await ev(Runtime, `${G}.reset()`, 120000);
    const start = parts(await ev(Runtime, `${G}.debugSnapshotSave()`, 300000));
    const listFailures = [];
    let listChecks = 0;
    for (let done = 0; done < o.steps; done += o.chunk) {
      await ev(Runtime, `${G}.debugStepSync(${o.chunk})`, 900000);
      await ev(Runtime, `${G}.setLive(false)`);
      if (indirect) {
        const r = await ev(Runtime, `${G}.debugCheckActiveLists()`, 60000);
        listChecks++;
        if (!r.ok) listFailures.push({ atStep: done + o.chunk, levels: r.levels.filter(l => !l.ok) });
      }
    }
    const end = parts(await ev(Runtime, `${G}.debugSnapshotSave()`, 300000));
    const nLevels = await ev(Runtime, `${G}.getNumLevels()`);
    const finestTiles = (await ev(Runtime, `${G}.debugListActiveBlocks(${nLevels - 1})`, 60000)).length;
    return { start, end, listFailures, listChecks, finestTiles };
  };

  const rows = [];
  try {
    console.log(`  ?indirect=1 vs ?indirect=0, ?detslots=1, ${o.steps} steps from reset()\n`);
    for (const cfg of configs) {
      const d1 = await leg(cfg, false);
      const d2 = await leg(cfg, false);
      const ind = await leg(cfg, true);
      const deterministic = hash(d1.end) === hash(d2.end);
      const same = hash(d1.end) === hash(ind.end);
      const alive = hash(d1.end) !== hash(d1.start);
      const refined = d1.finestTiles > 0 && ind.finestTiles > 0;
      const listsOk = ind.listFailures.length === 0 && ind.listChecks > 0;
      const ok = deterministic && same && alive && refined && listsOk;
      rows.push({ cfg, ok });
      const tag = !deterministic ? 'NDET' : !alive ? 'DEAD' : !refined ? 'NOTL' : ok ? 'ok  ' : 'FAIL';
      console.log(`  ${tag} ${cfg.name.padEnd(18)} direct ${hash(d1.end)}  direct' ${hash(d2.end)}  indirect ${hash(ind.end)}`
        + `  finest tiles ${d1.finestTiles}/${ind.finestTiles}  lists ${ind.listChecks - ind.listFailures.length}/${ind.listChecks}`);
      if (!deterministic) console.log('       the direct legs disagree with EACH OTHER -- detslots did not make the run deterministic, so no comparison here means anything');
      if (!alive) console.log('       stepping did not change the snapshot -- the equality below it proves nothing');
      if (!refined) console.log('       the finest level holds no tiles -- indirect dispatch had nothing to launch');
      if (deterministic && !same) for (const [k, n] of Object.entries(diffShape(d1.end, ind.end))) console.log(`       ${String(n).padStart(8)}  ${k}`);
      for (const f of ind.listFailures.slice(0, 3)) console.log(`       list wrong at step ${f.atStep}: ${JSON.stringify(f.levels)}`);
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server });
  }

  const bad = rows.filter(r => !r.ok);
  console.log('');
  if (bad.length) { console.log(`FAIL: ${bad.length} of ${rows.length} configuration(s).`); process.exit(1); }
  console.log(`PASS: ?indirect=1 is bit-identical to ?indirect=0 on ${rows.length} configuration(s), with a`);
  console.log('deterministic control, a moving state, a refined finest level, and every GPU list matching its host twin.');
  process.exit(0);
}

main().catch(e => { console.error('\n' + e.message); process.exit(1); });
