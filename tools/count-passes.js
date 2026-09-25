#!/usr/bin/env node
// Every compute pass an index-amr.html root step encodes, named by shader file
// and entry point, with how many per root step and how many workgroups each.
//
// WHY. The desktop is pass-count bound (plans/performance-snapshot.md: a root
// step costs ~7.5 us x passes there), so the pass list IS its cost model, and
// the per-pass GPU profile cannot give it: debugProfileMacroStep groups by
// label (the level-3 step, run 8 times, appears once) and the shared
// explode/coalesce seam opens its passes unlabelled. This counts what is
// actually encoded instead.
//
// HOW. A script injected before the page loads names every compute pipeline at
// creation -- "<entry shader file>:<entry point>[key overrides]" -- by matching
// the assembled module's first line against shaders/*.wgsl's (an assembled
// module starts with its entry file). Then beginComputePass/setPipeline/
// dispatchWorkgroups are wrapped for one 64-root-step debugStepSync batch.
// Refine rounds run every 16 root steps, so their passes show as fractions.
//
//   node tools/count-passes.js                                   # page defaults
//   node tools/count-passes.js 'interface=explode&res=5&levels=4&spongeW=2'
//   node tools/count-passes.js --remote --port=9229 --baseUrl=https://era:4471 '...'

const fs = require('fs');
const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const BL = require('./lib/browser-lifecycle');
const REPO_ROOT = path.join(__dirname, '..');

function parseArgs(argv) {
  const o = { baseUrl: 'https://localhost:4444', port: 9357, remote: false, warm: 2048, configs: [] };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = Number(a.slice(7));
    else if (a === '--remote') o.remote = true;
    else if (a.startsWith('--warm=')) o.warm = Number(a.slice(7));
    else if (a.startsWith('--')) { console.error(`unknown flag ${a}`); process.exit(2); }
    else o.configs.push(a);
  }
  if (!o.configs.length) o.configs.push('');
  return o;
}

function hookSource() {
  const SD = path.join(REPO_ROOT, 'shaders');
  const FIRST = {};
  for (const f of fs.readdirSync(SD)) {
    if (f.endsWith('.wgsl') && !f.startsWith('common_')) FIRST[fs.readFileSync(path.join(SD, f), 'utf8').split('\n')[0]] = f.replace('.wgsl', '');
  }
  return `(() => {
    const FIRST = ${JSON.stringify(FIRST)};
    const D = GPUDevice.prototype, cSM = D.createShaderModule, cCP = D.createComputePipeline;
    const modName = new WeakMap();
    D.createShaderModule = function (d) { const m = cSM.call(this, d); modName.set(m, FIRST[(d.code || '').split('\\n')[0]] || '?'); return m; };
    const KEYS = ['GHOST_ONLY', 'FINE_FINE_ONLY', 'NOOP', 'PARENT_GHOST', 'SKIP_GHOST', 'DIRECT_GHOST'];
    D.createComputePipeline = function (d) {
      const p = cCP.call(this, d), c = d.compute.constants || {};
      const k = KEYS.filter(x => x in c).map(x => x + '=' + c[x]).join(',');
      p.__name = (modName.get(d.compute.module) || '?') + ':' + (d.compute.entryPoint || 'main') + (k ? '[' + k + ']' : '');
      return p;
    };
  })();`;
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  let server = null, chrome = null, tabId;
  if (o.remote) {
    const t = (await CDP.List({ port: o.port })).find(x => x.type === 'page' && x.url.startsWith(o.baseUrl));
    if (!t) throw new Error(`--remote: no tab on ${o.baseUrl} at port ${o.port}`);
    tabId = t.id;
  } else {
    server = await BL.ensureServer(o.baseUrl, REPO_ROOT);
    chrome = await BL.ensureChrome(o.port);
    tabId = await BL.openTab(o.port, 'about:blank');
  }
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Page, Runtime } = client;
  await Page.enable(); await Runtime.enable();
  const { identifier } = await Page.addScriptToEvaluateOnNewDocument({ source: hookSource() });
  const ev = async (e) => {
    const r = await BL.evalExpr(Runtime, e, 900000);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  try {
    for (const q of o.configs) {
      await BL.navigateTo(Page, `${o.baseUrl}/index-amr.html?detslots=1&${q}`);
      await BL.waitForGlobal(Runtime, 'window.__AMR', 120000);
      const r = await ev(`(async()=>{const A=__AMR; A.setLive(false); await A.reset(); await A.debugStepSync(${o.warm});
        const E=GPUCommandEncoder.prototype, P=GPUComputePassEncoder.prototype;
        const k={b:E.beginComputePass, s:P.setPipeline, d:P.dispatchWorkgroups};
        const by=new Map(); let cur='?', passes=0;
        E.beginComputePass=function(d){passes++; return k.b.call(this,d);};
        P.setPipeline=function(pl){cur=pl.__name||'?'; return k.s.call(this,pl);};
        P.dispatchWorkgroups=function(x,y=1,z=1){const e=by.get(cur)||{n:0,wg:0}; e.n++; e.wg+=x*y*z; by.set(cur,e); return k.d.call(this,x,y,z);};
        try { await A.debugStepSync(64); } finally { E.beginComputePass=k.b; P.setPipeline=k.s; P.dispatchWorkgroups=k.d; }
        const t=performance.now(); await A.debugStepSync(1024); const wall=(performance.now()-t)/1024;
        const act={}; for(let m=1;m<A.getNumLevels();m++) act[m]=(await A.debugListActiveBlocks(m)).length;
        return {wall, passes:passes/64, act, rows:[...by].map(([n,v])=>[n, v.n/64, Math.round(v.wg/v.n)])};})()`);
      console.log(`\n=== index-amr.html?${q || '(defaults)'}  active tiles ${JSON.stringify(r.act)}`);
      console.log(`    ${r.passes.toFixed(1)} passes per root step, ${r.wall.toFixed(3)} ms per root step => ${(r.wall * 1000 / r.passes).toFixed(1)} us per pass`);
      r.rows.sort((a, b) => b[1] - a[1]);
      for (const [n, per, wg] of r.rows) {
        const f = per >= 1 ? String(+per.toFixed(2)) : '1/' + Math.round(1 / per);
        console.log(`   ${f.padStart(6)} x  ~${String(wg).padStart(5)} wg   ${n}`);
      }
    }
  } finally {
    await Page.removeScriptToEvaluateOnNewDocument({ identifier }).catch(() => {});
    await client.close();
    if (!o.remote) await BL.teardown({ port: o.port, tabId, chrome, server });
  }
}

main().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
