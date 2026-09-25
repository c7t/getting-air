// M0 register/spill spike driver (plans/3D.md sec 2.4, sec 5 M0).
//
// Ranked risk #1 for the 3D fork: at Q=27 the fused step kernel holds 54
// live f32 in two `private` arrays indexed by a loop variable, and those
// arrays are register-resident only if the compiler fully unrolls. If it
// does not, the arrays land in scratch memory and the whole solver runs at
// a fraction of its roofline -- with no error, no wrong answer, and no way
// to tell it from "3D is just slow". This page answers that before ~8000
// lines get written on top of the assumption.
//
// It is a BENCH PAGE, not a solver: no body, no AMR, no render, no
// interactivity beyond a Run button. shaders/common_d3_spike_body.wgsl has
// the three MODEs and how to read them against each other. The short
// version: compare GB/s, not GLUPS. GLUPS is expected to fall by 27/19
// between the velocity sets simply because each cell moves more bytes;
// GB/s falling is the signal that something other than bandwidth is the
// bound, and MODE=2 (perfectly coalesced, same register pressure) falling
// while MODE=1 (same traffic, no arithmetic) holds up is the spill
// signature specifically.
//
// URL parameters:
//   ?n=128         cube edge (clamped down if the adapter's binding limit
//                  cannot hold one f buffer; the clamp is reported, never
//                  silent)
//   ?q=19|27|both  velocity set(s), default both
//   ?mode=0|1|2|all  default all
//   ?steps=20      dispatches per timed rep (forced even, so the ping-pong
//                  ends where it started)
//   ?reps=5        timed reps per measurement; the median is reported
//   ?wg=4,4,4      workgroup shape
//   ?peak=717      optional peak device bandwidth in GB/s, for a % column.
//                  Deliberately not defaulted: every conclusion this page
//                  supports is a RATIO between its own rows, and a
//                  hardcoded peak would invite reading an absolute number
//                  off a machine it was not measured on.
//   ?autorun=0     don't start automatically

import { reportFatal, reportNoWebGPU, reportNoAdapter } from './error-overlay.mjs';
import { assembleShader } from './shader-loader.mjs';
import { SUPPORTED_Q } from './lattice-3d.mjs';

const statusEl = document.getElementById('status');
const outEl = document.getElementById('out');
const runBtn = document.getElementById('run');

const urlParams = new URLSearchParams(window.location.search);
const intParam = (k, d) => (urlParams.has(k) ? (parseInt(urlParams.get(k)) || d) : d);

const REQ_N = intParam('n', 128);
const STEPS = Math.max(2, intParam('steps', 20) & ~1); // even: ping-pong returns to f_a
const REPS = Math.max(1, intParam('reps', 5));
const PEAK_GBPS = urlParams.has('peak') ? parseFloat(urlParams.get('peak')) : null;
const AUTORUN = urlParams.get('autorun') !== '0';

// The three parsers that can REJECT their input are functions, not
// module-level consts, deliberately: a throw at module scope happens before
// init()'s .catch() is attached, so a typo'd ?q= would leave #status stuck
// at its initial text with no `error:` line -- which is the one failure
// shape tools/validate-all.js's boot smoke reports as "page may be stuck"
// rather than as the bad-parameter error it actually is.
function parseQs() {
  const v = (urlParams.get('q') || 'both').toLowerCase();
  if (v === 'both' || v === 'all') return SUPPORTED_Q.slice();
  const q = parseInt(v);
  if (!SUPPORTED_Q.includes(q)) throw new Error(`?q=${v}: expected one of ${SUPPORTED_Q.join(',')} or "both"`);
  return [q];
}
function parseModes() {
  const v = (urlParams.get('mode') || 'all').toLowerCase();
  if (v === 'all') return [0, 1, 2];
  const m = v.split(',').map(Number);
  if (!m.length || m.some(x => !(x >= 0 && x <= 2))) throw new Error(`?mode=${v}: expected 0, 1, 2, a comma-separated subset, or "all"`);
  return m;
}
function parseWg() {
  const v = (urlParams.get('wg') || '4,4,4').split(',').map(Number);
  if (v.length !== 3 || v.some(n => !(n >= 1))) throw new Error('?wg= expects three positive integers, e.g. 4,4,4');
  return v;
}

let QS, MODES, WG;

const MODE_NAME = ['full', 'stream', 'collide'];

// Median, not mean: a single rep that collided with compositor work or a
// clock ramp should not move the number, and with REPS=5 there is no
// meaningful distribution to summarize beyond the middle.
function median(xs) {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]);
}

const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '-' : x.toFixed(d));

function renderTable(results, meta) {
  const cols = [
    ['Q', r => r.q], ['mode', r => MODE_NAME[r.mode]], ['N', r => r.n],
    ['GPU ms', r => fmt(r.gpuMs)], ['wall ms', r => fmt(r.wallMs)],
    ['GLUPS', r => fmt(r.glups, 3)], ['GB/s', r => fmt(r.gbps, 1)],
    ...(PEAK_GBPS ? [['% peak', r => fmt(100 * r.gbps / PEAK_GBPS, 1)]] : []),
    ['finite', r => (r.finite ? 'ok' : 'NO')],
  ];
  const head = cols.map(([h]) => `<th>${h}</th>`).join('');
  const rows = results.map(r => `<tr>${cols.map(([, f]) => `<td>${f(r)}</td>`).join('')}</tr>`).join('');
  const m = [
    `adapter: ${meta.adapter}`,
    `grid: ${meta.n}^3 = ${meta.cells.toLocaleString()} cells${meta.clampedFrom ? ` (clamped from ${meta.clampedFrom}^3 by maxStorageBufferBindingSize)` : ''}`,
    `workgroup: ${WG.join('x')}   steps/rep: ${STEPS}   reps: ${REPS}   timing: ${meta.timing}`,
  ].join('\n');
  outEl.innerHTML = `<pre>${m}</pre><table>${head ? `<tr>${head}</tr>` : ''}${rows}</table>`;
}

async function init() {
  QS = parseQs(); MODES = parseModes(); WG = parseWg();
  if (!navigator.gpu) { reportNoWebGPU(statusEl); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { reportNoAdapter(statusEl); return; }

  const hasTimestamp = adapter.features.has('timestamp-query');
  const maxQ = Math.max(...QS);

  // One `f` buffer is cells * Q * 4 bytes and there are two of them. At the
  // WebGPU SPEC MINIMUM maxStorageBufferBindingSize of 128 MiB, 128^3 D3Q19
  // (152 MiB) is already past it -- plans/3D.md sec 2.2. Same treatment as
  // main-amr.js: request exactly what is needed, capped at what the adapter
  // really has, and if even that is not enough, step the grid DOWN and say
  // so rather than failing or, worse, quietly measuring a different problem.
  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const bytesFor = (n) => n * n * n * maxQ * 4;
  const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  let n = REQ_N;
  while (n > 8 && bytesFor(n) > limit) n -= 8;
  const clampedFrom = n !== REQ_N ? REQ_N : null;
  if (bytesFor(n) > limit) {
    statusEl.textContent = `error: even ${n}^3 D3Q${maxQ} needs ${(bytesFor(n) / 1048576).toFixed(0)} MiB per binding, this GPU allows ${(limit / 1048576).toFixed(0)} MiB`;
    return;
  }

  const needed = bytesFor(n);
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamp ? ['timestamp-query'] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(Math.max(needed, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(Math.max(needed, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
    },
  });
  device.lost.then((info) => {
    if (info.reason === 'destroyed') return;
    statusEl.textContent = `error: GPU device lost (${info.reason}): ${info.message}`;
  });

  const cells = n * n * n;
  const info = adapter.info || {};
  const adapterStr = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / ') || 'unknown';

  const querySet = hasTimestamp ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const queryResolve = hasTimestamp ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const queryRead = hasTimestamp ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;

  // Finiteness smoke: a kernel that faulted its way into NaN can be FASTER
  // (denormal/NaN paths aside, a wrong kernel is not a measurement), so
  // every row carries a sanity readback rather than trusting the timing
  // alone. 4 KiB off the front of plane 0 is enough to catch it.
  const SANITY_BYTES = 4096;
  const sanityBuf = device.createBuffer({ size: SANITY_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  // Buffers sized for the LARGEST velocity set being run, so both Q share
  // one allocation. A Q19 run then addresses the first 19 planes of a
  // 27-plane buffer, which is fine (the plane stride is NCELLS either way)
  // and avoids destroying/reallocating half a gigabyte between rows.
  const bufBytes = cells * maxQ * 4;
  const fA = device.createBuffer({ size: bufBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const fB = device.createBuffer({ size: bufBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bgAB = device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: fA } }, { binding: 1, resource: { buffer: fB } }] });
  const bgBA = device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: fA } }] });

  const modules = {};
  for (const q of QS) {
    const src = await assembleShader(`shaders/d3_spike_q${q}.wgsl`, (p) => fetch(p).then(r => {
      if (!r.ok) throw new Error(`failed to fetch ${p}: ${r.status}`);
      return r.text();
    }));
    modules[q] = device.createShaderModule({ code: src, label: `d3_spike_q${q}` });
  }

  const baseConstants = { NX: n, NY: n, NZ: n, WGX: WG[0], WGY: WG[1], WGZ: WG[2] };
  const disp = [Math.ceil(n / WG[0]), Math.ceil(n / WG[1]), Math.ceil(n / WG[2])];

  const initPipe = {};
  for (const q of QS) {
    initPipe[q] = await device.createComputePipelineAsync({
      layout, compute: { module: modules[q], entryPoint: 'init', constants: baseConstants },
    });
  }

  async function readSanity(buf) {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, sanityBuf, 0, SANITY_BYTES);
    device.queue.submit([enc.finish()]);
    await sanityBuf.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(sanityBuf.getMappedRange().slice(0));
    sanityBuf.unmap();
    return v.every(x => isFinite(x) && Math.abs(x) < 10);
  }

  async function seed(q) {
    const enc = device.createCommandEncoder();
    for (const bg of [bgAB, bgBA]) {
      const pass = enc.beginComputePass();
      pass.setPipeline(initPipe[q]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(disp[0], disp[1], disp[2]);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // One timed rep: STEPS dispatches inside ONE compute pass, one submit.
  // The pass boundary is where the timestamps go (encoder.writeTimestamp is
  // gone from WebGPU -- see main-amr.js's note), so a rep is the smallest
  // unit that can be timed on the GPU clock, and STEPS is what makes the
  // per-submit overhead negligible against it.
  async function timedRep(pipe) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(hasTimestamp
      ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
      : {});
    pass.setPipeline(pipe);
    for (let s = 0; s < STEPS; s++) {
      pass.setBindGroup(0, s % 2 === 0 ? bgAB : bgBA);
      pass.dispatchWorkgroups(disp[0], disp[1], disp[2]);
    }
    pass.end();
    if (hasTimestamp) {
      enc.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
      enc.copyBufferToBuffer(queryResolve, 0, queryRead, 0, 16);
    }
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - t0;
    let gpuMs = null;
    if (hasTimestamp) {
      await queryRead.mapAsync(GPUMapMode.READ);
      const t = new BigInt64Array(queryRead.getMappedRange().slice(0));
      queryRead.unmap();
      gpuMs = Number(t[1] - t[0]) / 1e6;
    }
    return { gpuMs, wallMs };
  }

  async function measure(q, mode) {
    const pipe = await device.createComputePipelineAsync({
      layout,
      compute: { module: modules[q], entryPoint: 'step', constants: { ...baseConstants, MODE: mode } },
    });
    await seed(q);
    await timedRep(pipe); // warmup: pipeline residency, clock ramp, first-touch
    const gpu = [], wall = [];
    for (let r = 0; r < REPS; r++) {
      const t = await timedRep(pipe);
      if (t.gpuMs != null) gpu.push(t.gpuMs);
      wall.push(t.wallMs);
    }
    const gpuMs = gpu.length ? median(gpu) : null;
    const wallMs = median(wall);
    const ms = gpuMs != null ? gpuMs : wallMs;
    // Traffic model: every mode reads Q planes and writes Q planes per cell,
    // f32. That is the roofline plans/3D.md sec 3.2 prices (152 B/cell at
    // Q19, 216 at Q27) and it is identical across the three modes, which is
    // the whole reason they are comparable in GB/s.
    const updates = cells * STEPS;
    return {
      q, mode, n, cells, steps: STEPS, reps: REPS, gpuMs, wallMs,
      glups: updates / ms / 1e6,
      gbps: (updates * q * 4 * 2) / ms / 1e6,
      finite: await readSanity(fA),
    };
  }

  const meta = { adapter: adapterStr, n, cells, clampedFrom, timing: hasTimestamp ? 'timestamp-query' : 'wall clock (no timestamp-query on this adapter)' };
  const results = [];
  window.__SPIKE3D = { ready: true, meta, results, params: { n, QS, MODES, STEPS, REPS, WG, PEAK_GBPS }, running: false, done: false };

  async function runAll() {
    if (window.__SPIKE3D.running) return window.__SPIKE3D;
    window.__SPIKE3D.running = true;
    window.__SPIKE3D.done = false;
    results.length = 0;
    runBtn.disabled = true;
    for (const q of QS) {
      for (const mode of MODES) {
        statusEl.textContent = `measuring D3Q${q} ${MODE_NAME[mode]} at ${n}^3 ...`;
        results.push(await measure(q, mode));
        renderTable(results, meta);
      }
    }
    window.__SPIKE3D.running = false;
    window.__SPIKE3D.done = true;
    runBtn.disabled = false;
    statusEl.textContent = `done: ${results.length} measurement(s) at ${n}^3, ${STEPS} steps x ${REPS} reps`;
    return window.__SPIKE3D;
  }
  window.__SPIKE3D.run = runAll;

  runBtn.onclick = () => { runAll(); };
  statusEl.textContent = `ready: ${n}^3, D3Q${QS.join('/')} , modes ${MODES.map(m => MODE_NAME[m]).join('/')}`;
  renderTable([], meta);
  if (AUTORUN) await runAll();
}

init().catch(e => reportFatal(statusEl, e));
