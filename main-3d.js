// Dense 3D LBM page -- plans/3D.md M1. D3Q19 by default, D3Q27 via ?q=27,
// no AMR, no body, axis-aligned slice view.
//
// ONE PAGE, THREE SCENARIOS, deliberately. The 2D side of this project has
// index.html / index-cylinder.html / index-channel.html / index-tgv.html,
// each with its own main*.js carrying its own copy of the same bind
// groups -- and CLAUDE.md records what that cost: commit 238e48c changed a
// shared shader's binding count, updated one copy of the bind group, missed
// the other, and broke a production page that no test had ever loaded. A 3D
// fork doubles that surface (plans/3D.md sec 6). So the scenario -- duct,
// beltrami, tgv -- is a URL parameter over ONE set of pipelines and ONE set
// of bind groups, and it differs only in the initial condition, the wall
// flags and the body force, all of which come from d3-scenarios.mjs. Both
// analytic gates and the interactive view therefore exercise the identical
// code path, which is the property the 2D layout does not have.
//
// URL parameters:
//   ?scenario=beltrami|duct|tgv   default beltrami
//   ?n=48        cube edge (clamped down if the adapter cannot bind one f)
//   ?q=19|27     velocity set, default 19
//   ?tau=0.8     BGK relaxation time; nu = (tau - 1/2)/3
//   ?u0=         scenario amplitude (duct: target peak velocity)
//   ?axis=x|y|z  slice normal, default z      ?slice=   index along it
//   ?mode=       speed | vorticity | normal (or 0 | 1 | 2). `vorticity` is
//                the out-of-plane component; `normal` is the velocity along
//                the slice normal, which is what shows a duct's u_x profile.
//   ?uscale= ?vscale= ?vortGamma=   render normalization overrides
//   ?spf=        solver steps per displayed frame
//   ?live=0      start paused (validation drives debugStepSync instead)

import { reportFatal, reportNoWebGPU, reportNoAdapter } from './error-overlay.mjs';
import { assembleShader } from './shader-loader.mjs';
import { SUPPORTED_Q } from './lattice-3d.mjs';
import { SCENARIOS, SCENARIO_NAMES, resolveScenario, nuFromTau } from './d3-scenarios.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
const numParam = (k, d) => (urlParams.has(k) ? (parseFloat(urlParams.get(k)) || d) : d);
const AXIS_NAMES = ['x', 'y', 'z'];

// Parsers that can REJECT live in functions, not module-scope consts: a
// throw at module scope happens before init()'s .catch() is attached, and
// tools/validate-all.js's boot smoke would then report "page may be stuck"
// instead of the bad-parameter error it actually is.
function parseScenarioName() {
  const v = urlParams.get('scenario') || 'beltrami';
  if (!SCENARIO_NAMES.includes(v)) throw new Error(`?scenario=${v}: expected one of ${SCENARIO_NAMES.join(', ')}`);
  return v;
}
function parseQ() {
  const q = urlParams.has('q') ? parseInt(urlParams.get('q')) : 19;
  if (!SUPPORTED_Q.includes(q)) throw new Error(`?q=${urlParams.get('q')}: expected one of ${SUPPORTED_Q.join(', ')}`);
  return q;
}
// Both forms accepted, on purpose: the UI's <select> carries numeric
// values, the URL reads better as a letter, and a validation config is
// clearer with a name. Rejecting one of them buys nothing and is a papercut
// whichever one you happen to have in hand.
function parseAxis() {
  const v = (urlParams.get('axis') || 'z').toLowerCase();
  const i = /^[0-2]$/.test(v) ? parseInt(v) : AXIS_NAMES.indexOf(v);
  if (i < 0) throw new Error(`?axis=${v}: expected x, y, z, or 0, 1, 2`);
  return i;
}

const MODE_NAMES = ['speed', 'vorticity', 'normal'];
// The default depends on the scenario, and has to: a duct's flow is
// unidirectional, so an x-normal slice has no in-plane velocity and
// therefore no out-of-plane vorticity either -- defaulting it to
// `vorticity` would render a correct simulation as a black canvas, which
// is the exact signal this project treats as a FAILURE (see the
// webgpu-verify skill). `normal` shows its u_x profile instead.
function defaultMode(scenarioName) { return scenarioName === 'duct' ? 'normal' : 'vorticity'; }
function parseMode(scenarioName) {
  const v = (urlParams.get('mode') || defaultMode(scenarioName)).toLowerCase();
  const i = /^[0-2]$/.test(v) ? parseInt(v) : MODE_NAMES.indexOf(v);
  if (i < 0) throw new Error(`?mode=${v}: expected ${MODE_NAMES.join(', ')}, or 0, 1, 2`);
  return i;
}

// Stride that keeps a full-field readback to a manageable number of points
// while still sampling the whole volume. Must DIVIDE N, so the sampled
// sites are exact lattice sites and the analytic reference can be evaluated
// at them with no interpolation.
function subsampleStride(N, maxPerAxis) {
  for (let s = Math.max(1, Math.ceil(N / maxPerAxis)); s <= N; s++) if (N % s === 0) return s;
  return N;
}

async function init() {
  const scenarioName = parseScenarioName();
  const Q = parseQ();
  const params = resolveScenario(scenarioName, {
    n: urlParams.has('n') ? parseInt(urlParams.get('n')) : null,
    tau: urlParams.has('tau') ? parseFloat(urlParams.get('tau')) : null,
    u0: urlParams.has('u0') ? parseFloat(urlParams.get('u0')) : null,
  });
  const sc = SCENARIOS[scenarioName];

  if (!navigator.gpu) { reportNoWebGPU(statusEl); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { reportNoAdapter(statusEl); return; }

  // Same raised-limit pattern as main-amr.js: WebGPU devices default to the
  // SPEC MINIMUM (128 MiB storage binding) no matter what the adapter can
  // really do, and 128^3 D3Q19 is 152 MiB -- already past it (plans/3D.md
  // sec 2.2). Request what is needed, capped at the adapter's real
  // capability, and fail with a legible message rather than a validation
  // error if this GPU genuinely cannot.
  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const N = params.n;
  const NCELLS = N * N * N;
  const fBytes = NCELLS * Q * 4;
  const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  if (fBytes > limit) {
    statusEl.textContent = `error: ${N}^3 D3Q${Q} needs a ${(fBytes / 1048576).toFixed(0)} MiB binding, this GPU's max is ${(limit / 1048576).toFixed(0)} MiB`;
    return;
  }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(Math.max(fBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(Math.max(fBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
    },
  });
  device.lost.then((info) => {
    if (info.reason === 'destroyed') return;
    statusEl.textContent = `error: GPU device lost (${info.reason}): ${info.message}`;
  });

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const U = GPUBufferUsage;
  const fA  = device.createBuffer({ size: fBytes, usage: U.STORAGE });
  const fB  = device.createBuffer({ size: fBytes, usage: U.STORAGE });
  const mac = device.createBuffer({ size: NCELLS * 4 * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const rpBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });

  const computeBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const renderBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
  ]});
  const bgAB = device.createBindGroup({ layout: computeBGL, entries: [
    { binding: 0, resource: { buffer: fA } }, { binding: 1, resource: { buffer: fB } }, { binding: 2, resource: { buffer: mac } }]});
  const bgBA = device.createBindGroup({ layout: computeBGL, entries: [
    { binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: fA } }, { binding: 2, resource: { buffer: mac } }]});
  const renderBG = device.createBindGroup({ layout: renderBGL, entries: [
    { binding: 0, resource: { buffer: mac } }, { binding: 1, resource: { buffer: rpBuf } }]});

  const loadShader = (p) => assembleShader(p, (f) => fetch(f).then(r => {
    if (!r.ok) throw new Error(`failed to fetch ${f}: ${r.status}`);
    return r.text();
  }));
  const stepModule = device.createShaderModule({ code: await loadShader(`shaders/d3_step_q${Q}.wgsl`), label: `d3_step_q${Q}` });
  const renderModule = device.createShaderModule({ code: await loadShader('shaders/d3_render_slice.wgsl'), label: 'd3_render_slice' });

  const WG = [4, 4, 4];
  const disp = [Math.ceil(N / WG[0]), Math.ceil(N / WG[1]), Math.ceil(N / WG[2])];
  const dims = { NX: N, NY: N, NZ: N };
  const stepConstants = {
    ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2],
    OMEGA: 1 / params.tau,
    FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
    WALL_X: params.walls.includes('x') ? 1 : 0,
    WALL_Y: params.walls.includes('y') ? 1 : 0,
    WALL_Z: params.walls.includes('z') ? 1 : 0,
  };
  const computePL = device.createPipelineLayout({ bindGroupLayouts: [computeBGL] });
  const initPipe = await device.createComputePipelineAsync({ layout: computePL, compute: { module: stepModule, entryPoint: 'initEq', constants: stepConstants } });
  const stepPipe = await device.createComputePipelineAsync({ layout: computePL, compute: { module: stepModule, entryPoint: 'step', constants: stepConstants } });
  const renderPipe = await device.createRenderPipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: { module: renderModule, entryPoint: 'vs_main', constants: dims },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format }], constants: {
      ...dims,
      // Pinned to 1 because the page pre-normalizes -- see U_SCALE/V_SCALE.
      VORT_SCALE: 1.0,
      // Below 1 lifts weak structure toward the top of the ramp instead of
      // leaving it in the near-black background; these fields are smooth and
      // have no vortex cores to protect from saturating.
      VORT_GAMMA: numParam('vortGamma', 0.7),
    } },
    primitive: { topology: 'triangle-list' },
  });

  // --- state ---------------------------------------------------------------
  let step = 0;
  let useB = false;               // true => the CURRENT field is in fB
  let live = urlParams.get('live') !== '0';
  let axis = parseAxis();
  let slice = urlParams.has('slice') ? Math.min(N - 1, Math.max(0, parseInt(urlParams.get('slice')))) : (N >> 1);
  let mode = parseMode(scenarioName);
  const STEPS_PER_FRAME = Math.max(1, Math.round(numParam('spf', Math.max(1, Math.round(262144 / NCELLS * 8)))));
  // Render normalizations, derived per scenario rather than inherited from
  // the 2D pages' constants. common_vortcolor.wgsl's tone curve is
  // calibrated for a shedding wake at |omega| ~ 0.05; every field here is
  // an order of magnitude gentler (a decaying Beltrami flow has
  // |omega| = k|u| ~ 0.008), so the page divides by a reference and hands
  // the shader a dimensionless ratio, with VORT_SCALE pinned to 1.
  //
  //   duct      u peaks at its target; shear runs across the half-width a.
  //   beltrami  |u| reaches ~2.4*u0 (three components of sin+cos), and the
  //             field is Beltrami so |omega| = k|u| exactly.
  //   tgv       same wavenumber, amplitude u0 per component.
  const U_SCALE = numParam('uscale', scenarioName === 'duct' ? params.uPeak : 2 * params.u0);
  const V_SCALE = numParam('vscale',
    scenarioName === 'duct' ? 2 * params.uPeak / params.a
      : (params.k || 2 * Math.PI / N) * U_SCALE);

  function writeRenderParams() {
    const b = new ArrayBuffer(32);
    new Uint32Array(b, 0, 4).set([axis, slice, mode, 0]);
    new Float32Array(b, 16, 4).set([U_SCALE, V_SCALE, 0, 0]);
    device.queue.writeBuffer(rpBuf, 0, b);
  }

  // Seed: upload the scenario's macroscopic initial condition and let the
  // GPU turn it into an equilibrium `f`. Both buffers are seeded so the
  // ping-pong parity does not matter after a reset.
  function reset() {
    device.queue.writeBuffer(mac, 0, sc.macro(N, params));
    const enc = device.createCommandEncoder();
    for (const bg of [bgAB, bgBA]) {
      const p = enc.beginComputePass();
      p.setPipeline(initPipe); p.setBindGroup(0, bg);
      p.dispatchWorkgroups(disp[0], disp[1], disp[2]);
      p.end();
    }
    device.queue.submit([enc.finish()]);
    // initEq READ mac and did not write it, so mac still holds the seed --
    // which is exactly the field the renderer should show at step 0.
    step = 0; useB = false;
  }

  function encodeSteps(enc, n) {
    const p = enc.beginComputePass();
    p.setPipeline(stepPipe);
    for (let s = 0; s < n; s++) {
      p.setBindGroup(0, useB ? bgBA : bgAB);
      p.dispatchWorkgroups(disp[0], disp[1], disp[2]);
      useB = !useB;
    }
    p.end();
    step += n;
  }

  // Chunked so a long run is not one enormous command buffer (the duct
  // needs O(10^4) steps to settle), and awaited per chunk so the queue
  // cannot run arbitrarily far ahead of the page.
  const SYNC_CHUNK = 500;
  async function debugStepSync(n) {
    live = false;
    playBtn.textContent = 'play';
    let done = 0;
    while (done < n) {
      const k = Math.min(SYNC_CHUNK, n - done);
      const enc = device.createCommandEncoder();
      encodeSteps(enc, k);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      done += k;
    }
    return { step };
  }

  // --- readback ------------------------------------------------------------
  const staging = device.createBuffer({ size: NCELLS * 4 * 4, usage: U.MAP_READ | U.COPY_DST });
  async function readMacro() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(mac, 0, staging, 0, NCELLS * 4 * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    return v;
  }

  // Strided full-volume sample. Returned to the validation tools instead of
  // the whole field because the whole field does not survive the trip: at
  // 48^3 it is 442k numbers, and CDP returns values as JSON. The stride
  // DIVIDES N, so every returned point is an exact lattice site and the
  // analytic reference needs no interpolation.
  async function readSubsampled(maxPerAxis) {
    const s = subsampleStride(N, maxPerAxis || 24);
    const m = await readMacro();
    const M = N / s;
    const out = { n: N, stride: s, m: M, rho: [], ux: [], uy: [], uz: [] };
    for (let z = 0; z < N; z += s) {
      for (let y = 0; y < N; y += s) {
        for (let x = 0; x < N; x += s) {
          const c = (z * N + y) * N + x;
          out.rho.push(m[4 * c]); out.ux.push(m[4 * c + 1]); out.uy.push(m[4 * c + 2]); out.uz.push(m[4 * c + 3]);
        }
      }
    }
    return out;
  }

  // Duct cross-section: u_x averaged along the periodic x axis, indexed
  // [z * N + y] to match d3-scenarios.mjs's ductProfile(). Averaging over x
  // is not just noise reduction -- a steady duct flow is x-invariant by
  // construction, so the SPREAD across x is itself a check that the run has
  // settled and that the periodic direction is behaving.
  async function readDuctProfile() {
    const m = await readMacro();
    const prof = new Array(N * N).fill(0);
    let maxSpread = 0;
    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        let sum = 0, lo = Infinity, hi = -Infinity;
        for (let x = 0; x < N; x++) {
          const v = m[4 * ((z * N + y) * N + x) + 1];
          sum += v; if (v < lo) lo = v; if (v > hi) hi = v;
        }
        prof[z * N + y] = sum / N;
        if (hi - lo > maxSpread) maxSpread = hi - lo;
      }
    }
    return { n: N, profile: prof, maxXSpread: maxSpread };
  }

  // Volume-averaged kinetic energy and enstrophy. Enstrophy is what the 3D
  // TGV benchmark's dissipation rate is built from (eps = 2 nu <omega^2/2>
  // for an incompressible periodic flow), so this is the reporting surface
  // for the tgv scenario -- see d3-scenarios.mjs on why tgv reports rather
  // than gates.
  async function readStats() {
    const m = await readMacro();
    const at = (x, y, z, c) => m[4 * ((((z + N) % N) * N + ((y + N) % N)) * N + ((x + N) % N)) + 1 + c];
    let ke = 0, ens = 0, maxSpeed = 0, rhoMin = Infinity, rhoMax = -Infinity, finite = true;
    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const c = (z * N + y) * N + x;
          const rho = m[4 * c], ux = m[4 * c + 1], uy = m[4 * c + 2], uz = m[4 * c + 3];
          if (!Number.isFinite(rho + ux + uy + uz)) { finite = false; continue; }
          const sp2 = ux * ux + uy * uy + uz * uz;
          ke += 0.5 * sp2;
          if (sp2 > maxSpeed) maxSpeed = sp2;
          if (rho < rhoMin) rhoMin = rho;
          if (rho > rhoMax) rhoMax = rho;
          const wx = (at(x, y + 1, z, 2) - at(x, y - 1, z, 2)) * 0.5 - (at(x, y, z + 1, 1) - at(x, y, z - 1, 1)) * 0.5;
          const wy = (at(x, y, z + 1, 0) - at(x, y, z - 1, 0)) * 0.5 - (at(x + 1, y, z, 2) - at(x - 1, y, z, 2)) * 0.5;
          const wz = (at(x + 1, y, z, 1) - at(x - 1, y, z, 1)) * 0.5 - (at(x, y + 1, z, 0) - at(x, y - 1, z, 0)) * 0.5;
          ens += 0.5 * (wx * wx + wy * wy + wz * wz);
        }
      }
    }
    const n3 = NCELLS;
    return {
      step, ke: ke / n3, enstrophy: ens / n3, dissipation: 2 * params.nu * (ens / n3),
      maxSpeed: Math.sqrt(maxSpeed), rhoMin, rhoMax, finite,
    };
  }

  // --- controls ------------------------------------------------------------
  const sliceSlider = document.getElementById('slider-SLICE');
  const sliceVal = document.getElementById('val-SLICE');
  const axisSel = document.getElementById('sel-AXIS');
  const modeSel = document.getElementById('sel-MODE');
  const playBtn = document.getElementById('btn-PLAY');
  sliceSlider.max = String(N - 1);
  sliceSlider.value = String(slice);
  sliceVal.textContent = String(slice);
  axisSel.value = String(axis);
  modeSel.value = String(mode);
  sliceSlider.oninput = () => { slice = parseInt(sliceSlider.value); sliceVal.textContent = sliceSlider.value; writeRenderParams(); };
  axisSel.onchange = () => { axis = parseInt(axisSel.value); writeRenderParams(); };
  modeSel.onchange = () => { mode = parseInt(modeSel.value); writeRenderParams(); };
  playBtn.onclick = () => { live = !live; playBtn.textContent = live ? 'pause' : 'play'; };
  document.getElementById('btn-RESET').onclick = () => { reset(); };
  playBtn.textContent = live ? 'pause' : 'play';
  document.getElementById('hint').textContent =
    `${scenarioName}  N=${N}  D3Q${Q}  tau=${params.tau}  nu=${params.nu.toFixed(4)}`;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
  }
  resize();
  window.addEventListener('resize', resize);

  writeRenderParams();
  reset();

  window.__D3 = {
    setLive: (v) => { live = !!v; playBtn.textContent = live ? 'pause' : 'play'; },
    isLive: () => live,
    reset,
    getStep: () => step,
    getParams: () => ({ ...params, Q, N, NCELLS, scenario: scenarioName }),
    readSubsampled, readDuctProfile, readStats,
    debugStepSync,
  };

  let lastStatus = 0;
  async function frame() {
    const enc = device.createCommandEncoder();
    if (live) encodeSteps(enc, STEPS_PER_FRAME);
    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0.07, g: 0.07, b: 0.1, a: 1 }, loadOp: 'clear', storeOp: 'store',
    }]});
    rp.setPipeline(renderPipe); rp.setBindGroup(0, renderBG); rp.draw(6); rp.end();
    device.queue.submit([enc.finish()]);

    const now = performance.now();
    if (now - lastStatus > 250) {
      lastStatus = now;
      const t = scenarioName === 'beltrami' ? `  t/td=${(step / params.td).toFixed(2)}`
        : scenarioName === 'duct' ? `  t/settle=${(step / params.settle).toFixed(2)}`
        : `  Re=${params.re.toFixed(0)}`;
      statusEl.textContent = `${scenarioName}  D3Q${Q}  ${N}^3  step ${step}${t}\n`
        + `${AXIS_NAMES[axis]}-slice ${slice}   ${live ? 'running' : 'paused'}`;
    }
    requestAnimationFrame(() => frame().catch(e => reportFatal(statusEl, e)));
  }
  frame().catch(e => reportFatal(statusEl, e));
}

init().catch(e => reportFatal(statusEl, e));
