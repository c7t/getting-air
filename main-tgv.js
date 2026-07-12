// Validation harness: 2D decaying Taylor-Green vortex -- a doubly-periodic
// flow with an EXACT closed-form solution at every point in space AND time
// (not just steady state), the standard LBM accuracy/convergence-order
// benchmark (see e.g. Krüger et al., "The Lattice Boltzmann Method:
// Principles and Practice," 2017, ch. 8). Reuses shaders/lbm_step.wgsl
// completely unchanged from Phase 1's channel-flow overrides: HAS_BODY=0,
// WALL_Y=0 (default -- no walls at all here), FORCE_X=FORCE_Y=0 (default),
// SPONGE_W=0 (streaming is already exactly periodic, no far-field to
// model). No new shader code was needed for this scenario at all -- the
// only genuinely new piece is the INITIAL CONDITION (seeded from the exact
// analytic field at t=0, not "start at rest" or "start at freestream").
//
// window.__CYL is shaped like main-channel.js's (setLive, reset, getStep,
// getParams, debugStepSync) but reports a full 2D velocity field
// (readField) instead of a 1D profile -- there's no steady state here to
// converge to (the vortex decays to zero), so there's no debugRunToSteady
// equivalent; tools/lib/tgv-metrics.js drives fixed step counts and
// compares against the analytic field/decay rate at each checkpoint.

import { assembleShader } from './shader-loader.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);

// N: square domain size (both the resolution AND the domain length -- the
// vortex's own wavelength is fixed at exactly one period across the
// domain, kx=ky=2*PI/N, so N alone parametrizes both without over- or
// under-resolving the periodic BC). U0: peak velocity amplitude at t=0,
// kept small (Ma well under 0.1) so the exact-incompressible analytic
// solution is a good match for what weakly-compressible LBM actually
// produces.
let N = parseInt(urlParams.get('res')) || 64;
if (N < 16) N = 16;
if (N > 256) N = 256;
let W = N, H = N;
let NCELLS = W * H;

let U0 = parseFloat(urlParams.get('u0')) || 0.04;
let TAU = parseFloat(urlParams.get('tau')) || 0.8;

function nuFromTau(tau) { return (tau - 0.5) / 3; }

const KX = 2 * Math.PI / N;
const KY = 2 * Math.PI / N;

// Analytic decay time -- 1/td = nu*(kx^2+ky^2) -- and the exact field at
// (x,y,t). rho0=1 (lattice units); the pressure/density term is second
// order in U0 (an O(Ma^2) compressibility correction on top of the
// leading-order incompressible velocity field), included here since the
// simulated density field carries the same term and comparing against a
// rho=1-everywhere target would just be measuring that known O(Ma^2)
// effect as if it were error.
function tdFromNu(nu) { return 1 / (nu * (KX * KX + KY * KY)); }

function analyticField(t, nu) {
  const td = tdFromNu(nu);
  const decay = Math.exp(-t / td);
  const decay2 = Math.exp(-2 * t / td);
  const ux = new Float64Array(NCELLS);
  const uy = new Float64Array(NCELLS);
  const rho = new Float64Array(NCELLS);
  const CS2 = 1 / 3;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const c = y * W + x;
      const cx = Math.cos(KX * x), sx = Math.sin(KX * x);
      const cy = Math.cos(KY * y), sy = Math.sin(KY * y);
      ux[c] = -U0 * cx * sy * decay;
      uy[c] = U0 * (KX / KY) * sx * cy * decay;
      const p = -(U0 * U0 / 4) * ((KY / KX) * Math.cos(2 * KX * x) + (KX / KY) * Math.cos(2 * KY * y)) * decay2;
      rho[c] = 1 + p / CS2;
    }
  }
  return { ux, uy, rho, td };
}

const resSlider = document.getElementById('slider-RES');
const resVal    = document.getElementById('val-RES');
resSlider.value = N;
resVal.textContent = N;
resSlider.onchange = () => {
  const url = new URL(window.location);
  url.searchParams.set('res', resSlider.value);
  window.location.href = url.href;
};
resSlider.oninput = () => {
  resVal.textContent = resSlider.value;
};

const u0Slider = document.getElementById('slider-U0');
const u0Val    = document.getElementById('val-U0');
u0Slider.value = U0;
u0Val.textContent = U0.toFixed(3);
u0Slider.onchange = () => {
  const url = new URL(window.location);
  url.searchParams.set('u0', u0Slider.value);
  window.location.href = url.href;
};
u0Slider.oninput = () => {
  u0Val.textContent = parseFloat(u0Slider.value).toFixed(3);
};

const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

// Seed every cell at the equilibrium distribution for the exact analytic
// t=0 field -- standard TGV LBM initialization (see file header). This is
// itself a small approximation (the true kinetic populations aren't
// exactly at equilibrium even for an exactly-incompressible initial
// velocity field), which is why even a "perfect" solver shows a brief
// initial-transient deviation before settling onto the analytic decay --
// expected, not a bug; tools/lib/tgv-metrics.js's checkpoints are chosen
// to sample after that transient.
function initF() {
  const { ux, uy, rho } = analyticField(0, nuFromTau(TAU));
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    for (let i = 0; i < 9; i++) f[i * NCELLS + c] = feq(rho[c], ux[c], uy[c], i);
  }
  return f;
}

async function loadShader(device, path) {
  const code = await assembleShader(path, async (p) => {
    const r = await fetch(p + '?v=' + Date.now());
    if (!r.ok) throw new Error(`failed to load ${p}`);
    return r.text();
  });
  return device.createShaderModule({ code });
}

function handleErr(e) {
  statusEl.textContent = `error: ${e.message}`;
  statusEl.style.color = '#f77';
  console.error('WebGPU Error:', e);
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }
  const device = await adapter.requestDevice();

  device.pushErrorScope('validation');

  const ctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    ctx.configure({ device, format: fmt, alphaMode: 'opaque' });
  }
  window.addEventListener('resize', resize);
  resize();

  const U = GPUBufferUsage;
  const fSize   = NCELLS * 9 * 4;
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_SRC });

  // Dummy CardState: HAS_BODY=0 means lbm_step.wgsl never lets these
  // fields affect the fluid (see main-channel.js's identical rationale).
  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST });
  function cardInit() {
    const card = new Float32Array(26);
    card[12] = 1; card[13] = 1; // a, b
    card[19] = TAU;
    return card;
  }
  device.queue.writeBuffer(cardStateBuf, 0, cardInit());
  device.queue.writeBuffer(f_a, 0, initF());

  const [stepSM, renSM] = await Promise.all([
    loadShader(device, 'shaders/lbm_step.wgsl'),
    loadShader(device, 'shaders/render.wgsl'),
  ]);

  const stepBGL = device.createBindGroupLayout({ label: 'stepBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

  // Fully periodic, no body, no walls, no force -- every override left at
  // its default except HAS_BODY and SPONGE_W (both must be explicitly
  // zeroed; their defaults are 1 and 4 respectively, tuned for the
  // falling-card/cylinder scenarios, not this one).
  const stepConstants = { W, H, HAS_BODY: 0, SPONGE_W: 0 };
  const stepPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
    compute: { module: stepSM, entryPoint: 'main', constants: stepConstants }
  });

  const constants = { W, H };
  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main', constants },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }], constants },
    primitive: { topology: 'triangle-list' },
  });

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;

  function dispatchMacroStep(enc) {
    const stepBG = useB ? stepBG_ba : stepBG_ab;
    const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
    useB = !useB;
  }

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    step = 0;
    useB = false;
  }

  async function debugStepSync(n) {
    liveMode = false;
    const enc = device.createCommandEncoder();
    for (let s = 0; s < n; s++) dispatchMacroStep(enc);
    step += n;
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return { step };
  }

  const stagingVel = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.MAP_READ | U.COPY_DST });

  // Full 2D velocity field (flat row-major, matching lbm_step.wgsl's dense
  // addressing exactly -- unlike the AMR harnesses, no block-major
  // conversion needed here). tools/lib/tgv-metrics.js compares this
  // directly against the analytic field at the current step (dt=1 in
  // lattice units, so "time" IS the step count).
  async function readField() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    device.queue.submit([enc.finish()]);
    await stagingVel.mapAsync(GPUMapMode.READ);
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    stagingVel.unmap();
    const ux = new Array(NCELLS), uy = new Array(NCELLS);
    for (let c = 0; c < NCELLS; c++) { ux[c] = vel[c * 2]; uy[c] = vel[c * 2 + 1]; }
    return { ux, uy };
  }

  window.__CYL = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    getStep: () => step,
    getDims: () => ({ W, H, N }),
    getParams: () => ({ N, W, H, TAU, U0, nu: nuFromTau(TAU), kx: KX, ky: KY, td: tdFromNu(nuFromTau(TAU)) }),
    readField,
    debugStepSync,
  };

  async function frame() {
    try {
      if (!liveMode) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      const enc = device.createCommandEncoder();
      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

      device.queue.submit([enc.finish()]);

      if (performance.now() - lastT > 250) {
        statusEl.textContent = `step ${step}  N=${N}  U0=${U0.toFixed(3)}`;
        lastT = performance.now();
      }

      requestAnimationFrame(() => frame().catch(handleErr));
    } catch (e) {
      handleErr(e);
    }
  }
  frame().catch(handleErr);
}
init().catch(handleErr);
