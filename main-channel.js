// Validation harness: plane Poiseuille (force-driven) or plane Couette
// (moving-wall-driven) channel flow -- both have an exact closed-form
// steady-state velocity profile, unlike the cylinder harness's literature-
// band Cd/St. Reuses shaders/lbm_step.wgsl unchanged (this scenario is
// just a different override configuration -- HAS_BODY=0, WALL_Y on,
// SPONGE_W=0 -- not a re-implemented physics path), so it inherits
// whatever the base solver does; it never adds numerics of its own.
//
// Domain: x (width W, small and periodic -- the flow is x-invariant, so W
// only needs to be a few cells) and y (height H, the real resolution
// parameter, walls at the y=0/y=H-1 edges via shaders/common_walls.wgsl).
// No sponge (SPONGE_W=0): x is exactly periodic, y has real walls, so
// there's no open/far-field edge left for the ALBC sponge to model.
//
// window.__CYL is the CDP-tooling surface (see tools/validate-channel.js),
// shaped like main-cylinder.js's but reporting a steady-state u(y) profile
// instead of a Cd/St time series -- there's no body here to drag/shed.

import { assembleShader } from './shader-loader.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);

// MODE picks which driving mechanism is active -- FORCE_X (Poiseuille) or
// WALL_U1 (Couette), never both. Reload-only (like BLOCKAGE/UPSTREAM in
// main-cylinder.js): it changes which analytic profile is being validated,
// not a live-tunable knob.
const MODE = urlParams.get('mode') === 'couette' ? 'couette' : 'poiseuille';

// H: channel height in cells -- the actual resolution/convergence-order
// parameter. W: streamwise width -- small and fixed regardless of H (the
// flow doesn't vary with x), just large enough for periodic D2Q9
// diagonal streaming to be meaningful.
let H = parseInt(urlParams.get('res')) || 32;
if (H < 8) H = 8;
if (H > 512) H = 512;
let W = parseInt(urlParams.get('w')) || 8;
if (W < 4) W = 4;
let NCELLS = W * H;

// TAU: fixed per page load (a numerical-stability/accuracy choice for the
// current resolution), not derived from Re -- unlike the cylinder harness
// (which fixes U0 and derives TAU from Re), here Re is hit by solving for
// the DRIVING parameter (FORCE_X or WALL_U1) at whatever TAU/nu is chosen,
// which is the standard way these forced-channel benchmarks are set up.
let TAU = parseFloat(urlParams.get('tau')) || 0.8;
let RE  = parseFloat(urlParams.get('re')) || 50;

function nuFromTau(tau) { return (tau - 0.5) / 3; }

// Poiseuille: exact steady-state relation U_max = FORCE_X*H^2/(8*nu)
// (halfway-bounce-back walls at y=-0.5/y=H-0.5, so the wall-to-wall
// physical height is exactly H -- see shaders/common_walls.wgsl's header).
// Solve for FORCE_X given a target centerline Re = U_max*H/nu.
// Couette: WALL_U1 = Re*nu/H directly (linear profile, no force).
function drivingParamsFromRe(re, tau) {
  const nu = nuFromTau(tau);
  const uMax = re * nu / H;
  if (MODE === 'poiseuille') {
    return { FORCE_X: 8 * nu * uMax / (H * H), WALL_U1: 0, uMax };
  }
  return { FORCE_X: 0, WALL_U1: uMax, uMax };
}

const resSlider = document.getElementById('slider-RES');
const resVal    = document.getElementById('val-RES');
resSlider.value = H;
resVal.textContent = H;
resSlider.onchange = () => {
  const url = new URL(window.location);
  url.searchParams.set('res', resSlider.value);
  window.location.href = url.href;
};
resSlider.oninput = () => {
  resVal.textContent = resSlider.value;
};

const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

// Start at rest -- the analytic profile is the STEADY-STATE solution this
// harness measures convergence toward, so the initial condition shouldn't
// already assume it.
function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    for (let i = 0; i < 9; i++) f[i * NCELLS + c] = feq(1, 0, 0, i);
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
  // fields affect the fluid, but the bind group layout still requires the
  // binding (fixed at pipeline-creation time). a=b=1 only avoids a
  // divide-by-zero in get_phi/render.wgsl's visualization shading --
  // irrelevant to the physics either way.
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

  let { FORCE_X, WALL_U1 } = drivingParamsFromRe(RE, TAU);

  // Re (via FORCE_X/WALL_U1) is a pipeline override, not a runtime buffer
  // field like TAU -- changing it means recreating the compute pipeline
  // (cheap: same shader module, new constants), not just a buffer write.
  function makeStepPipeline() {
    const stepConstants = {
      W, H,
      SPONGE_W: 0, // no domain-edge sponge -- x is periodic, y has real walls
      HAS_BODY: 0,
      WALL_Y: 1,
      WALL_U0: 0, WALL_U1,
      FORCE_X, FORCE_Y: 0,
    };
    return device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
      compute: { module: stepSM, entryPoint: 'main', constants: stepConstants }
    });
  }
  let stepPL = makeStepPipeline();

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

  // Re is baked into a pipeline override (FORCE_X/WALL_U1), so unlike the
  // cylinder harness's setRe (a single buffer write), this recreates the
  // step pipeline -- still cheap (same shader module, no recompilation
  // from source), and keeps the "no page reload between sweep cases"
  // convenience tools/lib/channel-metrics.js's runCase relies on.
  function setRe(re) {
    RE = re;
    ({ FORCE_X, WALL_U1 } = drivingParamsFromRe(RE, TAU));
    stepPL = makeStepPipeline();
    return { RE, TAU, FORCE_X, WALL_U1 };
  }

  const reSlider = document.getElementById('slider-RE');
  const reVal    = document.getElementById('val-RE');
  reSlider.value = RE;
  reVal.textContent = RE.toFixed(0);
  reSlider.oninput = () => {
    setRe(parseFloat(reSlider.value));
    reVal.textContent = RE.toFixed(0);
  };

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    step = 0;
    useB = false;
  }

  // Dedicated staging buffer for profile reads, separate from frame()'s
  // own rendering so a debug read can't race a live-mode frame in flight.
  const stagingVel = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.MAP_READ | U.COPY_DST });

  // u(y): the x-velocity profile, averaged over the (x-invariant-at-
  // steady-state) streamwise direction -- cheap noise cancellation with
  // no separate analysis pass, since the exact solution has no x
  // dependence at all.
  async function readProfile() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    device.queue.submit([enc.finish()]);
    await stagingVel.mapAsync(GPUMapMode.READ);
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    stagingVel.unmap();
    const uy = new Array(H).fill(0);
    for (let y = 0; y < H; y++) {
      let sum = 0;
      for (let x = 0; x < W; x++) sum += vel[(y * W + x) * 2];
      uy[y] = sum / W;
    }
    return uy;
  }

  // Deterministic, synchronous stepping for CDP tooling -- no rAF/
  // backpressure interaction, matching main-cylinder.js's
  // debugRunAndCollect in spirit.
  async function debugStepSync(n) {
    liveMode = false;
    const enc = device.createCommandEncoder();
    for (let s = 0; s < n; s++) dispatchMacroStep(enc);
    step += n;
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return step;
  }

  // Runs in blocks until the u(y) profile's relative L2 change between
  // successive blocks drops below `tol` (steady state reached), or
  // `maxSteps` is hit first (safety cap -- this is a validation tool, a
  // non-converging case should FAIL loudly, not hang).
  async function debugRunToSteady(opts = {}) {
    liveMode = false;
    const blockSteps = opts.blockSteps || 512;
    const maxSteps = opts.maxSteps || 200000;
    const tol = opts.tol || 1e-7;
    let prev = null;
    while (step < maxSteps) {
      await debugStepSync(blockSteps);
      const profile = await readProfile();
      if (prev) {
        let num = 0, den = 0;
        for (let y = 0; y < H; y++) {
          const d = profile[y] - prev[y];
          num += d * d;
          den += profile[y] * profile[y];
        }
        const rel = Math.sqrt(num / Math.max(den, 1e-30));
        if (rel < tol) return { step, profile, converged: true };
      }
      prev = profile;
    }
    return { step, profile: prev, converged: false };
  }

  window.__CYL = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    setRe,
    getStep: () => step,
    getDims: () => ({ W, H }),
    getParams: () => ({ mode: MODE, W, H, TAU, Re: RE, nu: nuFromTau(TAU), FORCE_X, WALL_U1 }),
    readProfile,
    debugStepSync,
    debugRunToSteady,
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
        statusEl.textContent = `step ${step}  mode=${MODE}  Re=${RE.toFixed(0)}`;
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
