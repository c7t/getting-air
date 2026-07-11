// Validation harness: a fixed circular cylinder in uniform crossflow.
// Reuses the exact same collide/stream/force/physics kernels as main.js
// (shaders/lbm_step.wgsl, lbm_force.wgsl, physics.wgsl) -- this scenario is
// only a different choice of CardState + sponge target, not a re-implemented
// physics path that could quietly drift from what main.js ships.
//
// The card is pinned (v_max = o_max = 0, g_eff = 0) so it never moves: the
// physics pass still runs every step (it drains the force atomics and
// copies them into state.fx/fy for readback), it just never lets that force
// change the body's velocity. The sponge, which normally relaxes domain
// edges toward a quiescent far field, is instead given a uniform freestream
// via lbm_step.wgsl's SPONGE_UX/UY pipeline overrides (default 0, so main.js
// is unaffected).
//
// window.__CYL is the CDP-tooling surface (see tools/validate-cylinder.js),
// modeled directly on main-amr.js's window.__AMR.

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseInt(urlParams.get('res')) || 9;
if (resLog2 < 7) resLog2 = 7;
if (resLog2 > 11) resLog2 = 11;

// Optional: sharp momentum-exchange bounce-back solid coupling instead of
// the default diffuse (Brinkman/Guo) volume penalization -- see
// shaders/lbm_step.wgsl/lbm_force.wgsl's own USE_BOUNCEBACK header for the
// method. Default off (0) reproduces today's exact behavior; main.js never
// sets this at all, so the falling-card scenario is untouched either way.
const USE_BOUNCEBACK = urlParams.has('bounceback') ? 1 : 0;

let W = 1 << resLog2;
let H = W;
let NCELLS = W * H;

// BLOCKAGE: domain height / D (D = cylinder diameter). UPSTREAM: distance
// from the left edge to the cylinder center, in diameters. R is *derived*
// from W and BLOCKAGE rather than taken directly in lattice units -- if R
// were an independent lattice-unit parameter, bumping it at fixed domain
// resolution silently changes both the blockage ratio and the upstream/
// downstream fetch (in diameters), confounding any attempt at a grid-
// convergence study (see the R=12->24 experiment that made Cd move the
// wrong way). Deriving R this way means bumping only `res` in the URL is
// a valid resolution sweep: blockage and fetch length in D-units stay
// fixed, only the diffuse-interface width relative to D shrinks.
let BLOCKAGE = parseFloat(urlParams.get('blockage')) || 24;
let UPSTREAM = parseFloat(urlParams.get('upstream')) || 8;
let R = W / (2 * BLOCKAGE);

// U0: freestream speed in lattice units/step (kept small so Ma = U0/cs
// stays well under 0.3). Re: Reynolds number, used only to derive
// TAU = 0.5 + 3*nu, nu = U0*D/Re. U0 is baked into the SPONGE_UX pipeline
// override, so changing it takes a page reload, same convention as the
// RES slider. TAU is dynamic (like main.js) so Re can be explored live
// without recompiling pipelines.
let U0 = parseFloat(urlParams.get('u0')) || 0.04;
let RE = parseFloat(urlParams.get('re')) || 100;

function tauFromRe(re) {
  const D  = 2 * R;
  const nu = U0 * D / re;
  return 0.5 + 3 * nu;
}
let TAU = tauFromRe(RE);

const resSlider = document.getElementById('slider-RES');
const resVal    = document.getElementById('val-RES');
resSlider.value = resLog2;
resVal.textContent = W;
resSlider.onchange = () => {
  const url = new URL(window.location);
  url.searchParams.set('res', resSlider.value);
  window.location.href = url.href;
};
resSlider.oninput = () => {
  resVal.textContent = 1 << parseInt(resSlider.value);
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

// PERTURB: amplitude (as a fraction of U0) of a small transverse-velocity
// perturbation seeded into the initial condition. Both the analytic circle
// and a pure freestream initial condition are exactly top/bottom symmetric,
// so without this, vortex shedding onset has to grow from whatever
// asymmetry floating-point round-off happens to provide -- which is a
// *smaller and slower-growing* seed at higher grid resolution (finer grids
// have less discretization error to seed from), making Cd/shedding-strength
// look like they're shrinking with resolution when really the higher-
// resolution runs just haven't finished saturating within the same step
// budget (this is what the res=8/9/10 sweep showed: Cd fell monotonically
// with resolution, and the Cl oscillation got visibly weaker each time).
// SEED is a fixed PRNG seed so runs stay reproducible for regression use.
const PERTURB = parseFloat(urlParams.get('perturb')) || 0.02;
const SEED    = parseInt(urlParams.get('seed'))      || 12345;

function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Seed the whole domain at the freestream velocity (rather than quiescent)
// -- this is standard practice for these benchmarks and shortens the
// startup transient that has to be discarded before measuring Cd/St -- plus
// a small per-cell random transverse perturbation (see PERTURB above) so
// shedding onset doesn't depend on grid-resolution-dependent round-off.
function initF() {
  const rng = mulberry32(SEED);
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    const uy = (rng() * 2 - 1) * PERTURB * U0;
    for (let i = 0; i < 9; i++) {
      f[i * NCELLS + c] = feq(1, U0, uy, i);
    }
  }
  return f;
}

async function loadShader(device, path) {
  const r = await fetch(path + '?v=' + Date.now());
  if (!r.ok) throw new Error(`failed to load ${path}`);
  const code = await r.text();
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
  const fSize    = NCELLS * 9 * 4;
  const f_a      = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST });
  const f_b      = device.createBuffer({ size: fSize, usage: U.STORAGE });
  const velBuf   = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 26 floats = 104 bytes. Cylinder placed UPSTREAM diameters
  // downstream of the inlet edge, centered transversely, matching the
  // "cylinder N diameters downstream" convention of the FPSC benchmark this
  // mirrors (amr-lbm.pdf S5.3) -- in diameters, not a fixed W/4, so it scales
  // correctly with R (see the BLOCKAGE/UPSTREAM comment above).
  const CX0 = UPSTREAM * 2 * R;
  const CY0 = H / 2;
  function cardInit() {
    return new Float32Array([
      CX0, CY0, 0,     // cx, cy, theta
      0, 0, 0,         // vx, vy, omega -- pinned (v_max/o_max = 0 below)
      0, 0, 0,         // fx, fy, tz
      1, 1, 0,         // mass, i_body, g_eff (no gravity, body is fixed)
      R, R,            // a, b (circle)
      0, 0,            // v_max, o_max -- 0 freezes the body exactly
      CX0, CY0, 0,     // cx_old, cy_old, th_old
      TAU,             // tau
      0, 0,            // y_total, x_total
      0, 0, 0, 0        // off_x, off_y, off_x_old, off_y_old
    ]);
  }

  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  device.queue.writeBuffer(cardStateBuf, 0, cardInit());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));

  // Only TAU is live-adjustable post-init (R/U0/RES are pipeline-baked, see
  // the RES/U0 sliders' reload-on-change above). Writes immediately rather
  // than through a paramsDirty flag so it works the same from the live
  // frame() loop and from CDP tooling's deterministic debugRunAndCollect,
  // which doesn't poll paramsDirty.
  function setRe(re) {
    RE = re;
    TAU = tauFromRe(RE);
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
    return { RE, TAU };
  }

  const reSlider = document.getElementById('slider-RE');
  const reVal    = document.getElementById('val-RE');
  reSlider.value = RE;
  reVal.textContent = RE.toFixed(0);
  reSlider.oninput = () => {
    setRe(parseFloat(reSlider.value));
    reVal.textContent = RE.toFixed(0);
  };

  const [stepSM, frcSM, phySM, renSM] = await Promise.all([
    loadShader(device, 'shaders/lbm_step.wgsl'),
    loadShader(device, 'shaders/lbm_force.wgsl'),
    loadShader(device, 'shaders/physics.wgsl'),
    loadShader(device, 'shaders/render.wgsl'),
  ]);

  const stepBGL = device.createBindGroupLayout({ label: 'stepBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const frcBGL = device.createBindGroupLayout({ label: 'frcBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const phyBGL = device.createBindGroupLayout({ label: 'phyBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

  const stepConstants = { W, H, SPONGE_UX: U0, SPONGE_UY: 0, USE_BOUNCEBACK };
  const constants     = { W, H };

  const stepPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
    compute: { module: stepSM, entryPoint: 'main', constants: stepConstants }
  });
  const frcPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }),
    compute: { module: frcSM, entryPoint: 'main', constants: { ...constants, USE_BOUNCEBACK } }
  });
  const phyPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [phyBGL] }),
    compute: { module: phySM, entryPoint: 'main', constants }
  });
  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main', constants },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }], constants },
    primitive: { topology: 'triangle-list' },
  });

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});

  const frcBG_a = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }]});
  const frcBG_b = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }]});

  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;

  const D = 2 * R;
  // trajectory rows: [step, fx, fy, Cd, Cl]. fx/fy are the raw hydrodynamic
  // force on the body in lattice units (state.fx/fy, populated by
  // physics.wgsl every step regardless of the body being pinned).
  const trajectory = [];

  function dispatchMacroStep(enc) {
    const stepBG = useB ? stepBG_ba : stepBG_ab;
    const frcBG  = useB ? frcBG_b  : frcBG_a;
    const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
    const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
    const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
    useB = !useB;
  }

  function resetSim() {
    device.queue.writeBuffer(cardStateBuf, 0, cardInit());
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    step = 0;
    useB = false;
    trajectory.length = 0;
  }

  document.getElementById('download').onclick = () => {
    const header = "step,fx,fy,Cd,Cl\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cylinder_Re${RE}_${W}x${H}.csv`;
    a.click();
  };

  // Triple-buffering for readbacks to avoid CPU-GPU stalls.
  const STAGES = 3;
  const stages = Array.from({ length: STAGES }, () => ({
    card: device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST }),
    inFlight: false,
    step: 0
  }));
  let currentStageIdx = 0;

  // Deterministic, synchronous stepping for CDP tooling (see
  // tools/validate-cylinder.js): no rAF/backpressure interaction, so a
  // measurement window is exactly nSteps long and every sample is real
  // (not interpolated/dropped under load), matching main-amr.js's
  // debugStepSync but also collecting the force history.
  async function debugRunAndCollect(nSteps) {
    liveMode = false;
    const stage = device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST });
    for (let k = 0; k < nSteps; k += STEPS_PER_FRAME) {
      const enc = device.createCommandEncoder();
      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;
      enc.copyBufferToBuffer(cardStateBuf, 0, stage, 0, 104);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      await stage.mapAsync(GPUMapMode.READ);
      const d = new Float32Array(stage.getMappedRange());
      const fx = d[6], fy = d[7];
      const Cd = 2 * fx / (U0 * U0 * D);
      const Cl = 2 * fy / (U0 * U0 * D);
      trajectory.push([step, fx, fy, Cd, Cl]);
      stage.unmap();
    }
    return { step, history: trajectory.slice() };
  }

  window.__CYL = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    setRe,
    getStep: () => step,
    getDims: () => ({ W, H }),
    getParams: () => ({ R, D: 2 * R, U0, Re: RE, TAU, blockage: BLOCKAGE, upstream: UPSTREAM, perturb: PERTURB, seed: SEED, W, H }),
    getForceHistory: () => trajectory.slice(),
    debugRunAndCollect,
  };

  async function frame() {
    try {
      if (!liveMode) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      const stage = stages[currentStageIdx];
      if (stage.inFlight) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      const enc = device.createCommandEncoder();
      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

      enc.copyBufferToBuffer(cardStateBuf, 0, stage.card, 0, 104);

      device.queue.submit([enc.finish()]);

      stage.inFlight = true;
      stage.step = step;

      const processReadback = async (st) => {
        await st.card.mapAsync(GPUMapMode.READ);
        const d = new Float32Array(st.card.getMappedRange());
        const fx = d[6], fy = d[7];
        const Cd = 2 * fx / (U0 * U0 * D);
        const Cl = 2 * fy / (U0 * U0 * D);
        if (st.step < 500000) trajectory.push([st.step, fx, fy, Cd, Cl]);

        if (performance.now() - lastT > 250) {
          statusEl.textContent = `step ${st.step}  Re=${RE.toFixed(0)}  Cd=${Cd.toFixed(3)}  Cl=${Cl.toFixed(3)}`;
          lastT = performance.now();
        }

        st.card.unmap();
        st.inFlight = false;
      };

      processReadback(stage);

      currentStageIdx = (currentStageIdx + 1) % STAGES;
      requestAnimationFrame(() => frame().catch(handleErr));
    } catch (e) {
      handleErr(e);
    }
  }
  frame().catch(handleErr);
}
init().catch(handleErr);
