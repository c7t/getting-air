import { assembleShader } from './shader-loader.mjs';
import {
  deriveCardParams, parseCardParams, parseResLog2, reynoldsFromTau,
  DENSE_DEFAULT_RES_LOG2,
} from './card-params.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseResLog2(urlParams, DENSE_DEFAULT_RES_LOG2);

let W = 1 << resLog2;
let H = W;
let NCELLS = W * H;

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

// ── Pesavento & Wang (2004) physical parameters ───────────────────────────────
// Paper: "Falling Paper: Navigating the Trade-Off between Density and Aspect
// Ratio". The parameterization and all the derived-quantity arithmetic live
// in card-params.mjs, shared verbatim with main-amr.js -- see that module's
// header for why it is shared rather than copied, and for the meaning of
// each quantity. The short version: card size and flow regime are stored as
// resolution-independent physical quantities (BLOCKAGE/ASPECT/RE), not raw
// lattice-cell counts, so pasting the same ?blockage=&aspect=&re=&ut= onto
// this page and index-amr.html reproduces the identical physical system.
let { BLOCKAGE, ASPECT, I_STAR, RE, U_T } = parseCardParams(urlParams);

// Derived in recalculate() below, from the physical parameters above plus W.
let A, B, TAU, RHO_B, MASS, I_BODY, G_LU, G_EFF;

function recalculate() {
  ({ A, B, TAU, RHO_B, MASS, I_BODY, G_LU, G_EFF } =
    deriveCardParams({ W, BLOCKAGE, ASPECT, I_STAR, RE, U_T }));
}
recalculate();


// FSCALE: Atomic Scaling Factor.
// Used to convert floating-point forces/torques to integers for the GPU atomics.
// Must be large enough for precision (1e4 = 0.0001 precision) but small enough
// to avoid 32-bit integer overflow when summing 1000s of cells.
const FSCALE  = 1e4;

const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    for (let i = 0; i < 9; i++) {
      f[i * NCELLS + c] = feq(1, 0, 0, i);
    }
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
  
  const hasTimestamp = 0 && adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamp ? ['timestamp-query'] : []
  });

  const querySet = hasTimestamp ? device.createQuerySet({
    type: 'timestamp',
    count: 2
  }) : null;
  const queryResolveBuffer = hasTimestamp ? device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
  }) : null;
  const queryReadBuffer = hasTimestamp ? device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  }) : null;
  
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
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 26 floats = 104 bytes
  const cardStateBuf   = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  const cardInit = new Float32Array([
    W/2, H/2, 0.2,   // cx, cy, theta
    0, 0, 0,         // vx, vy, omega
    0, 0, 0,         // fx, fy, tz
    MASS, I_BODY, G_EFF,
    A, B,
    0.3, 0.025,      // v_max, o_max
    W/2, H/2, 0.2,   // cx_old, cy_old, th_old
    TAU,             // tau
    0, 0,            // y_total, x_total
    0, 0, 0, 0       // off_x, off_y, off_x_old, off_y_old
  ]);
  device.queue.writeBuffer(cardStateBuf, 0, cardInit);
  device.queue.writeBuffer(f_a, 0, initF());

  let paramsDirty = false;
  const updateGPUParams = () => {
    const data = new Float32Array([MASS, I_BODY, G_EFF, A, B]);
    device.queue.writeBuffer(cardStateBuf, 9 * 4, data);
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
  };

  // BLOCKAGE/ASPECT/I_STAR/U_T all move the physical inputs to recalculate();
  // RE is the canonical flow-regime state (recalculate() always re-derives
  // TAU = tauFromReynolds(RE, A, U_T) from it), and TAU is the one exception
  // that goes the other way: dragging it back-solves RE first, so the two
  // stay mutually consistent regardless of which one the user drags.
  const blockageEl = document.getElementById('slider-BLOCKAGE');
  const aspectEl   = document.getElementById('slider-ASPECT');
  const iStarEl    = document.getElementById('slider-I_STAR');
  const reEl       = document.getElementById('slider-RE');
  const tauEl      = document.getElementById('slider-TAU');
  const utEl       = document.getElementById('slider-U_T');

  // Each readout shows the control's OWN value first, then the lattice-unit
  // quantity it derives, e.g. "2.0 (A=64.0)". An earlier revision showed only
  // the derived A/B under labels reading "Blockage"/"e (aspect)", so the
  // panel actively misreported what the slider was set to -- worth avoiding
  // in a project where a lot of debugging happens by reading this panel.
  const refreshDerivedReadouts = () => {
    document.getElementById('val-BLOCKAGE').textContent = `${BLOCKAGE.toFixed(1)} (A=${A.toFixed(1)})`;
    document.getElementById('val-ASPECT').textContent = `${ASPECT.toFixed(3)} (B=${B.toFixed(1)})`;
    document.getElementById('val-I_STAR').textContent = I_STAR.toFixed(2);
    document.getElementById('val-RE').textContent = Math.round(RE);
    document.getElementById('val-TAU').textContent = TAU.toFixed(4);
    document.getElementById('val-U_T').textContent = U_T.toFixed(3);
  };

  blockageEl.oninput = () => { BLOCKAGE = parseFloat(blockageEl.value); recalculate(); refreshDerivedReadouts(); paramsDirty = true; };
  aspectEl.oninput   = () => { ASPECT   = parseFloat(aspectEl.value);   recalculate(); refreshDerivedReadouts(); paramsDirty = true; };
  iStarEl.oninput    = () => { I_STAR   = parseFloat(iStarEl.value);    recalculate(); refreshDerivedReadouts(); paramsDirty = true; };
  utEl.oninput       = () => { U_T      = parseFloat(utEl.value);       recalculate(); refreshDerivedReadouts(); paramsDirty = true; };
  reEl.oninput       = () => { RE       = parseFloat(reEl.value);       recalculate(); refreshDerivedReadouts(); paramsDirty = true; };
  tauEl.oninput      = () => {
    const tau = parseFloat(tauEl.value);
    RE = reynoldsFromTau(tau, A, U_T);
    reEl.value = RE;
    recalculate(); // re-derives TAU from the just-updated RE (reproduces `tau`, mod float noise)
    refreshDerivedReadouts();
    paramsDirty = true;
  };

  // Sync widget positions to the actual initial state (fixes a pre-existing
  // bug where the HTML's hardcoded slider defaults didn't match the real
  // initial JS values) and show the initial derived readouts.
  blockageEl.value = BLOCKAGE;
  aspectEl.value   = ASPECT;
  iStarEl.value    = I_STAR;
  reEl.value       = RE;
  tauEl.value      = TAU;
  utEl.value       = U_T;
  refreshDerivedReadouts();

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

  const constants = { W, H };

  const stepPL = device.createComputePipeline({ 
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }), 
    compute: { module: stepSM, entryPoint: 'main', constants } 
  });
  const frcPL = device.createComputePipeline({ 
    layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }), 
    compute: { module: frcSM, entryPoint: 'main', constants } 
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

  const trajectory = [];

  document.getElementById('download').onclick = () => {
    const header = "step,cx,cy_total,cx_total,theta,vx,vy,omega,fx,fy,tz\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trajectory_${W}x${H}.csv`;
    a.click();
  };

  // Triple-buffering for readbacks to avoid CPU-GPU stalls
  const STAGES = 3;
  const stages = Array.from({ length: STAGES }, () => ({
    card: device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST }),
    query: hasTimestamp ? device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST }) : null,
    inFlight: false,
    step: 0
  }));
  let currentStageIdx = 0;

  const mlupsEl = document.getElementById('val-mlups');
  const gpuMsEl = document.getElementById('val-gpu-ms');
  const syncMsEl = document.getElementById('val-sync-ms');

  async function frame() {
    try {
      if (paramsDirty) {
        updateGPUParams();
        paramsDirty = false;
      }
      
      const stage = stages[currentStageIdx];
      // Backpressure: if the oldest stage is still in flight, we must wait.
      // With 3 buffers and STEPS_PER_FRAME=64, this should be rare.
      if (stage.inFlight) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      const enc = device.createCommandEncoder();
      
      if (hasTimestamp) {
        enc.writeTimestamp(querySet, 0);
      }

      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        const stepBG = useB ? stepBG_ba : stepBG_ab;
        const frcBG  = useB ? frcBG_b  : frcBG_a;
        
        const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
        const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
        const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
        
        useB = !useB;
      }
      step += STEPS_PER_FRAME;

      if (hasTimestamp) {
        enc.writeTimestamp(querySet, 1);
        enc.resolveQuerySet(querySet, 0, 2, queryResolveBuffer, 0);
        enc.copyBufferToBuffer(queryResolveBuffer, 0, stage.query, 0, 16);
      }

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();
      
      enc.copyBufferToBuffer(cardStateBuf, 0, stage.card, 0, 104);
      
      const tSubmit = performance.now();
      device.queue.submit([enc.finish()]);

      // Start asynchronous readback
      stage.inFlight = true;
      stage.step = step;
      
      const processReadback = async (st) => {
        const pCard = st.card.mapAsync(GPUMapMode.READ);
        const pQuery = hasTimestamp ? st.query.mapAsync(GPUMapMode.READ) : Promise.resolve();
        
        await Promise.all([pCard, pQuery]);
        
        const d = new Float32Array(st.card.getMappedRange());
        let gpuTime = 0;
        if (hasTimestamp) {
          const timestamps = new BigUint64Array(st.query.getMappedRange());
          gpuTime = Number(timestamps[1] - timestamps[0]) / 1e6;
          st.query.unmap();
        } else {
          // Without native timestamps, we measure CPU submission-to-read completion
          gpuTime = performance.now() - tSubmit;
        }

        // Update trajectory from this specific completed step
        if (st.step < 100000) {
          // Record: step, cx, cy_total, cx_total, theta, vx, vy, omega, fx, fy, tz
          trajectory.push([st.step, d[0], d[20], d[21], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]);
        }
        
        if (performance.now() - lastT > 250) {
          const mlups = (NCELLS * STEPS_PER_FRAME) / (gpuTime * 1e3);
          mlupsEl.textContent = mlups.toFixed(1);
          gpuMsEl.textContent = gpuTime.toFixed(2);
          syncMsEl.textContent = (performance.now() - tSubmit).toFixed(2);
          statusEl.textContent = `step ${st.step}  y=${d[20].toFixed(1)}  x=${d[21].toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`;
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
