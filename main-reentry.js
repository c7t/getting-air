// Dense reference for the "reentry" scenario (see main-reentry-amr.js's own
// header for the full motivation): the same thin-ellipse falling card as
// main.js, reusing the exact same collide/stream/force/physics kernels
// (shaders/lbm_step.wgsl, lbm_force.wgsl, physics.wgsl) -- this scenario is
// only a different choice of CardState + KINEMATIC pipeline override, not a
// re-implemented physics path that could quietly drift from what main.js
// ships (same "different CardState/sponge target, not a reimplementation"
// relationship main-cylinder.js has to main.js, which this file is
// structurally forked from for its CDP-tooling debug API).
//
// Unlike main-cylinder.js's pinned body (v_max=o_max=0, letting the
// UNMODIFIED physics.wgsl Newton-integration path settle at exactly zero
// velocity), this scenario's body is prescribed a CONSTANT nonzero vy/omega
// via shaders/physics.wgsl's KINEMATIC override -- same override, same
// VY_FIXED/OMEGA_FIXED defaults, as main-reentry-amr.js, so the two pages
// are directly comparable: an apples-to-apples dense-vs-AMR field diff
// without main-amr.js's own coupled dynamics being chaotic (trajectory-
// drift-sensitive) muddying the comparison. No sponge freestream (SPONGE_UX/
// UY left at their 0 default) -- the card falls through otherwise-quiescent
// fluid, matching main.js's own convention, not main-cylinder.js's uniform
// crossflow.
//
// window.__CYL is the CDP-tooling surface, same shape as main-cylinder.js's
// own (debugSnapshotSave produces the same layout:'flat' shape
// tools/lib/field-reconstruct.js's loadDenseFields already decodes).

import { assembleShader } from './shader-loader.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseInt(urlParams.get('res')) || 8;
if (resLog2 < 6) resLog2 = 6;
if (resLog2 > 11) resLog2 = 11;

let W = 1 << resLog2;
let H = W;
let NCELLS = W * H;

// Same thin-ellipse geometry as main-amr.js/main-reentry-amr.js -- see that
// file's header for why A/B are in these units.
let A = parseFloat(urlParams.get('a')) || 32;
let B = parseFloat(urlParams.get('b')) || 4;
let TAU = parseFloat(urlParams.get('tau')) || 0.509;

// Prescribed kinematics -- identical defaults to main-reentry-amr.js's own
// (see that file's header for how they were measured), so the two pages
// describe the exact same trajectory.
let VY = urlParams.has('vy') ? parseFloat(urlParams.get('vy')) : 0.0185;
let OMEGA = urlParams.has('omega') ? parseFloat(urlParams.get('omega')) : 0.00059;

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

const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

// Quiescent IC (rho=1, u=0 everywhere), matching main.js's own falling-card
// convention -- no freestream, no perturbation needed: unlike a pinned
// symmetric body waiting for shedding onset (main-cylinder.js's own
// PERTURB rationale), this scenario's asymmetry comes from the imposed
// nonzero omega from step 0, not from floating-point round-off.
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

// base64 chunked in 8192-byte pieces -- a single huge String.fromCharCode
// spread risks "Maximum call stack size exceeded" for larger grids. Copied
// verbatim from main-cylinder-amr.js's own helper (this project keeps each
// main*.js entry point self-contained rather than cross-importing, even for
// larger overlap than a 10-line helper).
function bytesToB64(bytes) {
  let binary = '';
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
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
  // f_a/velBuf need COPY_SRC (not just STORAGE|COPY_DST) so debugSnapshotSave
  // below can copyBufferToBuffer out of them -- main-cylinder-amr.js's own
  // debugSnapshotSave comment documents hitting exactly this "forgot
  // COPY_SRC" class of bug once already.
  const f_a      = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b      = device.createBuffer({ size: fSize, usage: U.STORAGE });
  const velBuf   = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 26 floats = 104 bytes. Centered in the domain, matching
  // main-amr.js's own convention (not main-cylinder.js's UPSTREAM-diameters
  // placement, which assumes a fixed body and an inlet/outlet, neither of
  // which apply here -- the card falls through the domain, kept on-screen
  // by the same moving-window panning main.js/main-amr.js already use).
  // vx/vy/omega start at 0 but that's irrelevant -- KINEMATIC mode
  // (shaders/physics.wgsl) sets them fresh from VY_FIXED/OMEGA_FIXED on the
  // very first step, before they're ever read for a position update.
  // mass/i_body/g_eff are dummy placeholders (unused under KINEMATIC), same
  // convention as main-cylinder.js's own pinned-body CardState.
  function cardInit() {
    return new Float32Array([
      W / 2, H / 2, 0.2, // cx, cy, theta (0.2 initial tilt, matching main-amr.js)
      0, 0, 0,           // vx, vy, omega
      0, 0, 0,           // fx, fy, tz
      1, 1, 0,           // mass, i_body, g_eff -- dummy, unused under KINEMATIC
      A, B,              // a, b (thin ellipse)
      0.3, 0.025,        // v_max, o_max -- dummy, unused under KINEMATIC
      W / 2, H / 2, 0.2, // cx_old, cy_old, th_old
      TAU,               // tau
      0, 0,              // y_total, x_total
      0, 0, 0, 0         // off_x, off_y, off_x_old, off_y_old
    ]);
  }

  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  device.queue.writeBuffer(cardStateBuf, 0, cardInit());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));

  // TAU is live-adjustable post-init, same "write immediately, not through a
  // paramsDirty flag" reasoning as main-cylinder.js's own setRe (works the
  // same from the live frame() loop and from CDP tooling's deterministic
  // debugStepSync, which doesn't poll paramsDirty).
  function setTau(tau) {
    TAU = tau;
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
    return { TAU };
  }

  const tauSlider = document.getElementById('slider-TAU');
  const tauVal    = document.getElementById('val-TAU');
  tauSlider.value = TAU;
  tauVal.textContent = TAU.toFixed(3);
  tauSlider.oninput = () => {
    setTau(parseFloat(tauSlider.value));
    tauVal.textContent = TAU.toFixed(3);
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

  // No SPONGE_UX/UY, no USE_BOUNCEBACK override -- both default (0), matching
  // main.js's own falling-card convention: quiescent far field, diffuse
  // (Brinkman/Guo) body coupling. See this file's own header.
  const constants = { W, H };

  const stepPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
    compute: { module: stepSM, entryPoint: 'main', constants }
  });
  const frcPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }),
    compute: { module: frcSM, entryPoint: 'main', constants }
  });
  // KINEMATIC=1 -- see shaders/physics.wgsl's own comment and
  // main-reentry-amr.js's identical phyPL construction.
  const phyPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [phyBGL] }),
    compute: { module: phySM, entryPoint: 'main', constants: { W, H, KINEMATIC: 1, VY_FIXED: VY, OMEGA_FIXED: OMEGA } }
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

  // trajectory rows: [step, fx, fy, tz]. Raw hydrodynamic force/torque on
  // the body in lattice units (state.fx/fy/tz, populated by physics.wgsl
  // every step regardless of KINEMATIC mode never feeding them back) -- no
  // Cd/Cl coefficient here (unlike main-cylinder.js): there's no freestream
  // D/U0 to normalize by, matching main-amr.js/main-reentry-amr.js's own
  // raw-force CSV convention.
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
    const header = "step,fx,fy,tz\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reentry_vy${VY}_omega${OMEGA}_${W}x${H}.csv`;
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
      const fx = d[6], fy = d[7], tz = d[8];
      trajectory.push([step, fx, fy, tz]);
      stage.unmap();
    }
    return { step, history: trajectory.slice() };
  }

  // Fast path for tooling that just wants to advance and snapshot (e.g.
  // dense-vs-AMR field comparisons) -- no per-STEPS_PER_FRAME-block card
  // readback/sync, matching main-amr.js's own debugStepSync.
  async function debugStepSync(n) {
    liveMode = false;
    for (let k = 0; k < n; k += STEPS_PER_FRAME) {
      const enc = device.createCommandEncoder();
      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      step += STEPS_PER_FRAME;
    }
    return { step };
  }

  // ── Debug/verification support: field snapshot ────────────────────────
  // Dedicated staging buffers, separate from the triple-buffered readback
  // stages above, so a debug read can't race frame()'s own in-flight
  // readback.
  const stagingF    = device.createBuffer({ size: fSize, usage: U.MAP_READ | U.COPY_DST });
  const stagingVel  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingCard = device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST });

  // Mirrors main-cylinder-amr.js's own debugSnapshotSave, minus pools (this
  // is the dense/single-level harness -- no AMR machinery to snapshot).
  // Produces the same shape tools/lib/field-reconstruct.js's
  // loadDenseFields already expects (layout:'flat', the pinned-cylinder
  // harnesses' shared convention -- see that function's own comment).
  // `params` intentionally matches main-reentry-amr.js's own params shape
  // (A,B,TAU,VY,OMEGA,resLog2): lets a dense-vs-AMR comparison tool assert
  // the two sides agree on the prescribed trajectory before ever reaching
  // the field diff.
  //
  // Invariant this relies on (same as main-cylinder-amr.js's own
  // debugSnapshotSave): only call while liveMode is false. STEPS_PER_FRAME
  // (64) is even, so useB always returns to its initial value (false) at a
  // macro-step-block boundary -- f_a (not f_b) is always the authoritative/
  // current buffer whenever no frame is mid-flight.
  async function debugSnapshotSave() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(f_a, 0, stagingF, 0, fSize);
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    enc.copyBufferToBuffer(cardStateBuf, 0, stagingCard, 0, 104);
    device.queue.submit([enc.finish()]);

    await Promise.all([stagingF, stagingVel, stagingCard].map(b => b.mapAsync(GPUMapMode.READ)));
    const f = new Float32Array(stagingF.getMappedRange()).slice();
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    const card = Array.from(new Float32Array(stagingCard.getMappedRange()).slice());
    stagingF.unmap();
    stagingVel.unmap();
    stagingCard.unmap();

    const snapshot = {
      formatVersion: 1,
      layout: 'flat', // plain row-major (cy*W+cx) -- matches shaders/lbm_step.wgsl, no block8 sub-tiling
      W, H, step,
      cardState: card,
      fB64: bytesToB64(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)),
      velB64: bytesToB64(new Uint8Array(vel.buffer, vel.byteOffset, vel.byteLength)),
      params: { A, B, TAU, VY, OMEGA, resLog2 },
    };
    console.log('[CYL snapshot] saved', { W, H, step });
    return snapshot;
  }

  window.__CYL = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    setTau,
    getStep: () => step,
    getDims: () => ({ W, H }),
    getParams: () => ({ A, B, TAU, VY, OMEGA, W, H, resLog2 }),
    getForceHistory: () => trajectory.slice(),
    debugRunAndCollect,
    debugStepSync,
    debugSnapshotSave,
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
        const fx = d[6], fy = d[7], tz = d[8];
        if (st.step < 500000) trajectory.push([st.step, fx, fy, tz]);

        if (performance.now() - lastT > 250) {
          statusEl.textContent = `[Reentry-dense] step ${st.step}  y=${d[20].toFixed(1)}  x=${d[21].toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${fy.toExponential(2)}  θ=${d[2].toFixed(2)}`;
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
