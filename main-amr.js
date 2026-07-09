// AMR dev build. Starts as a literal fork of main.js (same physics, same
// shaders content under new amr_*.wgsl filenames) so it can diverge without
// ever touching the reference sim in main.js/shaders/lbm_*.wgsl. See
// plans/AMR.md Milestone 0. Adds: a pause/resume + snapshot/restore debug
// API (window.__AMR) for CDP-driven verification tooling (tools/amr-*.js),
// modeled on the vpm branch's window.__VPM / debugSnapshotSave/Load, and a
// per-frame GPU validation error scope -- the vpm branch hit a real silent
// CPU<->GPU transfer failure from a buffer declared with the wrong usage
// flags (commit 83d3c8c), so this build checks eagerly rather than
// discovering that kind of bug from wrong-looking output.

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseInt(urlParams.get('res')) || 8;
if (resLog2 < 6) resLog2 = 6;
if (resLog2 > 11) resLog2 = 11;

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
// These constants define the "regime" of the simulation (Falling Paper).

let A = 64, B = 8;
let I_STAR = 0.34;
let TAU = 0.509;
let U_T = 0.05;

let RHO_B, MASS, I_BODY, G_LU, G_EFF;

function recalculate() {
  RHO_B  = I_STAR * 2 * A**3 / (B * (A**2 + B**2));
  RHO_B  = Math.max(1.05, RHO_B);
  MASS   = RHO_B * Math.PI * A * B;
  I_BODY = RHO_B * Math.PI * A * B * (A**2 + B**2) / 4;
  G_LU   = U_T**2 / (Math.PI * B * (RHO_B - 1));
  G_EFF  = G_LU * (1 - 1 / RHO_B);
}
recalculate();

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

function initCardState() {
  return new Float32Array([
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

// base64 chunked in 8192-byte pieces -- a single huge String.fromCharCode
// spread risks "Maximum call stack size exceeded" for larger grids.
function bytesToB64(bytes) {
  let binary = '';
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
function b64ToFloat32(b64, floatCount) {
  const binary = atob(b64);
  const bytes = new Uint8Array(floatCount * 4);
  for (let i = 0; i < bytes.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }

  // Milestone 6 needs real per-level GPU timing; leave this on for the AMR
  // dev build from the start (main.js keeps it off with `0 &&` -- don't
  // touch that file, this is deliberately different here).
  const hasTimestamp = adapter.features.has('timestamp-query');
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
  // COPY_SRC added on both f buffers (main.js's f_b lacks it) so debug
  // snapshotting can read back whichever buffer is authoritative without
  // needing a bind-group-layout-specific copy path. Flagged explicitly
  // because this exact class of bug (buffer usage flags silently wrong)
  // already bit the vpm branch once (commit 83d3c8c).
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 26 floats = 104 bytes
  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());

  let paramsDirty = false;
  const updateGPUParams = () => {
    const data = new Float32Array([MASS, I_BODY, G_EFF, A, B]);
    device.queue.writeBuffer(cardStateBuf, 9 * 4, data);
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
  };

  const sliders = [
    { id: 'A', setter: v => A = v, dp: 0 },
    { id: 'B', setter: v => B = v, dp: 0 },
    { id: 'I_STAR', setter: v => I_STAR = v, dp: 2 },
    { id: 'TAU', setter: v => TAU = v, dp: 3 },
    { id: 'U_T', setter: v => U_T = v, dp: 3 },
  ];
  sliders.forEach(s => {
    const el = document.getElementById(`slider-${s.id}`);
    const valEl = document.getElementById(`val-${s.id}`);
    el.oninput = () => {
      s.setter(parseFloat(el.value));
      valEl.textContent = el.value;
      recalculate();
      paramsDirty = true;
    };
  });

  const [stepSM, frcSM, phySM, renSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_force.wgsl'),
    loadShader(device, 'shaders/amr_physics.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
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
  let liveMode = true;

  const trajectory = [];

  document.getElementById('download').onclick = () => {
    const header = "step,cx,cy_total,cx_total,theta,vx,vy,omega,fx,fy,tz\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trajectory_amr_${W}x${H}.csv`;
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

  // ── Debug/verification support (window.__AMR) ────────────────────────────
  // Dedicated staging buffers, separate from the triple-buffered readback
  // stages above, so debug reads can't race frame()'s own in-flight readback.
  const stagingF     = device.createBuffer({ size: fSize, usage: U.MAP_READ | U.COPY_DST });
  const stagingVel   = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingCard  = device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST });

  // Invariant this relies on: STEPS_PER_FRAME is even, so useB always
  // returns to its initial value (false) at a frame boundary, meaning f_a
  // (not f_b) is always the authoritative/current buffer whenever no frame
  // is mid-flight. Only call snapshot save/load while liveMode is false.
  async function debugSnapshotSave() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(f_a, 0, stagingF, 0, fSize);
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    enc.copyBufferToBuffer(cardStateBuf, 0, stagingCard, 0, 104);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stagingF.mapAsync(GPUMapMode.READ),
      stagingVel.mapAsync(GPUMapMode.READ),
      stagingCard.mapAsync(GPUMapMode.READ),
    ]);
    const f = new Float32Array(stagingF.getMappedRange()).slice();
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    const card = Array.from(new Float32Array(stagingCard.getMappedRange()).slice());
    stagingF.unmap();
    stagingVel.unmap();
    stagingCard.unmap();

    const snapshot = {
      formatVersion: 1,
      W, H, step,
      cardState: card,
      fB64: bytesToB64(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)),
      velB64: bytesToB64(new Uint8Array(vel.buffer, vel.byteOffset, vel.byteLength)),
      params: { A, B, I_STAR, TAU, U_T, resLog2 },
    };
    console.log('[AMR snapshot] saved', { W, H, step });
    return snapshot;
  }

  async function debugSnapshotLoad(snapshot) {
    if (snapshot.W !== W || snapshot.H !== H) {
      throw new Error(`snapshot is ${snapshot.W}x${snapshot.H}, page is ${W}x${H} -- reload with ?res=${Math.log2(snapshot.W)}`);
    }
    const f = b64ToFloat32(snapshot.fB64, NCELLS * 9);
    const vel = b64ToFloat32(snapshot.velB64, NCELLS * 2);
    device.queue.writeBuffer(f_a, 0, f.buffer, f.byteOffset, fSize);
    // velBuf is a separate GPU buffer, not derived from f_a by anything
    // debugSnapshotLoad itself runs -- omitting this write left it holding
    // whatever was there before the load (stale ux/uy from a prior run)
    // until the next real step overwrote it. Caught by amr-diff.js: rho
    // (derived from f in the diff tool) round-tripped exactly, but ux/uy
    // (read from velBuf) didn't -- the asymmetry was the tell.
    device.queue.writeBuffer(velBuf, 0, vel.buffer, vel.byteOffset, NCELLS * 2 * 4);
    device.queue.writeBuffer(cardStateBuf, 0, new Float32Array(snapshot.cardState));
    useB = false;
    step = snapshot.step;
    console.log('[AMR snapshot] loaded', { W, H, step });
    return { step };
  }

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    useB = false;
    step = 0;
    trajectory.length = 0;
  }

  window.__AMR = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    getStep: () => step,
    getDims: () => ({ W, H }),
    debugSnapshotSave,
    debugSnapshotLoad,
  };

  async function frame() {
    try {
      if (!liveMode) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }
      if (paramsDirty) {
        updateGPUParams();
        paramsDirty = false;
      }

      const stage = stages[currentStageIdx];
      // Backpressure: if the oldest stage is still in flight, we must wait.
      if (stage.inFlight) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      device.pushErrorScope('validation');
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
      device.popErrorScope().then(err => { if (err) handleErr(err); });

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
          gpuTime = performance.now() - tSubmit;
        }

        if (st.step < 100000) {
          trajectory.push([st.step, d[0], d[20], d[21], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]);
        }

        if (performance.now() - lastT > 250) {
          const mlups = (NCELLS * STEPS_PER_FRAME) / (gpuTime * 1e3);
          mlupsEl.textContent = mlups.toFixed(1);
          gpuMsEl.textContent = gpuTime.toFixed(2);
          syncMsEl.textContent = (performance.now() - tSubmit).toFixed(2);
          statusEl.textContent = `[AMR-dev] step ${st.step}  y=${d[20].toFixed(1)}  x=${d[21].toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`;
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
