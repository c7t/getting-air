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

// ── Milestone 2 (plans/AMR.md): static 2-level refinement ─────────────────
// Fixed fine region, window-space-anchored (not buffer-space -- see the
// plan's design note). GHOST=2 fine-cell layers matches the paper's own
// reasoning (2 fine substeps need 2 ghost layers to stay correctly fed).
const GHOST = 2;
const FINE_COARSE_W = 16, FINE_COARSE_H = 16; // fine region size, in COARSE cells
const FW = FINE_COARSE_W * 2, FH = FINE_COARSE_H * 2; // fine "real" interior cells
const FBW = FW + 2 * GHOST, FBH = FH + 2 * GHOST; // fine buffer incl. ghosts
const NCELLS1 = FBW * FBH;
// Defaults to card-centered; overridable via URL for the silence test (see
// plans/AMR.md's Milestone 2 validation section), which wants the fine
// region placed somewhere the card ISN'T.
const FINE_ORIGIN_X = urlParams.has('fineOriginX') ? parseInt(urlParams.get('fineOriginX')) : Math.floor(W / 2 - FINE_COARSE_W / 2);
const FINE_ORIGIN_Y = urlParams.has('fineOriginY') ? parseInt(urlParams.get('fineOriginY')) : Math.floor(H / 2 - FINE_COARSE_H / 2);

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

// The IC is spatially uniform (rho=1, u=0 everywhere), so the fine grid's
// t=0 state is trivially also uniform equilibrium -- interpolating a
// uniform coarse field gives back the same uniform field. No need for a
// real GPU interpolation dispatch at init.
function initF1() {
  const f = new Float32Array(NCELLS1 * 9);
  for (let c = 0; c < NCELLS1; c++) {
    for (let i = 0; i < 9; i++) {
      f[i * NCELLS1 + c] = feq(1, 0, 0, i);
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

  // WebGPU devices default to the spec MINIMUM limits (128 MiB storage
  // buffer bindings, 256 MiB total buffer size) regardless of what the
  // adapter can actually do -- f_a/f_b (NCELLS*9*4 bytes) exceeds the
  // default storage-binding limit at any resolution >= ~1536^2 (144 MiB at
  // 2048^2). This is the "WebGPU allocation limit" this project has hit
  // before; it's a device-limit *request* that was never made, unrelated
  // to AMR block size (the coarse grid is still one dense NCELLS-sized
  // buffer through Milestone 2 -- AMR's actual memory-footprint payoff
  // doesn't land until Milestone 4's block pool). Request exactly what the
  // current resolution needs, capped at the adapter's real capability, and
  // fail with a clear message rather than a cryptic validation error if
  // the requested resolution genuinely exceeds this GPU.
  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const neededBufferBytes = NCELLS * 9 * 4; // f_a/f_b: the largest storage-bound buffers
  if (neededBufferBytes > adapter.limits.maxStorageBufferBindingSize) {
    const mib = (b) => (b / 1048576).toFixed(0);
    statusEl.textContent = `error: ${W}x${H} needs a ${mib(neededBufferBytes)} MiB buffer binding, this GPU's max is ${mib(adapter.limits.maxStorageBufferBindingSize)} MiB`;
    return;
  }
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
    maxBufferSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
  };
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamp ? ['timestamp-query'] : [],
    requiredLimits,
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

  // Fine grid (Milestone 2): plain flat buffer, no block-major layout and no
  // moving-window circular addressing -- see plans/AMR.md's Milestone 2
  // scope notes for why.
  const fSize1  = NCELLS1 * 9 * 4;
  const f1_a    = device.createBuffer({ size: fSize1, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f1_b    = device.createBuffer({ size: fSize1, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const vel1Buf = device.createBuffer({ size: NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_SRC });

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(f1_a, 0, initF1());

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

  const [stepSM, frcSM, phySM, renSM, interpSM, step1SM, avgSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_force.wgsl'),
    loadShader(device, 'shaders/amr_physics.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    loadShader(device, 'shaders/amr_interp_c2f.wgsl'),
    loadShader(device, 'shaders/amr_step1.wgsl'),
    loadShader(device, 'shaders/amr_average_f2c.wgsl'),
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
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

  // Milestone 2: interp (coarse->fine ghosts), fine step, average (fine->coarse).
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const step1BGL = device.createBindGroupLayout({ label: 'step1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const avgBGL = device.createBindGroupLayout({ label: 'avgBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});

  const constants = { W, H };
  const fineConstants = { W, H, FW, FH, FINE_ORIGIN_X, FINE_ORIGIN_Y };
  const step1Constants = { FW, FH, FINE_ORIGIN_X, FINE_ORIGIN_Y };

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
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }], constants: fineConstants },
    primitive: { topology: 'triangle-list' },
  });
  const interpPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: fineConstants }
  });
  const step1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
    compute: { module: step1SM, entryPoint: 'main', constants: step1Constants }
  });
  const avgPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),
    compute: { module: avgSM, entryPoint: 'main', constants: fineConstants }
  });

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});

  const frcBG_a = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }]});
  const frcBG_b = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }]});

  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }, { binding: 2, resource: { buffer: vel1Buf } }]});

  // Milestone 2 bind groups.
  // interp always WRITES f1_a (the fine grid's current-at-macro-step-
  // boundary buffer, mirroring f_a's own invariant -- 2 fine substeps per
  // macro-step is even), but READS whichever coarse buffer is "current"
  // this macro-step (same source the force pass reads).
  const interpBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f1_a } }]});
  const interpBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f1_a } }]});
  // Fine ping-pong within a macro-step is a fixed 2-call sequence (ab then
  // ba), not a persistent toggle like the coarse useB -- always call both,
  // in order, every macro-step.
  const step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f1_a } }, { binding: 2, resource: { buffer: f1_b } }, { binding: 3, resource: { buffer: vel1Buf } }]});
  const step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f1_b } }, { binding: 2, resource: { buffer: f1_a } }, { binding: 3, resource: { buffer: vel1Buf } }]});
  // average always READS f1_a (fine grid is current again after 2 substeps)
  // but WRITES whichever coarse buffer the coarse step just wrote this
  // macro-step -- named by target, matching stepBG_ba being the one that
  // writes f_a.
  const avgBG_targetA = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f1_a } }, { binding: 2, resource: { buffer: f_a } }]});
  const avgBG_targetB = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f1_a } }, { binding: 2, resource: { buffer: f_b } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  // Milestone 2: fine buffer dispatch (interp + fine step, full FBWxFBH),
  // and average dispatch (one thread per coarse cell in the fine region).
  const WGX1 = Math.ceil(FBW / 8), WGY1 = Math.ceil(FBH / 8);
  const WGX_avg = Math.ceil((FW / 2) / 8), WGY_avg = Math.ceil((FH / 2) / 8);
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
  const stagingF1    = device.createBuffer({ size: fSize1, usage: U.MAP_READ | U.COPY_DST });
  const stagingVel1  = device.createBuffer({ size: NCELLS1 * 2 * 4, usage: U.MAP_READ | U.COPY_DST });

  // Invariant this relies on: STEPS_PER_FRAME is even, so useB always
  // returns to its initial value (false) at a frame boundary, meaning f_a
  // (not f_b) is always the authoritative/current buffer whenever no frame
  // is mid-flight. Only call snapshot save/load while liveMode is false.
  async function debugSnapshotSave() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(f_a, 0, stagingF, 0, fSize);
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    enc.copyBufferToBuffer(cardStateBuf, 0, stagingCard, 0, 104);
    enc.copyBufferToBuffer(f1_a, 0, stagingF1, 0, fSize1);
    enc.copyBufferToBuffer(vel1Buf, 0, stagingVel1, 0, NCELLS1 * 2 * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stagingF.mapAsync(GPUMapMode.READ),
      stagingVel.mapAsync(GPUMapMode.READ),
      stagingCard.mapAsync(GPUMapMode.READ),
      stagingF1.mapAsync(GPUMapMode.READ),
      stagingVel1.mapAsync(GPUMapMode.READ),
    ]);
    const f = new Float32Array(stagingF.getMappedRange()).slice();
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    const card = Array.from(new Float32Array(stagingCard.getMappedRange()).slice());
    const f1 = new Float32Array(stagingF1.getMappedRange()).slice();
    const vel1 = new Float32Array(stagingVel1.getMappedRange()).slice();
    stagingF.unmap();
    stagingVel.unmap();
    stagingCard.unmap();
    stagingF1.unmap();
    stagingVel1.unmap();

    const snapshot = {
      formatVersion: 3,
      // 'block8': f/vel are laid out in fixed 8x8 buffer-space cell-blocks
      // (see shaders/amr_step.wgsl's cellIndex, Milestone 1 of
      // plans/AMR.md), not flat row-major -- tools/amr-diff.js needs this
      // tag to decode snapshots correctly.
      layout: 'block8',
      W, H, step,
      cardState: card,
      fB64: bytesToB64(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)),
      velB64: bytesToB64(new Uint8Array(vel.buffer, vel.byteOffset, vel.byteLength)),
      params: { A, B, I_STAR, TAU, U_T, resLog2 },
      // Milestone 2: fine grid, plain flat layout (no block-major, no
      // circular buffer -- see plans/AMR.md). fine.f/vel are indexed
      // fy*FBW+fx directly.
      fine: {
        FW, FH, FBW, FBH, GHOST, FINE_ORIGIN_X, FINE_ORIGIN_Y,
        fB64: bytesToB64(new Uint8Array(f1.buffer, f1.byteOffset, f1.byteLength)),
        velB64: bytesToB64(new Uint8Array(vel1.buffer, vel1.byteOffset, vel1.byteLength)),
      },
    };
    console.log('[AMR snapshot] saved', { W, H, step });
    return snapshot;
  }

  async function debugSnapshotLoad(snapshot) {
    if (snapshot.W !== W || snapshot.H !== H) {
      throw new Error(`snapshot is ${snapshot.W}x${snapshot.H}, page is ${W}x${H} -- reload with ?res=${Math.log2(snapshot.W)}`);
    }
    // Raw f_a/velBuf bytes are only meaningful under the layout they were
    // captured with (see debugSnapshotSave's 'layout' field) -- loading a
    // pre-Milestone-1 flat-row-major snapshot here would silently
    // reinterpret it as block-major and corrupt state with no thrown error,
    // exactly the class of silent-failure this project has learned to
    // guard against explicitly rather than discover from wrong output.
    if (snapshot.layout !== 'block8') {
      throw new Error(`snapshot layout is '${snapshot.layout}', this build expects 'block8'`);
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
    if (snapshot.fine) {
      if (snapshot.fine.FW !== FW || snapshot.fine.FH !== FH ||
          snapshot.fine.FINE_ORIGIN_X !== FINE_ORIGIN_X || snapshot.fine.FINE_ORIGIN_Y !== FINE_ORIGIN_Y) {
        throw new Error(`snapshot fine region (FW=${snapshot.fine.FW},FH=${snapshot.fine.FH},origin=${snapshot.fine.FINE_ORIGIN_X},${snapshot.fine.FINE_ORIGIN_Y}) doesn't match this page's (FW=${FW},FH=${FH},origin=${FINE_ORIGIN_X},${FINE_ORIGIN_Y})`);
      }
      const f1 = b64ToFloat32(snapshot.fine.fB64, NCELLS1 * 9);
      const vel1 = b64ToFloat32(snapshot.fine.velB64, NCELLS1 * 2);
      device.queue.writeBuffer(f1_a, 0, f1.buffer, f1.byteOffset, fSize1);
      device.queue.writeBuffer(vel1Buf, 0, vel1.buffer, vel1.byteOffset, NCELLS1 * 2 * 4);
    }
    useB = false;
    step = snapshot.step;
    console.log('[AMR snapshot] loaded', { W, H, step });
    return { step };
  }

  // Milestone 2 macro-step (plans/AMR.md): 1 coarse step + 2 fine substeps,
  // ordered per AGAL's Fig. 13 recursive routine -- interpolate ghosts from
  // the CURRENT (pre-step) coarse state, then coarse-step and fine-step-x2
  // independently (both read only pre-step data, so their relative order
  // doesn't matter), then average the now-twice-advanced fine interior back
  // onto the coarse cells the coarse step just (less accurately) computed.
  // Factored out of frame()'s loop so debugStepSync can reuse it exactly --
  // duplicating this 7-pass sequence would risk the two silently drifting
  // apart.
  function dispatchMacroStep(enc) {
    const stepBG   = useB ? stepBG_ba      : stepBG_ab;
    const frcBG    = useB ? frcBG_b        : frcBG_a;
    const interpBG = useB ? interpBG_readB : interpBG_readA;
    const avgBG    = useB ? avgBG_targetA  : avgBG_targetB;

    const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
    const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
    const ipl = enc.beginComputePass(); ipl.setPipeline(interpPL); ipl.setBindGroup(0, interpBG); ipl.dispatchWorkgroups(WGX1, WGY1); ipl.end();
    const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
    const f1a = enc.beginComputePass(); f1a.setPipeline(step1PL); f1a.setBindGroup(0, step1BG_ab); f1a.dispatchWorkgroups(WGX1, WGY1); f1a.end();
    const f1b = enc.beginComputePass(); f1b.setPipeline(step1PL); f1b.setBindGroup(0, step1BG_ba); f1b.dispatchWorkgroups(WGX1, WGY1); f1b.end();
    const avg = enc.beginComputePass(); avg.setPipeline(avgPL); avg.setBindGroup(0, avgBG); avg.dispatchWorkgroups(WGX_avg, WGY_avg); avg.end();

    useB = !useB;
  }

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(f1_a, 0, initF1());
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    useB = false;
    step = 0;
    trajectory.length = 0;
  }

  // Deterministic synchronous stepping, bypassing rAF entirely -- lets two
  // separate builds be driven to an EXACT matching step count for a fair
  // diff. Wall-clock polling of the normal rAF-driven `liveMode` loop can't
  // guarantee this: STEPS_PER_FRAME-sized jumps land unpredictably relative
  // to any external poll interval (confirmed directly while re-validating
  // Milestones 1 and 2 at 256x256 -- see plans/AMR.md).
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

  window.__AMR = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    getStep: () => step,
    getDims: () => ({ W, H }),
    debugSnapshotSave,
    debugSnapshotLoad,
    debugStepSync,
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

      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
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
