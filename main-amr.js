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

// ── Milestone 4 (plans/AMR.md): dynamic refinement via a fixed-capacity ───
// fine-block pool. Supersedes Milestone 2's single hardcoded fine region:
// refinement now happens at M1's own 8x8 coarse-block granularity, and any
// of MAX_FINE_BLOCKS pool slots can be assigned to any coarse block via
// blockSlot[]/slotToBlock[] indirection. Buffer-space-native throughout
// (unlike M2, which was window-anchored) -- M1's coarse blocks are already
// buffer-space, so this stays consistent; only the fine-level step kernel
// needs window coordinates, for the card SDF specifically.
const GHOST = 2;       // ghost layers per side, matches the 2-fine-substeps requirement
const BLOCK = 8;       // coarse block size (matches M1's cellIndex)
const RB = BLOCK;      // refine block size in coarse cells -- refine at block granularity
const FB = RB * 2 + 2 * GHOST; // per-slot fine buffer side length (20 for RB=8,GHOST=2)
const NCELLS1 = FB * FB; // cells per pool slot
const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 64;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY; // coarse block grid

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
// Fills the WHOLE pool (all MAX_FINE_BLOCKS slots), not just currently-
// assigned ones -- harmless since unassigned slots are never read (guarded
// by slotToBlock[slot]<0 in the shaders), and means a slot never holds
// uninitialized GPU memory between being freed and reassigned.
function initFPool() {
  const NPOOL = MAX_FINE_BLOCKS * NCELLS1;
  const f = new Float32Array(NPOOL * 9);
  for (let c = 0; c < NPOOL; c++) {
    for (let i = 0; i < 9; i++) {
      f[i * NPOOL + c] = feq(1, 0, 0, i);
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

  // Fine-block pool (Milestone 4): MAX_FINE_BLOCKS slots of NCELLS1 cells
  // each, plain flat layout within a slot (block-major-of-slots overall,
  // matching amr_step1.wgsl's `slot*(FB*FB) + local` indexing). Size is
  // independent of coarse domain size -- this is the actual memory-
  // footprint payoff (see plans/AMR.md's Milestone 4 design note).
  const fSizePool = MAX_FINE_BLOCKS * NCELLS1 * 9 * 4;
  const finePoolF_a   = device.createBuffer({ size: fSizePool, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const finePoolF_b   = device.createBuffer({ size: fSizePool, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const finePoolVel   = device.createBuffer({ size: MAX_FINE_BLOCKS * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  // blockSlot[NBLOCKS]: coarse block -> pool slot, or -1. slotToBlock is
  // its inverse (pool slot -> coarse block, or -1 if free) -- both are
  // needed since the coarse-side management pass indexes by block and the
  // fine-side interp/step/average passes are dispatched per pool slot (see
  // plans/AMR.md's Milestone 4 design note on why).
  const blockSlotBuf   = device.createBuffer({ size: NBLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const slotToBlockBuf = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(finePoolF_a, 0, initFPool());
  device.queue.writeBuffer(blockSlotBuf, 0, new Int32Array(NBLOCKS).fill(-1));
  device.queue.writeBuffer(slotToBlockBuf, 0, new Int32Array(MAX_FINE_BLOCKS).fill(-1));

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
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

  // Milestone 4: interp (coarse->fine ghosts), fine step, average (fine->coarse),
  // all pool-aware (an extra read-only slotToBlock/blockSlot binding vs. M2).
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const step1BGL = device.createBindGroupLayout({ label: 'step1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const avgBGL = device.createBindGroupLayout({ label: 'avgBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});

  const constants = { W, H };
  const fineConstants = { W, H, RB };
  // GHOST_ONLY=1: steady-state ghost-only reinterpolation (every macro-step).
  // GHOST_ONLY=0: full-slot fill, used once on block activation (see debugActivateBlock).
  const interpConstants = { W, H, RB, GHOST_ONLY: 1 };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0 };
  const step1Constants = { W, H, RB };

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
    compute: { module: interpSM, entryPoint: 'main', constants: interpConstants }
  });
  // Same module/entry point as interpPL, different override constant --
  // WGSL/WebGPU compiles this as a separate pipeline. Used once per newly-
  // activated slot to fill the whole region (no prior fine-level state to
  // evolve from), vs. interpPL's steady-state ghost-only reinterpolation.
  const interpInitPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: interpInitConstants }
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
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }, { binding: 2, resource: { buffer: finePoolVel } }, { binding: 3, resource: { buffer: blockSlotBuf } }]});

  // Milestone 4 bind groups (pool-aware, superseding M2's single-region ones).
  // interp always WRITES finePoolF_a (the pool's current-at-macro-step-
  // boundary buffer, mirroring f_a's own invariant -- 2 fine substeps per
  // macro-step is even), but READS whichever coarse buffer is "current"
  // this macro-step (same source the force pass reads).
  const interpBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  const interpBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  // Fine ping-pong within a macro-step is a fixed 2-call sequence (ab then
  // ba), not a persistent toggle like the coarse useB -- always call both,
  // in order, every macro-step.
  const step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: finePoolF_b } }, { binding: 3, resource: { buffer: finePoolVel } }, { binding: 4, resource: { buffer: slotToBlockBuf } }]});
  const step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: finePoolVel } }, { binding: 4, resource: { buffer: slotToBlockBuf } }]});
  // average always READS finePoolF_a (pool is current again after 2
  // substeps) but WRITES whichever coarse buffer the coarse step just
  // wrote this macro-step -- named by target, matching stepBG_ba being the
  // one that writes f_a.
  const avgBG_targetA = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  const avgBG_targetB = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  // Init variant (GHOST_ONLY=0, fills the whole slot): only ever called on
  // a just-activated slot immediately after coarse->fine interpolation
  // logically depends on the CURRENT coarse state, i.e. same source
  // selection as the steady-state interp bind groups above.
  const interpInitBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  const interpInitBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  // Milestone 4: interp/fine-step dispatch over (tile, tile, pool slot) --
  // cost scales with MAX_FINE_BLOCKS, not domain size (see plans/AMR.md's
  // Milestone 4 design note). average dispatches one workgroup per slot
  // exactly (RB*RB=8*8=64 cells = 1 workgroup, see amr_average_f2c.wgsl).
  const WGX1 = Math.ceil(FB / 8), WGY1 = Math.ceil(FB / 8);
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
  const stagingFPool   = device.createBuffer({ size: fSizePool, usage: U.MAP_READ | U.COPY_DST });
  const stagingVelPool = device.createBuffer({ size: MAX_FINE_BLOCKS * NCELLS1 * 2 * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingBlockSlot   = device.createBuffer({ size: NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingSlotToBlock = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });

  // Invariant this relies on: STEPS_PER_FRAME is even, so useB always
  // returns to its initial value (false) at a frame boundary, meaning f_a
  // (not f_b) is always the authoritative/current buffer whenever no frame
  // is mid-flight. Only call snapshot save/load while liveMode is false.
  async function debugSnapshotSave() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(f_a, 0, stagingF, 0, fSize);
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    enc.copyBufferToBuffer(cardStateBuf, 0, stagingCard, 0, 104);
    enc.copyBufferToBuffer(finePoolF_a, 0, stagingFPool, 0, fSizePool);
    enc.copyBufferToBuffer(finePoolVel, 0, stagingVelPool, 0, MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
    enc.copyBufferToBuffer(blockSlotBuf, 0, stagingBlockSlot, 0, NBLOCKS * 4);
    enc.copyBufferToBuffer(slotToBlockBuf, 0, stagingSlotToBlock, 0, MAX_FINE_BLOCKS * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stagingF.mapAsync(GPUMapMode.READ),
      stagingVel.mapAsync(GPUMapMode.READ),
      stagingCard.mapAsync(GPUMapMode.READ),
      stagingFPool.mapAsync(GPUMapMode.READ),
      stagingVelPool.mapAsync(GPUMapMode.READ),
      stagingBlockSlot.mapAsync(GPUMapMode.READ),
      stagingSlotToBlock.mapAsync(GPUMapMode.READ),
    ]);
    const f = new Float32Array(stagingF.getMappedRange()).slice();
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    const card = Array.from(new Float32Array(stagingCard.getMappedRange()).slice());
    const fPool = new Float32Array(stagingFPool.getMappedRange()).slice();
    const velPool = new Float32Array(stagingVelPool.getMappedRange()).slice();
    const blockSlotArr = Array.from(new Int32Array(stagingBlockSlot.getMappedRange()).slice());
    const slotToBlockArr = Array.from(new Int32Array(stagingSlotToBlock.getMappedRange()).slice());
    stagingF.unmap();
    stagingVel.unmap();
    stagingCard.unmap();
    stagingFPool.unmap();
    stagingVelPool.unmap();
    stagingBlockSlot.unmap();
    stagingSlotToBlock.unmap();

    const snapshot = {
      formatVersion: 4,
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
      // Milestone 4: fine-block pool (supersedes M2's single fine region).
      // pool.f/vel are indexed slot*(FB*FB)+local (see amr_step1.wgsl).
      pool: {
        RB, GHOST, FB, MAX_FINE_BLOCKS, NBLOCKS,
        blockSlot: blockSlotArr, slotToBlock: slotToBlockArr,
        fB64: bytesToB64(new Uint8Array(fPool.buffer, fPool.byteOffset, fPool.byteLength)),
        velB64: bytesToB64(new Uint8Array(velPool.buffer, velPool.byteOffset, velPool.byteLength)),
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
    if (snapshot.pool) {
      if (snapshot.pool.RB !== RB || snapshot.pool.MAX_FINE_BLOCKS !== MAX_FINE_BLOCKS || snapshot.pool.NBLOCKS !== NBLOCKS) {
        throw new Error(`snapshot pool (RB=${snapshot.pool.RB},MAX_FINE_BLOCKS=${snapshot.pool.MAX_FINE_BLOCKS},NBLOCKS=${snapshot.pool.NBLOCKS}) doesn't match this page's (RB=${RB},MAX_FINE_BLOCKS=${MAX_FINE_BLOCKS},NBLOCKS=${NBLOCKS})`);
      }
      const fPool = b64ToFloat32(snapshot.pool.fB64, MAX_FINE_BLOCKS * NCELLS1 * 9);
      const velPool = b64ToFloat32(snapshot.pool.velB64, MAX_FINE_BLOCKS * NCELLS1 * 2);
      device.queue.writeBuffer(finePoolF_a, 0, fPool.buffer, fPool.byteOffset, fSizePool);
      device.queue.writeBuffer(finePoolVel, 0, velPool.buffer, velPool.byteOffset, MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
      device.queue.writeBuffer(blockSlotBuf, 0, new Int32Array(snapshot.pool.blockSlot));
      device.queue.writeBuffer(slotToBlockBuf, 0, new Int32Array(snapshot.pool.slotToBlock));
      // Sync the CPU-side mirrors debugActivateBlock/debugDeactivateBlock
      // rely on -- omitting this would leave them reflecting whatever was
      // active before the load, not what the loaded snapshot actually has,
      // exactly the class of GPU/CPU-state desync bug this project has
      // already been bitten by once (see debugSnapshotSave's velBuf note).
      blockSlotCPU.set(snapshot.pool.blockSlot);
      slotToBlockCPU.set(snapshot.pool.slotToBlock);
      freeSlots = [];
      for (let slot = 0; slot < MAX_FINE_BLOCKS; slot++) {
        if (slotToBlockCPU[slot] === -1) freeSlots.push(slot);
      }
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
    // Z dimension selects pool slot -- cost scales with MAX_FINE_BLOCKS, not domain size.
    const ipl = enc.beginComputePass(); ipl.setPipeline(interpPL); ipl.setBindGroup(0, interpBG); ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); ipl.end();
    const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
    const f1a = enc.beginComputePass(); f1a.setPipeline(step1PL); f1a.setBindGroup(0, step1BG_ab); f1a.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1a.end();
    const f1b = enc.beginComputePass(); f1b.setPipeline(step1PL); f1b.setBindGroup(0, step1BG_ba); f1b.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1b.end();
    // Exactly one workgroup per slot (RB*RB=64 cells = 1 workgroup, see amr_average_f2c.wgsl).
    const avg = enc.beginComputePass(); avg.setPipeline(avgPL); avg.setBindGroup(0, avgBG); avg.dispatchWorkgroups(1, 1, MAX_FINE_BLOCKS); avg.end();

    useB = !useB;
  }

  // CPU-side mirror of blockSlot/slotToBlock, kept in sync with the GPU
  // buffers via small writeBuffer calls on every activate/deactivate.
  // Sub-step A (plans/AMR.md's Milestone 4 "staged landing" note): manual
  // CPU-orchestrated activation, proving the pool addressing mechanism
  // works, before wiring up the automatic vorticity criterion.
  let blockSlotCPU = new Int32Array(NBLOCKS).fill(-1);
  let slotToBlockCPU = new Int32Array(MAX_FINE_BLOCKS).fill(-1);
  let freeSlots = Array.from({ length: MAX_FINE_BLOCKS }, (_, i) => i);

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(finePoolF_a, 0, initFPool());
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    blockSlotCPU.fill(-1);
    slotToBlockCPU.fill(-1);
    device.queue.writeBuffer(blockSlotBuf, 0, blockSlotCPU);
    device.queue.writeBuffer(slotToBlockBuf, 0, slotToBlockCPU);
    freeSlots = Array.from({ length: MAX_FINE_BLOCKS }, (_, i) => i);
    useB = false;
    step = 0;
    trajectory.length = 0;
  }

  // Activates coarse block (bx,by) [0<=bx<NBX, 0<=by<NBY, buffer-space --
  // see plans/AMR.md's Milestone 4 design note on why block IDs are
  // buffer-space-native] against a free pool slot, filling the whole new
  // slot from the CURRENT coarse state (GHOST_ONLY=0 pipeline) since there
  // is no prior fine-level state for it to evolve from. Only valid while
  // liveMode is false, matching the debugSnapshotSave/Load convention --
  // dispatchMacroStep's useB toggling and this function's direct queue
  // writes would otherwise race the frame() loop's own encoder.
  async function debugActivateBlock(bx, by) {
    if (bx < 0 || bx >= NBX || by < 0 || by >= NBY) {
      throw new Error(`block (${bx},${by}) out of range [0,${NBX})x[0,${NBY})`);
    }
    const blockID = by * NBX + bx;
    if (blockSlotCPU[blockID] !== -1) return { slot: blockSlotCPU[blockID], alreadyActive: true };
    if (freeSlots.length === 0) throw new Error(`pool exhausted (MAX_FINE_BLOCKS=${MAX_FINE_BLOCKS})`);
    const slot = freeSlots.pop();
    blockSlotCPU[blockID] = slot;
    slotToBlockCPU[slot] = blockID;
    device.queue.writeBuffer(blockSlotBuf, blockID * 4, new Int32Array([slot]));
    device.queue.writeBuffer(slotToBlockBuf, slot * 4, new Int32Array([blockID]));

    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;
    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpInitPL);
    ipl.setBindGroup(0, interpInitBG);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return { slot, alreadyActive: false };
  }

  // Deactivates coarse block (bx,by). No explicit "final average" needed:
  // the average pass already runs every macro-step while the block is
  // active, so the coarse cells already reflect the latest fine-derived
  // state as of the most recent macro-step -- deactivation just stops
  // future fine-level evolution and frees the slot for reuse.
  function debugDeactivateBlock(bx, by) {
    if (bx < 0 || bx >= NBX || by < 0 || by >= NBY) {
      throw new Error(`block (${bx},${by}) out of range [0,${NBX})x[0,${NBY})`);
    }
    const blockID = by * NBX + bx;
    const slot = blockSlotCPU[blockID];
    if (slot === -1) return { wasActive: false };
    blockSlotCPU[blockID] = -1;
    slotToBlockCPU[slot] = -1;
    device.queue.writeBuffer(blockSlotBuf, blockID * 4, new Int32Array([-1]));
    device.queue.writeBuffer(slotToBlockBuf, slot * 4, new Int32Array([-1]));
    freeSlots.push(slot);
    return { wasActive: true, slot };
  }

  function debugListActiveBlocks() {
    const active = [];
    for (let blockID = 0; blockID < NBLOCKS; blockID++) {
      if (blockSlotCPU[blockID] !== -1) {
        active.push({ bx: blockID % NBX, by: Math.floor(blockID / NBX), slot: blockSlotCPU[blockID] });
      }
    }
    return active;
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
    debugActivateBlock,
    debugDeactivateBlock,
    debugListActiveBlocks,
    getBlockGridDims: () => ({ NBX, NBY, RB, MAX_FINE_BLOCKS }),
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
