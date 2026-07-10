// AMR validation harness: a fixed circular cylinder in uniform crossflow,
// driven by the exact same AMR machinery as main-amr.js (buffer layout,
// pipelines, dispatchMacroStep's refine/interp/step/average sequence --
// none of that is touched here). Only the scenario differs: pinned body
// (v_max=o_max=0, g_eff=0), a diameters-normalized radius/placement
// (BLOCKAGE/UPSTREAM, see main-cylinder.js's identical rationale), and a
// uniform freestream sponge target via amr_step.wgsl/amr_step1.wgsl's
// SPONGE_UX/UY overrides instead of the falling card's quiescent far field.
//
// Milestone 8 (plans/AMR-multilevel.md): this is still a fixed 2-level
// build (never got main-amr.js's Milestone 5-7 pools[]/N_LEVELS/S_Advance
// forward-ported -- that's Milestone 10's job, not this one), but it DOES
// need level 1's own force pass (amr_force1.wgsl, HAS_CHILD=0 since there's
// no level 2 here) and amr_force.wgsl's finest-wins masking, since both
// shader files are shared verbatim with main-amr.js and this harness is
// exactly the "coarse L0 + refined L1" scenario M8 exists to fix (the
// force/torque driving Cd/Cl was aliased at coarse-only integration).
// amr_step1.wgsl's epsilon scaling (1.5 -> 0.75 for L1) comes along for
// free from the shared shader file, no JS-side change needed for that part.
//
// window.__CYL matches main-cylinder.js's shape exactly (setRe, reset,
// getParams, getForceHistory, debugRunAndCollect) so tools/validate-
// cylinder.js drives this page unmodified -- only --url differs. That's
// what lets it (or a descendant) directly compare AMR against the dense
// solver using identical Cd/St extraction and the same literature table.

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseInt(urlParams.get('res')) || 9;
if (resLog2 < 7) resLog2 = 7;
if (resLog2 > 11) resLog2 = 11;

let W = 1 << resLog2;
let H = W;
let NCELLS = W * H;

// Milestone 4 fine-block pool sizing -- same defaults/URL params as
// main-amr.js (see that file's comments for the full rationale); unchanged
// by the cylinder scenario.
const GHOST = 2;
const BLOCK = 8;
const RB = BLOCK;
const FB = RB * 2 + 2 * GHOST;
const NCELLS1 = FB * FB;
const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 128;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY;

// REFINE_THRESH/COARSEN_THRESH default to -6/-7 in main-amr.js (calibrated
// for the falling-card scenario's vorticity scale). Retuned to -8/-9 here:
// swept -6/-7 (18 active blocks, clustered on the body, Cd~1.60 at Re=100
// vs a 1.35 literature target), -7/-8 (24-31 blocks, Cd~1.58), -8/-9 (62-75
// blocks, Cd~1.48) with maxFineBlocks temporarily uncapped to see true
// demand at each threshold before picking a default -- -8/-9 extends
// refinement into the near wake (not just the body halo) and measurably
// improves Cd, while peaking at ~75/128 pool slots (59% utilization), real
// headroom below MAX_FINE_BLOCKS so amr_manage.wgsl's refine() shouldn't
// hit its silent "pool exhausted this round -- stay coarse" fallback
// (which would otherwise produce patchy, dispatch-order-dependent
// refinement instead of a clean wake, with no error to signal it).
//
// Full validate-cylinder.js result at -8/-9: Re=100 Cd 1.590->1.482 (now
// PASSES), Re=200 Cd 1.728->1.805 (WORSE, moved further from the 1.34
// target), St unchanged at both Re (~0.13 vs 0.165 target at Re=100, ~0.15
// vs 0.197 at Re=200) despite active blocks going from 18 to ~70 -- so
// more wake refinement isn't the fix for St, and isn't a uniform win for
// Cd either. Kept as the default anyway (net improvement, and the
// regression is real information, not a reason to hide it) but the Re=200
// regression needs its own investigation before assuming lower thresholds
// are strictly better.
const REFINE_EVERY = urlParams.has('refineEvery') ? parseInt(urlParams.get('refineEvery')) : 16;
const REFINE_THRESH = urlParams.has('refineThresh') ? parseFloat(urlParams.get('refineThresh')) : -8;
const COARSEN_THRESH = urlParams.has('coarsenThresh') ? parseFloat(urlParams.get('coarsenThresh')) : -9;
const FORCE_REFINE_MARGIN = urlParams.has('forceRefineMargin') ? parseFloat(urlParams.get('forceRefineMargin')) : 8;
const FORCE_REFINE_LOOKAHEAD = urlParams.has('forceRefineLookahead') ? parseFloat(urlParams.get('forceRefineLookahead')) : REFINE_EVERY;

// BLOCKAGE/UPSTREAM/R/U0/RE/TAU: identical rationale to main-cylinder.js --
// R is derived from W and BLOCKAGE (not an independent lattice-unit param)
// so bumping `res` alone is a valid grid-convergence sweep, and the
// cylinder is placed UPSTREAM diameters from the inlet edge, not a fixed
// fraction of W.
let BLOCKAGE = parseFloat(urlParams.get('blockage')) || 24;
let UPSTREAM = parseFloat(urlParams.get('upstream')) || 8;
let R = W / (2 * BLOCKAGE);

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

// PERTURB/SEED: identical rationale to main-cylinder.js -- both the
// analytic circle and a pure freestream IC are exactly top/bottom
// symmetric, so shedding onset needs a deliberate, resolution-independent
// seed rather than relying on grid-dependent round-off. One rng, drawn
// across coarse init then pool init, both deterministic from SEED.
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
const rng = mulberry32(SEED);

function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    const uy = (rng() * 2 - 1) * PERTURB * U0;
    for (let i = 0; i < 9; i++) {
      f[i * NCELLS + c] = feq(1, U0, uy, i);
    }
  }
  return f;
}

// Pool cells all start at the (perturbed) freestream too, same rationale as
// main-amr.js's initFPool: a newly-activated slot is filled from the
// coarse state anyway, so this is just what an as-yet-unassigned slot holds.
function initFPool() {
  const NPOOL = MAX_FINE_BLOCKS * NCELLS1;
  const f = new Float32Array(NPOOL * 9);
  for (let c = 0; c < NPOOL; c++) {
    const uy = (rng() * 2 - 1) * PERTURB * U0;
    for (let i = 0; i < 9; i++) {
      f[i * NPOOL + c] = feq(1, U0, uy, i);
    }
  }
  return f;
}

// Cylinder placed UPSTREAM diameters from the inlet edge, centered
// transversely -- see main-cylinder.js's cardInit comment for why this is
// in diameters, not a fixed W/4 or W/2.
const CX0 = UPSTREAM * 2 * R;
const CY0 = H / 2;
function initCardState() {
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

  // Same device-limits handling as main-amr.js -- f_a/f_b exceed the
  // spec-minimum storage-binding limit above ~1536^2.
  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const neededBufferBytes = NCELLS * 9 * 4;
  if (neededBufferBytes > adapter.limits.maxStorageBufferBindingSize) {
    const mib = (b) => (b / 1048576).toFixed(0);
    statusEl.textContent = `error: ${W}x${H} needs a ${mib(neededBufferBytes)} MiB buffer binding, this GPU's max is ${mib(adapter.limits.maxStorageBufferBindingSize)} MiB`;
    return;
  }
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
    maxBufferSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
  };
  const device = await adapter.requestDevice({ requiredLimits });

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
  const f_a      = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b      = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const velBuf   = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  // Milestone 8: harmless placeholder for force1PL's "child level's
  // blockSlot" binding -- this harness has no level>=2 (HAS_CHILD=0 always
  // masks it out), see main-amr.js's identical dummyBlockSlotBuf comment.
  const dummyBlockSlotBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyBlockSlotBuf, 0, new Int32Array([-1]));

  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  const fSizePool = MAX_FINE_BLOCKS * NCELLS1 * 9 * 4;
  const finePoolF_a   = device.createBuffer({ size: fSizePool, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const finePoolF_b   = device.createBuffer({ size: fSizePool, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const finePoolVel   = device.createBuffer({ size: MAX_FINE_BLOCKS * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  const blockSlotBuf   = device.createBuffer({ size: NBLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const slotToBlockBuf = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  const blockCriterionBuf  = device.createBuffer({ size: NBLOCKS * 4, usage: U.STORAGE | U.COPY_DST });
  const freeListBuf        = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const freeCountBuf       = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const newlyActivatedBuf  = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST });

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(finePoolF_a, 0, initFPool());
  device.queue.writeBuffer(blockSlotBuf, 0, new Int32Array(NBLOCKS).fill(-1));
  device.queue.writeBuffer(slotToBlockBuf, 0, new Int32Array(MAX_FINE_BLOCKS).fill(-1));
  device.queue.writeBuffer(freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
  device.queue.writeBuffer(freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));

  // Only TAU is live-adjustable post-init (R/U0/RES are pipeline-baked).
  // Writes immediately (not through a paramsDirty flag polled by frame())
  // so it works the same from the live loop and from CDP tooling's
  // deterministic debugRunAndCollect -- see main-cylinder.js's identical note.
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

  const [stepSM, frcSM, phySM, renSM, interpSM, step1SM, avgSM, criterionSM, manageSM, force1SM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_force.wgsl'),
    loadShader(device, 'shaders/amr_physics.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    // Milestone 6 renamed this file (dense/pool-parent addressing split);
    // this harness has no level>=2, so it only ever needs the dense-parent
    // case (main-amr.js's own L0->L1 shader).
    loadShader(device, 'shaders/amr_interp_dense_parent.wgsl'),
    loadShader(device, 'shaders/amr_step1.wgsl'),
    loadShader(device, 'shaders/amr_average_f2c.wgsl'),
    loadShader(device, 'shaders/amr_criterion.wgsl'),
    loadShader(device, 'shaders/amr_manage.wgsl'),
    // Milestone 8: level 1's own force pass -- see amr_force1.wgsl's header.
    loadShader(device, 'shaders/amr_force1.wgsl'),
  ]);

  const stepBGL = device.createBindGroupLayout({ label: 'stepBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  // Milestone 8: binding 3 (blockSlot1) is the finest-wins masking check --
  // see amr_force.wgsl's header.
  const frcBGL = device.createBindGroupLayout({ label: 'frcBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 8: level 1's own force pass. This harness has no level>=2, so
  // HAS_CHILD is always 0 and binding 4 is a harmless dummy -- see
  // amr_force1.wgsl's header.
  const force1BGL = device.createBindGroupLayout({ label: 'force1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const phyBGL = device.createBindGroupLayout({ label: 'phyBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    // overlayOpacity (binding 4): refinement-coverage overlay opacity [0,1],
    // added to amr_render.wgsl by the pulled overlay-toggle change. No UI
    // slider in this harness (headless/CDP-driven) -- just bind a fixed
    // value so the render pipeline matches the shader's binding layout.
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
  ]});
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const criterionBGL = device.createBindGroupLayout({ label: 'criterionBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const manageBGL = device.createBindGroupLayout({ label: 'manageBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
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
  // Coarse step needs the freestream sponge target; force/physics/render/
  // criterion/manage don't reference SPONGE_UX/UY at all, so they keep the
  // plain {W,H} constants dict.
  const stepConstants = { W, H, SPONGE_UX: U0, SPONGE_UY: 0 };
  const fineConstants = { W, H, RB };
  const interpConstants = { W, H, RB, GHOST_ONLY: 1 };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0 };
  const interpFFConstants = { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1 };
  // Fine step also needs the freestream sponge target (see amr_step1.wgsl's
  // SPONGE_UX/UY addition -- it has its own copy of the sponge, not shared
  // with the coarse kernel).
  const step1Constants = { W, H, RB, SPONGE_UX: U0, SPONGE_UY: 0 };
  const criterionConstants = { W, H };
  const manageConstants = { W, H, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD };

  const stepPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
    compute: { module: stepSM, entryPoint: 'main', constants: stepConstants }
  });
  const frcPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }),
    compute: { module: frcSM, entryPoint: 'main', constants }
  });
  // Milestone 8: HAS_CHILD=0 -- this harness has no level>=2.
  const force1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [force1BGL] }),
    compute: { module: force1SM, entryPoint: 'main', constants: { W, H, RB, HAS_CHILD: 0 } }
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
  const interpInitPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: interpInitConstants }
  });
  const interpFFPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: interpFFConstants }
  });
  const step1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
    compute: { module: step1SM, entryPoint: 'main', constants: step1Constants }
  });
  const avgPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),
    compute: { module: avgSM, entryPoint: 'main', constants: fineConstants }
  });
  const criterionPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [criterionBGL] }),
    compute: { module: criterionSM, entryPoint: 'main', constants: criterionConstants }
  });
  const manageCoarsenPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'coarsen', constants: manageConstants }
  });
  const manageRefinePL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'refine', constants: manageConstants }
  });

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});

  const frcBG_a = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: blockSlotBuf } }]});
  const frcBG_b = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: blockSlotBuf } }]});
  // Milestone 8: level 1's own force pass. Always reads finePoolF_a --
  // level 1's own buffer is always "current" at a macro-step boundary
  // (same invariant as main-amr.js's own force1BG).
  const force1BG = device.createBindGroup({ layout: force1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: dummyBlockSlotBuf } }]});

  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const overlayOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(overlayOpacityBuf, 0, new Float32Array([1.0]));
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }, { binding: 2, resource: { buffer: finePoolVel } }, { binding: 3, resource: { buffer: blockSlotBuf } }, { binding: 4, resource: { buffer: overlayOpacityBuf } }]});

  const interpBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  const interpBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  const step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: finePoolF_b } }, { binding: 3, resource: { buffer: finePoolVel } }, { binding: 4, resource: { buffer: slotToBlockBuf } }]});
  const step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: finePoolVel } }, { binding: 4, resource: { buffer: slotToBlockBuf } }]});
  const interpFFBG_b = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_b } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  const avgBG_targetA = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  const avgBG_targetB = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  const interpInitBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  const interpInitBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});

  const criterionBG = device.createBindGroup({ layout: criterionBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: blockCriterionBuf } }]});
  const manageBG = device.createBindGroup({ layout: manageBGL, entries: [{ binding: 0, resource: { buffer: blockCriterionBuf } }, { binding: 1, resource: { buffer: blockSlotBuf } }, { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: freeListBuf } }, { binding: 4, resource: { buffer: freeCountBuf } }, { binding: 5, resource: { buffer: newlyActivatedBuf } }, { binding: 6, resource: { buffer: cardStateBuf } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const WGX1 = Math.ceil(FB / 8), WGY1 = Math.ceil(FB / 8);
  const WG_MANAGE = Math.ceil(NBLOCKS / 64);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;
  let autoRefine = true;
  let macroStepCounter = 0;

  const D = 2 * R;
  // trajectory rows: [step, fx, fy, Cd, Cl] -- same shape as main-cylinder.js.
  const trajectory = [];

  document.getElementById('download').onclick = () => {
    const header = "step,fx,fy,Cd,Cl\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cylinder_amr_Re${RE}_${W}x${H}.csv`;
    a.click();
  };

  const STAGES = 3;
  const stages = Array.from({ length: STAGES }, () => ({
    card: device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST }),
    inFlight: false,
    step: 0
  }));
  let currentStageIdx = 0;

  const stagingBlockSlot   = device.createBuffer({ size: NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingSlotToBlock = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
  async function readPoolIndirection() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(blockSlotBuf, 0, stagingBlockSlot, 0, NBLOCKS * 4);
    enc.copyBufferToBuffer(slotToBlockBuf, 0, stagingSlotToBlock, 0, MAX_FINE_BLOCKS * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stagingBlockSlot.mapAsync(GPUMapMode.READ),
      stagingSlotToBlock.mapAsync(GPUMapMode.READ),
    ]);
    const blockSlot = new Int32Array(stagingBlockSlot.getMappedRange()).slice();
    const slotToBlock = new Int32Array(stagingSlotToBlock.getMappedRange()).slice();
    stagingBlockSlot.unmap();
    stagingSlotToBlock.unmap();
    return { blockSlot, slotToBlock };
  }
  async function debugListActiveBlocks() {
    const { blockSlot } = await readPoolIndirection();
    const active = [];
    for (let blockID = 0; blockID < NBLOCKS; blockID++) {
      if (blockSlot[blockID] !== -1) {
        active.push({ bx: blockID % NBX, by: Math.floor(blockID / NBX), slot: blockSlot[blockID] });
      }
    }
    return active;
  }

  // Milestone 2/4 macro-step (unchanged from main-amr.js -- this sequence,
  // including the refine/coarsen re-evaluation cadence, is exactly what's
  // under test here, not something the cylinder scenario should alter).
  function dispatchMacroStep(enc) {
    const stepBG       = useB ? stepBG_ba          : stepBG_ab;
    const frcBG        = useB ? frcBG_b            : frcBG_a;
    const interpBG     = useB ? interpBG_readB     : interpBG_readA;
    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;
    const avgBG        = useB ? avgBG_targetA      : avgBG_targetB;

    if (autoRefine && macroStepCounter % REFINE_EVERY === 0) {
      enc.clearBuffer(newlyActivatedBuf);
      const crit = enc.beginComputePass(); crit.setPipeline(criterionPL); crit.setBindGroup(0, criterionBG); crit.dispatchWorkgroups(WGX, WGY); crit.end();
      const coarsenP = enc.beginComputePass(); coarsenP.setPipeline(manageCoarsenPL); coarsenP.setBindGroup(0, manageBG); coarsenP.dispatchWorkgroups(WG_MANAGE); coarsenP.end();
      const refineP = enc.beginComputePass(); refineP.setPipeline(manageRefinePL); refineP.setBindGroup(0, manageBG); refineP.dispatchWorkgroups(WG_MANAGE); refineP.end();
      const initP = enc.beginComputePass(); initP.setPipeline(interpInitPL); initP.setBindGroup(0, interpInitBG); initP.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); initP.end();
    }
    macroStepCounter++;

    const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
    // Milestone 8: level 1's own force contribution, before `phy` drains
    // the shared atomic forces[] buffer -- see main-amr.js's identical
    // comment on ordering/commutativity.
    const f1frc = enc.beginComputePass(); f1frc.setPipeline(force1PL); f1frc.setBindGroup(0, force1BG); f1frc.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1frc.end();
    const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
    const ipl = enc.beginComputePass(); ipl.setPipeline(interpPL); ipl.setBindGroup(0, interpBG); ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); ipl.end();
    const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
    const f1a = enc.beginComputePass(); f1a.setPipeline(step1PL); f1a.setBindGroup(0, step1BG_ab); f1a.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1a.end();
    const ff = enc.beginComputePass(); ff.setPipeline(interpFFPL); ff.setBindGroup(0, interpFFBG_b); ff.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); ff.end();
    const f1b = enc.beginComputePass(); f1b.setPipeline(step1PL); f1b.setBindGroup(0, step1BG_ba); f1b.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1b.end();
    const avg = enc.beginComputePass(); avg.setPipeline(avgPL); avg.setBindGroup(0, avgBG); avg.dispatchWorkgroups(1, 1, MAX_FINE_BLOCKS); avg.end();

    useB = !useB;
  }

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
    device.queue.writeBuffer(freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
    device.queue.writeBuffer(freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
    autoRefine = true;
    macroStepCounter = 0;
    useB = false;
    step = 0;
    trajectory.length = 0;
  }

  async function setAutoRefine(v) {
    autoRefine = !!v;
    if (!autoRefine) {
      const { blockSlot, slotToBlock } = await readPoolIndirection();
      blockSlotCPU.set(blockSlot);
      slotToBlockCPU.set(slotToBlock);
      freeSlots = [];
      for (let slot = 0; slot < MAX_FINE_BLOCKS; slot++) {
        if (slotToBlockCPU[slot] === -1) freeSlots.push(slot);
      }
    }
  }

  async function debugActivateBlock(bx, by) {
    if (autoRefine) throw new Error('debugActivateBlock: disable autoRefine first (setAutoRefine(false))');
    if (bx < 0 || bx >= NBX || by < 0 || by >= NBY) throw new Error(`block (${bx},${by}) out of range`);
    const blockID = by * NBX + bx;
    if (blockSlotCPU[blockID] !== -1) return { slot: blockSlotCPU[blockID], alreadyActive: true };
    if (freeSlots.length === 0) throw new Error(`pool exhausted (MAX_FINE_BLOCKS=${MAX_FINE_BLOCKS})`);
    const slot = freeSlots.pop();
    blockSlotCPU[blockID] = slot;
    slotToBlockCPU[slot] = blockID;
    device.queue.writeBuffer(blockSlotBuf, blockID * 4, new Int32Array([slot]));
    device.queue.writeBuffer(slotToBlockBuf, slot * 4, new Int32Array([blockID]));
    device.queue.writeBuffer(newlyActivatedBuf, slot * 4, new Uint32Array([1]));
    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;
    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpInitPL);
    ipl.setBindGroup(0, interpInitBG);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    device.queue.writeBuffer(newlyActivatedBuf, slot * 4, new Uint32Array([0]));
    return { slot, alreadyActive: false };
  }

  function debugDeactivateBlock(bx, by) {
    if (autoRefine) throw new Error('debugDeactivateBlock: disable autoRefine first (setAutoRefine(false))');
    if (bx < 0 || bx >= NBX || by < 0 || by >= NBY) throw new Error(`block (${bx},${by}) out of range`);
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

  // Deterministic, synchronous stepping for CDP tooling -- see
  // main-cylinder.js's identical rationale. Uses dispatchMacroStep (the
  // full refine/interp/step/average sequence), so this is measuring the
  // real AMR pipeline, not a simplified stand-in.
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
    debugListActiveBlocks,
    setAutoRefine,
    isAutoRefine: () => autoRefine,
    debugActivateBlock,
    debugDeactivateBlock,
    getBlockGridDims: () => ({ NBX, NBY, RB, MAX_FINE_BLOCKS }),
    getRefineParams: () => ({ REFINE_EVERY, REFINE_THRESH, COARSEN_THRESH }),
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
          statusEl.textContent = `[AMR] step ${st.step}  Re=${RE.toFixed(0)}  Cd=${Cd.toFixed(3)}  Cl=${Cl.toFixed(3)}`;
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
