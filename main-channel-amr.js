// AMR validation harness: plane Poiseuille/Couette channel flow, driven by
// the exact same multi-level machinery as main-amr.js/main-cylinder-amr.js
// (buffer layout, pipelines, S_Advance's recursive dispatch, per-level
// criterion/manage -- none of that is touched here, this file is generated
// by copying main-amr.js and swapping in the scenario-specific pieces
// below, same convention as main-cylinder-amr.js's own header). Unlike the
// cylinder harness, there is no body at all (HAS_BODY=0 everywhere) and no
// force/torque to integrate, so amr_force.wgsl/amr_force1.wgsl/
// amr_force1_pool.wgsl/amr_physics.wgsl are never dispatched -- matching
// main-channel.js's identical decision for the dense solver.
//
// Domain: square (W=H=2^resLog2), matching every other AMR page's shape
// convention, even though the flow only varies with y -- reusing the
// existing block-grid/pool machinery as-is (NBX/NBY, square) is far lower
// risk than teaching it to handle an asymmetric domain for the first time,
// and correctness is unaffected (more cells than strictly needed, not
// fewer). Walls at y=0/y=H-1 (shaders/common_walls.wgsl), x periodic.
//
// KNOWN GAP, CONFIRMED LIVE (not just theoretical): see
// shaders/amr_interp_dense_parent.wgsl's own header -- coarse-fine ghost
// interpolation assumes a periodic coarse level, so a block refined
// adjacent to the wall ghost-fills incorrectly (wraps instead of
// reflecting). The vorticity criterion (amr_criterion.wgsl) ALSO wraps
// periodically in y, which turned out to matter even without any real
// interior vorticity: it finite-differences across the y=0/y=H-1 "seam,"
// so an ASYMMETRIC wall-velocity profile (Couette: 0 at one wall,
// WALL_U1 at the other) reads as a large spurious jump there, independent
// of the smooth interior flow -- Poiseuille's symmetric zero-zero profile
// has no such jump and never triggers it. Live-verified: Couette N=2 with
// autoRefine left on from page load (the every-other-AMR-page default)
// reaches a stable-but-WRONG fixed point within ~600 steps, because that
// spurious trigger fires almost immediately and then hits the
// interpolation gap above. Fix: autoRefine defaults to OFF here (see its
// own declaration below), unlike every other AMR harness in this repo --
// `?autoRefine=1` opts back in for machinery-level testing (does the
// dispatch graph run without crashing for a walled domain), not physics
// validation, and is currently unsafe for Couette at any nonzero
// WALL_U1. Don't re-enable it by default without fixing the criterion's
// wall-awareness (and the interpolation gap) first.
//
// window.__CYL exposes the same shape main-channel.js's dense harness does
// (setRe, getParams, readProfile, debugRunToSteady) plus the AMR
// invariant-sweep surface tools/lib/amr-invariants.js needs (debugStepSync
// returning {step}, debugCheck21Balance, debugCheckGeometryCoverage --
// trivially {ok:true} here, no body to check coverage against --
// debugReadCardState).

import { assembleShader } from './shader-loader.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseInt(urlParams.get('res')) || 8;
// Floor of 5 (W=32, NBX=4), not the cylinder harness's 7 -- that floor came
// from a sensible-blockage-domain convention that doesn't apply here (no
// body, no blockage ratio); the only structural requirement is W a
// multiple of BLOCK=8 with more than one coarse block per axis. A lower
// floor matters for this harness specifically: channel flow's diffusive
// convergence time scales with H^2, so a cheap, frequently-run sweep needs
// small resolutions to actually be cheap.
if (resLog2 < 5) resLog2 = 5;
if (resLog2 > 11) resLog2 = 11;

let W = 1 << resLog2;
let H = W;
let NCELLS = W * H;

const MODE = urlParams.get('mode') === 'couette' ? 'couette' : 'poiseuille';

let TAU = parseFloat(urlParams.get('tau')) || 0.8;
let RE  = parseFloat(urlParams.get('re')) || 50;

function nuFromTau(tau) { return (tau - 0.5) / 3; }

// Same exact steady-state relation as main-channel.js -- see that file's
// header for the derivation (halfway-bounce-back walls, wall-to-wall
// height exactly H).
function drivingParamsFromRe(re, tau) {
  const nu = nuFromTau(tau);
  const uMax = re * nu / H;
  if (MODE === 'poiseuille') {
    return { FORCE_X: 8 * nu * uMax / (H * H), WALL_U1: 0, uMax };
  }
  return { FORCE_X: 0, WALL_U1: uMax, uMax };
}

const GHOST = 2;
const BLOCK = 8;
const RB = BLOCK;
const FB = RB * 2 + 2 * GHOST;
const NCELLS1 = FB * FB;
const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 128;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY;

const N_LEVELS = urlParams.has('levels') ? parseInt(urlParams.get('levels')) : 2;
if (N_LEVELS < 2) throw new Error(`?levels=${N_LEVELS} invalid -- must be >= 2 (L0 + at least one fine level)`);

// Vorticity-driven refinement -- same defaults as main-amr.js's falling-
// card build (calibrated for that scenario's much sharper vorticity, not
// retuned for channel flow's gentle gradient -- see this file's own header
// on why that's intentional here). FORCE_REFINE_MARGIN/LOOKAHEAD are wired
// through structurally (every manage pipeline expects them) but never
// gate anything: HAS_BODY=0 makes isNearBody/isNearBodyAt unconditionally
// false (shaders/amr_manage.wgsl/amr_manage_pool.wgsl).
const REFINE_EVERY = urlParams.has('refineEvery') ? parseInt(urlParams.get('refineEvery')) : 16;
const REFINE_THRESH = urlParams.has('refineThresh') ? parseFloat(urlParams.get('refineThresh')) : -6;
const COARSEN_THRESH = urlParams.has('coarsenThresh') ? parseFloat(urlParams.get('coarsenThresh')) : -7;
const FORCE_REFINE_MARGIN = urlParams.has('forceRefineMargin') ? parseFloat(urlParams.get('forceRefineMargin')) : 8;
const FORCE_REFINE_LOOKAHEAD = urlParams.has('forceRefineLookahead') ? parseFloat(urlParams.get('forceRefineLookahead')) : REFINE_EVERY;
// No SPONGE_EXCLUDE_W override needed (unlike the cylinder harness): there's
// no sponge at all (SPONGE_W=0 below) -- x is exactly periodic, y has real
// walls, so there's no open/far-field edge band to keep refinement out of.

function cellSizeL0AtLevel(m) { return 2 ** -m; }

// Per-child-level REFINE_THRESH/COARSEN_THRESH/FORCE_REFINE_MARGIN/
// FORCE_REFINE_LOOKAHEAD override -- same mechanism as main-cylinder-amr.js's
// identical function (childLevel=2,3,... can override the L(child-1)->L(child)
// decision via `?refineThresh{child}=` etc.), simplified since there's no
// per-level scaling-law history to carry over here (irrelevant while
// HAS_BODY=0 keeps isNearBody(At) unconditionally false).
function paramsForChildLevel(childLevel) {
  if (childLevel === 1) {
    return { REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD };
  }
  const get = (name, base) => urlParams.has(`${name}${childLevel}`) ? parseFloat(urlParams.get(`${name}${childLevel}`)) : base;
  return {
    REFINE_THRESH: get('refineThresh', REFINE_THRESH),
    COARSEN_THRESH: get('coarsenThresh', COARSEN_THRESH),
    FORCE_REFINE_MARGIN: get('forceRefineMargin', FORCE_REFINE_MARGIN),
    FORCE_REFINE_LOOKAHEAD: get('forceRefineLookahead', FORCE_REFINE_LOOKAHEAD),
  };
}

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

// Start at rest everywhere (both L0 and every pool level) -- the analytic
// profile is the STEADY-STATE solution this harness measures convergence
// toward. Uniform IC means the fine grid's t=0 state is trivially also
// uniform equilibrium, so no interpolation dispatch is needed at init
// (same reasoning as main-amr.js's own initFPool comment).
function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    for (let i = 0; i < 9; i++) f[i * NCELLS + c] = feq(1, 0, 0, i);
  }
  return f;
}
function initFPool(maxBlocks = MAX_FINE_BLOCKS) {
  const NPOOL = maxBlocks * NCELLS1;
  const f = new Float32Array(NPOOL * 9);
  for (let c = 0; c < NPOOL; c++) {
    for (let i = 0; i < 9; i++) f[i * NPOOL + c] = feq(1, 0, 0, i);
  }
  return f;
}

// Dummy CardState: HAS_BODY=0 means every step/force-adjacent shader never
// lets these fields affect the fluid, but the bind group layout still
// requires the binding. a=b=1 only avoids a divide-by-zero in get_phi/
// amr_render.wgsl's visualization shading -- irrelevant to the physics.
function initCardState() {
  const card = new Float32Array(26);
  card[12] = 1; card[13] = 1; // a, b
  card[19] = TAU;
  return card;
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

// Milestone 5 (plans/AMR-multilevel.md)-shaped level-generic pool
// allocation -- verbatim from main-amr.js's allocLevelPool (see that
// file's own extensive comment for the field-by-field rationale; nothing
// here is channel-flow-specific).
function allocLevelPool(device, U, m, NBX_m, NBY_m, maxFineBlocks) {
  const NBLOCKS_m = NBX_m * NBY_m;
  const fSizePool_m = maxFineBlocks * NCELLS1 * 9 * 4;
  const pool = {
    level: m,
    NBX: NBX_m, NBY: NBY_m, NBLOCKS: NBLOCKS_m,
    MAX_FINE_BLOCKS: maxFineBlocks,
    fSizePool: fSizePool_m,
    finePoolF_a: device.createBuffer({ size: fSizePool_m, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    finePoolF_b: device.createBuffer({ size: fSizePool_m, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    finePoolVel: device.createBuffer({ size: maxFineBlocks * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_SRC }),
    blockSlotBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    slotToBlockBuf: device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    blockCriterionBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST }),
    freeCountBuf: device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    newlyActivatedBuf: device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST }),
  };
  if (m === 1) {
    pool.freeListBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  } else {
    if (maxFineBlocks % 4 !== 0) {
      throw new Error(`level ${m}: MAX_FINE_BLOCKS (${maxFineBlocks}) must be a multiple of 4 (quad allocation)`);
    }
    pool.freeListBuf = device.createBuffer({ size: (maxFineBlocks / 4) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    const freeQuads_m = maxFineBlocks / 4;
    device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(freeQuads_m).map((_, i) => i));
    device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([freeQuads_m]));
    pool.parentSlotBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.quadrantBuf   = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.originXBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.originYBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  }
  device.queue.writeBuffer(pool.blockSlotBuf, 0, new Int32Array(NBLOCKS_m).fill(-1));
  device.queue.writeBuffer(pool.slotToBlockBuf, 0, new Int32Array(maxFineBlocks).fill(-1));
  return pool;
}

function tauAtLevel(m) {
  let t = TAU;
  for (let i = 0; i < m; i++) t = 2 * t - 0.5;
  return t;
}

// Block-major linear index for a cell at BUFFER coordinates (cx, cy) --
// matches shaders/amr_step.wgsl's cellIndex() exactly. velBuf is laid out
// this way, not flat row-major, so readProfile needs the same mapping.
function cellIndexJS(cx, cy) {
  const nbx = W / BLOCK;
  const bx = Math.floor(cx / BLOCK), by = Math.floor(cy / BLOCK);
  const lx = cx % BLOCK, ly = cy % BLOCK;
  const blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }

  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const neededBufferBytes = NCELLS * 9 * 4;
  if (neededBufferBytes > adapter.limits.maxStorageBufferBindingSize) {
    const mib = (b) => (b / 1048576).toFixed(0);
    statusEl.textContent = `error: ${W}x${H} needs a ${mib(neededBufferBytes)} MiB buffer binding, this GPU's max is ${mib(adapter.limits.maxStorageBufferBindingSize)} MiB`;
    return;
  }
  // Milestone 9 (plans/AMR-multilevel.md): amr_manage_pool.wgsl needs 16
  // storage bindings in one bind group -- see main-amr.js's identical
  // comment for the full rationale.
  const NEEDED_STORAGE_BUFFERS_PER_STAGE = 16;
  if (NEEDED_STORAGE_BUFFERS_PER_STAGE > adapter.limits.maxStorageBuffersPerShaderStage) {
    statusEl.textContent = `error: needs ${NEEDED_STORAGE_BUFFERS_PER_STAGE} storage buffers per shader stage, this GPU's max is ${adapter.limits.maxStorageBuffersPerShaderStage}`;
    return;
  }
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
    maxBufferSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
    maxStorageBuffersPerShaderStage: NEEDED_STORAGE_BUFFERS_PER_STAGE,
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
  const fSize   = NCELLS * 9 * 4;
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  // Harmless placeholders for "child level's blockSlot/blockCriterion"
  // bindings when no such level exists in this configuration -- see
  // main-amr.js's identical dummy buffers.
  const dummyBlockSlotBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyBlockSlotBuf, 0, new Int32Array([-1]));
  const dummyCriterionBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyCriterionBuf, 0, new Float32Array([0]));

  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  const pools = [undefined];
  {
    let curNBX = NBX, curNBY = NBY;
    for (let m = 1; m < N_LEVELS; m++) {
      const maxFineBlocks = m === 1
        ? MAX_FINE_BLOCKS
        : (urlParams.has(`maxFineBlocks${m}`) ? parseInt(urlParams.get(`maxFineBlocks${m}`)) : 128);
      const pool = allocLevelPool(device, U, m, curNBX, curNBY, maxFineBlocks);
      device.queue.writeBuffer(pool.finePoolF_a, 0, initFPool(maxFineBlocks));
      pools.push(pool);
      curNBX *= 2; curNBY *= 2;
    }
  }

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
  device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));

  for (let c = 2; c < N_LEVELS; c++) {
    const pool = pools[c];
    pool.levelParamsBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
    const staticBuf = new ArrayBuffer(32);
    const staticDv = new DataView(staticBuf);
    staticDv.setUint32(0, pool.NBX, true);
    staticDv.setUint32(4, pool.NBY, true);
    staticDv.setFloat32(12, cellSizeL0AtLevel(c), true);
    staticDv.setUint32(16, (c + 1) < N_LEVELS ? 1 : 0, true);
    device.queue.writeBuffer(pool.levelParamsBuf, 0, staticBuf);
  }
  function updateLevelParams() {
    for (let c = 2; c < N_LEVELS; c++) {
      device.queue.writeBuffer(pools[c].levelParamsBuf, 8, new Float32Array([tauAtLevel(c - 1)]));
    }
  }
  updateLevelParams();

  const overlaySlider = document.getElementById('slider-overlay');
  const overlayValEl = document.getElementById('val-overlay');

  const [stepSM, renSM, interpDenseSM, interpPoolSM, step1SM, step1PoolSM, avgSM, avgPoolSM, criterionSM, manageSM, criterionPoolSM, managePoolSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    loadShader(device, 'shaders/amr_interp_dense_parent.wgsl'),
    loadShader(device, 'shaders/amr_interp_pool_parent.wgsl'),
    loadShader(device, 'shaders/amr_step1.wgsl'),
    loadShader(device, 'shaders/amr_step1_pool.wgsl'),
    loadShader(device, 'shaders/amr_average_f2c.wgsl'),
    loadShader(device, 'shaders/amr_average_pool_parent.wgsl'),
    loadShader(device, 'shaders/amr_criterion.wgsl'),
    loadShader(device, 'shaders/amr_manage.wgsl'),
    loadShader(device, 'shaders/amr_criterion_pool.wgsl'),
    loadShader(device, 'shaders/amr_manage_pool.wgsl'),
  ]);

  const stepBGL = device.createBindGroupLayout({ label: 'stepBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
  ]});
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const interpPoolParentBGL = device.createBindGroupLayout({ label: 'interpPoolParentBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
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
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const criterionPoolBGL = device.createBindGroupLayout({ label: 'criterionPoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const managePoolBGL = device.createBindGroupLayout({ label: 'managePoolBGL', entries: [
    { binding: 0,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 6,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 8,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 9,  visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
  const step1PoolBGL = device.createBindGroupLayout({ label: 'step1PoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
  ]});
  const avgPoolBGL = device.createBindGroupLayout({ label: 'avgPoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});

  const constants = { W, H };
  let { FORCE_X, WALL_U1 } = drivingParamsFromRe(RE, TAU);

  // Channel-scenario overrides shared by every step-family pipeline --
  // HAS_BODY=0 (no interior geometry), WALL_Y=1 (real walls, not the
  // circular-cylinder harness's freestream sponge), SPONGE_W=0 (no sponge
  // at all -- x is exactly periodic, y has real walls). See
  // shaders/lbm_step.wgsl's identical overrides for the full rationale.
  function makeStepConstants() {
    return { W, H, HAS_BODY: 0, WALL_Y: 1, WALL_U0: 0, WALL_U1, FORCE_X, FORCE_Y: 0, SPONGE_W: 0 };
  }
  const fineConstants = { W, H, RB };

  const interpConstants = { W, H, RB, GHOST_ONLY: 1 };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0 };
  const interpFFConstants = { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1 };
  let stepConstants = makeStepConstants();
  let step1Constants = { ...stepConstants, RB };
  const criterionConstants = { W, H };
  // HAS_BODY=0: isNearBody is unconditionally false (shaders/amr_manage.wgsl).
  const manageConstants = { W, H, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD, HAS_LEVEL2: N_LEVELS > 2 ? 1 : 0, HAS_BODY: 0 };

  // Re (via FORCE_X/WALL_U1) is baked into the L0/L1(+pool) step pipelines
  // as an override -- changing it means recreating those pipelines (cheap:
  // same shader modules, no recompilation from source), matching
  // main-channel.js's identical setRe.
  let stepPL, step1PL, step1PoolPL;
  function makeStepPipelines() {
    stepConstants = makeStepConstants();
    step1Constants = { ...stepConstants, RB };
    stepPL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
      compute: { module: stepSM, entryPoint: 'main', constants: stepConstants }
    });
    step1PL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
      compute: { module: step1SM, entryPoint: 'main', constants: step1Constants }
    });
    step1PoolPL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [step1PoolBGL] }),
      compute: { module: step1PoolSM, entryPoint: 'main', constants: { ...step1Constants, K_EPS: 1.5 } }
    });
  }
  makeStepPipelines();

  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main', constants },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }], constants: fineConstants },
    primitive: { topology: 'triangle-list' },
  });
  const interpPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpDenseSM, entryPoint: 'main', constants: interpConstants }
  });
  const interpInitPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpDenseSM, entryPoint: 'main', constants: interpInitConstants }
  });
  const interpFFPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpDenseSM, entryPoint: 'main', constants: interpFFConstants }
  });
  const interpPoolConstants = { RB, GHOST_ONLY: 1 };
  const interpPoolInitConstants = { RB, GHOST_ONLY: 0 };
  const interpPoolFFConstants = { RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1 };
  const interpPoolParentPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }),
    compute: { module: interpPoolSM, entryPoint: 'main', constants: interpPoolConstants }
  });
  const interpPoolParentInitPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }),
    compute: { module: interpPoolSM, entryPoint: 'main', constants: interpPoolInitConstants }
  });
  const interpPoolParentFFPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }),
    compute: { module: interpPoolSM, entryPoint: 'main', constants: interpPoolFFConstants }
  });
  const avgPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),
    compute: { module: avgSM, entryPoint: 'main', constants: fineConstants }
  });
  const avgPoolPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgPoolBGL] }),
    compute: { module: avgPoolSM, entryPoint: 'main', constants: { RB } }
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

  const criterionPoolPLs = {};
  const managePoolCoarsenPLs = {};
  const managePoolRefinePLs = {};
  for (let m = 1; m < N_LEVELS - 1; m++) {
    const parentPool = pools[m];
    const parentIsDense = m === 1;
    const childParams = paramsForChildLevel(m + 1);
    const hasGrandchild = (m + 2) < N_LEVELS;
    const poolConstants = {
      W, H, RB,
      NBX_PARENT: parentPool.NBX, NBY_PARENT: parentPool.NBY,
      PARENT_CELL_SIZE_L0: cellSizeL0AtLevel(m),
      PARENT_HAS_CACHED_ORIGIN: parentIsDense ? 0 : 1,
      ...childParams,
      HAS_GRANDCHILD: hasGrandchild ? 1 : 0,
      HAS_BODY: 0,
    };
    criterionPoolPLs[m] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [criterionPoolBGL] }),
      compute: { module: criterionPoolSM, entryPoint: 'main', constants: { RB, NBX_PARENT: parentPool.NBX } }
    });
    managePoolCoarsenPLs[m] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [managePoolBGL] }),
      compute: { module: managePoolSM, entryPoint: 'coarsen', constants: poolConstants }
    });
    managePoolRefinePLs[m] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [managePoolBGL] }),
      compute: { module: managePoolSM, entryPoint: 'refine', constants: poolConstants }
    });
  }

  let stepBG_ab, stepBG_ba, step1BG_ab, step1BG_ba;
  function makeStepBindGroups() {
    stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
    stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});
    step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_b } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }]});
    step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }]});
  }
  makeStepBindGroups();

  const overlayOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(overlayOpacityBuf, 0, new Float32Array([1.0]));
  const outlineOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(outlineOpacityBuf, 0, new Float32Array([0.0]));
  if (overlaySlider) {
    overlaySlider.oninput = () => {
      const v = parseFloat(overlaySlider.value);
      overlayValEl.textContent = v.toFixed(2);
      device.queue.writeBuffer(overlayOpacityBuf, 0, new Float32Array([v]));
    };
  }
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }, { binding: 2, resource: { buffer: pools[1].finePoolVel } }, { binding: 3, resource: { buffer: pools[1].blockSlotBuf } }, { binding: 4, resource: { buffer: overlayOpacityBuf } }, { binding: 5, resource: { buffer: N_LEVELS > 2 ? pools[2].finePoolVel : pools[1].finePoolVel } }, { binding: 6, resource: { buffer: N_LEVELS > 2 ? pools[2].blockSlotBuf : dummyBlockSlotBuf } }, { binding: 7, resource: { buffer: outlineOpacityBuf } }]});

  const interpBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  const interpBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  const interpFFBG_b = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_b } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  const avgBG_targetA = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }]});
  const avgBG_targetB = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }]});
  const interpInitBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  const interpInitBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});

  const criterionBG = device.createBindGroup({ layout: criterionBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: pools[1].blockCriterionBuf } }]});
  const manageBG = device.createBindGroup({ layout: manageBGL, entries: [{ binding: 0, resource: { buffer: pools[1].blockCriterionBuf } }, { binding: 1, resource: { buffer: pools[1].blockSlotBuf } }, { binding: 2, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 3, resource: { buffer: pools[1].freeListBuf } }, { binding: 4, resource: { buffer: pools[1].freeCountBuf } }, { binding: 5, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 6, resource: { buffer: cardStateBuf } }, { binding: 7, resource: { buffer: N_LEVELS > 2 ? pools[2].blockCriterionBuf : dummyCriterionBuf } }, { binding: 8, resource: { buffer: N_LEVELS > 2 ? pools[2].blockSlotBuf : dummyBlockSlotBuf } }]});

  const criterionPoolBGs = {};
  const managePoolBGs = {};
  for (let m = 1; m < N_LEVELS - 1; m++) {
    const parentPool = pools[m];
    const childPool = pools[m + 1];
    const parentVel = m === 1 ? velBuf : parentPool.finePoolVel;
    const parentSlotToBlockBuf = m === 1 ? pools[1].slotToBlockBuf : parentPool.slotToBlockBuf;
    const parentBlockSlotBuf = m === 1 ? pools[1].blockSlotBuf : parentPool.blockSlotBuf;
    const parentOriginXBuf = m === 1 ? dummyBlockSlotBuf : parentPool.originXBuf;
    const parentOriginYBuf = m === 1 ? dummyBlockSlotBuf : parentPool.originYBuf;
    const grandchildPool = (m + 2) < N_LEVELS ? pools[m + 2] : null;
    const grandchildBlockSlotBuf = grandchildPool ? grandchildPool.blockSlotBuf : dummyBlockSlotBuf;

    criterionPoolBGs[m] = device.createBindGroup({ layout: criterionPoolBGL, entries: [
      { binding: 0, resource: { buffer: parentVel } },
      { binding: 1, resource: { buffer: parentSlotToBlockBuf } },
      { binding: 2, resource: { buffer: childPool.blockCriterionBuf } },
    ]});
    managePoolBGs[m] = device.createBindGroup({ layout: managePoolBGL, entries: [
      { binding: 0, resource: { buffer: childPool.blockCriterionBuf } },
      { binding: 1, resource: { buffer: childPool.blockSlotBuf } },
      { binding: 2, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 3, resource: { buffer: childPool.freeListBuf } },
      { binding: 4, resource: { buffer: childPool.freeCountBuf } },
      { binding: 5, resource: { buffer: childPool.newlyActivatedBuf } },
      { binding: 6, resource: { buffer: cardStateBuf } },
      { binding: 7, resource: { buffer: childPool.parentSlotBuf } },
      { binding: 8, resource: { buffer: childPool.quadrantBuf } },
      { binding: 9, resource: { buffer: childPool.originXBuf } },
      { binding: 10, resource: { buffer: childPool.originYBuf } },
      { binding: 11, resource: { buffer: parentBlockSlotBuf } },
      { binding: 12, resource: { buffer: parentSlotToBlockBuf } },
      { binding: 13, resource: { buffer: parentOriginXBuf } },
      { binding: 14, resource: { buffer: parentOriginYBuf } },
      { binding: 15, resource: { buffer: grandchildBlockSlotBuf } },
    ]});
  }

  for (let c = 2; c < N_LEVELS; c++) {
    const parentPool = pools[c - 1];
    const childPool = pools[c];
    const interpEntries = (parentBuf) => [
      { binding: 0, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 1, resource: { buffer: parentBuf } },
      { binding: 2, resource: { buffer: childPool.finePoolF_a } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.newlyActivatedBuf } },
      { binding: 5, resource: { buffer: childPool.blockSlotBuf } },
      { binding: 6, resource: { buffer: childPool.parentSlotBuf } },
      { binding: 7, resource: { buffer: childPool.quadrantBuf } },
    ];
    childPool.interpPoolParentBG_readA = device.createBindGroup({ layout: interpPoolParentBGL, entries: interpEntries(parentPool.finePoolF_a) });
    childPool.interpPoolParentBG_readB = device.createBindGroup({ layout: interpPoolParentBGL, entries: interpEntries(parentPool.finePoolF_b) });
    childPool.interpPoolParentFFBG_b = device.createBindGroup({ layout: interpPoolParentBGL, entries: interpEntries(parentPool.finePoolF_a).map((e, i) => i === 2 ? { binding: 2, resource: { buffer: childPool.finePoolF_b } } : e) });

    childPool.step1PoolBG_ab = device.createBindGroup({ layout: step1PoolBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: childPool.finePoolF_b } },
      { binding: 3, resource: { buffer: childPool.finePoolVel } },
      { binding: 4, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 5, resource: { buffer: childPool.originXBuf } },
      { binding: 6, resource: { buffer: childPool.originYBuf } },
      { binding: 7, resource: { buffer: childPool.levelParamsBuf } },
    ]});
    childPool.step1PoolBG_ba = device.createBindGroup({ layout: step1PoolBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_b } },
      { binding: 2, resource: { buffer: childPool.finePoolF_a } },
      { binding: 3, resource: { buffer: childPool.finePoolVel } },
      { binding: 4, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 5, resource: { buffer: childPool.originXBuf } },
      { binding: 6, resource: { buffer: childPool.originYBuf } },
      { binding: 7, resource: { buffer: childPool.levelParamsBuf } },
    ]});

    const avgEntries = (parentBuf) => [
      { binding: 0, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: parentBuf } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.parentSlotBuf } },
      { binding: 5, resource: { buffer: childPool.quadrantBuf } },
    ];
    childPool.avgPoolBG_targetA = device.createBindGroup({ layout: avgPoolBGL, entries: avgEntries(parentPool.finePoolF_a) });
    childPool.avgPoolBG_targetB = device.createBindGroup({ layout: avgPoolBGL, entries: avgEntries(parentPool.finePoolF_b) });
  }

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const WGX1 = Math.ceil(FB / 8), WGY1 = Math.ceil(FB / 8);
  const WG_MANAGE = Math.ceil(NBLOCKS / 64);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;
  // Default OFF, unlike every other AMR harness in this repo -- PROVEN
  // unsafe as a default here, not just theoretically risky. The vorticity
  // criterion (amr_criterion.wgsl) finite-differences the velocity field
  // with a PERIODIC wrap in y; with a real wall, that wrap compares the
  // near-y=0 velocity against the near-y=H-1 velocity across the "seam."
  // Poiseuille's profile is 0 at both walls, so the seam has no jump and
  // this never fires. Couette's profile is asymmetric (0 at one wall,
  // WALL_U1 at the other), so the wrap sees a spurious large jump --
  // independent of the real (tiny, uniform) interior vorticity -- and
  // triggers refinement for essentially any nonzero WALL_U1, hitting the
  // documented interpolation-periodic-wrap gap (see
  // shaders/amr_interp_dense_parent.wgsl's header) and corrupting the
  // coarse buffer. Live-verified: Couette N=2 with default thresholds
  // reaches a stable-but-wrong fixed point within ~600 steps if
  // autoRefine is left on from page load. `?autoRefine=1` opts back in
  // for machinery-level testing, not physics validation.
  let autoRefine = urlParams.get('autoRefine') === '1';
  let macroStepCounter = 0;

  // ── Recursive multi-level advance -- byte-for-byte identical dispatch
  // sequence to main-amr.js's own S_Advance (see that file for the design
  // rationale: interp-before-step commutativity, fine-fine refresh timing,
  // etc.) with only the force/torque integration removed (no body here).
  function S_Advance(level, enc) {
    const hasChild = (level + 1) < N_LEVELS;

    if (level === 0) {
      const stepBG = useB ? stepBG_ba : stepBG_ab;
      if (hasChild) {
        const readBG = useB ? interpBG_readB : interpBG_readA;
        const p = enc.beginComputePass(); p.setPipeline(interpPL); p.setBindGroup(0, readBG); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end();
      }
      const s = enc.beginComputePass(); s.setPipeline(stepPL); s.setBindGroup(0, stepBG); s.dispatchWorkgroups(WGX, WGY); s.end();
      if (hasChild) {
        S_Advance(1, enc);
        const avgBG = useB ? avgBG_targetA : avgBG_targetB;
        const a = enc.beginComputePass(); a.setPipeline(avgPL); a.setBindGroup(0, avgBG); a.dispatchWorkgroups(1, 1, MAX_FINE_BLOCKS); a.end();
      }
      return;
    }

    const pool = pools[level];
    const isL1 = level === 1;
    let cur = 'a';

    const interpIntoChild = (readCur) => {
      if (!hasChild) return;
      const childPool = pools[level + 1];
      const bg = readCur === 'a' ? childPool.interpPoolParentBG_readA : childPool.interpPoolParentBG_readB;
      const p = enc.beginComputePass(); p.setPipeline(interpPoolParentPL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, childPool.MAX_FINE_BLOCKS); p.end();
    };
    const averageFromChild = (writeCur) => {
      if (!hasChild) return;
      const childPool = pools[level + 1];
      const bg = writeCur === 'a' ? childPool.avgPoolBG_targetA : childPool.avgPoolBG_targetB;
      const p = enc.beginComputePass(); p.setPipeline(avgPoolPL); p.setBindGroup(0, bg); p.dispatchWorkgroups(1, 1, childPool.MAX_FINE_BLOCKS); p.end();
    };
    const substep = (readCur) => {
      if (isL1) {
        const bg = readCur === 'a' ? step1BG_ab : step1BG_ba;
        const p = enc.beginComputePass(); p.setPipeline(step1PL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end();
      } else {
        const bg = readCur === 'a' ? pool.step1PoolBG_ab : pool.step1PoolBG_ba;
        const p = enc.beginComputePass(); p.setPipeline(step1PoolPL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
    };
    const fineFineRefresh = () => {
      if (isL1) {
        const p = enc.beginComputePass(); p.setPipeline(interpFFPL); p.setBindGroup(0, interpFFBG_b); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end();
      } else {
        const p = enc.beginComputePass(); p.setPipeline(interpPoolParentFFPL); p.setBindGroup(0, pool.interpPoolParentFFBG_b); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
    };

    interpIntoChild(cur);
    substep(cur);
    cur = 'b';
    if (hasChild) {
      S_Advance(level + 1, enc);
      averageFromChild(cur);
      interpIntoChild(cur);
    }
    fineFineRefresh();
    substep(cur);
    cur = 'a';
    if (hasChild) {
      S_Advance(level + 1, enc);
      averageFromChild(cur);
    }
  }

  // Factored out of frame()'s loop so debugStepSync can reuse it exactly --
  // same rationale as main-amr.js's identical comment.
  function dispatchMacroStep(enc) {
    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;

    if (autoRefine && macroStepCounter % REFINE_EVERY === 0) {
      for (let m = 1; m < N_LEVELS; m++) {
        enc.clearBuffer(pools[m].newlyActivatedBuf);
      }

      const crit = enc.beginComputePass(); crit.setPipeline(criterionPL); crit.setBindGroup(0, criterionBG); crit.dispatchWorkgroups(WGX, WGY); crit.end();
      for (let m = 1; m < N_LEVELS - 1; m++) {
        const c = enc.beginComputePass(); c.setPipeline(criterionPoolPLs[m]); c.setBindGroup(0, criterionPoolBGs[m]); c.dispatchWorkgroups(2, 2, pools[m].MAX_FINE_BLOCKS); c.end();
      }

      const FIXED_POINT_ITERS = Math.max(1, N_LEVELS - 1);
      for (let iter = 0; iter < FIXED_POINT_ITERS; iter++) {
        for (let m = N_LEVELS - 1; m >= 1; m--) {
          if (m === 1) {
            const p = enc.beginComputePass(); p.setPipeline(manageCoarsenPL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
          } else {
            const parentLevel = m - 1;
            const wg = Math.ceil(pools[m].MAX_FINE_BLOCKS / 64);
            const p = enc.beginComputePass(); p.setPipeline(managePoolCoarsenPLs[parentLevel]); p.setBindGroup(0, managePoolBGs[parentLevel]); p.dispatchWorkgroups(wg); p.end();
          }
        }
        for (let m = 1; m < N_LEVELS; m++) {
          if (m === 1) {
            const p = enc.beginComputePass(); p.setPipeline(manageRefinePL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
          } else {
            const parentLevel = m - 1;
            const wg = Math.ceil(pools[parentLevel].MAX_FINE_BLOCKS / 64);
            const p = enc.beginComputePass(); p.setPipeline(managePoolRefinePLs[parentLevel]); p.setBindGroup(0, managePoolBGs[parentLevel]); p.dispatchWorkgroups(wg); p.end();
          }
        }
      }

      const init = enc.beginComputePass(); init.setPipeline(interpInitPL); init.setBindGroup(0, interpInitBG); init.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); init.end();
      for (let m = 2; m < N_LEVELS; m++) {
        const pool = pools[m];
        const p = enc.beginComputePass(); p.setPipeline(interpPoolParentInitPL); p.setBindGroup(0, pool.interpPoolParentBG_readA); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
    }
    macroStepCounter++;

    S_Advance(0, enc);

    useB = !useB;
  }

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(pools[1].finePoolF_a, 0, initFPool());
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(pools[1].blockSlotBuf, 0, new Int32Array(NBLOCKS).fill(-1));
    device.queue.writeBuffer(pools[1].slotToBlockBuf, 0, new Int32Array(MAX_FINE_BLOCKS).fill(-1));
    device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
    device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
    for (let c = 2; c < N_LEVELS; c++) {
      const pool = pools[c];
      device.queue.writeBuffer(pool.finePoolF_a, 0, initFPool(pool.MAX_FINE_BLOCKS));
      device.queue.writeBuffer(pool.blockSlotBuf, 0, new Int32Array(pool.NBLOCKS).fill(-1));
      device.queue.writeBuffer(pool.slotToBlockBuf, 0, new Int32Array(pool.MAX_FINE_BLOCKS).fill(-1));
      const freeQuads = Array.from({ length: pool.MAX_FINE_BLOCKS / 4 }, (_, i) => i);
      device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(freeQuads));
      device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([freeQuads.length]));
    }
    autoRefine = true;
    macroStepCounter = 0;
    useB = false;
    step = 0;
  }

  async function readPoolIndirection(level = 1) {
    const pool = pools[level];
    const stageBlockSlot = device.createBuffer({ size: pool.NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
    const stageSlotToBlock = device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(pool.blockSlotBuf, 0, stageBlockSlot, 0, pool.NBLOCKS * 4);
    enc.copyBufferToBuffer(pool.slotToBlockBuf, 0, stageSlotToBlock, 0, pool.MAX_FINE_BLOCKS * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([stageBlockSlot.mapAsync(GPUMapMode.READ), stageSlotToBlock.mapAsync(GPUMapMode.READ)]);
    const blockSlot = new Int32Array(stageBlockSlot.getMappedRange()).slice();
    const slotToBlock = new Int32Array(stageSlotToBlock.getMappedRange()).slice();
    stageBlockSlot.unmap(); stageSlotToBlock.unmap();
    stageBlockSlot.destroy(); stageSlotToBlock.destroy();
    return { blockSlot, slotToBlock };
  }

  async function setAutoRefine(v) { autoRefine = !!v; }

  async function debugListActiveBlocks(level = 1) {
    const pool = pools[level];
    const { blockSlot } = await readPoolIndirection(level);
    const active = [];
    for (let blockID = 0; blockID < pool.NBLOCKS; blockID++) {
      if (blockSlot[blockID] !== -1) {
        active.push({ bx: blockID % pool.NBX, by: Math.floor(blockID / pool.NBX), slot: blockSlot[blockID] });
      }
    }
    return active;
  }

  // Same border-aware 2:1-balance check as main-cylinder-amr.js's own
  // debugCheck21Balance (verbatim -- purely structural, no scenario
  // dependence at all).
  async function debugCheck21Balance() {
    const activeSets = {}, NBX_ = {}, NBY_ = {};
    for (let m = 1; m < N_LEVELS; m++) {
      const active = await debugListActiveBlocks(m);
      activeSets[m] = new Set(active.map(b => `${b.bx},${b.by}`));
      NBX_[m] = pools[m].NBX; NBY_[m] = pools[m].NBY;
    }
    function hasChild(m, bx, by) {
      return m + 1 < N_LEVELS && activeSets[m + 1].has(`${bx * 2},${by * 2}`);
    }
    function ancestorDepth(m, bx, by) {
      let level = m, x = bx, y = by;
      while (level >= 1) {
        if (activeSets[level].has(`${x},${y}`)) return level;
        x = Math.floor(x / 2); y = Math.floor(y / 2);
        level--;
      }
      return 0;
    }
    const EDGE_CHILDREN = { N: [[0, 0], [1, 0]], S: [[0, 1], [1, 1]], E: [[1, 0], [1, 1]], W: [[0, 0], [0, 1]] };
    const OPPOSITE = { N: 'S', S: 'N', E: 'W', W: 'E' };
    function borderMaxDepth(m, bx, by, edge) {
      if (!hasChild(m, bx, by)) return m;
      let maxD = m;
      for (const [dx, dy] of EDGE_CHILDREN[edge]) {
        maxD = Math.max(maxD, borderMaxDepth(m + 1, bx * 2 + dx, by * 2 + dy, edge));
      }
      return maxD;
    }
    const violations = [];
    const NEIGHBOR_OFFSETS = [['N', 0, -1], ['S', 0, 1], ['E', 1, 0], ['W', -1, 0]];
    for (let m = 1; m < N_LEVELS; m++) {
      for (const key of activeSets[m]) {
        const [bx, by] = key.split(',').map(Number);
        if (hasChild(m, bx, by)) continue;
        for (const [edge, dx, dy] of NEIGHBOR_OFFSETS) {
          const nbx = (bx + dx + NBX_[m]) % NBX_[m];
          const nby = (by + dy + NBY_[m]) % NBY_[m];
          const nDepth = activeSets[m].has(`${nbx},${nby}`)
            ? borderMaxDepth(m, nbx, nby, OPPOSITE[edge])
            : ancestorDepth(m, nbx, nby);
          if (Math.abs(m - nDepth) > 1) violations.push({ level: m, bx, by, myDepth: m, neighbor: [nbx, nby], nDepth, edge });
        }
      }
    }
    return { ok: violations.length === 0, violations };
  }

  // No body -- HAS_BODY=0 means isNearBody(At) is unconditionally false in
  // every manage shader, so there's nothing for a geometry-coverage check
  // to assert. Trivially passing, not a stand-in for a real check (see
  // this file's own header on the wall-refinement gap).
  async function debugCheckGeometryCoverage() {
    return { ok: true, violations: [] };
  }

  async function debugReadCardState() {
    const stage = device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(cardStateBuf, 0, stage, 0, 104);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const d = Array.from(new Float32Array(stage.getMappedRange()));
    stage.unmap();
    stage.destroy();
    const keys = ['cx','cy','theta','vx','vy','omega','fx','fy','tz','mass','i_body','g_eff','a','b','v_max','o_max','cx_old','cy_old','th_old','tau','y_total','x_total','off_x','off_y','off_x_old','off_y_old'];
    const out = {};
    keys.forEach((k,i) => out[k] = d[i]);
    return out;
  }

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

  function setRe(re) {
    RE = re;
    ({ FORCE_X, WALL_U1 } = drivingParamsFromRe(RE, TAU));
    makeStepPipelines();
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

  const stagingVel = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.MAP_READ | U.COPY_DST });

  // u(y): the x-velocity profile, averaged over the (x-invariant-at-
  // steady-state) streamwise direction. Reads velBuf directly -- L0's own
  // step kernel writes every cell every step regardless of refinement, and
  // the average pass (S_Advance) overwrites any refined block's L0 cells
  // with the finer level's restricted value, so velBuf always holds the
  // finest-available data even in a partially-refined domain. Block-major
  // indexing (cellIndexJS), not flat row-major -- see amr_step.wgsl.
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
      for (let x = 0; x < W; x++) sum += vel[cellIndexJS(x, y) * 2];
      uy[y] = sum / W;
    }
    return uy;
  }

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

  function getLevelPoolSizes() {
    return pools.slice(1).map(p => ({
      level: p.level,
      NBX: p.NBX, NBY: p.NBY, NBLOCKS: p.NBLOCKS,
      MAX_FINE_BLOCKS: p.MAX_FINE_BLOCKS,
    }));
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
    debugReadCardState,
    debugCheck21Balance,
    debugCheckGeometryCoverage,
    debugListActiveBlocks,
    setAutoRefine,
    isAutoRefine: () => autoRefine,
    getBlockGridDims: () => ({ NBX, NBY, RB, GHOST, FB, NCELLS1, MAX_FINE_BLOCKS }),
    getRefineParams: () => ({
      REFINE_EVERY, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD,
      perLevel: Array.from({ length: N_LEVELS - 1 }, (_, i) => ({ childLevel: i + 1, ...paramsForChildLevel(i + 1) })),
    }),
    getNumLevels: () => N_LEVELS,
    getLevelPoolSizes,
    tauAtLevel,
  };

  async function frame() {
    try {
      if (!liveMode) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      device.pushErrorScope('validation');
      const enc = device.createCommandEncoder();

      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

      device.queue.submit([enc.finish()]);
      device.popErrorScope().then(err => { if (err) handleErr(err); });

      if (performance.now() - lastT > 250) {
        statusEl.textContent = `[AMR] step ${step}  mode=${MODE}  Re=${RE.toFixed(0)}  levels=${N_LEVELS}`;
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
