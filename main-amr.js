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
// 64 was sized before geometry-forced refinement (see amr_manage.wgsl's
// isNearBody) existed, and was measured to permanently saturate: at the
// halved A=32,B=4 scale (see the A/B comment above), the card's own halo
// alone needs ~57-68 slots (down from ~100-115 pre-halving), and combined
// body+wake demand at the current, not-yet-recalibrated REFINE_THRESH/
// COARSEN_THRESH measures ~67-79 (debugListActiveBlocks() via
// ?refineThresh=999 isolates the body-only number, ?maxFineBlocks=<big>
// removes the cap so the count reflects real demand, not pool exhaustion).
// 128 gives headroom above that measured combined figure -- expected to
// matter once those thresholds are retuned for the new scale (smaller body
// means sharper vorticity gradients per unit length, so wake demand should
// go up, not down) -- while still costing less fine-pool memory (~4.0 MiB)
// than the coarse grid's own buffers (~5.24 MiB at the default W=256).
const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 128;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY; // coarse block grid

// ── Milestone 5 (plans/AMR-multilevel.md, plans/AMR-multilevel-M5.md):
// number of pool levels above L0. N_LEVELS=2 (default) is byte-identical
// to today's single-fine-level build (validated against a pre-M5
// baseline -- see the sub-plan). N_LEVELS>=3 allocates additional
// quadtree pool levels that no shader/dispatch reads yet (Milestone 6/7).
const N_LEVELS = urlParams.has('levels') ? parseInt(urlParams.get('levels')) : 2;
if (N_LEVELS < 2) throw new Error(`?levels=${N_LEVELS} invalid -- must be >= 2 (L0 + at least one fine level)`);

// ── Milestone 4b (plans/AMR.md): automatic vorticity-driven refinement ────
// Simplified AGAL Algorithm 3 for our 2-level case (see amr_criterion.wgsl/
// amr_manage.wgsl headers): a single refine threshold plus a lower coarsen
// threshold for hysteresis, both in log2|omega| units. Calibrated against
// an actual live run, not guessed: at step ~4096 (default IC, card still
// accelerating from rest) the true domain-wide max|omega| was only 0.0202
// (log2 ~= -5.63), measured directly from a debugSnapshotSave readback --
// the original guess of -5 never triggered any refinement at that stage.
// -6/-7 reliably triggers refinement tracking the wake. Still expect to
// retune as later milestones (larger domains, different A/B/tau) shift the
// sim's operating range.
const REFINE_EVERY = urlParams.has('refineEvery') ? parseInt(urlParams.get('refineEvery')) : 16;
const REFINE_THRESH = urlParams.has('refineThresh') ? parseFloat(urlParams.get('refineThresh')) : -6;
const COARSEN_THRESH = urlParams.has('coarsenThresh') ? parseFloat(urlParams.get('coarsenThresh')) : -7;

// Geometry-forced refinement (see amr_manage.wgsl's isNearBody): blocks near
// the card's SDF -- now or FORCE_REFINE_LOOKAHEAD macro-steps from now, by
// linear extrapolation of vx/vy/omega -- are always refined and never
// coarsened, independent of the vorticity criterion above. Fixes the
// "blunting" gap where a lagging vorticity signal leaves the card's own
// sharp geometry on the coarse grid (e.g. the whole startup transient,
// before any wake vorticity exists). Margin default (8 = one BLOCK) and
// lookahead default (matches REFINE_EVERY, the re-evaluation cadence it's
// meant to bridge) are starting points, not measured -- retune alongside
// REFINE_THRESH/COARSEN_THRESH once exercised against a live run.
const FORCE_REFINE_MARGIN = urlParams.has('forceRefineMargin') ? parseFloat(urlParams.get('forceRefineMargin')) : 8;
const FORCE_REFINE_LOOKAHEAD = urlParams.has('forceRefineLookahead') ? parseFloat(urlParams.get('forceRefineLookahead')) : REFINE_EVERY;

// Milestone 10: per-CHILD-level threshold overrides -- see
// main-cylinder-amr.js's copy of this function for the full rationale (a
// level-2 block's vorticity is measured on the same RB=8 stencil at half
// the physical spacing of level 1's, so the same physical feature reads as
// a different numeric |omega| one level down; reusing L0->L1's thresholds
// verbatim for L1->L2 is not expected to be correct by construction).
// `refineThresh{child}`/`coarsenThresh{child}`/etc. (child=2,3,...) override
// the L(child-1)->L(child) decision; unset levels fall back to the base
// (L0->L1) values, so a build that never sets them is unchanged.
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

// ── Pesavento & Wang (2004) physical parameters ───────────────────────────────
// These constants define the "regime" of the simulation (Falling Paper).
//
// A/B are in COARSE-grid units, and are deliberately HALF of main.js's dense-
// reference values (64,8) -- this is the AMR resource-savings fix: the card
// is defined as a fine-level body (64,8-equivalent), and only appears at
// that size where the refinement halo (see amr_manage.wgsl's isNearBody)
// actually resolves it at 2x. Everywhere else, the same W x H coarse buffer
// now spans a domain 2x wider (4x the area) in body-lengths for identical
// coarse-grid memory, vs. always running the card at dense-equivalent size.
let A = 32, B = 4;
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

// ── Milestone 6 (plans/AMR-multilevel.md): recursive fine tau. L0's own
// tau is TAU (the slider value, read live off CardState by the dense
// shader). Every deeper level's tau is the same Dupuis-Chopard relation
// amr_interp_dense_parent.wgsl already applies once (tau_fine =
// 2*tau_coarse - 0.5), just walked m times -- tauAtLevel(0) is L0's own
// tau, tauAtLevel(1) is L1's (what amr_interp_pool_parent.wgsl needs as
// `parentTau` when interpolating L1->L2), etc. Plain JS, not a GPU
// readback -- TAU is already a live JS variable the slider mutates
// directly, so this needs no round-trip.
function tauAtLevel(m) {
  let t = TAU;
  for (let i = 0; i < m; i++) t = 2 * t - 0.5;
  return t;
}

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
// Milestone 6: `maxBlocks` generalizes this beyond level 1's own capacity
// (default preserves the exact pre-M6 call sites) -- levels >=2 need the
// identical equilibrium pre-fill for the same reason level 1 already gets
// one (see the comment above initFPool's original call site): harmless
// since inactive slots are never read, and it means a slot never holds
// zero-initialized (rho=0, i.e. physically invalid) GPU memory between
// buffer creation and its first real activation.
function initFPool(maxBlocks = MAX_FINE_BLOCKS) {
  const NPOOL = maxBlocks * NCELLS1;
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

// ── Milestone 5 (plans/AMR-multilevel.md): level-generic pool allocation.
// Same buffer set as today's flat fine-pool globals, one instance per
// level, sized per plans/AMR-multilevel-M5.md's table. Level 1 is
// footprint-preserving with L0 (today's exact scheme, unchanged shapes --
// its "parent" is the dense L0 grid, addressed by blockID/cellIndex, not
// by anything this function allocates). Levels >=2 are genuine quadtree
// children of a level-(m-1) pool tile and carry two extra fields
// (parentSlot/quadrant) that level 1 has no need for. Buffers for levels
// >=2 are allocated eagerly (so ?levels=3 is a real allocation-only smoke
// test, not a no-op) but not bound into a pipeline until Milestone 6/7
// wires them up.
//
// Milestone 5's first draft also allocated ownBX/ownBY (a cached logical
// position per slot) -- Milestone 6 dropped them: a slot's own (bx,by) is
// always derivable from slotToBlock[slot] + this level's own NBX (one
// mod/div), EXACTLY what amr_interp_dense_parent.wgsl's main() already
// does every dispatch for level 1 today. Caching it would have been a
// second, redundant source of truth for zero performance benefit (the
// "expensive" derivation this would save is a single mod+div the project
// already pays for elsewhere in the same hot path) -- see
// shaders/amr_interp_pool_parent.wgsl's header for where the derivation
// actually happens.
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
    // Per-block allocation, unchanged from today -- L0 isn't itself
    // decomposed into quads, so there's no "quad" on this boundary.
    pool.freeListBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  } else {
    // Quad-unit allocation (decision 3, plans/AMR-multilevel.md:10):
    // refine/coarsen always grants or releases all 4 children of one
    // parent tile together, so the free list is indexed in quads (stride
    // 4), not individual slots.
    if (maxFineBlocks % 4 !== 0) {
      throw new Error(`level ${m}: MAX_FINE_BLOCKS (${maxFineBlocks}) must be a multiple of 4 (quad allocation)`);
    }
    pool.freeListBuf = device.createBuffer({ size: (maxFineBlocks / 4) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // New vs. level 1: a quadtree child needs its own parent lookup --
    // which parent-level slot it was carved from (parentSlot) and which
    // of the 4 quadrants it occupies (quadrant) -- see
    // plans/AMR-multilevel-M5.md §2 and shaders/amr_interp_pool_parent.wgsl.
    // COPY_SRC (not just STORAGE|COPY_DST): Milestone 10's debugSnapshotSave
    // reads these back via copyBufferToBuffer -- without it, that copy is an
    // invalid WebGPU command, which poisons the WHOLE shared command encoder
    // (all commands in an invalid GPUCommandBuffer become no-ops on submit),
    // silently zeroing out every OTHER staging buffer in the same save too.
    pool.parentSlotBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.quadrantBuf   = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // Milestone 7: a level>=2 tile's own physical (L0-buffer-space) origin,
    // cached at quad-activation time -- unlike ownBX/ownBY (correctly
    // dropped, see the amendment above), this is NOT cheaply re-derivable
    // per-dispatch: it requires walking the parent chain (this tile's
    // quadrant offset, scaled by the parent's own cell size in L0 units,
    // plus the parent's own origin, recursively), a cross-BUFFER,
    // cross-LEVEL computation, not a same-buffer mod/div. See
    // shaders/amr_step1_pool.wgsl's header.
    pool.originXBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.originYBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // parentSlot has no meaningful "unset" value read anywhere unless
    // slotToBlock already says active (initialized below) -- 0 is harmless
    // filler, not a correctness requirement, so left at WebGPU's own
    // zero-initialized default.
  }
  // BUGFIX: WebGPU zero-initializes new buffers by default -- 0 is a VALID
  // slot/blockID, not "unassigned" (that's -1, this pool's own convention
  // throughout). Every debug/reset path (resetSim, debugSnapshotLoad) was
  // careful to explicitly (re)write -1 before this milestone, but nothing
  // wrote it at bare ALLOCATION time for levels >=2 -- level 1 got it from
  // an explicit caller-side write (main-amr.js's init(), right after the
  // pools loop), but that was never generalized to every level. Exposed by
  // Milestone 8: with N_LEVELS>=3, a fresh page load (no explicit
  // AMR.reset() call) left level 2's entire pool looking "active, slot 0"
  // from frame 1 -- every slot's own force/step/average pass then ran for
  // real, all racing to write the SAME parent location (parentSlot also
  // defaulted to 0). Fixed at the source (every level, not just level 1)
  // rather than special-cased, so this can't recur if a future level's
  // caller-side init is ever forgotten again.
  device.queue.writeBuffer(pool.blockSlotBuf, 0, new Int32Array(NBLOCKS_m).fill(-1));
  device.queue.writeBuffer(pool.slotToBlockBuf, 0, new Int32Array(maxFineBlocks).fill(-1));
  return pool;
}

// Milestone 7: level m's own cell size, in L0-buffer-space units. Level 1's
// own cell is 0.5 L0 units (matches amr_step1.wgsl/amr_interp_dense_parent.
// wgsl's `fineToCoarseUnit`'s 0.5 factor); it halves again each level down.
function cellSizeL0AtLevel(m) {
  return 2 ** -m;
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }

  // Milestone 6 needs real per-level GPU timing; leave this on for the AMR
  // dev build from the start (main.js keeps it off with `0 &&` -- don't
  // touch that file, this is deliberately different here).
  const hasTimestamp = 0 && adapter.features.has('timestamp-query');

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
  // Milestone 9: shaders/amr_manage_pool.wgsl needs 15 storage bindings in
  // one bind group (childCriterion/blockSlot/slotToBlock/freeList/
  // freeCount/newlyActivated/state/parentSlot/quadrant/originX/originY,
  // plus 4 parent-level read-only mirrors for the 2:1-balance neighbor
  // checks) -- past the WebGPU spec-MINIMUM maxStorageBuffersPerShaderStage
  // of 8 (already the exact ceiling several existing 8-binding layouts sit
  // at, e.g. step1PoolBGL), same "spec minimum, not a real GPU limit"
  // situation as maxStorageBufferBindingSize above. Same treatment: request
  // what's needed, capped at the adapter's real capability, fail loud
  // (not a cryptic validation error) if this GPU genuinely can't do it.
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
  // Milestone 8: harmless placeholder for a "child level's blockSlot"
  // binding when no such level actually exists in this configuration (the
  // deepest configured level's own force pass still needs SOMETHING bound
  // there, even though its hasChild/HAS_CHILD gate means it's never read).
  // A single -1 entry is enough -- masking logic only ever indexes it when
  // hasChild is true, which is never the case for whoever binds this.
  const dummyBlockSlotBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyBlockSlotBuf, 0, new Int32Array([-1]));
  // Milestone 9: same idea, for a "child level's blockCriterion" binding
  // when HAS_LEVEL2=0 (N_LEVELS==2) -- amr_manage.wgsl's cascade/coarsen-
  // block checks are compiled out in that case, so this is never read.
  const dummyCriterionBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyCriterionBuf, 0, new Float32Array([0]));

  // CardState: 26 floats = 104 bytes
  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  // Fine-block pool (Milestone 4, generalized in Milestone 5): MAX_FINE_BLOCKS
  // slots of NCELLS1 cells each, plain flat layout within a slot (block-
  // major-of-slots overall, matching amr_step1.wgsl's `slot*(FB*FB) + local`
  // indexing). Size is independent of coarse domain size -- this is the
  // actual memory-footprint payoff (see plans/AMR.md's Milestone 4 design
  // note). `fSizePool` is kept as its own name (not just pools[1].fSizePool)
  // since it's still used standalone below by staging buffers/snapshot code
  // that, per plans/AMR-multilevel-M5.md's explicit non-goal, only ever
  // handles level 1 until Milestone 10.
  const fSizePool = MAX_FINE_BLOCKS * NCELLS1 * 9 * 4;
  const pools = [undefined]; // pools[0] unused -- L0 is the dense grid, not a pool level
  {
    let curNBX = NBX, curNBY = NBY; // level 1's logical grid = today's coarse block grid
    for (let m = 1; m < N_LEVELS; m++) {
      const maxFineBlocks = m === 1
        ? MAX_FINE_BLOCKS // unchanged param/default -- level 1 is byte-identical to today
        : (urlParams.has(`maxFineBlocks${m}`) ? parseInt(urlParams.get(`maxFineBlocks${m}`)) : 128);
      const pool = allocLevelPool(device, U, m, curNBX, curNBY, maxFineBlocks);
      device.queue.writeBuffer(pool.finePoolF_a, 0, initFPool(maxFineBlocks));
      pools.push(pool);
      curNBX *= 2; curNBY *= 2; // next level's logical grid extent (quadtree doubling per axis)
    }
  }

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());
  // pools[1].finePoolF_a's equilibrium pre-fill, and blockSlotBuf/
  // slotToBlockBuf's -1 fill, already happened above in allocLevelPool
  // (uniformly for every level, not just level 1 -- see its own comment).
  device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
  device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));

  // Milestone 6/8: per-level uniform (LevelParams) for every level>=2's
  // pool-parent interp/average/step1/force shaders. Layout: {nbx:u32,
  // nby:u32, parentTau:f32, dxL:f32, hasChild:u32, _pad1:u32, _pad2:u32,
  // _pad3:u32} = 32 bytes -- interp/average/step1_pool only declare the
  // first 4 fields (16 bytes) in their own WGSL struct, which is a valid
  // prefix of this same buffer; amr_force1_pool.wgsl (Milestone 8) is the
  // only reader of hasChild, so it declares the full 8-field struct. Only
  // levels >=2 need one -- level 1's parent is the dense L0 grid, addressed
  // via the dense shader's own CardState.tau read, not this uniform (its
  // OWN force pass, amr_force1.wgsl, gets a much smaller dedicated buffer --
  // see below).
  //
  // Split into a one-time static write (nbx/nby/dxL/hasChild -- fixed for
  // the whole session, geometric/topological, never change) and
  // updateLevelParams() below (parentTau only -- the one field that
  // actually moves, when the TAU slider changes).
  for (let c = 2; c < N_LEVELS; c++) {
    const pool = pools[c];
    pool.levelParamsBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
    const staticBuf = new ArrayBuffer(32);
    const staticDv = new DataView(staticBuf);
    staticDv.setUint32(0, pool.NBX, true);
    staticDv.setUint32(4, pool.NBY, true);
    staticDv.setFloat32(12, cellSizeL0AtLevel(c), true);
    staticDv.setUint32(16, (c + 1) < N_LEVELS ? 1 : 0, true); // does LEVEL c itself have a child?
    device.queue.writeBuffer(pool.levelParamsBuf, 0, staticBuf);
  }
  function updateLevelParams() {
    for (let c = 2; c < N_LEVELS; c++) {
      device.queue.writeBuffer(pools[c].levelParamsBuf, 8, new Float32Array([tauAtLevel(c - 1)])); // level c's parent is level c-1
    }
  }
  updateLevelParams();

  let paramsDirty = false;
  const updateGPUParams = () => {
    const data = new Float32Array([MASS, I_BODY, G_EFF, A, B]);
    device.queue.writeBuffer(cardStateBuf, 9 * 4, data);
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
    updateLevelParams(); // TAU changed -- every level's recursive tau shifts too
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

  // Refinement-coverage (green) overlay opacity. Render-only; does not affect
  // the simulation. Writing the uniform takes effect on the next frame.
  const overlaySlider = document.getElementById('slider-overlay');
  const overlayValEl = document.getElementById('val-overlay');
  if (overlaySlider) {
    overlaySlider.oninput = () => {
      const v = parseFloat(overlaySlider.value);
      overlayValEl.textContent = v.toFixed(2);
      device.queue.writeBuffer(overlayOpacityBuf, 0, new Float32Array([v]));
    };
  }

  // Quadtree outline opacity -- optional (off by default), separate from
  // the coverage fill above. White = level 1's own block edges, yellow =
  // level 2's own quadrant edges.
  const outlineSlider = document.getElementById('slider-outline');
  const outlineValEl = document.getElementById('val-outline');
  if (outlineSlider) {
    outlineSlider.oninput = () => {
      const v = parseFloat(outlineSlider.value);
      outlineValEl.textContent = v.toFixed(2);
      device.queue.writeBuffer(outlineOpacityBuf, 0, new Float32Array([v]));
    };
  }

  const [stepSM, frcSM, phySM, renSM, interpDenseSM, interpPoolSM, step1SM, step1PoolSM, avgSM, avgPoolSM, criterionSM, manageSM, force1SM, force1PoolSM, criterionPoolSM, managePoolSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_force.wgsl'),
    loadShader(device, 'shaders/amr_physics.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    loadShader(device, 'shaders/amr_interp_dense_parent.wgsl'),
    // Milestone 6: sibling shader for every L(m)->L(m+1) hop with m>=1 --
    // see shaders/amr_interp_pool_parent.wgsl's header for the addressing
    // split vs. the dense-parent module above.
    loadShader(device, 'shaders/amr_interp_pool_parent.wgsl'),
    loadShader(device, 'shaders/amr_step1.wgsl'),
    // Milestone 7: sibling shaders for every level>=2 (fine step + average),
    // same addressing split as M6's interp pair -- see
    // shaders/amr_step1_pool.wgsl / shaders/amr_average_pool_parent.wgsl.
    loadShader(device, 'shaders/amr_step1_pool.wgsl'),
    loadShader(device, 'shaders/amr_average_f2c.wgsl'),
    loadShader(device, 'shaders/amr_average_pool_parent.wgsl'),
    loadShader(device, 'shaders/amr_criterion.wgsl'),
    loadShader(device, 'shaders/amr_manage.wgsl'),
    // Milestone 8: per-level force/torque integration, same dense/pool
    // addressing split as everything else -- see amr_force1.wgsl's header.
    loadShader(device, 'shaders/amr_force1.wgsl'),
    loadShader(device, 'shaders/amr_force1_pool.wgsl'),
    // Milestone 9: per-level criterion + quad allocator/2:1-balance,
    // parent=level>=1 -- see amr_criterion_pool.wgsl/amr_manage_pool.wgsl.
    loadShader(device, 'shaders/amr_criterion_pool.wgsl'),
    loadShader(device, 'shaders/amr_manage_pool.wgsl'),
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
  const phyBGL = device.createBindGroupLayout({ label: 'phyBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  // Milestone 10: bindings 5/6 (level 2's own vel_pool/blockSlot) are for
  // finest-active-level-wins compositing -- harmless dummies when
  // N_LEVELS<3, see amr_render.wgsl's header.
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

  // Milestone 4: interp (coarse->fine ghosts), fine step, average (fine->coarse),
  // all pool-aware (an extra read-only slotToBlock/blockSlot binding vs. M2).
  // Binding 4 (newlyActivated) is Milestone 4b: only read by the GHOST_ONLY=0
  // init pipeline, but must still be present in the layout both pipelines share.
  // Milestone 4c: binding 5 (blockSlot) added so a ghost cell can check
  // whether its edge-adjacent neighbor block is also currently refined (see
  // amr_interp_c2f.wgsl's file header on fine-fine ghost consultation).
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 6: L(m)->L(m+1) (m>=1) ghost interpolation, shared by every
  // pool-to-pool level pair (decision 2 -- one pipeline, many levels, only
  // the bind group's buffers/uniform differ). Binding 0 is a small per-
  // child-level uniform (this level's own NBX/NBY + its parent's tau --
  // see shaders/amr_interp_pool_parent.wgsl's LevelParams), not the whole
  // CardState struct the dense layout uses -- a parent mid-chain doesn't
  // have a single domain-wide tau to read off CardState the way L0 does.
  // Bindings 6/7 (parentSlot/quadrant) are the only structurally new
  // per-slot fields vs. interpBGL, both this level's own.
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
  // Milestone 4b: criterion (per-block vorticity max) and manage (refine/coarsen decision).
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
    // Milestone 4c: geometry-forced refinement needs the card's pose/velocity.
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    // Milestone 9: level 2's own blockCriterion/blockSlot, for the cascade/
    // coarsen-block checks (harmless dummies when HAS_LEVEL2=0).
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 9: per-quadrant criterion for any level-(m+1) decision,
  // parent=level m -- see amr_criterion_pool.wgsl's header (one pipeline
  // per parent level, not shared, unlike the M6-M8 pool-parent shaders).
  const criterionPoolBGL = device.createBindGroupLayout({ label: 'criterionPoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  // Milestone 9: quad allocator + 2:1 balance for any level-(m+1) decision,
  // parent=level m>=1 -- see amr_manage_pool.wgsl's header.
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
    { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
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
  // Milestone 7: level>=2 fine step, shared across every level (decision 2)
  // -- bindings 5/6 (originX/originY) replace the dense case's blockID-
  // derived origin, binding 7 is the per-child-level uniform (parentTau;
  // nbx/nby unused here but shared verbatim with interpPoolParentBGL/
  // avgPoolBGL -- see shaders/amr_step1_pool.wgsl's header).
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
  // Milestone 7: level>=2 average, writing into a parent POOL tile via
  // parentSlot/quadrant instead of cellIndex() -- see
  // shaders/amr_average_pool_parent.wgsl.
  const avgPoolBGL = device.createBindGroupLayout({ label: 'avgPoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 8: level 1's own force pass. Binding 4 (childBlockSlot) is
  // level 2's blockSlot when HAS_CHILD=1, or a harmless dummy buffer when
  // HAS_CHILD=0 (N_LEVELS==2) -- see amr_force1.wgsl's header.
  const force1BGL = device.createBindGroupLayout({ label: 'force1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 8: level>=2's own force pass, one pipeline shared across every
  // such level (hasChild is a runtime LevelParams field here, not a
  // compile-time override -- see amr_force1_pool.wgsl's header).
  const force1PoolBGL = device.createBindGroupLayout({ label: 'force1PoolBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});

  const constants = { W, H };
  const fineConstants = { W, H, RB };
  // GHOST_ONLY=1: steady-state ghost-only reinterpolation (every macro-step).
  // GHOST_ONLY=0: full-slot fill, used once on block activation (see debugActivateBlock).
  const interpConstants = { W, H, RB, GHOST_ONLY: 1 };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0 };
  // Between-substep fine-fine-only ghost re-exchange (see amr_interp_c2f.wgsl's
  // FINE_FINE_ONLY note and the dispatch between f1a/f1b below).
  const interpFFConstants = { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1 };
  const step1Constants = { W, H, RB };
  const criterionConstants = { W, H };
  const manageConstants = { W, H, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD, HAS_LEVEL2: N_LEVELS > 2 ? 1 : 0 };

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
    compute: { module: interpDenseSM, entryPoint: 'main', constants: interpConstants }
  });
  // Same module/entry point as interpPL, different override constant --
  // WGSL/WebGPU compiles this as a separate pipeline. Used once per newly-
  // activated slot to fill the whole region (no prior fine-level state to
  // evolve from), vs. interpPL's steady-state ghost-only reinterpolation.
  const interpInitPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpDenseSM, entryPoint: 'main', constants: interpInitConstants }
  });
  // Fine-fine-only ghost re-exchange pipeline (same module, FINE_FINE_ONLY=1).
  const interpFFPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpDenseSM, entryPoint: 'main', constants: interpFFConstants }
  });
  // Milestone 6: pool-parent interp pipelines, mirroring the dense trio
  // above one-for-one (steady-state ghost-only / one-time full-slot-init /
  // fine-fine-only refresh) but from interpPoolSM. No W/H/NBX/NBY override
  // here -- unlike the dense case, this level's own grid extent is a
  // runtime uniform (levelParams), not baked into the pipeline, precisely
  // so ONE compiled pipeline object is reusable across every L(m)->L(m+1)
  // pair (see shaders/amr_interp_pool_parent.wgsl's header).
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
  const step1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
    compute: { module: step1SM, entryPoint: 'main', constants: step1Constants }
  });
  const avgPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),
    compute: { module: avgSM, entryPoint: 'main', constants: fineConstants }
  });
  // Milestone 7: level>=2 fine step / average -- one pipeline object each,
  // reused across every level pair (no per-level overrides needed; NBX/NBY/
  // parentTau are runtime uniform reads, not compile-time constants -- see
  // shaders/amr_step1_pool.wgsl's header, same reasoning as M6's
  // interpPoolParentPL).
  const step1PoolPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [step1PoolBGL] }),
    compute: { module: step1PoolSM, entryPoint: 'main', constants: step1Constants }
  });
  const avgPoolPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgPoolBGL] }),
    compute: { module: avgPoolSM, entryPoint: 'main', constants: { RB } }
  });
  // Milestone 8: level 1's own force pass. HAS_CHILD is baked in at
  // pipeline-creation time -- level 1 has exactly one dedicated pipeline
  // (not shared across levels), so whether level 2 exists is fixed for the
  // whole session (see amr_force1.wgsl's header).
  const force1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [force1BGL] }),
    compute: { module: force1SM, entryPoint: 'main', constants: { W, H, RB, HAS_CHILD: N_LEVELS > 2 ? 1 : 0 } }
  });
  // Milestone 8: level>=2's own force pass, one pipeline reused across
  // every such level (no per-level overrides -- hasChild/dxL are runtime
  // LevelParams reads, see amr_force1_pool.wgsl's header).
  const force1PoolPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [force1PoolBGL] }),
    compute: { module: force1PoolSM, entryPoint: 'main', constants: { W, H, RB } }
  });
  const criterionPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [criterionBGL] }),
    compute: { module: criterionSM, entryPoint: 'main', constants: criterionConstants }
  });
  // Two pipelines, same module, different entry points -- dispatched as two
  // SEPARATE passes (coarsen fully completing before refine starts) to
  // avoid a same-dispatch free-list race. See amr_manage.wgsl's header for
  // the bug this fixes (found by this milestone's own validation).
  const manageCoarsenPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'coarsen', constants: manageConstants }
  });
  const manageRefinePL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'refine', constants: manageConstants }
  });

  // Milestone 9: one criterion/manage pipeline PAIR per PARENT level
  // (1..N_LEVELS-2, i.e. every level that can itself have a child) --
  // NBX_PARENT/NBY_PARENT/PARENT_CELL_SIZE_L0/PARENT_HAS_CACHED_ORIGIN are
  // compile-time overrides, one pipeline object per parent level, not
  // shared the way M6-M8's pool-parent pipelines are (see
  // amr_manage_pool.wgsl's header for why that's the right tradeoff here).
  // Keyed by PARENT level m, deciding child level m+1.
  const criterionPoolPLs = {};
  const managePoolCoarsenPLs = {};
  const managePoolRefinePLs = {};
  for (let m = 1; m < N_LEVELS - 1; m++) {
    const parentPool = pools[m];
    const parentIsDense = m === 1; // level 1's parent is L0 -- see amr_step1.wgsl's header
    // childLevel = m+1: this loop only ever decides some child level >=2,
    // so it always picks up that child's own override (or falls back to
    // the base L0->L1 values if unset). See paramsForChildLevel's header.
    const childParams = paramsForChildLevel(m + 1);
    const poolConstants = {
      W, H, RB,
      NBX_PARENT: parentPool.NBX, NBY_PARENT: parentPool.NBY,
      PARENT_CELL_SIZE_L0: cellSizeL0AtLevel(m),
      PARENT_HAS_CACHED_ORIGIN: parentIsDense ? 0 : 1,
      ...childParams,
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

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});

  const frcBG_a = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: pools[1].blockSlotBuf } }]});
  const frcBG_b = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: pools[1].blockSlotBuf } }]});

  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const overlayOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(overlayOpacityBuf, 0, new Float32Array([1.0])); // overlay fully on by default
  // Quadtree outline opacity -- optional, off by default (see
  // shaders/amr_render.wgsl's own comment on why this is a separate
  // uniform from overlayOpacityBuf's fill).
  const outlineOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(outlineOpacityBuf, 0, new Float32Array([0.0]));
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }, { binding: 2, resource: { buffer: pools[1].finePoolVel } }, { binding: 3, resource: { buffer: pools[1].blockSlotBuf } }, { binding: 4, resource: { buffer: overlayOpacityBuf } }, { binding: 5, resource: { buffer: N_LEVELS > 2 ? pools[2].finePoolVel : pools[1].finePoolVel } }, { binding: 6, resource: { buffer: N_LEVELS > 2 ? pools[2].blockSlotBuf : dummyBlockSlotBuf } }, { binding: 7, resource: { buffer: outlineOpacityBuf } }]});

  // Milestone 4 bind groups (pool-aware, superseding M2's single-region ones).
  // interp always WRITES pools[1].finePoolF_a (the pool's current-at-macro-step-
  // boundary buffer, mirroring f_a's own invariant -- 2 fine substeps per
  // macro-step is even), but READS whichever coarse buffer is "current"
  // this macro-step (same source the force pass reads).
  const interpBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  const interpBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  // Fine ping-pong within a macro-step is a fixed 2-call sequence (ab then
  // ba), not a persistent toggle like the coarse useB -- always call both,
  // in order, every macro-step.
  const step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_b } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }]});
  const step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }]});
  // Fine-fine-only ghost re-exchange, run BETWEEN f1a and f1b. f1a writes the
  // post-substep-1 pool into pools[1].finePoolF_b (the buffer f1b then reads), so this
  // refreshes each block's fine-fine seam ghosts IN PLACE in pools[1].finePoolF_b from
  // the neighbor's just-updated interior. binding 1 (f_coarse) is unused in
  // FINE_FINE_ONLY mode; f_a is bound only to satisfy the shared layout.
  const interpFFBG_b = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_b } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  // average always READS pools[1].finePoolF_a (pool is current again after 2
  // substeps) but WRITES whichever coarse buffer the coarse step just
  // wrote this macro-step -- named by target, matching stepBG_ba being the
  // one that writes f_a.
  const avgBG_targetA = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }]});
  const avgBG_targetB = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }]});
  // Init variant (GHOST_ONLY=0, fills the whole slot): only ever called on
  // a just-activated slot immediately after coarse->fine interpolation
  // logically depends on the CURRENT coarse state, i.e. same source
  // selection as the steady-state interp bind groups above.
  const interpInitBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});
  const interpInitBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 5, resource: { buffer: pools[1].blockSlotBuf } }]});

  // Milestone 4b bind groups.
  const criterionBG = device.createBindGroup({ layout: criterionBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: pools[1].blockCriterionBuf } }]});
  const manageBG = device.createBindGroup({ layout: manageBGL, entries: [{ binding: 0, resource: { buffer: pools[1].blockCriterionBuf } }, { binding: 1, resource: { buffer: pools[1].blockSlotBuf } }, { binding: 2, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 3, resource: { buffer: pools[1].freeListBuf } }, { binding: 4, resource: { buffer: pools[1].freeCountBuf } }, { binding: 5, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 6, resource: { buffer: cardStateBuf } }, { binding: 7, resource: { buffer: N_LEVELS > 2 ? pools[2].blockCriterionBuf : dummyCriterionBuf } }, { binding: 8, resource: { buffer: N_LEVELS > 2 ? pools[2].blockSlotBuf : dummyBlockSlotBuf } }]});

  // Milestone 9: one criterion/manage bind group per PARENT level
  // (1..N_LEVELS-2), deciding child level m+1. Parent=level 1 sources from
  // the flat globals (velBuf/pools[1].*), parent=level>=2 from pools[m]
  // (both are equally valid "parent pool" shapes for this purpose -- the
  // dense-vs-cached-origin distinction is handled entirely by
  // PARENT_HAS_CACHED_ORIGIN, already baked into the pipeline above).
  const criterionPoolBGs = {};
  const managePoolBGs = {};
  for (let m = 1; m < N_LEVELS - 1; m++) {
    const parentPool = pools[m];
    const childPool = pools[m + 1];
    const parentVel = m === 1 ? velBuf : parentPool.finePoolVel;
    const parentSlotToBlockBuf = m === 1 ? pools[1].slotToBlockBuf : parentPool.slotToBlockBuf;
    const parentBlockSlotBuf = m === 1 ? pools[1].blockSlotBuf : parentPool.blockSlotBuf;
    const parentOriginXBuf = m === 1 ? dummyBlockSlotBuf : parentPool.originXBuf; // dummy: level-1 parent has no cached origin (PARENT_HAS_CACHED_ORIGIN=0 gates it out)
    const parentOriginYBuf = m === 1 ? dummyBlockSlotBuf : parentPool.originYBuf;

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
    ]});
  }

  // Bind groups for every level-pair (child level c=2..N_LEVELS-1, parent
  // c-1): interpolate (both parent-buffer variants -- Milestone 7 needs
  // BOTH now, unlike M6's manual-activation-only single readA variant,
  // since the recursive scheduler calls this mid-macro-step when either
  // of the parent's own buffers may be current), fine-fine refresh (always
  // targets the child's OWN _b, mirroring level 1's interpFFBG_b -- see
  // dispatchMacroStep's/S_Advance's fine-fine placement), this level's own
  // fine step (ab/ba, mirroring level 1's step1BG_ab/_ba exactly), and
  // average into the parent (both parent-buffer TARGET variants, mirroring
  // level 1's avgBG_targetA/_targetB).
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
    // Fine-fine-only refresh always operates on THIS level's own _b (the
    // buffer its own substep-1 just wrote) -- binding 1 (f_parent_pool) is
    // unused in FINE_FINE_ONLY mode, bound to parent's _a only to satisfy
    // the shared layout (mirrors dense's interpFFBG_b's f_a-unused note).
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

    // Milestone 8: level c's own force pass. childBlockSlot is level c+1's
    // blockSlot if it exists in this configuration, else the dummy --
    // matches this level's own levelParams.hasChild value written above.
    const childBlockSlotBuf = (c + 1 < N_LEVELS) ? pools[c + 1].blockSlotBuf : dummyBlockSlotBuf;
    childPool.force1PoolBG = device.createBindGroup({ layout: force1PoolBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: forceBuf } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.originXBuf } },
      { binding: 5, resource: { buffer: childPool.originYBuf } },
      { binding: 6, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 7, resource: { buffer: childBlockSlotBuf } },
    ]});
  }

  // Milestone 8: level 1's own force pass. Always reads pools[1].finePoolF_a
  // -- level 1's own buffer is always "current" (_a) at a macro-step
  // boundary, before S_Advance runs (see Milestone 7's own invariant), so
  // no ping-pong variant is needed here (unlike frcBG_a/frcBG_b, which DOES
  // depend on the persistent, cross-macro-step `useB` flag for L0's OWN
  // buffer choice).
  const force1BG = device.createBindGroup({ layout: force1BGL, entries: [
    { binding: 0, resource: { buffer: cardStateBuf } },
    { binding: 1, resource: { buffer: pools[1].finePoolF_a } },
    { binding: 2, resource: { buffer: forceBuf } },
    { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } },
    { binding: 4, resource: { buffer: N_LEVELS > 2 ? pools[2].blockSlotBuf : dummyBlockSlotBuf } },
  ]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  // Milestone 4: interp/fine-step dispatch over (tile, tile, pool slot) --
  // cost scales with MAX_FINE_BLOCKS, not domain size (see plans/AMR.md's
  // Milestone 4 design note). average dispatches one workgroup per slot
  // exactly (RB*RB=8*8=64 cells = 1 workgroup, see amr_average_f2c.wgsl).
  const WGX1 = Math.ceil(FB / 8), WGY1 = Math.ceil(FB / 8);
  // Milestone 4b: manage dispatches one thread per coarse block.
  const WG_MANAGE = Math.ceil(NBLOCKS / 64);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;
  let autoRefine = true; // Milestone 4b: on by default so refinement (and its coverage overlay) is visible without a console command; setAutoRefine(false) to disable for manual debugActivateBlock/debugDeactivateBlock testing
  let macroStepCounter = 0;

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
  // Milestone 10: pool array indexed by level (formatVersion 5), replacing
  // the old singular `pool` key (levels 1-only, formatVersion 4) -- see
  // debugSnapshotLoad's own explicit-reject note. Level 1's own staging
  // still uses the fixed global buffers (stagingFPool etc -- unchanged,
  // still the only thing debugProbeGhostFill/debugRunSteadyGhostFill
  // need); levels >=2 use ephemeral per-call staging buffers, same
  // approach as debugReadPool/readPoolIndirection, since N_LEVELS (and
  // each level's own sizes) are only known at runtime.
  async function debugSnapshotSave() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(f_a, 0, stagingF, 0, fSize);
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    enc.copyBufferToBuffer(cardStateBuf, 0, stagingCard, 0, 104);
    enc.copyBufferToBuffer(pools[1].finePoolF_a, 0, stagingFPool, 0, fSizePool);
    enc.copyBufferToBuffer(pools[1].finePoolVel, 0, stagingVelPool, 0, MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
    enc.copyBufferToBuffer(pools[1].blockSlotBuf, 0, stagingBlockSlot, 0, NBLOCKS * 4);
    enc.copyBufferToBuffer(pools[1].slotToBlockBuf, 0, stagingSlotToBlock, 0, MAX_FINE_BLOCKS * 4);

    const levelStaging = [];
    for (let m = 2; m < N_LEVELS; m++) {
      const pool = pools[m];
      const st = {
        f: device.createBuffer({ size: pool.fSizePool, usage: U.MAP_READ | U.COPY_DST }),
        vel: device.createBuffer({ size: pool.MAX_FINE_BLOCKS * NCELLS1 * 2 * 4, usage: U.MAP_READ | U.COPY_DST }),
        blockSlot: device.createBuffer({ size: pool.NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST }),
        slotToBlock: device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST }),
        parentSlot: device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST }),
        quadrant: device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST }),
        originX: device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST }),
        originY: device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST }),
      };
      enc.copyBufferToBuffer(pool.finePoolF_a, 0, st.f, 0, pool.fSizePool);
      enc.copyBufferToBuffer(pool.finePoolVel, 0, st.vel, 0, pool.MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
      enc.copyBufferToBuffer(pool.blockSlotBuf, 0, st.blockSlot, 0, pool.NBLOCKS * 4);
      enc.copyBufferToBuffer(pool.slotToBlockBuf, 0, st.slotToBlock, 0, pool.MAX_FINE_BLOCKS * 4);
      enc.copyBufferToBuffer(pool.parentSlotBuf, 0, st.parentSlot, 0, pool.MAX_FINE_BLOCKS * 4);
      enc.copyBufferToBuffer(pool.quadrantBuf, 0, st.quadrant, 0, pool.MAX_FINE_BLOCKS * 4);
      enc.copyBufferToBuffer(pool.originXBuf, 0, st.originX, 0, pool.MAX_FINE_BLOCKS * 4);
      enc.copyBufferToBuffer(pool.originYBuf, 0, st.originY, 0, pool.MAX_FINE_BLOCKS * 4);
      levelStaging.push(st);
    }

    device.queue.submit([enc.finish()]);
    const allBuffers = [stagingF, stagingVel, stagingCard, stagingFPool, stagingVelPool, stagingBlockSlot, stagingSlotToBlock];
    for (const st of levelStaging) allBuffers.push(st.f, st.vel, st.blockSlot, st.slotToBlock, st.parentSlot, st.quadrant, st.originX, st.originY);
    await Promise.all(allBuffers.map(b => b.mapAsync(GPUMapMode.READ)));

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

    const poolsOut = [
      null, // index 0 unused -- L0 is the dense grid, matches the live pools[] convention
      {
        level: 1, RB, GHOST, FB, MAX_FINE_BLOCKS, NBLOCKS, NBX, NBY,
        blockSlot: blockSlotArr, slotToBlock: slotToBlockArr,
        fB64: bytesToB64(new Uint8Array(fPool.buffer, fPool.byteOffset, fPool.byteLength)),
        velB64: bytesToB64(new Uint8Array(velPool.buffer, velPool.byteOffset, velPool.byteLength)),
      },
    ];
    for (let i = 0; i < levelStaging.length; i++) {
      const m = i + 2;
      const pool = pools[m];
      const st = levelStaging[i];
      const fPool_m = new Float32Array(st.f.getMappedRange()).slice();
      const velPool_m = new Float32Array(st.vel.getMappedRange()).slice();
      const blockSlotArr_m = Array.from(new Int32Array(st.blockSlot.getMappedRange()).slice());
      const slotToBlockArr_m = Array.from(new Int32Array(st.slotToBlock.getMappedRange()).slice());
      const parentSlotArr = Array.from(new Int32Array(st.parentSlot.getMappedRange()).slice());
      const quadrantArr = Array.from(new Uint32Array(st.quadrant.getMappedRange()).slice());
      const originXArr = Array.from(new Float32Array(st.originX.getMappedRange()).slice());
      const originYArr = Array.from(new Float32Array(st.originY.getMappedRange()).slice());
      for (const b of [st.f, st.vel, st.blockSlot, st.slotToBlock, st.parentSlot, st.quadrant, st.originX, st.originY]) { b.unmap(); b.destroy(); }
      poolsOut.push({
        level: m, RB, GHOST, FB, MAX_FINE_BLOCKS: pool.MAX_FINE_BLOCKS, NBLOCKS: pool.NBLOCKS, NBX: pool.NBX, NBY: pool.NBY,
        blockSlot: blockSlotArr_m, slotToBlock: slotToBlockArr_m,
        parentSlot: parentSlotArr, quadrant: quadrantArr, originX: originXArr, originY: originYArr,
        fB64: bytesToB64(new Uint8Array(fPool_m.buffer, fPool_m.byteOffset, fPool_m.byteLength)),
        velB64: bytesToB64(new Uint8Array(velPool_m.buffer, velPool_m.byteOffset, velPool_m.byteLength)),
      });
    }

    const snapshot = {
      formatVersion: 5,
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
      numLevels: N_LEVELS,
      pools: poolsOut,
    };
    console.log('[AMR snapshot] saved', { W, H, step, numLevels: N_LEVELS });
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
    // Milestone 10: formatVersion 4's singular `pool` key (level 1 only)
    // is REJECTED explicitly, not silently reinterpreted as pools[1] --
    // same "fail loud on layout mismatch" convention as the `layout`
    // check above, not a new one.
    if (snapshot.formatVersion < 5 || !Array.isArray(snapshot.pools)) {
      throw new Error(`snapshot formatVersion ${snapshot.formatVersion} uses the old singular 'pool' shape (pre-Milestone-10) -- this build expects a 'pools' array indexed by level. Re-capture the snapshot.`);
    }
    if (snapshot.numLevels !== N_LEVELS) {
      throw new Error(`snapshot has numLevels=${snapshot.numLevels}, this page has N_LEVELS=${N_LEVELS} -- reload with ?levels=${snapshot.numLevels}`);
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

    for (let m = 1; m < N_LEVELS; m++) {
      const pool = pools[m];
      const snapPool = snapshot.pools[m];
      if (!snapPool) throw new Error(`snapshot missing pools[${m}] (numLevels=${snapshot.numLevels} but this level's entry is absent)`);
      if (snapPool.RB !== RB || snapPool.MAX_FINE_BLOCKS !== pool.MAX_FINE_BLOCKS || snapPool.NBLOCKS !== pool.NBLOCKS) {
        throw new Error(`snapshot pools[${m}] (RB=${snapPool.RB},MAX_FINE_BLOCKS=${snapPool.MAX_FINE_BLOCKS},NBLOCKS=${snapPool.NBLOCKS}) doesn't match this page's (RB=${RB},MAX_FINE_BLOCKS=${pool.MAX_FINE_BLOCKS},NBLOCKS=${pool.NBLOCKS})`);
      }
      const fPool_m = b64ToFloat32(snapPool.fB64, pool.MAX_FINE_BLOCKS * NCELLS1 * 9);
      const velPool_m = b64ToFloat32(snapPool.velB64, pool.MAX_FINE_BLOCKS * NCELLS1 * 2);
      device.queue.writeBuffer(pool.finePoolF_a, 0, fPool_m.buffer, fPool_m.byteOffset, pool.fSizePool);
      device.queue.writeBuffer(pool.finePoolVel, 0, velPool_m.buffer, velPool_m.byteOffset, pool.MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
      device.queue.writeBuffer(pool.blockSlotBuf, 0, new Int32Array(snapPool.blockSlot));
      device.queue.writeBuffer(pool.slotToBlockBuf, 0, new Int32Array(snapPool.slotToBlock));

      if (m === 1) {
        // Sync the CPU-side mirrors debugActivateBlock/debugDeactivateBlock
        // rely on -- omitting this would leave them reflecting whatever was
        // active before the load, not what the loaded snapshot actually has,
        // exactly the class of GPU/CPU-state desync bug this project has
        // already been bitten by once (see debugSnapshotSave's velBuf note).
        blockSlotCPU.set(snapPool.blockSlot);
        slotToBlockCPU.set(snapPool.slotToBlock);
        freeSlots = [];
        for (let slot = 0; slot < MAX_FINE_BLOCKS; slot++) {
          if (slotToBlockCPU[slot] === -1) freeSlots.push(slot);
        }
        // Milestone 4b: the GPU-side freeList/freeCount (which the automatic
        // management pass owns) aren't part of the snapshot -- rebuild them
        // from the loaded slotToBlock instead of restoring a captured copy.
        // Free-list ORDER doesn't affect correctness (any permutation of the
        // free slots works equally as a stack), so this is exact, not an
        // approximation, and avoids growing the snapshot format for state
        // that's fully redundant with slotToBlock.
        device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(freeSlots));
        device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([freeSlots.length]));
      } else {
        device.queue.writeBuffer(pool.parentSlotBuf, 0, new Int32Array(snapPool.parentSlot));
        device.queue.writeBuffer(pool.quadrantBuf, 0, new Uint32Array(snapPool.quadrant));
        device.queue.writeBuffer(pool.originXBuf, 0, new Float32Array(snapPool.originX));
        device.queue.writeBuffer(pool.originYBuf, 0, new Float32Array(snapPool.originY));

        const qc = quadCPU[m];
        qc.blockSlotCPU.set(snapPool.blockSlot);
        qc.slotToBlockCPU.set(snapPool.slotToBlock);
        qc.originXCPU.set(snapPool.originX);
        qc.originYCPU.set(snapPool.originY);
        // Same free-list-is-redundant-with-slotToBlock reasoning as level 1
        // above, at quad granularity: quadrant 0's own slot stands for the
        // whole quad (decision 3's all-or-nothing invariant).
        qc.freeQuads = [];
        for (let quadIdx = 0; quadIdx < pool.MAX_FINE_BLOCKS / 4; quadIdx++) {
          if (qc.slotToBlockCPU[quadIdx * 4] === -1) qc.freeQuads.push(quadIdx);
        }
        device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(qc.freeQuads));
        device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([qc.freeQuads.length]));
      }
    }

    useB = false;
    step = snapshot.step;
    console.log('[AMR snapshot] loaded', { W, H, step, numLevels: N_LEVELS });
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
  // Milestone 7 (plans/AMR-multilevel.md): generic recursive dispatch,
  // replacing the old flat 7-pass sequence -- walks levels top-down in
  // AGAL's own S_Advance order (AGAL/src/solver_lbm/solver_lbm_advance.cu),
  // traced precisely rather than re-derived from the master plan's one-line
  // summary alone:
  //
  //   ROOT (level 0, no parent): interpolate INTO level 1 once (from L0's
  //   CURRENT state), L0's own ONE step, recurse into level 1 ONCE, average
  //   level 1 back into L0 once. Root never does a "second substep" -- its
  //   own dt IS the reference macro-step unit, nothing to catch up to.
  //
  //   NON-ROOT (level >= 1, always has an implicit parent -- whoever called
  //   it): interpolate INTO level+1 (if it exists) from THIS level's
  //   CURRENT state, this level's OWN substep A, then -- if level+1 exists
  //   -- recurse into level+1 ONCE, average level+1 back into THIS level,
  //   and re-interpolate INTO level+1 (using this level's just-averaged-
  //   into state) so level+1's NEXT cycle sees fresh ghosts. Then this
  //   level's own same-level fine-fine ghost refresh (a project-specific
  //   stand-in for AGAL's own neighbor-aware streaming -- see
  //   amr_interp_dense_parent.wgsl's FINE_FINE_ONLY note; AGAL's mesh
  //   doesn't need this pass because it addresses neighbor blocks directly
  //   during streaming instead of materializing ghost cells in a padded
  //   buffer). Then this level's OWN substep B, and -- again if level+1
  //   exists -- recurse into level+1 a SECOND time and average again.
  //   Every non-root level therefore does exactly 2 of its own substeps
  //   per call, and drives its child through exactly 2 full cycles (one
  //   per own substep) -- this is what makes level L+k run 2^k times more
  //   often than L0 per macro-step, the correct LBM refinement-ratio-2
  //   temporal scaling.
  //
  // "Current buffer" bookkeeping: L0 ping-pongs via the GLOBAL, persistent
  // `useB` flag (toggled once per macro-step, unchanged from before this
  // milestone). Every level >=1 instead starts EVERY call at its own _a
  // buffer and ends back at _a (substep A: a->b, substep B: b->a) -- a
  // purely LOCAL, per-call invariant needing no persistent state, and
  // exactly what today's pre-M7 code already did for level 1 alone (see
  // its own "fixed 2-call sequence, not a persistent toggle" comment,
  // preserved verbatim below). `cur` tracks it within one call.
  //
  // Level 1 is special the same way it is everywhere else in this codebase
  // (M5's addressing split, M6's shader split): its OWN substep/fine-fine
  // use the DENSE-shader pipelines (step1PL/interpFFPL, unchanged), since
  // its parent is L0. Every level's role as PARENT of level+1 (>=2) always
  // uses the POOL-shader pipelines (interpPoolParentPL/avgPoolPL), keyed by
  // the CHILD level's own bind groups -- this includes level 1 acting as
  // level 2's parent, which is why interpPoolParent* bind groups are built
  // per CHILD level (main-amr.js's per-level bind-group loop), not per
  // "is level 1" special case.
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
    let cur = 'a'; // THIS level's own current buffer, local to this call (see header)

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
    substep(cur);           // reads 'a', writes 'b'
    cur = 'b';
    if (hasChild) {
      S_Advance(level + 1, enc);
      averageFromChild(cur);  // level+1's full cycle #1 lands in level's CURRENT ('b')
      interpIntoChild(cur);   // re-interpolate level+1's ghosts from level's just-updated state
    }
    // Same-level fine-fine refresh, after any sibling's own average might
    // have just landed (see header) and before substep B reads it.
    fineFineRefresh();
    substep(cur);           // reads 'b', writes 'a'
    cur = 'a';
    if (hasChild) {
      S_Advance(level + 1, enc);
      averageFromChild(cur);  // level+1's full cycle #2 lands in level's CURRENT ('a')
    }
  }

  // Factored out of frame()'s loop so debugStepSync can reuse it exactly --
  // duplicating this dispatch sequence would risk the two silently drifting
  // apart.
  function dispatchMacroStep(enc) {
    const frcBG        = useB ? frcBG_b            : frcBG_a;
    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;

    // Milestone 4b/9: re-evaluate refinement every REFINE_EVERY macro-steps,
    // now generalized across every configured level. Runs BEFORE S_Advance
    // below so anything refined this round gets its one-time full-slot
    // fill before anything else this macro-step reads its pool slot. Reads
    // each level's own velocity field as populated by the PREVIOUS macro-
    // step (level 1's finePoolVel, level>=2's own), i.e. the same
    // "current, pre-step" data the force passes also read.
    if (autoRefine && macroStepCounter % REFINE_EVERY === 0) {
      for (let m = 1; m < N_LEVELS; m++) {
        enc.clearBuffer(pools[m].newlyActivatedBuf); // GPU-recorded, not queue.writeBuffer --
        // see plans/AMR.md's Milestone 4b note on why a JS-side writeBuffer
        // wouldn't interleave correctly with commands already recorded into
        // this same not-yet-submitted encoder.
      }

      // Criterion: level 1's own (dense, from velBuf, unchanged) plus every
      // parent level's own pool criterion (deciding levels 2..N_LEVELS-1).
      // Evaluated ONCE, before the fixed-point loop below -- a block's own
      // vorticity doesn't change just because a neighbor gets (de)activated
      // this round, so re-evaluating per iteration would be wasted work.
      const crit = enc.beginComputePass(); crit.setPipeline(criterionPL); crit.setBindGroup(0, criterionBG); crit.dispatchWorkgroups(WGX, WGY); crit.end();
      for (let m = 1; m < N_LEVELS - 1; m++) {
        const c = enc.beginComputePass(); c.setPipeline(criterionPoolPLs[m]); c.setBindGroup(0, criterionPoolBGs[m]); c.dispatchWorkgroups(2, 2, pools[m].MAX_FINE_BLOCKS); c.end();
      }

      // Milestone 9: 2:1-balance fixed-point loop -- coarsen finest-to-
      // coarsest (a level can't release while it's still a parent, or
      // while releasing would strand a neighbor's deeper child -- see
      // amr_manage.wgsl's header), then refine coarsest-to-finest (so a
      // neighbor cascade-forced active THIS iteration, at a shallower
      // level, is already reflected in blockSlot before a deeper level's
      // refine pass checks for it THIS SAME iteration). A handful of
      // iterations is enough to converge at this plan's validated depth
      // (N<=3, plans/AMR-multilevel.md's Milestone 9 text) -- see
      // amr_manage_pool.wgsl's header for why cascades don't chain deeper
      // than one hop there.
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

      // One-time full-slot fill for everything newly activated this round,
      // every level (level 1: dense-parent init pipeline, unchanged;
      // level>=2: pool-parent init pipeline, reading readA since a
      // level's own buffer is always "current" at a macro-step boundary --
      // same invariant debugActivateBlock already relies on).
      const init = enc.beginComputePass(); init.setPipeline(interpInitPL); init.setBindGroup(0, interpInitBG); init.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); init.end();
      for (let m = 2; m < N_LEVELS; m++) {
        const pool = pools[m];
        const p = enc.beginComputePass(); p.setPipeline(interpPoolParentInitPL); p.setBindGroup(0, pool.interpPoolParentBG_readA); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
    }
    macroStepCounter++;

    // Force integration + body dynamics: coarse-only still (Milestone 8's
    // scope, not this one -- see amr_step1.wgsl's own header), dispatched
    // once per macro-step, outside the fluid recursion entirely (same as
    // AGAL's own S_ComputeForces* calls, handled alongside S_Advance, not
    // inside it).
    const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
    // Milestone 8: every level's own force contribution, all before `phy`
    // drains+resets the shared atomic forces[] buffer. Order among these
    // (and vs. frc above) doesn't matter -- each reads only its own
    // level's "current, pre-macro-step" state and independently atomicAdds
    // into forces[], the same commutativity argument as interp-vs-step at
    // the root of S_Advance.
    const f1frc = enc.beginComputePass(); f1frc.setPipeline(force1PL); f1frc.setBindGroup(0, force1BG); f1frc.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1frc.end();
    for (let c = 2; c < N_LEVELS; c++) {
      const pool = pools[c];
      const p = enc.beginComputePass(); p.setPipeline(force1PoolPL); p.setBindGroup(0, pool.force1PoolBG); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
    }
    const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();

    S_Advance(0, enc);

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

  // Milestone 6: per-level (>=2) CPU mirrors for quad-granular activation
  // (decision 3, plans/AMR-multilevel.md:10) -- levels >=2 grant/release 4
  // slots as one unit, so `freeQuads` is a stack of QUAD indices (quad q
  // -> real slots q*4..q*4+3), same shape as level 1's `freeSlots` above,
  // just at 4-slot stride. Kept as a SEPARATE structure from level 1's
  // (rather than generalizing blockSlotCPU/slotToBlockCPU/freeSlots
  // themselves into per-level arrays) so level 1's already-working code
  // path above is untouched.
  const quadCPU = {};
  for (let c = 2; c < N_LEVELS; c++) {
    quadCPU[c] = {
      blockSlotCPU: new Int32Array(pools[c].NBLOCKS).fill(-1),
      slotToBlockCPU: new Int32Array(pools[c].MAX_FINE_BLOCKS).fill(-1),
      freeQuads: Array.from({ length: pools[c].MAX_FINE_BLOCKS / 4 }, (_, i) => i),
      // Milestone 7: CPU-side mirror of each active slot's own cached L0-
      // buffer-space origin (see allocLevelPool's originXBuf/originYBuf
      // comment) -- written once at activation, alongside blockSlotCPU.
      originXCPU: new Float32Array(pools[c].MAX_FINE_BLOCKS),
      originYCPU: new Float32Array(pools[c].MAX_FINE_BLOCKS),
    };
  }
  // This tile's own L0-buffer-space origin -- level 1 derives it cheaply
  // from its own (bx,by) (bx*RB, matching amr_step1.wgsl's unchanged
  // derivation exactly); level >=2 reads the cached mirror above (see
  // shaders/amr_step1_pool.wgsl's header on why level>=2 can't derive this
  // as cheaply). `bx,by` are only consulted for level===1.
  function tileOriginL0(level, slot, bx, by) {
    if (level === 1) return { x: bx * RB, y: by * RB };
    return { x: quadCPU[level].originXCPU[slot], y: quadCPU[level].originYCPU[slot] };
  }
  // This level's own blockSlotCPU mirror, whichever structure holds it --
  // level 1 uses the bare `blockSlotCPU` above, levels >=2 use quadCPU[c].
  function blockSlotCPUAtLevel(level) {
    return level === 1 ? blockSlotCPU : quadCPU[level].blockSlotCPU;
  }

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(pools[1].finePoolF_a, 0, initFPool());
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    blockSlotCPU.fill(-1);
    slotToBlockCPU.fill(-1);
    device.queue.writeBuffer(pools[1].blockSlotBuf, 0, blockSlotCPU);
    device.queue.writeBuffer(pools[1].slotToBlockBuf, 0, slotToBlockCPU);
    freeSlots = Array.from({ length: MAX_FINE_BLOCKS }, (_, i) => i);
    device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
    device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
    // Milestone 6: levels >=2 reset the same way, at quad granularity.
    for (let c = 2; c < N_LEVELS; c++) {
      const pool = pools[c];
      const qc = quadCPU[c];
      device.queue.writeBuffer(pool.finePoolF_a, 0, initFPool(pool.MAX_FINE_BLOCKS));
      qc.blockSlotCPU.fill(-1);
      qc.slotToBlockCPU.fill(-1);
      device.queue.writeBuffer(pool.blockSlotBuf, 0, qc.blockSlotCPU);
      device.queue.writeBuffer(pool.slotToBlockBuf, 0, qc.slotToBlockCPU);
      qc.freeQuads = Array.from({ length: pool.MAX_FINE_BLOCKS / 4 }, (_, i) => i);
      device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(qc.freeQuads));
      device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([qc.freeQuads.length]));
    }
    autoRefine = true; // matches the on-by-default initial state -- reset shouldn't silently disable it
    macroStepCounter = 0;
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
  // Reads blockSlot/slotToBlock directly from GPU -- the authoritative
  // source once Milestone 4b's automatic management can mutate pool state
  // without going through the CPU mirror at all.
  //
  // Milestone 6: generalized to take a level, using ephemeral staging
  // buffers sized to THAT level's own NBLOCKS/MAX_FINE_BLOCKS (levels
  // differ in both, see plans/AMR-multilevel-M5.md's table) instead of
  // the fixed-size `stagingBlockSlot`/`stagingSlotToBlock` globals (which
  // stay level-1-sized and are still used, unchanged, by
  // debugSnapshotSave's own level-1-only readback). Slightly more
  // allocation per call, but this is a debug/console function, not a hot
  // path, and it removes the old "not safe to call concurrently with
  // another in-flight readback through those buffers" caveat for free.
  async function readPoolIndirection(level = 1) {
    const pool = pools[level];
    const stageBlockSlot = device.createBuffer({ size: pool.NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
    const stageSlotToBlock = device.createBuffer({ size: pool.MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(pool.blockSlotBuf, 0, stageBlockSlot, 0, pool.NBLOCKS * 4);
    enc.copyBufferToBuffer(pool.slotToBlockBuf, 0, stageSlotToBlock, 0, pool.MAX_FINE_BLOCKS * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stageBlockSlot.mapAsync(GPUMapMode.READ),
      stageSlotToBlock.mapAsync(GPUMapMode.READ),
    ]);
    const blockSlot = new Int32Array(stageBlockSlot.getMappedRange()).slice();
    const slotToBlock = new Int32Array(stageSlotToBlock.getMappedRange()).slice();
    stageBlockSlot.unmap();
    stageSlotToBlock.unmap();
    stageBlockSlot.destroy();
    stageSlotToBlock.destroy();
    return { blockSlot, slotToBlock };
  }

  // Milestone 4b: toggles automatic vorticity-driven refinement. Manual
  // debugActivateBlock/debugDeactivateBlock are guarded against running
  // while this is on (see below) -- both mutate blockSlotCPU/slotToBlockCPU/
  // freeSlots directly, which would race the GPU-side free-list the
  // automatic management pass owns while enabled. Turning it off resyncs
  // those CPU mirrors from a fresh GPU readback, since automatic management
  // may have changed pool state the CPU mirror never saw. Level 1 only --
  // automatic management doesn't touch levels >=2 yet (Milestone 9's job),
  // so there's nothing for those levels to resync from.
  async function setAutoRefine(v) {
    autoRefine = !!v;
    if (!autoRefine) {
      const { blockSlot, slotToBlock } = await readPoolIndirection(1);
      blockSlotCPU.set(blockSlot);
      slotToBlockCPU.set(slotToBlock);
      freeSlots = [];
      for (let slot = 0; slot < MAX_FINE_BLOCKS; slot++) {
        if (slotToBlockCPU[slot] === -1) freeSlots.push(slot);
      }
    }
  }

  // Milestone 6: `level` defaults to 1 (today's exact behavior, unchanged
  // code path below). Levels >=2 activate at QUAD granularity (decision 3)
  // -- (bx,by) identifies ONE child in this level's own coordinate space,
  // but all 4 quadrant siblings are carved from a single parent quad (in
  // level (level-1)'s own pool) and activated together, since that parent
  // tile's own refine/coarsen decision was never made per-child (same
  // reasoning as amr_manage.wgsl's existing per-block criterion).
  async function debugActivateBlock(bx, by, level = 1) {
    if (autoRefine) throw new Error('debugActivateBlock: disable autoRefine first (setAutoRefine(false)) -- manual activation would race the GPU-side free-list');
    if (level < 1 || level >= N_LEVELS) throw new Error(`level ${level} out of range [1,${N_LEVELS})`);
    const pool = pools[level];
    if (bx < 0 || bx >= pool.NBX || by < 0 || by >= pool.NBY) {
      throw new Error(`level ${level} block (${bx},${by}) out of range [0,${pool.NBX})x[0,${pool.NBY})`);
    }

    if (level === 1) {
      const blockID = by * NBX + bx;
      if (blockSlotCPU[blockID] !== -1) return { slot: blockSlotCPU[blockID], alreadyActive: true };
      if (freeSlots.length === 0) throw new Error(`pool exhausted (MAX_FINE_BLOCKS=${MAX_FINE_BLOCKS})`);
      const slot = freeSlots.pop();
      blockSlotCPU[blockID] = slot;
      slotToBlockCPU[slot] = blockID;
      device.queue.writeBuffer(pools[1].blockSlotBuf, blockID * 4, new Int32Array([slot]));
      device.queue.writeBuffer(pools[1].slotToBlockBuf, slot * 4, new Int32Array([blockID]));
      // BUGFIX: the GHOST_ONLY=0 pipeline's own guard (see
      // amr_interp_dense_parent.wgsl) is
      // `if (GHOST_ONLY==0u && newlyActivated[slot]==0u) { return; }` --
      // without this write, every thread hits that guard and the dispatch
      // below silently does nothing, leaving the slot's fine pool at
      // whatever uniform-rest state initFPool() set it to. The automatic
      // refine() path in amr_manage.wgsl sets this correctly; this manual
      // CPU-driven path had never set it, meaning this debug function has
      // been silently non-functional (activating a slot without ever
      // actually initializing its fine data) since it was written. Reset
      // back to 0 after dispatch, matching the automatic path's per-round
      // clearBuffer lifecycle.
      device.queue.writeBuffer(pools[1].newlyActivatedBuf, slot * 4, new Uint32Array([1]));

      const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;
      const enc = device.createCommandEncoder();
      const ipl = enc.beginComputePass();
      ipl.setPipeline(interpInitPL);
      ipl.setBindGroup(0, interpInitBG);
      ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
      ipl.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      device.queue.writeBuffer(pools[1].newlyActivatedBuf, slot * 4, new Uint32Array([0]));
      return { slot, alreadyActive: false };
    }

    // Levels >=2: quad-granular activation against level (level-1)'s own
    // pool as parent.
    const qc = quadCPU[level];
    const blockID = by * pool.NBX + bx;
    if (qc.blockSlotCPU[blockID] !== -1) return { slot: qc.blockSlotCPU[blockID], alreadyActive: true };

    const parentPool = pools[level - 1];
    const parentBX = bx >> 1, parentBY = by >> 1;
    const parentBlockID = parentBY * parentPool.NBX + parentBX;
    const parentSlotVal = blockSlotCPUAtLevel(level - 1)[parentBlockID];
    if (parentSlotVal === -1) {
      throw new Error(`level ${level} block (${bx},${by}): parent level ${level - 1} block (${parentBX},${parentBY}) is not active -- activate it first`);
    }

    if (qc.freeQuads.length === 0) throw new Error(`level ${level} pool exhausted (MAX_FINE_BLOCKS=${pool.MAX_FINE_BLOCKS})`);
    const quadIdx = qc.freeQuads.pop();
    const baseSlot = quadIdx * 4;

    // Milestone 7: this quad's own L0-buffer-space origin, composed from
    // the PARENT's own cached (or, at level 1, cheaply-derived) origin --
    // see tileOriginL0/cellSizeL0AtLevel and shaders/amr_step1_pool.wgsl's
    // header for why this can't be re-derived per-dispatch the way ownBX/
    // ownBY could.
    const parentOrigin = tileOriginL0(level - 1, parentSlotVal, parentBX, parentBY);
    const parentCellSizeL0 = cellSizeL0AtLevel(level - 1);

    const slotsWritten = [];
    for (let qy = 0; qy <= 1; qy++) {
      for (let qx = 0; qx <= 1; qx++) {
        const quadrant = qx + 2 * qy;
        const slot = baseSlot + quadrant;
        const childBX = parentBX * 2 + qx, childBY = parentBY * 2 + qy;
        const childBlockID = childBY * pool.NBX + childBX;
        // BUGFIX (Milestone 10): see shaders/amr_manage_pool.wgsl's refine()
        // for the derivation -- a quadrant step is HALF the parent's own
        // block width, not the whole thing. Omitting *0.5 here (this file's
        // own mirror of the same formula) mis-registered every MANUALLY
        // activated level>=2 tile's physical origin the identical way the
        // GPU-side auto-refine path did.
        const originX_L0 = parentOrigin.x + qx * RB * parentCellSizeL0 * 0.5;
        const originY_L0 = parentOrigin.y + qy * RB * parentCellSizeL0 * 0.5;
        qc.blockSlotCPU[childBlockID] = slot;
        qc.slotToBlockCPU[slot] = childBlockID;
        qc.originXCPU[slot] = originX_L0;
        qc.originYCPU[slot] = originY_L0;
        device.queue.writeBuffer(pool.blockSlotBuf, childBlockID * 4, new Int32Array([slot]));
        device.queue.writeBuffer(pool.slotToBlockBuf, slot * 4, new Int32Array([childBlockID]));
        device.queue.writeBuffer(pool.parentSlotBuf, slot * 4, new Int32Array([parentSlotVal]));
        device.queue.writeBuffer(pool.quadrantBuf, slot * 4, new Uint32Array([quadrant]));
        device.queue.writeBuffer(pool.originXBuf, slot * 4, new Float32Array([originX_L0]));
        device.queue.writeBuffer(pool.originYBuf, slot * 4, new Float32Array([originY_L0]));
        device.queue.writeBuffer(pool.newlyActivatedBuf, slot * 4, new Uint32Array([1]));
        slotsWritten.push(slot);
      }
    }

    const enc = device.createCommandEncoder();
    const init = enc.beginComputePass();
    init.setPipeline(interpPoolParentInitPL);
    init.setBindGroup(0, pool.interpPoolParentBG_readA);
    init.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS);
    init.end();
    // Reconcile the quad's own 4 mutually-adjacent siblings' shared ghost
    // seams via the same-level fine-fine consultation, right after init --
    // GHOST_ONLY=0 deliberately skips that consultation (see
    // amr_interp_pool_parent.wgsl's header: two siblings could otherwise
    // race each other's still-being-written interior in the SAME
    // dispatch), so without this second pass each sibling would be left
    // holding its own independent bilinear guess at the shared boundary
    // instead of the exact neighbor-interior copy the steady-state pass
    // (and, once Milestone 7 wires it up, every live macro-step) produces.
    const steady = enc.beginComputePass();
    steady.setPipeline(interpPoolParentPL);
    steady.setBindGroup(0, pool.interpPoolParentBG_readA);
    steady.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS);
    steady.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    for (const slot of slotsWritten) {
      device.queue.writeBuffer(pool.newlyActivatedBuf, slot * 4, new Uint32Array([0]));
    }
    return { quadIdx, slots: slotsWritten, alreadyActive: false };
  }

  // Deactivates coarse block (bx,by) [level 1] or the whole quad (bx,by)
  // belongs to [level >=2]. No explicit "final average" needed: the
  // average pass already runs every macro-step while a block is active,
  // so the coarse cells already reflect the latest fine-derived state as
  // of the most recent macro-step -- deactivation just stops future
  // fine-level evolution and frees the slot(s) for reuse.
  function debugDeactivateBlock(bx, by, level = 1) {
    if (autoRefine) throw new Error('debugDeactivateBlock: disable autoRefine first (setAutoRefine(false)) -- manual deactivation would race the GPU-side free-list');
    if (level < 1 || level >= N_LEVELS) throw new Error(`level ${level} out of range [1,${N_LEVELS})`);
    const pool = pools[level];
    if (bx < 0 || bx >= pool.NBX || by < 0 || by >= pool.NBY) {
      throw new Error(`level ${level} block (${bx},${by}) out of range [0,${pool.NBX})x[0,${pool.NBY})`);
    }

    if (level === 1) {
      const blockID = by * NBX + bx;
      const slot = blockSlotCPU[blockID];
      if (slot === -1) return { wasActive: false };
      blockSlotCPU[blockID] = -1;
      slotToBlockCPU[slot] = -1;
      device.queue.writeBuffer(pools[1].blockSlotBuf, blockID * 4, new Int32Array([-1]));
      device.queue.writeBuffer(pools[1].slotToBlockBuf, slot * 4, new Int32Array([-1]));
      freeSlots.push(slot);
      return { wasActive: true, slot };
    }

    // Levels >=2: quad-granular deactivation -- releases all 4 quadrant
    // siblings of whichever quad (bx,by) belongs to, together.
    const qc = quadCPU[level];
    const blockID = by * pool.NBX + bx;
    const slot = qc.blockSlotCPU[blockID];
    if (slot === -1) return { wasActive: false };
    const quadIdx = Math.floor(slot / 4);
    const baseSlot = quadIdx * 4;
    const slotsCleared = [];
    for (let s = baseSlot; s < baseSlot + 4; s++) {
      const bID = qc.slotToBlockCPU[s];
      if (bID !== -1) {
        qc.blockSlotCPU[bID] = -1;
        device.queue.writeBuffer(pool.blockSlotBuf, bID * 4, new Int32Array([-1]));
      }
      qc.slotToBlockCPU[s] = -1;
      device.queue.writeBuffer(pool.slotToBlockBuf, s * 4, new Int32Array([-1]));
      slotsCleared.push(s);
    }
    qc.freeQuads.push(quadIdx);
    return { wasActive: true, quadIdx, slots: slotsCleared };
  }

  // TEMPORARY diagnostic (Milestone 4c investigation): writes a synthetic
  // f[0]=fx*100+fy marker into every pool cell, dispatches ONLY the
  // steady-state ghost-fill pass once (bypassing coarse step / fine step1 /
  // average entirely), and returns the resulting f[0] plane. Since the
  // marker survives untouched in every INTERIOR cell (this pass never
  // writes interior cells) and ghost cells get overwritten by whatever the
  // shader's neighbor-consultation logic picks, this directly reveals which
  // cell a ghost cell actually read from, with zero confounding from
  // streaming/collision. Remove once the fine-fine indexing bug is found.
  async function debugProbeGhostFill() {
    const NPOOL = MAX_FINE_BLOCKS * NCELLS1;
    const marker = new Float32Array(NPOOL * 9);
    for (let s = 0; s < MAX_FINE_BLOCKS; s++) {
      for (let fy = 0; fy < FB; fy++) {
        for (let fx = 0; fx < FB; fx++) {
          const cell = s * (FB * FB) + fy * FB + fx;
          marker[0 * NPOOL + cell] = fx * 100 + fy;
        }
      }
    }
    device.queue.writeBuffer(pools[1].finePoolF_a, 0, marker);

    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpPL);
    ipl.setBindGroup(0, interpBG_readA);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(pools[1].finePoolF_a, 0, stagingFPool, 0, fSizePool);
    device.queue.submit([enc2.finish()]);
    await stagingFPool.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingFPool.getMappedRange()).slice();
    stagingFPool.unmap();
    return Array.from(result.subarray(0, NPOOL));
  }

  // TEMPORARY diagnostic: dispatches ONLY the steady-state (GHOST_ONLY=1)
  // ghost-fill pass, in isolation, WITHOUT first overwriting pools[1].finePoolF_a --
  // unlike debugProbeGhostFill (which stomps the pool with a marker
  // pattern), this preserves whatever real interior data debugActivateBlock
  // already seeded, so it can be used to test the fine-fine consultation
  // path (which only runs in GHOST_ONLY=1, never in debugActivateBlock's own
  // GHOST_ONLY=0 init dispatch) against a known synthetic field's already-
  // correctly-interpolated interior, isolating exactly the mechanism the
  // Phase 4c ghost-consultation code exercises in real macro-steps.
  async function debugRunSteadyGhostFill() {
    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpPL);
    ipl.setBindGroup(0, useB ? interpBG_readB : interpBG_readA);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // Milestone 6: generic level-aware pool readback -- returns the raw
  // flat Float32Array from `level`'s own finePoolF_a, direction-major
  // across the WHOLE pool (f[i*(MAX_FINE_BLOCKS*FB*FB) + slot*(FB*FB) +
  // fy*FB + fx], matching every pool shader's own convention). Exists so
  // the M6 validation script can read back a manually-activated level-2
  // slot and compare it against the analytic Taylor-Green field, without
  // needing a full debugSnapshotSave (which, per
  // plans/AMR-multilevel-M5.md's explicit non-goal, only ever handles
  // level 1 until Milestone 10).
  async function debugReadPool(level = 1) {
    const pool = pools[level];
    const stage = device.createBuffer({ size: pool.fSizePool, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(pool.finePoolF_a, 0, stage, 0, pool.fSizePool);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(stage.getMappedRange()).slice();
    stage.unmap();
    stage.destroy();
    return Array.from(f);
  }

  // TEMPORARY diagnostic (root-cause investigation of the pre-existing
  // coarse<->fine interface artifact): overwrites f_a with a Taylor-Green-
  // like analytic vortex field (ux=-A*sin(2*pi*y/L), uy=A*sin(2*pi*x/L),
  // rho=1) instead of the usual uniform rest state. Unlike a linear ramp,
  // this has genuine curvature AND nonzero, smoothly-varying vorticity
  // (omega = A*(2*pi/L)*(cos(2*pi*x/L)+cos(2*pi*y/L))), so any error the
  // coarse->fine interpolation introduces at a block boundary shows up
  // against a known analytic ground truth, not against chaotic real flow
  // structure that's hard to reason about. Buffer-space coordinates (no
  // window conversion -- off_x/off_y are 0 right after reset() anyway).
  function debugInjectSyntheticField(A, L) {
    const f = new Float32Array(NCELLS * 9);
    for (let by = 0; by < NBY; by++) {
      for (let bx = 0; bx < NBX; bx++) {
        for (let ly = 0; ly < BLOCK; ly++) {
          for (let lx = 0; lx < BLOCK; lx++) {
            const x = bx * BLOCK + lx, y = by * BLOCK + ly;
            const blockID = by * NBX + bx;
            const cell = blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
            const ux = -A * Math.sin(2 * Math.PI * y / L);
            const uy = A * Math.sin(2 * Math.PI * x / L);
            for (let i = 0; i < 9; i++) f[i * NCELLS + cell] = feq(1, ux, uy, i);
          }
        }
      }
    }
    device.queue.writeBuffer(f_a, 0, f);
  }

  // Always reads GPU state directly (not the CPU mirror, which goes stale
  // the instant autoRefine's automatic management mutates pool state
  // without the CPU ever seeing it) -- see readPoolIndirection.
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

  // Milestone 9's own validation ask: walk all active tiles, confirm no
  // same-level-neighbor pair differs by more than 1 level -- cheap enough
  // to call periodically during development/validation, not wired into
  // the live per-macro-step path (that would need a GPU-side assertion
  // mechanism this project doesn't have; a readback-based debug function
  // is enough to catch a real violation during testing).
  async function debugCheck21Balance() {
    const level1 = await debugListActiveBlocks(1);
    const level1Set = new Set(level1.map(b => `${b.bx},${b.by}`));
    const level2Set = new Set();
    let level2Count = 0;
    if (N_LEVELS > 2) {
      const level2 = await debugListActiveBlocks(2);
      level2Count = level2.length;
      for (const b of level2) level2Set.add(`${Math.floor(b.bx / 2)},${Math.floor(b.by / 2)}`);
    }
    const { NBX, NBY } = pools[1];
    function levelOf(bx, by) {
      const key = `${bx},${by}`;
      if (!level1Set.has(key)) return 0;
      return level2Set.has(key) ? 2 : 1;
    }
    const violations = [];
    for (const b of level1) {
      const l = levelOf(b.bx, b.by);
      const neighbors = [
        [b.bx, (b.by + NBY - 1) % NBY], [b.bx, (b.by + 1) % NBY],
        [(b.bx + 1) % NBX, b.by], [(b.bx + NBX - 1) % NBX, b.by],
      ];
      for (const [nbx, nby] of neighbors) {
        const nl = levelOf(nbx, nby);
        if (Math.abs(l - nl) > 1) violations.push({ bx: b.bx, by: b.by, level: l, neighbor: [nbx, nby], neighborLevel: nl });
      }
    }
    return { ok: violations.length === 0, violations, level1Count: level1.length, level2Count };
  }

  // Deterministic synchronous stepping, bypassing rAF entirely -- lets two
  // separate builds be driven to an EXACT matching step count for a fair
  // diff. Wall-clock polling of the normal rAF-driven `liveMode` loop can't
  // guarantee this: STEPS_PER_FRAME-sized jumps land unpredictably relative
  // to any external poll interval (confirmed directly while re-validating
  // Milestones 1 and 2 at 256x256 -- see plans/AMR.md).
  // TEMPORARY diagnostic: single-macro-step granularity (debugStepSync is
  // locked to STEPS_PER_FRAME=64-step batches), for bisecting exactly which
  // macro-step a divergence first appears on.
  async function debugStepOne() {
    liveMode = false;
    const enc = device.createCommandEncoder();
    dispatchMacroStep(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    step += 1;
    return { step };
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

  // Milestone 5 (plans/AMR-multilevel-M5.md §6): pure-JS introspection of
  // per-level buffer sizes, for the "?levels=2 must allocate byte-
  // identical buffer sizes/counts to today's build" validation check --
  // no GPU readback needed, just GPUBuffer.size on what allocLevelPool
  // created.
  function getLevelPoolSizes() {
    return pools.slice(1).map(p => ({
      level: p.level,
      NBX: p.NBX, NBY: p.NBY, NBLOCKS: p.NBLOCKS,
      MAX_FINE_BLOCKS: p.MAX_FINE_BLOCKS,
      bytes: {
        finePoolF_a: p.finePoolF_a.size,
        finePoolF_b: p.finePoolF_b.size,
        finePoolVel: p.finePoolVel.size,
        blockSlotBuf: p.blockSlotBuf.size,
        slotToBlockBuf: p.slotToBlockBuf.size,
        blockCriterionBuf: p.blockCriterionBuf.size,
        freeListBuf: p.freeListBuf.size,
        freeCountBuf: p.freeCountBuf.size,
        newlyActivatedBuf: p.newlyActivatedBuf.size,
        ...(p.parentSlotBuf ? {
          parentSlotBuf: p.parentSlotBuf.size,
          quadrantBuf: p.quadrantBuf.size,
        } : {}),
      },
    }));
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
    debugStepOne,
    debugActivateBlock,
    debugDeactivateBlock,
    debugListActiveBlocks,
    debugCheck21Balance,
    debugProbeGhostFill,
    debugRunSteadyGhostFill,
    debugReadPool,
    debugInjectSyntheticField,
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
        // enc.writeTimestamp(querySet, 0);
      }

      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;

      if (hasTimestamp) {
        // enc.writeTimestamp(querySet, 1);
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
