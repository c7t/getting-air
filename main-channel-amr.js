// AMR validation harness: plane Poiseuille/Couette channel flow, driven by
// the exact same multi-level machinery as main-amr.js/main-cylinder-amr.js
// (buffer layout, pipelines, S_Advance's recursive dispatch, per-level
// criterion/manage -- none of that is touched here, this file is generated
// by copying main-amr.js and swapping in the scenario-specific pieces
// below, same convention as main-cylinder-amr.js's own header). Unlike the
// cylinder harness, there is no body at all (HAS_BODY=0 everywhere) and no
// force/torque to integrate, so amr_force.wgsl/amr_force1.wgsl/
// amr_physics.wgsl are never dispatched -- matching
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
// returning {step}, debugCheck21Balance, debugReadCardState). NOT
// debugCheckGeometryCoverage: this page has no body (HAS_BODY = 0), so
// tools/lib/amr-invariants.js's probe reports it SKIPPED, which is what that
// file's "a missing optional check is reported as skipped, never as a pass"
// design intends. It used to be defined here as a stub returning {ok:true} --
// a green tick standing for nothing, which is exactly what the probe exists
// to avoid.

import { reportFatal, refuseConfig, reportNoWebGPU, reportNoAdapter } from './error-overlay.mjs';
import { tauChainSingularity, tauSingularityMessage } from './amr2d.mjs';
import { loadShader } from './shader-loader.mjs';
import { packF, unpackF, fWords } from './f-pack.mjs';
import { check21BalanceOnGPU, allocLevelPool, readPoolIndirection as readPoolIndirectionOn, listActiveBlocks, readCardState , checkRefinementClosureOnGPU , makeCascadePipelines, encodeCascade, cascadeRoundTrip, makeCascadeSeeds, checkSlotQuadrantsOnGPU } from './amr2d-gpu.mjs';
// tauAtLevel: extracted to card-params.mjs by B3a-1, which landed the CALL
// in all five AMR pages and this IMPORT in only main-amr.js. The other four
// threw `ReferenceError: tauAtLevelOf is not defined` at init -- but only at
// ?levels>=3, because updateLevelParams's `for (c = 2; c < N_LEVELS; c++)`
// loop is VACUOUS at the levels=2 default every one of them ships. See
// plans/2D-backport.md B4.
import { tauAtLevel as tauAtLevelOf } from './card-params.mjs';
import { EX, EY, WT } from './lattice-2d.mjs';
import { makeCanvasFit } from './canvas-fit.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
// ?f16=1 / ?f16=2: real packed-half storage for `f` -- see shaders/common_fpack.wgsl
// and f-pack.mjs. Wired on EVERY page that consumes those shaders, including
// the ones with no accuracy check of their own: a page that quietly ignored
// ?f16= would make a green `validate-all --extra=f16=1` sweep look like it
// covered ground it never touched, which is the kind of false confidence
// this repo has been bitten by before.
const F16 = urlParams.has('f16') ? (parseInt(urlParams.get('f16')) || 0) : 0;

// ── ?dcpre=1 -- the legacy PRE-collision Dupuis-Chopard transfer factor ──────
// The coarse<->fine transfers rescale the non-equilibrium part of f, and the
// factor that shipped here until plans/2D-backport.md B1 was the textbook
// PRE-collision one -- while every buffer this solver transfers holds f AFTER
// collision, because amr_step.wgsl is a fused pull-stream + collide. At this
// page's default tau = 0.8 the two differ in magnitude AND SIGN (+0.6875
// against -0.25), which is why the analytic Poiseuille gate here -- not a
// Cd/St average -- is the instrument that can see it. amr2d.mjs's
// dcRescaleCoarseToFine carries the derivation and tools/test-amr2d.js gates
// both forms.
//
// Both factors live in one build so the defect can be RE-MEASURED rather than
// reconstructed from a checkout -- the same reason ?ghostcopy= and ?f16= are
// still here. It also has no tau = 1 singularity, so it is the escape hatch
// the refusal names.
const DC_PRE = urlParams.has('dcpre') ? (parseInt(urlParams.get('dcpre')) || 0) : 1;

// ── ?ghostcopy=1 -- legacy materialized same-level ghost cells ───────────────
// Default 0: the fine step resolves a source cell that falls outside its tile
// against the owning same-level tile directly (blockSlot), so the
// between-substep fine-fine ghost COPY pass is not encoded at all. Measured
// worth 9.9% of frame GPU time on the desktop and 13.1% on the phone (the pass
// itself was 16.7-17.5%; the fine step gives 3.6-7.5% of that back, since half
// a tile's threads now pull from another slot's memory) -- see
// plans/perf-characterization.md's "The one lead left" for the full
// decomposition on both devices.
//
// 1 restores the old path exactly (clamp at the slot's own buffer edge, plus
// the copy pass), so the two can be A/B'd for both speed and physics on one
// build: `node tools/validate-all.js --extra=ghostcopy=1` runs the whole
// validation sweep against the legacy path.
const GHOST_COPY = urlParams.has('ghostcopy') ? (parseInt(urlParams.get('ghostcopy')) || 0) : 0;

// ── ?diag=1 -- refinement convergence counters ──────────────────────────────
// Makes the refine round's fixed-point loop report whether it actually
// SETTLED, instead of just running out of iterations. Off by default and
// zero-cost when off: the DIAG override gates every atomic.
const DIAG = urlParams.has('diag') ? (parseInt(urlParams.get('diag')) || 0) : 0;

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
// ── ?demandCascade=1 -- the growth half of Milestone 9's refine cascade ────
// plans/AMR-multilevel.md specifies: "a quad may only refine to level m+1 if
// its same-level neighbors are already present; if a neighbor is more than one
// level coarser, force THAT neighbor to refine first (recursively, if the gap
// is >1)". Only the veto half was ever implemented, so a criterion-driven
// refine can be vetoed forever by a neighbour that would only ever have been
// created BY that refine. Measured live on index-amr.html: level 2 never
// extends past the geometry halo into the wake, which pins the L1/L2 boundary
// a few cells off the body so every shed vortex crosses it right there.
//
// The recursion the plan asks for is a poor GPU fit, but the depth is known up
// front and the fixed-point loop below is the bounded equivalent: growth
// advances one level per iteration and the loop already runs N_LEVELS-1 times,
// which is the exact bound. See shaders/amr_manage.wgsl's level2Wanted for the
// mechanism, why it is a union with (not a replacement for) the existence-based
// cascade that superseded it, and why it stops at the L0->L1 hop.
//

// The 2:1 rule is ONE closure on the want set (plans/2D-backport.md B2):
// decide -> shaders/amr_cascade.wgsl -> coarsen/refine, once. ?cascade=0 kept
// the old per-pass path alongside it for one commit so the difference could be
// measured (B2-2c: corner 2:1 violations 13 -> 0, level 2's x-extent 80 -> 160
// L0 units on this page); B2-2d deleted it.

const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 128;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY;

const N_LEVELS = urlParams.has('levels') ? parseInt(urlParams.get('levels')) : 2;
// refuseConfig, not throw: this runs at module scope, where
// init().catch(handleErr) can never see it -- see error-overlay.mjs.
if (N_LEVELS < 2) refuseConfig(statusEl, `?levels=${N_LEVELS} invalid -- must be >= 2 (L0 + at least one fine level)`);

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

// The D2Q9 basis, from the ONE place it is derived -- lattice-2d.mjs, which
// also generates shaders/common_lattice.wgsl. Typed out here (and in nine
// sibling pages) until 2026-09-14, in f64 EXACT FRACTIONS while the shader
// held eight-digit f32 decimals: the host built its initial condition from
// weights the GPU did not have. WT is now the shader's own f32 values.
// EX/EY were already identical everywhere and are unchanged.
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

function handleErr(e) {
  // Status line AND a legible on-page overlay -- see error-overlay.mjs for why
  // the 12px status line alone was not enough.
  reportFatal(statusEl, e);
}

// card-params.mjs owns this rule and tools/test-card-params.js tests it.
// Five pages inlined the same loop -- see plans/2D-backport.md B3a.
const tauAtLevel = (m) => tauAtLevelOf(TAU, m);
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
  if (!navigator.gpu) { reportNoWebGPU(statusEl); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { reportNoAdapter(statusEl); return; }

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

  // Canvas sizing and swapchain reconfiguration -- see canvas-fit.mjs,
  // which carries the reasoning for the changed-size guard (it was ten
  // identical copies of it).
  const resize = makeCanvasFit({ canvas, ctx, device, format: fmt });
  window.addEventListener('resize', resize);
  resize();

  const U = GPUBufferUsage;
  const fSize   = NCELLS * 9 * 4;
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  // COPY_DST: debugSnapshotLoad restores this with queue.writeBuffer.
  // Without the flag that write is a validation error and is silently
  // dropped, so a loaded snapshot keeps whatever ux/uy were already there.
  // The load path's own comment already describes this exact symptom
  // ("rho round-tripped exactly, but ux/uy didn't -- the asymmetry was the
  // tell") -- the writeBuffer call was added then, but the usage flag was
  // not, so the fix never actually took effect.
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  // Harmless placeholders for "child level's blockSlot/blockCriterion"
  // bindings when no such level exists in this configuration -- see
  // main-amr.js's identical dummy buffers.
  // ?diag=1 counters -- 8 u32 slots, read+zeroed via debugReadDiag().
  const diagBuf = device.createBuffer({ size: 8 * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  const diagReadBuf = device.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const dummyBlockSlotBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyBlockSlotBuf, 0, new Int32Array([-1]));

  // See main-amr.js's copy for the rationale: the GPU buffer holds packed
  // half pairs under F16, everything else speaks f32 plane-major, and these
  // two are the only places the two meet.
  const writeF = (buf, f32, ncells) => {
    const src = packF(f32, ncells, F16);
    device.queue.writeBuffer(buf, 0, src.buffer, src.byteOffset, ncells * fWords(F16) * 4);
  };
  const readF = (mapped, ncells) =>
    F16 ? unpackF(new Uint32Array(mapped), ncells, true) : new Float32Array(mapped).slice();

  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  const pools = [undefined];
  {
    let curNBX = NBX, curNBY = NBY;
    for (let m = 1; m < N_LEVELS; m++) {
      const maxFineBlocks = m === 1
        ? MAX_FINE_BLOCKS
        : (urlParams.has(`maxFineBlocks${m}`) ? parseInt(urlParams.get(`maxFineBlocks${m}`)) : 128);
      const pool = allocLevelPool(device, U, m, curNBX, curNBY, maxFineBlocks, NCELLS1);
      writeF(pool.finePoolF_a, initFPool(maxFineBlocks), maxFineBlocks * NCELLS1);
      pools.push(pool);
      curNBX *= 2; curNBY *= 2;
    }
  }

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  writeF(f_a, initF(), NCELLS);
  device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
  device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));

  for (let c = 1; c < N_LEVELS; c++) {
    const pool = pools[c];
    pool.levelParamsBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
    const staticBuf = new ArrayBuffer(32);
    const staticDv = new DataView(staticBuf);
    staticDv.setUint32(0, pool.NBX, true);
    staticDv.setUint32(4, pool.NBY, true);
    staticDv.setFloat32(12, cellSizeL0AtLevel(c), true);
    // The diffuse band, in units of THIS level's dx -- a per-level uniform
    // since B3-1's follow-up, not the compile-time K_EPS override the fine
    // step used to carry. One pipeline now serves every level, so an override
    // could only ever say one thing for the whole hierarchy. See
    // shaders/amr_step1.wgsl's get_chi.
    staticDv.setFloat32(20, 1.5, true);
    staticDv.setUint32(16, (c + 1) < N_LEVELS ? 1 : 0, true);
    device.queue.writeBuffer(pool.levelParamsBuf, 0, staticBuf);
  }
  function updateLevelParams() {
    // tau = 1 IS A REAL SINGULARITY for the post-collision transfer -- refuse
    // it, do not divide by it. See amr2d.mjs's tauChainSingularity. Reachable
    // here by hand: ?tau=0.75 puts level 1 exactly on 1, and ?tau=1 level 0.
    const sing = DC_PRE === 0 ? tauChainSingularity(TAU, N_LEVELS) : null;
    if (sing) refuseConfig(statusEl, tauSingularityMessage(sing));
    for (let c = 1; c < N_LEVELS; c++) {
      device.queue.writeBuffer(pools[c].levelParamsBuf, 8, new Float32Array([tauAtLevel(c - 1)]));
    }
  }
  updateLevelParams();

  const overlaySlider = document.getElementById('slider-overlay');
  const overlayValEl = document.getElementById('val-overlay');

  const [stepSM, renSM, interpDenseSM, interpPoolSM, step1SM, avgSM, avgPoolSM, criterionSM, manageSM, criterionPoolSM, managePoolSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    loadShader(device, 'shaders/amr_interp_dense_parent.wgsl'),
    loadShader(device, 'shaders/amr_interp_pool_parent.wgsl'),
    loadShader(device, 'shaders/amr_step1.wgsl'),
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
    // bindings 7/8 were level 2's blockCriterion/blockSlot, for the per-pass
    // cascade. Gone with it (B2-2d). Holes, not renumbered.
    // binding 9: ?diag=1 convergence counters. Always bound; never touched at DIAG=0.
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // binding 10: level 1's WANT array (B2). Always bound; only read when
    // CASCADE != 0, and only written by the decide() entry point.
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
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
    // binding 8 was childQuadrant (it held `slot % 4`); B2 is what that
    // recovery was for -- this is the child level's WANT array.
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    // bindings 9/10 were childOriginX/Y and 13/14 parentOriginX/Y. All four
    // gone (B3-5): the origin is `block * RB * 2^-(m-1)` in closed form, so
    // the kernel derives it -- see amr_manage_pool.wgsl's parentOriginL0.
    // binding 11 was parentBlockSlot, for the neighbour-active veto.
    // Gone with it (B2-2d).
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    // binding 15 was grandchildBlockSlot, for hasGrandchild.
    // Gone with it (B2-2d).
  ]});
  const avgBGL = device.createBindGroupLayout({ label: 'avgBGL', entries: [
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
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    // binding 6: blockSlot -- neighbour-addressed streaming (see the
    // DIRECT_GHOST override in shaders/amr_step1.wgsl). Present in the layout
    // even under ?ghostcopy=1, where the shader simply never reads it.
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
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
  // Split from fineConstants, which also drives the render fragment --
  // render.wgsl has no F16 override and WebGPU rejects an undeclared one.
  const avgConstants = { W, H, RB, F16, DC_PRE };

  const interpConstants = { W, H, RB, GHOST_ONLY: 1, F16, DC_PRE };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0, F16, DC_PRE };
  const interpFFConstants = { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1, F16, DC_PRE };
  let stepConstants = makeStepConstants();
  let step1Constants = { ...stepConstants, RB, DIRECT_GHOST: GHOST_COPY ? 0 : 1 };
  const criterionConstants = { W, H };
  // HAS_BODY=0: isNearBody is unconditionally false (shaders/amr_manage.wgsl).
  const manageConstants = { DIAG, W, H, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD, HAS_BODY: 0  };

  // Re (via FORCE_X/WALL_U1) is baked into the L0/L1(+pool) step pipelines
  // as an override -- changing it means recreating those pipelines (cheap:
  // same shader modules, no recompilation from source), matching
  // main-channel.js's identical setRe.
  let stepPL, step1PL;
  function makeStepPipelines() {
    stepConstants = makeStepConstants();
    step1Constants = { ...stepConstants, RB, DIRECT_GHOST: GHOST_COPY ? 0 : 1 };
    stepPL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
      compute: { module: stepSM, entryPoint: 'main', constants: { ...stepConstants, F16 } }
    });
    step1PL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
      compute: { module: step1SM, entryPoint: 'main', constants: { ...step1Constants, F16 } }
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
  const interpPoolConstants = { RB, GHOST_ONLY: 1, F16, DC_PRE };
  const interpPoolInitConstants = { RB, GHOST_ONLY: 0, F16, DC_PRE };
  const interpPoolFFConstants = { RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1, F16, DC_PRE };
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
    compute: { module: avgSM, entryPoint: 'main', constants: avgConstants }
  });
  const avgPoolPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgPoolBGL] }),
    compute: { module: avgPoolSM, entryPoint: 'main', constants: { RB, F16, DC_PRE } }
  });
  const criterionPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [criterionBGL] }),
    compute: { module: criterionSM, entryPoint: 'main', constants: criterionConstants }
  });
  // plans/2D-backport.md B2: the 2:1 closure's own pipelines. Built but
  // NOT yet in dispatchMacroStep -- see debugCascadeRoundTrip.
  const cascadeSM = await loadShader(device, 'shaders/amr_cascade.wgsl');
  const cascade = makeCascadePipelines(device, cascadeSM, pools, N_LEVELS);

  // B2: the want-set producers. Same bind group and constants as
  // coarsen/refine -- they are the same decision, minus the neighbours.
  const manageDecidePL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'decide', constants: manageConstants }
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
  const managePoolDecidePLs = {};
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
      ...childParams,
      HAS_BODY: 0,
    };
    criterionPoolPLs[m] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [criterionPoolBGL] }),
      compute: { module: criterionPoolSM, entryPoint: 'main', constants: { RB, NBX_PARENT: parentPool.NBX } }
    });
    managePoolDecidePLs[m] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [managePoolBGL] }),
      compute: { module: managePoolSM, entryPoint: 'decide', constants: poolConstants }
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

  let stepBG_ab, stepBG_ba;
  function makeStepBindGroups() {
    stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
    stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});
    // Level 1's own fine-step bind groups, in the SAME layout every other level
    // uses -- the c>=2 loop below builds the identical pair. Level 1 differs
    // only in what its levelParams says (dxL 0.5, parentTau = L0's own tau);
    // there is no longer a second kernel, layout or pipeline for it.
    // plans/2D-backport.md B3-1.
    pools[1].step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_b } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 5, resource: { buffer: pools[1].levelParamsBuf } }, { binding: 6, resource: { buffer: pools[1].blockSlotBuf } }]});
    pools[1].step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 5, resource: { buffer: pools[1].levelParamsBuf } }, { binding: 6, resource: { buffer: pools[1].blockSlotBuf } }]});
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
  const manageBG = device.createBindGroup({ layout: manageBGL, entries: [{ binding: 0, resource: { buffer: pools[1].blockCriterionBuf } }, { binding: 1, resource: { buffer: pools[1].blockSlotBuf } }, { binding: 2, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 3, resource: { buffer: pools[1].freeListBuf } }, { binding: 4, resource: { buffer: pools[1].freeCountBuf } }, { binding: 5, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 6, resource: { buffer: cardStateBuf } }, { binding: 9, resource: { buffer: diagBuf } }, { binding: 10, resource: { buffer: pools[1].wantBuf } }]});

  const criterionPoolBGs = {};
  const managePoolBGs = {};
  for (let m = 1; m < N_LEVELS - 1; m++) {
    const parentPool = pools[m];
    const childPool = pools[m + 1];
    // BUGFIX: ALWAYS the parent pool's own finePoolVel, never velBuf.
    // amr_criterion_pool.wgsl's binding 0 is the PARENT LEVEL's fine pool
    // velocity and it addresses that buffer BY POOL SLOT
    // (slot*(FB*FB) + fy*FB + fx). This shader only ever runs with a pool
    // level as its parent -- m >= 1 -- so there is no dense case to special-
    // case here; the `m === 1 ? velBuf : ...` this replaces was the
    // dense-parent pattern the neighbouring interp/step bind groups legitimately
    // use, copied to a shader that has no dense-parent variant.
    //
    // Handing it velBuf fed a dense, cellIndex-addressed L0 buffer to
    // slot-addressed reads: wrong layout for every slot, and past roughly the
    // first third of the slots the reads run off the end of a buffer less than
    // half the size the pool layout expects. The result was a level-2
    // blockCriterion of essentially ZERO everywhere -- measured 2^-39.86 for
    // all 85 active L1 parents, against a host reconstruction from level 1's
    // own field showing up to 2^-5.6.
    //
    // So amr_manage_pool.wgsl's refine() saw maxCrit ~= 0 for every parent,
    // desiredLevel(toPhysical(eps)) was 0, and the vorticity criterion could
    // NEVER promote a tile to level 2. Level 2 was 100% geometry-forced --
    // which is exactly what a phi scan showed independently before the cause
    // was known: L2 covered block centres at phi -0.5 .. 7.9, the
    // childLevel-2 forced halo (margin 8) and nothing beyond it. That pins
    // the L1/L2 boundary a few cells off the body, so every shed vortex
    // crosses it right at the trailing edge -- the reported block artifacts
    // and the lumpiness induced on the shed vortices.
    const parentVel = parentPool.finePoolVel;
    const parentSlotToBlockBuf = m === 1 ? pools[1].slotToBlockBuf : parentPool.slotToBlockBuf;
    const grandchildPool = (m + 2) < N_LEVELS ? pools[m + 2] : null;

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
      { binding: 8, resource: { buffer: childPool.wantBuf } },
      { binding: 12, resource: { buffer: parentSlotToBlockBuf } },
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

    childPool.step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: childPool.finePoolF_b } },
      { binding: 3, resource: { buffer: childPool.finePoolVel } },
      { binding: 4, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 5, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 6, resource: { buffer: childPool.blockSlotBuf } },
    ]});
    childPool.step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_b } },
      { binding: 2, resource: { buffer: childPool.finePoolF_a } },
      { binding: 3, resource: { buffer: childPool.finePoolVel } },
      { binding: 4, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 5, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 6, resource: { buffer: childPool.blockSlotBuf } },
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
      // NO LEVEL SPLIT SINCE B3-1: one kernel, one pipeline, every level.
      const bg = readCur === 'a' ? pool.step1BG_ab : pool.step1BG_ba;
      const p = enc.beginComputePass(); p.setPipeline(step1PL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
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
    // Legacy same-level fine-fine refresh (?ghostcopy=1 only). The default
    // path needs no pass here: substep B's own gather reaches into the
    // neighbour tile directly, so it reads the neighbour's post-`average`
    // interior rather than a copy taken before that average landed. See the
    // DIRECT_GHOST override in shaders/amr_step1.wgsl.
    if (GHOST_COPY) fineFineRefresh();
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

      // ── B2: ONE SWEEP, NO FIXED POINT ────────────────────────────────────
      //
      // decide (every level, own reason only) -> close under the 2:1 rule ->
      // coarsen finest-first -> refine coarsest-first. Once.
      //
      // WHY ONE PASS IS ENOUGH: the closure is transitive, so after it runs
      // want[m] already contains everything want[m+1] will need a parent for.
      // The legacy loop below iterates because its per-pass tests only ever
      // see one hop at a time.
      //
      // THE ORDERS STILL MATTER, but for the ALLOCATOR, not for balance.
      // Coarsen runs finest-first because a child's slots must return to the
      // free list before its parent's tile does; refine runs coarsest-first
      // because a level-(m+1) quad can only be carved from an ACTIVE level-m
      // parent slot. Neither is about 2:1 any more.
      // The want buffers must be CLEARED, not just overwritten: decide() at
      // level >= 2 is dispatched over PARENT slots and never visits a block
      // whose parent is inactive, so a stale want would survive there and
      // resurrect a tile the criterion has stopped asking for.
      for (let m = 1; m < N_LEVELS; m++) enc.clearBuffer(pools[m].wantBuf);

      { const p = enc.beginComputePass(); p.setPipeline(manageDecidePL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end(); }
      for (let m = 1; m < N_LEVELS - 1; m++) {
        const wg = Math.ceil(pools[m].MAX_FINE_BLOCKS / 64);
        const p = enc.beginComputePass(); p.setPipeline(managePoolDecidePLs[m]); p.setBindGroup(0, managePoolBGs[m]); p.dispatchWorkgroups(wg); p.end();
      }

      encodeCascade(enc, cascade, N_LEVELS);

      for (let m = N_LEVELS - 1; m >= 1; m--) {
        if (m === 1) {
          const p = enc.beginComputePass(); p.setPipeline(manageCoarsenPL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
        } else {
          const wg = Math.ceil(pools[m].MAX_FINE_BLOCKS / 64);
          const p = enc.beginComputePass(); p.setPipeline(managePoolCoarsenPLs[m - 1]); p.setBindGroup(0, managePoolBGs[m - 1]); p.dispatchWorkgroups(wg); p.end();
        }
      }
      for (let m = 1; m < N_LEVELS; m++) {
        if (m === 1) {
          const p = enc.beginComputePass(); p.setPipeline(manageRefinePL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
        } else {
          const wg = Math.ceil(pools[m - 1].MAX_FINE_BLOCKS / 64);
          const p = enc.beginComputePass(); p.setPipeline(managePoolRefinePLs[m - 1]); p.setBindGroup(0, managePoolBGs[m - 1]); p.dispatchWorkgroups(wg); p.end();
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
    writeF(f_a, initF(), NCELLS);
    writeF(pools[1].finePoolF_a, initFPool(), MAX_FINE_BLOCKS * NCELLS1);
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(pools[1].blockSlotBuf, 0, new Int32Array(NBLOCKS).fill(-1));
    device.queue.writeBuffer(pools[1].slotToBlockBuf, 0, new Int32Array(MAX_FINE_BLOCKS).fill(-1));
    device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
    device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
    for (let c = 2; c < N_LEVELS; c++) {
      const pool = pools[c];
      writeF(pool.finePoolF_a, initFPool(pool.MAX_FINE_BLOCKS), pool.MAX_FINE_BLOCKS * NCELLS1);
      device.queue.writeBuffer(pool.blockSlotBuf, 0, new Int32Array(pool.NBLOCKS).fill(-1));
      device.queue.writeBuffer(pool.slotToBlockBuf, 0, new Int32Array(pool.MAX_FINE_BLOCKS).fill(-1));
      const freeQuads = Array.from({ length: pool.MAX_FINE_BLOCKS / 4 }, (_, i) => i);
      device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(freeQuads));
      device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([freeQuads.length]));
    }
    // Deliberately does NOT reset autoRefine to true (unlike main-cylinder-
    // amr.js's identical reset, whose default IS true) -- this harness
    // defaults autoRefine OFF for a real reason (see its own declaration's
    // header), and every benchmark case calls reset() between runs; forcing
    // it back on here would silently re-enable the unsafe path on every
    // single case instead of just once at page load.
    macroStepCounter = 0;
    useB = false;
    step = 0;
  }

  // Pool indirection readback -- amr2d-gpu.mjs, five copies before B3a.
  const readPoolIndirection = (level = 1) => readPoolIndirectionOn(device, pools, level);
  async function setAutoRefine(v) { autoRefine = !!v; }

  // Active blocks of one level -- amr2d-gpu.mjs, five copies before B3a.
  const debugListActiveBlocks = (level = 1) => listActiveBlocks(device, pools, level);
  // 2:1 BALANCE. The rule, the multi-level readback and the corner-balance
  // decision all live in amr2d-gpu.mjs / amr2d.mjs now -- they were five
  // copies of one checker across the AMR pages, byte-identical in the
  // readback and already drifted in the checker. amr2d.mjs's pure half is
  // exercised by `make test` on inputs that VIOLATE the invariant, which an
  // in-page copy could never be.
  //
  // Cheap enough to call periodically during development and validation; not
  // wired into the live per-macro-step path, which would need a GPU-side
  // assertion mechanism this project does not have.
  const debugCheck21Balance = () => check21BalanceOnGPU(device, pools, N_LEVELS);
  // How far this topology is from the 2:1 closure -- amr2d-gpu.mjs.
  // Reports, does not gate: the shipped manager implements the rule as
  // per-pass tests and only the VETO half of the refine cascade exists,
  // so this is expected to be nonzero until plans/2D-backport.md B2.
  const debugCheckRefinementClosure = () => checkRefinementClosureOnGPU(device, pools, N_LEVELS);
  // A slot's quadrant is `slot % 4` by construction -- amr2d.mjs's
  // quadrantOfSlot. This scores the stored buffer against that rule over
  // live ACTIVE slots, which is what lets the pool manager stop writing it.
  const debugCheckSlotQuadrants = () => checkSlotQuadrantsOnGPU(device, pools, N_LEVELS);

  // THE GPU CASCADE, SCORED AGAINST THE HOST TWIN (plans/2D-backport.md B2-2).
  //
  // Nothing in the simulation consumes the want arrays yet -- this is the
  // closure built and proved correct BEFORE anything depends on it, the same
  // order B4-1 took before B4-2. `seeds` is a list of want sets in
  // amr2d.mjs's "bx,by" form; each is written to the GPU, closed by
  // shaders/amr_cascade.wgsl, and required to match cascade21's result
  // EXACTLY. Omit it and a default battery runs -- see makeCascadeSeeds.
  //
  // The seeds matter more than the driver: a closure only ever run on valid
  // input is indistinguishable from one that returns its input unchanged, so
  // the battery deliberately includes sets that VIOLATE 2:1 balance, sit on
  // the periodic seam, and want one child of a quad with no siblings.
  async function debugCascadeRoundTrip(seeds) {
    const battery = seeds || makeCascadeSeeds(pools, N_LEVELS);
    const results = [];
    for (const { name, sets } of battery) {
      const r = await cascadeRoundTrip(device, pools, N_LEVELS, cascade, sets);
      results.push({ name, ...r });
    }
    return { ok: results.every(r => r.ok), results };
  }


  // The rigid body's own state, keyed by common_geometry.wgsl's CardState --
  // amr2d-gpu.mjs, three byte-identical copies before B4.
  const debugReadCardState = () => readCardState(device, cardStateBuf);

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
    debugCheckRefinementClosure,
    debugCheckSlotQuadrants,
    debugCascadeRoundTrip,
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
