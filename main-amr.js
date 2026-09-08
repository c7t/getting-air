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

import { assembleShader } from './shader-loader.mjs';
import {
  deriveCardParams, parseCardParams, parseResLog2, reynoldsFromTau,
  AMR_DEFAULT_RES_LOG2, AMR_DEFAULT_LEVELS,
  tauAtLevel as tauAtLevelOf,
} from './card-params.mjs';
import { packF, unpackF, fWords } from './f-pack.mjs';

const canvas   = document.getElementById('c');
let deviceLost = false;
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
// Default is one step below main.js's own default (resLog2=8, W=256) --
// with the default levels=2, this reproduces the "lower far-field
// resolution, the fine level recovers the body's resolution" AMR win via
// the general BLOCKAGE/ASPECT/RE mechanism below (see the comment above
// `let BLOCKAGE`), generalizing what used to be a hardcoded A=32,B=4 "half
// of main.js's dense reference" special case.
let resLog2 = parseResLog2(urlParams, AMR_DEFAULT_RES_LOG2);

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
// RAISED 128 -> 256 together with the REFINE_THRESH retune above, and the
// two must move together: at -9/-10 the measured L1 demand over 84k steps
// was min 95 / median 112 / MAX 168, so leaving the cap at 128 would put the
// pool into permanent exhaustion. That failure is not graceful -- blocks are
// granted in blockID order, so the free list runs dry part-way through a row
// and the denied blocks form horizontal BANDS across the refined region (see
// the same failure diagnosed at 256^2 in the SDF commit). 256 gives ~1.5x
// headroom over the measured peak.
// 384, not 256. The 256 was sized from LATE-run demand (max 168 over 84k
// steps), but the early transient is the peak: with caps effectively removed,
// L1 demand rises to 182 around step 6k-12k before settling to ~110-130, and
// one measured run reached 229. 256 left only ~1.1x headroom over that, and
// running out is not graceful -- blocks are granted in blockID order, so the
// free list dries up part-way through a row and the denied blocks form
// horizontal BANDS across the refined region, which is the artifact this
// whole thread started from. Level 2's demand is stable at 132-140 (verified
// by varying its cap independently), so 256 there is ~1.8x and stays.
const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 384;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY; // coarse block grid

// ── Milestone 5 (plans/AMR-multilevel.md, plans/AMR-multilevel-M5.md):
// number of pool levels above L0. N_LEVELS=2 (default) is byte-identical
// to today's single-fine-level build (validated against a pre-M5
// baseline -- see the sub-plan). N_LEVELS>=3 allocates additional
// quadtree pool levels that no shader/dispatch reads yet (Milestone 6/7).
const N_LEVELS = urlParams.has('levels') ? parseInt(urlParams.get('levels')) : AMR_DEFAULT_LEVELS;
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
// Readback pipeline depth. Each stage is one frame in flight, so this also
// sets how many frames the app runs AHEAD of the readback -- at 250ms/frame
// on mobile, STAGES=3 means ~750ms of submit-to-readback latency (measured
// sync/gpu ~2.8-2.9, matching the depth) and three swapchain textures
// outstanding at once.
//
// Exposed because it is the remaining in-app lever on the "view twitches
// backward" report. Every simulation-side signal is clean -- step monotonic,
// field digest never repeating, window offset smooth -- which leaves
// PRESENTATION, and fewer frames in flight is less opportunity for the
// compositor to present them out of order or drop them. It is also a large
// latency win on a GPU-bound device, where the queue is full regardless, so
// shallower staging may cost little throughput.
//   ?stages=1  lowest latency, CPU waits on each readback
//   ?stages=2  compromise
//   ?stages=3  default, deepest pipeline
const STAGES_CFG = Math.max(1, Math.min(4, urlParams.has('stages') ? parseInt(urlParams.get('stages')) : 3));

const REFINE_EVERY = urlParams.has('refineEvery') ? parseInt(urlParams.get('refineEvery')) : 16;
// Vorticity refinement thresholds, log2|omega| per L0 block (see
// amr_criterion.wgsl, which reduces max|omega| per block, and
// amr_manage.wgsl's epsFor which does the comparison).
//
// RETUNED from -6/-7, which was leaving most of the shed wake unrefined.
// Measured on one frozen flow state at res=8 levels=3, blockage=3.3
// (1024 L0 blocks, domain max |omega| per block = 2.76e-2):
//
//   thresh   |omega| >=   blocks selected   % of wake (|omega|>=1e-3) covered
//     -6       1.56e-2         14                    7%
//     -7       7.81e-3         59                   29%
//     -8       3.91e-3         91                   45%
//     -9       1.95e-3        153                   76%
//    -10       9.77e-4        202                  100%
//
// At -6 the bar sat at 57% of the DOMAIN MAXIMUM: only blocks carrying more
// than half the peak vorticity anywhere got refined, which in practice meant
// the body's immediate surroundings and nothing else. 68 of the 77 refined
// blocks were coming from geometry forcing (isNearBody), not from this
// criterion at all. A vortex shed from the card stayed refined only while
// inside the body's geometric halo; once it convected out it dropped to L0
// and dissipated, so a trail that is evenly spaced on the dense solver came
// out of the AMR build with vortices MISSING -- the reported symptom.
//
// -9 covers 76% of vorticity-bearing blocks. -10 covers 100% but refines
// ~20% of the whole domain, which starts giving back the point of AMR;
// available via ?refineThresh=-10&coarsenThresh=-11 if fidelity matters more
// than cost for a given run.
//
// CAVEAT worth knowing before re-tuning: an absolute threshold is
// structurally fragile here. The domain peak |omega| was measured swinging
// 2.4x (2.4e-2 .. 5.8e-2) across runs at identical nominal physics, purely
// from where the card is in its tumble, so no single constant is right at
// all phases. A criterion relative to the current domain maximum would be
// scale- and phase-free; that is a design change to the refinement
// machinery, not a retune, and has not been attempted.
const REFINE_THRESH = urlParams.has('refineThresh') ? parseFloat(urlParams.get('refineThresh')) : -9;
const COARSEN_THRESH = urlParams.has('coarsenThresh') ? parseFloat(urlParams.get('coarsenThresh')) : -10;

// Geometry-forced refinement (see amr_manage.wgsl's isNearBody): blocks near
// the card's SDF -- now or FORCE_REFINE_LOOKAHEAD macro-steps from now, by
// linear extrapolation of vx/vy/omega -- are always refined and never
// coarsened, independent of the vorticity criterion above. Fixes the
// "blunting" gap where a lagging vorticity signal leaves the card's own
// sharp geometry on the coarse grid (e.g. the whole startup transient,
// before any wake vorticity exists).
//
// MARGIN: 16, retuned from the original 8 (which its own comment flagged as
// "a starting point, not measured"). This is the measured value; two things
// make 8 too small.
//
// 1. isNearBody tests the block's CENTRE, not its nearest point. A BLOCK=8
//    block's centre is up to 4*sqrt(2) ~ 5.66 cells from its own nearest
//    corner, so a margin of M only GUARANTEES refinement out to M - 5.66.
//    At M=8 that is 2.3 cells; measured coverage was indeed 100% only to
//    phi ~ 2.5, decaying to 52% at phi=8 and 5% by phi=12.
// 2. The boundary layer is thicker than that. Binning the field by distance
//    from the surface at page defaults (Re=1067, A=32), the body-relative
//    speed does not plateau until phi ~ 12-15 L0 cells, and |grad u| is
//    still ~21% of its peak where coverage has already fallen to half.
//
// So the coarse/fine interface sat INSIDE the boundary layer, and blocks
// flipped refined/coarse as the card translated past them -- the periodic
// artifacts visible at block boundaries. Measured directly, as the ratio of
// |d.omega| across level transitions to |d.omega| between same-level
// neighbours at the same distance from the body:
//
//   margin=8   437 seam pairs inside phi<8, ratio 1.39x-1.86x at phi 5-8
//   margin=16   38 seam pairs inside phi<8, ratio 1.02x  (artifact gone)
//
// The vorticity criterion cannot cover this gap on its own: measured BL
// vorticity is ~2.7e-3 = 2^-8.5, well under REFINE_THRESH=-6 (2^-6), so it
// never fires there and refinement near the body rests entirely on this
// margin.
//
// Cost is ~nil today: active L1 blocks go 54 -> ~92 (median), still inside
// the 128-slot default pool (max observed 100), and frame time is unchanged
// (median 7.2ms -> 6.0-7.0ms across repeats) because every pool pass already
// dispatches MAX_FINE_BLOCKS slots regardless of how many are active --
// inactive slots early-out, so the work was already being paid for. Raising
// MAX_FINE_BLOCKS *would* cost real time; raising this margin does not.
//
// Deliberately NOT applied to main-cylinder-amr.js, which has its own copy
// of this constant, a different geometry/regime, and currently-passing
// physics validation -- retuning it needs its own measurement. amr_manage.
// wgsl's isNearBody is shared by both pages, so the centre-vs-corner
// semantics are left alone rather than changed underneath the harness.
// Refinement LADDER (shaders/common_refine.wgsl, AGAL Algorithm 3).
// REFINE_THRESH above is the base rung -- the physical log2|omega| at which a
// region earns its FIRST level of refinement. Each further level costs
// N_REFINE_INC more octaves, so
//     desired >= k   <=>   log2|omega|_physical >= REFINE_THRESH + INC*(k-1)
// AGAL's own default INC is 1.0 (input/input.txt), i.e. one octave per level.
// N_REFINE_MAX clamps the criterion from above, as AGAL does.
const N_REFINE_INC = urlParams.has('refineInc') ? parseFloat(urlParams.get('refineInc')) : 1.0;
const N_REFINE_MAX = urlParams.has('refineMax') ? parseFloat(urlParams.get('refineMax')) : 1.0;

const FORCE_REFINE_MARGIN = urlParams.has('forceRefineMargin') ? parseFloat(urlParams.get('forceRefineMargin')) : 16;
// get_phi returns a cheap LOWER BOUND once it exceeds SDF_FAR=64 (see
// shaders/common_geometry.wgsl). That is exact for every consumer only while
// all phi thresholds stay below it, and FORCE_REFINE_MARGIN is URL-settable,
// so check rather than trust.
// ?sdfFar= overrides get_phi's far-field early-out cutoff (see
// shaders/common_geometry.wgsl). A very large value disables the early-out
// entirely, which is the A/B for whether it helps or hurts on a given GPU.
const SDF_FAR = urlParams.has('sdfFar') ? parseFloat(urlParams.get('sdfFar')) : 64;
// ?f16=1 / ?f16=2: store `f` as packed half pairs (5 u32 words/cell instead of 9
// f32s), cutting its traffic 1.80x. plans/perf-characterization.md measures
// the phone as bandwidth bound with `f` as essentially all the traffic, and
// settles the accuracy question by experiment -- see shaders/common_fpack.wgsl
// for the layout and f-pack.mjs for the host side.
//
// DEFAULT OFF while it earns its place on real devices. Off is not merely
// "the old numbers": F16=0 makes the shaders bitcast one f32 per u32, which
// is byte-for-byte the previous layout, so it is the same computation and not
// a second code path to keep honest.
//
// Buffer ALLOCATION is deliberately unchanged (still 9 words/cell) -- this is
// a bandwidth change, not yet a footprint one. That keeps every
// arrayLength(&f)/9u stride derivation in the shaders correct as-is and every
// page's buffer sizing untouched; shrinking the allocation is a separate step
// once this is proven.
const F16 = urlParams.has('f16') ? (parseInt(urlParams.get('f16')) || 0) : 0;

if (FORCE_REFINE_MARGIN >= SDF_FAR) {
  throw new Error(`?forceRefineMargin=${FORCE_REFINE_MARGIN} is at or above get_phi's SDF_FAR cutoff (${SDF_FAR}) -- ` +
    `beyond that the far-field early-out in shaders/common_geometry.wgsl returns a lower bound and isNearBody ` +
    `would silently under-refine. Raise SDF_FAR together with it if you really need a margin this large.`);
}

const FORCE_REFINE_LOOKAHEAD = urlParams.has('forceRefineLookahead') ? parseFloat(urlParams.get('forceRefineLookahead')) : REFINE_EVERY;
// L0 window-space edge band (coarse cells) excluded from vorticity-driven
// refinement -- keeps fine blocks out of the ALBC sponge (amr_step.wgsl
// SPONGE_W=4). A fixed L0-window strip, so the same value applies at every
// refinement level (unlike FORCE_REFINE_MARGIN). Default 8; ?spongeExclude=0
// disables it.
const SPONGE_EXCLUDE_W = urlParams.has('spongeExclude') ? parseFloat(urlParams.get('spongeExclude')) : 8;

// Milestone 10: per-CHILD-level threshold overrides -- see
// main-cylinder-amr.js's copy of this function for the full rationale (a
// level-2 block's vorticity is measured on the same RB=8 stencil at half
// the physical spacing of level 1's, so the same physical feature reads as
// a different numeric |omega| one level down; reusing L0->L1's thresholds
// verbatim for L1->L2 is not expected to be correct by construction).
// `refineThresh{child}`/`coarsenThresh{child}`/etc. (child=2,3,...) override
// the L(child-1)->L(child) decision; unset levels fall back to the base
// (L0->L1) values, so a build that never sets them is unchanged -- EXCEPT
// FORCE_REFINE_MARGIN, whose default fallback scales by the parent
// level's own cell size (cellSizeL0AtLevel(childLevel-1)) instead of
// reusing the raw base value -- see main-cylinder-amr.js's copy for why
// (live-verified: without scaling, N_LEVELS=4 saturates level 3's entire
// pool on first contact instead of forming a thin shell).
function paramsForChildLevel(childLevel) {
  if (childLevel === 1) {
    return { REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD };
  }
  const get = (name, base) => urlParams.has(`${name}${childLevel}`) ? parseFloat(urlParams.get(`${name}${childLevel}`)) : base;
  // LEVEL-CONSISTENT VORTICITY THRESHOLD.
  //
  // amr_criterion.wgsl and amr_criterion_pool.wgsl both reduce a raw
  // lattice-cell velocity DIFFERENCE (`(u[x+1]-u[x-1])*0.5`) with no division
  // by the cell size. AGAL, which this criterion is derived from, divides by
  // the level's own dx (solver_lbm_criterion.cu: `.../(2.0*dx_L)`), making
  // the criterion a physical velocity gradient that means the same thing at
  // every level.
  //
  // Without that division, a level-m cell is 2^-m the size of an L0 cell, so
  // the same PHYSICAL vorticity produces an omega 2^-m as large -- yet every
  // level was compared against the same absolute threshold. Deeper levels
  // were therefore systematically under-refined by 2^m: a level-1 block
  // needed 2x the physical vorticity of an L0 block to earn level-2
  // children, a level-2 block 4x. A vortex became LESS likely to keep its
  // refinement the moment it got refined, which is the compounding form of
  // the "shed vortices go missing" symptom.
  //
  // Correcting the threshold instead of the shader is exact, not an
  // approximation: the comparison is log2(omega) >= THRESH, and
  //   log2(omega_physical) = log2(omega_lattice / dx_m) = log2(omega_lattice) + m
  // so requiring log2(omega_lattice) >= THRESH - m is identical to dividing
  // by dx_m. m is the level the criterion is EVALUATED on, which for a
  // decision about creating childLevel tiles is childLevel-1. childLevel===1
  // is evaluated on L0 (m=0, dx=1) and is returned unchanged above, so the
  // measured L0 tuning is untouched.
  // NOTE: the per-level shift that used to live here is gone. The ladder in
  // common_refine.wgsl now converts the criterion to physical units itself
  // (amr_manage_pool.wgsl's toPhysical, from PARENT_CELL_SIZE_L0), so
  // applying a shift here as well would correct it twice.
  return {
    REFINE_THRESH: get('refineThresh', REFINE_THRESH),
    COARSEN_THRESH: get('coarsenThresh', COARSEN_THRESH),
    FORCE_REFINE_MARGIN: get('forceRefineMargin', FORCE_REFINE_MARGIN * cellSizeL0AtLevel(childLevel - 1)),
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
// Shared verbatim with main.js via card-params.mjs -- see that module's
// header for why these live in one place rather than two, and for what each
// quantity means. The property this page depends on specifically: because
// card size and flow regime are resolution-independent (BLOCKAGE/ASPECT/RE)
// rather than raw lattice-cell counts, pasting the same
// ?blockage=&aspect=&re=&ut= onto this page and main.js's reproduces the
// identical physical system. The AMR resource win is then purely a matter of
// choosing a LOWER `res` here than on the dense page, with `levels`
// recovering the missing resolution at the body -- generalizing what used to
// be a hardcoded A=32,B=4 "half of main.js's dense reference" special case.
// tools/test-card-params.js asserts that equivalence directly.
let { BLOCKAGE, ASPECT, I_STAR, RE, U_T } = parseCardParams(urlParams);

// Derived in recalculate() below. TAU here is this page's L0 (coarsest) tau;
// every finer level's own tau follows from it via tauAtLevel().
let A, B, TAU, RHO_B, MASS, I_BODY, G_LU, G_EFF;

function recalculate() {
  ({ A, B, TAU, RHO_B, MASS, I_BODY, G_LU, G_EFF } =
    deriveCardParams({ W, BLOCKAGE, ASPECT, I_STAR, RE, U_T }));
}
recalculate();

// ── Milestone 6 (plans/AMR-multilevel.md): recursive fine tau. L0's own tau
// is TAU (read live off CardState by the dense shader); every deeper level
// applies the Dupuis-Chopard relation amr_interp_dense_parent.wgsl already
// uses once (tau_fine = 2*tau_coarse - 0.5), walked m times. tauAtLevel(0)
// is L0's own tau, tauAtLevel(1) is L1's (what amr_interp_pool_parent.wgsl
// needs as `parentTau` when interpolating L1->L2), etc. Plain JS, not a GPU
// readback -- TAU is already a live JS variable the slider mutates directly.
// Thin wrapper over card-params.mjs's pure tauAtLevelOf(tau0, m) so callers
// here keep the existing one-argument form against the live TAU.
function tauAtLevel(m) {
  return tauAtLevelOf(TAU, m);
}

const FSCALE  = 1e7;

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
    // COPY_DST is load-bearing, not boilerplate: debugSnapshotLoad writes
    // this buffer via queue.writeBuffer, which is a validation error --
    // silently discarded -- without it. See velBuf's own note below.
    finePoolVel: device.createBuffer({ size: maxFineBlocks * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
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
    // BUGFIX: same "fix at the source, every level" gap as the blockSlot/
    // slotToBlock -1 init below, but for the free-list/free-count pair --
    // level 1 gets its eager freeListBuf/freeCountBuf write from an
    // explicit caller-side write right after the pools loop, but that was
    // never generalized to levels >=2 either. Left at WebGPU's zero-init
    // default, freeCountBuf reads back 0 ("no free quads"), so refine()
    // always takes the "pool exhausted" branch and NO level>=2 quad can
    // ever be granted until something explicitly calls resetSim() --
    // silently, with no GPU validation error, since this is application
    // logic, not an API misuse. resetSim()/debugSnapshotLoad already write
    // these correctly on their own paths; nothing wrote them at bare
    // allocation time, and nothing calls resetSim() automatically on page
    // load, so a fresh page (or any driver script that steps without
    // calling reset() first) saw permanent level>=2 refinement failure.
    const freeQuads_m = maxFineBlocks / 4;
    device.queue.writeBuffer(pool.freeListBuf, 0, new Int32Array(freeQuads_m).map((_, i) => i));
    device.queue.writeBuffer(pool.freeCountBuf, 0, new Int32Array([freeQuads_m]));
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
  // GPU-side timing. Restored from a `0 &&` hard-disable (commit 30c9e86,
  // "Nerf timestamp for mobile") which threw away desktop timing to
  // accommodate mobile -- but the feature test on this same line already
  // handles that: an adapter without 'timestamp-query' simply reports
  // wall-clock instead. The reason it had to be disabled rather than merely
  // feature-detected is that the old code used encoder.writeTimestamp(),
  // which was REMOVED from WebGPU (it needed the
  // chromium-experimental-timestamp-query-inside-passes flag); the supported
  // form is timestampWrites in a pass descriptor, which is what is used
  // below. Without this, "GPU" and "SYNC" in the overlay were the same
  // wall-clock number wearing two labels.
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
  // Milestone 9: shaders/amr_manage_pool.wgsl needs 16 storage bindings in
  // one bind group (childCriterion/blockSlot/slotToBlock/freeList/
  // freeCount/newlyActivated/state/parentSlot/quadrant/originX/originY,
  // 4 parent-level read-only mirrors for the 2:1-balance neighbor checks,
  // plus 1 grandchild-level read-only mirror (grandchildBlockSlot) added
  // for the N>=4 2:1-balance cascade fix -- see managePoolBGL's own
  // comment) -- past the WebGPU spec-MINIMUM maxStorageBuffersPerShaderStage
  // of 8 (already the exact ceiling several existing 8-binding layouts sit
  // at, e.g. step1PoolBGL), same "spec minimum, not a real GPU limit"
  // situation as maxStorageBufferBindingSize above. Same treatment: request
  // what's needed, capped at the adapter's real capability, fail loud (not
  // a cryptic validation error) if this GPU genuinely can't do it.
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

  // Slots 0/1 are the whole-frame span. 2..QUERY_CAP-1 are for
  // debugProfileMacroStep's per-pass breakdown (2 per pass), which is what
  // makes dispatch work attributable rather than guessed at. One macro-step
  // is ~9 passes at N=2 and ~20 at N=3, rising to ~32 on a refine step, so
  // 128 slots leaves generous headroom.
  const QUERY_CAP = 128;
  const querySet = hasTimestamp ? device.createQuerySet({
    type: 'timestamp',
    count: QUERY_CAP
  }) : null;
  const queryResolveBuffer = hasTimestamp ? device.createBuffer({
    size: QUERY_CAP * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
  }) : null;

  // A backgrounded tab on mobile is a common way to lose the GPU device, and
  // nothing here was watching for it: on loss every subsequent submit is
  // silently ignored, the frame loop keeps spinning, and the page just stops
  // advancing with no indication why. Surface it instead. `reason ===
  // 'destroyed'` is our own teardown and is not an error.
  device.lost.then((info) => {
    if (info.reason === 'destroyed') return;
    deviceLost = true;
    statusEl.textContent = `error: GPU device lost (${info.reason}) -- ${info.message || 'no message'}. Reload to restart.`;
    statusEl.style.color = '#f77';
    console.error('WebGPU device lost:', info);
  });

  device.pushErrorScope('validation');

  const ctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();

  // Reconfigure ONLY on a real size change. This used to run unconditionally
  // on every `resize` event, and both halves of it are destructive:
  // assigning canvas.width/height resets the drawing buffer even when the
  // value is unchanged, and ctx.configure() replaces the swapchain,
  // invalidating textures that in-flight command buffers still reference
  // (this page keeps up to STAGES frames in flight).
  //
  // On desktop `resize` fires when you resize the window, so the cost was
  // invisible. On a PHONE it fires constantly -- the URL bar hides and shows
  // on any scroll or drag, which includes touching the control sliders --
  // so the swapchain was being torn down and rebuilt underneath frames that
  // were already submitted. Reported symptom: the view "twitches back" a few
  // frames, correlated with moving sliders or switching away and back.
  //
  // Also guards the degenerate case: clientWidth/Height read 0 during some
  // layout transitions (and while hidden), and a 0-sized canvas is not a
  // valid configuration.
  let cfgW = 0, cfgH = 0;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (w <= 0 || h <= 0) return;      // mid-layout / hidden: nothing to configure
    if (w === cfgW && h === cfgH) return; // same size: reconfiguring is pure damage
    cfgW = w; cfgH = h;
    canvas.width = w;
    canvas.height = h;
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
  // COPY_DST: debugSnapshotLoad restores this with queue.writeBuffer.
  // Without the flag that write is a validation error and is silently
  // dropped, so a loaded snapshot keeps whatever ux/uy were already there.
  // The load path's own comment already describes this exact symptom
  // ("rho round-tripped exactly, but ux/uy didn't -- the asymmetry was the
  // tell") -- the writeBuffer call was added then, but the usage flag was
  // not, so the fix never actually took effect.
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
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

  // The ONLY two places the GPU's `f` layout and everyone else's meet.
  // Everything outside the GPU -- the initial equilibrium, the snapshot
  // format, the diagnostics below, and every tool under tools/ -- speaks f32
  // plane-major, unconditionally. Under F16 the GPU buffer holds packed half
  // pairs instead, so it is converted here on the way in and back on the way
  // out, and nothing downstream has to know. See f-pack.mjs for why that is
  // the boundary rather than teaching every consumer a second format.
  //
  // With F16 off both are the identity plus the copy the old code already
  // did, so this is not a new cost on the default path.
  const writeF = (buf, f32, ncells) => {
    const src = packF(f32, ncells, F16);
    device.queue.writeBuffer(buf, 0, src.buffer, src.byteOffset, ncells * fWords(F16) * 4);
  };
  const readF = (mapped, ncells) =>
    F16 ? unpackF(new Uint32Array(mapped), ncells, true) : new Float32Array(mapped).slice();
  const pools = [undefined]; // pools[0] unused -- L0 is the dense grid, not a pool level
  {
    let curNBX = NBX, curNBY = NBY; // level 1's logical grid = today's coarse block grid
    for (let m = 1; m < N_LEVELS; m++) {
      const maxFineBlocks = m === 1
        ? MAX_FINE_BLOCKS // unchanged param/default -- level 1 is byte-identical to today
        // 256, not 128: measured L2 demand at the retuned thresholds peaked
        // at 136 over 84k steps, and at res=8 levels=3 the old 128 default
        // saturated outright (128/128 active). See MAX_FINE_BLOCKS above for
        // why exhaustion here shows up as horizontal bands.
        : (urlParams.has(`maxFineBlocks${m}`) ? parseInt(urlParams.get(`maxFineBlocks${m}`)) : 256);
      const pool = allocLevelPool(device, U, m, curNBX, curNBY, maxFineBlocks);
      writeF(pool.finePoolF_a, initFPool(maxFineBlocks), maxFineBlocks * NCELLS1);
      pools.push(pool);
      curNBX *= 2; curNBY *= 2; // next level's logical grid extent (quadtree doubling per axis)
    }
  }

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  writeF(f_a, initF(), NCELLS);
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

  // See main.js's identical block for the full rationale: RE is the
  // canonical flow-regime state, TAU is the one control that goes the other
  // way (dragging it back-solves RE first).
  const blockageEl = document.getElementById('slider-BLOCKAGE');
  const aspectEl   = document.getElementById('slider-ASPECT');
  const iStarEl    = document.getElementById('slider-I_STAR');
  const reEl       = document.getElementById('slider-RE');
  const tauEl      = document.getElementById('slider-TAU');
  const utEl       = document.getElementById('slider-U_T');

  // Show the control's OWN value first, then the lattice-unit quantity it
  // derives -- see main.js's identical block for why the derived-only form
  // this replaces was actively misleading.
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
    recalculate();
    refreshDerivedReadouts();
    paramsDirty = true;
  };

  blockageEl.value = BLOCKAGE;
  aspectEl.value   = ASPECT;
  iStarEl.value    = I_STAR;
  reEl.value       = RE;
  tauEl.value      = TAU;
  utEl.value       = U_T;
  refreshDerivedReadouts();

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

  const [stepSM, frcSM, phySM, renSM, digestSM, interpDenseSM, interpPoolSM, step1SM, step1PoolSM, avgSM, avgPoolSM, criterionSM, manageSM, force1SM, force1PoolSM, criterionPoolSM, managePoolSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_force.wgsl'),
    loadShader(device, 'shaders/amr_physics.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    loadShader(device, 'shaders/amr_digest.wgsl'),
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
  // parent=level m>=1 -- see amr_manage_pool.wgsl's header. 16 bindings
  // (15 original + grandchildBlockSlot, added for the N>=4 2:1-balance
  // cascade fix -- both refine() and coarsen() only ever need EXISTENCE,
  // never level (m+2)'s criterion, so one shared buffer/layout covers
  // both) -- exactly this adapter's real maxStorageBuffersPerShaderStage,
  // not just the WebGPU spec minimum other buffer limits in this file hit.
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
    { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // grandchildBlockSlot
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
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    // TEMPORARY diagnostic (level-2 bounce-back sign investigation) -- see
    // amr_force1_pool.wgsl's own debugSlotForce header.
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});

  const constants = { W, H, SDF_FAR };
  // Same, plus the packed-f layout selector. Separate object because
  // `constants` is also fed to phy/render, whose modules don't declare F16 --
  // and WebGPU makes passing an undeclared override a pipeline-creation
  // error, not a warning. Every pipeline whose shader @includes
  // common_fpack.wgsl must get F16; no other pipeline may.
  const fConstants = { ...constants, F16 };
  const fineConstants = { W, H, RB, F16 };
  // Render fragment needs HAS_LEVEL2 to gate the level-2 override; keep it
  // separate from fineConstants, which is also fed to the avg compute
  // pipeline (whose shader has no HAS_LEVEL2 override).
  const renderConstants = { W, H, RB, HAS_LEVEL2: N_LEVELS > 2 ? 1 : 0 };
  // GHOST_ONLY=1: steady-state ghost-only reinterpolation (every macro-step).
  // GHOST_ONLY=0: full-slot fill, used once on block activation (see debugActivateBlock).
  const interpConstants = { W, H, RB, GHOST_ONLY: 1, F16 };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0, F16 };
  // Between-substep fine-fine-only ghost re-exchange (see amr_interp_c2f.wgsl's
  // FINE_FINE_ONLY note and the dispatch between f1a/f1b below).
  const interpFFConstants = { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1, F16 };
  const step1Constants = { W, H, RB, SDF_FAR, F16 };
  const criterionConstants = { W, H };
  const manageConstants = { W, H, SDF_FAR, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD, SPONGE_EXCLUDE_W, HAS_LEVEL2: N_LEVELS > 2 ? 1 : 0,
    N_REFINE_INC, N_REFINE_MAX, MAX_LEVEL: N_LEVELS - 1 };

  const stepPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
    compute: { module: stepSM, entryPoint: 'main', constants: fConstants }
  });
  const frcPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }),
    compute: { module: frcSM, entryPoint: 'main', constants: fConstants }
  });
  const phyPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [phyBGL] }),
    compute: { module: phySM, entryPoint: 'main', constants }
  });
  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main', constants },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }], constants: renderConstants },
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
  const interpPoolConstants = { RB, GHOST_ONLY: 1, F16 };
  const interpPoolInitConstants = { RB, GHOST_ONLY: 0, F16 };
  const interpPoolFFConstants = { RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1, F16 };
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
    compute: { module: avgPoolSM, entryPoint: 'main', constants: { RB, F16 } }
  });
  // ── Measurement-instrument pipeline twins (see benchSkip below) ──────────
  // Built unconditionally but only ever bound when the matching ?benchSkip=
  // group is set, so the normal dispatch path is untouched. Each is the SAME
  // shader module as its real counterpart with one override constant flipped,
  // which is what keeps them honest: a variant compiled from different source
  // could differ for reasons unrelated to the thing being measured (the trap
  // ?quantF16 fell into -- see plans/perf-characterization.md).
  //
  // *-noop: pass still encoded and dispatched at full width, returns before
  // touching any buffer. Difference vs. skipping the pass outright is the
  // fixed per-pass cost; the remainder is the work. See the NOOP override in
  // shaders/amr_interp_*.wgsl / amr_average_*.wgsl for the measured split.
  const noopPLs = {
    interpDense:  device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),           compute: { module: interpDenseSM, entryPoint: 'main', constants: { ...interpConstants,       NOOP: 1 } } }),
    interpDenseFF:device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),           compute: { module: interpDenseSM, entryPoint: 'main', constants: { ...interpFFConstants,     NOOP: 1 } } }),
    interpPool:   device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }), compute: { module: interpPoolSM,  entryPoint: 'main', constants: { ...interpPoolConstants,   NOOP: 1 } } }),
    interpPoolFF: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }), compute: { module: interpPoolSM,  entryPoint: 'main', constants: { ...interpPoolFFConstants, NOOP: 1 } } }),
    avg:          device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),              compute: { module: avgSM,         entryPoint: 'main', constants: { ...fineConstants,        NOOP: 1 } } }),
    avgPool:      device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [avgPoolBGL] }),          compute: { module: avgPoolSM,     entryPoint: 'main', constants: { RB, F16,                 NOOP: 1 } } }),
  };
  // step1-ring: fine step over the tile INTERIOR only, skipping the ghost
  // ring -- a proxy for FB 20 -> 16. See the SKIP_GHOST override in
  // shaders/amr_step1.wgsl for what it measured and what it corrects.
  const ringPLs = {
    step1:     device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),     compute: { module: step1SM,     entryPoint: 'main', constants: { ...step1Constants, SKIP_GHOST: 1 } } }),
    step1Pool: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [step1PoolBGL] }), compute: { module: step1PoolSM, entryPoint: 'main', constants: { ...step1Constants, SKIP_GHOST: 1 } } }),
  };
  // Milestone 8: level 1's own force pass. HAS_CHILD is baked in at
  // pipeline-creation time -- level 1 has exactly one dedicated pipeline
  // (not shared across levels), so whether level 2 exists is fixed for the
  // whole session (see amr_force1.wgsl's header).
  const force1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [force1BGL] }),
    compute: { module: force1SM, entryPoint: 'main', constants: { W, H, RB, HAS_CHILD: N_LEVELS > 2 ? 1 : 0, F16 } }
  });
  // Milestone 8: level>=2's own force pass, one pipeline reused across
  // every such level (no per-level overrides -- hasChild/dxL are runtime
  // LevelParams reads, see amr_force1_pool.wgsl's header).
  const force1PoolPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [force1PoolBGL] }),
    compute: { module: force1PoolSM, entryPoint: 'main', constants: { W, H, RB, F16 } }
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
    // HAS_GRANDCHILD (level m+2) for the 2:1-balance cascade -- see
    // amr_manage_pool.wgsl's header. Existence-based (hasGrandchild), not
    // criterion-based, so no separate grandchild-level threshold overrides
    // are needed here -- just whether that level exists at all.
    const hasGrandchild = (m + 2) < N_LEVELS;
    const poolConstants = {
      W, H, RB, SDF_FAR,
      NBX_PARENT: parentPool.NBX, NBY_PARENT: parentPool.NBY,
      PARENT_CELL_SIZE_L0: cellSizeL0AtLevel(m),
      PARENT_HAS_CACHED_ORIGIN: parentIsDense ? 0 : 1,
      SPONGE_EXCLUDE_W,
      ...childParams,
      N_REFINE_INC, N_REFINE_MAX, MAX_LEVEL: N_LEVELS - 1,
      HAS_GRANDCHILD: hasGrandchild ? 1 : 0,
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
  // Field digest (see shaders/amr_digest.wgsl): one dispatch per rendered
  // frame that fingerprints the L0 velocity field, so the watchdog can tell
  // whether a frame ever reproduces an EARLIER frame's field -- which
  // ordinary dynamics never does, but showing a stale buffer would.
  const digestBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC });
  const digestBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const digestBG = device.createBindGroup({ layout: digestBGL, entries: [
    { binding: 0, resource: { buffer: velBuf } },
    { binding: 1, resource: { buffer: digestBuf } },
  ]});
  const digestPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [digestBGL] }),
    compute: { module: digestSM, entryPoint: 'main', constants: { NCELLS } },
  });

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
    // Grandchild (level m+2) blockSlot for the 2:1-balance cascade -- dummy
    // when there's no such level (HAS_GRANDCHILD=0 gates it out of ever
    // being read, matching every other dummy-buffer fallback in this file).
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
    // TEMPORARY diagnostic (level-2 bounce-back sign investigation) -- see
    // amr_force1_pool.wgsl's own debugSlotForce header.
    childPool.debugSlotForceBuf = device.createBuffer({ size: childPool.MAX_FINE_BLOCKS * 8, usage: U.STORAGE | U.COPY_SRC });
    childPool.force1PoolBG = device.createBindGroup({ layout: force1PoolBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: childPool.finePoolF_a } },
      { binding: 2, resource: { buffer: forceBuf } },
      { binding: 3, resource: { buffer: childPool.slotToBlockBuf } },
      { binding: 4, resource: { buffer: childPool.originXBuf } },
      { binding: 5, resource: { buffer: childPool.originYBuf } },
      { binding: 6, resource: { buffer: childPool.levelParamsBuf } },
      { binding: 7, resource: { buffer: childBlockSlotBuf } },
      { binding: 8, resource: { buffer: childPool.debugSlotForceBuf } },
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
  const STAGES = STAGES_CFG;
  const stages = Array.from({ length: STAGES }, () => ({
    // 104 bytes of CardState + 16 bytes of field digest, read back together
    card: device.createBuffer({ size: 120, usage: U.MAP_READ | U.COPY_DST }),
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

    const f = readF(stagingF.getMappedRange(), NCELLS);
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    const card = Array.from(new Float32Array(stagingCard.getMappedRange()).slice());
    const fPool = readF(stagingFPool.getMappedRange(), MAX_FINE_BLOCKS * NCELLS1);
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
      const fPool_m = readF(st.f.getMappedRange(), pool.MAX_FINE_BLOCKS * NCELLS1);
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
      params: { A, B, BLOCKAGE, ASPECT, RE, I_STAR, TAU, U_T, resLog2 },
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
    writeF(f_a, f, NCELLS);
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
      writeF(pool.finePoolF_a, fPool_m, pool.MAX_FINE_BLOCKS * NCELLS1);
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
        if (!skipGroup('interp')) { const p = beginPass(enc, 'L0->L1 interp'); p.setPipeline(skipGroup('interp-noop') ? noopPLs.interpDense : interpPL); p.setBindGroup(0, readBG); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end(); }
      }
      const s = beginPass(enc, 'L0 step'); s.setPipeline(stepPL); s.setBindGroup(0, stepBG); s.dispatchWorkgroups(WGX, WGY); s.end();
      if (hasChild) {
        S_Advance(1, enc);
        const avgBG = useB ? avgBG_targetA : avgBG_targetB;
        if (!skipGroup('avg')) { const a = beginPass(enc, 'L1->L0 average'); a.setPipeline(skipGroup('avg-noop') ? noopPLs.avg : avgPL); a.setBindGroup(0, avgBG); a.dispatchWorkgroups(1, 1, MAX_FINE_BLOCKS); a.end(); }
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
      if (skipGroup('interp')) return; const p = beginPass(enc, `L${level}->L${level+1} interp`); p.setPipeline(skipGroup('interp-noop') ? noopPLs.interpPool : interpPoolParentPL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, childPool.MAX_FINE_BLOCKS); p.end();
    };
    const averageFromChild = (writeCur) => {
      if (!hasChild) return;
      const childPool = pools[level + 1];
      const bg = writeCur === 'a' ? childPool.avgPoolBG_targetA : childPool.avgPoolBG_targetB;
      if (skipGroup('avg')) return; const p = beginPass(enc, `L${level+1}->L${level} average`); p.setPipeline(skipGroup('avg-noop') ? noopPLs.avgPool : avgPoolPL); p.setBindGroup(0, bg); p.dispatchWorkgroups(1, 1, childPool.MAX_FINE_BLOCKS); p.end();
    };
    const substep = (readCur) => {
      if (isL1) {
        const bg = readCur === 'a' ? step1BG_ab : step1BG_ba;
        if (skipGroup('step1')) return; const p = beginPass(enc, 'L1 step'); p.setPipeline(skipGroup('step1-ring') ? ringPLs.step1 : step1PL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end();
      } else {
        const bg = readCur === 'a' ? pool.step1PoolBG_ab : pool.step1PoolBG_ba;
        if (skipGroup('step1')) return; const p = beginPass(enc, `L${level} step`); p.setPipeline(skipGroup('step1-ring') ? ringPLs.step1Pool : step1PoolPL); p.setBindGroup(0, bg); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
    };
    const fineFineRefresh = () => {
      if (isL1) {
        if (skipGroup('ghost')) return; const p = beginPass(enc, 'L1 fine-fine ghost'); p.setPipeline(skipGroup('ghost-noop') ? noopPLs.interpDenseFF : interpFFPL); p.setBindGroup(0, interpFFBG_b); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end();
      } else {
        if (skipGroup('ghost')) return; const p = beginPass(enc, `L${level} fine-fine ghost`); p.setPipeline(skipGroup('ghost-noop') ? noopPLs.interpPoolFF : interpPoolParentFFPL); p.setBindGroup(0, pool.interpPoolParentFFBG_b); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
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

  // ── GPU pass timing ──────────────────────────────────────────────────────
  // Every compute pass in the macro-step goes through beginPass() so it can
  // be individually timed on demand. When `profiler` is null (the normal
  // case) this is exactly enc.beginComputePass() with no overhead; when
  // debugProfileMacroStep() sets it, each pass gets its own timestamp pair
  // and a label, which is what turns "AMR is slow" into "this pass is slow".
  //
  // Timestamps come from the pass DESCRIPTOR (timestampWrites), not from
  // encoder.writeTimestamp() -- that entry point was removed from WebGPU and
  // is why timing was disabled here in the first place.
  // ── Pass-group skipping, for cost attribution on coarse-timer devices ────
  // The per-pass timestamp profile is useless on hardware whose timestamp
  // counter is coarse: the target PowerVR part ticks at 65536 ns, so a
  // ~1125us macro-step is only ~17 ticks spread across 9-12 passes and every
  // per-pass reading lands in a 1-8 tick bucket. (Confirmed: every value it
  // reports is an exact integer multiple of 65536 ns.) Frame time, at ~1100
  // ticks, is quantised by ~0.1% and is fine.
  //
  // So attribute at FRAME scale instead: skip a group of passes, measure the
  // change in frame GPU time, and the difference is that group's real cost.
  // ?benchSkip=force,ghost etc. This is a MEASUREMENT MODE -- skipping
  // passes makes the physics wrong by construction. It exists to answer
  // "what does this group cost", nothing else.
  // Groups. The first set REMOVE a pass; the rest are instrument variants that
  // keep the pass but change what it does, so a share can be split further:
  //
  //   force, phy, step1, interp, avg, ghost   -- pass not encoded at all
  //   interp-noop, avg-noop, ghost-noop       -- pass encoded and dispatched at
  //                                              full width, returns immediately
  //   step1-ring                              -- fine step over tile interior
  //                                              only (proxy for FB 20 -> 16)
  //
  // Why the -noop variants exist: removing a pass removes its fixed per-pass
  // cost AND its work at the same time, so a plain skip cannot tell you which
  // one you are looking at. Measured on desktop, removing a coupling pass is
  // worth 15-19% of frame GPU time while running it as a no-op is worth 2-6%
  // -- i.e. the AMR coupling cost is work, not pass count, so fusing coupling
  // passes would not pay. See the NOOP override in shaders/amr_interp_*.wgsl.
  //
  // step1-ring exists because the two target devices should disagree about it:
  // it measured 0.2% on the desktop (correcting a ~10.5% estimate in
  // plans/perf-characterization.md) but the phone is bandwidth-bound and the
  // traffic model predicts ~10% there.
  // Every valid group name. Enumerated rather than free-form because an
  // unrecognised name is otherwise INVISIBLE: it just never matches a
  // skipGroup() call, the passes all run, and the configuration reports a ~0%
  // share that reads as a real measurement. That is unrecoverable on the phone,
  // which has no CDP endpoint and gets one sweep per session -- the same class
  // of silent-failure trap as a sweep interrupted by backgrounding, which this
  // file already refuses to report quietly.
  const BENCH_GROUPS = new Set([
    'force', 'phy', 'step1', 'interp', 'avg', 'ghost',
    'interp-noop', 'avg-noop', 'ghost-noop', 'step1-ring',
  ]);
  function validateSkipGroups(groups, where) {
    const bad = [...groups].filter(g => !BENCH_GROUPS.has(g));
    if (bad.length) throw new Error(`${where}: unknown benchSkip group(s) ${bad.join(', ')} -- known: ${[...BENCH_GROUPS].join(', ')}`);
  }
  const benchSkip = new Set((urlParams.get('benchSkip') || '').split(',').filter(Boolean));
  validateSkipGroups(benchSkip, '?benchSkip=');
  function skipGroup(g) { return benchSkip.has(g); }

  let profiler = null;
  function beginPass(enc, label) {
    if (profiler && profiler.next + 2 <= profiler.cap) {
      const i = profiler.next;
      profiler.next += 2;
      profiler.labels.push({ label, i });
      return enc.beginComputePass({
        timestampWrites: { querySet, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 },
      });
    }
    return enc.beginComputePass();
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
      const crit = beginPass(enc, 'criterion L0'); crit.setPipeline(criterionPL); crit.setBindGroup(0, criterionBG); crit.dispatchWorkgroups(WGX, WGY); crit.end();
      for (let m = 1; m < N_LEVELS - 1; m++) {
        const c = beginPass(enc, `criterion L${m}`); c.setPipeline(criterionPoolPLs[m]); c.setBindGroup(0, criterionPoolBGs[m]); c.dispatchWorkgroups(2, 2, pools[m].MAX_FINE_BLOCKS); c.end();
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
            const p = beginPass(enc, 'manage coarsen L1'); p.setPipeline(manageCoarsenPL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
          } else {
            const parentLevel = m - 1;
            const wg = Math.ceil(pools[m].MAX_FINE_BLOCKS / 64);
            const p = beginPass(enc, `manage coarsen L${m}`); p.setPipeline(managePoolCoarsenPLs[parentLevel]); p.setBindGroup(0, managePoolBGs[parentLevel]); p.dispatchWorkgroups(wg); p.end();
          }
        }
        for (let m = 1; m < N_LEVELS; m++) {
          if (m === 1) {
            const p = beginPass(enc, 'manage refine L1'); p.setPipeline(manageRefinePL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
          } else {
            const parentLevel = m - 1;
            const wg = Math.ceil(pools[parentLevel].MAX_FINE_BLOCKS / 64);
            const p = beginPass(enc, `manage refine L${m}`); p.setPipeline(managePoolRefinePLs[parentLevel]); p.setBindGroup(0, managePoolBGs[parentLevel]); p.dispatchWorkgroups(wg); p.end();
          }
        }
      }

      // One-time full-slot fill for everything newly activated this round,
      // every level (level 1: dense-parent init pipeline, unchanged;
      // level>=2: pool-parent init pipeline, reading readA since a
      // level's own buffer is always "current" at a macro-step boundary --
      // same invariant debugActivateBlock already relies on).
      const init = beginPass(enc, 'L1 init fill'); init.setPipeline(interpInitPL); init.setBindGroup(0, interpInitBG); init.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); init.end();
      for (let m = 2; m < N_LEVELS; m++) {
        const pool = pools[m];
        const p = beginPass(enc, `L${m} init fill`); p.setPipeline(interpPoolParentInitPL); p.setBindGroup(0, pool.interpPoolParentBG_readA); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
    }
    macroStepCounter++;

    // Force integration + body dynamics: coarse-only still (Milestone 8's
    // scope, not this one -- see amr_step1.wgsl's own header), dispatched
    // once per macro-step, outside the fluid recursion entirely (same as
    // AGAL's own S_ComputeForces* calls, handled alongside S_Advance, not
    // inside it).
    if (!skipGroup('force') && !skipGroup('force0')) { const frc = beginPass(enc, 'force L0'); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end(); }
    // Milestone 8: every level's own force contribution, all before `phy`
    // drains+resets the shared atomic forces[] buffer. Order among these
    // (and vs. frc above) doesn't matter -- each reads only its own
    // level's "current, pre-macro-step" state and independently atomicAdds
    // into forces[], the same commutativity argument as interp-vs-step at
    // the root of S_Advance.
    if (!skipGroup('force')) { const f1frc = beginPass(enc, 'force L1'); f1frc.setPipeline(force1PL); f1frc.setBindGroup(0, force1BG); f1frc.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1frc.end(); }
    for (let c = 2; c < N_LEVELS; c++) {
      const pool = pools[c];
      if (!skipGroup('force')) { const p = beginPass(enc, `force L${c}`); p.setPipeline(force1PoolPL); p.setBindGroup(0, pool.force1PoolBG); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end(); }
    }
    if (!skipGroup('phy')) { const phy = beginPass(enc, 'body dynamics'); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end(); }

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
    writeF(f_a, initF(), NCELLS);
    writeF(pools[1].finePoolF_a, initFPool(), MAX_FINE_BLOCKS * NCELLS1);
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
      writeF(pool.finePoolF_a, initFPool(pool.MAX_FINE_BLOCKS), pool.MAX_FINE_BLOCKS * NCELLS1);
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
    writeF(pools[1].finePoolF_a, marker, NPOOL);

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
    const result = readF(stagingFPool.getMappedRange(), NPOOL);
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
    const f = readF(stage.getMappedRange(), pool.MAX_FINE_BLOCKS * NCELLS1);
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
    writeF(f_a, f, NCELLS);
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
  //
  // Generalized to N_LEVELS>=2 (was hardcoded to compare level 1 against
  // level 2 only, silently ignoring level 3+). Border-aware, not whole-
  // tile-max: a first version of this compared each tile's DEEPEST
  // descendant ANYWHERE in its footprint against its neighbor's deepest
  // descendant anywhere in ITS footprint -- wrong, and produced false-
  // positive "violations" live (e.g. a level-1 tile A next to a level-1
  // tile B where B's level-3 descendant sat on B's FAR side, away from A
  // -- the actual A/B shared edge only ever touched B's level-2 quadrants,
  // which IS balanced; B's unrelated far-side depth doesn't matter to
  // that boundary). True 2:1 balance is a property of ADJACENT CELLS, not
  // adjacent TILES-as-a-whole: only check LEAF tiles (active, no children
  // -- a non-leaf tile's own boundary correctness is already checked one
  // level deeper, at its children's own same-level neighbor checks, so
  // checking it again here would be redundant AND use the wrong
  // granularity), and for each neighbor, walk toward the SHARED edge
  // specifically (borderMaxDepth), not the neighbor's tile as a whole.
  async function debugCheck21Balance() {
    const activeSets = {}, NBX_ = {}, NBY_ = {}, counts = {};
    for (let m = 1; m < N_LEVELS; m++) {
      const active = await debugListActiveBlocks(m);
      activeSets[m] = new Set(active.map(b => `${b.bx},${b.by}`));
      NBX_[m] = pools[m].NBX; NBY_[m] = pools[m].NBY;
      counts[m] = active.length;
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
    // The 2 (of 4) quadrant children that lie along a given edge of their
    // parent -- e.g. a parent's SOUTH edge is covered by its qy=1 children
    // (both qx). Recursing with the SAME edge picks the correct
    // ever-deeper sliver along that edge, not the tile's max depth.
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
        if (hasChild(m, bx, by)) continue; // not a leaf -- checked one level deeper instead
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
    return { ok: violations.length === 0, violations, counts };
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
    // This path bypasses frame() entirely, so it has to flush pending
    // parameter changes itself -- otherwise a headless driver that sets a
    // slider and then steps would silently run the old values.
    if (paramsDirty) {
      updateGPUParams();
      paramsDirty = false;
    }
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

  // Per-pass GPU timing for ONE macro-step. Runs the identical dispatch
  // sequence the live loop runs (dispatchMacroStep, not a reimplementation),
  // with `profiler` set so every beginPass() call gets its own timestamp
  // pair, then resolves and returns {label, ms} in dispatch order. This is
  // the measurement that makes dispatch tuning falsifiable: it attributes
  // frame time to individual passes rather than leaving it as one number.
  //
  // `reps` runs the macro-step several times and returns the MEDIAN per
  // label -- a single macro-step is short enough that one sample is mostly
  // scheduling noise. Passes that only appear on refine steps (criterion /
  // manage / init fill, every REFINE_EVERY steps) will be present in some
  // reps and absent in others; each label reports its own sample count.
  async function debugProfileMacroStep(reps = 8) {
    if (!hasTimestamp) {
      throw new Error('debugProfileMacroStep: adapter lacks the timestamp-query feature -- no GPU timing available on this device');
    }
    const readBuf = device.createBuffer({ size: QUERY_CAP * 8, usage: U.MAP_READ | U.COPY_DST });
    const byLabel = new Map();
    for (let r = 0; r < reps; r++) {
      profiler = { cap: QUERY_CAP, next: 2, labels: [] }; // 0/1 reserved for the frame span
      const enc = device.createCommandEncoder();
      dispatchMacroStep(enc);
      const used = profiler.next;
      enc.resolveQuerySet(querySet, 0, used, queryResolveBuffer, 0);
      enc.copyBufferToBuffer(queryResolveBuffer, 0, readBuf, 0, used * 8);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(readBuf.getMappedRange()).slice();
      readBuf.unmap();
      for (const { label, i } of profiler.labels) {
        const ms = Number(ts[i + 1] - ts[i]) / 1e6;
        // A disjoint or unresolved query reads back as 0 (or negative). Drop
        // it rather than recording a pass as free -- a zero here means "not
        // measured", which is a very different claim from "costs nothing".
        if (!Number.isFinite(ms) || ms <= 0) continue;
        if (!byLabel.has(label)) byLabel.set(label, []);
        byLabel.get(label).push(ms);
      }
      profiler = null;
    }
    readBuf.destroy();
    const out = [];
    let total = 0;
    for (const [label, xs] of byLabel) {
      xs.sort((a, b) => a - b);
      const med = xs[Math.floor(xs.length / 2)];
      out.push({ label, ms: med, samples: xs.length });
      total += med;
    }
    return { passes: out, totalMs: total, reps };
  }

  // ── Telemetry back channel (opt-in: ?telemetry=1) ────────────────────────
  // A device that is not the dev machine -- a phone on the LAN -- has no CDP
  // endpoint to attach to, so its performance is otherwise unobservable, and
  // "realtime on desktop AND mobile" is a stated goal of this project. With
  // ?telemetry=1 the page POSTs a periodic sample to the dev server's
  // /_telemetry endpoint (see https.py), which appends it to telemetry.log.
  //
  // Off unless explicitly requested, same-origin only, and local-only: the
  // dev server writes a plain file next to the page and forwards nothing.
  // Failures are swallowed -- a page serving from anywhere without the
  // endpoint (GitHub Pages, say) must not break because a beacon 404s.
  const TELEMETRY = urlParams.get('telemetry') === '1';
  const TELEMETRY_EVERY_MS = 5000;
  // ?profile=1 additionally attaches a per-pass GPU breakdown, so a device
  // that cannot be attached to over CDP (a phone) can still report WHERE its
  // frame time goes, not just how much of it there is. Rate-limited hard --
  // the profile serializes one macro-step per rep, so it is far more
  // disruptive than a plain sample.
  const TELEMETRY_PROFILE = urlParams.get('profile') === '1';
  const TELEMETRY_PROFILE_EVERY_MS = 20000;
  let telemetryLast = 0;
  // -Infinity so the FIRST sample carries a profile; the rate limit applies
  // only to subsequent ones. Waiting 20s for the first breakdown makes a
  // short phone session report nothing useful.
  let telemetryProfileLast = -Infinity;
  let telemetryInfo = null;
  async function telemetrySample(gpuMs, syncMs, stepNow) {
    if (!TELEMETRY) return;
    const now = performance.now();
    if (now - telemetryLast < TELEMETRY_EVERY_MS) return;
    telemetryLast = now;
    if (!telemetryInfo) {
      let adapterInfo = {};
      try {
        const ai = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
        adapterInfo = { vendor: ai.vendor, architecture: ai.architecture, device: ai.device, description: ai.description };
      } catch { /* adapter info is optional and gated on some browsers */ }
      telemetryInfo = {
        page: 'index-amr.html',
        ua: navigator.userAgent,
        dpr: window.devicePixelRatio,
        screen: `${window.screen.width}x${window.screen.height}`,
        adapter: adapterInfo,
        hasTimestamp,
        config: {
          res: resLog2, W, levels: N_LEVELS, blockage: BLOCKAGE, aspect: ASPECT,
          re: RE, tau: TAU, maxFineBlocks: MAX_FINE_BLOCKS,
          forceRefineMargin: FORCE_REFINE_MARGIN, refineThresh: REFINE_THRESH,
          stepsPerFrame: STEPS_PER_FRAME,
        },
      };
    }
    // Active fine-block counts per level, read from the CPU-visible free
    // count rather than a GPU readback -- a readback here would stall the
    // very frame loop being measured.
    const body = {
      ...telemetryInfo,
      t: new Date().toISOString(),
      step: stepNow,
      gpuMs: Number.isFinite(gpuMs) ? +gpuMs.toFixed(3) : null,
      syncMs: Number.isFinite(syncMs) ? +syncMs.toFixed(3) : null,
      // L0-cell throughput only -- see the mlups comment in the frame loop.
      l0Mlups: gpuMs > 0 ? +((NCELLS * STEPS_PER_FRAME) / (gpuMs * 1e3)).toFixed(1) : null,
      watch: {
        frames: readbackWatch.n,
        stepBack: readbackWatch.stepBack,
        worstStepBack: readbackWatch.worstStepBack,
        posJump: readbackWatch.posJump,
        fieldRepeat: readbackWatch.fieldRepeat,
        offReversals: readbackWatch.offReversals,
        worstOffReversal: readbackWatch.worstOffReversal,
        offMaxStep: readbackWatch.offMaxStep,
        offBackFrames: readbackWatch.offBackFrames,
        offTrace: readbackWatch.offTrace,
        samples: readbackWatch.samples,
        diverged: readbackWatch.diverged,
        divergedAtStep: readbackWatch.divergedAtStep,
        runUp: readbackWatch.history,
      },
    };
    if (BENCH && !benchDone && !benchRunning && step >= BENCH_WARM) {
      benchRunning = true;
      try {
        const benchStartStep = step;
        body.bench = await runBenchSweep();
        body.bench.startedAtStep = benchStartStep;
        body.bench.endedAtStep = step;
        benchDone = true;
        statusEl.textContent = '[AMR-dev] benchmark sweep complete -- results sent';
      } catch (e) { body.benchError = String(e && e.message || e); benchDone = true; }
      finally { benchRunning = false; }
    }
    if (TELEMETRY_PROFILE && hasTimestamp && now - telemetryProfileLast > TELEMETRY_PROFILE_EVERY_MS) {
      telemetryProfileLast = now;
      // Pause stepping across the profile. Timestamps taken while the frame
      // loop is still submitting come back disjoint (several passes read
      // exactly 0.0000), because the profiler's own submissions interleave
      // with the live loop's. Restored in the finally below.
      const wasLive = liveMode;
      try {
        liveMode = false;
        await device.queue.onSubmittedWorkDone();
        const p = await debugProfileMacroStep(6);
        body.profile = { totalMs: +p.totalMs.toFixed(4), reps: p.reps,
          passes: p.passes.map(x => ({ label: x.label, ms: +x.ms.toFixed(4), n: x.samples })) };
        body.activeByLevel = {};
        for (let m = 1; m < N_LEVELS; m++) {
          try { body.activeByLevel[m] = (await debugListActiveBlocks(m)).length; } catch { /* best-effort */ }
        }
      } catch (e) {
        body.profileError = String(e && e.message || e);
      } finally {
        liveMode = wasLive;
      }
    }
    try {
      await fetch('/_telemetry', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body), keepalive: true,
      });
    } catch { /* no endpoint (static hosting) -- telemetry is best-effort */ }
  }

  // ── On-device benchmark sweep (?bench=1) ─────────────────────────────────
  // Frame-scale differential attribution, which is the only kind that works
  // on a coarse-timestamp device (see the benchSkip comment above). Runs a
  // sequence of pass-skip configurations, measuring median frame GPU time
  // for each, then POSTs one summary. Difference from the 'none' baseline is
  // that group's cost.
  //
  // Refinement is FROZEN for the duration (setAutoRefine(false)) so every
  // configuration sees the same block topology -- otherwise skipping the
  // criterion pass would change the active block count and the comparison
  // would be measuring two different simulations.
  //
  // The physics is deliberately wrong while this runs. It is a stopwatch,
  // not a simulation.
  const BENCH = urlParams.get('bench') === '1';
  // Sweep does not start until the flow and the refinement have developed.
  // The sweep used to begin on the first telemetry tick, ~5s after load --
  // about 1350 steps on the phone, where measured L1 demand is still climbing
  // and does not peak until 6k-12k. That measured an unrepresentative
  // topology and every published phone attribution inherited it.
  const BENCH_WARM = urlParams.has('benchWarm') ? parseInt(urlParams.get('benchWarm')) : 12000;
  const BENCH_ROUNDS = urlParams.has('benchRounds') ? parseInt(urlParams.get('benchRounds')) : 3;
  // Target wall-clock per timed run. Validated on the desktop at 2500ms, which
  // reproduced tools/bench-amr.js's headline (interp+avg+ghost 44.7% vs 44.1%)
  // at 4.5% spread but left the small instrument rows (-noop, step1-ring, all
  // 1-5% effects) at 13-20% spread and occasionally negative. 4000ms buys that
  // resolution back and still fits: switching from live-rAF sampling to timed
  // synchronous runs cut the per-configuration cost from a fixed 5s, so a
  // 13-config sweep at 3 rounds is ~3.5 min plus warm-up rather than longer.
  const BENCH_MEASURE_MS = urlParams.has('benchMeasureMs') ? parseInt(urlParams.get('benchMeasureMs')) : 4000;
  // 13 configurations at BENCH_ROUNDS=3 is (3+1)*13*5s = ~4.3 min of sweep on
  // top of the warm-up, so budget ~5 min of foregrounded, screen-on device.
  // The -noop and step1-ring entries are the instrument variants documented
  // at benchSkip above; they are here rather than desktop-only because the
  // phone has no CDP endpoint, so tools/bench-amr.js cannot reach it and this
  // sweep is the only way to ask that device the same questions.
  // ?benchConfigs=none,interp,avg,ghost,interp+avg+ghost trims the list. Worth
  // using on a device you have to hold in your hand: every configuration costs
  // (BENCH_ROUNDS+1) * BENCH_MEASURE_MS, and a shorter sweep also spends less
  // of itself inside this device's own thermal ramp. 'none' is always kept --
  // every share is relative to it.
  const BENCH_CONFIGS_DEFAULT = ['none', 'force', 'phy', 'force+phy', 'interp', 'avg', 'ghost', 'step1', 'interp+avg+ghost',
                                 'interp-noop', 'avg-noop', 'ghost-noop', 'step1-ring'];
  const BENCH_CONFIGS = urlParams.has('benchConfigs')
    // URLSearchParams decodes '+' as a space, and '+' is this list's own
    // combine operator, so an unencoded ?benchConfigs=interp+avg+ghost arrives
    // as 'interp avg ghost'. Accept both rather than rejecting a URL that a
    // reader would swear is correct -- this gets typed by hand on a phone.
    ? [...new Set(['none', ...urlParams.get('benchConfigs').split(',')
        .map(c => c.trim().replace(/\s+/g, '+')).filter(Boolean)])]
    : BENCH_CONFIGS_DEFAULT;
  async function runBenchSweep() {
    const wasAuto = autoRefine;
    await setAutoRefine(false);
    const activeByLevel = {};
    for (let m = 1; m < N_LEVELS; m++) {
      try { activeByLevel[m] = (await debugListActiveBlocks(m)).length; } catch { /* best-effort */ }
    }
    // ROUND-ROBIN over several rounds with the order rotated, not each
    // configuration measured to completion in turn. This device throttles
    // 24-57% within a session (see plans/perf-characterization.md), so a
    // sequential sweep measures the first configuration coldest and the last
    // hottest -- which lands entirely on the 'none' baseline, understating
    // every share. Interleaving spreads the ramp evenly instead. Same
    // reasoning, and the same fix, as tools/bench-amr.js --skip.
    // Backgrounding still invalidates the sweep, even now that timing no longer
    // depends on requestAnimationFrame: a hidden tab gets its GPU work
    // deprioritised and its timers throttled, so whichever configurations were
    // current while it was hidden are timed against a different machine. A
    // phone run was already lost to this once, silently. Record it and refuse
    // to report a sweep that was interrupted.
    let benchHidden = document.visibilityState === 'hidden';
    const onVis = () => { if (document.visibilityState === 'hidden') benchHidden = true; };
    document.addEventListener('visibilitychange', onVis);

    // Time a fixed number of macro-steps with the frame loop STOPPED, the same
    // way tools/bench-amr.js does -- NOT by sampling per-frame GPU timestamps
    // from the live rAF loop, which is what this used to do.
    //
    // The old method could not be trusted, and was measured failing: a desktop
    // run of it reported spreads of 25-88% and NEGATIVE shares down to -73%
    // (skipping work cannot make a frame slower), against a ground truth from
    // tools/bench-amr.js of interp+avg+ghost = 44%. The reason is structural
    // rather than statistical -- on a device that finishes its frame well
    // inside the vsync interval the GPU sits idle most of each frame and
    // clocks down, so per-frame timestamps scatter no matter how many are
    // averaged. Stopping the frame loop and timing a synchronous run removes
    // vsync, the compositor and the readback pipeline in one move; the same
    // instrument measured 2.5-5.6% spread that way.
    //
    // liveMode is restored at the end of the sweep (debugStepSync clears it).
    const measure = async () => {
      const t0 = performance.now();
      await debugStepSync(stepsPerMeasure);
      return performance.now() - t0;
    };
    const applySkip = (cfg) => {
      benchSkip.clear();
      if (cfg !== 'none') for (const g of cfg.split('+')) benchSkip.add(g);
    };
    // Fail before the sweep, not silently during it: a mistyped entry in
    // BENCH_CONFIGS would otherwise cost a whole ~5 min device session and
    // report a plausible-looking 0% share for that row.
    for (const cfg of BENCH_CONFIGS) {
      if (cfg !== 'none') validateSkipGroups(cfg.split('+'), `BENCH_CONFIGS entry "${cfg}"`);
    }
    // Discard a whole settling round -- one discarded run was measurably not
    // enough on the desktop (44-85% spread on the early rows).
    const totalSteps = (BENCH_ROUNDS + 1) * BENCH_CONFIGS.length;
    let doneSteps = 0;
    const progress = (label) => {
      const pct = Math.round((doneSteps / totalSteps) * 100);
      statusEl.textContent = `[AMR-dev] benchmark ${pct}% -- ${label} (do not switch away)`;
    };
    // How many macro-steps is ~BENCH_MEASURE_MS on THIS device? The two target
    // devices differ by ~40x in frame time, so a fixed step count would be
    // either far too short to time on the desktop or minutes per configuration
    // on the phone. Calibrated from a short probe instead, and reported in the
    // payload so a reader knows what the medians are medians OF.
    const probeSteps = 4 * STEPS_PER_FRAME;
    const probeT0 = performance.now();
    await debugStepSync(probeSteps);
    const msPerStep = (performance.now() - probeT0) / probeSteps;
    let stepsPerMeasure = Math.round(BENCH_MEASURE_MS / msPerStep / STEPS_PER_FRAME) * STEPS_PER_FRAME;
    stepsPerMeasure = Math.max(STEPS_PER_FRAME, Math.min(stepsPerMeasure, 200 * STEPS_PER_FRAME));

    progress('settling');
    for (const cfg of BENCH_CONFIGS) { applySkip(cfg); await measure(); doneSteps++; progress('settling ' + cfg); }

    const samples = new Map(BENCH_CONFIGS.map(c => [c, []]));
    for (let r = 0; r < BENCH_ROUNDS; r++) {
      for (let i = 0; i < BENCH_CONFIGS.length; i++) {
        const cfg = BENCH_CONFIGS[(i + r) % BENCH_CONFIGS.length];
        applySkip(cfg);
        const m = await measure();
        if (m != null) samples.get(cfg).push(m);
        doneSteps++;
        progress(`round ${r + 1}/${BENCH_ROUNDS}, ${cfg}`);
      }
    }
    const results = BENCH_CONFIGS.map(cfg => {
      const xs = samples.get(cfg).slice().sort((a, b) => a - b);
      return {
        cfg, n: xs.length,
        // Wall-clock ms for stepsPerMeasure macro-steps, not per-frame GPU ms.
        medianMs: xs.length ? +xs[Math.floor(xs.length / 2)].toFixed(1) : null,
        spreadPct: xs.length > 1 ? +(((xs[xs.length - 1] - xs[0]) / xs[Math.floor(xs.length / 2)]) * 100).toFixed(1) : null,
      };
    });
    benchSkip.clear();
    document.removeEventListener('visibilitychange', onVis);
    if (wasAuto) await setAutoRefine(true);
    liveMode = true; // debugStepSync cleared it; the page must resume after the sweep
    const base = results.find(r => r.cfg === 'none');
    for (const r of results) {
      r.deltaMs = (base && base.medianMs != null && r.medianMs != null)
        ? +(base.medianMs - r.medianMs).toFixed(1) : null;
      r.sharePct = (base && base.medianMs) ? +((r.deltaMs / base.medianMs) * 100).toFixed(1) : null;
    }
    return {
      activeByLevel, results,
      // What the medians are medians of, so a reader can sanity-check them.
      method: 'debugStepSync', stepsPerMeasure, msPerStepProbe: +msPerStep.toFixed(4),
      // Reported, not silently dropped: a caller who sees interrupted=true
      // should discard the numbers rather than wonder why they look odd.
      interrupted: benchHidden,
      // VALIDATED 2026-09-07 against tools/bench-amr.js on the same machine and
      // config, with every competing GPU client stopped. Large groups agree:
      //
      //   group             this sweep   bench-amr
      //   interp            13.8%        15.6-17.3%
      //   avg               12.2%        12.1-15.9%
      //   ghost             13.3%        13.7-18.5%
      //   step1             32.7%        27.5-31.7%
      //   interp+avg+ghost  40.5%        43.9-49.1%
      //
      // The small rows do NOT resolve, on any run: interp-noop/avg-noop/
      // ghost-noop/step1-ring are 0.2-5% effects by bench-amr and came back as
      // 7.9%, -3.6%, -0.9% and -12.2% here. The baseline itself carries ~15%
      // spread because the card keeps falling through the sweep while
      // refinement is frozen, so the workload drifts under every row equally.
      // Read anything under ~10% as "below the floor", not as a measurement --
      // and a NEGATIVE share means exactly that, since skipping work cannot
      // make a run slower. Use tools/bench-amr.js for the small effects on any
      // device that has a CDP endpoint; this sweep exists for the one that
      // does not.
      noiseFloorPct: 10,
    };
  }
  const readbackWatch = { lastStep: null, lastY: null, n: 0, stepBack: 0, worstStepBack: 0,
                          posJump: 0, fieldRepeat: 0, digests: [], samples: [],
                          lastOffX: null, lastOffY: null, offDirX: 0, offDirY: 0,
                          offReversals: 0, worstOffReversal: 0, offMaxStep: 0, offBackFrames: 0, offTrace: [], ring: [], diverged: false, divergedAtStep: null, history: null };

  let benchRunning = false;
  let benchDone = false;

  window.__AMR = {
    runBenchSweep,
    hasTimestamp: () => hasTimestamp,
    debugProfileMacroStep,
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
    getF16: () => F16,
    // Set the pass-skip set AFTER warm-up, which is the only way a skip A/B
    // is valid: passing ?benchSkip= in the URL means the warm-up itself runs
    // with the modified physics, so the card follows a different trajectory
    // and refinement settles on a different topology. Measured -- warming up
    // with force skipped gave 73 active L1 blocks against 123 for the
    // baseline, so the two runs were not doing comparable work at all. See
    // tools/bench-amr.js --skip.
    setBenchSkip: (groups) => {
      const next = (groups || []).filter(Boolean);
      validateSkipGroups(next, 'setBenchSkip');
      benchSkip.clear();
      for (const g of next) benchSkip.add(g);
      return [...benchSkip];
    },
    getLevelPoolSizes,
    tauAtLevel,
  };

  async function frame() {
    try {
      if (deviceLost) return; // stop the rAF chain; every submit would be a no-op
      // Flush pending slider changes BEFORE the liveMode early-return.
      // With this after it, a parameter changed while the sim was paused was
      // silently dropped, and debugStepSync (which never goes through this
      // function) would then advance the sim with the OLD values while the
      // control panel showed the new ones. Found while testing a mid-run
      // Blockage drag: A stayed at 38.8 for 16000 steps after the slider and
      // its readout had both moved to 20.3.
      if (paramsDirty) {
        updateGPUParams();
        paramsDirty = false;
      }
      if (!liveMode) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      const stage = stages[currentStageIdx];
      // Backpressure: if the oldest stage is still in flight, we must wait.
      if (stage.inFlight) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      device.pushErrorScope('validation');
      const enc = device.createCommandEncoder();

      // Whole-frame GPU span. A compute pass may carry timestampWrites
      // without dispatching anything, so an empty pass at each end brackets
      // the frame's real work without touching dispatchMacroStep.
      if (hasTimestamp) {
        const t0 = enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } });
        t0.end();
      }

      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;

      if (hasTimestamp) {
        const t1 = enc.beginComputePass({ timestampWrites: { querySet, endOfPassWriteIndex: 1 } });
        t1.end();
        enc.resolveQuerySet(querySet, 0, 2, queryResolveBuffer, 0);
        enc.copyBufferToBuffer(queryResolveBuffer, 0, stage.query, 0, 16);
      }

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

      // Only run when telemetry is on. It exists to answer a diagnostic
      // question, and a normal run should not pay for an instrument -- least
      // of all one whose cost cannot be measured on the machine it is
      // suspected on.
      if (TELEMETRY && !skipGroup('digest')) {
        const dg = beginPass(enc, 'field digest');
        dg.setPipeline(digestPL); dg.setBindGroup(0, digestBG); dg.dispatchWorkgroups(1); dg.end();
      }
      enc.copyBufferToBuffer(cardStateBuf, 0, stage.card, 0, 104);
      enc.copyBufferToBuffer(digestBuf, 0, stage.card, 104, 16);

      const tSubmit = performance.now();
      device.queue.submit([enc.finish()]);
      device.popErrorScope().then(err => { if (err) handleErr(err); });

      stage.inFlight = true;
      stage.step = step;

      // ── Ordering watchdog ────────────────────────────────────────────
      // Reported symptom: on a slow device the display appears to jump
      // BACKWARD a few frames now and then. The step counter itself is
      // monotonic (verified in telemetry), so any regression has to be in
      // what gets displayed -- which is read back per stage, and three
      // stages are in flight at once. If two readbacks are ever processed
      // out of order, the older one overwrites the newer one's status and
      // trajectory row, and the display goes back in time.
      //
      // Desktop cannot reproduce it (monotonic over hundreds of updates even
      // at sync/gpu = 3.3), so this records the evidence on whatever device
      // actually shows it, rather than guessing from here. Cheap enough to
      // leave on: two comparisons per frame.
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

        // Out-of-order / regression detection on the ACTUAL readback
        // sequence, not the 250ms-throttled status line.
        if (readbackWatch.lastStep !== null) {
          if (st.step < readbackWatch.lastStep) {
            readbackWatch.stepBack++;
            readbackWatch.worstStepBack = Math.max(readbackWatch.worstStepBack, readbackWatch.lastStep - st.step);
            if (readbackWatch.samples.length < 12) {
              readbackWatch.samples.push({ kind: 'step', from: readbackWatch.lastStep, to: st.step });
            }
          }
          // y_total/x_total are accumulated displacement: a physical
          // quantity that cannot jump discontinuously in one frame. A large
          // jump means we are looking at a stale readback, not new physics.
          //
          // The finiteness test is FIRST and separate on purpose. Every
          // comparison with NaN is false, so `dy > 50` silently passes a
          // solver that has already diverged -- which is exactly what
          // happened: a phone run blew up while this watchdog reported
          // frames=1127, stepBack=0, posJump=0. A NaN check cannot be
          // expressed as a magnitude threshold.
          if (!Number.isFinite(d[20]) || !Number.isFinite(d[4]) || !Number.isFinite(d[7])) {
            if (!readbackWatch.diverged) {
              readbackWatch.diverged = true;
              readbackWatch.divergedAtStep = st.step;
              // The run-up is the diagnostic, not the NaN itself.
              readbackWatch.history = readbackWatch.ring.slice();
            }
          } else {
            const dy = Math.abs(d[20] - readbackWatch.lastY);
            if (readbackWatch.lastY !== null && dy > 50) {
              readbackWatch.posJump++;
              if (readbackWatch.samples.length < 12) {
                readbackWatch.samples.push({ kind: 'y', from: +readbackWatch.lastY.toFixed(2), to: +d[20].toFixed(2), atStep: st.step });
              }
            }
          }
          // MOVING-WINDOW OFFSET TRACKING.
          // The render translates the ENTIRE field by state.off_x/off_y
          // (amr_render.wgsl's get_ux/get_uy and its bufX/bufY), while the
          // card is drawn at cx/cy which absorb the sub-cell remainder. So a
          // twitching offset moves the whole VIEW back and forth while the
          // card's orientation and the field's content stay unperturbed --
          // which is exactly the reported symptom, and is invisible to every
          // other signal here: y_total/x_total are accumulated displacement
          // and stay smooth across an offset reversal by construction
          // (amr_physics.wgsl splits position into floor -> off and
          // fraction -> cx).
          //
          // off wraps modulo W/H, so take the SIGNED SHORTEST delta -- a
          // 255 -> 0 step is +1, not -255.
          const wrapDelta = (cur, prev, n) => {
            let dd = (cur - prev) % n;
            if (dd > n / 2) dd -= n;
            if (dd < -n / 2) dd += n;
            return dd;
          };
          if (readbackWatch.lastOffX !== null) {
            const dxo = wrapDelta(d[22], readbackWatch.lastOffX, W);
            const dyo = wrapDelta(d[23], readbackWatch.lastOffY, H);
            // A reversal is a sign flip against the recent trend, not merely
            // a negative step: the card genuinely flutters, so sustained
            // motion in either direction is physical and expected.
            if (dxo !== 0) {
              if (readbackWatch.offDirX !== 0 && Math.sign(dxo) !== readbackWatch.offDirX) {
                readbackWatch.offReversals++;
                readbackWatch.worstOffReversal = Math.max(readbackWatch.worstOffReversal, Math.abs(dxo));
                if (readbackWatch.samples.length < 12) {
                  readbackWatch.samples.push({ kind: 'offX', atStep: st.step, delta: dxo, prevDir: readbackWatch.offDirX });
                }
              }
              readbackWatch.offDirX = Math.sign(dxo);
            }
            if (dyo !== 0) {
              if (readbackWatch.offDirY !== 0 && Math.sign(dyo) !== readbackWatch.offDirY) {
                readbackWatch.offReversals++;
                readbackWatch.worstOffReversal = Math.max(readbackWatch.worstOffReversal, Math.abs(dyo));
              }
              readbackWatch.offDirY = Math.sign(dyo);
            }
            readbackWatch.offMaxStep = Math.max(readbackWatch.offMaxStep, Math.abs(dxo), Math.abs(dyo));
            // Counting REVERSALS alone was the wrong measure: between two
            // reversals the offset can travel one way for dozens of frames,
            // and the viewer sees the scene translate backward on EVERY
            // frame with a negative delta, not just on the frame the
            // direction flips. Count those directly, and keep a short trace
            // so an observed twitch can be matched against what the window
            // actually did around that moment.
            if (dxo < 0 || dyo < 0) readbackWatch.offBackFrames++;
            readbackWatch.offTrace.push({ s: st.step, dx: dxo, dy: dyo });
            if (readbackWatch.offTrace.length > 32) readbackWatch.offTrace.shift();
          }
          readbackWatch.lastOffX = d[22];
          readbackWatch.lastOffY = d[23];

          // Rolling run-up buffer, kept regardless, so a divergence report
          // carries the frames BEFORE it rather than just the moment of.
          readbackWatch.ring.push({
            s: st.step, y: +Number(d[20]).toFixed(2), vy: +Number(d[4]).toFixed(5),
            om: +Number(d[5]).toFixed(6), fy: +Number(d[7]).toPrecision(4), th: +Number(d[2]).toFixed(3),
          });
          if (readbackWatch.ring.length > 24) readbackWatch.ring.shift();
        }
        // Field-repeat detection. `dig` fingerprints the whole L0 velocity
        // field; ordinary dynamics never reproduces an earlier frame's field
        // exactly, so an exact match against a recent frame means the
        // display went back in time rather than forward.
        // `d` already maps the WHOLE staging buffer; a second getMappedRange
        // for the digest would overlap it and throw. CardState occupies
        // floats 0..25 (104 bytes), the digest floats 26..29.
        const key = `${d[26]}|${d[27]}|${d[28]}`;
        const prevIdx = readbackWatch.digests.indexOf(key);
        if (prevIdx !== -1) {
          readbackWatch.fieldRepeat++;
          if (readbackWatch.samples.length < 12) {
            readbackWatch.samples.push({
              kind: 'fieldRepeat', atStep: st.step,
              framesBack: readbackWatch.digests.length - prevIdx,
            });
          }
        }
        readbackWatch.digests.push(key);
        if (readbackWatch.digests.length > 16) readbackWatch.digests.shift();

        readbackWatch.lastStep = st.step;
        readbackWatch.lastY = d[20];
        readbackWatch.n++;

        if (st.step < 100000) {
          trajectory.push([st.step, d[0], d[20], d[21], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]);
        }

        if (performance.now() - lastT > 250) {
          // L0 cells only -- it deliberately ignores every fine level, so it
          // is a coarse-grid-throughput figure, NOT total work done, and is
          // not comparable across level counts. tools/bench-amr.js computes
          // the honest cell-updates/s using live per-level active counts.
          const mlups = (NCELLS * STEPS_PER_FRAME) / (gpuTime * 1e3);
          mlupsEl.textContent = mlups.toFixed(1);
          gpuMsEl.textContent = gpuTime.toFixed(2);
          syncMsEl.textContent = (performance.now() - tSubmit).toFixed(2);
          telemetrySample(gpuTime, performance.now() - tSubmit, st.step);
          // Not while a benchmark sweep owns the status line: this runs on
          // every readback and silently overwrote the sweep's own progress
          // messages within a frame, so "benchmark round 2/3" was never
          // actually visible to anyone asked to watch for it.
          if (!benchRunning) {
            statusEl.textContent = `[AMR-dev] step ${st.step}  y=${d[20].toFixed(1)}  x=${d[21].toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`;
          }
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
