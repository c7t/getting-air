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

import { reportFatal, refuseConfig, setStatus, reportNoWebGPU, reportNoAdapter } from './error-overlay.mjs';
import { installVortControls } from './vort-controls.mjs';
import { createTrail } from './trajectory-trail.mjs';
import { createTotalUnwrapper } from './card-total.mjs';
import { createSimPacer, parseSimRate, DEFAULT_TU_PER_SEC } from './sim-rate.mjs';
import { installChromeToggle } from './ui-chrome.mjs';
import { loadShader } from './shader-loader.mjs';
import {
  deriveCardParams, parseCardParams, parseResLog2, reynoldsFromTau,
  AMR_DEFAULT_RES_LOG2, AMR_DEFAULT_LEVELS,
  tauAtLevel as tauAtLevelOf,
} from './card-params.mjs';
import { packF, unpackF, fWords } from './f-pack.mjs';
import { check21BalanceOnGPU, allocLevelPool, checkRootPoolIdentity, readConservedTotals, readPoolIndirection as readPoolIndirectionOn, listActiveBlocks, readCardState, checkGeometryCoverageOnGPU, makeRefusalWatch , checkRefinementClosureOnGPU , makeCascadePipelines, encodeCascade, cascadeRoundTrip, makeCascadeSeeds, checkSlotQuadrantsOnGPU, makeRenderBindGroup, renderPoolLevels, MAX_RENDER_POOL_LEVELS, makeAMRLayouts, makeCouplingPipelines, makeLevelBindGroups, makeManageBindGroups, makeScheduler, makeRefineRound, readRootFlags, allocRootPool, makeRootPool } from './amr2d-gpu.mjs';
import { poolSlotsFor, tauChainSingularity, tauSingularityMessage, rootPoolSpec, rootCellToDense, rootCellIndex } from './amr2d.mjs';
import { EX, EY, WT } from './lattice-2d.mjs';
import { makeCanvasFit } from './canvas-fit.mjs';

const canvas   = document.getElementById('c');
let deviceLost = false;
const statusEl = document.getElementById('status');

// The page ships with its chrome collapsed (index-amr.html's `class="ui-hidden"`);
// this is the single button that brings it back. Wired HERE, at module top
// level, rather than inside init() alongside the other controls: if WebGPU
// setup throws, init()'s remaining statements never run, and a toggle wired
// there would strand the page with no way to expand the UI. It needs nothing
// from the GPU. (error-overlay.mjs also force-reveals on any fatal, so the
// two paths are independent -- neither relies on the other having run.)
installChromeToggle(document.getElementById('ui-toggle'));

const urlParams = new URLSearchParams(window.location.search);

// THE DIFFUSE BAND'S WIDTH, in units of a level's own cell size:
// epsilon = K_EPS * dx_level (plans/2D-backport.md B7). Threaded into every
// shader that evaluates chi -- step, force and render, at every level -- so
// the band can be swept without touching a literal in nine files.
//
// Default 1.5 is the value every one of those sites already hardcoded, so
// this build is byte-identical to the previous one. ?kEps=0.75 halves it.
//
// It is a BAND ladder, not a resolution ladder, that settles the standing Cd
// red cells: CLAUDE.md diagnoses them as diffuse-interface width (the
// effective body radius exceeds the nominal one, so Cd converges from ABOVE),
// and a resolution ladder moves the band and everything else at once.
//
// THE RENDER PASS GETS IT TOO. Its chi is what draws the body outline, so a
// ladder run whose picture still showed the 1.5 band would be quietly
// misleading about the thing being swept.
const K_EPS = urlParams.has('kEps') ? parseFloat(urlParams.get('kEps')) : 1.5;
if (!(K_EPS > 0)) throw new Error(`?kEps=${urlParams.get('kEps')} must be > 0`);
// Default resLog2=8 (W=256) with levels=3: two octaves of refinement give
// the "lower far-field resolution, the fine levels recover the body's
// resolution" AMR win via the general BLOCKAGE/ASPECT/RE mechanism below
// (see the comment above `let BLOCKAGE`), generalizing what used to be a
// hardcoded A=32,B=4 "half of main.js's dense reference" special case. At
// the shipped BLOCKAGE=8 the card is A=16,B=2 here, A=32,B=4 at L1 --
// Pesavento & Wang's Fig. 2 ellipse in lattice units -- and A=64,B=8 at L2.
// Two octaves rather than one because B=2 at L0 leaves the card's thin
// dimension badly under-resolved.
//
// NOTE main.js now defaults to this SAME resLog2=8, not to the res 10 this
// page's L2 is equivalent to -- that grid does not fit on the mobile target.
// So a bare index.html is this page's L0, not its matched dense comparison;
// that run is `index.html?res=10` (card-params.mjs's
// AMR_EQUIVALENT_DENSE_RES_LOG2, and see DENSE_DEFAULT_RES_LOG2's comment).
let resLog2 = parseResLog2(urlParams, AMR_DEFAULT_RES_LOG2);

// Fixed simulation RATE (sim-rate.mjs). STEPS_PER_FRAME below is now a
// CEILING, not a target: the pacer asks for however many steps a wall-clock
// interval is worth, so the physics runs at the same speed on every device
// that can keep up, and slower devices are clamped to exactly their old
// behaviour. ?simRate= is in a/u_t per wall-second, the paper's own time unit.
const SIM_RATE = parseSimRate(urlParams);


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

const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY; // coarse block grid

// ── Milestone 5 (plans/AMR-multilevel.md, plans/AMR-multilevel-M5.md):
// number of pool levels above L0. N_LEVELS=2 (default) is byte-identical
// to today's single-fine-level build (validated against a pre-M5
// baseline -- see the sub-plan). N_LEVELS>=3 allocates additional
// quadtree pool levels that no shader/dispatch reads yet (Milestone 6/7).
const N_LEVELS = urlParams.has('levels') ? parseInt(urlParams.get('levels')) : AMR_DEFAULT_LEVELS;
// refuseConfig, not throw: this runs at module scope, where
// init().catch(handleErr) can never see it -- see error-overlay.mjs.
if (N_LEVELS < 2) refuseConfig(statusEl, `?levels=${N_LEVELS} invalid -- must be >= 2 (L0 + at least one fine level)`);
// U6: and the renderer's own ceiling, refused BEFORE anything is allocated for
// a configuration that cannot be drawn. A cap that silently drops the finest
// level is the defect U6 exists to fix -- at ?levels=4 a level-3 tile used to
// be refined, solved, force-reduced and then drawn as its level-2 parent, and
// `tools/validate-render-levels.js` measured the picture as BYTE-IDENTICAL
// after perturbing that level's whole velocity pool.
if (N_LEVELS - 1 > MAX_RENDER_POOL_LEVELS) {
  refuseConfig(statusEl, `?levels=${N_LEVELS} needs ${N_LEVELS - 1} pool levels in the renderer, `
    + `which binds ${MAX_RENDER_POOL_LEVELS} (shaders/amr_render.wgsl's walk). Raising it means one `
    + `more binding pair there, in amr2d-gpu.mjs's RENDER_LEVEL_BINDINGS, and in every page's renBGL.`);
}

// ── POOL CAPACITY PER LEVEL, MEASURED (2026-09-15) ───────────────────────────
// Max live tiles over 40000 steps with every cap lifted, on this page's
// default card and regime:
//
//                 levels=3   levels=4   levels=5      role at that depth
//   level 1         231        213        261         parent (always)
//   level 2         400        492        516         finest at 3, parent at 4/5
//   level 3          --        656        628         finest at 4, parent at 5
//   level 4          --         --        904         finest at 5
//
// Demand is set by the level INDEX far more than by total depth, and it STEPS
// UP when a level acquires a child (level 2: 400 as the finest level, ~500
// once level 3 exists) -- that is 2:1 closure forcing a parent tile for
// everything refined below. Hence the two regimes; see amr2d.mjs's
// poolSlotsFor for the headroom convention.
//
// At 1.7x headroom this gives, per level:
//   levels=3 (shipped):  L1 444, L2 680           (was 384, 512)
//   levels=4:            L1 444, L2 878, L3 1116  (was 384, 512, 512)
//   levels=5:            L1 444, L2 878, L3 1068, L4 1537
//
// THE OLD FLAT 512 FOR EVERY LEVEL >= 2 IS WHY ?levels=4 REFUSED: level 3
// wants 656 and was given 512. Raising ?maxFineBlocks= could not fix it --
// that parameter sizes level 1 only, which is also what the refusal message
// used to recommend (fixed alongside this).
const POOL_PEAKS = {
  finest: { 2: 400, 3: 656, 4: 904 },
  parent: { 1: 261, 2: 516, 3: 628 },
};

const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks')
  ? parseInt(urlParams.get('maxFineBlocks'))
  : poolSlotsFor(POOL_PEAKS, 1, N_LEVELS);

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
//
// ── Block artifacts in the wake: what was measured, 2026-09-08 ────────────
// Reported symptom: block-shaped artifacts in the wake, clearest at low
// vortGamma, plus lumpiness induced on the shed vortices. Measured with
// tools/measure-refinement.js (which persists these scans -- run it before
// touching any number here). Findings, in the order they change what you
// would do:
//
// 1. The artifacts sit on COARSE/FINE INTERFACES. Confirmed by capturing one
//    frozen state twice, with and without the quadtree outline overlay: the
//    visible square edges land on tile boundaries.
//
// 2. The interfaces cut THROUGH vortices. At the shipped -9, 77-79 of the
//    ~100 vorticity-bearing blocks (|omega| >= 1e-3) are selected -- so ~25%
//    of the wake sits on L0, and the level boundary runs across the outer
//    envelope of each shed vortex rather than around it. -10 reaches 100% of
//    the wake for 10.1% of the domain, notably cheaper than the ~20% this
//    comment's own table estimated (that table was measured at blockage=3.3;
//    the shipped card is now blockage=8, i.e. smaller).
//
// 3. It is NOT hysteresis thrashing, so do not narrow the band looking for
//    it. Churn is high -- level 2 turns over ~23-31% of its tiles per
//    refinement round -- but FLAPPING (a block re-created within a few rounds
//    of being released) is 0-3% of births. The churn is the refined region
//    following a convecting wake, which is what it is supposed to do.
//
// 4. LEVEL 2 IS GEOMETRY-ONLY, and that is the near-wake artifact. Measured
//    at 26k steps: the block centres covered by L2 span phi -0.5 .. 7.9,
//    i.e. EXACTLY the forced halo (childLevel-2 margin 8) and nothing beyond
//    it. The vorticity criterion never extends level 2 into the wake at all,
//    so the L1/L2 boundary is pinned a few cells off the card's surface and
//    every shed vortex crosses it right at the trailing edge -- which is
//    where the artifact is reported, and it is NOT the domain-edge sponge
//    band (an earlier pass here blamed that; the band is one block wide at
//    the WINDOW edge and cannot explain a near-wake artifact).
//
//    ROOT CAUSE, FOUND AND FIXED: the level-2 blockCriterion was identically
//    ZERO, always. criterionPoolBGs bound `m === 1 ? velBuf : ...` as
//    amr_criterion_pool.wgsl's parent velocity, handing a dense,
//    cellIndex-addressed L0 buffer to a shader that addresses BY POOL SLOT --
//    the dense-parent pattern the neighbouring interp/step bind groups
//    legitimately use, copied to the one shader that has no dense-parent
//    variant. So amr_manage_pool.wgsl's refine() saw maxCrit ~= 0 for every
//    parent and the vorticity criterion could NEVER promote a tile to level 2.
//    Level 2 was 100% geometry-forced, which is exactly what the phi scan
//    showed before the cause was known. See the parentVel BUGFIX in the
//    criterion/manage bind-group loop below.
//
//    Measured, same build, A/B on that one line (40k steps):
//      level-2 criterion non-zero entries   0 / 4096  ->  1132 / 4096
//      level-2 criterion max                0         ->  0.0308
//      active level-2 tiles                 64        ->  244
//      block-centre phi covered by L2   -0.5 .. 7.9   ->  -0.3 .. 57.7
//    The last line is the artifact: level 2 used to stop at the forced halo
//    (margin 8), so the L1/L2 boundary sat a few cells off the body and every
//    shed vortex crossed it right at the trailing edge. It now follows the
//    wake out to ~7x that distance.
//
//    A SEPARATE, SMALLER blocker was also real and is fixed independently:
//    the refine cascade's missing growth half (?demandCascade). 19 tiles were
//    in that state. See amr_manage.wgsl's level2Wanted.
//
//    A caution for whoever measures here next: blockCriterionBuf had no
//    COPY_SRC, so debugReadBlockCriterion's copy was a validation error, the
//    command buffer was dropped, and EVERY level read back as all zeros --
//    which looks exactly like "the criterion pass never ran" and did produce
//    one wrong reading before a level-1 control (known-good, since L1
//    refinement demonstrably works) exposed it. COPY_SRC is now set.
//
// NOT CHANGED HERE, deliberately. Any of these is a physics change to the
// page that Pages serves, and this page cannot validate one: see
// tools/measure-refinement.js's own warning -- changing refinement changes
// the trajectory, and the falling card is chaotic, so two runs are at
// different points in the tumble within a few thousand steps and their wakes
// are not comparable by eye or by any field norm. That is
// plans/AMR-vs-dense-validation.md's Finding #3 again. Validate a retune on
// the cylinder harness (statistically steady, literature Cd/St) or with
// tools/validate-divergence.js (both legs seeded from one state), then bring
// it back here.
// Vorticity color tone curve (shaders/common_vortcolor.wgsl). Overridable
// per-run so the look can be dialed against a live sim rather than guessed
// at: ?vortScale= moves the curve's knee, ?vortGamma= shapes the low end.
// Parsed identically on both pages -- the two views are meant to be compared
// by eye, so a knob that existed on only one of them would defeat that.
const VORT_SCALE = parseFloat(urlParams.get('vortScale')) || 40.0;
const VORT_GAMMA = parseFloat(urlParams.get('vortGamma')) || 1.2;

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

// ── ?dcpre=1 -- the legacy PRE-collision Dupuis-Chopard transfer factor ──────
// The coarse<->fine transfers rescale the non-equilibrium part of f, and the
// factor that shipped here until plans/2D-backport.md B1 was the textbook
// PRE-collision one -- while every buffer this solver transfers holds f AFTER
// collision, because amr_step.wgsl is a fused pull-stream + collide. At
// tau = 0.8 the two differ in magnitude AND SIGN. amr2d.mjs's
// dcRescaleCoarseToFine carries the derivation and tools/test-amr2d.js gates
// both forms; shaders/common_interp.wgsl and common_average.wgsl are the two
// sites, one per direction.
//
// Both factors live in one build so the defect can be RE-MEASURED rather than
// reconstructed from a checkout -- the same reason ?ghostcopy= and ?f16= are
// still here. It also has no tau = 1 singularity, so it is the escape hatch
// the refusal names.
const DC_PRE = urlParams.has('dcpre') ? (parseInt(urlParams.get('dcpre')) || 0) : 0;

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

if (FORCE_REFINE_MARGIN >= SDF_FAR) {
  refuseConfig(statusEl, `?forceRefineMargin=${FORCE_REFINE_MARGIN} is at or above get_phi's SDF_FAR cutoff (${SDF_FAR}) -- ` +
    `beyond that the far-field early-out in shaders/common_geometry.wgsl returns a lower bound and isNearBody ` +
    `would silently under-refine. Raise SDF_FAR together with it if you really need a margin this large.`);
}

const FORCE_REFINE_LOOKAHEAD = urlParams.has('forceRefineLookahead') ? parseFloat(urlParams.get('forceRefineLookahead')) : REFINE_EVERY;

// REFINE-AHEAD, AND WHY IT IS A HARD REQUIREMENT RATHER THAN A FUDGE.
//
// The constraint geometry-forced refinement enforces is "no leaf within
// FORCE_REFINE_MARGIN of the body, at EVERY step". The manager only gets to
// decide every REFINE_EVERY macro-steps, so a decision at t0 must keep that
// true all the way to t0 + REFINE_EVERY -- which it does by testing the body
// at its current pose AND at t0 + FORCE_REFINE_LOOKAHEAD. A lookahead shorter
// than the decision interval therefore leaves the interval it does not cover
// unprotected, by construction and regardless of margin or speed.
//
// This is not hypothetical bookkeeping: it is the exact asymmetry that made
// B4-1's coverage check fire on the first MOVING body it saw (the checker
// applied the lookahead, so it demanded coverage of a window the last
// decision was never responsible for). A pinned body cannot show either
// side of it.
//
// The default already satisfies this -- FORCE_REFINE_LOOKAHEAD defaults to
// REFINE_EVERY -- so this refuses nothing that ships. It exists because the
// relationship is invisible at the two declarations, and someone tuning
// ?forceRefineLookahead down to save refinement churn would be removing a
// correctness guarantee while appearing to tune a performance knob.
if (FORCE_REFINE_LOOKAHEAD < REFINE_EVERY) {
  refuseConfig(statusEl, `?forceRefineLookahead=${FORCE_REFINE_LOOKAHEAD} is below ?refineEvery=${REFINE_EVERY} -- ` +
    `geometry-forced refinement would leave the last ${REFINE_EVERY - FORCE_REFINE_LOOKAHEAD} macro-step(s) of ` +
    `every refinement round unprotected, because the body can reach a block the previous decision did not cover. ` +
    `The lookahead must be at least the decision interval.`);
}

// A BODY WITH NO GEOMETRY FORCING IS NOT A CONFIGURATION, it is a silent
// downgrade to "refine wherever the vorticity happens to look interesting".
// The vorticity criterion is a LAGGING signal (see shaders/amr_manage.wgsl's
// header: it only fires once the coarse grid has already produced incorrect
// under-resolved vorticity at the surface), so with the margin at zero the
// body's own surface is refined late or not at all -- and since B4-3 only the
// finest level computes any force at all, a coarse patch at the surface now
// contributes NOTHING rather than something crude.
if (!(FORCE_REFINE_MARGIN > 0)) {
  refuseConfig(statusEl, `?forceRefineMargin=${FORCE_REFINE_MARGIN} disables geometry-forced refinement on a page ` +
    `that HAS a body. Only the finest level computes force (plans/2D-backport.md B4-3), so the body must be ` +
    `guaranteed to reach it -- which is exactly what this margin does.`);
}

// ?boxrefine=0 restores the pre-B4 geometry TEST: ONE get_phi at a block's
// CENTRE against FORCE_REFINE_MARGIN, truncated window conversion included.
// Default 1 asks about the whole block (a Lipschitz branch and bound --
// shaders/common_geometry.wgsl's nearBodyBox), which is the question the
// margin was always meant to answer; the centre sample under-reports by the
// block's own circumradius, 5.66 L0 cells at RB=8. Both paths live in one
// build so the difference can be measured: plans/2D-backport.md B4.
//
// THE TEST, NOT THE CONFIGURATION. B4-2 also retired the childLevel===2
// margin special case that existed to compensate for the centre test (see
// paramsForChildLevel), and this flag does not bring it back. The full
// pre-B4 configuration is `?boxrefine=0&forceRefineMargin2=8`; ?boxrefine=0
// alone isolates the predicate with the margin held at its honest scaled
// default, which is usually the comparison you actually want.
const BOX_REFINE = urlParams.has('boxrefine') ? (parseInt(urlParams.get('boxrefine')) ? 1 : 0) : 1;
// L0 window-space edge band (coarse cells) excluded from vorticity-driven
// refinement -- keeps fine blocks out of the ALBC sponge (amr_step.wgsl
// SPONGE_W=4). A fixed L0-window strip, so the same value applies at every
// refinement level (unlike FORCE_REFINE_MARGIN). Default 8; ?spongeExclude=0
// disables it.
const SPONGE_EXCLUDE_W = urlParams.has('spongeExclude') ? parseFloat(urlParams.get('spongeExclude')) : 8;

// ?detslots=1 -- deterministic pool slot handout (plans/uniform-levels.md D0).
//
// Default 0 is the shipped atomic free-list race and is byte-identical to not
// having this flag. 1 replaces BOTH managers' handout with a single-threaded
// serial pass, which makes block->slot assignment a deterministic function of
// the run from resetSim() onward.
//
// IT EXISTS TO ANSWER ONE QUESTION, not to be fast: is slot assignment the
// ONLY live source of run-to-run nondeterminism in this solver? If it is, two
// runs of the same build under ?detslots=1 are BIT-IDENTICAL, and the whole
// attractor apparatus in CLAUDE.md -- "the repeat must be on both sides", the
// 0.013 spread under unknown load, "match the baseline in TWO different
// modes" -- is downstream of one atomicSub and can be engineered away rather
// than worked around. If they are NOT identical there is a second source, and
// finding that out costs a day rather than a refactor.
//
// See shaders/amr_manage.wgsl's DET_SLOTS for why the serial loop is the right
// shape for a MEASUREMENT and the wrong shape for a default.
const DET_SLOTS = urlParams.has('detslots') ? (parseInt(urlParams.get('detslots')) ? 1 : 0) : 0;

// ?rootpool= / ?rootstep= / ?rootcouple= / ?rootmanage= -- the root-pool
// staging flags (plans/uniform-levels.md U1..U5-4).
//
// The four `const`s and their sixty-five lines of design record now live in
// amr2d-gpu.mjs's readRootFlags, because U7-4 puts the root pool on all five
// pages and a flag's meaning copied five times is a flag that means five
// things. This page reads the full staging set: it is where U5-2, U5-3 and
// U5-4 are A/B'd in one build, which is what those three flags exist for.
const rootFlags = readRootFlags(urlParams);
const ROOT_POOL = rootFlags.pool;
const ROOT_STEP = rootFlags.step;

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
  // card-params.mjs owns this rule and tools/test-card-params.js tests it.
  // Five pages inlined the same loop -- see plans/2D-backport.md B3a.
  const tauAtLevel = (m) => tauAtLevelOf(TAU, m);
const FSCALE  = 1e7;

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

function handleErr(e) {
  // Status line AND a legible on-page overlay -- see error-overlay.mjs for why
  // the 12px status line alone was not enough.
  reportFatal(statusEl, e);
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

// Milestone 7: level m's own cell size, in L0-buffer-space units. Level 1's
// own cell is 0.5 L0 units (matches amr_step1.wgsl/amr_interp_dense_parent.
// wgsl's `fineToCoarseUnit`'s 0.5 factor); it halves again each level down.
function cellSizeL0AtLevel(m) {
  return 2 ** -m;
}

async function init() {
  if (!navigator.gpu) { reportNoWebGPU(statusEl); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { reportNoAdapter(statusEl); return; }

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
  // at, e.g. step1BGL), same "spec minimum, not a real GPU limit"
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

  // Canvas sizing and swapchain reconfiguration -- see canvas-fit.mjs,
  // which carries the reasoning for the changed-size guard (it was ten
  // identical copies of it).
  const resize = makeCanvasFit({ canvas, ctx, device, format: fmt });
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
  // ?diag=1 counters -- 8 u32 slots, read+zeroed via debugReadDiag().
  const diagBuf = device.createBuffer({ size: 8 * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  const diagReadBuf = device.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const dummyBlockSlotBuf = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(dummyBlockSlotBuf, 0, new Int32Array([-1]));

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
  // U5-4: level 1 allocates in QUADS once the root is its parent. The
  // predicate is derived from the flags ONCE -- in readRootFlags, because it
  // has to agree with the allocator, the reset, the cascade and the dispatch,
  // and a recomputed condition in four places is how those drift.
  const ROOT_MANAGED = rootFlags.managed;
  if (ROOT_POOL) pools[0] = allocRootPool(device, U, { W, H, RB });
  {
    let curNBX = NBX, curNBY = NBY; // level 1's logical grid = today's coarse block grid
    for (let m = 1; m < N_LEVELS; m++) {
      // ?maxFineBlocks= sizes LEVEL 1; each deeper level takes its own
      // ?maxFineBlocks<m>=. Defaults come from POOL_PEAKS via poolSlotsFor --
      // measured per level, not one flat number for all of them.
      const maxFineBlocks = m === 1
        ? MAX_FINE_BLOCKS
        : (urlParams.has(`maxFineBlocks${m}`)
            ? parseInt(urlParams.get(`maxFineBlocks${m}`))
            : poolSlotsFor(POOL_PEAKS, m, N_LEVELS));
      const pool = allocLevelPool(device, U, m, curNBX, curNBY, maxFineBlocks, NCELLS1,
        m === 1 ? { quadAlloc: ROOT_MANAGED } : {});
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
  // Level 1's eager free-list seed, for the PER-BLOCK allocator only. Under
  // quad allocation allocLevelPool has already written the quad-indexed pair
  // (and would be overwritten by a block-indexed one here), which is why this
  // is conditional rather than unconditional-and-harmless.
  if (!ROOT_MANAGED) {
    device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
    device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
  }

  // Milestone 6/8: per-level uniform (LevelParams) for every level>=2's
  // pool-parent interp/average/step1/force shaders. Layout: {nbx:u32,
  // nby:u32, parentTau:f32, dxL:f32, hasChild:u32, _pad1:u32, _pad2:u32,
  // _pad3:u32} = 32 bytes -- interp/average/step1_pool only declare the
  // first 4 fields (16 bytes) in their own WGSL struct, which is a valid
  // prefix of this same buffer; amr_force1.wgsl (Milestone 8) is the
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
  // The ROOT gets one too (U3). `parentTau` is deliberately left at 0: the
  // root's pipeline sets OWN_TAU, so that field is never read, and writing a
  // plausible value there would create a second source of truth for a number
  // `state.tau` already holds.
  for (let c = (ROOT_POOL ? 0 : 1); c < N_LEVELS; c++) {
    const pool = pools[c];
    if (!pool) continue;
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
    staticDv.setFloat32(20, K_EPS, true);
    device.queue.writeBuffer(pool.levelParamsBuf, 0, staticBuf);
  }
  function updateLevelParams() {
    // tau = 1 IS A REAL SINGULARITY for the post-collision transfer -- refuse
    // it, do not divide by it. See amr2d.mjs's tauChainSingularity.
    //
    // THE GUARD LIVES HERE because this is the one place a per-level tau ever
    // reaches the GPU: init calls it once, and every live TAU change funnels
    // through it. One guard therefore covers both, which is what keeps the
    // slider from walking into the singularity after a clean init -- reachable
    // on this page at ?levels>=4, where L0 tau = 0.5625 (inside the slider's
    // own 0.5005..0.6 range) puts level 3 exactly on 1.
    const sing = DC_PRE === 0 ? tauChainSingularity(TAU, N_LEVELS) : null;
    if (sing) refuseConfig(statusEl, tauSingularityMessage(sing));
    for (let c = 1; c < N_LEVELS; c++) {
      device.queue.writeBuffer(pools[c].levelParamsBuf, 8, new Float32Array([tauAtLevel(c - 1)])); // level c's parent is level c-1
    }
  }
  updateLevelParams();

  let paramsDirty = false;
  const updateGPUParams = () => {
    const data = new Float32Array([MASS, I_BODY, G_EFF, A, B]);
    device.queue.writeBuffer(cardStateBuf, 9 * 4, data);
    // BEFORE the L0 tau write, not after: the guard inside throws, and a
    // half-applied change (L0 on the new tau, every finer level still on the
    // old one) is exactly the silent degradation it exists to prevent.
    updateLevelParams(); // TAU changed -- every level's recursive tau shifts too
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
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
  //
  // The shipped default is the slider's own `value` in index-amr.html (0, i.e.
  // off) -- read back out by opacityFromSlider() where the uniform is created,
  // so the number lives in exactly one place. It used to be written here as a
  // literal 1.0 AND as a separate literal in the markup, which is two things
  // to keep in step for no reason.
  const overlaySlider = document.getElementById('slider-overlay');
  const overlayValEl = document.getElementById('val-overlay');
  if (overlaySlider) {
    if (overlayValEl) overlayValEl.textContent = parseFloat(overlaySlider.value).toFixed(2);
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
    if (outlineValEl) outlineValEl.textContent = parseFloat(outlineSlider.value).toFixed(2);
    outlineSlider.oninput = () => {
      const v = parseFloat(outlineSlider.value);
      outlineValEl.textContent = v.toFixed(2);
      device.queue.writeBuffer(outlineOpacityBuf, 0, new Float32Array([v]));
    };
  }

  const [stepSM, frcSM, phySM, renSM, digestSM, interpDenseSM, interpPoolSM, step1SM, avgSM, avgPoolSM, criterionSM, manageSM, force1SM, criterionPoolSM, managePoolSM, mirrorRootSM] = await Promise.all([
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
    // Milestone 7 / B3-1: ONE fine-step kernel for every level. The two
    // average entry files below are still two pipelines (their PARENTS have
    // different storage layouts), but share one body since B3-2 -- see
    // shaders/amr_step1.wgsl and shaders/common_average.wgsl.
    loadShader(device, 'shaders/amr_step1.wgsl'),
    loadShader(device, 'shaders/amr_average_f2c.wgsl'),
    loadShader(device, 'shaders/amr_average_pool_parent.wgsl'),
    loadShader(device, 'shaders/amr_criterion.wgsl'),
    loadShader(device, 'shaders/amr_manage.wgsl'),
    // Milestone 8: per-level force/torque integration, same dense/pool
    // addressing split as everything else -- see amr_force1.wgsl's header.
    loadShader(device, 'shaders/amr_force1.wgsl'),
    // Milestone 9: per-level criterion + quad allocator/2:1-balance,
    // parent=level>=1 -- see amr_criterion_pool.wgsl/amr_manage_pool.wgsl.
    loadShader(device, 'shaders/amr_criterion_pool.wgsl'),
    loadShader(device, 'shaders/amr_manage_pool.wgsl'),
    // U2: dense L0 -> root pool, an addressing proof and nothing else.
    // Loaded unconditionally (a module nothing instantiates costs nothing)
    // so the WGSL is compiled on every page load rather than only under
    // ?rootpool=1, where a syntax error would hide until someone set it.
    loadShader(device, 'shaders/amr_mirror_root.wgsl'),
  ]);

  // The six modules the shared coupling pipelines need, as one object.
  const modules = { interpDenseSM, interpPoolSM, avgSM, avgPoolSM, criterionSM, manageSM };

  // U7-0: the fourteen bind group layouts, from ONE place. They were spelled
  // out inline here and in four other pages, byte-identical in all of them --
  // see makeAMRLayouts for why that mattered more than the line count.
  const layouts = makeAMRLayouts(device);
  const {
    stepBGL, frcBGL, phyBGL, renBGL, interpBGL, interpPoolParentBGL,
    avgBGL, avgPoolBGL, criterionBGL, criterionPoolBGL, manageBGL,
    managePoolBGL, step1BGL, force1BGL,
  } = layouts;

  const constants = { W, H, SDF_FAR };
  // Same, plus the packed-f layout selector. Separate object because
  // `constants` is also fed to phy/render, whose modules don't declare F16 --
  // and WebGPU makes passing an undeclared override a pipeline-creation
  // error, not a warning. Every pipeline whose shader @includes
  // common_fpack.wgsl must get F16; no other pipeline may.
  const fConstants = { ...constants, F16, K_EPS };
  // VORT_SCALE/VORT_GAMMA are supplied by makeRenderPipeline below, which is
  // the only thing that ever varies them.
  // U6: how many pool levels the renderer walks. renderPoolLevels() REFUSES a
  // configuration deeper than the shader binds, rather than drawing it without
  // its finest level -- which is what this override replaced HAS_LEVEL2 for.
  const renderConstants = { W, H, RB, N_POOL_LEVELS: renderPoolLevels(N_LEVELS), K_EPS };
  const step1Constants = { W, H, RB, SDF_FAR, F16, DIRECT_GHOST: GHOST_COPY ? 0 : 1 };
  const manageConstants = { DIAG, W, H, SDF_FAR, REFINE_THRESH, COARSEN_THRESH, FORCE_REFINE_MARGIN, FORCE_REFINE_LOOKAHEAD, SPONGE_EXCLUDE_W, 
    N_REFINE_INC, N_REFINE_MAX, MAX_LEVEL: N_LEVELS - 1, BOX_REFINE, DET_SLOTS };

  // U7-1: the twelve coupling pipelines, from ONE place. Every page built
  // these identically; see makeCouplingPipelines for what stays per page and
  // why. `couplingConstants` carries the bundles it derived, so the
  // measurement twins and the root-parent variants below are built from the
  // SAME literal the real pipeline used rather than a second copy of it.
  const {
    constants: couplingConstants,
    interpPL, interpInitPL, interpFFPL,
    interpPoolParentPL, interpPoolParentInitPL, interpPoolParentFFPL,
    avgPL, avgPoolPL, criterionPL,
    manageDecidePL, manageCoarsenPL, manageRefinePL,
  } = makeCouplingPipelines(device, layouts, modules,
      { W, H, RB, F16, DC_PRE, manage: manageConstants });

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
  // VORT_SCALE/VORT_GAMMA are pipeline-overridable constants specialized into
  // the fragment shader here, so changing them live means rebuilding this one
  // pipeline. vort-controls.mjs owns the sliders and the per-frame coalescing
  // (and documents why these are not uniforms); this side owns only the
  // pipeline itself.
  const makeRenderPipeline = (scale, gamma) => device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main', constants },
    fragment: {
      module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }],
      constants: { ...renderConstants, VORT_SCALE: scale, VORT_GAMMA: gamma },
    },
    primitive: { topology: 'triangle-list' },
  });
  let renPL = makeRenderPipeline(VORT_SCALE, VORT_GAMMA);
  installVortControls({
    scale: VORT_SCALE, gamma: VORT_GAMMA,
    rebuild: (scale, gamma) => { renPL = makeRenderPipeline(scale, gamma); },
  });
  // Milestone 7: level>=2 fine step / average -- one pipeline object each,
  // reused across every level pair (no per-level overrides needed; NBX/NBY/
  // parentTau are runtime uniform reads, not compile-time constants -- see
  // shaders/amr_step1.wgsl's header, same reasoning as M6's
  // interpPoolParentPL).
  const step1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
    compute: { module: step1SM, entryPoint: 'main', constants: step1Constants }
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
    interpDense:  device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),           compute: { module: interpDenseSM, entryPoint: 'main', constants: { ...couplingConstants.interp,       NOOP: 1 } } }),
    interpPool:   device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }), compute: { module: interpPoolSM,  entryPoint: 'main', constants: { ...couplingConstants.interpPool,   NOOP: 1 } } }),
    avg:          device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),              compute: { module: avgSM,         entryPoint: 'main', constants: { ...couplingConstants.avg, NOOP: 1 } } }),
    avgPool:      device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [avgPoolBGL] }),          compute: { module: avgPoolSM,     entryPoint: 'main', constants: { RB, F16, DC_PRE,         NOOP: 1 } } }),
  };
  // step1-ring: fine step over the tile INTERIOR only, skipping the ghost
  // ring -- a proxy for FB 20 -> 16. See the SKIP_GHOST override in
  // shaders/amr_step1.wgsl for what it measured and what it corrects.
  const ringPLs = {
    step1: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }), compute: { module: step1SM, entryPoint: 'main', constants: { ...step1Constants, SKIP_GHOST: 1 } } }),
  };
  // ghostcopy: the legacy materialized-same-level-ghost path (DIRECT_GHOST=0
  // plus the fine-fine copy pass re-encoded), as a bench CONFIGURATION rather
  // than only a page-load flag. That matters on the phone, which gets one
  // sweep per session and whose medians are worthless across sessions because
  // it thermally ramps 24-57% -- ?bench=1 interleaves its configurations
  // within one run, so this is the only way to A/B the change there at all.
  // Note the sign: this config is SLOWER than the baseline, so its reported
  // share is negative, and the magnitude is what neighbour-addressed streaming
  // buys. Same module, one override flipped -- see the noopPLs comment on why
  // a variant compiled from different source would not be honest.
  const legacyGhostPLs = {
    step1: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }), compute: { module: step1SM, entryPoint: 'main', constants: { ...step1Constants, DIRECT_GHOST: 0 } } }),
  };
  // Milestone 8: level 1's own force pass. HAS_CHILD is baked in at
  // pipeline-creation time -- level 1 has exactly one dedicated pipeline
  // (not shared across levels), so whether level 2 exists is fixed for the
  // whole session (see amr_force1.wgsl's header).
  // Milestone 8: level>=2's own force pass, one pipeline reused across
  // every such level (no per-level overrides -- hasChild/dxL are runtime
  // LevelParams reads, see amr_force1.wgsl's header).
  const force1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [force1BGL] }),
    compute: { module: force1SM, entryPoint: 'main', constants: { W, H, RB, F16 } }
  });
  // Two pipelines, same module, different entry points -- dispatched as two
  // SEPARATE passes (coarsen fully completing before refine starts) to
  // avoid a same-dispatch free-list race. See amr_manage.wgsl's header for
  // the bug this fixes (found by this milestone's own validation).
  // plans/2D-backport.md B2: the 2:1 closure's own pipelines. Built but
  // NOT yet in dispatchMacroStep -- see debugCascadeRoundTrip.
  const cascadeSM = await loadShader(device, 'shaders/amr_cascade.wgsl');
  // U5-4: level 1's want set is closed into quads too, once level 1 allocates
  // in quads. See makeCascadePipelines' own note.
  const cascade = makeCascadePipelines(device, cascadeSM, pools, N_LEVELS,
    { quadCompleteFrom: ROOT_MANAGED ? 1 : 2 });

  // D0's deferred blockSlot writes.
  //
  // KEPT ON EVIDENCE, NOT ON ARGUMENT. Once scanCandidates took over the
  // ranking, refine()/coarsen() no longer read another thread's blockSlot, so
  // these looked redundant and were removed. Three runs later one of them had
  // diverged (levels=3 read 4aafcac0 against the established ce1bd4d8, with
  // 104/248 tiles against 103/240) and they went back. The mechanism is NOT
  // understood -- see plans/uniform-levels.md 1.2e. Do not remove them again
  // without a measurement that says they are inert.
  const manageLinkCoarsenPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'linkCoarsen', constants: manageConstants }
  });
  const manageLinkRefinePL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'linkRefine', constants: manageConstants }
  });
  // U7-4a: THE ROOT POOL'S SOLVER HALF, from amr2d-gpu.mjs.
  //
  // U2's mirror (and, since 1.2, the root's SEEDER), U3's root step, U4-1's
  // criterion redirect and U5-3's live coupling. Built here on the dev page
  // exactly as on the other four -- see makeRootPool for the solver/instrument
  // split and for why the interleaving, not the move, was the work.
  //
  // `beginPass` is this page's profiling wrapper, so `root step` and
  // `L1->root average` keep their labels in debugProfileMacroStep. A shipped
  // page passes nothing and gets a plain beginComputePass.
  const rootGpu = ROOT_POOL ? makeRootPool(device, U, layouts,
    { ...modules, step1SM, mirrorRootSM }, pools,
    { W, H, RB, F16, DC_PRE, step1Constants, couplingConstants,
      cardStateBuf, denseFBuf: f_a, flags: rootFlags, beginPass }) : null;

  // U4-1: the CRITERION, serving the ROOT.
  //
  // The same module every level >= 1 uses -- amr_criterion_pool.wgsl -- with
  // GHOST 0 and the root's own block grid. The root->level-1 relation IS the
  // pool parent->child relation: a root tile is 2*RB = 16 cells and level 1's
  // block grid is W/RB, exactly twice the root's W/(2*RB), so one root tile
  // carries four level-1 children and each 8x8 quadrant is one workgroup
  // producing one child criterion. That is the same shape the dense kernel
  // has, where one L0 8x8 block produces one -- which is why these two can be
  // compared at all.
  //
  // IT IS INERT. It writes its OWN buffer, never level 1's, so the page's
  // refinement decisions still come entirely from amr_criterion.wgsl. What it
  // buys is the differential test: same velocity field (U4-0 proved the root's
  // IS the dense one), two separately written kernels, and the question is
  // whether they agree bit for bit.
  let rootCritPL = null, rootCritBG = null, rootCritBuf = null;
  // The DENSE criterion's target once the root manages level 1. Built by
  // makeRootPool, because it is the solver's -- not this comparator's --
  // requirement: the shared refine round still encodes amr_criterion.wgsl, and
  // under quad management it must not land on level 1's real criterion buffer.
  const denseCritBuf = rootGpu ? rootGpu.denseCritBuf : null;
  if (ROOT_POOL) {
    const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
    rootCritBuf = device.createBuffer({
      size: pools[1].NBLOCKS * 4,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    rootCritPL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [criterionPoolBGL] }),
      compute: { module: criterionPoolSM, entryPoint: 'main',
                 constants: { RB, NBX_PARENT: spec.nbx, NBY_PARENT: spec.nby, GHOST: 0 } },
    });
    rootCritBG = device.createBindGroup({ layout: criterionPoolBGL, entries: [
      { binding: 0, resource: { buffer: pools[0].finePoolVel } },
      { binding: 1, resource: { buffer: pools[0].slotToBlockBuf } },
      { binding: 2, resource: { buffer: rootCritBuf } },
      { binding: 3, resource: { buffer: pools[0].blockSlotBuf } },
    ]});
    // U5-4 SWAPS THE ROLES, and the swap is what keeps the `crit` column from
    // going vacuous. Once the pool criterion at parent level 0 is the LIVE
    // writer of level 1's blockCriterion, running the inert root twin as well
    // would compare one kernel's output against its own -- a comparison that
    // cannot fail. So the DENSE kernel becomes the inert one, writing
    // `denseCritBuf` above, and debugCheckRootCriterion scores that against
    // the live buffer. Two independently written kernels either way; which one
    // is authoritative is the only thing that moved.
  }

  // U4-2: the FORCE pass, serving the ROOT.
  //
  // amr_force.wgsl IS ALREADY DEAD IN THE SOLVER and this is not what replaces
  // it there. Only the FINEST level's force pass is dispatched (B4-3), and
  // every AMR page refuses ?levels<2, so `finestLevel === 0` cannot be reached
  // -- the dense force kernel has not contributed to a shipped number in a
  // long time. What keeps it alive is main-cylinder-amr.js's
  // debugForceBreakdown, which runs each level's pass in isolation and is the
  // instrument that MEASURED the coarser levels contributing exactly zero
  // before the masking was deleted. So the dense kernel is a live INSTRUMENT
  // over a dead code path, and U4 retires it by making the root's own pass
  // reproduce it rather than by deleting it unmeasured.
  //
  // Two scratch accumulators so the comparison never touches the real one:
  // the body integrator reads forceBuf every macro-step, and a debug pass that
  // added to it would move the card.
  let rootForcePL = null, rootForceBG = null, rootForceBuf = null, denseForceBuf = null,
      denseForceBG = null, rootSlotForceBuf = null;
  if (ROOT_POOL) {
    const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
    const mkForce = () => device.createBuffer({ size: 4 * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    rootForceBuf = mkForce();
    denseForceBuf = mkForce();
    rootSlotForceBuf = device.createBuffer({ size: spec.slots * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
    rootForcePL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [force1BGL] }),
      compute: { module: force1SM, entryPoint: 'main',
                 constants: { W, H, RB, F16, GHOST: 0, NO_PARENT: 1 } },
    });
    rootForceBG = device.createBindGroup({ layout: force1BGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: pools[0].finePoolF_a } },
      { binding: 2, resource: { buffer: rootForceBuf } },
      { binding: 3, resource: { buffer: pools[0].slotToBlockBuf } },
      { binding: 4, resource: { buffer: pools[0].levelParamsBuf } },
      { binding: 5, resource: { buffer: rootSlotForceBuf } },
      { binding: 6, resource: { buffer: pools[0].blockSlotBuf } },
    ]});
    denseForceBG = device.createBindGroup({ layout: frcBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: f_a } },
      { binding: 2, resource: { buffer: denseForceBuf } },
    ]});
  }

  // U5-1: the COARSE->FINE INTERPOLATION into level 1, from the ROOT POOL.
  //
  // The same module every L(m)->L(m+1) hop with m>=1 uses --
  // amr_interp_pool_parent.wgsl -- with PARENT_GHOST 0. That override is the
  // whole of the change: level 1's parentTau is already tauAtLevel(0), i.e.
  // the dense accessor's own `state.tau`, and its parent SLOT and QUADRANT are
  // derivable from its own block index because the root is always full (see
  // shaders/common_interp_parent_pool.wgsl's parentSlotOf/quadrantOf). What is
  // NOT free is the fetch: a child covers one RB-wide quadrant of its parent
  // and the bilinear stencil reaches GHOST cells past it, so every level-1
  // tile needs root cells from outside its parent root tile on two of its four
  // sides. A ringed parent has those in its ring; the root has no ring, and
  // resolving them against the neighbouring ROOT TILE is what U5-1 adds.
  //
  // IT IS INERT, in the stronger sense U4's pipelines are not quite: BOTH
  // sides write scratch buffers, so the page's own level-1 pool is not touched
  // at all and the comparison's two legs start from bit-identical state. The
  // interp pass is idempotent on a paused page (a ghost cell's value depends
  // only on the parent and on INTERIOR cells, which it never writes), but
  // re-running the shipped pass on the shipped buffer would still leave the
  // page's ghosts holding something a macro-step did not produce.
  //
  // `debugCheckRootInterp` is the gate. See it for the protocol and for why
  // the wrong-parent leg is not optional.
  //
  // U7-4a LEFT THESE HERE AND TOOK THE LIVE ONES. Both legs write scratch and
  // are read by nothing but debugCheckRootInterp/debugCheckRootAverage, so
  // they are instrument; makeRootPool owns U5-3's live coupling. The two
  // objects the two halves SHARED -- the `unread` sentinel and `rootAvgPL` --
  // come back from it rather than being rebuilt, so the leg that scores the
  // live restriction is running the live pipeline and not a copy of it.
  let rootInterpPL = null, rootInterpBG = null, rootInterpStaleBG = null,
      denseInterpScratchBG = null, interpScratchDense = null, interpScratchRoot = null;
  let rootAvgBG_live = null, rootAvgBG_stale = null,
      denseAvgScratchBG = null, avgScratchDense = null, avgScratchRoot = null;
  const rootAvgPL = rootGpu ? rootGpu.rootAvgPL : null;
  if (ROOT_POOL) {
    const unread = rootGpu.unread;
    const mkScratch = () => device.createBuffer({
      size: pools[1].fSizePool, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    interpScratchDense = mkScratch();
    interpScratchRoot = mkScratch();
    rootInterpPL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [interpPoolParentBGL] }),
      compute: { module: interpPoolSM, entryPoint: 'main',
                 constants: { ...couplingConstants.interpPool, PARENT_GHOST: 0 } },
    });
    const rootBG = (parentF) => device.createBindGroup({ layout: interpPoolParentBGL, entries: [
      { binding: 0, resource: { buffer: pools[1].levelParamsBuf } },
      { binding: 1, resource: { buffer: parentF } },
      { binding: 2, resource: { buffer: interpScratchRoot } },
      { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } },
      { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } },
      { binding: 5, resource: { buffer: pools[1].blockSlotBuf } },
      { binding: 6, resource: { buffer: unread } },
      { binding: 7, resource: { buffer: unread } },
    ]});
    rootInterpBG = rootBG(pools[0].finePoolF_a);
    // The wrong-parent CONTROL: the root's other ping-pong buffer, which at
    // rest holds the state one substep back. Must come back DIRTY, and how
    // dirty is the count of ring cells that actually take the parent hop --
    // see debugCheckRootInterp.
    rootInterpStaleBG = rootBG(pools[0].finePoolF_b);
    // The dense leg, writing the other scratch. interpPL's own bind groups
    // target the real pool; this is the same pipeline against a copy.
    denseInterpScratchBG = device.createBindGroup({ layout: interpBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: f_a } },
      { binding: 2, resource: { buffer: interpScratchDense } },
      { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } },
      { binding: 4, resource: { buffer: pools[1].newlyActivatedBuf } },
      { binding: 5, resource: { buffer: pools[1].blockSlotBuf } },
    ]});

    // ── U5-2: the RESTRICTION, level 1 -> the ROOT POOL ──────────────────
    //
    // The reverse hop of U5-1 and the other half of the coupling, built the
    // same way: amr_average_pool_parent.wgsl with PARENT_GHOST 0, both legs
    // writing scratch so the page's own buffers are never touched.
    //
    // STRUCTURALLY EASIER THAN U5-1, and the asymmetry is the interesting
    // part: restriction writes one parent cell per child cell and the
    // destination is always inside the parent's own interior, so there is no
    // stencil and nothing to resolve against a neighbouring root tile. The
    // ring-free root costs this direction only the offset and the stride.
    //
    // The two scratches are the SAME SIZE, which is U1's identity showing up
    // as a line of code: the root pool has no ring, so tiling the domain costs
    // no padding and `spec.cells === W * H` exactly.
    avgScratchDense = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    avgScratchRoot = device.createBuffer({ size: pools[0].fSizePool, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    if (pools[0].fSizePool !== fSize) {
      throw new Error(`root pool f is ${pools[0].fSizePool} bytes, dense f ${fSize} -- U1's identity is broken`);
    }
    const rootAvgBG = (childF) => device.createBindGroup({ layout: avgPoolBGL, entries: [
      { binding: 0, resource: { buffer: pools[1].levelParamsBuf } },
      { binding: 1, resource: { buffer: childF } },
      { binding: 2, resource: { buffer: avgScratchRoot } },
      { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } },
      { binding: 4, resource: { buffer: unread } },
      { binding: 5, resource: { buffer: unread } },
    ]});
    rootAvgBG_live = rootAvgBG(pools[1].finePoolF_a);
    // The stale-CHILD control. `?rootstep=0` cannot discriminate this column
    // -- see debugCheckRootAverage -- so the liveness control has to change
    // the one input the restriction actually reads.
    rootAvgBG_stale = rootAvgBG(pools[1].finePoolF_b);
    denseAvgScratchBG = device.createBindGroup({ layout: avgBGL, entries: [
      { binding: 0, resource: { buffer: cardStateBuf } },
      { binding: 1, resource: { buffer: pools[1].finePoolF_a } },
      { binding: 2, resource: { buffer: avgScratchDense } },
      { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } },
    ]});
  }

  // D0's candidate scan: two pipelines from one entry point, so the grant and
  // release rules cannot drift into two spellings of "candidate". One
  // workgroup each -- see scanCandidates' header for why that is enough.
  const manageScanGrantPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'scanCandidates', constants: { ...manageConstants, SCAN_RELEASE: 0 } }
  });
  const manageScanReleasePL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'scanCandidates', constants: { ...manageConstants, SCAN_RELEASE: 1 } }
  });

  // Milestone 9: one criterion/manage pipeline PAIR per PARENT level
  // (1..N_LEVELS-2, i.e. every level that can itself have a child) --
  // NBX_PARENT/NBY_PARENT/PARENT_CELL_SIZE_L0/PARENT_HAS_CACHED_ORIGIN are
  // compile-time overrides, one pipeline object per parent level, not
  // shared the way M6-M8's pool-parent pipelines are (see
  // amr_manage_pool.wgsl's header for why that's the right tradeoff here).
  // Keyed by PARENT level m, deciding child level m+1.
  const criterionPoolPLs = {};
  const managePoolDecidePLs = {};
  const managePoolCoarsenPLs = {};
  const managePoolRefinePLs = {};
  // U5-4: `m` is the PARENT level, and it starts at the ROOT once the root is
  // a pool level. One manager, every level, which is the stage's whole point.
  for (let m = (ROOT_MANAGED ? 0 : 1); m < N_LEVELS - 1; m++) {
    const parentPool = pools[m];
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
      SPONGE_EXCLUDE_W,
      ...childParams,
      N_REFINE_INC, N_REFINE_MAX, MAX_LEVEL: N_LEVELS - 1,
      BOX_REFINE,
      // U5-4: the refusal counter is opt-in the same way the dense manager's
      // is, and for the same reason -- at DIAG=0 every counter stays 0 and the
      // pool-starvation gate would read TRUE VACUOUSLY.
      DIAG,
      DET_SLOTS,
    };
    criterionPoolPLs[m] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [criterionPoolBGL] }),
      // GHOST 0 at the root, which selects the ring-free stencil -- the same
      // constants U4-1's inert rootCritPL was built with, now on the live
      // pipeline. NBY_PARENT is only read on that path.
      compute: { module: criterionPoolSM, entryPoint: 'main',
                 constants: m === 0
                   ? { RB, NBX_PARENT: parentPool.NBX, NBY_PARENT: parentPool.NBY, GHOST: 0 }
                   : { RB, NBX_PARENT: parentPool.NBX } }
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

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});

  const frcBG_a = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }]});
  const frcBG_b = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }]});

  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  // Both opacity uniforms are seeded from their slider's shipped `value`, so
  // the default lives only in index-amr.html and the uniform cannot start out
  // disagreeing with the control that owns it. `fallback` covers a page that
  // has no such slider at all (the harness pages reuse parts of this file).
  const opacityFromSlider = (el, fallback) => {
    const v = el ? parseFloat(el.value) : NaN;
    return Number.isFinite(v) ? v : fallback;
  };
  const overlayOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(overlayOpacityBuf, 0,
    new Float32Array([opacityFromSlider(overlaySlider, 0.0)]));
  // Quadtree outline opacity -- optional, off by default (see
  // shaders/amr_render.wgsl's own comment on why this is a separate
  // uniform from overlayOpacityBuf's fill).
  const outlineOpacityBuf = device.createBuffer({ size: 4, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(outlineOpacityBuf, 0,
    new Float32Array([opacityFromSlider(outlineSlider, 0.0)]));
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

  // U4-3: the DIGEST, over the root pool.
  //
  // It needs NO shader change to serve the root -- it addresses a flat cell
  // index and the root pool holds exactly W*H cells, the same count. Only the
  // bound buffer moves. What it needs instead is a comparison that is
  // MEANINGFUL, and the shipped sampled form is not: it picks cells by STORAGE
  // INDEX, and the root pool is a permutation of the dense grid, so sample `i`
  // is a different physical cell in each. Two honest digests of one field,
  // legitimately unequal.
  //
  // FULL=1 reduces over every cell, which makes exactly one component
  // comparable: `max` is invariant under permutation AND summation order, so
  // digest[2] must match bit-for-bit. The two sums are not -- adding the same
  // 65536 floats in two orders is not required to give the same f32 -- and
  // they are reported rather than gated.
  let digestFullPL = null, digestDenseFullBG = null, digestRootFullBG = null,
      digestDenseFullBuf = null, digestRootFullBuf = null;
  if (ROOT_POOL) {
    const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
    const mk = () => device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC });
    digestDenseFullBuf = mk();
    digestRootFullBuf = mk();
    digestFullPL = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [digestBGL] }),
      compute: { module: digestSM, entryPoint: 'main', constants: { NCELLS, FULL: 1 } },
    });
    digestDenseFullBG = device.createBindGroup({ layout: digestBGL, entries: [
      { binding: 0, resource: { buffer: velBuf } },
      { binding: 1, resource: { buffer: digestDenseFullBuf } },
    ]});
    digestRootFullBG = device.createBindGroup({ layout: digestBGL, entries: [
      { binding: 0, resource: { buffer: pools[0].finePoolVel } },
      { binding: 1, resource: { buffer: digestRootFullBuf } },
    ]});
    // The root pool holds exactly the dense grid's cell count -- U1's whole
    // no-padding claim. If that ever stops being true the digest would silently
    // read past one of them, so it is asserted rather than assumed.
    if (spec.cells !== NCELLS) {
      throw new Error(`root pool holds ${spec.cells} cells, dense grid ${NCELLS}`);
    }
  }


  // U6: one velocity/indirection pair per POOL level, walked finest-first by
  // shaders/amr_render.wgsl. Shared with the other four AMR pages, which all
  // built this inline and three of which never passed the level-2 override at
  // all -- see makeRenderBindGroup.
  const renBG = makeRenderBindGroup(device, renBGL, pools,
    { velBuf, cardStateBuf, overlayOpacityBuf, outlineOpacityBuf });
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
  // Level 1's own fine-step bind groups, in the SAME layout every other level
  // uses -- see the c>=2 loop below, which builds the identical pair. Level 1
  // differs only in what its levelParams says (dxL 0.5, parentTau = L0's own
  // tau); there is no longer a second kernel, a second layout or a second
  // pipeline for it. plans/2D-backport.md B3-1.
  pools[1].step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_a } }, { binding: 2, resource: { buffer: pools[1].finePoolF_b } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 5, resource: { buffer: pools[1].levelParamsBuf } }, { binding: 6, resource: { buffer: pools[1].blockSlotBuf } }]});
  pools[1].step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: pools[1].finePoolF_b } }, { binding: 2, resource: { buffer: pools[1].finePoolF_a } }, { binding: 3, resource: { buffer: pools[1].finePoolVel } }, { binding: 4, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 5, resource: { buffer: pools[1].levelParamsBuf } }, { binding: 6, resource: { buffer: pools[1].blockSlotBuf } }]});
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

  // ── U5-3: which parent level 1 is coupled to, chosen ONCE ────────────────
  //
  // The dense-parent bundle goes in; whichever one level 1 is actually coupled
  // to comes back. The reasoning -- and the fallback, which is structural
  // rather than a ternary per dispatch site -- is makeRootPool's `coupleL1`.
  const denseParent = {
    interpPL, interpInitPL, interpFFPL, interpNoopPL: noopPLs.interpDense,
    interpBG:     (b) => b ? interpBG_readB : interpBG_readA,
    interpInitBG: (b) => b ? interpInitBG_readB : interpInitBG_readA,
    interpFFBG:   interpFFBG_b,
  };
  const {
    interpPL:     l1InterpPL,
    interpInitPL: l1InterpInitPL,
    interpFFPL:   l1InterpFFPL,
    interpNoopPL: l1InterpNoopPL,
    interpBG:     l1InterpBG,
    interpInitBG: l1InterpInitBG,
    interpFFBG:   l1InterpFFBG,
  } = rootGpu ? rootGpu.coupleL1(denseParent) : denseParent;

  // Milestone 4b bind groups.
  const criterionBG = device.createBindGroup({ layout: criterionBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: ROOT_MANAGED ? denseCritBuf : pools[1].blockCriterionBuf } }]});
  const manageBG = device.createBindGroup({ layout: manageBGL, entries: [{ binding: 0, resource: { buffer: pools[1].blockCriterionBuf } }, { binding: 1, resource: { buffer: pools[1].blockSlotBuf } }, { binding: 2, resource: { buffer: pools[1].slotToBlockBuf } }, { binding: 3, resource: { buffer: pools[1].freeListBuf } }, { binding: 4, resource: { buffer: pools[1].freeCountBuf } }, { binding: 5, resource: { buffer: pools[1].newlyActivatedBuf } }, { binding: 6, resource: { buffer: cardStateBuf } }, { binding: 7, resource: { buffer: pools[1].candRankBuf } }, { binding: 9, resource: { buffer: diagBuf } }, { binding: 10, resource: { buffer: pools[1].wantBuf } }]});

  // Milestone 9: one criterion/manage bind group per PARENT level
  // (1..N_LEVELS-2), deciding child level m+1. Parent=level 1 sources from
  // the flat globals (velBuf/pools[1].*), parent=level>=2 from pools[m]
  // (both are equally valid "parent pool" shapes for this purpose -- the
  // dense-vs-cached-origin distinction is handled entirely by
  // PARENT_HAS_CACHED_ORIGIN, already baked into the pipeline above).
  // U7-2: one criterion/manage bind-group pair per parent level, from ONE
  // place -- byte-identical across four pages before this, and differing on
  // the fifth only by where the loop starts. See makeManageBindGroups.
  const { criterionPoolBGs, managePoolBGs } = makeManageBindGroups(
    device, layouts, pools, N_LEVELS,
    { cardStateBuf, diagBuf, firstParentLevel: ROOT_MANAGED ? 0 : 1 });


  // U7-2: every level >= 2's interp / step / average / force bind groups, from
  // ONE place. Three pages had this loop byte-identical and two had it minus
  // the force block; see makeLevelBindGroups.
  makeLevelBindGroups(device, U, layouts, pools, N_LEVELS,
    { cardStateBuf, forceBuf });

  // Milestone 8: level 1's own force pass. Always reads pools[1].finePoolF_a
  // -- level 1's own buffer is always "current" (_a) at a macro-step
  // boundary, before S_Advance runs (see Milestone 7's own invariant), so
  // no ping-pong variant is needed here (unlike frcBG_a/frcBG_b, which DOES
  // depend on the persistent, cross-macro-step `useB` flag for L0's OWN
  // buffer choice).
  pools[1].debugSlotForceBuf = device.createBuffer({ size: pools[1].MAX_FINE_BLOCKS * 8, usage: U.STORAGE | U.COPY_SRC });
  pools[1].force1BG = device.createBindGroup({ layout: force1BGL, entries: [
    { binding: 0, resource: { buffer: cardStateBuf } },
    { binding: 1, resource: { buffer: pools[1].finePoolF_a } },
    { binding: 2, resource: { buffer: forceBuf } },
    { binding: 3, resource: { buffer: pools[1].slotToBlockBuf } },
    { binding: 4, resource: { buffer: pools[1].levelParamsBuf } },
    { binding: 5, resource: { buffer: pools[1].debugSlotForceBuf } },
      // Bound but never read at GHOST=2 -- these levels have a ring.
    { binding: 6, resource: { buffer: pools[1].blockSlotBuf } },
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

  // The pacer replaces the fixed 64-steps-per-frame loop. STEPS_PER_FRAME is
  // passed as the ceiling; see sim-rate.mjs for why that is what keeps the
  // phone at full speed while halving the desktop.
  const pacer = createSimPacer({ maxStepsPerFrame: STEPS_PER_FRAME, tuPerSec: SIM_RATE });
  const rateSlider = document.getElementById('slider-SIM_RATE');
  const rateValEl = document.getElementById('val-SIM_RATE');
  if (rateSlider) {
    rateSlider.value = SIM_RATE;   // ?simRate= wins over the markup
    if (rateValEl) rateValEl.textContent = SIM_RATE.toFixed(2);
    rateSlider.oninput = () => {
      const v = parseFloat(rateSlider.value);
      if (rateValEl) rateValEl.textContent = v.toFixed(2);
      pacer.setRate(v);
    };
  }

  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;
  let autoRefine = true; // Milestone 4b: on by default so refinement (and its coverage overlay) is visible without a console command; setAutoRefine(false) to disable for manual debugActivateBlock/debugDeactivateBlock testing
  let macroStepCounter = 0;

  const trajectory = [];

  // Rolling trajectory trail (trajectory-trail.mjs). Fed from the SAME
  // CardState readback that fills trajectory[] for the CSV export -- the
  // debug log and the on-screen line are one source of truth. The CSV keeps
  // the full run; only the trail's own buffer rolls, since it needs just
  // enough history to draw one window of descent.
  const trail = createTrail(document.getElementById('trail'));
  // See main.js's identical note and card-total.mjs: the shaders keep
  // x_total/y_total wrapped, this restores the true float64 totals. The
  // backward-jump watchdog below reads the unwrapped value too -- fed the raw
  // one it would count every wrap as a stale readback.
  const totals = createTotalUnwrapper(W, H);
  let trailOpacity = 1.0;
  const trailSlider = document.getElementById('slider-TRAIL');
  const trailValEl = document.getElementById('val-TRAIL');
  if (trailSlider) {
    trailOpacity = parseFloat(trailSlider.value);
    if (trailValEl) trailValEl.textContent = trailOpacity.toFixed(2);
    trailSlider.oninput = () => {
      trailOpacity = parseFloat(trailSlider.value);
      if (trailValEl) trailValEl.textContent = trailOpacity.toFixed(2);
    };
  }

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

  // ── U2: the mirror, and the proof it lands where the host says ───────────
  //
  // ON DEMAND, NOT PER MACRO-STEP. The plan first said "copy every macro-step";
  // that would make the pool's contents a function of WHEN you look, and the
  // thing under test -- the addressing -- is static. Mirroring on request and
  // comparing immediately tests the same map with none of that, and costs a
  // paused page rather than every frame.
  //
  // It binds f_a, which is the current dense buffer whenever the caller has
  // stopped on an even macro-step. debugSnapshotSave relies on exactly the
  // same invariant (STEPS_PER_FRAME is even, so useB returns to false), and
  // debugStepSync leaves it that way -- so "mirror, then check" is only
  // meaningful from the same rest state a snapshot is.
  // SEEDING THE ROOT POOL, AND WHY IT IS A CALL AND NOT A COMMENT.
  //
  // U1 deliberately gave the root pool NO initial field: "a buffer nothing
  // reads should not be given a state that could be mistaken for one." That
  // was right for three stages and became a defect the moment U5-3 made level
  // 1's ghosts read it -- every load and every reset() left the solver
  // interpolating from an unwritten buffer, and nothing said so, because every
  // gate to date called debugMirrorRoot() by hand before looking.
  //
  // MEASURED: `tools/measure-determinism.js --extra=rootpool=1` came back
  // DIFFERS on both level counts with `?detslots=1` -- the configuration D0
  // proved bit-reproducible. The tile counts moved run to run (49 vs 54 at
  // levels=2), so refinement itself was being driven by the unseeded field.
  // That is the fingerprint gate doing exactly what 1.2 promised: catching a
  // state bug no Cd comparison and no 512-step word diff had reported.
  //
  // The mirror is the seeder because it is the map U2 already validated
  // against a third route. At U7 the root is the only L0 and takes `initF()`
  // directly; until then this is one dispatch at init, at reset and after a
  // snapshot load, and never per frame (see debugMirrorRoot's own note).
  const seedRootFromDense = () => { if (ROOT_POOL) debugMirrorRoot(); };
  // The initial seed. resetSim() is NOT called at page load -- the dense grid
  // is written once inline, far above -- so this is a separate call site and
  // not a duplicate of the one in resetSim().
  seedRootFromDense();

  function debugMirrorRoot() {
    if (!rootGpu) throw new Error('debugMirrorRoot: no root pool (?rootpool=1)');
    return rootGpu.seedRootFromDense();
  }

  // Score a root-pool buffer against its dense counterpart, cell by cell,
  // EXACTLY.
  //
  // ONE comparator for every "is the root pool's X the dense X?" question.
  // U2/U3 ask it of `f`; U4 asks it of VELOCITY, because the criterion, the
  // force reduction and the digest all read `vel` and none of them can be
  // believed on the root until its `vel` is known to be the dense one. The
  // only real differences are the component count and whether components are
  // PLANE-MAJOR (`f`: wi*plane + cell) or INTERLEAVED (`vel`: cell*2 + comp),
  // so a second copy of this loop would be exactly the shape CLAUDE.md keeps
  // recording -- near-identical blocks, one of which later gets fixed.
  //
  // The comparison is on raw WORDS, so it is bit-exact and layout-agnostic --
  // under ?f16= it compares packed halves rather than round-tripped floats,
  // which is the regime where a mismatch would be easiest to lose.
  //
  // The host route is amr2d.mjs's rootCellToDense, which derives the dense
  // index from the spec, blockGridAtLevel and denseCellIndex. The shader
  // derives it from slotToBlock and its own overrides. Neither consults the
  // other -- though see plans/uniform-levels.md U2 for why that is necessary
  // and not sufficient, and what it cost when the two shared a premise.
  // Raw words out of any COPY_SRC buffer, on demand. One dedicated staging
  // buffer per call rather than a cached one: these run on a paused page from
  // a debug hook, never per frame, and a shared stage would have to be sized
  // for the largest caller and guarded against overlapping awaits.
  const readBuf = async (buf, bytes) => {
    const stage = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, stage, 0, bytes);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(stage.getMappedRange()).slice();
    stage.unmap(); stage.destroy();
    return out;
  };

  async function compareRootToDense({ denseBuf, poolBuf, comps, interleaved, asFloat, maxReport = 8 }) {
    if (!pools[0]) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
    const rootCells = spec.slots * spec.cellsPerSlot;
    const denseWords = NCELLS * comps;
    const rootWords = rootCells * comps;
    const stageDense = device.createBuffer({ size: denseWords * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const stageRoot = device.createBuffer({ size: rootWords * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(denseBuf, 0, stageDense, 0, denseWords * 4);
    enc.copyBufferToBuffer(poolBuf, 0, stageRoot, 0, rootWords * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([stageDense.mapAsync(GPUMapMode.READ), stageRoot.mapAsync(GPUMapMode.READ)]);
    const dense = new Uint32Array(stageDense.getMappedRange()).slice();
    const root = new Uint32Array(stageRoot.getMappedRange()).slice();
    stageDense.unmap(); stageRoot.unmap();
    stageDense.destroy(); stageRoot.destroy();

    // MAGNITUDE, not just inequality.
    //
    // Counting differing WORDS is the right metric for U2, where the mirror
    // must reproduce the dense grid exactly and any difference is an
    // addressing bug. It is the WRONG metric on its own for U3: a seed
    // difference reaches every cell within a hundred macro-steps, so the count
    // saturates and says nothing about how well the two agree.
    //
    // f32 views only -- F16 packs two halves per word and unpacking here would
    // duplicate f-pack.mjs for a diagnostic. The word count is still reported.
    const denseF = asFloat ? new Float32Array(dense.buffer) : null;
    const rootF = asFloat ? new Float32Array(root.buffer) : null;
    const dIdx = (ci, cell) => (interleaved ? cell * comps + ci : ci * NCELLS + cell);
    const rIdx = (ci, cell) => (interleaved ? cell * comps + ci : ci * rootCells + cell);
    let maxAbs = 0, maxRel = 0, sumSq = 0, sumRef = 0;
    const first = [];
    let checked = 0, mismatched = 0;
    for (let slot = 0; slot < spec.slots; slot++) {
      for (let ly = 0; ly < spec.side; ly++) {
        for (let lx = 0; lx < spec.side; lx++) {
          const denseCell = rootCellToDense({ dims: { W, H }, rb: RB }, slot, lx, ly);
          const rootCell = slot * spec.cellsPerSlot + ly * spec.side + lx;
          for (let ci = 0; ci < comps; ci++) {
            checked++;
            const a = root[rIdx(ci, rootCell)];
            const b = dense[dIdx(ci, denseCell)];
            if (denseF) {
              const va = rootF[rIdx(ci, rootCell)];
              const vb = denseF[dIdx(ci, denseCell)];
              const d = Math.abs(va - vb);
              if (d > maxAbs) maxAbs = d;
              const r = d / Math.max(Math.abs(vb), 1e-12);
              if (r > maxRel) maxRel = r;
              sumSq += d * d; sumRef += vb * vb;
            }
            if (a === b) continue;
            mismatched++;
            if (first.length < maxReport) first.push({ slot, lx, ly, comp: ci, root: a, dense: b, denseCell });
          }
        }
      }
    }
    return {
      ok: mismatched === 0, checked, mismatched, first,
      // relL2 is the field-level agreement: sqrt(sum d^2 / sum ref^2).
      maxAbs: denseF ? maxAbs : null,
      maxRel: denseF ? maxRel : null,
      relL2: denseF ? (sumRef > 0 ? Math.sqrt(sumSq / sumRef) : 0) : null,
    };
  }

  // U2/U3: the root pool's POPULATIONS against the dense grid's.
  const debugCheckRootMirror = (maxReport = 8) => compareRootToDense({
    denseBuf: f_a, poolBuf: pools[0] && pools[0].finePoolF_a,
    comps: fWords(F16), interleaved: false, asFloat: !F16, maxReport,
  });

  // U4: the root pool's VELOCITY against the dense grid's.
  //
  // The input gate for everything U4 moves. The criterion differences `vel`,
  // the force reduction integrates over it, and the digest summarises it --
  // none of which can be scored on the root while its `vel` is unproven, and
  // all three would otherwise report a difference that belongs to the step.
  // `vel` is 2 INTERLEAVED f32 per cell, unlike `f`'s nine planes, and is f32
  // under every packing (?f16= packs `f` only).
  const debugCheckRootVel = (maxReport = 8) => compareRootToDense({
    denseBuf: velBuf, poolBuf: pools[0] && pools[0].finePoolVel,
    comps: 2, interleaved: true, asFloat: true, maxReport,
  });

  // U4-1: the root's CRITERION against the dense kernel's, exactly.
  //
  // Both write one f32 per level-1 block, indexed the same way, so this is a
  // flat array comparison with no addressing in it -- deliberately. The
  // addressing question was settled by U2/U4-0; what is under test here is the
  // STENCIL, and specifically whether resolving a tap against the owning tile
  // reproduces what a periodic wrap over the whole dense grid produces.
  //
  // Raw words again, so it is bit-exact. A max-reduction has no accumulation
  // order to differ over, which is why exactness is the right bar and not an
  // ambitious one.
  async function debugCheckRootCriterion(maxReport = 8) {
    if (!rootCritBuf) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const n = pools[1].NBLOCKS;
    const mk = () => device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const sDense = mk(), sRoot = mk();
    // U5-4 SWAPPED WHICH SIDE IS AUTHORITATIVE, not what is compared. Before
    // it, the dense kernel wrote level 1's live blockCriterion and the root
    // twin wrote rootCritBuf; after it the pool criterion at parent level 0 is
    // the live writer and the DENSE kernel is the one on a scratch. Reading
    // the wrong pair would compare a kernel against itself, which is a
    // comparison that cannot fail -- see denseCritBuf's own note.
    const denseSrc = ROOT_MANAGED ? denseCritBuf : pools[1].blockCriterionBuf;
    const rootSrc  = ROOT_MANAGED ? pools[1].blockCriterionBuf : rootCritBuf;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(denseSrc, 0, sDense, 0, n * 4);
    enc.copyBufferToBuffer(rootSrc, 0, sRoot, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([sDense.mapAsync(GPUMapMode.READ), sRoot.mapAsync(GPUMapMode.READ)]);
    const dw = new Uint32Array(sDense.getMappedRange()).slice();
    const rw = new Uint32Array(sRoot.getMappedRange()).slice();
    sDense.unmap(); sRoot.unmap(); sDense.destroy(); sRoot.destroy();
    const df = new Float32Array(dw.buffer), rf = new Float32Array(rw.buffer);
    let mismatched = 0, maxAbs = 0, nonZero = 0;
    const first = [];
    for (let i = 0; i < n; i++) {
      if (df[i] !== 0) nonZero++;
      const d = Math.abs(rf[i] - df[i]);
      if (d > maxAbs) maxAbs = d;
      if (dw[i] === rw[i]) continue;
      mismatched++;
      if (first.length < maxReport) first.push({ block: i, root: rf[i], dense: df[i] });
    }
    // nonZero is the VACUITY guard: two all-zero criterion arrays agree
    // perfectly and say nothing. The dense kernel must have found some
    // vorticity for the comparison to mean anything.
    return { ok: mismatched === 0 && nonZero > 0, checked: n, mismatched, nonZero, maxAbs, first };
  }

  // U4-2: the root's FORCE against the dense kernel's, exactly.
  //
  // Runs BOTH passes on demand into scratch accumulators -- the solver
  // dispatches neither (the dense one is unreachable, the root one is inert),
  // so there is nothing to read after a step and this has to produce its own
  // numbers.
  //
  // THE BAR IS THE RAW i32 ACCUMULATORS, NOT A DERIVED FORCE. The reduction
  // atomically adds ONE TRUNCATED i32 PER WORKGROUP (FSCALE = 1e7), so the
  // partition matters: the dense kernel is one workgroup per 8x8 dense block,
  // and the root at GHOST=0 dispatches (2,2) over a 16-cell tile, which is the
  // same four 8x8 regions of the domain. Same partition, same partials, and
  // integer atomicAdd is associative and commutative -- so the sum is
  // order-independent and exact equality is the right bar rather than an
  // ambitious one. If the partitions ever diverge, this is where it shows.
  async function debugCheckRootForce() {
    if (!rootForcePL) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const zero = new Int32Array(4);
    device.queue.writeBuffer(rootForceBuf, 0, zero);
    device.queue.writeBuffer(denseForceBuf, 0, zero);
    const enc = device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      p.setPipeline(frcPL); p.setBindGroup(0, denseForceBG);
      p.dispatchWorkgroups(WGX, WGY); p.end();
    }
    {
      const p = enc.beginComputePass();
      p.setPipeline(rootForcePL); p.setBindGroup(0, rootForceBG);
      // (2,2): a ring-free tile is 2*RB = 16 cells, so four 8x8 workgroups --
      // the dense kernel's own partition. NOT WGX1/WGY1, which is sized for
      // FB = 20 and would dispatch a third, empty row and column.
      p.dispatchWorkgroups(2, 2, pools[0].MAX_FINE_BLOCKS); p.end();
    }
    const sD = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const sR = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(denseForceBuf, 0, sD, 0, 16);
    enc.copyBufferToBuffer(rootForceBuf, 0, sR, 0, 16);
    device.queue.submit([enc.finish()]);
    await Promise.all([sD.mapAsync(GPUMapMode.READ), sR.mapAsync(GPUMapMode.READ)]);
    const d = Array.from(new Int32Array(sD.getMappedRange()).slice(0, 3));
    const r = Array.from(new Int32Array(sR.getMappedRange()).slice(0, 3));
    sD.unmap(); sR.unmap(); sD.destroy(); sR.destroy();
    // VACUITY GUARD: two zero accumulators agree perfectly. The body has to be
    // in the fluid for this to mean anything, and on a falling card early in a
    // run it always is -- but "always" is what a guard is for.
    const nonZero = d.some(v => v !== 0);
    const diff = [0, 1, 2].map(i => r[i] - d[i]);
    // `exact` is reported, NOT gated. The caller owns the bound, because the
    // bound is only defensible next to a measured defect scale and that
    // measurement lives with the tool. See tools/validate-root-kernels.js.
    return {
      exact: diff.every(v => v === 0), nonZero,
      dense: d, root: r, diff, maxDiff: Math.max(...diff.map(Math.abs)),
    };
  }

  // U4-3: the root's DIGEST against the dense grid's, on the one component
  // that can be exact. See digestFullPL above for why only max qualifies.
  async function debugCheckRootDigest() {
    if (!digestFullPL) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const enc = device.createCommandEncoder();
    for (const [pl, bg] of [[digestFullPL, digestDenseFullBG], [digestFullPL, digestRootFullBG]]) {
      const p = enc.beginComputePass();
      p.setPipeline(pl); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end();
    }
    const sD = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const sR = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(digestDenseFullBuf, 0, sD, 0, 16);
    enc.copyBufferToBuffer(digestRootFullBuf, 0, sR, 0, 16);
    device.queue.submit([enc.finish()]);
    await Promise.all([sD.mapAsync(GPUMapMode.READ), sR.mapAsync(GPUMapMode.READ)]);
    const dw = new Uint32Array(sD.getMappedRange()).slice();
    const rw = new Uint32Array(sR.getMappedRange()).slice();
    const d = new Float32Array(dw.buffer), r = new Float32Array(rw.buffer);
    sD.unmap(); sR.unmap(); sD.destroy(); sR.destroy();
    // WORD equality on the max, so it cannot be softened by a tolerance.
    const maxExact = dw[2] === rw[2];
    const cellsMatch = dw[3] === rw[3];
    // The sums are order-dependent; report how far apart, as a sanity check
    // that the two really are digests of the same field and not of two fields.
    const relSum = Math.abs(r[0] - d[0]) / Math.max(Math.abs(d[0]), 1e-12);
    const relSq = Math.abs(r[1] - d[1]) / Math.max(Math.abs(d[1]), 1e-12);
    return {
      ok: maxExact && cellsMatch && d[2] > 0,
      maxExact, cellsMatch, maxU: d[2], rootMaxU: r[2], relSum, relSq,
      dense: Array.from(d), root: Array.from(r),
    };
  }

  // U4-4: the CONSERVED TOTALS, read off the root pool instead of the dense
  // grid -- and this one IS exactly equal, for a reason worth stating.
  //
  // readConservedTotals is already parameterised by a `cellIndex` callback: it
  // walks (x, y) in SPATIAL order and asks where that cell lives, then sums in
  // f64 on the host. So pointing it at the root pool changes the addressing and
  // NOTHING ELSE -- same values (U3), same order, same f64 reduction. Bit
  // equality is therefore the right bar here, where it was not for the digest
  // (whose GPU reduction order differs) or the force (whose per-cell arithmetic
  // differs). Three consumers, three different answers to "can this be exact",
  // and each one has a reason.
  async function debugCheckRootConserved() {
    if (!pools[0]) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
    const common = { W, H, ex: EX, ey: EY, decode: (m, n) => readF(m, n) };
    const dense = await readConservedTotals(device, {
      ...common, f: f_a, NCELLS,
      cellIndex: (x, y) => {
        const nbx = W / BLOCK;
        return (Math.floor(y / BLOCK) * nbx + Math.floor(x / BLOCK)) * BLOCK * BLOCK
             + (y % BLOCK) * BLOCK + (x % BLOCK);
      },
    });
    const root = await readConservedTotals(device, {
      ...common, f: pools[0].finePoolF_a, NCELLS: spec.cells,
      cellIndex: (x, y) => rootCellIndex({ dims: { W, H }, rb: RB }, x, y),
    });
    const keys = ['mass', 'momX', 'momY', 'rhoMin', 'rhoMax', 'maxU'];
    const diff = {};
    let exact = true;
    for (const k of keys) {
      if (!(k in dense)) continue;
      diff[k] = root[k] - dense[k];
      if (root[k] !== dense[k]) exact = false;
    }
    // VACUITY GUARD: two zero totals agree perfectly. rhoMax must have found a
    // real field, not an unwritten buffer.
    const live = Number.isFinite(dense.rhoMax) && dense.rhoMax > 0;
    return { ok: exact && live, exact, live, dense, root, diff };
  }

  // U5-1: level 1's ghost ring, interpolated from the ROOT POOL, against the
  // same ring interpolated from the dense L0 grid. EXACTLY.
  //
  // WHY BIT-IDENTITY IS THE RIGHT BAR AND NOT AN AMBITIOUS ONE. The two
  // accessors' `sampleParent` bodies are arithmetically the same statement --
  // same nine-term loop in the same order, same max(rho, 1e-6) floor, same
  // fneq -- and everything downstream of the fetch (fineToCoarseUnit, the
  // floor/frac split, interpCoarseToFine) is the SHARED kernel. Only the
  // address space moves, which is U3/U4's rule for when exactness survives.
  // The parent-local and dense-buffer coordinates differ by exactly the
  // parent root tile's integer origin, so even `tx`/`ty` are the same f32.
  //
  // THREE LEGS, AND THE THIRD IS THE ONE THAT MAKES THE FIRST MEAN ANYTHING.
  //
  //   dense   interpPL against a COPY of level 1's pool          the reference
  //   root    rootInterpPL, parent = the root pool               the subject
  //   stale   rootInterpPL, parent = the root's OTHER buffer     the control
  //
  // Both real legs start from a byte-identical copy of the live pool, so the
  // fine-fine consultation branch -- which reads the target buffer's own
  // INTERIOR and is indifferent to the parent -- resolves identically on each
  // and contributes guaranteed agreement. That is the vacuity risk here: if
  // every ring cell had an active same-level neighbour, the parent hop would
  // never run and `mismatched == 0` would be saying nothing at all. The stale
  // leg measures exactly that, by changing ONLY the parent: whatever it moves
  // is what takes the parent hop, and it must be a lot.
  //
  // `wrote` is the second guard, against the whole comparison running on a
  // pass that did nothing: it counts ring words the dense leg changed relative
  // to the seed.
  //
  // Scored over RING cells of ACTIVE slots only. Interiors are untouched by
  // both legs (GHOST_ONLY=1 returns early on them) and inactive slots by
  // neither, so including either would pad the denominator with words that
  // agree by construction -- and a rate diluted to meaninglessness is how a
  // checker stops being read.
  async function debugCheckRootInterp(maxReport = 8) {
    if (!rootInterpPL) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const pool = pools[1];
    const planeStride = pool.MAX_FINE_BLOCKS * NCELLS1;
    const nw = fWords(F16);

    // Seed a scratch from the live pool and run one leg into it. The seed is
    // what makes the legs comparable: the fine-fine branch reads this buffer.
    const leg = (dst, pl, bg) => {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(pool.finePoolF_a, 0, dst, 0, pool.fSizePool);
      const p = enc.beginComputePass();
      p.setPipeline(pl); p.setBindGroup(0, bg);
      p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS);
      p.end();
      device.queue.submit([enc.finish()]);
      return device.queue.onSubmittedWorkDone();
    };

    await leg(interpScratchDense, interpPL, denseInterpScratchBG);
    await leg(interpScratchRoot, rootInterpPL, rootInterpBG);
    const seed = await readBuf(pool.finePoolF_a, pool.fSizePool);
    const denseW = await readBuf(interpScratchDense, pool.fSizePool);
    const rootW = await readBuf(interpScratchRoot, pool.fSizePool);
    await leg(interpScratchRoot, rootInterpPL, rootInterpStaleBG);
    const staleW = await readBuf(interpScratchRoot, pool.fSizePool);
    const s2b = new Int32Array((await readBuf(pool.slotToBlockBuf, pool.MAX_FINE_BLOCKS * 4)).buffer);

    let checked = 0, mismatched = 0, wrote = 0, staleDiff = 0, activeSlots = 0;
    const first = [];
    for (let slot = 0; slot < pool.MAX_FINE_BLOCKS; slot++) {
      if (s2b[slot] < 0) continue;
      activeSlots++;
      for (let fy = 0; fy < FB; fy++) {
        for (let fx = 0; fx < FB; fx++) {
          const interior = fx >= GHOST && fx < GHOST + RB * 2 && fy >= GHOST && fy < GHOST + RB * 2;
          if (interior) continue;
          const cell = slot * NCELLS1 + fy * FB + fx;
          for (let wi = 0; wi < nw; wi++) {
            const i = wi * planeStride + cell;
            checked++;
            if (denseW[i] !== seed[i]) wrote++;
            if (denseW[i] !== staleW[i]) staleDiff++;
            if (denseW[i] === rootW[i]) continue;
            mismatched++;
            if (first.length < maxReport) {
              first.push({ slot, block: s2b[slot], fx, fy, word: wi, dense: denseW[i], root: rootW[i] });
            }
          }
        }
      }
    }
    // `wrote` IS REPORTED, NOT GATED, AND THAT DISTINCTION COST A RED SWEEP.
    // It counts ring words the dense leg changed relative to the seed, which
    // reads as a liveness guard right up until the pass it re-runs is ALREADY
    // LIVE in the macro-step: re-running an idempotent pass on a buffer that
    // already holds its output legitimately changes nothing, and `wrote == 0`
    // then means "the page is doing this correctly", not "the pass did
    // nothing". Measured on the U5-3 shipped-path rungs, where it inverted
    // both the gate and its control at once. `staleDiff` is the guard that
    // survives, because it changes an INPUT rather than looking for movement:
    // if the root leg wrote nothing at all, its stale twin would match the
    // dense leg and this would be zero.
    return {
      ok: mismatched === 0 && activeSlots > 0 && checked > 0 && staleDiff > 0,
      checked, mismatched, wrote, staleDiff, activeSlots, first,
    };
  }

  // U5-2: level 1 restricted into the ROOT POOL, against level 1 restricted
  // into the dense grid. EXACTLY.
  //
  // The reverse hop of debugCheckRootInterp, same three-leg shape, one
  // structural difference worth reading before the numbers.
  //
  // `?rootstep=0` CANNOT DISCRIMINATE THIS COLUMN, and that is a property of
  // restriction rather than a gap in the control. The average reads ONLY the
  // CHILD's populations and writes ONLY the parent -- the parent's prior
  // contents never enter the arithmetic -- so a stale root pool produces a
  // bit-identical result, and `validate-root-kernels.js`'s control rung reads
  // clean here while going dirty on all five of the others. That is the same
  // kind of abstention the starved-pool sweep gets from `field` and
  // `quadrants`: a control that reddens everything has not been shown to test
  // anything, and one that provably cannot redden a particular row should say
  // so rather than be quietly weakened until it does.
  //
  // So the liveness control changes the one input the restriction reads: the
  // third leg runs the root pipeline against level 1's OTHER ping-pong buffer,
  // which at rest holds the mid-macro-step state.
  //
  // Scored over the cells the restriction actually WRITES -- the L0 footprint
  // of the ACTIVE level-1 blocks, which is exactly RB*RB per active slot. The
  // rest of the domain is untouched by both legs and would agree by
  // construction.
  async function debugCheckRootAverage(maxReport = 8) {
    if (!rootAvgPL) return { ok: null, skipped: 'no root pool (?rootpool=1)' };
    const spec = rootPoolSpec({ dims: { W, H }, rb: RB });
    const nw = fWords(F16);
    const rootCells = spec.slots * spec.cellsPerSlot;

    const leg = (dst, src, pl, bg) => {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, dst, 0, fSize);
      const p = enc.beginComputePass();
      p.setPipeline(pl); p.setBindGroup(0, bg);
      p.dispatchWorkgroups(1, 1, pools[1].MAX_FINE_BLOCKS);
      p.end();
      device.queue.submit([enc.finish()]);
      return device.queue.onSubmittedWorkDone();
    };

    await leg(avgScratchDense, f_a, avgPL, denseAvgScratchBG);
    await leg(avgScratchRoot, pools[0].finePoolF_a, rootAvgPL, rootAvgBG_live);
    const seed = await readBuf(f_a, fSize);
    const denseW = await readBuf(avgScratchDense, fSize);
    const rootW = await readBuf(avgScratchRoot, fSize);
    await leg(avgScratchRoot, pools[0].finePoolF_a, rootAvgPL, rootAvgBG_stale);
    const staleW = await readBuf(avgScratchRoot, fSize);
    const blockSlot = new Int32Array((await readBuf(pools[1].blockSlotBuf, NBLOCKS * 4)).buffer);

    let checked = 0, mismatched = 0, wrote = 0, staleDiff = 0, activeBlocks = 0;
    for (let i = 0; i < NBLOCKS; i++) if (blockSlot[i] >= 0) activeBlocks++;
    const first = [];
    for (let slot = 0; slot < spec.slots; slot++) {
      for (let ly = 0; ly < spec.side; ly++) {
        for (let lx = 0; lx < spec.side; lx++) {
          // This root cell's own spatial position, hence which level-1 block
          // covers it. A root tile is 2*RB cells, i.e. exactly 2x2 level-1
          // blocks -- the quadrant relation U5 is built on, read here in the
          // direction that names the block.
          const gx = (slot % spec.nbx) * spec.side + lx;
          const gy = Math.floor(slot / spec.nbx) * spec.side + ly;
          if (blockSlot[Math.floor(gy / RB) * NBX + Math.floor(gx / RB)] < 0) continue;
          const denseCell = rootCellToDense({ dims: { W, H }, rb: RB }, slot, lx, ly);
          const rootCell = slot * spec.cellsPerSlot + ly * spec.side + lx;
          for (let wi = 0; wi < nw; wi++) {
            const d = wi * NCELLS + denseCell;
            const r = wi * rootCells + rootCell;
            checked++;
            if (denseW[d] !== seed[d]) wrote++;
            if (denseW[d] !== staleW[r]) staleDiff++;
            if (denseW[d] === rootW[r]) continue;
            mismatched++;
            if (first.length < maxReport) {
              first.push({ slot, lx, ly, gx, gy, word: wi, dense: denseW[d], root: rootW[r] });
            }
          }
        }
      }
    }
    // `wrote` reported, not gated -- see debugCheckRootInterp's note on why,
    // which this column is where it was actually measured.
    return {
      ok: mismatched === 0 && activeBlocks > 0 && checked > 0 && staleDiff > 0,
      checked, mismatched, wrote, staleDiff, activeBlocks, first,
    };
  }

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
      };
      enc.copyBufferToBuffer(pool.finePoolF_a, 0, st.f, 0, pool.fSizePool);
      enc.copyBufferToBuffer(pool.finePoolVel, 0, st.vel, 0, pool.MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
      enc.copyBufferToBuffer(pool.blockSlotBuf, 0, st.blockSlot, 0, pool.NBLOCKS * 4);
      enc.copyBufferToBuffer(pool.slotToBlockBuf, 0, st.slotToBlock, 0, pool.MAX_FINE_BLOCKS * 4);
      enc.copyBufferToBuffer(pool.parentSlotBuf, 0, st.parentSlot, 0, pool.MAX_FINE_BLOCKS * 4);
      enc.copyBufferToBuffer(pool.quadrantBuf, 0, st.quadrant, 0, pool.MAX_FINE_BLOCKS * 4);
      levelStaging.push(st);
    }

    device.queue.submit([enc.finish()]);
    const allBuffers = [stagingF, stagingVel, stagingCard, stagingFPool, stagingVelPool, stagingBlockSlot, stagingSlotToBlock];
    for (const st of levelStaging) allBuffers.push(st.f, st.vel, st.blockSlot, st.slotToBlock, st.parentSlot, st.quadrant);
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
      for (const b of [st.f, st.vel, st.blockSlot, st.slotToBlock, st.parentSlot, st.quadrant]) { b.unmap(); b.destroy(); }
      poolsOut.push({
        level: m, RB, GHOST, FB, MAX_FINE_BLOCKS: pool.MAX_FINE_BLOCKS, NBLOCKS: pool.NBLOCKS, NBX: pool.NBX, NBY: pool.NBY,
        blockSlot: blockSlotArr_m, slotToBlock: slotToBlockArr_m,
        parentSlot: parentSlotArr, quadrant: quadrantArr,
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
    // REFUSED RATHER THAN DEGRADED (found porting this path at U7-4b; it was
    // latent here before, under a flag no load-side gate exercises). Under
    // quad allocation level 1 carries a parentSlot/quadrant pair and a
    // QUAD-indexed free list, and the format records neither -- it only ever
    // saved those from level 2 up. Restoring would rebuild a block-indexed
    // free list over a quad pool, where slot `q` and quad `q` are different
    // things, and corrupt the pool with no thrown error. SAVE is deliberately
    // not refused: tools/measure-determinism.js fingerprints through it under
    // `--extra=rootpool=1`, and a hash of a consistent subset is still a
    // hash. Porting the format is U7-6's.
    if (ROOT_MANAGED) {
      throw new Error('debugSnapshotLoad: the snapshot format does not carry the root pool '
        + "or level 1's quad indirection yet (plans/uniform-levels.md U7-6) -- "
        + 'reload with ?rootmanage=0, or with ?rootpool=0');
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
    // U5-3: and the root pool, which is the same L0 in the other layout. The
    // snapshot format does not carry it yet (that is U7's); mirroring the
    // just-restored dense grid is exact by U2's proof, so a load lands both
    // representations in the same state rather than one.
    seedRootFromDense();
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

        const qc = quadCPU[m];
        qc.blockSlotCPU.set(snapPool.blockSlot);
        qc.slotToBlockCPU.set(snapPool.slotToBlock);
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

  // U7-3: WHAT each pass is, on the DEV page. The ORDER is amr2d-gpu.mjs's
  // makeScheduler, shared with the other four pages -- see its header for why
  // the seam is order vs. content, and why this page's content is the one that
  // legitimately differs: it carries the ?benchSkip= measurement twins, the
  // per-pass profiling labels, and the root pool's parallel passes, none of
  // which belong on a shipped page.
  const passes = {
    l0InterpIntoL1: (enc, useB) => {
      if (skipGroup('interp')) return;
      const p = beginPass(enc, 'L0->L1 interp');
      p.setPipeline(skipGroup('interp-noop') ? l1InterpNoopPL : l1InterpPL);
      p.setBindGroup(0, l1InterpBG(useB));
      p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end();
    },
    l0Step: (enc, useB) => {
      const s = beginPass(enc, 'L0 step');
      s.setPipeline(stepPL); s.setBindGroup(0, useB ? stepBG_ba : stepBG_ab);
      s.dispatchWorkgroups(WGX, WGY); s.end();
      // U3: the same step, on the root pool, in parallel -- see
      // makeRootPool's encodeRootStep.
      if (rootGpu) rootGpu.encodeRootStep(enc, useB);
    },
    l1AverageIntoL0: (enc, useB) => {
      if (skipGroup('avg')) return;
      const a = beginPass(enc, 'L1->L0 average');
      a.setPipeline(skipGroup('avg-noop') ? noopPLs.avg : avgPL);
      a.setBindGroup(0, useB ? avgBG_targetA : avgBG_targetB);
      a.dispatchWorkgroups(1, 1, MAX_FINE_BLOCKS); a.end();
      // U5-3: the SAME restriction into the root pool, in parallel -- see
      // makeRootPool's encodeRootAverage. Inside `avg`'s skip group, so
      // ?benchSkip=avg still isolates the step as the sole writer of either L0
      // representation, which validate-root-kernels.js relies on.
      if (rootGpu) rootGpu.encodeRootAverage(enc, useB);
    },
    interpIntoChild: (enc, level, readCur) => {
      if (skipGroup('interp')) return;
      const childPool = pools[level + 1];
      const bg = readCur === 'a' ? childPool.interpPoolParentBG_readA : childPool.interpPoolParentBG_readB;
      const p = beginPass(enc, `L${level}->L${level + 1} interp`);
      p.setPipeline(skipGroup('interp-noop') ? noopPLs.interpPool : interpPoolParentPL);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(WGX1, WGY1, childPool.MAX_FINE_BLOCKS); p.end();
    },
    averageFromChild: (enc, level, writeCur) => {
      if (skipGroup('avg')) return;
      const childPool = pools[level + 1];
      const bg = writeCur === 'a' ? childPool.avgPoolBG_targetA : childPool.avgPoolBG_targetB;
      const p = beginPass(enc, `L${level + 1}->L${level} average`);
      p.setPipeline(skipGroup('avg-noop') ? noopPLs.avgPool : avgPoolPL);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(1, 1, childPool.MAX_FINE_BLOCKS); p.end();
    },
    // NO LEVEL SPLIT SINCE B3-1: one kernel, one pipeline, every level.
    substep: (enc, level, readCur) => {
      if (skipGroup('step1')) return;
      const pool = pools[level];
      const bg = readCur === 'a' ? pool.step1BG_ab : pool.step1BG_ba;
      const legacyGhost = GHOST_COPY !== 0 || skipGroup('ghostcopy');
      const pl = skipGroup('step1-ring') ? ringPLs.step1 : (legacyGhost ? legacyGhostPLs.step1 : step1PL);
      const p = beginPass(enc, `L${level} step`);
      p.setPipeline(pl); p.setBindGroup(0, bg);
      p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
    },
    fineFineRefresh: (enc, level) => {
      const pool = pools[level];
      const p = beginPass(enc, level === 1 ? 'L1 fine-fine ghost' : `L${level} fine-fine ghost`);
      if (level === 1) { p.setPipeline(l1InterpFFPL); p.setBindGroup(0, l1InterpFFBG); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); }
      else { p.setPipeline(interpPoolParentFFPL); p.setBindGroup(0, pool.interpPoolParentFFBG_b); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); }
      p.end();
    },
  };
  const { S_Advance: S_AdvanceShared } = makeScheduler({
    nLevels: N_LEVELS,
    // The in-session bench configuration can turn the legacy path on as well
    // as the page-load flag, which is why this is a callback and not a value.
    ghostCopy: () => GHOST_COPY !== 0 || skipGroup('ghostcopy'),
    passes,
  });
  const S_Advance = (level, enc) => S_AdvanceShared(level, enc, useB);

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
  // ?benchSkip=force,interp etc. This is a MEASUREMENT MODE -- skipping
  // passes makes the physics wrong by construction. It exists to answer
  // "what does this group cost", nothing else.
  // Groups. The first set REMOVE a pass; the rest are instrument variants that
  // keep the pass but change what it does, so a share can be split further:
  //
  //   force, phy, step1, interp, avg   -- pass not encoded at all
  //   interp-noop, avg-noop            -- pass encoded and dispatched at full
  //                                       width, returns immediately
  //   step1-ring                       -- fine step over tile interior only
  //                                       (proxy for FB 20 -> 16)
  //   ghostcopy                        -- ADDS the legacy fine-fine ghost copy
  //                                       pass back and reverts the fine step
  //                                       to clamped streaming, so its share is
  //                                       NEGATIVE and its magnitude is what
  //                                       neighbour-addressed streaming buys
  //
  // The `ghost` and `ghost-noop` groups are gone with the pass they named --
  // measuring "skip the fine-fine copy" is meaningless now that the default
  // build never encodes it. `ghostcopy` asks the same question from the other
  // side. An old ?benchSkip=ghost is now REJECTED rather than silently
  // reporting ~0%, which is the whole point of enumerating these names.
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
  // which tools/bench-amr.js cannot drive and which gets one sweep per
  // session (see plans/perf-characterization.md on adb) -- the same class
  // of silent-failure trap as a sweep interrupted by backgrounding, which this
  // file already refuses to report quietly.
  const BENCH_GROUPS = new Set([
    'force', 'phy', 'step1', 'interp', 'avg',
    'interp-noop', 'avg-noop', 'step1-ring', 'ghostcopy',
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
    const frcBG = useB ? frcBG_b : frcBG_a;

    // Milestone 4b/9: re-evaluate refinement every REFINE_EVERY macro-steps,
    // now generalized across every configured level. Runs BEFORE S_Advance
    // below so anything refined this round gets its one-time full-slot
    // fill before anything else this macro-step reads its pool slot. Reads
    // each level's own velocity field as populated by the PREVIOUS macro-
    // step (level 1's finePoolVel, level>=2's own), i.e. the same
    // "current, pre-step" data the force passes also read.
    // U7-3: WHAT each pass of the refinement round is, on the DEV page. The
    // ORDER is amr2d-gpu.mjs's makeRefineRound, shared with the other four --
    // see its header. This page's content differs by the per-pass profiling
    // labels, the ?benchSkip= surface, D0's deterministic scan/link passes,
    // and the root's own criterion once it manages level 1.
    const refinePasses = {
      denseCriterion: (enc) => {
        const p = beginPass(enc, 'criterion L0'); p.setPipeline(criterionPL); p.setBindGroup(0, criterionBG); p.dispatchWorkgroups(WGX, WGY); p.end();
        // U4-1: the same decision from the root pool, into its own buffer.
        // Encoded immediately after the dense one so both read the SAME
        // velocity state -- a criterion compared across a step boundary would
        // differ for reasons that have nothing to do with the kernel. NOT
        // under ROOT_MANAGED: the pool criterion at parent level 0 is then the
        // LIVE writer, and running this too would make debugCheckRootCriterion
        // compare a kernel against its own output.
        if (rootCritPL && !ROOT_MANAGED) {
          const rc = beginPass(enc, 'criterion root'); rc.setPipeline(rootCritPL); rc.setBindGroup(0, rootCritBG);
          rc.dispatchWorkgroups(2, 2, pools[0].MAX_FINE_BLOCKS); rc.end();
        }
      },
      poolCriterion: (enc, m) => { const p = beginPass(enc, `criterion L${m}`); p.setPipeline(criterionPoolPLs[m]); p.setBindGroup(0, criterionPoolBGs[m]); p.dispatchWorkgroups(2, 2, pools[m].MAX_FINE_BLOCKS); p.end(); },
      denseDecide: (enc) => { const p = enc.beginComputePass(); p.setPipeline(manageDecidePL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end(); },
      poolDecide: (enc, m) => { const p = enc.beginComputePass(); p.setPipeline(managePoolDecidePLs[m]); p.setBindGroup(0, managePoolBGs[m]); p.dispatchWorkgroups(Math.ceil(pools[m].MAX_FINE_BLOCKS / 64)); p.end(); },
      denseCoarsen: (enc) => {
        // D0: the scan reads blockSlot as coarsen finds it, so it must run
        // BEFORE coarsen -- and after refine's own link pass from the previous
        // round, which is where blockSlot was last settled.
        if (DET_SLOTS) { const q = enc.beginComputePass(); q.setPipeline(manageScanReleasePL); q.setBindGroup(0, manageBG); q.dispatchWorkgroups(1); q.end(); }
        const p = enc.beginComputePass(); p.setPipeline(manageCoarsenPL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
        if (DET_SLOTS) { const q = enc.beginComputePass(); q.setPipeline(manageLinkCoarsenPL); q.setBindGroup(0, manageBG); q.dispatchWorkgroups(WG_MANAGE); q.end(); }
      },
      poolCoarsen: (enc, m) => { const p = enc.beginComputePass(); p.setPipeline(managePoolCoarsenPLs[m - 1]); p.setBindGroup(0, managePoolBGs[m - 1]); p.dispatchWorkgroups(Math.ceil(pools[m].MAX_FINE_BLOCKS / 64)); p.end(); },
      denseRefine: (enc) => {
        // After coarsen, so the blocks it released are already visible as
        // candidates for a grant in the same round.
        if (DET_SLOTS) { const q = enc.beginComputePass(); q.setPipeline(manageScanGrantPL); q.setBindGroup(0, manageBG); q.dispatchWorkgroups(1); q.end(); }
        const p = enc.beginComputePass(); p.setPipeline(manageRefinePL); p.setBindGroup(0, manageBG); p.dispatchWorkgroups(WG_MANAGE); p.end();
        if (DET_SLOTS) { const q = enc.beginComputePass(); q.setPipeline(manageLinkRefinePL); q.setBindGroup(0, manageBG); q.dispatchWorkgroups(Math.ceil(MAX_FINE_BLOCKS / 64)); q.end(); }
      },
      poolRefine: (enc, m) => { const p = enc.beginComputePass(); p.setPipeline(managePoolRefinePLs[m - 1]); p.setBindGroup(0, managePoolBGs[m - 1]); p.dispatchWorkgroups(Math.ceil(pools[m - 1].MAX_FINE_BLOCKS / 64)); p.end(); },
      l1InitFill: (enc) => { const p = beginPass(enc, 'L1 init fill'); p.setPipeline(l1InterpInitPL); p.setBindGroup(0, l1InterpInitBG(useB)); p.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); p.end(); },
      poolInitFill: (enc, m) => { const p = beginPass(enc, `L${m} init fill`); p.setPipeline(interpPoolParentInitPL); p.setBindGroup(0, pools[m].interpPoolParentBG_readA); p.dispatchWorkgroups(WGX1, WGY1, pools[m].MAX_FINE_BLOCKS); p.end(); },
    };
    if (autoRefine && macroStepCounter % REFINE_EVERY === 0) {
      makeRefineRound({
        nLevels: N_LEVELS, pools, cascade, encodeCascade,
        firstParentLevel: ROOT_MANAGED ? 0 : 1, passes: refinePasses,
      })(enc);
    }
    macroStepCounter++;

    // ONE FORCE PASS, AT THE FINEST LEVEL (plans/2D-backport.md B4-3).
    //
    // Milestone 8 ran EVERY level's force pass and had each mask itself out
    // where a finer level covered it, because `average` keeps a parent's
    // cells populated under an active child and summing them all would
    // double-count the same physical drag. The masking is gone: the body is
    // required to live entirely on the finest level (the geometry-forced
    // refinement hard constraint, asserted by debugCheckGeometryCoverage and
    // gated by tools/validate-amr-invariants.js), so every coarser pass was
    // contributing exactly zero -- measured, see amr_force.wgsl's header for
    // the per-level raw accumulator readings that established it before any
    // of this was deleted.
    //
    // A COVERAGE VIOLATION NOW SUBTRACTS RATHER THAN CORRUPTS. Before, a
    // body-adjacent block stuck at a coarse level contributed its own crude
    // force; now it contributes nothing at all. Neither is right and neither
    // is loud -- which is why B4's remaining work is to REFUSE rather than
    // degrade when the constraint cannot be met.
    const finestLevel = N_LEVELS - 1;
    if (!skipGroup('force')) {
      if (finestLevel === 0) {
        const frc = beginPass(enc, 'force L0'); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
      } else {
        // NO LEVEL SPLIT SINCE B3-4: one kernel, one pipeline, every level.
        const pool = pools[finestLevel];
        const p = beginPass(enc, `force L${finestLevel}`); p.setPipeline(force1PL); p.setBindGroup(0, pool.force1BG); p.dispatchWorkgroups(WGX1, WGY1, pool.MAX_FINE_BLOCKS); p.end();
      }
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
  // U5-4: level 1 joins this the moment it is quad-allocated. Its bare
  // blockSlotCPU/slotToBlockCPU/freeSlots mirror above stays -- the manual
  // debugActivateBlock path is the only thing that reads it, and that path
  // REFUSES under quad allocation (see its own note) rather than being
  // half-ported.
  const quadCPU = {};
  for (let c = (ROOT_MANAGED ? 1 : 2); c < N_LEVELS; c++) {
    quadCPU[c] = {
      blockSlotCPU: new Int32Array(pools[c].NBLOCKS).fill(-1),
      slotToBlockCPU: new Int32Array(pools[c].MAX_FINE_BLOCKS).fill(-1),
      freeQuads: Array.from({ length: pools[c].MAX_FINE_BLOCKS / 4 }, (_, i) => i),
    };
  }
  // This level's own blockSlotCPU mirror, whichever structure holds it --
  // level 1 uses the bare `blockSlotCPU` above, levels >=2 use quadCPU[c].
  function blockSlotCPUAtLevel(level) {
    return (level === 1 && !ROOT_MANAGED) ? blockSlotCPU : quadCPU[level].blockSlotCPU;
  }

  function resetSim() {
    writeF(f_a, initF(), NCELLS);
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    // Level 1's PER-BLOCK reset. Under quad allocation it is reset by the loop
    // below instead, like every other level -- U5-4's whole point is that
    // there stops being a level-1 case here. The CPU mirrors are still
    // cleared, so the refusing debugActivateBlock path cannot observe stale
    // state if it is ever re-enabled.
    blockSlotCPU.fill(-1);
    slotToBlockCPU.fill(-1);
    freeSlots = Array.from({ length: MAX_FINE_BLOCKS }, (_, i) => i);
    if (!ROOT_MANAGED) {
      writeF(pools[1].finePoolF_a, initFPool(), MAX_FINE_BLOCKS * NCELLS1);
      device.queue.writeBuffer(pools[1].blockSlotBuf, 0, blockSlotCPU);
      device.queue.writeBuffer(pools[1].slotToBlockBuf, 0, slotToBlockCPU);
      device.queue.writeBuffer(pools[1].freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
      device.queue.writeBuffer(pools[1].freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
    }
    // Milestone 6: levels >=2 reset the same way, at quad granularity -- and
    // level 1 too, once it is one of them (U5-4).
    for (let c = (ROOT_MANAGED ? 1 : 2); c < N_LEVELS; c++) {
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
    // U5-3: the root pool is L0 too, and reset() has to reach it. Ordered
    // after the writeBuffer above by the queue; `useB = false` above is the
    // phase the mirror writes (finePoolF_a).
    seedRootFromDense();
    step = 0;
    trajectory.length = 0;
    trail.clear();
    totals.reset();
    pacer.reset();
  }

  // Pool indirection readback -- amr2d-gpu.mjs, five copies before B3a.
  const readPoolIndirection = (level = 1) => readPoolIndirectionOn(device, pools, level);
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

  // Activates coarse block (bx,by) [0<=bx<NBX, 0<=by<NBY, buffer-space --
  // see plans/AMR.md's Milestone 4 design note on why block IDs are
  // buffer-space-native] against a free pool slot, filling the whole new
  // slot from the CURRENT coarse state (GHOST_ONLY=0 pipeline) since there
  // is no prior fine-level state for it to evolve from. Only valid while
  // liveMode is false, matching the debugSnapshotSave/Load convention --
  // dispatchMacroStep's useB toggling and this function's direct queue
  // writes would otherwise race the frame() loop's own encoder.
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

    // U5-4: this path is the PER-BLOCK allocator's, and it is refused rather
    // than half-ported once level 1 allocates in quads. A manual grant that
    // handed out one slot from a quad-indexed free list would corrupt the
    // pool silently -- the free list's entries are quad indices, and slot
    // `q` and quad `q` are different things. Levels >= 2 have always taken
    // the quad branch below; level 1 now joins them there, and the honest
    // version of this function for a quad pool is that branch, not this one.
    if (level === 1 && ROOT_MANAGED) {
      throw new Error('debugActivateBlock: level 1 is quad-allocated under ?rootmanage=1 -- '
        + 'use ?rootmanage=0 for the per-block manual path, or activate a level >= 2 block');
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

      const interpInitBG = l1InterpInitBG(useB);
      const enc = device.createCommandEncoder();
      const ipl = enc.beginComputePass();
      ipl.setPipeline(l1InterpInitPL);
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


    const slotsWritten = [];
    for (let qy = 0; qy <= 1; qy++) {
      for (let qx = 0; qx <= 1; qx++) {
        const quadrant = qx + 2 * qy;
        const slot = baseSlot + quadrant;
        const childBX = parentBX * 2 + qx, childBY = parentBY * 2 + qy;
        const childBlockID = childBY * pool.NBX + childBX;
        qc.blockSlotCPU[childBlockID] = slot;
        qc.slotToBlockCPU[slot] = childBlockID;
        device.queue.writeBuffer(pool.blockSlotBuf, childBlockID * 4, new Int32Array([slot]));
        device.queue.writeBuffer(pool.slotToBlockBuf, slot * 4, new Int32Array([childBlockID]));
        device.queue.writeBuffer(pool.parentSlotBuf, slot * 4, new Int32Array([parentSlotVal]));
        device.queue.writeBuffer(pool.quadrantBuf, slot * 4, new Uint32Array([quadrant]));
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
  // Raw per-block criterion for a level, straight off the GPU. Added while
  // chasing the near-wake artifact: the refine decision for level m+1 reads
  // pools[m+1].blockCriterionBuf, and reconstructing that value on the host
  // from a snapshot's velocity field only tells you what it SHOULD be. When
  // the two disagree, the criterion pass is at fault; when they agree, the
  // refine path is. Reading it is the only way to tell them apart.
  //
  // Indexed by that level's own block grid (pools[level].NBLOCKS), which is
  // the same indexing amr_manage_pool.wgsl's refine() uses for childCriterion
  // and amr_criterion_pool.wgsl uses when writing it.
  async function debugReadBlockCriterion(level = 2) {
    const pool = pools[level];
    if (!pool) throw new Error(`level ${level} has no pool (N_LEVELS=${N_LEVELS})`);
    const bytes = pool.NBLOCKS * 4;
    const stage = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(pool.blockCriterionBuf, 0, stage, 0, bytes);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const c = Array.from(new Float32Array(stage.getMappedRange()));
    stage.unmap();
    stage.destroy();
    return { level, NBX: pool.NBX, NBY: pool.NBY, criterion: c };
  }

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
  // amr2d-gpu.mjs. THIS PAGE HAD NONE, which is half of why it had no
  // geometry-coverage check either (plans/2D-backport.md B4).
  const debugReadCardState = () => readCardState(device, cardStateBuf);

  // GEOMETRY-FORCED REFINEMENT, THE HARD CONSTRAINT: every LEAF tile whose
  // footprint comes within that level's own FORCE_REFINE_MARGIN of the body
  // -- now, or FORCE_REFINE_LOOKAHEAD macro-steps out -- must already have a
  // child, unless it is already at the finest configured level. Read down the
  // levels it says the body lives entirely on the finest one.
  //
  // THIS PAGE MOVED A BODY THROUGH A REFINED REGION WITH NO SUCH GATE AT ALL.
  // main-cylinder-amr.js had the only implementation in the project, and
  // B3a-4 found that what looked like two more copies were `{ok: true}` stubs
  // on the two BODYLESS pages -- so the distribution was exactly backwards:
  // the check existed only where the body is PINNED, and was absent on both
  // pages where it moves. tools/lib/amr-invariants.js PROBES for this
  // function and reports SKIPPED when it is missing, precisely so a missing
  // check cannot look greener than a present one; this is that skip closed.
  const debugCheckGeometryCoverage = async () => checkGeometryCoverageOnGPU(device, pools, {
    nLevels: N_LEVELS, W, H, rb: RB, NBX, NBLOCKS,
    cardState: await debugReadCardState(),
    boxRefine: BOX_REFINE !== 0,
    paramsForChildLevel,
  });

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
        setStatus(statusEl, '[AMR-dev] benchmark sweep complete -- results sent');
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
  // phone cannot be driven the way tools/bench-amr.js drives the desktop (adb
  // does expose CDP, but debugStepSync over it killed Chrome -- see
  // plans/perf-characterization.md), so this sweep is the way to ask that
  // device the same questions.
  // ?benchConfigs=none,interp,avg,ghostcopy,interp+avg trims the list. Worth
  // using on a device you have to hold in your hand: every configuration costs
  // (BENCH_ROUNDS+1) * BENCH_MEASURE_MS, and a shorter sweep also spends less
  // of itself inside this device's own thermal ramp. 'none' is always kept --
  // every share is relative to it.
  const BENCH_CONFIGS_DEFAULT = ['none', 'force', 'phy', 'force+phy', 'interp', 'avg', 'ghostcopy', 'step1', 'interp+avg',
                                 'interp-noop', 'avg-noop', 'step1-ring'];
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
      setStatus(statusEl, `[AMR-dev] benchmark ${pct}% -- ${label} (do not switch away)`);
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
      // (`ghost` and `interp+avg+ghost` no longer exist as configurations --
      // the pass they skipped is gone; `ghostcopy` measures the same boundary
      // from the other side. The rows are kept because they are the only
      // published cross-check of this sweep against bench-amr.js.)
      //
      // The small rows do NOT resolve, on any run: interp-noop/avg-noop/
      // step1-ring are 0.2-5% effects by bench-amr and came back as 7.9%,
      // -3.6% and -12.2% here. The baseline itself carries ~15% spread because
      // the card keeps falling through the sweep while refinement is frozen,
      // so the workload drifts under every row equally. Read anything under
      // ~10% as "below the floor", not as a measurement -- and, for the
      // pass-REMOVING groups, a NEGATIVE share means exactly that, since
      // skipping work cannot make a run slower. `ghostcopy` is the one
      // configuration whose share is negative BY DESIGN (it adds work back),
      // so read its magnitude, not its sign, against the same ~10% floor. Use
      // tools/bench-amr.js for the small effects on any device that has a CDP
      // endpoint; this sweep exists for the one that does not.
      noiseFloorPct: 10,
    };
  }
  const readbackWatch = { lastStep: null, lastY: null, n: 0, stepBack: 0, worstStepBack: 0,
                          posJump: 0, fieldRepeat: 0, digests: [], samples: [],
                          lastOffX: null, lastOffY: null, offDirX: 0, offDirY: 0,
                          offReversals: 0, worstOffReversal: 0, offMaxStep: 0, offBackFrames: 0, offTrace: [], ring: [], diverged: false, divergedAtStep: null, history: null };

  let benchRunning = false;
  let benchDone = false;

  // ── ?diag=1 refinement convergence ───────────────────────────────────────
  // Reads AND ZEROES, so successive calls give per-interval counts rather than
  // a running total -- which is what makes "transient while refinement catches
  // up" vs "steady state" answerable.
  //
  // WHAT THESE MEANT AND WHAT THEY MEAN NOW. They described the FINAL
  // iteration of a fixed-point loop, and `converged` asked whether the 2:1
  // cascade had stopped propagating when that loop ran out of iterations.
  // B2 deleted the loop: the closure reaches its fixed point in one sweep by
  // construction, so there is no iteration budget to run out of and no
  // cascade-still-spreading state to detect.
  //
  // SO `converged` AND `refineByCascadeLastIter` ARE GONE rather than left
  // reading OK. diag[4] stopped being written at all in B2-2d, which made
  // `converged`'s first clause permanently true -- an always-true gate, the
  // fourth this project has found (B3a-4 had two, B4-3 one). What was left of
  // it was the pool-starvation half, so that is what it is now called.
  //
  // refineGranted is the tiles created in THIS round -- churn, not a fault.
  // refineStarved is refines refused for want of a slot, which IS a fault:
  // geometry-forced refinement being denied means a seam through the body,
  // and since B4-3 only the finest level computes force, so the refused
  // region contributes nothing at all. The live loop latches on it
  // (makeRefusalWatch); this is how the HARNESS path sees it, since
  // debugStepSync does not go through frame().
  async function debugReadDiag() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(diagBuf, 0, diagReadBuf, 0, 32);
    device.queue.submit([enc.finish()]);
    await diagReadBuf.mapAsync(GPUMapMode.READ);
    const v = Array.from(new Uint32Array(diagReadBuf.getMappedRange().slice(0)));
    diagReadBuf.unmap();
    device.queue.writeBuffer(diagBuf, 0, new Uint32Array(8));
    return {
      diagEnabled: DIAG !== 0,
      refineGranted: v[3],
      refineStarved: v[5],
      // The one thing here that is still a fault. NOT `granted === 0`: a
      // criterion-driven grant is normal operation (a block whose own
      // vorticity newly crossed threshold), and gating on it would turn a
      // healthy config red -- measured on main, amr-N2-bounceback reported
      // granted=1 at step 2048 with 2:1 balance passing at that same
      // checkpoint.
      poolOk: v[5] === 0,
    };
  }

  // ── The scene render, as ONE encoder path ────────────────────────────────
  // frame() and debugRenderOnce() both go through this. A second copy of the
  // pass descriptor is exactly the kind of duplication that lets a render
  // change land in the live path and not in the verification path -- which is
  // the failure mode tools/validate-render-levels.js exists to catch, so it
  // would be perverse for the gate to run against its own copy of the render.
  //
  // The TRAIL overlay is deliberately NOT here. It is a separate canvas with
  // its own submit and its own state, it advances with the card rather than
  // with the field, and including it would put a second moving part into a
  // screenshot the gate compares for exact equality.
  function encodeSceneRender(enc) {
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
    rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();
  }

  // Render one frame on demand, without stepping the simulation.
  //
  // frame() returns before the render pass whenever liveMode is false (see its
  // own early return), so a PAUSED page never redraws -- which means anything
  // that changes a buffer the renderer reads and then wants to see the result
  // has no way to ask for it. That is not a niche need: it is the only way to
  // score "does this level's data reach the picture at all" without letting
  // the solver overwrite the thing under test between the write and the draw.
  async function debugRenderOnce() {
    const enc = device.createCommandEncoder();
    encodeSceneRender(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // Overwrite EVERY cell of one level's velocity pool with a constant.
  //
  // The instrument for tools/validate-render-levels.js. The value is meant to
  // be far outside any physical velocity this solver produces, so that a
  // renderer which samples this level cannot fail to change colour -- the test
  // is "is there a path from this buffer to the picture", and a perturbation
  // small enough to be swallowed by the colour map would answer the wrong
  // question.
  //
  // Whole pool, not just the active slots: an inactive slot is not drawn, so
  // including it cannot create a false positive, and excluding it would need a
  // readback of the indirection this hook has no other reason to do.
  //
  // Restore by debugSnapshotLoad -- finePoolVel carries COPY_DST for exactly
  // that path (see allocLevelPool), and the gate takes its snapshot first.
  function debugPerturbLevelVel(level, ux, uy) {
    const pool = pools[level];
    if (!pool) throw new Error(`debugPerturbLevelVel: no pool at level ${level} (N_LEVELS=${N_LEVELS}, pools 1..${N_LEVELS - 1})`);
    const cells = pool.MAX_FINE_BLOCKS * NCELLS1;
    const a = new Float32Array(cells * 2);
    for (let i = 0; i < cells; i++) { a[2 * i] = ux; a[2 * i + 1] = uy; }
    device.queue.writeBuffer(pool.finePoolVel, 0, a.buffer, a.byteOffset, a.byteLength);
    return { level, cells };
  }

  window.__AMR = {
    debugReadDiag,
    runBenchSweep,
    hasTimestamp: () => hasTimestamp,
    debugProfileMacroStep,
    // reset() on resume: frame() returns early while paused without ever
    // reaching the pacer, so its last-timestamp would otherwise be stale by
    // the whole pause. (MAX_FRAME_DT_MS and the accumulator cap already
    // bound the damage to one frame; this makes it exactly zero.)
    setLive: (v) => { liveMode = !!v; if (liveMode) pacer.reset(); },
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
    debugCheckRootPool: () => checkRootPoolIdentity(device, pools),
    debugMirrorRoot,
    debugCheckRootMirror,
    debugCheckRootVel,
    debugCheckRootCriterion,
    debugCheckRootForce,
    debugCheckRootDigest,
    debugCheckRootConserved,
    debugCheckRootInterp,
    debugCheckRootAverage,
    getRootPool: () => (ROOT_POOL ? { ...rootPoolSpec({ dims: { W, H }, rb: RB }), stepped: !!ROOT_STEP } : null),
    debugRenderOnce,
    debugPerturbLevelVel,
    debugCheck21Balance,
    debugCheckRefinementClosure,
    debugCheckSlotQuadrants,
    debugCascadeRoundTrip,
    debugCheckGeometryCoverage,
    debugReadCardState,
    debugProbeGhostFill,
    debugRunSteadyGhostFill,
    debugReadPool,
    debugReadBlockCriterion,
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
    getDetSlots: () => DET_SLOTS,
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

  // GEOMETRY-FORCED REFINEMENT REFUSED -> stop, loudly. amr2d-gpu.mjs's
  // makeRefusalWatch explains the trip-wire and why the authority is the
  // coverage check; this is the "stop advancing" half. Latched: once set it
  // never clears, so the rAF chain ends here and the status line is not
  // overwritten by the next frame's own readout.
  const refusalWatch = makeRefusalWatch({
    device, pools, nLevels: N_LEVELS,
    checkCoverage: () => debugCheckGeometryCoverage(),
  });

  async function frame() {
    try {
      if (refusalWatch.error) { handleErr(new Error(refusalWatch.error)); return; }
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

      // Paced, not fixed: however many steps this frame's wall-clock interval
      // is worth, capped at STEPS_PER_FRAME. Always even (sim-rate.mjs), which
      // the useB ping-pong invariant above depends on.
      const nSteps = pacer.stepsForFrame(performance.now(), A / U_T);
      for (let s = 0; s < nSteps; s++) dispatchMacroStep(enc);
      step += nSteps;

      if (hasTimestamp) {
        const t1 = enc.beginComputePass({ timestampWrites: { querySet, endOfPassWriteIndex: 1 } });
        t1.end();
        enc.resolveQuerySet(querySet, 0, 2, queryResolveBuffer, 0);
        enc.copyBufferToBuffer(queryResolveBuffer, 0, stage.query, 0, 16);
      }

      encodeSceneRender(enc);
      trail.draw(2 * A, trailOpacity);

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
      // Steps THIS frame actually dispatched -- the MLUPS readout divides
      // the GPU span by it, and it is no longer a constant.
      stage.steps = nSteps;

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
        // d[21]/d[20] are the WRAPPED accumulators (see card-total.mjs).
        // Unwrap FIRST -- before the backward-jump watchdog below, which
        // would otherwise read each wrap as a stale readback.
        const { x: xTotal, y: yTotal } = totals.unwrap(d[21], d[20]);
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
          if (!Number.isFinite(yTotal) || !Number.isFinite(d[4]) || !Number.isFinite(d[7])) {
            if (!readbackWatch.diverged) {
              readbackWatch.diverged = true;
              readbackWatch.divergedAtStep = st.step;
              // The run-up is the diagnostic, not the NaN itself.
              readbackWatch.history = readbackWatch.ring.slice();
            }
          } else {
            const dy = Math.abs(yTotal - readbackWatch.lastY);
            if (readbackWatch.lastY !== null && dy > 50) {
              readbackWatch.posJump++;
              if (readbackWatch.samples.length < 12) {
                readbackWatch.samples.push({ kind: 'y', from: +readbackWatch.lastY.toFixed(2), to: +yTotal.toFixed(2), atStep: st.step });
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
            s: st.step, y: +Number(yTotal).toFixed(2), vy: +Number(d[4]).toFixed(5),
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
        readbackWatch.lastY = yTotal;
        readbackWatch.n++;

        if (st.step < 100000) {
          trajectory.push([st.step, d[0], yTotal, xTotal, d[2], d[3], d[4], d[5], d[6], d[7], d[8]]);
        }
        // The card's UNWRAPPED path. The wrapped cx/cy cannot be used: they
        // never leave the buffer centre.
        trail.push(xTotal, yTotal, 2 * A);

        if (performance.now() - lastT > 250) {
          // L0 cells only -- it deliberately ignores every fine level, so it
          // is a coarse-grid-throughput figure, NOT total work done, and is
          // not comparable across level counts. tools/bench-amr.js computes
          // the honest cell-updates/s using live per-level active counts.
          const mlups = (NCELLS * (st.steps || 0)) / (gpuTime * 1e3);
          mlupsEl.textContent = mlups.toFixed(1);
          gpuMsEl.textContent = gpuTime.toFixed(2);
          syncMsEl.textContent = (performance.now() - tSubmit).toFixed(2);
          telemetrySample(gpuTime, performance.now() - tSubmit, st.step);
          // Not while a benchmark sweep owns the status line: this runs on
          // every readback and silently overwrote the sweep's own progress
          // messages within a frame, so "benchmark round 2/3" was never
          // actually visible to anyone asked to watch for it.
          if (!benchRunning) {
            setStatus(statusEl, `[AMR-dev] step ${st.step}  y=${yTotal.toFixed(1)}  x=${xTotal.toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`);
          }
          lastT = performance.now();
        }

        st.card.unmap();
        st.inFlight = false;
      };

      processReadback(stage);
      // Self-throttled and fire-and-forget: at most one readback in flight
      // and at most one every 500ms, so this costs nothing per frame. It
      // only reads 4 bytes per level unless a pool is actually saturated.
      refusalWatch.poll();

      currentStageIdx = (currentStageIdx + 1) % STAGES;
      requestAnimationFrame(() => frame().catch(handleErr));
    } catch (e) {
      handleErr(e);
    }
  }
  frame().catch(handleErr);
}
init().catch(handleErr);
