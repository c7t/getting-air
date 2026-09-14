// Dense 3D LBM page -- plans/3D.md M1. D3Q19 by default, D3Q27 via ?q=27,
// no AMR, no body, axis-aligned slice view.
//
// ONE PAGE, THREE SCENARIOS, deliberately. The 2D side of this project has
// index.html / index-cylinder.html / index-channel.html / index-tgv.html,
// each with its own main*.js carrying its own copy of the same bind
// groups -- and CLAUDE.md records what that cost: commit 238e48c changed a
// shared shader's binding count, updated one copy of the bind group, missed
// the other, and broke a production page that no test had ever loaded. A 3D
// fork doubles that surface (plans/3D.md sec 6). So the scenario -- duct,
// beltrami, tgv -- is a URL parameter over ONE set of pipelines and ONE set
// of bind groups, and it differs only in the initial condition, the wall
// flags and the body force, all of which come from d3-scenarios.mjs. Both
// analytic gates and the interactive view therefore exercise the identical
// code path, which is the property the 2D layout does not have.
//
// URL parameters:
//   ?scenario=beltrami|duct|tgv   default beltrami
//   ?n=48        cube edge; for ?scenario=sphere it is the sphere DIAMETER
//                in cells and the domain is sized from it
//   ?re=         sphere Reynolds number (tau follows from it)
//   ?bounceback  sharp momentum-exchange coupling instead of diffuse chi
//   ?levels=2    turn on one fine AMR level (default 1 = dense only)
//   ?rb=4        coarse cells per block edge; FB = 2*RB + 2*GHOST
//   ?refine=     STATIC refinement region. `body` (default when there is
//                one) refines blocks within ?margin= coarse cells of the
//                surface; `box` refines a centred cube of ?boxfrac= of the
//                domain; `slab` refines a ?boxfrac= band in x spanning all
//                of y and z, so the seam is two FLAT faces with no edge or
//                corner anywhere -- the control for interface work, see
//                plans/3D.md M4; `all` refines everything (the noise floor
//                -- no coarse/fine interface anywhere).
//   ?margin=2  ?boxfrac=0.5
//   ?interface=  `explode` (DEFAULT since M4.1e) is Chen et al. 2006's
//                explode/coalesce: exactly conservative in mass AND momentum
//                on every rung of the geometry ladder (all/slab/bar/box),
//                field error within 8-10% of the no-interface control, and
//                a sphere in a refined shell reproduces the DENSE run's Cd
//                to 0.07%. `interp` is the M3 trilinear-plus-Dupuis-Chopard
//                coupling it replaced: NOT conservative and 3.6x worse on
//                the field, kept switchable so the conversion can be A/B'd
//                in one build. plans/3D.md M4.1b-e.
//   ?reflux=1    OPT-IN coarse/fine interface flux correction (M4). Makes
//                the interface exactly conservative in mass and momentum,
//                and on a seam with no convex corner (?refine=slab) halves
//                the field error. On one WITH a corner (?refine=box, and
//                any body-fitted shell) it is much WORSE than leaving it
//                off. Not the default for that reason -- plans/3D.md M4.
//   ?explin=0    M4.1c: restore M4.1b's UNIFORM explosion. Default 1 is the
//                linear one -- Ndot_i^f = Ndot_i^c + (r_f - r_c).F_i, with
//                F_i a central difference taken only on axes with two real
//                coarse neighbours and projected orthogonal to c_i. Only
//                meaningful with ?interface=explode.
//   ?orphans=0   TIMING ONLY: drop the coalesce orphan pass. This makes the
//                interface WRONG (it reinstates M4.1b's convex-edge mass
//                leak); it exists so tools/bench-d3-interface.js can price
//                the pass by differencing two runs of one build.
//   ?dcpre=1     restore the PRE-collision Dupuis-Chopard fneq factor at the
//                coarse/fine transfers. That is wrong for this solver's
//                post-collision buffers and was the M3 interface bug; the
//                knob exists so the defect can be re-measured in the shipped
//                build rather than reconstructed from a branch, exactly as
//                ?f16= and ?ghostcopy= do on the 2D side. Not a mode to run
//                physics in. See shaders/common_d3_pool.wgsl.
//   ?q=19|27     velocity set, default 19
//   ?tau=0.8     BGK relaxation time; nu = (tau - 1/2)/3
//   ?u0=         scenario amplitude (duct: target peak velocity)
//   ?axis=x|y|z  slice normal, default z      ?slice=   index along it
//   ?rotate=     quarter turns of the PICTURE, in degrees clockwise
//                (0|90|180|270). Defaults per scenario so that the
//                scenario's own `down` runs down the window -- `fall`
//                declares +x, so its z-slice turns a quarter and the body
//                falls instead of tracking sideways. An image rotation and
//                nothing more: |u|, the out-of-plane vorticity and the
//                normal velocity are all invariant inside the slice plane,
//                so no sign can move.
//   ?mode=       speed | vorticity | normal (or 0 | 1 | 2). `vorticity` is
//                the out-of-plane component; `normal` is the velocity along
//                the slice normal, which is what shows a duct's u_x profile.
//   ?uscale= ?vscale= ?vortGamma=   render normalization overrides
//   ?view=       slice (default) | volume. The volume view is the octree
//                raymarcher (plans/3D.md M6.3/M6.4): one dense volume per
//                LEVEL over that level's own bounding box, innermost box
//                wins per ray sample. Needs volumes, so it turns ?vol= on.
//   ?vol=N       a multiple of each level's OWN resolution: 1 gives every
//                level a volume matching its grid exactly. 0 (the slice
//                view's default) builds nothing.
//   ?volstack=0  one volume only, which is the single-resolution M6.3
//                render -- the CONTROL M6.4b's image difference is against.
//   ?volbox=     `domain` (default) or `refined`, the M6.1 single-volume
//                selector. Implies ?volstack=0.
//   ?volBudget=  MiB over the whole stack, default 256; REFUSES past it.
//   ?volMargin=  blocks of slack around a refined box (default 2 dynamic,
//                0 static), so a box that follows a body has room to move.
//   ?volfallback= 1 (default) uses a refined box only where its voxel came
//                from that level, falling back outward elsewhere; 0 restores
//                innermost-box-wins-unconditionally. See VOL_FALLBACK.
//   ?volsync=    1 (default) applies every refined box's origin and its ray
//                placement with no await between them; 0 restores the
//                pre-M6.4e staggered apply, for the A/B.
//   ?volh=       1 (default) differences each voxel over ITS OWN SOURCE
//                CELL; 0 restores the pre-M6.4d fixed one-voxel stride, for
//                the A/B. See d3_volume_scalar.wgsl's VOL_HSRC.
//   ?volfield=   q (default) | omega | speed | rho -- which channel of the
//                scalar volume the transfer function eats. All four are
//                filled by one pass, so this is a uniform, not a recompute.
//   ?volIso= ?volGain= ?volOpacity= ?volGamma= ?volStep= ?volSteps=
//                the transfer function and the march. Iso and gain are
//                DIMENSIONLESS (the scalar volume is normalized), and the Q
//                default is d3-criterion.mjs's measured threshold.
//   ?azim= ?elev= ?dist= ?fov= ?camtarget=body|domain
//                the orbit camera, in degrees and in units of the domain's
//                largest extent. Drag to orbit, wheel to zoom.
//   ?spf=        solver steps per displayed frame
//   ?live=0      start paused (validation drives debugStepSync instead)

import { reportFatal, reportNoWebGPU, reportNoAdapter } from './error-overlay.mjs';
import { assembleShader } from './shader-loader.mjs';
import { SUPPORTED_Q } from './lattice-3d.mjs';
import { SCENARIOS, SCENARIO_NAMES, resolveScenario, nuFromTau, beltramiVelocityAt } from './d3-scenarios.mjs';
import { packBodyState, unpackBodyState, BODY_FIELDS, sdfBody, qRotateInv,
         bodyCircumradius } from './d3-body.mjs';
import { parseWindowAxes, wrapDims, wrapDelta3, windowOffset3, windowCoord3,
         windowConstants } from './d3-window.mjs';
import { Q_THRESHOLD, gradU, qOfGrad, checkFieldCoverage } from './d3-criterion.mjs';
import { volumeStack, levelVolume as volLevelVolume, boxRatio as volBoxRatio } from './d3-volume.mjs';
import { makePool, refineHierarchy, nearBodyWant, storageRatio, GHOST,
         check21Balance, checkGeometryCoverage, checkRingParentCoverage,
         cellAtLevel, finestLevelAt } from './d3-amr.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
// `Number.isFinite`, NOT `|| d`. ZERO IS FALSY, so the old form silently
// returned the DEFAULT for every `?x=0` -- and 0 is a meaningful value for
// most of these (?qlead=0 removes the convection lead, ?slotWarn=0 silences
// the budget warning, ?margin=0, ?manageStart=0). It cost a measurement: a
// ?qlead=0 control leg came back byte-identical to the default leg, which
// reads exactly like "the term does nothing" and was in fact "the parameter
// never arrived". This file's own header names that failure -- a parameter
// silently dropped is worse than one rejected -- and this was an instance of
// it hiding in the parser.
const numParam = (k, d) => {
  if (!urlParams.has(k)) return d;
  const v = parseFloat(urlParams.get(k));
  return Number.isFinite(v) ? v : d;
};
const AXIS_NAMES = ['x', 'y', 'z'];

// Parsers that can REJECT live in functions, not module-scope consts: a
// throw at module scope happens before init()'s .catch() is attached, and
// tools/validate-all.js's boot smoke would then report "page may be stuck"
// instead of the bad-parameter error it actually is.
function parseScenarioName() {
  const v = urlParams.get('scenario') || 'beltrami';
  if (!SCENARIO_NAMES.includes(v)) throw new Error(`?scenario=${v}: expected one of ${SCENARIO_NAMES.join(', ')}`);
  return v;
}
function parseQ() {
  const q = urlParams.has('q') ? parseInt(urlParams.get('q')) : 19;
  if (!SUPPORTED_Q.includes(q)) throw new Error(`?q=${urlParams.get('q')}: expected one of ${SUPPORTED_Q.join(', ')}`);
  return q;
}
// Both forms accepted, on purpose: the UI's <select> carries numeric
// values, the URL reads better as a letter, and a validation config is
// clearer with a name. Rejecting one of them buys nothing and is a papercut
// whichever one you happen to have in hand.
function parseAxis() {
  const v = (urlParams.get('axis') || 'z').toLowerCase();
  const i = /^[0-2]$/.test(v) ? parseInt(v) : AXIS_NAMES.indexOf(v);
  if (i < 0) throw new Error(`?axis=${v}: expected x, y, z, or 0, 1, 2`);
  return i;
}

// Degrees clockwise, and only the four that are lattice-aligned: anything
// else would need interpolation across the slice and is a different feature.
function parseRotate() {
  const v = urlParams.get('rotate');
  const deg = parseFloat(v);
  if (!Number.isFinite(deg) || deg % 90 !== 0) {
    throw new Error(`?rotate=${v}: expected 0, 90, 180 or 270 (degrees clockwise)`);
  }
  return (((deg / 90) % 4) + 4) % 4;
}

const MODE_NAMES = ['speed', 'vorticity', 'normal'];
// The default depends on the scenario, and has to: a duct's flow is
// unidirectional, so an x-normal slice has no in-plane velocity and
// therefore no out-of-plane vorticity either -- defaulting it to
// `vorticity` would render a correct simulation as a black canvas, which
// is the exact signal this project treats as a FAILURE (see the
// webgpu-verify skill). `normal` shows its u_x profile instead.
function defaultMode(scenarioName) { return scenarioName === 'duct' ? 'normal' : 'vorticity'; }
function parseMode(scenarioName) {
  const v = (urlParams.get('mode') || defaultMode(scenarioName)).toLowerCase();
  const i = /^[0-2]$/.test(v) ? parseInt(v) : MODE_NAMES.indexOf(v);
  if (i < 0) throw new Error(`?mode=${v}: expected ${MODE_NAMES.join(', ')}, or 0, 1, 2`);
  return i;
}

// SLICE OR VOLUME. A URL parameter and not only a UI control, because the
// volume view needs VOLUMES -- textures sized and filled at init -- and a
// page that allocated them for a view nobody asked for would be paying the
// M6 ballpark's memory for a picture it is not drawing. ?vol= turns them on
// independently, which is what lets the M6.1 gate run them under the slice
// view; the <select> is enabled exactly when they exist.
const VIEW_NAMES = ['slice', 'volume'];
function parseView() {
  const v = (urlParams.get('view') || 'slice').toLowerCase();
  if (!VIEW_NAMES.includes(v)) throw new Error(`?view=${v}: expected ${VIEW_NAMES.join(' or ')}`);
  return v;
}

// Stride that keeps a full-field readback to a manageable number of points
// while still sampling the whole volume. Must DIVIDE N, so the sampled
// sites are exact lattice sites and the analytic reference can be evaluated
// at them with no interpolation.
function subsampleStride(N, maxPerAxis) {
  for (let s = Math.max(1, Math.ceil(N / maxPerAxis)); s <= N; s++) if (N % s === 0) return s;
  return N;
}

async function init() {
  const scenarioName = parseScenarioName();
  const Q = parseQ();
  const VIEW = parseView();
  // Every scenario-level knob is forwarded, and the list is derived from the
  // scenario's own defaults rather than hardcoded here. A parameter the page
  // silently DROPPED would be worse than one it rejected: `?re=20` was
  // ignored for exactly this reason during development, so a sweep that
  // believed it was measuring Re=20 was measuring Re=100 and reporting the
  // wrong reference alongside it. A validation harness that lies is worse
  // than no harness.
  const overrides = {};
  for (const k of Object.keys(SCENARIOS[scenarioName].defaults)) {
    if (!urlParams.has(k)) continue;
    const v = parseFloat(urlParams.get(k));
    if (!Number.isFinite(v)) throw new Error(`?${k}=${urlParams.get(k)}: expected a number`);
    overrides[k] = v;
  }
  // A parameter that is not one of this scenario's knobs is a typo, and
  // silently ignoring it is the failure above wearing a different hat.
  const PAGE_PARAMS = new Set(['scenario', 'q', 'axis', 'slice', 'mode', 'spf', 'live',
    'uscale', 'vscale', 'vortGamma', 'bounceback', 'chiEps', 'vmax', 'omax', 'rotate',
    'levels', 'rb', 'refine', 'margin', 'boxfrac', 'dcpre', 'reflux', 'interface',
    'explin', 'orphans', 'dynamic', 'manageEvery', 'slotHeadroom', 'amrskip',
    'manageMargin', 'manageStart',
    // M6: the resample volume stack and the raymarcher.
    'vol', 'volbox', 'volBudget', 'volstack', 'volMargin', 'volh', 'volsync',
    'volfallback',
    'view', 'volfield', 'volIso', 'volGain', 'volOpacity', 'volGamma', 'volStep', 'volSteps',
    'azim', 'elev', 'dist', 'fov', 'camtarget', 'winbox', 'proj',
    // M8.2a: the solid-interior reset, ?solideq=0 to disable for A/B.
    'solideq',
    // D1: the swept-cell force term, ?swept=0 to disable for A/B.
    'swept',
    // D3: Ladd's wall density, ?rhow=0 to restore the pinned 1 for A/B.
    'rhow',
    // M8.2b: the fall scenario's two reference frames.
    'tow', 'stream',
    // M8.3: the moving window, per axis. ?window=x, ?window=0 to force it off.
    'window',
    // M8.4a: the hand-placed wake box's extent, in body diameters.
    'wake', 'wakeR',
    // M8.4: the Q-criterion's threshold (its PRESENCE turns the criterion on),
    // its convection lead in L0 cells, and the fraction of the slot budget at
    // which the page starts saying so.
    'qthresh', 'qlead', 'slotWarn', 'qinc', 'qhyst',
    // M8.4: an absolute per-level slot budget. A field criterion's set is not
    // predicted by the initial one, so ?slotHeadroom= cannot size it.
    'slots']);
  for (const k of urlParams.keys()) {
    if (PAGE_PARAMS.has(k) || k in SCENARIOS[scenarioName].defaults) continue;
    throw new Error(`?${k}=: not a parameter of scenario "${scenarioName}" `
      + `(its knobs are ${Object.keys(SCENARIOS[scenarioName].defaults).join(', ')})`);
  }
  const params = resolveScenario(scenarioName, overrides);
  const sc = SCENARIOS[scenarioName];

  if (!navigator.gpu) { reportNoWebGPU(statusEl); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { reportNoAdapter(statusEl); return; }

  // Same raised-limit pattern as main-amr.js: WebGPU devices default to the
  // SPEC MINIMUM (128 MiB storage binding) no matter what the adapter can
  // really do, and 128^3 D3Q19 is 152 MiB -- already past it (plans/3D.md
  // sec 2.2). Request what is needed, capped at the adapter's real
  // capability, and fail with a legible message rather than a validation
  // error if this GPU genuinely cannot.
  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const [NX, NY, NZ] = params.dims;
  // `N` remains the scenario's own resolution parameter (cube edge, or the
  // sphere's diameter); NX/NY/NZ are the domain. They differ only for the
  // sphere, whose box is long and narrow.
  const N = params.n;
  const NCELLS = NX * NY * NZ;

  // --- the moving window (plans/3D.md M8.3) --------------------------------
  //
  // Per-axis, and NOT a translation of the field: the buffer is periodic and
  // the fluid never moves, the BODY wraps through it, and what follows the
  // body is the SPONGE -- see d3-window.mjs for why this reading and not the
  // 2D pages' off_x/off_y, and shaders/common_d3_window.wgsl for the shader
  // side. Everything here is 0 by default, at which point every function in
  // both of those files is the identity and folds out at pipeline-creation
  // time.
  //
  // THE ANCHOR IS THE BODY'S INITIAL CELL, so the offset is exactly 0 at step
  // 0 and a windowed run starts bit-identical to an unwindowed one. It also
  // means the scenario keeps saying where its body sits in its own domain --
  // `fall` deliberately puts a towed body near the FAR end so the wake has
  // the long side of the box, and the window then preserves that arrangement
  // forever instead of having it improve as the body travels.
  const WIN_AXES = parseWindowAxes(urlParams.get('window'), params.window || [0, 0, 0]);
  const WIN_ANCHOR = params.body ? params.body.x.map(Math.floor) : [0, 0, 0];
  const WIN_ON = WIN_AXES.some(Boolean);
  if (WIN_ON) {
    // Three refusals rather than three quiet degradations, in the shape M5.4a
    // settled on: a window that does not do what it says is worse than one
    // that will not start.
    //
    // NO BODY: the window follows the body. There is nothing else it could
    // follow, and a window centred on nothing is a sponge in a random place.
    if (!params.body) {
      statusEl.textContent = `error: ?window= needs a body to follow; scenario "${scenarioName}" has none`;
      return;
    }
    // A WALL: the window assumes the axis is periodic, because that is what
    // lets the body cross the seam and the wake wrap into the absorbing band.
    // A walled axis is neither.
    const walled = WIN_AXES.map((on, i) => on && params.walls.includes('xyz'[i]));
    if (walled.some(Boolean)) {
      statusEl.textContent = `error: ?window= on a walled axis (${walled.map((w, i) => w ? 'xyz'[i] : '').join('')});`
        + ' the window needs a periodic axis for the body to cross and the wake to be absorbed on';
      return;
    }
    // NO SPONGE: this is the one that would silently produce a wrong answer
    // rather than an obviously broken one. Without an absorbing band the wake
    // simply wraps around the periodic buffer and the body flies back into
    // it -- a run that looks perfectly healthy and is measuring a body in its
    // own exhaust.
    if (!(params.sponge && params.sponge.width > 0)) {
      statusEl.textContent = 'error: ?window= needs a sponge; without one the wake wraps around'
        + ' the periodic buffer and the body flies back into it';
      return;
    }
  }
  // 0 on an unwindowed axis -- the encoding both the module and the shader
  // read as "no window here".
  const WIN_N = wrapDims([NX, NY, NZ], WIN_AXES);
  // Handed to every pipeline whose shader includes common_d3_window.wgsl, and
  // ONLY those: an override a module does not declare is a validation error,
  // so this cannot simply be folded into poolConst.
  const WINC = windowConstants(WIN_N, WIN_ANCHOR);

  // --- AMR pool (plans/3D.md M3) -------------------------------------------
  // One fine level, STATIC refinement. Dynamic refinement and the 2:1
  // balance cascade are M4 -- the plan splits them deliberately, because the
  // 2D manager's own header documents three separately-found live bugs in
  // the balance logic and there is no reason to debug that at the same time
  // as the pool addressing underneath it.
  const LEVELS = Math.max(1, Math.round(numParam('levels', 1)));
  const RB = Math.max(2, Math.round(numParam('rb', 4)));
  // M5.0. ?levels=3 WAS accepted and silently ran two levels: exactly one
  // pool is allocated, `AMR` is a boolean, and every dispatch below is the
  // hand-flattened depth-2 sequence. That on its own is a missing feature;
  // what makes it a trap is debugCheck21Balance, which reports
  // `vacuous: LEVELS < 3` -- so at ?levels=3 the ONE tool that would notice
  // announced it was checking a hierarchy that did not exist, and every M5
  // stage could have looked green while running nothing.
  //
  // Refused rather than clamped, because clamping to 2 is the same silence
  // with a different spelling: a validation config written as ?levels=3
  // would pass and mean nothing. M5.3 lifts this, when the recursive
  // schedule has something to run.
  //
  // M5.3 LIFTED THE BLANKET REFUSAL: the recursive schedule drives any depth,
  // and the L1/L2 seam is gated (`amr-bar-explode-L3`, `amr-box-explode-L3`).
  // What remains refused is narrower and named, so that every depth a run
  // CAN reach is one something validates:
  //   - a BODY needs per-level force integration with finest-wins masking,
  //     or the same drag is summed once per level. M5.4.
  //   - the MANAGER has no balance pass yet, so a dynamic run at depth could
  //     build a tree 2:1 balance forbids. M5.5.
  // Both are refusals rather than warnings for M5.0's reason: a run that is
  // quietly wrong is worse than one that will not start.
  //
  // BOTH ARE NOW LIFTED -- M5.4b gave the body one force pass at the finest
  // level, and M5.5b gave the manager the cascade and a per-level allocator.
  // What is still refused at any depth is narrower and is not about depth at
  // all: a body with AMR needs ?refine=body, and ?margin= must cover the
  // force stencil's reach.
  //
  // BISECTION HOOK. Each name drops one pass of the pool-parent nest at every
  // level >= 2, which makes the physics wrong on purpose -- the point is to
  // find which pass a blowup lives in by removing passes one at a time, the
  // same attribution method ?benchSkip= uses for cost. It is what found the
  // unseeded-pool bug in M5.2b-ii, in one run.
  const AMR_SKIP = new Set((urlParams.get('amrskip') || '').split(',').filter(Boolean));
  const AMR = LEVELS >= 2;
  // `pool` is the LEVEL-1 tiling; `hier.byLevel[m]` is level m's, from
  // d3-amr.mjs's poolAtLevel. They coincide at m = 1 and diverge by a factor
  // of two per level after it. M5.1c.
  let pool = null, hier = null;
  // The refinement mode, and -- when it is geometry-forced -- the SDF and
  // margin that forced it, so debugCheckGeometryCoverage can restate the
  // requirement independently.
  let refineMode = null, geomForced = null;
  // The hand-placed wake box, when ?refine=wake put one there (M8.4a). Only
  // for reporting -- the set it produced is already in `hier`.
  let wakeBox = null;
  if (AMR) {
    if ([NX, NY, NZ].some(n => n % RB !== 0)) {
      statusEl.textContent = `error: ?rb=${RB} does not divide the ${NX}x${NY}x${NZ} domain`;
      return;
    }
    // FB = 2*RB + 2*GHOST must be a multiple of the 4-thread workgroup z
    // extent, because the pool kernels fold the slot into z as
    // `fz = gid.z % FB; slot = gid.z / FB` -- with FB not divisible by 4 the
    // fold straddles workgroups and threads land in the wrong tile. That
    // needs RB even, which is worth failing loudly on rather than producing
    // a subtly scrambled pool.
    if ((2 * RB + 2 * GHOST) % 4 !== 0) {
      statusEl.textContent = `error: ?rb=${RB} gives FB=${2 * RB + 2 * GHOST}, which must be a multiple of 4 (use an even RB)`;
      return;
    }
    // A post-collision grid transfer cannot work at omega = 1 on either
    // level: collision there maps f exactly onto its own equilibrium, so the
    // stored populations carry NO non-equilibrium part and the stress the
    // interface has to hand across simply is not in them. The rescale
    // expresses that as a division by (tau - 1). This is a property of the
    // physics, not of the formula, so it fails loudly rather than being
    // clamped -- see shaders/common_d3_pool.wgsl.
    const tauFine = 2 * params.tau - 0.5;
    const nearOne = (t) => Math.abs(t - 1) < 0.02;
    if (nearOne(params.tau) || nearOne(tauFine)) {
      statusEl.textContent = `error: ?tau=${params.tau} gives tau_coarse=${params.tau}, tau_fine=${tauFine};`
        + ' a level at tau = 1 carries no non-equilibrium part for the coarse/fine transfer to rescale';
      return;
    }
    pool = makePool({ dims: [NX, NY, NZ], rb: RB });
    const mode = urlParams.get('refine') || (params.body ? 'body' : 'box');
    refineMode = mode;
    // ONE predicate, in L0 CELL UNITS, whatever the depth. refineHierarchy
    // evaluates it at the FINEST level only and derives every coarser level
    // from the 2:1 closure -- see its header for why that is the right
    // reading of geometry-forced refinement rather than a shortcut.
    let want = null;
    if (mode === 'all') {
      want = () => true;
    } else if (mode === 'box') {
      // A centred cube. For a scenario with an ANALYTIC answer (beltrami)
      // this is the whole point: a refined region in the middle of a flow
      // whose exact solution is known, so any damage the coarse/fine
      // interface does shows up directly as a field error.
      const frac = numParam('boxfrac', 0.5);
      const lo = [NX, NY, NZ].map(n => n * (1 - frac) / 2);
      const hi = [NX, NY, NZ].map(n => n * (1 + frac) / 2);
      want = ({ mid }) => mid.every((c, i) => c >= lo[i] && c < hi[i]);
    } else if (mode === 'slab') {
      // A refined SLAB spanning the full domain in y and z: periodic in
      // both, so the seam is two FLAT faces with no edge and no corner
      // anywhere. That is not a convenience geometry, it is the control for
      // M4's flux correction -- see plans/3D.md. Where the seam is flat the
      // fine channels crossing it tile the coarse one exactly; at a convex
      // edge or corner they provably do not, and this separates the two.
      const frac = numParam('boxfrac', 0.5);
      const lo = NX * (1 - frac) / 2, hi = NX * (1 + frac) / 2;
      want = ({ mid }) => mid[0] >= lo && mid[0] < hi;
    } else if (mode === 'bar') {
      // Refined in x AND y, spanning all of z: four convex EDGES, no
      // corner. The middle rung of the interface-geometry ladder --
      // `slab` (flat only), `bar` (edges), `box` (edges and corners) --
      // which is what separates "the coupling is wrong" from "the coupling
      // is wrong where the seam turns". See plans/3D.md M4.1b.
      const frac = numParam('boxfrac', 0.5);
      const lo = [NX, NY].map(n => n * (1 - frac) / 2);
      const hi = [NX, NY].map(n => n * (1 + frac) / 2);
      want = ({ mid }) => [0, 1].every(i => mid[i] >= lo[i] && mid[i] < hi[i]);
    } else if (mode === 'body' || mode === 'wake') {
      if (!params.body) throw new Error(`?refine=${mode}: this scenario has no body`);
      const sh = params.body.shape, bx = params.body.x;
      const margin = numParam('margin', 2);
      // Sphere-only for now, which is what M3's validation needs; a general
      // SDF here would have to mirror d3-body.mjs's rotation handling and
      // that belongs with dynamic refinement in M4.
      // Kept so debugCheckGeometryCoverage can restate the requirement at
      // CELL granularity against the same SDF -- an independent route from
      // refineNearBody's block-corner sampling, which is the thing under
      // test rather than the reference.
      //
      // sdfAt takes the CENTRE, because the body MOVES (M4.2b-iii) and the
      // requirement moves with it. Checking against the initial position
      // would pass a manager that refined a shell and then left it behind,
      // which is precisely the failure this is meant to catch.
      // The delta takes the NEAREST PERIODIC IMAGE on any windowed axis, the
      // same arm shaders/common_d3_geometry.wgsl's bodyDelta3 takes -- so a
      // body that has wrapped through the buffer seam refines one shell here
      // and not two, and debugCheckGeometryCoverage restates the requirement
      // against the same reading the kernel used. The identity without a
      // window (wrapDelta3 at WIN_N = 0), so every existing case is
      // untouched.
      //
      // THE BODY'S OWN SDF, IN ITS OWN ORIENTATION -- not a sphere of radius
      // `shape.a` (fixed 2026-09-11). This used to be `length(delta) - sh.a`,
      // which is EXACT for a sphere and wrong for every other shape the body
      // module offers. For the falling plate it was wrong in both directions
      // at once: it required a BALL of radius a + margin, about half of which
      // the plate does not occupy, while the plate's own CORNERS at
      // sqrt(a^2 + b^2 + c^2) lie OUTSIDE that ball and so were not required
      // at all. Worse, the host BUILT the initial hierarchy from the same
      // wrong function, so debugCheckGeometryCoverage scored the host against
      // itself and passed -- a gate that goes green because both sides share
      // one mistake. The GPU manager has always used the true `get_phi3`, so
      // the first manager pass made the two disagree and the gate went red
      // pointing at the manager, which was the half that was right.
      //
      // Bit-identical for a sphere: sdfBody reduces to |p| - a there, and the
      // inverse rotation of a sphere is the identity. Every shipped case is a
      // sphere, and all of them are unchanged.
      //
      // The ORIENTATION has to be passed in for the same reason the CENTRE
      // does: a body that turns changes which cells are near it even standing
      // still, and a plate turns through 90 degrees of its own chord.
      geomForced = { radius: sh.a, margin,
        sdfAt: (c, bq) => (q) => sdfBody(
          qRotateInv(bq, wrapDelta3([q[0] - c[0], q[1] - c[1], q[2] - c[2]], WIN_N)), sh) };
      const bodyWant = nearBodyWant(geomForced.sdfAt(bx, params.body.q), margin);
      want = bodyWant;
      if (mode === 'wake') {
        // --- THE HAND-PLACED WAKE BOX (plans/3D.md M8.4a) ------------------
        //
        // NOT A CRITERION. It is the CONTROL that tells a criterion what it
        // has to achieve, and it exists before any criterion for the reason
        // M5.0 exists: M8.4's gate is "the shed wake stays resolved", and
        // writing a Q-criterion first would be tuning a threshold against an
        // imagined case. This refines the wake BY HAND, so the question
        // "what does resolving the wake buy" has an answer -- measured in St
        // against M8.1's yardstick -- before anything decides for itself.
        //
        // It is also the control M8.4c is scored against: a criterion-built
        // set that does not reproduce this set's St is not doing the job,
        // whatever its threshold sweep says.
        //
        // THE UNION WITH THE BODY SET IS BY CONSTRUCTION, not by hoping the
        // box contains the body. M5.4's hard requirement (the body lives
        // entirely on the finest level) is enforced by refusals that key off
        // `geomForced`, and debugCheckGeometryCoverage restates it at cell
        // granularity -- both keep working unchanged on a SUPERSET, and
        // neither would survive this being a box that merely ought to
        // overlap the body.
        //
        // Downstream is +x on every scenario that has a wake here, which is
        // the same assumption d3-scenarios.mjs's `sphere` and `fall` already
        // make about their own domains being long in x.
        const D = 2 * sh.a;
        const lenD = numParam('wake', 6);      // downstream extent, in diameters
        const radD = numParam('wakeR', 1);     // transverse half-width, in diameters
        const wlo = [bx[0] - margin, bx[1] - radD * D, bx[2] - radD * D];
        const whi = [bx[0] + lenD * D, bx[1] + radD * D, bx[2] + radD * D];
        // The block's BOX overlaps the wake box, not just its centre: a
        // centre test drops the blocks straddling the boundary, which is a
        // ragged edge on the one region whose job is to be a clean control.
        const inWake = ({ lo, hi }) => [0, 1, 2].every(i => hi[i] > wlo[i] && lo[i] < whi[i]);
        want = (b) => bodyWant(b) || inWake(b);
        wakeBox = { lo: wlo, hi: whi, lenD, radD };
      }
    } else {
      throw new Error(`?refine=${mode}: expected all, box, bar, slab, body or wake`);
    }
    hier = refineHierarchy(pool, { levels: LEVELS, want });
  }
  // Level 1's allocation, which is what every depth-2 code path below still
  // addresses directly. M5.3's recursive schedule is what removes the last
  // of these; until then naming it once is better than `hier.byLevel[1]`
  // scattered through the file.
  const poolAlloc = AMR ? hier.byLevel[1] : null;



  const fBytes = NCELLS * Q * 4;
  // The POOL, not the dense grid, is the largest binding once AMR is on: a
  // tile stores (FB/RB)^3 cells per coarse cell covered, which is 27x at
  // RB=4 (plans/3D.md sec 2.1's table). This has to be known BEFORE
  // requestDevice, because a device is created with the SPEC MINIMUM limits
  // unless asked otherwise -- and a 227 MiB pool binding on a device that
  // only asked for 128 MiB fails validation at buffer creation, which
  // presents as a fine level that silently never runs. It did exactly that
  // during development.
  // M4.2b-i. Dynamic refinement: the pool's block->slot map becomes writable
  // by a kernel instead of uploaded once. OPT-IN, and only with
  // ?refine=body, because that is the only criterion that can legitimately
  // change during a run -- a fixed box has no reason to be re-decided, and
  // letting it be re-decided would only add churn to test nothing.
  const DYNAMIC = urlParams.get('dynamic') === '1' ? 1 : 0;
  // How often the criterion is re-evaluated, in macro-steps. Re-deciding
  // every step is pure cost on a body that moves a fraction of a cell per
  // step; the 2D manager re-evaluates on an interval for the same reason.
  const MANAGE_EVERY = Math.max(1, Math.round(numParam('manageEvery', 16)));
  // The first step the manager may run on. Default 0, so it decides before
  // anything moves. Its reason for existing is testability: with a PINNED
  // body the wanted-set never changes after step 0, so an event placed at
  // step 0 happens while the fine and coarse levels still hold the same
  // initial condition -- and a drain that restricts one onto the other is
  // then indistinguishable from doing nothing. Delaying the first event lets
  // the flow develop first, which is the only way to see the drain work.
  const MANAGE_START = Math.max(0, Math.round(numParam('manageStart', 0)));
  // Slot headroom. The pool is sized to the slots actually in use (a 3D pool
  // sized for the whole domain would be (FB/RB)^3 = 27x the dense grid at
  // RB=4), so a manager that can only ever hand back what it already has is
  // not a manager. This is the fraction ABOVE the initial set it may grow
  // into, and it costs real memory -- at the sphere case's 160 tiles, 1.5x
  // is 63 MB against 42 MB.
  const SLOT_HEADROOM = Math.max(1, numParam('slotHeadroom', 1.5));
  // TEST HOOK. The margin the MANAGER's criterion uses, defaulting to the one
  // that built the initial set -- so by default the two agree and the manager
  // decides "no change", which is what M4.2b-i's bit-identical gate needs.
  //
  // Setting it DIFFERENT deliberately disagrees with the initial set, and
  // that is the only way to prove the manager ran at all: a gate that shows
  // "nothing changed" is equally consistent with a manager that never
  // executed. Smaller than ?margin= makes the coarsen pass fire (blocks are
  // released); larger makes refine fire (slots are popped). Both leave the
  // FIELD wrong until M4.2b-ii initializes and restricts, so this is for
  // STRUCTURAL checks only -- tools/validate-d3-invariants.js.
  const MANAGE_MARGIN = numParam('manageMargin', NaN);

  // Only ?refine=body has a criterion a kernel can re-evaluate. Every other
  // mode is a fixed region chosen once, and re-deciding it would add
  // allocation churn to test nothing.
  if (DYNAMIC && !geomForced) {
    throw new Error(`?dynamic=1 needs ?refine=body (a geometry-forced criterion); ?refine=${refineMode} is a fixed region`);
  }
  // AND ?refine=wake IS NOT DYNAMIC EITHER, even though it sets geomForced.
  // The manager's criterion is common_d3_manage.wgsl's blockWanted, which
  // knows the BODY and nothing else -- so on the first management event it
  // would coarsen the entire wake box away and report a perfectly consistent
  // pool while doing it. The run would look healthy and be measuring the
  // thing the box was added to avoid. Refused until M8.4b gives the kernel a
  // field criterion; this is the same refusal shape as the one above, for
  // the same reason.
  if (DYNAMIC && refineMode === 'wake') {
    throw new Error('?dynamic=1 does not yet work with ?refine=wake: the manager\'s criterion is'
      + ' geometry-only, so it would coarsen the wake box away on the first event (plans/3D.md M8.4a)');
  }

  // The slot budget. Static refinement knows its own answer up front and
  // allocates exactly that; dynamic refinement has to be able to GROW, so it
  // pays for headroom above the initial set. See SLOT_HEADROOM.
  //
  // PER LEVEL, and it has to be: measured on the sphere geometry, level 1
  // holds 160 tiles at ?levels=2 and 304 at ?levels=3, because a deeper
  // level demands a wider buffer shell around it. A budget derived from the
  // depth-2 answer would under-allocate the coarser levels of every deeper
  // run (plans/3D.md M5.1b).
  // THE SLOT BUDGET. ?slotHeadroom= scales the INITIAL set, which is the right
  // derivation for a geometry criterion -- the body's shell is the whole
  // answer and it does not grow. A FIELD criterion's set is not predicted by
  // the initial one at all: at reset the flow has not developed and the wake
  // does not exist yet, so the initial set is the geometry shell and the
  // steady-state set is several times it (measured: 160 blocks of geometry
  // against 379-628 of wake, plans/3D.md M8.4).
  //
  // So ?slots= sets it absolutely, and that is the knob a field criterion has
  // to use. It is sized from debugCriterion()'s sweep rather than guessed --
  // and running out is still a HARD failure, so the budget is a claim about
  // the flow that the run will check.
  const SLOT_ABS = Math.round(numParam('slots', NaN));
  const maxSlotsAt = (m) => {
    const a = hier.byLevel[m];
    if (Number.isFinite(SLOT_ABS) && SLOT_ABS > 0) {
      // Never BELOW the initial set: that would refuse a geometry-forced tile
      // at step 0, which is M5.4a's hard failure arriving before the run has
      // done anything.
      return Math.min(a.pool.nBlocks, Math.max(a.activeSlots, SLOT_ABS));
    }
    return Math.min(a.pool.nBlocks, Math.max(1, DYNAMIC
      ? Math.ceil(a.activeSlots * SLOT_HEADROOM)
      : a.activeSlots));
  };
  const SLOTS = [null];
  if (AMR) for (let m = 1; m < LEVELS; m++) SLOTS[m] = maxSlotsAt(m);
  const MAX_SLOTS = AMR ? SLOTS[1] : 0;
  // tileCells is the SAME at every level -- that is the uniform tile shape
  // (plans/3D.md M5, and d3-amr.mjs's poolAtLevel) showing up as one number
  // rather than a per-level one.
  const levelBytes = (m) => SLOTS[m] * pool.tileCells * Q * 4;
  let poolBytes = 0, biggestLevel = 1;
  if (AMR) for (let m = 1; m < LEVELS; m++) {
    if (levelBytes(m) > poolBytes) { poolBytes = levelBytes(m); biggestLevel = m; }
  }
  const needBytes = Math.max(fBytes, poolBytes);
  const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  if (needBytes > limit) {
    const what = poolBytes > fBytes
      ? `an L${biggestLevel} pool of ${SLOTS[biggestLevel]} tiles x ${pool.tileCells} cells`
      : `${NX}x${NY}x${NZ} D3Q${Q}`;
    statusEl.textContent = `error: ${what} needs a ${(needBytes / 1048576).toFixed(0)} MiB binding, this GPU's max is ${(limit / 1048576).toFixed(0)} MiB`;
    return;
  }
  // M6's tree sampler spans levels, so it binds the dense `mac` plus one
  // (mac, blockSlot) pair per pool level -- nine storage buffers at
  // MAX_SAMPLE_LEVELS = 4 -- and the probe that scores it against the host
  // adds an input and an output on top, for eleven in a COMPUTE stage. The
  // WebGPU spec MINIMUM is 8, which is not a real GPU limit: main-amr.js
  // already requests 16 for the 2D pool manager and runs on the phone.
  //
  // Same treatment as maxStorageBufferBindingSize above: ask for what is
  // needed, and fail LOUD rather than let the device come back with the
  // default and every affected pipeline turn into a cryptic
  // "Invalid PipelineLayout is invalid due to a previous error" -- which is
  // exactly how this was found, since a bind-group-layout failure is not an
  // exception, it just poisons everything downstream.
  const NEEDED_STORAGE_BUFFERS_PER_STAGE = 12;
  if (NEEDED_STORAGE_BUFFERS_PER_STAGE > adapter.limits.maxStorageBuffersPerShaderStage) {
    statusEl.textContent = `error: needs ${NEEDED_STORAGE_BUFFERS_PER_STAGE} storage buffers per shader stage,`
      + ` this GPU's max is ${adapter.limits.maxStorageBuffersPerShaderStage}`;
    return;
  }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(Math.max(needBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(Math.max(needBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
      maxStorageBuffersPerShaderStage: NEEDED_STORAGE_BUFFERS_PER_STAGE,
    },
  });
  device.lost.then((info) => {
    if (info.reason === 'destroyed') return;
    statusEl.textContent = `error: GPU device lost (${info.reason}): ${info.message}`;
  });
  // WebGPU validation errors are NOT exceptions: an over-large buffer, a
  // bind-group mismatch or a bad dispatch just makes the offending object
  // invalid and every use of it a no-op. Without this the symptom is a
  // kernel that silently does nothing -- which is precisely how the pool's
  // own size limit presented before it was found. Surfacing it into #status
  // puts it on the one channel tools/lib/browser-lifecycle.js's page watch
  // and validate-all.js's boot smoke both read.
  device.addEventListener('uncapturederror', (e) => {
    console.error('[getting-air] WebGPU error:', e.error.message);
    if (!/^error:/i.test(statusEl.textContent)) {
      statusEl.textContent = `error: WebGPU: ${e.error.message.split('\n')[0]}`;
    }
  });

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const U = GPUBufferUsage;
  const fA  = device.createBuffer({ size: fBytes, usage: U.STORAGE });
  const fB  = device.createBuffer({ size: fBytes, usage: U.STORAGE });
  const mac = device.createBuffer({ size: NCELLS * 4 * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const rpBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
  // Rigid body. Allocated for EVERY scenario, bodied or not, so there is one
  // bind-group layout and one set of pipelines -- see this file's header on
  // why the scenario is a parameter rather than a fork. HAS_BODY folds the
  // solid coupling out of the kernels entirely when there is no body.
  const HAS_BODY = params.body ? 1 : 0;
  const bodyBuf = device.createBuffer({ size: BODY_FIELDS.length * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  // 8 slots, not 6: forces[0..5] are fx,fy,fz,tx,ty,tz and the pad keeps the
  // clear kernel's one 8-lane workgroup exactly covering the buffer.
  const forceBuf = device.createBuffer({ size: 8 * 4, usage: U.STORAGE | U.COPY_SRC });

  // Pool buffers. Sized to the slots actually allocated, not to maxSlots:
  // static refinement knows its own answer up front, and a 3D pool sized for
  // the whole domain would be (FB/RB)^3 = 27x the dense grid at RB=4.
  let fluxAccBuf = null;
  // ONE ENTRY PER LEVEL (M5.1c). Everything the depth-2 code below still
  // names directly is an alias onto L[1]; the loop is what M5.2 and M5.3
  // extend, and building it now means those stages raise a level count
  // rather than restructure allocation while also debugging a new coupling.
  const L = [null];
  const makeLevelBuffers = (m) => {
    const alloc = hier.byLevel[m];
    const slots = SLOTS[m];
    const poolCells = slots * pool.tileCells;
    const nBlocks = alloc.pool.nBlocks;
    const lv = { level: m, pool: alloc.pool, alloc, slots, nBlocks };
    lv.fA = device.createBuffer({ size: levelBytes(m), usage: U.STORAGE });
    lv.fB = device.createBuffer({ size: levelBytes(m), usage: U.STORAGE });
    lv.mac = device.createBuffer({ size: poolCells * 4 * 4, usage: U.STORAGE | U.COPY_SRC });
    // COPY_SRC so debugCheck21Balance reads what the GPU HAS rather than the
    // host's copy of what it once uploaded. Identical today, because
    // refinement is static -- and exactly not identical the moment M4.2's
    // manager starts writing this buffer from a kernel, which is when the
    // checker has to already be right.
    lv.blockSlot = device.createBuffer({ size: nBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // COPY_SRC on slotToBlock too: it is blockSlot's inverse, and cross-
    // checking the two against each other is how the 2D free-list race was
    // confirmed -- they disagreed for exactly the colliding slot.
    lv.slotToBlock = device.createBuffer({ size: slots * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    lv.freeList = device.createBuffer({ size: Math.max(1, slots) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    lv.freeCount = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // Always allocated, even when static: `blockWant` and `slotNew` are bound
    // into the interp and average layouts, and a layout that exists in two
    // versions is exactly the 238e48c failure surface. One dummy element is
    // enough where there is no manager -- the pipelines that read them are
    // never created, and NEW_ONLY/DYING_ONLY fold the reads out of the ones
    // that are.
    // FULL SIZE ALWAYS (M5.5a). It used to be one dummy element without the
    // manager; the criterion-and-closure chain is now exercised on static
    // runs too (debugRunBalance), and it is nBlocks * 4 bytes -- 1.5 MB at
    // level 2 of the sphere case, against a pool measured in hundreds.
    lv.blockWant = device.createBuffer({ size: nBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    lv.slotNew = device.createBuffer({ size: (DYNAMIC ? slots : 1) * 4, usage: U.STORAGE | U.COPY_DST });
    // The initial set is not "new": it was uploaded with real data by the
    // host, and marking it new would have the fill pass overwrite every tile
    // with a coarse interpolation on the first step.
    uploadLevelTopology(lv);
    return lv;
  };
  // THE WHOLE OF A LEVEL'S BOOKKEEPING, uploaded from the host's initial
  // refinement. Split out of makeLevelBuffers because reset() has to restore
  // it too (M5.5b): the manager rewrites blockSlot, slotToBlock and the free
  // list as tiles change hands, so re-zeroing only freeCount -- which is what
  // reset used to do -- would leave a free list whose entries are slots that
  // are still in use. Static runs never noticed, because nothing there ever
  // changed. The field is re-seeded right after, so the topology and the
  // data are restored together or not at all.
  function uploadLevelTopology(lv) {
    const { alloc, slots, nBlocks } = lv;
    const s2b = new Int32Array(slots).fill(-1);
    s2b.set(alloc.slotToBlock.slice(0, Math.min(slots, alloc.activeSlots)));
    device.queue.writeBuffer(lv.blockSlot, 0, alloc.blockSlot);
    device.queue.writeBuffer(lv.slotToBlock, 0, s2b);
    // The free list, a classic GPU stack: freeCount is how many slots are
    // free, and the top of the stack lives at freeList[freeCount-1]. Slots
    // [0, activeSlots) start in use; everything above is free. freeCount[1]
    // is the refusal counter, and clearing it here is what makes the hard
    // failure per-RUN rather than permanent.
    const nFree = slots - alloc.activeSlots;
    const freeInit = new Int32Array(Math.max(1, slots));
    for (let i = 0; i < nFree; i++) freeInit[i] = alloc.activeSlots + i;
    device.queue.writeBuffer(lv.freeList, 0, freeInit);
    // [2] is the MINIMUM free count ever reached -- the exact high-water mark
    // of slot usage, recorded by the refine kernel. It seeds at the CURRENT
    // free count rather than at a large sentinel, so a run whose manager
    // never allocates still reports the truth (the initial set is the peak)
    // instead of a value nothing ever wrote.
    device.queue.writeBuffer(lv.freeCount, 0, new Int32Array([nFree, 0, nFree, 0]));
    device.queue.writeBuffer(lv.slotNew, 0, new Uint32Array(DYNAMIC ? slots : 1));
    device.queue.writeBuffer(lv.blockWant, 0, new Uint32Array(nBlocks));
  }
  if (AMR) {
    for (let m = 1; m < LEVELS; m++) L[m] = makeLevelBuffers(m);
    // Per coarse cell: the mass and momentum the FINE solver moved across
    // the seam this macro step, in coarse-cell units. Written by the flux
    // pass on substep A and added to on substep B; consumed by the reflux
    // pass. Only seam-adjacent cells are ever written OR read (both passes
    // share one reachability predicate), so it is deliberately not cleared.
    // L0-sized, so it is not per-level: it is the COARSE side of the seam.
    fluxAccBuf = device.createBuffer({ size: NCELLS * 4 * 4, usage: U.STORAGE | U.COPY_SRC });
  }
  // The depth-2 aliases. Every dispatch and bind group below still addresses
  // level 1 by name; M5.3's recursive schedule is what replaces them with a
  // walk over L. Naming them once here keeps that a localized change instead
  // of a rename across the file.
  const fPoolA = AMR ? L[1].fA : null;
  const fPoolB = AMR ? L[1].fB : null;
  const macPool = AMR ? L[1].mac : null;
  const blockSlotBuf = AMR ? L[1].blockSlot : null;
  const slotToBlockBuf = AMR ? L[1].slotToBlock : null;
  const freeListBuf = AMR ? L[1].freeList : null;
  const freeCountBuf = AMR ? L[1].freeCount : null;
  const blockWantBuf = AMR ? L[1].blockWant : null;
  const slotNewBuf = AMR ? L[1].slotNew : null;

  // `blockSlot` is bound to the coarse step on EVERY scenario so there is one
  // bind-group layout, the same arrangement `body` uses. With no pool it is a
  // single -1 element and HAS_POOL = 0 folds the lookup out at
  // pipeline-creation time -- see common_d3_step.wgsl's coveredByFiner.
  // Sec 6's 238e48c lesson is specifically about bind groups drifting between
  // near-identical code paths, so the layout does not fork.
  const blockSlotBound = blockSlotBuf ?? (() => {
    const b = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST });
    device.queue.writeBuffer(b, 0, new Int32Array([-1]));
    return b;
  })();

  const computeBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]});
  // M4.1d: binding 3 is blockSlot, for the finest-wins mask. Bound on EVERY
  // scenario with the same dummy the step kernel uses when there is no pool,
  // so this layout does not fork -- the 238e48c failure surface is exactly a
  // bind group that exists in two versions.
  // f_in, body, forces. No blockSlot: the coarse force kernel no longer masks
  // cells covered by a finer level, because under AMR it is not dispatched at
  // all -- the body lives entirely on the finest level (plans/3D.md M5.4b).
  const forceBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  // The fine-level force kernel: f_in, body, forces, slotToBlock.
  const forcePoolBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]});
  const physicsBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const zeroBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  // 0 = the dense L0 `mac`, 1 = the render params, and 2..9 = up to FOUR
  // pool levels as (mac, blockSlot) pairs -- common_d3_tree_sample.wgsl's
  // layout. FIXED IN NUMBER because WebGPU has no array of buffers; levels
  // the run does not have get a 4-byte dummy and SAMPLE_LEVELS folds them
  // out of the walk, which is the same shape the manager's child-want
  // binding uses and for the same reason.
  const MAX_SAMPLE_LEVELS = 4;
  // ONE DESCRIPTION OF THE SAMPLER'S BINDINGS, handed to both consumers --
  // the slice view's fragment stage and the probe's compute stage -- because
  // two copies of a ten-entry layout that must match one WGSL file is
  // exactly the shape of 238e48c.
  const renderBGL_entries = [
    { binding: 0, buffer: { type: 'read-only-storage' } },
    { binding: 1, buffer: { type: 'uniform' } },
    ...Array.from({ length: 2 * MAX_SAMPLE_LEVELS }, (_, i) => (
      { binding: 2 + i, buffer: { type: 'read-only-storage' } })),
  ];
  // The sampler's shared bindings, plus ONE this view owns: the body, for
  // the moving window's view offset (M8.3). Appended here rather than added
  // to renderBGL_entries so the probe and the resample -- which read the same
  // sampler and nothing else -- do not acquire a binding neither of them
  // uses. Same shape as the resample pass appending its output texture.
  const renderBGL = device.createBindGroupLayout({
    entries: [
      ...renderBGL_entries.map(e => ({ ...e, visibility: GPUShaderStage.FRAGMENT })),
      { binding: 12, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ] });
  const bgAB = device.createBindGroup({ layout: computeBGL, entries: [
    { binding: 0, resource: { buffer: fA } }, { binding: 1, resource: { buffer: fB } },
    { binding: 2, resource: { buffer: mac } }, { binding: 3, resource: { buffer: bodyBuf } },
    { binding: 4, resource: { buffer: blockSlotBound } }]});
  const bgBA = device.createBindGroup({ layout: computeBGL, entries: [
    { binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: fA } },
    { binding: 2, resource: { buffer: mac } }, { binding: 3, resource: { buffer: bodyBuf } },
    { binding: 4, resource: { buffer: blockSlotBound } }]});
  // The force kernel reads the SAME f_in the step kernel will read this
  // macro-step -- it is dispatched first -- so its populations are the
  // pre-streaming, time-t data both formulas want, with no separate
  // buffer-timing bookkeeping. Same arrangement as the 2D main-cylinder.js.
  const forceBGA = device.createBindGroup({ layout: forceBGL, entries: [
    { binding: 0, resource: { buffer: fA } }, { binding: 1, resource: { buffer: bodyBuf } },
    { binding: 2, resource: { buffer: forceBuf } }]});
  const forceBGB = device.createBindGroup({ layout: forceBGL, entries: [
    { binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: bodyBuf } },
    { binding: 2, resource: { buffer: forceBuf } }]});
  const physicsBG = device.createBindGroup({ layout: physicsBGL, entries: [
    { binding: 0, resource: { buffer: bodyBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const zeroBG = device.createBindGroup({ layout: zeroBGL, entries: [{ binding: 0, resource: { buffer: forceBuf } }]});
  // How deep the viewer can see. Clamped to the bindings that exist rather
  // than refused: running out of render bindings is no reason to refuse a
  // solve, and a run deeper than this is already past anything validated.
  // It is SAID OUT LOUD in #status when it bites, because a view that
  // quietly stops at level 4 of 6 looks exactly like a tree that stops
  // there.
  const SAMPLE_LEVELS = Math.min(MAX_SAMPLE_LEVELS, AMR ? LEVELS - 1 : 0);
  // Said out loud, every frame, for as long as it is true. A viewer that
  // silently stops one level short of the solver is the exact shape of thing
  // M5.0 refused ?levels=3 over: it renders a plausible picture of a
  // hierarchy that is not the one being solved.
  const viewDepthNote = AMR && SAMPLE_LEVELS < LEVELS - 1
    ? `   [view samples to L${SAMPLE_LEVELS} of L${LEVELS - 1}]` : '';
  const sampleDummy = device.createBuffer({ size: 4, usage: U.STORAGE });
  const treeSampleEntries = () => [
    { binding: 0, resource: { buffer: mac } }, { binding: 1, resource: { buffer: rpBuf } },
    ...Array.from({ length: MAX_SAMPLE_LEVELS }, (_, i) => {
      const lv = i + 1 <= SAMPLE_LEVELS ? L[i + 1] : null;
      return [
        { binding: 2 + 2 * i, resource: { buffer: lv ? lv.mac : sampleDummy } },
        { binding: 3 + 2 * i, resource: { buffer: lv ? lv.blockSlot : sampleDummy } },
      ];
    }).flat()];
  const renderBG = device.createBindGroup({ layout: renderBGL, entries: [
    ...treeSampleEntries(), { binding: 12, resource: { buffer: bodyBuf } }]});

  const loadShader = (p) => assembleShader(p, (f) => fetch(f).then(r => {
    if (!r.ok) throw new Error(`failed to fetch ${f}: ${r.status}`);
    return r.text();
  }));
  const stepModule = device.createShaderModule({ code: await loadShader(`shaders/d3_step_q${Q}.wgsl`), label: `d3_step_q${Q}` });
  const renderModule = device.createShaderModule({ code: await loadShader(`shaders/d3_render_slice_q${Q}.wgsl`), label: `d3_render_slice_q${Q}` });
  const forceModule = HAS_BODY ? device.createShaderModule({ code: await loadShader(`shaders/d3_force_q${Q}.wgsl`), label: `d3_force_q${Q}` }) : null;
  const physModule = HAS_BODY ? device.createShaderModule({ code: await loadShader('shaders/d3_physics.wgsl'), label: 'd3_physics' }) : null;
  const zeroModule = HAS_BODY ? device.createShaderModule({ code: await loadShader('shaders/d3_zero_forces.wgsl'), label: 'd3_zero_forces' }) : null;

  const WG = [4, 4, 4];
  const disp = [Math.ceil(NX / WG[0]), Math.ceil(NY / WG[1]), Math.ceil(NZ / WG[2])];
  const dims = { NX, NY, NZ };
  const USE_BOUNCEBACK = urlParams.has('bounceback') ? 1 : 0;
  // M8.2a. Hold the solid interior at feq(1, u_body) every step. Default ON;
  // ?solideq=0 restores the undamped interior for A/B. Bounce-back only --
  // the diffuse coupling damps its own interior through chi -- and passed to
  // BOTH step kernels from here, because the dense and pool paths must agree
  // about what is inside a body.
  const SOLID_EQ = urlParams.get('solideq') === '0' ? 0 : 1;
  // D1. Charge the body for the momentum of the cells its moving surface
  // buries and frees -- see shaders/common_d3_force.wgsl's SWEPT_FORCE
  // header. Default ON; ?swept=0 restores the link-only force for an A/B.
  // Identically zero on a PINNED body, so every validated sphere case is
  // bit-identical across it. Passed to BOTH force kernels from here, for the
  // reason SOLID_EQ is: the dense and pool paths must charge the same body
  // for the same thing.
  const SWEPT_FORCE = urlParams.get('swept') === '0' ? 0 : 1;
  // D3. Ladd's moving-wall correction carries the LOCAL density rather than a
  // pinned 1 -- see shaders/common_d3_force.wgsl's RHO_W_LOCAL header for why
  // that is what makes the coupling Galilean invariant. Default ON; ?rhow=0
  // restores the constant for an A/B. Identically zero on a PINNED body (the
  // correction itself vanishes at u_w = 0), so every pinned gate is
  // bit-identical. Passed to ALL FOUR kernels from here -- both step kernels
  // and both force kernels -- because the force reported must be the force
  // applied, and a mismatch there is invisible.
  const RHO_W_LOCAL = urlParams.get('rhow') === '0' ? 0 : 1;
  const CHI_EPS = numParam('chiEps', 1.5);
  const sponge = params.sponge || { width: 0, u: [0, 0, 0] };
  // Per-axis sponge mask (see common_d3_step.wgsl's SPONGE_AX). A scenario
  // that says nothing gets the band on every face, as before.
  const spongeAxes = sponge.axes || [1, 1, 1];
  const SPONGE_AXES = { SPONGE_AX: spongeAxes[0] ? 1 : 0, SPONGE_AY: spongeAxes[1] ? 1 : 0, SPONGE_AZ: spongeAxes[2] ? 1 : 0 };
  const stepConstants = {
    ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2],
    OMEGA: 1 / params.tau,
    FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
    WALL_X: params.walls.includes('x') ? 1 : 0,
    WALL_Y: params.walls.includes('y') ? 1 : 0,
    WALL_Z: params.walls.includes('z') ? 1 : 0,
    HAS_BODY, USE_BOUNCEBACK, CHI_EPS, SOLID_EQ, RHO_W_LOCAL, ...WINC,
    SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2], ...SPONGE_AXES,
    // M4.1a: skip cells a refined block covers. Folds out entirely when
    // there is no pool.
    HAS_POOL: AMR ? 1 : 0,
    RB, NBX: AMR ? pool.nb[0] : 1, NBY: AMR ? pool.nb[1] : 1, NBZ: AMR ? pool.nb[2] : 1,
  };
  const computePL = device.createPipelineLayout({ bindGroupLayouts: [computeBGL] });
  const initPipe = await device.createComputePipelineAsync({ layout: computePL, compute: { module: stepModule, entryPoint: 'initEq', constants: stepConstants } });
  const stepPipe = await device.createComputePipelineAsync({ layout: computePL, compute: { module: stepModule, entryPoint: 'step', constants: stepConstants } });
  // USE_BOUNCEBACK and CHI_EPS are passed to the force kernel from the SAME
  // constants the step kernel got. They are independent pipelines over the
  // same overrides, and a mismatch would integrate a force the fluid never
  // felt -- so they are created here as a pair, from one source, rather
  // than each being given its own copy of the value.
  // No pool overrides any more: the mask they fed is gone (M5.4b). Under AMR
  // this pipeline is created but never dispatched -- the finest level's pass
  // integrates the whole body.
  const forcePipe = HAS_BODY ? await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [forceBGL] }),
    compute: { module: forceModule, entryPoint: 'main', constants: {
      ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2], USE_BOUNCEBACK, CHI_EPS, SWEPT_FORCE, RHO_W_LOCAL, ...WINC } },
  }) : null;
  const physPipe = HAS_BODY ? await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [physicsBGL] }),
    compute: { module: physModule, entryPoint: 'main', constants: {
      GX: (params.gravity || [0, 0, 0])[0],
      GY: (params.gravity || [0, 0, 0])[1],
      GZ: (params.gravity || [0, 0, 0])[2],
      NO_FLUID_FORCE: params.noFluidForce ? 1 : 0,
      PLANAR: params.planar ? 1 : 0,
      ...WINC,
    } },
  }) : null;
  const zeroPipe = HAS_BODY ? await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [zeroBGL] }),
    compute: { module: zeroModule, entryPoint: 'main' },
  }) : null;

  // --- the Q-criterion, as a DIAGNOSTIC (plans/3D.md M8.4) -------------------
  //
  // WIRED READ-ONLY FIRST, AND ON PURPOSE. It reports per-block max Q and
  // drives nothing. What has to be answered before it may drive anything is
  // the SLOT BUDGET: maxSlotsAt() derives the pool size from the INITIAL
  // refined set, which for a geometry criterion is the whole answer and for a
  // field criterion is a domain with no wake in it yet. Running out of slots
  // is a HARD failure here (M5.4a) and should stay one -- slots come off an
  // atomicSub free list, so "drop what did not fit" would make the resolution
  // of the wake depend on allocation order, which is nondeterministic and
  // exactly the kind of run that looks healthy while being quietly worse. So
  // the budget has to be sized from a measurement, and this is the
  // measurement.
  //
  // It is evaluated on the FINEST level's candidate block grid, which is the
  // grid the criterion would flag into.
  // ON THE LEVEL-1 BLOCK GRID, at L0 resolution -- ONE evaluation, and the
  // LADDER is what grades it (plans/3D.md M8.4). That is the whole reason the
  // ladder is lighter than a per-level error estimator: `desiredLevel` turns a
  // single criterion value into "how deep does this region want to be", so
  // every level reads the same buffer with a shift instead of needing its own
  // field. The cost is that the grading is localized to a level-1 block,
  // which is the resolution the criterion was computed at anyway.
  const CRIT_LEVEL = 1;
  const CRIT_NB = [NX, NY, NZ].map(n => Math.max(1, Math.round(n / RB)));
  const CRIT_BLK_L0 = RB;
  // The localization limit the shader's header names: below one L0 cell per
  // candidate block the criterion cannot tell siblings apart, so it would
  // flag all of them. Said out loud rather than silently over-refining.
  const critNote = CRIT_BLK_L0 < 1
    ? `   [criterion at L${CRIT_LEVEL} spans ${CRIT_BLK_L0} L0 cells: below cell granularity]` : '';
  const critBuf = device.createBuffer({
    size: CRIT_NB[0] * CRIT_NB[1] * CRIT_NB[2] * 4,
    usage: U.STORAGE | U.COPY_SRC });
  const critBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const critBG = device.createBindGroup({ layout: critBGL, entries: [
    { binding: 0, resource: { buffer: mac } }, { binding: 1, resource: { buffer: critBuf } }]});
  // The convection lead, in L0 cells: how far the flow carries a vortex
  // between management events. Derived from the scenario's own velocity scale
  // the same way d3-criterion.mjs states it, and overridable for a sweep.
  const CRIT_U = params.u0 || params.u_t || params.uRel || 0;
  const CRIT_LEAD = numParam('qlead', MANAGE_EVERY * CRIT_U);
  // THE FIELD CRITERION IS OPT-IN, and the switch is the presence of
  // `?qthresh=` rather than a separate flag. Absent, Q_ABS folds the whole
  // test out of `decide` at pipeline-creation time and every scenario that
  // predates M8.4 is bit-identical rather than merely unaffected.
  const Q_ON = urlParams.has('qthresh');
  const Q_THRESH = numParam('qthresh', Q_THRESHOLD);
  // Dimensionless in, absolute out: the threshold is quoted against the
  // body's own shear scale so one number means the same thing at another Re
  // or resolution (d3-criterion.mjs's qRef).
  const CRIT_D = params.D || params.n;
  // The body's own shear scale, (U/D)^2 -- d3-criterion.mjs's qRef. ONE
  // expression, used by the criterion's absolute threshold here, by
  // debugCriterion's ladder and by the scalar volume's Q normalization
  // (M6.2): three places that must agree about what "Q = 0.1" means.
  const Q_REF_SCALE = (CRIT_U && CRIT_D) ? (CRIT_U / CRIT_D) ** 2 : 1;
  const Q_ABS = Q_THRESH * Q_REF_SCALE;
  // Octaves of rotation rate per further level, and the hysteresis band.
  // 2D ships INC = 1 and a one-octave REFINE/COARSEN gap; both are knobs here
  // for the same reason they are there -- the right values are properties of
  // the flow.
  const Q_INC = numParam('qinc', 1);
  const Q_HYST = numParam('qhyst', 1);
  if (Q_ON) {
    // The criterion writes on LEVEL 1's block grid, and every level's `decide`
    // indexes it by shifting its own block id down. Those two are derived
    // separately -- here from NX/RB, there from d3-amr.mjs's poolAtLevel -- so
    // they are checked against each other rather than assumed equal. A
    // mismatch would not crash; it would flag the wrong blocks.
    const nbCrit = AMR ? hier.byLevel[CRIT_LEVEL].pool.nb : CRIT_NB;
    if (CRIT_NB.some((n, i) => n !== nbCrit[i])) {
      statusEl.textContent = `error: the Q-criterion grid ${CRIT_NB.join('x')} does not match`
        + ` level ${CRIT_LEVEL}'s block grid ${nbCrit.join('x')}`;
      return;
    }
    if (!AMR) {
      statusEl.textContent = 'error: ?qthresh= is a refinement criterion and needs ?levels>=2;'
        + ' debugCriterion() reports it read-only on a dense run';
      return;
    }
    if (!DYNAMIC) {
      // A field criterion on a STATIC set is a contradiction: the set is
      // chosen once at reset, from a flow that has not developed, and can
      // never follow anything. Refused rather than quietly refining a
      // t = 0 wake forever.
      statusEl.textContent = 'error: ?qthresh= needs ?dynamic=1 -- a field criterion on a set'
        + ' chosen once at reset would refine a flow that has not developed yet';
      return;
    }
    if (CRIT_BLK_L0 < 1) {
      statusEl.textContent = `error: ?qthresh= at ?levels=${LEVELS} rb=${RB} gives ${CRIT_BLK_L0}`
        + ' L0 cells per candidate block; the criterion reads the L0 field and cannot tell'
        + ' siblings apart below one cell (plans/3D.md M8.4)';
      return;
    }
  }
  const critModule = device.createShaderModule({
    code: await loadShader(`shaders/d3_criterion_q${Q}.wgsl`), label: `d3_criterion_q${Q}` });
  const critPipe = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [critBGL] }),
    compute: { module: critModule, entryPoint: 'reduceQ', constants: {
      ...dims, BLK_L0: CRIT_BLK_L0,
      NBX: CRIT_NB[0], NBY: CRIT_NB[1], NBZ: CRIT_NB[2], LEAD: CRIT_LEAD } },
  });

  // --- AMR pipelines --------------------------------------------------------
  // tau_fine = 2*tau_coarse - 0.5 (the acoustic-scaling relation this
  // project uses at every level). Derived HERE, once, and handed to the
  // kernels as OMEGA_FINE / TAU_COARSE, so there is a single place that
  // owns the level -> tau mapping rather than each shader re-deriving it.
  const TAU_COARSE = params.tau;
  const TAU_FINE = 2 * TAU_COARSE - 0.5;
  // Level 0 is TAU_COARSE and each level down doubles the distance from 1/2,
  // which is the acoustic-scaling relation the depth-2 pair already encodes.
  const tauAtLevel = (m) => { let t = TAU_COARSE; for (let k = 0; k < m; k++) t = 2 * t - 0.5; return t; };
  // A level-m pool kernel works in level-(m-1) CELL UNITS, because that is
  // what fineToCoarseUnit3 returns; the body lives in L0 units. One rung is
  // L0 = 0.5*u - 0.25 (refinement is cell-centred, so it is affine, not a
  // scale), giving 2^-(m-1) and -0.5*(1 - 2^-(m-1)) over m-1 rungs. Identity
  // at m = 1, which is why every depth-2 number is untouched by this.
  const bodyFrameAt = (m) => ({
    L0_SCALE: 2 ** -(m - 1),
    L0_OFFSET: -0.5 * (1 - 2 ** -(m - 1)),
    CHI_SCALE: 2 ** -m,
  });
  // 1 puts the grid transfers back on the PRE-collision Dupuis-Chopard
  // factor, which is wrong for this solver's post-collision buffers and was
  // the M3 interface bug -- kept switchable so the defect can be measured
  // in the shipped build (shaders/common_d3_pool.wgsl derives both, and
  // tools/analyze-d3-interface.js is what drives the comparison).
  const DC_PRE = urlParams.get('dcpre') === '1' ? 1 : 0;
  // M4.1c. The explosion is LINEAR by default; ?explin=0 restores M4.1b's
  // uniform one so the two can be A/B'd on the same GPU in the same session.
  // Only meaningful with ?interface=explode.
  const EXPLODE_LINEAR = urlParams.get('explin') === '0' ? 0 : 1;
  // TIMING ONLY. ?orphans=0 drops the coalesce orphan pass, which makes the
  // interface WRONG -- it reinstates M4.1b's convex-edge mass leak. It is here
  // so tools/bench-d3-interface.js can price the pass by differencing two runs
  // of one build, the way plans/perf-characterization.md requires.
  const ORPHANS = urlParams.get('orphans') === '0' ? 0 : 1;
  let interpGhostPipe = null, interpFullPipe = null, step1Pipe = null, avgPipe = null;
  let interpBG = null, step1BG_AB = null, step1BG_BA = null, avgBGA = null, avgBGB = null;
  let fluxPipeSet = null, fluxPipeAdd = null, refluxPipe = null;
  let fluxBGA = null, fluxBGB = null, refluxBG = null;
  let explodePipe = null, coalescePipe = null;
  let explodeBG = null, coalesceBG = null;
  let forcePoolPipe = null, forcePoolBG = null, forcePoolDisp = null;
  // The manager's per-level pipelines live on L[m] (M5.5b); only the
  // criterion is a single pipeline, because only the finest level evaluates
  // it. manageBG is level 1's bind group, still named by the depth-2 aliases
  // below.
  let manageBG = null, manageDecidePipe = null;
  let fillPipe = null, drainPipe = null;
  // Per level, created only when there is a pool. See common_d3_moments.wgsl
  // for why a seeded level needs its `mac` filled explicitly.
  let momentsPipe = null;

  // M4's interface flux correction. OPT-IN (?reflux=1), and NOT the default
  // -- it does what it was built to do and that turned out not to be enough.
  // It restores exact global conservation (measured: the momentum drift a
  // partially-refined run leaks stops dead). On a seam with no convex corner
  // it also halves the field error. On a seam WITH one it is much worse than
  // no correction at all, because the fine lattice's diagonal channels
  // cannot tile the coarse one across a corner -- see plans/3D.md M4 and
  // shaders/common_d3_amr_flux.wgsl. Default off until that is solved.
  //
  // Two further situations make its accounting invalid rather than merely
  // inaccurate, and both fail loudly:
  //
  //   walls   the balance assumes every coarse population goes exactly one
  //           place. A wall bounce-back reflects instead, so a refined
  //           region touching a walled face would be silently mis-counted.
  //           No walled scenario uses AMR today; this is the guard for when
  //           one does. (A BODY is handled differently -- the correction is
  //           skipped near the surface rather than refused, since a refined
  //           shell around a body has its seam in clean fluid. See
  //           common_d3_amr_reflux.wgsl.)
  //   nb < 3  the flux pass resolves a ring cell against a tile that may not
  //           own it, using a centred periodic wrap. With fewer than three
  //           blocks on an axis a tile's ring wraps onto itself and the wrap
  //           is ambiguous.
  // M4.1e, 2026-09-10: `explode` is the DEFAULT. Chen et al. (2006)'s
  // explode/coalesce is exactly conservative in mass and momentum where the
  // M3 `interp` coupling is not, its field error tracks the no-interface
  // control to within 8-10% where interp is 3.6x worse, it reproduces the
  // dense sphere Cd to 0.07% where interp is 0.63%, and it costs +2.9% per
  // macro-step on a body-fitted shell and less than nothing at larger
  // refined fractions. plans/3D.md M4.1b-e has every number.
  //
  // `interp` is kept switchable because it is the only way to A/B the
  // conversion in one build, and because benchmarks/d3.json's amr-box-RB4
  // pins it as the control that says this work did not disturb the old
  // path -- not because it is a recommended mode. It is NOT conservative.
  const IFACE = urlParams.get('interface') || 'explode';
  if (!['explode', 'interp'].includes(IFACE)) throw new Error(`?interface=${IFACE}: expected explode or interp`);
  const EXPLODE = AMR && IFACE === 'explode';
  // Only the explode/coalesce coupling has a pool-parent variant (M5.2b-ii).
  // ?interface=interp is the superseded M3 path, kept switchable at depth 2
  // purely to A/B the conversion; giving it a second implementation at depth
  // would be two copies of a coupling that is not the default and is known
  // not to be conservative.
  if (LEVELS >= 3 && !EXPLODE) {
    statusEl.textContent = `error: ?levels=${LEVELS} needs ?interface=explode`
      + ' (the pool-parent coupling exists only there, plans/3D.md M5.2b-ii)';
    return;
  }
  // M5.4's HARD REQUIREMENT, as two preconditions. The body lives entirely on
  // the finest level; these are what make that enforced rather than assumed.
  //
  // 1. A body with AMR must be GEOMETRY-REFINED. With ?refine=box the body
  //    can sit wholly outside the refined region, and since the coarse force
  //    pass is no longer dispatched under AMR, the integrated force would be
  //    exactly ZERO -- silently. Refused instead.
  if (AMR && HAS_BODY && !geomForced) {
    statusEl.textContent = `error: a body with ?levels=${LEVELS} needs ?refine=body`
      + ` (the body must live entirely on the finest level, plans/3D.md M5.4);`
      + ` ?refine=${refineMode} is a fixed region that need not contain it`;
    return;
  }
  // 2. The margin must cover the force stencil's REACH. Bounce-back exchanges
  //    momentum across links crossing the surface (one cell); the diffuse
  //    penalty spans CHI_EPS. Both are in L0 cell units, and both must lie
  //    inside the refined region or the finest level's pass would miss part
  //    of the body. This gets SAFER with depth -- the finest cells shrink
  //    while the margin stays in L0 units -- so it is checked at level 1's
  //    scale, the tightest case.
  if (AMR && HAS_BODY) {
    const reach = USE_BOUNCEBACK ? 1 : CHI_EPS;
    if (geomForced.margin < reach) {
      statusEl.textContent = `error: ?margin=${geomForced.margin} is below the force stencil's`
        + ` reach (${reach} coarse cells for ${USE_BOUNCEBACK ? 'bounce-back' : 'diffuse chi'});`
        + ' part of the body would fall outside the finest level (plans/3D.md M5.4)';
      return;
    }
  }
  // M5.5b LIFTED THE LAST DEPTH REFUSAL. ?levels>=3 with ?dynamic=1 used to
  // stop here, because the manager had no 2:1 balance pass and could
  // therefore build a tree the invariant forbids. It now has one (M5.5a's
  // `completeOctets`/`balance`, scored against d3-amr.mjs's cascade21 on
  // real GPU data) and an allocator that runs at every level in the order
  // the tree requires. The gate is validate-d3-invariants.js's `drift3`.
  // M4.1d lifted the "explode cannot carry a body" restriction: the body
  // force is now integrated on the level that OWNS each region -- the coarse
  // kernel masks out cells a refined block covers and
  // shaders/common_d3_force_pool.wgsl sums those at fine resolution. Before
  // that, the L0 reduction under a refined region was reading whatever
  // coalesce had not written, which is stale, not merely coarse.

  const REFLUX = (() => {
    if (!AMR) return 0;
    if (urlParams.get('reflux') !== '1') return 0;
    // The flux correction measures and patches a mismatch that
    // explode/coalesce does not create. Running both would correct a seam
    // that is already conservative.
    if (EXPLODE) throw new Error('?reflux=1 needs ?interface=interp: explode/coalesce is conservative by construction');
    const walled = ['x', 'y', 'z'].some(a => params.walls.includes(a));
    if (walled) throw new Error(`?reflux: not valid with walls (?walls=${params.walls}); pass ?reflux=0 to run without the flux correction`);
    if (pool.nb.some(n => n < 3)) throw new Error(`?reflux: needs at least 3 blocks per axis, have ${pool.nb.join('x')} at ?rb=${RB}`);
    return 1;
  })();
  if (AMR) {
    const poolConst = { NX, NY, NZ, RB };
    const interpModule = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_interp_q${Q}.wgsl`), label: `d3_amr_interp_q${Q}` });
    const step1Module = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_step1_q${Q}.wgsl`), label: `d3_amr_step1_q${Q}` });
    const avgModule = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_average_q${Q}.wgsl`), label: `d3_amr_average_q${Q}` });

    const interpBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]});
    const step1BGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
              { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]});
    // 0 f_pool (ro), 1 the PARENT's f (rw), 2 slotToBlock (ro), 3 blockWant
    // (ro), 4 the PARENT's mac (rw). M5.5b swapped 3 and 4 so that binding 4
    // is the parent's macroscopic array in every kernel that writes one --
    // average as well as coalesce -- because that is the binding
    // common_d3_parentmac_{dense,pool}.wgsl declares, and that fragment pair
    // is the only thing standing between the two layouts and a silent
    // transposition. See its header for what one cost.
    const avgBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]});

    const mk = (bgl, module, constants) => device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: 'main', constants },
    });
    // Two interp pipelines: the steady-state ring refresh, and the one-time
    // full-tile fill at reset.
    interpGhostPipe = await mk(interpBGL, interpModule, { ...poolConst, TAU_COARSE, DC_PRE, GHOST_ONLY: 1, TIME_BLEND: 0.0 });
    interpFullPipe = await mk(interpBGL, interpModule, { ...poolConst, TAU_COARSE, DC_PRE, GHOST_ONLY: 0, TIME_BLEND: 0.0 });
    step1Pipe = await mk(step1BGL, step1Module, {
      ...poolConst,
      // A finer level below this one means this level stops solving where it
      // is covered -- what the dense step has always done. 0 at ?levels=2,
      // so the lookup folds out and the build is bit-identical.
      HAS_CHILD: L[2] ? 1 : 0,
      // Chen's coalesce averages advected-but-UNCOLLIDED interface states,
      // so under explode/coalesce the ring advects and stores only.
      COLLIDE_RING: EXPLODE ? 0 : 1,
      ...bodyFrameAt(1),
      OMEGA_FINE: 1 / TAU_FINE,
      FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
      HAS_BODY, USE_BOUNCEBACK, CHI_EPS, SOLID_EQ, RHO_W_LOCAL, ...WINC,
      SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2], ...SPONGE_AXES,
    });
    avgPipe = await mk(avgBGL, avgModule, { ...poolConst, TAU_COARSE, DC_PRE });
    // A level's `mac` from its own `f`. ONE pipeline for every level: the
    // kernel addresses a pool slot and nothing in it reads NX/NY/NZ, so
    // unlike the manager's passes there is no per-level frame to get wrong.
    // The bind group is per level because the buffers are.
    {
      const momentsModule = device.createShaderModule({
        code: await loadShader(`shaders/d3_moments_q${Q}.wgsl`), label: `d3_moments_q${Q}` });
      const momentsBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ]});
      momentsPipe = await mk(momentsBGL, momentsModule, poolConst);
      for (let m = 1; m < LEVELS; m++) {
        const lv = L[m];
        lv.momentsBG = device.createBindGroup({ layout: momentsBGL, entries: [
          { binding: 0, resource: { buffer: lv.fA } }, { binding: 1, resource: { buffer: lv.mac } },
          { binding: 2, resource: { buffer: lv.slotToBlock } }]});
      }
    }
    // M4.2b-ii. Third pipelines over the SAME two modules and the same two
    // layouts: a tile being born wants exactly interp's coarse->fine
    // transfer, and one being absorbed wants exactly average's restriction.
    // Separate shaders would be two more copies of transfers this file
    // already has, which is how they drift apart.
    if (DYNAMIC) {
      fillPipe = await mk(interpBGL, interpModule,
        { ...poolConst, TAU_COARSE, DC_PRE, GHOST_ONLY: 0, TIME_BLEND: 0.0, NEW_ONLY: 1 });
      drainPipe = await mk(avgBGL, avgModule, { ...poolConst, TAU_COARSE, DC_PRE, DYING_ONLY: 1 });
      L[1].fillPipe = fillPipe;
      L[1].drainPipe = drainPipe;
    }

    // M4.2b-i: the dynamic-refinement manager. TWO pipelines over ONE module
    // and ONE bind group -- the split is into separate PASSES, not separate
    // resources; see common_d3_manage.wgsl on why they must not share a
    // dispatch.
    if (geomForced) {
      const manageModule = device.createShaderModule({
        code: await loadShader(`shaders/d3_manage_q${Q}.wgsl`), label: `d3_manage_q${Q}` });
      const manageBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        // M8.4: per-block max Q from the field criterion, on the FINEST
        // level's grid. Bound on every level's group like childWant, so the
        // layout does not fork; read only by `decide`, and only when HAS_Q.
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ]});
      // MARGIN comes from the SAME value refineNearBody used to build the
      // static set, not from a second read of ?margin=: the bit-identical
      // gate needs the two criteria to agree exactly, and two independent
      // parses of one parameter is how they would silently stop agreeing.
      const manageConst = { ...poolConst, HAS_BODY, ...WINC, PLANAR: params.planar ? 1 : 0,
        // OFF unless ?qthresh= was given, at which point Q_ABS folds the
        // whole field test out at pipeline-creation time and every scenario
        // that predates M8.4 is bit-identical.
        HAS_Q: Q_ON ? 1 : 0,
        // The ladder's base, derived from the MEASURED `?qthresh=` rather
        // than typed as a second calibration: sqrt(Q) is the rotation rate
        // the ladder eats, so the base is log2 of the threshold's own
        // rotation rate. COARSEN sits ?qhyst= octaves below it -- the
        // anti-flicker band, without which a block on a rung refines and
        // coarsens on alternate evaluations and re-interpolates its region
        // every time.
        REFINE_THRESH: 0.5 * Math.log2(Math.max(Q_ABS, 1e-30)),
        COARSEN_THRESH: 0.5 * Math.log2(Math.max(Q_ABS, 1e-30)) - Q_HYST,
        N_REFINE_INC: Q_INC, N_REFINE_MAX: 1e30, MAX_LEVEL: LEVELS - 1,
        CNBX: CRIT_NB[0], CNBY: CRIT_NB[1], CNBZ: CRIT_NB[2],
        MARGIN: Number.isFinite(MANAGE_MARGIN) ? MANAGE_MARGIN : geomForced.margin,
        // The lead term is zero for a pinned body, so the criterion still
        // reproduces the host's initial set exactly and M4.2b-i's
        // bit-identical gate is untouched.
        MANAGE_EVERY };
      // THE CRITERION RUNS AT THE FINEST LEVEL ONLY (M5.1b), so `decide` is
      // one pipeline, built in that level's frame. Every coarser level's
      // want is derived by the closure below, never re-decided.
      const deepest = LEVELS - 1;
      const mkManageAt = (entry, m, extra) => device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
        compute: { module: manageModule, entryPoint: entry, constants: {
          ...manageConst,
          NX: NX * 2 ** (m - 1), NY: NY * 2 ** (m - 1), NZ: NZ * 2 ** (m - 1),
          BOX_SCALE: 2 ** -(m - 1), ...(extra || {}) } },
      });
      // PER LEVEL NOW, because the ladder grades: each level asks whether the
      // depth the flow wants reaches it. The finest level's pipeline is the
      // one the geometry criterion runs through (M5.1b is unchanged -- geometry
      // is still evaluated at the finest level only, and cascade21 forces its
      // ancestors); the coarser ones exist for the FIELD half.
      manageDecidePipe = await mkManageAt('decide', deepest, { LEVEL_M: deepest });
      for (let m = 1; m < LEVELS - 1; m++) {
        // HAS_BODY off below the finest level: geometry is evaluated there and
        // only there, and letting a coarser level re-evaluate it would refine
        // a shell the cascade is already responsible for -- two mechanisms for
        // one requirement, which is what M4.2b-i removed.
        L[m].decidePipe = await mkManageAt('decide', m, { LEVEL_M: m, HAS_BODY: 0 });
      }
      // The 2:1 cascade, per level: octet completion at m, then the parent
      // gather that writes m-1 from m. Both are one-thread-per-output, so
      // neither has a race to reason about -- see the shader.
      for (let m = 2; m < LEVELS; m++) L[m].octetsPipe = await mkManageAt('completeOctets', m);
      for (let m = 1; m < LEVELS - 1; m++) L[m].balancePipe = await mkManageAt('balance', m);
      // `balance` ORs into its level's want, so every level the criterion
      // does not write must be zeroed first -- and by a KERNEL, not a
      // writeBuffer, which is ordered at submit rather than where it is
      // encoded. See clearWant's own note.
      for (let m = 1; m < LEVELS; m++) L[m].clearWantPipe = await mkManageAt('clearWant', m);
      if (DYNAMIC) {
        // PER LEVEL (M5.5b). These used to be three pipelines built from
        // level 1's constants, which is the same silence M5.0 refused
        // ?levels=3 for: nbx() comes from NX, so a level-2 dispatch through
        // a level-1 pipeline would walk an eighth of the blocks and call the
        // rest unrefined.
        for (let m = 1; m < LEVELS; m++) {
          L[m].coarsenPipe = await mkManageAt('coarsen', m);
          L[m].refinePipe = await mkManageAt('refine', m);
          L[m].clearNewPipe = await mkManageAt('clearNew', m);
        }
      }
      const wantDummy = { blockWant: device.createBuffer({ size: 4, usage: U.STORAGE }) };
      manageBG = device.createBindGroup({ layout: manageBGL, entries: [
        { binding: 0, resource: { buffer: blockSlotBuf } },
        { binding: 1, resource: { buffer: slotToBlockBuf } },
        { binding: 2, resource: { buffer: freeListBuf } },
        { binding: 3, resource: { buffer: freeCountBuf } },
        { binding: 4, resource: { buffer: bodyBuf } },
        { binding: 5, resource: { buffer: blockWantBuf } },
        { binding: 6, resource: { buffer: slotNewBuf } },
        { binding: 7, resource: { buffer: wantDummy.blockWant } },
        { binding: 8, resource: { buffer: critBuf } }]});
      // Per level, so the closure can be dispatched down the tree. Binding 5
      // is THIS level's want and binding 7 is the CHILD's; at the deepest
      // level the child slot is bound to itself, which `balance` never reads
      // there because no pipeline is created for it.
      for (let m = 1; m < LEVELS; m++) {
        const lv = L[m];
        lv.manageBG = device.createBindGroup({ layout: manageBGL, entries: [
          { binding: 0, resource: { buffer: lv.blockSlot } },
          { binding: 1, resource: { buffer: lv.slotToBlock } },
          { binding: 2, resource: { buffer: lv.freeList } },
          { binding: 3, resource: { buffer: lv.freeCount } },
          { binding: 4, resource: { buffer: bodyBuf } },
          { binding: 5, resource: { buffer: lv.blockWant } },
          { binding: 6, resource: { buffer: lv.slotNew } },
          // NEVER `lv.blockWant` itself: binding 5 is writable and 7 is
          // read-only, and WebGPU rejects one buffer in both roles in a
          // single dispatch -- which it does by SKIPPING the dispatch, so
          // the symptom is a manager that decides nothing at all. Found
          // exactly that way. The deepest level's `balance` pipeline does
          // not exist, so what sits here is never read.
          { binding: 7, resource: { buffer: (L[m + 1] || wantDummy).blockWant } },
          // The SAME criterion buffer on every level's group. It is indexed
          // on the finest level's grid, which is the only level `decide`
          // runs on -- the coarser levels' groups carry it so the layout
          // stays one layout, exactly as binding 7 does.
          { binding: 8, resource: { buffer: critBuf } }]});
      }
    }

    // M4.1d: the fine level's own force/torque reduction. Created from the
    // SAME USE_BOUNCEBACK/CHI_EPS the step and coarse-force pipelines got --
    // three kernels over one pair of values, so a mismatch cannot arise from
    // three separate copies of them.
    if (HAS_BODY) {
      const forcePoolModule = device.createShaderModule({
        code: await loadShader(`shaders/d3_force_pool_q${Q}.wgsl`), label: `d3_force_pool_q${Q}` });
      // THE FINEST LEVEL, and only it (plans/3D.md M5.4b). The body lives
      // entirely there by hard requirement, so this one pass integrates the
      // whole of it and there is nothing to mask or partition. DX_WEIGHT is
      // 4^-m: the cross-level weight is dx^(D-1) = dx^2 and dx = 2^-m, so
      // 0.25 at depth 2 and 0.0625 at depth 3. It used to be baked at L1's
      // 0.25, which at depth 3 would have reported four times the drag.
      const fine = L[LEVELS - 1];
      const finePC = { NX: NX * 2 ** (LEVELS - 2), NY: NY * 2 ** (LEVELS - 2), NZ: NZ * 2 ** (LEVELS - 2), RB };
      // L0_SCALE / L0_OFFSET convert this level's parent-unit positions into
      // the L0 frame the body lives in -- the identity at level 1, affine
      // below it because refinement is cell-centred. See the shader.
      const m = LEVELS - 1;
      forcePoolPipe = await mk(forcePoolBGL, forcePoolModule, {
        // TWO WEIGHTS, NOT ONE. dx^(D-1) = 4^-m for the link sum, which is a
        // momentum per FINE timestep; dx^D = 8^-m for the swept sum, which is
        // a momentum per MACRO step because that is how often the body's pose
        // moves. See common_d3_force_pool.wgsl's SWEPT_WEIGHT header -- one
        // weight for both would over-charge a moving body by 2^m.
        ...finePC, USE_BOUNCEBACK, CHI_EPS, SWEPT_FORCE, RHO_W_LOCAL,
        DX_WEIGHT: 4 ** -m, SWEPT_WEIGHT: 8 ** -m, ...bodyFrameAt(m), ...WINC,
      });
      // Reads the pool buffer substep A will read, i.e. that level's time-t
      // state, matching the coarse kernel's own pre-streaming read.
      forcePoolBG = device.createBindGroup({ layout: forcePoolBGL, entries: [
        { binding: 0, resource: { buffer: fine.fA } }, { binding: 1, resource: { buffer: bodyBuf } },
        { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: fine.slotToBlock } }]});
      // Computed here rather than read from `fine.tileDisp`: the per-level
      // dispatch shapes are assigned below this block, so that field is not
      // populated yet. Same expression, same source (SLOTS).
      forcePoolDisp = [pool.FB / 4, pool.FB / 4, (pool.FB / 4) * Math.max(1, SLOTS[LEVELS - 1])];
    }

    // interp needs BOTH coarse states (t and t+dt, which are simply the two
    // ping-pong buffers once the coarse step runs first) and writes whichever
    // pool buffer the next substep will READ -- so four bind groups:
    // coarse parity x pool target. interpBG[coarseParity][poolTarget].
    const mkInterp = (c0, c1, dst) => device.createBindGroup({ layout: interpBGL, entries: [
      { binding: 0, resource: { buffer: c0 } }, { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: c1 } },
      { binding: 4, resource: { buffer: slotNewBuf } }]});
    interpBG = [
      [mkInterp(fA, fB, fPoolA), mkInterp(fA, fB, fPoolB)],   // coarse t in fA
      [mkInterp(fB, fA, fPoolA), mkInterp(fB, fA, fPoolB)],   // coarse t in fB
    ];
    // Binding 6 is the CHILD level's blockSlot where there is one and this
    // level's own where there is not -- a harmless self-reference that keeps
    // ONE layout, since HAS_CHILD = 0 folds every read of it away.
    const mkStep = (a, b) => device.createBindGroup({ layout: step1BGL, entries: [
      { binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: macPool } }, { binding: 3, resource: { buffer: bodyBuf } },
      { binding: 4, resource: { buffer: slotToBlockBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } },
      { binding: 6, resource: { buffer: (L[2] || L[1]).blockSlot } }]});
    step1BG_AB = mkStep(fPoolA, fPoolB);
    step1BG_BA = mkStep(fPoolB, fPoolA);
    L[1].step1Pipe = step1Pipe;
    // Indexed by PARITY: entry p reads this level's buffer p and writes the
    // other, so two substeps return the state to buffer 0.
    L[1].step1BG = [step1BG_AB, step1BG_BA];
    // average writes the coarse buffer the COARSE step just wrote, so it
    // also needs one per parity.
    const mkAvg = (dst) => device.createBindGroup({ layout: avgBGL, entries: [
      { binding: 0, resource: { buffer: fPoolA } }, { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: blockWantBuf } },
      { binding: 4, resource: { buffer: mac } }]});
    avgBGA = mkAvg(fA);
    avgBGB = mkAvg(fB);
    // M5.5b: level 1's DRAIN and FILL, indexed by COARSE PARITY. Both act on
    // the parent's time-t buffer, which at level 1 is the one L0 ping-pongs
    // -- and only at level 1: every deeper level begins and ends a macro-step
    // in its own fA (advanceLevel takes parity 0 then 1), so the pool-parent
    // pair below is parity-free and the index is ignored there.
    L[1].drainBG = [avgBGA, avgBGB];
    L[1].fillBG = [interpBG[0][0], interpBG[1][0]];

    // --- M4.1b explode / coalesce ----------------------------------------
    if (EXPLODE) {
      const explodeModule = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_explode_q${Q}.wgsl`), label: `d3_amr_explode_q${Q}` });
      const coalesceModule = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_coalesce_q${Q}.wgsl`), label: `d3_amr_coalesce_q${Q}` });
      const explodeBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ]});
      const coalesceBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]});
      explodePipe = await mk(explodeBGL, explodeModule, { ...poolConst, EXPLODE_LINEAR });
      coalescePipe = await mk(coalesceBGL, coalesceModule, { ...poolConst, ORPHANS });
      // Explode reads the coarse field at t and writes the ring of whichever
      // pool buffer substep A will read; coalesce writes back into that SAME
      // time-t coarse buffer, at the covered cells, so the coarse step gathers
      // it with no per-direction test. Both are indexed by coarse parity.
      const mkExplode = (src) => device.createBindGroup({ layout: explodeBGL, entries: [
        { binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: fPoolA } },
        { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: blockSlotBuf } }]});
      explodeBG = [mkExplode(fA), mkExplode(fB)];
      const mkCoalesce = (dst) => device.createBindGroup({ layout: coalesceBGL, entries: [
        { binding: 0, resource: { buffer: fPoolA } }, { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: blockSlotBuf } }, { binding: 3, resource: { buffer: macPool } },
        { binding: 4, resource: { buffer: mac } }]});
      coalesceBG = [mkCoalesce(fA), mkCoalesce(fB)];
      // Level 1 joins the SAME per-level record every deeper level uses, so
      // M5.3's schedule can walk levels instead of naming them. The level-1
      // entries differ from the rest in exactly one way -- its coalesce is
      // dispatched over the dense L0 grid rather than over parent tiles --
      // and that is carried as a dispatch shape, not as a branch.
      L[1].explodePipe = explodePipe; L[1].explodeBG = explodeBG;
      L[1].coalescePipe = coalescePipe; L[1].coalesceBG = coalesceBG;

      // --- M5.2b-ii: the same coupling with a POOL parent, level >= 2 ----
      //
      // Same two modules' worth of physics, assembled with
      // common_d3_parent_pool.wgsl instead of the dense fragment, and with
      // coalesce's parent-tile dispatch. The only per-level values are
      // poolConst's NX/NY/NZ -- read as THE PARENT GRID's dims, which is
      // what makes every other function in common_d3_pool.wgsl level-generic
      // (M5.2b-i) -- and the relaxation time.
      //
      // Bindings 8 and 9 are the parent's own pool bookkeeping and exist
      // only on this path; the dense parent needs neither, because a dense
      // cell is addressed arithmetically and always exists.
      for (let m = 2; m < LEVELS; m++) {
        const par = L[m - 1], lv = L[m];
        const pc = { NX: NX * 2 ** (m - 1), NY: NY * 2 ** (m - 1), NZ: NZ * 2 ** (m - 1), RB };
        const explodePoolModule = device.createShaderModule({
          code: await loadShader(`shaders/d3_amr_explode_pool_q${Q}.wgsl`), label: `d3_amr_explode_pool_q${Q}` });
        const coalescePoolModule = device.createShaderModule({
          code: await loadShader(`shaders/d3_amr_coalesce_pool_q${Q}.wgsl`), label: `d3_amr_coalesce_pool_q${Q}` });
        const ro = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
        const rw = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
        const explodePoolBGL = device.createBindGroupLayout({ entries: [ro(0), rw(1), ro(2), ro(3), ro(8)] });
        const coalescePoolBGL = device.createBindGroupLayout({ entries: [ro(0), rw(1), ro(2), ro(3), rw(4), ro(8), ro(9)] });
        lv.explodePipe = await mk(explodePoolBGL, explodePoolModule, { ...pc, EXPLODE_LINEAR });
        lv.coalescePipe = await mk(coalescePoolBGL, coalescePoolModule, { ...pc, ORPHANS });
        lv.step1Pipe = await mk(step1BGL, step1Module, {
          ...pc,
          COLLIDE_RING: EXPLODE ? 0 : 1,
          HAS_CHILD: L[m + 1] ? 1 : 0,
          ...bodyFrameAt(m),
          OMEGA_FINE: 1 / tauAtLevel(m),
          FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
          HAS_BODY, USE_BOUNCEBACK, CHI_EPS, SOLID_EQ, RHO_W_LOCAL, ...WINC,
          SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2], ...SPONGE_AXES,
        });
        // Explode reads the parent buffer holding time t -- the one the
        // parent's own upcoming substep will READ -- and writes the ring of
        // the buffer this level's substep A reads. Coalesce writes back into
        // that same parent buffer, exactly as the depth-2 pair does against
        // L0. Indexed by parent parity for the same reason.
        const mkEx = (src) => device.createBindGroup({ layout: explodePoolBGL, entries: [
          { binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: lv.fA } },
          { binding: 2, resource: { buffer: lv.slotToBlock } }, { binding: 3, resource: { buffer: lv.blockSlot } },
          { binding: 8, resource: { buffer: par.blockSlot } }]});
        lv.explodeBG = [mkEx(par.fA), mkEx(par.fB)];
        const mkCo = (dst) => device.createBindGroup({ layout: coalescePoolBGL, entries: [
          { binding: 0, resource: { buffer: lv.fA } }, { binding: 1, resource: { buffer: dst } },
          { binding: 2, resource: { buffer: lv.blockSlot } }, { binding: 3, resource: { buffer: lv.mac } },
          // The PARENT's mac pool, not L0's: coalesce republishes the
          // macroscopic field under the refined region, and at depth the
          // cells it speaks for live in the parent's pool. `parentIndex`
          // addresses both the same way, 4 floats per cell.
          { binding: 4, resource: { buffer: par.mac ?? mac } },
          { binding: 8, resource: { buffer: par.blockSlot } },
          { binding: 9, resource: { buffer: par.slotToBlock } }]});
        lv.coalesceBG = [mkCo(par.fA), mkCo(par.fB)];
        const mkStepL = (a, b) => device.createBindGroup({ layout: step1BGL, entries: [
          { binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } },
          { binding: 2, resource: { buffer: lv.mac } }, { binding: 3, resource: { buffer: bodyBuf } },
          { binding: 4, resource: { buffer: lv.slotToBlock } }, { binding: 5, resource: { buffer: lv.blockSlot } },
          { binding: 6, resource: { buffer: (L[m + 1] || lv).blockSlot } }]});
        lv.step1BG = [mkStepL(lv.fA, lv.fB), mkStepL(lv.fB, lv.fA)];

        // AND THE SEED. A level's pool starts as whatever the GPU zeroed it
        // to, and zero is not a fluid state: rho = 0 there, so the first
        // coalesce hands the parent nothing and the run detonates in one
        // step. That is exactly how this was found -- with L2 never
        // stepped, coalesce alone still blew L1 up, which pointed at L2's
        // CONTENTS rather than at the pass.
        //
        // Level 1 is seeded by interpFullPipe from the L0 initial condition
        // (see reset()); every level below it needs the same transfer from
        // ITS parent, which is the same module with the pool-parent
        // fragment. Seeding must cascade in order -- L2 reads an L1 that has
        // already been filled.
        const interpPoolModule = device.createShaderModule({
          code: await loadShader(`shaders/d3_amr_interp_pool_q${Q}.wgsl`), label: `d3_amr_interp_pool_q${Q}` });
        const interpPoolBGL = device.createBindGroupLayout({ entries: [ro(0), rw(1), ro(2), ro(3), ro(4), ro(8)] });
        lv.interpFullPipe = await mk(interpPoolBGL, interpPoolModule,
          { ...pc, TAU_COARSE: tauAtLevel(m - 1), DC_PRE, GHOST_ONLY: 0, TIME_BLEND: 0.0 });
        lv.interpSeedBG = device.createBindGroup({ layout: interpPoolBGL, entries: [
          { binding: 0, resource: { buffer: par.fA } }, { binding: 1, resource: { buffer: lv.fA } },
          { binding: 2, resource: { buffer: lv.slotToBlock } }, { binding: 3, resource: { buffer: par.fA } },
          { binding: 4, resource: { buffer: lv.slotNew } }, { binding: 8, resource: { buffer: par.blockSlot } }]});

        // --- M5.5b: the allocator's two transfers, at depth ---------------
        //
        // FILL is the seed pipeline with NEW_ONLY: same module, same bind
        // group, one override apart. A tile the manager has just allocated
        // wants exactly the coarse->fine transfer the seed performs; the
        // only difference is that the seed fills every tile and the fill
        // fills the ones `refine` flagged.
        //
        // The bind group is SHARED rather than rebuilt because it is
        // literally the same resources: the parent's time-t buffer is its
        // fA at every level >= 2 (see L[1].fillBG's note), and the
        // destination is this level's fA, which is the buffer its next
        // substep reads.
        lv.fillPipe = DYNAMIC ? await mk(interpPoolBGL, interpPoolModule,
          { ...pc, TAU_COARSE: tauAtLevel(m - 1), DC_PRE, GHOST_ONLY: 0, TIME_BLEND: 0.0, NEW_ONLY: 1 }) : null;
        lv.fillBG = [lv.interpSeedBG, lv.interpSeedBG];

        // DRAIN is the pool-parent sibling of level 1's `average` -- the
        // restriction that puts a dying tile's solution back into its parent
        // before `coarsen` frees the slot. It is the one piece M5.5b had to
        // build rather than re-parameterize, and it is built the same way
        // every other pool-parent kernel here is: the shared body with
        // common_d3_parent_pool.wgsl and common_d3_parentmac_pool.wgsl
        // spliced in, so there is no second copy of the transfer.
        if (DYNAMIC) {
          const avgPoolModule = device.createShaderModule({
            code: await loadShader(`shaders/d3_amr_average_pool_q${Q}.wgsl`), label: `d3_amr_average_pool_q${Q}` });
          const avgPoolBGL = device.createBindGroupLayout({ entries: [ro(0), rw(1), ro(2), ro(3), rw(4), ro(8)] });
          lv.drainPipe = await mk(avgPoolBGL, avgPoolModule,
            { ...pc, TAU_COARSE: tauAtLevel(m - 1), DC_PRE, DYING_ONLY: 1 });
          const drainBG = device.createBindGroup({ layout: avgPoolBGL, entries: [
            { binding: 0, resource: { buffer: lv.fA } }, { binding: 1, resource: { buffer: par.fA } },
            { binding: 2, resource: { buffer: lv.slotToBlock } }, { binding: 3, resource: { buffer: lv.blockWant } },
            // The PARENT's mac pool, planar -- the same array coalesce
            // republishes into, and for the same reason: a cell that stops
            // being covered must not keep the moments it had while it was.
            { binding: 4, resource: { buffer: par.mac } },
            { binding: 8, resource: { buffer: par.blockSlot } }]});
          lv.drainBG = [drainBG, drainBG];
        }
      }
    }

    // --- M4 interface flux correction ------------------------------------
    if (REFLUX) {
      const fluxModule = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_flux_q${Q}.wgsl`), label: `d3_amr_flux_q${Q}` });
      const refluxModule = device.createShaderModule({ code: await loadShader(`shaders/d3_amr_reflux_q${Q}.wgsl`), label: `d3_amr_reflux_q${Q}` });
      const fluxBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]});
      const refluxBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ]});
      // Two pipelines from one module: the substep-A pass WRITES, which is
      // what clears the accumulator, and the substep-B pass ADDS.
      fluxPipeSet = await mk(fluxBGL, fluxModule, { ...poolConst, ACCUM: 0 });
      fluxPipeAdd = await mk(fluxBGL, fluxModule, { ...poolConst, ACCUM: 1 });
      refluxPipe = await mk(refluxBGL, refluxModule, {
        ...poolConst, HAS_BODY, ...WINC,
        FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
      });
      const mkFlux = (src) => device.createBindGroup({ layout: fluxBGL, entries: [
        { binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: blockSlotBuf } },
        { binding: 2, resource: { buffer: fluxAccBuf } }]});
      // Each flux pass reads the pool buffer its substep STREAMS: substep A
      // streams fPoolA, substep B streams what A wrote into fPoolB.
      fluxBGA = mkFlux(fPoolA);
      fluxBGB = mkFlux(fPoolB);
      const mkReflux = (t0, t1) => device.createBindGroup({ layout: refluxBGL, entries: [
        { binding: 0, resource: { buffer: t0 } }, { binding: 1, resource: { buffer: t1 } },
        { binding: 2, resource: { buffer: mac } }, { binding: 3, resource: { buffer: blockSlotBuf } },
        { binding: 4, resource: { buffer: fluxAccBuf } }, { binding: 5, resource: { buffer: bodyBuf } }]});
      // [coarse parity]: which buffer holds t, which holds t + dt.
      refluxBG = [mkReflux(fA, fB), mkReflux(fB, fA)];
    }
  }

  // ONE STATEMENT OF THE SLICE VIEW'S OVERRIDES, because there are now two
  // pipelines built from them -- the canvas one here and the offscreen one
  // debugRenderFrame builds for the movie tool. Two copies of a constants
  // block that must match one WGSL file is exactly the shape of 238e48c, and
  // a divergence would show as a movie that did not look like the page.
  const renderConsts = {
    ...dims, RB, SAMPLE_LEVELS, ...WINC,
    // Pinned to 1 because the page pre-normalizes -- see U_SCALE/V_SCALE.
    VORT_SCALE: 1.0,
    // Below 1 lifts weak structure toward the top of the ramp instead of
    // leaving it in the near-black background; these fields are smooth and
    // have no vortex cores to protect from saturating.
    VORT_GAMMA: numParam('vortGamma', 0.7),
  };
  const renderPipe = await device.createRenderPipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: { module: renderModule, entryPoint: 'vs_main', constants: dims },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format }],
                constants: renderConsts },
    primitive: { topology: 'triangle-list' },
  });

  // --- M6: the tree sampler, reachable from the host -----------------------
  //
  // The renderer's other consumer of common_d3_tree_sample.wgsl, and the
  // only one whose output a check can score: same function, same bindings,
  // reached from a compute pass. Created always -- including on a dense run,
  // where SAMPLE_LEVELS is 0 and the answer must be "level 0 everywhere",
  // which is a claim worth being able to make.
  const PROBE_MAX = 4096;
  const probeInBuf = device.createBuffer({ size: PROBE_MAX * 16, usage: U.STORAGE | U.COPY_DST });
  const probeOutBuf = device.createBuffer({ size: PROBE_MAX * 32, usage: U.STORAGE | U.COPY_SRC });
  const probeStaging = device.createBuffer({ size: PROBE_MAX * 32, usage: U.MAP_READ | U.COPY_DST });
  const probeBGL = device.createBindGroupLayout({ entries: [
    ...renderBGL_entries.map(e => ({ ...e, visibility: GPUShaderStage.COMPUTE })),
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const probePipe = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [probeBGL] }),
    compute: {
      module: device.createShaderModule({
        code: await loadShader(`shaders/d3_tree_probe_q${Q}.wgsl`), label: `d3_tree_probe_q${Q}` }),
      entryPoint: 'main',
      constants: { ...dims, RB, SAMPLE_LEVELS },
    },
  });
  const probeBG = device.createBindGroup({ layout: probeBGL, entries: [
    ...treeSampleEntries(),
    { binding: 10, resource: { buffer: probeInBuf } },
    { binding: 11, resource: { buffer: probeOutBuf } }]});

  // Points are in L0 CELL UNITS with cell centres at integers -- the frame
  // d3-amr.mjs's cellAtLevel documents and tests. Returns, per point, the
  // level the sampler chose, that level's cell size, and the value.
  async function debugSampleTree(points) {
    if (points.length > PROBE_MAX) throw new Error(`debugSampleTree: ${points.length} points exceeds ${PROBE_MAX}`);
    const src = new Float32Array(points.length * 4);
    points.forEach((p, i) => { src[4 * i] = p[0]; src[4 * i + 1] = p[1]; src[4 * i + 2] = p[2]; });
    device.queue.writeBuffer(probeInBuf, 0, src);
    const enc = device.createCommandEncoder();
    const pp = enc.beginComputePass();
    pp.setPipeline(probePipe); pp.setBindGroup(0, probeBG);
    pp.dispatchWorkgroups(Math.ceil(points.length / 64)); pp.end();
    const bytes = points.length * 32;
    enc.copyBufferToBuffer(probeOutBuf, 0, probeStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await probeStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const v = new Float32Array(probeStaging.getMappedRange(0, bytes)).slice();
    probeStaging.unmap();
    return points.map((_, i) => ({
      level: v[8 * i + 4], h: v[8 * i + 5],
      rho: v[8 * i], u: [v[8 * i + 1], v[8 * i + 2], v[8 * i + 3]],
    }));
  }

  // --- state ---------------------------------------------------------------
  let step = 0;
  let useB = false;               // true => the CURRENT field is in fB
  // HAND_DEPTH runs are diagnostics, not solves: nothing schedules the
  // hierarchy yet (M5.3), so a live animation would present a hand-flattened
  // stanza as if it were the solver. Driven only by debugStepSync, from the
  // tools.
  let live = urlParams.get('live') !== '0';
  // HARD FAILURE STATE (M5.4a). Set once the manager has been refused a slot;
  // never cleared except by reset(). Refinement here is geometry-forced, so
  // a refusal means a coarse/fine seam is about to pass through the body --
  // the one configuration the solid coupling is not built for. 2D experience
  // is that such a run does not drift, it diverges.
  let poolExhausted = 0;
  let axis = parseAxis();
  const axisExtent = (a) => [NX, NY, NZ][a];
  // THE IN-PLANE AXES FOR A SLICE NORMAL, cyclically -- the single host-side
  // statement of planeDims()/fs_main's convention in
  // common_d3_render_slice.wgsl: the first runs across the screen and the
  // second down it (before any rotation). planeExtent() below indexes NX/NY/NZ
  // with exactly this.
  const PLANE_AXES = [[1, 2], [2, 0], [0, 1]];
  // QUARTER TURNS THAT PUT THE SCENARIO'S `down` DOWN THE WINDOW.
  //
  // `rot = 0` already draws the SECOND in-plane axis downward (fs_main's
  // `1 - uv.y`), so a scenario falling along that axis needs no turn at all;
  // one falling along the FIRST needs the picture turned a quarter clockwise.
  // A `down` that is the slice NORMAL is not in the picture and no rotation
  // can help, so it stays 0 rather than picking something arbitrary.
  //
  // Signed, because the answer for -x is not the answer for +x -- it is the
  // opposite turn, and a scenario that ever falls the other way should not
  // have to discover that by looking.
  function downTurn(a) {
    const d = params.down;
    if (!d) return 0;
    const [p, q] = PLANE_AXES[a];
    if (d[p] > 0) return 1;
    if (d[p] < 0) return 3;
    if (d[q] > 0) return 0;
    if (d[q] < 0) return 2;
    return 0;                          // `down` is the slice normal
  }
  // `?rotate=` overrides in DEGREES CLOCKWISE, which is how a person thinks
  // about turning a picture; the shader counts quarter turns.
  const ROT_OVERRIDE = urlParams.has('rotate') ? parseRotate() : null;
  const viewTurn = () => ROT_OVERRIDE !== null ? ROT_OVERRIDE : downTurn(axis);
  let slice = urlParams.has('slice')
    ? Math.min(axisExtent(parseAxis()) - 1, Math.max(0, parseInt(urlParams.get('slice'))))
    : (axisExtent(parseAxis()) >> 1);
  let mode = parseMode(scenarioName);
  const STEPS_PER_FRAME = Math.max(1, Math.round(numParam('spf', Math.max(1, Math.round(262144 / NCELLS * 8)))));
  const uRefDefault = params.scenario === 'duct' ? params.uPeak
    : params.scenario === 'sphere' ? params.u0 * 1.6      // the flow accelerates around the body
    // `fall` has no freestream at all -- the fluid starts at rest and the
    // only velocity scale is the body's own terminal one.
    : (params.scenario === 'fall' || params.scenario === 'card') ? params.u_t * 1.6
    : 2 * params.u0;
  // Render normalizations, derived per scenario rather than inherited from
  // the 2D pages' constants. common_vortcolor.wgsl's tone curve is
  // calibrated for a shedding wake at |omega| ~ 0.05; every field here is
  // an order of magnitude gentler (a decaying Beltrami flow has
  // |omega| = k|u| ~ 0.008), so the page divides by a reference and hands
  // the shader a dimensionless ratio, with VORT_SCALE pinned to 1.
  //
  //   duct      u peaks at its target; shear runs across the half-width a.
  //   beltrami  |u| reaches ~2.4*u0 (three components of sin+cos), and the
  //             field is Beltrami so |omega| = k|u| exactly.
  //   tgv       same wavenumber, amplitude u0 per component.
  const U_SCALE = numParam('uscale', uRefDefault);
  const V_SCALE = numParam('vscale',
    scenarioName === 'duct' ? 2 * params.uPeak / params.a
      : scenarioName === 'sphere' ? 2 * params.u0 / params.R
      : (scenarioName === 'fall' || scenarioName === 'card') ? 2 * params.u_t / params.R
        : (params.k || 2 * Math.PI / N) * U_SCALE);

  function writeRenderParams() {
    const b = new ArrayBuffer(32);
    new Uint32Array(b, 0, 4).set([axis, slice, mode, viewTurn()]);
    new Float32Array(b, 16, 4).set([U_SCALE, V_SCALE, 0, 0]);
    device.queue.writeBuffer(rpBuf, 0, b);
  }

  // --- M6.1-M6.4: the volume stack, the scalar field, the raymarcher -------
  //
  // ONE DENSE VOLUME PER LEVEL, EACH OVER ONLY THAT LEVEL'S BOUNDING BOX.
  // Every default here comes from the ballpark in plans/3D.md M6, measured
  // before any of it was written:
  //
  //   ?vol=N       a MULTIPLE OF EACH LEVEL'S OWN RESOLUTION. 1 means every
  //                level's volume matches its own grid exactly, so nothing
  //                is interpolated and nothing is thrown away; 2
  //                supersamples. Level m's grid already has 2^m cells per L0
  //                cell, so this is NOT "voxels per L0 cell" -- that phrasing
  //                makes one number mean three things in a three-level run.
  //                Defaults to 1 under ?view=volume and to 0 otherwise --
  //                the slice view samples the tree directly and owes this
  //                pass nothing, so filling volumes for it would be pure
  //                cost.
  //   ?volstack=0  one volume only (the L0 box), which is M6.3's
  //                single-resolution render. THE CONTROL for M6.4b: the
  //                stacked render must agree with it everywhere only L0
  //                exists and differ where a refined box covers.
  //   ?volbox=     `domain` (default) or `refined` -- the M6.1 single-volume
  //                selector, kept because `body3-vol-refined` is the case
  //                the ballpark exists for. It implies ?volstack=0.
  //   ?volBudget=  MiB, default 256, over the WHOLE stack. Exceeding it
  //                REFUSES rather than silently shrinking, for M5.0's
  //                reason: a view that quietly renders a coarser volume than
  //                asked for is a plausible picture of a grid nobody chose.
  //   ?volMargin=  blocks of slack around a refined box, default 2 on a
  //                dynamic run and 0 on a static one, so a box that FOLLOWS
  //                a body has somewhere to move to without the texture being
  //                reallocated.
  //
  // THE COST, and why the stack is the cheap answer rather than the clever
  // one: 16 B/voxel here (velocity rgba16float + the scalar rgba16float),
  // against the flagship case's 2.25 GB for ONE uniform volume at the finest
  // resolution. The three boxes are 24^3, 32^3 and the full domain -- 98 MB
  // together, because the refined region is 0.44% of the domain. That ratio
  // IS what AMR exists to exploit; materializing the whole domain at the
  // finest level is the dense run AMR replaced.
  const VOL_BUDGET_MB = Math.max(1, numParam('volBudget', 256));
  const volBoxMode = urlParams.get('volbox') || 'domain';
  if (!['domain', 'refined'].includes(volBoxMode)) {
    statusEl.textContent = `error: ?volbox=${volBoxMode} is not one of domain|refined`;
    return;
  }
  const VOL_MULT = Math.max(0, numParam('vol', VIEW === 'volume' ? 1 : 0));
  // A single volume over the finest level's box and a per-level stack are two
  // different box policies, and silently preferring one would make a URL that
  // asks for both mean something nobody wrote. Refused instead.
  if (volBoxMode === 'refined' && numParam('volstack', 0) > 0) {
    statusEl.textContent = 'error: ?volbox=refined is a single volume over the finest level\'s box;'
      + ' ?volstack=1 is one volume per level. Pick one.';
    return;
  }
  // ?volbox=refined names a LEVEL's bounding box, and a dense run has no
  // levels. Refused rather than quietly given the domain volume instead --
  // this file's own header names a silently-dropped parameter as worse than
  // a rejected one, and a harness that believed it was rendering the refined
  // box would be reporting on a picture nobody asked for.
  if (volBoxMode === 'refined' && !AMR) {
    statusEl.textContent = 'error: ?volbox=refined needs a refined level -- this run is ?levels=1';
    return;
  }
  const VOL_STACK = AMR && volBoxMode === 'domain' && numParam('volstack', 1) > 0;
  const VOL_MARGIN = Math.max(0, Math.round(numParam('volMargin', DYNAMIC ? 2 : 0)));
  // M6.4d, and ON by default: the gradient pass differences each voxel over
  // the source cell its data actually came from rather than over one voxel.
  // `?volh=0` restores the old fixed stride so the two can be rendered
  // side by side -- the control 96547af did not have.
  const VOL_HSRC = numParam('volh', 1) > 0 ? 1 : 0;
  // M6.4f. A refined box is used only where its voxel came from its OWN level;
  // elsewhere the stack falls back outward instead of overriding the coarser
  // level with that level's own data replicated onto a finer grid. `?volfallback=0`
  // restores innermost-box-wins-unconditionally for the A/B.
  const VOL_FALLBACK = numParam('volfallback', 1) > 0 ? 1 : 0;

  // THE SMALLEST PERIODIC INTERVAL covering a set of block indices on one
  // axis -- the largest circular GAP removed, rather than min..max.
  //
  // Not a nicety: with a moving window the body travels through a periodic
  // buffer and its refined shell travels with it, so twice a lap the shell
  // STRADDLES THE SEAM. Taken as min..max its bounding box is then the whole
  // axis, which at level 2 of the flagship case is a 1.5 GB texture request
  // instead of a 10 MB one -- i.e. the view refuses to start, periodically,
  // for a reason that looks like nothing.
  function axisSpan(used, n) {
    const idx = [];
    for (let i = 0; i < n; i++) if (used[i]) idx.push(i);
    if (!idx.length) return null;
    let gap = n - idx[idx.length - 1] + idx[0] - 1, at = idx[idx.length - 1];
    for (let i = 1; i < idx.length; i++) {
      const g = idx[i] - idx[i - 1] - 1;
      if (g > gap) { gap = g; at = idx[i - 1]; }
    }
    return { lo: (at + gap + 1) % n, len: n - gap };
  }

  // A level's refined set as a box in L0 CELL units, from a blockSlot map.
  // A level-m block spans RB * 2^-(m-1) L0 cells -- level 1's blocks are RB
  // cells wide because its parent IS L0, and each rung halves that. Checked
  // against the measured flagship case: level 2's box is 12 blocks per axis
  // at 2 L0 cells = 24.
  function boxFromBlockSlot(m, bs, marginBlocks = 0) {
    const lp = hier.byLevel[m].pool, nb = lp.nb, w = RB / 2 ** (m - 1);
    const used = nb.map((n) => new Uint8Array(n));
    let any = false;
    for (let id = 0; id < lp.nBlocks; id++) {
      if (bs[id] < 0) continue;
      any = true;
      const b = lp.blockOf(id);
      for (let k = 0; k < 3; k++) used[k][b[k]] = 1;
    }
    if (!any) return null;
    const lo = [0, 0, 0], hi = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const s = axisSpan(used[k], nb[k]);
      const len = Math.min(nb[k], s.len + 2 * marginBlocks);
      lo[k] = (s.lo - marginBlocks) * w;
      hi[k] = lo[k] + len * w;
    }
    return { lo, hi };
  }
  const refinedBoxL0 = (m, margin = VOL_MARGIN) =>
    boxFromBlockSlot(m, hier.byLevel[m].blockSlot, margin);

  // FIXED IN NUMBER, like every other per-level binding here, because WebGPU
  // has no array of differently-sized textures. A run deeper than this is
  // already past anything validated -- and past the tree sampler's own
  // MAX_SAMPLE_LEVELS, so a deeper volume would be filled from a capped walk
  // anyway. Clamped rather than refused, and SAID OUT LOUD, for the reason
  // viewDepthNote exists: a viewer that silently stops one level short of
  // the solver renders a plausible picture of a hierarchy nobody is solving.
  const MAX_VOL_LEVELS = 4;

  // The stack's DESCRIPTORS, from d3-volume.mjs -- the same module the
  // validation tool uses to predict which pixels a refined box can have
  // changed, which is what makes M6.4b's image difference a check rather
  // than the accused testifying.
  let volStack = [], volClamped = false;
  if (VOL_MULT > 0) {
    if (volBoxMode === 'refined') {
      const b = refinedBoxL0(LEVELS - 1);
      volStack = b ? [volLevelVolume({ lo: b.lo, ext: b.hi.map((h, k) => h - b.lo[k]),
                                       level: LEVELS - 1, mult: VOL_MULT, bytesPerVoxel: 16 })] : [];
    } else {
      const boxes = [null];
      if (VOL_STACK) for (let m = 1; m < LEVELS; m++) boxes[m] = refinedBoxL0(m);
      volStack = volumeStack({ dims: [NX, NY, NZ], levelBoxes: boxes, mult: VOL_MULT,
                               levels: VOL_STACK ? LEVELS : 1, bytesPerVoxel: 16 });
    }
    if (!volStack.length) {
      statusEl.textContent = `error: ?volbox=refined found no refined tiles at L${LEVELS - 1}`;
      return;
    }
    if (volStack.length > MAX_VOL_LEVELS) { volStack = volStack.slice(0, MAX_VOL_LEVELS); volClamped = true; }
    // M6.4d's companion is 4 B/voxel on the REFINED levels only (see the
    // lvlTex creation below for why L0 is a dummy). Counted here rather than
    // left out of the budget: a refusal is the whole point of the budget, and
    // an allocation the budget cannot see is one that OOMs instead.
    const lvlBytes = (VOL_HSRC || VOL_FALLBACK)
      ? volStack.reduce((a, v) => a + (v.level > 0 ? v.voxels * 4 : 0), 0) : 0;
    const bytes = volStack.reduce((a, v) => a + v.bytes, 0) + lvlBytes;
    if (bytes > VOL_BUDGET_MB * 1048576) {
      statusEl.textContent = `error: ?vol=${VOL_MULT} needs `
        + volStack.map(v => `L${v.level} ${v.res.join('x')}`).join(' + ')
        + ` = ${(bytes / 1048576).toFixed(0)} MiB, past ?volBudget=${VOL_BUDGET_MB}.`
        + ' Lower ?vol=, use ?volbox=refined, or raise the budget deliberately';
      return;
    }
    const lim = device.limits.maxTextureDimension3D;
    const big = volStack.find(v => Math.max(...v.res) > lim);
    if (big) {
      statusEl.textContent = `error: ?vol=${VOL_MULT} needs an L${big.level} volume of`
        + ` ${big.res.join('x')}, past this GPU's maxTextureDimension3D of ${lim}`;
      return;
    }
  }

  // Per-entry GPU state. Parallel to volStack, and rebuilt never: the shape
  // is fixed at creation (a texture has a size) and only the ORIGIN moves --
  // see common_d3_resample.wgsl's header for why that split is the whole of
  // the dynamic story.
  const volGPU = [];
  let volBoxTight = false;
  const encodeResample = (enc) => {
    if (!volGPU.length) return;
    const vp = enc.beginComputePass();
    for (const g of volGPU) {
      vp.setPipeline(g.pipe); vp.setBindGroup(0, g.bg);
      vp.dispatchWorkgroups(g.disp[0], g.disp[1], g.disp[2]);
    }
    vp.end();
  };
  // M6.2. The gradient pass, once per voxel, AFTER every velocity volume is
  // filled -- its own pass because it reads the texture the resample writes,
  // and a read of a storage texture another invocation in the same pass is
  // writing is a race, not a dependency.
  const encodeScalar = (enc) => {
    if (!volGPU.length || !volGPU[0].sclPipe) return;
    const sp = enc.beginComputePass();
    for (const g of volGPU) {
      sp.setPipeline(g.sclPipe); sp.setBindGroup(0, g.sclBG);
      sp.dispatchWorkgroups(g.disp[0], g.disp[1], g.disp[2]);
    }
    sp.end();
  };

  if (volStack.length) {
    const volBGL = device.createBindGroupLayout({ entries: [
      ...renderBGL_entries.map(e => ({ ...e, visibility: GPUShaderStage.COMPUTE })),
      { binding: 10, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // M6.4d's source-level companion. 12 rather than the next free number
      // by accident: renderBGL_entries owns 0..9 and this layout already
      // appends 10 and 11, so 12 is the first one nothing else in the
      // sampler's world speaks for.
      { binding: 12, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8uint', viewDimension: '3d' } },
    ]});
    const sclBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint', viewDimension: '3d' } },
    ]});
    const resampleCode = await loadShader(`shaders/d3_resample_q${Q}.wgsl`);
    const scalarCode = await loadShader('shaders/d3_volume_scalar.wgsl');
    const resampleModule = device.createShaderModule({ code: resampleCode, label: `d3_resample_q${Q}` });
    const scalarModule = device.createShaderModule({ code: scalarCode, label: 'd3_volume_scalar' });
    for (const v of volStack) {
      const mk = (label) => device.createTexture({
        size: v.res, dimension: '3d', format: 'rgba16float',
        // COPY_SRC so debugReadVolume can score them against the sampler they
        // were filled from; TEXTURE_BINDING because the gradient pass reads
        // the velocity one and the raymarcher reads the scalar one.
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        label });
      const velTex = mk(`d3_vol_vel_L${v.level}`);
      const sclTex = mk(`d3_vol_scl_L${v.level}`);
      // M6.4d. FULL SIZE ONLY WHERE IT IS READ. The L0 volume's own spacing
      // is already the coarsest in the tree, so its stride is 1 by
      // construction and it gets a 1x1x1 dummy -- the companion is paid for
      // by the small refined boxes and not by the 50 MB domain volume. A
      // binding cannot be optional; the WRITE and the READ are, through
      // WRITE_LEVEL and VOL_HSRC, and both fold at pipeline creation.
      const wantLvl = (VOL_HSRC || VOL_FALLBACK) && v.level > 0;
      const lvlTex = device.createTexture({
        size: wantLvl ? v.res : [1, 1, 1], dimension: '3d', format: 'rgba8uint',
        // COPY_SRC for the same reason velTex has it: the gate scores the
        // companion against debugSampleTree, and a buffer no check can read is
        // one two features now rest on and nothing verifies.
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
             | GPUTextureUsage.COPY_SRC,
        label: `d3_vol_lvl_L${v.level}` });
      const originBuf = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
      device.queue.writeBuffer(originBuf, 0, new Float32Array([v.lo[0], v.lo[1], v.lo[2], 0]));
      const pipe = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [volBGL] }),
        compute: { module: resampleModule, entryPoint: 'main',
          constants: { ...dims, RB, SAMPLE_LEVELS,
            VOL_NX: v.res[0], VOL_NY: v.res[1], VOL_NZ: v.res[2],
            VOL_HX: v.h[0], VOL_HY: v.h[1], VOL_HZ: v.h[2],
            WRITE_LEVEL: wantLvl ? 1 : 0 } },
      });
      const sclPipe = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [sclBGL] }),
        compute: { module: scalarModule, entryPoint: 'main',
          constants: {
            VOL_NX: v.res[0], VOL_NY: v.res[1], VOL_NZ: v.res[2],
            VOL_HX: v.h[0], VOL_HY: v.h[1], VOL_HZ: v.h[2],
            // Only the level-0 box over the WHOLE domain has a neighbour on
            // the other side of its own face.
            VOL_WRAP: (v.level === 0 && v.ext.every((e, k) => e === [NX, NY, NZ][k])) ? 1 : 0,
            // FROM VOL_HSRC, NOT FROM `wantLvl`. They were the same thing
            // until M6.4f gave the companion a second consumer -- and then
            // `?volh=0` still got the source-cell stencil, because the
            // companion was allocated for the RAY's sake. The A/B silently
            // became a no-op and the volh gate caught it.
            VOL_HSRC: (VOL_HSRC && v.level > 0) ? 1 : 0, VOL_LEVEL: v.level,
            Q_REF: Q_REF_SCALE, OM_REF: V_SCALE, U_REF: U_SCALE } },
      });
      volGPU.push({
        level: v.level, velTex, sclTex, lvlTex, hasLvl: !!wantLvl, originBuf, pipe, sclPipe,
        disp: v.res.map(r => Math.ceil(r / 4)),
        bg: device.createBindGroup({ layout: volBGL, entries: [
          ...treeSampleEntries(),
          { binding: 10, resource: velTex.createView() },
          { binding: 11, resource: { buffer: originBuf } },
          { binding: 12, resource: lvlTex.createView() }]}),
        sclBG: device.createBindGroup({ layout: sclBGL, entries: [
          { binding: 0, resource: velTex.createView() },
          { binding: 1, resource: sclTex.createView() },
          { binding: 2, resource: lvlTex.createView() }]}),
      });
    }
  }
  // Back-compat: the M6.1 gate and getParams both speak of "the volume",
  // which is now the stack's base. Kept as a name rather than resolved at
  // every use, so a tool written against M6.1 reads exactly what it read.
  const volInfo = volStack.length ? volStack[0] : null;

  // THE BOXES FOLLOW THE BODY (M6.4a, dynamic). Read on the status cadence,
  // not per frame: a blockSlot readback is a round trip, the manager runs
  // every ?manageEvery= steps anyway, and the box moves by whole blocks.
  //
  // A box that has OUTGROWN its allocated extent is said out loud rather than
  // clipped in silence -- the alternative is a view that shows part of a
  // refined region and looks exactly like one that shows all of it.
  //
  // SINGLE-FLIGHT, because the frame loop fires this without awaiting it. The
  // staging buffer underneath is serialized (see `serializedOn`), so an
  // overlapping call is no longer an error -- but it would QUEUE, and a
  // refresh that takes longer than the 250 ms cadence would then accumulate
  // one pending refresh per tick forever. Dropping is right here where
  // chaining is right there: these are polls of a current value, so a
  // skipped one is superseded by the next tick rather than lost.
  //
  // AND THE APPLY IS ATOMIC (M6.4e, 2026-09-13, `?volsync=0` to A/B). Two
  // buffers say where a refined box is and BOTH have to move together: the
  // resample's own `originBuf`, which decides what gets written INTO the
  // texture, and the ray params' `boxLo`, which decides where the raymarcher
  // reads it OUT of world space. Publish one without the other and the box's
  // whole contents are displaced by the difference.
  //
  // The old shape read and applied one level at a time, so the L1 origin was
  // written and then `await readBlockSlot(2)` yielded to the event loop --
  // and every frame that rendered during that readback drew L1's freshly
  // resampled contents through L1's STALE placement. The picture shifts en
  // masse by whole cells and snaps back when the refresh finishes.
  //
  // A DEPTH BUG, the same family as `serializedOn`'s: at `?levels=2` the loop
  // runs once and the only await is BEFORE the only write, so there is no gap
  // to observe. It needs a second refined level to exist at all.
  //
  // NOT NEAR THE BODY, WHICH IS WHY IT READS AS A FAR-FIELD FAULT: the
  // innermost box is applied LAST, with nothing but synchronous code between
  // its write and `writeRayParams`, so the finest level -- the one covering
  // the plate -- is always consistent. It is L1, which at ?volMargin=4 spans
  // 80x72x112 of a 192x160x192 domain, that goes out of register.
  //
  // The fix is to do every readback FIRST and every write after, with no
  // await between the first write and the publish. That also removes a
  // second inconsistency the per-level version had: innermost-box-wins means
  // a correctly-placed L2 could be composited over a stale L1.
  const VOL_SYNC = numParam('volsync', 1) > 0 ? 1 : 0;
  // Frames rendered, for the gate below. A counter and not a timestamp
  // because the question is "could anything have been DRAWN in the gap",
  // which a duration only implies.
  let volFrameSeq = 0;
  const volSyncStat = { updates: 0, partialFrames: 0, maxGapMs: 0 };
  let volBoxInFlight = null;
  function refreshVolumeBoxes() {
    if (volBoxInFlight) return volBoxInFlight;
    volBoxInFlight = refreshVolumeBoxesNow().finally(() => { volBoxInFlight = null; });
    return volBoxInFlight;
  }
  async function refreshVolumeBoxesNow() {
    if (!DYNAMIC || volStack.length < 2) return;
    volBoxTight = false;
    // EVERY READBACK FIRST. This is the only part that awaits, and hoisting it
    // out of the apply is the whole of the fix -- see the header above.
    // `?volsync=0` leaves the reads interleaved, which is the old behaviour
    // and the leg the gate measures against.
    const boxes = [];
    if (VOL_SYNC) {
      for (let i = 1; i < volStack.length; i++) {
        boxes[i] = boxFromBlockSlot(volStack[i].level, await readBlockSlot(volStack[i].level), 0);
      }
    }
    // ---- NO `await` BELOW THIS LINE WHEN VOL_SYNC IS ON ----
    let firstWrite = -1, tFirst = 0;
    for (let i = 1; i < volStack.length; i++) {
      const v = volStack[i];
      const box = VOL_SYNC ? boxes[i]
        : boxFromBlockSlot(v.level, await readBlockSlot(v.level), 0);
      if (!box) continue;
      const lo = v.lo.slice();
      for (let k = 0; k < 3; k++) {
        const len = box.hi[k] - box.lo[k];
        if (len > v.ext[k]) volBoxTight = true;
        // Centred on the set, and moved by WHOLE L0 cells so the volume's
        // voxel grid stays aligned with the level's own cell grid -- which is
        // what keeps the resample a copy rather than a resample.
        lo[k] = Math.round(box.lo[k] + (len - v.ext[k]) / 2);
      }
      if (lo.every((c, k) => c === v.lo[k])) continue;
      v.lo = lo;
      v.c0 = lo.map(c => c - 0.5);
      v.c1 = lo.map((c, k) => c + v.ext[k] - 0.5);
      device.queue.writeBuffer(volGPU[i].originBuf, 0, new Float32Array([lo[0], lo[1], lo[2], 0]));
      if (firstWrite < 0) { firstWrite = volFrameSeq; tFirst = performance.now(); }
    }
    writeRayParams();
    // THE GAP, MEASURED: frames drawn between the first origin write and the
    // publish that makes it meaningful. Zero BY CONSTRUCTION when VOL_SYNC is
    // on, which is exactly why it is worth counting -- a claim of "no await
    // here" is one refactor away from being false, and this notices.
    if (firstWrite >= 0) {
      volSyncStat.updates++;
      volSyncStat.partialFrames += volFrameSeq - firstWrite;
      volSyncStat.maxGapMs = Math.max(volSyncStat.maxGapMs, performance.now() - tFirst);
    }
  }

  // --- M6.3/M6.4b: the raymarcher ------------------------------------------
  //
  // Camera knobs, all four of them, and the defaults are the only interesting
  // part. `dist` is in units of the domain's largest extent so one default
  // frames a 64^3 box and the 192x128x128 flagship alike; `iso` and `gain`
  // are DIMENSIONLESS because the scalar volume is normalized
  // (d3_volume_scalar.wgsl), and the Q default is the MEASURED threshold from
  // d3-criterion.mjs rather than a number chosen by looking.
  const VOL_FIELDS = ['q', 'omega', 'speed', 'rho'];
  const volFieldName = (urlParams.get('volfield') || 'q').toLowerCase();
  if (!VOL_FIELDS.includes(volFieldName)) {
    statusEl.textContent = `error: ?volfield=${volFieldName} is not one of ${VOL_FIELDS.join('|')}`;
    return;
  }
  let volField = VOL_FIELDS.indexOf(volFieldName);
  const VOL_ISO_DEF = [Q_THRESH, 0.15, 0.0, 0.0];
  // The KNEE of the tone curve, per field: the value of (field - iso) at
  // which the curve is half way up. Not a clip -- nothing saturates -- so
  // this sets where the detail sits rather than what is thrown away.
  const VOL_GAIN_DEF = [4.0, 0.85, 1.0, 0.02];
  let camAzim = numParam('azim', 35) * Math.PI / 180;
  let camElev = numParam('elev', 18) * Math.PI / 180;
  let camDist = numParam('dist', 1.9) * Math.max(NX, NY, NZ);
  const CAM_FOV = numParam('fov', 38) * Math.PI / 180;
  // PERSPECTIVE OR ORTHOGRAPHIC. The default reads as a picture; the other is
  // an INSTRUMENT -- parallel rays and a known cells-per-pixel scale, so a
  // plane in the flow lands at a computable row and can be measured in cells
  // rather than estimated through foreshortening. The half-height is chosen
  // so the two FRAME THE SAME at the target plane, which is what makes
  // switching between them a comparison rather than a different experiment.
  const camProj = urlParams.get('proj') || 'persp';
  if (!['persp', 'ortho'].includes(camProj)) {
    statusEl.textContent = `error: ?proj=${camProj} is not one of persp|ortho`;
    return;
  }
  const CAM_ORTHO = camProj === 'ortho' ? 1 : 0;
  // The camera's UP is the scenario's own DOWN, negated -- the same field the
  // slice view's quarter turn reads. The solver has no opinion about down and
  // must not acquire one; this is the view's business.
  const CAM_DOWN = params.down || [0, 0, -1];
  const CAM_UP_AXIS = CAM_DOWN.findIndex(v => Math.abs(v) > 0.5);
  const CAM_UP_SIGN = CAM_DOWN[CAM_UP_AXIS] > 0 ? -1 : 1;
  // FOLLOW THE BODY, computed in the SHADER from the body buffer. A
  // host-written target is as stale as the last readback and the picture then
  // slides forward and snaps back every time one lands -- the same reason the
  // slice view reads the body directly (M8.3).
  const camTargetMode = urlParams.get('camtarget') || (HAS_BODY ? 'body' : 'domain');
  if (!['body', 'domain'].includes(camTargetMode)) {
    statusEl.textContent = `error: ?camtarget=${camTargetMode} is not one of body|domain`;
    return;
  }
  const CAM_FOLLOW = camTargetMode === 'body' && HAS_BODY ? 1 : 0;
  const CAM_TARGET = [(NX - 1) / 2, (NY - 1) / 2, (NZ - 1) / 2];
  // EXTINCTION PER L0 CELL at the top of the tone curve. The curve is
  // asymptotic to 1 (see d3_raymarch.wgsl's volTone), so this is a real
  // maximum rather than a scale that a strong core runs away with: at 0.6 a
  // core accumulates 1 - exp(-0.6) = 45% of the remaining light per cell and
  // a structure a fifth as strong accumulates 10%, which is the dynamic
  // range that lets a wake be visible beside a boundary layer.
  const VOL_OPACITY = numParam('volOpacity', 0.6);
  const VOL_STEP = numParam('volStep', 0.75);
  const VOL_STEPS = Math.max(16, Math.round(numParam('volSteps', 2048)));
  let volIso = numParam('volIso', VOL_ISO_DEF[volField]);
  let volGain = numParam('volGain', VOL_GAIN_DEF[volField]);

  // Only when the CLAMP bit -- a level with no tiles is dropped legitimately
  // (the coarser volume already holds the right answer there, because the
  // resample writes sampleTree, which falls back), and reporting that as a
  // truncation would cry wolf on the ordinary case.
  const volDepthNote = volClamped
    ? `   [volume stack stops at L${volStack[volStack.length - 1].level} of L${LEVELS - 1}]` : '';
  let rayPipe = null, rayBG = null, rayBuf = null;
  // THE OFFSCREEN TARGET AND ITS TWO PIPELINES, shared by M6.4b's image
  // difference and by tools/render-d3-movie.js. One texture, resized when a
  // caller asks for a different frame size, and one pipeline per view in the
  // offscreen format -- see debugRenderFrame below for why neither of them
  // goes through the canvas.
  const offscreen = { tex: null, w: 0, h: 0, volume: null, slice: null };
  // WHICH VIEW IS DRAWING. `?view=` chooses the initial one; the <select>
  // switches at runtime, and it can only offer `volume` where the volumes
  // exist -- which is `?vol=` and not `?view=`, since the volumes are the
  // memory and the view is free. So the raymarcher is built whenever there
  // is something for it to march, including under the slice view, where it
  // costs one pipeline and makes debugRenderFrame reachable from the M6.1
  // gate configs as well.
  let viewMode = VIEW;
  if (VIEW === 'volume' && !volStack.length) {
    statusEl.textContent = 'error: ?view=volume needs a volume -- ?vol=0 built none';
    return;
  }
  if (volStack.length) {
    rayBuf = device.createBuffer({ size: 192, usage: U.UNIFORM | U.COPY_DST });
    // 1x1x1, sampled by the levels this run does not have. rm.ctl.z folds
    // them out of the walk, so it is never read -- but a bind group with a
    // null texture is not a thing, exactly as the tree sampler's 4-byte
    // dummy buffers are not.
    const dummyVol = device.createTexture({
      size: [1, 1, 1], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING, label: 'd3_vol_dummy' });
    const dummyLvl = device.createTexture({
      size: [1, 1, 1], dimension: '3d', format: 'rgba8uint',
      usage: GPUTextureUsage.TEXTURE_BINDING, label: 'd3_lvl_dummy' });
    const rayBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ...Array.from({ length: MAX_VOL_LEVELS }, (_, i) => ({
        binding: 2 + i, visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '3d' } })),
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      // M6.4f's source-level companions, one per REFINED level: 7, 8, 9 are
      // stack entries 1, 2, 3. Entry 0 has none and needs none -- level 0's
      // data is never a coarser level's, because there is no coarser level.
      ...Array.from({ length: MAX_VOL_LEVELS - 1 }, (_, i) => ({
        binding: 7 + i, visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'uint', viewDimension: '3d' } })),
    ]});
    const rayModule = device.createShaderModule({
      code: await loadShader('shaders/d3_raymarch.wgsl'), label: 'd3_raymarch' });
    const rayConsts = { ...dims, ...WINC, HAS_BODY,
                        UP_AXIS: CAM_UP_AXIS, UP_SIGN: CAM_UP_SIGN,
                        VOL_GAMMA: numParam('volGamma', 0.7),
                        PROJ: CAM_ORTHO,
                        // ?winbox=0 marches the buffer's box instead of the
                        // window's. Default 1; measured worth nothing, kept
                        // because the window is the frame the picture is in.
                        WIN_BOX: urlParams.get('winbox') === '0' ? 0 : 1,
                        VOL_FALLBACK: VOL_FALLBACK };
    const mkRayPipe = (fmt) => device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [rayBGL] }),
      vertex: { module: rayModule, entryPoint: 'vs_main', constants: rayConsts },
      fragment: { module: rayModule, entryPoint: 'fs_main', targets: [{ format: fmt }], constants: rayConsts },
      primitive: { topology: 'triangle-list' },
    });
    rayPipe = await mkRayPipe(format);
    // The offscreen twin, in the format debugRenderFrame reads. Built here
    // rather than on first use because the volume view's whole reason to
    // exist is to be looked at, and both of the things that look at it (the
    // M6.4b gate and the movie tool) go through this pipeline.
    offscreen.volume = await mkRayPipe('rgba8unorm');
    rayBG = device.createBindGroup({ layout: rayBGL, entries: [
      { binding: 0, resource: { buffer: rayBuf } },
      { binding: 1, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
      ...Array.from({ length: MAX_VOL_LEVELS }, (_, i) => ({
        binding: 2 + i,
        resource: (volGPU[i] ? volGPU[i].sclTex : dummyVol).createView() })),
      { binding: 6, resource: { buffer: bodyBuf } },
      ...Array.from({ length: MAX_VOL_LEVELS - 1 }, (_, i) => ({
        binding: 7 + i,
        resource: (volGPU[i + 1] ? volGPU[i + 1].lvlTex : dummyLvl).createView() }))]});
  }

  // The uniform, rebuilt whenever anything in it moves -- a camera drag, a
  // field switch, a box following the body. 192 bytes, so there is nothing to
  // be clever about.
  function writeRayParams(aspectOverride) {
    if (!rayBuf) return;
    const aspect = aspectOverride || (canvas.width / Math.max(1, canvas.height));
    const b = new Float32Array(48);
    // cam.w is tan(fov/2) under perspective and the HALF-HEIGHT IN L0 CELLS
    // under orthographic -- the same number scaled by the distance, so the two
    // projections frame identically at the target plane.
    const halfTan = Math.tan(CAM_FOV / 2);
    b.set([camAzim, camElev, camDist, CAM_ORTHO ? camDist * halfTan : halfTan], 0);
    b.set([aspect, VOL_STEP, volIso, volGain], 4);
    b.set([volField, VOL_OPACITY, volStack.length, VOL_STEPS], 8);
    b.set([CAM_TARGET[0], CAM_TARGET[1], CAM_TARGET[2], CAM_FOLLOW], 12);
    for (let i = 0; i < MAX_VOL_LEVELS; i++) {
      const v = volStack[i];
      // w is the entry's LEVEL, not a present flag and not its index:
      // volumeStack drops a level with no tiles, so the stack can be (0, 2)
      // and M6.4f's "did this voxel come from MY level" needs the real one.
      // Presence is rm.ctl.z, which already folds absent entries out.
      b.set(v ? [v.c0[0], v.c0[1], v.c0[2], v.level] : [0, 0, 0, 0], 16 + 4 * i);
      // w is the voxel size the ray steps by inside this box -- the SMALLEST
      // of the three, so an anisotropic box does not step past a voxel on its
      // finest axis.
      b.set(v ? [v.ext[0], v.ext[1], v.ext[2], Math.min(...v.h)] : [1, 1, 1, 1], 32 + 4 * i);
    }
    device.queue.writeBuffer(rayBuf, 0, b);
  }

  // THE GATE (M6.1/M6.4a). Reads a volume back and hands it to the caller
  // alongside the voxel centres, so a tool can score it against
  // debugSampleTree at the SAME physical points -- the volume must agree with
  // the sampler it was filled from, which is the one claim that does not
  // depend on anything rendering.
  //
  // A z-SLAB rather than the whole volume: at the default resolution the
  // flagship case is 36 MB, which is a fine thing to hold on the GPU and a
  // silly thing to map into JS every time a check runs. `which` indexes the
  // STACK, so M6.4a's claim -- every level's volume agrees with the sampler
  // over its own box -- is the same check run once per level.
  async function debugReadVolume(z, which = 0, what = 'vel') {
    if (!volGPU.length) return { skipped: 'no volume (?vol=0)' };
    if (which >= volGPU.length) return { skipped: `no volume ${which} (stack of ${volGPU.length})` };
    // M6.4d/M6.4f both rest on the source-level companion being RIGHT, and
    // until it is readable nothing can say so. Refused rather than returned
    // empty where it does not exist: level 0 has no companion by design, and
    // an all-zero slab would read as "every voxel came from L0", which is the
    // very answer the check is trying to distinguish.
    if (what === 'lvl' && !volGPU[which].hasLvl) {
      return { skipped: `no source-level companion for stack entry ${which}`
        + ` (level ${volStack[which].level}${VOL_HSRC || VOL_FALLBACK ? '' : ', and ?volh=0&volfallback=0'})` };
    }
    // SAME REASON AS debugRenderFrame: the boxes are moved by the live frame
    // loop, and a tool reading the volume runs with `live=0`. Without this a
    // check reads a volume whose box the body left thousands of steps ago and
    // scores it against a sampler that is perfectly correct -- i.e. it
    // measures staleness and calls it disagreement.
    await refreshVolumeBoxes();
    const v = volStack[which], g = volGPU[which];
    const [nx, ny] = v.res;
    const zz = Math.min(v.res[2] - 1, Math.max(0, z | 0));
    // 256-byte row alignment is a copyTextureToBuffer requirement, not a
    // suggestion: an unpadded bytesPerRow is a validation error, and a
    // validation error here is silent.
    // 4 B/texel for the rgba8uint companion against 8 for the two rgba16float
    // volumes, so the row stride is not one constant.
    const bpt = what === 'lvl' ? 4 : 8;
    const rowBytes = Math.ceil(nx * bpt / 256) * 256;
    const buf = device.createBuffer({ size: rowBytes * ny, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    encodeResample(enc);
    if (what === 'scl') encodeScalar(enc);
    const srcTex = what === 'scl' ? g.sclTex : what === 'lvl' ? g.lvlTex : g.velTex;
    enc.copyTextureToBuffer({ texture: srcTex, origin: { x: 0, y: 0, z: zz } },
      { buffer: buf, bytesPerRow: rowBytes, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: 1 });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const mapped = buf.getMappedRange().slice(0);
    const raw = what === 'lvl' ? new Uint8Array(mapped) : new Uint16Array(mapped);
    buf.unmap(); buf.destroy();
    const out = [];
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const o = what === 'lvl' ? y * rowBytes + x * 4 : (y * rowBytes) / 2 + x * 4;
        const c = what === 'lvl' ? [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]
          : [half(raw[o]), half(raw[o + 1]), half(raw[o + 2]), half(raw[o + 3])];
        out.push({
          p: [v.lo[0] + (x + 0.5) * v.h[0] - 0.5,
              v.lo[1] + (y + 0.5) * v.h[1] - 0.5,
              v.lo[2] + (zz + 0.5) * v.h[2] - 0.5],
          // Named for what the two textures HOLD, not for their layout: the
          // velocity volume is (u, rho) and the scalar one is four separately
          // normalized fields. A caller that read `u` out of the scalar
          // volume would get Q in x and be none the wiser.
          ...(what === 'scl' ? { v: c }
            : what === 'lvl' ? { src: c[0] }
            : { u: c.slice(0, 3), rho: c[3] }),
        });
      }
    }
    return { info: { ...v, stack: volStack.length, which }, z: zz, texels: out };
  }
  // IEEE 754 binary16 -> Number. Exact, including subnormals and the zero
  // the volume is full of before anything has run -- rounding this would
  // turn the gate's tolerance into a property of the decoder.
  function half(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  }

  // The transfer function, from a tool. The <select> and the URL cover a
  // person; a CHECK needs to move the iso and see the picture respond,
  // because "lit pixels > 0" on its own cannot tell a normalized transfer
  // function from a threshold that happens to sit under the noise floor.
  // Returns what it set, so a caller reads the resolved values rather than
  // assuming its request was honoured.
  function debugSetVolume(opts = {}) {
    if (!rayBuf) return { skipped: 'no raymarcher (?vol=0)' };
    if (opts.field != null) {
      const f = typeof opts.field === 'string' ? VOL_FIELDS.indexOf(opts.field) : opts.field;
      if (f < 0 || f >= VOL_FIELDS.length) return { error: `no such field: ${opts.field}` };
      volField = f;
      volIso = VOL_ISO_DEF[f];
      volGain = VOL_GAIN_DEF[f];
    }
    if (opts.iso != null) volIso = opts.iso;
    if (opts.gain != null) volGain = opts.gain;
    volFieldSel.value = String(volField);
    writeRayParams();
    return { field: VOL_FIELDS[volField], iso: volIso, gain: volGain };
  }

  // The camera, from a tool. Same argument as debugSetVolume above: the drag
  // and the URL cover a person, but a MOVIE has to turn the camera between
  // frames, and reloading the page per frame would restart the flow.
  // Returns the resolved state, so a caller reads what it got.
  function debugSetCamera(opts = {}) {
    if (!rayBuf) return { skipped: 'no raymarcher (?vol=0)' };
    // Degrees in, radians held -- the same convention ?azim= and ?elev= use,
    // because a caller that has to convert is a caller that will sometimes
    // forget to.
    if (opts.azim != null) camAzim = opts.azim * Math.PI / 180;
    if (opts.elev != null) camElev = Math.max(-83, Math.min(83, opts.elev)) * Math.PI / 180;
    if (opts.dist != null) camDist = Math.max(1e-3, opts.dist) * Math.max(NX, NY, NZ);
    if (opts.dAzim != null) camAzim += opts.dAzim * Math.PI / 180;
    if (opts.dElev != null) camElev = Math.max(-1.45, Math.min(1.45, camElev + opts.dElev * Math.PI / 180));
    writeRayParams();
    return { azim: camAzim * 180 / Math.PI, elev: camElev * 180 / Math.PI,
             dist: camDist / Math.max(NX, NY, NZ) };
  }

  // ONE FRAME, RENDERED OFFSCREEN at a size the caller chooses, into a FIXED
  // format, returned as base64 RGBA. Two callers, and they want the same
  // three properties:
  //
  //   M6.4b's gate renders the same state twice in two builds and differs
  //   them pixel by pixel. The frame tools/render-d3-movie.js writes is one
  //   of 900 that have to be the same size and the same shape as each other.
  //
  // DELIBERATELY NOT THE CANVAS, for both. Going through the swap chain
  // would make the result depend on whatever size and device-pixel-ratio the
  // browser happened to give this run, would race the animation loop (so the
  // state rendered is not the state that was asked for), and on a headless
  // Chrome is not reliably readable at all. This path is encoded, submitted
  // and mapped by the caller's own await.
  //
  // THE SLICE VIEW IS RENDERED THE SAME WAY, through a second pipeline in the
  // offscreen format. It is not the more interesting picture of a tumbling
  // plate, but it is the picture this project debugs with, and a movie mode
  // that could only film one of the two views would send someone back to a
  // screen recorder for the other.
  // THE RAYMARCHER'S COST, OFF THE ANIMATION THREAD. plans/3D.md M6.4f.
  //
  // NOT rAF, and that is the whole point. A frame rate read off the frame loop
  // is the DISPLAY's: at 1152x720 and again at 2560x1440 an A/B of this shader
  // read 59.9 fps on both legs, because the frame costs under 16.6 ms either
  // way. Disabling vsync makes the two legs differ but then measures the
  // browser's frame pacing as well as the shader. So this encodes `reps` draws
  // into ONE command buffer, submits it, and waits for the queue -- no
  // presentation, no compositor, no rAF.
  //
  // AND NOT debugRenderFrame, WHICH IS ALSO OFF THE ANIMATION THREAD and is
  // the obvious thing to reach for. MEASURED at 1024x1024 on the flagship
  // card: **1181 ms per call against a 4.7 ms draw**, i.e. 250x the thing
  // being timed. It is not the copyTextureToBuffer -- it is base64 of 4 MiB
  // and the CDP round trip that returns it. That tool exists to produce an
  // IMAGE; timing through it would report its own encoder.
  //
  // AND IT REPORTS THE MINIMUM OF ITS BATCHES, which is bench-d3-interface.js's
  // rule on this desktop -- the median moves several percent between runs and
  // the minimum is the one that is reproducible.
  //
  // The resample and gradient passes are deliberately NOT encoded: the volume
  // does not change between reps, and the question is what the RAY costs.
  async function debugBenchRender(opts = {}) {
    if (!rayPipe || !offscreen.volume) return { skipped: 'no raymarcher (?vol=0)' };
    const w = Math.max(16, Math.min(4096, Math.round(opts.w || 1024)));
    const h = Math.max(16, Math.min(4096, Math.round(opts.h || 1024)));
    const reps = Math.max(1, Math.min(512, Math.round(opts.reps || 32)));
    const batches = Math.max(1, Math.min(64, Math.round(opts.batches || 5)));
    if (!offscreen.tex || offscreen.w !== w || offscreen.h !== h) {
      if (offscreen.tex) offscreen.tex.destroy();
      offscreen.tex = device.createTexture({
        size: [w, h], format: 'rgba8unorm', label: 'd3_offscreen',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      offscreen.w = w; offscreen.h = h;
    }
    await refreshVolumeBoxes();
    writeRayParams(w / h);
    const view = offscreen.tex.createView();
    const once = () => {
      const enc = device.createCommandEncoder();
      for (let i = 0; i < reps; i++) {
        const pass = enc.beginRenderPass({ colorAttachments: [{
          view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }]});
        pass.setPipeline(offscreen.volume); pass.setBindGroup(0, rayBG);
        pass.draw(6); pass.end();
      }
      device.queue.submit([enc.finish()]);
      return device.queue.onSubmittedWorkDone();
    };
    await once();                       // warm: pipeline, caches, clocks
    const ms = [];
    for (let b = 0; b < batches; b++) {
      const t0 = performance.now();
      await once();
      ms.push((performance.now() - t0) / reps);
    }
    ms.sort((a, b) => a - b);
    writeRayParams();
    return { w, h, reps, batches, msPerFrame: ms[0], median: ms[(ms.length - 1) >> 1],
             spread: (ms[ms.length - 1] - ms[0]) / ms[0], all: ms };
  }

  async function debugRenderFrame(opts = {}) {
    const o = typeof opts === 'number' ? { w: opts } : (opts || {});
    const view = o.view || viewMode;
    if (view === 'volume' && !rayPipe) return { skipped: 'no raymarcher (?vol=0)' };
    if (!['slice', 'volume'].includes(view)) return { error: `no such view: ${view}` };
    const w = Math.max(16, Math.min(2048, Math.round(o.w || 192)));
    // THE SLICE'S HEIGHT FOLLOWS ITS PLANE unless the caller insists. That is
    // resize()'s rule for the canvas and it is the same distortion either way:
    // a slice drawn to an aspect that is not the plane's renders a sphere as
    // an ellipse. The volume view has no such constraint -- a perspective
    // camera carries the aspect -- so there the default is square.
    const [pw, ph] = planeExtent(axis);
    const h = Math.max(16, Math.min(2048, Math.round(
      o.h || (view === 'slice' ? w * ph / pw : w))));
    if (!offscreen.tex || offscreen.w !== w || offscreen.h !== h) {
      if (offscreen.tex) offscreen.tex.destroy();
      offscreen.tex = device.createTexture({
        size: [w, h], format: 'rgba8unorm', label: 'd3_offscreen',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      offscreen.w = w; offscreen.h = h;
    }
    // The slice pipeline is built ON FIRST USE rather than alongside
    // renderPipe: every validation config in the suite creates that one, and
    // none of them wants a second copy of it in a format only a movie reads.
    if (view === 'slice' && !offscreen.slice) {
      offscreen.slice = await device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
        vertex: { module: renderModule, entryPoint: 'vs_main', constants: dims },
        fragment: { module: renderModule, entryPoint: 'fs_main',
          targets: [{ format: 'rgba8unorm' }], constants: renderConsts },
        primitive: { topology: 'triangle-list' },
      });
    }
    // THE BOXES MUST BE MADE TO FOLLOW THE BODY HERE, because nothing else on
    // this path does it (2026-09-12). `refreshVolumeBoxes` is driven by the
    // LIVE frame loop's status cadence, and an offscreen render runs with
    // `live=0` and no frame loop at all -- so every clip
    // `tools/render-d3-movie.js` ever filmed at ?levels>=2 with ?dynamic=1 had
    // refined volumes pinned to wherever the body was at step 0.
    //
    // IT IS WORSE THAN HAVING NO REFINED VOLUME. The stack rule is INNERMOST
    // BOX WINS, so a stale box does not politely fall back to L0 -- it
    // OVERRIDES L0 over its whole extent with coarse data replicated onto a
    // fine grid, and Q is a squared velocity gradient, so that replication
    // differentiates into stipple. Measured on the card at ?levels=3: the body
    // travelled from (128,80,96) to (180,26,96) over 2000 steps while both box
    // origins sat at their step-0 values, and 100% of the L2 volume's voxels
    // read a level COARSER than the volume they were stored in.
    //
    // Awaited BEFORE the encoder exists: it does its own readbacks and submits.
    if (view === 'volume') { await refreshVolumeBoxes(); writeRayParams(w / h); }
    const rowBytes = Math.ceil(w * 4 / 256) * 256;
    const buf = device.createBuffer({ size: rowBytes * h, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    if (view === 'volume') { encodeResample(enc); encodeScalar(enc); }
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: offscreen.tex.createView(),
      clearValue: { r: 0.07, g: 0.07, b: 0.1, a: 1 }, loadOp: 'clear', storeOp: 'store' }]});
    if (view === 'volume') { pass.setPipeline(offscreen.volume); pass.setBindGroup(0, rayBG); }
    else { pass.setPipeline(offscreen.slice); pass.setBindGroup(0, renderBG); }
    pass.draw(6); pass.end();
    enc.copyTextureToBuffer({ texture: offscreen.tex },
      { buffer: buf, bytesPerRow: rowBytes, rowsPerImage: h }, { width: w, height: h, depthOrArrayLayers: 1 });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    if (view === 'volume') writeRayParams();
    // Tightly packed, row padding removed, then base64 -- a 192x192 image is
    // 147 KB of bytes and 196 KB of text, which goes through Runtime.evaluate
    // without ceremony where an array of 147456 numbers would not.
    const px = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) px.set(raw.subarray(y * rowBytes, y * rowBytes + w * 4), y * w * 4);
    let s = '';
    for (let i = 0; i < px.length; i += 8192) s += String.fromCharCode(...px.subarray(i, i + 8192));
    return { w, h, view, step, rgba: btoa(s),
             camera: rayBuf ? { azim: camAzim, elev: camElev, dist: camDist, fov: CAM_FOV,
                                upAxis: CAM_UP_AXIS, upSign: CAM_UP_SIGN,
                                target: CAM_TARGET, follow: CAM_FOLLOW, aspect: w / h,
                                // Under ?proj=ortho, the half-height in L0
                                // CELLS -- so a caller converts pixels to
                                // cells with no camera model at all.
                                proj: camProj, orthoHalf: camDist * Math.tan(CAM_FOV / 2) } : null,
             stack: volStack.map(v => ({ level: v.level, lo: v.lo, ext: v.ext, res: v.res, h: v.h })) };
  }

  // Seed: upload the scenario's macroscopic initial condition and let the
  // GPU turn it into an equilibrium `f`. Both buffers are seeded so the
  // ping-pong parity does not matter after a reset.
  function reset() {
    device.queue.writeBuffer(mac, 0, sc.macro(params.dims, params));
    // TOPOLOGY FIRST, AND BEFORE THE SEED (M5.5b). The seed interpolates the
    // fresh coarse field into whichever blocks slotToBlock currently names,
    // so restoring the initial refinement after it would fill the tiles the
    // manager happened to leave behind and then relabel them. Only matters
    // once the manager can move a tile; it is unconditional anyway, because
    // "restore the topology" and "restore the field" are one operation and
    // splitting them by a flag is how they come apart.
    if (AMR) for (let m = 1; m < LEVELS; m++) uploadLevelTopology(L[m]);
    if (HAS_BODY) {
      device.queue.writeBuffer(bodyBuf, 0, packBodyState(params.body, {
        pinned: !!params.pinned,
        vMax: numParam('vmax', 0.2),
        // Generous by default: o_max is a blowup limiter, not a physical
        // limit, and a tumbling plate legitimately reaches high rates.
        oMax: numParam('omax', 0.5),
      }));
    }
    const enc = device.createCommandEncoder();
    // bgAB writes fB, bgBA writes fA -- both seeded, so the ping-pong parity
    // after a reset does not matter. fA is written last, and it is what the
    // pool interp below reads.
    for (const bg of [bgAB, bgBA]) {
      const p = enc.beginComputePass();
      p.setPipeline(initPipe); p.setBindGroup(0, bg);
      p.dispatchWorkgroups(disp[0], disp[1], disp[2]);
      p.end();
    }
    // Seed the pool by interpolating the freshly-initialized coarse field
    // into every tile, INTERIOR INCLUDED (GHOST_ONLY=0). There is no evolved
    // fine state to preserve at reset, and starting the fine level from a
    // coarse interpolation of the same initial condition is what makes an
    // AMR run comparable to the dense run of the same scenario.
    if (AMR) {
      const ip = enc.beginComputePass();
      ip.setPipeline(interpFullPipe);
      ip.setBindGroup(0, interpBG[0][0]);     // reset always leaves state in fA
      ip.dispatchWorkgroups(tileDisp[0], tileDisp[1], tileDisp[2]);
      ip.end();
      // ...then cascade the seed down every deeper level, IN ORDER: L2 reads
      // an L1 that this pass has already filled (M5.2b-ii). A separate
      // compute pass per level, because each must observe the previous
      // one's writes.
      for (let m = 2; m < LEVELS; m++) {
        const lp = enc.beginComputePass();
        lp.setPipeline(L[m].interpFullPipe);
        lp.setBindGroup(0, L[m].interpSeedBG);
        lp.dispatchWorkgroups(pool.FB / 4, pool.FB / 4, (pool.FB / 4) * Math.max(1, SLOTS[m]));
        lp.end();
      }
      // ...and give every level the MOMENTS of what was just seeded. `mac`
      // is derived, and until the first substep runs nothing else writes it
      // -- so without this a freshly reset pool reads as rho = 0, u = 0 to
      // anything that looks (readPoolStats, and the slice view's sampler).
      // After the seed, because it reads the `f` the seed wrote.
      for (let m = 1; m < LEVELS; m++) {
        const mp = enc.beginComputePass();
        mp.setPipeline(momentsPipe); mp.setBindGroup(0, L[m].momentsBG);
        mp.dispatchWorkgroups(pool.FB / 4, pool.FB / 4, (pool.FB / 4) * Math.max(1, SLOTS[m]));
        mp.end();
      }
    }
    device.queue.submit([enc.finish()]);
    // initEq READ mac and did not write it, so mac still holds the seed --
    // which is exactly the field the renderer should show at step 0.
    step = 0; useB = false;
    // The latch is per-RUN, not permanent: the upload above put every level
    // back on its initial refinement, which is by construction within budget,
    // and cleared freeCount[1] with it.
    poolExhausted = 0;
  }

  // Pool dispatch shapes. The slot is folded into z because 3D has no
  // fourth dispatch dimension (plans/3D.md sec 2.4).
  // Over MAX_SLOTS, not the initial active count: a slot the manager hands
  // out later must be stepped, and every pool kernel already returns early
  // on slotToBlock[slot] < 0, so covering the unused tail is free of
  // correctness risk and costs only the early-out. Identical to the old
  // shape whenever MAX_SLOTS == activeSlots, i.e. always when static.
  const tileDisp = AMR ? [Math.ceil(pool.FB / 4), Math.ceil(pool.FB / 4), (pool.FB / 4) * Math.max(1, MAX_SLOTS)] : null;
  // Per level, for M5.3's schedule. `tileDisp` walks THIS level's tiles;
  // `coalesceDisp` walks whatever the parent is -- the dense grid at level 1,
  // the parent's tiles below it (M5.2b-i's finding). Stored rather than
  // branched on, so the recursion has no level-1 special case in it.
  const poolDispFor = (slots) => [pool.FB / 4, pool.FB / 4, (pool.FB / 4) * Math.max(1, slots)];
  if (AMR) for (let m = 1; m < LEVELS; m++) {
    L[m].tileDisp = poolDispFor(SLOTS[m]);
    L[m].coalesceDisp = m === 1 ? disp : poolDispFor(SLOTS[m - 1]);
  }
  // The restriction walks a level's OWN slots: one workgroup per (block, z
  // sub-slab), the slot folded into z. ceil(RB/4)^3 workgroups cover a
  // block's RB^3 parent cells, which is 1 at RB = 4.
  const avgDispFor = (slots) => {
    const per = Math.ceil(RB / 4);
    return [per, per, per * Math.max(1, slots)];
  };
  const avgDisp = AMR ? avgDispFor(MAX_SLOTS) : null;
  if (AMR) for (let m = 1; m < LEVELS; m++) L[m].avgDisp = avgDispFor(SLOTS[m]);

  // M5.5a/M5.5b: THE DECIDE-AND-CLOSE CHAIN, encoded in one place.
  //
  // The criterion runs at the FINEST level only (M5.1b) and every coarser
  // level's want is derived from it by the 2:1 closure, deepest first: a
  // level's octets are completed before its parent gathers from it, which is
  // cascade21's own sweep order. `balance` ORs, so each derived level is
  // zeroed first.
  //
  // Shared by the manager and by debugRunBalance rather than written twice.
  // debugRunBalance is the gate that scores this chain against
  // d3-amr.mjs's refineHierarchy on real GPU data; if it ran a different
  // order from the one the solver runs, it would be scoring a chain nothing
  // uses.
  const encodeDecideChain = (enc, managePass) => {
    // THE FIELD CRITERION RUNS FIRST, because `decide` reads its answer.
    // Its own bind group, so it is encoded here rather than through
    // managePass -- and inside this function rather than at the call site, so
    // that every caller of the chain (the manager, and debugRunBalance) sees
    // the same passes in the same order. Skipped entirely when the criterion
    // is off, which is the default.
    if (Q_ON) {
      const cp = enc.beginComputePass();
      cp.setPipeline(critPipe); cp.setBindGroup(0, critBG);
      cp.dispatchWorkgroups(CRIT_NB[0], CRIT_NB[1], CRIT_NB[2]); cp.end();
    }
    for (let m = 1; m < LEVELS; m++) managePass(L[m].clearWantPipe, m);
    managePass(manageDecidePipe, LEVELS - 1);
    // The coarser levels' field half, before the cascade ORs 2:1 on top.
    // Skipped when the criterion is off, so the chain is exactly what it was.
    if (Q_ON) for (let m = 1; m < LEVELS - 1; m++) managePass(L[m].decidePipe, m);
    for (let m = LEVELS - 1; m >= 2; m--) {
      managePass(L[m].octetsPipe, m);
      managePass(L[m - 1].balancePipe, m - 1);
    }
  };

  // ORDER PER MACRO-STEP: zero -> force (+ fine force) -> physics -> step.
  //
  // force runs BEFORE step and reads the same f_in step will, so its
  // populations are the pre-streaming time-t data both the momentum-exchange
  // and the penalty formulas require. physics then integrates that force and
  // publishes the new body state, which step reads for chi and the local
  // solid velocity -- so the fluid always sees the body at the same instant
  // the force was measured on it.
  //
  // Each pass gets its own compute pass rather than one pass with several
  // pipelines, because a dispatch must observe the previous one's writes to
  // the body and force buffers, and within a single pass WebGPU offers no
  // such ordering.
  function encodeSteps(enc, n) {
    for (let s = 0; s < n; s++) {
      if (HAS_BODY) {
        const zp = enc.beginComputePass();
        zp.setPipeline(zeroPipe); zp.setBindGroup(0, zeroBG); zp.dispatchWorkgroups(1); zp.end();
        // EXACTLY ONE FORCE PASS (plans/3D.md M5.4b), and which one depends
        // only on whether there is a pool. The body lives entirely on the
        // finest level as a hard requirement, so there is no partition to
        // make and no masking to get right: with AMR the finest level's
        // kernel integrates the whole body, without it the dense kernel
        // does. Dispatching both would double-count, which is precisely
        // what the mask used to prevent -- and what the requirement now
        // makes impossible to need.
        //
        // Its own compute pass for the reason the block comment above gives,
        // and BEFORE physics, which reads the total.
        if (forcePoolPipe) {
          const fpp = enc.beginComputePass();
          fpp.setPipeline(forcePoolPipe); fpp.setBindGroup(0, forcePoolBG);
          fpp.dispatchWorkgroups(forcePoolDisp[0], forcePoolDisp[1], forcePoolDisp[2]); fpp.end();
        } else {
          const fp = enc.beginComputePass();
          fp.setPipeline(forcePipe); fp.setBindGroup(0, useB ? forceBGB : forceBGA);
          fp.dispatchWorkgroups(disp[0], disp[1], disp[2]); fp.end();
        }
        const pp = enc.beginComputePass();
        pp.setPipeline(physPipe); pp.setBindGroup(0, physicsBG); pp.dispatchWorkgroups(1); pp.end();
      }
      // --- AMR, N=2 (S_Advance) --------------------------------------------
      //
      // Both levels start the macro-step at time t. There are two couplings
      // and they want OPPOSITE pass orders, which is why this branches
      // rather than sharing a sequence:
      //
      //   explode/coalesce (M4.1b, default) puts the coarse step LAST,
      //     because coalesce writes the fine outflux into the covered cells
      //     of the time-t coarse buffer and the coarse gather then picks it
      //     up with no per-direction test.
      //   interp/average (M3, ?interface=interp) puts it FIRST, because the
      //     restriction has to land in the buffer that step just wrote.
      //
      // The coarse step does NO work under a refined region either way
      // (M4.1a).
      const cp = useB ? 1 : 0;    // which buffer holds the coarse state at t
      const coarseStep = () => {
        const p = enc.beginComputePass();
        p.setPipeline(stepPipe);
        p.setBindGroup(0, useB ? bgBA : bgAB);
        p.dispatchWorkgroups(disp[0], disp[1], disp[2]);
        p.end();
      };
      const tilePass = (pipe, bg) => {
        const tp = enc.beginComputePass();
        tp.setPipeline(pipe); tp.setBindGroup(0, bg);
        tp.dispatchWorkgroups(tileDisp[0], tileDisp[1], tileDisp[2]);
        tp.end();
      };
      const gridPass = (pipe, bg) => {
        const gp = enc.beginComputePass();
        gp.setPipeline(pipe); gp.setBindGroup(0, bg);
        gp.dispatchWorkgroups(disp[0], disp[1], disp[2]);
        gp.end();
      };
      const dispatchPass = (pipe, bg, d) => {
        const pp = enc.beginComputePass();
        pp.setPipeline(pipe); pp.setBindGroup(0, bg);
        pp.dispatchWorkgroups(d[0], d[1], d[2]);
        pp.end();
      };

      // --- M5.3: AGAL's S_Advance, as a recursion -------------------------
      //
      // ONE SUBSTEP OF LEVEL m. `parity` names which of level m's two
      // buffers holds time t; it is also the substep index, because substep
      // p reads buffer p and writes the other, so two substeps return the
      // state to buffer 0.
      //
      //   explode(m+1)   fill the child's ring from THIS level's time-t
      //                  buffer. Once per pair of child substeps: the ring
      //                  self-advances between them (GHOST = 2 is what pays
      //                  for that).
      //   recurse twice  the child takes two substeps for each of ours.
      //   coalesce(m+1)  the child's outflux back into this level's time-t
      //                  buffer, at the covered cells.
      //   step(m)        LAST, so it gathers those coalesced values as
      //                  ordinary neighbours and needs no per-direction test.
      //
      // LEVEL 0 IS THE BASE CASE AND IT IS A REAL ONE, not a convenience:
      // L0 is the dense grid with no ring and no tiles (plans/AMR-multilevel
      // decision 1, inherited). Everything above it is the uniform tile
      // abstraction, which is why every OTHER level is one line.
      //
      // THE RECURSION IS ON THE HOST. WebGPU has no recursion in command
      // encoding -- and does not need one, because this emits a FLAT command
      // stream. What the 2D plan's M7 warns about is the hand-flattened
      // stanza this replaces: at depth 3 it was already 14 dispatches in a
      // fixed order, and depth 4 would have been another copy of it.
      const advanceLevel = (m, parity) => {
        const ch = L[m + 1];
        if (ch) {
          if (!AMR_SKIP.has('explode')) dispatchPass(ch.explodePipe, ch.explodeBG[parity], ch.tileDisp);
          if (!AMR_SKIP.has('step')) { advanceLevel(m + 1, 0); advanceLevel(m + 1, 1); }
          if (!AMR_SKIP.has('coalesce')) dispatchPass(ch.coalescePipe, ch.coalesceBG[parity], ch.coalesceDisp);
        }
        if (m === 0) coarseStep();
        else dispatchPass(L[m].step1Pipe, L[m].step1BG[parity], L[m].tileDisp);
      };

      // --- dynamic refinement (M4.2b) ---------------------------------------
      //
      // BEFORE the levels advance, so the whole macro-step sees one topology
      // -- explode, both fine substeps, coalesce and the coarse step all
      // read the same blockSlot. Re-deciding mid-step would leave passes
      // disagreeing about which blocks exist.
      //
      // COARSEN AND REFINE ARE TWO PASSES and must stay two passes; the
      // shader's header has the free-list race that requires it.
      // `step + s`, NOT `step`: step only advances after the whole batch
      // (see debugStepSync's chunking), so testing `step` alone is constant
      // across every iteration of this loop -- the manager would run on all
      // 500 steps of one chunk and none of the next. A pinned body makes
      // that indistinguishable from correct, because nothing changes either
      // way, so the bit-identical gate below CANNOT catch it. Found by
      // reading, not by testing, and worth saying out loud: an
      // "it changed nothing" gate is blind to how often the nothing ran.
      if (DYNAMIC && manageDecidePipe && (step + s) >= MANAGE_START && ((step + s - MANAGE_START) % MANAGE_EVERY) === 0) {
        // THE ORDER IS THE WHOLE DESIGN (M4.2b-ii), and M5.5b makes every
        // line of it a sweep over levels rather than a single dispatch.
        //
        //   decide+close  evaluate the criterion at the FINEST level and
        //                 derive every coarser level's want from it by the
        //                 2:1 closure, ONCE, so that every pass below
        //                 answers the same question at every level. Two
        //                 passes each recomputing it is two chances to
        //                 disagree; at depth it would be 2*(LEVELS-1).
        //   drain         restrict a dying tile into its parent. MUST come
        //                 before coarsen: coarsen frees the slot and refine
        //                 can hand that same slot straight out in the next
        //                 pass, by which point the fine solution is gone.
        //   coarsen       free the slots. Only writes to freeList.
        //   refine        allocate. Only reads freeList -- see the manager's
        //                 own header for the race that forces this split.
        //   fill          initialize the just-allocated tiles from the
        //                 parent. MUST come after refine: there is no slot
        //                 to fill until refine has handed one out.
        //   clear         drop the just-filled flags, in its own pass
        //                 because clearing them inside fill is a race on
        //                 the very test fill uses to select its work.
        //
        // AND THE LEVEL ORDER IS THE OTHER HALF OF IT (M5.5b):
        //
        //   drain/coarsen run FINEST FIRST, because a dying tile restricts
        //   into its PARENT and the parent must still be holding its slot
        //   when it does. A whole subtree can die in one topology change,
        //   and finest-first collapses it one rung at a time: L3 restricts
        //   into L2, then L2 -- now carrying what L3 handed it -- restricts
        //   into L1. Coarsest-first would free the parent under the child
        //   and then average into a slot that has been handed to someone
        //   else.
        //
        //   refine/fill run COARSEST FIRST, the mirror image: a new tile is
        //   interpolated FROM its parent, so the parent must exist and must
        //   already hold real data. A newly-refined L2 tile can be the
        //   parent of a newly-refined L3 tile in the same event, and
        //   coarsest-first is what makes the second read the first's output
        //   instead of the previous owner's leftovers.
        //
        // The two sweeps are separate for the same reason coarsen and
        // refine are separate passes at one level: the free list cannot be
        // pushed and popped in one dispatch.
        const managePass = (pipe, m, wg) => {
          const mp = enc.beginComputePass();
          mp.setPipeline(pipe); mp.setBindGroup(0, L[m].manageBG);
          mp.dispatchWorkgroups(wg ?? Math.ceil(L[m].nBlocks / 64)); mp.end();
        };
        encodeDecideChain(enc, managePass);
        for (let m = LEVELS - 1; m >= 1; m--) {
          // avgDisp, NOT the L0 grid dispatch. The drain walks THIS LEVEL'S
          // SLOTS -- (per, per, per * slots), the shape its own shader
          // documents -- and it used to be encoded through gridPass, which
          // dispatches the coarse grid. At RB = 4 that made `per` 1 and the
          // slot index `wgid.z`, so the pass covered the first NZ/4 slots
          // and silently skipped every one above: 4 of 160 on the
          // `body-coarsen` config, 6 on `drift`. A tile coarsened out of a
          // skipped slot was freed without its solution ever reaching L0.
          // Invisible to every gate in place -- those configs are
          // structural, and the pool stays perfectly consistent while the
          // FIELD loses a tile's worth of fluid.
          dispatchPass(L[m].drainPipe, L[m].drainBG[cp], L[m].avgDisp);
          managePass(L[m].coarsenPipe, m);
        }
        for (let m = 1; m < LEVELS; m++) {
          managePass(L[m].refinePipe, m);
          dispatchPass(L[m].fillPipe, L[m].fillBG[cp], L[m].tileDisp);
        }
        // One thread per SLOT, so the dispatch is the slot budget rather
        // than the block count.
        for (let m = 1; m < LEVELS; m++) managePass(L[m].clearNewPipe, m, Math.ceil(L[m].slots / 64));
      }


      if (EXPLODE) {
        // M4.1b, Chen et al. 2006, driven by M5.3's recursion. At depth 2
        // this emits exactly the sequence it always did:
        //
        //   explode    L0 -> the ring, for the directions whose coarse target
        //              is covered. Reads the parent at t; no interpolation,
        //              no rescale.
        //   L1 A, B    two fine substeps. The ring ADVECTS but does not
        //              collide, which is what makes coalesce's average the
        //              uncollided states the scheme requires.
        //   coalesce   sum the ring back into the covered cells at t, and
        //              republish `mac` under the refined region.
        //   L0 x1      the coarse step LAST, now gathering coalesced values
        //              as if they were ordinary neighbours -- which is the
        //              whole reason it needs no per-direction test.
        //
        // `cp` is L0's parity, the one thing the dense base case still needs
        // from outside. See advanceLevel for what deeper levels do.
        advanceLevel(0, cp);
      } else {
        // The M3 coupling, kept for A/B only (?interface=interp). NOT
        // conservative -- see plans/3D.md M4.1b.
        coarseStep();
        if (AMR) {
          tilePass(interpGhostPipe, interpBG[cp][0]);
          if (REFLUX) gridPass(fluxPipeSet, fluxBGA);
          tilePass(step1Pipe, step1BG_AB);
          if (REFLUX) gridPass(fluxPipeAdd, fluxBGB);
          tilePass(step1Pipe, step1BG_BA);
          const ap = enc.beginComputePass();
          ap.setPipeline(avgPipe);
          ap.setBindGroup(0, useB ? avgBGA : avgBGB);
          ap.dispatchWorkgroups(avgDisp[0], avgDisp[1], avgDisp[2]);
          ap.end();
          if (REFLUX) gridPass(refluxPipe, refluxBG[cp]);
        }
      }

      useB = !useB;
    }
    step += n;
  }

  // Chunked so a long run is not one enormous command buffer (the duct
  // needs O(10^4) steps to settle), and awaited per chunk so the queue
  // cannot run arbitrarily far ahead of the page.
  const SYNC_CHUNK = 500;
  // Reads the manager's refusal counter, latches it, and turns it into a
  // stop. One 16-byte readback per chunk, on a path that already waits for
  // the queue -- so it costs a copy, not a sync. Only meaningful with the
  // manager running; static refinement can never be refused a slot.
  // LIVE SLOT ACCOUNTING (plans/3D.md M8.4). Per level: how many slots are in
  // use right now, and the HIGH-WATER MARK over the run.
  //
  // WHY THE PEAK IS THE NUMBER THAT MATTERS. Exhaustion is already a hard
  // failure (M5.4a) and that is right, but it is a CLIFF: up to the moment it
  // fires, a run one tile from the edge and a run at half its budget look
  // identical. The status line made that worse by reporting
  // `alloc.activeSlots`, which is the INITIAL static allocation and therefore
  // never moves on a dynamic run -- so the one number on screen was the one
  // number guaranteed not to answer the question.
  //
  // It matters because a FIELD criterion's set grows with the flow rather
  // than sitting still with the geometry: measured on a shed wake, the
  // Q-criterion wants 379 blocks at 20 D/U and 628 at 60 (plans/3D.md M8.4).
  // A run that survived is only evidence the budget was big enough FOR THAT
  // RUN, and without a peak there is no way to say whether it survived with
  // margin or by luck -- which is exactly what sizing the budget needs.
  const poolPeak = new Array(LEVELS).fill(0);
  const poolInUse = new Array(LEVELS).fill(0);
  // When to say so out loud, as a fraction of the budget. NOT an error -- a
  // run at 92% of its slots is producing correct physics and should finish --
  // but it is one flow excursion from a hard stop, and that is worth knowing
  // WHILE it is running rather than from the error message afterwards.
  // ?slotWarn=0 silences it.
  const SLOT_WARN = numParam('slotWarn', 0.85);
  // Single-flight, for `refreshVolumeBoxes`'s reason: the frame loop fires
  // this unawaited on the status cadence while the sync-step paths await it,
  // and the latch it sets is MONOTONIC -- so a caller that arrives mid-poll
  // is served by the poll already running rather than starting a second one.
  let poolCheckInFlight = null;
  function checkPoolExhausted() {
    if (!DYNAMIC || poolExhausted) return Promise.resolve(poolExhausted);
    if (poolCheckInFlight) return poolCheckInFlight;
    poolCheckInFlight = checkPoolExhaustedNow().finally(() => { poolCheckInFlight = null; });
    return poolCheckInFlight;
  }
  async function checkPoolExhaustedNow() {
    if (!DYNAMIC || poolExhausted) return poolExhausted;
    // EVERY LEVEL, not just level 1 (M5.5b). Each level owns its own free
    // list and its own refusal counter, and the level that runs out is the
    // FINEST one -- it holds the most tiles and it is the one the geometry
    // criterion drives directly -- so checking only level 1 would miss
    // exactly the case this exists to catch.
    for (let m = 1; m < LEVELS; m++) {
      const fc = await readI32(L[m].freeCount, 16);
      // [0] is the live free count and [2] the minimum it ever reached -- the
      // SAME readback the refusal check needs, so the accounting costs nothing
      // beyond the arithmetic. The peak comes from [2] rather than from the
      // sampled maximum of [0]: the kernel records it at the allocation, so a
      // spike between two polls is counted rather than missed.
      poolInUse[m] = L[m].slots - fc[0];
      poolPeak[m] = Math.max(poolPeak[m], L[m].slots - fc[2]);
      const n = fc[1];
      if (n <= 0) continue;
      poolExhausted = n;
      live = false;
      playBtn.textContent = 'play';
      statusEl.textContent = `error: out of pool slots at level ${m} -- the manager was refused`
        + ` a tile ${n} time(s) at step ${step}. Refinement is geometry-forced, so this puts a`
        + ` coarse/fine seam through the body; raise ?slotHeadroom= (now ${SLOT_HEADROOM})`
        + ' or refine less.'
        + ` [peak ${poolPeak[m]}/${L[m].slots} slots]`;
      break;
    }
    return poolExhausted;
  }

  // M8.1. STEP AND RECORD THE BODY FORCE, for the unsteady gate.
  //
  // The sphere cases up to now measure Cd ONCE, at the end, and assert it
  // has stopped moving. That is the right instrument for Re <= 210, where
  // Johnson & Patel (JFM 378, 1999) put the steady axisymmetric regime and
  // where there is genuinely nothing to time-average. Above Re ~ 270 the
  // wake sheds periodically, and a single late sample of a periodic signal
  // is not a mean -- it is a phase.
  //
  // So this returns the HISTORY, in the row shape tools/lib/cylinder-metrics
  // already analyses in 2D: [step, fx, fy, Cd, Cl, Cs, fz]. The first five
  // columns are exactly what its `analyze` reads, which is the point -- the
  // zero-crossing Strouhal estimator is tested code and does not want a
  // second copy.
  //
  // ONE readBody PER SAMPLE, not per step: the body state is 136 bytes and
  // the sample interval is chosen by the caller to give tens of samples per
  // shedding period, so the readback is noise against the steps between it.
  async function debugRunAndCollect(nSteps, every) {
    live = false;
    playBtn.textContent = 'play';
    const k = Math.max(1, Math.round(every) || 1);
    const history = [];
    let done = 0;
    if (await checkPoolExhausted()) return { step, history, exhausted: poolExhausted };
    while (done < nSteps) {
      const n = Math.min(k, nSteps - done);
      const enc = device.createCommandEncoder();
      encodeSteps(enc, n);          // advances `step` itself
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      done += n;
      const b = await readBody();
      history.push([step, b.fx, b.fy, b.cd ?? NaN, b.cl ?? NaN, b.cs ?? NaN, b.fz]);
      if (await checkPoolExhausted()) break;
    }
    return { step, history, exhausted: poolExhausted };
  }

  async function debugStepSync(n) {
    live = false;
    playBtn.textContent = 'play';
    if (await checkPoolExhausted()) return { step, exhausted: poolExhausted };
    let done = 0;
    while (done < n) {
      const k = Math.min(SYNC_CHUNK, n - done);
      const enc = device.createCommandEncoder();
      encodeSteps(enc, k);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      done += k;
      // STOPS ADVANCING rather than throwing: a tool that reads the returned
      // step count or debugPoolState sees it immediately, and one that does
      // neither gets a run that visibly goes nowhere with `error:` in the
      // status. Throwing would make the one config that EXPECTS exhaustion
      // (validate-d3-invariants.js's body-refine) harder to express than the
      // failure it is testing.
      if (await checkPoolExhausted()) break;
    }
    return { step, exhausted: poolExhausted };
  }

  // --- readback ------------------------------------------------------------
  const staging = device.createBuffer({ size: NCELLS * 4 * 4, usage: U.MAP_READ | U.COPY_DST });
  async function readMacro() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(mac, 0, staging, 0, NCELLS * 4 * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    return v;
  }

  // Strided full-volume sample. Returned to the validation tools instead of
  // the whole field because the whole field does not survive the trip: at
  // 48^3 it is 442k numbers, and CDP returns values as JSON. The stride
  // DIVIDES N, so every returned point is an exact lattice site and the
  // analytic reference needs no interpolation.
  async function readSubsampled(maxPerAxis) {
    const s = subsampleStride(N, maxPerAxis || 24);
    const m = await readMacro();
    const M = N / s;
    const out = { n: N, stride: s, m: M, rho: [], ux: [], uy: [], uz: [] };
    for (let z = 0; z < NZ; z += s) {
      for (let y = 0; y < NY; y += s) {
        for (let x = 0; x < NX; x += s) {
          const c = (z * NY + y) * NX + x;
          out.rho.push(m[4 * c]); out.ux.push(m[4 * c + 1]); out.uy.push(m[4 * c + 2]); out.uz.push(m[4 * c + 3]);
        }
      }
    }
    return out;
  }

  // Duct cross-section: u_x averaged along the periodic x axis, indexed
  // [z * N + y] to match d3-scenarios.mjs's ductProfile(). Averaging over x
  // is not just noise reduction -- a steady duct flow is x-invariant by
  // construction, so the SPREAD across x is itself a check that the run has
  // settled and that the periodic direction is behaving.
  async function readDuctProfile() {
    const m = await readMacro();
    const prof = new Array(N * N).fill(0);
    let maxSpread = 0;
    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        let sum = 0, lo = Infinity, hi = -Infinity;
        for (let x = 0; x < N; x++) {
          const v = m[4 * ((z * NY + y) * NX + x) + 1];
          sum += v; if (v < lo) lo = v; if (v > hi) hi = v;
        }
        prof[z * N + y] = sum / N;
        if (hi - lo > maxSpread) maxSpread = hi - lo;
      }
    }
    return { n: N, profile: prof, maxXSpread: maxSpread };
  }

  // Volume-averaged kinetic energy and enstrophy. Enstrophy is what the 3D
  // TGV benchmark's dissipation rate is built from (eps = 2 nu <omega^2/2>
  // for an incompressible periodic flow), so this is the reporting surface
  // for the tgv scenario -- see d3-scenarios.mjs on why tgv reports rather
  // than gates.
  //
  // TOTAL MASS AND TOTAL MOMENTUM RIDE ALONG, and they are not decoration:
  // they are the discriminator tools/analyze-d3-interface.js already uses at
  // the coarse/fine seam, and the same argument applies to a moving body.
  // In a PERIODIC domain with no sponge and no body force, the only thing
  // that can change the fluid's momentum is the body, so d(momentum)/dt must
  // equal minus the force the body reports -- and the two disagreeing says
  // the force is not the whole momentum transfer, which no drag comparison
  // can say (plans/3D.md D1). Summed over EVERY cell including the solid
  // interior, because the host cannot cheaply ask which cells those are;
  // under SOLID_EQ the interior contributes a constant `|S| * u_body`, so its
  // contribution to a DIFFERENCE is the few cells that changed hands.
  async function readStats() {
    const m = await readMacro();
    const at = (x, y, z, c) => m[4 * ((((z + NZ) % NZ) * NY + ((y + NY) % NY)) * NX + ((x + NX) % NX)) + 1 + c];
    let ke = 0, ens = 0, maxSpeed = 0, rhoMin = Infinity, rhoMax = -Infinity, finite = true;
    let mass = 0, px = 0, py = 0, pz = 0;
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
          const c = (z * NY + y) * NX + x;
          const rho = m[4 * c], ux = m[4 * c + 1], uy = m[4 * c + 2], uz = m[4 * c + 3];
          if (!Number.isFinite(rho + ux + uy + uz)) { finite = false; continue; }
          const sp2 = ux * ux + uy * uy + uz * uz;
          ke += 0.5 * sp2;
          mass += rho; px += rho * ux; py += rho * uy; pz += rho * uz;
          if (sp2 > maxSpeed) maxSpeed = sp2;
          if (rho < rhoMin) rhoMin = rho;
          if (rho > rhoMax) rhoMax = rho;
          const wx = (at(x, y + 1, z, 2) - at(x, y - 1, z, 2)) * 0.5 - (at(x, y, z + 1, 1) - at(x, y, z - 1, 1)) * 0.5;
          const wy = (at(x, y, z + 1, 0) - at(x, y, z - 1, 0)) * 0.5 - (at(x + 1, y, z, 2) - at(x - 1, y, z, 2)) * 0.5;
          const wz = (at(x + 1, y, z, 1) - at(x - 1, y, z, 1)) * 0.5 - (at(x, y + 1, z, 0) - at(x, y - 1, z, 0)) * 0.5;
          ens += 0.5 * (wx * wx + wy * wy + wz * wz);
        }
      }
    }
    const n3 = NCELLS;
    return {
      step, ke: ke / n3, enstrophy: ens / n3, dissipation: 2 * params.nu * (ens / n3),
      maxSpeed: Math.sqrt(maxSpeed), rhoMin, rhoMax, finite,
      // EXTENSIVE, not per-cell: these are compared against a FORCE, which is
      // also extensive, so dividing by the cell count here would put a factor
      // of NCELLS into every budget that used them.
      mass, px, py, pz,
    };
  }

  // HOW MUCH OF THE DOMAIN IS WORTH SAMPLING? plans/3D.md M8.5.
  //
  // Empty-space skipping pays in proportion to how much space is empty, and
  // M8.5's whole design rests on that fraction being small. It is cheaper to
  // measure it than to argue about it -- M8.0 is the standing reminder --
  // and this is the measurement: |omega| per cell, reduced to a MAX PER
  // BRICK, then the fraction of bricks whose max clears a threshold.
  //
  // BRICKS, NOT CELLS, and the difference is the entire point. A ray skips a
  // whole brick or none of it, so the cell-level occupancy flatters the
  // scheme: a vortex sheet one cell thick through the middle of every brick
  // is 1% of cells and 100% of bricks, and buys nothing at all. The pair is
  // reported together so that shape is visible rather than averaged away.
  //
  // |omega| rather than Q because readStats already computes the curl and
  // this is a sizing measurement, not a render. The two have the same
  // character here -- both concentrate in the wake -- and if the answer is
  // marginal on |omega| it is worth re-running on Q before building
  // anything.
  // THE SIZING MEASUREMENT (plans/3D.md M8.4). One dispatch, one readback,
  // every threshold answered on the host -- which is why the kernel reports
  // per-block max Q rather than a boolean.
  //
  // It reports three things per threshold, and the second two are the ones a
  // budget and a renderer actually need:
  //   count/frac  how many blocks the criterion wants, i.e. THE SLOT BUDGET.
  //   bbox        the flagged set's bounding box.
  //   boxRatio    bbox volume / flagged count -- M6.4c's trigger. M6.4's
  //               bounding boxes work because a geometry-forced set is ONE
  //               COMPACT REGION; a wake-following set is not, and above ~4x
  //               the boxes have stopped paying (plans/3D.md M8.4, M8.5).
  //               Measuring it here is what turns "it also invalidates M6.4's
  //               boxes" from an expectation into a number.
  const critStaging = device.createBuffer({
    size: critBuf.size, usage: U.MAP_READ | U.COPY_DST });
  async function debugCriterion(thresholds) {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(critPipe); p.setBindGroup(0, critBG);
    p.dispatchWorkgroups(CRIT_NB[0], CRIT_NB[1], CRIT_NB[2]);
    p.end();
    enc.copyBufferToBuffer(critBuf, 0, critStaging, 0, critBuf.size);
    device.queue.submit([enc.finish()]);
    await critStaging.mapAsync(GPUMapMode.READ);
    const q = new Float32Array(critStaging.getMappedRange()).slice();
    critStaging.unmap();
    const [nx, ny, nz] = CRIT_NB;
    const total = nx * ny * nz;
    // Normalized by the body's own shear scale, so the threshold is
    // dimensionless and the same number means the same thing at another Re or
    // resolution -- d3-criterion.mjs's qRef.
    const qr = Q_REF_SCALE;
    let qMax = -Infinity;
    for (let i = 0; i < total; i++) if (q[i] > qMax) qMax = q[i];
    const ths = thresholds || [0.5, 0.2, 0.1, 0.05, 0.02, 0.01];
    // The ladder's base is the ACTIVE threshold's own rotation rate, matching
    // what the manager is running (main-3d.js's REFINE_THRESH derivation).
    const cutBase = Q_THRESH * qr;
    // PERIODIC, VIA `axisSpan`, for `poolStateAt`'s reason and the third time
    // in this file (M6.4c fixed two). A plain min..max in BUFFER coordinates
    // reads the seam crossing as geometry: with a moving window the flagged
    // set travels through a periodic buffer and straddles the seam twice a
    // lap, and min..max is then nearly the whole axis. The M8.4 number this
    // path produced (boxRatio 4.6 on a wake criterion) SURVIVES -- it was
    // measured on a pinned sphere with no window, where the two agree -- but
    // the first wake number on the card would not have.
    const CRIT_NB_A = [nx, ny, nz];
    const rows = ths.map((t) => {
      const cut = t * qr;
      let n = 0;
      const used = CRIT_NB_A.map((m) => new Uint8Array(m));
      for (let i = 0; i < total; i++) {
        if (!(q[i] > cut)) continue;
        n++;
        const b = [i % nx, Math.floor(i / nx) % ny, Math.floor(i / (nx * ny))];
        for (let k = 0; k < 3; k++) used[k][b[k]] = 1;
      }
      const spans = used.map((u, k) => axisSpan(u, CRIT_NB_A[k]));
      // `hi` can sit BELOW `lo` on a wrapped span, so the extent is carried
      // rather than left to be recovered as hi - lo by every reader --
      // d3-volume.mjs's boxRatio refuses the latter instead of negating.
      const lo = spans.map((sp) => (sp ? sp.lo : 0));
      const hi = spans.map((sp, k) => (sp ? (sp.lo + sp.len - 1) % CRIT_NB_A[k] : -1));
      const ext = spans.map((sp) => (sp ? sp.len : 0));
      const bbox = n ? { lo, hi, ext,
        wraps: spans.map((sp, k) => !!sp && sp.lo + sp.len > CRIT_NB_A[k]) } : null;
      const eff = volBoxRatio(bbox, n);
      return { thresh: t, count: n, frac: n / total, bbox,
               boxVol: eff ? eff.boxBlocks : 0, boxRatio: eff ? eff.ratio : null };
    });
    // THE LADDER'S DEMAND, per level, WITHOUT ALLOCATING IT. Measuring demand
    // by refining it cannot see past the slot budget -- which is exactly
    // where the question lives, since a setting that over-refines is one that
    // exhausts. The kernel reports per-block Q for this reason: the whole
    // ladder is host arithmetic on one readback.
    //
    // An L1 block wanting level k implies its EIGHT children at each level
    // below (refinement is octet-complete from level 2 down, M5), so the tile
    // count at level j is 8^(j-1) times the number of L1 blocks wanting >= j.
    // That is the number the slot budget has to cover, and it is what makes
    // the difference between graded and ungraded concrete rather than
    // rhetorical.
    const ladderAt = (base, inc, maxLevel) => {
      const want = new Array(maxLevel + 1).fill(0);
      for (let i = 0; i < total; i++) {
        const eps = 0.5 * Math.log2(Math.max(q[i], 1e-30));
        let k = 0;
        for (let j = 1; j <= maxLevel; j++) if (eps >= base + inc * (j - 1)) k = j;
        for (let j = 1; j <= k; j++) want[j]++;
      }
      return want.slice(1).map((n, i) => ({ level: i + 1, blocks: n, tiles: n * 8 ** i }));
    };
    return { step, level: CRIT_LEVEL, nb: CRIT_NB.slice(), blkL0: CRIT_BLK_L0,
             lead: CRIT_LEAD, total, qMax, qRef: qr, qMaxNorm: qMax / qr, rows,
             ladder: (base, inc, maxLevel) => ladderAt(base, inc, maxLevel),
             // Precomputed for the common case, since a function does not
             // survive the structured clone a CDP readback does.
             demand: [0.5, 1, 2, 3, 4].map(inc => ({
               inc, levels: ladderAt(0.5 * Math.log2(Math.max(cutBase, 1e-30)), inc, LEVELS - 1) })) };
  }

  // THE ACCOUNTING, for a tool rather than an eye. A harness sizing the slot
  // budget wants the PEAK, not the instantaneous count, and wants it as a
  // number rather than out of the status line. Refreshes from the GPU so a
  // caller that has been driving debugStepSync (and therefore never let the
  // status cadence run) still gets a current answer.
  async function debugPoolUsage() {
    if (!AMR) return { amr: false, levels: [] };
    if (DYNAMIC) await checkPoolExhausted();
    return {
      amr: true, dynamic: !!DYNAMIC, step, exhausted: poolExhausted,
      warnAt: SLOT_WARN,
      levels: L.slice(1).map(l => ({
        level: l.level, slots: l.slots, blocks: l.nBlocks,
        // A static run's set is fixed at reset, so its "live" count is its
        // initial one and its peak is the same number -- said explicitly
        // rather than reporting a zero that looks like an empty pool.
        inUse: DYNAMIC ? poolInUse[l.level] : l.alloc.activeSlots,
        peak: DYNAMIC ? poolPeak[l.level] : l.alloc.activeSlots,
        peakFrac: (DYNAMIC ? poolPeak[l.level] : l.alloc.activeSlots) / l.slots,
      })),
    };
  }

  // THE GATE (plans/3D.md M8.4). Every cell whose Q exceeds the threshold must
  // sit in a refined block, checked HERE at cell granularity against a set
  // built from a per-block-max reduction on the GPU -- two different
  // computations, which is what makes agreement evidence rather than a
  // tautology. The same discipline debugCheckGeometryCoverage is held to.
  //
  // Q is recomputed on the host from `mac`, deliberately: reading the
  // criterion buffer back would check the manager against the criterion's own
  // answer and could not see the criterion itself being wrong.
  async function debugCheckQCoverage(thresh) {
    if (!AMR) return { skipped: 'no pool' };
    if (!Q_ON) return { skipped: 'the field criterion is off (?qthresh=)' };
    const m = await readMacro();
    const lv = L[LEVELS - 1];
    const bs = await readBlockSlot(LEVELS - 1);
    // The criterion's grid is the finest level's; its cells are L0 cells
    // scaled by the same factor CRIT_BLK_L0 carries.
    const at = (x, y, z) => {
      const c = 4 * ((((z + NZ) % NZ) * NY + ((y + NY) % NY)) * NX + ((x + NX) % NX));
      return [m[c + 1], m[c + 2], m[c + 3]];
    };
    const qAt = (x, y, z) => qOfGrad(gradU(
      [at(x + 1, y, z), at(x, y + 1, z), at(x, y, z + 1)],
      [at(x - 1, y, z), at(x, y - 1, z), at(x, y, z - 1)], 1));
    const cut = (thresh ?? Q_THRESH) * ((CRIT_U && CRIT_D) ? (CRIT_U / CRIT_D) ** 2 : 1);
    const r = checkFieldCoverage({
      dims: [NX, NY, NZ], rb: CRIT_BLK_L0, blockSlot: bs, qAt, thresh: cut });
    return { step, thresh: thresh ?? Q_THRESH, lead: CRIT_LEAD,
             manageEvery: MANAGE_EVERY, inUse: poolInUse[LEVELS - 1], slots: lv.slots, ...r };
  }

  // THE FALSE-NEGATIVE RATE OF THE CRITERION (plans/3D.md M8.4).
  //
  // THE PROBLEM THIS EXISTS FOR: under-refinement is SELF-CONCEALING. Fail to
  // refine a vortex, numerical dissipation kills it, Q drops, and the
  // criterion then correctly reports nothing there. It is stable rather than
  // transient -- once lost it stays lost -- so no measurement taken inside a
  // run that made the mistake can see it. Every honest detector has to give
  // the run information it would not otherwise have.
  //
  // THIS ONE USES THE INFORMATION THAT IS ALREADY LYING AROUND. Wherever a
  // level EXISTS, the field is in memory at two resolutions. The criterion
  // reads the PARENT's (L0), so evaluating it on the CHILD's as well gives
  // both the answer the criterion got and the answer it would have got with
  // twice the resolution. A block where the CHILD says refine and the PARENT
  // does not is a FALSE NEGATIVE the criterion would have made had that block
  // not already been refined -- which is exactly the L0-resolution limitation
  // the ladder carries, measured rather than argued.
  //
  // It is a rate on a SUBSET -- only where refinement already exists -- and
  // that is the honest limit of it: it cannot speak about regions nothing
  // refined. What it can do is say whether the criterion's view is adequate
  // WHERE IT CAN BE CHECKED, which is the near wake, which is where the
  // tightest vortices are.
  //
  // ON THE HOST, from the raw field, deliberately: a GPU pass reading the
  // criterion buffer back would be scoring the criterion against its own
  // answer. This recomputes Q from `mac_pool` by the same centred differences
  // d3-criterion.mjs states, which is the independent route.
  async function debugParentChildMiss(m = 1) {
    if (!AMR || !Q_ON) return { skipped: 'needs ?levels>=2 and ?qthresh=' };
    const lv = L[m];
    const bytes = Math.max(1, lv.slots) * pool.tileCells * 4 * 4;
    if (poolStagingBytes < bytes) {
      if (poolStaging) poolStaging.destroy();
      poolStaging = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
      poolStagingBytes = bytes;
    }
    const s2b = await readI32(lv.slotToBlock, lv.slots * 4);
    // The body as the GPU currently has it, so the geometry split below uses
    // where the body IS rather than where it started.
    const body = (await readBody()) || { cx: 0, cy: 0, cz: 0 };
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(lv.mac, 0, poolStaging, 0, bytes);
    // The parent's answer, as the criterion computed it this step.
    enc.copyBufferToBuffer(critBuf, 0, critStaging, 0, critBuf.size);
    device.queue.submit([enc.finish()]);
    await poolStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const mm = new Float32Array(poolStaging.getMappedRange(0, bytes)).slice();
    poolStaging.unmap();
    await critStaging.mapAsync(GPUMapMode.READ);
    const qp = new Float32Array(critStaging.getMappedRange()).slice();
    critStaging.unmap();

    const FB = pool.FB, plane = Math.max(1, lv.slots) * pool.tileCells;
    const uAt = (s, x, y, z) => {
      const c = s * pool.tileCells + (z * FB + y) * FB + x;
      return [mm[1 * plane + c], mm[2 * plane + c], mm[3 * plane + c]];
    };
    // THE LADDER'S BASE, and the child's value carries the physical-units
    // shift the parent's does not: a level-m cell is 2^-m of an L0 cell, so
    // Q_physical = Q_lattice * 4^m and eps = 0.5*log2(Q) + m. The `+ m` is
    // what makes the two comparable AT ALL -- without it the child's Q is
    // smaller by 4^m purely because its cells are smaller, and every block
    // would look like an over-refinement.
    const base = 0.5 * Math.log2(Math.max(Q_ABS, 1e-30));
    const epsOf = (q, lvl) => 0.5 * Math.log2(Math.max(q, 1e-30)) + lvl;
    // GEOMETRY-FORCED BLOCKS ARE A DIFFERENT POPULATION AND MUST BE COUNTED
    // SEPARATELY. The body shell is refined whatever Q says, and SOLID_EQ
    // holds the solid interior at feq(1, u_body), so the L0 field there is
    // uniform and its Q is ZERO -- while the child resolves the boundary
    // layer and asks for refinement. Every one of those scores as a "miss",
    // and it is a miss that costs nothing: geometry already refined the
    // block and the field criterion's opinion was never load-bearing there.
    //
    // Measured before splitting them: worstGap came out at 44 OCTAVES, a
    // factor of 1e13, which is not an under-read of anything -- it is
    // log2 of the zero floor. The number that answers the question is the
    // rate over the blocks the FIELD criterion is solely responsible for,
    // i.e. the wake.
    const bodyNear = geomForced
      ? nearBodyWant(geomForced.sdfAt([body.cx, body.cy, body.cz]), geomForced.margin)
      : () => false;
    let checked = 0, missed = 0, agreed = 0, over = 0, worstGap = 0;
    let fieldChecked = 0, fieldMissed = 0, fieldWorst = 0, geomChecked = 0;
    for (let sl = 0; sl < lv.slots; sl++) {
      const blk = s2b[sl];
      if (blk < 0) continue;
      // THE FULL INTERIOR, [GHOST, GHOST + 2*RB), not one cell in from it.
      // The +-1 the stencil needs is served by the RING, which is what GHOST=2
      // is for -- and shaving a cell off each face instead biases the child's
      // max LOW on every block, systematically, which reads as the parent
      // over-flagging. Measured that way first: `over` came back at 104-152
      // blocks against `missed` of 2, an asymmetry with no physical reading.
      let qc = 0;
      for (let z = GHOST; z < GHOST + 2 * RB; z++) {
        for (let y = GHOST; y < GHOST + 2 * RB; y++) {
          for (let x = GHOST; x < GHOST + 2 * RB; x++) {
            const J = gradU(
              [uAt(sl, x + 1, y, z), uAt(sl, x, y + 1, z), uAt(sl, x, y, z + 1)],
              [uAt(sl, x - 1, y, z), uAt(sl, x, y - 1, z), uAt(sl, x, y, z - 1)], 1);
            const q = qOfGrad(J);
            if (q > qc) qc = q;
          }
        }
      }
      // The parent's grid is level 1's; a level-m block maps onto it by a
      // shift, the same map `decide` uses.
      const nb = lv.pool.nb;
      const b = [blk % nb[0], Math.floor(blk / nb[0]) % nb[1], Math.floor(blk / (nb[0] * nb[1]))];
      const sh = Math.max(m - 1, 0);
      const c = [0, 1, 2].map(i => Math.min(b[i] >> sh, CRIT_NB[i] - 1));
      const qParent = qp[(c[2] * CRIT_NB[1] + c[1]) * CRIT_NB[0] + c[0]];
      const eC = epsOf(qc, m), eP = epsOf(qParent, 0);
      checked++;
      const childWants = eC >= base, parentWants = eP >= base;
      const isMiss = childWants && !parentWants;
      if (isMiss) { missed++; worstGap = Math.max(worstGap, eC - eP); }
      else if (childWants && parentWants) agreed++;
      else if (!childWants && parentWants) over++;
      // The block's L0-cell box, for the geometry test -- the same shape
      // nearBodyWant takes everywhere else.
      const scale = 2 ** -(m - 1);
      const lo = b.map(v => v * lv.pool.rb * scale);
      const hi = lo.map(v => v + lv.pool.rb * scale);
      if (bodyNear({ lo, hi, mid: lo.map((v, i) => (v + hi[i]) / 2) })) { geomChecked++; continue; }
      fieldChecked++;
      if (isMiss) { fieldMissed++; fieldWorst = Math.max(fieldWorst, eC - eP); }
    }
    return {
      step, level: m, checked,
      // THE COMPARISON IS ONLY FAIR AT ZERO LEAD, and the caller is told so
      // rather than left to discover it. The criterion's per-block value is a
      // max over the block DILATED by the convection lead, while the child's
      // here is a max over the block itself -- so with a lead the parent is
      // reading a strictly larger region and will look like it over-flags for
      // a reason that has nothing to do with resolution.
      lead: CRIT_LEAD, fairComparison: CRIT_LEAD === 0,
      // The headline: of the blocks where both views can be compared, how
      // often would the criterion's own (parent-resolution) view have said no
      // while the finer view says yes.
      missed, falseNegRate: checked ? missed / checked : 0,
      agreed, over,
      // In octaves of rotation rate -- i.e. how far the parent's view
      // under-read the worst case, in the ladder's own units.
      worstGapOctaves: worstGap,
      // THE NUMBER THAT ANSWERS THE QUESTION: the same rate over the blocks
      // the FIELD criterion alone is responsible for. Geometry-forced blocks
      // are excluded because they are refined whatever Q says, so a miss
      // there is free -- and because the solid interior's Q is identically
      // zero, which dominates the unsplit statistic.
      geomChecked, fieldChecked, fieldMissed,
      fieldFalseNegRate: fieldChecked ? fieldMissed / fieldChecked : 0,
      fieldWorstGapOctaves: fieldWorst,
    };
  }

  async function debugOccupancy(brick) {
    const B = Math.max(1, Math.round(brick || 8));
    const m = await readMacro();
    const at = (x, y, z, c) => m[4 * ((((z + NZ) % NZ) * NY + ((y + NY) % NY)) * NX + ((x + NX) % NX)) + 1 + c];
    const nb = [Math.ceil(NX / B), Math.ceil(NY / B), Math.ceil(NZ / B)];
    const brickMax = new Float64Array(nb[0] * nb[1] * nb[2]);
    const cells = [];
    let wMax = 0;
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
          const wx = (at(x, y + 1, z, 2) - at(x, y - 1, z, 2)) * 0.5 - (at(x, y, z + 1, 1) - at(x, y, z - 1, 1)) * 0.5;
          const wy = (at(x, y, z + 1, 0) - at(x, y, z - 1, 0)) * 0.5 - (at(x + 1, y, z, 2) - at(x - 1, y, z, 2)) * 0.5;
          const wz = (at(x + 1, y, z, 1) - at(x - 1, y, z, 1)) * 0.5 - (at(x, y + 1, z, 0) - at(x, y - 1, z, 0)) * 0.5;
          const w = Math.hypot(wx, wy, wz);
          if (!Number.isFinite(w)) continue;
          cells.push(w);
          if (w > wMax) wMax = w;
          const bi = ((z / B | 0) * nb[1] + (y / B | 0)) * nb[0] + (x / B | 0);
          if (w > brickMax[bi]) brickMax[bi] = w;
        }
      }
    }
    // Thresholds as fractions of the field's own maximum, because the
    // absolute scale is a scenario property and the question is about shape.
    const fracs = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01];
    const nBricks = brickMax.length;
    const rows = fracs.map(f => {
      const t = f * wMax;
      let nc = 0; for (const w of cells) if (w > t) nc++;
      let nbk = 0; for (const w of brickMax) if (w > t) nbk++;
      return { frac: f, threshold: t, cellFrac: nc / cells.length, brickFrac: nbk / nBricks };
    });
    return { step, brick: B, nBricks, nCells: cells.length, omegaMax: wMax, rows };
  }

  // WHERE, not just whether. plans/3D.md M8's tau probe.
  //
  // readStats answers "is the field finite" with a boolean, and a boolean
  // cannot tell a wall instability from a body instability from a seam
  // instability from BGK simply giving out in the bulk -- which is the whole
  // question when the relaxation time is being walked toward 1/2. So this
  // returns two LOCATIONS:
  //
  //   hot    the argmax of |u|, which exists on every step and whose
  //          MIGRATION is the early warning. A hotspot pinned one cell off
  //          the body surface is the bounce-back; one sitting on a
  //          coarse/fine seam is the interface; one wandering in the wake is
  //          the collision operator.
  //   first  the first non-finite cell, once there is one. By then the
  //          hotspot has usually already said where it came from, which is
  //          the point of reporting both.
  //
  // Each is annotated with the finest level covering it (d3-amr.mjs's
  // finestLevelAt on a real blockSlot readback -- the same route
  // debugCheckTreeSample scores the shader against), its distance to the
  // body surface in L0 cells, and its distance to the nearest wall. Those
  // three numbers are the diagnosis; everything else is narrative.
  async function debugHotspot() {
    const m = await readMacro();
    const sets = AMR ? await readLevelSets() : [null];
    const present = (lv, b) => sets[lv].has(b.join(','));
    const body = HAS_BODY ? await readBody() : null;
    const wallAxes = ['x', 'y', 'z'].filter(a => params.walls.includes(a));
    const dimsArr = [NX, NY, NZ];
    const annotate = (c) => {
      const x = c % NX, y = Math.floor(c / NX) % NY, z = Math.floor(c / (NX * NY));
      const p3 = [x, y, z];
      const o = { ijk: p3, rho: m[4 * c], u: [m[4 * c + 1], m[4 * c + 2], m[4 * c + 3]] };
      o.level = AMR ? finestLevelAt(p3, { levels: LEVELS, dims: dimsArr, rb: RB, present }) : 0;
      if (body) {
        // Body-frame distance, so it is the SDF the solver uses and not a
        // centre-distance that a non-spherical body would make meaningless.
        // Nearest periodic image under a moving window, matching bodyDelta3;
        // the plain difference without one.
        const d = wrapDelta3([p3[0] - body.cx, p3[1] - body.cy, p3[2] - body.cz], WIN_N);
        // The shape from the READBACK, not from params: that is what the
        // GPU is actually using, and on a free body it is also the only
        // copy that has moved.
        const shape = { kind: body.shape, a: body.a, b: body.b, c: body.c, r: body.r };
        o.sdf = sdfBody(qRotateInv([body.qw, body.qx, body.qy, body.qz], d), shape);
      }
      if (wallAxes.length) {
        o.wall = Math.min(...wallAxes.map(a => {
          const i = { x: 0, y: 1, z: 2 }[a];
          return Math.min(p3[i], dimsArr[i] - 1 - p3[i]);
        }));
      }
      return o;
    };
    let best = -1, bestC = 0, first = -1, nBad = 0;
    for (let c = 0; c < NCELLS; c++) {
      const rho = m[4 * c], ux = m[4 * c + 1], uy = m[4 * c + 2], uz = m[4 * c + 3];
      if (!Number.isFinite(rho + ux + uy + uz)) { nBad++; if (first < 0) first = c; continue; }
      const sp2 = ux * ux + uy * uy + uz * uz;
      if (sp2 > best) { best = sp2; bestC = c; }
    }
    return {
      step, finite: nBad === 0, nNonFinite: nBad,
      hot: best >= 0 ? { ...annotate(bestC), speed: Math.sqrt(best) } : null,
      first: first >= 0 ? annotate(first) : null,
    };
  }

  // Fine-level diagnostic: RMS speed over tile INTERIORS only (ring cells
  // are filled, not solved, and including them would blur exactly the
  // distinction this is for). Reported next to the coarse level's own RMS so
  // "is the fine level advancing at all" is answerable rather than inferred.
  //
  // EVERY LEVEL, and over the SLOT BUDGET rather than the initial active
  // count (M5.6). It used to read level 1 alone and walk slots
  // [0, activeSlots), which is wrong twice over once the tree is deeper than
  // two or the manager can move a tile: a blowup born on level 2 was
  // invisible here, and under ?dynamic=1 a slot below activeSlots may be
  // FREE (holding a dead tile's leftovers) while one above it is in use, so
  // the statistics were taken over the wrong set of cells entirely. The
  // authority on which slots are live is slotToBlock, so that is what it
  // reads. Same trap the codebase names elsewhere: a checker fed one level
  // of a three-level tree cannot fail.
  //
  // The staging buffer is allocated ON FIRST USE and grown to fit, because
  // sizing it up front to the deepest level's budget costs tens of MB of
  // MAP_READ memory on a run that may never call this.
  let poolStaging = null, poolStagingBytes = 0;
  async function readPoolStatsAt(m) {
    const lv = L[m];
    const bytes = Math.max(1, lv.slots) * pool.tileCells * 4 * 4;
    if (poolStagingBytes < bytes) {
      if (poolStaging) poolStaging.destroy();
      poolStaging = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
      poolStagingBytes = bytes;
    }
    const s2b = await readI32(lv.slotToBlock, lv.slots * 4);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(lv.mac, 0, poolStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await poolStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const mm = new Float32Array(poolStaging.getMappedRange(0, bytes)).slice();
    poolStaging.unmap();
    const FB = pool.FB, plane = Math.max(1, lv.slots) * pool.tileCells;
    let sq = 0, n = 0, finite = true, rhoMin = Infinity, rhoMax = -Infinity;
    for (let s = 0; s < lv.slots; s++) {
      if (s2b[s] < 0) continue;          // free slot: whatever the last owner left
      for (let z = GHOST; z < GHOST + 2 * RB; z++) {
        for (let y = GHOST; y < GHOST + 2 * RB; y++) {
          for (let x = GHOST; x < GHOST + 2 * RB; x++) {
            const c = s * pool.tileCells + (z * FB + y) * FB + x;
            const rho = mm[0 * plane + c], ux = mm[1 * plane + c], uy = mm[2 * plane + c], uz = mm[3 * plane + c];
            if (!Number.isFinite(rho + ux + uy + uz)) { finite = false; continue; }
            sq += ux * ux + uy * uy + uz * uz; n++;
            if (rho < rhoMin) rhoMin = rho;
            if (rho > rhoMax) rhoMax = rho;
          }
        }
      }
    }
    return { level: m, rms: Math.sqrt(sq / Math.max(n, 1)), cells: n, finite, rhoMin, rhoMax };
  }
  // The top-level fields stay LEVEL 1's, so every existing caller reads what
  // it always read; `byLevel` is the per-level surface. `finite` is the one
  // exception and it is deliberate: it is an ALL-levels answer, because a
  // NaN anywhere in the hierarchy is a blown-up run and reporting level 1 as
  // finite while level 2 is not would be the exact failure this is for.
  async function readPoolStats() {
    if (!AMR) return null;
    const byLevel = [];
    for (let m = 1; m < LEVELS; m++) byLevel.push(await readPoolStatsAt(m));
    const { level, ...lv1 } = byLevel[0];
    return { step, ...lv1, finite: byLevel.every(r => r.finite), byLevel };
  }

  // --- AMR structural invariants (plans/3D.md M4.2, risk #2) ---------------
  //
  // The 3D siblings of the 2D debugCheck21Balance / debugCheckGeometryCoverage
  // that tools/validate-amr-invariants.js drives. The LOGIC lives in
  // d3-amr.mjs so tools/test-d3-amr.js can run it -- including on inputs that
  // VIOLATE the invariant -- with no browser and no GPU; these are the thin
  // wrappers that hand it the live state.
  //
  // They exist before the manager does, deliberately: plans/3D.md sec 7 risk
  // #2 is that a checker written after a manager gets written to agree with
  // it, and the 2D manager's three live-verified balance bugs all sat under a
  // green suite.
  // Sized to the DEEPEST level's block grid, which is the largest: the block
  // count doubles per axis per level, so level 1's size would be 8x short at
  // level 2 -- a readback that silently returns a prefix, and a checker that
  // then reports a perfectly balanced tree because it never saw the rest.
  const blockSlotStaging = AMR
    ? device.createBuffer({ size: Math.max(...L.slice(1).map(l => l.nBlocks)) * 4, usage: U.MAP_READ | U.COPY_DST })
    : null;

  // ONE STAGING BUFFER PER KIND MEANS THE READBACKS MUST BE SERIALIZED, AND
  // THE SERIALIZATION BELONGS AT THE BUFFER (2026-09-12).
  //
  // A reader submits a copy into the staging buffer and then awaits
  // `mapAsync`. If a SECOND reader runs in that gap it encodes a copy into a
  // buffer that is pending map, which is a validation error -- "Buffer
  // (unlabeled) used in submit while pending map" -- and it takes the device
  // with it.
  //
  // The callers are a mix, which is why no single one of them can own this:
  // the sync-step paths AWAIT their readback, while the frame loop fires
  // `checkPoolExhausted` and `refreshVolumeBoxes` off the 250 ms status
  // cadence DELIBERATELY unawaited (a latch that stops the next frame is soon
  // enough, and a per-frame round trip is not). Only the buffer sees both.
  //
  // IT IS A DEPTH BUG, WHICH IS WHY ?levels=2 NEVER SHOWED IT. Each of those
  // two refreshes does one readback PER REFINED LEVEL, so depth 3 doubles the
  // window; and `mapAsync` resolves only once the queue drains, so a frame
  // that resamples and marches a 4x-resolution L2 volume can outlast the
  // cadence on its own. The two conditions arrive together.
  //
  // CHAIN, don't drop: an awaited caller wants an answer. The fire-and-forget
  // callers are single-flighted separately, so the chain cannot build a
  // backlog. A rejected link must not poison the chain, hence the `.catch`
  // on the tail but NOT on what the caller gets back.
  function serializedOn(tailRef, fn) {
    return (...args) => {
      const next = tailRef.tail.then(() => fn(...args));
      tailRef.tail = next.then(() => {}, () => {});
      return next;
    };
  }
  const blockSlotTail = { tail: Promise.resolve() };
  const readBlockSlot = serializedOn(blockSlotTail, async (m = 1) => {
    const bytes = L[m].nBlocks * 4;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(L[m].blockSlot, 0, blockSlotStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await blockSlotStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const v = new Int32Array(blockSlotStaging.getMappedRange(0, bytes)).slice();
    blockSlotStaging.unmap();
    return v;
  });

  // POOL BOOKKEEPING, read back from the GPU. blockSlot and slotToBlock are
  // inverses of each other, and freeCount + inUse must equal the slot
  // budget. Both statements are trivially true of an uploaded static map and
  // stop being trivial the moment a kernel writes them -- and cross-checking
  // blockSlot against its inverse is precisely how the 2D free-list race was
  // CONFIRMED: they disagreed for exactly the colliding slot.
  //
  // This also answers a question M4.2b-i's bit-identical gate cannot: that
  // gate shows the manager changed nothing, which is equally consistent with
  // the manager never having run. `ran` here is a positive observation --
  // freeCount is initialized by the host and only a kernel can move it, and
  // the invariants below are checked against what the GPU actually holds.
  const poolStateStaging = AMR ? device.createBuffer({
    size: Math.max(16, ...L.slice(1).map(l => l.slots * 4)), usage: U.MAP_READ | U.COPY_DST }) : null;
  // Serialized for the reason above: `checkPoolExhausted` reads through this
  // one, once per refined level, and the frame loop does not await it.
  const poolStateTail = { tail: Promise.resolve() };
  const readI32 = serializedOn(poolStateTail, async (src, bytes) => {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, poolStateStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await poolStateStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const v = new Int32Array(poolStateStaging.getMappedRange(0, bytes)).slice();
    poolStateStaging.unmap();
    return v;
  });
  // Per level, and the top-level fields still report LEVEL 1 so a caller
  // that predates the hierarchy reads what it always read. `byLevel` is the
  // surface the tools use: as of M5.6 tools/validate-d3-invariants.js takes
  // its `inUse` and `bbox` expectations from the FINEST level, because that
  // is the level the criterion is evaluated at and therefore the only one
  // where "the manager acted" is a statement about the criterion rather
  // than about the closure propagating something that happened below.
  //
  // `problems` and `slotsExhausted` are ALL-levels answers and always were:
  // a defect anywhere in the tree is a defect, and a checker fed one level
  // of a three-level tree cannot fail.
  async function debugPoolState() {
    if (!AMR) return { skipped: 'no pool (?levels=1)' };
    const byLevel = [null];
    for (let m = 1; m < LEVELS; m++) byLevel[m] = await poolStateAt(m);
    const bad = byLevel.slice(1).flatMap((r, i) => r.problems.map(p => ({ level: i + 1, ...p })));
    await checkPoolExhausted();
    return { ...byLevel[1], problems: bad, ok: bad.length === 0, byLevel: byLevel.slice(1),
             // M5.4a. Nonzero means the manager was refused a tile, which is
             // a hard failure rather than a degradation -- see the shader.
             slotsExhausted: poolExhausted };
  }
  async function poolStateAt(m) {
    const lv = L[m];
    const bs = await readBlockSlot(m);
    const s2b = await readI32(lv.slotToBlock, lv.slots * 4);
    const free = DYNAMIC ? (await readI32(lv.freeCount, 16))[0] : (lv.slots - lv.alloc.activeSlots);
    // These deliberately SHADOW the file-level level-1 names, so the body
    // below reads as it did before M5.1c while operating on level m. If you
    // are reading one of them here, it is this level's, not level 1's.
    const MAX_SLOTS = lv.slots, pool = lv.pool, poolAlloc = lv.alloc;
    const problems = [];
    let inUse = 0;
    for (let id = 0; id < pool.nBlocks; id++) {
      const slot = bs[id];
      if (slot < 0) continue;
      inUse++;
      if (slot >= MAX_SLOTS) { if (problems.length < 16) problems.push({ kind: 'slotOutOfRange', block: id, slot }); continue; }
      if (s2b[slot] !== id && problems.length < 16) problems.push({ kind: 'notInverse', block: id, slot, slotToBlock: s2b[slot] });
    }
    // And the other direction, which catches a slot handed out twice: two
    // blocks pointing at one slot leaves slotToBlock naming only one of them,
    // so the check above finds it -- but a slot marked in use that no block
    // claims is the mirror leak and needs its own pass.
    for (let slot = 0; slot < MAX_SLOTS; slot++) {
      const id = s2b[slot];
      if (id < 0) continue;
      if (bs[id] !== slot && problems.length < 16) problems.push({ kind: 'orphanSlot', slot, block: id, blockSlot: bs[id] });
    }
    // The bounding box of the refined set, in block coordinates. inUse alone
    // cannot show that a shell FOLLOWED a body: a sphere translating refines
    // as many blocks ahead as it coarsens behind, so the count is constant
    // while every tile changes hands. The box moves, and that is what
    // M4.2b-iii is actually claiming.
    //
    // PERIODIC, VIA `axisSpan` -- min..max IS WRONG HERE AND WAS WRONG UNTIL
    // 2026-09-12. With a moving window the body travels through a periodic
    // buffer and its shell travels with it, so twice a lap the shell
    // STRADDLES THE SEAM; taken as min..max its box is then nearly the whole
    // axis. The volume boxes have used the periodic span since M6.4 for
    // exactly this reason (see `axisSpan`'s own header) and this reporting
    // path simply never got it, so `boxRatio` -- M6.4c's "are boxes still the
    // right structure" number -- read the seam crossing as geometry. Measured
    // on the card: 12.29 at L2 at a straddling step against 2.0 away from
    // one, i.e. the statistic was mostly reporting where in its lap the body
    // happened to be. Same class as M6.5a: a quantity that is fine in buffer
    // coordinates and meaningless in window ones.
    const used = pool.nb.map((n) => new Uint8Array(n));
    for (let id = 0; id < pool.nBlocks; id++) {
      if (bs[id] < 0) continue;
      const b = pool.blockOf(id);
      for (let a = 0; a < 3; a++) used[a][b[a]] = 1;
    }
    const spans = used.map((u, a) => axisSpan(u, pool.nb[a]));
    const lo = spans.map((s) => (s ? s.lo : 1e9));
    const hi = spans.map((s, a) => (s ? (s.lo + s.len - 1) % pool.nb[a] : -1));
    // `hi` can be BELOW `lo` when the span wraps, so the extent is carried
    // explicitly rather than left to be recovered as hi - lo by every reader.
    const ext = spans.map((s) => (s ? s.len : 0));
    const bbox = inUse ? { lo, hi, ext, wraps: spans.map((s, a) => !!s && s.lo + s.len > pool.nb[a]) } : null;
    // M6.4c. BOX-UNION VOLUME OVER REFINED-SET VOLUME, in blocks: how much
    // M6.4's bounding boxes cost over the set they bound, and therefore
    // whether they are still the right structure. They work because a
    // geometry-forced set is ONE COMPACT REGION -- measured 1.35x on the
    // flagship case. A Q-criterion set following a shed wake is not, and
    // above ~4x the boxes have stopped paying and true per-ray descent is
    // the answer (plans/3D.md M6.4's "when this breaks", M8.5). Reported as
    // a number here rather than written down as a warning nobody re-reads.
    const boxEff = volBoxRatio(bbox, inUse);
    const budgetOk = inUse + free === MAX_SLOTS;
    if (!budgetOk && problems.length < 16) problems.push({ kind: 'budget', inUse, free, maxSlots: MAX_SLOTS });
    return { ok: problems.length === 0, problems, level: m, inUse, free, maxSlots: MAX_SLOTS,
             bbox, boxBlocks: boxEff ? boxEff.boxBlocks : null,
             boxRatio: boxEff ? boxEff.ratio : null,
             initialActive: poolAlloc.activeSlots, dynamic: !!DYNAMIC };
  }

  // EVERY level's map, read back from the GPU -- not the host's copy of what
  // it once uploaded, and not level 1 alone. A checker fed one level of a
  // three-level tree cannot fail, which is the same VACUOUS trap ?levels=3
  // used to spring before M5.0 refused it.
  async function readLevelSets() {
    const levelSets = [null];
    for (let m = 1; m < LEVELS; m++) {
      const bs = await readBlockSlot(m);
      const set = new Set();
      const lp = L[m].pool;
      for (let id = 0; id < lp.nBlocks; id++) if (bs[id] >= 0) set.add(lp.blockOf(id).join(','));
      levelSets[m] = set;
    }
    return levelSets;
  }
  // nbAt doubles per level, matching d3-amr.mjs's poolAtLevel -- the same
  // block grid cascade21 and refineHierarchy are written against.
  const nbAtLevel = (m) => pool.nb.map(n => n * 2 ** (m - 1));

  // M6. THE TREE SAMPLER, SCORED AGAINST THE HOST. The GPU walks the
  // hierarchy in shaders/common_d3_tree_sample.wgsl; d3-amr.mjs's
  // finestLevelAt walks it here from a blockSlot readback, with arithmetic
  // that shares nothing with the shader's. Two independent statements of
  // "finest active level wins", run against each other on real data -- the
  // same shape as debugRunBalance, and for the same reason.
  //
  // THE LEVEL IS THE CLAIM. Which level owns a point is a statement about
  // blockSlot alone, so the host can answer it exactly. Turning a point into
  // a value, given the level, is poolCell on an owning block, which
  // tools/test-d3-amr.js covers from both directions; what is checked here
  // is that the value is finite and that a refined region really does report
  // a refined level, because a sampler that silently fell back to L0
  // everywhere would otherwise agree with a host that did the same.
  async function debugCheckTreeSample(n) {
    if (!probePipe) return { skipped: 'no probe pipeline' };
    // A deterministic lattice at irrational-ish strides, so points land on
    // cell centres, on cell boundaries, and in both refined and unrefined
    // regions without being chosen to. Plus the wrap: negatives and points
    // past the far face must come back inside, since the domain is periodic
    // and a viewer that clamped instead would show a smeared edge.
    const k = Math.max(2, Math.round(Math.cbrt(n || 1728)));
    const pts = [];
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        for (let l = 0; l < k; l++) {
          pts.push([(i + 0.37) * NX / k - 1.0, (j + 0.63) * NY / k, (l + 0.5) * NZ / k + 0.25]);
        }
      }
    }
    const gpu = await debugSampleTree(pts);
    const sets = AMR ? await readLevelSets() : [null];
    const present = (m, b) => sets[m].has(b.join(','));
    const dims = [NX, NY, NZ];
    // The sampler sees SAMPLE_LEVELS pool levels, which is LEVELS - 1 unless
    // the run is deeper than it has bindings for. Scoring against LEVELS
    // there would report a disagreement that is the CLAMP, which is
    // announced in #status and is not a defect.
    const levels = SAMPLE_LEVELS + 1;
    let differ = 0, nonFinite = 0, first = null;
    const hist = new Array(levels).fill(0);
    for (let i = 0; i < pts.length; i++) {
      const want = AMR ? finestLevelAt(pts[i], { levels, dims, rb: RB, present }) : 0;
      const got = gpu[i].level;
      hist[Math.min(got, levels - 1)]++;
      if (got !== want) { differ++; if (!first) first = { p: pts[i], got, want }; }
      if (!Number.isFinite(gpu[i].rho) || !gpu[i].u.every(Number.isFinite)) {
        nonFinite++; if (!first) first = { p: pts[i], kind: 'nonFinite', got };
      }
      // The cell size must be the level's, or a finite difference taken
      // from it is scaled wrongly -- silently, and only in refined regions.
      if (gpu[i].h !== 2 ** -got) { differ++; if (!first) first = { p: pts[i], kind: 'h', got, h: gpu[i].h }; }
    }
    return { ok: differ === 0 && nonFinite === 0, points: pts.length, differ, nonFinite, first,
             byLevel: hist, sampleLevels: SAMPLE_LEVELS,
             // A run with a pool where EVERY point came back level 0 is the
             // failure this exists to catch, and it is not a mismatch --
             // the host would have to be wrong the same way to hide it.
             refinedHits: hist.slice(1).reduce((a, b) => a + b, 0) };
  }

  // RING PARENT COVERAGE (plans/3D.md M5.2a). Every ring cell's parent cell
  // must sit in an allocated PARENT tile, or explode has nothing to read.
  // Separate from 2:1 balance because it is a separate claim -- a perfectly
  // balanced tree can still have a corner ring cell with no parent, and
  // d3-amr.mjs's own test asserts the two checkers disagree on exactly that.
  async function debugCheckRingParents() {
    if (!AMR) return { skipped: 'no pool (?levels=1)' };
    const r = checkRingParentCoverage(await readLevelSets(), nbAtLevel, { levels: LEVELS });
    return {
      ok: r.violations.length === 0,
      violations: r.violations.slice(0, 16),
      nViolations: r.violations.length,
      required: r.required,
      // Nothing to check with one refined level: level 1's parent is the
      // dense L0 grid, which exists everywhere. Said out loud rather than
      // reported as a green tick, same as the 2:1 line.
      vacuous: LEVELS < 3,
    };
  }

  // M5.5a. Dispatch the criterion-and-closure chain and read back every
  // level's want set, then score it against d3-amr.mjs's refineHierarchy --
  // the host implementation `make test` already gates on violating inputs.
  // Two independent statements of one rule, run against each other, which is
  // the same discipline check21Balance and cascade21 are held to.
  //
  // It does NOT allocate: the passes here decide and close, and wiring that
  // into the allocator at depth is M5.5b. So this is a gate on the rule, not
  // on the manager.
  async function debugRunBalance() {
    if (!AMR || !manageDecidePipe) return { skipped: 'no geometry-forced criterion' };
    // The host reference this is scored against is refineHierarchy on the
    // GEOMETRY want alone, so with the field criterion on the two are
    // answering different questions and a disagreement would mean nothing.
    // Said rather than silently compared (M8.4).
    if (Q_ON) return { skipped: 'the field criterion is on; the host reference is geometry-only' };
    const enc = device.createCommandEncoder();
    // THE SAME CHAIN THE MANAGER ENCODES, not a second statement of it --
    // including the clear, which is a kernel rather than a writeBuffer for
    // the reason clearWant's own note gives.
    encodeDecideChain(enc, (pipe, m) => {
      const p = enc.beginComputePass();
      p.setPipeline(pipe); p.setBindGroup(0, L[m].manageBG);
      p.dispatchWorkgroups(Math.ceil(L[m].nBlocks / 64)); p.end();
    });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const gpu = [null];
    for (let m = 1; m < LEVELS; m++) {
      const bytes = L[m].nBlocks * 4;
      const e2 = device.createCommandEncoder();
      e2.copyBufferToBuffer(L[m].blockWant, 0, blockSlotStaging, 0, bytes);
      device.queue.submit([e2.finish()]);
      await blockSlotStaging.mapAsync(GPUMapMode.READ, 0, bytes);
      const v = new Uint32Array(blockSlotStaging.getMappedRange(0, bytes)).slice();
      blockSlotStaging.unmap();
      const set = new Set();
      for (let id = 0; id < L[m].nBlocks; id++) if (v[id]) set.add(L[m].pool.blockOf(id).join(','));
      gpu[m] = set;
    }

    // The host's answer for the body where it is NOW -- read back from the
    // GPU, not the startup position, because the body moves.
    const b = await readBody();
    const margin = Number.isFinite(MANAGE_MARGIN) ? MANAGE_MARGIN : geomForced.margin;
    // Translation AND rotation, the same bound common_d3_manage.wgsl's
    // blockWanted applies -- see its comment for why |omega| * circumradius is
    // the right term and why it is a bound rather than 2D's extrapolated pose.
    // In-plane radius under `planar`, as common_d3_manage.wgsl's lead.
    const radius = params.planar ? Math.hypot(params.body.shape.a, params.body.shape.c)
                                 : bodyCircumradius(params.body.shape);
    const lead = MANAGE_EVERY * (Math.hypot(b.vx, b.vy, b.vz)
      + Math.hypot(b.wx, b.wy, b.wz) * radius);
    const host = refineHierarchy(pool, {
      levels: LEVELS,
      want: nearBodyWant(geomForced.sdfAt([b.cx, b.cy, b.cz], [b.qw, b.qx, b.qy, b.qz]), margin + lead) });

    const diffs = [];
    for (let m = 1; m < LEVELS; m++) {
      const h = host.sets[m], g = gpu[m];
      let onlyGpu = 0, onlyHost = 0, firstOnlyGpu = null, firstOnlyHost = null;
      for (const k of g) if (!h.has(k)) { onlyGpu++; if (!firstOnlyGpu) firstOnlyGpu = k; }
      for (const k of h) if (!g.has(k)) { onlyHost++; if (!firstOnlyHost) firstOnlyHost = k; }
      diffs.push({ level: m, gpu: g.size, host: h.size, onlyGpu, onlyHost, firstOnlyGpu, firstOnlyHost });
    }
    return { ok: diffs.every(d => d.onlyGpu === 0 && d.onlyHost === 0), levels: diffs,
             vacuous: LEVELS < 3 };
  }

  async function debugCheck21Balance() {
    if (!AMR) return { skipped: 'no pool (?levels=1)' };
    const levelSets = await readLevelSets();
    const r = check21Balance(levelSets, nbAtLevel, { levels: LEVELS });
    return {
      ok: r.violations.length === 0,
      violations: r.violations.slice(0, 16),
      nViolations: r.violations.length,
      counts: r.counts,
      // Said out loud so a green result is not over-read: with one refined
      // level there is nothing that CAN violate 2:1. See check21Balance.
      vacuous: LEVELS < 3,
    };
  }

  async function debugCheckGeometryCoverage() {
    if (!AMR) return { skipped: 'no pool (?levels=1)' };
    // Only ?refine=body forces refinement from geometry. Every other mode
    // refines a fixed region that owes the body nothing, so there is no
    // requirement to check -- reported as SKIPPED, never as a pass.
    if (!geomForced) return { skipped: `?refine=${refineMode} is not geometry-forced` };
    const bs = await readBlockSlot();
    // The LIVE body position, read back from the GPU: the host does not step
    // the body, the physics kernel does, so this is the only place the truth
    // lives. Against the initial position instead, a manager that refined a
    // shell once and never moved it would pass.
    const b = await readBody();
    const r = checkGeometryCoverage(pool, bs,
      geomForced.sdfAt([b.cx, b.cy, b.cz], [b.qw, b.qx, b.qy, b.qz]), geomForced.margin);
    return { ok: r.nViolations === 0, violations: r.violations.slice(0, 16),
             nViolations: r.nViolations, required: r.required,
             bodyAt: [b.cx, b.cy, b.cz] };
  }

  // --- coarse/fine interface diagnostic ------------------------------------
  //
  // plans/3D.md M3 records an OPEN issue -- a partially-refined run grows a
  // seam error -- and quotes numbers no checked-in tool reproduced. This is
  // that measurement, plus the one control that can tell the two candidate
  // causes apart. tools/analyze-d3-interface.js drives it.
  //
  // It lives in the page because the reduction needs the WHOLE coarse field
  // and 48^3 x 4 floats does not survive a CDP round trip (see
  // readSubsampled's own note); only the summary crosses. The analytic
  // reference is d3-scenarios.mjs's, i.e. the same function the validation
  // tools score against -- not a second copy.
  //
  //   CONSERVATION. Total mass and total momentum over the coarse grid.
  //     Under a refined block `mac` holds the RESTRICTED fine moments, and
  //     the restriction is an exact arithmetic mean of rho and an exact
  //     mass-weighted mean of u, so this sum IS the hybrid system's total
  //     rather than an approximation of it. beltrami is periodic with no
  //     body and no body force, so each level ALONE conserves both exactly
  //     (streaming permutes populations; collision preserves the first two
  //     moments): any drift is the interface and nothing else.
  //
  //     That makes it the discriminator the open issue needs. If mass and
  //     momentum hold to the readback floor while the seam error grows,
  //     the interface is CONSISTENT-but-inaccurate and refluxing -- which
  //     restores conservation and nothing else -- is the wrong fix.
  //
  //   BUCKETED ERROR. Velocity error against the analytic solution, binned
  //     by signed Chebyshev distance in coarse cells to the coarse/fine
  //     interface: negative inside the refined region, positive outside,
  //     0 meaning "adjacent to it". A seam defect is a profile that peaks
  //     at |d| small and decays; a global accuracy problem is flat.
  const MAXD = 8;
  let bucketOf = null;      // per coarse cell, clamped signed distance
  function interfaceDistance() {
    if (bucketOf) return bucketOf;
    const refined = new Uint8Array(NCELLS);
    if (AMR) {
      for (let bz = 0; bz < pool.nb[2]; bz++) {
        for (let by = 0; by < pool.nb[1]; by++) {
          for (let bx = 0; bx < pool.nb[0]; bx++) {
            if (poolAlloc.blockSlot[pool.blockId(bx, by, bz)] < 0) continue;
            for (let z = bz * RB; z < (bz + 1) * RB; z++)
              for (let y = by * RB; y < (by + 1) * RB; y++)
                for (let x = bx * RB; x < (bx + 1) * RB; x++) refined[(z * NY + y) * NX + x] = 1;
          }
        }
      }
    }
    // Chebyshev distance to the opposite class, by BFS over the 26
    // neighbourhood -- so d = 0 is "touching the interface" on either side
    // and a diagonal neighbour counts, matching how a D3Q19/27 population
    // actually reaches across a seam. The grid is periodic, as the solver's
    // own streaming is.
    const dist = new Int32Array(NCELLS).fill(-1);
    let frontier = [];
    const idx = (x, y, z) => (((z + NZ) % NZ) * NY + ((y + NY) % NY)) * NX + ((x + NX) % NX);
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
          const c = (z * NY + y) * NX + x;
          let edge = false;
          for (let dz = -1; dz <= 1 && !edge; dz++)
            for (let dy = -1; dy <= 1 && !edge; dy++)
              for (let dx = -1; dx <= 1 && !edge; dx++)
                if (refined[idx(x + dx, y + dy, z + dz)] !== refined[c]) edge = true;
          if (edge) { dist[c] = 0; frontier.push(c); }
        }
      }
    }
    let d = 0;
    while (frontier.length) {
      const next = [];
      for (const c of frontier) {
        const x = c % NX, y = Math.floor(c / NX) % NY, z = Math.floor(c / (NX * NY));
        for (let dz = -1; dz <= 1; dz++)
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const n = idx(x + dx, y + dy, z + dz);
              if (dist[n] < 0) { dist[n] = d + 1; next.push(n); }
            }
      }
      frontier = next; d++;
    }
    bucketOf = new Int32Array(NCELLS);
    for (let c = 0; c < NCELLS; c++) {
      // dist is -1 everywhere when there is no interface at all (dense, or
      // ?refine=all): every cell then falls in the far bucket, which is what
      // makes those runs the flat control.
      const k = dist[c] < 0 ? MAXD : Math.min(dist[c], MAXD);
      bucketOf[c] = refined[c] ? -k : k;
    }
    return bucketOf;
  }

  // The per-cell reflux correction actually applied, summarised. A correct
  // correction is a DIFFERENCE of two nearly-equal seam fluxes, so it should
  // be orders of magnitude below either one; a correction the size of a
  // population means one of the two halves is not measuring what the other
  // is.
  const fluxStaging = AMR ? device.createBuffer({ size: NCELLS * 4 * 4, usage: U.MAP_READ | U.COPY_DST }) : null;
  async function readFluxAcc() {
    if (!AMR || !REFLUX) return null;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(fluxAccBuf, 0, fluxStaging, 0, NCELLS * 4 * 4);
    device.queue.submit([enc.finish()]);
    await fluxStaging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(fluxStaging.getMappedRange()).slice();
    fluxStaging.unmap();
    let n = 0, sumAbsM = 0, sumM = 0, maxAbsM = 0, sumAbsP = 0, maxAbsP = 0;
    for (let c = 0; c < NCELLS; c++) {
      const dm = v[4 * c], px = v[4 * c + 1], py = v[4 * c + 2], pz = v[4 * c + 3];
      if (dm === 0 && px === 0 && py === 0 && pz === 0) continue;
      const ap = Math.hypot(px, py, pz);
      n++; sumM += dm; sumAbsM += Math.abs(dm); sumAbsP += ap;
      if (Math.abs(dm) > maxAbsM) maxAbsM = Math.abs(dm);
      if (ap > maxAbsP) maxAbsP = ap;
    }
    return { step, cells: n, sumM, sumAbsM, meanAbsM: sumAbsM / Math.max(n, 1), maxAbsM,
             meanAbsP: sumAbsP / Math.max(n, 1), maxAbsP };
  }

  async function readInterfaceDiag(t) {
    const m = await readMacro();
    const bk = interfaceDistance();
    // Kahan, because the mass sum is ~NCELLS x 1 and the drift being looked
    // for is orders of magnitude below that. rho - 1 rather than rho for the
    // same reason.
    const sums = [0, 0, 0, 0], comp = [0, 0, 0, 0];   // mass, then momentum xyz
    const kadd = (a, v) => {
      const y = v - comp[a], u = sums[a] + y;
      comp[a] = (u - sums[a]) - y; sums[a] = u;
    };
    const nb = 2 * MAXD + 1;
    const buckets = Array.from({ length: nb }, () => ({ n: 0, err2: 0, ref2: 0, sim2: 0 }));
    const isBel = scenarioName === 'beltrami';
    let i = 0;
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++, i++) {
          const rho = m[4 * i], ux = m[4 * i + 1], uy = m[4 * i + 2], uz = m[4 * i + 3];
          kadd(0, rho - 1);
          kadd(1, rho * ux); kadd(2, rho * uy); kadd(3, rho * uz);
          const b = buckets[bk[i] + MAXD];
          b.n++;
          b.sim2 += ux * ux + uy * uy + uz * uz;
          if (isBel) {
            const e = beltramiVelocityAt(x, y, z, N, params.u0, params.nu, t);
            const dx = ux - e[0], dy = uy - e[1], dz = uz - e[2];
            b.err2 += dx * dx + dy * dy + dz * dz;
            b.ref2 += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
          }
        }
      }
    }
    return {
      step, t, cells: NCELLS, maxd: MAXD, scenario: scenarioName,
      mass: sums[0], mom: [sums[1], sums[2], sums[3]], hasRef: isBel,
      buckets: buckets.map((b, j) => ({ d: j - MAXD, ...b })).filter(b => b.n > 0),
    };
  }

  // Body state and the force measured on it. Read back through the body
  // buffer rather than the force accumulator, because d3_physics.wgsl drains
  // that accumulator with atomicExchange every step and copies the values
  // into the state -- including for a pinned body, which is the whole point
  // of the sphere scenario.
  const bodyStaging = device.createBuffer({ size: BODY_FIELDS.length * 4, usage: U.MAP_READ | U.COPY_DST });
  // SERIALIZED, because there is now more than one caller. The status line
  // reads the moving window's travel from this on its own cadence (M8.3)
  // while a tool drives its own reads between debugStepSync batches, and ONE
  // staging buffer cannot be mapped twice: the second mapAsync rejects and
  // the failure surfaces somewhere else entirely, as an unexplained "Uncaught
  // (in promise)" out of whichever caller happened to be second. Chaining is
  // cheaper than a second staging buffer and covers every future caller
  // rather than the two that exist. A failed read does not poison the chain.
  //
  // THIS WAS FIXED HERE FIRST AND ONLY HERE, which is why it came back on the
  // AMR staging buffers at ?levels=3 (2026-09-12). `serializedOn` is this
  // same chain, factored out, and blockSlot and poolState now go through it
  // too -- so reach for that rather than writing a third copy of this.
  const readBody = serializedOn({ tail: Promise.resolve() },
    async () => (HAS_BODY ? readBodyNow() : null));
  async function readBodyNow() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bodyBuf, 0, bodyStaging, 0, BODY_FIELDS.length * 4);
    device.queue.submit([enc.finish()]);
    await bodyStaging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(bodyStaging.getMappedRange()).slice();
    bodyStaging.unmap();
    const b = unpackBodyState(v);
    // Cd = Fx / (1/2 rho U^2 A), rho = 1 in lattice units. Reported only
    // where a reference area and a relative speed exist -- `uRel` is the
    // `fall` scenario's, which is the tow speed in one frame and the
    // freestream in the other, and the whole point of M8.2b is that the two
    // are the same number and must be normalized by the same one.
    const uRef = params.u0 || params.uRel;
    if (params.area && uRef) {
      b.cd = b.fx / (0.5 * uRef * uRef * params.area);
      b.cl = b.fy / (0.5 * uRef * uRef * params.area);
      b.cs = b.fz / (0.5 * uRef * uRef * params.area);
    }
    // THE WINDOW, reported rather than inferred (M8.3). `travel` is how far
    // the body has actually gone, which the wrap makes unreadable from the
    // position; `win` is where it sits IN THE WINDOW, which must stay at the
    // anchor to within a cell for as long as the run lasts -- that is the
    // invariant a moving-window harness asserts, and it is one readback.
    b.travel = Math.hypot(b.dx, b.dy, b.dz);
    b.winOff = windowOffset3([b.cx, b.cy, b.cz], WIN_ANCHOR, WIN_N);
    b.win = windowCoord3([b.cx, b.cy, b.cz], b.winOff, WIN_N);
    b.winAnchor = WIN_ANCHOR.slice();
    b.step = step;
    return b;
  }

  // Status-line window state. `winTravel` is the body's own accumulated
  // displacement -- the one thing the wrap makes unreadable from the position
  // -- and `winOff` is how far the window has panned. Both are REPORTING; the
  // view's own offset is computed in the shader and the solver's in the step
  // kernels, each from the body buffer directly.
  let winTravel = 0, winOff = [0, 0, 0];
  async function refreshWindowTravel() {
    // Swallowed: this is called without being awaited, so a failure here
    // would otherwise surface as an unhandled rejection rather than as the
    // status line being one quarter-second stale, which is what it is.
    const b = await readBody().catch(() => null);
    if (!b) return;
    winOff = b.winOff;
    winTravel = b.travel;
  }

  // --- controls ------------------------------------------------------------
  const sliceSlider = document.getElementById('slider-SLICE');
  const sliceVal = document.getElementById('val-SLICE');
  const axisSel = document.getElementById('sel-AXIS');
  const modeSel = document.getElementById('sel-MODE');
  const playBtn = document.getElementById('btn-PLAY');
  sliceSlider.max = String(axisExtent(axis) - 1);
  sliceSlider.value = String(slice);
  sliceVal.textContent = String(slice);
  axisSel.value = String(axis);
  modeSel.value = String(mode);
  sliceSlider.oninput = () => { slice = parseInt(sliceSlider.value); sliceVal.textContent = sliceSlider.value; writeRenderParams(); };
  axisSel.onchange = () => {
    axis = parseInt(axisSel.value);
    // The three axes can have different extents (the sphere's box is long
    // and narrow), so the slider's range follows the chosen normal.
    sliceSlider.max = String(axisExtent(axis) - 1);
    slice = Math.min(slice, axisExtent(axis) - 1);
    sliceSlider.value = String(slice); sliceVal.textContent = String(slice);
    resize();               // the new normal may have a different aspect
    writeRenderParams();
  };
  modeSel.onchange = () => { mode = parseInt(modeSel.value); writeRenderParams(); };
  // SLICE OR VOLUME, at runtime. The option is disabled where there are no
  // volumes rather than silently falling back: a viewer that draws a
  // different thing from the one selected is the shape of bug M5.0's refusal
  // and the view-depth note both exist to prevent.
  const viewSel = document.getElementById('sel-VIEW');
  const volFieldSel = document.getElementById('sel-VOLFIELD');
  viewSel.value = viewMode;
  volFieldSel.value = String(volField);
  viewSel.querySelector('option[value=volume]').disabled = !rayPipe;
  if (!rayPipe) viewSel.title = 'no volumes -- load with ?view=volume (or ?vol=1) to build them';
  function syncViewControls() {
    const vol = viewMode === 'volume';
    volFieldSel.style.display = vol ? '' : 'none';
    modeSel.style.display = vol ? 'none' : '';
    sliceSlider.disabled = vol;
    axisSel.disabled = vol;
    canvas.style.cursor = vol ? 'grab' : '';
  }
  // No volumes, no volume view: the option is DISABLED and says why, rather
  // than snapping back to `slice` with no explanation (which it used to do,
  // and which reads as a broken dropdown). The volumes are ?vol= / ?view=
  // at page load because they are the memory.
  if (!rayPipe) {
    const opt = viewSel.querySelector('option[value="volume"]');
    if (opt) { opt.disabled = true; opt.textContent = 'volume (add ?vol=1 to the URL)'; }
  }
  viewSel.onchange = () => {
    if (viewSel.value === 'volume' && !rayPipe) { viewSel.value = viewMode; return; }
    viewMode = viewSel.value;
    syncViewControls();
    resize();
  };
  volFieldSel.onchange = () => {
    volField = parseInt(volFieldSel.value);
    // The iso and the gain are per FIELD -- an iso chosen for Q/qRef means
    // nothing against |u|/uRef -- so switching field takes that field's own
    // measured defaults rather than carrying the last one across.
    volIso = VOL_ISO_DEF[volField];
    volGain = VOL_GAIN_DEF[volField];
    writeRayParams();
  };
  syncViewControls();
  playBtn.onclick = () => { live = !live; playBtn.textContent = live ? 'pause' : 'play'; };
  document.getElementById('btn-RESET').onclick = () => { reset(); };
  playBtn.textContent = live ? 'pause' : 'play';
  document.getElementById('hint').textContent =
    `${scenarioName}  ${NX}x${NY}x${NZ}  D3Q${Q}  tau=${params.tau.toFixed(4)}  nu=${params.nu.toFixed(4)}`
    + (params.re ? `  Re=${params.re}` : '')
    + (AMR ? `  levels=${LEVELS} RB=${RB} tiles=${poolAlloc.activeSlots}` : '');

  // The canvas box is the LARGEST box with the slice's in-plane aspect that
  // fits inside the container -- computed here, in pixels, rather than
  // handed to CSS as a ratio.
  //
  // The three axes have very different extents (the sphere scenario's domain
  // is 192x128x128), so a fixed square canvas renders a sphere as an
  // ellipse. The first fix for that set `aspect-ratio` from here and left
  // the stylesheet's `width: 100cqw; height: auto; max-height: 100cqh` in
  // place, which trades one distortion for the other: `max-height` caps the
  // height while the width stays pinned to the full container, so a slice
  // TALLER than the container is squashed vertically by exactly the amount
  // the cap bites. It looked correct from whichever view happened to be
  // wider than it was tall, which is how it survived being called fixed.
  //
  // A ratio cannot survive a one-sided clamp. So fit BOTH dimensions: scale
  // by the smaller of the two ratios (the standard "contain" fit) and set
  // width and height explicitly. Symmetric in the two axes by construction,
  // so it cannot compress in either direction.
  function planeExtent(a) {
    // The in-plane extents for slice normal `a`, from the one statement of
    // the convention (PLANE_AXES), THEN swapped for an odd quarter turn --
    // a rotated picture is as tall as the plane is wide. Getting this wrong
    // does not rotate anything, it stretches: the canvas would keep the
    // unrotated aspect and render a sphere as an ellipse, which is the exact
    // distortion this function's own history is about.
    const [p, q] = PLANE_AXES[a];
    const e = [axisExtent(p), axisExtent(q)];
    return (viewTurn() & 1) ? [e[1], e[0]] : e;
  }
  function resize() {
    // THE VOLUME VIEW HAS NO IN-PLANE ASPECT TO PRESERVE: a perspective
    // camera already carries the aspect (writeRayParams hands it to the
    // shader), so the canvas takes the whole container and the picture is
    // not distorted by it. The slice view's fit below is about a SLICE being
    // as wide as the plane is -- a different problem with a different answer.
    const box = canvas.parentElement.getBoundingClientRect();
    if (viewMode === 'volume') {
      const w = Math.max(1, Math.floor(box.width || 640));
      const h = Math.max(1, Math.floor(box.height || 480));
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      const dpr0 = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(w * dpr0));
      canvas.height = Math.max(1, Math.round(h * dpr0));
      writeRayParams();
      return;
    }
    const [pw, ph] = planeExtent(axis);
    // A container can be reported as zero-sized before first layout; fall
    // back to the plane's own extent so the canvas is never 0x0 (which is a
    // WebGPU error, not merely an ugly frame).
    const bw = box.width || pw, bh = box.height || ph;
    const scale = Math.min(bw / pw, bh / ph);
    const w = Math.max(1, Math.floor(pw * scale));
    const h = Math.max(1, Math.floor(ph * scale));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
  }
  resize();
  window.addEventListener('resize', resize);

  // ORBIT BY DRAGGING, ZOOM BY WHEEL. The camera is the one thing in this
  // view that cannot be checked by a gate -- "is it pointing at the
  // interesting part" has no reference value -- so it is made cheap to move
  // instead, and ?azim= ?elev= ?dist= reproduce any position a drag found.
  // Elevation stops short of the pole because `right` is cross(fwd, up) and
  // degenerates there; the shader has a fallback, and a view that quietly
  // rolled 90 degrees at the top of a drag would be worse than a stop.
  if (rayPipe) {
    let drag = null;
    if (viewMode === 'volume') canvas.style.cursor = 'grab';
    canvas.addEventListener('pointerdown', (e) => {
      if (viewMode !== 'volume') return;
      drag = [e.clientX, e.clientY];
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      camAzim -= (e.clientX - drag[0]) * 0.008;
      camElev = Math.max(-1.45, Math.min(1.45, camElev + (e.clientY - drag[1]) * 0.008));
      drag = [e.clientX, e.clientY];
      writeRayParams();
    });
    const endDrag = () => { drag = null; canvas.style.cursor = 'grab'; };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('wheel', (e) => {
      if (viewMode !== 'volume') return;
      e.preventDefault();
      const span = Math.max(NX, NY, NZ);
      camDist = Math.max(0.2 * span, Math.min(20 * span, camDist * Math.exp(e.deltaY * 0.001)));
      writeRayParams();
    }, { passive: false });
  }

  writeRenderParams();
  reset();

  window.__D3 = {
    setLive: (v) => { live = !!v; playBtn.textContent = live ? 'pause' : 'play'; },
    isLive: () => live,
    reset,
    getStep: () => step,
    getParams: () => ({
      ...params, Q, N, NX, NY, NZ, NCELLS, scenario: scenarioName,
      hasBody: !!HAS_BODY, bounceback: USE_BOUNCEBACK,
      // THE RESOLVED window, after `?window=`, not the scenario's own wish
      // -- `?window=0` on `fall` would otherwise report the window it was
      // switched out of, which is exactly the lie that makes a control
      // useless (M8.3).
      windowAxes: WIN_AXES.slice(), windowAnchor: WIN_ANCHOR.slice(), windowDims: WIN_N.slice(),
      // The refinement mode and, where there is one, the hand-placed wake box
      // (M8.4a) -- so a harness reports the region it actually ran rather
      // than the one its URL asked for.
      refineMode, wakeBox,
      levels: LEVELS, amr: AMR,
      // How deep the VIEWER can see, which is LEVELS - 1 unless the run is
      // deeper than the sampler has bindings for. Reported so a tool scores
      // the sampler against the depth it actually has rather than assume.
      sampleLevels: SAMPLE_LEVELS,
      // WHICH VIEW IS ACTUALLY DRAWING, resolved rather than echoed -- a
      // harness that asked for ?view=volume and got the slice view (no
      // volumes built) must be able to see that, not infer it.
      view: viewMode,
      // null when ?vol=0, which is still the default under the slice view.
      // `stack` is M6.4a's per-level list; the base volume's own fields stay
      // spread at the top level so a tool written against M6.1 is unchanged.
      volume: volInfo ? { ...volInfo, count: volStack.length,
        stack: volStack.map(v => ({ level: v.level, lo: v.lo.slice(), ext: v.ext.slice(),
                                    res: v.res.slice(), h: v.h.slice(), bytes: v.bytes })) } : null,
      // Everything d3-volume.mjs needs to reproduce this frame's rays on the
      // host, which is what makes M6.4b's image difference an independent
      // check rather than the page marking its own work.
      // The scalar volume's three normalizations, so a check can undo them
      // and compare against a quantity rather than against a ratio. They are
      // the SAME references the criterion and the slice view use, which is
      // the property worth being able to assert.
      volRefs: { q: Q_REF_SCALE, omega: V_SCALE, u: U_SCALE },
      camera: volStack.length ? { azim: camAzim, elev: camElev, dist: camDist, fov: CAM_FOV,
                          // Under ?proj=ortho this is the half-height in L0
                          // CELLS, so a tool converts pixels to cells with
                          // 2 * orthoHalf / imageHeight and no camera model.
                          proj: camProj, orthoHalf: camDist * Math.tan(CAM_FOV / 2),
                          upAxis: CAM_UP_AXIS, upSign: CAM_UP_SIGN, target: CAM_TARGET,
                          follow: CAM_FOLLOW, field: VOL_FIELDS[volField],
                          iso: volIso, gain: volGain } : null,
      ...(AMR ? {
        rb: RB, fb: pool.FB, ghost: GHOST,
        blocks: pool.nBlocks, activeSlots: poolAlloc.activeSlots,
        refinedFraction: poolAlloc.activeSlots / pool.nBlocks,
        // Per level, ADDED rather than replacing the level-1 fields above,
        // so every existing tool reads exactly what it read before.
        levelTiles: L.slice(1).map(l => ({ level: l.level, tiles: l.alloc.activeSlots,
                                           slots: l.slots, blocks: l.nBlocks })),
        tileCells: pool.tileCells,
        storageRatio: storageRatio(pool),
        tauCoarse: TAU_COARSE, tauFine: TAU_FINE, reflux: REFLUX,
      } : {}),
    }),
    readSubsampled, readDuctProfile, readStats, readBody, readPoolStats, debugCriterion,
    debugPoolUsage, debugCheckQCoverage, debugParentChildMiss,
    debugCheck21Balance, debugCheckGeometryCoverage, debugCheckRingParents, debugPoolState,
    debugRunBalance, debugSampleTree, debugCheckTreeSample, debugReadVolume,
    debugRenderFrame, debugBenchRender, debugSetVolume, debugSetCamera,
    // M6.4e. `partialFrames` must be 0: any frame drawn between a refined
    // box's origin write and the ray params that place it renders that box's
    // contents at the wrong world position.
    debugVolBoxSync: () => ({ ...volSyncStat, sync: VOL_SYNC, frames: volFrameSeq }),
    debugHotspot, debugRunAndCollect, debugOccupancy,
    readInterfaceDiag, readFluxAcc,
    debugStepSync,
  };

  let lastStatus = 0;
  async function frame() {
    // Counted for refreshVolumeBoxesNow's gap check: "how many frames could
    // have been drawn between a box moving and the raymarcher being told".
    volFrameSeq++;
    const enc = device.createCommandEncoder();
    if (live) encodeSteps(enc, STEPS_PER_FRAME);
    // AFTER the steps and before the render, so the volume is the state the
    // frame is about to show rather than the previous one. Both are no-ops at
    // ?vol=0, which is still the default under the slice view -- that view
    // samples the tree directly and owes these passes nothing.
    encodeResample(enc);
    encodeScalar(enc);
    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0.07, g: 0.07, b: 0.1, a: 1 }, loadOp: 'clear', storeOp: 'store',
    }]});
    if (viewMode === 'volume') { rp.setPipeline(rayPipe); rp.setBindGroup(0, rayBG); }
    else { rp.setPipeline(renderPipe); rp.setBindGroup(0, renderBG); }
    rp.draw(6); rp.end();
    device.queue.submit([enc.finish()]);

    const now = performance.now();
    if (now - lastStatus > 250) {
      lastStatus = now;
      // Checked on the status cadence, not per frame: one 16-byte readback
      // four times a second is free, and the manager runs far less often
      // than that anyway. Deliberately NOT awaited into the frame path --
      // the latch stops the next frame, which is soon enough for a condition
      // that is already unrecoverable.
      if (live) checkPoolExhausted().catch(() => {});
      if (poolExhausted) { requestAnimationFrame(frame); return; }
      // The window's travel, for the status line only -- the VIEW pans in the
      // shader from the body buffer itself (M8.3), because a host-refreshed
      // offset is stale between readbacks and the picture then slides forward
      // and snaps back every time one lands. Not awaited into the frame path.
      if (WIN_ON) refreshWindowTravel();
      // The refined boxes follow the body (M6.4a). On the status cadence and
      // not per frame: a blockSlot readback is a round trip and the box moves
      // by whole blocks. Not awaited into the frame path, for the same reason
      // the window's travel is not.
      if (live && viewMode === 'volume') refreshVolumeBoxes().catch(() => {});
      const t = scenarioName === 'beltrami' ? `  t/td=${(step / params.td).toFixed(2)}`
        : scenarioName === 'duct' ? `  t/settle=${(step / params.settle).toFixed(2)}`
        : scenarioName === 'sphere' ? `  t/(D/U)=${(step / params.convective).toFixed(2)}`
          : params.re ? `  Re=${params.re.toFixed(0)}` : '';
      // LIVE, on a dynamic run, with the peak alongside it -- see poolPeak.
      // A static run has no live count to report: its set is fixed at reset,
      // so alloc.activeSlots IS the answer there and stays what it was.
      const amrTxt = AMR
        ? `  ${L.slice(1).map(l => (DYNAMIC
            ? `L${l.level} ${poolInUse[l.level]}/${l.slots} slots (peak ${poolPeak[l.level]}`
              + `, ${(100 * poolPeak[l.level] / l.slots).toFixed(0)}%)`
            : `L${l.level} ${l.alloc.activeSlots}/${l.nBlocks}`)).join(' ')}`
          + ` blocks (RB=${RB}, FB=${pool.FB})`
        : '';
      // THE APPROACH TO THE CLIFF, said out loud. Exhaustion is a hard stop
      // and stays one; this is the warning that the run is close to it, which
      // is the part that was missing -- a dynamic run gave no signal at all
      // between "fine" and "stopped".
      const tight = DYNAMIC && SLOT_WARN > 0
        ? L.slice(1).filter(l => poolPeak[l.level] >= SLOT_WARN * l.slots) : [];
      const slotWarn = tight.length
        ? `\n   WARNING: L${tight.map(l => `${l.level} at ${(100 * poolPeak[l.level] / l.slots).toFixed(0)}%`).join(', L')}`
          + ` of the slot budget -- raise ?slotHeadroom= (now ${SLOT_HEADROOM}) before it refuses one`
        : '';
      // The window's travel, said out loud: a window is exactly the kind of
      // thing that looks identical whether it is working or switched off, and
      // "the body is at window x = 168 having travelled 412 cells" is the one
      // line that distinguishes them.
      const winTxt = WIN_ON
        ? `   window ${'xyz'.split('').filter((_, i) => WIN_AXES[i]).join('')}`
          + ` off ${winOff.map(v => v.toFixed(0)).join(',')} travelled ${winTravel.toFixed(1)}`
        : '';
      // WHICH VIEW, AND WHAT IT IS SHOWING. A volume render of the wrong
      // field at the wrong iso is a black screen, and a black screen is this
      // project's definition of a failure -- so the numbers that decide
      // whether anything can appear are on the screen next to it.
      const viewTxt = viewMode === 'volume'
        ? `volume ${VOL_FIELDS[volField]} iso=${volIso.toFixed(3)} gain=${volGain.toFixed(3)}`
          + `   ${volStack.map(v => `L${v.level} ${v.res.join('x')}`).join(' ')}`
          + volDepthNote
          + (volBoxTight ? '\n   WARNING: a refined box has outgrown its volume -- raise ?volMargin=' : '')
        : `${AXIS_NAMES[axis]}-slice ${slice}`;
      statusEl.textContent = `${scenarioName}  D3Q${Q}  ${NX}x${NY}x${NZ}  step ${step}${t}${amrTxt}\n`
        + `${viewTxt}   ${live ? 'running' : 'paused'}${viewDepthNote}${winTxt}${critNote}`
        + slotWarn;
    }
    requestAnimationFrame(() => frame().catch(e => reportFatal(statusEl, e)));
  }
  frame().catch(e => reportFatal(statusEl, e));
}

init().catch(e => reportFatal(statusEl, e));
