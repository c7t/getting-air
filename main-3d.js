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
//   ?mode=       speed | vorticity | normal (or 0 | 1 | 2). `vorticity` is
//                the out-of-plane component; `normal` is the velocity along
//                the slice normal, which is what shows a duct's u_x profile.
//   ?uscale= ?vscale= ?vortGamma=   render normalization overrides
//   ?spf=        solver steps per displayed frame
//   ?live=0      start paused (validation drives debugStepSync instead)

import { reportFatal, reportNoWebGPU, reportNoAdapter } from './error-overlay.mjs';
import { assembleShader } from './shader-loader.mjs';
import { SUPPORTED_Q } from './lattice-3d.mjs';
import { SCENARIOS, SCENARIO_NAMES, resolveScenario, nuFromTau, beltramiVelocityAt } from './d3-scenarios.mjs';
import { packBodyState, unpackBodyState, BODY_FIELDS, sdfBody, qRotateInv } from './d3-body.mjs';
import { makePool, refineHierarchy, nearBodyWant, storageRatio, GHOST,
         check21Balance, checkGeometryCoverage, checkRingParentCoverage,
         cellAtLevel, finestLevelAt } from './d3-amr.mjs';

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
const numParam = (k, d) => (urlParams.has(k) ? (parseFloat(urlParams.get(k)) || d) : d);
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
    'uscale', 'vscale', 'vortGamma', 'bounceback', 'chiEps', 'vmax', 'omax',
    'levels', 'rb', 'refine', 'margin', 'boxfrac', 'dcpre', 'reflux', 'interface',
    'explin', 'orphans', 'dynamic', 'manageEvery', 'slotHeadroom', 'amrskip',
    'manageMargin', 'manageStart',
    // M6: the resample volume.
    'vol', 'volbox', 'volBudget',
    // M8.2a: the solid-interior reset, ?solideq=0 to disable for A/B.
    'solideq']);
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
    } else if (mode === 'body') {
      if (!params.body) throw new Error('?refine=body: this scenario has no body');
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
      geomForced = { radius: sh.a, margin, sdfAt: (c) => (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - sh.a };
      want = nearBodyWant(geomForced.sdfAt(bx), margin);
    } else {
      throw new Error(`?refine=${mode}: expected all, box, bar, slab or body`);
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

  // The slot budget. Static refinement knows its own answer up front and
  // allocates exactly that; dynamic refinement has to be able to GROW, so it
  // pays for headroom above the initial set. See SLOT_HEADROOM.
  //
  // PER LEVEL, and it has to be: measured on the sphere geometry, level 1
  // holds 160 tiles at ?levels=2 and 304 at ?levels=3, because a deeper
  // level demands a wider buffer shell around it. A budget derived from the
  // depth-2 answer would under-allocate the coarser levels of every deeper
  // run (plans/3D.md M5.1b).
  const maxSlotsAt = (m) => {
    const a = hier.byLevel[m];
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
    device.queue.writeBuffer(lv.freeCount, 0, new Int32Array([nFree, 0, 0, 0]));
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
  const renderBGL = device.createBindGroupLayout({
    entries: renderBGL_entries.map(e => ({ ...e, visibility: GPUShaderStage.FRAGMENT })) });
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
  const renderBG = device.createBindGroup({ layout: renderBGL, entries: treeSampleEntries() });

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
  const CHI_EPS = numParam('chiEps', 1.5);
  const sponge = params.sponge || { width: 0, u: [0, 0, 0] };
  const stepConstants = {
    ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2],
    OMEGA: 1 / params.tau,
    FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
    WALL_X: params.walls.includes('x') ? 1 : 0,
    WALL_Y: params.walls.includes('y') ? 1 : 0,
    WALL_Z: params.walls.includes('z') ? 1 : 0,
    HAS_BODY, USE_BOUNCEBACK, CHI_EPS, SOLID_EQ,
    SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2],
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
      ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2], USE_BOUNCEBACK, CHI_EPS } },
  }) : null;
  const physPipe = HAS_BODY ? await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [physicsBGL] }),
    compute: { module: physModule, entryPoint: 'main', constants: {
      GX: (params.gravity || [0, 0, 0])[0],
      GY: (params.gravity || [0, 0, 0])[1],
      GZ: (params.gravity || [0, 0, 0])[2],
      NO_FLUID_FORCE: params.noFluidForce ? 1 : 0,
    } },
  }) : null;
  const zeroPipe = HAS_BODY ? await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [zeroBGL] }),
    compute: { module: zeroModule, entryPoint: 'main' },
  }) : null;

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
      HAS_BODY, USE_BOUNCEBACK, CHI_EPS, SOLID_EQ,
      SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2],
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
      ]});
      // MARGIN comes from the SAME value refineNearBody used to build the
      // static set, not from a second read of ?margin=: the bit-identical
      // gate needs the two criteria to agree exactly, and two independent
      // parses of one parameter is how they would silently stop agreeing.
      const manageConst = { ...poolConst, HAS_BODY,
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
      manageDecidePipe = await mkManageAt('decide', deepest);
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
        { binding: 7, resource: { buffer: wantDummy.blockWant } }]});
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
          { binding: 7, resource: { buffer: (L[m + 1] || wantDummy).blockWant } }]});
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
        ...finePC, USE_BOUNCEBACK, CHI_EPS, DX_WEIGHT: 4 ** -m, ...bodyFrameAt(m),
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
          HAS_BODY, USE_BOUNCEBACK, CHI_EPS, SOLID_EQ,
          SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2],
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
        ...poolConst, HAS_BODY,
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

  const renderPipe = await device.createRenderPipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: { module: renderModule, entryPoint: 'vs_main', constants: dims },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format }], constants: {
      ...dims, RB, SAMPLE_LEVELS,
      // Pinned to 1 because the page pre-normalizes -- see U_SCALE/V_SCALE.
      VORT_SCALE: 1.0,
      // Below 1 lifts weak structure toward the top of the ramp instead of
      // leaving it in the near-black background; these fields are smooth and
      // have no vortex cores to protect from saturating.
      VORT_GAMMA: numParam('vortGamma', 0.7),
    } },
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

  // --- M6.1: the resample volume -------------------------------------------
  //
  // The hierarchy onto a dense box, for the raymarcher M6.3 will point at
  // it. Every default here comes from the ballpark in plans/3D.md M6, not
  // from taste:
  //
  //   ?vol=N       voxels per L0 cell. 0 (the default) builds nothing --
  //                the slice view samples the tree directly and owes this
  //                pass nothing, so until M6.3 has something to look at,
  //                filling a volume every frame would be pure cost. The
  //                gate config turns it on.
  //   ?volbox=     `domain` (default) or `refined` -- the finest level's
  //                bounding box. That is the 10 MB case against the full
  //                domain's 2.25 GB for the SAME data at the SAME
  //                resolution, which is the ratio AMR exists to exploit.
  //   ?volBudget=  MiB, default 256. Exceeding it REFUSES rather than
  //                silently shrinking, for M5.0's reason: a view that
  //                quietly renders a coarser volume than asked for is a
  //                plausible picture of a grid nobody chose.
  const VOL_MULT = Math.max(0, numParam('vol', 0));
  const VOL_BUDGET_MB = Math.max(1, numParam('volBudget', 256));
  const volBoxMode = urlParams.get('volbox') || 'domain';
  if (!['domain', 'refined'].includes(volBoxMode)) {
    statusEl.textContent = `error: ?volbox=${volBoxMode} is not one of domain|refined`;
    return;
  }
  // A level's refined set in L0 CELL UNITS. A level-m block spans
  // RB * 2^-(m-1) L0 cells -- level 1's blocks are RB cells wide because its
  // parent IS L0, and each rung halves that. Checked against the measured
  // flagship case: level 2's box is 12 blocks per axis at 2 L0 cells = 24.
  const refinedBoxL0 = (m) => {
    const a = hier.byLevel[m], w = RB / 2 ** (m - 1);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let id = 0; id < a.pool.nBlocks; id++) {
      if (a.blockSlot[id] < 0) continue;
      const b = a.pool.blockOf(id);
      for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], b[k] * w); hi[k] = Math.max(hi[k], (b[k] + 1) * w); }
    }
    return Number.isFinite(lo[0]) ? { lo, hi } : null;
  };
  let volTex = null, volPipe = null, volBG = null, volDisp = null, volInfo = null;
  if (VOL_MULT > 0) {
    const box = volBoxMode === 'refined' && AMR ? refinedBoxL0(LEVELS - 1) : null;
    const lo = box ? box.lo : [0, 0, 0];
    const ext = box ? box.hi.map((h, k) => h - box.lo[k]) : [NX, NY, NZ];
    const res = ext.map(e => Math.max(1, Math.round(e * VOL_MULT)));
    const bytes = res[0] * res[1] * res[2] * 8;            // rgba16float
    if (bytes > VOL_BUDGET_MB * 1048576) {
      statusEl.textContent = `error: ?vol=${VOL_MULT} over ${volBoxMode} needs a`
        + ` ${res.join('x')} volume = ${(bytes / 1048576).toFixed(0)} MiB, past ?volBudget=${VOL_BUDGET_MB}.`
        + ' Lower ?vol=, use ?volbox=refined, or raise the budget deliberately';
      return;
    }
    const lim = device.limits.maxTextureDimension3D;
    if (Math.max(...res) > lim) {
      statusEl.textContent = `error: ?vol=${VOL_MULT} needs a ${res.join('x')} volume,`
        + ` past this GPU's maxTextureDimension3D of ${lim}`;
      return;
    }
    volTex = device.createTexture({
      size: res, dimension: '3d', format: 'rgba16float',
      // COPY_SRC so debugReadVolume can score it against the sampler it was
      // filled from; TEXTURE_BINDING for M6.2 and the raymarcher.
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      label: 'd3_volume' });
    const volBGL = device.createBindGroupLayout({ entries: [
      ...renderBGL_entries.map(e => ({ ...e, visibility: GPUShaderStage.COMPUTE })),
      { binding: 10, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
    ]});
    volPipe = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [volBGL] }),
      compute: {
        module: device.createShaderModule({
          code: await loadShader(`shaders/d3_resample_q${Q}.wgsl`), label: `d3_resample_q${Q}` }),
        entryPoint: 'main',
        constants: { ...dims, RB, SAMPLE_LEVELS,
          VOL_NX: res[0], VOL_NY: res[1], VOL_NZ: res[2],
          VOL_OX: lo[0], VOL_OY: lo[1], VOL_OZ: lo[2],
          VOL_HX: ext[0] / res[0], VOL_HY: ext[1] / res[1], VOL_HZ: ext[2] / res[2] },
      },
    });
    volBG = device.createBindGroup({ layout: volBGL, entries: [
      ...treeSampleEntries(), { binding: 10, resource: volTex.createView() }]});
    volDisp = res.map(r => Math.ceil(r / 4));
    volInfo = { res, lo, ext, h: ext.map((e, k) => e / res[k]), bytes, box: volBoxMode,
                mult: VOL_MULT, voxels: res[0] * res[1] * res[2] };
  }
  const encodeResample = (enc) => {
    if (!volPipe) return;
    const vp = enc.beginComputePass();
    vp.setPipeline(volPipe); vp.setBindGroup(0, volBG);
    vp.dispatchWorkgroups(volDisp[0], volDisp[1], volDisp[2]); vp.end();
  };

  // THE GATE (M6.1). Reads the volume back and hands it to the caller
  // alongside the voxel centres, so a tool can score it against
  // debugSampleTree at the SAME physical points -- the volume must agree
  // with the sampler it was filled from, which is the one claim that does
  // not depend on anything rendering.
  //
  // A z-SLAB rather than the whole volume: at the default resolution the
  // flagship case is 36 MB, which is a fine thing to hold on the GPU and a
  // silly thing to map into JS every time a check runs.
  async function debugReadVolume(z) {
    if (!volTex) return { skipped: 'no volume (?vol=0)' };
    const [nx, ny] = volInfo.res;
    const zz = Math.min(volInfo.res[2] - 1, Math.max(0, z | 0));
    // 256-byte row alignment is a copyTextureToBuffer requirement, not a
    // suggestion: an unpadded bytesPerRow is a validation error, and a
    // validation error here is silent.
    const rowBytes = Math.ceil(nx * 8 / 256) * 256;
    const buf = device.createBuffer({ size: rowBytes * ny, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    encodeResample(enc);
    enc.copyTextureToBuffer({ texture: volTex, origin: { x: 0, y: 0, z: zz } },
      { buffer: buf, bytesPerRow: rowBytes, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: 1 });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    const out = [];
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const o = (y * rowBytes) / 2 + x * 4;
        out.push({
          p: [volInfo.lo[0] + (x + 0.5) * volInfo.h[0] - 0.5,
              volInfo.lo[1] + (y + 0.5) * volInfo.h[1] - 0.5,
              volInfo.lo[2] + (zz + 0.5) * volInfo.h[2] - 0.5],
          u: [half(raw[o]), half(raw[o + 1]), half(raw[o + 2])],
          rho: half(raw[o + 3]),
        });
      }
    }
    return { info: volInfo, z: zz, texels: out };
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
  let slice = urlParams.has('slice')
    ? Math.min(axisExtent(parseAxis()) - 1, Math.max(0, parseInt(urlParams.get('slice'))))
    : (axisExtent(parseAxis()) >> 1);
  let mode = parseMode(scenarioName);
  const STEPS_PER_FRAME = Math.max(1, Math.round(numParam('spf', Math.max(1, Math.round(262144 / NCELLS * 8)))));
  const uRefDefault = params.scenario === 'duct' ? params.uPeak
    : params.scenario === 'sphere' ? params.u0 * 1.6      // the flow accelerates around the body
    // `fall` has no freestream at all -- the fluid starts at rest and the
    // only velocity scale is the body's own terminal one.
    : params.scenario === 'fall' ? params.u_t * 1.6
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
      : scenarioName === 'fall' ? 2 * params.u_t / params.R
        : (params.k || 2 * Math.PI / N) * U_SCALE);

  function writeRenderParams() {
    const b = new ArrayBuffer(32);
    new Uint32Array(b, 0, 4).set([axis, slice, mode, 0]);
    new Float32Array(b, 16, 4).set([U_SCALE, V_SCALE, 0, 0]);
    device.queue.writeBuffer(rpBuf, 0, b);
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
    for (let m = 1; m < LEVELS; m++) managePass(L[m].clearWantPipe, m);
    managePass(manageDecidePipe, LEVELS - 1);
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
  async function checkPoolExhausted() {
    if (!DYNAMIC || poolExhausted) return poolExhausted;
    // EVERY LEVEL, not just level 1 (M5.5b). Each level owns its own free
    // list and its own refusal counter, and the level that runs out is the
    // FINEST one -- it holds the most tiles and it is the one the geometry
    // criterion drives directly -- so checking only level 1 would miss
    // exactly the case this exists to catch.
    for (let m = 1; m < LEVELS; m++) {
      const n = (await readI32(L[m].freeCount, 16))[1];
      if (n <= 0) continue;
      poolExhausted = n;
      live = false;
      playBtn.textContent = 'play';
      statusEl.textContent = `error: out of pool slots at level ${m} -- the manager was refused`
        + ` a tile ${n} time(s) at step ${step}. Refinement is geometry-forced, so this puts a`
        + ` coarse/fine seam through the body; raise ?slotHeadroom= (now ${SLOT_HEADROOM})`
        + ' or refine less.';
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
  async function readStats() {
    const m = await readMacro();
    const at = (x, y, z, c) => m[4 * ((((z + NZ) % NZ) * NY + ((y + NY) % NY)) * NX + ((x + NX) % NX)) + 1 + c];
    let ke = 0, ens = 0, maxSpeed = 0, rhoMin = Infinity, rhoMax = -Infinity, finite = true;
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
          const c = (z * NY + y) * NX + x;
          const rho = m[4 * c], ux = m[4 * c + 1], uy = m[4 * c + 2], uz = m[4 * c + 3];
          if (!Number.isFinite(rho + ux + uy + uz)) { finite = false; continue; }
          const sp2 = ux * ux + uy * uy + uz * uz;
          ke += 0.5 * sp2;
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
        const d = [p3[0] - body.cx, p3[1] - body.cy, p3[2] - body.cz];
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
  async function readBlockSlot(m = 1) {
    const bytes = L[m].nBlocks * 4;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(L[m].blockSlot, 0, blockSlotStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await blockSlotStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const v = new Int32Array(blockSlotStaging.getMappedRange(0, bytes)).slice();
    blockSlotStaging.unmap();
    return v;
  }

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
  async function readI32(src, bytes) {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, poolStateStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await poolStateStaging.mapAsync(GPUMapMode.READ, 0, bytes);
    const v = new Int32Array(poolStateStaging.getMappedRange(0, bytes)).slice();
    poolStateStaging.unmap();
    return v;
  }
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
    let lo = [1e9, 1e9, 1e9], hi = [-1, -1, -1];
    for (let id = 0; id < pool.nBlocks; id++) {
      if (bs[id] < 0) continue;
      const b = pool.blockOf(id);
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], b[a]); hi[a] = Math.max(hi[a], b[a]); }
    }
    const bbox = inUse ? { lo, hi } : null;
    const budgetOk = inUse + free === MAX_SLOTS;
    if (!budgetOk && problems.length < 16) problems.push({ kind: 'budget', inUse, free, maxSlots: MAX_SLOTS });
    return { ok: problems.length === 0, problems, level: m, inUse, free, maxSlots: MAX_SLOTS,
             bbox, initialActive: poolAlloc.activeSlots, dynamic: !!DYNAMIC };
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
    const lead = MANAGE_EVERY * Math.hypot(b.vx, b.vy, b.vz);
    const host = refineHierarchy(pool, {
      levels: LEVELS, want: nearBodyWant(geomForced.sdfAt([b.cx, b.cy, b.cz]), margin + lead) });

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
    const r = checkGeometryCoverage(pool, bs, geomForced.sdfAt([b.cx, b.cy, b.cz]), geomForced.margin);
    return { ok: r.violations.length === 0, violations: r.violations.slice(0, 16),
             nViolations: r.violations.length, required: r.required,
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
  async function readBody() {
    if (!HAS_BODY) return null;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bodyBuf, 0, bodyStaging, 0, BODY_FIELDS.length * 4);
    device.queue.submit([enc.finish()]);
    await bodyStaging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(bodyStaging.getMappedRange()).slice();
    bodyStaging.unmap();
    const b = unpackBodyState(v);
    // Cd = Fx / (1/2 rho U^2 A), rho = 1 in lattice units. Reported only
    // where a reference area and a freestream exist.
    if (params.area && params.u0) {
      b.cd = b.fx / (0.5 * params.u0 * params.u0 * params.area);
      b.cl = b.fy / (0.5 * params.u0 * params.u0 * params.area);
      b.cs = b.fz / (0.5 * params.u0 * params.u0 * params.area);
    }
    b.step = step;
    return b;
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
    // The in-plane axes for slice normal `a`, cyclically -- and this MUST
    // agree with planeDims()/fs_main in common_d3_render_slice.wgsl, which
    // maps the first component across the screen and the second down it.
    if (a === 0) return [NY, NZ];
    if (a === 1) return [NZ, NX];
    return [NX, NY];
  }
  function resize() {
    const [pw, ph] = planeExtent(axis);
    const box = canvas.parentElement.getBoundingClientRect();
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
      levels: LEVELS, amr: AMR,
      // How deep the VIEWER can see, which is LEVELS - 1 unless the run is
      // deeper than the sampler has bindings for. Reported so a tool scores
      // the sampler against the depth it actually has rather than assume.
      sampleLevels: SAMPLE_LEVELS,
      // null when ?vol=0, which is the default -- see the resample block.
      volume: volInfo,
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
    readSubsampled, readDuctProfile, readStats, readBody, readPoolStats,
    debugCheck21Balance, debugCheckGeometryCoverage, debugCheckRingParents, debugPoolState,
    debugRunBalance, debugSampleTree, debugCheckTreeSample, debugReadVolume,
    debugHotspot, debugRunAndCollect, debugOccupancy,
    readInterfaceDiag, readFluxAcc,
    debugStepSync,
  };

  let lastStatus = 0;
  async function frame() {
    const enc = device.createCommandEncoder();
    if (live) encodeSteps(enc, STEPS_PER_FRAME);
    // AFTER the steps and before the render, so the volume is the state the
    // frame is about to show rather than the previous one. A no-op at
    // ?vol=0, which is the default until M6.3 has something to look at.
    encodeResample(enc);
    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0.07, g: 0.07, b: 0.1, a: 1 }, loadOp: 'clear', storeOp: 'store',
    }]});
    rp.setPipeline(renderPipe); rp.setBindGroup(0, renderBG); rp.draw(6); rp.end();
    device.queue.submit([enc.finish()]);

    const now = performance.now();
    if (now - lastStatus > 250) {
      lastStatus = now;
      // Checked on the status cadence, not per frame: one 16-byte readback
      // four times a second is free, and the manager runs far less often
      // than that anyway. Deliberately NOT awaited into the frame path --
      // the latch stops the next frame, which is soon enough for a condition
      // that is already unrecoverable.
      if (live) checkPoolExhausted();
      if (poolExhausted) { requestAnimationFrame(frame); return; }
      const t = scenarioName === 'beltrami' ? `  t/td=${(step / params.td).toFixed(2)}`
        : scenarioName === 'duct' ? `  t/settle=${(step / params.settle).toFixed(2)}`
        : scenarioName === 'sphere' ? `  t/(D/U)=${(step / params.convective).toFixed(2)}`
          : params.re ? `  Re=${params.re.toFixed(0)}` : '';
      const amrTxt = AMR
        ? `  ${L.slice(1).map(l => `L${l.level} ${l.alloc.activeSlots}/${l.nBlocks}`).join(' ')}`
          + ` blocks (RB=${RB}, FB=${pool.FB})`
        : '';
      statusEl.textContent = `${scenarioName}  D3Q${Q}  ${NX}x${NY}x${NZ}  step ${step}${t}${amrTxt}\n`
        + `${AXIS_NAMES[axis]}-slice ${slice}   ${live ? 'running' : 'paused'}${viewDepthNote}`;
    }
    requestAnimationFrame(() => frame().catch(e => reportFatal(statusEl, e)));
  }
  frame().catch(e => reportFatal(statusEl, e));
}

init().catch(e => reportFatal(statusEl, e));
