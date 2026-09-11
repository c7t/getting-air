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
import { packBodyState, unpackBodyState, BODY_FIELDS } from './d3-body.mjs';
import { makePool, refineHierarchy, nearBodyWant, storageRatio, GHOST,
         check21Balance, checkGeometryCoverage, checkRingParentCoverage } from './d3-amr.mjs';

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
    'manageMargin', 'manageStart']);
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
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(Math.max(needBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(Math.max(needBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
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
    const s2b = new Int32Array(slots).fill(-1);
    s2b.set(alloc.slotToBlock.slice(0, Math.min(slots, alloc.activeSlots)));
    device.queue.writeBuffer(lv.blockSlot, 0, alloc.blockSlot);
    device.queue.writeBuffer(lv.slotToBlock, 0, s2b);
    // The free list, a classic GPU stack: freeCount is how many slots are
    // free, and the top of the stack lives at freeList[freeCount-1]. Slots
    // [0, activeSlots) start in use; everything above is free.
    const nFree = slots - alloc.activeSlots;
    const freeInit = new Int32Array(Math.max(1, slots));
    for (let i = 0; i < nFree; i++) freeInit[i] = alloc.activeSlots + i;
    lv.freeList = device.createBuffer({ size: Math.max(1, slots) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    lv.freeCount = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    device.queue.writeBuffer(lv.freeList, 0, freeInit);
    device.queue.writeBuffer(lv.freeCount, 0, new Int32Array([nFree, 0, 0, 0]));
    // Always allocated, even when static: `blockWant` and `slotNew` are bound
    // into the interp and average layouts, and a layout that exists in two
    // versions is exactly the 238e48c failure surface. One dummy element is
    // enough where there is no manager -- the pipelines that read them are
    // never created, and NEW_ONLY/DYING_ONLY fold the reads out of the ones
    // that are.
    lv.blockWant = device.createBuffer({ size: (DYNAMIC ? nBlocks : 1) * 4, usage: U.STORAGE | U.COPY_DST });
    lv.slotNew = device.createBuffer({ size: (DYNAMIC ? slots : 1) * 4, usage: U.STORAGE | U.COPY_DST });
    // The initial set is not "new": it was uploaded with real data by the
    // host, and marking it new would have the fill pass overwrite every tile
    // with a coarse interpolation on the first step.
    device.queue.writeBuffer(lv.slotNew, 0, new Uint32Array(DYNAMIC ? slots : 1));
    device.queue.writeBuffer(lv.blockWant, 0, new Uint32Array(DYNAMIC ? nBlocks : 1));
    return lv;
  };
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
  const forceBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
  const renderBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
  ]});
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
    { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: blockSlotBound } }]});
  const forceBGB = device.createBindGroup({ layout: forceBGL, entries: [
    { binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: bodyBuf } },
    { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: blockSlotBound } }]});
  const physicsBG = device.createBindGroup({ layout: physicsBGL, entries: [
    { binding: 0, resource: { buffer: bodyBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const zeroBG = device.createBindGroup({ layout: zeroBGL, entries: [{ binding: 0, resource: { buffer: forceBuf } }]});
  const renderBG = device.createBindGroup({ layout: renderBGL, entries: [
    { binding: 0, resource: { buffer: mac } }, { binding: 1, resource: { buffer: rpBuf } }]});

  const loadShader = (p) => assembleShader(p, (f) => fetch(f).then(r => {
    if (!r.ok) throw new Error(`failed to fetch ${f}: ${r.status}`);
    return r.text();
  }));
  const stepModule = device.createShaderModule({ code: await loadShader(`shaders/d3_step_q${Q}.wgsl`), label: `d3_step_q${Q}` });
  const renderModule = device.createShaderModule({ code: await loadShader('shaders/d3_render_slice.wgsl'), label: 'd3_render_slice' });
  const forceModule = HAS_BODY ? device.createShaderModule({ code: await loadShader(`shaders/d3_force_q${Q}.wgsl`), label: `d3_force_q${Q}` }) : null;
  const physModule = HAS_BODY ? device.createShaderModule({ code: await loadShader('shaders/d3_physics.wgsl'), label: 'd3_physics' }) : null;
  const zeroModule = HAS_BODY ? device.createShaderModule({ code: await loadShader('shaders/d3_zero_forces.wgsl'), label: 'd3_zero_forces' }) : null;

  const WG = [4, 4, 4];
  const disp = [Math.ceil(NX / WG[0]), Math.ceil(NY / WG[1]), Math.ceil(NZ / WG[2])];
  const dims = { NX, NY, NZ };
  const USE_BOUNCEBACK = urlParams.has('bounceback') ? 1 : 0;
  const CHI_EPS = numParam('chiEps', 1.5);
  const sponge = params.sponge || { width: 0, u: [0, 0, 0] };
  const stepConstants = {
    ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2],
    OMEGA: 1 / params.tau,
    FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
    WALL_X: params.walls.includes('x') ? 1 : 0,
    WALL_Y: params.walls.includes('y') ? 1 : 0,
    WALL_Z: params.walls.includes('z') ? 1 : 0,
    HAS_BODY, USE_BOUNCEBACK, CHI_EPS,
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
  const forcePipe = HAS_BODY ? await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [forceBGL] }),
    compute: { module: forceModule, entryPoint: 'main', constants: {
      ...dims, WGX: WG[0], WGY: WG[1], WGZ: WG[2], USE_BOUNCEBACK, CHI_EPS,
      // Same source as the step kernel's, for the same reason the coupling
      // constants above are: a mask mismatch would double-count or drop a
      // whole region of the body.
      HAS_POOL: stepConstants.HAS_POOL, RB: stepConstants.RB,
      NBX: stepConstants.NBX, NBY: stepConstants.NBY, NBZ: stepConstants.NBZ } },
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
  let forcePoolPipe = null, forcePoolBG = null;
  let manageCoarsenPipe = null, manageRefinePipe = null, manageBG = null;
  let manageDecidePipe = null, manageClearPipe = null;
  let fillPipe = null, drainPipe = null;

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
  if (LEVELS >= 3 && HAS_BODY) {
    statusEl.textContent = `error: ?levels=${LEVELS} with a body needs per-level force`
      + ' integration and finest-wins masking (plans/3D.md M5.4); not built yet';
    return;
  }
  // UNREACHABLE TODAY and kept deliberately: ?dynamic=1 is itself refused
  // without ?refine=body, so the body refusal above always fires first. It
  // stops being unreachable the moment M5.4 lifts that one, which is exactly
  // when it has to already be here.
  if (LEVELS >= 3 && DYNAMIC) {
    statusEl.textContent = `error: ?levels=${LEVELS} with ?dynamic=1 needs the 2:1 balance`
      + ' pass in the manager (plans/3D.md M5.5); not built yet';
    return;
  }
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
    const avgBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
      OMEGA_FINE: 1 / TAU_FINE,
      FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
      HAS_BODY, USE_BOUNCEBACK, CHI_EPS,
      SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2],
    });
    avgPipe = await mk(avgBGL, avgModule, { ...poolConst, TAU_COARSE, DC_PRE });
    // M4.2b-ii. Third pipelines over the SAME two modules and the same two
    // layouts: a tile being born wants exactly interp's coarse->fine
    // transfer, and one being absorbed wants exactly average's restriction.
    // Separate shaders would be two more copies of transfers this file
    // already has, which is how they drift apart.
    if (DYNAMIC) {
      fillPipe = await mk(interpBGL, interpModule,
        { ...poolConst, TAU_COARSE, DC_PRE, GHOST_ONLY: 0, TIME_BLEND: 0.0, NEW_ONLY: 1 });
      drainPipe = await mk(avgBGL, avgModule, { ...poolConst, TAU_COARSE, DC_PRE, DYING_ONLY: 1 });
    }

    // M4.2b-i: the dynamic-refinement manager. TWO pipelines over ONE module
    // and ONE bind group -- the split is into separate PASSES, not separate
    // resources; see common_d3_manage.wgsl on why they must not share a
    // dispatch.
    if (DYNAMIC) {
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
      const mkManage = (entry) => device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
        compute: { module: manageModule, entryPoint: entry, constants: manageConst },
      });
      manageDecidePipe = await mkManage('decide');
      manageCoarsenPipe = await mkManage('coarsen');
      manageRefinePipe = await mkManage('refine');
      manageClearPipe = await mkManage('clearNew');
      manageBG = device.createBindGroup({ layout: manageBGL, entries: [
        { binding: 0, resource: { buffer: blockSlotBuf } },
        { binding: 1, resource: { buffer: slotToBlockBuf } },
        { binding: 2, resource: { buffer: freeListBuf } },
        { binding: 3, resource: { buffer: freeCountBuf } },
        { binding: 4, resource: { buffer: bodyBuf } },
        { binding: 5, resource: { buffer: blockWantBuf } },
        { binding: 6, resource: { buffer: slotNewBuf } }]});
    }

    // M4.1d: the fine level's own force/torque reduction. Created from the
    // SAME USE_BOUNCEBACK/CHI_EPS the step and coarse-force pipelines got --
    // three kernels over one pair of values, so a mismatch cannot arise from
    // three separate copies of them.
    if (HAS_BODY) {
      const forcePoolModule = device.createShaderModule({
        code: await loadShader(`shaders/d3_force_pool_q${Q}.wgsl`), label: `d3_force_pool_q${Q}` });
      forcePoolPipe = await mk(forcePoolBGL, forcePoolModule, { ...poolConst, USE_BOUNCEBACK, CHI_EPS });
      // Reads the pool buffer substep A will read, i.e. the fine level's
      // time-t state, matching the coarse kernel's own pre-streaming read.
      forcePoolBG = device.createBindGroup({ layout: forcePoolBGL, entries: [
        { binding: 0, resource: { buffer: fPoolA } }, { binding: 1, resource: { buffer: bodyBuf } },
        { binding: 2, resource: { buffer: forceBuf } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
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
      { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: mac } },
      { binding: 4, resource: { buffer: blockWantBuf } }]});
    avgBGA = mkAvg(fA);
    avgBGB = mkAvg(fB);

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
          OMEGA_FINE: 1 / tauAtLevel(m),
          FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
          HAS_BODY, USE_BOUNCEBACK, CHI_EPS,
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
      ...dims,
      // Pinned to 1 because the page pre-normalizes -- see U_SCALE/V_SCALE.
      VORT_SCALE: 1.0,
      // Below 1 lifts weak structure toward the top of the ramp instead of
      // leaving it in the near-black background; these fields are smooth and
      // have no vortex cores to protect from saturating.
      VORT_GAMMA: numParam('vortGamma', 0.7),
    } },
    primitive: { topology: 'triangle-list' },
  });

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
    }
    device.queue.submit([enc.finish()]);
    // initEq READ mac and did not write it, so mac still holds the seed --
    // which is exactly the field the renderer should show at step 0.
    step = 0; useB = false;
    // The latch is per-RUN, not permanent: reset re-seeds the pool from the
    // initial refinement, which is by construction within budget.
    poolExhausted = 0;
    if (DYNAMIC) device.queue.writeBuffer(freeCountBuf, 0, new Int32Array([MAX_SLOTS - poolAlloc.activeSlots, 0, 0, 0]));
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
  const avgDisp = AMR ? (() => {
    const per = Math.ceil(RB / 4);
    return [per, per, per * Math.max(1, MAX_SLOTS)];
  })() : null;

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
        const fp = enc.beginComputePass();
        fp.setPipeline(forcePipe); fp.setBindGroup(0, useB ? forceBGB : forceBGA);
        fp.dispatchWorkgroups(disp[0], disp[1], disp[2]); fp.end();
        // M4.1d: the fine level integrates the part of the body its own
        // tiles cover, and the coarse kernel above masked exactly those
        // cells out. Two kernels, one `forces` buffer, one atomic
        // accumulation -- so this is a partition of the integral, not a
        // second opinion on it. It goes in its own compute pass for the
        // reason the block comment above gives, and BEFORE physics, which
        // reads the total.
        if (forcePoolPipe) {
          const fpp = enc.beginComputePass();
          fpp.setPipeline(forcePoolPipe); fpp.setBindGroup(0, forcePoolBG);
          fpp.dispatchWorkgroups(tileDisp[0], tileDisp[1], tileDisp[2]); fpp.end();
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
      if (manageCoarsenPipe && (step + s) >= MANAGE_START && ((step + s - MANAGE_START) % MANAGE_EVERY) === 0) {
        const nbTot = pool.nb[0] * pool.nb[1] * pool.nb[2];
        const wgB = Math.ceil(nbTot / 64);
        const wgS = Math.ceil(MAX_SLOTS / 64);
        const blockPass = (pipe, n) => {
          const mp = enc.beginComputePass();
          mp.setPipeline(pipe); mp.setBindGroup(0, manageBG);
          mp.dispatchWorkgroups(n); mp.end();
        };
        // FIVE PASSES, AND THE ORDER IS THE WHOLE DESIGN (M4.2b-ii).
        //
        //   decide   evaluate the criterion ONCE into blockWant, so every
        //            pass below answers the same question. Two passes each
        //            recomputing it is two chances to disagree.
        //   (balance) the 2:1 closure would go HERE, on blockWant, before
        //            anything acts on it -- M4.2b-iv. It is absent because
        //            at ?levels=2 it is provably the identity: see
        //            common_d3_manage.wgsl's header and d3-amr.mjs's
        //            cascade21, which is the rule, written and tested.
        //   drain    restrict a dying tile onto its coarse cells. MUST come
        //            before coarsen: coarsen frees the slot and refine can
        //            hand that same slot straight out in the next pass, by
        //            which point the fine solution is gone.
        //   coarsen  free the slots. Only writes to freeList.
        //   refine   allocate. Only reads freeList -- see the manager's own
        //            header for the race that forces this split.
        //   fill     initialize the just-allocated tiles from the coarse
        //            field. MUST come after refine: there is no slot to fill
        //            until refine has handed one out.
        //   clear    drop the just-filled flags, in its own pass because
        //            clearing them inside fill is a race on the very test
        //            fill uses to select its work.
        blockPass(manageDecidePipe, wgB);
        gridPass(drainPipe, cp === 0 ? avgBGA : avgBGB);
        blockPass(manageCoarsenPipe, wgB);
        blockPass(manageRefinePipe, wgB);
        tilePass(fillPipe, interpBG[cp][0]);
        blockPass(manageClearPipe, wgS);
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
    const n = (await readI32(freeCountBuf, 16))[1];
    if (n > 0) {
      poolExhausted = n;
      live = false;
      playBtn.textContent = 'play';
      statusEl.textContent = `error: out of pool slots -- the manager was refused a tile ${n} time(s)`
        + ` at step ${step}. Refinement is geometry-forced, so this puts a coarse/fine seam`
        + ` through the body; raise ?slotHeadroom= (now ${SLOT_HEADROOM}) or refine less.`;
    }
    return poolExhausted;
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

  // Fine-level diagnostic: RMS speed over tile INTERIORS only (ring cells
  // are filled, not solved, and including them would blur exactly the
  // distinction this is for). Reported next to the coarse level's own RMS so
  // "is the fine level advancing at all" is answerable rather than inferred.
  const poolStaging = AMR ? device.createBuffer({
    size: Math.max(1, poolAlloc.activeSlots) * pool.tileCells * 4 * 4,
    usage: U.MAP_READ | U.COPY_DST,
  }) : null;
  async function readPoolStats() {
    if (!AMR) return null;
    const bytes = Math.max(1, poolAlloc.activeSlots) * pool.tileCells * 4 * 4;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(macPool, 0, poolStaging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await poolStaging.mapAsync(GPUMapMode.READ);
    const m = new Float32Array(poolStaging.getMappedRange()).slice();
    poolStaging.unmap();
    const FB = pool.FB, plane = Math.max(1, poolAlloc.activeSlots) * pool.tileCells;
    let sq = 0, n = 0, finite = true, rhoMin = Infinity, rhoMax = -Infinity;
    for (let s = 0; s < poolAlloc.activeSlots; s++) {
      for (let z = GHOST; z < GHOST + 2 * RB; z++) {
        for (let y = GHOST; y < GHOST + 2 * RB; y++) {
          for (let x = GHOST; x < GHOST + 2 * RB; x++) {
            const c = s * pool.tileCells + (z * FB + y) * FB + x;
            const rho = m[0 * plane + c], ux = m[1 * plane + c], uy = m[2 * plane + c], uz = m[3 * plane + c];
            if (!Number.isFinite(rho + ux + uy + uz)) { finite = false; continue; }
            sq += ux * ux + uy * uy + uz * uz; n++;
            if (rho < rhoMin) rhoMin = rho;
            if (rho > rhoMax) rhoMax = rho;
          }
        }
      }
    }
    return { step, rms: Math.sqrt(sq / Math.max(n, 1)), cells: n, finite, rhoMin, rhoMax };
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
  // Per level, and the top-level fields still report LEVEL 1 so every
  // existing caller reads what it always read (tools/validate-d3-invariants.js
  // asserts on `inUse`, `free` and `bbox` directly). `byLevel` is the new
  // surface; M5.6 is where the tools move onto it.
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
    // agree with planeDims()/fs_main in shaders/d3_render_slice.wgsl, which
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
    readInterfaceDiag, readFluxAcc,
    debugStepSync,
  };

  let lastStatus = 0;
  async function frame() {
    const enc = device.createCommandEncoder();
    if (live) encodeSteps(enc, STEPS_PER_FRAME);
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
        + `${AXIS_NAMES[axis]}-slice ${slice}   ${live ? 'running' : 'paused'}`;
    }
    requestAnimationFrame(() => frame().catch(e => reportFatal(statusEl, e)));
  }
  frame().catch(e => reportFatal(statusEl, e));
}

init().catch(e => reportFatal(statusEl, e));
