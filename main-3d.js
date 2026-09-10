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
//   ?reflux=1    OPT-IN coarse/fine interface flux correction (M4). Makes
//                the interface exactly conservative in mass and momentum,
//                and on a seam with no convex corner (?refine=slab) halves
//                the field error. On one WITH a corner (?refine=box, and
//                any body-fitted shell) it is much WORSE than leaving it
//                off. Not the default for that reason -- plans/3D.md M4.
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
import { makePool, refineWhere, refineNearBody, storageRatio, GHOST } from './d3-amr.mjs';

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
    'levels', 'rb', 'refine', 'margin', 'boxfrac', 'dcpre', 'reflux']);
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
  const AMR = LEVELS >= 2;
  let pool = null, poolAlloc = null;
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
    if (mode === 'all') {
      poolAlloc = refineWhere(pool, () => true);
    } else if (mode === 'box') {
      // A centred cube. For a scenario with an ANALYTIC answer (beltrami)
      // this is the whole point: a refined region in the middle of a flow
      // whose exact solution is known, so any damage the coarse/fine
      // interface does shows up directly as a field error.
      const frac = numParam('boxfrac', 0.5);
      const lo = [NX, NY, NZ].map(n => n * (1 - frac) / 2);
      const hi = [NX, NY, NZ].map(n => n * (1 + frac) / 2);
      poolAlloc = refineWhere(pool, ({ mid }) => mid.every((c, i) => c >= lo[i] && c < hi[i]));
    } else if (mode === 'slab') {
      // A refined SLAB spanning the full domain in y and z: periodic in
      // both, so the seam is two FLAT faces with no edge and no corner
      // anywhere. That is not a convenience geometry, it is the control for
      // M4's flux correction -- see plans/3D.md. Where the seam is flat the
      // fine channels crossing it tile the coarse one exactly; at a convex
      // edge or corner they provably do not, and this separates the two.
      const frac = numParam('boxfrac', 0.5);
      const lo = NX * (1 - frac) / 2, hi = NX * (1 + frac) / 2;
      poolAlloc = refineWhere(pool, ({ mid }) => mid[0] >= lo && mid[0] < hi);
    } else if (mode === 'body') {
      if (!params.body) throw new Error('?refine=body: this scenario has no body');
      const sh = params.body.shape, bx = params.body.x;
      const margin = numParam('margin', 2);
      // Sphere-only for now, which is what M3's validation needs; a general
      // SDF here would have to mirror d3-body.mjs's rotation handling and
      // that belongs with dynamic refinement in M4.
      poolAlloc = refineNearBody(pool, (q) => Math.hypot(q[0] - bx[0], q[1] - bx[1], q[2] - bx[2]) - sh.a, margin);
    } else {
      throw new Error(`?refine=${mode}: expected all, box, slab or body`);
    }
  }



  const fBytes = NCELLS * Q * 4;
  // The POOL, not the dense grid, is the largest binding once AMR is on: a
  // tile stores (FB/RB)^3 cells per coarse cell covered, which is 27x at
  // RB=4 (plans/3D.md sec 2.1's table). This has to be known BEFORE
  // requestDevice, because a device is created with the SPEC MINIMUM limits
  // unless asked otherwise -- and a 227 MiB pool binding on a device that
  // only asked for 128 MiB fails validation at buffer creation, which
  // presents as a fine level that silently never runs. It did exactly that
  // during development.
  const poolBytes = AMR ? Math.max(1, poolAlloc.activeSlots) * pool.tileCells * Q * 4 : 0;
  const needBytes = Math.max(fBytes, poolBytes);
  const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  if (needBytes > limit) {
    const what = poolBytes > fBytes
      ? `an L1 pool of ${poolAlloc.activeSlots} tiles x ${pool.tileCells} cells`
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
  let fPoolA = null, fPoolB = null, macPool = null, blockSlotBuf = null, slotToBlockBuf = null;
  let fluxAccBuf = null;
  if (AMR) {
    const slots = Math.max(1, poolAlloc.activeSlots);
    const poolCells = slots * pool.tileCells;
    fPoolA = device.createBuffer({ size: poolBytes, usage: U.STORAGE });
    fPoolB = device.createBuffer({ size: poolBytes, usage: U.STORAGE });
    macPool = device.createBuffer({ size: poolCells * 4 * 4, usage: U.STORAGE | U.COPY_SRC });
    blockSlotBuf = device.createBuffer({ size: pool.nBlocks * 4, usage: U.STORAGE | U.COPY_DST });
    slotToBlockBuf = device.createBuffer({ size: slots * 4, usage: U.STORAGE | U.COPY_DST });
    device.queue.writeBuffer(blockSlotBuf, 0, poolAlloc.blockSlot);
    device.queue.writeBuffer(slotToBlockBuf, 0, poolAlloc.slotToBlock.slice(0, slots));
    // Per coarse cell: the mass and momentum the FINE solver moved across
    // the seam this macro step, in coarse-cell units. Written by the flux
    // pass on substep A and added to on substep B; consumed by the reflux
    // pass. Only seam-adjacent cells are ever written OR read (both passes
    // share one reachability predicate), so it is deliberately not cleared.
    fluxAccBuf = device.createBuffer({ size: NCELLS * 4 * 4, usage: U.STORAGE | U.COPY_SRC });
  }

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
  const forceBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
    { binding: 0, resource: { buffer: fA } }, { binding: 1, resource: { buffer: bodyBuf } }, { binding: 2, resource: { buffer: forceBuf } }]});
  const forceBGB = device.createBindGroup({ layout: forceBGL, entries: [
    { binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: bodyBuf } }, { binding: 2, resource: { buffer: forceBuf } }]});
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
  // 1 puts the grid transfers back on the PRE-collision Dupuis-Chopard
  // factor, which is wrong for this solver's post-collision buffers and was
  // the M3 interface bug -- kept switchable so the defect can be measured
  // in the shipped build (shaders/common_d3_pool.wgsl derives both, and
  // tools/analyze-d3-interface.js is what drives the comparison).
  const DC_PRE = urlParams.get('dcpre') === '1' ? 1 : 0;
  let interpGhostPipe = null, interpFullPipe = null, step1Pipe = null, avgPipe = null;
  let interpBG = null, step1BG_AB = null, step1BG_BA = null, avgBGA = null, avgBGB = null;
  let fluxPipeSet = null, fluxPipeAdd = null, refluxPipe = null;
  let fluxBGA = null, fluxBGB = null, refluxBG = null;

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
  const REFLUX = (() => {
    if (!AMR) return 0;
    if (urlParams.get('reflux') !== '1') return 0;
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
    ]});
    const step1BGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]});
    const avgBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
      OMEGA_FINE: 1 / TAU_FINE,
      FORCE_X: params.force[0], FORCE_Y: params.force[1], FORCE_Z: params.force[2],
      HAS_BODY, USE_BOUNCEBACK, CHI_EPS,
      SPONGE_W: sponge.width, SPONGE_UX: sponge.u[0], SPONGE_UY: sponge.u[1], SPONGE_UZ: sponge.u[2],
    });
    avgPipe = await mk(avgBGL, avgModule, { ...poolConst, TAU_COARSE, DC_PRE });

    // interp needs BOTH coarse states (t and t+dt, which are simply the two
    // ping-pong buffers once the coarse step runs first) and writes whichever
    // pool buffer the next substep will READ -- so four bind groups:
    // coarse parity x pool target. interpBG[coarseParity][poolTarget].
    const mkInterp = (c0, c1, dst) => device.createBindGroup({ layout: interpBGL, entries: [
      { binding: 0, resource: { buffer: c0 } }, { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: c1 } }]});
    interpBG = [
      [mkInterp(fA, fB, fPoolA), mkInterp(fA, fB, fPoolB)],   // coarse t in fA
      [mkInterp(fB, fA, fPoolA), mkInterp(fB, fA, fPoolB)],   // coarse t in fB
    ];
    const mkStep = (a, b) => device.createBindGroup({ layout: step1BGL, entries: [
      { binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: macPool } }, { binding: 3, resource: { buffer: bodyBuf } },
      { binding: 4, resource: { buffer: slotToBlockBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
    step1BG_AB = mkStep(fPoolA, fPoolB);
    step1BG_BA = mkStep(fPoolB, fPoolA);
    // average writes the coarse buffer the COARSE step just wrote, so it
    // also needs one per parity.
    const mkAvg = (dst) => device.createBindGroup({ layout: avgBGL, entries: [
      { binding: 0, resource: { buffer: fPoolA } }, { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: mac } }]});
    avgBGA = mkAvg(fA);
    avgBGB = mkAvg(fB);

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
  let live = urlParams.get('live') !== '0';
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
    }
    device.queue.submit([enc.finish()]);
    // initEq READ mac and did not write it, so mac still holds the seed --
    // which is exactly the field the renderer should show at step 0.
    step = 0; useB = false;
  }

  // Pool dispatch shapes. The slot is folded into z because 3D has no
  // fourth dispatch dimension (plans/3D.md sec 2.4).
  const tileDisp = AMR ? [Math.ceil(pool.FB / 4), Math.ceil(pool.FB / 4), (pool.FB / 4) * Math.max(1, poolAlloc.activeSlots)] : null;
  const avgDisp = AMR ? (() => {
    const per = Math.ceil(RB / 4);
    return [per, per, per * Math.max(1, poolAlloc.activeSlots)];
  })() : null;

  // ORDER PER MACRO-STEP: zero -> force -> physics -> step.
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
        const pp = enc.beginComputePass();
        pp.setPipeline(physPipe); pp.setBindGroup(0, physicsBG); pp.dispatchWorkgroups(1); pp.end();
      }
      // --- AMR, N=2 (S_Advance) --------------------------------------------
      //
      // Both levels start the macro-step at time t. The order is:
      //
      //   L0 x1    the coarse step, taking L0 to t + dt. FIRST, so that the
      //            parent's state at BOTH ends of the step is available to
      //            the ring interpolation below -- t in the buffer it read,
      //            t + dt in the one it wrote.
      //   interp   fill L1's ring from the parent at time t
      //   L1 A     first fine substep, to t + dt/2
      //   interp   refill L1's ring from the parent at t + dt/2, blended
      //            between the two coarse states. Doing this ONCE per parent
      //            step instead leaves substep B half a coarse step stale at
      //            the seam, which is a first-order error that accumulates
      //            linearly -- see common_d3_amr_interp.wgsl's TIME_BLEND.
      //   L1 B     second fine substep, to t + dt
      //   average  restrict L1 onto L0 in the refined region, both at t + dt
      //   reflux   correct the unrefined coarse cells for the difference
      //            between the flux the COARSE step moved across the seam
      //            and the flux the FINE substeps actually moved (M4). Two
      //            `flux` passes measure the latter, one per substep,
      //            interleaved above because each reads the pool buffer its
      //            substep streams. Without this the interface is not
      //            conservative and the error is first order in dx --
      //            shaders/common_d3_amr_reflux.wgsl has the accounting.
      //
      // The coarse step does redundant work under the refined region (its
      // result is overwritten by average); that is the same trade the 2D
      // solver makes, and avoiding it would need a per-cell refined-mask
      // test in the hottest kernel there is.
      const cp = useB ? 1 : 0;    // which buffer holds the coarse state at t
      const p = enc.beginComputePass();
      p.setPipeline(stepPipe);
      p.setBindGroup(0, useB ? bgBA : bgAB);
      p.dispatchWorkgroups(disp[0], disp[1], disp[2]);
      p.end();

      if (AMR) {
        // ONE ring refresh per parent step, from the parent at time t, then
        // both fine substeps. Substep B's ring is what substep A advanced --
        // the GHOST=2 self-advance (see common_d3_amr_step1.wgsl).
        //
        // A second refresh at the half step, time-blended between the
        // parent's t and t+dt states, WAS built and measured, on the theory
        // that substep B was seeing stale interface data. It made the seam
        // error slightly WORSE (nearOut 3.24e-2 -> 3.93e-2 at t=64 on the
        // Beltrami box case), so staleness is not what the seam error is,
        // and the extra pass is not carried. The TIME_BLEND override stays
        // in the shader, defaulted to a no-op, because it is the natural
        // knob to re-try against a flux-corrected interface.
        const ip = enc.beginComputePass();
        ip.setPipeline(interpGhostPipe);
        ip.setBindGroup(0, interpBG[cp][0]);
        ip.dispatchWorkgroups(tileDisp[0], tileDisp[1], tileDisp[2]);
        ip.end();
        // M4 flux measurement, interleaved with the substeps rather than
        // deferred: each pass reads the pool buffer its substep STREAMS, and
        // substep B overwrites fPoolA, so the A measurement cannot wait.
        const flux = (pipe, bg) => {
          const fp = enc.beginComputePass();
          fp.setPipeline(pipe); fp.setBindGroup(0, bg);
          fp.dispatchWorkgroups(disp[0], disp[1], disp[2]);
          fp.end();
        };
        if (REFLUX) flux(fluxPipeSet, fluxBGA);       // writes; also the clear
        const substep = (bg) => {
          const sp = enc.beginComputePass();
          sp.setPipeline(step1Pipe); sp.setBindGroup(0, bg);
          sp.dispatchWorkgroups(tileDisp[0], tileDisp[1], tileDisp[2]);
          sp.end();
        };
        substep(step1BG_AB);
        if (REFLUX) flux(fluxPipeAdd, fluxBGB);
        substep(step1BG_BA);
        const ap = enc.beginComputePass();
        ap.setPipeline(avgPipe);
        // The coarse step just wrote the OTHER buffer, which is where the
        // restriction has to land.
        ap.setBindGroup(0, useB ? avgBGA : avgBGB);
        ap.dispatchWorkgroups(avgDisp[0], avgDisp[1], avgDisp[2]);
        ap.end();
        // Reflux LAST: it corrects the unrefined coarse cells against the
        // fine traffic just measured, and it reads the coarse field at t
        // (still intact -- the step wrote the other buffer) alongside the
        // t + dt field it corrects.
        if (REFLUX) {
          const rp = enc.beginComputePass();
          rp.setPipeline(refluxPipe); rp.setBindGroup(0, refluxBG[cp]);
          rp.dispatchWorkgroups(disp[0], disp[1], disp[2]);
          rp.end();
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
  async function debugStepSync(n) {
    live = false;
    playBtn.textContent = 'play';
    let done = 0;
    while (done < n) {
      const k = Math.min(SYNC_CHUNK, n - done);
      const enc = device.createCommandEncoder();
      encodeSteps(enc, k);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      done += k;
    }
    return { step };
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

  // The canvas box is given the SLICE's aspect ratio, so a sphere renders
  // round. The three axes can have very different extents (the sphere
  // scenario's domain is 192x128x128) and a fixed square canvas stretched
  // the x-long slices into ellipses -- which reads as a geometry bug on a
  // page whose entire job is showing geometry.
  function planeExtent(a) {
    if (a === 0) return [NY, NZ];
    if (a === 1) return [NZ, NX];
    return [NX, NY];
  }
  function resize() {
    const [pw, ph] = planeExtent(axis);
    canvas.style.aspectRatio = `${pw} / ${ph}`;
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
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
        tileCells: pool.tileCells,
        storageRatio: storageRatio(pool),
        tauCoarse: TAU_COARSE, tauFine: TAU_FINE, reflux: REFLUX,
      } : {}),
    }),
    readSubsampled, readDuctProfile, readStats, readBody, readPoolStats,
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
      const t = scenarioName === 'beltrami' ? `  t/td=${(step / params.td).toFixed(2)}`
        : scenarioName === 'duct' ? `  t/settle=${(step / params.settle).toFixed(2)}`
        : scenarioName === 'sphere' ? `  t/(D/U)=${(step / params.convective).toFixed(2)}`
          : params.re ? `  Re=${params.re.toFixed(0)}` : '';
      const amrTxt = AMR ? `  L1 ${poolAlloc.activeSlots}/${pool.nBlocks} blocks (RB=${RB}, FB=${pool.FB})` : '';
      statusEl.textContent = `${scenarioName}  D3Q${Q}  ${NX}x${NY}x${NZ}  step ${step}${t}${amrTxt}\n`
        + `${AXIS_NAMES[axis]}-slice ${slice}   ${live ? 'running' : 'paused'}`;
    }
    requestAnimationFrame(() => frame().catch(e => reportFatal(statusEl, e)));
  }
  frame().catch(e => reportFatal(statusEl, e));
}

init().catch(e => reportFatal(statusEl, e));
