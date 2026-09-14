// Scenario definitions and closed-form reference solutions for the dense
// 3D solver (index-3d.html / main-3d.js), plans/3D.md M1.
//
// Shared by the page and by the validation tools, the same way
// card-params.mjs is: the page uses these to build its initial condition
// and to derive tau/forcing from physical parameters, and
// tools/lib/d3-metrics.js uses them to score the result. One definition,
// not two that drift.
//
// WHY THESE TWO GATES, and not the ones plans/3D.md M1 originally named.
// M1 asked for a 3D Poiseuille duct plus "3D Taylor-Green vortex at
// Re=1600". The duct is here as specified. The Re=1600 TGV is NOT an
// analytic solution -- it is a DNS benchmark scored against digitized
// dissipation-rate curves (Brachet et al.) at 256^3-512^3, which is a
// literature-data and resolution problem, not a solver-correctness gate.
// What M1 actually needs to know is "does this solver realize the
// viscosity and the boundary conditions it claims", and for that a
// **Beltrami (ABC) flow** is a strictly better instrument and is exactly
// analytic in 3D:
//
//   For a Beltrami field, curl(u) = k*u, so u x omega = 0 and the
//   nonlinear term (u.grad)u collapses to grad(|u|^2/2) -- a pure
//   gradient, absorbed by the pressure. Navier-Stokes reduces to
//   du/dt = nu*lap(u), and since lap(u) = -k^2 u for this field,
//
//       u(x, t) = u0(x) * exp(-nu * k^2 * t)                (EXACT)
//
//   with p = p0 - |u|^2/2. That is a genuinely three-dimensional,
//   time-dependent, closed-form solution that exercises all three axes
//   symmetrically and pins the REALIZED viscosity to a scalar decay rate.
//   The 2D TGV harness this project already has (tools/lib/tgv-metrics.js)
//   is the same idea; this is its 3D counterpart, and unlike the 3D TGV it
//   does not degrade into "compare against a curve someone published".
//
// The 3D TGV is still here as a scenario, because it is the right
// *initial condition* for the eventual Re=1600 comparison and because it
// is what a viewer wants to look at. It reports energy and dissipation
// and is deliberately NOT a PASS/FAIL gate -- see benchmarks/d3.json.
//
// Same browser+Node dual-consumption pattern as lattice-3d.mjs: imported
// natively by the page, dynamically imported by the CommonJS tools.

import { SHAPE, makeBodyState, qFromAxisAngle, qMul } from './d3-body.mjs';

export const CS2 = 1 / 3;

export function nuFromTau(tau) { return CS2 * (tau - 0.5); }
export function tauFromNu(nu) { return nu / CS2 + 0.5; }

// --- duct: square-duct Poiseuille flow ------------------------------------
//
// Periodic in x, no-slip walls on BOTH y and z, driven by a uniform body
// force G (Guo forcing). Steady state has the classical series solution
// (White, Viscous Fluid Flow, sec 3-3.3) for the duct |y| <= a, |z| <= b
// with lap(u) = -G/mu:
//
//   u(y,z) = (16 G a^2)/(mu pi^3)
//            * SUM over odd i of  (-1)^((i-1)/2) / i^3
//              * [1 - cosh(i pi z / 2a) / cosh(i pi b / 2a)]
//              * cos(i pi y / 2a)
//
// Note which variable goes where: `cos` with the half-width-a axis, `cosh`
// with the half-width-b one. They are interchangeable only because this
// project's duct is square (a = b); getting it backwards on a non-square
// duct is a silent, plausible-looking error, so tools/test-d3-scenarios.js
// checks the series against lap(u) = -G/nu by finite differences rather
// than against a transcription of the same formula.
//
// WALL POSITION. shaders/common_walls.wgsl's convention (inherited by
// common_d3_step.wgsl) puts the physical wall halfway between the last
// fluid cell and the first nonexistent one, so N fluid cells span exactly
// N lattice units wall to wall: half-width a = N/2, and fluid cell j sits
// at centered coordinate (j + 0.5) - N/2. That half-cell offset is the
// whole reason a "which cell is the wall on" bug shows up as a few-percent
// profile error rather than as an obvious one.

// Centered wall-normal coordinate of fluid cell j in an N-cell span.
export function ductCoord(j, N) { return (j + 0.5) - N / 2; }

// Series terms. 24 odd terms is far past convergence (the i^3 denominator
// plus cosh growth makes term 24 O(1e-6) of term 1) and costs nothing --
// this is evaluated once per validation run, not per step.
const DUCT_TERMS = 24;

// u_x at centered coordinates (y, z) for a duct of half-widths a (the
// cos axis) and b (the cosh axis), driving force G, kinematic viscosity
// nu, rho = 1 (so mu = nu).
export function ductVelocityAt(y, z, a, b, G, nu) {
  let s = 0;
  for (let n = 0; n < DUCT_TERMS; n++) {
    const i = 2 * n + 1;
    const sgn = (n % 2 === 0) ? 1 : -1;          // (-1)^((i-1)/2)
    const kz = (i * Math.PI * z) / (2 * a);
    const kb = (i * Math.PI * b) / (2 * a);
    const ky = (i * Math.PI * y) / (2 * a);
    // cosh(kz)/cosh(kb) written as an exponential ratio: at N=128 and
    // i=23, cosh(kb) overflows f64 before the ratio does, and the naive
    // form silently yields NaN in exactly the large-N cases this is meant
    // to validate. cosh(u)/cosh(v) = (e^(u-v) + e^(-u-v)) / (1 + e^(-2v)).
    const ratio = (Math.exp(kz - kb) + Math.exp(-kz - kb)) / (1 + Math.exp(-2 * kb));
    s += (sgn / (i * i * i)) * (1 - ratio) * Math.cos(ky);
  }
  return (16 * G * a * a) / (nu * Math.PI ** 3) * s;
}

// Centerline (peak) velocity coefficient for a SQUARE duct: u_max =
// 0.2947 * G a^2 / nu. Computed rather than quoted so it tracks
// DUCT_TERMS, and asserted against the textbook 0.2947 by the unit test.
export function ductPeakCoeff() {
  return ductVelocityAt(0, 0, 1, 1, 1, 1);
}

// Body force G that produces a target peak velocity in a square duct of
// half-width a. This is how a scenario is specified: pick the Mach number
// you want, get the force that delivers it.
export function ductForceForPeak(uPeak, a, nu) {
  return (uPeak * nu) / (ductPeakCoeff() * a * a);
}

// Analytic steady u_x over the whole N x N cross-section, indexed
// [z * N + y] to match the page's readDuctProfile().
export function ductProfile(N, G, nu) {
  const a = N / 2;
  const out = new Float64Array(N * N);
  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      out[z * N + y] = ductVelocityAt(ductCoord(y, N), ductCoord(z, N), a, a, G, nu);
    }
  }
  return out;
}

// Slowest-decaying transient mode's time constant, in lattice steps: the
// (1,1) duct mode has lap eigenvalue -(pi/2a)^2 - (pi/2a)^2, so a
// quiescent start settles as exp(-t / tSettle). A run needs several of
// these, and expressing the run length this way makes it resolution- and
// viscosity-independent (the same idea as tgv.json's checkpoints being
// multiples of td).
export function ductSettleTime(N, nu) {
  const a = N / 2;
  return 1 / (nu * (Math.PI ** 2) / (2 * a * a));
}

// --- beltrami: decaying ABC flow ------------------------------------------
//
//   u = A sin(kz) + C cos(ky)
//   v = B sin(kx) + A cos(kz)
//   w = C sin(ky) + B cos(kx)
//
// curl(u) = k*u for any A, B, C (checked by finite differences in
// tools/test-d3-scenarios.js, not asserted here), so the flow is Beltrami
// and decays as exp(-nu k^2 t) with its spatial shape exactly preserved.
// k = 2*PI/N puts exactly one wavelength in the periodic box.
//
// A = B = C = u0 by default: the symmetric case, which is the point --
// an asymmetry in the solver between the three axes (a transposed index, a
// z-term dropped from feq, a wrong ez entry) breaks shape preservation,
// and shape preservation is what the field-L2 check measures.

export function beltramiK(N) { return 2 * Math.PI / N; }
export function beltramiDecayTime(N, nu) {
  const k = beltramiK(N);
  return 1 / (nu * k * k);
}

// Velocity at integer lattice site (x,y,z) and time t (lattice steps).
export function beltramiVelocityAt(x, y, z, N, u0, nu, t) {
  const k = beltramiK(N);
  const decay = Math.exp(-nu * k * k * t);
  return [
    u0 * (Math.sin(k * z) + Math.cos(k * y)) * decay,
    u0 * (Math.sin(k * x) + Math.cos(k * z)) * decay,
    u0 * (Math.sin(k * y) + Math.cos(k * x)) * decay,
  ];
}

// --- tgv: 3D Taylor-Green vortex (initial condition only) -----------------
//
// NOT a closed-form solution -- the nonlinear term does not vanish, which
// is exactly why it is the standard 3D transition benchmark. Present as an
// initial condition and a reporting scenario; see this file's header.
export function tgvVelocityAt(x, y, z, N, u0) {
  const k = 2 * Math.PI / N;
  return [
    u0 * Math.sin(k * x) * Math.cos(k * y) * Math.cos(k * z),
    -u0 * Math.cos(k * x) * Math.sin(k * y) * Math.cos(k * z),
    0,
  ];
}

// --- scenario table -------------------------------------------------------
//
// `walls` names the axes with no-slip boundaries; every other axis is
// periodic. `force` is the uniform body-force density (Guo). `macro`
// builds the initial [rho, ux, uy, uz] field the page uploads and the GPU
// turns into an equilibrium `f` -- see main-3d.js. Density carries the
// incompressible pressure p = p0 - |u|^2/2 as rho = 1 - |u|^2/(2 cs2),
// which is the O(Ma^2) correction an equilibrium seed otherwise starts
// with an error in.

// Non-cubic form. M2's sphere scenario needs a long, narrow box; the M1
// scenarios are all cubes and go through seedMacro() below.
// Deterministic PRNG, so a perturbed run is still reproducible for
// regression use. Same generator (mulberry32) and same default seed as
// main-cylinder.js, which is not an accident: the two projects' shedding
// onsets should be seeded the same way if their numbers are ever compared.
function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A SMALL TRANSVERSE PERTURBATION ON THE INITIAL CONDITION, and it exists
// because the 2D project already paid for its absence (main-cylinder.js's
// PERTURB, whose comment is worth reading in full).
//
// A uniform freestream past a centred body is EXACTLY symmetric, so shedding
// onset has to grow from whatever asymmetry round-off provides -- and that
// seed is SMALLER AND SLOWER-GROWING AT HIGHER RESOLUTION, because a finer
// grid has less discretization error to seed from. In 2D that made Cd and
// shedding strength look like they SHRANK with resolution, when really the
// finer runs had not saturated inside the same step budget. The trap is
// worse in 3D, not better: a sphere's wake must also select an azimuthal
// PLANE, and there is nothing in a symmetric initial condition to select it
// with.
//
// Defaults to 0 everywhere, so every existing case is bit-identical and the
// steady-regime gates are untouched. Opt in per case.
export function seedMacro3Perturbed(dims, velAt, amp, seed, axes) {
  const out = seedMacro3(dims, velAt);
  if (!amp) return out;
  const [NX, NY, NZ] = dims;
  const rng = mulberry32(seed | 0);
  const a = axes || [1, 2];            // transverse components by default
  for (let c = 0; c < NX * NY * NZ; c++) {
    for (const k of a) out[4 * c + 1 + k] += amp * (rng() * 2 - 1);
  }
  return out;
}

function seedMacro3(dims, velAt) {
  const [NX, NY, NZ] = dims;
  const out = new Float32Array(4 * NX * NY * NZ);
  for (let z = 0; z < NZ; z++) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const c = (z * NY + y) * NX + x;
        const [ux, uy, uz] = velAt(x, y, z);
        out[4 * c + 0] = 1 - (ux * ux + uy * uy + uz * uz) / (2 * CS2);
        out[4 * c + 1] = ux;
        out[4 * c + 2] = uy;
        out[4 * c + 3] = uz;
      }
    }
  }
  return out;
}

function seedMacro(N, velAt) {
  const n3 = N * N * N;
  const out = new Float32Array(4 * n3);
  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const c = (z * N + y) * N + x;
        const [ux, uy, uz] = velAt(x, y, z);
        out[4 * c + 0] = 1 - (ux * ux + uy * uy + uz * uz) / (2 * CS2);
        out[4 * c + 1] = ux;
        out[4 * c + 2] = uy;
        out[4 * c + 3] = uz;
      }
    }
  }
  return out;
}

// --- sphere: flow past a pinned sphere (the M2 gate) ----------------------
//
// Uniform crossflow along +x past a pinned sphere, with an ALBC sponge on
// every face relaxing toward the freestream -- the 3D counterpart of this
// project's 2D cylinder harness, and scored the same way: a time-averaged
// drag coefficient against literature.
//
//   Cd = Fx / (1/2 rho U^2 A),   A = pi R^2
//
// The reference is the Schiller-Naumann correlation,
//   Cd = 24/Re * (1 + 0.15 Re^0.687),
// which is the standard fit for a sphere below Re ~ 1000 and is quoted as
// good to a few percent. Below Re ~ 210 the wake is steady and
// axisymmetric, so unlike the 2D cylinder there is no shedding to
// time-average -- Cd simply settles, which makes this a cheaper and
// sharper gate than the cylinder's Cd/St pair.
export function schillerNaumann(re) {
  return (24 / re) * (1 + 0.15 * Math.pow(re, 0.687));
}

// Domain shaped in units of the sphere diameter: long enough downstream for
// the wake, wide enough that blockage is small. At the defaults below the
// blockage ratio is pi*(D/2)^2 / (span*D)^2 = 1.2%, which moves Cd by well
// under the tolerance.
const SPHERE_DOMAIN = { length: 12, span: 8, upstream: 3 };

export const SCENARIOS = {
  // Quiescent start, walls on y and z, driven to a steady analytic profile.
  // The initial condition is INDEPENDENT of the reference solution here
  // (u = 0 everywhere), so this gate cannot be passed by a shared mistake
  // between the seed and the check -- unlike beltrami, whose seed IS its
  // reference at t = 0.
  duct: {
    name: 'duct',
    defaults: { n: 48, tau: 0.8, u0: 0.05 },
    walls: ['y', 'z'],
    derive: ({ n, tau, u0 }) => {
      const nu = nuFromTau(tau);
      const a = n / 2;
      const G = ductForceForPeak(u0, a, nu);
      return { nu, a, G, force: [G, 0, 0], settle: ductSettleTime(n, nu), uPeak: u0 };
    },
    macro: (dims) => seedMacro3(dims, () => [0, 0, 0]),
  },

  beltrami: {
    name: 'beltrami',
    defaults: { n: 48, tau: 0.8, u0: 0.04 },
    walls: [],
    derive: ({ n, tau, u0 }) => {
      const nu = nuFromTau(tau);
      return { nu, k: beltramiK(n), td: beltramiDecayTime(n, nu), force: [0, 0, 0] };
    },
    macro: (dims, { u0, nu }) => seedMacro3(dims, (x, y, z) => beltramiVelocityAt(x, y, z, dims[0], u0, nu, 0)),
  },

  tgv: {
    name: 'tgv',
    defaults: { n: 64, tau: 0.8, u0: 0.04 },
    walls: [],
    derive: ({ n, tau, u0 }) => {
      const nu = nuFromTau(tau);
      const L = n / (2 * Math.PI);           // the benchmark's length scale
      return { nu, L, re: u0 * L / nu, force: [0, 0, 0] };
    },
    macro: (dims, { u0 }) => seedMacro3(dims, (x, y, z) => tgvVelocityAt(x, y, z, dims[0], u0)),
  },
};

SCENARIOS.sphere = {
  name: 'sphere',
  // `n` is the sphere DIAMETER in cells here, not the domain edge -- the
  // resolution that matters for a body is how many cells span it.
  // `perturb` is a fraction of u0 and defaults to 0, so every steady case is
  // bit-identical to before it existed. It is opt-in for the SHEDDING cases
  // -- see seedMacro3Perturbed for why a symmetric initial condition is a
  // trap rather than a neutral choice.
  defaults: { n: 16, tau: null, u0: 0.05, re: 100, perturb: 0, seed: 12345 },
  walls: [],
  dims: ({ n }) => [
    Math.round(SPHERE_DOMAIN.length * n),
    Math.round(SPHERE_DOMAIN.span * n),
    Math.round(SPHERE_DOMAIN.span * n),
  ],
  derive: (p) => {
    const { n, u0, re } = p;
    // tau follows from the target Re, not the other way round: Re is the
    // physical quantity a benchmark is stated in.
    const nu = u0 * n / re;
    const tau = tauFromNu(nu);
    const d = SCENARIOS.sphere.dims(p);
    const R = n / 2;
    const centre = [SPHERE_DOMAIN.upstream * n, d[1] / 2, d[2] / 2];
    const body = makeBodyState({ shape: { kind: SHAPE.SPHERE, a: R }, x: centre });
    return {
      nu, tau, re, R, D: n, dims: d, body, pinned: true,
      area: Math.PI * R * R,
      blockage: Math.PI * R * R / (d[1] * d[2]),
      sponge: { width: Math.max(6, Math.round(n / 2)), u: [u0, 0, 0] },
      force: [0, 0, 0],
      // One convective time D/U, the natural unit for how long a run needs.
      convective: n / u0,
      cdReference: schillerNaumann(re),
    };
  },
  // Started from the uniform freestream everywhere, including inside the
  // body -- the penalization drives the interior to the body's velocity
  // within a few hundred steps and starting from rest instead only adds an
  // acoustic transient to wait out.
  macro: (dims, p) => seedMacro3Perturbed(dims, () => [p.u0, 0, 0], p.perturb * p.u0, p.seed),
};

// --- fall: a body the FLUID actually moves (plans/3D.md M8.2) --------------
//
// THE FIRST 3D SCENARIO IN WHICH THE FLUID FORCE MOVES A BODY. Everything
// before it is pinned (`sphere`) or free with the coupling switched off
// (`drift`, `spin`, both `noFluidForce: true`), so the six-scalar force
// reduction, the 6-DOF integrator and the moving-body boundary condition
// have all been validated SEPARATELY and never against each other. That gap
// is the whole reason this exists -- see plans/3D.md M8.
//
// A SPHERE FIRST, DELIBERATELY, even though the target is a plate: its drag
// law is known, this suite already measures it (sphere_cases, and their
// ~+12% staircase offset against Schiller-Naumann), so the terminal velocity
// is PREDICTABLE and the gate can be a number rather than a look.
//
// THE PARAMETERS ARE THE TERMINAL STATE, not the forcing. You say what
// terminal velocity and Reynolds number you want and the gravity follows,
// exactly as card-params.mjs derives G_LU from U_T rather than the other way
// round -- because the regime is the physical input and the acceleration is
// an implementation detail of reaching it. At terminal the fluid force
// balances the effective weight:
//
//     rho_b V g_eff = 1/2 rho U_T^2 Cd A      =>  g_eff = U_T^2 Cd A / (2 rho_b V)
//
// G IS AN ACCELERATION AND IT IS ALREADY BUOYANCY-CORRECTED, matching
// d3_physics.wgsl's GX/GY/GZ and the 2D page's G_EFF. The momentum-exchange
// force this solver measures does NOT include buoyancy -- gravity is never
// applied to the FLUID, so there is no hydrostatic gradient to produce it --
// so the correction has to live here rather than being double counted.
SCENARIOS.fall = {
  name: 'fall',
  // `n` is the sphere DIAMETER in cells, as in `sphere`.
  // `tow` > 0 replaces free fall with PRESCRIBED motion at that speed:
  // gravity off, force measured and never applied. It is the reference the
  // free fall is scored against (plans/3D.md M8.2b), and it is the same
  // body in the same domain with the same sponge -- which is the whole
  // point. Comparing a fall against Schiller-Naumann compares it against a
  // different discretization AND a different domain at once; comparing it
  // against a tow isolates the fall.
  // Three modes, and the pair of them is the point (plans/3D.md M8.2b):
  //   tow=U     the body MOVES at U through still fluid, force recorded and
  //             never applied. This is the moving-body coupling under test.
  //   stream=U  the body is PINNED in a freestream U -- the same relative
  //             motion in the other Galilean frame, and the path every
  //             validated sphere case in this suite already uses.
  //   neither   free fall.
  // Cd(tow) and Cd(stream) must AGREE, in the same domain with the same
  // blockage and the same sponge, because they are the same flow seen from
  // two frames. That isolates the moving-body coupling from resolution,
  // blockage and the drag correlation all at once -- none of which a
  // comparison against Schiller-Naumann can separate.
  //
  // BOTH AT ONCE IS THE THIRD MODE, AND IT TURNS THAT PAIR INTO A SWEEP
  // (plans/3D.md D1). `tow` and `stream` are independent signed speeds: the
  // body moves at `tow`, the fluid (initial condition AND sponge target) sits
  // at `stream`, and the only thing the flow can depend on is the DIFFERENCE
  //
  //     uRelSigned = stream - tow
  //
  // because a rigid translation of body and fluid together is a change of
  // inertial frame and nothing else. So holding that difference fixed and
  // sliding the SPLIT is a one-parameter family of the SAME physical problem,
  // whose pinned end (tow = 0) is the validated path this suite already uses
  // and whose towed end (stream = 0) is the moving-body coupling under test.
  // Any dependence on the split is a defect, measured with no reference value
  // and no second domain -- see tools/probe-d3-galilean.js.
  //
  // Two endpoints of that family have EXACT answers, which is what makes it
  // an instrument rather than a comparison:
  //   uRelSigned = 0 (tow == stream)   F is exactly zero, because a uniform
  //       feq(1, u) field is a fixed point of the whole scheme: the moving-
  //       wall bounce-back returns feq_i exactly (feq_i - feq_opp is the
  //       correction term, identically), and the momentum-exchange sum
  //       pairs i with opp(i) with equal link counts on a closed body.
  //   tow = 0                          the pinned leg, already gated.
  // Neither is a tolerance on measured data, so a run that misses them is
  // reporting a defect rather than a discretization.
  //
  // BOTH SPEEDS ARE SIGNED. A sweep at fixed relative speed U runs
  // tow = a, stream = a - U for a in [0, U], which keeps every velocity in
  // the domain bounded by U -- the split must not smuggle in a Mach number
  // change alongside the frame change.
  defaults: { n: 12, re: 100, u_t: 0.04, rho_b: 4, tow: 0, stream: 0, perturb: 0, seed: 12345 },
  walls: [],
  // Long in the fall direction and no longer than it needs to be across:
  // the body has to reach terminal velocity AND then travel far enough to
  // measure it, which is the domain's whole job here.
  dims: ({ n }) => [Math.round(16 * n), Math.round(6 * n), Math.round(6 * n)],
  derive: (p) => {
    const { n, re, u_t, rho_b } = p;
    const nu = u_t * n / re;
    const tau = tauFromNu(nu);
    const d = SCENARIOS.fall.dims(p);
    const R = n / 2;
    const V = (4 / 3) * Math.PI * R ** 3;
    const area = Math.PI * R * R;
    const cd = schillerNaumann(re);
    // The balance above, solved for the acceleration.
    const gEff = u_t * u_t * cd * area / (2 * rho_b * V);
    const shape = { kind: SHAPE.SPHERE, a: R };
    // SIGNED, and `!== 0` rather than `> 0`: a Galilean split runs one of
    // these NEGATIVE (see the header), and `> 0` would have read a body towed
    // in -x as a free fall -- gravity back on, force applied, a different
    // experiment reported under the same name.
    const towed = p.tow !== 0, streamed = p.stream !== 0;
    // `prescribed` is "somebody is moving on rails", which is what separates
    // the measurement legs from the free fall; `uRelSigned` is the only thing
    // the FLOW can depend on, and every arm below keys off one of those two
    // rather than off which knob happens to be set.
    const prescribed = towed || streamed;
    const uRelSigned = p.stream - p.tow;
    // WHERE THE BODY SITS IS THE SAME ARRANGEMENT IN BOTH FRAMES, and under
    // the moving window it stays that way (plans/3D.md M8.3). A streamed
    // body sits 2n from the INLET, so its own wake has the other 14n and is
    // not the inlet condition -- which is where the sphere scenario puts
    // one. A body that MOVES leaves its wake BEHIND it, so it sits 2n from
    // the FAR end instead: the mirror image, 2n of undisturbed fluid ahead
    // and 14n of room for the wake.
    //
    // That mirroring is what makes the Galilean pair a comparison of the
    // COUPLING rather than of two domains. Before the window it held only at
    // step 0 -- a towed body carried its 14n of wake room with it only until
    // it ran out, and M8.2b measured Cd still RISING at 8 D/U (0.951 -> 1.067
    // against the streamed 1.362) because the room behind it was still
    // growing while the measurement was being taken.
    //
    // WHICH END IS "AHEAD" IS THE BODY'S MOTION RELATIVE TO THE FLUID, not
    // which knob is set: a split leg has both set, and keying off `streamed`
    // would flip the domain around halfway through a sweep that is supposed
    // to change one thing. A free fall has no prescribed speed and falls
    // along +x, so it takes the same arm as a tow.
    const bodyRelSign = prescribed ? Math.sign(p.tow - p.stream) : 1;
    const x0 = bodyRelSign >= 0 ? d[0] - 2 * n : 2 * n;
    // `v` at construction, not a vx assignment afterwards: packBodyState
    // reads `s.v[0]`, so setting `body.vx` writes a field nothing packs and
    // the body silently stays put. (It did, for one measurement.)
    const body = makeBodyState({ shape, x: [x0, d[1] / 2, d[2] / 2], density: rho_b,
                                 v: [p.tow, 0, 0] });
    return {
      nu, tau, re, R, D: n, dims: d, body,
      // PINNED means the body does not move, which is "the fluid is on rails
      // and the body is not" -- NOT "a freestream was asked for". A split leg
      // has both and the body must move.
      pinned: streamed && !towed,
      // A towed run integrates nothing: constant velocity, force recorded.
      // A streamed run does not move the body at all.
      noFluidForce: towed, tow: p.tow, stream: p.stream,
      // The relative speed under test, whichever frame it is expressed in,
      // and the SIGNED version alongside it: Cd is normalized by the
      // magnitude, but which way the drag points is the sign, and a tool that
      // infers that from `tow > 0` gets a split leg backwards.
      uRel: prescribed ? Math.abs(uRelSigned) : null,
      uRelSigned: prescribed ? uRelSigned : null,
      area, cd, rho_b, u_t,
      // THE MOVING WINDOW, on the axis the body travels along, and ONLY
      // where the body actually moves: a pinned streamed body has nothing to
      // follow and the window would be an inert branch in the one case that
      // is supposed to be the control. plans/3D.md M8.3.
      //
      // x only. The sphere's lateral drift is ~0.1 cells over 5000 steps
      // against 6n of domain, so a y/z window would buy nothing and would
      // put a moving discontinuity -- the sponge band's edge -- into the
      // narrow direction for no reason. A tumbling plate is the case that
      // changes that answer, and it can say so itself.
      // Keyed on whether the BODY MOVES, not on which knob is set: a split
      // leg streams AND tows, and `streamed` would switch the window off
      // under a body that is still travelling.
      window: streamed && !towed ? [0, 0, 0] : [1, 0, 0],
      // FALLS ALONG +x, the long axis. Nothing about the solver prefers an
      // axis; the domain does.
      gravity: towed ? [0, 0, 0] : [gEff, 0, 0],
      // WHICH WAY IS DOWN, as a signed lattice axis, for the VIEW alone.
      //
      // The solver does not have an opinion and must not acquire one: this
      // is read only by main-3d.js's downTurn(), which rotates the slice
      // picture so this direction runs down the window. Without it a fall
      // along +x on the default z-slice tracks ACROSS the screen, and which
      // way depends on the leg -- a towed body advances while a streamed one
      // is pinned in a flow going the other way -- so the same physics in two
      // Galilean frames reads as two different pictures.
      //
      // +x on ALL THREE LEGS, deliberately. A streamed body is not falling,
      // but it is the SAME flow as the tow seen from the other frame
      // (plans/3D.md M8.2b), and a control that is drawn differently from
      // the thing it controls is a control nobody can read.
      down: [1, 0, 0],
      gEff,
      // Quiescent fluid, and a sponge that holds it there rather than at a
      // freestream -- this is a body falling through still fluid, not a body
      // held in a flow.
      // Still fluid for a fall or a tow; the freestream for a streamed run.
      // `p.stream` directly -- it is already 0 when unset, and the guard it
      // replaces read as if a freestream were conditional on something other
      // than its own value.
      sponge: { width: Math.max(6, Math.round(n / 2)), u: [p.stream, 0, 0] },
      force: [0, 0, 0],
      blockage: area / (d[1] * d[2]),
      // Time to terminal is ~U_T/g_eff; the natural unit for how long a run
      // needs, and the tools size their windows from it.
      tSettle: u_t / gEff,
      convective: n / u_t,
      // THE REFERENCE MUST BE KEYED TO THE RELATIVE SPEED THE LEG ACTUALLY
      // RUNS AT, not to `re`. `re` sets the VISCOSITY and, for a free fall,
      // is the Reynolds number of the target terminal velocity `u_t` -- which
      // is what g_eff is derived from above and is correct there. A prescribed
      // leg runs at `uRel`, and when that differs from `u_t` the flow's
      // Reynolds number is uRel * D / nu, a different number. Reported against
      // `re` it looked like a probe leg was +474% over Schiller-Naumann when
      // the honest figure was +365%: the wrong reference, not the wrong
      // measurement. No gate reads this (the prescribed cases score against
      // each other or against recorded values), but a log that lies is how a
      // number gets quoted later.
      cdReference: prescribed ? schillerNaumann(Math.abs(uRelSigned) * n / nu) : cd,
    };
  },
  macro: (dims, p) => seedMacro3Perturbed(dims, () => [p.stream, 0, 0],
                                          p.perturb * p.u_t, p.seed),
};

// --- card: THE TARGET. A free plate, falling and turning ------------------
//
// plans/3D.md M8: the 3D analog of the 2D project's falling card
// (`card-params.mjs`, Pesavento & Wang 2004 Fig. 2). `fall` proved the
// machinery on a SPHERE, whose drag law is known so its terminal velocity is
// a number a gate can check. A plate is what the machinery was built for, and
// it is a different problem in exactly one way that matters: **its attitude
// is a degree of freedom that feeds back into its own drag.**
//
// WHAT IS DERIVED AND WHAT IS CHOSEN, because the 2D module's whole point is
// that the regime is the input and the lattice constants follow:
//
//   n        CHORD in cells. The resolution knob; everything else is a ratio.
//   aspect   thickness / chord. 0.125 is the paper's e, and at n = 32 that is
//            their a = 32, b = 4 reference card scaled by two.
//   span     span / chord. 1.0 is a SQUARE plate, which is NOT the paper's
//            quasi-2D card -- see the span note below. Cheap, and the thing
//            you can actually run.
//   re       2 * u_t * (chord/2) / nu, THE PAPER'S CONVENTION keyed to the
//            chord, matching card-params.mjs's after its factor-of-2 fix.
//   i_star   the dimensionless moment of inertia that selects the branch;
//            rho_b follows from it, exactly as in 2D.
//   tilt     initial pitch, in radians, about the span axis. A plate dropped
//            EXACTLY broadside is a symmetric initial condition and has
//            nothing to fall over from but round-off -- the same trap
//            seedMacro3Perturbed's header describes for the sphere's wake
//            plane, and worse here because the symmetry is the BODY's.
//
// THE SPAN IS THE HONEST GAP. Pesavento & Wang's card is quasi-2D: a strip
// wide enough that the ends do not matter, so I* is per unit span and the
// tumbling is a two-dimensional bifurcation. A square plate is a DIFFERENT
// body -- its tip vortices are not a correction, they are a comparable
// effect -- so `span` is a knob and not a constant, and a comparison against
// the paper needs it large. It is stated here rather than discovered later.
//
// WHAT THIS ACTUALLY DOES TODAY, measured 2026-09-11 at n = 32, span = 1,
// aspect = 0.125, tilt = 0.15, dense (?levels=1), and NOT yet a gate:
//
//   Re = 200 (tau = 0.524):  STABLE to 43000 steps. Falls broadside at
//       v_x = 0.043 against the target u_t = 0.05, with the measured fluid
//       force -1.35 against a weight m*g_eff = 1.408 -- a 4% closure of the
//       free-fall balance, which is the first time anything in 3D has closed
//       it on a non-sphere. max|u| holds at 0.069 and rho within +-1.3%.
//       It does NOT tumble: broadside is the stable branch here.
//   Re = 500 (tau = 0.5096): the FLUID diverges at ~20000 steps, and the
//       body follows rather than causes it -- max|u| goes 0.074 -> 0.16 ->
//       0.21 -> 0.45 -> NaN over about 1800 steps while the plate is still
//       within 3 degrees of broadside, with rho opening to [0.88, 1.07].
//       Capping the rotation (?omax=0.004) does not prevent it, which is
//       what rules out the body dynamics.
//
// So the target's Re ~ 1100 is NOT reachable at this resolution, and the
// reason is the one M8.0 already names: tau = 0.5 + 6 u_t a / Re, so the
// stability margin is bought with RESOLUTION, not with anything else. At
// n = 64 and Re = 1100 that is tau = 0.5087 -- the 2D card's own shipped
// tau0 -- and the body must then sit on a refined level, where
// tauAtLevel doubles the margin per rung. AMR is not an optimization for
// this case; it is the stability mechanism.
//
// AND IT MEASURES AS ONE (M8.7): on a controlled A/B the falling SPHERE at
// D = 12, Re = 300 dies at ~8000 steps dense and runs past 20000 under
// ?levels=2, and this card with its rotation frozen (?omax=0) dies by 5000
// dense and runs past 16000 under AMR. The one configuration where AMR
// loses is this card with its rotation FREE -- ~9000 against dense's
// ~23000 -- which is the combination nothing in the suite gates: `drift`
// translates with the fluid force OFF, `spin` rotates with it OFF, and no
// other body rotates at all.
//
// `tow` AND `stream` ARE HERE TOO, and on a plate they buy something the
// sphere's pair cannot (plans/3D.md D1). Exactly as in `fall`, the body moves
// at `tow`, the fluid sits at `stream`, only the difference is physical, and
// sliding the split at a fixed difference is the same flow seen from a moving
// frame. Two reasons this matters more on the plate than on the sphere:
//
//   A BROADSIDE PLATE HAS NO STAIRCASE ERROR. It is a ROUNDBOX with r = 0
//   whose body frame is a PERMUTATION of the world axes, so at `tilt` = 0 its
//   surface lies exactly on cell faces and the discrete body IS the intended
//   body -- no more area than the smooth one, no effective diameter to
//   exceed the nominal. Every other body in this suite is a sphere carrying a
//   standing +7..13% offset against Schiller-Naumann for exactly that reason,
//   and that offset is wide enough to hide most defects. This is the first
//   case here whose Cd can be put next to a literature value without it.
//
//   ITS SEPARATION IS FIXED BY ITS EDGES, not by a boundary layer, so Cd is
//   nearly Reynolds-independent above Re ~ 1e3 and a textbook number exists:
//   1.18 for a SQUARE plate normal to the flow, rising to 1.98 as the span
//   goes to infinity (Hoerner, Fluid-Dynamic Drag, ch. 3; Blevins, Applied
//   Fluid Dynamics Handbook, table 10-4 -- 1.18 at b/h = 1, 1.2 at 5, 1.5 at
//   20, 1.98 at infinity). At the Re this solver can reach the measured value
//   should sit somewhat ABOVE the plateau, because below ~1e3 the viscous
//   contribution has not yet become negligible; that is a stated expectation,
//   not a tolerance to tune.
SCENARIOS.card = {
  name: 'card',
  defaults: { n: 32, aspect: 0.125, span: 1, re: 500, u_t: 0.05, i_star: 0.17,
              tilt: 0.15, perturb: 0, seed: 12345, tow: 0, stream: 0, spanfill: 0,
              edge: 0 },
  walls: [],
  // THE SPANWISE-PERIODIC PLATE (`spanfill=1`, plans/3D.md M6.5 avenue 2).
  // Pesavento & Wang's card is a SECTION of an infinite strip: no tips, no
  // tip vortices, and a dimensionless moment of inertia defined per unit
  // span. A finite plate is a different body -- Field et al. (1997) put a 3D
  // disk of this project's I* (0.17 in the paper's units is ~0.017 in the
  // disk definition) in the fluttering/chaotic band, not the tumbling one,
  // and the six-lever sweep of M6.5 measured exactly that (flutter, glide,
  // never a turn). The domain is already periodic on every axis, so the
  // infinite strip costs nothing: the plate's span is made to FILL z, with a
  // pad past the periodic seam so an unwrapped neighbour position (the step
  // kernel evaluates `get_phi3` at `z - ez` BEFORE wrapping) reads inside
  // exactly when its wrapped cell does. `span` then sets the z extent in
  // chords -- the spanwise room the wake has to develop a 3D mode in -- and
  // the lateral width is 8n, the 2D card's own BLOCKAGE = 8 (12.5%), because
  // this leg exists to be compared with the 2D result and not with the
  // free-fall box's 5%. The body is PLANAR by symmetry: an infinite strip
  // cannot translate along its span or turn about a cross-span axis, so the
  // integrator is told to forbid both (d3-body.mjs `planar`), which is also
  // what keeps the pad from ever leaving the seam.
  //
  // This is the decisive test of the fork against the paper: a strip at
  // I* = 0.17, Re = 1100 that does not tumble says the solver is wrong; one
  // that does, while the finite plate flutters, says the finite plate's
  // fluttering is physics.
  spanfillZ: (p) => Math.round(Math.max(1, p.span) * p.n),
  // Long in the fall direction, and WIDE ACROSS IT, which is where this
  // differs from `fall`. A tumbling plate does not fall straight: it
  // translates along the direction it is turning, at a speed comparable to
  // its own descent. The moving window handles that (window: all three axes,
  // not `fall`'s x alone), but the SPONGE band still has to sit outside the
  // near wake on every face.
  //
  // A PRESCRIBED LEG GETS A WIDER DOMAIN, and it has to: the free-fall domain
  // puts 5.0% of its cross-section behind the plate at the defaults, and a
  // bluff body at 5% blockage carries a wall correction of order +12% -- the
  // same size as the effect a literature comparison is trying to resolve. The
  // prescribed domain is 8n across both lateral axes, i.e. 1.56%, which is
  // the blockage the validated sphere cases already run at. A free fall is
  // untouched: it needs LENGTH to fall down, not width, and widening it would
  // move every recorded number in this file for no gain.
  dims: (p) => {
    const { n, span } = p;
    const prescribed = (p.tow || 0) !== 0 || (p.stream || 0) !== 0;
    if (p.spanfill) return [Math.round(6 * n), Math.round(8 * n), SCENARIOS.card.spanfillZ(p)];
    return prescribed
      ? [Math.round(8 * n), Math.round(8 * n),
         Math.round(Math.max(8 * n, 2 * span * n + 6 * n))]
      : [Math.round(6 * n), Math.round(5 * n),
         Math.round(Math.max(3 * n, 2 * span * n + 2 * n))];
  },
  derive: (p) => {
    const { n, aspect, span, re, u_t, i_star } = p;
    // Body half-extents. The BODY frame is (chord, span, thickness) on
    // (x, y, z) -- principalInertia's ROUNDBOX case is stated in exactly
    // those axes, so naming them any other way would put the plate's large
    // moment on the wrong axis and the tumble would be about the wrong one.
    const spanfill = !!p.spanfill;
    // The pad: 2 cells past the seam on each side. Any value > 0.5 makes the
    // unwrapped-neighbour test agree with the wrapped cell; 2 leaves a cell
    // of slack for the chi band's own reach.
    const SPANFILL_PAD = 2;
    const a = n / 2;                       // semi-chord
    const b = spanfill ? SCENARIOS.card.spanfillZ(p) / 2 + SPANFILL_PAD
                       : span * n / 2;    // semi-span
    const c = aspect * n / 2;              // semi-thickness
    // Re on the CHORD, the paper's convention: Re = 2 u_t a / nu.
    const nu = 2 * u_t * a / re;
    const tau = tauFromNu(nu);
    // rho_b from I*, the SAME closed form card-params.mjs uses --
    //   I* = t(c^2 + t^2) rho_b / (2 c^3 rho_f)   [c = semi-chord, t = semi-thickness]
    // -- which is per unit span and therefore span-independent, which is why
    // it transfers to a 3D plate at all. Clamped above 1 for that module's
    // reason: a card lighter than the fluid is not this regime.
    const rho_b = Math.max(1.05, i_star * 2 * a ** 3 / (c * (a * a + c * c)));
    // `edge` rounds the plate's edges: r = edge * c, so 1 is a STADIUM
    // section (a full semicircle of radius c on each chord edge), the
    // nearest the ROUNDBOX comes to the 2D card, which is an ELLIPSE
    // (card-params.mjs). Volume and inertia keep the box formulas -- at
    // c/a = 1/8 the stadium is 2.7% lighter than the box, inside the I*
    // clamp's own slack -- so rho_b and g_eff are the box's. A true elliptic
    // cylinder is the next body if this one matters.
    const shape = { kind: SHAPE.ROUNDBOX, a, b, c, r: Math.max(0, Math.min(1, p.edge || 0)) * c };
    const V = 8 * a * b * c;
    // BROADSIDE drag balance, as in `fall`: the plate is dropped flat, and
    // the frontal area is then the whole face. Cd = 1.1 is the textbook flat
    // plate normal to the flow.
    //
    // THIS SETS THE TRANSIENT, NOT THE ANSWER. Once the plate turns, its own
    // drag is whatever the flow says it is; g_eff is only how hard it is
    // pulled. Worth checking against 2D anyway, because it is the same
    // balance written two ways: card-params.mjs ends at G_EFF = U_T^2/(pi t rho_b)
    // and this is U_T^2 Cd/(4 t rho_b), i.e. 1.1/4 = 0.275 against 1/pi = 0.318.
    const cd = 1.1;
    // The frontal area INSIDE the domain: for the periodic strip that is the
    // z extent, not the padded half-extent, and it is what blockage and any
    // Cd are normalized by. gEff is unchanged by the pad because area and V
    // scale together in b -- it is u_t^2 cd / (4 c rho_b) either way.
    const area = spanfill ? 4 * a * (SCENARIOS.card.spanfillZ(p) / 2) : 4 * a * b;
    const gEff = u_t * u_t * cd * (4 * a * b) / (2 * rho_b * V);
    const d = SCENARIOS.card.dims(p);
    // ORIENTATION, and it is the only fiddly part.
    //
    // The plate must fall FLAT -- its thin axis along the fall direction --
    // and tumble about its SPAN, and the tumble has to be VISIBLE in the
    // default z-slice, whose plane is (x, y). So the span must be world z.
    // That fixes the frame completely: body (chord, span, thickness) ->
    // world (y, z, x), which is the cyclic permutation x->y->z->x, i.e. a
    // turn of 2pi/3 about (1,1,1). Then `tilt` pitches it about world z, the
    // span, which is the one rotation that changes the angle of attack.
    //
    // Composed in WORLD order: qRotate(q, .) maps body to world, so a further
    // world-frame rotation multiplies on the LEFT.
    const qFlat = qFromAxisAngle([1, 1, 1], 2 * Math.PI / 3);
    // THE GALILEAN SPLIT, character for character the one `fall` carries --
    // see that scenario's header for why both speeds are signed and why the
    // arms key off `prescribed` and `uRelSigned` rather than off which knob
    // happens to be set.
    const towed = p.tow !== 0, streamed = p.stream !== 0;
    const prescribed = towed || streamed;
    const uRelSigned = p.stream - p.tow;
    const bodyRelSign = prescribed ? Math.sign(p.tow - p.stream) : 1;
    // WHERE THE PLATE SITS ON THE LATTICE, and for a measurement leg this is
    // not a detail -- it is the difference between a body with no
    // discretization error and one with a 12% area error.
    //
    // A cell is solid when its CENTRE has phi < 0, so an axis-aligned box of
    // half-extent h occupies the cells with |delta| < h. Cell centres sit at
    // integers, so:
    //   h INTEGER      -> the centre must sit at a HALF-integer (a cell
    //                     corner), and the count is exactly 2h.
    //                     Centred on a cell centre instead, |delta| < h
    //                     EXCLUDES both end cells and the count is 2h - 1.
    //   h HALF-INTEGER -> the centre must sit at a cell CENTRE, count 2h.
    // Get it wrong and the plate is one cell short on that axis. MEASURED, at
    // n = 16 span = 1 aspect = 0.125 with the centre on cell centres: the
    // discrete plate was 15 x 15 x 1 rather than 16 x 16 x 2, i.e. a frontal
    // area of 225 against the 256 `area` normalizes Cd by -- 12% low, and the
    // pinned Cd came out 1.00 against the square-plate 1.18 almost entirely
    // because of it.
    //
    // PRESCRIBED, AXIS-ALIGNED LEGS ONLY. A tilted plate cannot be aligned to
    // the lattice at all and a free fall tilts immediately, so there is
    // nothing to snap to; those keep the plain centring and every recorded
    // free-fall number is unchanged. The world axes here are
    // (thickness, chord, span) = (x, y, z) -- qFlat's permutation -- so the
    // half-extents pair with them in that order.
    const snap = (centre, h) => (Number.isInteger(h) ? Math.round(centre) + 0.5 : Math.round(centre));
    const aligned = prescribed && p.tilt === 0;
    const x0 = bodyRelSign >= 0 ? d[0] - 2 * n : 2 * n;
    const place = aligned
      ? [snap(x0, c), snap(d[1] / 2, a), snap(d[2] / 2, b)]
      : [x0, d[1] / 2, d[2] / 2];
    const body = makeBodyState({
      shape, density: rho_b,
      x: place,
      q: qMul(qFromAxisAngle([0, 0, 1], p.tilt), qFlat),
      v: [p.tow, 0, 0],
    });
    return {
      nu, tau, re, R: a, D: n, dims: d, body,
      pinned: streamed && !towed,
      noFluidForce: towed,
      tow: p.tow, stream: p.stream,
      uRel: prescribed ? Math.abs(uRelSigned) : null,
      uRelSigned: prescribed ? uRelSigned : null,
      area, cd, rho_b, u_t, aspect, span, i_star,
      // A PRESCRIBED LEG IS NOT FALLING. Gravity would accelerate nothing (the
      // body is on rails either way) but it would be reported, and a case
      // whose stated forcing is not the forcing it runs under is the kind of
      // lie this file's headers exist to prevent.
      gravity: prescribed ? [0, 0, 0] : [gEff, 0, 0], gEff,
      // ALL THREE AXES (plans/3D.md M8.3's own note: "A tumbling plate is
      // the case that changes that answer, and it can say so itself"). A
      // plate that tumbles translates across the domain as fast as it falls
      // down it, so an x-only window loses it sideways. A PINNED plate has
      // nothing for a window to follow; a towed one travels on x alone,
      // because it is on rails and cannot turn.
      window: prescribed ? (towed ? [1, 0, 0] : [0, 0, 0]) : (spanfill ? [1, 1, 0] : [1, 1, 1]),
      // The integrator's symmetry constraint for the periodic strip; see the
      // spanfill note above. Inert for every other leg.
      planar: spanfill,
      spanfill,
      // No band on the periodic span axis of the strip: there is no far
      // field there to relax toward (common_d3_step.wgsl SPONGE_AX).
      sponge: { width: Math.max(6, Math.round(n / 2)), u: [p.stream, 0, 0],
                axes: spanfill ? [1, 1, 0] : [1, 1, 1] },
      force: [0, 0, 0],
      blockage: area / (d[1] * d[2]),
      tSettle: u_t / gEff,
      convective: n / u_t,
      // THE REFERENCE IS THE SHAPE'S, NOT THE TRANSIENT'S. For a free fall
      // `cd` is the 1.1 that sets g_eff and is explicitly "the transient, not
      // the answer"; for a prescribed leg there is a real literature value to
      // be measured against, and it is the square-plate 1.18 of the header --
      // for a SQUARE plate, so it is reported only at span = 1 and left null
      // otherwise rather than quietly extrapolated off Blevins' table.
      cdReference: prescribed ? (span === 1 ? 1.18 : null) : cd,
      down: [1, 0, 0],
    };
  },
  // Seeded at the freestream, which is 0 for a free fall or a pure tow and
  // the prescribed value otherwise -- the same single expression `fall` uses,
  // for the same reason: a conditional here would be a second place for the
  // two to disagree about what the far field is.
  macro: (dims, p) => seedMacro3Perturbed(dims, () => [p.stream, 0, 0],
                                          p.perturb * p.u_t, p.seed),
};

// --- drift: a body that TRANSLATES, for the dynamic-refinement gate --------
//
// plans/3D.md M4.2b-iii. A free sphere given an initial linear velocity with
// the fluid force switched off (NO_FLUID_FORCE), so it crosses the domain in
// a straight line at a known constant speed.
//
// WHY A SCENARIO AND NOT A TWEAK TO `sphere`. Geometry-forced refinement has
// to follow a body that MOVES, and every other scenario's body is pinned or
// only rotates -- `spin` turns in place, which never moves the refined shell
// across a block boundary. Refining ahead of a body and coarsening behind it
// is the thing under test, and nothing in the suite could exercise it.
//
// WHY THE FLUID FORCE IS OFF, which makes this not a fluid case: the body's
// trajectory is then EXACTLY known -- x(t) = x0 + v t -- so the set of blocks
// the criterion must refine is known too, and any coverage failure is the
// manager and not the flow. With the force on, a coverage miss and a body
// that simply went somewhere else look identical.
//
// The fluid is still solved around it, so the refined tiles carry a real
// solution and a newly-refined tile that was filled wrongly still shows up
// as a blowup.
SCENARIOS.drift = {
  name: 'drift',
  defaults: { n: 24, tau: 0.8, u0: 0.02 },
  walls: [],
  dims: ({ n }) => [4 * n, 2 * n, 2 * n],
  derive: (p) => {
    const d = [4 * p.n, 2 * p.n, 2 * p.n];
    const shape = { kind: SHAPE.SPHERE, a: p.n / 3, b: p.n / 3, c: p.n / 3, r: 0 };
    // Starts a quarter of the way along x and drifts +x, so it has room to
    // cross many block boundaries before reaching the far side.
    const body = makeBodyState({
      shape, x: [d[0] / 4, d[1] / 2, d[2] / 2], v: [p.u0, 0, 0],
    });
    return { nu: nuFromTau(p.tau), body, pinned: false, noFluidForce: true,
             force: [0, 0, 0], dims: d };
  },
  macro: (dims) => seedMacro3(dims, () => [0, 0, 0]),
};

// --- spin: the 6-DOF integrator, with no fluid in the way ------------------
//
// A free body given an initial spin, with the fluid force switched off
// (NO_FLUID_FORCE). Not a fluid case at all: it exists so that
// shaders/d3_physics.wgsl -- the hardest new code in M2, and the one whose
// errors are least visible -- can be compared step for step against
// d3-body.mjs's stepFreeBody() on REAL GPU CODE, rather than being trusted
// because the host reference it mirrors is well tested.
//
// The default spin is near the INTERMEDIATE principal axis, so the
// trajectory being compared is the tumbling one: an integrator that agrees
// on a steady spin but not on a flip would pass a gentler test.
SCENARIOS.spin = {
  name: 'spin',
  // `fluid` COUPLES THE BODY TO THE FLOW, and it is 0 by default so the
  // M1 gate (benchmarks/d3.json's spin-N32) is bit-identical: that case
  // compares the GPU integrator against d3-body.mjs on a body the fluid must
  // not touch, and a fluid torque would make the comparison meaningless.
  //
  // 1 turns the fluid force ON, which is the ONE COMBINATION THIS SUITE HAD
  // NO CASE FOR: `drift` TRANSLATES with the coupling off, this scenario
  // ROTATES with it off, and every other body is pinned or does not turn. A
  // body that rotates under fluid torque on a REFINED grid is what M8.8's
  // rotation lead exists to protect, and nothing exercised it -- see
  // tools/validate-d3-invariants.js's `spin-amr`.
  // `rho_b` MATTERS ONLY ONCE `fluid` IS ON, and then it matters a great deal.
  // The default 1 is the fluid's own density, which for the uncoupled M1 gate
  // is an arbitrary scale that cancels. Coupled, it is the WORST case: a
  // neutrally-buoyant plate has no inertia to speak of against the fluid it
  // displaces, and measured here it sheds 140x of its spin in 250 steps and
  // then sits still -- a gate that stops rotating is a gate that stops
  // testing rotation. A denser body coasts, and the spin-down time scales
  // with it.
  defaults: { n: 32, tau: 0.8, u0: 0.02, fluid: 0, rho_b: 1 },
  walls: [],
  dims: ({ n }) => [n, n, n],
  derive: (p) => {
    const shape = { kind: SHAPE.ROUNDBOX, a: p.n / 4, b: p.n / 8, c: p.n / 24, r: 0 };
    const body = makeBodyState({
      shape, x: [p.n / 2, p.n / 2, p.n / 2],
      q: qFromAxisAngle([0.3, 0.5, 0.81], 0.4),   // a generic orientation, not axis-aligned
      omega: [1e-4 * p.u0, p.u0, 1e-4 * p.u0],    // near the intermediate axis
      density: p.rho_b,
    });
    return { nu: nuFromTau(p.tau), body, pinned: false, noFluidForce: !p.fluid,
             force: [0, 0, 0], dims: [p.n, p.n, p.n],
             // THE BODY TURNS IN PLACE, so there is nothing for a window to
             // follow and no `down` for the view to orient to.
             R: shape.a, D: 2 * shape.a };
  },
  macro: (dims) => seedMacro3(dims, () => [0, 0, 0]),
};

export const SCENARIO_NAMES = Object.keys(SCENARIOS);

// Resolve URL parameters into a fully-derived scenario. Throws on an
// unknown name rather than silently falling back, so a typo in a
// validation config fails loudly instead of quietly validating the wrong
// thing.
export function resolveScenario(name, overrides = {}) {
  const sc = SCENARIOS[name];
  if (!sc) throw new Error(`unknown scenario "${name}": expected one of ${SCENARIO_NAMES.join(', ')}`);
  const p = { ...sc.defaults, ...Object.fromEntries(Object.entries(overrides).filter(([, v]) => v != null)) };
  const derived = sc.derive(p);
  // Every scenario resolves to explicit dims. The M1 ones are cubes and say
  // so rather than leaving the page to infer it, so there is exactly one
  // place that decides a domain shape.
  const dims = derived.dims || (sc.dims ? sc.dims(p) : [p.n, p.n, p.n]);
  return { scenario: sc.name, walls: sc.walls, ...p, ...derived, dims };
}
