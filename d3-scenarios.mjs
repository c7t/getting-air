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

import { SHAPE, makeBodyState, qFromAxisAngle } from './d3-body.mjs';

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
  defaults: { n: 16, tau: null, u0: 0.05, re: 100 },
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
  macro: (dims, p) => seedMacro3(dims, () => [p.u0, 0, 0]),
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
  defaults: { n: 32, tau: 0.8, u0: 0.02 },
  walls: [],
  dims: ({ n }) => [n, n, n],
  derive: (p) => {
    const shape = { kind: SHAPE.ROUNDBOX, a: p.n / 4, b: p.n / 8, c: p.n / 24, r: 0 };
    const body = makeBodyState({
      shape, x: [p.n / 2, p.n / 2, p.n / 2],
      q: qFromAxisAngle([0.3, 0.5, 0.81], 0.4),   // a generic orientation, not axis-aligned
      omega: [1e-4 * p.u0, p.u0, 1e-4 * p.u0],    // near the intermediate axis
    });
    return { nu: nuFromTau(p.tau), body, pinned: false, noFluidForce: true, force: [0, 0, 0], dims: [p.n, p.n, p.n] };
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
