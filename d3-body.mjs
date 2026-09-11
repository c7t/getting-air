// 6-DOF rigid body for the dense 3D solver: shapes, signed distances,
// inertia tensors, quaternion algebra, and the free-body integrator.
// plans/3D.md M2, sec 1.3 ("the rigid body goes 3 DOF -> 6 DOF").
//
// Shared by the page (which uploads a BodyState3D built here) and by the
// tools (which score against it) -- and, for the integrator specifically,
// it is the REFERENCE the WGSL in shaders/d3_physics.wgsl mirrors line for
// line. That mirroring is checked on real GPU code by the `spin` scenario,
// which runs the body with the fluid force switched off and compares the
// GPU trajectory against stepFreeBody() below; tools/test-d3-body.js checks
// this side against the conservation laws it must obey.
//
// WHY ANGULAR MOMENTUM AND NOT THE EXPLICIT GYROSCOPIC TERM. plans/3D.md
// sec 1.3 names `omega x (I omega)` as the new term, integrating
//     omega_dot = I_world^-1 (torque - omega x (I_world omega)).
// That is correct but it is the worse of the two equivalent formulations:
// it needs an explicit cross product (a sign error there is invisible until
// a body tumbles), and the conservation of L it should imply holds only to
// the integrator's accuracy. Integrating L directly instead,
//     L    += torque * dt
//     omega = I_world^-1 L,       I_world = R I_body R^T
// makes L EXACTLY conserved under zero torque -- to floating point, not to
// truncation order -- and the gyroscopic effect emerges from R rotating
// inside I_world, which is where it physically comes from. The tumbling
// physics is identical; the bookkeeping is not. tools/test-d3-body.js
// asserts both the conservation and the qualitative consequence (the
// intermediate-axis instability), so this is checked, not asserted.
//
// Same browser+Node dual-consumption pattern as d3-scenarios.mjs and
// lattice-3d.mjs.

import { wrapPosition3 } from './d3-window.mjs';

// --- shapes ---------------------------------------------------------------
// Kept numeric and in one place because the WGSL switches on the same
// values; tools/test-d3-body.js asserts the shader's copies agree.
export const SHAPE = { SPHERE: 0, SPHEROID: 1, ROUNDBOX: 2 };
export const SHAPE_NAMES = ['sphere', 'spheroid', 'roundbox'];

// Every SDF here is a TRUE signed distance, not an algebraic surrogate.
// common_geometry.wgsl's header documents at length what the 2D algebraic
// form cost -- a chi band and a refinement margin that were anisotropic by
// the body's aspect ratio, 8x at the default card -- and plans/3D.md sec
// 1.3 notes the same error distorts along TWO axes in 3D. Getting it right
// from the start is cheaper than the retrofit was.
//
// A triaxial ellipsoid is DELIBERATELY not offered: its true distance needs
// a 6th-degree root or a full iteration, and an approximate one is exactly
// the trap above. A SPHEROID (two axes equal) is offered instead, because a
// surface of revolution reduces exactly to the 2D ellipse problem in the
// meridional plane, which this project already solves and validated.

// Exact: distance to a sphere of radius a.
function sdSphere(p, a) { return Math.hypot(p[0], p[1], p[2]) - a; }

// Exact: rounded box with half-extents (a,b,c) and corner radius r. The
// unrounded core `max(|p|-h, 0)` term is the exact exterior distance; the
// `min(max(...), 0)` term is the exact interior distance to the nearest
// face. Both are exact, which is why this and not a smooth approximation.
function sdRoundBox(p, a, b, c, r) {
  const qx = Math.abs(p[0]) - (a - r), qy = Math.abs(p[1]) - (b - r), qz = Math.abs(p[2]) - (c - r);
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
  const inside = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
  return outside + inside - r;
}

// Newton on the closest-point parameter of the meridional ellipse, ported
// from common_geometry.wgsl's 2D solver -- see that file for why 3
// iterations and why the seed is atan2(y*a, x*b). A spheroid with semi-axes
// (a, a, c) is the surface of revolution of the ellipse (a, c) about the
// local z axis, so the 3D distance from p is the 2D distance from
// (hypot(px,py), pz). Exact modulo the same Newton convergence the 2D
// version measured at 0.15% worst case.
function sdSpheroid(p, a, c) {
  const r0 = Math.hypot(p[0], p[1]);
  if (Math.abs(a - c) <= 1e-6 * Math.max(a, c)) return Math.hypot(r0, p[2]) - a;
  const x = Math.abs(r0), y = Math.abs(p[2]);
  const rr = Math.sqrt((x * x) / (a * a) + (y * y) / (c * c));
  const inside = rr < 1;
  let t = Math.atan2(y * a, x * c);
  for (let i = 0; i < 3; i++) {
    const ct = Math.cos(t), st = Math.sin(t);
    const ex = x - a * ct, ey = y - c * st;
    const F = ex * (-a * st) + ey * (c * ct);
    const Fp = -(a * a * st * st + c * c * ct * ct) + ex * (-a * ct) + ey * (-c * st);
    t -= F / (Math.abs(Fp) < 1e-9 ? -1e-9 : Fp);
    t = Math.min(Math.max(t, 0), Math.PI / 2);
  }
  const d = Math.hypot(x - a * Math.cos(t), y - c * Math.sin(t));
  return inside ? -d : d;
}

// Signed distance in the BODY frame (origin at the centre of mass, axes
// along the principal axes). World-frame callers rotate p in first.
export function sdfBody(p, shape) {
  switch (shape.kind) {
    case SHAPE.SPHERE: return sdSphere(p, shape.a);
    case SHAPE.SPHEROID: return sdSpheroid(p, shape.a, shape.c);
    case SHAPE.ROUNDBOX: return sdRoundBox(p, shape.a, shape.b, shape.c, shape.r);
    default: throw new Error(`unknown shape kind ${shape.kind}`);
  }
}

// Volume and principal moments of inertia for a uniform body of mass m.
// Principal axes are the body axes by construction for all three shapes.
export function bodyVolume(shape) {
  switch (shape.kind) {
    case SHAPE.SPHERE: return (4 / 3) * Math.PI * shape.a ** 3;
    case SHAPE.SPHEROID: return (4 / 3) * Math.PI * shape.a * shape.a * shape.c;
    // The rounded box's true volume is the box minus the corner/edge
    // shavings plus the rounds; for r << min(a,b,c) the box value is within
    // a fraction of a percent and every consumer here is a mass/inertia
    // scale, not a conservation law. Stated rather than silently assumed.
    case SHAPE.ROUNDBOX: return 8 * shape.a * shape.b * shape.c;
    default: throw new Error(`unknown shape kind ${shape.kind}`);
  }
}

export function principalInertia(shape, m) {
  switch (shape.kind) {
    case SHAPE.SPHERE: { const i = 0.4 * m * shape.a ** 2; return [i, i, i]; }
    case SHAPE.SPHEROID: {
      const ix = m * (shape.a ** 2 + shape.c ** 2) / 5;
      return [ix, ix, 2 * m * shape.a ** 2 / 5];
    }
    case SHAPE.ROUNDBOX: {
      const { a, b, c } = shape;
      return [m * (b * b + c * c) / 3, m * (a * a + c * c) / 3, m * (a * a + b * b) / 3];
    }
    default: throw new Error(`unknown shape kind ${shape.kind}`);
  }
}

// --- quaternions ----------------------------------------------------------
// Layout [w, x, y, z] everywhere, host and shader. A quaternion stored as
// [x,y,z,w] on one side and [w,x,y,z] on the other is the 3D analogue of a
// transposed `opp` table: it does not crash, it produces a plausible
// rotation, and it is checked by tools/test-d3-body.js against the shader's
// own ordering rather than trusted.

export function qIdentity() { return [1, 0, 0, 0]; }

export function qMul(p, q) {
  const [pw, px, py, pz] = p, [qw, qx, qy, qz] = q;
  return [
    pw * qw - px * qx - py * qy - pz * qz,
    pw * qx + px * qw + py * qz - pz * qy,
    pw * qy - px * qz + py * qw + pz * qx,
    pw * qz + px * qy - py * qx + pz * qw,
  ];
}

export function qConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }

export function qNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  // A zero quaternion cannot arise from a normalized start plus a bounded
  // update, but falling back to identity is cheaper than a NaN that would
  // propagate into every cell's SDF.
  if (!(n > 1e-12)) return qIdentity();
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function qFromAxisAngle(axis, angle) {
  const n = Math.hypot(...axis) || 1;
  const s = Math.sin(angle / 2) / n;
  return [Math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s];
}

// Rotate a vector from BODY to WORLD: v_world = q v q*.
export function qRotate(q, v) {
  const r = qMul(qMul(q, [0, ...v]), qConj(q));
  return [r[1], r[2], r[3]];
}

// Rotate a vector from WORLD to BODY: v_body = q* v q.
export function qRotateInv(q, v) {
  const r = qMul(qMul(qConj(q), [0, ...v]), q);
  return [r[1], r[2], r[3]];
}

// --- rigid-body integration ----------------------------------------------

// omega (world) from angular momentum L (world) and the body's principal
// moments: omega = R I_body^-1 R^T L. Written as rotate-into-body, divide,
// rotate-back rather than by forming the 3x3 I_world and inverting it --
// same result, no matrix, and no chance of an inverse going bad for a very
// anisotropic plate (Iz/Ix at the falling card's aspect is large).
// Exact rotation of angle |w|*dt about w, as a quaternion. Identity for a
// vanishing rate, which is the only case qFromAxisAngle's axis
// normalization could not handle.
export function expMap(w, dt) {
  const m = Math.hypot(...w);
  return m > 1e-12 ? qFromAxisAngle(w, m * dt) : qIdentity();
}

export function omegaFromL(q, ibody, L) {
  const Lb = qRotateInv(q, L);
  return qRotate(q, [Lb[0] / ibody[0], Lb[1] / ibody[1], Lb[2] / ibody[2]]);
}

export function rotationalEnergy(q, ibody, L) {
  const w = omegaFromL(q, ibody, L);
  return 0.5 * (L[0] * w[0] + L[1] * w[1] + L[2] * w[2]);
}

// One semi-implicit Euler step, dt = 1 lattice step (matching physics.wgsl's
// convention, where every `+=` is implicitly per-step).
//
// THE REFERENCE for shaders/d3_physics.wgsl. Any change here has to be
// mirrored there, and the `spin` scenario compares the two on real GPU code.
// `wrap` is the moving window's per-axis domain size, 0 on an axis with no
// window -- see d3-window.mjs. Default [0,0,0], so this is the unwindowed
// integrator it always was, bit for bit.
export function stepFreeBody(s, { force = [0, 0, 0], torque = [0, 0, 0], gravity = [0, 0, 0], dt = 1, vMax = Infinity, oMax = Infinity, wrap = [0, 0, 0] } = {}) {
  const m = s.mass;
  let v = [
    s.v[0] + dt * (force[0] / m + gravity[0]),
    s.v[1] + dt * (force[1] / m + gravity[1]),
    s.v[2] + dt * (force[2] / m + gravity[2]),
  ];
  // MAGNITUDE clamps, matching d3_physics.wgsl -- clamping components
  // independently would change the vector's direction, which is a limiter
  // silently steering a tumbling body rather than merely containing a
  // blowup. Both are no-ops at the default Infinity.
  const vmag = Math.hypot(...v);
  if (vmag > vMax) v = v.map(c => c * (vMax / vmag));
  let L = [s.L[0] + dt * torque[0], s.L[1] + dt * torque[1], s.L[2] + dt * torque[2]];
  // Position, wrapped into the buffer on every windowed axis (M8.3), with
  // the running DISPLACEMENT accumulated before the wrap -- it is the one
  // consumer that wants the unwrapped answer, and it is reporting only.
  // Mirrors shaders/d3_physics.wgsl.
  const x = wrapPosition3([s.x[0] + dt * v[0], s.x[1] + dt * v[1], s.x[2] + dt * v[2]], wrap);
  const d = [(s.d?.[0] ?? 0) + dt * v[0], (s.d?.[1] ?? 0) + dt * v[1], (s.d?.[2] ?? 0) + dt * v[2]];
  // Orientation update, from the WORLD-frame omega implied by the NEW L.
  //
  // MIDPOINT ON THE ROTATION GROUP, and this is load-bearing rather than
  // fastidious. For a free body L is constant but omega is NOT -- it is
  // R I_body^-1 R^T L, so it changes as the body turns, and that is exactly
  // the gyroscopic effect. Stepping with omega frozen at the start of the
  // interval (explicit Lie-Euler) is first-order in that variation, and
  // measured here it drifts **58.87% in rotational energy over 20000 steps**
  // and -- far worse -- makes a body spun about its MAJOR axis flip. Major-
  // axis spin is stable in reality; an integrator that invents an
  // instability is worse than a slow one, and no conservation check catches
  // it (|L| is exactly conserved throughout).
  //
  // Evaluating omega at the half step instead costs two extra
  // rotate-divide-rotates in what is a once-per-step scalar update, and
  // takes the same measurement to **0.0052%** with the stable axis stable
  // and the intermediate axis still flipping. tools/test-d3-body.js asserts
  // all three of those, which is what pins this down rather than leaving it
  // to taste.
  //
  // The exponential map (rather than the first-order tangent step
  // `q + (dt/2) w_quat q`) is used for each partial rotation: unit-norm by
  // construction, and exact for a constant omega over its interval.
  let w = omegaFromL(s.q, s.ibody, L);
  const wmag = Math.hypot(...w);
  if (wmag > oMax) { const k = oMax / wmag; w = w.map(c => c * k); L = L.map(c => c * k); }
  const qHalf = qNormalize(qMul(expMap(w, dt / 2), s.q));
  const wHalf = omegaFromL(qHalf, s.ibody, L);
  const q = qNormalize(qMul(expMap(wHalf, dt), s.q));
  return { ...s, x, v, q, L, d, omega: wHalf };
}

export function makeBodyState({ shape, x, q = qIdentity(), v = [0, 0, 0], omega = [0, 0, 0], density = 1 }) {
  const mass = density * bodyVolume(shape);
  const ibody = principalInertia(shape, mass);
  const Lb = qRotateInv(q, omega);
  const L = qRotate(q, [Lb[0] * ibody[0], Lb[1] * ibody[1], Lb[2] * ibody[2]]);
  // `d` is the displacement since construction, which is zero by definition.
  return { shape, mass, ibody, x: x.slice(), q: q.slice(), v: v.slice(), L, omega: omega.slice(), d: [0, 0, 0] };
}

// --- GPU layout -----------------------------------------------------------
// Flat f32 array matching shaders/common_d3_geometry.wgsl's BodyState3D
// EXACTLY. Field order is the contract; tools/test-d3-body.js parses the
// WGSL struct and asserts it, the same way tools/test-f-pack.js guards the
// packed-f layout.
export const BODY_FIELDS = [
  'cx', 'cy', 'cz',
  'qw', 'qx', 'qy', 'qz',
  'vx', 'vy', 'vz',
  'Lx', 'Ly', 'Lz',
  'wx', 'wy', 'wz',
  'fx', 'fy', 'fz',
  'tx', 'ty', 'tz',
  'mass', 'ix', 'iy', 'iz',
  'a', 'b', 'c', 'r',
  'shape', 'pinned', 'v_max', 'o_max',
  // Total displacement since reset -- REPORTING ONLY, see the WGSL struct's
  // own note for why its unbounded growth is harmless here and was not in 2D.
  'dx', 'dy', 'dz',
];

export function packBodyState(s, { pinned = false, vMax = 0.2, oMax = 0.1 } = {}) {
  const o = {
    cx: s.x[0], cy: s.x[1], cz: s.x[2],
    qw: s.q[0], qx: s.q[1], qy: s.q[2], qz: s.q[3],
    vx: s.v[0], vy: s.v[1], vz: s.v[2],
    Lx: s.L[0], Ly: s.L[1], Lz: s.L[2],
    wx: s.omega[0], wy: s.omega[1], wz: s.omega[2],
    fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0,
    mass: s.mass, ix: s.ibody[0], iy: s.ibody[1], iz: s.ibody[2],
    a: s.shape.a, b: s.shape.b ?? s.shape.a, c: s.shape.c ?? s.shape.a, r: s.shape.r ?? 0,
    shape: s.shape.kind, pinned: pinned ? 1 : 0, v_max: vMax, o_max: oMax,
    dx: s.d?.[0] ?? 0, dy: s.d?.[1] ?? 0, dz: s.d?.[2] ?? 0,
  };
  return new Float32Array(BODY_FIELDS.map(k => o[k]));
}

export function unpackBodyState(arr) {
  const o = {};
  BODY_FIELDS.forEach((k, i) => { o[k] = arr[i]; });
  return o;
}
