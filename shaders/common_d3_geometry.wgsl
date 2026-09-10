// 6-DOF rigid-body state and TRUE signed-distance geometry for the dense 3D
// solver -- the solid mask every step/force/render kernel tests against.
// plans/3D.md M2. Fragment only; included via
// `// @include "common_d3_geometry.wgsl"`.
//
// FIELD ORDER IS A CONTRACT with d3-body.mjs's BODY_FIELDS, which builds the
// buffer this reads. tools/test-d3-body.js parses this struct out of this
// file and asserts the two agree, the same way tools/test-f-pack.js guards
// the packed-f layout -- a mismatch here would not crash, it would rotate
// the body by someone else's quaternion.
//
// QUATERNION LAYOUT is [w, x, y, z], here and on the host. Storing [x,y,z,w]
// on one side is the 3D analogue of a transposed `opp` table: no crash, a
// plausible-looking wrong rotation.
//
// ANGULAR MOMENTUM, NOT ANGULAR VELOCITY, is the integrated state -- see
// d3-body.mjs's header for why (L is exactly conserved at zero torque, and
// the gyroscopic term emerges from R rotating inside I_world instead of
// needing an explicit, sign-error-prone cross product). `w*` is cached
// omega, recomputed each step by d3_physics.wgsl so the step and force
// kernels do not each have to redo the rotate-divide-rotate.

struct BodyState3D {
  cx : f32, cy : f32, cz : f32,
  qw : f32, qx : f32, qy : f32, qz : f32,
  vx : f32, vy : f32, vz : f32,
  Lx : f32, Ly : f32, Lz : f32,
  wx : f32, wy : f32, wz : f32,
  // Last measured fluid force and torque. Written by d3_physics.wgsl every
  // step, INCLUDING for a pinned body -- measuring the force on a body that
  // does not move is the entire point of the sphere validation scenario, so
  // this is the channel Cd is read back through.
  fx : f32, fy : f32, fz : f32,
  tx : f32, ty : f32, tz : f32,
  mass : f32, ix : f32, iy : f32, iz : f32,
  a : f32, b : f32, c : f32, r : f32,
  shape : f32, pinned : f32, v_max : f32, o_max : f32,
}

const SHAPE_SPHERE   = 0u;
const SHAPE_SPHEROID = 1u;
const SHAPE_ROUNDBOX = 2u;

// --- quaternion helpers ---------------------------------------------------

fn qMul(p: vec4<f32>, q: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    p.x*q.x - p.y*q.y - p.z*q.z - p.w*q.w,
    p.x*q.y + p.y*q.x + p.z*q.w - p.w*q.z,
    p.x*q.z - p.y*q.w + p.z*q.x + p.w*q.y,
    p.x*q.w + p.y*q.z - p.z*q.y + p.w*q.x,
  );
}
fn qConj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(q.x, -q.y, -q.z, -q.w); }

// BODY -> WORLD.
fn qRot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let r = qMul(qMul(q, vec4<f32>(0f, v)), qConj(q));
  return r.yzw;
}
// WORLD -> BODY.
fn qRotInv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let r = qMul(qMul(qConj(q), vec4<f32>(0f, v)), q);
  return r.yzw;
}

fn bodyQuat(s: BodyState3D) -> vec4<f32> { return vec4<f32>(s.qw, s.qx, s.qy, s.qz); }

// Exact rotation of angle |w|*dt about w, as a unit quaternion. Identity
// for a vanishing rate, which is the one case the axis normalization could
// not handle. Host counterpart: d3-body.mjs's expMap().
fn expMapQ(w: vec3<f32>, dt: f32) -> vec4<f32> {
  let m = length(w);
  if (m <= 1e-12f) { return vec4<f32>(1f, 0f, 0f, 0f); }
  let half = 0.5f * m * dt;
  return vec4<f32>(cos(half), sin(half) * w / m);
}

// omega = R I_body^-1 R^T L, written rotate-divide-rotate rather than by
// forming and inverting a 3x3 I_world: same result, no matrix, and no
// ill-conditioned inverse for a very anisotropic plate (I_zz / I_xx is
// large at a falling card's aspect ratio). Host counterpart:
// d3-body.mjs's omegaFromL().
fn omegaFromL(q: vec4<f32>, s: BodyState3D, L: vec3<f32>) -> vec3<f32> {
  let Lb = qRotInv(q, L);
  return qRot(q, vec3<f32>(Lb.x / s.ix, Lb.y / s.iy, Lb.z / s.iz));
}

// --- signed distances -----------------------------------------------------
//
// All three are TRUE signed distances. common_geometry.wgsl's header records
// in detail what the 2D algebraic surrogate cost -- a chi band and a
// refinement margin anisotropic by the body's aspect ratio, 8x at the
// default card -- and plans/3D.md sec 1.3 notes the distortion runs along
// TWO axes in 3D. So this is written correct from the start rather than
// retrofitted.
//
// A TRIAXIAL ELLIPSOID IS DELIBERATELY ABSENT. Its exact distance needs a
// 6th-degree root or a full iteration, and an approximate one is precisely
// the trap above. A SPHEROID is offered instead: a surface of revolution
// reduces exactly to the 2D ellipse problem in the meridional plane, which
// this project already solves and has validated.

fn sdSphere(p: vec3<f32>, a: f32) -> f32 { return length(p) - a; }

// Exact rounded box. `max(q, 0)` length is the exterior distance; the
// `min(max-component, 0)` term is the exact interior distance to the
// nearest face. Both exact -- which is the reason for this shape rather
// than a smooth blend of half-spaces.
fn sdRoundBox(p: vec3<f32>, h: vec3<f32>, r: f32) -> f32 {
  let q = abs(p) - (h - vec3<f32>(r));
  return length(max(q, vec3<f32>(0f))) + min(max(q.x, max(q.y, q.z)), 0f) - r;
}

// Newton on the closest-point parameter of the meridional ellipse -- the
// same iteration, seed and 3-step count common_geometry.wgsl measured at
// 0.15% worst-case relative error in 2D. Axisymmetric about the local z.
fn sdSpheroid(p: vec3<f32>, a: f32, c: f32) -> f32 {
  let r0 = length(p.xy);
  if (abs(a - c) <= 1e-6f * max(a, c)) { return length(vec2<f32>(r0, p.z)) - a; }
  let x = abs(r0); let y = abs(p.z);
  let rr = sqrt((x*x)/(a*a) + (y*y)/(c*c));
  let inside = rr < 1.0f;
  var t = atan2(y * a, x * c);
  for (var i = 0u; i < 3u; i++) {
    let ct = cos(t); let st = sin(t);
    let ex = x - a * ct; let ey = y - c * st;
    let F  = ex * (-a * st) + ey * (c * ct);
    let Fp = -(a*a*st*st + c*c*ct*ct) + ex * (-a * ct) + ey * (-c * st);
    t = t - F / select(Fp, -1e-9f, abs(Fp) < 1e-9f);
    t = clamp(t, 0f, 1.5707963f);
  }
  let d = length(vec2<f32>(x - a * cos(t), y - c * sin(t)));
  return select(d, -d, inside);
}

// World-space signed distance. NOT periodically wrapped, unlike the 2D
// get_phi: every M2 scenario has a sponged open far field or walls, no
// moving window, and a body near enough to a face for its own periodic
// image to matter is a badly-sized domain -- which should be visible as a
// blockage error, not silently absorbed by a wrap.
fn get_phi3(p: vec3<f32>, s: BodyState3D) -> f32 {
  let d = p - vec3<f32>(s.cx, s.cy, s.cz);
  let lp = qRotInv(bodyQuat(s), d);
  let kind = u32(s.shape);
  if (kind == SHAPE_SPHERE) { return sdSphere(lp, s.a); }
  if (kind == SHAPE_SPHEROID) { return sdSpheroid(lp, s.a, s.c); }
  return sdRoundBox(lp, vec3<f32>(s.a, s.b, s.c), s.r);
}

// Local solid velocity at world point p: v + omega x r.
fn bodyVelocity3(p: vec3<f32>, s: BodyState3D) -> vec3<f32> {
  let r = p - vec3<f32>(s.cx, s.cy, s.cz);
  let w = vec3<f32>(s.wx, s.wy, s.wz);
  return vec3<f32>(s.vx, s.vy, s.vz) + cross(w, r);
}

// Shared tanh-blend core, matching common_geometry.wgsl's chiFromPhiEps
// exactly so the 2D and 3D diffuse couplings are the same method.
fn chiFromPhiEps3(phi: f32, epsilon: f32) -> f32 {
  return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}
