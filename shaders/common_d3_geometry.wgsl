// 6-DOF rigid-body state and TRUE signed-distance geometry for the dense 3D
// solver -- the solid mask every step/force/render kernel tests against.
// plans/3D.md M2. Fragment only; included via
// `// @include "common_d3_geometry.wgsl"`, and it needs
// common_d3_window.wgsl listed BEFORE it -- bodyDelta3 takes the nearest
// periodic image of p - c on any axis the moving window is enabled on.
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
  // TOTAL DISPLACEMENT since reset, and REPORTING ONLY (M8.3). Under the
  // moving window cx/cy/cz are wrapped into the buffer every step, so "how
  // far has it fallen" is no longer readable from the position -- this is,
  // and it is the only thing that is.
  //
  // NOTHING IN THE SIMULATION READS IT BACK, which is what makes its
  // unbounded f32 growth harmless here where the 2D card's identical
  // accumulator was not: card-total.mjs exists because the 2D card's
  // SUB-CELL POSITION is the fractional part of this quantity, so a growing
  // ULP quantizes the body itself. Here the sub-cell position lives in cx,
  // which is bounded by the wrap, and the worst this can cost is a fraction
  // of a percent of a reported travel distance.
  dx : f32, dy : f32, dz : f32,
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

// Position of p RELATIVE TO the body centre, taking the nearest periodic
// image on any axis the moving window is enabled on (M8.3). ONE function,
// used by the SDF, by the local solid velocity and by the force kernels'
// torque arm, because all three ask the same question and a body that
// straddled the buffer seam in two of them and not the third would present
// as a torque with no force behind it.
//
// WITHOUT A WINDOW THIS IS EXACTLY `p - c` (winWrapDelta is the identity at
// WIN_N* = 0, the default), which is what keeps every scenario that predates
// M8.3 bit-identical rather than merely unaffected in practice. The
// no-window reading was deliberate and is worth keeping: a body near enough
// to a face for its own periodic image to matter is a badly-sized domain,
// and that should be visible as a blockage error rather than silently
// absorbed. A windowed axis is the one case where crossing the seam is the
// intended behaviour and not a mistake.
fn bodyDelta3(p: vec3<f32>, s: BodyState3D) -> vec3<f32> {
  return winWrapDelta(p - vec3<f32>(s.cx, s.cy, s.cz));
}

// THE SHAPE DISPATCH, on a position already in the BODY frame. Factored out
// of get_phi3 so that get_phi3Ahead below can ask the same question about the
// same body at a different POSE without a second copy of the shape table --
// a shape added to one and not the other would make the swept-cell term see a
// different body from the one the fluid is bouncing off.
fn sdfLocal3(lp: vec3<f32>, s: BodyState3D) -> f32 {
  let kind = u32(s.shape);
  if (kind == SHAPE_SPHERE) { return sdSphere(lp, s.a); }
  if (kind == SHAPE_SPHEROID) { return sdSpheroid(lp, s.a, s.c); }
  return sdRoundBox(lp, vec3<f32>(s.a, s.b, s.c), s.r);
}

// World-space signed distance.
fn get_phi3(p: vec3<f32>, s: BodyState3D) -> f32 {
  return sdfLocal3(qRotInv(bodyQuat(s), bodyDelta3(p, s)), s);
}

// THE SAME SIGNED DISTANCE, ONE STEP AHEAD -- the body advanced by `dt` at
// its current velocity and angular velocity, with everything else held.
//
// WHAT IT IS FOR (plans/3D.md D1). A moving body's discrete surface is a
// staircase that CHANGES, so every step a handful of cells cross between
// fluid and solid. The momentum those cells carry is created or destroyed,
// and the momentum exchange over the bounce-back links cannot see it -- it
// only sees what crosses a link. Charging the body for it needs to know
// WHICH cells are about to change hands, and that is this function against
// get_phi3: opposite signs means this cell changes hands on the next step.
//
// FORWARD, NOT BACKWARD, and the macro-step order is why. Per step the order
// is force -> physics -> step (main-3d.js's encodeSteps), so the force kernel
// sees the body at the pose the PREVIOUS step used and `f_in` still holding
// real fluid in the cells the NEXT step is about to bury. Looking back
// instead would name the cells whose data was already overwritten.
//
// PREDICTED WITH THE CURRENT VELOCITY, so it is EXACT for a body on rails
// (?tow=, ?stream=, ?dynamic= drift) and first order in the fluid force for a
// free one -- physics integrates v += F/m before moving, and F/m is ~3e-4 of
// v on every case in this suite. The geometry criterion already refines ahead
// on exactly this prediction (common_d3_manage.wgsl's MARGIN + manageEvery *
// |v|), so this is the same statement made in a second place, not a new
// assumption.
//
// The two unit quaternions multiply to a unit quaternion, so there is no
// renormalization here; this is a one-step geometric probe and not the
// integrator, which takes its own midpoint step on the rotation group.
fn get_phi3Ahead(p: vec3<f32>, s: BodyState3D, dt: f32) -> f32 {
  let c = vec3<f32>(s.cx, s.cy, s.cz) + vec3<f32>(s.vx, s.vy, s.vz) * dt;
  let q = qMul(expMapQ(vec3<f32>(s.wx, s.wy, s.wz), dt), bodyQuat(s));
  return sdfLocal3(qRotInv(q, winWrapDelta(p - c)), s);
}

// THE RADIUS OF THE SMALLEST SPHERE ABOUT THE CENTRE THAT CONTAINS THE BODY.
//
// Exists to bound how fast ROTATION can bring the surface toward a point that
// is standing still -- see blockWanted in common_d3_manage.wgsl. A surface
// point at body-frame radius r moves at |v + omega x r| <= |v| + |omega| * R,
// and phi is the distance to the NEAREST surface point, so |dphi/dt| is
// bounded by that same speed. One scalar, no second SDF evaluation.
//
// The rounded box's corner sits at sqrt((a-r)^2 + (b-r)^2 + (c-r)^2) + r,
// which is at most length(a,b,c) -- the corner radius only rounds INWARD --
// so the unrounded half-diagonal is a safe bound and is what is returned.
// d3-body.mjs's bodyCircumradius is the host statement of the same thing and
// make test scores them against a brute-force search over each surface.
fn bodyCircumradius3(s: BodyState3D) -> f32 {
  let kind = u32(s.shape);
  if (kind == 0u) { return s.a; }                        // SPHERE
  if (kind == 1u) { return max(s.a, s.c); }              // SPHEROID (a, a, c)
  return length(vec3<f32>(s.a, s.b, s.c));               // ROUNDBOX
}

// Local solid velocity at world point p: v + omega x r.
fn bodyVelocity3(p: vec3<f32>, s: BodyState3D) -> vec3<f32> {
  let r = bodyDelta3(p, s);
  let w = vec3<f32>(s.wx, s.wy, s.wz);
  return vec3<f32>(s.vx, s.vy, s.vz) + cross(w, r);
}

// Shared tanh-blend core, matching common_geometry.wgsl's chiFromPhiEps
// exactly so the 2D and 3D diffuse couplings are the same method.
fn chiFromPhiEps3(phi: f32, epsilon: f32) -> f32 {
  return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}
