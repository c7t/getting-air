// 6-DOF rigid-body integration for the dense 3D solver. plans/3D.md M2.
// Runs once per LBM step, @workgroup_size(1) -- like the 2D physics.wgsl,
// this is a scalar ODE step, not a field operation.
//
// THE HOST REFERENCE IS d3-body.mjs's stepFreeBody(), and this mirrors it
// line for line. That is not an aspiration: the `spin` scenario runs this
// exact kernel with the fluid force switched off and
// tools/lib/d3-body-metrics.js compares the GPU trajectory against the host
// one, so a divergence is a test failure rather than something noticed
// later in a tumbling plate.
//
// WHY ANGULAR MOMENTUM, NOT AN EXPLICIT GYROSCOPIC TERM. plans/3D.md sec
// 1.3 names omega x (I omega) as the new term:
//     omega_dot = I_world^-1 (torque - omega x (I_world omega)).
// Correct, but the worse of two equivalent formulations. Integrating L
// directly,
//     L    += torque
//     omega = R I_body^-1 R^T L
// makes L exactly conserved at zero torque -- to floating point, not to
// truncation order -- and the gyroscopic effect emerges from R rotating
// inside I_world, which is where it physically comes from. No explicit
// cross product means no place for a sign error that stays invisible until
// a body tumbles. The tumbling physics (the intermediate-axis instability
// that makes a falling plate interesting at all) is identical, and
// tools/test-d3-body.js asserts it appears.
//
// omega is not integrated but CACHED into the state here, so the step and
// force kernels can read it without each redoing rotate-divide-rotate per
// cell.

// @include "common_d3_geometry.wgsl"

@group(0) @binding(0) var<storage, read_write> body   : BodyState3D;
@group(0) @binding(1) var<storage, read_write> forces : array<atomic<i32>, 8>;

const FSCALE = 10000000.0f;

// Uniform acceleration (gravity, buoyancy-corrected). An ACCELERATION, not
// a force, matching the 2D physics.wgsl's g_eff.
override GX : f32 = 0.0f;
override GY : f32 = 0.0f;
override GZ : f32 = 0.0f;

// Ignore the fluid force entirely and integrate the body alone. This is
// what makes the integrator testable in isolation: the `spin` scenario uses
// it to compare this kernel against d3-body.mjs on real GPU code, with no
// fluid coupling in the way. Default 0 = normal coupled operation.
override NO_FLUID_FORCE : u32 = 0u;

@compute @workgroup_size(1)
fn main() {
  // Drain the accumulated impulse regardless of what is done with it --
  // leaving it in place under a pinned or force-free body would let it
  // accumulate across steps and appear as a spurious kick the moment
  // anything reads it.
  let fx = f32(atomicExchange(&forces[0], 0)) / FSCALE;
  let fy = f32(atomicExchange(&forces[1], 0)) / FSCALE;
  let fz = f32(atomicExchange(&forces[2], 0)) / FSCALE;
  let tx = f32(atomicExchange(&forces[3], 0)) / FSCALE;
  let ty = f32(atomicExchange(&forces[4], 0)) / FSCALE;
  let tz = f32(atomicExchange(&forces[5], 0)) / FSCALE;

  // Recorded BEFORE the pinned early-return: a pinned body still has its
  // force measured -- that is the whole point of the sphere validation
  // scenario -- it just does not move, and this is the channel Cd is read
  // back through.
  body.fx = fx; body.fy = fy; body.fz = fz;
  body.tx = tx; body.ty = ty; body.tz = tz;
  if (body.pinned != 0f) { return; }

  let useFluid = select(1.0f, 0.0f, NO_FLUID_FORCE != 0u);
  let force  = vec3<f32>(fx, fy, fz) * useFluid;
  let torque = vec3<f32>(tx, ty, tz) * useFluid;

  // 1. Linear: dt = 1, so every `+=` is one lattice step (the same
  // convention the 2D physics.wgsl uses).
  var v = vec3<f32>(body.vx, body.vy, body.vz)
        + force / body.mass + vec3<f32>(GX, GY, GZ);
  // MAGNITUDE clamp, not the 2D kernel's per-component one. Clamping
  // components independently changes the DIRECTION of the vector, which in
  // 2D was a limiter on a runaway and in 3D would silently steer a tumbling
  // body. These limiters exist to contain a blowup, not to shape the
  // trajectory, so they must not rotate anything.
  let vmag = length(v);
  if (vmag > body.v_max) { v = v * (body.v_max / vmag); }

  // 2. Angular momentum, then the orientation it implies.
  let q = bodyQuat(body);
  // omega = R I_body^-1 R^T L, written rotate-divide-rotate rather than by
  // forming and inverting a 3x3 I_world: same result, no matrix, and no
  // ill-conditioned inverse for a very anisotropic plate (Iz/Ix is large at
  // the falling card's aspect ratio).
  var Ln = vec3<f32>(body.Lx, body.Ly, body.Lz) + torque;
  var w = omegaFromL(q, body, Ln);
  // Magnitude clamp again, and applied to L as well as omega -- L is the
  // integrated state here, so clamping only the derived omega would leave
  // the two inconsistent and the limiter would do nothing that persists.
  // Scaling both by the same factor preserves the rotation AXIS, which
  // per-component clamping would not.
  let wmag = length(w);
  if (wmag > body.o_max) {
    let k = body.o_max / wmag;
    w = w * k; Ln = Ln * k;
  }

  // 3. Orientation update, from the WORLD-frame omega implied by the NEW L.
  //
  // MIDPOINT ON THE ROTATION GROUP, mirroring d3-body.mjs, and load-bearing
  // rather than fastidious. For a free body L is constant but omega is not
  // -- it is R I_body^-1 R^T L, changing as the body turns, which IS the
  // gyroscopic effect. Freezing omega at the start of the interval is
  // first-order in that variation and was measured on the host reference at
  // 58.87% rotational-energy drift over 20000 steps, and -- far worse --
  // it made a body spun about its MAJOR axis flip, an instability that does
  // not exist. |L| is exactly conserved the whole time, so no conservation
  // check would have caught it. Evaluating omega at the half step takes the
  // same measurement to 0.0052%. Two extra rotate-divide-rotates in a
  // once-per-step @workgroup_size(1) kernel: free.
  //
  // Each partial rotation uses the exponential map rather than the
  // first-order tangent step `q + (dt/2) w_quat q` -- unit-norm by
  // construction and exact for a constant omega over its interval.
  let qHalf = normalize(qMul(expMapQ(w, 0.5f), q));
  let wHalf = omegaFromL(qHalf, body, Ln);
  let qn = normalize(qMul(expMapQ(wHalf, 1.0f), q));

  // 4. Position.
  let x = vec3<f32>(body.cx, body.cy, body.cz) + v;

  body.cx = x.x; body.cy = x.y; body.cz = x.z;
  body.vx = v.x; body.vy = v.y; body.vz = v.z;
  body.Lx = Ln.x; body.Ly = Ln.y; body.Lz = Ln.z;
  body.wx = wHalf.x; body.wy = wHalf.y; body.wz = wHalf.z;
  body.qw = qn.x; body.qx = qn.y; body.qy = qn.z; body.qz = qn.w;
}
