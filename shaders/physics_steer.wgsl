// Rigid body integration for the STEERED card (index-steer.html /
// main-steer.js) -- shaders/physics.wgsl plus a centre of mass that is NOT
// the geometric centre, driven live from the phone's roll (steer-imu.mjs).
// A fork rather than an override on physics.wgsl: it adds a binding, and
// physics.wgsl is bound by six dense pages, each with its own bind group --
// the shape that shipped 238e48c. Nothing else on any page reads this file.
//
// THE CONVENTION THAT KEEPS THIS TO ONE KERNEL. CardState's cx/cy/vx/vy stay
// the GEOMETRIC centre's position and velocity. Every other kernel reads them
// only as (a) the centre of the ellipse get_phi draws and (b) the reference
// point of the rigid velocity field u = v + omega x (p - c) and of the torque
// sum -- and rigid kinematics holds about ANY body point, so none of them
// needs to know where the mass is. lbm_force.wgsl's tz is the torque about
// the geometric centre G; only Newton's laws care about the centre of mass C,
// and they live here:
//
//   r   = R(theta) d                       C - G, world frame
//   v_C = v_G + omega x r
//   tau_C = tau_G - r x F  +  (-r) x F_b   F_b = buoyancy, acting at G
//   integrate v_C, omega at C (I about C held at I_BODY, i.e. I* fixed)
//   v_G = v_C - omega x r' - R(theta') (d' - d)
//
// The last term is the REACTION to the ballast sliding: C's velocity is
// continuous across a change of d, so the shell moves the other way. It is
// the whole of what a weight shift does to the translation, and without it
// moving d would teleport the centre of mass.
//
// BUOYANCY IS WHY THIS NEEDS F_b AT ALL. card-params.mjs folds it into
// G_EFF = G_LU (1 - 1/RHO_B), which is correct for the FORCE, and was also
// correct for the torque while C and G coincided. With them apart, weight
// (M G_LU at C) and buoyancy (M G_LU / RHO_B at G, upward) form a couple --
// the heavy end goes down. At the default I* = 0.17, RHO_B ~ 2.7, so
// buoyancy is ~37% of the weight: not a small correction.
//
// d = 0 REDUCES EXACTLY to physics.wgsl: every added term is a product with
// r or (d' - d), both zero, so a page that never steers runs index.html's
// trajectory value-for-value (tools/validate-steer.js checks it on the GPU).
//
// Only the CHORD component of d is modelled: the card is 8x thinner than
// long and a ballast off the chord line would sit outside the body.

// @include "common_geometry.wgsl"

@group(0) @binding(0) var<storage, read_write> state  : CardState;
@group(0) @binding(1) var<storage, read_write> forces : array<atomic<i32>, 4>;

// Host writes the first five words (main-steer.js's writeSteer), this kernel
// alone writes d. Separate words, so a host writeBuffer of the inputs never
// clobbers the kernel's state.
struct Steer {
  input  : f32,  // steering input s in [-1, 1]
  reach  : f32,  // |d| at s = +-1, lattice cells
  mode   : f32,  // 0: d = s*reach along the chord (BODY frame)
                 // 1: d = s*reach*cos(theta) -- the WORLD-horizontal shift
                 //    projected onto the chord: steer right, weight goes
                 //    to whichever end is on the right (hang-glider)
  slew   : f32,  // max |change in d| per step, lattice cells
  fb     : f32,  // buoyancy magnitude M * G_LU / RHO_B
  d      : f32,  // CURRENT chord offset of C from G, lattice cells
  dd     : f32,  // the last step's change in d -- the ballast's velocity
  _p1    : f32,
}
@group(0) @binding(2) var<storage, read_write> steer : Steer;

override W : u32;
override H : u32;
const FSCALE = 10000000.0f;

// Domain-heights of accumulated displacement allowed before x_total/y_total
// wrap -- see step 4b below, and card-total.mjs, which unwraps them back into
// a true running total on the host and MUST use the same number.
override TOTAL_WRAP_SCREENS : u32 = 16u;

@compute @workgroup_size(1)
fn main() {
  // 0. Save current as old for the next step
  state.cx_old = state.cx;
  state.cy_old = state.cy;
  state.th_old = state.theta;
  state.off_x_old = state.off_x;
  state.off_y_old = state.off_y;

  // 1. Read accumulated impulse from atomic buffer
  let fx_fluid = f32(atomicExchange(&forces[0], 0)) / FSCALE;
  let fy_fluid = f32(atomicExchange(&forces[1], 0)) / FSCALE;
  let tz_fluid = f32(atomicExchange(&forces[2], 0)) / FSCALE;

  // 2. Newton integration, AT THE CENTRE OF MASS. The forces were
  // evaluated at the pose the step started from, so r is too.
  let d0 = steer.d;
  let c0 = cos(state.theta);
  let s0 = sin(state.theta);
  let r0 = d0 * vec2<f32>(c0, s0);
  // v_C = v_G + omega x r + R d', the exact inverse of 2c below. The last
  // term is not optional: leaving the ballast's own velocity out makes every
  // step's reaction kick PERMANENT -- G's velocity is read back as C's, so C
  // loses dd of momentum per step for as long as d is moving. Measured: vx
  // ran to the 0.3 clamp in 32 steps and the run went non-finite.
  var vcx = state.vx - state.omega * r0.y + steer.dd * c0;
  var vcy = state.vy + state.omega * r0.x + steer.dd * s0;
  // r x F as a z-component, the same sign convention as the force
  // kernels' tz = rx * fy - ry * fx. Buoyancy acts at G, i.e. at -r from
  // C, pointing -y (gravity is +y here): (-r) x (0, -fb) = r.x * fb.
  let tz_c = tz_fluid - (r0.x * fy_fluid - r0.y * fx_fluid) + r0.x * steer.fb;

  vcx         += fx_fluid / state.mass;
  vcy         += (fy_fluid + state.mass * state.g_eff) / state.mass;
  state.omega += tz_c / state.i_body;
  state.omega = clamp(state.omega, -state.o_max, state.o_max);

  // 2b. Slide the ballast toward the input, rate-limited: an unlimited
  // jump in d is an unlimited kick to G through the reaction term below.
  let th1 = state.theta + state.omega;
  let c1 = cos(th1);
  let s1 = sin(th1);
  var tgt = steer.input * steer.reach;
  if (steer.mode != 0.0f) { tgt *= c1; }
  let d1 = d0 + clamp(tgt - d0, -steer.slew, steer.slew);
  steer.d = d1;
  steer.dd = d1 - d0;
  let r1 = d1 * vec2<f32>(c1, s1);
  let dd = steer.dd;

  // 2c. Back to the geometric centre -- see the header.
  state.vx = vcx + state.omega * r1.y - dd * c1;
  state.vy = vcy - state.omega * r1.x - dd * s1;

  // 3. Clamping
  state.vx    = clamp(state.vx, -state.v_max, state.v_max);
  state.vy    = clamp(state.vy, -state.v_max, state.v_max);

  // 4. Position update (absolute)
  state.y_total += state.vy;
  state.x_total += state.vx;
  state.theta   += state.omega;

  // 4b. Keep the accumulators bounded. These grow without limit otherwise,
  // and they are f32, so their ULP grows with them -- which eats the
  // FRACTIONAL part these very lines hand to cx/cy below as the card's
  // sub-cell position, quantizing the body (and so get_phi, and so the whole
  // solid coupling) coarser and coarser the longer a page runs. See
  // card-total.mjs for the full argument and the measured scale.
  //
  // Wrapping by a whole multiple of the domain size is invisible to
  // everything downstream: off_x/off_y take floor(total) mod W/H, which a
  // K*W shift leaves alone, and cx/cy take the fractional part, which
  // subtracting an integer leaves alone. TOTAL_WRAP_SCREENS and the domain
  // are both powers of two and the wrap fires only just past the threshold,
  // so the subtraction is exact in f32 -- this adds no error, it only stops
  // the existing one growing. Wrapping toward zero (not into [0, wrap))
  // keeps both operands within a factor of two, which is what makes it
  // exact in the negative case too.
  let wrap_x = f32(W * TOTAL_WRAP_SCREENS);
  let wrap_y = f32(H * TOTAL_WRAP_SCREENS);
  if (state.x_total >= wrap_x) { state.x_total -= wrap_x; }
  else if (state.x_total <= -wrap_x) { state.x_total += wrap_x; }
  if (state.y_total >= wrap_y) { state.y_total -= wrap_y; }
  else if (state.y_total <= -wrap_y) { state.y_total += wrap_y; }

  // 5. THE BODY'S POSITION, AND THE VIEW, ARE NOW SEPARATE THINGS.
  //
  // The body is integrated in BUFFER coordinates and wrapped into
  // [0, W) x [0, H) every step, so its magnitude -- and therefore the ULP of
  // its sub-cell position, which get_phi and the whole solid coupling inherit
  // -- is bounded by the domain, not by how long the page has been running.
  //
  // THE VIEW TRACKS TRAVEL, NOT POSITION, and that distinction cost a
  // measurement to find. Deriving off from the body's absolute position
  // (floor(cx - W/2)) assumes the view is centred on the body. It is not:
  // main-cylinder.js places its cylinder UPSTREAM diameters in, at cx = 170.67
  // with W = 512, so that put off_x at 426, the sponge band landed mid-domain
  // and Cd came back 808.897 against a literature 1.35. off therefore keeps
  // deriving from x_total, and only its INTEGER part is ever consulted.
  //
  // So x_total/y_total are still load-bearing for the VIEW (and for the trail
  // and CSV export, via card-total.mjs) but no longer for the BODY. That is
  // the half of card-total.mjs's rationale that B5 retired -- see its header.
  state.cx = wrapf(state.cx + state.vx, f32(W));
  state.cy = wrapf(state.cy + state.vy, f32(H));
  let shift_x = i32(floor(state.x_total));
  let shift_y = i32(floor(state.y_total));
  state.off_x = f32((shift_x % i32(W) + i32(W)) % i32(W));
  state.off_y = f32((shift_y % i32(H) + i32(H)) % i32(H));

  state.fx = fx_fluid;
  state.fy = fy_fluid;
  state.tz = tz_fluid;
}
