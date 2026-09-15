// Rigid body integration and state management.
// Runs once per LBM step to move the card smoothly.

// @include "common_geometry.wgsl"

@group(0) @binding(0) var<storage, read_write> state  : CardState;
@group(0) @binding(1) var<storage, read_write> forces : array<atomic<i32>, 4>;

override W : u32;
override H : u32;
const FSCALE = 10000000.0f;

// KINEMATIC ("reentry") mode -- mirrors shaders/amr_physics.wgsl's identical
// override byte-for-byte (see that file's own comment for the rationale).
// Default 0u is an exact no-op: main.js/main-cylinder.js, which don't
// declare these overrides at all, run the unmodified step-2/3
// Newton-integration-plus-clamp path below exactly as before this was
// added.
// Domain-heights of accumulated displacement allowed before x_total/y_total
// wrap -- see step 4b below, and card-total.mjs, which unwraps them back into
// a true running total on the host and MUST use the same number.
override TOTAL_WRAP_SCREENS : u32 = 16u;
override KINEMATIC : u32 = 0u;
override VY_FIXED : f32 = 0.0f;
override OMEGA_FIXED : f32 = 0.0f;

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

  if (KINEMATIC != 0u) {
    // 2/3. Prescribed kinematics -- no force feedback, no clamping.
    state.vx = 0.0f;
    state.vy = VY_FIXED;
    state.omega = OMEGA_FIXED;
  } else {
    // 2. Newton integration
    state.vx    += fx_fluid / state.mass;
    state.vy    += (fy_fluid + state.mass * state.g_eff) / state.mass;
    state.omega += tz_fluid / state.i_body;

    // 3. Clamping
    state.vx    = clamp(state.vx, -state.v_max, state.v_max);
    state.vy    = clamp(state.vy, -state.v_max, state.v_max);
    state.omega = clamp(state.omega, -state.o_max, state.o_max);
  }

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

  // 5. Moving Window Panning
  // We want to keep (cx, cy) near (W/2, H*2/3)
  let initial_cx = f32(W) / 2.0f;
  let initial_cy = f32(H) / 2.0f;

  if (WINDOW_BODY != 0u) {
    // THE SHIPPED CONVENTION. The body is pinned to a fixed WINDOW position;
    // off_x/off_y take floor(total) and cx/cy take the fraction, so the
    // body's sub-cell position IS the fractional part of an accumulator that
    // grows without bound. That is what card-total.mjs and TOTAL_WRAP_SCREENS
    // above exist to contain.
    let shift_x = i32(floor(state.x_total));
    let shift_y = i32(floor(state.y_total));

    state.off_x = f32((shift_x % i32(W) + i32(W)) % i32(W));
    state.off_y = f32((shift_y % i32(H) + i32(H)) % i32(H));

    state.cx = initial_cx + (state.x_total - f32(shift_x));
    state.cy = initial_cy + (state.y_total - f32(shift_y));
  } else {
    // ?window=0 -- THE BODY LIVES IN BUFFER COORDINATES (plans/2D-backport.md
    // B5, 3D's convention). Integrated here and wrapped into [0, W) x [0, H)
    // every step, so its ULP is fixed at W*2^-24 forever BY CONSTRUCTION
    // rather than tracking an accumulator. x_total/y_total keep accumulating
    // but are REPORTING ONLY from here on -- nothing reads them back into the
    // simulation, which is exactly 3D's arrangement.
    //
    // off_x/off_y are UNCHANGED, and that is a correction to the first
    // attempt at this: I derived them from the body's absolute position
    // (floor(cx - W/2)), on the assumption that the view is centred on the
    // body. IT IS NOT. main-cylinder.js places its cylinder UPSTREAM
    // diameters in, at cx = 170.67 with W = 512, and deriving the view from
    // that put off_x at 426 instead of 0 -- the sponge band landed in the
    // middle of the domain and Cd came back 808.897 against a literature
    // 1.35. Measured, not reasoned: the pinned cylinder is the config where
    // the two conventions MUST agree, so it caught it on the first run.
    //
    // The view tracks the body's TRAVEL, not its position, and travel is
    // exactly what x_total already measures. So off keeps its existing
    // derivation, and only the INTEGER part of x_total is ever consulted --
    // which is not the precision-sensitive quantity. The sub-cell position,
    // which is, no longer comes from that accumulator at all.
    state.cx = wrapf(state.cx + state.vx, f32(W));
    state.cy = wrapf(state.cy + state.vy, f32(H));
    let shift_x = i32(floor(state.x_total));
    let shift_y = i32(floor(state.y_total));
    state.off_x = f32((shift_x % i32(W) + i32(W)) % i32(W));
    state.off_y = f32((shift_y % i32(H) + i32(H)) % i32(H));
  }

  state.fx = fx_fluid;
  state.fy = fy_fluid;
  state.tz = tz_fluid;
}
