// Rigid body integration and state management.
// Runs once per LBM step to move the card smoothly.

// @include "common_geometry.wgsl"

@group(0) @binding(0) var<storage, read_write> state  : CardState;
@group(0) @binding(1) var<storage, read_write> forces : array<atomic<i32>, 4>;

override W : u32;
override H : u32;
const FSCALE = 10000.0f;

// KINEMATIC ("reentry") mode: main-reentry-amr.js's own scenario, decoupling
// the body's motion from fluid force feedback entirely -- vx/vy/omega are
// PRESCRIBED constants (straight-down descent at a fixed sink rate, fixed
// tumble rate) instead of the usual force-integrated free-fall/tumble.
// Default 0u is an exact no-op: every other page (main-amr.js,
// main-cylinder-amr.js, the channel/TGV harnesses, none of which declare
// these overrides at all and so get WGSL's own default) runs the unmodified
// step-2/3 Newton-integration-plus-clamp path below, byte-for-byte as
// before this was added. Fluid forces are still drained from the atomic
// buffer and recorded into state.fx/fy/tz either way (below) -- KINEMATIC
// mode still reports what force the body WOULD feel, for diagnostics, it
// just never feeds back into the motion.
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
    // 2/3. Prescribed kinematics -- no force feedback, no clamping (nothing
    // to clamp: these ARE the velocities, by definition never exceeding
    // themselves).
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

  // 5. Moving Window Panning
  // We want to keep (cx, cy) near (W/2, H*2/3)
  let initial_cx = f32(W) / 2.0f;
  let initial_cy = f32(H) / 2.0f;

  let shift_x = i32(floor(state.x_total));
  let shift_y = i32(floor(state.y_total));

  state.off_x = f32((shift_x % i32(W) + i32(W)) % i32(W));
  state.off_y = f32((shift_y % i32(H) + i32(H)) % i32(H));

  state.cx = initial_cx + (state.x_total - f32(shift_x));
  state.cy = initial_cy + (state.y_total - f32(shift_y));

  state.fx = fx_fluid;
  state.fy = fy_fluid;
  state.tz = tz_fluid;
}
