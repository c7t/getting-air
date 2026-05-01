// Rigid body integration and state management.
// Runs once per LBM step to move the card smoothly.

struct CardState {
  cx     : f32,
  cy     : f32,
  theta  : f32,
  vx     : f32,
  vy     : f32,
  omega  : f32,
  fx     : f32,
  fy     : f32,
  tz     : f32,
  mass   : f32,
  i_body : f32,
  g_eff  : f32,
  a      : f32,  // semi-major axis
  b      : f32,  // semi-minor axis
  v_max  : f32,
  o_max  : f32,
  cx_old : f32,
  cy_old : f32,
  th_old : f32,
  tau    : f32,
  y_total: f32,
  _p1    : f32,
}

@group(0) @binding(0) var<storage, read_write> state  : CardState;
@group(0) @binding(1) var<storage, read_write> forces : array<atomic<i32>, 4>;

const W = 256.0f;
const H = 512.0f;
const FSCALE = 1000.0f;

@compute @workgroup_size(1)
fn main() {
  // 0. Save current as old for the next step's transition refill
  state.cx_old = state.cx;
  state.cy_old = state.cy;
  state.th_old = state.theta;

  // 1. Read accumulated impulse from atomic buffer
  let fx_fluid = f32(atomicExchange(&forces[0], 0)) / FSCALE;
  let fy_fluid = f32(atomicExchange(&forces[1], 0)) / FSCALE;
  let tz_fluid = f32(atomicExchange(&forces[2], 0)) / FSCALE;

  // 2. Newton integration
  state.vx    += fx_fluid / state.mass;
  state.vy    += (fy_fluid + state.mass * state.g_eff) / state.mass;
  state.omega += tz_fluid / state.i_body;

  // 3. Clamping
  state.vx    = clamp(state.vx, -state.v_max, state.v_max);
  state.vy    = clamp(state.vy, -state.v_max, state.v_max);
  state.omega = clamp(state.omega, -state.o_max, state.o_max);

  // 4. Position update
  state.cx    += state.vx;
  state.cy    += state.vy;
  state.theta += state.omega;
  state.y_total += state.vy;

  // 5. Toroidal wrapping
  state.cx = (state.cx % W + W) % W;
  state.cy = (state.cy % H + H) % H;
  
  state.fx = fx_fluid;
  state.fy = fy_fluid;
  state.tz = tz_fluid;
}
