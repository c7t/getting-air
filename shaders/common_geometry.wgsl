// Shared rigid-body state and ellipse signed-distance geometry -- the
// solid mask every step/force/manage/render kernel tests against.
// Included via `// @include "common_geometry.wgsl"`. Depends on the
// including file's own `override W : u32;`/`override H : u32;` (WGSL
// module-scope name resolution is order-independent, so it doesn't matter
// whether those overrides are declared before or after this include).

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
  a      : f32,
  b      : f32,
  v_max  : f32,
  o_max  : f32,
  cx_old : f32,
  cy_old : f32,
  th_old : f32,
  tau    : f32,
  y_total: f32,
  x_total: f32,
  off_x  : f32,
  off_y  : f32,
  off_x_old : f32,
  off_y_old : f32,
}

// Algebraic (non-Euclidean, scaled-by-semi-minor-axis) distance
// approximation for an ellipse centered at (state.cx, state.cy), rotated
// by state.theta, semi-axes (state.a, state.b). Positive outside, ~0 at
// the boundary. `p` and the state's center are periodically wrapped
// against the W/H domain before rotating into the body frame.
fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    var dx = p.x - state.cx;
    var dy = p.y - state.cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b;
}

// Shared tanh-blend core of every get_chi -- callers compute their own
// epsilon (it varies: fixed 1.5 at L0/L1, K_EPS*levelParams.dxL at
// level>=2's shared pipelines) and pass it in here.
fn chiFromPhiEps(phi: f32, epsilon: f32) -> f32 {
    // Clamp tanh arg: large |arg| overflows to NaN on some GPUs (e.g. Intel Gen12LP); saturated regime is unchanged. See PR.
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}
