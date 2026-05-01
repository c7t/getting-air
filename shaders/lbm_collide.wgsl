// D2Q9 Regularized BGK collision with GPU-side solid refilling.
// Uses an analytical distance function to determine the solid mask.

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
}

@group(0) @binding(0) var<storage, read>       state : CardState;
@group(0) @binding(1) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_col : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel   : array<f32>;

const W   = 256u;
const H   = 512u;
const CS2 = 0.33333333f;
const CS4 = 0.11111111f;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

fn is_solid(p: vec2<f32>, cx: f32, cy: f32, theta: f32, a: f32, b: f32) -> bool {
    let ca = cos(theta);
    let sa = sin(theta);
    var dx = p.x - cx;
    var dy = p.y - cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    return (lx*lx)/(a*a) + (ly*ly)/(b*b) <= 1.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell = y * W + x;
  let base = cell * 9u;
  let p = vec2<f32>(f32(x), f32(y));

  let solid_now = is_solid(p, state.cx, state.cy, state.theta, state.a, state.b);
  let solid_old = is_solid(p, state.cx_old, state.cy_old, state.th_old, state.a, state.b);

  if (solid_now) { return; }

  // Refilling: if was solid and is now fluid, initialize with LOCAL card velocity
  if (solid_old) {
    // Local velocity at point p: v_cm + omega x (p - cx)
    var rx = p.x - state.cx;
    var ry = p.y - state.cy;
    rx -= f32(W) * round(rx / f32(W));
    ry -= f32(H) * round(ry / f32(H));
    
    let ulx = state.vx - state.omega * ry;
    let uly = state.vy + state.omega * rx;
    let u2  = ulx*ulx + uly*uly;

    for (var i = 0u; i < 9u; i++) {
      let exf = f32(ex[i]); let eyf = f32(ey[i]);
      let eu  = exf * ulx + eyf * uly;
      f_col[base + i] = wt[i] * 1.0f * (1f + eu/CS2 + eu*eu/(2f*CS2*CS2) - u2/(2f*CS2));
    }
    vel[cell * 2u] = ulx; vel[cell * 2u + 1u] = uly;
    return;
  }

  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) { f[i] = f_in[base + i]; }

  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= rho; uy /= rho;

  if (rho < 0.5f || rho > 2.0f) {
    for (var i = 0u; i < 9u; i++) { f_col[base + i] = wt[i]; }
    return;
  }

  // Velocity cap (safety)
  let u_sq = ux*ux + uy*uy;
  if (u_sq > 0.09f) {
    let s = 0.3f / sqrt(u_sq);
    ux *= s; uy *= s;
  }

  vel[cell * 2u] = ux; vel[cell * 2u + 1u] = uy;

  var Pxx = -(rho * (ux*ux + CS2));
  var Pyy = -(rho * (uy*uy + CS2));
  var Pxy = -(rho * ux * uy);
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    Pxx += f[i] * exf * exf;
    Pyy += f[i] * eyf * eyf;
    Pxy += f[i] * exf * eyf;
  }

  let omgT = 1f - 1f / state.tau;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + eu/CS2 + eu*eu/(2f*CS2*CS2) - u_sq/(2f*CS2));
    let f_neq_reg = wt[i] * 4.5f * ((exf*exf - CS2)*Pxx + 2f*exf*eyf*Pxy + (eyf*eyf - CS2)*Pyy);
    f_col[base + i] = feq + omgT * f_neq_reg;
  }
}
