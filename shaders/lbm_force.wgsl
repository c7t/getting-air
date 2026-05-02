// Force and torque on the solid body via integration of the penalty force.

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

@group(0) @binding(0) var<storage, read>       state  : CardState;
@group(0) @binding(1) var<storage, read>       f_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;

const W      = 512u;
const H      = 1024u;
const FSCALE = 1000f;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

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

fn get_chi(phi: f32) -> f32 {
    let epsilon = 1.5f;
    return 0.5f * (1.0f - tanh(phi / epsilon));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell = y * W + x;
  let base = cell * 9u;
  let p = vec2<f32>(f32(x), f32(y));

  let phi = get_phi(p, state);
  let chi = get_chi(phi);
  if (chi < 1e-6) { return; }

  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    let fi = f_in[base + i];
    rho     += fi;
    ux_star += fi * f32(ex[i]);
    uy_star += fi * f32(ey[i]);
  }
  ux_star /= rho; uy_star /= rho;

  // Local solid velocity Us
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));
  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  // Penalty Force F = rho * chi * (Us - u*)
  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  // Integrate NEGATIVE of penalty force onto body
  let fx_body = -Fx;
  let fy_body = -Fy;
  let tz_body = rx * fy_body - ry * fx_body;

  atomicAdd(&forces[0], i32(fx_body * FSCALE));
  atomicAdd(&forces[1], i32(fy_body * FSCALE));
  atomicAdd(&forces[2], i32(tz_body * FSCALE));
}
