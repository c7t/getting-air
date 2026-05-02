// Volume Penalization IB-LBM (Direct Forcing) with Guo's scheme and Smagorinsky LES.

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

@group(0) @binding(0) var<storage, read>       state : CardState;
@group(0) @binding(1) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_col : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel   : array<f32>;

const W   = 512u;
const H   = 1024u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

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

  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) { f[i] = f_in[base + i]; }

  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  ux_star /= rho; uy_star /= rho;
  
  // Local solid velocity Us
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));
  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;
  
  let phi = get_phi(p, state);
  let chi = get_chi(phi);
  
  // Penalty Force F = rho * chi * (Us - u*)
  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);
  
  // Actual fluid velocity u = u* + F/(2rho)
  let ux = ux_star + Fx / (2.0f * rho);
  let uy = uy_star + Fy / (2.0f * rho);
  let u_sq = ux*ux + uy*uy;

  vel[cell * 2u] = ux; vel[cell * 2u + 1u] = uy;

  // 1. Calculate equilibrium and non-equilibrium tensor
  var feq: array<f32,9>;
  var Pxx = 0.0f; var Pyy = 0.0f; var Pxy = 0.0f;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    feq[i] = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);
    
    let fneq = f[i] - feq[i];
    Pxx += fneq * exf * exf;
    Pyy += fneq * eyf * eyf;
    Pxy += fneq * exf * eyf;
  }

  // 2. Calculate local strain magnitude Q = sqrt(2 * sum(Pi_neq^2))
  let Q = sqrt(2.0f * (Pxx*Pxx + Pyy*Pyy + 2.0f*Pxy*Pxy));

  // 3. Calculate Eddy Relaxation Time (Smagorinsky model)
  let Cs = 0.4f;
  let tau0 = state.tau;
  // tau_total = tau0 + 0.5 * (sqrt(tau0^2 + (18*Cs^2 / cs2) * Q) - tau0)
  // With cs2 = 1/3, 18/cs2 = 54.
  let tau_total = tau0 + 0.5f * (sqrt(tau0*tau0 + 54.0f * Cs*Cs * Q) - tau0);
  let omg = 1.0f / tau_total;

  // 4. Collide using tau_total
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    
    // Guo's Source Term Si using locally varying tau_total
    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );
    
    f_col[base + i] = f[i] - omg * (f[i] - feq[i]) + Si;
  }
}
