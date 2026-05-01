// Force and torque on the solid body via momentum exchange with analytical mask.

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

@group(0) @binding(0) var<storage, read>       state  : CardState;
@group(0) @binding(1) var<storage, read>       f_col  : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;

const W      = 512u;
const H      = 1024u;
const CS2    = 0.33333333f;
const FSCALE = 1000f;
const FI_MAX = 0.6f;

const ex  = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey  = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt  = array<f32,9>(
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
  let p = vec2<f32>(f32(x), f32(y));

  if (is_solid(p, state.cx, state.cy, state.theta, state.a, state.b)) { return; }

  let base = cell * 9u;
  for (var i = 1u; i < 9u; i++) {
    let nx = u32((i32(x) + ex[i] + i32(W)) % i32(W));
    let ny = u32((i32(y) + ey[i] + i32(H)) % i32(H));
    let np = vec2<f32>(f32(nx), f32(ny));

    if (is_solid(np, state.cx, state.cy, state.theta, state.a, state.b)) {
      let fi = clamp(f_col[base + i], 0f, FI_MAX);
      let mx = f32(x) + 0.5f * f32(ex[i]);
      let my = f32(y) + 0.5f * f32(ey[i]);
      var rx = mx - state.cx;
      var ry = my - state.cy;
      rx -= f32(W) * round(rx / f32(W));
      ry -= f32(H) * round(ry / f32(H));
      let ubx = state.vx - state.omega * ry;
      let uby = state.vy + state.omega * rx;
      let ei_ub = f32(ex[i]) * ubx + f32(ey[i]) * uby;
      let mag = 2f * fi - 2f * wt[i] * ei_ub / CS2;
      let fx = mag * f32(ex[i]);
      let fy = mag * f32(ey[i]);
      let tz = rx * fy - ry * fx;
      atomicAdd(&forces[0], i32(fx * FSCALE));
      atomicAdd(&forces[1], i32(fy * FSCALE));
      atomicAdd(&forces[2], i32(tz * FSCALE));
    }
  }
}
