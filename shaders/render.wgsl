// Visualization shader with analytical solid mask.

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
  _p0    : f32,
}

@group(0) @binding(0) var<storage, read> vel   : array<f32>;
@group(0) @binding(1) var<storage, read> state : CardState;

const W = 256.0f;
const H = 512.0f;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  let p = array<vec2<f32>,6>(
    vec2(-1f,-1f), vec2( 1f,-1f), vec2(-1f, 1f),
    vec2(-1f, 1f), vec2( 1f,-1f), vec2( 1f, 1f)
  );
  var out: VSOut;
  out.pos = vec4(p[vi], 0f, 1f);
  out.uv  = p[vi] * 0.5f + 0.5f;
  return out;
}

fn is_solid(p: vec2<f32>, cx: f32, cy: f32, theta: f32, a: f32, b: f32) -> bool {
    let ca = cos(theta);
    let sa = sin(theta);
    let half_len = a - b;
    var dx = p.x - cx;
    var dy = p.y - cy;
    dx -= W * round(dx / W);
    dy -= H * round(dy / H);
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    let cap_dist = max(0.0, abs(lx) - half_len);
    return (cap_dist * cap_dist + ly * ly) <= (b * b);
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let x = uv.x * W; let y = (1.0 - uv.y) * H;
  let cell = u32(y) * u32(W) + u32(x);
  
  if (is_solid(vec2(x, y), state.cx, state.cy, state.theta, state.a, state.b)) {
    return vec4(1.0, 0.8, 0.4, 1.0);
  }

  let ux = vel[cell * 2u];
  let uy = vel[cell * 2u + 1u];
  let speed = sqrt(ux*ux + uy*uy);
  
  // Colormap: blue for slow, red for fast
  let c = mix(vec3(0.1, 0.1, 0.4), vec3(0.9, 0.2, 0.1), clamp(speed * 10f, 0f, 1f));
  return vec4(c, 1.0);
}
