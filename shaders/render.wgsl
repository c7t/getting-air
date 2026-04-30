// Vorticity field + card major-axis line overlay.

struct Params {
  tau  : f32,
  gx   : f32,
  gy   : f32,
  cx   : f32,
  cy   : f32,
  a    : f32,
  theta: f32,
  _pad : f32,
}

@group(0) @binding(0) var<storage, read> vel   : array<f32>;
@group(0) @binding(1) var<storage, read> solid : array<u32>;
@group(0) @binding(2) var<uniform>       params: Params;

const W  = 256u;
const H  = 512u;
const Wf = 256f;
const Hf = 512f;

fn getVel(x: u32, y: u32) -> vec2f {
  let cx = clamp(x, 0u, W-1u);
  let cy = clamp(y, 0u, H-1u);
  let b  = (cy * W + cx) * 2u;
  return vec2f(vel[b], vel[b+1u]);
}

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f,6>(
    vec2f(-1,-1), vec2f(1,-1), vec2f(-1, 1),
    vec2f(-1, 1), vec2f(1,-1), vec2f( 1, 1)
  );
  var o: VSOut;
  o.pos = vec4f(p[vi], 0, 1);
  o.uv  = p[vi] * 0.5 + 0.5;
  return o;
}

// Signed distance from point p to segment (a,b)
fn segDist(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let t  = clamp(dot(p - a, ab) / dot(ab, ab), 0f, 1f);
  return length(p - (a + t * ab));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  let gxf = in.uv.x * Wf;
  let gyf = (1f - in.uv.y) * Hf;
  let gxi = u32(gxf);
  let gyi = u32(gyf);

  if (gxi >= W || gyi >= H) { return vec4f(0,0,0,1); }

  // Card line: from -a to +a along major axis
  let ca   = cos(params.theta);
  let sa   = sin(params.theta);
  let tip1 = vec2f(params.cx - params.a * ca, params.cy - params.a * sa);
  let tip2 = vec2f(params.cx + params.a * ca, params.cy + params.a * sa);
  let d    = segDist(vec2f(gxf, gyf), tip1, tip2);
  if (d < 0.8f) { return vec4f(1, 1, 0.2, 1); }  // bright yellow line

  // Vorticity
  let vR   = getVel(gxi+1u, gyi);
  let vL   = getVel(gxi-1u, gyi);
  let vU   = getVel(gxi, gyi+1u);
  let vD   = getVel(gxi, gyi-1u);
  let vort = (vR.y - vL.y - vU.x + vD.x) * 0.5f;

  let scale = 0.005f;
  let t = clamp(vort / scale, -1f, 1f);
  var col: vec3f;
  if (t >= 0f) {
    col = mix(vec3f(1,1,1), vec3f(1,0.15,0.1), t);
  } else {
    col = mix(vec3f(1,1,1), vec3f(0.1,0.3,1), -t);
  }
  return vec4f(col, 1);
}
