// Vorticity visualization + ellipse line overlay
// blue=CCW, red=CW, white=zero; solid cells drawn dark.

@group(0) @binding(0) var<storage, read> vel  : array<f32>;
@group(0) @binding(1) var<storage, read> solid: array<u32>;

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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  let gx = u32(in.uv.x * Wf);
  let gy = u32((1.0 - in.uv.y) * Hf);
  if (gx >= W || gy >= H) { return vec4f(0,0,0,1); }

  let cell = gy * W + gx;

  if (solid[cell] != 0u) {
    return vec4f(0.15, 0.15, 0.15, 1); // solid = dark grey
  }

  let vR   = getVel(gx+1u, gy);
  let vL   = getVel(gx-1u, gy);
  let vU   = getVel(gx, gy+1u);
  let vD   = getVel(gx, gy-1u);
  let vort = (vR.y - vL.y - vU.x + vD.x) * 0.5;

  let scale = 0.002f;
  let t = clamp(vort / scale, -1f, 1f);
  var col: vec3f;
  if (t >= 0f) {
    col = mix(vec3f(1,1,1), vec3f(1,0.15,0.1), t);
  } else {
    col = mix(vec3f(1,1,1), vec3f(0.1,0.3,1), -t);
  }
  return vec4f(col, 1);
}
