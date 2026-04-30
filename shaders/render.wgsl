// Vorticity visualization — full-screen quad
// Reads vel buffer, computes ωz = ∂uy/∂x − ∂ux/∂y, colors blue→white→red

@group(0) @binding(0) var<storage, read> vel: array<f32>;

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
  // two triangles covering clip space
  var p = array<vec2f,6>(
    vec2f(-1,-1), vec2f(1,-1), vec2f(-1, 1),
    vec2f(-1, 1), vec2f(1,-1), vec2f( 1, 1)
  );
  var o: VSOut;
  o.pos = vec4f(p[vi], 0, 1);
  o.uv  = p[vi] * 0.5 + 0.5;  // [0,1]
  return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
  // map UV → grid cell (y flipped: uv.y=0 is top of screen = y=0 in grid)
  let gx = u32(in.uv.x * Wf);
  let gy = u32((1.0 - in.uv.y) * Hf);

  if (gx >= W || gy >= H) { return vec4f(0,0,0,1); }

  // central-difference vorticity
  let vR = getVel(gx+1u, gy);
  let vL = getVel(gx-1u, gy);  // clamps at border
  let vU = getVel(gx, gy+1u);
  let vD = getVel(gx, gy-1u);
  let vort = (vR.y - vL.y - vU.x + vD.x) * 0.5;

  // map vorticity to color: blue=negative, white=zero, red=positive
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
