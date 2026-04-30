// Force and torque on the solid body via momentum exchange.
// For each fluid cell with a solid neighbour, sums 2*f_i*e_i over those links.
// Torque arm is from the ellipse centre (cx,cy) to the link midpoint.
// Outputs three atomically-accumulated i32 values scaled by FSCALE.

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

@group(0) @binding(0) var<storage, read>       f_col  : array<f32>;
@group(0) @binding(1) var<storage, read>       solid  : array<u32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;
@group(0) @binding(3) var<uniform>             params : Params;

const W      = 256u;
const H      = 512u;
const FSCALE  = 1000f;
const FI_MAX  = 0.6f;   // cap fi before accumulation to prevent int32 overflow

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell = y * W + x;
  if (solid[cell] != 0u) { return; }

  let base = cell * 9u;

  for (var i = 1u; i < 9u; i++) {   // skip rest direction (no momentum)
    let nx = u32((i32(x) + ex[i] + i32(W)) % i32(W));
    let ny = u32((i32(y) + ey[i] + i32(H)) % i32(H));

    if (solid[ny * W + nx] != 0u) {
      let fi = clamp(f_col[base + i], 0f, FI_MAX);
      let fx = 2f * fi * f32(ex[i]);
      let fy = 2f * fi * f32(ey[i]);

      // Moment arm: link midpoint relative to ellipse centre
      let rx = f32(x) + 0.5f * f32(ex[i]) - params.cx;
      let ry = f32(y) + 0.5f * f32(ey[i]) - params.cy;
      let tz = rx * fy - ry * fx;

      atomicAdd(&forces[0], i32(fx * FSCALE));
      atomicAdd(&forces[1], i32(fy * FSCALE));
      atomicAdd(&forces[2], i32(tz * FSCALE));
    }
  }
}
