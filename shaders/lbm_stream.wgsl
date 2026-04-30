// D2Q9 streaming — pull scheme, periodic in x and y.
// Moving bounce-back (Ladd 1994): corrects for card surface velocity so
// the no-slip condition enforces u=u_wall, not u=0.
// Without this, rotation at the tips (ω·a up to 0.64 lu/step) injects a
// systematic 390% error into the boundary distributions each step.

struct Params {
  tau  : f32,
  gx   : f32,
  gy   : f32,
  cx   : f32,
  cy   : f32,
  a    : f32,
  theta: f32,
  vx   : f32,   // card centre-of-mass velocity
  vy   : f32,
  omega: f32,
  _p0  : f32,
  _p1  : f32,
}

@group(0) @binding(0) var<storage, read>       f_col  : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out  : array<f32>;
@group(0) @binding(2) var<storage, read>       solid  : array<u32>;
@group(0) @binding(3) var<uniform>             params : Params;

const W   = 256u;
const H   = 512u;
const CS2 = 0.33333333f;

const ex  = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey  = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const opp = array<u32,9>( 0, 3, 4, 1, 2, 7, 8, 5, 6);
const wt  = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell     = y * W + x;
  let dst_base = cell * 9u;

  if (solid[cell] != 0u) {
    for (var i = 0u; i < 9u; i++) {
      f_out[dst_base + i] = f_col[dst_base + opp[i]];
    }
    return;
  }

  for (var i = 0u; i < 9u; i++) {
    let sx = u32((i32(x) - ex[i] + i32(W)) % i32(W));
    let sy = u32((i32(y) - ey[i] + i32(H)) % i32(H));

    let src_cell = sy * W + sx;
    if (solid[src_cell] != 0u) {
      // Moving bounce-back: link midpoint is (x - 0.5·ei, y - 0.5·ei)
      // Card velocity at that point: v_cm + ω × r
      let mx   = f32(x) - 0.5f * f32(ex[i]);
      let my   = f32(y) - 0.5f * f32(ey[i]);
      let rx   = mx - params.cx;
      let ry   = my - params.cy;
      let ubx  = params.vx - params.omega * ry;
      let uby  = params.vy + params.omega * rx;
      let ei_ub = f32(ex[i]) * ubx + f32(ey[i]) * uby;
      f_out[dst_base + i] = f_col[dst_base + opp[i]] + 2f * wt[i] * ei_ub / CS2;
    } else {
      f_out[dst_base + i] = f_col[src_cell * 9u + i];
    }
  }
}
