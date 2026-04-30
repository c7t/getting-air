// Force and torque on the solid body via momentum exchange (Ladd 1994).
//
// For each fluid-solid link the exchanged momentum is:
//   F_link = (f_before_to_solid + f_after_from_solid) * c_q
// where c_q = e_i points FROM solid TO fluid.
//
// For moving bounce-back:
//   f_after = f_col[opp(i)] + 2*w_i*(e_i·u_b)/cs²
//
// So:  F_link = 2*f_col[opp(i)] * e_i  +  2*w_i*(e_i·u_b)/cs² * e_i
//
// The original code used f_col[i] (wrong index — should be opp(i)).

struct Params {
  tau  : f32,
  gx   : f32,
  gy   : f32,
  cx   : f32,
  cy   : f32,
  a    : f32,
  theta: f32,
  vx   : f32,
  vy   : f32,
  omega: f32,
  _p0  : f32,
  _p1  : f32,
}

@group(0) @binding(0) var<storage, read>       f_col  : array<f32>;
@group(0) @binding(1) var<storage, read>       solid  : array<u32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;
@group(0) @binding(3) var<uniform>             params : Params;

const W      = 256u;
const H      = 512u;
const CS2    = 0.33333333f;
const FSCALE = 1000f;
const FI_MAX = 0.6f;

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

  let cell = y * W + x;
  if (solid[cell] != 0u) { return; }

  let base = cell * 9u;

  for (var i = 1u; i < 9u; i++) {
    let nx = u32((i32(x) + ex[i] + i32(W)) % i32(W));
    let ny = u32((i32(y) + ey[i] + i32(H)) % i32(H));

    if (solid[ny * W + nx] != 0u) {
      // f going *toward* the solid is f_col[opp(i)] (direction opp = from fluid toward solid)
      let fi = clamp(f_col[base + opp[i]], 0f, FI_MAX);

      // Moving-BC correction: boundary velocity at link midpoint
      let mx   = f32(x) + 0.5f * f32(ex[i]);
      let my   = f32(y) + 0.5f * f32(ey[i]);
      let ubx  = params.vx - params.omega * (my - params.cy);
      let uby  = params.vy + params.omega * (mx - params.cx);
      let ei_ub = f32(ex[i]) * ubx + f32(ey[i]) * uby;

      // Total momentum exchange per link
      let mag  = 2f * fi + 2f * wt[i] * ei_ub / CS2;
      let fx   = mag * f32(ex[i]);
      let fy   = mag * f32(ey[i]);

      let rx   = mx - params.cx;
      let ry   = my - params.cy;
      let tz   = rx * fy - ry * fx;

      // Negate: formula gives force of solid-on-fluid; we need fluid-on-solid.
      atomicAdd(&forces[0], i32(-fx * FSCALE));
      atomicAdd(&forces[1], i32(-fy * FSCALE));
      atomicAdd(&forces[2], i32(-tz * FSCALE));
    }
  }
}
