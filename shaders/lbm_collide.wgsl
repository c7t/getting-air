// D2Q9 BGK collision pass
// Skips solid nodes; adds body force via simple momentum source term.

struct Params {
  tau    : f32,
  u_inlet: f32,  // unused in this shader, reserved
  gx     : f32,
  gy     : f32,
}

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read>       f_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_col  : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel    : array<f32>;
@group(0) @binding(4) var<storage, read>       solid  : array<u32>;

const W   = 256u;
const H   = 512u;
const CS2 = 0.33333333f;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell = y * W + x;
  let base = cell * 9u;

  // solid nodes carry no fluid; skip collision
  if (solid[cell] != 0u) { return; }

  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) { f[i] = f_in[base + i]; }

  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= rho; uy /= rho;

  vel[cell * 2u]      = ux;
  vel[cell * 2u + 1u] = uy;

  let u2 = ux*ux + uy*uy;
  for (var i = 0u; i < 9u; i++) {
    let eu  = f32(ex[i])*ux + f32(ey[i])*uy;
    let feq = wt[i] * rho * (1f + eu/CS2 + eu*eu/(2f*CS2*CS2) - u2/(2f*CS2));
    // BGK collision + simple body-force source term w_i * 3*(e_i . g) * rho
    let eig = f32(ex[i])*params.gx + f32(ey[i])*params.gy;
    f_col[base + i] = f[i] - (f[i] - feq) / params.tau + wt[i] * 3f * eig * rho;
  }
}
