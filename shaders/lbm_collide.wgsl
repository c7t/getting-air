// D2Q9 Regularized BGK collision (Latt & Chopard 2006).
// Reconstructs non-equilibrium stress from Π_αβ rather than raw f^neq,
// filtering lattice-scale noise that triggers instability near sharp boundaries.

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

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read>       f_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_col  : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel    : array<f32>;
@group(0) @binding(4) var<storage, read>       solid  : array<u32>;

const W   = 256u;
const H   = 512u;
const CS2 = 0.33333333f;   // cs²
const CS4 = 0.11111111f;   // cs⁴ = 1/9

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

  if (solid[cell] != 0u) { return; }

  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) { f[i] = f_in[base + i]; }

  // Macroscopic density and velocity
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= rho; uy /= rho;

  // Safety reset: density outside [0.5,2] means the cell has blown up.
  if (rho < 0.5f || rho > 2.0f) {
    for (var i = 0u; i < 9u; i++) { f_col[base + i] = wt[i]; }
    vel[cell * 2u] = 0f; vel[cell * 2u + 1u] = 0f;
    return;
  }

  // Velocity cap — keeps feq positive and prevents tip spikes spreading
  let u_sq = ux*ux + uy*uy;
  if (u_sq > 0.04f) {
    let s = 0.2f / sqrt(u_sq);
    ux *= s; uy *= s;
  }

  vel[cell * 2u]      = ux;
  vel[cell * 2u + 1u] = uy;

  // Non-equilibrium stress tensor  Π_αβ = Σ_i f_i·e_iα·e_iβ − ρ(u_α·u_β + cs²δ_αβ)
  var Pxx = -(rho * (ux*ux + CS2));
  var Pyy = -(rho * (uy*uy + CS2));
  var Pxy = -(rho * ux * uy);
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    Pxx += f[i] * exf * exf;
    Pyy += f[i] * eyf * eyf;
    Pxy += f[i] * exf * eyf;
  }

  // Regularized BGK:  f_col = feq + (1 − 1/τ)·f_neq_reg + body-force term
  // f_neq_reg_i = w_i / (2·cs⁴) · [ (e_ix²−cs²)·Πxx + 2·e_ix·e_iy·Πxy + (e_iy²−cs²)·Πyy ]
  let u2      = ux*ux + uy*uy;
  let omgT    = 1f - 1f / params.tau;   // (1 - 1/τ)
  let half_cs4_inv = 0.5f / CS4;        // = 4.5

  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + eu/CS2 + eu*eu/(2f*CS2*CS2) - u2/(2f*CS2));

    let f_neq_reg = wt[i] * half_cs4_inv *
      ((exf*exf - CS2)*Pxx + 2f*exf*eyf*Pxy + (eyf*eyf - CS2)*Pyy);

    let eig = exf*params.gx + eyf*params.gy;
    f_col[base + i] = feq + omgT * f_neq_reg + wt[i] * 3f * eig * rho;
  }
}
