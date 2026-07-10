// Milestone 4 (plans/AMR.md): fine -> coarse averaging (restriction),
// POOL-AWARE. Supersedes Milestone 2's single-fixed-region version -- see
// amr_interp_c2f.wgsl's file header for the shared pool addressing scheme.
//
// Dispatched as (1, 1, MAX_FINE_BLOCKS) with workgroup_size(8,8): a coarse
// block is exactly RB*RB=8*8=64 cells, i.e. exactly one workgroup, so each
// thread handles exactly one coarse cell's 4 fine children -- no separate
// tile-index dimension needed here (unlike interp/step1, which iterate
// over the larger FBxFB slot buffer).
//
// rho: simple arithmetic mean of the 4 children (exactly mass-
// conservative). Velocity: mass-weighted average (momentum-conservative).
// Non-equilibrium part: inverse-Dupuis-Chopard-rescaled by
// tau_coarse/tau_fine (see amr_interp_c2f.wgsl for the forward direction).

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
  y_total: f32,
  x_total: f32,
  off_x  : f32,
  off_y  : f32,
  off_x_old : f32,
  off_y_old : f32,
}

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_pool      : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_coarse    : array<f32>;
@group(0) @binding(3) var<storage, read>       slotToBlock : array<i32>;

override W : u32;
override H : u32;
override RB : u32;
const BLOCK = 8u;
const GHOST = 2u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

fn feq(rho: f32, ux: f32, uy: f32, i: u32) -> f32 {
  let eu = f32(ex[i]) * ux + f32(ey[i]) * uy;
  return wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*(ux*ux+uy*uy));
}

fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let lcx = lid.x; let lcy = lid.y; // coarse-cell-local coords within the block
  let slot = wgid.z;

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  let FB = RB * 2u + 2u * GHOST;
  let poolPlaneStride = arrayLength(&f_pool) / 9u;

  // The 4 fine children of this coarse cell, in slot-local (ghost-inclusive) coords.
  let fx0 = GHOST + 2u * lcx; let fx1 = fx0 + 1u;
  let fy0 = GHOST + 2u * lcy; let fy1 = fy0 + 1u;
  let children = array<u32, 4>(
    slot * (FB * FB) + fy0 * FB + fx0, slot * (FB * FB) + fy0 * FB + fx1,
    slot * (FB * FB) + fy1 * FB + fx0, slot * (FB * FB) + fy1 * FB + fx1
  );

  var rho_sum = 0f;
  var rhou_x_sum = 0f; var rhou_y_sum = 0f;
  var f_children: array<array<f32, 9>, 4>;
  var rho_children: array<f32, 4>;
  var ux_children: array<f32, 4>;
  var uy_children: array<f32, 4>;

  for (var c = 0u; c < 4u; c++) {
    let cell = children[c];
    var rho = 0f; var ux = 0f; var uy = 0f;
    var f: array<f32, 9>;
    for (var i = 0u; i < 9u; i++) {
      f[i] = f_pool[i * poolPlaneStride + cell];
      rho += f[i];
      ux  += f[i] * f32(ex[i]);
      uy  += f[i] * f32(ey[i]);
    }
    ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f); // NaN-containment floor
    f_children[c] = f;
    rho_children[c] = rho; ux_children[c] = ux; uy_children[c] = uy;
    rho_sum += rho;
    rhou_x_sum += rho * ux;
    rhou_y_sum += rho * uy;
  }

  let rho_avg = rho_sum * 0.25f;
  let rho_sum_den = max(rho_sum, 1e-6f); // NaN-containment floor
  let ux_avg = rhou_x_sum / rho_sum_den;
  let uy_avg = rhou_y_sum / rho_sum_den;

  let tau_fine = 2.0f * state.tau - 0.5f;
  // Inverse of the coarse->fine rescale (see amr_interp_c2f.wgsl): the
  // fine->coarse factor is (tau_coarse/tau_fine) * (dx_coarse/dx_fine) =
  // (tau_coarse/tau_fine) * n, with refinement ratio n=2. The per-cell velocity
  // gradient doubles going to the coarser grid, so fneq must be scaled up by n.
  let rescale = 2.0f * state.tau / tau_fine;

  var fneq_avg: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    var s = 0f;
    for (var c = 0u; c < 4u; c++) {
      s += f_children[c][i] - feq(rho_children[c], ux_children[c], uy_children[c], i);
    }
    fneq_avg[i] = s * 0.25f;
  }

  // Destination: coarse BUFFER coords (no window conversion -- see file header).
  let nbx = W / BLOCK;
  let cbx = (u32(blockID) % nbx) * RB + lcx;
  let cby = (u32(blockID) / nbx) * RB + lcy;
  let coarseCell = cellIndex(cbx, cby);

  for (var i = 0u; i < 9u; i++) {
    f_coarse[i * (W * H) + coarseCell] = feq(rho_avg, ux_avg, uy_avg, i) + rescale * fneq_avg[i];
  }
}
