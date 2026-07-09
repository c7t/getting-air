// Milestone 2 (plans/AMR.md): fine -> coarse averaging (restriction).
//
// Runs once per macro-step, after both fine substeps complete, overwriting
// every coarse cell within the fine region's footprint with the average of
// its 4 fine children (AGAL Fig. 13 step 4). Simplification recorded in
// plans/AMR.md: this averages the WHOLE fine region back to its coarse
// parents, not just AGAL's narrower "interface layer" -- a correctness-
// neutral simplification at our scale, not a numerically different choice.
//
// rho is a simple arithmetic mean of the 4 children (exactly mass-
// conservative: each fine cell has 1/4 the coarse cell's area, so total
// mass over the 4 children equals mean(rho)*coarse_area). Velocity is a
// mass-weighted average of the children (momentum-conservative: coarse
// momentum = mean(rho_i * u_i), not a naive mean of u_i). The non-
// equilibrium part is inverse-Dupuis-Chopard-rescaled by tau_coarse/tau_fine
// (see amr_interp_c2f.wgsl for the forward direction).

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

@group(0) @binding(0) var<storage, read>       state    : CardState;
@group(0) @binding(1) var<storage, read>       f_fine   : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_coarse : array<f32>;

override W : u32;  // coarse grid dims
override H : u32;
override FW : u32; // fine grid "real" interior dims
override FH : u32;
override FINE_ORIGIN_X : i32;
override FINE_ORIGIN_Y : i32;

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lcx = gid.x; let lcy = gid.y; // coarse-cell-local coords within the fine region
  let CW = FW / 2u; let CH = FH / 2u;
  if (lcx >= CW || lcy >= CH) { return; }

  let FBW = FW + 2u * GHOST;
  let FBH = FH + 2u * GHOST;

  // The 4 fine children of this coarse cell, in fine-local (ghost-inclusive) coords.
  let fx0 = GHOST + 2u * lcx; let fx1 = fx0 + 1u;
  let fy0 = GHOST + 2u * lcy; let fy1 = fy0 + 1u;
  let children = array<u32, 4>(
    fy0 * FBW + fx0, fy0 * FBW + fx1,
    fy1 * FBW + fx0, fy1 * FBW + fx1
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
      f[i] = f_fine[i * (FBW * FBH) + cell];
      rho += f[i];
      ux  += f[i] * f32(ex[i]);
      uy  += f[i] * f32(ey[i]);
    }
    ux /= rho; uy /= rho;
    f_children[c] = f;
    rho_children[c] = rho; ux_children[c] = ux; uy_children[c] = uy;
    rho_sum += rho;
    rhou_x_sum += rho * ux;
    rhou_y_sum += rho * uy;
  }

  let rho_avg = rho_sum * 0.25f;
  let ux_avg = rhou_x_sum / rho_sum;
  let uy_avg = rhou_y_sum / rho_sum;

  let tau_fine = 2.0f * state.tau - 0.5f;
  let rescale = state.tau / tau_fine; // inverse of amr_interp_c2f.wgsl's rescale

  var fneq_avg: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    var s = 0f;
    for (var c = 0u; c < 4u; c++) {
      s += f_children[c][i] - feq(rho_children[c], ux_children[c], uy_children[c], i);
    }
    fneq_avg[i] = s * 0.25f;
  }

  // Destination: coarse WINDOW coords -> buffer coords -> block-major index.
  let cwx = u32(FINE_ORIGIN_X) + lcx;
  let cwy = u32(FINE_ORIGIN_Y) + lcy;
  let cbx = (cwx + u32(state.off_x)) % W;
  let cby = (cwy + u32(state.off_y)) % H;
  let coarseCell = cellIndex(cbx, cby);

  for (var i = 0u; i < 9u; i++) {
    f_coarse[i * (W * H) + coarseCell] = feq(rho_avg, ux_avg, uy_avg, i) + rescale * fneq_avg[i];
  }
}
