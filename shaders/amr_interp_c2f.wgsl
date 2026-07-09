// Milestone 2 (plans/AMR.md): coarse -> fine ghost-cell interpolation.
//
// Runs once per macro-step, before the coarse step overwrites f_coarse, so
// it always reads the coarse state at the START of the macro-step (matching
// AGAL's Fig. 13 step 1: "interpolation of data from the coarse grid to
// ghost cells on the fine grid" happens before step 2's coarse advance).
// Only writes the fine grid's 2-cell ghost border -- the "real" interior
// cells evolve via their own fine-level physics (amr_step1.wgsl) and are
// left untouched here.
//
// Dupuis-Chopard rescaling (cited in the paper's section 3.4): bilinearly
// interpolate rho/u from the 4 surrounding coarse cells, evaluate f_eq
// fresh at the fine target point from that interpolated rho/u, and
// separately bilinearly interpolate the non-equilibrium part
// f_neq = f - f_eq from the same 4 corners, rescaled by tau_fine/tau_coarse
// (see plans/AMR.md's Milestone 2 section for the tau_fine derivation).

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
@group(0) @binding(1) var<storage, read>       f_coarse : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_fine   : array<f32>;

override W : u32;  // coarse grid dims
override H : u32;
override FW : u32; // fine grid "real" interior dims
override FH : u32;
override FINE_ORIGIN_X : i32; // fine region's lower-left corner, coarse window units
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

// Block-major linear index for a cell at COARSE buffer coordinates (cx, cy).
// See amr_step.wgsl for the full derivation.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

fn wrapCoord(v: i32, n: u32) -> u32 {
  let m = i32(n);
  return u32(((v % m) + m) % m);
}

// Fine ghost-local coordinate -> position in COARSE window units. GHOST is
// the fine cell at the region's edge; each coarse cell splits into 2 fine
// cells of half-width, centered at +-0.25 coarse units from the coarse
// cell's own center.
fn fineToCoarseUnit(fCoord: u32, origin: i32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

// Reads macroscopic (rho, ux, uy) and all 9 f_neq components at a single
// COARSE window-space integer coordinate (wrapping/off_x-mapping handled
// internally, matching amr_step.wgsl's coarse addressing).
struct CoarseSample {
  rho: f32,
  ux: f32,
  uy: f32,
  fneq: array<f32, 9>,
}
fn sampleCoarse(wx: i32, wy: i32) -> CoarseSample {
  let wxu = wrapCoord(wx, W);
  let wyu = wrapCoord(wy, H);
  let bx = (wxu + u32(state.off_x)) % W;
  let by = (wyu + u32(state.off_y)) % H;
  let cell = cellIndex(bx, by);

  var f: array<f32, 9>;
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    f[i] = f_coarse[i * (W * H) + cell];
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= rho; uy /= rho;

  var out: CoarseSample;
  out.rho = rho; out.ux = ux; out.uy = uy;
  for (var i = 0u; i < 9u; i++) {
    out.fneq[i] = f[i] - feq(rho, ux, uy, i);
  }
  return out;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y;
  let FBW = FW + 2u * GHOST;
  let FBH = FH + 2u * GHOST;
  if (fx >= FBW || fy >= FBH) { return; }

  // Only the ghost border is (re-)interpolated; interior cells evolve via
  // their own fine-level physics in amr_step1.wgsl.
  let isInterior = fx >= GHOST && fx < GHOST + FW && fy >= GHOST && fy < GHOST + FH;
  if (isInterior) { return; }

  let px = fineToCoarseUnit(fx, FINE_ORIGIN_X);
  let py = fineToCoarseUnit(fy, FINE_ORIGIN_Y);

  let x0 = i32(floor(px)); let x1 = x0 + 1;
  let y0 = i32(floor(py)); let y1 = y0 + 1;
  let tx = px - f32(x0);
  let ty = py - f32(y0);

  let s00 = sampleCoarse(x0, y0);
  let s10 = sampleCoarse(x1, y0);
  let s01 = sampleCoarse(x0, y1);
  let s11 = sampleCoarse(x1, y1);

  let w00 = (1f - tx) * (1f - ty);
  let w10 = tx * (1f - ty);
  let w01 = (1f - tx) * ty;
  let w11 = tx * ty;

  let rho = w00*s00.rho + w10*s10.rho + w01*s01.rho + w11*s11.rho;
  let ux  = w00*s00.ux  + w10*s10.ux  + w01*s01.ux  + w11*s11.ux;
  let uy  = w00*s00.uy  + w10*s10.uy  + w01*s01.uy  + w11*s11.uy;

  let tau_coarse = state.tau;
  let tau_fine = 2.0f * tau_coarse - 0.5f;
  let rescale = tau_fine / tau_coarse;

  let fineCell = fy * FBW + fx;
  for (var i = 0u; i < 9u; i++) {
    let fneq = w00*s00.fneq[i] + w10*s10.fneq[i] + w01*s01.fneq[i] + w11*s11.fneq[i];
    f_fine[i * (FBW * FBH) + fineCell] = feq(rho, ux, uy, i) + rescale * fneq;
  }
}
