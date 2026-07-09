// Milestone 2 (plans/AMR.md): fine-level (L=1) LBM step.
//
// Structurally mirrors amr_step.wgsl (same D2Q9 BGK + Guo penalization
// physics, same card SDF), but for the fine grid specifically:
// - Plain flat buffer (fy*FBW+fx), not block-major -- see plans/AMR.md's
//   note on why block-structuring the fine level is deferred to Milestone 4.
// - No circular moving-window buffer, no ALBC sponge -- the fine region is a
//   small, simple, non-periodic buffer interior to the coarse domain.
// - Streaming clamps at the buffer edge instead of wrapping: the 2-cell
//   ghost border is expected to degrade over the 2 fine substeps (see
//   amr_interp_c2f.wgsl's header), so clamping there is harmless -- nothing
//   downstream depends on ghost-edge correctness within a macro-step.
// - Uses tau_fine = 2*tau_coarse - 0.5 (see plans/AMR.md's Milestone 2
//   section for the derivation). Card velocity/omega need no rescaling
//   (lattice velocity is level-invariant when dx_L/dt_L is preserved).
// - Force integration stays coarse-only this milestone (scope cut recorded
//   in plans/AMR.md) -- this kernel produces a physically consistent fine
//   flow field but doesn't write to forceBuf.

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

@group(0) @binding(0) var<storage, read>       state : CardState;
@group(0) @binding(1) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel   : array<f32>;

override FW : u32;
override FH : u32;
override FINE_ORIGIN_X : i32;
override FINE_ORIGIN_Y : i32;
const GHOST = 2u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

fn fineToCoarseUnit(fCoord: u32, origin: i32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    let dx = p.x - state.cx;
    let dy = p.y - state.cy;
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;

    // Algebraic distance approximation for ellipse. No periodic wrap here
    // (unlike the coarse SDF): the fine region is small and local, always
    // well within one card-length of state.cx/cy, so wrap never triggers.
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b;
}

fn get_chi(phi: f32) -> f32 {
    let epsilon = 1.5f;
    return 0.5f * (1.0f - tanh(phi / epsilon));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y;
  let FBW = FW + 2u * GHOST;
  let FBH = FH + 2u * GHOST;
  if (fx >= FBW || fy >= FBH) { return; }

  let cell = fy * FBW + fx;

  // 1. Pull Streaming: clamp at the buffer edge (see file header).
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    let srcX = clamp(i32(fx) - ex[i], 0, i32(FBW) - 1);
    let srcY = clamp(i32(fy) - ey[i], 0, i32(FBH) - 1);
    let srcCell = u32(srcY) * FBW + u32(srcX);
    f[i] = f_in[i * (FBW * FBH) + srcCell];
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  ux_star /= rho; uy_star /= rho;

  // 3. Penalty Force and Solid Coupling (window/coarse-unit space)
  let p = vec2<f32>(fineToCoarseUnit(fx, FINE_ORIGIN_X), fineToCoarseUnit(fy, FINE_ORIGIN_Y));
  let rx = p.x - state.cx;
  let ry = p.y - state.cy;

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);
  let chi = get_chi(phi);

  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  let ux = ux_star + Fx / (2.0f * rho);
  let uy = uy_star + Fy / (2.0f * rho);
  let u_sq = ux*ux + uy*uy;

  vel[cell * 2u] = ux; vel[cell * 2u + 1u] = uy;

  // 4. Collision (no sponge -- fine region is interior to the coarse domain)
  let tau_fine = 2.0f * state.tau - 0.5f;
  let omg = 1.0f / tau_fine;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );

    f_out[i * (FBW * FBH) + cell] = f[i] - omg * (f[i] - feq) + Si;
  }
}
