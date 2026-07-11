// Milestone 7 (plans/AMR-multilevel.md): fine-level LBM step for every
// L(m>=2)'s own pool tiles -- sibling of amr_step1.wgsl (which stays
// exactly as-is and is now specifically level 1's own step shader, exactly
// like amr_interp_dense_parent.wgsl is level 1's own interpolation shader).
// Same physics body (streaming/collision/penalization/sponge, identical
// math), one structural difference: WHERE this tile's own physical
// (buffer-space, L0 units) origin comes from.
//
// Level 1 derives its own origin from `blockID % nbx` (nbx = W/BLOCK) --
// valid ONLY because level 1's blockID space coincides with L0's own
// block-grid by construction (footprint-preserving 1:1 parent, decision 1).
// A level>=2 tile has no such shortcut: its blockID lives in ITS OWN
// (finer) coordinate space (see amr_interp_pool_parent.wgsl's header on
// how bx/by are derived there), and converting that to a true L0-buffer-
// space physical position requires walking the WHOLE parent chain (this
// tile's quadrant offset within its parent, scaled by the parent's own
// cell size in L0 units, plus the parent's own origin, recursively) --
// NOT a cheap single mod/div the way level 1's derivation is (that's
// exactly why M5's ownBX/ownBY got removed as redundant, but this origin
// is genuinely NOT redundant: it chains across different levels' buffers,
// unlike ownBX/ownBY which was one mod/div within a SINGLE level's own
// blockID). So it's cached per-slot at quad-activation time instead
// (main-amr.js), in L0-buffer-space float units -- the same units level
// 1's own `bx*RB` origin is already expressed in, so this shader's window-
// mapping math (off_x/off_y wrap, card SDF) is IDENTICAL to level 1's,
// just reading a stored originX/originY instead of computing bx*RB.
//
// tau: `state.tau` is L0's own tau; a level>=2 tile's relevant "coarse tau"
// (the one its own tau_fine is derived FROM) is its PARENT level's tau,
// not L0's -- supplied via `levelParams.parentTau`, the same per-child-
// level uniform amr_interp_pool_parent.wgsl and amr_average_pool_parent.wgsl
// already use (recursively: tauAtLevel(m) in main-amr.js).

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

struct LevelParams {
  nbx: u32,        // unused here (no cellIndex()/blockID-derived origin at this level -- see header) -- kept so this level's ONE uniform buffer is shared verbatim with amr_interp_pool_parent.wgsl/amr_average_pool_parent.wgsl, not a third near-duplicate.
  nby: u32,        // unused here, same reason.
  parentTau: f32,
  dxL: f32,        // Milestone 8: this level's own grid spacing in L0-buffer-
                   // space units, used below to scale epsilon (get_chi).
}

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_in        : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out       : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_pool    : array<f32>;
@group(0) @binding(4) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(5) var<storage, read>       originX     : array<f32>;
@group(0) @binding(6) var<storage, read>       originY     : array<f32>;
@group(0) @binding(7) var<uniform>             levelParams : LevelParams;

override W : u32; // GLOBAL domain dims (window periodicity), same at every level -- not level-specific, see header.
override H : u32;
override RB : u32;
const GHOST = 2u;

override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;

// Milestone 8 (plans/AMR-multilevel.md): epsilon = K_EPS * dx_L, not a fixed
// physical constant -- a fixed epsilon means refinement only ever improves
// *sampling* of an unchanging diffuse-boundary width, never the boundary's
// own sharpness. K_EPS=1.5 matches today's L0/L1 value exactly (dx_L0=1,
// dx_L1=0.5 are hardcoded literals in amr_step.wgsl/amr_step1.wgsl, which
// is why THEIR epsilon values don't change); this shared pipeline's dx_L
// genuinely varies per level (levelParams.dxL, runtime -- see header), so
// epsilon must be computed here, not hardcoded.
override K_EPS : f32 = 1.5f;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

fn fineToCoarseUnit(fCoord: u32, origin: f32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return origin - 0.25 + 0.5 * j;
}

fn wrapf(v: f32, n: f32) -> f32 {
  var r = v % n;
  if (r < 0.0) { r += n; }
  return r;
}

fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    var dx = p.x - state.cx;
    var dy = p.y - state.cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b;
}

fn get_chi(phi: f32) -> f32 {
    let epsilon = K_EPS * levelParams.dxL;
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y;
  let slot = gid.z;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  // The one structural difference vs. amr_step1.wgsl: origin comes from a
  // cached per-slot value (set at quad-activation time), not derived from
  // blockID.
  let originX_L0 = originX[slot];
  let originY_L0 = originY[slot];

  let poolPlaneStride = arrayLength(&f_in) / 9u;
  let cell = slot * (FB * FB) + fy * FB + fx;

  // 1. Pull Streaming: clamp at the slot's own buffer edge.
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    let srcX = clamp(i32(fx) - ex[i], 0, i32(FB) - 1);
    let srcY = clamp(i32(fy) - ey[i], 0, i32(FB) - 1);
    let srcCell = slot * (FB * FB) + u32(srcY) * FB + u32(srcX);
    f[i] = f_in[i * poolPlaneStride + srcCell];
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  let rhoDen = max(rho, 1e-6f);
  ux_star /= rhoDen; uy_star /= rhoDen;

  // 3. Penalty Force and Solid Coupling. Buffer-space fine position (L0
  // units, via the cached origin) -> window position by inverting
  // off_x/off_y (same as amr_step1.wgsl).
  let bufX = fineToCoarseUnit(fx, originX_L0);
  let bufY = fineToCoarseUnit(fy, originY_L0);
  let wx = wrapf(bufX - state.off_x, f32(W));
  let wy = wrapf(bufY - state.off_y, f32(H));
  let p = vec2<f32>(wx, wy);
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);
  let chi = get_chi(phi);

  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  let ux = ux_star + Fx / (2.0f * rhoDen);
  let uy = uy_star + Fy / (2.0f * rhoDen);
  let u_sq = ux*ux + uy*uy;

  vel_pool[cell * 2u] = ux; vel_pool[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC sponge -- same formula as amr_step1.wgsl.
  let SPONGE_W = 4.0f;
  let dist_x = min(wx, f32(W) - 1.0f - wx);
  let dist_y = min(wy, f32(H) - 1.0f - wy);
  var sponge_weight = clamp(1.0f - min(dist_x, dist_y) / SPONGE_W, 0.0f, 1.0f);
  sponge_weight = sponge_weight * sponge_weight * (3.0f - 2.0f * sponge_weight);

  // Relative to THIS level's own parent, not L0 -- see header.
  let tau_coarse = levelParams.parentTau;
  let tau_fine = 2.0f * tau_coarse - 0.5f;
  let omg = 1.0f / tau_fine;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );

    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let eu_far = exf*SPONGE_UX + eyf*SPONGE_UY;
    let f_target = wt[i] * (1.0f + 3.0f*eu_far + 4.5f*eu_far*eu_far - 1.5f*(SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY));
    f_out[i * poolPlaneStride + cell] = mix(f_collide, f_target, sponge_weight);
  }
}
