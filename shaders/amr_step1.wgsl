// Milestone 4 (plans/AMR.md): fine-level (L=1) LBM step, POOL-AWARE.
// Supersedes Milestone 2's single-fixed-region version -- see
// amr_interp_c2f.wgsl's file header for the pool addressing scheme this
// shares (dispatch over (tile, tile, slot), slotToBlock indirection,
// buffer-space-native coarse addressing).
//
// Unlike the interpolation pass, this kernel DOES need window coordinates
// (for the card SDF and penalization physics, both physically anchored),
// derived by inverting the moving-window off_x/off_y mapping -- the same
// derivation amr_step.wgsl's coarse kernel uses, just applied to a
// continuous fine-cell position instead of an integer coarse-cell one.
//
// Streaming clamps at the slot's own buffer edge (2-cell ghost border
// degrades over the 2 fine substeps by design -- see
// amr_interp_c2f.wgsl's header); force integration stays coarse-only this
// milestone (same scope cut as Milestone 2).

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_sponge.wgsl"
// @include "common_walls.wgsl"

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_in        : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out       : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_pool    : array<f32>;
@group(0) @binding(4) var<storage, read>       slotToBlock : array<i32>;

override W : u32; // coarse grid dims, needed for the off_x/off_y window wrap
override H : u32;
override RB : u32;
const BLOCK = 8u;
const GHOST = 2u;

// Sponge relaxation target velocity -- mirrors amr_step.wgsl's SPONGE_UX/UY
// exactly (same formula, see this file's sponge comment below for why the
// fine level needs its own copy of the sponge at all).
override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;
// Sponge ring width in cells -- see lbm_step.wgsl's identical override.
override SPONGE_W : f32 = 4.0f;

// Optional sharp bounce-back solid coupling -- see lbm_step.wgsl's header
// for the full method. At this level, "the geometrically-correct source"
// (used for the sharp inside test) and "the clamped-at-tile-edge source"
// (the buffer address normal streaming reads) are DIFFERENT things --
// clamping is purely a buffer-addressing artifact for cells whose true
// neighbor lies outside this tile's own FB x FB storage (that continuity
// is handled by the separate ghost-fill pass, not by this kernel), not a
// physical statement -- so the sharp test below deliberately uses the
// UNCLAMPED fine-index position (fineToCoarseUnitI, i32-accepting so it
// stays well-defined for an off-tile index), while the bounce-back VALUE
// substitution still reads this cell's own (in-tile, always valid) data.
override USE_BOUNCEBACK : u32 = 0u;

// Channel/TGV-scenario overrides -- see shaders/lbm_step.wgsl's identical
// set for the full rationale. All default to a no-op.
override HAS_BODY : u32 = 1u;
override WALL_Y : u32 = 0u;
override WALL_U0 : f32 = 0.0f;
override WALL_U1 : f32 = 0.0f;
override FORCE_X : f32 = 0.0f;
override FORCE_Y : f32 = 0.0f;

fn fineToCoarseUnit(fCoord: u32, origin: u32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

// Same formula as fineToCoarseUnit, but accepting a possibly-negative or
// possibly-past-FB fine index (a neighbor position that may lie outside
// this tile's own storage) -- see USE_BOUNCEBACK's own comment above.
fn fineToCoarseUnitI(fCoordI: i32, origin: u32) -> f32 {
  let j = f32(fCoordI - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

fn wrapf(v: f32, n: f32) -> f32 {
  var r = v % n;
  if (r < 0.0) { r += n; }
  return r;
}

// Milestone 8 (plans/AMR-multilevel.md): epsilon = K_EPS * dx_L1, not the
// bare physical constant amr_step.wgsl (L0) still uses -- L1's own dx is a
// fixed 0.5 (in L0-buffer-space units, matching amr_interp_dense_parent.
// wgsl's 0.5 factor; L1 is a single dedicated file/level, so this is a
// literal here, not a runtime lookup the way amr_step1_pool.wgsl's shared,
// multi-level pipeline needs). K_EPS=1.5 matches L0's own hardcoded value
// (dx_L0=1 there), so this is the SAME constant, just no longer coincident
// with dx=1 -- a genuine behavior change (0.75, not 1.5), fixing the
// under-resolved diffuse-boundary sampling this milestone targets.
const K_EPS = 1.5f;
fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, K_EPS * 0.5f);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y;
  let slot = gid.z;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  let nbx = W / BLOCK;
  let originX = (u32(blockID) % nbx) * RB;
  let originY = (u32(blockID) / nbx) * RB;

  let poolPlaneStride = arrayLength(&f_in) / 9u;
  let cell = slot * (FB * FB) + fy * FB + fx;

  // Position/solid-velocity terms, hoisted ABOVE the gather loop -- see
  // lbm_step.wgsl's identical hoist for why USE_BOUNCEBACK needs these
  // before streaming, not after. Buffer-space fine position -> window
  // position by inverting off_x/off_y (see file header).
  let bufX = fineToCoarseUnit(fx, originX);
  let bufY = fineToCoarseUnit(fy, originY);
  let wx = wrapf(bufX - state.off_x, f32(W));
  let wy = wrapf(bufY - state.off_y, f32(H));
  let p = vec2<f32>(wx, wy);
  // Periodic minimum-image lever arm, matching amr_step.wgsl / amr_force.wgsl
  // (the coarse step and force pass wrap rx/ry; the fine step previously did
  // not, so a cell reached across a seam got the wrong rotational velocity).
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);

  // 1. Pull Streaming: clamp at the slot's own buffer edge (or bounce back
  // off the solid -- see USE_BOUNCEBACK's own header comment).
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    if (USE_BOUNCEBACK != 0u && HAS_BODY != 0u) {
      let srcBufX = fineToCoarseUnitI(i32(fx) - ex[i], originX);
      let srcBufY = fineToCoarseUnitI(i32(fy) - ey[i], originY);
      let srcWx = wrapf(srcBufX - state.off_x, f32(W));
      let srcWy = wrapf(srcBufY - state.off_y, f32(H));
      if (get_phi(vec2<f32>(srcWx, srcWy), state) < 0f) {
        let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
        f[i] = f_in[opp[i] * poolPlaneStride + cell] + corr;
        continue;
      }
    }
    if (WALL_Y != 0u) {
      // Unwrapped buffer-space source position -- see
      // shaders/common_walls.wgsl's *F helpers' own comment on why this
      // (not wrapf's periodic result) is the right test, and why it
      // assumes off_y=0 (true for every WALL_Y-using scenario).
      let srcBufYUnwrapped = fineToCoarseUnitI(i32(fy) - ey[i], originY);
      if (wallSourceOutsideF(srcBufYUnwrapped)) {
        let wallUx = wallVelocityXF(srcBufYUnwrapped, WALL_U0, WALL_U1);
        let corr = 2f * wt[i] * f32(ex[i]) * wallUx / CS2;
        f[i] = f_in[opp[i] * poolPlaneStride + cell] + corr;
        continue;
      }
    }
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
  // NaN-containment floor (see amr_step.wgsl): finite velocity even if rho<=0.
  let rhoDen = max(rho, 1e-6f);
  ux_star /= rhoDen; uy_star /= rhoDen;

  // 3. Penalty Force and Solid Coupling -- chi forced to 0 under
  // USE_BOUNCEBACK or when there's no body at all, same as lbm_step.wgsl.
  let chi = select(get_chi(phi), 0f, USE_BOUNCEBACK != 0u || HAS_BODY == 0u);

  let Fx = rho * chi * (usx - ux_star) + FORCE_X;
  let Fy = rho * chi * (usy - uy_star) + FORCE_Y;

  let ux = ux_star + Fx / (2.0f * rhoDen);
  let uy = uy_star + Fy / (2.0f * rhoDen);
  let u_sq = ux*ux + uy*uy;

  vel_pool[cell * 2u] = ux; vel_pool[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC sponge. Milestone 4b fix: this used to skip the
  // sponge entirely on the (then-true) assumption that the fine region
  // never reaches the window edge -- valid when M2 hand-placed a single
  // static box, but false once refinement is criterion-driven and can
  // trigger anywhere, including near the sponge band where the coarse step
  // (amr_step.wgsl) DOES damp toward equilibrium. A refined block there
  // with no sponge of its own diverges from its damped coarse neighbors,
  // and the average pass then writes that undamped state back onto them --
  // exactly the boundary artifact this was fixed in response to. Same
  // formula as amr_step.wgsl's sponge, reusing the wx/wy already computed
  // above for the card SDF.
  let dist_x = min(wx, f32(W) - 1.0f - wx);
  let dist_y = min(wy, f32(H) - 1.0f - wy);
  let sponge_weight = spongeWeight(dist_x, dist_y, SPONGE_W);

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

    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let eu_far = exf*SPONGE_UX + eyf*SPONGE_UY;
    let f_target = wt[i] * (1.0f + 3.0f*eu_far + 4.5f*eu_far*eu_far - 1.5f*(SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY));
    f_out[i * poolPlaneStride + cell] = mix(f_collide, f_target, sponge_weight);
  }
}
