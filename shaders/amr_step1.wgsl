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
// Streaming resolves a source cell that falls outside this tile's interior
// against the OWNING same-level tile directly (blockSlot, binding 5) and only
// falls back to clamping at the slot's own buffer edge where no such neighbour
// exists -- see the DIRECT_GHOST override. Force integration stays coarse-only
// this milestone (same scope cut as Milestone 2).

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"
// @include "common_sponge.wgsl"
// @include "common_walls.wgsl"





@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_in        : array<u32>;
@group(0) @binding(2) var<storage, read_write> f_out       : array<u32>;
@group(0) @binding(3) var<storage, read_write> vel_pool    : array<f32>;
@group(0) @binding(4) var<storage, read>       slotToBlock : array<i32>;
// This level's own logical block grid -> pool slot, indexed by
// blockID = by*(W/BLOCK)+bx. Read-only, and the ONLY new input
// neighbour-addressed streaming needs (see DIRECT_GHOST) -- the neighbour
// tile's f data is already in scope, because f_in is the whole pool.
@group(0) @binding(5) var<storage, read>       blockSlot   : array<i32>;

override W : u32; // coarse grid dims, needed for the off_x/off_y window wrap
override H : u32;
override RB : u32;
// ── Measurement instrument: ?benchSkip=step1-ring ────────────────────────────
// Skips the ghost-RING cells of each tile, leaving only the 2*RB square
// interior. A proxy for what the fine step would cost if tiles carried no
// ghost padding at all (FB 20 -> 16, i.e. 36% fewer cells per tile), which is
// the second-order prize in "stop materializing same-level ghost cells".
//
// MEASUREMENT ONLY, and wrong by construction: substep B consumes the ring
// values substep A wrote for any cell the fine-fine exchange does not refresh
// (a tile at the edge of the refined region), so a run with this set is not
// physically correct.
//
// Measured 2026-09-07, desktop RTX 4080, res=8 levels=3: 0.2% of frame GPU
// time -- essentially free to skip, because ring threads share their 8x8
// workgroups with interior threads and removing them frees no scheduling slot.
// This CORRECTS plans/perf-characterization.md, which listed the same 36% as
// "about 10.5% of the whole frame ... the single largest identified piece of
// pure overhead". It is not, on this device. The phone is bandwidth-bound
// rather than occupancy-bound and the traffic model predicts ~10% there, so
// this knob exists to be re-read on that device, where the answer should
// differ. Guard placed after the slot lookup so the number stays comparable
// with the desktop figure above.
override SKIP_GHOST : u32 = 0u;

// ── Neighbour-addressed streaming (AGAL) ─────────────────────────────────────
// 1 (default): a pull whose source cell falls in this tile's ghost ring is
// resolved against the OWNING same-level tile's own interior, via blockSlot,
// instead of reading a materialized ghost cell. 0: the legacy path -- clamp at
// the slot's own buffer edge and rely on a separate fine-fine copy pass having
// filled the ring (?ghostcopy=1 restores it, for A/B).
//
// This is what lets the between-substep fine-fine ghost pass go away entirely
// (plans/perf-characterization.md's "the one lead left": 15.3% of frame on the
// phone, 13.3-18.5% on the desktop). It is not just cheaper, it is fresher:
// substep B now sees the neighbour's post-`average` interior directly, where
// the copy pass ran BEFORE the child's average landed, and a depth-2 ring cell
// -- which the copy path leaves clamp-degraded after substep A, and which the
// child's own bilinear parent sampling does read -- now streams correctly too.
//
// The ring is still materialized, and interp still fills it: a tile at the
// coarse/fine interface has no same-level neighbour there, so blockSlot is < 0
// and this falls back to the clamped read of a parent-interpolated ghost --
// exactly the legacy path, for exactly the cells that need it.
override DIRECT_GHOST : u32 = 1u;

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

  if (SKIP_GHOST != 0u) {
    let ringInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;
    if (!ringInterior) { return; } // see the SKIP_GHOST override above
  }

  let nbx = W / BLOCK;
  let nby = H / BLOCK;
  let bx = u32(blockID) % nbx;
  let by = u32(blockID) / nbx;
  let originX = bx * RB;
  let originY = by * RB;
  // Wrapped neighbour columns/rows, hoisted out of the 9-direction gather
  // below (the block grid is periodic, matching amr_interp_dense_parent.wgsl's
  // own fine-fine consultation exactly). Dead code under DIRECT_GHOST=0.
  let RB2 = RB * 2u;
  let bxm = (bx + nbx - 1u) % nbx;
  let bxp = (bx + 1u) % nbx;
  let bym = (by + nby - 1u) % nby;
  let byp = (by + 1u) % nby;

  // Neighbour-slot resolution, hoisted out of the 9-direction gather below.
  // A source cell is at most one cell away and a tile's interior is RB2 >= 2
  // wide, so at most ONE non-zero neighbour offset is reachable per axis --
  // which means the whole gather needs at most THREE neighbour slots (the
  // x-, y- and diagonal tiles), resolved once here instead of re-resolved,
  // with a fresh blockSlot load, on every one of the nine directions. A
  // fully-interior thread loads nothing at all.
  var offX = 0; var offY = 0;
  var nbrX = -1; var nbrY = -1; var nbrXY = -1;
  if (DIRECT_GHOST != 0u) {
    offX = select(select(0, 1, fx + 1u >= GHOST + RB2), -1, fx <= GHOST);
    offY = select(select(0, 1, fy + 1u >= GHOST + RB2), -1, fy <= GHOST);
    let cx = select(select(bx, bxp, offX > 0), bxm, offX < 0);
    let cy = select(select(by, byp, offY > 0), bym, offY < 0);
    if (offX != 0) { nbrX = blockSlot[by * nbx + cx]; }
    if (offY != 0) { nbrY = blockSlot[cy * nbx + bx]; }
    if (offX != 0 && offY != 0) { nbrXY = blockSlot[cy * nbx + cx]; }
  }

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
        f[i] = fUnpack(f_in[fIdx(opp[i], poolPlaneStride, cell)], opp[i]) + corr;
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
        f[i] = fUnpack(f_in[fIdx(opp[i], poolPlaneStride, cell)], opp[i]) + corr;
        continue;
      }
    }
    var sx = i32(fx) - ex[i];
    var sy = i32(fy) - ey[i];
    var srcSlot = slot;
    if (DIRECT_GHOST != 0u) {
      // Does this source cell leave the interior, and on which axes? If so
      // the offset can only be offX/offY (see the hoist above), so this is
      // pure register work -- no second blockSlot load.
      let ox = select(0, offX, sx < i32(GHOST) || sx >= i32(GHOST + RB2));
      let oy = select(0, offY, sy < i32(GHOST) || sy >= i32(GHOST + RB2));
      let ns = select(select(select(-1, nbrY, oy != 0), nbrX, ox != 0),
                      nbrXY, ox != 0 && oy != 0);
      if (ns >= 0) {
        // Re-express the source in the neighbour's own local coordinates. It
        // always lands in ITS interior -- sx in [-1, GHOST-1] maps to
        // [RB2-1, RB2+1], sx in [GHOST+RB2, FB] maps to [GHOST, GHOST+2] --
        // so the clamp below is a no-op on this path.
        srcSlot = u32(ns);
        sx -= ox * i32(RB2);
        sy -= oy * i32(RB2);
      }
    }
    // No same-level neighbour (or DIRECT_GHOST=0): clamp at the slot's own
    // buffer edge and read this tile's own ghost cell, which the interp pass
    // filled from the parent.
    let srcCell = srcSlot * (FB * FB)
                + u32(clamp(sy, 0, i32(FB) - 1)) * FB
                + u32(clamp(sx, 0, i32(FB) - 1));
    f[i] = fUnpack(f_in[fIdx(i, poolPlaneStride, srcCell)], i);
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
  // Gathered, then stored a whole cell at a time: under F16 two planes share
  // a word, so a per-plane store would be a read-modify-write race. See
  // common_fpack.wgsl.
  var fo: array<f32,9>;
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
    fo[i] = mix(f_collide, f_target, sponge_weight);
  }
  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    f_out[wi * poolPlaneStride + cell] = fPack(fo[fLo(wi)], fo[fHi(wi)], wi);
  }
}
