// Fused LBM Kernel: Pull-Streaming + Collision + Source Term
//
// Milestone 1 (plans/AMR.md): block-major buffer layout. Buffer cells are
// grouped into fixed BLOCK x BLOCK tiles laid out contiguously per tile --
// this is the addressable "cell-block" unit later milestones pool/refine/
// coarsen (AGAL section 3.2's cell-block decomposition), though at this
// milestone the block <-> position mapping is still a straight identity (no
// pool indirection yet -- that's Milestone 4).
//
// Dispatch is now over BUFFER coordinates (not window coordinates): gid.x/
// gid.y address a fixed memory location that never moves as the moving
// window (off_x/off_y) pans. Window/physical coordinates (needed only for
// the card SDF and the ALBC sponge, both physically anchored, not buffer-
// anchored) are derived per-thread by inverting the moving-window mapping.
// Streaming between buffer-adjacent cells is equivalent to streaming
// between window-adjacent cells because off_x/off_y is a single shift
// applied uniformly to every cell: if bx = (wx + off_x) % W for every
// cell, then the neighbor at window offset -ex lands at buffer offset -ex
// too, regardless of off_x's actual value.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_sponge.wgsl"

@group(0) @binding(0) var<storage, read>       state : CardState;
@group(0) @binding(1) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel   : array<f32>;

override W : u32;
override H : u32;
const BLOCK = 8u;

// Sponge relaxation target velocity. Default (0,0) reproduces the original
// quiescent far-field exactly; a validation scenario (e.g. cylinder in
// crossflow) sets these to a uniform freestream instead. Mirrors
// lbm_step.wgsl's SPONGE_UX/UY -- see that file for the rationale.
override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;
// Sponge ring width in cells -- see lbm_step.wgsl's identical override.
override SPONGE_W : f32 = 4.0f;

// Optional sharp bounce-back solid coupling -- mirrors lbm_step.wgsl's
// USE_BOUNCEBACK exactly (same formula, same default-0 no-op), just
// operating in this file's window/buffer split addressing. See that
// file's header for the full rationale.
override USE_BOUNCEBACK : u32 = 0u;

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// W and H are always exact multiples of BLOCK (resLog2 clamps W,H to
// powers of two >= 64), so this partitions the buffer exactly.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, 1.5f);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cx = gid.x; let cy = gid.y;
  if (cx >= W || cy >= H) { return; }

  // Window/physical coordinates: invert the moving-window shift.
  let wx = (cx + W - u32(state.off_x)) % W;
  let wy = (cy + H - u32(state.off_y)) % H;

  // Position/solid-velocity/own-cell-index terms, hoisted ABOVE the gather
  // loop -- see lbm_step.wgsl's identical hoist for why USE_BOUNCEBACK
  // needs these before streaming, not after.
  let p = vec2<f32>(f32(wx), f32(wy));
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);
  let cell = cellIndex(cx, cy);

  // 1. Pull Streaming: buffer-space neighbor (see file header derivation).
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    if (USE_BOUNCEBACK != 0u) {
      let wx_src = (wx + W - u32(ex[i])) % W;
      let wy_src = (wy + H - u32(ey[i])) % H;
      if (get_phi(vec2<f32>(f32(wx_src), f32(wy_src)), state) < 0f) {
        // Bounce-back -- see lbm_step.wgsl's identical branch for the
        // formula/derivation.
        let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
        f[i] = f_in[opp[i] * (W * H) + cell] + corr;
        continue;
      }
    }
    let bx_src = (cx + W - u32(ex[i])) % W;
    let by_src = (cy + H - u32(ey[i])) % H;
    f[i] = f_in[i * (W * H) + cellIndex(bx_src, by_src)];
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  // NaN-containment floor: divide by a positive rho so a pathological cell
  // (rho <= 0 during an incipient instability) yields a large-but-finite
  // velocity rather than Inf/NaN that streams and corrupts the whole field.
  // No-op in the valid regime (rho ~ 1). See also the tanh clamp in get_chi.
  let rhoDen = max(rho, 1e-6f);
  ux_star /= rhoDen; uy_star /= rhoDen;

  // 3. Penalty Force and Solid Coupling (window-space) -- chi forced to 0
  // under USE_BOUNCEBACK, same as lbm_step.wgsl.
  let chi = select(get_chi(phi), 0f, USE_BOUNCEBACK != 0u);

  // Penalty Force F = rho * chi * (Us - u*)
  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  // Actual fluid velocity u = u* + F/(2rho)
  let ux = ux_star + Fx / (2.0f * rhoDen);
  let uy = uy_star + Fy / (2.0f * rhoDen);
  let u_sq = ux*ux + uy*uy;

  // Store velocity for rendering (block-major buffer cell index)
  vel[cell * 2u] = ux; vel[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC Sponge (window-space distance to window edges)
  let dist_x = min(f32(wx), f32(W - 1u - wx));
  let dist_y = min(f32(wy), f32(H - 1u - wy));
  let sponge_weight = spongeWeight(dist_x, dist_y, SPONGE_W);

  let omg = 1.0f / state.tau;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    // Guo's Source Term Si = (1 - 1/(2tau)) * wi * [ (ei-u)/cs2 + (ei.u)/cs4 * ei ] . F
    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );

    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let eu_far = exf*SPONGE_UX + eyf*SPONGE_UY;
    let f_target = wt[i] * (1.0f + 3.0f*eu_far + 4.5f*eu_far*eu_far - 1.5f*(SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY)); // rho=1.0, u=(SPONGE_UX,SPONGE_UY) equilibrium

    f_out[i * (W * H) + cell] = mix(f_collide, f_target, sponge_weight);
  }
}
