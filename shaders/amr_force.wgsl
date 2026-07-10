// Force and torque on the solid body via integration of the penalty force.
//
// Milestone 1 (plans/AMR.md): block-major buffer layout, same rationale and
// derivation as amr_step.wgsl's file header -- dispatch over buffer-space
// coordinates, derive window coordinates per-thread for the card SDF.
//
// Milestone 8 (plans/AMR-multilevel.md): finest-wins masking. `average`
// keeps a parent's cells populated (if coarser) under an active child, so
// summing every level's force pass unconditionally would double-count the
// same physical drag -- once crudely here at L0, once accurately at L1's
// own force pass (amr_force1.wgsl). Refinement is always whole-block
// (never partial -- decision 3's quad granularity, and L0->L1's own
// footprint-preserving 1:1 relationship), so the skip is a per-BLOCK
// check, not per-cell: if blockSlot1[this cell's block] is active, an L1
// tile already covers this block's ENTIRE footprint more accurately, so
// this pass contributes nothing for any cell in it.

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

@group(0) @binding(0) var<storage, read>       state      : CardState;
@group(0) @binding(1) var<storage, read>       f_in       : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces     : array<atomic<i32>, 4>;
@group(0) @binding(3) var<storage, read>       blockSlot1 : array<i32>; // level 1's own blockSlot -- see header

override W : u32;
override H : u32;
const FSCALE = 10000f;
const BLOCK = 8u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// See amr_step.wgsl for the full derivation.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
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
    let epsilon = 1.5f;
    // Clamp tanh arg: large |arg| overflows to NaN on some GPUs (e.g. Intel Gen12LP); saturated regime is unchanged. See PR.
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}

// Sanitize NaN to 0 and clamp to the representable fixed-point range so the
// float->i32 conversion feeding the force atomics is always well-defined
// (WGSL leaves out-of-range and NaN float->i32 conversion implementation-
// defined). FSCALE=10000 and i32 max ~2.1e9, so +/-2e9 bounds |force| < 2e5.
fn safeFixed(x: f32) -> i32 {
    let s = select(x, 0.0f, x != x);
    return i32(clamp(s, -2.0e9f, 2.0e9f));
}

var<workgroup> wg_fx : array<f32, 64>;
var<workgroup> wg_fy : array<f32, 64>;
var<workgroup> wg_tz : array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32
) {
  let cx = gid.x; let cy = gid.y;

  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (cx < W && cy < H) {
    // Finest-wins masking (see header): skip this cell's whole block if L1
    // already covers it.
    let nbx1 = W / BLOCK;
    let blockID1 = (cy / BLOCK) * nbx1 + (cx / BLOCK);
    let coveredByFiner = blockSlot1[blockID1] >= 0;

    if (!coveredByFiner) {
      let wx   = (cx + W - u32(state.off_x)) % W;
      let wy   = (cy + H - u32(state.off_y)) % H;
      let cell = cellIndex(cx, cy);
      let p    = vec2<f32>(f32(wx), f32(wy));

      let phi = get_phi(p, state);
      let chi = get_chi(phi);

      if (chi >= 1e-6) {
        // Pull-gather from buffer-space neighbors, matching amr_step.wgsl's
        // streaming step exactly (see that file's header for the buffer-space
        // vs window-space derivation). Reading f_in[cell] directly here (as
        // this used to) computes rho/u* from the RAW pre-streaming buffer,
        // a different macroscopic field from what amr_step.wgsl uses for the
        // Guo forcing term it actually injects into the fluid, anywhere
        // there's a spatial gradient -- i.e. exactly the boundary layer/wake
        // region where chi > 0 (see the equivalent lbm_force.wgsl fix).
        var rho = 0f; var ux_star = 0f; var uy_star = 0f;
        for (var i = 0u; i < 9u; i++) {
          let bx_src = (cx + W - u32(ex[i])) % W;
          let by_src = (cy + H - u32(ey[i])) % H;
          let fi = f_in[i * (W * H) + cellIndex(bx_src, by_src)];
          rho     += fi;
          ux_star += fi * f32(ex[i]);
          uy_star += fi * f32(ey[i]);
        }
        ux_star /= max(rho, 1e-6f); uy_star /= max(rho, 1e-6f); // NaN-containment floor

        // Local solid velocity Us
        var rx = p.x - state.cx;
        var ry = p.y - state.cy;
        rx -= f32(W) * round(rx / f32(W));
        ry -= f32(H) * round(ry / f32(H));
        let usx = state.vx - state.omega * ry;
        let usy = state.vy + state.omega * rx;

        // Penalty Force F = rho * chi * (Us - u*)
        let Fx = rho * chi * (usx - ux_star);
        let Fy = rho * chi * (usy - uy_star);

        // Integrate NEGATIVE of penalty force onto body
        fx_body = -Fx;
        fy_body = -Fy;
        tz_body = rx * fy_body - ry * fx_body;
      }
    }
  }

  // Workgroup reduction
  wg_fx[lid] = fx_body;
  wg_fy[lid] = fy_body;
  wg_tz[lid] = tz_body;
  workgroupBarrier();

  // Simple reduction tree or linear sum for 64 elements
  if (lid == 0u) {
    var sum_fx = 0.0f;
    var sum_fy = 0.0f;
    var sum_tz = 0.0f;
    for (var i = 0u; i < 64u; i++) {
      sum_fx += wg_fx[i];
      sum_fy += wg_fy[i];
      sum_tz += wg_tz[i];
    }
    // Clamp + NaN-sanitize before float->i32: WGSL leaves out-of-range/NaN
    // float->i32 conversion implementation-defined (Intel and NVIDIA differ),
    // so an unbounded or NaN reduction here would corrupt the body force/torque
    // backend-specifically. See safeFixed().
    atomicAdd(&forces[0], safeFixed(sum_fx * FSCALE));
    atomicAdd(&forces[1], safeFixed(sum_fy * FSCALE));
    atomicAdd(&forces[2], safeFixed(sum_tz * FSCALE));
  }
}
