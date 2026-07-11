// Force and torque on the solid body via integration of the penalty force.

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

@group(0) @binding(0) var<storage, read>       state  : CardState;
@group(0) @binding(1) var<storage, read>       f_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;

override W : u32;
override H : u32;
const FSCALE = 10000f;

// Optional: sharp momentum-exchange bounce-back force instead of
// integrating the diffuse penalty force -- must match lbm_step.wgsl's own
// USE_BOUNCEBACK setting (they're independent pipelines/overrides, but
// main-cylinder.js always creates them in matching pairs). Standard MEM
// formula (Ladd 1994 / Mei-Luo-Shyy): for a fluid cell with a link into a
// solid neighbor in direction i, the momentum transferred to the solid
// this step is e_i * (2*f_opp(x) + correction), where f_opp(x) is this
// cell's OWN pre-streaming population in direction opp[i] (the population
// that was heading toward that same solid neighbor -- see
// lbm_step.wgsl's identical reflection for why opp[i], not i). f_in here
// is read at the SAME point in the dispatch sequence lbm_step.wgsl reads
// it as ITS OWN f_in this macro-step (force runs before step, same buffer
// selection -- see main-cylinder.js's dispatchMacroStep), so this is
// exactly the pre-streaming, time-t data the formula needs, with no
// separate buffer-timing bookkeeping required.
override USE_BOUNCEBACK : u32 = 0u;
const opp = array<u32,9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);
const CS2 = 0.33333333f;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

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

// Sanitize NaN to 0 and clamp to the fixed-point range so the float->i32 force
// cast is well-defined on every backend (parity with amr_force.wgsl).
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
  let x = gid.x; let y = gid.y;
  
  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (x < W && y < H) {
    let bx   = (x + u32(state.off_x)) % W;
    let by   = (y + u32(state.off_y)) % H;
    let cell = by * W + bx;
    let base = cell * 9u;
    let p    = vec2<f32>(f32(x), f32(y));

    let phi = get_phi(p, state);
    let chi = get_chi(phi);

    if (USE_BOUNCEBACK != 0u) {
      // Only a FLUID cell (phi>=0) with at least one link into a solid
      // neighbor contributes -- a solid-interior cell has no meaningful
      // "own outgoing population" to interpret as momentum transfer.
      if (phi >= 0f) {
        var rx = p.x - state.cx;
        var ry = p.y - state.cy;
        rx -= f32(W) * round(rx / f32(W));
        ry -= f32(H) * round(ry / f32(H));
        let usx = state.vx - state.omega * ry;
        let usy = state.vy + state.omega * rx;

        for (var i = 0u; i < 9u; i++) {
          let wx_src = (x + W - u32(ex[i])) % W;
          let wy_src = (y + H - u32(ey[i])) % H;
          if (get_phi(vec2<f32>(f32(wx_src), f32(wy_src)), state) < 0f) {
            let f_opp = f_in[opp[i] * (W * H) + cell];
            let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
            fx_body += -f32(ex[i]) * (2f * f_opp + corr);
            fy_body += -f32(ey[i]) * (2f * f_opp + corr);
          }
        }
        tz_body = rx * fy_body - ry * fx_body;
      }
    } else if (chi >= 1e-6) {
      // Pull-gather from upstream neighbors, matching lbm_step.wgsl's
      // streaming step exactly. Reading f_in[cell] directly here (as this
      // used to) computes rho/u* from the RAW pre-streaming buffer, which
      // is a different macroscopic field from what lbm_step.wgsl uses for
      // the Guo forcing term it actually injects into the fluid, anywhere
      // there's a spatial gradient (i.e. exactly the boundary layer/wake
      // region where chi > 0) -- so the force read back here for Cd/Cl (and
      // fed to the rigid-body integration) wasn't the force actually being
      // applied to the fluid.
      var rho = 0f; var ux_star = 0f; var uy_star = 0f;
      for (var i = 0u; i < 9u; i++) {
        let wx_src = (x + W - u32(ex[i])) % W;
        let wy_src = (y + H - u32(ey[i])) % H;
        let bx_src = (wx_src + u32(state.off_x)) % W;
        let by_src = (wy_src + u32(state.off_y)) % H;
        let fi = f_in[i * (W * H) + (by_src * W + bx_src)];
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
    atomicAdd(&forces[0], safeFixed(sum_fx * FSCALE));
    atomicAdd(&forces[1], safeFixed(sum_fy * FSCALE));
    atomicAdd(&forces[2], safeFixed(sum_tz * FSCALE));
  }
}
