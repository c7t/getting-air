// Force and torque on the solid body via integration of the penalty force.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"
// @include "common_reduce.wgsl"

@group(0) @binding(0) var<storage, read>       state  : CardState;
@group(0) @binding(1) var<storage, read>       f_in   : array<u32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;

override W : u32;
override H : u32;
// FSCALE: see shaders/amr_force1_pool.wgsl's FSCALE comment for why this
// is 1e7 and not 1e4 (per-workgroup truncation in the atomic reduction).
const FSCALE = 10000000f;

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

// THE DIFFUSE BAND'S WIDTH, as a multiple of THIS level's own cell size --
// epsilon = K_EPS * dx_level. It was a bare literal here and an override only
// on the pool path, so the one number that sets how sharp the solid boundary
// is could not be swept across the whole solver (plans/2D-backport.md B7).
//
// 1.5 is the value every one of these sites already had, so the default is
// byte-identical to the previous build. ?kEps= moves all of them together.
//
// WHY IT IS WORTH A KNOB. CLAUDE.md records `dense-reference` and
// `amr-N2-diffuse` failing Cd at Re=100 and diagnoses it as diffuse-interface
// width -- the band is a fixed number of cells regardless of resolution, so
// the effective body radius exceeds the nominal one and Cd converges from
// ABOVE. The instrument that settles that is a BAND ladder at fixed
// resolution, not a resolution ladder (which moves the band and everything
// else at once), and a band ladder needs this to be a parameter.
override K_EPS : f32 = 1.5f;
fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, K_EPS);
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
  // A THREAD OWNS A BUFFER CELL. This was the last kernel in the tree still
  // dispatched in WINDOW coordinates, converting to buffer at every load --
  // and converting back out again for every neighbour, which composed a shift
  // with its own inverse once per direction per cell:
  //
  //   window:  wx_src = (x - e) mod W,  bx_src = (wx_src + off) mod W
  //   buffer:  bx_src = (bx - e) mod W          with bx = (x + off) mod W
  //
  // Provably the same integer. This kernel touches no sponge and no walls, so
  // with the body buffer-native since B5 it needs NO window coordinate at all:
  // `state.off_x`/`off_y` do not appear in this file any more. That matters
  // beyond tidiness -- "only the render does raw off_x/off_y arithmetic" is
  // the audit that finds a kernel silently stuck in the wrong frame, and it
  // was false by exactly this one file (see plans/2D-backport.md B5-5, where
  // being in the wrong frame here read 226x wrong and no gate could see it).
  let bx = gid.x; let by = gid.y;
  
  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (bx < W && by < H) {
    let cell = by * W + bx;
    // The body's frame is this cell's own buffer position.
    let p = vec2<f32>(f32(bx), f32(by));

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
          let bx_nb = (bx + W - u32(ex[i])) % W;
          let by_nb = (by + H - u32(ey[i])) % H;
          if (get_phi(vec2<f32>(f32(bx_nb), f32(by_nb)), state) < 0f) {
            let f_opp = fUnpack(f_in[fIdx(opp[i], (W * H), cell)], opp[i]);
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
        let bx_src = (bx + W - u32(ex[i])) % W;
        let by_src = (by + H - u32(ey[i])) % H;
        let fi = fUnpack(f_in[fIdx(i, (W * H), (by_src * W + bx_src))], i);
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
  // Parallel tree reduction (common_reduce.wgsl) -- replaces a 64-step
  // serial sum that lane 0 used to run alone. See that file for the
  // on-device measurement that motivated it.
  wgReduceSum3(lid);
  if (lid == 0u) {
    let sum_fx = wg_fx[0];
    let sum_fy = wg_fy[0];
    let sum_tz = wg_tz[0];
    atomicAdd(&forces[0], safeFixed(sum_fx * FSCALE));
    atomicAdd(&forces[1], safeFixed(sum_fy * FSCALE));
    atomicAdd(&forces[2], safeFixed(sum_tz * FSCALE));
  }
}
