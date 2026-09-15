// Force and torque on the solid body via integration of the penalty force.
//
// Milestone 1 (plans/AMR.md): block-major buffer layout, same rationale and
// derivation as amr_step.wgsl's file header -- dispatch over buffer-space
// coordinates, derive window coordinates per-thread for the card SDF.
//
// Milestone 8 (plans/AMR-multilevel.md) added FINEST-WINS MASKING here:
// `average` keeps a parent's cells populated under an active child, so
// summing every level's force pass unconditionally would double-count the
// same physical drag -- once crudely at L0, once accurately at L1's own pass
// (amr_force1.wgsl). It skipped any block whose L1 tile was active.//
// Milestone 8's FINEST-WINS MASKING IS GONE (plans/2D-backport.md B4-3).
// Only the FINEST level's force pass is dispatched now, so no coarser pass
// can double-count a body cell and there is nothing to mask against. The
// premise is the geometry-forced-refinement hard constraint -- every leaf
// within FORCE_REFINE_MARGIN of the body is already at the finest level --
// which amr2d-gpu.mjs's checkGeometryCoverageOnGPU asserts on every AMR page
// and tools/validate-amr-invariants.js gates periodically through a run.
//
// MEASURED BEFORE DELETING, not argued. debugForceBreakdown runs each
// level's pass in isolation; with the masking still in place the coarser
// levels' raw i32 accumulators (FSCALE = 1e7) read, at 8192 steps:
//
//   levels=2            L0 0            L1 237344  (finest)
//   levels=3            L0 0    L1 1    L2 214591  (finest)
//   levels=2 bounceback L0 0            L1 201702  (finest)
//   levels=3 bounceback L0 0    L1 0    L2 207126  (finest)
//
// EXACTLY zero, bar a single 1e-7 unit on one config -- one workgroup's
// truncated partial (see amr_force1_pool.wgsl's FSCALE header), 5e-6 of the
// total and ~100x below the ~1e-3 reproducibility floor AMR Cd already has.
// Dead code, demonstrated.
//
// This pass therefore only runs at all when L0 IS the finest level, i.e. a
// build with no pool at all.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"
// @include "common_reduce.wgsl"

@group(0) @binding(0) var<storage, read>       state      : CardState;
@group(0) @binding(1) var<storage, read>       f_in       : array<u32>;
@group(0) @binding(2) var<storage, read_write> forces     : array<atomic<i32>, 4>;

override W : u32;
override H : u32;
// FSCALE: see shaders/amr_force1_pool.wgsl's FSCALE comment for why this
// is 1e7 and not 1e4 (per-workgroup truncation in the atomic reduction).
const FSCALE = 10000000f;
const BLOCK = 8u;

// Optional sharp momentum-exchange bounce-back force -- mirrors
// lbm_force.wgsl's USE_BOUNCEBACK exactly (same formula); see that file's
// header for the full rationale.
override USE_BOUNCEBACK : u32 = 0u;

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// See amr_step.wgsl for the full derivation.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

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

// Sanitize NaN to 0 and clamp to the representable fixed-point range so the
// float->i32 conversion feeding the force atomics is always well-defined
// (WGSL leaves out-of-range and NaN float->i32 conversion implementation-
// defined). FSCALE=1e7 and i32 max ~2.1e9, so +/-2e9 bounds |force| < 200
// -- still ~1000x the largest force either scenario produces.
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
    {
      let p    = bodyFrameCell(vec2<u32>(cx, cy), state);
      let cell = cellIndex(cx, cy);

      let phi = get_phi(p, state);
      let chi = get_chi(phi);

      if (USE_BOUNCEBACK != 0u) {
        // See lbm_force.wgsl's identical branch for the MEM formula.
        if (phi >= 0f) {
          var rx = p.x - state.cx;
          var ry = p.y - state.cy;
          rx -= f32(W) * round(rx / f32(W));
          ry -= f32(H) * round(ry / f32(H));
          let usx = state.vx - state.omega * ry;
          let usy = state.vy + state.omega * rx;

          for (var i = 0u; i < 9u; i++) {
            let bx_src = (cx + W - u32(ex[i])) % W;
            let by_src = (cy + H - u32(ey[i])) % H;
            if (get_phi(bodyFrameCell(vec2<u32>(bx_src, by_src), state), state) < 0f) {
              let f_opp = fUnpack(f_in[fIdx(opp[i], (W * H), cell)], opp[i]);
              let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
              fx_body += -f32(ex[i]) * (2f * f_opp + corr);
              fy_body += -f32(ey[i]) * (2f * f_opp + corr);
            }
          }
          tz_body = rx * fy_body - ry * fx_body;
        }
      } else if (chi >= 1e-6) {
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
          let fi = fUnpack(f_in[fIdx(i, (W * H), cellIndex(bx_src, by_src))], i);
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
  // Parallel tree reduction (common_reduce.wgsl) -- replaces a 64-step
  // serial sum that lane 0 used to run alone. See that file for the
  // on-device measurement that motivated it.
  wgReduceSum3(lid);
  if (lid == 0u) {
    let sum_fx = wg_fx[0];
    let sum_fy = wg_fy[0];
    let sum_tz = wg_tz[0];
    // Clamp + NaN-sanitize before float->i32: WGSL leaves out-of-range/NaN
    // float->i32 conversion implementation-defined (Intel and NVIDIA differ),
    // so an unbounded or NaN reduction here would corrupt the body force/torque
    // backend-specifically. See safeFixed().
    atomicAdd(&forces[0], safeFixed(sum_fx * FSCALE));
    atomicAdd(&forces[1], safeFixed(sum_fy * FSCALE));
    atomicAdd(&forces[2], safeFixed(sum_tz * FSCALE));
  }
}
