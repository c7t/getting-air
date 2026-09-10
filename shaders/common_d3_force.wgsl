// Force AND TORQUE on the 3D rigid body -- six scalars, against the 2D
// kernels' three (fx, fy, tz). plans/3D.md sec 1.3 flags this widening as
// "mechanical, but it touches every force kernel and their bind groups --
// i.e. the exact 238e48c failure surface CLAUDE.md warns about", which is
// why the reduction is a shared fragment (common_d3_reduce.wgsl) and why
// index-3d.html is in validate-all.js's boot smoke.
//
// Fragment only; the ENTRY files (d3_force_q{19,27}.wgsl) list every
// fragment, since shader-loader.mjs splices exactly one level.
//
// Both coupling methods, matching common_d3_step.wgsl's USE_BOUNCEBACK
// exactly -- they are independent pipelines over the same override, and
// main-3d.js always creates them as a matching pair. A mismatch would
// integrate a force the fluid never felt.
//
// DISPATCH ORDER MATTERS: this runs BEFORE the step kernel in each
// macro-step and reads the same f_in the step kernel will, so `f_in` here
// is the pre-streaming, time-t data both formulas want, with no separate
// buffer-timing bookkeeping. Same arrangement as the 2D main-cylinder.js.

@group(0) @binding(0) var<storage, read>       f_in   : array<f32>;
@group(0) @binding(1) var<storage, read>       body   : BodyState3D;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 8>;

override NX : u32;
override NY : u32;
override NZ : u32;
override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;
override USE_BOUNCEBACK : u32 = 0u;
override CHI_EPS : f32 = 1.5f;

// Fixed-point scale for the atomic reduction. 1e7, matching the 2D
// force kernels -- see shaders/amr_force1_pool.wgsl's FSCALE header for why
// the smaller 1e4 was not enough: the truncation is PER WORKGROUP, so the
// error accumulates with the workgroup count rather than averaging out.
// 3D has far more workgroups per body than 2D did, which makes this
// strictly more important here, not less.
const FSCALE = 10000000f;

// NaN -> 0 and clamp into the fixed-point range, so the f32 -> i32 cast is
// well-defined on every backend (parity with the 2D force kernels).
fn safeFixed(x: f32) -> i32 {
  let s = select(x, 0.0f, x != x);
  return i32(clamp(s, -2.0e9f, 2.0e9f));
}

var<workgroup> wg_f0 : array<f32, 64>;
var<workgroup> wg_f1 : array<f32, 64>;
var<workgroup> wg_f2 : array<f32, 64>;
var<workgroup> wg_f3 : array<f32, 64>;
var<workgroup> wg_f4 : array<f32, 64>;
var<workgroup> wg_f5 : array<f32, 64>;

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  var fb = vec3<f32>(0f);
  var tb = vec3<f32>(0f);

  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x < NX && y < NY && z < NZ) {
    let ncells = NX * NY * NZ;
    let cell = (z * NY + y) * NX + x;
    let p = vec3<f32>(f32(x), f32(y), f32(z));
    let phi = get_phi3(p, body);
    let r = p - vec3<f32>(body.cx, body.cy, body.cz);
    let us = bodyVelocity3(p, body);

    if (USE_BOUNCEBACK != 0u) {
      // Momentum exchange (Ladd 1994 / Mei-Luo-Shyy). Only a FLUID cell
      // with at least one link into a solid neighbour contributes: a
      // solid-interior cell has no meaningful outgoing population to read
      // as momentum transfer.
      if (phi >= 0f) {
        for (var i = 0u; i < QN; i++) {
          let ei = vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
          let sp = p - ei;
          if (get_phi3(sp, body) < 0f) {
            let fOpp = f_in[opp[i] * ncells + cell];
            let corr = 2f * wt[i] * dot(ei, us) / CS2;
            fb += -ei * (2f * fOpp + corr);
          }
        }
        tb = cross(r, fb);
      }
    } else {
      let chi = chiFromPhiEps3(phi, CHI_EPS);
      if (chi >= 1e-6f) {
        // Pull-gather from upstream neighbours, matching the step kernel's
        // streaming EXACTLY. Reading f_in[cell] directly instead would
        // compute rho/u* from the raw pre-streaming buffer -- a different
        // macroscopic field from the one the step kernel uses for the Guo
        // term it actually injects, anywhere there is a spatial gradient,
        // i.e. precisely the boundary layer where chi > 0. The 2D kernel
        // carries this same note because it was a real bug there: the force
        // read back for Cd was not the force being applied to the fluid.
        var rho = 0f; var m = vec3<f32>(0f);
        for (var i = 0u; i < QN; i++) {
          let sx = u32((i32(x) - ex[i] + i32(NX)) % i32(NX));
          let sy = u32((i32(y) - ey[i] + i32(NY)) % i32(NY));
          let sz = u32((i32(z) - ez[i] + i32(NZ)) % i32(NZ));
          let fi = f_in[i * ncells + ((sz * NY + sy) * NX + sx)];
          rho += fi;
          m += fi * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
        }
        let ustar = m / max(rho, 1e-6f);   // NaN-containment floor
        // Penalty force on the FLUID is rho chi (Us - u*); the body feels
        // its negative.
        fb = -(rho * chi * (us - ustar));
        tb = cross(r, fb);
      }
    }
  }

  wg_f0[lid] = fb.x; wg_f1[lid] = fb.y; wg_f2[lid] = fb.z;
  wg_f3[lid] = tb.x; wg_f4[lid] = tb.y; wg_f5[lid] = tb.z;
  workgroupBarrier();
  wgReduceSum6(lid);
  if (lid == 0u) {
    atomicAdd(&forces[0], safeFixed(wg_f0[0] * FSCALE));
    atomicAdd(&forces[1], safeFixed(wg_f1[0] * FSCALE));
    atomicAdd(&forces[2], safeFixed(wg_f2[0] * FSCALE));
    atomicAdd(&forces[3], safeFixed(wg_f3[0] * FSCALE));
    atomicAdd(&forces[4], safeFixed(wg_f4[0] * FSCALE));
    atomicAdd(&forces[5], safeFixed(wg_f5[0] * FSCALE));
  }
}
