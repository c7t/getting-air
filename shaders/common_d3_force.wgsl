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
// Level-1 block -> pool slot, or -1. Always bound with a single dummy -1
// element when there is no pool, exactly as common_d3_step.wgsl binds it, so

override NX : u32;
override NY : u32;
override NZ : u32;
override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;
override USE_BOUNCEBACK : u32 = 0u;
override CHI_EPS : f32 = 1.5f;

// --- THE SWEPT-CELL TERM (plans/3D.md D1) ----------------------------------
//
// THE MOMENTUM EXCHANGE OVER THE BOUNCE-BACK LINKS IS NOT THE WHOLE MOMENTUM
// TRANSFER, and for a MOVING body the part it misses is first order in the
// drag rather than a correction to it.
//
// A moving body's discrete surface is a staircase that CHANGES. Every step,
// cells on the leading face cross from fluid to solid and are overwritten
// with feq(1, u_body) by SOLID_EQ, destroying whatever fluid momentum they
// held; cells on the trailing face cross the other way and JOIN the fluid
// carrying the feq(1, u_body) that SOLID_EQ last wrote into them. Neither
// crosses a link, so neither appears in the sum above -- and the fluid's
// momentum changes anyway. Momentum that the fluid gained and the body was
// never charged for is drag the body never felt.
//
// IT IS NOT SMALL. The swept rate is A * u_body cells per step and each cell
// carries a momentum mismatch of order the near-surface slip, so the missing
// force goes as A * u_body * dU while the drag goes as (1/2) Cd A U^2. At
// D = 12, Re = 100 that ratio is ~0.3, MEASURED: the Galilean split
// (tools/probe-d3-galilean.js) reports Cd falling 31% linearly in the body's
// own speed with the grid, the Reynolds number, the domain and the blockage
// all held fixed -- which no resolution effect can do, because a resolution
// effect cannot depend on which inertial frame the same grid is described in.
// The momentum budget in a periodic sponge-free box was 48% short before this
// term and closes with it.
//
// WHY THIS IS THE WHOLE ACCOUNT. In a periodic domain with no other forcing
// the fluid's momentum can change for exactly one reason, so the body's force
// must be MINUS the fluid's momentum rate. Splitting that rate by where it
// happens gives two terms and only two: what crosses the links, and what
// changes hands. This is the second one, written out.
//
// BOUNCE-BACK ONLY. Under the diffuse (chi) coupling there is no sharp
// partition and no SOLID_EQ overwrite -- a cell in the band is penalized, not
// destroyed -- so there is nothing changing hands to account for.
//
// PROVABLY FREE ON A PINNED BODY, exactly as SOLID_EQ is: with v = 0 and
// omega = 0 the pose one step ahead IS the pose now, every cell tests equal
// and the term is identically zero. Every validated sphere case in this suite
// is pinned and is bit-identical across this change. `?swept=0` restores the
// old force for an A/B.
override SWEPT_FORCE : u32 = 1u;

// --- THE WALL DENSITY IN LADD'S CORRECTION (plans/3D.md D3) ----------------
//
// THE MOMENTUM-EXCHANGE FORCE IS GALILEAN-INVARIANT IF AND ONLY IF THE
// MOVING-WALL CORRECTION CARRIES THE LOCAL DENSITY. This is a derivation, not
// a preference, and it is short enough to write out.
//
// Boost the whole problem by a uniform velocity V. Nothing physical changes,
// so the force must not either. To first order in V the populations shift by
//
//     df_i = w_i * rho * (e_i . V) / cs^2
//
// and the body's velocity becomes u_w + V. Take the two halves of the sum
// below in turn, over the links where the source is solid:
//
//     the 2*f_opp half changes by   + SUM 2 w_i rho   e_i (e_i.V) / cs^2
//     the corr half changes by      - SUM 2 w_i rho_w e_i (e_i.V) / cs^2
//
// (the first picks up a sign because e_opp = -e_i). So the total change is
//
//     SUM 2 w_i (rho - rho_w) e_i (e_i . V) / cs^2
//
// which is EXACTLY ZERO iff rho_w is the density at that fluid node, and is
// otherwise a force linear in V that no amount of resolution removes.
//
// WITH rho_w PINNED AT 1 -- which is what both step kernels and both force
// kernels did until now, each calling it "the standard near-incompressible
// approximation" -- the residual is proportional to (rho - 1), i.e. to the
// compressibility, i.e. to Ma^2. That sounds negligible and is not: the
// momentum exchange is a small NET of large per-link terms, so a 0.3% error
// in each population, summed coherently over a whole surface, is tens of
// percent of the drag. MEASURED, on the Galilean split at fixed grid, Re,
// domain and blockage: Cd falls 15.5% at U = 0.0125 and 29.6% at U = 0.025 --
// DOUBLING with U, which is the signature this predicts (the error goes as
// (rho-1)*V ~ U^3 against a drag of U^2).
//
// AND THE FLOW IS NOT WHERE IT SHOWS. Enstrophy is exactly frame-invariant
// and matches to 0.3% between the two frames on both the dense and the pool
// path, while the reported Cd differs by 19% and 47%. So the fields agree and
// only the READING differs, which is what makes this a force-kernel defect
// and what makes the derivation above the whole story.
//
// IDENTICALLY ZERO ON A PINNED BODY, because corr itself is: u_w = 0 makes the
// whole correction vanish whatever rho_w is. Every pinned gate in this suite
// is bit-identical across it. `?rhow=0` restores the old constant for an A/B.
//
// THE STEP KERNELS TAKE THE SAME rho_w, and must: the force this kernel
// reports has to be the force the fluid actually felt. They read the same
// `f_in` at the same cell at the same instant, so the two agree exactly rather
// than approximately.
override RHO_W_LOCAL : u32 = 1u;

// --- FINEST-WINS MASKING (plans/3D.md M4.1d) -------------------------------
//
// A cell under a refined block is integrated by common_d3_force_pool.wgsl
// instead, at that level's own resolution. Summing it here as well would
// double-count the same physical volume -- the 2D sibling amr_force.wgsl:84
// carries the same mask for the same reason.
//
// NO FINEST-WINS MASKING, and its absence is the deliberate half of M5.4b.
// This kernel used to skip cells covered by the fine pool, so that the two
// force passes partitioned the integral between them. The body now lives
// ENTIRELY on the finest level as a hard requirement (plans/3D.md M5.4), so
// there is nothing to partition: when refinement is on, the host does not
// dispatch this kernel at all and the finest level's pass integrates the
// whole body. When refinement is off, there is nothing finer to be covered
// by. Either way the test was answering a question that can no longer be
// asked -- and the coupling at a coarse/fine seam through a body, which the
// partition would have had to get right, is now unreachable rather than
// merely untested.

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
    // Nearest periodic image under a moving window, and `p - c` without
    // one -- the SAME arm get_phi3 and bodyVelocity3 take, so the torque
    // cannot end up referred to a different image of the body than the
    // force that produced it.
    let r = bodyDelta3(p, body);
    let us = bodyVelocity3(p, body);

    if (USE_BOUNCEBACK != 0u) {
      // Momentum exchange (Ladd 1994 / Mei-Luo-Shyy). Only a FLUID cell
      // with at least one link into a solid neighbour contributes: a
      // solid-interior cell has no meaningful outgoing population to read
      // as momentum transfer.
      if (phi >= 0f) {
        // The wall density, from this cell's own pre-streaming populations --
        // see RHO_W_LOCAL. Only cells that can HAVE a solid neighbour pay for
        // it: a link reaches at most one cell, so phi < 2 covers every cell
        // the loop below can find one from, with a cell to spare.
        var rhoW = 1f;
        if (RHO_W_LOCAL != 0u && phi < 2f) {
          var sum = 0f;
          for (var i = 0u; i < QN; i++) { sum += f_in[i * ncells + cell]; }
          rhoW = sum;
        }
        for (var i = 0u; i < QN; i++) {
          let ei = vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
          let sp = p - ei;
          if (get_phi3(sp, body) < 0f) {
            let fOpp = f_in[opp[i] * ncells + cell];
            let corr = 2f * wt[i] * rhoW * dot(ei, us) / CS2;
            fb += -ei * (2f * fOpp + corr);
          }
        }
      }
      // THE SWEPT-CELL TERM -- see the SWEPT_FORCE header. Outside the
      // `phi >= 0` arm above on purpose: half of it lives on cells that are
      // SOLID right now and are about to be handed back to the fluid.
      if (SWEPT_FORCE != 0u) {
        let solidNow = phi < 0f;
        if (solidNow != (get_phi3Ahead(p, body, 1f) < 0f)) {
          // This cell's own momentum, as the next step will find it. For a
          // cell about to be buried that is real fluid; for one about to be
          // freed it is the feq(1, u_body) SOLID_EQ last wrote, i.e. exactly
          // u_body -- the same expression either way, which is why there is
          // one branch here and not two.
          var mc = vec3<f32>(0f);
          for (var i = 0u; i < QN; i++) {
            mc += f_in[i * ncells + cell] * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
          }
          // Fluid -> solid: the fluid is about to LOSE mc and the body is
          // what takes it. Solid -> fluid: the body gives mc back.
          fb += select(-mc, mc, !solidNow);
        }
      }
      // ONE cross product for both contributions, after both are in: they act
      // at the same arm, and the earlier `tb = cross(r, fb)` inside the link
      // loop's arm would have silently dropped the swept term's torque.
      tb = cross(r, fb);
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
