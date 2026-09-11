// Force AND TORQUE on the rigid body, integrated over the FINE level's
// octree pool. plans/3D.md M4.1d, and the 3D sibling of the 2D
// shaders/amr_force1.wgsl. Fragment only; the entry files list every
// include.
//
// WHY THIS EXISTS AT ALL, in two parts.
//
// 1. CORRECTNESS, on the explode path. Chen et al.'s coalesce writes only
//    the interface slots, so the coarse `f` under a refined region is stale
//    -- nothing maintains it. common_d3_force.wgsl integrating there would
//    be integrating last macro-step's data, which is why main-3d.js refused
//    ?interface=explode with a body until this landed. The coarse kernel now
//    masks those cells out (its own FINEST-WINS MASKING header) and this
//    kernel owns them.
//
// 2. ACCURACY, on every path. This is 2D Milestone 8's finding and it
//    transfers verbatim: a fixed PHYSICAL chi band under-samples at coarse
//    resolution, aliasing the very force that drives the body. More sample
//    points per unit volume is the point, so this dispatches over the full
//    pool tile shape, not at RB granularity.
//
// TWO THINGS A NAIVE COPY OF THE COARSE KERNEL GETS WRONG, both of which
// the 2D file records having actually got wrong first:
//
//   RING CELLS MUST NOT CONTRIBUTE. Unlike common_d3_amr_step1.wgsl, which
//   legitimately advects every cell because the ring has to evolve before
//   explode overwrites it, a ring cell here is either a copy of a
//   neighbour tile's interior or an exploded proxy for a coarse cell. Both
//   are owned by somebody else, and summing them double-counts a physical
//   point. Only isInterior3 contributes.
//
//   CROSS-LEVEL WEIGHTING IS dx^(D-1), WHICH IS dx^2 IN 3D -- NOT dx^3.
//   The per-cell momentum exchange below is not normalized by cell size or
//   by this level's timestep, so an unweighted sum would not integrate to
//   the same total regardless of which level owns a region. Both factors:
//
//       cell mass  ~ rho * dx^3      (3D volume measure)
//       timestep     dt = dx         (acoustic scaling, dx and dt halve together)
//       force = mass * du / dt   ->  dx^3 / dx  =  dx^2
//
//   The 2D file records an earlier version that applied only the volume
//   measure and silently dropped 1/dt, costing a factor of 2 at L1 and 4 at
//   L2 on the cylinder harness. The 3D version of that same mistake is
//   dx^3 = 0.125 here instead of dx^2 = 0.25, and it would look like a body
//   half the drag it should have. L1's dx is a fixed 0.5, so this is a
//   literal, matching amr_force1.wgsl (contrast amr_force1_pool.wgsl, whose
//   shared pipeline serves several levels and reads dxL at runtime).
//
// THE BODY LIVES IN COARSE UNITS. get_phi3 and bodyVelocity3 take L0
// coordinates, so every position here goes through fineToCoarseUnit3 --
// the same conversion common_d3_amr_step1.wgsl makes, for the same reason:
// one frame for the geometry, not a second copy of it per level.
//
// CHI_EPS ARRIVES IN COARSE UNITS AND IS HALVED, again matching the fine
// step kernel. A fine cell is half a coarse cell, so the same physical band
// is twice as many fine cells. Getting this wrong makes the body's surface
// a different width on the level that resolves it than on the level around
// it -- a discontinuity in the body itself at the interface.

@group(0) @binding(0) var<storage, read>       f_in        : array<f32>;
@group(0) @binding(1) var<storage, read>       body        : BodyState3D;
@group(0) @binding(2) var<storage, read_write> forces      : array<atomic<i32>, 8>;
@group(0) @binding(3) var<storage, read>       slotToBlock : array<i32>;

override USE_BOUNCEBACK : u32 = 0u;
override CHI_EPS : f32 = 1.5f;

// --- THE PARENT-UNIT TO L0 CONVERSION (plans/3D.md M5.4b) -----------------
//
// fineToCoarseUnit3 returns a position in the PARENT LEVEL's cell units, and
// the body lives in L0 units. At level 1 those coincide and these are the
// identity; below level 1 they do not, and the map is AFFINE rather than a
// pure scale, because refinement is CELL-CENTRED.
//
// One rung: a level-m global index g sits at parent-unit 0.5*g - 0.25, and a
// parent cell index c sits at L0 position 0.5*c - 0.25 by the same formula
// one level up. So L0 = 0.5*u - 0.25 per rung, and over (m-1) rungs
//
//     L0_SCALE  = 2^-(m-1)
//     L0_OFFSET = -0.5 * (1 - 2^-(m-1))
//
// Checked by hand at depth 3: L0 cell 0 has level-2 children at +-0.375 and
// +-0.125, and 0.25*g - 0.375 gives exactly those for g = 0..3.
//
// IT APPLIES TO EVERY POSITION IN THIS KERNEL, and the bounce-back branch
// has TWO: the cell's own, and the NEIGHBOUR's, which is what the link test
// asks about. Fixing only the first leaves the answer at exactly zero --
// every cell still fails "has a solid neighbour" -- which looks identical to
// not having fixed anything at all. It cost a debugging cycle here for that
// reason; the same transform appears twice in common_d3_amr_step1.wgsl for
// the same reason.
//
// WITHOUT THIS the kernel evaluates the body's SDF at coordinates twice too
// large per level, so every sample misses the body and the integrated force
// is EXACTLY ZERO -- which is what sphere-amr-L3 measured before it existed,
// Cd 0.0000. Silent, and indistinguishable from a body that fell outside the
// refined region, which is why that case gates both.
override L0_SCALE : f32 = 1.0f;
override L0_OFFSET : f32 = 0.0f;

// The chi band is a fixed number of CELLS at the level that resolves it, so
// it scales with this level's cell size: 0.5 at level 1, 2^-m in general.
// Only the diffuse path reads it; bounce-back is the gated coupling.
override CHI_SCALE : f32 = 0.5f;


// dx^2 for L1, i.e. 0.5^2. See the header -- this is the volume measure AND
// the timestep, not the volume measure alone.
//
// M5.4b: AN OVERRIDE, not a constant. It was baked at L1's 0.25 because L1
// was the only level that ever ran this kernel. The body now lives entirely
// on the FINEST level whatever that is (plans/3D.md M5.4), so the weight is
// 4^-m and the host sets it -- 0.25 at m=1, 0.0625 at m=2. A baked 0.25 at
// depth 3 would report four times the drag.
override DX_WEIGHT : f32 = 0.25f;

// Same fixed-point scale as the coarse kernel, and the reason it is 1e7
// rather than 1e4 bites HARDER here: the reduction atomicAdds one TRUNCATED
// i32 per WORKGROUP, and this level's per-cell contributions are already
// scaled down by DX_WEIGHT, so a workgroup whose partial falls below one
// fixed-point unit contributes exactly zero -- a systematic loss, not
// rounding noise. shaders/amr_force1_pool.wgsl's FSCALE header has the 2D
// measurement.
const FSCALE_POOL = 10000000f;

fn safeFixedPool(x: f32) -> i32 {
  let s = select(x, 0.0f, x != x);
  return i32(clamp(s, -2.0e9f, 2.0e9f));
}

var<workgroup> wg_f0 : array<f32, 64>;
var<workgroup> wg_f1 : array<f32, 64>;
var<workgroup> wg_f2 : array<f32, 64>;
var<workgroup> wg_f3 : array<f32, 64>;
var<workgroup> wg_f4 : array<f32, 64>;
var<workgroup> wg_f5 : array<f32, 64>;

@compute @workgroup_size(4, 4, 4)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  var fb = vec3<f32>(0f);
  var tb = vec3<f32>(0f);

  let FB = poolFB();
  let fz = gid.z % FB;
  let slot = gid.z / FB;
  let fi = vec3<u32>(gid.x, gid.y, fz);
  let blockID = select(-1, slotToBlock[slot], gid.x < FB && gid.y < FB);

  if (blockID >= 0 && isInterior3(fi)) {
    let b = blockXYZ(u32(blockID));
    let origin = b * RB;
    let poolPlane = arrayLength(&f_in) / QN;
    let cell = poolCell(slot, fi);

    let p = vec3<f32>(
      fineToCoarseUnit3(i32(fi.x), origin.x),
      fineToCoarseUnit3(i32(fi.y), origin.y),
      fineToCoarseUnit3(i32(fi.z), origin.z)) * L0_SCALE + L0_OFFSET;
    let phi = get_phi3(p, body);
    // Nearest periodic image under a moving window, and `p - c` without
    // one -- the SAME arm get_phi3 and bodyVelocity3 take, so the torque
    // cannot end up referred to a different image of the body than the
    // force that produced it.
    let r = bodyDelta3(p, body);
    let us = bodyVelocity3(p, body);

    if (USE_BOUNCEBACK != 0u) {
      // Momentum exchange (Ladd 1994 / Mei-Luo-Shyy), identical to the
      // coarse kernel's branch: only a FLUID cell with at least one link
      // into a solid neighbour contributes.
      if (phi >= 0f) {
        for (var i = 0u; i < QN; i++) {
          let ei = vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
          // The source in COARSE units is one FINE cell back, which is half
          // a coarse unit -- the geometry test has to be taken where the
          // link actually lands, not where a coarse link would land.
          let sp = vec3<f32>(
            fineToCoarseUnit3(i32(fi.x) - ex[i], origin.x),
            fineToCoarseUnit3(i32(fi.y) - ey[i], origin.y),
            fineToCoarseUnit3(i32(fi.z) - ez[i], origin.z)) * L0_SCALE + L0_OFFSET;
          if (get_phi3(sp, body) < 0f) {
            let fOpp = f_in[opp[i] * poolPlane + cell];
            let corr = 2f * wt[i] * dot(ei, us) / CS2;
            fb += -ei * (2f * fOpp + corr);
          }
        }
        tb = cross(r, fb);
      }
    } else {
      let chi = chiFromPhiEps3(phi, CHI_EPS * CHI_SCALE);
      if (chi >= 1e-6f) {
        // Pull-gather from upstream neighbours, matching the fine step
        // kernel's streaming, for the reason the coarse kernel's own note
        // gives: reading f_in[cell] directly computes rho/u* from a
        // different macroscopic field than the one the step kernel uses for
        // the Guo term it actually injects, anywhere there is a gradient --
        // i.e. precisely the boundary layer where chi > 0.
        //
        // A source that leaves this tile's INTERIOR is clamped rather than
        // resolved against the owning neighbour tile. That is deliberate and
        // it is safe: chi is nonzero only within a band of the body, the
        // refinement is geometry-forced so the body is never within one cell
        // of a tile edge that has no refined neighbour, and a clamp here
        // costs one cell of a gradient rather than an addressing fork of the
        // step kernel's whole neighbour-resolution hoist.
        var rho = 0f; var m = vec3<f32>(0f);
        for (var i = 0u; i < QN; i++) {
          let s = vec3<i32>(fi) - vec3<i32>(ex[i], ey[i], ez[i]);
          let sc = vec3<u32>(
            u32(clamp(s.x, 0, i32(FB) - 1)),
            u32(clamp(s.y, 0, i32(FB) - 1)),
            u32(clamp(s.z, 0, i32(FB) - 1)));
          let fv = f_in[i * poolPlane + poolCell(slot, sc)];
          rho += fv;
          m += fv * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
        }
        let ustar = m / max(rho, 1e-6f);
        fb = -(rho * chi * (us - ustar));
        tb = cross(r, fb);
      }
    }
  }

  fb *= DX_WEIGHT;
  tb *= DX_WEIGHT;

  wg_f0[lid] = fb.x; wg_f1[lid] = fb.y; wg_f2[lid] = fb.z;
  wg_f3[lid] = tb.x; wg_f4[lid] = tb.y; wg_f5[lid] = tb.z;
  workgroupBarrier();
  wgReduceSum6(lid);
  if (lid == 0u) {
    atomicAdd(&forces[0], safeFixedPool(wg_f0[0] * FSCALE_POOL));
    atomicAdd(&forces[1], safeFixedPool(wg_f1[0] * FSCALE_POOL));
    atomicAdd(&forces[2], safeFixedPool(wg_f2[0] * FSCALE_POOL));
    atomicAdd(&forces[3], safeFixedPool(wg_f3[0] * FSCALE_POOL));
    atomicAdd(&forces[4], safeFixedPool(wg_f4[0] * FSCALE_POOL));
    atomicAdd(&forces[5], safeFixedPool(wg_f5[0] * FSCALE_POOL));
  }
}
