// THE force/torque integration over a level's own pool tiles -- ONE kernel,
// every pool level (plans/2D-backport.md B3-4). Until then this was the
// level>=2 half of a pair, with a separate amr_force1.wgsl compiled and
// dispatched for level 1 alone.
//
// It collapsed for exactly the reasons B3-1 collapsed the fine-step pair, and
// the two files had exactly the same four differences -- origin, dxL, the
// diffuse band, and (here) the area/line weight, which IS dxL. The origin was
// the only structural one, and it was a wrong claim: a tile's physical origin
// is `block * RB * 2^-(m-1)` in closed form, i.e. `f32(bx*RB) * 2 *
// levelParams.dxL`, so the per-slot originX/originY buffers this used to read
// are gone from it -- and with the fine step off them since B3-1a, THIS WAS
// THEIR LAST READER. See plans/2D-backport.md B3-4 for what that retires.
//
// Generalizes amr_force.wgsl (L0's own, which stays: L0 is a dense,
// ghost-free grid) the same way amr_step1.wgsl generalizes amr_step.wgsl --
// same momentum-exchange math, dispatched over pool tiles (full FB*FB shape,
// Z=slot, same as amr_step1.wgsl and NOT amr_average_f2c.wgsl's
// RB-granularity one, since MORE sample points per unit area is the entire
// point of Milestone 8: a fixed physical epsilon under-sampled the chi
// transition band at coarse resolution, aliasing the force/torque that drives
// the body's own trajectory).
//
// TWO THINGS A NAIVE PER-LEVEL COPY OF amr_force.wgsl WOULD GET WRONG, and
// they are why this is not simply amr_force.wgsl with a different binding:
//
// 1. GHOST cells must NOT contribute. Unlike amr_step1.wgsl (which
//    legitimately collides/streams every cell, ghost included, since ghost
//    cells still need to evolve before the next ghost-refresh overwrites
//    them), a ghost cell here is either a copy of a NEIGHBOR's interior
//    (same-level fine-fine ghost) or a coarse-interpolated proxy -- summing
//    force there would double-count against whichever cell actually OWNS
//    that physical point. Only isInterior cells contribute.
//
// 2. Cross-level weighting. Fx/Fy here are a per-CELL momentum exchange, not
//    normalized by cell size or by this level's own timestep, so a raw
//    unweighted sum would not integrate to the same total regardless of which
//    level owns a region (the exact invariance Milestone 8's own validation
//    checks). The weight is dx_L^1, and BOTH factors matter:
//
//      cell mass  ~ rho * dx_L^2   (2D volume measure)
//      timestep     dt_L = dx_L    (acoustic scaling: dx and dt halve together)
//      force = mass * du / dt   ->  dx_L^2 / dx_L  =  dx_L
//
//    An earlier version used dx_L^2, applying only the volume measure and
//    silently dropping the 1/dt_L factor -- level L runs 2^L substeps per L0
//    macro-step, but this pass runs ONCE per macro-step and reads one
//    substep's momentum exchange, so the missing factor is exactly 2^L =
//    1/dx_L. Measured on the cylinder harness at Re=100: that bug cost 2x at
//    L1 and 4x at L2 (Cd 0.943 -> 1.430 at N=2, and the N=3 case went from
//    unusable to inside the literature band). The bounce-back branch uses the
//    SAME dx^1 -- not (as its old comment claimed) because one is a perimeter
//    integral and the other a volume integral, but because the mass and
//    timestep factors combine to dx^1 either way. Live-verified there too:
//    dx^2 gave Cd=0.631 (target 1.35) on the N=2 cylinder case, dx^1 gives
//    1.262.
//
// FINEST-WINS MASKING IS GONE (plans/2D-backport.md B4-3), along with its
// HAS_CHILD override, the childBlockSlot binding it read, and the levelParams
// nbx/nby/hasChild reads that served it. Only the finest level's force pass is
// dispatched now -- see amr_force.wgsl's header for the measurement that
// showed every coarser pass already contributing exactly zero.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"
// @include "common_reduce.wgsl"

// The shared 32-byte per-level uniform, same buffer every pool shader reads.
// This declares through kEps at offset 20; the fields before it that this
// kernel does not use are declared because WGSL has no way to skip them.
struct LevelParams {
  nbx: u32,        // this level's own block-grid extent -- used to derive the
                   // tile's own (bx,by), and with dxL its physical origin.
  nby: u32,        // unused here.
  parentTau: f32,  // unused here (force doesn't touch tau at all).
  dxL: f32,        // this level's own grid spacing in L0-buffer-space units:
                   // the diffuse band, the area/line weight, and half the
                   // origin derivation all scale with it.
  hasChild: u32,   // unused since the finest-wins masking went (see header);
                   // declared only to reach kEps.
  kEps: f32,       // the diffuse band in units of this level's dx. A per-level
                   // uniform, not an override -- one pipeline serves every
                   // level, so a compile-time constant could not say anything
                   // per-level. See shaders/amr_step1.wgsl's get_chi.
}

@group(0) @binding(0) var<storage, read>       state          : CardState;
@group(0) @binding(1) var<storage, read>       f_in           : array<u32>;
@group(0) @binding(2) var<storage, read_write> forces         : array<atomic<i32>, 4>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
@group(0) @binding(4) var<uniform>             levelParams    : LevelParams;
// Diagnostic (level-2 bounce-back sign investigation): per-slot (fx,fy)
// written unconditionally by every dispatch -- lets the JS side correlate
// sign against each tile's own position instead of only ever seeing the
// grand total (debugReadSlotForces).
@group(0) @binding(5) var<storage, read_write> debugSlotForce : array<vec2<f32>>;
// RENUMBERED CONTIGUOUS by B3-4. The layout had holes: 4/5 were
// originX/originY (gone -- the origin is derived, see header), 7 was the
// masking's childBlockSlot (gone in B4-3) and 8 sat past the hole because
// renumbering meant landing three pages' bind groups in lockstep. It is five
// pages now and one layout instead of two, so it is done once, here, with
// boot smoke on every page as the gate.

override W : u32;
override H : u32;
override RB : u32;
const GHOST = 2u;
// FSCALE: fixed-point scale for the atomic force accumulation. Raised from
// 1e4 to 1e7 because the reduction below atomicAdds ONE TRUNCATED i32 PER
// WORKGROUP (safeFixed's i32() cast truncates toward zero), so any workgroup
// whose partial sum falls below one fixed-point unit contributes exactly
// zero -- a systematic, one-directional loss, not rounding noise. Per-cell
// contributions shrink with the level's own dx weight, so deeper levels hit
// that floor hardest: measured on the cylinder harness at Re=100, at 1e4 the
// truncation cost ~10% of the force at L1 and ~32% at L2 (Cd 1.430 -> 1.593
// at N=2, 0.943 -> 1.390 at N=3). i32 max ~2.1e9 against the +/-2e9 clamp
// still bounds |force| < 200, ~1000x the largest force either scenario
// produces. A deeper hierarchy would eventually need a real fix (float
// atomics via CAS, or a two-stage reduction) rather than more scale.
const FSCALE = 10000000f;
// Optional sharp momentum-exchange bounce-back force -- see
// amr_step1.wgsl's USE_BOUNCEBACK header for the shared rationale.
override USE_BOUNCEBACK : u32 = 0u;

// Cell-centred refinement, shared with amr_step1.wgsl: the two children of
// parent cell c sit at c -/+ dx/2, so tile-local fine index j maps to
// origin - dx/2 + dx*(j - GHOST). amr2d.mjs's fineToCoarseUnit is the host
// twin. (Both files once hardcoded level 1's own dx=0.5 here, which was a
// real bug for every deeper level -- see amr_step1.wgsl.)
fn fineToCoarseUnit(fCoord: u32, origin: f32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return origin - 0.5 * levelParams.dxL + levelParams.dxL * j;
}

fn fineToCoarseUnitI(fCoordI: i32, origin: f32) -> f32 {
  let j = f32(fCoordI - i32(GHOST));
  return origin - 0.5 * levelParams.dxL + levelParams.dxL * j;
}

fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, levelParams.kEps * levelParams.dxL);
}

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
  let fx = gid.x; let fy = gid.y;
  let slot = gid.z;
  let FB = RB * 2u + 2u * GHOST;

  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (fx < FB && fy < FB) {
    let blockID = slotToBlock[slot];
    let isInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;

    if (blockID >= 0 && isInterior) {
      {
        // This tile's physical origin in L0 units, as one multiply --
        // `block * RB * 2^-(m-1)`, and `2^-(m-1)` is `2 * dxL`. It used to
        // be a per-slot buffer read; see the header, and amr2d.mjs's
        // tileOriginL0 for the host statement of the same closed form.
        let bx = u32(blockID) % levelParams.nbx;
        let by = u32(blockID) / levelParams.nbx;
        let originX_L0 = f32(bx * RB) * 2.0f * levelParams.dxL;
        let originY_L0 = f32(by * RB) * 2.0f * levelParams.dxL;
        let bufX = fineToCoarseUnit(fx, originX_L0);
        let bufY = fineToCoarseUnit(fy, originY_L0);
        let p = vec2<f32>(bufX, bufY);

        let phi = get_phi(p, state);
        let poolPlaneStride = arrayLength(&f_in) / 9u;
        let cell = slot * (FB * FB) + fy * FB + fx;
        // dx_L^1 for BOTH branches: a cell's mass scales as dx_L^2 but this
        // level's timestep is dt_L = dx_L (acoustic scaling), and force is
        // mass*du/dt, so the two factors combine to dx_L^1. See
        // amr_force1.wgsl's header point 2 -- the diffuse branch previously
        // used dx_L^2, applying the volume measure but dropping 1/dt_L,
        // which cost a factor of 2^L (4x at level 2).
        let areaWeight = levelParams.dxL;
        let lineWeight = levelParams.dxL;

        if (USE_BOUNCEBACK != 0u) {
          // See lbm_force.wgsl's identical branch for the MEM formula;
          // amr_step1.wgsl's own USE_BOUNCEBACK header for why the sharp
          // test uses the UNCLAMPED source position.
          if (phi >= 0f) {
            var rx = p.x - state.cx;
            var ry = p.y - state.cy;
            rx -= f32(W) * round(rx / f32(W));
            ry -= f32(H) * round(ry / f32(H));
            let usx = state.vx - state.omega * ry;
            let usy = state.vy + state.omega * rx;

            for (var i = 0u; i < 9u; i++) {
              let srcBufX = fineToCoarseUnitI(i32(fx) - ex[i], originX_L0);
              let srcBufY = fineToCoarseUnitI(i32(fy) - ey[i], originY_L0);
              if (get_phi(vec2<f32>(srcBufX, srcBufY), state) < 0f) {
                let f_opp = fUnpack(f_in[fIdx(opp[i], poolPlaneStride, cell)], opp[i]);
                let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
                fx_body += -f32(ex[i]) * (2f * f_opp + corr) * lineWeight;
                fy_body += -f32(ey[i]) * (2f * f_opp + corr) * lineWeight;
              }
            }
            tz_body = rx * fy_body - ry * fx_body;
          }
        } else {
          let chi = get_chi(phi);
          if (chi >= 1e-6) {
            var rho = 0f; var ux_star = 0f; var uy_star = 0f;
            for (var i = 0u; i < 9u; i++) {
              let srcX = clamp(i32(fx) - ex[i], 0, i32(FB) - 1);
              let srcY = clamp(i32(fy) - ey[i], 0, i32(FB) - 1);
              let srcCell = slot * (FB * FB) + u32(srcY) * FB + u32(srcX);
              let fi = fUnpack(f_in[fIdx(i, poolPlaneStride, srcCell)], i);
              rho     += fi;
              ux_star += fi * f32(ex[i]);
              uy_star += fi * f32(ey[i]);
            }
            ux_star /= max(rho, 1e-6f); uy_star /= max(rho, 1e-6f);

            var rx = p.x - state.cx;
            var ry = p.y - state.cy;
            rx -= f32(W) * round(rx / f32(W));
            ry -= f32(H) * round(ry / f32(H));
            let usx = state.vx - state.omega * ry;
            let usy = state.vy + state.omega * rx;

            let Fx = rho * chi * (usx - ux_star);
            let Fy = rho * chi * (usy - uy_star);

            fx_body = -Fx * areaWeight;
            fy_body = -Fy * areaWeight;
            tz_body = rx * fy_body - ry * fx_body;
          }
        }
      }
    }
  }

  wg_fx[lid] = fx_body;
  wg_fy[lid] = fy_body;
  wg_tz[lid] = tz_body;
  workgroupBarrier();

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
    debugSlotForce[slot] = vec2<f32>(sum_fx, sum_fy);
  }
}
