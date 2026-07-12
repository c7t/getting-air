// Milestone 8 (plans/AMR-multilevel.md): level>=2's own force/torque
// integration -- sibling of amr_force1.wgsl (level 1), same masking/area-
// weighting reasoning (see that file's header), one shared pipeline reused
// across every level>=2 the same way amr_step1_pool.wgsl is: origin comes
// from the cached per-slot buffer (not blockID -- see
// shaders/amr_step1_pool.wgsl's header for why), and dxL/hasChild come
// from the per-level LevelParams uniform at RUNTIME, not a compile-time
// override, since ONE pipeline here serves multiple levels that may or may
// not each have their own child (e.g. at N_LEVELS=4, level 2 has a child
// but level 3, the deepest configured level, doesn't).
//
// LevelParams here is the FULL 8-field struct (unlike
// amr_interp_pool_parent.wgsl/amr_average_pool_parent.wgsl/
// amr_step1_pool.wgsl, which only declare the first 4 fields of this same
// per-level buffer) -- this is the one shader that actually reads
// hasChild.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"

struct LevelParams {
  nbx: u32,        // THIS level's own NBX -- needed here (unlike the other
  nby: u32,        // pool-parent shaders) for the finest-wins masking
                   // check below; nby itself is still unused.
  parentTau: f32,  // unused here (force doesn't touch tau at all).
  dxL: f32,        // this level's own grid spacing in L0-buffer-space units.
  hasChild: u32,   // does THIS level itself have an active child level?
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read>       state          : CardState;
@group(0) @binding(1) var<storage, read>       f_in           : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces         : array<atomic<i32>, 4>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
@group(0) @binding(4) var<storage, read>       originX        : array<f32>;
@group(0) @binding(5) var<storage, read>       originY        : array<f32>;
@group(0) @binding(6) var<uniform>             levelParams    : LevelParams;
@group(0) @binding(7) var<storage, read>       childBlockSlot : array<i32>; // level+1's blockSlot, or a harmless dummy if hasChild==0
// TEMPORARY diagnostic (level-2 bounce-back sign investigation): per-slot
// (fx,fy) written unconditionally by every dispatch -- lets the JS side
// correlate sign against each tile's own position instead of only ever
// seeing the grand total.
@group(0) @binding(8) var<storage, read_write> debugSlotForce : array<vec2<f32>>;

override W : u32;
override H : u32;
override RB : u32;
const GHOST = 2u;
const FSCALE = 10000f;
override K_EPS : f32 = 1.5f;
// Optional sharp momentum-exchange bounce-back force -- see
// amr_step1_pool.wgsl's USE_BOUNCEBACK header for the shared rationale.
override USE_BOUNCEBACK : u32 = 0u;

// BUGFIX: see amr_step1_pool.wgsl's identical comment -- this file's copy
// of the same functions had the same pre-existing hardcoded-dx=0.5 bug
// (this level's own dx varies, levelParams.dxL, not a fixed 0.5/0.25).
fn fineToCoarseUnit(fCoord: u32, origin: f32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return origin - 0.5 * levelParams.dxL + levelParams.dxL * j;
}

fn fineToCoarseUnitI(fCoordI: i32, origin: f32) -> f32 {
  let j = f32(fCoordI - i32(GHOST));
  return origin - 0.5 * levelParams.dxL + levelParams.dxL * j;
}

fn wrapf(v: f32, n: f32) -> f32 {
  var r = v % n;
  if (r < 0.0) { r += n; }
  return r;
}

fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, K_EPS * levelParams.dxL);
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
      // This level's own logical (bx,by), derived from blockID exactly like
      // amr_interp_pool_parent.wgsl/amr_step1_pool.wgsl do (levelParams.nbx
      // is THIS level's own NBX here -- unlike those files, this shader
      // does need it, for the masking check below).
      let bxOwn = u32(blockID) % levelParams.nbx;
      let byOwn = u32(blockID) / levelParams.nbx;

      var maskedByFiner = false;
      if (levelParams.hasChild != 0u) {
        let nbxChild = levelParams.nbx * 2u; // child level's own NBX = this level's NBX * 2
        // Quadrant 0's own child block ID -- if it's active, all 4 are
        // (quad allocation is all-or-nothing, see amr_force1.wgsl's header).
        let childBlockID = (byOwn * 2u) * nbxChild + (bxOwn * 2u);
        maskedByFiner = childBlockSlot[childBlockID] >= 0;
      }

      if (!maskedByFiner) {
        let originX_L0 = originX[slot];
        let originY_L0 = originY[slot];
        let bufX = fineToCoarseUnit(fx, originX_L0);
        let bufY = fineToCoarseUnit(fy, originY_L0);
        let wx = wrapf(bufX - state.off_x, f32(W));
        let wy = wrapf(bufY - state.off_y, f32(H));
        let p = vec2<f32>(wx, wy);

        let phi = get_phi(p, state);
        let poolPlaneStride = arrayLength(&f_in) / 9u;
        let cell = slot * (FB * FB) + fy * FB + fx;
        let areaWeight = levelParams.dxL * levelParams.dxL;
        // Bounce-back's MEM sum is a PERIMETER integral, not the diffuse
        // method's VOLUME integral -- dx^1, not dx^2. See amr_force1.wgsl's
        // LINE_WEIGHT comment for the full rationale and live measurement.
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
              let srcWx = wrapf(srcBufX - state.off_x, f32(W));
              let srcWy = wrapf(srcBufY - state.off_y, f32(H));
              if (get_phi(vec2<f32>(srcWx, srcWy), state) < 0f) {
                let f_opp = f_in[opp[i] * poolPlaneStride + cell];
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
              let fi = f_in[i * poolPlaneStride + srcCell];
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
    debugSlotForce[slot] = vec2<f32>(sum_fx, sum_fy);
  }
}
