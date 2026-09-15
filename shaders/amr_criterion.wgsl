// Milestone 4b (plans/AMR.md): per-coarse-block refinement criterion.
//
// One workgroup per coarse block (RB*RB=8*8=64 cells=64 threads, same "one
// workgroup = one block" convention amr_average_f2c.wgsl uses), computing
// local vorticity from the CURRENT coarse velocity field via buffer-space
// periodic neighbor differences (no window conversion needed -- unlike the
// card SDF, this is pure flow-field data, and M1's coarse buffer is
// already buffer-space-native), then reducing to a single max|omega| per
// block via workgroup shared memory (same reduction pattern
// shaders/lbm_force.wgsl already uses for its force accumulator).
//
// Simplified AGAL Algorithm 3: the general form computes a desired level
// from log2|omega| against a ladder of Nstart-p*Ninc thresholds across
// Lmax levels. With only 2 levels here (coarse + 1 fine, no Milestone 3),
// that ladder collapses to a single number -- this shader just outputs
// max|omega| per block; amr_manage.wgsl does the (now-trivial) threshold
// comparison.
//
// NOT UNIFIED WITH amr_criterion_pool.wgsl, and plans/2D-backport.md B3-6
// records why: unlike the other four pairs B3 collapsed, these two do not
// differ in a parent ACCESSOR, they differ in the DISPATCH MAPPING. This one
// is one workgroup per L0 block producing ONE child criterion (level 1 is
// footprint-preserving 1:1 with L0 -- decision 1); the pool one is one
// workgroup per QUADRANT of a parent tile, producing FOUR. The shared part is
// the stencil and the reduction, and both are now in common_criterion.wgsl.

// @include "common_criterion.wgsl"

@group(0) @binding(0) var<storage, read>       vel             : array<f32>;
@group(0) @binding(1) var<storage, read_write> blockCriterion  : array<f32>;

override W : u32;
override H : u32;
const BLOCK = 8u;

fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

fn get_ux(cx: u32, cy: u32) -> f32 { return vel[cellIndex(cx % W, cy % H) * 2u]; }
fn get_uy(cx: u32, cy: u32) -> f32 { return vel[cellIndex(cx % W, cy % H) * 2u + 1u]; }

// Read by common_criterion.wgsl's wgReduceMax1 -- see its own comment.
var<workgroup> wg_omega : array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wgid: vec3<u32>
) {
  let cx = gid.x; let cy = gid.y;
  let xp1 = (cx + 1u) % W; let xm1 = (cx + W - 1u) % W;
  let yp1 = (cy + 1u) % H; let ym1 = (cy + H - 1u) % H;

  let omega = discreteCurl(get_uy(xp1, cy), get_uy(xm1, cy),
                           get_ux(cx, yp1), get_ux(cx, ym1));

  wg_omega[lid] = abs(omega);
  workgroupBarrier();
  wgReduceMax1(lid);

  if (lid == 0u) {
    let m = wg_omega[0];
    let nbx = W / BLOCK;
    let blockID = wgid.y * nbx + wgid.x;
    blockCriterion[blockID] = m;
  }
}
