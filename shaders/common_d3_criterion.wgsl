// THE Q-CRITERION, shader side: flag the blocks a DETACHED VORTEX needs.
// plans/3D.md M8.4. Fragment only; the entry files list every include.
//
// Host counterpart and the full argument: d3-criterion.mjs. The short
// version, because it is the part that decides the shape of this file:
//
//   THE DESIGN INPUT IS DETACHED VORTICES. A geometry-forced set is one
//   compact region around the body; the target regime (Re ~ 1100, tight
//   isolated vortices shed into the wake) needs a set that appears where the
//   body is NOT and that MOVES WITH THE FLOW. So this criterion is a field
//   test, evaluated everywhere, and the set it produces is deliberately not
//   compact -- which is also what invalidates M6.4's bounding boxes.
//
// IT READS THE DENSE L0 `mac`, AND THAT IS THE DESIGN, not a shortcut.
// A criterion has to answer "should this block be refined" where the block is
// NOT YET refined, so the field it reads must exist independently of the
// answer. L0 is the only level with that property -- and it holds a valid
// field everywhere including UNDER refinement, because the coalesce chain
// republishes each level's moments into its parent's `mac` (measured: with
// ?refine=all at ?levels=3 the dense array is written entirely by two
// coalesce hops and still scores 5.59e-3 against the analytic Beltrami
// solution). A criterion reading a level's own pool could only ever confirm
// refinement that already exists.
//
// The cost of that choice is stated rather than hidden: a vortex is flagged
// from its L0 footprint, so the localization is one L0 cell, and at a depth
// where a finest block is smaller than an L0 cell (RB * 2^-(m-1) < 1, i.e.
// m > log2(RB) + 1) this cannot resolve which sibling to flag and will flag
// all of them. The host refuses that case rather than silently over-refining.
//
// ONE WORKGROUP PER CANDIDATE BLOCK, striding over the L0 cells it covers.
// Neighbouring blocks re-read overlapping cells because of the convection
// dilation below, which is redundant work -- but this runs once every
// MANAGE_EVERY macro-steps, against a step kernel that runs every step, and
// a separate per-cell Q buffer would cost a full field's worth of memory and
// bandwidth to avoid it. Priced deliberately, not overlooked.

// The dense L0 macroscopic field, [rho, ux, uy, uz] per cell. INTERLEAVED,
// and the vec4 type is what says so -- see common_d3_parentmac_dense.wgsl
// for what confusing this with a pool's planar layout cost once already.
@group(0) @binding(0) var<storage, read>       macL0     : array<vec4<f32>>;
// PER-BLOCK MAX Q, not a boolean want, and the difference is deliberate --
// it is also the shape 2D's amr_criterion_pool.wgsl settled on
// (`childCriterion : array<f32>`).
//
// Folding the threshold in here would make every question about the
// threshold cost a GPU pass: "how many blocks does qthresh=0.05 flag" and
// "how many does 0.02 flag" are then two runs of the reduction rather than
// two lines of host arithmetic on one readback. That matters because THE
// THRESHOLD IS THE UNMEASURED PARAMETER -- it sets the size of the refined
// set, which sets the slot budget, and the budget is currently derived from
// the INITIAL set (main-3d.js's maxSlotsAt), i.e. from a domain with no wake
// in it yet. Sizing that honestly needs a sweep, and this makes the sweep
// free.
//
// It also keeps this kernel from having to know that geometry has already
// written a want somewhere: the union of "near the body" and "inside a
// vortex" is taken by whoever consumes both, not by whichever happens to run
// second.
@group(0) @binding(1) var<storage, read_write> blockQ : array<f32>;

override NX : u32;
override NY : u32;
override NZ : u32;

// L0 cells per edge of a candidate block at the level being flagged:
// RB * 2^-(m-1). 4 at level 1 with RB=4, 2 at level 2, 1 at level 3.
override BLK_L0 : f32 = 4.0f;
// Candidate-block counts on the level being flagged.
override NBX : u32 = 1u;
override NBY : u32 = 1u;
override NBZ : u32 = 1u;

// HOW FAR AHEAD TO FLAG, in L0 cells. A detached vortex convects; a set that
// only covers where it IS is stale before the next management event, exactly
// as common_d3_manage.wgsl's geometry criterion is without its own
// `MANAGE_EVERY * |v|` lead term. Zero reproduces the undilated set.
override LEAD : f32 = 0.0f;

var<workgroup> wg_q : array<f32, 64>;

fn l0Cell(x: i32, y: i32, z: i32) -> u32 {
  let cx = u32(((x % i32(NX)) + i32(NX)) % i32(NX));
  let cy = u32(((y % i32(NY)) + i32(NY)) % i32(NY));
  let cz = u32(((z % i32(NZ)) + i32(NZ)) % i32(NZ));
  return (cz * NY + cy) * NX + cx;
}

fn uAt(x: i32, y: i32, z: i32) -> vec3<f32> { return macL0[l0Cell(x, y, z)].yzw; }

// Q = 1/2(|Omega|^2 - |S|^2) at an L0 cell, by centred differences over the
// six face neighbours. The step is ONE L0 CELL, so Q comes out in L0 units,
// and so does any threshold compared against it -- the criterion is a
// statement about the flow, and the flow does not know what level is asking.
//
// Written as the difference of the two Frobenius norms rather than the
// equivalent -1/2 J_ij J_ji: the compact form is one transposition away from
// being silently wrong, and the two norms are separately testable.
// tools/test-d3-criterion.js scores the host twin of this against closed-form
// answers, including the one that matters -- PURE SHEAR IS EXACTLY ZERO,
// which is the entire reason this is Q and not |omega|.
fn qAt(x: i32, y: i32, z: i32) -> f32 {
  var J : array<vec3<f32>, 3>;             // J[j] = du/dx_j
  J[0] = (uAt(x + 1, y, z) - uAt(x - 1, y, z)) * 0.5f;
  J[1] = (uAt(x, y + 1, z) - uAt(x, y - 1, z)) * 0.5f;
  J[2] = (uAt(x, y, z + 1) - uAt(x, y, z - 1)) * 0.5f;
  var s2 = 0f; var om2 = 0f;
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      // J[j][i] is du_i/dx_j.
      let a = J[j][i];
      let b = J[i][j];
      let S = 0.5f * (a + b);
      let O = 0.5f * (a - b);
      s2 += S * S;
      om2 += O * O;
    }
  }
  return 0.5f * (om2 - s2);
}

@compute @workgroup_size(64)
fn reduceQ(@builtin(workgroup_id) wg: vec3<u32>,
         @builtin(local_invocation_index) lid: u32) {
  if (wg.x >= NBX || wg.y >= NBY || wg.z >= NBZ) { return; }

  // The L0 box this block covers, dilated by the convection lead. floor/ceil
  // rather than round: the dilated box must CONTAIN the region, and a block
  // that is half a cell short of a vortex is a block that does not refine.
  let lo = vec3<f32>(vec3<u32>(wg.x, wg.y, wg.z)) * BLK_L0 - LEAD;
  let hi = lo + BLK_L0 + 2f * LEAD;
  let i0 = vec3<i32>(floor(lo));
  let i1 = vec3<i32>(ceil(hi));
  let n = i1 - i0;
  let total = u32(max(n.x, 0) * max(n.y, 0) * max(n.z, 0));

  var best = -1e30f;
  for (var k = lid; k < total; k += 64u) {
    let kx = i32(k) % n.x;
    let ky = (i32(k) / n.x) % n.y;
    let kz = i32(k) / (n.x * n.y);
    best = max(best, qAt(i0.x + kx, i0.y + ky, i0.z + kz));
  }
  wg_q[lid] = best;
  workgroupBarrier();
  // Tree reduction over the 64 lanes, the same shape common_d3_reduce.wgsl
  // uses for the force sum.
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid < s) { wg_q[lid] = max(wg_q[lid], wg_q[lid + s]); }
    workgroupBarrier();
  }
  if (lid != 0u) { return; }
  // ASSIGNED, not accumulated: this array is this kernel's own answer about
  // the flow, and it is recomputed from scratch every time the criterion
  // runs. The union with the geometry-forced set happens downstream, where
  // both answers are in hand.
  blockQ[(wg.z * NBY + wg.y) * NBX + wg.x] = wg_q[0];
}
