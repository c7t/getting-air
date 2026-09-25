// Workgroup reduction of SIX per-thread partials (force xyz + torque xyz)
// for the 3D force kernel. Fragment only.
//
// The 3D counterpart of common_reduce.wgsl, which reduces three (fx, fy,
// tz) -- plans/3D.md sec 1.3 flags exactly this widening as "mechanical,
// but it touches every force kernel and their bind groups, i.e. the exact
// 238e48c failure surface". So it is a shared fragment rather than a
// hand-copy per kernel, and there is one place to get the tree right.
//
// Assumes 64 threads per workgroup (@workgroup_size(4,4,4) here, against
// the 2D kernels' (8,8) -- same count, so the same log2(64) = 6-step
// halving). See common_reduce.wgsl for the on-device measurement that made
// the tree worth having over a linear sum by lane 0.
//
// BARRIER UNIFORMITY: workgroupBarrier() must be reached by every
// invocation, so it sits OUTSIDE the `if (lid < s)` and the loop bound is a
// uniform constant. Both are required for well-defined WGSL.
//
// DEPENDS ON THE INCLUDER declaring wg_f0..wg_f5 as
// `var<workgroup> array<f32, 64>` -- naga rejects passing workgroup
// pointers into functions ("a pointer of space WorkGroup ... can't be
// passed into functions") and `make wgsl` is the gate every shader clears,
// so the same arrangement common_reduce.wgsl already uses applies here.
fn wgReduceSum6(lid: u32) {
  for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      wg_f0[lid] = wg_f0[lid] + wg_f0[lid + s];
      wg_f1[lid] = wg_f1[lid] + wg_f1[lid + s];
      wg_f2[lid] = wg_f2[lid] + wg_f2[lid + s];
      wg_f3[lid] = wg_f3[lid] + wg_f3[lid + s];
      wg_f4[lid] = wg_f4[lid] + wg_f4[lid + s];
      wg_f5[lid] = wg_f5[lid] + wg_f5[lid + s];
    }
    workgroupBarrier();
  }
}
