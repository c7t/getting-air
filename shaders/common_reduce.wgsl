// Workgroup reduction for the force/torque kernels. Included via
// `// @include "common_reduce.wgsl"`. Assumes a 64-thread workgroup
// (@workgroup_size(8, 8)), which every force kernel in this project uses.
//
// WHY THIS EXISTS: all four force shaders previously reduced their 64
// per-thread partials with a LINEAR sum executed entirely by lane 0 --
//
//     if (lid == 0u) { for (var i = 0u; i < 64u; i++) { sum += wg[i]; } }
//
// -- 64 sequential, dependent float adds per array, three arrays, so 192
// serialised adds while the other 63 lanes idle. Their own comment
// ("Simple reduction tree or linear sum for 64 elements") shows the tree was
// considered and the linear form chosen.
//
// On a wide desktop GPU that latency hides behind other resident workgroups
// and the force passes look no worse than the step kernels. On the mobile
// PowerVR part this project actually targets it does not hide: a per-pass
// GPU profile taken on-device (via the ?telemetry=1&profile=1 back channel)
// put `force L0` and `force L1` at 18.6% of the macro-step EACH -- four
// times the cost of `L0 step`, which does a full LBM collide+stream over all
// 65,536 cells, while a force pass only evaluates chi and reduces. Work per
// cell is strictly lower; only the reduction differs.
//
// The tree below is the standard log2(64) = 6-step halving. Every lane stays
// busy for the first steps and the dependent chain drops from 64 to 6.
//
// BARRIER UNIFORMITY: workgroupBarrier() must be reached by every invocation
// in the workgroup, so it sits OUTSIDE the `if (lid < s)`, and the loop bound
// is a uniform constant -- both required for well-defined WGSL.
//
// NUMERICS: this changes the summation ORDER, so results differ from the
// linear version in the last bits of the float. Pairwise/tree summation is
// if anything more accurate than a long sequential chain (shorter dependent
// error accumulation), and the sum is quantised to fixed point immediately
// afterwards anyway. Validated against the cylinder harness's Cd/St bands
// rather than assumed harmless.
// DEPENDS ON THE INCLUDER declaring `wg_fx`, `wg_fy` and `wg_tz` as
// `var<workgroup> array<f32, 64>` -- the same arrangement common_geometry.wgsl
// already uses for the includer's own `override W`/`override H`. The
// alternative, passing them as ptr<workgroup, ...> parameters, is legal WGSL
// but naga rejects it ("a pointer of space WorkGroup ... can't be passed into
// functions"), and `make wgsl` is the gate every shader here has to clear.
// Module-scope name resolution in WGSL is order-independent, so this
// resolves regardless of where the include lands in the file.
fn wgReduceSum3(lid: u32) {
  for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      wg_fx[lid] = wg_fx[lid] + wg_fx[lid + s];
      wg_fy[lid] = wg_fy[lid] + wg_fy[lid + s];
      wg_tz[lid] = wg_tz[lid] + wg_tz[lid + s];
    }
    workgroupBarrier();
  }
}
