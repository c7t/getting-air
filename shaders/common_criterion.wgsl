// The two pieces the refinement-criterion kernels share: the vorticity
// stencil, and the 64-lane max reduction. Included by amr_criterion_pool.wgsl
// (parent = L0's dense grid) and amr_criterion_pool.wgsl (parent = a pool
// tile). plans/2D-backport.md B3-6.
//
// WHY THIS IS A SEPARATE FRAGMENT FROM common_reduce.wgsl, which is the
// obvious home for a reduction: that one's `wgReduceSum3` depends on the
// includer declaring `wg_fx`/`wg_fy`/`wg_tz`. Putting a second reduction
// beside it, with a different dependency (`wg_omega`), would make the whole
// fragment usable only by a file that declares ALL of them -- which the force
// kernels and the criterion kernels do not. Measured, not guessed: the first
// attempt did exactly that and `make wgsl` refused it with `no definition in
// scope for identifier: wg_fx`. A fragment that reads includer-declared
// globals has to be split along the same lines as those globals.
//
// AND WHY THE TWO CRITERION KERNELS ARE NOT UNIFIED, unlike the four pairs
// B3-1..B3-4 collapsed. Those differed in a parent ACCESSOR -- the same
// kernel over a differently-addressed parent. These differ in the DISPATCH
// MAPPING itself:
//
//   dense   one workgroup per L0 block -> ONE child criterion. Level 1 is
//           footprint-preserving 1:1 with L0 (plans/AMR-multilevel.md
//           decision 1), so a parent block IS a child block.
//   pool    one workgroup per QUADRANT of a parent tile's 2*RB x 2*RB
//           interior -> FOUR child criteria per parent slot.
//
// That is not an accessor; it is a different loop nest, and the asymmetry is
// the real footprint-preserving-parent decision rather than an accident of
// storage. What is genuinely shared is below, and it is the whole overlap.

// DISCRETE VORTICITY, du_y/dx - du_x/dy, from four already-fetched velocity
// samples. Central difference over +/-1 cell of whatever level the caller is
// on, hence the 0.5 -- it is per-CELL, so a caller wanting a per-L0-unit
// normalization scales afterwards (shaders/amr_render.wgsl does exactly that,
// per level, and deliberately keeps its own variants; see its comments).
//
// It takes scalars rather than doing the fetch because the fetch is the one
// thing the two kernels genuinely do differently: the dense one wraps
// periodically over the whole L0 domain, the pool one reads its tile's own
// ghost ring with no wrap at all.
fn discreteCurl(uy_xp: f32, uy_xm: f32, ux_yp: f32, ux_ym: f32) -> f32 {
  return (uy_xp - uy_xm) * 0.5f - (ux_yp - ux_ym) * 0.5f;
}

// The log2(64) = 6-step halving tree from common_reduce.wgsl, for a single
// array and with max instead of +. Result lands in wg_omega[0].
//
// Both criterion kernels ran the exact linear form that file's header exists
// to argue against --
//
//     if (lid == 0u) { for (var i = 0u; i < 64u; i++) { m = max(m, wg[i]); } }
//
// -- 64 sequential dependent ops in lane 0 while 63 lanes idle. They were
// missed when the force kernels were converted, which is its own small
// lesson: that fix went where the on-device profile pointed and not to every
// instance of the pattern.
//
// UNLIKE THE SUM, THIS IS BIT-IDENTICAL BY CONSTRUCTION. max is associative
// and exact on floats, so reordering cannot change the answer -- the numerics
// caveat common_reduce.wgsl carries does not apply here, and nothing about
// the VALUE needs re-validating, only that the kernel still runs.
//
// DEPENDS ON THE INCLUDER declaring `wg_omega` as
// `var<workgroup> array<f32, 64>`, the same arrangement wgReduceSum3 uses.
// BARRIER UNIFORMITY: must be reached by every invocation in the workgroup,
// so call it unconditionally and guard only the store that follows.
fn wgReduceMax1(lid: u32) {
  for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      wg_omega[lid] = max(wg_omega[lid], wg_omega[lid + s]);
    }
    workgroupBarrier();
  }
}
