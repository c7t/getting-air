// Fine -> coarse averaging (restriction) for the L1 -> L0 hop: the parent is
// the DENSE grid. Milestone 4 (plans/AMR.md).
//
// The kernel itself is shaders/common_average.wgsl, shared with
// amr_average_pool_parent.wgsl since plans/2D-backport.md B3-2; the dense
// parent's tau and destination addressing are
// shaders/common_avg_parent_dense.wgsl. What is left here is the binding
// layout (which must match avgBGL in the pages one-to-one) and the two
// overrides only the dense side needs.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_pool      : array<u32>;
@group(0) @binding(2) var<storage, read_write> f_coarse    : array<u32>;
@group(0) @binding(3) var<storage, read>       slotToBlock : array<i32>;

override W : u32;
override H : u32;

// @include "common_avg_parent_dense.wgsl"
// @include "common_average.wgsl"
