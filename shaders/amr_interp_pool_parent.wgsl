// Coarse -> fine ghost-cell interpolation for EVERY L(m) -> L(m+1) hop: the
// parent is a POOL tile. Milestone 6 (plans/AMR-multilevel.md / -M5.md); m>=1
// only until U7-6f, when the dense L0 grid and its own interp entry file went
// and the root became the parent of level 1 like any other.
//
// The kernel itself is shaders/common_interp_kernel.wgsl, one copy since
// plans/2D-backport.md B3-3 and now with one includer; the pool
// parent's tau, block grid, origin and fetch are
// shaders/common_interp_parent_pool.wgsl (read that one for why this side
// needs no periodic wrap and no second slot lookup). What is left here is the
// binding layout, which must match interpPoolParentBGL in the pages
// one-to-one.

// @include "common_lattice.wgsl"
// @include "common_interp.wgsl"
// @include "common_fpack.wgsl"

struct LevelParams {
  nbx: u32,
  nby: u32,
  parentTau: f32,
  dxL: f32, // this level's own grid spacing in L0-buffer-space units --
            // unused here (no epsilon/geometry math in this shader), but
            // real (not padding): Milestone 8's amr_force1_pool.wgsl reads
            // it from this SAME per-level buffer, hence the shared name.
}

@group(0) @binding(0) var<uniform>             levelParams    : LevelParams;
@group(0) @binding(1) var<storage, read>       f_parent_pool  : array<u32>;
@group(0) @binding(2) var<storage, read_write> f_pool         : array<u32>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
@group(0) @binding(4) var<storage, read>       newlyActivated : array<u32>;
@group(0) @binding(5) var<storage, read>       blockSlot      : array<i32>;
// The two genuinely new per-slot fields the quadtree parent hop needs --
// see common_interp_parent_pool.wgsl's header.
//
// UNREAD ON THE ROOT-PARENT PIPELINE (PARENT_GHOST = 0), where both are
// derived from the child's own block index instead. They stay DECLARED because
// WGSL module scope has no conditional bindings and this entry file serves both
// pipelines; main-amr.js binds a sentinel buffer there rather than a plausible
// one, so a read that should not happen cannot return a number that looks like
// data (plans/2D-backport.md B6-9c).
@group(0) @binding(6) var<storage, read>       parentSlot     : array<i32>;
@group(0) @binding(7) var<storage, read>       quadrant       : array<u32>;

// @include "common_interp_parent_pool.wgsl"
// @include "common_interp_kernel.wgsl"
