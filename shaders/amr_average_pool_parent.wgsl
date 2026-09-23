// Fine -> parent-pool averaging (restriction) for every L(m) -> L(m-1) hop
// with m>=2: the parent is another POOL tile, not the dense grid.
// Milestone 7 (plans/AMR-multilevel.md).
//
// The kernel itself is shaders/common_average.wgsl, shared with
// the deleted dense-parent entry file since plans/2D-backport.md B3-2; the pool parent's tau
// and destination addressing are shaders/common_avg_parent_pool.wgsl. What is
// left here is the binding layout (which must match avgPoolBGL in the pages
// one-to-one).

// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"

struct LevelParams {
  nbx: u32,        // READ ON THE ROOT-PARENT PATH ONLY (PARENT_GHOST = 0),
  nby: u32,        // where parentSlot/quadrant are derived from this level's
                   // own block index instead of read -- see
                   // common_avg_parent_pool.wgsl. Unused on every ringed
                   // parent, where the destination is parentSlot+quadrant and
                   // not a block-index lookup. Shared verbatim with the
                   // interp/step1 pool-parent shaders' uniform, not a
                   // near-duplicate. `nby` genuinely is unused here.
  parentTau: f32,
  dxL: f32,        // also unused here -- see amr_interp_pool_parent.wgsl's
                   // comment; real field, not padding (amr_force1_pool.wgsl
                   // reads it from this same per-level buffer).
}

@group(0) @binding(0) var<uniform>             levelParams   : LevelParams;
@group(0) @binding(1) var<storage, read>       f_pool        : array<u32>;
@group(0) @binding(2) var<storage, read_write> f_parent_pool : array<u32>;
@group(0) @binding(3) var<storage, read>       slotToBlock   : array<i32>;
// UNREAD ON THE ROOT-PARENT PIPELINE (PARENT_GHOST = 0), where both are
// derived from the child's own block index instead. They stay DECLARED because
// WGSL module scope has no conditional bindings and this entry file serves both
// pipelines; main-amr.js binds a sentinel buffer there rather than a plausible
// one, so a read that should not happen cannot return a number that looks like
// data (plans/2D-backport.md B6-9c). Same treatment as the interp entry's.
@group(0) @binding(4) var<storage, read>       parentSlot    : array<i32>;
@group(0) @binding(5) var<storage, read>       quadrant      : array<u32>;

// @include "common_avg_parent_pool.wgsl"
// @include "common_average.wgsl"
