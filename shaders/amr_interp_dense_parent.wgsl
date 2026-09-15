// Coarse -> fine ghost-cell interpolation for the L0 -> L1 hop: the parent is
// the DENSE, cellIndex()-addressed L0 buffer. Milestone 4 (plans/AMR.md),
// renamed from amr_interp_c2f.wgsl by Milestone 6.
//
// The kernel itself is shaders/common_interp_kernel.wgsl, shared with
// amr_interp_pool_parent.wgsl since plans/2D-backport.md B3-3; the dense
// parent's tau, block grid, origin and fetch are
// shaders/common_interp_parent_dense.wgsl (read that one for the known
// WALL_Y/periodic-wrap gap). What is left here is the binding layout (which
// must match interpBGL in the pages one-to-one) and the two overrides only
// the dense side needs.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_interp.wgsl"
// @include "common_fpack.wgsl"

@group(0) @binding(0) var<storage, read>       state          : CardState;
@group(0) @binding(1) var<storage, read>       f_coarse       : array<u32>;
@group(0) @binding(2) var<storage, read_write> f_pool         : array<u32>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
// Milestone 4b: which slots were JUST assigned this refine/coarsen round --
// only read when GHOST_ONLY=0 (the one-time full-slot-fill pipeline), to
// avoid re-filling an already-active slot's evolved interior with a fresh
// (and by now stale) coarse interpolation. The steady-state GHOST_ONLY=1
// pipeline shares this bind group layout but never reads this binding.
@group(0) @binding(4) var<storage, read>       newlyActivated : array<u32>;
// Milestone 4c: coarse-block -> pool-slot map (inverse of slotToBlock), so a
// ghost cell can check whether its EDGE-adjacent neighbor block is also
// currently refined, and if so pull directly from that neighbor's fine
// interior instead of round-tripping through the coarse level. Read-write
// elsewhere (main.js owns writes); read-only here.
@group(0) @binding(5) var<storage, read>       blockSlot      : array<i32>;

override W : u32;   // coarse grid dims
override H : u32;

// @include "common_interp_parent_dense.wgsl"
// @include "common_interp_kernel.wgsl"
