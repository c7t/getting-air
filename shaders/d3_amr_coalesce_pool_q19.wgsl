// D3Q19 interface coalesce with a POOL parent (level >= 2). Same physics as
// d3_amr_coalesce_q19.wgsl; the parent-address fragment differs and the
// dispatch walks parent tiles instead of a dense grid.
// shader-loader.mjs splices one level, so every fragment is listed here.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_pool.wgsl"
// @include "common_d3_parent_pool.wgsl"
// @include "common_d3_amr_coalesce.wgsl"
// @include "common_d3_amr_coalesce_main_pool.wgsl"
