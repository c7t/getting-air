// D3Q19 interface explode with a POOL parent (level >= 2). Same physics as
// d3_amr_explode_q19.wgsl; only the parent-address fragment differs.
// shader-loader.mjs splices one level, so every fragment is listed here.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_pool.wgsl"
// @include "common_d3_parent_pool.wgsl"
// @include "common_d3_amr_explode.wgsl"
