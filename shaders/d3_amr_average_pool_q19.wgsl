// D3Q19 fine->coarse restriction with a POOL parent (level >= 2). Same
// transfer as d3_amr_average_q19.wgsl -- moments plus the Dupuis-Chopard
// fneq rescale -- with only the parent-address and parent-mac fragments
// swapped. It is M5.5b's DRAIN: the sibling of the pool-parent interp that
// fills a newly-allocated tile.
// shader-loader.mjs splices one level, so every fragment is listed here.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_pool.wgsl"
// @include "common_d3_parent_pool.wgsl"
// @include "common_d3_parentmac_pool.wgsl"
// @include "common_d3_amr_average.wgsl"
