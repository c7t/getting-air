// D3Q19 coarse->fine interpolation with a POOL parent (level >= 2). Same
// transfer as d3_amr_interp_q19.wgsl -- trilinear plus the post-collision
// Dupuis-Chopard fneq rescale -- with only the parent-address fragment
// swapped. It seeds a newly-created level at startup and, from M5.5, fills
// a tile the manager has just allocated.
// shader-loader.mjs splices one level, so every fragment is listed here.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_pool.wgsl"
// @include "common_d3_parent_pool.wgsl"
// @include "common_d3_amr_interp.wgsl"
