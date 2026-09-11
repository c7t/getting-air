// D3Q19 pool moments: rho and u from f, into a level's own mac. Used at
// reset(), after the seed has filled f and before anything reads mac.
// See common_d3_moments.wgsl.
// shader-loader.mjs splices one level, so every fragment is listed here.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_pool.wgsl"
// @include "common_d3_moments.wgsl"
