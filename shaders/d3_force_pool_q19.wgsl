// D3Q19 variant of the fine-level force/torque kernel. See
// shaders/common_d3_force_pool.wgsl. Every fragment is listed here, not
// inside the body: shader-loader.mjs splices exactly one level.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_geometry.wgsl"
// @include "common_d3_reduce.wgsl"
// @include "common_d3_pool.wgsl"
// @include "common_d3_force_pool.wgsl"
