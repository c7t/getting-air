// D3Q27 variant of the 3D force/torque kernel. See
// shaders/common_d3_force.wgsl. Every fragment is listed here, not inside
// the body: shader-loader.mjs splices exactly one level.

// @include "common_d3q27_lattice.wgsl"
// @include "common_d3_geometry.wgsl"
// @include "common_d3_reduce.wgsl"
// @include "common_d3_force.wgsl"
