// D3Q19 variant of the Q-criterion refinement flag. See
// shaders/common_d3_criterion.wgsl. Every fragment is listed here, not inside
// the body: shader-loader.mjs splices exactly one level.
//
// The lattice is not reachable from this shader -- it reads `mac`, not `f` --
// but it is named anyway, per the convention common_d3_render_slice.wgsl's
// header states: naming the WRONG one is the kind of latent mismatch this
// project keeps finding the expensive way, and a per-Q entry file costs
// nothing.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_criterion.wgsl"
