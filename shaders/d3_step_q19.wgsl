// D3Q19 variant of the dense 3D fused step (plans/3D.md M1). The velocity
// set is chosen entirely by which lattice fragment is included here --
// see shaders/common_d3_step.wgsl for what the kernel does and why, and
// shaders/d3_step_q27.wgsl, which is this file with one line changed.

// @include "common_d3q19_lattice.wgsl"
// @include "common_d3_step.wgsl"
