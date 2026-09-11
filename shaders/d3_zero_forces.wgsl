// Clears the force/torque accumulator between macro-steps.
//
// Separate from d3_physics.wgsl (which drains it with atomicExchange)
// because a PINNED body returns early there, and because a scenario may run
// the force kernel for measurement without running the integrator at all.
// One dispatch, 8 lanes, negligible -- and it means no path can leave a
// stale partial sum for the next step to inherit.

@group(0) @binding(0) var<storage, read_write> forces : array<atomic<i32>, 8>;

@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  atomicStore(&forces[gid.x], 0);
}
