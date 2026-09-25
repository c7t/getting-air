// COALESCE's entry point for the DENSE parent (level 1). One thread per L0
// cell, dispatched over the grid -- which is what this pass has always done;
// it is a separate fragment only so the pool-parent sibling can replace it
// without forking coalesceAt. See that function's header.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY || gid.z >= NZ) { return; }
  coalesceAt(vec3<u32>(gid.x, gid.y, gid.z));
}
