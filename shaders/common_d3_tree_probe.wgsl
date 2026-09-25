// A HOST-DRIVEN PROBE of common_d3_tree_sample.wgsl. plans/3D.md M6.
// Fragment only; the entry files list every include.
//
// The sampler's only other consumer is a FRAGMENT shader, whose output is
// pixels -- and a picture is not something a check can score. This is the
// same function, over the same bindings, reached from a compute pass so that
// tools/validate-d3-invariants.js can ask it where a list of points landed
// and compare the answer against d3-amr.mjs's finestLevelAt.
//
// WHAT IS ACTUALLY BEING SCORED IS THE LEVEL. "Finest active level wins" is
// a statement about blockSlot alone, and the host can answer it from a
// readback with arithmetic that shares nothing with the shader's. Given the
// level, turning a point into a value is poolCell on an owning block, which
// tools/test-d3-amr.js already covers from both directions. The value is
// returned too, and the tool checks it is finite and that a hit really did
// come from the level it claims -- but the level is the claim.
@group(0) @binding(10) var<storage, read>       probeIn  : array<vec4<f32>>;
// Two entries per point: [2i] = (rho, ux, uy, uz), [2i+1] = (level, h, 0, 0).
@group(0) @binding(11) var<storage, read_write> probeOut : array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&probeIn)) { return; }
  let s = sampleTree(probeIn[i].xyz);
  probeOut[2u * i] = s.v;
  probeOut[2u * i + 1u] = vec4<f32>(f32(s.level), s.h, 0f, 0f);
}
