// RESAMPLE: the AMR hierarchy onto a dense 3D texture. plans/3D.md M6.1.
// Fragment only; the entry files list every include.
//
// The GPU sibling of tools/lib/field-reconstruct.js, and there is almost
// nothing to it -- common_d3_tree_sample.wgsl already answers "what is the
// field here", so this is that question asked once per voxel over a box.
// What the file actually decides is the BOX, and that decision came from
// measurement rather than taste (plans/3D.md M6's ballpark):
//
//   - the refined region of the flagship case is 24^3 L0 cells out of
//     192x128x128, i.e. 0.44% of the domain;
//   - a uniform volume over the whole domain at the FINEST level is 2.25 GB
//     and ~52 ms/frame, which is the dense run AMR exists to replace;
//   - the same data, over the region that actually HAS that resolution, is
//     10 MB and ~0.23 ms.
//
// So the box is a parameter: ORIGIN and EXTENT in L0 cell units, and a
// RESOLUTION in voxels. The host picks the full domain at L0 by default and
// the refined bounding box at full depth on request, and neither is baked
// in here.
//
// rgba16float, NOT r16float. Measured on an RTX 4080 with no optional
// features: a 3D r16float texture with STORAGE_BINDING is REJECTED by core
// WebGPU. rgba16float is storage-writable AND filterable, which is what the
// raymarcher needs from it.
//
// WHY VELOCITY AND NOT THE SCALAR DIRECTLY. Q-criterion needs the velocity
// gradient, so computing it here would cost SEVEN tree samples per voxel
// (centre and +-h on three axes) against ONE. Resampling velocity first and
// differencing this dense, coherent volume afterwards is ~4x cheaper for one
// extra volume -- the same argument that puts the gradient in a pass rather
// than in the ray loop. M6.2 is that pass.

@group(0) @binding(10) var volOut : texture_storage_3d<rgba16float, write>;

// THE ORIGIN IS A UNIFORM AND THE SHAPE IS AN OVERRIDE, and the split is the
// whole of M6.4a's dynamic story. A refined box FOLLOWS the body: a
// translating sphere holds `inUse` constant while every tile changes hands,
// so the set's SIZE is steady and its POSITION is not (M4.2b-iii). Baking the
// origin in as an override would mean rebuilding the pipeline, the texture
// and the bind group every time the box moved -- for a viewer, every few
// frames. So the extent and the resolution are fixed at creation, where they
// have to be (a texture has a size), and the origin slides through a uniform.
//
// A box that OUTGROWS its allocated extent is said out loud in #status rather
// than silently clipped: a view that quietly shows part of a refined region
// is a plausible picture of a hierarchy nobody chose, which is the failure
// mode M5.0's refusal exists for.
@group(0) @binding(11) var<uniform> volOrigin : vec4<f32>;

// THE LEVEL EACH VOXEL'S DATA ACTUALLY CAME FROM (M6.4d). `sampleTree`
// already returns it and this pass used to throw it away; the gradient pass
// (d3_volume_scalar.wgsl) needs it, because a voxel filled from a coarser
// level is one of a replicated block and must be differenced over the SOURCE
// cell rather than over one voxel. d3-volume.mjs's `stencilStride` is the
// host statement of that rule.
//
// rgba8uint AND NOT r8uint: r8 is not a storage-writable format in core
// WebGPU (the same wall d3_volume_scalar.wgsl's header hit with r16float), so
// 4 bytes per voxel is the floor. It is paid ONLY by the refined levels --
// the L0 volume gets a 1x1x1 dummy and WRITE_LEVEL = 0, because its own
// spacing is already the coarsest in the tree and its stride is 1 by
// construction. On the flagship that is the 12 MB volume, not the 50 MB one.
@group(0) @binding(12) var lvlOut : texture_storage_3d<rgba8uint, write>;

// 0 for the L0 volume, whose companion is a 1x1x1 dummy. A binding cannot be
// optional, so the WRITE is.
override WRITE_LEVEL : u32 = 0u;

// The box SHAPE, in L0 cell units with cell centres at integers -- the frame
// common_d3_tree_sample.wgsl documents.
override VOL_NX : u32 = 1u;
override VOL_NY : u32 = 1u;
override VOL_NZ : u32 = 1u;
// L0 cells per voxel on each axis. 1 means the volume matches L0; 0.25 means
// four voxels per L0 cell, i.e. level-2 resolution.
override VOL_HX : f32 = 1f;
override VOL_HY : f32 = 1f;
override VOL_HZ : f32 = 1f;

// Voxel v's CENTRE, in L0 cell units. The half-voxel offset is the same
// convention the rest of the solver uses for cells, so a volume at VOL_H = 1
// aligned to the origin lands exactly on L0 cell centres and the resample is
// then a copy rather than a resample -- which is what makes the 1x case
// checkable against `mac` directly.
fn volCentre(v: vec3<u32>) -> vec3<f32> {
  return volOrigin.xyz
       + (vec3<f32>(v) + vec3<f32>(0.5f)) * vec3<f32>(VOL_HX, VOL_HY, VOL_HZ)
       - vec3<f32>(0.5f);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= VOL_NX || gid.y >= VOL_NY || gid.z >= VOL_NZ) { return; }
  let s = sampleTree(volCentre(gid));
  // (ux, uy, uz, rho). Velocity first because M6.2 differences the xyz and
  // never touches w, so the gradient pass reads three contiguous halves of
  // each texel rather than striding past a density it does not want.
  textureStore(volOut, vec3<i32>(gid), vec4<f32>(s.v.yzw, s.v.x));
  if (WRITE_LEVEL == 1u) { textureStore(lvlOut, vec3<i32>(gid), vec4<u32>(s.level, 0u, 0u, 0u)); }
}
