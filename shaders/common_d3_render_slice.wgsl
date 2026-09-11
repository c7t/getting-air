// Axis-aligned slice view of the macroscopic field (plans/3D.md M1's
// visual, sec 1.3 rendering option 1), sampling the FINEST level that
// covers each pixel (M6.0b). Fragment only; the entry files list every
// include -- and they are per-Q, like every other kernel here, because
// common_d3_tree_sample.wgsl brings common_d3_pool.wgsl's tile layout with
// it and that in turn references the lattice. None of the lattice is
// reachable from this shader today; naming the WRONG one anyway is the kind
// of latent mismatch this project keeps finding the expensive way.
//
// Deliberately the cheapest thing that works, and deliberately first: it
// reuses shaders/common_vortcolor.wgsl verbatim, it is what you actually
// want when debugging a seam or a wall, and it keeps the volume renderer
// (option 2, resample-and-raymarch) out of M1's critical path where it
// would otherwise eat the schedule -- ranked risk #6.
//
// Reads the same [rho, ux, uy, uz] buffer common_d3_step.wgsl writes, so
// there is no separate render-side field to keep in step.

// The signed ramp and its tone curve come from common_vortcolor.wgsl,
// shared with both 2D views so the sign convention and the colors mean the
// same thing everywhere. Its VORT_SCALE is overridden to 1.0 by the page,
// though, and that is the important part: the 2D default of 40 is
// calibrated for a shedding wake at |omega| ~ 0.05, and every field on this
// page is an order of magnitude gentler than that -- a decaying Beltrami
// flow has |omega| = k|u| ~ 0.008. Rendered on the 2D calibration it is a
// nearly-black smudge. So the page normalizes each value by a
// SCENARIO-DERIVED reference (see main-3d.js's vRef/uRef) and passes a
// dimensionless ratio here, which is what the tone curve should have been
// eating all along.

// Binding 0 (the dense L0 `mac`) and bindings 2-9 (every pool level's `mac`
// and `blockSlot`) belong to common_d3_tree_sample.wgsl, which is where this
// view's sampling now comes from -- see its header. This file owns only
// binding 1.
@group(0) @binding(1) var<uniform>       rp  : RParams;

// axis: 0/1/2 = the slice's NORMAL is x/y/z. slice: index along it.
// mode: 0 = speed, 1 = out-of-plane vorticity, 2 = the velocity component
// along the slice normal (which is what shows a duct's u_x profile on a
// y-z slice -- the in-plane view of a unidirectional flow is blank).
struct RParams {
  axis  : u32,
  slice : u32,
  mode  : u32,
  _pad0 : u32,
  uScale : f32,      // velocity reference for modes 0 and 2
  vScale : f32,      // vorticity reference for mode 1
  _pad1 : f32,
  _pad2 : f32,
}

// NX/NY/NZ come from common_d3_pool.wgsl, which the tree sampler needs
// anyway for poolCell and the tile layout. One declaration, not two.

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

const quad = array<vec2<f32>,6>(
  vec2(-1f,-1f), vec2( 1f,-1f), vec2(-1f, 1f),
  vec2(-1f, 1f), vec2( 1f,-1f), vec2( 1f, 1f)
);

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out: VSOut;
  out.pos = vec4(quad[vi], 0f, 1f);
  out.uv  = quad[vi] * 0.5f + 0.5f;
  return out;
}

// In-plane axis extents for the current slice normal. The two in-plane
// axes are taken in cyclic order (x-normal -> (y,z), y-normal -> (z,x),
// z-normal -> (x,y)) so the out-of-plane vorticity component below is the
// right-handed one and its sign means the same thing on all three views.
fn planeDims() -> vec2<u32> {
  if (rp.axis == 0u) { return vec2<u32>(NY, NZ); }
  if (rp.axis == 1u) { return vec2<u32>(NZ, NX); }
  return vec2<u32>(NX, NY);
}

// The 3D point at in-plane CONTINUOUS coordinates (a, b) on the current
// slice, in L0 cell units with cell centres at integers -- the frame
// common_d3_tree_sample.wgsl documents. Continuous rather than an integer
// cell index because that is the whole point of M6: the pixel asks where it
// is and the tree answers at whatever resolution it has there, instead of
// the view quantising to L0 before it ever looks.
fn planePoint(a: f32, b: f32) -> vec3<f32> {
  let sl = f32(rp.slice);
  if (rp.axis == 0u) { return vec3<f32>(sl, a, b); }   // (y, z) plane at x = slice
  if (rp.axis == 1u) { return vec3<f32>(b, sl, a); }   // (z, x) plane at y = slice
  return vec3<f32>(a, b, sl);                          // (x, y) plane at z = slice
}

// A sampled velocity rotated into (in-plane u, in-plane v, out-of-plane w),
// so the caller never has to re-derive the axis permutation. The cyclic
// order is what makes the out-of-plane vorticity right-handed on all three
// views -- see planeDims.
fn rotateUVW(u: vec3<f32>) -> vec3<f32> {
  if (rp.axis == 0u) { return vec3<f32>(u.y, u.z, u.x); }
  if (rp.axis == 1u) { return vec3<f32>(u.z, u.x, u.y); }
  return u;
}

// Velocity at in-plane continuous coordinates, from the finest level that
// covers the point.
fn sampleUVW(a: f32, b: f32) -> vec3<f32> {
  return rotateUVW(sampleTree(planePoint(a, b)).v.yzw);
}

// Dark-blue -> cyan -> yellow ramp for unsigned magnitudes. Distinct from
// the signed blue/red vorticity ramp on purpose: the two modes must not be
// confusable at a glance, since "is that a strong positive vortex or a
// fast region" is exactly the question a slice view is being asked.
fn magColor(t: f32) -> vec3<f32> {
  let c = clamp(t, 0f, 1f);
  let lo = mix(vec3(0.05, 0.05, 0.12), vec3(0.1, 0.65, 0.75), smoothstep(0f, 0.55f, c));
  return mix(lo, vec3(1.0, 0.92, 0.35), smoothstep(0.55f, 1f, c));
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let d = planeDims();
  // uv -> L0 cell units. Cell i spans [i - 1/2, i + 1/2), so the -0.5 is
  // what makes floor(p + 1/2) reproduce the integer index this used to
  // compute directly, at every zoom level.
  let pa = uv.x * f32(d.x) - 0.5f;
  let pb = (1.0 - uv.y) * f32(d.y) - 0.5f;

  let t = sampleTree(planePoint(pa, pb));
  let s = rotateUVW(t.v.yzw);

  if (rp.mode == 1u) {
    // Out-of-plane vorticity: d(v)/d(a) - d(u)/d(b) with (a,b) the in-plane
    // axes, by central differences.
    //
    // THE STEP IS THE LOCAL CELL SIZE, not a fixed one, and that is forced
    // rather than chosen. A fixed step of one L0 cell would throw away
    // exactly the resolution this view was changed to show; a fixed step of
    // one FINEST cell would sample the same coarse cell twice outside the
    // refined region and paint it flat zero -- which is not a small error,
    // it is the whole coarse field reading as irrotational. So the
    // difference is taken over the cell size of the level that answered,
    // which is a true local gradient estimate on either side of a seam.
    //
    // Across a seam one arm can land on a coarser cell, and the staircase
    // that produces is not hidden: this is the view for looking at seams.
    let h = t.h;
    let omega = (sampleUVW(pa + h, pb).y - sampleUVW(pa - h, pb).y)
              - (sampleUVW(pa, pb + h).x - sampleUVW(pa, pb - h).x);
    return vec4(vorticityColor(omega / (2f * h) / max(rp.vScale, 1e-12f)), 1.0);
  }
  if (rp.mode == 2u) {
    // Signed velocity along the slice normal, on the same ramp so the sign
    // reads the same way (blue negative, red positive). This is the mode
    // that shows a unidirectional flow: a duct's u_x is invisible in both
    // of the others, since an x-normal slice has no in-plane velocity and
    // therefore no out-of-plane vorticity either.
    return vec4(vorticityColor(s.z / max(rp.uScale, 1e-9f)), 1.0);
  }
  return vec4(magColor(length(s) / max(rp.uScale, 1e-9f)), 1.0);
}
