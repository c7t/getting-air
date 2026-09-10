// Axis-aligned slice view of the dense 3D macroscopic field (plans/3D.md
// M1's visual, sec 1.3 rendering option 1).
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
// @include "common_vortcolor.wgsl"

@group(0) @binding(0) var<storage, read> mac : array<f32>;
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

override NX : u32;
override NY : u32;
override NZ : u32;

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

// Velocity at in-plane integer coordinates (a, b), wrapped. Returns the
// three components rotated into (in-plane u, in-plane v, out-of-plane w)
// so the caller never has to re-derive the axis permutation.
fn sampleUVW(a: i32, b: i32) -> vec3<f32> {
  let d = planeDims();
  let ai = u32((a + i32(d.x) * 64) % i32(d.x));
  let bi = u32((b + i32(d.y) * 64) % i32(d.y));
  var cell : u32;
  var uvw : vec3<u32>;
  if (rp.axis == 0u) {
    cell = (bi * NY + ai) * NX + rp.slice;      // (y, z) plane at x = slice
    uvw = vec3<u32>(1u, 2u, 0u);
  } else if (rp.axis == 1u) {
    cell = (ai * NY + rp.slice) * NX + bi;      // (z, x) plane at y = slice
    uvw = vec3<u32>(2u, 0u, 1u);
  } else {
    cell = (rp.slice * NY + bi) * NX + ai;      // (x, y) plane at z = slice
    uvw = vec3<u32>(0u, 1u, 2u);
  }
  return vec3<f32>(
    mac[4u * cell + 1u + uvw.x],
    mac[4u * cell + 1u + uvw.y],
    mac[4u * cell + 1u + uvw.z],
  );
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
  let fa = uv.x * f32(d.x);
  let fb = (1.0 - uv.y) * f32(d.y);
  let ia = i32(fa); let ib = i32(fb);

  let s = sampleUVW(ia, ib);

  if (rp.mode == 1u) {
    // Out-of-plane vorticity: d(v)/d(a) - d(u)/d(b) with (a,b) the in-plane
    // axes. Central differences over one cell, same 0.5 factor as the 2D
    // dense view, so common_vortcolor's tone curve is calibrated the same.
    let omega = (sampleUVW(ia + 1, ib).y - sampleUVW(ia - 1, ib).y) * 0.5f
              - (sampleUVW(ia, ib + 1).x - sampleUVW(ia, ib - 1).x) * 0.5f;
    return vec4(vorticityColor(omega / max(rp.vScale, 1e-12f)), 1.0);
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
