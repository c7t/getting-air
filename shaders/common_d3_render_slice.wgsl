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

// THE BODY, for the moving window's view offset (M8.3). A binding this view
// owns, appended AFTER the sampler's shared ones exactly as the resample
// pass appends its output texture -- the shared description stays one
// description, and the probe and the resample do not acquire a binding
// neither of them reads.
//
// READ DIRECTLY, NOT PASSED IN. A host-written offset is as stale as the
// last body readback, and the body advances between them: the picture then
// slides forward and snaps back once a refresh lands, which is a visible
// pumping rather than a subtle lag.
@group(0) @binding(12) var<storage, read> body : BodyState3D;

// THE VIEW'S OFFSET IS CONTINUOUS WHERE THE SOLVER'S IS INTEGER, and the
// difference is deliberate. winOffset() floors, because the SPONGE BAND has
// to be aligned to the cell grid -- a band edge sliding through a cell would
// flicker. A viewer has no such constraint, and inheriting the floor would
// leave the body sawing back and forth across one cell as its sub-cell part
// ramps from 0 to 1: a full cell of jitter, at the exact frequency of the
// body's own cell crossings, on a body that is physically standing still in
// this frame.
fn winViewOffset() -> vec3<f32> {
  let c = vec3<f32>(body.cx, body.cy, body.cz);
  let a = vec3<f32>(WIN_AX, WIN_AY, WIN_AZ);
  let d = winDims();
  return vec3<f32>(
    select(0f, c.x - a.x, d.x > 0f),
    select(0f, c.y - a.y, d.y > 0f),
    select(0f, c.z - a.z, d.z > 0f));
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
//
// THE SLICE IS IN WINDOW COORDINATES and the offset converts it to the
// buffer position the sampler reads (M8.3). Without a window the offset is
// zero and this is what it always was. With one, the view pans with the
// body instead of watching it wrap across the screen -- and `slice` keeps
// meaning "this many cells into the window", so a slice through the body
// stays through the body as it travels. sampleTree wraps periodically
// already, so an offset point past the far face needs no clamping here.
fn planePoint(a: f32, b: f32) -> vec3<f32> {
  let sl = f32(rp.slice);
  let off = winViewOffset();
  if (rp.axis == 0u) { return vec3<f32>(sl, a, b) + off; }   // (y, z) plane at x = slice
  if (rp.axis == 1u) { return vec3<f32>(b, sl, a) + off; }   // (z, x) plane at y = slice
  return vec3<f32>(a, b, sl) + off;                          // (x, y) plane at z = slice
}

// A sampled velocity rotated into (in-plane u, in-plane v, out-of-plane w),
// so the caller never has to re-derive the axis permutation. The cyclic
// order is what makes the out-of-plane vorticity right-handed on all three
// views -- see planeDims. Applied to a POSITION as well as a velocity below:
// the same permutation takes a cell centre into (a, b, normal).
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

// The same sample, plus WHERE IT CAME FROM in the in-plane frame. A nearest
// sampler answers at the centre of the cell containing the query, not at the
// query -- so a finite difference has to divide by the separation of the two
// CENTRES, not by the step it asked for. See TreeSample.c.
struct PlaneSample { uvw : vec3<f32>, ab : vec3<f32> }
fn samplePlane(a: f32, b: f32) -> PlaneSample {
  let t = sampleTree(planePoint(a, b));
  return PlaneSample(rotateUVW(t.v.yzw), rotateUVW(t.c));
}

// One centred derivative, over the TRUE separation of the samples returned.
//
// THE GUARD IS FOR A TREE MID-REBUILD, not for the steady case. Under 2:1
// balance the arms cannot land in the same cell: the coarsest neighbour of a
// level-m cell is level m-1, whose size is exactly 2h, so a 2h separation
// always crosses a boundary. This sampler is documented as safe to call
// while the manager is halfway through a topology change, though, and a zero
// separation there would be a divide by zero rather than a stale pixel.
fn dUdX(plus: PlaneSample, minus: PlaneSample, comp: u32, axis: u32, h: f32) -> f32 {
  let dx = plus.ab[axis] - minus.ab[axis];
  if (abs(dx) < 1e-6f * max(h, 1e-6f)) { return 0f; }
  return (plus.uvw[comp] - minus.uvw[comp]) / dx;
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
    // THE STEP ASKED FOR IS THE LOCAL CELL SIZE, and that much is forced
    // rather than chosen. A fixed step of one L0 cell would throw away
    // exactly the resolution this view was changed to show; a fixed step of
    // one FINEST cell would sample the same coarse cell twice outside the
    // refined region and paint it flat zero -- which is not a small error,
    // it is the whole coarse field reading as irrotational.
    //
    // BUT THE STEP DELIVERED IS NOT THE STEP ASKED FOR, and dividing by the
    // latter was a real bug (fixed 2026-09-11). sampleTree is NEAREST: it
    // answers at the centre of the cell containing the query. Across a seam
    // the `+h` arm can land in a coarser cell whose centre is up to half that
    // cell away, so the true separation of the two samples is anywhere from
    // about 0.5 to 1.5 times 2h -- and dividing by 2h regardless mis-scales
    // omega by an O(1) factor EXACTLY at tile boundaries. That is not the
    // resolution staircase an earlier version of this comment waved at; it
    // is a wrong value, and it showed up as artifacts along the seams in the
    // one view that exists to look at them.
    //
    // The tell was that `speed` -- a point sample with no derivative in it --
    // is smooth across the very seams where `vorticity` is not. A field that
    // cannot express the artifact not expressing it is what separates a
    // rendering bug from a solver one.
    let h = t.h;
    let ap = samplePlane(pa + h, pb); let am = samplePlane(pa - h, pb);
    let bp = samplePlane(pa, pb + h); let bm = samplePlane(pa, pb - h);
    let omega = dUdX(ap, am, 1u, 0u, h) - dUdX(bp, bm, 0u, 1u, h);
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
