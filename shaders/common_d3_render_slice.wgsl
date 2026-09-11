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
  // QUARTER TURNS OF THE PICTURE, CLOCKWISE. 0 is the historical view.
  //
  // IT IS AN IMAGE ROTATION AND NOTHING ELSE, which is the whole reason it
  // is safe. Every field this view draws -- |u|, the OUT-OF-PLANE vorticity,
  // the velocity along the slice NORMAL -- is invariant under a rotation
  // within the slice plane, so only the uv -> (a, b) mapping below changes
  // and not one line of the sampling or the differencing. A rotation that
  // had to re-sign omega would be a reflection, and there is no reflection
  // here: `rot` is a rotation subgroup on purpose.
  //
  // WHY IT EXISTS: a body falling along +x rendered on a z-slice moves
  // ACROSS the window, because the in-plane axes are (x, y) and x is the
  // horizontal one. The scenario says which way is DOWN and main-3d.js picks
  // the turn that puts it down the screen -- see downTurn(). Nothing about
  // the solver prefers an axis, and now nothing about the view forces the
  // reader to remember which one the scenario chose.
  rot   : u32,
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

// The same sample, plus WHERE IT CAME FROM and WHICH LEVEL ANSWERED, both in
// the in-plane frame. A nearest sampler answers at the centre of the cell
// containing the query, not at the query -- so a finite difference has to
// divide by the separation of the two CENTRES, not by the step it asked for.
// See TreeSample.c. `ab` is (a, b, normal): rotateUVW takes a POSITION into
// the plane frame exactly as it takes a velocity, so `.z` is the offset along
// the slice normal, which is the half of this that dPlane below is about.
struct PlaneSample { uvw : vec3<f32>, ab : vec3<f32>, level : u32 }
fn samplePlaneAt(a: f32, b: f32, mMax: u32) -> PlaneSample {
  let t = sampleTreeAtMost(planePoint(a, b), mMax);
  return PlaneSample(rotateUVW(t.v.yzw), rotateUVW(t.c), t.level);
}

// ONE CENTRED DERIVATIVE, FROM A LEVEL-CONSISTENT PAIR.
//
// THE TWO ARMS MUST COME FROM THE SAME LEVEL, and that is a correctness
// requirement rather than a tidiness one. sampleTree is NEAREST: it answers at
// a cell CENTRE, which differs from the query on ALL THREE axes, by up to half
// that cell. So a pair straddling a seam is displaced not only along the axis
// being differenced -- which the divisor below already handles -- but along the
// other in-plane axis and along the SLICE NORMAL too, and those displacements
// do not cancel. The difference then carries the transverse gradients,
// amplified by (transverse offset) / (in-plane step):
//
//     est(du/db) = du/db + (da/db_step) du/da + (dn/db_step) du/dn
//
// with dn up to a quarter of a coarse cell and db_step around one, so the
// contamination is O(1) in the seam row. MEASURED, on a Beltrami box where the
// exact vorticity is k*u: the seam row read 9.94% of max|omega| high, against
// 10.00% predicted from the slice-normal term alone. Away from the seam the
// error is 0.30% rms. That was the whole of the residual artifact after the
// divisor fix earlier the same day -- which cut the same row from 25.6% to
// 9.94% by fixing the denominator, and could not touch this because it is in
// the numerator.
//
// THE FIX IS TO RE-ASK, NOT TO CORRECT. Two samples at the SAME level share
// the transverse offset exactly, so it cancels in the difference and nothing
// has to be estimated. When the first pair disagrees, both arms are re-asked
// capped at the coarser of the two levels, at that level's own step. Each pass
// strictly lowers the cap, and level 0 is present everywhere, so the loop ends;
// in practice it runs once for a seam pixel and not at all for any other, which
// is why the cost is only on the ~7% of pixels that straddle a seam.
//
// The seam row is then a COARSE estimate rather than a wrong fine one, which
// is the honest answer: the finest level available to BOTH sides of a
// difference is the finest level that difference can be taken at.
//
// THE ZERO GUARD IS FOR A TREE MID-REBUILD, not for the steady case. Under
// 2:1 balance a level-consistent pair is exactly 2h apart. This sampler is
// documented as safe to call while the manager is halfway through a topology
// change, though, and a zero separation there would be a divide by zero
// rather than a stale pixel.
fn dPlane(pa: f32, pb: f32, comp: u32, axis: u32, mStart: u32) -> f32 {
  var m = mStart;
  var h = 1f / f32(1u << m);
  var plus  = samplePlaneAt(pa + select(0f, h, axis == 0u), pb + select(0f, h, axis == 1u), m);
  var minus = samplePlaneAt(pa - select(0f, h, axis == 0u), pb - select(0f, h, axis == 1u), m);
  for (var it = 0u; it < SAMPLE_LEVELS && plus.level != minus.level; it++) {
    m = min(plus.level, minus.level);
    h = 1f / f32(1u << m);
    plus  = samplePlaneAt(pa + select(0f, h, axis == 0u), pb + select(0f, h, axis == 1u), m);
    minus = samplePlaneAt(pa - select(0f, h, axis == 0u), pb - select(0f, h, axis == 1u), m);
  }
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

// uv -> in-plane L0 cell units, through `rot` quarter turns CLOCKWISE.
//
// Cell i spans [i - 1/2, i + 1/2), so the -0.5 is what makes floor(p + 1/2)
// reproduce the integer index this used to compute directly, at every zoom
// level. uv.y = 1 is the TOP of the screen, so the unrotated form already
// draws +b DOWNWARD; rot = 1 hands that role to +a instead.
//
// Each case is the composition of the previous one with a quarter turn, and
// they are written out rather than derived from a matrix because there are
// exactly four of them and the closed forms are one line each.
fn uvToPlane(uv: vec2<f32>, d: vec2<u32>) -> vec2<f32> {
  let dx = f32(d.x); let dy = f32(d.y);
  if (rp.rot == 1u) { return vec2((1f - uv.y) * dx - 0.5f,   (1f - uv.x) * dy - 0.5f); }
  if (rp.rot == 2u) { return vec2((1f - uv.x) * dx - 0.5f,   uv.y * dy - 0.5f); }
  if (rp.rot == 3u) { return vec2(uv.y * dx - 0.5f,          uv.x * dy - 0.5f); }
  return vec2(uv.x * dx - 0.5f, (1f - uv.y) * dy - 0.5f);
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let d = planeDims();
  let p = uvToPlane(uv, d);
  let pa = p.x;
  let pb = p.y;

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
    // BUT THE STEP DELIVERED IS NOT THE STEP ASKED FOR, and the pair is not
    // even taken at one resolution unless it is made to be. sampleTree is
    // NEAREST -- it answers at the centre of the cell containing the query --
    // so across a seam the two arms differ both in SEPARATION (an O(1) error
    // in the divisor, fixed 2026-09-11) and in POSITION ALONG THE OTHER TWO
    // AXES, including the slice normal, which contaminates the numerator with
    // the transverse gradients and was worth another 10% of max|omega| in the
    // seam row. dPlane handles both: it re-asks the pair at a common level.
    //
    // The tell was that `speed` -- a point sample with no derivative in it --
    // is smooth across the very seams where `vorticity` is not. A field that
    // cannot express the artifact not expressing it is what separates a
    // rendering bug from a solver one.
    let omega = dPlane(pa, pb, 1u, 0u, t.level) - dPlane(pa, pb, 0u, 1u, t.level);
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
