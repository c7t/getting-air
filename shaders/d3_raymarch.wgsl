// THE OCTREE RAYMARCHER: one dense volume PER LEVEL, innermost box wins.
// plans/3D.md M6.3 + M6.4b.
//
// WHAT MAKES THIS NOT A RESEARCH PROJECT. sec 1.3 says "do not attempt to
// raymarch the quadtree/octree pool directly -- it is accurate and it is a
// research project", and that stays true of the POOL: per-ray descent through
// tiles, rings and free-list slots, with no coherence and no filtering. This
// is the other thing, and the two had been conflated. A level's refined set
// is ONE COMPACT REGION (geometry-forced refinement guarantees it -- the body
// is one body), so its BOUNDING BOX is small: measured on the flagship case,
// 24^3 L0 cells out of 192x128x128, i.e. 0.44% of the domain. Three dense
// volumes over three such boxes are 49 MB against 2.25 GB for one uniform
// volume at the finest resolution -- and a ray sample is then 2-3 box tests
// with the LAST hit winning, no descent, no pointer chasing, no stack.
//
// THE HOLES ARE NOT WRONG. A level's box is solid and its refined set is a
// shell, so the box contains cells that level does not have (measured 1.35x
// on the flagship). The resample pass writes sampleTree, which FALLS BACK to
// the coarsest level present -- so a hole holds the correct coarser value and
// sampling the innermost covering box is right everywhere, not just where
// that level exists.
//
// WHEN IT BREAKS, said before it does: bounding boxes pay because the refined
// set is one region. A Q-criterion set following a shed wake is NOT -- it
// breaks into scattered patches and a union of boxes degenerates toward the
// full domain. debugPoolState reports box-union / set volume per level
// (M6.4c) so the trigger is a number rather than a judgement call; above ~4x
// the boxes have stopped paying and true traversal is the answer.
//
// @include "common_d3_window.wgsl"
// @include "common_d3_geometry.wgsl"

struct RayParams {
  // azimuth, elevation (radians), distance (L0 cells), tan(fov/2)
  cam   : vec4<f32>,
  // aspect (w/h), step multiplier (voxels per ray step), iso, gain
  look  : vec4<f32>,
  // field index, opacity scale, level count, max ray steps
  ctl   : vec4<f32>,
  // the orbit target when not following the body, and w = follow flag
  tgt   : vec4<f32>,
  // Per level: the CONTINUOUS box (see d3-volume.mjs's header -- cell i spans
  // [i-1/2, i+1/2), so a box of `ext` cells from index `lo` spans
  // [lo-1/2, lo+ext-1/2)), and the extent with the voxel size in w.
  // boxLo[i].w is the entry's LEVEL, which is not its index: volumeStack drops
  // a level with no tiles, so the stack can be (0, 2). VOL_FALLBACK needs it
  // to ask "did this voxel come from MY level".
  boxLo : array<vec4<f32>, 4>,
  boxExt: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> rm : RayParams;
@group(0) @binding(1) var volSamp : sampler;
// FIXED IN NUMBER, like every other per-level binding here, because WebGPU
// has no array of buffers or of differently-sized textures. Levels the run
// does not have are bound to a 1x1x1 dummy and rm.ctl.z folds them out.
@group(0) @binding(2) var vol0 : texture_3d<f32>;
@group(0) @binding(3) var vol1 : texture_3d<f32>;
@group(0) @binding(4) var vol2 : texture_3d<f32>;
@group(0) @binding(5) var vol3 : texture_3d<f32>;
@group(0) @binding(6) var<storage, read> body : BodyState3D;
// M6.4f. The source level per voxel, the companion common_d3_resample.wgsl
// writes. Only levels 1..3 have one -- level 0's data is never a coarser
// level's replicated onto a finer grid, because there is no coarser level.
// uint, so textureLoad and integer coordinates: a filtered level number is
// not a level number.
@group(0) @binding(7) var lvl1 : texture_3d<u32>;
@group(0) @binding(8) var lvl2 : texture_3d<u32>;
@group(0) @binding(9) var lvl3 : texture_3d<u32>;

override NX : u32 = 1u;
override NY : u32 = 1u;
override NZ : u32 = 1u;
// Which world axis is UP for the camera, and its sign. The scenario declares
// which way is DOWN (d3-scenarios.mjs's `down`, already read by the slice
// view's quarter turn) and the host negates it. The SOLVER has no opinion
// about down and must not acquire one: this is the view's business, exactly
// as the slice rotation is.
override UP_AXIS : u32 = 2u;
override UP_SIGN : f32 = 1f;
override HAS_BODY : u32 = 0u;
// M6.4f. 1 = a box is only used where its voxel came from its own level;
// 0 = innermost box wins unconditionally, which is the pre-M6.4f behaviour and
// the leg the A/B measures against.
override VOL_FALLBACK : u32 = 0u;
// 0 = perspective (the default, and the one that reads as a picture),
// 1 = ORTHOGRAPHIC. See camRay for why the second one exists.
override PROJ : u32 = 0u;
// Express the march interval in WINDOW coordinates (1, the default) or in the
// buffer's own (0). See fs_main for what this is and is not worth.
override WIN_BOX : u32 = 1u;

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

fn axisVec(a: u32, s: f32) -> vec3<f32> {
  return vec3<f32>(select(0f, s, a == 0u), select(0f, s, a == 1u), select(0f, s, a == 2u));
}

// The orbit frame: `up`, and the two axes orthogonal to it taken cyclically
// so the triple is right-handed whichever axis up is. d3-volume.mjs's
// orbitBasis is the same three lines in JS, and tools/test-d3-volume.js
// scores the pair against each other -- a camera is not checkable by looking
// at the picture it produces, which is the whole difficulty with a viewer.
struct Orbit { e1 : vec3<f32>, e2 : vec3<f32>, up : vec3<f32> }
fn orbitAxes() -> Orbit {
  return Orbit(axisVec((UP_AXIS + 1u) % 3u, 1f),
               axisVec((UP_AXIS + 2u) % 3u, UP_SIGN),
               axisVec(UP_AXIS, UP_SIGN));
}

// WHERE THE CAMERA LOOKS, AND WHY IT IS COMPUTED HERE RATHER THAN HANDED IN.
// A body-following camera needs the body's position, and a HOST-WRITTEN one
// is as stale as the last readback: the picture then slides forward and snaps
// back every time a refresh lands, which is a visible pumping rather than a
// subtle lag. The slice view reads the body buffer directly for exactly this
// reason (M8.3) -- same binding, same argument.
fn camTarget() -> vec3<f32> {
  if (rm.tgt.w > 0.5f && HAS_BODY == 1u) {
    return vec3<f32>(body.cx, body.cy, body.cz);
  }
  return rm.tgt.xyz;
}

struct Cam { o : vec3<f32>, d : vec3<f32> }
fn camRay(uv: vec2<f32>) -> Cam {
  let ax = orbitAxes();
  let t = camTarget();
  let ca = cos(rm.cam.x); let sa = sin(rm.cam.x);
  let ce = cos(rm.cam.y); let se = sin(rm.cam.y);
  let eye = t + rm.cam.z * (ce * (ca * ax.e1 + sa * ax.e2) + se * ax.up);
  let fwd = normalize(t - eye);
  var right = cross(fwd, ax.up);
  if (length(right) < 1e-6f) { right = cross(fwd, vec3<f32>(1f, 0f, 0f)); }
  right = normalize(right);
  let up = normalize(cross(right, fwd));
  // PERSPECTIVE puts the eye at a point and fans the rays; ORTHOGRAPHIC makes
  // them parallel and slides the ORIGIN across the image plane instead.
  //
  // THE ORTHOGRAPHIC MODE IS AN INSTRUMENT, and it earns its place by making a
  // class of measurement possible rather than by looking better. Under it the
  // image plane is an AFFINE MAP OF THE LATTICE: `rm.cam.w` is the half-height
  // in L0 CELLS, so a pixel is a known number of cells, a plane in world space
  // is a straight line at a computable row, and "where is that cut, in cells"
  // stops being a question about foreshortening. Every mis-measurement in
  // M6.5a's write-up -- an A/B at a step where the cut fell outside the frame,
  // an image statistic censored by the frame edge -- is one a parallel
  // projection with a known scale would have made impossible.
  //
  // A sphere's outline is then a circle rather than a conic, too, which is
  // what lets a host predict a silhouette by closed form instead of by
  // ray/sphere per pixel.
  if (PROJ == 1u) {
    let ox = (uv.x * 2f - 1f) * rm.look.x * rm.cam.w;
    let oy = (uv.y * 2f - 1f) * rm.cam.w;
    return Cam(eye + ox * right + oy * up, fwd);
  }
  let px = (uv.x * 2f - 1f) * rm.look.x * rm.cam.w;
  let py = (uv.y * 2f - 1f) * rm.cam.w;
  return Cam(eye, normalize(fwd + px * right + py * up));
}

// Slab method, with t0 clamped to 0 so an eye INSIDE the box still marches.
// Returns (t0, t1) with t1 < t0 on a miss.
fn rayBox(o: vec3<f32>, d: vec3<f32>, c0: vec3<f32>, c1: vec3<f32>) -> vec2<f32> {
  let inv = 1f / d;                       // +-inf on an axis-parallel ray
  let a = (c0 - o) * inv;
  let b = (c1 - o) * inv;
  let lo = min(a, b);
  let hi = max(a, b);
  return vec2<f32>(max(max(lo.x, lo.y), max(lo.z, 0f)), min(hi.x, min(hi.y, hi.z)));
}

// Normalized position within box i, PERIODICALLY. The half-cell in `boxLo` is
// already continuous (see RayParams), so the box map is one subtraction and
// one division -- and it is the SAME map the resample kernel's volCentre
// inverts, which is what makes the M6.1 gate (the volume scored against the
// sampler at the same physical points) also a check on this line.
//
// THE WRAP IS NOT DEFENSIVE. With a moving window the body travels through a
// periodic buffer and its refined shell travels with it, so the shell
// STRADDLES THE SEAM twice per lap -- and the bounding box of a straddling
// set, taken naively, is the whole domain. The host takes the smallest
// periodic span instead (main-3d.js's refinedBoxL0), which can leave an
// origin outside [0, N); wrapping here is what makes that box mean the cells
// it actually covers. It is also a no-op for every box that does not
// straddle, since a wrapped coordinate can only re-enter a box that already
// spans nearly the whole axis.
fn boxLocal(p: vec3<f32>, i: u32) -> vec3<f32> {
  let n = vec3<f32>(f32(NX), f32(NY), f32(NZ));
  var q = p - rm.boxLo[i].xyz;
  q = q - n * floor(q / n);
  return q / rm.boxExt[i].xyz;
}

// M6.4f. Did entry i's voxel at `tc` actually COME from level i, or is it a
// coarser level's value replicated onto i's finer grid? The resample records
// the answer per voxel; this reads it back with integer coordinates, because
// a filtered level number is not a level number.
//
// `>=` and not `==`: the level-m volume resamples with the same unrestricted
// sampleTree as everything else, so inside a level-(m+1) refined region its
// voxels legitimately hold level-(m+1) data -- undersampled onto m's grid, but
// its own level's or finer, which is what "not replicated" means here.
fn srcAtLeast(i: u32, tc: vec3<f32>) -> bool {
  let lv = u32(rm.boxLo[i].w);
  var d = vec3<u32>(1u);
  if (i == 1u) { d = textureDimensions(lvl1); }
  else if (i == 2u) { d = textureDimensions(lvl2); }
  else if (i == 3u) { d = textureDimensions(lvl3); }
  else { return true; }
  let f = vec3<f32>(d);
  let c = vec3<i32>(clamp(tc * f, vec3<f32>(0f), f - vec3<f32>(1f)));
  var src = 0u;
  if (i == 1u) { src = textureLoad(lvl1, c, 0).r; }
  else if (i == 2u) { src = textureLoad(lvl2, c, 0).r; }
  else { src = textureLoad(lvl3, c, 0).r; }
  return src >= lv;
}

// INNERMOST WINS -- BUT ONLY WHERE THE INNERMOST BOX ACTUALLY HAS THE DATA
// (M6.4f, `?volfallback=0` to A/B).
//
// A level's volume is a dense grid over a BOUNDING BOX and its refined set is
// a shell, so most of the box is filled from a coarser level, replicated onto
// the finer grid -- 58% of the structured voxels on the flagship card, and
// more again at ?volMargin=4. Taking the innermost box unconditionally hands
// the ray that replication: a plateau it then samples at the FINE voxel size,
// so it pays fine steps and fine interpolation for data that has neither.
// M6.4d fixed what that did to the GRADIENT; this is the other half, which is
// what the picture is made of.
//
// FALLING BACK IS ALSO CHEAPER, and that is not a coincidence: `dt` is the
// chosen box's voxel size, so a ray crossing a replicated region now steps at
// the resolution the data actually has instead of 2^m times finer.
//
// DESCENDING, so the first box that both contains p and owns its data wins and
// the rest are never touched. That bounds the extra fetches by the number of
// boxes actually entered -- typically one, and none at all outside them.
struct Hit { lv : u32, tc : vec3<f32> }
fn levelAt(p: vec3<f32>) -> Hit {
  let n = u32(rm.ctl.z);
  for (var k = n; k > 1u; k--) {
    let i = k - 1u;
    let tc = boxLocal(p, i);
    if (all(tc <= vec3<f32>(1f))) {
      if (VOL_FALLBACK != 1u || srcAtLeast(i, tc)) { return Hit(i, tc); }
    }
  }
  return Hit(0u, boxLocal(p, 0u));
}

// THE SAMPLER CLAMPS AT A BOX'S EDGE, and that is the right choice of the two
// available rather than a free one. Trilinear at the last half-voxel of a
// refined box has no neighbour inside the box, so it repeats the edge voxel
// where the physically right value is the COARSER level's -- a half-fine-voxel
// smear along the box face, which is 1/8 of an L0 cell at level 2. The
// alternative, wrapping, would fetch the opposite face of the box: the whole
// domain away, and wrong by the field itself rather than by a half voxel.
// Blending the two levels across the face would remove it and is a different
// feature (and one the slice view's dPlane argues AGAINST doing casually --
// two samples from different levels do not sit at the same place).
fn sampleLevel(i: u32, tc: vec3<f32>) -> vec4<f32> {
  // textureSampleLevel, NOT textureSample: an explicit LOD has no implicit
  // derivatives, so it is legal in the non-uniform control flow this branch
  // necessarily is.
  if (i == 1u) { return textureSampleLevel(vol1, volSamp, tc, 0f); }
  if (i == 2u) { return textureSampleLevel(vol2, volSamp, tc, 0f); }
  if (i == 3u) { return textureSampleLevel(vol3, volSamp, tc, 0f); }
  return textureSampleLevel(vol0, volSamp, tc, 0f);
}

// Which channel of the scalar volume to show. d3_volume_scalar.wgsl fills all
// four in one pass, so this is a uniform rather than a recompute.
fn fieldOf(v: vec4<f32>) -> f32 {
  let f = u32(rm.ctl.x);
  if (f == 1u) { return v.y; }            // |omega| / vRef
  if (f == 2u) { return v.z; }            // |u| / uRef
  if (f == 3u) { return abs(v.w); }       // |rho - 1|
  return v.x;                             // Q / qRef
}

// THE TRANSFER FUNCTION IS THREE THINGS: the `iso` subtracted in fs_main, the
// tone curve below it, and this ramp. THE M1 LESSON APPLIES TO IT FIRST -- a
// transfer function not normalized to the actual field renders a correct
// simulation as a black screen -- and it cannot be normalized HERE, because
// this shader has no idea what scale the flow is at. So the SCALAR VOLUME
// carries the normalization instead (d3_volume_scalar.wgsl divides every
// channel by the same reference the criterion and the slice view use), and
// `iso` is then a dimensionless number whose default is the MEASURED one:
// d3-criterion.mjs's Q_THRESHOLD, 0.1 of the body's own shear scale, which
// flags zero blocks on a seeded initial field and 379-628 in a developed wake.
fn volColor(t: f32) -> vec3<f32> {
  let c = clamp(t, 0f, 1f);
  let lo = mix(vec3(0.10, 0.16, 0.55), vec3(0.10, 0.75, 0.85), smoothstep(0f, 0.45f, c));
  return mix(lo, vec3(1.0, 0.93, 0.45), smoothstep(0.45f, 1f, c));
}

// THE TONE CURVE, AND IT IS THE ONE common_vortcolor.wgsl ALREADY ARGUES FOR
// -- Reinhard, a/(1+a), monotonic and asymptotic to 1 but never reaching it.
// The same reason applies here and bites harder. A hard clip collapses
// everything above the knee to one colour AND to one opacity, so the front
// face of the strongest structure becomes opaque and the render degenerates
// into a smooth isosurface of the iso value: measured on a Re = 300 sphere,
// where the near-wall Q is over a hundred times the wake's, a clipped mapping
// showed the boundary layer as a featureless shell and the wake not at all.
// Compressing instead keeps both in one picture, which is the entire reason
// to raymarch rather than to draw one isosurface.
//
// VOL_GAMMA then shapes the low end, exactly as it does on the 2D pages:
// below 1 it lifts weak structure out of the background rather than leaving
// it in a haze. The default matches the slice view's 0.7 for the same fields.
override VOL_GAMMA : f32 = 0.7;
fn volTone(x: f32) -> f32 {
  let a = max(x, 0f);
  return pow(a / (1f + a), VOL_GAMMA);
}

// THE BODY, BY SPHERE TRACING ITS OWN SDF -- not by thresholding the volume.
// d3-body.mjs's shapes are TRUE signed distances (sphere and rounded box
// exact, spheroid by the meridional Newton), which is what makes tracing them
// valid rather than merely plausible, and it means the surface is drawn at
// full precision instead of at whatever resolution the volume happens to be.
// Returns the hit distance, or -1.
fn traceBody(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32) -> f32 {
  if (HAS_BODY != 1u) { return -1f; }
  var t = t0;
  for (var i = 0u; i < 128u; i++) {
    if (t > t1) { break; }
    let phi = get_phi3(o + d * t, body);
    if (phi < 0.05f) { return t; }
    t += max(phi, 0.05f);
  }
  return -1f;
}

fn bodyNormal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.15f;
  return normalize(vec3<f32>(
    get_phi3(p + vec3<f32>(e, 0f, 0f), body) - get_phi3(p - vec3<f32>(e, 0f, 0f), body),
    get_phi3(p + vec3<f32>(0f, e, 0f), body) - get_phi3(p - vec3<f32>(0f, e, 0f), body),
    get_phi3(p + vec3<f32>(0f, 0f, e), body) - get_phi3(p - vec3<f32>(0f, 0f, e), body)));
}

const BG = vec3<f32>(0.05, 0.05, 0.09);

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let ray = camRay(uv);
  // The march is bounded by the LEVEL-0 volume, which covers the whole
  // domain: every refined box is inside it by construction, so one interval
  // is enough and a ray that misses it has nothing to show.
  //
  // AND THE INTERVAL IS IN WINDOW COORDINATES, NOT BUFFER ONES (M8.3). THIS
  // IS NOT COSMETIC -- IT IS THE WHOLE PICTURE, PERIODICALLY.
  //
  // The buffer is periodic and the body WRAPS THROUGH IT, but the wake does
  // not: it trails 112 cells behind the plate whatever the plate's buffer
  // position is. So the moment the body crosses the seam, its entire wake is
  // on the far side of it -- and a march bounded by [0, N) stops dead at the
  // seam and shows almost nothing. As the body falls on, more of the wake
  // comes back inside [0, N) and the picture grows again, one wrap period at
  // a time.
  //
  // MEASURED IN CELLS, under ?proj=ortho with the camera down a lattice axis,
  // which makes the image an affine map of the grid and a cut plane a row that
  // converts to a coordinate: on the falling card at the wrap (buffer
  // x = 0.1, step 14600), the buffer box's lit region ENDS AT BUFFER
  // x = -0.3 +- 0.25 cells. The box face is at -0.5. The plane IS the face,
  // identified to within one pixel rather than by eye -- and the wake, which
  // the window box shows running 49.5 cells further, is simply not drawn.
  // In frame terms that is 8.22% of the picture lit against 38.45%.
  // `?winbox=0` restores the buffer box for the A/B.
  //
  // WHY IT IS EASY TO MIS-MEASURE, since it was, twice: away from the seam
  // the cut plane is far from the body and lands OUTSIDE the frame, so an A/B
  // at an arbitrary step reads as 34 pixels in 160,000 and looks like
  // nothing. The step to test is the one where the body's buffer position is
  // near 0 or N. An image statistic clipped by the frame edge -- "the lit
  // region reaches row 0" -- is likewise censored, not constant, and reading
  // it as constant is what made this look like a non-problem.
  //
  // THE SAMPLING WAS NEVER WRONG, and that is worth keeping straight because
  // it is what rules out the resample: boxLocal already wraps, and the L0
  // volume agrees with the tree sampler to 9.5e-4 at every step including
  // across a wrap. The field was always right; only the interval the ray was
  // allowed to walk was stated in the wrong frame.
  let woff = select(vec3<f32>(0f), winOffset(vec3<f32>(body.cx, body.cy, body.cz)), WIN_BOX == 1u);
  let c0 = rm.boxLo[0].xyz + woff;
  let span = rayBox(ray.o, ray.d, c0, c0 + rm.boxExt[0].xyz);
  if (span.y <= span.x) { return vec4(BG, 1f); }

  let tBody = traceBody(ray.o, ray.d, span.x, span.y);
  let tEnd = select(span.y, min(span.y, tBody), tBody >= 0f);

  let iso = rm.look.z;
  let gain = max(rm.look.w, 1e-6f);
  let opacity = rm.ctl.y;
  let maxSteps = u32(rm.ctl.w);

  var acc = vec4<f32>(0f);
  var t = span.x;
  for (var i = 0u; i < maxSteps; i++) {
    if (t >= tEnd || acc.w > 0.995f) { break; }
    let p = ray.o + ray.d * t;
    let hit = levelAt(p);
    let lv = hit.lv;
    // THE STEP IS THE INNERMOST BOX'S VOXEL SIZE, which is the payoff of the
    // stack and not an optimization bolted onto it: a ray takes coarse steps
    // through the coarse field and fine steps through the refined shell, so
    // resolving the near-wall sheet costs steps only where that sheet is. A
    // fixed finest step would march the whole domain at the finest rate --
    // the same "2.25 GB against 10 MB" ratio, paid in time instead of memory.
    let dt = max(rm.boxExt[lv].w * rm.look.y, 1e-3f);
    let s = fieldOf(sampleLevel(lv, hit.tc));
    let tone = volTone(max(s - iso, 0f) / gain);
    if (tone > 0f) {
      // Opacity per unit LENGTH, not per sample: with an adaptive step the
      // naive per-sample alpha would make the refined region darker or
      // brighter than the coarse one purely because it is sampled more often,
      // which is a rendering artifact that looks exactly like physics.
      let alpha = 1f - exp(-tone * opacity * dt);
      let col = volColor(tone);
      acc = vec4<f32>(acc.xyz + (1f - acc.w) * alpha * col, acc.w + (1f - acc.w) * alpha);
    }
    t += dt;
  }

  var rgb = acc.xyz + (1f - acc.w) * BG;
  if (tBody >= 0f && tBody <= span.y) {
    let n = bodyNormal(ray.o + ray.d * tBody);
    // Two lights, one from the camera and a dim fill from `up`, so a smooth
    // body is not a flat silhouette and its orientation is readable -- which
    // for a TUMBLING card is the whole thing being looked at.
    let lam = 0.35f + 0.5f * max(dot(n, -ray.d), 0f) + 0.15f * max(dot(n, orbitAxes().up), 0f);
    rgb = acc.xyz + (1f - acc.w) * vec3<f32>(0.55f, 0.56f, 0.60f) * lam;
  }
  return vec4(rgb, 1f);
}
