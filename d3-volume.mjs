// THE VOLUME STACK AND THE CAMERA: the host statement of what the raymarcher
// does. plans/3D.md M6.2-M6.4. Shared by main-3d.js (which builds the
// volumes and drives the render) and by tools/validate-d3-raymarch.js (which
// has to predict, INDEPENDENTLY of the picture, which pixels a refined box
// can possibly have changed).
//
// WHY A HOST MODULE AT ALL, when the shader is the thing that runs. Because
// M6.4b's gate is "the stacked render agrees with the single-volume render
// everywhere only L0 exists, and differs where a refined box covers", and a
// gate that asks the page where its boxes project to is asking the accused
// to testify. The ray/box arithmetic here is a second route to the same
// answer -- the same relationship d3-amr.mjs's cellAtLevel has to
// common_d3_tree_sample.wgsl, and tools/test-d3-volume.js mutation-checks it
// the same way.
//
// THE FRAME IS THE SAMPLER'S FRAME: L0 cell units with cell CENTRES at
// integers (common_d3_tree_sample.wgsl's header derives it). So cell i spans
// [i - 1/2, i + 1/2) and a box of `ext` cells starting at cell index `lo`
// spans the CONTINUOUS interval [lo - 1/2, lo + ext - 1/2). Every box below
// carries both: `lo`/`ext` in cell-index units, because that is what the
// resample kernel's origin is, and `c0`/`c1` continuous, because that is
// what a ray is intersected against. Conflating them is a half-cell shift,
// which at ?vol=1 over the domain is exactly the error the M6.1 alignment
// case exists to catch.

// A LEVEL'S VOLUME. `lo`/`ext` are in L0 CELL-INDEX units.
//
// `mult` IS A MULTIPLE OF THE LEVEL'S OWN RESOLUTION, not a count of voxels
// per L0 cell -- level m's grid already has 2^m cells per L0 cell, so mult = 1
// means every level's volume matches its own grid EXACTLY and nothing is
// interpolated or thrown away. Stating it the other way ("voxels per L0
// cell") makes one number mean a different thing at each level and was the
// first version of this; it survived until a gate asked for 4 and got
// sixteen.
//
// EVERY LEVEL AT FULL RESOLUTION OVER ITS OWN BOX -- which is the whole
// measurement M6.4 rests on: 49 MB for the flagship case's three levels
// against 2.25 GB for one uniform volume at the finest resolution, because
// the refined region is 0.44% of the domain.
export function levelVolume({ lo, ext, level, mult, bytesPerVoxel = 8 }) {
  const s = mult * 2 ** level;
  const res = ext.map((e) => Math.max(1, Math.round(e * s)));
  const h = ext.map((e, k) => e / res[k]);
  return {
    level, mult, lo: lo.slice(), ext: ext.slice(), res, h,
    // The continuous span, for ray intersection. See the header.
    c0: lo.map((v) => v - 0.5),
    c1: lo.map((v, k) => v + ext[k] - 0.5),
    voxels: res[0] * res[1] * res[2],
    bytes: res[0] * res[1] * res[2] * bytesPerVoxel,
  };
}

// THE STACK. `levelBoxes[m]` is level m's refined bounding box in L0 cell
// units, or null where that level has no tiles; index 0 is ignored and the
// dense domain is used instead, because level 0 is present everywhere by
// definition.
//
// A level whose box is missing is DROPPED rather than given an empty volume:
// a zero-extent texture is not creatable, and "this level has no tiles" is
// exactly the case where the coarser volume already holds the right answer
// (the resample writes sampleTree, which falls back).
export function volumeStack({ dims, levelBoxes = [], mult = 1, levels = 1, bytesPerVoxel = 8 }) {
  const out = [levelVolume({ lo: [0, 0, 0], ext: dims.slice(), level: 0, mult, bytesPerVoxel })];
  for (let m = 1; m < levels; m++) {
    const b = levelBoxes[m];
    if (!b) continue;
    out.push(levelVolume({
      lo: b.lo.slice(), ext: b.hi.map((h, k) => h - b.lo[k]), level: m, mult, bytesPerVoxel }));
  }
  return out;
}

// The BOX-EFFICIENCY ratio M6.4c reports: how much a solid bounding box costs
// over the set it bounds. A body-fitted shell is a shell and the box is
// solid, so this is never 1 -- the flagship case measured 1.35x. Above ~4x
// the boxes have stopped paying and true per-ray descent is the answer
// (plans/3D.md M6.4's "when this breaks"), which is why this is a number
// rather than a warning.
// `ext` WINS OVER hi - lo + 1 WHEN IT IS THERE, because a periodic span can
// WRAP and then `hi` is below `lo`. A caller on a windowed run hands the
// extent explicitly; hi - lo + 1 on a wrapped span is negative, and before
// 2026-09-12 this quietly reported the seam crossing as geometry.
export function boxRatio(bbox, inUse) {
  if (!bbox || !inUse) return null;
  const ext = bbox.ext
    || [0, 1, 2].map((k) => bbox.hi[k] - bbox.lo[k] + 1);
  if (ext.some((e) => e <= 0)) return null;
  const v = ext[0] * ext[1] * ext[2];
  return { boxBlocks: v, ratio: v / inUse };
}

// --- the camera ------------------------------------------------------------
//
// ONE ORBIT, stated once. Azimuth turns about `up`, elevation lifts out of
// the plane orthogonal to it, and `dist` is in units of the domain's largest
// extent so a camera default means the same thing on a 64^3 box and on the
// 192x128x128 flagship.
//
// WHY `up` IS A PARAMETER AND NOT +z: the scenarios already declare which way
// is DOWN (main-3d.js's downTurn reads it for the slice view's quarter turn),
// and a volume view that rendered a falling card sideways would be the same
// papercut one dimension up. The solver has no opinion about down and must
// not acquire one -- this is the view's business, exactly as the slice
// rotation is.
export function orbitBasis(up) {
  // The two axes orthogonal to `up`, taken cyclically so the triple is
  // right-handed whichever axis `up` is. Only axis-aligned ups occur here
  // (a scenario's `down` is a lattice direction), so this is exact.
  const a = up.findIndex((v) => Math.abs(v) > 0.5);
  const s = up[a] > 0 ? 1 : -1;
  const e1 = [0, 0, 0], e2 = [0, 0, 0], u = [0, 0, 0];
  e1[(a + 1) % 3] = 1;
  e2[(a + 2) % 3] = s;
  u[a] = s;
  return { e1, e2, up: u };
}

export function orbitEye(target, { azim, elev, dist, up = [0, 0, 1], scale = 1 }) {
  const { e1, e2, up: u } = orbitBasis(up);
  const ca = Math.cos(azim), sa = Math.sin(azim), ce = Math.cos(elev), se = Math.sin(elev);
  const r = dist * scale;
  return target.map((t, k) => t + r * (ce * (ca * e1[k] + sa * e2[k]) + se * u[k]));
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const n = Math.hypot(...a) || 1; return a.map((v) => v / n); };

// The pinhole basis, mirroring d3_raymarch.wgsl's camBasis(). Right-handed
// with `fwd` toward the target, so a pixel at uv = (0.5, 0.5) looks straight
// down `fwd` and the image is not mirrored -- a reflection here would flip
// the sign of nothing (the fields are scalars) and simply lie about which
// side of the body the wake is on, which is worse for being invisible.
export function cameraBasis(eye, target, up = [0, 0, 1]) {
  const fwd = norm(sub(target, eye));
  const { up: u } = orbitBasis(up);
  let right = cross(fwd, u);
  if (Math.hypot(...right) < 1e-6) right = cross(fwd, [1, 0, 0]);
  right = norm(right);
  return { fwd, right, up: norm(cross(right, fwd)) };
}

// uv in [0,1]^2 with uv.y UP the screen (the quad's own convention in
// d3_raymarch.wgsl), aspect = width/height.
//
// TWO PROJECTIONS, and the second is an instrument. Perspective fans the rays
// from a point; ORTHOGRAPHIC makes them parallel and slides the ORIGIN across
// the image plane, with `halfHeight` in L0 CELLS. Under it the image is an
// affine map of the lattice -- a pixel is a known number of cells and a plane
// in the flow is a straight line at a computable row -- which is what makes a
// cut measurable in cells instead of estimated through foreshortening.
export function cameraRay(uv, { eye, fwd, right, up, tanHalfFov, aspect, ortho = false, halfHeight = 0 }) {
  const sx = (uv[0] * 2 - 1) * aspect;
  const sy = uv[1] * 2 - 1;
  if (ortho) {
    return { o: eye.map((e, k) => e + sx * halfHeight * right[k] + sy * halfHeight * up[k]),
             d: fwd.slice() };
  }
  const px = sx * tanHalfFov, py = sy * tanHalfFov;
  return { o: eye.slice(), d: norm(fwd.map((f, k) => f + px * right[k] + py * up[k])) };
}

// CELLS PER PIXEL under the orthographic camera. One line, and it is the whole
// reason that mode exists: an image measurement converts to lattice units with
// no camera model at all.
export function orthoScale(halfHeight, imageHeight) { return 2 * halfHeight / imageHeight; }

// Slab method. Returns null when the ray misses, else the [t0, t1] interval
// with t0 clamped to 0 so an eye INSIDE the box still marches.
export function rayBox(o, d, c0, c1) {
  let t0 = -Infinity, t1 = Infinity;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < 1e-12) {
      if (o[k] < c0[k] || o[k] > c1[k]) return null;
      continue;
    }
    const a = (c0[k] - o[k]) / d[k], b = (c1[k] - o[k]) / d[k];
    t0 = Math.max(t0, Math.min(a, b));
    t1 = Math.min(t1, Math.max(a, b));
  }
  if (t1 < Math.max(t0, 0)) return null;
  return [Math.max(t0, 0), t1];
}

// Normalized position within a box, PERIODICALLY -- d3_raymarch.wgsl's
// boxLocal. `dims` omitted means no wrap, which is the right reading for a
// box that does not straddle the seam and is what the M6.4b gate asserts of
// its own configs before relying on this.
export function boxLocal(p, v, dims = null) {
  return p.map((x, k) => {
    let q = x - v.c0[k];
    if (dims) q -= dims[k] * Math.floor(q / dims[k]);
    return q / v.ext[k];
  });
}

// INNERMOST BOX WINS (M6.4b). The stack is ordered coarsest-first, so the
// last box containing the point is the finest one that does -- no descent, no
// pointer chasing, and the holes in a level's box are not wrong because the
// resample wrote sampleTree into them, which falls back to the coarser value.
export function innermostAt(p, stack, dims = null) {
  let best = -1;
  for (let i = 0; i < stack.length; i++) {
    const tc = boxLocal(p, stack[i], dims);
    if (tc.every((v) => v >= 0 && v <= 1)) best = i;
  }
  return best;
}

// Does this ray touch any box past the first, i.e. CAN the stacked render
// differ from the single-volume one along it? The M6.4b gate's independent
// route: a pixel whose ray misses every refined box must be bit-identical
// between the two builds, and one that hits must be allowed to differ.
//
// Deliberately UNWRAPPED, and the caller checks that: a straddling box would
// need every periodic image tested, and a gate that quietly widened its
// "allowed to differ" set until the test passed would be no gate at all.
export function rayTouchesRefined(o, d, stack, tMax = Infinity) {
  for (let i = 1; i < stack.length; i++) {
    const hit = rayBox(o, d, stack[i].c0, stack[i].c1);
    if (hit && hit[0] <= tMax) return true;
  }
  return false;
}

// Is every refined box strictly inside the domain, i.e. does no box straddle
// the periodic seam? rayTouchesRefined is only a valid predicate when this
// holds.
export function stackUnwrapped(stack, dims) {
  return stack.slice(1).every((v) => v.c0.every((c, k) => c >= -0.5 && v.c1[k] <= dims[k] - 0.5));
}

// THE STENCIL STRIDE (M6.4d). How far apart, IN VOXELS, the gradient pass's
// central difference should reach for a voxel whose data came from level
// `srcLevel`, given the volume's own voxel size `h` in L0 cell units.
//
// WHY IT IS NOT ALWAYS 1. A level's volume is a dense grid over a BOUNDING
// BOX and the refined set inside it is not box-shaped, so voxels inside the
// box but outside the set are filled from a COARSER level -- 48-63% of the
// structured voxels on the flagship card, measured by
// tools/probe-d3-volume-crunch.js. The resample samples the tree NEAREST, so
// every voxel landing in one source cell holds a BIT-IDENTICAL value.
// Differencing those at the VOLUME's spacing gives exactly zero across the
// plateau and the whole source-cell jump across one voxel at its edge, then
// divides by a step h_src/h too small -- a staircase differentiated into
// speckle. That is the stipple along the vortex tubes, and it is why the
// artifact vanishes at ?volstack=0 (no refined box, no replicated voxels)
// while every other knob only scales it.
//
// So the stencil spans ONE SOURCE CELL rather than one voxel. This is
// common_d3_tree_sample.wgsl's own standing rule -- "any finite difference
// taken from it must use the sampler's own h" -- applied one pass later,
// where until 2026-09-12 the h was a pipeline constant and the same for
// every voxel in the texture.
//
// A NO-OP WHERE THE VOLUME ALREADY HAS THE DATA: srcLevel equal to the
// volume's level gives 2^-src / h = mult >= 1, and mult = 1 (the default and
// every gate's setting) gives exactly 1. So the refined region renders
// bit-identically and only the replicated population moves, which is what
// `?volh=0` exists to demonstrate rather than assert.
//
// floor(x + 0.5) rather than Math.round, because the shader mirrors this and
// WGSL's `round` is half-to-EVEN where JS's is half-UP. The ratio is an exact
// integer for every box whose extent lands on the level's own grid, so the
// two agree anyway -- but a rule stated twice should not differ on a case
// merely because nobody reaches it.
export function stencilStride(h, srcLevel) {
  const hs = 2 ** -srcLevel;
  return h.map((hk) => Math.max(1, Math.floor(hs / hk + 0.5)));
}
