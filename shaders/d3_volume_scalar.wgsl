// THE SCALAR VOLUME: the fields a raymarcher can actually show, differenced
// off the resampled velocity volume. plans/3D.md M6.2.
//
// NOT PER-Q, and it has no includes: this kernel never touches a population,
// a tile or the lattice. It reads one 3D texture and writes another. Naming
// a velocity set here would be a lie of the kind this project keeps paying
// for elsewhere (see common_d3_render_slice.wgsl's header for the opposite
// case, where the lattice IS reachable and so is named).
//
// WHY A SEPARATE PASS AND NOT Q STRAIGHT FROM THE TREE. Q needs the velocity
// GRADIENT, so sampling it from the hierarchy costs SEVEN tree samples per
// voxel (centre plus +-h on three axes), each of which is a walk down the
// levels and four scattered planar loads. Resampling velocity once and
// differencing this dense, coherent volume is ~4x cheaper for one extra
// texture -- and the gradient is then evaluated once per VOXEL instead of
// once per RAY SAMPLE, which is the same argument one level further up.
//
// WHY rgba16float AND NOT r32float, which the M6 ballpark expected.
// Measured, and then reconsidered:
//
//   - r16float is NOT storage-writable in core WebGPU (rejected outright on
//     an RTX 4080 with no optional features), so it was never available.
//   - r32float IS storage-writable, but FILTERING it needs the optional
//     float32-filterable feature, and the raymarcher wants trilinear. An
//     optional feature is a dependency, and the mobile target may not have
//     it.
//   - rgba16float is storage-writable AND filterable in core.
//
// The objection to half precision is real and is answered by NORMALIZING
// rather than by widening: Q has units of 1/time^2 and here runs ~(U/D)^2 ~
// 1e-5, which is below binary16's smallest NORMAL (6.1e-5) and would land in
// the subnormals. Dividing by the same reference the criterion and the slice
// view already divide by (d3-criterion.mjs's qRef, main-3d.js's U_SCALE /
// V_SCALE) puts every channel at order 1, where fp16's 2^-11 relative step is
// far finer than anything a transfer function resolves. "Hand the shader a
// ratio, not a quantity" is this codebase's existing answer to this exact
// question.
//
// And having paid 8 B/voxel, the other three channels are free, so they are
// filled rather than padded: ONE PASS PRODUCES EVERY FIELD THE VIEW CAN
// SHOW, and switching between them is a uniform rather than a recompute.
//
//   .r   Q / qRef        rotation-dominated: vortex CORES, not shear layers
//   .g   |omega| / vRef  every shear layer, which is what 2D renders
//   .b   |u| / uRef      the point sample, no derivative in it -- the field
//                        that CANNOT express a differencing artifact, which
//                        is what separated a rendering bug from a solver one
//                        in M6.0b and is worth keeping reachable
//   .a   rho - 1         the acoustic field; also the first thing to look at
//                        when a run is about to blow up

@group(0) @binding(0) var velIn  : texture_3d<f32>;
@group(0) @binding(1) var sclOut : texture_storage_3d<rgba16float, write>;
// The level each voxel's data came from, written by common_d3_resample.wgsl.
// See VOL_HSRC. A 1x1x1 dummy on the L0 volume, which never reads it.
@group(0) @binding(2) var lvlIn  : texture_3d<u32>;

override VOL_NX : u32 = 1u;
override VOL_NY : u32 = 1u;
override VOL_NZ : u32 = 1u;
// The voxel size on each axis, in L0 cell units. The gradient comes out in
// L0 units at every level, which is what makes one threshold mean one thing
// across the stack -- the same level-independence argument
// common_vortcolor.wgsl makes for omega, and the reason a refined level
// legitimately reads HIGHER Q than the coarse grid over the same flow: the
// finer stencil resolves a core the coarse difference was averaging flat.
override VOL_HX : f32 = 1f;
override VOL_HY : f32 = 1f;
override VOL_HZ : f32 = 1f;
// 1 where the volume covers the WHOLE periodic domain, so a difference at the
// face reaches round to the other side; 0 for a sub-box, where there is no
// neighbour and the stencil has to shorten instead.
override VOL_WRAP : u32 = 0u;

// M6.4d. THE STENCIL REACHES ONE SOURCE CELL, NOT ONE VOXEL, and until
// 2026-09-12 it always reached one voxel.
//
// A level's volume is a dense grid over a BOUNDING BOX while the refined set
// inside it is a shell, so most of the box is filled from a COARSER level --
// 48-63% of the structured voxels on the flagship card, measured. The
// resample samples the tree NEAREST, so all the voxels inside one source cell
// are BIT-IDENTICAL. Differencing that plateau at VOL_H gives exactly zero
// through the interior and the entire source-cell jump across a single voxel
// at its edge, divided by a step 2^(level-src) too small: the replication
// staircase differentiated into isolated speckle, which is the stipple a
// ?levels=3 card render shows along its vortex tubes.
//
// THE FIX IS common_d3_tree_sample.wgsl's OWN RULE -- "any finite difference
// taken from it must use the sampler's own h" -- which this pass could not
// obey while h was one pipeline constant for the whole texture. Now the
// resample hands over the source level per voxel and the stride follows.
//
// IT IS A NO-OP WHERE THE VOLUME ALREADY HAS THE DATA (src == level gives
// stride 1 at ?vol=1), so the refined region renders bit-identically and only
// the replicated population moves. `?volh=0` sets this to 0 and restores the
// old fixed stride, which is how that claim is measured rather than asserted.
override VOL_HSRC : u32 = 0u;
// This volume's own level, so the shader can tell "the data is mine" from
// "the data is a coarser level's, replicated".
override VOL_LEVEL : u32 = 0u;

// The normalizations. All three default to 1, which makes this kernel report
// raw lattice quantities -- correct, and useless at fp16, which is why the
// host always sets them.
override Q_REF  : f32 = 1f;
override OM_REF : f32 = 1f;
override U_REF  : f32 = 1f;

// A neighbour index on one axis, and THE SEPARATION IT ACTUALLY GIVES.
//
// Returning both is the M6.0b lesson in its cheapest form: a stencil that
// clamps at a face and then divides by the step it ASKED for is wrong by 2x
// on the whole boundary layer of the volume -- which for a refined box is
// precisely the seam region the view exists to look at. So the shortened
// stencil reports its own width and the caller divides by that.
struct Tap { ip : i32, im : i32, sep : f32 }
fn tap(i: i32, n: i32, h: f32, stride: i32) -> Tap {
  // Never past the volume: a stride wider than the texture would collapse
  // ip onto im and divide by zero. Clamping is right rather than refusing --
  // a box thinner than one source cell has no wider difference to take.
  let d = max(1, min(stride, n - 1));
  if (VOL_WRAP == 1u) { return Tap(((i + d) % n + n) % n, ((i - d) % n + n) % n, 2f * f32(d) * h); }
  let ip = min(i + d, n - 1);
  let im = max(i - d, 0);
  return Tap(ip, im, f32(ip - im) * h);
}

// d3-volume.mjs's `stencilStride`, in WGSL. floor(x + 0.5) and not `round`:
// WGSL rounds half-to-EVEN and JS's Math.round is half-UP, and one rule
// stated in two languages should not differ on a case merely because the box
// extents that reach it are rare.
fn strideFor(srcLevel: u32, h: f32) -> i32 {
  if (VOL_HSRC != 1u || srcLevel >= VOL_LEVEL) { return 1; }
  return max(1, i32(floor(exp2(-f32(srcLevel)) / h + 0.5f)));
}

fn uAt(x: i32, y: i32, z: i32) -> vec3<f32> {
  return textureLoad(velIn, vec3<i32>(x, y, z), 0).xyz;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= VOL_NX || gid.y >= VOL_NY || gid.z >= VOL_NZ) { return; }
  let v = vec3<i32>(gid);
  // The level THIS voxel was filled from -- not the volume's own level, which
  // is only what it was filled INTO. See VOL_HSRC.
  // An `if` and not `select`, which evaluates both arms: VOL_HSRC is an
  // override, so this folds away entirely and the L0 volume -- the big one --
  // pays no fetch at all for a companion texture it does not have.
  var src : u32 = VOL_LEVEL;
  if (VOL_HSRC == 1u) { src = textureLoad(lvlIn, v, 0).r; }
  let tx = tap(v.x, i32(VOL_NX), VOL_HX, strideFor(src, VOL_HX));
  let ty = tap(v.y, i32(VOL_NY), VOL_HY, strideFor(src, VOL_HY));
  let tz = tap(v.z, i32(VOL_NZ), VOL_HZ, strideFor(src, VOL_HZ));

  // J[j] = du/dx_j, so J[j][i] is du_i/dx_j -- the same indexing
  // common_d3_criterion.wgsl's qAt uses, deliberately, because these two are
  // the same quantity computed on two grids and a transposition between them
  // would show up as a criterion that refines where the picture shows
  // nothing.
  var J : array<vec3<f32>, 3>;
  J[0] = (uAt(tx.ip, v.y, v.z) - uAt(tx.im, v.y, v.z)) / tx.sep;
  J[1] = (uAt(v.x, ty.ip, v.z) - uAt(v.x, ty.im, v.z)) / ty.sep;
  J[2] = (uAt(v.x, v.y, tz.ip) - uAt(v.x, v.y, tz.im)) / tz.sep;

  // Q = 1/2(|Omega|^2 - |S|^2), as the difference of the two Frobenius norms
  // rather than the equivalent -1/2 J_ij J_ji: the compact form is one
  // transposition away from being silently wrong, and the two norms are
  // separately meaningful. d3-criterion.mjs says the same thing in JS and
  // tools/test-d3-criterion.js scores it against the case that matters --
  // PURE SHEAR IS EXACTLY ZERO, which is the whole reason this is Q and not
  // |omega|.
  var s2 = 0f; var om2 = 0f;
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      let a = J[j][i];
      let b = J[i][j];
      let S = 0.5f * (a + b);
      let O = 0.5f * (a - b);
      s2 += S * S;
      om2 += O * O;
    }
  }
  let q = 0.5f * (om2 - s2);
  // The curl, from the same tensor. Free here, and the field 2D renders.
  let omega = vec3<f32>(J[1][2] - J[2][1], J[2][0] - J[0][2], J[0][1] - J[1][0]);
  let c = textureLoad(velIn, v, 0);

  textureStore(sclOut, v, vec4<f32>(
    q / Q_REF,
    length(omega) / OM_REF,
    length(c.xyz) / U_REF,
    c.w - 1f));
}
