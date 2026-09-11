// THE MOVING WINDOW, shader side. plans/3D.md M8.3. Fragment only; the ENTRY
// files list it, and it must appear before common_d3_geometry.wgsl and
// common_d3_sponge.wgsl, which both call into it.
//
// Host counterpart and the full argument: d3-window.mjs. The short version,
// because it is the part that is easy to get backwards while reading a
// kernel:
//
//   THE FLUID NEVER MOVES. The buffer is periodic and holds the lab frame.
//   The BODY advances through it and wraps, as it physically does through
//   the fluid. What follows the body is the SPONGE -- the band of cells
//   held at the far-field state -- and it follows by having each cell
//   convert its buffer position into a WINDOW coordinate before measuring
//   its distance to a window face. Nothing else in the solver is aware of
//   any of this: streaming, the tiles, the rings and the coarse/fine
//   interface are all written against buffer indices and stay that way.
//
// WIN_N* IS BOTH THE SIZE AND THE SWITCH: the domain size on a windowed
// axis, 0 on an axis with no window, where every function here is the
// identity and folds out at pipeline-creation time. Default 0 on all three,
// so every scenario that predates this is bit-identical rather than merely
// unaffected in practice.

override WIN_NX : f32 = 0.0f;
override WIN_NY : f32 = 0.0f;
override WIN_NZ : f32 = 0.0f;

// Where the body sits in the window, in L0 cells. Its INITIAL position, so
// the offset is exactly 0 at step 0 -- see d3-window.mjs.
override WIN_AX : f32 = 0.0f;
override WIN_AY : f32 = 0.0f;
override WIN_AZ : f32 = 0.0f;

fn winDims() -> vec3<f32> { return vec3<f32>(WIN_NX, WIN_NY, WIN_NZ); }

// Nearest-image delta on one axis; the identity where there is no window.
// `round` here ties to the EVEN integer where d3-window.mjs's Math.round ties
// upward, so the two disagree by a sign exactly at |d| = n/2 -- see that
// file's note for why the two equidistant images are interchangeable there.
fn winWrap1(d: f32, n: f32) -> f32 {
  if (n <= 0f) { return d; }
  return d - n * round(d / n);
}

fn winWrapDelta(d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(winWrap1(d.x, WIN_NX), winWrap1(d.y, WIN_NY), winWrap1(d.z, WIN_NZ));
}

// How far the window has travelled, in WHOLE cells, from the body's centre.
// Integer so the sponge band stays aligned to the cell grid rather than
// having its edge slide through a cell; the sub-cell remainder is the body's
// own position within its anchor cell, which is where it belongs.
fn winOffset(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    select(0f, floor(c.x) - WIN_AX, WIN_NX > 0f),
    select(0f, floor(c.y) - WIN_AY, WIN_NY > 0f),
    select(0f, floor(c.z) - WIN_AZ, WIN_NZ > 0f));
}

fn winCoord1(p: f32, off: f32, n: f32) -> f32 {
  if (n <= 0f) { return p; }
  let w = p - off;
  return w - n * floor(w / n);
}

// Buffer position -> window coordinate, in [0, n). The identity on an axis
// with no window, where the buffer IS the window.
fn winCoord(p: vec3<f32>, off: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    winCoord1(p.x, off.x, WIN_NX),
    winCoord1(p.y, off.y, WIN_NY),
    winCoord1(p.z, off.z, WIN_NZ));
}

// Keep a body position inside the buffer -- the counterpart of winCoord.
// One wraps a fluid cell into the window; this wraps the body into the
// buffer, and together they are what lets the body cross the periodic seam
// without anything else in the solver noticing.
fn winWrapPos1(c: f32, n: f32) -> f32 {
  if (n <= 0f) { return c; }
  return c - n * floor(c / n);
}

fn winWrapPos(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    winWrapPos1(c.x, WIN_NX), winWrapPos1(c.y, WIN_NY), winWrapPos1(c.z, WIN_NZ));
}
