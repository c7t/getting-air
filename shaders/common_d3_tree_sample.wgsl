// FINEST-ACTIVE-LEVEL-WINS SAMPLING of the macroscopic field, at a
// continuous point. plans/3D.md M6. Fragment only; the entry files list
// every include.
//
// THE ONE PLACE IN THIS CODEBASE THAT SPANS LEVELS. Every other kernel is
// written for a single level and takes its parent through
// common_d3_parent_{dense,pool}.wgsl -- that is what makes them all the same
// code at every depth. A viewer cannot do that: "show me the best answer
// available here" is a question about the whole tree at once, so this
// declares every level's bookkeeping and walks down from the deepest.
//
// WHY A VIEWER NEEDS IT AT ALL, given that the dense L0 `mac` already holds
// the finest solution. It does -- the coalesce chain republishes each
// level's macroscopic field into its PARENT's `mac` under the refined
// region, so L2 reaches L1 reaches L0 (measured: with ?refine=all at
// ?levels=3 the dense array is written entirely by two coalesce hops and
// still scores 5.59e-3 against the analytic Beltrami solution). What L0
// cannot carry is RESOLUTION: a refined region arrives correct and coarse,
// so the refinement is invisible in the one view built for looking at seams.
// This is the part that was missing, and it is the same sampler the resample
// pass needs, one dimension down.
//
// THE COORDINATE FRAME IS L0 CELL UNITS WITH CELL CENTRES AT INTEGERS, which
// is what the rest of the solver means by a coarse coordinate:
// fineToCoarseUnit3 places the two children of parent cell `origin` at
// origin -+ 1/4, so `origin` is the parent cell's CENTRE and the cell spans
// [origin - 1/2, origin + 1/2).
//
// From that, one line places a point at any level. A level-m cell has size
// 2^-m in L0 units, and its centres continue the same convention, so
//
//     centre(g, m) = (g + 1/2) * 2^-m - 1/2      and therefore
//     g(p, m)      = floor((p + 1/2) * 2^m)
//
// which is m = 0 -> floor(p + 1/2), the dense cell containing p, and
// m = 1 -> the fine cell whose centre is g/2 - 1/4. That is the GLOBAL FINE
// route d3-amr.mjs uses as the INDEPENDENT check on tile addressing --
// ownership by one division, no ring and no offsets -- which is exactly why
// it is the right frame for a sampler: a viewer must never reach a ring
// cell, and going through the owning block by construction means it cannot.
//
// PER-LEVEL BINDINGS, FIXED IN NUMBER, because WebGPU has no array of
// buffers. Levels the run does not have are bound to a 4-byte dummy and
// SAMPLE_LEVELS folds them out of the walk. The alternative -- one buffer
// per level chosen at runtime -- does not exist in this API.

// Level 0, the dense grid. INTERLEAVED, and the type says so -- see
// common_d3_parentmac_dense.wgsl for what the other convention cost.
@group(0) @binding(0) var<storage, read> mac : array<vec4<f32>>;
// Levels 1..4, each as (mac pool, blockSlot). PLANAR, like every pool array.
@group(0) @binding(2) var<storage, read> macP1 : array<f32>;
@group(0) @binding(3) var<storage, read> bsP1  : array<i32>;
@group(0) @binding(4) var<storage, read> macP2 : array<f32>;
@group(0) @binding(5) var<storage, read> bsP2  : array<i32>;
@group(0) @binding(6) var<storage, read> macP3 : array<f32>;
@group(0) @binding(7) var<storage, read> bsP3  : array<i32>;
@group(0) @binding(8) var<storage, read> macP4 : array<f32>;
@group(0) @binding(9) var<storage, read> bsP4  : array<i32>;

// How many POOL levels exist, i.e. LEVELS - 1. Zero on a dense run, and then
// every branch below folds away and this file costs nothing.
override SAMPLE_LEVELS : u32 = 0u;

struct TreeSample {
  // [rho, ux, uy, uz], whichever level won.
  v     : vec4<f32>,
  // The level it came from: 0 for the dense grid. Returned rather than
  // inferred because it is the ONLY thing about this function that a host
  // check can score exactly -- "which level owns this point" is a statement
  // about blockSlot alone, and d3-amr.mjs can answer it independently.
  // Given the level, the value is one indexing expression that
  // tools/test-d3-amr.js already covers.
  level : u32,
  // The level's cell size in L0 units, 2^-level. The caller needs it for
  // any finite difference: differencing over a FIXED step would read the
  // same cell twice inside a coarse region (a flat zero) or smear across
  // several inside a fine one.
  h     : f32,
}

fn treeWrap(v: i32, n: i32) -> u32 { return u32(((v % n) + n) % n); }

// Is level m present at p, and if so what does it hold? Returns level 0 in
// `level` as the miss signal; callers test it.
fn sampleAtLevel(p: vec3<f32>, m: u32) -> TreeSample {
  var out: TreeSample;
  out.level = 0u; out.h = 1f; out.v = vec4<f32>(0f);
  let s = f32(1u << m);                       // 2^m
  let dims = vec3<i32>(i32(NX), i32(NY), i32(NZ)) * i32(1u << m);
  let gf = floor((p + vec3<f32>(0.5f)) * s);
  let g = vec3<u32>(
    treeWrap(i32(gf.x), dims.x),
    treeWrap(i32(gf.y), dims.y),
    treeWrap(i32(gf.z), dims.z));
  // The block that OWNS this cell: one division, and the local index then
  // lands in the tile's INTERIOR by construction, never its ring.
  let RB2 = 2u * RB;
  let b = g / RB2;
  // Level m's block counts: level m-1's grid is L0 scaled by 2^(m-1), and a
  // block spans RB of its cells.
  let nb = vec3<u32>(NX, NY, NZ) * (1u << (m - 1u)) / RB;
  let id = (b.z * nb.y + b.y) * nb.x + b.x;

  var slot = -1;
  if (m == 1u) { slot = bsP1[id]; }
  else if (m == 2u) { slot = bsP2[id]; }
  else if (m == 3u) { slot = bsP3[id]; }
  else if (m == 4u) { slot = bsP4[id]; }
  if (slot < 0) { return out; }

  let cell = poolCell(u32(slot), g - b * RB2 + vec3<u32>(GHOST));
  var plane = 1u;
  if (m == 1u) { plane = arrayLength(&macP1) / 4u; }
  else if (m == 2u) { plane = arrayLength(&macP2) / 4u; }
  else if (m == 3u) { plane = arrayLength(&macP3) / 4u; }
  else if (m == 4u) { plane = arrayLength(&macP4) / 4u; }
  if (cell >= plane) { return out; }          // unreachable; a slot is in range

  var v: vec4<f32>;
  if (m == 1u) { v = vec4<f32>(macP1[cell], macP1[plane + cell], macP1[2u * plane + cell], macP1[3u * plane + cell]); }
  else if (m == 2u) { v = vec4<f32>(macP2[cell], macP2[plane + cell], macP2[2u * plane + cell], macP2[3u * plane + cell]); }
  else if (m == 3u) { v = vec4<f32>(macP3[cell], macP3[plane + cell], macP3[2u * plane + cell], macP3[3u * plane + cell]); }
  else { v = vec4<f32>(macP4[cell], macP4[plane + cell], macP4[2u * plane + cell], macP4[3u * plane + cell]); }
  out.v = v; out.level = m; out.h = 1f / s;
  return out;
}

// The dense grid, which is present everywhere and therefore the base case.
fn sampleDense(p: vec3<f32>) -> TreeSample {
  var out: TreeSample;
  let gf = floor(p + vec3<f32>(0.5f));
  let c = vec3<u32>(
    treeWrap(i32(gf.x), i32(NX)),
    treeWrap(i32(gf.y), i32(NY)),
    treeWrap(i32(gf.z), i32(NZ)));
  out.v = mac[coarseCell(c)];
  out.level = 0u;
  out.h = 1f;
  return out;
}

// DEEPEST FIRST, and stop at the first hit. The tree is 2:1 balanced and
// octet-complete, but neither property is needed here: the question is only
// "is there a tile at this level covering p", and the deepest one that says
// yes is by definition the finest active level. That makes this loop correct
// on a hierarchy the manager is halfway through rebuilding, which matters
// because the renderer runs between macro-steps and has no lock.
fn sampleTree(p: vec3<f32>) -> TreeSample {
  for (var m = SAMPLE_LEVELS; m >= 1u; m--) {
    let s = sampleAtLevel(p, m);
    if (s.level != 0u) { return s; }
  }
  return sampleDense(p);
}
