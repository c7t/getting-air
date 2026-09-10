// GENERATED FILE -- do not edit. Source: lattice-3d.mjs (latticeWGSL(19)),
// regenerate with `node tools/gen-lattice-3d.js`, guarded by
// tools/test-lattice-3d.js (which also re-checks the moment conditions
// against the tables PARSED BACK OUT of this file, so an agreed-upon wrong
// table still fails).
//
// D3Q19 lattice basis, weights, bounce-back pairing and BGK equilibrium.
// Fragment only -- included via `// @include "common_d3q19_lattice.wgsl"`
// (see shader-loader.mjs), never compiled alone. Every 3D kernel body
// fragment is written against these names and is therefore velocity-set
// agnostic; the entry-point file chooses Q by choosing which of these two
// lattice fragments to include.
//
// Direction order: rest, then face, then edge -- and within each
// shell every direction is immediately followed by its own negation, so
// opp[i] is i+1 for odd i and i-1 for even i>0. Index ranges:
//   rest   [0, 1)
//   face   [1, 7)
//   edge   [7, 19)
//   corner [19, 19)   <- empty: D3Q19 has no corner directions, so a
//                     same-level gather never needs a corner neighbour
// D3Q19's indices are a strict PREFIX of D3Q27's: direction i is the same
// physical direction in both sets.

const QN : u32 = 19u;
const CS2 = 1.0f/3.0f; // lattice speed of sound squared

const ex = array<i32,19>(
   0,  1, -1,  0,  0,  0,  0,  1, -1,  1,
  -1,  1, -1,  1, -1,  0,  0,  0,  0
);
const ey = array<i32,19>(
   0,  0,  0,  1, -1,  0,  0,  1, -1, -1,
   1,  0,  0,  0,  0,  1, -1,  1, -1
);
const ez = array<i32,19>(
   0,  0,  0,  0,  0,  1, -1,  0,  0,  0,
   0,  1, -1, -1,  1,  1, -1, -1,  1
);
const wt = array<f32,19>(
  1.0f/3.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
  1.0f/18.0f, 1.0f/18.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
  1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
  1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
);
// Bounce-back pairing: opposite-direction index for each of the 19 links.
const opp = array<u32,19>(
   0u,  2u,  1u,  4u,  3u,  6u,  5u,  8u,  7u, 10u,
   9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u
);

// Named feqD3Q, not feq -- the fused step kernels already declare a local
// `let feq = ...` inside their collision loop, and a distinct name keeps
// the two apart without relying on shadowing rules (same reason
// common_lattice.wgsl names the 2D one feqD2Q9).
fn feqD3Q(rho: f32, ux: f32, uy: f32, uz: f32, i: u32) -> f32 {
  let eu = f32(ex[i]) * ux + f32(ey[i]) * uy + f32(ez[i]) * uz;
  return wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*(ux*ux + uy*uy + uz*uz));
}
