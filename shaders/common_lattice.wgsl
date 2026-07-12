// Shared D2Q9 lattice basis, weights, bounce-back opposite-direction
// pairing, and the BGK equilibrium distribution -- identical across every
// kernel that streams/collides/interpolates on the D2Q9 lattice. Included
// via `// @include "common_lattice.wgsl"` (see shader-loader.mjs) rather
// than hand-copied.

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);
// Bounce-back pairing: opposite direction index for each of the 9 links
// (rest is self-opposite; axis pairs 1<->3, 2<->4; diagonal pairs 5<->7,
// 6<->8), matching the ex/ey table above exactly.
const opp = array<u32,9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
const CS2 = 0.33333333f; // D2Q9 lattice speed of sound squared (1/3)

// Named feqD2Q9, not feq -- several including files (the fused step
// kernels) already have their own local `let feq = ...` inside their
// collision loop; a distinct name avoids relying on shadowing rules to
// keep the two apart.
fn feqD2Q9(rho: f32, ux: f32, uy: f32, i: u32) -> f32 {
  let eu = f32(ex[i]) * ux + f32(ey[i]) * uy;
  return wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*(ux*ux+uy*uy));
}
