// Shared coarse->fine interpolation core -- the half of the interp kernels
// that does NOT depend on where the four parent samples came from.
// Included via `// @include "common_interp.wgsl"`.
//
// WHY THIS EXISTS. amr_interp_dense_parent.wgsl and
// amr_interp_pool_parent.wgsl differ in exactly one thing: how they FETCH a
// parent cell. The dense one reads the L0 grid at wrapped buffer coordinates
// (sampleCoarse); the pool one reads a parent tile at parent-local interior
// coordinates (sampleParentPool). Everything after those four fetches --
// the bilinear blend, the Dupuis-Chopard rescale, the reconstruction of f --
// was character-for-character identical in both, and is now here once.
//
// It is a fragment, so per shader-loader.mjs it must not itself @include.
// It depends on `feqD2Q9` from common_lattice.wgsl; every file that includes
// this one already includes that one. WGSL module-scope resolution is
// order-independent, so the two @include lines may appear in either order.
//
// The reason it is worth sharing rather than leaving duplicated: the
// ghost-free tile layout (plans/ghost-free.md) needs the STEP kernels to run
// this same reconstruction inline, at the point a gather finds no same-level
// neighbour, instead of reading a value a separate interp pass stored into a
// materialized ring. Three copies of this math would be two too many.

struct CoarseSample {
  rho: f32,
  ux: f32,
  uy: f32,
  fneq: array<f32, 9>,
}

// Dupuis-Chopard non-equilibrium rescale, coarse->fine. The factor is
// (tau_fine/tau_coarse) * (dx_fine/dx_coarse) = (tau_fine/tau_coarse) * (1/n),
// with refinement ratio n=2. fneq scales as tau * (velocity gradient per
// lattice cell); the same physical shear spans 2x as many fine cells, so the
// per-cell gradient (and hence fneq) is halved on the fine grid. Omitting the
// 1/n factor leaves an O(1) (2x) non-equilibrium stress discontinuity at every
// fine<->coarse interface, injecting spurious vorticity there.
//
// tauCoarse is the PARENT's own tau, which is not the same thing at every
// level: L0's is state.tau, but a mid-chain parent (L1 acting as parent to L2)
// has its own, tau_fine = 2*tau_coarse - 0.5 applied recursively -- see
// main-amr.js's tauAtLevel() and amr_interp_pool_parent.wgsl's levelParams.
fn dupuisChopardRescale(tauCoarse: f32) -> f32 {
  let tauFine = 2.0f * tauCoarse - 0.5f;
  return 0.5f * tauFine / tauCoarse;
}

// Bilinearly blend four parent samples into one fine cell's distribution.
// (tx,ty) are the fractional position of the fine cell within the parent cell
// at s00, i.e. s00=(x0,y0), s10=(x1,y0), s01=(x0,y1), s11=(x1,y1).
//
// rho/u are blended and then fed to feq, rather than blending f directly:
// equilibrium is reconstructed from the interpolated macroscopic state and
// only the non-equilibrium part is interpolated-and-rescaled. That is what
// makes the rescale above applicable to fneq alone.
fn interpCoarseToFine(
  s00: CoarseSample, s10: CoarseSample, s01: CoarseSample, s11: CoarseSample,
  tx: f32, ty: f32, tauCoarse: f32
) -> array<f32, 9> {
  let w00 = (1f - tx) * (1f - ty);
  let w10 = tx * (1f - ty);
  let w01 = (1f - tx) * ty;
  let w11 = tx * ty;

  let rho = w00*s00.rho + w10*s10.rho + w01*s01.rho + w11*s11.rho;
  let ux  = w00*s00.ux  + w10*s10.ux  + w01*s01.ux  + w11*s11.ux;
  let uy  = w00*s00.uy  + w10*s10.uy  + w01*s01.uy  + w11*s11.uy;

  let rescale = dupuisChopardRescale(tauCoarse);

  var fo: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    let fneq = w00*s00.fneq[i] + w10*s10.fneq[i] + w01*s01.fneq[i] + w11*s11.fneq[i];
    fo[i] = feqD2Q9(rho, ux, uy, i) + rescale * fneq;
  }
  return fo;
}
