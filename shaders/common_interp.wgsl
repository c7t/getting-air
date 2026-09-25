// Shared coarse->fine interpolation core -- the half of the interp kernels
// that does NOT depend on where the four parent samples came from.
// Included via `// @include "common_interp.wgsl"`.
//
// WHY THIS EXISTS. There were two interp entry files until U7-6f --
// amr_interp_dense_parent.wgsl and amr_interp_pool_parent.wgsl -- differing in
// exactly one thing: how they FETCH a parent cell. The dense one read the L0
// grid at wrapped buffer coordinates (sampleCoarse); the pool one reads a
// parent tile at parent-local interior coordinates (sampleParentPool).
// Everything after those four fetches -- the bilinear blend, the
// Dupuis-Chopard rescale, the reconstruction of f -- was
// character-for-character identical in both, and is here once. The dense one
// is gone and this stays shared: the STEP kernels run the same reconstruction
// inline for the ghost-free work (plans/ghost-free.md).
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

// ── ?dcpre=1 -- the legacy PRE-collision Dupuis-Chopard factor ──────────────
// 0 (default) is the POST-collision factor, which is the one this solver
// needs; 1 restores the factor that shipped until plans/2D-backport.md B1.
// Both live in one build so the defect can be RE-MEASURED rather than
// reconstructed from a checkout -- the same reason ?ghostcopy= and ?f16= are
// still here. Folded at pipeline creation, so ?dcpre=1 is exactly the old
// computation and not a second code path carrying a runtime branch.
override DC_PRE : u32 = 0u;

// Dupuis-Chopard non-equilibrium rescale, coarse->fine.
//
// THE TEXTBOOK FACTOR IS FOR PRE-COLLISION f AND THIS SOLVER DOES NOT HAVE
// ANY. Dupuis-Chopard gives
//
//   fneq_pre,fine = (tau_f/tau_c) * (dx_f/dx_c) * fneq_pre,coarse
//                 = (tau_f/tau_c) * (1/n) * fneq_pre,coarse,   n = 2
//
// because fneq scales as tau * (velocity gradient per lattice cell) and the
// same physical shear spans 2x as many fine cells, halving the per-cell
// gradient. But the step kernel is a FUSED pull-stream + collide, so every
// buffer on both sides of this transfer holds f AFTER collision. BGK gives
// fneq* = ((tau - 1)/tau) * fneq_pre at whichever level it is evaluated, and
// composing decollide -> the relation above -> recollide cancels both tau's
// in the middle:
//
//   fneq*_fine = (1/2) * (tau_f - 1)/(tau_c - 1) * fneq*_coarse
//
// At tau_c = 0.8 the two differ in magnitude AND SIGN (-0.25 against
// +0.6875). Near tau = 0.5 they agree to a few percent, which is why this
// survived on the pages that have a Cd/St number -- see amr2d.mjs's
// dcRescaleCoarseToFine, which is the host statement of both forms and what
// tools/test-amr2d.js scores them against.
//
// TAU = 1 MAKES THIS 0/0 and is refused at init, not divided by: at omega = 1
// the post-collision populations ARE the equilibrium and carry no stress for
// a post-collision transfer to move. See amr2d.mjs's tauChainSingularity.
//
// tauCoarse is the PARENT's own tau, which is not the same thing at every
// level: L0's is state.tau, but a mid-chain parent (L1 acting as parent to L2)
// has its own, tau_fine = 2*tau_coarse - 0.5 applied recursively -- see
// card-params.mjs's tauAtLevel() and amr_interp_pool_parent.wgsl's levelParams.
fn dcRescaleCoarseToFine(tauCoarse: f32, tauFine: f32) -> f32 {
  if (DC_PRE != 0u) { return 0.5f * tauFine / tauCoarse; }
  return 0.5f * (tauFine - 1.0f) / (tauCoarse - 1.0f);
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

  let rescale = dcRescaleCoarseToFine(tauCoarse, 2.0f * tauCoarse - 0.5f);

  var fo: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    let fneq = w00*s00.fneq[i] + w10*s10.fneq[i] + w01*s01.fneq[i] + w11*s11.fneq[i];
    fo[i] = feqD2Q9(rho, ux, uy, i) + rescale * fneq;
  }
  return fo;
}
