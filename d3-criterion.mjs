// THE Q-CRITERION: which blocks a DETACHED VORTEX needs refined.
// plans/3D.md M8.4, sec 1.3. One statement of the rule, shared by the page,
// shaders/common_d3_criterion.wgsl (which mirrors it) and the tools.
//
// THE DESIGN INPUT IS THE TARGET, not a measurement: Re ~ 1100, a narrow
// shear layer on the body, and TIGHT ISOLATED VORTICES shed into the wake.
// The 2D project established that a geometry-forced set does not support
// that regime. So the requirement here is not "refine more", it is
// **FOLLOW STRUCTURES THAT HAVE DETACHED FROM THE BODY** -- a set that is
// not one compact region, that appears where the body is not, and that moves
// with the flow rather than with the geometry.
//
// WHY Q AND NOT |omega|, WHICH IS WHAT 2D USES. In 2D vorticity is a scalar
// and the body is a plate whose boundary layer is a small part of the
// domain. In 3D |omega| flags EVERY shear layer -- including the boundary
// layer over the whole body surface and every free shear layer feeding the
// wake -- which at this Reynolds number is most of the interesting volume.
// The Q-criterion
//
//     Q = 1/2 (|Omega|^2 - |S|^2),     J = grad u,
//     S = (J + J^T)/2,  Omega = (J - J^T)/2
//
// is positive only where ROTATION dominates STRAIN, which is what isolates a
// vortex CORE from the shear layer that produced it. AGAL offers exactly
// this as S_CRITERION for exactly this reason.
//
// THE PROPERTY THAT MATTERS, and the one tools/test-d3-criterion.js asserts
// rather than the arithmetic: **PURE SHEAR HAS Q = 0 AND |omega| = gamma.**
// A criterion built on |omega| cannot tell a boundary layer from a vortex; Q
// can, identically to zero, at every shear rate. That is the whole decision,
// and it is checkable in closed form.
//
// IT NEEDS THE FULL GRADIENT TENSOR -- nine components, not one curl -- which
// is the substantive difference from amr_criterion_pool.wgsl and the reason
// this is its own module rather than a tweak to the 2D one.

// The velocity gradient J_ij = du_i/dx_j by central differences, from the
// six neighbours of a cell. `up[j]` / `um[j]` are the velocity vectors one
// step in +j / -j, and `h` is the step in the SAME units the caller wants the
// gradient in.
//
// SECOND-ORDER AND CENTRED, matching how tools/test-d3-scenarios.js
// differentiates the Beltrami field to check curl(u) = k u: a criterion
// scored against an analytic field has to use the same order of accuracy the
// check does, or the discrepancy is the stencil rather than the criterion.
export function gradU(up, um, h) {
  const J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 3; i++) J[i][j] = (up[j][i] - um[j][i]) / (2 * h);
  }
  return J;
}

// Q from a gradient tensor. Written as the difference of the two Frobenius
// norms rather than the equivalent -1/2 J_ij J_ji, because the two norms are
// individually meaningful (and individually testable) and the compact form
// is one transposition away from being silently wrong.
export function qOfGrad(J) {
  let om2 = 0, s2 = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const S = 0.5 * (J[i][j] + J[j][i]);
      const O = 0.5 * (J[i][j] - J[j][i]);
      s2 += S * S;
      om2 += O * O;
    }
  }
  return 0.5 * (om2 - s2);
}

// Vorticity magnitude from the same tensor -- NOT used by the criterion, and
// present so the test can state the two side by side on one field. The whole
// argument for Q is a claim about how these two differ, and a module that
// only computes the one it uses cannot support the claim.
export function omegaOfGrad(J) {
  const wx = J[2][1] - J[1][2];
  const wy = J[0][2] - J[2][0];
  const wz = J[1][0] - J[0][1];
  return Math.hypot(wx, wy, wz);
}

// THE NORMALIZATION, and it is what makes a threshold portable. Q has units
// of 1/time^2, so a bare value means nothing across scenarios -- the same
// flow at half the velocity scale has a quarter the Q. The reference is the
// body's own shear scale (U/D)^2, so the threshold below is dimensionless and
// a case at another Re or another resolution keeps the same number.
//
// The renderer made this same move for the same reason (main-3d.js's
// U_SCALE / V_SCALE, with VORT_SCALE pinned to 1): hand the shader a ratio,
// not a quantity.
export function qRef(u, d) { return (u / d) ** 2; }

// THE DEFAULT THRESHOLD, in units of qRef.
//
// STATED AS A CHOICE WITH A REASON, not tuned: Q > 0 is the textbook vortex
// definition and would flag every rotation-dominated cell including the
// numerical noise floor, so a bare Q > 0 is not a refinement criterion, it is
// a way to refine most of the domain. The value here is the smallest ROUND
// number that is clear of that floor at the resolutions this solver runs --
// and it is a knob (`?qthresh=`) precisely because the right value is a
// property of the flow, not of this file.
export const Q_THRESHOLD = 0.05;

// HOW FAR AHEAD TO REFINE, in L0 cells, for a structure that CONVECTS.
//
// The geometry criterion already carries this idea and states it as
// `MARGIN + manageEvery * |v|` (common_d3_manage.wgsl): a shell that only
// just covers the body at decision time is stale on the very next step, and
// the constraint has to hold at EVERY step rather than the ones the manager
// runs on. A detached vortex is the same problem with a different velocity --
// it does not sit still while the manager is not looking, it convects at
// roughly the freestream. So the flagged region is dilated by how far the
// flow carries it between decisions.
//
// This is the term whose absence does not look like a bug: the set would
// simply lag the vortices by `manageEvery * u` cells, refining where they
// WERE. `?refine=body` on the drift scenario is the 2D-shaped precedent for
// that failing quietly.
export function convectionLead(manageEvery, u) { return manageEvery * Math.abs(u); }
