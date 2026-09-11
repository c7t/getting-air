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

// THE DEFAULT THRESHOLD, in units of qRef. MEASURED, 2026-09-11, not chosen.
//
// Q > 0 is the textbook vortex definition and would flag every
// rotation-dominated cell including the noise floor, so a bare Q > 0 is not a
// refinement criterion -- it is a way to refine most of the domain. The
// question is where the floor actually sits, and `debugCriterion()` answers
// it on a real shed wake (dense sphere, Re=300, D=16, perturb=0.02):
//
//   t = 0 D/U    max Q/qRef = 0.093   -- freestream plus the 2% symmetry-
//                breaking seed, and NOTHING ELSE. At 0.05 that flags 49.93%
//                of the domain and at 0.02, 99.98%.
//   t = 20..60   max Q/qRef = 12.8..13.1 in the developed wake, i.e. 140x
//                the seeded floor.
//
// So the separation is enormous and the only mistake available is setting the
// threshold INSIDE the floor. 0.05 -- the value first written here, by eye --
// was exactly that mistake: it flags half the domain at step 0, which on a
// dynamic run is slot exhaustion before the flow has developed at all, for
// reasons that have nothing to do with vortices.
//
// 0.1 is the smallest round value that flags ZERO blocks on the seeded
// initial field while flagging 379-628 blocks in the developed wake. It stays
// a knob (`?qthresh=`) because the right value is a property of the flow, not
// of this file -- but it is now a measured default rather than a guess.
export const Q_THRESHOLD = 0.1;

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

// --- COVERAGE: the gate (plans/3D.md M8.4) ---------------------------------
//
// THE FIELD CRITERION'S ANALOGUE OF d3-amr.mjs's checkGeometryCoverage, and
// deliberately the same shape. That one states the geometry criterion's hard
// requirement at CELL granularity -- every cell within `margin` of the body
// must sit in a refined block -- by an INDEPENDENT route from the
// block-corner sampling that built the set. This states the field
// criterion's: **every cell whose Q exceeds the threshold must sit in a
// refined block**, checked per cell against a per-BLOCK-max reduction, which
// is likewise a different computation and not a re-run of the same one.
//
// IT MUST HOLD AT EVERY STEP, NOT AT THE ONES THE MANAGER RUNS ON. That is
// not a new standard: common_d3_manage.wgsl's geometry criterion already
// makes exactly this claim, and its `MANAGE_EVERY * |v|` lead term exists to
// satisfy it -- "a shell that only just covers the body at decision time is
// already stale on the very next step". convectionLead() above is the same
// term for a structure that convects instead of being carried by a body.
//
// WHAT THIS CAN CATCH THAT NOTHING ELSE DOES. A vortex leaving the refined
// region between management events does not blow up, does not drift the
// conservation checks, and does not change the picture: it is simply solved
// at L0 while the run reports a healthy pool. Structural pool checks pass
// (the pool stays perfectly consistent), the interface checks pass (the seam
// is fine, it is just in the wrong place), and Cd moves by an amount nobody
// can attribute. This is the check that names it.
//
// ONE ASYMMETRY WITH THE GEOMETRY VERSION, and it is why the bound here is
// measured rather than declared: a body does not teleport, but a VORTEX CAN
// BE CREATED above threshold somewhere new between two decisions. Convection
// lead cannot cover that, and no lead can. So violations decompose into
// convection (which the lead should remove, and `?qlead=0` should restore)
// and creation (which only a shorter MANAGE_EVERY reduces) -- and the two are
// told apart by whether the violating cell is adjacent to the refined set.
export function checkFieldCoverage({ dims, rb, blockSlot, qAt, thresh, maxReport = 32 }) {
  const [NX, NY, NZ] = dims;
  const nbx = Math.round(NX / rb), nby = Math.round(NY / rb);
  const refinedAt = (x, y, z) => {
    const id = ((Math.floor(z / rb) * nby) + Math.floor(y / rb)) * nbx + Math.floor(x / rb);
    return blockSlot[id] >= 0;
  };
  const violations = [];
  let required = 0, covered = 0, adjacent = 0, qWorst = -Infinity;
  for (let z = 0; z < NZ; z++) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const q = qAt(x, y, z);
        if (!(q > thresh)) continue;
        required++;
        if (refinedAt(x, y, z)) { covered++; continue; }
        // ADJACENT to the refined set means the structure was covered and has
        // moved off the edge of it -- convection, i.e. the lead was too
        // short. Isolated means it appeared where nothing was refined --
        // creation, which only a shorter decision interval reduces. The
        // distinction is the actionable part; a bare violation count is not.
        let near = false;
        for (let d = 0; d < 6 && !near; d++) {
          const o = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]][d];
          const p = [(x + o[0] + NX) % NX, (y + o[1] + NY) % NY, (z + o[2] + NZ) % NZ];
          near = refinedAt(p[0], p[1], p[2]);
        }
        if (near) adjacent++;
        if (q > qWorst) qWorst = q;
        if (violations.length < maxReport) violations.push({ cell: [x, y, z], q, adjacent: near });
      }
    }
  }
  const n = required - covered;
  return {
    required, covered, violations: n, reported: violations,
    // Split by cause, and as a FRACTION of what was required -- an absolute
    // count means nothing without knowing how much there was to cover.
    frac: required ? n / required : 0,
    convection: adjacent, creation: n - adjacent,
    qWorst: n ? qWorst : null,
  };
}
