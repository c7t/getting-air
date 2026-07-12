// Compute-savings accounting for tools/validate-amr-vs-dense.js: "AMR did
// X% of the work a dense grid at the same finest resolution would have."
// This is a genuinely different question from field-relL2 accuracy (the
// job of tools/lib/field-reconstruct.js) -- a run can be cheap AND wrong, or
// expensive AND right, so this module never sees field data, only block
// counts and geometry constants.

// Formula A (primary): real GPU dispatch cost in "cell-substep" units for
// ONE AMR L0 macro-step, vs. what a dense grid at the matched finest
// resolution needs to cover the same physical time.
//
//   costAMR         = W0*H0 + sum_{m=1..nLevels-1} activeCountsByLevel[m] * (2*RB)^2 * 2^m
//   costDenseTarget  = W_target*H_target * 2^(nLevels-1)
//   savingsFraction  = costAMR / costDenseTarget
//
// (2*RB)^2 = interior fine cells per active block, fixed size at every
// level (only its physical footprint shrinks as level increases) --
// GHOST cells are bookkeeping overhead, not physics, so they're excluded
// from the interior count here (this is a "how much real physics work
// happened" metric, not a memory-footprint one).
//
// 2^m = substeps per L0 macro-step at level m (main-cylinder-amr.js's
// S_Advance comment, ~line 1690-1700: "level L+k runs 2^k times more often
// than L0" -- the standard LBM refinement-ratio-2 temporal scaling).
//
// activeCountsByLevel[m] counts ALL active blocks at level m, INCLUDING
// ones that themselves have an active child -- the real GPU dispatch still
// runs a parent block's own substep (ghost-fill / 2:1-balance bookkeeping)
// even where a finer child is authoritative for physics/rendering, so this
// is actual-compute-performed, not an idealized non-overlapping partition
// of the domain.
//
// costDenseTarget's own 2^(nLevels-1) factor is not a normalization choice:
// it falls directly out of this codebase's acoustic/convective LBM
// scaling (U0 fixed in lattice units, D/TAU scale with resolution -- see
// tools/lib/amr-resolution-mapping.js's own header) -- a dense run at
// 2^(nLevels-1)x the resolution needs that many more of its own native
// steps to cover the same physical time as one AMR L0 macro-step, exactly
// matching what the AMR multi-rate substepping already encodes per level.
function computeCostSavings({ W0, H0, RB, nLevels, W_target, H_target, activeCountsByLevel }) {
  let costAMR = W0 * H0;
  for (let m = 1; m < nLevels; m++) {
    const active = activeCountsByLevel[m] || 0;
    costAMR += active * (2 * RB) * (2 * RB) * (2 ** m);
  }
  const costDenseTarget = W_target * H_target * (2 ** (nLevels - 1));
  const savingsFraction = costAMR / costDenseTarget;
  return { costAMR, costDenseTarget, savingsFraction, workDoneVsDensePercent: 100 * savingsFraction };
}

module.exports = { computeCostSavings };
