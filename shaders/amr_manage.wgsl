// Milestone 4b (plans/AMR.md): refine/coarsen decision + free-list
// allocation, one thread per coarse block.
//
// eps = min(1, log2(max(maxOmega, EPS_FLOOR))) -- log2 scale so the
// threshold discriminates by order-of-magnitude vorticity changes, EPS_FLOOR
// avoids log2(0)=-inf for genuinely-still blocks (which then always read as
// far below COARSEN_THRESH, no special-casing needed). REFINE_THRESH >
// COARSEN_THRESH gives a hysteresis band so a block hovering near the
// boundary doesn't refine/coarsen every re-evaluation.
//
// Free list is a classic GPU stack: freeCount (atomic) tracks how many
// slots are currently free; a slot index lives at freeList[freeCount-1]
// (top of stack). Pop: atomicSub returns the OLD count; if positive, the
// slot to use is freeList[oldCount-1]. Push: atomicAdd returns the OLD
// count, which is exactly the index to write the freed slot into.
//
// TWO SEPARATE ENTRY POINTS, dispatched as two SEPARATE compute passes
// (coarsen fully completing before refine starts) -- not a simplification,
// a correctness requirement found by testing: atomics only guarantee the
// freeCount COUNTER is serialized across threads; they say nothing about
// when one thread's plain (non-atomic) write to freeList becomes visible
// to another thread's plain read of the same array within the SAME
// dispatch. A coarsening block's push and a refining block's pop in one
// combined dispatch raced on freeList, producing two different coarse
// blocks with blockSlot[] pointing at the identical pool slot -- caught by
// the interface-continuity check (a ~70% jump instead of the usual few
// percent) and confirmed by cross-validating blockSlot against its
// inverse slotToBlock, which disagreed for exactly the colliding slot.
// Splitting into ordered passes sidesteps the whole hazard: within
// "coarsen" only writes happen (each thread's freeList index is unique, no
// two threads write the same slot); within "refine" only reads happen (of
// data written by the prior, now-complete pass), each at a unique index no
// longer being concurrently written -- no race in either phase.
//
// Milestone 9 (plans/AMR-multilevel.md): 2:1 balance vs. level 2 (see
// shaders/amr_manage_pool.wgsl's header for the full design and the
// coarsen scope-limit note it documents). Two additions, gated by
// HAS_LEVEL2 (0 when N_LEVELS==2, so this file's behavior at today's
// depth is provably unchanged -- childCriterionL2/childBlockSlotL2 are
// harmless dummy buffers in that case, never read):
// - refine(): an L0 block whose own criterion/isNearBody does NOT call
//   for L1 can still be CASCADE-forced to L1 if a same-level (L1)
//   EDGE-NEIGHBOR already wants a level-2 child (reads that neighbor's own
//   4 quadrant blockCriterion[2] values + reuses isNearBody, since a
//   position-based body-proximity test doesn't care which level is being
//   decided -- see amr_manage_pool.wgsl's header on why isNearBody(P) is
//   exactly the right reusable test here). Without this, a level-2 region
//   could end up directly adjacent to a level-0-only region -- a 2-level
//   gap.
//   That cascade is EXISTENCE-based, and it has to be: see this file's own
//   note on binding 7. But existence alone only HOLDS balance around a deep
//   region that already exists -- it cannot GROW one, which left the other
//   half of plans/AMR-multilevel.md's Milestone 9 spec unimplemented and
//   deadlocked. ?demandCascade restores the growth half; see level2Wanted.
// - coarsen(): blocked if this block itself has an active level-2 child
//   (can't release a tile still needed as a parent), or if a same-level
//   (L1) EDGE-NEIGHBOR has one (releasing would leave that neighbor's
//   level-2 child directly adjacent to a level-0-only region once this
//   block drops).
//
// Geometry-forced refinement (blunting fix): the vorticity criterion above
// is a LAGGING signal -- it only fires once the coarse grid has already
// produced (incorrect, under-resolved) vorticity near the card's surface.
// At rest, or whenever the card sweeps into a block faster than the
// REFINE_EVERY re-evaluation cadence catches it, that block sits on coarse
// solid-coupling for a while, "blunting" the card's sharp/thin geometry
// before refinement ever notices (confirmed by this project's own earlier
// tuning note: at step ~4096, live-measured max|omega| was too low to
// trigger the original -5 threshold at all -- i.e. nothing near the card
// was refined during that whole transient). isNearBody() below is an
// unconditional, vorticity-independent test: any block whose center is
// within FORCE_REFINE_MARGIN of the card's ellipse SDF -- evaluated at
// both the CURRENT pose and a linearly-extrapolated pose
// FORCE_REFINE_LOOKAHEAD macro-steps ahead (using the card's current
// vx/vy/omega) -- is forced refined and exempted from coarsening,
// regardless of what the fluid is doing. The lookahead covers the gap
// between refine/coarsen re-evaluations (see main-amr.js's REFINE_EVERY);
// two pose samples (now + future) rather than a continuous sweep is a
// deliberately cheap approximation, generous margin compensating for the
// coarse sampling -- fine to retune alongside REFINE_THRESH/COARSEN_THRESH
// once this is exercised against a live run.

// @include "common_geometry.wgsl"
// @include "common_refine.wgsl"

@group(0) @binding(0) var<storage, read>       blockCriterion  : array<f32>;
@group(0) @binding(1) var<storage, read_write> blockSlot       : array<i32>;
@group(0) @binding(2) var<storage, read_write> slotToBlock     : array<i32>;
@group(0) @binding(3) var<storage, read_write> freeList        : array<i32>;
@group(0) @binding(4) var<storage, read_write> freeCount       : atomic<i32>;
@group(0) @binding(5) var<storage, read_write> newlyActivated  : array<u32>;
@group(0) @binding(6) var<storage, read>       state           : CardState;
// Milestone 9: level 2's own blockCriterion/blockSlot, for the cascade/
// coarsen-block checks -- harmless dummies when HAS_LEVEL2=0 (N_LEVELS==2).
// blockCriterionL2 was unread for a while, after the cascade below switched
// to hasLevel2Child (existence, not criterion-based "wants") -- left bound
// rather than removed, to avoid a bind-group-shape change across every JS
// call site. ?demandCascade reads it again, for GROWTH only (level2Wanted),
// which is the use the existence switch never covered. Keeping the binding
// is what makes that fix free: managePoolBGL is already at the 16-storage-
// buffer per-stage limit, so a new buffer for this would not have fit.
@group(0) @binding(7) var<storage, read>       blockCriterionL2 : array<f32>;
@group(0) @binding(8) var<storage, read>       blockSlotL2      : array<i32>;
// ── Refinement convergence counters (?diag=1) ────────────────────────────────
// The refine round is a FIXED-POINT LOOP run a fixed N_LEVELS-1 times with no
// convergence check -- it stops because the counter ran out, not because the
// structure settled. amr_manage_pool.wgsl's own header records that this
// "didn't always converge in time". These make the LAST iteration report
// itself, so "what is left" is a number rather than something found later as
// an artifact.
//   [3] refines granted this iteration (any reason)
//   [4] ... of which the CASCADE was the only reason -- i.e. 2:1 balance was
//       still propagating outward when the loop stopped. The lag signal.
//   [5] refines wanted but the pool was exhausted
// Gated by DIAG: at 0 no atomic is touched.
@group(0) @binding(9) var<storage, read_write> diag : array<atomic<u32>, 8>;

override DIAG : u32 = 0u;

override W : u32;
override H : u32;
override REFINE_THRESH : f32;
override COARSEN_THRESH : f32;
// Ladder parameters -- see common_refine.wgsl. MAX_LEVEL is the finest
// configured level (N_LEVELS-1); this shader evaluates LEVEL 0 blocks, whose
// dx is 1, so their lattice-unit criterion is already the physical one and
// needs no shift.
override N_REFINE_INC : f32 = 1.0f;
override N_REFINE_MAX : f32 = 1.0f;
override MAX_LEVEL : i32 = 1;
override FORCE_REFINE_MARGIN : f32;
override FORCE_REFINE_LOOKAHEAD : f32;
override HAS_LEVEL2 : u32 = 0u;
// 0 leaves level2Wanted uncalled, so behaviour is exactly as before.
override DEMAND_CASCADE : u32 = 0u;
// When 0, isNearBody is unconditionally false -- no interior geometry to
// force-refine toward (channel-flow/TGV scenarios), so refinement is
// purely vorticity-driven. See shaders/lbm_step.wgsl's identical override.
override HAS_BODY : u32 = 1u;
// L0 window-space edge band (coarse cells) excluded from vorticity-driven
// refinement -- keeps fine blocks out of the ALBC sponge (amr_step.wgsl
// SPONGE_W=4). Default 0 disables it; the JS default is 8. Preserves
// isNearBody (the body is window-centered, never in the edge band) and the
// 2:1-balance cascade, gating only the vorticity (epsFor) term.
override SPONGE_EXCLUDE_W : f32 = 0.0f;
const BLOCK = 8u;
const EPS_FLOOR = 1e-6f;

fn epsFor(blockID: u32) -> f32 {
  return min(1.0f, log2(max(blockCriterion[blockID], EPS_FLOOR)));
}

// True if blockID's center is within FORCE_REFINE_MARGIN of the card's
// surface either right now, or FORCE_REFINE_LOOKAHEAD macro-steps from now.
// Window-space conversion mirrors amr_step.wgsl's wx/wy derivation exactly
// (buffer blocks are fixed in memory; the card is anchored in window space).
//
// The "future" test does NOT extrapolate cx/cy forward -- amr_physics.wgsl's
// moving window keeps cx/cy pinned near (W/2,H/2) by construction (bulk
// translation is absorbed into off_x/off_y, not cx/cy), so cx += vx*lookahead
// would displace a phantom ellipse that doesn't correspond to where the card
// (or this buffer block, relative to it) actually will be. What DOES move,
// relative to the window-anchored card, is a fixed buffer cell's window-space
// position: wx(t) = cx_buf - off_x(t), and off_x grows at rate vx (it tracks
// the card's total world-frame displacement), so wx(t) = wx(now) - vx*t.
// Extrapolate the TEST POINT backward instead of the ellipse forward. theta
// is the one quantity the window doesn't absorb, so it still extrapolates
// forward normally.
fn isNearBody(blockID: u32) -> bool {
  if (HAS_BODY == 0u) { return false; }
  let nbx = W / BLOCK;
  let bx = blockID % nbx; let by = blockID / nbx;
  let cx_buf = bx * BLOCK + BLOCK / 2u;
  let cy_buf = by * BLOCK + BLOCK / 2u;
  let wx = (cx_buf + W - u32(state.off_x)) % W;
  let wy = (cy_buf + H - u32(state.off_y)) % H;
  let p_now = vec2<f32>(f32(wx), f32(wy));

  let phi_now = get_phi(p_now, state);

  let p_future = p_now - vec2<f32>(state.vx, state.vy) * FORCE_REFINE_LOOKAHEAD;
  var future = state;
  future.theta += state.omega * FORCE_REFINE_LOOKAHEAD;
  let phi_future = get_phi(p_future, future);

  return min(phi_now, phi_future) < FORCE_REFINE_MARGIN;
}

// True if blockID's center lies within SPONGE_EXCLUDE_W (coarse cells) of any
// window edge, i.e. inside/near the ALBC sponge band (amr_step.wgsl SPONGE_W).
// Window-space conversion mirrors isNearBody exactly. Gated off when
// SPONGE_EXCLUDE_W <= 0 (the JS default is 8, ?spongeExclude=0 disables it).
fn inSpongeBand(blockID: u32) -> bool {
  if (SPONGE_EXCLUDE_W <= 0.0f) { return false; }
  let nbx = W / BLOCK;
  let bx = blockID % nbx; let by = blockID / nbx;
  let cx_buf = bx * BLOCK + BLOCK / 2u;
  let cy_buf = by * BLOCK + BLOCK / 2u;
  let wx = (cx_buf + W - u32(state.off_x)) % W;
  let wy = (cy_buf + H - u32(state.off_y)) % H;
  let distX = min(f32(wx), f32(W - wx));
  let distY = min(f32(wy), f32(H - wy));
  return min(distX, distY) < SPONGE_EXCLUDE_W;
}

// True if the L1 tile at `blockID1` currently has an active level-2 child
// (quadrant 0 stands for all 4 -- decision 3's all-or-nothing invariant).
fn hasLevel2Child(blockID1: u32) -> bool {
  let nbx = W / BLOCK;
  let bx = blockID1 % nbx; let by = blockID1 / nbx;
  let nbxL2 = nbx * 2u;
  let childBlockID0 = (by * 2u) * nbxL2 + (bx * 2u);
  return blockSlotL2[childBlockID0] >= 0;
}

// True if the L1 tile at `blockID1` -- which must already be active -- has a
// vorticity criterion that ASKS for a level-2 child, whether or not it has
// one yet. The growth half of Milestone 9's refine cascade: "if a neighbor is
// more than one level coarser, force THAT neighbor to refine first".
//
// WHY THIS IS NEEDED, AND WHY IT IS NOT THE TEST THAT WAS REMOVED.
// amr_manage_pool.wgsl's refine() vetoes a criterion-driven L1->L2 refine
// unless all 4 of the parent's same-level neighbours are already active. The
// veto shipped; the forcing never did. So: L1 tile A's criterion asks for L2
// children, but A's neighbour B has no L1 tile, so A is vetoed -- and B is
// created only by its own criterion, by geometry, or by the cascade above,
// which fires on a neighbour that ALREADY HAS an L2 child. A has none,
// because it was just vetoed. B is never created, A never refines. Measured
// live: level 2 never extends past the geometry halo into the wake at all, so
// the L1/L2 boundary is pinned a few cells off the body and every shed vortex
// crosses it there. Self-reinforcing, too -- the thin ragged L1 region is
// what keeps producing the vetoes.
//
// The earlier criterion-based cascade was removed because a "wants" signal is
// re-evaluated fresh every round and can read false for a child that is still
// genuinely active, which let a real imbalance go uncascaded. That argument is
// about HOLDING balance around an existing deep region, and it stands -- which
// is why this is a UNION with the existence test, never a replacement. Used
// only to CREATE, a flickering signal is harmless: a round where it reads
// false simply does not grow the region, and nothing is released on its
// account (coarsen does not consult it). Raising a block from level 0 to 1
// also cannot break 2:1 balance by itself; it can only close a gap.
//
// BOUNDED, not recursive. The plan says "recursively, if the gap is >1", which
// is a poor GPU fit -- but the depth is known up front, so the fixed-point
// loop in the JS dispatch is the bounded equivalent: this grows the region by
// one level per iteration, and it already runs N_LEVELS-1 times, which is the
// exact bound because the refine passes run coarsest-to-finest within an
// iteration. AGAL does the same thing rather than recursing (mesh_amr.cu
// drives a cblock_ID_ref mark field through staged kernels).
//
// SCOPE: this is the L0->L1 hop only. The same deadlock exists one level down
// at N_LEVELS>=4 (an L2 tile wanting L3 children, vetoed for want of an L2
// neighbour), and it cannot be fixed the same way -- amr_manage_pool.wgsl
// would need the level-(m+2) criterion, and it has no binding left. N>=4 is
// unvalidated anyway; fixing it needs a binding freed first.
fn level2Wanted(blockID1: u32) -> bool {
  let nbx = W / BLOCK;
  let bx = blockID1 % nbx; let by = blockID1 / nbx;
  let nbxL2 = nbx * 2u;
  var m = 0f;
  for (var qy = 0u; qy < 2u; qy++) {
    for (var qx = 0u; qx < 2u; qx++) {
      m = max(m, blockCriterionL2[(by * 2u + qy) * nbxL2 + (bx * 2u + qx)]);
    }
  }
  // blockCriterionL2 is reduced from LEVEL 1's own field, so it is in level-1
  // lattice units; the ladder wants physical. A level-1 cell is half an L0
  // cell, so physical log2|omega| = lattice + 1 -- the same +m shift
  // amr_manage_pool.wgsl's toPhysical() applies, and common_refine.wgsl's
  // "PHYSICAL UNITS" note explains why omitting it under-refines by 2^m.
  let eps = min(1.0f, log2(max(m, EPS_FLOOR)));
  return desiredLevel(eps + 1.0f) >= 2;
}

// Same 4 edge-neighbor blockIDs every fine-fine/manage neighbor lookup in
// this codebase uses, factored out since both the cascade and coarsen-
// block checks below need them.
fn edgeNeighbors(blockID: u32) -> array<u32, 4> {
  let nbx = W / BLOCK; let nby = H / BLOCK;
  let bx = blockID % nbx; let by = blockID / nbx;
  return array<u32, 4>(
    ((by + nby - 1u) % nby) * nbx + bx,
    ((by + 1u) % nby) * nbx + bx,
    by * nbx + ((bx + 1u) % nbx),
    by * nbx + ((bx + nbx - 1u) % nbx),
  );
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  let nblocks = (W / BLOCK) * (H / BLOCK);
  if (blockID >= nblocks) { return; }

  let currentSlot = blockSlot[blockID];
  // Ladder: release this block's level-1 tile only if its own flow no longer
  // asks for ANY refinement (desired < 1), hysteresis-shifted. Level 0's dx
  // is 1, so epsFor is already the physical log2|omega|.
  if ((desiredLevelCoarsen(epsFor(blockID)) < 1 || inSpongeBand(blockID)) && currentSlot >= 0 && !isNearBody(blockID)) {
    // Milestone 9: can't release a tile that's still a parent, or whose
    // release would leave a neighbor's level-2 child directly adjacent to
    // a level-0-only region -- see this file's header.
    if (HAS_LEVEL2 != 0u) {
      if (hasLevel2Child(blockID)) { return; }
      let neighbors = edgeNeighbors(blockID);
      for (var i = 0u; i < 4u; i++) {
        if (blockSlot[neighbors[i]] >= 0 && hasLevel2Child(neighbors[i])) { return; }
      }
    }
    let oldCount = atomicAdd(&freeCount, 1);
    freeList[u32(oldCount)] = currentSlot;
    blockSlot[blockID] = -1;
    slotToBlock[u32(currentSlot)] = -1;
  }
}

@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  let nblocks = (W / BLOCK) * (H / BLOCK);
  if (blockID >= nblocks) { return; }

  let currentSlot = blockSlot[blockID];
  // Milestone 9: cascade -- refine even if blockID's OWN criterion doesn't
  // call for it, if an ALREADY-ACTIVE same-level neighbor ALREADY HAS a
  // level-2 child (2:1 balance: that neighbor's level-2 child can't sit
  // directly next to a level-0-only region) -- see this file's header.
  // EXISTENCE (hasLevel2Child), not desire -- see amr_manage_pool.wgsl's
  // own identical fix/rationale (a criterion-based "wants" test can
  // flicker false for a still-genuinely-active child whose criterion
  // dipped this round, letting a real imbalance go uncascaded).
  var cascadeWanted = false;
  if (HAS_LEVEL2 != 0u && currentSlot < 0) {
    let neighbors = edgeNeighbors(blockID);
    for (var i = 0u; i < 4u; i++) {
      if (blockSlot[neighbors[i]] >= 0 && hasLevel2Child(neighbors[i])) { cascadeWanted = true; }
      // GROWTH: the neighbour has no L2 child yet but its criterion asks for
      // one, and it cannot get one until this block exists. Union with the
      // existence test above, never a replacement -- see level2Wanted.
      if (DEMAND_CASCADE != 0u && blockSlot[neighbors[i]] >= 0 && level2Wanted(neighbors[i])) { cascadeWanted = true; }
    }
  }
  let ownReason = (desiredLevel(epsFor(blockID)) >= 1 && !inSpongeBand(blockID)) || isNearBody(blockID);
  if ((ownReason || cascadeWanted) && currentSlot < 0) {
    let oldCount = atomicSub(&freeCount, 1);
    if (oldCount > 0) {
      let slot = freeList[u32(oldCount - 1)];
      blockSlot[blockID] = slot;
      slotToBlock[u32(slot)] = i32(blockID);
      newlyActivated[u32(slot)] = 1u;
      if (DIAG != 0u) {
        atomicAdd(&diag[3], 1u);
        // Attributed to the cascade only when nothing else asked for it: a
        // block its own criterion or geometry already wanted is not evidence
        // that the balance cascade is still propagating.
        if (cascadeWanted && !ownReason) { atomicAdd(&diag[4], 1u); }
      }
    } else {
      atomicAdd(&freeCount, 1); // pool exhausted this round -- undo, stay coarse
      if (DIAG != 0u) { atomicAdd(&diag[5], 1u); }
    }
  }
}
