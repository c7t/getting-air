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
// bindings 7 and 8 WERE level 2's own blockCriterion and blockSlot, read by
// the per-pass cascade tests. Gone with them (B2-2d) -- the closure needs no
// cross-level read here at all, because shaders/amr_cascade.wgsl has already
// put level 2's demands into level 1's want array. Holes, not renumbered:
// see amr_manage_pool.wgsl on why.
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
// binding 7: D0's candidate RANK, one i32 per block, -1 for a non-candidate.
// Reclaims one of the two holes B2-2d left (7 and 8 were level 2's
// blockCriterion/blockSlot for the per-pass cascade) rather than renumbering,
// so no existing binding index moves and the four pages that never set
// DET_SLOTS need only bind a buffer, not re-read their layouts.
//
// It holds the GRANT rank or the RELEASE rank depending on which scan last
// ran, and the two never collide: a block wants a tile and lacks one, or has
// one and is no longer wanted, or neither. It cannot be both.
@group(0) @binding(7)  var<storage, read_write> candRank : array<i32>;

@group(0) @binding(9) var<storage, read_write> diag : array<atomic<u32>, 8>;
// Level 1's WANT array, one u32 per L0 block -- written by decide(), closed
// by shaders/amr_cascade.wgsl, consumed by coarsen/refine when CASCADE != 0.
@group(0) @binding(10) var<storage, read_write> want : array<u32>;

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
// 0 leaves level2Wanted uncalled, so behaviour is exactly as before.
// When 0, isNearBody is unconditionally false -- no interior geometry to
// force-refine toward (channel-flow/TGV scenarios), so refinement is
// purely vorticity-driven. See shaders/lbm_step.wgsl's identical override.
override HAS_BODY : u32 = 1u;
// When 0, isNearBody reverts to the pre-B4 single sample at the block CENTRE
// (?boxrefine=0). Default 1: the whole block is tested. See isNearBody.
override BOX_REFINE : u32 = 1u;

// THE 2:1 DECISION IS NOT IN THIS FILE (plans/2D-backport.md B2). decide()
// writes each block's OWN reason into the want array, shaders/amr_cascade.wgsl
// closes it under the rule, and coarsen/refine below are pure lookups.
//
// WHY THE VETOES WERE DELETED RATHER THAN PORTED. Every balance test that used
// to live here was one direction of the closure read locally, and the pool
// manager's equivalent was STRONGER than the rule it approximated -- which is
// what deadlocked growth rather than what protected it. Applied to the want
// set, "refine forced by a deeper neighbour" and "coarsen blocked by one" are
// the same function. ?cascade=0 kept both paths in one build for one commit;
// the measurement is in B2-2c and the path is gone.

// L0 window-space edge band (coarse cells) excluded from vorticity-driven
// refinement -- keeps fine blocks out of the ALBC sponge (amr_step.wgsl
// SPONGE_W=4). Default 0 disables it; the JS default is 8. Preserves
// isNearBody (the body is window-centered, never in the edge band) and the
// 2:1-balance cascade, gating only the vorticity (epsFor) term.
override SPONGE_EXCLUDE_W : f32 = 0.0f;

// ── D0: ?detslots=1 -- deterministic slot handout (MEASUREMENT MODE) ────────
// Default 0 is the shipped atomic free-list race, byte-identical when absent.
//
// WHAT IT IS FOR. Which slot a block gets depends on which thread reaches the
// `atomicSub` below first, so block->slot assignment varies run to run; and
// amr_force1.wgsl atomicAdds one TRUNCATED i32 per workgroup, so regrouping
// the slots regroups the partials and they truncate differently. That is the
// mechanism behind every attractor table in CLAUDE.md. This flag exists to
// answer ONE question before any of it is engineered away: is slot assignment
// the ONLY live source of run-to-run nondeterminism? If it is, `?detslots=1`
// makes repeated runs BIT-IDENTICAL, and no statistics are needed to say so.
// (The second half of the chain is already exact: integer atomicAdd is
// associative and commutative with no rounding, so once each workgroup's
// partial is fixed the sum is order-independent.)
//
// IT IS NOT THE SHIPPING IMPLEMENTATION. One thread does the whole handout in
// a serial loop -- obviously correct, obviously deterministic, and far too
// slow to default on. The shipping version gives every candidate a RANK from
// a prefix sum over the want set and keeps the parallel dispatch.
//
// AND IT PINS A WEAKER ORDER THAN THE SHIPPING ONE SHOULD. Candidates are
// taken in DISPATCH-INDEX order, which is reproducible only from a
// deterministic initial state (resetSim writes an identity free list), so the
// assignment is a function of the whole run rather than of the current state.
// That is enough to answer the question above and is not enough to ship: the
// real rule should order by BLOCK ID, which is geometry and therefore makes a
// snapshot reload reproduce the same assignment. amr2d.mjs's
// grantAssignment/releaseAssignment state that rule, and tools/test-amr2d.js
// mutation-checks it.
override DET_SLOTS : u32 = 0u;
const BLOCK = 8u;

// This block's own centre and half-extent in L0 buffer space. An L0 block is
// BLOCK cells of size 1, so its footprint is BLOCK L0 units -- the same
// footprint a level-1 tile has at half the cell size, which is what makes
// level 1 footprint-preserving 1:1 with L0 (decision 1). The predicates that
// consume these are common_refine.wgsl's, shared with amr_manage_pool.wgsl.
const HALF_EXTENT_L0 = f32(BLOCK) * 0.5f;
fn blockCentreL0(blockID: u32) -> vec2<f32> {
  let nbx = W / BLOCK;
  let bx = blockID % nbx; let by = blockID / nbx;
  return vec2<f32>(f32(bx * BLOCK + BLOCK / 2u), f32(by * BLOCK + BLOCK / 2u));
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


// THIS BLOCK'S OWN REASON TO EXIST -- criterion, geometry, sponge, and nothing
// about its neighbours. Factored out so decide() and the legacy refine() path
// cannot drift: it is verbatim what `ownReason` was computed as inline.
fn ownWant(blockID: u32) -> bool {
  let c = blockCentreL0(blockID);
  return (desiredLevel(epsOf(blockCriterion[blockID])) >= 1 && !inSpongeBandAt(c))
      || nearBodyAt(c, HALF_EXTENT_L0);
}

// Write the want set. No allocation, no neighbour test, no ordering
// requirement -- every thread writes its own block and reads nothing another
// thread writes, which is what makes the closure that follows well-defined.
@compute @workgroup_size(64)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  if (blockID >= (W / BLOCK) * (H / BLOCK)) { return; }
  want[blockID] = select(0u, 1u, ownWant(blockID));
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  let nblocks = (W / BLOCK) * (H / BLOCK);
  if (blockID >= nblocks) { return; }

  let currentSlot = blockSlot[blockID];
  // A tile exists if and only if it is WANTED. The want set arrives already
  // closed under the 2:1 rule (decide -> shaders/amr_cascade.wgsl -> here), so
  // "can't release a tile that is still a parent" needs no test: a wanted
  // level-2 block implies a wanted level-1 parent by construction. The three
  // per-pass tests that used to live here -- hasLevel2Child on this block, on
  // each edge neighbour, and the criterion ladder -- are all one direction of
  // that closure read locally, and they are gone with plans/2D-backport.md
  // B2-2d.
  if (DET_SLOTS != 0u) {
    // The mirror of refine()'s path: rank among the releasing blocks, push at
    // count0 + rank. amr2d.mjs's releaseAssignment is the rule.
    //
    // Same deferral, same reason: the rank reads blockSlot, so this path does
    // not write it. It clears slotToBlock (which the rank does not read) and
    // leaves blockSlot to linkCoarsen() below. Releasing ALL FOUR of a quad is
    // not a concern here -- the dense manager allocates per block, not per
    // quad (see allocLevelPool's m === 1 branch).
    if (want[blockID] != 0u || blockSlot[blockID] < 0) { return; }
    let rank = candRank[blockID];
    if (rank < 0) { return; } // the scan and this predicate disagree -- refuse rather than corrupt
    // See refine()'s note: linkCoarsen() derives the new free count, so no
    // thread here re-scans the block range to compute a total every other
    // thread is computing too.
    let count0 = atomicLoad(&freeCount);
    let s = blockSlot[blockID];
    freeList[u32(count0 + rank)] = s;
    slotToBlock[u32(s)] = -1;
    return;
  }

  if (want[blockID] == 0u && currentSlot >= 0) {
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
  // Milestone 9's cascade lived here as two tests over the four edge
  // neighbours -- an EXISTENCE one (does a neighbour already have a level-2
  // child) and, behind ?demandCascade, a DESIRE one (does it want one). They
  // were the same rule read twice from one side, and only the veto half was
  // ever complete: a criterion-driven refine could be blocked forever by a
  // neighbour that would only ever have been created BY that refine (B2-1
  // measured level 2 pinned to half its allowed reach because of it).
  // The want set arrives closed, so this is now a lookup.
  if (DET_SLOTS != 0u) {
    // PARALLEL and deterministic: every candidate counts the candidates below
    // it and takes the free-list entry that rank names. amr2d.mjs's
    // grantAssignment is the rule; `freeList[count0 - 1 - rank]` is its
    // `freeIndex`, and refusing at rank >= count0 is its "lowest ids win when
    // the pool is short".
    //
    // ONLY CANDIDATES SCAN, which is what makes an O(nblocks) loop affordable
    // here and not in the serial version this replaces: in steady state a
    // refine round has tens of new candidates, not thousands, and their scans
    // run concurrently. Measured: the serial loop cost ~0.23 ms per refine
    // round, 28% of total sim time at levels=2 (plans/uniform-levels.md 1.2b).
    //
    // THE RANK'S INPUTS MUST NOT MOVE WHILE IT IS BEING COUNTED, and that is
    // why this path does NOT write blockSlot. `want` is read-only during
    // refine; blockSlot is what the racing path mutates, and a thread that
    // scanned after a lower-numbered thread had granted would count one
    // candidate too few and collide on a slot. The blockSlot half of the
    // grant is deferred to linkRefine() below -- a separate dispatch, so the
    // ordering is guaranteed rather than hoped for.
    if (want[blockID] == 0u || blockSlot[blockID] >= 0) { return; }
    let rank = candRank[blockID];
    // Safe to read: nothing writes freeCount during this dispatch. The new
    // value is NOT computed here -- linkRefine() derives it from the pool
    // itself.
    //
    // It used to be, by having every candidate re-scan the whole block range
    // for the total and store the same answer. That was measured and it was
    // the dominant cost: at levels=2 the parallel rank saved almost nothing
    // over the serial loop it replaced (+26.8% against +27.9%), because each
    // of a few hundred candidates was walking 2 x nblocks entries instead of
    // one. The scan a thread cannot avoid is its own rank; the total is the
    // same number for everyone and belongs somewhere it is computed once.
    let count0 = atomicLoad(&freeCount);
    if (rank < 0 || rank >= count0) {
      if (DIAG != 0u) { atomicAdd(&diag[5], 1u); } // pool exhausted -- stay coarse
      return;
    }
    let s = freeList[u32(count0 - 1 - rank)];
    slotToBlock[u32(s)] = i32(blockID);
    newlyActivated[u32(s)] = 1u;
    if (DIAG != 0u) { atomicAdd(&diag[3], 1u); }
    return;
  }

  if (want[blockID] != 0u && currentSlot < 0) {
    let oldCount = atomicSub(&freeCount, 1);
    if (oldCount > 0) {
      let slot = freeList[u32(oldCount - 1)];
      blockSlot[blockID] = slot;
      slotToBlock[u32(slot)] = i32(blockID);
      newlyActivated[u32(slot)] = 1u;
      if (DIAG != 0u) {
        atomicAdd(&diag[3], 1u);
        // diag[4] counted cascade-only grants -- refines nothing else asked
        // for. The closure has no such category (a want is a want), and
        // nothing reads the slot any more: `converged` was retired with it,
        // because its other clause had made it an always-true gate.
      }
    } else {
      atomicAdd(&freeCount, 1); // pool exhausted this round -- undo, stay coarse
      if (DIAG != 0u) { atomicAdd(&diag[5], 1u); }
    }
  }
}

// ── D0's deferred blockSlot writes ──────────────────────────────────────────
//
// refine()/coarsen()'s deterministic paths rank their candidates by scanning
// `blockSlot`, so neither may write it -- see the rank comments there. These
// two passes finish the job afterwards, as separate dispatches, which is what
// makes the ordering a guarantee instead of an assumption.
//
// They are no-ops under the default (racing) path, which writes blockSlot
// inline, and the host does not encode them there at all. Both are also
// IDEMPOTENT: re-running either reproduces the same state, so an accidental
// double dispatch is harmless.

// ── D0's candidate scan ─────────────────────────────────────────────────────
//
// WHY A SCAN AND NOT A PER-CANDIDATE COUNT. The version this replaces had each
// candidate count the candidates below it -- O(candidates x nblocks), no
// buffer, and it read well. It was measured and it did not work: at levels=2
// the card page has hundreds of live level-1 blocks and the criterion churns
// them, so a refine round has hundreds of candidates each walking up to 4096
// entries, and the parallel handout cost the same +27% the serial loop had
// (plans/uniform-levels.md 1.2c-OPEN). The design note's "tens of candidates,
// not thousands" was simply wrong about this page.
//
// ONE WORKGROUP, COOPERATIVE. Each of 256 threads counts its own contiguous
// chunk, thread 0 scans the 256 chunk totals, then each thread writes its
// chunk's ranks from its own offset. Serial depth is nblocks/256 + 256 rather
// than nblocks: 272 against 4096 at this resolution, and it grows with the
// domain far more slowly than the thing it replaces. A two-level scan across
// many workgroups would be faster still and needs a second buffer and a third
// dispatch; this is enough until it is measured not to be.
//
// SCAN_RELEASE picks the predicate. Two pipelines from one entry point, so the
// grant and release rules cannot drift apart into two spellings of "candidate"
// the way refine() and coarsen() once each ran their own geometric test at
// different points (see coarsen()'s own history note).
override SCAN_RELEASE : u32 = 0u;
const SCAN_WG : u32 = 256u;
var<workgroup> chunkTotal : array<i32, SCAN_WG>;

fn isCandidate(b: u32) -> bool {
  if (SCAN_RELEASE != 0u) { return want[b] == 0u && blockSlot[b] >= 0; }
  return want[b] != 0u && blockSlot[b] < 0;
}

@compute @workgroup_size(256)
fn scanCandidates(@builtin(local_invocation_id) lid: vec3<u32>) {
  // Uniform (an override constant), so the barriers below stay in uniform
  // control flow.
  if (DET_SLOTS == 0u) { return; }
  let nblocks = (W / BLOCK) * (H / BLOCK);
  let t = lid.x;
  let per = (nblocks + SCAN_WG - 1u) / SCAN_WG;
  let lo = min(t * per, nblocks);
  let hi = min(lo + per, nblocks);

  var c = 0;
  for (var b = lo; b < hi; b++) { if (isCandidate(b)) { c = c + 1; } }
  chunkTotal[t] = c;
  workgroupBarrier();

  if (t == 0u) {
    var acc = 0;
    for (var i = 0u; i < SCAN_WG; i++) {
      let v = chunkTotal[i];
      chunkTotal[i] = acc;      // exclusive
      acc = acc + v;
    }
  }
  workgroupBarrier();

  var r = chunkTotal[t];
  for (var b = lo; b < hi; b++) {
    if (isCandidate(b)) { candRank[b] = r; r = r + 1; } else { candRank[b] = -1; }
  }
}

// The free count, derived rather than accumulated.
//
// `freeCount` is the number of unallocated slots, so it is a FUNCTION of
// slotToBlock and does not have to be maintained incrementally. Deriving it
// here costs one thread a walk over the POOL (a few hundred slots), where
// maintaining it in refine/coarsen cost every candidate a walk over the
// DOMAIN (thousands of blocks). One thread does it, and it runs in a dispatch
// where nothing else writes slotToBlock, so there is no race to reason about.
//
// This also means the deterministic path never has to get the incremental
// bookkeeping right -- a class of off-by-one that the racing path's
// atomicSub/atomicAdd pair exists to handle and that would have had to be
// re-derived here.
fn recountFree() {
  let nSlots = arrayLength(&slotToBlock);
  var alloc = 0;
  for (var t = 0u; t < nSlots; t++) {
    if (slotToBlock[t] >= 0) { alloc = alloc + 1; }
  }
  atomicStore(&freeCount, i32(nSlots) - alloc);
}

// After coarsen: a block whose slot no longer points back at it was released.
@compute @workgroup_size(64)
fn linkCoarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (DET_SLOTS == 0u) { return; }
  let blockID = gid.x;
  let nblocks = (W / BLOCK) * (H / BLOCK);
  if (blockID >= nblocks) { return; }
  let s = blockSlot[blockID];
  if (s >= 0 && slotToBlock[u32(s)] != i32(blockID)) { blockSlot[blockID] = -1; }
  if (blockID == 0u) { recountFree(); }
}

// After refine: every live slot publishes itself back to its block. Existing
// slots rewrite the value they already had; newly granted ones install theirs.
@compute @workgroup_size(64)
fn linkRefine(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (DET_SLOTS == 0u) { return; }
  let s = gid.x;
  if (s >= arrayLength(&slotToBlock)) { return; }
  let b = slotToBlock[s];
  if (b >= 0) { blockSlot[u32(b)] = i32(s); }
  if (s == 0u) { recountFree(); }
}
