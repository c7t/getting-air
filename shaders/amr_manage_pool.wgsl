// Milestone 9 (plans/AMR-multilevel.md): refine/coarsen decision + quad
// allocation for any L(m)->L(m+1) hop.
//
// EVERY HOP, INCLUDING L0->L1, SINCE U7-6f. It served m>=1 only while a
// separate dense-parent manager (amr_manage.wgsl) owned the L0->L1 decision
// and allocated level 1 per BLOCK rather than per quad; U5-4 made the root the
// parent of level 1 and U7-6f deleted the other manager, so this is the only
// allocator in the project and the pool is uniform from the root down.
//
// One pipeline instance PER PARENT LEVEL (NBX_PARENT/NBY_PARENT/
// PARENT_CELL_SIZE_L0/PARENT_HAS_CACHED_ORIGIN baked as overrides) --
// unlike interp/step1/average/force's pool-parent shaders, criterion/
// manage never had an established "one pipeline shared across every
// level>=2" pattern before this milestone, and at this plan's actual
// validated depth (N=3) this shader only ever runs with parent=level 1
// anyway. One pipeline per parent level, override-baked, is simpler than
// adding new per-parent-level runtime uniforms just to future-proof past
// N=3 -- retarget if a later milestone actually needs N>=4.
//
// PARENT_HAS_CACHED_ORIGIN selects how a candidate's parent origin is
// obtained: 0 when parent=level 1 (origin derived cheaply from blockID,
// like amr_step1.wgsl -- see that file's header on why level 1 doesn't
// cache origin at all), 1 when parent=level>=2 (origin read from that
// level's own cached originX/Y buffers, like amr_force1_pool.wgsl). When 0,
// parentOriginX/parentOriginY are harmless dummy buffers, never read.
//
// 2:1 balance (decision from plans/AMR-multilevel.md's Milestone 9):
// - refine(): a parent slot may only spawn a level-(m+1) quad if its own
//   same-level (level-m) edge-neighbors are ALSO active, UNLESS this
//   refine is HARD-REQUIRED (geometry -- isNearBodyAt -- or a mandatory
//   2:1-balance cascade from a deeper neighbor), in which case the
//   neighbor-active gate doesn't apply at all. Changed from an earlier
//   version where the gate applied unconditionally: geometry-forced
//   refinement is meant to be a HARD constraint (the body's surface
//   reaches the finest configured level regardless of what its coarser
//   neighborhood happens to look like this round), with 2:1 balance then
//   DRIVEN FROM that outward, not used as a precondition that can veto
//   it. Live-verified this was a real, live bug, not just a theoretical
//   gap: candidates well within the geometric margin (e.g. phi=1.98
//   against a ~4-unit margin) were found NOT getting their required
//   child, because a same-level neighbor happened not to be active yet
//   -- the cascade meant to catch this up within the same round didn't
//   always converge in time. The OUTWARD cascade that restores 2:1
//   balance after a hard-required deep refine is a shallower instance of
//   THIS file's own cascade -- existence- (not
//   criterion-) based for the same reason this file's own grandchild
//   cascade is, below. A criterion (vorticity)-only refine -- no
//   geometric or cascade reason -- still needs the gate, so vorticity-
//   driven growth alone can't outrun its own coarser neighborhood; only
//   the two MANDATORY reasons bypass it.
// - coarsen()/refine() BOTH also need a THIRD level's data (grandchild,
//   level m+2) to stay exact once N_LEVELS>=4 -- FIXED, not scope-limited
//   anymore. Discovered live: forcing N_LEVELS=4 (after the free-list
//   eager-init fix elsewhere in this milestone let level>=2 refinement
//   actually engage for the first time) produced real, reproducible
//   debugCheck21Balance violations -- a level-1-only tile directly
//   adjacent to a level-3 tile (depth diff 2). Root cause: refine()'s
//   existing same-level-m neighbor check only verifies neighbors are
//   ACTIVE at level m, which is necessary but not sufficient -- it says
//   nothing about whether one of those neighbors is ITSELF about to grow
//   a level-(m+2) grandchild, which would make this parent's own level-
//   (m+1) presence mandatory too, not optional. Two additions, gated by
//   HAS_GRANDCHILD (0 when m+2>=N_LEVELS, so behavior at this level's
//   own deepest-configured case is provably unchanged -- grandchildBlockSlot
//   is a harmless dummy buffer in that case, never read):
//   - refine(): cascade -- force this parent to spawn its level-(m+1) quad
//     even if its own criterion doesn't call for it, if a level-(m+1) tile
//     SHARING A FACE with this parent's own 2x2 child footprint already HAS
//     an active level-(m+2) grandchild. (That "sharing a face" is load-
//     bearing and was got wrong once -- see refine()'s own BUGFIX comment on
//     why testing the neighbor PARENT's quadrant-0 child instead produced
//     both missed cascades AND a refine/coarsen oscillation.) EXISTENCE
//     (hasGrandchild), not desire -- an earlier version of this fix used a
//     criterion-based "does the neighbor's child WANT a grandchild" test
//     (mirroring the deleted dense manager's own level2WantsRefine one level down)
//     and it was live-verified wrong: debugCheck21Balance still caught
//     real depth-1-vs-depth-3 violations with it, because criterion is
//     re-evaluated fresh every round and can read as "doesn't want it
//     anymore" for a grandchild that's still genuinely active (coarsen()
//     hasn't released it yet) -- the cascade must react to what's actually
//     THERE right now, not to a lagging/flickering desire signal.
//   - coarsen(): blocked if any of the 4 children about to be released
//     itself has an active level-(m+2) child, OR if any of those 4
//     children's own same-level-(m+1) EDGE-NEIGHBORS has one (releasing
//     would leave that neighbor's grandchild directly adjacent to this
//     now-coarser region). Same-level-(m+1) neighbor lookup reuses
//     childBlockSlot (already bound for this file's own quad bookkeeping),
//     not a parent-level traversal -- level (m+1) is the level actually
//     being coarsened, so its own same-level neighbor structure is what
//     2:1 balance is checked against, exactly mirroring how refine()'s
//     existing check operates at the PARENT's own level m, not one level
//     removed from what's being decided.

// @include "common_geometry.wgsl"
// @include "common_refine.wgsl"

@group(0) @binding(0)  var<storage, read>       childCriterion    : array<f32>;
@group(0) @binding(1)  var<storage, read_write> childBlockSlot    : array<i32>;
@group(0) @binding(2)  var<storage, read_write> childSlotToBlock  : array<i32>;
@group(0) @binding(3)  var<storage, read_write> childFreeList     : array<i32>; // quad-indexed
@group(0) @binding(4)  var<storage, read_write> childFreeCount    : atomic<i32>; // quad-indexed
@group(0) @binding(5)  var<storage, read_write> childNewlyActivated : array<u32>;
@group(0) @binding(6)  var<storage, read>       state             : CardState;
@group(0) @binding(7)  var<storage, read_write> childParentSlot   : array<i32>;
// binding 8 WAS childQuadrant, and it held `slot % 4`.
//
// A slot's quadrant is a function of the slot index, not stored data: both
// allocators compose the slot as `quadIdx*4 + quadrant` (refine() below, and
// main-amr.js's debugActivateBlock), so the buffer stored a constant and
// refine() rewrote it on every allocation. amr2d.mjs's `quadrantOfSlot` is
// the rule; amr2d-gpu.mjs's checkSlotQuadrantsOnGPU scored the live buffer
// against it over 352 and 212 active slots, at ?levels=3 and ?levels=4, after
// 8000+ steps of real refinement, before this binding was removed.
//
// WHY IT MATTERED ENOUGH TO CHASE. This kernel declared EXACTLY 16 storage
// buffers, which is `maxStorageBuffersPerShaderStage` on the target hardware
// -- see CLAUDE.md. There was no room for anything, and the next thing that
// needs room is plans/2D-backport.md B2's want array. The buffer itself stays
// (several other shaders read it, and the allocator now writes it once at
// allocation instead of per-refine); it is this kernel's BINDING that is
// recovered. 8 is left as a hole rather than renumbered: a renumber is five
// separate pages' bind groups to land in lockstep, which is the shape that
// shipped 238e48c.
//
// AND B2 IS WHAT IT WAS RECOVERED FOR. The child level's WANT array now sits
// here -- written by decide(), closed by shaders/amr_cascade.wgsl, consumed by
// coarsen/refine when CASCADE != 0. That puts this kernel back at exactly 16,
// which is legal but leaves no slack again; B2-2c's deletions
// (parentBlockSlot, grandchildBlockSlot -- both dead under the closure) take
// it to 14.
@group(0) @binding(8)  var<storage, read_write> childWant         : array<u32>;
// ── Refinement refusal counter (?diag=1) ─────────────────────────────────────
//
// Binding 9: diag[5] counts refines refused for want of a pool slot. The
// number was chosen to match the dense manager's binding for the SAME buffer
// and the same slot within it; that manager is gone, the numbering stays.
//
// ADDED AT plans/uniform-levels.md U5-4, because that stage took level 1 off
// the dense manager and the refusal counter went with it. Measured before and
// after on `amr-dev-invariants --extra=maxFineBlocks=16&diag=1`: the dense
// manager reported `pool STARVED (2140 refine(s) refused)` and the pool
// manager reported `pool OK` on an identically starved run. That is a gate
// going VACUOUS rather than a gate going green, and CLAUDE.md's own count of
// how many vacuous gates this project has collected is the reason it is fixed
// here rather than noticed later.
//
// It is also the instrument POOL_PEAKS depends on: the defaults are a measured
// peak times 1.7, and the refusal watch is what reports a wrong guess.
@group(0) @binding(9)  var<storage, read_write> diag              : array<atomic<u32>, 8>;
override DIAG : u32 = 0u;
// bindings 9/10 WERE childOriginX/childOriginY, written here at refine time,
// and 13/14 WERE parentOriginX/parentOriginY, read here to write them. All
// four are gone (plans/2D-backport.md B3-5): a tile's physical origin is
// `block * RB * 2^-(m-1)` in closed form -- see parentOriginL0() below -- so
// the recursion this kernel ran, and the per-slot buffers it ran it into, were
// storing something cheaper to compute. B3-1a and B3-4 took the last two
// READERS (the fine step and the force pass) off those buffers; this removes
// the writer, the buffers, and PARENT_HAS_CACHED_ORIGIN with them. 14 -> 10
// declared, on the kernel that used to sit at the 16 ceiling.
// binding 11 WAS parentBlockSlot, read only by the neighbour-active veto.
// Gone with it (B2-2d).
@group(0) @binding(12) var<storage, read>       parentSlotToBlock : array<i32>;
// Grandchild (level m+2) blockSlot, for the 2:1-balance cascade/coarsen-
// block checks -- dummy if !HAS_GRANDCHILD, see header. Both refine() and
// coarsen() only ever need EXISTENCE (hasGrandchild), never level (m+2)'s
// criterion -- see hasGrandchild's own comment on why "wants" isn't the
// right test for maintaining balance against an already-active grandchild.
// binding 15 WAS grandchildBlockSlot, read only by hasGrandchild. Gone with
// it (B2-2d). Holes at 11 and 15 rather than a renumber: five separate
// pages' bind groups would have to land in lockstep, which is the shape
// that shipped 238e48c. 16 -> 14 declared.

override W : u32;
override H : u32;
override RB : u32;
override NBX_PARENT : u32;
override NBY_PARENT : u32;
override PARENT_CELL_SIZE_L0 : f32;
// When 0, isNearBodyAt is unconditionally false -- the two bodyless pages
// (TGV, channel) pass it.
override HAS_BODY : u32 = 1u;
// When 0, isNearBodyAt reverts to the pre-B4 single sample at the tile CENTRE
// (?boxrefine=0).
override BOX_REFINE : u32 = 1u;

// THE 2:1 DECISION IS NOT IN THIS FILE (plans/2D-backport.md B2). decide()
// writes each parent's OWN reason into the child level's want array,
// shaders/amr_cascade.wgsl closes it under the rule, and coarsen/refine are
// pure lookups. ?cascade=0 kept both paths in one build for one commit; the
// measurement is in B2-2c and the path is gone.

override REFINE_THRESH : f32;
override COARSEN_THRESH : f32;
// Ladder parameters -- see common_refine.wgsl.
override N_REFINE_INC : f32 = 1.0f;
override N_REFINE_MAX : f32 = 1.0f;
override MAX_LEVEL : i32 = 2;

// This shader evaluates blocks at the PARENT level m, whose cells are
// PARENT_CELL_SIZE_L0 = 2^-m of an L0 cell. The criterion kernels reduce a
// raw lattice-cell difference, so converting to a physical gradient means
// dividing by that -- i.e. adding m in log2. This is AGAL's /(2.0*dx_L).
fn myLevel() -> i32 { return i32(round(-log2(PARENT_CELL_SIZE_L0))); }

// A TILE'S PHYSICAL ORIGIN IN L0 UNITS, AS ONE MULTIPLY -- here, the PARENT
// tile's, from the parent's own block index. Every level's block grid is
// globally anchored and quadtree-uniform, so the origin is
// `block * RB * 2^-(m-1)`, and `2^-(m-1)` is `2 * PARENT_CELL_SIZE_L0` at the
// parent level. amr2d.mjs's tileOriginL0 is the host statement of the same
// closed form, and tools/test-amr2d.js scores it against the recursive route
// this kernel used to run.
//
// That recursion (parent origin + quadrant offset * the parent's cell size,
// cached per slot) is what got transposed once and cost a wrong level-2 force.
// It is not here to get wrong any more.
fn parentOriginL0(bxP: u32, byP: u32) -> vec2<f32> {
  let s = f32(RB) * 2.0f * PARENT_CELL_SIZE_L0;
  return vec2<f32>(f32(bxP) * s, f32(byP) * s);
}
fn toPhysical(eps: f32) -> f32 { return eps - log2(PARENT_CELL_SIZE_L0); }

// This parent tile's centre and half-extent in L0 buffer space, the pair
// common_refine.wgsl's predicates consume. The parent's interior is 2*RB cells
// of size PARENT_CELL_SIZE_L0, so the footprint is 2*RB*PARENT_CELL_SIZE_L0
// and the half of it is RB*PARENT_CELL_SIZE_L0 -- RB is ALREADY "half the
// interior" by construction, which is why no further *0.5 belongs in either.
//
// Deriving the centre and the half-extent from the same expression is
// deliberate: if one is ever wrong the other is wrong with it, rather than the
// box silently straddling a tile it does not belong to. Both bugs this file's
// git history records (a centre off by half a tile, and a quadrant step off by
// the same factor) were exactly that pair disagreeing.
fn parentHalfExtentL0() -> f32 { return f32(RB) * PARENT_CELL_SIZE_L0; }
fn parentCentreL0(bxP: u32, byP: u32) -> vec2<f32> {
  return parentOriginL0(bxP, byP) + vec2<f32>(parentHalfExtentL0(), parentHalfExtentL0());
}
override FORCE_REFINE_MARGIN : f32;
override FORCE_REFINE_LOOKAHEAD : f32;
// L0 window-space edge band excluded from vorticity-driven refinement -- a
// fixed L0-window strip, UNSCALED per level, since the sponge it shadows is
// itself a fixed L0 strip. Gated off when <= 0.
override SPONGE_EXCLUDE_W : f32 = 0.0f;

// THE WANT SET FOR THE CHILD LEVEL. Dispatched over PARENT slots, like
// refine(), because that is where the criterion and the geometry live.
//
// TWO DIFFERENCES FROM refine(), both load-bearing:
//
//   NO "already refined" EARLY-OUT. Want is about desire, not about what
//   exists. A quad that already exists and is still wanted must be MARKED
//   wanted, or coarsen() releases it on the same round.
//   NO CASCADE AND NO VETO. Neighbours are not this function's business --
//   that is what shaders/amr_cascade.wgsl is for, and keeping it out of here
//   is the entire point of B2.
//
// Blocks whose parent slot is inactive are never visited, so the caller must
// CLEAR the want buffer before dispatching this.
@compute @workgroup_size(64)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
  let parentSlot = gid.x;
  if (parentSlot >= arrayLength(&parentSlotToBlock)) { return; }
  let parentBlockID = parentSlotToBlock[parentSlot];
  if (parentBlockID < 0) { return; }

  let bxP = u32(parentBlockID) % NBX_PARENT;
  let byP = u32(parentBlockID) / NBX_PARENT;
  let nbxChild = NBX_PARENT * 2u;

  var maxCrit = 0f;
  for (var qy = 0u; qy < 2u; qy++) {
    for (var qx = 0u; qx < 2u; qx++) {
      let cb = (byP * 2u + qy) * nbxChild + (bxP * 2u + qx);
      maxCrit = max(maxCrit, childCriterion[cb]);
    }
  }
  let eps = epsOf(maxCrit);

  let centre = parentCentreL0(bxP, byP);

  let wanted = nearBodyAt(centre, parentHalfExtentL0())
    || (desiredLevel(toPhysical(eps)) > myLevel() && !inSpongeBandAt(centre));

  // All four children, because the pool allocates quads and nothing else --
  // so the want this level produces is quad-complete by construction, and
  // amr_cascade.wgsl's completeQuads only ever has the CLOSURE's own
  // single-block additions left to finish.
  let v = select(0u, 1u, wanted);
  for (var q = 0u; q < 4u; q++) {
    let cbx = bxP * 2u + (q & 1u);
    let cby = byP * 2u + ((q >> 1u) & 1u);
    childWant[cby * nbxChild + cbx] = v;
  }
}

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

// Does this parent slot want a child quad it does not already have? Shared by
// both handout paths so the racing and deterministic routes cannot disagree
// about WHICH blocks are candidates -- only about the order they are served.
fn refineWants(parentSlot: u32) -> bool {
  if (parentSlot >= arrayLength(&parentSlotToBlock)) { return false; }
  let parentBlockID = parentSlotToBlock[parentSlot];
  if (parentBlockID < 0) { return false; } // parent not active -- not a candidate at all
  let bxP = u32(parentBlockID) % NBX_PARENT;
  let byP = u32(parentBlockID) / NBX_PARENT;
  let nbxChild = NBX_PARENT * 2u;
  // Already refined? Quadrant 0 stands for all 4 (decision 3's all-or-nothing
  // invariant).
  let childBlockID0 = (byP * 2u) * nbxChild + (bxP * 2u);
  if (childBlockSlot[childBlockID0] >= 0) { return false; }

  // NO CRITERION AND NO GEOMETRY HERE. Both were still being computed at this
  // point -- a max over the 4 prospective quadrants, its log2, the parent's
  // origin and its centre -- and NOTHING below read any of it. B2-2d deleted
  // the three tests that consumed them (the grandchild cascade, the
  // hard-required split, the neighbour-active gate) and left the inputs
  // behind. Dead since then; removed in B3-7 (plans/2D-backport.md).
  //
  // Legal WGSL and free at runtime (any compiler drops an unused `let`), which
  // is exactly why it survived: the cost was that this function READ as though
  // it still weighed the criterion, next to a 20-line BUGFIX comment about a
  // centre it no longer computed for anything. Deleting a mechanism has to
  // include deleting what fed it -- the same sweep B2-2d ran over the CHECKS
  // should have run over the INPUTS.

  // The grandchild-cascade essay that stood here is in git history: two
  // separately live-verified index bugs in an 8-cell ring walk, and a
  // "wants"-vs-"has" flicker that forced an existence test. All of it was
  // this file computing one hop of a transitive closure by hand.
  // A QUAD EXISTS IF AND ONLY IF IT IS WANTED (plans/2D-backport.md B2-2d).
  //
  // Three separate mechanisms used to stand between here and the allocator,
  // and all three were the 2:1 rule read locally from one side:
  //
  //   the GRANDCHILD CASCADE -- an 8-cell ring walk testing hasGrandchild,
  //     whose own removed comment records getting the wrong child index on
  //     three sides out of four, twice, with both error directions
  //     live-verified at N=4;
  //   the HARD-REQUIRED split -- geometry and cascade bypassing the gate
  //     while a criterion-only refine kept it, which is the distinction the
  //     closure erases (a want is a want);
  //   the NEIGHBOUR-ACTIVE GATE -- all four of the PARENT's same-level
  //     neighbours active, which is STRICTLY STRONGER than 2:1 balance. The
  //     rule only demands the parents of this child's own neighbours. That
  //     extra strength was not a safety margin, it was the deadlock: a refine
  //     blocked by a neighbour that would only ever have been created BY that
  //     refine, measured (B2-1) as level 2 pinned to half its allowed reach.
  //
  // The want array arrives closed under the rule, so none of it has anything
  // left to decide.
  return childWant[childBlockID0] != 0u;
}

// Hand one free quad to one parent slot. Lifted verbatim out of refine() so the
// serial path can reuse it; the arithmetic and every comment below are
// unchanged.
fn grantQuad(parentSlot: u32, quadIdx: i32) {
  let parentBlockID = parentSlotToBlock[parentSlot];
  let bxP = u32(parentBlockID) % NBX_PARENT;
  let byP = u32(parentBlockID) / NBX_PARENT;
  let nbxChild = NBX_PARENT * 2u;
  let baseSlot = u32(quadIdx) * 4u;
  for (var qy = 0u; qy < 2u; qy++) {
    for (var qx = 0u; qx < 2u; qx++) {
      let quadrant = qx + 2u * qy;
      let slot = baseSlot + quadrant;
      let childBX = bxP * 2u + qx;
      let childBY = byP * 2u + qy;
      let childBlockID = childBY * nbxChild + childBX;
      childBlockSlot[childBlockID] = i32(slot);
      childSlotToBlock[slot] = i32(childBlockID);
      childParentSlot[slot] = i32(parentSlot);
      // BUGFIX (L2 bounce-back sign/magnitude investigation): the
      // "Milestone 10 BUGFIX" that used to sit here had it backwards --
      // see parentCenterX_L0's own BUGFIX comment above (same file, same
      // root confusion): the parent's own interior is 2*RB cells (not
      // RB -- amr_criterion_pool.wgsl's header), so RB is ALREADY half
      // the parent's own physical width (RB*PARENT_CELL_SIZE_L0 out of a
      // full 2*RB*PARENT_CELL_SIZE_L0), and that IS the correct quadrant
      // step -- no further *0.5f belongs here, same as parentCenterX_L0
      // needed none. The removed *0.5f halved every qx=1/qy=1 child's
      // offset from its parent's origin, so the 4 children of a quad no
      // longer tiled the parent's footprint 2x2 with no gap/overlap:
      // quadrant 1 sat overlapping half of quadrant 0's true territory
      // and left the outer half of the parent's footprint uncovered by
      // any tile at all (masked-off at the parent level too, since
      // masking only checks quadrant 0's existence, not its registered
      // position) -- exactly the kind of corruption that would produce
      // a wrong-sign, wrong-magnitude level>=2 bounce-back force while
      // leaving 2:1-balance (an index-only check) and the field-finite
      // check clean. Live-verified via a per-slot force readback
      // (amr_force1_pool.wgsl's debugSlotForce / main-cylinder-amr.js's
      // debugReadSlotForces) correlating each level-2 slot's own (fx,fy)
      // against its geometric position -- restoring this formula to
      // match the ORIGINAL (pre-Milestone-10) version fixes it.
      childNewlyActivated[slot] = 1u;
    }
  }
}

@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (DET_SLOTS != 0u) {
    // One thread, dispatch order, no atomics in the decision. See DET_SLOTS.
    if (gid.x != 0u) { return; }
    var count = atomicLoad(&childFreeCount);
    let nParentSlots = arrayLength(&parentSlotToBlock);
    for (var ps = 0u; ps < nParentSlots; ps++) {
      if (!refineWants(ps)) { continue; }
      if (count <= 0) { // pool exhausted this round -- stay coarse
        if (DIAG != 0u) { atomicAdd(&diag[5], 1u); }
        continue;
      }
      count = count - 1;
      grantQuad(ps, childFreeList[u32(count)]);
    }
    atomicStore(&childFreeCount, count);
    return;
  }

  let parentSlot = gid.x;
  if (!refineWants(parentSlot)) { return; }

  let oldCount = atomicSub(&childFreeCount, 1);
  if (oldCount > 0) {
    grantQuad(parentSlot, childFreeList[u32(oldCount - 1)]);
  } else {
    atomicAdd(&childFreeCount, 1); // pool exhausted this round -- undo, stay coarse
    if (DIAG != 0u) { atomicAdd(&diag[5], 1u); }
  }
}

// Release one quad's four slots. Lifted out of coarsen() so the serial path can
// reuse it.
fn releaseQuadSlots(quadIdx: u32) {
  for (var q = 0u; q < 4u; q++) {
    let s = quadIdx * 4u + q;
    let bID = childSlotToBlock[s];
    if (bID >= 0) { childBlockSlot[u32(bID)] = -1; }
    childSlotToBlock[s] = -1;
  }
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (DET_SLOTS != 0u) {
    // One thread, slot order, no atomics. See DET_SLOTS.
    if (gid.x != 0u) { return; }
    var count = atomicLoad(&childFreeCount);
    let nSlots = arrayLength(&childSlotToBlock);
    for (var s = 0u; s < nSlots; s += 4u) {
      let bID = childSlotToBlock[s];
      if (bID < 0) { continue; }
      if (childWant[u32(bID)] != 0u) { continue; }
      childFreeList[u32(count)] = i32(s / 4u);
      count = count + 1;
      releaseQuadSlots(s / 4u);
    }
    atomicStore(&childFreeCount, count);
    return;
  }

  let slot = gid.x;
  if (slot >= arrayLength(&childSlotToBlock)) { return; }
  let blockID = childSlotToBlock[slot];
  if (blockID < 0) { return; }
  // Only quadrant 0 drives the decision -- all 4 release together
  // (decision 3), so evaluating (and freeing) once per quad, not once per
  // slot, avoids 4 threads racing the same freeList push.
  // One thread per QUAD, and quadrant 0's slot is the one divisible by 4 --
  // see the binding-8 note above on why this is arithmetic rather than a read.
  if (slot % 4u != 0u) { return; }

  // NO CRITERION AND NO GEOMETRY HERE EITHER, and for the same reason as
  // refine() above -- this computed the child's eps, walked up to the parent
  // slot, and derived the parent's centre, none of which anything below reads.
  // Dead since B2-2d; removed in B3-7.
  //
  // What that code was FOR is worth keeping, because it is the sharpest
  // argument in this file for why the closure replaced it. refine() and
  // coarsen() each ran their own geometric test, and for a while they ran it
  // at DIFFERENT POINTS -- refine() at the parent's centre, coarsen() at
  // quadrant 0's. Live-verified: a parent at phi=1.98 (well inside a 4-unit
  // margin) whose own quadrant-0 centre measured phi=4.56 (just outside it).
  // refine() recreated the quad every fixed-point iteration and coarsen()
  // released it again immediately, a create/destroy oscillation on EVERY
  // evaluation rather than a rare edge case -- and that starved the L0->L1
  // cascade, which checked hasLevel2Child, of ever seeing the child exist
  // (false at 20 of 20 consecutive evaluations, from a GPU-side capture).
  // It was the mechanism behind a debugCheck21Balance violation that
  // reproduced identically across thousands of steps.
  //
  // TWO FUNCTIONS DECIDING THE SAME THING SEPARATELY IS THE BUG CLASS. Now
  // they both read one want array that one pass wrote.

  // Want-only, for the same reason refine() is: the closure guarantees a
  // wanted grandchild implies a wanted child, so a tile still needed as a
  // parent is still wanted. The HAS_GRANDCHILD walk that used to guard this
  // -- each of the 4 releasing children plus each of their own edge
  // neighbours -- is gone with it.
  if (childWant[u32(childSlotToBlock[slot])] == 0u) {
    let quadIdx = slot / 4u; // slot IS quadrant 0's own slot (slot % 4 == 0 checked above), so quadIdx*4u==slot
    let oldCount = atomicAdd(&childFreeCount, 1);
    childFreeList[u32(oldCount)] = i32(quadIdx);
    releaseQuadSlots(quadIdx);
  }
}
