// Milestone 9 (plans/AMR-multilevel.md): refine/coarsen decision + quad
// allocation for any L(m)->L(m+1) hop with m>=1 -- sibling of
// amr_manage.wgsl (which stays the L0->L1 decision, dense-parent-specific,
// but gains a matching cascade/coarsen-block check against THIS file's own
// child level -- see that file's header).
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
// level's own cached originX/Y buffers, like amr_step1_pool.wgsl). When 0,
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
//   balance after a hard-required deep refine is amr_manage.wgsl's own
//   cascade (or, recursively, a shallower amr_manage_pool.wgsl
//   instance) -- see that file's header, now also existence- (not
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
//     even if its own criterion doesn't call for it, if a same-level-m
//     EDGE-NEIGHBOR already has an ACTIVE level-(m+1) child that itself
//     already HAS an active level-(m+2) grandchild. EXISTENCE
//     (hasGrandchild), not desire -- an earlier version of this fix used a
//     criterion-based "does the neighbor's child WANT a grandchild" test
//     (mirroring amr_manage.wgsl's own level2WantsRefine one level down)
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

struct CardState {
  cx     : f32,
  cy     : f32,
  theta  : f32,
  vx     : f32,
  vy     : f32,
  omega  : f32,
  fx     : f32,
  fy     : f32,
  tz     : f32,
  mass   : f32,
  i_body : f32,
  g_eff  : f32,
  a      : f32,
  b      : f32,
  v_max  : f32,
  o_max  : f32,
  cx_old : f32,
  cy_old : f32,
  th_old : f32,
  tau    : f32,
  y_total: f32,
  x_total: f32,
  off_x  : f32,
  off_y  : f32,
  off_x_old : f32,
  off_y_old : f32,
}

@group(0) @binding(0)  var<storage, read>       childCriterion    : array<f32>;
@group(0) @binding(1)  var<storage, read_write> childBlockSlot    : array<i32>;
@group(0) @binding(2)  var<storage, read_write> childSlotToBlock  : array<i32>;
@group(0) @binding(3)  var<storage, read_write> childFreeList     : array<i32>; // quad-indexed
@group(0) @binding(4)  var<storage, read_write> childFreeCount    : atomic<i32>; // quad-indexed
@group(0) @binding(5)  var<storage, read_write> childNewlyActivated : array<u32>;
@group(0) @binding(6)  var<storage, read>       state             : CardState;
@group(0) @binding(7)  var<storage, read_write> childParentSlot   : array<i32>;
@group(0) @binding(8)  var<storage, read_write> childQuadrant     : array<u32>;
@group(0) @binding(9)  var<storage, read_write> childOriginX      : array<f32>;
@group(0) @binding(10) var<storage, read_write> childOriginY      : array<f32>;
@group(0) @binding(11) var<storage, read>       parentBlockSlot   : array<i32>;
@group(0) @binding(12) var<storage, read>       parentSlotToBlock : array<i32>;
@group(0) @binding(13) var<storage, read>       parentOriginX     : array<f32>; // dummy if !PARENT_HAS_CACHED_ORIGIN
@group(0) @binding(14) var<storage, read>       parentOriginY     : array<f32>; // dummy if !PARENT_HAS_CACHED_ORIGIN
// Grandchild (level m+2) blockSlot, for the 2:1-balance cascade/coarsen-
// block checks -- dummy if !HAS_GRANDCHILD, see header. Both refine() and
// coarsen() only ever need EXISTENCE (hasGrandchild), never level (m+2)'s
// criterion -- see hasGrandchild's own comment on why "wants" isn't the
// right test for maintaining balance against an already-active grandchild.
@group(0) @binding(15) var<storage, read>       grandchildBlockSlot : array<i32>;

override W : u32;
override H : u32;
override RB : u32;
override NBX_PARENT : u32;
override NBY_PARENT : u32;
override PARENT_CELL_SIZE_L0 : f32;
override PARENT_HAS_CACHED_ORIGIN : u32;
override HAS_GRANDCHILD : u32 = 0u;
override REFINE_THRESH : f32;
override COARSEN_THRESH : f32;
override FORCE_REFINE_MARGIN : f32;
override FORCE_REFINE_LOOKAHEAD : f32;
// L0 window-space edge band excluded from vorticity-driven refinement (same
// fixed L0-window strip as amr_manage.wgsl -- unscaled per level, since the
// sponge is a fixed L0 strip). Gated off when <= 0.
override SPONGE_EXCLUDE_W : f32 = 0.0f;
const EPS_FLOOR = 1e-6f;

fn get_phi(p: vec2<f32>, s: CardState) -> f32 {
  let ca = cos(s.theta);
  let sa = sin(s.theta);
  var dx = p.x - s.cx;
  var dy = p.y - s.cy;
  dx -= f32(W) * round(dx / f32(W));
  dy -= f32(H) * round(dy / f32(H));
  let lx = dx * ca + dy * sa;
  let ly = -dx * sa + dy * ca;
  let d = sqrt((lx*lx)/(s.a*s.a) + (ly*ly)/(s.b*s.b)) - 1.0;
  return d * s.b;
}

// Same now/future test as amr_manage.wgsl's isNearBody, parametrized by an
// already-computed L0-buffer-space center instead of deriving it from a
// dense blockID -- see that file's header for the extrapolation rationale
// (test point moves backward at -v, not the ellipse forward, since the
// moving window absorbs bulk translation into off_x/off_y).
fn isNearBodyAt(centerX_L0: f32, centerY_L0: f32) -> bool {
  let wx = (u32(centerX_L0) + W - u32(state.off_x)) % W;
  let wy = (u32(centerY_L0) + H - u32(state.off_y)) % H;
  let p_now = vec2<f32>(f32(wx), f32(wy));

  let phi_now = get_phi(p_now, state);

  let p_future = p_now - vec2<f32>(state.vx, state.vy) * FORCE_REFINE_LOOKAHEAD;
  var future = state;
  future.theta += state.omega * FORCE_REFINE_LOOKAHEAD;
  let phi_future = get_phi(p_future, future);

  return min(phi_now, phi_future) < FORCE_REFINE_MARGIN;
}

// True if the given L0-buffer-space center lies within SPONGE_EXCLUDE_W of any
// window edge (the ALBC sponge band). Mirrors amr_manage.wgsl's inSpongeBand,
// parametrized by an already-computed L0 center like isNearBodyAt above.
fn inSpongeBandAt(centerX_L0: f32, centerY_L0: f32) -> bool {
  if (SPONGE_EXCLUDE_W <= 0.0f) { return false; }
  let wx = (u32(centerX_L0) + W - u32(state.off_x)) % W;
  let wy = (u32(centerY_L0) + H - u32(state.off_y)) % H;
  let distX = min(f32(wx), f32(W - wx));
  let distY = min(f32(wy), f32(H - wy));
  return min(distX, distY) < SPONGE_EXCLUDE_W;
}

// True if the level-(m+1) tile at `childBlockID` currently has an active
// level-(m+2) child (quadrant 0 stands for all 4 -- decision 3's all-or-
// nothing invariant, same as hasLevel2Child in amr_manage.wgsl).
fn hasGrandchild(childBlockID: u32) -> bool {
  let nbxChild = NBX_PARENT * 2u;
  let bx = childBlockID % nbxChild; let by = childBlockID / nbxChild;
  let nbxGrandchild = nbxChild * 2u;
  let gcBlockID0 = (by * 2u) * nbxGrandchild + (bx * 2u);
  return grandchildBlockSlot[gcBlockID0] >= 0;
}

// Level-(m+1)'s own 4 edge-neighbor blockIDs (level-(m+1) coordinate
// space, NBX_PARENT*2 wide) -- same shape as amr_manage.wgsl's
// edgeNeighbors, just at the child level instead of the dense L0/L1 one.
fn childEdgeNeighbors(childBlockID: u32) -> array<u32, 4> {
  let nbxChild = NBX_PARENT * 2u; let nbyChild = NBY_PARENT * 2u;
  let bx = childBlockID % nbxChild; let by = childBlockID / nbxChild;
  return array<u32, 4>(
    ((by + nbyChild - 1u) % nbyChild) * nbxChild + bx,
    ((by + 1u) % nbyChild) * nbxChild + bx,
    by * nbxChild + ((bx + 1u) % nbxChild),
    by * nbxChild + ((bx + nbxChild - 1u) % nbxChild),
  );
}

@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let parentSlot = gid.x;
  if (parentSlot >= arrayLength(&parentSlotToBlock)) { return; }
  let parentBlockID = parentSlotToBlock[parentSlot];
  if (parentBlockID < 0) { return; } // parent not active -- not a candidate at all

  let bxP = u32(parentBlockID) % NBX_PARENT;
  let byP = u32(parentBlockID) / NBX_PARENT;
  let nbxChild = NBX_PARENT * 2u;

  // Already refined? Quadrant 0 stands for all 4 (decision 3's all-or-
  // nothing invariant).
  let childBlockID0 = (byP * 2u) * nbxChild + (bxP * 2u);
  if (childBlockSlot[childBlockID0] >= 0) { return; }

  // Own criterion: max over the 4 prospective quadrants.
  var maxCrit = 0f;
  for (var qy = 0u; qy < 2u; qy++) {
    for (var qx = 0u; qx < 2u; qx++) {
      let cb = (byP * 2u + qy) * nbxChild + (bxP * 2u + qx);
      maxCrit = max(maxCrit, childCriterion[cb]);
    }
  }
  let eps = min(1.0f, log2(max(maxCrit, EPS_FLOOR)));

  // Parent's own physical origin -- see header for the dense-vs-cached split.
  var parentOriginX_L0 = f32(bxP * RB);
  var parentOriginY_L0 = f32(byP * RB);
  if (PARENT_HAS_CACHED_ORIGIN != 0u) {
    parentOriginX_L0 = parentOriginX[parentSlot];
    parentOriginY_L0 = parentOriginY[parentSlot];
  }
  // BUGFIX: center = origin + HALF the block's own physical width. The
  // parent's own interior is 2*RB cells (not RB -- see amr_criterion_pool.
  // wgsl's own header: "a parent slot's own interior is 2*RB x 2*RB
  // cells"), each PARENT_CELL_SIZE_L0 wide, so the full physical width is
  // 2*RB*PARENT_CELL_SIZE_L0 and the HALF-width is RB*PARENT_CELL_SIZE_L0
  // -- RB is already "half the interior" by construction, so no further
  // *0.5 belongs here. The previous "Milestone 10 BUGFIX" comment at this
  // exact spot claimed to match amr_manage.wgsl's own (correct, already-
  // validated) `bx*BLOCK+BLOCK/2u` convention but actually computed HALF
  // of that (RB=BLOCK=8, PARENT_CELL_SIZE_L0=0.5 at m=1 numerically gives
  // RB*dx*0.5=2, not BLOCK/2=4) -- a real, live-verified bug: candidates
  // well within the geometric force-refine margin (phi as low as ~2
  // against a ~4-unit margin) were STILL not getting their required
  // level-(m+1) child, because THIS shader's own idea of "near the body"
  // was evaluated 2 L0-units off from where amr_manage.wgsl (and every
  // other shader's own chi/phi position, which all use the correct,
  // unscaled-by-an-extra-0.5 physical center) actually place it.
  let parentCenterX_L0 = parentOriginX_L0 + f32(RB) * PARENT_CELL_SIZE_L0;
  let parentCenterY_L0 = parentOriginY_L0 + f32(RB) * PARENT_CELL_SIZE_L0;

  // Grandchild cascade (see header): even if THIS parent's own criterion
  // doesn't call for a level-(m+1) child, force one anyway if a same-
  // level-m edge-neighbor ALREADY HAS an active level-(m+1) child that
  // itself already has an active level-(m+2) grandchild -- otherwise that
  // grandchild sits directly adjacent to this still-level-m region, a
  // 2-level gap. Existence (hasGrandchild), not desire: a "would level-
  // (m+2) want to refine here right now" test can flicker false on a round
  // where the ALREADY-ACTIVE grandchild's own criterion has since dropped
  // (flow evolved, criterion re-evaluated fresh each round) even though
  // coarsen() hasn't released it yet -- using "wants" here let a genuinely
  // still-active depth-(m+2) region go uncascaded for however many rounds
  // its own criterion stayed fresh-per-round negative, a live-verified bug
  // in an earlier version of this fix (debugCheck21Balance caught real
  // depth-1-vs-depth-3 violations with the "wants" version).
  var cascadeWanted = false;
  if (HAS_GRANDCHILD != 0u) {
    let nbrBX = array<u32, 4>(bxP, bxP, (bxP + 1u) % NBX_PARENT, (bxP + NBX_PARENT - 1u) % NBX_PARENT);
    let nbrBY = array<u32, 4>((byP + NBY_PARENT - 1u) % NBY_PARENT, (byP + 1u) % NBY_PARENT, byP, byP);
    for (var i = 0u; i < 4u; i++) {
      let nbxN = nbrBX[i]; let nbyN = nbrBY[i];
      let childBlockIDN = (nbyN * 2u) * nbxChild + (nbxN * 2u);
      if (childBlockSlot[childBlockIDN] >= 0 && hasGrandchild(childBlockIDN)) {
        cascadeWanted = true;
      }
    }
  }

  // HARD constraints, never blocked by the 2:1-balance neighbor gate below
  // -- geometry (this candidate's own surface proximity) and cascade (a
  // same-level neighbor ALREADY has a 2-level-deeper descendant) are both
  // mandatory: the body's surface must reach the finest configured level
  // regardless of a same-level neighbor's current activity, and 2:1
  // balance is then DRIVEN from that requirement outward (neighbors get
  // pulled up to satisfy it -- see amr_manage.wgsl's own cascade, now
  // existence- not criterion-based, for the mechanism one level further
  // out) rather than used to VETO the requirement itself. Only a
  // criterion (vorticity)-only refine -- no geometric or cascade reason,
  // just "this cell's own flow looks interesting" -- keeps the
  // conservative gate, so criterion-driven growth alone still can't
  // outrun its own coarser neighborhood.
  let isHardRequired = isNearBodyAt(parentCenterX_L0, parentCenterY_L0) || cascadeWanted;
  let wantsRefine = isHardRequired || (eps >= REFINE_THRESH && !inSpongeBandAt(parentCenterX_L0, parentCenterY_L0));
  if (!wantsRefine) { return; }

  if (!isHardRequired) {
    // 2:1 balance (see header): all 4 same-level (level-m) edge-neighbors
    // of the PARENT must already be active, or this refine is blocked
    // this round -- whichever shader manages the parent's own level is
    // responsible for cascading them active (reading THIS shader's own
    // childCriterion to detect the demand -- see amr_manage.wgsl's header).
    let neighborN = parentBlockSlot[((byP + NBY_PARENT - 1u) % NBY_PARENT) * NBX_PARENT + bxP];
    let neighborS = parentBlockSlot[((byP + 1u) % NBY_PARENT) * NBX_PARENT + bxP];
    let neighborE = parentBlockSlot[byP * NBX_PARENT + ((bxP + 1u) % NBX_PARENT)];
    let neighborW = parentBlockSlot[byP * NBX_PARENT + ((bxP + NBX_PARENT - 1u) % NBX_PARENT)];
    if (neighborN < 0 || neighborS < 0 || neighborE < 0 || neighborW < 0) { return; }
  }

  let oldCount = atomicSub(&childFreeCount, 1);
  if (oldCount > 0) {
    let quadIdx = childFreeList[u32(oldCount - 1)];
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
        childQuadrant[slot] = quadrant;
        // BUGFIX (Milestone 10): a quadrant step must offset by ONE
        // QUADRANT-WIDTH, not one full PARENT-block-width. The parent's own
        // interior is RB cells at PARENT_CELL_SIZE_L0 each (physical width
        // RB*PARENT_CELL_SIZE_L0); the quad's 4 children tile that SAME
        // footprint 2x2, so each quadrant only spans HALF of it per axis --
        // omitting the *0.5f here (as this did before) placed qx=1/qy=1
        // children a full parent-block-width away from the parent's origin
        // (double the correct offset), silently mis-registering every
        // auto-refined level>=2 tile's physical position against the body
        // geometry it exists to resolve. Caught because it made N=3
        // integrated force ~17x too small (most cells' chi evaluated far
        // outside the true epsilon band), not from any topology/2:1-balance
        // check (those only look at bx/by indices, which this bug never
        // touched -- see main-cylinder-amr.js's debugActivateBlock, which
        // had the identical bug in its own JS-side mirror of this formula).
        childOriginX[slot] = parentOriginX_L0 + f32(qx) * f32(RB) * PARENT_CELL_SIZE_L0 * 0.5f;
        childOriginY[slot] = parentOriginY_L0 + f32(qy) * f32(RB) * PARENT_CELL_SIZE_L0 * 0.5f;
        childNewlyActivated[slot] = 1u;
      }
    }
  } else {
    atomicAdd(&childFreeCount, 1); // pool exhausted this round -- undo, stay coarse
  }
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= arrayLength(&childSlotToBlock)) { return; }
  let blockID = childSlotToBlock[slot];
  if (blockID < 0) { return; }
  // Only quadrant 0 drives the decision -- all 4 release together
  // (decision 3), so evaluating (and freeing) once per quad, not once per
  // slot, avoids 4 threads racing the same freeList push.
  if (childQuadrant[slot] != 0u) { return; }

  let eps = min(1.0f, log2(max(childCriterion[u32(blockID)], EPS_FLOOR)));
  // This quad's own center (quadrant 0's cached origin is this quad's own
  // origin -- see header) -- same generous-margin isNearBody test as refine.
  // BUGFIX: see refine()'s parentCenterX_L0 for the full derivation -- this
  // level's own interior is 2*RB cells (not RB) at this level's own dx
  // (PARENT_CELL_SIZE_L0*0.5f, i.e. dx_child=dx_parent/2), so the HALF-
  // width is RB*dx_child directly, no further *0.5.
  let centerX_L0 = childOriginX[slot] + f32(RB) * (PARENT_CELL_SIZE_L0 * 0.5f);
  let centerY_L0 = childOriginY[slot] + f32(RB) * (PARENT_CELL_SIZE_L0 * 0.5f);

  if ((eps < COARSEN_THRESH || inSpongeBandAt(centerX_L0, centerY_L0)) && !isNearBodyAt(centerX_L0, centerY_L0)) {
    let quadIdx = slot / 4u; // slot IS quadrant 0's own slot (childQuadrant[slot]==0 checked above), so quadIdx*4u==slot
    // See header: blocked if any of the 4 children about to release has
    // an active level-(m+2) grandchild itself, or if any of their own
    // same-level-(m+1) edge-neighbors does -- releasing either would leave
    // that grandchild directly adjacent to this now-coarser region.
    if (HAS_GRANDCHILD != 0u) {
      for (var q = 0u; q < 4u; q++) {
        let s = quadIdx * 4u + q;
        let bID = childSlotToBlock[s];
        if (bID < 0) { continue; }
        if (hasGrandchild(u32(bID))) { return; }
        let nbrs = childEdgeNeighbors(u32(bID));
        for (var i = 0u; i < 4u; i++) {
          if (childBlockSlot[nbrs[i]] >= 0 && hasGrandchild(nbrs[i])) { return; }
        }
      }
    }
    let oldCount = atomicAdd(&childFreeCount, 1);
    childFreeList[u32(oldCount)] = i32(quadIdx);
    for (var q = 0u; q < 4u; q++) {
      let s = quadIdx * 4u + q;
      let bID = childSlotToBlock[s];
      if (bID >= 0) { childBlockSlot[u32(bID)] = -1; }
      childSlotToBlock[s] = -1;
    }
  }
}
