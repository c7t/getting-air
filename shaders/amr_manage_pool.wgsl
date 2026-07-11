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
//   same-level (level-m) edge-neighbors are ALSO active. If not, this
//   quad's refine is blocked THIS round -- amr_manage.wgsl's own cascade
//   check (or, recursively, a shallower amr_manage_pool.wgsl instance)
//   is what forces those missing neighbors active, in the SAME
//   coarsest-to-finest sweep, so a later same-round refine (of a level
//   closer to the root) unblocks this one before it's re-evaluated.
// - coarsen(): SCOPE LIMIT, flagged explicitly rather than silently
//   incomplete -- a fully general implementation would also need to check
//   (a) does this quad itself have active level-(m+2) children, and
//   (b) does a same-level neighbor have active level-(m+2) children,
//   both requiring a THIRD level's blockSlot data. Neither check is
//   implemented here because at this plan's validated depth (N=3), this
//   shader's only real instantiation (parent=level 1, deciding level 2)
//   has level 2 as the DEEPEST configured level -- there is no level 3
//   for either condition to ever be true, so omitting them is exact, not
//   approximate, for N<=3. Revisit before ever running this at N>=4.

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

override W : u32;
override H : u32;
override RB : u32;
override NBX_PARENT : u32;
override NBY_PARENT : u32;
override PARENT_CELL_SIZE_L0 : f32;
override PARENT_HAS_CACHED_ORIGIN : u32;
override REFINE_THRESH : f32;
override COARSEN_THRESH : f32;
override FORCE_REFINE_MARGIN : f32;
override FORCE_REFINE_LOOKAHEAD : f32;
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
  // BUGFIX (Milestone 10): center = origin + HALF the block's own physical
  // width, not the full width -- compare amr_manage.wgsl's own isNearBody,
  // which correctly uses `bx * BLOCK + BLOCK / 2u`. This omitted-*0.5 bug
  // shifted every geometry-forced-refinement candidate's position estimate
  // by a full block-width toward (+X,+Y), biasing which blocks isNearBodyAt
  // considers "near" and (independently of the childOriginX/Y quadrant-
  // offset bug fixed above) contributing to level 2's refined region coming
  // out far smaller/off-center than the vorticity-only criterion alone
  // would predict.
  let parentCenterX_L0 = parentOriginX_L0 + f32(RB) * PARENT_CELL_SIZE_L0 * 0.5f;
  let parentCenterY_L0 = parentOriginY_L0 + f32(RB) * PARENT_CELL_SIZE_L0 * 0.5f;

  let wantsRefine = eps >= REFINE_THRESH || isNearBodyAt(parentCenterX_L0, parentCenterY_L0);
  if (!wantsRefine) { return; }

  // 2:1 balance (see header): all 4 same-level (level-m) edge-neighbors of
  // the PARENT must already be active, or this refine is blocked this
  // round -- whichever shader manages the parent's own level is
  // responsible for cascading them active (reading THIS shader's own
  // childCriterion to detect the demand -- see amr_manage.wgsl's header).
  let neighborN = parentBlockSlot[((byP + NBY_PARENT - 1u) % NBY_PARENT) * NBX_PARENT + bxP];
  let neighborS = parentBlockSlot[((byP + 1u) % NBY_PARENT) * NBX_PARENT + bxP];
  let neighborE = parentBlockSlot[byP * NBX_PARENT + ((bxP + 1u) % NBX_PARENT)];
  let neighborW = parentBlockSlot[byP * NBX_PARENT + ((bxP + NBX_PARENT - 1u) % NBX_PARENT)];
  if (neighborN < 0 || neighborS < 0 || neighborE < 0 || neighborW < 0) { return; }

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
  // BUGFIX (Milestone 10): same missing-*0.5 center-vs-origin bug as
  // refine()'s parentCenterX_L0 above -- (PARENT_CELL_SIZE_L0*0.5f) is
  // already this CHILD level's own cell size (dx_child = dx_parent/2), so
  // `f32(RB) * dx_child` is the child block's FULL width; its center needs
  // one more *0.5.
  let centerX_L0 = childOriginX[slot] + f32(RB) * (PARENT_CELL_SIZE_L0 * 0.5f) * 0.5f;
  let centerY_L0 = childOriginY[slot] + f32(RB) * (PARENT_CELL_SIZE_L0 * 0.5f) * 0.5f;

  if (eps < COARSEN_THRESH && !isNearBodyAt(centerX_L0, centerY_L0)) {
    // See header's coarsen scope-limit note: no check here for "do I (or a
    // same-level neighbor) have active level-(m+2) children" -- exact, not
    // approximate, as long as this level is the deepest configured one.
    let quadIdx = slot / 4u;
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
