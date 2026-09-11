// Dynamic refinement: the refine/coarsen decision and the free-list
// allocation that serves it. plans/3D.md M4.2b-i, and the 3D sibling of the
// 2D shaders/amr_manage.wgsl. Fragment only; the entry files list every
// include.
//
// COARSEN AND REFINE ARE TWO SEPARATE COMPUTE PASSES, coarsen fully
// completing before refine starts. That is not tidiness, it is a correctness
// requirement, and the 2D file records finding it the hard way in a live
// run: atomics guarantee that the free COUNTER is serialized across threads,
// and say NOTHING about when one thread's plain (non-atomic) write to
// freeList becomes visible to another thread's plain read of the same array
// within the SAME dispatch. A coarsening block's push racing a refining
// block's pop produced two different coarse blocks whose blockSlot pointed
// at one pool slot -- caught only by cross-checking blockSlot against its
// inverse slotToBlock, which disagreed for exactly the colliding slot.
//
// Splitting into ordered passes removes the hazard rather than managing it:
// within `coarsen` only writes happen, each thread at its own unique index;
// within `refine` only reads happen, of data written by a prior and now
// complete pass. Neither phase has two threads touching one freeList entry.
//
// THE CRITERION IS GEOMETRY-FORCED and nothing else, deliberately. A block
// is wanted at the fine level iff its coarse-cell box comes within MARGIN of
// the body. That is the same predicate d3-amr.mjs's refineNearBody applies
// on the host to build the STATIC set -- same eight corners, same centre,
// same `min <= margin` -- because M4.2b-i's gate is that a pinned body gives
// a BIT-IDENTICAL run with the manager switched on. A criterion that
// disagreed with the host's by one block would break that gate and there
// would be no way to tell it from a real defect in the allocator.
//
// Vorticity- or error-driven refinement is deliberately NOT here. It changes
// which blocks are wanted from step to step, which is exactly what the
// bit-identical gate needs to not happen yet.
//
// 2:1 BALANCE IS A CLOSURE ON THE WANT SET, and it therefore belongs between
// `decide` and the drain rather than as a test inside coarsen and refine:
// "refine forced by a neighbour that wants a deeper child" and "coarsen
// blocked by a neighbour that has one" are one rule read in two directions,
// and by the time those two passes run the answer is already balanced. The
// rule is d3-amr.mjs's `cascade21`, gated by tools/test-d3-amr.js against
// check21Balance -- a checker written FIRST, in M4.2a, deliberately before
// there was a manager to be tempted to agree with. `completeOctets` and
// `balance` below are its GPU mirror (M5.5a), scored against it on real GPU
// data by tools/validate-d3-invariants.js.
//
// At ?levels=2 that closure is the IDENTITY -- a level-1 block's parent
// level is the dense L0 grid, which is present everywhere -- so neither pass
// is created there and neither could write anything back if it were.
//
// REFINEMENT IS OCTET-COMPLETE FROM LEVEL 2 DOWN. A parent spawns all eight
// children or none, exactly as amr_manage_pool.wgsl spawns a whole quad.
// Nothing in `refine` enforces that, and nothing needs to: `completeOctets`
// has already made the WANT set octet-complete, so acting on it per block
// allocates and frees whole octets. cascade21's header records what a
// closure that forgets this does to check21Balance.
//
// WHAT THIS FILE DOES NOT DO: a slot handed out here is not initialized, and
// a slot released here is not restricted into its parent. Those are the FILL
// and DRAIN passes -- interp with NEW_ONLY and average with DYING_ONLY --
// and the order the host encodes them in is the design. See main-3d.js's
// manager stanza, and plans/3D.md M4.2b-ii / M5.5b.

@group(0) @binding(0) var<storage, read_write> blockSlot   : array<i32>;
@group(0) @binding(1) var<storage, read_write> slotToBlock : array<i32>;
@group(0) @binding(2) var<storage, read_write> freeList    : array<i32>;
// [0] = how many slots are free. [1] = HOW MANY TIMES A REFINE WAS REFUSED
// FOR WANT OF A SLOT, and it is sticky: the host treats any nonzero value as
// a hard failure and stops the run. See `refine` for why that is not an
// over-reaction. [2] = the MINIMUM free count ever reached, i.e. the exact
// high-water mark of slot usage -- recorded by `refine` because a host that
// samples the count can only report the peaks it happened to look at.
@group(0) @binding(3) var<storage, read_write> freeCount   : array<atomic<i32>, 4>;
@group(0) @binding(4) var<storage, read>       body        : BodyState3D;
// The criterion's answer, computed ONCE by `decide` and read by every pass
// after it. M4.2b-i evaluated blockWanted() separately in coarsen and in
// refine, which is two chances to disagree about one question -- and once
// the drain and fill passes also need the answer it becomes four. One
// producer, three consumers.
@group(0) @binding(5) var<storage, read_write> blockWant   : array<u32>;
// Per SLOT: set by `refine` when it hands the slot out, cleared by the fill
// pass once the tile has been initialized from the coarse field. A slot with
// this set is holding whatever the last owner left behind.
@group(0) @binding(6) var<storage, read_write> slotNew     : array<u32>;
// The CHILD level's want array, read by `balance` (plans/3D.md M5.5). Bound
// always so the layout does not fork; read only by that entry point, which is
// created only where there is a child level.
@group(0) @binding(7) var<storage, read> childWant : array<u32>;
// PER-BLOCK MAX Q from common_d3_criterion.wgsl (plans/3D.md M8.4), indexed
// on the FINEST level's block grid -- the grid `decide` runs on. Bound always,
// like childWant, so the layout does not fork; read only where HAS_Q is set.
@group(0) @binding(8) var<storage, read> blockQ : array<f32>;

// THE FIELD CRITERION, off by default so every scenario that predates M8.4 is
// bit-identical rather than merely unaffected.
//
// IT IS A LADDER, NOT A THRESHOLD, and that is not a refinement of the idea
// -- it is the idea. A single yes/no per block answers "is there a vortex
// here", refines it to the FINEST level, and so decides depth by proximity
// to the body rather than by what the flow is doing. 2D MEASURED that exact
// failure before replacing it (common_refine.wgsl's header): at N=3, 118 of
// 132 level-2 blocks were inside the geometry halo and the other 14 were the
// 2:1 cascade shell -- "level 2 never reached the wake at all, however
// energetic the wake was".
//
// So this includes 2D's `common_refine.wgsl` verbatim rather than restating
// it. That file is pure scalar arithmetic over four overrides with nothing
// two-dimensional in it, and one statement of a ladder is worth more than a
// second copy that agrees today.
//
// THE UNITS. The criterion kernel reports max Q per block, and Q ~ (du/dx)^2
// -- so sqrt(Q) is a ROTATION RATE, directly comparable to the |omega| the
// 2D ladder was calibrated against, and epsPhys = log2(sqrt(Q)) puts the two
// on one scale. The `+ m` physical-units shift common_refine.wgsl documents
// is ZERO here because the field is read at L0; it becomes 2*m (not m) the
// day this is evaluated per level, since Q carries the gradient squared.
override HAS_Q : u32 = 0u;
// Which level THIS dispatch decides for. The ladder answers "how deep does
// this region want to be", so each level asks whether that answer reaches it.
override LEVEL_M : i32 = 1;
// Level-1 block counts, for mapping a level-m block onto the criterion grid.
override CNBX : u32 = 1u;
override CNBY : u32 = 1u;
override CNBZ : u32 = 1u;
const Q_FLOOR = 1e-30f;
// THE LADDER'S OWN OVERRIDES, which common_refine.wgsl requires its includer
// to declare -- the same arrangement common_geometry.wgsl uses for W/H.
//
// REFINE_THRESH is DERIVED BY THE HOST from `?qthresh=`, not typed here:
// qthresh is dimensionless against the body's own shear scale and its default
// was measured (d3-criterion.mjs), so the ladder's base is
// 0.5*log2(qthresh * qRef) and keeps that measurement rather than replacing
// it with a second number calibrated by eye. N_REFINE_INC is the only new
// free parameter: how many octaves of rotation rate each further level costs.
override REFINE_THRESH : f32 = -1e30f;
override COARSEN_THRESH : f32 = -1e30f;
override N_REFINE_INC : f32 = 1.0f;
override N_REFINE_MAX : f32 = 1.0f;
override MAX_LEVEL : i32 = 1;

override MARGIN : f32 = 2.0f;
// How many macro-steps pass before the criterion is re-evaluated. The
// manager refines AHEAD by however far the body can travel in that time --
// see blockWanted. 1 makes the lead term vanish.
override MANAGE_EVERY : f32 = 1.0f;
// The body's own radius is baked into get_phi3, so the criterion needs
// nothing else about the shape.
override HAS_BODY : u32 = 0u;
// A level-m manager works in level-(m-1) CELL UNITS and the body lives in L0
// units. 2^-(m-1), identity at level 1.
//
// A PURE SCALE, NOT THE AFFINE MAP the step and force kernels use (M5.4b),
// AND THAT IS DELIBERATE. Those convert a cell CENTRE and must land on it
// exactly. This converts a BLOCK BOX used as a conservative bound, and the
// only thing that must be exact about it is that it AGREES with
// d3-amr.mjs's refineHierarchy, which builds the initial set with the same
// pure scale -- M4.2b-i's gate is that the kernel's criterion and the host's
// produce the same set, and two nearly-right conversions that differ would
// break it. The difference is the cell-centring offset, at most ~0.4 L0
// cells at depth 3 against a margin of 2, and the real guarantee is the
// CELL-GRANULAR checkGeometryCoverage rather than this box either way.
override BOX_SCALE : f32 = 1.0f;

// Does this block's coarse-cell box come within MARGIN of the body? Eight
// corners plus the centre, minimum signed distance -- character for
// character the test refineNearBody makes on the host, including its
// limitation: a body small against a block can fall between the samples.
// d3-amr.mjs's checkGeometryCoverage is the independent cell-granular route
// that catches that, and tools/validate-d3-invariants.js is what runs it.
fn blockWanted(b: vec3<u32>) -> bool {
  if (HAS_BODY == 0u) { return false; }
  // REFINE AHEAD BY WHERE THE BODY WILL BE, not only where it is. This
  // criterion is re-evaluated every MANAGE_EVERY steps, so a shell that
  // only just covers the body at decision time is already stale on the very
  // next step -- and the geometry-forced constraint is supposed to hold at
  // EVERY step, not at the ones the manager happens to run on.
  //
  // Found by measurement, not foresight: M4.2b-iii's `drift` gate reported
  // coverage violations at exactly the cells sitting on the margin
  // boundary, and the body had moved MANAGE_EVERY * |v| = 0.08 cells since
  // the decision. Tiny, and enough, because a boundary is a boundary.
  //
  // Rotation is deliberately not in this term. It matters for a long body
  // spun about a short axis, and every body that currently uses
  // ?refine=body is a sphere, for which it is exactly zero. Adding it
  // speculatively would be an untested term in a criterion whose whole job
  // is to be checkable.
  let lead = MANAGE_EVERY * length(vec3<f32>(body.vx, body.vy, body.vz));
  let reach = MARGIN + lead;
  let lo = vec3<f32>(b * RB) * BOX_SCALE;
  let hi = lo + f32(RB) * BOX_SCALE;
  var best = 1e30f;
  for (var k = 0u; k < 8u; k++) {
    let p = vec3<f32>(
      select(lo.x, hi.x, (k & 1u) != 0u),
      select(lo.y, hi.y, (k & 2u) != 0u),
      select(lo.z, hi.z, (k & 4u) != 0u));
    best = min(best, get_phi3(p, body));
  }
  best = min(best, get_phi3((lo + hi) * 0.5f, body));
  return best <= reach;
}

// --- THE 2:1 CASCADE, M5.5 -------------------------------------------------
//
// The GPU mirror of d3-amr.mjs's `cascade21`, which is the tested statement
// of the rule (M4.2b-iv) and what tools/test-d3-amr.js scores this against.
// Two passes, both GATHERS -- one thread per OUTPUT block, reading inputs it
// does not write. There is no scatter and therefore no race to reason about,
// which is the opposite of how the 2D manager does it and deliberately so.
//
// ONLY THE FINEST LEVEL EVALUATES THE CRITERION (M5.1b). Every coarser level
// is whatever these two passes require, so `decide` runs on one pipeline and
// the rest of the tree is derived.

// PASS 1a. OCTET COMPLETION, level >= 2. A block exists because its parent
// spawned all eight children, so a want for one is a want for all eight --
// see cascade21's header for what a closure that forgets this does to
// check21Balance (which reads octant (0,0,0) alone and calls the rest a
// leaf). Level 1 is exempt: its parent is the dense L0 grid and its blocks
// are refined individually, which is exactly what refineNearBody builds.
//
// Safe in place despite reading siblings it may also write, because each
// index is written by exactly ONE thread and the value only ever goes 0->1:
// every thread computes the OR over the same eight originals.
@compute @workgroup_size(64)
fn completeOctets(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let n = vec3<u32>(nbx(), nby(), nbz());
  if (id >= n.x * n.y * n.z) { return; }
  let b = blockOfId(id);
  let base = (b / 2u) * 2u;
  var any = 0u;
  for (var k = 0u; k < 8u; k++) {
    let c = base + vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
    any = any | blockWant[(c.z * n.y + c.y) * n.x + c.x];
  }
  blockWant[id] = any;
}

// PASS 1b. THE PARENT CLOSURE. This level's want, gathered from the child's:
// parent p is wanted if ANY of the 64 child blocks in [2p-1, 2p+2]^3 is.
//
// WHY 64 AND NOT 8. cascade21's rule is present(m,b) => present(m-1,
// parent(b+d)) for all 27 offsets d, not just the six faces: d = 0 is the
// tree property, the faces are 2:1 balance, and the twelve edges and eight
// corners are the RING -- a corner ring cell's parent tile must exist or
// explode has nothing to read (M5.2a). Inverting that scatter gives this
// gather: p's own eight children, dilated by one block in every direction.
@compute @workgroup_size(64)
fn balance(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let n = vec3<u32>(nbx(), nby(), nbz());
  if (id >= n.x * n.y * n.z) { return; }
  let p = blockOfId(id);
  let cn = n * 2u;                       // the child level's block counts
  var any = 0u;
  for (var k = 0u; k < 64u; k++) {
    // [2p-1, 2p+2] on each axis, periodic like every other block lookup.
    let o = vec3<i32>(i32(k & 3u), i32((k >> 2u) & 3u), i32((k >> 4u) & 3u)) - 1;
    let c = vec3<u32>(
      wrapu(i32(p.x) * 2 + o.x, cn.x),
      wrapu(i32(p.y) * 2 + o.y, cn.y),
      wrapu(i32(p.z) * 2 + o.z, cn.z));
    any = any | childWant[(c.z * cn.y + c.y) * cn.x + c.x];
  }
  // OR, never overwrite: this level may also have been wanted in its own
  // right. Today it cannot be -- only the finest level runs the criterion --
  // but a vorticity criterion per level would, and silently dropping it
  // would be the kind of thing nothing here would catch.
  blockWant[id] = blockWant[id] | any;
}

fn blockOfId(id: u32) -> vec3<u32> {
  let nx = nbx(); let ny = nby();
  return vec3<u32>(id % nx, (id / nx) % ny, id / (nx * ny));
}

// PASS 0a. Zero this level's want, one thread per block.
//
// A GPU PASS RATHER THAN A writeBuffer, because the manager is encoded in
// the middle of a command buffer that already holds hundreds of macro-steps:
// device.queue.writeBuffer is ordered at SUBMIT, not where it is called, so
// it would clear the want set of whichever step happened to be encoded last
// and leave every other step's cascade accumulating into stale data.
//
// Needed because `balance` ORs rather than overwrites (see its own note), so
// every level the criterion does not write must start at zero. The finest
// level does not need it -- `decide` assigns -- but it is dispatched there
// too rather than special-cased, since a skipped clear and an assigning
// `decide` are indistinguishable right up until someone adds a second
// criterion.
@compute @workgroup_size(64)
fn clearWant(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= nbx() * nby() * nbz()) { return; }
  blockWant[id] = 0u;
}

// PASS 0b. Evaluate the criterion, once, into blockWant. Everything after
// this reads the answer rather than recomputing it -- see blockWant's own
// note. It also fixes the answer for the whole topology change, so the drain
// pass and the coarsen pass provably agree about which tiles are dying.
//
// THE UNION OF THE TWO CRITERIA IS TAKEN HERE, and here only (M8.4). A body
// and a detached vortex are two independent reasons to refine a block, and
// `decide` is already the single producer every other pass reads -- so the
// alternative, having the criterion kernel OR into blockWant itself, would
// make the answer depend on which of two dispatches ran second. One producer,
// one union, three consumers.
//
// The geometry half stays FIRST in the expression deliberately: it is the one
// whose failure is a seam through the body (M5.4), and `||` short-circuits,
// so a block the body needs is never subject to the field test at all.
//
// RUNS AT EVERY LEVEL NOW, not only the finest, which is what makes the
// refinement GRADED: the ladder says how deep a region wants to be, and each
// level asks whether that answer reaches it. `balance` then ORs the 2:1
// closure on top -- and its own note already expected this ("this level may
// also have been wanted in its own right ... a vorticity criterion per level
// would"), which is why it ORs rather than overwrites.
//
// HYSTERESIS, mirroring 2D: a block that is ALREADY refined is held against
// the coarsen ladder, one band below the refine ladder, so it must fall a
// full band below the level it holds before releasing it. Without it a block
// sitting on a rung refines and coarsens on alternate evaluations, and every
// one of those transitions re-interpolates its region from the parent --
// which 2D records as visible wake "lumpiness" it was already bitten by.
@compute @workgroup_size(64)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= nbx() * nby() * nbz()) { return; }
  let b = blockOfId(id);
  let geom = blockWanted(b);
  var field = false;
  if (HAS_Q != 0u) {
    // This level's block onto the criterion's LEVEL-1 grid: level m's blocks
    // are 2^(m-1) times finer, so the map is a shift per axis. One integer
    // divide rather than a reduction, at the cost of the criterion being
    // localized to a level-1 block -- which is the resolution the criterion
    // was computed at anyway.
    let sh = u32(max(LEVEL_M - 1, 0));
    let c = vec3<u32>(min(b.x >> sh, CNBX - 1u), min(b.y >> sh, CNBY - 1u), min(b.z >> sh, CNBZ - 1u));
    let q = blockQ[(c.z * CNBY + c.y) * CNBX + c.x];
    // sqrt(Q) is the rotation rate; log2 of it is what the ladder eats.
    let epsPhys = 0.5f * log2(max(q, Q_FLOOR));
    let held = blockSlot[id] >= 0;
    let want = select(desiredLevel(epsPhys), desiredLevelCoarsen(epsPhys), held);
    field = want >= LEVEL_M;
  }
  blockWant[id] = select(0u, 1u, geom || field);
}

// PASS 2. Release the slot of any refined block the criterion no longer
// wants. Only writes: each thread pushes at its own atomically-reserved
// index, so no two threads touch one freeList entry.
//
// BY THE TIME THIS RUNS the drain pass has already restricted the tile onto
// its coarse cells (M4.2b-ii). Freeing first and restricting later cannot
// work: `refine` may hand the same slot out in the very next pass, and the
// data would be gone.
@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= nbx() * nby() * nbz()) { return; }
  let slot = blockSlot[id];
  if (slot < 0) { return; }
  if (blockWant[id] != 0u) { return; }
  blockSlot[id] = -1;
  slotToBlock[slot] = -1;
  // atomicAdd returns the OLD count, which is exactly the index to write.
  let at = atomicAdd(&freeCount[0], 1);
  freeList[at] = slot;
}

// PASS 5, after the fill. Clears the just-filled flags, one thread per slot.
//
// This is a separate pass rather than a line at the end of the fill kernel
// because clearing it there is a RACE: the fill kernel tests slotNew to
// decide whether to touch a cell, and a thread that cleared it would make
// every sibling that had not yet reached the test return early, leaving the
// tile filled in part. Threads within a dispatch are not ordered, so there
// is no "last" thread to do it from.
@compute @workgroup_size(64)
fn clearNew(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= arrayLength(&slotNew)) { return; }
  slotNew[slot] = 0u;
}

// PASS 3. Give a slot to any unrefined block the criterion now wants. Only
// reads of freeList, at an index no longer being concurrently written.
@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= nbx() * nby() * nbz()) { return; }
  if (blockSlot[id] >= 0) { return; }
  if (blockWant[id] == 0u) { return; }
  // atomicSub returns the OLD count; the slot to take is freeList[old-1].
  let old = atomicSub(&freeCount[0], 1);
  if (old <= 0) {
    // OUT OF SLOTS, AND THIS IS A HARD FAILURE, not a degradation.
    //
    // The counter still goes back and the block still stays coarse -- a
    // scrambled free list would be worse than an unrefined block -- but the
    // event is RECORDED, and the host stops the run on it.
    //
    // WHY IT IS NOT "gracefully less refined". Refinement here is
    // geometry-forced: the block that could not be refined is one the body
    // is about to occupy. Leaving it coarse does not cost accuracy at the
    // margin, it puts a coarse/fine seam THROUGH the body, which is the one
    // configuration the solid coupling is not built for (plans/3D.md M5.4,
    // the hard requirement). The 2D experience is the evidence: a pool that
    // runs out does not drift, it diverges.
    //
    // It is also recorded rather than merely inferred from
    // debugCheckGeometryCoverage failing, because that check is expensive,
    // runs only when a tool asks, and answers a different question -- it
    // says the constraint is violated NOW, not that the allocator is the
    // reason. This says the reason.
    atomicAdd(&freeCount[0], 1);
    atomicAdd(&freeCount[1], 1);
    return;
  }
  // THE EXACT HIGH-WATER MARK, recorded where the allocation happens.
  //
  // The host also samples the free count, but only when it polls -- so a
  // spike between two polls is invisible, and "we never saw it above 66%" is
  // a weaker claim than "it never went above 66%". Sizing a slot budget wants
  // the second one. `old - 1` is this allocation's resulting free count, and
  // the minimum of that over the run is the peak usage; atomicMin makes it
  // exact regardless of how many threads allocate in the same dispatch.
  //
  // freeCount[2] rather than a new buffer: the array is
  // array<atomic<i32>, 4> and [2]/[3] were already spare.
  atomicMin(&freeCount[2], old - 1);
  let slot = freeList[old - 1];
  blockSlot[id] = slot;
  slotToBlock[slot] = i32(id);
  // The tile holds whatever the previous owner left. The fill pass, which
  // runs after this one, turns it into an interpolation of the coarse field
  // and clears the flag. Until M4.2b-ii this was left set to nothing at all,
  // and validate-d3-invariants.js's body-refine config blew the field up --
  // deliberately, as the evidence that this line had to exist.
  slotNew[slot] = 1u;
}
