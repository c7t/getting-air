// Dynamic refinement: the refine/coarsen decision and the free-list
// allocation that serves it. plans/3D.md M4.2b-i, and the 3D sibling of the
// 2D shaders/amr_manage.wgsl. Fragment only; the entry files list every
// include.
//
// TWO ENTRY POINTS, DISPATCHED AS TWO SEPARATE COMPUTE PASSES, coarsen fully
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
// WHAT THIS STAGE DOES NOT DO: a slot handed out here is NOT initialized,
// and a slot released here is NOT restricted back to the coarse grid first.
// Both are M4.2b-ii. So a criterion that actually FIRES produces an
// uninitialized tile today -- which is why ?dynamic=1 is opt-in, why it is
// refused unless ?refine=body, and why the only case that exercises it has a
// PINNED body whose wanted-set never changes.

@group(0) @binding(0) var<storage, read_write> blockSlot   : array<i32>;
@group(0) @binding(1) var<storage, read_write> slotToBlock : array<i32>;
@group(0) @binding(2) var<storage, read_write> freeList    : array<i32>;
@group(0) @binding(3) var<storage, read_write> freeCount   : atomic<i32>;
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

override MARGIN : f32 = 2.0f;
// The body's own radius is baked into get_phi3, so the criterion needs
// nothing else about the shape.
override HAS_BODY : u32 = 0u;

// Does this block's coarse-cell box come within MARGIN of the body? Eight
// corners plus the centre, minimum signed distance -- character for
// character the test refineNearBody makes on the host, including its
// limitation: a body small against a block can fall between the samples.
// d3-amr.mjs's checkGeometryCoverage is the independent cell-granular route
// that catches that, and tools/validate-d3-invariants.js is what runs it.
fn blockWanted(b: vec3<u32>) -> bool {
  if (HAS_BODY == 0u) { return false; }
  let lo = vec3<f32>(b * RB);
  let hi = lo + f32(RB);
  var best = 1e30f;
  for (var k = 0u; k < 8u; k++) {
    let p = vec3<f32>(
      select(lo.x, hi.x, (k & 1u) != 0u),
      select(lo.y, hi.y, (k & 2u) != 0u),
      select(lo.z, hi.z, (k & 4u) != 0u));
    best = min(best, get_phi3(p, body));
  }
  best = min(best, get_phi3((lo + hi) * 0.5f, body));
  return best <= MARGIN;
}

fn blockOfId(id: u32) -> vec3<u32> {
  let nx = nbx(); let ny = nby();
  return vec3<u32>(id % nx, (id / nx) % ny, id / (nx * ny));
}

// PASS 0. Evaluate the criterion, once, into blockWant. Everything after
// this reads the answer rather than recomputing it -- see blockWant's own
// note. It also fixes the answer for the whole topology change, so the drain
// pass and the coarsen pass provably agree about which tiles are dying.
@compute @workgroup_size(64)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= nbx() * nby() * nbz()) { return; }
  blockWant[id] = select(0u, 1u, blockWanted(blockOfId(id)));
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
  let at = atomicAdd(&freeCount, 1);
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
  let old = atomicSub(&freeCount, 1);
  if (old <= 0) {
    // Out of slots. Put the counter back and leave the block coarse rather
    // than corrupting the list -- running out of headroom is a capacity
    // problem (?slotHeadroom=), and it must degrade to "less refined than
    // asked for" rather than to a scrambled pool. The host sees it as
    // debugCheckGeometryCoverage failing, which is the right report: the
    // geometry-forced constraint is genuinely not being met.
    atomicAdd(&freeCount, 1);
    return;
  }
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
