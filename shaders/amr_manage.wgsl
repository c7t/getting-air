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

@group(0) @binding(0) var<storage, read>       blockCriterion  : array<f32>;
@group(0) @binding(1) var<storage, read_write> blockSlot       : array<i32>;
@group(0) @binding(2) var<storage, read_write> slotToBlock     : array<i32>;
@group(0) @binding(3) var<storage, read_write> freeList        : array<i32>;
@group(0) @binding(4) var<storage, read_write> freeCount       : atomic<i32>;
@group(0) @binding(5) var<storage, read_write> newlyActivated  : array<u32>;

override W : u32;
override H : u32;
override REFINE_THRESH : f32;
override COARSEN_THRESH : f32;
const BLOCK = 8u;
const EPS_FLOOR = 1e-6f;

fn epsFor(blockID: u32) -> f32 {
  return min(1.0f, log2(max(blockCriterion[blockID], EPS_FLOOR)));
}

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  let nblocks = (W / BLOCK) * (H / BLOCK);
  if (blockID >= nblocks) { return; }

  let currentSlot = blockSlot[blockID];
  if (epsFor(blockID) < COARSEN_THRESH && currentSlot >= 0) {
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
  if (epsFor(blockID) >= REFINE_THRESH && currentSlot < 0) {
    let oldCount = atomicSub(&freeCount, 1);
    if (oldCount > 0) {
      let slot = freeList[u32(oldCount - 1)];
      blockSlot[blockID] = slot;
      slotToBlock[u32(slot)] = i32(blockID);
      newlyActivated[u32(slot)] = 1u;
    } else {
      atomicAdd(&freeCount, 1); // pool exhausted this round -- undo, stay coarse
    }
  }
}
