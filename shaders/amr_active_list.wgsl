// THE ACTIVE-SLOT LIST: one pool level's in-use slots, compacted, plus the
// indirect-dispatch arguments that launch exactly that many tiles.
//
// WHY. Every per-slot pass dispatches `(x, y, MAX_FINE_BLOCKS)` and lets each
// workgroup of an unassigned slot read `slotToBlock[slot] < 0` and return.
// That return is NOT free: measured on the phone (img-tec, 2026-09-24) an
// early-returning 8x8 workgroup costs ~14 ns, per thread launched rather than
// per workgroup, and the finest level runs 2^(levels-1) substeps per root
// step -- so at ?levels=4 one root step launched ~305k workgroups for ~13k
// live tiles, ~4 of its 9.4 ms. plans/perf-characterization.md has the
// measurement. Consumers launched through `dispatchWorkgroupsIndirect` from
// `args` see z in [0, count) and read their slot as `activeSlots[z]` (each
// kernel's `mainIndirect` entry point); a `main` entry point never reads
// this list, so every other page and every out-of-macro-step caller is
// unaffected.
//
// ORDER IS ASCENDING SLOT, and deterministic by construction -- one workgroup,
// each thread owning a contiguous CHUNK of slots, an exclusive scan of the
// per-thread counts, then each thread writing its chunk in order. No atomics,
// so the list, and with it every consumer's launch order, is a pure function
// of `slotToBlock`. amr2d.mjs's `activeSlotList` is the host statement.
// Launch order cannot change a result anyway (each tile's work is independent
// of which z runs it, and the force partials are summed with INTEGER atomics),
// which is what lets ?indirect=1 be gated bit-for-bit against ?indirect=0.
//
// `args` holds three (x, y, z) triples, one per consumer SHAPE, all with the
// same z = count: (SHAPE0_XY)^2 for the FB-square passes (step, force,
// explode, interp), (SHAPE1_XY)^2 for coalesce's 2*RB square, (SHAPE2_XY)^2
// for average's one-workgroup-per-slot. The page picks the triple by byte
// offset (0, 12, 24).

@group(0) @binding(0) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(1) var<storage, read_write> activeSlots : array<u32>;
@group(0) @binding(2) var<storage, read_write> args        : array<u32, 9>;

override N_SLOTS : u32;
override SHAPE0_XY : u32;
override SHAPE1_XY : u32;
override SHAPE2_XY : u32;

const WG : u32 = 256u;
var<workgroup> scan : array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  let chunk = (N_SLOTS + WG - 1u) / WG;
  let lo = min(t * chunk, N_SLOTS);
  let hi = min(lo + chunk, N_SLOTS);

  var n = 0u;
  for (var s = lo; s < hi; s++) {
    if (slotToBlock[s] >= 0) { n++; }
  }

  // Inclusive Hillis-Steele scan over the 256 per-thread counts.
  scan[t] = n;
  workgroupBarrier();
  for (var d = 1u; d < WG; d = d << 1u) {
    let add = select(0u, scan[t - d], t >= d);
    workgroupBarrier();
    scan[t] += add;
    workgroupBarrier();
  }

  var k = scan[t] - n;   // exclusive prefix: where this chunk starts
  for (var s = lo; s < hi; s++) {
    if (slotToBlock[s] >= 0) { activeSlots[k] = s; k++; }
  }

  if (t == WG - 1u) {
    let count = scan[t];
    args[0] = SHAPE0_XY; args[1] = SHAPE0_XY; args[2] = count;
    args[3] = SHAPE1_XY; args[4] = SHAPE1_XY; args[5] = count;
    args[6] = SHAPE2_XY; args[7] = SHAPE2_XY; args[8] = count;
  }
}
