// THE ACTIVE-SLOT LIST: one pool level's in-use slots, compacted, with their
// count (`ActiveList`, shaders/common_active_list.wgsl).
//
// WHY. Every per-slot pass dispatches `(x, y, MAX_FINE_BLOCKS)` and lets each
// workgroup of an unassigned slot read `slotToBlock[slot] < 0` and return.
// That return is NOT free: measured on the phone (img-tec, 2026-09-24) an
// early-returning 8x8 workgroup costs ~14 ns, per thread launched rather than
// per workgroup, and the finest level runs 2^(levels-1) substeps per root
// step -- so at ?levels=4 one root step launched ~305k workgroups for ~13k
// live tiles, ~4 of its 9.4 ms. plans/perf-characterization.md has the
// measurement. main-amr.js's ?launch=stride instead dispatches K workgroups in
// z and has each walk this list at stride K (each kernel's `mainStride`). A
// `main` entry point never reads it, so every other page and every
// out-of-macro-step caller is unaffected.
//
// NOT dispatchWorkgroupsIndirect, which is the obvious consumer of a count and
// was built first: it was bit-identical and LOST on both devices, because each
// indirect call carries ~230 us (phone) / ~390 us (desktop) in this Chrome
// against ~29 / ~5 us for a direct one -- phone 10.3 -> 13.2 ms per root step,
// desktop 0.55 -> 21.5 ms. Commit 8e67789 has that build and its measurement.
//
// ORDER IS ASCENDING SLOT, and deterministic by construction -- one workgroup,
// each thread owning a contiguous CHUNK of slots, an exclusive scan of the
// per-thread counts, then each thread writing its chunk in order. No atomics,
// so the list is a pure function of `slotToBlock`; amr2d.mjs's
// `activeSlotList` is the host statement. Launch order cannot change a result
// anyway (each tile's work is independent of which workgroup runs it, and the
// force partials are summed with INTEGER atomics), which is what lets
// ?launch=stride be gated bit-for-bit against ?launch=all.

@group(0) @binding(0) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(1) var<storage, read_write> activeList  : ActiveList;
// @include "common_active_list.wgsl"

override N_SLOTS : u32;

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
    if (slotToBlock[s] >= 0) { activeList.slots[k] = s; k++; }
  }

  if (t == WG - 1u) {
    activeList.count = scan[t];
  }
}
