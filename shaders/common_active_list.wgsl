// A pool level's ACTIVE-SLOT LIST, as its readers and its writer
// (amr_active_list.wgsl) see it: the in-use slots in ascending order, with
// their count in front. The count is IN the buffer, not beside it, so a kernel
// that walks the list needs no second binding -- and the layout is carried by
// this type rather than by a convention about index 0.
//
// Every includer declares `activeList` at its own binding; only its
// `mainIndirect`/`mainStride` entry points read it, so a pipeline built on
// `main` never needs it bound.
struct ActiveList {
  count : u32,
  slots : array<u32>,
}

// ?launch=stride: the list's length as a WORKGROUP-UNIFORM value, so a loop
// bounded by it may contain barriers. `li` is local_invocation_index.
var<workgroup> wg_activeCount : u32;
fn activeListCount(li: u32) -> u32 {
  if (li == 0u) { wg_activeCount = activeList.count; }
  return workgroupUniformLoad(&wg_activeCount);
}
