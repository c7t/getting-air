// THE PARENT ADDRESS, pool case: level m >= 2, whose parent is level m-1 and
// therefore a POOL of tiles rather than a dense grid. plans/3D.md M5.2b.
// Fragment only; the entry files list every include. The sibling is
// common_d3_parent_dense.wgsl, and its header explains why this is the only
// piece that differs between the two.
//
// COORDINATES. NX/NY/NZ are the PARENT grid's dimensions for this pipeline
// -- L0's scaled by 2^(m-1) -- so `v` here is a global level-(m-1) cell.
// The parent's own tiles hold 2*RB parent cells across, so:
//
//     parent block   = v / (2*RB)
//     parent-local   = v - block * 2*RB + GHOST
//
// which is `fromGlobalFine` in d3-amr.mjs, and tools/test-d3-amr.js checks
// that route against tile-local arithmetic at levels 2 and 3.
//
// IT ALWAYS LANDS IN THE PARENT'S INTERIOR, never in the parent's own ring,
// and that is the point rather than a happy accident: v / (2*RB) names the
// tile that OWNS v, so the local index falls in [GHOST, GHOST + 2*RB) by
// construction. A ring cell of the parent is the parent's own interface
// ghost -- reading one would hand the child a value the parent never solved,
// which is the same mistake DIRECT_GHOST exists to avoid one level down.
//
// ABSENCE IS REAL HERE. parentIndex returns -1 when the parent tile is not
// allocated, where the dense sibling can never fail. Two different callers
// rely on two different guarantees:
//   - the cells explode WRITES are covered by M5.2a's ring-parent invariant,
//     checked on the host by checkRingParentCoverage and gated by
//     tools/validate-d3-invariants.js. The -1 branch there is unreachable,
//     the same way explode's `i = 0 cannot reach here` is.
//   - the linear explosion's gradient reaches one cell FURTHER than that
//     invariant covers, so it tests parentPresent and degrades to a uniform
//     explosion where the neighbour is missing. That path is reachable, is
//     first-order, and is exactly conservative.

@group(0) @binding(8) var<storage, read> parentBlockSlot : array<i32>;

fn parentNb() -> vec3<u32> { return vec3<u32>(NX, NY, NZ) / (2u * RB); }

fn parentBlockOfCell(v: vec3<u32>) -> vec3<u32> { return v / (2u * RB); }

fn parentBlockIdOf(pb: vec3<u32>) -> u32 {
  let n = parentNb();
  return (pb.z * n.y + pb.y) * n.x + pb.x;
}

fn parentSlotOf(v: vec3<u32>) -> i32 {
  return parentBlockSlot[parentBlockIdOf(parentBlockOfCell(v))];
}

fn parentIndex(v: vec3<u32>) -> i32 {
  let slot = parentSlotOf(v);
  if (slot < 0) { return -1; }
  let pb = parentBlockOfCell(v);
  let loc = v - pb * (2u * RB) + vec3<u32>(GHOST);
  return i32(poolCell(u32(slot), loc));
}

fn parentPresent(v: vec3<u32>) -> bool { return parentSlotOf(v) >= 0; }
