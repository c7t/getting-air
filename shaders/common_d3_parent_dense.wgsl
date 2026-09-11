// THE PARENT ADDRESS, dense case: level 1, whose parent is the L0 grid.
// plans/3D.md M5.2b. Fragment only; the entry files list every include.
//
// WHY THIS IS A SEPARATE FRAGMENT AT ALL. Everything else in
// common_d3_pool.wgsl is already level-generic once NX/NY/NZ are read as
// "the PARENT grid's dimensions" rather than "L0's" -- nbx() = NX/RB is then
// that level's block count, fineDim() = 2*NX is its child-cell count, and
// blockOfCoarse, coarseOfFine, blockOfFine and localInTile3 all follow. That
// is the uniform tile shape (d3-amr.mjs's poolAtLevel) paying off in the
// kernels: at level 1 the parent grid IS L0 and nothing changes.
//
// The ONE thing that is genuinely different at depth is how a parent CELL is
// addressed in the parent's f array: a dense index here, a tile lookup
// through the parent's own blockSlot at level >= 2. So that is the only
// thing these two fragments provide, and the interface coupling
// (explode/coalesce) is written once against it.
//
// PRESENCE. A dense parent cell always exists, so parentPresent is
// constant-true and every guard that consults it folds away. It is not
// constant at depth -- a parent tile may simply not be allocated -- and the
// invariant that makes the cells explode actually READS safe is
// d3-amr.mjs's checkRingParentCoverage (M5.2a), gated by
// tools/validate-d3-invariants.js.

fn parentIndex(v: vec3<u32>) -> i32 { return i32(coarseCell(v)); }
fn parentPresent(v: vec3<u32>) -> bool { return true; }

// THE PARENT'S MACROSCOPIC ARRAY HAS A DIFFERENT LAYOUT AT DEPTH, and that
// is not a style difference -- it is a silent bug if assumed away. The dense
// L0 `mac` is INTERLEAVED, [rho,ux,uy,uz] per cell, because the renderer and
// every host readback walk it that way. A pool `mac_pool` is PLANAR,
// component-major, because pool kernels stride by slot. Coalesce writes one
// of them and reads the other, so it has to ask.
fn parentMacIndex(cell: u32, comp: u32, plane: u32) -> u32 { return 4u * cell + comp; }
