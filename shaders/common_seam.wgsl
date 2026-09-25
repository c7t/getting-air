// The coarse/fine SEAM's shared geometry, for explode and coalesce
// (plans/2D-backport.md B6-1). Fragment only; its includers declare
// `levelParams` (the CHILD level's uniform) and `blockSlot` (the CHILD level's
// indirection) and include common_lattice.wgsl first.
//
// THREE COORDINATE FRAMES, and every function here names the one it takes:
//
//   fine    the CHILD level's global cell grid, nbxC * 2RB across -- the grid
//           a ring cell's position is expressed in.
//   coarse  the PARENT level's global cell grid, half that: nbxC * RB across.
//           A fine cell's parent is fine >> 1; cell-centred refinement puts
//           the two children of parent c at 2c and 2c + 1 on each axis.
//   block   a CHILD block covers RB coarse cells = 2RB fine cells, so the
//           child block of coarse cell c is c / RB and of fine cell g is
//           g / (2RB). One child block <-> one child tile.
//
// "COVERED" MEANS "OWNED BY THE CHILD LEVEL" (plans/uniform-levels.md 2.5a's
// second meaning): a coarse cell whose child block holds a tile. Refinement is
// whole-block, so it is one `blockSlot` load for any cell of the block.
//
// Everything is periodic. Positions are carried UNWRAPPED where a caller
// needs tile-local offsets to stay contiguous across the domain edge, and
// wrapped only at the point of a lookup.

fn seamNbxC() -> u32 { return levelParams.nbx; }
fn seamNbyC() -> u32 { return levelParams.nby; }

fn wrapi(a: i32, n: u32) -> i32 { return ((a % i32(n)) + i32(n)) % i32(n); }

fn wrapFine(g: vec2<i32>) -> vec2<i32> {
  return vec2<i32>(wrapi(g.x, seamNbxC() * 2u * RB), wrapi(g.y, seamNbyC() * 2u * RB));
}
fn wrapCoarse(c: vec2<i32>) -> vec2<i32> {
  return vec2<i32>(wrapi(c.x, seamNbxC() * RB), wrapi(c.y, seamNbyC() * RB));
}

// A WRAPPED coarse cell -> is it owned by the child level?
fn coveredCoarse(c: vec2<i32>) -> bool {
  let b = vec2<u32>(c) / RB;
  return blockSlot[b.y * seamNbxC() + b.x] >= 0;
}
// Any coarse cell, wrapped here.
fn coveredCoarseU(c: vec2<i32>) -> bool { return coveredCoarse(wrapCoarse(c)); }

// A WRAPPED fine cell -> its child block (x, y).
fn blockOfFine(g: vec2<i32>) -> vec2<i32> { return g / i32(2u * RB); }
