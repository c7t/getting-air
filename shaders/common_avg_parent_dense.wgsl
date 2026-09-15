// The DENSE-parent half of the averaging accessor (plans/2D-backport.md B3-2)
// -- i.e. the L1 -> L0 hop, where the destination is the dense, ghost-free,
// cellIndex()-addressed coarse grid rather than another pool tile.
//
// Three functions and nothing else; common_average.wgsl is the kernel. See
// its header for the contract and for why this asymmetry is deliberate.
//
// Reads `state` and writes `f_coarse`, both declared by the entry file
// (amr_average_f2c.wgsl), as are the W/H overrides -- a fragment must not
// @include, and WGSL module scope is order-independent.

// Level 1's parent is L0, whose tau IS the card state's tau. (Every deeper
// level's parent tau is a recursive tauAtLevel -- see the pool half.)
fn parentTau() -> f32 { return state.tau; }

// L0's own block-linear addressing: BLOCK x BLOCK coarse cells per block,
// blocks in row-major order. shaders/amr_step.wgsl has the same function for
// the coarse step.
const BLOCK = 8u;
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

// Destination: coarse BUFFER coords (no window conversion). Level 1's blockID
// space coincides with L0's own block grid by construction (the 1:1
// footprint-preserving parent, plans/AMR-multilevel.md decision 1), which is
// what makes this a bare mod/div rather than a parentSlot lookup.
fn parentCellForChild(slot: u32, blockID: i32, lcx: u32, lcy: u32) -> u32 {
  let nbx = W / BLOCK;
  let cbx = (u32(blockID) % nbx) * RB + lcx;
  let cby = (u32(blockID) / nbx) * RB + lcy;
  return cellIndex(cbx, cby);
}

fn parentStoreWord(cell: u32, wi: u32, word: u32) {
  f_coarse[wi * (W * H) + cell] = word;
}
