// Copy the dense L0 grid into the ROOT POOL, for U2's addressing proof.
// plans/uniform-levels.md U2.
//
// WHY THIS EXISTS AND WHY IT IS NOT THE POINT. Nothing in the solver needs
// this pass. It exists so the root pool's addressing can be scored against the
// live dense buffer BEFORE any kernel depends on it -- the same order of work
// B3-5 used when it proved `tileOriginL0`'s closed form against the per-slot
// origin buffers and only then deleted them. An addressing or stride error
// found here costs an afternoon; found at U3 it looks like a physics
// regression, because by then the mapping is inside a step kernel and the
// symptom is a wrong field.
//
// WORD-FOR-WORD, NOT VALUE-BY-VALUE. It moves `fWords()` u32s per cell rather
// than unpacking and repacking nine f32s, so it is correct under both the
// plain `array<f32>` layout and the packed-half one (?f16=1|2) without knowing
// which is in use -- exactly the trick common_interp_kernel.wgsl's fine-fine
// copy already uses. It also makes the comparison downstream EXACT: a mirror
// that round-tripped through f32 could hide a low-bit difference under F16,
// which is the one regime where this check matters most.
//
// THE ROOT HAS NO RING (amr2d.mjs's ghostDepthAtLevel(0) is 0), so a slot is
// exactly 2*RB x 2*RB of its own cells and its local coordinates ARE its
// cells. There is no GHOST offset anywhere below, and its absence is part of
// what this pass proves.
//
// AND THE DENSE SIDE IS NOT ROW-MAJOR. This wrote `gy*W + gx` until
// 2026-09-17, and so did amr2d.mjs's rootCellToDense, so U2's two "independent
// routes" agreed with each other over a mirror that fetched the wrong dense
// cell for 98.4% of the root pool. The dense L0 grid is 8x8 block-major
// (amr_step.wgsl's cellIndex, and every snapshot's `layout: 'block8'` tag);
// the root pool is 2*RB x 2*RB tiles, row-major within a tile. Those are
// DIFFERENT layouts and the mirror is the permutation between them -- which is
// the whole reason this pass has to exist and cannot be a memcpy. See
// plans/uniform-levels.md U2.

// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"

@group(0) @binding(0) var<storage, read>       f_dense     : array<u32>;
@group(0) @binding(1) var<storage, read_write> f_root      : array<u32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;

override W  : u32;   // domain, in root cells
override H  : u32;
override RB : u32;

// The DENSE grid's sub-tile, not a level's RB -- see the header. They are
// equal today (main-amr.js's `RB = BLOCK`) and that coincidence is exactly
// what let the row-major slip look plausible, so it is named separately.
const DENSE_BLOCK : u32 = 8u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side = RB * 2u;              // a root tile's own cells, per side
  let lx = gid.x; let ly = gid.y; let slot = gid.z;
  if (lx >= side || ly >= side) { return; }
  if (slot >= arrayLength(&slotToBlock)) { return; }

  // The root is always full, so this IS the identity -- and it is read rather
  // than assumed, so a pool whose indirection was not the identity would
  // mirror to the wrong place and the comparison would say so.
  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  let nbx = W / side;
  let bx = u32(blockID) % nbx;
  let by = u32(blockID) / nbx;

  let gx = bx * side + lx;         // the dense cell this root cell stands for
  let gy = by * side + ly;
  if (gx >= W || gy >= H) { return; }

  // The dense grid's OWN index, restated here rather than @include'd: every
  // dense-parent accessor in this directory carries its own copy of these four
  // lines (common_interp_parent_dense.wgsl, common_avg_parent_dense.wgsl,
  // amr_force.wgsl, amr_criterion.wgsl, amr_render.wgsl) and this one is
  // scored against amr2d.mjs's denseCellIndex, which is scored in turn against
  // the host decoder that reads real snapshots.
  let denseNBX = W / DENSE_BLOCK;
  let dbx = gx / DENSE_BLOCK; let dby = gy / DENSE_BLOCK;
  let dlx = gx % DENSE_BLOCK; let dly = gy % DENSE_BLOCK;

  let densePlane = W * H;                            // cells per direction plane
  // PLANE STRIDE IS arrayLength/9, NEVER /fWords(). Every `f` buffer in this
  // project is sized for nine planes whatever the packing is, and under ?f16=
  // only the first five are used -- so the STRIDE does not move, the plane
  // COUNT does. Dividing by fWords() here put four planes of every cell at the
  // wrong offset and U2 caught it on the first run with ?f16=1, which is the
  // whole reason this stage exists before a kernel depends on the mapping.
  // amr_step1.wgsl, common_average.wgsl and common_interp_kernel.wgsl all say
  // arrayLength(&f_pool) / 9u; this is the same convention.
  let rootPlane  = arrayLength(&f_root) / 9u;
  let denseCell  = (dby * denseNBX + dbx) * (DENSE_BLOCK * DENSE_BLOCK) + dly * DENSE_BLOCK + dlx;
  let rootCell   = slot * (side * side) + ly * side + lx;

  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    f_root[wi * rootPlane + rootCell] = f_dense[wi * densePlane + denseCell];
  }
}
