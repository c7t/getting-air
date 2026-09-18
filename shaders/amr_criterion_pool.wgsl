// Milestone 9 (plans/AMR-multilevel.md): per-quadrant vorticity criterion
// for deciding whether a level-(m+1) child should exist -- sibling of
// amr_criterion.wgsl (which stays exactly as-is: it decides L0->L1,
// reading L0's own dense velBuf; this decides L(m)->L(m+1) for any m>=1,
// reading level m's own finePoolVel).
//
// Dispatch: (2, 2, MAX_FINE_BLOCKS[m]) with workgroup_size(8,8). A parent
// slot's own interior is 2*RB x 2*RB cells -- exactly 4 RB*RB=64-cell
// quadrants, each exactly one workgroup (same "one workgroup = one
// reduction unit" convention as amr_criterion.wgsl/amr_force*.wgsl).
// workgroup_id.xy IS the quadrant (qx,qy) directly -- no separate
// quadrant math needed the way amr_step1.wgsl's quadrant lookup
// requires, since here we're producing a criterion for a NOT-YET-existing
// child, not consuming an already-assigned quadrant.
//
// NBX_PARENT is a compile-time override, not a runtime LevelParams field
// like the interp/step1/average/force pool-parent shaders use -- unlike
// those, there's no established "one pipeline shared across every
// level>=2" need here yet (criterion/manage never had that pattern before
// this milestone, and at this plan's actual validated depth (N=3) this
// shader only ever runs with parent=level 1 anyway). One pipeline
// instance per parent level, override-baked, is simpler than adding a new
// per-level uniform buffer just for level 1 (which no other level-1
// shader needs) purely to future-proof past N=3.
//
// Same-slot neighbor lookup for the vorticity finite difference uses
// PLAIN (not periodic-wrapped) +-1 indexing within the slot's own FB*FB
// buffer -- valid without clamping because the 2-cell ghost border
// (GHOST=2) already provides the +-1 margin every interior cell needs,
// and ghost cells already hold genuine neighboring data (fine-fine
// consultation or coarse interpolation), unlike the dense L0 criterion
// which must wrap around the WHOLE domain itself.

// @include "common_criterion.wgsl"

@group(0) @binding(0) var<storage, read>       vel            : array<f32>; // parent level's finePoolVel
@group(0) @binding(1) var<storage, read>       slotToBlock    : array<i32>; // parent level's own
@group(0) @binding(2) var<storage, read_write> childCriterion : array<f32>; // child level's blockCriterion
// Parent level's blockSlot, for the ring-free stencil below. Always bound;
// only read when GHOST == 0.
@group(0) @binding(3) var<storage, read>       blockSlot      : array<i32>; // parent level's own

override RB : u32;
override NBX_PARENT : u32;
// Only read on the ring-free path; every ringed pipeline leaves it at the
// default, where it is folded away unused.
override NBY_PARENT : u32 = 1u;

// GHOST is an OVERRIDE since plans/uniform-levels.md U4-1: the ROOT level has
// no ring (amr2d.mjs's ghostDepthAtLevel(0) is 0). Default 2 keeps every
// existing pipeline byte-identical.
override GHOST : u32 = 2u;

fn velAt(slot: u32, fx: u32, fy: u32, FB: u32, comp: u32) -> f32 {
  return vel[(slot * (FB * FB) + fy * FB + fx) * 2u + comp];
}

// One stencil tap, resolved against the OWNING same-level tile when it leaves
// this one's interior.
//
// THE RING-FREE PATH, AND IT IS DERIVED FROM GHOST RATHER THAN FLAGGED. At
// GHOST == 0 a slot is exactly its own 2*RB x 2*RB cells and there is no ring
// to read, so resolving against the neighbour is not an option among several
// -- it is the only correct behaviour. A separate override could be left unset
// on a ring-free pipeline, and U3 already paid for exactly that shape once
// (the root inheriting DIRECT_GHOST: 0 from step1Constants, which asked a
// level with no ring to read a ghost cell nothing fills).
//
// The rule is amr2d.mjs's resolveSource, which reads the POOL's own ring depth
// -- at 0 a tap of -1 lands at 2*RB - 1 in the tile on the low side, and one
// of 2*RB lands at 0 in the tile on the high side. The block grid is periodic,
// matching every other kernel here.
//
// THE ROOT IS ALWAYS FULL, so `blockSlot` is the identity and the `< 0` branch
// is unreachable there. It is written anyway because the same ring-free path
// would be wrong to leave open-coded if a future level is ever ring-free and
// sparse, and because a `cannot happen` branch that returns a plausible number
// silently is exactly what plans/2D-backport.md B6-9c is about: 0 velocity is
// what a quiescent cell reads.
fn tapVel(bx: u32, by: u32, sx: i32, sy: i32, FB: u32, comp: u32) -> f32 {
  var nx = sx; var ny = sy;
  var tbx = bx; var tby = by;
  if (nx < 0)            { nx += i32(FB); tbx = (bx + NBX_PARENT - 1u) % NBX_PARENT; }
  else if (nx >= i32(FB)) { nx -= i32(FB); tbx = (bx + 1u) % NBX_PARENT; }
  if (ny < 0)            { ny += i32(FB); tby = (by + NBY_PARENT - 1u) % NBY_PARENT; }
  else if (ny >= i32(FB)) { ny -= i32(FB); tby = (by + 1u) % NBY_PARENT; }
  let s = blockSlot[tby * NBX_PARENT + tbx];
  if (s < 0) { return 0f; }
  return velAt(u32(s), u32(nx), u32(ny), FB, comp);
}

// Read by common_criterion.wgsl's wgReduceMax1 -- see its own comment.
var<workgroup> wg_omega : array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wgid: vec3<u32>
) {
  let lx = gid.x; let ly = gid.y; // interior-local coords, [0, 2*RB)
  let slot = gid.z;
  let qx = wgid.x; let qy = wgid.y; // quadrant IS the workgroup id -- see header

  let FB = RB * 2u + 2u * GHOST;
  let blockID = slotToBlock[slot];

  var omega = 0f;
  if (blockID >= 0) {
    let fx = lx + GHOST; let fy = ly + GHOST;
    if (GHOST == 0u) {
      // Ring-free: every tap that leaves the tile is resolved against the
      // owning one. See tapVel.
      let bx = u32(blockID) % NBX_PARENT;
      let by = u32(blockID) / NBX_PARENT;
      let ix = i32(fx); let iy = i32(fy);
      omega = discreteCurl(tapVel(bx, by, ix + 1, iy, FB, 1u), tapVel(bx, by, ix - 1, iy, FB, 1u),
                           tapVel(bx, by, ix, iy + 1, FB, 0u), tapVel(bx, by, ix, iy - 1, FB, 0u));
    } else {
      // The ghost border guarantees fx+-1 is in range (see header), which is
      // the whole difference from the dense kernel's periodic wrap.
      omega = discreteCurl(velAt(slot, fx + 1u, fy, FB, 1u), velAt(slot, fx - 1u, fy, FB, 1u),
                           velAt(slot, fx, fy + 1u, FB, 0u), velAt(slot, fx, fy - 1u, FB, 0u));
    }
  }

  wg_omega[lid] = abs(omega);
  workgroupBarrier();
  // Unconditional: every invocation must reach the barriers inside.
  wgReduceMax1(lid);

  if (lid == 0u && blockID >= 0) {
    let m = wg_omega[0];
    let bx = u32(blockID) % NBX_PARENT;
    let by = u32(blockID) / NBX_PARENT;
    let nbxChild = NBX_PARENT * 2u;
    let childBlockID = (by * 2u + qy) * nbxChild + (bx * 2u + qx);
    childCriterion[childBlockID] = m;
  }
}
