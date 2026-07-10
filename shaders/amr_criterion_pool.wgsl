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
// quadrant math needed the way amr_step1_pool.wgsl's quadrant lookup
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

@group(0) @binding(0) var<storage, read>       vel            : array<f32>; // parent level's finePoolVel
@group(0) @binding(1) var<storage, read>       slotToBlock    : array<i32>; // parent level's own
@group(0) @binding(2) var<storage, read_write> childCriterion : array<f32>; // child level's blockCriterion

override RB : u32;
override NBX_PARENT : u32;
const GHOST = 2u;

fn velAt(slot: u32, fx: u32, fy: u32, FB: u32, comp: u32) -> f32 {
  return vel[(slot * (FB * FB) + fy * FB + fx) * 2u + comp];
}

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
    // Discrete vorticity: du_y/dx - du_x/dy, same formula as
    // amr_criterion.wgsl/amr_render.wgsl -- ghost border guarantees fx+-1
    // in range (see header).
    omega = (velAt(slot, fx + 1u, fy, FB, 1u) - velAt(slot, fx - 1u, fy, FB, 1u)) * 0.5f
          - (velAt(slot, fx, fy + 1u, FB, 0u) - velAt(slot, fx, fy - 1u, FB, 0u)) * 0.5f;
  }

  wg_omega[lid] = abs(omega);
  workgroupBarrier();

  if (lid == 0u && blockID >= 0) {
    var m = 0f;
    for (var i = 0u; i < 64u; i++) { m = max(m, wg_omega[i]); }
    let bx = u32(blockID) % NBX_PARENT;
    let by = u32(blockID) / NBX_PARENT;
    let nbxChild = NBX_PARENT * 2u;
    let childBlockID = (by * 2u + qy) * nbxChild + (bx * 2u + qx);
    childCriterion[childBlockID] = m;
  }
}
