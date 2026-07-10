// Milestone 7 (plans/AMR-multilevel.md): fine -> parent-pool averaging
// (restriction) for every L(m)->L(m-1) hop with m>=2 -- sibling of
// amr_average_f2c.wgsl, which stays exactly as-is and is now specifically
// the L1->L0 case (writing into the dense buffer via cellIndex()).
//
// Same restriction math (arithmetic-mean rho, momentum-weighted velocity,
// inverse-Dupuis-Chopard-rescaled fneq) -- the only structural difference
// is WHERE the averaged result is written: not a dense cellIndex()
// address, but a specific (lx,ly) position inside the PARENT's own FB*FB
// tile, found via this slot's own parentSlot+quadrant (the same two
// fields amr_interp_pool_parent.wgsl already reads for the forward/
// prolongation direction -- no new per-slot fields needed here).
//
// Dispatch shape identical to amr_average_f2c.wgsl: (1, 1, MAX_FINE_BLOCKS)
// with workgroup_size(8,8) -- RB*RB=64 cells is exactly one workgroup, one
// thread per coarse-equivalent cell of THIS level's own footprint.

struct LevelParams {
  nbx: u32,        // unused here (destination is parentSlot+quadrant, not a
  nby: u32,        // cellIndex() lookup) -- shared verbatim with the interp/
                   // step1 pool-parent shaders' uniform, not a near-duplicate.
  parentTau: f32,
  _pad: f32,
}

@group(0) @binding(0) var<uniform>             levelParams   : LevelParams;
@group(0) @binding(1) var<storage, read>       f_pool        : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_parent_pool : array<f32>;
@group(0) @binding(3) var<storage, read>       slotToBlock   : array<i32>;
@group(0) @binding(4) var<storage, read>       parentSlot    : array<i32>;
@group(0) @binding(5) var<storage, read>       quadrant      : array<u32>;

override RB : u32;
const GHOST = 2u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

fn feq(rho: f32, ux: f32, uy: f32, i: u32) -> f32 {
  let eu = f32(ex[i]) * ux + f32(ey[i]) * uy;
  return wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*(ux*ux+uy*uy));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let lcx = lid.x; let lcy = lid.y; // coarse-cell-local coords within this level's own footprint
  let slot = wgid.z;

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  let FB = RB * 2u + 2u * GHOST;
  let poolPlaneStride = arrayLength(&f_pool) / 9u;

  let fx0 = GHOST + 2u * lcx; let fx1 = fx0 + 1u;
  let fy0 = GHOST + 2u * lcy; let fy1 = fy0 + 1u;
  let children = array<u32, 4>(
    slot * (FB * FB) + fy0 * FB + fx0, slot * (FB * FB) + fy0 * FB + fx1,
    slot * (FB * FB) + fy1 * FB + fx0, slot * (FB * FB) + fy1 * FB + fx1
  );

  var rho_sum = 0f;
  var rhou_x_sum = 0f; var rhou_y_sum = 0f;
  var f_children: array<array<f32, 9>, 4>;
  var rho_children: array<f32, 4>;
  var ux_children: array<f32, 4>;
  var uy_children: array<f32, 4>;

  for (var c = 0u; c < 4u; c++) {
    let cell = children[c];
    var rho = 0f; var ux = 0f; var uy = 0f;
    var f: array<f32, 9>;
    for (var i = 0u; i < 9u; i++) {
      f[i] = f_pool[i * poolPlaneStride + cell];
      rho += f[i];
      ux  += f[i] * f32(ex[i]);
      uy  += f[i] * f32(ey[i]);
    }
    ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f);
    f_children[c] = f;
    rho_children[c] = rho; ux_children[c] = ux; uy_children[c] = uy;
    rho_sum += rho;
    rhou_x_sum += rho * ux;
    rhou_y_sum += rho * uy;
  }

  let rho_avg = rho_sum * 0.25f;
  let rho_sum_den = max(rho_sum, 1e-6f);
  let ux_avg = rhou_x_sum / rho_sum_den;
  let uy_avg = rhou_y_sum / rho_sum_den;

  // Relative to THIS level's own parent, not L0 -- see header.
  let tau_coarse = levelParams.parentTau;
  let tau_fine = 2.0f * tau_coarse - 0.5f;
  let rescale = 2.0f * tau_coarse / tau_fine;

  var fneq_avg: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    var s = 0f;
    for (var c = 0u; c < 4u; c++) {
      s += f_children[c][i] - feq(rho_children[c], ux_children[c], uy_children[c], i);
    }
    fneq_avg[i] = s * 0.25f;
  }

  // Destination: a specific (lx,ly) inside the PARENT's own tile, found via
  // this slot's parentSlot+quadrant (no cellIndex() -- the parent isn't the
  // dense grid, see file header).
  let pSlot = u32(parentSlot[slot]);
  let q = quadrant[slot];
  let qx = q & 1u;
  let qy = (q >> 1u) & 1u;
  let plx = qx * RB + lcx;
  let ply = qy * RB + lcy;
  let parentPlaneStride = arrayLength(&f_parent_pool) / 9u;
  let parentCell = pSlot * (FB * FB) + (ply + GHOST) * FB + (plx + GHOST);

  for (var i = 0u; i < 9u; i++) {
    f_parent_pool[i * parentPlaneStride + parentCell] = feq(rho_avg, ux_avg, uy_avg, i) + rescale * fneq_avg[i];
  }
}
