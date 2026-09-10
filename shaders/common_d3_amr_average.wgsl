// Fine (L1 pool) -> coarse (L0 dense) restriction. plans/3D.md M3.
// Fragment only; the entry files list every include.
//
// Each thread owns one coarse cell and averages its EIGHT fine children
// (against 2D's four). Dispatched so one workgroup covers one block's
// RB^3 coarse cells.
//
//   rho  arithmetic mean of the 8 children -- exactly mass-conservative.
//   u    mass-weighted mean -- exactly momentum-conservative.
//   fneq mean, then rescaled by the exact INVERSE of the coarse->fine
//        factor -- dcRescaleF2C, whose derivation (and the post-collision
//        correction that is NOT the textbook form) is in
//        common_d3_pool.wgsl. Dropping it leaves the same O(1) stress
//        discontinuity at the interface that dropping the forward factor
//        does, just in the other direction.
//
// Conservation is the point: this is what keeps the coarse level's mass and
// momentum consistent with the fine solution living inside it, and it is
// why the restriction is written from the children's moments rather than by
// averaging f directly (averaging f would conserve mass and momentum too,
// but would NOT let fneq carry its own rescale).

@group(0) @binding(0) var<storage, read>       f_pool      : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_coarse    : array<f32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;
// The coarse macroscopic field, written here as well as the coarse `f`.
// Without this, `mac` under a refined region would keep whatever the COARSE
// step computed from its own (now superseded) populations, so the renderer
// and every readback would show the coarse solution in exactly the region
// that was refined to improve it. The averaged moments are already in hand,
// so this costs four stores.
@group(0) @binding(3) var<storage, read_write> mac_coarse  : array<f32>;

override TAU_COARSE : f32 = 0.8f;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  // One workgroup per (block, z sub-slab). RB may exceed the 4^3 workgroup,
  // so a block's RB^3 coarse cells are covered by ceil(RB/4)^3 workgroups;
  // x and y come from the dispatch directly and only z is folded in
  // alongside the slot, because 3D has no fourth dispatch dimension
  // (plans/3D.md sec 2.4). Dispatch is (per, per, per * slots).
  let per = (RB + 3u) / 4u;
  let slot = wgid.z / per;
  let lc = vec3<u32>(
    wgid.x * 4u + lid.x,
    wgid.y * 4u + lid.y,
    (wgid.z % per) * 4u + lid.z);
  if (lc.x >= RB || lc.y >= RB || lc.z >= RB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }
  let b = blockXYZ(u32(blockID));

  let poolPlane = arrayLength(&f_pool) / QN;
  let f0 = vec3<u32>(GHOST) + 2u * lc;   // first child, tile-local

  var rhoSum = 0f;
  var momSum = vec3<f32>(0f);
  var fc: array<array<f32, QN>, 8>;
  var rhoC: array<f32, 8>;
  var uC: array<vec3<f32>, 8>;

  for (var c = 0u; c < 8u; c++) {
    let off = vec3<u32>(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u);
    let cell = poolCell(slot, f0 + off);
    var rho = 0f;
    var m = vec3<f32>(0f);
    var f: array<f32, QN>;
    for (var i = 0u; i < QN; i++) {
      f[i] = f_pool[i * poolPlane + cell];
      rho += f[i];
      m += f[i] * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
    }
    let u = m / max(rho, 1e-6f);   // NaN-containment floor
    fc[c] = f; rhoC[c] = rho; uC[c] = u;
    rhoSum += rho;
    momSum += rho * u;
  }

  let rhoAvg = rhoSum * 0.125f;
  let uAvg = momSum / max(rhoSum, 1e-6f);

  let rescale = dcRescaleF2C(TAU_COARSE);

  var fo: array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    var s = 0f;
    for (var c = 0u; c < 8u; c++) {
      s += fc[c][i] - feqD3Q(rhoC[c], uC[c].x, uC[c].y, uC[c].z, i);
    }
    fo[i] = feqD3Q(rhoAvg, uAvg.x, uAvg.y, uAvg.z, i) + rescale * (s * 0.125f);
  }

  let cc = b * RB + lc;
  let ncells = NX * NY * NZ;
  let dst = coarseCell(cc);
  for (var i = 0u; i < QN; i++) { f_coarse[i * ncells + dst] = fo[i]; }
  mac_coarse[4u * dst + 0u] = rhoAvg;
  mac_coarse[4u * dst + 1u] = uAvg.x;
  mac_coarse[4u * dst + 2u] = uAvg.y;
  mac_coarse[4u * dst + 3u] = uAvg.z;
}
