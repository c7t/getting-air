// Fine -> coarse restriction: a level-m tile onto its PARENT's cells.
// plans/3D.md M3, made parent-generic by M5.5b. Fragment only; the entry
// files list every include.
//
// LEVEL-GENERIC, the same way interp and coalesce are: the parent is the
// dense L0 grid at level 1 and a pool of tiles below it, and the only thing
// that differs is how a parent cell is addressed. That is
// common_d3_parent_{dense,pool}.wgsl's whole job, so this file is assembled
// against one or the other and says nothing about which.
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
@group(0) @binding(1) var<storage, read_write> f_parent    : array<f32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;
// The criterion's answer, per block. Always bound -- so this layout does not
// fork -- and read only when DYING_ONLY folds the test in.
@group(0) @binding(3) var<storage, read>       blockWant   : array<u32>;
// binding 4 is the PARENT's macroscopic array, declared by whichever of
// common_d3_parentmac_{dense,pool}.wgsl this module was assembled with, and
// binding 8 (pool parent only) is the parent's blockSlot. This body cannot
// name either array and addresses the parent through parentIndex /
// parentMacStore, for the reason those files' headers give: the two
// macroscopic layouts are different AND their types are different, so a
// transposition is a compile error rather than a silent one.
//
// Writing `mac` at all is not optional. Without it the parent's macroscopic
// field under a refined region would keep whatever the parent's own step
// computed from its (now superseded) populations, so the renderer and every
// readback would show the coarse solution in exactly the region that was
// refined to improve it. The averaged moments are already in hand, so it
// costs four stores.

override TAU_COARSE : f32 = 0.8f;

// M4.2b-ii, the DRAIN pipeline. 1 restricts ONLY the tiles the manager is
// about to release, and it must run BEFORE the coarsen pass frees them:
// `refine` can hand the same slot straight back out in the next pass, and by
// then the fine solution is gone.
//
// This is the mirror of interp's NEW_ONLY, and a third pipeline over this
// same module for the same reason -- a tile being absorbed back into the
// coarse grid wants exactly the restriction `average` already performs, and
// a second copy of it is how the two would drift.
//
// THE DUPUIS-CHOPARD RESCALE IS CORRECT HERE. M4.2b-i's plan note said
// reaching for `average` was the obvious move and was wrong, on the grounds
// that Chen's scheme wants no rescale. That reasoning does not transfer:
// Chen's no-rescale is about the INTERFACE, where one state is moved between
// two bookkeepings of a single volume and the a = (n-1)/2n offset absorbs
// the tau difference. A tile being destroyed is an ordinary GRID TRANSFER --
// the coarse cell must carry the same rho, u and viscous stress the fine
// cells did, and fneq carries the stress, so it scales. The note was wrong
// and is corrected in plans/3D.md.
override DYING_ONLY : u32 = 0u;

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
  if (DYING_ONLY != 0u && blockWant[u32(blockID)] != 0u) { return; }
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

  // The PARENT cell this thread's eight children live in. parentIndex, not
  // coarseCell: a dense index at level 1, a tile lookup through the parent's
  // own blockSlot at level >= 2 (M5.2b), and the stride comes from the
  // binding for the same reason -- a pool parent is slots x tileCells, not a
  // grid.
  //
  // THE -1 BRANCH IS UNREACHABLE for the two callers that exist, and is a
  // return rather than a clamp anyway. A child tile exists because its
  // parent spawned it, so the parent of any tile this dispatch walks is
  // allocated; on the drain path M5.5b's ordering is what keeps it that way
  // while a subtree collapses (coarsen finest-first, so the parent is still
  // holding its slot when the child restricts into it).
  let cc = b * RB + lc;
  let pidx = parentIndex(cc);
  if (pidx < 0) { return; }
  let dst = u32(pidx);
  let ncells = arrayLength(&f_parent) / QN;
  for (var i = 0u; i < QN; i++) { f_parent[i * ncells + dst] = fo[i]; }
  parentMacStore(dst, vec4<f32>(rhoAvg, uAvg.x, uAvg.y, uAvg.z));
}
