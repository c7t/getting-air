// COALESCE: fine -> coarse at the seam (plans/2D-backport.md B6-1). The 2D port
// of the 3D fork's common_d3_amr_coalesce.wgsl, the other half of Chen et al.
// (2006); read that file's header for the full argument.
//
// WHAT A COALESCE IS. Over the child's two substeps, populations leaving the
// refined region advect into the child's ring -- the 2x2 subdivision of the
// first UNCOVERED parent cell outside. After substep B the two ring depths hold
// the outflux of the two substeps, one each (the ring advects without
// colliding, COLLIDE_RING = 0), so the parent population is their MEAN:
//
//     f_i(parent) = (1/4) * sum over the 4 ring cells covering it of f_i
//
// 1/4 is the 2D cell-volume ratio, not a weight. No rescale.
//
// WHERE IT IS WRITTEN, which is the load-bearing trick. The value is what the
// covered cell c = t - e_i would hand to its uncovered neighbour t, so it goes
// into c's OWN slot i in the parent's TIME-t buffer, and the unmodified parent
// step -- which runs AFTER this (makeScheduler, `explode`) -- pulls it from c
// exactly as it pulls anything else. No per-direction test in the hot kernel.
// For a given (c, i) exactly one cell, t, pulls from it, so no two writes meet.
//
// THE ORPHAN PASS, on the UNCOVERED cells. A ring cell p holds, after substep
// B, its source's outflux at t (depth 2) or at t + dt/2 (depth 1). The coalesce
// proper claims p in direction i from coarse(p) - e_i when that is covered.
// For the t + dt/2 value the source moved only ONE fine cell, so where the seam
// turns coarse(p) - e_i can be UNCOVERED -- nobody would claim p and the mass
// would vanish at the next explode (3D measured it: -1.2e-2 per edge cell per
// step). Here the uncovered cell c claims it, reading p from the tile that owns
// its source q = p - e_i, and ADDS it to its own slot i: mass that transited c
// on its way out. Net of the IN-ORPHANS amr_explode.wgsl injected from c into
// children whose one-fine-step destination is covered, so in a uniform flow the
// two cancel exactly -- conservation AND consistency, which 3D found are
// separate requirements that a conservation gate alone cannot tell apart.
// Same test, opposite sign, as the coalesce proper: no double counting.
//
// NOT gathering by ORIGIN. Also exactly conservative, and 3D measured it 4.6x
// worse in the field: it hands a coarse cell mass that sits in a neighbour.
//
// EVERY WRITE IS A WHOLE CELL (common_fpack.wgsl's rule under ?f16=): each
// thread owns its cell c, reads all nine, replaces some, writes all nine.
//
// Dispatched over PARENT tiles: (ceil(2RB/8), ceil(2RB/8), parent slots), one
// thread per parent INTERIOR cell. A parent's ring is not a cell it solves.

// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"

struct LevelParams {
  nbx: u32,        // the CHILD level's block grid -- see common_seam.wgsl
  nby: u32,
  parentTau: f32,
  dxL: f32,
}

@group(0) @binding(0) var<uniform>             levelParams       : LevelParams;
@group(0) @binding(1) var<storage, read>       f_pool            : array<u32>; // child, after both substeps
@group(0) @binding(2) var<storage, read_write> f_parent          : array<u32>; // parent's TIME-t buffer
@group(0) @binding(3) var<storage, read>       blockSlot         : array<i32>; // child -- "covered", and the ring's owner
@group(0) @binding(4) var<storage, read>       parentSlotToBlock : array<i32>; // parent -- which block this thread's slot is
@group(0) @binding(5) var<storage, read>       parentActiveSlots : array<u32>; // parent -- read only by `mainIndirect`

override RB : u32;
override PARENT_GHOST : u32 = 2u;
const GHOST = 2u;

// @include "common_seam.wgsl"

// A child-tile cell, by slot and TILE-LOCAL (ring-inclusive) position.
fn childCell(s: u32, l: vec2<i32>) -> u32 {
  let FB = RB * 2u + 2u * GHOST;
  return s * (FB * FB) + u32(l.y) * FB + u32(l.x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { coalesceCell(gid, gid.z); }

// ?indirect=1 (main-amr.js): launched over this pool's active-slot list
// (amr_active_list.wgsl), so z indexes the list rather than the pool. `main`
// never reads `activeSlots`, so its layout -- every other page's -- is unchanged.
@compute @workgroup_size(8, 8)
fn mainIndirect(@builtin(global_invocation_id) gid: vec3<u32>) { coalesceCell(gid, parentActiveSlots[gid.z]); }

fn coalesceCell(gid: vec3<u32>, pslot: u32) {
  let RB2 = RB * 2u;
  if (gid.x >= RB2 || gid.y >= RB2) { return; }
  if (pslot >= arrayLength(&parentSlotToBlock)) { return; }
  let pbid = parentSlotToBlock[pslot];
  if (pbid < 0) { return; }
  let nbxP = seamNbxC() >> 1u;
  let pb = vec2<i32>(i32(u32(pbid) % nbxP), i32(u32(pbid) / nbxP));
  let c = pb * i32(RB2) + vec2<i32>(i32(gid.x), i32(gid.y));   // wrapped already

  let side = RB2 + 2u * PARENT_GHOST;
  let cIdx = pslot * side * side + (gid.y + PARENT_GHOST) * side + gid.x + PARENT_GHOST;
  let pplane = arrayLength(&f_parent) / 9u;
  let cplane = arrayLength(&f_pool) / 9u;

  var fc: array<f32, 9>;
  var changed = false;

  if (coveredCoarse(c)) {
    // --- the coalesce proper --------------------------------------------------
    // The target t is at most one parent cell outside c's block, so its four
    // fine cells are c's own tile's ring. Local offsets from UNWRAPPED
    // positions, so a tile at the domain edge stays contiguous.
    let cb = c / i32(RB);
    let cslot = blockSlot[u32(cb.y) * seamNbxC() + u32(cb.x)];
    if (cslot < 0) { return; }
    for (var i = 0u; i < 9u; i++) { fc[i] = fUnpack(f_parent[fIdx(i, pplane, cIdx)], i); }
    for (var i = 0u; i < 9u; i++) {
      let e = vec2<i32>(ex[i], ey[i]);
      let tU = c + e;
      if (coveredCoarseU(tU)) { continue; }   // no parent cell waiting to pull; i = 0 lands here
      var s = 0f;
      for (var k = 0; k < 4; k++) {
        let pU = tU * 2 + vec2<i32>(k & 1, k >> 1);
        let l = pU - cb * i32(RB2) + i32(GHOST);
        s += fUnpack(f_pool[fIdx(i, cplane, childCell(u32(cslot), l))], i);
      }
      fc[i] = s * 0.25f;
      changed = true;
    }
  } else {
    // --- the orphan pass ------------------------------------------------------
    // Both orphan kinds need a covered cell among c's own lattice neighbours
    // (the 3D file's guard proof carries over: every sub-vector of a D2Q9
    // direction is a D2Q9 direction or zero), so everything else leaves now.
    var near = false;
    for (var i = 1u; i < 9u; i++) { near = near || coveredCoarseU(c + vec2<i32>(ex[i], ey[i])); }
    if (!near) { return; }
    for (var i = 0u; i < 9u; i++) { fc[i] = fUnpack(f_parent[fIdx(i, pplane, cIdx)], i); }
    for (var i = 1u; i < 9u; i++) {
      let e = vec2<i32>(ex[i], ey[i]);
      let tU = c + e;
      // A covered target: this slot is exploded rather than pulled; nothing here.
      if (coveredCoarseU(tU)) { continue; }
      var s = 0f;
      var nIn = 0f;
      for (var k = 0; k < 4; k++) {
        let off = vec2<i32>(k & 1, k >> 1);
        // OUT: fine cell p of t, whose source q = p - e_i may sit in a tile.
        let pU = tU * 2 + off;
        let q = wrapFine(pU - e);
        let qb = blockOfFine(q);
        let qs = blockSlot[u32(qb.y) * seamNbxC() + u32(qb.x)];
        if (qs >= 0) {
          let l = q - qb * i32(RB2) + i32(GHOST) + e;   // p, in q's tile
          s += fUnpack(f_pool[fIdx(i, cplane, childCell(u32(qs), l))], i);
        }
        // IN: c's child 2c + off, one fine step along e_i, into a covered cell.
        let ch = wrapFine(c * 2 + off + e);
        if (coveredCoarse(ch / 2)) { nIn += 1f; }
      }
      let net = s - nIn * fc[i];
      if (net != 0f) { fc[i] += net * 0.25f; changed = true; }
    }
  }

  if (!changed) { return; }
  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    f_parent[wi * pplane + cIdx] = fPack(fc[fLo(wi)], fc[fHi(wi)], wi);
  }
}
