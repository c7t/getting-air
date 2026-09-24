// EXPLODE: coarse -> fine at the seam (plans/2D-backport.md B6-1). The 2D port
// of the 3D fork's common_d3_amr_explode.wgsl -- Chen, Filippova, Hoch,
// Molvig, Shock, Teixeira & Zhang (2006), Physica A 362(1) 158-167, the
// PowerFLOW grid-refinement scheme. Read that file's header for the full
// argument; what follows is what the 2D version does and where it differs.
//
// WHAT AN EXPLODE IS. A child ring cell lies inside one PARENT cell v (GHOST 2
// is one parent cell deep, so the ring is v's 2x2 subdivision). If v is
// UNCOVERED -- the parent level still solves there -- then for each direction
// i whose parent target v + e_i IS covered, v's post-collision population has
// no parent destination (that cell's parent solution is a restriction nobody
// pulls), so it is placed in v's four children at UNCHANGED DENSITY: a plain
// copy, mass 1/4 each, totalling exactly what v held. No interpolation, no
// Dupuis-Chopard rescale: with tau_f = 2 tau_c - 1/2 (Chen's omega_f at n = 2)
// the scheme's own time offset absorbs the tau change, and the ring does not
// collide (COLLIDE_RING = 0 in amr_step1.wgsl), so there is nothing to convert.
//
// ONE EXPLODE FEEDS BOTH SUBSTEPS. Both ring depths get the same value, and
// the ring self-advects: after substep A depth 1 holds what depth 2 held, so
// substep B gathers the same state -- Chen's "reoccurrence".
//
// THE IN-ORPHAN (3D's M4.1c-i), handled by the SAME loop. If v + e_i is NOT
// covered the direction is not exploded -- but this particular child's own
// one-FINE-step destination g + e_i can still be covered, which happens only
// where the seam turns (a child straddles it on one axis and v does not). That
// child then gets f_i(v) too, and amr_coalesce.wgsl's orphan pass takes
// exactly that amount back off v. Uniform deliberately: only some children
// fire, so a linear term would not sum out.
//
// DIRECTIONS NOT EXPLODED ARE ZEROED. An interior fine cell only gathers from
// a ring cell along a direction pointing INTO the tile, which are exactly the
// exploded ones; the outward directions are refilled by the substeps and
// harvested by coalesce. Zero makes that checkable instead of merely true.
//
// 2D DIFFERENCE: THE PARENT IS READ THROUGH ITS OWNING TILE, not through the
// ring of the child's own parent tile. A pool parent tile carries a ring too,
// and under this scheme a ring is an inbox/outbox, not a distribution
// (plans/uniform-levels.md 2.5), so reading v out of it would hand the child
// whatever the parent's own seam last left there. `parentBlockSlot` names the
// tile that SOLVES v. The root is ringless and always full, so there the
// owner is its block index (PARENT_GHOST = 0, same fold as the interp
// accessor's).
//
// LINEAR EXPLOSION (B6-2, `EXPLODE_LINEAR`, ?explin=0 restores uniform) --
// 3D's M4.1c-ii. The uniform explosion is a piecewise-CONSTANT reconstruction:
// first order at the seam, and B6-1's field channel measured what that costs
// -- the error is almost entirely the mode's AMPLITUDE, i.e. the seam
// dissipates. Chen et al.'s second-order form is
//
//     f_i(child) = f_i(v) + (r_child - r_v) . G_i
//
// with G_i a central difference of f_i on the parent grid and the child offset
// +-1/4 of a parent cell per axis. Two constraints, both inherited:
//
//   1. ONLY AXES WITH TWO REAL, UNCOVERED PARENT NEIGHBOURS. A covered
//      neighbour's parent solution is a restriction nobody pulls (and its
//      slots may hold coalesced outbox values), and a pool parent two cells
//      out may have no tile at all. So the test is per axis: it excludes the
//      seam normal automatically (v + e_i is covered by definition) and
//      degrades to UNIFORM at a concave corner. No geometry is classified.
//   2. PROJECTED ORTHOGONAL TO e_i. One explosion feeds both substeps because
//      the exploded state RECURS under translation by -e_i, and a linear field
//      does so iff G_i . e_i = 0. A component along e_i would inject a
//      different value on each substep from one parent state -- a spurious
//      time variation at the seam, an acoustic source.
//
// CONSERVATION IS UNTOUCHED BY CONSTRUCTION: the four children's offsets sum
// to zero on each axis, and the two substeps draw from the two layers along
// e_i, whose offsets are equal and opposite. If a conservation gate moves when
// this is switched on, it is a bug here, not a tolerance to widen. The
// IN-orphan stays UNIFORM for the reason given above it.
override EXPLODE_LINEAR : u32 = 1u;

// Dispatched over (ceil(FB/8), ceil(FB/8), child slots), same shape as interp.

// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"

struct LevelParams {
  nbx: u32,        // the CHILD level's block grid -- see common_seam.wgsl's frames
  nby: u32,
  parentTau: f32,  // unused: nothing on this path is rescaled
  dxL: f32,
}

@group(0) @binding(0) var<uniform>             levelParams     : LevelParams;
@group(0) @binding(1) var<storage, read>       f_parent        : array<u32>; // parent at time t
@group(0) @binding(2) var<storage, read_write> f_pool          : array<u32>; // child, the buffer substep A reads
@group(0) @binding(3) var<storage, read>       slotToBlock     : array<i32>; // child
@group(0) @binding(4) var<storage, read>       blockSlot       : array<i32>; // child -- "covered"
@group(0) @binding(5) var<storage, read>       parentBlockSlot : array<i32>; // parent -- owner of a parent cell; unread at the root
@group(0) @binding(6) var<storage, read>       activeSlots     : array<u32>; // child -- read only by `mainIndirect`

override RB : u32;
override PARENT_GHOST : u32 = 2u;
const GHOST = 2u;

// @include "common_seam.wgsl"

// A WRAPPED coarse cell -> its index in the parent's storage, through the
// tile that owns it; -1 if no parent tile holds it (the ring-parent invariant
// says that cannot happen for a cell a live ring sits in -- see the caller).
fn parentCellIndex(c: vec2<i32>) -> i32 {
  let nbxP = seamNbxC() >> 1u;
  let pb = vec2<u32>(c) / (2u * RB);
  let pbid = pb.y * nbxP + pb.x;
  var pslot = i32(pbid);
  if (PARENT_GHOST != 0u) { pslot = parentBlockSlot[pbid]; }
  if (pslot < 0) { return -1; }
  let side = 2u * RB + 2u * PARENT_GHOST;
  let l = vec2<u32>(c) - pb * (2u * RB) + vec2<u32>(PARENT_GHOST);
  return i32(u32(pslot) * side * side + l.y * side + l.x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { explodeCell(gid, gid.z); }

// ?indirect=1 (main-amr.js): launched over this pool's active-slot list
// (amr_active_list.wgsl), so z indexes the list rather than the pool. `main`
// never reads `activeSlots`, so its layout -- every other page's -- is unchanged.
@compute @workgroup_size(8, 8)
fn mainIndirect(@builtin(global_invocation_id) gid: vec3<u32>) { explodeCell(gid, activeSlots[gid.z]); }

fn explodeCell(gid: vec3<u32>, slot: u32) {
  let fx = gid.x; let fy = gid.y;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }
  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }
  // Ring cells only. The interior is the child's own evolved state.
  let RB2 = RB * 2u;
  if (fx >= GHOST && fx < GHOST + RB2 && fy >= GHOST && fy < GHOST + RB2) { return; }

  let b = vec2<i32>(i32(u32(blockID) % seamNbxC()), i32(u32(blockID) / seamNbxC()));
  let g = wrapFine(b * i32(RB2) + vec2<i32>(i32(fx), i32(fy)) - i32(GHOST));
  let v = g / 2;
  // A ring cell inside a COVERED parent cell faces a same-level neighbour
  // tile; the step reaches through it directly (DIRECT_GHOST), so nothing is
  // exploded here and the cell is left exactly as interp wrote it -- which is
  // what a new tile's bilinear init reads.
  if (coveredCoarse(v)) { return; }

  let vIdx = parentCellIndex(v);
  if (vIdx < 0) { return; }
  let pplane = arrayLength(&f_parent) / 9u;
  var fv: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    fv[i] = fUnpack(f_parent[fIdx(i, pplane, u32(vIdx))], i);
  }

  // The four parent neighbours for the linear term, and whether each axis can
  // be central-differenced (both neighbours real and uncovered). Hoisted: they
  // depend on v alone.
  var okx = false; var oky = false;
  var ixm = -1; var ixp = -1; var iym = -1; var iyp = -1;
  if (EXPLODE_LINEAR != 0u) {
    let vxm = wrapCoarse(v + vec2<i32>(-1, 0)); let vxp = wrapCoarse(v + vec2<i32>(1, 0));
    let vym = wrapCoarse(v + vec2<i32>(0, -1)); let vyp = wrapCoarse(v + vec2<i32>(0, 1));
    ixm = parentCellIndex(vxm); ixp = parentCellIndex(vxp);
    iym = parentCellIndex(vym); iyp = parentCellIndex(vyp);
    okx = !coveredCoarse(vxm) && !coveredCoarse(vxp) && ixm >= 0 && ixp >= 0;
    oky = !coveredCoarse(vym) && !coveredCoarse(vyp) && iym >= 0 && iyp >= 0;
  }
  // This child's offset from v's centre, in PARENT cells: -1/4 or +1/4.
  let d = (vec2<f32>(g - v * 2) - 0.5f) * 0.5f;

  var fo: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) {
    let e = vec2<i32>(ex[i], ey[i]);
    if (coveredCoarseU(v + e)) {
      fo[i] = fv[i];                          // the explode proper
      if (EXPLODE_LINEAR != 0u && (okx || oky)) {
        var G = vec2<f32>(0f, 0f);
        if (okx) { G.x = 0.5f * (fUnpack(f_parent[fIdx(i, pplane, u32(ixp))], i) - fUnpack(f_parent[fIdx(i, pplane, u32(ixm))], i)); }
        if (oky) { G.y = 0.5f * (fUnpack(f_parent[fIdx(i, pplane, u32(iyp))], i) - fUnpack(f_parent[fIdx(i, pplane, u32(iym))], i)); }
        // i = 0 cannot reach here (e_0 = 0 lands on v, which is uncovered),
        // so |e_i|^2 >= 1.
        let ef = vec2<f32>(e);
        G -= ef * (dot(G, ef) / dot(ef, ef));
        fo[i] += dot(d, G);
      }
    } else {
      let dst = wrapFine(g + e);              // the IN-orphan test
      let intoTile = coveredCoarse(dst / 2) && all(blockOfFine(dst) == b);
      fo[i] = select(0f, fv[i], intoTile);
    }
  }

  let plane = arrayLength(&f_pool) / 9u;
  let cell = slot * (FB * FB) + fy * FB + fx;
  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    f_pool[wi * plane + cell] = fPack(fo[fLo(wi)], fo[fHi(wi)], wi);
  }
}
