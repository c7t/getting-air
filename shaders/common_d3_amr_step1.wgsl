// Fine-level (L1) LBM step over the octree pool. plans/3D.md M3.
// Fragment only; the entry files list every include.
//
// Same fused pull-stream + BGK + Guo + body coupling as
// common_d3_step.wgsl, but addressed through the pool and with
// NEIGHBOUR-ADDRESSED STREAMING: a source cell that leaves this tile's
// interior is resolved against the OWNING same-level tile directly, via
// blockSlot, rather than read out of a materialized ghost cell. Only where
// no such tile exists -- a coarse/fine interface -- does the gather fall
// back to the ring, which the interp pass filled from the parent.
//
// plans/3D.md sec 1.1 calls DIRECT_GHOST "the single most valuable thing
// the 2D work bought the 3D fork", and it is worth more here: it removes
// the between-substep fine-fine ghost copy entirely, and the copy pass was
// 10-13% of frame in 2D where the ring is a smaller fraction of a tile.
//
// D3Q19 NEVER NEEDS A CORNER TILE. Its 18 non-rest directions are 6 face +
// 12 edge, with no (+-1,+-1,+-1) among them, so a source displaces at most
// two axes and the corner neighbour is unreachable. D3Q27 does need it.
// That is a real simplification D3Q19 buys and D3Q27 gives back
// (plans/3D.md sec 2.3), and it is visible here as nbrXYZ being resolved
// but never selected at Q=19 -- the lattice's own shell ranges make it
// dead code the compiler folds out.

@group(0) @binding(0) var<storage, read>       f_in        : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out       : array<f32>;
@group(0) @binding(2) var<storage, read_write> mac_pool    : array<f32>;
@group(0) @binding(3) var<storage, read>       body        : BodyState3D;
@group(0) @binding(4) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(5) var<storage, read>       blockSlot   : array<i32>;

// The FINE level's own relaxation rate, 1/tau_fine with
// tau_fine = 2*tau_coarse - 0.5. Passed in rather than derived so there is
// one place (main-3d.js) that owns the level->tau mapping.
override OMEGA_FINE : f32 = 1.0f;

override FORCE_X : f32 = 0.0f;
override FORCE_Y : f32 = 0.0f;
override FORCE_Z : f32 = 0.0f;
override HAS_BODY : u32 = 0u;
override USE_BOUNCEBACK : u32 = 0u;
// Chi band in FINE cells. The band is a physical width, and a fine cell is
// half a coarse cell, so the same physical band is 2x as many fine cells --
// which is exactly the correction the 2D solver's K_EPS * dx_L1 makes, and
// exactly the thing plans/3D.md's M3 note warns will bite where a refined
// shell wraps a body. Expressed in COARSE units by the caller and halved
// here, so one number means one physical width at every level.
override CHI_EPS : f32 = 1.5f;

override SPONGE_W : f32 = 0.0f;
override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;
override SPONGE_UZ : f32 = 0.0f;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let FB = poolFB();
  let fz = gid.z % FB;
  let slot = gid.z / FB;
  if (gid.x >= FB || gid.y >= FB) { return; }
  let fi = vec3<u32>(gid.x, gid.y, fz);

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  // THE WHOLE TILE IS STEPPED, RING INCLUDED. This looks like waste -- at
  // RB=4 the ring is 70% of a tile (1216 of 1728 cells) and only the
  // interior is the tile's own solution -- but it is the mechanism GHOST=2
  // exists for, and stepping the interior alone is silently wrong.
  //
  // Why: the pool ping-pongs across the two fine substeps, A -> B -> A. If a
  // substep writes only interiors, the buffer it wrote has an UNINITIALIZED
  // ring, and the next substep's interface gather -- the one case the ring
  // still serves after DIRECT_GHOST -- reads it. That is not a small error;
  // it is NaN on the first macro-step. It also hides completely under
  // ?refine=all, where every neighbour tile exists and no ring cell is ever
  // read, which is exactly how it was found: `all` was clean to 1.6e-3
  // against the analytic solution while a partially-refined box was NaN.
  //
  // So the ring SELF-ADVANCES: substep A's gather on a depth-1 ring cell
  // reaches into depth 2, leaving depth 1 valid for substep B. A depth-2
  // cell's own sources fall outside the tile and clamp, so depth 2 is
  // degraded after substep A -- which is fine, because nothing reads it
  // again before interp refreshes the whole ring next macro-step.
  // plans/3D.md sec 2.1 records the same reasoning for the 2D solver, and
  // its "don't STEP the ghost cells" lead is precisely about recovering
  // this cost without giving up the mechanism.

  let b = blockXYZ(u32(blockID));
  let origin = b * RB;
  let RB2 = 2u * RB;

  // Neighbour-slot resolution, hoisted out of the QN-direction gather.
  // A source is at most one cell away and an interior is RB2 >= 2 wide, so
  // at most ONE non-zero block offset is reachable per axis -- the whole
  // gather needs at most SEVEN neighbour tiles (3 face, 3 edge, 1 corner),
  // resolved once here instead of re-resolved with a fresh blockSlot load on
  // every direction. A fully-interior thread loads nothing at all.
  var off = vec3<i32>(0, 0, 0);
  off.x = select(select(0, 1, fi.x + 1u >= GHOST + RB2), -1, fi.x <= GHOST);
  off.y = select(select(0, 1, fi.y + 1u >= GHOST + RB2), -1, fi.y <= GHOST);
  off.z = select(select(0, 1, fi.z + 1u >= GHOST + RB2), -1, fi.z <= GHOST);
  // The block grid is periodic, matching the dense level's own periodicity.
  let nb = vec3<u32>(nbx(), nby(), nbz());
  let bn = vec3<u32>(
    wrapu(i32(b.x) + off.x, nb.x),
    wrapu(i32(b.y) + off.y, nb.y),
    wrapu(i32(b.z) + off.z, nb.z));

  var nbrX = -1; var nbrY = -1; var nbrZ = -1;
  var nbrXY = -1; var nbrXZ = -1; var nbrYZ = -1; var nbrXYZ = -1;
  if (off.x != 0) { nbrX = blockSlot[blockIdOf(vec3<u32>(bn.x, b.y, b.z))]; }
  if (off.y != 0) { nbrY = blockSlot[blockIdOf(vec3<u32>(b.x, bn.y, b.z))]; }
  if (off.z != 0) { nbrZ = blockSlot[blockIdOf(vec3<u32>(b.x, b.y, bn.z))]; }
  if (off.x != 0 && off.y != 0) { nbrXY = blockSlot[blockIdOf(vec3<u32>(bn.x, bn.y, b.z))]; }
  if (off.x != 0 && off.z != 0) { nbrXZ = blockSlot[blockIdOf(vec3<u32>(bn.x, b.y, bn.z))]; }
  if (off.y != 0 && off.z != 0) { nbrYZ = blockSlot[blockIdOf(vec3<u32>(b.x, bn.y, bn.z))]; }
  if (off.x != 0 && off.y != 0 && off.z != 0) { nbrXYZ = blockSlot[blockIdOf(bn)]; }

  let poolPlane = arrayLength(&f_in) / QN;
  let cell = poolCell(slot, fi);

  // Body position, in COARSE units -- the SDF and every body quantity live
  // in the dense level's coordinates, so the fine level converts rather than
  // keeping a second frame.
  let p = vec3<f32>(
    fineToCoarseUnit3(i32(fi.x), origin.x),
    fineToCoarseUnit3(i32(fi.y), origin.y),
    fineToCoarseUnit3(i32(fi.z), origin.z));
  let phi = select(1e30f, get_phi3(p, body), HAS_BODY != 0u);
  let us = select(vec3<f32>(0f), bodyVelocity3(p, body), HAS_BODY != 0u);

  var f: array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    let ei = vec3<i32>(ex[i], ey[i], ez[i]);
    var s = vec3<i32>(fi) - ei;

    if (HAS_BODY != 0u && USE_BOUNCEBACK != 0u) {
      // The sharp inside test uses the UNCLAMPED, geometrically-correct
      // source position, while the reflected VALUE comes from this cell's
      // own (always valid) data -- the same distinction the 2D fine step
      // draws, and for the same reason: a tile edge is a buffer-addressing
      // artifact, not a physical statement.
      let sp = vec3<f32>(
        fineToCoarseUnit3(s.x, origin.x),
        fineToCoarseUnit3(s.y, origin.y),
        fineToCoarseUnit3(s.z, origin.z));
      if (get_phi3(sp, body) < 0f) {
        let corr = 2f * wt[i] * dot(vec3<f32>(f32(ei.x), f32(ei.y), f32(ei.z)), us) / CS2;
        f[i] = f_in[opp[i] * poolPlane + cell] + corr;
        continue;
      }
    }

    // Does the source leave the interior, and on which axes? If so the
    // offset can only be off.{x,y,z} (see the hoist above), so this is pure
    // register work -- no second blockSlot load.
    let ox = select(0, off.x, s.x < i32(GHOST) || s.x >= i32(GHOST + RB2));
    let oy = select(0, off.y, s.y < i32(GHOST) || s.y >= i32(GHOST + RB2));
    let oz = select(0, off.z, s.z < i32(GHOST) || s.z >= i32(GHOST + RB2));
    var ns = -1;
    if (ox != 0 && oy != 0 && oz != 0) { ns = nbrXYZ; }
    else if (ox != 0 && oy != 0) { ns = nbrXY; }
    else if (ox != 0 && oz != 0) { ns = nbrXZ; }
    else if (oy != 0 && oz != 0) { ns = nbrYZ; }
    else if (ox != 0) { ns = nbrX; }
    else if (oy != 0) { ns = nbrY; }
    else if (oz != 0) { ns = nbrZ; }

    var srcSlot = slot;
    if (ns >= 0) {
      // Re-express the source in the neighbour's own local frame. It always
      // lands in ITS INTERIOR: s in [-1, GHOST-1] maps to [RB2-1, RB2+1],
      // and s in [GHOST+RB2, FB] maps to [GHOST, GHOST+2]. So this reads
      // real same-level data, never the neighbour's own ring -- which is
      // the property that makes the fine-fine ghost copy unnecessary, and
      // which tools/test-d3-amr.js asserts directly.
      srcSlot = u32(ns);
      s -= vec3<i32>(ox, oy, oz) * i32(RB2);
    }
    // With no owning tile this is a coarse/fine interface and the ring is
    // the fallback -- the ring's one remaining job after DIRECT_GHOST. The
    // clamp is a no-op on the neighbour path (see above) and only bites
    // here, where the source may sit one past the ring at a corner.
    let sc = vec3<u32>(
      u32(clamp(s.x, 0, i32(FB) - 1)),
      u32(clamp(s.y, 0, i32(FB) - 1)),
      u32(clamp(s.z, 0, i32(FB) - 1)));
    f[i] = f_in[i * poolPlane + poolCell(srcSlot, sc)];
  }

  var rho = 0f;
  var m = vec3<f32>(0f);
  for (var i = 0u; i < QN; i++) {
    rho += f[i];
    m += f[i] * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
  }
  let rhoDen = max(rho, 1e-6f);
  let ustar = m / rhoDen;

  // CHI_EPS arrives in COARSE units and is halved: a fine cell is half a
  // coarse cell, so the same physical band width is twice as many fine
  // cells. Getting this wrong makes the solid boundary sharper or blurrier
  // on the refined level than on the level around it -- a discontinuity in
  // the body itself at the interface.
  let chi = select(0f, chiFromPhiEps3(phi, CHI_EPS * 0.5f),
                   HAS_BODY != 0u && USE_BOUNCEBACK == 0u);
  let F = rho * chi * (us - ustar) + vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
  let u = ustar + F / (2.0f * rhoDen);
  let u_sq = dot(u, u);

  let macPlane = arrayLength(&mac_pool) / 4u;
  mac_pool[0u * macPlane + cell] = rho;
  mac_pool[1u * macPlane + cell] = u.x;
  mac_pool[2u * macPlane + cell] = u.y;
  mac_pool[3u * macPlane + cell] = u.z;

  // Sponge distances are measured in COARSE units against the L0 domain, so
  // a refined tile near a domain face absorbs exactly as its parent does.
  let spongeW = spongeWeight3(
    min(p.x, f32(NX - 1u) - p.x),
    min(p.y, f32(NY - 1u) - p.y),
    min(p.z, f32(NZ - 1u) - p.z), SPONGE_W);

  var fo: array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    let ei = vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
    let eu = dot(ei, u);
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);
    let t1 = (ei - u) * 3.0f;
    let Si = (1.0f - 0.5f * OMEGA_FINE) * wt[i] * dot(t1 + (eu * 9.0f) * ei, F);
    let fCollide = f[i] - OMEGA_FINE * (f[i] - feq) + Si;
    let euFar = ei.x*SPONGE_UX + ei.y*SPONGE_UY + ei.z*SPONGE_UZ;
    let uFarSq = SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY + SPONGE_UZ*SPONGE_UZ;
    let fTarget = wt[i] * (1.0f + 3.0f*euFar + 4.5f*euFar*euFar - 1.5f*uFarSq);
    fo[i] = mix(fCollide, fTarget, spongeW);
  }
  for (var i = 0u; i < QN; i++) { f_out[i * poolPlane + cell] = fo[i]; }
}
