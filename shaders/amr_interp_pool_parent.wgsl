// Milestone 6 (plans/AMR-multilevel.md/-M5.md): parent-pool -> child-pool
// ghost-cell interpolation, for every L(m)->L(m+1) hop with m>=1 (the
// quadtree side of the addressing split -- see amr_interp_dense_parent.wgsl,
// which is the L0->L1, dense-buffer-parent sibling of this shader and
// shares its interpolation math verbatim).
//
// Genuinely simpler than the dense-parent case, for one structural reason:
// the "coarse" data this shader samples is a SINGLE parent pool slot's own
// FB*FB buffer (interior + its own already-ghost-filled border), never a
// second, different parent tile. A dense-parent lookup has to reach
// arbitrary coarse cells via cellIndex()+wrapCoord (the coarse buffer holds
// the WHOLE domain). Here, because every level>=1 tile uses the identical
// RB/GHOST/FB shape (decision 2, plans/AMR-multilevel.md:9) and a child
// covers exactly one RB*RB quadrant of its parent's own 2*RB*2*RB interior,
// the full stencil radius fineToCoarseUnit() below can ever require (checked
// directly: for origin in {0, RB} and fx/fy spanning the child's whole
// FB-wide buffer, the resulting parent-local-interior index always lands in
// [-GHOST, 2*RB-1+GHOST]) maps 1:1, via a plain +GHOST offset, onto the
// parent's own already-valid [0,FB) index range -- including the parent's
// ghost cells, which is exactly why this never needs a second slot lookup
// or a periodic wrap the way the dense case's sampleCoarse() does.
//
// A quadtree child's parent identity is NOT derivable from anything else
// (unlike this level's own logical (bx,by), which IS derivable from
// slotToBlock[slot] + this level's own NBX, exactly as the dense shader
// already derives it -- see main()) -- so parentSlot/quadrant are the two
// genuinely new per-slot fields this shader needs (bindings 6/7), written
// once at allocation time (see main-amr.js's quad-activation code) and
// read-only here.
//
// Same-level (sibling) fine-fine ghost consultation (including diagonal
// corners) is ported VERBATIM from amr_interp_dense_parent.wgsl -- it's a
// same-level concern, indifferent to how the parent hop works, and this
// level's own blockSlot/NBX/NBY (the levelParams uniform, binding 0) are
// exactly the inputs it already needed.
//
// levelParams.parentTau replaces the dense shader's `state.tau` read:
// CardState carries only L0's own tau, but a mid-chain parent (e.g. L1
// acting as parent to L2) has its OWN tau (tau_fine = 2*tau_coarse - 0.5,
// applied recursively -- see main-amr.js's tauAtLevel()), which has to be
// supplied per child level rather than read from the single domain-wide
// CardState field. levelParams.nbx/nby are this level's own logical grid
// extent (plans/AMR-multilevel-M5.md's NBX[m]/NBY[m]) -- runtime uniform
// values, deliberately NOT `override` constants, since (unlike RB/GHOST,
// which are identical at every level) NBX/NBY differ per child level and
// this one compiled pipeline is reused across all of them (decision 2).

struct LevelParams {
  nbx: u32,
  nby: u32,
  parentTau: f32,
  _pad: f32,
}

@group(0) @binding(0) var<uniform>             levelParams    : LevelParams;
@group(0) @binding(1) var<storage, read>       f_parent_pool  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_pool         : array<f32>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
@group(0) @binding(4) var<storage, read>       newlyActivated : array<u32>;
@group(0) @binding(5) var<storage, read>       blockSlot      : array<i32>;
@group(0) @binding(6) var<storage, read>       parentSlot     : array<i32>;
@group(0) @binding(7) var<storage, read>       quadrant       : array<u32>;

override RB : u32;  // identical at every level (decision 2) -- refine block size in coarse-equivalent units
override GHOST_ONLY : u32;
// See amr_interp_dense_parent.wgsl's header for what this mode does --
// ported unchanged, same same-level-only early return.
override FINE_FINE_ONLY : u32 = 0u;

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

// Fine ghost-local coordinate -> position in parent-local-interior units
// (identical formula to amr_interp_dense_parent.wgsl's fineToCoarseUnit;
// `origin` there was a buffer-space coarse coordinate, here it's 0 or RB
// -- the quadrant's own offset within the parent's 2*RB-wide interior).
fn fineToCoarseUnit(fCoord: u32, origin: u32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

fn wrapCoord(v: i32, n: u32) -> u32 {
  let m = i32(n);
  return u32(((v % m) + m) % m);
}

// Parent-local-interior-unit coordinates (ix,iy) -> flat index into the
// parent's own FB*FB tile at `pSlot`. No wrap, no neighbor-slot lookup --
// see the file header for why the +GHOST offset alone always lands in the
// parent's own valid [0,FB) range for every (ix,iy) this shader ever calls
// this with.
fn parentCellIndex(pSlot: u32, ix: i32, iy: i32) -> u32 {
  let FB = RB * 2u + 2u * GHOST;
  let fxp = u32(ix + i32(GHOST));
  let fyp = u32(iy + i32(GHOST));
  return pSlot * (FB * FB) + fyp * FB + fxp;
}

struct CoarseSample {
  rho: f32,
  ux: f32,
  uy: f32,
  fneq: array<f32, 9>,
}
fn sampleParentPool(pSlot: u32, ix: i32, iy: i32) -> CoarseSample {
  let cell = parentCellIndex(pSlot, ix, iy);
  let parentPlaneStride = arrayLength(&f_parent_pool) / 9u;

  var f: array<f32, 9>;
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    f[i] = f_parent_pool[i * parentPlaneStride + cell];
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f); // NaN-containment floor, matching the dense shader

  var out: CoarseSample;
  out.rho = rho; out.ux = ux; out.uy = uy;
  for (var i = 0u; i < 9u; i++) {
    out.fneq[i] = f[i] - feq(rho, ux, uy, i);
  }
  return out;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y;
  let slot = gid.z;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; } // slot not currently assigned

  let isInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;
  if (isInterior && GHOST_ONLY != 0u) { return; }
  if (GHOST_ONLY == 0u && newlyActivated[slot] == 0u) { return; }

  let nbx = levelParams.nbx;
  let nby = levelParams.nby;
  // This level's own logical (bx,by), derived from blockID exactly like
  // amr_interp_dense_parent.wgsl derives it -- blockID already encodes
  // by*nbx+bx by construction (see main-amr.js's quad-activation code),
  // so no separate cached ownBX/ownBY array is needed (a redundant extra
  // buffer M5's first draft allocated and M6 removed -- see
  // plans/AMR-multilevel-M5.md's amendment note).
  let bx = u32(blockID) % nbx;
  let by = u32(blockID) / nbx;

  // Same-level fine-fine + diagonal-corner ghost consultation, ported
  // verbatim from amr_interp_dense_parent.wgsl (see that file's header for
  // why diagonal corners need their own case, not just the two edges).
  let RB2 = RB * 2u;
  let inXRange = fx >= GHOST && fx < GHOST + RB2;
  let inYRange = fy >= GHOST && fy < GHOST + RB2;
  if (GHOST_ONLY != 0u) {
    var neighborSlot = -1;
    var nfx = fx; var nfy = fy;
    if (inYRange && fx < GHOST) {
      neighborSlot = blockSlot[by * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2;
    } else if (inYRange && fx >= GHOST + RB2) {
      neighborSlot = blockSlot[by * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2;
    } else if (inXRange && fy < GHOST) {
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + bx];
      nfy = fy + RB2;
    } else if (inXRange && fy >= GHOST + RB2) {
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + bx];
      nfy = fy - RB2;
    } else if (fx < GHOST && fy < GHOST) {
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2; nfy = fy + RB2;
    } else if (fx >= GHOST + RB2 && fy < GHOST) {
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2; nfy = fy + RB2;
    } else if (fx < GHOST && fy >= GHOST + RB2) {
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2; nfy = fy - RB2;
    } else if (fx >= GHOST + RB2 && fy >= GHOST + RB2) {
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2; nfy = fy - RB2;
    }

    if (neighborSlot >= 0) {
      let poolPlaneStride = arrayLength(&f_pool) / 9u;
      let poolCellBase = slot * (FB * FB) + fy * FB + fx;
      let neighborCellBase = u32(neighborSlot) * (FB * FB) + nfy * FB + nfx;
      for (var i = 0u; i < 9u; i++) {
        f_pool[i * poolPlaneStride + poolCellBase] = f_pool[i * poolPlaneStride + neighborCellBase];
      }
      return;
    }
  }

  if (FINE_FINE_ONLY != 0u) { return; }

  // The actual generalization vs. the dense shader: parent lookup via
  // parentSlot+quadrant instead of cellIndex().
  let pSlot = u32(parentSlot[slot]);
  let q = quadrant[slot];
  let qx = q & 1u;
  let qy = (q >> 1u) & 1u;
  let originX = qx * RB;
  let originY = qy * RB;

  let px = fineToCoarseUnit(fx, originX);
  let py = fineToCoarseUnit(fy, originY);

  let x0 = i32(floor(px)); let x1 = x0 + 1;
  let y0 = i32(floor(py)); let y1 = y0 + 1;
  let tx = px - f32(x0);
  let ty = py - f32(y0);

  let s00 = sampleParentPool(pSlot, x0, y0);
  let s10 = sampleParentPool(pSlot, x1, y0);
  let s01 = sampleParentPool(pSlot, x0, y1);
  let s11 = sampleParentPool(pSlot, x1, y1);

  let w00 = (1f - tx) * (1f - ty);
  let w10 = tx * (1f - ty);
  let w01 = (1f - tx) * ty;
  let w11 = tx * ty;

  let rho = w00*s00.rho + w10*s10.rho + w01*s01.rho + w11*s11.rho;
  let ux  = w00*s00.ux  + w10*s10.ux  + w01*s01.ux  + w11*s11.ux;
  let uy  = w00*s00.uy  + w10*s10.uy  + w01*s01.uy  + w11*s11.uy;

  let tau_coarse = levelParams.parentTau;
  let tau_fine = 2.0f * tau_coarse - 0.5f;
  // Dupuis-Chopard non-equilibrium rescale, same factor/reasoning as
  // amr_interp_dense_parent.wgsl -- see that file for the derivation.
  let rescale = 0.5f * tau_fine / tau_coarse;

  let poolPlaneStride = arrayLength(&f_pool) / 9u;
  let poolCellBase = slot * (FB * FB) + fy * FB + fx;
  for (var i = 0u; i < 9u; i++) {
    let fneq = w00*s00.fneq[i] + w10*s10.fneq[i] + w01*s01.fneq[i] + w11*s11.fneq[i];
    f_pool[i * poolPlaneStride + poolCellBase] = feq(rho, ux, uy, i) + rescale * fneq;
  }
}
