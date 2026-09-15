// The POOL-parent half of the interpolation accessor
// (plans/2D-backport.md B3-3) -- i.e. every L(m) -> L(m+1) hop with m>=1,
// where the "coarse" data is a SINGLE parent pool slot's own FB*FB tile.
//
// Five functions and nothing else; common_interp_kernel.wgsl is the kernel.
// See its header for the contract, and for why parentOrigin and sampleParent
// are one fragment: they must agree on a frame, and this one is PARENT-LOCAL
// INTERIOR coordinates.
//
// Reads `levelParams`, `f_parent_pool`, `parentSlot` and `quadrant`, all
// declared by the entry file (amr_interp_pool_parent.wgsl) -- a fragment must
// not @include, and WGSL module scope is order-independent.
//
// GENUINELY SIMPLER THAN THE DENSE HALF, for one structural reason: the data
// sampled here is one parent tile's own buffer (interior + its own
// already-ghost-filled border), never a second, different parent tile. A
// dense-parent lookup has to reach arbitrary coarse cells via
// cellIndex()+wrapCoord, because the coarse buffer holds the WHOLE domain.
// Here, because every level>=1 tile uses the identical RB/GHOST/FB shape
// (decision 2, plans/AMR-multilevel.md:9) and a child covers exactly one RB*RB
// quadrant of its parent's own 2*RB*2*RB interior, the full stencil radius
// fineToCoarseUnit can ever require (checked directly: for origin in {0, RB}
// and fx/fy spanning the child's whole FB-wide buffer, the resulting
// parent-local-interior index always lands in [-GHOST, 2*RB-1+GHOST]) maps
// 1:1, via a plain +GHOST offset, onto the parent's own already-valid [0,FB)
// index range -- including the parent's ghost cells. That is exactly why this
// needs no wrap and no second slot lookup.
//
// A quadtree child's parent identity is NOT derivable from anything else
// (unlike this level's own logical (bx,by), which IS derivable from
// slotToBlock[slot] + this level's own NBX -- see the kernel), so
// parentSlot/quadrant are the two genuinely new per-slot fields this side
// needs, written once at allocation time and read-only here.

// A mid-chain parent has its OWN tau: CardState carries only L0's, but e.g.
// L1 acting as parent to L2 has tau_fine = 2*tau_coarse - 0.5 applied
// recursively (card-params.mjs's tauAtLevel), supplied per child level as
// levelParams.parentTau rather than read from the domain-wide CardState.
fn parentTau() -> f32 { return levelParams.parentTau; }

// This level's own logical grid extent (plans/AMR-multilevel-M5.md's
// NBX[m]/NBY[m]) -- runtime uniform values, deliberately NOT `override`
// constants, since (unlike RB/GHOST, which are identical at every level) they
// differ per child level and one compiled pipeline serves all of them
// (decision 2).
fn levelNbx() -> u32 { return levelParams.nbx; }
fn levelNby() -> u32 { return levelParams.nby; }

// Origin: this child's own quadrant offset within its parent's 2*RB-wide
// interior -- 0 or RB per axis, from the quadrant bits alone, with no spatial
// coordinate involved (amr2d.mjs's quadrantOrigin is the host twin). The
// block's own (bx,by) plays no part here; the parent is found by slot, not by
// position.
fn parentOrigin(slot: u32, bx: u32, by: u32) -> vec2<u32> {
  let q = quadrant[slot];
  return vec2<u32>((q & 1u) * RB, ((q >> 1u) & 1u) * RB);
}

// Parent-local-interior coordinates (ix,iy) -> flat index into the parent's
// own FB*FB tile. No wrap, no neighbor-slot lookup -- see this file's header
// for why the +GHOST offset alone always lands in the parent's valid [0,FB)
// range for every (ix,iy) the kernel ever calls this with.
fn parentCellIndex(pSlot: u32, ix: i32, iy: i32) -> u32 {
  let FB = RB * 2u + 2u * GHOST;
  let fxp = u32(ix + i32(GHOST));
  let fyp = u32(iy + i32(GHOST));
  return pSlot * (FB * FB) + fyp * FB + fxp;
}

// CoarseSample lives in common_interp.wgsl, alongside the blend that consumes
// it -- this supplies only the FETCH.
fn sampleParent(slot: u32, ix: i32, iy: i32) -> CoarseSample {
  let pSlot = u32(parentSlot[slot]);
  let cell = parentCellIndex(pSlot, ix, iy);
  let parentPlaneStride = arrayLength(&f_parent_pool) / 9u;

  var f: array<f32, 9>;
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    f[i] = fUnpack(f_parent_pool[fIdx(i, parentPlaneStride, cell)], i);
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f); // NaN-containment floor

  var out: CoarseSample;
  out.rho = rho; out.ux = ux; out.uy = uy;
  for (var i = 0u; i < 9u; i++) {
    out.fneq[i] = f[i] - feqD2Q9(rho, ux, uy, i);
  }
  return out;
}
