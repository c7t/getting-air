// The DENSE-parent half of the interpolation accessor
// (plans/2D-backport.md B3-3) -- i.e. the L0 -> L1 hop, where the "coarse"
// data is the dense, cellIndex()-addressed L0 buffer covering the WHOLE
// periodic domain.
//
// Five functions and nothing else; common_interp_kernel.wgsl is the kernel.
// See its header for the contract, and in particular for why parentOrigin and
// sampleParent are one fragment: they must agree on a frame, and this one is
// coarse BUFFER coordinates.
//
// Reads `state` and `f_coarse`, and the W/H overrides, all declared by the
// entry file (amr_interp_dense_parent.wgsl) -- a fragment must not @include,
// and WGSL module scope is order-independent.
//
// Buffer-space native: L0's coarse blocks are defined in buffer space (fixed
// in memory regardless of the moving window's off_x/off_y), so a pool slot's
// coarse-cell lookups need no window conversion at all -- only the (separate,
// in amr_step1.wgsl) card SDF physics needs window coordinates.
//
// KNOWN GAP (channel-flow/WALL_Y scenarios, shaders/common_walls.wgsl):
// wrapCoord assumes the coarse level is periodic in BOTH x and y. A fine block
// refined adjacent to a real y=0/y=H-1 wall would have its ghost cells sample
// the periodic image across the domain instead of reflecting off the wall --
// wrong physics, not just imprecise. Not fixed yet: main-channel-amr.js's
// default refine thresholds are deliberately kept high enough that refinement
// never reaches the wall-adjacent blocks, so this stays latent rather than
// exercised. Fix properly (wall-aware ghost rule, not a periodic wrap) before
// validating any AMR channel-flow case that refines near a wall.

// Level 1's parent is L0, whose tau IS the card state's tau.
fn parentTau() -> f32 { return state.tau; }

// Level 1's own block grid coincides with L0's by construction (the 1:1
// footprint-preserving parent, plans/AMR-multilevel.md decision 1).
const BLOCK = 8u;
fn levelNbx() -> u32 { return W / BLOCK; }
fn levelNby() -> u32 { return H / BLOCK; }

// Origin: the coarse block's own buffer-space lower-left corner. The quadrant
// plays no part here -- level 1 covers its parent block whole.
fn parentOrigin(slot: u32, bx: u32, by: u32) -> vec2<u32> {
  return vec2<u32>(bx * RB, by * RB);
}

// Block-major linear index for a cell at COARSE buffer coordinates (cx, cy).
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

fn wrapCoord(v: i32, n: u32) -> u32 {
  let m = i32(n);
  return u32(((v % m) + m) % m);
}

// CoarseSample lives in common_interp.wgsl, alongside the blend that consumes
// it -- this supplies only the FETCH. (ix,iy) are BUFFER-space integer
// coordinates, periodic, no off_x mapping needed.
fn sampleParent(slot: u32, ix: i32, iy: i32) -> CoarseSample {
  let cx = wrapCoord(ix, W);
  let cy = wrapCoord(iy, H);
  let cell = cellIndex(cx, cy);

  var f: array<f32, 9>;
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    f[i] = fUnpack(f_coarse[fIdx(i, (W * H), cell)], i);
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
