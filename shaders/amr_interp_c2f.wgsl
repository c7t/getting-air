// Milestone 4 (plans/AMR.md): coarse -> fine ghost-cell interpolation,
// POOL-AWARE. Supersedes Milestone 2's single-fixed-region version:
// instead of one hardcoded window-anchored fine region, this operates on
// whichever of MAX_FINE_BLOCKS pool slots are currently assigned to a
// coarse block (slotToBlock[slot] != -1).
//
// Dispatched over (tileX, tileY, slot) -- the Z dimension selects pool
// slot, so cost scales with pool CAPACITY, not domain size (see
// plans/AMR.md's Milestone 4 design note on why this dispatch shape is the
// one part not worth simplifying away).
//
// Buffer-space native (unlike M2): M1's coarse blocks are already defined
// in buffer space (fixed in memory regardless of the moving window's
// off_x/off_y), so a pool slot's coarse-cell lookups need no window
// conversion at all -- only the (separate, in amr_step1.wgsl) card SDF
// physics needs window coordinates.
//
// GHOST_ONLY selects between two compiled pipelines from this one module:
// GHOST_ONLY=1 (steady-state, every macro-step) only re-interpolates the
// ghost border, matching M2's behavior. GHOST_ONLY=0 (one-time, on
// activation) fills the WHOLE slot including the "real" interior, needed
// because a freshly-activated slot has no prior fine-level state to
// evolve from.

struct CardState {
  cx     : f32,
  cy     : f32,
  theta  : f32,
  vx     : f32,
  vy     : f32,
  omega  : f32,
  fx     : f32,
  fy     : f32,
  tz     : f32,
  mass   : f32,
  i_body : f32,
  g_eff  : f32,
  a      : f32,
  b      : f32,
  v_max  : f32,
  o_max  : f32,
  cx_old : f32,
  cy_old : f32,
  th_old : f32,
  tau    : f32,
  y_total: f32,
  x_total: f32,
  off_x  : f32,
  off_y  : f32,
  off_x_old : f32,
  off_y_old : f32,
}

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_coarse       : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_pool         : array<f32>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
// Milestone 4b: which slots were JUST assigned this refine/coarsen round --
// only read when GHOST_ONLY=0 (the one-time full-slot-fill pipeline), to
// avoid re-filling an already-active slot's evolved interior with a fresh
// (and by now stale) coarse interpolation. The steady-state GHOST_ONLY=1
// pipeline shares this bind group layout but never reads this binding.
@group(0) @binding(4) var<storage, read>       newlyActivated : array<u32>;
// Milestone 4c: coarse-block -> pool-slot map (inverse of slotToBlock), so a
// ghost cell can check whether its EDGE-adjacent neighbor block is also
// currently refined, and if so pull directly from that neighbor's fine
// interior instead of round-tripping through the coarse level. Read-write
// elsewhere (main.js owns writes); read-only here.
@group(0) @binding(5) var<storage, read>       blockSlot      : array<i32>;

override W : u32;   // coarse grid dims
override H : u32;
override RB : u32;  // refine block size in coarse cells (matches M1's BLOCK)
override GHOST_ONLY : u32;
// When set, this pass does ONLY the fine-fine (edge + diagonal-corner) ghost
// copy below and returns -- it never falls through to coarse interpolation and
// never touches coarse-adjacent ghosts. Used for the between-substep fine-fine
// re-exchange (see main-amr.js): the once-per-macro-step interp fills fine-fine
// seams exactly for the FIRST fine substep only; without a refresh, the SECOND
// substep streams a stale/edge-clamped neighbor halo, injecting a periodic
// error at every fine-fine seam once per macro-step (the block-pitch artifact).
// Coarse-adjacent ghosts intentionally keep their "degrade over 2 substeps"
// behavior (correct multi-rate coupling; coarse is quasi-static), so this mode
// leaves them alone. Default 0u so the existing interp/interpInit pipelines,
// which don't set it, still compile.
override FINE_FINE_ONLY : u32 = 0u;

const BLOCK = 8u;
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

// Block-major linear index for a cell at COARSE buffer coordinates (cx, cy).
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

// Fine ghost-local coordinate -> position in BUFFER-space coarse units.
// origin is the coarse block's own buffer-space lower-left corner
// (blockBX*RB, blockBY*RB). No window conversion: coarse-cell lookups
// stay in buffer space throughout, matching M1's own addressing.
fn fineToCoarseUnit(fCoord: u32, origin: u32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

fn wrapCoord(v: i32, n: u32) -> u32 {
  let m = i32(n);
  return u32(((v % m) + m) % m);
}

struct CoarseSample {
  rho: f32,
  ux: f32,
  uy: f32,
  fneq: array<f32, 9>,
}
// wx, wy here are BUFFER-space integer coordinates (periodic, no off_x
// mapping needed -- see file header).
fn sampleCoarse(bx_in: i32, by_in: i32) -> CoarseSample {
  let bx = wrapCoord(bx_in, W);
  let by = wrapCoord(by_in, H);
  let cell = cellIndex(bx, by);

  var f: array<f32, 9>;
  var rho = 0f; var ux = 0f; var uy = 0f;
  for (var i = 0u; i < 9u; i++) {
    f[i] = f_coarse[i * (W * H) + cell];
    rho += f[i];
    ux  += f[i] * f32(ex[i]);
    uy  += f[i] * f32(ey[i]);
  }
  ux /= max(rho, 1e-6f); uy /= max(rho, 1e-6f); // NaN-containment floor

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
  // Milestone 4b: init mode only fills genuinely-new slots (see binding 4's
  // comment above) -- an already-active slot reaching this pipeline (only
  // possible if callers dispatch it too broadly) must not be touched.
  if (GHOST_ONLY == 0u && newlyActivated[slot] == 0u) { return; }

  let nbx = W / BLOCK;
  let nby = H / BLOCK;
  let bx = u32(blockID) % nbx;
  let by = u32(blockID) / nbx;
  let originX = bx * RB;
  let originY = by * RB;

  // Milestone 4c: fine-fine ghost consultation. Only in the steady-state
  // pass (GHOST_ONLY=1) -- in the init pass (GHOST_ONLY=0) two edge-adjacent
  // blocks can both be newly-activated in the SAME dispatch, and this pass
  // also fills their interiors (see the isInterior/GHOST_ONLY skip above),
  // so reading a neighbor's interior there would race its own write.
  // Steady-state has no such race: interior cells are only ever written by
  // the PREVIOUS macro-step's step1PL (this pass returns early on interior
  // cells, see above), so a neighbor's interior is always stable data
  // regardless of dispatch order across slots/workgroups.
  //
  // A ghost cell lies along exactly one edge (not a diagonal corner) when
  // exactly one of fx/fy is out of the interior range. Diagonal corners
  // (both out of range) consult the diagonal neighbor block the same way --
  // found to matter in practice, not just a completeness nicety: a corner
  // still on the old coarse-interpolated path, sitting between two edges
  // now made exact by the fine-fine consultation below, goes measurably
  // inconsistent with its (now-precise) edge neighbors. Since the D2Q9
  // stencil streams diagonally too, that inconsistency feeds directly into
  // the interior cell nearest the corner every fine substep and grows
  // there until it diverges -- confirmed directly: a controlled 3x3
  // refined cluster seeded with a smooth analytic (Taylor-Green) field and
  // no card/turbulence involved still blew up within ~2800 macro-steps,
  // bisected to a single corner cell (the one corner of a cluster-corner
  // block that happens to have an active diagonal neighbor) climbing from
  // ~0.55 to over 1 before cascading to NaN.
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
      // NW diagonal corner.
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2; nfy = fy + RB2;
    } else if (fx >= GHOST + RB2 && fy < GHOST) {
      // NE diagonal corner.
      neighborSlot = blockSlot[((by + nby - 1u) % nby) * nbx + ((bx + 1u) % nbx)];
      nfx = fx - RB2; nfy = fy + RB2;
    } else if (fx < GHOST && fy >= GHOST + RB2) {
      // SW diagonal corner.
      neighborSlot = blockSlot[((by + 1u) % nby) * nbx + ((bx + nbx - 1u) % nbx)];
      nfx = fx + RB2; nfy = fy - RB2;
    } else if (fx >= GHOST + RB2 && fy >= GHOST + RB2) {
      // SE diagonal corner.
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

  // Fine-fine-only mode: this ghost cell has no active fine neighbor (the copy
  // above didn't fire), so there is nothing to refresh -- leave it exactly as
  // the previous fine substep produced it (coarse-adjacent ghosts keep their
  // intended multi-rate "degrade over 2 substeps" behavior) and do NOT run the
  // coarse interpolation below.
  if (FINE_FINE_ONLY != 0u) { return; }

  let px = fineToCoarseUnit(fx, originX);
  let py = fineToCoarseUnit(fy, originY);

  let x0 = i32(floor(px)); let x1 = x0 + 1;
  let y0 = i32(floor(py)); let y1 = y0 + 1;
  let tx = px - f32(x0);
  let ty = py - f32(y0);

  let s00 = sampleCoarse(x0, y0);
  let s10 = sampleCoarse(x1, y0);
  let s01 = sampleCoarse(x0, y1);
  let s11 = sampleCoarse(x1, y1);

  let w00 = (1f - tx) * (1f - ty);
  let w10 = tx * (1f - ty);
  let w01 = (1f - tx) * ty;
  let w11 = tx * ty;

  let rho = w00*s00.rho + w10*s10.rho + w01*s01.rho + w11*s11.rho;
  let ux  = w00*s00.ux  + w10*s10.ux  + w01*s01.ux  + w11*s11.ux;
  let uy  = w00*s00.uy  + w10*s10.uy  + w01*s01.uy  + w11*s11.uy;

  let tau_coarse = state.tau;
  let tau_fine = 2.0f * tau_coarse - 0.5f;
  // Dupuis-Chopard non-equilibrium rescale, coarse->fine. The factor is
  // (tau_fine/tau_coarse) * (dx_fine/dx_coarse) = (tau_fine/tau_coarse) * (1/n),
  // with refinement ratio n=2. fneq scales as tau * (velocity gradient per
  // lattice cell); the same physical shear spans 2x as many fine cells, so the
  // per-cell gradient (and hence fneq) is halved on the fine grid. Omitting the
  // 1/n factor leaves an O(1) (2x) non-equilibrium stress discontinuity at every
  // fine<->coarse interface, injecting spurious vorticity there.
  let rescale = 0.5f * tau_fine / tau_coarse;

  // f_pool is direction-major across the WHOLE pool (matching the coarse
  // f_coarse convention): plane stride = MAX_FINE_BLOCKS*FB*FB, derived via
  // arrayLength instead of a separate override (the buffer's actual size
  // already encodes it).
  let poolPlaneStride = arrayLength(&f_pool) / 9u;
  let poolCellBase = slot * (FB * FB) + fy * FB + fx;
  for (var i = 0u; i < 9u; i++) {
    let fneq = w00*s00.fneq[i] + w10*s10.fneq[i] + w01*s01.fneq[i] + w11*s11.fneq[i];
    f_pool[i * poolPlaneStride + poolCellBase] = feq(rho, ux, uy, i) + rescale * fneq;
  }
}
