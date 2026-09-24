// THE fine-level LBM step -- ONE kernel, every pool level (plans/2D-backport.md
// B3-1). Until then this was the level>=2 half of a pair, with a separate
// amr_step1.wgsl compiled and dispatched for level 1 alone: 349 and 359 lines
// stating the same streaming/collision/penalization/sponge body twice.
//
// THE PAIR EXISTED FOR ONE REASON, AND IT WAS NOT A REAL ONE. The level-1 file
// derived its tile's physical origin as `bx * RB` from blockID; this one read a
// per-slot originX/originY buffer, on the argument that a level>=2 tile's
// origin needs a walk up the parent chain (quadrant offset scaled by the
// parent's cell size, plus the parent's origin, recursively) and so is not
// something a per-dispatch kernel should redo. But every level's block grid is
// globally anchored and quadtree-uniform, so that recursion has a closed form
// -- amr2d.mjs's tileOriginL0, `block * RB * 2^-(m-1)`, which is
// `f32(bx*RB) * 2 * levelParams.dxL` here. The two routes agree exactly in f32
// (every term is an integer times a power of two): tools/test-amr2d.js scores
// the host pair against each other, and debugCheckTileOrigins scores the live
// buffer against the same closed form at every invariant checkpoint.
//
// With the origin derived, nothing structural is left, and each of the four
// remaining differences was already a field of the per-level uniform this
// kernel had:
//
//   origin  ->  bx/by (levelParams.nbx) and levelParams.dxL, below
//   dxL     ->  levelParams.dxL          (level 1: 0.5, the old literal)
//   tau     ->  levelParams.parentTau    (level 1: L0's own state.tau)
//   K_EPS   ->  levelParams.kEps, and kEps * dxL at level 1 is kEps * 0.5,
//               which is exactly what the level-1 file computed. It was an
//               override on both files; see get_chi below for why one
//               pipeline made that untenable.
//
// tau is worth spelling out because it is the one that is not geometry: a
// tile's tau_fine derives from its PARENT level's tau, not from L0's. At level
// 1 those coincide, which is why the old level-1 file could read state.tau
// directly; at level>=2 they do not. levelParams.parentTau is tauAtLevel(m-1)
// (card-params.mjs), written per level by the host and re-written when the TAU
// slider moves.
//
// L0 IS A POOL LEVEL TOO SINCE U7-6f, served by THIS kernel with GHOST 0 and
// NO_PARENT 1 -- see amr2d-gpu.mjs's makeRootPool. It was a dense,
// ghost-free, cellIndex()-addressed grid with its own separate kernel through
// plans/AMR-multilevel.md decision 1 and every rung up to U7-6e; the root tile
// being RINGLESS is all that survives of that asymmetry.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"
// @include "common_sponge.wgsl"
// @include "common_walls.wgsl"





struct LevelParams {
  nbx: u32,        // this level's own logical block-grid extent. Names same-level neighbours for DIRECT_GHOST streaming, and (with dxL) places the tile physically -- see the origin derivation in main(). Shared verbatim with amr_interp_pool_parent.wgsl/amr_average_pool_parent.wgsl, not a third near-duplicate.
  nby: u32,        // same.
  parentTau: f32,
  dxL: f32,        // Milestone 8: this level's own grid spacing in L0-buffer-
                   // space units, used below to scale epsilon (get_chi) and
                   // to place the tile (2*dxL is the parent's cell size).
  hasChild: u32,   // unused here -- declared only to reach kEps at offset 20.
                   // Real field of the shared 32-byte buffer (amr_force1_pool.
                   // wgsl reads it), not padding.
  kEps: f32,       // the diffuse band in units of THIS level's dx. A per-level
                   // uniform and NOT an override since B3-1's follow-up: one
                   // pipeline serves every level now, so a compile-time
                   // constant could no longer say anything per-level, and
                   // main-cylinder-amr.js's ?kEpsPool= (which singles out
                   // levels >=2) had silently become a whole-hierarchy flag.
}

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_in        : array<u32>;
@group(0) @binding(2) var<storage, read_write> f_out       : array<u32>;
@group(0) @binding(3) var<storage, read_write> vel_pool    : array<f32>;
@group(0) @binding(4) var<storage, read>       slotToBlock : array<i32>;
@group(0) @binding(5) var<uniform>             levelParams : LevelParams;
// This level's own logical block grid -> pool slot, indexed by
// blockID = by*levelParams.nbx+bx -- the one new input neighbour-addressed
// streaming needs (see DIRECT_GHOST), the same buffer and the same indexing
// amr_interp_pool_parent.wgsl's fine-fine consultation already uses.
@group(0) @binding(6) var<storage, read>       blockSlot   : array<i32>;
// Read only by `mainIndirect` below.
@group(0) @binding(7) var<storage, read>       activeSlots : array<u32>;

override W : u32; // GLOBAL domain dims (window periodicity), same at every level -- not level-specific, see header.
override H : u32;
override RB : u32;
// ── Measurement instrument: ?benchSkip=step1-ring ────────────────────────────
// Skips the ghost-RING cells of each tile, leaving only the 2*RB square
// interior. A proxy for what the fine step would cost if tiles carried no
// ghost padding at all (FB 20 -> 16, i.e. 36% fewer cells per tile), which is
// the second-order prize in "stop materializing same-level ghost cells".
//
// MEASUREMENT ONLY, and wrong by construction: substep B consumes the ring
// values substep A wrote for any cell the fine-fine exchange does not refresh
// (a tile at the edge of the refined region), so a run with this set is not
// physically correct.
//
// Measured 2026-09-07, desktop RTX 4080, res=8 levels=3: 0.2% of frame GPU
// time -- essentially free to skip, because ring threads share their 8x8
// workgroups with interior threads and removing them frees no scheduling slot.
// This CORRECTS plans/perf-characterization.md, which listed the same 36% as
// "about 10.5% of the whole frame ... the single largest identified piece of
// pure overhead". It is not, on this device. The phone is bandwidth-bound
// rather than occupancy-bound and the traffic model predicts ~10% there, so
// this knob exists to be re-read on that device, where the answer should
// differ. Guard placed after the slot lookup so the number stays comparable
// with the desktop figure above.
override SKIP_GHOST : u32 = 0u;

// ── Neighbour-addressed streaming (AGAL) ─────────────────────────────────────
// 1 (default): a pull whose source cell falls in this tile's ghost ring is
// resolved against the OWNING same-level tile's own interior, via blockSlot,
// instead of reading a materialized ghost cell. 0: the legacy path -- clamp at
// the slot's own buffer edge and rely on a separate fine-fine copy pass having
// filled the ring (?ghostcopy=1 restores it, for A/B).
//
// This is what lets the between-substep fine-fine ghost pass go away entirely
// (plans/perf-characterization.md's "the one lead left": 15.3% of frame on the
// phone, 13.3-18.5% on the desktop). It is not just cheaper, it is fresher:
// substep B now sees the neighbour's post-`average` interior directly, where
// the copy pass ran BEFORE the child's average landed, and a depth-2 ring cell
// -- which the copy path leaves clamp-degraded after substep A, and which the
// child's own bilinear parent sampling does read -- now streams correctly too.
//
// The ring is still materialized, and interp still fills it: a tile at the
// coarse/fine interface has no same-level neighbour there, so blockSlot is < 0
// and this falls back to the clamped read of a parent-interpolated ghost --
// exactly the legacy path, for exactly the cells that need it.
override DIRECT_GHOST : u32 = 1u;

// GHOST is an OVERRIDE, not a const, since plans/uniform-levels.md U3: the
// ROOT level has no ring at all (amr2d.mjs's ghostDepthAtLevel(0) is 0),
// because it has no parent to receive from and is always full, so DIRECT_GHOST
// always resolves against an owning same-level tile and never falls back.
// Default 2 keeps every existing pipeline byte-identical.
//
// It flows into FB = RB*2 + 2*GHOST, so at 0 a slot is exactly its own
// 2*RB x 2*RB cells and the local coordinates ARE the cells.
override GHOST : u32 = 2u;

// COLLIDE_RING (plans/2D-backport.md B6-1). 0 on the explode/coalesce path:
// a RING cell advects and stores what it gathered, uncollided, and the
// interior collides as ever. Chen et al.'s coalesce averages advected-but-
// UNCOLLIDED states -- "an arithmetic average of N_i's rather than of
// Ntilde_i's would invalidate the correctness of non-equilibrium distributions
// on the coarse grid" -- and the explode's ring self-advance is what lets one
// explosion feed both substeps. Colliding the ring would relax the parent's
// populations at the child's tau, which is the thing the Dupuis-Chopard factor
// existed to undo; not colliding is why this path has no factor. Default 1 is
// the interp path, byte-identical. The root (GHOST 0) has no ring, so it is
// indifferent.
override COLLIDE_RING : u32 = 1u;

// NO_PARENT: this level is the ROOT. Two things follow, and both are
// consequences of the same fact rather than two switches.
//
// 1. TAU IS ITS OWN. `2*parentTau - 0.5` has no meaning without a parent. The
//    arithmetically equivalent dodge -- write `(tau+0.5)/2` into parentTau and
//    let the kernel undo it -- is NOT used: `fl(fl(tau + 0.5) - 0.5)` is not
//    guaranteed to be `tau` in f32, so a bit-identity gate could fail on the
//    round trip rather than on anything real. It reads `state.tau`, the same
//    word the dense kernel reads.
//
// 2. THE HALF-CELL OFFSET GOES AWAY. fineToCoarseUnit places cell j at
//    `origin - 0.5*dxL + dxL*j`, where `origin` is the centre of the first
//    PARENT cell the tile covers and the two children straddle it at
//    +-dxL/2. At the root `origin` is the first cell's OWN centre -- there is
//    nothing to straddle -- so the offset is 0 and cell j sits at
//    `origin + dxL*j`.
//
//    This was found by U3's differential test, not by reading: the root and
//    dense kernels disagreed at the very first cell, and a half-cell shift in
//    the body's phi and the sponge's distance is exactly what that looks like.
//    It would not have been visible in any single-kernel test, because the
//    root is self-consistent with it -- just displaced half a cell from the
//    grid it is supposed to reproduce.
override NO_PARENT : u32 = 0u;

// SPONGE_CELL_SNAP IS GONE, AND THE WINDOW POSITION IS ONE CONVENTION.
//
// It existed so the root could take the DELETED dense L0 step's convention:
// `bufferToWindowCell`, u32 modular arithmetic, which TRUNCATES `off_x`/
// `off_y` to whole cells, while every other level used `bufferToWindowPos`
// and kept the sub-cell part. U3 found the disagreement and deliberately did
// not fix it -- U3 was a REPRESENTATION stage, its job was to show the root
// pool reproduces the dense grid exactly, not to improve it -- and U7-6f
// deleted the grid but kept the convention, because unifying them moves
// published numbers and that rung's gate was bit-identity.
//
// `wx`/`wy` feed the SPONGE BAND and nothing else in this file (the body's
// frame has been the buffer position itself since B5), so this was level 0's
// band snapping to whole cells while a fractional window offset slid
// underneath it -- a staircase where every finer level had a ramp. See
// plans/uniform-levels.md "U7-6f -- WHAT IT LEFT BEHIND" for the measurement
// that retired it, including what the disagreement was actually worth.

// Sponge relaxation target velocity -- the same formula lbm_step.wgsl uses on
// the dense reference pages (see this file's sponge comment below for why a
// refined level needs its own copy of the sponge at all).
override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;
// Sponge ring width in cells -- see lbm_step.wgsl's identical override.
override SPONGE_W : f32 = 4.0f;

// Milestone 8 (plans/AMR-multilevel.md): epsilon = kEps * dx_L, not a fixed
// physical constant -- a fixed epsilon means refinement only ever improves
// *sampling* of an unchanging diffuse-boundary width, never the boundary's
// own sharpness. kEps=1.5 matches today's L0/L1 value exactly (dx_L0=1,
// dx_L1=0.5; L0's was a hardcoded literal in the deleted dense step, which is why
// ITS epsilon does not change).
//
// BOTH factors are now per-level runtime uniforms, and the second one had to
// become one: B3-1 made this a single pipeline for every level, so a K_EPS
// OVERRIDE could only ever carry one value for the whole hierarchy -- which
// silently turned main-cylinder-amr.js's ?kEpsPool= (documented as singling
// out levels >=2) into a flag that moved level 1 too. ?kEps= still sweeps
// every level at once; the host writes the per-level value.

// Optional sharp momentum-exchange bounce-back solid coupling -- see
// lbm_step.wgsl's header for the full method. At this level, "the
// geometrically-correct source" (used for the sharp inside test) and "the
// clamped-at-tile-edge source" (the buffer address normal streaming reads)
// are DIFFERENT things -- clamping is purely a buffer-addressing artifact
// for cells whose true neighbor lies outside this tile's own FB x FB storage
// (that continuity is handled by the separate ghost-fill pass, not by this
// kernel), not a physical statement -- so the sharp test below deliberately
// uses the UNCLAMPED fine-index position (fineToCoarseUnitI, i32-accepting
// so it stays well-defined for an off-tile index), while the bounce-back
// VALUE substitution still reads this cell's own (in-tile, always valid)
// data.
override USE_BOUNCEBACK : u32 = 0u;

// SOLID_EQ: under BOUNCE-BACK, hold cells INSIDE the body at the local solid
// equilibrium instead of letting them evolve (plans/2D-backport.md B8).
//
// THE HAZARD, and why 2D cannot currently see it. Under bounce-back `chi` is
// forced to 0, so the penalty term that damps the interior under the diffuse
// coupling is not there -- and the gather above only redirects a source that
// is solid, which says nothing about a cell that IS solid. Interior cells are
// therefore stepped as ordinary fluid with reflected gathers and nothing
// bounds them: 3D measured max|u| inside the body at 6x the body's own speed
// at tau=0.6, and a moving body dead in 200 steps at tau=0.514.
//
// A PINNED body never notices, because nothing reads a solid cell: the
// bounce-back branch reads f_in[opp[i]] at the FLUID cell, and the force
// kernel runs only where phi >= 0. 2D's only bounce-back body is pinned
// (?bounceback lives in main-cylinder.js and main-cylinder-amr.js, both
// fixed), so this fix cannot move a single 2D number today -- which is
// exactly why it is worth landing now rather than after ?bounceback=1 is
// first pointed at a moving body.
//
// IT IS ALSO THE FRESH-NODE REFILL, done unconditionally. A cell the body
// vacates becomes fluid holding whatever it last had; writing the solid
// equilibrium every step means the value it is uncovered with is already the
// right one, at no cost and with no need to detect the uncovering. rho = 1 is
// the same near-incompressible choice the bounce-back correction term above
// already makes.
//
// ?solideq=0 restores the old behaviour for A/B.
override SOLID_EQ : u32 = 1u;

// Channel/TGV-scenario overrides -- see shaders/lbm_step.wgsl's identical
// set for the full rationale. All default to a no-op.
override HAS_BODY : u32 = 1u;
override WALL_Y : u32 = 0u;
override WALL_U0 : f32 = 0.0f;
override WALL_U1 : f32 = 0.0f;
override FORCE_X : f32 = 0.0f;
override FORCE_Y : f32 = 0.0f;

// BUGFIX: this file serves EVERY pool level through one shared pipeline
// (dx varies per level -- 0.5 at level 1, 0.25 at level 2, 0.125 at level
// 3, ...), but this function once hardcoded level 1's OWN fixed
// dx=0.5/half-cell=0.25 (correct only for the level-1-DEDICATED file this
// was copied from, since deleted) instead of reading levelParams.dxL -- a
// pre-existing bug (confirmed present before this session's own changes),
// not something introduced by the bounce-back work that surfaced it.
// Effect: every level>=2 cell's computed physical position was stretched
// by (0.5/dxL)x too wide relative to its tile's true origin-anchored
// footprint -- e.g. 2x at level 2, 4x at level 3 -- corrupting BOTH the
// diffuse method's chi/phi and bounce-back's sharp inside/outside test.
// Live-verified impact, NEITHER fully resolved by this fix alone (both
// have at least one more separate, unresolved issue -- flagged near
// N_LEVELS in main-cylinder-amr.js for bounce-back's own remaining one):
// diffuse-method Cd at N=3 moved from 0.152 to 0.228 (still far below the
// ~1.35 target -- confirms this bug was A contributor to the project's
// separately-tracked ~9x AMR Cd deficit, not the sole cause); bounce-
// back's level-2-ONLY force (L1 forced to the diffuse path, isolating
// level 2) moved from wrong-signed fx=-7.09 (1201 boundary-link triggers)
// to still-wrong-but-improved fx=-6.05 (797 triggers) -- real
// measured progress, not a full fix.
// The half-cell straddle is a PARENT-RELATIVE term -- see NO_PARENT.
// CELL_CENTRE_AFFINE (plans/uniform-levels.md S8-2). Root cells are centred
// on INTEGERS, so cell g of level m covers [g*dx - 1/2, (g+1)*dx - 1/2] and
// its centre is (g + 1/2)*dx - 1/2 -- the affine map, compounding per rung.
// From `origin = block*2*RB*dx` that is an offset of (1 - dx)/2: 0 at the
// root, 1/4 at level 1, 3/8 at level 2, 7/16 at level 3.
//
// The legacy offset was `dx/2`, on the premise that `origin` is the centre of
// the tile's first PARENT cell. That holds at level 1 (a root cell's centre is
// an integer) and nowhere below it: a level-1 cell's centre is x.25 or x.75,
// and `origin` is a multiple of RB*dx*2. So from level 2 down every cell was
// placed 1/2 - dx of a root cell high in x AND y -- 1/4, 3/8, 7/16 -- while
// the transfers, which pair children (GHOST+2p, GHOST+2p+1) with parent cell
// q*RB + p by INDEX, kept the data where it belongs. The body, the sponge and
// the walls were evaluated in the wrong place relative to the flow on every
// level >= 2. Measured on the pinned cylinder with the body on the tile
// partition's mirror axis and no seed: startup |Cl| 0.053 (levels=3) and 0.43
// (res 7 levels=4) against 2e-4 at levels=2 -- and 2e-4 at every depth with
// this rule. A y-displacement is the only thing that can make a symmetric
// problem lift. ?cellcentre=0 restores the legacy offset.
override CELL_CENTRE_AFFINE : u32 = 1u;
fn cellCentreOffset() -> f32 {
  if (CELL_CENTRE_AFFINE != 0u) { return 0.5f * (1.0f - levelParams.dxL); }
  return select(0.5f * levelParams.dxL, 0.0f, NO_PARENT != 0u);
}

fn fineToCoarseUnit(fCoord: u32, origin: f32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return origin - cellCentreOffset() + levelParams.dxL * j;
}

fn fineToCoarseUnitI(fCoordI: i32, origin: f32) -> f32 {
  let j = f32(fCoordI - i32(GHOST));
  return origin - cellCentreOffset() + levelParams.dxL * j;
}

fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, levelParams.kEps * levelParams.dxL);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { stepCell(gid, gid.z); }

// ?indirect=1 (main-amr.js): launched over this pool's active-slot list
// (amr_active_list.wgsl), so z indexes the list rather than the pool. `main`
// never reads `activeSlots`, so its layout -- every other page's -- is unchanged.
@compute @workgroup_size(8, 8)
fn mainIndirect(@builtin(global_invocation_id) gid: vec3<u32>) { stepCell(gid, activeSlots[gid.z]); }

fn stepCell(gid: vec3<u32>, slot: u32) {
  let fx = gid.x; let fy = gid.y;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  if (SKIP_GHOST != 0u) {
    let ringInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;
    if (!ringInterior) { return; } // see the SKIP_GHOST override above
  }

  // Logical (bx,by) within THIS level's own block grid, derived from blockID
  // exactly as amr_interp_pool_parent.wgsl derives it. Used for two things:
  // naming this tile's same-level neighbours (DIRECT_GHOST), and placing the
  // tile physically, just below.
  let nbx = levelParams.nbx;
  let nby = levelParams.nby;
  let bx = u32(blockID) % nbx;
  let by = u32(blockID) / nbx;

  // THIS TILE'S PHYSICAL ORIGIN IN L0 UNITS, AS ONE MULTIPLY.
  //
  // It used to be read from a per-slot originX/originY buffer, on the
  // argument (this file's own former header) that a level>=2 tile's origin
  // needs a walk up the parent chain and is therefore not something a
  // per-dispatch kernel should redo. That argument is wrong, and amr2d.mjs
  // has said so since B0: every level's block grid is globally anchored and
  // quadtree-uniform, so tileOriginL0 is `block * RB * 2^-(m-1)` in closed
  // form -- and `2^-(m-1)` is `2 * dxL`. The recursion the manager runs and
  // this multiply agree exactly in f32 (every term is an integer times a
  // power of two); tools/test-amr2d.js scores the two host routes against
  // each other, and debugCheckTileOrigins scores the live buffer against
  // this same closed form at every invariant checkpoint.
  //
  // This is what makes the file level-generic: with the origin derived, the
  // ONE structural difference between this kernel and the old level-1-only
  // amr_step1.wgsl is gone, and level 1 is just `dxL = 0.5`.
  let originX_L0 = f32(bx * RB) * 2.0f * levelParams.dxL;
  let originY_L0 = f32(by * RB) * 2.0f * levelParams.dxL;
  let RB2 = RB * 2u;
  let bxm = (bx + nbx - 1u) % nbx;
  let bxp = (bx + 1u) % nbx;
  let bym = (by + nby - 1u) % nby;
  let byp = (by + 1u) % nby;

  // Neighbour-slot resolution, hoisted out of the 9-direction gather below.
  // A source cell is at most one cell away and a tile's interior is RB2 >= 2
  // wide, so at most ONE non-zero neighbour offset is reachable per axis --
  // which means the whole gather needs at most THREE neighbour slots (the
  // x-, y- and diagonal tiles), resolved once here instead of re-resolved,
  // with a fresh blockSlot load, on every one of the nine directions. A
  // fully-interior thread loads nothing at all.
  var offX = 0; var offY = 0;
  var nbrX = -1; var nbrY = -1; var nbrXY = -1;
  if (DIRECT_GHOST != 0u) {
    offX = select(select(0, 1, fx + 1u >= GHOST + RB2), -1, fx <= GHOST);
    offY = select(select(0, 1, fy + 1u >= GHOST + RB2), -1, fy <= GHOST);
    let cx = select(select(bx, bxp, offX > 0), bxm, offX < 0);
    let cy = select(select(by, byp, offY > 0), bym, offY < 0);
    if (offX != 0) { nbrX = blockSlot[by * nbx + cx]; }
    if (offY != 0) { nbrY = blockSlot[cy * nbx + bx]; }
    if (offX != 0 && offY != 0) { nbrXY = blockSlot[cy * nbx + cx]; }
  }

  let poolPlaneStride = arrayLength(&f_in) / 9u;
  let cell = slot * (FB * FB) + fy * FB + fx;

  // Position/solid-velocity terms, hoisted ABOVE the gather loop -- see
  // lbm_step.wgsl's identical hoist for why USE_BOUNCEBACK needs these
  // before streaming, not after. Buffer-space fine position (L0 units, via
  // the origin derived above) -> window position by inverting off_x/off_y
  // (see file header).
  let bufX = fineToCoarseUnit(fx, originX_L0);
  let bufY = fineToCoarseUnit(fy, originY_L0);
  // WINDOW position, for the sponge band below -- ONE convention at every
  // level, including the root, since SPONGE_CELL_SNAP went (see its old home
  // above). The BODY's frame is just the buffer position itself since B5
  // (common_geometry.wgsl).
  let wpos = bufferToWindowPos(vec2<f32>(bufX, bufY), state);
  let wx = wpos.x; let wy = wpos.y;
  let p = vec2<f32>(bufX, bufY);
  // Periodic minimum-image lever arm, matching amr_force1.wgsl and
  // lbm_step.wgsl (the force pass and the dense reference step wrap rx/ry; the
  // fine step previously did not, so a cell reached across a seam got the
  // wrong rotational velocity).
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);

  // 1. Pull Streaming: clamp at the slot's own buffer edge (or bounce back
  // off the solid -- see USE_BOUNCEBACK's own header comment).
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    if (USE_BOUNCEBACK != 0u && HAS_BODY != 0u) {
      let srcBufX = fineToCoarseUnitI(i32(fx) - ex[i], originX_L0);
      let srcBufY = fineToCoarseUnitI(i32(fy) - ey[i], originY_L0);
      if (get_phi(vec2<f32>(srcBufX, srcBufY), state) < 0f) {
        let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
        f[i] = fUnpack(f_in[fIdx(opp[i], poolPlaneStride, cell)], opp[i]) + corr;
        continue;
      }
    }
    if (WALL_Y != 0u) {
      // Unwrapped buffer-space source position -- see
      // shaders/common_walls.wgsl's *F helpers' own comment on why this
      // (not wrapf's periodic result) is the right test, and why it
      // assumes off_y=0 (true for every WALL_Y-using scenario).
      let srcBufYUnwrapped = fineToCoarseUnitI(i32(fy) - ey[i], originY_L0);
      if (wallSourceOutsideF(srcBufYUnwrapped)) {
        let wallUx = wallVelocityXF(srcBufYUnwrapped, WALL_U0, WALL_U1);
        let corr = 2f * wt[i] * f32(ex[i]) * wallUx / CS2;
        f[i] = fUnpack(f_in[fIdx(opp[i], poolPlaneStride, cell)], opp[i]) + corr;
        continue;
      }
    }
    var sx = i32(fx) - ex[i];
    var sy = i32(fy) - ey[i];
    var srcSlot = slot;
    if (DIRECT_GHOST != 0u) {
      // Does this source cell leave the interior, and on which axes? If so
      // the offset can only be offX/offY (see the hoist above), so this is
      // pure register work -- no second blockSlot load.
      let ox = select(0, offX, sx < i32(GHOST) || sx >= i32(GHOST + RB2));
      let oy = select(0, offY, sy < i32(GHOST) || sy >= i32(GHOST + RB2));
      let ns = select(select(select(-1, nbrY, oy != 0), nbrX, ox != 0),
                      nbrXY, ox != 0 && oy != 0);
      if (ns >= 0) {
        // Re-express the source in the neighbour's own local coordinates. It
        // always lands in ITS interior -- sx in [-1, GHOST-1] maps to
        // [RB2-1, RB2+1], sx in [GHOST+RB2, FB] maps to [GHOST, GHOST+2] --
        // so the clamp below is a no-op on this path.
        srcSlot = u32(ns);
        sx -= ox * i32(RB2);
        sy -= oy * i32(RB2);
      }
    }
    // No same-level neighbour (or DIRECT_GHOST=0): clamp at the slot's own
    // buffer edge and read this tile's own ghost cell, which the interp pass
    // filled from the parent.
    let srcCell = srcSlot * (FB * FB)
                + u32(clamp(sy, 0, i32(FB) - 1)) * FB
                + u32(clamp(sx, 0, i32(FB) - 1));
    f[i] = fUnpack(f_in[fIdx(i, poolPlaneStride, srcCell)], i);
  }

  // A ring cell advects and stores -- see COLLIDE_RING. Its velocity is still
  // written, from the gathered moments, so nothing that samples a ring reads a
  // value left over from before the path changed; it is NOT a fluid velocity
  // (a ring cell is an inbox/outbox, plans/uniform-levels.md 2.5).
  if (COLLIDE_RING == 0u && GHOST > 0u) {
    let inInterior = fx >= GHOST && fx < GHOST + RB2 && fy >= GHOST && fy < GHOST + RB2;
    if (!inInterior) {
      var rr = 0f; var mx = 0f; var my = 0f;
      for (var i = 0u; i < 9u; i++) { rr += f[i]; mx += f[i] * f32(ex[i]); my += f[i] * f32(ey[i]); }
      let rd = max(rr, 1e-6f);
      vel_pool[cell * 2u] = mx / rd; vel_pool[cell * 2u + 1u] = my / rd;
      let nwr = fWords();
      for (var wi = 0u; wi < nwr; wi++) {
        f_out[wi * poolPlaneStride + cell] = fPack(f[fLo(wi)], f[fHi(wi)], wi);
      }
      return;
    }
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  // NaN-containment floor (see lbm_step.wgsl): finite velocity even if rho<=0.
  let rhoDen = max(rho, 1e-6f);
  ux_star /= rhoDen; uy_star /= rhoDen;

  // 3. Penalty Force and Solid Coupling -- chi forced to 0 under
  // USE_BOUNCEBACK or when there's no body at all, same as lbm_step.wgsl.
  let chi = select(get_chi(phi), 0f, USE_BOUNCEBACK != 0u || HAS_BODY == 0u);

  let Fx = rho * chi * (usx - ux_star) + FORCE_X;
  let Fy = rho * chi * (usy - uy_star) + FORCE_Y;

  let ux = ux_star + Fx / (2.0f * rhoDen);
  let uy = uy_star + Fy / (2.0f * rhoDen);
  let u_sq = ux*ux + uy*uy;

  vel_pool[cell * 2u] = ux; vel_pool[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC sponge. Milestone 4b fix: this used to skip the
  // sponge entirely on the (then-true) assumption that the fine region
  // never reaches the window edge -- valid when M2 hand-placed a single
  // static box, but false once refinement is criterion-driven and can
  // trigger anywhere, including near the sponge band where the COARSER level
  // DOES damp toward equilibrium. A refined block there with no sponge of its
  // own diverges from its damped coarse neighbours, and the average pass then
  // writes that undamped state back onto them -- exactly the boundary artifact
  // this was fixed in response to. One formula for every level since U7-6f
  // (level 0 runs this kernel too), from the wx/wy computed above. That
  // window position is now THE ONLY thing in this file that needs one: the
  // card SDF it used to share has read the buffer position directly since B5.
  let dist_x = min(wx, f32(W) - 1.0f - wx);
  let dist_y = min(wy, f32(H) - 1.0f - wy);
  let sponge_weight = spongeWeight(dist_x, dist_y, SPONGE_W);

  // Relative to THIS level's own parent, not L0 -- see header. At the root
  // there is no parent and `state.tau` IS this level's tau; see OWN_TAU.
  let tau_coarse = levelParams.parentTau;
  let tau_fine = select(2.0f * tau_coarse - 0.5f, state.tau, NO_PARENT != 0u);
  let omg = 1.0f / tau_fine;
  // Gathered, then stored a whole cell at a time: under F16 two planes share
  // a word, so a per-plane store would be a read-modify-write race. See
  // common_fpack.wgsl.
  // Uniform over this cell: is it INSIDE the body, on the bounce-back path?
  let inSolid = SOLID_EQ != 0u && USE_BOUNCEBACK != 0u && HAS_BODY != 0u && phi < 0f;
  var fo: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );

    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let eu_far = exf*SPONGE_UX + eyf*SPONGE_UY;
    let f_target = wt[i] * (1.0f + 3.0f*eu_far + 4.5f*eu_far*eu_far - 1.5f*(SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY));
    // SOLID_EQ (see header): inside the body, discard the collision entirely
    // and write the local solid equilibrium. Hoisted flag, per-direction
    // value -- the branch is uniform across the cell.
    fo[i] = select(mix(f_collide, f_target, sponge_weight),
                   feqD2Q9(1.0f, usx, usy, i), inSolid);
  }
  let nw = fWords();
  for (var wi = 0u; wi < nw; wi++) {
    f_out[wi * poolPlaneStride + cell] = fPack(fo[fLo(wi)], fo[fHi(wi)], wi);
  }
}
