// THE force/torque integration over a level's own pool tiles -- ONE kernel,
// every pool level (plans/2D-backport.md B3-4). Until then this was the
// level>=2 half of a pair, with a separate amr_force1.wgsl compiled and
// dispatched for level 1 alone.
//
// It collapsed for exactly the reasons B3-1 collapsed the fine-step pair, and
// the two files had exactly the same four differences -- origin, dxL, the
// diffuse band, and (here) the area/line weight, which IS dxL. The origin was
// the only structural one, and it was a wrong claim: a tile's physical origin
// is `block * RB * 2^-(m-1)` in closed form, i.e. `f32(bx*RB) * 2 *
// levelParams.dxL`, so the per-slot originX/originY buffers this used to read
// are gone from it -- and with the fine step off them since B3-1a, THIS WAS
// THEIR LAST READER. See plans/2D-backport.md B3-4 for what that retires.
//
// THE ONLY FORCE PASS SINCE U7-6f. It generalized a dense L0 one
// (amr_force.wgsl, deleted with the dense grid) the same way amr_step1.wgsl
// generalized the dense step -- same momentum-exchange math, dispatched over
// pool tiles (full FB*FB shape, Z=slot, and NOT the restriction's
// RB-granularity one, since MORE sample points per unit area is the entire
// point of Milestone 8: a fixed physical epsilon under-sampled the chi
// transition band at coarse resolution, aliasing the force/torque that drives
// the body's own trajectory).
//
// TWO THINGS A NAIVE PER-LEVEL COPY OF A DENSE FORCE PASS WOULD GET WRONG,
// and they are why this was not simply that file with a different binding:
//
// 1. GHOST cells must NOT contribute. Unlike amr_step1.wgsl (which
//    legitimately collides/streams every cell, ghost included, since ghost
//    cells still need to evolve before the next ghost-refresh overwrites
//    them), a ghost cell here is either a copy of a NEIGHBOR's interior
//    (same-level fine-fine ghost) or a coarse-interpolated proxy -- summing
//    force there would double-count against whichever cell actually OWNS
//    that physical point. Only isInterior cells contribute.
//
// 2. Cross-level weighting. Fx/Fy here are a per-CELL momentum exchange, not
//    normalized by cell size or by this level's own timestep, so a raw
//    unweighted sum would not integrate to the same total regardless of which
//    level owns a region (the exact invariance Milestone 8's own validation
//    checks). The weight is dx_L^1, and BOTH factors matter:
//
//      cell mass  ~ rho * dx_L^2   (2D volume measure)
//      timestep     dt_L = dx_L    (acoustic scaling: dx and dt halve together)
//      force = mass * du / dt   ->  dx_L^2 / dx_L  =  dx_L
//
//    An earlier version used dx_L^2, applying only the volume measure and
//    silently dropping the 1/dt_L factor -- level L runs 2^L substeps per L0
//    macro-step, but this pass runs ONCE per macro-step and reads one
//    substep's momentum exchange, so the missing factor is exactly 2^L =
//    1/dx_L. Measured on the cylinder harness at Re=100: that bug cost 2x at
//    L1 and 4x at L2 (Cd 0.943 -> 1.430 at N=2, and the N=3 case went from
//    unusable to inside the literature band). The bounce-back branch uses the
//    SAME dx^1 -- not (as its old comment claimed) because one is a perimeter
//    integral and the other a volume integral, but because the mass and
//    timestep factors combine to dx^1 either way. Live-verified there too:
//    dx^2 gave Cd=0.631 (target 1.35) on the N=2 cylinder case, dx^1 gives
//    1.262.
//
// FINEST-WINS MASKING IS GONE (plans/2D-backport.md B4-3), along with its
// HAS_CHILD override, the childBlockSlot binding it read, and the levelParams
// nbx/nby/hasChild reads that served it. Only the finest level's force pass is
// dispatched now. `average` keeps a parent's cells populated under an active
// child, so summing every level unconditionally would double-count the same
// physical drag; the premise that makes masking unnecessary rather than merely
// absent is the geometry-forced-refinement hard constraint -- every leaf
// within FORCE_REFINE_MARGIN of the body is already at the finest level --
// which amr2d-gpu.mjs's checkGeometryCoverageOnGPU asserts on every AMR page
// and tools/validate-amr-invariants.js gates periodically through a run.
//
// MEASURED BEFORE DELETING, not argued. This record lived in the dense force
// pass's header and moves here with U7-6f rather than going with the file.
// debugForceBreakdown ran each level's pass in isolation; with the masking
// still in place the coarser levels' raw i32 accumulators (FSCALE = 1e7)
// read, at 8192 steps:
//
//   levels=2            L0 0            L1 237344  (finest)
//   levels=3            L0 0    L1 1    L2 214591  (finest)
//   levels=2 bounceback L0 0            L1 201702  (finest)
//   levels=3 bounceback L0 0    L1 0    L2 207126  (finest)
//
// EXACTLY zero, bar a single 1e-7 unit on one config -- one workgroup's
// truncated partial (see the FSCALE header below), 5e-6 of the total and
// ~100x below the ~1e-3 reproducibility floor AMR Cd already has. Dead code,
// demonstrated.

// @include "common_geometry.wgsl"
// @include "common_lattice.wgsl"
// @include "common_fpack.wgsl"
// @include "common_reduce.wgsl"

// The shared 32-byte per-level uniform, same buffer every pool shader reads.
// This declares through kEps at offset 20; the fields before it that this
// kernel does not use are declared because WGSL has no way to skip them.
struct LevelParams {
  nbx: u32,        // this level's own block-grid extent -- used to derive the
                   // tile's own (bx,by), and with dxL its physical origin.
  nby: u32,        // unused here.
  parentTau: f32,  // unused here (force doesn't touch tau at all).
  dxL: f32,        // this level's own grid spacing in L0-buffer-space units:
                   // the diffuse band, the area/line weight, and half the
                   // origin derivation all scale with it.
  hasChild: u32,   // unused since the finest-wins masking went (see header);
                   // declared only to reach kEps.
  kEps: f32,       // the diffuse band in units of this level's dx. A per-level
                   // uniform, not an override -- one pipeline serves every
                   // level, so a compile-time constant could not say anything
                   // per-level. See shaders/amr_step1.wgsl's get_chi.
}

@group(0) @binding(0) var<storage, read>       state          : CardState;
@group(0) @binding(1) var<storage, read>       f_in           : array<u32>;
@group(0) @binding(2) var<storage, read_write> forces         : array<atomic<i32>, 4>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
@group(0) @binding(4) var<uniform>             levelParams    : LevelParams;
// Diagnostic (level-2 bounce-back sign investigation): per-slot (fx,fy)
// written unconditionally by every dispatch -- lets the JS side correlate
// sign against each tile's own position instead of only ever seeing the
// grand total (debugReadSlotForces).
@group(0) @binding(5) var<storage, read_write> debugSlotForce : array<vec2<f32>>;
// This level's own blockSlot, for the ring-free gather below. Always bound;
// only read when GHOST == 0.
@group(0) @binding(6) var<storage, read>       blockSlot      : array<i32>;
// Read only by `mainStride` below.
@group(0) @binding(7) var<storage, read>       activeList     : ActiveList;
// @include "common_active_list.wgsl"
// RENUMBERED CONTIGUOUS by B3-4. The layout had holes: 4/5 were
// originX/originY (gone -- the origin is derived, see header), 7 was the
// masking's childBlockSlot (gone in B4-3) and 8 sat past the hole because
// renumbering meant landing three pages' bind groups in lockstep. It is five
// pages now and one layout instead of two, so it is done once, here, with
// boot smoke on every page as the gate.

override W : u32;
override H : u32;
override RB : u32;
// GHOST is an OVERRIDE since plans/uniform-levels.md U4-2: the ROOT level has
// no ring (amr2d.mjs's ghostDepthAtLevel(0) is 0). Default 2 keeps every
// existing pipeline byte-identical. It flows into FB = RB*2 + 2*GHOST, into
// the `isInterior` test -- which at 0 admits every thread, because a ring-free
// tile IS its interior -- and into the half-cell straddle below.
override GHOST : u32 = 2u;

// NO_PARENT: this level is the ROOT, so the half-cell straddle goes away.
//
// fineToCoarseUnit places cell j at `origin - 0.5*dxL + dxL*j`, where `origin`
// is the centre of the first PARENT cell the tile covers and the two children
// straddle it. The root's `origin` is its own first cell's centre -- there is
// nothing to straddle -- so the term is 0 there. Identical to amr_step1.wgsl's
// NO_PARENT, and for the identical reason: left in, the body's phi sits half a
// cell off the grid it is meant to reproduce, and NO SINGLE-KERNEL TEST CAN
// SEE IT, because the root stays perfectly self-consistent with the offset.
override NO_PARENT : u32 = 0u;
// FSCALE: fixed-point scale for the atomic force accumulation. Raised from
// 1e4 to 1e7 because the reduction below atomicAdds ONE TRUNCATED i32 PER
// WORKGROUP (safeFixed's i32() cast truncates toward zero), so any workgroup
// whose partial sum falls below one fixed-point unit contributes exactly
// zero -- a systematic, one-directional loss, not rounding noise. Per-cell
// contributions shrink with the level's own dx weight, so deeper levels hit
// that floor hardest: measured on the cylinder harness at Re=100, at 1e4 the
// truncation cost ~10% of the force at L1 and ~32% at L2 (Cd 1.430 -> 1.593
// at N=2, 0.943 -> 1.390 at N=3). i32 max ~2.1e9 against the +/-2e9 clamp
// still bounds |force| < 200, ~1000x the largest force either scenario
// produces. A deeper hierarchy would eventually need a real fix (float
// atomics via CAS, or a two-stage reduction) rather than more scale.
const FSCALE = 10000000f;
// Optional sharp momentum-exchange bounce-back force -- see
// amr_step1.wgsl's USE_BOUNCEBACK header for the shared rationale.
override USE_BOUNCEBACK : u32 = 0u;

// Cell-centred refinement, shared with amr_step1.wgsl: the two children of
// parent cell c sit at c -/+ dx/2, so tile-local fine index j maps to
// origin - dx/2 + dx*(j - GHOST). amr2d.mjs's fineToCoarseUnit is the host
// twin. (Both files once hardcoded level 1's own dx=0.5 here, which was a
// real bug for every deeper level -- see amr_step1.wgsl.)
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

// One gathered source cell, resolved against the OWNING same-level tile when
// it leaves this one.
//
// THE RING-FREE PATH, derived from GHOST rather than flagged -- see
// amr_criterion_pool.wgsl's tapVel for the full argument, which is the same
// one. At GHOST == 0 a slot is exactly its own 2*RB x 2*RB cells, so the clamp
// the ringed path uses would fold a cell back onto itself instead of reaching
// the neighbour, and the dense kernel it must reproduce wraps periodically
// over the whole domain.
//
// The rule is amr2d.mjs's resolveSource. The `< 0` fallback is unreachable at
// the root, which is always full; it clamps rather than inventing a value, so
// a sparse ring-free level would degrade exactly the way the ringed path does
// rather than in some third way.
// RING_FREE_FORCE (plans/2D-backport.md B6-2). The diffuse branch below
// re-does the streaming gather to get rho and u*, and on a RINGED level it
// clamped its sources into the tile's own ring -- where the STEP does not
// read: the step reaches a same-level neighbour's interior directly
// (DIRECT_GHOST, amr_step1.wgsl). So at every tile edge near the body the
// force was built from a different gather than the one the fluid felt. On the
// interp path the ring held a collided, roughly-neighbour-like state and the
// discrepancy hid; on the explode path the ring is an uncollided inbox/outbox
// and it did not: a FULLY REFINED amr-N2-diffuse (no coarse seam anywhere)
// read Cd 1.655 on interp and 1.778 on explode, while bounce-back -- whose
// link sum reads only the cell's own data -- did not move. With this set,
// sources outside the interior resolve exactly as the step's do: into the
// same-level neighbour's interior, falling back to the ring only where there
// is no neighbour (a coarse seam, which B4's geometry rule keeps away from the
// body). Default 0 keeps the published numbers byte-identical until the
// default is deliberately flipped.
override RING_FREE_FORCE : u32 = 0u;

fn srcCellRinged(slot: u32, blockID: i32, sx: i32, sy: i32, FB: u32) -> u32 {
  let RB2 = i32(RB * 2u);
  let G = i32(GHOST);
  let offX = select(select(0, 1, sx >= G + RB2), -1, sx < G);
  let offY = select(select(0, 1, sy >= G + RB2), -1, sy < G);
  let clamped = slot * (FB * FB) + u32(clamp(sy, 0, i32(FB) - 1)) * FB + u32(clamp(sx, 0, i32(FB) - 1));
  if (offX == 0 && offY == 0) { return clamped; }
  let bx = u32(blockID) % levelParams.nbx;
  let by = u32(blockID) / levelParams.nbx;
  let tbx = u32((i32(bx) + offX + i32(levelParams.nbx)) % i32(levelParams.nbx));
  let tby = u32((i32(by) + offY + i32(levelParams.nby)) % i32(levelParams.nby));
  let s = blockSlot[tby * levelParams.nbx + tbx];
  if (s < 0) { return clamped; }
  return u32(s) * (FB * FB) + u32(sy - offY * RB2) * FB + u32(sx - offX * RB2);
}

fn srcCellResolved(slot: u32, blockID: i32, sx: i32, sy: i32, FB: u32) -> u32 {
  var nx = sx; var ny = sy;
  let bx = u32(blockID) % levelParams.nbx;
  let by = u32(blockID) / levelParams.nbx;
  var tbx = bx; var tby = by;
  if (nx < 0)             { nx += i32(FB); tbx = (bx + levelParams.nbx - 1u) % levelParams.nbx; }
  else if (nx >= i32(FB)) { nx -= i32(FB); tbx = (bx + 1u) % levelParams.nbx; }
  if (ny < 0)             { ny += i32(FB); tby = (by + levelParams.nby - 1u) % levelParams.nby; }
  else if (ny >= i32(FB)) { ny -= i32(FB); tby = (by + 1u) % levelParams.nby; }
  let s = blockSlot[tby * levelParams.nbx + tbx];
  if (s < 0) {
    return slot * (FB * FB) + u32(clamp(sy, 0, i32(FB) - 1)) * FB + u32(clamp(sx, 0, i32(FB) - 1));
  }
  return u32(s) * (FB * FB) + u32(ny) * FB + u32(nx);
}

fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, levelParams.kEps * levelParams.dxL);
}

// CONTAINMENT THAT COUNTS ITSELF. Ported from amr2d/backport's B6-9c (1677fb8).
//
// The substitution is correct and the SILENCE was not. NaN is never a valid
// force, and neither is one past the fixed-point range, so both branches below
// are "cannot happen" guards -- and they returned a plausible number when it
// did. Zero force is the worst plausible number available: it is exactly what a
// body in still fluid feels, so a wrecked flow reads as a calm one. Measured on
// that branch: the whole level-2 pool NaN, and the only gate watching the body
// reported OK at every checkpoint for 8192 steps.
//
// So keep the containment -- one sick cell must not stream Inf through the
// field or make the float->i32 cast implementation-defined -- and RECORD that
// it fired, in forces[3]: the accumulator's fourth slot, which nothing else
// reads or clears, so the count is sticky from the host's reset (which zeroes
// all four). Counted per call, i.e. per reduced partial, not per cell: nonzero
// means the force reduction substituted a value, and never anything else.
// Read back by amr2d-gpu.mjs's checkFieldFinite.
//
// THE NaN TEST IS ON THE BITS, NOT `x != x`. WGSL lets an implementation
// assume floats are finite, so `x != x` may be folded to false -- and measured
// on this machine's Dawn/Vulkan it was: with every population on level 2 set
// to NaN and 256 steps run, the `x != x` form counted ZERO substitutions. That
// means the NaN branch of this containment had never done anything either. An
// exponent of all ones is NaN or Inf, and an integer test cannot be assumed
// away. Healthy runs never take the branch, so they are bit-identical.
fn safeFixed(x: f32) -> i32 {
    let nonFinite = (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u;
    if (nonFinite || x > 2.0e9f || x < -2.0e9f) { atomicAdd(&forces[3], 1); }
    let s = select(x, 0.0f, nonFinite);
    return i32(clamp(s, -2.0e9f, 2.0e9f));
}

// FORCE_CULL (plans/uniform-levels.md S8-7): a workgroup whose 8x8 sub-tile
// lies entirely beyond the body's reach returns before reading any `f`.
// EXACT, not approximate -- it only skips workgroups whose sum is provably
// zero already:
//   diffuse      a cell contributes only if chi >= 1e-6 (below), i.e.
//                phi < eps * atanh(1 - 2e-6) = 7.25 eps;
//   bounce-back  only if phi >= 0 with a SOLID source one link away, i.e.
//                phi < sqrt(2) dx.
// phi is a 1-Lipschitz distance (or, past SDF_FAR, a proven LOWER bound on
// one), so phi(centre) minus the box's half-diagonal bounds every cell in it
// from below -- the same argument common_geometry.wgsl's nearBodyBox rests
// on. The margin, 9 eps + 2 dx, clears both reaches and the Newton residual.
//
// It matters because BODY_SUBSTEP (main-amr.js) runs this pass before every
// finest substep: 2^(levels-1) times per root step over every finest tile,
// of which only the body's shell can contribute. ?forcecull=0 on the card
// page restores the full sweep.
override FORCE_CULL : u32 = 1u;
var<workgroup> wg_cull : u32;

fn cullWorkgroup(wid: vec3<u32>, slot: u32) -> u32 {
  let blockID = slotToBlock[slot];
  if (blockID < 0) { return 1u; }
  let lo = max(wid.xy * 8u, vec2<u32>(GHOST, GHOST));
  let hi = min(wid.xy * 8u + vec2<u32>(7u, 7u), vec2<u32>(GHOST + RB * 2u - 1u, GHOST + RB * 2u - 1u));
  if (lo.x > hi.x || lo.y > hi.y) { return 1u; }   // ring-only workgroup
  let bx = u32(blockID) % levelParams.nbx;
  let by = u32(blockID) / levelParams.nbx;
  let originX_L0 = f32(bx * RB) * 2.0f * levelParams.dxL;
  let originY_L0 = f32(by * RB) * 2.0f * levelParams.dxL;
  let pLo = vec2<f32>(fineToCoarseUnit(lo.x, originX_L0), fineToCoarseUnit(lo.y, originY_L0));
  let pHi = vec2<f32>(fineToCoarseUnit(hi.x, originX_L0), fineToCoarseUnit(hi.y, originY_L0));
  let R = length(0.5f * (pHi - pLo));
  let reach = 9.0f * levelParams.kEps * levelParams.dxL + 2.0f * levelParams.dxL;
  return select(0u, 1u, get_phi(0.5f * (pLo + pHi), state) - R > reach);
}

var<workgroup> wg_fx : array<f32, 64>;
var<workgroup> wg_fy : array<f32, 64>;
var<workgroup> wg_tz : array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>
) { forceCell(gid, lid, wid, gid.z); }

// ?launch=stride (main-amr.js): a DIRECT dispatch of K workgroups in z, each
// walking this pool's active-slot list (amr_active_list.wgsl) at stride K,
// so an empty slot is never launched. K is the host's lagged estimate of the
// count and only sets the parallelism; the loop covers every entry whatever
// it is. The barrier lets the body reuse workgroup memory. `main` never reads
// `activeList`, so its layout -- every other page's -- is unchanged.
@compute @workgroup_size(8, 8)
fn mainStride(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>
) {
  let n = activeListCount(lid);
  for (var z = wid.z; z < n; z += nwg.z) { forceCell(gid, lid, wid, activeList.slots[z]); workgroupBarrier(); }
}

fn forceCell(gid: vec3<u32>, lid: u32, wid: vec3<u32>, slot: u32) {
  if (FORCE_CULL != 0u) {
    if (lid == 0u) { wg_cull = cullWorkgroup(wid, slot); }
    if (workgroupUniformLoad(&wg_cull) != 0u) { return; }
  }
  let fx = gid.x; let fy = gid.y;
  let FB = RB * 2u + 2u * GHOST;

  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (fx < FB && fy < FB) {
    let blockID = slotToBlock[slot];
    let isInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;

    if (blockID >= 0 && isInterior) {
      {
        // This tile's physical origin in L0 units, as one multiply --
        // `block * RB * 2^-(m-1)`, and `2^-(m-1)` is `2 * dxL`. It used to
        // be a per-slot buffer read; see the header, and amr2d.mjs's
        // tileOriginL0 for the host statement of the same closed form.
        let bx = u32(blockID) % levelParams.nbx;
        let by = u32(blockID) / levelParams.nbx;
        let originX_L0 = f32(bx * RB) * 2.0f * levelParams.dxL;
        let originY_L0 = f32(by * RB) * 2.0f * levelParams.dxL;
        let bufX = fineToCoarseUnit(fx, originX_L0);
        let bufY = fineToCoarseUnit(fy, originY_L0);
        let p = vec2<f32>(bufX, bufY);

        let phi = get_phi(p, state);
        let poolPlaneStride = arrayLength(&f_in) / 9u;
        let cell = slot * (FB * FB) + fy * FB + fx;
        // dx_L^1 for BOTH branches: a cell's mass scales as dx_L^2 but this
        // level's timestep is dt_L = dx_L (acoustic scaling), and force is
        // mass*du/dt, so the two factors combine to dx_L^1. See
        // amr_force1.wgsl's header point 2 -- the diffuse branch previously
        // used dx_L^2, applying the volume measure but dropping 1/dt_L,
        // which cost a factor of 2^L (4x at level 2).
        let areaWeight = levelParams.dxL;
        let lineWeight = levelParams.dxL;

        if (USE_BOUNCEBACK != 0u) {
          // See lbm_force.wgsl's identical branch for the MEM formula;
          // amr_step1.wgsl's own USE_BOUNCEBACK header for why the sharp
          // test uses the UNCLAMPED source position.
          if (phi >= 0f) {
            var rx = p.x - state.cx;
            var ry = p.y - state.cy;
            rx -= f32(W) * round(rx / f32(W));
            ry -= f32(H) * round(ry / f32(H));
            let usx = state.vx - state.omega * ry;
            let usy = state.vy + state.omega * rx;

            for (var i = 0u; i < 9u; i++) {
              let srcBufX = fineToCoarseUnitI(i32(fx) - ex[i], originX_L0);
              let srcBufY = fineToCoarseUnitI(i32(fy) - ey[i], originY_L0);
              if (get_phi(vec2<f32>(srcBufX, srcBufY), state) < 0f) {
                let f_opp = fUnpack(f_in[fIdx(opp[i], poolPlaneStride, cell)], opp[i]);
                let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
                fx_body += -f32(ex[i]) * (2f * f_opp + corr) * lineWeight;
                fy_body += -f32(ey[i]) * (2f * f_opp + corr) * lineWeight;
              }
            }
            tz_body = rx * fy_body - ry * fx_body;
          }
        } else {
          let chi = get_chi(phi);
          if (chi >= 1e-6) {
            var rho = 0f; var ux_star = 0f; var uy_star = 0f;
            for (var i = 0u; i < 9u; i++) {
              let sx = i32(fx) - ex[i];
              let sy = i32(fy) - ey[i];
              var srcCell = slot * (FB * FB)
                          + u32(clamp(sy, 0, i32(FB) - 1)) * FB
                          + u32(clamp(sx, 0, i32(FB) - 1));
              if (GHOST == 0u) { srcCell = srcCellResolved(slot, blockID, sx, sy, FB); }
              else if (RING_FREE_FORCE != 0u) { srcCell = srcCellRinged(slot, blockID, sx, sy, FB); }
              let fi = fUnpack(f_in[fIdx(i, poolPlaneStride, srcCell)], i);
              rho     += fi;
              ux_star += fi * f32(ex[i]);
              uy_star += fi * f32(ey[i]);
            }
            ux_star /= max(rho, 1e-6f); uy_star /= max(rho, 1e-6f);

            var rx = p.x - state.cx;
            var ry = p.y - state.cy;
            rx -= f32(W) * round(rx / f32(W));
            ry -= f32(H) * round(ry / f32(H));
            let usx = state.vx - state.omega * ry;
            let usy = state.vy + state.omega * rx;

            let Fx = rho * chi * (usx - ux_star);
            let Fy = rho * chi * (usy - uy_star);

            fx_body = -Fx * areaWeight;
            fy_body = -Fy * areaWeight;
            tz_body = rx * fy_body - ry * fx_body;
          }
        }
      }
    }
  }

  wg_fx[lid] = fx_body;
  wg_fy[lid] = fy_body;
  wg_tz[lid] = tz_body;
  workgroupBarrier();

  // Parallel tree reduction (common_reduce.wgsl) -- replaces a 64-step
  // serial sum that lane 0 used to run alone. See that file for the
  // on-device measurement that motivated it.
  wgReduceSum3(lid);
  if (lid == 0u) {
    let sum_fx = wg_fx[0];
    let sum_fy = wg_fy[0];
    let sum_tz = wg_tz[0];
    atomicAdd(&forces[0], safeFixed(sum_fx * FSCALE));
    atomicAdd(&forces[1], safeFixed(sum_fy * FSCALE));
    atomicAdd(&forces[2], safeFixed(sum_tz * FSCALE));
    debugSlotForce[slot] = vec2<f32>(sum_fx, sum_fy);
  }
}
