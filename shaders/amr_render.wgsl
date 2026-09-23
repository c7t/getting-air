// Visualization shader with smooth analytical mask and vorticity calculation.

// @include "common_geometry.wgsl"
// @include "common_vortcolor.wgsl"

// BINDING 0 IS LEVEL 0'S VELOCITY: the ROOT POOL's, addressed by
// `rootCellIndex` through binding 12's indirection (plans/uniform-levels.md
// U7-6b). It carried the dense L0 grid in block8 layout until U7-6f, selected
// by a ROOT_IS_POOL override -- the two addressings are not interchangeable,
// so that flag moved the BUFFER as well as the arithmetic. There is one
// representation of level 0 now and no flag.
@group(0) @binding(0) var<storage, read> vel         : array<f32>;
@group(0) @binding(1) var<storage, read> state       : CardState;
// ONE VELOCITY/INDIRECTION PAIR PER POOL LEVEL, AND THE WALK OVER THEM IS A
// LOOP (plans/uniform-levels.md U6).
//
// It used to be two hard-coded tiers -- `vel_pool`/`blockSlot` for level 1 and
// `vel_pool2`/`blockSlot2` for level 2 -- with the compositing written out
// twice. There was no third pair, so at `?levels=4` a level-3 tile was
// refined, solved, stepped twice per level-2 substep, force-reduced and
// invariant-checked, and then DRAWN AS ITS LEVEL-2 PARENT. The finest level in
// the hierarchy, which is the whole reason the hierarchy exists, was invisible.
// `tools/validate-render-levels.js` measured it directly: perturbing level 3's
// velocity pool by u=(9,9) over 446400 cells left the picture byte-identical.
//
// WGSL CANNOT INDEX AN ARRAY OF STORAGE BUFFERS, so the buffers stay one
// binding each and `poolVel`/`poolSlotOf` below are a 4-way `if` ladder. That
// ladder is the irreducible part. What is NOT irreducible -- and is what
// actually had the bug -- is the per-level block arithmetic, the bilinear
// sample, the finest-wins precedence and the outline colour, all of which were
// copy-pasted per tier and are now written ONCE inside `for (m = 1; m <=
// N_POOL_LEVELS; m++)`.
//
// Levels above `N_POOL_LEVELS` are bound to a harmless duplicate of level 1's
// buffers and never read: the loop stops, rather than the bindings being
// absent. Reading a dummy `blockSlot` out of bounds is NOT safe -- Dawn clamps
// to element 0, other stacks need not -- which is the hazard the old
// `HAS_LEVEL2` gate existed for, and the loop bound is now what enforces it.
@group(0) @binding(2) var<storage, read> vel_pool    : array<f32>;
@group(0) @binding(3) var<storage, read> blockSlot   : array<i32>;
@group(0) @binding(4) var<uniform>       overlayOpacity : f32;     // refinement-coverage overlay opacity [0,1]
@group(0) @binding(5) var<storage, read> vel_pool2   : array<f32>;
@group(0) @binding(6) var<storage, read> blockSlot2  : array<i32>;
@group(0) @binding(8) var<storage, read> vel_pool3   : array<f32>;
@group(0) @binding(9) var<storage, read> blockSlot3  : array<i32>;
@group(0) @binding(10) var<storage, read> vel_pool4  : array<f32>;
@group(0) @binding(11) var<storage, read> blockSlot4 : array<i32>;
// The ROOT pool's block->slot indirection, for binding 0.
@group(0) @binding(12) var<storage, read> blockSlot0 : array<i32>;
// Quadtree outline opacity [0,1] -- optional, off (0) by default. Separate
// uniform from overlayOpacity (the coverage FILL) so the two can be toggled
// independently -- an outline-only view is useful precisely when the fill
// is turned down/off to see the underlying flow field.
@group(0) @binding(7) var<uniform>       outlineOpacity : f32;

override W : u32;
override H : u32;
const BLOCK = 8u;

// Milestone 4 (plans/AMR.md): fine-region visual overlay, so a seam or
// discontinuity at the coarse-fine interface is immediately visible rather
// than only showing up in a numerical diff. Pool-aware -- supersedes
// Milestone 2's single-fixed-region version.
override RB : u32;
const GHOST = 2u;

// How many POOL levels this configuration actually has, i.e. N_LEVELS - 1.
// The walk below runs m = 1 .. N_POOL_LEVELS and never touches a deeper
// binding, which is what makes the dummy bindings safe (see their note above).
//
// MAX_RENDER_POOL_LEVELS IS 4 AND THE PAGE MUST REFUSE ABOVE IT. A cap that
// silently drops the finest level is precisely the defect U6 exists to fix, so
// a configuration this shader cannot draw has to fail at init with a message
// rather than render a lie. main-amr.js checks it.
override N_POOL_LEVELS : u32 = 1u;

// A buffer cell, addressed through the ROOT POOL.
//
// THE ROOT TILE HAS NO GHOST RING, and that is the one thing that stops level
// 0 from simply joining `poolVel`'s ladder below. `amr2d.mjs`'s
// ghostDepthAtLevel(0) is 0 -- a ring exists to hold a parent interface and
// the root has no parent -- so the root tile is 2*RB cells square where every
// other level's is 2*RB + 2*GHOST, and there is no GHOST offset to add to the
// local coordinate. Writing this as `poolVel(0, ...)` would have silently
// applied level>=1's geometry to it.
//
// The root is also ALWAYS FULL (slots == nblocks, every block active), which
// is why there is no "is this level active here" test: level 0 covers every
// pixel, which is exactly what makes it the base case of the walk rather than
// one more iteration of it.
fn rootCellIndex(cx: u32, cy: u32) -> u32 {
  let tile = RB * 2u;
  let nbx = W / tile;
  let blockID = (cy / tile) * nbx + (cx / tile);
  let slot = u32(blockSlot0[blockID]);
  return slot * (tile * tile) + (cy % tile) * tile + (cx % tile);
}

// Level 0's velocity at a BUFFER cell.
fn rootVel(bx: u32, by: u32) -> vec2<f32> {
  let i = rootCellIndex(bx, by);
  return vec2<f32>(vel[i * 2u], vel[i * 2u + 1u]);
}

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

const p = array<vec2<f32>,6>(
  vec2(-1f,-1f), vec2( 1f,-1f), vec2(-1f, 1f),
  vec2(-1f, 1f), vec2( 1f,-1f), vec2( 1f, 1f)
);

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out: VSOut;
  out.pos = vec4(p[vi], 0f, 1f);
  out.uv  = p[vi] * 0.5f + 0.5f;
  return out;
}

// THE DIFFUSE BAND'S WIDTH, as a multiple of THIS level's own cell size --
// epsilon = K_EPS * dx_level. It was a bare literal here and an override only
// on the pool path, so the one number that sets how sharp the solid boundary
// is could not be swept across the whole solver (plans/2D-backport.md B7).
//
// 1.5 is the value every one of these sites already had, so the default is
// byte-identical to the previous build. ?kEps= moves all of them together.
//
// WHY IT IS WORTH A KNOB. CLAUDE.md records `dense-reference` and
// `amr-N2-diffuse` failing Cd at Re=100 and diagnoses it as diffuse-interface
// width -- the band is a fixed number of cells regardless of resolution, so
// the effective body radius exceeds the nominal one and Cd converges from
// ABOVE. The instrument that settles that is a BAND ladder at fixed
// resolution, not a resolution ladder (which moves the band and everything
// else at once), and a band ladder needs this to be a parameter.
override K_EPS : f32 = 1.5f;
fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, K_EPS);
}

fn get_uy(x: i32, y: i32) -> f32 {
  let wx = (u32(x) + W) % W;
  let wy = (u32(y) + H) % H;
  let bx = (wx + u32(state.off_x)) % W;
  let by = (wy + u32(state.off_y)) % H;
  return rootVel(bx, by).y;
}

fn get_ux(x: i32, y: i32) -> f32 {
  let wx = (u32(x) + W) % W;
  let wy = (u32(y) + H) % H;
  let bx = (wx + u32(state.off_x)) % W;
  let by = (wy + u32(state.off_y)) % H;
  return rootVel(bx, by).x;
}

// wrapf lives in common_geometry.wgsl (B5-1), included above.

// Coarse-grid vorticity at integer-centred cell (cx, cy) in WINDOW coords
// (get_ux/get_uy add the window offset and wrap). Central difference, per
// coarse cell (dx=1) => the 0.5 factor.
fn coarseOmegaCell(cx: i32, cy: i32) -> f32 {
  return (get_uy(cx + 1, cy) - get_uy(cx - 1, cy)) * 0.5f
       - (get_ux(cx, cy + 1) - get_ux(cx, cy - 1)) * 0.5f;
}

// THE BUFFER SELECTOR, and it is the one part WGSL forces to be a ladder.
// There is no way to index an array of storage buffers, so `m` picks a binding
// here and nowhere else -- every caller below works in `m` and never names a
// buffer. The `>= 4u` fallthrough rather than `== 4u` keeps the function total
// without a default that could be reached by a level the loop never visits.
fn poolVel(m: u32, slot: u32, cx: i32, cy: i32) -> vec2<f32> {
  let FBl = RB * 2u + 2u * GHOST;
  let ix = u32(clamp(cx, 0, i32(FBl) - 1));
  let iy = u32(clamp(cy, 0, i32(FBl) - 1));
  let cell = slot * (FBl * FBl) + iy * FBl + ix;
  if (m == 1u) { return vec2<f32>(vel_pool[cell * 2u],  vel_pool[cell * 2u + 1u]); }
  if (m == 2u) { return vec2<f32>(vel_pool2[cell * 2u], vel_pool2[cell * 2u + 1u]); }
  if (m == 3u) { return vec2<f32>(vel_pool3[cell * 2u], vel_pool3[cell * 2u + 1u]); }
  return vec2<f32>(vel_pool4[cell * 2u], vel_pool4[cell * 2u + 1u]);
}
fn poolSlotOf(m: u32, blockID: i32) -> i32 {
  if (m == 1u) { return blockSlot[blockID]; }
  if (m == 2u) { return blockSlot2[blockID]; }
  if (m == 3u) { return blockSlot3[blockID]; }
  return blockSlot4[blockID];
}

// Fine-grid vorticity at fine cell (cx, cy) of a level-m slot.
//
// The ghost ring (cells [0,1] and [FB-2,FB-1]) is c2f-filled from the parent,
// so a fine stencil that reaches into the ring stays consistent with the level
// above -- this is what lets the perimeter fine curl match the coarse curl
// without a hard operator switch. Clamp keeps out-of-range taps in the ring.
//
// NORMALISATION IS PER-COARSE-UNIT AT EVERY LEVEL, which is what makes the
// levels directly comparable at an interface. Level m's spacing is 2^-m coarse
// units and the central difference spans +/-1 fine cell, so the factor is
// 1/(2 * 2^-m) = 2^(m-1): 1 at level 1, 2 at level 2, and so on. The two
// hand-written copies this replaces carried exactly those two constants.
fn fineOmegaAt(m: u32, slot: u32, cx: i32, cy: i32) -> f32 {
  let uyp = poolVel(m, slot, cx + 1, cy).y;
  let uym = poolVel(m, slot, cx - 1, cy).y;
  let uxp = poolVel(m, slot, cx, cy + 1).x;
  let uxm = poolVel(m, slot, cx, cy - 1).x;
  return ((uyp - uym) - (uxp - uxm)) * exp2(f32(m) - 1.0f);
}

// This level's own outline colour. White at level 1, then yellow, cyan,
// magenta -- distinct hues rather than a ramp, because the question the
// outline answers is "which level owns this footprint", not "how deep".
fn levelOutlineColor(m: u32) -> vec3<f32> {
  if (m == 1u) { return vec3(1.0f, 1.0f, 1.0f); }
  if (m == 2u) { return vec3(1.0f, 1.0f, 0.0f); }
  if (m == 3u) { return vec3(0.0f, 1.0f, 1.0f); }
  return vec3(1.0f, 0.0f, 1.0f);
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  // SUB-CELL PAN. The moving window pans the FIELD by state.off_x/off_y, which
  // are integers (floor of the accumulated displacement, mod W/H -- see
  // amr_physics.wgsl step 5), while the CARD is drawn at state.cx/cy, which
  // carry the leftover FRACTION of that same displacement. So over one cell
  // crossing the card slides smoothly by up to a cell and the field snaps back
  // by exactly one cell at the moment the fraction wraps. The two stay
  // consistent with each other -- the card is anchored in window space, which
  // is the whole design -- but the composite wobbles by up to one cell in the
  // VIEWPORT. At W=512 that is 1/512 of the frame and invisible; at W=64 it is
  // an obvious jitter of the card and the field together, which is how it was
  // reported (on a phone, "more obvious at the lower resolution").
  //
  // Fix: shift the SAMPLE POINT by that same fraction, for both the body SDF
  // and the field lookup, so the two shift together and nothing moves relative
  // to anything else. The card then lands exactly at the window centre (the
  // pixel at fx = W/2 evaluates phi at W/2 + subX == state.cx) and the field
  // scrolls smoothly sub-cell instead of a cell at a time, which the bilinear
  // reconstruction below already resolves for free.
  //
  // Taken from x_total/y_total rather than from cx - W/2, so it does not
  // assume where a scenario put initial_cx/cy. This is RENDER ONLY -- off_x/
  // off_y, cx/cy and every physics path are untouched, and a scenario with a
  // pinned body (v=0, so x_total stays 0) is byte-identical.
  let subX = state.x_total - floor(state.x_total);
  let subY = state.y_total - floor(state.y_total);
  let fx = uv.x * f32(W) + subX; let fy = (1.0 - uv.y) * f32(H) + subY;

  // The render draws the WINDOW; under ?window=0 the body lives in buffer
  // coordinates, so this is the one place that converts the other way.
  let chi = get_chi(get_phi(windowToBufferPos(vec2(fx, fy), state), state));

  // Level-consistent vorticity RECONSTRUCTION. Compute the discrete curl as a
  // cell-centred scalar FIELD (per coarse cell, and per fine cell in refined
  // blocks) and bilinearly interpolate that field. This replaces the old
  // approach of differencing NEAREST-sampled velocity per pixel, which made
  // omega piecewise-constant per cell (visible stair-steps) and hard-switched
  // between a coarse (+/-1 coarse cell) and fine (+/-1 fine cell) operator at
  // perimeters (a 1-band seam). The velocity field is smooth and continuous
  // (verified), so interpolating the curl field is a FAITHFUL reconstruction,
  // not a blur/mask.

  // Coarse field: bilinear over the 4 surrounding integer-centred coarse cells.
  let cx0 = i32(floor(fx)); let cy0 = i32(floor(fy));
  let ctx = fx - f32(cx0);  let cty = fy - f32(cy0);
  var omega = mix(
      mix(coarseOmegaCell(cx0, cy0),     coarseOmegaCell(cx0 + 1, cy0),     ctx),
      mix(coarseOmegaCell(cx0, cy0 + 1), coarseOmegaCell(cx0 + 1, cy0 + 1), ctx),
      cty);

  // If this pixel falls in a refined block, override with the fine field
  // (bilinear over the 4 surrounding fine cells). Block resolution uses the
  // same nearest-coarse-cell-centre (+0.5) rule as the sim. The fine-local
  // offset uses PERIODIC distance (bufX-originX wrapped): without it, a
  // position in the top half-cell of the buffer wrap resolves its block to the
  // opposite side and maps to a bogus ghost cell -- a real moving-wrap seam.
  // Fine-edge cells read the ghost ring via poolVelCell. Internal fine-fine
  // block edges are exactly C0 (ghosts copy the neighbour's real interior). At
  // a TRUE fine/coarse perimeter the coarse-adjacent ghosts are c2f-derived and
  // evolve across the 2 fine substeps (multi-rate), so the fine curl there is
  // close-but-not-bit-identical to the coarse curl -- a small bounded residual,
  // far smaller than the old hard-fallback stencil-width seam (no hard switch).
  let bufX = wrapf(fx + state.off_x, f32(W));
  let bufY = wrapf(fy + state.off_y, f32(H));
  let nbx = W / BLOCK;
  // THE WALK. Finest-active-level wins, one level per iteration, and every
  // quantity below is the same expression evaluated at this level's own scale
  // (plans/uniform-levels.md U6).
  //
  // Level m's tile covers RB * 2^(1-m) L0 units, its block grid is nbx *
  // 2^(m-1) wide, and a point's fine coordinate inside the tile is
  // GHOST + 2^m * (offset within the tile) + 0.5. Check those against the two
  // tiers this replaces: level 1 gave footprint RB, grid nbx, coefficient 2;
  // level 2 gave RB/2, 2*nbx, coefficient 4. Both fall out of the same three
  // lines now.
  //
  // THE LOOP STOPS AT THE FIRST ABSENT TILE, and that is a statement about the
  // quadtree rather than an optimisation: a level-(m+1) tile exists only as a
  // quad carved from an active level-m parent, so once this pixel's level-m
  // tile is absent no deeper one can cover it. That is also what keeps the
  // deeper `blockSlot` reads in range -- the index is only ever formed from a
  // parent that exists.
  var deepest = 0u;          // finest level that actually covers this pixel
  var deepestSlot = 0u;
  var deepestLocal = vec2<f32>(0.0f, 0.0f); // offset within that tile, L0 units
  var deepestFootprint = f32(RB);
  {
    var footprint = f32(RB);   // level 1's tile, in L0 units
    var nbxL = nbx;
    for (var m = 1u; m <= N_POOL_LEVELS; m++) {
      // THE HALF-CELL SHIFT IS PER LEVEL, AND GETTING THAT WRONG IS VISIBLE.
      //
      // A tile is picked by which CELL of the level above contains this point,
      // so the shift is half a level-(m-1) cell = 2^-m in L0 units, i.e.
      // 1/dens. That puts the tile boundary on a cell centre and leaves the
      // bilinear stencil reaching exactly 0.5 FINE CELLS into the ring, at
      // every level -- which is the reach level 1 has always had, and the ring
      // is c2f-filled precisely so that tap is consistent across the seam.
      //
      // Written as a flat 0.5 it is half an L0 cell, which is 2^(m-1) FINE
      // cells: 0.5 at level 1, 1.5 at level 2, 3.5 at level 3, 7.5 at level 4.
      // GHOST is 2, so from level 3 down the tap lands outside the ring
      // entirely, `poolVel`'s clamp pins it to the tile edge, and a band of
      // pixels along every level-3 tile boundary reads one frozen value. On
      // screen that is a dark lattice over the refined region -- caught by
      // eye, not by the reachability gate, which only asks whether a level
      // changes the picture at all.
      let dens = exp2(f32(m));
      let shift = 1.0f / dens;
      let bX = u32(wrapf(bufX + shift, f32(W)) / footprint);
      let bY = u32(wrapf(bufY + shift, f32(H)) / footprint);
      let sl = poolSlotOf(m, i32(bY * nbxL + bX));
      if (sl < 0) { break; }
      var dx = bufX - f32(bX) * footprint; dx -= f32(W) * round(dx / f32(W));
      var dy = bufY - f32(bY) * footprint; dy -= f32(H) * round(dy / f32(H));
      deepest = m;
      deepestSlot = u32(sl);
      deepestLocal = vec2<f32>(dx, dy);
      deepestFootprint = footprint;
      footprint = footprint * 0.5f;
      nbxL = nbxL * 2u;
    }
  }
  // Quadtree outline: additive line color, drawn along each ACTIVE block's
  // own 4 edges (not a fixed background grid -- only where a level actually
  // owns this footprint), one color per level so the tree structure itself
  // is legible, not just "some refinement happened here" (that's what the
  // fill overlay below already shows). LINE_WIDTH is in L0-buffer-space
  // units, so it renders thinner at higher sim resolution, same as every
  // other buffer-space-native visual element here.
  const LINE_WIDTH = 0.15f;
  var outlineColor = vec3(0.0f);
  if (deepest > 0u) {
    // Sample the level the walk settled on. `exp2(f32(deepest))` is the
    // fine-cells-per-L0-unit density -- 2 at level 1, 4 at level 2 -- which is
    // the one coefficient the two hand-written tiers spelled out by hand. It is
    // the same expression the walk used for its own shift, and it has to stay
    // that way: the shift is defined as half a cell of the level ABOVE, and
    // this is what converts the offset it produced into fine coordinates.
    let dens = exp2(f32(deepest));
    let fxc = f32(GHOST) + dens * deepestLocal.x + 0.5;
    let fyc = f32(GHOST) + dens * deepestLocal.y + 0.5;
    let fx0 = i32(floor(fxc)); let fy0 = i32(floor(fyc));
    let ftx = fxc - f32(fx0);  let fty = fyc - f32(fy0);
    omega = mix(
        mix(fineOmegaAt(deepest, deepestSlot, fx0, fy0),         fineOmegaAt(deepest, deepestSlot, fx0 + 1, fy0),     ftx),
        mix(fineOmegaAt(deepest, deepestSlot, fx0, fy0 + 1),     fineOmegaAt(deepest, deepestSlot, fx0 + 1, fy0 + 1), ftx),
        fty);

    // This tile's own edge distance, periodic within its own footprint. Only
    // the FINEST level's outline is drawn, which is the same finest-wins
    // precedence the omega field just used -- level 2's yellow took over from
    // level 1's white before, and it still does.
    let edgeDist = min(min(deepestLocal.x, deepestFootprint - deepestLocal.x),
                       min(deepestLocal.y, deepestFootprint - deepestLocal.y));
    if (edgeDist < LINE_WIDTH) { outlineColor = levelOutlineColor(deepest); }
  }

  // Blue for clockwise (negative), red for counter-clockwise (positive).
  // Shared verbatim with the dense view so the two are comparable by eye --
  // see shaders/common_vortcolor.wgsl, including why this mapping is level-
  // INdependent even though refined regions legitimately show sharper cores.
  var c = vorticityColor(omega);

  // Refined-block coverage overlay: additive green (not a mix toward gray --
  // a mix is barely visible against the near-black low-vorticity
  // background where coverage most needs to be legible) over the whole
  // footprint of the tile the walk settled on, not just its sampled interior.
  // Reuses `deepest` already computed above at no extra cost. Canvas output is
  // unorm, so this saturates harmlessly in already-bright (high-vorticity or
  // solid-body) regions.
  //
  // Milestone 10 used two shades so the two refinement tiers were visually
  // distinguishable; with the walk there can be more than two, so the green
  // brightens with DEPTH on the same ramp the two fixed shades sat on --
  // (0, 0.22, 0) at level 1 and (0, 0.32, 0.12) at level 2, continued.
  if (deepest > 0u) {
    let t = f32(deepest - 1u);
    c += (vec3(0.0, 0.22, 0.0) + vec3(0.0, 0.10, 0.12) * t) * overlayOpacity;
  }

  // Blend with solid color
  let solid_color = vec3(1.0, 0.8, 0.4);
  c = mix(c, solid_color, chi);

  // Quadtree outline, drawn last (on top of the solid body too) so block
  // structure stays legible even where it crosses the body -- mix rather
  // than additive, since the line colors are already saturated and an
  // additive white/yellow would blow out unpredictably over a bright
  // vorticity or solid-body pixel.
  c = mix(c, outlineColor, outlineOpacity * step(0.5f, dot(outlineColor, outlineColor)));

  return vec4(c, 1.0);
}
