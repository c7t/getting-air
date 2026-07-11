// Visualization shader with smooth analytical mask and vorticity calculation.

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

@group(0) @binding(0) var<storage, read> vel         : array<f32>;
@group(0) @binding(1) var<storage, read> state       : CardState;
@group(0) @binding(2) var<storage, read> vel_pool    : array<f32>; // Milestone 4: fine pool (level 1)
@group(0) @binding(3) var<storage, read> blockSlot   : array<i32>; // Milestone 4: coarse block -> pool slot (level 1)
@group(0) @binding(4) var<uniform>       overlayOpacity : f32;     // refinement-coverage overlay opacity [0,1]
// Milestone 10: level 2's own vel_pool/blockSlot, for finest-active-level-
// wins compositing -- harmless dummy buffers (blockSlot2 all -1) when
// N_LEVELS<3, in which case level 2 is simply never "active" anywhere and
// this file's behavior is exactly today's fixed two-tier logic. See this
// file's header comment on why a third tier doesn't need a bigger
// redesign (bindless/dynamic-arity level arrays) -- this plan's own scope
// stops at N<=3 (plans/AMR-multilevel.md's own scalability notes), so one
// more fixed set of bindings, not a generalized loop, is the right size
// of change here.
@group(0) @binding(5) var<storage, read> vel_pool2   : array<f32>;
@group(0) @binding(6) var<storage, read> blockSlot2  : array<i32>;
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

// Whether level 2 exists (N_LEVELS > 2). When 0, blockSlot2/vel_pool2 are
// harmless dummy bindings and the level-2 override below MUST be gated off:
// the dummy blockSlot2 is a single -1 element, so the childBlockID index is
// out of bounds, and an OOB storage read is NOT guaranteed to return the
// in-bounds -1 (Dawn clamps to element 0, but other WebGPU stacks can return
// >=0, which falsely activates the L2 override and renders every refined
// block black). Mirror amr_manage.wgsl's own HAS_LEVEL2 gate.
override HAS_LEVEL2 : u32 = 0u;

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// See amr_step.wgsl for the full derivation; vel is laid out this way
// (Milestone 1, plans/AMR.md) instead of flat row-major.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
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

fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    var dx = p.x - state.cx;
    var dy = p.y - state.cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b;
}

fn get_chi(phi: f32) -> f32 {
    let epsilon = 1.5f;
    // Clamp tanh arg: large |arg| overflows to NaN on some GPUs (e.g. Intel Gen12LP); saturated regime is unchanged. See PR.
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}

fn get_uy(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[cellIndex(bx, by) * 2u + 1u];
}

fn get_ux(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[cellIndex(bx, by) * 2u];
}

fn wrapf(v: f32, n: f32) -> f32 {
    var r = v % n;
    if (r < 0.0) { r += n; }
    return r;
}

// Coarse-grid vorticity at integer-centred cell (cx, cy) in WINDOW coords
// (get_ux/get_uy add the window offset and wrap). Central difference, per
// coarse cell (dx=1) => the 0.5 factor.
fn coarseOmegaCell(cx: i32, cy: i32) -> f32 {
  return (get_uy(cx + 1, cy) - get_uy(cx - 1, cy)) * 0.5f
       - (get_ux(cx, cy + 1) - get_ux(cx, cy - 1)) * 0.5f;
}

// Fine-pool velocity at a fine-cell INDEX (cx, cy) within a slot. The ghost
// ring (cells [0,1] and [FB-2,FB-1]) is c2f-filled from the coarse field, so a
// fine stencil that reaches into the ring stays consistent with the coarse
// level -- this is what lets the perimeter fine curl match the coarse curl
// without a hard operator switch. Clamp keeps out-of-range taps in the ring.
fn poolVelCell(slot: u32, cx: i32, cy: i32) -> vec2<f32> {
  let FBl = RB * 2u + 2u * GHOST;
  let ix = u32(clamp(cx, 0, i32(FBl) - 1));
  let iy = u32(clamp(cy, 0, i32(FBl) - 1));
  let cell = slot * (FBl * FBl) + iy * FBl + ix;
  return vec2<f32>(vel_pool[cell * 2u], vel_pool[cell * 2u + 1u]);
}

// Fine-grid vorticity at fine cell (cx, cy) of a slot. Fine spacing is 0.5
// coarse units and the central difference spans +/-1 fine cell, so the factor
// is 1/(2*0.5) = 1 -- the SAME per-coarse-unit normalization as
// coarseOmegaCell, so the two levels are directly comparable at interfaces.
fn fineOmegaCell(slot: u32, cx: i32, cy: i32) -> f32 {
  let uyp = poolVelCell(slot, cx + 1, cy).y;
  let uym = poolVelCell(slot, cx - 1, cy).y;
  let uxp = poolVelCell(slot, cx, cy + 1).x;
  let uxm = poolVelCell(slot, cx, cy - 1).x;
  return (uyp - uym) - (uxp - uxm);
}

// Milestone 10: level 2's own vel_pool/omega, same shape as
// poolVelCell/fineOmegaCell above, one level deeper. Level 2's own spacing
// is 0.25 coarse units, so the normalizing factor is 1/(2*0.25)=2 (not 1
// like level 1) -- same per-coarse-unit convention, still directly
// comparable at interfaces.
fn poolVelCell2(slot: u32, cx: i32, cy: i32) -> vec2<f32> {
  let FBl = RB * 2u + 2u * GHOST;
  let ix = u32(clamp(cx, 0, i32(FBl) - 1));
  let iy = u32(clamp(cy, 0, i32(FBl) - 1));
  let cell = slot * (FBl * FBl) + iy * FBl + ix;
  return vec2<f32>(vel_pool2[cell * 2u], vel_pool2[cell * 2u + 1u]);
}
fn fineOmegaCell2(slot: u32, cx: i32, cy: i32) -> f32 {
  let uyp = poolVelCell2(slot, cx + 1, cy).y;
  let uym = poolVelCell2(slot, cx - 1, cy).y;
  let uxp = poolVelCell2(slot, cx, cy + 1).x;
  let uxm = poolVelCell2(slot, cx, cy - 1).x;
  return ((uyp - uym) - (uxp - uxm)) * 2.0f;
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let fx = uv.x * f32(W); let fy = (1.0 - uv.y) * f32(H);

  let chi = get_chi(get_phi(vec2(fx, fy), state));

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
  let bBX = u32(wrapf(bufX + 0.5, f32(W))) / RB;
  let bBY = u32(wrapf(bufY + 0.5, f32(H))) / RB;
  let slot = blockSlot[i32(bBY * nbx + bBX)];
  var level2Active = false;
  // Quadtree outline: additive line color, drawn along each ACTIVE block's
  // own 4 edges (not a fixed background grid -- only where a level actually
  // owns this footprint), one color per level so the tree structure itself
  // is legible, not just "some refinement happened here" (that's what the
  // fill overlay below already shows). LINE_WIDTH is in L0-buffer-space
  // units, so it renders thinner at higher sim resolution, same as every
  // other buffer-space-native visual element here.
  const LINE_WIDTH = 0.15f;
  var outlineColor = vec3(0.0f);
  if (slot >= 0) {
    let s = u32(slot);
    var dxr = bufX - f32(bBX * RB); dxr -= f32(W) * round(dxr / f32(W));
    var dyr = bufY - f32(bBY * RB); dyr -= f32(H) * round(dyr / f32(H));
    let fxc = f32(GHOST) + 2.0 * dxr + 0.5;
    let fyc = f32(GHOST) + 2.0 * dyr + 0.5;
    let fx0 = i32(floor(fxc)); let fy0 = i32(floor(fyc));
    let ftx = fxc - f32(fx0);  let fty = fyc - f32(fy0);
    omega = mix(
        mix(fineOmegaCell(s, fx0, fy0),     fineOmegaCell(s, fx0 + 1, fy0),     ftx),
        mix(fineOmegaCell(s, fx0, fy0 + 1), fineOmegaCell(s, fx0 + 1, fy0 + 1), ftx),
        fty);

    // This L1 block's own edge distance (periodic within [0,RB)), white.
    let edgeDist1 = min(min(dxr, f32(RB) - dxr), min(dyr, f32(RB) - dyr));
    if (edgeDist1 < LINE_WIDTH) { outlineColor = vec3(1.0f, 1.0f, 1.0f); }

    // Milestone 10: finest-active-level-wins. If this L1 block also has an
    // active level-2 child covering this pixel's own quadrant, override
    // again with level 2's own (denser) field. Which quadrant (0/1 on each
    // axis) is exactly dxr/dyr's own half of the RB-wide L1 footprint --
    // same test amr_force1.wgsl's masking check uses, just reading instead
    // of masking.
    let halfRB = f32(RB) * 0.5f;
    let qx = select(0u, 1u, dxr >= halfRB);
    let qy = select(0u, 1u, dyr >= halfRB);
    let nbxL2 = nbx * 2u;
    let childBlockID = (bBY * 2u + qy) * nbxL2 + (bBX * 2u + qx);
    let slot2 = blockSlot2[i32(childBlockID)];
    if (HAS_LEVEL2 != 0u && slot2 >= 0) {
      level2Active = true;
      let s2 = u32(slot2);
      // This pixel's own offset WITHIN the quadrant (L0 units, [0,halfRB)),
      // then the same fine-coordinate mapping as level 1's own above, just
      // at level 2's own 4x-of-L0 density (coefficient 4.0, not 2.0).
      let dxr2 = dxr - f32(qx) * halfRB;
      let dyr2 = dyr - f32(qy) * halfRB;
      let fxc2 = f32(GHOST) + 4.0 * dxr2 + 0.5;
      let fyc2 = f32(GHOST) + 4.0 * dyr2 + 0.5;
      let fx02 = i32(floor(fxc2)); let fy02 = i32(floor(fyc2));
      let ftx2 = fxc2 - f32(fx02);  let fty2 = fyc2 - f32(fy02);
      omega = mix(
          mix(fineOmegaCell2(s2, fx02, fy02),     fineOmegaCell2(s2, fx02 + 1, fy02),     ftx2),
          mix(fineOmegaCell2(s2, fx02, fy02 + 1), fineOmegaCell2(s2, fx02 + 1, fy02 + 1), ftx2),
          fty2);

      // This L2 quadrant's own edge distance (periodic within [0,halfRB)),
      // yellow -- takes over from level 1's white where finer, same
      // finest-wins precedence the omega field itself just used above.
      let edgeDist2 = min(min(dxr2, halfRB - dxr2), min(dyr2, halfRB - dyr2));
      if (edgeDist2 < LINE_WIDTH) { outlineColor = vec3(1.0f, 1.0f, 0.0f); }
    }
  }

  // Blue for clockwise (negative), red for counter-clockwise (positive)
  let val = clamp(omega * 80.0f, -1.0f, 1.0f);
  var c: vec3<f32>;
  if (val > 0.0) {
    c = mix(vec3(0.05, 0.05, 0.1), vec3(1.0, 0.3, 0.2), val);
  } else {
    c = mix(vec3(0.05, 0.05, 0.1), vec3(0.2, 0.5, 1.0), -val);
  }

  // Refined-block coverage overlay: additive green (not a mix toward gray --
  // a mix is barely visible against the near-black low-vorticity
  // background where coverage most needs to be legible) over the whole
  // coarse block footprint currently holding a pool slot (slot>=0), not
  // just its sampled interior. Reuses slot already computed above at no
  // extra cost. Canvas output is unorm, so this saturates harmlessly in
  // already-bright (high-vorticity or solid-body) regions.
  //
  // Milestone 10: brighter/bluer green for level 2's own quadrant
  // footprint, so the two refinement tiers are visually distinguishable,
  // not just "some refinement happened here" -- level2Active is already
  // computed per-pixel above (this level's own quadrant only, not the
  // whole L1 block, since level 2 doesn't necessarily cover all 4).
  if (level2Active) {
    c += vec3(0.0, 0.32, 0.12) * overlayOpacity;
  } else if (slot >= 0) {
    c += vec3(0.0, 0.22, 0.0) * overlayOpacity;
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
