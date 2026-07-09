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
@group(0) @binding(2) var<storage, read> vel_pool    : array<f32>; // Milestone 4: fine pool
@group(0) @binding(3) var<storage, read> blockSlot   : array<i32>; // Milestone 4: coarse block -> pool slot

override W : u32;
override H : u32;
const BLOCK = 8u;

// Milestone 4 (plans/AMR.md): fine-region visual overlay, so a seam or
// discontinuity at the coarse-fine interface is immediately visible rather
// than only showing up in a numerical diff. Pool-aware -- supersedes
// Milestone 2's single-fixed-region version.
override RB : u32;
const GHOST = 2u;

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
    return 0.5f * (1.0f - tanh(phi / epsilon));
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

// Continuous window coords -> (fine-local x, fine-local y, pool slot).
// slot = -1 if the coarse block this window position falls in isn't
// currently refined. Buffer-space-native lookup (adds off_x/off_y once,
// same inversion amr_step1.wgsl's kernel uses), matching Milestone 4's
// pool addressing scheme -- see amr_interp_c2f.wgsl's file header.
fn windowToPool(wx: f32, wy: f32) -> vec3<i32> {
    let bufX = wrapf(wx + state.off_x, f32(W));
    let bufY = wrapf(wy + state.off_y, f32(H));
    let nbx = W / BLOCK;
    let blockBX = u32(bufX) / RB;
    let blockBY = u32(bufY) / RB;
    let blockID = i32(blockBY * nbx + blockBX);
    let slot = blockSlot[blockID];
    if (slot < 0) { return vec3<i32>(0, 0, -1); }
    let originX = blockBX * RB;
    let originY = blockBY * RB;
    let fxc = f32(GHOST) + 2.0 * (bufX - f32(originX)) + 0.5;
    let fyc = f32(GHOST) + 2.0 * (bufY - f32(originY)) + 0.5;
    return vec3<i32>(i32(round(fxc)), i32(round(fyc)), slot);
}

// True only when the full 4-point vorticity stencil around the pool
// coordinate lands in the slot's "real" interior (not its ghost border,
// which holds approximate/stale data by design -- see amr_interp_c2f.wgsl).
fn inPoolInterior(fx: i32, fy: i32) -> bool {
    return fx > i32(GHOST) && fx < i32(GHOST + RB * 2u) - 1 &&
           fy > i32(GHOST) && fy < i32(GHOST + RB * 2u) - 1;
}

fn get_uy1(fx: i32, fy: i32, slot: i32) -> f32 {
    let FB = RB * 2u + 2u * GHOST;
    let cx = u32(clamp(fx, 0, i32(FB) - 1));
    let cy = u32(clamp(fy, 0, i32(FB) - 1));
    let cell = u32(slot) * (FB * FB) + cy * FB + cx;
    return vel_pool[cell * 2u + 1u];
}

fn get_ux1(fx: i32, fy: i32, slot: i32) -> f32 {
    let FB = RB * 2u + 2u * GHOST;
    let cx = u32(clamp(fx, 0, i32(FB) - 1));
    let cy = u32(clamp(fy, 0, i32(FB) - 1));
    let cell = u32(slot) * (FB * FB) + cy * FB + cx;
    return vel_pool[cell * 2u];
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let fx = uv.x * f32(W); let fy = (1.0 - uv.y) * f32(H);
  let ix = i32(fx); let iy = i32(fy);

  let chi = get_chi(get_phi(vec2(fx, fy), state));

  // Discrete vorticity: du_y/dx - du_x/dy, coarse (always computed as the
  // fallback -- see below).
  var omega = (get_uy(ix + 1, iy) - get_uy(ix - 1, iy)) * 0.5f
            - (get_ux(ix, iy + 1) - get_ux(ix, iy - 1)) * 0.5f;

  // Milestone 4 (plans/AMR.md): sample the fine pool instead of the coarse
  // grid wherever this window position falls in a currently-refined block
  // and away from its ghost border -- a seam here would mean a real
  // interpolation/averaging bug, not just a small numerical diff.
  let pool = windowToPool(fx, fy);
  if (pool.z >= 0 && inPoolInterior(pool.x, pool.y)) {
    // Central difference df/dx ~= (f(x+dx)-f(x-dx))/(2*dx). Fine cells are
    // spaced dx=0.5 coarse units apart (not the coarse grid's dx=1), so the
    // factor is 1/(2*0.5)=1, not the coarse path's 0.5 -- no separate scale
    // needed here.
    omega = (get_uy1(pool.x + 1, pool.y, pool.z) - get_uy1(pool.x - 1, pool.y, pool.z))
          - (get_ux1(pool.x, pool.y + 1, pool.z) - get_ux1(pool.x, pool.y - 1, pool.z));
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
  // coarse block footprint currently holding a pool slot (pool.z>=0), not
  // just its sampled interior. Reuses pool.z already computed above at no
  // extra cost. Canvas output is unorm, so this saturates harmlessly in
  // already-bright (high-vorticity or solid-body) regions.
  if (pool.z >= 0) {
    c += vec3(0.0, 0.22, 0.0);
  }

  // Blend with solid color
  let solid_color = vec3(1.0, 0.8, 0.4);
  c = mix(c, solid_color, chi);

  return vec4(c, 1.0);
}
