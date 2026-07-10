// Milestone 8 (plans/AMR-multilevel.md): level 1's own force/torque
// integration -- generalizes amr_force.wgsl the same way amr_step1.wgsl
// generalized amr_step.wgsl: same momentum-exchange math, dispatched over
// this level's own pool tiles (full FB*FB shape, Z=slot -- same dispatch
// shape as amr_step1.wgsl, NOT amr_average_f2c.wgsl's RB-granularity one,
// since MORE sample points per unit area is the entire point of this
// milestone: a fixed physical epsilon under-sampled the chi transition
// band at coarse resolution, aliasing the force/torque that drives the
// body's own trajectory).
//
// Two things a naive per-level copy of amr_force.wgsl would get wrong:
//
// 1. GHOST cells must NOT contribute. Unlike amr_step1.wgsl (which
//    legitimately collides/streams every cell, ghost included, since ghost
//    cells still need to evolve before the next ghost-refresh overwrites
//    them), a ghost cell here is either a copy of a NEIGHBOR's interior
//    (same-level fine-fine ghost) or a coarse-interpolated proxy -- summing
//    force there would double-count against whichever cell actually OWNS
//    that physical point. Only isInterior cells contribute.
//
// 2. Area weighting. Fx/Fy here are a per-CELL momentum exchange, not
//    normalized by cell size -- summing raw per-cell values unweighted
//    across levels would make a refined region report ~4x the coarse
//    level's total force per doubling of resolution (a 2x2 patch of fine
//    cells replacing 1 coarse cell, each of comparable magnitude, summed
//    without correction). To integrate to the SAME total regardless of
//    which level currently owns a region (the exact invariance this
//    milestone's own validation checks), each level's contribution must be
//    weighted by its own dx_L^2 relative to L0's dx_L0=1. L1's dx is a
//    fixed 0.5 (footprint-preserving with L0, same reasoning as
//    amr_step1.wgsl's literal epsilon), so AREA_WEIGHT=0.25 is a literal
//    here, not a runtime lookup (contrast amr_force1_pool.wgsl, whose
//    shared pipeline serves multiple levels and needs it from
//    levelParams.dxL instead).
//
// Finest-wins masking (see amr_force.wgsl's header for the general
// rationale): whether THIS tile is superseded by an active level-2 child
// is uniform across its whole interior (quad allocation is all-or-nothing,
// decision 3), so it's one lookup per invocation, not per-cell. HAS_CHILD
// is a compile-time override, not a runtime uniform like
// amr_force1_pool.wgsl's -- level 1 has exactly ONE dedicated pipeline (not
// shared across levels), so whether level 2 exists at all is fixed for the
// whole session, known at pipeline-creation time.

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

@group(0) @binding(0) var<storage, read>       state          : CardState;
@group(0) @binding(1) var<storage, read>       f_in           : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces         : array<atomic<i32>, 4>;
@group(0) @binding(3) var<storage, read>       slotToBlock    : array<i32>;
@group(0) @binding(4) var<storage, read>       childBlockSlot : array<i32>; // level 2's blockSlot, or a harmless dummy if HAS_CHILD=0 -- see header

override W : u32;
override H : u32;
override RB : u32;
override HAS_CHILD : u32 = 0u;
const GHOST = 2u;
const BLOCK = 8u;
const FSCALE = 10000f;
const K_EPS = 1.5f;
const AREA_WEIGHT = 0.25f; // dx_L1^2 = 0.5^2 -- see header

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

fn fineToCoarseUnit(fCoord: u32, origin: u32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

fn wrapf(v: f32, n: f32) -> f32 {
  var r = v % n;
  if (r < 0.0) { r += n; }
  return r;
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
    let epsilon = K_EPS * 0.5f; // see header -- L1's own dx is a fixed literal
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}

fn safeFixed(x: f32) -> i32 {
    let s = select(x, 0.0f, x != x);
    return i32(clamp(s, -2.0e9f, 2.0e9f));
}

var<workgroup> wg_fx : array<f32, 64>;
var<workgroup> wg_fy : array<f32, 64>;
var<workgroup> wg_tz : array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32
) {
  let fx = gid.x; let fy = gid.y;
  let slot = gid.z;
  let FB = RB * 2u + 2u * GHOST;

  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (fx < FB && fy < FB) {
    let blockID = slotToBlock[slot];
    let isInterior = fx >= GHOST && fx < GHOST + RB * 2u && fy >= GHOST && fy < GHOST + RB * 2u;

    if (blockID >= 0 && isInterior) {
      var maskedByFiner = false;
      if (HAS_CHILD != 0u) {
        let nbx1 = W / BLOCK;
        let bx1 = u32(blockID) % nbx1;
        let by1 = u32(blockID) / nbx1;
        let nbx2 = nbx1 * 2u;
        // Quadrant 0's own child block ID -- if it's active, all 4 are
        // (quad allocation is all-or-nothing, see header).
        let childBlockID = (by1 * 2u) * nbx2 + (bx1 * 2u);
        maskedByFiner = childBlockSlot[childBlockID] >= 0;
      }

      if (!maskedByFiner) {
        let nbx = W / BLOCK;
        let originX = (u32(blockID) % nbx) * RB;
        let originY = (u32(blockID) / nbx) * RB;
        let bufX = fineToCoarseUnit(fx, originX);
        let bufY = fineToCoarseUnit(fy, originY);
        let wx = wrapf(bufX - state.off_x, f32(W));
        let wy = wrapf(bufY - state.off_y, f32(H));
        let p = vec2<f32>(wx, wy);

        let phi = get_phi(p, state);
        let chi = get_chi(phi);

        if (chi >= 1e-6) {
          let poolPlaneStride = arrayLength(&f_in) / 9u;
          let cell = slot * (FB * FB) + fy * FB + fx;

          // Pull-gather within this slot's own buffer, clamped at its edge
          // (matching amr_step1.wgsl's streaming -- this is a pool tile,
          // not the periodic dense grid amr_force.wgsl reads).
          var rho = 0f; var ux_star = 0f; var uy_star = 0f;
          for (var i = 0u; i < 9u; i++) {
            let srcX = clamp(i32(fx) - ex[i], 0, i32(FB) - 1);
            let srcY = clamp(i32(fy) - ey[i], 0, i32(FB) - 1);
            let srcCell = slot * (FB * FB) + u32(srcY) * FB + u32(srcX);
            let fi = f_in[i * poolPlaneStride + srcCell];
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

          fx_body = -Fx * AREA_WEIGHT;
          fy_body = -Fy * AREA_WEIGHT;
          tz_body = rx * fy_body - ry * fx_body;
        }
      }
    }
  }

  wg_fx[lid] = fx_body;
  wg_fy[lid] = fy_body;
  wg_tz[lid] = tz_body;
  workgroupBarrier();

  if (lid == 0u) {
    var sum_fx = 0.0f;
    var sum_fy = 0.0f;
    var sum_tz = 0.0f;
    for (var i = 0u; i < 64u; i++) {
      sum_fx += wg_fx[i];
      sum_fy += wg_fy[i];
      sum_tz += wg_tz[i];
    }
    atomicAdd(&forces[0], safeFixed(sum_fx * FSCALE));
    atomicAdd(&forces[1], safeFixed(sum_fy * FSCALE));
    atomicAdd(&forces[2], safeFixed(sum_tz * FSCALE));
  }
}
