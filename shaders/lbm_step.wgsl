// Fused LBM Kernel: Pull-Streaming + Collision + Source Term
// This kernel performs a full LBM step in one pass over memory.

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

@group(0) @binding(0) var<storage, read>       state : CardState;
@group(0) @binding(1) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel   : array<f32>;

override W : u32;
override H : u32;

// Sponge relaxation target velocity. Default (0,0) reproduces the original
// quiescent far-field exactly; a validation scenario (e.g. cylinder in
// crossflow) sets these to a uniform freestream instead.
override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;

// Optional: replace the diffuse (Brinkman/Guo) volume-penalization solid
// coupling with a sharp momentum-exchange bounce-back boundary condition
// (the "standard AGAL" method) -- default 0 reproduces today's chi-based
// scheme exactly (this override, and every line it gates, is a no-op when
// left at 0, so existing scenarios/builds are byte-for-byte unaffected).
// Optional (not the new default) because it changes the numerical method,
// not just a tunable within the existing one -- direct comparison against
// the diffuse method needs BOTH still available. See lbm_force.wgsl's
// header for the momentum-exchange force formula this pairs with, and
// shaders/amr_step*.wgsl for the AMR-side copies of this same flag.
override USE_BOUNCEBACK : u32 = 0u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);
// Bounce-back pairing: opposite direction index for each of the 9 links
// (rest is self-opposite; axis pairs 1<->3, 2<->4; diagonal pairs 5<->7,
// 6<->8), matching the ex/ey table above exactly.
const opp = array<u32,9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
const CS2 = 0.33333333f; // D2Q9 lattice speed of sound squared (1/3)

fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    var dx = p.x - state.cx;
    var dy = p.y - state.cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    
    // Algebraic distance approximation for ellipse
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b; 
}

fn get_chi(phi: f32) -> f32 {
    let epsilon = 1.5f;
    // Clamp tanh arg: large |arg| overflows to NaN on some GPUs (e.g. Intel Gen12LP); saturated regime is unchanged. See PR.
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  // Position/solid-velocity/own-cell-index terms, hoisted ABOVE the gather
  // loop (unchanged math, just moved earlier from where section "3" used
  // to compute them) -- USE_BOUNCEBACK's gather-time sharp test and
  // own-cell reflection lookup both need these before streaming, not after.
  let p = vec2<f32>(f32(x), f32(y));
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);

  let bx = (x + u32(state.off_x)) % W;
  let by = (y + u32(state.off_y)) % H;
  let cell = by * W + bx;

  // 1. Pull Streaming: Read populations from neighbors that will arrive at (x,y)
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    // Window coordinates of source neighbor
    let wx_src = (x + W - u32(ex[i])) % W;
    let wy_src = (y + H - u32(ey[i])) % H;

    if (USE_BOUNCEBACK != 0u && get_phi(vec2<f32>(f32(wx_src), f32(wy_src)), state) < 0f) {
      // Bounce-back: the streaming source is inside the solid, so there's
      // no valid fluid population to pull -- reflect this cell's OWN
      // pre-streaming population that was heading toward that same
      // neighbor instead (opp[i], since direction i's source being solid
      // means the wall lies in direction opp[i] FROM this cell). Ladd's
      // moving-wall correction term is included for parity with the
      // diffuse method's rigid-body coupling (usx,usy); it's exactly 0 for
      // a pinned body (the cylinder validation scenario), so this reduces
      // to textbook simple bounce-back there. rho=1 in the correction
      // (near-incompressible approximation, standard practice for this
      // term specifically) avoids a circular dependency on this cell's own
      // not-yet-gathered rho.
      let corr = 2f * wt[i] * (f32(ex[i]) * usx + f32(ey[i]) * usy) / CS2;
      f[i] = f_in[opp[i] * (W * H) + cell] + corr;
    } else {
      // Map window source to buffer source
      let bx_src = (wx_src + u32(state.off_x)) % W;
      let by_src = (wy_src + u32(state.off_y)) % H;
      f[i] = f_in[i * (W * H) + (by_src * W + bx_src)];
    }
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  let rhoDen = max(rho, 1e-6f); // NaN-containment floor (parity with amr_step)
  ux_star /= rhoDen; uy_star /= rhoDen;

  // 3. Penalty Force and Solid Coupling -- chi forced to 0 under
  // USE_BOUNCEBACK (the sharp bounce-back streaming above is the ENTIRE
  // solid boundary condition in that mode; Fx/Fy/Si below collapse to 0
  // and this becomes plain BGK relaxation everywhere, exactly as intended).
  let chi = select(get_chi(phi), 0f, USE_BOUNCEBACK != 0u);

  // Penalty Force F = rho * chi * (Us - u*)
  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  // Actual fluid velocity u = u* + F/(2rho)
  let ux = ux_star + Fx / (2.0f * rhoDen);
  let uy = uy_star + Fy / (2.0f * rhoDen);
  let u_sq = ux*ux + uy*uy;

  // Store velocity for rendering (buffer cell index)
  vel[cell * 2u] = ux; vel[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC Sponge
  let SPONGE_W = 4.0f;
  let dist_x = min(f32(x), f32(W - 1u - x));
  let dist_y = min(f32(y), f32(H - 1u - y));
  var sponge_weight = clamp(1.0f - min(dist_x, dist_y) / SPONGE_W, 0.0f, 1.0f);
  sponge_weight = sponge_weight * sponge_weight * (3.0f - 2.0f * sponge_weight);

  let omg = 1.0f / state.tau;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);
    
    // Guo's Source Term Si = (1 - 1/(2tau)) * wi * [ (ei-u)/cs2 + (ei.u)/cs4 * ei ] . F
    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );
    
    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let eu_far = exf*SPONGE_UX + eyf*SPONGE_UY;
    let f_target = wt[i] * (1.0f + 3.0f*eu_far + 4.5f*eu_far*eu_far - 1.5f*(SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY)); // rho=1.0, u=(SPONGE_UX,SPONGE_UY) equilibrium

    f_out[i * (W * H) + cell] = mix(f_collide, f_target, sponge_weight);
  }
}
