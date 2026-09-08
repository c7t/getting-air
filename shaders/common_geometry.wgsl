// Shared rigid-body state and ellipse signed-distance geometry -- the
// solid mask every step/force/manage/render kernel tests against.
// Included via `// @include "common_geometry.wgsl"`. Depends on the
// including file's own `override W : u32;`/`override H : u32;` (WGSL
// module-scope name resolution is order-independent, so it doesn't matter
// whether those overrides are declared before or after this include).

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

// TRUE signed distance to the ellipse centered at (state.cx, state.cy),
// rotated by state.theta, semi-axes (state.a, state.b). Positive outside,
// negative inside, zero on the boundary. `p` and the state's center are
// periodically wrapped against the W/H domain before rotating into the
// body frame.
//
// WHY THIS IS A REAL DISTANCE AND NOT THE OLD ALGEBRAIC FORM
//
// This used to return `(r - 1) * b` with `r = hypot(lx/a, ly/b)` -- an
// algebraic distance scaled by the semi-MINOR axis. That is exact on the
// minor axis and wrong everywhere else, by a factor that grows with the
// aspect ratio: on the MAJOR axis it under-reports the true distance by
// exactly a/b. At the falling card's default 8:1 aspect that is 8x, and
// every consumer of phi inherits the error as an anisotropy:
//
//   - get_chi's blend band (epsilon is in phi units) was ~4 cells wide at
//     the flat faces but ~32 cells at the tips, smearing the penalization
//     over a region far larger than the local surface radius.
//   - isNearBody's refinement margin reached 16 cells at the faces but 128
//     at the tips, inflating the refined halo ~3x over what the geometry
//     needs and exhausting the fine-block pool at larger field sizes.
//
// Both are one root cause, so both are fixed here rather than by tuning
// epsilon and the margin around a distorted field.
//
// Method: Newton iteration on the closest-point parameter t of the
// parametric boundary s(t) = (a cos t, b sin t), solving F(t) = (p-s)·s' = 0
// in the first quadrant (the problem is symmetric in both axes, so |lx|,|ly|
// suffices and the sign is restored at the end). Seeded from the algebraic
// direction atan2(y*a, x*b), 3 iterations reach 0.15% worst-case relative
// error against a brute-force reference over the region that matters --
// versus 41% for 1 iteration and 4.3% for 2. Measured, not assumed.
//
// A CIRCLE (a == b) takes an exact closed form and skips the iteration
// entirely. That is not just an optimization: it makes this change
// bit-identical for every circular-body harness in the project
// (main-cylinder*.js and friends), so their validated physics cannot move,
// and it keeps the bounce-back inner loop -- which calls get_phi once per
// lattice direction -- as cheap as it was.
//
// INTERIOR CAVEAT: for points well inside the body, Newton can converge to
// a local rather than the global closest point (e.g. from the centre it
// finds a tip, not the nearer flat face), so |phi| deep inside may be an
// overestimate. Harmless for every current consumer: chi saturates to 1
// long before that error appears (tanh clamps), isNearBody only needs
// phi < margin, and the bounce-back branch only needs the sign. The SIGN is
// always exact -- it comes from the algebraic test, not from the iteration.
// Distance beyond which the cheap algebraic lower bound is returned as-is;
// see the FAR-FIELD EARLY-OUT note in get_phi. Every phi threshold any
// caller uses must stay below this.
const SDF_FAR = 64.0f;

fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    var dx = p.x - state.cx;
    var dy = p.y - state.cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;

    let a = state.a;
    let b = state.b;

    // Circle: |p| - R is already the exact signed distance.
    if (abs(a - b) <= 1e-6f * max(a, b)) {
        return length(vec2<f32>(lx, ly)) - a;
    }

    let x = abs(lx);
    let y = abs(ly);

    // FAR-FIELD EARLY-OUT. The old algebraic form `(r-1)*b` is a provable
    // LOWER BOUND on the true distance outside the body: d = (r-1)/|grad r|
    // and |grad r| ranges over [1/a, 1/b], so dividing by the largest
    // possible gradient (1/b, since b <= a) can only under-estimate. Measured
    // against brute force it is 0% to -88% -- never over.
    //
    // So when the bound alone already exceeds SDF_FAR, the true distance does
    // too, and no caller can tell the difference: every consumer compares phi
    // against a threshold well under SDF_FAR (chi saturates to 0 within a few
    // epsilon, isNearBody tests against FORCE_REFINE_MARGIN), and the sign is
    // unambiguous out there. Returning the bound skips the Newton iteration
    // for the overwhelming majority of cells -- which matters because
    // get_phi is called once per cell in BOTH the step and force kernels,
    // and the force passes measured 40% of the desktop frame.
    //
    // Callers must keep every phi threshold below SDF_FAR. At 64 that is ~4x
    // the default FORCE_REFINE_MARGIN and ~40x the chi band, with room to
    // spare.
    let r = sqrt((x*x)/(a*a) + (y*y)/(b*b));
    let algebraic = (r - 1.0f) * b;
    if (algebraic > SDF_FAR) { return algebraic; }

    // Sign from the algebraic test -- exact, and independent of the
    // iteration below (see INTERIOR CAVEAT above).
    let inside = r < 1.0f;

    var t = atan2(y * a, x * b); // == atan2(y/b, x/a), without the divides
    for (var i = 0u; i < 3u; i++) {
        let ct = cos(t);
        let st = sin(t);
        let ex = x - a * ct;
        let ey = y - b * st;
        // F  = (p-s)·s',  s' = (-a sin t, b cos t)
        let F = ex * (-a * st) + ey * (b * ct);
        // F' = -|s'|^2 + (p-s)·s'',  s'' = (-a cos t, -b sin t)
        let Fp = -(a*a*st*st + b*b*ct*ct) + ex * (-a * ct) + ey * (-b * st);
        // Fp is negative and O(a^2) for any non-degenerate ellipse; the
        // guard only protects the a==b==0 pathology, which cannot reach here.
        t = t - F / select(Fp, -1e-9f, abs(Fp) < 1e-9f);
        t = clamp(t, 0.0f, 1.5707963f);
    }
    let d = length(vec2<f32>(x - a * cos(t), y - b * sin(t)));
    return select(d, -d, inside);
}

// Shared tanh-blend core of every get_chi -- callers compute their own
// epsilon (it varies: fixed 1.5 at L0/L1, K_EPS*levelParams.dxL at
// level>=2's shared pipelines) and pass it in here.
fn chiFromPhiEps(phi: f32, epsilon: f32) -> f32 {
    // Clamp tanh arg: large |arg| overflows to NaN on some GPUs (e.g. Intel Gen12LP); saturated regime is unchanged. See PR.
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}
