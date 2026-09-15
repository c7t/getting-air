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
// An OVERRIDE, not a const, so it can be A/B'd at runtime: set it very large
// and the early-out never fires, restoring the always-Newton behaviour. That
// matters because the branch is a clear win on desktop but unmeasurable from
// here on the mobile part this project targets, where a divergent branch has
// a different cost profile.
override SDF_FAR : f32 = 64.0f;

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

// --- "does any point of this BOX come within `margin` of the body?" ---------
//
// THE KERNELS USED TO ASK ABOUT A BLOCK'S CENTRE, and that is a different
// question. The gap is the block's own circumradius -- 5.66 L0 cells for an
// RB=8 level-1 block, against a default L1->L2 margin of 4 -- so a tile whose
// CENTRE just missed the margin could still have an edge, or at shallow
// surface incidence a corner, touching the body.
//
// main-cylinder-amr.js:224-236 records that live-verified, with the symptom
// "L1's own force pass sat at a bit-identical fx~-0.19 for 20,000+ steps" --
// a permanently-stuck coarse patch next to the body, not unsteady flow. It
// was fixed then by ENLARGING THE MARGIN (paramsForChildLevel's
// childLevel===2 special case), which papers over the gap at one level and
// leaves it everywhere else. This asks the real question instead, so the
// margin means what it says at every level.
//
// THE METHOD: a signed distance is 1-Lipschitz, so ONE evaluation at a box
// centre brackets the minimum over the whole box as
//
//     phi(c) - h*sqrt(2)  <=  min over box  <=  phi(c)
//
// which decides most boxes outright and leaves only an ambiguous shell to
// subdivide. It is CONSERVATIVE at every depth -- it can over-refine, never
// miss -- which is the right direction for a HARD geometric constraint.
//
// This is amr2d.mjs's `nearBodyWant`, and the two must agree: that module is
// the specification (GPU-free, mutation-checked in tools/test-amr2d.js,
// scored against a brute-force closest point) and this is what actually runs.
// The two constants below are typed in both places and tools/test-amr2d.js
// parses them back OUT of this file to compare.

// sqrt(2), ROUNDED UP -- rounding it down would make the bound false by a
// hair, on exactly the borderline blocks this exists to catch.
const SQRT2_UP : f32 = 1.4142136;

// Depth of the branch and bound. 3 leaves a residual slack of RB*sqrt(2)/16 =
// 0.71 L0 cells at RB=8, and costs at most 1+4+16+64 = 85 get_phi calls for a
// box the bound cannot decide -- which is a thin shell around the surface,
// once per block, once every REFINE_EVERY macro-steps. Everything else is
// rejected on the first evaluation.
const BLOCK_BB_DEPTH : u32 = 3u;

// The distance the refinement test uses: the smaller of the body's distance
// NOW and FORCE_REFINE_LOOKAHEAD macro-steps from now.
//
// The future pose does NOT extrapolate the body's centre forward.
// amr_physics.wgsl's moving window keeps cx/cy pinned near (W/2, H/2) by
// construction -- bulk translation is absorbed into off_x/off_y -- so
// `cx += vx*lookahead` would displace a phantom ellipse that does not
// correspond to where the card, or this buffer block relative to it, actually
// will be. What DOES move relative to the window-anchored card is a fixed
// buffer cell's window position: wx(t) = wx(now) - vx*t. So the TEST POINT
// runs backward and the ellipse stays put. theta is the one quantity the
// window does not absorb, so it still extrapolates forward normally.
fn phiMinPose(p: vec2<f32>, lookahead: f32, state: CardState) -> f32 {
    let phi_now = get_phi(p, state);
    var future = state;
    future.theta = state.theta + state.omega * lookahead;
    let phi_future = get_phi(p - vec2<f32>(state.vx, state.vy) * lookahead, future);
    return min(phi_now, phi_future);
}

// WGSL has no recursion, so the depth-first walk carries its own stack. It
// holds at most 3*BLOCK_BB_DEPTH + 1 = 10 entries: each pop reuses the popped
// slot and pushes 4, and the deepest level pushes nothing.
fn nearBodyBox(center: vec2<f32>, half: f32, margin: f32, lookahead: f32, state: CardState) -> bool {
    var stackC : array<vec2<f32>, 16>;
    var stackH : array<f32, 16>;
    var stackD : array<u32, 16>;
    stackC[0] = center;
    stackH[0] = half;
    stackD[0] = BLOCK_BB_DEPTH;
    var sp : u32 = 1u;
    loop {
        if (sp == 0u) { break; }
        sp -= 1u;
        let c = stackC[sp];
        let h = stackH[sp];
        let d = stackD[sp];
        let phi = phiMinPose(c, lookahead, state);
        if (phi - h * SQRT2_UP > margin) { continue; }   // provably clear
        if (phi <= margin) { return true; }              // the centre is in
        // The deepest level is the only place the answer is a BOUND rather
        // than a decision: having failed to reject, accept. That is where the
        // residual slack lives, and it is on the safe side.
        if (d == 0u) { return true; }
        let hh = h * 0.5f;
        for (var i = 0u; i < 4u; i++) {
            let off = vec2<f32>(select(-hh, hh, (i & 1u) != 0u), select(-hh, hh, (i & 2u) != 0u));
            stackC[sp] = c + off;
            stackH[sp] = hh;
            stackD[sp] = d - 1u;
            sp += 1u;
        }
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// BUFFER -> WINDOW, THE ONLY TWO PLACES IT HAPPENS (plans/2D-backport.md B5-1)
//
// The solver is buffer-coordinate native: a thread owns a buffer cell and
// everything it indexes is a buffer index. Since B5 the BODY is buffer-native
// too, so only things anchored to the VIEW convert -- the ALBC sponge band,
// the WALL_Y walls, and the render. That used to include the body SDF and the
// refinement geometry predicates, and each of NINE call sites spelled the
// conversion out for itself, in one of two idioms that are NOT
// interchangeable:
//
//   cells      (c + N - u32(off)) % N       integer, truncates off
//   positions  wrapf(p - off, N)            continuous, keeps off's fraction
//
// Keeping both named matters even now that the body no longer uses either:
// the sponge is sampled per cell and the fine levels evaluate at fractional
// L0 positions, so a site that picks the wrong one by copy-paste gets a
// sub-cell error that nothing rounds away. B5-1 counted them; B5-2 and B5-5
// took the body off them entirely.
//
// DEPENDS ON THE INCLUDER declaring `override W` / `override H`, like get_phi
// above.

fn wrapf(v: f32, n: f32) -> f32 {
  var r = v % n;
  if (r < 0.0) { r += n; }
  return r;
}

// Integer buffer CELL -> integer window cell.
fn bufferToWindowCell(c: vec2<u32>, state: CardState) -> vec2<u32> {
  return vec2<u32>((c.x + W - u32(state.off_x)) % W,
                   (c.y + H - u32(state.off_y)) % H);
}

// Continuous buffer POSITION -> continuous window position, keeping the
// sub-cell part of both the position and the offset.
fn bufferToWindowPos(p: vec2<f32>, state: CardState) -> vec2<f32> {
  return vec2<f32>(wrapf(p.x - state.off_x, f32(W)),
                   wrapf(p.y - state.off_y, f32(H)));
}

// ─────────────────────────────────────────────────────────────────────────────
// WHICH FRAME IS THE BODY IN? (plans/2D-backport.md B5)
//
// THE BODY LIVES IN BUFFER COORDINATES. `cx`/`cy` are buffer positions,
// integrated and wrapped into [0, W) x [0, H) every step, and the SDF is
// evaluated directly on a cell's own buffer position -- get_phi takes the
// nearest periodic image, so there is no conversion and no wrap to do. A
// kernel that has a buffer cell already has everything the body needs.
//
// So the buffer->window conversion below survives for exactly three things,
// and all three are genuinely anchored to the VIEW rather than to the fluid:
// the ALBC sponge band, the WALL_Y channel walls, and the render.
//
// THIS REPLACED A WINDOW-ANCHORED BODY (?window=1, deleted 2026-09-15 after
// B5-5). Under that convention the body was pinned to a fixed WINDOW position
// and the buffer panned beneath it, so every kernel touching the SDF
// converted first -- nine hand-written sites across six files, in two
// different flavours, which is what B5-1 collapsed and B5-2/B5-5 finished.
// Two things are worth keeping from it:
//
//   - The conversions were not interchangeable. The CELL form truncated the
//     offset, the POSITION form kept its fraction, and the fine levels
//     evaluate at fractional L0 positions. Anything reintroducing a
//     conversion has to pick deliberately; see bufferToWindowCell vs.
//     bufferToWindowPos above.
//   - The two conventions did NOT merely re-label cells, they integrated the
//     position differently, so a free body's trajectory diverged from step
//     one. Measured at the swap: a pinned body agreed exactly (every Cd/St
//     config to the digit), while index-reentry.html's prescribed moving body
//     differed by relL2 2.1e-3 in ux. That is rounding, not a frame error --
//     a frame error is O(1), as B5-5's 226x force bug was.

// A kernel that needs the body's frame writes its own buffer cell inline, as
// `vec2<f32>(f32(bx), f32(by))`. There is no accessor: an identity function
// named after a frame is exactly the thing that let lbm_force.wgsl look
// converted while it was not.

// The inverse of bufferToWindowPos, for the one caller that legitimately
// starts in WINDOW space: the render, which rasterizes the VIEW and therefore
// has a window position in hand, but must ask about the body, which is in
// buffer coordinates.
fn windowToBufferPos(pWin: vec2<f32>, state: CardState) -> vec2<f32> {
  return pWin + vec2<f32>(state.off_x, state.off_y);
}
