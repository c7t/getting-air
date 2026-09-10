// Fused 3D LBM step: pull-streaming + BGK collision + Guo forcing, plus the
// macroscopic-field output the renderer and the validation readbacks use.
// plans/3D.md M1.
//
// Fragment only, and fragments cannot themselves @include (shader-loader.mjs
// splices one level, deliberately -- no cycle detection needed). So the
// ENTRY files list every fragment this body needs: the lattice,
// common_d3_geometry.wgsl and common_d3_sponge.wgsl. Included by
// shaders/d3_step_q19.wgsl and shaders/d3_step_q27.wgsl, each pairing it
// with a different common_d3q{19,27}_lattice.wgsl -- that pairing is the ENTIRE difference
// between the velocity sets, so `?q=19|27` costs one extra entry file and
// no branches (plans/3D.md sec 4). Nothing here names a Q; it speaks only
// QN / ex / ey / ez / wt / opp / feqD3Q.
//
// Direct 3D translation of shaders/lbm_step.wgsl, with the parts M1 does
// not have deliberately absent: no body/SDF/chi coupling, no sponge, no
// moving walls. Those belong to M2 and later, and adding them speculatively
// here would mean carrying untested branches through every kernel that
// inherits this one.
//
// WALLS. WALL_X/Y/Z each turn the two domain faces normal to that axis into
// stationary no-slip walls; every other axis stays periodic. The convention
// is shaders/common_walls.wgsl's, unchanged: the physical wall sits HALFWAY
// between the last fluid cell and the first nonexistent one, so N fluid
// cells span exactly N lattice units wall to wall and the duct half-width
// is exactly N/2. d3-scenarios.mjs's ductCoord() is the host-side statement
// of the same convention, and the two have to agree or the analytic profile
// is compared against a duct half a cell wider than the one being
// simulated -- which presents as a few-percent profile error, not as an
// obvious failure.
//
// Walls are stationary, so Ladd's moving-wall correction term is identically
// zero and is omitted rather than written out and multiplied by zero. A
// Couette or moving-boundary scenario has to add it back (see
// common_walls.wgsl's wallVelocityX for the 2D form).
//
// A direction whose streaming source is outside on TWO axes at once (an
// edge direction at a duct corner cell) reflects the same way as a
// single-axis one -- textbook simple bounce-back at a concave corner, and
// the reason the test is `outside on any axis` rather than per-axis.

@group(0) @binding(0) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out : array<f32>;
// [rho, ux, uy, uz] per cell. Named `mac`, not `macro` -- WGSL reserves
// `macro` as a keyword. Dual-purpose, deliberately: the host uploads
// the initial condition into it and `initEq` turns that into an equilibrium
// `f`, after which every `step` overwrites it with the CURRENT macroscopic
// field for the renderer and for readback. One buffer, one meaning ("the
// macroscopic state right now"), and re-seeding is upload + initEq.
//
// KNOWN COST, accepted for M1: `step` writes this every step, which is 16
// B/cell on top of the 152 (D3Q19) the solver already moves -- about 10%
// more write traffic, and M0 established this kernel is purely
// bandwidth-bound, so it is roughly 10% of frame. The fix is a second
// pipeline variant with the write overridden off, used for every step of a
// batch except the last one the renderer or a readback actually observes.
// Deliberately NOT built yet: plans/perf-characterization.md's standing
// lesson is that optimizations here get measured, not assumed, and M1 has
// no measurement apparatus for the 3D solver yet.
@group(0) @binding(2) var<storage, read_write> mac : array<f32>;
// Rigid body. Always bound (one bind-group layout for every scenario, see
// main-3d.js), and completely inert when HAS_BODY is 0 -- these are
// pipeline-overridable constants, so the whole solid-coupling path folds
// out at pipeline-creation time for the M1 scenarios.
@group(0) @binding(3) var<storage, read>       body : BodyState3D;
// Level-1 block -> pool slot, or -1. Always bound (one bind-group layout for
// every scenario, like `body`) and a single dummy -1 element when there is no
// pool, which HAS_POOL = 0 folds out at pipeline-creation time.
@group(0) @binding(4) var<storage, read>       blockSlot : array<i32>;

override NX : u32;
override NY : u32;
override NZ : u32;

// --- coarse/fine partitioning (plans/3D.md M4.1a) --------------------------
//
// THE COARSE GRID DOES NOT SOLVE UNDER A REFINED REGION. Chen et al. (2006)
// -- the explode/coalesce scheme M4.1 converts this interface to -- treats
// coarse and fine as a PARTITION: coarse voxels stop where fine ones start,
// except for the one-voxel dual interface layer. This solver started as an
// OVERLAP instead, with L0 solving everywhere and `average` overwriting its
// result under refinement, and that redundancy is exactly what makes the
// seam's mass balance ambiguous -- the same population is accounted on both
// levels.
//
// Skipping a covered cell is numerically a NO-OP today, which is why this
// lands as its own stage: `average` overwrites both `f_out` and `mac` at
// every covered cell after this kernel runs, so the work being skipped was
// already being discarded. What the stage actually buys is the per-cell mask
// itself, in the kernel sec 2.4 measured at 77% of device peak, introduced
// against a bit-identical gate rather than alongside a physics change. M4.1b
// makes it load-bearing, when `average` becomes a coalesce that writes only
// the interface layer and there is no longer anything overwriting the deep
// interior.
//
// The test is per-BLOCK, not per-cell: refinement is always whole-block, so
// one blockSlot load answers it for every cell in the block. Same shape as
// the 2D `amr_force.wgsl:84` finest-wins mask.
override HAS_POOL : u32 = 0u;
override RB : u32 = 4u;
override NBX : u32 = 1u;
override NBY : u32 = 1u;
override NBZ : u32 = 1u;

fn coveredByFiner(x: u32, y: u32, z: u32) -> bool {
  if (HAS_POOL == 0u) { return false; }
  let b = ((z / RB) * NBY + (y / RB)) * NBX + (x / RB);
  return blockSlot[b] >= 0;
}

override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

// BGK relaxation rate, 1/tau.
override OMEGA : f32 = 1.25f;

// Uniform body-force density (Guo), e.g. the duct's driving force. All
// zero is a no-op and folds out at pipeline-creation time, since these are
// pipeline-overridable constants rather than uniforms.
override FORCE_X : f32 = 0.0f;
override FORCE_Y : f32 = 0.0f;
override FORCE_Z : f32 = 0.0f;

// Stationary no-slip walls on the two faces normal to each axis. 0 =
// periodic (the default on every axis).
override WALL_X : u32 = 0u;
override WALL_Y : u32 = 0u;
override WALL_Z : u32 = 0u;

// --- solid body (M2) -------------------------------------------------------
// 0 = no interior body at all; skip every get_phi3/chi evaluation. Default,
// and exactly what the M1 scenarios (duct, beltrami, tgv) run.
override HAS_BODY : u32 = 0u;

// Sharp momentum-exchange bounce-back instead of the diffuse
// (Brinkman/Guo) volume penalization. Same both-methods-live arrangement as
// the 2D lbm_step.wgsl: direct comparison of the two needs both available,
// and it changes the numerical method rather than a tunable within one.
// MUST match d3_force.wgsl's own setting -- main-3d.js always creates the
// pair together.
override USE_BOUNCEBACK : u32 = 0u;

// Width of the diffuse chi band, in lattice cells. 1.5 matches the 2D
// dense kernel exactly.
override CHI_EPS : f32 = 1.5f;

// ALBC sponge: relax toward a uniform freestream within SPONGE_W cells of
// every domain face. <= 0 disables it (the periodic and walled scenarios).
override SPONGE_W : f32 = 0.0f;
override SPONGE_UX : f32 = 0.0f;
override SPONGE_UY : f32 = 0.0f;
override SPONGE_UZ : f32 = 0.0f;

fn cellIndex(x: u32, y: u32, z: u32) -> u32 {
  return (z * NY + y) * NX + x;
}

// True if direction i's streaming source lies outside a walled face.
// Signed arithmetic throughout: the 2D kernels' `(x + W - u32(ex[i])) % W`
// idiom leans on u32 wraparound making `- u32(-1)` mean `+ 1`, which is
// correct but is not something anyone should have to re-derive while
// reading a three-axis version of it.
fn wallSourceOutside3(x: u32, y: u32, z: u32, i: u32) -> bool {
  let sx = i32(x) - ex[i];
  let sy = i32(y) - ey[i];
  let sz = i32(z) - ez[i];
  return (WALL_X != 0u && (sx < 0 || sx >= i32(NX)))
      || (WALL_Y != 0u && (sy < 0 || sy >= i32(NY)))
      || (WALL_Z != 0u && (sz < 0 || sz >= i32(NZ)));
}

// Turns the host-uploaded macroscopic field in `mac` into an equilibrium
// `f`. Seeding from (rho, u) rather than uploading `f` directly keeps the
// initial condition defined in ONE place -- d3-scenarios.mjs, which is also
// what the validation tools score against -- and keeps the upload to 4
// floats per cell instead of QN (at 128^3 D3Q19 that is 34 MB instead of
// 152 MB, which is the difference between a fraction of a second and a
// visible stall).
@compute @workgroup_size(WGX, WGY, WGZ)
fn initEq(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x >= NX || y >= NY || z >= NZ) { return; }
  let ncells = NX * NY * NZ;
  let cell = cellIndex(x, y, z);

  let rho = mac[4u * cell + 0u];
  let ux  = mac[4u * cell + 1u];
  let uy  = mac[4u * cell + 2u];
  let uz  = mac[4u * cell + 3u];

  for (var i = 0u; i < QN; i++) {
    f_out[i * ncells + cell] = feqD3Q(rho, ux, uy, uz, i);
  }
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x >= NX || y >= NY || z >= NZ) { return; }
  if (coveredByFiner(x, y, z)) { return; }
  let ncells = NX * NY * NZ;
  let cell = cellIndex(x, y, z);

  // 1. Pull-stream, with bounce-back where the source is outside a wall.
  let p = vec3<f32>(f32(x), f32(y), f32(z));
  let phi = select(1e30f, get_phi3(p, body), HAS_BODY != 0u);
  let us = select(vec3<f32>(0f), bodyVelocity3(p, body), HAS_BODY != 0u);

  var f : array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    let sxi = i32(x) - ex[i];
    let syi = i32(y) - ey[i];
    let szi = i32(z) - ez[i];
    if (HAS_BODY != 0u && USE_BOUNCEBACK != 0u
        && get_phi3(vec3<f32>(f32(sxi), f32(syi), f32(szi)), body) < 0f) {
      // The streaming source is inside the solid: reflect this cell's own
      // population that was heading toward it (opp[i], since direction i's
      // source being solid puts the wall in direction opp[i] from here),
      // with Ladd's moving-wall correction for the body's local velocity.
      // rho = 1 in the correction -- the standard near-incompressible
      // approximation for this term specifically, which avoids a circular
      // dependency on this cell's own not-yet-gathered rho.
      let corr = 2f * wt[i] * dot(vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i])), us) / CS2;
      f[i] = f_in[opp[i] * ncells + cell] + corr;
    } else if ((WALL_X | WALL_Y | WALL_Z) != 0u && wallSourceOutside3(x, y, z, i)) {
      // The source is inside the wall, so there is no fluid population to
      // pull: reflect this cell's OWN pre-streaming population that was
      // heading the other way. Direction i's source being solid means the
      // wall lies in direction opp[i] from here.
      f[i] = f_in[opp[i] * ncells + cell];
    } else {
      let sx = u32((sxi + i32(NX)) % i32(NX));
      let sy = u32((syi + i32(NY)) % i32(NY));
      let sz = u32((szi + i32(NZ)) % i32(NZ));
      f[i] = f_in[i * ncells + cellIndex(sx, sy, sz)];
    }
  }

  // 2. Moments.
  var rho = 0f; var mx = 0f; var my = 0f; var mz = 0f;
  for (var i = 0u; i < QN; i++) {
    rho += f[i];
    mx  += f[i] * f32(ex[i]);
    my  += f[i] * f32(ey[i]);
    mz  += f[i] * f32(ez[i]);
  }
  let rhoDen = max(rho, 1e-6f);  // NaN-containment floor, parity with lbm_step
  let ux_star = mx / rhoDen; let uy_star = my / rhoDen; let uz_star = mz / rhoDen;

  // 3. Solid coupling and forcing.
  //
  // chi is forced to 0 under USE_BOUNCEBACK -- the sharp reflection in the
  // gather above IS the entire boundary condition in that mode, and the
  // penalty force below collapses to the uniform body force alone. Same
  // arrangement as the 2D lbm_step.wgsl.
  let chi = select(0f, chiFromPhiEps3(phi, CHI_EPS),
                   HAS_BODY != 0u && USE_BOUNCEBACK == 0u);
  let ustar = vec3<f32>(ux_star, uy_star, uz_star);
  // Penalty force F = rho chi (Us - u*), plus the uniform body force.
  let F = rho * chi * (us - ustar) + vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);

  // Guo forcing: the actual fluid velocity is u = u* + F/(2 rho).
  let u = ustar + F / (2.0f * rhoDen);
  let ux = u.x; let uy = u.y; let uz = u.z;
  let u_sq = dot(u, u);

  mac[4u * cell + 0u] = rho;
  mac[4u * cell + 1u] = ux;
  mac[4u * cell + 2u] = uy;
  mac[4u * cell + 3u] = uz;

  let spongeW = spongeWeight3(
    min(f32(x), f32(NX - 1u - x)),
    min(f32(y), f32(NY - 1u - y)),
    min(f32(z), f32(NZ - 1u - z)), SPONGE_W);

  // 4. Collide. Gathered whole and stored whole, matching lbm_step.wgsl --
  // see common_fpack.wgsl for why a per-plane store is not an option once
  // packed storage is in play, even though M1 does not use it.
  var fo : array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]); let ezf = f32(ez[i]);
    let eu  = exf*ux + eyf*uy + ezf*uz;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    // Guo's source term:
    //   Si = (1 - 1/(2 tau)) wi [ (ei - u)/cs2 + (ei.u)/cs4 ei ] . F
    // with cs2 = 1/3 so 1/cs2 = 3 and 1/cs4 = 9.
    let ei = vec3<f32>(exf, eyf, ezf);
    let t1 = (ei - u) * 3.0f;
    let t2 = eu * 9.0f;
    let Si = (1.0f - 0.5f * OMEGA) * wt[i] * dot(t1 + t2 * ei, F);

    let fCollide = f[i] - OMEGA * (f[i] - feq) + Si;

    // ALBC sponge: relax toward the uniform freestream equilibrium at
    // rho = 1 near the domain faces. Folds out entirely at SPONGE_W <= 0.
    let euFar = exf*SPONGE_UX + eyf*SPONGE_UY + ezf*SPONGE_UZ;
    let uFarSq = SPONGE_UX*SPONGE_UX + SPONGE_UY*SPONGE_UY + SPONGE_UZ*SPONGE_UZ;
    let fTarget = wt[i] * (1.0f + 3.0f*euFar + 4.5f*euFar*euFar - 1.5f*uFarSq);
    fo[i] = mix(fCollide, fTarget, spongeW);
  }

  for (var i = 0u; i < QN; i++) {
    f_out[i * ncells + cell] = fo[i];
  }
}
