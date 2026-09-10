// Fused 3D LBM step: pull-streaming + BGK collision + Guo forcing, plus the
// macroscopic-field output the renderer and the validation readbacks use.
// plans/3D.md M1.
//
// Fragment only. Included by shaders/d3_step_q19.wgsl and
// shaders/d3_step_q27.wgsl, each pairing it with a different
// common_d3q{19,27}_lattice.wgsl -- that pairing is the ENTIRE difference
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

override NX : u32;
override NY : u32;
override NZ : u32;

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
  let ncells = NX * NY * NZ;
  let cell = cellIndex(x, y, z);

  // 1. Pull-stream, with bounce-back where the source is outside a wall.
  var f : array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    if ((WALL_X | WALL_Y | WALL_Z) != 0u && wallSourceOutside3(x, y, z, i)) {
      // The source is inside the wall, so there is no fluid population to
      // pull: reflect this cell's OWN pre-streaming population that was
      // heading the other way. Direction i's source being solid means the
      // wall lies in direction opp[i] from here.
      f[i] = f_in[opp[i] * ncells + cell];
    } else {
      let sx = u32((i32(x) - ex[i] + i32(NX)) % i32(NX));
      let sy = u32((i32(y) - ey[i] + i32(NY)) % i32(NY));
      let sz = u32((i32(z) - ez[i] + i32(NZ)) % i32(NZ));
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

  // 3. Guo forcing: the actual fluid velocity is u = u* + F/(2 rho).
  let ux = ux_star + FORCE_X / (2.0f * rhoDen);
  let uy = uy_star + FORCE_Y / (2.0f * rhoDen);
  let uz = uz_star + FORCE_Z / (2.0f * rhoDen);
  let u_sq = ux*ux + uy*uy + uz*uz;

  mac[4u * cell + 0u] = rho;
  mac[4u * cell + 1u] = ux;
  mac[4u * cell + 2u] = uy;
  mac[4u * cell + 3u] = uz;

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
    let t1x = (exf - ux) * 3.0f;
    let t1y = (eyf - uy) * 3.0f;
    let t1z = (ezf - uz) * 3.0f;
    let t2  = eu * 9.0f;
    let Si = (1.0f - 0.5f * OMEGA) * wt[i]
           * ((t1x + t2*exf)*FORCE_X + (t1y + t2*eyf)*FORCE_Y + (t1z + t2*ezf)*FORCE_Z);

    fo[i] = f[i] - OMEGA * (f[i] - feq) + Si;
  }

  for (var i = 0u; i < QN; i++) {
    f_out[i * ncells + cell] = fo[i];
  }
}
