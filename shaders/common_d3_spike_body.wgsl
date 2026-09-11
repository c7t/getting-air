// M0 register/spill spike kernel body (plans/3D.md sec 2.4, sec 5 M0).
//
// Fragment only. Included by shaders/d3_spike_q19.wgsl and
// shaders/d3_spike_q27.wgsl, each of which pairs it with a different
// common_d3q{19,27}_lattice.wgsl. Those two files are the entire difference
// between the velocity sets: this body never names a Q, only QN/ex/ey/ez/
// wt/feqD3Q. That is the `?q=19|27` mechanism plans/3D.md sec 4 asks for,
// done at the @include level because WGSL has no preprocessor and an array
// size must be a const-expression (an `override Q` cannot size
// `array<f32,Q>`, so one module cannot serve both sets).
//
// WHAT THIS IS FOR. Ranked risk #1 for the 3D fork is that a `private`
// array indexed by a loop variable is only register-resident if the
// compiler fully unrolls the loop. At Q=9 it does. At Q=27, through
// Tint -> SPIR-V/DXIL -> driver, it might not -- and a 27-element array
// spilled to scratch would be catastrophic AND SILENT: it would present as
// "3D is just slow", with no error and no wrong answer, after the entire
// solver had been written on top of it. This kernel exists to answer that
// question in ~200 lines instead of ~8000.
//
// The measurement is achieved bandwidth against the per-cell traffic
// roofline. Three MODEs, all with identical memory traffic (QN planes in,
// QN planes out) and therefore directly comparable in GB/s:
//
//   MODE=0  full     pull-stream + BGK collide. The real thing. Holds `f`
//                    and `fo` live simultaneously -- 38 f32 at Q19, 54 at
//                    Q27 -- which is exactly the register pressure in
//                    question.
//   MODE=1  stream   gather and write straight back out. The ACCESS-PATTERN
//                    CEILING: same addresses, same bytes, no arithmetic.
//                    Note it is not a register control -- a compiler is
//                    free to fuse each gather into its own store and never
//                    hold QN values live, which is precisely why it is the
//                    ceiling and not the baseline.
//   MODE=2  collide  read own cell, collide, write. Same traffic, same
//                    register pressure as MODE=0, but perfectly coalesced
//                    addresses. Separates "the gather's scattered reads are
//                    expensive" from "the collision's registers are
//                    expensive", which is the distinction that decides
//                    whether the design has to be restructured.
//
// Reading the result: if all three land near the same GB/s, memory is the
// bound and registers are fine. If MODE=2 at Q27 falls well below MODE=2 at
// Q19 *in GB/s* (not in GLUPS -- GLUPS is expected to fall by 27/19), that
// is the spill signature, because spill traffic is real traffic the roofline
// cannot see.
//
// Deliberately absent: body/geometry, sponge, walls, forcing, AMR, render,
// f16 packing. This is not a solver and must not grow into one; M1 is where
// a real dense 3D page gets written.

@group(0) @binding(0) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out : array<f32>;

override NX : u32;
override NY : u32;
override NZ : u32;

// Workgroup shape, overridable so the spike can sweep it -- occupancy is
// half the question here and 3D gives a genuine choice (4x4x4 = 64 threads
// against 2D's 8x8, same count but a different cache footprint per group).
// The host must dispatch ceil(N/WG) per axis to match.
override WGX : u32 = 4u;
override WGY : u32 = 4u;
override WGZ : u32 = 4u;

// 0 = full (stream + collide), 1 = stream only, 2 = collide only.
override MODE : u32 = 0u;

// BGK relaxation rate 1/tau. 1.0 (tau = 1) is a stable, physically sensible
// default; the spike is a throughput measurement, not a physics run, but
// running it at a value that would blow up would put denormals in the
// arithmetic and quietly change what is being measured.
override OMEGA : f32 = 1.0f;

// Initial-condition amplitude: a Taylor-Green vortex, which is both a
// non-degenerate field (no uniform-value shortcuts available to the
// compiler, no denormals) and the actual M1 validation case, so the
// initializer is not throwaway.
override U0 : f32 = 0.04f;

fn cellIndex(x: u32, y: u32, z: u32) -> u32 {
  return (z * NY + y) * NX + x;
}

// Seeds both `f` buffers with the BGK equilibrium of a Taylor-Green vortex.
// Done on the GPU rather than uploaded: at 128^3 x Q27 the host-side array
// would be 226 MiB per buffer, which is minutes of JS and a pointless
// upload for a field defined by a closed form.
@compute @workgroup_size(WGX, WGY, WGZ)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x >= NX || y >= NY || z >= NZ) { return; }
  let ncells = NX * NY * NZ;
  let cell = cellIndex(x, y, z);

  let TAU2 = 6.2831853f;
  let px = TAU2 * f32(x) / f32(NX);
  let py = TAU2 * f32(y) / f32(NY);
  let pz = TAU2 * f32(z) / f32(NZ);

  let ux =  U0 * sin(px) * cos(py) * cos(pz);
  let uy = -U0 * cos(px) * sin(py) * cos(pz);
  let uz =  0f;
  let rho = 1f;

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

  // 1. Gather. Periodic in all three axes, done in signed arithmetic
  // rather than the 2D kernels' `(x + W - u32(ex[i])) % W` -- that idiom
  // relies on u32 wraparound making `- u32(-1)` equivalent to `+ 1`, which
  // is correct but is not something anyone should have to re-derive while
  // reading a 3-axis version of it.
  var f : array<f32, QN>;
  if (MODE == 2u) {
    for (var i = 0u; i < QN; i++) { f[i] = f_in[i * ncells + cell]; }
  } else {
    for (var i = 0u; i < QN; i++) {
      let sx = u32((i32(x) - ex[i] + i32(NX)) % i32(NX));
      let sy = u32((i32(y) - ey[i] + i32(NY)) % i32(NY));
      let sz = u32((i32(z) - ez[i] + i32(NZ)) % i32(NZ));
      f[i] = f_in[i * ncells + cellIndex(sx, sy, sz)];
    }
  }

  // MODE=1: write the gathered populations straight back. Everything below
  // this point is the arithmetic being priced.
  if (MODE == 1u) {
    for (var i = 0u; i < QN; i++) { f_out[i * ncells + cell] = f[i]; }
    return;
  }

  // 2. Moments.
  var rho = 0f; var mx = 0f; var my = 0f; var mz = 0f;
  for (var i = 0u; i < QN; i++) {
    rho += f[i];
    mx  += f[i] * f32(ex[i]);
    my  += f[i] * f32(ey[i]);
    mz  += f[i] * f32(ez[i]);
  }
  let rhoDen = max(rho, 1e-6f); // NaN-containment floor, parity with lbm_step
  let ux = mx / rhoDen; let uy = my / rhoDen; let uz = mz / rhoDen;

  // 3. BGK collision into a SECOND array before storing. Both arrays live
  // at once is not incidental -- it is the register footprint under test,
  // and it is also what the real kernels need (common_fpack.wgsl forbids a
  // per-plane store because two planes share a word under F16, so the
  // whole cell must be resident before any of it is written).
  var fo : array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    fo[i] = f[i] - OMEGA * (f[i] - feqD3Q(rho, ux, uy, uz, i));
  }

  // 4. Store.
  for (var i = 0u; i < QN; i++) {
    f_out[i * ncells + cell] = fo[i];
  }
}
