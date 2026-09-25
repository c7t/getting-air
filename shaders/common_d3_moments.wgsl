// Moments of a pool level's populations into its own `mac`. Fragment only;
// the entry files list every include.
//
// WHY THIS EXISTS AT ALL, given that every step kernel already writes `mac`
// as a side effect of collision. A pool's `mac` is a DERIVED buffer: the
// step writes it for the cells that level solves and the child's coalesce
// writes it for the cells the child covers, so it is correct from the first
// substep onward and is simply UNWRITTEN before then. reset() seeds `f`
// (interp from the parent, cascading down the levels) and stops there.
//
// That was invisible while nothing read a pool's `mac` off the GPU. It stops
// being invisible the moment something does:
//
//   - readPoolStats reported rms 0 at step 0 on every level, on cases whose
//     L0 energy is plainly nonzero (plans/3D.md M5.6), and
//     validate-d3-invariants.js had to special-case step 0 to avoid calling
//     a correctly-seeded level "not advancing".
//   - the slice view samples the finest level that covers a point (M6.0),
//     so at step 0 a refined region would render as a HOLE -- black where
//     the coarse field around it is not -- which looks exactly like a
//     broken sampler.
//
// Neither is a reason to loosen a check or to special-case a viewer. The
// buffer should simply hold the moments of the populations that are there,
// and this is the eleven lines that make that true.
//
// EVERY CELL OF A LIVE TILE, ring included. The ring is filled rather than
// solved, so its `mac` is never consumed -- the sampler always reaches a
// cell through the block that OWNS it, which lands in the interior by
// construction. Writing it anyway costs one predicate and means there is no
// cell of a live tile whose `mac` is undefined, which is a simpler thing to
// state than the truth about which subset matters.

@group(0) @binding(0) var<storage, read>       f_pool      : array<f32>;
@group(0) @binding(1) var<storage, read_write> mac_pool    : array<f32>;
@group(0) @binding(2) var<storage, read>       slotToBlock : array<i32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let FB = poolFB();
  let fz = gid.z % FB;
  let slot = gid.z / FB;
  if (gid.x >= FB || gid.y >= FB) { return; }
  if (slot >= arrayLength(&slotToBlock)) { return; }
  if (slotToBlock[slot] < 0) { return; }

  let cell = poolCell(slot, vec3<u32>(gid.x, gid.y, fz));
  let poolPlane = arrayLength(&f_pool) / QN;
  var rho = 0f;
  var m = vec3<f32>(0f);
  for (var i = 0u; i < QN; i++) {
    let f = f_pool[i * poolPlane + cell];
    rho += f;
    m += f * vec3<f32>(f32(ex[i]), f32(ey[i]), f32(ez[i]));
  }
  let u = m / max(rho, 1e-6f);   // the same NaN-containment floor the
                                 // transfers use, for the same reason
  let macPlane = arrayLength(&mac_pool) / 4u;
  mac_pool[0u * macPlane + cell] = rho;
  mac_pool[1u * macPlane + cell] = u.x;
  mac_pool[2u * macPlane + cell] = u.y;
  mac_pool[3u * macPlane + cell] = u.z;
}
