// Per-frame field digest, for the "the view jumps backward a few frames"
// investigation.
//
// The existing readback watchdog only sees the CARD STATE, and that has come
// back monotonic on every run including ones where the symptom was reported
// live -- so whatever rewinds is the FLUID FIELD, which nothing was
// measuring. This reduces the L0 velocity field to a few scalars so a frame
// can be compared against the frames before it.
//
// The point is not the magnitude of any one number: it is whether a frame's
// digest ever EXACTLY REPEATS a digest from several frames earlier. Ordinary
// dynamics never reproduces a previous state bit-for-bit; a display or
// buffer-selection fault that shows an older field does. That distinction is
// what makes this decisive where "does it look wrong" is not.
//
// One dispatch per rendered FRAME, not per macro-step (64 of those), so at
// ~9 passes per macro-step this is under 0.2% of the frame's pass count.
//
// Strided rather than exhaustive: a full reduction would need a second pass
// to combine workgroups. One workgroup walking the field with a large stride
// is enough to fingerprint it, and keeps this to a single dispatch with no
// inter-workgroup dependency.

@group(0) @binding(0) var<storage, read>       vel    : array<f32>;
@group(0) @binding(1) var<storage, read_write> digest : array<f32, 4>;

override NCELLS : u32;

var<workgroup> wg_a : array<f32, 64>;
var<workgroup> wg_b : array<f32, 64>;
var<workgroup> wg_m : array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
  var s = 0f;      // sum of ux (sign-sensitive: catches a shifted field)
  var q = 0f;      // sum of squares (energy-like)
  var m = 0f;      // max |u|
  // Stride so consecutive threads read consecutive cells (coalesced) while
  // the workgroup as a whole samples the whole domain.
  var i = lid;
  loop {
    if (i >= NCELLS) { break; }
    let ux = vel[i * 2u];
    let uy = vel[i * 2u + 1u];
    s += ux + 2f * uy;              // asymmetric weight so ux/uy cannot cancel
    q += ux * ux + uy * uy;
    m = max(m, max(abs(ux), abs(uy)));
    i += 64u * 37u;                 // 37: coprime with the block sizes, so the
                                    // walk does not alias onto one block column
  }
  wg_a[lid] = s; wg_b[lid] = q; wg_m[lid] = m;
  workgroupBarrier();
  for (var d: u32 = 32u; d > 0u; d = d >> 1u) {
    if (lid < d) {
      wg_a[lid] = wg_a[lid] + wg_a[lid + d];
      wg_b[lid] = wg_b[lid] + wg_b[lid + d];
      wg_m[lid] = max(wg_m[lid], wg_m[lid + d]);
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    digest[0] = wg_a[0];
    digest[1] = wg_b[0];
    digest[2] = wg_m[0];
    digest[3] = f32(NCELLS);
  }
}
