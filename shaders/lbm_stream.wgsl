// D2Q9 streaming pass — pull scheme
// Periodic BC in x; bounce-back (no-slip) walls at y=0 and y=H-1
// Reads f_col, writes f_out

@group(0) @binding(0) var<storage, read>       f_col : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out : array<f32>;

const W = 256u;
const H = 512u;

const ex  = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey  = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const opp = array<u32,9>( 0, 3, 4, 1, 2, 7, 8, 5, 6);  // opposite direction index

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let dst_base = (y * W + x) * 9u;

  for (var i = 0u; i < 9u; i++) {
    // source cell for pull streaming
    let sx_i = i32(x) - ex[i];
    let sy_i = i32(y) - ey[i];

    var src_base: u32;
    var src_dir:  u32;

    if (sy_i < 0 || sy_i >= i32(H)) {
      // bounce-back: reflect off top/bottom wall — read opposite from same cell
      src_base = (y * W + x) * 9u;
      src_dir  = opp[i];
    } else {
      // periodic in x
      let sx = u32((sx_i + i32(W)) % i32(W));
      src_base = (u32(sy_i) * W + sx) * 9u;
      src_dir  = i;
    }

    f_out[dst_base + i] = f_col[src_base + src_dir];
  }
}
