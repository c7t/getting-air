// Simple periodic streaming for D2Q9.

@group(0) @binding(1) var<storage, read>       f_col : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out : array<f32>;

const W   = 512u;
const H   = 1024u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell     = y * W + x;
  let dst_base = cell * 9u;

  for (var i = 0u; i < 9u; i++) {
    let sx = u32((i32(x) - ex[i] + i32(W)) % i32(W));
    let sy = u32((i32(y) - ey[i] + i32(H)) % i32(H));
    f_out[dst_base + i] = f_col[(sy * W + sx) * 9u + i];
  }
}
