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

  let bx   = (x + u32(state.off_x)) % W;
  let by   = (y + u32(state.off_y)) % H;
  let cell = by * W + bx;
  let dst_base = cell * 9u;

  for (var i = 0u; i < 9u; i++) {
    // Window coordinates of source neighbor
    let wx_src = (x + W - u32(ex[i])) % W;
    let wy_src = (y + H - u32(ey[i])) % H;
    
    // Map window source to buffer source
    let bx_src = (wx_src + u32(state.off_x)) % W;
    let by_src = (wy_src + u32(state.off_y)) % H;
    
    f_out[dst_base + i] = f_col[(by_src * W + bx_src) * 9u + i];
  }
}
