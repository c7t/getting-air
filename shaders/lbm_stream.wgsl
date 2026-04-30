// D2Q9 streaming — pull scheme, periodic in x and y.
// Bounce-back at solid nodes: if the source cell is solid, reflect.

@group(0) @binding(0) var<storage, read>       f_col  : array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out  : array<f32>;
@group(0) @binding(2) var<storage, read>       solid  : array<u32>;

const W = 256u;
const H = 512u;

const ex  = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey  = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const opp = array<u32,9>( 0, 3, 4, 1, 2, 7, 8, 5, 6);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= W || y >= H) { return; }

  let cell     = y * W + x;
  let dst_base = cell * 9u;

  // solid nodes just reflect their own distributions (won't affect fluid)
  if (solid[cell] != 0u) {
    for (var i = 0u; i < 9u; i++) {
      f_out[dst_base + i] = f_col[dst_base + opp[i]];
    }
    return;
  }

  for (var i = 0u; i < 9u; i++) {
    // pull: where did direction i come from?
    let sx = u32((i32(x) - ex[i] + i32(W)) % i32(W));
    let sy = u32((i32(y) - ey[i] + i32(H)) % i32(H));

    let src_cell = sy * W + sx;
    if (solid[src_cell] != 0u) {
      // source is solid: bounce-back — reflect from current cell, opposite dir
      f_out[dst_base + i] = f_col[dst_base + opp[i]];
    } else {
      f_out[dst_base + i] = f_col[src_cell * 9u + i];
    }
  }
}
