// THE PARENT'S MACROSCOPIC ARRAY, pool case (level >= 2, parent = a level of
// tiles). The sibling of common_d3_parentmac_dense.wgsl -- read its header
// for why this pair exists at all and what the layout difference cost.
//
// PLANAR, component-major, the same way a pool's f array is: a component's
// plane is `arrayLength/4` long and the cell index strides within it. The
// type is array<f32> rather than array<vec4<f32>> and that difference is
// load-bearing -- it is what makes writing this one in the dense layout a
// compile error rather than a silent transposition.
@group(0) @binding(4) var<storage, read_write> mac : array<f32>;

fn parentMacStore(cell: u32, v: vec4<f32>) {
  let plane = arrayLength(&mac) / 4u;
  mac[0u * plane + cell] = v.x;
  mac[1u * plane + cell] = v.y;
  mac[2u * plane + cell] = v.z;
  mac[3u * plane + cell] = v.w;
}
