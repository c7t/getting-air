// THE PARENT'S MACROSCOPIC ARRAY, dense case (level 1, parent = the L0 grid).
// plans/3D.md M5.2b-iii. Fragment only; the entry files list every include.
//
// WHY THE BINDING LIVES HERE AND NOT IN common_d3_amr_coalesce.wgsl. The two
// macroscopic arrays in this solver have DIFFERENT MEMORY LAYOUTS:
//
//   dense L0 `mac`   INTERLEAVED, [rho,ux,uy,uz] per cell, because the
//                    renderer and every host readback walk it that way.
//   pool `mac_pool`  PLANAR, component-major, like the pool's own f array.
//
// Coalesce writes one of them and reads the other, and at depth 3 the thing
// it writes IS a pool. Writing it in the dense layout put every moment in a
// different cell -- mass drift -8.0e+3 against a bound of 5 -- and it
// survived a first look because both are f32 arrays of exactly the right
// length. The binding is valid, the writes are in range, and nothing said
// which convention applied.
//
// So the shared coalesce body does not declare `mac` at all. It cannot
// express either layout, because the name is not in scope there; it can only
// call parentMacStore. That is the part that actually prevents the bug.
//
// AND THE TYPE CARRIES THE LAYOUT. array<vec4<f32>> is not decoration: a
// module that hand-indexes this as 4*cell+c stops compiling, and the pool
// sibling's array<f32> is a different type so the two cannot be confused
// inside a fragment either.
//
// MEASURED IN CHROME on an RTX 4080, 2026-09-10, because naga accepting a
// thing is not the same as Tint accepting it and neither is the same as it
// being free:
//   - the binding and a DYNAMIC component index (mac[i][k], the render
//     shader's pattern) both compile and run correctly;
//   - vec4<f32> is exactly four consecutive floats with NO padding, so the
//     bytes are identical to the old 4*cell+c layout and every host readback
//     and bind-group layout is unchanged;
//   - min of 6 alternating rounds of 200 dispatches over 2^20 cells: vec4
//     0.953x the scalar version, i.e. inside this desktop's noise floor
//     (plans/perf-characterization.md: sub-10% differences are not results).
@group(0) @binding(4) var<storage, read_write> mac : array<vec4<f32>>;

fn parentMacStore(cell: u32, v: vec4<f32>) { mac[cell] = v; }
