// Visualization shader with smooth analytical mask and vorticity calculation.

// @include "common_geometry.wgsl"
// @include "common_vortcolor.wgsl"

@group(0) @binding(0) var<storage, read> vel   : array<f32>;
@group(0) @binding(1) var<storage, read> state : CardState;

override W : u32;
override H : u32;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

const p = array<vec2<f32>,6>(
  vec2(-1f,-1f), vec2( 1f,-1f), vec2(-1f, 1f),
  vec2(-1f, 1f), vec2( 1f,-1f), vec2( 1f, 1f)
);

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out: VSOut;
  out.pos = vec4(p[vi], 0f, 1f);
  out.uv  = p[vi] * 0.5f + 0.5f;
  return out;
}

fn get_chi(phi: f32) -> f32 {
    return chiFromPhiEps(phi, 1.5f);
}

fn get_uy(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[(by * W + bx) * 2u + 1u];
}

fn get_ux(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[(by * W + bx) * 2u];
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  // SUB-CELL PAN. The moving window pans the FIELD by state.off_x/off_y, which
  // are integers (floor of the accumulated displacement, mod W/H -- see
  // amr_physics.wgsl step 5), while the CARD is drawn at state.cx/cy, which
  // carry the leftover FRACTION of that same displacement. So over one cell
  // crossing the card slides smoothly by up to a cell and the field snaps back
  // by exactly one cell at the moment the fraction wraps. The two stay
  // consistent with each other -- the card is anchored in window space, which
  // is the whole design -- but the composite wobbles by up to one cell in the
  // VIEWPORT. At W=512 that is 1/512 of the frame and invisible; at W=64 it is
  // an obvious jitter of the card and the field together, which is how it was
  // reported (on a phone, "more obvious at the lower resolution").
  //
  // Fix: shift the SAMPLE POINT by that same fraction, for both the body SDF
  // and the field lookup, so the two shift together and nothing moves relative
  // to anything else. The card then lands exactly at the window centre (the
  // pixel at fx = W/2 evaluates phi at W/2 + subX == state.cx) and the field
  // scrolls smoothly sub-cell instead of a cell at a time, which the bilinear
  // reconstruction below already resolves for free.
  //
  // Taken from x_total/y_total rather than from cx - W/2, so it does not
  // assume where a scenario put initial_cx/cy. This is RENDER ONLY -- off_x/
  // off_y, cx/cy and every physics path are untouched, and a scenario with a
  // pinned body (v=0, so x_total stays 0) is byte-identical.
  let subX = state.x_total - floor(state.x_total);
  let subY = state.y_total - floor(state.y_total);
  let fx = uv.x * f32(W) + subX; let fy = (1.0 - uv.y) * f32(H) + subY;
  let ix = i32(fx); let iy = i32(fy);
  
  let chi = get_chi(get_phi(vec2(fx, fy), state));

  // Discrete vorticity: du_y/dx - du_x/dy
  let omega = (get_uy(ix + 1, iy) - get_uy(ix - 1, iy)) * 0.5f
            - (get_ux(ix, iy + 1) - get_ux(ix, iy - 1)) * 0.5f;

  // Blue for clockwise (negative), red for counter-clockwise (positive).
  // Shared with the AMR view -- see shaders/common_vortcolor.wgsl.
  var c = vorticityColor(omega);

  // Blend with solid color
  let solid_color = vec3(1.0, 0.8, 0.4);
  c = mix(c, solid_color, chi);
  
  return vec4(c, 1.0);
}
