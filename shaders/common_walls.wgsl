// Axis-aligned no-slip channel walls at the y=0 and y=H-1 domain edges --
// a position-based bounce-back test (not an SDF against a CardState body),
// for scenarios with real tunnel walls instead of (or in addition to) an
// interior body: plane Poiseuille/Couette channel flow. Included via
// `// @include "common_walls.wgsl"`.
//
// Convention: the physical wall sits HALFWAY between the last fluid row
// and the (nonexistent) row just past it -- e.g. the "y=0 edge" wall is at
// y=-0.5, and the "y=H-1 edge" wall is at y=H-0.5. So the physical
// wall-to-wall channel height equals H exactly (the two half-cell offsets
// on either end cancel), not H+1 or H-1. Depends on the including file's
// own `override H : u32;`.

// True if direction i's streaming source for a cell at row `y` lies
// outside the channel (y in [0,H)) -- i.e. a bounce-back wall reflection
// is needed here, not the default periodic-wrap streaming read.
fn wallSourceOutside(y: u32, eyi: i32) -> bool {
  let ySrc = i32(y) - eyi;
  return ySrc < 0 || ySrc >= i32(H);
}

// The wall's own local x-velocity for a reflection at row `y` in
// direction i -- u0 for the y=0 edge, u1 for the y=H-1 edge. The walls
// are horizontal and don't move vertically, so usy is always 0 (callers
// only need this, not a usy counterpart).
fn wallVelocityX(y: u32, eyi: i32, u0: f32, u1: f32) -> f32 {
  let ySrc = i32(y) - eyi;
  return select(u1, u0, ySrc < 0);
}

// Continuous-position counterparts of the two functions above, for AMR
// fine-level kernels where a cell's coarse-equivalent y position is a
// float (fractional fine-cell offsets), not an integer row index. Callers
// pass the UNWRAPPED buffer-space coarse-unit position of the streaming
// source directly (e.g. amr_step1.wgsl's own fineToCoarseUnitI output) --
// this assumes off_y is always 0 for any scenario using WALL_Y (no
// moving-window panning), so buffer-space position IS window-space
// position with no further conversion needed.
fn wallSourceOutsideF(yPos: f32) -> bool {
  return yPos < 0.0f || yPos >= f32(H);
}

fn wallVelocityXF(yPos: f32, u0: f32, u1: f32) -> f32 {
  return select(u1, u0, yPos < 0.0f);
}
