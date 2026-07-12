// Shared "Absorbing/Artificial Lattice Boundary Condition" (ALBC) sponge
// weight -- every step kernel relaxes its post-collision populations
// toward a far-field equilibrium within SPONGE_W cells of a domain edge,
// blended by this cubic-smoothstep weight. SPONGE_W<=0 disables it
// entirely (returns 0 everywhere) -- needed for scenarios with real walls
// or full periodicity instead of an open/free-stream far field.
fn spongeWeight(distX: f32, distY: f32, sw: f32) -> f32 {
  if (sw <= 0.0f) { return 0.0f; }
  var w = clamp(1.0f - min(distX, distY) / sw, 0.0f, 1.0f);
  return w * w * (3.0f - 2.0f * w);
}
