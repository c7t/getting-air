// Three-axis ALBC sponge weight -- the 3D counterpart of
// common_sponge.wgsl, which takes two distances. Every step kernel relaxes
// its post-collision populations toward a far-field equilibrium within
// SPONGE_W cells of a domain face, blended by this cubic smoothstep.
// SPONGE_W <= 0 disables it entirely, which is what the periodic and
// walled scenarios use.
//
// min() over all three distances, not a product: a corner should be fully
// absorbing, not absorbing-cubed.
fn spongeWeight3(dx: f32, dy: f32, dz: f32, sw: f32) -> f32 {
  if (sw <= 0.0f) { return 0.0f; }
  var w = clamp(1.0f - min(dx, min(dy, dz)) / sw, 0.0f, 1.0f);
  return w * w * (3.0f - 2.0f * w);
}
