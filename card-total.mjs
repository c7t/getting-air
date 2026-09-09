// Unwraps the card's accumulated displacement (x_total/y_total) back into a
// true running total, from the WRAPPED value the physics shaders now keep.
//
// WHY THE SHADERS WRAP AT ALL. physics.wgsl/amr_physics.wgsl accumulate
// `state.y_total += state.vy` every step in f32, and that accumulator used to
// grow without bound. f32 carries 24 bits of mantissa, so its ULP grows with
// the value: at y_total ~1e5 the ULP is ~0.012 and at ~1e6 it is ~0.12,
// against a per-step increment of |vy| ~0.04. Two things degrade together:
//
//   - The TRAIL. trajectory-trail.mjs plots (x_total, y_total), so once the
//     ULP approaches the per-sample increment the y coordinate advances in
//     visible stair-steps while x keeps moving smoothly -- the flat spots in
//     the white line on a long-running page.
//   - The CARD ITSELF, which matters more. Both shaders take the card's
//     sub-cell position straight out of the accumulator's FRACTIONAL part
//     (`cy = initial_cy + (y_total - floor(y_total))`). The fraction is the
//     part the ULP eats first, so the sub-cell position quantizes to coarser
//     and coarser steps -- and get_phi tests that position, so the whole
//     solid coupling inherits the quantization. That is a physics effect, not
//     a cosmetic one, and it has no lower bound: run long enough and the
//     fraction disappears entirely.
//
// WHY WRAPPING IS EXACT, AND INVISIBLE TO THE SIMULATION. The shaders use the
// accumulator for exactly two things, and a wrap by a whole multiple of the
// domain size changes neither:
//   - `off_y = floor(y_total) mod H` -- subtracting K*H shifts floor(y_total)
//     by K*H, an exact multiple of H, so the modulo is unchanged.
//   - `cy = initial_cy + frac(y_total)` -- subtracting an INTEGER leaves the
//     fractional part alone.
// The wrap is a power of two (the domain is, and TOTAL_WRAP_SCREENS is), and
// fires only when the value has just crossed it, so the subtraction is
// between two numbers within a factor of two of each other and is exact in
// f32 -- it introduces no error of its own, it only stops the error growing.
//
// The consumers that want the TRUE total -- the trail, the CSV export, the
// status line, and main-amr.js's backward-jump watchdog, which would
// otherwise read every wrap as a stale readback -- get it from here, in
// float64, whose ULP at 1e9 is still ~1e-7.

// Domain-heights of accumulated displacement to allow before wrapping. Must
// match TOTAL_WRAP_SCREENS in shaders/physics.wgsl and amr_physics.wgsl.
//
// The tension: SMALLER keeps more mantissa for the sub-cell fraction, LARGER
// keeps the unwrap below unambiguous. 16 screens is far more than any single
// readback can cross (a frame moves the card a few lattice units at most,
// against a half-wrap of 8 domain heights) while holding the ULP at W=256 to
// 16*256*2^-23 ~= 4.9e-4 of a cell, permanently, instead of letting it grow.
export const TOTAL_WRAP_SCREENS = 16;

// Phase-unwraps a pair of wrapped accumulators into float64 running totals.
// `span` is the domain size in lattice units (W for x, H for y -- equal on
// every page here, but kept as two arguments so it stays correct if that ever
// stops being true).
export function createTotalUnwrapper(spanX, spanY) {
  const wrapX = spanX * TOTAL_WRAP_SCREENS;
  const wrapY = spanY * TOTAL_WRAP_SCREENS;
  let prevX = null, prevY = null;
  let totalX = 0, totalY = 0;

  // One axis of the standard phase unwrap: attribute a jump larger than half
  // the wrap to the wrap rather than to motion. Safe because the shader wraps
  // by exactly `wrap` and real motion per readback is orders of magnitude
  // smaller.
  const step = (raw, prev, wrap) => {
    let d = raw - prev;
    if (d < -wrap * 0.5) d += wrap;
    else if (d > wrap * 0.5) d -= wrap;
    return d;
  };

  return {
    // Call once per readback, with the raw (wrapped) values straight out of
    // the CardState buffer. Returns the true totals.
    unwrap(rawX, rawY) {
      if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) {
        // A diverged solver must not corrupt the accumulator -- pass the NaN
        // through so the caller's own finiteness checks still see it, and
        // leave prev/total alone so a recovery re-seeds cleanly.
        return { x: rawX, y: rawY };
      }
      if (prevX === null) { totalX = rawX; totalY = rawY; }
      else {
        totalX += step(rawX, prevX, wrapX);
        totalY += step(rawY, prevY, wrapY);
      }
      prevX = rawX; prevY = rawY;
      return { x: totalX, y: totalY };
    },
    // After a reset or a snapshot load the GPU accumulator jumps to an
    // unrelated value; re-seed rather than unwrap across the discontinuity.
    reset() { prevX = null; prevY = null; totalX = 0; totalY = 0; },
  };
}
