// Vorticity -> color, shared by the dense view (shaders/render.wgsl) and the
// AMR view (shaders/amr_render.wgsl).
//
// WHY SHARED: those two views exist to be compared by eye -- the whole point
// of the BLOCKAGE/ASPECT/RE parameterization is that the same physical system
// can be put on both pages and looked at side by side. A duplicated color ramp
// is a duplicated thing to keep in step, and "the AMR page looks more
// saturated than index.html" is exactly the observation that is impossible to
// interpret if the two are not provably using the same mapping. They were
// byte-identical copies before this file existed, so there was nothing to
// reconcile in moving them here.
//
// LEVEL INDEPENDENCE. The `omega` arriving here is already normalized to
// "d(u_lattice) / d(x in L0 cells)" at every level, so this mapping is NOT
// level-dependent and must not try to compensate for level:
//
//   level m has spacing dx_m = 2^-m L0 cells, and u in lattice units is
//   IDENTICAL across levels (acoustic scaling -- halving dx and dt together
//   leaves u alone; see card-params.mjs's tauAtLevel). A central difference
//   spanning +/-1 cell of level m is therefore divided by 2*2^-m, i.e.
//   multiplied by 2^(m-1):
//       L0 -> 0.5   (coarseOmegaCell's 0.5)
//       L1 -> 1.0   (fineOmegaCell, no factor)
//       L2 -> 2.0   (fineOmegaCell2's 2.0)
//   which is exactly what those three functions apply.
//
// So a refined region legitimately shows LARGER |omega| than the same flow
// on the coarse grid: the finer stencil resolves a vortex core the coarse
// central difference was averaging flat. That is real detail, not a scaling
// bug -- which is why the answer to "the fine levels look over-saturated" is
// the tone curve below rather than a per-level fudge factor.
//
// THE TONE CURVE. The mapping used to be `clamp(omega * 80, -1, 1)` into a
// linear ramp: every |omega| >= 1/80 collapsed to exactly the same fully
// saturated color, so vortex cores -- the most interesting part of the field,
// and the part the fine levels exist to resolve -- rendered as flat blobs.
// Worse, adding levels made it WORSE, because sharper cores mean more of the
// field sits above the clip.
//
// Reinhard (a/(1+a)) is used instead: monotonic, asymptotic to 1 but never
// reaching it, so there is always some gradient left at the top end no matter
// how strong the core. VORT_GAMMA then shapes the low end, either lifting it
// back toward the old linear ramp's brightness (below 1) or deepening the
// background so the structure stands out against it (above 1).
//
// Both knobs are pipeline-overridable and driven from `?vortScale=` /
// `?vortGamma=` on either page (main.js and main-amr.js parse them
// identically), so the look can be dialed against a live run instead of
// guessed at and recompiled. Note the OLD look is not exactly reproducible by
// any setting here -- the hard clip is the thing being removed.

// Reference scale: |omega| * VORT_SCALE == 1 is the curve's knee. 80 was the
// old mapping's saturation point, so 40 puts the knee at twice the vorticity
// -- less of the field pushed bright. Larger = brighter overall.
override VORT_SCALE : f32 = 40.0;
// Shapes the compressed magnitude. 1.0 = pure Reinhard. Below 1 lifts weak
// vorticity toward its old brightness while leaving the top end compressed;
// above 1 darkens the field generally. 1.2 deepens the near-black background
// so the wake reads against it instead of sitting in a haze.
override VORT_GAMMA : f32 = 1.2;

// Near-black background, warm red for counter-clockwise (positive), cool blue
// for clockwise (negative). Unchanged from the mapping this replaced.
const VORT_BG  = vec3<f32>(0.05, 0.05, 0.1);
const VORT_POS = vec3<f32>(1.0, 0.3, 0.2);
const VORT_NEG = vec3<f32>(0.2, 0.5, 1.0);

// Signed, non-clipping tone response in (-1, 1).
fn vorticityTone(omega: f32) -> f32 {
  let a = abs(omega) * VORT_SCALE;
  let compressed = a / (1.0 + a);   // in [0,1), never saturates
  let mag = pow(compressed, VORT_GAMMA);
  return select(-mag, mag, omega >= 0.0);
}

fn vorticityColor(omega: f32) -> vec3<f32> {
  let val = vorticityTone(omega);
  if (val > 0.0) {
    return mix(VORT_BG, VORT_POS, val);
  }
  return mix(VORT_BG, VORT_NEG, -val);
}
