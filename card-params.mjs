// Pesavento & Wang (2004) "falling card" physical parameters, shared by the
// dense reference (main.js) and the AMR build (main-amr.js).
//
// WHY THIS IS ITS OWN MODULE, not a block copied into each page: the whole
// point of the BLOCKAGE/ASPECT/RE parameterization is that pasting the same
// ?blockage=&aspect=&re=&ut= onto BOTH pages reproduces the identical
// physical system, so the AMR resource win comes purely from picking a lower
// `res` there and letting `levels` recover the body's resolution. That claim
// is only true if both pages derive their parameters from the SAME
// arithmetic -- two hand-synced copies of these formulas would let the pages
// drift apart silently, which is exactly the failure mode that broke
// index-amr.html once already (commit 238e48c: a binding change mirrored
// into main-cylinder-amr.js's bind group but not main-amr.js's own copy of
// the same one). Being a plain .mjs with no DOM/WebGPU dependency also makes
// the arithmetic testable in Node without a GPU -- see
// tools/test-card-params.js, which asserts the cross-page equivalence claim
// directly rather than leaving it to be eyeballed off the two pages' control
// panels.
//
// Same browser+Node dual-consumption pattern as shader-loader.mjs: imported
// natively by the pages' `<script type="module">`, dynamically imported by
// the CommonJS tools under tools/.
//
// SCOPE: main.js and main-amr.js only. The scenario harnesses
// (main-cylinder*.js, main-channel*.js, main-tgv*.js, main-reentry*.js) keep
// their own parameter blocks -- they're pinned-body/uniform-crossflow
// scenarios with a different regime (their own BLOCKAGE/UPSTREAM/RE
// conventions), and they're the currently-validated physics harnesses, so
// folding them in here is a separate, riskier change that should be done
// against a green validation run rather than bundled into this one.

// ── Defaults ─────────────────────────────────────────────────────────────────
// These reproduce the historical hardcoded constants at each page's own
// default resolution -- see CARD_PARAM_DEFAULTS' individual comments and
// tools/test-card-params.js's "legacy defaults" case, which pins them.
export const CARD_PARAM_DEFAULTS = {
  // BLOCKAGE: domain height (W) / chord (2*A), mirroring main-cylinder.js's
  // own BLOCKAGE. Deriving A from W this way (rather than taking A directly
  // as a lattice-unit slider) means the card's size relative to the domain
  // no longer silently changes when the Resolution slider does -- and it is
  // what makes the same card expressible at any grid resolution.
  // 2.0 => A = W/4, i.e. the historical A=64 at the dense page's W=256.
  BLOCKAGE: 2.0,

  // ASPECT: e = B/A, Pesavento & Wang's aspect ratio. Their reference card
  // is a=32, b=4 [lu], e=0.125.
  ASPECT: 0.125,

  // I_STAR: dimensionless moment of inertia,
  //   I* = b(a^2+b^2)rho_b / (2 a^3 rho_f)
  // characterizing the rotation dynamics -- see deriveCardParams' RHO_B.
  I_STAR: 0.34,

  // RE: Reynolds number, the canonical stored flow-regime quantity. TAU is
  // always DERIVED from it (see deriveCardParams) so that changing card size
  // (BLOCKAGE/ASPECT) or U_T holds Re fixed instead of silently drifting it
  // -- the whole point of exposing Re at all.
  //   nu = (TAU-0.5)/3;  Re := U_T*A/nu
  // 1066.667 reproduces main.js's old hand-computed comment exactly:
  // A=64, U_T=0.05, TAU=0.509 -> Re ~= 1066.7 ("Re ~ 1100").
  RE: 1066.667,

  // U_T: target terminal velocity [lattice units/step]. Small enough to keep
  // Ma < 0.1 during the free-fall transient, so even fast tumbles stay well
  // below the Ma < 0.3 limit where LBM accuracy degrades.
  U_T: 0.05,
};

// ── The dense/AMR default pairing ────────────────────────────────────────────
// These two defaults are not independent: the AMR page is meant to run its L0
// grid AMR_DEFAULT_LEVELS-1 octaves COARSER than the dense reference, and to
// recover the body's resolution with that many refinement levels. Out of the
// box the two pages should therefore describe the same physical system at the
// same effective resolution at the body, differing only in how much of the
// far field is resolved -- which is the entire AMR claim.
//
//   AMR_DEFAULT_RES_LOG2 === DENSE_DEFAULT_RES_LOG2 - (AMR_DEFAULT_LEVELS - 1)
//
// tools/test-card-params.js asserts exactly that, so changing one of these
// without the other fails a test instead of silently making the pages'
// default runs incomparable.
export const DENSE_DEFAULT_RES_LOG2 = 8;
export const AMR_DEFAULT_RES_LOG2 = 7;
export const AMR_DEFAULT_LEVELS = 2;

// Resolution clamp both pages enforce. Kept here so the AMR-vs-dense
// resolution ladder (tools/lib/amr-resolution-mapping.js) and the pages
// agree on one set of bounds.
export const RES_LOG2_MIN = 6;
export const RES_LOG2_MAX = 11;

// ── Reynolds <-> tau ─────────────────────────────────────────────────────────
// nu = (tau - 0.5)/3 and Re = u*a/nu, solved each way. `a` is the semi-major
// axis in the LATTICE UNITS OF THE LEVEL WHOSE TAU THIS IS -- for the AMR
// page that means L0's own A with L0's own tau (finer levels then follow
// from tauAtLevel, which preserves Re by construction; see its comment).

export function tauFromReynolds(re, a, u_t) {
  return 0.5 + 3 * u_t * a / re;
}

export function reynoldsFromTau(tau, a, u_t) {
  return 3 * u_t * a / (tau - 0.5);
}

// ── Per-level tau (AMR) ──────────────────────────────────────────────────────
// The Dupuis-Chopard refinement-ratio-2 relation amr_interp_dense_parent.wgsl
// already applies once (tau_fine = 2*tau_coarse - 0.5), walked m times.
// tauAtLevel(tau0, 0) is L0's own tau, tauAtLevel(tau0, 1) is L1's (what
// amr_interp_pool_parent.wgsl needs as `parentTau` when interpolating
// L1->L2), etc.
//
// This is acoustic/convective scaling: halving dx AND dt leaves u in lattice
// units unchanged and doubles nu in lattice units, since
// nu_lattice = nu_phys * dt/dx^2 -> nu_phys * (1/2)/(1/4) = 2*nu_lattice.
// Re is therefore invariant down the hierarchy -- level m has 2^m times the
// cells across the body AND 2^m times the lattice viscosity, so
// Re = u*(2^m*A)/nu_m is unchanged. tools/test-card-params.js asserts exactly
// that, which is what makes the cross-page equivalence claim checkable.
export function tauAtLevel(tau0, m) {
  let t = tau0;
  for (let i = 0; i < m; i++) t = 2 * t - 0.5;
  return t;
}

// ── Derived card/regime quantities ───────────────────────────────────────────
// Pure function of the physical inputs plus the grid width W: everything the
// pages' own `recalculate()` used to compute inline. Returns a fresh object;
// the pages destructure it into their existing module-scope variables so the
// (many) downstream references to A/B/TAU/G_EFF/... keep working unchanged.
export function deriveCardParams({ W, BLOCKAGE, ASPECT, I_STAR, RE, U_T }) {
  // Card geometry from the resolution-independent physical parameters.
  const A = W / (2 * BLOCKAGE);
  const B = A * ASPECT;

  // TAU derived from the canonical RE (see CARD_PARAM_DEFAULTS.RE).
  const TAU = tauFromReynolds(RE, A, U_T);

  // RHO_B: solid-to-fluid density ratio (rho_body/rho_fluid), chosen to
  // satisfy I_STAR. Fluid density is 1.0 in LBM. Higher RHO_B makes the card
  // "heavier" and less affected by small fluid gusts.
  //   I* = b(a^2+b^2)rho_b / (2 a^3)  =>  rho_b = I* * 2a^3 / (b(a^2+b^2))
  // Clamped at 1.05 so the card stays denser than the fluid: G_LU below
  // divides by (RHO_B - 1), which is a buoyancy-neutral singularity at
  // exactly 1.0, and a card lighter than the fluid isn't the modeled regime.
  let RHO_B = I_STAR * 2 * A ** 3 / (B * (A ** 2 + B ** 2));
  RHO_B = Math.max(1.05, RHO_B);

  // MASS: total mass of the 2D ellipse (area * density).
  const MASS = RHO_B * Math.PI * A * B;

  // I_BODY: moment of inertia of a 2D ellipse -- how much torque is needed
  // to change the card's rotation rate.
  const I_BODY = RHO_B * Math.PI * A * B * (A ** 2 + B ** 2) / 4;

  // G_LU: raw gravity, the constant needed to reach U_T against viscous drag.
  const G_LU = U_T ** 2 / (Math.PI * B * (RHO_B - 1));

  // G_EFF: buoyancy-corrected effective gravity -- the card's weight minus
  // that of the displaced fluid, since the fluid pushes back up on it.
  const G_EFF = G_LU * (1 - 1 / RHO_B);

  return { A, B, TAU, RHO_B, MASS, I_BODY, G_LU, G_EFF };
}

// ── URL parsing ──────────────────────────────────────────────────────────────
// One place that knows the query-parameter names, so the two pages can't
// drift on spelling either. `params` is a URLSearchParams (available in both
// the browser and Node).
//
// NOTE the `|| default` idiom is preserved verbatim from the pages: it treats
// an explicit 0 as absent. That is deliberate here -- every one of these
// quantities is a positive physical scale (a zero-chord card, zero Reynolds
// number or zero terminal velocity is not a runnable configuration), so
// falling back to the default is a strictly better outcome than dividing by
// zero several lines later.
export function parseCardParams(params) {
  return {
    BLOCKAGE: parseFloat(params.get('blockage')) || CARD_PARAM_DEFAULTS.BLOCKAGE,
    ASPECT: parseFloat(params.get('aspect')) || CARD_PARAM_DEFAULTS.ASPECT,
    I_STAR: parseFloat(params.get('istar')) || CARD_PARAM_DEFAULTS.I_STAR,
    RE: parseFloat(params.get('re')) || CARD_PARAM_DEFAULTS.RE,
    U_T: parseFloat(params.get('ut')) || CARD_PARAM_DEFAULTS.U_T,
  };
}

// resLog2 with each page's own default (dense defaults to 8; the AMR page
// defaults one step lower, so that at its default levels=2 it covers the same
// physical extent as the dense page with a coarser far field -- the AMR win).
export function parseResLog2(params, fallback) {
  let resLog2 = parseInt(params.get('res')) || fallback;
  if (resLog2 < RES_LOG2_MIN) resLog2 = RES_LOG2_MIN;
  if (resLog2 > RES_LOG2_MAX) resLog2 = RES_LOG2_MAX;
  return resLog2;
}
