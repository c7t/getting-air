// The D2Q9 lattice, DERIVED rather than typed -- plans/2D-backport.md B9.
// Sibling of lattice-3d.mjs, which the 3D fork built first and which is where
// the defect below was found.
//
// WHY THIS FILE EXISTS, and it is not tidiness. Before it, the D2Q9 basis was
// typed out in eleven places: `shaders/common_lattice.wgsl` and the `EX`/`EY`/
// `WT` consts in main.js, main-amr.js, main-cylinder.js, main-cylinder-amr.js,
// main-reentry.js, main-reentry-amr.js, main-tgv.js, main-tgv-amr.js,
// main-channel.js and main-channel-amr.js. The shader's and the hosts' values
// were NOT THE SAME NUMBERS, and nothing said so.
//
// THE SHIPPED f32 WEIGHTS WERE NOT THE CORRECTLY ROUNDED FRACTIONS. The
// checked-in shader held eight-digit decimal literals, and `0.02777778f` is
// one f32 ulp ABOVE `fround(1/36)` -- four diagonals, so:
//
//     sum of the shader's f32 weights          = 1 + 1.4901e-8
//     sum of the correctly rounded fractions   = 1 + 7.4506e-9
//
// Every collision computes `feq_i = w_i rho (...)`, so `sum_i feq_i =
// rho (1 + eps)` and `f - omega (f - feq)` INJECTS `omega * rho * eps` of mass
// per cell per step. That compounds linearly for as long as a run lasts.
//
// MEASURED IN THE RUNNING 2D SOLVER, 2026-09-14, by
// tools/analyze-amr-interface.js on TGV (periodic, force-free, so each level
// alone conserves mass exactly and any drift is real). N=128, tau=0.8, 512
// steps, mass gain per cell per step:
//
//     nothing refined      1.861e-8   against eps/tau0        = 1.863e-8
//     everything refined   2.707e-8   against 2*eps/tau1      = 2.709e-8
//
// 0.1% on both paths. The second row is the same defect seen through the AMR
// hierarchy -- a refined cell collides TWICE per macro-step at
// tau_1 = 2*tau_0 - 1/2 rather than once at tau_0 -- so it confirms the
// mechanism, tauAtLevel and the substep count at the same time.
//
// --- AND THE 3D FORK'S FIX DOES NOT TRANSFER. THIS IS THE PART TO READ. ---
//
// lattice-3d.mjs nudges each shell weight by a whole number of f32 ulps so the
// multiplicity-weighted sum is EXACTLY 1, and treats the resulting few-1e-9
// error in the moment conditions as harmless ("a cs2 off by 1e-8 is a pressure
// off by 1e-8, while a conservation law off by 1e-8 compounds"). Ported
// literally to D2Q9 that gives k = [0, 0, -1], and it is WRONG HERE, because
// the moment error does not just sit there -- IT COMPOUNDS TOO, in the
// velocity. Through one BGK collision, with S = sum_i w_i and
// Sig2 = sum_i w_i e_ix^2:
//
//     rho -> rho (1 + omega (S - 1))                  mass
//     M   -> M   (1 + omega (3 Sig2 - 1))             momentum
//     u = M/rho -> u (1 + omega (3 Sig2 - S))         WHAT A GATE MEASURES
//
// So the velocity drift is governed by `3 Sig2 - S`, which for D2Q9 is
// `2 w1 + 8 w2 - w0`. Per step, in units of omega:
//
//     shipped today (0.02777778f)     mass  1.490e-8    velocity  1.490e-8
//     correctly rounded fractions     mass  7.451e-9    velocity  0 EXACTLY
//     the 3D-style tweak [0, 0, -1]   mass  0           velocity -1.490e-8
//
// The correctly rounded fractions already satisfy `2 w1 + 8 w2 - w0 = 0` to
// the BIT -- not approximately: all three are multiples of 2^-26 and the
// combination cancels exactly. The tweak that buys exact mass SPENDS that,
// and creates a velocity drift where there was none. An exhaustive search over
// |k| <= 4 finds NO tweak that zeroes both.
//
// SO D2Q9 SHIPS THE PLAIN FRACTIONS, AT FULL PRECISION. Against what was
// shipped, that halves the mass injection AND removes the velocity drift
// entirely. Mass drift with exact velocity is the benign half of the trade: a
// uniform rho rise in a periodic box has zero gradient and so no dynamics,
// while a velocity drift changes Re and every measured quantity, growing with
// run length.
//
// The tweak machinery below is kept, set to zero, so the search is documented
// and a future velocity set can use it -- and so the two constraints are
// stated where anyone changing the weights will see both.
//
// NOTE FOR THE 3D SIDE, not acted on here: the same arithmetic says D3Q27's
// tweak [-1, 1, -1, 0] trades an EXACTLY zero velocity drift for an exactly
// zero mass drift (plain: mass 7.451e-9, velocity 0; tweaked: mass 0, velocity
// 7.451e-9). D3Q19's [1, -4, 2] is a genuine improvement (mass 1.490e-8 -> 0
// with the velocity drift unchanged in magnitude at 7.451e-9), so the decision
// there was right for the default velocity set and questionable for the
// `?q=27` variant.

// The WGSL is generated from here by tools/gen-lattice-2d.js and CHECKED IN
// (there is no build step -- the source IS the artifact).
// tools/test-lattice-2d.js asserts the checked-in file still matches, and,
// independently, that the tables PARSED BACK OUT of the WGSL satisfy the
// moment conditions and a searched-not-assumed `opp` involution. The second
// half is what catches a wrong table the generator and the file agree about.

// Lattice speed of sound squared. 1/3, as for D3Q19 and D3Q27 -- the velocity
// sets differ in isotropy order, not in cs2.
export const CS2 = 1 / 3;

// THE VELOCITY SET IS ENUMERATED, NOT TYPED, and the order is the one every
// kernel and every host already uses:
//
//     0            rest
//     1..4         axis,     +x, +y, -x, -y
//     5..8         diagonal, (+1,+1), (-1,+1), (-1,-1), (+1,-1)
//
// Both shells go round counter-clockwise from the +x side, which is what makes
// `opp` the +4/-4 rotation within each shell -- a structural property rather
// than a table, so there is nothing to transpose. tools/test-lattice-2d.js
// SEARCHES for the involution rather than assuming it, so a re-ordering that
// broke the property would fail rather than silently ship.
const AXIS = [[1, 0], [0, 1], [-1, 0], [0, -1]];
const DIAG = [[1, 1], [-1, 1], [-1, -1], [1, -1]];

export const E = [[0, 0], ...AXIS, ...DIAG];
export const EX = E.map(e => e[0]);
export const EY = E.map(e => e[1]);

// Which shell each direction belongs to: |e|^2 = 0, 1, 2.
const SHELL = E.map(([x, y]) => x * x + y * y);

// Weights by shell, as exact fractions. The moment conditions in
// tools/test-lattice-2d.js are what actually pin these down, so a typo here
// fails a test rather than quietly detuning cs2.
const WEIGHT_FRACTIONS = [[4, 9], [1, 9], [1, 36]];
const WEIGHT_MULTIPLICITY = [1, 4, 4];

// ZERO -- see the header. Any nonzero entry here buys exact mass by creating a
// velocity drift, and velocity is the channel the gates measure. Kept so the
// mechanism is in one place rather than rediscovered.
const WEIGHT_ULP_TWEAK = [0, 0, 0];

// One f32 ulp of x. frexp-free: the exponent of x is that of the largest
// power of two not exceeding |x|.
function f32ulp(x) {
  const e = Math.floor(Math.log2(Math.abs(x)));
  return 2 ** (e - 23);
}

const SHELL_W = WEIGHT_FRACTIONS.map(([num, den], s) => {
  const base = Math.fround(num / den);
  const w = base + WEIGHT_ULP_TWEAK[s] * f32ulp(base);
  if (Math.fround(w) !== w) throw new Error(`D2Q9 shell ${s} weight ${w} is not an f32`);
  return w;
});

export const WT = SHELL.map(s => SHELL_W[s]);

// Bounce-back pairing, DERIVED from the velocity set rather than typed: the
// opposite of e is -e, and there is exactly one direction carrying it.
export const OPP = E.map(([x, y]) => E.findIndex(([a, b]) => a === -x && b === -y));

// The exact-arithmetic sum of the multiplicity-weighted f32 weights. Exact in
// f64 because every weight is a multiple of 2^-26 and the partial sums stay
// below 2. `weightSum() - 1` is the per-step MASS injection, in units of omega.
export function weightSum() {
  return SHELL_W.reduce((acc, w, s) => acc + w * WEIGHT_MULTIPLICITY[s], 0);
}

// `3*Sig2 - S` = `2 w1 + 8 w2 - w0` for D2Q9: the per-step relative VELOCITY
// drift through one BGK collision, in units of omega. See the header -- this
// is the invariant that must be exactly zero, and the reason the 3D fork's
// weight tweak is not ported. Computed from the shell weights in exact f64,
// which is faithful for the same reason weightSum is.
export function velocityDrift() {
  const [w0, w1, w2] = SHELL_W;
  return 2 * w1 + 8 * w2 - w0;
}

// The fraction each shell's weight was rounded from, and how many ulps it was
// moved -- exported so the test can BOUND the tweak rather than trust it.
export function weightProvenance() {
  return WEIGHT_FRACTIONS.map(([num, den], s) => ({
    exact: num / den, f32: SHELL_W[s], ulps: WEIGHT_ULP_TWEAK[s],
    ulp: f32ulp(Math.fround(num / den)), multiplicity: WEIGHT_MULTIPLICITY[s],
  }));
}

// 17 SIGNIFICANT DIGITS, which is what makes the shader's f32 and this
// module's f32 the SAME NUMBER. A correctly rounding WGSL front end maps a
// 17-digit decimal back to the identical f32; the eight-digit literals this
// replaces did not, and that is the whole of the defect above.
const f32lit = (x) => `${x.toPrecision(17)}f`;

export function latticeWGSL() {
  return `// GENERATED by tools/gen-lattice-2d.js from lattice-2d.mjs. DO NOT EDIT.
// Regenerate with \`node tools/gen-lattice-2d.js\`; tools/test-lattice-2d.js
// fails on drift between the two, and independently checks the lattice moment
// conditions on the tables parsed back OUT of this file.
//
// Shared D2Q9 lattice basis, weights, bounce-back opposite-direction pairing,
// and the BGK equilibrium distribution -- identical across every kernel that
// streams/collides/interpolates on the D2Q9 lattice. Included via
// \`// @include "common_lattice.wgsl"\` (see shader-loader.mjs) rather than
// hand-copied.
//
// THE WEIGHTS ARE NOT THE PLAIN ROUNDED FRACTIONS. Each is moved by a whole
// number of f32 ulps so that they sum to EXACTLY 1; without that, every
// collision injects omega*rho*1.49e-8 of mass per cell per step, which
// compounds linearly for the length of a run. lattice-2d.mjs carries the
// derivation, the measurement and what the trade costs the moment conditions.
// The 17-digit literals are load-bearing: they are what make this file's f32
// and the host's the same number.

const ex = array<i32,9>(${EX.map(v => String(v).padStart(2)).join(',')});
const ey = array<i32,9>(${EY.map(v => String(v).padStart(2)).join(',')});
const wt = array<f32,9>(
${WT.map(w => `  ${f32lit(w)}`).join(',\n')}
);
// Opposite-direction index for each of the 9 links. Structural rather than a
// table: each shell is enumerated counter-clockwise, so the opposite is the
// half-turn within the shell.
const opp = array<u32,9>(${OPP.map(v => `${v}u`).join(', ')});
const CS2 = ${f32lit(Math.fround(CS2))}; // D2Q9 lattice speed of sound squared (1/3)

// Named feqD2Q9, not feq -- several including files (the fused step kernels)
// already have their own local \`let feq = ...\` inside their collision loop; a
// distinct name avoids relying on shadowing rules to keep the two apart.
fn feqD2Q9(rho: f32, ux: f32, uy: f32, i: u32) -> f32 {
  let eu = f32(ex[i]) * ux + f32(ey[i]) * uy;
  return wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*(ux*ux+uy*uy));
}
`;
}
