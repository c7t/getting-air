// D3Q19 / D3Q27 lattice tables -- the single source of truth for the 3D
// velocity sets, shared by the host (JS) and the GPU (WGSL).
//
// WHY THIS EXISTS AT ALL. plans/3D.md ranks a transposed `opp` table as
// "the highest-consequence, lowest-visibility bug available in this port":
// it does not crash, it does not NaN, it produces a plausible-looking wrong
// flow that a Cd/St harness will happily average into a number. The 2D
// tables (shaders/common_lattice.wgsl) are 9 entries hand-written once and
// eyeballed; 27 entries x 5 tables is not eyeballable, and there are two
// velocity sets that must agree with each other.
//
// So the tables are DERIVED, not typed: enumerate {-1,0,1}^3, bucket by
// shell, and emit each velocity immediately followed by its own negation.
// Three consequences fall out for free and are relied on downstream:
//
//   1. `opp` is structural: opp[0] = 0, and otherwise odd i pairs with i+1.
//      There is no table to transpose. tools/test-lattice-3d.js still
//      re-derives it from the emitted ex/ey/ez and checks it, so a change
//      to the ordering cannot silently break bounce-back.
//   2. D3Q19 is a strict PREFIX of D3Q27 (rest, 6 face, 12 edge, then the
//      8 corners Q27 adds). Direction index i means the same thing in both
//      sets, so a Q19 field and a Q27 field are comparable plane by plane
//      and `?q=19|27` (plans/3D.md sec 4) is a loop bound, not a remap.
//   3. Shell boundaries are exact: 1..6 are the face directions, 7..18 the
//      edge directions, 19..26 the corner directions. DIRECT_GHOST needs
//      only the neighbours the lattice points at, and this is what lets
//      "D3Q19 never needs a corner neighbour" (plans/3D.md sec 2.3) be
//      read straight off the index.
//
// Same browser+Node dual-consumption pattern as shader-loader.mjs,
// card-params.mjs and f-pack.mjs: imported natively by the pages, and
// dynamically imported by the CommonJS tools. The WGSL side is GENERATED
// from here (latticeWGSL below, written out by tools/gen-lattice-3d.js and
// checked in, since there is no build step) and tools/test-lattice-3d.js
// asserts the checked-in file still matches -- plus, independently, that
// the tables PARSED BACK OUT of the WGSL satisfy the lattice moment
// conditions. The second half is what catches a wrong table that the
// generator and the file agree about.

export const SUPPORTED_Q = [19, 27];

// Lattice speed of sound squared. 1/3 for both D3Q19 and D3Q27 (and for
// D2Q9) -- the velocity sets differ in isotropy order, not in cs2.
export const CS2 = 1 / 3;

// Weights by shell (|e|^2 = 0, 1, 2, 3). These are the standard sets; the
// moment conditions in tools/test-lattice-3d.js are what actually pin them
// down, so a typo here fails a test rather than quietly detuning cs2.
const WEIGHTS = {
  19: [1 / 3, 1 / 18, 1 / 36],
  27: [8 / 27, 2 / 27, 1 / 54, 1 / 216],
};

// Exact-fraction source for the WGSL emitter, so the shader carries
// `1.0f/18.0f` rather than a truncated decimal that would have to be
// checked for f32 agreement digit by digit. Parallel to WEIGHTS by index.
const WEIGHT_FRACTIONS = {
  19: [[1, 3], [1, 18], [1, 36]],
  27: [[8, 27], [2, 27], [1, 54], [1, 216]],
};

// Per-axis enumeration order. +1 before -1 before 0 is what makes the first
// member of each opposite pair the POSITIVE one (so e_1 is +x, not -x) and
// what puts the face shell in x,y,z order.
const AXIS_ORDER = [1, -1, 0];

// All 27 offsets, in the order the axis loops visit them.
function allOffsets() {
  const out = [];
  for (const dx of AXIS_ORDER) {
    for (const dy of AXIS_ORDER) {
      for (const dz of AXIS_ORDER) out.push([dx, dy, dz]);
    }
  }
  return out;
}

// Velocity set for Q, as an array of [ex, ey, ez]. Shell by shell, and
// within a shell each velocity immediately followed by its negation -- see
// the header for the three properties that ordering buys.
export function velocities(Q) {
  if (!SUPPORTED_Q.includes(Q)) throw new Error(`unsupported velocity set D3Q${Q}`);
  const maxShell = Q === 19 ? 2 : 3;
  const offsets = allOffsets();
  const out = [];
  const seen = new Set();
  const key = (e) => e.join(',');
  for (let shell = 0; shell <= maxShell; shell++) {
    for (const e of offsets) {
      if (e[0] * e[0] + e[1] * e[1] + e[2] * e[2] !== shell) continue;
      if (seen.has(key(e))) continue;
      // `c === 0 ? 0 : -c`, not plain `-c`: negating a 0 component yields
      // -0, which is invisible in printed output but is NOT deepStrictEqual
      // to 0 -- it would make every table comparison downstream subtly
      // order-dependent for no reason.
      const neg = e.map((c) => (c === 0 ? 0 : -c));
      seen.add(key(e));
      out.push(e);
      if (shell !== 0) { seen.add(key(neg)); out.push(neg); }
    }
  }
  if (out.length !== Q) throw new Error(`built ${out.length} velocities for D3Q${Q}`);
  return out;
}

// Weight per direction, by shell.
export function weights(Q) {
  return velocities(Q).map(([x, y, z]) => WEIGHTS[Q][x * x + y * y + z * z]);
}

// Opposite-direction index for bounce-back. Structural given the ordering
// above (rest is self-opposite; every other direction pairs with its
// neighbour), and tools/test-lattice-3d.js re-derives it from the velocity
// table by search rather than trusting this closed form.
export function opposites(Q) {
  const n = velocities(Q).length;
  const out = new Array(n);
  out[0] = 0;
  for (let i = 1; i < n; i++) out[i] = (i % 2 === 1) ? i + 1 : i - 1;
  return out;
}

// Index ranges of each shell, as [start, end) -- rest, face, edge, corner.
// `corner` is an empty range for D3Q19, which is exactly the statement that
// D3Q19 needs no corner neighbour.
export function shellRanges(Q) {
  const v = velocities(Q);
  const norm = (e) => e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  const range = (s) => {
    const idx = v.map((e, i) => [norm(e), i]).filter(([n]) => n === s).map(([, i]) => i);
    return idx.length ? [idx[0], idx[idx.length - 1] + 1] : [v.length, v.length];
  };
  return { rest: range(0), face: range(1), edge: range(2), corner: range(3) };
}

// BGK equilibrium, host side -- the same expression the WGSL feqD3Q emits.
// Used by the host initializer and by any tool that needs to build an
// equilibrium field without a GPU.
export function feq(Q, rho, ux, uy, uz, i) {
  const e = velocities(Q)[i];
  const w = weights(Q)[i];
  const eu = e[0] * ux + e[1] * uy + e[2] * uz;
  const usq = ux * ux + uy * uy + uz * uz;
  return w * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usq);
}

// --- WGSL emission --------------------------------------------------------

function wrapList(items, perLine, indent) {
  const lines = [];
  for (let i = 0; i < items.length; i += perLine) {
    lines.push(indent + items.slice(i, i + perLine).join(', ') + (i + perLine < items.length ? ',' : ''));
  }
  return lines.join('\n');
}

// The checked-in shaders/common_d3q{19,27}_lattice.wgsl, verbatim. Both
// files declare the SAME symbol names (QN, ex, ey, ez, wt, opp, CS2,
// feqD3Q), which is what lets one kernel body fragment be included by a
// Q19 entry file and a Q27 entry file with no other difference -- WGSL has
// no preprocessor, so "one source, two pipeline variants" has to be done
// at the @include level (see plans/3D.md sec 6).
export function latticeWGSL(Q) {
  const v = velocities(Q);
  const opp = opposites(Q);
  const fr = WEIGHT_FRACTIONS[Q];
  const sh = shellRanges(Q);
  const pad = (n) => String(n).padStart(2, ' ');
  const wtItems = v.map(([x, y, z]) => {
    const [num, den] = fr[x * x + y * y + z * z];
    return `${num}.0f/${den}.0f`;
  });
  return `// GENERATED FILE -- do not edit. Source: lattice-3d.mjs (latticeWGSL(${Q})),
// regenerate with \`node tools/gen-lattice-3d.js\`, guarded by
// tools/test-lattice-3d.js (which also re-checks the moment conditions
// against the tables PARSED BACK OUT of this file, so an agreed-upon wrong
// table still fails).
//
// D3Q${Q} lattice basis, weights, bounce-back pairing and BGK equilibrium.
// Fragment only -- included via \`// @include "common_d3q${Q}_lattice.wgsl"\`
// (see shader-loader.mjs), never compiled alone. Every 3D kernel body
// fragment is written against these names and is therefore velocity-set
// agnostic; the entry-point file chooses Q by choosing which of these two
// lattice fragments to include.
//
// Direction order: rest, then face, then edge${Q === 27 ? ', then corner' : ''} -- and within each
// shell every direction is immediately followed by its own negation, so
// opp[i] is i+1 for odd i and i-1 for even i>0. Index ranges:
//   rest   [${sh.rest[0]}, ${sh.rest[1]})
//   face   [${sh.face[0]}, ${sh.face[1]})
//   edge   [${sh.edge[0]}, ${sh.edge[1]})
//   corner [${sh.corner[0]}, ${sh.corner[1]})${Q === 19 ? '   <- empty: D3Q19 has no corner directions, so a\n//                     same-level gather never needs a corner neighbour' : ''}
// D3Q19's indices are a strict PREFIX of D3Q27's: direction i is the same
// physical direction in both sets.

const QN : u32 = ${Q}u;
const CS2 = 1.0f/3.0f; // lattice speed of sound squared

const ex = array<i32,${Q}>(
${wrapList(v.map((e) => pad(e[0])), 10, '  ')}
);
const ey = array<i32,${Q}>(
${wrapList(v.map((e) => pad(e[1])), 10, '  ')}
);
const ez = array<i32,${Q}>(
${wrapList(v.map((e) => pad(e[2])), 10, '  ')}
);
const wt = array<f32,${Q}>(
${wrapList(wtItems, 5, '  ')}
);
// Bounce-back pairing: opposite-direction index for each of the ${Q} links.
const opp = array<u32,${Q}>(
${wrapList(opp.map((o) => `${pad(o)}u`), 10, '  ')}
);

// Named feqD3Q, not feq -- the fused step kernels already declare a local
// \`let feq = ...\` inside their collision loop, and a distinct name keeps
// the two apart without relying on shadowing rules (same reason
// common_lattice.wgsl names the 2D one feqD2Q9).
fn feqD3Q(rho: f32, ux: f32, uy: f32, uz: f32, i: u32) -> f32 {
  let eu = f32(ex[i]) * ux + f32(ey[i]) * uy + f32(ez[i]) * uz;
  return wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*(ux*ux + uy*uy + uz*uz));
}
`;
}
