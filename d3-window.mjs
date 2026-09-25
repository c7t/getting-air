// THE MOVING WINDOW for the 3D fork. plans/3D.md M8.3. One statement of the
// convention, shared by the page, the shaders (via shaders/common_d3_window.wgsl,
// which mirrors these four functions line for line) and the tools.
//
// WHY A WINDOW AT ALL, and it is NOT "so a falling body does not leave the
// domain". M8.2b measured the sharper reason: a towed body travels U per
// step, so reaching the 30 D/U the validated sphere cases settle for carries
// it 360 cells through a 192-cell domain. **No moving-body drag measurement
// can converge in a finite domain**, because the body's surroundings keep
// changing while the measurement is being taken -- the towed Cd was still
// rising (0.951 at 4 D/U, 1.067 at 8) toward the streamed 1.362 when it ran
// out of room. The window's job is to make the body's surroundings TIME
// INVARIANT, which is the precondition for any such measurement to converge
// at all.
//
// THIS IS NOT HOW THE 2D PAGES DO IT, deliberately. shaders/physics.wgsl
// keeps off_x/off_y and every 2D kernel treats its DISPATCH index as a
// WINDOW coordinate, converting to a buffer index with `(x + off) % W` at
// every load and store. That works, and in 3D it would mean touching every
// addressing path in the solver -- the dense step, the pool step, moments,
// interp, average, explode, coalesce, force, and the M6 tree sampler --
// each of which is currently free of any window bookkeeping at all.
//
// The other reading of the same simulation costs almost none of that. The
// buffer is periodic; the fluid data never moves; the BODY simply advances
// through the buffer and wraps, exactly as it physically does through the
// fluid. Then:
//
//   - every f_in/f_out index, every tile, every ring and every interface
//     is UNCHANGED, because none of them ever asked where the body was;
//   - the body's SDF gains a nearest-image wrap, so a body straddling the
//     buffer seam is one body and not two;
//   - and the SPONGE -- the one thing that must stay put RELATIVE TO THE
//     BODY -- is moved instead, by converting a buffer position into a
//     window coordinate before measuring its distance to a window face.
//
// **SO THE WINDOW IS NOT A TRANSLATION OF THE FIELD. IT IS A TRANSLATION OF
// WHICH CELLS THE SPONGE HOLDS AT REST.** The band of absorbing cells
// travels with the body: fluid entering it behind the body has its wake
// eaten, and emerges ahead of the body at rest, which is precisely the
// physical statement "the body is falling through undisturbed fluid and
// leaving a wake that does not come back". No frame transformation appears
// anywhere -- the fluid is in the lab frame throughout and the sponge
// relaxes toward the lab-frame far field (still fluid for a fall or a tow),
// which is why nothing in the collision, the forcing or the interface has
// to know the window exists.
//
// THE ANCHOR IS WHERE THE BODY SITS IN THE WINDOW, in L0 cells, and it is
// set to the body's INITIAL position by every caller here. Two consequences
// worth having: the offset is exactly 0 at step 0, so a windowed run starts
// bit-identical to an unwindowed one, and the scenario keeps saying where
// its body sits in its own domain rather than having a second, hidden
// notion of "the middle".
//
// PRECISION IS BETTER HERE THAN IN 2D, and for free. card-total.mjs exists
// because the 2D card's sub-cell position is the FRACTIONAL part of an
// unbounded f32 accumulator, whose ULP eventually eats it. Here the body's
// position is wrapped into [0, N) every step, so it is bounded by
// construction and its ULP is fixed at N * 2^-24 (~1.1e-5 cells at N = 192)
// forever. The running total below is REPORTING ONLY -- nothing reads it
// back into the simulation -- so its own growing ULP costs a fraction of a
// percent of a reported travel distance and nothing else.

// The pipeline-override names shaders/common_d3_window.wgsl declares, in the
// order (size, anchor) x (x, y, z). THE CONTRACT between this module and that
// file, the same way BODY_FIELDS is the contract with BodyState3D: an
// override a module does not declare is a WebGPU validation error, and one it
// declares and nobody sets is a window that silently never moves.
// tools/test-d3-window.js parses the WGSL and asserts the two agree.
export const WINDOW_OVERRIDES = ['WIN_NX', 'WIN_NY', 'WIN_NZ', 'WIN_AX', 'WIN_AY', 'WIN_AZ'];

// The override values for a run: sizes (0 where there is no window) then the
// anchor. Built here so the page and any tool produce the same six numbers
// from the same two inputs.
export function windowConstants(wrapN, anchor) {
  const v = [...wrapN, ...anchor];
  return Object.fromEntries(WINDOW_OVERRIDES.map((k, i) => [k, v[i]]));
}

// Parse a `?window=` spec into a per-axis flag triple. '', '0' and 'none'
// all disable it; otherwise the axes named in the string are enabled.
// Rejects an unknown character rather than silently ignoring it, matching
// d3-scenarios.mjs's refusal to fall back on a typo.
export function parseWindowAxes(spec, fallback = [0, 0, 0]) {
  if (spec === null || spec === undefined) return fallback.slice();
  const s = String(spec).trim().toLowerCase();
  if (s === '' || s === '0' || s === 'none') return [0, 0, 0];
  if (s === '1' || s === 'all' || s === 'xyz') return [1, 1, 1];
  const out = [0, 0, 0];
  for (const ch of s) {
    const i = 'xyz'.indexOf(ch);
    if (i < 0) throw new Error(`?window=${spec}: expected some of xyz, or 0/none/all`);
    out[i] = 1;
  }
  return out;
}

// Nearest-image delta on one axis. `n <= 0` means the axis is not windowed
// and the delta is returned unchanged -- which is what keeps every existing
// scenario bit-identical rather than merely unaffected in practice.
//
// EXACTLY AT THE HALF PERIOD the WGSL mirror disagrees with this by a sign,
// and it does not matter: WGSL's `round` resolves a tie to the even integer
// and JS's `Math.round` resolves it upward, so at |d| = n/2 one returns +n/2
// and the other -n/2. Those are the two equidistant images of the same point,
// every consumer here takes a distance or a body-frame magnitude from it, and
// a cell exactly half a domain from the body is far outside any chi band or
// margin in the first place. Named rather than left as a latent difference
// between two files that are supposed to be one statement.
export function wrapDelta(d, n) {
  if (!(n > 0)) return d;
  return d - n * Math.round(d / n);
}

export function wrapDelta3(d, wrapN) {
  return [wrapDelta(d[0], wrapN[0]), wrapDelta(d[1], wrapN[1]), wrapDelta(d[2], wrapN[2])];
}

// How far the window has travelled, in whole cells. INTEGER, so the sponge
// band stays aligned to the cell grid instead of its edge sliding through a
// cell and flickering; the leftover sub-cell part is simply the body's own
// position within its anchor cell, which is where it belongs.
export function windowOffset(c, anchor, n) {
  if (!(n > 0)) return 0;
  return Math.floor(c) - anchor;
}

export function windowOffset3(c, anchor, wrapN) {
  return [0, 1, 2].map(i => windowOffset(c[i], anchor[i], wrapN[i]));
}

// Buffer position -> window coordinate, in [0, n). The identity on an axis
// with no window, where the buffer IS the window.
export function windowCoord(p, off, n) {
  if (!(n > 0)) return p;
  const w = p - off;
  return w - n * Math.floor(w / n);
}

export function windowCoord3(p, off, wrapN) {
  return [0, 1, 2].map(i => windowCoord(p[i], off[i], wrapN[i]));
}

// Keep a body position inside the buffer. The counterpart of windowCoord:
// one wraps a fluid cell into the window, this wraps the body into the
// buffer, and together they are what lets the body cross the periodic seam
// without anything else in the solver noticing.
export function wrapPosition(c, n) {
  if (!(n > 0)) return c;
  return c - n * Math.floor(c / n);
}

export function wrapPosition3(c, wrapN) {
  return [0, 1, 2].map(i => wrapPosition(c[i], wrapN[i]));
}

// The per-axis domain size to hand the wrap functions: the size on a
// windowed axis, 0 on an unwindowed one. The single place the "0 means off"
// encoding is produced, so no caller has to remember it.
export function wrapDims(dims, axes) {
  return [0, 1, 2].map(i => (axes[i] ? dims[i] : 0));
}
