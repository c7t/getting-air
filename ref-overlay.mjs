// A world-fixed position reference, so the card's SPEED can be read off the
// screen. Three candidates, switchable live so they can be compared:
//
//   grid     fine lines at a fixed world spacing
//   stars    a jittered "star field" -- one point per world cell at a hashed
//            position, so it is irregular but never clumps (a jittered grid,
//            the cheap cousin of blue noise)
//   crosses  faint crosses at grid intersections only, like the reseau marks
//            on Apollo Hasselblad frames
//
// WHY IT IS NEEDED AT ALL. The card is drawn pinned at the screen centre and
// the flow scrolls past it (see trajectory-trail.mjs's header and
// render.wgsl's SUB-CELL PAN), so nothing on screen is at rest in the frame
// the card is falling through. The air far from the card is nearly still,
// but "nearly still vorticity" is precisely what the colour map renders as
// black. A reference fixed in WORLD coordinates is the missing ruler.
//
// THE MAPPING. render.wgsl draws the pixel at screen fraction (u, v) (v down)
// from window coordinate (u*W + subX, v*H + subY), with the card exactly at
// the centre by default -- so the world point at the centre of the screen is
// the card's (x_total, y_total), and one screen width is W lattice units. A
// page that starts the card off-centre (?cardY=) passes the centre's world
// position, not the card's. Nothing here
// knows about off_x/off_y: the overlay moves with x_total smoothly, exactly as
// the rendered field does since the sub-cell pan.
//
// The caller passes the card's position AT THE STEP BEING SHOWN. The CardState
// readback it comes from is a frame or two stale, so the pages extrapolate by
// the card's own velocity (cardAt, below) -- otherwise the reference lags the
// field by a few cells and visibly swims against it.
//
// A 2D canvas, for trajectory-trail.mjs's reason: no GPU state, so it cannot
// break the simulation if it is wrong.

export const REF_MODES = ['off', 'grid', 'stars', 'crosses'];

// Integer hash -> [0, 1). Deterministic per world cell, so a star is a fixed
// object that scrolls, not a flicker. (Two rounds of a multiply-xorshift;
// quality only has to beat the eye.)
function hash01(ix, iy, salt) {
  let h = (Math.imul(ix, 0x27d4eb2d) ^ Math.imul(iy, 0x165667b1) ^ Math.imul(salt, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// The card's position at `step`, from a stale readback taken at `rb.step`.
// Linear in the card's own velocity (cells per step) -- over the one to three
// frames of readback latency the card's acceleration is negligible.
export function cardAt(rb, step) {
  if (!rb) return null;
  const dt = step - rb.step;
  return { x: rb.x + rb.vx * dt, y: rb.y + rb.vy * dt };
}

export function createRefOverlay(canvas) {
  const ctx = canvas ? canvas.getContext('2d') : null;

  // cx, cy: world coordinates at the screen centre (the card).
  // spanX, spanY: world units across the screen (W, H).
  // spacing: world units between reference marks (the pages pass a chord
  // fraction, so the reference means the same thing at any resolution).
  const draw = (mode, { cx, cy, spanX, spanY, spacing }) => {
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);
    if (mode === 'off' || !Number.isFinite(cx) || !Number.isFinite(cy) || !(spacing > 0)) return;

    const sx = w / spanX, sy = h / spanY;          // px per world unit
    const x0 = cx - spanX / 2, y0 = cy - spanY / 2; // world at the top-left
    const px = (X) => (X - x0) * sx;
    const py = (Y) => (Y - y0) * sy;
    // The world cells the screen overlaps, with one cell of slack so marks
    // enter from off-screen rather than popping in at the edge.
    const i0 = Math.floor(x0 / spacing) - 1, i1 = Math.ceil((x0 + spanX) / spacing) + 1;
    const j0 = Math.floor(y0 / spacing) - 1, j1 = Math.ceil((y0 + spanY) / spacing) + 1;

    // A cool grey that no vorticity colour uses (the map is red/blue on
    // near-black; the card is amber; the trail is white).
    ctx.fillStyle = ctx.strokeStyle = '#b8c7d9';

    if (mode === 'grid') {
      // Pixel-snapped half-pixel offsets keep 1-px lines crisp instead of
      // smeared across two columns.
      ctx.globalAlpha = 0.16;
      ctx.lineWidth = Math.max(1, Math.round(dpr));
      const snap = (v) => Math.round(v) + (ctx.lineWidth % 2 ? 0.5 : 0);
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) { const x = snap(px(i * spacing)); ctx.moveTo(x, 0); ctx.lineTo(x, h); }
      for (let j = j0; j <= j1; j++) { const y = snap(py(j * spacing)); ctx.moveTo(0, y); ctx.lineTo(w, y); }
      ctx.stroke();
    } else if (mode === 'stars') {
      // One star per world cell, jittered within the inner 80% of it, so no
      // two stars come closer than a fifth of a cell: irregular, not clumpy.
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const X = (i + 0.1 + 0.8 * hash01(i, j, 1)) * spacing;
        const Y = (j + 0.1 + 0.8 * hash01(i, j, 2)) * spacing;
        const b = hash01(i, j, 3);
        ctx.globalAlpha = 0.28 + 0.42 * b;
        ctx.beginPath();
        ctx.arc(px(X), py(Y), (0.7 + 0.8 * b) * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (mode === 'crosses') {
      ctx.globalAlpha = 0.38;
      ctx.lineWidth = Math.max(1, Math.round(dpr));
      const arm = Math.max(3 * dpr, Math.min(9 * dpr, 0.12 * spacing * sx));
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const x = Math.round(px(i * spacing)) + 0.5, y = Math.round(py(j * spacing)) + 0.5;
        ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y);
        ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  };

  return { draw };
}
