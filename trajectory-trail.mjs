// Rolling trajectory trail -- the white line tracing the falling card's path,
// drawn as a 2D-canvas overlay on top of the WebGPU canvas. Shared by the
// dense view (main.js) and the AMR view (main-amr.js), which log the card's
// position identically.
//
// WHY A 2D OVERLAY AND NOT A GPU PASS. Drawing a polyline in the render
// shader would mean a new storage binding for the point buffer, and ten pages
// each own a private copy of renBGL/renBG for the two shared render shaders --
// the 238e48c hazard CLAUDE.md's boot-smoke note exists because of. The path
// is already on the CPU (it comes from the same CardState readback that feeds
// the CSV export), so a 2D canvas needs no GPU state at all and cannot break
// the simulation if it is wrong.
//
// WHY IT IS A PLOT, NOT A WORLD-SPACE OVERLAY. The card is drawn at the
// SCREEN CENTRE and never moves: amr_physics.wgsl pins it at buffer
// (W/2, H/2) and pans off_x/off_y by floor(x_total/y_total), so the field
// scrolls past a stationary card, and common_geometry.wgsl's get_phi tests
// screen coordinates directly against state.cx/cy. A trail drawn in the
// flow's own frame would therefore always end at the screen centre and extend
// upward behind the card, and could never show more than half a screen --
// four chords at the shipped defaults -- before running out of canvas.
//
// So this is a diagram in the spirit of Pesavento & Wang's Fig. 2(b): its own
// frame, its own anchoring, sharing only the canvas. It is NOT spatially
// registered with the rendered flow, and the head of the line is not where
// the rendered card is.
//
// HOW MUCH TO SHOW. The paper's Fig. 2(b) covers about 7 chords of descent
// over five tumbling periods and Fig. 5 about 10, so DEFAULT_CHORDS_TALL = 8
// sits between them. One chord is 2A, which at the shipped BLOCKAGE=8 and
// W=256 makes 8 chords exactly one domain height -- the trail fills the
// canvas without the framing having to be told about the resolution.
//
// FRAMING, vertically: the trail starts at the top of the canvas and grows
// downward; once it has descended a full window the frame tracks the head, so
// the newest point sits near the bottom and older points scroll off the top.
// Both regimes are the same expression -- see yTop below.
//
// FRAMING, horizontally: the frame re-centres on the buffered path's own
// horizontal extent, and zooms OUT (isotropically, never in past the
// chords-tall framing) if that extent would not otherwise fit. Both paper
// figures drift further sideways than they fall -- 9 chords across 7, and 12
// across 10 -- so a fixed-origin frame would run the line off the canvas
// within one buffer length. Scale and centre are both low-pass filtered, or
// the frame twitches on every shed vortex.

export const DEFAULT_CHORDS_TALL = 8;

// Fraction of the visible height the head sits at once the trail is
// scrolling. Short of 1.0 so the newest point is not clipped by the edge.
const HEAD_FRAC = 0.94;
// Keep a little more than one window of history, so re-centring and zooming
// out have something to work with rather than popping as points expire.
const KEEP_SLACK = 1.15;
// Leave a margin when fitting horizontally, so a path that only just fits is
// not drawn hard against the edges.
const FIT_MARGIN = 1.12;
// Points closer together than this (in lattice units) add nothing at any
// plausible zoom, and only cost memory and draw time.
const MIN_POINT_SPACING = 0.15;
// Hard ceiling, for the pathological case of a card that hovers instead of
// descending: distance-based pruning alone would never fire.
const MAX_POINTS = 6000;
// Exponential smoothing for the frame. Low enough that the frame reads as
// steady, high enough that it keeps up with a card that starts gliding.
const FRAME_LERP = 0.06;

export function createTrail(canvas, opts = {}) {
  const chordsTall = opts.chordsTall || DEFAULT_CHORDS_TALL;
  const ctx = canvas ? canvas.getContext('2d') : null;

  // Parallel arrays rather than {x,y} objects: this is appended to on every
  // readback and walked in full on every frame.
  let xs = [];
  let ys = [];
  // Smoothed frame state; null until the first draw seeds them, so the trail
  // does not visibly slide into place on its first frame.
  let viewCx = null;
  let viewScale = null;

  const clear = () => {
    xs = []; ys = [];
    viewCx = null; viewScale = null;
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const push = (x, y, chord) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const n = xs.length;
    if (n) {
      const dx = x - xs[n - 1], dy = y - ys[n - 1];
      if (dx * dx + dy * dy < MIN_POINT_SPACING * MIN_POINT_SPACING) return;
    }
    xs.push(x); ys.push(y);

    // Distance-based pruning: the buffer holds a fixed span of DESCENT, not a
    // fixed number of samples, so it means the same thing at any frame rate,
    // any resolution and any steps-per-frame.
    const keep = chordsTall * chord * KEEP_SLACK;
    const cut = y - keep;
    let drop = 0;
    while (drop < xs.length && ys[drop] < cut) drop++;
    if (drop > 0) { xs.splice(0, drop); ys.splice(0, drop); }
    if (xs.length > MAX_POINTS) {
      const excess = xs.length - MAX_POINTS;
      xs.splice(0, excess); ys.splice(0, excess);
    }
  };

  const draw = (chord, opacity) => {
    if (!ctx) return;
    // Match the backing store to the CSS box, at device resolution -- the
    // canvas is sized by the same min(100cqw,100cqh) rule as the WebGPU one,
    // so this follows the window without needing its own layout maths.
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);
    if (opacity <= 0 || xs.length < 2 || !(chord > 0)) return;

    const n = xs.length;
    const yNow = ys[n - 1];
    let xMin = xs[0], xMax = xs[0];
    for (let i = 1; i < n; i++) {
      if (xs[i] < xMin) xMin = xs[i];
      if (xs[i] > xMax) xMax = xs[i];
    }

    // Isotropic scale: the chords-tall framing is the CEILING, so a narrow
    // path gets exactly the intended window and only a wide one zooms out.
    // Anisotropic fitting would distort the path's shape, which is the whole
    // thing being looked at -- a glide and a tumble differ by shape.
    const spanYTarget = chordsTall * chord;
    const spanXNeeded = Math.max((xMax - xMin) * FIT_MARGIN, 1e-6);
    const targetScale = Math.min(h / spanYTarget, w / spanXNeeded);
    const targetCx = (xMin + xMax) * 0.5;

    if (viewScale === null) { viewScale = targetScale; viewCx = targetCx; }
    else {
      viewScale += (targetScale - viewScale) * FRAME_LERP;
      viewCx += (targetCx - viewCx) * FRAME_LERP;
    }

    // HARD CONTAINMENT, applied after the filter. Smoothing decides how the
    // frame MOVES; it must not be able to hide the line. A fast lateral
    // excursion outruns a 0.06 lerp by several frames, which showed up as the
    // head running off the left edge -- measured, not hypothesised. Clamping
    // here rather than raising FRAME_LERP keeps the frame steady in the usual
    // case and only intervenes when it would otherwise clip.
    if (viewScale > w / spanXNeeded) viewScale = w / spanXNeeded;
    // Clamp to an INSET, not to the edge itself. Clamping to halfW pins the
    // offending extreme exactly on the boundary, which still puts ink in the
    // outermost column (measured: 5 edge pixels at xMax=0.999). spanXNeeded
    // carries FIT_MARGIN=1.12, so 0.97*halfW is still strictly wider than
    // half the path and the two clamps cannot fight each other.
    const insetW = w * 0.5 / viewScale * 0.97;
    if (xMin < viewCx - insetW) viewCx = xMin + insetW;
    if (xMax > viewCx + insetW) viewCx = xMax - insetW;

    // One expression for both regimes. Early on, yNow - window sits above the
    // oldest point, so yTop pins to it and the trail hangs from the top of
    // the canvas. Once the descent exceeds the window the second term wins
    // and the frame tracks the head, scrolling the oldest points off.
    const spanYEff = h / viewScale;
    const yTop = Math.max(ys[0], yNow - spanYEff * HEAD_FRAC);
    const sx = (x) => w * 0.5 + (x - viewCx) * viewScale;
    const sy = (y) => (y - yTop) * viewScale;

    // Drawn oldest-to-newest in segments with rising alpha, so the tail fades
    // out rather than ending in a hard cut where the buffer happens to stop.
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(1, 1.6 * dpr);
    ctx.strokeStyle = '#fff';
    const SEGMENTS = 24;
    for (let s = 0; s < SEGMENTS; s++) {
      const i0 = Math.floor((s * (n - 1)) / SEGMENTS);
      const i1 = Math.floor(((s + 1) * (n - 1)) / SEGMENTS);
      if (i1 <= i0) continue;
      ctx.globalAlpha = opacity * (0.12 + 0.88 * ((s + 1) / SEGMENTS));
      ctx.beginPath();
      ctx.moveTo(sx(xs[i0]), sy(ys[i0]));
      for (let i = i0 + 1; i <= i1; i++) ctx.lineTo(sx(xs[i]), sy(ys[i]));
      ctx.stroke();
    }

    // Head marker: the line alone does not say which end is "now", and the
    // rendered card cannot say it either -- that one is pinned at the screen
    // centre and belongs to a different frame entirely (see the header).
    ctx.globalAlpha = opacity;
    ctx.beginPath();
    ctx.arc(sx(xs[n - 1]), sy(yNow), Math.max(1.5, 2.2 * dpr), 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.globalAlpha = 1;
  };

  return { push, draw, clear, get length() { return xs.length; } };
}
