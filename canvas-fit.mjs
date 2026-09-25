// Canvas sizing and swapchain configuration -- plans/2D-backport.md B3a.
//
// TEN IDENTICAL COPIES, one per main*.js page, byte for byte. Not a
// coincidence: every page opens the same way, and the reasoning below was
// written once and pasted nine times, which means a fix to it lands in one
// page and not the others. That is the shape CLAUDE.md records producing
// 238e48c.
//
// ONLY RECONFIGURE WHEN THE SIZE ACTUALLY CHANGED, and that guard is the
// whole point of the function. Assigning canvas.width/height resets the
// drawing buffer even when the value is unchanged, and ctx.configure()
// replaces the swapchain, invalidating textures that in-flight command
// buffers still reference (these pages keep up to STAGES frames in flight).
//
// On desktop `resize` fires when you resize the window, so the cost was
// invisible. On a PHONE it fires constantly -- the URL bar hides and shows on
// any scroll or drag, which includes touching the control sliders -- so the
// swapchain was being torn down and rebuilt underneath frames that were
// already submitted. Reported symptom: the view "twitches back" a few frames,
// correlated with moving sliders or switching away and back.
//
// Also guards the degenerate case: clientWidth/Height read 0 during some
// layout transitions (and while hidden), and a 0-sized canvas is not a valid
// configuration.
//
// A FACTORY, not a bare function, because cfgW/cfgH are the state the guard
// is made of. Each page gets its own.
export function makeCanvasFit({ canvas, ctx, device, format }) {
  let cfgW = 0, cfgH = 0;
  return function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (w <= 0 || h <= 0) return;         // mid-layout / hidden: nothing to configure
    if (w === cfgW && h === cfgH) return; // same size: reconfiguring is pure damage
    cfgW = w; cfgH = h;
    canvas.width = w;
    canvas.height = h;
    ctx.configure({ device, format, alphaMode: 'opaque' });
  };
}
