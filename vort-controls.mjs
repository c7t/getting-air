// Live controls for the vorticity tone curve (shaders/common_vortcolor.wgsl),
// shared by the dense view (main.js) and the AMR view (main-amr.js).
//
// WHY THESE KNOBS NEED A MODULE AT ALL. VORT_SCALE and VORT_GAMMA are
// pipeline-OVERRIDABLE CONSTANTS -- specialized into the fragment shader when
// the render pipeline is created -- not uniforms. So "move the slider" means
// "rebuild that pipeline", which is real work on a UI event, which means
// coalescing, which means state. That is enough machinery that having two
// hand-synced copies of it (one per page) is the usual mistake, and the two
// pages must behave identically here: the whole point of sharing one color
// mapping is that the two views can be compared by eye, and a knob that
// behaved differently on one of them would quietly defeat that.
//
// WHY NOT JUST MAKE THEM UNIFORMS, which is the obvious way to make an
// override live: ten pages (main.js, main-amr.js and the eight scenario
// harnesses) each own a private copy of renBGL/renBG for the two shared
// render shaders. Adding a binding means editing all ten in lockstep, and
// getting it wrong in exactly one of them is 238e48c -- the bug CLAUDE.md's
// boot-smoke note exists because of. Respecializing one small fragment
// pipeline on a UI action costs a fraction of a frame and touches no binding,
// no bind group layout, and no shader.
//
// This module owns the DOM and the coalescing; the caller owns pipeline
// creation, via the `rebuild` callback. It deliberately knows nothing about
// WebGPU.

// Element ids each page's markup is expected to use. Both are optional -- a
// page with neither simply keeps whatever its pipeline was built with.
const SLIDERS = [
  { key: 'scale', el: 'slider-VORT_SCALE', out: 'val-VORT_SCALE', digits: 0 },
  { key: 'gamma', el: 'slider-VORT_GAMMA', out: 'val-VORT_GAMMA', digits: 2 },
];

// `scale`/`gamma` are the values the pipeline was ALREADY built with -- i.e.
// after ?vortScale=/?vortGamma= have been applied. The sliders are seeded from
// them rather than from their own markup, so a URL-pinned look is not silently
// overridden by the shipped default the moment the page loads.
//
// `rebuild(scale, gamma)` is called at most once per animation frame, with the
// latest values, and never with stale ones. Throwing from it is caught and
// logged: a visualization knob must not be able to take down the simulation.
export function installVortControls({ scale, gamma, rebuild }) {
  const current = { scale, gamma };
  let pending = null;   // non-null == a frame is already queued

  const queue = () => {
    if (pending) { pending = { ...current }; return; }
    pending = { ...current };
    requestAnimationFrame(() => {
      const target = pending;
      pending = null;
      try {
        rebuild(target.scale, target.gamma);
      } catch (e) {
        console.error('[getting-air] vorticity pipeline rebuild failed:', e);
      }
    });
  };

  for (const { key, el, out, digits } of SLIDERS) {
    const slider = document.getElementById(el);
    if (!slider) continue;
    const valEl = document.getElementById(out);
    const show = (v) => { if (valEl) valEl.textContent = v.toFixed(digits); };

    slider.value = current[key];
    show(current[key]);

    slider.oninput = () => {
      const v = parseFloat(slider.value);
      if (!Number.isFinite(v)) return;
      current[key] = v;
      // Readout updates immediately even though the rebuild waits for the
      // next frame, so the control still feels direct under a fast drag.
      show(v);
      queue();
    };
  }
}
