// Collapsible page chrome -- the sliders, the #status line, the #perf
// readouts, the download button -- plus the one guarantee that makes hiding
// them safe: a fatal error always brings them back.
//
// WHY A MODULE for what is, today, a single page. This is the same shape as
// error-overlay.mjs, and it exists for the reason that module's header
// records: ten near-identical copies of a few lines is exactly how 238e48c
// happened, one copy updated and the others not. The reveal-on-error contract
// is the part that must not drift, and it is enforced from the OTHER side --
// error-overlay.mjs's showFatal() calls revealChrome() -- so any page that
// later adopts the toggle inherits the guarantee instead of reimplementing
// it, and a page that forgets to call installChromeToggle() simply never
// hides anything rather than being able to hide its own error reporting.
//
// The hiding itself is pure CSS, keyed off one class on <body>. Nothing here
// sets style.display on individual elements: the boot smoke
// (tools/validate-all.js's runBootSmoke) reads #status's textContent over
// CDP, which display:none does not affect, and keeping the mechanism to a
// single class keeps it that way by construction.

// The class each page's stylesheet hangs its `display: none` rules off.
// Present on <body> in the markup, so the chrome is hidden from first paint
// rather than flashing visible for a frame and being switched off by script.
export const CHROME_HIDDEN_CLASS = 'ui-hidden';

// Resync callbacks for every installed toggle button, so that a class change
// made from outside (revealChrome, below) still leaves the button's label and
// aria-expanded describing reality. Without this, an error-triggered reveal
// left the button reading "controls" over an already-expanded UI.
const syncs = [];

export function isChromeHidden() {
  return !!(document.body && document.body.classList.contains(CHROME_HIDDEN_CLASS));
}

// Force the chrome visible, whatever the toggle state was.
//
// Called by error-overlay.mjs's showFatal(). The #fatal-overlay box is itself
// never hidden (it is appended to #canvas-container, which is not part of the
// chrome), so this is not what makes the error legible -- it is what puts the
// #status line, carrying the same `error: ...` text, back on screen alongside
// it, and what stops a user from being left with a collapsed UI they then
// have to think to expand. Deliberately one-way: nothing in this module
// re-hides the chrome on its own.
export function revealChrome() {
  if (document.body) document.body.classList.remove(CHROME_HIDDEN_CLASS);
  for (const sync of syncs) sync();
}

// Wire the single show/hide button.
//
// CALL THIS AT MODULE TOP LEVEL, BEFORE init() -- not from inside it. If
// WebGPU setup throws, the rest of init() never runs, so a toggle wired there
// would leave the page collapsed with no way to bring the controls back. The
// button needs nothing from the GPU, so there is no reason for it to wait.
//
// Returns a resync function (for a caller that changes the class itself), or
// null if the page has no toggle button -- in which case the page simply
// keeps whatever state its markup shipped with.
export function installChromeToggle(button, opts = {}) {
  if (!button) return null;
  const showLabel = opts.showLabel || 'controls';
  const hideLabel = opts.hideLabel || 'hide';

  const sync = () => {
    const hidden = isChromeHidden();
    button.textContent = hidden ? showLabel : hideLabel;
    // The button is the only thing on screen when collapsed, so it carries
    // the state for assistive tech too, not just the label text.
    button.setAttribute('aria-expanded', hidden ? 'false' : 'true');
  };

  button.onclick = () => {
    if (document.body) document.body.classList.toggle(CHROME_HIDDEN_CLASS);
    sync();
  };

  sync();
  syncs.push(sync);
  return sync;
}
