// The steering INPUT for index-steer.html: the phone held as a wheel
// (steer-imu.mjs does the fusion), with a slider and the arrow keys as the
// desktop fallback, and a small HUD wheel that shows what the page is reading.
//
// One output, `value()`, the steering input s in [-1, 1]; main-steer.js
// writes it to the GPU whenever it changes. The sensor wins while it is
// active; otherwise the slider (which the keys move) is the input.
//
// WHY A BUTTON: motion sensors need a user gesture on iOS
// (DeviceMotionEvent.requestPermission), and so do fullscreen and
// screen.orientation.lock -- which matter here more than they look. Turning a
// phone like a wheel is exactly the motion that trips AUTO-ROTATE at ~45 deg,
// and the page would flip under your hands mid-turn. Locking the orientation
// needs fullscreen on Android Chrome.
//
// THE ZERO IS GRAVITY, NOT YOUR HANDS. The wheel is centred when the SCREEN is
// level -- its own "up" pointing away from the ground -- for whichever
// orientation the display is in when steering starts: portrait 0, landscape
// +-90 deg of device roll. It is read from screen.orientation.angle
// (window.orientation on older iOS) and NOT from the pose the phone happens to
// be in at the tap, which is what this did first: tap while tilted and the
// page steered forever. The zero is taken ONCE, at the start (and again on a
// tap of the wheel), deliberately not on every orientation change: if the
// lock is refused and the display auto-rotates at 45 deg mid-turn, following
// it would flip the steering under your hands.

import { createSteerFilter, steerInput, wrapAngle, motionSample, createAxisDetector, MAPPINGS } from './steer-imu.mjs';

export function installSteerInput({ slider, valEl, button, hud, maxDeg = 45, deadDeg = 2 }) {
  const filter = createSteerFilter();
  let sensorOn = false, neutral = null, lastT = null, psi = 0, sensorS = 0;
  let events = 0, signChecked = false, accelInverted = false;
  // The rotationRate axis mapping (steer-imu.mjs's MAPPINGS): browsers do not
  // agree on it, so it is DETECTED from the first ~25 deg of turning and the
  // verdict remembered for this browser. Until then, Chrome's -- the one this
  // was measured on. A wrong guess costs responsiveness, not correctness: the
  // accelerometer term still holds the angle, only slowly.
  const AXES_KEY = 'getting-air.steer.axes';
  let mapping = 'xyz', axesKnown = false;
  try { const m = localStorage.getItem(AXES_KEY); if (m && MAPPINGS[m]) { mapping = m; axesKnown = true; } } catch (e) { /* storage off */ }
  const axes = createAxisDetector();
  const hudCtx = hud ? hud.getContext('2d') : null;

  const manual = () => (slider ? parseFloat(slider.value) : 0);
  const value = () => (sensorOn && neutral !== null ? sensorS : manual());

  const onMotion = (ev) => {
    const smp = motionSample(ev, mapping);
    if (!smp) return;
    events++;
    // event.timeStamp, not event.interval: the latter is a nominal rate some
    // browsers report as a constant regardless of what actually arrived.
    const t = ev.timeStamp / 1000;
    const dt = lastT === null ? 0 : t - lastT;
    lastT = t;
    if (!axesKnown) {
      const v = axes.push(smp.raw, smp.accel, dt);
      if (v) {
        axesKnown = true;
        if (v !== mapping) { mapping = v; smp.omega = MAPPINGS[v](smp.raw); }
        try { localStorage.setItem(AXES_KEY, v); } catch (e) { /* storage off */ }
      }
    }
    filter.update(smp.omega, smp.accel, dt);
    if (!filter.ready()) return;
    if (neutral === null) return;
    // THE ACCELEROMETER'S SIGN, once. iOS has reported
    // accelerationIncludingGravity negated relative to the spec, which the
    // filter cannot see (u and -u obey the same equation) and which a
    // RELATIVE zero cancelled. An absolute zero does not: negated, level would
    // read 180 deg. So assume the one thing a person starting to steer
    // certainly does -- holds the phone within 90 deg of level -- and flip the
    // reading if the first one says otherwise.
    if (!signChecked) {
      signChecked = true;
      accelInverted = Math.abs(wrapAngle(filter.roll() - neutral)) > Math.PI / 2;
    }
    psi = wrapAngle(filter.roll() + (accelInverted ? Math.PI : 0) - neutral);
    // Screen facing the sky: the wheel angle is undefined there (gravity has
    // no component in the screen plane) and unobservable, so fade to centre
    // rather than steer on a gyro-only estimate that is free to drift.
    const q = filter.inPlane();
    const fade = Math.min(1, Math.max(0, (q - 0.25) / 0.25));
    sensorS = steerInput(psi, { maxDeg, deadDeg }) * fade;
  };

  // The display's rotation from the device's natural (portrait) orientation,
  // degrees, as screen.orientation reports it: 90 when the phone is turned
  // counter-clockwise onto its side. In this module's CLOCKWISE-positive roll
  // that pose is -90 deg, so level is the negative of the screen angle.
  const screenAngle = () => {
    let a = 0;
    if (typeof screen !== 'undefined' && screen.orientation && Number.isFinite(screen.orientation.angle)) a = screen.orientation.angle;
    else if (typeof window.orientation === 'number') a = window.orientation;
    return ((a % 360) + 360) % 360;
  };
  const recenter = () => { neutral = wrapAngle(-screenAngle() * Math.PI / 180); };

  async function start() {
    if (sensorOn) { stop(); return; }
    try {
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        const r = await DeviceMotionEvent.requestPermission();
        if (r !== 'granted') throw new Error(`motion permission ${r}`);
      }
    } catch (e) { if (button) button.textContent = 'no motion access'; return; }
    window.addEventListener('devicemotion', onMotion);
    sensorOn = true; signChecked = false; recenter();
    if (button) { button.textContent = 'steering: tilt'; button.setAttribute('aria-pressed', 'true'); }
    // Best effort; each can refuse (desktop, iOS, an embedded view).
    try {
      if (!document.fullscreenElement && document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      const type = screen.orientation && screen.orientation.type;
      if (type && screen.orientation.lock) await screen.orientation.lock(type);
    } catch (e) { /* the wheel still works; the display may auto-rotate */ }
    // No sensor at all (a desktop fires nothing, or all-null events):
    // say so rather than silently steering on the slider.
    setTimeout(() => { if (sensorOn && events === 0 && button) button.textContent = 'no motion sensor'; }, 1500);
  }
  function stop() {
    window.removeEventListener('devicemotion', onMotion);
    sensorOn = false; neutral = null; lastT = null;
    if (button) { button.textContent = 'steer with phone'; button.setAttribute('aria-pressed', 'false'); }
    try { if (screen.orientation && screen.orientation.unlock) screen.orientation.unlock(); } catch (e) { /* */ }
  }
  if (button) button.onclick = start;
  // Tap the wheel to re-read the screen's orientation -- the way back to a
  // correct zero after the display turned without the lock.
  if (hud) hud.onclick = () => { if (sensorOn) recenter(); else if (slider) { slider.value = 0; slider.oninput && slider.oninput(); } };

  if (slider) slider.oninput = () => { if (valEl) valEl.textContent = manual().toFixed(2); };
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight' && e.key !== 'ArrowDown') return;
    const tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || !slider) return;
    const v = e.key === 'ArrowDown' ? 0 : Math.max(-1, Math.min(1, manual() + (e.key === 'ArrowRight' ? 0.1 : -0.1)));
    slider.value = v.toFixed(2);
    slider.oninput();
    e.preventDefault();
  });

  // HUD: a wheel turned by the input (the SENSED angle while the sensor is
  // on, so a mismatch with your hands is visible), with the input as an arc
  // and the accelerometer trust as a dot -- red while a shake is being
  // ridden out on the gyro alone.
  function drawHud() {
    if (!hudCtx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(hud.clientWidth * dpr), h = Math.round(hud.clientHeight * dpr);
    if (hud.width !== w || hud.height !== h) { hud.width = w; hud.height = h; }
    const c = hudCtx, r = 0.4 * Math.min(w, h), s = value();
    c.clearRect(0, 0, w, h);
    c.save();
    c.translate(w / 2, h / 2);
    c.lineWidth = Math.max(1, 0.06 * r);
    c.strokeStyle = 'rgba(255,255,255,0.25)';
    c.beginPath(); c.arc(0, 0, r, 0, 2 * Math.PI); c.stroke();
    // input arc from the top
    c.strokeStyle = '#ffd76a';
    c.lineWidth = Math.max(2, 0.14 * r);
    c.beginPath(); c.arc(0, 0, r, -Math.PI / 2, -Math.PI / 2 + s * Math.PI / 2, s < 0); c.stroke();
    const ang = sensorOn && neutral !== null ? psi : s * maxDeg * Math.PI / 180;
    c.rotate(ang);
    c.strokeStyle = 'rgba(255,255,255,0.8)';
    c.lineWidth = Math.max(1, 0.08 * r);
    c.beginPath(); c.moveTo(-0.8 * r, 0); c.lineTo(0.8 * r, 0); c.moveTo(0, 0); c.lineTo(0, 0.8 * r); c.stroke();
    c.fillStyle = 'rgba(255,255,255,0.8)';
    c.beginPath(); c.arc(0, -0.8 * r, 0.1 * r, 0, 2 * Math.PI); c.fill();
    c.restore();
    if (sensorOn) {
      const wgt = filter.accelWeight();
      c.fillStyle = wgt > 0.5 ? '#7f7' : wgt > 0.1 ? '#fd6' : '#f55';
      c.beginPath(); c.arc(w - 0.12 * r, 0.12 * r, 0.08 * r, 0, 2 * Math.PI); c.fill();
    }
    if (valEl) valEl.textContent = s.toFixed(2);
  }

  return {
    value,
    drawHud,
    sensorActive: () => sensorOn,
    // For tools and the console: the fused state.
    diag: () => ({ sensorOn, events, mapping, axesKnown, axesEvidence: axes.errors(),
      screenAngle: screenAngle(), neutralDeg: neutral === null ? null : neutral * 180 / Math.PI, accelInverted, psiDeg: psi * 180 / Math.PI, s: value(),
      inPlane: filter.inPlane(), accelWeight: filter.accelWeight(), bias: filter.bias(), up: filter.up() }),
  };
}
