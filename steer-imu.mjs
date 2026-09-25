// A DRIFT-FREE STEERING WHEEL from a phone's gyro and accelerometer.
//
// Pure (no DOM), so tools/test-steer-imu.js can drive it with a simulated
// phone whose true orientation is known. main-steer.js feeds it
// DeviceMotionEvent samples.
//
// WHAT "STEERING ANGLE" MEANS HERE: the phone's rotation about its own screen
// normal (device z), measured against gravity. Hold the phone like a wheel and
// turn it; that is the angle. It is the one rotation that is fully OBSERVABLE
// from the accelerometer whenever the screen is not facing the sky: gravity,
// seen in the device frame, turns in the screen plane by exactly the wheel
// angle, whatever the phone's pitch (how far it is reclined) and whatever its
// yaw (which way you face). Neither of those moves it -- see roll() below.
//
// SO THE FILTER TRACKS ONE VECTOR, NOT AN ORIENTATION. `u` is the world's UP
// direction expressed in device coordinates. A full attitude quaternion would
// carry yaw, which nothing here can correct (no magnetometer) and nothing
// here needs.
//
//   PREDICT (gyro, fast, drifts): u is fixed in the world, so in the device
//     frame it turns the other way from the device:  du/dt = u x omega.
//   CORRECT (accelerometer, slow, noisy, never drifts): pull u toward the
//     measured up direction a/|a| with gain kP -- a Mahony complementary
//     filter restricted to one vector. The correction is itself a rotation
//     rate, e = a_n x u, added to omega before integrating: u x (a_n x u) is
//     exactly a_n's component perpendicular to u, so it only ever TURNS u,
//     and its component along u is identically zero.
//   GYRO BIAS (integral term, kI): a constant bias b would otherwise leave a
//     standing tilt error of ~b/kP (1 deg/s of bias, a poor but real phone
//     gyro, is ~0.6 deg at kP = 1.5). The integral estimates b and removes
//     it, so the steady state is exact, not merely bounded.
//
// DRIFT-FREE BY CONSTRUCTION, not by tuning: the only state that could
// accumulate error is u's direction about the device z axis, and that is a
// tilt axis relative to gravity whenever the screen is upright-ish, i.e. a
// direction the accelerometer term corrects. The gyro contributes response
// (no lag, no shake noise); the accelerometer contributes the reference.
//
// LINEAR ACCELERATION: the accelerometer measures gravity PLUS whatever the
// hand is doing. `accelGate` down-weights a sample by how far |a| is from 1 g,
// so a shake is carried through on the gyro alone. A sustained sideways
// acceleration at exactly 1 g total cannot be told from a tilt by any IMU;
// that is physics, not a defect here.
//
// SIGN-AGNOSTIC in the accelerometer: iOS has historically reported
// accelerationIncludingGravity with the opposite sign to the spec. Negating
// a_n negates the u the filter settles on, du/dt = u x omega is linear so -u
// obeys it too, and the wheel angle is measured RELATIVE to a calibrated
// neutral, so a constant 180 deg offset cancels.

const G = 9.80665;

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

// Rodrigues: v rotated by angle `ang` about unit axis k.
function rotate(v, k, ang) {
  const c = Math.cos(ang), s = Math.sin(ang);
  const kxv = cross(k, v), kv = dot(k, v);
  return [0, 1, 2].map(i => v[i] * c + kxv[i] * s + k[i] * kv * (1 - c));
}

export const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export function createSteerFilter({ kP = 1.5, kI = 0.3, accelGate = 0.25, angleGate = 8, gateFloor = 0.05, biasMax = 0.2 } = {}) {
  const sGate = Math.sin(angleGate * Math.PI / 180);
  let u = null;           // world up in device coords (unit), null until the first sample
  let bias = [0, 0, 0];   // gyro bias estimate, rad/s
  let weight = 0;         // last accelerometer trust in [0, 1], for the HUD

  // du/dt = u x w, exactly, over dt: u turned by -|w| dt about w.
  const turn = (w, dt) => {
    const wn = norm(w);
    if (wn > 1e-12) u = rotate(u, w.map(x => x / wn), -wn * dt);
    const un = norm(u);
    u = u.map(x => x / un);
  };

  return {
    // omega: device-frame angular velocity, rad/s. accel: device-frame
    // accelerationIncludingGravity, m/s^2 (either sign convention). dt: s.
    update(omega, accel, dt) {
      const an = norm(accel);
      if (!u) {                                  // first sample: trust the accelerometer outright
        if (!(an > 0.3 * G)) return;
        u = accel.map(x => x / an);
        return;
      }
      if (!(dt > 0)) return;
      dt = Math.min(dt, 0.1);                    // a stalled event loop must not fling u
      // PREDICT, THEN CORRECT against the accelerometer AT THE SAME INSTANT.
      // Comparing the accel sample (end of the interval) with u from the
      // START of it counts the step's own motion as error: the correction
      // then chases the gyro and settles at a lag of ~omega*dt -- measured
      // 1.2 deg on a 1.3 Hz flick with perfect sensors, gone with this order.
      turn([0, 1, 2].map(i => omega[i] - bias[i]), dt);
      weight = an > 0 ? Math.max(0, 1 - Math.abs(an / G - 1) / accelGate) : 0;
      if (weight > 0) {
        let e = cross(accel.map(x => x / an), u);
        // ...and by how far it DISAGREES with the gyro's prediction. |a| alone
        // cannot see a shake perpendicular to gravity (6 m/s^2 sideways moves
        // |a| by only 17% while tilting a/|a| by 31 deg), but the gyro,
        // integrated over a fraction of a second, knows u to well under a
        // degree. Full trust inside angleGate, falling to gateFloor at twice
        // it. The floor is what keeps this from latching: a real large
        // disagreement (a clipped gyro, a bad first sample) still converges,
        // just ~20x more slowly.
        const s = norm(e);
        weight *= Math.max(gateFloor, Math.min(1, 2 - s / sGate));
        e = e.map(x => x * weight);
        // Bias integral. The steady state of the correction is
        // kP e = -(b - bias), so moving bias against e drives it onto b.
        bias = bias.map((b, i) => Math.max(-biasMax, Math.min(biasMax, b - kI * e[i] * dt)));
        turn(e.map(x => kP * x), dt);
      }
    },
    // The wheel angle, radians, CLOCKWISE-positive as seen facing the screen.
    // Pitch- and yaw-invariant: a rotation about device z turns (u_x, u_y)
    // rigidly, and pitch and yaw only scale or leave that pair alone.
    roll() { return u ? Math.atan2(-u[0], u[1]) : 0; },
    // |(u_x, u_y)|: 1 with the screen vertical, 0 with it facing the sky --
    // where the wheel angle is undefined (and unobservable), so callers fade.
    inPlane() { return u ? Math.hypot(u[0], u[1]) : 0; },
    ready() { return !!u; },
    up() { return u ? u.slice() : null; },
    bias() { return bias.slice(); },
    accelWeight() { return weight; },
  };
}

// Wheel angle (radians, relative to neutral) -> steering input in [-1, 1],
// with a continuous deadband so the card does not twitch on a held wheel.
export function steerInput(psi, { maxDeg = 45, deadDeg = 2 } = {}) {
  const deg = Math.abs(psi) * 180 / Math.PI;
  const mag = Math.min(1, Math.max(0, deg - deadDeg) / (maxDeg - deadDeg));
  return Math.sign(psi) * mag;
}

// DeviceMotionEvent -> the filter's units.
//
// rotationRate is in DEGREES per second everywhere that matters, but WHICH
// AXIS each of alpha/beta/gamma names IS NOT AGREED ON. The spec says alpha,
// beta, gamma are about device z, x, y. Chrome delivers them as x, y, z --
// measured, not assumed: an emulated gyroscope reading of (0, 0, -30 deg/s)
// arrives as {alpha: 0, beta: 0, gamma: -30}, through the same conversion
// real hardware takes. Reading it the spec's way feeds the steering rate to
// a tilt axis, and the wheel then moves only as fast as the accelerometer
// term drags it (7 deg of a 30 deg turn, the first time this was driven).
//
// So the mapping is not hard-coded: createAxisDetector below settles it from
// the data, and MAPPINGS names the candidates. Returns null for an event with
// no gyro (some desktops fire devicemotion with every field null).
const RAD = Math.PI / 180;
export const MAPPINGS = {
  xyz: (r) => [r.alpha * RAD, r.beta * RAD, r.gamma * RAD],   // Chrome
  spec: (r) => [r.beta * RAD, r.gamma * RAD, r.alpha * RAD],  // the spec's words
};
export function motionSample(ev, mapping = 'xyz') {
  const r = ev.rotationRate, a = ev.accelerationIncludingGravity;
  if (!r || !a || r.alpha == null || a.x == null) return null;
  return { omega: MAPPINGS[mapping](r), accel: [a.x, a.y, a.z], raw: r };
}

// WHICH MAPPING IS THIS BROWSER USING? The accelerometer answers it: the
// measured up direction turns as du/dt = u x omega, and only the correct
// mapping predicts the turn. Each candidate integrates its own reading of the
// gyro from an ANCHOR (a quiet accelerometer sample) across a window of
// samples, and its squared miss against the accelerometer at the window's end
// is summed. A window and not consecutive samples, because one sample's turn
// (~1 deg at a brisk 60 deg/s) is comparable to the accelerometer's noise,
// and for a phone held in landscape one of the wrong permutations moves the
// prediction very little -- measured: per-sample comparison could not tell a
// spec browser from a Chrome one in 1400 deg of steering. Over a window the
// signal grows with its length and the noise does not.
//
// Only quiet samples (|a| near 1 g) anchor or close a window, and a verdict
// needs enough rotation seen AND one candidate explaining the motion clearly
// better than the other.
export function createAxisDetector({ minTurnDeg = 25, ratio = 0.3, window = 10 } = {}) {
  const keys = Object.keys(MAPPINGS);
  const err = Object.fromEntries(keys.map(k => [k, 0]));
  let pred = null, n = 0, seen = 0, verdict = null;
  return {
    push(raw, accel, dt) {
      if (verdict) return verdict;
      const an = norm(accel);
      const quiet = Math.abs(an / G - 1) < 0.05;
      const a = accel.map(x => x / an);
      if (!(dt > 0 && dt < 0.1)) { pred = null; }
      else if (pred) {
        for (const k of keys) {
          const w = MAPPINGS[k](raw), wn = norm(w);
          if (wn > 1e-12) pred[k] = rotate(pred[k], w.map(x => x / wn), -wn * dt);
        }
        // Only real turning counts toward the verdict's evidence: gyro noise
        // at rest integrates to tens of degrees over a minute, and must not.
        const turn = norm(MAPPINGS.xyz(raw)) * dt;
        if (turn > 0.1 * RAD) seen += turn;
        n++;
      }
      if (pred && n >= window) {
        if (quiet) for (const k of keys) err[k] += (a[0] - pred[k][0]) ** 2 + (a[1] - pred[k][1]) ** 2 + (a[2] - pred[k][2]) ** 2;
        pred = null;
      }
      if (!pred && quiet) { pred = Object.fromEntries(keys.map(k => [k, a.slice()])); n = 0; }
      if (seen > minTurnDeg * RAD) {
        const ks = keys.slice().sort((p, q) => err[p] - err[q]);
        if (err[ks[0]] < ratio * err[ks[1]]) verdict = ks[0];
      }
      return verdict;
    },
    verdict: () => verdict,
    errors: () => ({ ...err, seenDeg: seen / RAD }),
  };
}
