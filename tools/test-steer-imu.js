#!/usr/bin/env node
// steer-imu.mjs against a SIMULATED PHONE whose true orientation is known.
//
// The phone is held like a steering wheel: landscape, reclined, turned
// through a scripted wheel angle, while the holder also turns on the spot
// (yaw) and bobs the pitch -- neither of which is steering and neither of
// which may move the output. Its gyro carries a large constant bias and
// noise; its accelerometer carries noise and violent shakes. Rates are the
// ~60 Hz Chrome delivers devicemotion at.
//
// Every accuracy claim has a CONTROL that fails it, so the test cannot pass
// by measuring nothing: gyro-only integration must drift by hundreds of
// degrees over the same run, and accelerometer-only must be thrown off by the
// shakes. And the bias integral is shown to MATTER by turning it off.

import { createSteerFilter, steerInput, wrapAngle, motionSample, createAxisDetector, MAPPINGS } from '../steer-imu.mjs';

let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); if (!ok) failures++; };

// ── tiny 3x3 rotation algebra (row-major arrays of rows) ───────────────────
const mul = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const T = (A) => [0, 1, 2].map(i => [0, 1, 2].map(j => A[j][i]));
const mv = (A, v) => A.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const Rx = (t) => [[1, 0, 0], [0, Math.cos(t), -Math.sin(t)], [0, Math.sin(t), Math.cos(t)]];
const Ry = (t) => [[Math.cos(t), 0, Math.sin(t)], [0, 1, 0], [-Math.sin(t), 0, Math.cos(t)]];
const Rz = (t) => [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];
// Body rate from two consecutive device->world rotations: M0^T M1 = exp([w dt]x).
function bodyRate(M0, M1, dt) {
  const D = mul(T(M0), M1);
  const ang = Math.acos(Math.max(-1, Math.min(1, (D[0][0] + D[1][1] + D[2][2] - 1) / 2)));
  const s = ang < 1e-12 ? 1 : ang / (2 * Math.sin(ang));
  return [(D[2][1] - D[1][2]) * s / dt, (D[0][2] - D[2][0]) * s / dt, (D[1][0] - D[0][1]) * s / dt];
}

// Deterministic noise (the test must be reproducible).
let seed = 12345;
const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

const G = 9.80665, UP = [0, 1, 0];   // world: +Y is up
const DEG = Math.PI / 180;

// The true wheel angle: slow sweeps, a held turn, and fast flicks.
const wheel = (t) => 30 * DEG * Math.sin(2 * Math.PI * t / 7)
  + 15 * DEG * Math.sin(2 * Math.PI * t / 1.3)
  + (t % 40 > 30 ? 25 * DEG : 0) * Math.min(1, (t % 40 - 30) * 4);
// Device -> world: roll by the wheel (CW seen from the front = -z), landscape
// (+90 deg), reclined ~40 deg with some bob, and the holder turning around.
const pose = (t) => mul(Ry(0.3 * t), mul(Rx(-(40 + 10 * Math.sin(t * 0.9)) * DEG), Rz(-(wheel(t) + 90 * DEG))));

// browser: how the simulated browser packs rotationRate (see MAPPINGS). When
// set, samples go through the page's own path -- raw event -> axis detector
// -> mapping -> filter -- starting from the WRONG guess for a spec browser.
function run({ seconds, filterOpts = {}, gyroBias = [0.02, -0.015, 0.02], signFlip = false, shakes = true, browser = null }) {
  const dt = 1 / 60;
  const f = createSteerFilter(filterOpts);
  // Scored against the ABSOLUTE angle, which the simulation knows: the
  // landscape hold adds 90 deg and a negated accelerometer another 180. (An
  // early version scored against a neutral captured 1 s in, which bakes the
  // not-yet-learned bias into the reference and made the bias integral look
  // HARMFUL -- the no-integral run's standing error cancelled against itself.)
  const offset = 90 * DEG + (signFlip ? Math.PI : 0);
  const det = browser ? createAxisDetector() : null;
  let mapping = 'xyz', verdictAt = null;
  let gyroOnly = 0, prevM = pose(0);
  const errs = [], errsAcc = [];
  let gyroDrift = 0;
  for (let k = 1; k <= seconds * 60; k++) {
    const t = k * dt, M = pose(t);
    const w = bodyRate(prevM, M, dt); prevM = M;
    const omega = w.map((x, i) => x + gyroBias[i] + 0.003 * gauss());
    // Shakes: 0.4 s bursts of 6 m/s^2 sideways, twice every 20 s.
    const shaking = shakes && ((t % 20 > 5 && t % 20 < 5.4) || (t % 20 > 13 && t % 20 < 13.4));
    const lin = shaking ? [6 * Math.sin(40 * t), 3 * Math.cos(33 * t), 0] : [0, 0, 0];
    let accel = mv(T(M), UP.map((x, i) => G * x + lin[i])).map(x => x + 0.05 * gauss());
    if (signFlip) accel = accel.map(x => -x);
    if (det) {
      const d = 180 / Math.PI;
      const raw = browser === 'xyz' ? { alpha: omega[0] * d, beta: omega[1] * d, gamma: omega[2] * d }
                                    : { alpha: omega[2] * d, beta: omega[0] * d, gamma: omega[1] * d };
      const v = det.push(raw, accel, dt);
      if (v && verdictAt === null) { verdictAt = t; mapping = v; }
      f.update(MAPPINGS[mapping](raw), accel, dt);
    } else f.update(omega, accel, dt);
    // Gyro-only control: the same device-z rate, bias and all, integrated
    // with no reference -- minus the TRUE z rate, so what is left is purely
    // the error it accumulates (unwrapped: it must be seen to grow).
    gyroOnly += (omega[2] - w[2]) * dt;
    gyroDrift = Math.abs(gyroOnly) / DEG;
    // Accel-only control: the angle of the raw measured up vector.
    const accRoll = Math.atan2(-accel[0], accel[1]);
    if (k > (det ? 60 * 20 : 120)) {   // 2 s to converge from the first sample (20 s with detection)
      const truth = wheel(t) + offset;
      errs.push(Math.abs(wrapAngle(f.roll() - truth)));
      errsAcc.push({ shaking, e: Math.abs(wrapAngle(accRoll - truth)) });
    }
  }
  const rms = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length) / DEG;
  const max = (a) => Math.max(...a) / DEG;
  return { f, rms: rms(errs), max: max(errs), last: max(errs.slice(-60 * 60)),
    gyroDrift, verdict: det && det.verdict(), verdictAt, accShakeMax: max(errsAcc.filter(x => x.shaking).map(x => x.e)) };
}

// 1. The real thing: 10 minutes, biased gyro, shakes.
{
  const r = run({ seconds: 600 });
  check(r.rms < 0.5, `fused wheel angle RMS error ${r.rms.toFixed(3)} deg < 0.5 over 10 min`);
  check(r.max < 2.0, `fused worst error ${r.max.toFixed(2)} deg < 2.0 (shakes included)`);
  check(r.last < 1.0, `no drift: worst error in the LAST minute ${r.last.toFixed(3)} deg < 1.0`);
  check(r.gyroDrift > 100, `control: gyro-only integration drifted ${r.gyroDrift.toFixed(0)} deg by the end (> 100)`);
  check(r.accShakeMax > 10, `control: accelerometer-only is thrown ${r.accShakeMax.toFixed(1)} deg by a shake (> 10)`);
  const b = r.f.bias();
  check(Math.abs(b[2] - 0.02) < 0.004, `bias integral found the z bias: ${b[2].toFixed(4)} rad/s (true 0.0200)`);
}

// 2. The integral term is load-bearing: without it the steady error is ~b/kP.
{
  const withI = run({ seconds: 120, shakes: false, gyroBias: [0, 0, 0.05] });
  const noI = run({ seconds: 120, shakes: false, gyroBias: [0, 0, 0.05], filterOpts: { kI: 0 } });
  check(noI.last > 1.2 && withI.last < 0.5 * noI.last,
    `bias integral matters: last-minute worst ${withI.last.toFixed(3)} deg with it vs ${noI.last.toFixed(3)} deg without`);
}

// 3. Sign-agnostic in the accelerometer (iOS's historical convention).
{
  const r = run({ seconds: 120, signFlip: true });
  check(r.rms < 0.5, `negated accelerometer: RMS ${r.rms.toFixed(3)} deg < 0.5 (settles 180 deg round, as it must)`);
}

// 4. The axis mapping is DETECTED, both ways round, and a browser whose
// mapping the page guessed wrong still ends up steering correctly.
for (const browser of ['xyz', 'spec']) {
  const r = run({ seconds: 120, browser });
  check(r.verdict === browser && r.verdictAt < 10,
    `axis detector: a '${browser}' browser is identified as '${r.verdict}' after ${r.verdictAt?.toFixed(1)} s (< 10)`);
  check(r.rms < 0.5, `  ...and then steers with RMS ${r.rms.toFixed(3)} deg < 0.5`);
}

// Control: a phone at rest has told the detector nothing, so it must not
// guess -- a verdict there would be a coin flip on the noise.
{
  const det = createAxisDetector();
  for (let k = 0; k < 5 * 60 * 60; k++) {
    det.push({ alpha: 0.2 * gauss(), beta: 0.2 * gauss(), gamma: 0.2 * gauss() },
      [0.05 * gauss(), G + 0.05 * gauss(), 0.05 * gauss()], 1 / 60);
  }
  check(det.verdict() === null, `axis detector: no verdict from five minutes of a phone at rest (counted ${det.errors().seenDeg.toFixed(1)} deg as turning)`);
}

// 5. The input map and the event adapter.
check(steerInput(0) === 0 && steerInput(1.5 * DEG) === 0, 'deadband holds a centred wheel at exactly 0');
check(steerInput(45 * DEG) === 1 && steerInput(-90 * DEG) === -1, 'saturates at +-1');
check(Math.abs(steerInput(23.5 * DEG) - 0.5) < 1e-12, 'linear between deadband and max');
{
  const ev = { rotationRate: { alpha: 0, beta: 0, gamma: -30 }, accelerationIncludingGravity: { x: 0, y: 9.8, z: 0 } };
  const s = motionSample(ev);
  // The Chrome event measured in index-steer.html: a (0, 0, -30 deg/s) gyro.
  check(Math.abs(s.omega[2] + 30 * DEG) < 1e-12 && s.omega[0] === 0, "default 'xyz' mapping reads Chrome's gamma as the device z rate, in rad/s");
  check(Math.abs(motionSample(ev, 'spec').omega[1] + 30 * DEG) < 1e-12, "the 'spec' mapping reads the same event's gamma as device y");
  check(motionSample({ rotationRate: { alpha: null }, accelerationIncludingGravity: { x: null } }) === null, 'an all-null desktop event is ignored');
}

if (failures) { console.error(`\n${failures} FAILED`); process.exit(1); }
console.log('\nsteer-imu: all checks passed');
