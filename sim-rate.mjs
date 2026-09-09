// Fixed simulation RATE, shared by main.js and main-amr.js.
//
// THE PROBLEM. Both pages advanced a fixed STEPS_PER_FRAME=64 per
// requestAnimationFrame, so the physics rate was `64 x fps` -- and both terms
// are properties of the hardware. Measured on one machine, same page, same
// config (W=256, levels=3):
//
//     desktop index-amr  (nvidia)    1465 steps/s   22.9 fps
//     desktop index.html (nvidia)    3750 steps/s   59.0 fps
//     phone   index-amr  (img-tec)    212 steps/s    3.3 fps
//
// -- a 7x spread between desktop and phone and 2.6x between the two desktop
// pages. There was no "default speed": there was whatever the device managed.
//
// THE UNIT. Pesavento & Wang measure time in a/u_t (Fig. 3's axis is
// literally `t [a/u_t]`; Fig. 2(d)'s frames are 4a/u_t apart), so that is what
// the rate is expressed in here rather than steps or frames. One a/u_t is
// A/U_T lattice steps -- 320 at the shipped defaults -- and callers pass that
// in, so the rate means the same thing at any resolution, card size or
// terminal velocity. The three numbers above are 4.58, 11.72 and 0.66 a/u_t/s.
//
// IS THERE A PHYSICALLY CORRECT RATE? No -- only a physically correct unit.
// a/u_t is scale-free, so a wall-clock rate needs a physical realisation, and
// the shipped dimensionless numbers (Re=1100, I*=0.17, e=0.125, so
// rho_b/rho_f=2.678) then fix the length scale: in water that is a 7.2 mm
// chord and real time is 42.4 a/u_t/s; in air a 43.8 mm chord and 17.2
// a/u_t/s. A 2.5x spread over equally valid realisations, and both far above
// what we run at, so the default below is slow motion by choice, not physics.
//
// HOW THE CLAMP KEEPS SLOW DEVICES AT FULL SPEED. maxStepsPerFrame is a
// CEILING, not a target. A device that cannot reach the rate asks for more
// steps than the ceiling every frame, gets the ceiling, and therefore behaves
// exactly as it did before this existed. The phone above manages 212 steps/s
// against a 733 steps/s target -- 3.5x short -- so it is untouched, with
// enough margin to stay untouched if it got three times faster. A faster
// desktop, conversely, does the same 733 steps/s and simply uses less of its
// GPU, instead of spending the headroom making the card fall faster.

// Half the desktop AMR page's measured 4.58 a/u_t/s, rounded onto the Sim
// speed slider's 0.05 step (2.29 -> 2.30, a 0.4% difference). Subjective --
// see above; there is no physical rate to match, and every plausible
// real-time rate is several times faster than this.
export const DEFAULT_TU_PER_SEC = 2.3;

// A frame gap longer than this is treated as this long. Belt-and-braces
// alongside the accumulator cap below: a backgrounded tab or a long stall
// must not bank simulation time and then discharge it as one huge burst.
const MAX_FRAME_DT_MS = 250;

export function parseSimRate(params, fallback = DEFAULT_TU_PER_SEC) {
  const v = parseFloat(params.get('simRate'));
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

// `maxStepsPerFrame` must be EVEN and is the old STEPS_PER_FRAME. Every count
// returned is even too: both pages flip a ping-pong buffer (`useB`) once per
// step, and main-amr.js's snapshot code documents the invariant that useB is
// back to its initial value at a frame boundary -- an odd count would leave
// f_b authoritative and silently corrupt save/load.
export function createSimPacer({ maxStepsPerFrame, tuPerSec = DEFAULT_TU_PER_SEC }) {
  const maxSteps = Math.max(2, maxStepsPerFrame - (maxStepsPerFrame & 1));
  let rate = tuPerSec;
  let last = null;
  let acc = 0;   // steps owed, carried fractionally across frames

  return {
    get rate() { return rate; },
    setRate(v) { if (Number.isFinite(v) && v > 0) rate = v; },
    // Call on reset/resume so a pause does not bank time.
    reset() { last = null; acc = 0; },

    // `stepsPerTimeUnit` is A/U_T, read live so the rate survives a Blockage
    // or Ut drag. Returns an even step count in [0, maxStepsPerFrame].
    stepsForFrame(nowMs, stepsPerTimeUnit) {
      if (last === null) { last = nowMs; return 0; }
      const dt = Math.min(nowMs - last, MAX_FRAME_DT_MS);
      last = nowMs;
      if (!(stepsPerTimeUnit > 0)) return 0;

      acc += (dt / 1000) * rate * stepsPerTimeUnit;
      // Cap the backlog at one frame's ceiling. This is what makes a device
      // that cannot keep up simply run flat out rather than accumulating an
      // ever-growing debt it would try to repay in bursts.
      if (acc > maxSteps) acc = maxSteps;

      let n = Math.floor(acc);
      n -= n & 1;                 // parity -- see the header
      acc -= n;                   // keep the remainder, so the average rate is exact
      return n;
    },
  };
}
