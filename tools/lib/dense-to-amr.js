// dense -> AMR field injection: overwrite an AMR snapshot's field data with
// the dense solver's state, so both solvers can be run forward from THE SAME
// initial condition and diffed over a short horizon.
//
// WHY THIS EXISTS
//
// tools/validate-amr-vs-dense.js compares single instantaneous snapshots of
// two INDEPENDENTLY-timed runs. plans/AMR-vs-dense-validation.md's Finding #3
// records why that stops being meaningful the moment the flow sheds: at
// Re=100 both solvers were individually correct (all Cd inside the
// literature band) yet `uy` relL2 came out at 1.63, because two correct
// periodic solutions sampled at uncorrelated phase disagree enormously.
// Nothing about that number tells you whether the AMR machinery is right.
//
// Starting both solvers from ONE state removes phase from the comparison
// entirely. What's left is divergence, and its SHAPE is the diagnostic:
//
//   - nonzero at step 0        -> injection/interpolation error, not dynamics
//   - concentrated at tile edges (anomalies clustering mod 2*RB)
//                              -> ghost/seam bug
//   - at a half-tile offset    -> the tile-registration bug class this
//                                 project has already hit twice (be16b4b,
//                                 73162a4)
//   - growing in refined interiors, not at their edges
//                              -> per-level tau
//   - smooth, diffuse, growing from zero
//                              -> ordinary truncation error, i.e. correct
//
// That is the automated form of the by-hand analysis already recorded in
// plans/AMR-vs-dense-validation.md ("per-column max-abs-diff and a histogram
// of anomaly locations mod tile size"), which had to be redone manually every
// time something looked off.
//
// WHAT KEEPS THIS SMALL AND CHECKABLE
//
// 1. It does NOT synthesize AMR topology. It takes a REAL AMR snapshot and
//    overwrites only the field arrays, leaving blockSlot/slotToBlock/
//    parentSlot/originX/originY exactly as the solver produced them. The
//    entire "did I build a valid quadtree" bug class therefore cannot occur
//    -- the hierarchy comes from the solver.
//
// 2. It walks the SAME quadtree recursion as
//    field-reconstruct.js's reconstructAMRToResolution (paintQuad), so the
//    two are inverses by construction rather than by coincidence.
//
// 3. It is therefore validated by ROUND-TRIP against that already-fixture-
//    tested reconstructor (tools/test-dense-to-amr.js), not by eye: inject a
//    known dense field, reconstruct it back, and require the original. New
//    code checked against existing checked code by an identity.
//
// The output is a snapshot object in exactly the shape main-amr.js's
// debugSnapshotLoad already accepts, so nothing new is needed on the page.

const { b64ToFloat32, unshiftField, rhoFromF } = require('./field-reconstruct');

const BLOCK = 8; // matches shaders/amr_step.wgsl's block8 cellIndex

const EX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const EY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

// Must stay identical to common_lattice.wgsl's feqD2Q9 -- the injected f is
// rebuilt as feq + rescale*fneq, so a mismatch here would show up as a
// uniform offset the moment the solver takes its first step.
function feq(rho, ux, uy, i) {
  const eu = EX[i] * ux + EY[i] * uy;
  return WT[i] * rho * (1 + eu * 3 + eu * eu * 4.5 - (ux * ux + uy * uy) * 1.5);
}

function float32ToB64(arr) {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64');
}

// Same relation as card-params.mjs's tauAtLevel and main-amr.js's own, kept
// here as a local copy because this file is CommonJS and the pages' module is
// ESM; the value is pinned against it in tools/test-dense-to-amr.js.
function tauAtLevel(tau0, m) {
  let t = tau0;
  for (let i = 0; i < m; i++) t = 2 * t - 0.5;
  return t;
}

// Composed Dupuis-Chopard fneq rescale from level `src` down to level `dst`
// (dst <= src, i.e. fine -> coarse).
//
// One hop, from shaders/amr_average_f2c.wgsl: 2 * tau_coarse / tau_fine.
// Composing k hops telescopes, since each hop's numerator tau cancels the
// next hop's denominator:
//   (2*t_L/t_{L+1}) * (2*t_{L+1}/t_{L+2}) * ... = 2^k * t_dst / t_src
// Averaging is linear and the rescale is a scalar, so averaging the whole
// footprint once and applying the composed factor once is EXACTLY equal to
// the solver's repeated average-then-rescale-per-hop -- this is an identity,
// not an approximation.
function fneqRescale(tau0, src, dst) {
  return 2 ** (src - dst) * tauAtLevel(tau0, dst) / tauAtLevel(tau0, src);
}

// ---------------------------------------------------------------------
// Dense side: decode a main-cylinder.js/main.js debugSnapshotSave payload
// into window space, keeping f (which loadDenseFields discards -- it only
// needs macroscopic fields) because the non-equilibrium part is exactly what
// must survive injection. Dropping it and injecting f=feq would throw away
// the stress/shear state and make the first steps of every comparison a
// re-development transient rather than a measurement.
// ---------------------------------------------------------------------
function loadDenseStateWithF(snapshot) {
  const { W, H } = snapshot;
  const NCELLS = W * H;
  const offX = snapshot.cardState[22], offY = snapshot.cardState[23];
  const layout = snapshot.layout || 'flat';

  const velRaw = b64ToFloat32(snapshot.velB64, NCELLS * 2);
  const vel = unshiftField(velRaw, W, H, 2, offX, offY, layout);
  const ux = new Float32Array(NCELLS), uy = new Float32Array(NCELLS);
  for (let c = 0; c < NCELLS; c++) { ux[c] = vel[c * 2]; uy[c] = vel[c * 2 + 1]; }

  const fRaw = b64ToFloat32(snapshot.fB64, NCELLS * 9);
  const f = new Float32Array(NCELLS * 9);
  for (let i = 0; i < 9; i++) {
    const plane = fRaw.subarray(i * NCELLS, (i + 1) * NCELLS);
    f.set(unshiftField(plane, W, H, 1, offX, offY, layout), i * NCELLS);
  }
  const rho = rhoFromF(f, W, H);

  // fneq = f - feq(rho,u), the part that carries stress. Precomputed once for
  // the whole grid rather than per sampled footprint -- every dense cell is
  // read at least once, most exactly once.
  const fneq = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    for (let i = 0; i < 9; i++) {
      fneq[i * NCELLS + c] = f[i * NCELLS + c] - feq(rho[c], ux[c], uy[c], i);
    }
  }
  return { W, H, ux, uy, rho, fneq };
}

// ---------------------------------------------------------------------
// Quadtree walk. Deliberately mirrors reconstructAMRToResolution's paintQuad
// so both sides agree on every slot's physical origin. Unlike paintQuad this
// visits EVERY active slot, including ones that have active children: the
// solver still runs a parent's own substep (ghost-fill / 2:1 bookkeeping)
// under an active child, so the parent's cells must hold real data too, not
// just the quadrants where it happens to be the finest level.
// ---------------------------------------------------------------------
function walkActiveSlots(snapshot, cb) {
  const { W: W0, H: H0, numLevels, pools } = snapshot;
  const RB = pools[1].RB;
  const NBX0 = pools[1].NBX, NBY0 = pools[1].NBY;

  function recurse(level, slot, bx, by, originX, originY) {
    cb(level, slot, bx, by, originX, originY);
    const childLevel = level + 1;
    if (childLevel >= numLevels) return;
    const child = pools[childLevel];
    const nbxChild = NBX0 * (1 << level);
    if (child.NBX !== nbxChild) {
      throw new Error(`dense-to-amr: level ${childLevel} NBX=${child.NBX}, expected ${nbxChild} from quadtree doubling`);
    }
    const dxLparent = 1 / (1 << level);
    for (let qy = 0; qy < 2; qy++) {
      for (let qx = 0; qx < 2; qx++) {
        const childBX = bx * 2 + qx, childBY = by * 2 + qy;
        const childSlot = child.blockSlot[childBY * nbxChild + childBX];
        if (childSlot === -1) continue;
        recurse(childLevel, childSlot, childBX, childBY,
          originX + qx * RB * dxLparent, originY + qy * RB * dxLparent);
      }
    }
  }

  const L1 = pools[1];
  for (let by = 0; by < NBY0; by++) {
    for (let bx = 0; bx < NBX0; bx++) {
      const slot1 = L1.blockSlot[by * NBX0 + bx];
      if (slot1 !== -1) recurse(1, slot1, bx, by, bx * RB, by * RB);
    }
  }
  void W0; void H0;
}

// ---------------------------------------------------------------------
// The injector.
// ---------------------------------------------------------------------
//
// denseSnapshot : main-cylinder.js/main.js debugSnapshotSave payload, at the
//                 AMR's own finest resolution (W0 * 2^(numLevels-1)).
// amrSnapshot   : main-amr.js/main-cylinder-amr.js debugSnapshotSave payload,
//                 supplying the refinement TOPOLOGY (and everything else that
//                 isn't field data).
//
// Returns a new snapshot object; the input is not mutated.
function injectDenseIntoAMRSnapshot({ denseSnapshot, amrSnapshot, tau0 = null, fillGhosts = true }) {
  const { W: W0, H: H0, numLevels, pools } = amrSnapshot;
  if (!pools || numLevels == null) {
    throw new Error('injectDenseIntoAMRSnapshot: amrSnapshot missing pools[]/numLevels -- not an AMR debugSnapshotSave payload');
  }
  const finestLevel = numLevels - 1;
  const mult = 1 << finestLevel;            // target cells per L0 cell per axis
  const targetW = W0 * mult;

  if (denseSnapshot.W !== targetW || denseSnapshot.H !== H0 * mult) {
    throw new Error(
      `injectDenseIntoAMRSnapshot: dense snapshot is ${denseSnapshot.W}x${denseSnapshot.H}, but this AMR snapshot's finest ` +
      `resolution is ${W0}*2^${finestLevel} = ${targetW} (x${H0 * mult}). Run the dense reference at res=${Math.log2(targetW)}, ` +
      `or see tools/lib/amr-resolution-mapping.js which exists to derive a consistent pair.`);
  }

  // L0's own tau. cardState[19] per shaders/common_geometry.wgsl's CardState.
  const TAU0 = tau0 != null ? tau0 : amrSnapshot.cardState[19];
  if (!(TAU0 > 0.5)) {
    throw new Error(`injectDenseIntoAMRSnapshot: L0 tau=${TAU0} from cardState[19] is not above the BGK floor 0.5 -- wrong snapshot or wrong CardState layout?`);
  }

  const dense = loadDenseStateWithF(denseSnapshot);
  const DW = dense.W, DH = dense.H;

  // Moving-window pan, in target-grid units -- same convention
  // reconstructAMRToResolution uses when it writes, so this reads back from
  // the same place it would have written.
  const offX = amrSnapshot.cardState[22], offY = amrSnapshot.cardState[23];
  const offXTarget = offX * mult, offYTarget = offY * mult;

  // Average one axis-aligned square of the dense target grid. rho is an
  // arithmetic mean (mass-conservative) and velocity is mass-weighted
  // (momentum-conservative), exactly matching amr_average_f2c.wgsl's own
  // choice; fneq is an arithmetic mean, which is what makes the composed
  // rescale in fneqRescale an identity rather than an approximation.
  const acc = new Float32Array(9);
  function sampleSquare(bufX, bufY, extentL0) {
    const x0 = Math.round(bufX * mult - offXTarget);
    const y0 = Math.round(bufY * mult - offYTarget);
    const size = Math.round(extentL0 * mult);
    let rhoSum = 0, mux = 0, muy = 0, n = 0;
    acc.fill(0);
    for (let dy = 0; dy < size; dy++) {
      const wy = ((y0 + dy) % DH + DH) % DH;
      for (let dx = 0; dx < size; dx++) {
        const wx = ((x0 + dx) % DW + DW) % DW;
        const c = wy * DW + wx;
        const r = dense.rho[c];
        rhoSum += r;
        mux += r * dense.ux[c];
        muy += r * dense.uy[c];
        for (let i = 0; i < 9; i++) acc[i] += dense.fneq[i * DW * DH + c];
        n++;
      }
    }
    const rho = rhoSum / n;
    // rhoSum is a sum of LBM densities, all ~1.0, so it cannot be zero in any
    // physical state; guard anyway so a corrupt snapshot fails loudly here
    // rather than seeding NaNs into the solver.
    if (!(rhoSum > 0)) throw new Error(`injectDenseIntoAMRSnapshot: non-positive density sum ${rhoSum} at buffer (${bufX},${bufY}) -- corrupt dense snapshot?`);
    for (let i = 0; i < 9; i++) acc[i] /= n;
    return { rho, ux: mux / rhoSum, uy: muy / rhoSum };
  }

  // ── L0 ──────────────────────────────────────────────────────────────────
  // Filled everywhere, including under active tiles: the solver steps the
  // whole dense L0 grid every macro-step regardless of what covers it.
  const NCELLS0 = W0 * H0;
  const fL0 = new Float32Array(NCELLS0 * 9);
  const velL0 = new Float32Array(NCELLS0 * 2);
  const rescale0 = fneqRescale(TAU0, finestLevel, 0);
  for (let cy = 0; cy < H0; cy++) {
    for (let cx = 0; cx < W0; cx++) {
      const s = sampleSquare(cx, cy, 1);
      // block8: L0 storage is block-major, NOT row-major (see
      // shaders/amr_step.wgsl's cellIndex) -- writing row-major here would
      // scramble the field in a way that still decodes without error.
      const nbx = W0 / BLOCK;
      const bID = Math.floor(cy / BLOCK) * nbx + Math.floor(cx / BLOCK);
      const cell = bID * (BLOCK * BLOCK) + (cy % BLOCK) * BLOCK + (cx % BLOCK);
      velL0[cell * 2] = s.ux; velL0[cell * 2 + 1] = s.uy;
      for (let i = 0; i < 9; i++) {
        fL0[i * NCELLS0 + cell] = feq(s.rho, s.ux, s.uy, i) + rescale0 * acc[i];
      }
    }
  }

  // ── Pool levels ─────────────────────────────────────────────────────────
  // Start from the template's own arrays so INACTIVE slots keep whatever the
  // solver last left in them (they're skipped by every shader via
  // slotToBlock[slot] < 0, so their contents are irrelevant -- but preserving
  // them keeps the output a faithful edit of the input rather than a partly
  // synthesized one).
  const RB = pools[1].RB, GHOST = pools[1].GHOST, FB = pools[1].FB;
  const outPools = [undefined];
  const poolF = [null], poolVel = [null];
  for (let m = 1; m < numLevels; m++) {
    const p = pools[m];
    const cells = p.MAX_FINE_BLOCKS * FB * FB;
    poolF.push(Float32Array.from(b64ToFloat32(p.fB64, cells * 9)));
    poolVel.push(Float32Array.from(b64ToFloat32(p.velB64, cells * 2)));
  }

  // Ghost cells are filled too. The solver refills them from the parent and
  // from same-level neighbours at the start of each macro-step, so they would
  // be corrected anyway -- but only AFTER the first substep has already read
  // them. Seeding them from the dense field costs nothing and removes a
  // one-step artifact at every tile border, which is exactly the region this
  // whole comparison is trying to measure.
  const lo = fillGhosts ? 0 : GHOST;
  const hi = fillGhosts ? FB : GHOST + 2 * RB;

  const visited = [];
  for (let m = 0; m < numLevels; m++) visited.push(new Set());

  walkActiveSlots(amrSnapshot, (level, slot, bx, by, originX, originY) => {
    visited[level].add(slot);
    const dxL = 1 / (1 << level);
    const rescale = fneqRescale(TAU0, finestLevel, level);
    const f = poolF[level], vel = poolVel[level];
    const planeStride = pools[level].MAX_FINE_BLOCKS * FB * FB;
    for (let fy = lo; fy < hi; fy++) {
      for (let fx = lo; fx < hi; fx++) {
        const sx = originX + (fx - GHOST) * dxL;
        const sy = originY + (fy - GHOST) * dxL;
        const s = sampleSquare(sx, sy, dxL);
        const cellIdx = slot * (FB * FB) + fy * FB + fx;
        vel[cellIdx * 2] = s.ux; vel[cellIdx * 2 + 1] = s.uy;
        for (let i = 0; i < 9; i++) {
          f[i * planeStride + cellIdx] = feq(s.rho, s.ux, s.uy, i) + rescale * acc[i];
        }
      }
    }
  });

  // Every slot marked active in blockSlot must have been reachable from the
  // L1 roots. An unreachable-but-active slot is an orphan -- a real AMR
  // structural bug (a leaked allocation, or a parent coarsened without its
  // children) -- and injecting into a snapshot containing one would silently
  // leave that slot holding stale data while the solver kept stepping it.
  // Cheap to check here, so check it.
  for (let m = 1; m < numLevels; m++) {
    const p = pools[m];
    const active = [];
    for (let slot = 0; slot < p.MAX_FINE_BLOCKS; slot++) {
      if (p.slotToBlock[slot] !== -1) active.push(slot);
    }
    const orphans = active.filter(s => !visited[m].has(s));
    if (orphans.length) {
      throw new Error(
        `injectDenseIntoAMRSnapshot: level ${m} has ${orphans.length} active slot(s) not reachable from the L1 roots ` +
        `(e.g. slot ${orphans[0]} -> block ${p.slotToBlock[orphans[0]]}). That is an orphaned allocation in the source ` +
        `snapshot, not an injection failure -- the AMR tree and its slot table disagree.`);
    }
  }

  for (let m = 1; m < numLevels; m++) {
    outPools.push({ ...pools[m], fB64: float32ToB64(poolF[m]), velB64: float32ToB64(poolVel[m]) });
  }

  return {
    ...amrSnapshot,
    fB64: float32ToB64(fL0),
    velB64: float32ToB64(velL0),
    pools: outPools,
    // Provenance, so a snapshot that has been injected into is identifiable
    // after the fact rather than looking like an ordinary captured run.
    injectedFrom: {
      denseW: denseSnapshot.W, denseH: denseSnapshot.H,
      denseStep: denseSnapshot.step, amrStepReplaced: amrSnapshot.step,
      tau0: TAU0, fillGhosts,
    },
  };
}

module.exports = {
  injectDenseIntoAMRSnapshot, loadDenseStateWithF, walkActiveSlots,
  fneqRescale, tauAtLevel, feq, float32ToB64,
};
