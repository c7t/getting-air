#!/usr/bin/env node
// Numerically diffs two AMR-dev-build snapshots (see main-amr.js's
// debugSnapshotSave / tools/amr-snapshot.js) instead of comparing
// screenshots -- this is the regression check plans/AMR.md's milestones
// rely on ("does the AMR build's output match a reference to tolerance").
// Pure Node, no browser/CDP needed once the snapshots are on disk.
//
// Milestone 0/1 scope: both snapshots must be the same W x H (e.g. two
// snapshots of the same run, or main-amr.js vs. a from-scratch reference at
// identical resolution). Comparing a fine sub-region against a coarser
// uniform-grid reference (needed from Milestone 2 onward, once resolutions
// actually differ) needs windowing/subsampling this doesn't do yet -- add
// that when M2 needs it, don't build it speculatively now.
//
// Usage: node tools/amr-diff.js <a.json> <b.json>

const fs = require('fs');

function b64ToFloat32(b64, floatCount) {
  const binary = Buffer.from(b64, 'base64');
  return new Float32Array(binary.buffer, binary.byteOffset, floatCount);
}

// Buffer storage is a circular window (main-amr.js's moving-window
// off_x/off_y, see .plan-albc-moving-window.md): window coordinate wx maps
// to buffer coordinate bx = (wx + off_x) % W. Unshift so field comparisons
// are in physically meaningful window coordinates, not raw (and
// step-dependent) buffer layout.
function unshiftField(buf, W, H, comps, offX, offY) {
  const out = new Float32Array(W * H * comps);
  for (let wy = 0; wy < H; wy++) {
    const by = (wy + offY) % H;
    for (let wx = 0; wx < W; wx++) {
      const bx = (wx + offX) % W;
      const srcCell = by * W + bx;
      const dstCell = wy * W + wx;
      for (let c = 0; c < comps; c++) out[dstCell * comps + c] = buf[srcCell * comps + c];
    }
  }
  return out;
}

function rhoFromF(fUnshifted, W, H) {
  const NCELLS = W * H;
  const rho = new Float32Array(NCELLS);
  for (let cell = 0; cell < NCELLS; cell++) {
    let s = 0;
    for (let i = 0; i < 9; i++) s += fUnshifted[i * NCELLS + cell];
    rho[cell] = s;
  }
  return rho;
}

// Same periodic central-difference formula as shaders/amr_render.wgsl's
// get_ux/get_uy + fs_main, applied to already-unshifted (window-coordinate)
// ux/uy fields, so the sign/scale of any reported vorticity diff matches
// what the two runs actually rendered.
function vorticityField(ux, uy, W, H) {
  const omega = new Float32Array(W * H);
  const wrap = (v, n) => ((v % n) + n) % n;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const uyxp1 = uy[(y * W) + wrap(x + 1, W)];
      const uyxm1 = uy[(y * W) + wrap(x - 1, W)];
      const uxyp1 = ux[(wrap(y + 1, H) * W) + x];
      const uxym1 = ux[(wrap(y - 1, H) * W) + x];
      omega[y * W + x] = (uyxp1 - uyxm1) * 0.5 - (uxyp1 - uxym1) * 0.5;
    }
  }
  return omega;
}

function diffStats(a, b, label) {
  let maxAbsDiff = 0, sumSqDiff = 0, sumSqRef = 0, maxAbsRef = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    maxAbsDiff = Math.max(maxAbsDiff, d);
    maxAbsRef = Math.max(maxAbsRef, Math.abs(a[i]));
    sumSqDiff += d * d;
    sumSqRef += a[i] * a[i];
  }
  const relL2 = sumSqRef > 0 ? Math.sqrt(sumSqDiff / sumSqRef) : (sumSqDiff > 0 ? Infinity : 0);
  console.log(`${label}: maxAbsDiff=${maxAbsDiff.toExponential(4)}  maxAbsRef=${maxAbsRef.toExponential(4)}  relL2=${relL2.toExponential(4)}`);
  return { maxAbsDiff, maxAbsRef, relL2 };
}

function loadFields(snapshot) {
  const { W, H } = snapshot;
  const NCELLS = W * H;
  const offX = snapshot.cardState[22], offY = snapshot.cardState[23];

  const velRaw = b64ToFloat32(snapshot.velB64, NCELLS * 2);
  const vel = unshiftField(velRaw, W, H, 2, offX, offY);
  const ux = new Float32Array(NCELLS), uy = new Float32Array(NCELLS);
  for (let c = 0; c < NCELLS; c++) { ux[c] = vel[c * 2]; uy[c] = vel[c * 2 + 1]; }

  const fRaw = b64ToFloat32(snapshot.fB64, NCELLS * 9);
  // f is laid out i*(W*H) + cell (see amr_step.wgsl); unshift each of the 9
  // direction planes independently, then treat the whole thing as a single
  // 9-component field so rhoFromF's i*(W*H)+cell indexing still holds.
  const fUnshifted = new Float32Array(NCELLS * 9);
  for (let i = 0; i < 9; i++) {
    const plane = fRaw.subarray(i * NCELLS, (i + 1) * NCELLS);
    const planeVec = unshiftField(plane, W, H, 1, offX, offY);
    fUnshifted.set(planeVec, i * NCELLS);
  }
  const rho = rhoFromF(fUnshifted, W, H);
  const omega = vorticityField(ux, uy, W, H);

  return { W, H, step: snapshot.step, ux, uy, rho, omega };
}

function main() {
  const [aFile, bFile] = process.argv.slice(2);
  if (!aFile || !bFile) {
    console.error('usage: amr-diff.js <a.json> <b.json>');
    process.exit(1);
  }
  const a = JSON.parse(fs.readFileSync(aFile, 'utf8'));
  const b = JSON.parse(fs.readFileSync(bFile, 'utf8'));
  if (a.W !== b.W || a.H !== b.H) {
    console.error(`resolution mismatch: ${aFile} is ${a.W}x${a.H}, ${bFile} is ${b.W}x${b.H} -- not yet supported (see file header)`);
    process.exit(1);
  }
  if (a.step !== b.step) {
    console.warn(`WARNING: comparing different step counts (${a.step} vs ${b.step}) -- diffs below reflect that, not just numerical divergence`);
  }

  const fa = loadFields(a), fb = loadFields(b);
  console.log(`Comparing ${aFile} (step ${fa.step}) vs ${bFile} (step ${fb.step}), ${fa.W}x${fa.H}`);
  const results = {
    ux: diffStats(fa.ux, fb.ux, 'ux   '),
    uy: diffStats(fa.uy, fb.uy, 'uy   '),
    rho: diffStats(fa.rho, fb.rho, 'rho  '),
    omega: diffStats(fa.omega, fb.omega, 'omega'),
  };
  const allZero = Object.values(results).every(r => r.maxAbsDiff === 0);
  console.log(allZero ? 'IDENTICAL' : 'DIFFERS');
  process.exit(0);
}
main();
