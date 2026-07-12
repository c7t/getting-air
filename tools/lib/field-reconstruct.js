// Shared field-decoding/comparison primitives, factored out of
// tools/amr-diff.js (which used to define all of these locally) so
// tools/validate-amr-vs-dense.js can reuse the exact same snapshot-decoding
// logic for its own, harder problem: reconstructing an AMR run's quadtree
// pool data onto a UNIFORM grid at a different (finer) resolution than the
// snapshot's own W/H, so it can be diffed against a same-resolution dense
// reference. tools/amr-diff.js explicitly punted on that ("Comparing a fine
// sub-region against a coarser uniform-grid reference... needs windowing/
// subsampling this doesn't do yet") -- reconstructAMRToResolution below is
// that missing piece.
//
// Everything in the "shared with amr-diff.js" section is a pure relocation,
// byte-for-byte identical logic to what tools/amr-diff.js had inline before
// this file existed (see git history) -- no behavior change there.

const BLOCK = 8; // matches shaders/amr_step.wgsl's block8 cellIndex

function b64ToFloat32(b64, floatCount) {
  const binary = Buffer.from(b64, 'base64');
  return new Float32Array(binary.buffer, binary.byteOffset, floatCount);
}

// Raw array index for a cell at BUFFER coordinates (cx, cy), given the raw
// GPU buffer's layout. 'flat' is main.js/main-cylinder.js's plain row-major
// (cy*W+cx -- matches shaders/lbm_step.wgsl, no block sub-tiling). 'block8'
// is main-amr.js/main-cylinder-amr.js's L0 layout (see shaders/amr_step.wgsl's
// cellIndex) -- fixed 8x8 buffer-space blocks, block-major, row-major within
// a block.
function rawIndex(cx, cy, W, H, layout) {
  if (layout === 'flat') return cy * W + cx;
  if (layout === 'block8') {
    const nbx = W / BLOCK;
    const bx = Math.floor(cx / BLOCK), by = Math.floor(cy / BLOCK);
    const lx = cx % BLOCK, ly = cy % BLOCK;
    const blockID = by * nbx + bx;
    return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
  }
  throw new Error(`unknown snapshot layout '${layout}'`);
}

// Buffer storage is a circular window (main-amr.js's moving-window
// off_x/off_y, see .plan-albc-moving-window.md): window coordinate wx maps
// to buffer coordinate bx = (wx + off_x) % W. Unshift so field comparisons
// are in physically meaningful window coordinates (and a canonical flat
// row-major layout, regardless of the source snapshot's raw layout), not
// raw (and step-dependent, and layout-dependent) buffer storage.
//
// The pinned-cylinder harnesses (main-cylinder.js/main-cylinder-amr.js) never
// move the body, so off_x/off_y are always 0 there -- but this still routes
// through the same conversion rather than special-casing zero, so it stays
// correct if that ever changes and stays consistent with every other
// consumer of a snapshot's cardState[22]/[23].
function unshiftField(buf, W, H, comps, offX, offY, layout) {
  const out = new Float32Array(W * H * comps);
  for (let wy = 0; wy < H; wy++) {
    const by = (wy + offY) % H;
    for (let wx = 0; wx < W; wx++) {
      const bx = (wx + offX) % W;
      const srcCell = rawIndex(bx, by, W, H, layout);
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

// rho for a pool level's flat-per-slot layout (cell = slot*(FB*FB)+fy*FB+fx,
// per-direction plane stride i*(MAX_FINE_BLOCKS*FB*FB) -- see
// shaders/amr_step1_pool.wgsl / main-cylinder-amr.js's allocLevelPool).
// Computed once over the WHOLE pool array (not per reconstructed target
// cell): a fine cell can fan out to many target cells when the AMR's finest
// level is coarser than the comparison's target resolution, and re-summing
// 9 f-components per fan-out would be wasted, resolution-dependent work.
function rhoFromPoolF(fPoolArr, maxFineBlocks, FB) {
  const cellsPerPool = maxFineBlocks * FB * FB;
  const rho = new Float32Array(cellsPerPool);
  for (let cell = 0; cell < cellsPerPool; cell++) {
    let s = 0;
    for (let i = 0; i < 9; i++) s += fPoolArr[i * cellsPerPool + cell];
    rho[cell] = s;
  }
  return rho;
}

// Same periodic central-difference formula as shaders/amr_render.wgsl's
// get_ux/get_uy + fs_main, applied to already-unshifted (window-coordinate)
// ux/uy fields, so the sign/scale of any reported vorticity diff matches
// what the two runs actually rendered. Applied to the FINAL, fully
// composited target-resolution field (once, after reconstruction) -- not
// per source level -- matching how amr_render.wgsl computes curl per-pixel
// on whichever field values won compositing, not blended across levels.
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
  if (label) console.log(`${label}: maxAbsDiff=${maxAbsDiff.toExponential(4)}  maxAbsRef=${maxAbsRef.toExponential(4)}  relL2=${relL2.toExponential(4)}`);
  return { maxAbsDiff, maxAbsRef, relL2 };
}

// Decodes a SINGLE-resolution snapshot (dense main.js/main-cylinder.js
// 'flat', or same-resolution AMR L0 'block8') into window-space
// {ux,uy,rho,omega}. Renamed from amr-diff.js's own loadFields -- this is
// the "one solver, one resolution" case; reconstructAMRToResolution below
// is the "AMR at a different, finer resolution" case.
function loadDenseFields(snapshot) {
  const { W, H } = snapshot;
  const NCELLS = W * H;
  const offX = snapshot.cardState[22], offY = snapshot.cardState[23];
  // formatVersion 1 snapshots (pre-Milestone-1, and main-cylinder.js's own
  // dense debugSnapshotSave) predate/omit the 'layout' field and are always
  // flat row-major.
  const layout = snapshot.layout || 'flat';

  const velRaw = b64ToFloat32(snapshot.velB64, NCELLS * 2);
  const vel = unshiftField(velRaw, W, H, 2, offX, offY, layout);
  const ux = new Float32Array(NCELLS), uy = new Float32Array(NCELLS);
  for (let c = 0; c < NCELLS; c++) { ux[c] = vel[c * 2]; uy[c] = vel[c * 2 + 1]; }

  const fRaw = b64ToFloat32(snapshot.fB64, NCELLS * 9);
  // f is laid out i*(W*H) + cell; unshift each of the 9 direction planes
  // independently, then treat the whole thing as a single 9-component field
  // so rhoFromF's i*(W*H)+cell indexing still holds.
  const fUnshifted = new Float32Array(NCELLS * 9);
  for (let i = 0; i < 9; i++) {
    const plane = fRaw.subarray(i * NCELLS, (i + 1) * NCELLS);
    const planeVec = unshiftField(plane, W, H, 1, offX, offY, layout);
    fUnshifted.set(planeVec, i * NCELLS);
  }
  const rho = rhoFromF(fUnshifted, W, H);
  const omega = vorticityField(ux, uy, W, H);

  return { W, H, step: snapshot.step, layout, ux, uy, rho, omega };
}

// ---------------------------------------------------------------------
// reconstructAMRToResolution: the new piece. Decodes an AMR
// debugSnapshotSave() payload (main-cylinder-amr.js, formatVersion 5,
// pools[1..numLevels-1]) into a uniform window-space grid at
// targetResLog2 -- which must equal the AMR's own finest configured
// resolution (baseResLog2 + numLevels - 1) for a meaningful comparison;
// see tools/lib/amr-resolution-mapping.js, which is what guarantees that
// invariant on the URL-construction side. This function itself only
// asserts it (throws if 2^targetResLog2 isn't consistent with W*2^(numLevels-1)),
// it does not derive resolutions.
//
// Algorithm derived directly from shaders/amr_render.wgsl's fs_main (lines
// ~197-343) per-pixel level-compositing logic, generalized from the
// shader's hardcoded 2-tier (L0/L1/L2) wiring to an arbitrary numLevels,
// and from per-pixel to per-quadtree-node (painting whole regions at once
// rather than one target cell at a time, since a source cell can fan out to
// many target cells when a coarser level stands in for a finer one nearby).
//
// A level's four child quadrants can be independently active (an L1 block's
// four L2 children need not all exist together) -- see
// main-cylinder-amr.js's debugActivateBlock, which activates any single
// (level, bx, by) independently -- so this recurses per-quadrant, not per
// whole parent block.
// ---------------------------------------------------------------------

// Inverse of shaders/amr_step1.wgsl / amr_step1_pool.wgsl's
// fineToCoarseUnit: given a level's own dxL (cellSizeL0AtLevel(level) =
// 2^-level) and physical L0-buffer-space origin, map a fine local index
// fx/fy in [0,FB) back to L0-buffer-space physical position. Kept only as
// documentation of the relationship paintQuad exploits directly via integer
// block/quadrant arithmetic (RB is always a power-of-two block size, so no
// interpolation/rounding is needed anywhere in the reconstruction -- every
// target cell maps to exactly one source cell at exactly one level).
// fineToCoarseUnit(fCoord, origin, dxL) = origin - 0.5*dxL + dxL*(fCoord - GHOST)

function reconstructAMRToResolution(snapshot, targetResLog2) {
  const { W: W0, H: H0, numLevels, pools } = snapshot;
  if (!pools || numLevels == null) throw new Error('reconstructAMRToResolution: snapshot missing pools[]/numLevels -- not an AMR debugSnapshotSave payload');
  const targetW = 1 << targetResLog2;
  const expectedTargetW = W0 * (1 << (numLevels - 1));
  if (targetW !== expectedTargetW) {
    throw new Error(`reconstructAMRToResolution: targetResLog2=${targetResLog2} (W=${targetW}) is inconsistent with this snapshot's own finest resolution ` +
      `W0*2^(numLevels-1) = ${W0}*2^${numLevels - 1} = ${expectedTargetW} -- see tools/lib/amr-resolution-mapping.js, which is responsible for deriving ` +
      `a base resLog2 that makes this hold`);
  }

  const offX = snapshot.cardState[22], offY = snapshot.cardState[23];
  // Every pool's blockSlot/originX/originY (and, below, the per-tile
  // (bx,by)->target-cell placement math) is expressed directly in
  // BUFFER-space L0 units -- unlike loadDenseFields, which unshifts the
  // dense grid into window space up front and stays there throughout, this
  // function stays in buffer space for the whole quadtree walk (matching
  // amr_render.wgsl's own bufX/bufY-then-blockSlot-lookup order) and would
  // need a buffer->window conversion applied AT THE POINT OF WRITING each
  // painted target cell (not on the raw L0 array up front, which is what
  // unshiftField below actually does) to be correct for a nonzero off_x/
  // off_y. Every current caller of this function is the pinned-cylinder
  // harness family (main-cylinder-amr.js), which never moves its body and
  // so always has off_x=off_y=0 -- assert that explicitly rather than
  // silently mis-place fields if this is ever reused for a moving body.
  if (offX !== 0 || offY !== 0) {
    throw new Error(`reconstructAMRToResolution: off_x=${offX} off_y=${offY} (nonzero) -- this function has only been verified for the pinned-cylinder ` +
      `harnesses' fixed off_x=off_y=0 case; a moving-window snapshot needs the buffer->window conversion applied per painted target cell, not up front ` +
      `on the raw L0 array the way it's written now. Don't use this against a falling-card (main-amr.js) snapshot without fixing that first.`);
  }
  const NCELLS0 = W0 * H0;
  const velRawL0 = unshiftField(b64ToFloat32(snapshot.velB64, NCELLS0 * 2), W0, H0, 2, offX, offY, 'block8');
  const fRawL0raw = b64ToFloat32(snapshot.fB64, NCELLS0 * 9);
  const fL0 = new Float32Array(NCELLS0 * 9);
  for (let i = 0; i < 9; i++) {
    const plane = fRawL0raw.subarray(i * NCELLS0, (i + 1) * NCELLS0);
    fL0.set(unshiftField(plane, W0, H0, 1, offX, offY, 'block8'), i * NCELLS0);
  }
  const rhoL0 = rhoFromF(fL0, W0, H0);

  const RB = pools[1].RB, GHOST = pools[1].GHOST, FB = pools[1].FB;
  const NBX0 = pools[1].NBX, NBY0 = pools[1].NBY;
  const INTERIOR = 2 * RB; // interior fine cells per axis per pool slot

  // Decode each level's vel/rho once, up front -- these are read many times
  // (once per painted target cell) during the walk below.
  const levelData = [null]; // index 0 unused, L0 handled separately above
  for (let m = 1; m < numLevels; m++) {
    const pool = pools[m];
    const cellsPerPool = pool.MAX_FINE_BLOCKS * FB * FB;
    const velArr = b64ToFloat32(pool.velB64, cellsPerPool * 2);
    const fArr = b64ToFloat32(pool.fB64, cellsPerPool * 9);
    const rhoArr = rhoFromPoolF(fArr, pool.MAX_FINE_BLOCKS, FB);
    levelData.push({ blockSlot: pool.blockSlot, velArr, rhoArr, NBX: pool.NBX, NBY: pool.NBY });
  }

  const ux = new Float32Array(targetW * targetW);
  const uy = new Float32Array(targetW * targetW);
  const rho = new Float32Array(targetW * targetW);
  let finestCells = 0;
  const totalCells = targetW * targetW;

  // Paints an axis-aligned square target-grid region
  // [originX,originX+extent) x [originY,originY+extent) (in L0-buffer-space
  // units, extent measured in L0 cells) with ONE source sample -- used both
  // for whole unrefined L0 cells and for a pool slot's own interior
  // sub-block standing in for a not-(yet)-refined child quadrant. `mult` is
  // target-cells-per-axis per L0-unit of extent (i.e. targetW/W0), so the
  // painted footprint is always exactly `extent*mult` x `extent*mult`
  // target cells -- an integer since RB/extent are powers of two and mult
  // is too.
  function paintUniform(originX, originY, extent, sampleUx, sampleUy, sampleRho, isFinest) {
    const mult = targetW / W0;
    const x0 = Math.round(originX * mult), y0 = Math.round(originY * mult);
    const size = Math.round(extent * mult);
    for (let dy = 0; dy < size; dy++) {
      const wy = ((y0 + dy) % targetW + targetW) % targetW;
      for (let dx = 0; dx < size; dx++) {
        const wx = ((x0 + dx) % targetW + targetW) % targetW;
        const cell = wy * targetW + wx;
        ux[cell] = sampleUx; uy[cell] = sampleUy; rho[cell] = sampleRho;
      }
    }
    if (isFinest) finestCells += size * size;
  }

  // Paints a pool slot's own RBxRB sub-block (quadrant qx,qy of its own
  // 2RBx2RB interior) sample-by-sample -- used when that quadrant has no
  // active child of its own, so THIS level is the finest available there.
  // Each of the RB*RB fine cells is painted individually (not as one flat
  // color) since within-slot values genuinely vary cell to cell.
  function paintSlotSubblock(level, slot, qx, qy, originX, originY, isFinest) {
    const { velArr, rhoArr } = levelData[level];
    const mult = targetW / W0;
    const dxL = 1 / (1 << level); // cellSizeL0AtLevel(level), L0 units per this level's cell
    for (let ly = 0; ly < RB; ly++) {
      const fy = GHOST + qy * RB + ly;
      for (let lx = 0; lx < RB; lx++) {
        const fx = GHOST + qx * RB + lx;
        const cellIdx = slot * (FB * FB) + fy * FB + fx;
        const sx = originX + qx * RB * dxL + lx * dxL;
        const sy = originY + qy * RB * dxL + ly * dxL;
        paintUniform(sx, sy, dxL, velArr[cellIdx * 2], velArr[cellIdx * 2 + 1], rhoArr[cellIdx], isFinest);
      }
      // paintUniform above is called per fine cell (dxL extent) rather than
      // once for the whole RBxRB sub-block -- necessary because cell values
      // vary within the block; the mult/size math in paintUniform still
      // collapses each individual call to its own exact target-cell square.
    }
    void mult;
  }

  // Paints an entire slot's full interior ((2*RB)x(2*RB) fine cells) --
  // used when level+1 >= numLevels (no possible child at all, this is
  // structurally the finest level anywhere).
  function paintSlotFull(level, slot, originX, originY) {
    const { velArr, rhoArr } = levelData[level];
    const dxL = 1 / (1 << level);
    for (let fy = GHOST; fy < GHOST + INTERIOR; fy++) {
      for (let fx = GHOST; fx < GHOST + INTERIOR; fx++) {
        const cellIdx = slot * (FB * FB) + fy * FB + fx;
        const sx = originX + (fx - GHOST) * dxL;
        const sy = originY + (fy - GHOST) * dxL;
        paintUniform(sx, sy, dxL, velArr[cellIdx * 2], velArr[cellIdx * 2 + 1], rhoArr[cellIdx], true);
      }
    }
  }

  // level: which level's `slot` we're currently inside (>=1). bx,by: this
  // slot's OWN block-id coordinates in level `level`'s own NBX x NBY grid.
  // originX/originY: this slot's L0-buffer-space physical origin.
  function paintQuad(level, slot, bx, by, originX, originY) {
    const childLevel = level + 1;
    if (childLevel >= numLevels) {
      paintSlotFull(level, slot, originX, originY);
      return;
    }
    const child = levelData[childLevel];
    const nbxChild = NBX0 * (1 << level); // = child.NBX, sanity-checked below
    if (child.NBX !== nbxChild) {
      throw new Error(`reconstructAMRToResolution: level ${childLevel} NBX=${child.NBX}, expected ${nbxChild} from doubling -- pool grid geometry assumption violated`);
    }
    const dxLparent = 1 / (1 << level);
    for (let qy = 0; qy < 2; qy++) {
      for (let qx = 0; qx < 2; qx++) {
        const childBX = bx * 2 + qx, childBY = by * 2 + qy;
        const childSlot = child.blockSlot[childBY * nbxChild + childBX];
        const childOriginX = originX + qx * RB * dxLparent;
        const childOriginY = originY + qy * RB * dxLparent;
        if (childSlot === -1) {
          paintSlotSubblock(level, slot, qx, qy, originX, originY, childLevel >= numLevels);
        } else {
          paintQuad(childLevel, childSlot, childBX, childBY, childOriginX, childOriginY);
        }
      }
    }
  }

  const L1 = levelData[1];
  for (let by = 0; by < NBY0; by++) {
    for (let bx = 0; bx < NBX0; bx++) {
      const slot1 = L1.blockSlot[by * NBX0 + bx];
      if (slot1 === -1) {
        // No L1 tile here at all -- L0 is authoritative for this whole
        // RBxRB block, one L0 cell at a time (values vary cell to cell).
        for (let ly = 0; ly < RB; ly++) {
          for (let lx = 0; lx < RB; lx++) {
            const cx = bx * RB + lx, cy = by * RB + ly;
            const cell = cy * W0 + cx;
            paintUniform(cx, cy, 1, velRawL0[cell * 2], velRawL0[cell * 2 + 1], rhoL0[cell], numLevels === 1);
          }
        }
      } else {
        paintQuad(1, slot1, bx, by, bx * RB, by * RB);
      }
    }
  }

  const omega = vorticityField(ux, uy, targetW, targetW);
  return { W: targetW, H: targetW, step: snapshot.step, ux, uy, rho, omega, finestCoverageFraction: finestCells / totalCells };
}

module.exports = {
  b64ToFloat32, rawIndex, unshiftField, rhoFromF, rhoFromPoolF, vorticityField, diffStats,
  loadDenseFields, reconstructAMRToResolution,
};
