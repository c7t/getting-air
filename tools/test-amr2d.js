#!/usr/bin/env node
// GPU-free tests for amr2d.mjs -- the 2D quadtree pool's addressing and
// structural invariants (plans/2D-backport.md B0).
//
// WHY THIS IS THE TEST THAT MATTERS. Before this file, every 2D AMR
// invariant lived inside main-amr.js and could only run in a browser against
// a live GPU, so an AMR refactor was gated by a Cd/St sweep that already
// carries two standing red cells and whose AMR numbers are only reproducible
// to ~1e-3. That is not a gate you can move an allocator behind.
//
// Three rules, all taken from tools/test-d3-amr.js, which is why the 3D
// fork's M4/M5 survived:
//
//   1. CHECK AGAINST AN INDEPENDENT ROUTE. Neighbour resolution is scored
//      against GLOBAL FINE COORDINATES, where "which tile owns this cell" is
//      a single division with no ring, no offsets and no periodic block wrap
//      to get wrong. resolveSource and the global route share no arithmetic,
//      so agreement between them is evidence rather than tautology.
//   2. MUTATION-CHECK. A formula agreeing with a route written alongside it
//      proves nothing. Every closed form here is also given a set of
//      plausible slips, and the test asserts each slip is CAUGHT. Where the
//      repo has a record of the actual historical bug (the quadrant origin
//      scaled by 2*RB instead of RB, shaders/amr_manage_pool.wgsl:380-403)
//      that bug is one of the mutants.
//   3. RUN THE CHECKERS ON INPUTS THAT VIOLATE THE INVARIANT. A checker only
//      ever run on valid input is indistinguishable from one that returns
//      nothing.
//
// AND THE SUITE ITSELF WAS SCORED THE SAME WAY, because a green run says
// nothing about what they would catch. amr2d.mjs was mutated one edit at a
// time and this suite re-run; every mutant below must break at least one
// check, and two of them did NOT until the checks named beside them were
// added:
//
//   edge-only closure (drop the diagonals)              3 failures
//   no quad completion                                  4
//   whole-tile depth, not the shared edge               3
//   non-periodic block wrap in the checker              0 -> 2  (added "THE
//                                                       BLOCK GRID IS PERIODIC")
//   ring coverage checks edges only                     2
//   ancestorDepth stops one level short                 5
//   cascade sweeps coarsest-first, not deepest-first    5
//   branch-and-bound accepts on the bound               0 -> 1  (added "the
//                                                       branch-and-bound
//                                                       SUBDIVIDES")
//
// Both holes were the same shape: every fixture sat in the MIDDLE of the grid
// and every geometry case had a body large enough that subdivision never
// mattered, so the two mutants produced answers that were merely conservative
// rather than wrong. Add a fixture at the seam and one with a small body
// before trusting a new check here.
//
// Run: node tools/test-amr2d.js   (also picked up by `make test`)

const assert = require('assert');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}
const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b}`);

const key = (b) => `${b[0]},${b[1]}`;
const sorted = (s) => [...s].sort();

(async () => {
  const A = await import(path.join(__dirname, '..', 'amr2d.mjs'));
  const {
    GHOST, RB_DEFAULT, fineToCoarseUnit, coarseUnitToFine, cellSizeL0AtLevel,
    grantAssignment, releaseAssignment,
    tileCellsAtLevel, ghostDepthAtLevel, tileSideAtLevel, blockGridAtLevel,
    parentCellOfFineCell, fineCellsOfParentCell, ringDepth, ringSlotRole,
    poolInverseViolations,
    makePool, poolAtLevel, nbAtLevel, parentOfBlock,
    quadrantOfBlock, quadrantOrigin, tileOriginL0, tileOriginL0Recursive,
    refineWhere, nearBodyWant, nearBodyWantCentre, refineNearBody,
    resolveSource, toGlobalFine, fromGlobalFine, storageRatio,
    check21Balance, checkRingParentCoverage, checkGeometryCoverage, cascade21,
    SQRT2_UP, BLOCK_BB_DEPTH, quadrantOfSlot,
    dcRescaleCoarseToFine, dcRescaleFineToCoarse,
    dcRescaleCoarseToFinePre, dcRescaleFineToCoarsePre,
    tauChainSingularity, tauSingularityMessage, TAU_UNITY_BAND,
  } = A;

  // ── pool geometry ────────────────────────────────────────────────────────

  ok('pool geometry follows FB = 2*RB + 2*GHOST and rejects a non-dividing RB', () => {
    for (const rb of [2, 4, 8]) {
      const p = makePool({ dims: [64, 64], rb });
      assert.strictEqual(p.FB, 2 * rb + 2 * GHOST, `FB at RB=${rb}`);
      assert.strictEqual(p.nBlocks, (64 / rb) ** 2);
      assert.strictEqual(p.tileCells, p.FB ** 2);
    }
    assert.strictEqual(makePool({ dims: [256, 256] }).rb, RB_DEFAULT);
    assert.throws(() => makePool({ dims: [60, 64], rb: 8 }), /not a multiple of RB/);
    assert.throws(() => makePool({ dims: [64.5, 64], rb: 8 }), /positive integer/);
  });

  ok('storage ratio is (FB/RB)^2 -- 6.25 at the solver\'s own RB=8', () => {
    close(storageRatio(makePool({ dims: [64, 64], rb: 8 })), 6.25, 1e-12, 'RB=8');
    close(storageRatio(makePool({ dims: [64, 64], rb: 4 })), 9.0, 1e-12, 'RB=4');
  });

  ok('block id and its inverse round-trip over the whole grid', () => {
    const p = makePool({ dims: [48, 32], rb: 8 });
    let n = 0;
    for (let by = 0; by < p.nb[1]; by++) {
      for (let bx = 0; bx < p.nb[0]; bx++) {
        const id = p.blockId(bx, by);
        assert.strictEqual(id, n++, `blockId(${bx},${by}) is not row-major-contiguous`);
        assert.deepStrictEqual(p.blockOf(id), [bx, by]);
      }
    }
    assert.strictEqual(n, p.nBlocks);
  });

  ok('interior is [GHOST, GHOST+2*RB) and the ring is everything else', () => {
    const p = makePool({ dims: [64, 64], rb: 8 });
    const interior = [];
    for (let j = 0; j < p.FB; j++) if (p.isInterior(j)) interior.push(j);
    assert.strictEqual(interior.length, 2 * p.rb);
    assert.strictEqual(interior[0], GHOST);
    assert.strictEqual(interior[interior.length - 1], GHOST + 2 * p.rb - 1);
  });

  // ── the fine<->coarse mapping ────────────────────────────────────────────

  ok('refinement is CELL-CENTRED: a parent cell\'s two children sit at +-dx/2', () => {
    // The first interior fine cell (j = GHOST) is the first child of the
    // block's first parent cell, whose centre is at `origin`.
    for (const dx of [0.5, 0.25, 0.125]) {
      close(fineToCoarseUnit(GHOST, 0, dx), -dx / 2, 1e-12, `child 0 at dx=${dx}`);
      close(fineToCoarseUnit(GHOST + 1, 0, dx), +dx / 2, 1e-12, `child 1 at dx=${dx}`);
      // ...and the pair straddles the parent centre exactly.
      close((fineToCoarseUnit(GHOST, 0, dx) + fineToCoarseUnit(GHOST + 1, 0, dx)) / 2,
        0, 1e-12, `pair midpoint at dx=${dx}`);
    }
    // Level 1's own numbers, as shaders/amr_step1.wgsl:120 hardcodes them.
    close(fineToCoarseUnit(GHOST, 24, 0.5), 23.75, 1e-12, 'origin 24, j=GHOST');
  });

  ok('coarseUnitToFine inverts fineToCoarseUnit, ring indices included', () => {
    for (const dx of [0.5, 0.25]) {
      for (const origin of [0, 8, 24.5]) {
        for (let j = -2; j < 22; j++) {
          assert.strictEqual(coarseUnitToFine(fineToCoarseUnit(j, origin, dx), origin, dx), j,
            `round-trip failed at j=${j}, origin=${origin}, dx=${dx}`);
        }
      }
    }
  });

  ok('the fine<->coarse mapping breaks under a lost half-cell, a wrong dx and a lost GHOST', () => {
    // MUTATION-CHECKED. Each of these is a plausible slip in one line, and
    // each must produce a different answer somewhere in the range a kernel
    // actually evaluates.
    const mutants = {
      'dropped the -dx/2 (cell corners, not centres)': (j, o, dx) => o + dx * (j - GHOST),
      'used the parent cell size for dx': (j, o) => o - 0.5 + 1.0 * (j - GHOST),
      'forgot GHOST (ring counted as interior)': (j, o, dx) => o - 0.5 * dx + dx * j,
      'sign-flipped the half-cell': (j, o, dx) => o + 0.5 * dx + dx * (j - GHOST),
    };
    for (const [name, mut] of Object.entries(mutants)) {
      let differs = false;
      for (const dx of [0.5, 0.25]) {
        for (let j = -2; j < 22; j++) {
          if (Math.abs(fineToCoarseUnit(j, 8, dx) - mut(j, 8, dx)) > 1e-12) differs = true;
        }
      }
      assert.ok(differs, `mutant "${name}" was not caught -- the check cannot see it`);
    }
  });

  // ── the uniform tile shape ───────────────────────────────────────────────

  ok('poolAtLevel is makePool on a scaled domain -- the tile shape is quadtree-uniform', () => {
    const base = makePool({ dims: [256, 256], rb: 8 });
    for (let m = 1; m <= 4; m++) {
      const p = poolAtLevel(base, m);
      assert.strictEqual(p.rb, base.rb, `RB changed at level ${m}`);
      assert.strictEqual(p.FB, base.FB, `FB changed at level ${m} -- the tile is NOT uniform`);
      assert.deepStrictEqual(p.nb, nbAtLevel(base, m), `nb disagrees at level ${m}`);
      assert.deepStrictEqual(p.nb, [base.nb[0] * 2 ** (m - 1), base.nb[1] * 2 ** (m - 1)]);
      // A level-m block's L0 footprint is rb * 2^(1-m), and the blocks must
      // still tile the domain exactly.
      const footprint = p.rb * cellSizeL0AtLevel(m - 1);
      close(p.nb[0] * footprint, base.dims[0], 1e-12, `level ${m} does not tile the domain`);
    }
    assert.throws(() => poolAtLevel(base, 0), /not a pool level/);
  });

  ok('parent/quadrant decompose a block index and recompose it', () => {
    for (const b of [[0, 0], [1, 0], [0, 1], [1, 1], [13, 7], [64, 65]]) {
      const p = parentOfBlock(b), q = quadrantOfBlock(b);
      assert.deepStrictEqual([p[0] * 2 + q[0], p[1] * 2 + q[1]], b, `recompose ${b}`);
      assert.ok(q[0] === 0 || q[0] === 1);
    }
    // The child's interior offset inside the parent's tile-local frame.
    assert.strictEqual(quadrantOrigin(8, 0), GHOST);
    assert.strictEqual(quadrantOrigin(8, 1), GHOST + 8);
    // ...and the two quadrants exactly halve the parent's 2*RB interior.
    assert.strictEqual(quadrantOrigin(8, 1) + 8, GHOST + 2 * 8);
  });

  ok('a tile\'s L0 origin agrees between the closed form and the GPU\'s parent-chain walk', () => {
    // INDEPENDENT ROUTES. shaders/amr_manage_pool.wgsl:403 builds the origin
    // recursively at allocation time; tileOriginL0 is one multiply. They must
    // agree at every level and every block, and the recursion is the one that
    // was transposed once.
    for (let m = 1; m <= 4; m++) {
      const n = 4 * 2 ** (m - 1);
      for (let by = 0; by < n; by++) {
        for (let bx = 0; bx < n; bx++) {
          const a = tileOriginL0([bx, by], m);
          const b = tileOriginL0Recursive([bx, by], m);
          close(a[0], b[0], 1e-12, `origin x at level ${m} block ${bx},${by}`);
          close(a[1], b[1], 1e-12, `origin y at level ${m} block ${bx},${by}`);
        }
      }
    }
    // Level 1's origin is shaders/amr_step1.wgsl's own `bx * RB`.
    assert.deepStrictEqual(tileOriginL0([3, 5], 1), [24, 40]);
    // A level-2 tile is one quadrant of its parent's footprint.
    assert.deepStrictEqual(tileOriginL0([6, 10], 2), [24, 40]);
    assert.deepStrictEqual(tileOriginL0([7, 11], 2), [28, 44]);
  });

  ok('sibling tiles TILE their parent\'s footprint -- no gap, no overlap', () => {
    // This is the property shaders/amr_manage_pool.wgsl:380-403 records
    // losing: "quadrant 1 sat overlapping half of quadrant 0's true
    // territory and left the outer half of the parent's footprint uncovered
    // by any tile at all". 2:1 balance is an index-only check and could not
    // see it; this can.
    for (let m = 2; m <= 4; m++) {
      const parentFootprint = RB_DEFAULT * cellSizeL0AtLevel(m - 2);
      const childFootprint = RB_DEFAULT * cellSizeL0AtLevel(m - 1);
      close(2 * childFootprint, parentFootprint, 1e-12, `two children span the parent at level ${m}`);
      const p = [3, 5];
      const origins = [];
      for (let k = 0; k < 4; k++) {
        origins.push(tileOriginL0([p[0] * 2 + (k & 1), p[1] * 2 + ((k >> 1) & 1)], m));
      }
      const po = tileOriginL0(p, m - 1);
      const want = [
        [po[0], po[1]], [po[0] + childFootprint, po[1]],
        [po[0], po[1] + childFootprint], [po[0] + childFootprint, po[1] + childFootprint],
      ];
      for (let k = 0; k < 4; k++) {
        close(origins[k][0], want[k][0], 1e-12, `quadrant ${k} x at level ${m}`);
        close(origins[k][1], want[k][1], 1e-12, `quadrant ${k} y at level ${m}`);
      }
    }
  });

  ok('the tile origin breaks under a transposed quadrant and the historical 2*RB slip', () => {
    // MUTATION-CHECKED, and the second mutant IS the bug
    // shaders/amr_manage_pool.wgsl:380-403 live-verified: the quadrant offset
    // scaled by the parent's full interior (2*RB) instead of RB, because "RB
    // is already half the interior, no further factor belongs" was applied to
    // the sibling-centre computation and never back-applied to the origin.
    const mutants = {
      'transposed the quadrant (qx and qy swapped)': (block, m) => {
        let b = block.slice(); const quads = [];
        for (let lv = m; lv >= 2; lv--) { const q = quadrantOfBlock(b); quads.unshift([q[1], q[0]]); b = parentOfBlock(b); }
        let o = [b[0] * RB_DEFAULT, b[1] * RB_DEFAULT];
        for (let lv = 2; lv <= m; lv++) {
          const q = quads[lv - 2], pcs = cellSizeL0AtLevel(lv - 1);
          o = [o[0] + q[0] * RB_DEFAULT * pcs, o[1] + q[1] * RB_DEFAULT * pcs];
        }
        return o;
      },
      'scaled the quadrant offset by 2*RB, the parent\'s whole interior': (block, m) => {
        let b = block.slice(); const quads = [];
        for (let lv = m; lv >= 2; lv--) { quads.unshift(quadrantOfBlock(b)); b = parentOfBlock(b); }
        let o = [b[0] * RB_DEFAULT, b[1] * RB_DEFAULT];
        for (let lv = 2; lv <= m; lv++) {
          const q = quads[lv - 2], pcs = cellSizeL0AtLevel(lv - 1);
          o = [o[0] + q[0] * 2 * RB_DEFAULT * pcs, o[1] + q[1] * 2 * RB_DEFAULT * pcs];
        }
        return o;
      },
      'used the CHILD\'s cell size instead of the parent\'s': (block, m) => {
        let b = block.slice(); const quads = [];
        for (let lv = m; lv >= 2; lv--) { quads.unshift(quadrantOfBlock(b)); b = parentOfBlock(b); }
        let o = [b[0] * RB_DEFAULT, b[1] * RB_DEFAULT];
        for (let lv = 2; lv <= m; lv++) {
          const q = quads[lv - 2], ccs = cellSizeL0AtLevel(lv);
          o = [o[0] + q[0] * RB_DEFAULT * ccs, o[1] + q[1] * RB_DEFAULT * ccs];
        }
        return o;
      },
    };
    for (const [name, mut] of Object.entries(mutants)) {
      let differs = false;
      for (let m = 2; m <= 3; m++) {
        for (let by = 0; by < 8; by++) for (let bx = 0; bx < 8; bx++) {
          const a = tileOriginL0([bx, by], m), b = mut([bx, by], m);
          if (Math.abs(a[0] - b[0]) > 1e-12 || Math.abs(a[1] - b[1]) > 1e-12) differs = true;
        }
      }
      assert.ok(differs, `mutant "${name}" was not caught -- the check cannot see it`);
    }
  });

  // ── neighbour resolution, against the independent route ──────────────────

  ok('resolveSource agrees with the global-fine route at every level and every offset', () => {
    // The two share no arithmetic: resolveSource works in tile-local
    // coordinates with a ring and a periodic block wrap; the global route is
    // one division.
    for (const rb of [4, 8]) {
      const base = makePool({ dims: [64, 64], rb });
      for (const m of [1, 2, 3]) {
        const pool = poolAtLevel(base, m);
        // Everything refined, so "no tile owns it" never fires and every
        // offset is exercised.
        const blockSlot = new Int32Array(pool.nBlocks).map((_, i) => i);
        const slotToBlockId = new Int32Array(pool.nBlocks).map((_, i) => i);
        for (const b of [[0, 0], [1, 2], [pool.nb[0] - 1, pool.nb[1] - 1]]) {
          for (let sy = -1; sy <= pool.FB; sy++) {
            for (let sx = -1; sx <= pool.FB; sx++) {
              const r = resolveSource(pool, blockSlot, b, [sx, sy]);
              assert.ok(r !== null, `no owner at level ${m} rb=${rb} b=${b} src=${sx},${sy}`);
              const owner = r.own ? b : pool.blockOf(slotToBlockId[r.slot]);
              const viaResolve = toGlobalFine(pool, owner, [r.fx, r.fy]);
              // The independent route.
              const g = toGlobalFine(pool, b, [sx, sy]);
              const direct = fromGlobalFine(pool, g);
              const viaGlobal = toGlobalFine(pool, direct.block, direct.local);
              assert.deepStrictEqual(viaResolve, viaGlobal,
                `level ${m} rb=${rb} b=${b} src=${sx},${sy}: resolve says ${viaResolve}, global route says ${viaGlobal}`);
              // And the owner really is the owner.
              assert.deepStrictEqual(owner, direct.block,
                `level ${m} rb=${rb} b=${b} src=${sx},${sy}: wrong owning tile`);
              assert.ok(pool.isInterior(r.fx) && pool.isInterior(r.fy) || r.own,
                `a resolved neighbour cell must land in the neighbour's INTERIOR, never its ring`);
            }
          }
        }
      }
    }
  });

  ok('RB=4 and RB=8 resolve the same physical cell -- RB only cuts the domain into tiles', () => {
    // Asserted EXACTLY, not to a tolerance: RB changes how the domain is
    // partitioned and nothing else, so the global fine cell a gather lands on
    // cannot depend on it.
    const dims = [64, 64];
    const answers = {};
    for (const rb of [4, 8]) {
      const pool = makePool({ dims, rb });
      const blockSlot = new Int32Array(pool.nBlocks).map((_, i) => i);
      const out = [];
      for (let gy = 0; gy < 2 * dims[1]; gy += 7) {
        for (let gx = 0; gx < 2 * dims[0]; gx += 5) {
          for (const [ex, ey] of [[1, 0], [0, 1], [1, 1], [-1, 1], [-1, -1]]) {
            const here = fromGlobalFine(pool, [gx, gy]);
            const src = [here.local[0] - ex, here.local[1] - ey];
            const r = resolveSource(pool, blockSlot, here.block, src);
            const owner = r.own ? here.block : pool.blockOf(r.slot);
            out.push(toGlobalFine(pool, owner, [r.fx, r.fy]).join(','));
          }
        }
      }
      answers[rb] = out;
    }
    assert.deepStrictEqual(answers[4], answers[8], 'RB=4 and RB=8 disagree about where a gather reads');
  });

  ok('resolveSource returns null where no tile owns the source -- the ring\'s one job', () => {
    const pool = makePool({ dims: [64, 64], rb: 8 });
    const blockSlot = new Int32Array(pool.nBlocks).fill(-1);
    blockSlot[pool.blockId(3, 3)] = 0;                 // an isolated tile
    // Interior: never leaves, so no neighbour is consulted at all.
    const inside = resolveSource(pool, blockSlot, [3, 3], [GHOST + 4, GHOST + 4]);
    assert.strictEqual(inside.own, true);
    assert.strictEqual(inside.slot, null);
    // A source one cell past the interior wants the neighbour, which is absent.
    assert.strictEqual(resolveSource(pool, blockSlot, [3, 3], [GHOST - 1, GHOST + 4]), null);
    assert.strictEqual(resolveSource(pool, blockSlot, [3, 3], [GHOST - 1, GHOST - 1]), null);
    // With the neighbour present it resolves into that tile's interior.
    blockSlot[pool.blockId(2, 3)] = 1;
    const r = resolveSource(pool, blockSlot, [3, 3], [GHOST - 1, GHOST + 4]);
    assert.strictEqual(r.slot, 1);
    assert.strictEqual(r.fx, GHOST - 1 + 2 * pool.rb);
    assert.ok(pool.isInterior(r.fx));
  });

  ok('neighbour resolution breaks under a lost re-expression, a wrong bound and a lost wrap', () => {
    // MUTATION-CHECKED against the global route, which is the only reason the
    // agreement above means anything.
    const pool = makePool({ dims: [64, 64], rb: 8 });
    const blockSlot = new Int32Array(pool.nBlocks).map((_, i) => i);
    const RB2 = 2 * pool.rb;
    const mutants = {
      'forgot to re-express the source in the neighbour\'s frame': (b, src) => {
        const nbr = [0, 1].map(a => (src[a] < GHOST ? -1 : src[a] >= GHOST + RB2 ? 1 : 0));
        const ob = [0, 1].map(a => (b[a] + nbr[a] + pool.nb[a]) % pool.nb[a]);
        return toGlobalFine(pool, ob, src);
      },
      'tested the interior bound with > instead of >=': (b, src) => {
        const out = [0, 0], nbr = [0, 0];
        for (let a = 0; a < 2; a++) {
          const s = src[a];
          if (s < GHOST) { nbr[a] = -1; out[a] = s + RB2; }
          else if (s > GHOST + RB2) { nbr[a] = 1; out[a] = s - RB2; }
          else { nbr[a] = 0; out[a] = s; }
        }
        const ob = [0, 1].map(a => (b[a] + nbr[a] + pool.nb[a]) % pool.nb[a]);
        return toGlobalFine(pool, ob, out);
      },
      'dropped the periodic block wrap': (b, src) => {
        const out = [0, 0], nbr = [0, 0];
        for (let a = 0; a < 2; a++) {
          const s = src[a];
          if (s < GHOST) { nbr[a] = -1; out[a] = s + RB2; }
          else if (s >= GHOST + RB2) { nbr[a] = 1; out[a] = s - RB2; }
          else { nbr[a] = 0; out[a] = s; }
        }
        return toGlobalFine(pool, [b[0] + nbr[0], b[1] + nbr[1]], out);
      },
    };
    for (const [name, mut] of Object.entries(mutants)) {
      let differs = false;
      for (const b of [[0, 0], [2, 5], [pool.nb[0] - 1, 0]]) {
        for (let sy = -1; sy <= pool.FB; sy++) for (let sx = -1; sx <= pool.FB; sx++) {
          const truth = resolvedGlobalFor(pool, blockSlot, b, [sx, sy]);
          const got = mut(b, [sx, sy]);
          if (truth.join(',') !== got.join(',')) differs = true;
        }
      }
      assert.ok(differs, `mutant "${name}" was not caught -- the check cannot see it`);
    }
    function resolvedGlobalFor(p, bs, b, src) {
      const r = resolveSource(p, bs, b, src);
      const owner = r.own ? b : p.blockOf(r.slot);
      return toGlobalFine(p, owner, [r.fx, r.fy]);
    }
  });

  // ── the body's own distance, against a brute-force closest point ─────────

  // THE INDEPENDENT ROUTE: sample the ellipse boundary densely and take the
  // nearest sample. It shares nothing with the Newton iteration -- no seed,
  // no derivative, no quadrant folding -- so agreement is evidence.
  const bruteEllipsePhi = (px, py, { cx, cy, theta, a, b }, { W, H }, N = 200000) => {
    const ca = Math.cos(theta), sa = Math.sin(theta);
    let dx = px - cx, dy = py - cy;
    dx -= W * Math.round(dx / W);
    dy -= H * Math.round(dy / H);
    const lx = dx * ca + dy * sa, ly = -dx * sa + dy * ca;
    let best = Infinity;
    for (let i = 0; i < N; i++) {
      const t = 2 * Math.PI * i / N;
      const ex = lx - a * Math.cos(t), ey = ly - b * Math.sin(t);
      const d2 = ex * ex + ey * ey;
      if (d2 < best) best = d2;
    }
    const inside = (lx * lx) / (a * a) + (ly * ly) / (b * b) < 1;
    return inside ? -Math.sqrt(best) : Math.sqrt(best);
  };

  ok('ellipsePhi is a TRUE distance -- within 0.2% of a brute-force closest point', () => {
    // The tolerance is common_geometry.wgsl's own measured claim for three
    // Newton iterations ("0.15% worst-case relative error against a
    // brute-force reference over the region that matters"), not a number
    // chosen to make this pass. The falling card's default aspect is 8:1,
    // which is exactly where the OLD algebraic form is worst.
    const dims = { W: 256, H: 256 };
    const card = { cx: 128, cy: 128, theta: 0.7, a: 24, b: 3, vx: 0, vy: 0, omega: 0 };
    let worst = 0, worstAt = null, samples = 0;
    for (let px = 96; px <= 160; px += 4) {
      for (let py = 96; py <= 160; py += 4) {
        const got = A.ellipsePhi(px, py, card, dims);
        const want = bruteEllipsePhi(px, py, card, dims);
        if (want < 0) continue;   // interior: only the SIGN is guaranteed there
        if (want < 1e-6) continue;
        samples++;
        const rel = Math.abs(got - want) / Math.max(want, 1);
        if (rel > worst) { worst = rel; worstAt = [px, py, got, want]; }
      }
    }
    assert.ok(samples > 100, `only ${samples} exterior samples -- the fixture stopped testing anything`);
    assert.ok(worst <= 2e-3, `worst relative error ${worst} at ${JSON.stringify(worstAt)}`);
  });

  ok('ellipsePhi\'s SIGN is exact everywhere, inside included', () => {
    const dims = { W: 256, H: 256 };
    const card = { cx: 128, cy: 128, theta: -1.1, a: 24, b: 3 };
    for (let px = 100; px <= 156; px += 1) {
      for (let py = 100; py <= 156; py += 1) {
        const got = A.ellipsePhi(px, py, card, dims);
        const want = bruteEllipsePhi(px, py, card, dims, 20000);
        if (Math.abs(want) < 1e-3) continue;   // on the boundary, sign is moot
        assert.strictEqual(Math.sign(got), Math.sign(want), `sign at ${px},${py}: ${got} vs ${want}`);
      }
    }
  });

  ok('the ALGEBRAIC shortcut is the one main-cylinder-amr.js could not share', () => {
    // Not a check of amr2d.mjs but of WHY this function had to be written:
    // the only existing 2D coverage checker used `(r - 1) * b`, which is
    // exact for a circle and under-reports by up to a/b on an ellipse. On the
    // card's 8:1 default that is the difference between a correct check and
    // one that flags a ring of tiles the kernel never considered near.
    const dims = { W: 256, H: 256 };
    const disc = { cx: 128, cy: 128, theta: 0.3, a: 9, b: 9 };
    const algebraic = (px, py, { cx, cy, theta, a, b }) => {
      const ca = Math.cos(theta), sa = Math.sin(theta);
      const dx = px - cx, dy = py - cy;
      const lx = dx * ca + dy * sa, ly = -dx * sa + dy * ca;
      return (Math.sqrt((lx * lx) / (a * a) + (ly * ly) / (b * b)) - 1) * b;
    };
    // A circle: the two agree, which is why the cylinder page never noticed.
    for (const [px, py] of [[140, 128], [128, 150], [137, 141]]) {
      close(A.ellipsePhi(px, py, disc, dims), algebraic(px, py, disc), 1e-9, `circle at ${px},${py}`);
    }
    // An 8:1 card on its major axis: the shortcut is short by ~8x.
    const card = { cx: 128, cy: 128, theta: 0, a: 24, b: 3 };
    const ratio = A.ellipsePhi(160, 128, card, dims) / algebraic(160, 128, card, dims);
    assert.ok(ratio > 7 && ratio < 8.1, `major-axis ratio is ${ratio}, expected ~a/b = 8`);
  });

  ok('ellipsePhi takes the NEAREST PERIODIC IMAGE, so a body at the seam is not far away', () => {
    // The fixture the mutation sweep found missing the first time: every
    // earlier geometry case sat in the MIDDLE of the grid, where a wrap bug
    // is invisible.
    const dims = { W: 64, H: 64 };
    const body = { cx: 1, cy: 32, theta: 0, a: 4, b: 4 };
    // x = 63 is two cells from cx = 1 the short way round, 62 the long way.
    close(A.ellipsePhi(63, 32, body, dims), -2, 1e-6, 'across the x seam');
    close(A.ellipsePhi(5, 32, body, dims), 0, 1e-6, 'the same distance the other side');
    const tall = { cx: 32, cy: 63, theta: 0, a: 4, b: 4 };
    close(A.ellipsePhi(32, 1, tall, dims), -2, 1e-6, 'across the y seam');
  });

  ok('bodyPhiL0 takes the MINIMUM over the current and extrapolated poses', () => {
    const dims = { W: 256, H: 256 };
    // Moving in +x at 0.5 cells/macro-step. The kernel extrapolates the TEST
    // POINT backward at -v rather than the body forward, because the moving
    // window absorbs bulk translation into off_x/off_y -- so a point AHEAD of
    // the body in x is the one the future pose brings closer.
    const card = { cx: 128, cy: 128, theta: 0, a: 6, b: 6, vx: 0.5, vy: 0, omega: 0 };
    const now = A.ellipsePhi(150, 128, card, dims);
    const both = A.bodyPhiL0(150, 128, card, dims, 20);
    close(now, 16, 1e-6, 'the current pose');
    close(both, 6, 1e-6, 'the test point moved back by vx*lookahead = 10');
    // Zero lookahead is the current pose alone, not a min against itself.
    close(A.bodyPhiL0(150, 128, card, dims, 0), now, 0, 'lookahead 0');
    // Rotation still runs FORWARD -- the window absorbs translation, not spin.
    const spin = { cx: 128, cy: 128, theta: 0, a: 24, b: 3, vx: 0, vy: 0, omega: Math.PI / 2 / 10 };
    const spun = A.bodyPhiL0(128, 145, spin, dims, 10);
    assert.ok(spun < A.ellipsePhi(128, 145, spin, dims) - 10,
      `a quarter turn should bring the major axis under the test point (got ${spun})`);
  });

  ok('pool capacity is sized per level, and a level with children gets more', () => {
    // The card page's measured table (main-amr.js POOL_PEAKS). The point of
    // the test is the SHAPE, not the digits: a flat default per level is what
    // made ?levels=4 refuse, so the thing worth pinning is that deeper levels
    // get strictly more, and that acquiring a child raises a level's size.
    const peaks = {
      finest: { 2: 400, 3: 656, 4: 904 },
      parent: { 1: 261, 2: 516, 3: 628 },
    };
    const H = A.POOL_HEADROOM;

    // Level 2 as the FINEST level (levels=3) vs. with a child (levels=4).
    const quad = (n) => Math.ceil(n / 4) * 4;   // slots come in quads; see below
    const finest2 = A.poolSlotsFor(peaks, 2, 3);
    const parent2 = A.poolSlotsFor(peaks, 2, 4);
    assert.strictEqual(finest2, quad(400 * H));
    assert.strictEqual(parent2, quad(516 * H));
    assert.ok(parent2 > finest2,
      `a level that must parent a finer one needs MORE, got ${parent2} <= ${finest2}`);

    // Deeper levels want strictly more -- the property the flat 512 violated.
    const l1 = A.poolSlotsFor(peaks, 1, 5);
    const l2 = A.poolSlotsFor(peaks, 2, 5);
    const l3 = A.poolSlotsFor(peaks, 3, 5);
    const l4 = A.poolSlotsFor(peaks, 4, 5);
    assert.ok(l1 < l2 && l2 < l3 && l3 < l4,
      `demand must grow with depth, got ${[l1, l2, l3, l4].join(' ')}`);

    // Every default must clear the measured peak it was derived from --
    // otherwise the pool is in permanent exhaustion by construction.
    assert.ok(A.poolSlotsFor(peaks, 3, 4) > 656, 'level 3 must exceed its own measured peak');
    assert.ok(A.poolSlotsFor(peaks, 4, 5) > 904, 'level 4 must exceed its own measured peak');
    // 512 -- the old flat default -- did NOT, which is the bug in one line.
    assert.ok(A.poolSlotsFor(peaks, 3, 4) > 512, 'the old flat 512 sat below level 3 demand');

    // A level nobody measured still gets a number, and a bigger one than the
    // deepest measured level rather than a silent fallback to something small.
    const l5 = A.poolSlotsFor(peaks, 5, 6);
    assert.ok(l5 > l4, `extrapolated level 5 (${l5}) must exceed measured level 4 (${l4})`);

    // EVERY size must be a multiple of 4: slots are allocated a QUAD at a time
    // (one per child quadrant) and allocLevelPool refuses anything else at
    // init. Live-verified by getting it wrong -- a bare 1.7x gave level 2 878
    // and index-amr.html?levels=4 would not boot at all.
    for (const [m, n] of [[1, 5], [2, 5], [3, 5], [4, 5], [2, 3], [3, 4], [5, 6]]) {
      const v = A.poolSlotsFor(peaks, m, n);
      assert.strictEqual(v % 4, 0, `level ${m} of ${n}: ${v} is not a multiple of 4`);
    }
  });

  ok('the body frame is the buffer frame, and its legacy form truncates', () => {
    // THE BODY IS IN BUFFER COORDINATES since plans/2D-backport.md B5, so the
    // frame accessor the coverage checker uses is the IDENTITY -- and it is
    // named rather than inlined precisely so this test can pin it. The day
    // the host and the kernels disagree about which frame the body is in is
    // the day the coverage gate silently stops measuring anything; that is
    // B5-5's bug (lbm_force.wgsl kept a window position while state.cx had
    // become a buffer one) one layer up.
    const st = { off_x: 10.75, off_y: -3.25 };
    const close2 = (got, want, what) => {
      close(got[0], want[0], 1e-12, what + ' x');
      close(got[1], want[1], 1e-12, what + ' y');
    };
    // Identity, and INDEPENDENT OF off -- that is the whole point of B5.
    close2(A.bodyFrameL0(34.25, 7.9, st), [34.25, 7.9], 'exact');
    close2(A.bodyFrameL0(34.25, 7.9, { off_x: 0, off_y: 0 }), [34.25, 7.9], 'off-independent');
    // The ?boxrefine=0 path still reaches the kernel through vec2<u32>(...),
    // so it truncates a FRACTIONAL tile centre by up to a cell -- compared
    // against a margin of a few, on exactly the borderline blocks a coverage
    // check is about. common_refine.wgsl's BOX_REFINE == 0u branch mirrors it.
    close2(A.bodyFrameL0Legacy(34.25, 7.9, st), [34, 7], 'legacy truncates');
    // trunc, not floor: a negative coordinate must not shift by a whole cell.
    assert.strictEqual(A.bodyFrameL0Legacy(-3.25, -0.5, st)[0], -3);
  });

  // ── the geometry predicate, and the gap the kernel still has ─────────────

  const circle = (cx, cy, r) => (x, y) => Math.hypot(x - cx, y - cy) - r;

  ok('the box predicate never misses a block a dense sample says is within the margin', () => {
    // The Lipschitz branch-and-bound is CONSERVATIVE: it may over-refine, it
    // may never miss. A dense sample over the block is the independent route.
    const sdf = circle(31.3, 27.9, 9.5);
    const margin = 3.0;
    const want = nearBodyWant(sdf, margin);
    const pool = makePool({ dims: [64, 64], rb: 8 });
    let checked = 0, over = 0;
    for (let id = 0; id < pool.nBlocks; id++) {
      const [bx, by] = pool.blockOf(id);
      const lo = [bx * pool.rb, by * pool.rb];
      const hi = [lo[0] + pool.rb, lo[1] + pool.rb];
      const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
      let dense = false;
      const N = 33;
      for (let i = 0; i <= N && !dense; i++) {
        for (let j = 0; j <= N && !dense; j++) {
          const x = lo[0] + (hi[0] - lo[0]) * i / N;
          const y = lo[1] + (hi[1] - lo[1]) * j / N;
          if (sdf(x, y) <= margin) dense = true;
        }
      }
      const box = want({ lo, hi, mid });
      assert.ok(!(dense && !box), `block ${bx},${by} is within the margin and the predicate MISSED it`);
      if (box && !dense) over++;
      checked++;
    }
    assert.ok(checked > 0);
    // Slack exists (it is a bound), but it must not be the whole domain.
    assert.ok(over < checked / 4, `over-refinement is ${over}/${checked} -- the bound has gone slack`);
  });

  ok('the branch-and-bound SUBDIVIDES -- it rejects blocks its own one-shot bound cannot', () => {
    // Without this the predicate could "accept on the bound" -- reject where
    // phi(mid) - R > margin and accept everything else -- and still pass the
    // never-misses check above, because that mutant is strictly MORE
    // conservative. Conservative is safe and expensive: at RB=8 the one-shot
    // bound is slack by the block circumradius, 5.66 L0 cells, which on this
    // fixture is the difference between refining 4 blocks and refining 12.
    //
    // Asserted EXACTLY, with no tolerance: every block the one-shot bound
    // accepts while the block is PROVABLY clear (its densely-sampled minimum
    // distance exceeds the margin) must be rejected here.
    const sdf = circle(32, 32, 5);
    const margin = 2.0;
    const want = nearBodyWant(sdf, margin);
    const pool = makePool({ dims: [64, 64], rb: 8 });
    const SQRT2_UP = 1.4142136;
    let provablyClearButBoundAccepts = 0;
    for (let id = 0; id < pool.nBlocks; id++) {
      const [bx, by] = pool.blockOf(id);
      const lo = [bx * pool.rb, by * pool.rb];
      const hi = [lo[0] + pool.rb, lo[1] + pool.rb];
      const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
      const oneShot = sdf(mid[0], mid[1]) - (hi[0] - lo[0]) / 2 * SQRT2_UP <= margin;
      if (!oneShot) continue;
      let dmin = Infinity;
      const N = 64;
      for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) {
        dmin = Math.min(dmin, sdf(lo[0] + (hi[0] - lo[0]) * i / N, lo[1] + (hi[1] - lo[1]) * j / N));
      }
      if (dmin <= margin) continue;      // genuinely near -- both must accept
      provablyClearButBoundAccepts++;
      assert.ok(!want({ lo, hi, mid }),
        `block ${bx},${by} is provably clear (min distance ${dmin.toFixed(3)} > ${margin}) and was accepted`);
    }
    assert.ok(provablyClearButBoundAccepts >= 4,
      `the fixture only produced ${provablyClearButBoundAccepts} distinguishing block(s) -- it no longer tests subdivision`);
  });

  ok('the CENTRE predicate misses blocks the box predicate catches -- the live-verified gap', () => {
    // main-cylinder-amr.js:224-236 records this as a real, live-verified bug
    // ("a tile whose CENTER just missed that margin could still have an edge
    // ... touching the body", symptom: L1's force pass stuck at a
    // bit-identical fx~-0.19 for 20,000+ steps) and records the fix as
    // ENLARGING THE MARGIN rather than fixing the test. So the gap is still
    // in the shipped kernel, and this is it, captured as an assertion instead
    // of a comment.
    const sdf = circle(32, 32, 10);
    const margin = 1.0;
    const pool = makePool({ dims: [64, 64], rb: 8 });
    const box = nearBodyWant(sdf, margin);
    const centre = nearBodyWantCentre(sdf, margin);
    const missed = [];
    for (let id = 0; id < pool.nBlocks; id++) {
      const [bx, by] = pool.blockOf(id);
      const lo = [bx * pool.rb, by * pool.rb];
      const hi = [lo[0] + pool.rb, lo[1] + pool.rb];
      const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
      const b = box({ lo, hi, mid }), c = centre({ lo, hi, mid });
      if (b && !c) missed.push([bx, by]);
      // The centre test can never be BROADER than the box test: a centre
      // within the margin is a point of the box within the margin.
      assert.ok(!(c && !b), `block ${bx},${by}: centre says near, box says clear -- impossible`);
    }
    assert.ok(missed.length > 0,
      'the two predicates agree everywhere -- this fixture no longer exercises the gap');
    // The gap is bounded by the block circumradius: RB*sqrt(2)/2 = 5.66 L0
    // cells at RB=8, which is why a margin smaller than that is where it bites.
    assert.ok(missed.length >= 4, `expected the gap on several blocks, saw ${missed.length}`);
  });

  ok('the WGSL box test types the SAME two constants amr2d.mjs does', () => {
    // The kernel and the host run different code for the same rule, so the
    // rule's two magic numbers are the seam. Parsed back OUT of the checked-in
    // WGSL, which is what fails when the two are edited apart -- the half of
    // tools/test-lattice-2d.js's design that catches a generator and a file
    // being wrong together.
    const wgsl = require('fs').readFileSync(
      path.join(__dirname, '..', 'shaders', 'common_geometry.wgsl'), 'utf8');
    const num = (re, what) => {
      const m = wgsl.match(re);
      assert.ok(m, `${what} is not declared in common_geometry.wgsl any more`);
      return parseFloat(m[1]);
    };
    assert.strictEqual(num(/const\s+SQRT2_UP\s*:\s*f32\s*=\s*([0-9.]+)/, 'SQRT2_UP'), SQRT2_UP);
    assert.strictEqual(num(/const\s+BLOCK_BB_DEPTH\s*:\s*u32\s*=\s*([0-9]+)u/, 'BLOCK_BB_DEPTH'), BLOCK_BB_DEPTH);
    // And SQRT2_UP must still be an OVER-estimate of sqrt(2): rounding it down
    // makes the Lipschitz bound false by a hair, on exactly the borderline
    // blocks the whole test exists to catch.
    assert.ok(SQRT2_UP >= Math.SQRT2, `SQRT2_UP=${SQRT2_UP} is below sqrt(2)`);
    assert.ok(SQRT2_UP < Math.SQRT2 + 1e-6, `SQRT2_UP=${SQRT2_UP} is slack, not rounded up`);
  });

  ok('the WGSL stack cannot overflow at BLOCK_BB_DEPTH', () => {
    // nearBodyBox's explicit stack is a fixed-size WGSL array, and an
    // overflowing index there is a clamp, not a crash -- it would silently
    // re-test a stale box and report a wrong answer. Each pop reuses its own
    // slot and pushes 4, and the deepest level pushes nothing, so the high
    // water mark is 3*depth + 1.
    const wgsl = require('fs').readFileSync(
      path.join(__dirname, '..', 'shaders', 'common_geometry.wgsl'), 'utf8');
    const sizes = [...wgsl.matchAll(/var\s+stack[CHD]\s*:\s*array<[^,]+,\s*(\d+)\s*>/g)].map(m => +m[1]);
    assert.strictEqual(sizes.length, 3, 'expected exactly three stack arrays');
    for (const n of sizes) {
      assert.ok(n >= 3 * BLOCK_BB_DEPTH + 1,
        `stack of ${n} is too small for depth ${BLOCK_BB_DEPTH} (needs ${3 * BLOCK_BB_DEPTH + 1})`);
    }
  });

  ok('a slot\'s quadrant is slot % 4 -- the same composition both allocators use', () => {
    // Both allocators build the slot as `quadIdx*4 + quadrant`:
    // shaders/amr_manage_pool.wgsl's refine() and main-amr.js's
    // debugActivateBlock. So the quadrant is recoverable from the index, and
    // the pool's per-slot quadrant buffer was storing a constant -- which is
    // what let amr_manage_pool.wgsl drop a binding it had no room for (it sat
    // at exactly maxStorageBuffersPerShaderStage; see CLAUDE.md).
    //
    // Checked against the COMPOSITION rather than against `slot % 4` restated,
    // so this is the two routes agreeing and not one route twice.
    for (let quadIdx = 0; quadIdx < 6; quadIdx++) {
      for (let quadrant = 0; quadrant < 4; quadrant++) {
        const slot = quadIdx * 4 + quadrant;
        assert.strictEqual(quadrantOfSlot(slot), quadrant,
          `slot ${slot} (quad ${quadIdx}, quadrant ${quadrant})`);
      }
    }
    // And quadrant 0's slot is the one the coarsen pass selects on, which is
    // the single use that became arithmetic.
    for (let quadIdx = 0; quadIdx < 6; quadIdx++) {
      assert.strictEqual(quadrantOfSlot(quadIdx * 4), 0);
      for (let q = 1; q < 4; q++) assert.notStrictEqual(quadrantOfSlot(quadIdx * 4 + q), 0);
    }
  });

  ok('refineWhere assigns slots deterministically and refuses to overflow the pool', () => {
    const pool = makePool({ dims: [64, 64], rb: 8, maxSlots: 8 });
    const sdf = circle(32, 32, 4);
    const a = refineNearBody(pool, sdf, 1.0);
    const b = refineNearBody(pool, sdf, 1.0);
    assert.deepStrictEqual([...a.blockSlot], [...b.blockSlot], 'slot assignment is not deterministic');
    assert.ok(a.activeSlots > 0 && a.activeSlots <= 8);
    // Slots are handed out in ascending block order, so slotToBlock is sorted.
    const used = [...a.slotToBlock].filter(v => v >= 0);
    assert.deepStrictEqual(used, [...used].sort((x, y) => x - y));
    assert.throws(() => refineWhere(makePool({ dims: [64, 64], rb: 8, maxSlots: 2 }), () => true),
      /more than maxSlots/);
  });

  // ── 2:1 balance ─────────────────────────────────────────────────────────

  // Level-1 blocks over a 128x128 L0 domain at RB=8: nb = [16,16].
  const BASE = makePool({ dims: [128, 128], rb: 8 });
  const nbAt = (m) => nbAtLevel(BASE, m);

  // A tower: one level-1 block, its full level-2 quad, and one of those
  // children's full level-3 quad. Legal by construction.
  function tower() {
    const s = [null, new Set(), new Set(), new Set()];
    s[1].add(key([8, 8]));
    for (let k = 0; k < 4; k++) s[2].add(key([16 + (k & 1), 16 + ((k >> 1) & 1)]));
    for (let k = 0; k < 4; k++) s[3].add(key([32 + (k & 1), 32 + ((k >> 1) & 1)]));
    return s;
  }

  ok('check21Balance reports a REAL depth-3-next-to-depth-0 gap', () => {
    // A violating input, which is the only way to know the checker checks.
    const sets = tower();
    const r = check21Balance(sets, nbAt, { levels: 4 });
    assert.ok(!r.ok, 'a bare tower is 2:1-unbalanced and the checker said it was fine');
    assert.ok(r.violations.some(v => v.level === 3 && v.nDepth === 0),
      `expected a level-3 leaf facing depth 0, got ${JSON.stringify(r.violations.slice(0, 3))}`);
  });

  ok('check21Balance is clean once the closure has run', () => {
    const sets = tower();
    const { sets: balanced } = cascade21(sets, nbAt, { levels: 4 });
    const r = check21Balance(balanced, nbAt, { levels: 4 });
    assert.ok(r.ok, `cascade21 left ${r.violations.length} violation(s): ${JSON.stringify(r.violations[0])}`);
  });

  ok('the balance walk follows the SHARED EDGE: far side is legal, near side is not', () => {
    // The false positive main-amr.js:2946-2960 records a first version
    // producing: comparing each tile's deepest descendant ANYWHERE against
    // its neighbour's deepest ANYWHERE. True 2:1 balance is a property of
    // adjacent CELLS, so the walk has to follow the SHARED edge.
    //
    // A is a LEAF at level 1. B is its +x neighbour at level 1 with a full
    // level-2 quad, and ONE of those level-2 children carries a level-3 quad.
    // Which one is the entire experiment: on B's FAR (+x) half the A/B edge
    // only ever touches level-2 cells and A is fine; on B's NEAR (-x) half it
    // touches level 3 and A is a genuine violation. A checker that takes
    // whole-tile depth cannot tell those apart, and reports both.
    const A = [8, 8], B = [9, 8];
    const build = (deepChild) => {
      const s = [null, new Set(), new Set(), new Set()];
      s[1].add(key(A)); s[1].add(key(B));
      for (let k = 0; k < 4; k++) s[2].add(key([B[0] * 2 + (k & 1), B[1] * 2 + ((k >> 1) & 1)]));
      for (let k = 0; k < 4; k++) s[3].add(key([deepChild[0] * 2 + (k & 1), deepChild[1] * 2 + ((k >> 1) & 1)]));
      return s;
    };
    const atA = (sets) => check21Balance(sets, nbAt, { levels: 4 })
      .violations.filter(v => v.level === 1 && v.block[0] === A[0] && v.block[1] === A[1]
                              && v.axis === 0 && v.dir === 1);
    // B's level-2 children are x = 18 (near A) and x = 19 (far from A).
    assert.strictEqual(atA(build([19, 16])).length, 0,
      'A was reported against a level-3 quad on B\'s FAR side, which its edge never touches');
    assert.strictEqual(atA(build([18, 16])).length, 1,
      'A faces a level-3 quad across the shared edge and was NOT reported');
  });

  ok('THE BLOCK GRID IS PERIODIC: the seam is a neighbour, not an edge', () => {
    // Every kernel here wraps the block grid (shaders/amr_step1.wgsl's
    // bxm/bxp/bym/byp), because the 2D domain is periodic and the body pans
    // through it. A checker that treated the seam as a boundary would invent
    // violations there and miss real ones -- and it would do so ONLY at
    // x = 0 and x = nb-1, which is exactly where no centred fixture looks.
    const levels = 3;
    const build = (withSeamNeighbour) => {
      const s = [null, new Set(), new Set()];
      s[1].add(key([0, 0]));
      if (withSeamNeighbour) s[1].add(key([nbAt(1)[0] - 1, 0]));
      for (let k = 0; k < 4; k++) s[2].add(key([k & 1, (k >> 1) & 1]));
      return s;
    };
    const across = (sets) => check21Balance(sets, nbAt, { levels })
      .violations.filter(v => v.level === 2 && v.block[0] === 0 && v.axis === 0 && v.dir === -1);
    // With the level-1 tile present on the far side of the seam, the level-2
    // tile at x=0 faces depth 1 and is balanced.
    assert.strictEqual(across(build(true)).length, 0,
      'the seam neighbour is present and the level-2 tile at x=0 was reported anyway');
    // Without it, both level-2 tiles on the x=0 column face depth 0 across the
    // seam, and both are real violations.
    const v = across(build(false));
    assert.strictEqual(v.length, 2, 'a genuine gap across the seam was not reported');
    for (const hit of v) {
      assert.strictEqual(hit.neighbour[0], nbAt(2)[0] - 1, 'the neighbour was not wrapped');
      assert.strictEqual(hit.nDepth, 0);
    }
  });

  ok('the closure wraps too -- a quad at the origin forces parents across the seam', () => {
    const levels = 3;
    const want = [null, new Set(), new Set()];
    for (let k = 0; k < 4; k++) want[2].add(key([k & 1, (k >> 1) & 1]));
    const { sets } = cascade21(want, nbAt, { levels });
    const last = nbAt(1)[0] - 1;
    // Neighbours of level-2 block (0,0) include (-1,-1) -> (31,31), whose
    // parent is (15,15). All four seam-corner parents must be present.
    for (const p of [[last, last], [last, 0], [0, last], [0, 0]]) {
      assert.ok(sets[1].has(key(p)), `the closure did not force parent ${key(p)} across the seam`);
    }
    assert.ok(check21Balance(sets, nbAt, { levels }).ok);
    assert.ok(checkRingParentCoverage(sets, nbAt, { levels }).ok);
  });

  ok('corner violations are reported SEPARATELY and do not gate ok', () => {
    // main-amr.js:3061-3090's deliberate call: corner balance is a
    // requirement of the ghost-free path, not of the default one.
    const s = [null, new Set(), new Set()];
    s[1].add(key([8, 8]));
    for (let k = 0; k < 4; k++) s[2].add(key([16 + (k & 1), 16 + ((k >> 1) & 1)]));
    // Diagonal neighbour of level-2 block (16,16) is (15,15), whose ancestor
    // chain is empty -> depth 0, a |2-0| corner gap. Its EDGE neighbours are
    // supplied by level 1, so edge balance is a separate question.
    const r = check21Balance(s, nbAt, { levels: 3 });
    assert.ok(r.cornerViolations.length > 0, 'the corner gap was not reported at all');
    assert.strictEqual(typeof r.ok, 'boolean');
    assert.strictEqual(typeof r.cornerOk, 'boolean');
    assert.strictEqual(r.cornerOk, false);
    assert.ok(r.violations.every(v => v.level !== undefined));
  });

  // ── ring parent coverage ────────────────────────────────────────────────

  ok('ring parent coverage catches the DIAGONAL parent an edge-only closure leaves out', () => {
    // The concrete case from checkRingParentCoverage's header: child block
    // (2p, 2p) has diagonal neighbour (2p-1, 2p-1) whose parent is (p-1,p-1),
    // which the four edges never supply.
    const p = 8;
    const s = [null, new Set(), new Set()];
    // Level 2 is the full quad of parent (p, p): blocks (2p..2p+1)^2, whose
    // 9-neighbourhoods span (2p-1 .. 2p+2) and therefore need every parent in
    // (p-1 .. p+1)^2. Supply all NINE except the one corner an edge-only
    // closure would never produce.
    for (let j = p - 1; j <= p + 1; j++) {
      for (let i = p - 1; i <= p + 1; i++) {
        if (i === p - 1 && j === p - 1) continue;
        s[1].add(key([i, j]));
      }
    }
    for (let k = 0; k < 4; k++) s[2].add(key([2 * p + (k & 1), 2 * p + ((k >> 1) & 1)]));
    const r = checkRingParentCoverage(s, nbAt, { levels: 3 });
    assert.ok(!r.ok, 'the diagonal parent was missing and the checker said coverage was fine');
    // EXACTLY that parent, and nothing else -- otherwise this would pass on a
    // fixture that was simply under-refined everywhere.
    const missing = new Set(r.violations.map(v => key(v.parent)));
    assert.deepStrictEqual([...missing], [key([p - 1, p - 1])],
      `expected only parent (${p - 1},${p - 1}) to be missing, got ${[...missing]}`);
    // Adding it closes the gap, and nothing else was needed.
    s[1].add(key([p - 1, p - 1]));
    assert.ok(checkRingParentCoverage(s, nbAt, { levels: 3 }).ok,
      'the diagonal parent was supplied and coverage is still incomplete');
  });

  ok('cascade21 output always has full ring parent coverage', () => {
    const { sets } = cascade21(tower(), nbAt, { levels: 4 });
    const r = checkRingParentCoverage(sets, nbAt, { levels: 4 });
    assert.ok(r.ok, `cascade21 left ${r.violations.length} ring parent(s) unallocated`);
    assert.ok(r.required > 0, 'nothing was checked -- the fixture has no pool-parent level');
  });

  // ── the closure ─────────────────────────────────────────────────────────

  ok('cascade21 is the IDENTITY at levels=2 -- which is what "provably" means', () => {
    // A level-1 block's parent is the dense L0 grid, present everywhere, so
    // there is nothing to require and nothing to add. Twenty random sets.
    let rng = 12345;
    const rand = () => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (let t = 0; t < 20; t++) {
      const want = [null, new Set()];
      for (let i = 0; i < 25; i++) {
        want[1].add(key([Math.floor(rand() * 16), Math.floor(rand() * 16)]));
      }
      const { sets, forced } = cascade21(want, nbAt, { levels: 2 });
      assert.deepStrictEqual(sorted(sets[1]), sorted(want[1]), `set changed on trial ${t}`);
      assert.strictEqual(forced.length, 0, `${forced.length} block(s) forced at levels=2`);
    }
  });

  ok('cascade21 output is QUAD-COMPLETE from level 2 down', () => {
    // A want for one child is a want for the parent's refinement, so the
    // input is completed rather than rejected. check21Balance's hasChild
    // leans on this: a parent holding only quadrant (1,0) would read as a
    // LEAF. 3D's first cascade added single blocks and the checker was right
    // to complain.
    const want = [null, new Set(), new Set(), new Set()];
    want[3].add(key([33, 32]));      // a single, non-(0,0) child
    want[2].add(key([9, 12]));       // another
    const { sets } = cascade21(want, nbAt, { levels: 4 });
    for (let m = 2; m <= 3; m++) {
      for (const k of sets[m]) {
        const b = k.split(',').map(Number);
        const p = parentOfBlock(b);
        for (let q = 0; q < 4; q++) {
          const sib = key([p[0] * 2 + (q & 1), p[1] * 2 + ((q >> 1) & 1)]);
          assert.ok(sets[m].has(sib), `level ${m}: ${k} is present without sibling ${sib}`);
        }
      }
    }
    assert.ok(check21Balance(sets, nbAt, { levels: 4 }).ok);
  });

  ok('cascade21 is IDEMPOTENT -- one deepest-first sweep IS the fixed point', () => {
    // Checked, not argued. This is the property that lets the closure replace
    // main-amr.js:2383-2418's fixed-point loop, whose iteration count is a
    // guess and which is documented as sometimes running out
    // (main-amr.js:3585: "balance was still spreading outward when the loop
    // ran out").
    for (const want of [tower(), (() => {
      const s = [null, new Set(), new Set(), new Set()];
      s[3].add(key([40, 40])); s[3].add(key([7, 60])); s[2].add(key([3, 3]));
      return s;
    })()]) {
      const once = cascade21(want, nbAt, { levels: 4 });
      const twice = cascade21(once.sets, nbAt, { levels: 4 });
      for (let m = 1; m <= 3; m++) {
        assert.deepStrictEqual(sorted(twice.sets[m]), sorted(once.sets[m]), `level ${m} moved on a second sweep`);
      }
      assert.strictEqual(twice.forced.length, 0, 'a second sweep still forced blocks');
    }
  });

  ok('cascade21 is MINIMAL -- removing any forced quad breaks the tree', () => {
    // Without this, "refine a halo to be safe" passes every other check: a
    // superset of a balanced set is still balanced.
    const { sets, forced } = cascade21(tower(), nbAt, { levels: 4 });
    const levels = 4;
    // Group the forced blocks into the units the allocator actually grants:
    // a quad at level >= 2, a single block at level 1.
    const units = new Map();
    for (const f of forced) {
      const id = f.level >= 2 ? `${f.level}:q${f.quad[0]},${f.quad[1]}` : `1:${key(f.block)}`;
      if (!units.has(id)) units.set(id, { level: f.level, blocks: [] });
      units.get(id).blocks.push(f.block);
    }
    assert.ok(units.size > 0, 'nothing was forced -- the fixture does not exercise minimality');
    const parented = (s) => {
      for (let m = 2; m < levels; m++) {
        for (const k of s[m] || []) {
          const p = parentOfBlock(k.split(',').map(Number));
          if (!(s[m - 1] && s[m - 1].has(key(p)))) return false;
        }
      }
      return true;
    };
    for (const [id, u] of units) {
      const trial = sets.map((s, m) => (m === 0 ? null : new Set(s)));
      for (const b of u.blocks) trial[u.level].delete(key(b));
      const stillFine = check21Balance(trial, nbAt, { levels }).ok
        && checkRingParentCoverage(trial, nbAt, { levels }).ok
        && parented(trial);
      assert.ok(!stillFine, `unit ${id} was forced but removing it breaks nothing -- the cascade over-refines`);
    }
  });

  ok('a REAL nested shell -- three independently-evaluated geometry levels -- is unbalanced, then is not', () => {
    // The shape a geometry-forced criterion actually produces, and unbalanced
    // BY CONSTRUCTION because no per-level distance test knows about the
    // level above it. This is the case plans/2D-backport.md B2 exists for.
    const levels = 4;
    const sdf = circle(64, 64, 12);
    const want = [null, new Set(), new Set(), new Set()];
    for (let m = 1; m < levels; m++) {
      const pool = poolAtLevel(BASE, m);
      const cell = cellSizeL0AtLevel(m - 1);            // the PARENT cell size, in L0 units
      const margin = 4 * cell;                          // a per-level margin, evaluated independently
      const near = nearBodyWant(sdf, margin);
      for (let id = 0; id < pool.nBlocks; id++) {
        const [bx, by] = pool.blockOf(id);
        const lo = [bx * pool.rb * cell, by * pool.rb * cell];
        const hi = [lo[0] + pool.rb * cell, lo[1] + pool.rb * cell];
        const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
        if (near({ lo, hi, mid })) want[m].add(key([bx, by]));
      }
    }
    const before = check21Balance(want, nbAt, { levels });
    assert.ok(!before.ok,
      'the independently-evaluated shells came out balanced -- this fixture is not exercising the closure');
    const { sets, counts, forced } = cascade21(want, nbAt, { levels });
    const after = check21Balance(sets, nbAt, { levels });
    assert.ok(after.ok, `cascade21 left ${after.violations.length} violation(s)`);
    assert.ok(checkRingParentCoverage(sets, nbAt, { levels }).ok);
    // The closure only ever ADDS.
    for (let m = 1; m < levels; m++) {
      for (const k of want[m]) assert.ok(sets[m].has(k), `level ${m}: the closure dropped ${k}`);
      assert.ok(counts[m] >= want[m].size);
    }
    // THE FINEST LEVEL IS THE CRITERION'S OWN ANSWER, and the closure must not
    // second-guess it: the only thing it may add there is a QUAD COMPLETION,
    // never a 2:1 or ring requirement (those always land one level up). A
    // level-m want forces presence at m-1, so nothing can force the deepest
    // level for any other reason -- asserted rather than argued, because if it
    // ever did, the geometry criterion would have been quietly overridden.
    for (const f of forced) {
      if (f.level !== levels - 1) continue;
      assert.ok(f.because && f.because.siblingOf !== undefined,
        `the closure added ${key(f.block)} to the FINEST level for reason ${JSON.stringify(f.because)}`);
    }
  });

  // ── geometry coverage ───────────────────────────────────────────────────

  ok('checkGeometryCoverage reports a hole at the finest level, and takes the predicate', () => {
    const levels = 3;
    const sdf = circle(64, 64, 10);
    const margin = 2.0;
    const finest = levels - 1;
    const pool = poolAtLevel(BASE, finest);
    const cell = cellSizeL0AtLevel(finest - 1);
    const want = nearBodyWant(sdf, margin);
    const sets = [null, new Set(), new Set()];
    const covered = [];
    for (let id = 0; id < pool.nBlocks; id++) {
      const [bx, by] = pool.blockOf(id);
      const lo = [bx * pool.rb * cell, by * pool.rb * cell];
      const hi = [lo[0] + pool.rb * cell, lo[1] + pool.rb * cell];
      const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
      if (want({ lo, hi, mid })) { sets[finest].add(key([bx, by])); covered.push([bx, by]); }
    }
    assert.ok(covered.length > 0, 'the fixture refines nothing');
    assert.ok(checkGeometryCoverage(BASE, sets, want, { levels }).ok, 'a fully covered body reported a hole');
    // Punch one out.
    sets[finest].delete(key(covered[0]));
    const r = checkGeometryCoverage(BASE, sets, want, { levels });
    assert.ok(!r.ok, 'a missing body-adjacent tile was not reported');
    assert.strictEqual(r.violations.length, 1);
    assert.deepStrictEqual(r.violations[0].block, covered[0]);
    // The CENTRE predicate asks a weaker question, so it must require no more
    // than the box predicate does -- which is exactly why swapping them is a
    // change of meaning and not a refactor.
    const centreR = checkGeometryCoverage(BASE, sets, nearBodyWantCentre(sdf, margin), { levels });
    assert.ok(centreR.required <= r.required,
      'the centre predicate demanded MORE tiles than the box predicate -- impossible');
  });

  // ── the Dupuis-Chopard transfer factor (plans/2D-backport.md B1) ──────────
  //
  // Rule 1, THE INDEPENDENT ROUTE, matters more here than anywhere else in
  // this file, because the defect being fixed was a plausible-looking closed
  // form that had survived for years. Re-deriving the same algebra alongside
  // it would prove nothing. So the route below COMPOSES THREE ELEMENTARY
  // FACTS instead -- BGK post-collision, the textbook PRE-collision
  // Dupuis-Chopard relation, and BGK again at the finer level -- and never
  // writes the answer down:
  //
  //   fneq*_c  = (1 - 1/tau_c) * q         decollide is its inverse
  //   q_f      = (1/2)(tau_f/tau_c) * q    the relation, PRE-collision
  //   fneq*_f  = (1 - 1/tau_f) * q_f
  //
  // The claimed factor is fneq*_f / fneq*_c, for an arbitrary pre-collision
  // q. Agreement is then evidence, not a tautology.
  const routePost = (tauC, tauF, q = 0.137) => {
    const fneqPostC = (1 - 1 / tauC) * q;          // BGK at the coarse level
    const qF = 0.5 * (tauF / tauC) * q;            // Dupuis-Chopard, PRE
    const fneqPostF = (1 - 1 / tauF) * qF;         // BGK at the fine level
    return fneqPostF / fneqPostC;
  };
  const TAU_CASES = [0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 1.2, 1.7, 2.5];
  const tauPair = (tauC) => [tauC, 2 * tauC - 0.5];

  ok('dcRescaleCoarseToFine matches decollide -> PRE -> recollide', () => {
    for (const tauC of TAU_CASES) {
      const [c, f] = tauPair(tauC);
      close(dcRescaleCoarseToFine(c, f), routePost(c, f), 1e-12, `tau_c=${c}`);
      // and it does not depend on the q the route was probed with.
      close(routePost(c, f, 1), routePost(c, f, -7.25), 1e-12, `q-dependence at tau_c=${c}`);
    }
  });

  ok('dcRescaleFineToCoarse is the exact inverse', () => {
    for (const tauC of TAU_CASES) {
      const [c, f] = tauPair(tauC);
      const p = dcRescaleCoarseToFine(c, f) * dcRescaleFineToCoarse(c, f);
      // One ulp, not zero: the two are algebraically reciprocal but each is
      // evaluated as its own division. A round trip through the interface is
      // inert to that, which is the property being claimed.
      assert.ok(Math.abs(p - 1) <= 2 * Number.EPSILON,
        `round trip at tau_c=${c} came back ${p}`);
    }
  });

  ok('the legacy pair is the PRE-collision relation, and is also an inverse pair', () => {
    for (const tauC of TAU_CASES) {
      const [c, f] = tauPair(tauC);
      close(dcRescaleCoarseToFinePre(c, f), 0.5 * f / c, 1e-15, `pre c2f at tau_c=${c}`);
      const p = dcRescaleCoarseToFinePre(c, f) * dcRescaleFineToCoarsePre(c, f);
      assert.ok(Math.abs(p - 1) <= 2 * Number.EPSILON, `pre round trip at tau_c=${c}`);
    }
  });

  ok('the two factors differ in MAGNITUDE AND SIGN at tau = 0.8', () => {
    // The number plans/2D-backport.md B1 quotes, and the reason this is not a
    // refinement of the old factor: at the channel/TGV default they do not
    // even agree on which way the non-equilibrium stress points.
    close(dcRescaleCoarseToFine(0.8, 1.1), -0.25, 1e-12, 'post at tau=0.8');
    close(dcRescaleCoarseToFinePre(0.8, 1.1), 0.6875, 1e-12, 'pre at tau=0.8');
    // ...and near tau = 0.5, where the card and cylinder pages live, they
    // very nearly agree. That is WHY this shipped: the pages with a Cd/St
    // number could barely see it.
    const [c, f] = tauPair(0.5044);
    assert.ok(Math.abs(dcRescaleCoarseToFine(c, f) / dcRescaleCoarseToFinePre(c, f) - 1) < 0.02,
      'the two factors were expected to agree to ~2% near tau=0.5');
  });

  // Rule 2, MUTATION-CHECK. Each of these is a slip someone could plausibly
  // make (or did): dropping the 1/n, the direction reversed, the historical
  // pre-collision form, and a sign slip on the shift.
  ok('plausible slips in the factor are all caught by the independent route', () => {
    const mutants = {
      'dropped the 1/n': (c, f) => (f - 1) / (c - 1),
      'direction reversed': (c, f) => 0.5 * (c - 1) / (f - 1),
      'the historical PRE-collision form': (c, f) => 0.5 * f / c,
      'shift sign slipped': (c, f) => 0.5 * (f + 1) / (c + 1),
      'shifted only the numerator': (c, f) => 0.5 * (f - 1) / c,
    };
    for (const [name, m] of Object.entries(mutants)) {
      const caught = TAU_CASES.some((tauC) => {
        const [c, f] = tauPair(tauC);
        return Math.abs(m(c, f) - routePost(c, f)) > 1e-9;
      });
      assert.ok(caught, `mutant "${name}" was NOT caught at any tau in the sweep`);
    }
  });

  // AGREEMENT WITH THE SHADER, which is the thing actually at risk -- the
  // same reason tools/test-f-pack.js re-implements common_fpack.wgsl's
  // addressing rather than only checking the host against itself. A host
  // module that is right while the WGSL is wrong buys nothing: the GPU does
  // every transfer. So both WGSL forms are lifted out of the shader source
  // and evaluated against the host's.
  //
  // A FAILED EXTRACTION IS A FAILURE, NOT A SKIP. This project has collected
  // four gates that silently became vacuous; a regex that stops matching
  // after an innocuous edit is exactly that shape, so it fails loudly and
  // the next person re-points it.
  const wgslReturns = (file, fnName) => {
    const src = require('fs').readFileSync(path.join(__dirname, '..', 'shaders', file), 'utf8');
    const fn = new RegExp(`fn ${fnName}\\([^)]*\\)\\s*->\\s*f32\\s*\\{([\\s\\S]*?)\\n\\}`).exec(src);
    assert.ok(fn, `could not find fn ${fnName} in shaders/${file} -- re-point this extraction`);
    const rets = [...fn[1].matchAll(/return ([^;]+);/g)].map((m) => m[1]);
    assert.strictEqual(rets.length, 2,
      `expected exactly 2 returns in ${fnName} (DC_PRE branch, then default), got ${rets.length}`);
    // WGSL f32 literals (0.5f, 1.0f) -> JS numbers. Nothing else in these
    // expressions differs from JS.
    return rets.map((e) => new Function('tauCoarse', 'tauFine',
      `return ${e.replace(/(\d)f\b/g, '$1')};`));
  };

  ok('shaders/common_interp.wgsl states the same two coarse->fine factors', () => {
    const [pre, post] = wgslReturns('common_interp.wgsl', 'dcRescaleCoarseToFine');
    for (const tauC of TAU_CASES) {
      const [c, f] = tauPair(tauC);
      close(post(c, f), dcRescaleCoarseToFine(c, f), 1e-12, `WGSL post at tau_c=${c}`);
      close(pre(c, f), dcRescaleCoarseToFinePre(c, f), 1e-12, `WGSL pre at tau_c=${c}`);
    }
  });

  ok('shaders/common_average.wgsl states the same two fine->coarse factors', () => {
    const [pre, post] = wgslReturns('common_average.wgsl', 'dcRescaleFineToCoarse');
    for (const tauC of TAU_CASES) {
      const [c, f] = tauPair(tauC);
      close(post(c, f), dcRescaleFineToCoarse(c, f), 1e-12, `WGSL post at tau_c=${c}`);
      close(pre(c, f), dcRescaleFineToCoarsePre(c, f), 1e-12, `WGSL pre at tau_c=${c}`);
    }
  });

  // Rule 3, RUN THE CHECKER ON INPUTS THAT VIOLATE THE INVARIANT.
  ok('tauChainSingularity finds the singular level, and only when there is one', () => {
    // tau_m = 1 exactly when tau_0 = 1/2 + 2^-(m+1). Every one of these is a
    // value a user can type, and the last is inside index-amr.html's own
    // slider range.
    for (const [tau0, level] of [[1.0, 0], [0.75, 1], [0.625, 2], [0.5625, 3]]) {
      const hit = tauChainSingularity(tau0, level + 1);
      assert.ok(hit, `tau0=${tau0} should be singular at level ${level}`);
      assert.strictEqual(hit.level, level);
      close(hit.tau, 1, 1e-12, `level ${level} tau`);
      // ...and a hierarchy that stops ABOVE it is clean, which is what makes
      // this a per-config answer rather than a blanket ban on a tau.
      assert.strictEqual(tauChainSingularity(tau0, level), null,
        `tau0=${tau0} at ${level} level(s) has no singular level and was flagged`);
      // The message has to name the level -- a refusal that says only "bad
      // tau" sends the reader to the wrong knob on a four-level hierarchy.
      assert.ok(tauSingularityMessage(hit).includes(`level ${level}`),
        'the refusal text does not name the level it found');
    }
    // The shipped defaults are clear of it at every depth this project runs.
    for (const tau0 of [0.5044, 0.5087, 0.5256, 0.8]) {
      assert.strictEqual(tauChainSingularity(tau0, 5), null,
        `tau0=${tau0} was flagged singular and should not be`);
    }
  });

  ok('the singularity band is a band, not an equality test', () => {
    // Exact equality would never fire on a float chain: what poisons the
    // transfer is the neighbourhood, where the factor is finite but enormous.
    const justInside = 0.75 + 0.4 * TAU_UNITY_BAND / 2;   // level 1 lands inside
    assert.ok(tauChainSingularity(justInside, 2), 'a tau just inside the band was not caught');
    const wellOutside = 0.75 + 4 * TAU_UNITY_BAND;
    assert.strictEqual(tauChainSingularity(wellOutside, 2), null,
      'a tau well outside the band was refused');
  });


  // ── D0: the deterministic slot handout ───────────────────────────────────
  //
  // MUTATION-CHECKED, per this file's rule 2. Every check below was run
  // against a deliberately broken grantAssignment/releaseAssignment and the
  // mutant it catches is named beside it. The one that matters most is
  // PERMUTATION INVARIANCE: it is the entire property being bought, and a
  // rule that merely "looks ordered" passes everything else.

  ok('grant is invariant under the order candidates arrive in', () => {
    // MUTANT: drop the sort (serve in arrival order) -> this is the only
    // check that fires, which is exactly why it exists.
    const freeCount = 6;
    const ids = [41, 7, 19, 3, 28];
    const ref = JSON.stringify(grantAssignment({ candidates: ids, freeCount }));
    const perms = [
      [3, 7, 19, 28, 41], [41, 28, 19, 7, 3], [19, 3, 41, 28, 7], [7, 41, 3, 19, 28],
    ];
    for (const p of perms) {
      assert.strictEqual(JSON.stringify(grantAssignment({ candidates: p, freeCount })), ref,
        `arrival order ${p.join(',')} changed the assignment`);
    }
  });

  ok('grant pops down from the top of the stack, one index each', () => {
    // MUTANT: freeIndex = i (pop from the bottom) -> caught here.
    // MUTANT: count-- after the read (off by one) -> caught here.
    const { granted, refused, freeCount } = grantAssignment({ candidates: [5, 2, 9], freeCount: 4 });
    assert.deepStrictEqual(granted.map(g => g.id), [2, 5, 9], 'not served in id order');
    assert.deepStrictEqual(granted.map(g => g.freeIndex), [3, 2, 1], 'wrong free-list indices');
    assert.deepStrictEqual(refused, [], 'refused something with slots to spare');
    assert.strictEqual(freeCount, 1, 'free count not decremented once per grant');
  });

  ok('a starved pool starves the HIGHEST ids, and never over-grants', () => {
    // MUTANT: sort descending -> the wrong ids survive, caught here.
    // MUTANT: `if (count < 0)` -> grants one too many, caught here.
    const { granted, refused, freeCount } = grantAssignment({ candidates: [8, 1, 5, 3, 9], freeCount: 2 });
    assert.deepStrictEqual(granted.map(g => g.id), [1, 3], 'starvation did not favour low ids');
    assert.deepStrictEqual(refused, [5, 8, 9], 'wrong candidates refused');
    assert.strictEqual(freeCount, 0, 'free count went past empty');
    for (const g of granted) assert.ok(g.freeIndex >= 0 && g.freeIndex < 2, `freeIndex ${g.freeIndex} out of the valid region`);
  });

  ok('an empty pool grants nothing and is not an error', () => {
    const r = grantAssignment({ candidates: [4, 1], freeCount: 0 });
    assert.deepStrictEqual(r.granted, [], 'granted from an empty free list');
    assert.deepStrictEqual(r.refused, [1, 4], 'refusals not in id order');
    assert.strictEqual(r.freeCount, 0, 'free count went negative');
  });

  ok('every granted free-list index is distinct', () => {
    // The property a race cannot guarantee and the one corruption would be
    // silent: two blocks pointed at one quad.
    const ids = [31, 4, 17, 22, 9, 40, 12];
    const { granted } = grantAssignment({ candidates: ids, freeCount: ids.length });
    const seen = new Set(granted.map(g => g.freeIndex));
    assert.strictEqual(seen.size, granted.length, 'two candidates were given the same free-list index');
  });

  ok('release pushes in id order from the top', () => {
    // MUTANT: write at freeCount + n - 1 - j -> caught here.
    const { writes, freeCount } = releaseAssignment({ releases: [12, 3, 7], freeCount: 5 });
    assert.deepStrictEqual(writes.map(w => w.id), [3, 7, 12], 'not released in id order');
    assert.deepStrictEqual(writes.map(w => w.freeIndex), [5, 6, 7], 'wrong push indices');
    assert.strictEqual(freeCount, 8, 'free count not incremented once per release');
  });

  ok('release is invariant under arrival order too', () => {
    const ref = JSON.stringify(releaseAssignment({ releases: [12, 3, 7], freeCount: 5 }));
    for (const p of [[3, 7, 12], [12, 7, 3], [7, 12, 3]]) {
      assert.strictEqual(JSON.stringify(releaseAssignment({ releases: p, freeCount: 5 })), ref,
        `arrival order ${p.join(',')} changed the release`);
    }
  });

  ok('grant then release restores the free count', () => {
    const ids = [6, 2, 11];
    const g = grantAssignment({ candidates: ids, freeCount: 5 });
    const r = releaseAssignment({ releases: g.granted.map(x => x.id), freeCount: g.freeCount });
    assert.strictEqual(r.freeCount, 5, 'a grant/release round trip leaked or invented free entries');
    // The indices touched on the way out are the ones touched on the way back.
    assert.deepStrictEqual(
      g.granted.map(x => x.freeIndex).sort((a, b) => a - b),
      r.writes.map(x => x.freeIndex).sort((a, b) => a - b),
      'release wrote to a different region than grant read');
  });

  ok('a refused candidate consumes nothing', () => {
    // The failure this guards: counting refusals against the free list, which
    // would leave the count wrong for the NEXT round rather than this one --
    // a defect that only shows up a refine round later.
    // The extras must genuinely EXCEED capacity or they are grants, not
    // refusals -- the first version of this check used freeCount 3 with five
    // candidates, where the third one is served and the premise is simply
    // wrong. The test caught that before the code did.
    const a = grantAssignment({ candidates: [1, 2], freeCount: 2 });
    const b = grantAssignment({ candidates: [1, 2, 99, 98, 97], freeCount: 2 });
    assert.strictEqual(a.freeCount, b.freeCount,
      'refusals changed the free count');
    assert.deepStrictEqual(a.granted, b.granted.slice(0, 2),
      'refusals disturbed the grants that did succeed');
  });


  // ── U0: the uniform level model ──────────────────────────────────────────

  ok('the root has no ring; every level below has two cells of one', () => {
    assert.strictEqual(ghostDepthAtLevel(0), 0, 'the root was given a ring');
    for (const m of [1, 2, 3, 4]) assert.strictEqual(ghostDepthAtLevel(m), GHOST, `level ${m}`);
    assert.strictEqual(tileSideAtLevel(0, 8), 16, 'root tile side');
    assert.strictEqual(tileSideAtLevel(1, 8), 20, 'level-1 tile side');
    // The whole point of the root having no ring: its storage is exactly the
    // domain, not the domain plus padding.
    const dims = { W: 512, H: 512 };
    const [nbx, nby] = blockGridAtLevel(dims, 0, 8);
    assert.strictEqual(nbx * nby * tileCellsAtLevel(0, 8) ** 2, dims.W * dims.H,
      'root tiles do not tile the domain exactly');
  });

  ok('the block grid agrees with the convention it replaces, at every level', () => {
    // INDEPENDENT ROUTE (rule 1): today's grid is "L1 is 1:1 with L0 blocks,
    // then double per level", derived from W/RB. The new one is "the domain
    // over a whole root tile, then double per level", derived from W/(2*RB).
    // They share no arithmetic; agreeing is evidence.
    const dims = { W: 512, H: 256 }, rb = 8;
    const legacy = (m) => [(dims.W / rb) * (1 << (m - 1)), (dims.H / rb) * (1 << (m - 1))];
    for (const m of [1, 2, 3, 4]) {
      assert.deepStrictEqual(blockGridAtLevel(dims, m, rb), legacy(m),
        `level ${m} disagrees with the pre-U0 convention`);
    }
    // And it extends DOWN, which the old one could not express at all.
    assert.deepStrictEqual(blockGridAtLevel(dims, 0, rb), [32, 16], 'root grid');
  });

  ok('a domain that does not tile is refused, not rounded', () => {
    assert.throws(() => blockGridAtLevel({ W: 500, H: 512 }, 0, 8), /does not divide/,
      'a ragged domain was silently accepted');
  });

  ok('fine <-> parent cell is an exact inverse across the ring too', () => {
    // ROUND TRIP, not eyeballing -- the same discipline dense-to-amr.js and
    // field-reconstruct.js are scored by.
    const m = 1, rb = 8;
    for (let p = -1; p <= rb; p++) {
      const [a, b] = fineCellsOfParentCell(p, m, rb);
      assert.strictEqual(parentCellOfFineCell(a, m, rb), p, `low child of parent ${p}`);
      assert.strictEqual(parentCellOfFineCell(b, m, rb), p, `high child of parent ${p}`);
      assert.strictEqual(b, a + 1, `parent ${p}'s children are not adjacent`);
    }
  });

  ok('the ring is EXACTLY one parent cell deep', () => {
    // This is the load-bearing geometric fact: it is why GHOST is 2, why two
    // fine substeps traverse the ring exactly once, and why a 2x2 ring block
    // is one parent cell in both directions.
    const m = 1, rb = 8, g = GHOST;
    for (let f = 0; f < g; f++) {
      assert.strictEqual(parentCellOfFineCell(f, m, rb), -1, `low ring cell ${f} is not in parent -1`);
    }
    const hi = g + 2 * rb;
    for (let f = hi; f < hi + g; f++) {
      assert.strictEqual(parentCellOfFineCell(f, m, rb), rb, `high ring cell ${f} is not in parent ${rb}`);
    }
  });

  ok('ring depth is 0 inside, 1..2 out, and identically 0 at the root', () => {
    const rb = 8, g = GHOST, hi = g + 2 * rb;
    assert.strictEqual(ringDepth(g, g, 1, rb), 0, 'interior corner read as ring');
    assert.strictEqual(ringDepth(hi - 1, hi - 1, 1, rb), 0, 'far interior corner read as ring');
    assert.strictEqual(ringDepth(g - 1, g, 1, rb), 1, 'first ring cell');
    assert.strictEqual(ringDepth(0, g, 1, rb), 2, 'outer ring cell');
    assert.strictEqual(ringDepth(0, 0, 1, rb), 2, 'ring corner takes its deepest axis');
    assert.strictEqual(ringDepth(g - 1, 0, 1, rb), 2, 'mixed-depth corner takes the max');
    // The root has no ring cells at all, at any coordinate it has.
    for (const f of [0, 1, 7, 15]) {
      assert.strictEqual(ringDepth(f, f, 0, rb), 0, `root cell ${f} read as ring`);
    }
  });

  ok('a ring slot is inbox or outbox by DIRECTION, not by cell', () => {
    const m = 1, rb = 8, g = GHOST;
    // One cell, two directions, two different roles -- which is the claim.
    const fx = g - 1, fy = g + 4;        // depth 1 on the low-x side
    assert.strictEqual(ringSlotRole(fx, fy, +1, 0, m, rb), 'inbox', 'inward is not inbox');
    assert.strictEqual(ringSlotRole(fx, fy, -1, 0, m, rb), 'outbox', 'outward is not outbox');
    assert.strictEqual(ringSlotRole(fx, fy, 0, +1, m, rb), 'tangential', 'along the seam is not tangential');
    assert.strictEqual(ringSlotRole(fx, fy, 0, 0, m, rb), 'rest', 'the rest population has no direction');
    // A diagonal that reduces depth is delivery, not a tangent.
    assert.strictEqual(ringSlotRole(0, 0, +1, +1, m, rb), 'inbox', 'inward diagonal at a corner');
    assert.strictEqual(ringSlotRole(g, g, +1, 0, m, rb), 'interior', 'an interior cell has no ring role');
    assert.strictEqual(ringSlotRole(0, 5, -1, 0, m, rb), 'offtile', 'a step off the buffer is not a role');
  });

  ok('pool inverse violations are caught from BOTH directions', () => {
    // RUN THE CHECKER ON INPUTS THAT VIOLATE IT (rule 3). A checker only ever
    // run on valid input is indistinguishable from one that returns nothing.
    const blockSlot = [-1, 0, -1, 1];
    const slotToBlock = [1, 3, -1];
    assert.deepStrictEqual(poolInverseViolations(blockSlot, slotToBlock), [], 'a consistent pool was flagged');

    // block -> slot -> a DIFFERENT block
    const a = poolInverseViolations([-1, 0, -1, 1], [2, 3, -1]);
    assert.ok(a.some(v => v.kind === 'block-slot-block'), 'a forward inconsistency was missed');

    // a slot claiming a block that does not claim it back -- the direction a
    // one-sided check would miss, and the one linkRefine would silently repair
    const b = poolInverseViolations([-1, -1, -1, -1], [1, -1, -1]);
    assert.ok(b.some(v => v.kind === 'slot-block-slot'), 'a reverse inconsistency was missed');

    assert.ok(poolInverseViolations([-1, 99], [-1]).some(v => v.kind === 'slot-out-of-range'),
      'an out-of-range slot was missed');
    assert.ok(poolInverseViolations([-1], [-1, 42]).some(v => v.kind === 'block-out-of-range'),
      'an out-of-range block was missed');
  });
  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
