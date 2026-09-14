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
// AND THE SUITE ITSELF WAS SCORED THE SAME WAY, because "35 checks pass" says
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
    tauAtLevel, makePool, poolAtLevel, nbAtLevel, parentOfBlock,
    quadrantOfBlock, quadrantOrigin, tileOriginL0, tileOriginL0Recursive,
    refineWhere, nearBodyWant, nearBodyWantCentre, refineNearBody,
    resolveSource, toGlobalFine, fromGlobalFine, storageRatio,
    check21Balance, checkRingParentCoverage, checkGeometryCoverage, cascade21,
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

  ok('tau doubles (tau - 1/2) per rung, which is what makes AMR a stability mechanism', () => {
    close(tauAtLevel(0.8, 0), 0.8, 1e-12, 'level 0');
    close(tauAtLevel(0.8, 1), 1.1, 1e-12, 'level 1');
    close(tauAtLevel(0.8, 2), 1.7, 1e-12, 'level 2');
    for (const tau0 of [0.51, 0.6, 0.8]) {
      for (let m = 0; m < 4; m++) {
        close(tauAtLevel(tau0, m) - 0.5, (tau0 - 0.5) * 2 ** m, 1e-12, `tau-1/2 at level ${m}`);
      }
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

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
