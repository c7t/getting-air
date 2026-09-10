#!/usr/bin/env node
// GPU-free tests for d3-amr.mjs -- the octree pool's addressing
// (plans/3D.md M3).
//
// WHY THIS IS THE TEST THAT MATTERS. plans/3D.md ranks 3D 2:1 balance and
// the pool manager as risk #2, and notes that the 2D versions of those
// shaders carry three separately-documented live-verified bugs found only
// by an invariant checker. The addressing underneath them fails the same
// way: an off-by-one in the fine<->coarse mapping, or a neighbour lookup
// that picks the tile next door, does not crash and does not NaN -- it
// produces a plausible flow with a seam, and the seam is invisible until a
// validation number moves for reasons nobody can trace.
//
// So the neighbour resolution is checked against an INDEPENDENT route:
// GLOBAL FINE COORDINATES, in which "which tile owns this cell" is a single
// division with no ring, no offsets and no periodic block wrap to get
// wrong. resolveSource() and the global route share no arithmetic, so
// agreement between them is evidence rather than tautology.
//
// Run: node tools/test-d3-amr.js   (also picked up by `make test`)

const assert = require('assert');
const path = require('path');

let pass = 0;
function ok(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
}
const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b}`);

(async () => {
  const A = await import(path.join(__dirname, '..', 'd3-amr.mjs'));
  const {
    GHOST, fineToCoarseUnit, coarseUnitToFine, makePool, refineWhere,
    refineNearBody, resolveSource, toGlobalFine, fromGlobalFine, storageRatio,
    check21Balance, checkGeometryCoverage,
  } = A;

  ok('pool geometry follows FB = 2*RB + 2*GHOST and rejects a non-dividing RB', () => {
    for (const rb of [2, 4, 8]) {
      const p = makePool({ dims: [32, 32, 32], rb });
      assert.strictEqual(p.FB, 2 * rb + 2 * GHOST, `FB at RB=${rb}`);
      assert.strictEqual(p.nBlocks, (32 / rb) ** 3);
      assert.strictEqual(p.tileCells, p.FB ** 3);
    }
    assert.throws(() => makePool({ dims: [30, 32, 32], rb: 4 }), /not a multiple of RB/);
  });

  ok('storage ratio matches the table in plans/3D.md sec 2.1', () => {
    close(storageRatio(makePool({ dims: [32, 32, 32], rb: 4 })), 27.0, 1e-9, 'RB=4');
    close(storageRatio(makePool({ dims: [32, 32, 32], rb: 8 })), 15.625, 1e-9, 'RB=8');
  });

  ok('block id and its inverse round-trip over the whole grid', () => {
    const p = makePool({ dims: [24, 16, 32], rb: 4 });
    let n = 0;
    for (let bz = 0; bz < p.nb[2]; bz++) {
      for (let by = 0; by < p.nb[1]; by++) {
        for (let bx = 0; bx < p.nb[0]; bx++) {
          const id = p.blockId(bx, by, bz);
          assert.deepStrictEqual(p.blockOf(id), [bx, by, bz], `block ${bx},${by},${bz}`);
          n++;
        }
      }
    }
    assert.strictEqual(n, p.nBlocks);
  });

  // The half-cell offset is the whole reason a fine/coarse mapping is easy
  // to get wrong: the children of coarse cell c are at c +- 1/4, NOT at c
  // and c + 1/2.
  ok('fine <-> coarse-unit mapping is cell-centred and round-trips', () => {
    const origin = 12;
    close(fineToCoarseUnit(GHOST, origin), origin - 0.25, 1e-12, 'first interior child');
    close(fineToCoarseUnit(GHOST + 1, origin), origin + 0.25, 1e-12, 'second child');
    close(fineToCoarseUnit(GHOST + 2, origin), origin + 0.75, 1e-12, 'first child of the next coarse cell');
    // Ring cells continue the same line, on both sides.
    close(fineToCoarseUnit(0, origin), origin - 1.25, 1e-12, 'depth-2 ring below');
    close(fineToCoarseUnit(GHOST - 1, origin), origin - 0.75, 1e-12, 'depth-1 ring below');
    for (let j = 0; j < 12; j++) {
      assert.strictEqual(coarseUnitToFine(fineToCoarseUnit(j, origin), origin), j, `round-trip at j=${j}`);
    }
    // The two children of a coarse cell straddle it symmetrically, so their
    // mean is the coarse cell centre -- which is what makes the 8-cell
    // average a centred restriction.
    for (let c = 0; c < 4; c++) {
      const a = fineToCoarseUnit(GHOST + 2 * c, origin), b = fineToCoarseUnit(GHOST + 2 * c + 1, origin);
      close((a + b) / 2, origin + c, 1e-12, `children of coarse cell ${origin + c} straddle it`);
    }
  });

  ok('global-fine round-trip: tile-local -> global -> tile-local', () => {
    const p = makePool({ dims: [32, 24, 16], rb: 4 });
    for (const b of [[0, 0, 0], [3, 2, 1], [7, 5, 3]]) {
      for (const l of [[GHOST, GHOST, GHOST], [GHOST + 3, GHOST + 7, GHOST + 1], [GHOST + 2 * 4 - 1, GHOST, GHOST + 5]]) {
        const g = toGlobalFine(p, b, l);
        const back = fromGlobalFine(p, g);
        assert.deepStrictEqual(back.block, b, `block for local ${l} of block ${b}`);
        assert.deepStrictEqual(back.local, l, `local for ${l} of block ${b}`);
      }
    }
  });

  // THE ONE. Every interior cell of every active tile, every direction of
  // D3Q27 (a superset of D3Q19), checked against the global-fine route.
  ok('resolveSource agrees with the global-fine route for every cell and direction', () => {
    const p = makePool({ dims: [16, 16, 16], rb: 4 });
    // A refined region with a deliberately ragged edge, so plenty of cells
    // have SOME neighbours present and others absent -- a fully-refined
    // domain would never exercise the null path, and a single tile would
    // never exercise the present path.
    const { blockSlot } = refineWhere(p, ({ bx, by, bz }) => (bx + by + bz) % 3 !== 2);
    const RB2 = 2 * p.rb;
    const dirs = [];
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) dirs.push([dx, dy, dz]);

    let checked = 0, viaNeighbour = 0, viaNull = 0;
    for (let id = 0; id < p.nBlocks; id++) {
      if (blockSlot[id] < 0) continue;
      const b = p.blockOf(id);
      // Interior cells only: a ring cell is not stepped, it is filled.
      for (let fz = GHOST; fz < GHOST + RB2; fz += 3) {
        for (let fy = GHOST; fy < GHOST + RB2; fy += 3) {
          for (let fx = GHOST; fx < GHOST + RB2; fx += 3) {
            for (const d of dirs) {
              const src = [fx - d[0], fy - d[1], fz - d[2]];
              const got = resolveSource(p, blockSlot, b, src);

              // Independent route: where is this source in GLOBAL fine
              // coordinates, and who owns that?
              const g = toGlobalFine(p, b, src);
              const want = fromGlobalFine(p, g);
              const wantSlot = blockSlot[p.blockId(...want.block)];

              checked++;
              if (want.block.every((c, i) => c === b[i])) {
                assert.ok(got && got.own, `expected own-tile for ${src} in block ${b}`);
                assert.deepStrictEqual([got.fx, got.fy, got.fz], want.local, `own-tile local coords for ${src}`);
              } else if (wantSlot < 0) {
                assert.strictEqual(got, null, `expected null (no owning tile) for ${src} in block ${b}`);
                viaNull++;
              } else {
                assert.ok(got && !got.own, `expected neighbour tile for ${src} in block ${b}`);
                assert.strictEqual(got.slot, wantSlot, `neighbour slot for ${src} in block ${b}`);
                assert.deepStrictEqual([got.fx, got.fy, got.fz], want.local, `neighbour local coords for ${src}`);
                viaNeighbour++;
              }
            }
          }
        }
      }
    }
    // The test is only meaningful if it actually visited all three outcomes.
    assert.ok(checked > 5000, `only ${checked} combinations checked`);
    assert.ok(viaNeighbour > 100, `neighbour path exercised only ${viaNeighbour} times`);
    assert.ok(viaNull > 100, `missing-neighbour path exercised only ${viaNull} times`);
  });

  ok('a source resolved into a neighbour always lands in that tile INTERIOR', () => {
    // This is the property that makes the ring unnecessary for same-level
    // streaming: the neighbour's copy of the cell is real data, not its own
    // ring. If it landed in the neighbour's ring, DIRECT_GHOST would be
    // reading a stale interpolated value and calling it a same-level
    // exchange.
    const p = makePool({ dims: [16, 16, 16], rb: 4 });
    const { blockSlot } = refineWhere(p, () => true);
    const RB2 = 2 * p.rb;
    for (const b of [[1, 1, 1], [0, 2, 3]]) {
      for (let fz = GHOST; fz < GHOST + RB2; fz++) {
        for (let fy = GHOST; fy < GHOST + RB2; fy++) {
          for (let fx = GHOST; fx < GHOST + RB2; fx++) {
            for (const d of [[1, 0, 0], [-1, 0, 0], [0, 1, 1], [1, 1, 1], [-1, -1, -1]]) {
              const r = resolveSource(p, blockSlot, b, [fx - d[0], fy - d[1], fz - d[2]]);
              assert.ok(r, 'fully refined domain: every source must resolve');
              for (const c of [r.fx, r.fy, r.fz]) {
                assert.ok(p.isInterior(c), `resolved to local ${c}, which is ring, not interior`);
              }
            }
          }
        }
      }
    }
  });

  ok('refineWhere assigns slots deterministically and consistently both ways', () => {
    const p = makePool({ dims: [16, 16, 16], rb: 4 });
    const pred = ({ mid }) => Math.hypot(mid[0] - 8, mid[1] - 8, mid[2] - 8) < 5;
    const a = refineWhere(p, pred), b = refineWhere(p, pred);
    assert.deepStrictEqual(Array.from(a.blockSlot), Array.from(b.blockSlot), 'not deterministic');
    assert.ok(a.activeSlots > 0 && a.activeSlots < p.nBlocks, `selected ${a.activeSlots} of ${p.nBlocks}`);
    for (let id = 0; id < p.nBlocks; id++) {
      const s = a.blockSlot[id];
      if (s < 0) continue;
      assert.strictEqual(a.slotToBlock[s], id, `slotToBlock/blockSlot disagree at block ${id}`);
    }
    for (let s = 0; s < a.activeSlots; s++) {
      assert.ok(a.slotToBlock[s] >= 0, `slot ${s} below activeSlots must be occupied`);
      assert.strictEqual(a.blockSlot[a.slotToBlock[s]], s, `round-trip at slot ${s}`);
    }
    for (let s = a.activeSlots; s < p.maxSlots; s++) {
      assert.strictEqual(a.slotToBlock[s], -1, `slot ${s} above activeSlots must be free`);
    }
  });

  ok('refineWhere fails loudly rather than silently dropping tiles', () => {
    const p = makePool({ dims: [16, 16, 16], rb: 4, maxSlots: 3 });
    assert.throws(() => refineWhere(p, () => true), /more than maxSlots/);
  });

  ok('refineNearBody covers the body and stops within the margin', () => {
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    const centre = [16, 16, 16], R = 6;
    const sdf = (q) => Math.hypot(q[0] - centre[0], q[1] - centre[1], q[2] - centre[2]) - R;
    const margin = 2;
    const { blockSlot, activeSlots } = refineNearBody(p, sdf, margin);
    assert.ok(activeSlots > 0, 'nothing refined');
    for (let id = 0; id < p.nBlocks; id++) {
      const [bx, by, bz] = p.blockOf(id);
      const lo = [bx * p.rb, by * p.rb, bz * p.rb];
      const hi = lo.map(c => c + p.rb);
      // Any block containing a point ON the body surface must be refined --
      // this is the geometry-forced-refinement constraint, in static form.
      const mid = lo.map((c, i) => (c + hi[i]) / 2);
      const nearest = Math.abs(sdf(mid));
      if (nearest < 1e-9) assert.ok(blockSlot[id] >= 0, `block ${id} straddles the surface but is not refined`);
      // And nothing far away should be.
      if (sdf(mid) > margin + p.rb * Math.sqrt(3)) {
        assert.strictEqual(blockSlot[id], -1, `block ${id} is far from the body but was refined`);
      }
    }
  });


  // --- structural invariants (plans/3D.md M4.2, risk #2) -------------------
  //
  // Every one of these feeds the checker an input that VIOLATES the
  // invariant and asserts it is caught. A checker only ever run on valid
  // input is indistinguishable from `return { violations: [] }`, and that is
  // exactly the failure mode plans/3D.md sec 7 risk #2 is about: the 2D
  // manager's three live-verified balance bugs all sat under a green suite.

  const setOf = (...keys) => new Set(keys);

  ok('2:1 balance passes a flat one-level refinement, and says why that is weak', () => {
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    const { blockSlot } = refineWhere(p, ({ bx }) => bx < 4);
    const lv1 = new Set();
    for (let id = 0; id < p.nBlocks; id++) if (blockSlot[id] >= 0) lv1.add(p.blockOf(id).join(','));
    const r = check21Balance([null, lv1], () => p.nb, { levels: 2 });
    assert.strictEqual(r.violations.length, 0);
    // The honest half: with one refined level the check CANNOT fail, so this
    // asserts the machinery runs, not that balance holds. If a future change
    // makes a one-level domain able to violate 2:1, this assertion is the
    // thing that should be revisited rather than the checker.
    assert.strictEqual(r.levels, 2);
  });

  // These build SMALL, deliberately incomplete trees and assert about ONE
  // named tile, filtering the violation list rather than demanding a
  // globally legal arrangement. Building a fully balanced 3-level nest by
  // hand is itself error-prone -- the first draft of these tests called
  // three different arrangements "legal" and the checker was right about
  // all three -- and a test whose premise is hard to get right is a test
  // that will later be "fixed" by loosening the checker.
  // Filter to ONE tile and ONE axis. These trees are deliberately extended
  // along x only, so the y and z faces of every tile look out onto nothing
  // and violate 2:1 by construction -- which is correct, and not what any of
  // these tests is about.
  const at = (r, bx, axis = 0) => r.violations.filter(v => v.block[0] === bx && v.axis === axis);

  ok('2:1 balance CATCHES a level-3 leaf whose face neighbour resolves at level 1', () => {
    const nbAt = (m) => [8 * 2 ** (m - 1), 8 * 2 ** (m - 1), 8 * 2 ** (m - 1)];
    // A properly parented level-3 tile at (12,0,0): level-2 (6,0,0),
    // level-1 (3,0,0). Its -x face neighbour is level-3 (11,0,0), absent;
    // walking up gives level-2 (5,0,0).
    const lv1 = setOf('3,0,0', '2,0,0');
    const lv2 = setOf('6,0,0', '5,0,0');
    const lv3 = setOf('12,0,0');
    const ok3 = check21Balance([null, lv1, lv2, lv3], nbAt, { levels: 4 });
    assert.strictEqual(at(ok3, 12).length, 0,
      `level-3 (12,0,0) next to a level-2 tile is legal on x: ${JSON.stringify(at(ok3, 12)[0])}`);
    // Remove that level-2 tile and the same face resolves at level 1 -- a
    // gap of 2, which is exactly what 2:1 balance forbids.
    const bad = check21Balance([null, lv1, setOf('6,0,0'), lv3], nbAt, { levels: 4 });
    assert.ok(at(bad, 12).some(v => v.axis === 0 && v.dir === -1 && v.nDepth === 1),
      `a 3-vs-1 gap on the -x face must be caught: ${JSON.stringify(bad.violations)}`);
  });

  ok('2:1 balance reads the SHARED FACE of a neighbour, not its deepest corner', () => {
    // A level-1 leaf at (6,0,0). Its -x neighbour (5,0,0) is refined to
    // level 2 (children at x = 10, 11). Refine the level-2 child on the FAR
    // side (x=10) down to level 3: the shared face is still level 2, so this
    // is legal. A checker that took the neighbour tile's maximum depth
    // ANYWHERE would report a violation here -- and would then be loosened,
    // which is how a real gap gets through later.
    const nbAt = (m) => [8 * 2 ** (m - 1), 8 * 2 ** (m - 1), 8 * 2 ** (m - 1)];
    const lv1 = setOf('6,0,0', '5,0,0');
    const lv2 = setOf('10,0,0', '11,0,0');
    const far = check21Balance([null, lv1, lv2, setOf('20,0,0')], nbAt, { levels: 4 });
    assert.strictEqual(at(far, 6).length, 0,
      `a level-3 sliver on the FAR side of the neighbour is legal on x: ${JSON.stringify(at(far, 6)[0])}`);
    // Move it to the NEAR side (level-2 x=11, whose children are 22,23) and
    // the level-1 tile is suddenly face-to-face with level 3.
    const near = check21Balance([null, lv1, lv2, setOf('22,0,0')], nbAt, { levels: 4 });
    assert.ok(near.violations.some(v => v.block[0] === 6 && v.axis === 0 && v.dir === -1 && v.nDepth === 3),
      `a level-3 sliver on the SHARED face must be caught: ${JSON.stringify(near.violations)}`);
  });

  ok('2:1 balance wraps periodically, matching the block grid', () => {
    // A level-3 tile at x = 0 has a -x face neighbour that WRAPS to the far
    // side of the domain, where there is nothing. A checker that clamped
    // instead of wrapping would miss every violation on the domain face.
    const nbAt = (m) => [8 * 2 ** (m - 1), 8 * 2 ** (m - 1), 8 * 2 ** (m - 1)];
    const r = check21Balance([null, setOf('0,0,0'), setOf('0,0,0'), setOf('0,0,0')], nbAt, { levels: 4 });
    assert.ok(r.violations.some(v => v.block[0] === 0 && v.axis === 0 && v.dir === -1 && v.nDepth === 0),
      `the periodic wrap must be checked, not clamped: ${JSON.stringify(r.violations)}`);
  });

  ok('geometry coverage passes a real refineNearBody result', () => {
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    const c = [16, 16, 16], R = 5, margin = 2;
    const sdf = (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - R;
    const { blockSlot } = refineNearBody(p, sdf, margin);
    const r = checkGeometryCoverage(p, blockSlot, sdf, margin);
    assert.ok(r.required > 0, 'the test body must actually require refinement');
    assert.strictEqual(r.violations.length, 0, JSON.stringify(r.violations.slice(0, 3)));
  });

  ok('geometry coverage CATCHES a hole punched in the refined shell', () => {
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    const c = [16, 16, 16], R = 5, margin = 2;
    const sdf = (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - R;
    const { blockSlot } = refineNearBody(p, sdf, margin);
    // Punch the hole in the block the body CENTRE sits in, not merely the
    // first refined one: refineNearBody's block-corner test refines a wider
    // set than the cell-granular requirement, so unrefining an outer block
    // of the shell legitimately changes nothing and would make this test
    // pass for the wrong reason.
    const hole = p.blockId(...c.map(v => Math.floor(v / p.rb)));
    assert.ok(blockSlot[hole] >= 0, 'the body centre must start out refined');
    blockSlot[hole] = -1;
    const r = checkGeometryCoverage(p, blockSlot, sdf, margin);
    assert.ok(r.violations.length > 0, 'unrefining the block under the body must be caught');
    assert.ok(r.violations.every(v => v.block === hole));
  });

  ok('geometry coverage is an INDEPENDENT route: it catches a body smaller than a block', () => {
    // refineNearBody samples a block's corners and centre, so a body that
    // fits between those samples refines nothing -- its own header says so.
    // The cell-granular scan is what turns that from a comment into a
    // failure, which is the whole reason it does not reuse that predicate.
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    // Centred ON a cell, so the cell scan sees it, but small enough to slip
    // between the block's own corner and centre samples: block (0,0,0)
    // spans cells 0..3, and both (0,0,0) and its centre (2,2,2) are sqrt(3)
    // away, well outside R + margin.
    const c = [1, 1, 1], R = 0.6, margin = 0.1;
    const sdf = (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - R;
    const { blockSlot, activeSlots } = refineNearBody(p, sdf, margin);
    assert.strictEqual(activeSlots, 0, 'the premise of this test is that the block test misses it');
    const r = checkGeometryCoverage(p, blockSlot, sdf, margin);
    assert.ok(r.violations.length > 0, 'the cell-granular scan must catch what the block test missed');
  });

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
