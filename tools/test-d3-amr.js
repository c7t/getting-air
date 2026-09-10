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

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
