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
    check21Balance, checkGeometryCoverage, cascade21,
    poolAtLevel, parentOfBlock, octantOfBlock, octantOrigin, refineHierarchy,
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



  // --- the pool-parent path (plans/3D.md M5.1a) ----------------------------
  //
  // The claim these check is that there is NO second addressing scheme at
  // depth: level m's tiling is level 1's tiling of a 2^(m-1)-times larger
  // domain, so every function above works unchanged. A claim like that is
  // exactly the kind that is true when written and quietly false two
  // milestones later, so it is re-run at depth rather than argued.

  ok('poolAtLevel(m=1) IS the pool, and deeper levels double the block grid', () => {
    const p = makePool({ dims: [16, 16, 16], rb: 4 });
    const l1 = poolAtLevel(p, 1);
    assert.deepStrictEqual(l1.nb, p.nb);
    assert.strictEqual(l1.FB, p.FB);
    assert.strictEqual(l1.tileCells, p.tileCells);
    for (let m = 2; m <= 4; m++) {
      const lm = poolAtLevel(p, m);
      // Same tile, more of them -- the uniform-tile-shape claim in one line.
      assert.strictEqual(lm.FB, p.FB, `FB changed at level ${m}`);
      assert.strictEqual(lm.rb, p.rb, `RB changed at level ${m}`);
      assert.deepStrictEqual(lm.nb, p.nb.map(n => n * 2 ** (m - 1)), `nb at level ${m}`);
      // ...and it agrees with the block grid check21Balance/cascade21 were
      // written against in M4.2a and M4.2b-iv, which is the whole reason
      // those two need no change at depth.
      assert.deepStrictEqual(lm.nb, p.nb.map(n => n * 2 ** (m - 1)));
    }
    assert.throws(() => poolAtLevel(p, 0), /no pool/);
  });

  ok('resolveSource agrees with the global route at level 2 AND level 3', () => {
    // The same check as "THE ONE" above, at depth. If level m needed its own
    // addressing this is where it would show, because the global route is
    // recomputed from that level's own dims and shares no arithmetic with
    // resolveSource.
    const base = makePool({ dims: [8, 8, 8], rb: 2 });
    for (const m of [2, 3]) {
      const p = poolAtLevel(base, m);
      const { blockSlot } = refineWhere(p, ({ bx, by, bz }) => (bx + by + bz) % 3 !== 2);
      const RB2 = 2 * p.rb;
      let checked = 0, viaNeighbour = 0, viaNull = 0;
      for (let id = 0; id < p.nBlocks; id++) {
        if (blockSlot[id] < 0) continue;
        const b = p.blockOf(id);
        for (let fz = GHOST; fz < GHOST + RB2; fz++) {
          for (let fy = GHOST; fy < GHOST + RB2; fy++) {
            for (const d of [[1, 0, 0], [-1, 1, 0], [0, -1, 1], [1, 1, 1], [-1, -1, -1]]) {
              const src = [GHOST - d[0], fy - d[1], fz - d[2]];
              const got = resolveSource(p, blockSlot, b, src);
              const want = fromGlobalFine(p, toGlobalFine(p, b, src));
              const wantSlot = blockSlot[p.blockId(...want.block)];
              checked++;
              if (want.block.every((c, i) => c === b[i])) {
                assert.ok(got && got.own, `L${m}: expected own-tile for ${src} in ${b}`);
                assert.deepStrictEqual([got.fx, got.fy, got.fz], want.local, `L${m}: own local for ${src}`);
              } else if (wantSlot < 0) {
                assert.strictEqual(got, null, `L${m}: expected null for ${src} in ${b}`);
                viaNull++;
              } else {
                assert.ok(got && !got.own, `L${m}: expected neighbour for ${src} in ${b}`);
                assert.strictEqual(got.slot, wantSlot, `L${m}: neighbour slot for ${src} in ${b}`);
                assert.deepStrictEqual([got.fx, got.fy, got.fz], want.local, `L${m}: neighbour local for ${src}`);
                viaNeighbour++;
              }
            }
          }
        }
      }
      assert.ok(checked > 500, `L${m}: only ${checked} combinations checked`);
      assert.ok(viaNeighbour > 20, `L${m}: neighbour path exercised only ${viaNeighbour} times`);
      assert.ok(viaNull > 20, `L${m}: missing-neighbour path exercised only ${viaNull} times`);
    }
  });

  ok('the octant bits alone place a child inside its parent tile', () => {
    // The pool-parent path's one genuinely new piece: no spatial lookup and
    // no parent-chain walk, just q. Checked against the GLOBAL route at both
    // levels -- child global cell -> parent global cell by one halving ->
    // parent block by one division -> parent-local index. That route shares
    // nothing with octantOrigin's closed form.
    const base = makePool({ dims: [8, 8, 8], rb: 2 });
    for (const m of [2, 3]) {
      const child = poolAtLevel(base, m);
      const par = poolAtLevel(base, m - 1);
      const RB2 = 2 * child.rb;
      let checked = 0;
      for (let id = 0; id < child.nBlocks; id += 5) {
        const b = child.blockOf(id);
        const pb = parentOfBlock(b), q = octantOfBlock(b);
        // The parent block is the one that geometrically contains the child.
        assert.deepStrictEqual(pb, b.map(x => Math.floor(x / 2)), `L${m}: parent of ${b}`);
        assert.ok(pb.every((c, i) => c < par.nb[i]), `L${m}: parent ${pb} outside the level-${m - 1} grid`);
        for (const j of [0, 1, GHOST, GHOST + 1, GHOST + RB2 - 1, GHOST + RB2, child.FB - 1]) {
          for (let a = 0; a < 3; a++) {
            // Independent route, in global coordinates at each level.
            const gChild = b[a] * RB2 + (j - GHOST);
            const nParent = par.dims[a] * 2;                  // parent cells per axis, globally
            const gParent = Math.floor(gChild / 2);
            const wrapped = ((gParent % nParent) + nParent) % nParent;
            const pBlock = Math.floor(wrapped / (2 * par.rb));
            const pLocal = wrapped - pBlock * 2 * par.rb + GHOST;
            // Interior cells must land in the parent block the octant names;
            // ring cells may leave it, which is the ring's whole job.
            if (j >= GHOST && j < GHOST + RB2) {
              assert.strictEqual(pBlock, pb[a], `L${m}: axis ${a}, child ${b} local ${j} -> parent block`);
              assert.strictEqual(pLocal, octantOrigin(child.rb, q[a]) + ((j - GHOST) >> 1),
                `L${m}: axis ${a}, child ${b} octant ${q[a]} local ${j} -> parent local`);
            }
            // The CONTINUOUS mapping, which is what the interpolation
            // stencil actually consumes: the same fineToCoarseUnit the
            // dense-parent path uses, with the octant offset as its origin.
            // Independent route: match physical positions -- a child cell
            // centre is at (g + 0.5) * dx_child, a parent-local coordinate u
            // is at (pBlock*2RB + u - GHOST + 0.5) * dx_parent, and
            // dx_child = dx_parent / 2.
            const u = fineToCoarseUnit(j, octantOrigin(child.rb, q[a]));
            const uWant = GHOST - 0.5 + (gChild + 0.5) / 2 - pb[a] * 2 * par.rb;
            close(u, uWant, 1e-12, `L${m}: axis ${a}, child ${b} local ${j} continuous parent coord`);
            checked++;
          }
        }
      }
      assert.ok(checked > 100, `L${m}: only ${checked} placements checked`);
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


  // --- the 2:1 cascade (plans/3D.md M4.2b-iv) ------------------------------
  //
  // cascade21 is scored against check21Balance, which was written first, in
  // M4.2a, deliberately before there was a manager to be tempted to agree
  // with. So these are two independently-written statements of one
  // invariant run against each other: the cascade proposes a balanced set
  // and the checker -- which knows nothing about the closure rule -- either
  // finds a gap in it or does not.
  //
  // The trees below are FULL, not the deliberately-incomplete ones the
  // checker's own tests use: the cascade's output is asserted to be
  // globally balanced, so a tree whose y and z faces look out onto nothing
  // would fail for reasons that are not about the cascade.

  const nbAt3 = (m) => [4 * 2 ** (m - 1), 4 * 2 ** (m - 1), 4 * 2 ** (m - 1)];
  // The eight children of the parent that contains `b` -- the smallest
  // refinement the pool can actually hold at level >= 2.
  const octet = (b) => {
    const p = [b[0] >> 1, b[1] >> 1, b[2] >> 1], s = new Set();
    for (let k = 0; k < 8; k++) s.add(`${p[0] * 2 + (k & 1)},${p[1] * 2 + ((k >> 1) & 1)},${p[2] * 2 + ((k >> 2) & 1)}`);
    return s;
  };
  const keys = (sets) => sets.map(s => (s ? [...s].sort().join('|') : null));
  // The tree property check21Balance does NOT make: every block's parent
  // must exist. Needed by the minimality test below, because an orphaned
  // sub-tree can be structurally illegal while still being 2:1 balanced.
  const orphans = (sets, levels) => {
    const out = [];
    for (let m = 2; m < levels; m++) {
      for (const k of sets[m] || []) {
        const b = k.split(',').map(Number);
        if (!sets[m - 1].has(`${b[0] >> 1},${b[1] >> 1},${b[2] >> 1}`)) out.push({ level: m, block: b });
      }
    }
    return out;
  };

  ok('the cascade is the IDENTITY at levels=2, which is why there is no balance kernel', () => {
    // The claim shaders/common_d3_manage.wgsl relies on: with one refined
    // level the closure has nothing to close, because a level-1 block's
    // parent level is the dense L0 grid and that is present everywhere. So
    // the manager ships WITHOUT a balance pass rather than with a kernel
    // that provably does nothing -- and this is the check on "provably".
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    let rng = 12345;
    const rand = () => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (let trial = 0; trial < 20; trial++) {
      const lv1 = new Set();
      for (let id = 0; id < p.nBlocks; id++) if (rand() < 0.3) lv1.add(p.blockOf(id).join(','));
      const r = cascade21([null, lv1], () => p.nb, { levels: 2 });
      assert.strictEqual(r.forced.length, 0, `trial ${trial}: forced ${JSON.stringify(r.forced[0])}`);
      assert.deepStrictEqual(keys(r.sets), keys([null, lv1]), `trial ${trial}: the set changed`);
    }
  });

  ok('the cascade CLOSES a 3-vs-1 gap the checker reports, and check21Balance agrees', () => {
    // The same shape as the checker's own 3-vs-1 test, with the level-2 tile
    // that would have made it legal left out. The checker must see the gap,
    // the cascade must close it, and the checker must then see nothing.
    const lv1 = new Set(['3,0,0']);
    const lv2 = new Set(['6,0,0']);
    const lv3 = new Set(['12,0,0']);
    const before = check21Balance([null, lv1, lv2, lv3], nbAt3, { levels: 4 });
    assert.ok(before.violations.length > 0, 'the premise: this tree is unbalanced');
    const c = cascade21([null, lv1, lv2, lv3], nbAt3, { levels: 4 });
    assert.ok(c.forced.length > 0, 'the cascade must have had work to do');
    const after = check21Balance(c.sets, nbAt3, { levels: 4 });
    assert.strictEqual(after.violations.length, 0,
      `the cascade left ${after.violations.length} violations, e.g. ${JSON.stringify(after.violations[0])}`);
    assert.strictEqual(orphans(c.sets, 4).length, 0, 'the cascade left an orphaned sub-tree');
    // It grew the set rather than editing it: nothing the criterion asked
    // for may be dropped, or the manager would be silently un-refining what
    // geometry forced.
    for (const k of lv3) assert.ok(c.sets[3].has(k), `dropped a wanted block ${k}`);
  });

  ok('the cascade materializes the parents of a block that has none', () => {
    // The manager evaluates its criterion per level and nothing in it looks
    // up, so a level-3 want with no level-2 or level-1 ancestor is a state
    // it can genuinely produce.
    const c = cascade21([null, new Set(), new Set(), new Set(['12,4,4'])], nbAt3, { levels: 4 });
    assert.ok(c.sets[2].has('6,2,2'), `level-2 parent missing: ${[...c.sets[2]].join(' ')}`);
    assert.ok(c.sets[1].has('3,1,1'), `level-1 grandparent missing: ${[...c.sets[1]].join(' ')}`);
    assert.strictEqual(orphans(c.sets, 4).length, 0);
    assert.strictEqual(check21Balance(c.sets, nbAt3, { levels: 4 }).violations.length, 0);
  });

  ok('the cascade is idempotent, so the single deepest-first sweep IS a fixed point', () => {
    // The sweep runs once and relies on additions at level m-1 being made
    // before the loop reaches m-1. That argument is easy to state and easy
    // to get wrong, so it is checked rather than trusted.
    const start = [null, new Set(['3,0,0']), new Set(['6,0,0']), new Set(['12,0,0', '13,5,2'])];
    const once = cascade21(start, nbAt3, { levels: 4 });
    const twice = cascade21(once.sets, nbAt3, { levels: 4 });
    assert.strictEqual(twice.forced.length, 0, `a second sweep still forced ${JSON.stringify(twice.forced[0])}`);
    assert.deepStrictEqual(keys(twice.sets), keys(once.sets));
  });

  ok('the cascade adds NOTHING to a uniformly refined tree', () => {
    // Every block present at every level: balanced by construction, and the
    // case a closure that over-fires would still get right -- included
    // because the minimality test below is about a sparse tree and this one
    // is about not touching a dense one.
    const sets = [null];
    for (let m = 1; m <= 3; m++) {
      const nb = nbAt3(m), s = new Set();
      for (let z = 0; z < nb[2]; z++) for (let y = 0; y < nb[1]; y++) for (let x = 0; x < nb[0]; x++) s.add(`${x},${y},${z}`);
      sets.push(s);
    }
    const c = cascade21(sets, nbAt3, { levels: 4 });
    assert.strictEqual(c.forced.length, 0, `forced ${JSON.stringify(c.forced[0])} on a uniform tree`);
  });

  ok('the cascade balances a REAL nested-shell refinement, not just a hand-built tree', () => {
    // The shape M5 will actually produce: three geometry-driven shells
    // around a sphere, each level's criterion evaluated independently. That
    // is what makes it unbalanced -- nothing in a per-level distance test
    // knows about the level above it, so a tight level-3 shell ends up face
    // to face with level-1 territory. The hand-built trees above check ONE
    // named tile; this checks that a whole realistic set comes out clean.
    const nb1 = 8, R = 5, ctr = 16;          // 32^3 cells at RB=4
    const nbAt = (m) => Array(3).fill(nb1 * 2 ** (m - 1));
    const shell = (m, band) => {
      const nb = nbAt(m)[0], w = 32 / nb, out = new Set();
      for (let z = 0; z < nb; z++) for (let y = 0; y < nb; y++) for (let x = 0; x < nb; x++) {
        const d = Math.hypot((x + 0.5) * w - ctr, (y + 0.5) * w - ctr, (z + 0.5) * w - ctr) - R;
        if (Math.abs(d) <= band) out.add(`${x},${y},${z}`);
      }
      return out;
    };
    const sets = [null, shell(1, 6), shell(2, 3), shell(3, 2.5)];
    const before = check21Balance(sets, nbAt, { levels: 4 });
    assert.ok(before.violations.length > 0,
      'the premise: independently-evaluated shells are NOT 2:1 balanced');
    const c = cascade21(sets, nbAt, { levels: 4 });
    const after = check21Balance(c.sets, nbAt, { levels: 4 });
    assert.strictEqual(after.violations.length, 0,
      `${after.violations.length} left, e.g. ${JSON.stringify(after.violations[0])}`);
    assert.strictEqual(orphans(c.sets, 4).length, 0);
    // Grew, never shrank: the criterion's own wants must all survive, or the
    // manager would be quietly un-refining what geometry forced.
    for (let m = 1; m < 4; m++) for (const k of sets[m]) assert.ok(c.sets[m].has(k), `dropped ${k} at level ${m}`);
    // And it did not answer "balanced" by refining the world. RECORDED
    // VALUES, measured 2026-09-10, not derived: 88/272/1680 wanted ->
    // 184/704/2432 balanced, against 512/4096/32768 blocks per level. If
    // these move, the closure changed -- and whether the new numbers are
    // right is a question for the minimality test above, not for this
    // assertion, which only makes the move visible.
    assert.deepStrictEqual(c.counts.slice(1), [184, 704, 2432],
      `balanced counts moved from the recorded 184/704/2432`);
  });

  ok('the cascade output is OCTET-COMPLETE, which is what the checker assumes', () => {
    // check21Balance's hasChild tests octant (0,0,0) alone, so a parent
    // holding only some of its children reads as a LEAF and the checker
    // reports a violation that is not one. That is not a shortcut to be
    // fixed: a tile is allocated per block and the 2D pool manager spawns a
    // whole quad, so a partially-refined parent does not exist. This is the
    // assertion that keeps the cascade on the same model.
    const c = cascade21([null, new Set(), new Set(), new Set(['12,4,4'])], nbAt3, { levels: 4 });
    for (let m = 2; m < 4; m++) {
      for (const k of c.sets[m]) {
        const b = k.split(',').map(Number);
        const par = [b[0] >> 1, b[1] >> 1, b[2] >> 1];
        for (let q = 0; q < 8; q++) {
          const sib = `${par[0] * 2 + (q & 1)},${par[1] * 2 + ((q >> 1) & 1)},${par[2] * 2 + ((q >> 2) & 1)}`;
          assert.ok(c.sets[m].has(sib), `level ${m} block ${k} is present without its sibling ${sib}`);
        }
      }
    }
  });

  ok('every OCTET the cascade forces is NECESSARY -- it does not over-refine', () => {
    // The failure this guards against is a closure that refines a halo
    // "to be safe": it would pass every test above, since a superset of a
    // balanced set is still balanced, and would cost slots for nothing.
    //
    // THE UNIT IS THE OCTET, NOT THE BLOCK, and that is the octet-
    // completeness rule again rather than a weakening of the test: removing
    // one child of a spawned octet does not describe any state the pool can
    // be in. Removing the octet does. Each removal must break the tree --
    // either 2:1 balance (check21Balance, written independently) or the
    // parent property (orphans, above), both statements about the RESULT
    // and not about the rule that produced it.
    // The input is given OCTET-COMPLETE on purpose, so that every group
    // below is something the CLOSURE decided rather than the octet rule
    // restating the caller's own want. Testing "removing it breaks the
    // tree" against a sibling the octet rule added would be a tautology.
    const start = [null, new Set(), new Set(), octet([12, 4, 4])];
    const c = cascade21(start, nbAt3, { levels: 4 });
    const groups = new Map();
    for (const f of c.forced) {
      const g = `${f.level}:${f.octet ? f.octet.join(',') : f.block.join(',')}`;
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(f);
    }
    assert.ok(groups.size > 3, `only ${groups.size} octets to test`);
    for (const [g, members] of groups) {
      const cut = c.sets.map(x => (x ? new Set(x) : null));
      for (const f of members) cut[f.level].delete(f.block.join(','));
      const bal = check21Balance(cut, nbAt3, { levels: 4 }).violations.length;
      const orp = orphans(cut, 4).length;
      assert.ok(bal > 0 || orp > 0, `removing forced octet ${g} left a legal tree: it was not necessary`);
    }
  });

  // --- the refinement hierarchy (plans/3D.md M5.1b) -------------------------

  ok('refineHierarchy at levels=2 is BIT-IDENTICAL to refineWhere/refineNearBody', () => {
    // M5.1b's gate, and the reason the whole hierarchy can be introduced
    // against an unchanged answer: at levels=2 the finest level IS level 1,
    // the L0 rescale is a no-op, and cascade21 is the identity. If this ever
    // stops holding, the GPU side's own bit-identical gate is measuring
    // something that already moved.
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    const cases = [
      ['box', ({ mid }) => mid.every(c => c >= 8 && c < 24)],
      ['all', () => true],
      ['ragged', ({ bx, by, bz }) => (bx + by + bz) % 3 !== 2],
    ];
    for (const [name, pred] of cases) {
      const flat = refineWhere(p, pred);
      const h = refineHierarchy(p, { levels: 2, want: pred });
      assert.deepStrictEqual(Array.from(h.byLevel[1].blockSlot), Array.from(flat.blockSlot), `${name}: blockSlot`);
      assert.deepStrictEqual(Array.from(h.byLevel[1].slotToBlock), Array.from(flat.slotToBlock), `${name}: slotToBlock`);
      assert.strictEqual(h.byLevel[1].activeSlots, flat.activeSlots, `${name}: activeSlots`);
    }
    // ...including the real geometry-forced path, predicate and all.
    const c = [16, 16, 16], R = 5, margin = 2;
    const sdf = (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - R;
    const body = ({ lo, hi }) => {
      let best = Infinity;
      for (let i = 0; i < 8; i++) best = Math.min(best, sdf([i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]));
      return Math.min(best, sdf([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])) <= margin;
    };
    const flat = refineNearBody(p, sdf, margin);
    const h = refineHierarchy(p, { levels: 2, want: body });
    assert.deepStrictEqual(Array.from(h.byLevel[1].blockSlot), Array.from(flat.blockSlot), 'body: blockSlot');
    assert.ok(flat.activeSlots > 0, 'the body must actually refine something');
  });

  ok('refineHierarchy builds a 2:1-balanced, properly parented tree at depth', () => {
    const p = makePool({ dims: [32, 32, 32], rb: 4 });
    const c = [16, 16, 16], R = 5, margin = 2;
    const sdf = (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - R;
    const body = ({ lo, hi }) => {
      let best = Infinity;
      for (let i = 0; i < 8; i++) best = Math.min(best, sdf([i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]));
      return Math.min(best, sdf([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])) <= margin;
    };
    for (const levels of [3, 4]) {
      const h = refineHierarchy(p, { levels, want: body });
      const nbAt = (m) => poolAtLevel(p, m).nb;
      // The invariant, by the checker written in M4.2a -- not by re-running
      // the closure that produced the set.
      const bal = check21Balance(h.sets, nbAt, { levels });
      assert.strictEqual(bal.violations.length, 0,
        `levels=${levels}: ${bal.violations.length} violations, e.g. ${JSON.stringify(bal.violations[0])}`);
      // Every level is populated and properly parented: a hierarchy that
      // refined only the finest level would pass 2:1 above only if the
      // closure were broken, so this is the cross-check on that.
      for (let m = 1; m < levels; m++) {
        assert.ok(h.byLevel[m].activeSlots > 0, `levels=${levels}: level ${m} is empty`);
        for (const k of h.sets[m]) {
          if (m < 2) continue;
          const b = k.split(',').map(Number);
          assert.ok(h.sets[m - 1].has(parentOfBlock(b).join(',')),
            `levels=${levels}: level-${m} block ${k} has no parent`);
        }
      }
      // Coarser levels must ENCLOSE finer ones, which is what a shell IS.
      // Checked in physical L0 units so it is a statement about geometry
      // rather than about indices.
      for (let m = 2; m < levels; m++) {
        const wFine = poolAtLevel(p, m).rb / 2 ** (m - 1);
        const wCoarse = poolAtLevel(p, m - 1).rb / 2 ** (m - 2);
        let maxFine = -Infinity, maxCoarse = -Infinity;
        for (const k of h.sets[m]) maxFine = Math.max(maxFine, (Number(k.split(',')[0]) + 1) * wFine);
        for (const k of h.sets[m - 1]) maxCoarse = Math.max(maxCoarse, (Number(k.split(',')[0]) + 1) * wCoarse);
        assert.ok(maxCoarse >= maxFine,
          `levels=${levels}: level ${m - 1} (x<=${maxCoarse}) does not enclose level ${m} (x<=${maxFine})`);
      }
    }
  });

  ok('refineHierarchy covers the body at the FINEST level, at cell granularity', () => {
    // The requirement geometry-forced refinement actually makes, checked the
    // independent way checkGeometryCoverage already checks it at level 1:
    // scan cells against the SDF rather than re-running the block predicate.
    // Driven at depth 3, where the finest view's cells are 4x smaller than
    // an L0 cell -- if refineHierarchy's L0 rescale were wrong, the shell
    // would be the wrong physical size and this is what would say so.
    const p = makePool({ dims: [16, 16, 16], rb: 4 });
    const c = [8, 8, 8], R = 3, margin = 1.5;
    const sdf = (q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) - R;
    const body = ({ lo, hi }) => {
      let best = Infinity;
      for (let i = 0; i < 8; i++) best = Math.min(best, sdf([i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]));
      return Math.min(best, sdf([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])) <= margin;
    };
    const levels = 3, h = refineHierarchy(p, { levels, want: body });
    const fine = h.byLevel[levels - 1];
    // The finest view's cells are 2^(m-1) per L0 cell, so the SDF is
    // evaluated on the rescaled grid.
    const s = 2 ** (levels - 2);
    const r = checkGeometryCoverage(fine.pool, fine.blockSlot, (q) => sdf(q.map(v => v / s)) * s, margin * s);
    assert.ok(r.required > 0, 'the body must require cells at the finest level');
    assert.strictEqual(r.violations.length, 0,
      `${r.violations.length} cells within the margin sit outside the finest level, e.g. ${JSON.stringify(r.violations[0])}`);
  });

  if (!process.exitCode) console.log(`\n${pass} check(s) passed`);
  else console.log('\nFAILED');
})();
