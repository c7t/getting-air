#!/usr/bin/env node
// Pure-Node assertions for card-params.mjs. No GPU, no browser, no server --
// run it with `make test` (or `node tools/test-card-params.js`).
//
// WHY: the BLOCKAGE/ASPECT/RE parameterization makes a specific, load-bearing
// claim -- that index.html at res=R and index-amr.html at res=R-(levels-1)
// with the same ?blockage=&aspect=&re=&ut= are THE SAME PHYSICAL SYSTEM,
// resolved identically at the body and differing only in how much of the far
// field is resolved. That claim is the entire basis for comparing the two
// solvers' cost, and until this file existed the only way to check it was to
// stare at two control panels and two renders and decide whether the card
// "looked" the same size -- which is exactly the debugging-by-eye loop this
// project is trying to get out of. It is also, as it happens, pure
// arithmetic, so it can simply be asserted.
//
// Same CommonJS-script-that-dynamic-imports-an-.mjs shape as
// tools/assemble-shader.js.

const assert = require('assert');
const path = require('path');

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok    ${name}`);
  } catch (e) {
    failures.push({ name, e });
    console.log(`  FAIL  ${name}`);
    console.log(String(e.message).split('\n').map(l => `        ${l}`).join('\n'));
  }
}

// Relative closeness, so the tolerance means the same thing at tau~0.5 and at
// I_BODY~1e9. Exact-zero expectations fall back to an absolute compare.
function close(actual, expected, rel, msg) {
  const tol = expected === 0 ? rel : Math.abs(expected) * rel;
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${msg}: got ${actual}, expected ${expected} (|diff|=${Math.abs(actual - expected)} > tol=${tol})`
  );
}

async function main() {
  const CP = await import(path.join(__dirname, '..', 'card-params.mjs'));
  const {
    CARD_PARAM_DEFAULTS, DENSE_DEFAULT_RES_LOG2, AMR_DEFAULT_RES_LOG2,
    AMR_DEFAULT_LEVELS, AMR_EQUIVALENT_DENSE_RES_LOG2, RES_LOG2_MIN, RES_LOG2_MAX,
    tauFromReynolds, reynoldsFromTau, tauAtLevel, deriveCardParams,
    parseCardParams, parseResLog2,
  } = CP;

  const D = CARD_PARAM_DEFAULTS;
  const EPS = 1e-12;   // "exact modulo IEEE754 reassociation"
  const LOOSE = 1e-9;

  // main.js's pre-refactor hardcoded literals, spelled out here rather than
  // read off CARD_PARAM_DEFAULTS. The legacy cases below exist to prove the
  // parameterization can still EXPRESS the old system exactly; riding on the
  // live defaults instead would make them silently re-target themselves the
  // moment those defaults are retuned (as they were when the shipped card
  // moved to Pesavento & Wang's Fig. 2 case), testing nothing.
  // RE is 2133.333, not the 1066.667 that used to be stored, because Re is
  // now the paper's chord-based 2*u_t*a/nu. Same physical system, same tau:
  // the old number WAS this one, just expressed in the module's old half-Re
  // units. That this still lands on TAU=0.509 is the check that the
  // convention change was a pure relabeling and not a change of regime.
  const LEGACY = { BLOCKAGE: 2.0, ASPECT: 0.125, I_STAR: 0.34, RE: 2133.333, U_T: 0.05 };

  // ── 1. Legacy constants ────────────────────────────────────────────────────
  // The parameterization was introduced as a refactor of hardcoded values, so
  // its first duty is to reproduce them. These are main.js's pre-refactor
  // literals: A=64, B=8 at W=256, and TAU=0.509 for "Re ~ 1100".
  test('legacy defaults: dense page at W=256 reproduces A=64, B=8, TAU=0.509', () => {
    const p = deriveCardParams({ W: 256, ...LEGACY });
    close(p.A, 64, EPS, 'A');
    close(p.B, 8, EPS, 'B');
    close(p.TAU, 0.509, 1e-5, 'TAU');
  });

  test('legacy defaults: derived body quantities match the closed forms', () => {
    const { A, B, RHO_B, MASS, I_BODY, G_LU, G_EFF } = deriveCardParams({ W: 256, ...LEGACY });
    // Independently recomputed here from the paper's formulas rather than
    // copied from the module, so a sign/factor slip in either shows up.
    const rho = LEGACY.I_STAR * 2 * A ** 3 / (B * (A ** 2 + B ** 2));
    close(RHO_B, rho, EPS, 'RHO_B');
    assert.ok(RHO_B > 1, `RHO_B must exceed fluid density 1.0, got ${RHO_B}`);
    close(MASS, rho * Math.PI * A * B, EPS, 'MASS');
    close(I_BODY, rho * Math.PI * A * B * (A ** 2 + B ** 2) / 4, EPS, 'I_BODY');
    close(G_LU, LEGACY.U_T ** 2 / (Math.PI * B * (rho - 1)), EPS, 'G_LU');
    close(G_EFF, G_LU * (1 - 1 / rho), EPS, 'G_EFF');
    assert.ok(G_EFF > 0 && G_EFF < G_LU, `G_EFF should be a positive fraction of G_LU, got ${G_EFF} vs ${G_LU}`);
  });

  // ── 2. tau <-> Re is a genuine inverse pair ────────────────────────────────
  // The TAU slider back-solves RE through reynoldsFromTau and then lets
  // recalculate() re-derive TAU from it, so a non-inverse pair would make the
  // slider drift its own value on every drag.
  test('tauFromReynolds/reynoldsFromTau round-trip in both directions', () => {
    for (const re of [20, 40, 100, 200, 1066.667, 3000]) {
      for (const a of [8, 32, 64, 256]) {
        for (const u of [0.01, 0.04, 0.05, 0.1]) {
          const tau = tauFromReynolds(re, a, u);
          close(reynoldsFromTau(tau, a, u), re, LOOSE, `Re round-trip (re=${re},a=${a},u=${u})`);
          close(tauFromReynolds(reynoldsFromTau(tau, a, u), a, u), tau, LOOSE, `tau round-trip (re=${re},a=${a},u=${u})`);
        }
      }
    }
  });

  test('every default config sits strictly above the BGK stability floor', () => {
    for (let r = RES_LOG2_MIN; r <= RES_LOG2_MAX; r++) {
      const { TAU } = deriveCardParams({ W: 1 << r, ...D });
      assert.ok(TAU > 0.5, `resLog2=${r}: TAU=${TAU} is at or below the BGK floor 0.5`);
    }
  });

  // ── 3. tauAtLevel is the Dupuis-Chopard relation ───────────────────────────
  test('tauAtLevel matches the closed form 0.5 + 2^m*(tau0-0.5)', () => {
    for (const tau0 of [0.5005, 0.5045, 0.509, 0.52, 0.6]) {
      for (let m = 0; m <= 4; m++) {
        close(tauAtLevel(tau0, m), 0.5 + 2 ** m * (tau0 - 0.5), LOOSE, `tauAtLevel(${tau0},${m})`);
      }
    }
  });

  test('tauAtLevel(tau0, 0) is the identity (L0 is its own level)', () => {
    for (const tau0 of [0.5005, 0.509, 0.6]) assert.strictEqual(tauAtLevel(tau0, 0), tau0);
  });

  // Re is what refinement is supposed to PRESERVE: level m has 2^m times the
  // cells across the body and 2^m times the lattice viscosity, so the
  // Reynolds number the body actually sees is level-independent. If this ever
  // failed, refining would silently change the flow regime.
  test('Re is invariant down the level hierarchy', () => {
    for (const W of [128, 256, 512]) {
      for (const re of [20, 100, 1066.667]) {
        const { A, TAU } = deriveCardParams({ W, ...D, RE: re });
        for (let m = 0; m <= 3; m++) {
          const reAtM = reynoldsFromTau(tauAtLevel(TAU, m), A * 2 ** m, D.U_T);
          close(reAtM, re, LOOSE, `Re at level ${m} (W=${W}, re=${re})`);
        }
      }
    }
  });

  // ── 4. THE CROSS-PAGE EQUIVALENCE CLAIM ────────────────────────────────────
  // This is the headline assertion. Dense at resLog2=R vs AMR at
  // resLog2=R-(n-1) with n levels, same physical parameters:
  //   (a) the AMR's FINEST level resolves the card at exactly the dense
  //       page's cell count across the body, and
  //   (b) the AMR's finest-level tau equals the dense page's tau,
  // so the two are the same physical system at the same resolution AT THE
  // BODY -- the AMR page simply carries a coarser far field. Anything else
  // and the "AMR does less work for the same physics" comparison is invalid,
  // because it would be doing DIFFERENT physics.
  test('cross-page equivalence: AMR finest level matches the dense reference', () => {
    const grid = [];
    for (const R of [8, 9, 10]) {
      for (const n of [2, 3, 4]) {
        if (R - (n - 1) < RES_LOG2_MIN) continue;
        for (const blockage of [1.5, 2.0, 8.0]) {
          for (const aspect of [0.05, 0.125, 0.4]) {
            for (const re of [20, 100, 1066.667, 3000]) {
              for (const u of [0.02, 0.05]) {
                grid.push({ R, n, blockage, aspect, re, u });
              }
            }
          }
        }
      }
    }
    assert.ok(grid.length > 100, `expected a real sweep, got ${grid.length} cases`);

    for (const { R, n, blockage, aspect, re, u } of grid) {
      const phys = { BLOCKAGE: blockage, ASPECT: aspect, I_STAR: D.I_STAR, RE: re, U_T: u };
      const dense = deriveCardParams({ W: 1 << R, ...phys });
      const amr = deriveCardParams({ W: 1 << (R - (n - 1)), ...phys });
      const label = `R=${R},levels=${n},blockage=${blockage},aspect=${aspect},re=${re},ut=${u}`;

      // (a) body resolution at the finest level
      close(amr.A * 2 ** (n - 1), dense.A, LOOSE, `${label}: finest-level A`);
      close(amr.B * 2 ** (n - 1), dense.B, LOOSE, `${label}: finest-level B`);

      // (b) finest-level tau
      close(tauAtLevel(amr.TAU, n - 1), dense.TAU, LOOSE, `${label}: finest-level tau`);

      // and therefore the same Reynolds number, from both sides independently
      close(reynoldsFromTau(dense.TAU, dense.A, u), re, LOOSE, `${label}: dense Re`);
      close(reynoldsFromTau(tauAtLevel(amr.TAU, n - 1), amr.A * 2 ** (n - 1), u), re, LOOSE, `${label}: AMR finest Re`);
    }
  });

  // AMR_EQUIVALENT_DENSE_RES_LOG2 is the dense run the AMR default is a claim
  // ABOUT, so it has to satisfy the pairing exactly.
  test('AMR_EQUIVALENT_DENSE_RES_LOG2 is the AMR default\'s matched dense run', () => {
    assert.strictEqual(
      AMR_EQUIVALENT_DENSE_RES_LOG2,
      AMR_DEFAULT_RES_LOG2 + (AMR_DEFAULT_LEVELS - 1),
      `equivalent dense res (${AMR_EQUIVALENT_DENSE_RES_LOG2}) must be the AMR default ` +
      `(${AMR_DEFAULT_RES_LOG2}) plus levels-1 (${AMR_DEFAULT_LEVELS - 1})`
    );
    assert.ok(AMR_EQUIVALENT_DENSE_RES_LOG2 <= RES_LOG2_MAX,
      `equivalent dense res ${AMR_EQUIVALENT_DENSE_RES_LOG2} exceeds the clamp ${RES_LOG2_MAX}, ` +
      `so index.html could not actually be run at it`);
    const dense = deriveCardParams({ W: 1 << AMR_EQUIVALENT_DENSE_RES_LOG2, ...D });
    const amr = deriveCardParams({ W: 1 << AMR_DEFAULT_RES_LOG2, ...D });
    close(amr.A * 2 ** (AMR_DEFAULT_LEVELS - 1), dense.A, LOOSE, 'equivalent run: finest-level A');
    close(tauAtLevel(amr.TAU, AMR_DEFAULT_LEVELS - 1), dense.TAU, LOOSE, 'equivalent run: finest-level tau');
    // Pins the concrete numbers the rest of the project's comments quote.
    // The AMR page's L0 card is A=16,B=2 at its W=256; one octave up is
    // A=32,B=4 -- Pesavento & Wang's reference ellipse in lattice units --
    // and two octaves up is A=64,B=8, the equivalent dense run at W=1024.
    close(amr.A, 16, EPS, 'AMR default A');
    close(amr.B, 2, EPS, 'AMR default B');
    close(amr.TAU, 0.504364, 1e-4, 'AMR default L0 tau');
    close(dense.A, 64, EPS, 'equivalent dense A');
    close(dense.B, 8, EPS, 'equivalent dense B');
    close(dense.TAU, 0.517455, 1e-4, 'equivalent dense tau');
  });

  // The shipped dense default is deliberately NOT the equivalent run -- see
  // DENSE_DEFAULT_RES_LOG2's comment: res 10 allocates 16x the cells up
  // front and does not fit on the mobile target, so index.html would fail to
  // start there rather than merely run slowly. Asserted rather than left to
  // the comment, so that "the defaults are not a matched pair" stays a stated
  // property of the module: someone re-pairing them would have to delete this
  // test and read why it existed, instead of quietly reintroducing a page
  // that cannot boot on a phone.
  test('shipped dense default is deliberately below the equivalent run', () => {
    assert.ok(
      DENSE_DEFAULT_RES_LOG2 < AMR_EQUIVALENT_DENSE_RES_LOG2,
      `dense default (${DENSE_DEFAULT_RES_LOG2}) is expected to sit BELOW the AMR-equivalent ` +
      `res (${AMR_EQUIVALENT_DENSE_RES_LOG2}) for mobile allocation headroom`
    );
    // It still shows the SAME physical card -- only the grid it is sampled on
    // differs. That is what keeps it a usable coarse view rather than a
    // different experiment.
    const denseDefault = deriveCardParams({ W: 1 << DENSE_DEFAULT_RES_LOG2, ...D });
    const amr = deriveCardParams({ W: 1 << AMR_DEFAULT_RES_LOG2, ...D });
    close(denseDefault.B / denseDefault.A, D.ASPECT, LOOSE, 'dense default e');
    close(reynoldsFromTau(denseDefault.TAU, denseDefault.A, D.U_T), D.RE, LOOSE, 'dense default Re');
    // At the shipped defaults both pages happen to share one L0 grid, so the
    // dense page is exactly the AMR page's coarsest level.
    close(denseDefault.A, amr.A, EPS, 'dense default A == AMR L0 A');
    close(denseDefault.TAU, amr.TAU, LOOSE, 'dense default tau == AMR L0 tau');
  });

  // The shipped card is meant to BE the paper's Fig. 2 case. Its two purely
  // dimensionless quantities are stored directly, so they can be compared to
  // the published values without any unit reasoning -- see
  // 2004_PRL_Pesavento_Wang.pdf p.2, which gives Re=1100, I*=0.17, e=0.125
  // with I* = b(a^2+b^2)rho_b/(2a^3 rho_f) and e = b/a.
  //
  test('shipped defaults are the paper Fig. 2 card (Re=1100, I*=0.17, e=0.125)', () => {
    close(D.RE, 1100, EPS, 'Re (paper Fig. 2)');
    close(D.I_STAR, 0.17, EPS, 'I* (paper Fig. 2)');
    close(D.ASPECT, 0.125, EPS, 'e = b/a (paper Fig. 2)');
    // e is a ratio of derived lengths too, at either page's default width.
    for (const r of [DENSE_DEFAULT_RES_LOG2, AMR_DEFAULT_RES_LOG2]) {
      const { A, B } = deriveCardParams({ W: 1 << r, ...D });
      close(B / A, 0.125, LOOSE, `e from derived A,B at res=${r}`);
    }
  });

  // THE CONVENTION ITSELF. reynoldsFromTau's factor of 2 is the whole content
  // of the fix, and it is invisible in every round-trip test above (both
  // directions carry it, so they stay inverses either way). Assert it against
  // the paper's own definition, recomputed here from nu:
  //   2004_PRL_Pesavento_Wang.pdf p.2: Re = 2*u_t*a/nu, keyed to the chord 2a.
  // Without this, silently reverting to u_t*a/nu would pass the entire rest
  // of this file -- which is exactly how the half-Re bug survived before.
  test('Re is the paper chord-based 2*u_t*a/nu, not the semi-axis u_t*a/nu', () => {
    for (const tau of [0.5005, 0.504364, 0.509, 0.52, 0.6]) {
      for (const a of [4, 16, 32, 64]) {
        for (const u of [0.02, 0.05, 0.1]) {
          const nu = (tau - 0.5) / 3;
          close(reynoldsFromTau(tau, a, u), 2 * u * a / nu, LOOSE,
            `chord-based Re (tau=${tau},a=${a},u=${u})`);
          close(tauFromReynolds(2 * u * a / nu, a, u), tau, LOOSE,
            `chord-based tau (tau=${tau},a=${a},u=${u})`);
        }
      }
    }
  });

  // The shipped card must BE Re=1100 where the paper's ellipse actually
  // lives: the AMR page's L1, whose A=32,B=4 is the published a,b in lattice
  // units. Re is level-invariant (asserted above), so this holds at every
  // rung -- but pinning it at that specific one is what ties the shipped
  // configuration to the published figure rather than to an arbitrary level.
  test('the paper ellipse rung (A=32,B=4) runs at the paper Reynolds number', () => {
    const amr = deriveCardParams({ W: 1 << AMR_DEFAULT_RES_LOG2, ...D });
    close(amr.A * 2, 32, EPS, 'L1 A');
    close(amr.B * 2, 4, EPS, 'L1 B');
    close(reynoldsFromTau(tauAtLevel(amr.TAU, 1), amr.A * 2, D.U_T), 1100, LOOSE, 'Re at L1');
  });

  // ── 5. Invariance properties the parameterization exists to provide ────────
  test('RE is held fixed when card size or U_T changes', () => {
    // The old A/B-slider parameterization drifted Re silently whenever the
    // card was resized, because TAU was independent state. Now TAU follows.
    const base = { W: 256, ...D };
    for (const blockage of [1.0, 2.0, 4.0, 16.0]) {
      for (const u of [0.02, 0.05, 0.1]) {
        const p = deriveCardParams({ ...base, BLOCKAGE: blockage, U_T: u });
        close(reynoldsFromTau(p.TAU, p.A, u), D.RE, LOOSE, `Re held at blockage=${blockage},ut=${u}`);
      }
    }
  });

  test('card size relative to the domain is resolution-independent', () => {
    // A/W must depend only on BLOCKAGE -- this is what stops the card
    // silently changing physical size when the Resolution slider moves.
    for (const blockage of [1.5, 2.0, 8.0]) {
      for (let r = RES_LOG2_MIN; r <= RES_LOG2_MAX; r++) {
        const W = 1 << r;
        const { A, B } = deriveCardParams({ W, ...D, BLOCKAGE: blockage });
        close(A / W, 1 / (2 * blockage), LOOSE, `A/W at blockage=${blockage}, res=${r}`);
        close(B / A, D.ASPECT, LOOSE, `B/A at blockage=${blockage}, res=${r}`);
      }
    }
  });

  test('RHO_B clamp keeps the card denser than the fluid', () => {
    // A thick, low-I* card drives the closed form below 1.0, where G_LU's
    // 1/(RHO_B-1) would blow up or go negative. The clamp is the guard.
    const p = deriveCardParams({ W: 256, ...D, ASPECT: 0.5, I_STAR: 0.05 });
    assert.ok(p.RHO_B >= 1.05, `RHO_B should be clamped up to 1.05, got ${p.RHO_B}`);
    assert.ok(Number.isFinite(p.G_LU) && p.G_LU > 0, `G_LU must stay finite and positive, got ${p.G_LU}`);
    assert.ok(Number.isFinite(p.G_EFF) && p.G_EFF > 0, `G_EFF must stay finite and positive, got ${p.G_EFF}`);
  });

  // ── 6. URL parsing ─────────────────────────────────────────────────────────
  test('parseCardParams falls back to defaults on absent/garbage input', () => {
    assert.deepStrictEqual(parseCardParams(new URLSearchParams('')), { ...D });
    assert.deepStrictEqual(parseCardParams(new URLSearchParams('blockage=&re=nonsense')), { ...D });
  });

  test('parseCardParams reads every documented query parameter', () => {
    const got = parseCardParams(new URLSearchParams(
      'blockage=4&aspect=0.2&istar=0.5&re=250&ut=0.03'));
    assert.deepStrictEqual(got, { BLOCKAGE: 4, ASPECT: 0.2, I_STAR: 0.5, RE: 250, U_T: 0.03 });
  });

  test('parseResLog2 clamps to the range both pages enforce', () => {
    assert.strictEqual(parseResLog2(new URLSearchParams('res=3'), 8), RES_LOG2_MIN);
    assert.strictEqual(parseResLog2(new URLSearchParams('res=99'), 8), RES_LOG2_MAX);
    assert.strictEqual(parseResLog2(new URLSearchParams('res=9'), 8), 9);
    assert.strictEqual(parseResLog2(new URLSearchParams(''), 8), 8);
  });

  // ── Report ─────────────────────────────────────────────────────────────────
  console.log();
  if (failures.length) {
    console.log(`card-params: FAILED (${failures.length} failing, ${passed} passing)`);
    process.exit(1);
  }
  console.log(`card-params: ${passed} test(s) passed`);
}

main().catch(e => { console.error(e); process.exit(1); });
