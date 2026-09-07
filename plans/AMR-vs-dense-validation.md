# Dense-vs-AMR full-field correctness: status and continuation notes

Paused 2026-07-11. This is a working-notes doc, not a milestone plan like
`AMR-multilevel.md` — written to let this work resume cold, either in a new
session or by someone else. Everything below reflects the actual measured
state as of the last commit (`faa11e5`, "Add dense-vs-AMR full-field
correctness tool"), not aspirations.

## Why this exists

User wants: run a scenario at high resolution (e.g. 1024²) on the dense
reference (`main.js`/`main-cylinder.js`), run AMR (`main-amr.js`/
`main-cylinder-amr.js`) refined down to that *same* physical resolution at
the cylinder surface (coarser in the bulk, to demonstrate AMR's actual
purpose), and directly compare results — not just each solver
independently matching literature Cd/St (which `tools/validate-cylinder.js`
already does), but a sharper apples-to-apples check that AMR reproduces
what the dense solver computes at matched resolution. Full field
(velocity/vorticity) comparison was explicitly chosen over Cd/St-only, and
the tool is standalone/opt-in (not folded into `validate-all.js`'s routine
sweep — a 1024² dense run is much more expensive than that suite's default
res=9 configs).

The full design plan (options considered, rejected alternatives) is in this
session's Claude Code plan-mode output, not duplicated here — this doc is
about *current state and what to do next*, not the original design
rationale (though the "why" of specific decisions is inlined below where it
matters for continuation).

## What's built (all committed, `faa11e5`)

- **`tools/lib/amr-resolution-mapping.js`** — `deriveAMRParams({targetResLog2,
  nLevels, blockage, u0, re, allowUnvalidatedLevels, allowMarginalTau})`
  derives `baseResLog2 = targetResLog2 - (nLevels-1)` and validates it
  against three independent gates: the `[7,11]` resLog2 clamp both pages
  enforce, the project's own validated-levels cap (currently `nLevels<=3`,
  read from `main-cylinder-amr.js`'s own comment above `N_LEVELS` — **that
  comment is the living source of truth per CLAUDE.md, re-read it before
  trusting the cap hardcoded in amr-resolution-mapping.js**), and a NEW
  check I derived this session: base-grid relaxation time
  `tauBase = 0.5 + 0.005*(W_base/Re)` (closed form of
  `main-cylinder.js`'s own `tauFromRe`, using project defaults
  `BLOCKAGE=24, U0=0.04`) must stay comfortably above the BGK stability
  floor (0.5) — warns below 0.52, throws (needs `--allowMarginalTau`) below
  0.505. This matters because every *existing* validated AMR config in this
  project defaults to `?res=9` (W_base=512) regardless of level count, so
  "nLevels<=3 is validated" has never actually been exercised at the
  smaller base grids a fixed high target resolution forces you into.
  `deriveSharedURLParams`/`buildDenseUrl`/`buildAMRUrl` build both pages'
  URLs from one shared param set so they can't drift apart.

- **`tools/lib/field-reconstruct.js`** — absorbed `tools/amr-diff.js`'s old
  guts (`b64ToFloat32`, `rawIndex`, `unshiftField`, `rhoFromF`,
  `vorticityField`, `diffStats`, `loadFields`→`loadDenseFields`) as exports,
  plus the new piece: `reconstructAMRToResolution(snapshot, targetResLog2)`
  walks an AMR `debugSnapshotSave()` payload's quadtree pools and paints a
  uniform target-resolution grid, per-quadrant recursive (`paintQuad`),
  derived from `shaders/amr_render.wgsl`'s own per-pixel compositing
  algorithm generalized from its hardcoded 2-tier wiring to arbitrary
  `numLevels`. Verified by a **pure-Node fixture test**
  (`tools/test-field-reconstruct.js`, no GPU) with a hand-built mixed-depth
  snapshot (one L1 block whose 4 L2 quadrants are
  `[active,-1,active,-1]`) — asserts exact per-cell placement at both the
  finest-injection and standing-in-fan-out branches, plus a poison-value
  leak check and an exact `finestCoverageFraction` assertion. **This test
  passes and gives real confidence the reconstruction geometry/indexing is
  correct** — the large field-diff numbers seen in real runs (below) are
  NOT reconstruction bugs, see Findings.
  STALE AS OF 2026-09-06 — the moving-window restriction described here has
  since been LIFTED. `reconstructAMRToResolution` now stays in buffer space
  for the whole quadtree walk and applies the buffer→window conversion at
  the point of writing each painted cell (`paintUniform`), verified against
  a real moving-window snapshot (`main-reentry-amr.js`'s falling card, with
  off_x/off_y routinely nonzero) via `tools/analyze-reentry-seam.js`. See
  the function's own header comment, which is current. Nonzero
  `cardState[22]/[23]` is handled; it no longer throws.

- **`tools/lib/amr-cost.js`** — `computeCostSavings` implements "AMR did X%
  of the work a dense grid at the same finest resolution would have" in
  cell-substep units, using `activeCountsByLevel` (real block counts,
  including parents-with-active-children, since the GPU dispatch still runs
  those) and level substep multipliers `2^m`. Sanity-checked: sparse
  refinement → ~3% cost (97% savings); a fully-refined domain costs
  *slightly more* than dense (~112%, ghost-cell/multi-rate overhead) —
  confirms the formula isn't trivially returning "savings" for everything.

- **`tools/lib/browser-lifecycle.js`** — Chrome/HTTPS-server lifecycle
  helpers extracted out of `validate-all.js` (pure refactor, verified
  behavior-identical by re-running `validate-all.js --configs=index-boot`
  and `--configs=amr-N2-bounceback` after the extraction — both still PASS).

- **`main-cylinder.js`** gained `debugSnapshotSave` (didn't exist before —
  only the AMR side had field-snapshot capability). Needed `COPY_SRC` added
  to `f_a`/`velBuf`'s buffer usage flags (weren't copyable out before).
  Verified end-to-end against a live GPU: buffer copies succeed,
  `loadDenseFields` decodes it, fields are finite and physically sensible.

- **`tools/validate-amr-vs-dense.js`** — the top-level CLI. Flags: `--res=`
  (target resLog2 list), `--levels=` (nLevels list, default `2,3`), `--re=`
  (default `20,40` — steady cases, see Findings for why), `--mode=`
  (`adaptive`/`fullrefine`/`both`, default `both`), `--bounceback`/
  `--diffuse` (**defaults to bounceback** — see Findings, this is load-
  bearing not stylistic), `--tol=`/`--tolOmega=`/`--tolFullrefine=`,
  `--allowUnvalidatedLevels`, `--allowMarginalTau`, `--dryRun`,
  `--saveSnapshots=<dir>`. Owns Chrome/server lifecycle like `validate-all.js`
  (one tab, `Page.navigate` between configs). Caches the dense reference per
  `(targetRes, re)` and reuses it across the `--levels=`/`--mode=` sweep
  (the dense leg doesn't depend on AMR params, so this avoids re-running the
  expensive part `|levels|x|modes|` times).

  `fullrefine` mode force-activates every block at every level
  (`setAutoRefine(false)` + a single batched in-page `Runtime.evaluate`
  loop over `debugActivateBlock`, not one CDP round-trip per block) before
  running — this is the "zero coarse/fine interface anywhere" case meant to
  isolate reconstruction-code correctness from AMR's own refine/coarsen
  accuracy.

## Findings so far (the actual point of this session)

### 1. RESOLVED (2026-09-06): the diffuse-coupling failure was two force bugs plus one method limitation

**This finding is closed. The text below replaces the original "not yet
root-caused ... could be a real code regression, or a GPU/driver-specific
numerical issue" entry, which sent the reader looking for the wrong thing.**

The original symptom, on unmodified `main` at the default Re=100:

| config | Cd measured | Cd target |
|---|---|---|
| dense-reference | 1.908 | 1.35±0.15 |
| amr-N2-diffuse | 0.650 | 1.35±0.15 |
| amr-N3-diffuse | 0.085 | 1.35±0.15 |

It was three separate things. Ruled OUT first, by measurement (not by
reading): finest-wins masking is correct, and the chi band width is not the
cause at level 2 (sweeping `?kEpsPool` over 1.5/3/6/12 barely moved it).
The decisive observation was that **bounce-back gets the same levels,
masking, registration and dispatch right** (L1 Cd=1.277 at N=2, L2 Cd=1.318
at N=3) while diffuse collapsed -- which isolates the fault to the diffuse
branch of the per-level force shaders. `debugForceBreakdown` is the tool
that localizes this; use it first next time.

1. **Cross-level weight (fixed).** The diffuse branch weighted each level by
   `dx_L^2` -- the 2D volume measure -- and dropped the timestep. Under
   acoustic scaling `dt_L = dx_L`, and force is `mass*du/dt`, so the weight
   is `dx_L^2/dx_L = dx_L^1`. Level L runs `2^L` substeps per L0 macro-step
   but the force pass runs once and reads one substep, so the missing factor
   is exactly `2^L`. See `shaders/amr_force1.wgsl`'s header point 2.

2. **Fixed-point truncation (fixed).** The reduction `atomicAdd`s one
   TRUNCATED i32 per WORKGROUP, so any workgroup whose partial sum falls
   below one fixed-point unit contributes exactly zero -- systematic loss,
   worst at depth where the dx weight is smallest. `FSCALE` 1e4 -> 1e7. See
   `shaders/amr_force1_pool.wgsl`'s FSCALE comment; it is a stopgap, and a
   deeper hierarchy will need float atomics or a two-stage reduction.

   Contributions measured separately (instantaneous Cd, Re=100):

   | config | orig | FSCALE only | dx^1 only | both |
   |---|---|---|---|---|
   | N=2 | 0.645 | 0.792 | 1.430 | 1.593 |
   | N=3 | 0.076 | 0.347 | 0.943 | 1.390 |

3. **Interface width (NOT a bug, still failing).** `get_phi` is an exact
   Euclidean distance in lattice units for a circle and epsilon is a fixed
   1.5, so the chi band is ~±4 cells wide however big the body is; the
   effective hydrodynamic radius exceeds nominal by a roughly
   resolution-independent ~2.5-4.5 cells. Cd high AND St low together is the
   signature. Confirmed by convergence: dense Re=100 goes Cd 1.908 -> 1.597
   and St 0.126 -> 0.148 from res=9 to res=10, both monotonically toward
   1.35/0.165, excess radius near-constant in cells while R doubles.

**Current state after the fixes** -- every config now shows only #3, ordered
by its own epsilon:

| config | epsilon | Cd | result |
|---|---|---|---|
| dense-reference | 1.5 | 1.951 | FAIL |
| amr-N2-diffuse | 0.75 | 1.620 | FAIL |
| amr-N3-diffuse | 0.375 | — | **PASS** |

`amr-N3-diffuse` passes for the first time; both bounce-back configs and
both AMR invariant checks still pass. A diffuse Cd/St failure at res=9 is
now EXPECTED for dense and N=2 -- see `main-cylinder-amr.js`'s own comment
above `N_LEVELS`, which is the living source of truth. Closing the gap needs
a calibrated effective-radius offset or a sharper epsilon; not attempted.

### 2. Field reconstruction machinery is sound; steady-flow field comparison mostly works

`--res=9 --levels=2 --re=20 --mode=fullrefine` (bounceback, steady, no
perturbation, cleanest self-consistency case: dense@512 vs AMR base=256
fully force-refined, so its finest level exactly matches dense — zero
coarse/fine interface anywhere):

- Cd matched to 0.5% (dense=2.180, amr=2.191) — strong.
- `ux`/`rho` matched tight: relL2 9.7e-3 / 5.1e-5.
- `uy`/`omega` did NOT meet `--tolFullrefine` (relL2 0.22 / 0.70) — but
  traced by hand, not hand-waved: both solvers' own peak `|uy|` locations
  and magnitudes agree to within 1 cell and <2%. The worst individual-cell
  disagreements are a small, localized cluster (~0.13% of cells, ~336 out
  of 262144) right near the peak-shear-layer region, NOT spread across the
  domain and NOT at tile/domain-edge boundaries (both explicitly ruled out
  as reconstruction-indexing-bug explanations by direct inspection — see
  the investigation trail below if you need to redo this check).
  Re=20 is steady and top/bottom-symmetric (no perturbation): `uy`'s own
  magnitude is tiny (~0.02 peak) almost everywhere except right at that one
  shear layer, so relL2 (normalized by the reference field's own norm) is a
  harsh metric here — a handful of cells where two INDEPENDENTLY-implemented
  WGSL kernels (`lbm_step.wgsl` vs `amr_step.wgsl`/`amr_step1.wgsl`)
  disagree by a small absolute amount, right where gradients are sharpest,
  dominates the norm of an otherwise-near-zero field.
  **Conclusion for steady Re: Cd/ux/rho are reliable corroborating
  signals; uy/omega relL2 tolerances need to be interpreted magnitude-
  aware, not taken as a flat pass/fail at face value.**

Investigation trail (if you need to redo/extend this): saved snapshots via
`--saveSnapshots=`, then in Node: `loadDenseFields`/`reconstructAMRToResolution`
on the two snapshot JSONs, computed per-column max-abs-diff and a
histogram of anomaly locations mod tile size (16) to check for a
boundary/indexing bug — ruled out (anomalies aren't concentrated at tile
edges, aren't periodic in the tile-boundary sense, and peak-field locations
between solvers agree to 1 cell). This was real, hands-on numerical
detective work, not guessing — worth redoing the same way if a future
"unexpected relL2" needs triage, rather than assuming it's automatically
either "definitely a bug" or "definitely fine."

### 3. Instantaneous field comparison does NOT work at shedding Re — this is the open question

`--res=9 --levels=2 --re=100 --mode=both` (bounceback, shedding):

- Cd stayed well-behaved: dense=1.327, adaptive=1.442, fullrefine=1.416 —
  **all three independently within the literature band (1.35±0.15)**, and
  AMR within ~7-9% of dense. Corroborates AMR physics at Re=100, consistent
  with the existing `validate-cylinder.js` suite.
- But `uy` relL2 got MUCH WORSE than at Re=20: 1.63 (adaptive) / 1.53
  (fullrefine) — i.e. over 150% relative error, the opposite of what was
  predicted going in (the hypothesis was that Re=100's domain-wide-magnitude
  `uy` field would make relL2 a fairer metric than Re=20's near-zero one).

**Why, and this is the actual finding**: Re=100 sheds periodically. Both
snapshots are a single INSTANT captured after independently running dense
and AMR forward from the same initial condition but through two physically
different code paths (different discretization, different step count even
— dense runs ~70k steps, AMR ~35k steps at half the... no wait, they're
matched via `computeWindow` against each page's own resolution-scaled `D`,
so total *physical time* covered should match, but the two solvers'
transient-to-limit-cycle behavior isn't necessarily phase-locked). By the
time you snapshot near the end of the run, dense's vortex might be
shedding from the top while AMR's is shedding from the bottom of the same
limit cycle — two CORRECT, physically-equivalent periodic solutions,
sampled at uncorrelated phase, will show huge instantaneous field
disagreement despite both being right. This is fundamentally different
from the Re=20 finding (a metric-normalization artifact on real near-
agreement) — this is a phase-mismatch problem inherent to comparing
single instantaneous snapshots of an unsteady/periodic flow between two
independently-timed solvers.

**Practical implication, not yet acted on**: raw instantaneous
field-snapshot diffing (as currently implemented) is only a meaningful
check at STEADY Re (20, 40) — where it worked well modulo the magnitude-
normalization caveat above. At shedding Re (100, 200), Cd/St (already
time-averaged over the measurement window, which is what
`validate-cylinder.js`/`cylinder-metrics.js`'s `analyze()` already computes)
remain the right/only currently-meaningful check; the new field-diff
machinery shouldn't be trusted there without further work.

## Open threads / what to do next

**Status update 2026-09-06.** Option 3 below (root-cause the diffuse
failure) is DONE — see the rewritten Finding #1. Option 2 (make comparison
meaningful at shedding Re) has been superseded by a different and cheaper
approach: rather than phase-ALIGNING two independently-timed runs, start
both solvers from ONE state and measure short-horizon divergence, which
removes phase from the comparison entirely. `tools/lib/dense-to-amr.js`
(the injector, with `tools/test-dense-to-amr.js` round-tripping it against
the fixture-tested reconstructor) is built; the driver that runs both legs
forward and diffs them is the remaining piece. Option 1's resolution ladder
is untouched and still open.

Original options, as they stood when this was paused:

1. **Accept the current split** (field-diff meaningful at steady Re only;
   Cd/St covers shedding Re, which is already what `validate-cylinder.js`
   does) and move forward with the originally-planned resolution ladder
   using steady Re for field checks:
   - `--res=9 --levels=3 --re=20` (or 40) — more genuinely untested
     territory (both the AMR-mechanism validated-levels cap and the new
     base-tau margin get tighter at levels=3; tauBase≈0.506 at Re=200,
     ≈0.564 at Re=20 per the table computed this session — recompute via
     `deriveAMRParams` before running, don't trust stale numbers).
   - Scale to the user's actual original target, `--res=10` (1024²), once
     smaller cases are clean. Expensive — use `--dryRun` first to see the
     `fullrefine` memory estimate before committing to a real run.
   - `adaptive` mode hasn't been exercised at all except the one Re=100 run
     above (which is compromised by the phase-mismatch issue for its field
     numbers, though its Cd/cost numbers are still valid) — worth a clean
     steady-Re adaptive run to get a first honest cost-savings number
     (`workDoneVsDensePercent`) from a REAL (not artificially-fully-refined)
     AMR run.

2. **Invest in phase-aligned/time-averaged field comparison** to make
   instantaneous-field validation meaningful at shedding Re too. This is a
   real, separate piece of work (not a quick fix) — would need either (a)
   detecting a common phase reference (e.g. align on a Cl zero-crossing,
   similar to how `cylinder-metrics.js`'s Strouhal estimator already finds
   zero-crossings) and comparing phase-shifted snapshots, or (b) comparing
   time/phase-averaged fields (mean flow + RMS fluctuation amplitude, not
   instantaneous values) between the two solvers instead of raw
   snapshots. Not started.

3. **Root-cause the diffuse-coupling regression** (Finding #1) — separate
   from this tool's own scope entirely, but is the more consequential
   finding of the session in terms of project health (it means the
   project's own default validation command is currently red for its
   flagship non-bounce-back coupling method). Not started; not
   root-caused; not distinguished between "real regression" and
   "environment-specific" causes.

No decision has been made on which to pursue next — that's exactly where
this got paused.

## Practical notes for resuming

- All Node tools in this repo (`tools/*.js`) are CommonJS; `main*.js` are
  ES modules loaded by the HTML pages directly (no bundler, static site).
- A WebGPU-capable Chrome + `https.py` dev server need to be running for
  any of the GPU-driven tools; `validate-all.js`/`validate-amr-vs-dense.js`
  will launch both themselves if not already up (`ensureServer`/
  `ensureChrome` in `tools/lib/browser-lifecycle.js`), and tear down
  whatever they started when done (pass `--keepOpen` to leave Chrome open
  for follow-up ad-hoc CDP scripts, e.g. the kind used in the Finding #2/#3
  investigation trail above).
- `--saveSnapshots=<dir>` is the easiest way to capture raw data for
  offline investigation (Node scripts against `tools/lib/field-reconstruct.js`
  directly, no GPU needed) rather than re-running the GPU sweep every time
  you want to inspect something.
- Re-run `tools/test-field-reconstruct.js` (fast, no GPU) any time the
  reconstruction code changes, or any time a real-run result looks
  suspicious, to rule the reconstruction machinery in or out before
  suspecting AMR itself.
- `main-cylinder-amr.js`'s own comment above `N_LEVELS` is the living
  source of truth on which `?levels=N` combinations are currently
  considered validated — CLAUDE.md explicitly defers to it, and so does
  `amr-resolution-mapping.js`'s hardcoded `VALIDATED_MAX_LEVELS` constant,
  which should be re-checked against that comment if it's been a while.
