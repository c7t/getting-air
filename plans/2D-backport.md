# Back-porting the 3D fork's findings into the 2D solver

Written 2026-09-14 against `3d/m0-m1-dense-solver` @ cd106f8. The 3D branch
has touched **no 2D solver code at all** (`git diff main...HEAD` outside
`plans/`, `*d3*`, `*3d*` is `CLAUDE.md`, `.gitignore`, `tools/validate-all.js`,
`tools/lib/browser-lifecycle.js`, `tools/lib/url-query.js`), so everything
below is genuinely new work on the 2D side rather than a merge.

**The one-line summary.** Five of the six things worth bringing back are
architecture, not physics, and the sixth (the interface) is the only one that
moves a published number. The blocker is the same for all of them: **2D has
no GPU-free host statement of its own AMR geometry**, so every refactor here
is currently gated only by "does the cylinder still shed". 3D has
`d3-amr.mjs` + `tools/test-d3-amr.js` (25 mutation-checked assertions, no
GPU) and that is what made M4/M5 survivable. Build the 2D equivalent first.

---

## 1. The verdict, up front

| # | Finding (3D) | Applies to 2D? | Why it is worth it | Size |
|---|---|---|---|---|
| B0 | `d3-amr.mjs` + mutation-checked host tests | **DONE** | `amr2d.mjs` + `amr2d-gpu.mjs`, 34 GPU-free checks; deleted 680 lines of five duplicated checkers | M |
| B0b | The interface instrument (mass/momentum drift) | **DONE** | `tools/analyze-amr-interface.js`; measured B9 as 93% of the mass channel and the seam at 9.6x the no-interface floor in momentum | M |
| B1 | Post-collision Dupuis-Chopard `fneq` factor | **yes, a live bug** | 2D uses the pre-collision form on post-collision populations | S (code) / L (re-baseline) |
| B2 | `cascade21`: 2:1 as ONE closure on the WANT set | **DONE** | corner 2:1 violations 13 → 0 and L2's extent 80 → 160 L0 units on the shipped page; `?cascade=0` keeps the old path as a control | M |
| B3 | One kernel per stage + a `parent_{dense,pool}` accessor | **yes** | deletes ~1100 lines of near-duplicate WGSL across 6 kernel pairs | M |
| B3a | The same sweep in JS: the AMR pages duplicate each other | **largely DONE** | duplication outside `init`/`frame` 3,319 → 2,419 lines; byte-identical 1,129 → 613. Found a stranded comment and two always-true gates on the way | M |
| B4 | The body lives entirely on the finest level | **DONE** | coverage gate written for the two moving-body pages, the block predicate fixed, ~120 lines of masking + 3 bindings + an override deleted, and 3 refusals now under test | S |
| B5 | The window is a translation of the SPONGE | **half already true in 2D AMR, not in `main.js`** | removes window bookkeeping from 5 hot kernels; retires `card-total.mjs`'s problem | M |
| B6 | Explode/coalesce at the coarse/fine interface | **yes, but measure first** | 2D's interface is not conservative, and nothing in 2D measures that | L |
| B7 | The diffuse band ladder | **DONE** | `?kEps=` sweeps all nine chi sites; the band is FIRST order in 2D (3D's 2nd does not transfer) and both red cells are quantified band error, closing at `?kEps=0.375` | S |
| B8 | `SOLID_EQ` on the bounce-back path | **DONE** | exact on the deterministic leg; the interior it fixes is at 78% of free-stream today, so the landmine was armed | S |
| B9 | f32 lattice weights do not sum to 1 | **DONE** | `lattice-2d.mjs` derives the basis once (was eleven copies); mass injection halved, velocity drift removed; couette gates improved 26-53% | S (code) / L (re-baseline) |

Sizes are of the *change*, not of the validation. B1, B6 and B9 each move the
published benchmark surface and `main` **is** the site, so each needs its own
re-baseline pass.

---

## 2. What does NOT transfer, and why — so it is not re-argued

These came up while surveying and each has a reason to stay in 3D.

- **`SWEPT_FORCE` (D1) and the fresh-node refill.** Structurally impossible
  in 2D as shipped. `USE_BOUNCEBACK` is set from a URL parameter in exactly
  one file, `main-cylinder.js`, and that body is **pinned**; `main.js`,
  `main-amr.js`, `main-reentry.js` and `main-reentry-amr.js` never set the
  override, so every moving 2D body runs the **diffuse (chi)** coupling.
  `chi` never partitions cells into solid and fluid, so nothing ever changes
  hands and the whole covered/fresh-node class cannot arise. (plans/3D.md
  M8.2a's own text says 2D's moving bounce-back body is `index-reentry.html`;
  that is the superseded reading — CLAUDE.md's 2026-09-11 correction is the
  current one. Do not restore it.)
- **TRT.** Measured on branch `worktree-3d-trt` and NOT merged: it fixes the
  static wall and makes both of the moving card's real limits worse. Nothing
  to bring back except the lattice-weight finding, which is B9.
- **Ghost-free tiles.** Decided against in 2D first (`plans/ghost-free.md`),
  re-decided against in 3D (plans/3D.md sec 2.1). Closed both ways.
- **The multi-level volume stack, the raymarcher, `serializedOn`, the
  volume-box sync.** All 3D-render-only. 2D's own sub-cell pan fix
  (`shaders/render.wgsl:52`, `amr_render.wgsl:158`) is the *source* of 3D's
  "the view's offset is continuous where the solver's is integer" lesson, not
  a destination for it.
- **`?levels=2` makes a balance check vacuous.** 2D has run `?levels=3` in
  the standing suite (`amr-N3-diffuse`, `amr-N3-bounceback`) since before the
  3D fork, so 2D's checker has never been vacuous. Nothing to do.
- **The per-level chi band scaling (`CHI_SCALE = 2^-m`).** 2D already has it
  on the pool path: `amr_step1_pool.wgsl` uses `K_EPS * levelParams.dxL`.
  What 2D does *not* have is the same knob on L0/L1 — that is B7, and it is a
  measurement, not a scaling fix.
- **The six ordered manager passes and the opposite level orders.** 2D
  already coarsens finest-first and refines coarsest-first
  (`main-amr.js:2383-2418`), and does not need a `drain`: 2D's `average` runs
  for every level at the end of every macro-step (`S_Advance`,
  `main-amr.js:2185`) and the manager runs after it, so a tile's fine
  solution has already reached its parent before it can be released. **That
  is nowhere written down** — B0 should assert it rather than leave it as a
  coincidence, and B6 must re-check it, because explode/coalesce is an
  interface-layer transfer and does not restrict a tile's interior.

---

## 3. The verification gap — this is the actual deliverable

2D's AMR invariants live **inside `main-amr.js`** and can only run in a
browser against a live GPU:

- `debugCheck21Balance` (`main-amr.js:3010`) — edge-only, with corner balance
  reported separately and deliberately not gating (`main-amr.js:3061-3090`).
- `debugCheckGeometryCoverage` — geometry-forced refinement.
- Everything else is scored end-to-end: Cd/St against literature
  (`tools/validate-cylinder.js`), structural sweeps
  (`tools/validate-amr-invariants.js`), field diffs against dense
  (`tools/validate-amr-vs-dense.js`).

`make test`'s 2D half is `test-card-params`, `test-card-total`,
`test-dense-to-amr`, `test-f-pack`, `test-field-reconstruct`, `test-sim-rate`.
Only `field-reconstruct` knows anything about the quadtree, and only enough to
walk it for a snapshot — it has no notion of a neighbour, a ring, a parent, or
2:1 balance.

So today a 2D AMR refactor is gated by **a red cell in a run that already has
two standing red cells** (`dense-reference` and `amr-N2-diffuse` at Re=100)
and whose AMR Cd is only reproducible to +/-0.001. That is not a gate you can
refactor an allocator behind.

### B0 — `amr2d.mjs`, the host statement (do this first, alone)

A pure ES module, no GPU, no DOM, mirroring `d3-amr.mjs`:

```
makePool({ dims, rb, maxSlots })     tiling, FB = 2*RB + 2*GHOST, slot count
poolAtLevel(pool, m)                 level-m view  (the 3D one is a one-liner;
                                     check whether 2D's is, see B3)
resolveSource(pool, blockSlot, b, s) where a gather that leaves a tile lands
toGlobalFine / fromGlobalFine        the independent route, for cross-checking
cellAtLevel / finestLevelAt          the "which level owns this point" rule
check21Balance(levelSets, nbAt)      edge AND corner, one function
checkRingParentCoverage(...)         every ring cell's parent tile exists
checkGeometryCoverage(pool, sdf, m)  the body reaches the finest level
cascade21(wantSets, nbAt)            the closure (B2 consumes this)
```

The three rules that make it a test and not a restatement, all taken from
`tools/test-d3-amr.js`:

1. **Check against an INDEPENDENT route, never against the same arithmetic.**
   `resolveSource` is scored against global fine coordinates, where ownership
   is one division with no ring and no offsets. RB=8 and RB=4 must give
   **bit-identical** results, asserted exactly, not to a tolerance.
2. **MUTATION-CHECK every assertion.** An off-by-one in the quadrant offset, a
   flipped quadrant and a transposed `parentOfBlock` must each fail at least
   two checks. A test written alongside the closed form it checks proves
   nothing.
3. **Run the checkers on inputs that VIOLATE the invariant.** A checker only
   ever run on valid input is indistinguishable from one that returns nothing.

**Gate:** `tools/test-amr2d.js` in `make test`, GPU-free, ~20-25 assertions.
Then rewrite `main-amr.js`'s `debugCheck21Balance` to import and call
`check21Balance` rather than reimplement it, and have
`tools/validate-amr-invariants.js` compare the GPU's blockSlot against the
host rule. That is the shape 3D uses: **the host function is the
specification and the test oracle; the GPU checker scores real GPU state
against it.**

Keep `debugCheck21Balance`'s two hard-won properties while moving it: every
level read in ONE submit (the torn-snapshot bug, `main-amr.js:2964-2985`) and
`borderMaxDepth` rather than whole-tile depth (`main-amr.js:2946-2960`).

### B0 — DONE (2026-09-14)

`amr2d.mjs` (pure, `make test`) + `amr2d-gpu.mjs` (the readback) + 34
mutation-checked assertions in `tools/test-amr2d.js`. The wiring replaced all
five pages' private `debugCheck21Balance`/`readAllBlockSlots` -- 680 lines
deleted -- and was gated on real GPU state: the corner-violation sequence over
8192 steps of live refinement on `index-amr.html?levels=3` is identical before
and after, and reproducible on a same-build repeat first.

**Three things the work changed about the rest of this plan.**

1. **The suite had two holes and only a mutation sweep found them.** Eight
   mutants of `amr2d.mjs`; two broke NOTHING on the first pass -- a
   non-periodic block wrap, and a branch-and-bound that accepts on its own
   bound instead of subdividing. Both were the same shape: every fixture sat
   in the middle of the grid, and every geometry case had a body large enough
   that subdivision never mattered, so the mutants came out merely
   CONSERVATIVE rather than wrong. **Score every new checker here by mutation,
   not by a green run**, and put a fixture at the seam and one with a small
   body.

2. **B4 is bigger than it looked, and it has a live-verified precedent.** The
   kernels' geometry test is a SINGLE CENTRE SAMPLE (`amr_manage.wgsl`'s
   `isNearBody`, `amr_manage_pool.wgsl`'s `isNearBodyAt`: one `get_phi` at the
   block centre against `FORCE_REFINE_MARGIN`). The gap to "does any cell of
   this block come within the margin" is the block circumradius -- 5.66 L0
   cells at RB=8. `main-cylinder-amr.js:224-236` records that gap
   live-verified, with the symptom "L1's own force pass sat at a
   bit-identical fx~-0.19 for 20,000+ steps", and records the fix as
   ENLARGING THE MARGIN (`paramsForChildLevel`'s `childLevel===2` special
   case) rather than fixing the test. So B4 should replace the centre sample
   with the Lipschitz branch and bound `amr2d.mjs` now carries, which makes
   the margin mean what it says and lets that special case go.
   **Until then the checker must keep asking the kernel's question**, which is
   why `checkGeometryCoverage` TAKES the predicate and both are exported.

3. **The five-copy problem is not confined to the shaders.** B3 counts six
   duplicated WGSL kernel pairs; the same rule had five copies in JS, and two
   had already drifted. Measured across the five AMR pages afterwards: 40
   shared functions, 3,319 duplicated lines, 18 already drifted. **Tracked as
   B3a**, which blocks B3 and B4.

### B0b — the interface instrument (prerequisite for B1 and B6)

There is no 2D equivalent of `tools/analyze-d3-interface.js`. Build it, and
build it **before** touching the seam:

- Run a refined box on a **periodic, force-free** case — `index-tgv-amr.html`
  is exactly that — against its own no-interface controls (`?levels=1`, and a
  fully-refined run).
- Report **total mass and total momentum drift**, and field error bucketed by
  signed distance to the seam.
- The asymmetry is the discriminator and it is what found the 3D rescale bug:
  `fneq` has no zeroth or first moment, so a wrong rescale leaves **mass at
  the f32 floor while momentum leaks**. Addressing and flux bugs do not look
  like that.

**Gate:** reports, does not PASS/FAIL. There is no literature value for how
fast two discretizations diverge. It exists so B1 and B6 have a number.

---

## 4. The staged plan

Each stage is independently mergeable and has its own gate. Stages B1/B6/B9
move published numbers; B2/B3/B4/B5 must not.

### B0b — DONE (2026-09-14)

`tools/analyze-amr-interface.js`, plus `readConservedTotals` in
`amr2d-gpu.mjs` and `debugConservedTotals` on `main-tgv-amr.js`. Three rungs
in one invocation -- none / half / all -- reporting mass and momentum drift
against the two no-interface controls. Reports; does not PASS/FAIL.

**It changed the ordering of the rest of this plan, on its first run.**

1. **THE MASS CHANNEL IS THE LATTICE, NOT THE INTERFACE. B9 now blocks B6's
   mass half.** Measured N=128, tau=0.8, 512 steps:

   ```
   rung  refined   d mass    per cell/step   predicted (B9)   ratio
   none       0%   1.561e-1     1.861e-8        1.863e-8      0.999
   half      50%   2.057e-1     2.452e-8        2.286e-8      1.073
   all      100%   2.271e-1     2.707e-8        2.709e-8      0.999
   ```

   B9's weight excess, predicted statically, measured directly. `all` costs
   1.4545x `none` because a refined cell collides TWICE per macro-step at
   `tau_1 = 2*tau_0 - 1/2` rather than once at `tau_0` -- `(2/1.1)/(1/0.8)`;
   measured 1.4557. The prediction therefore holds on BOTH paths to 0.1%,
   which confirms the mechanism, `tauAtLevel` and the substep count at once.
   The seam's own mass contribution is the **7.3% on the `half` rung, and only
   there** -- real, but sitting on a floor 13x its size. **Fix B9 before
   making any interface mass claim.**

2. **The momentum channel is clean and the seam is loud in it.** none
   5.73e-6, `half` **1.157e-3 = 9.6x the no-interface floor**, all 1.20e-4 --
   and growing as `step^0.79` rather than settling, which is what says it is a
   per-step source at the seam rather than a transient from the freeze. So
   **B1 does not have to wait for B9**: the weight excess is isotropic and
   cannot move momentum, and `fneq` has no zeroth or first moment so B1's
   rescale cannot move mass. The two are separable by which channel moves.

3. **A declared `?refine=` ladder is needed after all, and it is B6's.** The
   `half` rung's 128-of-256 blocks is the POOL CAP, not an answer the
   criterion gave: slots are granted in blockID order, the free list dries up
   part-way, and the denied blocks form a band. Reproducible and usable, but
   emergent and jagged rather than declared -- and B6 needs a FLAT seam
   (3D's `?refine=slab`) as the control that separates a correction bug from a
   convex corner. The tool flags a rung sitting at its cap rather than
   reporting the count as geometry.

**One trap worth carrying, because any tool here can hit it.**
`debugStepSync(n)` runs in `STEPS_PER_FRAME = 64` batches and **rounds up**,
so asking for 16 runs 64. A first version normalized by the REQUEST, and a
ladder of deltas 16/16/32/64/128 actually ran 384 steps while being labelled
256 -- the mass channel then read exactly **1.500x** its predicted floor on
every rung, which looks like a real 50% excess and is entirely the ladder.
Normalize by the step count the PAGE reports, never by the request.

**And one pre-existing defect found in passing, not fixed here:**
`index-tgv-amr.html?levels=1` never finishes init -- `#status` sits at
"initializing...". `resetSim` dereferences `pools[1]` unconditionally while
the allocation loop runs zero times at `N_LEVELS = 1`. It fails silently
rather than with `error:`, so `validate-all.js`'s boot smoke would not catch
it even if a levels=1 config existed. Belongs with B4's "refuse rather than
degrade" work.

### B1 — the post-collision Dupuis-Chopard factor

**The defect.** 2D's step kernels are fused pull-stream + collide
(`amr_step.wgsl`'s own title), so every buffer holds `f` **after** collision.
The transfers use the textbook **pre-collision** factor:

- `shaders/common_interp.wgsl:43` — `0.5 * tauFine / tauCoarse`
- `shaders/amr_average_f2c.wgsl:117` — `2.0 * state.tau / tau_fine`
- `shaders/amr_average_pool_parent.wgsl:116` — `2.0 * tau_coarse / tau_fine`

The correct post-collision factor is `((tau_f - 1)/(tau_c - 1)) / 2`
coarse->fine, and its exact inverse fine->coarse
(`shaders/common_d3_pool.wgsl` carries the derivation). At tau_c = 0.8 the two
differ in magnitude **and sign** (-0.25 against +0.6875). This was the large
half of 3D's M3 seam error.

**Why it was not fixed with the 3D one:** it moves the entire 2D benchmark
surface at once and `main` is the published site.

**Do it as:** `?dcpre=1` restoring the old factor, exactly as 3D did, so both
paths live in one build and the defect can be re-measured rather than
reconstructed from a branch. Default 0 (correct).

**Refuse tau = 1.** It is a real singularity, not a formula artifact: at
omega = 1 the post-collision populations *are* the equilibrium and carry no
stress, so no post-collision transfer can recover it. `tauAtLevel()`
(`main-amr.js:528`) must be checked at every level at init and refused
loudly, not divided by ~0 quietly.

**Gate:** B0b's momentum drift falls; then the full
`node tools/validate-all.js` sweep re-baselined, with the AMR Cd
reproducibility caveat honoured (same-build repeats, not a comparison against
a number recorded in another session — dense stays bit-identical and is the
strict check).

### B1 — DONE (2026-09-15). The seam's momentum drift falls 27x.

Two commits, staged the way B2-2b/c were: **B1-1** lands both factors behind
`?dcpre=` with the default still on the legacy one (inert, and proved so),
**B1-2** flips the default. A revert of the default does not take the
mechanism, and the re-baselining change does not ride along with an inert one.

**TWO SITES, ONE PER DIRECTION** — `shaders/common_interp.wgsl` and
`common_average.wgsl`. This section was written against THREE files; B3-2
collapsed the average pair into one body, so B1 turned out to be a two-line
change where it was scoped as a sweep. That is B3 paying for itself in a stage
that was planned before it.

**THE MEASUREMENT, same build, one flag** (`tools/analyze-amr-interface.js`,
N=128, tau=0.8, 512 steps):

```
                    ?dcpre=1 (legacy)      ?dcpre=0 (correct)
  none  (no seam)   3.598e-5               3.598e-5   bit-identical
  half  (real seam) 3.562e-1 = 3085x floor 1.302e-2 = 23.3x floor
```

**27.4x.** And `none` coming back BIT-IDENTICAL is what makes it evidence
rather than a number: no interface, nothing to rescale, so the instrument is
reading the seam and not the solver.

**Three things worth carrying.**

1. **THE ANALYTIC AMR GATES CANNOT SEE AN INTERFACE CHANGE, BECAUSE THEY DO
   NOT REFINE.** `channel-poiseuille-amr-N2`, `channel-couette-amr-N2`,
   `tgv-amr-N2` and `tgv-amr-N3` came back BIT-IDENTICAL across the flip
   (9.0634e-4 / 8.0601e-4 / 3.0470e-4 / 1.3453e-4, and maxL2rel 1.3208e-3 on
   both TGV rungs). Measured directly with `debugListActiveBlocks` after 2048
   steps: **every one of them has ZERO active tiles at every level.** The
   pages say so themselves and mean it — `main-channel-amr.js` defaults
   `autoRefine` OFF (its header explains the Couette wall-seam trigger that
   forced it), and `main-tgv-amr.js` keeps thresholds that deliberately never
   fire at TGV's own vorticity scale. So `tgv-amr-N2 == tgv-amr-N3` exactly
   because both are the same dense run.

   This is not a defect in those pages; it is a defect in what the SUITE is
   read to mean. **CLAUDE.md's "for any change to precision or storage layout
   the analytic field checks are the gate" does NOT extend to a coarse/fine
   COUPLING change** — there, the `*-amr-*` analytic configs are structurally
   blind, and B0b's conservation channel is the only instrument in the tree
   that is not. B6 needs to know this before it starts: its gate cannot be
   these configs either.

   **And it retro-invalidates one reading.** B2-2c cited
   `channel-poiseuille-amr-N2` at "9.06e-4 against a 5.0e-3 tolerance" as the
   analytic gate that stayed clean while Cd/St moved. That number is exactly
   what a zero-tile dense run reads and has not changed through B1 either, so
   it never had the power to say anything about a refinement-region change.
   The CONCLUSION there still stands on the corner-balance and closure
   evidence; the analytic corroboration does not.

2. **NEAR tau = 0.5 THE TWO FACTORS AGREE TO ~2%, AND THAT IS WHY THIS
   SHIPPED.** The difference is +0.6875 against -0.25 at tau = 0.8 — magnitude
   and sign — but the card page runs tau ~ 0.5044 and the cylinder ~ 0.5256,
   where they differ by a couple of percent. So the pages carrying a number
   could barely see it, and the pages whose tau would have made it obvious
   were the ones that do not refine. A defect can be simultaneously enormous
   and invisible if the configurations that would show it are the ones nobody
   scores.

3. **MASS MOVED, THOUGH `fneq` HAS NO ZEROTH MOMENT.** B0b predicted this
   rescale could not touch mass, and on the seam rung it did: -2.256e-8 per
   cell/step (ratio -1.974 against the lattice-weight floor, WRONG-SIGNED)
   under the legacy factor, 7.546e-9 (0.660) under the correct one. The
   prediction is right about the direct route and incomplete: the seam has no
   flux correction, so a corrupted stress changes the flow, and a different
   flow leaks a different amount of mass through it. The channel separation
   ("mass drifts -> weights, momentum drifts -> coupling") is a statement
   about FIRST-ORDER routes, not an exclusion.

   Related, and a caveat on the tool: the `all` rung is NOT a clean
   no-interface control for this change. Its total is read off the dense L0
   grid, whose content under full refinement IS the restriction this factor
   scales — and its refined fraction itself moved (100% -> 98%), because the
   criterion reads a field the factor changes. `none` is the control here.

**TAU = 1 IS REFUSED.** At omega = 1 the post-collision populations ARE the
equilibrium, so no post-collision transfer can recover the stress, and the
closed form says so as 0/0. Level m is singular at `tau_0 = 1/2 + 2^-(m+1)`,
which puts `?tau=1` and `?tau=0.75` on channel/TGV squarely on it and
`tau_0 = 0.5625` — inside `index-amr.html`'s OWN slider range — on level 3.
`amr2d.mjs`'s `tauChainSingularity` is the rule; the guard sits inside
`updateLevelParams()`, the one place a per-level tau reaches the GPU, so init
and every live TAU/RE control are covered by ONE check. Three call sites were
reordered so it runs BEFORE the L0 tau write — otherwise a refused change
still leaves L0 on the new tau with every finer level on the old one, which is
the silent degradation the guard exists to prevent.

**And the refusal is gated BOTH WAYS**, which B4-4's rule asks for and which
one config cannot do: `refuse-tau-unity` (`?levels=2&tau=0.75`) requires it to
fire, and `tau-unity-ok-under-dcpre` (the same URL plus `?dcpre=1`) requires
the page to BOOT. The singularity belongs to the post-collision form alone, so
without the negative half a guard that simply banned tau near 1 outright would
look identical. Note also that the value the USER types is not the value that
is singular — L0 at 0.75 is ordinary, level 1 is the problem — so a guard that
checked only L0 would pass the gating config.

**THE HOST STATEMENT, and the route that makes it evidence.** `amr2d.mjs`
carries both pairs and `tools/test-amr2d.js` scores them against an
INDEPENDENT ROUTE — compose BGK post-collision, the textbook pre-collision
relation, and BGK again — rather than restating the algebra alongside them,
which would have proved nothing about a closed form that had already survived
for years. Five plausible slips are mutation-checked, including the historical
one. Both WGSL forms are lifted out of the shader source and evaluated against
the host's, because a host module that is right while the WGSL is wrong buys
nothing; a failed extraction FAILS rather than skips. Both new gates were made
to fail before being trusted.

**THE RE-BASELINE.** Two full sweeps on the flipped default, reproducing each
other EXACTLY on all five configs — the same-build repeat this plan's Risk #1
demands, so these moves are real and not the free list:

```
                      legacy            post-collision     move
  dense-reference     1.951 / 0.1260    1.951 / 0.1260     none (no transfer)
  amr-N2-diffuse      1.642 / 0.1477    1.631 / 0.1466    -0.011 / -0.0011
  amr-N2-bounceback   1.321 / 0.1617    1.343 / 0.1624    +0.022 / +0.0007
  amr-N3-diffuse      1.425 / 0.1551    1.463 / 0.1565    +0.038 / +0.0014
  amr-N3-bounceback   1.351 / 0.1616    1.348 / 0.1627    -0.003 / +0.0011
```

Verdicts unchanged: the same two standing red cells and no others, all seven
invariant gates OK at every checkpoint, every boot smoke and every refusal
config PASS. **St rose on all four AMR configs**, i.e. toward the literature
0.165 in every case, while Cd moved both ways — consistent with a stress
coupling that was wrong at the seam rather than a systematic bias. Do not read
more into the Cd column than that: these are still time-averaged surface
integrals, and finding #1 above is the reminder that the suite's analytic half
was not watching.

**`dense-reference` bit-identical across the flip** is the strict check doing
its job — that page has no coarse/fine transfer at all, so a move there would
have meant the change had escaped its blast radius.

**One process note.** A mid-sweep `refuse-pool-exhausted` FAIL ("the page kept
running (status "initializing...")") did not reproduce in isolation or on the
repeat sweep, and is recorded here as a flake rather than quietly dropped.
Separately, `pgrep -f "validate-all.js …"` matches ITS OWN shell's command
line, so a waiter built on it never exits and reports a finished sweep as
still running — use the log's mtime or a pid captured at launch.

### B2-1 — the before-number — DONE (2026-09-14)

`debugCheckRefinementClosure` on all five AMR pages runs `cascade21` against
the live present set; reported at every checkpoint of the standing sweep,
never gated. **Two measurements, and the second exists because of what the
first cannot do.**

1. **Every closure violation on the shipped pages is the DIAGONAL.**
   `{parentOf: 0, siblingOf: 0, edge: 0, diagonal: 14}` on
   `index-amr.html?levels=3`, and the same shape on the cylinder. So of the
   four defects this section lists, the three that were fixed really are
   fixed; what is left in the TOPOLOGY is exactly the corner half
   `debugCheck21Balance` already reports and deliberately does not gate.
   Closing a set and walking shared edges are independent computations and
   they agree, with `forced <= cornerViolations` always — two violating PAIRS
   can demand the same parent BLOCK, which is why the counts differ by one or
   two rather than never.

2. **THE CLOSURE CHECK STRUCTURALLY CANNOT SEE THE VETO DEFECT**, and this is
   the part worth carrying. Closing the PRESENT set only finds blocks implied
   by what exists. The fourth bug is about wants that were never granted, and
   a block that was never created cannot be observed as missing from a closure
   over what is. **That is the argument for putting the closure on the WANT
   set stated as a measurement rather than a design preference** — and it
   means the natural instrument covers one half only.

   The other half, via the knob that already exists (`?demandCascade=1`),
   `index-amr.html?levels=3` at ~8300 steps:

```
leg                 L1 tiles   L2 tiles   L2 bx range   L2 x-extent (L0)
default (veto only)    189        352       [22,41]          80
?demandCascade=1       259        468       [16,59]         176
```

   **Level 2's extent more than doubles.** L1 barely moves (184 -> 200 L0
   units) while L2 goes 80 -> 176, so the constraint is specifically on the
   deeper level — the L1/L2 boundary pinned a few cells off the body, exactly
   as described. That is B2-2's target, now a number.

**What this changes about B2-2.** The rewrite's gate is no longer "expect Cd/St
to move and report it": the two measurements above are the gate, and they are
directional. `closure -> 0` (the ring half) and `L2 extent -> ~176 without
?demandCascade` (the veto half). Cd/St moving is a consequence to be reported,
not the thing being checked — which matters because the AMR configs' own
reproducibility floor (+/-0.002 at N=3, CLAUDE.md) is wide enough to hide a
real regression underneath a change this size.

### B2-2a — the GPU closure, proved before it is used — DONE (2026-09-14)

`shaders/amr_cascade.wgsl`: `completeQuads` + `balance`, no includes at all
(the rule is about the TREE, so it needs no SDF, lattice or populations).
Want buffers in `allocLevelPool`; pipelines built; `dispatchMacroStep`
untouched, so the physics is inert. `cascadeRoundTrip` seeds both
implementations and requires an EXACT match.

**One sweep, deepest first — the fixed-point loop is not needed and that is
now checked**, via three already-closed seeds, rather than argued.

**Two things worth carrying.**

1. **The seeds are the work, not the driver.** All 10 agree on all four AMR
   pages, but agreement on valid input is worth nothing on its own, so the
   battery is built from failure modes: quad-partial (odd in both axes),
   seam (0,0), seam-corner (both wraps at once), and `dense-band` — a full
   finest-level row, which is the only seed dense enough that two threads
   write the same parent slot in one dispatch, i.e. the only one that could
   expose a non-monotone write. Mutation-scored: 7 mutants of the shader,
   5–9 seed failures each.

2. **`d = (0,0)` IS REDUNDANT, measured.** It is the one mutant that breaks
   nothing — 0 of 10 GPU seeds, and 0 of 42 host checks with the same mutant
   applied to `cascade21`. A quad's members are adjacent and share a parent,
   so for any block at least one x-face neighbour has the same parent
   (`bx` even → `(bx+1)>>1 == bx>>1`; odd → `(bx-1)>>1`; NBX is a power of
   two so the wrap has no edge case). **So this plan's own "d = (0,0) is the
   tree property, the edges are 2:1 balance, the diagonals are the ring"
   overstates it: the tree property is implied by the edges.** Keep the
   offset — it costs a ninth of a cheap pass, it makes the rule self-evident,
   and the redundancy depends on the faces being present — but a green suite
   would support deleting it, which is why it is written into both files.

**Left for B2-2b:** the `decide` entry points (criterion + geometry → want,
with the balance vetoes removed), switching coarsen/refine to consume the want
set, and deleting the per-pass tests, `FIXED_POINT_ITERS`, and
`?demandCascade`. Gate is B2-1's two directional numbers: `closure -> 0` and
L2's extent `80 -> ~176` L0 units without the flag.

### B2-2b — BLOCKED ON THE BINDING CEILING, and that decides its shape

**Attempted and reverted, 2026-09-14.** The intent was the house pattern every
other stage here used: land the switch-over behind `?cascade=1` with the
default byte-identical, measure the A/B in one build, flip the default in a
later commit. **That is impossible for the pool manager**, and the reason is a
hard limit rather than a design preference:

```
error: The number of storage buffers (17) in the Compute stage
       exceeds the maximum per-stage limit (16).
       While validating [BindGroupLayoutDescriptor "managePoolBGL"]
```

`shaders/amr_manage_pool.wgsl` already declares **16 storage buffers — exactly
the per-stage maximum**, which every AMR page already knows it is sitting on
(`NEEDED_STORAGE_BUFFERS_PER_STAGE = 16`, checked against the adapter at init).
Adding the child level's want array is a 17th, and no grouping avoids it: the
limit is per STAGE, not per bind group. Caught on the first boot-smoke run,
before any of it could be believed.

**So the switch-over cannot be staged behind a flag — it has to be the
replacement, in one commit.** That is not a loss: two of the pool manager's
bindings become dead under the closure, and they are exactly the two the
legacy tests read — `parentBlockSlot` (the neighbour-active veto) and
`grandchildBlockSlot` (the grandchild guard). Delete those and add `childWant`
and the count goes 16 → 15, with room. But the deletions and the switch are
one atomic change, and the A/B has to be done ACROSS commits (against B2-1's
recorded numbers) rather than within one build.

**Two ways forward, and the first is probably worth doing on its own.**

1. **B2-2b0: get the pool manager off the ceiling first — DONE
   (2026-09-14).** 16 → 15. `childQuadrant` held `slot % 4`: both allocators
   compose a slot as `quadIdx*4 + quadrant`, so a whole binding was spent on a
   constant. The buffer stays (five other shaders read it) and
   `allocLevelPool` writes it once instead of refine() rewriting it per
   allocation.

   **The measurement is the transferable part.** The claim being relied on was
   not "the arithmetic is right" — that is two lines — but "nothing on any
   path has ever put a different value there", including `debugSnapshotLoad`,
   which writes whatever a snapshot recorded. So `quadrantOfSlot` went into
   `amr2d.mjs` as the rule, `checkSlotQuadrantsOnGPU` into `amr2d-gpu.mjs`,
   and the rule was scored against live state on the UNCHANGED build first:
   holds over 352 active slots at `?levels=3` and 212 at `?levels=4` after
   8000+ steps of real refinement. **A derivable-looking field is a claim
   about every writer, not about the formula.**

   Binding 8 is left as a HOLE rather than renumbered — a renumber is five
   pages' bind groups in lockstep, the 238e48c shape.

2. **B2-2b as one replacement**, accepting that the default moves in the same
   commit that adds the closure. Gated by B2-1's directional numbers rather
   than by a same-build A/B: `closure -> 0`, L2's x-extent `80 -> ~176` L0
   units with no flag set, plus the AMR Cd/St reported as a change.

**What was already written and works, for whoever picks this up:** the
`decide` entry points for both managers (the criterion/geometry half of
coarsen-refine with the neighbour tests removed), the `select`-based rewiring
of coarsen/refine onto the want array, and the one-sweep dispatch
(decide → `encodeCascade` → coarsen finest-first → refine coarsest-first).
All of it passed `make check` and `naga`; only the pool manager's bind group
could not be created. The dense manager (`amr_manage.wgsl`, 11 buffers) has
ample room and its half of the change is unaffected by any of this.

### B2-2b / B2-2c — DONE (2026-09-14). The manager acts on the closed set.

`decide` -> close -> coarsen -> refine, once, with `?cascade=0` kept as a
control. Default flipped in its own commit so a revert of the default does not
take the mechanism.

```
index-amr.html?levels=3       ?cascade=0   default
corner 2:1 violations             13          0
closure missing                   13          0
L1 / L2 tiles                 182 / 340   243 / 420
L2 x-extent (L0 units)            80         160
```

**Corner balance is clean for the first time.** `check21BalanceOnGPU` has
reported and not gated it since B0 — the ring path tolerates a missing
diagonal parent, so failing every run on it would have been wrong. Under the
closure it is zero at every checkpoint, because the diagonals are four of the
nine offsets the rule was always written with and the per-pass implementation
only ever covered the faces. **A check that had to be non-gating because the
code could not satisfy it can now become a gate.**

**Three things worth carrying.**

1. **The veto was STRONGER than the rule, and that was the bug.** The
   neighbour-active gate demanded the PARENT's four same-level neighbours be
   active; 2:1 balance only demands the parents of the block's OWN
   neighbours. The extra strength is not a safety margin, it is the deadlock
   — a refine blocked by a neighbour that would only ever have been created BY
   that refine. **A conservative-looking local test is worth checking against
   the rule it approximates**; this one cost level 2 half its reach for as
   long as it shipped.

2. **The right instrument is not Cd/St, and this is a clean example.** The
   change moves St on the two bounce-back configs ~0.004 AWAY from the
   literature centre (0.1654 → 0.1616) while making the topology correct. No
   verdict changes, and `channel-poiseuille-amr-N2` — the analytic gate with a
   closed-form answer — reads 9.06e-4 against a 5.0e-3 tolerance. CLAUDE.md
   already says Cd/St averages far-field structure away; a refinement-region
   change is squarely in that class, so **report the Cd/St move and gate on
   the analytic one.**

3. **The staging only worked because B2-2b0 came first.** The flagged
   switch-over was impossible at 16 bindings and trivial at 15. Two attempts:
   the first hit `CreateBindGroupLayout` and was reverted whole; the second
   was the same code with the want array in a recovered slot. **When a change
   is blocked by a hard limit, the unblocking step is its own stage** — it is
   cheaper, it is bit-identical, and it converts a leap back into a step.

### B2-2d — DONE (2026-09-14). The per-pass path is gone.

-766/+294. `amr_manage.wgsl` 11 -> 9 bindings, `amr_manage_pool.wgsl` 16 -> 14,
and with them `hasLevel2Child`, `level2Wanted`, `edgeNeighbors`,
`hasGrandchild`, `childEdgeNeighbors`, the 8-cell grandchild ring walk, the
`isHardRequired` split, the neighbour-active veto, `FIXED_POINT_ITERS`,
`?demandCascade`, `?refineIters`, `HAS_LEVEL2`, `HAS_GRANDCHILD`,
`dummyCriterionBuf`. Inert against B2-2c: `amr-N3-diffuse` 1.425 / 0.1551 is
exact, the bounce-back pair within 0.001.

**CORNER 2:1 BALANCE IS NOW A GATE** (`requireCornerBalance` defaults true,
and `validate-all`'s `?ghostfree=1` special case is gone -- that path alone
used to need it, now everything does). It was reported-and-not-gated from B0
until now *for a good reason*: the ring path tolerates a missing diagonal
parent, the per-pass cascade covered only the faces, so the count was reliably
13 on the shipped page and gating it would have failed every run for a defect
already written down. **A check that had to be non-gating because the code
could not satisfy it becomes a gate the moment the code can -- and leaving it
reporting-only after that is how a fixed invariant silently regresses.** Worth
scanning the rest of this project's checkers for the same shape.

**And a process note that nearly cost the tree.** The first attempt at
collapsing the dispatch brace-matched across template literals and truncated
`main-amr.js` from 3833 lines to 152 -- and **`make js` PASSED**, because a
truncated file is still valid JavaScript. It was caught by reading
`git diff --stat` (11,544 deletions), not by any check. That is B3a's finding
#1 for the third time, now with a positive rule attached: **for scripted
surgery in these files, the diff stat is part of the gate**, and a diff an
order of magnitude larger than the change described is a failure regardless of
what `node --check` says.

**B3's `manage` pair is now a refactor rather than a rewrite**, which is what
B3 sequenced it last for: the two files no longer contain two different
implementations of a balance rule, because they no longer contain one at all.

### The checker sweep B2-2d called for — DONE (2026-09-14)

Scanning the rest of the project's checks for the same shape found three, and
they were three different failure modes — worth separating, because only one
of them is the shape B2-2d described:

1. **Reporting-only because the code could not comply.** The closure check
   (`checkRefinementClosureOnGPU`), deliberately non-gating in B2-1. Now
   gated. **Its meaning changed with the promotion**, which is worth noticing:
   it used to ask "is the rule implemented correctly" — true by construction
   now — and it now asks "could the ALLOCATOR deliver what the rule demanded",
   a resource question.

2. **An ALWAYS-TRUE GATE, the fourth in this project** (B3a-4 found two, B4-3
   one). `converged` read `v[4] === 0 && v[5] === 0` and B2-2d stopped writing
   `v[4]` entirely, so the first clause was permanently satisfied; its whole
   rationale described the deleted fixed-point loop, down to advising a flag
   (`?refineIters=`) that B2-2d also deleted. **Deleting a mechanism silently
   promotes every check that referenced it to vacuous** — that is now three
   instances, and it is worth grepping the checks for a mechanism's name
   whenever one is removed. What remained was the pool-starvation half, so it
   is named for that now.

3. **A check that existed and NEVER RAN** — worse than reporting-only, because
   it looks like coverage. `checkSlotQuadrantsOnGPU` was added in B2-2b0 to
   justify removing a binding and then only ever driven by a one-off probe.
   Now gated at every checkpoint.

The sweep is seven gates, all gating: `2:1-balance, corner, coverage, field,
pool, closure, quadrants`. **Eight since B3-1's prerequisite** -- `origins`,
the same shape as `quadrants` and added for the same reason (a per-slot buffer
whose content is a function of cheaper data, scored before that is relied on).

**And they were checked for being vacuous**, which is the point of the whole
exercise. Under `?maxFineBlocks=16`: six fire (2:1-balance 28, corner 55,
coverage 28, pool STARVED 2235, closure 19) and **`quadrants` correctly does
not**, because the quadrant rule is independent of starvation. Six firing and
one not is the discrimination that says they are not all reading one cause —
a sweep where everything goes red together has not been shown to test seven
things.

### B2 — 2:1 balance as one closure

**What 2D has.** The rule is spread across `amr_manage.wgsl` and
`amr_manage_pool.wgsl` as per-pass tests, wrapped in a fixed-point loop
(`main-amr.js:2383-2418`) whose iteration count is `N_LEVELS - 1`. The pool
manager's header (`shaders/amr_manage_pool.wgsl:26-90`) documents **three
separately live-verified bugs** produced by that shape:

1. the neighbour-active gate vetoing geometry-forced refinement;
2. a grandchild (level m+2) blind spot that produced real depth-1-next-to-
   depth-3 violations at `N_LEVELS=4`;
3. a criterion-based ("wants") cascade test that had to become an existence
   ("has") test, because criterion is re-evaluated every round and flickers.

And a fourth, still open and shipped: **only the veto half of the refine
cascade was ever implemented** (`main-amr.js:112-131`). A criterion-driven
refine can be vetoed forever by a neighbour that would only ever have been
created *by* that refine. Measured live on `index-amr.html`: level 2 never
extends past the geometry halo into the wake, which pins the L1/L2 boundary a
few cells off the body **so every shed vortex crosses it right there**. The
fix exists as `?demandCascade=1`, default off and never validated.

**What 3D has.** One host function, `d3-amr.mjs:464`, ~50 lines:

```
present(m, b)  =>  present(m-1, parent(n))   for each neighbour n of b, and b itself
```

Read forwards it is "refine forced by a deeper neighbour"; read backwards it
is "coarsen blocked by one". They are not two mechanisms, which is precisely
why it belongs *between* decide and coarsen/refine rather than as tests inside
them. It runs on the **WANT** set, so by the time coarsen and refine execute
the answer is already balanced and neither needs to know anything about
levels. The unbounded fixed-point loop disappears; so does the grandchild
special case (the closure is transitive by construction); so does the
existence-vs-desire distinction (there is only one set).

**Two details 2D must copy exactly:**

- **All 9 offsets in 2D, not the 4 faces.** `d = (0,0)` is the tree property
  (a block with no parent is not a refinement of anything), the faces are 2:1
  balance, and the **four diagonals are the RING**. Face-only closure leaves a
  corner ring cell's parent tile unallocated, and `amr_step1.wgsl` reads
  exactly that cell whenever the diagonal same-level neighbour is absent.
  This is also what finally makes `debugCheck21Balance`'s corner list a gate
  instead of a report (`main-amr.js:3061`).
- **Refinement is QUAD-COMPLETE from level 2 down.** `amr_manage_pool.wgsl`
  already says so ("a parent slot may only spawn a level-(m+1) quad"), and
  `hasChild` leans on it. A want for one child is a want for the parent's
  refinement, so the input is **completed to quads**, not rejected. 3D's first
  cascade got this wrong and the checker was right.

**Do it as:** host `cascade21` in `amr2d.mjs` first, gated by `make test`
against `check21Balance` — including **minimality** (removing any forced quad
leaves a tree that violates 2:1 or orphans a subtree), without which "refine a
halo to be safe" passes everything. Then mirror it into WGSL as
`completeQuads` + `balance` entry points on the want array, delete the
per-pass tests, delete the fixed-point loop, delete `?demandCascade` (it
becomes unconditional and correct).

**Gate:** `make test`; then `validate-all --configs=amr-N3-diffuse,
amr-N3-bounceback,amr-dev-invariants` with corner balance **required**; then
the wake defect measured as closed (level 2 extends past the geometry halo —
`debugListActiveBlocks` / `measure-refinement.js`). Expect Cd/St to move on
the AMR configs, because the refined region genuinely changes; that is the
point, and it must be reported as a change rather than absorbed.

### B3 — one kernel per stage, one accessor per parent kind

**What 2D has.** Six near-duplicate pairs, dense-parent and pool-parent:

| level-1 file | level>=2 file | lines |
|---|---|---|
| `amr_step1.wgsl` | `amr_step1_pool.wgsl` | 349 / 359 |
| `amr_interp_dense_parent.wgsl` | `amr_interp_pool_parent.wgsl` | 308 / 269 |
| `amr_average_f2c.wgsl` | `amr_average_pool_parent.wgsl` | 142 / 147 |
| `amr_force1.wgsl` | `amr_force1_pool.wgsl` | 245 / 236 |
| `amr_criterion.wgsl` | `amr_criterion_pool.wgsl` | 63 / 83 |
| `amr_manage.wgsl` | `amr_manage_pool.wgsl` | 364 / 492 |

`common_interp.wgsl` already shares the *middle* of the interp pair and its
header says exactly why ("they differ in exactly one thing: how they FETCH a
parent cell"). 3D took that observation all the way.

**The 3D shape.** `shaders/common_d3_parent_{dense,pool}.wgsl` declare two
functions and nothing else:

```wgsl
fn parentIndex(v) -> i32     // -1 when the parent tile is absent
fn parentPresent(v) -> bool  // constant-true in the dense case; it folds away
```

Everything else is level-generic *once `W`/`H` are read as "the PARENT grid's
dimensions" rather than "L0's"* — then `nbx() = W/RB` is that level's block
count, `fineDim() = 2*W` is its child-cell count, and every block/cell mapping
follows. The entry file composes: `d3_amr_interp_q19.wgsl` is seven lines of
`@include`. `d3_amr_step1_q19.wgsl` needs **no parent fragment at all** — the
fine step never reads a parent, so there is exactly one step kernel for every
level.

**The 2D diff is already this small.** Code-only, `amr_step1.wgsl` vs
`amr_step1_pool.wgsl` differ in: origin (derived from blockID vs read from
cached `originX/Y`), `dxL` (hardcoded 0.5/0.25 vs `levelParams.dxL`), tau
(`state.tau` vs `levelParams.parentTau`), and `K_EPS` (const vs override).
All four are expressible in the per-level uniform the pool version **already
has**. Give level 1 an `originX/originY` pair at allocation and
`amr_step1.wgsl` deletes outright.

**Do it as, in this order, one pair per commit:**

1. `step1` — delete `amr_step1.wgsl`, level 1 runs `amr_step1_pool.wgsl`.
   This is the hot kernel and the largest single win. **DONE — see B3-1
   below; it went the other way round in the end (the pool file survived and
   took the name), and the blocker was a wrong claim in its own header.**
2. `average` — one body, two parent fragments. **DONE — see B3-2 below.
   The accessor needed a third function (`parentStoreWord`) that 3D's
   `parentIndex`/`parentPresent` pair does not have, because the two parents
   are different BUFFERS and WGSL cannot pass one to a function.**
3. `interp` — `common_interp.wgsl` already holds the shared half; move the
   fetch behind `parentIndex` and collapse the two entry files. **DONE — see
   B3-3 below. The fetch could not move behind an index alone: the origin has
   to move with it, because the two halves work in different FRAMES.**
4. `force1` — same, and it shrinks again under B4. **DONE — see B3-4 below.
   It collapsed to ONE kernel like step1, not to a kernel-plus-accessor like
   average/interp, and it leaves `originX`/`originY` with no reader.**
5. `criterion` — smallest, do it for uniformity. **DONE, and the answer was
   NO — see B3-6 below. These two do not differ in a parent accessor, they
   differ in the dispatch mapping, and ~12 of 146 lines overlap. The stencil
   and the reduction are shared; the kernels stay two.**
6. `manage` — **last**, and only after B2 has already deleted the per-pass
   balance tests. Unifying 364+492 lines of two different balance
   implementations is not a refactor, it is a rewrite; B2 makes it a refactor.
   **DONE — see B3-7 below. B2 had made it small: the predicates unified, the
   per-block vs per-quad ALLOCATORS did not and should not, and the job turned
   up dead code B2 itself had left behind.**

**Gate:** bit-identical. Each of these is a pure code motion and must be
proved so — `tools/amr-diff.js` against a pre-change snapshot, and
`validate-all --configs=amr-N2-diffuse,amr-N3-diffuse` reproducing the
*dense-style* exactness where available. Where an AMR config's Cd moves in the
4th digit, that is the `atomicSub` free-list / truncated-atomic
non-determinism (CLAUDE.md), **not** evidence — take a same-build repeat
before believing either direction.

**The asymmetry that stays, and should:** L0 remains a dense, ghost-free,
`cellIndex()`-addressed grid (plans/AMR-multilevel.md decision 1). 3D kept it
too. It is genuinely cheaper per cell on what is usually the majority of live
cells, and the *point* of the accessor split is that this asymmetry then costs
two functions instead of six files. Do not "unify" L0 into a pool.

### B3-1 — DONE (2026-09-14). One fine step, every level.

`amr_step1.wgsl` (349 lines, level 1 only) and `amr_step1_pool.wgsl` (359,
level>=2) are one file, `amr_step1.wgsl`, dispatched for every pool level. Two
commits, both proved inert.

**The pair's whole reason for existing was an argument, and the argument was
wrong.** The level-1 file derived its tile's physical origin as `bx * RB`; the
pool file read a per-slot `originX/originY` buffer, because (its own header)
a level>=2 tile's origin needs a walk up the parent chain. It does not: every
level's block grid is globally anchored and quadtree-uniform, so the origin
has a closed form -- `block * RB * 2^-(m-1)`, i.e. `f32(bx*RB) * 2 *
levelParams.dxL`. amr2d.mjs has held both routes and scored them against each
other since B0 (`tileOriginL0` vs `tileOriginL0Recursive`). **The refactor was
blocked on a claim the project had already disproved and not noticed.**

Deriving it (B3-1a) deleted two bindings from every page's step1 layout AND
removed the one structural difference between the files. What was left was
four differences, each already a field of the per-level uniform the pool file
had: `dxL` (0.5 at level 1, the old literal), `parentTau` (L0's own tau at
level 1), `nbx/nby`, and `K_EPS` -- whose `K_EPS * dxL` at level 1 is exactly
the `K_EPS * 0.5` the level-1 file computed. So B3-1b is `for (let c = 2;` ->
`c = 1` on the levelParams loops, one bind-group pair moved onto `pools[1]`,
and `substep()` losing its `if (isL1)`.

**PROVING IT INERT NEEDED A NEW INSTRUMENT, AND THAT IS THE REUSABLE PART.**
The stated gate was "bit-identical, `tools/amr-diff.js` against a pre-change
snapshot". That gate is available here -- a PINNED cylinder does not feed force
back into the flow, so unlike the falling card its field is reproducible where
its Cd is not -- but it is a whole-run comparison, and a whole-run comparison
can only ever say "something moved". The better gate was cheaper and came
first: `debugCheckTileOrigins` (the eighth invariant gate) scores the LIVE
origin buffer against the same closed form at every checkpoint, so "the value
the kernel now computes is the value it used to load" is measured per slot,
exactly, before anything is removed on the strength of it. Mutation-tested by
transposing x/y -- the defect class that cost a wrong level-2 force -- which
turns it red with the other seven green. **Same move as B2-2b0's
`checkSlotQuadrants`: when a refactor rests on "this stored value is
redundant", score the stored value first; the gate is then per-slot and exact
instead of whole-field and statistical.**

Measured, `index-cylinder-amr.html?levels=3`, 4096 steps from reset:

    ?levels=3   pre x2, B3-1a x3, B3-1b x1   all bit-IDENTICAL
                (maxAbsDiff 0 on ux/uy/rho/omega)

**And the gate turned out to be config-dependent, which is the thing to carry
forward.** One `?levels=3` run in seven came back different (ux relL2 2.6e-5)
and has not recurred. At `?levels=2` NO two runs of one build agree -- four
runs, four different fields, at the SAME magnitude as the pre-vs-post
difference there. So `amr-diff` cannot gate a change at N=2 at all, and the
N=2 pre/post DIFFERS is not evidence of anything; N=3 is where the comparison
has power, and it reads IDENTICAL across both commits. Note this INVERTS the
Cd picture, where N=3 is the wider spread -- reproducibility does not transfer
between configs in either direction. Written down in CLAUDE.md next to the Cd
figures, because the natural move (measure the cheap config, assume the deep
one is no better) would have picked exactly the wrong one.

**What is left of B3:** `average`, `interp`, `force1`, `criterion`, `manage`.
`force1` gets easier than the table below suggests -- `amr_force1_pool.wgsl` is
now the ONLY reader of `originX/originY`, so unifying it the same way retires
those two buffers and `amr_manage_pool.wgsl`'s write of them, which is the
binding CLAUDE.md's 16-buffer-ceiling note nominates next.

### B3-2 — DONE (2026-09-14). The average pair, as one body and an accessor.

`amr_average_f2c.wgsl` (142) and `amr_average_pool_parent.wgsl` (147) were
near-identical copies. They are now 24 and 32 lines -- their binding layout,
their overrides, and two `@include`s each:

    shaders/common_average.wgsl             the kernel, once
    shaders/common_avg_parent_dense.wgsl    parent = L0's dense grid
    shaders/common_avg_parent_pool.wgsl     parent = another pool tile

**This is the accessor shape B3 describes, and it is worth stating what the
accessor turned out to BE**, since it is what B3-3 and B3-4 will reuse. 3D's
pair declares `parentIndex(v)` / `parentPresent(v)`. That is not quite enough
in WGSL here, because the dense parent and the pool parent are DIFFERENT
BUFFERS with different plane strides, and a buffer cannot be passed to a
function. So the contract is three functions, and the third is the store
itself:

    fn parentTau() -> f32
    fn parentCellForChild(slot, blockID, lcx, lcy) -> u32
    fn parentStoreWord(cell, wi, word)

The dense fragment reads `state` and writes `f_coarse`; the pool one reads
`levelParams` and writes `f_parent_pool`. **The fragments reference bindings
the ENTRY file declares**, which is new for this project -- every existing
`common_*.wgsl` is a pure function of its arguments -- and it is the thing
that lets the entry files hold the binding layout (one place, matching the
page's BGL one-to-one) while the fragment holds the access. Nothing in
shader-loader.mjs needed to change; it is a textual splice and WGSL module
scope is order-independent.

**Unlike B3-1, there is no deeper simplification hiding here.** B3-1 collapsed
its pair because the difference was a wrong claim; this pair's difference is
real -- L0 is a dense, ghost-free, cellIndex()-addressed grid and stays one
(plans/AMR-multilevel.md decision 1). The win is that the asymmetry now costs
three small functions instead of a second copy of the restriction.

**Measured inert**, `index-cylinder-amr.html?levels=3`, 4096 steps from reset:
bit-IDENTICAL to the pre-B3 baseline. Boot smoke PASS on all seven pages;
invariants PASS (all eight gates) on amr-dev and both N2/N3 x
diffuse/bounceback; analytic channel/TGV PASS; Cd/St unchanged apart from the
two standing red cells.

**AND THE FIRST RUN SAID DIFFERS.** It was the `?levels=3` excursion B3-1
measured (1 run in 7), and the repeat came back IDENTICAL -- the first live
use of the rule B3-1 wrote down, on the first opportunity to get it wrong.
Take the repeat.

### B3-3 — DONE (2026-09-14). Interp, and what the accessor is really for.

`amr_interp_dense_parent.wgsl` (308) and `amr_interp_pool_parent.wgsl` (269)
are 39 lines each -- bindings and two `@include`s. 577 lines become 504, and
the ~120-line `main()` they each carried becomes one.

    shaders/common_interp_kernel.wgsl          the kernel, once
    shaders/common_interp_parent_dense.wgsl    parent = L0's dense grid
    shaders/common_interp_parent_pool.wgsl     parent = another pool tile

`common_interp.wgsl` -- the BLEND -- is untouched and keeps its name: it is
shared more widely than this pair (plans/ghost-free.md needs the STEP kernels
to run the same reconstruction inline), which is why the new file is
`common_interp_kernel.wgsl` rather than folded into it. One is the math inside
the kernel; the other is the kernel around it.

**The accessor grew by two functions and, more usefully, by a CONSTRAINT.**

    fn parentTau() -> f32
    fn levelNbx() -> u32
    fn levelNby() -> u32
    fn parentOrigin(slot, bx, by) -> vec2<u32>
    fn sampleParent(slot, ix, iy) -> CoarseSample

`parentOrigin` and `sampleParent` are a PAIR and must agree on a FRAME. The
dense half works in coarse BUFFER coordinates over the whole periodic domain
(origin `bx*RB`, sample wraps and goes through `cellIndex()`); the pool half
works in PARENT-LOCAL INTERIOR coordinates inside one tile (origin the
quadrant's own 0-or-RB offset, sample a bare `+GHOST` shift, no wrap and no
second slot lookup). Neither is meaningful without the other, which is the
argument for one fragment per parent kind rather than a set of independent
knobs -- and it is the thing a "just parameterize the differences" refactor
would have got wrong by exposing origin and fetch as separate overrides.

**What this bought that the line count does not show.** The dense half's
KNOWN GAP -- `wrapCoord` assumes the coarse level is periodic in BOTH axes, so
a block refined against a real WALL_Y wall would sample the periodic image
instead of reflecting -- was documented in the entry file's header, where it
read as a property of "the interp shader". It is a property of the DENSE
FETCH, and it now sits in the file that contains that fetch, next to it. The
pool half never had the gap and no longer carries the warning.

**Measured inert**, `index-cylinder-amr.html?levels=3`, 4096 steps from reset:
bit-IDENTICAL to the pre-B3 baseline, first run. Boot smoke PASS on all seven
pages; invariants PASS (all eight gates) on amr-dev and both N2/N3 x
diffuse/bounceback; analytic channel/TGV PASS; Cd/St unchanged apart from the
two standing red cells.

### B3-4 — DONE (2026-09-14). force1, and the origin buffers are now dead.

`amr_force1.wgsl` (231, level 1) and `amr_force1_pool.wgsl` (224, level>=2)
are one file under the first name. It collapsed for exactly B3-1's reasons and
had exactly B3-1's four differences -- origin, dxL, the diffuse band, and the
area/line weight, which IS dxL. So this stage was cheap; what it sets up is
not.

**`originX`/`originY` HAVE NO READER LEFT.** B3-1a took the fine step off them;
this takes the force pass off them, and those were the only two. What still
touches them is `shaders/amr_manage_pool.wgsl`, which WRITES `childOriginX/Y`
at refine time and READS `parentOriginX/Y` to do it -- four of that kernel's
bindings, on the kernel CLAUDE.md records as having sat at the exact
16-storage-buffer per-stage ceiling. The parent origin it reads has the same
closed form (`f32(bxP*RB) * 2 * PARENT_CELL_SIZE_L0`), so the whole cycle is
removable and `PARENT_HAS_CACHED_ORIGIN` with it. **That also retires the
eighth gate**: `debugCheckTileOrigins` scores a buffer that would no longer
exist, and leaving a checker pointed at deleted state is how this project got
its three vacuous gates (B2-2d). Delete it in the same commit, deliberately --
the rule stops needing a checker because it stops being stored.

**The binding layout was RENUMBERED CONTIGUOUS**, which the old file
explicitly declined to do ("a renumber is three separate pages' bind groups to
land in lockstep and this project has shipped that bug before"). It had two
holes by then -- 4/5 from the origin buffers, 7 from B4-3's masking. Done here
because it is now ONE layout instead of two, the edit is scripted with an
assert per site, and boot smoke on every page is the gate. Which earned its
keep immediately: the first attempt shipped a surviving `K_EPS` in the
pipeline constants that the shader no longer declares, and boot smoke caught
it as `Pipeline overridable constant "K_EPS" not found` rather than anything
subtler.

**Measured inert.** The bit-identity gate does NOT reach level 1's force at
`?levels=3` -- only the finest level's force pass is dispatched (B4-3), so
that run exercises level 3. The configs that exercise level 1's force are the
N=2 ones, and they are exact to the digit across builds:

    amr-N2-diffuse      Cd 1.642  St 0.1477     (unchanged, 3 builds)
    amr-N2-bounceback   Cd 1.322  St 0.1617     (unchanged; level 1 bounce-back)
    amr-N3-diffuse      Cd 1.425  St 0.1551
    amr-N3-bounceback   Cd 1.351  St 0.1616

plus bit-IDENTICAL at `?levels=3` in TWO different modes (see below), boot
smoke on all five AMR pages, and invariants PASS on all eight gates.

**AND THE GATE'S MODEL GOT CORRECTED TWICE, which is the durable output of
this whole stage.** B3-1b called a stray DIFFERS a "rare excursion". The kEps
commit found it was not noise but a second EXACTLY-reproducible attractor.
This stage found a third. Sixteen runs over six builds: 9 in mode A, 4 in B, 2
in C, every run bit-exact within its mode and the modes stable ACROSS BUILDS.
So a DIFFERS carries no information beyond "different mode", an IDENTICAL is
conclusive, and the strong form -- used here -- is to match the baseline in
two different modes, which no race can forge. Written into CLAUDE.md.

### B3-5 — DONE (2026-09-14). The origin buffers, and the gate that guarded them.

`originX`/`originY` are gone. `shaders/amr_manage_pool.wgsl` goes from 14
declared storage buffers to **10** -- it wrote `childOriginX/Y` at refine time
and read `parentOriginX/Y` to do it, and all four bindings went with the
closed form (`parentOriginL0`, one multiply). `PARENT_HAS_CACHED_ORIGIN` went
with them. That is the kernel CLAUDE.md records as having sat at EXACTLY the
16-per-stage ceiling; it now has six spare.

It also removed, on three pages: the per-slot GPU buffers, their CPU mirrors
(`originXCPU`/`originYCPU`), the host `tileOriginL0(level, slot, bx, by)` that
read them, the origin composition inside `debugActivateBlock`, and the
`originX`/`originY` arrays in the snapshot format.

**THE COMPOSITION WAS DRIFTED, AND NOBODY KNEW.** `debugActivateBlock`'s
JS-side mirror of the child-origin formula reads

    main-cylinder-amr.js   parentOrigin.x + qx * RB * parentCellSizeL0
    main-amr.js            parentOrigin.x + qx * RB * parentCellSizeL0 * 0.5
    main-reentry-amr.js    parentOrigin.x + qx * RB * parentCellSizeL0 * 0.5

The cylinder page carries a BUGFIX comment saying the `* 0.5` is wrong (RB is
already half the parent's interior) and that it mis-registered every manually
activated level>=2 tile. **That fix was never propagated to the other two**,
which are the pages that MOVE A BODY. It was latent -- only the manual
activation path writes it, and B3-1a/B3-4 had already taken the solver off
reading it -- but it is B3a's finding #3 again: a rule written down three
times, fixed in one copy. Deleting the rule is the fix that cannot drift.

**And the eighth gate went in the same commit as its subject.**
`debugCheckTileOrigins` (added as B3-1's prerequisite, three days of commits
ago) scored the live buffers against the closed form, precisely so the kernels
could be taken off them. With the buffers gone it had nothing to read. Leaving
it would have made it the FOURTH always-true gate in this project (B3a-4 found
two, B4-3 one, B2-2d one more). **Retiring a checker WITH its subject is the
counterpart to B2-2d's lesson, not an exception to it** -- the rule stops
needing a checker because it stops being stored. Back to seven gates.

**Measured inert.** Bit-IDENTICAL at `?levels=3` (mode A, twice), boot smoke
PASS on all five AMR pages, all three `refuse-*` configs PASS, channel/TGV
analytic PASS, and Cd/St to the digit:

    amr-N2-diffuse 1.642 / 0.1478   amr-N2-bounceback 1.322 / 0.1617
    amr-N3-diffuse 1.425 / 0.1551   amr-N3-bounceback 1.351 / 0.1616

Invariants PASS on all seven gates at every checkpoint, and the starved-pool
discrimination still holds (`--extra=maxFineBlocks=16`: five go red, `field`
and `quadrants` do not).

### B3-6 — DONE (2026-09-14). Criterion does NOT fit the shape, and that is the finding.

The plan said "smallest, do it for uniformity". Doing it measured why it is
not the same job as the other four, and the pair stays two files.

**They do not differ in a parent ACCESSOR. They differ in the DISPATCH
MAPPING.**

    amr_criterion.wgsl       one workgroup per L0 block -> ONE child criterion
    amr_criterion_pool.wgsl  one workgroup per QUADRANT of a parent tile's
                             2*RB x 2*RB interior -> FOUR per parent slot

B3-1..B3-4 each collapsed a pair that was the same kernel over a
differently-addressed parent; an accessor is exactly the right tool for that.
Here the loop nest itself is different, and the reason is not storage: level 1
is footprint-preserving 1:1 with L0 (plans/AMR-multilevel.md decision 1), so
its parent block IS its own block, while every deeper level takes a quadrant.
Forcing a shared kernel would mean parameterizing the dispatch shape, which
buys indirection and removes nothing. **63 + 83 lines, of which ~12 overlap** --
the worst ratio of the six, and the one place where "unify for uniformity"
would have made the code worse.

**What IS shared is now shared**, in `shaders/common_criterion.wgsl`:

- `discreteCurl(...)`, the vorticity stencil, which was written out twice
  (three times counting amr_render.wgsl, which legitimately keeps its own
  per-level normalized variants and is left alone).
- `wgReduceMax1(lid)`, the 64-lane tree max. **Both criterion kernels were
  still running the exact linear form `common_reduce.wgsl`'s header exists to
  argue against** -- 64 sequential dependent ops in lane 0 while 63 lanes
  idle, the pattern an on-device profile flagged at 18.6% of the macro-step
  per force pass. They were missed when the force kernels were converted.
  *The fix went where the profile pointed and not to every instance of the
  pattern* -- worth a grep next time a shape like that is fixed.

**AND THE FRAGMENT COULD NOT LIVE IN `common_reduce.wgsl`**, which is where it
obviously belongs. That file's `wgReduceSum3` reads includer-declared
`wg_fx`/`wg_fy`/`wg_tz`; adding a reduction that reads `wg_omega` would make
the fragment compile only in a file declaring all four. Measured, not guessed
-- the first attempt did that and `make wgsl` refused it with `no definition
in scope for identifier: wg_fx`. **A fragment that reads includer-declared
globals must be split along the same lines as those globals**, which is a real
constraint on this project's whole `common_*.wgsl` convention and was not
written down anywhere before.

**Measured inert.** The tree max is bit-identical BY CONSTRUCTION (max is
associative and exact on floats, unlike the sum `common_reduce.wgsl` had to
re-validate), and it reads that way: bit-IDENTICAL at `?levels=3`, boot smoke
PASS on all five AMR pages, invariants PASS on all seven gates, analytic
channel/TGV PASS, Cd/St unchanged.

### B3-7 — DONE (2026-09-14). manage: the predicates unify, the allocators do not.

The stage the plan flagged as able to eat the schedule (risk 2: "364 + 492
lines implementing two different balance schemes"). B2 had already removed
both balance schemes, so what was left was a small, sharp job -- 840 lines to
875 across three files, of which the two managers drop 343+497 to 295+427.

**WHAT UNIFIED: the decision predicates.** `isNearBody`/`inSpongeBand` existed
in both files. They are the same tests; they differed only in how the
candidate's centre and half-extent were derived, which is the one thing each
manager genuinely knows for itself:

    dense   an L0 block is BLOCK cells of size 1   -> half-extent BLOCK/2
    pool    a level-m tile's interior is 2*RB cells of size 2^-m
                                                   -> half-extent RB * 2^-m

Those are the SAME FOOTPRINT written two ways (decision 1's
footprint-preserving 1:1), and **at level 1 they are the same NUMBER** -- the
exact arrangement that lets two copies drift while agreeing on every test
anyone runs. The pool file's own comment said so and left it. B3-5 had just
found that shape biting for real: a sibling formula fixed on one page and left
wrong on the two that move a body.

So the caller supplies the centre and half-extent, and `nearBodyAt` /
`inSpongeBandAt` / `epsOf` live once in `common_refine.wgsl` -- which is
already the two managers' shared decision fragment and is included by nothing
else. Same split as B3-3's `parentOrigin`/`sampleParent`: what varies stays
with the caller, the rule is shared.

**WHAT DID NOT, and should not: the allocators.** Per-BLOCK (dense, one slot
from a slot free list) vs per-QUAD (pool, four slots from a quad free list) is
plans/AMR-multilevel.md decision 3, not an addressing accident. Like B3-6's
criterion pair, these differ in the DISPATCH AND ALLOCATION shape, and forcing
one kernel would parameterize the granularity to remove nothing. **Two of six
pairs did not fit the accessor shape, and both for the same reason** -- worth
saying plainly, because "six near-duplicate pairs" in B3's own table reads as
six of one kind and it is four and two.

**AND IT FOUND DEAD CODE THAT B2 LEFT.** `refine()` still computed a max over
its four prospective quadrants, its log2, the parent's origin and the parent's
centre. `coarsen()` still computed the child's eps, walked up to the parent
slot and derived the parent's centre. **Nothing below read any of it** -- B2-2d
deleted the three tests that consumed them (the grandchild cascade, the
hard-required split, the neighbour-active gate) and left the inputs standing.
Legal WGSL, free at runtime (a compiler drops an unused `let`), which is
exactly why it survived: the cost was that `refine()` READ as though it still
weighed the criterion, beside a twenty-line BUGFIX comment about a centre it
no longer computed for anything.

**Deleting a mechanism has to include deleting what fed it.** B2-2d ran a
sweep over the CHECKS that referenced the deleted machinery and found three
vacuous gates; the same sweep over its INPUTS was never run. That is the third
distinct way this project has been bitten by a removal's blast radius (vacuous
gates, a flag whose meaning silently widened in B3-1, and now dead inputs) --
and the general rule is: after removing a mechanism, grep for its name in
checks, in flags, AND in what computed its arguments.

**Measured inert.** Bit-IDENTICAL at `?levels=3` to the pre-B3 baseline; boot
smoke PASS on all five AMR pages; invariants PASS on all seven gates at every
checkpoint on amr-dev and N2/N3 x diffuse/bounceback; all three `refuse-*`
PASS; channel/TGV analytic PASS; Cd/St to the digit (1.642/0.1478,
1.322/0.1617, 1.425/0.1551, 1.351/0.1616).

**A FOURTH ATTRACTOR.** The first `?levels=3` run matched none of the three
known modes; the repeat was bit-identical to mode A. Consistent with the
family (ux relL2 1.9e-5, same scale as the others) and with the rule as
written -- a DIFFERS names the mode and nothing else, an IDENTICAL is
conclusive. Do not read "none of the known modes" as evidence of a change.

**B3 IS COMPLETE.** Six pairs: four collapsed to one kernel or one
kernel-plus-accessor, two measured as not that shape and left as two with
their genuine overlap extracted.

### B3a — the same sweep, in JS

**Measured 2026-09-14**, after B0 found one rule in five copies. Across
`main-amr.js`, `main-cylinder-amr.js`, `main-reentry-amr.js`,
`main-tgv-amr.js` and `main-channel-amr.js`, counting brace-matched function
bodies with comments and whitespace stripped:

```
42 shared function names
 - excluding init() and frame(), which ARE the page and should differ:
   40 names, 3,319 duplicated lines
 - 18 of those have already DRIFTED between copies
 - 1,129 duplicated lines are byte-identical across every copy
```

`init()` is 3,019 lines in all five and `frame()` 276, with five variants
each. Those are the pages themselves and are not the target; a page per
scenario is the 2D side's actual design, and B3's WGSL work is what reduces
what they each have to say. Everything else on the list is a rule that exists
once and is written down between two and five times.

**Take the byte-identical ones first**, because the gate is exact and the
review is trivial:

| function | copies | lines | note |
|---|---|---|---|
| `allocLevelPool` | 5 | 97 | the pool allocator itself |
| `debugSnapshotLoad` | 3 | 99 | |
| `debugDeactivateBlock` | 3 | 42 | |
| `debugProbeGhostFill` | 3 | 30 | |
| `debugInjectSyntheticField` | 3 | 18 | |
| `debugReadCardState`, `debugReadPool`, `debugReadDiag` | 2-3 | 13-26 | |
| `debugListActiveBlocks`, `resize`, `loadShader` | 5 | 9-12 | |

**Then the three that a later stage is about to touch**, which is why this is
a prerequisite and not tidying:

- **`tauAtLevel` -- 5 copies, 2 variants, and `card-params.mjs` already
  exports and tests it.** `main-amr.js` delegates to it; the other four still
  inline `for (i < m) t = 2*t - 0.5`. Same answer today. This is the identical
  duplication B0 removed from `amr2d.mjs`, still live in four pages, on a
  *physics* rule. Cheapest item on the list.
- **`debugCheckGeometryCoverage` -- 3 copies, 2 variants, 100 lines**, and B4
  changes what question it asks. Extract it BEFORE B4, or B4 is three edits
  and a chance to land in two of them.
- **`debugSnapshotSave` -- 3 copies, 3 variants, 103 lines**, i.e. fully
  drifted already, and B2 and B3 both move the snapshot's shape.

**One thing on the list that is NOT drift, checked rather than assumed.**
`S_Advance` reports 2 variants, and CLAUDE.md calls it dimension-agnostic and
verbatim -- so this looked like rot in the scheduler. It is not: the
*schedule* is identical in all five, and the whole difference is
`main-amr.js`'s `beginPass`/`skipGroup` instrumentation -- the `?bench=1`
pass-group attribution sweep and its `legacyGhostPLs` second pipeline set,
which exist so the bench harness can A/B without a page reload. A deliberate
main-amr.js-only capability. **And the related worry is also cleared:**
`?ghostcopy=1` selects `DIRECT_GHOST: 0` at pipeline creation on every page
(`main-cylinder-amr.js:1198` as well as `main-amr.js:1351`), so
`--extra=ghostcopy=1` does A/B the real legacy path across the whole suite,
exactly as CLAUDE.md says. Do not "fix" either of these.

**Gate: bit-identical, per extraction.** Each is a pure code motion.
`tools/amr-diff.js` against a pre-change snapshot for anything touching the
solver; for the debug-only ones, B0's own gate shape -- run a
`validate-all.js` config before and after and require the same numbers, on a
build whose repeat-reproducibility was established first.

### B3a — LARGELY DONE (2026-09-14)

Four commits. Duplication outside `init`/`frame` is down from 3,319 lines to
2,419; the byte-identical share from 1,129 to 613.

```
B3a-1  loadShader, resize, tauAtLevel          -408 lines  (TEN pages each,
                                                            not five)
B3a-2  allocLevelPool                           -380
B3a-3  readPoolIndirection, listActiveBlocks    -203
B3a-4  two always-true geometry-coverage stubs   -15
```

**THREE THINGS FOUND BY DOING IT, and each is worth more than the lines.**

1. **`make check` CANNOT GATE A CODE MOTION IN THESE FILES.** My extraction
   script had an off-by-one and left the `f` of `function allocLevelPool`
   behind in five pages, as a bare expression statement. `node --check` is a
   PARSE check: `f` on its own line is syntactically valid. Every page threw
   `ReferenceError: f is not defined` at module evaluation and never defined
   its global, and `make check` was green throughout. **`validate-all.js`'s
   BOOT SMOKE is what caught it** — which is exactly the gap CLAUDE.md records
   that config being added for after 238e48c. Boot-smoke every page after any
   code motion here, every time.

2. **A DOC COMMENT HAD BEEN STRANDED FIFTY LINES FROM ITS FUNCTION, in three
   files.** `debugActivateBlock`'s description sat above `readPoolIndirection`,
   which had been inserted between them; a second, newer comment was later
   written above the function, so it had half its documentation adjacent and
   half orphaned. Found because the extraction tooling prints the first and
   last line of every block before deleting it — without that it would have
   been deleted silently. **Any scripted surgery here should print what it is
   about to cut.**

3. **`debugCheckGeometryCoverage` WAS NEVER THREE COPIES.** It is one
   implementation and two STUBS, distributed backwards:

   ```
   main-amr.js           has a body (the falling card)   NO coverage check
   main-reentry-amr.js   has a body                      NO coverage check
   main-cylinder-amr.js  has a body                      the real check
   main-tgv-amr.js       HAS_BODY = 0                    stub: {ok:true}
   main-channel-amr.js   HAS_BODY = 0                    stub: {ok:true}
   ```

   And the stubs defeat a design decision: `tools/lib/amr-invariants.js`
   PROBES for the function and reports it SKIPPED when absent precisely so a
   missing check "can't quietly look greener than it is". Returning `{ok:true}`
   converts that honest skip into a green tick standing for nothing. Deleted.

   **So B3a's entry for it is closed, and B4's just got bigger:** the shipped
   falling-card page and the reentry page both move a body through a refined
   region **with no geometry-coverage gate at all**. B4 has to write one for
   them, not just change the predicate in an existing one.

**What is left**, all still byte-identical and none of it blocking:
`debugSnapshotLoad` (2 dup copies, 99 lines), `debugDeactivateBlock` (84),
`debugProbeGhostFill` (60), `debugInjectSyntheticField`, `debugReadCardState`,
`debugReadPool`, `debugReadDiag`. `debugSnapshotSave` is 3 copies in 3
variants and should wait for B2/B3, which move the snapshot's shape anyway.

### B4-1 — the coverage gate — DONE (2026-09-14)

B3a's finding #3 said B4 had to WRITE a geometry-coverage check for the two
pages that move a body, not just change the predicate in an existing one.
Done: `ellipsePhi`/`bodyPhiL0`/`bufferToWindow` in `amr2d.mjs`,
`checkGeometryCoverageOnGPU` + `readCardState` in `amr2d-gpu.mjs`, six more
mutation-scored checks in `tools/test-amr2d.js`, and the check now live on
`main-amr.js`, `main-reentry-amr.js` and `main-cylinder-amr.js`. The
predicate is an ARGUMENT and still defaults to the kernel's centre sample, so
B4-2 flips both sides together.

**Three things it changed about the rest of B4.**

1. **THE LOOKAHEAD IS NOT PART OF THE CONSTRAINT.** The check fired on the
   first moving body it saw — 2 violations at step 1024 of `index-amr.html` —
   and it was the checker that was wrong. `FORCE_REFINE_LOOKAHEAD` is the
   kernel's MECHANISM, not the requirement: the manager decides every
   `REFINE_EVERY` steps, so at t0 it covers `[t0, t0+LOOKAHEAD]` and thereby
   keeps the *now*-condition true until `t0 + REFINE_EVERY`. A checker that
   also applies the lookahead asks about a window the last decision was never
   responsible for, and reports the leading edge of a moving body as a
   violation on a correct run. **A pinned body cannot show this** (v = omega =
   0 makes the future pose the current one), which is why the only existing
   implementation had it backwards and nothing noticed. B4's own text already
   says the refine-ahead framing is the criterion's job — this is the other
   half of that sentence, and it belongs in the checker's header, not in a
   commit message.

2. **THE OLD CHECK HAD ALREADY DRIFTED, in the direction its own header
   denied.** It was defended as "an INDEPENDENT re-derivation to catch drift
   in the shader's version". Its `get_phi` computed the superseded ALGEBRAIC
   distance `(hypot(lx/a, ly/b) - 1) * b`, which `common_geometry.wgsl`
   replaced with a true Newton distance — exact for a CIRCLE, and short by up
   to a/b (8x on the card's default aspect) for an ellipse. It was never a
   second opinion; it was the only one, on the only page where the error is
   invisible, and copying it to the card page would have flagged a ring of
   tiles the kernel never considered near the body. **Duplication defended as
   independence is the specific thing to distrust here** — the independent
   route has to be a different DERIVATION (here: a brute-force closest point
   on the boundary), not a second transcription.

3. **`?levels>=3` HAD A FATAL INIT ERROR ON FOUR OF THE FIVE AMR PAGES, and
   the suite could not see it.** B3a-1's `tauAtLevel` extraction landed the
   CALL in all five pages and the IMPORT in one, so the other four threw
   `ReferenceError: tauAtLevelOf is not defined` at init. Invisible at every
   page's own `levels=2` default, because `updateLevelParams`'s
   `for (c = 2; c < N_LEVELS; c++)` loop is VACUOUS there — the same shape as
   B3a-4's always-true gates and the `?levels=1` tgv defect, which is now
   three instances of one class. `make check` was green throughout (B3a's
   finding #1 again: it is a PARSE check). Fixed, and gated by three new
   `checkBoots` configs at `?levels=3` — `cylinder-amr-boot-N3`,
   `tgv-amr-boot-N3`, `channel-amr-boot-N3`. **Any new URL-selected mode
   wants a boot config at the setting where its loop is not vacuous.**

**What is left in B4**, unchanged in substance by the above: the box
predicate in both managers (and `paramsForChildLevel`'s `childLevel===2`
margin special case retired with it), the finest-wins masking and
`childBlockSlot` bindings deleted from the three force kernels, the init-time
refusals (margin vs. stencil reach; a body with AMR and no geometry-forced
refinement), and pool exhaustion latched as `error:` instead of degrading
silently.

### B4-2 — the box predicate — DONE (2026-09-14)

Both managers' geometry test is now `nearBodyBox` (`shaders/common_geometry.wgsl`,
the WGSL twin of `amr2d.mjs`'s `nearBodyWant`), the `u32()` window truncation
is gone, `paramsForChildLevel`'s `childLevel===2` margin special case is
retired, and `?boxrefine=0` keeps the old test in the same build.

**The two halves are ONE change, and trying to stage them proved it.** Fixing
the test while keeping the margin slack that existed to compensate for the
broken test SATURATED level 2's pool — 128/128, `MAX_FINE_BLOCKS` — so
geometry-forced refinement was being silently refused. Two safety factors for
one hazard is worse than either. Measured at `?levels=3&bounceback`, 8192
steps, with `debugForceBreakdown` reading the coarser levels' own force passes
(they must be **exactly zero**, which is the sharp form of "the body lives
entirely on the finest level"):

```
predicate  margin2   L1 tiles  L2 tiles   L0 force   L1 force
centre        8         51        64        0.0000    0.0000     before
BOX           8         67       128 (!)    0.0000    0.0000
BOX           4         52        64        0.0000    0.0000     now
```

The box test at the honest margin reproduces the centre test at the inflated
one, tile for tile. That is B4's claim demonstrated rather than argued.

**Three things worth carrying forward.**

1. **`debugForceBreakdown` is the instrument for B4's remaining work, not
   Cd.** "Every coarser level's force pass integrates to exactly zero" is a
   direct, unambiguous statement of the premise the masking deletion rests on;
   Cd is a time-averaged surface integral that would absorb a small violation
   without comment. Same lesson CLAUDE.md already records for precision work
   (the analytic gates, not Cd/St) — it generalizes.

2. **`?levels=4` IS BROKEN, and this is the first thing to say so
   concretely.** At `levels=4&bounceback`: coverage reports 20 violations,
   level 3 sits at 128/128, and `debugForceBreakdown` reads L2 fx=-2.33
   against L3 fx=+2.26 — large cancelling contributions, i.e. exactly the
   "body-adjacent block computing its force at a coarser level than the build
   is configured for" pathology the coverage check exists to catch.
   Consistent with `main-cylinder-amr.js`'s own "N_LEVELS>=4 is untested"
   note. **B4's refuse-rather-than-degrade item now has a live reproducer**,
   which is worth more than the refusal itself: it means the latch can be
   tested by something other than a deliberately under-provisioned pool.

3. **A `?flag=0` control restores a TEST, not a CONFIGURATION, and the
   difference has to be written on the flag.** `?boxrefine=0` alone leaves the
   retired margin special case retired, so the full pre-B4 leg is
   `?boxrefine=0&forceRefineMargin2=8`. Every remaining stage here ships this
   shape of control (`?dcpre=1`, `?window=`, `?solideq=0`); each should say
   what its `0` does and does not restore, or an A/B measured a year from now
   will quietly compare two things that differ in three ways.

**And `validate-all` now prints Cd/St on PASS** — the gap 46157cd closed for
the analytic gates was still open on the cylinder path, so the numbers were
computed and discarded unless a config was already red. None of the table
above would have been visible otherwise, and every remaining stage that moves
the refined region needs it.

### B4-3 / B4-4 — the masking, and the refusals — DONE (2026-09-14)

**B4-3.** One force pass, at the finest level. The finest-wins masking is gone
from all three force kernels, with `HAS_CHILD`, three `childBlockSlot`
bindings, and the last reader of `LevelParams`'s `nbx`/`nby`/`hasChild` — so
`amr_force1_pool.wgsl` now declares the same 4-field prefix every other pool
shader does instead of being the one file that needed all 8.

**Measured before deleting, at 8192 steps, raw i32 accumulators (FSCALE=1e7):**

```
levels=2              L0 0             L1 237344  (finest)
levels=3              L0 0    L1 1     L2 214591  (finest)
levels=2 bounceback   L0 0             L1 201702  (finest)
levels=3 bounceback   L0 0    L1 0     L2 207126  (finest)
```

Exactly zero bar one 1e-7 unit — a single workgroup's truncated partial.
**The 4-decimal float form of the same readings says "0.0000" against a total
of ~0.02**, which still leaves room for 0.25% of the drag on a level about to
stop being computed. Take the raw integer when the question is "is this
exactly zero"; the convenience formatting is not the instrument.

**B4-4.** The refusals, and — more usefully — three configs that require each
one to actually fire (`runBootSmoke`'s new `expectError`). *A guard that never
fires and a guard that cannot fire are indistinguishable from a green suite*,
and the rest of this plan adds several more guards.

**Two findings, both the same shape one layer apart, and both found by the
gate rather than by reading the code.**

1. **`?levels=1` was never the bug B0b recorded.** It is not `resetSim`
   dereferencing `pools[1]`: all five pages already had
   `if (N_LEVELS < 2) throw`, and it was always correct. It throws at MODULE
   SCOPE, which `init().catch(handleErr)` cannot catch — module evaluation has
   already failed, so `init()` never runs and its `.catch` never attaches. The
   page refused correctly and said nothing. `refuseConfig` reports first and
   throws second.

2. **The status line could not hold a fatal either.** Every live page rewrites
   `#status` every ~250ms from an async readback, so a fatal raised between
   two of those writes is overwritten by a step counter that never advances
   again. The pool-exhaustion refusal fired, wrote its message, and vanished;
   `validate-all` reported *"the page kept running (status [AMR] step 832…)"*
   for a page that had already stopped. `reportFatal` now latches the element.

   **The general lesson for the rest of this plan: a diagnostic is not
   delivered until it survives the next 250ms.** Both of these were correct
   code that produced a misleading observation, and in both cases the
   misleading observation pointed *away* from the real cause.

**And a mechanism note worth reusing.** The runtime latch needed no shader
change and no new binding: a cheap always-sound-in-one-direction TRIP-WIRE
(every level's `freeCount`, 4 bytes, one submit — exhaustion implies zero, so
a zero cannot be missed) gated onto the AUTHORITATIVE check (coverage) only
when it trips. The direct signal would have been a refusal counter in the
manager, but `amr_manage_pool.wgsl` has no diag binding at all, so that is a
new binding across five pages' bind groups — the 238e48c shape. **Prefer a
host-side trip-wire over a new binding whenever the authoritative check
already exists.**

### B4 — the body lives entirely on the finest level

**2D already asserts this** — `debugCheckGeometryCoverage`: every leaf tile
near the body must already be at the finest configured level, and
`tools/validate-amr-invariants.js` gates it. So finest-wins masking in the
force kernels is **provably dead code**:

- `shaders/amr_force.wgsl:80`
- `shaders/amr_force1.wgsl:139-151` (+ `childBlockSlot` at binding 4)
- `shaders/amr_force1_pool.wgsl:128-137` (+ `childBlockSlot` at binding 7,
  `hasChild` in `LevelParams`)

Delete the masking, the `childBlockSlot` bindings, the `HAS_CHILD` override,
and run exactly ONE force pass, at the finest level. Two consequences worth
writing into the headers: no finest-wins masking anywhere, and **no
coarse/fine seam ever touches the body or its margin** — which makes "how
should an interface behave with a body crossing it" unreachable rather than
merely unanswered, and that matters for B6.

**Enforce it rather than assume it.** 3D turned the same statement into two
refusals: a body with AMR needs geometry-forced refinement enabled, and the
refinement margin must cover the force stencil's reach. 2D has
`FORCE_REFINE_MARGIN` (`main-amr.js:358`) and `FORCE_REFINE_LOOKAHEAD`
(`main-amr.js:417`) — check them at init against the stencil reach and refuse.
And adopt 3D's **refine-ahead** framing explicitly: the criterion must use
`MARGIN + refineEvery * |v|`, because a shell that only just covers the body
at decision time is stale on the next step and the constraint has to hold at
*every* step. 2D's `FORCE_REFINE_LOOKAHEAD` already defaults to
`REFINE_EVERY`, so this is a rename and a comment, not a change — but say it,
because 3D's `drift` scenario fails without it and someone will delete it as
an unexplained fudge.

**And adopt the hard-failure rule (3D M5.4a).** Running out of pool slots is
geometry-forced refinement being *refused*, which means a seam through the
body. 2D currently degrades silently (`MAX_FINE_BLOCKS`, and
`main-amr.js:3585`'s note that `granted=1/byCascade=0` means balance was still
spreading when the loop ran out). Latch it, write `error:` into `#status`,
stop advancing. `validate-all.js`'s boot smoke already polls `#status` for
exactly this shape.

**Gate:** bit-identical on every pinned-body config (the masking never fired);
`amr-dev-invariants` unchanged; a deliberately under-provisioned
`?maxFineBlocks=` run must now FAIL loudly, and a config should exist that
proves the refusal fires.

### B5 — the window is a translation of the sponge — **CLOSED (2026-09-15)**

**OUTCOME.** The body lives in BUFFER coordinates. `cx`/`cy` are buffer
positions, integrated and wrapped into [0, W) x [0, H) every step; a kernel
that owns a buffer cell already has the body's frame and converts nothing.
Buffer->window survives for exactly three things, all anchored to the VIEW
rather than the fluid: the ALBC sponge band, the WALL_Y walls, and the render.
The window-anchored convention, its `?window=` flag and its four frame
accessors are DELETED -- there is one convention, not a switch.

Stages: B5-0 (dense step buffer-dispatched, two dead kernels deleted), B5-1
(nine hand-written conversions -> two functions), B5-2 (the frame split, behind
a flag), B5-3 (the body's initial placement honoured; `?upstream=` made real),
B5-4 (the precision claim measured, and RETIRED -- it was `TOTAL_WRAP_SCREENS`,
not the convention), B5-5 (`lbm_force.wgsl`, the one kernel that silently
ignored the flag), B5-6 (default flipped, old path deleted), B5-7 (that
kernel's DISPATCH, the last window bookkeeping in the tree).

**WHAT IT COST AND WHAT IT BOUGHT.** Net ~150 lines removed. Every pinned-body
config is bit-identical (`off = 0` makes the two conventions the same map);
the re-baselining is confined to MOVING bodies and is small --
`index-reentry.html` at 4096 steps moves by relL2 2.1e-3 in ux. The stated
precision benefit did NOT survive measurement (B5-4). What B5 actually bought
is the conversion count: nine sites in six files, in two non-interchangeable
idioms, down to three view-anchored callers.

**NO RESIDUAL.** B5-7 took `lbm_force.wgsl`, the last window-DISPATCHED
kernel, onto buffer dispatch, so the plan's wording ("the step, the force
kernels and the manager lose their window bookkeeping entirely") is now met
without exception. **`state.off_x`/`off_y` are read by the render alone**, and
written by `physics.wgsl`/`amr_physics.wgsl` -- which makes the audit that
found B5-5's bug a one-line grep with no exceptions to remember.

**Everything below this line is the original statement of the problem and the
stage-by-stage record.** It is kept because the reasoning is the useful part;
read the file/line references as historical.


**Where 2D actually stands** (CLAUDE.md's description of the 2D scheme is one
generation out of date):

- `main.js` / `lbm_*.wgsl` — **still window-coordinate dispatch**, converting
  to buffer at every load and store (`lbm_step.wgsl:90,125`,
  `lbm_stream.wgsl:47,58`, `lbm_collide.wgsl:74`, `lbm_force.wgsl:62,109`).
  This is the shape 3D rejected.
- The AMR pages — **already buffer-coordinate dispatch**
  (`amr_step.wgsl:9-19`), converting buffer->window only where something is
  physically anchored. Half the job is done.

**What is left.** The body is still held at a fixed *window* position
(`amr_physics.wgsl:95-106`: `cx = W/2 + frac`, `off_x = floor(x_total) mod W`),
so every kernel that evaluates the SDF still converts:

- `shaders/amr_step.wgsl:87`
- `shaders/amr_step1.wgsl:211,234,262` (and the continuous `wrapf` path)
- `shaders/amr_force.wgsl:87`, `amr_force1.wgsl:157,180`
- `shaders/amr_manage.wgsl:175,199`

3D's reading puts the body in **buffer** coordinates with a nearest-image wrap
in the SDF, and converts **only in the sponge**. Then the step, the force
kernels and the manager lose their window bookkeeping entirely, and the
continuous `wrapf` conversions — the ones with the sub-cell precision
exposure — go with them.

**Two things come free:**

- **Bounded precision.** `card-total.mjs` exists because the 2D card's
  sub-cell position is the fractional part of an unbounded f32 accumulator
  whose ULP eventually eats it; `TOTAL_WRAP_SCREENS` (`amr_physics.wgsl:72-93`)
  is the patch. Under the 3D convention the body's position is wrapped into
  `[0, W)` every step, so its ULP is fixed at `W * 2^-24` forever by
  construction. `x_total` survives as **reporting only** — nothing reads it
  back into the simulation — which is exactly 3D's arrangement.
- **A real gate.** The two schemes are related by a pure *permutation of
  buffer cells* (both floor the offset and keep the fraction on the body), so
  after unshifting, the fields must agree **bit-for-bit**.
  `tools/lib/field-reconstruct.js`'s `unshiftField` already does the
  unshifting. That is a stronger gate than anything else in this document.

**Do it as:** `?window=` selecting old/new, exactly as 3D ships `?window=0` as
its control, so both live in one build. Refuse rather than degrade: no body,
a walled axis, or no sponge (without an absorbing band the wake wraps round
and the body flies back into it, which looks perfectly healthy).

**Scope note.** Converting `main.js`/`lbm_*.wgsl` to buffer-coordinate
dispatch is a separate, smaller commit and should go first — it makes the
base page consistent with the AMR pages and is gated by `index-boot` plus a
dense snapshot diff.

### B5-0 — DONE (2026-09-14). The dense step is buffer-dispatched.

`shaders/lbm_step.wgsl` now owns a BUFFER cell per thread and converts to
window coordinates only for the three physically-anchored things: the body
SDF, the ALBC sponge band, and the WALL_Y channel walls. It was the last
window-dispatch kernel in the tree.

**The round trip it removes is the point.** Streaming used to compute a window
source and then map it straight back to a buffer source -- composing a shift
with its own inverse, once per direction per cell:

    window:  wsrc = (x - e) mod W,  bsrc = (wsrc + off) mod W
    buffer:  bsrc = (bx - e) mod W            with x = (bx - off) mod W

Provably the same integer, no reduction anywhere in the kernel, and the cell
each thread WRITES has not moved -- so B5's "unshift and compare bit-for-bit"
gate degenerates here to a plain comparison.

**AND THE GATE HAD TO BE CHOSEN CAREFULLY, which is worth recording.**
`dense-reference` is the obvious dense config and it is the WRONG ONE: a
PINNED cylinder never moves the window, so `off` stays 0, the shift is the
identity, and the config cannot tell the two readings apart at all. The page
that can is `index-reentry.html` -- dense, a MOVING body, and (unlike
`index.html`) it exposes `window.__CYL` with `debugSnapshotSave`. Measured
there, 4096 steps from reset:

    same build, twice   IDENTICAL   (the dense pages are exactly
                                     deterministic -- no pool free list)
    pre vs post         IDENTICAL

Plus `dense-reference` Cd 1.951 / St 0.1260 exactly (CLAUDE.md's recorded
bit-exact value), `index-boot`/`reentry-boot` PASS, and
channel-poiseuille/couette-dense + tgv-dense PASS.

**`lbm_stream.wgsl` and `lbm_collide.wgsl` are DELETED.** The plan cited them
as evidence that the dense path was window-dispatched; they are in fact DEAD
-- unreferenced by any page, tool or Makefile target since the fused
`lbm_step.wgsl` replaced them, and only kept green by `make wgsl` compiling
every file in the directory. They were also **the last two copies of `struct
CardState` outside `common_geometry.wgsl`** (checked: not yet drifted), which
is the struct amr2d-gpu.mjs's `CARD_STATE_KEYS` warns can rot silently because
inserting a field re-labels every field after it rather than failing. A dead
file that still compiles is a rot surface `make check` actively protects.

### B5-1 — DONE (2026-09-14). Nine conversions become two functions.

Every buffer->window conversion in the solver now goes through
`bufferToWindowCell` or `bufferToWindowPos` in `common_geometry.wgsl`. There
were NINE hand-written sites across six files, and **they were not all the
same conversion** -- which is the reason to count them before changing them:

    cells      (c + N - u32(off)) % N      integer; TRUNCATES off
    positions  wrapf(p - off, N)           continuous; keeps off's fraction

Five sites used the first (`amr_step`, `amr_force`, `lbm_step`,
`common_refine`'s legacy `?boxrefine=0` path and `inSpongeBandAt`), four the
second (`amr_step1` x2, `amr_force1` x2). The second family is exactly what B5
exists to remove -- `off` is a float carrying the body's fractional travel, and
the fine levels evaluate at fractional L0 positions -- so they had to be
distinguishable, not merged.

**This is where the `?window=` switch goes.** B5 proper changes what "convert"
means; doing it across nine sites in six files is the shape that ships a
half-converted build. It is now one function.

**`wrapf` HAD THREE COPIES** -- `amr_step1.wgsl`, `amr_force1.wgsl` and
`amr_render.wgsl`, byte-identical. Found because moving it to
`common_geometry.wgsl` made `make wgsl` refuse the render page with
`redefinition of wrapf`; the other two were deleted deliberately, the third
was found by the compiler. Worth noting which gate catches which: `make check`
cannot see a code motion (B3a finding #1) but it does see a name collision.

**And the host already disagrees with all nine.** `amr2d.mjs`'s
`bufferToWindow` is a BARE SUBTRACTION with no wrap, because `get_phi` already
takes the nearest periodic image -- so every kernel's wrap is work `get_phi`
immediately redoes. The kernels keep it for now: this commit is inert by
construction and dropping the wrap is a real (if tiny) numerical change. It
belongs with B5 proper, where it is one line instead of nine.

**Measured inert.** Bit-IDENTICAL on `index-reentry.html` (dense, moving
window) AND on `index-cylinder-amr.html?levels=3`; boot smoke PASS on all
seven pages; invariants PASS on all seven gates; every dense and AMR
analytic/physics config unchanged to the digit.

### B5-2 — the buffer convention exists, behind `?window=0`, and does NOT validate yet.

*(Superseded: B5-3 made it validate, B5-6 made it the only convention.)*

Implemented and gated as the plan asked (both conventions in one build,
default unchanged). The DEFAULT path is bit-IDENTICAL on
`index-reentry.html` and `index-cylinder-amr.html?levels=3`, boot smoke PASSes
on all seven pages, invariants PASS on all seven gates, and every physics
number is unchanged. `?window=0` runs. It does not yet reproduce the pinned
cylinder, which it must, so the default stays 1 and nothing was flipped.

`common_geometry.wgsl` gains `WINDOW_BODY` and four frame accessors --
`bodyFrameCell`, `bodyFrame`, `bodyFrameBare`, `windowToBody` -- and every SDF
site now asks for the BODY's frame while the sponge and the WALL_Y walls keep
asking for the WINDOW's. That separation is the substance of B5 and it is
done; what is unresolved is where the body's position comes from.

**TWO THINGS MEASURED, AND THE SECOND IS NOT ABOUT B5 AT ALL.**

**1. The view tracks TRAVEL, not position.** My first derivation set
`off_x = floor(cx - W/2)`, assuming the view is centred on the body. It is
not: `main-cylinder.js` places its cylinder UPSTREAM diameters in, at
cx = 170.67 with W = 512, so that put `off_x` at 426, the sponge band landed
mid-domain, and Cd came back **808.897** against a literature 1.35. Caught on
the first run, because the pinned cylinder is the one config where the two
conventions MUST agree exactly. `off` now keeps its existing derivation from
`x_total`, and only the INTEGER part of that accumulator is ever consulted --
the sub-cell position, which is the precision-sensitive part, no longer comes
from it at all. Cd went 808.897 -> 2.028.

**2. `?upstream=` IS INERT ON THE DENSE CYLINDER PAGE, and has been all
along.** That remaining 2.028-vs-1.951 gap is not a B5 bug. `physics.wgsl`
step 5 overwrites `cx` with `initial_cx + frac(x_total)` = W/2 EVERY STEP,
discarding the page's own `CX0 = UPSTREAM * 2 * R` placement. The buffer
convention integrates the body's position instead of recomputing it, so it
HONOURS that placement -- which is why its cylinder sits at 170.67 and drags
differently. Confirmed directly on the shipped path:

    ?upstream=4    Cd 1.951  St 0.1260
    ?upstream=8    Cd 1.951  St 0.1260     (the default)
    ?upstream=20   Cd 1.951  St 0.1260

Bit-identical across a 5x change in a documented knob, on this project's
reference config. `dense-reference` is not running the cylinder where its own
parameter says it is.

**AND THE DEBUG PATH DISAGREES WITH THE LIVE PATH**, which is why this hid.
`debugStepSync` leaves `cx` at 170.67; the live loop parks it at 256. A probe
of the card state says one thing and the harness measures another, so
`?upstream=` looks like it is doing something to anyone who checks it the
cheap way.

**What B5 needs next, and it is a DECISION not a fix.** The window convention
throws the body's initial placement away; the buffer convention keeps it.
Making `?window=0` reproduce `?window=1` on the cylinder means replicating
"overwrite cx with W/2 every step" -- i.e. deliberately discarding the
placement freedom the buffer convention exists to provide. The alternative is
to fix `?upstream=` so it means what it says, which MOVES `dense-reference`'s
numbers and re-baselines the project's reference config. That is a physics
call, not a refactor, and it belongs to whoever owns the validation harness.

**Also still open:** `lbm_force.wgsl` is the last window-DISPATCHED kernel. It
needs no window coordinates at all under `?window=0` (its only window use is
the SDF), so converting it is free once the default flips -- and it is the
same atomicAdd-regrouping re-baselining noted in B5-0.

**What is left of B5** is the real change: the BODY moves into buffer
coordinates, the SDF takes the nearest image, and the conversion survives only
in the sponge -- with `?window=` selecting old/new, and a refusal (not a
degradation) when there is no body, a walled axis, or no sponge. Note that
`lbm_force.wgsl` is deliberately NOT converted here: its atomicAdd reduction
is per-WORKGROUP, so re-tiling the dispatch regroups the truncated partials
and the total moves in the last fixed-point digit whenever `off` != 0 -- the
same class as the AMR free-list note in CLAUDE.md. That is a re-baselining
change and belongs with B5 proper, not bundled into an inert one.

### B5-3 — DONE (2026-09-15). The placement is honoured, and `?window=0` validates.

B5-2 ended on a decision, not a fix: the window convention threw the body's
initial placement away and the buffer convention kept it, so the two could not
agree on the pinned cylinder. **The decision taken was the third option B5-2
did not see** -- honour the placement AND move `?upstream=`'s default to the
value that was actually running, so the knob becomes real without moving a
single baseline.

**The arithmetic that makes it free.** `CX0 = UPSTREAM * 2 * R` and
`R = W / (2 * BLOCKAGE)`, so `CX0 = UPSTREAM * W / BLOCKAGE`. At the default
BLOCKAGE = 24, **UPSTREAM = 12 IS exactly W/2** -- for any W and any `res`.
The shipped cylinder has therefore been running at 12 diameters all along
while its own parameter said 8. Setting the default to 12 is documentation
catching up with reality, and it is why nothing re-baselines.

`physics.wgsl`/`amr_physics.wgsl` gain `INITIAL_CX`/`INITIAL_CY` overrides
replacing the hardcoded `f32(W)/2.0`. **They have NO DEFAULT on purpose.** A
sentinel default would let a page that forgets to supply one degrade quietly
back to W/2 -- the exact bug being fixed -- whereas a missing override is a
`CreateComputePipeline` error, which every page turns into `error:` in
`#status` and the boot smoke catches. All six pages that dispatch a physics
kernel supply them: `W/2, H/2` for `main.js`, `main-amr.js` and both reentry
pages (bit-identical to the old constant), `CX0, CY0` for the two cylinder
pages.

**WHAT THE OLD BUILD ACTUALLY DID ON STEP 0, which is the part worth keeping.**
`dispatchMacroStep` runs force BEFORE physics. `cardInit()` seeded cx = CX0 =
170.67 and physics overwrote it with 256 on that same step -- so step 0 applied
the body's solid coupling at a position the rest of the run never used. Seed and
pin now agree. That inconsistency was invisible to every gate because it lasted
one step.

**MEASURED -- `?upstream=` IS LIVE, and its old value was a bad configuration.**
`dense-reference`, Re=100:

    upstream    Cd       St         cx (probed live)
    8          2.028    0.1277     170.667
    12         1.951    0.1260     256.0       <- the shipped default
    20         2.209    2.0328     426.667

St 2.03 at upstream=20 is not a bug: 426 of 512 puts the cylinder ~4D from the
outlet, the wake is destroyed, and the shedding peak is meaningless. The knob
now reports that instead of hiding it. Compare B5-2's table, where 4, 8 and 20
all gave 1.951 bit-identically.

**AND `?window=0` NOW REPRODUCES THE PINNED CYLINDER -- AT TWO PLACEMENTS.**

    default (upstream=12)   window=1  1.951/0.1260   window=0  1.951/0.1260
    upstream=8              window=1  2.028/0.1277   window=0  2.028/0.1277

Agreeing at ONE placement could be coincidence of a shared default; agreeing at
two, including the one that used to read 2.028 against 1.951, is the
conventions genuinely meeting. B5-2's blocker is closed and B5 proper is
unblocked.

**INERT WHERE IT MUST BE, proved rather than argued.** `index-reentry.html`
(dense, MOVING window) at 4096 steps from `reset()`, via `debugSnapshotSave` +
`tools/amr-diff.js`: same build twice IDENTICAL (relL2 0 on ux/uy/rho/omega,
which is what makes the comparison meaningful), and **pristine HEAD vs this
build IDENTICAL**. `dense-reference` 1.951/0.1260 on both trees -- and dense is
bit-exact deterministic, so identical there is conclusive, not indicative.
Boot smoke PASS on all seven pages, all three refusal configs PASS, all seven
invariant gates PASS on every AMR config, every channel/TGV analytic gate PASS.

**CLAUDE.md's `amr-N2-diffuse` Cd 1.620 IS STALE, and that is not this
change.** Measured on a pristine HEAD checkout, properly served: 1.642, 1.642.
This build: 1.643, 1.643. Both stable to four digits, 0.001 apart -- the
documented floor. The recorded 1.619/1.620 predates some earlier stage; do not
read a 0.02 gap against it as a regression.

**THE MEASUREMENT HAZARD THAT NEARLY SANK THIS STAGE.** `ensureServer` only
checks that *something* answers on the baseUrl port and reuses it, whatever
tree that server's cwd is. With several worktrees live, the default
`https://localhost:4444` belonged to the MAIN checkout and 4445 to the 3d-trt
worktree -- so a full sweep and its "pristine A/B" both ran, agreed with each
other, looked entirely clean, and measured neither this branch nor its
baseline. Every number had to be thrown away. Two runs agreeing is not evidence
they ran your code; `curl` the served file for a token you added, and pass both
`--baseUrl=` and a private `--port=`.

**Still open, unchanged from B5-2:** `lbm_force.wgsl` is the last
window-DISPATCHED kernel, and converting it re-groups its per-workgroup
truncated atomicAdds, so it is a re-baselining change that belongs with B5
proper rather than an inert stage.

### B5-4 — DONE (2026-09-15). B5's PRECISION ARGUMENT DOES NOT SURVIVE MEASUREMENT.

B5's goal is the conversion count -- one coordinate convention, kernels that
carry no window bookkeeping. Alongside it this document lists **two things
[that] come free**, and the first is **bounded precision**: "the body's position
is wrapped into `[0, W)` every step, so its ULP is fixed at `W * 2^-24` forever
by construction". That freebie is now measured, and **it is not a property of
the convention at all.** It is a property of `TOTAL_WRAP_SCREENS`, which is one
constant. This retires a side-claim; it does not touch the goal.

**THE INSTRUMENT.** `?wrapScreens=N` on `index-reentry.html` (same shape as
B7's `?kEps=` ladder), threaded to BOTH `physics.wgsl`'s override and
`card-total.mjs`'s unwrapper -- they are two halves of one convention and a
mismatch does not fail loudly, it silently re-reads every wrap as motion.
Default 16 is unchanged and the page is bit-IDENTICAL with the flag absent.

**WHY THIS PAGE.** `index-reentry.html` is KINEMATIC: `vy` is set to exactly
`VY_FIXED` every step, so after n steps the body's true displacement is exactly
`n * f32(VY)` -- an exact reference, in closed form, with no solver in the way.
It is also the only dense page whose body MOVES, which is what the pinned
cylinder cannot test (`off = 0` there makes the two conventions trivially
agree -- the same blind spot B5-0 recorded about `dense-reference`).

**THE LADDER.** Body position error vs. that exact trajectory, 262144 steps:

    leg                          err (cells)   vs shipped
    window=1  wrapScreens=16      8.541e+0        1.0x      <- shipped
    window=1  wrapScreens=4       1.413e+0        6.0x
    window=1  wrapScreens=2       1.162e+0        7.3x
    window=1  wrapScreens=1       7.114e-1       12.0x
    window=0  (B5)                7.176e-1       11.9x

**`window=1 wrapScreens=1` AND `window=0` ARE THE SAME NUMBER** (0.9% apart).
The buffer convention holds the body in `[0, H)`; `wrapScreens=1` holds the
accumulator in `[0, H)`. Same magnitude, same ULP, same answer. B5 buys nothing
here that one constant does not already buy.

**AND THERE IS NO FIXED ULP, under either convention.** The plan's wording says
the error is pinned "forever by construction". It is not: the REPRESENTATION's
ULP is fixed, but the accumulated drift still grows roughly linearly in the
step count, because every step rounds. Measured on the shipped convention:

    step       window=1      window=0
    4096       2.609e-3      2.600e-2
    16384      5.386e-2      5.386e-2
    65536      2.548e-1      1.933e-1
    262144     8.541e+0      7.176e-1
    524288     1.868e+1      1.435e+0

**AND THE BUFFER CONVENTION IS TEN TIMES WORSE FOR THE FIRST ~7000 STEPS.**
`cy` starts at `H/2` and is ~128 from step zero, while `y_total` starts at 0
and is small for a long time -- so the shipped convention has the smaller
magnitude, and therefore the finer ULP, until `y_total` grows past `H/2`. The
crossover sits between 4096 and 16384 steps, where `H/2 / VY = 6919` predicts
it. A short run is not a small version of a long one here; the ordering
reverses.

(The 16384 row reading identically for both is a coincidence of the crossover
-- the two accumulations are independent, and they differ at every other
sample. Single samples of an accumulating rounding process, so read the decade,
not the digit.)

**WHAT THIS LEAVES B5 WITH.** The conversion-count argument, which is real and
already largely banked (B5-1 collapsed nine hand-written sites into two
functions; B5-2 split the body's frame from the window's). What is NOT
supported is flipping the default FOR PRECISION: that is a re-baselining change
across every page, and the same 12x is available from `TOTAL_WRAP_SCREENS`
without touching the convention. **Both are re-baselining and neither is done
here** -- this stage is inert by construction, and the choice belongs to
whoever owns the published numbers.

**THE CHEAPER CANDIDATE, for whoever takes that decision.** `card-total.mjs`
picked 16 for unwrap headroom, not for precision: "far more than any single
readback can cross (a frame moves the card a few lattice units at most,
against a half-wrap of 8 domain heights)". At `wrapScreens=1` the half-wrap is
still 128 cells against that same few-lattice-unit frame, so the headroom
argument does not obviously require 16 -- but it has not been tested against a
live readback cadence, only against this harness's, and that is the check it
would need first.

### B5-5 — DONE (2026-09-15). The kernel that declared the flag and ignored it.

**THE DEFECT.** Under `?window=0` the dense force kernel computed force and
torque against a body displaced by exactly `(off_x, off_y)` -- however far the
window had panned. Measured on `index-reentry.html`: at `off_y = 75`, `fy` read
-7.102e+0 against a correct -3.140e-2, a factor of **226**. At `off_y = 1` the
two conventions agreed to 0.7%, and at `off_y = 0` exactly.

**THE CAUSE.** `lbm_force.wgsl` built its sample point straight from the thread
index, which is a WINDOW cell:

    let p = vec2<f32>(f32(x), f32(y));   // a WINDOW position
    let phi = get_phi(p, state);         // compared against state.cx/cy
    var rx = p.x - state.cx;             // ...and the lever arm, so torque too

Correct under the shipped convention, where `state.cx` IS a window coordinate.
Wrong under `?window=0`, where it is a buffer coordinate. Every other
body-touching kernel went through the `bodyFrame*` accessors; this one never
did.

**AND IT WAS SILENT, which is the transferable part.** The file `@include`s
`common_geometry.wgsl`, so it DECLARED `WINDOW_BODY` and both dense pages duly
supplied it -- it simply never read it. Supplying an override a shader does not
declare is a hard pipeline error; supplying one it declares and IGNORES is
invisible. That is a different failure class from the one this project already
guards against.

**WHY NO GATE CAUGHT IT.** Every config ever pointed at `?window=0` has
`off == 0`. B5-3 measured the two conventions agreeing at two different
placements and called it the strong form; it was not -- both legs are a PINNED
cylinder, so window and buffer coincide and the frame separation was never
exercised. Same blind spot B5-0 recorded about `dense-reference`, one layer up.
**The check that generalises is the audit, not the config:** grep every shader
for raw `off_x`/`off_y` arithmetic. Done before B5-6 flipped anything, it found
this was the only physics kernel doing it -- everything else was the render (a
view by definition) or `physics.wgsl`/`amr_physics.wgsl` WRITING `off`.

**THE FIX, and why it was inert.** The sample point now goes via the buffer
cell the thread already computes: `bodyFrameCell(vec2<u32>(bx, by), state)`.
Under `WINDOW_BODY=1` that is `bufferToWindowCell((x + off) % W) = x` exactly,
in integer arithmetic, so the shipped path stayed bit-identical INCLUDING where
`off != 0` -- verified on `index-reentry.html` at 4096 steps with `off_y = 75`,
field relL2 0 and the reported force identical to the digit. Afterwards the two
conventions agreed to ~0.03% at `off_y = 75` (the residue is B5-4's
inter-convention position drift) and exactly at `off_y = 1`.

### B5-6 — DONE (2026-09-15). The buffer convention is the default, and the window convention is gone.

The flip and the cleanup, in one commit because they are one change: a flip
that left the HOST still modelling the body in window coordinates would have
had the coverage gate scoring the kernels in a frame they no longer use.

**WHAT MOVED, AND WHAT DID NOT.** Everything with a PINNED body is bit-
identical, because `off = 0` makes the two conventions the same map -- every
Cd/St config to the digit (`dense-reference` 1.951/0.1260, `amr-N2-diffuse`
1.643/0.1478, `amr-N2-bounceback` 1.321/0.1617, `amr-N3-diffuse` 1.425/0.1551,
`amr-N3-bounceback` 1.351/0.1616), every channel and TGV gate, all seven
invariant gates on every AMR config, all three refusal configs, boot smoke on
all seven pages.

The re-baselining is confined to MOVING bodies, and it is small:
`index-reentry.html` at 4096 steps differs by relL2 **2.1e-3** in ux, 6.1e-3 in
vorticity. That is rounding, not a frame error -- the two conventions integrate
the position differently (B5-4), so the body sits ~0.026 cells apart by that
step. A frame error is O(1), as B5-5's 226x force bug was. Control: the new
path is exactly deterministic (same build twice, relL2 0), so the difference is
reproducible rather than noise.

**THE AUDIT THAT PRECEDED IT.** B5-5 found one kernel silently ignoring the
convention, so before flipping anything, every shader was grepped for raw
`off_x`/`off_y` arithmetic. `lbm_force.wgsl` was the only physics kernel doing
it; everything else was either the render (a view by definition, correctly
window-dispatched) or `physics.wgsl`/`amr_physics.wgsl` WRITING `off`. That is
the check to repeat before trusting any future frame change -- the accessors
being available does not mean they are used.

**WHAT WENT.** `override WINDOW_BODY` and its four frame accessors
(`bodyFrameCell`, `bodyFrame`, `bodyFrameBare`) collapse: 13 call sites across
7 shaders now write `vec2<f32>(f32(bx), f32(by))` inline, because the body's
frame IS the buffer cell they already hold. `windowToBody` survives as
`windowToBufferPos` for the render, the one caller that legitimately starts in
window space. `INITIAL_CX`/`INITIAL_CY` (B5-3) are deleted from both physics
shaders and all six pages -- the buffer convention honours whatever
`cardInit()` seeded, by construction, so the override that re-pinned the body
every step has nothing left to do. `?window=` goes from both dense pages.
Host-side, `bufferToWindow`/`bufferToWindowLegacy` are retired WITH their test,
which is re-pointed at `bodyFrameL0`/`bodyFrameL0Legacy` so the truncation
lesson stays gated rather than deleted (B3-5's rule).

Net 152 lines removed. **`?upstream=`'s default stays 12** -- that is what
keeps the cylinder at W/2 and the baselines intact (B5-3).

**NO ACCESSOR SURVIVES AS AN IDENTITY.** `bodyFrameCell` would now be
`vec2<f32>(c)`, and a function named after a frame that does nothing is exactly
what let `lbm_force.wgsl` look converted while it was not. The host keeps its
named identity (`bodyFrameL0`) on purpose, because there it is what the
coverage gate is scoring and the test pins it; the shaders do not, because
there it would be the camouflage.

**AND `card-total.mjs` LOST HALF ITS REASON TO EXIST.** Its header argued the
accumulator's ULP mattered for the trail AND for "THE CARD ITSELF, which
matters more", since the body's sub-cell position was `frac(y_total)`. The body
no longer reads the accumulator at all: what `x_total` still feeds the
simulation is `off`, and only through `floor()`, an integer the ULP does not
reach. `TOTAL_WRAP_SCREENS` now buys trail/CSV resolution, not physics. Both
comments are corrected in place rather than left to rot.

### The gate hole this uncovered, which is not about B5

A scripted edit mangled an object literal in `main.js`; `make js` reported
"57 module(s) parse"; `index.html` then wedged at "initializing..." with an
uncaught SyntaxError, caught only by the GPU boot smoke.

**`node --check FILE` PARSES AS A SCRIPT AND SILENTLY EXITS 0 ON ANY FILE THAT
USES ESM `import`.** Measured on node v26.1.0:

    printf 'const a = {p: 1\nb();\n'                            > s.js  -> exit 1  caught
    printf 'import x from "./y.mjs";\nconst a = {p: 1\nb();\n' > m.js  -> exit 0  MISSED

Every page entry point opens with `import`, so `make js` -- CLAUDE.md's first
gate -- had never checked a single one of them, and had not since they became
modules. The fix is one word: pipe the file to stdin under
`--input-type=module`. Applied to EVERY file, not just the ones that look like
modules, because CommonJS is valid module syntax too (`require(...)` is a call,
`module.exports = x` an assignment) -- so it widens coverage rather than
trading one blind spot for another.

**And the fixed gate was made to fail before it was trusted:** with the brace
removed again `make js` exits 2 pointing at the line, and 0 once restored. A
gate that has not been shown to fire is the thing this project keeps finding.

### B5-7 — DONE (2026-09-15). The last window-dispatched kernel.

B5-5 fixed `lbm_force.wgsl`'s FRAME; its DISPATCH stayed in window coordinates,
converting to buffer at every load and back out again for every neighbour --
composing a shift with its own inverse once per direction per cell, exactly
what B5-0 removed from `lbm_step.wgsl`:

    window:  wx_src = (x - e) mod W,  bx_src = (wx_src + off) mod W
    buffer:  bx_src = (bx - e) mod W          with bx = (x + off) mod W

The kernel touches no sponge and no walls, so with the body buffer-native it
needs no window coordinate at all: `state.off_x`/`off_y` are gone from the file.

**THE REASON IS THE AUDIT, NOT THE CODE SHAPE.** "Only the render does raw
`off_x`/`off_y` arithmetic" is the check that finds a kernel silently stuck in
the wrong frame -- the thing no gate could see in B5-5, because every config
ever pointed at the flag had `off == 0`. That invariant was false by exactly
this one file, and an invariant with one remembered exception is much weaker
than one with none. It now holds with no exceptions.

**THE COST, PREDICTED THEN MEASURED.** The reduction truncates PER WORKGROUP:
each 8x8 group sums 64 contributions and `atomicAdd`s one i32 at FSCALE = 1e7.
Re-tiling changes which cells share a workgroup, so the partials truncate
differently. It is inert exactly when `off ≡ 0 (mod 8)`, because then a window
tile maps onto a buffer tile and the PARTITION is unchanged. So:

    index-cylinder.html   pinned, off = 0   bit-identical (dense-reference 1.951/0.1260)
    index-reentry.html    prescribed        FIELD bit-identical; reported force moves
    index.html            free body         trajectory diverges (no gate but boot smoke)

Measured on `index-reentry.html` at 4096 steps, where `off_y = 75` and
75 mod 8 = 3, so the partition genuinely differs:

    fx   -1.7560500652e-02 -> -1.7560500652e-02   unchanged
    fy   -3.1409200281e-02 -> -3.1409598887e-02   -4.0e-07  (1.3e-05 rel)
    tz    2.6860749722e-01 ->  2.6860710979e-01   -3.9e-07  (1.4e-06 rel)

~4 units at FSCALE across 1024 workgroups -- the predicted mechanism at the
predicted magnitude. The FIELD is bit-identical (relL2 0 on ux/uy/rho/omega)
because this page is KINEMATIC: the force is reported and discarded, and this
kernel never writes `f`. Channel and TGV are untouched by construction -- they
have no body and never dispatch it.

### B9 — DONE (2026-09-14)

`lattice-2d.mjs` + `tools/gen-lattice-2d.js` + `tools/test-lattice-2d.js`, and
the eleven hand-typed copies of the D2Q9 basis are now one derivation.

**THE 3D FIX DOES NOT TRANSFER, AND THAT IS THE FINDING.** `lattice-3d.mjs`
nudges the weights by whole ulps so they sum to exactly 1, on the argument
that the resulting moment error is harmless because it does not compound. For
D2Q9 that argument is false: the moment error compounds too, in the
**velocity**. Through one BGK collision,

```
rho -> rho (1 + omega (S - 1))                 S    = sum_i w_i
M   -> M   (1 + omega (3 Sig2 - 1))            Sig2 = sum_i w_i e_ix^2
u   -> u   (1 + omega (3 Sig2 - S))            <- what every gate measures
```

and `3 Sig2 - S` is `2 w1 + 8 w2 - w0` for D2Q9. Per step, in units of omega:

```
shipped (0.02777778f)         mass  1.490e-8    velocity  1.490e-8
correctly rounded fractions   mass  7.451e-9    velocity  0 EXACTLY
the 3D-style tweak [0,0,-1]   mass  0           velocity -1.490e-8
```

The plain fractions satisfy the velocity identity **to the bit**, and no tweak
in `|k| <= 4` zeroes both. So 2D ships the plain fractions at full precision:
half the mass injection AND no velocity drift. Mass drift with exact velocity
is the benign half — a uniform rho rise in a periodic box has zero gradient
and so no dynamics.

**Measured, before and after, on the channel gates** (every case improved or
held; none worsened): couette −26% to −53%, poiseuille −0.5% to −10%, and the
worst case converged in **13312 steps against 27648** — a drift growing with
run length had been defeating the steady-state detector. TGV is mixed and
reported as such: the two longest dense cases improve 17% and 21%, shorter
ones move within noise. Cylinder `dense-reference` Cd 1.950 → 1.951, St 0.1258
→ 0.1260, both already outside their bands and still are — which is
CLAUDE.md's own point about Cd/St being the wrong instrument here, confirmed.

**The interface instrument now reads what it should:** its mass floor halved
(1.861e-8 → 9.377e-9 per cell per step, predicted 9.313e-9) while the seam's
ABSOLUTE contribution is unchanged (1.66e-9 → 1.64e-9). The interface stands
at 14.4% of a floor half the size, instead of 7.3%. **B6's mass half is
unblocked.**

**For the 3D side, not acted on here:** the same arithmetic says D3Q27's tweak
`[-1,1,-1,0]` trades an exactly zero velocity drift for an exactly zero mass
drift (plain: mass 7.451e-9, velocity 0; tweaked: mass 0, velocity 7.451e-9).
D3Q19's `[1,-4,2]` is a genuine improvement — mass 1.490e-8 → 0 with the
velocity drift unchanged in magnitude — so the decision there was right for
the default velocity set and questionable for the `?q=27` variant.

### B6 — the coarse/fine interface

### B6-0 — THE BEFORE-NUMBER, AND WHY IT READ AS FIXED (2026-09-23)

Re-measured before starting, and the first reading said there was nothing to
fix: the `half` rung's momentum drift at 512 steps was **7.8e-5, 0.75x the `all`
control**, against B1's recorded 1.302e-2. It was bisected rather than
believed, and **nothing was fixed -- the test geometry had become symmetric.**

```
build / mode                                   half rung @512   geometry
a230ace  (B1)                                  1.302e-2         jagged band (holes, stub)
ae90867  dense coupling                        1.302e-2         same, bit-identical
ae90867  ?rootpool=1  + OLD manager            1.302e-2         same, bit-identical
ae90867  ?rootpool=1  + quad manager (U5-4)    7.836e-5         FLAT band, rows 0-7
HEAD                                           7.836e-5         same, bit-identical
```

The root-pool COUPLING reproduces the dense leak bit for bit; the drop is the
MANAGER. U5-4 allocates level 1 in quads, so the pool-capped `half` rung became
a flat full-width band of exactly half the domain -- two seams half a
Taylor-Green period apart, where the flow is negated and the seams face
opposite ways, so their leaks cancel in the GLOBAL sum the instrument reads.
Breaking the symmetry with the cap, same build (HEAD), flat seams throughout:

```
cap   band        seams          max|d mom| @512      growth   mass ratio
 64   rows 0-3    y=0, y=32      6.608e-3 = 184x      t^0.91   0.803
 96   rows 0-5    y=0, y=48      3.198e-3 =  89x      t^0.88   0.867
128   rows 0-7    y=0, y=64      7.836e-5 = 2.2x      t^1.72   0.927
```

**So the leak is real, linear in t (a per-step source), and present on FLAT
seams** -- it is not a convex-corner effect, which matters for B6: the corner
is not where it lives. `tools/analyze-amr-interface.js` now defaults to
`--halfCap=96`; do not choose a cap whose refined fraction is one half.

**Two things to carry.** A global conservation total is blind to any geometry
with a symmetry that negates the per-seam flux; a declared asymmetric seam (or
a per-seam budget) is the instrument, not a lucky cap. And this bisect was
first run on a debug Chrome whose GPU process had crashed, on SwiftShader --
two of its legs came out wrong (a spurious 8.49e-3, and a false "does not
boot"). Every row above is from a health-checked RTX run
(`tools/check-browser.js`, now in every tool's teardown).

### B6-1 — the design, before the code (2026-09-23)

3D's M4.1b + M4.1c-i ported to D2Q9, behind `?interface=explode` with
`interp` still the default (3D's own staging). Everything below is inherited
from `common_d3_amr_explode.wgsl` / `common_d3_amr_coalesce.wgsl` except where
marked 2D.

- **Explode** (`shaders/amr_explode.wgsl`): for each child ring cell whose
  parent cell `v` is UNCOVERED, direction `i` gets `f_i(v)` -- a plain copy,
  no interpolation, no rescale -- if `v + e_i` is covered; otherwise the
  IN-orphan rule (`f_i(v)` iff the child's one-fine-step destination is
  covered AND in this tile, else 0). **2D: parent cells are resolved through
  their OWNING tile** (parent `blockSlot`), never through the parent tile's
  ring, which is not a distribution under this scheme.
- **Coalesce** (`shaders/amr_coalesce.wgsl`): dispatched over parent-tile
  interiors. A covered `c` gets, for each `i` whose target `c + e_i` is
  uncovered, the MEAN of the four ring cells covering the target (1/4 = the
  2D volume ratio) written into its own slot of the parent's TIME-t buffer,
  so the unmodified parent step pulls it. An uncovered `c` next to a covered
  cell runs the OUT-orphan pass (half-step exits, found by the owner of
  `p - e_i`) net of the IN-orphans explode injected, `f_i(c) += net/4`.
  Both orphans land together: 3D measured the pair as one defect.
- **The ring advects and does not collide** (`COLLIDE_RING = 0` on the
  explode path; the interior still collides). No Dupuis-Chopard factor on
  this path -- `tau_f = 2 tau_c - 1/2` is Chen's `omega_f` at n = 2.
- **Order** (`makeScheduler`, `explode`): per parent substep
  `interp -> explode -> child cycle -> coalesce -> parent step -> average`.
  **2D DIFFERENCE, deliberate: `average` is KEPT, after the parent step.**
  Nothing in the dynamics reads it (an uncovered cell pulls only coalesced
  slots from a covered one; explode reads only uncovered cells), but it keeps
  every covered cell a true restriction between macro-steps, so: no drain
  pass is needed (sec 2's argument still holds), and the conservation
  instrument, snapshots, render and criterion keep their meaning. `interp`
  is kept too, BEFORE explode, as the source of the same-level ring values a
  new tile's bilinear init reads; explode then overwrites the coarse-seam
  ring cells. Both are cost, not physics, and are B6-perf's to remove.

**Gate:** `analyze-amr-interface.js --extra=interface=explode` -- cap 96 and
cap 64 fall from 89x/184x to the no-interface floor in momentum and to ratio
1.000 in mass; `none` and `all` bit-identical to `interp` (no coarse seam to
transfer across). Then field error against the no-interface control, then
`validate-all.js` with the default untouched.


**Do not start here.** Start at B0b. 2D's interface today is interp (ring
ghosts) + average (restriction), with no flux correction: coarse cells at the
seam stream from their own coarse neighbours while the fine tile streams from
its ring, so **mass and momentum are not conserved across the seam**, and
nothing in the 2D suite measures either. That is the same state 3D was in
before M4.

The 3D answer is **Chen et al. 2006 explode/coalesce, delivered by
DESTINATION** — and every one of its hard-won details is 2D-applicable:

- **Conservation and consistency are separate requirements**, and a geometry
  ladder cannot tell them apart. 3D's M4.1b ended exactly conservative and did
  *not* preserve a uniform flow; the conservation gate went green over a
  defect worth 8x the field error.
- **The half-step EXIT and the half-step ENTRY are one defect seen from both
  sides.** They land on the same (cell, direction) slot, and in a uniform flow
  their counts are equal so the net write is zero. **Never add one without the
  other.**
- **The orphan pass.** A ring cell's content is claimed by `coarse(p) - e_i`,
  and on a diagonal direction at a convex corner that cell is outside the
  refined region, so nobody claims it and the mass is dropped. 3D measured
  -1.2e-2 per edge coarse cell per macro-step against a predicted 1.4e-2.
- **Do not "simplify" it into gathering by ORIGIN.** Also exactly
  conservative, smaller, and measured **4.6x worse in the field**.
- **Refluxing is superseded. Do not extend it.** In 3D the reason is that the
  corner does not tile: fine channels tile a coarse FACE exactly and a coarse
  CORNER not at all, because D3Q19 has no (1,1,±1). **That counting argument
  does NOT transfer to 2D unexamined** — D2Q9 *does* carry the (±1,±1)
  diagonals, so the 2D corner has to be re-derived, not inherited. Do not
  assume it transfers and do not assume it doesn't: build the 2D analogue of
  `?refine=slab` (a corner-free seam) as the control that separates a
  correction bug from a corner, and let it answer. What *is* inherited without
  re-derivation is the conclusion — explode/coalesce puts mass exactly one
  place by construction, so the corner question never has to be settled to
  ship it.
- **A fully-refined run cannot see an interface bug.** With everything
  refined, the restriction's rescaled coarse `f` is consumed by a coarse step
  whose result is discarded. 2D needs a pinned partially-refined control
  config, the way `amr-box-RB4` is pinned to `interp` in 3D — and it must stay
  pinned, because a control that tracks the default is not a control.
- **AND NEITHER CAN A RUN THAT REFINES NOTHING, WHICH IS WHAT THE SUITE'S
  ANALYTIC AMR CONFIGS ACTUALLY DO.** B1's finding #1: all four of
  `channel-poiseuille-amr-N2`, `channel-couette-amr-N2`, `tgv-amr-N2` and
  `tgv-amr-N3` hold ZERO active tiles at every level, by their pages' own
  deliberate defaults. They were bit-identical across B1's flip for that
  reason, not because the change was small. **Do not reach for them as B6's
  gate.** Making them useful means turning refinement on there, which
  `main-channel-amr.js`'s header says is currently unsafe for Couette — so
  that is its own piece of work, and it is a prerequisite if B6 wants an
  analytic answer rather than only a conservation one.
- **A stage that measures as a no-op may be downstream of something bigger.**
  3D's linear explosion measured as noise against an interface that still had
  an O(1) edge inconsistency; with that fixed the same change was worth a
  factor of two.

**Cost.** This is the one large item. It needs a compaction pass (active
blocks -> a dispatch list), which 2D wants anyway, and it will need a **drain**
pass that 2D does not currently need (see §2) — explode/coalesce is an
interface-layer transfer and does not restrict a tile's interior, so the
"average already did it" argument stops holding the moment `average` stops
running every step for every level. Check that before, not after.

**Gate:** B0b's mass and momentum drift, on the TGV harness, at every rung of
a 2D geometry ladder; then the field error tracking a no-interface control;
then `validate-amr-vs-dense.js` and `validate-divergence.js` (whose `edge`
column exists for exactly this class) showing the interface error falling
toward the `fullrefine` noise floor.

### B6-1 — BUILT (2026-09-23). Conservative; 3.5x better in the field; not yet consistent.

Opt-in, `?interface=explode`, on all five AMR pages. Every number below is from
a health-checked RTX run (`tools/check-browser.js`); TGV N=128, tau=0.8.

**CONSERVATION -- MET.** Momentum, max|d mom| against the no-seam controls in
the same run:

```
seam rung       interp (same build)     explode             control
cap 64  @512    6.61e-3 = 184x          3.80e-5 = 1.06x     none 3.60e-5
cap 96  @512    3.20e-3 =  89x          5.73e-5 = 0.55x     all  1.05e-4
cap 96  @256    1.77e-3                 1.60e-5             none 6.3e-5
cap 96  @4096   6.78e-3                 1.46e-3             all  2.54e-3
```

Mass ratio against the lattice floor 0.997 / 1.000 / 0.996 (interp 0.80-0.98).
`none` and `all` are BIT-IDENTICAL to interp, as they must be: no coarse seam,
nothing to transfer. The late-time explode drift sits under `all`, which has no
seam at all -- it is the f32 rounding that scales with fine stepping.

**THE DEFAULT PATH IS BYTE-IDENTICAL**, measured: `measure-determinism.js`
detslots=1 fingerprints before and after, same hashes on the dev page
(`dec9eb05032bcced`, `cc0dd492cd6ced2c`) and the cylinder page
(`57f916c1cf590e12`, `72718a827cbccd57`) at levels 2 and 3.

**A RING CELL IS NOT A STATE, AND THE CRITERION WAS READING ONE.** The first
explode run of `amr-dev-invariants` starved the pool (2669 refines refused at
step 5120) and then broke geometry coverage. The criterion's +-1 vorticity
stencil at a tile's edge reads the RING's velocity; on interp that is a
collided, interpolated state, on explode it is an inbox/outbox, and its curl is
noise that drove level-2 refinement. `RING_FREE_TAPS` resolves those taps into
the same-level neighbour tile (one-sided at a coarse seam) on the explode path;
with it, all seven invariants are green through 8192 steps. This was
uniform-levels 2.5's warning, met in practice. `amr_render.wgsl`'s stencil is
the other listed consumer and is visual-only; not yet audited.

**CONSISTENCY -- NOT MET, AND THE INSTRUMENT SAYS WHY.** The field channel
(`analyze-amr-interface.js`, new: L2rel against the exact Taylor-Green
solution, bucketed by distance to the seam, and split into the mode's
AMPLITUDE and the residual SHAPE):

```
@2048 (~1 t_d)       whole     d0-1      far       amp        shape
none (no seam)       3.07e-4   -         3.07e-4   +1.9e-4    2.41e-4
all  (no seam)       2.69e-4   -         2.69e-4   -1.1e-4    2.45e-4
half interp          8.34e-3   9.11e-3   8.03e-3   -7.70e-3   3.21e-3
half explode         2.42e-3   2.55e-3   2.35e-3   -2.14e-3   1.11e-3
```

The seam error is almost entirely AMPLITUDE -- extra dissipation, growing
linearly (explode ~-1e-6 of the mode per step, ~0.2% of the analytic decay
rate; interp ~4x that) -- and it is NOT localized (d0-1 = far), because the
mode is global and a seam that dissipates lowers it everywhere. That is the
signature of a first-order reconstruction at the seam, which the UNIFORM
explosion is. **B6-2 is the linear explosion** (3D's M4.1c-ii, worth 2x there),
gated on `amp` falling toward the controls.

**Found on the way, and fixed separately:** `browser-lifecycle.js`'s
`ensureServer` had been broken by the health-check commit (f9e170d), and the
explode scheduler read the dev page's `ghostCopy` callback before the state it
closes over existed (caught by the new health check on its first run).

### B6-2 — DONE (2026-09-23). The linear explosion: 12x interp in the field, 2.3x the controls.

3D's M4.1c-ii in `amr_explode.wgsl` (`EXPLODE_LINEAR`, default on the explode
path, `?explin=0` for uniform): central difference on axes with two real,
uncovered parent neighbours, projected orthogonal to `e_i`. Same build, cap 96:

```
@2048                  whole     amp        shape     conservation
controls none / all    3.1e-4    +1.9/-1.1e-4  2.4e-4
interp                 8.34e-3   -7.70e-3   3.21e-3   89x floor
explode uniform        2.42e-3   -2.14e-3   1.11e-3   0.55x, mass 1.000
explode LINEAR         7.14e-4   -5.3e-4    4.7e-4    0.50x, mass 0.998
```

The seam's dissipation fell ~5x and the error stopped growing (6.7e-4 at 512,
7.1e-4 at 2048). Conservation did not move, as the construction says it
cannot. `?explin=0` reproduced B6-1's uniform numbers to every digit.

### B6-3 — DONE (2026-09-23). The force read the ring too; the published table on explode.

**THE FIRST EXPLODE CYLINDER TABLE LOOKED LIKE A PHYSICS REGRESSION AND WAS A
READOUT.** Both diffuse configs rose +0.12 in Cd (N3-diffuse 1.473 -> 1.594, a
new red cell) while bounce-back barely moved. The referee was the SAME config
FULLY REFINED -- no coarse seam anywhere -- and it still split, interp 1.655
against explode 1.778. So not the interface: the diffuse branch of
`amr_force1.wgsl` re-does the streaming gather for rho and u*, and on a ringed
level it clamped its sources into the RING, which the step never reads (it goes
through the neighbour tile, DIRECT_GHOST). Explode's ring is uncollided, so the
force was built from a gather the fluid never felt; bounce-back's link sum
reads only the cell's own data, which is why it did not move.
`RING_FREE_FORCE` resolves the gather exactly as the step does. Fully refined,
interp and explode then agree TO THE DIGIT (1.655 / 0.1497 both), and interp's
own number did not move, i.e. its ring happened to hold the right thing.

**THE PUBLISHED TABLE ON `?interface=explode`** (health-checked RTX):

```
config              interp (baseline)   explode            literature
amr-N2-diffuse      1.652 / 0.1484      1.652 / 0.1494     1.35 / 0.165
amr-N2-bounceback   1.356 / 0.1642      1.373 / 0.1665
amr-N3-diffuse      1.473 / 0.1570      1.493 / 0.1603
amr-N3-bounceback   1.365 / 0.1645      1.328 / 0.1680
```

No new red cell; the documented diffuse-band one is unchanged. St moves toward
0.165 on three of four. All invariants green. The default path is unchanged
(cylinder fingerprints `57f916c1cf590e12` / `72718a827cbccd57` before and
after).

**What is left before explode can be the default** -- that call moves the
published surface, so it is the user's, not this plan's:
- `amr_render.wgsl`'s stencil, the last listed ring consumer (visual only).
- `validate-divergence.js` / `validate-amr-vs-dense.js` on explode (the `edge`
  column is built for exactly this class).
- Cost: explode adds two passes and keeps interp and average (B6-perf).
- The card page's moving body and the reentry pages have booted and passed
  invariants on explode, but nothing has scored their physics.

### B6-4 — DONE (2026-09-23). The bright corner cells were the RENDER reading the ring.

Reported by eye on the cylinder page under explode: single very bright cells at
the corners of refinement boundaries. Two candidates look identical on screen
-- the renderer reading a ring cell, or a real spike where the orphan passes
act -- so it was MEASURED IN THE FLUID FIRST, as CLAUDE.md's M6.4 lesson says
(never an image statistic): the tile INTERIORS, reconstructed at step 8192,
have their sharpest vorticity outliers on the body on both paths (8.5e-4 on
each, 32-46 fine cells from any corner). So it was the render: its bilinear
blend plus the curl's +-1 reach two cells past the interior, into a ring that
at a convex corner holds mostly zeroed/clamped directions under explode.

`RING_FREE_RENDER` in `amr_render.wgsl` resolves those taps through the
same-level neighbour tile (own edge at a coarse seam), default on the explode
path, `?ringfreerender=` to A/B. Magnified crops of all 8 convex corners, same
build (`plans/img/b6-4-corner-cells.png`): interp clean; explode with the ring
read shows the bright cells at every corner; explode ring-free matches interp.

**That makes four ring consumers on this path, all found by a failing check,
never by reading**: the criterion (pool starvation), the diffuse force (Cd
+0.12 with no seam), the render (corner cells), and -- by construction -- the
step itself, which never reads a ring where a neighbour exists. uniform-levels
2.5's "a ring cell is not a state" is the rule all four broke.

**Found in passing, not yet explained:** the explode interior has DENSITY
outliers the interp one does not -- 1.0e-4 (5.4x its p99, against interp's
2.1x) in 2x2 fine clusters, i.e. single COARSE cells, symmetric about the wake
axis at x ~ 575, y ~ 480/543, ~32 fine cells from any corner: on a straight
downstream seam. Too small for the vorticity picture; a real seam signature.

### B7 — DONE (2026-09-14)

`?kEps=` threads one `K_EPS` override through all nine chi sites (the render
pass included — its chi draws the body outline, and a ladder whose picture
still showed the old band would misrepresent the thing being swept).
`?kEpsPool=` survives and now defaults to `K_EPS` rather than a second
hardcoded 1.5. Default unchanged and byte-identical, asserted on the one
deterministic config in the suite.

**The ladder**, `dense-reference`, Re=100, D=21.33:

```
kEps     Cd      St        dCd      ratio
1.5     1.951   0.1260
0.75    1.617   0.1485   -0.334
0.375   1.453   0.1571   -0.164    2.04
0.1875  1.366   0.1593   -0.087    1.89
bounce-back (sharp, same body)  Cd 1.327  St 0.1605
                                literature 1.35+/-0.15, 0.165+/-0.015
```

**Three things, in order of how much they change what is written above.**

1. **FIRST order, not second. 3D's convergence rate does not transfer.** This
   section proposed inheriting "the diffuse body converges at SECOND order in
   the band width" from 3D's D3Q19 sphere (ratio 3.85 ~ 4). 2D measures 2.04
   and 1.89, i.e. p ~ 1. The neighbouring B6 bullet already says not to
   inherit D3Q19's corner-counting argument without re-deriving it; **the same
   caution was needed here and was not stated**, which is the generalizable
   part — a 3D *rate* is as unportable as a 3D *counting argument*, and both
   look like physics rather than geometry when you read them.

2. **The gate's first branch is the answer: the red cells are a quantified
   band error, not a second defect.** The shipped band inflates Cd by ~47% and
   depresses St by 21%, both first-order, and both red cells close when it is
   narrowed (`amr-N2-diffuse` at `?kEps=0.375`: Cd 1.380 / St 0.1643, from
   failing both to passing both). CLAUDE.md now carries the ladder instead of
   the "open issue" framing.

3. **Richardson is the weakest number here, and the write-up should not lean
   on it.** Extrapolating the last pair gives Cd -> 1.279, but at kEps=0.1875
   the tanh transition is ~0.4 cells — NARROWER THAN THE MESH — so the smooth
   power-law it assumes is already contaminated. The tell is that the sign of
   the disagreement with the sharp answer flips relative to 3D (there 1.478
   against bounce-back's 1.3951, above; here 1.279 against 1.327, below).
   **The sharp-interface run on the same body is the better anchor for "zero
   band"** and needs no extrapolation at all — 2D has one and 3D's write-up
   used it only as a check.

**The default stays 1.5, deliberately.** Every number here is a PINNED
cylinder; the shipped page is a falling card. Narrowing the band moves toward
bounce-back's sharpness, which 3D's D4 measured as worse for a MOVING body
(9x on frame consistency, 600x on force noise). Adopting a narrower default is
a separate change gated by the moving-card harnesses — making it on the
strength of a pinned-body sweep is exactly the mistake this section's own
closing note warns about, one parameter over.

### B7 — the diffuse band, and the standing Cd red cells

`dense-reference` and `amr-N2-diffuse` fail at Re=100 (Cd 1.950 and 1.620
against 1.35+/-0.15), and CLAUDE.md records the diagnosis as
"diffuse-interface width — the chi band is ~+/-4 cells regardless of
resolution, so the effective radius exceeds the nominal one and Cd converges
from above."

3D measured the same thing properly and the result is directly usable: **the
diffuse body converges at SECOND order in the band width, to the number
bounce-back gives directly.** D=12, Re=100, pinned, band 1.5 / 0.75 / 0.375
coarse cells: Cd 5.080 / 2.413 / 1.721, a difference ratio of 3.85 against 4
for halving; Richardson to zero band gives 1.478 against bounce-back's 1.3951
on the same body. **The instrument is a BAND ladder, not a resolution
ladder** — and 2D's own recorded evidence (1.908 at res=9 -> 1.597 at res=10)
is a resolution ladder, which conflates band width with everything else.

**Blocker:** the band is hardcoded on the dense/L1 path —
`lbm_collide.wgsl:64`, `lbm_step.wgsl:67`, `lbm_force.wgsl:36`,
`amr_step.wgsl:78`, `amr_force.wgsl:51`, all `1.5f`. Only the pool path has a
knob (`?kEpsPool=`, `main-cylinder-amr.js:296`). Thread one `K_EPS` override
through all of them, defaulting to 1.5 so every existing number is
byte-identical.

**Gate:** the ladder itself. Either the extrapolation lands inside the
1.35+/-0.15 band — in which case the red cells are a known, quantified,
second-order band error and the benchmark should say so — or it does not, and
2D has a second defect that the band explanation has been covering.

**Also worth noting, and free:** 3D's D4 measured that for a body that MOVES
at low tau, the diffuse coupling is frame-consistent and noiseless where
bounce-back is neither (9x better on the frame, 600x on the force noise). 2D
chose chi for a completely different reason (a resolution-starved phone). The
3D measurement retroactively justifies it. Do not "upgrade" 2D's falling card
to bounce-back on the strength of 3D's *pinned* sphere numbers.

### B8 — DONE (2026-09-14)

`override SOLID_EQ` in all four step kernels, `?solideq=0` to A/B. The gate is
met in its strongest available form: `dense-reference` is the one
DETERMINISTIC config in the suite, and `?bounceback` against
`?bounceback&solideq=0` reads Cd 1.327 / St 0.1605 on both legs — exact, not
within-a-floor.

**"Latent only" was right about the observables and wrong about the
magnitude.** This section says the fix "cannot move a single 2D number", which
is true and is the gate — but it reads as though there were nothing there to
fix. Measured in the velocity field inside the body, where no observable ever
looks (dense bounce-back, 4096 steps, U0 = 0.04):

```
leg                  max|u| inside    max|u| outside
SOLID_EQ (default)     2.05e-4          5.06e-2
?solideq=0             4.00e-2          5.11e-2
```

**The body's interior currently carries 78% of the free-stream velocity.** The
hazard is real in 2D and was never inherited from 3D's measurement — it is
simply never read. That is worth knowing before someone points `?bounceback=1`
at a moving body and reads the result as a solver bug.

**And the control is the reusable part.** `max|u| OUTSIDE` also differed
between the two legs, by 0.9% — which would have contradicted the
bit-identical Cd above if taken at face value. Running the SAME leg twice gave
a ~0.6% spread, because the page runs live for an indeterminate number of
frames between load and `setLive(false)` and this is an instantaneous field
quantity in a shedding flow. **A new instrument needs its own noise floor
measured before either direction is believed**, exactly as CLAUDE.md
prescribes for AMR Cd — and the first thing this one produced, unchecked, was
a false contradiction with a result that was already exact.

### B8 — `SOLID_EQ` on the 2D bounce-back path

Latent, cheap, and free. Under bounce-back `chi` is 0, so interior cells are
stepped with reflected gathers and nothing damps them — 3D measured max|u|
inside the body at 6x the body's own speed at tau=0.6, and a moving body dead
in 200 steps at tau=0.514. **A pinned body never notices**, because nothing
reads a solid cell: the bounce-back branch reads `f_in[opp[i]]` at the FLUID
cell and the force kernel runs only where `phi >= 0`.

2D's only bounce-back body is pinned, so today this cannot fire — and for the
same reason the fix **cannot move a single 2D number**. Port it anyway
(`?solideq=0` to A/B) so that `?bounceback=1` on a moving 2D body is not a
landmine, and note in the header that it is also the fresh-node refill done
unconditionally because that is cheaper than detecting when to do it.

**Gate:** bit-identical on `dense-reference`, `amr-N2-bounceback`,
`amr-N3-bounceback`. If anything moves, something reads a solid cell and that
is the finding.

### B9 — the f32 lattice weights

**And it is TWICE what CLAUDE.md records, measured while writing this.**
CLAUDE.md says "The 2D lattice (D2Q9, 1 + 7.5e-9) deliberately still has it",
which is the number for `fround(4/9) + 4*fround(1/9) + 4*fround(1/36)`. But the
checked-in WGSL does not hold those values — `shaders/common_lattice.wgsl:9-13`
types eight-digit decimal literals, and `0.02777778f` is **one step above**
`fround(1/36)` by 1.86e-9. Four diagonals, so:

```
actual sum of shaders/common_lattice.wgsl's f32 weights  = 1 + 1.49e-8
the value CLAUDE.md records                              = 1 + 7.45e-9
```

i.e. exactly the D3Q19 figure, not half of it. `0.44444444f` and
`0.11111111f` do round to `fround(4/9)` and `fround(1/9)`, and
`CS2 = 0.33333333f` is exactly `fround(1/3)`, so the diagonal literal is the
whole of the extra half. Every 2D collision therefore injects
`omega * rho * 1.49e-8` of mass per step — a uniform density rise that reads
as a **velocity deficit growing with run length**. In 3D that same magnitude
was 80% of the tau=0.6 duct gate's recorded error. **A gate whose error grows
with the run is measuring a drift.**

**There is a second, smaller instance of the same class:** the host-side `WT`
arrays (`main.js:89`, `main-amr.js:536`, and six more) use *exact fractions*,
so the initial condition the host writes and the equilibrium the shader
computes disagree by 4 x 1.86e-9. Same shape as the packed-`f` host/shader
layout agreement that `tools/test-f-pack.js` already gates — and the right
fix is the same one: derive the numbers once and check the two against each
other.

The 3D fix is `lattice-3d.mjs`'s `WEIGHT_ULP_TWEAK`: nudge each shell's weight
by a whole number of f32 ULPs so the multiplicity-weighted sum is exactly 1,
and assert the tweaked value is still an f32.

2D's weights are typed in ten-plus places (`shaders/common_lattice.wgsl:9`,
`main.js:89`, `main-amr.js:536`, `main-cylinder-amr.js:454`, `main-channel*.js`,
`main-tgv*.js`, `main-reentry.js`, ...). Do the same
thing 3D does: **derive them once and generate the WGSL**, with a
`tools/test-lattice-2d.js` that guards the checked-in file two ways — exact
drift against the generator, and the lattice moment conditions computed from
the tables **parsed back out of the WGSL**. The second half is what fails when
the generator and the file are wrong together.

**This moves every 2D number.** It is the right change and it is the one most
likely to be deferred; if it is deferred, say so explicitly in `CLAUDE.md`
rather than letting it be rediscovered (it already is recorded there — keep it
current).

---

## 5. Ordering

```
B0   host module + mutation-checked tests        ──┐ DONE
B0b  interface instrument (mass/momentum drift)  ──┤ DONE
                                                   │
B3a  JS duplication sweep                        ──┤ LARGELY DONE
                                                   │
B4-1 the coverage gate itself                    ──┤ DONE (it was never
                                                   │  three copies -- it had
                                                   │  to be WRITTEN)
B4-2 the box predicate + the margin slack         ──┤ DONE (one change, not
                                                   │  two -- see B4-2)
B4-3 masking deleted, one force pass              ──┤ DONE
B4-4 refusals, + 3 configs that require them      ──┘ DONE
                                                      B4 COMPLETE. It did
                                                      prove B0's harness --
                                                      every one of its four
                                                      stages was gated by
                                                      something B0/B4-1 built,
                                                      and three real defects
                                                      were found BY those
                                                      gates rather than by
                                                      reading code.
B7   chi band ladder                                   DONE -- red cells are
                                                       quantified band error;
                                                       default unchanged (it
                                                       needs the moving-card
                                                       gates, not this one)
B8   SOLID_EQ                                          DONE -- exact on the
                                                       deterministic leg
B9   lattice weights                                   DONE -- unblocked B6's
                                                       mass half
B2-1 the before-number                             ◄── DONE
B2-2a the GPU closure, proved, unused             ◄── DONE
B2-2b0 get the pool manager off the 16-buffer      ◄── DONE, 16 -> 15
       ceiling                                          (childQuadrant held
                                                        slot % 4)
B2-2b the switch-over, measured, default off      ◄── DONE
B2-2c default flipped                             ◄── DONE
B2-2d delete the ?cascade=0 path (16 -> 14),       ◄── DONE; corner balance
      corner balance GATES                              now gates, and B3's
                                                        manage pair is a
                                                        refactor not a rewrite
B2   cascade21                                     ◄── DONE, via B2-1..B2-2d
                                                        above; the per-pass
                                                        path is deleted, not
                                                        merely superseded
B3-1 step1: one kernel, every level               ◄── DONE, bit-identical
B3-2 average: one body, two parent fragments      ◄── DONE, bit-identical
B3-3 interp: kernel + two parent fragments        ◄── DONE, bit-identical
B3-4 force1: one kernel, every level              ◄── DONE, bit-identical
B3-5 retire originX/originY + the 8th gate         ◄── DONE; manage_pool
                                                        14 -> 10 bindings
B3-6 criterion: shared stencil + reduction only   ◄── DONE; NOT unified,
                                                        and that is measured
B3-7 manage: predicates unify, allocators do not  ◄── DONE. B3 COMPLETE.
B5   window = sponge translation                    ◄── DONE. One convention:
                                                        the body is buffer-
                                                        native, only the view
                                                        converts. Precision
                                                        claim retired (B5-4);
                                                        lbm_force's DISPATCH is
                                                        the one named residual.
B1   post-collision rescale                         ◄── DONE. Seam momentum
                                                        drift 3085x floor ->
                                                        23.3x, one flag, one
                                                        build. tau=1 refused,
                                                        gated both ways. And
                                                        the analytic AMR gates
                                                        turn out to refine
                                                        NOTHING -- read B1's
                                                        finding #1 before
                                                        trusting them for B6.
B6   explode/coalesce                               ◄── needs B0b, B1, B3, B4
                                                        -- B1 is now done, so
                                                        B6 is unblocked and is
                                                        the only stage left
```

B4, B7 and B8 are deliberately first among the real changes: each is small,
each has a gate that says "nothing moved", and together they prove B0's
harness catches what it claims to.

**Do not combine a bit-identical stage with a re-baselining one in a commit.**
B1, B6, B7 and B9 each move the published surface; B2 moves the AMR
configs because the refined region genuinely changes. Everything else must be
provably inert, and the value of that is destroyed if it rides along with
something that isn't.

## 6. Risks

1. **The gate is the weakest part of this.** The default `validate-all.js`
   sweep is *not* all-green on `main` and AMR Cd is only reproducible to
   ~+/-0.001 (the `atomicSub` free-list re-groups the truncated per-workgroup
   atomics). A 4th-digit AMR move is not evidence in either direction. Every
   build-vs-build claim needs a same-build repeat; the dense configs are
   bit-identical and are the strict check.
2. **B3's `manage` pair is the one that can eat the schedule.** 364 + 492
   lines implementing two different balance schemes. It is sequenced last on
   purpose; if B2 slips, drop `manage` from B3 rather than unifying two
   implementations of a rule that is about to be deleted.
3. **B6 is a milestone, not a stage.** 3D took M3 -> M4.1e across several
   attempts including one (refluxing) that had to be abandoned after it
   measured the corner. Budget it that way, and treat B0b's report as the
   thing that decides whether to start.
4. **`main` is the site.** Merging publishes within a minute. Every stage that
   moves a number wants its GPU validation *before* the merge, not before a
   release that does not exist.
5. **Two concurrent forks.** Doing this work on 2D while 3D continues means
   `common_refine.wgsl` (shared verbatim today) and any future shared fragment
   move under both. Prefer sharing *more* fragments, not fewer — that is what
   B3 is for — but land 2D-side changes to shared files on their own.
