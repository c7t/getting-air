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
4. `force1` — same, and it shrinks again under B4.
5. `criterion` — smallest, do it for uniformity.
6. `manage` — **last**, and only after B2 has already deleted the per-pass
   balance tests. Unifying 364+492 lines of two different balance
   implementations is not a refactor, it is a rewrite; B2 makes it a refactor.

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

### B5 — the window is a translation of the sponge

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
B2   cascade21                                     ◄── needs B0
B3-1 step1: one kernel, every level               ◄── DONE, bit-identical
B3-2 average: one body, two parent fragments      ◄── DONE, bit-identical
B3-3 interp: kernel + two parent fragments        ◄── DONE, bit-identical
B3   the rest (force1, criterion, manage LAST)     ◄── needs B2 and B3a
B5   window = sponge translation                    ◄── independent of B2/B3,
                                                        but smaller after B3
B1   post-collision rescale                         ◄── needs B0b only; its
                                                        channel is MOMENTUM, so
                                                        it does not wait for B9
B6   explode/coalesce                               ◄── needs B0b, B1, B3, B4
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
