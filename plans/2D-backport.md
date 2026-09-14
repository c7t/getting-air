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
| B1 | Post-collision Dupuis-Chopard `fneq` factor | **yes, a live bug** | 2D uses the pre-collision form on post-collision populations | S (code) / L (re-baseline) |
| B2 | `cascade21`: 2:1 as ONE closure on the WANT set | **yes** | replaces 3 live balance bugs, an unbounded fixed-point loop, and closes a measured refinement defect on the shipped page | M |
| B3 | One kernel per stage + a `parent_{dense,pool}` accessor | **yes** | deletes ~1100 lines of near-duplicate WGSL across 6 kernel pairs | M |
| B4 | The body lives entirely on the finest level | **yes, already true** | makes finest-wins masking in 3 force kernels provably dead code | S |
| B5 | The window is a translation of the SPONGE | **half already true in 2D AMR, not in `main.js`** | removes window bookkeeping from 5 hot kernels; retires `card-total.mjs`'s problem | M |
| B6 | Explode/coalesce at the coarse/fine interface | **yes, but measure first** | 2D's interface is not conservative, and nothing in 2D measures that | L |
| B7 | The diffuse band converges at 2nd order; Richardson it | **yes** | may close the standing `dense-reference`/`amr-N2-diffuse` Cd=1.95-vs-1.35 red cells | S |
| B8 | `SOLID_EQ` on the bounce-back path | **latent only** | free on every 2D gate (all bounce-back bodies are pinned); disarms a landmine | S |
| B9 | f32 lattice weights do not sum to 1 | **yes, and 2x worse than recorded** | measured 1 + 1.49e-8, not the 7.45e-9 CLAUDE.md states; a gate whose error grows with run length is measuring a drift | S (code) / L (re-baseline) |

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
   had already drifted. Expect more of this in `main-*-amr.js` -- worth a
   sweep of its own before B3 rather than during it.

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
   This is the hot kernel and the largest single win.
2. `average` — one body, two parent fragments.
3. `interp` — `common_interp.wgsl` already holds the shared half; move the
   fetch behind `parentIndex` and collapse the two entry files.
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
B0   host module + mutation-checked tests        ──┐
B0b  interface instrument (mass/momentum drift)  ──┤ prerequisites
                                                   │
B4   finest-level-only + hard failure            ◄──┘  (smallest, proves B0)
B7   chi band ladder                                  (independent, cheap)
B8   SOLID_EQ                                         (independent, free)
B2   cascade21                                   ◄── needs B0
B3   kernel unification, manage LAST              ◄── needs B2
B5   window = sponge translation                  ◄── independent of B2/B3,
                                                      but smaller after B3
B1   post-collision rescale                       ◄── needs B0b
B9   lattice weights                                  (independent; re-baseline)
B6   explode/coalesce                             ◄── needs B0b, B1, B3, B4
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
