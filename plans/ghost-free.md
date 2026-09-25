# Ghost-free tiles: stop materializing the ring, not just the copy pass

**Status: planned, not started.** Written 2026-09-09. This is 2D work on
`main`, with its own payoff; it is also the prerequisite decision for
`plans/3D.md` (see §7).

`DIRECT_GHOST` (shipped 2026-09-08, `plans/perf-characterization.md`'s "the
one lead left") removed the fine-fine ghost COPY PASS. It did not remove the
**ring itself**. This removes the ring: `FB = 2*RB` exactly, `GHOST = 0`.

---

## 1. The two things called "ghost", and which one is left

1. **The ring** — `FB > 2*RB`, real allocated memory holding copies of
   neighbouring data. **Still there.**
2. **The fine-fine copy pass** — filled the ring from same-level neighbours
   between substeps. **Gone since 2026-09-08.**

After `DIRECT_GHOST`, the ring serves exactly one case, stated in
`shaders/amr_step1.wgsl`'s own header:

> The ring is still materialized, and interp still fills it: a tile at the
> coarse/fine interface has no same-level neighbour there, so `blockSlot` is
> < 0 and this falls back to the clamped read of a parent-interpolated ghost.

So the whole remaining job of the ring is: **hold coarse-parent data for
interface tiles.** Ghost-free replaces that stored value with an **inline**
parent reconstruction in the gather — the same bilinear rho/u/`fneq` plus
Dupuis-Chopard rescale `amr_interp_pool_parent.wgsl:275-286` already
computes, moved from a dedicated pass into the point of use.

### Why `GHOST=2` and not 1

Load-bearing, not slack. Each level takes two substeps per parent step with
no ring refresh between them: `S_Advance` interpolates once per child
*cycle*, and the parent's state does not change between the child's own A and
B, so re-interpolating would produce identical values. Two layers let the
ring self-advance — substep A's gather on a depth-1 ring cell reaches into
depth 2, so depth 1 is still valid for substep B.

Inline sampling makes depth meaningless: the parent is sampled fresh at each
substep, at the point of use. So `GHOST` does not go 2 -> 1, it goes to 0.

---

## 2. What it is worth in 2D

**The steady-state `interp` pass stops being encoded.** Measured at
**13.8-17.3% of frame on the desktop and 19.2% on the phone**
(`plans/perf-characterization.md`, "AMR coupling, measured on BOTH devices").
That is the largest single remaining item after `step1`.

**Be precise about which interp goes.** `amr_interp_dense_parent.wgsl` compiles
two pipelines from one module via `GHOST_ONLY`:

- `GHOST_ONLY=1` — the steady-state, every-macro-step ring refresh. **This is
  what disappears**, and it is what the `interp` bench group measures.
- `GHOST_ONLY=0` — the one-time full-slot fill on a newly activated tile.
  **This stays.** A tile that has just been refined still needs its interior
  seeded from the parent. It is not in the per-macro-step budget.

**Pool memory 6.25 -> 4.0 cells stored per coarse cell covered**
(`(FB/RB)^2`: 400/64 against the ideal 256/64), i.e. ~11 MB -> ~7 MB.
Irrelevant in 2D; stated only because it is the whole argument in 3D.

**The fine step processes 36% fewer cells** (FB 20 -> 16). Expect this to be
small: `?benchSkip=step1-ring` measured **0.2% desktop / 0.9% phone**, because
ring threads share their 8x8 workgroups with interior threads and free no
scheduling slot. That instrument skipped ring *threads* while leaving their
reads and the whole `interp` pass in place, so a real ghost-free build should
beat 1% — but the original 10.5% prediction from the traffic model was wrong
by 10x and should not be re-derived.

### What it costs, and the honest expected range

Interface threads pay the parent reconstruction inline, potentially once per
direction. Only tiles with `blockSlot < 0` on some face pay it, so in 2D this
is a minority of tiles, and 23% of a tile's cells touch a boundary at RB=8.

Precedent for the shape of the trade: `DIRECT_GHOST` removed a 17.5% pass and
gave back **+7.5% desktop / +3.6% phone** on the fine step, netting
9.9% / 13.1%. Inline parent sampling is materially more expensive per thread
than the neighbour read `DIRECT_GHOST` substituted, so expect a worse
give-back.

**Honest expected range: -5% (a net loss) to +15%.** This is not a confident
prediction and should not be reported as one.

**What would falsify it:** if the fine step's increase exceeds the interp
saving, ghost-free loses on speed in 2D. See §7 for why that is a warning for
3D rather than a veto.

---

## 3. What has to change

### 3.1 `GHOST` is hardcoded in TEN shaders -- DONE (step 1)

    shaders/amr_step1.wgsl               shaders/amr_interp_dense_parent.wgsl
    shaders/amr_step1_pool.wgsl          shaders/amr_interp_pool_parent.wgsl
    shaders/amr_average_f2c.wgsl         shaders/amr_force1.wgsl
    shaders/amr_average_pool_parent.wgsl shaders/amr_force1_pool.wgsl
    shaders/amr_criterion_pool.wgsl      shaders/amr_render.wgsl

(An earlier draft of this section said seven. It missed `amr_force1.wgsl` and
both interp shaders. `grep -ln '^const GHOST = 2u;' shaders/*.wgsl` is the
authoritative list -- exactly ten.)

All are now `override GHOST : u32 = 2u;`. No module-scope `const`, workgroup
array size, or other const-expression context depends on `GHOST` in any of
them, so the conversion is mechanical; every kernel already derives `FB` from
it with a `let` rather than hardcoding 20.

**The hazard in this step is the JS side, and it points the opposite way from
what you would guess.** `main-amr.js`'s own comment states it: *"WebGPU makes
passing an undeclared override a pipeline-creation error, not a warning."* So
the dangerous mistake is passing `GHOST` to a pipeline whose shader does NOT
declare it -- that is a hard failure at boot, on whichever page happens to
build that pipeline. Missing a pipeline that DOES declare it is harmless now
(it falls back to the default 2) and silently wrong later, at step 3.

Both directions therefore have to be audited, not just eyeballed. Map each
`XSM` module variable to its `.wgsl` (the `loadShader` array in each page),
then for every `module: XSM ... constants: <expr>` check whether `GHOST`
reaches it, directly or through a named object or spread. Two sites were
found only this way, and neither matches the obvious pattern:

- `avgConstants` in `main-cylinder-amr.js`, `main-channel-amr.js` and
  `main-tgv-amr.js` -- a separate object from `fineConstants`, feeding `avgSM`.
- `main-channel-amr.js`'s `step1Constants`, which is declared `let` and
  reassigned, so a regex anchored on `const` misses both occurrences.

Note also that in `main-cylinder/channel/tgv-amr.js`, `fineConstants` drives
the **render fragment**, not the average pass -- the object names do not mean
the same thing across the five pages. Do not assume the pages are copies of
each other; they diverged (this is the 238e48c surface).

Gates run: `make check`; full `validate-all.js` default sweep; boot smoke PASS
on all four pages; invariants PASS; all seven analytic `channel-*`/`tgv-*`
configs PASS; `amr-N2-bounceback`, `amr-N3-diffuse`, `amr-N3-bounceback` PASS.
The two documented FAILs (`dense-reference`, `amr-N2-diffuse`) reproduce.

#### FINDING: "byte-identical" is the WRONG standard here, and AMR Cd is not reproducible to 4 digits

The first draft of this plan said step 1 would be "byte-identical at 2". It is
not, and chasing that produced a more useful result than the step itself.

Measured on the desktop, `amr-N2-diffuse` at Re=100:

| build | run | Cd |
|---|---|---|
| `main` (control) | 1 | **1.620** |
| `override GHOST` | 1 | **1.619** |
| `override GHOST` | 2 | **1.620** |

**Two runs of the SAME build gave 1.619 and 1.620.** The change is exonerated
-- the delta sits inside the harness's own run-to-run spread -- but the real
result is that spread existing at all.

`dense-reference` was bit-identical across every run (Cd 1.950, St 0.1258),
which is what makes this diagnosable rather than ambient noise: the dense page
has **no pool**, and the nondeterminism is in the pool. Both halves of the
mechanism are documented in the shaders themselves:

- **Slot assignment is nondeterministic.** The free list is a LIFO popped with
  `atomicSub(&freeCount, 1)` (`amr_manage.wgsl:330`, `amr_manage_pool.wgsl:365`),
  so which block lands in which slot depends on the order workgroups reach the
  atomic. Semantically irrelevant -- physically, it regroups the data.
- **The force reduction truncates PER WORKGROUP.** `amr_force1_pool.wgsl`'s own
  header: it "atomicAdds ONE TRUNCATED i32 PER WORKGROUP (safeFixed's i32()
  cast truncates toward zero)". Integer `atomicAdd` is order-independent, so
  the reduction is not the problem -- but the truncation is applied to each
  workgroup's *partial* sum, and a workgroup covers a slot. Regroup the slots
  and the partials truncate differently.

That header also establishes the truncation is material, not theoretical: at
`FSCALE=1e4` it cost ~10% of the force at L1 and ~32% at L2, moving Cd from
1.430 to 1.593 at N=2. `FSCALE=1e7` shrinks that ~1000x -- landing the residual
at exactly the ~1e-3-on-Cd scale observed above.

**Consequences for the rest of this plan:**

- **Cd at three decimals cannot be step 3's gate.** A `+/-0.001` move on an AMR
  config is indistinguishable from a re-run. The gate is what CLAUDE.md already
  says it is for a layout change -- the analytic `channel-*`/`tgv-*` checks --
  plus `validate-divergence`, which reports a SHAPE (`edge` vs `half` vs level
  share) rather than a single scalar and is therefore immune to this.
- **A dense-page Cd change IS meaningful** at full precision, because that path
  is deterministic. Use `dense-reference` as the strict check and the AMR
  configs as the loose one.
- **Any future build-vs-build claim on an AMR config needs the same-build
  repeat**, not a comparison against a number recorded in another session.
  Getting this backwards is how `plans/perf-characterization.md` records two
  earlier wrong conclusions being reached.

### 3.2 The neighbour-resolution hoist already degenerates correctly

Worth knowing before planning around it. `amr_step1.wgsl:191-193`:

    offX = select(select(0, 1, fx + 1u >= GHOST + RB2), -1, fx <= GHOST);

At `GHOST=2, RB2=16, FB=20`: `fx in {0,1,2} -> -1`, `fx in {17,18,19} -> +1`.
At `GHOST=0, RB2=16, FB=16`: `fx == 0 -> -1`, `fx == 15 -> +1`. Correct in
both cases — only the outermost interior cells can reach outside a 1-cell
pull. **No change needed to the hoist, the three-slot resolution, or the
neighbour-local re-expression.** The change is confined to the `ns < 0`
fallback.

### 3.3 The parent sampler has to become a shared include -- DONE (step 2)

Done as `shaders/common_interp.wgsl`: `CoarseSample`, `dupuisChopardRescale()`
and `interpCoarseToFine()`, included by both interp shaders. 57 lines removed
against 15 added; the two shaders differed in NOTHING but how they fetch a
parent cell.

Two `shader-loader.mjs` properties constrain what a fragment may do, and both
matter for step 3 when the step kernels include it too:

- **One level of include only** ("Fragments must not themselves @include"), so
  `common_interp.wgsl` cannot pull in `common_lattice.wgsl` for `feqD2Q9`. It
  relies on the includer having done so. Both interp shaders already did;
  `amr_step1{,_pool}.wgsl` already do too, so step 3 needs no new include
  there beyond `common_interp.wgsl` itself.
- **No dedup** -- each `@include` splices verbatim, so a double include is a
  duplicate-declaration error. Check the assembled output, not the source.

### 3.3b TWO FINDINGS FROM DESIGNING STEP 3 -- read before implementing

Both were missed by this plan's first draft, and both change what step 3 is.

#### (i) The ring is a SNAPSHOT IN THE SCHEDULE, not a cache of a value you can recompute

"Replace the stored ghost with an inline parent sample" assumes the parent
looks the same at both moments. It does not. Traced through `S_Advance`:

    ROOT: interpolate INTO L1  (from L0's CURRENT state, i.e. time t)
          L0's own ONE step    (L0 is now at t+1)
          recurse into L1      (L1 does its 2 substeps HERE)
          average L1 back into L0

The interp that fills L1's ring reads L0's **pre-step** buffer -- confirmed in
`main-amr.js`: `interpBG_readB` is chosen on the same `useB` that selects
`stepBG_ba`, so interp and the step read the same buffer, and the step writes
the other one. But `S_Advance(1)` runs AFTER `L0 step`. So a naive inline
sample during L1's substeps would read L0 at **t+1**, where the ring holds
**t**. That is a different interface scheme, not the same one computed later.

**Fix, and it is clean at level 1:** bind the parent's PRE-STEP buffer to the
step kernel -- the very buffer interp read, untouched for the rest of the
macro-step because L0 steps only once. Inline sampling then reproduces the
ring's cycle-start value *exactly*.

**What still differs, unavoidably:** the ring SELF-ADVANCES. It is filled once
per child cycle, and the ring cells are collided along with the interior, so at
the child's substep B a depth-1 ring cell holds parent(t) collided once, not
parent(t). Inline sampling gives parent(t) at both substeps. So ghost-free
changes the temporal coupling at the interface -- from "snapshot then let it
evolve under the fine LBM" to "hold the parent's start-of-step state fixed
across both substeps".

`amr_interp_dense_parent.wgsl` calls the existing behaviour the "intended
multi-rate degrade over 2 substeps", and the word chosen is *degrade*; the
DIRECT_GHOST work already found the ring path leaves a depth-2 cell
"clamp-degraded after substep A". Neither scheme is temporal interpolation of
the parent, which is what the literature actually prescribes. So this is a
change between two approximations, not a regression from a correct scheme --
but **step 3 is a numerical-scheme change, not plumbing, and must be gated as
one.**

**The gate that makes this tractable already exists.** `validate-divergence
--mode=fullrefine` has no coarse/fine interface at all, so the `ns < 0` path
never fires and ghost-free must be **exactly** identical there. That is a hard
exactness check on everything except the interface. `--mode=adaptive` then
isolates the interface change, and the `edge` column is precisely where this
scheme change should appear. Run both; expect fullrefine unchanged to the last
digit and adaptive to move.

#### (ii) The ring is MEMOIZATION, and that is most of why it is not pure overhead

Interp reconstructs each ring cell once; each ring cell is then read by up to
three interior threads' gathers. Inline reconstruction repeats the work per
(thread, direction), and each reconstruction costs four parent fetches of nine
`f` values apiece -- 36 reads to produce one direction's value if done naively
inside the 9-direction loop.

Counted properly, per tile:

- **2D, RB=8:** ring is `400 - 256 = 144` cells. Inline: boundary interior
  cells `16^2 - 14^2 = 60`, each with 3 (edge) or 5 (corner) out-of-interior
  source directions ~= **164** reconstructions. Only **~1.14x** redundancy.
- **3D, RB=4:** shell is `1728 - 512 = 1216`. Inline: `8^3 - 6^3 = 296`
  boundary cells at roughly 7 out-of-tile directions each ~= **2072**, i.e.
  **~1.7x**.

So the redundancy is real but bounded, and it is swamped by the other half of
the accounting: **the ring is materialized and interpolated for EVERY tile,
while the inline path fires only where `blockSlot < 0`.** A tile in the
interior of a refined region pays full ring cost today and would pay *zero*
inline cost. The trade is therefore "everywhere, cheaply" against "only at
interfaces, ~1.5x more expensively" -- which is a good trade exactly when
interface tiles are a minority, and a poor one for a thin refined shell where
nearly every tile touches an interface. **That is the affordability question,
now stated precisely enough to predict rather than guess.**

Implementation consequence: `common_interp.wgsl` needs a single-direction
variant so the naive per-direction cost is not paid. Even so, the four parent
fetches must be shared across directions that resolve to the same source cell,
or the 36-reads-per-direction figure above is what gets paid.

### 3.4 The genuinely fiddly part: a child sampling its parent's edge

`amr_interp_pool_parent.wgsl`'s header notes that a child's parent sampling
reaches into its parent's ring. With `GHOST=0` at level 1, a level-2 tile at
its parent's edge needs data from the **parent's neighbour tile**. Solvable
with the same `blockSlot` mechanism one level up — the parent level's own
`blockSlot` is already bound in several of these layouts — but it is the part
most likely to be got wrong, and it should be its own commit with
`validate-divergence` run against it specifically.

### 3.5 `amr_criterion_pool.wgsl` depends on the ring for stencil margin

Its own header (line 29): *"(GHOST=2) already provides the +-1 margin every
interior cell needs"*. At `GHOST=0` a vorticity stencil at a tile edge has no
margin. Needs either the same neighbour resolution or an explicit clamp, and
a clamp changes the refinement criterion at tile edges — which is a physics
change, not a plumbing one. **Do not let this one ride.**

### 3.6 JS and offline tooling

`FB` derivation, pool sizing, `getBlockGridDims`, snapshot header. All five
AMR pages carry their own copy (the 238e48c surface — change them together
and run boot smoke on all of them).

**The offline tooling needs no changes.** `tools/lib/dense-to-amr.js:286` and
`tools/lib/field-reconstruct.js:241` both read `GHOST`/`FB` out of the
snapshot header rather than hardcoding, and their loop bounds
(`lo = fillGhosts ? 0 : GHOST`, `hi = fillGhosts ? FB : GHOST + 2*RB`)
degenerate correctly at `GHOST=0`. Verified by reading, not assumed.

Snapshots taken under one flag will not load under the other (`FB` differs).
The existing loader already validates `RB`/`MAX_FINE_BLOCKS` against the page
and throws a clear message; extend it to `FB`. Fair trade for a URL flag.

---

## 4. The flag

`?ghostfree=1`, default 0, both paths live in one build — the same shape as
`?ghostcopy=` and `?f16=`, for the same reasons: A/B-able for speed AND for
physics without a second build, and

    node tools/validate-all.js --extra=ghostfree=1

sweeps the entire validation suite against it for free. Add a `ghostfree`
group to the `?bench=1` default sweep so the phone can be measured in one
session (the only way to beat that device's 24-57% thermal ramp).

Default flips to `GHOST=0` only after §5 passes and §6 measures a win, at
which point the flag inverts to `?ghostfree=0` as the restore path — exactly
what `?ghostcopy=1` is today.

---

## 5. Validation gates

All of these exist. None of them had to be written for this.

- **`tools/validate-divergence.js` is the primary gate**, and it is
  purpose-built for this class. It seeds dense and AMR from one shared state
  and reports WHERE they diverge: a high `edge` column is the ghost/seam
  class, `half` is tile registration, `fullrefine` is the noise floor and the
  ratio to it is the interface error. **A ghost-free change is a seam
  change.** Run `--mode=both` (default) and compare the interface excess
  ratio against the numbers the DIRECT_GHOST work recorded: 1.12x, `edge`
  0.95-0.96, adaptive final relL2(ux) 7.74e-3, fullrefine 6.91e-3.
- **The analytic `channel-*` / `tgv-*` checks are the gate**, per CLAUDE.md —
  Cd/St is not, for anything that changes what is stored or how it is
  reconstructed. This qualifies.
- **`tools/validate-amr-invariants.js`** — 2:1 balance and geometry coverage,
  asserted periodically through a run.
- **Cd/St** against the documented baseline, including its known failures
  (`dense-reference` Cd 1.950, `amr-N2-diffuse` 1.620 — the open
  diffuse-interface-width issue, not a regression).
- **Boot smoke on all four pages** (`index-boot`, `amr-dev-boot`,
  `reentry-boot`, `reentry-amr-boot`). This change touches bind groups on
  shaders five AMR pages share. That is precisely the 238e48c failure mode.
- `make check`.

---

## 5b. VALIDATION RESULT (2026-09-09, step 3, level 1, N=2)

`node tools/validate-divergence.js --res=9 --levels=2 --re=20 --mode=both`,
with and without `--extra=ghostfree=1` (that passthrough was added to
validate-divergence and buildAMRUrl for this run; it appends to the AMR page
ONLY, since the dense reference is a different build with none of these flags
and silently appending an ignored option would make a run look controlled when
only one leg changed).

Note the control was run at **levels=2 to match**. The 1.12x / edge 0.95-0.96
figures in perf-characterization.md are levels=3 and are NOT a valid control
for this.

**fullrefine -- no coarse/fine interface, so INLINE_PARENT never fires:**

    step      control (ring)   ghost-free
    0                0.00e+0      0.00e+0     <- seeding, exact on both
    64               6.06e-3      6.06e-3
    128              6.18e-3      6.19e-3
    192              6.26e-3      6.26e-3
    256              6.32e-3      6.32e-3

Agreement to <=0.16%, i.e. at the harness's own ~1e-3 reproducibility floor
(see CLAUDE.md / 93ca558). Seeding error is **exactly zero on both**, and that
part IS deterministic -- injection is a copy, no atomics -- so the `FB` 20->16
layout change and the snapshot/reconstruction round-trip are exact.
**The implementation is correct.**

**adaptive -- where the scheme change lives:**

    step      control     ghost-free    delta
    64        6.20e-3     6.21e-3       +0.2%
    128       6.34e-3     6.36e-3       +0.3%
    192       6.44e-3     6.47e-3       +0.5%
    256       6.52e-3     6.56e-3       +0.6%

    interface excess   1.032x  ->  1.038x
    edge               0.91    ->  0.92     (both well below 1.0)
    half               1.16    ->  1.15

**Ghost-free is very slightly WORSE at the interface -- ~0.6%, growing with
time, same direction at all four checkpoints.** Monotone 4/4 argues real rather
than noise, which scatters. This is exactly the predicted cost of losing the
ring's self-advance (sec 3.3b(i)): substep B sees parent(t) rather than
parent(t) evolved one substep.

**`edge` did NOT rise**, so this is a small diffuse accuracy cost, not a seam
defect. A mis-indexed inline reconstruction is precisely what the `edge` column
exists to catch, and it is not showing one.

### The Cd shift is UNEXPLAINED and should not be read as a win

`amr-N2-diffuse` Cd moved from the documented 1.620 FAIL into the 1.35+/-0.15
band under ghostfree -- a >=7% move, >=100x the harness's own +/-0.001 repeat
noise, so it is real. But the field-level divergence above says the interface
coupling changed by **0.6%**. Those do not reconcile easily.

The regimes differ (divergence is Re=20 steady bounceback; the Cd config is
Re=100 diffuse shedding), so "a small coupling change moves where vortices
cross the interface and shifts shedding" is plausible -- but it is a
hypothesis, not a finding. **The divergence test does NOT corroborate the Cd
move as an accuracy improvement**, and Cd is the metric CLAUDE.md specifically
says not to trust for a layout change (the packed-fp16 precedent: passed Cd/St
on both AMR cylinder configs while missing every channel tolerance by 5-20x).
Treat it as an open observation to explain, not as evidence for the change.

## 6. Measurement plan

Desktop via `tools/bench-amr.js --skip` with competing GPU clients stopped
and `nvidia-smi` showing the SM clock near idle first. Phone via one in-page
`?bench=1` sweep with `ghostfree` in the configuration list, `?telemetry=1`,
`interrupted: false`, Chrome foregrounded throughout.

Four configurations, so the net splits into its two halves — the same design
that made the `DIRECT_GHOST` measurement legible:

    none (ghost-free)                 ghostfree=1
    step1 skipped (ghost-free)        ghostfree=1 & benchSkip=step1
    none (legacy ring)                ghostfree=0
    step1 skipped (legacy ring)       ghostfree=0 & benchSkip=step1

That separates "the interp pass went away" from "the fine step got more
expensive", instead of reporting one net number that hides both.

Read anything under ~10% on the desktop as below the measurement floor, and a
negative share as proof of it.

---

## 6b. BENCHMARK RESULT (2026-09-09, step 3, level 1, N=2)

### A matched A/B is structurally hard on this page, and that is the first finding

`tools/bench-amr.js` only works on `index-amr.html` (the only page exposing
`window.__AMR`) -- and that page is the FALLING CARD. Ghost-free perturbs the
flow, which changes the forces, which changes the card's trajectory, which
moves the geometry-forced refinement somewhere else by the time refinement
freezes. So the two arms do not see the same topology:

    default warm-up (20000 steps)        L1 = 63 vs 80   (27% apart)
    + refineThresh=999 (geometry only)   L1 = 69 vs 54   (still 22% apart)
    + refineThresh=999 AND --warm=0      L1 = 30 vs 31   (3.3% -- usable)

`--warm=0` is what makes it comparable: the block set is then a pure function
of the card's INITIAL position, identical in both arms. `refineThresh=999`
alone is not enough, because the card has already moved by the freeze point.

**A future A/B of anything that perturbs the flow needs a PINNED body.** The
cylinder page has one, and cannot be used, because it does not expose
`window.__AMR`. Exposing it there is the right fix and is not done.

### The numbers

Desktop RTX 4080, 9 reps of 10000 macro-steps, competing GPU clients closed
mid-session (the first attempt carried an 87% spread against ~6-13% after --
a Google Meet tab; `nvidia-smi` showed the SM clock idle at 210 MHz the whole
time, which is exactly the "utilization% alone is misleading, and so is the
clock alone" trap):

    config                                   median    spread   L1
    res=8&levels=2&refineThresh=999          3161 ms   13.7%    30
    ... &ghostfree=1                         2792 ms   12.1%    31

    time -11.7%, with ghost-free doing 0.6% MORE useful work (31 tiles vs 30)
    -> roughly 12% faster per useful cell update.

Corroborated independently by the unmatched-topology run, normalized by work.
`bench-amr.js`'s own Mcell metric counts `FB*FB` -- DISPATCHED cells, ghosts
included -- which turns out to be the useful way to read it:

    config       tiles  FB   dispatched   useful    Mdisp/s   Museful/s
    ring            63  20      115936     97792      360.7      304.3
    ghost-free      80  16      106496    106496      359.8      359.8

**Dispatched throughput is identical (360.7 vs 359.8, 0.25% apart).** The GPU
chews cells at the same rate either way; what changed is that ghost-free's
cells are all useful. +18.2% on that normalization.

### Read it as 10-18%, not as a number

The spreads in the matched run are 12-14%, and
`plans/perf-characterization.md` says to read anything under ~10% on the
desktop as below the floor. So this is at the edge of what this harness can
resolve, and the two independent readings (12%, 18%) bracket it.

**What the shape says, which matters more than the value:** the win is NOT
that inline reconstruction is cheap. It is that the ring path dispatches 400
cells per tile to update 256 of them -- 36% waste -- and runs an interp pass
on top. The inline gathers cost about what they replace, which is consistent
with the ~1.14x 2D redundancy estimated in sec 3.3b(ii). That also predicts the
3D case will be worse on this axis (~1.7x redundancy, and 58% of a tile's cells
on the boundary at RB=4), while the dispatch-waste half of the win gets BIGGER
(a ghosted 3D tile at RB=4 dispatches 1728 cells to update 512 -- 70% waste).

### Cost side, for the same change

`validate-divergence` (sec 5b): interface error ~0.6% worse, monotone across
all four checkpoints, `edge` unchanged at 0.91 -> 0.92. So the trade at N=2 is
**~12-18% faster for ~0.6% more interface error**, plus a ~1.56x pool memory
reduction that does not matter in 2D.

## 6c. N=3 RESULT (2026-09-09) -- and a correction to 6b

### The bug that made the first N=3 benchmark meaningless

`?ghostfree=1` at N_LEVELS=3 produced **Cd=-5815 / -12308** on the cylinder
harness. amr_step1_pool.wgsl's own `fineToCoarseUnit` maps to L0 BUFFER SPACE
(`levelParams.dxL`, for the card SDF); the parent-local mapping is
`origin - 0.25 + 0.5*j` (ratio always 2). Using the former to index the latter
put every level>=2 parent sample at half scale. Fixed as `fineToParentUnitI`.

Three things this cost, all worth keeping:

- **`invariants PASS` throughout**, including the FIELD-FINITE check, while the
  solution was garbage. Invariants passing is not evidence the physics is right.
- **`tgv-amr-N3` PASSED throughout.** TGV has no body, hence no
  geometry-forced refinement, hence the level>=2 near-body parent sampling is
  never exercised. The analytic checks are the gate for a LAYOUT change and are
  NOT sufficient for an ADDRESSING change that only fires near a body.
- **The N=3 gates could not run at all** until the cylinder/TGV pages got the
  four-way parent-buffer wiring, so an unvalidatable configuration was
  benchmarked. Same structural failure the boot-smoke configs exist to prevent.

### Corrected benchmark: +9.8%, not +18.9%

Re-measured on the fixed solver, and for the first time with **exactly matched
topology** (`activeL1..n delta [0,0]`) -- which is itself the confirmation that
the garbage field was driving the earlier 40-vs-48 L2 discrepancy:

    N=3, 6000 steps, 30 L1 / 40 L2 in BOTH arms
    ring         4688 ms  (spread 7.7%)   155.9 Museful/s
    ghost-free   4268 ms  (spread 9.9%)   171.3 Museful/s
    -> -8.9% wall time = +9.8% throughput, no normalization needed

**Sec 6b's +18.9% is withdrawn.** It came from the broken run, where ghost-free
carried 20% more L2 tiles and the useful-work normalization credited it for
that extra work. The N=2 figure (+13.9%) was taken at 30-vs-31 tiles and is
only near-matched; treat it as indicative too.

### The accuracy cost at N=3 is an order of magnitude worse than at N=2

`validate-divergence --res=9 --levels=3 --re=20 --mode=both`, control at the
same levels:

    fullrefine (no interface): BIT-IDENTICAL in every column and checkpoint.
      The exactness test passes perfectly -- no addressing bug remains in the
      path that should not fire.

    adaptive:   step   control    ghost-free   delta
                 64    6.96e-3    7.04e-3      +1.1%
                128    7.27e-3    7.48e-3      +2.9%
                192    7.55e-3    7.92e-3      +4.9%
                256    7.80e-3    8.33e-3      +6.8%

    interface excess  1.124x -> 1.200x   (interface error contribution +60%)
    edge              0.95   -> 0.96     (still below 1: diffuse, not a seam defect)

At N=2 the same measurement gave +0.2% -> +0.6%. Two interfaces instead of one,
AND level 2's inline path carries the grandparent clamp that level 1 does not.

### Verdict at N=3, which is the default config

    speed            +9.8%
    interface error  +6.8% divergence, excess 1.12 -> 1.20
    gates            amr-N3-diffuse St regresses PASS -> 0.1494 (needs 0.150)
                     amr-N3-bounceback PASS, tgv-amr-N3 PASS

**That is a bad trade as it stands.** ~10% throughput for a 60% increase in
interface error contribution and a gate regression. The grandparent clamp
(sec 3.4 / the override header in amr_step1_pool.wgsl) is the thing to attack
before re-judging it: it is the one part of the level>=2 path that is knowingly
approximate, and it fires exactly where geometry-forced refinement bypasses the
2:1 neighbour gate -- i.e. around the body, which is where Cd/St are decided.

## 7. The relationship to `plans/3D.md`

3D is the reason this is worth doing now rather than eventually. Two
questions, and 2D answers them very unevenly:

- **Is inline parent reconstruction CORRECT at the seam?** 2D answers this
  **completely**, with the apparatus in §5 — none of which exists for 3D yet.
  This answer transfers whole.
- **Is it AFFORDABLE?** 2D gives a **lower bound only**. The cost scales with
  the fraction of a tile's cells that touch a boundary: **23% in 2D at RB=8,
  58% in 3D at RB=4** — 2D understates it by ~2.5x. And the memory prize that
  motivates it in 3D does not exist in 2D at all.

**The asymmetry that matters:** a 2D *speed* loss is a warning for 3D, not a
veto — 3D has a memory argument 2D lacks, and it decouples tile size from the
memory budget there (see `plans/3D.md` §2.1). A 2D *correctness* failure is a
veto, and finding it here costs days; finding it inside a brand-new 3D solver
with no validated baseline and no seam classifier costs weeks, and its failure
mode is a plausible-looking wrong flow rather than a crash.

Doing this first also means the 3D pool shaders get written against a
mechanism that has already been proven, rather than one being invented at the
same time as the solver around it.

---

## 8. Order of work

Each step is separately verifiable, and the first two are semantic no-ops
(which is not the same as bit-identical -- see §3.1).

1. `override GHOST` in the ten shaders + pipeline constants. **DONE.**
   Semantically a no-op at the default, but NOT bit-identical -- see §3.1's
   finding. Gate: `make check`, boot smoke, full `validate-all.js` sweep, and
   a same-build repeat of `amr-N2-diffuse` to separate the change from the
   harness's own spread.
2. Extract the parent sampler into `shaders/common_interp.wgsl`, used by both
   interp shaders. **DONE.** Pure refactor; `amr-N2-diffuse` came back at
   1.620, i.e. exactly the control draw, and all four analytic AMR configs
   PASS.
3. **DONE (level 1, N=2).** `?ghostfree=1`: `GHOST=0` variant; the `ns < 0` fallback in
   `amr_step1{,_pool}.wgsl` calls the shared sampler inline; the steady-state
   interp dispatch is not encoded. The `GHOST_ONLY=0` activation fill stays.
4. §3.4 — child sampling across its parent's tile edge. Own commit, own
   `validate-divergence` run.
5. §3.5 — the criterion's stencil margin. Own commit; a clamp here is a
   physics change and needs the analytic gate, not just Cd/St.
6. §5 in full, then §6 on both devices.
7. Decide the default. If it flips, invert the flag to `?ghostfree=0` and
   **document it in CLAUDE.md's "Performance work" section** alongside
   `?ghostcopy=1` — CLAUDE.md documents what exists, so it changes when this
   lands, not before.
