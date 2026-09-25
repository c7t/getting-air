# Performance snapshot — 2D AMR, 2026-09-24

A dated record of where `index-amr.html`'s time goes on the two target devices,
and a ballpark cost model fitted to it. It is a SNAPSHOT: numbers here describe
build `8aea1e9` on `amr2d/b6-explode`, with `?stride=1` (the default launch). For
the history of how these conclusions were reached, the dead ends included, see
`plans/perf-characterization.md`. To refresh this document, rerun the commands in
[Reproducing](#reproducing) and write a new dated snapshot rather than editing
this one's numbers in place.

| | desktop | phone |
|---|---|---|
| device | RTX 4080, Linux, Chrome (Vulkan) | Pixel 11 Pro XL (img-tec), Chrome 153.0.8010.52 |
| bound by | **pass count** (latency per dependent pass) | **work** (cells stepped and coupled), with a large per-pass cost on top |
| driven by | a debug-port Chrome (`tools/lib/browser-lifecycle.js`) | CDP over `adb forward`, `local: true` (`tools/test-cdp-local.js`) |

The phone is newer than the one `plans/perf-characterization.md` measured
(a Pixel 10 Pro XL), and the software has moved too, so numbers there are not
directly comparable with the ones here.

## Headlines

- **Desktop: a root step is ~7.5 µs × the number of passes.** Cell work does not
  register at these grid sizes. Fewer passes is the only lever.
- **Phone: ~66 µs per pass, ~3 ns per cell stepped, ~4.3 µs per coupled child tile.**
  At the page defaults the steps are ~42% of a root step, level coupling ~32%, force
  and body ~13%.
- **AMR loses to the flat solver at a 256² finest grid and wins 3–7× at 1024².**
  It costs 2–3× the flat solver per cell update, so it only wins where refinement
  cuts the cell count by more than that.

## The passes in a root step

Counted with `tools/count-passes.js`: every compute pass one root step encodes,
named by entry shader file and entry point. `wg` is workgroups per dispatch, averaged over one
64-root-step batch. Refinement rounds run every 16 root steps, so their passes
appear as fractions. Passes that list `mainStride` walk the active-slot list
(`?stride=1`); the others launch every slot.

**Page defaults** (`interface=interp`, `levels=3`, `res=8`; 108 / 188 active tiles) —
**23.8 passes per root step**:

| per root step | ~wg each | pass | what it is |
|---|---|---|---|
| 6 | 1401 | `amr_step1:mainStride` | level 1 step × 2, level 2 step × 4 |
| 4 | 1611 | `amr_force1:mainStride` | body force before each finest substep |
| 4 | 1 | `physics:main` | body update after each force pass |
| 2 | 1611 | `amr_interp_pool_parent:mainStride` | level 1 → 2 ghost interpolation, per level-1 substep |
| 2 | 179 | `amr_average_pool_parent:mainStride` | level 2 → 1 restriction |
| 2 | 1 | `amr_active_list:main` | active-slot list rebuild, levels 1–2 |
| 1 | 5472 | `amr_interp_pool_parent:main` (root parent) | root → level 1 interpolation, every slot |
| 1 | 1024 | `amr_step1:main` | the root step |
| 1 | 608 | `amr_average_pool_parent:main` (root parent) | level 1 → root restriction, every slot |
| 1/8 each | 7–1728 | `amr_criterion_pool`, `amr_manage_pool` decide / coarsen / refine, `amr_cascade:completeQuads` | refinement round |
| 1/16 each | 64–6336 | `amr_cascade:balance`, interp init fills | refinement round |

**Explode, levels=3** (same tiles) — **26.8 passes**: as above, with interp
replaced by `amr_explode` (2 + 1 from the root) and `amr_coalesce` (2 + 1).

**Explode, res=5, levels=4** (`spongeW=2`; 16 / 56 / 56 tiles) — **56.3 passes per root step**:

| per root step | ~wg each | pass |
|---|---|---|
| 14 | 627 | `amr_step1:mainStride` (2 + 4 + 8 substeps) |
| 8 + 8 | 756 / 1 | `amr_force1:mainStride` + `physics:main`, per finest substep |
| 6 + 1 | 696 / 216 | `amr_explode` (pool parents / root parent) |
| 6 + 1 | 203 / 16 | `amr_coalesce` (pool parents / root parent) |
| 6 + 1 | 77 / 608 | `amr_average_pool_parent` (pool parents / into the root) |
| 3 | 1 | `amr_active_list:main` |
| 1 | 16 | `amr_step1:main` (root step) |
| ~1 | — | refinement round, amortized |

**The pass count in closed form** (for `levels = L`; `tools/fit-cost-model.js`'s
`passesPerRootStep`, which matches the counted 23.8 / 26.8 / 56.3):

    steps       Σ_{m=1}^{L-1} 2^m               every pool level's substeps
    body        2 · 2^(L-1)                     force + body update per finest substep
    coupling    k · (Σ_{m=1}^{L-2} 2^m + 1)     k = 2 (interp+average) or 3 (explode+coalesce+average)
    + (L-1) list rebuilds + 1 root step + ~1 amortized refinement pass

The finest level's force/body pair and step (3 passes × 2^(L-1)) are why depth is
expensive on the desktop: 24 of the 56 passes at `levels=4`.

## Per-device constants

| | desktop | phone | source |
|---|---|---|---|
| a real dependent pass | ~7.5 µs | ~56–66 µs | cost-model fit; phone cross-checked by the skip sweep (removing the one-workgroup `physics` passes saved 56–57 µs each) |
| a trivial dependent pass (floor) | ~1.5–5 µs | ~29–30 µs | isolated early-return kernel, one submit |
| one early-returning 8×8 workgroup | negligible | ~14 ns (≈0.23 ns per thread launched) | isolated kernel, 1164–4096 slots |
| one `dispatchWorkgroupsIndirect` call | ~390 µs | ~233 µs | isolated kernel; per dispatch, not per pass. Rules indirect dispatch out on both |
| flat solver, per cell | 0.14 ns (1024²) – 0.30 ns (256²) | 5.4–5.8 ns | `index.html`, sync stepping, after pass cost |
| per-pass GPU timestamps | usable (~1 µs) | unusable (65.5 µs ticks) | |

## The cost model

For an `index-amr.html` root step:

    root step (ms)  =  a · passes  +  b · stepCells  +  c · coupledTiles

- `passes` — the closed form above.
- `stepCells` — the root grid (W²) plus every stepped tile at its **full 20×20**
  (the ghost ring is stepped too): Σ_m tiles_m · 2^m · 400.
- `coupledTiles` — one per child tile per parent substep: Σ_m tiles_m · 2^(m-1).

Fitted by non-negative least squares to 15 timings per device (five configs × three
points in one fall; data in `benchmarks/perf-snapshot-2026-09-24.json`):

| | a (per pass) | b (per step cell) | c (per coupled tile) | fit RMS / max | **predicting a held-out config** RMS / max |
|---|---|---|---|---|---|
| desktop | 7.53 µs | 0.09 ns | 0 | 8.0% / 18.2% | **9.1% / 20.1%** |
| phone | 65.9 µs | 3.02 ns | 4.29 µs | 8.1% / 22.0% | **11.9% / 25.4%** |

**Use it for totals, to about ±10–25%, and not for the split between terms.**
Passes and tiles rise together across these configurations, so the fit cannot
cleanly assign time between them. The phone skip sweep (below), which measures
the groups directly, splits explode res5 L4 as steps ~2.6 ms and coupling ~2.1 ms.
The model's split for the same config, each term with its own passes, is ~1.8 and
~3.0. The worst-predicted config is `res=7&levels=2` on the phone, underpredicted
by 10–22%.

Worked example, phone, explode res5 L4 at t = 2 a/u_t (56 passes, 263 424 step
cells, 328 coupled tiles): 56 × 65.9 µs + 263 424 × 3.02 ns + 328 × 4.29 µs =
3.69 + 0.80 + 1.41 = **5.89 ms** against 5.84 measured. Passes dominate even on
the phone here, which is why this config has the biggest gap to the flat solver.

## Where the phone's time goes (skip sweep)

`tools/bench-amr.js --remote --skip=…` on the phone, refinement frozen after a
2048-step warm-up, three interleaved rounds, median of 512 root steps. The share is
the time saved by not encoding the group.

| skipped group | explode res5 L4 (6.2 ms/root step) | page defaults (4.1 ms/root step) |
|---|---|---|
| step1 (all pool-level steps) | **42.0%** | **41.5%** |
| coupling: explode + coalesce + avg / interp + avg | **34.0%** | **32.1%** |
| of which explode / interp | 13.4% | 23.9% |
| of which coalesce | 15.3% | — |
| of which avg | 7.0% | 8.2% |
| force + phy | 14.3% | 13.2% |
| of which phy alone (1-workgroup passes) | 7.4% | 5.4% |

Spreads 0.5–6.6%. The `-noop` and `step1-ring` twins were left out: they are
direct-launch pipelines, so under `?stride=1` they would change the launch as well
as the pass.

## AMR vs flat, same physics and domain

`tools/bench-amr-vs-dense.js`. Each AMR config runs against `index.html` at its
finest resolution (flat `res = R + L − 1`), with the same card and the same
finest-level τ (checked; a mismatched pair is refused). The flat sponge width is
set to AMR's `spongeW × 2^(L−1)`, because the pages measure it in different cells.
The score is simulated time (a/u_t) per wall second. AMR is timed at 2 / 10 / 40
a/u_t into one `?detslots=1` fall, and flat before and after, scored on the faster
reading.

| AMR ÷ flat speed, t = 2 / 10 / 40 a/u_t (>1: AMR faster) | desktop | phone |
|---|---|---|
| explode res5 L4 (256²) | 0.73 / 0.77 / 0.84 | 0.79 / 0.78 / 0.67 |
| res6 L3 (256²) | 0.95 / 0.97 / 1.01 | 0.81 / 0.78 / 0.72 |
| res7 L2 (256²) | 1.01 / 0.96 / 0.98 | 0.81 / 0.85 / 0.71 |
| res8 L3, page defaults (1024²) | 2.99 / 2.91 / 3.11 | 7.04 / 4.54 / 3.86 |
| explode res8 L3 (1024²) | 2.82 / 2.67 / 2.64 | 5.89 / 3.82 / 3.34 |

Flat absolute: desktop 70.8–74.6 a/u_t/s at 256², 4.5 at 1024²; phone 5.4–5.8 at
256², **0.12 at 1024²** (6.26 ms/step, 168 MLUPS).

Why: an AMR cell update costs 2–3× a flat one on both devices. At 256², tiles are
8 root cells wide and margins are measured in root cells, so a 32² or 64² root
grid ends up mostly refined. `res5 L4` holds 16 of 16 level-1 blocks, and AMR
still does ~45% of flat's cell updates. At 1024², level 2 covers ~7% of the
domain, and AMR does ~11% of flat's updates. On the phone the win also narrows
through a fall (7.0× → 3.9×) as the wake refines. **Accuracy is not part of this
comparison**: equal body resolution is the premise, and what AMR gives up is a
coarser far field.

## Tile size (model-based; one term unmeasured)

Tiles are `RB = 8` root-parent cells (a 16×16 interior at their own level) plus a
2-cell ghost ring, `FB = 2·RB + 4 = 20`. Changing `RB` moves three things:

1. **Ring overhead** — stored ÷ interior = (1 + 2/RB)²: 2.25 at RB=4, **1.56 at
   RB=8**, 1.27 at RB=16, 1.13 at RB=32. Both the step and the coupling scale with
   stored cells (explode launches FB², average restricts the whole interior), so
   RB 8 → 16 cuts per-cell work ~19%.
2. **Over-refinement** — refinement is tile-granular, so the refined band around
   body and wake widens with RB. Of 272 level-2 tiles at the defaults, ≳20% sit on
   the refined region's edge even if it were compact, and the wake is elongated.
   Doubling RB plausibly adds 10–30% refined area, before the 2:1 closure (also in
   tiles) adds more at coarser levels. **Not measured.**
3. **Pass count** — unchanged.

Estimate: **desktop indifferent** (pass-bound). **Phone: RB 8–16, a shallow
optimum**, with 16 likely within ±10% of today and RB=4 clearly worse (+44% per-cell
work for a smaller over-refinement saving). Also constrained by the root grid,
which must tile evenly. In 3D the ring term is cubed, (1 + 2/RB)³ = 1.95 at RB=8
and 1.42 at RB=16, so tile size matters much more there.

What would settle it cheaply: take snapshots from one fall, re-run the refinement
criterion, margins and 2:1 closure on the host at RB = 4 / 8 / 16, count refined
area, and feed it to the model. Measuring it on the GPU would take code changes:
`BLOCK = 8` is a constant in `main-amr.js`, and the average kernel assumes one
8×8 workgroup per tile.

## Measurement hazards

- **Phone thermals**: 24–57% slowdown within a session
  (`perf-characterization.md`). Interleave arms; the contest times flat before and
  after AMR.
- **Desktop background load**: re-running `tools/count-passes.js` the same day with
  the user's browser and `mpv` on the GPU (22% utilization) gave root steps
  1.4–3.5× slower than the table above, with identical pass counts and tiles.
  Check `nvidia-smi` before trusting a desktop timing; pass counts are unaffected.
- **Topology**: use `?detslots=1`, or two runs of one config refine differently and
  do different work.
- **Launch mode**: these numbers are `?stride=1`. `?stride=0` launches every pool
  slot; at `levels=4` that costs the phone ~4 ms per root step in empty workgroups.
- **Phone CDP**: always `local: true`. A `GET /json/protocol` crashes Chrome for
  Android.

## Reproducing

    # desktop (launches its own Chrome; check nvidia-smi first)
    node tools/count-passes.js '' 'interface=explode' 'interface=explode&res=5&levels=4&spongeW=2'
    node tools/bench-amr-vs-dense.js
    node tools/fit-cost-model.js                        # the model, from the JSON below

    # phone (Chrome foregrounded, a tab open on the dev server)
    adb forward tcp:9229 localabstract:chrome_devtools_remote
    node tools/bench-amr-vs-dense.js --remote --port=9229 --baseUrl=https://era:4471
    node tools/bench-amr.js --remote --port=9229 --baseUrl=https://era:4471 \
         --warm=2048 --steps=512 --reps=3 \
         --skip=none,step1,force,phy,force+phy,explode,coalesce,avg,explode+coalesce+avg \
         'interface=explode&res=5&levels=4&spongeW=2&detslots=1'

Measurements: `benchmarks/perf-snapshot-2026-09-24.json`. A new snapshot should
add a new JSON and a new dated document; `tools/fit-cost-model.js` takes the JSON
path as its argument.
