# CLAUDE.md — agent operating notes for `getting-air`

WebGPU (WGSL) D2Q9 lattice-Boltzmann fluid simulator. **Static page, no build
step** — the source *is* the artifact; GitHub Pages serves it directly.

## Layout
- `index.html` / `main.js` — base single-level LBM. Shaders: `shaders/lbm_*.wgsl`.
- `index-amr.html` / `main-amr.js` — 2-level block-structured AMR. Shaders:
  `shaders/amr_*.wgsl`. **Most active work is here.**
- `index-cylinder*.html` / `main-cylinder*.js` — cylinder-in-crossflow validation
  harness (base + AMR variants).
- `shaders/` — all WGSL. `Makefile` — validation + release helpers.

## Validate before committing (no GPU needed)
Run `make check` and make it pass before committing shader/JS changes:
- `make js` — `node --check` every `*.js` and `*.mjs` (needs Node). The
  shared `.mjs` modules were outside this glob until 2026-09-08.
- `make test` — the GPU-free unit tests (`tools/test-*.js`): the shared
  card/regime parameterization, the AMR field reconstructor, the dense→AMR
  injector, and the packed-`f` host/shader layout agreement. Add a `tools/test-<name>.js` and it is picked up
  automatically; each must run with no server/browser/GPU and exit nonzero
  on failure.
- `make wgsl` — validate every `shaders/*.wgsl` with `naga` (needs `naga`;
  `make tools` installs it via cargo/Rust). If `naga` is absent, `make check`
  still runs the JS checks and skips WGSL with a note.
`make check` needs no network and no `gh`. This is static validation only —
it does **not** prove the app renders; for that use the `webgpu-verify` skill
(`.claude/skills/webgpu-verify/`), which drives a real GPU Chrome and
screenshots. A black/blank canvas is a failure, not success.

## Physics & AMR validation (needs a GPU)
`make check` doesn't run the simulation. These do — they drive a real WebGPU
Chrome via CDP and either compare against known physics or assert structural
invariants the AMR machinery depends on:

- **`tools/validate-all.js`** — single top-level harness, run this by
  default. Owns the whole Chrome/HTTPS-server lifecycle (launches its own
  dedicated debug-port Chrome if none is running; one tab reused across every
  config via `Page.navigate`, never more than one WebGPU context alive at
  once). Runs both checks below across the dense reference and every AMR
  levels/bounce-back combination (`dense-reference`, `amr-N2-diffuse`,
  `amr-N2-bounceback`, `amr-N3-diffuse`, `amr-N3-bounceback`), plus a cheap
  boot smoke check (`index-boot`, `amr-dev-boot`, `reentry-boot`,
  `reentry-amr-boot`) against `index.html`, `index-amr.html` and the two
  reentry pages — pages with no `window.__CYL`, so the Cd/St and invariant
  checks don't apply, but that used to mean this suite never loaded them at
  all. Added after a shared-shader/JS-bind-group mismatch
  broke `index-amr.html` in production (a WGSL binding count change was
  mirrored into `main-cylinder-amr.js`'s own bind group but not
  `main-amr.js`'s separate copy of the same one) without failing anything
  here, since nothing had ever visited that page — `runBootSmoke` polls
  `#status`: a pipeline-creation failure is caught internally by that page's
  own `init().catch(handleErr)`, written as `error: ...` into `#status` (not
  an uncaught exception `Runtime.exceptionThrown` would catch), so this
  checks the status text itself advances past its initial value without
  ever starting with `error:`. Prints one aggregated PASS/FAIL report.
      node tools/validate-all.js                        # full default sweep
      node tools/validate-all.js --configs=amr-N2-bounceback
      node tools/validate-all.js --configs=index-boot,amr-dev-boot
      node tools/validate-all.js --re=20,40,100,200 --steps=20000
      node tools/validate-all.js --extra=f16=2      # append to every config's URL
  `--extra=` appends query parameters to every config, including the channel
  and TGV harnesses that build their own per-case URLs, so a pipeline override
  can be swept across the whole suite without a second copy of the table.

  **The default sweep is not currently all-green on `main`.**
  `dense-reference` and `amr-N2-diffuse` fail at Re=100 (Cd 1.950 and 1.620
  against 1.35±0.15). That is the open diffuse-interface-width issue, not a
  regression: the chi band is ~±4 cells regardless of resolution, so the
  effective body radius exceeds the nominal one and Cd converges from above
  (1.908 at res=9 → 1.597 at res=10). The bounce-back variants of the same
  configs pass, which is consistent. Re-baseline against these numbers rather
  than assuming a red cell is yours.
- **`tools/validate-cylinder.js`** — physics: pinned cylinder in uniform
  crossflow, time-averaged Cd/Strouhal vs. literature values in
  `benchmarks/cylinder.json`. Assumes a Chrome + page are already up (see
  `webgpu-verify`) — `validate-all.js` is the one-command version.
- **`tools/validate-amr-invariants.js`** — AMR structural invariants,
  asserted periodically through a run (not just at the end, so a transient
  violation can't slip past): 2:1 balance between neighboring tiles
  (`window.__CYL.debugCheck21Balance`) and the geometry-forced-refinement
  hard constraint — every leaf tile near the body must already be at the
  finest configured level (`debugCheckGeometryCoverage`) — plus a cheap
  field-finite (NaN/blowup) smoke check.
- **`tools/validate-amr-vs-dense.js`** — standalone/opt-in, **not** part of
  `validate-all.js`'s default sweep (a high-resolution dense run is far more
  expensive than that suite's default configs). Runs the dense reference at
  a high target resolution and AMR refined down to that *same* physical
  resolution at the cylinder surface, then diffs the two solvers'
  velocity/density/vorticity **fields** directly against each other (not
  just each vs. literature Cd/St, which is all the tools above check) —
  reconstructing AMR's quadtree pool data onto a uniform grid for the
  comparison (`tools/lib/field-reconstruct.js`). The base-resolution/levels
  scaling law and its own validity checks (clamp, validated-levels cap, a
  base-grid relaxation-time stability margin) live in
  `tools/lib/amr-resolution-mapping.js`. Own header comment documents a
  currently-open finding worth reading before trusting a run.
      node tools/validate-amr-vs-dense.js --res=10 --levels=2,3 --re=20,40
      node tools/validate-amr-vs-dense.js --res=8 --levels=2 --re=20 --mode=fullrefine
- **`tools/validate-divergence.js`** — standalone/opt-in. Runs the dense
  reference and AMR forward from **one shared initial state** (seeded via
  `tools/lib/dense-to-amr.js`, which overwrites a real AMR snapshot's field
  data with the dense solver's and hands it to `debugSnapshotLoad`) and
  reports how, where and how fast they diverge. Exists because comparing two
  *independently-timed* runs is meaningless once the flow sheds — two correct
  periodic solutions at uncorrelated phase disagree enormously (see
  `plans/AMR-vs-dense-validation.md`'s Finding #3). Seeding both legs
  removes phase from the comparison. Read the SHAPE, not the magnitude:
  nonzero at step 0 is injection, a high `edge` column is a ghost/seam bug,
  a high `half` column is the tile-registration class, the per-level error
  share catches per-level tau/force, and low concentration means ordinary
  truncation error. Always run `--mode=both` (the default): `fullrefine` has
  no coarse/fine interface, so it is the noise floor, and the ratio to
  `adaptive` is the interface error. Reports, does not PASS/FAIL — there is
  no literature value for how fast two discretizations should diverge.
      node tools/validate-divergence.js --res=9 --levels=3 --re=20
      node tools/validate-divergence.js --diffuse --saveSnapshots=/tmp/div
  A `fullrefine` seeding error must be exactly zero (the finest level maps
  1:1 onto the dense grid, so injection is an exact copy); the tool warns if
  it is not, and that means the injection/reconstruction path is at fault,
  not the solver.
- Shared analysis code lives in `tools/lib/` (`cylinder-metrics.js`,
  `amr-invariants.js`, `field-reconstruct.js`, `amr-resolution-mapping.js`,
  `amr-cost.js`, `browser-lifecycle.js`, `dense-to-amr.js`) — both the leaf
  tools and `validate-all.js`/`validate-amr-vs-dense.js`/
  `validate-divergence.js` call the same logic, not independently-drifting
  copies. `dense-to-amr.js` (dense snapshot → AMR hierarchy) and
  `field-reconstruct.js`'s `reconstructAMRToResolution` (AMR → uniform grid)
  are inverses of each other by construction — they walk the same quadtree
  recursion — which is what lets `tools/test-dense-to-amr.js` validate the
  injector by round-trip instead of by eye.

## Performance work
Read `plans/perf-characterization.md` BEFORE optimizing anything here. The
two target devices have **opposite** bottlenecks — the desktop is
pass-count/latency bound, the mobile PowerVR is memory-bandwidth bound — and
the obvious optimization (indirect dispatch off the active block count) was
measured to help neither. It also records the trap that produced a confident
wrong conclusion: per-pass GPU timestamps are unusable on that mobile part
(65536 ns counter granularity vs sub-tick passes), so attribution has to be
done at frame scale via `?bench=1`.

**For any change to precision or storage layout, the analytic field checks
(`channel-*`, `tgv-*`) are the gate — Cd/St is not.** Cd and St are
time-averaged surface integrals dominated by the near-body region where
`fneq` is largest; they average far-field noise away instead of reporting it.
Measured directly: packed-fp16 mode 2 PASSES the Cd/St harness on both AMR
cylinder configs while missing every channel tolerance by 5-20x. The same
document records how the previous fp16 answer came out wrong — an emulation
the driver optimized away, with a control that could not detect that.

`?f16=1|2` — real packed-half storage for `f` (`shaders/common_fpack.wgsl`,
`f-pack.mjs`). Default 0 and byte-identical to the previous `array<f32>`
layout. Measured NOT viable as a default; kept for re-measurement, not for
shipping.

`?ghostcopy=1` — restores the legacy materialized same-level ghost cells on
every AMR page. Default 0: the fine step addresses neighbour tiles directly
during streaming (`DIRECT_GHOST` in `shaders/amr_step1*.wgsl`), so the
between-substep fine-fine ghost COPY pass is not encoded at all. Worth ~10% of frame GPU time on the
desktop and ~13% on the phone — see `plans/perf-characterization.md`'s "The one
lead left" for the per-device decomposition and for why the ~30% it originally
predicted was wrong. Both
paths are live in one build, so they can be A/B'd for speed
(`?benchSkip=ghostcopy`, also in the `?bench=1` default sweep) and for physics
(`node tools/validate-all.js --extra=ghostcopy=1`).

Timing/measurement entry points:
- `?telemetry=1` — POSTs periodic samples (device, adapter, config, frame
  GPU/sync ms) to the dev server's `/_telemetry`, appended to
  `telemetry.log`. The way to see a device `tools/bench-amr.js` cannot drive,
  e.g. a phone on the LAN. (A phone on USB does expose CDP via `adb forward
  tcp:9222 localabstract:chrome_devtools_remote`, but only while Chrome is
  foregrounded, and driving `debugStepSync` over it killed Chrome — see
  `plans/perf-characterization.md`. Use `?bench=1` + this.) Opt-in,
  same-origin, local-only.
- `?profile=1` — attaches `debugProfileMacroStep`'s per-pass breakdown to
  those samples. Trustworthy only where the timestamp counter is
  fine-grained; check the values are not all multiples of one number first.
- `?bench=1` — runs the frame-scale pass-group attribution sweep and posts
  one summary. `?benchSkip=force,interp` applies a skip by hand. Freezes
  refinement so every configuration sees identical topology; the physics is
  deliberately wrong while it runs.

Current known-issue state (e.g. which `?levels=N` combinations are physics-
validated) drifts with active work — see `main-cylinder-amr.js`'s own
comment above `N_LEVELS`, not this file, for what's current.

## Branch model
- **`main`** — canonical, always-buildable, default branch, **and the published
  site**: GitHub Pages serves `main` directly, so anything merged is live within
  a minute. All PRs land here.
- **feature branches** — short-lived, off `main`, PR back into `main`, deleted on
  merge. (Fork contributors use `user/<handle>/<topic>`.)
- `gh-pages` and `pub` are **gone** (deleted 2026-09-08). `gh-pages` was a
  separate published snapshot advanced by a release step; `pub` was an earlier
  attempt at the same idea that in practice was either immediately stale or
  ahead of `main`. Both are replaced by "Pages serves `main`".

## Publishing
There is no release step. **Merging to `main` publishes** — Pages rebuilds from
it automatically. `make publish` is a guarded no-op kept only because the old
muscle memory is now dangerous: it used to push `HEAD` to the Pages branch, and
with the Pages branch being `main` that would push whatever branch you are on
straight to main, bypassing the PR flow.

    make status         # origin repo, current branch, and what Pages serves

`make status` talks to GitHub via `gh` and always acts on **your own `origin`**
(derived from the remote URL, never `gh repo view`, which resolves a fork to its
parent). `gh` is required only for `status`; run `make require-gh` to check it is
installed, current, and signed in.

Because merging is publishing, `make check` (and, for anything physics-affecting,
the GPU validation above) matters *before the merge*, not before a separate
release.

### If Pages ever needs re-pointing
Derive `OWNER/REPO` from `origin` (`git remote get-url origin`), then:

    gh api --method PUT "repos/OWNER/REPO/pages" --input - <<< '{"source":{"branch":"main","path":"/"}}'
    gh api --method POST "repos/OWNER/REPO/pages/builds"   # force a rebuild now

Repoint **before** deleting whatever branch Pages currently serves, so the site
never unpublishes, and confirm `gh api repos/OWNER/REPO/pages/builds/latest`
reports `built` at the expected commit before removing the old branch.

## Conventions
- Never hardcode the repo owner/name — derive identity from `origin`.
- Prefer ecosystem tooling (`make`, `naga`, `node`) over ad-hoc scripts.
- Small, focused commits; do not commit secrets or local dev artifacts.
