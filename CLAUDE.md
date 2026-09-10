# CLAUDE.md — agent operating notes for `getting-air`

WebGPU (WGSL) D2Q9 lattice-Boltzmann fluid simulator. **Static page, no build
step** — the source *is* the artifact; GitHub Pages serves it directly.

## Layout
- `index.html` / `main.js` — base single-level LBM. Shaders: `shaders/lbm_*.wgsl`.
- `index-amr.html` / `main-amr.js` — 2-level block-structured AMR. Shaders:
  `shaders/amr_*.wgsl`. **Most active work is here.**
- `index-cylinder*.html` / `main-cylinder*.js` — cylinder-in-crossflow validation
  harness (base + AMR variants).
- `index-3d.html` / `main-3d.js` — dense 3D LBM (D3Q19/D3Q27), no AMR,
  slice view, with a 6-DOF rigid body. **One page, five scenarios** via
  `?scenario=duct|beltrami|tgv|sphere|spin`. Shaders: `shaders/d3_*.wgsl`.
  See "The 3D fork" below.
- `index-3d-spike.html` / `main-3d-spike.js` — 3D bench page, the M0
  milestone of `plans/3D.md`. Not a solver: dense periodic grid, no body,
  no AMR, no render.
- `shaders/` — all WGSL. `Makefile` — validation + release helpers.

## Validate before committing (no GPU needed)
Run `make check` and make it pass before committing shader/JS changes:
- `make js` — `node --check` every `*.js` and `*.mjs` (needs Node). The
  shared `.mjs` modules were outside this glob until 2026-09-08.
- `make test` — the GPU-free unit tests (`tools/test-*.js`): the shared
  card/regime parameterization, the AMR field reconstructor, the dense→AMR
  injector, the packed-`f` host/shader layout agreement, the 3D lattice
  tables, and the 3D analytic reference solutions. Add a `tools/test-<name>.js` and it is picked up
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
  **AMR Cd is only reproducible to ~+/-0.001; dense Cd is exact.** Measured
  2026-09-09: two runs of the SAME build gave `amr-N2-diffuse` Cd 1.619 and
  1.620, while `dense-reference` was bit-identical (1.950 / St 0.1258) across
  every run. Cause: pool slots are handed out by an `atomicSub` free-list, so
  block->slot assignment varies run to run, and `amr_force1*.wgsl` atomicAdds
  one TRUNCATED i32 PER WORKGROUP -- regrouping the slots regroups the
  partials, which truncate differently (see that file's own FSCALE header for
  why the truncation is material, and `plans/ghost-free.md` for the full
  trace). So a 4th-digit Cd move on an AMR config is NOT evidence of anything:
  a build-vs-build claim there needs a same-build repeat, not a comparison
  against a number recorded in another session. The dense configs remain
  deterministic and usable as a strict check.
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

## The 3D fork (`plans/3D.md`)

**M0-M3 are done, and M4 is in progress**: M4.1a, M4.1b and M4.1c have
landed. The coarse/fine interface on the `?interface=explode` path is now
exactly conservative in mass AND momentum, and its field error has fallen
to the no-interface control's level -- a refined box tracks `refine=all` to
within 8-10% at N = 32/48/64, against the default `interp` path's 3.6x.
M4.1d has landed too -- the fine level now integrates the body force
(`common_d3_force_pool.wgsl`, weight dx^(D-1) = dx^2, NOT dx^3) and a sphere
in a refined shell reproduces the dense run's Cd to 0.07%. M4.1e, making
explode the DEFAULT, is next and is a blast-radius decision rather than an
evidence one: merging publishes, and the flip moves every recorded AMR
number at once.
Read `plans/3D.md` before touching any of this — in particular its decision
table at the top, which records what is settled so it does not get
re-argued. Two things are settled and load-bearing:

- **The coarse/fine interface uses the materialized RING. Ghost-free is not
  being pursued** (2D measured it at 5-9% against a ~10% floor, it imposes a
  corner-balance refinement constraint the codebase never had, and there is
  memory headroom). See `plans/ghost-free.md` for the measurement and
  `plans/3D.md` sec 2.1 for the decision.
- **f32 registers are not a constraint**, measured, so nothing should be
  restructured around avoiding them. `plans/3D.md` sec 2.4.
- **The 6-DOF body integrates angular MOMENTUM, not angular velocity**, and
  takes a MIDPOINT step on the rotation group. Both are load-bearing: the
  first makes |L| exactly conserved (measured 0.00e+0 over 12500 GPU steps)
  and removes an explicit gyroscopic cross product; the second was worth
  58.87% -> 0.0052% rotational-energy drift, and without it a body spun
  about its MAJOR axis *flipped* — an instability that does not exist, which
  no conservation check catches. `d3-body.mjs` is the reference
  `shaders/d3_physics.wgsl` mirrors; the `spin` scenario compares them on
  real GPU code.
- **Bounce-back is the accurate solid coupling in 3D; diffuse (chi) is
  not.** Sphere Cd measures +7..13% against Schiller-Naumann with
  bounce-back, and +49..131% with diffuse — converging from above with
  resolution, the same open diffuse-interface-width issue recorded above for
  the 2D dense-reference cylinder, amplified because the frontal-area error
  goes as (1+d/R)^2 in 3D rather than (1+d/R). The diffuse sphere cases are
  REGRESSION checks against recorded values, not physics checks; do not
  re-baseline them against literature.

Also settled, and worth knowing before adding a page:

- **One 3D page, scenarios by URL parameter** — `index-3d.html?scenario=`,
  not a page per scenario. The 2D side has four near-identical `main*.js`
  each carrying its own copy of the same bind groups, which is exactly the
  shape that produced 238e48c. Do not fork `main-3d.js`; add a scenario to
  `d3-scenarios.mjs`.

- **Every 3D grid transfer uses the POST-collision Dupuis-Chopard fneq
  factor, `(tau_f - 1)/(tau_c - 1) / 2`, NOT the textbook
  `(tau_f/tau_c) * dx_f/dx_c`.** The step kernels are fused
  pull-stream + collide, so every buffer holds f after collision, and the
  textbook form is for pre-collision populations. At tau=0.8 the two differ
  in magnitude AND sign (-0.25 vs +0.6875). Using the wrong one was the
  large half of M3's seam error, found 2026-09-09. `common_d3_pool.wgsl`
  derives both; `?dcpre=1` restores the wrong one for re-measurement.
  **The 2D solver still has this bug** (`common_interp.wgsl:44`,
  `amr_average_f2c.wgsl:117`, `amr_average_pool_parent.wgsl:116`) --
  deliberately not fixed in the same change, because it would move the
  entire 2D benchmark surface at once and `main` is the published site.

- **The 3D AMR coarse/fine interface IS conservative on the explode path,
  as of M4.1b (`?interface=explode`)** — mass and momentum both, on every
  rung of the geometry ladder (`?refine=all|slab|bar|box`). The DEFAULT
  path (`?interface=interp`) is unchanged and still is not: a
  partially-refined run there converges at FIRST order, whole-domain L2rel
  1.36e-2 / 9.02e-3 / 7.04e-3 at N = 32/48/64 on the analytic Beltrami box
  case, against no-interface controls at 4.6e-3 / 2.3e-3 / 2.0e-3. So **do
  not trust a quantitative number from a partially-refined 3D run on the
  default path**, which is why there is still no sphere-with-AMR case in
  the suite. Explode is not the default yet because it has no body coupling
  and its remaining seam error wants M4.1c.

- **Explode/coalesce is Chen et al. 2006's scheme (the PowerFLOW
  algorithm), and it delivers by DESTINATION.** The interface coarse layer
  is subdivided so coarse and fine voxels overlap the same volume; a state
  whose coarse neighbour does not exist is exploded into the fine voxels
  and REMOVED from coarse dynamics, and the matching fine states are
  coalesced back and removed from fine dynamics. Mass goes exactly one
  place by construction, so the convex corner that defeated refluxing never
  arises. No Dupuis-Chopard rescale anywhere on this path.

  **CONSERVATION AND CONSISTENCY ARE SEPARATE REQUIREMENTS HERE, and the
  geometry ladder cannot tell them apart.** M4.1b ended exactly
  conservative and did NOT preserve a uniform flow, and the conservation
  gate went green over a defect worth 8x the field error. At a convex edge
  a coarse cell's CHILD can cross the seam in one fine step while the
  coarse cell itself does not, and it cuts both ways: a half-step EXIT no
  coarse bucket can absorb (drop it and mass leaks; add it and the
  receiving cell gets more than a uniform state holds), and a half-step
  ENTRY no coarse cell explodes (leave it and the fine region gathers zero
  where it should gather the parent's population). They are one defect seen
  from both sides, they land on the SAME (cell, direction) slot, and in a
  uniform flow their counts are equal so the net write is zero. **Never add
  one without the other.**

  **The trap that cost a full debugging cycle: a ring cell's content
  is claimed by `coarse(p) - e_i`, and that cell is not always covered.**
  The `t + dt/2` outflux has moved one FINE cell, so on a diagonal
  direction `coarse(p)` advances on only some of `e_i`'s axes and
  subtracting `e_i` steps back OUT of the refined region on the others. On
  a flat face it lands on another covered cell (harmless, a lateral smear);
  at a convex EDGE it lands on the diagonally-outside unrefined cell, so
  nobody claimed it and the mass was dropped — measured -1.2e-2 per edge
  coarse cell per macro-step against a predicted f/2 ~ 1.4e-2. The fix is
  the ORPHAN PASS in `common_d3_amr_coalesce.wgsl`, which runs the same
  claim test with the opposite sign on the unrefined cells and ADDS the
  orphan to their own slot. **Do not "simplify" this into gathering by
  ORIGIN** — that is also exactly conservative, is a smaller change, and
  was measured 4.6x worse in the field, because at a convex corner it hands
  a coarse cell mass that physically sits in a different neighbour.
  Conservation is necessary and not sufficient.

- **A stage that measures as a no-op may be downstream of something
  bigger.** M4.1c's linear explosion, measured against M4.1b's interface,
  moved the box case from 2.23e-2 to 2.19e-2 -- inside run-to-run noise,
  and a reasonable read would have been "the explosion is not where the
  error is". It was masked: the edge inconsistency above is O(1) and
  swamped a second-order term. With that fixed, the SAME change is worth a
  factor of two on every geometry. Before concluding a change does
  nothing, check whether a larger defect is saturating the measurement.

- **Refluxing (`?reflux=1`) is SUPERSEDED. Do not extend it.** It is what
  measured the corner and is why explode/coalesce was found: it makes the
  interface exactly conservative and on a corner-free seam (`?refine=slab`)
  halves the field error, but on a seam with a convex corner (`?refine=box`,
  and any body-fitted shell) it is far worse than leaving it off. **The
  corner does not tile**: at a convex corner the coarse grid crosses one
  full channel while the fine grid crosses only two cells x two substeps =
  half of it, because D3Q19 has no (1,1,±1). Fine channels tile a coarse
  FACE exactly and a coarse CORNER not at all, which is the Berger-Colella
  assumption LBM breaks. `shaders/common_d3_amr_flux.wgsl` and plans/3D.md
  M4 carry the full trace, and `?refine=slab` is the control that separates
  a correction bug from the corner.

- **`?refine=all` cannot see an interface bug**, so never treat it as
  coverage for one. With every block refined, the restriction's rescaled
  coarse `f` is consumed by a coarse step whose result is then discarded,
  the rings DIRECT_GHOST never reads are the only thing interp writes, and
  `mac` comes from moments the rescale does not touch. All three `amr-all`
  gates are bit-identical across the rescale fix. `amr-box-RB4` /
  `d3-amr-box` exist because of this.

What exists today:

- `lattice-3d.mjs` — D3Q19/D3Q27 velocity sets, weights and bounce-back
  pairing, **derived rather than typed**. D3Q19 is a strict index-prefix of
  D3Q27, and `opp` is structural (each velocity is emitted immediately
  followed by its own negation), so there is no table to transpose.
- `shaders/common_d3q{19,27}_lattice.wgsl` — **generated** from it by
  `node tools/gen-lattice-3d.js` and checked in (no build step). Do not
  hand-edit; regenerate.
- `tools/test-lattice-3d.js` (in `make test`) — guards them two ways: exact
  drift against the generator, *and* the lattice moment conditions plus a
  searched-not-assumed `opp` involution computed from the tables **parsed
  back out of the WGSL**. The second half is what fails when the generator
  and the checked-in file are wrong together.
- `index-3d-spike.html` — the M0 bench page. Three kernel modes with
  identical memory traffic (`full` / `stream` / `collide`), so they are
  comparable in **GB/s and only in GB/s** — GLUPS necessarily falls by 27/19
  between the velocity sets, and reading that as "Q27 is slow" is the
  mistake. Wired into `validate-all.js` as `d3-spike-boot`.

      node tools/spike-d3-registers.js            # owns Chrome; prints a verdict
      node tools/spike-d3-registers.js --n=96 --peak=717 --wg=8,8,1

- `d3-scenarios.mjs` — the three scenarios AND their closed-form reference
  solutions, shared by the page (which uses them to build initial
  conditions) and by the validation tools (which score against them).
  `tools/test-d3-scenarios.js` guards the references against the PDEs they
  solve — the duct series against an independent SOR Poisson solve verified
  to converge to it at second order, the Beltrami field against
  `curl(u) = k u` by finite differences — rather than against a
  transcription of the same formula.
- `d3-body.mjs` — 6-DOF shapes, TRUE signed distances (sphere and rounded
  box exact; spheroid via the 2D ellipse Newton in the meridional plane; a
  triaxial ellipsoid deliberately absent, since an approximate one is the
  trap `common_geometry.wgsl`'s header documents), inertia tensors,
  quaternions, and the free-body integrator. `tools/test-d3-body.js` guards
  the struct layout against the shader, asserts `|grad phi| = 1` and a
  brute-force nearest-point search, and asserts the tennis-racket theorem
  (a body spun about its intermediate axis must flip, one about a stable
  axis must not).
- `tools/validate-3d.js` + `benchmarks/d3.json` — the M1, M2 and M3 gates.
  Analytic PASS/FAIL for duct, Beltrami, spin and the bounce-back sphere;
  the 3D TGV only reports; the diffuse sphere cases and `amr-box-RB4` are
  regression checks against recorded values (read `amr_box_note` before
  re-baselining — when refluxing lands those numbers should FALL).

      node tools/validate-3d.js                   # owns Chrome; 27 cases
      node tools/validate-3d.js --cases=spin-N32
      node tools/validate-3d.js --skip=tgv,sphere,sphere-diffuse

- `tools/analyze-d3-interface.js` — the coarse/fine interface diagnostic.
  Reports, does not PASS/FAIL. Runs the refined box against its own
  no-interface controls (`refine=all`, dense) in one invocation and prints
  two things: the field error bucketed by signed distance to the seam, and
  **total mass and total momentum drift**. The second is the discriminator,
  and it is what found the rescale bug — Beltrami is periodic and
  force-free so each level alone conserves both exactly, and fneq has no
  zeroth or first moment, so "mass at the f32 floor while momentum leaks"
  says the defect is in the non-equilibrium coupling and not in the
  addressing or the flux scheme. Reach for it before building anything at
  the interface.

      node tools/analyze-d3-interface.js
      node tools/analyze-d3-interface.js --n=64 --td=0.25,0.5
      node tools/analyze-d3-interface.js --configs=box --extra=dcpre=1

- `d3-amr.mjs` + `tools/test-d3-amr.js` — the octree pool's geometry and
  addressing (`?levels=2`, `?rb=`, `?refine=all|box|body`). The neighbour
  resolution is checked against an INDEPENDENT route -- global fine
  coordinates, where ownership is one division with no ring and no offsets
  -- rather than by re-running the same arithmetic. RB=4 and RB=8 must give
  BIT-IDENTICAL results, and the suite asserts that exactly, not to a
  tolerance: RB changes only how the domain is cut into tiles.

  **Read `benchmarks/d3.json`'s `tolerances_note` before re-baselining
  anything** -- the tolerances were set from measurement, and it records
  which cases are deliberately off the bounce-back "magic" tau so the
  wall-position error stays visible instead of tuned away. All 13 cases
  were green 2026-09-09: duct cross-section L2rel 6.3e-5..1.7e-3, Beltrami
  decay-rate error 1.2e-5..2.1e-3 converging at second order in N. The
  `?scenario=tgv` case deliberately has no tolerances and only reports --
  the Re=1600 comparison needs literature data this project does not have.

  Measured 2026-09-09, RTX 4080, 128^3: D3Q19 **3.62 GLUPS / 551 GB/s (77% of
  peak)**, D3Q27 **2.35 GLUPS / 508 GB/s**, with the full kernel matching its
  own stream-only ceiling at both — i.e. purely bandwidth-bound, no spill.
  Repeat spread 0.2%, so unlike the AMR Cd numbers above this harness is
  reproducible to well under a percent — a build-vs-build claim here does
  not need the same-build repeat those do (the tool takes one anyway and
  prints the spread alongside its verdict).

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
