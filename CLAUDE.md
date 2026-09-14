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

**M0-M6 are done -- M6.4's octree raymarcher and M6.5's movie mode included,
so a refined run can now be LOOKED at at every level's own resolution, and
filmed. M8.0-M8.3 are done: the target Re runs, the solver sheds at the
right frequency, the solid interior is held at the body's equilibrium, and a
moving window (M8.3) makes a moving-body measurement converge at all.** Depth is real:
`?levels=N` runs at any depth, static or `?dynamic=1`, the body lives
entirely on the finest level, and the 2:1 balance and ring-parent checks are
gates there rather than VACUOUS lines. The
coarse/fine interface on the `?interface=explode` path -- the DEFAULT since
M4.1e -- is exactly conservative in mass AND momentum, and its field error
has fallen to the no-interface control's level: a refined box tracks
`refine=all` to within 8-10% at N = 32/48/64, against the old `interp`
path's 3.6x. The fine level integrates the body force
(`common_d3_force_pool.wgsl`, weight dx^(D-1) = dx^2, NOT dx^3) and a sphere
in a refined shell reproduces the dense run's Cd to 0.07%.

- **Dynamic refinement works behind `?dynamic=1` (M4.2b), at any depth
  (M5.5b), refused unless `?refine=body`.** A topology change is SIX ORDERED
  PASSES -- decide(+close), drain, coarsen, refine, fill, clear -- and the
  order is the design, not style. `drain` must precede `coarsen` (freeing
  first loses the fine solution, since `refine` can re-hand the slot in the
  next pass); `fill` must follow `refine`; `coarsen` and `refine` must stay
  separate passes (a live 2D free-list race put two blocks on one slot); and
  `clear` is its own pass because clearing the new-flag inside `fill` races
  the very test `fill` uses to pick its work.

- **AND THE LEVEL ORDER IS OPPOSITE FOR THE TWO HALVES** (M5.5b).
  Drain+coarsen sweep FINEST-FIRST, because a dying tile restricts into its
  PARENT and the parent must still hold its slot when it does; refine+fill
  sweep COARSEST-FIRST, because a new tile is interpolated FROM its parent.
  A whole subtree can be born or die in one event, so neither order is a
  preference. The manager's want-clear is a KERNEL (`clearWant`) and not a
  `writeBuffer`: the manager is encoded mid-command-buffer and
  `device.queue.writeBuffer` is ordered at SUBMIT, so a host-side clear
  would zero one step's want and leave every other step accumulating into
  stale data.

- **A dispatch that is the right SHAPE for one buffer is not thereby right
  for another.** The depth-2 drain was encoded through the coarse-grid
  dispatch instead of `average`'s own `(per, per, per * slots)`, so at RB=4
  it covered the first `NZ/4` slots and skipped every one above -- 4 of 160
  on one config. A tile coarsened out of a skipped slot was freed without
  its solution ever reaching L0, and NOTHING in the suite could see it: the
  structural configs check the pool, which stays perfectly consistent, and
  the one field gate has a PINNED body that never drains. Found in M5.5b by
  reading, not by a red cell.

- **A tile being born or absorbed IS an ordinary grid transfer, so it KEEPS
  the Dupuis-Chopard rescale.** Chen's no-rescale applies to the interface,
  where one state moves between two bookkeepings of one volume and the
  a = (n-1)/2n offset absorbs tau. Creation and destruction are not that:
  the two grids must carry the same rho, u and stress, and fneq carries the
  stress. `interp` (`NEW_ONLY`) and `average` (`DYING_ONLY`) are the right
  tools, rescale included. An earlier note in plans/3D.md said the opposite
  and was wrong.

- **The geometry criterion REFINES AHEAD: `MARGIN + manageEvery * |v|`.** A
  shell that only just covers the body at decision time is stale on the next
  step, and the constraint has to hold at every step, not the ones the
  manager runs on. Zero for a pinned body, so it does not disturb the static
  cases. Do not delete it as an unexplained fudge -- `?refine=body` on the
  `drift` scenario fails without it.

- Two lessons worth carrying from that work: a "nothing changed" gate cannot
  tell a no-op manager from one that never ran (use `?manageMargin=` to force
  it to act, `?manageStart=` to place the event after the flow develops, and
  watch `debugPoolState()`), and `step` only advances after a whole
  `debugStepSync` batch, so any per-step interval must use `step + s`. A
  third: a count is not a set -- a translating body holds `inUse` constant
  while every tile changes hands, which is why `debugPoolState` reports the
  refined set's bounding box.

- **The AMR invariant checker is GREEN, and its 2:1 half is VACUOUS at
  `?levels=2` only** (`tools/validate-d3-invariants.js`, `d3-amr.mjs`'s
  `check21Balance` / `checkGeometryCoverage`). At `?levels=2` a leaf's
  neighbour is level 1 or level 0 and both are legal, so the check cannot
  fail; the tool says so rather than showing a green tick. The depth-3 rows
  are where it is a gate -- `box3`/`bar3`/`body3` on a HOST-built tree, and
  `body3-dynamic`/`drift3` on one the MANAGER rebuilds every few steps.
  Geometry coverage IS real at every depth. The pure functions are unit-tested on inputs that
  VIOLATE the invariant (`make test`) -- a checker only ever run on valid
  input is indistinguishable from one that returns nothing.

- **The 2:1 forcing rule is ONE closure on the WANT set, and it ships as a
  host function with NO kernel** (`d3-amr.mjs`'s `cascade21`, M4.2b-iv).
  `present(m,b)` requires `present(m-1, parent(n))` for each face neighbour
  n; read forwards that is "refine forced by a deeper neighbour", read
  backwards it is "coarsen blocked by one", and they are not two mechanisms
  -- which is why it belongs between `decide` and `drain` rather than as
  tests inside coarsen and refine, where the 2D manager's three live balance
  bugs live. At `?levels=2` the closure is the IDENTITY (a level-1 block's
  parent level is the dense L0 grid), so `common_d3_manage.wgsl` has no
  `balance` entry point on purpose: a kernel that provably writes nothing
  back is untestable code. `make test` gates the rule against
  `check21Balance` instead -- including minimality, without which "refine a
  halo to be safe" passes everything.

- **THE BODY LIVES ENTIRELY ON THE FINEST LEVEL. Hard requirement, not a
  policy** (plans/3D.md M5.4). The geometry criterion is evaluated at the
  finest level only and every coarser level is whatever `cascade21` requires,
  so this holds by construction; M5.4 depends on it rather than defending
  against it. Two consequences: exactly ONE force pass, at the finest level,
  with no finest-wins masking anywhere; and NO COARSE/FINE SEAM EVER TOUCHES
  THE BODY OR ITS MARGIN, so how an interface should behave with a body
  crossing it is unreachable rather than merely unanswered. Enforced by two
  refusals -- a body with AMR needs `?refine=body`, and `?margin=` must cover
  the force stencil's reach -- and by M5.4a below.

- **RUNNING OUT OF POOL SLOTS IS A HARD FAILURE** (M5.4a). Refinement is
  geometry-forced, so a refused tile means a seam through the body; 2D
  experience is that such a run does not drift, it diverges. The manager
  records refusals in `freeCount[1]` and the host latches it, writes `error:`
  into `#status` and stops advancing. `validate-d3-invariants.js`'s
  `body-refine` config exists to prove it FIRES.

- **A POOL KERNEL WORKS IN ITS PARENT LEVEL'S CELL UNITS, and the body lives
  in L0 units.** `fineToCoarseUnit3` returns parent units; at level 1 those
  coincide with L0 and below it they do not. The map is AFFINE
  (L0 = 0.5u - 0.25 per rung) because refinement is cell-centred, hence
  `L0_SCALE`/`L0_OFFSET`. It applies to EVERY position in a kernel --
  `common_d3_amr_step1.wgsl` has two and `common_d3_force_pool.wgsl` has two
  -- and missing the bounce-back link test's NEIGHBOUR position leaves the
  integrated force at exactly zero while everything else looks right.

- **A BUFFER'S LAYOUT IS CARRIED BY ITS WGSL TYPE, not by convention.** The
  dense `mac` is `array<vec4<f32>>` (interleaved); a pool's `mac_pool` is
  `array<f32>` (planar, component-major). They were both `array<f32>` and
  coalesce wrote the pool one in the dense layout at depth 3 -- every moment
  landed in a different cell, mass drift -8.0e+3 against a bound of 5, and
  it survived a first look because both are f32 arrays of exactly the right
  length. The shared coalesce body now cannot NAME either array; it calls
  `parentMacStore`, declared by `common_d3_parentmac_{dense,pool}.wgsl`.
  Measured in Chrome: `vec4<f32>` is four consecutive floats with no padding
  (so host readbacks and bind-group layouts are unchanged), a dynamic
  component index `mac[i][k]` compiles and runs, and it costs nothing
  (0.953x, inside noise). **`const_assert` does NOT work on `override`
  expressions**, and nearly every shape here is an override -- so the type
  is the only static check available.

- **REFINEMENT IS OCTET-COMPLETE FROM LEVEL 2 DOWN: a parent spawns all
  eight children or none.** `amr_manage_pool.wgsl` says it for the 2D quad,
  a tile is allocated per block, and `check21Balance`'s `hasChild` tests
  octant (0,0,0) alone because of it. The first `cascade21` added single
  blocks; a parent holding only octant (1,0,0) still read as a LEAF and the
  checker reported the violation the cascade was meant to remove. The
  checker was right. M5 will meet this again.

- **Explode is not slower in any case that matters** (`tools/bench-d3-interface.js`,
  new). +2.9% per macro-step on a body-fitted shell, -4.5% at a larger
  refined fraction, +31% only for a small refined box in a small domain. All
  of the cost is the coalesce ORPHAN pass -- `?orphans=0` lands on interp
  everywhere -- and the linear explosion is free. Price a pass by REMOVING
  it (`?orphans=0`, `?explin=0`, both physics-wrong), never by per-pass
  timestamps. The desktop is noisy enough that the tool reports the MINIMUM
  of its reps, not the median; treat sub-10% differences as noise.
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
- **THE SOLID INTERIOR IS HELD AT `feq(1, u_body)` EVERY STEP UNDER
  BOUNCE-BACK** (`SOLID_EQ`, both step kernels; `?solideq=0` to A/B). Under
  bounce-back `chi` is 0, so interior cells are stepped with reflected
  gathers and NOTHING DAMPS THEM -- measured, max|u| reached 6x the body's
  own speed INSIDE the body at tau=0.6, and a moving body blew up in 200
  steps at tau=0.514. A PINNED body never notices, because nothing reads a
  solid cell (the step's bounce-back branch reads `f_in[opp[i]]` at the
  FLUID cell; the force kernel runs only where `phi >= 0`), which is why
  every gate in the suite missed it for as long as no body moved. That same
  property makes the fix provably free: it cannot change a pinned-body
  number, and does not. It is ALSO the fresh-node refill -- a cell entering
  the fluid arrives at equilibrium with the body's velocity -- done every
  step because that is cheaper than detecting when to do it. **2D DOES NOT
  HAVE THIS PROBLEM, AND AS OF 2026-09-11 THE REASON IS ESTABLISHED: 2D HAS
  NO MOVING BOUNCE-BACK BODY AT ALL.** `USE_BOUNCEBACK` is set from a URL
  parameter in exactly one file, `main-cylinder.js`, and that body is PINNED;
  `main.js`, `main-amr.js`, `main-reentry.js` and `main-reentry-amr.js` never
  set the override, so every moving 2D body -- the falling card and both
  reentry pages included -- runs the DIFFUSE (chi) coupling.
  `main-reentry.js:270` says so outright, and the commit that introduced 2D
  bounce-back (04bac4b) is explicit that it was for the pinned cylinder:
  "Ladd's moving-wall correction included for parity ... though it's exactly
  0 for the pinned cylinder validation case". So the earlier note here, that
  reentry is a moving bounce-back body, was wrong on the coupling.

  That matters beyond the bookkeeping: chi never partitions cells into solid
  and fluid, so NOTHING EVER CHANGES HANDS, and the whole class of
  covered/fresh-node defects -- including plans/3D.md's D1, the swept-cell
  momentum the force kernel could not see -- cannot arise in 2D. 3D cannot
  follow: its diffuse sphere Cd is +49..131%. There is no 2D precedent to
  inherit here, neither a solution nor a recorded dead end.

- **A MOVING BODY IS CHARGED FOR THE FLUID IT SWEEPS UP, and before
  2026-09-11 it was not** (`SWEPT_FORCE`, both 3D force kernels, `?swept=0`
  to A/B; plans/3D.md D1). A moving body's discrete surface is a staircase
  that CHANGES: each step, cells on the leading face cross into the solid and
  `SOLID_EQ` destroys their momentum, and cells on the trailing face cross
  back carrying the `feq(1, u_body)` last written into them. NEITHER CROSSES A
  LINK, so a momentum exchange summed over bounce-back links cannot see
  either, and the fluid's momentum changes anyway. It is FIRST ORDER in the
  drag, not a correction to it -- the swept rate is `A * u_body` cells per
  step -- which is how a body-scale error hid inside a surface integral.
  **IDENTICALLY ZERO ON A PINNED BODY** (v = omega = 0 makes the pose one step
  ahead the pose now), so every sphere gate is bit-identical across it, and
  measures so.

- **THE INSTRUMENT FOR ANY MOVING-BODY CLAIM IS THE GALILEAN SPLIT, NOT A
  RESOLUTION LADDER** (`tools/probe-d3-galilean.js`; `fall` and `card` both
  take signed `?tow=` and `?stream=`). The body moves at `tow`, the fluid sits
  at `stream`, only the DIFFERENCE is physical, so holding it fixed and
  sliding the split is one flow seen from a moving frame -- same grid, same
  Re, same domain, same blockage, and the reference value is the other legs
  rather than a paper. Cd(a) must be flat in a. **A resolution ladder cannot
  do this job**: M8.2c read the tow/stream gap as resolution because it
  assumed a missing momentum term goes as `A U^2` with no D dependence, but
  the real term goes as `A * u_body * dU` with `dU` a boundary-layer slip,
  which also shrinks with resolution. That claim is retracted in part; the
  split found the same defect at ONE resolution, falling 31% linearly in the
  body's own speed.

- **THE BROADSIDE PLATE IS THE ONLY BODY HERE WITH NO STAIRCASE ERROR, and it
  is the first whose Cd is comparable to literature without a caveat**
  (`?scenario=card&tilt=0` with `tow`/`stream`, `plate_cases`). A ROUNDBOX
  with r = 0 whose body frame permutes the world axes lies exactly on cell
  faces. Measured Cd 1.2066 (chord 16) and 1.2012 (chord 24) against the
  square-plate 1.18 -- +2.3% and +1.8%, and 0.4% APART, where every sphere
  case sits at +7..13% and moves several percent per rung. **But only if it is
  PLACED right**: a cell is solid when its CENTRE has phi < 0, so an integer
  half-extent needs the body on a cell CORNER and a half-integer one needs a
  cell CENTRE. Centred naively the plate was 15x15x1 instead of 16x16x2 and Cd
  read 1.0024 -- a 12% area error reported as physics. `make test` asserts it
  by COUNTING SOLID CELLS.

- **NO MOVING-BODY DRAG NUMBER BELOW tau ~ 0.55 IS TRUSTWORTHY, on the dense
  path OR the pool** (plans/3D.md D1/D3). The bounce-back moving-body coupling
  degrades as tau -> 1/2. Measured by momentum budget in a periodic
  sponge-free box, where the fluid's momentum can only change through the body
  so `sum F` must equal `-dP` exactly, the force is short by
  **16.6% / 3.6% / 1.5% / 0.0% at tau = 0.509 / 0.55 / 0.6 / 0.8**, with
  `deficit * (tau - 1/2)` constant at ~1.6e-3 -- i.e. it goes as
  1/(tau - 1/2), linear in Re at fixed U and D. The Galilean split says it
  from the other side: at tau = 0.509 the dense path reads -18.5% across the
  frame and the pool +47.3%, and **at tau = 0.6 both are flat to ~1% and agree
  with each other to ~1%**. So the two are ONE defect on two discretizations,
  not a dense bug and a pool bug. This is Lambda = (tau-1/2)^2 biting: 8.1e-5
  at tau = 0.509, against the 3/16 at which halfway bounce-back places the
  wall exactly halfway (benchmarks/d3.json's tolerances_note already records
  that constant for the STATIC duct cases). A TRT collision, which fixes
  Lambda independently of tau, is the standard remedy and would be a real
  change to this solver.

- **AND IT SHARPENS M8.0 RATHER THAN CONTRADICTING IT.** M8.0 says the card's
  stability margin is bought with RESOLUTION because tau = 1/2 + 6 u_t a / Re,
  and that AMR is the stability mechanism rather than an optimization.
  Accuracy now says the same thing for the same reason: `tauAtLevel` doubles
  (tau - 1/2) per rung, so a body on a refined level has a proportionally more
  accurate moving-body coupling as well as a more stable one.

- **A PINNED BODY AT LOW tau IS FINE, which is why nothing caught this** --
  sphere-Re200-D16 runs at tau = 0.512 and sits +7.0% against
  Schiller-Naumann, because its wall-position error is a FIXED offset that
  lands inside the standing staircase allowance. It is the MOVING coupling
  that degrades, and until D1 nothing in this suite moved a body and measured
  its drag. `tow_amr_cases` now gates the frame agreement at tau = 0.6 (towed
  within 3% of pinned, measured 1.0%) and pins the tau = 0.509 number as a
  regression marker.

- **ENSTROPHY IS THE FRAME-INVARIANT DISCRIMINATOR, and it is what separated
  "the flow is wrong" from "the reading is wrong".** Vorticity does not see a
  uniform velocity shift, so two Galilean frames of one flow must report the
  same enstrophy whatever their Cd says. They match to 0.3% on both paths
  while Cd differs by 19% and 47% -- which retired the fresh-node refill as a
  suspect in one measurement, after a whole entry had been written blaming it.
  Reach for it before theorizing about a moving-body coupling.

- **RULED OUT AND DERIVED, so it is not re-litigated: the wall density in
  Ladd's correction.** The momentum-exchange force is Galilean invariant to
  FIRST order in the frame velocity iff the correction carries the LOCAL
  density rather than a pinned 1 (the two halves of the sum shift by
  `+SUM 2 w_i rho e_i(e_i.V)/cs^2` and `-SUM 2 w_i rho_w e_i(e_i.V)/cs^2`, and
  cancel exactly when rho_w = rho). All four kernels now do that
  (`RHO_W_LOCAL`, `?rhow=0` to A/B, identically zero on a pinned body). **It
  is worth 0.04%** -- front (rho > 1) and rear (rho < 1) cancel over a closed
  surface. Kept because it is the correct formula and costs nothing, NOT
  because it fixed anything.

- **FOR A BODY THAT MOVES AT LOW tau, THE DIFFUSE COUPLING IS FRAME-CONSISTENT
  AND NOISELESS AND BOUNCE-BACK IS NEITHER** (plans/3D.md D4). Measured on the
  Galilean split at D = 12, Re = 100, tau = 0.509 -- the configuration where
  bounce-back is worst:

      coupling        pinned    a=1      frame gap   force rms/mean
      bounce-back     1.3951   1.1373      -18.5%         295%
      diffuse (chi)   5.0433   5.1419       +2.0%          0.5%

  Nine times better on the frame and six hundred on the noise, AND PREDICTED:
  momentum exchange is a raw sum of individual populations over a SUBSET of
  directions, which is not a moment, so ghost content enters it; the diffuse
  penalty reads `rho` and `u*`, which ARE moments and which ghosts are
  orthogonal to. **The price is that the body is the wrong SIZE** -- diffuse
  Cd is +362% over Schiller-Naumann here, an effective D of ~26 for a nominal
  12. It is a REGULARIZATION, not a wrong model: Brinkman penalization
  converges to no-slip as the band narrows, so "convergent and not converged"
  is the accurate phrase, not "non-physical".

- **CHI CARRIES THE FALLING CARD, AND AT A NARROW BAND IT REPRODUCES
  BOUNCE-BACK'S TERMINAL VELOCITY TO 0.7%** (plans/3D.md D4). Re = 500,
  n = 32, 26000 steps: `v_x` = 0.0316 / 0.0358 / 0.0428 at band 1.5 / 0.75 /
  0.375 coarse cells against bounce-back's 0.0425, with the free-fall balance
  |f|/mg at 1.00 throughout and neither coupling tumbling. The 22% slower fall
  at the DEFAULT band is the fat body quantitatively -- v ~ 1/sqrt(Cd), and the
  pinned-plate ladder predicts 0.79 against a measured 0.78 -- not a second
  defect. **0.375 coarse cells is what `?levels=3` gives for free at the
  default `chiEps`**, because the band is a fixed number of cells at the body's
  own level. chi's density field is also 5x quieter (+-0.3% against +-1.7%),
  which matters because rho excursions are what precede a blowup.

- **THE COUPLING DOES NOT MOVE THE Re = 500 STABILITY WALL, AND THAT IS THE
  TAXONOMY BEARING OUT.** Four ways to 26000-30000 steps on the card: today's
  default clean at 26000, a pre-D1 build (`?swept=0&rhow=0`) dead at ~25000,
  `?swept=0` clean at 30000, chi at band 0.375 clean at 26000. **The
  divergence STEP is a single draw good to a factor of two and is not a
  discriminator** -- do NOT read the one death as "D1's fix extended the
  card's life". What IS established: the blowup is the FIRST tau limit (ghost
  modes undamped as omega -> 2), a property of the BULK collision operator, so
  no boundary treatment touches it. chi buys the moving-boundary limit and the
  noise; only TRT/MRT buys the NaNs. See plans/TRT.md.

- **THE DIFFUSE BODY CONVERGES AT SECOND ORDER IN THE BAND WIDTH, TO THE
  NUMBER BOUNCE-BACK GIVES DIRECTLY.** Measured, D = 12, Re = 100, pinned,
  band in coarse cells 1.5 / 0.75 / 0.375: Cd 5.080 / 2.413 / 1.721, a
  difference ratio of 3.85 against 4 for halving. Richardson to zero band
  gives 1.478 (+35% over Schiller-Naumann) against BOUNCE-BACK's 1.3951
  (+28%) on the same body. So chi at best MATCHES bounce-back's accuracy and
  its whole advantage is the moving case -- and the two converging to one
  answer is the check that says it regularizes the same problem rather than
  solving a different one. **The frame consistency survives sharpening**
  (-0.2% at depth 3, -1.9% on a sub-cell dense band), which was the thing most
  at risk since a narrower band is a stiffer force. **At a FIXED physical band
  refining is worse** (2.017 at depth 3 against 1.721 dense), because a
  sub-cell dense band makes `tanh` nearly a step and is quietly approximating
  bounce-back. So the shipping combination is BOTH knobs: refine so the band
  is several cells wide at the body's own level (smooth, quiet,
  frame-consistent) while its PHYSICAL width is small (a body near the right
  size).

- **THE CHI BAND IS A FIXED NUMBER OF CELLS AT THE LEVEL THAT RESOLVES THE
  BODY, so its PHYSICAL width halves per rung of refinement** (`CHI_SCALE =
  2^-m`, with `phi` in L0 units at every level): 1.5 coarse units dense, 0.75
  at depth 2, 0.375 at depth 3. So AMR shrinks the diffuse coupling's only
  real error automatically, at the same time as it doubles (tau - 1/2) for
  bounce-back's. **That is why the coupling choice is live again**: 2D chose
  chi because a phone was resolution-starved and a fat body was the lesser
  evil, 3D chose bounce-back because it is sharp and second-order, and neither
  argument survives AMR unchanged. `common_d3_amr_step1.wgsl` used to claim
  the scaling preserved a constant PHYSICAL width, which is the opposite of
  what it does; corrected 2026-09-11.

- **Bounce-back is the accurate solid coupling in 3D FOR A PINNED BODY;
  diffuse (chi) is
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

- **`?interface=explode` is the DEFAULT as of M4.1e (2026-09-10), and the
  3D AMR coarse/fine interface is conservative.** Mass and momentum both, on
  every rung of the geometry ladder (`?refine=all|slab|bar|box`), with a
  field error that tracks the no-interface control to within 8-10% and a
  sphere-in-a-refined-shell that reproduces the dense Cd to 0.07%. A
  partially-refined 3D run is now trustworthy for a quantitative number,
  which it was not before M4.
  `?interface=interp` is the M3 coupling it replaced — NOT conservative,
  first-order, 3.6x worse on the field (whole-domain L2rel 1.36e-2 /
  9.02e-3 / 7.04e-3 at N = 32/48/64 against controls at 4.6e-3 / 2.3e-3 /
  2.0e-3) — kept switchable only to A/B the conversion in one build.
  `amr-box-RB4` is PINNED to it and must stay pinned: it is the control that
  says M4 left the old path alone, and a control that tracks the default is
  not a control.

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
- **THE MOVING WINDOW IS A TRANSLATION OF THE SPONGE, NOT OF THE FIELD**
  (M8.3, `?window=x|xyz|0`). The 2D pages make every kernel's dispatch index
  a WINDOW coordinate and convert to a buffer index at every load and store
  (`off_x`/`off_y`); in 3D that would be a window conversion in the dense
  step, the pool step, moments, interp, average, explode, coalesce, both
  force kernels and the tree sampler. So it is read the other way: the buffer
  is periodic and **THE FLUID NEVER MOVES**, the BODY wraps through it, and
  the only thing that has to follow the body is the ABSORBING BAND -- both
  step kernels convert a buffer position into a window coordinate before
  measuring its distance to a window face. Every addressing path, every tile
  and the whole interface are untouched, because none of them ever asked
  where the body was; the resample box's origin needed nothing at all.
  `d3-window.mjs` and `shaders/common_d3_window.wgsl` are the two statements
  of the convention and `make test` gates them against each other. Three
  things to know:
  - **The anchor is the body's INITIAL cell**, so the offset is exactly 0 at
    step 0 and a windowed run starts bit-identical to an unwindowed one --
    which is what makes `?window=0` a control rather than a different
    experiment. Refused rather than degraded: no body, a walled axis, or no
    sponge (without an absorbing band the wake wraps round and the body flies
    back into it, which looks perfectly healthy).
  - **The VIEW's offset is continuous where the SOLVER's is integer.** The
    solver floors so the sponge band stays cell-aligned; a viewer that
    inherited the floor leaves the body sawing across one cell at the exact
    frequency of its own cell crossings. The slice view reads the body buffer
    directly for the same class of reason -- a host-written offset is stale
    between readbacks and the picture slides forward and snaps back.
  - **SAMPLING A MOVING BODY'S FORCE ONCE PER CELL IT CROSSES MEASURES ONE
    PHASE OF THE STAIRCASE, NOT THE MEAN.** The force oscillates with a
    period of exactly 1/U steps and an amplitude comparable to the drag, so
    the obvious sampling rate IS the alias. It produced a smooth, reproducible
    Cd "collapse" from 1.24 to 0.30 over a run whose kinetic energy,
    enstrophy, max|u| and density range were constant to four digits.
    `tools/probe-d3-window.js` samples every step; `--sample=25` reproduces
    the trap. The tow is reproducible to 0.2% on a same-build repeat, and the
    same fall runs 40000 steps and 1681 cells through a 192-cell domain
    without leaving.
  - **Cd(tow) IS BELOW Cd(stream), AND THE GAP IS RESOLUTION.** -29.5% /
    -25.2% / -17.3% / -13.9% at D = 12 / 16 / 24 / 32 at FIXED 2.18%
    blockage, closing from BOTH sides (tow rising, stream falling) at roughly
    the first-order rate a staircased bounce-back sphere converges at. That
    is the discriminator: a missing momentum term in the moving-body coupling
    would go as A*U^2 against a drag of Cd*(1/2)*U^2*A and hold the RATIO
    fixed. Pinned, the staircase is one fixed shape (hence this suite's
    standing +9..13% over Schiller-Naumann); towed, its phase sweeps and the
    effective surface is smoothed. **Both frames extrapolate to Cd = 1.22** (a
    first-order Richardson on the tow gives 1.220/1.226/1.223 and the
    stream's 16-24 pair 1.224), which is +11.7% over Schiller-Naumann -- the
    suite's own pinned-sphere offset. So the gap goes to ZERO, not to a
    floor, and the D = 12 number is not a coupling defect: D = 12 is coarser
    than every other case here. ~8% residual at the target's D = 64.

      node tools/probe-d3-window.js                  # the Galilean pair + the no-window control
      node tools/probe-d3-window.js --legs=tow --td=60

  A windowed page is otherwise visited by nothing in the standing suite, so
  `validate-all.js` boot-smokes one as `d3-window-boot`.

- **EVERY SHARED STAGING BUFFER MUST BE SERIALIZED, AND THE SERIALIZATION
  BELONGS AT THE BUFFER** (`serializedOn` in `main-3d.js`). A reader submits a
  copy into the staging buffer and then awaits `mapAsync`; a second reader
  entering that gap encodes a copy into a buffer that is pending map, which is
  a validation error -- "Buffer (unlabeled) used in submit while pending map"
  -- and it takes the DEVICE with it. The callers cannot own this because they
  are a mix: the sync-step paths AWAIT their readback, while the frame loop
  fires `checkPoolExhausted` and `refreshVolumeBoxes` off the 250 ms status
  cadence deliberately UNAWAITED. Only the buffer sees both.
  - **IT IS A DEPTH BUG, WHICH IS WHY `?levels=2` NEVER SHOWED IT.** Each of
    those refreshes does one readback PER REFINED LEVEL, so depth 3 doubles
    the window, and `mapAsync` resolves only once the queue drains -- so a
    frame that resamples and marches a 4x-resolution L2 volume can outlast the
    cadence on its own. The two conditions arrive together. Measured on
    `card` at `?levels=3`: **30 map-race errors in 13 s and a dead device**,
    against 0 over 90 s with the fix.
  - **CHAIN AT THE BUFFER, DROP AT THE CALLER.** An awaited caller wants an
    answer, so the buffer QUEUES; the cadence callers poll a current value and
    are single-flighted, so a refresh slower than the cadence cannot
    accumulate one pending call per tick forever.
  - **IT HAD ALREADY BEEN FIXED ONCE, FOR `bodyStaging` ONLY**, under a
    comment claiming the chain "covers every future caller". It covered that
    one buffer. `readBody` now goes through `serializedOn` too, so there is
    one mechanism rather than a third copy waiting to be written.
  - **A DEBUGSTEPSYNC TOOL CANNOT REPRODUCE IT**, because it awaits every
    readback -- the race lives only on the rAF path. It takes a LIVE page and
    real elapsed wall time, which is a shape nothing in the standing suite
    has.

- **THE VOLUME VIEW IS ONE DENSE VOLUME PER LEVEL AND THE INNERMOST BOX WINS
  -- no per-ray tree descent** (`?view=volume`, M6.3/M6.4). sec 1.3's "do not
  raymarch the octree pool, it is a research project" stays true OF THE POOL;
  it is not true of this, and the two had been conflated. A level's refined
  set is ONE COMPACT REGION because refinement is geometry-forced, so its
  bounding box is small -- measured 24^3 L0 cells out of 192x128x128 -- and
  three dense volumes over three such boxes are 98 MB against **2.25 GB** for
  one uniform volume at the finest resolution. A ray sample is then 2-3 box
  tests with the LAST hit winning, and it STEPS by the innermost box's voxel
  size, so the near-wall sheet costs steps only where it is.
  - **`?vol=N` is a MULTIPLE OF EACH LEVEL'S OWN RESOLUTION**, not voxels per
    L0 cell -- level m already has 2^m cells per L0 cell, so the other
    phrasing means three things in a three-level run. It meant them until a
    gate asked for 4 at level 2 and got sixteen.
  - **The scalar volume is `rgba16float` and every channel is NORMALIZED**
    (`Q/qRef`, `|omega|/vRef`, `|u|/uRef`, `rho - 1`). Raw Q is ~1e-5, BELOW
    binary16's smallest normal; dividing by the reference the criterion
    already uses puts it at order 1 and removes the `float32-filterable`
    dependency the mobile target may not have. One pass fills all four, so
    `?volfield=` is a uniform write.
  - **The tone curve is Reinhard, for `common_vortcolor.wgsl`'s reason.** A
    clip makes the strongest structure's front face opaque and the render
    degenerates into an isosurface of the iso value: measured on a Re = 300
    sphere, near-wall Q is >100x the wake's, and clipped the wake vanished.
  - **A CAMERA IS ONLY CHECKABLE THROUGH THE BODY'S SILHOUETTE.** "Is it
    pointing at the interesting part" has no reference value; with the volume
    switched off (`?volIso=` above every value) the render is the body alone
    and the host predicts it by ray/sphere -- measured IoU 1.000.
    `tools/validate-d3-raymarch.js` is the gate (M6.2's Q against
    `d3-criterion.mjs` on the same fp16 texels, M6.3's transfer function,
    M6.4b's image difference against `?volstack=0` with `d3-volume.mjs`
    saying independently which pixels may differ: 234 differ, 0 outside).
  - **THE REFINED BOXES ONLY FOLLOW THE BODY FROM THE LIVE FRAME LOOP, and
    before 2026-09-12 the OFFSCREEN path did not refresh them at all.**
    `refreshVolumeBoxes` is driven by the status cadence; `debugRenderFrame`
    and `debugReadVolume` run with `live=0` and no frame loop, so both now
    await it themselves. **Every clip `tools/render-d3-movie.js` filmed at
    ?levels>=2 with ?dynamic=1 had refined volumes pinned to step 0**:
    measured, the card's body travelled (128,80,96) -> (180,26,96) over 2000
    steps while both box origins sat unmoved at their initial values.
    **A STALE BOX IS WORSE THAN NO BOX**, because the stack rule is INNERMOST
    BOX WINS -- it does not fall back to L0, it OVERRIDES L0 over its whole
    extent with coarse data replicated onto a fine grid.
    **BUT IT IS NOT WHAT A LIVE PAGE SHOWS**, and the commit that fixed it
    said otherwise: the frame loop has refreshed the boxes on its 250 ms
    cadence since M6.4a, and `debugRenderFrame`/`debugReadVolume` are reachable
    only from `window.__D3`. So this is a MOVIE and TOOL defect, whole and
    entire. The live speckle is the next entry, which that commit diagnosed
    correctly and did not change. (Its claim to be quoting an M6.4a
    "known-missing gate" is also withdrawn -- no such line has ever been in
    the repo; a history search finds the phrase only in the commit that
    quotes it.)
  - **A VOXEL THAT READS A COARSER LEVEL THAN THE VOLUME IT IS STORED IN IS
    WHERE THE SPECKLE IS** (`tools/probe-d3-volume-crunch.js`). A level's
    volume is a dense grid over a BOUNDING BOX while the refined set inside it
    is not box-shaped, so those voxels get coarse data replicated onto a fine
    grid -- and Q is a squared velocity GRADIENT, so replication differentiates
    into stipple. Measured on the card at ?levels=3, normalized |lap Q| on the
    two populations: **2.52x (dynamic) and 8.14x (static)**, with **48-63% of
    structured voxels in the box reading a coarser level**. That is the
    dominant population, not an edge effect. Measure it in the VOLUME, never in
    the image: an image statistic cannot separate a speckled volume from a
    raymarcher undersampling a clean one, and M6.5a is the standing lesson on
    trusting image statistics here.
  - **AND THE AMPLIFIER IS THE GRADIENT PASS'S FIXED STENCIL** (M6.4d,
    2026-09-12, `?volh=0` to A/B). Replication alone is only a plateau; what
    turns it into stipple is that `d3_volume_scalar.wgsl` differenced EVERY
    voxel over one voxel, `VOL_H` being a pipeline constant for the whole
    texture. The resample samples the tree NEAREST, so a source cell's voxels
    are BIT-IDENTICAL -- the difference is exactly zero across the plateau and
    the entire source-cell jump across the one voxel at its edge, over a step
    2^(level-src) too small. So the staircase is differentiated, and Q, being
    a squared gradient, squares the result.
    **This is `common_d3_tree_sample.wgsl`'s OWN standing rule** -- "any finite
    difference taken from it must use the sampler's own `h`" -- which this pass
    structurally could not obey, because the resample threw `TreeSample.level`
    away. It now writes it to a companion `rgba8uint` volume and the stencil
    reaches ONE SOURCE CELL: `stride = max(1, round(2^-src / h))` voxels,
    stated in `d3-volume.mjs`'s `stencilStride` and mirrored in `strideFor`.
    **IT IS BIT-IDENTICAL WHERE THE VOLUME ALREADY HAS THE DATA** (stride 1 at
    `?vol=1`), so M6.4b's sharp near-wall sheet is untouched and only the
    replicated population moves -- gated exactly that way, not to a tolerance,
    by `validate-d3-raymarch.js --cases=volh`: 0 of 4096 at-level voxels
    differ across the A/B, 5111 of 5120 replicated ones do.
    **WORTH 4.9x ON THE CARD** (`probe-d3-volume-crunch.js --legs=full,fixedh`,
    n=32 `?levels=3`, 1000 steps): normalized |lap Q| on the replicated voxels
    falls 0.4695 -> 0.0959 while the at-level ones hold at 0.1328 -> 0.1318,
    with `boxRatio` 1.884 and `inUse` 6192 IDENTICAL across the pair. The
    replicated-over-at-level ratio inverts, 3.54 -> 0.73: coarse data is now
    SMOOTHER than refined data, which is what it should be.
    **THE COMPANION IS PAID FOR BY THE SMALL BOXES ONLY.** 4 B/voxel, and the
    L0 volume gets a 1x1x1 dummy with `WRITE_LEVEL`/`VOL_HSRC` folded to 0 --
    its own spacing is already the coarsest in the tree, so its stride is 1 by
    construction. r8uint is not storage-writable in core WebGPU (the same wall
    r16float hit), so 4 B is the floor. It is counted in `?volBudget=`: an
    allocation the budget cannot see is one that OOMs instead of refusing.
  - **A REFINED BOX'S PLACEMENT LIVES IN TWO BUFFERS AND THEY MUST MOVE
    TOGETHER** (M6.4e, 2026-09-13, `?volsync=0` to A/B). The resample's
    `originBuf` says what is written INTO a box's texture; the ray params'
    `boxLo` says where the raymarcher reads it back OUT of world space.
    `refreshVolumeBoxesNow` used to read and apply ONE LEVEL AT A TIME, so it
    wrote L1's origin and then `await readBlockSlot(2)` yielded to the event
    loop -- and every frame drawn during that readback showed L1's freshly
    resampled contents at L1's STALE position. The picture shifts EN MASSE by
    whole cells and snaps back when the refresh lands. Every readback now
    happens before every write, with no await between the first write and the
    publish.
    **MEASURED, card `?levels=3`, 150 s a leg** (`tools/probe-d3-volbox-sync.js`):
    **710 of 4357 rendered frames -- 16.3% -- against 0**, max gap 362 ms
    against 0.2 ms, at the same update count (125/132) and the same frame
    count. Free.
    **A DEPTH BUG, like `serializedOn`'s**: at `?levels=2` the loop runs once
    and the only await is BEFORE the only write, so no gap exists to observe.
    **AND IT LOOKS LIKE A FAR-FIELD FAULT because the INNERMOST box is applied
    LAST**, with only synchronous code between its write and the publish -- so
    the level covering the plate is always in register and the one that is not
    is L1, which at `?volMargin=4` spans 80x72x112 of a 192x160x192 domain.
    "Near the body is fine, further out shifts" is the signature of the apply
    ORDER, not of anything physical.
    **NOTHING IN THE STANDING SUITE CAN SEE IT**, for the reason `serializedOn`
    records: it lives between an unawaited refresh and the frame loop, and a
    `debugStepSync` tool has no frame loop. `debugVolBoxSync()` keeps the count
    on the page so a live run can be asked.
  - **A REFINED BOX IS ONLY USED WHERE ITS VOXEL CAME FROM ITS OWN LEVEL**
    (M6.4f, 2026-09-13, `?volfallback=0` to A/B). M6.4d fixed what replication
    did to the GRADIENT; this is the other half. `levelAt` took the innermost
    containing box unconditionally, so across the majority of a box that its
    refined set does not cover -- 58% of structured voxels measured, more again
    at `?volMargin=4` -- the ray was handed a coarser level's value replicated
    onto a fine grid, and then sampled it at the FINE voxel size. It now
    descends and takes the first box that both contains the point AND owns its
    data, falling through to L0.
    **9.36% of pixels change, worst 37/255**, so it is not cosmetic.
    Descending rather than scanning is what keeps it to one companion fetch in
    the common case; falling back also coarsens `dt`, so the guess was that it
    would pay for itself. COST: see `debugBenchRender` below.
    **NEVER PRICE A SHADER ON THE ANIMATION THREAD.** Three attempts at this
    number went wrong the same way. A frame rate read off the rAF loop is the
    DISPLAY's: 1152x720 and 2560x1440 BOTH read 59.9 fps on both legs, because
    the frame costs under 16.6 ms either way, and "equal" was really "capped".
    Disabling vsync separates the legs but puts the browser's frame pacing in
    the number instead. `debugBenchRender` is the instrument -- N draws in ONE
    command buffer, one submit, `onSubmittedWorkDone`, no presentation and no
    readback (debugRenderFrame's copy and map cost more than the draw), minimum
    of its batches as `bench-d3-interface.js` does. A build-vs-build claim
    still needs a same-build repeat.
    **`debugRenderFrame` IS ALSO OFF THE ANIMATION THREAD AND STILL CANNOT DO
    IT: 1181 ms per call at 1024^2 against a 4.7 ms draw** -- base64 of 4 MiB
    and the CDP round trip, 250x the thing being timed. Off-thread is
    necessary and not sufficient.
    **AND THE COST IS BELOW THE FLOOR.** A same-build repeat differs by 6.2%,
    the A/B by 3.9% -- with the SIGN not even stable across instruments -- so
    the honest statement is |cost| < ~6% of a 1024^2 render, at four times the
    live view's pixel count, against a change worth 9.36% of the picture.
    Tightening it means A/B-ing both pipelines inside ONE page; the
    fresh-Chrome-per-leg shape is what the 6% is.
  - **`boxRatio` WAS READING THE SEAM CROSSING AS GEOMETRY, and the "flat
    plates are pathological for boxes" claim built on it is RETRACTED**
    (2026-09-12). `poolStateAt`'s bbox was a plain min..max in BUFFER
    coordinates; with a moving window the shell straddles the seam twice a lap
    and min..max is then nearly the whole axis. It now uses the same
    `axisSpan` the volume boxes have used since M6.4. Measured on the card at
    L2: **12.29 at a straddling step, ~2.0 away from one** -- so the statistic
    was mostly reporting where in its lap the body happened to be, and the
    honest number is comfortably UNDER M6.4c's ~4x threshold. A wrapped span
    has `hi` BELOW `lo`, so `bbox` now carries `ext` explicitly and
    `d3-volume.mjs`'s `boxRatio` prefers it; feeding it a wrapped span with no
    `ext` is REFUSED rather than silently negated, and `make test` gates that.
    Same class as M6.5a: a quantity that is fine in buffer coordinates and
    meaningless in window ones.
  - **THE VOLUME'S EXTENT IS FIXED AT CREATION AND A ROTATING BODY OUTGROWS
    IT** -- `?volMargin=` is slack added ONCE, to the step-0 box, and only the
    ORIGIN moves thereafter (a texture has a size). A plate turning edge-on to
    broadside swings its AXIS-ALIGNED bbox by far more than a sphere's, so the
    set walks out of a box that was correctly sized at step 0. Measured on the
    card at ?levels=3, L2 y-extent against its volume: 0.80 at rest, **1.20 at
    peak |omega|**, and the overflow tracks |omega| rather than the step
    number. The box then CLIPS the refined region and the render shows a hard
    rectangular boundary -- which is the OPPOSITE geometry to the speckle
    above (box wider than set) and a different artifact. `?volMargin=4` covers
    the measured peak on this case (`?volBudget=512` with it); the principled
    fix, unrun, is to size a rotating body's box from its BOUNDING SPHERE
    (2*sqrt(a^2+b^2+c^2)) rather than from one pose.
  - **`debugPoolState` reports `boxRatio` per level** (M6.4c) -- box-union
    over refined-set volume, 1.04x/1.14x on a geometry-forced shell against
    the ~4x at which boxes stop paying. M8.4 already measured **4.6** on a
    wake-following criterion, so the successor is measured and waiting.

- **THE TREE SAMPLER (`shaders/common_d3_tree_sample.wgsl`) IS THE ONLY
  THING HERE THAT SPANS LEVELS**, and it is what the slice view now samples
  (M6.0b). Every solver kernel is written for one level and reaches its
  parent through `common_d3_parent_{dense,pool}.wgsl`; a viewer cannot, since
  "the best answer available here" is a question about the whole tree.
  Its frame is forced, not chosen: `fineToCoarseUnit` puts cell centres at
  INTEGERS, so `g(p, m) = floor((p + 1/2) * 2^m)` and ownership is one
  division by `2*RB` -- which is why a viewer can never land in a tile's
  RING. `d3-amr.mjs`'s `cellAtLevel`/`finestLevelAt` are the host statement
  of the same rule; `make test` mutation-checks them and
  `debugCheckTreeSample` scores the real shader against them on GPU data.
  **Any finite difference taken from it must use the sampler's own `h`** --
  a fixed L0 step discards the resolution, a fixed finest step paints the
  whole coarse field flat zero.

- **A POOL LEVEL'S `mac` IS SEEDED AT RESET** (M6.0a,
  `common_d3_moments.wgsl`). It is otherwise a derived buffer -- written by
  the step and by the child's coalesce -- so before that it read as zero at
  step 0 on every level, which a viewer renders as a hole and a check has to
  special-case.

- **`main-3d.js` NEEDS 12 STORAGE BUFFERS PER SHADER STAGE**, past the spec
  minimum of 8, because the tree sampler binds the dense `mac` plus a
  (`mac`, `blockSlot`) pair per level and its probe adds two more. Requested
  with a loud failure, the treatment `main-amr.js` already settled on. A
  bind-group-layout failure is NOT an exception: it poisons every pipeline
  built on it and the only symptom is `Invalid PipelineLayout is invalid due
  to a previous error`.

- `tools/probe-d3-galilean.js` — the moving-body coupling measured against
  ITSELF (D1). Sweeps the tow/stream split at a fixed relative speed and
  reports Cd against the split, plus an optional MOMENTUM BUDGET leg in a
  periodic sponge-free box (`?scenario=drift`) where the fluid's momentum can
  only change through the body, so `sum F` must equal `-dP` and the residual
  is the force the kernel is not measuring. Reports, does not gate.

      node tools/probe-d3-galilean.js                     # the sphere sweep
      node tools/probe-d3-galilean.js --budget            # the mechanism
      node tools/probe-d3-galilean.js --scenario=card --n=16 --re=200 \
           "--scenarioExtra=tilt=0&span=1&aspect=0.125&u_t=0.05"

  Two traps it now documents from having hit them: the budget's first version
  was 80% its own INITIAL CONDITION (SOLID_EQ filling the body interior on
  step 1 is `|S| * u_body` of momentum appearing at once), and `drift` is a
  body pumping momentum into a CLOSED BOX, so a long window does not settle,
  it diverges -- the field's max|u| and density range are printed beside the
  closure for that reason.

- **A CLIP IS RENDERED, NOT SCREEN-RECORDED** (`tools/render-d3-movie.js`,
  M6.5). It drives the page a declared number of steps per frame, renders each
  frame OFFSCREEN through `debugRenderFrame` (either view), and pipes raw RGBA into
  ffmpeg. A recorder would film the SOLVER'S frame rate, so an expensive
  configuration's clip runs slow and two builds are not comparable; here
  `--seconds`/`--fps` describe the FILM and `--steps` the FLOW.
  - **THE RAY'S MARCH INTERVAL IS IN WINDOW COORDINATES, NOT BUFFER ONES**
    (M6.5a, `?winbox=0` to A/B). The body wraps through the periodic buffer
    and its wake does not, so a march bounded by [0, N) stops at the seam and
    the picture COLLAPSES the moment the body crosses it -- 8.22% of the frame
    lit against 38.45%, measured at buffer x = 0. Gated as an invariance
    (`--cases=window`): the lit fraction must not jump between samples 200
    steps apart. 1.00x fixed, 1.84x with `?winbox=0`.
  - **`?proj=ortho` IS THE INSTRUMENT; PERSPECTIVE IS THE PICTURE** (M6.5b).
    Under a parallel projection with a known scale the image is an AFFINE MAP
    OF THE LATTICE: `orthoScale` turns pixels into L0 cells with one division,
    a plane in the flow is a straight line at a computable row, and a sphere's
    outline is a circle rather than a conic. Reach for it before reasoning
    about what a picture shows -- it turned "8.22% of the frame against
    38.45%" into "the cut is at buffer x = -0.3 +- 0.25 and the box face is at
    -0.5". The two projections frame identically at the target plane, so
    switching is a comparison and not a different experiment.
  - **AND IT IS INVISIBLE UNLESS YOU MEASURE AT THE SEAM.** Away from it the
    cut plane falls outside the frame and an A/B reads as 34 pixels in
    160,000. Two measurements said "no effect" before one said 4.7x: the other
    was an image statistic pinned at row 0, which is CENSORED by the frame
    edge, not constant. Test where the body's buffer position is near 0 or N.
  - **IT REPORTS THE TRAJECTORY, and `INT |omega| dt` IS NOT THE ROTATION.**
    That integral is ARC LENGTH -- a plate rocking back and forth accumulates
    it without ever turning over, and the first version duly called the
    default card "0.44 revolutions". It now reports `net = |INT omega dt|`
    beside `arc = INT |omega| dt`: tumbling is net ~ arc, fluttering is
    arc >> net.
  - **THE CARD FLUTTERS, IT DOES NOT TUMBLE**, and five configurations say
    so (n = 32, 26000 steps, chi). net/arc revolutions: 0.04/0.44 at the
    default, 0.12/0.76 at tilt 0.7, 0.10/1.03 at `?levels=2`, 0.09/0.91 at
    span 2, 0.07/1.80 at the TARGET Re = 1100, and **0.18/2.41 at Re = 1100
    with a 2:1 plate** -- the closest any of them comes. `arc` grows 5x while
    `net` never reaches a fifth of a turn, which is the regime being wrong
    rather than the initial condition. Re = 1100 SURVIVING 26000 steps at
    `?levels=2` is itself new, and is M8.0's "AMR is the stability mechanism"
    holding.
  - **SIX LEVERS, ONE ANSWER, AND EVERY ONE MOVES `arc` NOT `net`** -- span
    1->2, Re 500->1100, release 0.15->1.047 rad, I* 0.17->0.68, chi band
    0.75->0.375, and the plate pre-thinned so its EFFECTIVE thickness is the
    nominal 4 cells. Arc rises 40% across the thickness axis; net max rises
    20% to 0.367 with DECELERATING increments, converging on ~0.4 rather than
    a revolution. **`?bounceback=1` is not available at tau = 0.50436** -- it
    blew up at step 5000 with |v| pinned at v_max -- so an exact-thickness
    body and Re = 1100 are currently mutually exclusive here.
  - **THE TWO SUSPECTS LEFT ARE BOTH THE DOMAIN, and plans/3D.md M6.5 lists
    five avenues with costs.** (a) **NARROW DOMAINS DAMP**, and `card`'s
    free-fall box is narrow: blockage **5.00% at span 1, 6.67% at span 2,
    8.00% at span 4**, against the 1.56% its own prescribed legs use -- and
    the scenario header already says 5% carries a wall correction of order
    +12%. **This confounds the span result**: going 1 -> 2 raised net max AND
    tightened the box, so the true span effect is larger than measured.
    Widen before sweeping span again. (b) **TIP VORTICES** -- P&W is 2D
    (infinite span, no tips) and a finite plate sheds them; the domain is
    already periodic, so a plate whose span exactly FILLS z has none, but
    `card`'s dims put z = 2*span*n + 2n so the gap is built in. Both need a
    few lines in `d3-scenarios.mjs`, and they pull OPPOSITE ways on the same
    axis (widen z for clearance, fill z to remove tips) -- two experiments,
    not one knob.
  - **FOUR AXES SWEPT, ONE ANSWER** -- span 1->2, Re 500->1100, release angle
    0.15->1.047 rad, and I* 0.17->0.68 (`tools/probe-d3-tumble.js`). A
    FOURFOLD change in inertia moves the peak net rotation 0.306 -> 0.361,
    18%, and nothing crosses. **The suspect is now the plate's effective
    THICKNESS, not a parameter**: the card is 4 cells thick at n = 32, the chi
    band adds 0.75 coarse cells per side at `?levels=2`, so its aspect is
    ~0.17 against a nominal 0.125. Aspect is a primary parameter of that
    boundary. `?bounceback=1` (sharp, nominal thickness) and `?levels=3`
    (band down to 0.375 cells) are the two cheap tests. UNRUN.
  - **FLUTTER IS AN ATTRACTOR: THE INITIAL CONDITION IS NOT THE OBSTACLE.**
    Released near VERTICAL (`?tilt=1.047`, 30 degrees off edge-on) the 2:1
    plate at Re = 1100 swoops IMMEDIATELY -- net/arc = **1.00** over the first
    2000 steps, 79 degrees all one way, more net rotation than any other case
    reaches in 26000 -- and then relaxes onto the same bounded limit cycle,
    `net` oscillating in 0.05..0.31 while `arc` climbs linearly to 2.74. The
    same attractor is reached from `tilt` 0.15 and from 1.047. So the release
    angle is settled and `i_star` is the axis left.
  - **`net` IS A WANDERING QUANTITY UNDER FLUTTER**, so read the SERIES, not
    the end: 0.120 final against a 0.306 maximum on the same run. Tumbling is
    `net` growing without bound, which a single end-of-run number cannot show.
  - **THE 2:1 PLATE GLIDES, AND THAT IS THE REAL SIGNAL**: v = (0.024,
    -0.022, 0.009), a lateral speed nearly equal to its descent, against the
    square plate's (0.031, 0.003, 0.003) straight down. Fluttering WITH GLIDE
    sits immediately next to tumbling in Pesavento & Wang. Span and Re have
    both now been swept with the same answer, so the untried axis is
    `i_star` -- 0.17 throughout, and the boundary in that diagram is a curve
    in I*-Re rather than a Reynolds threshold.

      node tools/render-d3-movie.js                      # the card, 30 s
      node tools/render-d3-movie.js --view=slice --seconds=10 --fps=24

- `tools/validate-d3-raymarch.js` — the volume renderer's gate (M6.2-M6.4b).
  Standalone and opt-in; owns its own Chrome. Three claims, none of them "the
  picture looks right": the scalar volume is the gradient of the velocity
  volume (host Q via `d3-criterion.mjs` on the SAME fp16 texels), the
  transfer function is normalized to the field (an A/B against an iso raised
  above everything), and the stack changes the picture EXACTLY where a
  refined box covers. `d3-volume.mjs` + `tools/test-d3-volume.js` are the
  host statement of the boxes and the camera, mutation-checked.

      node tools/validate-d3-raymarch.js
      node tools/validate-d3-raymarch.js --cases=stack --size=256

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
