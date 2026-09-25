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

**THERE IS NO DENSE LEVEL 0 ON AN AMR PAGE** (plans/uniform-levels.md U7-6f,
2026-09-22). Level 0 is the ROOT POOL: 2\*RB-square tiles, RINGLESS
(`ghostDepthAtLevel(0)` is 0 — a ring holds a parent interface and the root has
no parent), indirection the identity, always full. It runs the same
`amr_step1.wgsl` every other level runs, is allocated in quads by the same
`amr_manage_pool.wgsl`, and is addressed by `amr2d.mjs`'s `rootCellIndex`.

`amr_step.wgsl`, `amr_criterion.wgsl`, `amr_manage.wgsl`, `amr_force.wgsl`,
`amr_interp_dense_parent.wgsl`, `amr_average_f2c.wgsl`,
`common_interp_parent_dense.wgsl`, `common_avg_parent_dense.wgsl` and
`amr_mirror_root.wgsl` are GONE, along with `?rootpool=`, `?rootstep=`,
`?rootcouple=`, `?rootmanage=`, `?densel0=` and `?rootIsPool=`. The five
NON-AMR reference pages (`index.html`, `index-cylinder.html`,
`index-reentry.html`, `index-channel.html`, `index-tgv.html`) keep their own
dense grids and `lbm_step.wgsl` — untouched, and still the control the AMR
pages are scored against.

**The block8 layout is still real, and it is theirs.** `amr2d.mjs`'s
`denseCellIndex` owns the rule; it is what `initF()` builds an initial
condition in (the seeder permutes it into root tiles) and what
`tools/lib/dense-to-amr.js` reads a dense capture in. A page that needs both
names them apart — see `main-tgv-amr.js`'s `icCellIndex` vs `cellIndexJS`,
which were one function until they meant two things.

**Snapshots are formatVersion 7** and carry level 0 as `root`. A pre-7 capture
is REFUSED on load rather than read with a missing level 0.

## Validate before committing (no GPU needed)
Run `make check` and make it pass before committing shader/JS changes:
- `make js` — syntax-check every `*.js` and `*.mjs` (needs Node). The
  shared `.mjs` modules were outside this glob until 2026-09-08.
  **It parsed as a SCRIPT until 2026-09-15, which meant it silently checked
  NOTHING in any file using ESM `import`** — i.e. every page entry point. On
  node v26.1.0 `node --check m.js` exits 0 on a file with a real syntax error
  if that file opens with an `import`; only `node --input-type=module --check
  < m.js` reports it, which is what the target does now. A mangled object
  literal in `main.js` passed this gate and wedged `index.html` at
  "initializing..."; the GPU boot smoke was what caught it.
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

**A GPU result is only valid if the BROWSER was healthy** — run
`node tools/check-browser.js --port=<p> --chromeLog=<log>` at launch and read
every tool's `[teardown] browser health:` line and exit code
(`browser-lifecycle.js`'s teardown runs the same check). On 2026-09-23 the debug
Chrome's GPU process crashed mid-session and WebGPU fell back SILENTLY to
SwiftShader (CPU, 10 storage buffers); pages kept booting and producing
plausible numbers. See the webgpu-verify skill's section 0.

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
  `dense-reference` and `amr-N2-diffuse` fail at Re=100 (Cd 1.951 and 1.652
  against 1.35±0.15). That is the diffuse-interface-width issue, not a
  regression. Re-baseline against these numbers rather than assuming a red
  cell is yours.
  **`amr-N2-diffuse` READ 1.642 UNTIL 2026-09-18 AND THE INITIAL CONDITION
  MOVED UNDER IT.** D1-a fixed two defects in `main-cylinder-amr.js`'s
  `resetSim` -- a module-scoped perturbation RNG that was never re-seeded, so
  every reset drew a different field, and velocity/parentSlot buffers reset
  never wrote, so the first refine round after every reset ran against page
  history (plans/uniform-levels.md D1-a). `reset()` is now a bit-exact fixed
  point, which it was not before, and the whole AMR cylinder table shifted:
  N2-diffuse 1.652/0.1484, N2-bounceback 1.356/0.1642, N3-diffuse
  1.473/0.1570, N3-bounceback 1.365/0.1645. `dense-reference` is UNCHANGED at
  1.951/0.1260 -- `main-cylinder.js` seeds its rng inside `initF()` and never
  had either defect, which is why it is the control.
  **`amr-N2-diffuse` READ 1.620 HERE UNTIL 2026-09-15 AND THAT NUMBER WAS
  STALE.** Measured that day on a pristine checkout of the then-HEAD, twice:
  1.642, 1.642 (stable to four digits). Some earlier stage moved it and did
  not re-record it here, so a ~0.02 gap against the old figure is not a
  regression -- take a pristine-tree reading before believing one.
  **AND IT IS NO LONGER OPEN — IT IS MEASURED (2026-09-14, `?kEps=`).** The
  band is `K_EPS * dx_level` and `?kEps=` now sweeps every level at once
  (plans/2D-backport.md B7); the instrument is a BAND ladder at fixed
  resolution, not the resolution ladder this note used to cite (that moves the
  band and everything else together). `dense-reference`, Re=100:

      kEps     Cd      St          kEps 1.5 is the shipped default
      1.5     1.951   0.1260
      0.75    1.617   0.1485
      0.375   1.453   0.1571
      0.1875  1.366   0.1593
      bounce-back (sharp, same body)   Cd 1.327   St 0.1605

  **First order in the band width** — halving it halves the error (ratios 2.04,
  1.89). Note that 3D measured SECOND order for its sphere; it does not
  transfer. So the shipped band inflates Cd by ~47% and depresses St by 21%,
  and both red cells close when it is narrowed: `amr-N2-diffuse` at
  `?kEps=0.375` reads Cd 1.380 / St 0.1643, passing both.
  **The default deliberately stays 1.5.** Every number above is a PINNED
  cylinder, and the page this project ships is a falling card; narrowing the
  band moves toward bounce-back's sharpness, which 3D's D4 measured as worse
  for a MOVING body (frame consistency and force noise). Adopting a narrower
  default needs the moving-card gates, not this ladder.
  **AND THE RENDER'S OWN chi WAS NOT THIS BAND UNTIL 2026-09-22.** B7 threaded
  `?kEps=` into the render precisely so a ladder's picture would not be
  misleading, but `amr_render.wgsl`'s `get_chi` passed `K_EPS` with no `dx`
  factor while the step and force kernels pass `kEps * levelParams.dxL` -- so
  the DRAWN body edge was a fixed 1.5 ROOT cells and the solver's was 1.5
  FINEST-level cells, a factor of `2^(levels-1)`. Fixed (the shader derives
  dx_finest from its existing `N_POOL_LEVELS` override), which makes the
  shipped picture's body edge `2^(levels-1)` times crisper. Any screenshot
  taken before that date shows a halo the solver never used -- see
  plans/uniform-levels.md U7-6g item 5 for the measured widths.
  **THE SUITE'S CYLINDER WINDOW OPENS ON THE SHEDDING ONSET RAMP**
  (plans/uniform-levels.md S8-4, 2026-09-23). Onset is 41-60 D/U0 after reset;
  `cylinder-metrics.js`'s transient is 40. Settled (150 D/U0) values read
  0.6-3% HIGHER than the suite: N2-diffuse 1.688/0.1495, N2-bounceback
  1.365/0.1651, N3-diffuse 1.476/0.1570, N3-bounceback 1.370/0.1649. The
  suite's numbers are still reproducible regression markers; they are not the
  flow's Cd. `tools/probe-field-scaling.js` uses 150 and reports `settle`.
  **AND S8-2 MOVED THE LEVELS>=3 ROWS** (level >= 2 cells were placed 1/2 - dx
  of a root cell off their data; `?cellcentre=0` restores that): suite-window
  N3-diffuse 1.473 -> 1.433, N3-bounceback 1.365 -> 1.353. Level-1 CELLS are
  placed identically by construction, but the manager's proximity box moved
  half a root cell at every level, so N2 can move in the last digit:
  N2-bounceback is bit-identical (1.356/0.1642), N2-diffuse St 0.1484 -> 0.1485.
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
  **AND THE FIELD IS NOT REPRODUCIBLE EITHER, BUT NOT UNIFORMLY -- IT
  DEPENDS ON THE CONFIG, THE OPPOSITE WAY ROUND FROM Cd.** Measured
  2026-09-14 with `debugSnapshotSave` + `tools/amr-diff.js` on
  `index-cylinder-amr.html`, 4096 steps from `reset()` (a PINNED body, so
  force does not feed back into the flow -- unlike the falling card, whose
  field cannot be compared this way at all):

      ?levels=3   at least FOUR modes, each EXACTLY reproducible. Over 22
                  runs and 8 builds: mode A 12 runs, B 4, C 3, D 1. Modes
                  differ from each other by ux relL2 1.6-3.0e-5, vorticity
                  3.4-5.8e-4 -- and every run reproduces its mode BIT-FOR-BIT,
                  including across different builds. The count keeps going up
                  as more runs are taken: treat it as "several", not as a
                  closed set, and never as "it must be A or B".
      ?levels=2   every one of 4 runs DIFFERS, at ux relL2 2-4e-5 (more modes,
                  or none)

  **It is NOT noise and NOT an excursion** -- that was the first reading and it
  was wrong. A run lands in one of a small number of attractors (the atomicSub
  free list resolving a race one way or another at some early refine), and each
  attractor is bit-exact and build-independent. So:

  - a DIFFERS tells you only WHICH MODE you are in. It is never, on its own,
    evidence that a change moved anything.
  - an IDENTICAL against a baseline run is conclusive -- no race can forge a
    bit-exact match.
  - the strong form is to match the baseline in TWO DIFFERENT MODES. Two
    builds agreeing bit-for-bit in each of two attractors is far better
    evidence than one IDENTICAL, and it costs one extra run.

  N=2 has no reproducible baseline at all and `amr-diff` cannot gate there;
  use the Cd/St numbers, which ARE stable there to the digit across builds.
  **THAT MAY NO LONGER BE TRUE — re-measure before relying on it either way.**
  2026-09-22, same protocol (fresh load, `reset()`, 4096 steps), 4 runs at
  `?levels=2`: THREE distinct snapshot hashes, but ux and vorticity relL2
  EXACTLY 0 in every pairing, plus two more runs in a separate probe, also
  identical. The race still permutes which SLOT holds a tile -- that is what
  moves the hash -- but no longer which blocks are refined, so level 0's field
  reproduces. Six runs, ONE session, ONE build, which is not the
  several-builds standard this file asks for, so the figures above stand as
  recorded; if it holds up, `amr-diff` gains a gate at N=2 it has never had.
  Candidates for what closed it: D1-a's reset fixes (2026-09-18), U7-6f's
  deletion, or U7-6g's root-velocity seed.

  **AND THERE IS A THIRD FORM, WHICH IS STRONGER THAN EITHER AND COSTS
  NOTHING: TAKE THE FREE LIST OUT OF THE LEG.** `setAutoRefine(false)`
  immediately after `reset()` leaves every level >= 1 empty, so the ROOT ALONE
  steps, no slot is ever handed out, and the run is bit-deterministic —
  attractors cannot arise because the race that chooses them never happens.
  A build A/B is then simply conclusive, with no repeat and no mode-matching.
  It cannot see anything that needs a coarse/fine interface, so it is not a
  replacement for the modes above; it IS the right instrument for anything
  level 0 owns alone (the sponge, the root step, geometry in the far field).
  Assert the zero tile count in the leg, or it is a different experiment from
  the one you think you ran. Worked example: plans/uniform-levels.md U7-6g,
  which retired `SPONGE_CELL_SNAP` on an IDENTICAL with a half-cell control
  three orders away.

  **`tools/amr-diff.js` WAS DEAD FROM 7dc30e8 UNTIL 2026-09-22** — it read the
  dense arrays U7-6f deleted and threw a raw TypeError on every snapshot this
  project can produce. Fixed (it takes level 0 from the snapshot's `root`), but
  worth knowing when reading a number above: everything in this section was
  measured before that commit.

  Note this inverts the Cd picture, where N=3 is the WIDER spread: do not
  assume one config's reproducibility from another's, in either direction.

  **AND THE SPREAD WIDENS WITH DEPTH -- ~+/-0.002 at N=3, not +/-0.001.**
  The figure above is `amr-N2-diffuse`'s. Measured 2026-09-14, three runs of
  one build: `amr-N3-diffuse` gave Cd 1.411 / 1.408 / 1.407. Same mechanism,
  more of it -- a third pool level is a third free list regrouping a third set
  of truncated partials. Calibrate against the config you are actually
  comparing, and take the repeat on THAT config; N=2's floor is not N=3's.
  **VERIFY WHICH TREE THE DEV SERVER IS SERVING BEFORE BELIEVING A RUN.**
  `ensureServer` only checks that *something* answers on the baseUrl port; if
  it finds one it prints "HTTPS dev server already up" and reuses it, whatever
  that server's cwd is. With more than one worktree live, the default
  `https://localhost:4444` is routinely the MAIN checkout's server, not yours.
  Measured 2026-09-15: a full sweep and a "pristine A/B" both ran green and
  agreed with each other while serving the main checkout and the `3d-trt`
  worktree respectively -- neither was the branch under test, and every number
  had to be discarded. Two runs agreeing is NOT evidence they ran your code.
  Prove the tree, then pin both ports:

      ss -lptn | grep -oE 'pid=[0-9]+'      # then readlink /proc/<pid>/cwd
      curl -sk https://localhost:<port>/shaders/physics.wgsl | grep -c <TOKEN-YOU-ADDED>
      node tools/validate-all.js --baseUrl=https://localhost:<yours> --port=<your chrome>

  `https.py` takes `GA_PORT=`; it is UNTRACKED, so an archived checkout used
  as a baseline will not have it.

- **`tools/validate-cylinder.js`** — physics: pinned cylinder in uniform
  crossflow, time-averaged Cd/Strouhal vs. literature values in
  `benchmarks/cylinder.json`. Assumes a Chrome + page are already up (see
  `webgpu-verify`) — `validate-all.js` is the one-command version.
- **`tools/validate-amr-invariants.js`** — AMR structural invariants,
  asserted periodically through a run (not just at the end, so a transient
  violation can't slip past). **Seven gates as of 2026-09-14, all gating:**
  2:1 balance (`debugCheck21Balance`), CORNER 2:1 balance (same call — gating
  only since the closure made it satisfiable), geometry-forced refinement
  (`debugCheckGeometryCoverage` — every leaf near the body already at the
  finest level), field-finite (NaN/blowup), pool starvation (refines refused
  for want of a slot; needs `?diag=1` or it reads vacuously true — and needs
  the page to expose `debugReadDiag`, which the channel, TGV and reentry pages
  did NOT until 2026-09-22, so this gate was absent rather than green on
  three of five AMR pages), the 2:1
  CLOSURE (`debugCheckRefinementClosure` — what the rule requires but the
  allocator did not deliver), and the slot-quadrant rule
  (`debugCheckSlotQuadrants`).
  **It was eight for a day.** `debugCheckTileOrigins` scored the per-slot
  origin buffers against their closed form, to justify taking the kernels off
  them; once B3-5 deleted the buffers it had nothing to read, and it went in
  the same commit rather than being left pointing at deleted state — which is
  how this project collected three vacuous gates. Retiring a checker WITH its
  subject is the counterpart of that lesson, not an exception to it.
  **Sanity-check the sweep against a starved pool** (`--extra=maxFineBlocks=16`)
  after touching it. Measured 2026-09-14 on `amr-dev-invariants` at 1024 steps:
  FIVE go red (2:1-balance 14, corner 26, coverage 36, pool STARVED 2454,
  closure 11 missing) and `field` and `quadrants` do NOT. Five-of-seven, with
  the two abstainers being the ones whose rules are genuinely independent of
  starvation, is the discrimination that says these read seven different things
  rather than one -- a sweep where everything goes red together has not been
  shown to test anything.
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
- **`tools/validate-snapshot-roundtrip.js`** — standalone/opt-in. Steps N from
  `reset()`, saves, loads what it just saved, steps N more, and requires the
  result to be BIT-IDENTICAL to an uninterrupted 2N. The second N is what makes
  it a gate rather than a restore check: it contains refinement rounds, so
  anything the format failed to carry about the ALLOCATOR shows up as a
  different slot handout. `tools/lib/render-levels.js` also round-trips a
  snapshot but renders immediately after the load, so it scores only what the
  renderer reads and passed while the format was still wrong.
      node tools/validate-snapshot-roundtrip.js
      node tools/validate-snapshot-roundtrip.js --steps=1024
  Three controls, and none is decoration: `alive` (the capture at N must differ
  from the one at 2N — every row is an EQUALITY, so a capture that carries
  nothing passes them all), `stale` (loading a snapshot 64 steps later must
  MOVE the outcome, which is what proves the load is load-bearing), and
  `refuse` (a level-1 per-block/QUAD granularity mismatch must THROW). The
  `stale` control is the one that caught a save bug three gated rows had gone
  green on — see plans/uniform-levels.md U7-6a.
  **A COMMAND-ENCODER VALIDATION ERROR DROPS THE WHOLE COMMAND BUFFER**, which
  is how that bug presented: one over-long `copyBufferToBuffer` discarded every
  other copy in the same submit, and `debugSnapshotSave` returned an all-zero
  dense grid at `?levels>=3` while the page itself ran perfectly. If a readback
  comes back all zeros, suspect a sibling copy in the same encoder before you
  suspect the sim.
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

## The 16-storage-buffer ceiling
`shaders/amr_manage_pool.wgsl` declares **exactly 16 storage buffers**, which
IS `maxStorageBuffersPerShaderStage` on the target hardware — every AMR page
checks for it at init (`NEEDED_STORAGE_BUFFERS_PER_STAGE = 16`) and refuses
loudly if the adapter offers fewer. So that kernel has **no free binding**:
adding one is a hard `CreateBindGroupLayout` failure, the page does not boot,
and no bind-group regrouping helps because the limit is per STAGE. Measured
2026-09-14 while trying to add one array (plans/2D-backport.md B2-2b, which
that discovery re-shaped).

**One was recovered on 2026-09-14 and the count is now 15** (B2-2b0):
`childQuadrant` held `slot % 4` -- both allocators compose a slot as
`quadIdx*4 + quadrant`, so it was a buffer storing a constant. The buffer
itself stays (five other shaders read it) and `allocLevelPool` now writes it
once; only this kernel's binding went. `amr2d.mjs`'s `quadrantOfSlot` is the
rule and `checkSlotQuadrantsOnGPU` scores the live buffer against it.

**AND IT IS 10 SINCE B3-5 (2026-09-14), with real slack for the first time.**
`childOriginX`/`childOriginY` (written here at refine time) and
`parentOriginX`/`parentOriginY` (read here to write them) are all gone: a
tile's origin is `block * RB * 2^-(m-1)` in closed form (`amr2d.mjs`'s
`tileOriginL0`), so it is derived in the three kernels that need it and stored
nowhere. Same shape as `childQuadrant` -- a buffer holding something cheaper
to compute -- and the same order of work: prove the closed form against the
live buffer FIRST, then remove.

Still check the count BEFORE designing around a new buffer -- the failure mode
is a `CreateBindGroupLayout` error at init, i.e. a page that does not boot, not
a warning.

## Performance work
**Current numbers: `plans/performance-snapshot.md`** — per-device constants, the
passes in a root step, a fitted cost model (`tools/fit-cost-model.js`, ±10–25% on
totals) and AMR vs flat. Read `plans/perf-characterization.md` BEFORE optimizing
anything here: it is the history, dead ends included. The
two target devices have **opposite** bottlenecks — the desktop is
pass-count/latency bound, the mobile PowerVR is memory-bandwidth bound — and
the obvious optimization (indirect dispatch off the active block count) was
measured to help neither **at `?levels=2` with 128 slots** -- and at depth
empty slots are the largest item on the phone (~14 ns per empty workgroup,
scaling with slots x 2^m substeps). **`dispatchWorkgroupsIndirect` itself costs
~230 us (phone) / ~390 us (desktop) PER CALL in this Chrome**, so the fix that
shipped is `?stride=1` -- a direct dispatch walking an active-slot list -- not
indirect dispatch (2026-09-24, same document). It also records the trap that produced a confident
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

**BUT NOT FOR A COARSE/FINE COUPLING CHANGE: THE ANALYTIC *AMR* CONFIGS
REFINE NOTHING.** `channel-poiseuille-amr-N2`, `channel-couette-amr-N2`,
`tgv-amr-N2` and `tgv-amr-N3` all hold ZERO active tiles at every level —
measured with `debugListActiveBlocks` after 2048 steps, and deliberate on both
pages (`main-channel-amr.js` defaults `autoRefine` OFF over a Couette
wall-seam trigger; `main-tgv-amr.js` keeps thresholds that never fire at TGV's
vorticity scale). `tgv-amr-N2` and `tgv-amr-N3` therefore print the IDENTICAL
number, which is the tell. They gate the solver, not the seam: all four were
bit-identical across B1's rescale flip, which moved the seam's momentum drift
by 27x. For anything touching interp/average/ghosts, the instrument is
`tools/analyze-amr-interface.js`'s conservation channel (plans/2D-backport.md
B1, finding #1). A `*-amr-*` name in this suite does not by itself mean a run
had an interface in it.
**THE TGV HALF OF THAT ATTRIBUTION IS NOW MEASURED, NOT INFERRED, AND IT
STANDS** (2026-09-22, plans/uniform-levels.md U7-6g). Until then the root's
first criterion round read a ZERO velocity field on that page, so zero tiles
at step 0 was guaranteed for a reason unrelated to the thresholds. The root is
now seeded with the true Taylor-Green velocity (`denseL0ToRootVel`) and it is
STILL zero at every level, at 0/64/512/2048 steps and at both `?levels=`.
Controls: `readField()` reads `pools[0].finePoolVel` directly and reports
max|ux| = max|uy| = U0 exactly with 4032/4096 cells nonzero, so the seed
landed; and `?refineThresh=-8&coarsenThresh=-9` gives L1 = 64 active, so the
count can go nonzero.

`?f16=1|2` — real packed-half storage for `f` (`shaders/common_fpack.wgsl`,
`f-pack.mjs`). Default 0 and byte-identical to the previous `array<f32>`
layout. Measured NOT viable as a default; kept for re-measurement, not for
shipping.

**Pool capacity is per level, and `?maxFineBlocks=` sizes LEVEL 1 ONLY.**
Each deeper level takes its own `?maxFineBlocks2=`, `?maxFineBlocks3=`, ... —
which was undocumented anywhere outside the allocation loop until 2026-09-15,
while the pool-exhaustion refusal told you to raise `?maxFineBlocks=` whatever
level had actually run dry. On `?levels=4` that advice is a no-op, so raising
it looked like the refusal was spurious rather than the knob being wrong. The
message now names the saturated level's own knob.

Defaults come from measured demand (`POOL_PEAKS` on each AMR page, sized by
`amr2d.mjs`'s `poolSlotsFor` at 1.7x the peak), not from one flat number per
page. Max live tiles over 40k steps with every cap lifted:

      falling card (index-amr.html)        cylinder (index-cylinder-amr.html)
      L=3    L=4    L=5                    L=3    L=4
  1   231    213    261                1    78     95
  2   400    492    516                2    96    160
  3    --    656    628                3    --    208
  4    --     --    904

Demand tracks the level INDEX, and STEPS UP when a level acquires a child
(level 2: 400 as the finest level, ~500 once level 3 exists) — that is 2:1
closure forcing a parent tile for everything refined below. The old flat 512
(card) / 128 (cylinder) for every level >= 2 is exactly why `?levels=4`
refused, and is CLAUDE.md's own recorded "level 3 at 128/128 with 20 coverage
violations" on the cylinder page. Levels beyond the table are extrapolated and
say so; the refusal watch remains the thing that reports a wrong guess.

**The body lives in BUFFER coordinates** (plans/2D-backport.md B5, completed
2026-09-15). `cx`/`cy` are buffer positions, integrated and wrapped into
[0, W) x [0, H) every step; a kernel that has a buffer cell already has the
body's frame and needs no conversion. Only the ALBC sponge band, the WALL_Y
walls and the render convert (`bufferToWindowCell`/`bufferToWindowPos` in
`common_geometry.wgsl`). The old window-anchored convention and its `?window=`
flag are DELETED — there is one convention, not a switch.

If you add a kernel that touches the body, it needs no frame call at all. The
failure mode to know about: `lbm_force.wgsl` spent B5-2 and B5-3 building a
WINDOW position while `state.cx` had become a buffer one, and because the two
coincide whenever `off == 0` — which is every pinned-cylinder config, i.e.
every config in the sweep — nothing caught it until a moving window was
measured directly (B5-5, 226x wrong in fy). **Grep new kernels for raw
`off_x`/`off_y` arithmetic; only the render should have any.**

`?wrapScreens=N` (index-reentry.html) — domain-heights of accumulated travel
before `x_total`/`y_total` wrap; default 16, byte-identical when absent. The
instrument for the body's sub-cell precision, because under the shipped window
convention that position IS `frac(y_total)`, so the accumulator's magnitude is
the resolution. Measured 2026-09-15 (plans/2D-backport.md B5-4), 262144 steps
against an exact prescribed trajectory: wrap=16 errs 8.5 cells, wrap=1 errs
0.71 -- and `?window=0`, the whole point of B5, errs 0.72. **The buffer
convention and `wrapScreens=1` are the same number**, so B5 buys no precision
that one constant does not. Note also there is NO fixed ULP under either: the
drift grows roughly linearly with step count, and the buffer convention is
~10x WORSE below ~7000 steps (its `cy` starts at H/2 while `y_total` starts
at 0). Both candidate defaults are re-baselining; neither has been adopted.

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

`?stride=0` -- restores the every-slot launch on `index-amr.html`. Default 1:
the per-substep passes walk each level's active-slot list
(`shaders/amr_active_list.wgsl`, `mainStride` in each kernel) instead of
launching every pool slot; -10% on the phone at the defaults, -37% at res=5
levels=4. Bit-identical by construction and gated so by
`tools/validate-stride.js`. The other four AMR pages still launch every slot:
their layouts and bind groups are untouched, because only a `mainStride`
entry point reads the list.

Timing/measurement entry points:
- `?telemetry=1` — POSTs periodic samples (device, adapter, config, frame
  GPU/sync ms) to the dev server's `/_telemetry`, appended to
  `telemetry.log`. The way to see a device `tools/bench-amr.js` cannot drive,
  e.g. a phone on the LAN. (A phone on USB CAN be driven over CDP via `adb
  forward tcp:9222 localabstract:chrome_devtools_remote`, only while Chrome is
  foregrounded, and **only with chrome-remote-interface's `local: true`**:
  Chrome for Android crashes outright on the `/json/protocol` request CRI
  otherwise makes on every connect. That crash was the "debugStepSync killed
  Chrome" once recorded here. `tools/test-cdp-local.js` gates it — see
  `plans/perf-characterization.md`.) Opt-in, same-origin, local-only.
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
