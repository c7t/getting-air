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
- `make js` — `node --check` every `*.js` (needs Node).
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
  boot smoke check (`index-boot`, `amr-dev-boot`) against `index.html` and
  `index-amr.html` — the two dev pages with no `window.__CYL`, so the Cd/St
  and invariant checks don't apply, but that used to mean this suite never
  loaded them at all. Added after a shared-shader/JS-bind-group mismatch
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
- Shared analysis code lives in `tools/lib/` (`cylinder-metrics.js`,
  `amr-invariants.js`, `field-reconstruct.js`, `amr-resolution-mapping.js`,
  `amr-cost.js`, `browser-lifecycle.js`) — both the leaf tools and
  `validate-all.js`/`validate-amr-vs-dense.js` call the same logic, not
  independently-drifting copies.

Current known-issue state (e.g. which `?levels=N` combinations are physics-
validated) drifts with active work — see `main-cylinder-amr.js`'s own
comment above `N_LEVELS`, not this file, for what's current.

## Branch model
- **`main`** — canonical, always-buildable, default branch. All PRs land here.
- **`gh-pages`** — the *published* snapshot; the GitHub Pages source. **Never edit
  or commit to it directly** — it is only ever advanced by a release (below).
- **feature branches** — short-lived, off `main`, PR back into `main`, deleted on
  merge. (Fork contributors use `user/<handle>/<topic>`.)

## Releasing (publishing the live site)
Publishing is a deliberate step, separate from merging to `main`:

    make publish        # fast-forward origin's Pages branch to the current commit

`make publish` and `make status` talk to GitHub via `gh` and always act on **your
own `origin`** (derived from the remote URL, never `gh repo view`, which resolves
a fork to its parent). So the maintainer publishes the canonical site; a fork
owner publishes their own fork. `publish` is guarded: it runs `make check` first,
requires a clean tree, is fast-forward-only (needs `FORCE=1` to overwrite a
diverged Pages branch), no-ops if already published, and prompts for confirmation
(`CONFIRM=1` to skip, `DRYRUN=1` to preview). Never push `gh-pages` by hand.

`gh` is required only for `status`/`publish`; run `make require-gh` to check it is
installed, current, and signed in.

### First-time Pages setup (one-time, if `gh-pages` doesn't exist yet)
Derive `OWNER/REPO` from `origin` (`git remote get-url origin`), then:

    git push origin main:refs/heads/gh-pages                 # seed the published branch
    gh api --method PUT "repos/OWNER/REPO/pages" --input - <<< '{"source":{"branch":"gh-pages","path":"/"}}'

After that the Pages source is `gh-pages` and `make publish` maintains it. (If the
repo currently serves Pages from another branch, this PUT repoints it; do it
before retiring the old branch so the site never unpublishes.)

## Conventions
- Never hardcode the repo owner/name — derive identity from `origin`.
- Prefer ecosystem tooling (`make`, `naga`, `node`) over ad-hoc scripts.
- Small, focused commits; do not commit secrets or local dev artifacts.
