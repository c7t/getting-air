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
