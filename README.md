# getting-air

A WebGPU (WGSL) D2Q9 lattice-Boltzmann fluid simulator, including a 2-level
block-structured AMR build. It is a **static page** — no build step — served
directly by GitHub Pages.

## Development

No build step: edit the `.html` / `.js` / `shaders/*.wgsl` files directly and
reload in a WebGPU-capable browser. The `Makefile` provides GPU-free static
checks and a release helper (`make help` lists everything).

### Prerequisites
- **Node** — for `make js` (`node --check` on the JS).
- **`naga`** (optional) — for `make wgsl` (WGSL validation). `make tools`
  installs it via cargo (needs Rust). Without it, `make check` still runs the
  JS checks and skips WGSL.
- **`gh`** (GitHub CLI) — only for `make status` / `make publish`.
  `make require-gh` checks it is installed, current, and signed in.

### Validate before committing
```
make check      # JS always; WGSL too if naga is installed
```
This is static validation only. To confirm the app actually renders, run it in
a real GPU browser (see `.claude/skills/webgpu-verify/`).

### Branches
- **`main`** — the canonical branch; always buildable; where PRs land.
- **`gh-pages`** — the published snapshot that GitHub Pages serves. Updated only
  by a release; never edited directly.
- **feature branches** — short-lived, off `main`, merged back via PR.

### Publishing a release
Merging to `main` does **not** change the live site; publishing is deliberate:
```
make publish    # fast-forward your origin's Pages branch to the current commit
```
`make publish` / `make status` act on your own `origin`, so a fork owner
publishes their fork and the maintainer publishes the canonical site. `publish`
runs `make check` first, requires a clean tree, is fast-forward-only
(`FORCE=1` to override), and confirms before pushing (`DRYRUN=1` to preview).

### Contributing
Fork, branch off `main`, open a PR against `main`. You can preview your own copy
live with `make publish` against your fork.
