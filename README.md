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
- **`gh`** (GitHub CLI) — only for `make status`.
  `make require-gh` checks it is installed, current, and signed in.

### Validate before committing
```
make check      # JS always; WGSL too if naga is installed
```
This is static validation only. To confirm the app actually renders, run it in
a real GPU browser (see `.claude/skills/webgpu-verify/`).

### Branches
- **`main`** — the canonical branch; always buildable; where PRs land; **and the
  published site** (GitHub Pages serves it directly).
- **feature branches** — short-lived, off `main`, merged back via PR.

### Publishing
There is no release step: **merging to `main` publishes**, and Pages rebuilds
within about a minute. So run `make check` (and the GPU validation, for anything
physics-affecting) *before the merge*.
```
make status     # origin repo, current branch, and what Pages serves
```
`make status` acts on your own `origin`, so a fork owner sees their fork and the
maintainer sees the canonical site. `make publish` remains only as a no-op that
explains this; it no longer pushes anything.

### Contributing
Fork, branch off `main`, open a PR against `main`. To preview your own copy live,
enable Pages on your fork's `main` (Settings → Pages).
