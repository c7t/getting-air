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
make check      # JS + Python syntax; WGSL too if naga is installed
```
This is static validation only. To confirm the app actually renders, run it in
a real GPU browser (see `.claude/skills/webgpu-verify/`).

### Dev server (optional, for local iteration)
`tools/devserver.py` is a small, dependency-free (Python standard library only)
static server for local development. It serves the repo with `no-cache` (so every
edit shows on a plain reload) and accepts snapshot/artifact uploads on
`POST /collect`.
```
make serve                 # http on localhost:8080 (same-machine use)
make serve ARGS=--https    # https with a self-signed cert (see below)
# or directly:
python3 tools/devserver.py [--https] [--port N] [--lan]
```

**Why HTTPS (and why self-signed is fine).** WebGPU (`navigator.gpu`) is only
exposed in a **secure context**. Plain HTTP counts as secure *only* for loopback
(`http://localhost`, `127.0.0.1`, `[::1]`); any other origin — for example
reaching the server by the host machine's own network address so a browser on a
different device can load it — is **not** a secure context over plain HTTP, and
the browser refuses WebGPU there. (Chrome does this even with
`--unsafely-treat-insecure-origin-as-secure`; the flag does not enable WebGPU on a
non-loopback HTTP origin, by design.) Serving over **HTTPS** supplies the secure
context so the simulation can initialize.

Because this is a **local development** server, a **self-signed** certificate is
sufficient: its only job is to make the origin a secure context. The browser shows
a one-time "not trusted" warning that you accept once; WebGPU then works regardless
of certificate trust. `--https` mints the certificate automatically (needs
`openssl`), with subject-alternative names for `localhost` and the host's own
address so it matches by name.

Guidance: use `--http` (the default) when the browser and server are on the **same
machine** and you open `http://localhost:PORT`; use `--https` for anything reached
by network address. `POST /collect` is gated by a per-run token printed at startup
and only accepts same-origin requests.

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
