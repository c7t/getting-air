# getting-air -- validation helpers.  Requires GNU make.
#
# Fast, GPU-free static checks a contributor (or an AI agent) can run to verify
# changes before loading the sim in a browser. There is no build step for the
# app itself (it is a static page); these targets only VALIDATE.
#
#   make check   run all static checks (JS + unit tests always; WGSL if naga is)
#   make test     run the GPU-free unit tests (tools/test-*.js)
#   make wgsl     validate every WGSL shader with naga (needs Rust-built naga)
#   make js       syntax-check every JS module with `node --check`
#   make tools    install the validation tools (naga-cli via cargo; needs Rust)
#   make help     list targets
#
# JS validation and the unit tests only need Node; WGSL validation needs `naga`
# (a Rust tool). `make check` runs JS and the unit tests unconditionally and
# skips WGSL with a note if naga is not installed, so it is useful even without
# a Rust toolchain.
#
# NOTE what `make check` still does NOT prove: that anything renders, or that
# the physics is right. `node --check` is syntax only, and `make test` covers
# just the pure-arithmetic/pure-data code paths that can run without a GPU.
# The GPU suites under tools/ (validate-all.js and friends, see CLAUDE.md) are
# the ones that actually exercise the solvers.

# Prefer a cargo-installed naga if present, without clobbering an existing one
# elsewhere on PATH (respects CARGO_HOME; falls back to ~/.cargo).
export PATH := $(if $(CARGO_HOME),$(CARGO_HOME),$(HOME)/.cargo)/bin:$(PATH)

# naga-cli version requirement for `make tools` (override: make tools NAGA_VERSION=31).
NAGA_VERSION ?= 30

SHADERS := $(wildcard shaders/*.wgsl)
# Recursive, not $(wildcard *.js) -- that only ever matched root-level files,
# silently skipping tools/ and tools/lib/ (real, actively-used JS, not just
# root main*.js). venv/ (Python virtualenv) and AGAL/ (vendored C++/CUDA
# sub-repo) are excluded -- neither is this project's own JS source.
JS      := $(shell find . -name '*.js' -not -path './node_modules/*' -not -path './.git/*' -not -path './venv/*' -not -path './AGAL/*')

# GPU-free unit tests. Every tools/test-*.js is expected to be runnable as
# `node tools/test-foo.js` with no server, no browser and no GPU, and to exit
# nonzero on failure -- that contract is what lets this be a plain wildcard
# rather than a hand-maintained list.
TESTS   := $(sort $(wildcard tools/test-*.js))

.DEFAULT_GOAL := help

.PHONY: help
help: ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  make %-10s %s\n", $$1, $$2}'

.PHONY: check
check: js test ## run all static checks (JS + unit tests always; WGSL if naga is available)
	@if command -v naga >/dev/null 2>&1; then \
	  $(MAKE) --no-print-directory wgsl; \
	else \
	  echo "note: naga not installed -- skipping WGSL validation (run 'make tools')"; \
	fi
	@echo "OK: static checks passed"

.PHONY: test
test: ## run the GPU-free unit tests (tools/test-*.js)
	@command -v node >/dev/null 2>&1 || { echo "node not found -- install Node.js"; exit 1; }
	@test -n "$(strip $(TESTS))" || { echo "no tests found matching tools/test-*.js (run from repo root?)"; exit 1; }
	@rc=0; for t in $(TESTS); do \
	  echo "== $$t"; \
	  if node "$$t"; then :; else rc=1; fi; \
	done; \
	if [ $$rc -eq 0 ]; then echo "test: $(words $(TESTS)) suite(s) passed"; else echo "test: FAILED"; fi; \
	exit $$rc

.PHONY: wgsl
wgsl: ## validate every WGSL shader with naga (needs Rust-built naga)
	@command -v naga >/dev/null 2>&1 || { echo "naga not found -- run 'make tools' (needs Rust)"; exit 1; }
	@test -n "$(strip $(SHADERS))" || { echo "no shaders found matching shaders/*.wgsl (run from repo root?)"; exit 1; }
	@rc=0; for f in $(SHADERS); do \
	  case "$$f" in \
	    shaders/common_*.wgsl) echo "  skip  $$f (fragment-only, never compiled alone -- validated via every includer's assembled output)"; continue;; \
	  esac; \
	  if grep -q '^// @include' "$$f"; then \
	    command -v node >/dev/null 2>&1 || { echo "node not found -- needed to assemble $$f's @include fragments"; exit 1; }; \
	    tmp=$$(mktemp --suffix=.wgsl); \
	    if ! node tools/assemble-shader.js "$$f" > "$$tmp" 2>"$$tmp.err"; then \
	      echo "  FAIL  $$f (assemble)"; sed 's/^/        /' "$$tmp.err"; rc=1; rm -f "$$tmp" "$$tmp.err"; continue; \
	    fi; \
	    rm -f "$$tmp.err"; \
	    if out=$$(naga "$$tmp" 2>&1); then echo "  ok    $$f (assembled)"; \
	    else echo "  FAIL  $$f"; echo "$$out" | sed 's/^/        /'; rc=1; fi; \
	    rm -f "$$tmp"; \
	  else \
	    if out=$$(naga "$$f" 2>&1); then echo "  ok    $$f"; \
	    else echo "  FAIL  $$f"; echo "$$out" | sed 's/^/        /'; rc=1; fi; \
	  fi; \
	done; \
	if [ $$rc -eq 0 ]; then echo "wgsl: $(words $(SHADERS)) shader(s) valid"; else echo "wgsl: FAILED"; fi; \
	exit $$rc

.PHONY: js
js: ## syntax-check every JS module with `node --check`
	@command -v node >/dev/null 2>&1 || { echo "node not found -- install Node.js"; exit 1; }
	@test -n "$(strip $(JS))" || { echo "no JS modules found matching *.js (run from repo root?)"; exit 1; }
	@rc=0; for f in $(JS); do \
	  if out=$$(node --check "$$f" 2>&1); then echo "  ok    $$f"; \
	  else echo "  FAIL  $$f"; echo "$$out" | sed 's/^/        /'; rc=1; fi; \
	done; \
	if [ $$rc -eq 0 ]; then echo "js: $(words $(JS)) module(s) parse"; else echo "js: FAILED"; fi; \
	exit $$rc

.PHONY: tools
tools: ## install validation tools (naga-cli via cargo; needs Rust)
	@command -v cargo >/dev/null 2>&1 || { echo "need Rust/cargo to install naga-cli -- see https://rustup.rs"; exit 1; }
	cargo install naga-cli --version '^$(NAGA_VERSION)' --locked

# --- Publishing / release --------------------------------------------------
# `make status` talks to GitHub via `gh` and ALWAYS acts on your own `origin`
# remote -- derived from the remote URL, NOT `gh repo view` (which resolves a
# fork to its PARENT and would make you report on the wrong repo). So it shows
# whichever repo you cloned from: the maintainer sees the canonical site, a
# fork owner sees their own. The validation targets above need no network and
# no gh; only this one does.
#
# There is no release step any more: Pages serves `main` directly, so merging
# to main publishes. See the `publish` target below for why it still exists.

# Minimum major version of gh we rely on (Pages API + `gh auth status`).
MIN_GH_MAJOR := 2

# origin owner/repo from the remote URL (handles https and ssh forms).
ORIGIN_SLUG := $(shell git remote get-url origin 2>/dev/null | sed -E 's#(git@|https://)([^/:]+)[/:]##; s#\.git$$##')

.PHONY: require-gh
require-gh: ## check GitHub CLI (gh) is installed, current, and authenticated
	@command -v gh >/dev/null 2>&1 || { \
	  echo "gh (GitHub CLI) is not installed -- needed for 'make status'."; \
	  echo "  install: https://github.com/cli/cli#installation"; \
	  echo "           (e.g. 'brew install gh', 'sudo apt install gh', 'sudo dnf install gh')"; \
	  exit 1; }
	@have=$$(gh --version | sed -n 's/^gh version \([0-9]*\).*/\1/p'); \
	  if [ "$${have:-0}" -lt $(MIN_GH_MAJOR) ]; then \
	    echo "gh is too old (found v$$have, need v$(MIN_GH_MAJOR)+) -- please update it."; \
	    echo "  update: https://github.com/cli/cli#installation"; \
	    exit 1; fi
	@gh auth status >/dev/null 2>&1 || { \
	  echo "gh is installed but not signed in."; \
	  echo "  run: gh auth login"; \
	  exit 1; }

.PHONY: status
status: ## show origin repo, current branch, and what is published (Pages)
	@echo "origin:  $(ORIGIN_SLUG)"
	@echo "branch:  $$(git rev-parse --abbrev-ref HEAD)"
	@echo "HEAD:    $$(git rev-parse --short HEAD)  $$(git log -1 --format=%s)"
	@if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then \
	  pb=$$(gh api "repos/$(ORIGIN_SLUG)/pages" --jq .source.branch 2>/dev/null); \
	  pu=$$(gh api "repos/$(ORIGIN_SLUG)/pages" --jq .html_url 2>/dev/null); \
	  if [ -n "$$pb" ]; then echo "pages:   branch '$$pb'  ->  $$pu"; \
	  else echo "pages:   not configured for $(ORIGIN_SLUG)"; fi; \
	else \
	  echo "pages:   (install & sign in to gh to show what's published -- 'make require-gh')"; \
	fi

# Pages serves `main` directly (changed 2026-09-08), so publishing IS merging:
# every push to main rebuilds the site. This target used to fast-forward a
# separate `gh-pages` branch to the current commit; kept as a guarded no-op
# rather than deleted, because the old muscle memory is dangerous now -- it
# pushed HEAD to the Pages branch, and with the Pages branch being `main` that
# would push whatever branch you happen to be on straight to main, bypassing
# the PR flow entirely.
.PHONY: publish
publish: ## (no-op) Pages serves main directly -- merging to main publishes
	@pb=$$(gh api "repos/$(ORIGIN_SLUG)/pages" --jq .source.branch 2>/dev/null); \
	pu=$$(gh api "repos/$(ORIGIN_SLUG)/pages" --jq .html_url 2>/dev/null); \
	cur=$$(git rev-parse --abbrev-ref HEAD); \
	echo "nothing to publish: Pages serves '$${pb:-main}' directly, so every push to it is already live."; \
	if [ -n "$$pu" ]; then echo "  live at: $$pu"; fi; \
	if [ "$$cur" != "$${pb:-main}" ]; then \
	  echo "  you are on '$$cur' -- open a PR into '$${pb:-main}'; merging is what publishes."; \
	fi
