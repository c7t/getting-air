# getting-air -- validation helpers.  Requires GNU make.
#
# Fast, GPU-free static checks a contributor (or an AI agent) can run to verify
# changes before loading the sim in a browser. There is no build step for the
# app itself (it is a static page); these targets only VALIDATE.
#
#   make check   run all static checks (JS always; WGSL if naga is available)
#   make wgsl     validate every WGSL shader with naga (needs Rust-built naga)
#   make js       syntax-check every JS module with `node --check`
#   make tools    install the validation tools (naga-cli via cargo; needs Rust)
#   make help     list targets
#
# JS validation only needs Node; WGSL validation needs `naga` (a Rust tool).
# `make check` runs JS unconditionally and skips WGSL with a note if naga is
# not installed, so it is useful even without a Rust toolchain.

# Prefer a cargo-installed naga if present, without clobbering an existing one
# elsewhere on PATH (respects CARGO_HOME; falls back to ~/.cargo).
export PATH := $(if $(CARGO_HOME),$(CARGO_HOME),$(HOME)/.cargo)/bin:$(PATH)

# naga-cli version requirement for `make tools` (override: make tools NAGA_VERSION=31).
NAGA_VERSION ?= 30

SHADERS := $(wildcard shaders/*.wgsl)
JS      := $(wildcard *.js)

.DEFAULT_GOAL := help

.PHONY: help
help: ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  make %-10s %s\n", $$1, $$2}'

.PHONY: check
check: js ## run all static checks (JS always; WGSL if naga is available)
	@if command -v naga >/dev/null 2>&1; then \
	  $(MAKE) --no-print-directory wgsl; \
	else \
	  echo "note: naga not installed -- skipping WGSL validation (run 'make tools')"; \
	fi
	@echo "OK: static checks passed"

.PHONY: wgsl
wgsl: ## validate every WGSL shader with naga (needs Rust-built naga)
	@command -v naga >/dev/null 2>&1 || { echo "naga not found -- run 'make tools' (needs Rust)"; exit 1; }
	@test -n "$(strip $(SHADERS))" || { echo "no shaders found matching shaders/*.wgsl (run from repo root?)"; exit 1; }
	@rc=0; for f in $(SHADERS); do \
	  if out=$$(naga "$$f" 2>&1); then echo "  ok    $$f"; \
	  else echo "  FAIL  $$f"; echo "$$out" | sed 's/^/        /'; rc=1; fi; \
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
# These targets talk to GitHub via `gh` and ALWAYS act on your own `origin`
# remote -- derived from the remote URL, NOT `gh repo view` (which resolves a
# fork to its PARENT and would make you operate on the wrong repo). So `make
# publish` publishes whichever repo you cloned from: run by the maintainer it
# publishes the canonical site; run by a fork owner it publishes their fork.
# The validation targets above need no network and no gh; only these do.

# Minimum major version of gh we rely on (Pages API + `gh auth status`).
MIN_GH_MAJOR := 2

# origin owner/repo from the remote URL (handles https and ssh forms).
ORIGIN_SLUG := $(shell git remote get-url origin 2>/dev/null | sed -E 's#(git@|https://)([^/:]+)[/:]##; s#\.git$$##')

.PHONY: require-gh
require-gh: ## check GitHub CLI (gh) is installed, current, and authenticated
	@command -v gh >/dev/null 2>&1 || { \
	  echo "gh (GitHub CLI) is not installed -- needed for 'make publish'/'make status'."; \
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

.PHONY: publish
publish: require-gh check ## publish current branch to origin's Pages branch (release)
	@slug="$(ORIGIN_SLUG)"; \
	pb=$$(gh api "repos/$$slug/pages" --jq .source.branch 2>/dev/null); \
	pu=$$(gh api "repos/$$slug/pages" --jq .html_url 2>/dev/null); \
	cur=$$(git rev-parse --abbrev-ref HEAD); \
	if [ -z "$$pb" ]; then echo "no GitHub Pages configured for $$slug -- set Settings > Pages first"; exit 1; fi; \
	if [ "$$pb" = "$$cur" ]; then \
	  echo "Pages serves the current branch ('$$pb') directly -- every push is already live; nothing to publish."; exit 0; fi; \
	if [ -n "$$(git status --porcelain)" ]; then echo "working tree not clean -- commit or stash first"; exit 1; fi; \
	src=$$(git rev-parse HEAD); short=$$(git rev-parse --short HEAD); \
	remote_pb=$$(git ls-remote origin "refs/heads/$$pb" 2>/dev/null | cut -f1); \
	if [ "$$remote_pb" = "$$src" ]; then echo "already published: $$slug@$$pb is at $$short"; exit 0; fi; \
	force=""; \
	if [ -n "$$remote_pb" ] && ! git merge-base --is-ancestor "$$remote_pb" "$$src" 2>/dev/null; then \
	  if [ "$$FORCE" != "1" ]; then \
	    echo "'$$pb' has commits not contained in $$cur (diverged)."; \
	    echo "  reconcile, or re-run with FORCE=1 to overwrite."; exit 1; fi; \
	  force="--force"; fi; \
	echo "publish $$short ($$(git log -1 --format=%s))  ->  $$slug@$$pb"; \
	echo "        live at: $$pu"; \
	if [ "$$DRYRUN" = "1" ]; then echo "DRYRUN: would run 'git push $$force origin HEAD:refs/heads/$$pb'"; exit 0; fi; \
	if [ "$$CONFIRM" != "1" ]; then printf "proceed? [y/N] "; read -r ans; case "$$ans" in y|Y) ;; *) echo "aborted."; exit 1;; esac; fi; \
	git push $$force origin "HEAD:refs/heads/$$pb"; \
	echo "published $$short -> $$slug@$$pb ; Pages rebuilds in ~1 min: $$pu"
