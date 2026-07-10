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
	  | awk 'BEGIN{FS=":.*## "}{printf "  make %-8s %s\n", $$1, $$2}'

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
