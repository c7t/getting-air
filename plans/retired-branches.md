# Retired branches and worktrees

What each dead branch was, what came out of it, and what is lost if it is
deleted. Written 2026-09-25, before the three worktrees under
`.claude/worktrees/` were removed, so that removing them costs nothing that is
not written down here.

A deleted branch's commits stay reachable through `git reflog` until garbage
collection (about 90 days by default), so an accidental delete is recoverable
for a while -- but only on the machine it happened on, and only for branches
that were never pushed if nobody has run `git gc`.

## Deleted, fully merged -- nothing lost

| Branch | Last head | Merged by | Notes |
|---|---|---|---|
| `3d/m0-m1-dense-solver` | `18e01a6` | PR #23 | The whole 3D fork, M0-M6.5 and M8.0-M8.3. Named for its first milestone and stale within days. |
| `3d/periodic-plate` | `181b7bc` | via #23 | Was checked out in a worktree misleadingly named `3d-trt`; nothing to do with TRT. |
| `amr2d/b6-merge` | `c4a3af0` | PR #24 | The integration branch that merged `amr2d/b6-explode` into a `main` that already had 3D. |
| `amr2d/b6-explode` | `b82518e` | PR #24 | The 2D back-port that shipped: explode/coalesce as the 2D interface, B0-B9, uniform levels, S8, the stride launch. Never pushed; fully contained in `main`. |
| `audit/m64-raymarch-boxes` | `8e502fc` | via #23 | An audit of the M6.4 raymarch boxes; every commit is in `main`. |

## `amr2d/backport` -- DEAD END, 29 commits never merged, never pushed

Head `79325fe` (2026-09-17), forked from `main`'s history at `a230ace` (B1-2).
It was the FIRST attempt at 2D B6, the coarse/fine interface; `amr2d/b6-explode`
redid B6 from the same starting point with a different explode/coalesce
implementation, and that is the one that shipped. So most of this branch is
an abandoned implementation of something that now exists another way.

**Salvaged into `main` (PR #25, 2026-09-24), so not lost:**

- `af1a31e` -- the pause button on `index-amr.html` and ONE writer for
  `liveMode`. Ported to both 2D pages, plus the paused-repaint fix it lacked
  (a paused page went black on resize).
- `ad46a2a` (B6-9b) -- the finding that NO AMR invariant read the fluid: the
  `field` column is the card body's numbers, which stay finite over a
  completely NaN fluid. Ported as `debugCheckFieldFinite` / the sweep's
  `fluid` column.
- `1677fb8` (B6-9c) -- NaN containment that counts itself (`forces[3]`).
  Ported, and in porting it we found the branch's own `x != x` test never
  fired (WGSL may assume finite floats); `main` tests the exponent bits.

**Only on this branch, and lost when it is deleted:**

- B6-1 .. B6-8e: the first explode/coalesce implementation, its consistency
  gate, and the corner/inflow-seam investigation. Superseded by
  `amr2d/b6-explode`'s B6, which reached the same kind of finding
  independently (its B6-3 force, B6-4 render and B6-6 newborn-tile ring reads).
- B6-9, B6-9d, B6-9e, B6-9f: fixes to ring consumers (criterion, force,
  renderer taps, the ring's velocity) OF THAT IMPLEMENTATION. The lessons
  carry -- "a ring under explode/coalesce is a delivery buffer, not a state",
  and "an admissibility rule for populations does not transfer to velocity"
  -- but the code does not. On `main` the second lesson is in
  `plans/uniform-levels.md`; the first survives only as a comment on
  `checkFieldFinite` in `amr2d-gpu.mjs`, so this line is its fullest record.
- `tools/validate-uniform-seam.js`, built on that branch's debug hooks.
- The CLAUDE.md / `plans/2D-backport.md` narrative of the above, from B6 on.
  `main`'s `plans/2D-backport.md` is `amr2d/b6-explode`'s version.

**To keep the history anyway**, before deleting:

    git tag archive/amr2d-backport amr2d/backport
    git push origin archive/amr2d-backport

A tag costs nothing, is not a branch anyone will build on, and survives the
worktree and the branch being removed.

## `worktree-3d-trt` -- KEPT on purpose, but it is LOCAL ONLY

Head `2ea1d12` (2026-09-13), 8 commits not in `main`, **never pushed**.
CLAUDE.md keeps it deliberately as the record of the TRT collision experiment
(built, measured, not adopted -- `plans/TRT.md` sec 8.10-8.14 is on `main`,
the code is only here). But a record that exists on one machine is one disk
failure from gone. Worth pushing as a branch or an `archive/` tag like the
above.
