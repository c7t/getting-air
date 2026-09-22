# Uniform levels: retiring the dense L0, and the interface buffering B6 needs

**The one-line summary.** The coarse/fine interface wants a *mailbox* — a
two-cell-deep ring that is an inbox in the inward directions and an
accumulator in the outward ones — and the tile already has one. What it does
not have is a substrate where that mailbox means the same thing at every seam,
because L0 is a dense grid, L1 is a pool with a dense parent, and levels >= 2
are pools with pool parents. **Three regimes, so every interface feature is
written and validated three times, and the one seam every shipped page uses is
the one least covered by tests.** This plan collapses them to one: the root is
a pool level like any other, special only in that it has no parent, is always
full, and is never allocated or freed.

B6 (explode/coalesce, plans/2D-backport.md) comes after. This is the substrate
it runs on.

---

## 1. The verdict, up front

| # | Stage | Status (2026-09-17) | Gate |
|---|---|---|---|
| U0 | The uniform level model on the host | **DONE** — 77 GPU-free checks, 8+6 mutants caught | `tools/test-amr2d.js` |
| — | B6's claimant rule, front-loaded | **DONE, and it ANSWERED the open question** (2.6a) | `transferLedger`, 9 shapes, scored against a wrong rule |
| U1 | The root pool exists, unused | **DONE** — identity proved on live buffers, page provably inert | `checkRootPoolIdentity`; hashes unmoved |
| U2 | The mirror | **WAS VACUOUS, NOW FIXED** — both "independent" routes wrote `gy*W+gx`; the dense grid is 8x8 block-major, so 98.4% of the pool read the wrong cell | a THIRD route (`field-reconstruct.js`'s `rawIndex`), GPU-free in `make check` |
| U3 | The step kernel serves the root | **DONE — BIT-IDENTICAL** on 8 rungs, 512 macro-steps; controls saturate at 98.4% | word equality + field health, `?rootstep=0` as control |
| U4 | criterion, force, digest, conserved totals | **DONE** — all four consumers on the root. Exact on the criterion, digest max and conserved totals; the force to the truncation floor | `tools/validate-root-kernels.js`, 8 rungs + 2 controls |
| U5 | L1 becomes a quad child of the root | **U5-0…U5-4 DONE under `?rootpool=1`** — both hops of the coupling and the manager. Level 1 is quad-allocated and quad-managed; tiles +28-32%; new fingerprints, default unmoved. `amr_manage.wgsl` cannot retire until the other four AMR pages get a root pool, and Cd/St is unmeasurable until then | `validate-root-kernels.js` (11 rungs, 4 controls); `validate-all.js` invariants ± starved pool; `measure-determinism.js` |
| U6 | The renderer walks levels | **DONE** — and it found that three of the five pages never drew level 2 either. Level 1 bit-identical to before; gate green on two pages and in the default sweep | `tools/validate-render-levels.js`, `tools/lib/render-levels.js` |
| U7 | One implementation across the five pages, then delete the dense path | **U7-0…U7-4 DONE** (layouts, coupling pipelines, per-level bind groups, both orderings, and now the root pool itself shared — net −1559 lines through U7-3, every gate unmoved; U7-4 puts a root pool on all five pages behind `?rootpool=1` and MEASURES U5-4's Cd/St, which U5 could not); **U7-5, U7-5a and U7-6a DONE** — `?rootpool=1` is the DEFAULT, pool demand re-measured at quad granularity, and the snapshot format now carries level 1's quad indirection AND the free list. The flip moved published Cd within the noise -- and that noise turned out to be the cylinder page's RACING allocator, whose attractor spread (~±0.009) is 10x the same-build repeat the gates quote. **D1 (port DET_SLOTS to the cylinder page) is now sequenced ahead of U7-6**, which deletes its control. U7-6 (delete the dense path) remains. **Split into U7-0…U7-6** — U1–U5 all landed on `main-amr.js` alone, so the remaining work is propagation before deletion. Measured: all 14 bind group layouts are byte-identical across five pages, and `S_Advance` is byte-identical across the other four | each rung byte-identical on the default path (`measure-determinism.js`), except U7-5 which is the one that moves numbers |

**Where the risk actually sits.** U0–U2 went in clean and each found something
(a half-cell convention, an f16 stride, a window-convention split between L0 and
the pool levels). U3 is the first stage where a kernel depends on the mapping,
and it is failing — which is the staging working as designed: the defect is
sitting in a flag-gated, inert code path with an instrument pointed at it,
rather than in the shipped solver.

**This plan corrected itself here, and then the correction was itself wrong.**
The table originally said U3–U4 are scored by bit-identity, justified by "the
arithmetic per cell is unchanged; only the address space moves". That was
retracted on the argument that `amr_step.wgsl` and `amr_step1.wgsl` are
separately written and owe each other no f32 association, and the bar was
lowered to agreement after one macro-step scored by magnitude.

**MEASURED, the original bar was right for the step and the retraction was
right for the force.** U3 is bit-identical on eight rungs at 512 macro-steps;
U4's force is not, on the same protocol. The rule that survives both is
narrower than either:

> Exactness survives a change of ADDRESS SPACE. It does not survive a change of
> ARITHMETIC or of REDUCTION ORDER.

Which of those a consumer suffers is decidable by reading it, BEFORE the
measurement — see U4-3/U4-4's table, where four consumers split three ways.
Bit-identity is still the right bar at U7, where the comparison becomes one
kernel against itself across builds.

And one prerequisite that is **not** part of this plan's thesis but should
probably come before all of it — see 1.1:

| # | Stage | What it does | Gate | Size |
|---|---|---|---|---|
| D0 | Deterministic slot assignment | replace the `atomicSub` free-list race with a rank from a prefix sum over the want set, on both grant and release | every config bit-reproducible **run to run**, not just within an attractor | S/M |

Sizes are of the *change*. U5 is the only stage that moves a published number,
and it moves it for a reason this project has already characterised (slot
assignment regroups the force reduction's truncated partials — see CLAUDE.md's
reproducibility note). Everything before U5 is a pure representation change and
must be **bit-identical**, which is a far stronger gate than this project
usually gets and is the main reason to stage it this way.

### 1.1 Sequencing — SETTLED 2026-09-17

**The order is: D0, then uniform levels, then conservation, then the sponge.**
Two things are pulled forward out of their stages: B6's host statement of the
claimant rule (into U0) and the sponge's *no-seam-in-the-ramp invariant* (before
conservation). The forced-refinement policy and the field-scaling ladders stay
at the end.

    D0   deterministic slot handout                    <- in progress
    U0   host model + B6's claimant rule
    U1..U5  uniform levels, on index-amr.html
    U7-0..3   share the layouts, pipelines, bind groups and scheduler
    U6        the renderer walks levels -- ride it on U7-2
    U7-4      the root pool everywhere (4a: share the solver half; 4b: call it)
    U7-6a     the snapshot format -- a PREREQUISITE for the flip, DONE
    U7-5a     pool demand at quad granularity, DONE
    U7-5      flip the default, DONE
    D1        DET_SLOTS on the cylinder page -- the Cd gate cannot resolve
              anything below ~0.02 without it, and U7-6 deletes its control
    U7-6      delete the dense path
    --   the sponge seam INVARIANT (gate only)
    B6   explode/coalesce
    --   sponge refinement policy, then section 8's ladders

The reasoning that produced it is below, kept because the counter-arguments
are the part worth re-reading if someone wants to reopen it.

---

The base order is right and the reasons are real: B6 written once instead of
twice; and the sponge/geometry work only *means* anything once a coarse root is
possible, which is after both. Three amendments, in descending order of how
much they matter.

**(a) D0 first, and it is the strongest thing on this page.** Most of the
argument for doing the L0 unification before B6 is a *validation* argument
(3.2): the L0/L1 seam is the one no exact-equality gate can reach. But the
reason it cannot be reached is not that L0 is dense. It is
`amr_manage_pool.wgsl:331`'s `atomicSub(&childFreeCount, 1)` — which slot a
block gets depends on which thread reaches the atomic first, so block->slot
assignment varies run to run, and `amr_force1.wgsl`'s per-workgroup truncated
`i32` partials then regroup and truncate differently.

Determinise the handout and the whole solver plausibly becomes bit-reproducible
run to run. Note the second half of that chain is already exact: integer
`atomicAdd` is associative and commutative with no rounding, so once each
workgroup's partial is fixed, the sum is order-independent. **Slot assignment
is the only live source of nondeterminism this file knows of.** Every attractor
table, every "the repeat must be on both sides", every "0.013 spread under
unknown load" in CLAUDE.md exists downstream of it.

The change is contained: each wanting block computes its rank among wanting
blocks in blockID order (a prefix sum over `childWant`, `NBLOCKS` entries — a
few thousand u32s, not per-cell), then takes `freeList[freeCount - 1 - rank]`.
The release path at line 418 needs the same treatment or the free list's
*contents* stay history-dependent. Binding budget is 10/16, so there is room.

**Test the hypothesis before building the compaction.** Add a debug mode that
assigns slots from the host in blockID order and writes them directly, then run
`?levels=2` and `?levels=3` four times each and diff. If they come back
bit-identical, the mechanism is confirmed and the parallel version is worth
writing. If they do not, there is a second source of nondeterminism and
everything below needs rethinking anyway. That is a day, and it de-risks every
stage that follows.

**(b) Front-load B6's host statement, whatever the implementation order.** The
one genuinely open question in this whole programme is the D2Q9 diagonal
claimant at a convex corner (2.6). It has no dependency on the substrate —
it is a rule about lattice directions and refinement geometry, statable in
`amr2d.mjs` and mutation-checkable with no GPU. Their ordering puts the riskiest
unknown late. Write the rule and its tests during U0, and if the corner turns
out to have a nasty answer, it changes what the substrate has to support while
the substrate is still being designed.

**(c) Split the sponge work: the invariant is a precondition for conservation,
the policy is not.** "No coarse/fine seam intersects the sponge ramp" (8.5) is
cheap, needs no coarse root, and is a *precondition for believing B6's numbers*
— inside the ramp the collision is modified, so a seam there transfers
populations produced under a different effective relaxation on each side, and
B6's transfer has no term for that. Land the gate before B6 so the sweep can
prove no config has one. The forced-refinement *policy* and the `root + j`
ladder stay where they are, at the end.

**The argument for conservation before uniform levels, and why it does not
win.** B6 is a known defect on the shipped page; the L0 unification is a large
refactor of the hottest kernel that moves published numbers at U5 and delivers
no physics. If the refactor stalls, the budget is gone and the defect is still
there. Against that: the mailbox belongs to the *child* tile, so B6's
duplication across a dense and a pool parent is maybe sixty lines in two
accessor files — not the cost. The cost is maintaining two copies of the most
subtle code in the project *while debugging it*, and doing it in the one place
the tests are weakest. With D0 done, that second objection loses most of its
force too — which is another reason D0 comes first: it is the cheapest way to
find out how much of stage 1's case was really about stage 1.

### 1.2 D0 — MEASURED 2026-09-17. Slot assignment is the only source.

`tools/measure-determinism.js`, `index-amr.html`, 4096 steps from `reset()`,
two runs per configuration, same build, scored by exact equality of every
`debugSnapshotSave` field payload. Quiet machine, no other Chrome holding a GPU
context; dedicated server on a pinned port, proven to serve this worktree by a
token grep before the run.

                           verdict     min ms / 4096 steps
    levels=2 detslots=1    IDENTICAL       286      7ac54e170f903ac3 x3
    levels=3 detslots=1    IDENTICAL       620      ce1bd4d8a3a1055c x3
    levels=2 detslots=0    DIFFERS         224      3846d936 c25b577b c88413a0
    levels=3 detslots=0    DIFFERS         562      4c515c11 033fdb8a 411a188c

**The hypothesis holds, and the discrimination rung is what makes it mean
something.** Without the flag `levels=2` produced three different states in
three runs, exactly as CLAUDE.md records; with it, three identical ones. So the
flag is not merely failing to perturb a configuration that was already stable.

**Stronger than the table shows: the `detslots=1` hashes reproduced ACROSS
SESSIONS.** An earlier two-run pass of the same build gave `7ac54e17...` and
`ce1bd4d8...`, and the three-run pass above reproduced both exactly, on a
separately launched Chrome. Five bit-identical runs at `levels=2` and five at
`levels=3`.

**One caveat on the baseline rows, because it is easy to over-read them.** This
tool's fingerprint covers the WHOLE snapshot — the pool indirection
(`blockSlot`, `slotToBlock`, `parentSlot`) as well as the field. CLAUDE.md's
attractor counts are field comparisons via `tools/amr-diff.js`. Two runs can
sit in the same field attractor and still differ here, because their slot
assignments differ. So "5 of 5 baseline runs differ" is NOT a contradiction of
"`levels=3` has about four reproducible field modes" — it is a stricter
instrument answering a stricter question. It makes the IDENTICAL rows stronger
(field *and* indirection agree) and the DIFFERS rows less comparable to the
recorded mode counts.

**What this buys, and it is larger than D0 itself.** The whole apparatus this
project uses to reason about AMR changes — attractor tables, "the repeat must
be on both sides", the +-0.001 / +-0.002 per-config reproducibility floors, the
0.013 spread measured under unknown load, "match the baseline in TWO different
modes" — is downstream of one `atomicSub`. With a deterministic handout, **any
AMR change can be scored by exact equality**, which is immune to GPU load and
cannot be forged. That is a better gate than anything in the current sweep, and
it is available to every stage after this one.

### 1.2b The cost, and what it localises

Same run, min of three, timed in the page around `debugStepSync`:

    levels=2     224 -> 286 ms     +27.9%
    levels=3     562 -> 620 ms     +10.3%

**Too expensive to default, so the parallel version is warranted** — decided by
measurement rather than by assuming a serial loop must be slow.

**But look at the shape, because it localises the remaining work to one
kernel.** The overhead is ~62 ms at `levels=2` and ~58 ms at `levels=3`: a
roughly CONSTANT absolute cost, not one that grows with level count. At
`levels=2` only `amr_manage.wgsl` runs; at `levels=3` `amr_manage_pool.wgsl`
runs too and adds essentially nothing. So **the dense manager's serial loop is
the whole cost and the pool manager's is free**, which is what the block counts
predict — the dense loop walks `NBX*NBY` = 4096 blocks while the pool loop
walks a few hundred parent slots. At `REFINE_EVERY = 16` that is 256 refine
rounds over 4096 steps, i.e. **~0.23 ms per round**, all of it in one thread.

**So only `amr_manage.wgsl` needs parallelising**, and the pool manager can
keep the serial loop it has.

### 1.2c-DONE The parallel handout is built, and it is PROVEN EQUIVALENT

`amr_manage.wgsl`'s deterministic path is now parallel: every candidate counts
the candidates below it and takes the free-list entry that rank names, with the
`blockSlot` half of the write deferred to `linkCoarsen`/`linkRefine` — separate
dispatches, so the rank's inputs cannot move while it is being counted. The
pool manager keeps its serial loop, which 1.2b measured as free. No new buffer,
no new binding, one page touched.

**The equivalence is exact, at both level counts, and that is the gate.**

    levels=2 detslots=1   7ac54e170f903ac3   serial build AND parallel build
    levels=3 detslots=1   ce1bd4d8a3a1055c   serial build AND parallel build

A parallel rank reproducing a serial loop bit-for-bit over 4096 steps of live
refinement churn is not something a wrong implementation does by accident. It
is worth more than any tolerance-based comparison of the two could have been,
and it was available only because D0 made the runs deterministic in the first
place — the gate paid for itself one stage after it was built.

**AND THE ONE INTERMEDIATE VERSION THAT DIFFERED EXPLAINS ITSELF.** The first
parallel draft returned `83197cbb...` at `levels=3` instead of `ce1bd4d8...`.
The cause was a race I had explicitly considered and wrongly dismissed: that
draft maintained `freeCount` incrementally, with every candidate thread storing
`count0 - min(granted, count0)` — so a thread could `atomicLoad` the count
AFTER another thread had already stored the new one, read the wrong `count0`,
and take the wrong free-list index. Deriving the count instead (see below)
removed every write from the dispatch, and the hash returned to the serial
build's.

**The lesson is the general one, not the specific bug.** "All threads compute
the same value, so concurrent identical stores are benign" is true of the
STORES and says nothing about the READS in the same dispatch. A dispatch that
writes a value it also reads has an ordering problem no matter how agreeable
the writers are.

**Derive the free count, do not accumulate it.** `freeCount` is the number of
unallocated slots, so it is a function of `slotToBlock`. `recountFree()` walks
the POOL (a few hundred slots, one thread, in a dispatch that writes nothing
else) instead of every candidate walking the DOMAIN. That removed the race
above, and it also means the deterministic path never has to re-derive the
incremental bookkeeping the racing path's `atomicSub`/`atomicAdd` pair exists
to get right.

### 1.2c-OPEN The cost did NOT come down, and the reason refutes the design note

    build                          levels=2        levels=3
    serial handout                 +27.9%          +10.3%
    parallel, incremental count    +26.8%          +3.8%
    parallel, derived count        +27.2%          +8.8%

`levels=3` moves around in the 4-10% band, which is the run-to-run spread on
this quantity, not a trend. **`levels=2` did not improve at all.**

So the rank scan itself is the cost, and the design note below was wrong where
it argued the scan would be cheap because "in steady state a refine round has
tens of new candidates, not thousands". On the card page at `levels=2` there
are hundreds of live level-1 blocks and the criterion churns them, so a refine
round has hundreds of candidates, each walking up to `nblocks` = 4096 entries.
That is ~1e6 element visits per round and 256 rounds in this measurement.

**So the remaining work is a real prefix sum — O(nblocks) total rather than
O(candidates x nblocks) — and it does need the buffer this design avoided.**
That is one `array<u32>` of `NBLOCKS` per level plus a scan pass, and it is a
FIVE-PAGE change (`main-amr.js` and the cylinder, channel, TGV and reentry AMR
pages), which is the change shape CLAUDE.md records a production breakage from.
It should land as one commit with a boot smoke on every page.

Until then `?detslots=1` stays a validation mode: correct, deterministic,
proven equivalent, and about 25% too slow to default at `levels=2`.

### 1.2d The prefix scan landed. The cost still has not.

`scanCandidates` in `amr_manage.wgsl`: one workgroup, 256 threads, each
counting its own chunk, thread 0 scanning the 256 chunk totals, each thread
then writing its chunk's ranks from its own offset. Serial depth
`nblocks/256 + 256` instead of `nblocks`. Two pipelines from one entry point
(`SCAN_RELEASE`) so the grant and release predicates cannot drift into two
spellings of "candidate".

The rank buffer took **binding 7**, one of the two holes B2-2d left, so no
existing index moved. It is allocated in `allocLevelPool` — one place, not
five — and bound by all five AMR pages. **All six pages boot-smoked** after the
binding change, which is the check CLAUDE.md records a production breakage for
skipping.

**Third implementation, same hashes.** Serial loop, per-candidate scan, and
prefix scan all produce `7ac54e170f903ac3` at `levels=2` and
`ce1bd4d8a3a1055c` at `levels=3`. Three independent routes agreeing bit-for-bit
is about as strong as correctness evidence gets here.

**And the cost did not move.**

    build                          levels=2        levels=3
    serial handout                 +27.9%          +10.3%
    parallel, per-candidate scan   +27.2%          +8.8%
    parallel, prefix scan          +26.5%          +6.8%

**It is overhead, not physics.** Active tile counts are IDENTICAL between
`detslots=0` and `detslots=1` — `[75]` at `levels=2`, `[103,240]` at
`levels=3`. A deterministic handout could have landed on a busier topology and
cost more honestly; it did not. `tools/measure-determinism.js` now reports tile
counts next to the timings, because a ms column without them cannot distinguish
the two.

**THE COST IS PARKED, DELIBERATELY, AND THAT IS THE END OF IT FOR NOW.**
`?detslots=1` is default-off and byte-identical when absent. Its job is to prove
two builds equivalent by bit-identity, and it does that on a run that takes
280 ms. A validation mode costing 26% more than the thing it validates is not a
problem; nothing ships with it on.

Three guesses at the cost were already wrong — the redundant total scan, the
per-candidate rank scan, and (below) the link passes — which is itself the
signal to stop. **Re-open this only if determinism is proposed as the
DEFAULT**, and then attribute it with the instrument rather than a fourth
guess: the remaining suspects are the four extra dispatches per refine round on
a desktop `plans/perf-characterization.md` characterises as pass-count bound,
and `recountFree()`'s single-threaded pool walk. That document's own lesson is
that the obvious optimisation here helped neither device.

### 1.2e The link passes are NOT redundant, and nobody knows why

Once `scanCandidates` took over the ranking, `refine`/`coarsen` no longer read
any other thread's `blockSlot` — each thread touches only its own entry. So the
deferral looked like two dispatches paying for a race that no longer existed,
and the link passes were removed and the writes put back inline.

**Three runs later, one had diverged**: `levels=3 detslots=1` read
`4aafcac0add6256c` on run 1 against the established `ce1bd4d8a3a1055c` on runs
2 and 3, with `[104,248]` tiles against `[103,240]`. Restoring the link passes
restored 3/3 IDENTICAL at both level counts.

**The mechanism is not understood.** The argument for removing them still looks
correct on inspection and is evidently incomplete. Candidates worth checking
before anyone tries again:

- `linkRefine` rewrites `blockSlot[slotToBlock[s]] = s` for EVERY live slot, so
  it is an idempotent repair. If the pool carries a latent
  blockSlot/slotToBlock inconsistency from somewhere else, the link pass has
  been hiding it every round — which would make its removal the messenger, not
  the cause. `debugCheckSlotQuadrants` and the indirection readback are the
  tools; an explicit inverse-consistency check does not exist yet and probably
  should.
- The pool manager at level 2 reads level 1's `slotToBlock` as
  `parentSlotToBlock`. Ordering there is by dispatch, and it was not re-derived
  when the link passes moved.

**The disposition until then: they stay.** They are two dispatches; the
divergence is a wrong answer. And the lesson is the same one this stage keeps
teaching — the incremental `freeCount` was also removed on an argument that
read correctly, and was also wrong. **On this kernel, an argument that a pass
is inert is a hypothesis with a cheap test; run the test.**

### 1.2c-DESIGN The reasoning that produced the current version

**Rank by scanning, do not store a scan.** Only CANDIDATES need a rank, and in
steady state there are tens of them, not thousands. Each candidate thread
counts the candidates below it:

    var rank = 0;
    for (var b = 0u; b < blockID; b++) {
      if (want[b] != 0u && blockSlot[b] < 0) { rank++; }
    }
    // then take freeList[freeCount - 1 - rank], refuse if rank >= freeCount

That is O(blockID) per candidate, in parallel, with no new buffer and no new
binding — which matters more than it sounds, because a binding change to these
two shaders is a **five-page** change (`main-amr.js` plus the cylinder, channel,
TGV and reentry AMR pages) and CLAUDE.md records a production breakage from
exactly that: a binding count mirrored into one page's bind group and not
another's.

**There is one race and it has a cheap fix.** The rank reads `blockSlot`, which
refine itself writes — a thread that scans after a lower-numbered thread has
granted counts one candidate too few and collides on a quad. `want` is
read-only during refine; `blockSlot` is not. The fix is to make it read-only
too: **have refine write only `slotToBlock` and `newlyActivated`, and derive
`blockSlot` in a following pass** that scatters `blockSlot[slotToBlock[s]] = s`
over slots. Both arrays are already bound, so it is a new ENTRY POINT, not a
new binding. `coarsen` has already cleared the released entries by then (the
order is decide -> cascade -> coarsen -> refine), so the scatter only has to
add, never clear.

The follow-up pass is a per-page dispatch addition — smaller than a binding
change, but still five pages, and it should land as one commit with a boot
smoke on every one of them.

**Then re-run `tools/measure-determinism.js` against it.** It must reproduce
the four verdicts, and the timing rows should close most of the gap above. The
`detslots=1` HASHES should be re-derived rather than expected to match: the
parallel version ranks by block id where the serial one serves in dispatch
order, so a *different* deterministic assignment is correct and not a
regression. Only the verdicts transfer, not the hashes.

### 1.2d The rule, on the host, mutation-checked

`amr2d.mjs`'s `grantAssignment` / `releaseAssignment`, with 9 GPU-free checks
in `tools/test-amr2d.js` (62 total, was 53). They return a FREE-LIST INDEX, not
a slot — the list's contents are shared state, and what the rule fixes is which
index each candidate reads, so keeping the two apart lets the GPU be scored
without the test having to model the pool's history.

Scored against six deliberate mutants, all caught, and with real
discrimination rather than everything reddening together:

    grant: no sort (arrival order)          4 checks fire
    grant: sort descending                  4
    grant: pop from the bottom of the stack 2
    grant: off-by-one capacity test         3
    release: push order reversed            1
    release: no sort                        2

The permutation-invariance check is the load-bearing one: it is the entire
property being bought, and a rule that merely looks ordered passes everything
else. One check was itself wrong on the first run — it asserted that extra
candidates were "refused" at a free count that could still serve them — and the
suite caught the fixture before it could bless the code.

**A note the rule records, correcting 1.2a's caution.** Serving in dispatch
order is not the weakness it was first written up as. A slot assignment is part
of the state and `debugSnapshotLoad` restores it, so dispatch order is
state-determined too. Ordering by block id is still the better rule — it is
geometry, so two runs that reached the same state by different routes agree —
but that is a refinement, not a defect to be engineered around.

**One process note worth keeping.** The first run of this reported "the page
failed to boot" on every configuration. The page was fine; the harness was
wrong — `waitForGlobal` throws on timeout and returns `undefined` on success,
so `if (!await waitForGlobal(...))` fires on the success path. Both new tools
had it. A boot-failure report from a *new* harness deserves a direct probe of
`#status` before it is believed.

---

### 1.2a The instrument, as built

`?detslots=1` on `index-amr.html`, default 0 and byte-identical when absent.
It replaces the handout in **both** managers — `amr_manage.wgsl` (L0->L1) and
`amr_manage_pool.wgsl` (level >= 2) — so `?levels=2` and `?levels=3` are both
covered. `__AMR.getDetSlots()` reports it, so a run can assert it was actually
on rather than assuming the URL took.

**How.** One thread does the whole handout in a serial loop; no atomic decides
anything. The pop discipline is unchanged (`freeList[count-1]`, then
decrement), so the only thing that moves is *which* candidate is served *when*.
The qualification test was factored into `refineWants` and the grant body into
`grantQuad`, shared by both paths, so the racing and serial routes cannot
disagree about which blocks are candidates — only about order.

**Two things it deliberately is not.**

1. **Not the shipping implementation.** A single thread is obviously correct
   and obviously deterministic, and far too slow to default on. The shipping
   version ranks candidates from a prefix sum over the want set and keeps the
   parallel dispatch. Build that only if the measurement below says the
   mechanism is what we think it is.
2. **Not the strongest available order.** The pool manager serves candidates in
   *dispatch-index* order, which is reproducible only from a deterministic
   initial state (`resetSim` writes an identity free list), so the assignment
   is a function of the whole run rather than of the current state. Enough to
   answer the question; not enough to ship — the real rule should order by
   **block ID**, which is geometry, and therefore makes a snapshot reload
   reproduce the same assignment. The dense manager already dispatches over
   blocks, so its serial path is block-ordered for free.

**The measurement** is `tools/measure-determinism.js`, one command:

    node tools/measure-determinism.js --baseUrl=https://localhost:<yours> --port=<yours>

The control rungs are not optional and are built into it. `?levels=3` already
reproduces bit-for-bit *within* an attractor, so a green `detslots=1` pair
there is consistent with the flag doing nothing — the discrimination is
`?levels=2`, which differs on every run today and must stop differing. The tool
refuses to report at all if that rung comes back IDENTICAL, because then it is
not measuring nondeterminism and no other row means anything. It also asserts
`__AMR.getDetSlots()` matches what the URL asked for, so a typo cannot read as
a clean negative.

---

## 2. Why the interface wants a mailbox, and why the ring already is one

This section derives the buffering. It does not depend on the rest of the plan
and should be read first, because the staging only makes sense once the target
is clear.

### 2.1 The requirement

On a uniform grid LBM conserves exactly because streaming is a **permutation**
of stored populations: every population leaves one cell and arrives in exactly
one. The composite grid must restore that property. Two rules do it:

1. **Partition, not overlap.** Every point of the domain is owned by the
   finest level covering it. A level-*m* cell covered by level *m+1* does not
   solve. Its storage is free for other use.
2. **Delivery by destination.** Whatever crosses a seam is written by the
   receiving side, so every population has exactly one claimant.

Rule 1 is also the *debit*, and this is the part that is easy to miss: once a
covered coarse cell stops solving, the population its uncovered neighbour
streams inward is read by nobody at level *m* — which is exactly right,
because the fine level consumed it. **No explicit "remove it from the coarse
books" pass is needed.** The partition rule pays for the explosion by
construction.

### 2.2 The geometry is already right

A level-(m+1) tile's ring is `GHOST = 2` fine cells deep. Two fine cells is
**exactly one parent cell**. So:

- A 2x2 block of ring cells corresponds to exactly one level-*m* cell.
- Over a macro-step the fine level takes 2 substeps, and a population travels
  one fine cell per substep, so it traverses **exactly the ring depth**.

Those two facts are the whole design. `main-amr.js:100` already says GHOST=2
"matches the 2-fine-substeps requirement"; what has never been written down is
that the same 2 buys the *outward* direction as well.

### 2.3 Inbox: explode

At time *t*, the parent cell P adjacent to the refined region has its
populations copied into the 2x2 ring block representing P's volume — **both
depths**. Then:

- substep A: the interior pulls from ring depth 1; ring depth 2 advects into
  depth 1; nothing refills depth 2.
- substep B: the interior pulls from depth 1 (which is depth 2's old content).
- at *t + dt* the ring inbox is exhausted, ready for the next explode.

**So the "half-step staleness" of the present scheme is not a defect to be
corrected here — it is structurally absorbed.** One explode per macro-step
feeds both substeps, because the ring is two deep and advects.

Conservation, in the density convention: P gives up `f_i(P)` over volume `h^2`.
Each of the 4 ring cells receives `f_i(P)` over volume `h^2/4`. Total
delivered = `f_i(P) * h^2`. Exact, and **explode is a plain copy** — no factor.
A *linear* reconstruction across the 4 children is the accuracy refinement, and
it stays conservative as long as the correction has zero mean over the block.

### 2.4 Outbox: coalesce

The same ring cells, in the outward-pointing directions, are the accumulator.
The fine interior streams outward into the ring; over 2 substeps that outflux
advects through the two ring depths. At *t + dt* the 2x2 ring block covering P
holds four contributions — 2 face cells x 2 substeps — one per ring cell.

Conservation gives the factor directly. Each ring cell holds mass
`f_i * h^2/4`; P must receive `f_i(P) * h^2`; therefore

```
f_i(P) = (1/4) * sum over the 4 ring cells of f_i
```

**Coalesce is the plain average of the 2x2 ring block, per direction.** No
moment rebuild, no rescale. The accumulation over both substeps is performed by
the ring's own advection, not by an atomic counter or a separate buffer.

### 2.5 Two consequences worth stating loudly

**(a) The ring's 9 direction-slots split by role, and the split is per
direction, not per cell.** For a ring cell R:

| directions where | role | written by | read by |
|---|---|---|---|
| `R + e_i` is interior or further in | **inbox** | explode | the fine step |
| `R + e_i` is further out | **outbox** | the fine step | coalesce |
| `R + e_i` is inside a refined *same-level* neighbour | neither | nobody | nobody (the step reaches through via `DIRECT_GHOST`) |

A ring cell is therefore **not a state**. Its nine numbers do not form a
distribution, its moments are not a density or a velocity, and any consumer
that treats them as one is wrong. *(CLAUDE.md on the other branch records this
as measured, with three consumers found violating it — `amr_force1.wgsl`,
`amr_criterion_pool.wgsl`, and `amr_render.wgsl`'s stencil clamp. Treat that as
a checklist of places to audit, not as a design input.)*

**(b) The ring must advect and must NOT collide, and that is what removes the
Dupuis-Chopard factor from the transfer.** Colliding a ring cell would apply
the fine level's relaxation to populations that belong to the parent's time
scale. The whole purpose of the Dupuis-Chopard rescale is to convert the
non-equilibrium part between the two levels' scalings — and if the delivered
populations are never relaxed at the wrong level, there is nothing to convert.
The compensation is exact, not approximate.

**This splits `interp` into two jobs that are currently one, and they should
not stay one:**

| job | when | what it does | rescale? |
|---|---|---|---|
| **initialise** a newly-activated tile | on refine, once | rebuild `f` from the parent's moments — you are *creating state* that did not exist | **yes** — B1's factor, `?dcpre=` stays live here |
| **deliver** across a seam | every macro-step | copy/average populations — you are *moving flux* | **no** |

Today `common_interp_kernel.wgsl` does both with `GHOST_ONLY` selecting
between them, and both go through `interpCoarseToFine`. They are different
operations and conflating them is why the rescale looked like it belonged at
the seam.

### 2.5a "COVERED" MEANS TWO DIFFERENT THINGS, and 2.6a depends on the second

Worth pinning before the derivation that rests on it.

A level-*m* cell is **covered** when it lies under an active level-*(m+1)*
tile. The granularity is the BLOCK, not the cell: refinement is granted per
`RB x RB` block, all or nothing, so coverage is uniform within a block.

What differs is what "covered" then implies.

**Today it means "solved twice, and one answer thrown away."** Verified in the
source, not assumed:

1. `amr_step.wgsl` contains no coverage test of any kind. It steps every cell
   of L0, including the ones under tiles, and `average` -- dispatched
   `(1,1,MAX_FINE_BLOCKS)` at `workgroup_size(8,8)`, i.e. exactly `RB x RB`
   parent cells per block -- then overwrites every covered parent cell with the
   restriction of its four children.
2. The RING is the same story one cell further out. A tile's ring is a
   fine-resolution copy of the parent cells around it, and those cells are
   UNCOVERED, so they really are solved at coarse pitch -- while `SKIP_GHOST`
   defaults to 0 and its skip sits inside `if (SKIP_GHOST != 0u)`, so the fine
   step collides and streams the ring as well. `interp` overwrites it next
   macro-step.
3. With three levels the chain repeats: a cell under a level-2 tile is
   integrated at level 0, level 1 and level 2. Three pitches, two discarded.

The wasted area is small -- 75 tiles x 64 = 4800 L0 cells of 262144 at W=512,
about 1.8%, plus the ring shells -- and the waste is not the problem. The
problem is that the duplicate leaves a PLAUSIBLE value in the covered coarse
cells, which the uncovered cell at the seam then pulls from. That is one of the
four unbalanced transfers: the coarse side consuming a restriction the fine
level was never debited for.

**Under the partition rule it means "not solved here at all."** Each point is
integrated by exactly one level, the finest covering it. Covered cells still
STORE values, as inbox and outbox; the ring stops being fluid entirely,
advecting without colliding.

**2.6a and `transferLedger` use the SECOND meaning.** `covered(x, y)` there is
"owned by the fine level", and the ledger's premise -- a population crosses the
seam iff one end is covered and the other is not -- is only true once the
coarse level has stopped solving covered cells. Applied to today's solver the
ledger would be describing a scheme that does not exist yet. It is a statement
about the target, and the partition is part of what has to be true for it.

### 2.6a ANSWERED (2026-09-17): the D2Q9 corner needs no special case

Derived, not inherited. A coarse cell `Q` that is not covered still solves, and
its update pulls `f_i` from `Q - e_i`. If `Q - e_i` is covered, that population
must come from the fine level. Over one macro-step a fine population travels
two fine cells — exactly one parent cell, exactly the ring depth — so the mass
that left `Q - e_i` heading along `i` has advected into the ring cells
occupying **Q's own volume**. Hence

    f_i(Q) = mean of the 4 ring cells covering Q, in the ring of the tile
             containing Q - e_i,    for every i where Q - e_i is covered

The mean, not the sum: each ring cell carries `f_i · (h/2)²` and Q must receive
`f_i(Q) · h²`. In 3D the factor would be 1/8.

**There is no separate corner case, and that is the finding.** A diagonal `i`
is handled by the same sentence as an axis direction, because `Q - e_i` is a
single parent cell either way and a parent cell lies in exactly one block. The
claim is decided by the coverage of **one cell**, never by the shape of the
boundary — so convex corners, concave corners and staircases are not distinct
situations.

**Scored by a ledger that can fail.** `transferLedger` in `amr2d.mjs` builds
the EXPECTED set of seam crossings from coverage alone (a population crosses
iff one end is covered and the other is not) and the ACTUAL set by applying the
claimant rule, then compares. On nine shapes — a lone tile (four convex
corners), two abutting tiles, a 2×2 square, an L, a plus, a diagonal staircase,
a hole with a coarse cell enclosed by fine on all eight sides, and both
degenerate cases — every crossing has exactly one claimant and every claim has
exactly one crossing, in both directions.

Three things keep it from being vacuous:

- The interesting shapes are asserted to carry traffic, and the two degenerate
  ones to carry none — the discrimination that says the ledger reads shape.
- The diagonals are asserted to carry a real share (`all.exits > axis.exits`),
  so a corner rule could not have hidden in a channel nothing uses.
- **The ledger is run against a WRONG rule** — claiming at `Q + e_i`, the most
  plausible slip and the one that looks right in a diagram — and it reports
  unclaimed exits. An audit that merely re-applies the rule it audits proves
  nothing, and this project has three gates like that already.

**What it does NOT say.** This is bookkeeping on the geometry: it says the
transfer CAN be a bijection and names the rule that makes it one. It says
nothing about the implementation, and nothing about accuracy —
**conservation does not imply consistency**, which is section 7's separate
warning and the reason B6 needs two gates rather than one.

### 2.6 What is genuinely open

**The diagonal claimant at a convex corner.** D2Q9 carries `(+-1,+-1)`, so the
3D counting argument (D3Q19 has no `(1,1,+-1)`) transfers nothing. For a
diagonal direction the ring block that delivers to P is not the block directly
across the seam, and at a convex corner the claimant can fall outside the
refined region, inside a *sibling* tile, or nowhere. The rule must be **stated
once on the host** (`amr2d.mjs`), mutation-checked GPU-free, and scored against
the GPU — the same treatment `cascade21` got in B2. Do not let it exist only in
WGSL.

The instrument that separates a corner effect from a correction bug is a
**declared, corner-free seam geometry** as the control. That is what
`?refine=slab` is for on the TGV page.

---

## 3. What L0's denseness actually costs

### 3.1 Three regimes, measured

| | L0 | L1 | L >= 2 |
|---|---|---|---|
| storage | dense `W*H`, `cellIndex()` + wrap | pool, `(slot, lx, ly)` | pool, `(slot, lx, ly)` |
| ring | **none** | 2 cells | 2 cells |
| parent | none | **the dense grid** | a pool tile |
| allocation | n/a | **per block** | per quad (stride 4) |
| `parentSlot` / `quadrant` | n/a | **absent** | present |
| step kernel | `amr_step.wgsl` (245) | `amr_step1.wgsl` (464) | same |
| criterion | `amr_criterion.wgsl` (73) | `amr_criterion_pool.wgsl` (86) | same |
| manager | `amr_manage.wgsl` (295) | `amr_manage_pool.wgsl` (427) | same |
| force | `amr_force.wgsl` (206) | `amr_force1.wgsl` (273) | same |
| interp accessor | `common_interp_parent_dense.wgsl` (83) | `..._pool.wgsl` (93) | same |
| average accessor | `common_avg_parent_dense.wgsl` (41) | `..._pool.wgsl` (40) | same |

**~940 lines of WGSL exist only because the root is dense**, plus the host
bind groups, pipelines and staging buffers that feed them. Every one of those
pairs is a place where a change lands on one side and not the other. CLAUDE.md
already records that failure mode twice: the binding-count change mirrored into
`main-cylinder-amr.js` but not `main-amr.js`, and B6-8e's "N>=3's defect pinned
to ONE pass".

### 3.2 The validation asymmetry is the real argument

This is the part that matters more than the line count.

- The seam every shipped page uses is **L0/L1**. It is the one with a dense
  parent, so it is the one the pool machinery's gates cannot reach.
- `?levels=2` has **no reproducible baseline** — every run differs at ux relL2
  2-4e-5, so `tools/amr-diff.js` cannot gate there at all.
- `?levels=3` reproduces bit-for-bit within an attractor, so it *can* be gated
  by exact equality.

So today the only seam that can be scored by bit-identity is the one no default
configuration exercises, and making L1 a quad child of the root makes the root
seam **structurally** identical to the L1/L2 seam.

**But structural identity is not reproducibility, and an earlier draft of this
section overclaimed it.** `?levels=3`'s attractors and `?levels=2`'s lack of
them are empirical properties of the `atomicSub` free-list race, not of the
dense/pool split. Quad allocation puts one thread per parent tile instead of
one per block, so there are 4x fewer racers and possibly fewer modes — "fewer
modes" is not "reproducible". **Nothing in U1-U7 is guaranteed to make the root
seam bit-testable; D0 is what does that** (see 1.1). Read this section as an
argument about duplication and drift, which it is, and not as one about
testability, which D0 owns.

### 3.3 Honest counter-argument

The dense root is **not** a structural blocker for section 2's design. The
mailbox belongs to the *child* tile, so a dense parent can be exploded from and
coalesced into perfectly well; the accessor pair just needs two more entry
points. If the only goal were "ship B6", this plan is optional.

The reasons to do it anyway are duplication, drift, and the testability
asymmetry above — not impossibility. Anyone re-reading this should weigh U5's
risk against that, not against a claim that B6 cannot proceed otherwise.

---

## 4. The design

### 4.1 One rule, three consequences

> **The root level is a pool level that has no parent, is always full, and is
> never allocated or freed.**

**Tile shape.** Every level's tile holds the same `2*RB x 2*RB = 16x16`
interior of its *own* cells. Today an L1 tile covers `RB x RB` L0 cells; making
the root's tile `2*RB x 2*RB` root cells means an L1 tile covers exactly one
**quadrant** of a root tile — which is precisely the relation L2 already has to
L1. The three regimes collapse to two: *root*, and *everything else*.

Root block grid: `NBX_0 = W / (2*RB)`, i.e. half today's `NBX = W / BLOCK` in
each axis. `W = H = 2^resLog2` (`main-amr.js:88`), so `NBX` is a power of two
and the halving is exact by construction at every supported resolution — no new
constraint on the user.

**Ghost depth is per level, and the root's is 0.**

```
GHOST_m = 0 for m = 0,   2 otherwise
FB_m    = 2*RB + 2*GHOST_m      ->  16 at the root, 20 elsewhere
```

The root has no parent, so no parent-interface mailbox; and it is always full,
so `DIRECT_GHOST` always resolves an out-of-tile source against the owning
same-level tile and never falls back. **The root's ring would be dead storage,
so it does not get one.** This keeps root memory at exactly today's `W*H` cells
— `(W/(2RB)) * (H/(2RB)) * (2RB)^2 = W*H` — rather than paying the 56% ring
overhead a uniform `FB` would cost. Per-level strides are already derived per
buffer (`arrayLength(&f_pool) / 9u`), so this is cheaper to do than it sounds.

**The indirection is the identity.** `slotToBlock[i] = i`, `blockSlot[i] = i`,
permanently. Compile an `IDENTITY_SLOTS` override that folds the lookup away,
default on for the root — so if the extra buffer load costs anything on the
bandwidth-bound phone, the escape hatch is already there and A/B-able rather
than a redesign.

### 4.2 The macro-step, re-derived

Under partition the pass order **inverts**, and the derivation is one line:
coalesce must have finished before the level it delivers into takes its step,
and coalesce cannot finish until the child's substeps have run. Therefore
**a level's own step runs last in its own cycle**, where interp/average runs it
first.

```
S_Advance(m):                       # advances level m by two of its own steps
    explode  m -> m+1               # from m's state at t, fills both ring depths
    S_Advance(m+1)                  # child's full cycle
    coalesce m+1 -> m               # 2x2 ring average into m's seam cells
    step m  (substep A)             # reads m at t, plus what coalesce delivered
    explode  m -> m+1               # from m's new state
    S_Advance(m+1)
    coalesce m+1 -> m
    step m  (substep B)
```

At the root, `step 0` runs once per macro-step and runs **last**.

Two instruments become invalid the moment this lands and must be fixed in the
same change, not after: anything that sums the root level alone (a covered root
cell is now an outbox, not fluid) and anything that scores ring cells as a
distribution (explode writes some directions zero *on purpose*).

### 4.3 What stays exactly as it is

- `index.html` / `main.js` and the dense cylinder/channel/TGV/reentry pages.
  They are a separate single-level solver; the dense reference
  `validate-all.js` compares against is untouched.
- `RB = 8`, `GHOST = 2` for non-root levels, the 2:1 balance closure, the
  criterion, the body-in-buffer-coordinates convention (B5), the sponge, the
  walls, bounce-back, `SOLID_EQ`.
- The refinement *policy*. This changes representation, not when tiles appear.
  **One exception is already known and it is not small: the sponge.** Its band
  is a fixed strip of L0 window cells, which is only meaningful while L0 is the
  root. The moment a coarser root is possible the rule has to be restated, and
  it probably inverts from "exclude refinement here" to "force it". See
  section 8.5 — it is the one term that can break section 8.2's cost claim.

---

## 5. The staged plan

Each stage is independently mergeable. **Merging publishes**, so every stage
needs its own gate before the merge, not before a release.

### U0 — PARTLY DONE (2026-09-17). The level model and the mailbox geometry.

In `amr2d.mjs`, mutation-checked GPU-free in `tools/test-amr2d.js` (70 checks,
was 62). Nothing consumes them yet — they exist first and alone, which is the
order B0 used.

    tileCellsAtLevel(m, rb)        2*RB at every level, root included
    ghostDepthAtLevel(m)           0 at the root, GHOST below it
    tileSideAtLevel(m, rb)         16 at the root, 20 elsewhere
    blockGridAtLevel(dims, m, rb)  domain / a whole root tile, doubling per level
    parentCellOfFineCell(f, m)     tile-local fine -> parent-local, -1 across the ring
    fineCellsOfParentCell(p, m)    the inverse
    ringDepth(fx, fy, m, rb)       0 interior, 1..2 ring, identically 0 at the root
    ringSlotRole(fx, fy, ex, ey)   interior | inbox | outbox | tangential | rest | offtile
    poolInverseViolations(bs, stb) blockSlot and slotToBlock are inverses

**The block grid is scored against an independent route.** Today's convention
is "level 1 is 1:1 with L0 blocks, then double per level", derived from `W/RB`;
the new one is "the domain over a whole root tile, then double per level",
derived from `W/(2*RB)`. They share no arithmetic, and they agree at every
level from 1 to 4 — which is evidence rather than tautology, and it is also
what says the new model extends the old one downward rather than replacing it.

**The ring-is-one-parent-cell fact is now asserted, not assumed.**
`parentCellOfFineCell` returns `-1` across the whole low ring and `rb` across
the high one, and the two directions are proved inverses by round trip over the
ring-inclusive range — the same discipline `dense-to-amr.js` and
`field-reconstruct.js` are scored by, rather than by eye.

**`ringSlotRole` names `tangential` instead of guessing.** Whether a direction
whose target sits at the same ring depth must be written, zeroed or left alone
is section 7's open question 2, and a function that answered it would hide it.

**`poolInverseViolations` is written down because D0 gave a reason to doubt
it** (1.2e): removing the link passes — an idempotent rewrite of `blockSlot`
from `slotToBlock` — made a deterministic run diverge with no nameable
mechanism. This is the predicate a GPU checker should score, and it is checked
from BOTH directions, because the reverse one is exactly what a one-sided check
would miss and what `linkRefine` would silently repair.

Scored against eight mutants, all caught, with discrimination (2/3/1/1/2/2/1/1
checks firing).

**B6's claimant rule is DONE too, and it answered the open question** — see
2.6a. `coalesceSource`, `explodeTarget` and `transferLedger` in `amr2d.mjs`,
with four checks (74 total, was 70): the ledger balances on nine shapes
including a hole and a staircase, the shapes are asserted to carry traffic, the
diagonals are asserted to carry a real share of it, and the ledger is run
against a deliberately wrong rule and reports the failure.

Front-loading this was worth it. The answer is that **there is no corner case**,
which means the substrate does not have to support one — and that was exactly
the thing worth knowing before designing the substrate rather than after.

**Still to do in U0:** nothing blocking. The remaining host work is whatever
U1–U7 turn out to need, and the GPU half of 2.6a (scoring the shipped kernel
against `transferLedger`) belongs with B6, not here.

### U0 — the original statement, for reference

Extend `amr2d.mjs` with the uniform level model and the mailbox rules:

```
tileCellsAtLevel(m)            2*RB always
ghostDepthAtLevel(m)           0 at the root, 2 otherwise
blockGridAtLevel(m)            root NBX/2, then the quad recursion
rootIsFull / rootHasNoParent   the two constraints, as predicates
ringBlockOfParentCell(...)     parent cell  -> its 2x2 ring block
parentCellOfRingBlock(...)     the inverse, by construction
ringSlotRole(cell, i)          inbox | outbox | unused        (section 2.5)
claimantOfRingCell(cell, i)    the delivery rule, corners included (section 2.6)
```

**Gate:** `tools/test-amr2d.js`, GPU-free, mutation-checked. The two
`ringBlockOfParentCell` / `parentCellOfRingBlock` directions must be proved
inverses the way `dense-to-amr.js` and `reconstructAMRToResolution` already
are — by round trip, not by eye.

Nothing in the app changes. This is B0's shape and it is what makes every later
stage scorable against something.

### U1 — DONE (2026-09-17). The root pool exists and is provably inert.

`?rootpool=1` on `index-amr.html`, default 0. With the flag off nothing is
allocated at all — this is not "allocated and unused", it is absent, so the
phone never pays ~21 MB for a buffer no kernel reads. With it on the pool
exists, carries the identity indirection, and is still read by nothing.

`allocLevelPool` learned `m === 0`: same buffers, same addressing, differing
only in the three things that follow from having no parent — every slot
permanently assigned, nothing ever granted or released, `newlyActivated` never
firing. **It REFUSES a root pool that is not exactly full**
(`maxFineBlocks !== NBLOCKS_m`), because a root short of its own domain would
be a silent hole in the grid, and `poolSlotsFor`'s 1.7x headroom has no meaning
for a level that never grows.

**The identity is scored on the LIVE buffers**, not on the host's intent —
`checkRootPoolIdentity` reads `blockSlot`/`slotToBlock` back and checks both
that each is the identity and that they are inverses
(`poolInverseViolations`), the same discipline `debugCheckSlotQuadrants` uses
on `quadrantBuf`. Measured at `?levels=2,3,4`:

    ?levels=2                spec=null    check={"ok":null,"skipped":"no root pool allocated"}
    ?levels=2&rootpool=1     nbx=16 nby=16 slots=256 side=16 cells=65536   ok, 0 violations
    ?levels=3&rootpool=1     (same)                                        ok, 0 violations
    ?levels=4&rootpool=1     (same)                                        ok, 0 violations

`cells = 65536 = 256 x 256`, which IS the domain — the tiled root costs exactly
the dense grid it will replace, no padding. That identity is asserted on the
host too, across W = 128..1024, next to the contrast that a *ringed* root would
cost 1.56x. Ghost depth 0 at the root is load-bearing, not tidiness.

**The gate: it changes no number.** `--extra=rootpool=1` across the whole
determinism table still reads

    levels=2 detslots=1    IDENTICAL   7ac54e170f903ac3
    levels=3 detslots=1    IDENTICAL   ce1bd4d8a3a1055c

the exact hashes established before the root pool existed, with tile counts
unchanged. A correctly shaped root pool is inert, which is the whole of U1's
claim.

**One tool bug found and fixed on the way.** `--extra=` was in
`measure-determinism.js`'s defaults and its usage text and NOT in `parseArgs`,
so it was rejected as an unknown argument. That is the better of the two
possible failures and only by luck — had it been parsed and dropped, this gate
would have reported a clean pass for a run that never enabled the flag.

### U1 — the original statement, for reference

Allocate root tiles through `allocLevelPool` with a new `m === 0` branch (no
free list, no `newlyActivated`, identity indirection written once). Add the
root to the invariant sweep's tile-origin and slot-quadrant checks, where it
should pass trivially.

**Gate:** the page is byte-identical with the pool allocated and unused. Assert
the identity indirection on the GPU.

### U2 — CORRECTED (2026-09-17). IT WAS VACUOUS: BOTH "INDEPENDENT" ROUTES SHARED A PREMISE.

Everything below in the original U2 entry is true and none of it was the gate
it claimed to be. The mirror and `rootCellToDense` both ended in `gy*W + gx`.
**The dense L0 grid is not row-major.** `shaders/amr_step.wgsl`'s `cellIndex`
groups it into fixed 8x8 buffer-space blocks, block-major, row-major within a
block, and every snapshot carries `layout: 'block8'` to say so. So the mirror
fetched the wrong dense cell for **98.4% of the root pool** (measured over
512x256 at RB=8; 99.2% at 256x256), the checker scored it through the same
wrong map, and U2 read `0/589824` on every rung.

**The stale-must-be-dirty control did not help, and could not.** It asks
whether the comparison reads live data. It does. A consistent permutation is
live data compared against itself.

**THE PROPERTY THE TWO-ROUTE CHECK ACTUALLY TESTED IS BIJECTIVITY, NOT
CORRECTNESS.** `gy*W + gx` is a perfectly good bijection from root cells onto
dense cells; it is simply the wrong one. That is why the fix has a third route
rather than a better second one:

    amr_mirror_root.wgsl      the shader's dense index
    amr2d.mjs denseCellIndex  the host's, stated once and exported
    field-reconstruct.js      rawIndex(..., 'block8') -- the decoder that reads
                              REAL GPU SNAPSHOTS into the fields the validation
                              tools compare against literature

The third is the anchor, because it is validated by data rather than by another
formula in this repo. `tools/test-amr2d.js` scores `denseCellIndex` against it
over every cell of three domains, asserts the root map is a bijection, asserts
each root cell lands on the dense cell at its own COORDINATES, and carries the
shipped row-major map as a named mutant. GPU-free, in `make check`.

**The mutation check is itself informative, and worth reading before writing
the next "independent route" claim.** Of the three new checks, exactly ONE
catches the shipped map:

    the dense index agrees with the snapshot decoder      passes either way (*)
    every root pool cell maps ONE-TO-ONE                  passes either way
    a root cell stands for the dense cell at its COORDS   FAILS on the mutant

    (*) it fails if denseCellIndex itself is mutated; it cannot see
        rootCellToDense at all

A bijection check and an agreement check can both be green over the exact bug
they were written for. The check that bites is the one that COMPOSES the two
sides -- take the (gx,gy) a root cell geometrically IS, ask the dense grid
where that lives, and require the map to have landed there.

**Independence is about PREMISES, not authorship.** Two routes written by
different people at different times, consulting nothing, still agree if they
share an unstated assumption -- and the assumption here was never written down
anywhere either route could be checked against. It is now, in `amr2d.mjs`'s
`DENSE_BLOCK`/`denseCellIndex`.

**What it cost:** U3 spent a day on "a systematic defect in the root step, not
localised", ruling out `dxL`, the diffuse band, the tile origin, the
DIRECT_GHOST offsets and the sponge convention one at a time. The defect was
not in the step kernel. See U3 for the A/B.

**`tools/validate-root-mirror.js` still passes, and its stale column MOVED** —
`levels=2` reads 281669/589824 after 64 steps where it read 582330 before.
That is the root now stepping the RIGHT field and therefore tracking the dense
grid instead of diverging from it. The rung that says the fix landed is U3's,
not this tool's: this one was green over the bug and is green over the fix.

### U2 — as it was written (2026-09-17). The addressing is proved, and it found a *stride* bug.

`shaders/amr_mirror_root.wgsl` copies the dense L0 grid into the root pool;
`tools/validate-root-mirror.js` compares the two word for word against an
INDEPENDENT host route — `amr2d.mjs`'s `rootCellToDense`, which derives the
dense index from `blockGridAtLevel` where the shader derives it from
`slotToBlock` and its own overrides. Neither consults the other.

**The clean result is not the gate.** A comparison that never executes, or that
compares a buffer against itself, reports zero mismatches too. So every rung
runs twice — mirror then check (must be CLEAN), step 64 then check again (must
be DIRTY) — and a stale pool that compares clean is reported as a broken
instrument rather than a pass.

    levels=2&rootpool=1          mirrored 0/589824    after 64 steps 582330/589824
    levels=3&rootpool=1          mirrored 0/589824    after 64 steps 584924/589824
    levels=4&rootpool=1          mirrored 0/589824    after 64 steps 584992/589824
    levels=2&rootpool=1&f16=1    mirrored 0/327680    after 64 steps 306802/327680
    levels=2&rootpool=1&res=9    mirrored 0/2359296   after 64 steps 2343571/2359296

**IT EARNED ITS KEEP ON THE FIRST RUN.** The mirror computed its plane stride
as `arrayLength(&f_root) / fWords()`, which is correct at the default and puts
four of every cell's nine planes at the wrong offset under `?f16=1`. Every `f`
buffer in this project is sized for NINE planes whatever the packing is, and
under `?f16=` only the first five are used — so the STRIDE does not move, the
plane COUNT does. `amr_step1.wgsl`, `common_average.wgsl` and
`common_interp_kernel.wgsl` all say `arrayLength(&f_pool) / 9u`.

**Only the f16 rung saw it**, which is the argument for the rungs not all being
the default: levels vary the pool count, `f16` varies the packing, `res` varies
the domain and so the root block grid. A single-rung version of this gate would
have passed and handed U3 a stride bug to find as a wrong field.

That is exactly the trade this stage exists to make. An addressing error found
here cost an afternoon; found at U3 it arrives as a wrong field and reads like a
physics regression, because by then the mapping is inside a step kernel.

**One deviation from the original plan, deliberately.** It mirrors ON DEMAND,
not every macro-step. A per-macro-step copy would make the pool's contents a
function of when you look, and the thing under test — the addressing — is
static. It also binds `f_a`, which is the current dense buffer only at rest on
an even macro-step; `debugSnapshotSave` relies on the same invariant, so
"mirror, then check" is meaningful from exactly the state a snapshot is.

**And the page is still inert.** `--extra=rootpool=1` across the determinism
table still reads `7ac54e170f903ac3` and `ce1bd4d8a3a1055c` with the mirror
machinery present. (That run's timings were visibly noisy — which is the
standing argument for scoring these by bit-identity rather than by
milliseconds.)

### U2 — the original statement, for reference

Add a debug pass that copies the dense L0 into the root pool each macro-step
and a checker that asserts **bit-equality** cell by cell. Run it across every
config in the sweep.

This is deliberately the same order of work as B3-5: *prove the new addressing
against the live buffer first, then move anything onto it.* A tiling or stride
error found here costs an afternoon; found at U3 it looks like a physics
regression.

**Gate:** zero mismatched cells, on every config, at every checkpoint. Any
nonzero count is an addressing bug, full stop.

### U3 — DONE (2026-09-17). BIT-IDENTICAL, at the bar this plan had talked itself out of.

`amr_step1.wgsl` compiled a second time with `GHOST = 0`, `NO_PARENT = 1`,
`SPONGE_CELL_SNAP = 1` and `DIRECT_GHOST = 1`, dispatched on the root pool
alongside the dense L0 step and sharing its `useB` ping-pong. It reads and
writes the root pool only; the dense path stays authoritative.

**THE RESULT: THE TWO KERNELS PRODUCE THE SAME FIELD, WORD FOR WORD.** With
`?benchSkip=avg` — the step kernel is then the ONLY writer of the dense L0
buffers, see below — 512 macro-steps on `index-amr.html`:

```
                                          differing/checked   maxAbs    field
  levels=2                                      0/589824      0.0e+0    ok, max|u| 0.0166
  levels=3                                      0/589824      0.0e+0    ok, max|u| 0.0207
  levels=4                                      0/589824      0.0e+0    ok, max|u| 0.0248
  levels=2 f16=1                                0/327680        --      ok
  levels=2 f16=2                                0/327680        --      ok
  levels=2 res=9                                0/2359296     0.0e+0    ok, max|u| 0.0123
  levels=2 ghostcopy=1                          0/589824      0.0e+0    ok
  levels=2 dcpre=1                              0/589824      0.0e+0    ok
  CONTROL ?rootstep=0 levels=2             580623/589824      5.7e-3    ok
  CONTROL ?rootstep=0 levels=3             580631/589824      7.0e-3    ok
```

**The control is what makes the zeros mean anything**: mirror, step, do NOT step
the root, and 98.4% of every word differs. The comparison is live and it
saturates.

**SO THE PLAN'S ORIGINAL GATE WAS RIGHT AND ITS RETRACTION WAS WRONG.** U3-as-
written said score this by bit-identity; the in-progress entry retracted that on
the argument that `amr_step.wgsl` and `amr_step1.wgsl` are separately written
kernels and "nothing entitles them to associate their f32 identically". Nothing
entitles them to in general -- but they do, because they perform the same
operations in the same order per cell and only the ADDRESS SPACE moves. The
retraction was reasoning about what the code might do instead of asking it.

**IT WAS NOT THE STEP KERNEL. IT WAS U2'S MIRROR.** The root was stepping a
PERMUTED field: correct neighbour topology, wrong data in it. A/B on this tree,
one protocol, 512 macro-steps, `?levels=2`:

```
                              row-major mirror          block8 mirror
  ?benchSkip=avg rootstep=1   584021/589824  3.10e-3    0/589824  0.0e+0
                 rootstep=0   580623/589824  2.22e-3    580623/589824  2.22e-3
  shipped path   rootstep=1   588047/589824  2.88e-3    580531/589824  7.08e-4
                 rootstep=0   580619/589824  1.93e-3    580619/589824  1.93e-3
```

**Both controls are BIT-IDENTICAL across the two builds** — 580623 and 580619,
to the word, and relL2 to every digit. That is exactly what a bijective
permutation must give (the same cells, renamed) and it is the anchor saying
nothing but the root step moved. It is also why the control could not have
detected the bug: the permutation cancels in it.

**Why the wrong field read as a plausible 2e-3 rather than an obvious O(1).** A
root tile is 2*RB = 16 cells and a dense block is 8, so a root tile is exactly
2x2 dense blocks — the same cells in a different order. The error is therefore
bounded by the field's own spatial variation, which two macro-steps from
`reset()` is ~1e-2 in `f`. A scramble of a nearly-uniform field still looks like
a field.

**THE RESIDUAL ON THE SHIPPED PATH IS `average`, AND THAT IS NOT A DEFECT.**
7.08e-4 against a 1.93e-3 control — the root tracks the dense grid four times
better than freezing it does, but not exactly. The attribution is structural
rather than statistical: the only two pipelines that BIND a dense L0 `f` buffer
as writable are `stepPL` (`stepBG_ab/ba` binding 2) and `avgPL`
(`avgBG_targetA/B` binding 2). Everything else — force, interp, criterion,
render — reads. So with `avg` skipped the step kernel is the sole writer and
bit-identity is a complete statement about the two kernels; with `avg` on, the
whole residual is what the L1->L0 restriction writes, plus its downstream
spread. The root pool receives no restriction yet. **That is U4/U5's subject,
not U3's.**

**One real defect found on the way, and it is the sort a rung ladder exists
for.** `?ghostcopy=1` read 580623/589824 — indistinguishable from not stepping
the root — while every other rung was zero. Cause: `step1Constants` carries
`DIRECT_GHOST: GHOST_COPY ? 0 : 1` and the root pipeline spread it. The legacy
path clamps at the slot's own buffer edge and reads a ghost cell a separate
fine-fine copy pass filled; the root has `GHOST = 0`, so there is no ring to
clamp into and no pass that fills one, and it streamed from its own edge cells.
The root is also always full, so the direct path never falls back and the legacy
path has nothing to offer it. `DIRECT_GHOST` is now pinned to 1 on that
pipeline, with the measurement in the comment.

**Two findings from the in-progress entry stand, both fixed there:**

1. **The half-cell offset.** `fineToCoarseUnit` places cell j at
   `origin - 0.5*dxL + dxL*j`, where `origin` is the centre of the first PARENT
   cell and the two children straddle it. The root's `origin` is its own first
   cell's centre — nothing to straddle — so the term is 0 there (`NO_PARENT`).
   No single-kernel test could have seen it: the root is perfectly
   self-consistent with the offset, just displaced.
2. **L0 and the pool levels use DIFFERENT WINDOW CONVENTIONS (recorded, not
   fixed).** `amr_step.wgsl` converts with `bufferToWindowCell` — u32 modular
   arithmetic, so it TRUNCATES `off_x`/`off_y` to whole cells. `amr_step1.wgsl`
   uses `bufferToWindowPos`, which keeps the sub-cell part. So whenever the
   window offset is fractional — essentially always on the falling card — L0's
   sponge band sits up to half a cell from where every finer level's does.
   `SPONGE_CELL_SNAP = 1` makes the root reproduce L0 because U3 is a
   REPRESENTATION stage; unifying the conventions is a physics change that
   belongs in its own commit with its own gate. **It is still open.**

#### Two instrument traps this stage hit, both recorded in section 6

- **`?benchSkip=interp,avg` drives the dense field 100% non-finite**, and a
  word-equality comparison then reports `0/589824` differing and `maxAbs
  0.0e+0`, because identical NaN bit patterns ARE identical words. That ladder
  was written up as a result before the field was checked. The instrument now
  scores field health FIRST and refuses to report a row whose field is not
  finite. `?benchSkip=avg` alone is the valid isolation: level 1 is still
  interpolated and stepped, so the force reduction stays finite, and only the
  write-back is removed.
- **Chrome caches `https.py`'s header-less responses**, so a `Page.navigate`
  that only changes the query string can run the PREVIOUS build. The
  `DIRECT_GHOST` fix above appeared to do nothing for one whole measurement
  because of this. `Network.setCacheDisabled(true)` on the CDP session, always.
- **`debugStepSync(n)` advances in whole frames of `STEPS_PER_FRAME = 64`**
  (`for (k = 0; k < n; k += STEPS_PER_FRAME)`), so `debugStepSync(1)` and
  `debugStepSync(2)` both run 64 macro-steps and return the identical field.
  "After 2 macro-steps" anywhere in this file means 64. It also means the
  smallest advance available is 64, which is why the residual above is
  attributed structurally (who binds the buffer) rather than spatially: in 64
  steps a difference at a covered cell reaches 64 cells, and the domain is 256
  wide.

### U3 — the in-progress entry, for reference (the defect it was chasing was not there)


The wiring is in and inert: `amr_step1.wgsl` compiled a second time with
`GHOST = 0` and `NO_PARENT = 1`, dispatched on the root pool alongside the
dense L0 step and sharing its `useB` ping-pong. It reads and writes the root
pool only; the dense path stays authoritative.

**The default path is provably unaffected.** `amr_step1.wgsl` is the kernel
EVERY level uses, so the new overrides had to be inert at their defaults — and
the determinism table still reads `7ac54e170f903ac3` and `ce1bd4d8a3a1055c`
without `?rootpool=`. That was the change with real blast radius and it is
clean.

**Two findings, one fixed.**

1. **The half-cell offset (fixed).** `fineToCoarseUnit` places cell j at
   `origin - 0.5*dxL + dxL*j`, where `origin` is the centre of the first
   PARENT cell and the two children straddle it at ±dxL/2. The root's `origin`
   is its own first cell's centre — there is nothing to straddle — so the term
   is 0 there. Left in, the root's body SDF and sponge sat half a cell off the
   grid they are meant to reproduce. **No single-kernel test could have seen
   it**: the root is perfectly self-consistent with the offset, just displaced.
2. **L0 and the pool levels use DIFFERENT WINDOW CONVENTIONS (recorded, not
   fixed).** `amr_step.wgsl` converts with `bufferToWindowCell` — u32 modular
   arithmetic, so it TRUNCATES `off_x`/`off_y` to whole cells.
   `amr_step1.wgsl` uses `bufferToWindowPos`, which keeps the sub-cell part. So
   whenever the window offset is fractional — essentially always on the falling
   card — L0's sponge band sits up to half a cell from where every finer
   level's does. `?spongecellsnap=` is not a flag; the root pipeline sets
   `SPONGE_CELL_SNAP = 1` to reproduce L0, because U3 is a REPRESENTATION
   stage and unifying the conventions is a physics change that belongs in its
   own commit with its own gate.

**THE GATE'S JUSTIFICATION WAS WRONG, AND THAT IS THE MORE USEFUL FINDING.**
This plan said U3 should be scored by bit-identity because "the arithmetic per
cell is unchanged; only the address space moves". That is true of ONE kernel
run against a different buffer. It is false here: `amr_step.wgsl` and
`amr_step1.wgsl` are separately written kernels, and nothing entitles them to
associate their f32 identically. Bit-identity against the DENSE kernel was
never a sound bar. It becomes sound at U7, when the dense kernel is gone and
the comparison is one kernel against itself across builds.

**And comparing after many steps is the wrong instrument for a different
reason** — `plans/AMR-vs-dense-validation.md`'s Finding #3 already says it:
seeded or not, two discretisations separate, and the rate is not a pass/fail
quantity. A word-inequality count saturates (588073 of 589824) and says
nothing.

**The instrument that does work: ONE macro-step, scored by magnitude.** No
amplification has happened yet, so any difference is the kernels' own
arithmetic. `debugCheckRootMirror` now reports `maxAbs` and `relL2` alongside
the word count, and `?rootstep=0` is the live control -- same everything, root
simply not stepped.

    after 2 macro-steps        maxAbs     relL2
    rootstep=1  levels=2      9.59e-3   2.23e-3
    rootstep=1  levels=3      1.14e-2   2.41e-3
    rootstep=0  (control)     1.03e-4   1.06e-5

**Read the control row first.** Not stepping the root at all leaves it 1.06e-5
from the dense grid -- that is simply how much two macro-steps move the field.
Stepping it leaves it 2.2e-3 away, **two hundred times further**. The root step
is not slightly off, it is producing a materially different answer, and it is
worse than doing nothing.

**So there is a systematic defect in the root step, and it is not localised.**
Ruled out so far: `dxL` (`cellSizeL0AtLevel(0)` is 1), the diffuse band
(`kEps * dxL` = 1.5 exactly), the tile origin (`bx*RB*2*dxL` = `bx*2RB`, exact
in f32), the DIRECT_GHOST offsets at `GHOST = 0` (the neighbour hoist and the
re-expression both check out by hand at 0), and the sponge convention (snapping
it changed the magnitude by nothing). Not yet checked: the gather's clamp
fallback, `nbx/nby` in the root's `levelParams` against what the kernel expects,
and whether the root's `MAX_FINE_BLOCKS` dispatch covers exactly the live
slots.

**Next, and it should be a bisection rather than a fourth guess**: seed a
uniform field (`?init=uniform` exists on the TGV page, not this one), where the
exact answer is known and a single wrong neighbour shows up as a single wrong
cell rather than a diffuse 2e-3. That is the same move `validate-uniform-seam.js`
makes for the interface, for the same reason.

### U3 — the original statement, for reference

Compile `amr_step1.wgsl` a second time with `GHOST = 0`, `IDENTITY_SLOTS = 1`,
and root `levelParams`, and dispatch it for the root behind `?rootpool=1`.
`amr_step.wgsl` stays, unreferenced under the flag.

**Gate: BIT-IDENTITY, not tolerance.** The arithmetic per cell is unchanged;
only the address space moves, and each output cell is an independent gather, so
f32 results must match exactly. Score with `debugSnapshotSave` +
`tools/amr-diff.js`, two runs per build, on the analytic configs
(`channel-poiseuille`, `channel-couette`, `tgv-*`) and on
`index-cylinder-amr.html?levels=3`.

A DIFFERS here is a real defect, not an attractor: at this stage nothing about
slot assignment has changed, so the free-list race that makes `?levels=3`
multi-modal is not in play. **That is the reason to do the step before the
allocator** — it buys one stage where exact equality is unambiguous.

Then measure: `?bench=1` on the desktop, `?telemetry=1` on the phone, with and
without `IDENTITY_SLOTS`. Read `plans/perf-characterization.md` first; per-pass
timestamps are unusable on the PowerVR part and attribution has to be at frame
scale.

### U4-0 — DONE (2026-09-17). The root's VELOCITY is the dense velocity, bit for bit.

U4's INPUT GATE, and it is not a formality. The criterion differences `vel`,
the force reduction integrates over it, and the digest summarises it. Score any
of those on the root while its `vel` is unproven and a difference that belongs
to the STEP is reported against the consumer -- which is the shape U3 already
lost a day to, in the other direction.

`debugCheckRootVel`, 512 macro-steps, `?benchSkip=avg` (the U3 isolation: the
step kernel is then the only writer of the dense L0 buffers):

```
  levels 2 / 3 / 4, f16=1, f16=2, res=9, ghostcopy=1, dcpre=1
      all 0 differing words -- 0/131072, and 0/524288 at res=9
```

**AND THE TWO CONTROLS ARE NOT EQUALLY GOOD, which is worth saying rather than
reporting two dirty columns.** `?rootstep=0` leaves the root's `vel` UNWRITTEN
-- the root step is its only writer -- so that column is scored against zeros
and reads relL2 exactly 1.0. It proves the comparison reads live data, which is
its job, and nothing more. The SHIPPED-path row is the real stale control:
there `average` moves the dense `vel` and not the root's, and it reads
130941/131072 differing. A control that compares against a zero buffer and one
that compares against a genuinely diverged field are different claims.

One comparator now serves both questions (`compareRootToDense`), differing only
in the component count and in whether components are PLANE-MAJOR (`f`) or
INTERLEAVED (`vel`). A second copy of that loop is the shape CLAUDE.md keeps
recording.

### U4-1 — DONE (2026-09-17). The criterion on the root, ring-free, BIT-IDENTICAL.

`amr_criterion_pool.wgsl` now serves the root: the same module every level >= 1
uses, with `GHOST = 0` and the root's own block grid. It writes its OWN buffer,
never level 1's, so the page's refinement decisions still come entirely from
`amr_criterion.wgsl` and this stage moves nothing.

**The root->level-1 relation IS the pool parent->child relation, which is why
the two kernels are comparable at all.** A root tile is `2*RB = 16` cells and
level 1's block grid is `W/RB`, exactly twice the root's `W/(2*RB)` -- so one
root tile carries four level-1 children and each 8x8 quadrant is one workgroup
producing one child criterion, which is the same shape the dense kernel has,
where one L0 8x8 block produces one.

```
  512 macro-steps, ?benchSkip=avg      criterion differing / blocks
    levels=2                                  0 / 1024
    levels=3                                  0 / 1024
    res=9                                     0 / 4096
  CONTROL ?rootstep=0                      1024 / 1024
  CONTROL shipped path (avg on)            1024 / 1024
```

**THE OBSTACLE WAS THE STENCIL, AND IT IS THE SHAPE EVERY U4 CONSUMER HAS.**
The pool criterion differences +-1 and relies on the ring holding the
neighbour's data; the root has no ring. So a tap that leaves the tile resolves
against the OWNING tile instead -- the same move `amr_step1.wgsl`'s
DIRECT_GHOST already makes, and the same rule `amr2d.mjs`'s `resolveSource`
already states. U4-1a made that rule read the pool's own ring depth instead of
the module constant, which it had been ignoring since B0.

**It is DERIVED from `GHOST == 0`, not put behind its own flag.** At ring depth
0 there is no other correct behaviour, and a separate override is something a
ring-free pipeline can be given wrongly -- which U3 already paid for once, when
the root inherited `DIRECT_GHOST: 0` from `step1Constants` and was asked to
read a ghost cell nothing fills.

**SCORED AGAINST TWO GPU MUTANTS, because a clean comparison is also what a
blind one produces:**

```
  clamp inside the tile instead of resolving      891 / 1024 caught
  quadrant axes swapped (qx <-> qy)               512 / 1024 caught
```

The second is exactly half, which is what a diagonal-preserving swap must give
-- the off-diagonal quadrants move and the diagonal ones do not. **The first is
the more interesting number, because it is NOT 1024.** The criterion reduces by
MAX over a quadrant, so a wrong edge value is invisible wherever the maximum
happens to land on an interior cell, and 133 blocks agreed with a stencil that
was outright wrong. A max-reduction gate is sharp but not total, and the tool's
header says so.

The comparison also carries a VACUITY guard (`nonZero > 0`): two all-zero
criterion arrays agree perfectly and say nothing.

### U4-2 — DONE (2026-09-17). The force on the root, and the ONE prediction this plan got wrong.

`amr_force1.wgsl` now serves the root: `GHOST = 0`, `NO_PARENT = 1` (the same
half-cell straddle U3 removed from the step), the ring-free gather, and a (2,2)
dispatch over the 16-cell tile. It accumulates into a SCRATCH buffer, never
`forceBuf`, so the body integrator is untouched.

**FIRST, WHAT U4 SAID TO RETIRE IS ALREADY DEAD -- AND IS STILL LOAD-BEARING.**
`amr_force.wgsl` has not contributed to a shipped number in a long time: only
the FINEST level's force pass is dispatched (B4-3), `finestLevel === 0` means
`N_LEVELS === 1`, and every AMR page refuses `?levels<2` at module scope. The
branch cannot be reached. What keeps the kernel alive is
`main-cylinder-amr.js`'s `debugForceBreakdown`, which runs each level's pass in
isolation and is the INSTRUMENT that measured the coarser levels contributing
exactly zero before the masking was deleted. So it is a live instrument over a
dead code path, and the way to retire it is to make the root's own pass
reproduce it -- not to delete it unmeasured.

**THE PREDICTION WAS BIT-IDENTITY AND IT IS FALSIFIED.** U4-as-written argued
the partials would group identically because "the workgroup partition of the
root is unchanged". The partition IS unchanged -- the dense kernel is one
workgroup per 8x8 dense block, and the root at `GHOST = 0` dispatches (2,2)
over a 16-cell tile, the same four 8x8 regions of the domain. The partials are
still not identical:

```
  512 macro-steps, ?benchSkip=avg        |diff| in raw i32 units (FSCALE = 1e7)
    levels=2 / 3 / 4, res=8                    <= 1        (1024 workgroups)
    res=9                                      <= 5        (4096 workgroups)
    f16=1                                       exact
  CONTROL ?rootstep=0                     <= 1.6e7
  CONTROL shipped path (avg on)           <= 1.8e5
```

Deterministic: reproduced bit-for-bit across runs on both sides.

**SO U3's BIT-IDENTITY WAS THE SPECIAL CASE, NOT THE RULE.** The step's per-cell
expression is spelled identically in both kernels and only the address space
moves, which is why exact equality held there. The force's is not: the pool
kernel carries `-Fx * areaWeight` where `areaWeight` is exactly 1.0 at the
root, and the torque is a mul-sub the compiler may or may not contract. The
plan's earlier retraction of the bit-identity bar -- "nothing entitles two
separately written kernels to associate their f32 identically" -- was wrong
about the step and right about this.

**WHY A BOUND IS ACCEPTABLE HERE AND NOT LAZY.** It cannot be a mis-gathered or
mis-included CELL. Totals run ~5e5 raw units over a few hundred diffuse-band
cells, so one cell is worth ~1e3 units; a residual of 1 is a thousandth of a
single cell, and no cell can be wrong by that little. It also grows with
workgroup count (1 at 1024, 5 at 4096), which is what per-workgroup truncation
must do. And the defect scale is MEASURED rather than imagined: disabling the
ring-free gather -- clamping at the tile edge, as the ringed path does -- reads
`[-1512, 743, 6477]`. `FORCE_TOL = 64` therefore sits ~13x above the observed
residual and ~11x below the smallest component of a real defect.

**THE MECHANISM IS NOT ESTABLISHED, AND ONE EXPERIMENT WAS DISCARDED.** Testing
the `* areaWeight` suspicion by deleting that multiply was CONFOUNDED:
`amr_force1.wgsl` is the LIVE force pass at level 1, so the edit changed the
card's motion and the two runs were no longer comparable -- the dense total
itself moved from -486022 to -269736. Recorded so the same experiment is not
run again. Isolating it needs a diagnostic that leaves level 1 alone.

### U4-3/U4-4 — DONE (2026-09-17). The digest and the conserved totals, and THREE different answers to "can this be exact".

Both read the dense L0 buffers that U7 deletes, and both now read the root pool
instead. Neither needed a new kernel.

**U4-3, THE DIGEST.** It needs no shader change at all to serve the root: it
addresses a flat cell index and the root pool holds exactly `W*H` cells, so
only the bound buffer moves. What it needed was a comparison that MEANS
something. The shipped form samples 4096 cells by STORAGE INDEX, and the root
pool is a permutation of the dense grid, so sample `i` is a different physical
cell in each -- two honest digests of one field, legitimately unequal.

`FULL=1` (default 0, byte-identical) reduces over every cell instead, which
makes exactly one component comparable: `max` is invariant under BOTH
permutation and summation order, so `digest[2]` must match bit-for-bit. The two
sums are not required to -- adding the same 65536 floats in two orders need not
give the same f32 -- so they are reported and not gated. (In practice they came
back identical at `levels=2` and `res=9` and 6.2e-8 apart at `levels=3`; that
is luck, not a guarantee, and gating it would be gating luck.)

**U4-4, THE CONSERVED TOTALS, AND THIS ONE IS EXACTLY EQUAL.**
`readConservedTotals` was already parameterised by a `cellIndex` callback: it
walks `(x, y)` in SPATIAL order and asks where that cell lives, then sums in
f64 on the host. So pointing it at the root pool changes the addressing and
NOTHING ELSE -- same values (U3), same order, same f64 reduction. `mass`,
`momX`, `momY`, `rhoMin`, `rhoMax` and `maxU` all match exactly.
`amr2d.mjs`'s `rootCellIndex` is the addressing, scored against
`rootCellToDense` as a ROUND TRIP rather than side by side -- U2's lesson is
that two routes written together agree over a shared mistake, and a round trip
cannot.

**THE THREE ANSWERS, WHICH IS THE RESULT WORTH CARRYING PAST U4.**

```
  f, vel, criterion    EXACT      same per-cell arithmetic; only the address
                                  space moves
  conserved totals     EXACT      same values, same spatial order, f64 on the
                                  host -- only the addressing moves
  digest               max EXACT  sampled by storage index, so only a
                       sums not   permutation- and order-invariant reduction
                                  can be compared at all
  force                |d| <= 5   per-workgroup truncation; the pool kernel
                                  also carries a `* areaWeight` the dense one
                                  does not
```

U3 concluded that bit-identity was achievable and the plan's retraction of it
had been wrong. That was right for the step and wrong as a general rule. The
useful statement is narrower: **exactness survives a change of ADDRESS SPACE,
and does not survive a change of ARITHMETIC or of REDUCTION ORDER** -- and
which of those a consumer suffers is a property of the consumer, decided before
the measurement rather than discovered by it.

### U4 — criterion, force, digest, conserved totals

Point each at the root pool; retire `amr_criterion.wgsl`, `amr_force.wgsl`, the
dense half of `amr_digest.wgsl`, and `readConservedTotals`' dense path. Same
bit-identity bar as U3 — the force reduction still atomically adds one
truncated i32 per workgroup, but at this stage the *workgroup partition* of the
root is unchanged, so the partials group identically.

Check the 16-storage-buffer ceiling before designing each of these, not after.
`amr_manage_pool.wgsl` sits at 10 since B3-5, so there is slack, but the
failure mode is a `CreateBindGroupLayout` error at init — a page that does not
boot.

### U5-0 — DONE (2026-09-17). The quadrant offset was a THIRD site of one mistake.

The host rule U5 rests on, stated and mutation-checked before anything moves.

`quadrantOrigin(rb, q)` returned `GHOST + q*RB` -- where `GHOST` is the
**parent's** ring depth, read from the module constant. Every pool parent has
one, so it was right everywhere it had ever been used. Under U5 the parent is
the ROOT, which has none, and a quadrant-1 child would be placed at 10..18
inside a root tile whose cells stop at 15.

**THAT IS THE THIRD SITE OF THE SAME MISTAKE**, and naming the pattern is worth
more than the fix:

```
  U4-1a  resolveSource                 read the module constant, not the pool's
  U4-1   amr_criterion_pool.wgsl       assumed a ring in its STENCIL
  U4-2   amr_force1.wgsl               assumed a ring in its GATHER
  U5-0   quadrantOrigin                assumed a ring in the PARENT's frame
```

**A ring depth written as a constant is an assertion that every level has a
ring, and it is wrong exactly once -- at the root.** Anything that will serve
level 0 should be audited for a literal `GHOST` before it is pointed there,
rather than after.

Gated the same way U4-1a was, and for the same reason: the wrong answer is a
VALID-LOOKING INDEX, not a crash. `quadrantFitsParent` makes containment the
invariant, and the mutation -- the shipped constant-GHOST formula -- is caught
by two checks. Also gated: an L1 block is a quadrant of a root block at every
block, scored against the two block grids (each root block gets exactly four
distinct children, and every root block gets some), not asserted.

### U5-1 — DONE (2026-09-17), BIT-IDENTICAL on 8 rungs. The interp accessor was the FOURTH ring site, and the one that says so out loud.

Read this before starting it: the swap is **not** the override change it looks
like.

`common_interp_parent_pool.wgsl` is nearly ready to serve a root parent. Its
`sampleParent` is arithmetically IDENTICAL to the dense accessor's -- same
loop, same order, same `max(rho, 1e-6)` floor, same `fneq` -- so only the FETCH
addressing differs, and by U4's rule that means the swap should be
BIT-IDENTICAL. Three of the four things it needs are free:

  - `parentTau()` already reads `levelParams.parentTau`, and level 1's is
    already `tauAtLevel(0)`. No change.
  - the parent SLOT needs no buffer and no allocator change. The root is always
    full and its indirection is the identity, so the parent of L1 block
    `(bx, by)` is root block `(by>>1)*(nbx/2) + (bx>>1)` -- derivable from the
    child's own `slotToBlock` entry and `levelParams.nbx`.
  - the QUADRANT likewise: `(by&1)<<1 | (bx&1)`. No `quadrant` buffer needed.

**The fourth is the problem, and the file states it as the reason it is
simple.** Its header justifies having no wrap and no neighbour lookup like
this: a child's parent-local index lands in `[-GHOST, 2*RB-1+GHOST]` and maps
via a plain `+GHOST` offset onto the parent's already-valid `[0, FB)` range --
*"including the parent's ghost cells"*. **The root has none.** Index -1 maps to
-1.

And it is not an edge case. A child tile's ring is GHOST = 2 fine cells = ONE
parent cell deep, so a child in quadrant 0 needs parent cells from -1, and one
in quadrant 1 needs them through 2*RB. **Every L1 tile needs a root cell
outside its parent root tile, on two of its four sides.** Under a ringed parent
those land in the parent's ring, which interp filled. At the root they must
resolve into the NEIGHBOURING ROOT TILE.

**So this is the FOURTH site of the constant-GHOST mistake** (after
`resolveSource`, the criterion's stencil, the force's gather, and
`quadrantOrigin`'s parent frame -- five, counting U5-0). The resolution is the
same rule each time, `amr2d.mjs`'s `resolveSource`, and here it is cheaper than
anywhere else: because the root's indirection is the identity, resolving to a
neighbour tile is modular arithmetic on the root block grid and needs **no new
binding** -- the interp kernel's existing `blockSlot` at binding 5 is the
CHILD's and is not what this wants.

**The gate needs no second pool.** Run the dense-parent interp, copy level 1's
`f` to one scratch buffer, run the root-parent interp into the same pool, and
compare -- one buffer the size of L1's pool rather than a whole second
hierarchy.

**Predicted result: bit-identical**, on the reasoning above. If it is not, the
difference is in the FETCH and nowhere else, which is a narrow place to look.

#### What was built, and the prediction held

`PARENT_GHOST` is the whole of the change: one override on
`common_interp_parent_pool.wgsl`, default 2 so every existing pipeline is
byte-identical, 0 on the root-parent pipeline. Everything root-specific is
DERIVED from it rather than flagged separately -- at 0 the parent has no ring,
which means it has no parent, which means it is the root, which means it is
always full. That chain is what makes `parentSlotOf`/`quadrantOf` derivable and
what makes resolving an out-of-tile index against the owning tile the only
correct behaviour rather than one option among several. (`amr_criterion_pool`'s
GHOST override records the same reasoning; U3 already paid once for the other
shape.)

**One contract change, in the shared kernel.** `sampleParent` now takes
`(slot, bx, by, ix, iy)`. A root parent needs the child's block coordinates to
find the neighbouring root tile, and `parentOrigin` already took them --
handing them to only one of a PAIR that must agree on a frame is the split this
fragment exists to prevent. The dense half ignores them.

**Gate: `tools/validate-root-kernels.js` gained an `interp` column** rather
than getting a tool of its own. It is the one row that is not a kernel on the
root -- it asks whether level 1's ghost ring interpolated FROM the root
reproduces the ring interpolated from the dense grid -- but it is the same
differential protocol against the same reference and it wants the same rungs,
including `ghostcopy`, which is the rung that caught U3's one real defect.

**THREE LEGS, AND THE THIRD IS WHAT MAKES THE FIRST MEAN ANYTHING.** Both real
legs write SCRATCH buffers seeded from a byte-identical copy of the live pool,
so the page's own level-1 pool is never touched and the fine-fine consultation
branch -- which reads the target buffer's interior and is indifferent to the
parent -- resolves identically on each. That is the vacuity risk: if every ring
cell had an active same-level neighbour, the parent hop would never run and
`mismatched == 0` would say nothing. The third leg changes ONLY the parent (the
root's other ping-pong buffer), so whatever it moves is exactly what takes the
parent hop.

    rung (512 macro-steps, index-amr.html)     ring words   differ   parent hop
    levels=2                                       53136        0         9688
    levels=3                                       62208        0        10420
    levels=4                                       66096        0        10500
    levels=2 f16=1                                 29520        0         4893
    levels=2 f16=2                                 29520        0         5314
    levels=2 res=9                                 93312        0        13690
    levels=2 ghostcopy=1                           53136        0         9688
    levels=2 dcpre=1                               53136        0         9699

**18% of ring words take the parent hop; the rest are fine-fine copies.** That
fraction is the number to watch if this ever goes green for the wrong reason.

**THREE MUTANTS, MEASURED, all on the `levels=2` rung (53136 ring words):**

    clamp instead of resolving to the neighbour tile      6695 differ
    quadrant 0 for every child (drop the derivation)      5868 differ
    PARENT_GHOST left at 2 -- the named constant-GHOST     9720 differ
      mistake, i.e. exactly what U5-0 caught on the host

Note the first two are caught on ~60-70% of the hop words, not all of them:
wherever a bilinear tap happened to land inside the tile anyway, or wherever
quadrant 0 happened to be right, the wrong code returns the right number. **A
mutant that survives on a third of its own footprint is the normal case here**,
which is why the gate is exact equality and not a rate.

**THE SHIPPED PATH DID NOT MOVE, AND IT IS PROVEN BY EQUALITY, NOT BY Cd.**
`tools/measure-determinism.js` with `?detslots=1` returned `7ac54e170f903ac3`
(levels=2) and `ce1bd4d8a3a1055c` (levels=3) -- the SAME fingerprints 1.2c
recorded for the pre-U5-1 build, over 4096 steps of live refinement churn,
covering the field and the whole pool indirection. That is D0's promised
dividend arriving: an AMR change scored by exact equality across builds,
immune to GPU load and unforgeable. Do not reach for a Cd A/B when this is
available -- and note `amr-N2-diffuse` reads **Cd 1.631 / St 0.1466** on this
branch, not CLAUDE.md's 1.642, because D0 moved slot assignment. That is a
BRANCH baseline to re-record, not a regression, and the fingerprint is what
says so.

#### Would a dummy level -1 help? No, and the reason is worth keeping

The idea: give the root a parent so the root gets a ring, and the accessor's
`+GHOST` offset works unchanged at every level with no ring-free path at all.
It does not survive contact with what a ring is FOR.

1. **It does not terminate.** L(-1) would itself be parentless and need its own
   ring, filled by L(-2). Some level has to be ring-free; the question is only
   which, and the root is the one that costs nothing to make so.
2. **It would be WRONG, not just wasteful.** A ring holds an interpolation of
   the parent -- a lossy, coarser reconstruction. The root's neighbour cells
   are not coarser: they are the same resolution, exact, and already in memory
   one tile over. Filling a root ring from an L(-1) would substitute a
   coarse-interpolated approximation for exact data that is sitting right
   there, at every root tile boundary, i.e. everywhere. It also forfeits
   bit-identity, which is this stage's gate.
3. **It costs 56% of the root's memory.** The root is the whole domain and
   always full, so ringing it takes its tiles from 16x16 to 20x20 -- and
   `ghostDepthAtLevel(0) == 0` is exactly what keeps root memory at today's
   `W*H` (U1's `cells === dims.W * dims.H` identity). Plus an L(-1) pool, its
   step, and a fill pass.
4. **The ring-free path is three lines.** Modular arithmetic on the root block
   grid, no new binding, no new buffer, bit-identical.

The useful way to say it: **the root's ring is a VIEW, not storage.** It is
always full, so every out-of-tile index has an owner, and `resolveSource` names
it. That is the same sentence U3 used to justify pinning `DIRECT_GHOST` on the
root step and U4-1 used for the criterion's stencil -- the fourth application
of one rule, not a fourth special case.

### U5-2 — DONE (2026-09-17), BIT-IDENTICAL on 8 rungs. The restriction, and why it was the easy half.

The reverse hop of U5-1, built the same way: `PARENT_GHOST` on
`common_avg_parent_pool.wgsl`, default 2 (every ringed pipeline byte-identical)
and 0 on the root-parent pipeline, with `parentSlot`/`quadrant` derived from
the child's own blockID rather than read.

**STRUCTURALLY EASIER THAN THE FORWARD DIRECTION, and the asymmetry is the
result worth carrying.** Restriction writes ONE parent cell per child cell and
the destination is always inside the parent's own interior -- there is no
stencil, so nothing ever reaches past the parent tile. U5-1 had to resolve an
out-of-tile index against the neighbouring ROOT TILE; this direction has no
out-of-tile index to resolve at all. **A ring-free parent costs prolongation a
neighbour lookup and costs restriction nothing but an offset and a stride.**
That is a statement about which direction the ring is FOR, and it is the same
statement B6's mailbox makes: the ring is an inbox inward and an accumulator
outward, and only the inward half needs to reach.

    rung (512 macro-steps, index-amr.html)     written words   differ
    levels=2                                          23616        0
    levels=3                                          27648        0
    levels=4                                          29376        0
    levels=2 f16=1                                    13120        0
    levels=2 f16=2                                    13120        0
    levels=2 res=9                                    41472        0
    levels=2 ghostcopy=1                              23616        0
    levels=2 dcpre=1                                  23616        0

Scored over the cells the restriction actually WRITES -- the L0 footprint of
the ACTIVE level-1 blocks, exactly `RB*RB` per active slot. Two mutants, on the
`levels=2` rung: quadrant 0 for every child **20544/23616**, `PARENT_GHOST`
left at 2 **23616/23616**.

**AND `?rootstep=0` CANNOT DISCRIMINATE THIS COLUMN, WHICH IS A FACT ABOUT
RESTRICTION AND NOT A HOLE IN THE CONTROL.** The tool's control rung makes the
root pool stale, which moves every row that READS it. The average reads only
the CHILD and writes only the parent -- the parent's prior contents never enter
the arithmetic -- so a stale parent produces a bit-identical result. That
column reads clean on the control while the other five go dirty, and it is
excluded from the control's assertion with the reason written down. Its own
liveness control lives in the page hook and changes the child instead. **A
control that provably cannot discriminate a row should say so rather than be
quietly weakened until it appears to** -- the same discipline as the
starved-pool sweep's `field`/`quadrants` abstention.

**THE PLAN WAS WRONG ABOUT WHAT L1 NEEDS, and U5-1 is why.** The umbrella
below says "L1 gains `parentSlot`/`quadrant`". It does not: the root is always
full, so its slot IS its block index, and both fields fall out of the child's
own block coordinates in three lines of arithmetic. **Two buffers per level and
two allocator writes were budgeted for and are not needed** -- the same shape
as B2-2b0's `childQuadrant` and B3-5's origin buffers, found a third time. The
bindings stay declared (WGSL has no conditional bindings, and one entry file
serves both pipelines) but are bound to a sentinel on the root pipeline.

### U5-3 — DONE (2026-09-17). The coupling goes live, and the whole simulation is BIT-IDENTICAL.

U5-1 and U5-2 built both hops as inert twins. U5-3 wires them into the solver:
under `?rootpool=1`, level 1's ghost ring is interpolated FROM the root pool
and level 1's restriction is written TO the root pool.

**THE DENSE GRID KEEPS ITS OWN STEP AND ITS OWN RESTRICTION, and that staging
device is most of the stage.** Both L0 representations are stepped (U3) and
both now receive the restriction, so they stay byte-identical *indefinitely*
rather than only until the first `average`. Every remaining dense consumer --
the renderer, the criterion, the force, the digest, `debugSnapshotSave`, the
whole host and tool tail -- therefore needs no change at this stage and cannot
be broken by it. They get flipped one at a time afterwards, each against a
buffer already proven equal, and the dense writers go at U7. The cost is one
duplicated L0 step and one duplicated restriction per macro-step, on an opt-in
flag.

Exactly one thing is REPLACED rather than duplicated: the dense-parent interp.
Both would write the same ghost cells of the same pool, and U5-1's result is
that they write the same words, so a race would buy nothing.

`?rootcouple=0` turns the coupling off while keeping the root pool -- U5-2's
configuration exactly, in the same build, which is what makes the coupling
A/B-able rather than requiring two checkouts.

**Gate 1: `validate-root-kernels.js`'s shipped-path rows, promoted from
REPORTED to GATED and green.**

    rung (512 macro-steps)          f differing / 589824      the control
    levels=2 rootpool=1                    0                  580531  (rootcouple=0)
    levels=3 rootpool=1                    0                  580570  (rootcouple=0)
    levels=4 rootpool=1                    0                  --

`vel`, `crit`, `dig` and `cons` are exact on those rows too and the force is
exact or within 2. The control is the same page with `?rootcouple=0`: the root
is stepped and correct but receives no restriction, and 98.4% of it goes dirty.
That is the discrimination the REPORTED section was waiting for, and the plan's
own note -- "the root pool receives no restriction until U4/U5" -- is now spent.

**Gate 2, and it is the stronger one: `?rootpool=1` produces the SAME
FINGERPRINT as `?rootpool=0`.**

    levels=2 detslots=1    7ac54e170f903ac3    rootpool=0 AND rootpool=1
    levels=3 detslots=1    ce1bd4d8a3a1055c    rootpool=0 AND rootpool=1

Two different coupling implementations, 4096 steps of live refinement churn,
one fingerprint covering the field AND the whole pool indirection. This is the
second stage in a row where D0's determinism has paid for itself.

#### The fingerprint caught a real bug that nothing else did

The first run of gate 2 came back **DIFFERS on both level counts with
`?detslots=1`** -- the configuration D0 proved bit-reproducible -- and the tile
counts moved run to run (49 vs 54 at `levels=2`), so refinement itself was
being driven by something unstable.

**The root pool was never seeded.** U1 deliberately gave it no initial field:
*"a buffer nothing reads should not be given a state that could be mistaken for
one."* That was right for three stages and became a defect the instant level 1
started reading it -- every page load and every `reset()` left the solver
interpolating from an unwritten buffer.

**And no existing gate could have found it**, which is the part worth keeping.
Every root-pool checker to date calls `debugMirrorRoot()` by hand before it
looks, so the buffer was always seeded *by the instrument* and never by the
page. The 512-step word diff was green, the invariant sweep was green, the boot
smoke was green, and Cd would have been within its own noise. Only a gate that
asks "does this build reproduce itself exactly" could see it.

The fix seeds from the mirror -- U2's validated dense->root map -- at init, at
`reset()` and after `debugSnapshotLoad`. At U7 the root is the only L0 and takes
`initF()` directly.

#### And one instrument trap, measured

`debugCheckRootInterp`/`debugCheckRootAverage` originally gated on
`wrote > 0`: ring words the dense leg changed relative to the seed, as a
liveness guard. **It inverts the moment the pass it re-runs is already LIVE in
the macro-step.** Re-running an idempotent pass on a buffer that already holds
its output legitimately changes nothing, so `wrote == 0` means "the page is
doing this correctly", not "the pass did nothing" -- and it took out the new
gate and its control in the same sweep.

`staleDiff` is the guard that survives, because it changes an INPUT rather than
looking for movement: if the root leg wrote nothing at all, its stale twin
would match the dense leg and it would read zero. **A liveness control has to
perturb something, not merely observe something.**

### U5-4 — DONE (2026-09-17). One manager, every level. The stage that moves numbers, and it moved more than the plan said.

`amr_manage_pool.wgsl` now decides and allocates LEVEL 1, with the root as the
parent level; `amr_manage.wgsl` is not dispatched at all under `?rootpool=1`.
Level 1 allocates in QUADS, its want set is closed into quads by
`amr_cascade.wgsl`, and its reset is the same quad reset every deeper level
already had. `?rootmanage=0` keeps the dense manager in the same build.

Almost all of the shader fitted the root unchanged, which is the thesis
holding: `myLevel()` reads 0, `parentOriginL0` is `block * 16`,
`parentHalfExtentL0` is 8, and the 2:1 neighbour-active gate is trivially
satisfied because the root is always full. The pieces that moved are host-side
-- a `quadAlloc` option on `allocLevelPool`, a `quadCompleteFrom` option on the
cascade, three loops that now start at parent level 0, and level 1 joining
`quadCPU`.

**THE PLAN UNDERSTATED THE MOVE, AND IT IS WORTH SAYING WHY.** It predicted
"Cd moves in the 4th digit", attributing the change to slot regrouping alone.
But the pool manager also DECIDES over the parent's footprint and allocates in
quads, so level 1's refinement granularity goes from one 8-cell block to a
16-cell quad. Measured, `index-amr.html`, 4096 steps:

    active tiles            dense manager      pool manager
    levels=2                      75                 96      +28%
    levels=3               [103, 240]         [136, 240]     +32% at L1

That is a differently-shaped refined region, not a 4th-digit perturbation. It
is strictly MORE refinement, so it costs memory and time rather than accuracy
-- but `POOL_PEAKS` was measured at block granularity and its 40k-step
re-measurement is OUTSTANDING. The refusal watch is what would report a wrong
guess, which is why the next paragraph is part of this stage and not a
follow-up.

**IT TOOK A GATE VACUOUS AND THE STAGE PUT IT BACK.** Level 1's
pool-exhaustion counter lived in `amr_manage.wgsl`, so moving level 1 off it
silently unhooked the starvation gate. Measured side by side on
`amr-dev-invariants --extra=maxFineBlocks=16&diag=1`:

    dense manager   pool STARVED (2140 refine(s) refused)
    pool manager    pool OK                                  <- vacuous
    after the fix   pool STARVED (1028 refine(s) refused)

`amr_manage_pool.wgsl` gained binding 9 -- the same `diag` buffer at the same
slot as the dense manager's -- and the binding was mirrored into **all five**
AMR pages' `managePoolBGL` and bind groups, which is CLAUDE.md's own recorded
trap walked deliberately with the boot smoke run afterwards. The starved sweep
is back to five-of-seven red with `field` and `quadrants` abstaining, which is
the discrimination that says the gates read seven different things.

**Gates.**

1. `validate-root-kernels.js`: all 11 gated rungs still bit-identical, all 4
   controls dirty. The `crit` column needed its roles SWAPPED to stay
   meaningful -- the pool criterion at parent level 0 is now the live writer of
   level 1's `blockCriterion`, so the DENSE kernel is the one on a scratch
   buffer. Left alone, that column would have compared a kernel against its own
   output, which is a comparison that cannot fail.
2. Invariants: seven-of-seven green at 8192 steps; five-of-seven red under
   `--extra=maxFineBlocks=16`, as above.
3. Reproducibility: `?detslots=1` IDENTICAL at both level counts. New
   fingerprints, and they SHOULD be new:

        levels=2   f71bce9d33945265     (was 7ac54e170f903ac3)
        levels=3   71560c03a3d34c21     (was ce1bd4d8a3a1055c)

4. **The default did not move at all**: `?rootpool=0` still returns
   `7ac54e170f903ac3` / `ce1bd4d8a3a1055c`, so the five-page binding change and
   everything else in this stage is byte-identical on the shipped path.

**AND THE levels=2 BASELINE RUNG STOPS DISCRIMINATING UNDER THIS MANAGER.**
4 of 4 runs at `?rootpool=1&levels=2` with `detslots=0` came back not just
identical to each other but identical to the `detslots=1` hash: the racing free
list produces the same assignment every time there. So
`measure-determinism.js --extra=rootpool=1` exits nonzero on that row, and the
gate is deliberately NOT relaxed for it -- the row is doing its job by saying
it can no longer tell "the flag worked" from "nothing raced". `levels=3
detslots=0` still DIFFERS and is the rung that keeps the instrument honest.
Plausibly the pool manager's dispatch (1024 parent slots, a few dozen sparse
candidates) simply does not race on this hardware where the dense manager's
4096 blocks did; that is a scheduling accident, not a guarantee, and it should
not be relied on.

#### What U5 did NOT do, and what that costs

**`amr_manage.wgsl` cannot retire here.** Four other AMR pages --
`main-cylinder-amr.js`, `main-tgv-amr.js`, `main-channel-amr.js`,
`main-reentry-amr.js` -- have no root pool at all, so they still dispatch it.
U1 through U5 only ever touched `main-amr.js`. Deleting the dense manager
therefore needs the root pool ported to the other four pages, which is U7's
"host and tool tail" work rather than this stage's.

**So the Cd/St consequence of U5-4 IS NOT MEASURED, and cannot be yet.** The
cylinder harness is `index-cylinder-amr.html`, which has no root pool, so
`amr-N2-diffuse`/`amr-N3-diffuse` run the dense manager whatever this page
does. The plan's "take a same-build repeat on both sides" protocol has nothing
to compare. **That is the prerequisite for making `?rootpool=1` the default**,
and it is a bigger item than it looks: the cylinder page is the only harness
with literature values attached.

### U5 — L1 becomes a quad child of the root

The load-bearing stage. `amr_manage.wgsl` retires; `amr_manage_pool.wgsl`
serves every level >= 1; L1 gains `parentSlot`/`quadrant` and quad-stride
allocation; `amr_interp_dense_parent.wgsl`, `amr_average_f2c.wgsl` and both
dense accessors retire.

**This moves published numbers, and it should.** Quad allocation changes which
slot a block gets, which regroups the force reduction's truncated partials, so
Cd moves in the 4th digit on AMR configs. CLAUDE.md's own reproducibility note
is the calibration: `amr-N2-diffuse` is reproducible to about +-0.001 and
`amr-N3-diffuse` to about +-0.002, and **the repeat has to be taken on both
sides**, on the config actually being compared.

**Gate, three parts, all required:**
1. The analytic AMR configs hold **zero active tiles** at every level
   (measured, `debugListActiveBlocks`) — so they must be **bit-identical**
   across this stage. They gate the solver, not the seam, and that is exactly
   what is wanted here.
2. The full 9-gate invariant sweep, plus the starved-pool sanity check
   (`--extra=maxFineBlocks=16`): the discrimination that says the gates read
   nine different things is that five go red and two abstain, not that
   everything reddens together.
3. `?levels=3` lands in a **known attractor**, bit-for-bit, in two runs. The
   strong form — matching the baseline in two *different* attractors — costs
   one extra run and is worth it here.

Do **not** take a Cd delta as evidence at this stage without a same-build
repeat on both sides. And verify which tree the dev server is serving before
believing any of it (`ss -lptn`, then `readlink /proc/<pid>/cwd`, then pin
`--baseUrl=` and `--port=`).

### U6 — the renderer walks levels

`amr_render.wgsl` binds exactly three velocity sources today — dense `vel`,
`vel_pool` (L1), `vel_pool2` (L2) — and `main-amr.js:1584`'s `renBG` wires them
to `velBuf`, `pools[1]` and `pools[2]` **by number, not by depth**. There is no
`vel_pool3`, so **a level-3 tile's velocity has no path into the picture at
all**: at `?levels=4` it is refined, solved, stepped twice per level-2 substep,
force-reduced and invariant-checked, and then drawn as its level-2 parent.
Replace the fixed set with a per-level walk.

**This is a bug and it already has a gate that fails on it.**
`tools/validate-render-levels.js` pauses the sim, overwrites one level's
velocity pool with a value no physical flow produces, redraws via
`debugRenderOnce`, and asks whether the picture moved. Levels 1 and 2 go green;
level 3 goes red. Two page hooks support it (`debugRenderOnce`,
`debugPerturbLevelVel`) and `frame()` now shares one `encodeSceneRender`
encoder path with them, so the gate cannot end up scoring a second copy of the
render.

It is deliberately a *capability* test, not a static one. Counting `vel_pool*`
bindings in the WGSL would keep passing the day someone binds a fourth buffer
and forgets to sample it. It is also deliberately kept out of
`validate-all.js`'s default sweep until this stage closes it — a known-red gate
does not get to turn the whole sweep red — but **it goes into the sweep in the
same commit that fixes the renderer**, not later.

Its own trustworthiness is scored before its result is: two unchanged renders
must be byte-identical or the run aborts, the snapshot restore must return to
the baseline frame, and a level with no active tiles **abstains** rather than
passing, because "nothing to draw" and "drawn nowhere" are the same screenshot.

Note what this makes the renderer: at a true coarse/fine seam it is the *only*
live consumer of what the ring publishes, because no physics gate reads a ring
cell. A change to what explode writes into the ring reaches the picture and
nothing else. Score it with screenshots, deliberately, rather than assuming a
green sweep covers it.

### U6 — DONE (2026-09-18). The renderer walks levels, and the defect was bigger than the one it was written for.

`amr_render.wgsl` binds one velocity/indirection pair per POOL level (up to
`MAX_RENDER_POOL_LEVELS` = 4) and walks them finest-first in a loop.
`N_POOL_LEVELS` replaces `HAS_LEVEL2`, and a configuration deeper than the
shader can draw is now REFUSED at init rather than rendered without its finest
level.

**WGSL cannot index an array of storage buffers**, so the buffers stay one
binding each and `poolVel`/`poolSlotOf` are a four-way `if` ladder. That ladder
is irreducible. What is NOT, and is what actually carried the bug, is the
per-level block arithmetic, the bilinear sample, the finest-wins precedence and
the outline colour -- all copy-pasted per tier before, all written once now.

#### It found a second, larger instance the moment it was pointed at another page

The gate was written for one defect: level 3 invisible at `?levels=4` on
`index-amr.html`. Running it against the cylinder page found another.

**Three of the five AMR pages -- cylinder, TGV and channel -- never passed a
level override to the render fragment at all.** They pass `fineConstants`,
which has no `HAS_LEVEL2` field, so the shader's declared default of `0u` stood
and the level-2 branch was statically dead. Measured on
`index-cylinder-amr.html?levels=3` before the fix: perturbing level 2's whole
velocity pool (65600 cells, u=(9,9)) left the picture **byte-identical**, while
level 1 moved.

That is the page the Cd/St numbers come from. Every screenshot anyone has taken
of the cylinder at `?levels=3` showed level-1 resolution.

**The gate could only ever have found it by being pointed there**, which needed
two page hooks the cylinder page did not have. Those are ~25 lines and they are
now in it. The lesson is not about the renderer: a capability gate that exists
on one page measures one page.

#### And the first draft of the fix had a visible artifact, caught by eye

The walk generalises the tile lookup, and the half-cell shift in it does not
generalise the way it looks.

A tile is picked by which CELL OF THE LEVEL ABOVE contains the point, so the
shift is **half a level-(m-1) cell = 2^-m in L0 units**, i.e. `1/dens`. Written
as a flat `0.5` -- which is what level 1's original line says, because at level
1 those are the same number -- it is half an L0 cell, which is `2^(m-1)` FINE
cells:

```
  level   shift as written (0.5 L0)   in fine cells   ring is 2 deep
    1                                      0.5        ok
    2                                      1.5        ok
    3                                      3.5        OUTSIDE THE RING
    4                                      7.5        OUTSIDE THE RING
```

Past level 2 the bilinear tap lands outside the ring entirely, `poolVel`'s clamp
pins it to the tile edge, and a band of pixels along every level-3 tile boundary
reads one frozen value. On screen: **a dark lattice over the refined region**,
which is what it looked like and how it was found -- the user watching the debug
Chrome during the test runs, not any gate.

**NO GATE HERE WOULD HAVE CAUGHT IT, AND THAT IS THE POINT WORTH KEEPING.** The
reachability gate asks whether a level changes the picture AT ALL; a level whose
every seam is wrong still changes it. The paused baseline was byte-reproducible
(6 of 6 renders identical). The field was finite. Live frames showed no
alternation (12 distinct hashes, no 2-cycle). Every instrument was green over a
visibly broken picture. **This stage's own note already said to score the
renderer with screenshots deliberately rather than assume a green sweep covers
it, and that note earned itself inside one stage.**

With `shift = 1/dens` the ring reach is exactly 0.5 fine cells at EVERY level --
the reach level 1 has always had.

#### Gates

- **Level 1 is BIT-IDENTICAL to the pre-U6 renderer.** `?levels=2&detslots=1`,
  paused after 4096 steps, screenshot hash `04635a0ad09a721b` on both builds.
  That is the check that says the walk reproduces the old level-1 path rather
  than merely resembling it, and it is available only because D0 made the
  state reproducible.
- **`tools/validate-render-levels.js` green on both pages**: card at
  `?levels=4` (levels 1, 2, 3 all PASS), cylinder at `?levels=3` (1, 2 PASS).
- **In the default sweep**, as this stage's own note required, as
  `render-levels-card` and `render-levels-cylinder`. The check moved to
  `tools/lib/render-levels.js` so the leaf tool and the sweep call ONE
  implementation.
- **The solver did not move**: `measure-determinism.js` returns
  `7ac54e170f903ac3` / `ce1bd4d8a3a1055c`, unchanged. The renderer is
  render-only and this proves it rather than asserting it.

Level 2's assignment DID change (`?levels=3` hashes differ): the old code split
an L1 tile at `halfRB` with no shift at all, so the seam fell between cell
centres and the two halves each sampled their own interior with no shared ring
tap. The new rule is the same one level 1 uses, one octave down.

#### What the gate can and cannot say now that it is green

The pass/fail split used to supply its own discrimination -- levels 1 and 2
green while level 3 went red. With everything green that is gone, so the two
guards that remain are restated in the tool rather than left to be inferred:
the baseline must be reproducible (or the run aborts), and the restore must
return to baseline before every level (or that level abstains). Without the
first, "the picture changed" means nothing; without the second, each level is
compared against the previous level's perturbation and passes trivially.

### U7 — one implementation, five pages, then delete the dense path

**U7 AS ORIGINALLY WRITTEN IS ONE TASK WITH A TEN-ITEM LIST AND NO GATE UNTIL
THE END, AND IT IS TOO BIG.** U5 found out why: every stage from U1 to U5-4
landed on `main-amr.js` alone, so the root pool exists on the dev page and
nowhere else. `amr_manage.wgsl` cannot retire while four other pages dispatch
it, and U5-4's own Cd/St consequence cannot be measured at all, because the
cylinder harness has no root pool to measure. The missing work is not
*deletion*, it is **propagation** -- and propagation done by copying would put
five copies of the root pool in the tree, which is the exact shape this project
keeps paying for.

So U7 splits into seven sub-stages. The first six SHARE the implementation, and
each one is byte-identical on the default path and provable by fingerprint. The
seventh is the deletion, which becomes small once there is one copy of
everything to delete.

#### The measurement that shapes the split (2026-09-18)

```
  bind group layouts     14 kinds across 5 pages, and every one of them is
                         BYTE-IDENTICAL (compared with comments and whitespace
                         stripped). 658 lines of them in total.
  S_Advance              byte-identical across cylinder / reentry / tgv /
                         channel. main-amr.js's differs ONLY by the root pool,
                         and differed before this session too.
  dispatchMacroStep      three variants: amr / (cylinder, reentry) /
                         (tgv, channel). The last pair is shorter for one
                         reason -- no body.
  resetSim, snapshots    genuinely per-page: four and three variants. These
                         stay per-page.
  the root-pool work     ~1290 added lines in main-amr.js across U1..U5, of
                         which the COMPARATORS (debugCheckRoot*, the mirror
                         checker, compareRootToDense) are roughly 500 and
                         belong on the dev page, not on the other four.
```

**That last line is the whole argument for the order.** Everything above the
line is measured; this next figure is an ESTIMATE and is marked as one. The
part of U1-U5 the other pages actually need -- allocate the pool, seed it, step
it, couple level 1 to it, manage level 1 from it -- looks like roughly 270
lines of solver once the comparators, the flag headers and the inert twins are
set aside. Copied into four pages that is ~1100 lines of new duplication.
Shared first it is close to zero, because the four pages already run a
byte-identical scheduler. The estimate does not have to be right for the
ordering to be: the measured half (14 identical layouts, one identical
`S_Advance`) already says the sharing is free and the copying is not.

#### U7-0 — DONE (2026-09-18). The bind group layouts are one function.

`makeAMRLayouts(device)` in `amr2d-gpu.mjs` returns the fourteen layouts; five
pages call it and deleted their own. **865 lines removed, 278 added, a net
−590.** Every page destructures only what it uses, so the two bodyless pages
name eleven and the three with a body name fourteen -- the function builds all
fourteen regardless, because a layout nobody binds costs nothing and selecting
a subset there would reintroduce exactly the per-page variation this removes.

**THIS IS THE RUNG THAT PERMANENTLY KILLS THE BINDING-MIRROR TRAP**, and that
is worth more than the line count. CLAUDE.md records it producing 238e48c; U4-1
and U4-2 walked into it again in the other direction and stopped four pages
booting; U5-4 walked it a third time deliberately, with the boot smoke as the
net. After this rung a binding added to a shared shader has exactly one layout
to reach.

**The one thing to watch:** the layouts are identical TODAY. If the shared
function grows per-page parameters it becomes five copies with extra steps. If
a page ever genuinely needs a different layout, that is a signal the SHADER
should not be shared either -- take that conversation rather than adding a flag.

*Gate, all four green:* `make check`; boot smoke on all seven configs plus both
render-reachability configs; `measure-determinism.js` returned
`7ac54e170f903ac3` / `ce1bd4d8a3a1055c` on the default, unmoved;
`amr-N2-diffuse` read **Cd 1.631 / St 0.1466**, unmoved to four digits.

**One instrument caveat, measured while running it.** The `levels=2 detslots=0`
baseline rung came back IDENTICAL over TWO runs and then gave three distinct
hashes over FOUR runs of the same build. The race lands the same way twice
often enough that a 2-run baseline is not evidence it has stopped racing --
which matters because U5-4 recorded a case where four runs DID all agree, and
the two look identical at `--runs=2`. The tool's header now says to read that
rung at `--runs=4`.

#### U7-1 — DONE (2026-09-18). The coupling pipelines are one function.

`makeCouplingPipelines(device, layouts, modules, {W, H, RB, F16, DC_PRE, manage})`
builds the twelve every AMR page built identically: the six interp variants
(dense parent and pool parent, each steady-state / init / fine-fine-only), the
two averages, the criterion, and manage's three entry points. **377 lines
removed, 178 added, a net −199.**

Measured before touching anything, comments and whitespace stripped: eleven of
the twelve were byte-identical across all five pages, and the twelfth (`avgPL`)
differed only in the NAME of its constants object -- `fineConstants` on two
pages, `avgConstants` on three, with the same five fields in both.

**IT TAKES FIVE SCALARS, NOT TEN PREBUILT BUNDLES, AND THAT IS THE POINT.**
`GHOST_ONLY` and `FINE_FINE_ONLY` name MODES OF THE KERNEL -- steady-state ghost
refresh, one-time full-slot fill on activation, the between-substep fine-fine
re-exchange. Five pages were each spelling that triple out from `W, H, RB, F16,
DC_PRE`, so five pages could each get it wrong. The modes now belong to the
shared code and the scenario stays with the page.

**And it returns the bundles it derived.** A page needing a VARIANT -- the
`?benchSkip=` no-op twins, the `SKIP_GHOST` ring twin, U5's root-parent twins --
now builds it from `couplingConstants.*` rather than from its own second copy of
the same literal. `main-amr.js` had exactly that: its no-op measurement twins
were spread from separately-declared objects, so a twin could drift from the
thing it measures. Eight now-dead bundle declarations went with the change.

#### What stays per page, and the line is the scenario

The step, the force, the physics integrator and the render fragment stay per
page, because their override sets are genuinely different things -- measured,
`step1Constants` differs on every single page:

```
  amr        W, H, RB, SDF_FAR, F16, DIRECT_GHOST
  cylinder   + SPONGE_UX/UY, USE_BOUNCEBACK, SOLID_EQ
  reentry    W, H, RB, F16, DIRECT_GHOST
  tgv        ...stepConstants, RB, DIRECT_GHOST
  channel    (built inline, no named bundle)
```

Scenario overrides are data; which pipelines exist, and which override selects
which mode of a kernel, is not. `manageConstants` is the one bundle the shared
function takes whole, because thresholds and geometry are scenario.

D0's deterministic-handout pipelines (`manageScan*`, `manageLink*`) and the
`?benchSkip=` twins stay on `main-amr.js` -- they exist on one page and will
move when they reach the others.

*Gate, all five green:* `make check`; boot smoke on seven configs plus both
render-reachability configs; `measure-determinism.js` unmoved at
`7ac54e170f903ac3` / `ce1bd4d8a3a1055c`; **the four analytic AMR configs
(`channel-poiseuille-amr-N2`, `channel-couette-amr-N2`, `tgv-amr-N2`,
`tgv-amr-N3`) all PASS** -- they hold zero active tiles, so they gate the solver
rather than the seam, which is exactly what a pipeline refactor wants;
`amr-N2-diffuse` unmoved at Cd 1.631 / St 0.1466.

#### U7-2 — DONE (2026-09-18). The per-level bind groups are two functions.

**621 lines removed, 216 added, a net −405.** Three things moved:

- `makeLevelBindGroups` — every level >= 2's interp / step / average / force
  bind groups. Byte-identical across the card, cylinder and reentry pages;
  TGV and channel had the same loop minus the force block, a strict subset.
  `forceBuf` is the CONDITION for that block rather than a flag: a page with no
  body has no force accumulator, so there is nothing to decide.
- `makeManageBindGroups` — one criterion/manage pair per parent level.
  Byte-identical across four pages; `main-amr.js` differed only by where the
  loop starts since U5-4 made the root a parent level. `firstParentLevel` is
  that and nothing else, and it disappears at U7-5.
- `makeRenderBindGroup` — already moved at U6, listed here because it belongs
  to this rung.

**A dead line went with the move.** Four of the five copies still computed
`grandchildPool` for a grandchild cascade B2-2d deleted -- legal, free at
runtime, and reading as though the loop still weighed it.
`amr_manage_pool.wgsl`'s own header records the identical lesson about
`refineWants`: deleting a mechanism has to include deleting what fed it.

*Gate, all five green:* `make check`; boot smoke on seven configs plus both
render configs; `measure-determinism.js` unmoved at `7ac54e170f903ac3` /
`ce1bd4d8a3a1055c`; the analytic AMR configs PASS; `amr-N2-diffuse` unmoved at
Cd 1.631 / St 0.1466 with its invariants green.

#### AND THE POOL-ALLOCATION SCAFFOLDING DOES NOT BELONG HERE — a correction

This rung was scoped to "take the pool-allocation and reset scaffolding with
it", on the argument that more than half of U7-4's per-page residue is a loop
bound in one of those loops. **Measured before doing it, that advice did not
hold**, and the difference is the whole basis of this ladder:

```
  the level>=2 bind-group loop   2 variants, one a strict SUBSET of the other
  the manage bind-group loop     2 variants, differing by a loop bound
  the pool-ALLOCATION loop       3 variants
  the levelParams loop           4 variants across 5 pages
```

Everything shared in U7-0, U7-1 and U7-2 had ONE premise: it was already
byte-identical, and the change made that structural. The allocation and
levelParams loops are not — they differ in genuine per-scenario policy (see
below). Sharing them would be a parameterising refactor without that premise,
which is exactly how a shared function acquires per-page options and becomes
five copies with extra steps -- the hazard `makeAMRLayouts`'s own header names.
**So U7-4 keeps its handful of loop-bound edits, and that is the cheaper
trade.**

**One finding while measuring it, not acted on.** The pages differ in pool
SIZING policy: the card and cylinder pages use `poolSlotsFor(POOL_PEAKS, m,
N_LEVELS)`, measured per level; **the reentry, TGV and channel pages still use
a flat `128` for every level >= 2.** That flat-per-level default is the exact
shape CLAUDE.md records as the cause of a `?levels=4` refusal ("the old flat 512
(card) / 128 (cylinder) for every level >= 2 is exactly why `?levels=4`
refused"). It is latent rather than live on the two bodyless pages, which hold
zero active tiles; on the reentry page it is not obviously latent. Fixing it is
a behaviour change that needs its own measurement, not a drive-by inside a
refactor rung.

**U6 WENT FIRST AND SHARED ITS OWN BIND GROUP RATHER THAN WRITING IT FIVE
TIMES** (2026-09-18). `makeRenderBindGroup` is in `amr2d-gpu.mjs` and every page
calls it; fold it into this rung's shared builder when that exists, rather than
leaving two homes for the same wiring.

#### U7-3 — DONE (2026-09-18). The two orderings are one function each.

**922 lines removed, 557 added, a net −365.** Two functions, and both were
byte-identical across four pages before it:

- **`makeScheduler`** — `S_Advance`, AGAL's recursive multi-rate advance order.
  Byte-identical on cylinder / reentry / TGV / channel; `main-amr.js`'s differed
  only by the root pool and its measurement decoration.
- **`makeRefineRound`** — everything one `?refineEvery=` round encodes.
  Byte-identical on **all four** of those pages too.

**THE SEAM IS ORDER vs. CONTENT, and that is the whole design.** The shared
functions own WHEN each pass is encoded and the recursion or the loops that get
there. A `passes` object owns WHAT a pass is: which pipeline, which bind group,
which dispatch size, and whatever profiling or `?benchSkip=` decoration the page
wants around it. The order was identical five times over; the content
legitimately differs, because `main-amr.js` carries measurement twins, D0's
scan/link passes and the root's parallel passes that no shipped page has.

That split is why this rung was worth the risk. A mis-ordered pass here is a
physics bug, not a crash -- it still runs, still produces a field, and still
looks like a simulation -- and three separate parts of the refinement round are
subtle for three different reasons (criterion evaluated once before the sweep;
want buffers CLEARED not overwritten; coarsen finest-first but refine
coarsest-first, for the ALLOCATOR rather than for balance). All of that now has
one home, with the reasoning attached to it.

*Gate, everything, all green:* `make check`; boot smoke on seven configs plus
both render configs; `measure-determinism.js` unmoved at `7ac54e170f903ac3` /
`ce1bd4d8a3a1055c` -- which also covers `?detslots=1`, so the dev page's D0
scan/link ordering inside `denseCoarsen`/`denseRefine` is proven unchanged;
all four analytic AMR configs PASS; `amr-N2-diffuse` Cd 1.631 / St 0.1466 and
`amr-N3-diffuse` Cd 1.463 / St 0.1565, both unmoved with invariants green;
`validate-root-kernels.js` green on all eleven rungs; `amr-dev-invariants`
seven-of-seven with `?rootpool=1`.

**`dispatchMacroStep` ITSELF IS NOT SHARED, and the measurement says not to.**
It has three variants (amr / cylinder+reentry / tgv+channel), because what
surrounds the refinement round -- the force pass, the physics integrator, the
snapshot bookkeeping -- is scenario. The byte-identical part of it WAS the
refinement round, and that is what moved. This is the same line U7-2 drew:
share what is already identical, and leave what genuinely differs.

#### U7-4 — the root pool on every page

Allocate, seed, and expose the flags. **But NOT quite "everything else is
already shared" — U7-3 changed that, and the itemisation below now applies only
after a step U7-4 has to take first.**

**WHAT U7-3 ACTUALLY LEFT.** This rung was scoped on the assumption that
sharing the scheduler would carry the root pool to every page "by
construction". It did not, and the reason is the seam U7-3 chose: the shared
scheduler owns the ORDER, and a `passes` object owns the CONTENT. **The root's
step and the root's restriction are content.** So they live in `main-amr.js`'s
`passes` and nowhere else, and four pages still have no way to encode them.

Measured 2026-09-18, `main-amr.js`'s root-pool construction block is **372
lines**, and it splits cleanly:

```
  SOLVER      ~200   the mirror, the root step, the root criterion, and U5-3's
                     live interp/average pipelines and bind groups.
                     Needed by every page.
  INSTRUMENT  ~170   the root force pass, the full digest, and U5-1/U5-2's
                     inert scratch legs. Dev page only.
```
plus roughly 280 lines of `debugCheckRoot*` comparators, which stay on the dev
page for the reason already given.

**So U7-4 is two steps, not one:**

- **U7-4a — extract the SOLVER half** into `amr2d-gpu.mjs` as a
  `makeRootPool(...)` returning the mirror, the step, the criterion and the
  live coupling's pipelines and bind groups, plus the two `passes` entries the
  scheduler needs. `main-amr.js` becomes its first caller and keeps only the
  instrument and the comparators. *Gate:* `measure-determinism.js` unmoved BOTH
  with and without `?rootpool=1` (the second is what covers the live coupling),
  and `validate-root-kernels.js` green on all eleven rungs.
  **The interleaving is the work.** The instrument and the solver currently sit
  inside one `if (ROOT_POOL)` block with shared locals (the sentinel buffer,
  the scratch allocator), so the extraction is an untangle rather than a move.
- **U7-4b — the four pages call it**, which is then the itemisation below.

**The `~20 new lines and 7 single-token edits` figure is U7-4b's, and it holds
only once U7-4a is done.** It was written before U7-3 fixed the seam, and it
assumed a sharing that U7-3 deliberately did not do.

**"~60 LINES PER PAGE" WAS AN ESTIMATE AND IT WAS ABOUT 3x HIGH — for U7-4b.**
Itemised
against the real sites in `main-amr.js` (2026-09-18), what a page actually
grows once U7-0…U7-3 are done is roughly **20 new lines and 7 single-token
edits**:

```
  NEW LINES
   2   call a shared readRootFlags(urlParams) and destructure it
   3   if (ROOT_POOL) pools[0] = allocRootPool(...)
   1   quadAlloc argument on the existing allocLevelPool call
   2   the `if (!ROOT_MANAGED)` guard around level 1's per-block free-list seed
   1   quadCompleteFrom argument on makeCascadePipelines
   3   seedRootFromDense() at init, in resetSim, after debugSnapshotLoad
   2   the same guard inside resetSim
   4   debugActivateBlock's refusal under quad allocation
   1   getRootPool on the debug surface
  ---
  ~19  and 5 of those (the activate refusal, the snapshot seed) exist only on
       the three pages that HAVE those functions. tgv and channel have neither,
       and their resetSim keeps no CPU mirrors at all, so they grow ~4 lines.

  SINGLE-TOKEN EDITS TO EXISTING LINES
   the levelParams loop start          (ROOT_POOL ? 0 : 1)
   the pipelines loop start            (ROOT_MANAGED ? 0 : 1)
   the bind-groups loop start          (ROOT_MANAGED ? 0 : 1)
   the quadCPU loop start              (ROOT_MANAGED ? 1 : 2)
   the resetSim loop start             (ROOT_MANAGED ? 1 : 2)
   blockSlotCPUAtLevel's ternary
   criterionBG's target buffer
```

**AND THE SHAPE OF THAT RESIDUE IS THE USEFUL PART.** More than half of it is
not root-pool code at all -- it is LOOP BOUNDS in code that is itself
duplicated five times (the pool-allocation loop, the levelParams loop, the
`quadCPU`/reset scaffolding). Every one of those edits disappears if U7-2
shares the loop instead of only the bind groups it builds. So this itemisation
is also a scope note for U7-2: **share the pool-allocation and reset
scaffolding, not just the bind groups**, and U7-4 shrinks to the handful of
lines that are genuinely about having a root.

**The flags are the one thing that must NOT be copied.** The four `const`
declarations are four lines; the design record attached to them in
`main-amr.js` is about sixty-five lines of comment, and that belongs in one
place. A shared `readRootFlags` is where it goes.

**And the other four pages should probably never get `?rootcouple` or
`?rootmanage` at all.** They are staging flags -- they exist so U5-2, U5-3 and
U5-4 could be A/B'd in one build on the page being developed, and U7-5
collapses them. Giving four more pages a knob that is already scheduled for
deletion is work in both directions.

The COMPARATORS stay on `index-amr.html` -- they are the dev page's instrument
and five copies of them would be five copies of a checker, which is how this
project collected its vacuous gates in the first place.

One thing this rung was expected to make possible stays open and MOVES:
**`POOL_PEAKS` re-measured at quad granularity**, 40k steps, both pages, all
level counts. U5-4 measured +28-32% tiles at 4096 steps and left the peak table
outstanding; `?rootpool=1` is still opt-in on every page after U7-4, so that
demand is not yet anybody's shipped demand. It belongs with **U7-5**, which is
the rung that makes it one. The refusal watch reports a wrong guess either way,
and nothing refused over the 8192-step invariant sweep at the current defaults.
**DONE as U7-5a below** -- and the answer is narrower than U5-4's figure: the
rise is level 1's alone.

#### U7-4a — DONE (2026-09-18). The root pool's solver half is one function.

`makeRootPool(device, U, layouts, modules, pools, {...})` in `amr2d-gpu.mjs`,
with `allocRootPool` and `readRootFlags` beside it. **-6 lines net, which is
not the point: U7-4b is.**

The 372-line construction block split as the estimate said, and the two halves
landed where the estimate said:

```
  SOLVER      the mirror (which is also the SEEDER since 1.2), U3's root step,
              U4-1's criterion REDIRECT, U5-3's live coupling, and the two
              pass bodies the scheduler needs          -> amr2d-gpu.mjs
  INSTRUMENT  the root force pass, the full digest, U5-1/U5-2's inert scratch
              legs, and ~280 lines of debugCheckRoot*  -> stays on index-amr
```

**THE INTERLEAVING WAS THE WORK, exactly as scoped.** The two halves shared
locals -- the `unread` sentinel and `rootAvgPL`, both declared inside an inert
block and both read by the live path. Both are RETURNED rather than rebuilt, so
the instrument leg that scores the live restriction runs the live pipeline and
not a second copy of it that could drift from its subject.

**THE INERT CRITERION TWIN STAYED BEHIND, and that corrects this stage's own
itemisation**, which listed "the criterion" as solver. Under `managed` the live
level-1 criterion is `criterionPoolPLs[0]` -- built by each page's existing
per-parent-level loop once its bound starts at 0 -- so U4-1's `rootCritPL` is
never the live writer in any configuration; it is an instrument in both. What
the SOLVER needs from that stage is the REDIRECT: the shared refine round still
encodes `amr_criterion.wgsl`, and under quad management it must not land on
level 1's real criterion buffer. `denseCritBuf` is that target, and it is the
only piece of U4-1 in the shared half.

**`readRootFlags` takes `{ staging }`, and that is where "the other four pages
should probably never get `?rootcouple`" landed.** The dev page reads the full
set; the four shipped pages call `readRootFlags(urlParams, { staging: false })`
and get `?rootpool=` only, with the other three pinned on. One reader, one copy
of the sixty-five lines of design record, and U7-5 has one place to collapse.

*Gate, all four green:*
- `make check`; boot smoke on all seven configs plus both render configs.
- `measure-determinism.js` unmoved on the DEFAULT path (`7ac54e170f903ac3` /
  `ce1bd4d8a3a1055c`) **and** under `--extra=rootpool=1`, which is what covers
  the live coupling: `f71bce9d33945265` / `71560c03a3d34c21`, U5-4's own
  figures to the digit. Its `levels=2 detslots=0` row still comes back
  IDENTICAL under rootpool=1 and still exits nonzero for it, as U5-4 recorded
  and deliberately did not relax; `levels=3 detslots=0` still DIFFERS.
- `validate-root-kernels.js` green on all eleven rungs and all four controls.

#### U7-4b — DONE (2026-09-18). The four pages call it.

**The `~20 new lines and 7 single-token edits` estimate held.** Measured across
the four pages the shape is what the itemisation predicted: the three pages
with a body grow ~20 lines each, the two bodyless ones ~12 (they have no
`debugActivateBlock` refusal and no snapshot seed, but they do have their own
reset and the coupling selection, so "~4" was low). More than half of every
page's diff is still LOOP BOUNDS, as predicted -- and U7-2's decision not to
share the allocation and levelParams loops still looks right, because a bound
is one token and a shared loop with three per-page policies is not.

One thing the itemisation did not have:

**`makeRootPool` gained `rebuildStep`, because one page's step constants are
not fixed for the session.** `main-channel-amr.js` bakes Re into
`FORCE_X`/`WALL_U1` and recreates its step pipelines on every `setRe`. A root
left on the pipeline built at init would be driving a different flow from the
dense grid it is supposed to be a second copy of -- silently, since both still
step and neither goes NaN. The bind groups do not depend on the constants, so
only the pipeline is rebuilt. This is the same class as U3's inherited
`DIRECT_GHOST`: a constant the root must not be allowed to fall behind on.

*Gate.* Default path first, because this rung must not move it:
- `make check`; boot smoke on all seven configs plus both render configs.
- The four analytic AMR configs PASS.
- `amr-N2-diffuse` **Cd 1.630 / St 0.1466** and `amr-N3-diffuse` **Cd 1.463 /
  St 0.1565** -- unmoved from U7-3's readings (N2's 1.630 against 1.631 is
  inside that config's own ±0.001 floor; N3 is exact).
- `render-levels-cylinder` came back **bit-identical** to its pre-edit
  screenshot hash (`66b3e770d32be4c7`, 11884 B). An intermediate run of the
  same build gave `d978e035217d9400` instead, which is that page's known
  several-attractor behaviour and not a change -- the IDENTICAL is the
  conclusive reading, since no race can forge a bit-exact match.

Then the leg this rung exists to make possible, `--extra=rootpool=1`:
- Boot smoke green on all seven configs, **including the four pages that had
  never had a root pool before**.
- All four analytic AMR configs PASS, and both cylinder configs' seven
  invariants green at every checkpoint through 8192 steps.

**AND THE Cd/St CONSEQUENCE OF U5-4 IS MEASURED, which U5 said it could not
be.** First reading on a harness with literature values attached:

```
                      rootpool=0            rootpool=1 (quad-managed L1)
  amr-N2-diffuse   Cd 1.630  St 0.1466    Cd 1.607  St 0.1446
  amr-N3-diffuse   Cd 1.463  St 0.1565    Cd 1.473  St 0.1571
```

Both moves are far outside their configs' reproducibility floors (±0.001 at
N=2, ±0.002 at N=3), so they are real and not slot regrouping. N=3 still passes
both bands either way; N=2 stays the known-red cell, and the root pool does not
rescue it -- which is the expected answer, since that cell is the
diffuse-band-width issue and U5-4 changes refinement granularity, not the band.
**These are single readings.** U7-5's gate is where they get same-build repeats
on both sides, which is the protocol CLAUDE.md requires before a build-vs-build
claim on an AMR config.

#### `validate-root-kernels.js --page=` is the wrong instrument, and withdrawn

This stage promised that the tool "gains `--page=` and runs against the cylinder
page", so that U5's coupling is finally scored on a harness with literature
values. Those two things pull opposite ways, and the same stage says why: every
row in that tool is a differential against a dense counterpart, and the
comparators that compute it (`compareRootToDense` and the `debugCheckRoot*`
family, ~280 lines) are deliberately ONE COPY on the dev page. `--page=` would
mean porting a checker to a second page, which is precisely the move this
project has three vacuous gates from.

And it would buy nothing: after U7-4a the root's kernels on the cylinder page
are the SAME PIPELINE OBJECTS, from the same shared function, as the ones the
dev page already scores bit-for-bit. What the literature harness adds is not a
word diff, it is Cd and St -- which is `validate-all.js --extra=rootpool=1`,
run above. **That is the substitute, and the `--page=` item is withdrawn.**

#### AND U7-6's SNAPSHOT WORK IS A PREREQUISITE FOR U7-5, NOT A FOLLOWER

Found while porting `debugSnapshotLoad` (2026-09-18). The snapshot format only
ever saved `parentSlot`/`quadrant` from level 2 up, and rebuilds level 1's free
list at BLOCK granularity from `slotToBlock`. Under quad allocation level 1's
free list holds QUAD indices -- slot `q` and quad `q` are different things --
so a load writes a block-indexed list over a quad pool and the next refine
hands out overlapping quads. No thrown error, no NaN: a corrupted allocator.

This was latent on `main-amr.js` before this rung too, under `?rootpool=1`,
because no load-side gate has ever run with that flag set. It is now REFUSED
loudly on all three pages that have snapshots rather than degraded, with the
message naming U7-6. SAVE is deliberately not refused:
`tools/measure-determinism.js` fingerprints through it under
`--extra=rootpool=1`, and a hash of a consistent subset is still a hash.

**The refusal immediately turned `render-levels-card` red under
`?rootpool=1`**, because `tools/lib/render-levels.js` restores its baseline
through `debugSnapshotLoad`. That is the sequencing finding: at U7-5 the flag
becomes the default, and on that day the render-levels gate, `amr-diff.js`,
`validate-divergence.js` and every snapshot round-trip go with it. So the
snapshot half of U7-6's host-and-tool tail moves BEFORE U7-5. The two render
configs were consequently scoped out of THIS rung's `?rootpool=1` leg; **U7-6a
below does that work and they pass there**, along with a second format gap
nobody had predicted.

#### U7-6a — DONE (2026-09-18). The snapshot format carries what a load needs.

Split out of U7-6 and moved AHEAD of U7-5 for the reason U7-4b found: at U7-5
`?rootpool=1` becomes the default, and on that day every snapshot round trip
goes with it. Two things were missing from the format, and only the first was
the one U7-4b predicted.

**1. Level 1's quad indirection.** The format captured `parentSlot`/`quadrant`
from level 2 up and rebuilt level 1's free list at BLOCK granularity. Under a
root pool level 1 is quad-allocated (U5-4), so the restore wrote an indirection
at the wrong granularity over a quad pool. Level 1's entry now carries the pair
plus a `quadAlloc` marker, the restore splits on HOW THE LEVEL IS ALLOCATED
rather than on `m === 1`, and a marker mismatch is refused loudly instead of
loading. A capture from before this has no marker, which reads correctly as
"per-block", and there is no correct QUAD capture from before it to be
incompatible with -- save never wrote those fields.

**2. THE FREE LIST IS STATE, NOT BOOKKEEPING, and that one was not predicted.**
The load rebuilt it ASCENDING from `slotToBlock`, on a comment that argued:
"Free-list ORDER doesn't affect correctness (any permutation of the free slots
works equally as a stack), so this is exact, not an approximation, and avoids
growing the snapshot format for state that's fully redundant with slotToBlock."
Every clause of that is true and the conclusion does not follow. The order
decides WHICH slot the next grant hands out, so a rebuilt list gives the run an
equally-correct-but-different pool layout from the one it had. Measured on a
save+load in the middle of an otherwise identical run:

```
  levels=2   2 blockSlot + 2 slotToBlock entries moved
  levels=3   25 + 25, and 20 parentSlot entries
  both       fB64 and velB64 followed the permutation
```

So the round trip was never reproducible against an uninterrupted run, and
`?detslots=1` -- D0's whole point -- stopped meaning anything across a load.
The list is now saved and restored verbatim, with the old rebuild kept as the
fallback for captures that predate this.

#### `tools/validate-snapshot-roundtrip.js` — the gate, and why render-levels could not be it

`tools/lib/render-levels.js` already round-trips a snapshot and requires the
restore to return to baseline bit-for-bit. That check is real and it is not
this one: it renders IMMEDIATELY after the load, so it scores only what the
RENDERER reads -- the velocity pools and the indirection -- and never asks the
ALLOCATOR anything. Both defects above need a refine round to express
themselves, so render-levels passed through all of it on both pages. **It
passed under `?rootpool=1` on both pages while the format was still wrong.**

The new tool steps N, saves, loads, steps N more, and requires the result to be
bit-identical to an uninterrupted 2N. The second N is what makes it a gate:
it contains refinement rounds, so a wrong free list lands different slots and
moves the fingerprint. Three controls, all of which earned their place:

```
  alive     the capture at N must differ from the one at 2N
  stale     loading a snapshot 64 steps later must MOVE the outcome
  refuse    a level-1 granularity mismatch must THROW, not load
```

#### The self-inflicted bug this rung's own controls caught, which is the part worth keeping

The first cut of the save copied a quad-allocated level's free list at
per-SLOT length. `allocLevelPool` sizes `freeListBuf` as `maxFineBlocks * 4`
for a per-block level but `(maxFineBlocks / 4) * 4` for a quad one -- one i32
per QUAD. So the copy overran its source.

**A COMMAND-ENCODER VALIDATION ERROR DROPS THE WHOLE COMMAND BUFFER, NOT THE
ONE BAD COPY.** Every other `copyBufferToBuffer` in the same submit was
discarded with it, so `debugSnapshotSave` returned a snapshot whose dense `f`,
`vel` and `cardState` were ALL ZEROS -- at `?levels>=3` only, because that is
when the per-level loop runs. The page itself was fine throughout: refinement
ran, tiles were allocated, the card fell. Only the instrument was blind.

**Three of the four gated rows went GREEN on it**, because zeros equal zeros.
What did not go green was the `stale` control, which said the outcome at 1088
steps was identical to the outcome at 1024 -- an impossible claim, and the only
reason the defect was found at all. That is the case for controls in one
paragraph: the gate reported a pass, and the control reported something that
could not be true.

The guard that would have caught it directly is now in the tool: the reference
leg fingerprints at N as well as 2N and ABORTS the row if they are equal,
because every row below is an EQUALITY and a capture that carries nothing
passes all of them. The copy length now comes from `freeListBuf.size` rather
than being recomputed from `MAX_FINE_BLOCKS` at the call site -- the free
list's SIZE is granularity-dependent the same way its CONTENTS are, and that
is one rule, in one place, in the allocator that owns it.

#### The determinism fingerprints move ONCE, and the solver does not

`measure-determinism.js` hashes the whole snapshot, so adding the free list to
the format moves every recorded value. Re-baselined on the default path:

```
  levels=2 detslots=1    b2b4d310a03ab626    (was 7ac54e170f903ac3)
  levels=3 detslots=1    d799e95d23ea47f0    (was ce1bd4d8a3a1055c)
```

**The tile counts are unchanged** -- `[75]` and `[103,240]`, the same numbers
U7-0…U7-4 recorded -- and the four analytic AMR configs and both cylinder
Cd/St readings are unmoved, which is the evidence that what changed is the
fingerprint's SUBJECT and not the solver. A snapshot field cannot affect a run
that never loads one.

And the new subject is the better one: the fingerprint now also gates that slot
handout state is reproducible, which is exactly what D0 is about.

*Gate, all green:* `make check`; `validate-snapshot-roundtrip.js` 12 of 12
(4 gated x {levels 2,3} x {rootpool 0,1}, plus 4 stale and 4 refusal controls);
boot smoke on seven configs plus both render configs, default and
`?rootpool=1`; both render configs PASS under `?rootpool=1`, which they could
not before this rung; the four analytic AMR configs PASS; `amr-N2-diffuse` and
`amr-N3-diffuse` unmoved.

#### What is LEFT in U7-6 after this

The snapshot's own remaining item is the one that belongs with the deletion,
not before it: **the format gains the root pool and loses the dense arrays.**
Until U7-6 the root is a byte-identical second copy of the dense L0 and
`seedRootFromDense` reconstructs it exactly on load, so the format does not
need to carry it yet. The rest of the host-and-tool tail is unchanged.

#### The quad granularity: what it is, and what actually forces it (2026-09-18)

U5-4 and U7-4b both state "level 1's refinement granularity goes from one
8-cell block to a 16-cell quad" as a fact without the mechanism, and the
mechanism turns out to matter for U7-5's decision. Traced through the shaders:

**The two managers are dispatched over different things.**

```
  amr_manage.wgsl        one thread per level-1 BLOCK; grants ONE slot from a
  (dense L0 parent)      per-SLOT free list. Unit of refinement: one block,
                         RB = 8 cells per axis.

  amr_manage_pool.wgsl   one thread per PARENT SLOT; grantQuad() hands out
  (pool parent)          FOUR slots at once, from a per-QUAD free list.
                         refineWants tests quadrant 0 only ("quadrant 0 stands
                         for all 4"); coarsen releases all four together.
```

`rootPoolSpec` makes a root tile `2*RB` = 16 cells per axis and level 1's block
grid exactly twice the root's, so one root tile carries four level-1 blocks:

```
  root tile (16 cells/axis)        its four level-1 children (8 cells/axis each)
  +---------------+                +-------+-------+
  |               |                |  q0   |  q1   |
  |               |      -->       +-------+-------+
  |               |                |  q2   |  q3   |
  +---------------+                +-------+-------+
      dense manager grants any ONE of these; the pool manager grants all FOUR
```

The want-set closure agrees with the allocator rather than fighting it:
`quadCompleteFrom` goes 2 -> 1 at U5-4, so the 2:1 cascade promotes "I want
this block" into "I want this quad" before the manager ever sees it. A region
that used to earn one 8-cell block now earns a 16-cell quad. That is the
+28-32% tiles, and it is why Cd moves.

**AND IT IS A CONVENTION, NOT SOMETHING THE ADDRESSING FORCES.** This is the
part worth having written down, because both earlier stages read as though the
quad were structural. `shaders/common_interp_parent_pool.wgsl`:

```wgsl
fn quadrantOf(slot, bx, by)   { if (PARENT_GHOST == 0u) { return ((by & 1u) << 1u) | (bx & 1u); } return quadrant[slot]; }
fn parentSlotOf(slot, bx, by) { if (PARENT_GHOST == 0u) { return (by >> 1u) * parentNbx() + (bx >> 1u); } return u32(parentSlot[slot]); }
```

Under a ROOT parent (`PARENT_GHOST == 0`) a child derives its parent slot and
its quadrant from its OWN BLOCK COORDINATES, because the root is always full --
not from its slot index. So a level-1 child does not have to sit in an aligned
group of four for the coupling to address it. At levels >= 2 those two values
come from `parentSlot[]` / `quadrant[]` BUFFERS, which would tolerate any
grouping just as well.

What the quad grouping actually is: "decision 3, all-or-nothing" from
plans/AMR-multilevel.md (not in this tree), which the code then EXPLOITS --
B2-2b0 recovered a storage binding from the 16-buffer ceiling precisely because
`quadrant == slot % 4` holds under it. `allocLevelPool`'s own comment names the
trigger: per-block allocation is "level 1 while its parent is the DENSE grid,
which is not itself decomposed into quads, so there is no quad on that
boundary." Once L0 becomes tiles, level 1 inherits the sibling convention every
deeper level already follows.

**So the coarser granularity is the price of UNIFORMITY -- one manager, one
allocator, `amr_manage.wgsl` retires -- and not a consequence the scheme
compels.** If the tile cost or the Cd move turns out to matter, "quad-allocate
level 1" is negotiable in a way U5-4 and U7-4b both implied it was not. What is
NOT established either way is whether per-block granularity is BETTER: the
moves measured so far are 1.630 -> 1.607 at N=2 (toward the literature 1.35)
and 1.463 -> 1.473 at N=3 (away from it), both small and both single readings.
That is a question for a measurement, not for this note.

#### U7-5a — DONE (2026-09-18). Pool demand re-measured at quad granularity.

`tools/measure-pool-peaks.js`, the persisted form of the ad hoc scan that
produced each page's `POOL_PEAKS` on 2026-09-15. Both pages, both allocators,
40000 steps, every level's cap lifted.

```
  index-amr.html (chaotic -- read the ABSOLUTE peaks, not the ratio)
                 rootpool=0                rootpool=1
    levels=3     230  412                  288  408
    levels=4     267  512  684             324  468  632
    levels=5     268  564  660  872        356  564  784  1184

  index-cylinder-amr.html (pinned, steady -- read the RATIO here)
                 rootpool=0                rootpool=1
    levels=3      83  100                  132  100
    levels=4     101  164  208             132  160  208
```

**THE COST IS LEVEL 1 AND NOTHING ELSE.** On the steady page L2 is 100 against
100 and L3 is 208 against 208 -- identical to the tile across the allocator
change -- while L1 goes +59% at levels=3 and +31% at levels=4. Under quad
allocation L1 lands on 132 at BOTH depths, because level 1's tile set is
decided by which root quads are wanted (geometry plus 2:1 closure) and that
does not depend on how deep the hierarchy goes below it.

That is narrower than U5-4's "+28-32% tiles", which averaged the rise over the
whole hierarchy and therefore understated it at level 1 and overstated it
everywhere else. It is also what the mechanism predicts: level 1 is the only
level whose allocator changed.

**AND U7-5 IS NOT BLOCKED BY POOL SIZING, WHICH CORRECTS THIS PLAN'S OWN
WORRY.** The concern was that flipping without re-sizing risks refusals.
Measured against the CURRENTLY SHIPPED defaults, nothing refuses at any depth
on either page; what happens is that the 1.7x headroom convention erodes:

```
  index-amr.html   levels=3  L1 288/444 = 1.54x   L2 408/680 = 1.67x
                   levels=4  L1 324/444 = 1.37x   ...
                   levels=5  L1 356/444 = 1.25x   L4 1184/1537 = 1.30x
```

So the sizing change is a margin restoration, not a rescue.

**THE CONSTANTS ARE NOT CHANGED ON THIS RUNG, and that is the measurement
being honest about its own resolution.** The rootpool=0 control leg reproduces
the shipped table to within ~9% on the card and ~6% on the cylinder, which is
this instrument's precision on each page; every rootpool=0 delta is inside it.
Adopting them would be fitting noise and paying VRAM for it. The quad numbers
go in WITH the flip at U7-5, because a peak table belongs to the allocator it
describes:

```
  main-amr.js          finest { 2: 412, 3: 684, 4: 1184 }
                       parent { 1: 356, 2: 564, 3:  784 }
  main-cylinder-amr.js finest { 2: 100, 3:  208 }
                       parent { 1: 132, 2:  164 }
```

Max over both legs, since `?rootpool=` stays selectable until U7-6. At 1.7x
that moves the shipped card configuration from L1 444 / L2 680 to L1 608 /
L2 704 -- **about +2 MB, which is the whole memory cost of the flip.**

**One instrument note worth carrying.** The peak is a SAMPLED max: nothing in
the page tracks a high-water mark, so the tool polls every `--sample=` steps
and a spike between samples is missed. The 1.7x convention absorbs that, and
the numbers being replaced were obtained the same way, so the comparison is
like for like. A level sitting at its cap is reported as a CLIP rather than a
demand and exits nonzero -- a clipped peak is the one number this measurement
must not hand on.

#### U7-5 — DONE (2026-09-18). The default is flipped.

`readRootFlags` defaults `rootpool` to 1. The root is a pool level, level 1 is
its quad child, `amr_manage.wgsl` is not dispatched. `?rootpool=0` survives as
the ESCAPE and goes with the dense path at U7-6. Both `POOL_PEAKS` tables
adopted in the same commit, because a peak table belongs to the allocator it
describes.

**THE STAGING FLAGS DID NOT COLLAPSE, AND THE GATES ARE WHY.** This rung was
written as "`?rootcouple` and `?rootmanage` collapse into it or go". Measured
against what reads them, that is wrong for two of the three: ALL FOUR of
`validate-root-kernels.js`'s CONTROL rows are `?rootstep=0` / `?rootcouple=0`,
and those controls are what make its eleven gated rows mean anything. Deleting
an instrument to satisfy a plan line is how this project collected vacuous
gates. They go at U7-6 with the dense path they compare against. `rootmanage`
has no tool reading it and could have gone; it stays because U7-5a made its
question live -- it is the only handle on whether the quad convention is the
better one.

#### The published numbers barely moved, which is the opposite of the prediction

This rung was billed as "where the published numbers move for real". With each
path properly sized they mostly do not. Same build, same `POOL_PEAKS`,
same-build repeats on both sides -- though see "The bisect, and the retraction
it forced" below for why +/-0.001 is a WITHIN-ATTRACTOR figure on this page and
understates the real uncertainty by about ten times:

```
                     run A            run B           floor
  rootpool=1   N2    1.624 / 0.1450   1.623 / 0.1450  +/-0.001
               N3    1.473 / 0.1570   1.473 / 0.1570  +/-0.002
  rootpool=0   N2    1.623 / 0.1458   1.623 / 0.1458
               N3    1.463 / 0.1565   1.463 / 0.1565
```

`amr-N2-diffuse`'s Cd does not move at all (1.623 against 1.6235, inside its
own floor). `amr-N3-diffuse` moves +0.010 against a +/-0.002 floor, which is
real and is the whole measured Cd consequence of the flip. Every St shift is
<= 0.0008.

**BOTH SIDES ARE THE SAME BUILD, which is better than the protocol asked for.**
The gate called for same-build repeats against a re-baselined sweep; after the
flip `?rootpool=0` IS the old path, so the A/B needs no second checkout and a
difference cannot be a build difference.

#### POOL CAPACITY IS NOT INERT, and nothing recorded that before

> **RETRACTED.** The heading is wrong and is kept for the trail. Capacity IS
> inert once slot assignment is deterministic -- measured bit-identical on
> `index-amr.html?detslots=1` for 444 against 1024. What varies below is the
> cylinder page's racing allocator. See "The bisect, and the retraction it
> forced". The practical consequence is REPLACED by a sharper one: that page's
> Cd carries an attractor spread of ~+/-0.009, ten times the same-build repeat.

The finding that cost the most to get to, and the one to carry forward.

U7-4b measured `?rootpool=1` at Cd 1.607 and the dense path at 1.631 -- a 0.024
spread it reported as U5-4's Cd consequence. Those two readings were taken on
the OLD `POOL_PEAKS`. On the adopted ones the same two configurations read
1.624 and 1.623. So the pool table moved BOTH legs, including the dense one,
whose allocator did not change at all:

```
                   rootpool=0   rootpool=1
  old POOL_PEAKS   1.631        1.607
  new POOL_PEAKS   1.623        1.624
```

**It is not starvation.** Measured directly at the depth `amr-N2-diffuse` runs,
which U7-5a's sweep had skipped: level-1 demand at `levels=2` is 61 tiles on
the dense path and 104 under quad, against a pool that went 164 -> 228. Neither
path was ever close to binding, and nothing was refused.

So the mechanism is slot REGROUPING: `MAX_FINE_BLOCKS` sets the free-list length
and the dispatch depth, and CLAUDE.md's own note says regrouping slots regroups
`amr_force1.wgsl`'s truncated per-workgroup partials. What is new is the SIZE
of it. CLAUDE.md records that as a 4th-digit effect, "+/-0.001, a build-vs-build
claim needs a same-build repeat"; here changing ONE capacity constant, with
demand untouched, moved Cd by **0.008** -- eight times that.

Two consequences, both worth acting on:

- **A capacity constant chosen for headroom is a physics parameter.** Any
  future `POOL_PEAKS` retune moves the published Cd, and nothing in the repo
  said so. It is recorded at both pages' tables now.
- **Cd readings are only comparable within one pool table.** The plan's earlier
  figures (1.631, 1.607, 1.642 in CLAUDE.md) each belong to whatever table was
  in force, and cross-table comparison is what made U7-4b's 0.024 look like an
  allocator effect when most of it was not.

RESOLVED, and it opened something larger -- see "The pool-capacity sweep"
below. Short version: it is scatter of about +/-0.009, the ALLOCATOR is not the
variable at all (matched capacities agree to <= 0.001), and the refined BLOCK
SET genuinely differs between capacities on a page where force cannot feed
back. The mechanism is not established.

#### One instrument caught a flip consequence, which is the argument for its assertion

`validate-snapshot-roundtrip.js`'s `rootpool=0` rows relied on the flag
DEFAULTING to 0, so after the flip both of its legs were the quad allocator and
the suite would have gone green on four copies of one configuration. `open()`'s
"assert the flags took" check turned that into an error instead. Both legs now
name `?rootpool=` explicitly.

*Gate, all green:*
- `make check`, `make test` (8 suites).
- Boot smoke on seven configs plus both render configs.
- All four analytic AMR configs PASS.
- `amr-dev-invariants` seven-of-seven; the starved-pool control still
  discriminates **five-of-seven**, with `field` and `quadrants` abstaining.
- `validate-snapshot-roundtrip.js` 12 of 12, both legs genuinely different
  allocators, all four `stale` controls moving.
- `measure-determinism.js` re-baselined on the new default:
  `8ddd3ff2f84a697f` (levels=2) / `3d80fa737af6bf9e` (levels=3), IDENTICAL over
  FOUR runs each. **Read at `--runs=4`, and that mattered**: at two runs the
  `levels=3 detslots=0` baseline came back IDENTICAL, which would have left
  every row in the tool reading IDENTICAL and the claim unfalsifiable. At four
  it DIFFERS, so the instrument keeps its discrimination. That is the tool's
  own header being right about itself.


#### The pool-capacity sweep, and what it found instead (2026-09-18)

> **SUPERSEDED IN ITS CONCLUSION by "The bisect, and the retraction it forced"
> below.** Capacity is NOT a physics parameter; this section measured the
> cylinder page's RACING allocator, which has no `detslots` implementation. The
> measurements here are sound and the reasoning from them was not.

U7-5 left open "whether 0.008 is the true scale of slot regrouping or whether a
second mechanism is involved". Swept `?maxFineBlocks=` on `amr-N2-diffuse`
(`index-cylinder-amr.html?levels=2`), where measured level-1 demand is 61 tiles
dense and 104 quad, so every capacity below is far above demand and nothing is
ever refused. Only the pool's SIZE varies.

```
  maxFineBlocks     rootpool=0            rootpool=1
        128      Cd 1.641  St 0.1475
        164      Cd 1.631  St 0.1466          <- the OLD POOL_PEAKS value
        228      Cd 1.623  St 0.1458      Cd 1.623  St 0.1449   <- the NEW one
        320      Cd 1.639  St 0.1474
        512      Cd 1.638  St 0.1473
       1024      Cd 1.639  St 0.1474      Cd 1.640  St 0.1480
```

**THREE THINGS, AND THE THIRD IS THE ONE THAT MATTERS.**

**1. It is scatter, not a trend.** The first three points fall monotonically
and that is a coincidence; 320, 512 and 1024 come back up. Range 1.623-1.641,
about +/-0.009 around 1.633. So it is not a systematic count-of-truncations
effect.

**2. THE ALLOCATOR IS NOT THE VARIABLE; CAPACITY IS.** At matched capacity the
two allocators agree to <= 0.001 (228: 1.623 vs 1.623; 1024: 1.639 vs 1.640).
That settles U7-5's headline independently: flipping `?rootpool=` does not move
Cd. It also means the capacity sensitivity is NOT something U5-4 or U7-5
introduced -- it is on the dense path too, and predates all of this.

**AND IT RE-SCOPES CLAUDE.md's REPRODUCIBILITY NOTE.** "AMR Cd is only
reproducible to ~+/-0.001" holds WITHIN a fixed configuration. Against any
perturbation that regroups slots -- capacity, allocator, plausibly refine
cadence -- the nuisance spread is ~+/-0.01, ten times larger. `amr-N2-diffuse`
therefore cannot resolve a physics difference below roughly 0.02, and U7-5's
"+0.010 at N=3, real" should be read as "inside the nuisance band", not as a
physics result.

**3. St MOVES WITH Cd, AND THE REFINED SET ACTUALLY DIFFERS.** Strouhal is a
frequency and is robust to offsets in force magnitude, so its tracking Cd
argues against a pure force-integration artifact. Checked directly, comparing
ACTIVE BLOCK SETS (slot-independent) at a fixed step:

```
  index-cylinder-amr.html?levels=2&detslots=1&rootpool=0, 20000 steps
    CONTROL  cap 228 vs cap 228    SAME    44 blocks, 0 differences
    QUESTION cap 228 vs cap 1024   DIFFER  44 vs 46 blocks, 1 only-A, 3 only-B
             the differing blocks are bx=63 -- NBX is 64, so the LAST block
             column, the downstream outflow edge
```

So refinement itself depends on pool capacity with the pool nowhere near full.
The control is what makes that readable: the same capacity twice gives the
identical 44-block set, so `?detslots=1` is pinning the page and the difference
is not the racing free list.

**THE MECHANISM IS NOT ESTABLISHED, and the cylinder being PINNED rules out the
easy explanation.** Force cannot feed back into the flow on this page, so a
force-rounding difference cannot move the refinement. Something else is
capacity-dependent. Two candidates, neither verified:

- a pass reading INACTIVE slots -- their stale contents differ when there are
  978 of them instead of 184. `plans/2D-backport.md` B6-9c is this class, and
  makeRootPool's `unread` sentinel exists because of it.
- an order-dependent float accumulation not yet located.

**THE DISCRIMINATING EXPERIMENT IS A BISECT ON STEP COUNT.** Compare the two
capacities' active sets (and a reconstructed, slot-independent field) at
increasing step counts and find the FIRST divergence. Diverging at the very
first refine round is structural -- something reads capacity directly.
Diverging later and growing is amplification of a tiny numerical difference,
which points at the inactive-slot candidate. The probe is small enough to
restate:

```js
  // per capacity: navigate ?levels=2&detslots=1&rootpool=0&maxFineBlocks=CAP,
  // reset, debugStepSync(N), then
  debugListActiveBlocks(1).then(a => a.map(b => b.bx + ',' + b.by).sort())
  // KEY ON (bx,by), NOT the object: listActiveBlocks returns {bx,by,slot} and
  // stringifying it gives "[object Object]" for every entry, which collapses
  // the comparison to SAME. The first run of this probe did exactly that and
  // was caught only because it printed SAME next to |A|=44 |B|=46.
```

**THIS IS NOT A U7 BLOCKER.** It is on the dense path too, it long predates the
root pool, and U7-6 deletes neither the pool allocator nor the criterion. But
U7-6 DOES delete the dense path, which is the control this comparison uses, so
the bisect is cheaper before it than after.

#### The bisect, and the retraction it forced (2026-09-18)

The pool-capacity section above concluded "pool capacity is not inert" and
"a constant chosen for headroom is a physics parameter". **BOTH ARE WRONG, and
the bisect is what showed it.** Left in place above because the route to the
right answer is the useful part; read this section as its correction.

**CAPACITY IS INERT WHEN THE HANDOUT IS DETERMINISTIC.** On `index-amr.html`
under `?detslots=1&levels=3`, capacities 444 against 1024 -- both far above the
~230-tile demand -- are BIT-IDENTICAL in the dense L0 field AND in `cardState`
at every checkpoint from step 16 through 4096. Not close: identical.

```
  cap  444 A dense: 16:1b75dd 32:2c8908 ... 2048:61ed8d 4096:00fdd3
  cap  444 B dense: 16:1b75dd 32:2c8908 ... 2048:61ed8d 4096:00fdd3
  cap 1024   dense: 16:1b75dd 32:2c8908 ... 2048:61ed8d 4096:00fdd3
  (cardState likewise identical on all three)
```

**WHAT THE CYLINDER SWEEP ACTUALLY MEASURED IS THE RACE.**
`main-cylinder-amr.js` HAS NO `detslots` IMPLEMENTATION -- D0's deterministic
handout is main-amr.js-only, which U7-1 records and which this investigation
forgot. So `?detslots=1` there is an inert URL parameter and the `atomicSub`
free list races. Changing the capacity changes the race's outcome; slot
assignment regroups; `amr_force1.wgsl`'s TRUNCATED per-workgroup partials
regroup with it. That is CLAUDE.md's own mechanism, and its "several
attractors, each bit-exact" is the same observation from the other side: the
same capacity lands in the same attractor and repeats to +/-0.001, a different
capacity can land in one 0.018 away.

**THE CONSEQUENCE THAT SURVIVES, AND IT IS THE VALUABLE ONE.** The +/-0.001
same-build repeat that CLAUDE.md records, and that U7-5's gate leaned on, is a
WITHIN-ATTRACTOR figure. It understates the real uncertainty of an AMR Cd on
`index-cylinder-amr.html` by roughly TEN TIMES. Anything that perturbs the race
-- pool capacity, the allocator, the refine cadence -- can move Cd by up to
0.018 with no physics change whatsoever. That is the page every published Cd/St
number in this project comes from.

So U7-5's "+0.010 at N=3" is NOT a measured consequence of the flip. It is
inside the attractor spread, and this harness cannot currently resolve it.

**THE FIX IS NAMED AND NOT DONE: port DET_SLOTS to main-cylinder-amr.js.** It
is the one change that would make that page's Cd genuinely reproducible and
retire the nuisance instead of budgeting for it. D0 already did the work once;
U7-1 deliberately left the scan/link pipelines on the dev page because nothing
else needed them yet. This is the thing that needs them. **Staged as D1 below**,
ahead of U7-6, which deletes both its subject and its control.

#### Three instrument errors this cost, all worth keeping

They are recorded because each one produced a confident wrong reading first.

1. **A flag that the page does not implement reads as a clean negative.**
   `?detslots=1` on the cylinder page. The control diverging was interpreted as
   "the cylinder is anomalously nondeterministic" before the grep. Tools that
   assert their flags took -- `measure-pool-peaks.js`, `validate-snapshot-
   roundtrip.js` -- exist because of exactly this, and this probe did not.

2. **A whole-snapshot fingerprint is SLOT-SENSITIVE and cannot answer a
   slot-regrouping question.** Two capacities hand out different slots by
   construction (the free list starts `[0..cap-1]` and refine takes the top),
   so the pool's `f`/`vel`/`blockSlot`/`slotToBlock` differ even when the
   physics is bit-identical. The first bisect reported "diverges at every
   checkpoint" on that basis. The dense L0 grid is indexed by (x,y) and carries
   level 1's restriction, so it is the state with the bookkeeping projected
   out; `cardState` separates the force path from the flow.

3. **`listActiveBlocks` returns `{bx, by, slot}`, and stringifying the object
   gives `"[object Object]"` for every entry** -- which collapses a Set to one
   element and makes every comparison come back SAME. Caught only because the
   probe printed SAME next to `|A|=44 |B|=46`. Key on `bx + ',' + by`, which is
   also the slot-independent thing to compare.

**AND ONE REAL, SMALL FINDING ON THE SIDE.** With the whole-snapshot
fingerprint, two same-capacity runs on `index-amr.html?detslots=1` differed at
step 16 and were identical from 32 onward. Dense-only, they are identical
throughout. So something slot-indexed and transient is not pinned by
`?detslots=1` in the first refine round -- most likely the free list's residual
ORDER, which U7-6a put into the snapshot. It heals, and it does not touch the
physics, but it means the snapshot fingerprint is very slightly stronger than
"the solution" and could produce a spurious DIFFERS at a fine checkpoint.

### D1 — the deterministic handout on the cylinder page

**WHY, in one line:** `index-cylinder-amr.html` is where every published Cd/St
number comes from, and its allocator races, so those numbers carry an attractor
spread of about +/-0.009 -- ten times the +/-0.001 same-build repeat the gates
quote. See "The bisect, and the retraction it forced". D0 already built the
deterministic handout; U7-1 deliberately left it on `main-amr.js` because
nothing else needed it yet. This needs it.

**IT IS SMALLER THAN IT SOUNDS, and the split is along the path U7-5 made the
default.** Measured against the two managers:

```
  amr_manage_pool.wgsl   DET_SLOTS is ENTIRELY IN-SHADER -- refine() and
  (the default path      coarsen() each carry a serial one-thread branch. No
   after U7-5)           new pipelines, no new bindings. The port is an
                         override and a flag.

  amr_manage.wgsl        DET_SLOTS needs FOUR extra pipelines host-side --
  (only under            scanCandidates x2 (grant/release) and linkCoarsen /
   ?rootpool=0)          linkRefine -- plus their encoding around coarsen and
                         refine, in an order that is load-bearing.
```

The bind group needs nothing either way: `candRankBuf` is already bound at
binding 7 on every page, because U7-0 made the fourteen layouts shared. That is
the rung paying back exactly as it promised.

#### D1-0 — DONE (2026-09-18). The instrument addresses any page, and gates only where it has measured.

`tools/measure-determinism.js` gained `--global=`, defaulted from the same page
table `tools/measure-pool-peaks.js` uses (`index-amr` / `index-reentry-amr` ->
`window.__AMR`, the cylinder/TGV/channel harnesses -> `window.__CYL`), and the
six hardcoded `window.__AMR` references became one `G`.

The gate table moved with it. `CONFIGS`' expectations are `index-amr.html`'s
MEASURED behaviour, so `GATES_BY_PAGE` now decides whether a page is SCORED at
all: a page not in that table prints every verdict, says why it is not gated,
and exits 0. `--gate` / `--no-gate` override by hand, which is what the
measuring run itself uses.

*Gate, met:* the tool reproduced `8ddd3ff2f84a697f` (levels=2) and
`3d80fa737af6bf9e` (levels=3) on `index-amr.html` at `--runs=4`, bit for bit,
after the refactor.

**THOSE TWO HASHES ARE FORMAT-v5 AND ARE SUPERSEDED.** `fingerprint` walks the
whole snapshot object, so U7-6c adding a `root` key moved every hash in the
table without moving any physics: `e0b1b22bbe47cbfb` / `4705ea2b06226c8b` at
formatVersion 6, re-measured 2026-09-22, IDENTICAL over four runs each. Check
the format version before reading a moved hash here as a regression.

**AND THE CARD PAGE'S OWN DISCRIMINATION RUNG IS NOW VACUOUS IN THE DEFAULT
CONFIGURATION.** `levels=2 detslots=0` came back IDENTICAL over 4 runs
(`8ddd3ff2f84a697f`, the same hash as `detslots=1`). The tool's header
predicted exactly this for quad-managed level 1 and recorded it under
`--extra=rootpool=1`; U7-5 then made `rootpool=1` the default, so the header's
"under that flag" is now "with no arguments at all". `node
tools/measure-determinism.js` therefore exits 1 on a healthy build, by design.
`levels=3 detslots=0` still DIFFERS (three distinct hashes in four runs) and is
what keeps that page honest. The header now says so; the gate was NOT relaxed.

#### D1-a — DONE (2026-09-18). Three edits, as predicted. And then the page had two defects and a third source.

The port itself was exactly the three edits the rung specified -- `DET_SLOTS`
from `?detslots=`, into `poolConstants`, and `getDetSlots` on `window.__CYL` --
because `amr_manage_pool.wgsl` carries the serial handout in-shader and U7-5
made that the path level 1 runs on. No pipelines, no bindings.

*Gate:* NOT MET, and the reason is the second of the two outcomes the rung
said not to paper over. It took two page fixes to find out which.

##### The first measurement, and the harness confound under it

Reported, not gated (D1-0's rule), 4 runs x 4096 steps:

```
  levels=2 detslots=1   IDENTICAL
  levels=3 detslots=1   IDENTICAL
  levels=2 detslots=0   DIFFERS   four distinct hashes in four runs
  levels=3 detslots=0   DIFFERS   two attractors, 3+1
```

On that reading this page DISCRIMINATES BETTER THAN THE CARD PAGE DOES TODAY --
both baseline rungs race here, where `index-amr.html`'s `levels=2` no longer
does. So `GATES_BY_PAGE` gained the page and the run was repeated with scoring
on. **It then failed, and the repeat is the only reason any of the rest of this
was found.** One green sweep would have closed the rung on a wrong result.

The first thing the repeat exposed was the harness's own: every run line read
`steps=4160` for a 4096-step request, against `4096` on the card page. Both
pages boot `liveMode = true`, so an uncounted 64-step rAF frame was landing
between the `reset()` round trip and the `debugStepSync()` one. The tool now
calls `setLive(false)` before `reset()` and ERRORS if the returned step count
is not the one it asked for. **The two pages had not been being measured the
same way, and reading the failure before fixing that would have been a
conclusion about the harness.**

##### Defect 1: `resetSim()` was not reproducible, and not even a fixed point

With the step counts clean the failure survived, so the initial state was the
next thing to doubt. Three consecutive `reset()` calls in one page load, no
stepping at all between them, then `debugSnapshotSave`:

```
  resets  1        2        3          each internally stable to the bit,
  state   A        B        C          all three DIFFERENT
```

`reset()` was not idempotent. The snapshot localised the difference to `fB64`
and `pools[1].fB64` and NOTHING else -- the field, never the allocator.

Cause: `const rng = mulberry32(SEED)` is MODULE-SCOPED on this page. That is
deliberate and the comment says why (the coarse and pool inits draw one stream,
not two copies of one), but it was created once at load and never re-seeded, so
`initF()`'s draws were a function of how many times `initF`/`initFPool` had
already been called. `main-cylinder.js` -- the dense reference whose Cd
CLAUDE.md records as bit-exact across every run -- builds its rng INSIDE
`initF()` and never had this. Note also that boot draws pool-then-coarse while
`resetSim` draws coarse-then-pool, so the booted field and the post-reset field
are different patterns either way; every tool resets before measuring, so the
reset one is the one that matters.

Fixed: `let rng` plus `reseedRng()`, called first thing in `resetSim()`.

##### Defect 2: the velocity fields are state, and reset did not write them

`reset()` became a fixed point within a page load and STILL differed BETWEEN
navigates. The same probe, diffing navigate-to-navigate at reset 1:

```
  .velB64 x1, .pools[].velB64 x1, .pools[].parentSlot[] x8
```

`velBuf`, `finePoolVel` and `parentSlotBuf` were never written by `resetSim`,
so they held whatever ran before it. That is not cosmetic: `dispatchMacroStep`
runs the refinement round BEFORE `S_Advance`, reading "each level's own
velocity field as populated by the PREVIOUS macro-step", and `macroStepCounter
= 0` means **the first macro-step after a reset IS a refinement round**. So the
first refinement after every reset was against page history.

Fixed: `initF`/`initFPool` fill a velocity array from the SAME draw (`f` is an
equilibrium distribution, so its velocity is exactly `(U0, uy)` and no moment
has to be taken), `resetSim` writes it, and `parentSlotBuf` goes back to the
zero fill `allocLevelPool` left. `quadrantBuf` is deliberately not cleared --
it holds `slot % 4`, a constant, not allocator state.

After both fixes, three navigates x three resets: **ONE hash, `4e8eb1f5a4c02158`,
everywhere.** Reset is a complete fixed point, cold start included.

**BOTH FIXES MOVE THIS PAGE'S PUBLISHED Cd/St.** The initial condition changed:
the perturbation field is now the one `SEED` names rather than one further down
the stream, and the first refine round now sees the IC rather than a leftover.
Re-baseline against a fresh measurement, not against the numbers in CLAUDE.md.
`index-cylinder.html`, the dense reference, is untouched and is the control.

##### And the gate still fails, which is the finding

From a provably bit-identical initial state, 6 runs x 4096 steps,
`?levels=3&detslots=1`: five runs agree, one does not. The outlier is no longer
the first navigate, so it is not the cold-start artefact either.

What differs is the same shape every time, and it is NOT the field first:

```
  pools[1].blockSlot / slotToBlock / parentSlot   shifted by exactly 4 -- ONE QUAD
  fB64, velB64, pools[].fB64, pools[].velB64      downstream of that
  cardState[6..8]                                 the force accumulators
```

A uniform one-quad shift in the handout means the free list was one quad
different at some grant round. `DET_SLOTS` makes `refine()` and `coarsen()`
single-threaded in dispatch order with no atomics in the decision, so the
handout cannot reorder itself -- which suggested the WANT SET differed by one
tile at some round, with the 2:1-closure cascade as the first candidate.
**MEASURED, AND THAT GUESS WAS WRONG** -- see "One more shot at the residual"
below: the want set never diverges, and the force does, from step 128.

**THIS IS THE OUTCOME THE RUNG SAID WOULD BE THE MOST VALUABLE THING D1 COULD
FIND, AND IT SHOULD NOT BE PAPERED OVER.** Note what it does NOT say: D0's
claim is about SLOT ASSIGNMENT, and slot assignment is still pinned -- the
divergence enters upstream of it. Note also it is rare (1 in 6 at levels=3, and
`levels=2` showed the same 1-in-6 shape), which is why `--runs=2` cannot see it
and why every rung above was measured at 4 or more.

##### The cost, measured, since the rung asked for it

```
                       detslots=0   detslots=1
  levels=2                 319 ms      381 ms    +19.5%
  levels=3                 552 ms      612 ms    +10.8%
```

4096 steps, min of 4 runs, identical tile counts on both legs (so this is
overhead, not a different topology). The card page measured +0.7% / +3.4% on
the same build -- D0's original +28.7% / +0.9% no longer reproduces there
either. A Cd run is 69888 steps, so this is a real 10-20% on the wall clock of
every cylinder case.

##### Where this leaves D1-b and D1-c

`GATES_BY_PAGE` keeps `index-cylinder-amr.html` scored and RED. That is
correct: the rung's own prediction is that `detslots=1` is IDENTICAL, one run
refutes it, and the red cell is the refutation standing until the want-set
source is found. Do not add the page to a report-only list to make it green.

D1-c's capacity sweep (the direct confirmation of the retraction) is NOT
runnable yet at the precision it needs. `?detslots=1` cuts the variation from
four-distinct-in-four to one-in-six, which is a large improvement and not a
pin, and the 1.623-1.641 spread it was meant to collapse is the same order as
what remains. Finding the want-set source comes first.

##### One more shot at the residual: the want set is exonerated, the FORCE is not

Bisected with a small-payload probe rather than 20 MB snapshots -- per level,
every 128 steps, 8 runs of `?levels=3&detslots=1` from the now-fixed-point
reset: the active block list (sorted -> the want SET) and the card state.

```
  WANT SET   (which blocks are refined)   never diverges, 8 runs x 32 checkpoints
  CARD STATE (the force accumulators)     diverges at step 128, 2 attractors, 3/5
```

**So the criterion and the 2:1 cascade are exonerated**, which also matches
reading them: `completeQuads` writes the same four entries any wanting sibling
would write, and `balance` writes `1u` into a parent -- both idempotent, so
neither has a race that can change an OUTCOME however the threads interleave.
The earlier guess in D1-a that the want set differs upstream was wrong.

What is left is the force path, and it diverges EARLY and OFTEN -- step 128,
not step 4000, and 3/5 rather than 1/6. `amr_force1.wgsl` atomicAdds one
TRUNCATED i32 per WORKGROUP, and CLAUDE.md already records that regrouping the
slots regroups those partials. The open question this leaves is the sharp one:
with `DET_SLOTS` pinning the handout, what regroups them? Either the handout is
not as pinned as the serial loop implies, or the per-workgroup reduction itself
is not deterministic.

**THE SNAPSHOT SAYS THE HANDOUT STILL PERMUTES**: the diverging pair differed
in `pools[1].blockSlot` / `slotToBlock` / `parentSlot` by exactly 4 -- one quad
-- on the same block set. That is the thread to pull next.

*Two limits on the above, stated because the probe nearly hid both:*

- **The probe's "handout" channel was a duplicate, not a measurement.**
  `debugListActiveBlocks` returns block IDs already sorted, so hashing it
  unsorted gives the same string as sorting it, and it CANNOT see a
  permutation. It printed "never diverges" and that verdict is worth nothing.
  A real handout channel has to read `blockSlot` itself.
- **The first version of this probe had a vacuous card channel** --
  `debugReadCardState` returns an object, `Array.from` of it is `[]`, and the
  hash was constant. It reported "never diverges" over 8 runs at both 2048 and
  4096 steps, which read as a clean negative and was nothing at all. The
  liveness control (distinct values ACROSS checkpoints within one run) now
  prints next to every verdict, and it is what caught it.

**And this is where determinism work stops for now.** The point of it was to
make AMR comparable against the flat baseline; the flat baseline has no pool,
no slots and no per-workgroup partials, which is exactly why `dense-reference`
is bit-exact and why it is the thing AMR has to be scored against. But the
flat baseline is itself only as good as its agreement with the literature, and
it currently MISSES at Re=100 (Cd 1.951 against 1.35+/-0.15, St 0.1260 against
0.165+/-0.015) for the diffuse-band reason. Closing a +/-0.009 attractor spread
underneath a baseline that is 45% off its own reference buys little. The band
width is the higher-value target.

##### The re-baseline, and the sweep that says nothing else moved

Full `validate-all.js` after both fixes. Every boot, refusal, render,
invariant and analytic config PASS; the only red cells are the two CLAUDE.md
already records as the diffuse-band issue.

```
  config              Cd      St        was (CLAUDE.md)   note
  dense-reference     1.951   0.1260    1.951 / 0.1260    UNCHANGED -- the control.
                                                          main-cylinder.js was not
                                                          touched, and it is bit-exact
                                                          run to run, so this agreeing
                                                          is what says the sweep itself
                                                          did not move.
  amr-N2-diffuse      1.652   0.1484    1.642 / 0.1485    +0.010
  amr-N2-bounceback   1.356   0.1642
  amr-N3-diffuse      1.473   0.1570
  amr-N3-bounceback   1.365   0.1645
```

`amr-N2-diffuse` moving +0.010 is the size of the attractor spread CLAUDE.md
records for this page (+/-0.009), which is what an initial-condition change
should look like here and is NOT evidence either way about the fixes. Both red
cells are red for the reason they were before -- the diffuse band width, still
first order, still `K_EPS * dx_level`.

**These are the numbers to compare against from here, and the comparison is
still only valid within one configuration** until the want-set source above is
closed. `?detslots=1` was NOT adopted for the sweep: D1-c's decision is
deliberately left open, and pinning to a handout that is not yet a pin would
buy nothing.

#### D1-0 — the original statement, for reference

`tools/measure-determinism.js` hardcodes `window.__AMR` in six places and
asserts `getDetSlots()`. Before any of D1 can be scored it needs `--global=`,
defaulted from a page table the way `tools/measure-pool-peaks.js` does it --
`index-amr` and `index-reentry-amr` expose `window.__AMR`, the cylinder/TGV/
channel harnesses expose `window.__CYL`, and that does NOT follow from the page
name.

**AND ITS GATE TABLE IS THE CARD PAGE'S, NOT A UNIVERSAL ONE.** `CONFIGS`
predicts `levels=2 detslots=0 -> DIFFERS`; that is a measured property of
`index-amr.html`, and U5-4 already recorded it ceasing to hold there under quad
management. On a new page the predictions are unknown. So: report, do not gate,
until the page's own behaviour is measured -- then record it and gate. Measure
first, gate after, which is this project's order everywhere else.

*Gate:* the tool runs unchanged against `index-amr.html` and reproduces
`8ddd3ff2f84a697f` / `3d80fa737af6bf9e`. A refactor of the harness that moves
the card page's numbers is a refactor that broke something.

#### D1-a — the original statement, for reference

1. `const DET_SLOTS = urlParams.has('detslots') ? ... : 0` on
   `main-cylinder-amr.js`, pointing at `shaders/amr_manage.wgsl`'s DET_SLOTS
   header for why the serial loop is the right shape for a MEASUREMENT and the
   wrong shape for a default.
2. `DET_SLOTS` into `poolConstants` (it already carries `DIAG` for the same
   opt-in reason).
3. `getDetSlots` on `window.__CYL`.

Three edits, and after U7-5 they cover the default configuration completely,
because level 1 is quad-managed.

*Gate:* `measure-determinism --page=index-cylinder-amr.html`, `--runs=4`.
`detslots=1` IDENTICAL at levels 2 and 3. `detslots=0` must DIFFER **on at
least one level count**, or the instrument cannot see nondeterminism on this
page and no green row means anything -- the same guard the card page's
`levels=2` rung provides.

**TWO OUTCOMES TO EXPECT AND NOT PAPER OVER:**

- **`levels=2 detslots=0` may come back IDENTICAL.** U5-4 measured exactly that
  on the card page once the pool manager owned level 1: the racing free list
  produced the same assignment in 4 of 4 runs. If it happens here, `levels=3`
  carries the discrimination and that gets recorded, not relaxed away.
- **`detslots=1` may NOT come back IDENTICAL.** Then there is a second source
  of nondeterminism on this page that the card page does not have, and that is
  the most valuable thing D1 could find. It is information, not a bug to widen
  the comparison around -- D0's own stance on its own failure mode.

*Also measure, do not assume:* the cost. D0 measured the serial handout at
+28.7% (levels=2) and +0.9% (levels=3) on the card page at 4096 steps. A Cd run
here is 69888 steps. If it is expensive that is a fact to record next to the
gate, not a reason to skip it.

#### U7-6c — DONE (2026-09-22). The snapshot carries the root pool, and the round-trip suite turns out to be RED for another reason.

`debugSnapshotSave`/`Load` on the three pages that have them now carry
`pools[0]`'s field and velocity, at `formatVersion: 6`.

**SHARED, unlike the rest of those two functions.** CLAUDE.md's rule is that
they stay per-page because "they serialise whatever state that page owns" --
and the root pool is the one part that is NOT page state. Its shape comes from
`rootPoolSpec` and is identical everywhere, so `encodeRootCapture` /
`readRootCapture` / `restoreRootCapture` live in `amr2d-gpu.mjs`. Three copies
of a format is three chances to carry a different one.

**WHAT IS SAVED IS `f` AND `vel`, AND NOTHING ELSE.** The root's
blockSlot/slotToBlock are the IDENTITY, written once by `allocLevelPool` and
never touched -- there is no grant or release at a level with no parent -- so
they are shape, not state, and `checkRootPoolIdentity` already scores them
against that rule. The free list is allocated and left alone for the same
reason.

**THE VELOCITY IS STATE, AND THAT IS THE WHOLE RUNG.** It looks derived, since
the step kernel rewrites it from `f` every macro-step. But `dispatchMacroStep`
runs the REFINEMENT ROUND FIRST, reading "each level's own velocity field as
populated by the PREVIOUS macro-step" -- so the first refine round after a load
reads velocity the load must have put there. **This is the identical defect
D1-a found in `resetSim`, at a second site**: a buffer rewritten every step is
still state if something reads it before the first step. Worth stating as a
rule rather than as two coincidences.

Before this, a load called `seedRootFromDense()`, and `amr_mirror_root.wgsl`
writes `f_root` ONLY -- the root's velocity survived a load untouched. A
version-5 capture still loads, via that same fallback, and now says out loud
what it does not carry rather than looking clean.

*Gate:* U7-6b's own workaround, removed. That rung had to run level 0 LAST in
`tools/lib/render-levels.js` because a level-0 perturbation survived the
restore and poisoned every later row (one PASS, then three `ABSTAIN restore did
not return to baseline`). The loop is back in natural order and all rows pass
on both pages:

```
  index-amr.html            L0 PASS  L1 PASS  L2 PASS  L3 PASS
  index-cylinder-amr.html   L0 PASS  L1 PASS  L2 PASS
```

Level 0 runs FIRST there now, so the later rows passing IS the restore working.
Plus `make check`, `make test` (8 suites) and the full `validate-all.js` sweep.

##### `validate-snapshot-roundtrip.js` IS RED, AND IT WAS RED BEFORE THIS RUNG

> **CHASED AND CLOSED THE SAME DAY -- see "U7-6c-fix" below. The suite is 12 of
> 12.** Everything measured in this section is accurate; the CONCLUSION at the
> end of it ("U7-6a's record does not reproduce") was wrong. The cause was
> `resetSim` on `index-amr.html` not being a fixed point, which made the gate
> depend on what had run before it.

All four gated rows fail. They also fail with this rung reverted, at the
session's starting commit `2fe46e3`, and **at `6ffe437` -- U7-6a itself, the
commit whose entry records "12 of 12, both legs genuinely different
allocators, all four `stale` controls moving"**. Measured 2026-09-22 by
checking each commit out and running it:

```
  6ffe437  U7-6a          FAIL x4   ref e3ffeb0f4dfaa5b2 / roundtrip 82a654e0a81062e0 (L2,rootpool=0)
  2fe46e3  session start  FAIL x4   ref 01b703842ec46b05 / roundtrip fe009ff51bf1f856
  HEAD     with U7-6c     FAIL x4   ref 3c72db437a061930 / roundtrip b4836bdd4d1174f0
```

The hashes differ per commit because the code does; the VERDICT does not.

**IT IS NOT THIS RUNG'S, and the diff says so directly**: at every commit the
differing keys are `.pools[].parentSlot`, `.blockSlot`, `.slotToBlock`,
`.freeList` and one or two `.fB64`/`.velB64` -- and **never `.root.*`**, at
HEAD, where `root` exists. The root round-trips exactly. U7-6c also did not
change the failure: the `rootpool=1` rows read `32 .pools[].parentSlot + 1
.pools[].velB64` both before and after it, to the entry.

**IT IS NOT THE ENVIRONMENT EITHER.** Chrome has been 151.0.7922.173 since
2026-08-24 and the machine has 5 weeks of uptime, so nothing moved under the
code between U7-6a's green record and this red one. Both of the suite's
controls (`stale` moves the outcome, `refuse` throws) PASS at every commit
above, so the instrument is live and this is a real red, not a vacuous one.

That leaves U7-6a's record not reproducing. The most likely explanation is the
trap CLAUDE.md records having cost two full sweeps already -- `ensureServer`
reusing a server whose cwd is another checkout -- but that cannot be proved
retroactively and is not being asserted here. What IS established: **the
snapshot round trip is not reproducible today, has not been for at least as
long as the tool has existed, and the level-1/2 allocator state is where it
diverges.** Not fixed here, and deliberately not folded into this rung.

##### The D1-0 gate hashes moved, and that is the format, not the physics

`8ddd3ff2f84a697f` / `3d80fa737af6bf9e` were formatVersion 5. `fingerprint`
walks the whole snapshot object, so adding `root` moved both without moving any
physics: **`e0b1b22bbe47cbfb` / `4705ea2b06226c8b` at version 6**, IDENTICAL
over four runs each. Recorded in the tool beside the table, because a moved
hash there is exactly the shape of a regression and this one is not.

#### U7-6c-fix — the round-trip suite is GREEN (2026-09-22). `reset()` on the card page was not a fixed point either.

`tools/validate-snapshot-roundtrip.js` was red on all four gated rows, at HEAD,
at the session's start, and at `6ffe437` -- U7-6a itself, whose entry records
12 of 12. Chased and closed. **12 of 12.**

##### It was not the snapshot format. It was `resetSim`, for the third time.

The chase, in the order it actually went, because two of the three steps were
wrong and the wrong ones are the instructive part:

**1. `newlyActivated` -- WRONG.** It is read by the init-fill (`GHOST_ONLY==0`)
and nothing in the shaders clears it, which looked exactly like the
"derived buffer that is actually state" class D1-a and U7-6c had just found
twice. It is cleared, on the GPU, at the top of every refine round
(`makeRefineRound`'s `enc.clearBuffer`) -- deliberately GPU-recorded so it
interleaves with commands already in the encoder. Hypothesis dead in one grep.

**2. The live rAF loop -- ALSO WRONG, and it cost a fix that changed nothing.**
`open()` never paused the page, and `debugStepSync` only pauses on entry, so
the FIRST leg after a navigate -- the reference leg -- was exposed to stray
frames the round-trip leg was not. That is the identical confound
`measure-determinism.js` carried until D1-a, so it looked like the answer.
`setLive(false)` was added and the hashes did not move by one bit. The step
assertion added alongside it never tripped either: `getStep()` was already
exactly 2N, which says no frame ever stepped. **Both were kept** -- the
assertion is a real guard and the pause is correct -- but neither is the fix,
and saying so is the point.

**3. `resetSim`, and the probe that found it.** A probe comparing the two legs'
indirection came back IDENTICAL for `levels=2 rootpool=1` where the tool
reported 32 `parentSlot` entries moved. The difference between probe and tool
was the one thing left: **the probe navigated fresh for each leg; the tool runs
both legs in ONE page**, reference first, then `reset()` and the round-trip
leg. So the question became whether `reset()` puts the page back where a fresh
load starts. Measured directly:

```
  A   fresh load          -> reset -> snapshot    87aecb4b36ab9036
  B   after 2048 steps    -> reset -> snapshot    bb5106599df78f12
  differing:  .pools[].parentSlot x44, .pools[].velB64, .root.velB64, .velB64
```

**That is the gate's failure signature, key for key.** `reset()` on
`index-amr.html` never wrote the velocity buffers or cleared `parentSlotBuf` --
the same defect D1-a fixed on `main-cylinder-amr.js` and U7-6c fixed for the
root pool, at its third site. The refinement round runs BEFORE the step and
`macroStepCounter = 0`, so the first refine after every reset read velocity
nobody had written.

**ZERO IS THE RIGHT VALUE ON THIS PAGE, and that is not luck.** `index-amr.html`
starts at uniform rest -- `initF` is `feq(1, 0, 0, i)` everywhere -- so the
velocity the restored `f` represents IS `(0,0)`. `main-cylinder-amr.js` has a
perturbed freestream and had to derive its velocity from the same rng draw.
Same defect, two initial conditions, two correct answers.

*After:* `reset IS a fixed point`, and the suite is 12 of 12.

##### Why the fix moves nothing that a fresh page does

The reference leg's four hashes are BYTE-IDENTICAL before and after
(`3c72db437a061930`, `fef986e81e68eed9`, `28f5419eb0f1c14c`,
`676a9e59b9b483aa`). Only the round-trip leg moved, into agreement with it.
That is the shape a correct fix has here: WebGPU zero-initialises buffers, so a
FRESH load already had zero velocity and was always right -- only a reset after
history was wrong. `measure-determinism` navigates fresh per run and reads
`e0b1b22bbe47cbfb` at `levels=2 detslots=1` before and after, unchanged.

##### What this says about U7-6a's green record

It reproduces after all, and the explanation is ordinary. U7-6a's own run would
have been green whenever its reference leg happened to start from a page whose
history did not perturb the stale buffers -- and the buffers in question are
velocity and free-slot `parentSlot`, whose contents depend on how far the
previous config had run. The suite runs four configs in sequence through one
Chrome, so config 1 sees a different history from config 4. **A gate that
depends on what ran before it is not flaky, it is under-specified**, and the
under-specification was `reset()`.

So the earlier entry's "U7-6a's record does not reproduce" was the right
observation and the wrong conclusion: the record was real, the gate was simply
never measuring only what it claimed to.

#### The reset defect, closed structurally (2026-09-22)

Three instances in three days -- D1-a on the cylinder page, U7-6c on the root
pool, U7-6c-fix on the card page -- all of one false inference:

> **"WebGPU zero-initialises buffers, so this one needs no write."**
> True at ALLOCATION. False at RESET. A fresh buffer is zeros; a buffer being
> reset holds the last run's history.

The first response was to write the rule down. A rule relies on the next person
remembering it, and this project has three data points saying that does not
work. Replaced with two structures and an audit.

##### 1. One statement of a pool's initial state

`amr2d-gpu.mjs`'s **`writePoolInitialState(device, pool, { velFill })`**,
called by `allocLevelPool` AND by every page's `resetSim`. **Allocation is just
the first reset.** The two were hand-written copies of one list living far
apart in six files, and they had drifted in four different ways:

```
  blockSlot / slotToBlock   both wrote it
  freeList / freeCount      alloc wrote it for QUAD levels only; the per-block
                            level got an eager caller-side write on one page
  quadrant                  alloc only (it is a constant -- but "constant" was
                            a fact to remember rather than one to enforce)
  parentSlot                NEITHER wrote it -- the comment said zero-init was
                            enough, which was true where it was written
  finePoolVel               NEITHER wrote it -- and the refine round reads it
  wantBuf / newlyActivated / candRank / blockCriterion
                            neither; each rewritten-before-read WITHIN a round,
                            which is the per-buffer reasoning that failed
```

`f` stays the caller's, and that is the boundary the function draws: everything
inside it is a statement about the ALLOCATOR and is identical on every page;
`initF`/`initFPool` are the scenario's subject and differ.

`velFill` takes a constant `[ux, uy]` or a `Float32Array`. The array form is
not a convenience -- the cylinder's perturbation is per-cell and must come from
the SAME rng draw as the `f` it describes, so that page builds both in one pass
and hands the velocity in. A function that re-derived it would silently
disagree.

##### 2. A test that fails when someone adds a buffer and does not decide

`tools/test-pool-initial-state.js`, GPU-free, in `make test`. It runs
`allocLevelPool` against a recording stand-in for `GPUDevice`, enumerates every
buffer-valued key the pool declares, and requires each to be either WRITTEN by
`writePoolInitialState` or named in an explicit `NOT_RESET_STATE` list with a
reason. Two entries are on that list today (`finePoolF_a`, `finePoolF_b`), and
a second test fails if a name on it stops being a buffer, so the exemptions
cannot go stale either.

**Mutation-checked, against the exact buffer that caused the bug.** Deleting
the `parentSlotBuf` write turns three of the four cases red and names
`parentSlotBuf` in the failure text. A reviewer cannot be relied on to notice a
buffer MISSING from a list; this makes the omission the thing that fails.

##### 3. `tools/validate-reset-fixed-point.js` -- the property itself, on the GPU

The abstraction covers the pools. This covers whatever a page adds next to
them, which is where the dense `velBuf` instance lived.

```
  A   navigate, pause, reset, snapshot
  B   step N, reset, snapshot
  require A == B
```

Five configurations -- card, cylinder and reentry, at two level counts --
because every defect was in per-level reset code. It reports the SHAPE of the
difference (`.pools[].parentSlot x44`), which names the offending buffer
directly. A control requires that stepping moved the state at all, since A == B
is also what a comparison that cannot see anything produces.

*All five PASS.* `reentry` passes without ever having been fixed by hand -- it
inherited the shared call, which is the point.

##### The audit, and what it found beyond the three known sites

Five `resetSim` implementations. Beyond the pools, the DENSE `velBuf` was
unwritten on **all five** pages; it is now written on all five and goes with
the dense grid at U7-6f. `main-tgv-amr.js` and `main-channel-amr.js` had never
been touched by any of the three fixes -- their pools hold zero active tiles so
nothing bit, which is exactly the kind of latency that makes an audit worth
more than a rule.

*Gate:* `make check`; `make test` (9 suites, the new one included);
`validate-reset-fixed-point` 5/5; `validate-snapshot-roundtrip` **12 of 12**;
`measure-determinism` on the card page **unchanged** at `e0b1b22bbe47cbfb` /
`4705ea2b06226c8b`, which is what says the refactor is behaviour-preserving;
and the full `validate-all.js` sweep unchanged from baseline -- `dense-reference`
exactly 1.951/0.1260 and `amr-N2-diffuse` 1.652/0.1485, the two known
diffuse-band cells and nothing else.

#### U7-6d — DONE (2026-09-22). The dense L0 is provably inert, and `?densel0=0` proves it.

**THE RUNG'S OWN ITEMISATION WAS STALE, and the audit is most of the result.**
It named four consumers to port. Measured:

```
  readConservedTotals        ALREADY ROOT-CAPABLE. It is parameterised by
                             `cellIndex` and `decode`, and main-amr.js has
                             called it on pools[0].finePoolF_a since U4-3.
  readPoolUniformDeviation   DOES NOT EXIST. No such symbol anywhere in the
                             repo. Carried in the plan from an earlier sketch.
  the dense criterion read   Not a port. Under a root pool `denseCriterion` is
                             REDIRECTED into `rootGpu.denseCritBuf`, a scratch
                             buffer that exists so the shared refine round has
                             somewhere harmless to land. Nothing reads it.
  the dense force            `frcPL` dispatches only when N_LEVELS === 1, which
                             every AMR page REFUSES at init (`refuse-levels-1`).
                             Dead in every reachable configuration.
```

So the work was not porting consumers. **U7-6b and U7-6c had already ported the
last two that mattered** -- the renderer and the snapshot -- and what remained
was that the dense L0 is still fully STEPPED and RESTRICTED every macro-step
(U5-3 kept both running deliberately, so the two representations stayed
byte-identical), feeding nothing.

##### `?densel0=0`

Default 1, byte-identical to not having the flag. 0 stops encoding the dense
step, the dense restriction and the dense criterion once a root pool exists.
Three call sites per page, five pages, plus `getDenseL0()` on each debug
surface. It goes away at U7-6f along with the path it switches.

*Gate -- the claim is bit-equality of everything that is not the dense grid:*

```
  card levels=2        densel0=1 07dbf161762d2265  densel0=0 07dbf161762d2265
  card levels=3        densel0=1 b47c77656d4a854c  densel0=0 b47c77656d4a854c
  cylinder levels=2    densel0=1 f7102f1332c6eafd  densel0=0 f7102f1332c6eafd
  cylinder levels=3    densel0=1 4668084f39fbdb31  densel0=0 4668084f39fbdb31
```

2048 steps, `.fB64`/`.velB64` (the dense arrays, expected to freeze) excluded
by name; `.root.*` and `.pools[]` are NOT excluded, because they are the claim.
**Two controls, both active:** the probe asserts `getDenseL0()` matches the URL
(a flag that did not take reads as a clean pass -- this project's most common
false green), and requires 64 further steps to MOVE the compared state, since
every row is an equality and a filter that excluded everything meaningful would
pass them all.

*And the end-to-end version, which is stronger than the bit comparison:* the
cylinder physics suite under `--extra=densel0=0`, **69888 steps a case**, is
identical to the recorded baseline -- `amr-N2-diffuse` Cd 1.652 / St 0.1485,
`amr-N3-diffuse` Cd 1.473 / St 0.1570. Not "within tolerance": the same
numbers.

##### The cost it removes, measured badly on purpose to say so

```
  levels=2   1634 -> 1420 ms      (4096 steps, min of 2)
  levels=3   3029 -> 1308 ms
```

**DO NOT QUOTE THESE AS A SPEEDUP.** They were taken on a contended GPU --
earlier the same day the same configurations read 380 ms and 610 ms, so the
machine was roughly 4x loaded and the ratio is not stable under that. The
DIRECTION is certain (a whole dense L0 step and restriction per macro-step stop
being encoded) and the magnitude is not measured. `plans/perf-characterization.md`
is the protocol for doing it properly, and this was not that.

Note also that `measure-determinism`'s hashes differ under `?densel0=0`
(`c2f03ff0787bdd59` against `e0b1b22bbe47cbfb`) and that is CORRECT: its
fingerprint walks the whole snapshot including the now-frozen dense arrays.
The probe above is the comparison that excludes them.

##### What this leaves for U7-6f

The dense grid is now demonstrably a spectator on every page. What still
touches it: `debugSnapshotSave` (which also carries the root since U7-6c), the
dev page's root COMPARATORS -- whose entire job is scoring the two
representations against each other, so they need it stepping and are why
`?densel0=1` stays the default for now -- and `tools/lib/field-reconstruct.js`
/ `dense-to-amr.js`, which are U7-6e.

*Gate, all green:* `make check`; `make test` (9 suites);
`validate-reset-fixed-point` 5/5; `validate-snapshot-roundtrip` 12 of 12; the
full default sweep unchanged from baseline.

#### D1-b — SKIPPED, deliberately, 2026-09-18. The decision the rung asked to be made explicitly.

The rung says: "If D1-b is judged not worth doing on a path scheduled for
deletion, that is a defensible call -- but it has to be made explicitly and
written down." Made, and here is the reasoning.

**D1-b is not worth doing, and D1-a is why -- not the deletion schedule.**
The rung's argument FOR it was "an unpinned control is not a control": pin the
dense manager so U7-6's before/after A/B has a pinned baseline on both legs.
D1-a measured that `?detslots=1` DOES NOT PIN THIS PAGE. The force diverges at
step 128 into two attractors with the handout serial and the want set
identical. So four new pipelines on `amr_manage.wgsl` would buy an
*equally unpinned* control, on a file the next rung deletes. The premise the
rung rested on did not survive the rung before it.

**What replaces it as U7-6's gate**, because "inherits the +/-0.009" is a real
consequence and needs an answer rather than a shrug:

- `index-cylinder.html` (`dense-reference`) is NOT touched by U7-6 -- it has no
  pool, no slots and no per-workgroup partials, and it reproduces BIT-EXACTLY
  run to run. It is a true bit-exact control and it is already in the sweep.
- The four analytic AMR configs (`channel-*-amr-N2`, `tgv-amr-N2/N3`) were
  bit-identical across B1's rescale flip and are the solver gate. Their known
  limitation -- they hold ZERO active tiles, so they do not exercise the seam
  -- is unchanged by U7-6 and is why they are not the whole gate.
- `validate-snapshot-roundtrip.js` is a BIT-IDENTICAL gate with three live
  controls, and U7-6 changes the snapshot format, so it is directly on point.
- `validate-amr-invariants.js`'s seven gates are structural and do not care
  about Cd's spread at all.

So U7-6's Cd/St cells are read as BAND membership against the literature, which
is what they were always for, and the bit-exact claims come from the four
instruments above. That is a better gate than a pinned-Cd A/B would have been,
and it does not depend on a pin that does not exist.

#### D1-c — BLOCKED on the force, not deferred. 2026-09-18.

Its three items were: Cd/St on both legs at `?detslots=1`; re-run the capacity
sweep under it, predicting the 1.623-1.641 spread collapses to the repeat
floor; and decide whether `validate-all.js` adopts the flag.

**Item 2 cannot be run at the precision it needs, and item 3 answers itself.**
The prediction was that pinning the handout collapses the spread. D1-a measured
the force diverging at step 128 WITH the handout pinned, so the flag does not
deliver the floor the prediction is stated against -- a sweep run now would be
measuring the residual, not the capacity, and would read exactly like the
result it is supposed to refute.

So: **`validate-all.js` does NOT adopt `?detslots=1`.** It costs +10-20% on
every cylinder case (measured, D1-a) and buys a pin that is not a pin. Revisit
when the force source is closed, at which point item 2 becomes runnable and is
worth running.

The retraction ("pool capacity is inert; the cylinder page's allocator races")
therefore stands on the CARD page's evidence alone, as it did before. That is
where it was established and it has not weakened; it is only the cylinder-page
confirmation that is still owed.

---

### U7-6 — staged, because it is not one rung

The rung as written is "delete the dense path", and its own text is honest that
the deletion is small only because U7-0..U7-3 made each thing named once. But
the deletion cannot happen until every CONSUMER of the dense L0 is ported, and
the survey says that is the larger half:

```
  f_a / f_b references     37 / 30 / 17 / 14 / 30  across the five AMR pages
  velBuf references        16 / 14 /  8 /  9 / 12
  ROOT_POOL/ROOT_MANAGED   31 / 21 / 14 / 14 / 21  guards to collapse
```

and the renderer still reads the dense `velBuf` at binding 0 for level 0 --
`makeRenderBindGroup` takes it as a parameter, and `amr_render.wgsl` walks
`m = 1 .. N_POOL_LEVELS` with level 0 handled by separate flat-array
accessors. U6 made levels 1..4 uniform and deliberately left the root alone.

Sub-rungs, in dependency order. Every one of them is "port a consumer"; only
the last deletes anything:

```
  U7-6b  the RENDERER's level 0 becomes a pool level, like 1..4 already are
         -- DONE 2026-09-18
  U7-6c  debugSnapshotSave/Load carry the root pool  -- DONE 2026-09-22
         (the dense arrays go at U7-6f, with the buffers themselves)
  U7-6d  the remaining host consumers  -- DONE 2026-09-22, and the
         itemisation was stale: two of the four were already done or never
         existed. The real content was that the dense L0 still STEPS.
  U7-6e  tools/lib/field-reconstruct.js and tools/lib/dense-to-amr.js, which
         are inverses by construction and have a round-trip test already
  U7-6f  THE DELETION: the shaders, the f_a/f_b/velBuf buffers and their bind
         groups, `?rootpool=`, and the dense-only checkers retired in the same
         commit as their subjects
```

**The ordering is not arbitrary and the last one is last for a reason.** While
`?rootpool=0` still exists, every sub-rung above can be A/B'd against the path
it is replacing IN ONE BUILD. After U7-6f there is nothing to compare against
-- which is the same argument that put D1 ahead of U7-6, and it applies with
more force to the consumers than it did to the allocator.

#### U7-6b — DONE (2026-09-18). The renderer's level 0, and the gate row that never existed.

`amr_render.wgsl` binding 0 is still "level 0's velocity"; what it CONTAINS is
now chosen by a `ROOT_IS_POOL` override -- the dense L0 grid addressed by
`cellIndex`'s block8 layout, or the ROOT POOL addressed by a new
`rootCellIndex` through binding 12's indirection.

**LEVEL 0 DID NOT JOIN `poolVel`'s LADDER, and that was the one real trap.**
`ghostDepthAtLevel(0)` is 0 -- a ring holds a parent interface and the root has
no parent -- so the root tile is `2*RB` cells square where every other level's
is `2*RB + 2*GHOST`, with no GHOST offset on the local coordinate. Writing it
as `poolVel(0, ...)` would have applied level>=1's geometry to it silently.
The root is also always FULL, which is why level 0 stays the BASE CASE of the
walk (`coarseOmegaCell`) rather than becoming one more iteration of it.

`renBGL` gained one binding, in ONE place, because U7-0 made the layouts
shared. Five pages needed one constant each.

##### The gate: one build, two addressings, same picture

Both representations hold the same field while `?rootpool=1` keeps the dense
grid stepped, so `?rootIsPool=0|1` points the renderer at one state through
each addressing IN ONE BUILD. It moves the BUFFER and the arithmetic together
(`renderRootIsPool` is the single statement of that rule; switching one without
the other draws garbage, not the other representation).

```
  index-amr.html         874a14e804705e8e   IDENTICAL on both legs,
                                            and identical to the pre-U7-6b build
```

That card-page hash is the trustworthy one: the page is deterministic, and it
has read `874a14e804705e8e` in every run of this rung, before and after.

**THE CYLINDER PAGE'S BASELINE HASH IS NOT A CROSS-RUN COMPARATOR, and this
rung nearly drew a wrong conclusion from treating it as one.** It read
`ff93b070db8ddc9d` before the change, `cdbcb1e9d5efb375` after -- stable across
three consecutive runs, which looked like a real move. It is the attractor:
both values recur on both legs, and three runs is three draws from the dominant
one. D1-a measured that residue directly; the mistake here was forgetting it
applied to pictures too. Score this page's render against the CARD page's
invariance, or against a same-build A/B, never against a hash from another run.

##### The level-0 row, and the instrument bug it immediately found

`tools/lib/render-levels.js` walked `m = 1 .. nLevels-1` and had never scored
level 0 -- it could not, before this rung. It does now, and the first thing it
reported was FAIL on BOTH legs:

```
  level 0  FAIL  picture UNCHANGED after overwriting 102400 cells
```

102400 = 256 blocks x 400, and 400 is `(2*RB + 2*GHOST)^2` -- level>=1's
geometry. The ringless root slot holds 256. `debugPerturbLevelVel` reached for
the page's module-level `NCELLS1`, so it wrote 400 cells per root block into a
buffer holding 256, and **an oversized `writeBuffer` is a validation error that
discards the write** -- so the picture did not change and the gate read "this
level has no path into the renderer". A correct-looking red cell, from the
instrument, about a path that was fine.

Fixed by giving each pool its own `cellsPerSlot` and having the perturb read
it. That is the same shape as this rung's own trap, one layer up: the root's
geometry is not level 1's, and anything that assumes otherwise is wrong exactly
at the root.

*Gate, all green:*
```
  index-amr.html            L1 PASS  L2 PASS  L3 PASS  L0 PASS
  index-cylinder-amr.html   L1 PASS  L2 PASS           L0 PASS
  ?rootIsPool=0 control     L0 FAIL on both  <- REQUIRED. The renderer reads
                            the dense grid there, so perturbing the root pool
                            must NOT reach the picture. Without this row the
                            L0 PASS above is consistent with the flag doing
                            nothing.
```
plus `make check`, `make test` (8 suites), and the full `validate-all.js`
sweep: every boot, refusal, invariant, render and analytic config PASS, with
only `dense-reference` and `amr-N2-diffuse` red for the diffuse-band reason.

##### One thing this rung uncovered and did NOT fix

Running level 0 FIRST produced one PASS and three `ABSTAIN  restore did not
return to baseline`. The restore is `debugSnapshotLoad`, and it carries each
POOL level's velocity but not the ROOT pool's -- so a level-0 perturbation
survives it and poisons every later row. **The ABSTAIN was correct and the gate
was right to refuse to score**, which is the behaviour that makes it worth
having. Level 0 now runs LAST so the rows stay independent; U7-6c is where the
snapshot gains the root pool, and that ordering comment goes with it.

#### D1-b — the original statement, for reference

Only reachable under `?rootpool=0`, which after U7-5 exists solely as the A/B
escape -- and an unpinned control is not a control. Four pipelines and their
encoding:

1. `DET_SLOTS` into `manageConstants`.
2. `manageScanGrantPL` / `manageScanReleasePL` (one entry point,
   `SCAN_RELEASE` 0/1) and `manageLinkCoarsenPL` / `manageLinkRefinePL`.
3. Encode them in `refinePasses.denseCoarsen` and `denseRefine`.

**THE ORDER IS LOAD-BEARING AND main-amr.js RECORDS WHY.** The release scan
runs BEFORE coarsen, because it reads `blockSlot` as coarsen finds it -- and
after the previous round's refine link, which is where `blockSlot` was last
settled. The grant scan runs AFTER coarsen, so blocks released this round are
visible as candidates in the same round. Copy the reasoning with the code; a
mis-ordered pass here is a physics difference that still produces a field.

**AND D0's "KEPT ON EVIDENCE, NOT ON ARGUMENT" TRAVELS WITH THEM.** The link
passes looked redundant once `scanCandidates` owned the ranking, were removed,
and three runs later one configuration had diverged. The mechanism is still not
understood. Do not drop them here on the same reasoning that was already wrong
once.

*Gate:* `detslots=1` IDENTICAL over 4 runs under `?rootpool=0` as well.

#### D1-c — the original statement, for reference

1. Cd/St on both legs at `?detslots=1`, with repeats. These become the
   comparable numbers.
2. **Re-run the capacity sweep with `?detslots=1`.** Prediction: the
   1.623-1.641 spread collapses to the repeat floor. That is the direct
   confirmation of the retraction on the page where it was observed -- so far
   capacity has only been shown inert on the CARD page, and this closes it.
3. Decide whether `validate-all.js`'s cylinder configs adopt `?detslots=1`.

**THE DECISION IN 3 IS A REAL ONE AND SHOULD NOT BE PRE-EMPTED. Pinning removes
VARIANCE, NOT BIAS.** The deterministic handout selects ONE attractor -- the
serial, dispatch-order one -- and its Cd is one sample from the set, possibly
~0.009 from the set's centre. So:

```
  build vs build        strictly better pinned: both sides on the same
                        attractor, and the +/-0.001 floor becomes real
  against literature    it is ONE SAMPLE, not a mean. Adopting it moves the
                        recorded Cd once, and the new number is not more
                        "correct" -- only reproducible
```

The alternative worth naming rather than dismissing: run N attractors and
report the SPREAD, which measures the uncertainty instead of hiding it. Almost
certainly too expensive for the default sweep at 69888 steps a case; possibly
right for a once-per-release characterisation.

#### Sequencing, and why this order

```
  D1-0   the tool can address the page          (nothing is scoreable before it)
  D1-a   pool manager  -- the DEFAULT path      (3 edits, covers what ships)
  D1-b   dense manager -- the CONTROL           (only while ?rootpool=0 exists)
  D1-c   re-measure, and decide about the gate
```

**ALL OF IT BEFORE U7-6.** That rung deletes `amr_manage.wgsl`, which is D1-b's
entire subject, and `?rootpool=0`, which is the control D1-c's comparison uses.
After U7-6 there is no dense path to pin and nothing to A/B against. This is
the same sequencing argument that moved U7-6a ahead of U7-5, and that one held.

If D1-b is judged not worth doing on a path scheduled for deletion, that is a
defensible call -- but it has to be made explicitly and written down, because
the consequence is that U7-6's own before/after comparison runs against an
unpinned baseline and inherits the +/-0.009.

#### What this does NOT cover

`main-reentry-amr.js`, `main-tgv-amr.js` and `main-channel-amr.js` also lack
DET_SLOTS. They carry no Cd/St gate, and the analytic configs hold zero active
tiles so their allocator never runs -- the nondeterminism has nothing to bite.
Port it there when something needs it, not before.

#### U7-6 — delete the dense path

`amr_step.wgsl`, `amr_criterion.wgsl`, `amr_manage.wgsl`, `amr_force.wgsl`,
`amr_interp_dense_parent.wgsl`, `amr_average_f2c.wgsl`,
`common_interp_parent_dense.wgsl`, `common_avg_parent_dense.wgsl`, the `f_a` /
`f_b` / `velBuf` dense buffers and their bind groups.

By this point they are named in ONE place each, which is what makes this rung
small. It is also the rung that pays back U7-0 through U7-3: a deletion spread
over five copies is five chances to leave one behind.

**Retire the dense-only checkers in the same commit as their subject.** That is
B3-5's lesson (`debugCheckTileOrigins` went with the buffers it scored) and its
converse is how this project collected three vacuous gates.

Host and tool tail, all of which currently assume a dense L0 on the AMR side
and must be ported or explicitly scoped out:

- `tools/lib/field-reconstruct.js` — the root becomes one more level of the
  same quadtree walk, which should make it *simpler*.
- `tools/lib/dense-to-amr.js` — the injector's target gains root tiles. Its
  round-trip test against `field-reconstruct` is the gate and already exists.
- `tools/validate-amr-vs-dense.js`, `tools/validate-divergence.js` — their
  *dense reference* side is `index-cylinder.html` and is unaffected; only the
  AMR side's reconstruction changes.
- `debugSnapshotSave` / `debugSnapshotLoad` — the snapshot format gains the
  root pool and loses the dense arrays. Old snapshots stop loading; say so
  loudly rather than mis-reading them.
- `amr2d-gpu.mjs`'s `readConservedTotals`, `readPoolUniformDeviation`.

**Gate:** `make check`, then the full `validate-all.js` sweep. Re-baseline
against the numbers in CLAUDE.md, not against a memory of them — and remember
the default sweep is **not** all-green on `main` today (`dense-reference` and
`amr-N2-diffuse` fail at Re=100, which is the diffuse-band-width issue and not
a regression).

#### What deliberately stays per-page

Sharing is not the goal; ONE STATEMENT OF EACH RULE is. Three things are
genuinely different per page and sharing them would be sharing a coincidence:

- **`resetSim`** — four variants, because the initial field is the page's
  subject (a card at rest, a cylinder in crossflow, a Taylor-Green vortex, a
  channel profile).
- **`debugSnapshotSave` / `debugSnapshotLoad`** — three variants, and only
  three pages have them at all. They serialise whatever state that page owns.
- **The root COMPARATORS** — `debugCheckRoot*`, `compareRootToDense`, the
  mirror checker. Roughly 500 lines that exist to score one representation
  against another, and the dev page is where that scoring happens. Copying
  them to five pages would be copying a checker, which is how three gates went
  vacuous here before.

The test for whether something belongs in the shared half: **does every page
have to agree about it for the solver to be correct?** A bind group layout,
yes. A pipeline set, yes. The dispatch order, emphatically yes. An initial
condition, no.

---

## 6. Traps, carried forward

**The dev-server trap applies to every stage here.** `ensureServer` reuses
whatever answers on the port. With the main checkout, `3d-trt` and this
worktree all live, the default `https://localhost:4444` is routinely somebody
else's tree. A full sweep and a "pristine A/B" have both already run green
against the wrong checkout and had to be discarded. Prove the tree with a token
you added, then pin both ports.

**A BINDING ADDED TO A SHARED SHADER MUST BE MIRRORED INTO FIVE BIND GROUP
LAYOUTS, AND THIS TRAP CAUGHT U4 EVEN THOUGH CLAUDE.md DOCUMENTS IT.** U4-1
added binding 3 to `amr_criterion_pool.wgsl` and U4-2 added binding 6 to
`amr_force1.wgsl`; both were mirrored into `main-amr.js` and nowhere else, and
every other AMR page stopped booting with `Binding doesn't exist in
[BindGroupLayoutInternal "force1BGL"]`. That is 238e48c's shape with the
direction reversed. The boot smoke found it in seconds -- which is the whole
reason CLAUDE.md added it -- but only because it was RUN. Run it after any
shader binding change, before believing the page you happen to be testing is
the only one that matters:

    node tools/validate-all.js --configs=index-boot,amr-dev-boot,reentry-boot,reentry-amr-boot

Note that list does NOT include the cylinder, channel or TGV pages, so it is
necessary and not sufficient; those three have their own boot configs
(`cylinder-amr-boot-N3`, `channel-amr-boot-N3`, `tgv-amr-boot-N3`) and the
default sweep runs them.

**AND DO NOT RUN `validate-all.js` WITHOUT PINNING ITS PORTS.** It takes no
`--help`; an unrecognised flag makes it run the FULL default sweep on
`https://localhost:4444` and debug port 9333, which starts a second dev server
and a second WebGPU Chrome. Killing the node process leaves both behind, and an
orphaned Chrome holding a GPU context is the exact condition CLAUDE.md records
as having already produced one confidently-wrong reading. Clean up by PROFILE
DIR, never by process name -- `make chrome-clean` kills every debug Chrome
under `/tmp/vpm-chrome-profile`, including the one you are driving.

**TWO ROUTES ARE ONLY INDEPENDENT IF THEY DO NOT SHARE A PREMISE.** U2 scored
a shader against a host function, neither consulting the other, and both wrote
`gy*W + gx` over a dense grid that is 8x8 block-major. The check was green over
a mirror that fetched the wrong cell 98.4% of the time. Authorship is not
independence. When the shared premise is a LAYOUT or a CONVENTION, write it
down somewhere both routes can be scored against, and prefer a third route that
is validated by DATA (`field-reconstruct.js`'s `rawIndex` reads real snapshots)
over a third formula. And prefer a check that COMPOSES the two sides to one
that compares them: a bijection check and an agreement check were both green
over this bug; "does this root cell land on the dense cell at its own
coordinates" was not.

**A WORD-EQUALITY GATE CANNOT TELL A BIT-IDENTICAL SOLUTION FROM TWO IDENTICAL
NaN FIELDS.** `?benchSkip=interp,avg` drives 589824/589824 dense populations
non-finite, and the mirror checker then reports `0/589824` differing with
`maxAbs 0.0e+0` — its most emphatic possible pass. Score field health FIRST and
refuse to report a row whose field is not finite. This is the same lesson as
the `field` column below, arriving through a different door: there, a gate read
something other than the fluid; here, a gate read the fluid and the fluid was
NaN.

**CHROME CACHES `https.py`'s HEADER-LESS RESPONSES.** A `Page.navigate` that
only changes the query string can silently run the previous build's JS and
WGSL. One U3 measurement concluded a fix "did nothing" on exactly this.
`Network.setCacheDisabled({cacheDisabled: true})` on every CDP session, and if a
change appears inert, check that before believing it.

**`debugStepSync(n)` ADVANCES IN WHOLE FRAMES OF 64.** Its loop is
`for (k = 0; k < n; k += STEPS_PER_FRAME)` with `STEPS_PER_FRAME = 64`, so
`debugStepSync(1)` and `debugStepSync(2)` return the identical field. Any "after
N macro-steps" in this file with N < 64 means 64. The smallest advance
available is therefore 64 steps, which is longer than it takes a single-cell
difference to reach a quarter of the domain — so localising a residual in SPACE
is not available through this API, and structural attribution (which pipeline
BINDS the buffer as writable) is the instrument that works.

**A green gate is not necessarily reading what you think.** The invariant
sweep's `field` column is `debugReadCardState` — the rigid body's five numbers,
not the fluid — and it cannot fail: `safeFixed` maps NaN to 0, every moment
divides by `max(rho, 1e-6)`, and the integrated velocity is clamped. It has
reported OK through 8192 steps over a run whose level-2 pool was entirely NaN.
`debugCheckFieldFinite` is the one that reads the fluid.

**An admissibility rule derived for one quantity does not transfer to another
just because it is evaluated at the same cell.** Reusing it reads as
conservative and is a silent zero. This will come up again in section 2's
inbox/outbox split, where the rule for *populations* and the rule for
*velocity* at a ring cell are genuinely different.

**Every "cannot happen" branch should be reachable by a counter even when it is
not reachable by the physics.** Containment and silence are separable: a
substitution that leaves no trace makes the robustness that keeps the page
running the same mechanism that keeps the failure from being reported.

**Check the storage-buffer count before designing around a new buffer.** The
ceiling is 16 per stage on the target hardware and the failure mode is a page
that does not boot. Section 4's design deliberately adds no new buffer: the
mailbox reuses the ring, and the accumulation is done by advection rather than
by atomics.

---

## 7. Open questions this plan does not answer

1. ~~**The diagonal claimant at a convex corner.**~~ **ANSWERED 2026-09-17 —
   see 2.6a.** Derived for D2Q9 and scored by a falsifiable ledger over nine
   shapes: the claim is decided by the coverage of the single cell `Q - e_i`,
   so there is no corner case at all. What remains of this item is the GPU
   half — scoring the shipped kernel against `transferLedger` — and the
   corner-free seam geometry is still the control worth having, now as a
   consistency instrument rather than a conservation one.
2. **Whether the ring's tangential directions need a role at all.** Section
   2.5's table covers inward and outward; a direction whose target is another
   ring cell at the same depth is neither, and whether it must be written,
   zeroed, or left alone is not settled here.
3. **What U3 costs on the phone.** The root step is the single hottest kernel
   and it gains an indirection. `IDENTITY_SLOTS` is the escape hatch; whether
   it is needed is a measurement, not a prediction.
4. **Whether the root should be allowed to be non-full later.** This plan
   fixes it full, per the stated constraint. The machinery would permit a
   coarser root, which would decouple domain size from base memory cost — see
   section 8, which is the reason that matters.
5. **Which sponge policy to adopt** (section 8.5). The three are a genuine
   trade, not a right answer, and the choice decides whether section 8.2's cost
   claim survives a coarsening root. The recommendation is "root + j levels",
   but `j` is a measurement and the reflection cost of a narrower ramp has
   never been measured here at all.

---

## 8. What this is ultimately for: field scaling

Not part of this plan's stages. It is the question the plan makes *askable*,
and it is worth writing down now so the substrate is not designed in a way that
forecloses it.

### 8.1 The question

> For a fixed body-refinement resolution: what does increasing the simulated
> fluid volume do to the answer, how much volume does fidelity actually
> require, and how can that volume be reduced to save compute without losing
> fidelity?

These are three knobs that this project has never been able to turn
independently, because **today "more domain" and "more body resolution" are the
same knob.** `W = H = 2^resLog2` sets the dense L0 extent, and the body's
resolution is `W * 2^-(N-1)`. Wanting twice the domain at the same body
resolution means doubling `W` (4x the L0 cells, quadratic) *and* adding a level
to claw the body resolution back. So the cost of asking the question has always
been superlinear, and the question has never been asked.

### 8.2 Why uniform levels reframes it

With a root that is a level like any other, the three knobs separate cleanly:

| knob | expressed as | cost |
|---|---|---|
| body resolution | the **finest** level's `dx` | fixed by the refined region's tile count, not by the domain |
| domain volume | how many levels sit **above** the body's | **one extra level doubles the extent in each axis** |
| domain *shape* | which tiles exist at each level | anisotropic, per tile |

The middle row is the prize. Prepending a coarser root keeps the tile count
roughly constant while each tile covers 4x the area, so **domain extent becomes
exponential in level count while cost grows roughly linearly** — the added cost
per level is the 2:1 closure shell around the refined region, which scales with
its perimeter, not its area. That converts "how much volume does fidelity need"
from an experiment nobody can afford into a ladder, the same shape of
instrument as B7's `?kEps=` band ladder.

The third row matters too and is easy to overlook: a shedding wake needs length
downstream, not width. Today the domain is literally square by construction. A
uniform level treatment makes domain *shape* a refinement question rather than
a `W`/`H` question, which is one of the cheapest available ways to answer
"reduce volume without reducing fidelity".

### 8.3 Two hard prerequisites, and one is not optional

**(a) B6 must land first, and this is a mechanism, not a preference.** The
present interface is a per-step momentum source sitting on the seam, and its
magnitude scales with seam *area*. Every coarse level prepended to the root
adds a 2:1 closure shell — which is more seam. So a domain-extent ladder run on
today's interface would move for two reasons at once and could not attribute
the movement to either. **The ladder measures the seam until the seam stops
being a source.**

**(b) The coarse end of the level chain runs out of tau, and the ceiling has to
be measured rather than derived.** Viscosity is fixed physically, so
`tau_m = 1/2 + 2^m * (tau_0 - 1/2)`: going *coarser* halves the distance to
`tau = 1/2` at every level. From `tau_0 = 0.8` the chain reads 0.8, 0.65,
0.575, 0.5375, 0.51875 — so there is a hard, fairly shallow ceiling on how many
coarse levels can be prepended before the root is at the BGK stability limit,
and it depends on `tau_0` and on the local velocity, not on a clean inequality.

**Find it by running until the field goes to NaN and reporting where**, not by
arguing it from a textbook bound. `debugCheckFieldFinite` is the instrument
that already exists for exactly this, and its rho band of [0.1, 10] catches the
excursion a few hundred steps before the NaN. There is also a known singular
set to stay clear of while sweeping — level *m* is singular at
`tau_0 = 1/2 + 2^-(m+1)` under the default post-collision transfer — so the
ladder must record which `tau_0` values it skipped and why.

### 8.4 The shape of the eventual instrument

Nothing here is a commitment; it is what the plan is trying not to foreclose.

- **Metric.** The cylinder harness, because it is the only configuration with
  literature values to converge *toward* — `benchmarks/cylinder.json`, Cd and
  St. The falling card has no external reference, so it can only show
  self-consistency.
- **Ladder.** Fix the finest level's `dx`; sweep the number of levels above it;
  report Cd/St against domain extent in body diameters. The answer to "how much
  volume" is the rung where the metric stops moving by more than the
  reproducibility floor — which is **+-0.001 at N=2 and +-0.002 at N=3**, and
  has to be re-measured on whatever depth the ladder actually reaches, because
  those floors are per-config and this ladder changes the config every rung.
- **Controls.** The rung that answers the question is the one where *only* the
  extent changed. Blockage ratio, sponge width and sponge placement all move
  with the extent unless they are pinned, and a ladder that lets them drift is
  measuring the sponge.
- **The reduction question** is then a second ladder over far-field treatment
  at fixed extent: for a target error, what extent does each treatment need?
  That is the number that says whether a better outflow condition is worth
  writing.

### 8.5 The sponge is the term that can break 8.2, and the edge probably has to be FORCE-REFINED

Section 8.2 claims domain extent grows exponentially in level count while cost
grows roughly linearly. That is true of the *interior* — the 2:1 closure shell
around the body scales with its perimeter, which does not move when the domain
grows. **The boundary treatment is the term that can break it**, and it needs
deciding before the root is allowed to coarsen, not after.

**Today's rule is the opposite of what will be needed.** `SPONGE_W = 4` is
measured against window coordinates in **L0 units** (`amr_step1.wgsl:427`), so
the ramp is a fixed physical strip four root cells wide. And
`SPONGE_EXCLUDE_W = 8`, also in L0 window cells, **forbids** vorticity-driven
refinement inside a band twice that wide — it exists to keep fine blocks *out*
of the sponge. Both rules are written in root cells, which is only a meaningful
unit while the root is the finest thing in the far field.

**What happens when the root coarsens.** Prepending `k` coarser levels makes
`dx_root` grow as `2^k`, so a sponge of four root cells grows physically as
`2^k` too. The ramp's *area fraction* stays constant (perimeter grows with the
domain at the same rate), which sounds fine — but it means the **usable**
domain is a constant fraction of the total, so growing the domain never buys
proportionally more usable fluid. That is precisely the thing section 8.1 set
out to get.

**The three policies, and the trade is real.** Let `k` be the number of coarse
levels prepended, `L` the ramp's physical width, and `s` the level the ramp is
resolved at.

| policy | `L` | edge-ring tile cost | usable-domain fraction |
|---|---|---|---|
| ramp fixed in **root cells** (today) | grows as `2^k` | constant | **constant** — growing the domain buys nothing |
| ramp fixed in **physical units**, fixed level `s` | constant | **grows as `2^k`** (perimeter x width / tile area) | approaches 1 |
| ramp fixed at **root + j levels**, `j` a small constant | grows as `2^k`, but `2^j` smaller | constant | constant, improved by `2^j` |

The middle row is the one that breaks section 8.2: an edge ring of fixed
physical width and fixed level costs tiles proportional to the domain
perimeter, i.e. `2^k` — the same rate as the extent, and asymptotically
dominating the `k`-ish interior closure cost. **You can have cheap, or you can
have a usable fraction that improves with domain size, but not both — unless
the required physical width `L` itself shrinks, which is the "better far-field
treatment" lever from 8.4.**

**So the recommendation is the third row**: express the sponge's resolution as a
fixed depth below the root, not as a fixed physical width and not as a fixed
count of root cells. It keeps the tile cost flat, improves the usable fraction
by a factor of `2^j`, and hands the ladder a clean knob (`j`) to sweep. And it
means **the domain edge is force-refined**, which is the inversion this section
is named for: the band that today refuses refinement becomes a band that
demands it, for the same underlying reason — to control where the seam is.

**Refining the ramp does not by itself make a thinner ramp work, and the note
should not be read as claiming it does.** An absorbing layer's reflection
depends mainly on how abrupt the ramp is relative to the disturbance's length
scale; cell count fixes the *discretisation* of the ramp, not its physical
abruptness. What forced refinement buys is **decoupling**: today "how physically
wide is the sponge" and "how many cells resolve it" are the same number, and
below four cells the smoothstep stops being smooth. Separating them makes the
width a parameter that can be swept instead of a consequence of where the root
happens to be. Whether a given width is affordable in reflection terms is then
the 8.4 measurement.

**A seam must not cross the ramp, and this project has already been bitten by
it.** `amr_step1.wgsl:415-426` carries the incident: a refined block inside the
sponge with no sponge of its own diverged from its damped coarse neighbours,
and `average` then wrote that undamped state back onto them. The fine level was
given its own copy of the sponge in response. That fix makes a seam in the ramp
*survivable*; it does not make it *correct*, and under explode/coalesce it gets
worse rather than better — inside the ramp the collision is modified, so
populations delivered across a seam there were produced under a different
effective relaxation on each side, and B6's transfer has no term for that.

So the forced-refined edge ring must be **wider than the ramp plus a margin**,
so the whole ramp lives at one level. That is the same rule, for the same
reason, as the geometry-forced refinement that keeps every leaf near the body at
the finest level: do not put a seam where the physics is doing something
special.

**Two implementation consequences.**

1. **The ring moves.** The sponge band is window-anchored while the buffer is
   periodic and the window translates with the card, so the band sweeps through
   buffer space. The forced-refined ring is therefore a dynamic refinement rule
   evaluated every refine round, exactly like the body's — not a static
   allocation made once at init. It also means its tiles are continuously
   created and retired, so it is a steady load on the pool's free list and
   `POOL_PEAKS` will have to be re-measured, not extrapolated.
2. **A latent hole to close while here.** `SPONGE_EXCLUDE_W` gates only the
   vorticity term; it deliberately preserves `isNearBody` and the 2:1 cascade.
   So a body-forced refinement *can* already put a seam in the sponge today.
   On `index-amr.html` it cannot happen in practice because the body is
   window-centred, which is a guard by geometry rather than by rule. Check the
   pinned-body pages, where nothing centres anything.

**The gate this needs** is a sponge analogue of `debugCheckGeometryCoverage`:
every cell within the ramp plus margin is at the designated sponge level, and
no coarse/fine seam intersects the ramp. It belongs in the invariant sweep
alongside the other eight, and — per this project's own standard — it has to be
shown to go red on a deliberately bad configuration (`?spongeExclude=0` with a
criterion that fires at the edge) before it is believed when it is green.

