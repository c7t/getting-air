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
| U4 | criterion, force, digest, conserved totals | **U4-0/1/2 DONE** — `vel` and the criterion bit-identical; the FORCE agrees to the truncation floor and the bit-identity prediction is falsified. digest + conserved totals remain | `tools/validate-root-kernels.js`, 3 GPU mutants |
| U5 | L1 becomes a quad child of the root | not started | — |
| U6 | The renderer walks levels | not started; **its gate already exists and already fails** | `tools/validate-render-levels.js` |
| U7 | Delete the dense path | not started | — |

**Where the risk actually sits.** U0–U2 went in clean and each found something
(a half-cell convention, an f16 stride, a window-convention split between L0 and
the pool levels). U3 is the first stage where a kernel depends on the mapping,
and it is failing — which is the staging working as designed: the defect is
sitting in a flag-gated, inert code path with an instrument pointed at it,
rather than in the shipped solver.

**One correction to this plan's own reasoning, worth reading before U4.** The
table used to say U3–U4 are scored by "bit-identity", justified by "the
arithmetic per cell is unchanged; only the address space moves". That holds for
ONE kernel on a different buffer and NOT for `amr_step.wgsl` against
`amr_step1.wgsl`, which are separately written and owe each other no f32
association. Bit-identity is the right bar at U7, where the comparison becomes
one kernel against itself across builds. Until then the bar is agreement after
a SINGLE macro-step, scored by magnitude — see U3.

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
    U1..U7  uniform levels
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

### U7 — delete the dense path

`amr_step.wgsl`, `amr_criterion.wgsl`, `amr_manage.wgsl`, `amr_force.wgsl`,
`amr_interp_dense_parent.wgsl`, `amr_average_f2c.wgsl`,
`common_interp_parent_dense.wgsl`, `common_avg_parent_dense.wgsl`, the `f_a` /
`f_b` / `velBuf` dense buffers and their bind groups.

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

---

## 6. Traps, carried forward

**The dev-server trap applies to every stage here.** `ensureServer` reuses
whatever answers on the port. With the main checkout, `3d-trt` and this
worktree all live, the default `https://localhost:4444` is routinely somebody
else's tree. A full sweep and a "pristine A/B" have both already run green
against the wrong checkout and had to be discarded. Prove the tree with a token
you added, then pin both ports.

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

