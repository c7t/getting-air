# Sub-plan: Milestone 5 — Data-layout spec + level-generic buffer allocation

Expands `plans/AMR-multilevel.md`'s Milestone 5 (one paragraph) into an
implementation-ready spec. No shader changes, no behavior change — this
milestone is pure JS-side data-structure work, gated by a byte-identical
`?levels=2` check against today's build.

**Amendment (caught while implementing Milestone 6):** the `ownBX`/`ownBY`
fields this doc originally specified (§2, §4) were dropped from the actual
implementation. They're fully redundant with `slotToBlock[slot]` + this
level's own `NBX`/`NBY` (`bx = blockID % NBX[m]`, `by = blockID / NBX[m]`)
— exactly the derivation `amr_interp_dense_parent.wgsl` already performs
every dispatch for level 1 today, so caching it separately bought no real
performance benefit (the "expensive" derivation it would have saved is a
single mod+div the project already pays for elsewhere in the same hot
path) at the cost of a second, redundant source of truth. Read `ownBX`/
`ownBY` below as historical — the shipped code only allocates
`parentSlot`/`quadrant` for levels ≥2. See
`shaders/amr_interp_pool_parent.wgsl`'s header for where the derivation
actually happens.

## 0. Level-indexing convention (pins down an ambiguity the master plan
   leaves implicit)

- **Level 0 (L0)**: the existing dense coarse grid. Untouched (decision 1
  in the master plan). Not part of the `pools[]` array below.
- **Pool levels 1..N-1**: `N` = the new `?levels=` param, default 2 (today's
  behavior). Level 1 is *exactly* today's fine pool: footprint-preserving,
  1 L0 block → 1 L1 tile at 2x density, covering the SAME physical
  footprint as its parent block (`amr_interp_c2f.wgsl`'s `fineToCoarseUnit`,
  main-amr.js:32-35's `RB`/`FB` constants). This relationship is **not**
  quadtree and doesn't change in this plan at all — decision 1 keeps L0 as
  today's dense buffer, and L0↔L1 is the addressing pair built around that
  dense buffer.
- **L(m)→L(m+1) for m≥1**: genuine quadtree, 1 parent tile → 4 children,
  each covering one quadrant of the parent's own footprint at 2x density.
  This is the new topology decision 2 describes. It first applies at
  L1→L2, i.e. it's invisible at `N=2` and only exercised once `N≥3`.

This means level 1 is special (footprint-preserving parent = the dense
grid) and levels ≥2 are uniform among themselves (quadtree parent = a pool
tile). Two addressing schemes, as the master plan says — but it's worth
being explicit that the *split point* is between L1 and L2, not between L0
and L1-through-LN. Level 1's own data shapes are unchanged from today;
only levels ≥2 are new.

## 1. Per-level logical grid extent

Every pool level needs its own "how many logical block positions exist at
this level" for the dense `blockSlot`/`blockCriterion` arrays (see master
plan's neighbor-lookup note, plans/AMR-multilevel.md:23):

```
NBX[1] = NBX0 = W / BLOCK        // unchanged from today (main-amr.js:51)
NBY[1] = NBY0 = H / BLOCK
NBX[m+1] = 2 * NBX[m]            // quadtree doubles the logical grid per axis
NBY[m+1] = 2 * NBY[m]            // per additional level, m >= 1
NBLOCKS[m] = NBX[m] * NBY[m]
```

At `N=2` only `NBLOCKS[1]` exists and equals today's `NBLOCKS`. Confirms
the master plan's flag (plans/AMR-multilevel.md:23) that this dense
scheme is fine for 2-3 levels and stops scaling for deep hierarchies —
`NBLOCKS[3]` is already 16x `NBLOCKS[1]`.

## 2. Per-level buffer inventory

Same fields as today's flat globals (main-amr.js:293-311), generalized.
`NCELLS1 = FB*FB` is level-independent (every level ≥1 uses the identical
tile shape, decision 2) — only the *capacity* (`MAX_FINE_BLOCKS[m]`) and
the *logical grid extent* (`NBLOCKS[m]`) vary per level.

| Buffer | Size formula | m=1 (today, unchanged) | m≥2 (new) |
|---|---|---|---|
| `finePoolF_a`/`_b` | `MAX_FINE_BLOCKS[m] * NCELLS1 * 9 * 4` | same as today's `fSizePool` | new alloc |
| `finePoolVel` | `MAX_FINE_BLOCKS[m] * NCELLS1 * 2 * 4` | same as today | new alloc |
| `blockSlot` | `NBLOCKS[m] * 4` | same as today's `blockSlotBuf` | new alloc |
| `slotToBlock` | `MAX_FINE_BLOCKS[m] * 4` | same as today | new alloc |
| `blockCriterion` | `NBLOCKS[m] * 4` | same as today | new alloc |
| `freeList` | `MAX_FINE_BLOCKS[m] * 4` (m=1) / `(MAX_FINE_BLOCKS[m]/4) * 4` (m≥2, quad units) | same as today | **different stride**, see §3 |
| `freeCount` | `4` (single atomic) | same as today | same, counts quads not slots |
| `newlyActivated` | `MAX_FINE_BLOCKS[m] * 4` | same as today | new alloc |
| `parentSlot` | — (m=1 uses `cellIndex`/`blockID`, not this) | **N/A** | `MAX_FINE_BLOCKS[m] * 4`, new field |
| `quadrant` | — | **N/A** | `MAX_FINE_BLOCKS[m] * 4` (or packed 2 bits; start as `u32` for simplicity, pack later if memory matters) |
| `ownBX`/`ownBY` | — | **N/A** (derivable: `bx=blockID%NBX0`) | `MAX_FINE_BLOCKS[m] * 4` each, cached at allocation |

`parentSlot`/`quadrant`/`ownBX`/`ownBY` only exist for m≥2 — they encode
exactly the two addressing gaps the master plan calls out
(plans/AMR-multilevel.md:22-24): "which parent slot + quadrant" and "own
logical position, cached rather than re-derived."

Field semantics for m≥2:
- `parentSlot[slot]`: the level-`(m-1)` pool slot this tile's parent quad
  lives at. All 4 sibling slots born from the same refine event share the
  same `parentSlot` value.
- `quadrant[slot]`: 0-3, fixed at allocation. Convention (pin now, so M6
  doesn't have to invent it under time pressure): quadrant bit 0 = +x half,
  bit 1 = +y half, i.e. `quadrant = dx + 2*dy` with `dx,dy ∈ {0,1}`.
- `ownBX[slot]`, `ownBY[slot]`: `ownBX = parentOwnBX*2 + (quadrant & 1)`,
  `ownBY = parentOwnBY*2 + (quadrant >> 1)`, computed once at allocation
  time from the parent's own cached `ownBX`/`ownBY` (or, for a level-2
  quad whose parent is level 1, from the parent's `blockID % NBX[1]` /
  `blockID / NBX[1]`, since level 1 has no separate `ownBX`/`ownBY` array).

## 3. Allocation granularity split (a third asymmetry, worth flagging
   explicitly — not stated as plainly as the addressing split in the
   master plan)

Level 1 keeps **today's per-block allocation** exactly as-is: one L0
block activates one L1 slot, individually, matching `debugActivateBlock`'s
existing per-`(bx,by)` granularity (main-amr.js:869-905). There is no
"quad" on the L0↔L1 boundary — L0 isn't itself decomposed into quads, it's
the dense grid decision 1 keeps unchanged.

Levels ≥2 use decision 3's **quad-unit allocation**: refine/coarsen always
grants or releases 4 slots at once (one full quadtree split of a single
parent tile). Concretely:
- `freeList[m]` (m≥2) is indexed in **quad slots**, length
  `MAX_FINE_BLOCKS[m]/4`. A popped quad slot `q` expands to 4 real pool
  slots `q*4 + 0..3`.
- `atomicAdd`/`atomicSub` on `freeCount[m]` (m≥2) move by 1 **quad** at a
  time, same mechanism as today's per-slot stack, just reinterpreted at
  4-slot stride — this is decision 3's own wording
  (plans/AMR-multilevel.md:10), restated here specifically to make clear
  it does *not* apply to level 1.
- **Constraint**: `MAX_FINE_BLOCKS[m]` must be a multiple of 4 for m≥2.
  Assert this at allocation time and throw (matching the project's
  existing "fail loud on layout mismatch" convention, e.g.
  debugSnapshotLoad's format-version checks at main-amr.js:685-687,701)
  rather than silently truncating.

## 4. JS structural refactor

Replace the flat globals at main-amr.js:293-311 (`finePoolF_a`,
`finePoolF_b`, `finePoolVel`, `blockSlotBuf`, `slotToBlockBuf`,
`blockCriterionBuf`, `freeListBuf`, `freeCountBuf`, `newlyActivatedBuf`)
with a `pools` array, 1-indexed by level (`pools[0]` unused/undefined to
keep the level-1 naming intuitive, or use a `Map`/plain object keyed by
level — either is fine, pick whichever reads better against existing
`slot`/`blockID` naming):

```js
function allocLevelPool(device, U, m, { NBX_m, NBY_m, maxFineBlocks }) {
  const NBLOCKS_m = NBX_m * NBY_m;
  const fSizePool_m = maxFineBlocks * NCELLS1 * 9 * 4;
  const pool = {
    level: m,
    NBX: NBX_m, NBY: NBY_m, NBLOCKS: NBLOCKS_m,
    MAX_FINE_BLOCKS: maxFineBlocks,
    finePoolF_a: device.createBuffer({ size: fSizePool_m, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    finePoolF_b: device.createBuffer({ size: fSizePool_m, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    finePoolVel: device.createBuffer({ size: maxFineBlocks * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_SRC }),
    blockSlotBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    slotToBlockBuf: device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    blockCriterionBuf: device.createBuffer({ size: NBLOCKS_m * 4, usage: U.STORAGE | U.COPY_DST }),
    freeListBuf: null,   // sized below, granularity depends on m
    freeCountBuf: device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }),
    newlyActivatedBuf: device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST }),
  };
  if (m === 1) {
    pool.freeListBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  } else {
    if (maxFineBlocks % 4 !== 0) throw new Error(`level ${m}: MAX_FINE_BLOCKS (${maxFineBlocks}) must be a multiple of 4 (quad allocation)`);
    pool.freeListBuf = device.createBuffer({ size: (maxFineBlocks / 4) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    pool.parentSlotBuf = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST });
    pool.quadrantBuf   = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST });
    pool.ownBXBuf      = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST });
    pool.ownBYBuf      = device.createBuffer({ size: maxFineBlocks * 4, usage: U.STORAGE | U.COPY_DST });
  }
  return pool;
}

const N_LEVELS = urlParams.has('levels') ? parseInt(urlParams.get('levels')) : 2;
const pools = [undefined]; // pools[0] unused; pools[1..N_LEVELS-1] populated
let NBX_m = NBX, NBY_m = NBY; // level 1 starts at today's coarse block grid
for (let m = 1; m < N_LEVELS; m++) {
  const maxFineBlocks = m === 1
    ? MAX_FINE_BLOCKS // unchanged param, unchanged default (128)
    : (urlParams.has(`maxFineBlocks${m}`) ? parseInt(urlParams.get(`maxFineBlocks${m}`)) : 128);
  pools[m] = allocLevelPool(device, U, m, { NBX_m, NBY_m, maxFineBlocks });
  NBX_m *= 2; NBY_m *= 2; // next level's logical grid, quadtree doubling
}
```

Every place today's code references `finePoolF_a`, `blockSlotBuf`, etc.
directly (bind group creation at main-amr.js:519-550, staging buffers at
main-amr.js:604-610, `debugSnapshotSave`/`Load` at 616-734,
`debugActivateBlock`/`debugDeactivateBlock` at 869-926,
`readPoolIndirection`, `resetSim`) gets threaded through `pools[1].*`
instead. This mechanical rename is the bulk of M5's actual diff — **not**
the new-level allocation loop above, which is comparatively small and
inert until M6/M7 wire a shader to read it. The rename touching every one
of those call sites, without changing a single one of their semantics, is
the real regression risk in this milestone; treat it with the same care
as any refactor that must produce a byte-identical result.

## 5. Explicit non-goals (deferred to later milestones, called out so M5
   doesn't quietly grow scope)

- **Snapshot format stays singular.** `debugSnapshotSave`/`Load`'s
  `pool: {...}` key (main-amr.js:664-669, 699-729) keeps serializing only
  `pools[1]` under that same singular key, unchanged shape, even though
  `pools[]` is now internally an array. Milestone 10 owns turning this
  into an array-of-pools with a `formatVersion` bump
  (plans/AMR-multilevel.md:73) — doing it early here would be exactly the
  kind of "fold two milestones together" the master plan warns against
  for M7 (plans/AMR-multilevel.md:42), and for the same reason: it adds a
  second axis of change under the same commit as the buffer-layout
  refactor, when the whole point of M5 is to be checkable byte-for-byte
  against today.
- **No new shader files, no new bind groups for m≥2.** `parentSlot`/
  `quadrant`/`ownBX`/`ownBY` buffers get allocated (so M6 has something to
  bind to) but are never bound into a pipeline or written by anything
  other than a future refine dispatch. At `N=2` they don't exist at all
  (loop only runs `m=1`).
- **No dispatch changes.** `dispatchMacroStep` (main-amr.js:745-788)
  keeps referencing `pools[1].*` fields by their new path but its actual
  pass sequence, ordering, and pipeline objects are untouched.

## 6. Validation plan

1. **Byte-identical allocation at `?levels=2`.** Add a debug introspection
   hook, e.g. `window.__AMR.getLevelPoolSizes()`, returning per-level
   `{level, NBX, NBY, NBLOCKS, MAX_FINE_BLOCKS, byteSize per buffer}`.
   Record today's (pre-M5) values once, then assert the post-M5 `N=2`
   output matches exactly — same numbers, same buffer count. This is a
   pure-JS check, no GPU readback needed, fast enough to run on every
   iteration while doing the refactor.
2. **Full numeric regression via existing tooling.** Capture a baseline
   snapshot on the pre-M5 code with `tools/amr-snapshot.js run-to-step`,
   then do the same on post-M5 `?levels=2` code, and diff with
   `tools/amr-diff.js`. Expect **zero** difference (not "within
   tolerance" — nothing computational changed, only how JS refers to the
   same buffers), so treat any nonzero diff as a real regression from the
   rename, not a numerics question.
3. **N≥3 allocates cleanly, does nothing yet.** With `?levels=3`, confirm
   (a) no GPU validation error from the extra buffer creation (the
   existing `device.popErrorScope()` check at main-amr.js:556-557 already
   catches this), (b) `pools[2].MAX_FINE_BLOCKS % 4 === 0` is enforced —
   deliberately pass an non-multiple-of-4 `?maxFineBlocks2=` value and
   confirm the thrown error, matching this project's fail-loud convention,
   (c) the sim's actual behavior (trajectory, snapshot diff) at `N=3` is
   **identical** to `N=2` — pools[2] existing and being initialized but
   never dispatched into must be a true no-op.
4. **Existing debug API still works unmodified at `N=2`.**
   `debugActivateBlock`/`debugDeactivateBlock`/`debugListActiveBlocks`/
   `debugSnapshotSave`/`Load`/`debugStepSync` all keep working exactly as
   before — run through `tools/validate-cylinder.js`'s existing flow
   end-to-end, not just a unit-level check, since that harness exercises
   most of this surface already.

## 7. Open questions carried into Milestone 6 (deliberately not resolved
   here — M5 allocates, M6 consumes)

- Whether `quadrant` is worth bit-packing (2 bits into a shared word with
  something else) vs. a plain `u32` array — start with the plain array
  (this table already has enough new arrays to reason about without also
  optimizing their bit layout); revisit only if M9/M10's memory budget
  demands it.
- `MAX_FINE_BLOCKS[m≥2]` defaults (128 placeholder here, same as level 1)
  are not measured — Milestone 10 explicitly owns retuning capacity/
  thresholds per level once there's a real `N=3` scenario to measure
  against (plans/AMR-multilevel.md:72).
- `?maxFineBlocks2=`, `?maxFineBlocks3=`-style per-level param naming is a
  placeholder; fine to bikeshed later, not load-bearing for M5's own
  validation.

## Files touched
- `main-amr.js` — the buffer-alloc block
  (main-amr.js:287-311), staging buffers (604-610), `debugSnapshotSave`/
  `Load` (616-734), `debugActivateBlock`/`debugDeactivateBlock` (869-926),
  `readPoolIndirection`, `resetSim`, bind-group construction (519-550),
  `window.__AMR` exposed getters.
- No shader files. No `index-amr.html` changes (`?levels=` is a URL param
  like `?res=`, no new UI control needed at this milestone).
