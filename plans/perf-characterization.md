# Performance characterization: the two target devices have opposite bottlenecks

Measured 2026-09-07 on `index-amr.html?res=8&blockage=3.3`, levels=2, 66
active L1 blocks, `MAX_FINE_BLOCKS=128`, `STEPS_PER_FRAME=64`.

This exists because the obvious optimization — indirect dispatch off the
active block count, so the pool passes stop dispatching 128 slots when only
66 are live — turns out to help **neither** device measurably, and it took
three wrong turns to establish that. Read this before optimizing anything
here.

## How to measure (and how not to)

`?bench=1` runs a frame-scale differential sweep: skip a group of passes,
measure the change in median frame GPU time, repeat. `?benchSkip=force,ghost`
applies a skip manually. Results post over the telemetry channel
(`?telemetry=1`), which is the only way to see a phone.

**Do not trust per-pass timestamps on mobile.** The PowerVR part's timestamp
counter ticks at 65536 ns. A macro-step is ~1125 µs — about 17 ticks spread
across 9–12 passes — so every per-pass reading lands in a 1–8 tick bucket and
the "profile" is quantization noise. This was not noticed at first and led to
a confident, wrong conclusion (see the "force chain = 56%" claim in commit
1358432's message, corrected in 51f1edb). The tell: every value the device
reports is an exact integer multiple of 65536 ns. Frame time, at ~1100 ticks,
is quantized by ~0.1% and is fine — hence frame-scale differencing.

Desktop timestamps are fine-grained and per-pass profiling
(`debugProfileMacroStep`) works there.

## The measurement

Percentage of frame GPU time recovered by skipping each pass group:

| skipped group | phone (PowerVR) | desktop (NVIDIA Lovelace) |
|---|---|---|
| step1 (fine LBM, 2 substeps) | 23.9% | 34.9% |
| force (all levels) | 18.8% | 28.7% |
| interp (coarse→fine ghosts) | 14.6% | 28.2% |
| ghost (fine-fine) | 11.5% | 32.4% |
| avg (fine→coarse) | 4.1% | 19.6% |
| **phy (body integration)** | **2.3%** | **25.4%** |

Baselines: phone 69.5 ms, desktop 4.35 ms per frame (64 macro-steps).

## What it means

**`phy` is the discriminator.** It is a single `dispatchWorkgroups(1)` — one
workgroup, integrating the rigid body, moving essentially no memory. It costs
25.4% of the desktop frame and 2.3% of the phone's. A pass that does no work
and no I/O can only be expensive if *passes themselves* are expensive.

**The additivity test confirms it.** Skipping interp, avg and ghost together
versus separately:

- **phone**: parts sum to 30.2%, measured together 30.0% → **additive**, so
  each pass's cost is its own throughput. No shared floor.
- **desktop**: parts sum to 80.2%, measured together 34.5% → **strongly
  sublinear**, so most of each pass's apparent cost is a fixed per-pass
  charge that does not disappear until the pass does.

So:

- **Desktop is latency/pass-count bound.** The lever is fewer compute passes
  per macro-step. Nine passes at N=2, each carrying a fixed cost that dwarfs
  its own work.
- **Phone is bandwidth bound.** At res=8 the solver moves ~1.6 GB per
  displayed frame; against 69.5 ms that is ~23 GB/s, squarely in mobile
  memory-bus territory. Measured savings track the traffic model
  (step1 29.7% predicted / 23.9% measured; force 16.9% / 18.8%; ghost 14.8% /
  11.5%). The lever is fewer bytes moved.

## Why indirect dispatch is NOT the answer

It was the plan, and the measurement killed it:

- **On the phone** an inactive pool slot reads `slotToBlock[slot]`, sees < 0
  and returns before touching `f`. It costs a workgroup launch and ~4 bytes.
  Since the device is bandwidth bound, and `phy` proves a whole extra pass is
  worth only 2.3%, culling 62 of 128 slot launches is far below that.
- **On the desktop** it does not reduce the number of passes, which is the
  actual bottleneck. Narrower dispatches, same pass count, same floor.

Confirmed independently by an earlier sweep: at fixed physics (~66 active
blocks) desktop frame time went 2.78 ms at `maxFineBlocks=128` and 4.24 ms at
256, so dispatch width is not free — but it is second-order next to pass
count.

## What to do instead

**Fuse the force accumulation into the step kernels.** It is the one change
that attacks both bottlenecks with the same edit:

- removes a whole pass per level (desktop: `force` alone is 28.7%),
- removes a redundant full read of `f` (phone: `force` is 18.8%) — the force
  pass reads exactly the same buffer, at exactly the same point in the
  sequence, as the step pass that follows it. Its own header says so: force
  runs before step and reads the same pre-streaming data.

Cost: `phy` currently runs *before* `S_Advance` and drains the force
accumulator, so fusing force into the step means `phy` must move after, and
the body update lags by one macro-step. That is a real change to the time
discretization (a few thousand macro-steps into a run it should be
negligible, and it is arguably the more natural ordering) but it is a physics
change and must be validated against the cylinder Cd/St bands, not assumed.

Secondary, unquantified: a fine tile is `FB=20` square with only `2*RB=16`
square of interior, so 36% of every fine-level step is ghost cells. That
ratio is structural to `GHOST=2` and 2 substeps, but it scales with tile
size — `RB=16` would cut it to 27%. `RB` is currently welded to `BLOCK=8` by
the L0↔L1 footprint-preserving scheme, so this is not a knob today.

Not worth doing: shrinking `MAX_FINE_BLOCKS` to fit demand. It reduces
dispatch width, which is the thing that does not matter on either device, and
it removes the headroom that stops the pool exhausting (see
`main-cylinder-amr.js`'s note on the banding failure that caused).

## fp16 storage for `f`: measured feasibility (2026-09-07)

The phone is bandwidth bound and `f` (9 x f32 = 36 B/cell) is essentially all
the traffic, so halving its width is worth far more than any pass-level
change. Two questions: how much does it buy, and does the physics survive.

**It does not need the `shader-f16` feature.** `pack2x16float` /
`unpack2x16float` are core WGSL builtins. Storage becomes packed halves in a
`u32` array; arithmetic stays f32. Worth knowing because the desktop adapter
here does NOT expose `shader-f16` at all.

### Accuracy: measured, not argued

`fneq = f - feq` is the viscous stress and is a small difference of large
numbers, so it is what quantisation eats first. Scaling `fneq/f ~ 3*tau*|grad u|`
against fp16's 4.88e-4 relative precision:

| region | \|grad u\|/cell | fneq/f | vs fp16 noise |
|---|---|---|---|
| peak shear at the body | 8.75e-3 | 1.31e-2 | 26.9x |
| mid boundary layer | 2.70e-3 | 4.05e-3 | 8.3x |
| outer BL / near wake | 1.20e-3 | 1.80e-3 | 3.7x |
| quiescent far field | 1.0e-4 | 1.5e-4 | **0.3x** |

So the prediction was: fine near the body, far-field viscous stress lost in
noise. Tested rather than trusted, by emulating fp16 STORAGE precision inside
the existing f32 pipeline (`?quantF16=`, `QUANT_F16` override in
lbm_step.wgsl / amr_step*.wgsl) and running the real Cd/St harness:

| config | mode 0 (f32) | mode 1 (fp16 precision) |
|---|---|---|
| dense, Re=100, bounceback | Cd 1.327 / St 0.1605 | Cd 1.327 / St 0.1605 |
| AMR levels=2 | Cd 1.284 / St 0.1642 | Cd 1.284 / St 0.1642 |
| AMR levels=3 | Cd 1.323 / St 0.1656 | Cd 1.322 / St 0.1655 |

A null result needs a control, so `QUANT_F16=3` is a deliberately coarse
8-bit canary: it gives Cd 2.298 / St 1.842, both FAIL. The override reaches
the shader, and modes 1/2 really are indistinguishable.

The only movement anywhere is at levels=3 (0.08% on Cd), which is exactly
where `amr_interp_*`/`amr_average_*` rescale `fneq` by 2^k across levels and
would amplify quantisation noise. Consistent with the mechanism, far inside
tolerance.

Mode 2 (store `f - w_i`, ~6.7x more resolution on the informative part) turns
out to be unnecessary — mode 1 is already indistinguishable. Keep mode 2 in
mind only if a future regime pushes closer to the floor.

Cd/St are bulk time-averaged quantities and would not necessarily reveal
far-field quantisation noise that the table above predicts. Before shipping
this, look at a vorticity render in the quiescent region, or diff fields with
tools/validate-divergence.js, rather than treating Cd/St as the whole story.

### Gain

9 halves is an odd count; the practical layout is 5 `u32` planes (four packed
pairs plus one half-used) with each thread owning a whole cell, so there is no
read-modify-write hazard:

| layout | B/cell | traffic reduction |
|---|---|---|
| f32 today | 36 | 1.00x |
| 5 u32 planes | 20 | 1.80x |
| ideal tight 9 x f16 | 18 | 2.00x |

Phone projection (frame time scales with `f` traffic):

| config | now | at 1.80x | conservative 1.55x |
|---|---|---|---|
| res=8 N=2 (finest 512-equiv) | 14.4 fps | 25.9 | 22.3 |
| res=8 N=3 (finest 1024-equiv) | 6.2 fps | 11.2 | 9.6 |
| res=9 N=3 (finest 2048-equiv) | 2.2 fps | 4.0 | 3.4 |

The conservative column allows for `velBuf`, indirection and atomics staying
f32, and for some passes becoming latency-bound once their traffic halves.

Versus ~15-20% for fusing the force pass into the step: fp16 is worth roughly
**four times more** on this device, and the two are independent — fusion
removes a redundant read, fp16 halves every read. On the DESKTOP the ranking
reverses, since it is pass-count bound and fp16 does not remove passes.

## What the refinement overlay is actually showing (2026-09-07)

Reported from a phone screenshot: the green overlay looks blotchy and does
not appear to track the field's detail. Measured rather than eyeballed, at
res=8 levels=3 blockage=3.3 (two runs, different flow states):

| metric | run A | run B |
|---|---|---|
| refined L1 blocks | 144 / 1024 | 193 / 1024 |
| mean refined edge-neighbours | 3.33 | 3.19 |
| isolated (0 neighbours) | 0 | 0 |
| refined but BELOW threshold | 36% | 40% |
| above threshold but NOT refined | 28% | 15% |

Two findings, and neither is the one the picture suggests.

**The overlay is not a checkerboard.** Mean 3.2-3.3 refined edge-neighbours
out of 4, with ZERO isolated blocks, is a coherent region, not scattered
speckle. The blocky appearance is the 8-cell block granularity against a
smooth field, not incoherent refinement.

**The blocks above threshold that go unrefined are 100% in the outermost
block ring** -- every one of them, in both runs, at distance 0 from the
window edge. That is `SPONGE_EXCLUDE_W = 8` doing its job: vorticity-driven
refinement is deliberately kept out of the ALBC sponge, because refining an
absorbing layer wastes slots and can seed reflections. It is not a tracking
failure. (Their |omega| runs up to 1.2e-2, well above the 1.95e-3 threshold,
which does say a lot of wake vorticity is reaching the sponge -- expected for
a falling card whose window leaves its own wake behind.)

**What IS true is the inverse of the impression.** 36-40% of refined blocks
are BELOW the vorticity threshold -- they are held by geometry forcing (the
FORCE_REFINE_MARGIN=16 halo around the body) and by the coarsen hysteresis.
So a large fraction of the green is tracking the BODY, not the field, and
that is why it reads as broader and less feature-following than the visible
vorticity. Combined with the level-2 finding above (89% geometry-halo-driven),
the picture is consistent: resolution in this solver is allocated mostly by
proximity to the body, with the criterion contributing at the margins.

That is a deliberate design (the halo exists because the boundary layer
needs it -- see main-amr.js's FORCE_REFINE_MARGIN comment and its
measurements), not a defect. But it does mean "make the refinement follow the
flow" is a question about the halo/criterion BALANCE, not about tuning the
criterion alone, which is where several sessions of threshold sweeps kept
running aground.
