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

## fp16 storage for `f`: IMPLEMENTED, and it does NOT hold (2026-09-07)

**Superseded section.** What stood here claimed fp16 storage was measured safe.
It was not measured at all. The claim came from `?quantF16=`, which emulated
half precision by rounding each stored value through
`unpack2x16float(pack2x16float(x))` while keeping f32 storage. That round trip
is a no-op the driver is free to fold away, and it does: with the real packed
implementation now in the tree for comparison,

    dense-reference Re=100, f32 baseline      Cd 1.950 / St 0.1258
    dense-reference Re=100, ?quantF16=1       Cd 1.950 / St 0.1258
    dense-reference Re=100, ?f16=1 (real)     Cd 2.282 / St 0.3080

Four significant figures of agreement between the f32 run and the "quantised"
one, across a 48512-step shedding measurement, is not a null result -- it is
proof that nothing was being quantised. The canary (mode 3,
`round(x*256)/256`) DID move Cd, and that was read as proving the override
reached the shader and therefore that modes 1 and 2 were live. It proves only
the first half: mode 3 is arithmetic the compiler cannot fold, modes 1 and 2
are not. **A control has to be the same KIND of operation as the thing it
controls for.** `QUANT_F16` has been deleted rather than fixed, so its numbers
cannot be read again.

### What the real implementation measures

`shaders/common_fpack.wgsl` + `f-pack.mjs`, `?f16=1` (pack `f_i`) and `?f16=2`
(pack the deviation `f_i - w_i`, ~8x more resolution on the part that carries
the flow, since f_i sits close to its lattice weight). `?f16=0` is the default
and is byte-identical to the previous `array<f32>` layout -- verified by the
full 14-config sweep coming back unchanged.

Against the analytic benchmarks, which is where this had to be tested and
never was:

| check | tol | f32 | ?f16=1 | ?f16=2 |
|---|---|---|---|---|
| poiseuille res=16 Re=10 | 0.01  | PASS | -     | 3.40e-2 |
| poiseuille res=32 Re=10 | 0.005 | PASS | 9.995e-1 | 1.087e-1 |
| couette res=16 Re=10    | 0.01  | PASS | 1.692e-1 | 2.09e-2 |
| couette res=32 Re=10    | 0.005 | PASS | 6.709e-1 | 7.31e-2 |
| tgv N=32 field          | 0.01  | PASS | 5.765e-2 | PASS |
| tgv N=64 field          | 0.01  | PASS | 3.964e-1 | PASS |
| tgv N=128 field         | 0.01  | PASS | 1.138e+0 | 2.59e-2 |

And against the bulk force metrics, which is where it WOULD have been signed
off:

| config | f32 | ?f16=2 |
|---|---|---|
| amr-N2-bounceback Re=100 | PASS | **PASS** |
| amr-N3-diffuse Re=100    | PASS | **PASS** |
| dense-reference Re=100   | Cd 1.950 | Cd 2.008 |

That contrast is the reusable part. Mode 2 passes the Cd/St harness outright
on both AMR configs and is within 3% on the dense one, while being 5-20x
outside tolerance on every channel and TGV field check. Cd and St are
time-averaged integrals over the body surface; they are dominated by the
near-body region, which is exactly where `fneq` is largest and quantisation
hurts least, and they average away the far-field noise rather than reporting
it. The superseded section above even said so in its own last paragraph --
"Cd/St are bulk time-averaged quantities and would not necessarily reveal
far-field quantisation noise... before shipping this, look at a vorticity
render in the quiescent region, or diff fields" -- and then the ship/no-ship
call was made on Cd/St anyway. **For anything that changes precision or
storage, the analytic field checks (`channel-*`, `tgv-*`) are the gate and
Cd/St is not.**

Mode 2 is worth 10-40x over mode 1 and rescues the coarser TGV cases, but
**neither mode meets these tolerances**, and the error GROWS WITH RESOLUTION
in both. That scaling is the whole story: a finer grid resolves the same
physical gradient over more cells, so the per-cell `fneq` -- the entire
information content of the distribution beyond equilibrium -- shrinks toward
the quantum. Refining the grid is exactly what this project does for a living,
so the format gets worse precisely where the work is going.

Poiseuille under mode 1 is the clearest single number: `L2rel = 0.9995`, i.e.
the velocity is essentially zero. It is body-force driven, and the per-step
force increment is smaller than an ulp of `f_i`, so the flow never develops at
all. Mode 2 fixes the mechanism (the increment is representable against the
deviation, not against the weight) and the error drops by 30x -- but not below
tolerance.

### What this does and does not rule out

It rules out fp16 as a DEFAULT. It does not rule out the machinery, which is
why it is staying in the tree at `?f16=0`: the layout is proven correct (the
failure signature is resolution-dependent precision loss, not the
direction-transposition a layout bug would give, and `tools/test-f-pack.js`
asserts host/shader addressing agreement per plane), and it is one URL
parameter away from being re-measured on a device or a format that changes the
answer. Worth revisiting only with something that actually addresses the
signal-to-quantum ratio -- bf16 is worse, not better; a per-cell shared
exponent or storing `fneq` separately from a low-order equilibrium would be
real changes, not a flag.

The traffic arithmetic below was never in question and still holds, if a
format is ever found that keeps the physics:

| layout | B/cell | traffic reduction |
|---|---|---|
| f32 today | 36 | 1.00x |
| 5 u32 planes | 20 | 1.80x |
| ideal tight 9 x f16 | 18 | 2.00x |

### The lesson worth keeping

The accuracy question was asked, an experiment was designed, a control was
run, and the answer was still wrong -- because the experiment measured f32
storage the whole time and the control could not detect that. The cheap
insurance would have been to check `?quantF16=1` against `?quantF16=0` for ANY
difference at all, at any step count. Identical output from a perturbed solver
is a bug report, not a pass.

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
