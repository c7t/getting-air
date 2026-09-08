# Performance characterization: the two target devices have opposite bottlenecks

Measured 2026-09-07 on `index-amr.html?res=8&blockage=3.3`, levels=2, 66
active L1 blocks, `MAX_FINE_BLOCKS=128`, `STEPS_PER_FRAME=64`.

This exists because the obvious optimization — indirect dispatch off the
active block count, so the pool passes stop dispatching 128 slots when only
66 are live — turns out to help **neither** device measurably, and it took
three wrong turns to establish that. Read this before optimizing anything
here.

## The phone throttles, and it invalidates every median (2026-09-07)

Three `?telemetry=1` sessions from the phone (img-tec / PowerVR, Android,
`index-amr.html`), 774 watched frames:

| config | cold | hot | ramp | fps cold -> hot | syncMs/gpuMs |
|---|---|---|---|---|---|
| res=9 N=2 | 174 ms @step 640 | 272 ms @step 17344 | **+57%** | 5.8 -> 3.7 | 2.95x |
| res=8 N=3 | 235 ms @step 320 | 308 ms @step 10368 | **+31%** | 4.3 -> 3.2 | 2.97x |
| res=8 N=2 | 87 ms @step 1280 | 108 ms @step 4736 | +24% | 11.5 -> 9.3 | 2.94x |

The ramp is monotonic from the first sample and plateaus after a few thousand
steps; res=9 N=2 partially recovers late (272 -> 245 ms), consistent with the
device cooling. **This is larger than any optimization on the table.** Force
fusion was worth 5-8%; the thermal state of the phone is worth 24-57%.

Consequences, which apply to everything measured on this device:

- **A median is meaningless without the thermal state.** An earlier session
  concluded the phone had got slower between builds, from min 166 / median
  222 ms against min 226 / median 377 ms. Those are not comparable: the second
  run may simply have been hotter for longer. That conclusion should not be
  relied on.
- **Only the COLD minimum is a build comparison**, and only if the device
  starts from a similar temperature and the sample is taken in the first few
  hundred steps. Take the first telemetry sample, not the median.
- **A phone A/B needs a cooldown between arms**, or interleaving, which
  `?bench=1` gets right for free -- it sweeps configurations within one
  session, so drift hits every arm rather than only the later ones. The same
  reasoning that forced round-robin into `tools/bench-amr.js --skip`.

`syncMs/gpuMs = 2.95x` in every config, to within 1%. That is the 3-deep
submit->readback pipeline showing up exactly as designed, not a hitch -- but
note the absolute number: at 280 ms frames it is ~850 ms of latency between
submitting work and reading its result.

## How to measure (and how not to)

`?bench=1` runs a frame-scale differential sweep: skip a group of passes,
measure the change in median frame GPU time, repeat. `?benchSkip=force,ghost`
applies a skip manually. Results post over the telemetry channel
(`?telemetry=1`), which is the only way to see a phone.

**`?bench=1` was rebuilt on 2026-09-08 and the old numbers from it are void.**
It used to sample per-frame GPU timestamps from the LIVE rAF loop. That does
not work on a device which finishes its frame well inside the vsync interval:
the GPU idles most of each frame and clocks down, so the timestamps scatter no
matter how many are averaged. Measured failing on the desktop -- spreads of
25-88% and shares as low as **-73%** (skipping work cannot make a run slower),
with `interp+avg+ghost` reported at 2.9% against a `tools/bench-amr.js` ground
truth of 44%. It now does what bench-amr.js does: stops the frame loop and
times one synchronous `debugStepSync` run per configuration, with the step
count calibrated per device from a short probe (the two targets differ ~40x in
frame time, so no fixed count serves both). `stepsPerMeasure` and
`msPerStepProbe` are reported in the payload.

**It is validated for the large groups and still cannot resolve small ones.**
Against bench-amr.js on the same machine and config, competing GPU clients
stopped: interp 13.8% vs 15.6-17.3, avg 12.2% vs 12.1-15.9, ghost 13.3% vs
13.7-18.5, step1 32.7% vs 27.5-31.7, interp+avg+ghost 40.5% vs 43.9-49.1. But
0.2-5% effects still come back as 7.9%, -3.6%, -0.9% and -12.2%. On the
desktop the `none` baseline itself carries ~15% spread, because a fast device
runs ~286k steps across the sweep and the card keeps falling while refinement
is frozen, so the workload drifts under every row. **Read anything under ~10%
as below the floor, and a negative share as proof of it.** The phone does far
better (spreads 0.5-6.1%) precisely because it is slow: 16k steps across the
whole sweep, so almost no drift. Capping `stepsPerMeasure` rather than
targeting a fixed wall-clock would fix the desktop the same way; not done yet.

**Stop every other GPU client before measuring, including your own leftovers.**
A benchmark tab left open keeps simulating at full tilt. Three of them
accumulated across one session and took its desktop baseline from 14.8s to
18.9s, which was misattributed to ambient load from the user's browser before
the real cause was found: a driver that closed its CDP client but not its tab,
leaving one live WebGPU context per run plus a GPU process pegged at 98% CPU.
Check `nvidia-smi` reports the SM clock near idle (a few hundred MHz) before
trusting a run; utilization% alone is misleading.

**Do not run `?bench=1` on a device you might switch away from.** A hidden tab
gets its GPU work deprioritised and its timers throttled, so whichever
configurations were current while it was hidden are timed against a different
machine. A phone run was lost to exactly that, silently. The sweep watches
`visibilitychange` and returns `interrupted: true`; discard those numbers. The
full 13-config list is ~4.5 min at the default 3 rounds; `?benchConfigs=` trims
it (7 configs is ~2.7 min on the phone) and `?benchMeasureMs=` sets the target
per-run wall-clock. Progress goes to the `#status` line, which the frame loop
leaves alone while a sweep owns it. An unknown group name is rejected up front
rather than silently reporting ~0% -- a mistyped entry used to be
indistinguishable from a real measurement, which is unrecoverable on a device
that gets one sweep per session.

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

**Superseded for the coupling groups on BOTH columns.** This table is
levels=2 with `MAX_FINE_BLOCKS=128` and predates the threshold retune, the
pool resize, the true-SDF `get_phi` and the FSCALE fix. The desktop column is
re-measured at current defaults further down ("RE-MEASURED at current
defaults"), and both columns are re-measured for interp/avg/ghost/step1 in
"AMR coupling, measured on BOTH devices". Kept for the `phy` additivity
argument below, which is what it was collected for.

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

## RE-MEASURED at current defaults (2026-09-07) -- the fusion case is gone

The table above is levels=2, 66 active L1 blocks, `MAX_FINE_BLOCKS=128`, and
predates the threshold retune, the pool resize to 384/256, the true-SDF
`get_phi` and the FSCALE fix. Re-run at today's defaults
(`res=8&levels=3&blockage=3.3`, frozen at L1=113 / L2=140) with
`tools/bench-amr.js --skip`:

| skipped group | share (levels=3, now) | share (levels=2, old table) |
|---|---|---|
| step1 (fine LBM) | **29.3%** | 34.9% |
| interp (coarse->fine ghosts) | 15.0% | 28.2% |
| ghost (fine-fine) | 12.5% | 32.4% |
| avg (fine->coarse) | 11.8% | 19.6% |
| force (all levels) | **11.4%** | 28.7% |
| force0 (L0 only) | 3.3% | -- |
| phy (body integration) | **3.5%** | 25.4% |

**`phy` collapsing from 25.4% to 3.5% is the headline.** That single
`dispatchWorkgroups(1)` was the entire evidence for "passes themselves are
expensive, so the lever is fewer passes". At levels=3 there is enough real
work per macro-step (253 active tiles against 66) to amortize the fixed
per-pass cost, and the pass-count argument largely dissolves with it.

So **do not fuse the force pass.** All force work is now 11.4%, of which the
L0 pass is 3.3%; fusion does not remove all of that (the step kernel still
does the workgroup reduction and the atomics), so the realistic prize is
5-8% -- barely outside this measurement's own 4-15% spread. Against that:
bind-group layout changes to three shaders shared by five AMR pages (the
238e48c failure mode), a pipeline variant or a redefinition of the body force
to deal with level m stepping 2^m times per macro-step against one force pass,
and a one-macro-step shift in the body update needing full Cd/St
revalidation. That is a bad trade at 5-8%.

**The AMR coupling overhead is the target instead.** interp + ghost + avg =
**39.3%** of the frame is spent moving data between levels, which is more than
the fine solve it exists to serve (29.3%). That is the direct, measured form
of the complaint that started this work -- that the hierarchical version does
not beat the flat one. Two concrete leads, neither yet measured:

- ~~**Ghost-cell fraction in the fine step.**~~ TESTED 2026-09-08, and it is
  not a lead. The reasoning was: a tile is FB=20 square with only 2*RB=16
  square of interior, so 36% of every fine-level step is ghost cells -- about
  10.5% of the whole frame, "the single largest identified piece of pure
  overhead". Measured with `?benchSkip=step1-ring`, which runs the fine step
  over the interior only: **0.2% on the desktop and 0.9% on the phone.**
  Both devices, ~1%, not 10.5%.

  Two reasons the arithmetic overstated it. The ring threads share their 8x8
  workgroups with interior threads, so removing them frees no scheduling slot
  on either device. And skipping them does not remove the ring's *reads* --
  interior threads still stream from those cells -- so the traffic actually
  saved is a fraction of 36%, which is why the bandwidth-bound device did not
  behave differently from the latency-bound one. A prediction of ~10% for the
  phone, made from the traffic model, was wrong by 10x.

  This also removes the second-order case for shrinking `FB` 20 -> 16 (which
  is what eliminating materialized ghosts would allow): the pool would get 36%
  smaller, but the frame would not get measurably faster.
- ~~**Dispatch shape of the three coupling passes.**~~ TESTED, and it is not
  a lead. Doubling the pool allocation doubles every pool pass's dispatch
  width; if width cost anything, their shares would rise. Measured at
  `maxFineBlocks` 384/256 vs 768/512, comparing SHARES within each run (the
  frozen topologies differ, L1 116/L2 148 vs L1 104/L2 132, so absolute times
  are not comparable -- see tools/bench-amr.js's header):

  | skipped | 384/256 | 768/512 |
  |---|---|---|
  | interp | 13.8% | 13.7% |
  | avg | 12.0% | 11.8% |
  | ghost | 16.6% | 10.4% |
  | step1 | 29.2% | 27.2% |
  | force | 12.5% | 11.4% |

  Flat, or slightly lower in line with the ~10% fewer active tiles. An
  inactive slot reads `slotToBlock[slot] < 0` and returns, and that really is
  free. So indirect dispatch buys nothing -- the same conclusion the levels=2
  sweep reached, but now established on the current build and WITHOUT relying
  on the pass-count model `phy` falsified, which is what made it worth
  re-deriving rather than inheriting.

  Note what this also means: the coupling cost is proportional to ACTIVE
  tiles, so it does not shrink by dispatching more cleverly. It shrinks only
  by moving less per tile, or by needing fewer tiles.

## AMR coupling, measured on BOTH devices (2026-09-08)

The question was whether coarse<->fine coupling accounts for much of the
compute time. It does, it is the largest single item in the frame, and it is
the same size on two devices with opposite bottlenecks. `res=8&levels=3&
blockage=3.3` throughout.

| skipped group | phone (PowerVR) | desktop (RTX 4080) |
|---|---|---|
| interp (coarse->fine ghosts) | 19.2% | 13.8-17.3% |
| avg (fine->coarse) | 6.0% | 12.1-15.9% |
| ghost (fine-fine) | 15.3% | 13.3-18.5% |
| **interp+avg+ghost together** | **42.4%** | **40.5-49.1%** |
| step1 (fine LBM) | 32.5% | 27.5-32.7% |

Phone: one `?bench=1` sweep, spreads 0.5-6.1%, `interrupted: false`. Desktop:
six `tools/bench-amr.js --skip` runs plus one validated in-page sweep; the
range is across runs with different frozen topologies, not measurement error.

**Coupling costs more than the fine solve it exists to serve, on both.** That
is the direct, measured form of the complaint that started this work.

**It is additive now, unlike at levels=2.** Parts sum to 43.8/44.3 against
43.9/44.1 measured together on the desktop, and 40.5 against 42.4 on the
phone. The levels=2 desktop numbers were strongly sublinear (parts summed to
80.2%, together 34.5%), so summing individual shares WAS invalid then and IS
valid now. Do not carry the old caution forward without re-checking it.

**Structural reason:** at levels=3 the macro-step encodes **9 coupling passes**
(3 interp, 3 avg, 3 fine-fine ghost) against **7 solve passes** (1 L0 + 2 L1 +
4 L2). See `S_Advance` in main-amr.js.

### It is the WORK, not the pass count

`?benchSkip=<group>-noop` dispatches a coupling pass at full width with a
shader that returns before touching any buffer. Removing the pass entirely
removes its fixed per-pass cost AND its work; the no-op variant removes only
the work, so the difference splits them. Desktop, `tools/bench-amr.js`:

| group | pass removed | dispatched as no-op | -> work |
|---|---|---|---|
| ghost | 15.5-18.5% | 1.7-4.3% | ~13% |
| interp | 17.3% | 2.0% | ~15% |
| avg | 15.9% | 5.6% | ~10% |

Pass existence is ~10% of the frame in total; the work is ~41%. **So fusing
coupling passes is not a lever** -- the same verdict the force-fusion analysis
reached by a different route, now established for the coupling passes too.

### The interpolation math is nearly free; the plumbing is not

`interp` and `ghost` have identical dispatch shapes, run 3x each per
macro-step, and share the same fine-fine copy code. `interp` only adds the
bilinear parent sampling on top. So `interp - ghost` isolates the actual
coarse<->fine transfer math: **+2.0, +0.4 and -1.2 points across desktop runs
(i.e. <=2% and within noise), +3.9 points on the phone.**

Nearly all of what is labelled "interp" is the same-level ghost plumbing that
happens to live inside it, not coarse-fine transfer at all. On the desktop,
running the ghost pass's full prologue and neighbour resolution but skipping
the data copy recovered only 1.8-3.9% of its 15.5-18.5% -- the copy is cheap,
the per-thread prologue and neighbour resolution are not.

### Ruled out, measured, at the +/-3% level

None of these moved the frame; do not re-derive them. Skipping the ghost ring
inside step1 (0.2% desktop / 0.9% phone). Splitting the ghost copy by edge
class to test coalescing (~0%). Replacing the pool shader's runtime integer
`%`/`/` with mask/shift, valid because NBX is always a power of two (0.6%).
Halving `avg`'s live register footprint with a bit-identical single-pass
accumulator (~0%). Indirect dispatch was already ruled out twice.

### The one lead left

**Stop materializing same-level ghost cells.** `amr_interp_pool_parent.wgsl`'s
own header names it: AGAL "addresses neighbor blocks directly during streaming
instead of materializing ghost cells in a padded buffer". That removes the
ghost pass outright (15.3% phone / 13.3-18.5% desktop) and interp's fine-fine
branch, which is nearly all of interp (~15% on both) -- an envelope around
**30% of frame on both devices**, minus whatever the neighbour lookup costs
step1's gather.

Tractable because `step1` already binds the whole pool as `f_in`, so a
neighbour tile's data is in scope; the only new input is `blockSlot` as one
read-only binding. No new data buffers.

Do NOT justify it by the `FB` 20 -> 16 shrink it would also allow -- that was
measured at ~1% (see the struck-out ghost-cell-fraction lead above).

Gates: it touches shaders shared by five AMR pages and changes bind groups,
which is the 238e48c failure mode `validate-all.js`'s boot smoke exists for.
It is physics-affecting, so the analytic `channel-*`/`tgv-*` checks are the
gate, then Cd/St, then AMR invariants, then `validate-divergence` (whose
`fullrefine` leg is the interface-error noise floor and should be unchanged).

`avg` is not worth attacking: it is near its traffic floor (4 reads + 1 write
per coarse cell), the register rewrite did nothing, and it is *cheaper* on the
phone (6.0%) than the desktop (12-16%).

## Superseded: what the levels=2 numbers said to do instead

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
