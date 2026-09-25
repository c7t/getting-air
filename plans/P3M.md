# Plan: Implement the remaining VPM phases (P3M velocity + sparse remeshing), 2D-scoped

## Context

The LBM panel shows fine tip-vortex shedding; the VPM panel doesn't, even after tuning ε/remesh-h/ν/body-grid-h down. The prior investigation this session found the tuning helps but doesn't remove the ceiling: `MAX_N = 2048` (`vpm.js:37`) hard-caps every particle buffer, and `remeshParticles()` (`vpm.js:264-299`) enforces that cap by sorting occupied lattice nodes by `|alpha|` and discarding the rest whenever it's exceeded — which deletes exactly the low-circulation fine structure (tip vortices) first, keeping the bulk wake. `vpm_hybrid_method.md` (the project's own roadmap) calls this out as expected at this stage: Phase 3 (P3M/Ewald-split Biot-Savart, grid+FFT far-field) and the sparse-block half of Phase 4 (remeshing) were never implemented — the sim is still doing brute-force/hash-truncated O(N²)-ish work, which is why N is capped so low. The user has decided to stay 2D and extend the existing LBM-vs-VPM comparison scaffold rather than chase the doc's 3D/Jetson framing — the doc's *phase structure* still applies, just re-scoped.

Goal: raise the particle budget ~10-25x (2048 → 8k-30k) without the CPU-side JS or the O(N²) GPU kernels becoming the bottleneck, by finishing Phase 3 and the sparse half of Phase 4, so tip-vortex-scale detail can actually survive remeshing.

**Key 2D simplification worth flagging to whoever implements this:** the doc's cost model is written for 3D vector-potential VPM. In 2D, vorticity and the stream function are scalars: `∇²ψ = -ω`, `u = (∂ψ/∂y, -∂ψ/∂x)`. In Fourier space (k=0 zeroed):
```
ux_hat =  i*ky/|k|^2 * omega_hat
uy_hat = -i*kx/|k|^2 * omega_hat
```
So the far-field solve is one forward FFT of a scalar field plus two inverse FFTs — cheaper than the doc's 3-component estimate.

No FFT/Poisson/spectral code exists anywhere in the repo today (verified by grep) — this is new infrastructure, hand-written WGSL (no bundler, no libraries, shaders fetched as raw text via the existing `loadShader()`).

## Milestone 0 — Cheap wins + baseline capture (do first, gates everything else)

Before investing in P3M, apply the tuning already diagnosed and capture a baseline so later milestones have something concrete to compare against:
- Lower `NU` default (`vpm.js:61`) from 0.05 toward ~0.005-0.01 to match LBM's `ν=(τ-0.5)/3` at default `τ=0.509` (≈0.003) instead of exceeding LBM's entire slider range.
- Lower `EPSILON` (`vpm.js:58`) default from 8 toward 2-4, `REMESH_H` (`vpm.js:63`) matched to it, `BODY_GRID_H` (`vpm.js:86`) from 4 toward 1-2.
- Run `compare.js`'s lock-step + `cmp-calibrate` (existing UI, no code change) at these new defaults with `bodyMode='lbm'`, capture the `Fy`/`Tz` `corr`/`scale`/`relResidual` numbers as the **pre-P3M baseline** — every later milestone's validation step re-runs this same comparison and checks it hasn't regressed.

## Milestone 1 — FFT infrastructure only, inert (no sim behavior change)

New files: `shaders/vpm_fft.wgsl`, `shaders/vpm_transpose.wgsl`.

Implement a **Stockham radix-2 FFT as separately-dispatched global-memory compute passes** (row pass → transpose → row pass again as the column pass), not a single shared-memory kernel:
- A shared-memory single-kernel FFT at N=1024 needs exactly `1024*2*4*2 = 16384` bytes — right at WebGPU's spec-minimum `maxComputeWorkgroupStorageSize`, with no queried/verified margin (nothing in this codebase queries device limits today). A silent off-by-one in a shared-memory autosort index is hard to localize.
- Multi-pass matches this codebase's existing convention: `shaders/vpm_hash.wgsl` already decomposes into 4 separately-dispatched passes (`clear`/`count`/`prefixSum`/`scatter`) rather than one monolithic kernel.
- Per the doc's own cost estimate, FFT dispatch overhead is not the bottleneck at this domain size — don't trade debuggability for it.

Domain: `gridN = 1024` (2x zero-pad over the existing 512x512 domain, matching the doc's own free-space-BC assumption). Two full-size ping-pong buffers per transform (`vec2<f32>` complex, `gridN*gridN*2*4` bytes).

**Validation (zero risk — nothing live changes):** round-trip identity test — forward FFT then inverse FFT reproduces a known input field (reuse the `gaussian_blob` preset's analytic profile) to float32 tolerance. Add as a debug-only telemetry readout, not wired into `dispatchSubstep`.

## Milestone 2 — Offline P3M velocity, validated, not wired into the sim loop

New files: `shaders/vpm_p2g.wgsl`, `shaders/vpm_g2p.wgsl`, `shaders/vpm_p3m_spectral.wgsl`, `shaders/vpm_velocity_p3m_near.wgsl`.

Ewald split reuses the existing kernel formula (`shaders/vpm_velocity.wgsl`'s `(alpha/(2π r²))·(1−exp(−r²/ε²))·perp(r)`) at two widths — no new kernel derivation:
```
G_far(r)  = kernel(r, sigma_far)              // smooth, grid-resolvable
G_near(r) = kernel(r, epsilon) - kernel(r, sigma_far)   // decays fast past a few sigma_far
```
Pipeline (replacing what will become the two `vpm_velocity.wgsl` calls in `dispatchSubstep`, but not wired yet this milestone):
1. **P2G** (`vpm_p2g.wgsl`): gather-based M4′ scatter of `alpha` onto the far-field lattice (spacing = `REMESH_H`), reusing `cellStart`/`cellCount`/`sortedIdx` from `vpm_hash.wgsl` — each lattice node gathers M4′-weighted contributions from particles in its own + neighboring hash cells (same access pattern `vpm_pse.wgsl` already uses). Gather, not scatter-with-atomics, sidesteps WGSL having no `atomic<f32>`. Requires tying `CELL_SIZE` to `REMESH_H`.
2. **Forward FFT** of `omega_grid` (Milestone 1's shaders).
3. **Spectral multiply** (`vpm_p3m_spectral.wgsl`): pointwise pass producing `ux_hat`/`uy_hat` from `omega_hat` per the §0 formulas, with an Ewald damping factor (exact form derived/verified this milestone, not assumed up front) and `k=0` zeroed.
4. **Two inverse FFTs** → `ux_grid`, `uy_grid`.
5. **G2P** (`vpm_g2p.wgsl`): M4′ gather of grid velocity back to each particle's actual position (direct lattice indexing, same M4′ weights style as `remeshParticles`).
6. **Near-field correction** (`vpm_velocity_p3m_near.wgsl`): same hash-bounded loop structure as `shaders/vpm_velocity_hashed.wgsl`, but with a **small** `NEAR_CUTOFF` (a few `sigma_far`) summing `G_near`, added to G2P's output.

New buffers in `vpm.js`: `omegaGridBufA/B`, `uxGridBufA/B`, `uyGridBufA/B` (ping-pong, `gridN*gridN*2*4` bytes each) plus bind group layouts mirroring the existing `hashVelBGL` pattern.

**Validation:** diff combined (far+near) output against brute-force `vpm_velocity.wgsl` on `'gaussian_blob'`/`'cloud'` snapshots, reusing the exact max-diff pattern `updateHashDiffTelemetry` already implements for the hash-vs-brute-force check. Also: grid-convergence check on the `'single'` preset (known exact answer) — confirm `G_far` at width `sigma_far`, FFT-solved, converges to the brute-force kernel at that same width as grid resolution increases. Sim behavior still unchanged.

## Milestone 3 — Wire P3M into the live sim, `MAX_N` still 2048

Replace the two `vpm_velocity.wgsl` dispatches in `dispatchSubstep()` (`vpm.js:927-942`) with the Milestone 2 pipeline. Keep the brute-force path available for a live diff (repurpose the `hashdiff` telemetry to compare P3M vs. brute-force instead of hash vs. brute-force).

**Validation:** existing analytic-preset telemetry must still match (`omega`/dipole-speed/Lamb-Oseen-M2 checks already in `updateTelemetry`, `vpm.js:775-843`). Re-run Milestone 0's `compare.js` lock-step + `fitStats` against LBM, confirm `corr`/`relResidual` hasn't regressed from the captured baseline. **This milestone alone is fully shippable** even if nothing further is done.

## Milestone 4 — GPU sparse remeshing (finishes Phase 4)

Note on scope: the doc's "sparse grid of fixed-size blocks, active only where vorticity is present" was sized for its 3D, N~3×10⁵ target. At 2D/N~10-30k with a ~10⁶-point grid, a dense FFT over the whole padded domain is already cheap, and free-space FFT wants a dense transform anyway — chasing block-sparsity in the FFT grid itself isn't warranted at this scale. What *does* still matter: bounding the cost of finding "which lattice nodes have particles" for remeshing so it doesn't degrade toward an O(domain) scan.

Extend `vpm_hash.wgsl`'s `count` pass: when `atomicAdd(&cellCount[cell], 1u)` returns 0 (previously-empty cell), that thread also does `atomicAdd(&activeCellCount, 1u)` and appends `cell` to a new compact `activeCellList` buffer — a standard GPU append-buffer variant, reusing atomics already present in this file. Remeshing then reuses the Milestone 2 P2G grid (same lattice, same spacing — this is the actual sharing opportunity) and extracts new particles only from `activeCellList`'s footprint via a new `vpm_g2p.wgsl` entry point, replacing the CPU `Map`-based `remeshParticles()` (`vpm.js:264-299`) entirely. The `|alpha|`-sort-and-truncate-when-over-`MAX_N` step can stay a CPU sort for now (cheap even at 30k entries); flag a GPU bitonic sort as optional/deferred, not required.

**Validation:** `logDebugSample`'s existing `sumA`/`sumAbsA` conservation checks (`vpm.js:877-921`), plus a re-run of `compare.js`'s `fitStats`.

## Milestone 5 — Move `computePenalizationInjection` off brute force

`computePenalizationInjection` (`vpm.js:201-246`) is currently O(gridPoints²·N) — a local body-patch grid where every point does an O(N) `inducedVelocity()` call, no hash acceleration at all, worst-scaling function in the pipeline as N grows.
- **Quick fix:** reuse `shaders/vpm_velocity_hashed.wgsl` as-is with the body-patch grid as `queryPos` (its bind group already separates `queryPos` from `srcPos`) — O(gridPoints × neighbors-within-cutoff) instead of O(gridPoints × N). Minimal new code; do this first.
- **Target end-state:** once the Milestone 2 P3M grid exists, sample `u_far` via G2P-style interpolation at the body-patch points (O(gridPoints)) plus a small hash-bounded near correction only close to the body surface — requires generalizing G2P to accept arbitrary query points, not just particles. New CPU↔GPU staging round-trip (upload body-patch grid, dispatch, read back), following the existing `stagingInFlight` pattern in `vpm.js`.

**Validation:** existing `updateBodyTelemetry`'s `|u-u_body|` convergence check (`vpm.js:845-856`), plus before/after timing of the readback callback.

## Milestone 6 — Raise `MAX_N` in stages

2048 → 8192 → 16384 → 30000, resizing every `MAX_N`-sized buffer (`posBuf`, `alphaBuf`, `midPosBuf`, `vel1Buf`, `vel2Buf`, staging buffers, `particleCellIdBuf`, `sortedIdxBuf`, `refVelBuf`, `hashVelBuf`, `dAlphaBuf` — full list confirmed in `vpm.js:433-459`). Re-run the full regression suite (analytic-preset telemetry, LBM force/torque `fitStats`) at each step. Keep the live brute-force reference/hashdiff check gated below some N (e.g. ~4096, since it's itself O(N²)) — treat higher-N validation as periodic/offline snapshots, not a per-frame live diff.

**This is the actual payoff check:** confirm LBM-vs-VPM force/torque correlation holds or improves as N increases, and visually confirm tip-vortex detail now survives remeshing instead of being pruned.

## UI changes (index.html / vpm.js)

- New sliders: `vpm-slider-P3M-SIGMAFAR` (Ewald far-width) and a `vpm-slider-P3M-NEARCUTOFF` (small near-field radius). The existing `CUTOFF` slider (`vpm-slider-CUTOFF`) keeps driving PSE diffusion's neighbor radius (`writePSEParams`) — relabel it "VPM PSE cutoff R" once P3M takes over the velocity near-field role, don't delete it.
- A debug toggle for offline high-N P3M-vs-brute-force validation snapshots (Milestone 6's gated check).
- `compare.js` needs no structural change (agnostic to `stepOnce`'s internals); worth surfacing live particle count `N` alongside its fit-stats output so fidelity can be correlated against N over Milestone 6's stages.

## Phase 6 (boundary vorticity fallback) — not scheduled

Per the doc, only pursue this if Milestone 6's results show Brinkman penalization still under-resolves tip separation even at N~30k with a properly resolved core/near-field. Don't build it speculatively.

## Critical files
- `vpm.js`
- `shaders/vpm_hash.wgsl`
- `shaders/vpm_velocity_hashed.wgsl`
- `shaders/vpm_velocity.wgsl`
- `shaders/vpm_pse.wgsl`
- `compare.js`
- `index.html`
- New: `shaders/vpm_fft.wgsl`, `shaders/vpm_transpose.wgsl`, `shaders/vpm_p2g.wgsl`, `shaders/vpm_g2p.wgsl`, `shaders/vpm_p3m_spectral.wgsl`, `shaders/vpm_velocity_p3m_near.wgsl`
