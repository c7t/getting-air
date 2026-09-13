# TRT: fixing the tau family at its root

**Status: BUILT AND MEASURED, 2026-09-13; the verdict is NEGATIVE for the
card and is in sec 8.14. THE CODE IS NOT MERGED: `?trt=1&lambda=` exists only
on branch `worktree-3d-trt` (tip 2ea1d12), kept there as a measured dead end.
What this branch carries from that work is the weights fix, the D32 control
case, and this record.** On that branch the operator is off by default, dense
and pool, bit-identical to BGK when off.
It fixes the static wall exactly as the theory says (duct error down 3-6x,
pinned sphere +7% -> +4% over Schiller-Naumann) and makes BOTH of the
card's actual limits worse: the moving-body force deficit doubles (8.12)
and the stability wall RISES at the Mach numbers this project runs (8.13).
Along the way it found a mass leak in every 3D collision -- the f32 lattice
weights did not sum to 1 (8.10, commit 69feb72) -- which is the lasting
result. Sections 1-7 are the 2026-09-11 design note, kept as written so the
predictions can be read against the measurements; sec 8.0-8.9 is the plan
as it stood before building; 8.10-8.14 are what happened.

---

## 1. What the measurements say, so the motivation is not assumed

`tau = 1/2 + 3*nu`, and `Re = U*D/nu`, so

    tau = 1/2 + 3*U*D/Re

Reaching a high Reynolds number at fixed resolution FORCES tau toward 1/2.
M8.0 already states the consequence for stability ("the margin is bought with
RESOLUTION, not with anything else"). D3 measured a second consequence, and it
bites long before the solver becomes unstable.

**The moving-body force is short as `1/(tau - 1/2)`.** Measured by momentum
budget in a periodic sponge-free box, where the fluid's momentum can only
change through the body so `sum F` must equal `-dP` exactly:

    tau            0.509     0.55      0.6      0.8
    short by       16.6%     3.6%     1.5%     0.0%
    x (tau-1/2)   1.5e-3   1.8e-3   1.5e-3      --

**And the Galilean split says the same thing from the other side.** One flow in
two inertial frames must give one Cd. At tau = 0.509 the dense path reads
-18.5% across the frame and the pool +47.3%; at tau = 0.6 both are flat to ~1%
and agree with each other to ~1%.

**A PINNED body at the same tau is fine** -- `sphere-Re200-D16` runs at
tau = 0.512 and sits +7.0% against Schiller-Naumann. So this is specifically
the MOVING coupling, which is why nothing in the suite caught it until a body
moved and its drag was measured (plans/3D.md D1).

**The card's own target is inside the bad region.** n = 64, Re = 1100 gives
tau = 0.5087 dense. AMR helps -- `tauAtLevel` doubles `(tau - 1/2)` per rung,
so two rungs put the body's own level at tau = 0.535 -- but that is a factor of
4, bought with an 8x memory/compute step per rung in 3D, against a defect that
needs roughly 10x to become negligible.

---

## 2. Why tau -> 1/2 breaks things, in three separable ways

BGK has ONE relaxation rate `omega = 1/tau` for every non-conserved moment.
That single number is asked to do three unrelated jobs, and at low tau it
cannot do them all.

**(a) It sets the viscosity.** `nu = cs^2 (tau - 1/2)`. This is the job you
actually want it for, and the only one.

**(b) It damps the non-hydrodynamic ("ghost") moments.** D3Q19 has 19 moments:
1 density, 3 momentum, 6 stress -- and 9 GHOST moments that carry no
hydrodynamics at all. BGK relaxes them at the same `omega`. As `omega -> 2` the
collision `f -> 2*f_eq - f` becomes an INVOLUTION: it reflects populations
about equilibrium, so ghost content is neither damped nor amplified. It
persists, and the `u^2` terms in `f_eq` couple it back into the hydrodynamics.
That is the NaN mechanism, and the ghost damping rate goes as
`(2 - omega) = 2(tau - 1/2)/tau`, i.e. as `(tau - 1/2)`.

**(c) It sets where a bounce-back wall actually is.** Halfway bounce-back
places the wall exactly at the link midpoint iff

    Lambda = (1/omega_+ - 1/2)(1/omega_- - 1/2) = 3/16

(Ginzburg & d'Humieres). BGK ties `omega_+ = omega_- = omega`, so
`Lambda = (tau - 1/2)^2` and you do not get to choose it. benchmarks/d3.json
already records the consequence for the STATIC duct cases:
tau = 0.9330127 = 1/2 + sqrt(3)/4 is where `Lambda = 3/16`, and the tau = 0.8
and 0.6 cases sit deliberately off it "to keep the wall-position error visible
rather than tuned away". At tau = 0.509, `Lambda = 8.1e-5` -- **2300x below the
value at which the boundary is exact**.

**The suspected link between (b) and D3's law, stated as a hypothesis.**
Momentum exchange is a raw sum of INDIVIDUAL populations over a SUBSET of
directions. That is not a moment, so ghost content enters it directly. A moving
boundary continuously pumps ghosts (fresh nodes injected at equilibrium with no
`fneq`, covered nodes destroyed, the staircase reshaping), they accumulate to
an amplitude `~ 1/(2 - omega) ~ 1/(tau - 1/2)`, and the force reading inherits
it. A PINNED body pumps a CONSTANT ghost field, which is a fixed offset
absorbed into the standing staircase allowance.

Two independent measurements are consistent with it and neither proves it:
the `1/(tau - 1/2)` law itself, and the fact that **enstrophy** -- which ghosts
cannot touch, being orthogonal to `rho` and `u` by construction -- matches to
0.3% between the two frames while Cd differs by 19% and 47%. The diffuse
coupling, whose force reads `rho` and `u*` (both MOMENTS), shows a 2.0% frame
gap where bounce-back shows -18.5% (plans/3D.md D4). That is the prediction the
hypothesis makes, and it held.

---

## 3. What TRT is

Split the populations into even and odd parts about each direction pair:

    f_i^+ = (f_i + f_ibar)/2        f_i^- = (f_i - f_ibar)/2

and relax the two with DIFFERENT rates:

    f_i^new = f_i - omega_+ (f_i^+ - f_i^{eq,+}) - omega_- (f_i^- - f_i^{eq,-})

`omega_+` carries the viscosity exactly as BGK's `omega` did:
`nu = cs^2 (1/omega_+ - 1/2)`. `omega_-` is then FREE, and you spend it on
`Lambda`:

    Lambda = (1/omega_+ - 1/2)(1/omega_- - 1/2) = 3/16

**So `Lambda` becomes a constant you choose rather than `(tau - 1/2)^2` that
the Reynolds number chooses for you.** Job (c) is decoupled from job (a). And
because `omega_-` stays O(1) as `omega_+ -> 2`, the odd ghost moments keep
being damped at O(1) -- which addresses job (b) for half the ghost spectrum.

BGK is the special case `omega_- = omega_+`, so **a TRT implementation with
`omega_- = omega_+` must be bit-identical to the current solver**. That is the
gate to build first (see sec 5).

**MRT is the generalization** -- relax every moment at its own rate -- and is
strictly more powerful and strictly more code (a 19x19 transform, or a
hand-optimized moment basis). TRT gets most of the benefit for a small fraction
of the work and has the clean `Lambda` story. Start at TRT; MRT only if
measurement says TRT is not enough.

---

## 4. What it would cost here

**The collision appears in exactly three 3D kernels**: `common_d3_step.wgsl`,
`common_d3_amr_step1.wgsl`, `common_d3_spike_body.wgsl` (the M0 bench, which
does not need it). So two real sites, and the 2D solver is untouched.

The change at each site, replacing `f[i] - OMEGA*(f[i] - feq)`:

    f_i^+  = (f[i] + f[opp[i]]) / 2          // needs f[opp[i]] -- see below
    fe_i^+ = (feq_i + feq_opp) / 2
    f_i^-  = (f[i] - f[opp[i]]) / 2
    fe_i^- = (feq_i - feq_opp) / 2
    fCollide = f[i] - OMEGA_P*(f_i^+ - fe_i^+) - OMEGA_M*(f_i^- - fe_i^-)

**`f[opp[i]]` is already in registers.** Both kernels gather the whole `f`
array into `var f: array<f32, QN>` before colliding, so the pair is a register
read, not a second load. The even/odd equilibria are cheaper than they look:
`feq_i` and `feq_opp` differ only in the sign of `(e_i . u)`, so

    fe^+ = w_i rho (1 + 4.5 eu^2 - 1.5 u^2)      fe^- = w_i rho * 3 eu

i.e. **the split equilibrium is cheaper to compute than two full `feq`s**, and
arguably cheaper than the current single one.

**Cost estimate: near zero.** plans/perf-characterization.md's standing finding
is that these kernels are memory-bandwidth bound (M0 measured the full kernel
matching its own stream-only ceiling at both velocity sets, 77%/71% of device
peak, no spill). TRT adds arithmetic and no traffic. **Measure it rather than
assume it** -- `tools/spike-d3-registers.js` is the tool, and the one thing
that could bite is register pressure pushing the kernel off its ceiling, which
is exactly what sec 2.4 retired for Q27 and would have to be re-checked.

**The forcing term needs care.** Guo forcing has its own `(1 - omega/2)`
prefactor; under TRT the even and odd parts of the source take `omega_+` and
`omega_-` respectively. Getting this wrong is a silent momentum error in
exactly the duct case that would otherwise gate it.

**Every grid transfer carries tau.** `common_d3_pool.wgsl` derives the
Dupuis-Chopard `fneq` rescale `(tau_f - 1)/(tau_c - 1)/2`, and `tauAtLevel`
walks `tau -> 2 tau - 1/2` per rung. Under TRT the rescale is a statement about
the EVEN (viscous) relaxation, so it should key on `omega_+`, and `omega_-`
needs its own per-level rule to hold `Lambda` constant at every level. **That
is the part most likely to be got wrong**, and it is the part with the sharpest
existing gate (`amr_conserve_cases`, `amr_box_explode_cases`).

---

## 5. How it would be validated, in order

The point of this ordering is that each step has an EXACT expected answer, so a
failure localizes.

1. **`omega_- = omega_+` is BIT-IDENTICAL to today.** Not a tolerance. TRT
   reduces to BGK algebraically, so every gate in benchmarks/d3.json must
   return its recorded value to the last bit. This is the same shape of gate
   M4.1a and M4.2b-i used to introduce machinery against an unchanged answer
   before making it load-bearing, and it is the only step that catches a
   transcription error in the even/odd split.

2. **The duct at `Lambda = 3/16` beats the duct off it.** `duct_cases` already
   runs tau = 0.9330127 (where BGK happens to sit at `Lambda = 3/16`) next to
   tau = 0.8 and 0.6 which do not. Under TRT, tau = 0.8 and 0.6 with
   `omega_-` pinned to `Lambda = 3/16` must move TOWARD the 0.9330127 case's
   error, and their `tolerances_note` records exactly what they measure today
   (L2rel 6.3e-5 .. 1.7e-3). **This is the gate that proves `Lambda` is doing
   what the theory says**, on a case with an analytic answer and no body.

3. **The moving-body law flattens.** D3's momentum budget at tau = 0.509 is
   short by 16.6%; with `Lambda` pinned it should fall toward the tau = 0.8
   figure. `tools/probe-d3-galilean.js --budget --budgetExtra=tau=...` is the
   tool and the numbers to beat are in sec 1.

4. **The Galilean split flattens at LOW tau.** `tow_amr_cases` currently gates
   frame agreement at tau = 0.6 (measured 1.0%) precisely because tau = 0.509
   could not be gated. With TRT the tau = 0.509 pair should come inside the
   same 3%, at which point `sphere-amr-tow-Re100-D12` stops being a recorded
   regression marker and becomes a matched pair -- **re-baseline it DOWNWARD,
   which benchmarks/d3.json's `tow_amr_note` already says to do.**

5. **The card runs at its own Reynolds number.** The end of the line: n = 64,
   Re = 1100, tau = 0.5087, which is the configuration plans/3D.md M8 exists to
   reach and which sec 1 says is currently inside the bad region.

---

## 6. What TRT does NOT fix

Worth stating so it is not oversold.

- **The cubic defect.** LBM's truncated equilibrium has `O(u^3)` errors that
  act as a spurious velocity-dependent viscosity. When `nu` is small that is a
  large FRACTION of the real viscosity, so the effective Reynolds number is not
  the one requested. TRT does not touch the equilibrium.
- **Boundary-layer resolution.** `delta/D ~ Re^-1/2`. At Re = 1100 and D = 64
  the layer is ~2 cells. No collision operator helps.
- **The staircase.** A bounce-back body is still a staircase, still carries the
  standing +7..13% offset on a sphere, and a MOVING one still has a force whose
  RMS is comparable to its mean (295% measured, against the diffuse coupling's
  0.5%). TRT addresses accuracy, not noise. **This is the one place the diffuse
  interface is strictly better and TRT does not catch up.**
- **Even ghosts.** `omega_-` rescues the odd ghost moments; the even ones still
  relax at `omega_+ -> 2`. Full MRT is what fixes that half.

---

## 7. The honest alternative, and when to prefer it

plans/3D.md D4 measured the diffuse (chi) coupling as the other way out, and it
is not obviously worse:

    coupling        pinned    a=1      frame gap   force rms/mean
    bounce-back     1.3951   1.1373      -18.5%         295%
    diffuse (chi)   5.0433   5.1419       +2.0%          0.5%

Chi has no fresh nodes, no covered nodes, no staircase, and its force reads
MOMENTS -- so the entire D1/D3 class cannot arise. Its price is that the body
is the wrong SIZE, and D4's ladder measured that error converging at SECOND
ORDER in the band width to `Cd = 1.478` against bounce-back's own 1.3951 --
i.e. **the two couplings converge to the same answer**, which is what says chi
regularizes the same problem rather than solving a different one.

**So the choice is not accuracy versus inaccuracy. It is:**

- **TRT** keeps exact geometry and fixes the coupling, at the cost of a change
  to the collision operator that touches every grid transfer's tau handling.
- **chi** keeps the collision operator and fixes the coupling by not having the
  mechanism, at the cost of a tuning knob (`?chiEps=`) and a body that is only
  the right size in the limit.

They are not exclusive, and the strongest argument for doing TRT anyway is that
it also buys (a) and (b) -- the NaNs -- which chi does nothing for, and the
NaNs are what currently stop the card at Re = 500.

**That measurement has now been made** (plans/3D.md D4). chi carries the card:
at Re = 500, n = 32, both couplings close the free-fall balance to 1.00, and
narrowing the band walks the terminal velocity onto bounce-back's --
`v_x` = 0.0316 / 0.0358 / 0.0428 at band 1.5 / 0.75 / 0.375 against
bounce-back's 0.0425, i.e. **0.7% apart at the narrow band**. So the diffuse
error really is just the band, the band really is a knob, and `?levels=3` turns
it down for free.

**AND THE Re = 500 STABILITY WALL IS UNMOVED BY EITHER COUPLING**, which is
this note's sec 2(b) bearing out under measurement. Four configurations to
26000-30000 steps: today's default clean at 26000, a pre-D1 build dead at
~25000, `?swept=0` clean at 30000, and chi at band 0.375 clean at 26000. The
divergence STEP is a single draw good to a factor of two and is not a
discriminator -- but the shape is: **nothing about the boundary changes the
wall, because the wall is the bulk collision operator.**

**So the split of labour is now measured, not argued:**

    limit                            chi      TRT
    (a) ghost instability, the NaNs   no      yes
    (b) wall position / Lambda        n/a     yes
    (c) moving-body force reading    yes      yes
    staircase force noise (295%)     yes       no
    body is the right size            no      yes

chi is available today and fixes the column the card needs to MEASURE
something. TRT is the only thing on the list that lets the card RUN at its own
Reynolds number. They are complementary, and the case for doing both is
stronger than the case for either.

---

## 8. PLAN OF RECORD (2026-09-12)

Written on branch `worktree-3d-trt` against `3d/m0-m1-dense-solver` @ 070b96e,
after reading the two collision kernels, the pool's rescale, `main-3d.js`'s
tau plumbing, `benchmarks/d3.json` and plans/3D.md's M6.5 avenues. The
GOAL this plan serves is plans/3D.md M8's: **a freely falling plate that
TUMBLES at Re ~ 1100, as the 2D card does.** TRT is not a tumbling fix. It is
the change that removes the three tau-family limits standing between the
current card and the experiments that would find out why it flutters -- the
sharp plate at Re = 1100 cannot run (M6.5 avenue 5), no run can go much past
26000 steps (avenue 3), and no moving-body drag below tau ~ 0.55 can be
believed (D3). Sec 8.7 is where the tumbling question is actually asked, and
it is asked with the domain controls M6.5 says confound it, because a TRT
run at the narrow domain that still flutters would answer only one axis.

**The machine is shared -- about half a GPU is available while this runs.**
Every gate below is chosen at the smallest configuration that can make its
claim (D = 12 spheres, N = 48 ducts), legs are ordered cheapest-first so a
failure is found before an expensive run, every tool runs its legs
sequentially in ONE Chrome, and the timing milestone (8.6) is explicitly
deferred to a quiet machine because a bandwidth number taken on a shared GPU
is not a number. Costs are stated as solver steps and as a wall-clock
ESTIMATE derived from plans/3D.md sec 3.3's measured rates (~200
macro-steps/s for the card's L0 grid at full GPU, so roughly half that here);
the first leg of each milestone should print its own rate and the estimates
be corrected from it, not trusted.

### 8.0 Decisions, stated up front so they are not re-argued mid-build

1. **The delta formulation, so BGK is the zero of a knob rather than a
   separate code path.** Not sec 4's four-line even/odd form. With
   `dw = OMEGA_M - OMEGA`:

       f_minus     = 0.5 * (f[i] - f[opp[i]])
       feq_minus   = 3 * wt[i] * rho * eu                // (feq_i - feq_opp)/2
       S_minus     = 3 * wt[i] * dot(e_i, F)             // Guo's odd part
       fCollide    = f[i] - OMEGA*(f[i] - feq) + Si      // TODAY'S LINE, untouched
                     - dw * (f_minus - feq_minus)        // TRT
                     - 0.5 * dw * S_minus                // TRT forcing

   Algebraically identical to sec 3's form (the even part relaxes at
   `OMEGA`, the odd part at `OMEGA + dw = OMEGA_M`). At `dw = 0` every added
   term is an exact IEEE zero product and `x - 0 == x` exactly, so **the
   Δω = 0 build is bit-identical to BGK by arithmetic, not by tolerance**,
   whatever FMA contraction the compiler chooses for the new terms. The one
   residual risk is the compiler re-contracting the EXISTING terms because
   the expression tree grew; 8.2's first gate is a measurement of exactly
   that. The rest direction is self-opposite (`opp[0] = 0`), so its odd part
   is identically zero and needs no special case.

2. **`omega_-` is HOST-DERIVED and passed as an override**, `OMEGA_M` in the
   dense kernel and `OMEGA_M_FINE` in the pool kernel, exactly as `OMEGA` /
   `OMEGA_FINE` are today. `main-3d.js` already owns the level->tau mapping
   ("one place owns it", its own comment at the AMR pipelines); it now owns
   two: `tauAtLevel(m)` for the even rate as today, and

       tauMinusAtLevel(m) = 1/2 + LAMBDA / (tauAtLevel(m) - 1/2)

   Lambda is DIMENSIONLESS, so holding it constant across levels is the
   right invariant and needs no per-rung rule of its own. The body lives on
   the finest level (M5.4), so the finest level's Lambda is the one the wall
   sees; coarser levels' `omega_-` matters only for bulk ghost damping and
   for the transfers. At the card's target (tau_+ = 0.5087 at L0) Lambda =
   3/16 gives tau_- = 22.0 at L0 and 11.3 at L1: `omega_-` ~ 0.05-0.09, a
   SLOW odd relaxation, which is the standard regime for magic-Lambda TRT
   at low viscosity and not a mistake to be "fixed" by clamping.

3. **URL surface: `?trt=1` and `?lambda=0.1875`.** Off by default for the
   whole of this plan. `?lambda=bgk` is a sentinel that sets `OMEGA_M :=
   OMEGA` (the identical constant, not a recomputed one), which is how the
   TRT code path is exercised at an exact Δω = 0. The default-flip is a
   SEPARATE decision (8.8), after every gate is green and every recorded
   regression value is re-baselined in one commit -- half of
   benchmarks/d3.json's numbers move when Lambda moves, and the diffuse
   sphere cases, `amr-box-RB4` and `sphere-amr-tow-Re100-D12` are recorded
   values that must move deliberately, not silently.

4. **The grid transfers rescale fneq PER PARITY.** `common_d3_pool.wgsl`'s
   derivation (post-collision factor `(tau_f - 1)/(tau_c - 1)/2`) holds
   unchanged for the even part with `tau_+`, and the same derivation with
   `tau_-` holds for the odd part: Chapman-Enskog gives
   `fneq_i^- = -tau_- (d_t feq_i^- + e_i . grad feq_i^+)`, which scales
   with `tau_-` and with the gradient per cell, and collision multiplies it
   by `(1 - omega_-)`. So `dcRescaleC2F/F2C` become a PAIR of factors and
   `interpCoarseToFine3` / the average kernel split fneq by `opp[]` before
   scaling. **The odd part carries no stress**, so its own `tau_- = 1`
   singularity is handled by ZEROING the odd rescale rather than refusing
   the run (at Lambda = 3/16 that lands at tau_+ = 0.875, and a refusal there
   would be a new restriction with no physics behind it). The even part's
   tau = 1 refusal stays exactly as it is.

5. **The explode/coalesce interface (the default) gets NO change and one
   measurement.** Chen's scheme transfers post-collision populations with no
   rescale, and its consistency rests on `tau_f = 2 tau_c - 1/2` (the
   coalesce header, line 38). Under TRT the even rates still satisfy that;
   the odd rates do not, and whether that matters is not derivable -- it is
   the seam-error measurement in 8.3. If the seam signature comes back, the
   explode path needs a per-parity treatment, and that would be NEW work,
   sized then.

6. **The forcing splits by parity, prefactors `(1 - omega_+/2)` on the even
   part and `(1 - omega_-/2)` on the odd part** (Guo & Zheng's MRT forcing
   restricted to two rates). Derived, not assumed: the odd equilibrium's
   first moment is `rho u` with `u = u* + F/(2 rho)`, so `f^- - feq^-` has
   momentum `-F/2`; relaxing it at `omega_-` contributes `+omega_- F/2` and
   the odd source `(1 - omega_-/2) F`, total `F` for ANY `omega_-`. The
   momentum input is exact by construction and 8.1's `test-d3-trt.js`
   asserts it on the real lattice tables. The chi penalty force goes
   through the same source term, so chi + TRT is covered by the same line.

7. **The 2D solver is not touched**, for the reason common_d3_pool.wgsl
   already gives for its own rescale fix: `main` is the published site and
   this would move the whole 2D benchmark surface. `common_d3_spike_body.wgsl`
   stays BGK too (it exists to measure bandwidth, and 8.6 says how TRT's
   cost is priced instead).

8. **Not in scope: MRT.** Sec 6 says what TRT does not fix (the even ghosts).
   If 8.5 measures the stability wall as unmoved, MRT is the next design
   note, not a stretch goal of this one.

### 8.1 T0 -- the host statement and its unit test (no GPU)

Files: `d3-trt.mjs` (new), `tools/test-d3-trt.js` (new; picked up by
`make test` automatically).

`d3-trt.mjs` exports `tauMinusOf(tauPlus, lambda)`, `lambdaOf(tauPlus,
tauMinus)`, `tauMinusAtLevel(tau0, lambda, m)` (composed with
card-params.mjs's `tauAtLevel`, which is the existing statement of the even
ladder), a REFERENCE collision `trtCollide(f, rho, u, F, omegaP, omegaM,
lattice)` in the delta form of 8.0(1), and `dcRescalePair(tauPlusC,
tauMinusC)`. The page imports the tau functions from it; the collision is
the test's oracle only.

`test-d3-trt.js` asserts, on `lattice-3d.mjs`'s real Q19 and Q27 tables and
random populations:
- `trtCollide` at `omegaM === omegaP` equals the BGK line EXACTLY (f64,
  the delta form makes it `x - 0`).
- Mass and momentum are conserved for random `omegaM`, and with forcing the
  momentum change is exactly `F` (8.0(6)).
- The split equilibrium identities: `feq_i - feq_opp == 6 wt rho eu` and the
  even part is `wt rho (1 + 4.5 eu^2 - 1.5 u^2)`.
- MUTATION: the second moment of `f_out - feq` (the stress) is scaled by
  `(1 - omegaP)` and NOT by `(1 - omegaM)`; an odd ghost moment (any odd
  combination with zero first moment) by `(1 - omegaM)`. This is the test
  that fails when even and odd are swapped, and it is the only GPU-free
  thing that can.
- `lambdaOf(tau, tauMinusOf(tau, L)) == L` and Lambda is constant down a
  3-level ladder; `dcRescalePair` C2F followed by F2C is the identity on
  both parities.

Cost: none. Gate: `make check`.

### 8.2 T1 -- the dense kernel: `?trt=1`, and three gates that localize

Files: `shaders/common_d3_step.wgsl` (`override OMEGA_M`, the three delta
lines), `main-3d.js` (parse `trt`/`lambda`, `stepConstants.OMEGA_M`, the
status line shows `tau-` and Lambda beside `tau`, `getParams()` exposes both
so tools can print what ran), `tools/lib/d3-metrics.js` and
`tools/lib/d3-body-metrics.js` (BOTH `caseUrl`s pass `trt`/`lambda` through
-- they are two copies of one function; note it in passing, do not merge
them in this change).

Gates, in order, each with an exact expected outcome:

1. **Bit-identical at `trt=0`.** `node tools/validate-3d.js` with no extra:
   every DENSE case must reproduce its recorded digits (dense Cd is
   deterministic -- CLAUDE.md; sphere-Re20-D16 2.9436, duct/beltrami L2rel
   as printed). If any dense digit moves, 8.0(1)'s residual risk fired: put
   the delta lines behind `if (TRT != 0u)` on a `u32` override so the BGK
   expression is textually unchanged, and re-run. Cost: the standard suite;
   run `--skip=tgv` and skip the AMR groups (untouched at this milestone).
2. **Beltrami is the transcription detector.** `--cases=bel-N48-tau06
   --extra=trt=1`: at Lambda = 3/16 and tau = 0.6, `tau_- = 2.375`. The
   decay-rate error must stay inside its recorded tolerance (viscosity is
   `omega_+`'s alone); an even/odd swap puts it off by ~6x and cannot be
   missed. Then `bel-N48` (tau 0.8) and `bel-N48-tau10`. Cost: seconds each.
3. **The duct at Lambda = 3/16 lands on the magic case.** `duct-N48-tau06`
   and `duct-N48` with `--extra=trt=1`: their L2rel (recorded 1.7e-3 and
   ~6e-4 off-magic) must fall to the `duct-N48-magic` band (6.3e-5..2e-4).
   THIS is the gate that proves Lambda does what the theory says, on an
   analytic case with no body. Then pin it: add `duct-N48-tau06-trt` and
   `duct-N24-trt` cases (`trt: true, lambda: 0.1875`) with the magic case's
   `field_l2_tol: 0.002`, so the wall position is gated permanently and not
   just observed once. Cost: settle is `a^2/nu`, so tau = 0.6 is the slow
   one -- ~1 minute.
4. **Pinned spheres report, not gate.** `sphere-Re20-D16` and
   `sphere-Re200-D16` with `--extra=trt=1`: a pinned body's Cd WILL move
   (the wall moves), by an amount worth recording beside the staircase
   offset (+6.9..+12.8%). If it moves OUTSIDE the 15% tolerance, that is
   news, not a failure to hide -- record and continue.
5. **Ladder Lambda once, on the duct only**: `--extra=trt=1&lambda=` at
   1/12, 1/6, 1/4. 3/16 must be the minimum of the wall error, which is the
   cheap confirmation that the sign convention in `tauMinusOf` is right.

### 8.3 T2 -- the pool kernel and every grid transfer

Files: `shaders/common_d3_amr_step1.wgsl` (`OMEGA_M_FINE`, same three
lines), `shaders/common_d3_pool.wgsl` (`dcRescalePair`, `interpCoarseToFine3`
takes `tauMinusCoarse` and splits by `opp`), `shaders/common_d3_amr_interp.wgsl`
and `common_d3_amr_average.wgsl` (`TAU_M_COARSE` beside `TAU_COARSE`),
`main-3d.js` (every pipeline-constant site that passes `TAU_COARSE` or
`OMEGA_FINE` -- the depth-2 pair near lines 1370-1416 and the per-level
loop near 1704-1788 -- passes the odd twin; there are SEVEN such sites and
missing one is the 238e48c shape, so grep for `TAU_COARSE` when done and
count).

Gates:
1. `amr-all-RB4/RB8/Q27` and `amr-box-RB4` at `trt=0`: recorded digits. As
   CLAUDE.md warns, AMR Cd reproduces only to ~1e-3, so the sphere-amr
   cases are checked against tolerance, the field/conservation cases
   against their recorded digits.
2. `amr_conserve_cases` with `--extra=trt=1`: mass and momentum drift must
   stay inside the SAME bounds -- the collision conserves both for any
   `omega_-` (8.0(6)) and the explode path has no rescale, so nothing here
   may move by more than the floor. A momentum leak with mass exact is the
   old signature (`amr_interface_note`) and would say the odd transfer is
   wrong.
3. `amr_box_explode_cases` (field, depth 2 and 3) with `trt=1`, then
   `node tools/analyze-d3-interface.js --extra=trt=1`: the seam error must
   track the `refine=all` control within the same 8-10%. This is 8.0(5)'s
   measurement. If the ratio rises, stop and size the per-parity explode
   treatment before continuing; do not widen the bound.
4. `sphere_amr_cases` (pinned, Cd matches dense to 0.5%) with `trt=1` --
   the dense and pool paths must move TOGETHER under TRT. And
   `sphere-amr-dynamic-*`'s exact-equality gate, which does not know tau
   exists and must not start to.
5. `node tools/validate-d3-invariants.js` once, as a boot smoke: nothing
   structural reads tau.

Cost: the AMR groups of validate-3d, ~10 cases at N = 48 / D = 16, a few
minutes total; `analyze-d3-interface.js` a few more.

### 8.4 T3 -- THE CLAIM: does the moving-body law flatten?

This is what the whole note was written for, and it is a measurement with
two admissible outcomes. Tool: `tools/probe-d3-galilean.js`, unchanged
(`--extra`/`--budgetExtra` already exist).

1. **The momentum budget.** `--budget` at tau = 0.509 / 0.55 / 0.6 / 0.8,
   BGK and `trt=1`. BGK reproduces sec 1's 16.6 / 3.6 / 1.5 / 0.0%. Under
   TRT the deficit at 0.509 should fall toward the 0.8 figure. Same leg at a
   SECOND U and a SECOND D (plans/3D.md sec 8 item 1 asks for this before
   the 1/(tau-1/2) law is leaned on; the drift box at D = 12 is cheap enough
   that it costs nothing to do it here). Cost: `drift` is a closed box and
   the tool prints max|u| and the density range beside the closure for that
   reason -- short windows, ~20 s per point.
2. **The Galilean split at the bad tau.** The sphere sweep at tau = 0.509,
   dense then `levels=2`: BGK reads -18.5% and +47.3% across the frame;
   TRT should bring both inside the ~1% the tau = 0.6 pair already shows.
   Enstrophy across the frame must remain 0.3%-flat in both builds -- that
   is the check that the FLOW did not change, only the reading.
3. **Then pin whichever way it went.**
   - Flattens: add `sphere-amr-tow-Re100-D12-trt` as a MATCHED PAIR with a
     new `sphere-amr-pin-Re100-D12-trt` (`cd_matches_tol: 0.03`, the tau =
     0.6 pair's shape), and leave `sphere-amr-tow-Re100-D12` recording the
     BGK defect until 8.8 retires it. That is `tow_amr_note`'s own
     instruction ("re-baseline downward ... converted to a matched pair").
   - Does not flatten: the hypothesis in sec 2 (odd ghost content in the
     force reading) is wrong or insufficient, and the defect lives in the
     momentum-exchange formula on a MOVING staircase rather than in Lambda.
     Record it in plans/3D.md D3, keep chi as the card's coupling (D4), and
     continue -- 8.5 and 8.7 do not depend on this outcome.

### 8.5 T4 -- stability: where does the wall move?

Tool: `tools/probe-d3-tau.js --extra=trt=1`, which already prints the
lowest surviving tau per config against M8.0's recorded BGK table
(beltrami 0.50300, tgv 0.50218, duct 0.50050, sphere-D16 0.50218 dense /
0.50150 refined). Sec 2(b) predicts the odd ghosts stop being the limit;
sec 6 says the even ghosts still relax at `omega_+ -> 2`, so the honest
expectation is "lower, not gone".

1. The four bodiless/pinned rows, one Lambda (3/16). This is the expensive
   probe of the plan short of the card: 20000 steps per tau, ~7 taus per
   config -- run `--configs=sphere-D16` first (it is the row the target
   resembles), the others after. Estimate 10-20 min per config at half GPU.
2. Lambda sweep on `sphere-D16` only: 1/4 is the literature's stability
   optimum (Ginzburg/d'Humieres/Kuzmin), 3/16 the wall optimum. If 1/4
   survives a lower tau than 3/16, record the trade; the default stays 3/16
   unless the card in 8.7 needs the margin.
3. The two recorded blowups, both ways: the card at Re = 500 dense (dies at
   ~20000 steps, d3-scenarios.mjs's card header) and D2 (rotation free,
   AMR, ~9000). A single divergence step is a draw good to a factor of two
   (plans/3D.md), so the claim is "survives 3x the recorded step", not "dies
   later". ~26000 and ~30000 steps on the card at n = 32 -- a few minutes
   each.

### 8.6 T5 -- cost, DEFERRED to a quiet GPU

`tools/bench-d3-interface.js` prices a change by differencing two runs of
one build and reports the MINIMUM of its reps; on a shared GPU that minimum
is whatever the other job's idle moments allowed, i.e. not a measurement.
So this milestone is a one-liner that runs when the machine is free: BGK
against `trt=1` on the body-fitted shell config, expecting inside the
tool's own ~10% noise floor (sec 4: arithmetic only, no traffic). Not a
blocker for anything above or below. If someone wants the register/ceiling
verdict too, `common_d3_spike_body.wgsl` takes the same three lines and
`spike-d3-registers.js` prints it -- optional.

### 8.7 T6 -- THE CARD: the sharp plate at Re = 1100, and the tumbling matrix

Tool: `tools/probe-d3-tumble.js` (its BASE is already the closest-to-tumbling
configuration: n = 32, span 2, Re = 1100, tilt 1.047, levels 2, dynamic).
Read the SERIES, not the end: tumbling is `net` growing without bound;
flutter is `net` bounded in 0.05..0.31 rev while `arc` climbs. Each leg is
26000 steps -- estimate 3-5 minutes at half GPU from sec 3.3's rate, to be
corrected from the first leg's printout.

Legs, in this order, each answering one question:

1. **`--extra=bounceback=1&trt=1`.** The exact-thickness plate at the
   target Re, which `?bounceback=1` alone could not run (blew up at step
   5000, |v| pinned at v_max). Claim A: it runs 26000 steps. Claim B, the
   one that matters: does `net` grow? M6.5 named the plate's EFFECTIVE
   THICKNESS (chi band + 4 cells = aspect ~0.17 against 0.125) as the
   leading non-domain suspect, and this is the first configuration that
   can test it at the target Re.
2. **`--extra=trt=1`** (chi, the default coupling). Control: chi's force
   reads moments, so TRT should change its flutter only through bulk
   stability. If leg 2 tumbles where BGK-chi did not, the tau wall was
   truncating the run, not the coupling.
3. **Length.** Whichever of 1/2 survives, at `--steps=100000` (avenue 3: 26000
   steps is ~8 flutter periods and a long transient looks exactly like a
   limit cycle). ~15-20 minutes each at half GPU; this is the one place the
   shared machine really costs, so run it once, on the best leg.
4. **The domain controls, because M6.5 says they confound everything
   above.** Two small additions to `d3-scenarios.mjs`'s `card`: a lateral
   dims override (`?wide=1` gives the prescribed leg's `[8n, 8n, ...]`,
   1.56% blockage against the free-fall box's 5-6.67%) and a spanwise-fill
   option (`?spanfill=1` makes z = 2*span*n exactly, so the periodic domain
   removes the tips). They pull opposite ways on one axis, so they are two
   legs, not one -- run each with the better of legs 1/2. `?wide=1` is ~2x
   the cells (256x256x192), so ~2x the time. If a narrow-domain leg
   flutters and the wide one tumbles, TRT was necessary and the domain was
   sufficient; if neither tumbles, the remaining axis is I*-Re (avenue 4)
   and the answer to "does TRT buy tumbling" is no, and recorded.
5. **`--extra=trt=1&levels=3`.** Chi at band 0.375 (D4's shipping
   combination) with the body's level at tau_+ = 0.535 and Lambda pinned.
   The configuration closest to what the 2D card actually runs
   (levels = 3, tau0 = 0.50873). Depth 3 doubles the staging-buffer
   window (CLAUDE.md), which is fixed, but watch for it.
6. One clip, `tools/render-d3-movie.js`, of whichever leg tumbles or comes
   closest -- for the eye, not as evidence.

What is deliberately NOT in this matrix: i_star and span sweeps (done, six
levers, one answer -- M6.5), the release angle (flutter is an attractor from
0.15 and from 1.047), and a resolution ladder (the Galilean split is the
instrument for moving-body claims, and 8.4 already used it).

### 8.8 The default flip -- a separate decision, after all of the above

Only when 8.2-8.4 are green: flip `trt` on by default, set every case in
benchmarks/d3.json that records a BGK number (the diffuse spheres,
`amr-box-RB4`, `sphere-amr-tow-Re100-D12`, the duct/beltrami tolerances)
to its TRT value IN ONE COMMIT with the re-baseline stated case by case, and
retire the `-trt` twins added in 8.2/8.4 by making them the cases. Keep
`?trt=0` live as the A/B, the `?dcpre=` / `?ghostcopy=` arrangement. The
2D solver stays as it is. Update CLAUDE.md's 3D section and plans/3D.md's
sec 8 list (item 1 is this).

### 8.9 Cost summary and stop conditions

    milestone   GPU                       est. at half GPU   stops the plan if
    8.1 T0      none                      --                 --
    8.2 T1      dense suite + 3 extras    ~10 min            dense digits move at trt=0 and the
                                                             TRT-guard arrangement does not fix it
    8.3 T2      AMR groups + interface    ~15 min            seam ratio rises under trt=1 (8.0(5))
    8.4 T3      drift budget + splits     ~10 min            never -- either outcome is recorded
    8.5 T4      tau ladders               ~1 h (4 configs)   never -- reports
    8.6 T5      bench, quiet GPU only     ~5 min, later      cost > ~10% (would be a surprise;
                                                             then find the spill)
    8.7 T6      6+ card legs              ~1-2 h             never -- this IS the question

The order is the order: 8.4 cannot be interpreted without 8.3's seam
measurement (a pool-path Cd is meaningless if the interface is leaking), and
8.7's legs 1-3 need 8.5's answer to know whether a survival is TRT or luck.
Nothing after 8.3 blocks on 8.6.

### 8.10 T1 RESULTS (2026-09-12) -- the dense kernel is in, and it found a leak

`?trt=1&lambda=` on the dense kernel, delta form, `OMEGA_M` host-derived.
Every gate in 8.2 ran, on my own ports (4445/9444; the other session's
server on 4444 serves ITS checkout, not this one -- a worktree needs its own
`https.py`).

1. **Bit-identical at `trt=0`, and at `?lambda=bgk`.** A reference tree
   (the T0 commit) served on a third port, the same four cases run on
   both, detail blocks diffed: identical to every printed digit (Q19 and
   Q27, duct, Beltrami, sphere). The sentinel path reproduces `trt=0` the
   same way. The compiler did not re-contract the BGK line.

2. **Beltrami keeps its viscosity.** Decay-rate error under `trt=1` at
   Lambda = 3/16: -1.8e-4 (tau 0.8), -5.5e-4 (tau 0.6, where tau_- = 2.375),
   +7.3e-4 (tau 1.0). Tolerance 5e-3. The field L2rel IMPROVED (3.7e-3 ->
   1.5e-3 at tau 0.8). No even/odd swap.

3. **The duct did NOT land on the magic band at first, and the reason was
   not TRT.** tau = 0.6, N = 48: BGK 1.595e-3 -> TRT 1.421e-3, barely
   moved, and 3x the settle time DOUBLED it to 3.06e-3. The velocity was
   falling linearly in time. `readStats` said why: **mass rose at 2.6e-8
   per step per cell, uniformly, with the momentum settled**, on BGK and
   TRT alike, on Beltrami too. An f32-emulated 1D channel reproduced the
   rate (2.69e-8) and a term-by-term decomposition put it in `sum(feq) -
   rho`, not the polynomial: **the rounded weights sum to 1 + 1.49e-8**, and
   `omega * rho * 1.49e-8` is 2.5e-8 at tau = 0.6. Fixed in
   lattice-3d.mjs (sum-exact weights, commit 69feb72): duct drift 2e-9/step,
   Beltrami -3e-10. THEN the gate reads as the theory says:

       duct              BGK        TRT Lambda=3/16
       N48 magic         1.02e-4    1.02e-4   (identical; omega_- == omega_+)
       N48 tau 0.8       4.28e-4    1.36e-4
       N48 tau 0.6       8.41e-4    4.50e-4   (x36 settle: 3.96e-4 -- stationary)
       N24 tau 0.8       1.70e-3    4.19e-4
       N24 tau 0.6       2.84e-3    4.79e-4

   Lambda ladder, N48 tau 0.6: 6.35e-4 / 4.81e-4 / **4.50e-4** / 4.72e-4 /
   1.11e-3 at 1/12, 1/6, 3/16, 1/4, 1/2. The minimum is at 3/16, so the
   sign convention is right, and the ~4.5e-4 floor there is Lambda-flat --
   not wall position, and not this plan's business. Pinned as
   `duct-N24-tau06-trt`, `duct-N48-tau06-trt` and a `-x36` twin that guards
   against the drift returning (tol 1e-3, ~2x measured).

   Two lessons for the record. **A gate whose error grows with run length
   is measuring a drift, not a discretization**, and `settle_multiple` was
   hiding one in plain sight because every case ran the same multiple.
   And **an f32 emulation can rank rounding sources but not predict the net
   rate**: two emulations differing only in whether a 1e-10 `u_y` entered
   `e.u` gave 2.7e-8 and 0.7e-8 for the same weights. The analytic bias
   was the robust number; the GPU was the instrument.

4. **Pinned spheres move TOWARD Schiller-Naumann.** sphere-Re20-D16
   2.9436 (+12.8%) -> 2.8772 (+10.3%); sphere-Re200-D16 at tau = 0.512
   0.8622 (+7.0%) -> 0.8386 (+4.1%). The staircase is unchanged; the wall
   inside it sits closer to where it should. Not a gate; recorded.

5. Every dense case green on the fixed weights, sphere Cd unchanged at
   the printed digits (Re100: 1.1924 -> 1.1922).

Not done in T1: the cost (8.6, quiet GPU) and the spike kernel. Next is
8.3, the pool kernel and the per-parity transfers.

### 8.11 T2 RESULTS (2026-09-13) -- the pool kernel and the per-parity transfers

`OMEGA_M_FINE` on the pool step; `dcRescaleOddC2F/F2C` in common_d3_pool.wgsl
with the odd taus HOST-DERIVED (`tauMinusLevel`, `trtXfer(m)` beside every
`TAU_COARSE`; 8 sites, not the 7 estimated); interp and average split fneq
by `opp` and apply the odd correction as a delta that is exactly zero with
`TRT = 0`. A closed form fell out that the shader does not need but the
reader does: with Lambda held, **tau_- HALVES its distance from 1/2 per rung
while tau_+ doubles it**.

1. **Bit-identical at `trt=0` at depth.** All 17 AMR cases (all/box/bar,
   explode conservation and field at depth 2 and 3, the four sphere-amr,
   the three tow-amr) diffed against a T1-commit reference tree: identical
   to every printed digit -- including the pool-path Cd, which reproduced
   exactly on this same-build repeat.

2. **Conservation holds; the explode path's momentum leak rises 4-8x and
   stays 20-50x under the bound.** Mass unchanged. Momentum bar/box/bar-L3/
   box-L3: 1.85e-3 / 9.3e-4 / 2.50e-3 / 1.31e-3 against BGK's 2.6e-4 /
   2.2e-4 / 3.1e-4 / 2.0e-4, bound 0.05. That is 8.0(5)'s prediction: the
   two levels' odd rates are not in the 2tau-1/2 relation the even ones
   are, and explode/coalesce does not rescale. Recorded in
   amr_conserve_note; a per-parity explode treatment is the fix if it ever
   matters, and it does not yet.

3. **No seam signature.** amr-box-explode-field 2.655e-3 -> 2.589e-3,
   L3 6.134e-3 -> 6.307e-3, both inside bounds; analyze-d3-interface's
   bucket-0 error sits between its neighbours at every checkpoint.

4. **THE 0.06% CROSS-LEVEL "INVARIANCE" WAS A COINCIDENCE.** Under TRT the
   pinned pair reads dense D16 2.8772 vs AMR 2.9185, +1.4%, failing the
   0.5% bound. But TRT moved each level by an amount monotone in how far
   BGK's Lambda sat from 3/16: dense D16 (tau 0.62, Lambda 0.014) -2.3%,
   L1 D32 (0.74, 0.058) -0.8%, L2 D64 (0.98, 0.23) -0.1%. BGK's
   tau-dependent wall offsets had made a D16 dense body agree with a D32
   refined one. Measured against the honest control, dense AT THE SAME
   FINEST RESOLUTION (`sphere-Re20-D32`, added): BGK 2.9072 vs AMR 2.9417
   (+1.2%), TRT 2.8924 vs 2.9185 (+0.9%). The pool path under TRT is as
   consistent with dense as under BGK, slightly more so, and the ~1% is
   the coarser far field. The case's `cd_matches` should move to D32 at
   8.8.

5. **The tau = 0.6 tow pair passes at +2.0%** (bound 3%; BGK +1.0%). The
   dynamic exact-equality gates pass. `validate-d3-invariants` untouched.

6. **THE tau = 0.509 TOWED BODY READS Cd 5.23 UNDER TRT, against BGK's 1.91
   and a pinned ~1.30.** Four times the pinned leg. L1's tau_- there is
   10.9, omega_- = 0.09. This is 8.4's question and it arrived early; it
   is NOT a transfer bug (the same machinery passes at tau = 0.6, and the
   pinned dense sphere at tau 0.512 improved under TRT). Note for 8.4:
   sec 3's "omega_- stays O(1) as omega_+ -> 2" is WRONG at fixed Lambda --
   omega_- = (tau_+ - 1/2)/Lambda -> 0. The odd ghosts are damped
   MONOTONICALLY (factor 1 - omega_- ~ 0.91/step) instead of BGK's
   sign-alternating 1 - omega ~ -0.93/step; either way slowly. Whether that
   is what the momentum-exchange reading sees is the measurement to make.

### 8.12 T3 RESULTS (2026-09-13) -- THE MOVING-BODY LAW DOES NOT FLATTEN. IT DOUBLES.

`tools/probe-d3-galilean.js --budget` on the `drift` box (96x48x48, periodic,
no sponge, D = 12, U = 0.025, 300 steps after a 300-step warm-up), where the
fluid's momentum can only change through the body so `sum F + dP` is the
force the kernel is not measuring:

    tau       BGK short by    TRT (Lambda = 3/16) short by
    0.509        16.6%               36.2%
    0.55          3.6%                7.4%
    0.6           1.5%                3.2%
    0.8           0.0%                0.3%

TRT roughly DOUBLES the deficit at every tau and the 1/(tau - 1/2) shape
survives. The dense Galilean split at tau = 0.509 says the same thing from
the other side: Cd falls -37.5% from the pinned leg to a = 1 under TRT
against BGK's -18.5%, linearly in a (1.2989 / 1.1938 / 1.0822 / 0.8828 /
0.8120 at a = 0 / 0.25 / 0.5 / 0.75 / 1), and the pool at tau = 0.509 reads
Cd 5.23 towed against ~1.3 pinned (8.11 item 6). **Sec 2's hypothesis --
that the missing force is odd-ghost content that Lambda would damp -- is
refuted**, which is the second of the two outcomes 8.4 pre-committed to.

What the numbers say instead. At fixed Lambda, omega_- = (tau_+ - 1/2)/Lambda
goes to ZERO as tau_+ -> 1/2 (sec 3's "stays O(1)" was wrong; it holds for a
fixed tau_-, not a fixed Lambda). So the odd non-equilibrium near the body is
~tau_- times LARGER under TRT than under BGK -- 22x at the card's tau -- and
whatever the moving-body coupling fails to account for scales with it. The
deficit doubling rather than growing 22x says the missing term is not simply
proportional to fneq^-, but its direction is unambiguous: the defect lives in
the momentum-exchange coupling on a moving staircase (the swept and fresh
cells, D1's territory), not in the collision operator, and no choice of
Lambda reaches it. The PINNED body improves under TRT at every tau measured
(sphere-Re200-D16 at tau 0.512: +7.0% -> +4.1% over S-N; the D = 12 pinned leg
here: +28% -> +19%) -- so TRT buys the STATIC wall exactly as advertised and
the moving one not at all.

Consequences for the rest of this plan:
- `sphere-amr-tow-Re100-D12` stays pinned to its BGK value. No matched pair.
- The card's coupling stays chi (D4), where the force reads moments and the
  frame gap was 2.0% at this tau. TRT + bounce-back on a moving body at low
  tau is quantitatively WORSE than BGK + bounce-back, so 8.7 leg 1 (the sharp
  plate at Re = 1100 under TRT) can answer "does it run" but not "what is its
  drag", and it is demoted below the chi legs.
- What TRT can still buy is 8.5: the bulk stability wall. That is measured
  next, and it is now the whole case for the operator on the card.
- plans/3D.md sec 8 item 1 ("TRT ... fixes Lambda independently of tau and is
  the standard remedy" for D3) is answered in the negative and should be
  rewritten when this branch merges.

### 8.13 T4 RESULTS (2026-09-13) -- TRT AT Lambda = 3/16 LOWERS THE STABILITY WALL

`tools/probe-d3-tau.js --extra=trt=1`, 20000 steps per point, against M8.0's
BGK table:

    config        BGK lowest surviving     TRT Lambda=3/16
    sphere-D16    0.50218                  NaN @ 1000 at 0.50218 (and at every Lambda tried:
                                           1/4, 0.05, 0.01, 0.002 -- all dead by step 3000)
    beltrami      0.50300                  NaN @ 2000 at 0.504, at 0.503, and below
    tgv           0.50218                  NaN @ 2000 at 0.504 and below

TRT is LESS stable than BGK on every row, bodiless or not. The hotspot on the
sphere is at the surface (sdf = -0.1), on Beltrami in the bulk, so it is not
the wall; and **the f64 host reference collision reproduces it** (24^3
Beltrami at tau_+ = 0.503, tau_- = 1.5: NaN at step 1525 while BGK runs
clean), so it is the scheme and not the shader or f32.

**The boundary, measured on Beltrami N = 48 at tau_+ = 0.503, u0 = 0.04
(peak Ma 0.17), tau_- swept through Lambda:**

    tau_-    0.503(bgk)  0.51   0.52   0.6    0.65   0.7    0.8    0.9    1.0    1.1   1.5   3.8   10   34   63
    outcome  ok          ok     ok     ok     ok     ok     NaN    NaN    NaN    NaN   NaN   NaN   NaN  NaN  NaN
                                                            @3000  @2000  @2000  @1000

So at this tau_+ the odd sector must be OVER-relaxed nearly as hard as the
even one: tau_- <= ~0.75, omega_- >= ~1.33, Lambda <= ~7e-4 -- 270x below
3/16. It is not a rule about Lambda and not a rule about tau_- alone (the
same flow at tau_+ = 0.6 ran its whole gate at tau_- = 2.375): it is a joint
constraint that bites as omega_+ -> 2, which is precisely where the plan
wanted TRT. **And it is a Mach effect**: at tau_+ = 0.503 with Lambda = 3/16,
u0 = 0.01 (peak Ma 0.04) survives 20000 steps decaying cleanly, u0 = 0.02
(Ma 0.08) grows from step 8000 and dies by 10000, u0 = 0.04 dies by 2000.
The card runs u_t = 0.05 with peaks of 2-3x.

Sec 2(b) had it backwards for this regime. BGK's odd ghosts at omega -> 2
are marginal and SIGN-ALTERNATING; TRT's at omega_- -> 0 are marginal and
MONOTONE, and a persistent non-alternating odd ghost couples coherently
into the u^2 terms of the equilibrium where an alternating one averages
out. That is a hypothesis with one measured consequence (the Mach
dependence) and no linear analysis behind it; the measurement is the
result. Literature stability optima for TRT (Lambda = 1/4) are
advection-diffusion results at vanishing Mach and do not transfer.

### 8.14 VERDICT, and what remains of the plan

Three things were asked of TRT (sec 7's table). Measured:

    limit                            chi     TRT, measured
    (a) ghost instability, the NaNs   no     WORSE at the Mach this project runs (8.13)
    (b) wall position / Lambda        n/a    YES -- duct 3-6x, pinned sphere +7% -> +4% (8.10)
    (c) moving-body force reading    yes     WORSE, deficit doubles (8.12)

**TRT does not help the card.** It fixes the one thing that was never the
card's problem (a pinned straight wall) and worsens both things that were.
The operator stays in the build, off by default, as the A/B it now is; it is
not a candidate for the default flip (8.8 is withdrawn), and 8.6 (cost) and
8.7 legs 1-3 (the card under TRT) are not worth GPU time. What the TRT work
DID buy, and keeps:

- **The f32 lattice weights now sum to exactly 1** (commit 69feb72). Every
  3D collision was injecting omega * rho * 1.49e-8 of mass per step; the
  tau = 0.6 duct gate's "wall error" was 80% drift. That is the solver
  finding of this branch, and it found itself only because the duct gate
  refused to behave as the theory said.
- **`sphere-Re20-D32`**, the honest control for the cross-level claim; the
  recorded 0.06% invariance was a coincidence of BGK's tau-dependent wall
  offsets (8.11).
- The pinned static-wall improvement is real, and a static-body user of
  this solver (the duct and sphere gates) could take it. It is not this
  project's problem.
- The delta-form TRT with host-derived per-level odd taus, the per-parity
  transfers and the unit test are the cheapest possible substrate for a
  MRT (or a regularized/entropic collision) if the stability wall is ever
  attacked at the collision operator again -- and 8.13's table is the
  first thing such an attempt should reproduce.

**For the tumbling question itself, the operator was never the axis.** The
levers that remain are plans/3D.md M6.5's avenues 1 and 2 -- the free-fall
domain is narrow (5-6.7% blockage against 1.56%) and the plate has tips
where Pesavento & Wang's does not -- both scenario changes, both cheap, both
confounding every sweep to date. That is where the next GPU hour goes.
