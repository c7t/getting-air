# TRT: fixing the tau family at its root

**Status: a design note, not a plan of record. Nothing here is built.** Written
2026-09-11 out of plans/3D.md's D1/D3/D4, which measured the problem TRT exists
to solve and then measured the alternative (a diffuse interface) that avoids it
instead.

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
