# Implementation Plan: LBM Moving Window with ALBC Sponge

## Objective
Implement a moving window simulation using a circular buffer to allow the falling card to "pan" seamlessly through an infinite space of still air. To prevent acoustic shocks at the boundaries, an Absorbing Layer Boundary Condition (ALBC) "sponge" will be applied to all four sides of the domain.

## Key Files & Context
- `main.js`: CPU-side buffer management and initialization.
- `shaders/physics.wgsl`: Card state integration and grid shift logic.
- `shaders/lbm_collide.wgsl`: LBM collision step and ALBC sponge application.
- `shaders/lbm_stream.wgsl`: LBM streaming step.
- `shaders/lbm_force.wgsl`: Penalty force calculation.
- `shaders/render.wgsl`: Visualization.

## Implementation Steps

This implementation is designed to be progressive, allowing for testing in isolation after each phase. **The agent will pause execution after each phase to allow the user to manually verify the simulation before proceeding to the next phase.**

### Phase 1: Foundation (No visual change expected)
1. **Update `CardState` Buffer (CPU & GPU)**
   - Expand `CardState` by 4 floats (16 bytes) to include:
     `off_x: f32`, `off_y: f32`, `off_x_old: f32`, `off_y_old: f32`.
   - Update `main.js` to initialize these to `0` and increase buffer sizes to 104 bytes.
   - Update `CardState` struct in all `.wgsl` files.
   - **Verification:** Run the simulation. It should execute identically to the current version with no errors or visual changes.

### Phase 2: Absorbing Boundaries (Wake absorption expected)
2. **Implement ALBC Sponge in `lbm_collide.wgsl`**
   - Define a sponge width (e.g., `SPONGE_W = 64.0`).
   - Calculate a blending weight `W(wx, wy)` that is 0 in the center and ramps up to 1 at all four boundaries using a smooth cubic function.
   - After calculating the post-collision distribution `f_col`, blend it with the target resting state:
     ```wgsl
     let target_f = wt[i] * 1.0; // rho=1.0, u=0
     f_col_final = mix(f_col_computed, target_f, W(wx, wy));
     ```
   - **Verification:** Run the simulation. The falling card will eventually hit the bottom sponge and its wake will hit the top sponge. Instead of wrapping around (periodic boundary), the fluid activity should smoothly dissipate at the edges.

### Phase 3: Circular Buffer Plumbing (No visual change expected)
3. **Implement Circular Buffer Indexing in All Shaders**
   - Introduce a mapping from **window coordinates** `(wx, wy)` to **buffer coordinates** `(bx, by)`:
     ```wgsl
     // Example logic (actual implementation will need carefully placed u32 casts)
     let bx = (wx + u32(state.off_x)) % W;
     let by = (wy + u32(state.off_y)) % H;
     ```
   - Apply this mapping to:
     - `lbm_collide.wgsl`: Read/write `f_in`, `f_col`, `vel`.
     - `lbm_stream.wgsl`: Read `f_col` (source), write `f_out` (destination). Both source and destination coordinates must be mapped to buffer coordinates.
     - `lbm_force.wgsl`: Read `f_in`.
     - `render.wgsl`: Read `vel`.
   - **Verification:** Run the simulation. Since `off_x` and `off_y` are still exactly 0, the indexing math should result in identical memory access patterns. The visual result should perfectly match Phase 2.

### Phase 4: Activation (Moving Window expected)
4. **Update `physics.wgsl` (Moving Window Logic)**
   - Change position integration to track absolute position (`x_total`, `y_total`).
   - Calculate the target grid shift to keep the card centered near its initial position (e.g., `cx ≈ W/2`, `cy ≈ H/4`).
   - Update `off_x` and `off_y` based on this shift (accumulating the integer displacement).
   - Calculate `state.cx` and `state.cy` as the sub-pixel window coordinates (relative to the shifted grid) so the visualization and force calculations remain stable.
   - **Verification:** Run the simulation. The card should appear to stay roughly in the upper middle of the screen while the surrounding still air is continuously shifted into view and the wake flows "upwards" and out of the domain, absorbed silently by the sponge.

## Verification & Testing
- Use the progressive verifications listed in the steps above.
- Download the trajectory CSV to ensure physical position tracking (`y_total`, `x_total`) remains correct and continuous.