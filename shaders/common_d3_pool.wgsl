// Octree-pool addressing and the coarse->fine interpolation core, shared by
// the 3D AMR kernels. plans/3D.md M3. Fragment only.
//
// Host counterpart: d3-amr.mjs, whose tools/test-d3-amr.js checks the same
// mapping against an INDEPENDENT route (global fine coordinates, where
// ownership is one division with no ring and no offsets). The arithmetic
// here mirrors that module; a disagreement is a seam nobody would see.
//
// THE RING IS KEPT and GHOST is 2 -- plans/3D.md sec 2.1, settled. Two ring
// layers are load-bearing, not slack: each level takes two substeps per
// parent step with no ring refresh between them, so substep A's gather on a
// depth-1 ring cell reaches into depth 2, leaving depth 1 valid for B.

const GHOST : u32 = 2u;

override RB : u32 = 4u;          // coarse cells per block edge
override NX : u32;               // L0 dims
override NY : u32;
override NZ : u32;

fn poolFB() -> u32 { return 2u * RB + 2u * GHOST; }
fn nbx() -> u32 { return NX / RB; }
fn nby() -> u32 { return NY / RB; }
fn nbz() -> u32 { return NZ / RB; }

fn blockIdOf(b: vec3<u32>) -> u32 { return (b.z * nby() + b.y) * nbx() + b.x; }
fn blockXYZ(id: u32) -> vec3<u32> {
  return vec3<u32>(id % nbx(), (id / nbx()) % nby(), id / (nbx() * nby()));
}

fn poolCell(slot: u32, f: vec3<u32>) -> u32 {
  let FB = poolFB();
  return slot * FB * FB * FB + (f.z * FB + f.y) * FB + f.x;
}

fn coarseCell(c: vec3<u32>) -> u32 { return (c.z * NY + c.y) * NX + c.x; }

// Cell-centred refinement: the two children of coarse cell `origin` sit at
// origin -+ 1/4, NOT at origin and origin + 1/2. Signed index so ring cells
// continue the same line on both sides.
fn fineToCoarseUnit3(j: i32, origin: u32) -> f32 {
  return f32(origin) - 0.25f + 0.5f * f32(j - i32(GHOST));
}

fn isInteriorIdx(j: u32) -> bool { return j >= GHOST && j < GHOST + 2u * RB; }
fn isInterior3(f: vec3<u32>) -> bool {
  return isInteriorIdx(f.x) && isInteriorIdx(f.y) && isInteriorIdx(f.z);
}

fn wrapu(v: i32, n: u32) -> u32 { return u32(((v % i32(n)) + i32(n)) % i32(n)); }

// --- GLOBAL FINE coordinates ----------------------------------------------
//
// The frame d3-amr.mjs calls GLOBAL FINE: integer fine cell over the whole
// domain, [0, 2*NX) etc., where "which tile owns this cell" is one division
// with no ring and no offsets. Everything that has to reason about a cell
// belonging to one tile while being addressed through ANOTHER -- a ring cell
// is exactly that -- goes through here rather than doing its own arithmetic.

fn fineDim() -> vec3<i32> { return vec3<i32>(i32(2u * NX), i32(2u * NY), i32(2u * NZ)); }

fn wrapFine3(g: vec3<i32>) -> vec3<i32> {
  let n = fineDim();
  return vec3<i32>(
    ((g.x % n.x) + n.x) % n.x,
    ((g.y % n.y) + n.y) % n.y,
    ((g.z % n.z) + n.z) % n.z);
}

fn blockOfFine(g: vec3<i32>) -> vec3<u32> {
  let RB2 = 2u * RB;
  return vec3<u32>(u32(g.x) / RB2, u32(g.y) / RB2, u32(g.z) / RB2);
}

// The coarse cell a global fine cell sits in. Cell-centred refinement puts
// fine cells 2c and 2c+1 inside coarse cell c, so this is one shift.
fn coarseOfFine(g: vec3<i32>) -> vec3<u32> {
  return vec3<u32>(u32(g.x) / 2u, u32(g.y) / 2u, u32(g.z) / 2u);
}

// Tile-local index of GLOBAL FINE cell g inside the tile owning block b --
// which need NOT be g's own block. Callers use this where g is at most one
// fine cell outside b's interior (a depth-1 ring cell of it) or inside it, so
// the result always lands in [0, FB). The centred wrap is what makes it
// periodic, and it needs at least 3 blocks on the axis; main-3d.js checks
// that before enabling any path that calls this.
fn localInTile(g: i32, b: u32, nFine: i32) -> i32 {
  let RB2 = i32(2u * RB);
  var d = g - i32(b) * RB2;
  if (d < -nFine / 2) { d += nFine; }
  if (d >  nFine / 2) { d -= nFine; }
  return d + i32(GHOST);
}

fn localInTile3(g: vec3<i32>, b: vec3<u32>) -> vec3<u32> {
  let n = fineDim();
  return vec3<u32>(
    u32(localInTile(g.x, b.x, n.x)),
    u32(localInTile(g.y, b.y, n.y)),
    u32(localInTile(g.z, b.z, n.z)));
}

// The block a coarse cell belongs to.
fn blockOfCoarse(c: vec3<u32>) -> vec3<u32> {
  return vec3<u32>(c.x / RB, c.y / RB, c.z / RB);
}

// --- coarse -> fine interpolation core ------------------------------------
//
// The 3D counterpart of common_interp.wgsl. rho/u are blended and fed to
// feq, and only the NON-EQUILIBRIUM part is interpolated and rescaled --
// which is what makes the Dupuis-Chopard factor applicable at all.

struct CoarseSample3 {
  rho : f32,
  u   : vec3<f32>,
  fneq: array<f32, QN>,
}

// --- the Dupuis-Chopard non-equilibrium rescale ---------------------------
//
// THE STORED POPULATIONS ARE POST-COLLISION, and that changes the factor.
// common_d3_step.wgsl and common_d3_amr_step1.wgsl are both fused
// pull-stream + collide kernels: what lands in the buffer is f AFTER
// collision, which is exactly what makes the next step's gather a pure
// stream. Every grid transfer therefore reads post-collision f and must
// write post-collision f.
//
// The textbook statement is about PRE-collision populations:
//
//   fneq_pre,f / fneq_pre,c = (tau_f/tau_c) * (dx_f/dx_c) = (tau_f/tau_c)/2
//
// -- fneq scales as tau times the velocity gradient per lattice cell, and
// the same physical shear spans twice as many fine cells, so the per-cell
// gradient, and hence fneq, is halved. Collision then multiplies fneq by
// (1 - 1/tau), which is a DIFFERENT number on each level, so:
//
//   fneq_post,f     (1 - 1/tau_f)     tau_f    1     tau_f - 1    1
//   -----------  =  -------------  *  ----- *  -  =  ---------  * -
//   fneq_post,c     (1 - 1/tau_c)     tau_c    2     tau_c - 1    2
//
// At tau_c = 0.8 (so tau_f = 1.1) that is -0.25, against the pre-collision
// form's +0.6875: a different magnitude AND the opposite sign, because
// tau_c < 1 is over-relaxed and flips fneq through the collision while
// tau_f > 1 does not.
//
// APPLYING THE PRE-COLLISION FACTOR TO POST-COLLISION POPULATIONS WAS THIS
// SOLVER'S COARSE/FINE INTERFACE BUG (plans/3D.md M3). Its signature is
// worth knowing, because it is what makes the class findable: it leaves
// mass EXACTLY conserved -- fneq has no zeroth or first moment, so it
// cannot perturb a mass flux at all -- while corrupting the viscous
// stress, which IS the second moment and hence the momentum flux. So the
// seam leaks momentum and not mass. tools/analyze-d3-interface.js measures
// both and reads that asymmetry directly.
//
// AGAL, the reference implementation this port follows, uses the
// pre-collision factor correctly: its S_Interpolate runs BEFORE S_Collide
// (AGAL/src/solver_lbm/solver_lbm_advance.cu), so its buffers hold
// post-STREAM, pre-collision f. The formula transferred; the convention it
// belongs to did not.
//
// tau = 1 IS A REAL SINGULARITY, not a formula artifact. At omega = 1 the
// post-collision populations ARE the equilibrium and carry no stress
// whatsoever, so NO post-collision transfer can recover it -- the
// information is gone, on either level. main-3d.js refuses a tau that puts
// either level there instead of dividing by ~0 quietly.
//
// 1 restores the old, wrong pre-collision factor, so the defect can be
// re-measured in the shipped build rather than reconstructed from a branch
// -- the same arrangement as ?f16= and ?ghostcopy=. It is not a mode to
// run physics in.
override DC_PRE : u32 = 0u;

fn dcRescaleC2F(tauCoarse: f32) -> f32 {
  let tauFine = 2.0f * tauCoarse - 0.5f;
  if (DC_PRE != 0u) { return 0.5f * tauFine / tauCoarse; }
  return 0.5f * (tauFine - 1.0f) / (tauCoarse - 1.0f);
}

// The exact inverse, so that a restriction immediately following an
// interpolation is the identity on fneq. Written as its own expression
// rather than 1/dcRescaleC2F so the two read as a matched pair.
fn dcRescaleF2C(tauCoarse: f32) -> f32 {
  let tauFine = 2.0f * tauCoarse - 0.5f;
  if (DC_PRE != 0u) { return 2.0f * tauCoarse / tauFine; }
  return 2.0f * (tauCoarse - 1.0f) / (tauFine - 1.0f);
}

// Trilinear blend of the 8 surrounding parent samples. `t` is the fine
// cell's fractional position inside the parent cell at s[0]; samples are
// indexed s[(k*2 + j)*2 + i] for corner (i, j, k).
fn interpCoarseToFine3(s: array<CoarseSample3, 8>, t: vec3<f32>, tauCoarse: f32) -> array<f32, QN> {
  var w: array<f32, 8>;
  for (var k = 0u; k < 2u; k++) {
    let wz = select(1f - t.z, t.z, k == 1u);
    for (var j = 0u; j < 2u; j++) {
      let wy = select(1f - t.y, t.y, j == 1u);
      for (var i = 0u; i < 2u; i++) {
        let wx = select(1f - t.x, t.x, i == 1u);
        w[(k * 2u + j) * 2u + i] = wx * wy * wz;
      }
    }
  }

  var rho = 0f;
  var u = vec3<f32>(0f);
  for (var c = 0u; c < 8u; c++) {
    rho += w[c] * s[c].rho;
    u += w[c] * s[c].u;
  }

  let rescale = dcRescaleC2F(tauCoarse);
  var fo: array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    var fneq = 0f;
    for (var c = 0u; c < 8u; c++) { fneq += w[c] * s[c].fneq[i]; }
    fo[i] = feqD3Q(rho, u.x, u.y, u.z, i) + rescale * fneq;
  }
  return fo;
}
