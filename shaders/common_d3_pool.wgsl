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

// Dupuis-Chopard non-equilibrium rescale, coarse -> fine. The factor is
// (tau_fine/tau_coarse) * (dx_fine/dx_coarse) = (tau_fine/tau_coarse) * 1/2.
// fneq scales as tau * (velocity gradient per lattice cell); the same
// physical shear spans twice as many fine cells, so the per-cell gradient --
// and hence fneq -- is halved. Omitting the 1/2 leaves an O(1)
// non-equilibrium stress discontinuity at every interface, injecting
// spurious vorticity there. Identical to the 2D form; this is one of the
// pieces plans/3D.md sec 1.1 lists as transferring verbatim.
fn dupuisChopardRescale3(tauCoarse: f32) -> f32 {
  let tauFine = 2.0f * tauCoarse - 0.5f;
  return 0.5f * tauFine / tauCoarse;
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

  let rescale = dupuisChopardRescale3(tauCoarse);
  var fo: array<f32, QN>;
  for (var i = 0u; i < QN; i++) {
    var fneq = 0f;
    for (var c = 0u; c < 8u; c++) { fneq += w[c] * s[c].fneq[i]; }
    fo[i] = feqD3Q(rho, u.x, u.y, u.z, i) + rescale * fneq;
  }
  return fo;
}
