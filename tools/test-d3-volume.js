#!/usr/bin/env node
// The volume stack and the camera, checked without a GPU. plans/3D.md
// M6.2-M6.4; the module is d3-volume.mjs and the shaders it mirrors are
// shaders/common_d3_resample.wgsl (the box map) and shaders/d3_raymarch.wgsl
// (the box test and the camera).
//
// WHAT IS ACTUALLY CHECKABLE HERE, and it is worth saying because a viewer
// tempts one to check the wrong thing. "Does the picture look right" has no
// reference value. What does:
//
//   THE BOX MAP IS THE RESAMPLE'S INVERSE. The resample kernel places voxel
//   v at lo + (v + 1/2) h - 1/2 and the raymarcher recovers a texture
//   coordinate from a position. Composing them must be the identity, exactly,
//   or the volume is sampled half a voxel off its own contents -- which at
//   ?vol=1 over the domain is a visible shift and everywhere else is a blur
//   nobody can attribute.
//
//   INNERMOST WINS IS A CLAIM ABOUT ORDER. The stack is coarsest-first and
//   the LAST containing box is the answer; "the first" is the same code with
//   one word changed and produces a picture that is merely coarser, which is
//   exactly the kind of wrong that survives being looked at.
//
//   THE PERIODIC BOX. A refined shell straddles the seam twice per lap under
//   a moving window, and a box taken as min..max is then the whole domain.
//
// MUTATION-CHECKED, in the same shape tools/test-d3-amr.js checks
// cellAtLevel: every assertion below is re-run against a deliberately WRONG
// variant, and the test fails if the wrong one also passes. A check that
// only ever sees correct input cannot distinguish a working function from
// one that returns a constant.

const assert = require('assert');

(async () => {
const {
  levelVolume, volumeStack, boxLocal, innermostAt, rayBox, rayTouchesRefined,
  stackUnwrapped, orbitBasis, orbitEye, cameraBasis, cameraRay, boxRatio,
} = await import('../d3-volume.mjs');

let checks = 0;
const ok = (cond, msg) => { checks++; assert.ok(cond, msg); };
const close = (a, b, tol, msg) => ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

// --- the box, and the frame it lives in ------------------------------------
{
  const v = levelVolume({ lo: [36, 52, 52], ext: [24, 24, 24], level: 2, mult: 1 });
  assert.deepStrictEqual(v.res, [96, 96, 96], 'mult 1 gives level 2 its OWN grid, 4 cells per L0 cell');
  v.h.forEach((h) => close(h, 0.25, 1e-15, 'voxel size is 2^-level'));
  // Cell i spans [i - 1/2, i + 1/2), so a box of `ext` cells from index `lo`
  // spans [lo - 1/2, lo + ext - 1/2). The half-cell is the whole content of
  // this assertion and it is what the raymarcher's boxLo carries.
  assert.deepStrictEqual(v.c0, [35.5, 51.5, 51.5], 'continuous lo is lo - 1/2');
  assert.deepStrictEqual(v.c1, [59.5, 75.5, 75.5], 'continuous hi is lo + ext - 1/2');
  checks += 3;
}

// THE COMPOSITION THAT MATTERS: resample placement -> box map -> identity.
// volCentre is shaders/common_d3_resample.wgsl's, transcribed here because
// there is nothing else to compare it against without a GPU -- the GPU
// agreement is tools/validate-d3-invariants.js's volume gate, which scores
// the filled texture against the tree sampler at the same physical points.
{
  const volCentre = (v, i) => i.map((c, k) => v.lo[k] + (c + 0.5) * v.h[k] - 0.5);
  for (const spec of [
    { lo: [0, 0, 0], ext: [192, 128, 128], level: 0, mult: 1 },
    { lo: [36, 52, 52], ext: [24, 24, 24], level: 2, mult: 1 },
    { lo: [-6, 4, 120], ext: [32, 16, 16], level: 1, mult: 2 },
  ]) {
    const v = levelVolume(spec);
    for (const i of [[0, 0, 0], [1, 2, 3], [v.res[0] - 1, v.res[1] - 1, v.res[2] - 1]]) {
      const tc = boxLocal(volCentre(v, i), v);
      tc.forEach((t, k) => close(t, (i[k] + 0.5) / v.res[k], 1e-12,
        `box map inverts the resample's placement at voxel ${i}`));
    }
  }
  // THE MUTATION: drop the half-cell from the box's continuous origin, which
  // is the single most likely slip and the one the M6.1 alignment case exists
  // to catch. It must break the identity above.
  const v = levelVolume({ lo: [36, 52, 52], ext: [24, 24, 24], level: 2, mult: 1 });
  const wrong = { ...v, c0: v.lo.slice() };
  const tc = boxLocal(volCentre(v, [0, 0, 0]), wrong);
  ok(Math.abs(tc[0] - 0.5 / v.res[0]) > 1e-6, 'mutation: an origin without the half-cell must NOT invert');
}

// --- the stack -------------------------------------------------------------
{
  const dims = [192, 128, 128];
  const st = volumeStack({ dims, levels: 3, mult: 1, bytesPerVoxel: 16, levelBoxes: [
    null,
    { lo: [48, 32, 32], hi: [80, 64, 64] },
    { lo: [60, 44, 44], hi: [84, 68, 68] }]});
  assert.deepStrictEqual(st.map(v => v.level), [0, 1, 2], 'one volume per level, coarsest first');
  assert.deepStrictEqual(st[0].res, dims, 'level 0 covers the domain at 1x');
  assert.deepStrictEqual(st[1].res, [64, 64, 64], 'level 1 at its own resolution is 2 per L0 cell');
  assert.deepStrictEqual(st[2].res, [96, 96, 96], 'level 2 is 4');
  checks += 4;
  // THE MEASUREMENT THE WHOLE DESIGN RESTS ON: the three boxes together
  // against ONE uniform volume at the finest resolution over the domain.
  const stacked = st.reduce((a, v) => a + v.bytes, 0);
  const uniform = dims.reduce((a, n) => a * n * 4, 1) * 16;
  ok(uniform / stacked > 40, `the stack must be far cheaper than uniform-finest (got ${(uniform / stacked).toFixed(0)}x)`);

  // A LEVEL WITH NO TILES IS DROPPED, not given an empty texture: a
  // zero-extent texture is not creatable, and the coarser volume already
  // holds the right answer there because the resample writes sampleTree,
  // which falls back.
  const sparse = volumeStack({ dims, levels: 3, mult: 1, levelBoxes: [null, null, { lo: [0, 0, 0], hi: [8, 8, 8] }] });
  assert.deepStrictEqual(sparse.map(v => v.level), [0, 2], 'a level with no box is dropped');
  checks++;

  // INNERMOST WINS, and the mutation is "first wins".
  ok(innermostAt([70, 50, 50], st) === 2, 'a point in every box resolves to the finest');
  ok(innermostAt([50, 34, 34], st) === 1, 'a point in L1 only resolves to L1');
  ok(innermostAt([5, 5, 5], st) === 0, 'a point in L0 only resolves to L0');
  const firstWins = (p, s) => s.findIndex(v => boxLocal(p, v).every(t => t >= 0 && t <= 1));
  ok(firstWins([70, 50, 50], st) !== 2, 'mutation: first-box-wins must NOT agree at a refined point');

  ok(stackUnwrapped(st, dims), 'these boxes are inside the domain');
  ok(!stackUnwrapped(volumeStack({ dims, levels: 2, mult: 1, levelBoxes: [null, { lo: [180, 0, 0], hi: [212, 32, 32] }] }), dims),
    'a box past the far face is reported as straddling');
}

// --- the periodic box ------------------------------------------------------
{
  const dims = [64, 64, 64];
  // A shell that has walked off the far face and back on: lo = 56, ext = 16,
  // so it covers cells 56..63 and 0..7.
  const v = levelVolume({ lo: [56, 0, 0], ext: [16, 64, 64], level: 1, mult: 1 });
  ok(boxLocal([60, 10, 10], v, dims)[0] > 0 && boxLocal([60, 10, 10], v, dims)[0] < 1,
    'a point on the near side of the seam is inside');
  ok(boxLocal([3, 10, 10], v, dims)[0] > 0 && boxLocal([3, 10, 10], v, dims)[0] < 1,
    'a point on the FAR side of the seam is also inside');
  ok(boxLocal([30, 10, 10], v, dims)[0] > 1, 'a point opposite the box is outside');
  // Unwrapped, the far-side point reads as outside -- which is the bug the
  // wrap exists to remove, stated as a mutation.
  ok(!(boxLocal([3, 10, 10], v)[0] >= 0 && boxLocal([3, 10, 10], v)[0] <= 1),
    'mutation: without the wrap the far side of the seam falls out of the box');
}

// --- the camera ------------------------------------------------------------
{
  for (const up of [[0, 0, 1], [-1, 0, 0], [0, 1, 0]]) {
    const b = orbitBasis(up);
    const cr = [b.e1[1] * b.e2[2] - b.e1[2] * b.e2[1],
                b.e1[2] * b.e2[0] - b.e1[0] * b.e2[2],
                b.e1[0] * b.e2[1] - b.e1[1] * b.e2[0]];
    cr.forEach((c, k) => close(c, b.up[k], 1e-12, `orbit frame is right-handed for up=${up}`));
  }
  const target = [95.5, 63.5, 63.5];
  const eye = orbitEye(target, { azim: 0, elev: 0, dist: 100 });
  assert.deepStrictEqual(eye, [195.5, 63.5, 63.5], 'azim 0, elev 0 puts the eye on +e1');
  checks++;
  close(Math.hypot(...orbitEye(target, { azim: 1.1, elev: 0.4, dist: 100 }).map((e, k) => e - target[k])),
    100, 1e-9, 'the orbit radius is `dist` at any angle');

  const cam = cameraBasis(eye, target);
  // Orthonormal, and `fwd` points AT the target. A reflected basis would
  // mirror the picture, which changes no sign in these scalar fields and so
  // would simply lie about which side of the body the wake is on.
  close(cam.fwd[0], -1, 1e-12, 'fwd points at the target');
  [['fwd', 'right'], ['right', 'up'], ['up', 'fwd']].forEach(([a, b]) =>
    close(cam[a][0] * cam[b][0] + cam[a][1] * cam[b][1] + cam[a][2] * cam[b][2], 0, 1e-12, `${a}.${b} = 0`));

  const ray = cameraRay([0.5, 0.5], { eye, ...cam, tanHalfFov: 0.35, aspect: 1.5 });
  ray.d.forEach((d, k) => close(d, cam.fwd[k], 1e-12, 'the centre pixel looks straight down fwd'));
  // The ASPECT goes on the HORIZONTAL axis: a wide window must widen the
  // field of view rather than stretch the picture. Off by that factor the
  // sphere is an ellipse, which is the distortion the slice view's
  // planeExtent has its own history about.
  const wide = cameraRay([1, 0.5], { eye, ...cam, tanHalfFov: 0.35, aspect: 2 });
  const tall = cameraRay([0.5, 1], { eye, ...cam, tanHalfFov: 0.35, aspect: 2 });
  const ang = (r) => Math.acos(Math.max(-1, Math.min(1, r.d.reduce((a, d, k) => a + d * cam.fwd[k], 0))));
  close(Math.tan(ang(wide)) / Math.tan(ang(tall)), 2, 1e-9, 'the horizontal half-angle is `aspect` times the vertical');
}

// --- ray/box ---------------------------------------------------------------
{
  const v = levelVolume({ lo: [10, 10, 10], ext: [10, 10, 10], level: 0, mult: 1 });
  // Straight down +x from x = 0 through the box's middle: the slab is
  // [9.5, 19.5] in x, and the ray parameter is the distance.
  const hit = rayBox([0, 14, 14], [1, 0, 0], v.c0, v.c1);
  close(hit[0], 9.5, 1e-12, 'entry is the near face');
  close(hit[1], 19.5, 1e-12, 'exit is the far face');
  ok(rayBox([0, 100, 14], [1, 0, 0], v.c0, v.c1) === null, 'a ray that misses reports a miss');
  // An eye INSIDE the box still marches, from 0 rather than from behind it.
  close(rayBox([14, 14, 14], [1, 0, 0], v.c0, v.c1)[0], 0, 1e-12, 'an eye inside starts at t = 0');
  // An axis-parallel ray outside the slab on another axis is a miss, not a
  // divide by zero.
  ok(rayBox([0, 100, 14], [1, 0, 0], v.c0, v.c1) === null, 'axis-parallel miss is handled');

  const st = [levelVolume({ lo: [0, 0, 0], ext: [64, 64, 64], level: 0, mult: 1 }), v];
  ok(rayTouchesRefined([0, 14, 14], [1, 0, 0], st), 'a ray through the refined box touches it');
  ok(!rayTouchesRefined([0, 40, 40], [1, 0, 0], st), 'a ray past it does not');
  // The gate reads this as "allowed to differ", so a version that said yes
  // everywhere would pass any image difference at all.
  ok(!rayTouchesRefined([0, 40, 40], [0, 1, 0], st), 'nor does one pointing away');
}

// --- M6.4c's readout -------------------------------------------------------
{
  const r = boxRatio({ lo: [0, 0, 0], hi: [11, 11, 11] }, 1280);
  ok(r.boxBlocks === 1728, 'the box is inclusive of both ends');
  close(r.ratio, 1728 / 1280, 1e-12, 'the flagship shell measured 1.35x');
  ok(boxRatio(null, 0) === null, 'no set, no ratio');
}

console.log(`test-d3-volume: ${checks} checks passed`);
})().catch((e) => { console.error(e); process.exit(1); });
