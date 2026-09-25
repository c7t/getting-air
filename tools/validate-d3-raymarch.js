#!/usr/bin/env node
// THE VOLUME RENDERER, GATED. plans/3D.md M6.2 (the scalar field), M6.3 (the
// raymarcher) and M6.4b (innermost-box-wins).
//
// WHY IT IS ITS OWN TOOL. tools/validate-d3-invariants.js asks one question
// per config -- "is this run structurally sound" -- and answers it from
// inside the page. Two of the three claims here are about TWO RUNS
// (?volstack=1 against ?volstack=0) or about agreement between the page and
// an INDEPENDENT host computation, which is a different shape: it needs two
// navigations and a host-side comparison, not a deeper probe.
//
// THE THREE CLAIMS, and none of them is "the picture looks right":
//
//   M6.2  THE SCALAR VOLUME IS THE GRADIENT OF THE VELOCITY VOLUME.
//         Q is recomputed on the host from the SAME fp16 velocity texels the
//         kernel differenced, through d3-criterion.mjs's qOfGrad -- the
//         module tools/test-d3-criterion.js already scores against closed
//         forms, including the one that matters (pure shear is exactly
//         zero). So this is the GPU against an independently-tested host
//         routine on identical inputs, not against a second transcription.
//
//   M6.3  THE TRANSFER FUNCTION IS NORMALIZED TO THE ACTUAL FIELD. The M1
//         lesson applies here first: a transfer function not normalized to
//         the field renders a correct simulation as a black screen, and a
//         black canvas is this project's definition of a FAILURE rather than
//         of a clean run. Checked as coverage -- some pixels lit, not all of
//         them saturated -- rather than as a look.
//
//   M6.4b INNERMOST-BOX-WINS CHANGES THE PICTURE EXACTLY WHERE A REFINED BOX
//         COVERS. Two builds of one state, differenced pixel by pixel, with
//         d3-volume.mjs saying INDEPENDENTLY which pixels are allowed to
//         differ: the ones whose ray touches a refined box. A render that
//         changed everywhere would mean the box transform is wrong, not that
//         it got sharper; one that changed nowhere would mean the stack is
//         not being sampled at all.
//
// AND A FOURTH, WHICH IS REALLY ABOUT THE CAMERA. The body is sphere-traced
// from its own SDF, so with the volume turned off (?volIso= above every
// value) the render is a silhouette and nothing else -- and the host knows
// exactly where a sphere projects to. That is the only way to check a camera:
// "is it pointing at the interesting part" has no reference value, but "does
// the body land where the host says it lands" is arithmetic.
//
// Owns the whole lifecycle (HTTPS server + a dedicated debug-port Chrome if
// neither is up, one tab reused via Page.navigate), like tools/validate-3d.js.
//
//   node tools/validate-d3-raymarch.js
//   node tools/validate-d3-raymarch.js --cases=scalar,stack
//   node tools/validate-d3-raymarch.js --size=256 --steps=200

const path = require('path');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  steps: 400, size: 192, cases: null, extra: '', timeout: 600, keepOpen: false,
};

// THE FLAGSHIP SHAPE, small enough to run in seconds: a sphere at ?levels=3
// with a geometry-forced shell, which is the configuration every number in
// the M6 ballpark was measured on. `?live=0` because every check drives
// debugStepSync itself -- a render gate that raced the animation loop would
// be comparing two different states and calling it a rendering difference.
const BASE = 'scenario=sphere&n=8&re=100&u0=0.05&q=19&bounceback=1&live=0'
  + '&levels=3&rb=4&refine=body&interface=explode&vol=1&view=volume';
// A body that TRAVELS, for the window case. n = 16 keeps a wrap inside a few
// thousand steps; the defaults do the rest.
const CARD = 'scenario=card&n=16&live=0&vol=1&view=volume';

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--size=')) o.size = parseInt(a.slice(7));
    else if (a.startsWith('--cases=')) o.cases = a.slice(8).split(',');
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--timeout=')) o.timeout = parseInt(a.slice(10));
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, what, timeoutMs = 300000) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

const G = 'window.__D3';
const decode = (b64) => Buffer.from(b64, 'base64');

// --- M6.2 ------------------------------------------------------------------
//
// The scalar volume against the host's own Q, on the SAME inputs. The
// stencil's SEPARATION is reproduced exactly, including the shortening at a
// face: a sub-box volume has no neighbour outside itself, so the kernel
// clamps the index AND divides by the width it actually got. Dividing by the
// step it asked for would be wrong by 2x over the whole boundary layer of the
// volume, which for a refined box is precisely the seam region -- so the
// check reproduces the shortening rather than skipping the faces, which would
// leave it untested.
function hostQSlab({ slabs, res, h, wrap, qOfGrad, omegaOfGrad }) {
  const [nx, ny] = res;
  const at = (s, x, y) => slabs[s][y * nx + x].u;
  const tap = (i, n) => (wrap
    ? { ip: (i + 1) % n, im: ((i - 1) % n + n) % n, sep: 2 }
    : { ip: Math.min(i + 1, n - 1), im: Math.max(i - 1, 0), sep: Math.min(i + 1, n - 1) - Math.max(i - 1, 0) });
  const out = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const tx = tap(x, nx), ty = tap(y, ny);
      // The z arms are the neighbouring SLABS, so the centre slab is index 1
      // and the separation is always the full two steps -- the z faces of the
      // volume are deliberately not sampled here rather than sampled wrong.
      const up = [at(1, tx.ip, y), at(1, x, ty.ip), at(2, x, y)];
      const um = [at(1, tx.im, y), at(1, x, ty.im), at(0, x, y)];
      const sep = [tx.sep * h[0], ty.sep * h[1], 2 * h[2]];
      // J[i][j] = du_i/dx_j, d3-criterion.mjs's convention.
      const J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) J[i][j] = (up[j][i] - um[j][i]) / sep[j];
      out.push({ x, y, q: qOfGrad(J), omega: omegaOfGrad(J), speed: Math.hypot(...at(1, x, y)) });
    }
  }
  return out;
}

// --- M6.3 / M6.4b ----------------------------------------------------------

function imageStats(px) {
  let lit = 0, saturated = 0, bodyish = 0;
  for (let i = 0; i < px.length; i += 4) {
    const r = px[i] / 255, g = px[i + 1] / 255, b = px[i + 2] / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    // The background is (0.05, 0.05, 0.09) -- dark AND slightly blue. Lit
    // means "brighter than that by more than encoding noise".
    if (mx > 0.13) lit++;
    if (mx > 0.97) saturated++;
    // The body is shaded neutral grey; the volume ramp is strongly
    // saturated everywhere along it. So low chroma AND not dark is the body.
    if (mx - mn < 0.06 && mx > 0.15) bodyish++;
  }
  const n = px.length / 4;
  return { pixels: n, lit, litFrac: lit / n, saturated, satFrac: saturated / n,
           body: bodyish, bodyFrac: bodyish / n };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const { qOfGrad, omegaOfGrad } = await import('../d3-criterion.mjs');
  const { levelVolume, cameraBasis, cameraRay, orbitEye, rayBox, rayTouchesRefined,
          stackUnwrapped } = await import('../d3-volume.mjs');

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ local: true, port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = await attachPageWatch(client, {
    onError: (e) => console.error(`    !! [${e.kind}] ${e.text.split('\n')[0]}`),
  });

  const rows = [];
  const want = (name) => !o.cases || o.cases.includes(name);
  const open = async (extra, label) => {
      // The window case needs a body that TRAVELS, which the pinned sphere in
    // BASE does not; `card` is the scenario whose clip found this.
    const base = extra.includes('window=') ? CARD : BASE;
    const url = `${o.baseUrl}/index-3d.html?${base}${extra}${o.extra ? `&${o.extra}` : ''}`;
    console.log(`\n=== ${label} (${url})`);
    await navigateTo(Page, url);
    await waitForGlobal(Runtime, `${G}`, 60000);
    await assertPageHealthy(Runtime, watch, label);
    await ev(Runtime, `${G}.debugStepSync(${o.steps})`, 'debugStepSync');
    return await ev(Runtime, `${G}.getParams()`, 'getParams');
  };

  try {
    // --- M6.2: the scalar volume is the gradient of the velocity volume ----
    if (want('scalar')) {
      const p = await open('', 'scalar (M6.2)');
      const stack = p.volume.stack;
      const per = [];
      for (let which = 0; which < stack.length; which++) {
        const zc = Math.floor(stack[which].res[2] / 2);
        const slabs = [];
        for (const dz of [-1, 0, 1]) {
          slabs.push((await ev(Runtime, `${G}.debugReadVolume(${zc + dz}, ${which}, 'vel')`, 'debugReadVolume')).texels);
        }
        const scl = await ev(Runtime, `${G}.debugReadVolume(${zc}, ${which}, 'scl')`, 'debugReadVolume');
        const v = stack[which];
        const wrap = v.level === 0 && v.ext.every((e, k) => e === [p.NX, p.NY, p.NZ][k]);
        const host = hostQSlab({ slabs, res: v.res, h: v.h, wrap, qOfGrad, omegaOfGrad });
        // M6.4d. `hostQSlab` differences at a FIXED one-voxel stride, which is
        // what the kernel does for a voxel the volume's own level covers and
        // NOT what it does for one filled from a coarser level -- there the
        // stencil reaches a whole source cell. So the comparison is restricted
        // to the at-level population, where the host formula is exact, and the
        // replicated one is gated by the `volh` case instead.
        //
        // COUNTED AND PRINTED rather than silently filtered: a config whose
        // box grew until every voxel was replicated would otherwise pass this
        // by comparing nothing at all, and the count is the only thing that
        // would say so. On this config it is 0 -- a static run defaults
        // ?volMargin=0, so the box is the shell's tight bbox.
        const srcLvl = new Int32Array(scl.texels.length);
        for (let i = 0; i < scl.texels.length; i += 4096) {
          const pts = scl.texels.slice(i, i + 4096).map((t) => t.p);
          const r = await ev(Runtime, `${G}.debugSampleTree(${JSON.stringify(pts)})`, 'debugSampleTree');
          for (let j = 0; j < r.length; j++) srcLvl[i + j] = r[j].level;
        }
        let skipped = 0;
        for (let i = 0; i < srcLvl.length; i++) if (srcLvl[i] < v.level) skipped++;
        if (skipped === srcLvl.length) {
          throw new Error(`L${v.level}: every voxel in the slab is replicated from a coarser`
            + ' level, so this case would compare nothing -- see M6.4d');
        }
        // Scaled by the SLAB's own maximum, not per texel: Q is a difference
        // of two Frobenius norms and is legitimately near zero over most of a
        // volume, so a per-texel relative error would be dominated by
        // cancellation in the quiet field and would say nothing about the
        // part anyone looks at.
        let qMax = 0, omMax = 0;
        for (const hq of host) { qMax = Math.max(qMax, Math.abs(hq.q)); omMax = Math.max(omMax, hq.omega); }
        let worstQ = 0, worstOm = 0, worstU = 0;
        for (let i = 0; i < host.length; i++) {
          if (srcLvl[i] < v.level) continue;
          const g = scl.texels[i].v;
          worstQ = Math.max(worstQ, Math.abs(g[0] * p.volRefs.q - host[i].q) / Math.max(qMax, 1e-30));
          worstOm = Math.max(worstOm, Math.abs(g[1] * p.volRefs.omega - host[i].omega) / Math.max(omMax, 1e-30));
          worstU = Math.max(worstU, Math.abs(g[2] * p.volRefs.u - host[i].speed) / Math.max(1e-9, host[i].speed));
        }
        per.push({ level: v.level, res: v.res, texels: host.length - skipped, qMax, worstQ, worstOm, worstU });
        console.log(`    L${v.level} ${v.res.join('x')}: max|Q| ${qMax.toExponential(2)}`
          + `  worst Q ${worstQ.toExponential(2)}  |omega| ${worstOm.toExponential(2)}  |u| ${worstU.toExponential(2)}`
          + `  (${host.length - skipped} at-level texels, ${skipped} replicated and skipped)`);
      }
      // 1% of the slab's own maximum. The inputs are IDENTICAL fp16 texels,
      // so the only sources of disagreement are f32-against-f64 arithmetic in
      // the sum of nine squares and the final binary16 rounding of the
      // stored ratio -- the latter alone is 2^-11 of the stored value, and Q
      // is a small difference of larger numbers.
      const ok = per.every(r => r.worstQ < 1e-2 && r.worstOm < 1e-2 && r.worstU < 1e-2);
      rows.push({ name: 'scalar (M6.2)', ok, detail: `worst Q ${Math.max(...per.map(r => r.worstQ)).toExponential(2)}` });
    }

    // --- the camera, by the body's silhouette ------------------------------
    if (want('camera')) {
      // ?volIso above every value in the volume leaves the body and the
      // background and nothing else -- a silhouette the host can predict in
      // closed form, which is the only checkable claim a camera has.
      const p = await open('&volIso=1e9', 'camera (silhouette)');
      const img = await ev(Runtime, `${G}.debugRenderFrame(${o.size})`, 'debugRenderFrame');
      const px = decode(img.rgba);
      const st = imageStats(px);
      const body = await ev(Runtime, `${G}.readBody()`, 'readBody');
      const c = img.camera;
      const up = [0, 0, 0];
      up[c.upAxis] = c.upSign;
      const target = c.follow ? [body.cx, body.cy, body.cz] : c.target;
      const eye = orbitEye(target, { azim: c.azim, elev: c.elev, dist: c.dist, up });
      const basis = cameraBasis(eye, target, up);
      const R = p.R || p.body?.a || 0;
      // A RAY/SPHERE TEST, not a projection: the body is at a finite distance
      // under a perspective camera, so its outline is a conic and not a
      // circle, and a projected-radius check would be wrong by the same
      // amount everywhere -- which is exactly the kind of error that then
      // gets absorbed into a tolerance.
      let hit = 0, agree = 0, disagree = 0;
      for (let y = 0; y < img.h; y++) {
        for (let x = 0; x < img.w; x++) {
          // uv.y is UP the screen and the image's first row is the TOP.
          const uv = [(x + 0.5) / img.w, 1 - (y + 0.5) / img.h];
          const ray = cameraRay(uv, { eye, ...basis, tanHalfFov: Math.tan(c.fov / 2), aspect: c.aspect });
          const oc = ray.o.map((v, k) => v - target[k]);
          const b = 2 * oc.reduce((a, v, k) => a + v * ray.d[k], 0);
          const cc = oc.reduce((a, v) => a + v * v, 0) - R * R;
          const disc = b * b - 4 * cc;
          const inHost = disc > 0 && -b - Math.sqrt(disc) > 0;
          const i = (y * img.w + x) * 4;
          const mx = Math.max(px[i], px[i + 1], px[i + 2]) / 255;
          const mn = Math.min(px[i], px[i + 1], px[i + 2]) / 255;
          const inImg = mx - mn < 0.06 && mx > 0.15;
          if (inHost) hit++;
          if (inHost && inImg) agree++;
          else if (inHost !== inImg) disagree++;
        }
      }
      // Intersection over union. The silhouette is a staircase of pixels
      // either way, so the boundary row is allowed to disagree -- the claim
      // is that the body is WHERE the host says, not that antialiasing
      // matches.
      const iou = agree / Math.max(1, hit + disagree - (hit - agree));
      const ok = hit > 50 && agree / Math.max(1, hit) > 0.95 && disagree / Math.max(1, hit) < 0.15;
      console.log(`    silhouette: host ${hit} px, image agrees on ${agree}`
        + ` (${(100 * agree / Math.max(1, hit)).toFixed(1)}%), disagree ${disagree}, IoU ${iou.toFixed(3)}`);
      console.log(`    image: ${st.lit} lit / ${st.pixels}, ${st.body} body-shaded`);
      rows.push({ name: 'camera', ok, detail: `IoU ${iou.toFixed(3)} over ${hit} px` });
    }

    // --- M6.3: the transfer function is normalized -------------------------
    if (want('transfer')) {
      const p = await open('', 'transfer (M6.3)');
      // WHAT IS ACTUALLY IN THE VOLUME, before asking what the picture shows.
      // "Some pixels are lit" cannot tell a transfer function scaled to the
      // field from a threshold sitting under the noise floor -- the field's
      // own maximum against the iso is what separates them, and it is a
      // number the volume can be asked for.
      const st0 = p.volume.stack;
      let fieldMax = 0;
      for (let which = 0; which < st0.length; which++) {
        const zc = Math.floor(st0[which].res[2] / 2);
        const scl = await ev(Runtime, `${G}.debugReadVolume(${zc}, ${which}, 'scl')`, 'debugReadVolume');
        for (const t of scl.texels) fieldMax = Math.max(fieldMax, t.v[0]);
      }
      const set = await ev(Runtime, `${G}.debugSetVolume({})`, 'debugSetVolume');
      const img = await ev(Runtime, `${G}.debugRenderFrame(${o.size})`, 'debugRenderFrame');
      const on = imageStats(decode(img.rgba));
      // THE A/B THAT MAKES IT A CHECK: raise the iso above everything in the
      // volume and the volume must vanish, leaving the body and the
      // background. The DIFFERENCE between the two counts is the volume's own
      // contribution, which is the thing being claimed to exist.
      await ev(Runtime, `${G}.debugSetVolume({ iso: 1e9 })`, 'debugSetVolume');
      const imgOff = await ev(Runtime, `${G}.debugRenderFrame(${o.size})`, 'debugRenderFrame');
      const off = imageStats(decode(imgOff.rgba));
      await ev(Runtime, `${G}.debugSetVolume({ iso: ${set.iso} })`, 'debugSetVolume');
      console.log(`    field max ${fieldMax.toFixed(3)} against iso ${set.iso} (${set.field}/ref)`);
      console.log(`    lit ${on.lit}/${on.pixels} (${(100 * on.litFrac).toFixed(2)}%) with the volume,`
        + ` ${off.lit} (${(100 * off.litFrac).toFixed(2)}%) without it;`
        + ` saturated ${on.saturated} (${(100 * on.satFrac).toFixed(2)}%)`);
      // A BLACK CANVAS IS A FAILURE and so is a white one -- and so, less
      // obviously, is a picture that does not change when the volume is
      // switched off, which is what a render reading an empty texture looks
      // like. The saturation bound is what says the scale is a SCALE and not
      // a clip.
      const ok = fieldMax > set.iso && on.lit > off.lit && on.satFrac < 0.25
        && on.litFrac < 0.6 && off.bodyFrac > 0.001;
      rows.push({ name: 'transfer (M6.3)', ok,
                  detail: `max ${fieldMax.toFixed(2)} > iso ${set.iso}, lit ${off.lit} -> ${on.lit}` });
    }

    // --- the window: the picture must not depend on the buffer's origin ----
    //
    // A windowed run's body WRAPS through a periodic buffer while its wake
    // stays a fixed distance behind it, so where the body happens to sit in
    // the buffer is a bookkeeping detail the picture must be blind to. It was
    // not: with the march interval in buffer coordinates the ray stopped at
    // the seam, so the moment the body crossed it the whole wake was outside
    // the interval and the render collapsed -- 8.22% of the frame lit against
    // 38.45%, measured on the card at buffer x = 0.
    //
    // GATED AS CONTINUITY, which is the form that needs no reference picture:
    // the flow changes very little in 200 steps, so the lit fraction must not
    // JUMP between adjacent samples. A seam cut makes it collapse in one
    // sample and recover over the next several, which is exactly the sawtooth
    // that gave this away in a clip.
    if (want('window')) {
      const p = await open('&window=xyz&dist=0.8&volIso=0.03&volGain=2.5&volOpacity=0.6',
                           'window (buffer-origin invariance)');
      const lit = [];
      for (let i = 0; i < 18; i++) {
        if (i) await ev(Runtime, `${G}.debugStepSync(200)`, 'debugStepSync');
        const img = await ev(Runtime, `${G}.debugRenderFrame({ w: 160, h: 160, view: 'volume' })`, 'debugRenderFrame');
        const b = await ev(Runtime, `${G}.readBody()`, 'readBody');
        lit.push({ step: img.step, bufX: b.cx, frac: imageStats(decode(img.rgba)).litFrac });
      }
      let worst = 1, at = null;
      for (let i = 1; i < lit.length; i++) {
        // Only DROPS matter: a wake that grows is the flow, a wake that
        // vanishes between two adjacent samples is the seam.
        const r = lit[i - 1].frac / Math.max(lit[i].frac, 1e-6);
        if (r > worst) { worst = r; at = lit[i]; }
      }
      console.log('    lit fraction across a wrap: '
        + lit.map(l => `${(100 * l.frac).toFixed(0)}%`).join(' '));
      console.log(`    buffer x: ${lit.map(l => l.bufX.toFixed(0)).join(' ')}`);
      console.log(`    worst one-sample drop ${worst.toFixed(2)}x`
        + (at ? ` at step ${at.step} (buffer x ${at.bufX.toFixed(0)})` : ''));
      // THE THRESHOLD IS SET FROM THE SEPARATION, not from taste, and the
      // first value chosen by taste was wrong: 2x let the broken build PASS.
      // Measured on this config, fixed against ?winbox=0: worst drop 1.00x
      // against 1.84x, the latter landing exactly on the sample where the
      // body's buffer x wraps. 1.25x sits between them with margin on both
      // sides. The n = 32 card collapses 4.7x, so a bigger domain would make
      // this easier -- this config is deliberately the harder one.
      //
      // A drop of ANY size is suspect with the fix in, because 200 steps of a
      // slowly-developing wake cannot remove lit volume; the allowance is for
      // sampling noise at the iso, not for physics.
      const ok = worst < 1.25 && lit.every(l => l.frac > 0.01);
      rows.push({ name: 'window', ok, detail: `worst drop ${worst.toFixed(2)}x over a wrap` });
    }

    // --- M6.4b: innermost box wins -----------------------------------------
    if (want('stack')) {
      const a = await open('', 'stack (M6.4b)');
      const imgA = await ev(Runtime, `${G}.debugRenderFrame(${o.size})`, 'debugRenderFrame');
      const b = await open('&volstack=0', 'stack control (?volstack=0)');
      const imgB = await ev(Runtime, `${G}.debugRenderFrame(${o.size})`, 'debugRenderFrame');
      const bodyA = await ev(Runtime, `${G}.readBody()`, 'readBody');

      const stack = imgA.stack.map(v => levelVolume({ lo: v.lo, ext: v.ext, level: v.level, mult: 1 }));
      // Restated rather than assumed: rayTouchesRefined is only a valid
      // predicate where no box straddles the periodic seam, and a gate that
      // quietly widened its own allowed set until it passed would be no gate.
      if (!stackUnwrapped(stack, [a.NX, a.NY, a.NZ])) {
        throw new Error('a refined box straddles the periodic seam; the host predicate does not cover that');
      }
      if (imgB.stack.length !== 1) throw new Error(`?volstack=0 built ${imgB.stack.length} volumes, expected 1`);
      if (stack.length < 2) throw new Error('the stacked leg has only one volume -- nothing to compare');

      const c = imgA.camera;
      const up = [0, 0, 0];
      up[c.upAxis] = c.upSign;
      const target = c.follow ? [bodyA.cx, bodyA.cy, bodyA.cz] : c.target;
      const eye = orbitEye(target, { azim: c.azim, elev: c.elev, dist: c.dist, up });
      const basis = cameraBasis(eye, target, up);
      const pa = decode(imgA.rgba), pb = decode(imgB.rgba);
      let differ = 0, differAllowed = 0, differForbidden = 0, allowed = 0, worst = 0, firstBad = null;
      for (let y = 0; y < imgA.h; y++) {
        for (let x = 0; x < imgA.w; x++) {
          const uv = [(x + 0.5) / imgA.w, 1 - (y + 0.5) / imgA.h];
          const ray = cameraRay(uv, { eye, ...basis, tanHalfFov: Math.tan(c.fov / 2), aspect: c.aspect });
          const span = rayBox(ray.o, ray.d, stack[0].c0, stack[0].c1);
          const may = span ? rayTouchesRefined(ray.o, ray.d, stack, span[1]) : false;
          if (may) allowed++;
          const i = (y * imgA.w + x) * 4;
          const d = Math.max(Math.abs(pa[i] - pb[i]), Math.abs(pa[i + 1] - pb[i + 1]), Math.abs(pa[i + 2] - pb[i + 2]));
          // 2/255 of tolerance, for 8-bit quantization of two renders that
          // take DIFFERENT numbers of ray steps through the same field --
          // the stacked leg steps by the innermost voxel size, so even where
          // it reads the same values it accumulates them at different
          // points. Anything real is far larger.
          if (d <= 2) continue;
          differ++;
          if (may) differAllowed++;
          else {
            differForbidden++;
            if (d > worst) { worst = d; firstBad = { x, y, d }; }
          }
        }
      }
      console.log(`    ${differ} pixels differ of ${imgA.w * imgA.h}`
        + ` (${allowed} could: their ray touches a refined box)`);
      console.log(`    allowed ${differAllowed}, FORBIDDEN ${differForbidden}`
        + (firstBad ? `  worst at (${firstBad.x},${firstBad.y}) by ${firstBad.d}/255` : ''));
      // BOTH DIRECTIONS. Nothing differing means the stack is not being
      // sampled; something differing outside every refined box means the box
      // transform is wrong.
      const ok = differForbidden === 0 && differAllowed > 0.05 * Math.max(1, allowed);
      rows.push({ name: 'stack (M6.4b)', ok,
                  detail: `${differAllowed} allowed / ${differForbidden} forbidden, ${allowed} could` });
    }

    // --- M6.4d: the gradient stencil reaches one SOURCE cell ---------------
    //
    // A level's volume is a dense grid over a bounding box and the refined set
    // inside it is a shell, so most of the box is filled from a COARSER level.
    // The resample samples NEAREST, so those voxels come in bit-identical
    // blocks, and differencing them at the volume's own spacing turns the
    // replication staircase into speckle. `?volh=1` (the default) differences
    // each voxel over its own source cell instead.
    //
    // TWO CLAIMS, AND THE FIRST IS THE ONE THAT MAKES THE CHANGE SAFE:
    //
    //   1. IT IS A NO-OP WHERE THE VOLUME ALREADY HAS THE DATA. Every voxel
    //      the refined level actually covers must be BIT-IDENTICAL across the
    //      A/B -- not close, identical, because its stride is 1 either way and
    //      the arithmetic is the same instructions on the same texels. This is
    //      what says M6.4b's sharp near-wall sheet is untouched.
    //   2. AND IT CHANGES THE REPLICATED ONES. Something has to move, or the
    //      flag is wired to nothing -- the failure 96547af had no control for.
    //
    // The classifier is `debugSampleTree`, i.e. the SHADER's own walk, so this
    // asks what the resample actually read rather than what the box implies.
    if (want('volh')) {
      // `?volMargin=2` IS PART OF THE CASE, not a tweak to make it pass. A
      // STATIC run defaults the margin to 0, so the box is the tight bbox of
      // the shell and a small sphere's shell very nearly fills it -- the first
      // run of this gate found 4096 at-level voxels and ZERO replicated ones,
      // i.e. nothing for the stride to act on. The configuration that shows
      // the artifact pads the box (the flagship clip runs ?volMargin=4), and
      // padding is exactly what puts a known-coarse ring inside a refined
      // level's volume.
      const AB = '&volMargin=2&volh=';
      const a = await open(`${AB}1`, 'volh (M6.4d)');
      const which = a.volume.count - 1;
      if (which < 1) throw new Error(`no refined volume in the stack (count ${a.volume.count})`);

      // THE SLICE IS CHOSEN, NOT ASSUMED TO BE THE MIDDLE, and the score is
      // min(at-level, replicated): a slice with only one population cannot
      // test either claim, and the middle of a body-fitted box is exactly
      // where the refined set is solid.
      const classify = async (z) => {
        const scl = await ev(Runtime, `${G}.debugReadVolume(${z}, ${which}, 'scl')`, 'debugReadVolume');
        const lv = new Int32Array(scl.texels.length);
        for (let i = 0; i < scl.texels.length; i += 4096) {
          const pts = scl.texels.slice(i, i + 4096).map((t) => t.p);
          const r = await ev(Runtime, `${G}.debugSampleTree(${JSON.stringify(pts)})`, 'debugSampleTree');
          for (let j = 0; j < r.length; j++) lv[i + j] = r[j].level;
        }
        let at = 0, co = 0;
        for (let i = 0; i < lv.length; i++) { if (lv[i] >= scl.info.level) at++; else co++; }
        return { scl, lv, at, co };
      };
      const nz = a.volume.stack[which].res[2];
      let pick = null;
      for (let z = 0; z < nz; z += Math.max(1, Math.floor(nz / 8))) {
        const c = await classify(z);
        console.log(`    z=${z}: ${c.at} at-level, ${c.co} replicated`);
        if (!pick || Math.min(c.at, c.co) > Math.min(pick.at, pick.co)) pick = { ...c, z };
      }
      if (!pick || pick.co === 0) {
        throw new Error('no slice of the refined volume holds a replicated voxel'
          + ' -- the box covers its set exactly, so there is nothing for M6.4d to act on');
      }
      const { scl: sclA, lv: lvl, z: zc } = pick;
      const volLevel = sclA.info.level;
      console.log(`    slice z=${zc}: ${pick.at} at-level, ${pick.co} replicated`);

      await open(`${AB}0`, 'volh control (?volh=0)');
      const sclB = await ev(Runtime, `${G}.debugReadVolume(${zc}, ${which}, 'scl')`, 'debugReadVolume');
      if (sclB.texels.length !== sclA.texels.length) {
        throw new Error('the A/B produced different volume shapes -- not a control');
      }

      // Bit-identical on the at-level population; SOMETHING different on the
      // replicated one. Q is channel 0.
      let atLevel = 0, atLevelDiff = 0, coarse = 0, coarseDiff = 0;
      for (let i = 0; i < sclA.texels.length; i++) {
        const qa = sclA.texels[i].v[0], qb = sclB.texels[i].v[0];
        const same = Object.is(qa, qb);
        if (lvl[i] >= volLevel) { atLevel++; if (!same) atLevelDiff++; }
        else { coarse++; if (!same) coarseDiff++; }
      }

      // AND THE SPECKLE ITSELF, by the statistic probe-d3-volume-crunch.js
      // uses: the normalized in-plane Laplacian of Q, averaged over the
      // replicated population only, where structure exists to speckle.
      const [nx, ny] = sclA.info.res;
      const crunch = (t) => {
        const q = t.map((x) => x.v[0]);
        let acc = 0, n = 0;
        for (let y = 1; y < ny - 1; y++) {
          for (let x = 1; x < nx - 1; x++) {
            const i = y * nx + x;
            if (lvl[i] >= volLevel) continue;
            const c = q[i];
            const n4 = q[i + 1] + q[i - 1] + q[i + nx] + q[i - nx];
            const mag = Math.abs(c) + Math.abs(n4) / 4;
            if (mag < 0.05) continue;
            acc += Math.abs(4 * c - n4) / (mag + 2 ** -14); n++;
          }
        }
        return n ? acc / n : null;
      };
      const cA = crunch(sclA.texels), cB = crunch(sclB.texels);

      // The two structural claims GATE; the crunch statistic gates only when
      // it has a population to average over. A young wake at 200 steps can
      // leave the replicated ring below the structure floor, and a gate that
      // fails on an empty average is reporting its own sample size.
      const ok = atLevelDiff === 0 && coarseDiff > 0
        && (cA == null || cB == null || cA < cB);
      console.log(`    at-level ${atLevel} voxels, ${atLevelDiff} differ (must be 0)`);
      console.log(`    replicated ${coarse} voxels, ${coarseDiff} differ (must be > 0)`);
      console.log(`    crunch on the replicated population: ${cA?.toFixed(4)} with,`
        + ` ${cB?.toFixed(4)} without (${cB && cA ? (cB / cA).toFixed(2) : '-'}x)`);
      rows.push({ name: 'volh (M6.4d)', ok,
                  detail: `${atLevelDiff}/${atLevel} at-level differ, ${coarseDiff}/${coarse} replicated,`
                    + ` crunch ${cA == null ? '-' : cA.toFixed(3)} vs ${cB == null ? '-' : cB.toFixed(3)}` });
    }
    // --- M6.4d/M6.4f: the source-level companion says the truth ------------
    //
    // BOTH features rest on one buffer. M6.4d picks a voxel's stencil from it
    // and M6.4f decides which box the ray may use from it, so a companion that
    // is merely plausible would make both of them confidently wrong in the
    // same direction -- and neither's own gate would notice, because both ask
    // whether the flag CHANGED something, not whether it changed it where it
    // should have.
    //
    // The claim is exact and there is nothing to tolerance: the companion holds
    // what `sampleTree` returned at that voxel, and `debugSampleTree` is that
    // same shader walk asked from the host at the same point. Every texel, or
    // it is wrong.
    if (want('srclevel')) {
      const p = await open('&volMargin=2', 'srclevel (M6.4d/f)');
      const per = [];
      for (let which = 1; which < p.volume.count; which++) {
        const zc = Math.floor(p.volume.stack[which].res[2] / 2);
        const lvl = await ev(Runtime, `${G}.debugReadVolume(${zc}, ${which}, 'lvl')`, 'debugReadVolume');
        if (lvl.skipped) throw new Error(`L${which}: ${lvl.skipped}`);
        let bad = 0, first = null, hist = {};
        for (let i = 0; i < lvl.texels.length; i += 4096) {
          const chunk = lvl.texels.slice(i, i + 4096);
          const r = await ev(Runtime, `${G}.debugSampleTree(${JSON.stringify(chunk.map((t) => t.p))})`,
            'debugSampleTree');
          for (let j = 0; j < r.length; j++) {
            const got = chunk[j].src, want_ = r[j].level;
            hist[got] = (hist[got] || 0) + 1;
            if (got !== want_) { bad++; if (!first) first = { p: chunk[j].p, got, want: want_ }; }
          }
        }
        per.push({ level: lvl.info.level, texels: lvl.texels.length, bad, hist });
        console.log(`    L${lvl.info.level} ${lvl.info.res.join('x')}: ${lvl.texels.length} texels,`
          + ` ${bad} disagree with debugSampleTree`
          + `  levels seen ${JSON.stringify(hist)}`);
        if (first) console.log(`      first: at ${first.p.map((v) => v.toFixed(2))}`
          + ` companion says ${first.got}, sampler says ${first.want}`);
      }
      // A SLAB THAT IS ALL ONE LEVEL PROVES NOTHING -- it would pass against a
      // companion hardwired to that number. ?volMargin=2 is there to guarantee
      // both populations, and this is the assertion that it did.
      const mixed = per.every((r) => Object.keys(r.hist).length > 1);
      const ok = per.length > 0 && per.every((r) => r.bad === 0) && mixed;
      if (!mixed) console.log('    !! a slab holds only ONE source level -- nothing is discriminated');
      rows.push({ name: 'srclevel (M6.4d/f)', ok,
                  detail: `${per.reduce((a, r) => a + r.bad, 0)} disagree of `
                    + `${per.reduce((a, r) => a + r.texels, 0)}, ${mixed ? 'mixed' : 'SINGLE-LEVEL'}` });
    }
  } finally {
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  console.log('\n' + '='.repeat(92));
  console.log(`SUMMARY  ${o.steps} steps, ${o.size}x${o.size} offscreen renders`);
  console.log('='.repeat(92));
  let exit = 0;
  for (const r of rows) {
    if (!r.ok) exit = 1;
    console.log(`${String(r.name).padEnd(20)}${String(r.detail).padEnd(58)}${(r.ok ? 'PASS' : 'FAIL').padStart(8)}`);
  }
  if (!rows.length) { console.log('no cases selected'); exit = 2; }
  console.log(exit === 0 ? '\nALL PASS' : '\nFAILURES ABOVE');
  process.exit(exit);
}

main().catch((e) => { console.error(e); process.exit(1); });
