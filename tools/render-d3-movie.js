#!/usr/bin/env node
// RENDER A CLIP OF A 3D RUN. plans/3D.md M6.5.
//
// Drives index-3d.html forward a fixed number of steps per frame, renders
// each frame OFFSCREEN at a fixed size, and pipes the raw pixels straight
// into ffmpeg. No screen recorder, no temp PNGs, no compositor.
//
// WHY NOT A SCREEN RECORDER, which is the obvious way to film a canvas. Three
// reasons, and they are the same three that put M6.4b's gate on this path:
//
//   THE FRAME RATE WOULD BE THE SOLVER'S. The page advances ?spf= steps per
//   animation frame and the browser paints when it can, so a recording's
//   time axis is whatever the GPU managed that second -- a clip of a slow
//   configuration runs slow, and two clips of two builds are not comparable.
//   Here the STEPS PER FRAME are declared and the wall clock is irrelevant:
//   `--seconds` and `--fps` describe the FILM, `--steps` describes the FLOW,
//   and the three together fix exactly how much simulated time a second of
//   footage is worth.
//
//   THE SIZE WOULD BE THE WINDOW'S. An offscreen target is whatever was
//   asked for, on any machine, headless or not.
//
//   THE STATE WOULD RACE THE LOOP. debugStepSync advances and returns; the
//   render that follows is of the state that was asked for and not of
//   whatever the animation loop had reached.
//
// WHAT IT REPORTS, because a tool here should not only produce a file: the
// body's trajectory is read back every frame (a 150-byte readback) and the
// clip ends with how far it fell, how far it drifted across, how many turns
// it made and how fast it was going. A movie of a tumbling plate that does
// not tumble is a thing worth being TOLD rather than left to notice.
//
//   node tools/render-d3-movie.js                        # the card, 30 s
//   node tools/render-d3-movie.js --seconds=10 --fps=24 --size=640x480
//   node tools/render-d3-movie.js --view=slice --out=/tmp/card-slice.mp4
//   node tools/render-d3-movie.js --scenario=sphere --extra=re=300 --spin=90
//   node tools/render-d3-movie.js --gif --seconds=6 --size=480x360
//   node tools/render-d3-movie.js --encode=rgba                # raw frames on the pipe

const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const CDP = require('/usr/lib/node_modules/chrome-remote-interface');
const {
  ensureServer, ensureChrome, openTab, firstTab, navigateTo, waitForGlobal,
  attachPageWatch, assertPageHealthy, teardown,
} = require('./lib/browser-lifecycle');

const REPO_ROOT = path.join(__dirname, '..');

// PER-SCENARIO FRAMING, and it is the only thing in here chosen by eye.
// Everything else -- the iso, the gain, the tone curve -- is measured or
// derived (see d3_volume_scalar.wgsl and d3_raymarch.wgsl). Where a camera
// points has no reference value, so it is a default to be overridden rather
// than a claim.
//
// `steps` is the SIMULATED length of the clip, and it is the number worth
// arguing about: the card's is ~26000, which is where plans/3D.md D4 measured
// its terminal velocity and about where the Re = 500 run reaches the first
// tau limit. Asking for more footage than the run survives produces a clip
// that ends in a blowup, which the tool says out loud rather than encoding.
const PRESETS = {
  // THE CARD'S TRANSFER FUNCTION IS MEASURED FROM ITS OWN FIELD, and the
  // first version was not -- it inherited an iso tuned on an n = 16, Re = 200
  // flow and rendered the n = 32, Re = 500 one as a solid cyan cloud with the
  // plate invisible inside it. That is M1's lesson from the other side: a
  // transfer function not normalized to the actual field renders a correct
  // simulation as a BLANK screen, and blank can be white as easily as black.
  //
  // Measured on the developed wake (Q/qRef over three z-slabs, steps 2000 to
  // 26000): max 12-13, p99.9 3.0 -> 5.4, p99.5 0.5 -> 3.5, p99 0.04 -> 2.3,
  // median ~0. So the iso sits near p99 and the tone knee (iso + gain) near
  // p99.9, which lights the cores and leaves the background alone. The
  // OPACITY is the other half: at 0.6 per L0 cell a 190-cell domain is
  // opaque whatever the iso is, and 0.12 is what lets a ray cross the wake
  // and still show the plate behind it.
  card:   { url: 'scenario=card&n=32', steps: 26000, view: 'volume',
            vol: 'vol=1&volIso=1.5&volGain=4&volOpacity=0.12&dist=0.95&azim=28&elev=12' },
  fall:   { url: 'scenario=fall&n=16&bounceback=1', steps: 20000, view: 'volume',
            vol: 'vol=1&volIso=0.05&volGain=3&dist=0.8&azim=30&elev=14' },
  sphere: { url: 'scenario=sphere&n=16&re=300&perturb=0.02&bounceback=1&levels=3&refine=body',
            steps: 12000, view: 'volume',
            vol: 'vol=1&volIso=0.1&volGain=4&dist=1.0&azim=25&elev=14' },
};

const DEFAULTS = {
  baseUrl: 'https://localhost:4444', port: 9333,
  scenario: 'card', extra: '', url: null,
  seconds: 30, fps: 30, size: '800x600', view: null,
  steps: null, spf: null, warmup: 0, spin: 0,
  out: null, crf: 18, gif: false, keepOpen: false, quiet: false, encode: 'png',
};

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (const a of argv) {
    if (a.startsWith('--baseUrl=')) o.baseUrl = a.slice(10);
    else if (a.startsWith('--port=')) o.port = parseInt(a.slice(7));
    else if (a.startsWith('--scenario=')) o.scenario = a.slice(11);
    else if (a.startsWith('--extra=')) o.extra = a.slice(8).replace(/^[?&]/, '');
    else if (a.startsWith('--url=')) o.url = a.slice(6).replace(/^[?&]/, '');
    else if (a.startsWith('--seconds=')) o.seconds = parseFloat(a.slice(10));
    else if (a.startsWith('--fps=')) o.fps = parseInt(a.slice(6));
    else if (a.startsWith('--size=')) o.size = a.slice(7);
    else if (a.startsWith('--view=')) o.view = a.slice(7);
    else if (a.startsWith('--steps=')) o.steps = parseInt(a.slice(8));
    else if (a.startsWith('--spf=')) o.spf = parseInt(a.slice(6));
    else if (a.startsWith('--warmup=')) o.warmup = parseInt(a.slice(9));
    else if (a.startsWith('--spin=')) o.spin = parseFloat(a.slice(7));
    else if (a.startsWith('--out=')) o.out = a.slice(6);
    else if (a.startsWith('--crf=')) o.crf = parseInt(a.slice(6));
    else if (a.startsWith('--encode=')) o.encode = a.slice(9);
    else if (a === '--gif') o.gif = true;
    else if (a === '--quiet') o.quiet = true;
    else if (a === '--keepOpen') o.keepOpen = true;
    else { console.error(`unknown argument: ${a}`); process.exit(2); }
  }
  return o;
}

async function ev(Runtime, expr, what, timeoutMs = 600000) {
  const r = await Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: timeoutMs });
  if (r.exceptionDetails) throw new Error(`${what}: ${r.exceptionDetails.text}`);
  return r.result.value;
}

// Write to a pipe that can say no. ffmpeg encodes slower than the GPU renders
// at small sizes and faster at large ones, so the backpressure is real in one
// direction or the other and ignoring it either grows an unbounded buffer or
// drops frames silently.
//
// The error handler is attached ONCE, by the caller, and this only ever waits
// for 'drain'. A `once('error')` per frame leaks a listener per frame on a
// socket that never errors -- 900 of them, and Node says so at eleven.
function write(stream, buf) {
  if (stream.write(buf)) return Promise.resolve();
  return new Promise((resolve) => stream.once('drain', resolve));
}

// PNG FRAMES ON THE PIPE, NOT RAW RGBA. See main-3d.js's encodePNG: the page
// produces a PNG in less time than it took to base64 the raw bytes, and ships
// ~23x fewer of them. image2pipe reads one PNG after another off a single
// stream and takes the size from the frames, so nothing here has to agree with
// the page about the geometry any more -- which is one fewer way to get a torn
// picture.
//
// `--encode=rgba` restores the RAW WIRE FORMAT and nothing else -- the faster
// base64 underneath it is unconditional, so that leg is not the pre-M6.5d tool
// and does not measure as one. Per frame, 800x600, a developed card wake:
// 146 ms before, 49 ms with the base64 fix alone, 13 ms with PNG as well.
function ffmpegArgs(o, w, h) {
  const input = o.encode === 'rgba'
    ? ['-y', '-f', 'rawvideo', '-pixel_format', 'rgba',
       '-video_size', `${w}x${h}`, '-framerate', String(o.fps), '-i', 'pipe:0']
    : ['-y', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(o.fps), '-i', 'pipe:0'];
  if (o.gif) {
    // Two passes over one read: `split` duplicates the decoded stream inside
    // the filter graph, so palettegen sees every frame without the input
    // being opened twice -- which a PIPE could not do anyway.
    return [...input, '-filter_complex',
      'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
      '-loop', '0', o.out];
  }
  return [...input, '-c:v', 'libx264', '-preset', 'medium', '-crf', String(o.crf),
          // yuv420p for players that will not touch anything else; it is also
          // why the frame size is forced EVEN below.
          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', o.out];
}

const q = (v) => (Number.isFinite(v) ? v.toFixed(3) : String(v));

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const preset = PRESETS[o.scenario] || { url: `scenario=${o.scenario}`, steps: 20000, view: 'volume', vol: 'vol=1' };
  const view = o.view || preset.view || 'volume';
  if (!['slice', 'volume'].includes(view)) { console.error(`--view=${view}: expected slice or volume`); process.exit(2); }
  if (!['png', 'rgba'].includes(o.encode)) { console.error(`--encode=${o.encode}: expected png or rgba`); process.exit(2); }
  const m = /^(\d+)x(\d+)$/.exec(o.size);
  if (!m) { console.error(`--size=${o.size}: expected WIDTHxHEIGHT`); process.exit(2); }
  // EVEN, because yuv420p subsamples chroma by two and an odd dimension is a
  // hard ffmpeg error rather than a rounding.
  let W = parseInt(m[1]) & ~1, H = parseInt(m[2]) & ~1;
  const frames = Math.max(1, Math.round(o.seconds * o.fps));
  // N frames hold N - 1 INTERVALS: the first is rendered at the state the
  // clip starts from, and every later one after a batch of steps. Dividing
  // by `frames` would quietly make the clip one batch short of the run it
  // says it covers, which for the card is 29 steps of a 26000-step fall --
  // invisible, and wrong in the direction of a number this file prints.
  const gaps = Math.max(1, frames - 1);
  const totalSteps = o.spf != null ? o.spf * gaps : (o.steps ?? preset.steps);
  const spf = o.spf ?? Math.max(1, Math.round(totalSteps / gaps));
  o.out = o.out || path.join(REPO_ROOT, `d3-${o.scenario}-${view}.${o.gif ? 'gif' : 'mp4'}`);
  if (o.gif && !o.out.endsWith('.gif')) { console.error('--gif needs an --out ending in .gif'); process.exit(2); }

  // ?live=0 because every step is driven from here. A page that also stepped
  // in its animation loop would put an unknown number of extra steps between
  // two frames, which is the exact property this tool exists to remove.
  const qs = [preset.url, 'live=0', `view=${view}`,
              view === 'volume' ? preset.vol : null, o.extra]
    .filter(Boolean).join('&');
  const url = `${o.baseUrl}/index-3d.html?${o.url || qs}`;

  console.log(`rendering ${frames} frames (${o.seconds}s at ${o.fps} fps), ${spf} steps/frame`
    + ` = ${spf * gaps} steps${o.warmup ? ` after a ${o.warmup}-step warmup` : ''}, ${W}x${H} ${view}`);
  console.log(`  ${url}`);
  console.log(`  -> ${o.out}`);

  const server = await ensureServer(o.baseUrl, REPO_ROOT);
  const chrome = await ensureChrome(o.port);
  if (chrome.started) await new Promise(r => setTimeout(r, 2000));
  const tabId = chrome.started ? await firstTab(o.port) : await openTab(o.port, 'about:blank');
  const client = await CDP({ port: o.port, target: tabId });
  const { Runtime, Page } = client;
  await Runtime.enable();
  await Page.enable();
  const watch = await attachPageWatch(client, {
    onError: (e) => console.error(`  !! [${e.kind}] ${e.text.split('\n')[0]}`),
  });

  let ff = null, encoded = 0, stopped = null;
  const track = [];
  try {
    await navigateTo(Page, url);
    await waitForGlobal(Runtime, 'window.__D3', 60000);
    await assertPageHealthy(Runtime, watch, o.scenario);
    const params = await ev(Runtime, 'window.__D3.getParams()', 'getParams');
    if (view === 'volume' && params.view !== 'volume') {
      throw new Error('the page did not come up in the volume view -- ?vol= built no volumes');
    }
    // The first frame settles the offscreen target's size, and the SLICE view
    // may choose its own height from the plane's aspect. Believe the page
    // rather than the request: an encoder fed a size the frames are not is a
    // torn picture, not an error.
    const probe = await ev(Runtime, `window.__D3.debugRenderFrame({ w: ${W}, h: ${H}, view: '${view}', encode: '${o.encode}' })`, 'debugRenderFrame');
    if (probe.skipped || probe.error) throw new Error(probe.skipped || probe.error);
    if (probe.w !== W || probe.h !== H) {
      console.log(`  page chose ${probe.w}x${probe.h} (the slice keeps its plane's aspect)`);
      W = probe.w & ~1; H = probe.h & ~1;
      if (W !== probe.w || H !== probe.h) {
        throw new Error(`the page's ${probe.w}x${probe.h} is odd and yuv420p needs even -- pass --size= explicitly`);
      }
    }
    if (o.warmup > 0) {
      process.stdout.write(`  warmup ${o.warmup} steps...`);
      await ev(Runtime, `window.__D3.debugStepSync(${o.warmup})`, 'debugStepSync');
      process.stdout.write(' done\n');
    }

    ff = spawn('ffmpeg', ffmpegArgs(o, W, H), { stdio: ['pipe', 'ignore', 'pipe'] });
    let ffErr = '';
    ff.stderr.on('data', (d) => { ffErr += d.toString(); if (ffErr.length > 8000) ffErr = ffErr.slice(-8000); });
    // One handler, for the life of the process: a write to a pipe whose far
    // end has died is an EPIPE, and unhandled it takes the tool down with a
    // stack trace instead of ffmpeg's own last words.
    let pipeBroke = null;
    ff.stdin.on('error', (e) => { pipeBroke = e; });
    const ffDone = new Promise((resolve, reject) => {
      ff.on('error', reject);
      ff.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}:\n${ffErr.split('\n').slice(-12).join('\n')}`))));
    });
    // Claimed immediately, because the loop below can throw: ffmpeg then
    // exits nonzero and rejects this promise with nobody awaiting it, which
    // Node reports as an unhandled rejection ON TOP OF the real error. The
    // real error is the one worth reading, so this one is swallowed here and
    // awaited for its own sake at the end.
    ffDone.catch(() => {});

    const t0 = Date.now();
    let lastLog = t0;
    // The camera turns over the WHOLE clip, not per frame, so --spin=90 means
    // a quarter turn from first frame to last whatever the frame count is.
    const dAzim = o.spin / Math.max(1, frames - 1);
    for (let i = 0; i < frames; i++) {
      if (i > 0) await ev(Runtime, `window.__D3.debugStepSync(${spf})`, 'debugStepSync');
      if (o.spin && view === 'volume' && i > 0) {
        await ev(Runtime, `window.__D3.debugSetCamera({ dAzim: ${dAzim} })`, 'debugSetCamera');
      }
      const img = await ev(Runtime,
        `window.__D3.debugRenderFrame({ w: ${W}, h: ${H}, view: '${view}', encode: '${o.encode}' })`, 'debugRenderFrame');
      if (img.skipped || img.error) throw new Error(img.skipped || img.error);
      await write(ff.stdin, Buffer.from(img.png || img.rgba, 'base64'));
      if (pipeBroke) throw new Error(`ffmpeg stopped reading (${pipeBroke.code}):\n${ffErr.split('\n').slice(-12).join('\n')}`);
      encoded++;

      // THE BODY, EVERY FRAME. A 150-byte readback, and it is what turns this
      // from a file-producer into something that reports: the trajectory is
      // the answer to "did it tumble", and a NaN in it is the answer to "did
      // the run survive the clip".
      if (params.hasBody) {
        const b = await ev(Runtime, 'window.__D3.readBody()', 'readBody');
        const finite = Number.isFinite(b.cx + b.cy + b.cz + b.qw + b.vx + b.wz);
        track.push({ step: img.step, ...b });
        if (!finite) { stopped = `the body state went non-finite at step ${img.step}`; break; }
      }
      const now = Date.now();
      if (!o.quiet && (now - lastLog > 2000 || i === frames - 1)) {
        lastLog = now;
        const rate = (i + 1) / ((now - t0) / 1000);
        const eta = (frames - i - 1) / Math.max(rate, 1e-6);
        process.stdout.write(`\r  frame ${i + 1}/${frames}  ${rate.toFixed(1)} fps`
          + `  eta ${Math.round(eta)}s        `);
      }
    }
    if (!o.quiet) process.stdout.write('\n');
    // A page that errored mid-clip (a pool refusal, a device error) has said
    // so in #status, and a clip encoded past that point is footage of a
    // broken run presented as a result.
    try { await assertPageHealthy(Runtime, watch, o.scenario); }
    catch (e) { stopped = stopped || e.message; }

    ff.stdin.end();
    await ffDone;
  } finally {
    if (ff && !ff.killed && ff.exitCode === null) { try { ff.stdin.end(); } catch (e) { /* already closed */ } }
    await client.close();
    await teardown({ port: o.port, tabId, chrome, server, keepOpen: o.keepOpen });
  }

  const st = fs.existsSync(o.out) ? fs.statSync(o.out) : null;
  console.log('\n' + '='.repeat(78));
  console.log(`${o.out}  ${st ? (st.size / 1048576).toFixed(1) + ' MiB' : 'MISSING'}`
    + `  ${encoded} frames, ${(encoded / o.fps).toFixed(1)}s at ${o.fps} fps`
    + `  ${spf} steps/frame`);
  if (track.length > 1) {
    const a = track[0], z = track[track.length - 1];
    // Total DISPLACEMENT, from the body's own dx/dy/dz, which accumulate
    // across the periodic seam -- the centre alone wraps under a moving
    // window and would report a plate that fell 1100 cells as having moved
    // eleven.
    const fell = z.dx - a.dx, across = Math.hypot(z.dy - a.dy, z.dz - a.dz);
    // TWO ROTATION NUMBERS, AND THE PAIR IS THE ANSWER. Integrating |omega|
    // alone counts a plate rocking back and forth as turning, which is
    // exactly the case this is trying to tell apart -- the first version of
    // this said a fluttering plate had "turned 0.44 revolutions", which is
    // true of the arc length and false of the plate.
    //
    //   net  = |INT omega dt| / 2pi   -- the VECTOR integral, so opposite
    //          swings cancel. About the fixed axis a tumbling plate turns
    //          about, this is the number of revolutions.
    //   arc  = INT |omega| dt / 2pi   -- how far it turned in total, sign
    //          ignored.
    //
    // Tumbling is net ~ arc and both above one. Fluttering is arc >> net.
    // Neither number is read from the QUATERNION, which says only where the
    // plate ended up and reads zero for one that made exactly one turn.
    const O = [0, 0, 0];
    let arc = 0;
    for (let i = 1; i < track.length; i++) {
      const p = track[i - 1], c = track[i], dt = c.step - p.step;
      O[0] += 0.5 * (p.wx + c.wx) * dt;
      O[1] += 0.5 * (p.wy + c.wy) * dt;
      O[2] += 0.5 * (p.wz + c.wz) * dt;
      arc += 0.5 * (Math.hypot(p.wx, p.wy, p.wz) + Math.hypot(c.wx, c.wy, c.wz)) * dt;
    }
    const net = Math.hypot(...O) / (2 * Math.PI);
    arc /= 2 * Math.PI;
    console.log(`body: fell ${fell.toFixed(1)} cells, drifted ${across.toFixed(1)} across`);
    console.log(`      rotation: ${net.toFixed(2)} net revolutions, ${arc.toFixed(2)} of arc`
      + `  (net/arc ${(net / Math.max(arc, 1e-12)).toFixed(2)})`);
    console.log(`      v = (${q(z.vx)}, ${q(z.vy)}, ${q(z.vz)})`
      + `  |omega| = ${Math.hypot(z.wx, z.wy, z.wz).toExponential(2)}  step ${z.step}`);
    // "It tumbled" is a claim with a threshold, so it is stated with both
    // numbers beside it rather than asserted.
    if (net > 0.75 && net / Math.max(arc, 1e-12) > 0.6) {
      console.log('      TUMBLING: it turned through more than three quarters of a revolution,');
      console.log('      and most of its rotation went one way.');
    } else if (arc > 0.25) {
      console.log(`      FLUTTERING: ${arc.toFixed(2)} revolutions of arc but only ${net.toFixed(2)} net,`);
      console.log('      i.e. it rocked rather than turned over.');
    } else {
      console.log(`      FALLING FLAT: ${arc.toFixed(2)} revolutions of arc. It barely turned at all.`);
    }
  }
  if (stopped) {
    console.log(`\nSTOPPED EARLY: ${stopped}`);
    console.log('The clip holds the frames up to that point, which is footage of a run that did');
    console.log('not survive its own length -- shorten --seconds, or raise tau (--extra=re=...).');
    process.exit(1);
  }
}

main().catch((e) => { console.error('\n' + (e.stack || e.message)); process.exit(1); });
