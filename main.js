const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');
// Size canvas to device pixels so it fills the screen on any DPR
const dpr = window.devicePixelRatio || 1;
canvas.width  = Math.round(canvas.clientWidth  * dpr);
canvas.height = Math.round(canvas.clientHeight * dpr);

const W = 256, H = 512;
const NCELLS = W * H;

// --- Pesavento & Wang (2004) ellipse parameters ---
const A     = 32;          // semi-major axis [lu]
const B     = 4;           // semi-minor axis [lu] → e = B/A = 0.125
const THETA = 0.2;         // initial angle from horizontal [rad]
const CX    = W / 2;       // ellipse centre x
const CY    = H / 2;       // ellipse centre y

// LBM parameters targeting Re = 1100
// ν = (τ - 0.5) * cs² = (τ - 0.5)/3
// Re = u * 2A / ν  →  u = Re * ν / (2A)
const TAU     = 0.52;
const NU      = (TAU - 0.5) / 3;
const U_INLET = 1100 * NU / (2 * A);   // ≈ 0.1146 lu/step

// body force: tiny constant x-acceleration to sustain flow against drag
// set empirically; ~1e-6 is a good starting point
const GX = 1e-6;

// D2Q9
const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  const u2 = ux*ux + uy*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - u2*1.5);
}

function buildSolid() {
  const s  = new Uint32Array(NCELLS);
  const ca = Math.cos(THETA), sa = Math.sin(THETA);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - CX, dy = y - CY;
      const lx = dx*ca + dy*sa;   // along major axis
      const ly = -dx*sa + dy*ca;  // along minor axis
      if ((lx/A)**2 + (ly/B)**2 <= 1) s[y*W + x] = 1;
    }
  }
  return s;
}

function initF() {
  const f  = new Float32Array(NCELLS * 9);
  const ux = U_INLET, uy = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const base = (y*W + x) * 9;
      for (let i = 0; i < 9; i++) f[base+i] = feq(1.0, ux, uy, i);
    }
  }
  return f;
}

async function loadShader(device, path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`failed to load ${path}`);
  return device.createShaderModule({ code: await r.text() });
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { statusEl.textContent = 'no adapter'; return; }
  const device = await adapter.requestDevice();
  const ctx    = canvas.getContext('webgpu');
  const fmt    = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: fmt, alphaMode: 'opaque' });

  // --- buffers ---
  const fSize    = NCELLS * 9 * 4;
  const velSize  = NCELLS * 2 * 4;
  const solSize  = NCELLS * 4;
  const U = GPUBufferUsage;

  const mkBuf = (size, usage) => device.createBuffer({ size, usage });

  const f_a    = mkBuf(fSize,   U.STORAGE | U.COPY_DST);
  const f_b    = mkBuf(fSize,   U.STORAGE);
  const f_c    = mkBuf(fSize,   U.STORAGE | U.COPY_SRC);
  const velBuf = mkBuf(velSize, U.STORAGE);
  const solBuf = mkBuf(solSize, U.STORAGE | U.COPY_DST);

  const solidData = buildSolid();
  device.queue.writeBuffer(solBuf, 0, solidData);
  device.queue.writeBuffer(f_a,   0, initF());

  // params: tau, u_inlet, gx, gy
  const paramsArr = new Float32Array([TAU, U_INLET, GX, 0.0]);
  const paramsBuf = mkBuf(16, U.UNIFORM | U.COPY_DST);
  device.queue.writeBuffer(paramsBuf, 0, paramsArr);

  // --- shaders ---
  const [colSM, strSM, renSM] = await Promise.all([
    loadShader(device, 'shaders/lbm_collide.wgsl'),
    loadShader(device, 'shaders/lbm_stream.wgsl'),
    loadShader(device, 'shaders/render.wgsl'),
  ]);

  // --- bind group layouts ---
  const colBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]});
  const strBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]});
  const renBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  ]});

  // --- pipelines ---
  const mkLayout = (...bgls) =>
    device.createPipelineLayout({ bindGroupLayouts: bgls });

  const colPL = device.createComputePipeline({
    layout: mkLayout(colBGL),
    compute: { module: colSM, entryPoint: 'main' },
  });
  const strPL = device.createComputePipeline({
    layout: mkLayout(strBGL),
    compute: { module: strSM, entryPoint: 'main' },
  });
  const renPL = device.createRenderPipeline({
    layout: mkLayout(renBGL),
    vertex:   { module: renSM, entryPoint: 'vs_main' },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }] },
    primitive: { topology: 'triangle-list' },
  });

  // --- bind groups ---
  const colBG = device.createBindGroup({ layout: colBGL, entries: [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: { buffer: f_a } },
    { binding: 2, resource: { buffer: f_b } },
    { binding: 3, resource: { buffer: velBuf } },
    { binding: 4, resource: { buffer: solBuf } },
  ]});
  const strBG = device.createBindGroup({ layout: strBGL, entries: [
    { binding: 0, resource: { buffer: f_b } },
    { binding: 1, resource: { buffer: f_c } },
    { binding: 2, resource: { buffer: solBuf } },
  ]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [
    { binding: 0, resource: { buffer: velBuf } },
    { binding: 1, resource: { buffer: solBuf } },
  ]});

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  let step = 0, lastT = performance.now();

  function frame() {
    const stepsPerFrame = 8;
    const enc = device.createCommandEncoder();

    for (let s = 0; s < stepsPerFrame; s++) {
      const col = enc.beginComputePass();
      col.setPipeline(colPL); col.setBindGroup(0, colBG);
      col.dispatchWorkgroups(WGX, WGY); col.end();

      const str = enc.beginComputePass();
      str.setPipeline(strPL); str.setBindGroup(0, strBG);
      str.dispatchWorkgroups(WGX, WGY); str.end();

      enc.copyBufferToBuffer(f_c, 0, f_a, 0, fSize);
    }
    step += stepsPerFrame;

    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r:0.07, g:0.07, b:0.10, a:1 },
      loadOp: 'clear', storeOp: 'store',
    }]});
    rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

    device.queue.submit([enc.finish()]);

    const now = performance.now();
    if (now - lastT > 300) {
      statusEl.textContent =
        `step ${step}  Re≈1100  τ=${TAU}  u_in=${U_INLET.toFixed(4)}`;
      lastT = now;
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

init().catch(e => {
  statusEl.textContent = `error: ${e.message}`;
  statusEl.style.color = '#f77';
  console.error(e);
});
