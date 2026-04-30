const canvas  = document.getElementById('c');
const statusEl = document.getElementById('status');
canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;

const W = 256, H = 512;
const NCELLS = W * H;

// D2Q9 weights and velocities (JS side, for initialization)
const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  const u2 = ux*ux + uy*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - u2*1.5);
}

function initF(ux0 = 0, uy0 = 0) {
  const f = new Float32Array(NCELLS * 9);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // small sinusoidal vortex perturbation
      const eps = 0.05;
      const ux = ux0 + eps * Math.sin(2*Math.PI*y/H) * Math.cos(2*Math.PI*x/W);
      const uy = uy0 - eps * Math.cos(2*Math.PI*y/H) * Math.sin(2*Math.PI*x/W);
      const base = (y*W + x)*9;
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
  const fSize  = NCELLS * 9 * 4;   // bytes
  const velSize = NCELLS * 2 * 4;

  const mkBuf = (size, usage) => device.createBuffer({ size, usage });
  const U = GPUBufferUsage;

  const f_a   = mkBuf(fSize,   U.STORAGE | U.COPY_DST);
  const f_b   = mkBuf(fSize,   U.STORAGE);              // f_col (post-collision)
  const f_c   = mkBuf(fSize,   U.STORAGE | U.COPY_SRC); // f_out (post-stream) → copied to f_a
  const velBuf = mkBuf(velSize, U.STORAGE);

  // upload initial f
  device.queue.writeBuffer(f_a, 0, initF());

  // params uniform: tau, gx, gy, pad
  const paramsArr = new Float32Array([0.60, 0.0, 0.0, 0.0]);
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
  ]});
  const strBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const renBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  ]});

  // --- pipelines ---
  const colPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [colBGL] }),
    compute: { module: colSM, entryPoint: 'main' },
  });
  const strPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [strBGL] }),
    compute: { module: strSM, entryPoint: 'main' },
  });
  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex:   { module: renSM, entryPoint: 'vs_main' },
    fragment: { module: renSM, entryPoint: 'fs_main',
      targets: [{ format: fmt }] },
    primitive: { topology: 'triangle-list' },
  });

  // --- bind groups (collide reads f_a, writes f_b + vel) ---
  const colBG = device.createBindGroup({ layout: colBGL, entries: [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: { buffer: f_a } },
    { binding: 2, resource: { buffer: f_b } },
    { binding: 3, resource: { buffer: velBuf } },
  ]});
  const strBG = device.createBindGroup({ layout: strBGL, entries: [
    { binding: 0, resource: { buffer: f_b } },
    { binding: 1, resource: { buffer: f_c } },
  ]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [
    { binding: 0, resource: { buffer: velBuf } },
  ]});

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);

  let step = 0;
  let lastT = performance.now();

  function frame() {
    // run several LBM steps per render frame for speed
    const stepsPerFrame = 4;
    const enc = device.createCommandEncoder();

    for (let s = 0; s < stepsPerFrame; s++) {
      // collision
      const col = enc.beginComputePass();
      col.setPipeline(colPL);
      col.setBindGroup(0, colBG);
      col.dispatchWorkgroups(WGX, WGY);
      col.end();

      // streaming
      const str = enc.beginComputePass();
      str.setPipeline(strPL);
      str.setBindGroup(0, strBG);
      str.dispatchWorkgroups(WGX, WGY);
      str.end();

      // copy f_c → f_a for next step
      enc.copyBufferToBuffer(f_c, 0, f_a, 0, fSize);
    }
    step += stepsPerFrame;

    // render
    const rp = enc.beginRenderPass({ colorAttachments: [{
      view:      ctx.getCurrentTexture().createView(),
      clearValue: { r: 0.07, g: 0.07, b: 0.10, a: 1 },
      loadOp:  'clear', storeOp: 'store',
    }]});
    rp.setPipeline(renPL);
    rp.setBindGroup(0, renBG);
    rp.draw(6);
    rp.end();

    device.queue.submit([enc.finish()]);

    const now = performance.now();
    if (now - lastT > 200) {
      statusEl.textContent = `step ${step}  τ=${paramsArr[0].toFixed(3)}`;
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
