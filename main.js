const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');
const dpr = window.devicePixelRatio || 1;
canvas.width  = Math.round(canvas.clientWidth  * dpr);
canvas.height = Math.round(canvas.clientHeight * dpr);

const W = 256, H = 512, NCELLS = W * H;

const A = 32, B = 4;
const I_STAR = 0.17;
const RHO_B  = I_STAR * 2 * A**3 / (B * (A**2 + B**2));
const MASS   = RHO_B * Math.PI * A * B;
const I_BODY = RHO_B * Math.PI * A * B * (A**2 + B**2) / 4;

const TAU     = 0.52;
const U_T     = 0.05;
const G_LU    = U_T**2 / (Math.PI * B * (RHO_B - 1));
const G_EFF   = G_LU * (1 - 1 / RHO_B);

const FSCALE  = 1e3;

const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    const base = c * 9;
    for (let i = 0; i < 9; i++) f[base+i] = feq(1, 0, 0, i);
  }
  return f;
}

async function loadShader(device, path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`failed to load ${path}`);
  const code = await r.text();
  return device.createShaderModule({ code });
}

function handleErr(e) {
  statusEl.textContent = `error: ${e.message}`;
  statusEl.style.color = '#f77';
  console.error(e);
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: fmt, alphaMode: 'opaque' });

  const U = GPUBufferUsage;
  const fSize   = NCELLS * 9 * 4;
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE });
  const f_c     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_SRC });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 20 floats
  const cardStateBuf   = device.createBuffer({ size: 80, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const cardStateStage = device.createBuffer({ size: 80, usage: U.MAP_READ | U.COPY_DST });

  const cardInit = new Float32Array([
    W/2, H/4, 0.2,   // cx, cy, theta
    0, 0, 0,         // vx, vy, omega
    0, 0, 0,         // fx, fy, tz
    MASS, I_BODY, G_EFF,
    A, B,
    0.3, 0.05,       // v_max, o_max
    W/2, H/4, 0.2,   // cx_old, cy_old, th_old
    TAU              // tau (reusing _p0)
  ]);
  device.queue.writeBuffer(cardStateBuf, 0, cardInit);
  device.queue.writeBuffer(f_a, 0, initF());

  const [colSM, strSM, frcSM, phySM, renSM] = await Promise.all([
    loadShader(device, 'shaders/lbm_collide.wgsl'),
    loadShader(device, 'shaders/lbm_stream.wgsl'),
    loadShader(device, 'shaders/lbm_force.wgsl'),
    loadShader(device, 'shaders/physics.wgsl'),
    loadShader(device, 'shaders/render.wgsl'),
  ]);

  const mkBGL = (entries) => device.createBindGroupLayout({ entries });
  const mkBG  = (layout, entries) => device.createBindGroup({ layout, entries });
  
  const colBGL = mkBGL([
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]);
  const strBGL = mkBGL([
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]);
  const frcBGL = mkBGL([
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]);
  const phyBGL = mkBGL([
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]);
  const renBGL = mkBGL([
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]);

  const colPL = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [colBGL] }), compute: { module: colSM, entryPoint: 'main' } });
  const strPL = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [strBGL] }), compute: { module: strSM, entryPoint: 'main' } });
  const frcPL = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }), compute: { module: frcSM, entryPoint: 'main' } });
  const phyPL = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [phyBGL] }), compute: { module: phySM, entryPoint: 'main' } });

  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main' },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }] },
    primitive: { topology: 'triangle-list' },
  });

  const colBG = mkBG(colBGL, [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]);
  const strBG = mkBG(strBGL, [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_c } }]);
  const frcBG = mkBG(frcBGL, [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }]);
  const phyBG = mkBG(phyBGL, [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]);
  const renBG = mkBG(renBGL, [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }]);

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const STEPS_PER_FRAME = 8;
  let step = 0, lastT = performance.now();

  async function frame() {
    const enc = device.createCommandEncoder();
    for (let s = 0; s < STEPS_PER_FRAME; s++) {
      const col = enc.beginComputePass(); col.setPipeline(colPL); col.setBindGroup(0, colBG); col.dispatchWorkgroups(WGX, WGY); col.end();
      const str = enc.beginComputePass(); str.setPipeline(strPL); str.setBindGroup(0, strBG); str.dispatchWorkgroups(WGX, WGY); str.end();
      enc.copyBufferToBuffer(f_c, 0, f_a, 0, fSize);
      const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
      const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
    }
    step += STEPS_PER_FRAME;

    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
    rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();
    
    enc.copyBufferToBuffer(cardStateBuf, 0, cardStateStage, 0, 80);
    device.queue.submit([enc.finish()]);

    if (performance.now() - lastT > 300) {
      await cardStateStage.mapAsync(GPUMapMode.READ);
      const d = new Float32Array(cardStateStage.getMappedRange());
      statusEl.textContent = `step ${step}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`;
      cardStateStage.unmap();
      lastT = performance.now();
    }
    requestAnimationFrame(() => frame().catch(handleErr));
  }
  frame().catch(handleErr);
}
init().catch(handleErr);
