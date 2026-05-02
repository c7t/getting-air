const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const W = 512, H = 512, NCELLS = W * H;

// ── Pesavento & Wang (2004) physical parameters ───────────────────────────────
// These constants define the "regime" of the simulation (Falling Paper).

// Paper: "Falling Paper: Navigating the Trade-Off between Density and Aspect Ratio"
// Semi-axes: a=32, b=4 [lu].  Aspect ratio e = b/a = 0.125.
// Geometry: Semi-major (A) and semi-minor (B) axes of the ellipse in lattice units.
// The card is 2*A long and 2*B thick.
let A = 64, B = 8;

// Dimensionless moment of inertia: I* = b(a²+b²)ρ_b / (2a³ρ_f)  = 0.17
// → ρ_b/ρ_f = I* · 2a³ / (b·(a²+b²)) ≈ 2.678
// This characterizes the rotation dynamics. A value of 0.17 is typical for 
// a card whose mass distribution allows for stable tumbling.
let I_STAR = 0.34;

// TAU: LBM Relaxation Time. 
// Related to Kinematic Viscosity (ν) by: ν = (TAU - 0.5) / 3.
// TAU = 0.5 corresponds to zero viscosity (unstable). 
// TAU = 0.52 is "thin" fluid (high Reynolds number, e.g., Re ≈ 1100).
// Target Re = 1100 requires τ ≈ 0.509; we start at 0.52 for stability.
let TAU = 0.509;

// U_T: Target Terminal Velocity (in lattice units per step).
// Target u_t small enough to keep Ma < 0.1 during free-fall transient.
// We aim for 0.05 so that even during fast tumbles, the tip velocity
// stays well below the Mach limit (Ma < 0.3) where LBM becomes inaccurate.
let U_T = 0.05;

let RHO_B, MASS, I_BODY, G_LU, G_EFF;

function recalculate() {
  // RHO_B: Solid-to-fluid density ratio (ρ_body / ρ_fluid).
  // Calculated to satisfy the I_STAR requirement. In LBM, fluid density is 1.0.
  // Higher RHO_B makes the card "heavier" and less affected by small fluid gusts.
  RHO_B  = I_STAR * 2 * A**3 / (B * (A**2 + B**2));
  RHO_B  = Math.max(1.05, RHO_B);

  // MASS: Total mass of the 2D ellipse (Area * Density).
  MASS   = RHO_B * Math.PI * A * B;

  // I_BODY: Moment of inertia for a 2D ellipse. 
  // Determines how much torque is needed to change the card's rotation speed.
  I_BODY = RHO_B * Math.PI * A * B * (A**2 + B**2) / 4;

  // G_LU: Raw Gravity. 
  // The gravitational constant needed to reach U_T against viscous drag.
  G_LU   = U_T**2 / (Math.PI * B * (RHO_B - 1));

  // G_EFF: Effective Gravity (Buoyancy-corrected).
  // In a coupled simulation, the fluid pushes up on the card. 
  // G_EFF accounts for the weight of the card minus the weight of the displaced fluid.
  G_EFF  = G_LU * (1 - 1 / RHO_B);
}
recalculate();


// FSCALE: Atomic Scaling Factor.
// Used to convert floating-point forces/torques to integers for the GPU atomics.
// Must be large enough for precision (1e4 = 0.0001 precision) but small enough
// to avoid 32-bit integer overflow when summing 1000s of cells.
const FSCALE  = 1e4;

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
  const r = await fetch(path + '?v=' + Date.now());
  if (!r.ok) throw new Error(`failed to load ${path}`);
  const code = await r.text();
  return device.createShaderModule({ code });
}

function handleErr(e) {
  statusEl.textContent = `error: ${e.message}`;
  statusEl.style.color = '#f77';
  console.error('WebGPU Error:', e);
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }
  
  const hasTimestamp = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamp ? ['timestamp-query'] : []
  });

  const querySet = hasTimestamp ? device.createQuerySet({
    type: 'timestamp',
    count: 2
  }) : null;
  const queryResolveBuffer = hasTimestamp ? device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
  }) : null;
  const queryReadBuffer = hasTimestamp ? device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  }) : null;
  
  device.pushErrorScope('validation');

  const ctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    ctx.configure({ device, format: fmt, alphaMode: 'opaque' });
  }
  window.addEventListener('resize', resize);
  resize();

  const U = GPUBufferUsage;
  const fSize   = NCELLS * 9 * 4;
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE });
  const f_c     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_SRC });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 26 floats = 104 bytes
  const cardStateBuf   = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const cardStateStage = device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST });

  const cardInit = new Float32Array([
    W/2, H/2, 0.2,   // cx, cy, theta
    0, 0, 0,         // vx, vy, omega
    0, 0, 0,         // fx, fy, tz
    MASS, I_BODY, G_EFF,
    A, B,
    0.3, 0.025,      // v_max, o_max
    W/2, H/2, 0.2,   // cx_old, cy_old, th_old
    TAU,             // tau
    0, 0,            // y_total, x_total
    0, 0, 0, 0       // off_x, off_y, off_x_old, off_y_old
  ]);
  device.queue.writeBuffer(cardStateBuf, 0, cardInit);
  device.queue.writeBuffer(f_a, 0, initF());

  let paramsDirty = false;
  const updateGPUParams = () => {
    const data = new Float32Array([MASS, I_BODY, G_EFF, A, B]);
    device.queue.writeBuffer(cardStateBuf, 9 * 4, data);
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
  };

  const sliders = [
    { id: 'A', setter: v => A = v, dp: 0 },
    { id: 'B', setter: v => B = v, dp: 0 },
    { id: 'I_STAR', setter: v => I_STAR = v, dp: 2 },
    { id: 'TAU', setter: v => TAU = v, dp: 3 },
    { id: 'U_T', setter: v => U_T = v, dp: 3 },
  ];
  sliders.forEach(s => {
    const el = document.getElementById(`slider-${s.id}`);
    const valEl = document.getElementById(`val-${s.id}`);
    el.oninput = () => {
      s.setter(parseFloat(el.value));
      valEl.textContent = el.value;
      recalculate();
      paramsDirty = true;
    };
  });

  const [colSM, strSM, frcSM, phySM, renSM] = await Promise.all([
    loadShader(device, 'shaders/lbm_collide.wgsl'),
    loadShader(device, 'shaders/lbm_stream.wgsl'),
    loadShader(device, 'shaders/lbm_force.wgsl'),
    loadShader(device, 'shaders/physics.wgsl'),
    loadShader(device, 'shaders/render.wgsl'),
  ]);

  const colBGL = device.createBindGroupLayout({ label: 'colBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const strBGL = device.createBindGroupLayout({ label: 'strBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const frcBGL = device.createBindGroupLayout({ label: 'frcBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const phyBGL = device.createBindGroupLayout({ label: 'phyBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

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

  const colBG = device.createBindGroup({ layout: colBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const strBG = device.createBindGroup({ layout: strBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_c } }]});
  const frcBG = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }]});
  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();

  const trajectory = [];

  document.getElementById('download').onclick = () => {
    const header = "step,cx,cy_total,cx_total,theta,vx,vy,omega,fx,fy,tz\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trajectory_${W}x${H}.csv`;
    a.click();
  };

  const mlupsEl = document.getElementById('val-mlups');
  const gpuMsEl = document.getElementById('val-gpu-ms');
  const syncMsEl = document.getElementById('val-sync-ms');

  async function frame() {
    try {
      if (paramsDirty) {
        updateGPUParams();
        paramsDirty = false;
      }
      const enc = device.createCommandEncoder();
      
      if (hasTimestamp) {
        enc.writeTimestamp(querySet, 0);
      }

      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        const col = enc.beginComputePass(); col.setPipeline(colPL); col.setBindGroup(0, colBG); col.dispatchWorkgroups(WGX, WGY); col.end();
        const str = enc.beginComputePass(); str.setPipeline(strPL); str.setBindGroup(0, strBG); str.dispatchWorkgroups(WGX, WGY); str.end();
        enc.copyBufferToBuffer(f_c, 0, f_a, 0, fSize);
        const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
        const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
      }
      step += STEPS_PER_FRAME;

      if (hasTimestamp) {
        enc.writeTimestamp(querySet, 1);
        enc.resolveQuerySet(querySet, 0, 2, queryResolveBuffer, 0);
        enc.copyBufferToBuffer(queryResolveBuffer, 0, queryReadBuffer, 0, 16);
      }

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();
      
      enc.copyBufferToBuffer(cardStateBuf, 0, cardStateStage, 0, 104);
      
      const tSubmit = performance.now();
      device.queue.submit([enc.finish()]);

      // Read back state every frame for trajectory
      await cardStateStage.mapAsync(GPUMapMode.READ);
      const tRead = performance.now();
      
      const d = new Float32Array(cardStateStage.getMappedRange());
      
      let gpuTime = 0;
      if (hasTimestamp) {
        await queryReadBuffer.mapAsync(GPUMapMode.READ);
        const timestamps = new BigUint64Array(queryReadBuffer.getMappedRange());
        gpuTime = Number(timestamps[1] - timestamps[0]) / 1e6;
        queryReadBuffer.unmap();
      } else {
        // Fallback to CPU-side timing of the submission block
        gpuTime = tRead - tSubmit;
      }

      // Update trajectory
      if (step < 100000) {
        // Record: step, cx, cy_total, cx_total, theta, vx, vy, omega, fx, fy, tz
        trajectory.push([step, d[0], d[20], d[21], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]);
      }
      
      if (performance.now() - lastT > 250) {
        const mlups = (NCELLS * STEPS_PER_FRAME) / (gpuTime * 1e3);
        mlupsEl.textContent = mlups.toFixed(1);
        gpuMsEl.textContent = gpuTime.toFixed(2);
        syncMsEl.textContent = (tRead - tSubmit).toFixed(2);
        statusEl.textContent = `step ${step}  y=${d[20].toFixed(1)}  x=${d[21].toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`;
        lastT = performance.now();
      }
      cardStateStage.unmap();

      requestAnimationFrame(() => frame().catch(handleErr));
    } catch (e) {
      handleErr(e);
    }
  }
  frame().catch(handleErr);
}
init().catch(handleErr);
