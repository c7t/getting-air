const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');
const dpr = window.devicePixelRatio || 1;
canvas.width  = Math.round(canvas.clientWidth  * dpr);
canvas.height = Math.round(canvas.clientHeight * dpr);

// ── Grid ──────────────────────────────────────────────────────────────────────
const W = 256, H = 512, NCELLS = W * H;

// ── Pesavento & Wang (2004) physical parameters ───────────────────────────────
const A     = 32;        // semi-major axis [lu]
const B     = 4;         // semi-minor axis [lu]  e = B/A = 0.125
const E     = B / A;     // aspect ratio

// Dimensionless moment of inertia: I* = b(a²+b²)ρ_b / (2a³ρ_f)  = 0.17
// → ρ_b/ρ_f = I* · 2a³ / (b·(a²+b²))
const I_STAR = 0.17;
const RHO_B  = I_STAR * 2 * A**3 / (B * (A**2 + B**2));  // ≈ 2.678
const PI_AB  = Math.PI * A * B;

const MASS   = RHO_B * PI_AB;                          // card mass [lu²]
const I_BODY = RHO_B * PI_AB * (A**2 + B**2) / 4;     // moment of inertia

// LBM parameters.
// Re=1100 needs τ≈0.509 which is BGK-unstable in the transient.
// Use τ=0.6 (Re≈220 at u_t=0.05) for Phase 3 stability; will tighten τ later.
const TAU     = 0.6;
const NU      = (TAU - 0.5) / 3;                       // ≈ 0.0333
// Target u_t small enough to keep Ma < 0.1 during free-fall transient
const U_T     = 0.05;                                  // lu/step
// u_t = sqrt(π·b·g·(ρ_b/ρ_f − 1))  →  g = u_t² / (π·b·(ρ_b−1))
const G_LU    = U_T**2 / (Math.PI * B * (RHO_B - 1)); // ≈ 1.19e-4 lu/step²
// Net downward acceleration on card (gravity minus buoyancy)
const G_EFF   = G_LU * (1 - 1 / RHO_B);               // ≈ 7.44e-5

// Force scale for int32 atomic accumulation
const FSCALE  = 1e3;

const CARD_ENABLED = true;

// ── D2Q9 (JS side for initialisation) ────────────────────────────────────────
const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

// ── Card state ────────────────────────────────────────────────────────────────
let cardCX    = W / 2;
let cardCY    = H / 4;      // start in upper quarter, falls down
let cardTheta = 0.2;        // initial angle [rad], matches the paper
let cardVX    = 0;
let cardVY    = 0;
let cardOmega = 0;

function buildSolid(cx, cy, theta) {
  const s  = new Uint32Array(NCELLS);
  if (!CARD_ENABLED) return s;   // fluid-only sanity check mode
  const ca = Math.cos(theta), sa = Math.sin(theta);
  const halfLen = A - B;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let dx = x - cx;
      let dy = y - cy;
      // Minimum image convention for toroidal wrapping
      dx -= W * Math.round(dx / W);
      dy -= H * Math.round(dy / H);

      const lx = dx*ca + dy*sa;
      const ly = -dx*sa + dy*ca;
      const capDist = Math.max(0, Math.abs(lx) - halfLen);
      if (capDist*capDist + ly*ly <= B*B) s[y*W + x] = 1;
    }
  }
  return s;
}

function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    const base = c * 9;
    for (let i = 0; i < 9; i++) f[base+i] = feq(1, 0, 0, i);
  }
  return f;
}

function makeParams(cx, cy) {
  // Must match Params struct in all shaders: 12 floats = 48 bytes
  return new Float32Array([TAU, 0, 0, cx, cy, A, cardTheta,
                           cardVX, cardVY, cardOmega, 0, 0]);
}

async function loadShader(device, path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`failed to load ${path}`);
  return device.createShaderModule({ code: await r.text() });
}

// ── WebGPU init ───────────────────────────────────────────────────────────────
async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { statusEl.textContent = 'no adapter'; return; }
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: fmt, alphaMode: 'opaque' });

  const U = GPUBufferUsage;
  const mkBuf = (size, usage) => device.createBuffer({ size, usage });

  const fSize   = NCELLS * 9 * 4;
  const velSize = NCELLS * 2 * 4;
  const solSize = NCELLS * 4;

  const f_a       = mkBuf(fSize,   U.STORAGE | U.COPY_DST);
  const f_b       = mkBuf(fSize,   U.STORAGE);
  const f_c       = mkBuf(fSize,   U.STORAGE | U.COPY_SRC);
  const velBuf    = mkBuf(velSize, U.STORAGE);
  const solBuf    = mkBuf(solSize, U.STORAGE | U.COPY_DST);
  // Force accumulator: [Fx, Fy, Tz, pad] as atomic i32
  const forceBuf  = mkBuf(16, U.STORAGE | U.COPY_SRC | U.COPY_DST);
  const forceStage= mkBuf(16, U.COPY_DST | U.MAP_READ);

  let solidData = buildSolid(cardCX, cardCY, cardTheta);
  device.queue.writeBuffer(solBuf, 0, solidData);
  device.queue.writeBuffer(f_a,   0, initF());

  const paramsBuf = mkBuf(48, U.UNIFORM | U.COPY_DST);
  device.queue.writeBuffer(paramsBuf, 0, makeParams(cardCX, cardCY));

  const [colSM, strSM, frcSM, renSM] = await Promise.all([
    loadShader(device, 'shaders/lbm_collide.wgsl'),
    loadShader(device, 'shaders/lbm_stream.wgsl'),
    loadShader(device, 'shaders/lbm_force.wgsl'),
    loadShader(device, 'shaders/render.wgsl'),
  ]);

  // ── Bind group layouts ─────────────────────────────────────────────────────
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
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});
  const frcBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});
  const renBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
  ]});

  const mkPL = (...bgls) =>
    device.createPipelineLayout({ bindGroupLayouts: bgls });

  const colPL = device.createComputePipeline({ layout: mkPL(colBGL), compute: { module: colSM, entryPoint: 'main' } });
  const strPL = device.createComputePipeline({ layout: mkPL(strBGL), compute: { module: strSM, entryPoint: 'main' } });
  const frcPL = device.createComputePipeline({ layout: mkPL(frcBGL), compute: { module: frcSM, entryPoint: 'main' } });
  const renPL = device.createRenderPipeline({
    layout: mkPL(renBGL),
    vertex:   { module: renSM, entryPoint: 'vs_main' },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }] },
    primitive: { topology: 'triangle-list' },
  });

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
    { binding: 3, resource: { buffer: paramsBuf } },
  ]});
  const frcBG = device.createBindGroup({ layout: frcBGL, entries: [
    { binding: 0, resource: { buffer: f_b } },   // post-collision distributions
    { binding: 1, resource: { buffer: solBuf } },
    { binding: 2, resource: { buffer: forceBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
  ]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [
    { binding: 0, resource: { buffer: velBuf } },
    { binding: 1, resource: { buffer: solBuf } },
    { binding: 2, resource: { buffer: paramsBuf } },
  ]});

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  const STEPS_PER_FRAME = 8;

  let step = 0, lastT = performance.now();

  // ── Rigid body integration ─────────────────────────────────────────────────
  function integrateBody(Fx, Fy, Tz, dt) {
    // Fx/Fy/Tz are the total impulse accumulated over dt LBM steps.
    const V_MAX = 0.30;   // Increased to allow full transient (Ma < 0.5)
    const O_MAX = 0.05;   // Increased to allow faster rotation

    cardVX    += Fx / MASS;
    cardVY    += (Fy + MASS * G_EFF * dt) / MASS;
    cardOmega += Tz / I_BODY;

    // Clamp to physical bounds
    cardVX    = Math.max(-V_MAX, Math.min(V_MAX,  cardVX));
    cardVY    = Math.max(-V_MAX, Math.min(V_MAX,  cardVY));
    cardOmega = Math.max(-O_MAX, Math.min(O_MAX,  cardOmega));

    cardCX    += cardVX * dt;
    cardCY    += cardVY * dt;
    cardTheta += cardOmega * dt;
    cardCX = ((cardCX % W) + W) % W;
    cardCY = ((cardCY % H) + H) % H;
  }

  function uploadSolid(cx, cy, theta) {
    const oldSolid = solidData;
    solidData = buildSolid(cx, cy, theta);

    // Refill solid→fluid transitions with card-velocity feq
    const fRefill = new Float32Array(9);
    for (let idx = 0; idx < NCELLS; idx++) {
      if (oldSolid[idx] === 1 && solidData[idx] === 0) {
        for (let i = 0; i < 9; i++) fRefill[i] = feq(1, cardVX, cardVY, i);
        device.queue.writeBuffer(f_a, idx * 9 * 4, fRefill);
      }
    }
    device.queue.writeBuffer(solBuf,   0, solidData);
    device.queue.writeBuffer(paramsBuf, 0, makeParams(cx, cy));
  }

  // ── Render loop — synchronous force readback ─────────────────────────────────
  // Await mapAsync before scheduling next frame so forceStage is never mapped
  // when the GPU encoder copies to it.
  async function frame() {
    const enc = device.createCommandEncoder();
    enc.clearBuffer(forceBuf, 0, 16);

    for (let s = 0; s < STEPS_PER_FRAME; s++) {
      const col = enc.beginComputePass();
      col.setPipeline(colPL); col.setBindGroup(0, colBG);
      col.dispatchWorkgroups(WGX, WGY); col.end();

      const str = enc.beginComputePass();
      str.setPipeline(strPL); str.setBindGroup(0, strBG);
      str.dispatchWorkgroups(WGX, WGY); str.end();

      enc.copyBufferToBuffer(f_c, 0, f_a, 0, fSize);

      const frc = enc.beginComputePass();
      frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG);
      frc.dispatchWorkgroups(WGX, WGY); frc.end();
    }
    step += STEPS_PER_FRAME;

    enc.copyBufferToBuffer(forceBuf, 0, forceStage, 0, 16);

    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r:0.07, g:0.07, b:0.10, a:1 },
      loadOp: 'clear', storeOp: 'store',
    }]});
    rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

    device.queue.submit([enc.finish()]);

    // Block until GPU has written forces into forceStage, then read back.
    await forceStage.mapAsync(GPUMapMode.READ);
    const d  = new Int32Array(forceStage.getMappedRange());
    const Fx = d[0] / FSCALE, Fy = d[1] / FSCALE, Tz = d[2] / FSCALE;
    forceStage.unmap();

    integrateBody(Fx, Fy, Tz, STEPS_PER_FRAME);
    uploadSolid(cardCX, cardCY, cardTheta);

    if (performance.now() - lastT > 300) {
      statusEl.textContent = CARD_ENABLED
        ? `step ${step}  vy=${cardVY.toFixed(4)}  Fy=${Fy.toExponential(2)}  Tz=${Tz.toExponential(2)}  θ=${cardTheta.toFixed(2)}`
        : `step ${step}  (no card)`;
      lastT = performance.now();
    }

    // Schedule next frame only after GPU work + readback are complete.
    requestAnimationFrame(() => frame().catch(handleErr));
  }

  function handleErr(e) {
    statusEl.textContent = `error: ${e.message}`;
    statusEl.style.color = '#f77';
    console.error(e);
  }

  frame().catch(handleErr);
}

init().catch(e => {
  statusEl.textContent = `error: ${e.message}`;
  statusEl.style.color = '#f77';
  console.error(e);
});
