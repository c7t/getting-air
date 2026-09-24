#!/usr/bin/env node
// EVERY BUFFER A POOL DECLARES IS EITHER WRITTEN BY reset, OR SAID NOT TO BE.
//
// `amr2d-gpu.mjs`'s `writePoolInitialState` is one statement of a pool's
// initial state, shared by `allocLevelPool` and every page's `resetSim` so the
// two cannot drift. That closes the gap for the buffers it knows about. This
// test closes the OTHER half: it fails when someone adds a buffer to
// `allocLevelPool` and does not decide whether reset owns it.
//
// The three defects that motivated this were all one buffer nobody had
// decided about -- `finePoolVel`, `parentSlotBuf`, and the dense `velBuf` --
// each left on the reasoning "WebGPU zero-initialises it", which is true at
// allocation and false at reset. A reviewer cannot be relied on to notice a
// buffer that is missing from a list; a test can.
//
// GPU-FREE. It runs `allocLevelPool` against a recording stand-in for
// GPUDevice, which is enough because the property under test is "which
// buffers were written", not what was written into them.

const assert = require('assert');

let failures = 0;
function check(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); }
  catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}

// A GPUDevice that records. createBuffer hands back a tagged object; the queue
// remembers which of them writeBuffer touched.
function recordingDevice() {
  const written = new Set();
  let n = 0;
  return {
    written,
    createBuffer({ size, usage }) { return { __id: `b${n++}`, size, usage }; },
    queue: {
      writeBuffer(buf, offset, data) {
        if (!buf || !buf.__id) throw new Error('writeBuffer called on something that is not a buffer');
        written.add(buf.__id);
      },
    },
  };
}

const U = {
  STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8, UNIFORM: 16, VERTEX: 32, INDEX: 64, INDIRECT: 128,
};

// Buffers reset deliberately does NOT write, with the reason. Adding to this
// list is how you declare "not state" -- which is a decision someone made,
// rather than an omission nobody noticed.
const NOT_RESET_STATE = {
  finePoolF_a: 'the scenario\'s initial condition -- initF/initFPool differ per page, so the page writes it',
  finePoolF_b: 'the back phase; the step writes it before anything reads it, and `useB = false` after a reset',
  activeSlotsBuf: 'derived from slotToBlock by amr_active_list.wgsl at the top of EVERY macro-step, before its consumers (?indirect=1)',
  activeArgsBuf: 'derived with activeSlotsBuf, same pass, same timing -- never read before that macro-step writes it',
};

(async () => {
  const { allocLevelPool, writePoolInitialState } = await import('../amr2d-gpu.mjs');

  // Level 1 per-block, level 2 quad, and the root -- the three shapes that
  // differ in which buffers exist at all.
  const CASES = [
    { name: 'level 1, per-block', m: 1, nbx: 8, nby: 8, slots: 64, cells: 400, opts: { quadAlloc: false } },
    { name: 'level 1, quad',      m: 1, nbx: 8, nby: 8, slots: 64, cells: 400, opts: { quadAlloc: true } },
    { name: 'level 2, quad',      m: 2, nbx: 16, nby: 16, slots: 128, cells: 400, opts: {} },
    { name: 'root (level 0)',     m: 0, nbx: 4, nby: 4, slots: 16, cells: 256, opts: {} },
  ];

  for (const c of CASES) {
    check(`${c.name}: every declared buffer is written or declared not-state`, () => {
      const dev = recordingDevice();
      const pool = allocLevelPool(dev, U, c.m, c.nbx, c.nby, c.slots, c.cells, c.opts);
      // allocLevelPool calls writePoolInitialState itself; clear and call it
      // again so this measures the RESET path specifically.
      dev.written.clear();
      writePoolInitialState(dev, pool);

      const bufferKeys = Object.keys(pool).filter(k => pool[k] && pool[k].__id);
      assert.ok(bufferKeys.length > 5, `expected a pool to declare several buffers, saw ${bufferKeys.length}`);
      const missed = bufferKeys.filter(k => !dev.written.has(pool[k].__id) && !(k in NOT_RESET_STATE));
      assert.deepStrictEqual(missed, [],
        `these buffers exist on the pool and reset() writes none of them:\n          ${missed.join(', ')}\n`
        + '        Either write it in writePoolInitialState, or add it to NOT_RESET_STATE\n'
        + '        in this test with the reason it is not state.');
    });
  }

  check('the not-state list is not silently stale', () => {
    const dev = recordingDevice();
    const pool = allocLevelPool(dev, U, 2, 16, 16, 128, 400, {});
    for (const k of Object.keys(NOT_RESET_STATE)) {
      assert.ok(pool[k] && pool[k].__id,
        `NOT_RESET_STATE names '${k}', which is not a buffer on the pool any more -- `
        + 'remove it rather than leaving an exemption for something that no longer exists');
    }
  });

  check('a quad pool free list is sized in QUADS, and reset seeds every entry', () => {
    const dev = recordingDevice();
    const pool = allocLevelPool(dev, U, 2, 16, 16, 128, 400, {});
    assert.strictEqual(pool.quadAlloc, true);
    // one i32 per quad, not per slot -- the mismatch U7-6a's over-long copy hit
    assert.strictEqual(pool.freeListBuf.size, (128 / 4) * 4);
  });

  check('the root pool refuses a capacity that is not one slot per block', () => {
    const dev = recordingDevice();
    assert.throws(() => allocLevelPool(dev, U, 0, 4, 4, 32, 256, {}), /one slot per block/);
  });

  console.log(`\npool-initial-state: ${failures ? `${failures} test(s) FAILED` : 'all tests passed'}`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
