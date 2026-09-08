// Host-side counterpart to shaders/common_fpack.wgsl: converts the `f`
// distribution function between the f32 plane-major layout everything
// OUTSIDE the GPU speaks and the packed-halves layout the shaders use when
// F16 is on.
//
// WHY A CONVERSION AND NOT A SECOND FORMAT. Snapshots, the dense->AMR
// injector (tools/lib/dense-to-amr.js), the field reconstructor
// (tools/lib/field-reconstruct.js) and every validation tool read and write
// `f` as a Float32Array of 9 planes. Making all of them f16-aware would
// double the surface where the two layouts could drift apart -- the exact
// failure mode this repo has already been bitten by (commit 238e48c). So the
// wire format stays f32 plane-major unconditionally, and packing is a
// property of the GPU buffer alone: pack on the way in, unpack on the way
// out. Nothing downstream changes.
//
// With f16 off both functions are the identity (no copy, no arithmetic), so
// the default path costs nothing.
//
// Same browser+Node dual-consumption pattern as shader-loader.mjs and
// card-params.mjs: imported natively by the pages, dynamically imported by
// the CommonJS tools.

const _buf = new ArrayBuffer(4);
const _f32 = new Float32Array(_buf);
const _u32 = new Uint32Array(_buf);

// f32 -> IEEE binary16 bits, round-to-nearest-even, with subnormal and
// overflow handling. Written out rather than using Math.f16round because
// that is very recent (and this must also run under whatever Node the tools
// happen to have); the point is to match what pack2x16float does on the GPU,
// so a snapshot round-trip is exact rather than approximately right.
//
// TAKES f32 SEMANTICS, deliberately, and that is not the same as
// Math.f16round. This rounds through single precision (the input is stored
// into a Float32Array to get at its bits), so a JS double that sits exactly
// on an f16 tie ONCE NARROWED to f32 rounds ties-to-even from there, whereas
// Math.f16round rounds the full double in one step and can land one ulp
// away. Example: 0.40246582615253534 -> 0.40234375 here, 0.402587890625 from
// Math.f16round. f32 is the right semantics because every real caller passes
// an element of a Float32Array -- already an f32 -- and the value has to
// match what the GPU's pack2x16float produces from the same f32. Feeding a
// raw double here means double rounding; narrow it with Math.fround first if
// that matters to the caller.
export function f32ToF16Bits(v) {
  _f32[0] = v;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let man = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0); // Inf / NaN
  const e = exp - 112;                                        // -127 + 15
  if (e >= 0x1f) return sign | 0x7c00;                        // overflow -> Inf
  if (e <= 0) {
    if (e < -10) return sign;                                 // underflow -> +-0
    man |= 0x800000;                                          // implicit leading 1
    const shift = 14 - e;                                     // in [14, 24]
    let h = man >>> shift;
    const rem = man & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h += 1;      // ties to even
    return sign | h;
  }
  let h = (e << 10) | (man >>> 13);
  const rem = man & 0x1fff;
  // A carry out of the mantissa lands in the exponent field, which is the
  // correct result, so no special case is needed here.
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h += 1;
  return sign | h;
}

export function f16BitsToF32(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 5.9604644775390625e-8;          // 2^-24
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}

// Number of u32 words per cell for a given mode. Mirrors fWords() in
// common_fpack.wgsl.
export function fWords(f16) { return f16 ? 5 : 9; }

// D2Q9 lattice weights, duplicated from shaders/common_lattice.wgsl because
// this module is deliberately dependency-free (it is imported by ten pages
// and by the CommonJS tools). Only mode 2 uses them; the constants are fixed
// by the lattice, not by anything tunable.
const WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

// f32 plane-major (9 * ncells) -> what the GPU buffer wants.
// f16 off: the same array, untouched (the shader bitcasts it back).
// f16 on:  a Uint32Array of 5 * ncells words, planes (2p, 2p+1) paired.
export function packF(f, ncells, f16) {
  if (!f16) return f;
  const dev = f16 === 2;                       // mode 2 stores f_i - w_i
  const w = new Uint32Array(5 * ncells);
  for (let p = 0; p < 5; p++) {
    const lo = 2 * p, hi = Math.min(2 * p + 1, 8);
    const loBase = lo * ncells, hiBase = hi * ncells, wBase = p * ncells;
    const wlo = dev ? WT[lo] : 0, whi = dev ? WT[hi] : 0;
    for (let c = 0; c < ncells; c++) {
      w[wBase + c] = ((f32ToF16Bits(f[hiBase + c] - whi) << 16) | f32ToF16Bits(f[loBase + c] - wlo)) >>> 0;
    }
  }
  return w;
}

// Inverse of packF. `src` is whatever was read back from the GPU buffer:
// a Float32Array when f16 is off (returned as-is), or a Uint32Array of
// packed words when it is on.
export function unpackF(src, ncells, f16) {
  if (!f16) return src;
  const off = f16 === 2 ? 1 : 0;
  const f = new Float32Array(9 * ncells);
  for (let i = 0; i < 9; i++) {
    const p = i >> 1, high = (i & 1) === 1, w0 = off ? WT[i] : 0;
    const wBase = p * ncells, fBase = i * ncells;
    for (let c = 0; c < ncells; c++) {
      const w = src[wBase + c];
      f[fBase + c] = f16BitsToF32(high ? (w >>> 16) : (w & 0xffff)) + w0;
    }
  }
  return f;
}
