// Packed storage for the distribution function `f`.
// Included via `// @include "common_fpack.wgsl"`.
//
// WHY IT EXISTS, and why it is DEFAULT OFF.
//
// plans/perf-characterization.md measures the phone (PowerVR) as BANDWIDTH
// bound with `f` -- 9 x f32 = 36 B/cell -- as essentially all of the traffic,
// so halving its width is worth ~4x more there than any pass-level change.
// That made it worth implementing. It does not survive the physics.
//
// Measured with this code, against the analytic benchmarks: mode 1 gives
// L2rel 0.9995 on channel Poiseuille (the flow does not develop at all --
// the per-step body-force increment is below an ulp of f_i) and 1.14 on
// TGV N=128. Mode 2 is 10-40x better and rescues the coarser TGV cases, but
// still misses every channel tolerance by 5-20x. In BOTH modes the error
// grows with resolution, because a finer grid spreads the same physical
// gradient over more cells and shrinks the per-cell fneq -- the whole
// information content beyond equilibrium -- toward the quantum. Refining is
// what this project does, so the format degrades exactly where the work goes.
//
// It is kept, at F16=0, because the layout is proven correct (the failure
// signature is resolution-dependent precision loss, not the direction
// transposition a layout bug gives, and tools/test-f-pack.js asserts
// host/shader addressing agreement plane by plane) and because it is one URL
// parameter from being re-measured if the format or the device changes.
//
// A NOTE ON HOW THE OLD ANSWER WAS WRONG, since it cost a session: the
// accuracy question was previously "answered" by ?quantF16=, which rounded
// each stored value through unpack2x16float(pack2x16float(x)) at f32 width.
// The driver folds that round trip away, so it measured f32 and reported no
// difference. Its canary (round(x*256)/256) DID move Cd, which was read as
// proof the override was live -- but the canary is arithmetic the compiler
// cannot fold and the modes under test were not. A control has to be the same
// KIND of operation as the thing it controls for. Note also that Cd/St PASS
// under mode 2 on both AMR cylinder configs; bulk surface integrals are
// dominated by the near-body region where fneq is largest, and they average
// the far-field noise away instead of reporting it. The analytic field checks
// are the gate for a precision change.
//
// No `shader-f16` feature needed: pack2x16float/unpack2x16float are core
// WGSL builtins. The desktop adapter here does not expose `shader-f16` at
// all, so a f16-typed-storage approach would not even compile there.
//
// HOW, and why the f32 path goes through here too. Every `f` buffer is now
// declared `array<u32>` unconditionally, and F16 selects the interpretation:
//
//   F16 == 0  one bitcast f32 per u32. BYTE-IDENTICAL to the old
//             array<f32> declaration, which is the point -- a page that
//             never sets F16 is unaffected, JS still writes its initial
//             equilibrium as a plain Float32Array, snapshots still read back
//             as Float32Array, and `arrayLength(&f)/9u` still yields the
//             per-plane cell stride. One code path, not two.
//   F16 == 1  planes (2p, 2p+1) share u32 plane p, so 9 planes become 5.
//   F16 == 2  the same packing, but storing the DEVIATION f_i - w_i.
//
// WHY MODE 2 EXISTS, measured. Mode 1 spends its 11 mantissa bits on f_i
// itself, and f_i sits close to its lattice weight w_i: at U0=0.04 the
// deviation carrying all the flow information is ~12% of the value, so an
// ulp of f_i is ~1/8 of an ulp's worth of signal. Storing f_i - w_i moves
// the exponent down to the signal's own magnitude and recovers that factor
// (~8x for the rest/axis links). Mode 1 measured Cd 2.282 / St 0.3080 on
// the dense Re=100 reference against f32's 1.950 / 0.1258 -- not a subtle
// shift -- which is the precision floor the fneq table in
// plans/perf-characterization.md predicted for the outer wake and far field
// and which no bulk Cd/St emulation had actually tested.
//
// PAIRING. Consecutive indices, which is not arbitrary. Streaming reads
// plane i from the cell at -e_i, so a word is only fetched once for both of
// its halves to the extent the two planes' source cells fall in the same
// cache lines. The worst pairing would be (i, opp[i]) -- offsets +e and -e,
// maximally separated. With this lattice's ordering (0 rest, 1..4 axes,
// 5..8 diagonals) consecutive pairs are (0,1) (2,3) (4,5) (6,7) and none of
// them is an opposite pair; the largest separation is (1,2) cells, for pair
// (4,5). A whole level-1 tile is FB*FB*5*4 = 8 KB, so the shifted re-reads
// stay resident and the traffic saving tracks the footprint saving.
//
// WHAT IS AND IS NOT SAVED. The footprint (and so the write traffic, and the
// read traffic to the extent above) goes 36 -> 20 B/cell, 1.80x -- 5 words,
// not the ideal 4.5, because 9 halves is an odd count and word 4 carries
// plane 8 with its high half unused. Buffer ALLOCATION is deliberately left
// at 9 words/cell for now, so this is a bandwidth change and not yet a
// footprint change; that keeps every arrayLength-derived stride and every
// page's buffer sizing untouched.
//
// EVERY WRITER MUST WRITE A WHOLE CELL. Packing two planes into one word
// means a partial-plane store would be a read-modify-write race. All the
// current writers already loop i = 0..8 over a cell they own exclusively;
// a future kernel that wants to touch a single plane cannot, and would need
// to fall back to the f32 layout.

override F16 : u32 = 0u;

// Number of u32 words per cell actually in use.
fn fWords() -> u32 { return select(9u, 5u, F16 != 0u); }

// Word index holding plane `i` of cell `c`, given a per-plane cell stride S.
fn fIdx(i: u32, S: u32, c: u32) -> u32 {
  return select(i, i >> 1u, F16 != 0u) * S + c;
}

fn fUnpack(w: u32, i: u32) -> f32 {
  if (F16 == 0u) { return bitcast<f32>(w); }
  let h = unpack2x16float(w);
  let v = select(h.x, h.y, (i & 1u) == 1u);
  return select(v, v + wt[i], F16 == 2u);
}

// Low/high plane indices carried by word p. fHi is clamped rather than
// branched: at p=4 the true high plane is 9, out of range, so word 4 stores
// plane 8 in BOTH halves. The high half is never read back (plane 8 maps to
// word 4's LOW half), so the duplicate is inert padding.
fn fLo(p: u32) -> u32 { return select(p, p << 1u, F16 != 0u); }
fn fHi(p: u32) -> u32 { return min(fLo(p) + 1u, 8u); }

// Takes the WORD index so mode 2 can subtract the right weight from each
// half; fLo/fHi name the two planes the word carries.
fn fPack(lo: f32, hi: f32, p: u32) -> u32 {
  if (F16 == 0u) { return bitcast<u32>(lo); }
  if (F16 == 2u) { return pack2x16float(vec2<f32>(lo - wt[fLo(p)], hi - wt[fHi(p)])); }
  return pack2x16float(vec2<f32>(lo, hi));
}
