// Runtime WGSL "#include" shim: WGSL has no module/include system, but every
// shader here is fetched as plain text before compiling (see loadShader() in
// main*.js), so a small textual splice at fetch time is enough to share
// fragments (shaders/common_*.wgsl) across kernels instead of hand-copying
// them -- see shaders/common_lattice.wgsl and shaders/common_geometry.wgsl
// for what's actually shared. Directive: a line matching
// `// @include "name.wgsl"` is replaced with that file's own contents.
// Fragments must not themselves @include -- one level only, no cycle
// detection needed. Used identically from a browser (readFile backed by
// fetch) and from Node (tools/assemble-shader.js, backed by fs), which is
// also what lets `make wgsl` validate exactly what the browser compiles.
const INCLUDE_RE = /^\s*\/\/\s*@include\s+"([^"]+)"\s*$/;

export async function assembleShader(entryPath, readFile) {
  const dir = entryPath.slice(0, entryPath.lastIndexOf('/') + 1);
  const src = await readFile(entryPath);
  const lines = src.split('\n');
  const out = [];
  for (const line of lines) {
    const m = line.match(INCLUDE_RE);
    if (m) {
      out.push(await readFile(dir + m[1]));
    } else {
      out.push(line);
    }
  }
  return out.join('\n');
}
