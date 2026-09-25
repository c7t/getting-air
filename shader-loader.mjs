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

// Fetch a WGSL entry file, splice its `@include`s, and compile it.
//
// TEN IDENTICAL COPIES before plans/2D-backport.md B3a -- one per main*.js
// page, byte for byte. It belongs here because `assembleShader` above is the
// thing it exists to wrap, and because the cache-buster is a decision
// (`?v=Date.now()`: these pages are served by a dev server with no cache
// headers, and a stale shader after an edit is indistinguishable from a
// shader that does not work).
//
// The HTTP status check is load-bearing: without it a 404 returns the dev
// server's HTML error page, which reaches the WGSL front end as a syntax
// error pointing at `<!DOCTYPE`, several steps removed from "that path is
// wrong".
export async function loadShader(device, path) {
  const code = await assembleShader(path, async (p) => {
    const r = await fetch(p + '?v=' + Date.now());
    if (!r.ok) throw new Error(`failed to load ${p} (HTTP ${r.status} ${r.statusText})`);
    return r.text();
  });
  return device.createShaderModule({ code });
}
