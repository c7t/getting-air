#!/usr/bin/env node
// Thin CLI used by `make wgsl` to validate an assembled (fragments-inlined)
// shader the same way the browser will compile it -- see shader-loader.mjs
// (the shared assembly logic, an ES module dynamically imported here since
// this file itself is a plain CommonJS script) and the Makefile's `wgsl`
// target.
//
// Usage: node tools/assemble-shader.js shaders/lbm_step.wgsl > /tmp/out.wgsl

const fs = require('fs');
const path = require('path');

async function main() {
  const entryPath = process.argv[2];
  if (!entryPath) {
    console.error('usage: assemble-shader.js <path/to/shader.wgsl>');
    process.exit(1);
  }
  const { assembleShader } = await import(path.join(__dirname, '..', 'shader-loader.mjs'));
  const readFile = (p) => fs.promises.readFile(p, 'utf8');
  const code = await assembleShader(entryPath, readFile);
  process.stdout.write(code);
}

main().catch(e => { console.error(e); process.exit(1); });
