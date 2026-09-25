#!/usr/bin/env node
// Writes shaders/common_lattice.wgsl from lattice-2d.mjs. There is no build
// step in this project (the source IS the artifact), so the generated file is
// CHECKED IN and this is a regenerate-on-demand tool, not part of any
// pipeline. Drift between the generator and the checked-in file is a
// `make test` failure -- see tools/test-lattice-2d.js.
//
// Run: node tools/gen-lattice-2d.js

const fs = require('fs');
const path = require('path');

(async () => {
  const root = path.join(__dirname, '..');
  const { latticeWGSL } = await import(path.join(root, 'lattice-2d.mjs'));
  const dest = path.join(root, 'shaders', 'common_lattice.wgsl');
  fs.writeFileSync(dest, latticeWGSL());
  console.log(`wrote ${path.relative(root, dest)}`);
})();
