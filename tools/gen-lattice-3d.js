#!/usr/bin/env node
// Writes shaders/common_d3q19_lattice.wgsl and common_d3q27_lattice.wgsl
// from lattice-3d.mjs. There is no build step in this project (the source
// IS the artifact), so the generated files are CHECKED IN and this is a
// regenerate-on-demand tool, not part of any pipeline. Drift between the
// generator and the checked-in files is a `make test` failure -- see
// tools/test-lattice-3d.js.
//
// Run: node tools/gen-lattice-3d.js

const fs = require('fs');
const path = require('path');

(async () => {
  const root = path.join(__dirname, '..');
  const { SUPPORTED_Q, latticeWGSL } = await import(path.join(root, 'lattice-3d.mjs'));
  for (const Q of SUPPORTED_Q) {
    const dest = path.join(root, 'shaders', `common_d3q${Q}_lattice.wgsl`);
    fs.writeFileSync(dest, latticeWGSL(Q));
    console.log(`wrote ${path.relative(root, dest)}`);
  }
})();
