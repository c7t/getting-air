#!/usr/bin/env node
// Numerically diffs two AMR-dev-build snapshots (see main-amr.js's
// debugSnapshotSave / tools/amr-snapshot.js) instead of comparing
// screenshots -- this is the regression check plans/AMR.md's milestones
// rely on ("does the AMR build's output match a reference to tolerance").
// Pure Node, no browser/CDP needed once the snapshots are on disk.
//
// Milestone 0/1 scope: both snapshots must be the same W x H (e.g. two
// snapshots of the same run, or main-amr.js vs. a from-scratch reference at
// identical resolution). Comparing a fine sub-region against a coarser
// uniform-grid reference (needed once resolutions actually differ) is
// handled by tools/lib/field-reconstruct.js's reconstructAMRToResolution,
// consumed by tools/validate-amr-vs-dense.js -- not by this CLI, which stays
// same-resolution-only.
//
// The actual field-decoding primitives (b64 decode, window-shift unrotate,
// rho-from-f, vorticity, relL2 diff stats) live in
// tools/lib/field-reconstruct.js, shared with tools/validate-amr-vs-dense.js
// so there's one decoder, not two independently-drifting copies.
//
// Usage: node tools/amr-diff.js <a.json> <b.json>

const fs = require('fs');
const { diffStats, loadDenseFields } = require('./lib/field-reconstruct');

function main() {
  const [aFile, bFile] = process.argv.slice(2);
  if (!aFile || !bFile) {
    console.error('usage: amr-diff.js <a.json> <b.json>');
    process.exit(1);
  }
  const a = JSON.parse(fs.readFileSync(aFile, 'utf8'));
  const b = JSON.parse(fs.readFileSync(bFile, 'utf8'));
  if (a.W !== b.W || a.H !== b.H) {
    console.error(`resolution mismatch: ${aFile} is ${a.W}x${a.H}, ${bFile} is ${b.W}x${b.H} -- not yet supported (see file header)`);
    process.exit(1);
  }
  if (a.step !== b.step) {
    console.warn(`WARNING: comparing different step counts (${a.step} vs ${b.step}) -- diffs below reflect that, not just numerical divergence`);
  }

  const fa = loadDenseFields(a), fb = loadDenseFields(b);
  console.log(`Comparing ${aFile} (step ${fa.step}, layout ${fa.layout}) vs ${bFile} (step ${fb.step}, layout ${fb.layout}), ${fa.W}x${fa.H}`);
  const results = {
    ux: diffStats(fa.ux, fb.ux, 'ux   '),
    uy: diffStats(fa.uy, fb.uy, 'uy   '),
    rho: diffStats(fa.rho, fb.rho, 'rho  '),
    omega: diffStats(fa.omega, fb.omega, 'omega'),
  };
  const allZero = Object.values(results).every(r => r.maxAbsDiff === 0);
  console.log(allZero ? 'IDENTICAL' : 'DIFFERS');
  process.exit(0);
}
main();
