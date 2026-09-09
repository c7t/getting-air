#!/usr/bin/env node
// Reclaims the debug Chromes the GPU validation tools leave behind, and their
// profile dirs. Run it when you are done validating -- `make chrome-clean`.
//
// WHY THIS IS A SEPARATE, EXPLICIT COMMAND rather than something teardown
// does. tools/lib/browser-lifecycle.js deliberately REUSES a Chrome that is
// already listening on the debug port: that is what keeps exactly one WebGPU
// context alive across a whole run instead of one per config. The cost is that
// a run which adopts a Chrome does not own it and must not kill it -- another
// run may be adopting the same one, and the webgpu-verify skill leaves one
// open on purpose for manual driving. So no individual run can safely decide
// to reap; you can.
//
// What accumulates without it: the first run to be killed part-way (a timeout,
// a Ctrl-C, a crash before teardown) orphans its Chrome, which then keeps a
// GPU context and slows every later measurement -- the skill's own "orphaned
// Chrome competing for the GPU" warning, which has already produced one
// confidently-wrong reading in this project's history. ensureChrome now sweeps
// stale PROFILE DIRS on every launch, so disk no longer grows unbounded, but
// only this command reclaims the processes.
//
// SAFETY: a Chrome is only ever identified by its --user-data-dir living under
// /tmp/vpm-chrome-profile, read from /proc. Never by process name, so the
// user's own browser cannot be matched. --dry-run prints what it would do.

const fs = require('fs');
const { PROFILE_ROOT, liveProfileDirs, sweepStaleProfiles, reapAllChromes } = require('./lib/browser-lifecycle');

const dryRun = process.argv.includes('--dry-run');

function du(dir) {
  let total = 0;
  const walk = (p) => {
    let st;
    try { st = fs.lstatSync(p); } catch { return; }
    if (st.isDirectory()) {
      let kids = [];
      try { kids = fs.readdirSync(p); } catch { return; }
      for (const k of kids) walk(`${p}/${k}`);
    } else total += st.size;
  };
  walk(dir);
  return total;
}

const live = liveProfileDirs();
let dirs = [];
try { dirs = fs.readdirSync(PROFILE_ROOT).map(d => `${PROFILE_ROOT}/${d}`); } catch { /* nothing there */ }

console.log(`profile root: ${PROFILE_ROOT}`);
console.log(`  profile dirs:        ${dirs.length}`);
console.log(`  with a live Chrome:  ${live.size}`);
console.log(`  disk:                ${(du(PROFILE_ROOT) / 1e6).toFixed(0)} MB`);

if (dryRun) {
  for (const d of dirs) console.log(`  ${live.has(d) ? 'LIVE ' : 'stale'}  ${d}`);
  console.log('\n--dry-run: nothing killed or deleted.');
  process.exit(0);
}

const { killed } = reapAllChromes();
console.log(`\nsent SIGTERM to ${killed} Chrome process(es) using that root`);

// Give them a moment to exit so their dirs count as stale, then sweep.
setTimeout(() => {
  const removed = sweepStaleProfiles();
  console.log(`removed ${removed} profile dir(s)`);
  const stillLive = liveProfileDirs().size;
  if (stillLive) {
    console.log(`${stillLive} Chrome(s) still running -- their dirs were kept; rerun if they exit.`);
  } else {
    try { fs.rmdirSync(PROFILE_ROOT); } catch { /* not empty or not there */ }
  }
  console.log('done.');
}, 1500);
