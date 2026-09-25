// LATER WINS, and it has to. A leg appended to a base query as
// `...&levels=2&...&levels=3` is read by URLSearchParams.get as the FIRST
// occurrence, so the leg is silently ignored and the run reports on the BASE
// configuration under the leg's name. That happened in probe-d3-tumble.js: a
// `levels=3` leg came back BIT-IDENTICAL to the `levels=2` run, which is the
// only reason it was caught. This codebase's own rule is that a parameter
// silently DROPPED is worse than one rejected, and a duplicated one is that
// failure wearing a hat.
//
// Shared rather than copied because the failure is invisible: a tool that
// appends without merging produces a plausible run of the wrong thing, and a
// second copy of this reasoning is a second chance to not have it. Every tool
// that composes a page URL out of a base plus caller-supplied fragments should
// go through here.
function mergeQuery(...parts) {
  const q = new URLSearchParams();
  for (const part of parts) {
    if (!part) continue;
    for (const [k, v] of new URLSearchParams(part)) q.set(k, v);
  }
  return q.toString();
}

module.exports = { mergeQuery };
