// Derives an apples-to-apples (dense-target-resolution, AMR-base-resolution)
// pairing for tools/validate-amr-vs-dense.js, and builds the two pages' URLs
// from ONE shared param set so they cannot silently drift apart.
//
// The scaling law: main-cylinder.js and main-cylinder-amr.js both derive the
// cylinder radius from their own grid width and a BLOCKAGE ratio
// (R = W/(2*BLOCKAGE)), so a resolution change alone keeps the domain's
// physical proportions (blockage, upstream fetch, all in diameters) fixed --
// see main-cylinder.js's own comment above its BLOCKAGE/UPSTREAM/R block.
// AMR's ?levels=N doubles resolution per level (cellSizeL0AtLevel(m) =
// 2^-m, main-cylinder-amr.js:577-579). So matching AMR's FINEST level to a
// dense reference at targetResLog2 means:
//
//   baseResLog2 = targetResLog2 - (nLevels - 1)
//
// Three independent constraints gate whether a (targetResLog2, nLevels)
// pair is a legitimate apples-to-apples comparison, not just arithmetic:
//
//  1. baseResLog2 must land in [7,11] -- the resLog2 clamp both
//     main-cylinder.js and main-cylinder-amr.js already enforce
//     (silently, by re-clamping) at load time.
//  2. nLevels itself must be within what main-cylinder-amr.js's own AMR
//     mechanism currently validates -- see that file's comment block above
//     `const N_LEVELS` (a living document per CLAUDE.md, re-check it if
//     this cap ever looks stale). As of this writing: nLevels<=3 is
//     validated (2:1 balance, geometry coverage, both coupling modes);
//     nLevels>3 is untested, and bounce-back specifically hard-blocks it
//     without ?forceBounceback.
//  3. The BASE level's own relaxation time must have enough margin above
//     the BGK stability floor (tau=0.5) -- see computeBaseTau below. This
//     is a genuinely separate axis from #2: every existing validated AMR
//     config in this project defaults to ?res=9 (W_base=512) regardless of
//     level count, so "nLevels<=3 is validated" has never been exercised
//     at the SMALLER base grids a fixed high target resolution forces you
//     into (e.g. target=1024, nLevels=3 -> base=256, a base grid nothing
//     in this project's history has run at 3 levels before).

const RES_LOG2_MIN = 7, RES_LOG2_MAX = 11; // matches both pages' own clamp
const VALIDATED_MAX_LEVELS = 3; // main-cylinder-amr.js's own N_LEVELS comment

// tau_base = 0.5 + 3*U0*D_base/Re, D_base = W_base/BLOCKAGE (main-cylinder.js's
// tauFromRe, reproduced here in closed form so it can be evaluated BEFORE
// ever opening a page -- this is a pure pre-flight check, not a GPU
// readback). Every finer level's own tau is DERIVED from this one via the
// Dupuis-Chopard relation tau_fine = 2*tau_coarse - 0.5
// (main-cylinder-amr.js:337-349), which moves tau AWAY from 0.5 at each
// finer level -- so the base/coarsest level is always the tightest
// stability constraint in the whole hierarchy, never the finest.
function computeBaseTau({ W_base, blockage, u0, re }) {
  const D = W_base / blockage;
  const nu = u0 * D / re;
  return 0.5 + 3 * nu;
}

// Below this, expect visible numerical noise (not necessarily blowup, BGK
// is nominally stable down to tau>0.5) -- an unvalidated-in-this-project
// regime, worth a loud warning rather than a silent go-ahead.
const TAU_WARN_BELOW = 0.52;
// Below this, close enough to the tau=0.5 floor that a field-relL2 FAIL
// from this tool would be as likely to reflect base-grid noise as an AMR
// bug -- require an explicit opt-in rather than silently running it.
const TAU_BLOCK_BELOW = 0.505;

// Throws with a specific, actionable message naming the exact violated
// bound -- these errors are meant to be read by whoever typed the --res/
// --levels flags, not just logged.
function deriveAMRParams({ targetResLog2, nLevels, blockage = 24, u0 = 0.04, re = 100, allowUnvalidatedLevels = false, allowMarginalTau = false }) {
  if (!Number.isInteger(nLevels) || nLevels < 2) {
    throw new Error(`deriveAMRParams: nLevels=${nLevels} invalid -- must be an integer >= 2 (L0 + at least one fine level)`);
  }
  if (nLevels > VALIDATED_MAX_LEVELS && !allowUnvalidatedLevels) {
    throw new Error(`deriveAMRParams: nLevels=${nLevels} exceeds this project's currently-validated cap of ${VALIDATED_MAX_LEVELS} ` +
      `(see main-cylinder-amr.js's own comment above N_LEVELS -- re-read it, this cap drifts with active work per CLAUDE.md). ` +
      `Pass allowUnvalidatedLevels:true / --allowUnvalidatedLevels to proceed anyway (this will also forward &forceBounceback if --bounceback is set).`);
  }
  const baseResLog2 = targetResLog2 - (nLevels - 1);
  if (baseResLog2 < RES_LOG2_MIN || baseResLog2 > RES_LOG2_MAX) {
    throw new Error(`deriveAMRParams: targetResLog2=${targetResLog2} with nLevels=${nLevels} needs baseResLog2=${baseResLog2}, ` +
      `outside the [${RES_LOG2_MIN},${RES_LOG2_MAX}] clamp both main-cylinder.js and main-cylinder-amr.js enforce -- ` +
      `either lower nLevels, lower targetResLog2, or raise it (whichever moves baseResLog2 back into range).`);
  }
  const W_target = 1 << targetResLog2;
  const W_base = 1 << baseResLog2;

  const tauBase = computeBaseTau({ W_base, blockage, u0, re });
  if (tauBase < TAU_BLOCK_BELOW && !allowMarginalTau) {
    throw new Error(`deriveAMRParams: base-level tau=${tauBase.toFixed(4)} (W_base=${W_base}, Re=${re}, blockage=${blockage}, u0=${u0}) is within ` +
      `${(tauBase - 0.5).toFixed(4)} of the BGK stability floor (tau=0.5) -- every finer AMR level derives its own tau FROM this one via ` +
      `tau_fine=2*tau_coarse-0.5 (main-cylinder-amr.js:337-349), so the base level is always the tightest stability constraint. ` +
      `A field-relL2 FAIL at this combination would be as likely to reflect base-grid numerical noise as a real AMR bug. ` +
      `Pass allowMarginalTau:true / --allowMarginalTau to proceed anyway, or reduce nLevels / raise targetResLog2 to grow W_base.`);
  }
  const tauWarning = tauBase < TAU_WARN_BELOW
    ? `base-level tau=${tauBase.toFixed(4)} (W_base=${W_base}, Re=${re}) is below ${TAU_WARN_BELOW} -- unvalidated-in-this-project numerical-noise ` +
      `territory (see tools/lib/amr-resolution-mapping.js's own header). Not blocking, but treat a FAIL here with extra scrutiny.`
    : null;

  return { baseResLog2, targetResLog2, nLevels, W_target, W_base, tauBase, tauWarning };
}

// One query-string fragment consumed by BOTH URL builders below -- a future
// param addition to one can't silently diverge from the other because
// there's only one place params are turned into a string. `bounceback` is
// NOT AMR-specific: main-cylinder.js has the identical ?bounceback flag
// (see its own header, ~line 26-30) since it's a coupling-method choice,
// not an AMR concept -- forwarded to both pages identically.
function deriveSharedURLParams({ blockage = 24, upstream = 8, re = 100, u0 = 0.04, seed = 12345, perturb = 0, bounceback = false }) {
  const params = new URLSearchParams();
  params.set('blockage', String(blockage));
  params.set('upstream', String(upstream));
  params.set('re', String(re));
  params.set('u0', String(u0));
  params.set('seed', String(seed));
  params.set('perturb', String(perturb));
  if (bounceback) params.set('bounceback', '');
  return params;
}

function buildDenseUrl(baseUrl, { targetResLog2, sharedParams }) {
  const params = new URLSearchParams(sharedParams);
  params.set('res', String(targetResLog2));
  return `${baseUrl}/index-cylinder.html?${params.toString()}`;
}

function buildAMRUrl(baseUrl, { baseResLog2, nLevels, sharedParams, maxFineBlocksByLevel = {}, forceBounceback = false }) {
  const params = new URLSearchParams(sharedParams);
  params.set('res', String(baseResLog2));
  params.set('levels', String(nLevels));
  for (const [level, count] of Object.entries(maxFineBlocksByLevel)) {
    // level 1's cap is ?maxFineBlocks=, levels >=2 are ?maxFineBlocks2=,
    // ?maxFineBlocks3=, ... (main-cylinder-amr.js:70, :708-709).
    const key = Number(level) === 1 ? 'maxFineBlocks' : `maxFineBlocks${level}`;
    params.set(key, String(count));
  }
  if (forceBounceback) params.set('forceBounceback', '');
  return `${baseUrl}/index-cylinder-amr.html?${params.toString()}`;
}

module.exports = {
  RES_LOG2_MIN, RES_LOG2_MAX, VALIDATED_MAX_LEVELS, TAU_WARN_BELOW, TAU_BLOCK_BELOW,
  computeBaseTau, deriveAMRParams, deriveSharedURLParams, buildDenseUrl, buildAMRUrl,
};
