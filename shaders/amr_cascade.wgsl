// 2:1 BALANCE AS ONE CLOSURE ON THE WANT SET -- plans/2D-backport.md B2.
//
// THE RULE, in full:
//
//     want(m, b)  =>  want(m-1, parent(n))   for every neighbour n of b at
//                     level m, and for b itself (offset (0,0))
//
// Read forwards it is "refine forced by a neighbour that wants a deeper
// child"; read BACKWARDS -- a block the criterion did not want, added back --
// it is "coarsen blocked by a neighbour that has one". THEY ARE NOT TWO
// MECHANISMS. That is the whole reason this is one function applied to the
// WANT set between the criterion and the allocator, rather than tests bolted
// onto coarsen and refine.
//
// WHAT THE PER-PASS SHAPE COST. shaders/amr_manage_pool.wgsl's header
// documents three separately live-verified bugs it produced -- a
// neighbour-active gate vetoing geometry-forced refinement, a grandchild
// (level m+2) blind spot that made real depth-1-next-to-depth-3 violations at
// N_LEVELS=4, and a criterion-based ("wants") cascade test that had to become
// an existence ("has") test because criterion flickers -- and a fourth is
// still shipped: only the VETO half of the refine cascade was ever
// implemented, so a criterion-driven refine can be blocked forever by a
// neighbour that would only ever have been created BY that refine. Measured
// (B2-1): level 2's x-extent on index-amr.html is 80 L0 units against 176
// with ?demandCascade=1, i.e. the L1/L2 boundary is pinned a few cells off
// the body and every shed vortex crosses it there.
//
// The closure has none of those failure modes by construction: it is
// transitive (no grandchild special case), it runs on WANTS rather than on
// what exists (no flicker), and the growth half IS the veto half.
//
// ALL 9 OFFSETS, NOT THE FOUR FACES. d = (0,0) is the TREE PROPERTY (a block
// with no parent is not a refinement of anything), the four faces are 2:1
// balance, and the four DIAGONALS are the RING -- amr_step1.wgsl reads the
// corner cell whenever the diagonal same-level neighbour is absent, and that
// read has to land in an existing parent tile. B2-1 measured the shipped
// topology's violations as 100% diagonal, so the faces are already right and
// this is the half that is not.//
// AND d = (0,0) IS REDUNDANT -- MEASURED, NOT ASSUMED. The three-jobs reading
// above ("tree property, faces, ring") is how this rule is usually written
// down, and it overstates the case: the own-parent offset is IMPLIED by the
// four faces. A quad's members are adjacent and share a parent, so for any
// block at least one x-face neighbour has the same parent -- bx even gives
// (bx+1)>>1 == bx>>1, bx odd gives (bx-1)>>1 == bx>>1, and NBX is a power of
// two so the wrap has no edge case. Dropping (0,0) therefore changes nothing,
// and it is the one mutant of this closure that breaks NOTHING: 0 of 42 host
// checks and 0 of 10 GPU seeds, against 5-9 failures for every other mutant
// tried (face-only, no quad completion, non-periodic wrap, one-axis parent
// shift, transposed parent, quad completion rounding up).
//
// KEEP IT ANYWAY. It costs one ninth of a cheap pass, it makes the rule
// self-evident rather than a consequence, and the redundancy depends on the
// face offsets being present -- so a future change that narrowed the offset
// set would silently take the tree property with it. What is NOT ok is
// deleting it as an optimization on the strength of a green suite, which is
// exactly what a green suite would support here.
//
// ONE SWEEP, DEEPEST FIRST -- NO FIXED-POINT LOOP. A want at level m forces
// wants at level m-1 only; the propagation is strictly one-directional down
// the levels, so visiting levels from finest to 2 reaches the fixed point in
// one pass. tools/test-amr2d.js asserts exactly that about the host twin
// ("cascade21 is IDEMPOTENT -- one deepest-first sweep IS the fixed point"),
// which is what lets main-amr.js's FIXED_POINT_ITERS loop go away.
//
// THE HOST TWIN IS THE SPECIFICATION. amr2d.mjs's `cascade21` is the same
// rule, GPU-free and mutation-scored, and amr2d-gpu.mjs drives this shader
// against it on seeded want sets -- including sets that VIOLATE the invariant,
// which is the only way to know a closure closes anything.
//
// WHY THE WRITES DO NOT RACE, since both entry points are read-modify-write
// over a shared array and neither takes a lock:
//
//   MONOTONE. Every write is `= 1u`. No thread ever clears a want, so two
//   threads writing the same slot write the same value, and a read that
//   misses another thread's write can only UNDER-report -- never produce a
//   value nobody wrote.
//   completeQuads reads and writes ONE level. A thread whose own block is
//   wanted sets all four siblings, so the quad is completed by that thread
//   alone; a thread whose block is not wanted does nothing and cannot undo
//   it. The result is independent of ordering.
//   balance reads level m and writes level m-1 -- DIFFERENT arrays, so there
//   is no intra-pass hazard at all, and the dispatch boundary between the two
//   entry points is the only ordering this needs.

// NO INCLUDES. This is pure index arithmetic on the block grids -- no SDF,
// no lattice, no populations. That it needs none of that is the point: the
// 2:1 rule is a statement about the TREE, and the shipped version entangled
// it with criterion values and geometry tests inside coarsen/refine.

// This level's own want array, and its parent level's.
@group(0) @binding(0) var<storage, read_write> want       : array<u32>;
@group(0) @binding(1) var<storage, read_write> wantParent : array<u32>;

// THIS level's block grid. The parent's is exactly half on each axis -- a
// level-m tile is one QUADRANT of its parent's footprint at doubled density
// (amr2d.mjs's poolAtLevel), so there is no second pair of dimensions to get
// out of step with these.
override NBX : u32;
override NBY : u32;

// Level 1 is the exception and not an inconsistency: its parent is the DENSE
// L0 grid, present everywhere, so level-1 blocks are refined individually and
// there is no quad to complete. Levels >= 2 exist only as a quad spawned by
// one parent tile -- there is no such thing as a quarter-refined parent, and
// amr2d.mjs's check21Balance leans on that too. 3D's first cascade got this
// wrong and its checker was right.
override QUAD_COMPLETE : u32 = 1u;

// A want for one child is a want for the parent's refinement, so the input is
// COMPLETED to quads rather than rejected. That says what the want MEANS
// instead of discarding information the criterion genuinely produced -- the
// criterion is evaluated per block and nothing in it looks sideways, so a
// want for one child with no siblings is a state it really can reach.
@compute @workgroup_size(64)
fn completeQuads(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= NBX * NBY) { return; }
  if (QUAD_COMPLETE == 0u) { return; }
  if (want[b] == 0u) { return; }

  let bx = b % NBX;
  let by = b / NBX;
  let qx0 = (bx & ~1u);
  let qy0 = (by & ~1u);
  for (var q = 0u; q < 4u; q++) {
    let cx = qx0 + (q & 1u);
    let cy = qy0 + ((q >> 1u) & 1u);
    want[cy * NBX + cx] = 1u;
  }
}

// THE CLOSURE ITSELF. For every wanted block at this level, its own parent
// and the parents of all 8 neighbours must be wanted one level up.
//
// THE BLOCK GRID IS PERIODIC, and the wrap is not decoration: a quad at the
// origin forces parents across the seam. tools/test-amr2d.js has a fixture at
// the seam for exactly this, added after a mutant that dropped the wrap broke
// nothing -- every earlier fixture sat in the middle of the grid.
@compute @workgroup_size(64)
fn balance(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= NBX * NBY) { return; }
  if (want[b] == 0u) { return; }

  let bx = b % NBX;
  let by = b / NBX;
  let nbxP = NBX / 2u;

  for (var k = 0u; k < 9u; k++) {
    let dx = k % 3u;          // 0,1,2 -> -1,0,+1
    let dy = k / 3u;
    let nx = (bx + NBX + dx - 1u) % NBX;
    let ny = (by + NBY + dy - 1u) % NBY;
    // parent(n) -- one shift per axis, which is amr2d.mjs's parentOfBlock.
    wantParent[(ny >> 1u) * nbxP + (nx >> 1u)] = 1u;
  }
}
