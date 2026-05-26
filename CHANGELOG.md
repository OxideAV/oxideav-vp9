# Changelog

All notable changes to `oxideav-vp9` are recorded here.

## [Unreleased]

### Added

* **Round 23: §6.3.9 `read_inter_mode_probs( )` + §6.3.10
  `read_interp_filter_probs( )` compressed-header sweeps.** The two
  nested-loop companions to the round-22 §6.3.11 primitive that
  together front-load the inter-arm dispatch's first 33 cells:
  * `INTER_MODES = 4` / `INTER_MODE_CONTEXTS = 7` constants from §3
    (`vp9-spec.txt` lines 506-507).
  * `DEFAULT_INTER_MODE_PROBS[INTER_MODE_CONTEXTS][INTER_MODES - 1]`
    transcribed verbatim from the §10.5 listing (lines 7758-7766) and
    re-exported from `mode_info` into `compressed` as the single source
    of truth.
  * `read_inter_mode_probs( coder, inter_mode_probs )` per §6.3.9
    (`vp9-spec.txt` lines 2138-2143) — row-major
    `INTER_MODE_CONTEXTS x (INTER_MODES - 1) = 21` cell sweep
    consuming one `B(252)` `update_prob` flag per slot and, on 1, a
    `decode_term_subexp` + `inv_remap_prob` cascade updating
    `inter_mode_probs[ ][ ]` in place. Feeds the §6.4.16
    `inter_block_mode_info( )` per-block decoder via the §9.3.2 ctx
    once reference-buffer state lands.
  * `SWITCHABLE_FILTERS = 3` / `INTERP_FILTER_CONTEXTS = 4` constants
    from §3 (`vp9-spec.txt` lines 487, 495).
  * `DEFAULT_INTERP_FILTER_PROBS[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`
    transcribed verbatim from the §10.5 listing (lines 7769-7775) and
    re-exported from `mode_info` into `compressed`.
  * `read_interp_filter_probs( coder, interp_filter_probs )` per
    §6.3.10 (`vp9-spec.txt` lines 2146-2151) — row-major
    `INTERP_FILTER_CONTEXTS x (SWITCHABLE_FILTERS - 1) = 8` cell
    sweep updating `interp_filter_probs[ ][ ]` in place.
  * 13 new unit tests covering: the §10.5 default-table re-export
    equality for both sweeps, the §3 `INTER_MODES = 4` /
    `INTER_MODE_CONTEXTS = 7` / `SWITCHABLE_FILTERS = 3` /
    `INTERP_FILTER_CONTEXTS = 4` constant pins, the zero-buffer
    `update_prob = 0` path passing every cell through unchanged on
    both default and custom starting grids, row-major cursor + value
    equivalence between each sweep and its explicit nested
    `read_diff_update_prob` loop, single-`B(252)` flag consumption per
    cell, and a cross-sweep §6.3.9 → §6.3.10 → §6.3.11 chain check
    confirming the 21 + 8 + 4 = 33 cell total advances the cursor
    identically to 33 explicit `read_diff_update_prob` calls (the
    shape the §6.3 outer-dispatch inter arm will adopt once it grows).
  * The `FrameIsIntra == 0`-gated outer dispatch in
    `parse_compressed_header` still skips the calls — §6.3.12..§6.3.17
    must land first so the coder cursor lines up across the whole
    inter branch. Round-23 surface is internal-only.

* **Round 22: §6.3.11 `read_is_inter_probs( )` compressed-header
  sweep.** Unconditional `IS_INTER_CONTEXTS = 4` `diff_update_prob`
  walk over the §10.5 `default_is_inter_prob[ IS_INTER_CONTEXTS ] =
  {9, 102, 187, 225}` initials, populating the running
  `is_inter_prob[ ]` table the round-21 §6.4.13 `read_is_inter( )`
  per-block decoder consumes via the §9.3.2 ctx:
  * `read_is_inter_probs( coder, is_inter_prob )` per §6.3.11
    (`vp9-spec.txt` lines 2154-2167) — four sequential
    `read_diff_update_prob` calls, one `B(252)` `update_prob` flag per
    slot and on 1 a `decode_term_subexp` + `inv_remap_prob` cascade
    updating `is_inter_prob[ ]` in place.
  * `DEFAULT_IS_INTER_PROB_TABLE` re-export of the round-21
    `mode_info::DEFAULT_IS_INTER_PROB` constant — single source of
    truth for the §10.5 default table across both the §6.4.13
    per-block decoder and the §6.3.11 compressed-header sweep.
  * 7 unit tests covering the §10.5 default re-export equality, the
    zero-buffer `update_prob = 0` path passing every cell through
    unchanged, four-context cell-count visiting with a custom prob
    array, equivalence between the sweep and an explicit
    `read_diff_update_prob` × 4 call sequence (probs + cursor parity),
    the §3 `IS_INTER_CONTEXTS = 4` constant pin, an exhaustive
    starting-tuple round-trip on the zero buffer, and a
    cursor-equivalence check against four explicit `B(252)` reads.
* **Round 21: §6.4.13 `read_is_inter( )` + §9.3.2 `is_inter` ctx +
  §10.5 `default_is_inter_prob`.** Adds the per-block inter/intra
  decoder the §6.4.11 `inter_frame_mode_info( )` driver fires after
  `read_skip( )`:
  * `SEG_LVL_REF_FRAME = 2` / `IS_INTER_CONTEXTS = 4` constants from §3.
  * `default_is_inter_prob[IS_INTER_CONTEXTS] = {9, 102, 187, 225}`
    transcribed verbatim from §10.5.
  * `IsInterNeighbours { above: Option<i32>, left: Option<i32> }` —
    the §6.4.11 above/left `RefFrames[ ][ ][ 0 ]` view (`None` = §6.4.11
    "unavailable → force to `INTRA_FRAME`" rule).
  * `is_inter_context( nb )` (§9.3.2) — the four-branch ctx derivation:
    both available + both intra → 3; both available + one intra → 1;
    both available + neither intra → 0; only above / only left →
    `2 * intra_flag`; neither → 0. Returns `0..=3` indexing
    `is_inter_prob[ ]`.
  * `read_is_inter( coder, seg_feature_ref_frame_active,
    segment_ref_frame_data, is_inter_prob, nb )` (§6.4.13) — the
    two-path reader: when `seg_feature_active( SEG_LVL_REF_FRAME )` is
    set, `is_inter = FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] !=
    INTRA_FRAME` without consuming any coder bits; otherwise the §9.3.3
    `BINARY_TREE` walk under `is_inter_prob[ ctx ]`.
  * 15 unit tests pinning the §10.5 default constants, the §3
    `SEG_LVL_REF_FRAME` value, every branch of `is_inter_context` (both
    unavailable / both-intra / one-intra / neither-intra / only-above
    intra-and-inter / only-left intra-and-inter / `NONE` ref-frame
    sentinel treated as intra), the §6.4.13 seg-feature path for both
    `INTRA_FRAME` and each of `LAST/GOLDEN/ALTREF_FRAME` overrides, the
    zero-coder bit=0 path, the bias-coder bit=1 path with both-intra
    neighbours, the ctx-indexes-into-`is_inter_prob` sweep across all
    four ctxs, and the path-1 short-circuit ignoring both neighbours
    and the coder.
* **Round 20: §6.4.12 `inter_segment_id( )` + §6.4.14 `get_segment_id( )`
  + §7.4 segmentation-prediction context strips.** Lands the inter-frame
  companion to the round-16 `intra_segment_id` primitive — the per-block
  segment-id reader the §6.4.11 `inter_frame_mode_info( )` driver fires
  before `read_skip( )` / `read_is_inter( )` / `read_tx_size( )`:
  * §10.2 `num_8x8_blocks_high_lookup[ BLOCK_SIZES ]` =
    `{1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8}` transcribed verbatim into
    the `partition` module alongside the existing `_WIDE_LOOKUP`.
  * `PrevSegmentIds<'a>` — borrowed row-major `MiRows × MiCols` view of
    the previous frame's segment-id plane.
  * `get_segment_id( prev, mi_row, mi_col, mi_size )` (§6.4.14) — the
    `bw` / `bh` clamp via `Min( MiCols - MiCol, bw )` /
    `Min( MiRows - MiRow, bh )` and the `seg = 7; seg = Min( seg,
    PrevSegmentIds[ … ] )` spatial-min sweep.
  * `SegPredContextState { above[MiCols], left[MiRows] }` — the §7.4.1
    / §7.4.2 strip storage with `new( )` zero-init, `clear_left( )`
    per-superblock-row reset, and `above( )` / `left( )` ctx accessors.
  * `read_seg_id_predicted( )` — the §9.3.2 `ctx =
    LeftSegPredContext[ MiRow ] + AboveSegPredContext[ MiCol ]`
    derivation + §9.3.1 `binary_tree` one-bit decode under
    `segmentation_pred_prob[ ctx ]`.
  * `inter_segment_id( )` (§6.4.12) — the four-path orchestrator:
    `!enabled` → 0; `enabled && !update_map` → `predictedSegmentId`;
    `update_map && !temporal_update` → `read_segment_id`;
    `update_map && temporal_update` → `read_seg_id_predicted` then
    either `predictedSegmentId` or a fresh `read_segment_id`, followed
    by the trailing write-back of `seg_id_predicted` into
    `AboveSegPredContext[ MiCol + i ]` / `LeftSegPredContext[ MiRow +
    i ]` over the `num_8x8_blocks_*_lookup` sub-strips.
  * 12 unit tests: `get_segment_id` (interior 2x2 min, partial-edge
    clamp, all-7 fallback), the §7.4 zero-init contract, `clear_left`
    not touching Above, the §9.3.2 ctx wiring of
    `read_seg_id_predicted`, each of the four §6.4.12 paths,
    `Error::InvalidBitstream` on missing `tree_probs` / `pred_prob`,
    and a partial-edge `BLOCK_32X32` write-back clamp.
  * Provenance: VP9 Bitstream & Decoding Process Specification v0.7
    (`docs/video/vp9/vp9-spec.txt` §6.4.4 lines 2395-2437, §6.4.7 lines
    2480-2494, §6.4.12 lines 2562-2586, §6.4.14 lines 2607-2620, §7.4.1
    lines 3824-3830, §7.4.2 lines 3831-3838, §9.3.2 lines 6313-6314,
    §10.2 line 7117). No external library source consulted.

* **Round 19: §6.4.3 recursive `decode_partition( )` driver (extending
  the crate-local `partition` module).** Composes the round-18
  `decode_partition_type( )` primitive into the recursive §6.4.3
  partition driver — the per-superblock walker the §6.4.2
  `decode_tile( )` outer loop fires at every `(r, c, BLOCK_64X64)`
  cell:
  * `decode_partition( coder, r, c, bsize, mi_rows, mi_cols,
    ctx_state, probs_kind, leaves )` — walks the §6.4.3 listing
    line-for-line: the `(r >= MiRows || c >= MiCols)` quadrant
    short-circuit, the `num8x8 = num_8x8_blocks_wide_lookup[bsize]`
    / `halfBlock8x8 = num8x8 >> 1` / `hasRows = (r + halfBlock8x8) <
    MiRows` / `hasCols = (c + halfBlock8x8) < MiCols` derivation, the
    `partition` decode via [`decode_partition_type`] (using the
    §9.3.2 `partition_plane_context` ctx + the per-frame probability
    source), the four-way dispatch on the decoded `PARTITION_*` value
    (with the HORZ second-leaf gated by `hasRows`, the VERT
    second-leaf gated by `hasCols`, and SPLIT recursing in spec
    order TL → TR → BL → BR per spec lines 2381-2384), and the
    §6.4.3 tail write-back into the partition-context strips
    (gated by `bsize == BLOCK_8X8 || partition != PARTITION_SPLIT`,
    writing `15 >> b_width_log2_lookup[ subsize ]` into
    `AbovePartitionContext[ c + i ]` and `15 >>
    b_height_log2_lookup[ subsize ]` into `LeftPartitionContext[ r +
    i ]` for `i ∈ 0..num8x8`).
  * `PartitionContextState` — the `AbovePartitionContext[ ]` /
    `LeftPartitionContext[ ]` strips (sized `Sb64Cols * 8` /
    `Sb64Rows * 8` per the §7.4 listing). Exposes
    `new( mi_cols, mi_rows )` with the §7.4 zero-reset, and
    `clear_left( )` for the §6.4.2 per-superblock-row reset
    invoked by the §6.4.2 tile driver.
  * `PartitionProbsKind` — the per-frame probability source enum:
    `Keyframe` indexes [`KF_PARTITION_PROBS`] directly per the
    §9.3.2 `FrameIsIntra == 1` arm; `Inter(&[[u8; 3]; 16])` indexes
    the caller's running `partition_probs` table (typically
    initialised from [`DEFAULT_PARTITION_PROBS`] and conditionally
    updated by the §6.3 `read_partition_probs( )` sweep — still
    pending in a later round).
  * `LeafBlock { r, c, subsize }` log records — emitted in §6.4.3
    traversal order in lieu of the §6.4.4 `decode_block( r, c,
    subsize )` call site (the per-block `mode_info` / `residual`
    decode is downstream of this driver and not yet wired). The
    deferred-leaf log is the validation surface for the recursion
    layout this round.
  * Test-only minimal range encoder (`RangeEncoder` in the
    `partition::tests` module) — a forward-simulation bounded
    brute-force search over `BoolValue ∈ 0..128` plus a DFS over
    per-renorm refill bits. For each `(bool_value, stream)`
    candidate, the §9.2.2 decoder is forward-simulated against the
    target `(bit, p)` sequence; the first candidate that produces
    every target bit wins. The trailing tail is zero-padded so any
    further renorm reads past the strictly-required bits resolve to
    0. No external library / source consulted; the search loop
    walks the §9.2.2 listing verbatim.
  * 8 new unit tests covering the recursive driver: a `RangeEncoder`
    roundtrip across an arbitrary 8-element `(bit, p)` sequence; a
    roundtrip with extreme probabilities (`p ∈ { 1, 128, 255 }`); a
    single-leaf 64x64 `PARTITION_NONE` hand-built bitstream (one
    leaf at `{ 0, 0, BLOCK_64X64 }` + §6.4.3 tail `15 >> 4 = 0`
    write-back); a four-leaf SPLIT-into-32x32-NONE hand-built
    bitstream (four leaves in TL → TR → BL → BR order at
    `{ (0,0), (0,4), (4,0), (4,4), BLOCK_32X32 }` + §6.4.3 tail
    `15 >> 3 = 1` write-back per child but not the parent SPLIT); a
    mixed HORZ/VERT quadrant hand-built bitstream (8 leaves: TL
    HORZ → 2 at BLOCK_32X16, TR VERT → 2 at BLOCK_16X32, BL HORZ,
    BR VERT, exercising both the HORZ / VERT second-leaf paths and
    the §9.3.2 ctx-derivation lockstep against the successively
    populated strip state); the `(r >= mi_rows || c >= mi_cols)`
    short-circuit invariant (no leaves emitted, strips untouched);
    the `PartitionContextState::clear_left( )` zero-the-left-strip
    invariant (above strip unchanged); and the
    `PartitionProbsKind::Inter` table dispatch matching the
    caller's row across `ctx ∈ 0..16`.

  Out of scope for round 19 (deferred):
  * The §6.3 `read_partition_probs( )` compressed-header sweep
    (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
    `diff_update_prob` cells against `DEFAULT_PARTITION_PROBS`) —
    the driver consumes the `Inter` running table, but constructing
    it lands in a later round.
  * The §6.4.4 `decode_block( )` mode-info + residual decode that
    `LeafBlock` stands in for — wiring it into this driver is
    downstream of all the §6.4 mode-info readers landing first.
  * The §6.4.2 `decode_tile( )` outer loop (the `r += 8, c += 8`
    superblock walk + per-row `clear_left_context( )`) — composes
    this driver but is a separate round.
  * The §8.4 `counts_partition` probability-adaption accumulator
    (§9.3.4 bookkeeping) for inter-frame `partition_probs[ ]`
    adaption.

  The round-19 surface stays internal-only (`pub(crate)`); the
  public API still exposes `parse_uncompressed_header`,
  `parse_compressed_header` and their result types exclusively.

* **§6.4.3 `decode_partition_type( )` per-call partition reader (new
  crate-local `partition` module).** The single-call decoder the
  recursive §6.4.3 `decode_partition( r, c, bsize )` driver (later
  round) fires once per `(r, c, bsize)` quadrant inside a tile:
  * §9.3.1 partition trees `PARTITION_TREE[6]`, `COLS_PARTITION_TREE[2]`
    and `ROWS_PARTITION_TREE[2]` transcribed verbatim from
    `docs/video/vp9/vp9-spec.txt`.
  * §3 partition enumeration: `PARTITION_NONE = 0`, `PARTITION_HORZ = 1`,
    `PARTITION_VERT = 2`, `PARTITION_SPLIT = 3`, plus dimensions
    `PARTITION_TYPES = 4` and `PARTITION_CONTEXTS = 16`.
  * §10.2 lookups transcribed verbatim: `B_WIDTH_LOG2_LOOKUP` /
    `B_HEIGHT_LOG2_LOOKUP` (the §6.4.3 tail `15 >>
    b_*_log2_lookup[subsize]` write-back inputs),
    `MI_WIDTH_LOG2_LOOKUP` (the §9.3.2 `bsl` derivation input),
    `NUM_8X8_BLOCKS_WIDE_LOOKUP` (the §6.4.3 `num8x8` input).
  * §10.2 `SUBSIZE_LOOKUP[4][13]` (`PARTITION → child block size`)
    transcribed verbatim, with `BLOCK_INVALID = 14` for the
    horizontal / vertical / split combinations that have no legal
    child at non-square parents.
  * §10.4 `KF_PARTITION_PROBS[16][3]` (keyframe / intra-only fixed
    probabilities) and §10.5 `DEFAULT_PARTITION_PROBS[16][3]` (inter
    frame initial probabilities, prior to the §6.3
    `read_partition_probs( )` sweep) transcribed verbatim. Each
    table has a shape + listing-anchor + §9.2-minimum-prob test.
  * `partition_plane_context( bsize, above_ctx, left_ctx )` — the
    §9.3.2 `ctx = bsl * 4 + left * 2 + above` derivation, with
    `bsl = mi_width_log2_lookup[bsize]`, `boffset = 3 - bsl`, and
    an OR-fold of the `AbovePartitionContext[ ]` /
    `LeftPartitionContext[ ]` strips across `num8x8` cells.
  * `decode_partition_type( coder, has_rows, has_cols, probs )` —
    the §6.4.3 reader proper: dispatches on `(has_rows, has_cols)`
    per the §9.3.1 tree-selection rule (interior → 6-entry tree,
    right-edge → 2-entry `cols_partition_tree`, bottom-edge →
    2-entry `rows_partition_tree`, corner → return
    `PARTITION_SPLIT` without consuming bits) and remaps the §9.3.3
    walker's node index per the §9.3.2 `node2` rule (interior:
    `node2 = node`, right-edge: `node2 = 1`, bottom-edge:
    `node2 = 2`). Returns one of the four `PARTITION_*` constants.
  * 37 new unit tests covering: every §3 constant (4 partition
    values + 2 dimensions); the four §10.2 lookups against the
    spec listings; `SUBSIZE_LOOKUP` `PARTITION_NONE` identity /
    `PARTITION_SPLIT` superblock anchors / `PARTITION_HORZ` +
    `PARTITION_VERT` superblock anchors / `BLOCK_INVALID` at
    non-square parents; all three §9.3.1 trees verbatim;
    `KF_PARTITION_PROBS` + `DEFAULT_PARTITION_PROBS` shape + four
    listing anchors each + §9.2 min-prob sanity;
    `partition_plane_context` zero-strip + above-only + left-only +
    both-bits-set cases across each of the four superblock sizes,
    the OR-fold across the strip, unrelated-bit masking, the
    panic-on-invalid-bsize / mismatched-strip guards, and an
    exhaustive sweep proving the 16 ctx values 0..=15 are all
    reachable; `decode_partition_type` against the zero coder
    (every arm picks its first leaf), the all-ones coder (interior
    walks every right-branch → `PARTITION_SPLIT`), the
    one-then-zero coder (each arm's first-right-then-left walk),
    the corner case (consumes zero bits + leaves bool-coder
    untouched), and an exhaustive arm × buffer × probability
    smoke-test confirming every output stays in `0..=3`.

  The recursive §6.4.3 `decode_partition( )` driver itself (which
  threads `SUBSIZE_LOOKUP[partition][bsize]` into four recursive
  calls when `PARTITION_SPLIT` and writes back the
  `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips with
  `15 >> b_*_log2_lookup[subsize]`) and the §6.3
  `read_partition_probs( )` compressed-header sweep both land in a
  later round; the round-18 surface is internal-only, `pub(crate)`.

* **§6.4.15 `intra_block_mode_info( )` inter-frame intra-block reader
  (extending the crate-local `mode_info` module).** The companion to
  the §6.4.6 keyframe driver, for intra blocks within non-keyframe
  frames:
  * §9.3.2 `SIZE_GROUP_LOOKUP[BLOCK_SIZES]`
    (`{0,0,0,1,1,1,2,2,2,3,3,3,3}`) plus §9.3
    `DEFAULT_Y_MODE_PROBS[BLOCK_SIZE_GROUPS][INTRA_MODES - 1]` (4 × 9)
    and `DEFAULT_UV_MODE_PROBS[INTRA_MODES][INTRA_MODES - 1]` (10 × 9)
    transcribed verbatim from `docs/video/vp9/vp9-spec.txt` — the
    compressed-header `y_mode_probs` / `uv_mode_probs` defaults
    (distinct from the §10.5 keyframe `kf_*_mode_probs`).
  * `intra_mode( coder, y_mode_probs, mi_size )` (§9.3.2 ctx =
    `size_group_lookup[MiSize]`), `sub_intra_mode( coder, y_mode_probs )`
    (ctx = 0), and `uv_mode( coder, uv_mode_probs, y_mode )` (ctx =
    `y_mode`) — §9.3.3 walks over `INTRA_MODE_TREE` with the §9.3
    compressed-header rows.
  * `intra_block_mode_info( )` (§6.4.15) returning
    `Vp9IntraBlockModeInfo { ref_frame_0, ref_frame_1, y_mode,
    sub_modes[4], uv_mode }`. Sets `ref_frame[0] = INTRA_FRAME`,
    `ref_frame[1] = NONE`; the `MiSize >= BLOCK_8X8` arm decodes one
    `intra_mode` replicated across `sub_modes[ ]`, the sub-8x8 arm
    walks the `(idy, idx)` grid decoding one `sub_intra_mode` per cell
    (`y_mode` = last decoded). Reads modes only — `segment_id` / `skip`
    / `tx_size` are decoded by the §6.4.11 driver beforehand.
  * §6.4.5 `mode_info( )` dispatch: a `Vp9ModeInfo` enum
    (`IntraFrame` / `InterFrameIntraBlock`) plus
    `inter_frame_intra_block_mode_info( )` wiring the §6.4.15 path
    alongside the existing §6.4.6 keyframe path.
  * Per-table shape + anchor + §9.2 min-prob tests for
    `SIZE_GROUP_LOOKUP` / `DEFAULT_Y_MODE_PROBS` / `DEFAULT_UV_MODE_PROBS`,
    instrumented-callback ctx-row tests for each reader, hand-traced
    bias-buffer decodes, and a per-block decode scenario (BLOCK_8X8
    bias buffer → `y_mode = D207_PRED`, `uv_mode = D153_PRED`). The
    surface stays crate-internal (`pub(crate)`).
* **Round 17: §6.4.6 `intra_frame_mode_info( )` keyframe driver
  (extending the crate-local `mode_info` module).** Wires the rounds
  15 / 16 primitives into the top-level §6.4.6 per-block mode-info
  reader for keyframe (and intra-only) frames:
  * §9.3.1 `intra_mode_tree[18]` constant
    `{ -DC_PRED, 2, -TM_PRED, 4, -V_PRED, 6, 8, 12, -H_PRED, 10,
    -D135_PRED, -D117_PRED, -D45_PRED, 14, -D63_PRED, 16, -D153_PRED,
    -D207_PRED }` transcribed verbatim — the 18-entry / 10-leaf tree
    shared by `default_intra_mode` / `default_uv_mode` / `intra_mode`
    / `sub_intra_mode` / `uv_mode`.
  * §10.5 `KF_Y_MODE_PROBS[10][10][9]` (a 900-byte 3D table indexed by
    `[abovemode][leftmode][node]` per the §9.3.2 `default_intra_mode`
    listing) transcribed verbatim from the spec listing
    (lines 7463–7599).
  * §10.5 `KF_UV_MODE_PROBS[10][9]` (a 90-byte 2D table indexed by
    `[y_mode][node]` per the §9.3.2 `default_uv_mode` listing)
    transcribed verbatim from the spec listing (lines 7602–7613).
  * `default_intra_mode( coder, abovemode, leftmode )` and
    `default_uv_mode( coder, y_mode )` — §9.3.3 walks over
    `INTRA_MODE_TREE` with the respective `kf_*_mode_probs` row.
  * `intra_frame_mode_info()` (§6.4.6) — the orchestrator threading
    `intra_segment_id( )` + `read_skip( )` + `read_tx_size( 1 )` +
    `default_intra_mode` + `default_uv_mode` into a
    `Vp9IntraMiBlock { segment_id, skip, tx_size, ref_frame_0,
    ref_frame_1, is_inter, y_mode, sub_modes[4], uv_mode }`. The
    §6.4.6 `ref_frame[0] = INTRA_FRAME = 0` / `ref_frame[1] = NONE =
    -1` / `is_inter = false` triple is hardwired per the spec
    listing. Handles both the `MiSize >= BLOCK_8X8` single-mode
    partition (one `default_intra_mode` decode replicated into all
    four `sub_modes[ ]` cells) and the `MiSize < BLOCK_8X8` sub-mode
    walk (the §6.4.6 `(idy, idx)` grid stepped by
    `num_4x4_blocks_high_lookup[MiSize]` /
    `num_4x4_blocks_wide_lookup[MiSize]` — 4 reads for BLOCK_4X4, 2
    for BLOCK_4X8 / BLOCK_8X4 — with each cell receiving its own
    decoded mode replicated across the (num4x4h × num4x4w)
    `sub_modes[ ]` sub-grid; `y_mode` set to the *last* decoded
    `default_intra_mode`).
  * `IntraFrameNeighbours` bundle — per-MI-block neighbour state a
    tile driver builds from its frame-wide `SubModes[ ][ ][ ]` array
    (positions {2, 3} of the above neighbour, positions {1, 3} of the
    left neighbour, plus the §7.4.4 `AvailU` / `AvailL` flags). The
    §9.3.2 listing reads only those four cells; `DC_PRED` is
    substituted when the corresponding `avail_*` flag is false.
* **Round 16: §6.4.7 `intra_segment_id( )` + §9.3.1 `segment_tree[14]`
  (extending the crate-local `mode_info` module).** Lands the next
  slice of the §6.4.6 `intra_frame_mode_info()` orchestrator's
  primitives that round 15 left deferred:
  * The §9.3.1 `segment_tree[14]` constant
    `{ 2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7 }` transcribed
    verbatim — the 7-leaf binary tree used by every `segment_id`
    decode site (intra §6.4.7 + inter §6.4.12).
  * `read_segment_id( coder, tree_probs )` — the §9.3.3 walk over the
    new `SEGMENT_TREE` with per-node probability
    `segmentation_tree_probs[node]` per the §9.3.2 listing's
    `segment_id` entry. Returns the decoded segment id in `0..=7`.
  * `intra_segment_id( coder, segmentation_enabled,
    segmentation_update_map, tree_probs )` (§6.4.7) — the
    `segmentation_enabled && segmentation_update_map` gate around
    `read_segment_id`, falling through to `segment_id = 0` otherwise
    (the intra-only path has no `segmentation_temporal_update` /
    `seg_id_predicted` machinery — that's inter-only and lands when
    the §6.4.12 syntax does).
* **Round 15: §6.4.8 `read_skip` + §6.4.10 `read_tx_size` + §9.3.3
  `tree_decode` (crate-local `mode_info` module).** The first slice of
  the §6.4 per-block mode-info decode that the round-14
  `residual_intra` driver currently consumes via a caller-supplied
  bundle — unblocks the per-block `BoolCoder`-driven mode-info wiring
  the §6.4.6 `intra_frame_mode_info()` orchestrator will need.
  * `tree_decode( coder, tree, prob )` — the §9.3.3 generic tree
    decoding loop `do { n = T[n + read_bool(P(n >> 1))] } while (n >
    0)` that every tree-coded syntax element routes through. The
    probability callback is a `FnMut(usize) -> u8` so call-sites can
    splice the right §9.3.2 row in without the helper needing to know
    which syntax element it's decoding.
  * §9.3.1 trees `tx_size_8_tree[2]` / `tx_size_16_tree[4]` /
    `tx_size_32_tree[6]` and `binary_tree[2]` transcribed verbatim
    from the spec listing.
  * `skip_context( nb )` (§9.3.2) — the `Skips[MiRow-1][MiCol] +
    Skips[MiRow][MiCol-1]` ctx derivation with `AvailU` / `AvailL`
    gating; `tx_size_context( nb, max_tx_size )` (§9.3.2) — the
    `(above + left) > maxTxSize` ctx derivation that consults
    neighbour `TxSizes[ ]` only on unskipped MI blocks (and mirrors
    the side when a neighbour is unavailable).
  * `read_skip( coder, seg_feature_skip_active, skip_prob, nb )`
    (§6.4.8) — the §6.4.9 `seg_feature_active(SEG_LVL_SKIP)`
    early-return rule plus the §9.3.2 binary tree decode under
    `skip_prob[skip_context(nb)]`.
  * `read_tx_size( coder, allow_select, tx_mode, mi_size, tx_probs,
    nb )` (§6.4.10) — the `allow_select && tx_mode == TX_MODE_SELECT
    && MiSize >= BLOCK_8X8` path picking the §9.3.1 tree by
    `max_txsize_lookup[MiSize]` and the §9.3.2 ctx, falling through
    to `Min(maxTxSize, tx_mode_to_biggest_tx_size[tx_mode])` per the
    spec's `else` branch.
  * `NeighbourSkips` / `NeighbourTxSizes` — the per-MI-block
    neighbour-state bundles a tile driver builds from its
    frame-wide `Skips[ ][ ]` / `TxSizes[ ][ ]` arrays.
  * 22 unit tests: the §9.3.1 tree-listing anchors (verbatim
    transcription check), the §9.3.3 walker with a zero-coder
    (every read yields bit=0, every tree picks its first leaf),
    a "bias buffer" prefix `[0x7F, 0x00, ...]` whose post-marker
    coder state lets the first `read_bool(255)` flip to 1 (one
    right-branch step in any tree), the node-index argument-order
    invariant for `tree_decode`'s prob callback, exhaustive
    `skip_context` cases (no neighbours, single-side, both-sides
    mixed and matching), `tx_size_context` against the §9.3.2
    listing across max_tx_size 0..=3 plus the skipped-neighbour
    fallback and the `!AvailL` / `!AvailU` mirroring, the §6.4.8
    `seg_feature_active` early-return path, `read_skip` against
    both the zero coder (always false) and the bias coder + p=255
    (true), the §6.4.10 `else`-branch `Min(max, biggest)` fallback
    for `allow_select == false` / non-SELECT `tx_mode` / sub-8x8
    `MiSize`, the §9.3.1 tree-dispatch for `BLOCK_8X8` /
    `BLOCK_16X16` / `BLOCK_32X32`, and the row-by-ctx selection
    correctness via the spec-derived ctx derivation lockstep
    (both ctx=0 and ctx=1 evaluated against `tx_size_context`).
    Every test consumes the §9.2 BoolCoder through valid byte
    buffers (marker-bit conformant) hand-derived by walking the
    §9.2 listing — no external library or source was consulted.
  * Out of scope this round: the §6.4.6 `intra_frame_mode_info()`
    orchestrator (which wires `read_skip` + `read_tx_size` + the
    deferred §6.4.7 `intra_segment_id` + §6.4.15
    `intra_block_mode_info` into a single `Vp9IntraMiBlock`); the
    `Skips[ ][ ]` / `TxSizes[ ][ ]` frame-wide array write-back
    (left to the §6.4.6 driver); inter-frame mode info (§6.4.11+,
    needs reference-buffer state); and the §8.4 `counts_skip` /
    `counts_tx_size` probability-adaption accumulators. The
    round-15 surface is internal-only; the public API still
    exposes `parse_uncompressed_header`, `parse_compressed_header`
    and their result types exclusively.
* **Round 14: §6.4.21 `residual( )` intra driver (crate-local `residual`
  module).** The §6.4.21 outer loop for the intra path — the per-plane,
  per-4x4-sub-block walk that owns the `AboveNonzeroContext` /
  `LeftNonzeroContext` write-back across a whole MI block, drives the
  round-13 §6.4.24 `tokens( )` per-block decode, and feeds the round-11
  §8.6.2 `reconstruct_block` with real per-block `Tokens` arrays,
  availability flags and plane/quantizer state.
  * §10.2 `num_4x4_blocks_wide_lookup[13]` / `num_4x4_blocks_high_lookup[13]`,
    §6.4.10 `max_txsize_lookup[13]`, and §6.4.23 `ss_size_lookup[13][2][2]`
    tables transcribed verbatim, alongside the `BLOCK_4X4 .. BLOCK_64X64`
    / `BLOCK_INVALID` `subsize` constants from §3.
  * `get_plane_block_size( subsize, plane, subsampling_x, subsampling_y )`
    (§6.4.23) and `get_uv_tx_size( tx_size, mi_size, subsampling_x,
    subsampling_y )` (§6.4.22) — the chroma-plane block-size /
    transform-size derivations that key the per-plane loop.
  * `ResidualBlockCtx` — the per-MI-block / per-frame bundle (`MiCol` /
    `MiRow` / `MiCols` / `MiRows`, `MiSize`, `tx_size`, `subsampling_x` /
    `y`, `skip`, `Lossless`, `BitDepth`, the per-block `PredMode` for
    luma and chroma, and the per-plane DC/AC quantizers from round 8);
    plus `AvailFlags` for §7.4.4 `AvailL` / `AvailU` and a
    `PlaneBuffers` wrapper for the three `CurrFrame[ plane ]` planes.
  * `residual_intra( planes, nz, block, avail, token_source )` — the
    §6.4.21 driver proper: per plane, computes `bsize = MiSize <
    BLOCK_8X8 ? BLOCK_8X8 : MiSize`, the per-plane `planeSz` +
    `num4x4w` / `num4x4h` dimensions and chroma `txSz`, then walks the
    `(y, x)` 4x4 grid stepping by `step = 1 << txSz`. For each in-bounds
    transform block (`startX < maxx && startY < maxy`) it calls the
    round-10 `predict_intra` with the resolved `have_left` /
    `have_above` / `not_on_right` flags, pulls `Tokens[ ]` from a
    per-block `TokenSource` callback (when `!skip`), derives the §6.4.25
    `TxType` (chroma / `TX_32X32` / lossless force `DCT_DCT`; luma intra
    uses round-11 `tx_type_for_intra`), runs the round-11
    `reconstruct_block`, and writes
    `AboveNonzeroContext[ plane ][ x4 + i ] = LeftNonzeroContext[
    plane ][ y4 + i ] = nonzero` for `i ∈ 0..step` per the §6.4.21
    trailing write-back.
  * 12 unit tests: the §10.2 / §6.4.10 / §6.4.23 table-anchor checks
    (luma-identity invariant + 4:2:0 / asymmetric subsampling
    anchors), the §6.4.22 `get_uv_tx_size` chroma cap and the sub-8x8
    short-circuit, the `skip = true` path leaving every strip cell at 0
    (no token decode), the full `skip = false` walk firing
    `token_source` exactly 16 luma + 4 U + 4 V times for a `BLOCK_16X16`
    MI block at `tx_size = TX_4X4` (each call recorded with `(plane,
    block_idx, tx_sz)`), the `nonzero = true` strip write-back over
    `step = 1 << tx_sz` 4-sample units for `tx_size = TX_8X8`, the
    out-of-bounds block skip with intact zero context, a DC-only luma
    block at MI (1,1) lockstep against an independent `predict_intra` +
    `reconstruct_block` probe, and the `bsize = max(MiSize, BLOCK_8X8)`
    widening for a `BLOCK_4X4` MI block. No external library / source
    was consulted; every formula and table is transcribed directly from
    the §6.4.21 / §6.4.22 / §6.4.23 / §10.2 listings.
  * The `is_inter` branch of §6.4.21 (which calls `predict_inter( )`
    before the per-block loop) is deferred until the §8.5.2 inter
    prediction process and reference-buffer state land; the per-block
    mode-info decode (`y_mode` / `sub_modes` / `tx_size` / `skip` /
    `segment_id` from §6.4) that the residual loop reads is also a
    later-round increment. The round-14 surface is internal-only.

* **Round 13: §6.4.24 `tokens( )` per-block coefficient driver
  (crate-local `tokens` module).** Walks the round-12 §6.4.25 scan
  order and feeds each scan position through the round-7
  `read_coef_token` pipeline, recovering one transform block's
  quantised coefficients into a `Tokens[ ]` array.
  * The §10 band tables — `coefband_4x4[ 16 ]` transcribed verbatim and
    `coefband_8x8plus[ 1024 ]` built from the verbatim 21-entry prefix
    plus the all-`5` tail — selected by `coef_band( c, txSz )` per the
    §6.4.24 `(txSz == TX_4X4) ? coefband_4x4 : coefband_8x8plus` rule.
  * `token_cache_neighbours( c, pos, txSz, txType )` — the §9.3.2
    neighbour pair (`nb[ 0 ]` / `nb[ 1 ]`): `(0, 0)` for the DC
    coefficient, and for `c > 0` the above (`(i-1)*n + j`) / left
    (`i*n + j-1`) raster cells with the `DCT_ADST` (double above) /
    `ADST_DCT` (double left) / first-row / first-column variants
    (`n = 4 << txSz`).
  * `build_token_probs( cell )` — the §9.3.2 10-node probability array:
    node 0 → `cell[1]`, node 1 → `cell[2]`, node `2..=9` →
    `pareto( node, cell[2] )`.
  * `NonzeroContext` (the per-plane `AboveNonzeroContext` /
    `LeftNonzeroContext` 4-sample strips) and `TokenBlockCtx` (the
    per-block / per-frame state `tokens( )` reads — `plane`,
    `is_inter`, resolved `TxType`, `BitDepth`, `x4` / `y4`, `maxX` /
    `maxY`).
  * `tokens( coder, block, txSz, scan, coef_probs, nz, token_cache,
    tokens )` — the §6.4.24 driver: `segEob = 16 << (txSz << 1)`, the
    `checkEob` gating, the §9.3.2 per-coefficient `ctx` (DC from the
    non-zero strips, `c > 0` from `TokenCache`), the
    `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell pick, the
    `more_coefs` / `token` / `read_coef` / `sign_bit` decode, the
    `TokenCache[ pos ] = energy_class[ token ]` write, the
    `ZERO_TOKEN`-clears-`checkEob` rule, the trailing `Tokens[ scan[ i
    ] ] = 0` zero-fill, and the `nonzero = c > 0` return.
  * 15 unit tests: the band tables vs the §10 listing + the
    `coef_band` dispatch, the §9.3.2 neighbour derivation (DC, interior
    DCT_DCT / ADST variants, first row / column, 8x8 width scaling),
    `build_token_probs` node mapping, and the `tokens( )` driver
    (zero-buffer immediate EOB + all-zero fill, the
    `ZERO_TOKEN`-clears-`checkEob` block fill, a lockstep match against
    an independent `read_coef_token` walk, the DC non-zero-context cell
    routing, the trailing zero-fill, and the `NonzeroContext::new`
    all-zero invariant). No external library source was consulted; the
    band tables and every formula are transcribed directly from the
    §6.4.24 / §9.3.2 / §10 listings. The §6.4.21 residual loop that
    threads `NonzeroContext` across the frame and feeds the round-11
    reconstruct driver lands in a later round; the round-13 surface is
    internal-only.

* **Round 12: §6.4.25 `get_scan` scan-order selection (crate-local
  `scan` module).** The first step of the §6.4.24 `tokens( )` per-block
  driver — it picks the scan order (the sequence of raster positions
  `pos = scan[ c ]` the coefficient loop visits) for a transform block.
  * The §10.1 scan tables transcribed verbatim: `default_scan_4x4` /
    `col_scan_4x4` / `row_scan_4x4` (16 entries), the 8x8 trio (64),
    the 16x16 trio (256), and `default_scan_32x32` (1024). Element
    type `u16` so the 32x32 table's `0..=1023` range fits.
  * `get_scan( plane, txSz, txType )` — the §6.4.25 selection:
    `ADST_DCT` → `row_scan`, `DCT_ADST` → `col_scan`, else
    (`DCT_DCT` / `ADST_ADST`) → `default`, applying the §6.4.25 first
    half (a chroma plane or a `TX_32X32` block forces `TxType =
    DCT_DCT`). The mode-info-dependent `mode2txfm_map[ y_mode ]`
    `TxType` derivation already lives in
    [`reconstruct::tx_type_for_intra`]; the per-block mode-info state
    is owned by the (deferred) §6.4.21 residual driver.
  * `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` `txSz` index
    constants (§3).
  * 10 unit tests: every table's spec length, the permutation
    invariant (each raster position appears exactly once), the
    DC-first invariant, §10.1 listing anchors (first four + last
    entry of each table), the §6.4.25 `txType` → table selection for
    4x4 / 8x8 / 16x16, the `TX_32X32`-always-default and
    chroma-forces-default first-half overrides, and the
    `16 << (txSz << 1)` `segEob` length match. No external library
    source was consulted; the tables are transcribed directly from the
    §10.1 listing.

* **Round 11: §8.6.2 reconstruct driver (crate-local `reconstruct`
  module).** Ties the rounds 7-10 pieces together at the conceptual
  `reconstruct( plane, startX, startY, txSz )` call site of the
  §6.4.21 residual syntax.
  * `tx_type_for_intra( mode )` — the §6.4.25 `mode2txfm_map[ y_mode ]`
    lookup selecting the `TxType` (`DCT_DCT` / `ADST_DCT` / `DCT_ADST`
    / `ADST_ADST`) for a luma intra block from its `PredMode`. The
    10-entry intra prefix of `mode2txfm_map` (§10.5) is transcribed
    verbatim.
  * `reconstruct_block( plane_buf, x, y, tx_sz, tokens, dc_quant,
    ac_quant, tx_type, lossless, bit_depth )` (§8.6.2) — sets
    `dqDenom = 2` for `txSz == TX_32X32` else `1`, `n = 2 + txSz`,
    `n0 = 1 << n`; step 1 `Dequant[i][j] = (Tokens[i*n0+j] *
    get_ac_quant) / dqDenom`; step 2 the `Dequant[0][0] = (Tokens[0] *
    get_dc_quant) / dqDenom` DC override; step 3 the §8.7.2
    `inverse_transform_2d`; step 4 `CurrFrame[plane][y+i][x+j] =
    Clip1( CurrFrame[...] + Dequant[i][j] )`. Integer division
    truncates toward zero per §4.1 (Rust's `i64 /` matches).
  * `reconstruct_intra_block( … )` — the end-to-end one-block driver:
    predicts via §8.5.1 `predict_intra` (round 10), derives the
    `TxType` with the §6.4.25 `TX_32X32` / lossless `DCT_DCT`
    overrides, then runs `reconstruct_block`. The shape the deferred
    §6.4.21 residual loop will call once it threads real availability
    and quantizer state.
  * Crate-local `clip1` (§3) helper operating in `i64` so the
    high-precision residual sum does not overflow before clamping.
  * 10 unit tests: the `mode2txfm_map` intra prefix vs the §10.5
    listing, `clip1` range clamping, an all-zero `Tokens` block
    leaving the prediction unchanged, a DC-only `Tokens` block adding
    a flat residual to a flat prediction, step-4 clipping at both the
    bit-depth max and zero, the `TX_32X32` `dqDenom = 2` halving
    through the real driver, `reconstruct_intra_block` with DC_PRED +
    zero tokens equalling the pure DC prediction, DC_PRED + a known DC
    residual reconstructing the expected pixels, and the lossless WHT
    path driven via `reconstruct_intra_block`.

* **Round 10: §8.5.1 intra prediction process (crate-local `intra`
  module).**
  * `PredMode` — the 10 §7.4.5 intra prediction modes with
    discriminants matching the spec numbering exactly (`DC_PRED` = 0,
    `V_PRED` = 1, `H_PRED` = 2, `D45_PRED` = 3, `D135_PRED` = 4,
    `D117_PRED` = 5, `D153_PRED` = 6, `D207_PRED` = 7, `D63_PRED` = 8,
    `TM_PRED` = 9) plus `from_raw` for the (deferred) mode-info decode.
  * `Plane` — a minimal row-major `i32` plane buffer standing in for
    `CurrFrame[ plane ]`; the §8.5.1 process reads neighbour samples
    from it and writes the prediction back.
  * `predict_intra( plane, x, y, have_left, have_above, not_on_right,
    tx_sz, mode, max_x, max_y, bit_depth )` (§8.5.1) — builds
    `aboveRow[-1 .. 2*size-1]` and `leftCol[0 .. size-1]` per the
    `haveAbove` / `haveLeft` / `notOnRight` availability rules (the
    upper-right extension fires only for `txSz == 0`; missing
    neighbours fill `(1<<(BitDepth-1)) ± 1`), forms the `pred` block
    for the selected mode — `V`/`H` copies; the four `DC` neighbour
    cases (`avg` = `(sum + size) >> (log2Size+1)`, `leftAvg` /
    `aboveAvg` = `(sum + (1<<(log2Size-1))) >> log2Size`, and the
    `1<<(BitDepth-1)` no-neighbour fill); the `D45`/`D63`/`D117`/
    `D135`/`D153`/`D207` directional `Round2` recurrences (including
    the reverse-order `D207` step 5); and `TM` with `Clip1` — then
    stores it back. Neighbour reads clamp with `Min(maxX, .)` /
    `Min(maxY, .)`. Crate-local `round2` (§3) and `clip1` (§3)
    helpers.
  * 16 unit tests: `PredMode::from_raw` round-tripping the §7.4.5
    numbering (and the discriminant values), `round2` / `clip1`
    against their §3 definitions, `V_PRED` / `H_PRED` copy semantics,
    all four `DC_PRED` neighbour cases, the `TM_PRED` formula plus
    out-of-range clipping, a constant-neighbour invariant collapsing
    every directional mode to the neighbour constant across all four
    transform sizes, the `D207_PRED` bottom-row / step-2 formulas, the
    `notOnRight`-gated upper-right `aboveRow` extension (via
    `D45_PRED`, enabled only for `txSz == 0`), and the `Min(maxX, .)`
    plane-edge clamping of neighbour reads. No external library /
    source was consulted; every formula is transcribed directly from
    the spec §8.5.1 listing. The §8.6.2 reconstruct driver that
    supplies the real availability flags and adds the round-9
    inverse-transformed residual to this prediction remains deferred
    to a future round; the round-10 surface is internal-only.

* **Round 9: §8.7 inverse transform process (crate-local `idct`
  module).**
  * The §8.7.1.1 butterfly primitives — `B` (butterfly rotation,
    including the `16 + 32*k` two-multiply fast path), `H` (Hadamard
    rotation), `SB` (butterfly into the high-precision `S` array) and
    `SH` (Hadamard rotation + `Round2(·,14)` out of `S`) — plus the
    `cos64` / `sin64` angle functions backed by the verbatim 33-entry
    `COS64_LOOKUP` quarter-wave table and the `brev` bit-reversal
    helper. All fixed-point intermediates are evaluated in `i64`
    (the spec notes `S` needs `24 + BitDepth` bits).
  * `inverse_dct( t, n )` (§8.7.1.2 + §8.7.1.3) — the inverse-DCT
    array permutation (`T[i] = copyT[ brev(n, i) ]`) followed by the
    recursive inverse DCT process for `2 <= n <= 5`.
  * `inverse_adst( t, n )` (§8.7.1.4 .. §8.7.1.9) — the ADST
    input/output permutations and the ADST4 / ADST8 / ADST16
    processes (the `SINPI_1_9 .. SINPI_4_9` constants transcribed
    verbatim) dispatched by `n` for `2 <= n <= 4`.
  * `inverse_wht( t, shift )` (§8.7.1.10) — the in-place inverse
    Walsh-Hadamard transform with the `shift` pre-scaling argument.
  * `inverse_transform_2d( dequant, n, tx_type, lossless )` (§8.7.2)
    — the 2D driver: per-`TxType` row transform then column
    transform over a `(1<<n)` by `(1<<n)` `Dequant` block, the
    lossless WHT path (`shift = 2` rows / `0` columns), and the
    `Round2( T[i], Min(6, n+2) )` column rounding. `TxType` constants
    `DCT_DCT` / `ADST_DCT` / `DCT_ADST` / `ADST_ADST` are defined per
    §3.
  * 20 unit tests: `cos64` quarter-wave symmetry + periodicity,
    `sin64` shift, `brev`, the Hadamard sum/difference + flip
    semantics, the `16+32*k` butterfly fast-path equivalence, the
    DC-only "flat output" property of the 4/8/16/32-point inverse
    DCT, zero-in/zero-out for all ADST sizes, the §8.7.1.5 output
    permutation indices, and the 2D driver's zero-in/zero-out (all
    four `TxType`s, `n = 2..5`) + DC-only flat-block property (lossy
    and lossless paths). No external library / source was consulted;
    the `cos64_lookup` table and `SINPI_*_9` constants are
    transcribed directly from the spec §8.7.1 listings. The §8.6.2
    reconstruct driver that builds the `Dequant` input (round-7 token
    magnitudes scaled by the round-8 quantizers) and adds the
    residual to the prediction remains deferred to a future round.

* **Round 8: §8.6.1 dequantization functions (crate-local `dequant`
  module).**
  * `dc_q( bit_depth, b )` / `ac_q( bit_depth, b )` (§8.6.1) — index
    the `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
    `(BitDepth - 8) >> 1` row and the `Clip3(0, 255, b)` column.
    Both 256-entry tables are transcribed verbatim from the §8.6.1
    listing into `DC_QLOOKUP` / `AC_QLOOKUP`.
  * `seg_feature_active( seg, segment_id, feature )` (§6.4.9) —
    `segmentation_enabled && FeatureEnabled[ segment_id ][ feature ]`.
  * `get_qindex( seg, quant, segment_id )` (§8.6.1) — applies the
    `SEG_LVL_ALT_Q` segment feature (absolute update replaces
    `base_q_idx`, delta update offsets it, then `Clip3(0, 255, .)`)
    or returns `base_q_idx` when the feature is inactive.
  * `get_dc_quant( plane, .. )` / `get_ac_quant( plane, .. )`
    (§8.6.1) — combine `get_qindex()` with the plane-specific header
    delta (`delta_q_y_dc` luma DC, `delta_q_uv_dc` chroma DC,
    `delta_q_uv_ac` chroma AC; luma AC has none) and dispatch to
    `dc_q` / `ac_q`.
  * `SEG_LVL_ALT_Q` constant (§3 table of constants) plus a private
    `clip3` helper (§5.1).
  * 13 unit tests including table-shape / spec-anchor checks, the
    `clip3` clamp branches, `dc_q` / `ac_q` index clipping + row
    selection, all three `get_qindex` paths, plane-delta selection
    for `get_dc_quant` / `get_ac_quant`, and the high-bit-depth
    divergence of the same qindex across the three table rows. The
    §8.6.2 reconstruct driver that consumes these helpers (scaling
    the round-7 `Tokens` array, with the `dqDenom = 2` halving for
    `TX_32X32`) remains deferred to a future round. No external
    library / source was consulted; the lookup tables are
    transcribed directly from the spec §8.6.1 listing.

* **Round 7: §6.4.24 / §6.4.26 coefficient-token decoder (crate-local
  `tokens` module).**
  * `read_token( coder, &probs )` (§6.4.24, §9.3.3) — walks the
    20-entry `token_tree` returning one of `ZERO_TOKEN` ..
    `DCT_VAL_CATEGORY6`. `probs[0..=9]` are the 10 internal-node
    `read_bool` probabilities pre-derived from
    `coef_probs[...][1]` / `coef_probs[...][2]` via `pareto`.
  * `pareto( node, prob )` (§9.3.2) — short-circuits to `prob` for
    `node < 2`; otherwise looks up
    `PARETO_TABLE[ (prob - 1) / 2 ][ node - 2 ]` (odd `prob`) or
    interpolates two adjacent rows (even `prob`). The full 128×8
    pareto table is transcribed verbatim from spec §10.3.
  * `read_more_coefs( coder, prob )` (§6.4.24, §9.3.2) — single
    `B(p)` returning `true` (continue scan) / `false` (EOB).
  * `read_coef( coder, token, bit_depth )` (§6.4.26) — extra-bits
    decoder. For tokens `ONE`..`FOUR` no bits are read and the base
    coef is returned. For `DCT_VAL_CATEGORY1..6` the `numExtra`
    `B(p)` reads against `cat_probs[cat]` build the magnitude in
    `EXTRA_BITS[token][2] + (extra_bits_value)`. For `CAT6` at
    `bit_depth ∈ {10, 12}` an additional `BitDepth - 8` `B(255)`
    `high_bit` reads prepend MSBs at shift `5 + BitDepth - e`.
  * `read_coef_token( coder, check_eob, more_coefs_prob,
    &token_probs, bit_depth )` — driver returning
    `CoefStep::Eob | Coef { token, value }`. Folds `read_more_coefs`
    + `read_token` + `read_coef` + `L(1) sign_bit` together. The
    `checkEob` flag itself stays in the caller's residual loop per
    §6.4.24.
  * `EXTRA_BITS[11][3]`, `CAT_PROBS[7][14]`, `ENERGY_CLASS[12]`,
    `TOKEN_TREE[20]`, `PARETO_TABLE[128][8]` constants — all
    transcribed verbatim from spec §6.4.26 / §10.2 / §10.3 / §9.3.
  * 28 unit tests including hand-traced golden buffers for
    `ONE_TOKEN` (`0x40 0x00 …`) and `TWO_TOKEN` (`0x60 0x00 …`)
    derived by stepping the §9.2 decoder by hand. No external
    library / source was consulted; the §6.4.21 `residual( )` driver
    that will consume these helpers remains deferred to a future
    round.

* **Round 6: §6.3.7 `read_coef_probs` 6D coefficient-probability
  sweep wired into `parse_compressed_header`.**
  * `read_coef_probs(&mut coder, tx_mode, &mut coef_probs)` (§6.3.7)
    — walks `txSz ∈ [TX_4X4, maxTxSize]` with `maxTxSize =
    tx_mode_to_biggest_tx_size[ tx_mode ]` (§10.5). Per active
    tx-size, reads an outer `L(1) update_probs` flag and, on 1,
    drives a nested `(i, j, k, l, m)` sweep over `BLOCK_TYPES=2 ×
    REF_TYPES=2 × COEF_BANDS=6 × maxL(k) × UNCONSTRAINED_NODES=3`
    cells, with `maxL = (k == 0) ? 3 : 6` per §6.3.7 (band 0 has
    only 3 valid previous-coef contexts). Each cell becomes
    `read_diff_update_prob( coder, cell )`. Fully-active inner
    walk is 396 cells per tx-size, 1584 for a TX_MODE_SELECT
    frame.
  * `tx_mode_to_biggest_tx_size( tx_mode )` const-fn (§10.5) —
    maps `TxMode` to the spec's biggest-tx-size index, with the
    `ALLOW_32X32` and `TX_MODE_SELECT` rows both mapping to
    `TX_32X32 = 3`.
  * `coef_probs::DEFAULT_COEF_PROBS: CoefProbs` constant in the
    new `src/coef_probs.rs` module — the §10
    `default_coef_probs[TX_SIZES=4][BLOCK_TYPES=2][REF_TYPES=2][
    COEF_BANDS=6][PREV_COEF_CONTEXTS=6][UNCONSTRAINED_NODES=3]`
    table (1728 u8 entries) transcribed verbatim from the spec
    listing. Band-0 trailing `{0, 0, 0} // unused` rows preserved
    as in-table sentinels matching the `maxL = 3` clamp.
  * `CoefProbs` public type alias re-exported from the crate root
    (`pub use coef_probs::CoefProbs;`), naming the 6D array shape.
  * `Vp9CompressedHeader` extended with `pub coef_probs:
    CoefProbs`. `parse_compressed_header` now runs the sweeps in
    spec order: `read_tx_mode` → optional `read_tx_mode_probs` →
    `read_coef_probs` → `read_skip_prob`.
  * `Vp9CompressedHeader` no longer derives `Copy` — the 1728-byte
    `coef_probs` field makes silent copies costly. `Clone` is
    retained.
  * 7 new unit tests: `tx_mode_to_biggest_tx_size` against the
    §10.5 listing, `read_coef_probs` zero-buffer passthrough for
    `ONLY_4X4` (1-slab) and `TX_MODE_SELECT` (4-slab) modes, an
    across-mode sweep, `DEFAULT_COEF_PROBS` shape +
    spec-listing anchors (TX_4X4 / block-type 0 / Intra / band 0
    / ctx 0 = {195, 29, 183}; TX_32X32 / block-type 1 / Inter /
    band 5 / ctx 5 = {1, 16, 6}), the band-0 unused-row sentinel
    invariant, and the inner-sweep `2 × 2 × (3 + 5 × 6) × 3 =
    396` cell-count check.
  * 2 new end-to-end integration tests
    (`tests/compressed_header.rs`):
    `end_to_end_tx_mode_select_runs_coef_probs_sweep` driving the
    full TX_MODE_SELECT → tx_mode_probs → coef_probs → skip_prob
    chain through `parse_uncompressed_header` +
    `parse_compressed_header` and verifying two §10 default
    anchors survive verbatim; and
    `end_to_end_only_4x4_visits_only_first_tx_size_coef_slab`
    confirming the outer-loop tx-size clipping for the ONLY_4X4
    path.

* **Round 5: §6.3.2 `tx_mode_probs` + §6.3.8 `read_skip_prob` sweeps
  wired into `parse_compressed_header`.**
  * `read_tx_mode_probs(&mut coder, &mut tx_probs)` (§6.3.2) —
    three nested sweeps (`TX_SIZE_CONTEXTS * (1+2+3) = 12` cells)
    walking `tx_probs_8x8`, `tx_probs_16x16`, `tx_probs_32x32` via
    `read_diff_update_prob`. Gated on
    `tx_mode == TX_MODE_SELECT` per the §6.3 compressed-header
    syntax dispatch.
  * `read_skip_prob(&mut coder, &mut skip_prob)` (§6.3.8) —
    unconditional `SKIP_CONTEXTS = 3` sweep via
    `read_diff_update_prob`.
  * `DEFAULT_TX_PROBS: [[[u8; 3]; 2]; 4]` and
    `DEFAULT_SKIP_PROB: [u8; 3] = [192, 128, 64]` constants
    transcribed verbatim from the §10 default-tables listing.
  * `Vp9CompressedHeader` extended with `tx_probs` and `skip_prob`
    fields exposing the post-sweep tables to callers.
    `parse_compressed_header` runs the sweeps in spec order:
    `read_tx_mode` → optional `read_tx_mode_probs` →
    `read_skip_prob`.
  * 10 new unit tests covering `DEFAULT_TX_PROBS` shape + value
    spot-check, `DEFAULT_SKIP_PROB` value, `read_tx_mode_probs`
    zero-buffer passthrough, the 12-cell sweep count, the row-0
    (`TX_4X4`) non-modification invariant, `read_skip_prob`
    zero-buffer passthrough across `SKIP_CONTEXTS = 3`, and three
    `parse_compressed_header` integration scenarios (ONLY_4X4
    non-lossless, lossless, and TX_MODE_SELECT). 2 new end-to-end
    integration tests in `tests/compressed_header.rs` driving the
    full uncompressed-header splice plus the new sweeps.
  * `read_diff_update_prob`, `decode_term_subexp`,
    `inv_remap_prob`, `inv_recenter_nonneg`, and `INV_MAP_TABLE`
    drop their round-4 `#[allow(dead_code)]` markers — they are
    now driven live by the §6.3.2 / §6.3.8 sweeps.

* **Round 4: §6.3.3 `diff_update_prob` chain (`decode_term_subexp` +
  `inv_remap_prob` + `inv_recenter_nonneg` + 255-entry
  `inv_map_table`).**
  * `read_diff_update_prob( coder, base_prob )` (§6.3.3) — reads the
    `B(252)` `update_prob` flag, on 1 pulls a `decode_term_subexp`
    value and remaps the previous probability through
    `inv_remap_prob`. On 0, passes the base probability straight
    through.
  * `decode_term_subexp( )` (§6.3.4) — the 5-leg
    `L(1) → L(4)` / `L(1) → L(4)+16` / `L(1) → L(5)+32` /
    `L(7) → +64` / `L(7), L(1) → (v<<1)-1+bit` cascade producing a
    value in `0..=254`.
  * `inv_remap_prob( delta_prob, prob )` (§6.3.5) — low-half /
    high-half piecewise remap calling `inv_recenter_nonneg`.
  * `inv_recenter_nonneg( v, m )` (§6.3.6) — pure arithmetic
    helper covering the `v > 2*m` short-circuit plus the
    odd / even split.
  * `INV_MAP_TABLE: [u8; 255]` — the §6.3.5 listing transcribed
    verbatim (a permutation of 1..=254 with a duplicated trailing
    253).
  * 16 new unit tests covering `inv_recenter_nonneg` (all three
    piecewise branches plus the `v == 2*m` boundary),
    `INV_MAP_TABLE` length + spot-checks, `inv_remap_prob` against
    both low-half / high-half branches and the `v > 2m`
    short-circuit, `decode_term_subexp` against hand-derived §9.2
    buffers (leg-1 zero plus a sweep that confirms the result
    stays in `0..=254`), and `read_diff_update_prob` confirming
    the `update_prob == 0` passthrough across the full
    1..=255 base-probability sweep.
  * The chain is structural; no caller in §6.3.2 / §6.3.7+ uses it
    yet, so each helper carries `#[allow(dead_code)]` until the
    next round wires them into the table sweeps.

* **Round 3: §9.2 Boolean decoder primitives + §6.3.1 `read_tx_mode`
  walk.**
  * New `src/bool_coder.rs` module implementing the four §9.2
    primitives: `init_bool( sz )` (§9.2.1) with the marker-bit
    zero-conformance check, `read_bool( p )` (§9.2.2) with the
    `split = 1 + (((BoolRange-1) * p) >> 8)` narrow and the
    `range < 128` renormalisation refill, `read_literal( n )`
    (§9.2.4) folded over `read_bool(128)`, and `exit_bool( )`
    (§9.2.3) consuming the remaining `BoolMaxBits` with a
    zero-pad conformance check. `BoolMaxBits` underflow raises
    `InvalidBitstream` instead of silently injecting 0.
  * New `src/compressed.rs` module with `Vp9CompressedHeader`,
    `TxMode` (5-variant enum mapping §3 `TX_MODES`), and the
    `parse_compressed_header(payload, lossless)` entry point.
    `read_tx_mode( )` (§6.3.1) short-circuits to `ONLY_4X4` when
    `Lossless == 1` (§6.2.9), otherwise reads `L(2)` and (for the
    `ALLOW_32X32` raw value) an extra `L(1)` `tx_mode_select` to
    distinguish `ALLOW_32X32` from `TX_MODE_SELECT`.
  * 9 `bool_coder` unit tests + 7 `compressed` unit tests + 4
    end-to-end integration tests (`tests/compressed_header.rs`)
    splicing a hand-derived §9.2 payload past
    `Vp9FrameHeader::uncompressed_header_size_bytes`. Every byte
    vector used was derived by stepping the §9.2 decoder, not
    borrowed from any third-party VP9 implementation.
  * The §6.3.2+ syntax (`tx_mode_probs`, `read_coef_probs`,
    `read_skip_prob`, `read_inter_mode_probs`,
    `read_interp_filter_probs`, …) all flow through the §6.3.3
    `diff_update_prob` chain (`decode_term_subexp` (§6.3.4) +
    `inv_remap_prob` (§6.3.5) + `inv_recenter_nonneg` (§6.3.6) +
    the 255-entry `inv_map_table` constant) and have been
    deferred to the next round so this drop lands a verifiable
    Boolean-coder primitive plus the §6.3.1 walk in isolation.

* **Round 2: full §6.2 uncompressed-header walk.** Extends the round-1
  walker through the end of `uncompressed_header()` and the §6.1.1
  `trailing_bits()` zero-fill alignment:
  * `s(n)` signed-integer reader per spec §4.9.2 plus
    `BitReader::trailing_bits()` zero-pad consumer with §7.1.1
    zero-bit conformance check (`src/bitreader.rs`).
  * `read_loop_filter_params()` (§6.2.8) with full `delta_enabled` /
    `delta_update` / per-ref / per-mode `s(6)` delta walk.
  * `read_quantization_params()` (§6.2.9) with `read_delta_q()`
    (§6.2.10) for `delta_q_y_dc` / `_uv_dc` / `_uv_ac` and the
    `Lossless` derivation.
  * `read_segmentation_params()` (§6.2.11) with `read_prob()`
    (§6.2.12), the 7 `tree_probs`, the 3 `pred_prob` (with
    `temporal_update` switching between f(0)-implicit-255 and
    `read_prob()`), and the per-segment / per-feature
    `feature_enabled` / `feature_value` / optional `feature_sign`
    loop driven by `segmentation_feature_bits` /
    `segmentation_feature_signed`.
  * `read_tile_info()` (§6.2.13) with `Sb64Cols` computed from
    `FrameWidth` via §6.2.6 (`MiCols = (W+7)>>3`,
    `Sb64Cols = (MiCols+7)>>3`) and `calc_min_log2_tile_cols` /
    `calc_max_log2_tile_cols` per §6.2.14
    (`MIN_TILE_WIDTH_B64 = 4`, `MAX_TILE_WIDTH_B64 = 64`). The
    §7.2.11 `tile_cols_log2 <= 6` conformance constraint is checked.
  * f(16) `header_size_in_bytes`, then `trailing_bits()` consumed so
    `uncompressed_header_size_bytes` exposes the byte-aligned offset
    at which the §6.3 compressed header starts.
  * `refresh_frame_context`, `frame_parallel_decoding_mode`,
    `frame_context_idx` (with the `FrameIsIntra ||
    error_resilient_mode` reset to 0 per §6.2 `setup_past_independence`),
    `reset_frame_context`, `refresh_frame_flags` (0xFF for key
    frames per spec).
  * New public types: `LoopFilterParams`, `QuantizationParams`,
    `SegmentationParams`, `TileInfo`, plus constants `MAX_SEGMENTS`,
    `SEG_LVL_MAX`, `SEGMENTATION_FEATURE_BITS`,
    `SEGMENTATION_FEATURE_SIGNED`. `Vp9FrameHeader` extended with all
    the new fields plus `uncompressed_header_size_bytes`.
  * 6 additional bit-reader tests (`s(n)` round-trip,
    `trailing_bits` accept/reject/no-op) + 2 `tile_info`
    arithmetic tests + 4 additional integration tests
    (loop_filter delta-update, segmentation full update,
    `tile_info` 4K increment walk, nonzero trailing bit rejection).
    Existing integration tests extended with the new tail.

* **Round 1: uncompressed-header walker.** A clean-room implementation
  of the structural portion of VP9 spec v0.7 §6.2 / §7.2:
  * MSB-first `f(n)` bit reader (`src/bitreader.rs`) per spec §9.1.
  * `parse_uncompressed_header()` walking `frame_marker`, `Profile`
    (including the `profile == 3` reserved bit), `show_existing_frame`
    early-return with `frame_to_show_map_idx`, `frame_type`,
    `show_frame`, `error_resilient_mode`, `frame_sync_code`,
    `color_config()` (with full §7.2.2 constraint checks including
    CS_RGB-on-profile-0/2 rejection and `reserved_zero` enforcement),
    `frame_size()` and `render_size()`.
  * Public `Vp9FrameHeader`, `ColorConfig`, `ColorSpace`, `FrameType`
    types in the crate root.
  * `Error::UnexpectedEof`, `Error::InvalidBitstream`,
    `Error::Unsupported` variants in addition to the existing
    `NotImplemented`.
  * 3 internal bit-reader tests plus 8 integration tests
    (`tests/uncompressed_header.rs`) covering all four profiles,
    studio/full-swing color ranges, render-size overrides, the
    `show_existing_frame` early return, the intra-only inter-frame
    branch, and three failure paths (bad `frame_marker`, bad
    `frame_sync_code`, truncated buffer).

  `decode_vp9()` / `encode_vp9()` continue to return
  `Error::NotImplemented`; their full pipelines land in later rounds.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule (no external library source as
  reference, not even as a sanity check). Per the workspace's
  Implementer-Round procedure, such audit failures are unrecoverable
  via incremental cleanup and require an orphan rebuild.

  Every public API path returned `Error::NotImplemented`. A
  clean-room re-implementation against the VP9 Bitstream & Decoding
  Process Specification (v0.7) has now begun (see "Added" above).

  No `old` branch is retained; long-standing audit failures forfeit
  the archive per workspace policy.
