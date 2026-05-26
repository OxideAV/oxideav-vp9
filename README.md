# oxideav-vp9

Pure-Rust VP9 codec — clean-room re-implementation against the VP9
Bitstream & Decoding Process Specification v0.7.

## Status — 2026-05-27 (round 24)

**Round 24: §6.3.14 `read_y_mode_probs( )` compressed-header sweep.**
Round 24 extends the §6.3 inter-arm primitives chain by one cell
alongside the round-22 §6.3.11 `read_is_inter_probs( )` and round-23
§6.3.9 / §6.3.10 sweeps:

* `read_y_mode_probs( coder, y_mode_probs )` per §6.3.14
  (`vp9-spec.txt` lines 2220-2225). A `BLOCK_SIZE_GROUPS = 4` (§3 line
  460) × `INTRA_MODES - 1 = 9` (§3 line 505) = 36-cell row-major
  sweep of `read_diff_update_prob` calls — one `B(252)` `update_prob`
  flag per cell and, on 1, a `decode_term_subexp( )` +
  `inv_remap_prob( )` cascade — updating `y_mode_probs[ ][ ]` in
  place.
* `DEFAULT_Y_MODE_PROBS` (`mode_info`, transcribed verbatim from §9.3 /
  §10.5) carries the inter-frame `intra_mode` initial / reset
  probabilities (row annotations preserved: 0 = block_size < 8x8,
  1 = < 16x16, 2 = < 32x32, 3 = >= 32x32). The same constant feeds the
  (still-deferred) §7.4.5 intra-mode tree decoder of
  `inter_block_mode_info( )`.
* `DEFAULT_Y_MODE_PROBS_TABLE` re-export in `compressed.rs` keeps
  `mode_info` as the single source of truth (mirroring the round-22
  `DEFAULT_IS_INTER_PROB_TABLE` and round-23 inter-mode / interp-filter
  staging patterns).

Validation (+8 tests, lib total 345 → 353; suite total 365 → 373)
covers the §3 constant pinning for `BLOCK_SIZE_GROUPS = 4` and
`INTRA_MODES = 10`; the verbatim transcription of the §9.3
`default_y_mode_probs` table; the zero-buffer `update_prob = 0`
pass-through preserving the starting table; an all-cells-visited
check with a non-uniform custom starting table; a cursor-equivalence
proof that the sweep consumes exactly 36 `B(252)` flags; explicit
row-major walk equivalence against a parallel-coder reference (two
starting tables); and a single-source-of-truth check that the
`compressed.rs` re-export equals the `mode_info` constant.

Out of scope for round 24: the §6.3 outer-dispatch `FrameIsIntra == 0`
arm itself — the y-mode sweep lives alongside `read_is_inter_probs( )`
+ `read_inter_mode_probs( )` + `read_interp_filter_probs( )` between
`read_skip_prob( )` (§6.3.8) and the (still-deferred) §6.3.15
`read_partition_probs( )` / §6.3.16 `mv_probs( )` cells. Wiring any
subset into `parse_compressed_header` before §6.3.12 / §6.3.13
(`frame_reference_mode( )` + `frame_reference_mode_probs( )`) land
would mis-position the coder cursor — and §6.3.12 needs
`ref_frame_sign_bias[ ]` state the uncompressed-header walker still
rejects with `Error::Unsupported`. The round-24 surface stays
internal-only (`pub(crate)` with `#[allow(dead_code)]` until the
outer dispatch grows the inter arm); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-26 (round 23)

**Round 23: §6.3.9 `read_inter_mode_probs( )` + §6.3.10
`read_interp_filter_probs( )` compressed-header sweeps.** Round 23
extends the §6.3 inter-arm primitives chain by two cells alongside
the round-22 §6.3.11 `read_is_inter_probs( )`:

* `read_inter_mode_probs( coder, inter_mode_probs )` per §6.3.9
  (`vp9-spec.txt` lines 2138-2143). An `INTER_MODE_CONTEXTS = 7` (§3
  line 507) × `INTER_MODES - 1 = 3` (§3 line 506) = 21-cell
  row-major sweep of `read_diff_update_prob` calls.
* `read_interp_filter_probs( coder, interp_filter_probs )` per
  §6.3.10 (`vp9-spec.txt` lines 2146-2151). An
  `INTERP_FILTER_CONTEXTS = 4` (§3 line 495) × `SWITCHABLE_FILTERS - 1
  = 2` (§3 line 487) = 8-cell row-major sweep. The spec swaps the
  loop-index names (outer `j`, inner `i`) — the visit order still
  matches the array layout `[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`.
* `DEFAULT_INTER_MODE_PROBS` (`mode_info`, transcribed verbatim from
  §10.5 lines 7758-7766) and `DEFAULT_INTERP_FILTER_PROBS`
  (`mode_info`, transcribed verbatim from §10.5 lines 7769-7775)
  carry the spec's annotated initial / reset values. The same
  constants will feed the (still-deferred) §6.4.16
  `inter_block_mode_info( )` per-block reader once that lands.
* `DEFAULT_INTER_MODE_PROBS_TABLE` and `DEFAULT_INTERP_FILTER_PROBS_TABLE`
  re-exports in `compressed.rs` keep `mode_info` as the single source
  of truth (mirroring the round-22 `DEFAULT_IS_INTER_PROB_TABLE`
  staging pattern).

Validation (+16 tests, lib total 329 → 345; suite total 349 → 365)
covers the §3 constant pinning for `INTER_MODE_CONTEXTS = 7`,
`INTER_MODES = 4`, `INTERP_FILTER_CONTEXTS = 4`, `SWITCHABLE_FILTERS = 3`;
the verbatim transcription of each §10.5 default table; the
zero-buffer `update_prob = 0` pass-through preserving each starting
table; an all-cells-visited check with non-uniform starts (custom
tables preserved across the sweep); cursor-equivalence proofs that
each sweep consumes exactly its prescribed cell count of `B(252)`
flags (21 for inter-mode, 8 for interp-filter); explicit row-major
walk equivalence against a parallel-coder reference; and
single-source-of-truth checks that the `compressed.rs` re-exports
equal the `mode_info` constants.

Out of scope for round 23: the §6.3 outer-dispatch `FrameIsIntra == 0`
arm itself — the inter-mode / interp-filter sweeps live alongside
`read_is_inter_probs( )` between `read_skip_prob( )` (§6.3.8) and
`frame_reference_mode( )` (§6.3.12). Wiring any subset into
`parse_compressed_header` before §6.3.12 / §6.3.13 land would
mis-position the coder cursor. The round-23 surface stays
internal-only (`pub(crate)` with `#[allow(dead_code)]` until the
outer dispatch grows the inter arm); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-26 (round 22)

**Round 22: §6.3.11 `read_is_inter_probs( )` compressed-header sweep.**
Round 22 lands the unconditional `IS_INTER_CONTEXTS = 4`
`diff_update_prob` walk that populates the running `is_inter_prob[ ]`
table the round-21 §6.4.13 `read_is_inter( )` per-block decoder
consumes via the §9.3.2 ctx:

* `read_is_inter_probs( coder, is_inter_prob )` per §6.3.11
  (`vp9-spec.txt` lines 2154-2167). Four sequential
  `read_diff_update_prob` calls — one `B(252)` `update_prob` flag per
  slot and, on 1, a `decode_term_subexp` + `inv_remap_prob` cascade —
  updating `is_inter_prob[0..IS_INTER_CONTEXTS]` in place.
* `DEFAULT_IS_INTER_PROB_TABLE` re-export of the round-21
  `mode_info::DEFAULT_IS_INTER_PROB = {9, 102, 187, 225}` (§10.5)
  initial / reset table — same constant feeds both the §6.4.13
  per-block decoder and the §6.3.11 compressed-header sweep so there's
  a single source of truth.

Validation covers the §10.5 default re-export matching the
`mode_info` source-of-truth constant, the zero-buffer
`update_prob = 0` path passing every cell through unchanged, the
four-context cell-count visiting (custom probabilities preserved
across the sweep), an explicit equivalence test that the sweep is
identical to four sequential `read_diff_update_prob` calls against a
parallel coder (proves cursor advancement matches and the order of
slots is preserved), the §3 `IS_INTER_CONTEXTS = 4` constant pinning,
a tuple-sweep exhaustive round-trip across distinct starting
probabilities, and an independent cursor-equivalence check confirming
the function consumes exactly four `B(252)` flags on the zero buffer.

Out of scope for round 22: the §6.3 outer-dispatch `FrameIsIntra == 0`
arm itself — `read_is_inter_probs( )` lives *between* `read_skip_prob`
(§6.3.8) and `frame_reference_mode( )` (§6.3.12) inside the gated
inter branch alongside `read_inter_mode_probs( )` (§6.3.9) and
`read_interp_filter_probs( )` (§6.3.10); wiring it into
`parse_compressed_header` ahead of those companions would mis-position
the coder cursor. Each of those primitives lands in its own round; the
outer dispatch grows the inter arm only when §6.3.9 / §6.3.10 are also
in place. The round-22 surface stays internal-only (`pub(crate)`); the
public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-26 (round 21)

**Round 21: §6.4.13 `read_is_inter( )` + §9.3.2 `is_inter` context +
§10.5 `default_is_inter_prob`.** Round 21 lands the per-block
inter/intra reader the §6.4.11 `inter_frame_mode_info( )` driver fires
between `read_skip( )` and `read_tx_size( !skip || !is_inter )`:

* §3 constants `SEG_LVL_REF_FRAME = 2` and `IS_INTER_CONTEXTS = 4`
  transcribed verbatim.
* §10.5 `default_is_inter_prob[IS_INTER_CONTEXTS] = {9, 102, 187, 225}`
  transcribed verbatim (the initial / reset table the §6.3.10
  `read_is_inter_probs( )` compressed-header sweep updates with
  `diff_update_prob` deltas — that sweep lands in a separate round
  alongside the rest of §6.3.9..§6.3.16).
* `IsInterNeighbours { above: Option<i32>, left: Option<i32> }` — the
  §6.4.11 above/left `RefFrames[ ][ ][ 0 ]` view: `Some( rf )` carries
  the neighbour's `ref_frame[0]` (`INTRA_FRAME` / `LAST_FRAME` /
  `GOLDEN_FRAME` / `ALTREF_FRAME` / `NONE`), `None` encodes
  `!AvailU` / `!AvailL` (the §6.4.11 listing forces the neighbour to
  `INTRA_FRAME` when unavailable, so the §9.3.2 `*Intra` rule resolves
  the same way).
* `is_inter_context( nb )` (§9.3.2) — the four-branch ctx derivation:
  - both available, both intra → `3`
  - both available, exactly one intra → `1` (`true || false = 1`)
  - both available, neither intra → `0`
  - only one available, that one intra → `2`
  - only one available, that one inter → `0`
  - neither available → `0`
  Returns one of `0..=3` indexing `is_inter_prob[ ctx ]`.
* `read_is_inter( coder, seg_feature_ref_frame_active,
  segment_ref_frame_data, is_inter_prob, nb )` (§6.4.13) — two paths:
  - `seg_feature_active( SEG_LVL_REF_FRAME )` → `is_inter =
    FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME`
    without consuming any coder bits.
  - otherwise → one §9.3.3 `BINARY_TREE` bit under
    `is_inter_prob[ is_inter_context( nb ) ]`.

Validation covers the §10.5 / §3 constants, every branch of the
§9.3.2 ctx derivation (including the `NONE = -1` ref-frame sentinel
which satisfies `<= INTRA_FRAME` and is treated as intra-side), the
§6.4.13 seg-feature path for `INTRA_FRAME` (→ false) and each of
`LAST_FRAME` / `GOLDEN_FRAME` / `ALTREF_FRAME` overrides (→ true), the
zero-coder bit=0 path, the bias-coder bit=1 path with both-intra
neighbours, an `is_inter_prob[ ctx ]` indexing sweep across all four
ctxs that confirms no panic / out-of-range, and a `seg_feature_active`
short-circuit test that ignores both neighbours and the coder.

Out of scope for round 21: the §6.4.11 `inter_frame_mode_info( )`
orchestrator itself (still composes this with `inter_segment_id( )` /
`read_skip( )` / `read_tx_size( )` / `inter_block_mode_info( )` or
`intra_block_mode_info( )`); the §6.3.10 `read_is_inter_probs( )`
compressed-header sweep (lands with the rest of §6.3.9..§6.3.16);
`inter_block_mode_info( )` (§6.4.16 — blocked on reference-buffer
state and MV decode); the §8.4 `counts_is_inter` probability-adaption
accumulator (§9.3.4 bookkeeping for end-of-frame adaption). The
round-21 surface stays internal-only (`pub(crate)`); the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

## Status — 2026-05-26 (round 20)

**Round 20: §6.4.12 `inter_segment_id( )` + §6.4.14 `get_segment_id( )`
+ §7.4 `AboveSegPredContext` / `LeftSegPredContext` strips.** Round 20
lands the inter-frame companion to the round-16 `intra_segment_id`
primitive — the per-block segment-id reader the §6.4.11
`inter_frame_mode_info( )` driver fires before `read_skip( )` /
`read_is_inter( )` / `read_tx_size( )`:

* §10.2 `num_8x8_blocks_high_lookup[ BLOCK_SIZES ]` =
  `{1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8}` transcribed verbatim
  (alongside the existing `num_8x8_blocks_wide_lookup`), keying both
  the §6.4.14 `bh` clamp and the §6.4.12 `LeftSegPredContext[ ]`
  write-back length.
* `PrevSegmentIds<'a>` — a borrowed row-major `MiRows × MiCols` view
  of the previous frame's segment-id plane (the §6.4.4 `SegmentIds[ ][
  ]` write-back).
* `get_segment_id( prev, mi_row, mi_col, mi_size )` (§6.4.14) — the
  `bw` / `bh` clamp via `Min( MiCols - MiCol, bw )` /
  `Min( MiRows - MiRow, bh )` and the `seg = 7; seg = Min( seg,
  PrevSegmentIds[ … ] )` spatial-minimum sweep.
* `SegPredContextState { above[MiCols], left[MiRows] }` — the §7.4.1 /
  §7.4.2 strip storage with `new( )` zero-init,
  `clear_left( )` per-superblock-row reset, and `above( ) ` /
  `left( )` ctx accessors.
* `read_seg_id_predicted( coder, pred_prob, seg_pred_ctx, mi_row,
  mi_col )` — the §9.3.2 `ctx = LeftSegPredContext[ MiRow ] +
  AboveSegPredContext[ MiCol ]` derivation and §9.3.1 `BINARY_TREE`
  one-bit decode under `segmentation_pred_prob[ ctx ]`.
* `inter_segment_id( )` (§6.4.12) — the four-path orchestrator:
  (1) `!segmentation_enabled` → 0; (2) enabled but `!update_map` →
  `predictedSegmentId`; (3) `update_map && !temporal_update` →
  `read_segment_id` (the round-16 §9.3.1 walk); (4) `update_map &&
  temporal_update` → `read_seg_id_predicted` then either the
  predictor or a fresh `read_segment_id`, followed by the spec's
  trailing write-back of `seg_id_predicted` into
  `AboveSegPredContext[ MiCol + i ]` (`i ∈ 0..bw`) and
  `LeftSegPredContext[ MiRow + i ]` (`i ∈ 0..bh`).

Validation covers `get_segment_id` (interior 2x2 min, partial-edge
clamp via `Min( MiCols - MiCol, bw )`, all-7 fallback), the §7.4
zero-init contract (and `clear_left` not touching `Above`), the §9.3.2
`Left + Above` ctx wiring of `read_seg_id_predicted`, each of the
four §6.4.12 paths independently, the `Error::InvalidBitstream`
surface when `tree_probs` (paths 3 / 4-not-predicted) or `pred_prob`
(path 4) are missing, and the §6.4.12 trailing write-back clamping on
a partial-edge `BLOCK_32X32` at `(1, 1)` of a 3-wide frame.

Out of scope for round 20: the §6.4.11 `inter_frame_mode_info( )`
orchestrator itself (composes this primitive with `read_skip( )` /
`read_is_inter( )` / `read_tx_size( )` / `inter_block_mode_info( )`
or `intra_block_mode_info( )`); `read_is_inter( )` (§6.4.13 — needs
the §9.3.2 `is_inter` ctx and the `is_inter_prob[ 4 ]` compressed-
header table); `inter_block_mode_info( )` (§6.4.16 — blocked on
reference-buffer state and MV decode); the `PrevSegmentIds[ ][ ]` /
`SegmentIds[ ][ ]` frame-wide write-back (left to the §6.4.4 driver);
and the §8.4 `counts_*` probability-adaption accumulators (§9.3.4
bookkeeping for end-of-frame adaption). The round-20 surface stays
internal-only (`pub(crate)`); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-26

**Round 19: §6.4.3 recursive `decode_partition()` driver (extending the
`partition` module).** Round 19 composes the round-18
`decode_partition_type()` primitive into the full §6.4.3 recursive
partition driver:

* `decode_partition(coder, r, c, bsize, mi_rows, mi_cols, ctx_state,
  probs_kind, leaves)` — walks the §6.4.3 listing line-for-line: the
  `(r >= MiRows || c >= MiCols)` quadrant short-circuit, the `num8x8`
  / `halfBlock8x8` / `hasRows` / `hasCols` derivation, the
  `partition` decode via the round-18 primitive (with the §9.3.2
  `partition_plane_context` ctx + the per-frame probability source),
  the four-way dispatch on the decoded `PARTITION_*` value (with
  HORZ second-leaf gated by `hasRows`, VERT second-leaf gated by
  `hasCols`, and SPLIT recursing in spec order TL → TR → BL → BR),
  and the §6.4.3 tail write-back into the partition-context strips
  (gated by `bsize == BLOCK_8X8 || partition != PARTITION_SPLIT`,
  writing `15 >> b_*_log2_lookup[subsize]` per cell).
* `PartitionContextState` — the `AbovePartitionContext[]` /
  `LeftPartitionContext[]` strips (sized `Sb64Cols * 8` /
  `Sb64Rows * 8` per the §7.4 listing). Exposes `new(mi_cols, mi_rows)`
  with the §7.4 zero-reset and `clear_left()` for the §6.4.2
  per-superblock-row reset.
* `PartitionProbsKind` — the per-frame probability source enum
  (`Keyframe` → indexes the §10.4 `KF_PARTITION_PROBS` directly;
  `Inter(&table)` → indexes the caller's running 16 × 3 table,
  typically initialised from §10.5 `DEFAULT_PARTITION_PROBS` and
  conditionally updated by the §6.3 `read_partition_probs()` sweep —
  still pending in a later round).
* `LeafBlock { r, c, subsize }` log records — emitted in §6.4.3
  traversal order in lieu of the §6.4.4 `decode_block(r, c, subsize)`
  call site (the per-block `mode_info` / `residual` decode is
  downstream and not yet wired into this driver).

Validation includes three hand-built bitstreams produced by a
test-only minimal range encoder that mirrors the §9.2.2 decode steps
inverse-by-inverse:

* a single 64x64 superblock with `PARTITION_NONE` → one leaf
  `{ 0, 0, BLOCK_64X64 }` and the §6.4.3 tail `15 >> 4 = 0`
  write-back;
* a single 64x64 superblock with `PARTITION_SPLIT` then four 32x32
  `PARTITION_NONE` children → four leaves in TL → TR → BL → BR order,
  with the §6.4.3 tail write-back firing on each child (`15 >> 3 = 1`)
  but not on the parent SPLIT;
* a 64x64 superblock split with mixed HORZ / VERT children (TL HORZ →
  2 leaves at BLOCK_32X16, TR VERT → 2 leaves at BLOCK_16X32, BL
  HORZ, BR VERT) exercising the HORZ second-leaf and VERT
  second-leaf §6.4.3 paths plus the §6.4.3 tail write-back across
  mixed partitions and the §9.3.2 ctx derivation across the
  successively-populated strip state.

Out of scope for round 19:

* The §6.3 `read_partition_probs()` compressed-header sweep
  (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
  `diff_update_prob` cells against `DEFAULT_PARTITION_PROBS`) — the
  driver consumes the `Inter` running table, but constructing it
  lands in a later round.
* The §6.4.4 `decode_block()` mode-info + residual decode that
  `LeafBlock` stands in for — wiring it into this driver is
  downstream of all the §6.4 mode-info readers landing first.
* The §6.4.2 `decode_tile()` outer loop (the `r += 8, c += 8`
  superblock walk + per-row `clear_left_context()`) — composes this
  driver but is a separate round.

The round-19 surface stays internal-only (`pub(crate)`); the public
API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-26 (round 18)

**Round 18: §6.4.3 `decode_partition_type()` per-call partition reader (new
`partition` module).** Round 18 lands the single-call partition decoder
the recursive §6.4.3 `decode_partition(r, c, bsize)` driver fires once per
quadrant inside a tile, plus every §3 / §10.2 / §10.4 / §10.5 surface it
consumes:

* §3 enumeration `PARTITION_NONE = 0`, `PARTITION_HORZ = 1`,
  `PARTITION_VERT = 2`, `PARTITION_SPLIT = 3` plus `PARTITION_TYPES = 4`
  and `PARTITION_CONTEXTS = 16`.
* §9.3.1 trees `PARTITION_TREE[6]`, `COLS_PARTITION_TREE[2]`,
  `ROWS_PARTITION_TREE[2]` transcribed verbatim from the spec listing.
* §10.2 lookups `B_WIDTH_LOG2_LOOKUP` / `B_HEIGHT_LOG2_LOOKUP` (the
  §6.4.3 tail `15 >> b_*_log2_lookup[subsize]` write-back inputs),
  `MI_WIDTH_LOG2_LOOKUP` (the §9.3.2 `bsl` input), and
  `NUM_8X8_BLOCKS_WIDE_LOOKUP` (the §6.4.3 `num8x8` input) — all
  transcribed verbatim.
* §10.2 `SUBSIZE_LOOKUP[4][13]` transcribed verbatim (with
  `BLOCK_INVALID = 14` for the HORZ / VERT / SPLIT entries with no
  legal child at non-square parents).
* §10.4 `KF_PARTITION_PROBS[16][3]` (keyframe / intra-only fixed table)
  and §10.5 `DEFAULT_PARTITION_PROBS[16][3]` (inter-frame initial
  table prior to §6.3 `read_partition_probs()`) transcribed verbatim
  with per-table shape + listing-anchor + §9.2 min-prob tests.
* `partition_plane_context(bsize, above_ctx, left_ctx)` — the §9.3.2
  `ctx = bsl * 4 + left * 2 + above` derivation: `bsl =
  mi_width_log2_lookup[bsize]`, `boffset = 3 - bsl`, OR-fold of the
  `AbovePartitionContext[]` / `LeftPartitionContext[]` strips across
  `num8x8` cells, extract the `bsl`-th bit. Covers `bsl ∈ {0, 1, 2, 3}`
  for the four superblock recursion sizes (`BLOCK_8X8` / `BLOCK_16X16`
  / `BLOCK_32X32` / `BLOCK_64X64`) with the full `ctx ∈ 0..=15`
  reached via the included exhaustive sweep.
* `decode_partition_type(coder, has_rows, has_cols, probs)` — the
  §6.4.3 reader proper. Dispatches on `(has_rows, has_cols)` per the
  §9.3.1 tree-selection rule: interior (`(true, true)`) walks the
  6-entry `PARTITION_TREE` with `node2 = node`; right-edge
  (`(false, true)`) walks 2-entry `COLS_PARTITION_TREE` with
  `node2 = 1`; bottom-edge (`(true, false)`) walks 2-entry
  `ROWS_PARTITION_TREE` with `node2 = 2`; corner (`(false, false)`)
  returns `PARTITION_SPLIT` directly without consuming any bool-coder
  bits. Returns one of the four `PARTITION_*` constants.

The recursive §6.4.3 driver itself — which threads
`SUBSIZE_LOOKUP[partition][bsize]` into four recursive calls when
`PARTITION_SPLIT` and writes back the `AbovePartitionContext[]` /
`LeftPartitionContext[]` strips with `15 >> b_*_log2_lookup[subsize]` —
lands in a later round. The §6.3 `read_partition_probs()`
compressed-header sweep (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) =
16 × 3 = 48` `diff_update_prob` cells against `DEFAULT_PARTITION_PROBS`)
also lands in a later round. The round-18 surface is internal-only;
the public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-25

**§6.4.15 `intra_block_mode_info()` inter-frame intra-block reader.**
The companion to the §6.4.6 `intra_frame_mode_info()` keyframe driver,
fired by the §6.4.11 `inter_frame_mode_info()` path when a block in a
non-keyframe frame is coded intra (`is_inter == 0`):

* §9.3.2 `size_group_lookup[BLOCK_SIZES]`
  (`{0,0,0,1,1,1,2,2,2,3,3,3,3}`) and §9.3
  `default_y_mode_probs[BLOCK_SIZE_GROUPS][INTRA_MODES - 1]` (4 × 9) /
  `default_uv_mode_probs[INTRA_MODES][INTRA_MODES - 1]` (10 × 9)
  transcribed verbatim — the compressed-header `y_mode_probs` /
  `uv_mode_probs` defaults, distinct from the §10.5 keyframe
  `kf_*_mode_probs`. Per-table shape + anchor + §9.2 min-prob tests.
* `intra_mode( coder, y_mode_probs, mi_size )` — §9.3.3 walk over
  `intra_mode_tree` with ctx `size_group_lookup[MiSize]`;
  `sub_intra_mode( coder, y_mode_probs )` — ctx fixed at 0;
  `uv_mode( coder, uv_mode_probs, y_mode )` — ctx `y_mode`. Each ctx
  derivation has an instrumented-callback test pinning the row reached,
  plus hand-traced bias-buffer cases (`intra_mode` / `uv_mode` →
  `D207_PRED`).
* `intra_block_mode_info()` (§6.4.15) → `Vp9IntraBlockModeInfo
  { ref_frame_0, ref_frame_1, y_mode, sub_modes[4], uv_mode }`.
  `ref_frame[0] = INTRA_FRAME`, `ref_frame[1] = NONE`; the
  `MiSize >= BLOCK_8X8` arm decodes one `intra_mode` replicated across
  `sub_modes[]`, the sub-8x8 arm walks the `(idy, idx)` grid decoding
  one `sub_intra_mode` per cell (`y_mode` = last). Unlike §6.4.6 it
  reads **only** modes — `segment_id` / `skip` / `tx_size` are decoded
  by the §6.4.11 driver beforehand. A per-block bias-buffer scenario
  pins the contiguous `intra_mode → uv_mode` decode
  (`D207_PRED` then `D153_PRED`).
* §6.4.5 `mode_info()` dispatch shape: a `Vp9ModeInfo` enum with
  `IntraFrame(Vp9IntraMiBlock)` (the `FrameIsIntra` /
  `intra_frame_mode_info()` path) and
  `InterFrameIntraBlock(Vp9IntraBlockModeInfo)` (the `!FrameIsIntra`,
  `is_inter == 0` / `intra_block_mode_info()` sub-path), plus
  `inter_frame_intra_block_mode_info()` wiring the latter. The
  surrounding §6.4.11 prelude (`inter_segment_id` / `read_is_inter` /
  the `inter_block_mode_info()` arm) lands when its
  reference-buffer-dependent primitives do.

**Round 17: §6.4.6 `intra_frame_mode_info()` keyframe driver.** Round
17 wires the rounds 15 / 16 primitives into the §6.4.6 per-block
mode-info reader for keyframe (and intra-only) frames — the spec's
top-level entry point for an intra MI block:

* §9.3.1 `intra_mode_tree[18]` — the 18-entry / 10-leaf tree shared by
  `default_intra_mode` / `default_uv_mode` / `intra_mode` /
  `sub_intra_mode` / `uv_mode`. Transcribed verbatim
  (`{ -DC_PRED, 2, -TM_PRED, 4, -V_PRED, 6, 8, 12, -H_PRED, 10,
  -D135_PRED, -D117_PRED, -D45_PRED, 14, -D63_PRED, 16, -D153_PRED,
  -D207_PRED }`); the all-bit-0 walk lands on the `-DC_PRED` leaf
  (= 0).
* §10.5 `kf_y_mode_probs[INTRA_MODES][INTRA_MODES][INTRA_MODES - 1]`
  (a 10 × 10 × 9 = 900-byte table indexed by `[abovemode][leftmode]
  [node]` per the §9.3.2 `default_intra_mode` listing) transcribed
  verbatim from the §10.5 listing. Anchor checks pin five rows
  including `[dc][dc]`, `[dc][tm]`, `[h][h]`, and the last row
  `[tm][tm]`.
* §10.5 `kf_uv_mode_probs[INTRA_MODES][INTRA_MODES - 1]` (10 × 9 = 90
  bytes, indexed by `[y_mode][node]` per the §9.3.2 `default_uv_mode`
  listing) transcribed verbatim. Anchor checks pin the `[dc]`, `[h]`
  and `[tm]` rows.
* `default_intra_mode( coder, abovemode, leftmode )` — the §9.3.3
  walk over `intra_mode_tree` with `kf_y_mode_probs[above][left][node]`
  per row. Hand-traced bias-buffer test pins
  `default_intra_mode( DC_PRED, DC_PRED ) -> D207_PRED` (right-branch
  on every node, terminating at the §9.3.1 `-D207_PRED` leaf).
* `default_uv_mode( coder, y_mode )` — the §9.3.3 walk with
  `kf_uv_mode_probs[y_mode][node]`. Same hand-traced bias-buffer test
  pins `default_uv_mode( DC_PRED ) -> D207_PRED`.
* `intra_frame_mode_info()` (§6.4.6) — the orchestrator threading
  `intra_segment_id( )` (round 16) + `read_skip( )` (round 15) +
  `read_tx_size( 1 )` (round 15) + `default_intra_mode` +
  `default_uv_mode` into a `Vp9IntraMiBlock { segment_id, skip,
  tx_size, ref_frame_0, ref_frame_1, is_inter, y_mode, sub_modes[4],
  uv_mode }`. `ref_frame[0] = INTRA_FRAME = 0`, `ref_frame[1] = NONE
  = -1`, `is_inter = false` are hardwired per the §6.4.6 listing
  (NONE = -1 derived from `isCompound = ref_frame[1] > NONE` plus
  `ref_frame[1] > INTRA_FRAME = 0` for compound; the unique integer
  strictly below INTRA_FRAME). The `MiSize >= BLOCK_8X8` arm decodes
  one `default_intra_mode` and replicates it into all four
  `sub_modes[ ]` cells; the `MiSize < BLOCK_8X8` arm walks the §6.4.6
  `(idy, idx)` grid stepped by `num_4x4_blocks_high_lookup[MiSize]` /
  `num_4x4_blocks_wide_lookup[MiSize]` — 4 reads for BLOCK_4X4, 2 for
  BLOCK_4X8 / BLOCK_8X4 — with each cell receiving its own decoded
  mode replicated across the (num4x4h × num4x4w) `sub_modes[ ]`
  sub-grid, and `y_mode` set to the *last* decoded
  `default_intra_mode` per the spec listing.
* `IntraFrameNeighbours { avail_u, avail_l, above_sub_modes_23[2],
  left_sub_modes_13[2] }` — the per-MI-block neighbour bundle a tile
  driver builds from its frame-wide `SubModes[ ][ ][ ]` array. The
  §9.3.2 listing reads only positions {2, 3} of the above neighbour's
  `sub_modes[ ]` and positions {1, 3} of the left neighbour's
  `sub_modes[ ]`, so the bundle only carries those two cells per
  side; `DC_PRED` is substituted when `avail_u` / `avail_l` is false
  per the §9.3.2 fallback.

Out of scope for round 17 (the §6.4.15 `intra_block_mode_info` reader
itself landed subsequently — see the section above):
inter-frame mode info (§6.4.11+, blocked on reference-buffer state);
the §8.4 `counts_intra_mode` / `counts_uv_mode` probability-adaption
accumulators (§9.3.4 bookkeeping); and the `SubModes[ ][ ][ ]` /
`YModes[ ][ ]` frame-wide write-back the next MI block consumes from
the just-decoded `Vp9IntraMiBlock` (left to the §6.4.4 driver). The
round-17 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 16: §6.4.7 `intra_segment_id` + §9.3.1 `segment_tree`.** Round
16 extends the round-15 `mode_info` module (crate-internal,
`pub(crate)`) with the next slice of the §6.4.6
`intra_frame_mode_info()` orchestrator's primitives:

* §9.3.1 `segment_tree[14]` —
  `{ 2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7 }` transcribed
  verbatim. A 7-leaf binary tree mapping to segment ids `0..=7`; note
  the §9.3.1 packing means the all-bit-0 walk visits node indices
  `{0, 1, 3}` (not the contiguous `0..3` a regular binary tree would).
* `read_segment_id( coder, tree_probs )` — the §9.3.3 `tree_decode`
  walk over `SEGMENT_TREE` with per-node probability
  `segmentation_tree_probs[node]` per the §9.3.2 listing's
  `segment_id` entry. Returns the decoded segment id directly (the
  §9.3.3 post-loop `-n` already produces it).
* `intra_segment_id( coder, segmentation_enabled,
  segmentation_update_map, tree_probs )` (§6.4.7) — the
  `segmentation_enabled && segmentation_update_map` gate around
  `read_segment_id`. The intra-only path has no
  `segmentation_temporal_update` / `seg_id_predicted` machinery; when
  the gate fails `segment_id = 0` per the spec listing.

Out of scope for round 16: the §6.4.12 `inter_segment_id( )` syntax
(with the `predictedSegmentId = get_segment_id( )` spatial-prediction
helper + the `seg_id_predicted` binary decode + the
`AboveSegPredContext` / `LeftSegPredContext` write-back) — that's an
inter-frame primitive blocked on the reference-buffer state the
round-2 header walker still rejects with `Error::Unsupported`. The
§6.4.15 `intra_block_mode_info` (`default_intra_mode` /
`default_uv_mode` decode against the §10.5 `kf_y_mode_probs` /
`kf_uv_mode_probs` 3D / 2D tables) and the §6.4.6
`intra_frame_mode_info()` orchestrator that composes
`intra_segment_id` + `read_skip` + `read_tx_size` +
`intra_block_mode_info` into a single `Vp9IntraMiBlock` are deferred
to the next round. The round-16 surface is internal-only; the public
API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

**Round 15: §6.4.8 `read_skip` + §6.4.10 `read_tx_size` + §9.3.3
`tree_decode`.** Round 15 adds a `mode_info` module (crate-internal,
`pub(crate)`) implementing the first slice of the §6.4 per-block
mode-info decode that the round-14 `residual_intra` driver currently
consumes via a caller-supplied bundle — the building blocks the §6.4.6
`intra_frame_mode_info()` orchestrator will compose:

* §9.3.3 `tree_decode( coder, tree, prob )` — the generic
  `do { n = T[n + read_bool(P(n >> 1))] } while (n > 0)` walker that
  every tree-coded syntax element (skip, tx_size, intra_mode, …)
  routes through; the probability callback is a `FnMut(usize) -> u8`
  so call-sites splice the right §9.3.2 row in without the helper
  needing to know which syntax element it's decoding.
* §9.3.1 trees `tx_size_8_tree[2]`, `tx_size_16_tree[4]`,
  `tx_size_32_tree[6]`, and `binary_tree[2]` transcribed verbatim
  from the spec listing.
* §9.3.2 `skip_context` (the `Skips[MiRow-1][MiCol] +
  Skips[MiRow][MiCol-1]` derivation modulated by `AvailU` / `AvailL`)
  and `tx_size_context` (the `(above + left) > maxTxSize` rule that
  consults neighbour `TxSizes[ ]` only on unskipped MI blocks, and
  mirrors the side when a neighbour is unavailable).
* §6.4.8 `read_skip` — the `seg_feature_active(SEG_LVL_SKIP)`
  early-return rule plus the §9.3.2 binary-tree decode under
  `skip_prob[skip_context(nb)]`.
* §6.4.10 `read_tx_size` — the `allow_select && tx_mode ==
  TX_MODE_SELECT && MiSize >= BLOCK_8X8` path that walks the §9.3.1
  tree picked by `max_txsize_lookup[MiSize]`, indexed by the §9.3.2
  ctx; falls through to `Min(maxTxSize,
  tx_mode_to_biggest_tx_size[tx_mode])` per the spec's `else` branch.
* `NeighbourSkips` / `NeighbourTxSizes` — per-MI-block neighbour-state
  bundles a tile driver builds from its frame-wide `Skips[ ][ ]` /
  `TxSizes[ ][ ]` arrays.

Out of scope for round 15: the §6.4.6 `intra_frame_mode_info()`
orchestrator (which composes `read_skip` + `read_tx_size` + the
deferred §6.4.7 `intra_segment_id` + §6.4.15 `intra_block_mode_info`
into a single MI block); the `Skips[ ][ ]` / `TxSizes[ ][ ]`
frame-wide write-back (left to the §6.4.6 driver); inter-frame mode
info (§6.4.11+, needs reference-buffer state); and the §8.4
`counts_skip` / `counts_tx_size` probability-adaption accumulators
(§9.3.4 bookkeeping for the end-of-frame adaption round). The
round-15 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 14: §6.4.21 `residual( )` intra driver.** Round 14 adds a
`residual` module (crate-internal, `pub(crate)`) implementing the §6.4.21
`residual( )` outer loop for the **intra** path — the per-plane,
per-4x4-sub-block walk that owns the `AboveNonzeroContext` /
`LeftNonzeroContext` write-back across a whole MI block, drives the
round-13 §6.4.24 `tokens( )` per-block decode, and feeds the round-11
§8.6.2 `reconstruct_block` with real per-block `Tokens` arrays,
availability flags and plane/quantizer state:

* The §10.2 `num_4x4_blocks_wide_lookup` / `num_4x4_blocks_high_lookup`
  / `max_txsize_lookup` tables and the §6.4.23 `ss_size_lookup[ 13 ][ 2
  ][ 2 ]` table transcribed verbatim, alongside the `BLOCK_4X4 ..
  BLOCK_64X64` / `BLOCK_INVALID` `subsize` constants from §3.
* `get_plane_block_size( subsize, plane, subsampling_x, subsampling_y )`
  (§6.4.23) and `get_uv_tx_size( tx_size, mi_size, subsampling_x,
  subsampling_y )` (§6.4.22) — the chroma-plane block-size /
  transform-size derivations that key the per-plane loop.
* `ResidualBlockCtx` — the per-MI-block / per-frame bundle (`MiCol` /
  `MiRow` / `MiCols` / `MiRows`, `MiSize`, `tx_size`, `subsampling_x` /
  `y`, `skip`, `Lossless`, `BitDepth`, the per-block `PredMode` for luma
  + chroma, and the per-plane DC/AC quantizers from round 8); plus
  `AvailFlags` for §7.4.4 `AvailL` / `AvailU` and a `PlaneBuffers`
  wrapper for the three `CurrFrame[ plane ]` planes.
* `residual_intra` — the §6.4.21 driver proper: per plane, computes
  `bsize = MiSize < BLOCK_8X8 ? BLOCK_8X8 : MiSize`, the per-plane
  `planeSz` + `num4x4w` / `num4x4h` dimensions and chroma `txSz`, then
  walks the `(y, x)` 4x4 grid stepping by `step = 1 << txSz`. For each
  in-bounds transform block (`startX < maxx && startY < maxy`) it calls
  the round-10 `predict_intra` with the resolved `have_left` /
  `have_above` / `not_on_right` flags, pulls `Tokens[ ]` from a per-block
  `TokenSource` callback (when `!skip`), derives the §6.4.25 `TxType`
  (chroma / `TX_32X32` / lossless force `DCT_DCT`; luma intra uses
  round-11 `tx_type_for_intra`), runs the round-11 `reconstruct_block`,
  and writes `AboveNonzeroContext[ plane ][ x4 + i ] =
  LeftNonzeroContext[ plane ][ y4 + i ] = nonzero` for `i ∈ 0..step` per
  the §6.4.21 trailing write-back.

The `is_inter` branch of §6.4.21 (which calls `predict_inter( )` before
the per-block loop) is deferred until the §8.5.2 inter prediction
process and reference-buffer state land in a later round; the
per-block mode-info decode (`y_mode` / `sub_modes` / `tx_size` / `skip`
/ `segment_id` from §6.4) that the residual loop reads is also a
later-round increment — for round 14 the per-block mode-info bundle is
passed in by the test caller, and a production caller would thread it
in once the §6.4.6 / §6.4.7 / §6.4.10 mode-info syntax lands. The
round-14 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their result
types exclusively.

**Round 13: §6.4.24 `tokens( )` per-block coefficient driver.** Round 13
adds the §6.4.24 `tokens( )` driver to the `tokens` module — the per-block
loop that walks the round-12 §6.4.25 scan order (`pos = scan[ c ]`) and
feeds each scan position through the round-7 `read_coef_token` pipeline,
recovering one transform block's quantised coefficients into a `Tokens[ ]`
array:

* The §10 band tables — `coefband_4x4[ 16 ]` transcribed verbatim and
  `coefband_8x8plus[ 1024 ]` built from the verbatim 21-entry prefix plus
  the all-`5` tail — picked by `coef_band( c, txSz )` per the §6.4.24
  `(txSz == TX_4X4) ? coefband_4x4 : coefband_8x8plus` rule.
* `token_cache_neighbours( c, pos, txSz, txType )` — the §9.3.2 neighbour
  pair (`nb[ 0 ]` / `nb[ 1 ]`): `(0, 0)` for the DC coefficient, and for
  `c > 0` the above (`(i-1)*n + j`) / left (`i*n + j-1`) raster cells with
  the `DCT_ADST` (double above) / `ADST_DCT` (double left) / first-row /
  first-column variants (`n = 4 << txSz`).
* `build_token_probs( cell )` — the §9.3.2 10-node probability array
  (node 0 → `cell[1]`, node 1 → `cell[2]`, node `2..=9` →
  `pareto( node, cell[2] )`).
* `NonzeroContext` (the per-plane `AboveNonzeroContext` /
  `LeftNonzeroContext` 4-sample strips) and `TokenBlockCtx` (the per-block
  / per-frame state the driver reads).
* `tokens( coder, block, txSz, scan, coef_probs, nz, token_cache, tokens )`
  — the §6.4.24 driver proper: `segEob = 16 << (txSz << 1)`, the `checkEob`
  gating, the §9.3.2 per-coefficient `ctx` (DC from the non-zero strips,
  `c > 0` from `TokenCache`), the
  `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell pick, the
  `more_coefs` / `token` / `read_coef` / `sign_bit` decode, the
  `TokenCache[ pos ] = energy_class[ token ]` write, the
  `ZERO_TOKEN`-clears-`checkEob` rule, the trailing `Tokens[ scan[ i ] ] =
  0` zero-fill, and the `nonzero = c > 0` return.

The §6.4.21 `residual( )` plane / sub-block driver — which owns the
`AboveNonzeroContext` / `LeftNonzeroContext` write-back across a whole
frame, the per-block mode-info decode, and the wiring into the round-11
§8.6.2 `reconstruct_block` — lands in a later round. The round-13 surface
is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their result
types exclusively.

**Round 12: §6.4.25 `get_scan` scan-order selection.** Round 12 adds a
`scan` module (crate-internal, `pub(crate)`) implementing the §6.4.25
`get_scan( )` process — the first step of the §6.4.24 `tokens( )`
per-block driver, which selects the scan order (the sequence of raster
positions `pos = scan[ c ]` the coefficient loop walks) for a transform
block:

* The §10.1 scan tables transcribed verbatim — `default_scan_4x4` /
  `col_scan_4x4` / `row_scan_4x4` (16), the 8x8 trio (64), the 16x16
  trio (256), and `default_scan_32x32` (1024). The element type is
  `u16` so the 32x32 table's `0..=1023` raster positions fit.
* `get_scan( plane, tx_sz, tx_type )` (§6.4.25) — selects between the
  tables by the resolved `TxType`: `ADST_DCT` → `row_scan`, `DCT_ADST`
  → `col_scan`, else (`DCT_DCT` / `ADST_ADST`) → `default`. The §6.4.25
  first half (a chroma plane `plane > 0` or a `TX_32X32` block forces
  `TxType = DCT_DCT`) is applied here, so a caller passing the luma
  `TxType` for every plane still selects the right scan. The
  mode-info-dependent `mode2txfm_map[ y_mode ]` `TxType` derivation
  already lives in `reconstruct::tx_type_for_intra` (round 11); the
  per-block mode-info state (`y_mode`, `sub_modes`, `Lossless`,
  `is_inter`) is owned by the deferred §6.4.21 residual driver.
* `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` `txSz` index constants
  (§3).

The §6.4.24 `tokens( )` loop that walks `pos = scan[ c ]`, derives the
per-coefficient `ctx` from `AboveNonzeroContext` / `LeftNonzeroContext`
and `TokenCache`, and feeds the round-7 token decode lands in a later
round. The round-12 surface is internal-only; the public API still
exposes `parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 11: §8.6.2 reconstruct driver.** Round 11 adds a `reconstruct`
module (crate-internal, `pub(crate)`) implementing the §8.6.2
reconstruct process — the conceptual `reconstruct( plane, startX,
startY, txSz )` call site of the §6.4.21 residual syntax — that finally
ties the rounds 7-10 pieces into `reconstruct = predict + residual`:

* `tx_type_for_intra( mode )` (§6.4.25) — the `mode2txfm_map[ y_mode ]`
  lookup selecting the `TxType` (`DCT_DCT` / `ADST_DCT` / `DCT_ADST` /
  `ADST_ADST`) for a luma intra block from its `PredMode`. The 10-entry
  intra prefix of `mode2txfm_map` (§10.5) is transcribed verbatim (the
  four inter-mode entries, all `DCT_DCT`, are omitted as the helper
  only indexes the intra prefix).
* `reconstruct_block( plane_buf, x, y, tx_sz, tokens, dc_quant,
  ac_quant, tx_type, lossless, bit_depth )` (§8.6.2) — sets `dqDenom =
  2` for `txSz == TX_32X32` else `1`, `n = 2 + txSz`, `n0 = 1 << n`;
  step 1 `Dequant[i][j] = (Tokens[i*n0+j] * get_ac_quant) / dqDenom`,
  step 2 the `Dequant[0][0] = (Tokens[0] * get_dc_quant) / dqDenom` DC
  override, step 3 the round-9 §8.7.2 `inverse_transform_2d`, step 4
  `CurrFrame[ plane ][ y+i ][ x+j ] = Clip1( CurrFrame[..] +
  Dequant[i][j] )`. Integer division truncates toward zero per §4.1.
* `reconstruct_intra_block( .. )` — the end-to-end one-block driver:
  predicts via the round-10 §8.5.1 `predict_intra`, derives the
  `TxType` with the §6.4.25 `TX_32X32` / lossless `DCT_DCT` overrides,
  then runs `reconstruct_block`. This is the single-block shape the
  deferred §6.4.21 residual loop will drive once it threads real
  availability and quantizer state.

The §6.4.21 residual loop — which supplies the real per-block `Tokens`
arrays, availability flags and segment/quantizer state across a whole
frame, and wires this driver into a public decode path — lands in a
later round. The round-11 surface is internal-only; the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

**Round 10: §8.5.1 intra prediction process.** Round 10 adds an
`intra` module (crate-internal, `pub(crate)`) implementing the §8.5.1
intra prediction the §8.6.2 reconstruct process invokes for intra
blocks (the prediction half of `reconstruct = predict + residual`):

* `PredMode` — the 10 §7.4.5 intra prediction modes, with
  discriminants matching the spec numbering exactly (`DC_PRED` = 0,
  `V_PRED` = 1, `H_PRED` = 2, `D45_PRED` = 3, `D135_PRED` = 4,
  `D117_PRED` = 5, `D153_PRED` = 6, `D207_PRED` = 7, `D63_PRED` = 8,
  `TM_PRED` = 9), plus `from_raw` for the (deferred) mode-info decode.
* `Plane` — a minimal row-major `i32` plane buffer standing in for
  `CurrFrame[ plane ]`, read for neighbour samples and written with
  the prediction.
* `predict_intra( .. )` (§8.5.1) — builds the `aboveRow[-1 .. 2*size-1]`
  and `leftCol[0 .. size-1]` neighbour arrays per the `haveAbove` /
  `haveLeft` / `notOnRight` availability rules (including the
  upper-right extension that fires only for `txSz == 0` and the
  `(1<<(BitDepth-1)) ± 1` no-neighbour fills), then forms the `pred`
  block for the selected mode: `V`/`H` copies, the four `DC` neighbour
  cases (`avg` / `leftAvg` / `aboveAvg` / midpoint), the
  `D45`/`D63`/`D117`/`D135`/`D153`/`D207` directional `Round2`
  recurrences, and `TM` with `Clip1`. Neighbour reads clamp with
  `Min(maxX, .)` / `Min(maxY, .)`; the result is stored back into the
  plane.

The §8.6.2 reconstruct driver — which supplies the real `haveAbove` /
`haveLeft` / `notOnRight` flags (from tile / frame edges) and adds the
round-9 inverse-transformed residual to this prediction — lands in a
later round. The round-10 surface is internal-only; the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

**Round 9: §8.7 inverse transform process.** Round 9 adds an `idct`
module (crate-internal, `pub(crate)`) implementing the §8.7 inverse
transform stage the §8.6.2 reconstruct process invokes after the
round-8 dequantization step:

* The §8.7.1.1 butterfly primitives — `B` (butterfly rotation, with
  the `16 + 32*k` two-multiply fast path), `H` (Hadamard rotation),
  `SB` (butterfly into the high-precision `S` array) and `SH`
  (Hadamard rotation + `Round2(·, 14)` out of `S`) — plus `cos64` /
  `sin64` backed by the verbatim 33-entry `cos64_lookup` quarter-wave
  table and the `brev` bit-reversal helper. Fixed-point intermediates
  are evaluated in `i64` (the spec notes `S` needs `24 + BitDepth`
  bits of precision).
* `inverse_dct( t, n )` (§8.7.1.2 + §8.7.1.3) — the inverse-DCT array
  bit-reversal permutation followed by the recursive inverse DCT
  process for `2 <= n <= 5` (4/8/16/32-point).
* `inverse_adst( t, n )` (§8.7.1.4 .. §8.7.1.9) — the ADST
  input/output permutations and the ADST4 / ADST8 / ADST16 processes
  (with the `SINPI_1_9 .. SINPI_4_9` constants transcribed verbatim)
  dispatched by `n` for `2 <= n <= 4`.
* `inverse_wht( t, shift )` (§8.7.1.10) — the in-place inverse
  Walsh-Hadamard transform.
* `inverse_transform_2d( dequant, n, tx_type, lossless )` (§8.7.2) —
  the 2D driver applying the per-`TxType` row transform then column
  transform over a `(1<<n)` by `(1<<n)` block, the lossless WHT path
  (`shift = 2` rows / `0` columns), and the `Round2( T[i], Min(6,
  n+2) )` column rounding. `TxType` constants `DCT_DCT` / `ADST_DCT`
  / `DCT_ADST` / `ADST_ADST` follow §3.

The §8.6.2 reconstruct driver — which builds the `Dequant` input
(round-7 token magnitudes scaled by the round-8 quantizers, with the
`dqDenom = 2` halving for `TX_32X32`), calls this transform layer and
adds the residual to the prediction — lands in a later round. The
round-9 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 8: §8.6.1 dequantization functions.** Round 8 adds a
`dequant` module (crate-internal, `pub(crate)`) implementing the
quantizer-value derivation the §8.6.2 reconstruct process consumes
between the round-7 coefficient-token decode and the §8.7 inverse
transform:

* `dc_q( bit_depth, b )` / `ac_q( bit_depth, b )` (§8.6.1) — index
  the `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
  `(BitDepth - 8) >> 1` row (0 / 1 / 2 for 8- / 10- / 12-bit) and the
  `Clip3(0, 255, b)` column. Both 256-entry tables are transcribed
  verbatim from the §8.6.1 listing into `DC_QLOOKUP` / `AC_QLOOKUP`.
* `seg_feature_active( seg, segment_id, feature )` (§6.4.9) —
  `segmentation_enabled && FeatureEnabled[ segment_id ][ feature ]`.
* `get_qindex( seg, quant, segment_id )` (§8.6.1) — the per-block
  quantizer index. When `seg_feature_active( SEG_LVL_ALT_Q )`, the
  segment's `FeatureData` either replaces `base_q_idx` (absolute
  update) or offsets it (delta update), then `Clip3(0, 255, .)`;
  otherwise `base_q_idx` is returned directly.
* `get_dc_quant( plane, .. )` / `get_ac_quant( plane, .. )`
  (§8.6.1) — combine `get_qindex()` with the plane-appropriate
  header delta (`delta_q_y_dc` luma DC, `delta_q_uv_dc` chroma DC,
  `delta_q_uv_ac` chroma AC; luma AC has no delta in VP9) and
  dispatch to `dc_q` / `ac_q`.

The §8.6.2 reconstruct driver — which scales the round-7 `Tokens`
array by these quantizers (with the `dqDenom = 2` halving for
`TX_32X32`), runs the §8.7 inverse transform and adds the residual
to the prediction — lands in a later round. The round-8 surface is
internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 7: §6.4.24 / §6.4.26 coefficient-token decoder.** Round 7
adds a `tokens` module (crate-internal, `pub(crate)`) implementing
the pieces needed to recover one quantised DCT coefficient at a time
from the §9.2 Boolean coder, given a pre-selected
`coef_probs[ txSz ][ plane>0 ][ is_inter ][ band ][ ctx ][..3 ]` cell:

* The §9.3 `token_tree[20]` walker — `read_token( coder, &probs )` —
  returning one of the 11 spec tokens (`ZERO_TOKEN` through
  `DCT_VAL_CATEGORY6`).
* The §9.3.2 `pareto( node, prob )` helper backed by the §10.3
  128×8 `pareto_table` (transcribed verbatim).
* The §6.4.24 `more_coefs` `B(p)` reader.
* The §6.4.26 `read_coef( coder, token, bit_depth )` extra-bits
  decoder consuming the `extra_bits[11][3]` and `cat_probs[7][14]`
  tables (transcribed verbatim) plus the CAT6 `high_bit` `B(255)`
  loop that prepends `BitDepth - 8` MSBs for 10- and 12-bit profiles
  at shift positions `5 + BitDepth - e`.
* A `read_coef_token` driver that wires `more_coefs` + `read_token` +
  `read_coef` + `L(1) sign_bit` together and returns
  `CoefStep::{ Eob, Coef { token, value } }`.
* The §10.2 `energy_class[12]` table (for `TokenCache` once the
  residual driver lands).

The hand-traced golden buffers in the test suite cover every token
slot's "all-zero extra bits" base value, both legs of the
`more_coefs` decision, and two non-trivial tree-walk paths
(`ONE_TOKEN` from buffer `0x40 0x00 …`, `TWO_TOKEN` from buffer
`0x60 0x00 …`).

The §6.4.21 `residual( )` plane/sub-block driver still lands in a
later round — it owns the `AboveNonzeroContext` / `LeftNonzeroContext`
arrays and the per-block `ctx` derivation that picks the right
`coef_probs` cell. The round-7 surface is internal-only; the public
API still exposes `parse_uncompressed_header` and
`parse_compressed_header` exclusively.

**Round 6: §6.3.7 `read_coef_probs` 6D coefficient-probability sweep
wired in.** Round 6 inserts the §6.3.7 walker between the round-5
§6.3.2 `tx_mode_probs` and §6.3.8 `read_skip_prob` calls. The
walker:

* Visits `txSz ∈ [TX_4X4, maxTxSize]` where `maxTxSize =
  tx_mode_to_biggest_tx_size[ tx_mode ]` per §10.5 (1, 2, 3, 4 or 4
  active tx-size slabs for `ONLY_4X4` / `ALLOW_8X8` / `ALLOW_16X16`
  / `ALLOW_32X32` / `TX_MODE_SELECT` respectively).
* For each active slab, reads an outer `L(1) update_probs` flag.
* If 1, walks the nested `(i, j, k, l, m)` sweep — `BLOCK_TYPES=2 ×
  REF_TYPES=2 × COEF_BANDS=6 × maxL(k) × UNCONSTRAINED_NODES=3`
  cells where `maxL = (k == 0) ? 3 : 6` (band 0 has only 3 valid
  previous-coef contexts; the §10 listing trails it with `{0, 0, 0}
  // unused` rows).
* Each cell becomes `read_diff_update_prob( coder, cell )` against
  the running `coef_probs[ txSz ][ i ][ j ][ k ][ l ][ m ]` table.

When `update_probs == 1` the inner sweep traverses `2 × 2 × (3 + 5
× 6) × 3 = 396` cells per active tx-size, totalling up to `4 × 396
= 1584` `diff_update_prob` calls for a fully-active
`TX_MODE_SELECT` frame.

Initial probabilities come from the §10 `default_coef_probs[
TX_SIZES ][ BLOCK_TYPES ][ REF_TYPES ][ COEF_BANDS ][
PREV_COEF_CONTEXTS ][ UNCONSTRAINED_NODES ]` listing (1728 u8
entries) transcribed verbatim into a new `DEFAULT_COEF_PROBS`
constant in `src/coef_probs.rs`. The public type alias `CoefProbs`
names the 6D shape so callers of [`parse_compressed_header`] can
match on the new `Vp9CompressedHeader::coef_probs` field.

The §6.3.7 walker is **structural-parse only**: it advances the
§9.2 Boolean coder by exactly the bits the spec dictates and
updates the running `coef_probs` table, but the actual entropy
decode of residual coefficients (§6.4) still lands in a later
round.

**Round 5: §6.3.2 `tx_mode_probs` + §6.3.8 `read_skip_prob` sweeps
wired into the compressed-header walker.** Round 5 consumes the
round-4 `read_diff_update_prob` primitive in two table sweeps:

* `tx_mode_probs( )` (§6.3.2) — gated on `tx_mode == TX_MODE_SELECT`,
  walks `tx_probs_8x8[2][1]`, `tx_probs_16x16[2][2]`,
  `tx_probs_32x32[2][3]` (12 cells total) via
  `read_diff_update_prob` starting from the §10 `default_tx_probs`
  initials transcribed verbatim into `DEFAULT_TX_PROBS`.
* `read_skip_prob( )` (§6.3.8) — unconditional 3-element
  (`SKIP_CONTEXTS = 3`) sweep over the §10 `default_skip_prob = {
  192, 128, 64 }` initials transcribed verbatim into
  `DEFAULT_SKIP_PROB`.

`Vp9CompressedHeader` now exposes the post-sweep `tx_probs` /
`skip_prob` tables. When `tx_mode != TX_MODE_SELECT` the §6.3
syntax skips `tx_mode_probs( )` entirely, so the field is left
equal to `DEFAULT_TX_PROBS`. `parse_compressed_header` runs both
sweeps in spec order: `read_tx_mode` → optional `read_tx_mode_probs`
→ `read_skip_prob`.

The inter-only §6.3.9+ syntax (`read_inter_mode_probs`,
`read_interp_filter_probs`, `read_is_inter_probs`,
`frame_reference_mode`, `mv_probs`) remains deferred — those fire
only on `FrameIsIntra == 0` and need reference-buffer state which
the uncompressed-header walker still rejects with
`Error::Unsupported`.

**Round 4: §6.3.3 `diff_update_prob` chain (`decode_term_subexp` +
`inv_remap_prob` + `inv_recenter_nonneg` + 255-entry
`inv_map_table`).** Round 4 lands the helper chain every §6.3.7+
probability sweep is built on:

* `read_diff_update_prob( coder, base_prob )` (§6.3.3) — reads the
  `B(252)` `update_prob` flag and, on 1, pulls a
  `decode_term_subexp` value (§6.3.4) then remaps the previous
  probability through `inv_remap_prob` (§6.3.5). On 0, passes the
  base probability straight through.
* `decode_term_subexp` (§6.3.4) — the 5-leg cascade
  (`L(1) → L(4)` / `L(1) → L(4)+16` / `L(1) → L(5)+32` /
  `L(7) → +64` / `L(7), L(1) → (v<<1)-1+bit`) yielding a value in
  `0..=254`.
* `inv_remap_prob` (§6.3.5) — the low-half / high-half piecewise
  remap, with the 255-entry `INV_MAP_TABLE` constant transcribed
  verbatim from the §6.3.5 listing.
* `inv_recenter_nonneg` (§6.3.6) — pure arithmetic helper.

The chain is structural — no caller in §6.3.2 / §6.3.7+ uses it yet.
That's deferred to the next round so this drop lands the primitive
in isolation. All four helpers are `pub(crate)` with explicit
`#[allow(dead_code)]` until the next round wires them into the
table sweeps.

**Round 3: §9.2 Boolean decoder + §6.3.1 `read_tx_mode` walk.** Building
on round 2's full §6.2 uncompressed-header walker, round 3 lands the
arithmetic-decode primitive — `init_bool( sz )` / `read_bool( p )` /
`read_literal( n )` / `exit_bool( )` per spec §9.2.1–§9.2.4 — plus the
first §6.3 compressed-header field (`read_tx_mode( )`, §6.3.1) exposed
via `parse_compressed_header(payload, lossless) -> Vp9CompressedHeader`.

Decoded `tx_mode` covers all five §3 values (`ONLY_4X4`, `ALLOW_8X8`,
`ALLOW_16X16`, `ALLOW_32X32`, `TX_MODE_SELECT`), including the lossless
short-circuit to `ONLY_4X4`. The §9.2.1 marker-bit zero-conformance
check is enforced (`InvalidBitstream` on a nonzero marker), and the
§9.2.2 `BoolMaxBits` underflow path is surfaced rather than silently
papered over.

The remaining §6.3.7+ syntax — `tx_mode_probs`, `read_coef_probs`,
`read_skip_prob`, `read_inter_mode_probs`,
`read_interp_filter_probs`, … — all funnel through
`read_diff_update_prob` (landed this round) and the table sweeps
themselves land in the next round. `read_inter_mode_probs` /
`read_interp_filter_probs` only fire on inter-frame headers, which
already return `Error::Unsupported` until reference-buffer state
lands.

**Round 2: full §6.2 uncompressed-header walk.** Building on round 1,
`parse_uncompressed_header(&[u8]) -> Result<Vp9FrameHeader, Error>` now
walks the entire `uncompressed_header()` syntax tree plus the §6.1.1
`trailing_bits()` zero-fill alignment. Per-field coverage:

* Round 1: `frame_marker`, `Profile` (with the `profile == 3`
  `reserved_zero` bit), `show_existing_frame` early-return path
  (returns `frame_to_show_map_idx`), `frame_type`, `show_frame`,
  `error_resilient_mode`, `frame_sync_code` (`0x49 / 0x83 / 0x42`),
  `color_config()` (bit depth 8/10/12, all 8 `color_space` values,
  `color_range`, `subsampling_x` / `subsampling_y` with the §7.2.2
  reserved-zero and CS_RGB-on-profile-0/2 constraint checks), and
  `frame_size` / `render_size` (with the
  `render_and_frame_size_different == 1` override).
* Round 2: post-`render_size` syntax — `refresh_frame_context`,
  `frame_parallel_decoding_mode`, `frame_context_idx`, the
  `setup_past_independence` `frame_context_idx = 0` reset for intra /
  error-resilient frames, `loop_filter_params()` (§6.2.8) including
  full delta-update walk with `s(6)` ref/mode deltas,
  `quantization_params()` (§6.2.9) with `read_delta_q()` (§6.2.10),
  `segmentation_params()` (§6.2.11) with `read_prob()` (§6.2.12) +
  per-segment / per-feature data using the `segmentation_feature_bits`
  / `segmentation_feature_signed` tables, `tile_info()` (§6.2.13)
  driven by `Sb64Cols` computed from `FrameWidth` per §6.2.6 /
  §6.2.14, the f(16) `header_size_in_bytes`, and the §6.1.1
  `trailing_bits()` zero-pad alignment with §7.1.1 zero-bit
  conformance check.

`decode_vp9()` / `encode_vp9()` still return `Error::NotImplemented`;
the compressed header (§6.3) plus entropy decoder, intra/inter
prediction, transforms and loop filter land in later rounds. Inter
(non-intra-only) headers — which require `frame_size_with_refs` and
reference-buffer state — return `Error::Unsupported` from the header
walker for now.

The `Vp9FrameHeader` struct now also exposes
`uncompressed_header_size_bytes` (the byte-aligned offset at which the
§6.3 compressed header starts), giving callers everything needed to
slice the compressed-header payload of `header_size_in_bytes` from the
remainder of the frame.

## Test surface

* `cargo test`: 285 unit tests + 20 integration tests (8 in
  `tests/compressed_header.rs` plus 12 in
  `tests/uncompressed_header.rs`).
* Round-18 additions: 37 unit tests covering the §3 partition
  enumeration (`PARTITION_NONE` / `_HORZ` / `_VERT` / `_SPLIT` =
  0..=3) and dimensions (`PARTITION_TYPES = 4`,
  `PARTITION_CONTEXTS = 16`); the four §10.2 lookups
  (`B_WIDTH_LOG2_LOOKUP` / `B_HEIGHT_LOG2_LOOKUP` /
  `MI_WIDTH_LOG2_LOOKUP` / `NUM_8X8_BLOCKS_WIDE_LOOKUP`) against
  the spec listings; `SUBSIZE_LOOKUP` `PARTITION_NONE` identity,
  `PARTITION_SPLIT` superblock anchors (`BLOCK_8X8 -> BLOCK_4X4`,
  …, `BLOCK_64X64 -> BLOCK_32X32`), `PARTITION_HORZ` /
  `PARTITION_VERT` superblock anchors, and the
  `BLOCK_INVALID = 14` cell at non-square parents; the three
  §9.3.1 trees (`PARTITION_TREE[6]`, `COLS_PARTITION_TREE[2]`,
  `ROWS_PARTITION_TREE[2]`) verbatim transcription;
  `KF_PARTITION_PROBS` + `DEFAULT_PARTITION_PROBS` shape + four
  per-table listing anchors + §9.2 minimum-probability sanity
  check; `partition_plane_context` zero-strips + above-bit-set +
  left-bit-set + both-set across each of the four superblock
  sizes, the OR-fold across the strip width, unrelated-bit
  masking, the panic-on-`BLOCK_INVALID` / mismatched-strip
  guards, and an exhaustive sweep that proves all 16
  `ctx ∈ 0..=15` are reachable across (`bsize`, `above_bit`,
  `left_bit`); `decode_partition_type` against the zero coder
  (`PARTITION_NONE` / `_HORZ` / `_VERT` for the four arm
  combinations), the all-ones coder (interior right-branch walk
  → `PARTITION_SPLIT`), the one-then-zero coder (interior →
  `PARTITION_HORZ`, right-edge → `PARTITION_SPLIT`, bottom-edge →
  `PARTITION_SPLIT`, with a follow-up confirming the same outcome
  for uniform probs = 255), the corner (`(false, false)`) case
  with a bool-coder reuse probe proving zero bits are consumed,
  and an exhaustive 4 × 3 (arm × buffer) smoke-test confirming
  the output stays in `0..=3`.
* Round-16 additions: 7 unit tests covering the §9.3.1
  `segment_tree[14]` verbatim transcription, `read_segment_id` against
  the zero coder (every bit=0 walks `tree[0]=2 → tree[2]=6 →
  tree[6]=0` and returns segment 0), the bias-buffer + `[255; 7]`
  trace producing segment id 4 (first read flips to bit=1 picking
  inner branch 4, the next two collapse to bit=0 picking leaf -4), the
  §9.3.1 node-index packing fact that a pure-left walk visits node
  indices `{0, 1, 3}` (not contiguous), `intra_segment_id` short-
  circuiting to `segment_id = 0` when either `segmentation_enabled`
  or `segmentation_update_map` is false (without dereferencing
  `tree_probs`), the active path decoding through `read_segment_id`,
  and the `Error::InvalidBitstream` surface when the caller forgets
  to thread `tree_probs` despite the active gate.
* Round-15 additions: 22 unit tests covering the §9.3.1 tree-listing
  anchors (`tx_size_8_tree` / `tx_size_16_tree` / `tx_size_32_tree` /
  `binary_tree` verbatim transcription), the §9.3.3 `tree_decode`
  walker with a zero-coder (every read yields bit=0, every tree picks
  its first leaf), a "bias buffer" prefix `[0x7F, 0x00, ...]` whose
  post-marker coder state lets the first `read_bool(255)` flip to 1
  (one right-branch step), the node-index argument-order invariant of
  the prob callback, exhaustive `skip_context` cases (no neighbours,
  single-side, both-sides mixed and matching), `tx_size_context`
  against the §9.3.2 listing across `max_tx_size = 0..=3` plus the
  skipped-neighbour fallback and the `!AvailL` / `!AvailU` mirroring,
  the §6.4.8 `seg_feature_active(SEG_LVL_SKIP)` early-return path,
  `read_skip` against both the zero coder (always false) and the bias
  coder + `p=255` (true), the §6.4.10 `else`-branch
  `Min(maxTxSize, biggest)` fallback for `allow_select == false` /
  non-SELECT `tx_mode` / sub-8x8 `MiSize`, the §9.3.1 tree-dispatch
  for `BLOCK_8X8` / `BLOCK_16X16` / `BLOCK_32X32`, and the row-by-ctx
  selection correctness via lockstep against `tx_size_context` at
  both ctx=0 and ctx=1.
* Round-14 additions: 12 unit tests covering the §10.2
  `num_4x4_blocks_wide_lookup` / `_high_lookup` and §6.4.10
  `max_txsize_lookup` listings, the §6.4.23 `ss_size_lookup` luma
  identity invariant + the 4:2:0 / asymmetric-subsampling anchors
  (`BLOCK_8X8 -> BLOCK_4X4`, `BLOCK_64X64 -> BLOCK_32X32`, the
  `(1,0)` / `(0,1)` mixed chroma cases), the §6.4.22 `get_uv_tx_size`
  chroma cap (`MiSize=BLOCK_16X16 -> chroma TX_8X8`) and the sub-8x8
  short-circuit, the `skip = true` path leaving every `AboveNonzero`
  / `LeftNonzero` cell at 0 (no token decode), the full `skip = false`
  walk firing `token_source` exactly 16 luma + 4 U + 4 V times for a
  `BLOCK_16X16` MI block at `tx_size = TX_4X4` (each call recorded
  with `(plane, block_idx, tx_sz)`), the `nonzero = true` strip
  write-back over `step = 1 << tx_sz` 4-sample units for `tx_size =
  TX_8X8`, the out-of-bounds (`startX >= maxx`) block skip with
  intact zero context, a DC-only luma block at MI (1,1) lockstep
  against an independent `predict_intra` + `reconstruct_block` probe
  (proving the residual loop ties the rounds 10/11/13 pieces
  identically), and the §6.4.21 `bsize = max(MiSize, BLOCK_8X8)`
  widening for a `BLOCK_4X4` MI block (4 luma + 1 U + 1 V blocks
  decoded).
* Round-13 additions: 15 unit tests covering `coefband_4x4` /
  `coefband_8x8plus` against the §10 listing (21-entry prefix + all-`5`
  tail) and the `coef_band` tx-size dispatch, the §9.3.2
  `token_cache_neighbours` derivation (DC origin, interior DCT_DCT,
  the `DCT_ADST` / `ADST_DCT` double-neighbour variants, first-row /
  first-column fallbacks, and the `n = 4 << txSz` 8x8 width scaling),
  `build_token_probs` node mapping (node 0 → `cell[1]`, node 1 →
  `cell[2]`, node `2..=9` → `pareto`), and the `tokens( )` driver
  itself: a zero-buffer immediate EOB (returns `nonzero = false`, all
  `Tokens` zeroed), the `ZERO_TOKEN`-clears-`checkEob` block fill
  (`nonzero = true`, explicit zeros), a lockstep equality against an
  independent `read_coef_token` walk over the same scan / buffer
  (`Tokens` + `TokenCache` + return value), the DC non-zero-context
  cell routing (a non-zero strip selects the `ctx = 2` cell), the
  trailing `c..segEob` zero-fill on an 8x8 block, and the
  `NonzeroContext::new` all-zero invariant.
* Round-12 additions: 10 unit tests covering every §10.1 scan table's
  spec length, the permutation invariant (each raster position appears
  exactly once — the property the §6.4.24 loop relies on to zero
  untouched coefficients), the DC-first invariant, §10.1 listing
  anchors (first four + last entry of each table), the §6.4.25
  `txType` → table selection for 4x4 / 8x8 / 16x16, the
  `TX_32X32`-always-default and chroma-forces-`DCT_DCT` first-half
  overrides, and the `16 << (txSz << 1)` `segEob` scan-length match.
* Round-11 additions: 10 unit tests covering the `mode2txfm_map` intra
  prefix vs the §10.5 listing, the local `clip1` range clamping, an
  all-zero `Tokens` block leaving the prediction unchanged, a DC-only
  `Tokens` block adding a flat residual to a flat prediction, step-4
  `Clip1` saturation at both the bit-depth max and zero, the
  `TX_32X32` `dqDenom = 2` halving exercised through the real
  `reconstruct_block`, and three `reconstruct_intra_block` end-to-end
  cases (DC_PRED + zero tokens equalling the pure DC prediction,
  DC_PRED + a known DC residual reconstructing the expected pixels,
  and the lossless WHT path).
* Round-10 additions: 16 unit tests covering `PredMode::from_raw`
  round-tripping the §7.4.5 numbering (and rejecting 10+), the local
  `round2` / `clip1` helpers against their §3 definitions, `V_PRED`
  copying `aboveRow` down and `H_PRED` copying `leftCol` across, all
  four `DC_PRED` neighbour cases (both / left-only / above-only /
  none), `TM_PRED`'s `Clip1(aboveRow[j] + leftCol[i] - aboveRow[-1])`
  formula (including out-of-range clipping), a constant-neighbour
  invariant that collapses every directional mode to the neighbour
  constant across all four transform sizes, the `D207_PRED` bottom-row
  / step-2 formula, the `notOnRight`-gated upper-right extension of
  `aboveRow` (enabled only for `txSz == 0`) via `D45_PRED`, and the
  `Min(maxX, .)` plane-edge clamping of neighbour reads.
* Round-8 additions: 13 unit tests covering the `DC_QLOOKUP` /
  `AC_QLOOKUP` shape (256 entries per row) and §8.6.1 listing
  anchors (first/last entry of every row plus two interior anchors),
  `qlookup_row` bit-depth mapping (8→0 / 10→1 / 12→2), `clip3`
  both-end clamping, `dc_q` / `ac_q` index clipping + bit-depth row
  selection, `get_qindex` for the segmentation-off / delta-update /
  absolute-update paths (with the `Clip3` clamps), `seg_feature_active`
  needing both `segmentation_enabled` and the per-segment bit,
  `get_dc_quant` / `get_ac_quant` plane-delta selection (luma AC has
  no delta), a segment-overridden qindex threaded through both, and
  the high-bit-depth divergence of the same qindex across the three
  table rows.
* Round-6 additions: 7 unit tests covering
  `tx_mode_to_biggest_tx_size` against the §10.5 listing,
  `read_coef_probs` zero-buffer passthrough for both `ONLY_4X4`
  (1-slab) and `TX_MODE_SELECT` (4-slab) paths plus an across-mode
  sweep, `DEFAULT_COEF_PROBS` shape (4 × 2 × 2 × 6 × 6 × 3 = 1728
  entries) + four hand-picked anchor values from the §10 listing,
  the band-0 unused-row sentinel invariant (every `(txSz, i, j)`
  triple's band 0 has `{0, 0, 0}` at contexts 3..6), and the inner
  sweep cell-count check `2 × 2 × (3 + 5 × 6) × 3 = 396`. 2 new
  end-to-end integration tests (`tests/compressed_header.rs`)
  splicing a TX_MODE_SELECT frame and an ONLY_4X4 frame through
  the full pipeline and verifying §10 default anchors survive the
  zero-buffer §6.3.7 sweep verbatim.
* Round-5 additions: 10 unit tests covering `DEFAULT_TX_PROBS`
  shape + value spot-check against the §10 listing, `DEFAULT_SKIP_PROB
  = [192, 128, 64]`, `read_tx_mode_probs` on a zero buffer
  (defaults pass through; row 0 (`TX_4X4`) is never touched), the
  12-cell loop-count shape, `read_skip_prob` on a zero buffer plus
  a 3-context-visit sanity check, and three `parse_compressed_header`
  integration tests confirming the §10 defaults survive the
  zero-buffer sweep for the `ONLY_4X4`, lossless, and `TX_MODE_SELECT`
  paths. 2 new end-to-end integration tests
  (`tests/compressed_header.rs`): one driving the
  `TX_MODE_SELECT` → `tx_mode_probs` → `read_skip_prob` chain from
  a 64×64 uncompressed-header splice, one confirming `ALLOW_16X16`
  skips the §6.3.2 sweep.
* Round-4 additions: 16 unit tests covering `inv_recenter_nonneg`
  (all three piecewise branches plus the `v == 2*m` boundary),
  `INV_MAP_TABLE` length + spot-checks against the spec listing
  anchors (`[0]=7`, `[19]=254`, `[20]=1`, `[254]=253` with the
  duplicated trailing 253), `inv_remap_prob` covering both
  low-half and high-half `(m << 1) ≤ 255` branches plus the
  `v > 2m` short-circuit, `decode_term_subexp` against
  hand-derived §9.2 buffers (leg-1 zero plus a sweep that confirms
  the result stays in `0..=254` for every first-byte family), and
  `read_diff_update_prob` confirming the `update_prob == 0`
  passthrough against the full 1..=255 base-probability sweep.
* Round-3 additions: 9 `bool_coder` unit tests covering
  `init_bool` (zero-size / short-slice / nonzero-marker rejection +
  the zero-buffer accept path), `read_bool` against hand-traced
  golden buffers (mixed probabilities, extreme p=255 run), `read_literal`,
  and `exit_bool` accept/reject paths; plus 7 `compressed` unit tests
  for `parse_compressed_header` covering each `TxMode` value
  (lossless short-circuit, `ONLY_4X4`/`ALLOW_8X8`/`ALLOW_16X16`/
  `ALLOW_32X32`/`TX_MODE_SELECT` golden buffers) and the
  marker / empty-buffer rejection paths; and 4 end-to-end
  integration tests (`tests/compressed_header.rs`) that build a
  64x64 key-frame uncompressed header, splice a §9.2 payload past
  the byte-aligned `trailing_bits` pad, and verify the walker
  picks up the right `tx_mode` from the spliced payload.
* Uncompressed-header coverage (rounds 1 + 2, unchanged) spans the
  four profiles, studio/full-swing color ranges, render-size
  overrides, the `show_existing_frame` early return, the intra-only
  inter-frame branch (with the spec's BT.601 / 4:2:0 / 8-bit defaults
  for Profile 0), full `loop_filter_params` delta update with mixed
  `update_ref_delta` / `update_mode_delta` flags and signed `s(6)`
  deltas, `quantization_params` with a nonzero `base_q_idx` and
  signed `delta_q_y_dc`, full segmentation with `update_map` +
  `temporal_update` + `update_data` driving the per-segment /
  per-feature inner loop including the 0-magnitude-bit skip feature,
  `tile_info` increment-walk for a 4K-wide frame, plus three failure
  paths (bad `frame_marker`, bad `frame_sync_code`, truncated buffer)
  and the §7.1.1 nonzero trailing-bit rejection.
* No external fixtures are involved yet; each test constructs its
  input bit-by-bit (and §9.2 golden buffers are hand-derived by
  stepping the decoder, not borrowed from any third-party VP9
  implementation).

## Provenance

Single source of truth: VP9 Bitstream & Decoding Process Specification
v0.7 (`docs/video/vp9/vp9-spec.txt`). No external library source —
`libvpx`, `libaom`, FFmpeg's `libavcodec/vp9*`, `dav1d`, `libgav1`,
prior 0.0.x releases of this crate — has been consulted, quoted, or
cross-checked. Black-box `ffmpeg` binary invocations remain
permissible as opaque validators but are not yet wired into the test
harness.

## Roadmap

Future rounds, roughly in order:

1. Per-block mode-info decode (`y_mode` / `sub_modes` / `tx_size` /
   `skip` / `segment_id`) per §6.4.6 / §6.4.7 / §6.4.10. Round 15
   landed §6.4.8 `read_skip` + §6.4.10 `read_tx_size` + the §9.3.3
   `tree_decode` walker; round 16 added §6.4.7 `intra_segment_id` +
   the §9.3.1 `segment_tree[14]`; round 17 added the §6.4.6
   `intra_frame_mode_info()` keyframe orchestrator on top of the
   §9.3.1 `intra_mode_tree[18]` + §10.5 `kf_y_mode_probs` /
   `kf_uv_mode_probs` decode (`default_intra_mode` plus
   `default_uv_mode`, with the `MiSize < BLOCK_8X8` `(idy, idx)`
   sub-block walk fanning a per-cell `default_intra_mode` into the
   `sub_modes[ ]` array, and the §9.3.2 above/left neighbour
   derivation handled by the `IntraFrameNeighbours` bundle). The
   remaining intra piece is §6.4.15 `intra_block_mode_info` — the
   intra-block branch used inside inter frames, which uses
   `y_mode_probs[size_group_lookup[MiSize]][node]` /
   `y_mode_probs[0][node]` / `uv_mode_probs[y_mode][node]` from the
   compressed header rather than the keyframe `kf_*_mode_probs`
   tables; it lands alongside the §6.4.11 inter-frame driver. Once
   that lands the per-block `BoolCoder` token decode can replace the
   round-14 `TokenSource` callback to expose a public single-MI-block
   intra decode path; the partition-tree walk (§6.4.3) closes the
   per-tile loop.
2. Inter (non-intra-only) header path — `frame_size_with_refs`,
   `allow_high_precision_mv`, `read_interpolation_filter` — plus
   the inter-only §6.3.9–§6.3.16 syntax (`read_inter_mode_probs`,
   `read_interp_filter_probs`, `read_is_inter_probs`,
   `frame_reference_mode`, `mv_probs`) once reference-buffer state
   is in place.
3. Per-tile partition-tree walk (§6.4) including the §6.4.21
   `residual()` driver that finally consumes the round-7 tokens and
   the round-6 `coef_probs` tables, plus the per-block mode-info
   decode that feeds `predict_intra`. Round 18 added the §6.4.3
   `decode_partition_type()` per-call primitive plus the §10.2
   `subsize_lookup` / `b_*_log2_lookup` / `num_8x8_blocks_wide_lookup`
   tables and the §10.4 `kf_partition_probs` / §10.5
   `default_partition_probs` probability tables; the recursive
   `decode_partition(r, c, bsize)` driver that splits on the
   decoded `partition`, threads `subsize_lookup[partition][bsize]`
   into four recursive calls when `PARTITION_SPLIT`, and writes
   back `AbovePartitionContext[]` / `LeftPartitionContext[]`
   with `15 >> b_*_log2_lookup[subsize]` lands in the next round,
   alongside the §6.3 `read_partition_probs()` compressed-header
   sweep that updates `default_partition_probs` on inter frames.
4. Inter prediction (§8.5.2), loop filter (§8.8), multi-tile, then
   encoder paths.

## License

MIT. See `LICENSE`.
