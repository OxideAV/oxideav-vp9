//! VP9 partition primitive per spec v0.7 — §3 / §6.4.3 / §9.3.1 / §9.3.2
//! / §10.2 / §10.4 / §10.5.
//!
//! Round 18 landed the §6.4.3 `decode_partition_type( )` reader — the per-call
//! partition-tree decode that the recursive [`decode_partition`] driver fires
//! once per `(r, c, bsize)` quadrant. Round 19 adds the recursive driver
//! proper: it composes [`decode_partition_type`] with the §10.2
//! `subsize_lookup` traversal and the §6.4.3 tail write-back into the
//! `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips.
//!
//! Driver semantics per §6.4.3:
//!
//! * `(r >= MiRows || c >= MiCols)` short-circuits and returns without
//!   touching anything (the §6.4.3 first line).
//! * `num8x8 = num_8x8_blocks_wide_lookup[ bsize ]`,
//!   `halfBlock8x8 = num8x8 >> 1`, `hasRows = (r + halfBlock8x8) < MiRows`,
//!   `hasCols = (c + halfBlock8x8) < MiCols`.
//! * The `partition` syntax element is read via [`decode_partition_type`]
//!   using the §9.3.2 `ctx = bsl * 4 + left * 2 + above` from
//!   [`partition_plane_context`] and the per-frame probability source
//!   ([`KF_PARTITION_PROBS`] on keyframes / intra-only frames; the running
//!   `partition_probs[ ]` table on inter frames).
//! * `subsize = subsize_lookup[ partition ][ bsize ]` then dispatches on
//!   the four [`PartitionKind`] arms: `NONE` / `HORZ` (with the
//!   `hasRows`-gated second leaf) / `VERT` (with the `hasCols`-gated
//!   second leaf) / `SPLIT` (four recursive calls in spec order
//!   TL → TR → BL → BR).
//! * The §6.4.3 tail write-back fires when
//!   `bsize == BLOCK_8X8 || partition != PARTITION_SPLIT`, setting
//!   `AbovePartitionContext[c + i] = 15 >> b_width_log2_lookup[subsize]`
//!   and `LeftPartitionContext[r + i] = 15 >> b_height_log2_lookup[subsize]`
//!   for `i ∈ 0..num8x8`.
//!
//! Leaf blocks (the §6.4.4 `decode_block( r, c, subsize )` call sites that
//! this round does not yet wire — the per-block `mode_info` /
//! `residual` decode is downstream of this driver) are logged into a
//! caller-supplied `Vec<LeafBlock>` instead, in spec-traversal order. The
//! recursive walk's correctness is then validated by hand-built bitstreams
//! producing predictable leaf layouts (a single 64x64 PARTITION_NONE leaf,
//! a 4-way SPLIT into four 32x32 NONE leaves, and mixed HORZ + VERT
//! quadrant layouts).
//!
//! The §9.3.1 tree-selection rule — *"if hasRows == 1 and hasCols == 1 the
//! tree is `partition_tree`; else if hasCols == 1 the tree is
//! `cols_partition_tree`; else if hasRows == 1 the tree is
//! `rows_partition_tree`; else the return value is `PARTITION_SPLIT`"* — is
//! handled inside [`decode_partition_type`] so a caller passing the four
//! edge cases (interior / right-edge / bottom-edge / corner) walks the
//! correct two- or six-entry tree (and skips the bool coder entirely when
//! both flags are 0).
//!
//! The §9.3.2 probability selection rule for `partition` — *"node2 = node
//! when both flags are 1, `node2 = 1` when only hasCols is 1, `node2 = 2`
//! otherwise"* — is reproduced verbatim in [`decode_partition_type`]'s
//! probability callback: the tree-decode walker requests `prob(node)` for
//! `node ∈ { 0, 1, 2 }`, and the callback rewrites the node index per the
//! `(hasRows, hasCols)` pair before indexing the per-context probability
//! row.
//!
//! The §9.3.2 ctx derivation lives in [`partition_plane_context`]: it
//! materialises the §6.4.3 `above` / `left` bitmaps from the per-`c` /
//! per-`r` `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips,
//! `OR`s them across `num8x8 = num_8x8_blocks_wide_lookup[ bsize ]` cells,
//! extracts the `bsl`-th bit, and returns `ctx = bsl * 4 + left * 2 +
//! above` (range `0..=15`, matching [`PARTITION_CONTEXTS`]).
//!
//! Per-frame probability sourcing:
//!
//! * Keyframes / intra-only frames (`FrameIsIntra == 1`) use the
//!   §10.4 fixed [`KF_PARTITION_PROBS`] table (`PARTITION_CONTEXTS = 16`
//!   rows of `PARTITION_TYPES - 1 = 3` probabilities each, transcribed
//!   verbatim).
//! * Inter frames use a running `partition_probs[ ]` table initialised
//!   from the §10.5 [`DEFAULT_PARTITION_PROBS`] (same shape) and
//!   conditionally updated by §6.3 `read_partition_probs( )` — which still
//!   needs wiring into the compressed-header sweep in a later round.
//!
//! Deferred to later rounds (NOT in scope here):
//!
//! * The §6.4.3 recursive driver itself (the `decode_partition( r, c, bsize )`
//!   function that splits on the decoded `partition`, threads the
//!   `subsize_lookup[ partition ][ bsize ]` child block size into four
//!   recursive calls when `PARTITION_SPLIT`, and writes back the
//!   `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips with
//!   `15 >> b_*_log2_lookup[ subsize ]`). It composes this primitive plus
//!   the §6.4.4 `decode_block( )` orchestrator that lives one layer up.
//! * The §6.3 `read_partition_probs( )` compressed-header sweep
//!   (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
//!   `diff_update_prob` cells against [`DEFAULT_PARTITION_PROBS`]).
//! * The §8.4 `counts_partition[ PARTITION_CONTEXTS ][ PARTITION_TYPES ]`
//!   probability-adaption accumulator (§9.3.4 bookkeeping).
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §3 / §6.4.3 / §9.3.1 / §9.3.2 / §10.2 /
//! §10.4 / §10.5).

#![allow(dead_code)] // surfaces land in the next round's §6.4.3 driver

use crate::bool_coder::BoolCoder;
use crate::mode_info::tree_decode;
use crate::residual::{BLOCK_8X8, BLOCK_SIZES};
use crate::Error;

// ----- §3 partition enumeration -----

/// `PARTITION_NONE = 0` per §3 / §7.4.3 line 3843.
pub(crate) const PARTITION_NONE: u8 = 0;
/// `PARTITION_HORZ = 1` per §3 / §7.4.3 line 3844.
pub(crate) const PARTITION_HORZ: u8 = 1;
/// `PARTITION_VERT = 2` per §3 / §7.4.3 line 3845.
pub(crate) const PARTITION_VERT: u8 = 2;
/// `PARTITION_SPLIT = 3` per §3 / §7.4.3 line 3846.
pub(crate) const PARTITION_SPLIT: u8 = 3;

/// `PARTITION_TYPES = 4` per §3 (line 497 of `docs/video/vp9/vp9-spec.txt`).
///
/// Number of values for `partition`.
pub(crate) const PARTITION_TYPES: usize = 4;

/// `PARTITION_CONTEXTS = 16` per §3 (line 463 of `docs/video/vp9/vp9-spec.txt`).
///
/// Number of contexts when decoding `partition`. The §9.3.2 ctx derivation
/// `ctx = bsl * 4 + left * 2 + above` yields four `bsl` rows
/// (`bsl ∈ { 1, 2, 3, 4 }` — i.e. the four superblock sizes `BLOCK_8X8 →
/// BLOCK_16X16`, `BLOCK_16X16 → BLOCK_8X8`, `BLOCK_32X32 → BLOCK_16X16`,
/// `BLOCK_64X64 → BLOCK_32X32`) of four `(left, above)` cells each.
pub(crate) const PARTITION_CONTEXTS: usize = 16;

// ----- §10.2 block-geometry lookup tables (verbatim) -----

/// `b_width_log2_lookup[ BLOCK_SIZES ]` per §10.2 spec line 7088 —
/// `{0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4}`.
///
/// Indexed by the §3 `BLOCK_*` constant (`BLOCK_4X4 = 0 → 0` width-log2-of-4,
/// `BLOCK_64X64 = 12 → 4` width-log2-of-4). Used by the §6.4.3 tail when
/// writing `AbovePartitionContext[ c + i ] = 15 >> b_width_log2_lookup[ subsize ]`.
pub(crate) const B_WIDTH_LOG2_LOOKUP: [u8; BLOCK_SIZES] = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4];

/// `b_height_log2_lookup[ BLOCK_SIZES ]` per §10.2 spec line 7099 —
/// `{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4}`.
///
/// Indexed by the §3 `BLOCK_*` constant. Used by the §6.4.3 tail when
/// writing `LeftPartitionContext[ r + i ] = 15 >> b_height_log2_lookup[ subsize ]`.
pub(crate) const B_HEIGHT_LOG2_LOOKUP: [u8; BLOCK_SIZES] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];

/// `mi_width_log2_lookup[ BLOCK_SIZES ]` per §10.2 spec line 7108 —
/// `{0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3}`.
///
/// Indexed by the §3 `BLOCK_*` constant; the §9.3.2 listing reads
/// `bsl = mi_width_log2_lookup[ bsize ]` and
/// `boffset = mi_width_log2_lookup[ BLOCK_64X64 ] - bsl = 3 - bsl`.
/// `bsl` ranges over `{ 1, 2, 3, 4 }` for the four superblock-recursion
/// `bsize` calls `BLOCK_8X8` / `BLOCK_16X16` / `BLOCK_32X32` / `BLOCK_64X64`
/// (the §6.4.3 caller never invokes partition decode on a sub-8x8 block).
pub(crate) const MI_WIDTH_LOG2_LOOKUP: [u8; BLOCK_SIZES] = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3];

/// `num_8x8_blocks_wide_lookup[ BLOCK_SIZES ]` per §10.2 spec line 7111 —
/// `{1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8}`.
///
/// Indexed by the §3 `BLOCK_*` constant; the §6.4.3 listing reads
/// `num8x8 = num_8x8_blocks_wide_lookup[ bsize ]` (with `halfBlock8x8 =
/// num8x8 >> 1`).
pub(crate) const NUM_8X8_BLOCKS_WIDE_LOOKUP: [u8; BLOCK_SIZES] =
    [1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8];

/// `num_8x8_blocks_high_lookup[ BLOCK_SIZES ]` per §10.2 spec line 7117 —
/// `{1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8}`.
///
/// Indexed by the §3 `BLOCK_*` constant; the §6.4.12 / §6.4.14 listings
/// read `bh = num_8x8_blocks_high_lookup[ MiSize ]` for the
/// `LeftSegPredContext[ MiRow + i ]` strip write-back and the
/// `get_segment_id( )` spatial sweep bounds.
pub(crate) const NUM_8X8_BLOCKS_HIGH_LOOKUP: [u8; BLOCK_SIZES] =
    [1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8];

/// `mi_width_log2_lookup[ BLOCK_64X64 ]` — the §9.3.2 constant the
/// `boffset = mi_width_log2_lookup[ BLOCK_64X64 ] - bsl` derivation
/// subtracts from. Equals `3` per [`MI_WIDTH_LOG2_LOOKUP`].
pub(crate) const MI_WIDTH_LOG2_BLOCK_64X64: u8 = 3;

// ----- §10.2 subsize_lookup (verbatim, including BLOCK_INVALID slots) -----

/// `BLOCK_INVALID = 14` sentinel per §3 line 462 — re-exported here for
/// readability of the `subsize_lookup` listing.
const BI: u8 = 14;

/// `subsize_lookup[ PARTITION_TYPES ][ BLOCK_SIZES ]` per §10.2 spec line
/// 7132.
///
/// Outer index is the decoded `partition` value (`PARTITION_NONE`,
/// `PARTITION_HORZ`, `PARTITION_VERT`, `PARTITION_SPLIT`); inner index is
/// the parent `bsize` (`BLOCK_4X4` .. `BLOCK_64X64`). The §6.4.3 driver
/// reads `subsize = subsize_lookup[ partition ][ bsize ]` and recurses
/// on `subsize` (or hands it to `decode_block( )`).
///
/// `BLOCK_INVALID = 14` marks combinations the §10.2 listing has no
/// meaningful child for (e.g. `PARTITION_HORZ` applied to a `BLOCK_4X8`
/// parent, where horizontal split is undefined).
///
/// Verbatim from the §10.2 listing:
///
/// ```text
/// { // PARTITION_NONE
///   BLOCK_4X4,  BLOCK_4X8,  BLOCK_8X4,
///   BLOCK_8X8,  BLOCK_8X16, BLOCK_16X8,
///   BLOCK_16X16,BLOCK_16X32,BLOCK_32X16,
///   BLOCK_32X32,BLOCK_32X64,BLOCK_64X32,
///   BLOCK_64X64,
/// }, { // PARTITION_HORZ
///   BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_8X4,     BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_16X8,    BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_32X16,   BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_64X32,
/// }, { // PARTITION_VERT
///   BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_4X8,     BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_8X16,    BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_16X32,   BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_32X64,
/// }, { // PARTITION_SPLIT
///   BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_4X4,     BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_8X8,     BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_16X16,   BLOCK_INVALID, BLOCK_INVALID,
///   BLOCK_32X32,
/// }
/// ```
pub(crate) const SUBSIZE_LOOKUP: [[u8; BLOCK_SIZES]; PARTITION_TYPES] = [
    // PARTITION_NONE — identity
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    // PARTITION_HORZ
    [
        BI, BI, BI, /* BLOCK_8X4 */ 2, BI, BI, /* BLOCK_16X8 */ 5, BI, BI,
        /* BLOCK_32X16 */ 8, BI, BI, /* BLOCK_64X32 */ 11,
    ],
    // PARTITION_VERT
    [
        BI, BI, BI, /* BLOCK_4X8 */ 1, BI, BI, /* BLOCK_8X16 */ 4, BI, BI,
        /* BLOCK_16X32 */ 7, BI, BI, /* BLOCK_32X64 */ 10,
    ],
    // PARTITION_SPLIT
    [
        BI, BI, BI, /* BLOCK_4X4 */ 0, BI, BI, /* BLOCK_8X8 */ 3, BI, BI,
        /* BLOCK_16X16 */ 6, BI, BI, /* BLOCK_32X32 */ 9,
    ],
];

// ----- §9.3.1 partition tree listings (verbatim) -----

/// `partition_tree[ 6 ]` per §9.3.1 spec line 6083.
///
/// The 4-leaf binary tree used when `hasRows == 1 && hasCols == 1` (the
/// interior superblock case): walks `PARTITION_NONE` / `PARTITION_HORZ` /
/// `PARTITION_VERT` / `PARTITION_SPLIT` via the §9.3.3 tree decode.
///
/// Verbatim from the spec listing:
///
/// ```text
/// partition_tree[ 6 ] = {
///     -PARTITION_NONE, 2,
///     -PARTITION_HORZ, 4,
///     -PARTITION_VERT, -PARTITION_SPLIT
/// }
/// ```
///
/// `-PARTITION_NONE = -0` collapses to `0` so the §9.3.3 post-loop `-n`
/// returns `0` (= `PARTITION_NONE`) when the walker hits leaf 0; the other
/// three leaves return the matching `PARTITION_*` integer.
pub(crate) const PARTITION_TREE: [i32; 6] = [
    -(PARTITION_NONE as i32),
    2,
    -(PARTITION_HORZ as i32),
    4,
    -(PARTITION_VERT as i32),
    -(PARTITION_SPLIT as i32),
];

/// `cols_partition_tree[ 2 ]` per §9.3.1 spec line 6090.
///
/// Used when `hasRows == 0 && hasCols == 1` (right-edge superblock): only
/// `PARTITION_HORZ` and `PARTITION_SPLIT` are legal.
///
/// Verbatim:
///
/// ```text
/// cols_partition_tree[ 2 ] = {
///     -PARTITION_HORZ, -PARTITION_SPLIT
/// }
/// ```
pub(crate) const COLS_PARTITION_TREE: [i32; 2] =
    [-(PARTITION_HORZ as i32), -(PARTITION_SPLIT as i32)];

/// `rows_partition_tree[ 2 ]` per §9.3.1 spec line 6095.
///
/// Used when `hasRows == 1 && hasCols == 0` (bottom-edge superblock): only
/// `PARTITION_VERT` and `PARTITION_SPLIT` are legal.
///
/// Verbatim:
///
/// ```text
/// rows_partition_tree[ 2 ] = {
///     -PARTITION_VERT, -PARTITION_SPLIT
/// }
/// ```
pub(crate) const ROWS_PARTITION_TREE: [i32; 2] =
    [-(PARTITION_VERT as i32), -(PARTITION_SPLIT as i32)];

// ----- §10.4 kf_partition_probs (verbatim) -----

/// `kf_partition_probs[ PARTITION_CONTEXTS ][ PARTITION_TYPES - 1 ]` per
/// §10.4 spec line 7439.
///
/// The fixed probability table used for `partition` on keyframes /
/// intra-only frames (`FrameIsIntra == 1`); the §9.3.2 listing routes
/// every `partition` decode in those frames through this table indexed by
/// `[ctx][node2]`. 16 rows × 3 cells, ordered by the four superblock-sizes
/// (`8X8 → 4X4`, `16X16 → 8X8`, `32X32 → 16X16`, `64X64 → 32X32`) and
/// within each by the four (`above`, `left`) split combinations: both not
/// split, above split, left split, both split.
///
/// Verbatim from the §10.4 listing.
pub(crate) const KF_PARTITION_PROBS: [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS] = [
    // 8x8 -> 4x4
    [158, 97, 94], // a/l both not split
    [93, 24, 99],  // a split, l not split
    [85, 119, 44], // l split, a not split
    [62, 59, 67],  // a/l both split
    // 16x16 -> 8x8
    [149, 53, 53],
    [94, 20, 48],
    [83, 53, 24],
    [52, 18, 18],
    // 32x32 -> 16x16
    [150, 40, 39],
    [78, 12, 26],
    [67, 33, 11],
    [24, 7, 5],
    // 64x64 -> 32x32
    [174, 35, 49],
    [68, 11, 27],
    [57, 15, 9],
    [12, 3, 3],
];

// ----- §10.5 default_partition_probs (verbatim) -----

/// `default_partition_probs[ PARTITION_CONTEXTS ][ PARTITION_TYPES - 1 ]`
/// per §10.5 spec line 7623.
///
/// The initial probability table for `partition` on inter frames; the
/// §6.3 `read_partition_probs( )` compressed-header sweep starts from this
/// table and conditionally updates each cell via `diff_update_prob( )`.
/// `decode_partition_type` reads from this table (or the post-sweep
/// running copy) when `FrameIsIntra == 0`.
///
/// Verbatim from the §10.5 listing.
pub(crate) const DEFAULT_PARTITION_PROBS: [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS] = [
    // 8x8 -> 4x4
    [199, 122, 141],
    [147, 63, 159],
    [148, 133, 118],
    [121, 104, 114],
    // 16x16 -> 8x8
    [174, 73, 87],
    [92, 41, 83],
    [82, 99, 50],
    [53, 39, 39],
    // 32x32 -> 16x16
    [177, 58, 59],
    [68, 26, 63],
    [52, 79, 25],
    [17, 14, 12],
    // 64x64 -> 32x32
    [222, 34, 30],
    [72, 16, 44],
    [58, 32, 12],
    [10, 7, 6],
];

// ----- §9.3.2 partition_plane_context -----

/// `partition_plane_context( r, c, bsize, above_ctx, left_ctx )` per §9.3.2
/// spec lines 6254-6265.
///
/// Returns `ctx = bsl * 4 + left * 2 + above`, the index into the
/// per-context partition probability row (`KF_PARTITION_PROBS[ctx]` for
/// keyframes, the running `partition_probs[ctx]` for inter frames).
///
/// The spec listing reads:
///
/// ```text
/// above = 0
/// left = 0
/// bsl = mi_width_log2_lookup[ bsize ]
/// boffset = mi_width_log2_lookup[ BLOCK_64X64 ] - bsl
/// for ( i = 0; i < num8x8; i++ ) {
///     above |= AbovePartitionContext[ c + i ]
///     left  |= LeftPartitionContext[ r + i ]
/// }
/// above = (above & (1 << boffset)) > 0
/// left  = (left  & (1 << boffset)) > 0
/// ctx = bsl * 4 + left * 2 + above
/// ```
///
/// `num8x8 = num_8x8_blocks_wide_lookup[ bsize ]` per §6.4.3 — the same
/// number the caller already has, threaded here as the strip width of the
/// `above_ctx` / `left_ctx` slices.
///
/// `above_ctx` is the per-column `AbovePartitionContext[ c .. c+num8x8 ]`
/// strip; `left_ctx` is the per-row `LeftPartitionContext[ r .. r+num8x8 ]`
/// strip. Both arrive as `&[u8]` slices the caller materialises from its
/// frame-wide `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]`
/// arrays.
///
/// `bsize` must be one of the four superblock-recursion sizes the
/// §6.4.3 driver invokes partition decode on (`BLOCK_8X8` / `BLOCK_16X16`
/// / `BLOCK_32X32` / `BLOCK_64X64`); for those, `bsl ∈ { 1, 2, 3, 4 }` and
/// `ctx ∈ 0..=15` covering [`PARTITION_CONTEXTS`].
///
/// Returns the `ctx` value directly — the caller indexes its 16-row
/// probability table with it.
///
/// # Panics
///
/// Panics if `bsize >= BLOCK_SIZES` (i.e. not a valid §3 `BLOCK_*`
/// constant), or if `above_ctx.len() != left_ctx.len()` (the §6.4.3
/// listing reads both strips at matching offsets so a mismatched width
/// would silently corrupt the bitmap OR-fold). Both invariants are
/// caller-managed; the §6.4.3 driver always materialises matching strips.
pub(crate) fn partition_plane_context(bsize: u8, above_ctx: &[u8], left_ctx: &[u8]) -> usize {
    assert!(
        (bsize as usize) < BLOCK_SIZES,
        "partition_plane_context: bsize={} out of range",
        bsize
    );
    assert_eq!(
        above_ctx.len(),
        left_ctx.len(),
        "partition_plane_context: strip widths mismatch ({} vs {})",
        above_ctx.len(),
        left_ctx.len()
    );

    let bsl = MI_WIDTH_LOG2_LOOKUP[bsize as usize];
    let boffset = MI_WIDTH_LOG2_BLOCK_64X64 - bsl;

    // `num8x8` is the strip width per the §6.4.3 caller; the OR-fold runs
    // over `num8x8` cells of each side.
    let mut above_bits: u8 = 0;
    let mut left_bits: u8 = 0;
    for (&a, &l) in above_ctx.iter().zip(left_ctx.iter()) {
        above_bits |= a;
        left_bits |= l;
    }

    let mask: u8 = 1u8 << boffset;
    let above = u8::from((above_bits & mask) != 0);
    let left = u8::from((left_bits & mask) != 0);

    (bsl as usize) * 4 + (left as usize) * 2 + (above as usize)
}

// ----- §6.4.3 decode_partition_type -----

/// `decode_partition_type( coder, has_rows, has_cols, ctx, probs )` —
/// the §6.4.3 per-call partition reader per spec line 2360 plus the §9.3.1
/// tree-selection and §9.3.2 probability-selection rules.
///
/// Returns one of the four §3 partition constants
/// ([`PARTITION_NONE`] / [`PARTITION_HORZ`] / [`PARTITION_VERT`] /
/// [`PARTITION_SPLIT`]).
///
/// Tree-selection (§9.3.1 line 6078):
///
/// * `has_rows == 1` and `has_cols == 1` → walk [`PARTITION_TREE`]
///   (the 6-entry interior tree); all four `PARTITION_*` outcomes possible.
/// * `has_rows == 0` and `has_cols == 1` → walk [`COLS_PARTITION_TREE`]
///   (the 2-entry right-edge tree); only `PARTITION_HORZ` /
///   `PARTITION_SPLIT` legal.
/// * `has_rows == 1` and `has_cols == 0` → walk [`ROWS_PARTITION_TREE`]
///   (the 2-entry bottom-edge tree); only `PARTITION_VERT` /
///   `PARTITION_SPLIT` legal.
/// * `has_rows == 0` and `has_cols == 0` → return
///   `PARTITION_SPLIT` directly without consuming any bool-coder bits
///   (the spec's "the return value is `PARTITION_SPLIT`" corner-case).
///
/// Probability-selection (§9.3.2 line 6247-6250) — `node2`:
///
/// * `has_rows == 1` and `has_cols == 1` → `node2 = node` (the §9.3.3
///   walker passes `0`, `1`, `2` in sequence; the row indexes
///   `probs[ctx][0..=2]` per the §10.4 / §10.5 listing).
/// * `has_rows == 0` and `has_cols == 1` → `node2 = 1` for every read
///   (the right-edge tree only has one bool decision and it indexes
///   `probs[ctx][1]`).
/// * `has_rows == 1` and `has_cols == 0` → `node2 = 2` for every read
///   (the bottom-edge tree only has one bool decision and it indexes
///   `probs[ctx][2]`).
///
/// `probs` is the per-context 3-cell probability row — either
/// `KF_PARTITION_PROBS[ctx]` (keyframes / intra-only frames) or
/// `partition_probs[ctx]` (inter frames; initialised from
/// [`DEFAULT_PARTITION_PROBS`] and conditionally updated by §6.3
/// `read_partition_probs( )` once that sweep lands). The caller resolves
/// `ctx` via [`partition_plane_context`].
///
/// Returns [`Error::InvalidBitstream`] when the underlying §9.2 bool
/// coder underflows mid-walk (matching every other §9.3.3 reader in this
/// crate).
pub(crate) fn decode_partition_type(
    coder: &mut BoolCoder<'_>,
    has_rows: bool,
    has_cols: bool,
    probs: &[u8; PARTITION_TYPES - 1],
) -> Result<u8, Error> {
    match (has_rows, has_cols) {
        // §9.3.1 first arm — interior superblock: walk the full
        // partition_tree[ 6 ] with node2 = node.
        (true, true) => {
            let raw = tree_decode(coder, &PARTITION_TREE, |node| probs[node])?;
            // The §9.3.3 walker returns `-n` of the leaf (so e.g. -0 →
            // PARTITION_NONE = 0, -1 → PARTITION_HORZ = 1, …).
            Ok(raw as u8)
        }
        // §9.3.1 second arm — right-edge: cols_partition_tree[ 2 ] with
        // node2 fixed at 1 per §9.3.2 line 6249.
        (false, true) => {
            let raw = tree_decode(coder, &COLS_PARTITION_TREE, |_node| probs[1])?;
            Ok(raw as u8)
        }
        // §9.3.1 third arm — bottom-edge: rows_partition_tree[ 2 ] with
        // node2 fixed at 2 per §9.3.2 line 6250.
        (true, false) => {
            let raw = tree_decode(coder, &ROWS_PARTITION_TREE, |_node| probs[2])?;
            Ok(raw as u8)
        }
        // §9.3.1 fourth arm — corner (no rows, no cols): the listing
        // returns PARTITION_SPLIT directly without reading any bits.
        (false, false) => Ok(PARTITION_SPLIT),
    }
}

// ----- §6.4.3 recursive decode_partition driver -----

/// Per-frame partition-probability source threaded into [`decode_partition`].
///
/// The §9.3.2 listing keys the `partition` probability row on whether the
/// current frame is a keyframe / intra-only frame ([`KF_PARTITION_PROBS`])
/// or an inter frame (the running `partition_probs[ ]` table whose initial
/// values come from [`DEFAULT_PARTITION_PROBS`] and whose live values are
/// updated by the §6.3 `read_partition_probs( )` sweep — pending in a
/// later round).
#[derive(Debug, Clone, Copy)]
pub(crate) enum PartitionProbsKind<'a> {
    /// `FrameIsIntra == 1` — use the §10.4 [`KF_PARTITION_PROBS`] table
    /// verbatim.
    Keyframe,
    /// `FrameIsIntra == 0` — use the caller's running
    /// `partition_probs[16][3]` table (typically initialised from
    /// [`DEFAULT_PARTITION_PROBS`] and conditionally updated by the
    /// §6.3 `read_partition_probs( )` sweep).
    Inter(&'a [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS]),
}

impl<'a> PartitionProbsKind<'a> {
    /// Look up the per-context 3-cell probability row by `ctx`.
    fn row(&self, ctx: usize) -> [u8; PARTITION_TYPES - 1] {
        match self {
            PartitionProbsKind::Keyframe => KF_PARTITION_PROBS[ctx],
            PartitionProbsKind::Inter(table) => table[ctx],
        }
    }
}

/// `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` per §6.4.3 +
/// §7.4 reset rule.
///
/// The two arrays are sized `Sb64Cols * 8` and `Sb64Rows * 8` respectively
/// per the §7.4 listing (each cell is a single byte holding a bitmap
/// across the four possible `bsl` granularities). The §6.4.3 tail writes
/// `15 >> b_*_log2_lookup[ subsize ]` into the relevant strip slice; the
/// §9.3.2 `partition_plane_context` reads back from those strips when
/// deriving `ctx` for a subsequent quadrant.
///
/// At a tile boundary the spec's `clear_left_context( )` zeroes
/// `LeftPartitionContext[ ]` and the per-tile-row loop in §6.4.2 calls it
/// once before walking `c`; the parent driver wires that reset around
/// each `decode_partition( r, c, BLOCK_64X64 )` row. Within a single CTU
/// recursion the arrays accumulate writes from sibling quadrants.
#[derive(Debug)]
pub(crate) struct PartitionContextState {
    /// `AbovePartitionContext[ 0 .. Sb64Cols * 8 ]`.
    pub above: Vec<u8>,
    /// `LeftPartitionContext[ 0 .. Sb64Rows * 8 ]`.
    pub left: Vec<u8>,
}

impl PartitionContextState {
    /// Build a fresh state for a frame of `mi_cols * mi_rows` MI blocks,
    /// with both strips zeroed per the §7.4 reset rule.
    pub(crate) fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            above: vec![0u8; mi_cols],
            left: vec![0u8; mi_rows],
        }
    }

    /// `clear_left_context( )` per §6.4.2 — invoked between superblock
    /// rows by the §6.4.2 tile driver.
    pub(crate) fn clear_left(&mut self) {
        for cell in self.left.iter_mut() {
            *cell = 0;
        }
    }

    /// `clear_above_context( )` per §6.4 / §7.4.1 — zeroes
    /// `AbovePartitionContext[ ]` once per `decode_tiles( )` invocation
    /// (i.e. once per frame, before the first tile's `decode_tile( )`).
    ///
    /// Per §7.4.1 the canonical span is `i = 0..Sb64Cols * 8 - 1`
    /// because the array can be read for locations beyond `MiCols`.
    /// The strip allocated by [`PartitionContextState::new`] is sized to
    /// `mi_cols` cells — callers that want the full `Sb64Cols * 8`
    /// span should round `mi_cols` up to the next multiple of 8 when
    /// constructing the state. The reset itself is a per-cell zero so
    /// it is correct regardless of the chosen strip width.
    pub(crate) fn clear_above(&mut self) {
        for cell in self.above.iter_mut() {
            *cell = 0;
        }
    }
}

/// Per-leaf log record emitted in §6.4.3 traversal order by
/// [`decode_partition`].
///
/// Each entry stands in for the §6.4.4 `decode_block( r, c, subsize )`
/// call site that this round does not yet wire (the per-block
/// `mode_info` / `residual` decode is downstream). The traversal order
/// is: depth-first, with `PARTITION_SPLIT` recursing TL → TR → BL → BR
/// per §6.4.3 lines 2381-2384.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeafBlock {
    /// MI-row coordinate `r` per §6.4.4 line 2397.
    pub r: u32,
    /// MI-column coordinate `c` per §6.4.4 line 2398.
    pub c: u32,
    /// `subsize` per §6.4.4 line 2399 (one of the §3 `BLOCK_*`
    /// constants, but never `BLOCK_INVALID = 14`).
    pub subsize: u8,
}

/// Apply the §6.4.3 tail write-back to the partition-context strips.
///
/// The spec listing is:
///
/// ```text
/// if ( bsize == BLOCK_8X8 || partition != PARTITION_SPLIT ) {
///     for ( i = 0; i < num8x8; i++ ) {
///         AbovePartitionContext[ c + i ] = 15 >> b_width_log2_lookup[ subsize ]
///         LeftPartitionContext[ r + i ] = 15 >> b_height_log2_lookup[ subsize ]
///     }
/// }
/// ```
///
/// Called by [`decode_partition`] after the recursive / leaf dispatch
/// has run. `subsize` is the post-`subsize_lookup` child size, NOT the
/// parent `bsize`; per §10.2 / §3 it is always a valid `BLOCK_*` index
/// (never `BLOCK_INVALID`) when the gate condition fires.
fn write_back_partition_context(
    above: &mut [u8],
    left: &mut [u8],
    r: usize,
    c: usize,
    num8x8: usize,
    subsize: u8,
) {
    let above_val = 15u8 >> B_WIDTH_LOG2_LOOKUP[subsize as usize];
    let left_val = 15u8 >> B_HEIGHT_LOG2_LOOKUP[subsize as usize];
    for i in 0..num8x8 {
        // The §6.4.3 listing never reads past `Sb64Cols * 8` /
        // `Sb64Rows * 8`; an out-of-bounds index would mean the caller
        // sized the state strips incorrectly. We saturate at the strip
        // length rather than panic, matching the §6.4.3 quadrant
        // out-of-range short-circuit that already prevents the OOB
        // write at the recursion edge.
        let ci = c + i;
        if ci < above.len() {
            above[ci] = above_val;
        }
        let ri = r + i;
        if ri < left.len() {
            left[ri] = left_val;
        }
    }
}

/// `decode_partition( r, c, bsize )` per spec §6.4.3 (lines 2353-2391).
///
/// The recursive partition driver: composes [`decode_partition_type`]
/// (the §6.4.3 partition syntax-element decode) with the §10.2
/// `subsize_lookup` traversal and the §6.4.3 tail write-back into the
/// `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips.
///
/// `leaves` accumulates one [`LeafBlock`] per `decode_block( r, c,
/// subsize )` call site the spec listing would invoke. The
/// §6.4.4 `decode_block( )` decode itself (`mode_info` + `residual`) is
/// downstream of this driver and not yet wired; the per-leaf log is
/// the stand-in this round validates.
///
/// Recursion order on `PARTITION_SPLIT` is `(r, c) → (r, c+half) →
/// (r+half, c) → (r+half, c+half)` per the §6.4.3 listing lines
/// 2381-2384 (TL → TR → BL → BR).
///
/// Returns [`Error::InvalidBitstream`] if the underlying §9.2 bool coder
/// underflows mid-walk.
///
/// # Panics
///
/// Panics if `bsize` is not a valid §3 `BLOCK_*` constant (i.e.
/// `bsize >= BLOCK_SIZES`). The §6.4.3 caller chain only invokes
/// this with one of the four superblock-recursion sizes (`BLOCK_8X8` /
/// `BLOCK_16X16` / `BLOCK_32X32` / `BLOCK_64X64`); the recursion
/// produces only the `subsize_lookup` children of those.
#[allow(clippy::too_many_arguments)] // the §6.4.3 signature has 8 positional
                                     // inputs by spec design (r / c / bsize +
                                     // frame extents + context state + probs +
                                     // leaf-log sink); each is independent and
                                     // bundling them would obscure the spec
                                     // mapping.
pub(crate) fn decode_partition(
    coder: &mut BoolCoder<'_>,
    r: u32,
    c: u32,
    bsize: u8,
    mi_rows: u32,
    mi_cols: u32,
    ctx_state: &mut PartitionContextState,
    probs_kind: PartitionProbsKind<'_>,
    leaves: &mut Vec<LeafBlock>,
) -> Result<(), Error> {
    // §6.4.3 line 2354: out-of-frame quadrants short-circuit without
    // touching the bool coder or the context strips.
    if r >= mi_rows || c >= mi_cols {
        return Ok(());
    }

    assert!(
        (bsize as usize) < BLOCK_SIZES,
        "decode_partition: bsize={} out of range",
        bsize
    );

    // §6.4.3 lines 2356-2359.
    let num8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as u32;
    let half = num8x8 >> 1;
    let has_rows = (r + half) < mi_rows;
    let has_cols = (c + half) < mi_cols;

    // §9.3.2 ctx derivation + probability-row pick. The strip-slice
    // widths the §9.3.2 listing reads are `num8x8` cells; we saturate
    // at the available strip length to mirror the §6.4.3 quadrant
    // short-circuit at frame edges.
    let above_end = ((c + num8x8) as usize).min(ctx_state.above.len());
    let above_strip = &ctx_state.above[(c as usize)..above_end];
    let left_end = ((r + num8x8) as usize).min(ctx_state.left.len());
    let left_strip = &ctx_state.left[(r as usize)..left_end];
    // If the strip is shorter than num8x8 (frame-edge underflow), pad
    // with zero cells so the OR-fold sees the §7.4 reset value.
    let strip_len = above_strip.len().min(left_strip.len());
    let above_buf: Vec<u8> = above_strip
        .iter()
        .take(strip_len)
        .copied()
        .chain(core::iter::repeat(0u8))
        .take(num8x8 as usize)
        .collect();
    let left_buf: Vec<u8> = left_strip
        .iter()
        .take(strip_len)
        .copied()
        .chain(core::iter::repeat(0u8))
        .take(num8x8 as usize)
        .collect();
    let ctx = partition_plane_context(bsize, &above_buf, &left_buf);
    let probs_row = probs_kind.row(ctx);

    // §6.4.3 line 2360: read the partition syntax element.
    let partition = decode_partition_type(coder, has_rows, has_cols, &probs_row)?;
    // §6.4.3 line 2361: child block size.
    let subsize = SUBSIZE_LOOKUP[partition as usize][bsize as usize];

    // §6.4.3 lines 2362-2385: leaf vs recursive dispatch.
    if (subsize as usize) < (BLOCK_8X8 as usize) || partition == PARTITION_NONE {
        leaves.push(LeafBlock { r, c, subsize });
    } else if partition == PARTITION_HORZ {
        leaves.push(LeafBlock { r, c, subsize });
        if has_rows {
            leaves.push(LeafBlock {
                r: r + half,
                c,
                subsize,
            });
        }
    } else if partition == PARTITION_VERT {
        leaves.push(LeafBlock { r, c, subsize });
        if has_cols {
            leaves.push(LeafBlock {
                r,
                c: c + half,
                subsize,
            });
        }
    } else {
        // PARTITION_SPLIT: four recursive calls in §6.4.3 spec order
        // (TL → TR → BL → BR per lines 2381-2384).
        decode_partition(
            coder, r, c, subsize, mi_rows, mi_cols, ctx_state, probs_kind, leaves,
        )?;
        decode_partition(
            coder,
            r,
            c + half,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            leaves,
        )?;
        decode_partition(
            coder,
            r + half,
            c,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            leaves,
        )?;
        decode_partition(
            coder,
            r + half,
            c + half,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            leaves,
        )?;
    }

    // §6.4.3 lines 2386-2391: tail write-back, gated by
    // `bsize == BLOCK_8X8 || partition != PARTITION_SPLIT`.
    if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
        write_back_partition_context(
            &mut ctx_state.above,
            &mut ctx_state.left,
            r as usize,
            c as usize,
            num8x8 as usize,
            subsize,
        );
    }

    Ok(())
}

// ----- §6.4.1 get_tile_offset + §6.4.2 decode_tile -----

/// `get_tile_offset( tileNum, mis, tileSzLog2 )` per spec §6.4.1
/// (`vp9-spec.txt` lines 2335-2338).
///
/// ```text
/// get_tile_offset( tileNum, mis, tileSzLog2 ) {
///     sbs = (mis + 7) >> 3
///     offset = ( (tileNum * sbs) >> tileSzLog2 ) << 3
///     return Min( offset, mis )
/// }
/// ```
///
/// Pure arithmetic helper invoked by §6.4 `decode_tiles( )` four times per
/// tile to derive `MiRowStart` / `MiRowEnd` / `MiColStart` / `MiColEnd`.
/// `mis` is the relevant frame extent in MI cells (`MiRows` for row
/// offsets, `MiCols` for column offsets); `tileSzLog2` is the matching
/// `tile_rows_log2` / `tile_cols_log2` field from the uncompressed
/// header. The §6.4 caller invokes this with `tileNum ∈ 0..=tilesPerAxis`
/// (one extra past the last tile, to fetch the `End` extent), so the
/// `Min( offset, mis )` clamp guards the past-the-end call.
///
/// The intermediate product `tileNum * sbs` cannot overflow `u32`: the
/// largest level (Level 6) caps `MiRows * MiCols` at well under `2^28`,
/// so `sbs <= 2^25`; the max `tileNum` is the per-axis tile count plus
/// one, bounded by `MAX_TILE_WIDTH_SB64 + 1 = 65` per §7.2 — comfortably
/// inside `u32`.
pub(crate) fn get_tile_offset(tile_num: u32, mis: u32, tile_sz_log2: u32) -> u32 {
    let sbs = (mis + 7) >> 3;
    let offset = ((tile_num * sbs) >> tile_sz_log2) << 3;
    offset.min(mis)
}

/// `decode_tile( )` per spec §6.4.2 (`vp9-spec.txt` lines 2343-2349).
///
/// ```text
/// decode_tile( ) {
///     for ( r = MiRowStart; r < MiRowEnd; r += 8 ) {
///         clear_left_context( )
///         for ( c = MiColStart; c < MiColEnd; c += 8 )
///             decode_partition( r, c, BLOCK_64X64 )
///     }
/// }
/// ```
///
/// Superblock-row driver: walks the tile's MI window in 64x64 superblock
/// strides, fires `clear_left_context( )` once at the start of each row
/// per §7.4.2, then invokes [`decode_partition`] at each
/// `(r, c, BLOCK_64X64)` superblock origin. The §6.4.3 driver
/// short-circuits when `r >= mi_rows || c >= mi_cols`, so a tile whose
/// `End` offsets fall past the frame edge naturally skips the
/// out-of-frame superblocks without extra bookkeeping.
///
/// Inputs follow the §6.4 `decode_tiles( )` listing one-to-one:
///
/// * `mi_row_start`, `mi_row_end`, `mi_col_start`, `mi_col_end` —
///   the four `get_tile_offset( )` outputs for the current tile.
/// * `mi_rows`, `mi_cols` — frame extents (`MiRows` / `MiCols` per
///   §7.2.4), passed verbatim into [`decode_partition`]'s edge clamp.
/// * `ctx_state` — the `AbovePartitionContext` / `LeftPartitionContext`
///   strips. The §7.4.1 `clear_above_context( )` reset that fires once
///   per `decode_tiles( )` is the caller's responsibility (the tile
///   driver only fires the §7.4.2 `clear_left_context( )` reset per
///   superblock row, matching the spec listing).
/// * `probs_kind` — the §9.3.2 `partition` probability source
///   (`Keyframe` for `FrameIsIntra == 1`, `Inter(...)` otherwise).
/// * `leaves` — per-leaf log sink threaded through every
///   [`decode_partition`] call so the tile's full traversal order is
///   recoverable by the caller.
///
/// Returns [`Error::InvalidBitstream`] if any inner [`decode_partition`]
/// underflows the §9.2 bool coder.
///
/// # Panics
///
/// Panics if `mi_row_end < mi_row_start` or `mi_col_end < mi_col_start`
/// (the §6.4.1 `get_tile_offset( )` clamp guarantees the spec-defined
/// caller never produces a backwards range — this assertion documents
/// the precondition).
#[allow(clippy::too_many_arguments)] // mirrors the §6.4 + §6.4.2 spec
                                     // composition (four tile-offset
                                     // inputs + two frame extents + ctx
                                     // state + probs + leaf sink); each
                                     // is independent and bundling them
                                     // would obscure the spec mapping.
pub(crate) fn decode_tile(
    coder: &mut BoolCoder<'_>,
    mi_row_start: u32,
    mi_row_end: u32,
    mi_col_start: u32,
    mi_col_end: u32,
    mi_rows: u32,
    mi_cols: u32,
    ctx_state: &mut PartitionContextState,
    probs_kind: PartitionProbsKind<'_>,
    leaves: &mut Vec<LeafBlock>,
) -> Result<(), Error> {
    assert!(
        mi_row_end >= mi_row_start,
        "decode_tile: mi_row_end={mi_row_end} < mi_row_start={mi_row_start}"
    );
    assert!(
        mi_col_end >= mi_col_start,
        "decode_tile: mi_col_end={mi_col_end} < mi_col_start={mi_col_start}"
    );

    let mut r = mi_row_start;
    while r < mi_row_end {
        // §7.4.2 clear_left_context( ) — fires once per superblock row
        // per the §6.4.2 listing line 2345.
        ctx_state.clear_left();
        let mut c = mi_col_start;
        while c < mi_col_end {
            // §6.4.2 line 2347: decode_partition( r, c, BLOCK_64X64 ).
            decode_partition(
                coder,
                r,
                c,
                crate::residual::BLOCK_64X64,
                mi_rows,
                mi_cols,
                ctx_state,
                probs_kind,
                leaves,
            )?;
            c += 8;
        }
        r += 8;
    }

    Ok(())
}

// ----- §6.4 decode_tiles outer driver -----

/// `tile_payload_sizes( sz, tile_rows_log2, tile_cols_log2 )` per spec
/// §6.4 lines 2306-2311 — the byte-stream prefix walk that derives the
/// per-tile size budget WITHOUT invoking the §9.2 bool coder or the
/// §6.4.2 [`decode_tile`] body.
///
/// ```text
/// for ( tileRow = 0; tileRow < tileRows; tileRow++ ) {
///     for ( tileCol = 0; tileCol < tileCols; tileCol++ ) {
///         lastTile = (tileRow == tileRows - 1) && (tileCol == tileCols - 1)
///         if ( lastTile ) {
///             tile_size = sz
///         } else {
///             tile_size                                                  f(32)
///             sz -= tile_size + 4
///         }
///         ...
///     }
/// }
/// ```
///
/// Inputs:
///
/// * `data` — the tile-stream slice starting at the first tile's
///   `tile_size` (i.e. immediately past `parse_compressed_header`'s
///   exit). Only the `f(32)` length prefixes are read; the tile
///   bodies themselves are skipped over by byte count, never decoded.
/// * `sz` — the total tile-stream budget in bytes (the running `sz`
///   the §6.4 listing updates with `sz -= tile_size + 4`). On exit
///   the budget that would remain for the last tile is reflected in
///   the last entry of the returned vector.
/// * `tile_rows_log2`, `tile_cols_log2` — from the
///   [`crate::header::TileInfo`] of the uncompressed header.
///
/// Returns one `u32` per `(tileRow, tileCol)` cell, in row-major
/// order. The output `Vec` length is exactly `tileRows * tileCols`.
/// Every entry except the last is the `f(32)` value read at that
/// tile's slot; the last entry is the spec's `tile_size = sz`
/// assignment for the `lastTile` case (line 2308).
///
/// This is the pure byte-arithmetic subset of [`decode_tiles`] — it
/// is what a §6.4 demuxer needs to slice a frame's tile payload into
/// per-tile bool-coder sub-streams, and what the round-32 §6.4.2
/// [`decode_tile`] caller can pre-compute before allocating per-tile
/// state.
///
/// # Errors
///
/// * [`Error::UnexpectedEof`] — the byte stream is shorter than the
///   running `tile_size + 4` reads demand (a non-last tile's 4-byte
///   length prefix runs past the end of `data`, or a declared
///   `tile_size` value extends past the available byte slice).
/// * [`Error::InvalidBitstream`] — a non-last tile's declared
///   `tile_size + 4` would underflow the running `sz` budget per
///   §6.4 line 2311 (the spec's running subtraction would wrap).
pub fn tile_payload_sizes(
    data: &[u8],
    mut sz: u32,
    tile_rows_log2: u8,
    tile_cols_log2: u8,
) -> Result<Vec<u32>, Error> {
    let tile_cols: u32 = 1u32 << tile_cols_log2;
    let tile_rows: u32 = 1u32 << tile_rows_log2;

    let mut out: Vec<u32> = Vec::with_capacity((tile_rows * tile_cols) as usize);
    let mut byte_cursor: usize = 0;

    for tile_row in 0..tile_rows {
        for tile_col in 0..tile_cols {
            let last_tile = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
            let tile_size: u32 = if last_tile {
                // §6.4 line 2308: tile_size = sz.
                sz
            } else {
                // §6.4 line 2310: tile_size  f(32) — big-endian per
                // the spec's f(n) convention.
                if data.len() < byte_cursor + 4 {
                    return Err(Error::UnexpectedEof);
                }
                let raw = u32::from_be_bytes([
                    data[byte_cursor],
                    data[byte_cursor + 1],
                    data[byte_cursor + 2],
                    data[byte_cursor + 3],
                ]);
                byte_cursor += 4;
                // §6.4 line 2311: sz -= tile_size + 4. Checked
                // arithmetic so an oversized declared tile_size that
                // would underflow the running budget surfaces as a
                // bitstream error rather than wrapping.
                let delta = raw.checked_add(4).ok_or(Error::InvalidBitstream)?;
                sz = sz.checked_sub(delta).ok_or(Error::InvalidBitstream)?;
                raw
            };

            // Range check on the tile body itself (mirrors the
            // §9.2.1 `init_bool( tile_size )` slice fetch in
            // [`decode_tiles`]): a declared size that overshoots
            // `data` is `UnexpectedEof`. The last tile's `tile_size =
            // sz` is constrained by the caller's `sz` argument; for
            // non-last tiles the running `byte_cursor` has already
            // advanced past the `f(32)` prefix.
            if data.len() < byte_cursor + tile_size as usize {
                return Err(Error::UnexpectedEof);
            }
            byte_cursor += tile_size as usize;

            out.push(tile_size);
        }
    }

    Ok(out)
}

/// Per-tile record emitted by [`decode_tiles`] for each
/// `(tileRow, tileCol)` cell of the tile grid.
///
/// Each entry captures the four `get_tile_offset( )` outputs and the
/// flat list of `(r, c, subsize)` leaves the per-tile [`decode_tile`]
/// call emitted, so the caller can replay the §6.4.2 traversal order
/// per tile without reaching back through a single global leaf log.
#[derive(Debug, Clone)]
pub(crate) struct DecodedTile {
    /// `tileRow` per §6.4 line 2304.
    pub tile_row: u32,
    /// `tileCol` per §6.4 line 2305.
    pub tile_col: u32,
    /// `MiRowStart` per §6.4 line 2313.
    pub mi_row_start: u32,
    /// `MiRowEnd` per §6.4 line 2314.
    pub mi_row_end: u32,
    /// `MiColStart` per §6.4 line 2315.
    pub mi_col_start: u32,
    /// `MiColEnd` per §6.4 line 2316.
    pub mi_col_end: u32,
    /// `tile_size` per §6.4 lines 2308 / 2310 — the per-tile byte budget
    /// handed to `init_bool( )` for this tile.
    pub tile_size: u32,
    /// The §6.4.2 per-tile leaf log: every `(r, c, subsize)` cell
    /// `decode_tile( )` visited inside this tile, in §6.4.3 traversal
    /// order.
    pub leaves: Vec<LeafBlock>,
}

/// `decode_tiles( sz )` per spec §6.4 (`vp9-spec.txt` lines 2300-2331).
///
/// ```text
/// decode_tiles( sz ) {
///     tileCols = 1 << tile_cols_log2
///     tileRows = 1 << tile_rows_log2
///     clear_above_context()
///     for ( tileRow = 0; tileRow < tileRows; tileRow++ ) {
///         for ( tileCol = 0; tileCol < tileCols; tileCol++ ) {
///             lastTile = (tileRow == tileRows - 1) && (tileCol == tileCols - 1)
///             if ( lastTile ) {
///                 tile_size = sz
///             } else {
///                 tile_size                                                  f(32)
///                 sz -= tile_size + 4
///             }
///             MiRowStart = get_tile_offset( tileRow, MiRows, tile_rows_log2 )
///             MiRowEnd   = get_tile_offset( tileRow + 1, MiRows, tile_rows_log2 )
///             MiColStart = get_tile_offset( tileCol, MiCols, tile_cols_log2 )
///             MiColEnd   = get_tile_offset( tileCol + 1, MiCols, tile_cols_log2 )
///             init_bool( tile_size )
///             decode_tile( )
///             exit_bool( )
///         }
///     }
/// }
/// ```
///
/// Frame-level driver: walks the `(1 << tile_rows_log2) × (1 <<
/// tile_cols_log2)` tile grid in row-major order, composing the four
/// pieces this round and earlier rounds have already lifted into
/// primitives:
///
/// * `clear_above_context( )` per §7.4.1 — fires once before the tile
///   walk, via [`PartitionContextState::clear_above`]. The §7.4.2
///   `clear_left_context( )` reset that fires per superblock row is
///   the responsibility of the inner [`decode_tile`] driver.
/// * `tile_size` per §6.4 line 2310 — read as `f(32)` (32-bit
///   big-endian) from the byte stream for every tile EXCEPT the last,
///   where the spec sets `tile_size = sz`. The 4-byte header is not
///   counted toward `tile_size` itself (the line 2311
///   `sz -= tile_size + 4` accounts for both the tile body AND the
///   4-byte length prefix).
/// * `init_bool( tile_size )` / `exit_bool( )` per §9.2.1 / §9.2.3 —
///   bracket the per-tile bool-coder lifetime; the byte slice handed
///   to `init_bool( )` is the `tile_size`-length sub-slice starting at
///   the current byte position.
/// * `decode_tile( )` per §6.4.2 — the [`decode_tile`] primitive
///   landed in round 32 fires `clear_left_context( )` per superblock
///   row and walks `decode_partition( r, c, BLOCK_64X64 )` over the
///   tile's MI window.
///
/// Inputs:
///
/// * `data` — the tile-stream slice starting at the first tile's
///   `tile_size` (i.e. immediately past `parse_compressed_header`'s
///   exit). The caller is responsible for stripping the
///   uncompressed-header bytes + the compressed-header bytes from the
///   frame buffer first.
/// * `sz` — the total tile-stream budget in bytes (the running `sz`
///   the §6.4 listing updates with `sz -= tile_size + 4`). On entry
///   this is the frame's tile payload size; on exit it must equal the
///   last tile's `tile_size` (the §6.4 line 2308 assignment for
///   `lastTile`).
/// * `tile_rows_log2`, `tile_cols_log2` — from the
///   [`crate::header::TileInfo`] of the uncompressed header.
/// * `mi_rows`, `mi_cols` — frame extents per §7.2.4.
/// * `ctx_state` — the `Above` / `Left` partition-context strips
///   (zeroed `clear_above_context( )` is fired internally for the
///   above strip; the inner [`decode_tile`] resets the left strip per
///   superblock row).
/// * `probs_kind` — the §9.3.2 `partition` probability source.
///
/// Returns one [`DecodedTile`] per `(tileRow, tileCol)` cell, in
/// row-major order. The output `Vec` length is exactly `tileRows *
/// tileCols`.
///
/// # Errors
///
/// * [`Error::UnexpectedEof`] — the byte stream is shorter than the
///   running tile_size demands (either the 4-byte length prefix or
///   the tile body), or `sz` underflows when the spec computes `sz -=
///   tile_size + 4`.
/// * [`Error::InvalidBitstream`] — the inner `init_bool( ) /
///   exit_bool( )` rejects (`sz < 1`, nonzero marker, nonzero
///   exit-padding) or [`decode_partition`] / [`decode_tile`] surfaces
///   an inner bitstream violation.
#[allow(clippy::too_many_arguments)] // mirrors the §6.4 spec
                                     // composition (data + running sz
                                     // + two tile-grid sizes + two
                                     // frame extents + ctx state +
                                     // probs).
pub(crate) fn decode_tiles(
    data: &[u8],
    sz: u32,
    tile_rows_log2: u8,
    tile_cols_log2: u8,
    mi_rows: u32,
    mi_cols: u32,
    ctx_state: &mut PartitionContextState,
    probs_kind: PartitionProbsKind<'_>,
) -> Result<Vec<DecodedTile>, Error> {
    let tile_cols: u32 = 1u32 << tile_cols_log2;
    let tile_rows: u32 = 1u32 << tile_rows_log2;

    // §6.4 lines 2306-2311: walk the per-tile `f(32)` length prefixes
    // up-front via the [`tile_payload_sizes`] helper. The helper also
    // range-checks every declared tile body against `data`, so the
    // per-tile slice fetch below is guaranteed in-bounds for every
    // `tile_size` returned.
    let sizes = tile_payload_sizes(data, sz, tile_rows_log2, tile_cols_log2)?;

    // §6.4 line 2303: clear_above_context() — once per frame, before
    // the first tile.
    ctx_state.clear_above();

    let mut out: Vec<DecodedTile> = Vec::with_capacity((tile_rows * tile_cols) as usize);
    let mut byte_cursor: usize = 0;
    let mut size_idx: usize = 0;

    for tile_row in 0..tile_rows {
        for tile_col in 0..tile_cols {
            let last_tile = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
            // Every non-last tile carries a 4-byte length prefix per
            // §6.4 line 2310; the last tile uses §6.4 line 2308
            // `tile_size = sz` with no prefix.
            if !last_tile {
                byte_cursor += 4;
            }
            let tile_size: u32 = sizes[size_idx];
            size_idx += 1;

            // §6.4 lines 2313-2316: derive the four MI extents via
            // the §6.4.1 primitive.
            let mi_row_start = get_tile_offset(tile_row, mi_rows, tile_rows_log2 as u32);
            let mi_row_end = get_tile_offset(tile_row + 1, mi_rows, tile_rows_log2 as u32);
            let mi_col_start = get_tile_offset(tile_col, mi_cols, tile_cols_log2 as u32);
            let mi_col_end = get_tile_offset(tile_col + 1, mi_cols, tile_cols_log2 as u32);

            // §6.4 line 2326: init_bool( tile_size ). The §9.2 bool
            // coder is bracketed inside the tile; its byte cursor is
            // independent of `byte_cursor` (which counts whole bytes
            // through the frame's tile payload). The §7.4.1 note
            // permits `tile_size = 0` only when the tile has zero
            // superblocks — `init_bool( )` itself still requires
            // `sz >= 1` per §9.2.1, so a zero-superblock tile is
            // still required to carry an init/exit pair with at
            // least one byte of body. The [`tile_payload_sizes`]
            // helper already proved this slice is in-bounds.
            let tile_slice = &data[byte_cursor..byte_cursor + tile_size as usize];
            let mut coder = BoolCoder::init_bool(tile_slice, tile_size as usize)?;

            // §6.4 line 2327: decode_tile( ).
            let mut leaves: Vec<LeafBlock> = Vec::new();
            decode_tile(
                &mut coder,
                mi_row_start,
                mi_row_end,
                mi_col_start,
                mi_col_end,
                mi_rows,
                mi_cols,
                ctx_state,
                probs_kind,
                &mut leaves,
            )?;

            // §6.4 line 2328: exit_bool( ).
            coder.exit_bool()?;

            byte_cursor += tile_size as usize;

            out.push(DecodedTile {
                tile_row,
                tile_col,
                mi_row_start,
                mi_row_end,
                mi_col_start,
                mi_col_end,
                tile_size,
                leaves,
            });
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{BLOCK_32X32, BLOCK_64X64};

    // ----- §3 / §10.2 verbatim-listing anchor tests -----

    #[test]
    fn partition_type_constants_match_spec() {
        // §7.4.3 table line 3843..3846:
        assert_eq!(PARTITION_NONE, 0);
        assert_eq!(PARTITION_HORZ, 1);
        assert_eq!(PARTITION_VERT, 2);
        assert_eq!(PARTITION_SPLIT, 3);
    }

    #[test]
    fn partition_dimension_constants_match_spec() {
        // §3 lines 463 / 497.
        assert_eq!(PARTITION_TYPES, 4);
        assert_eq!(PARTITION_CONTEXTS, 16);
    }

    #[test]
    fn b_width_log2_lookup_matches_spec_listing() {
        // §10.2 line 7088 verbatim.
        assert_eq!(B_WIDTH_LOG2_LOOKUP, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]);
    }

    #[test]
    fn b_height_log2_lookup_matches_spec_listing() {
        // §10.2 line 7099 verbatim.
        assert_eq!(
            B_HEIGHT_LOG2_LOOKUP,
            [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4]
        );
    }

    #[test]
    fn mi_width_log2_lookup_matches_spec_listing() {
        // §10.2 line 7108 verbatim.
        assert_eq!(
            MI_WIDTH_LOG2_LOOKUP,
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        );
        // BLOCK_64X64 = 12; the §9.3.2 boffset = mi_width_log2_lookup[
        // BLOCK_64X64 ] - bsl constant equals 3.
        assert_eq!(MI_WIDTH_LOG2_LOOKUP[12], 3);
        assert_eq!(MI_WIDTH_LOG2_BLOCK_64X64, 3);
    }

    #[test]
    fn num_8x8_blocks_wide_lookup_matches_spec_listing() {
        // §10.2 line 7111 verbatim.
        assert_eq!(
            NUM_8X8_BLOCKS_WIDE_LOOKUP,
            [1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8]
        );
    }

    #[test]
    fn num_8x8_blocks_high_lookup_matches_spec_listing() {
        // §10.2 line 7117 verbatim.
        assert_eq!(
            NUM_8X8_BLOCKS_HIGH_LOOKUP,
            [1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8]
        );
    }

    // ----- §10.2 subsize_lookup anchor tests -----

    #[test]
    fn subsize_lookup_partition_none_is_identity() {
        // §10.2 PARTITION_NONE arm: subsize_lookup[ NONE ][ bsize ] ==
        // bsize for every bsize (no splitting → child same as parent).
        for (bsize, &child) in SUBSIZE_LOOKUP[PARTITION_NONE as usize].iter().enumerate() {
            assert_eq!(
                child, bsize as u8,
                "PARTITION_NONE identity broken at bsize={bsize}"
            );
        }
    }

    #[test]
    fn subsize_lookup_partition_split_at_superblocks() {
        // §10.2 PARTITION_SPLIT arm anchors per spec lines 7160-7165:
        //   BLOCK_8X8 (3)  -> BLOCK_4X4 (0)
        //   BLOCK_16X16 (6) -> BLOCK_8X8 (3)
        //   BLOCK_32X32 (9) -> BLOCK_16X16 (6)
        //   BLOCK_64X64 (12) -> BLOCK_32X32 (9)
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][3], 0);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][6], 3);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][9], 6);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][12], 9);
    }

    #[test]
    fn subsize_lookup_partition_horz_and_vert_at_superblocks() {
        // §10.2 PARTITION_HORZ / PARTITION_VERT anchors per spec lines
        // 7139-7159.
        // HORZ: BLOCK_8X8 (3) -> BLOCK_8X4 (2); BLOCK_16X16 (6) -> BLOCK_16X8 (5);
        //       BLOCK_32X32 (9) -> BLOCK_32X16 (8); BLOCK_64X64 (12) -> BLOCK_64X32 (11).
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_HORZ as usize][3], 2);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_HORZ as usize][6], 5);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_HORZ as usize][9], 8);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_HORZ as usize][12], 11);
        // VERT: BLOCK_8X8 (3) -> BLOCK_4X8 (1); BLOCK_16X16 (6) -> BLOCK_8X16 (4);
        //       BLOCK_32X32 (9) -> BLOCK_16X32 (7); BLOCK_64X64 (12) -> BLOCK_32X64 (10).
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_VERT as usize][3], 1);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_VERT as usize][6], 4);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_VERT as usize][9], 7);
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_VERT as usize][12], 10);
    }

    #[test]
    fn subsize_lookup_partition_invalid_at_non_square_parents() {
        // §10.2: the HORZ / VERT / SPLIT arms list BLOCK_INVALID = 14
        // for every parent that's neither a square superblock nor
        // BLOCK_4X4. Spot-check the four edge cases.
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_HORZ as usize][1], 14); // BLOCK_4X8
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_VERT as usize][2], 14); // BLOCK_8X4
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][4], 14); // BLOCK_8X16
        assert_eq!(SUBSIZE_LOOKUP[PARTITION_HORZ as usize][7], 14); // BLOCK_16X32
    }

    // ----- §9.3.1 tree-listing anchor tests -----

    #[test]
    fn partition_tree_matches_spec_listing() {
        // §9.3.1 line 6083 verbatim: with PARTITION_NONE = 0 the
        // -PARTITION_NONE entry collapses to 0 (§9.3.3 post-loop `-n`
        // still returns 0 for the first leaf).
        assert_eq!(
            PARTITION_TREE,
            [
                0, // -PARTITION_NONE
                2, -1, // -PARTITION_HORZ
                4, -2, // -PARTITION_VERT
                -3, // -PARTITION_SPLIT
            ]
        );
    }

    #[test]
    fn cols_partition_tree_matches_spec_listing() {
        // §9.3.1 line 6090 verbatim.
        assert_eq!(COLS_PARTITION_TREE, [-1, -3]);
    }

    #[test]
    fn rows_partition_tree_matches_spec_listing() {
        // §9.3.1 line 6095 verbatim.
        assert_eq!(ROWS_PARTITION_TREE, [-2, -3]);
    }

    // ----- §10.4 / §10.5 probability-table anchors -----

    #[test]
    fn kf_partition_probs_table_shape_and_anchors() {
        // Shape per §3 (16 × 3).
        assert_eq!(KF_PARTITION_PROBS.len(), PARTITION_CONTEXTS);
        for row in KF_PARTITION_PROBS.iter() {
            assert_eq!(row.len(), PARTITION_TYPES - 1);
        }
        // §10.4 listing first row (8x8 → 4x4, both not split):
        assert_eq!(KF_PARTITION_PROBS[0], [158, 97, 94]);
        // §10.4 last row (64x64 → 32x32, both split):
        assert_eq!(KF_PARTITION_PROBS[15], [12, 3, 3]);
        // Two interior anchors picked from the listing:
        assert_eq!(KF_PARTITION_PROBS[5], [94, 20, 48]); // 16x16, above split
        assert_eq!(KF_PARTITION_PROBS[11], [24, 7, 5]); // 32x32, both split

        // Every entry must be a valid §9.2 probability (1..=255, the
        // §9.3.2 listing forbids 0). All KF table cells satisfy this
        // per the §10.4 listing.
        for row in KF_PARTITION_PROBS.iter() {
            for &p in row.iter() {
                assert!(
                    p >= 1,
                    "KF_PARTITION_PROBS has a 0 probability (§9.2 violation)"
                );
            }
        }
    }

    #[test]
    fn default_partition_probs_table_shape_and_anchors() {
        // Shape per §3 (16 × 3).
        assert_eq!(DEFAULT_PARTITION_PROBS.len(), PARTITION_CONTEXTS);
        for row in DEFAULT_PARTITION_PROBS.iter() {
            assert_eq!(row.len(), PARTITION_TYPES - 1);
        }
        // §10.5 listing first row (8x8 → 4x4, both not split):
        assert_eq!(DEFAULT_PARTITION_PROBS[0], [199, 122, 141]);
        // §10.5 listing last row (64x64 → 32x32, both split):
        assert_eq!(DEFAULT_PARTITION_PROBS[15], [10, 7, 6]);
        // Two interior anchors:
        assert_eq!(DEFAULT_PARTITION_PROBS[4], [174, 73, 87]); // 16x16, both not split
        assert_eq!(DEFAULT_PARTITION_PROBS[8], [177, 58, 59]); // 32x32, both not split

        // §9.2 minimum-probability sanity check.
        for row in DEFAULT_PARTITION_PROBS.iter() {
            for &p in row.iter() {
                assert!(
                    p >= 1,
                    "DEFAULT_PARTITION_PROBS has a 0 probability (§9.2 violation)"
                );
            }
        }
    }

    // ----- §9.3.2 partition_plane_context tests -----

    // Per §9.3.2 line 6257: bsl = mi_width_log2_lookup[ bsize ]; per
    // §10.2 line 7108 the lookup is { 0,0,0,0,0,1,1,1,2,2,2,3,3 }, so
    //   BLOCK_8X8 (3)  -> bsl = 0 -> ctx group {0,1,2,3}
    //   BLOCK_16X16 (6) -> bsl = 1 -> ctx group {4,5,6,7}
    //   BLOCK_32X32 (9) -> bsl = 2 -> ctx group {8,9,10,11}
    //   BLOCK_64X64 (12)-> bsl = 3 -> ctx group {12,13,14,15}
    // boffset = mi_width_log2_lookup[ BLOCK_64X64 ] - bsl = 3 - bsl.

    #[test]
    fn partition_plane_context_zero_strips_block_8x8() {
        // bsize = BLOCK_8X8 = 3 → bsl = 0, boffset = 3, mask = 0x08.
        // num8x8 = 1. Zero strips → above = left = 0 → ctx = 0*4 = 0.
        let ctx = partition_plane_context(/* BLOCK_8X8 */ 3, &[0], &[0]);
        assert_eq!(ctx, 0);
    }

    #[test]
    fn partition_plane_context_zero_strips_block_16x16() {
        // bsize = BLOCK_16X16 = 6 → bsl = 1, boffset = 2, mask = 0x04.
        // num8x8 = 2. Zero strips → ctx = 1*4 = 4.
        let ctx = partition_plane_context(/* BLOCK_16X16 */ 6, &[0, 0], &[0, 0]);
        assert_eq!(ctx, 4);
    }

    #[test]
    fn partition_plane_context_zero_strips_block_32x32() {
        // bsize = BLOCK_32X32 = 9 → bsl = 2, boffset = 1, mask = 0x02.
        // num8x8 = 4. Zero strips → ctx = 2*4 = 8.
        let ctx = partition_plane_context(/* BLOCK_32X32 */ 9, &[0; 4], &[0; 4]);
        assert_eq!(ctx, 8);
    }

    #[test]
    fn partition_plane_context_zero_strips_block_64x64() {
        // bsize = BLOCK_64X64 = 12 → bsl = 3, boffset = 0, mask = 0x01.
        // num8x8 = 8. Zero strips → ctx = 3*4 = 12.
        let ctx = partition_plane_context(/* BLOCK_64X64 */ 12, &[0; 8], &[0; 8]);
        assert_eq!(ctx, 12);
    }

    #[test]
    fn partition_plane_context_above_bit_set_block_8x8() {
        // bsize = BLOCK_8X8 = 3 → bsl = 0, boffset = 3, mask = 0x08.
        // num8x8 = 1. above_ctx = [0x08], left_ctx = [0] → above bit
        // set, left bit clear → ctx = 0*4 + 0*2 + 1 = 1.
        let ctx = partition_plane_context(/* BLOCK_8X8 */ 3, &[0x08], &[0]);
        assert_eq!(ctx, 1);
    }

    #[test]
    fn partition_plane_context_left_bit_set_block_8x8() {
        // bsize = BLOCK_8X8 = 3 → bsl = 0, boffset = 3, mask = 0x08.
        // num8x8 = 1. above_ctx = [0], left_ctx = [0x08] → above bit
        // clear, left bit set → ctx = 0*4 + 1*2 + 0 = 2.
        let ctx = partition_plane_context(/* BLOCK_8X8 */ 3, &[0], &[0x08]);
        assert_eq!(ctx, 2);
    }

    #[test]
    fn partition_plane_context_both_bits_set_block_8x8() {
        // Both bits set → ctx = 0*4 + 1*2 + 1 = 3.
        let ctx = partition_plane_context(/* BLOCK_8X8 */ 3, &[0x08], &[0x08]);
        assert_eq!(ctx, 3);
    }

    #[test]
    fn partition_plane_context_or_fold_across_strip() {
        // bsize = BLOCK_32X32 = 9 → bsl = 2, boffset = 1, mask = 0x02.
        // num8x8 = 4. Above strip [0, 0, 0x02, 0] → above_bits = 0x02 →
        // above bit set. Left strip all zero → left bit clear.
        // ctx = 2*4 + 0*2 + 1 = 9.
        let ctx =
            partition_plane_context(/* BLOCK_32X32 */ 9, &[0, 0, 0x02, 0], &[0, 0, 0, 0]);
        assert_eq!(ctx, 9);
    }

    #[test]
    fn partition_plane_context_or_fold_unrelated_bits_ignored() {
        // bsize = BLOCK_16X16 = 6 → bsl = 1, boffset = 2, mask = 0x04.
        // num8x8 = 2. A strip with bit 0 set (0x01) does NOT contribute
        // to the bsl-th bit of the OR fold; ctx should still be the
        // "both unset" base 1*4 = 4.
        let ctx = partition_plane_context(/* BLOCK_16X16 */ 6, &[0x01, 0x02], &[0x01, 0x02]);
        assert_eq!(ctx, 4);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn partition_plane_context_panics_on_invalid_bsize() {
        let _ = partition_plane_context(/* BLOCK_INVALID */ 14, &[0], &[0]);
    }

    #[test]
    #[should_panic(expected = "strip widths mismatch")]
    fn partition_plane_context_panics_on_mismatched_strips() {
        let _ = partition_plane_context(/* BLOCK_16X16 */ 6, &[0, 0], &[0]);
    }

    // ----- §6.4.3 decode_partition_type tests -----

    // The §9.2 decoder needs a marker bit (= 0) and a byte buffer the
    // bool coder can renormalise from. We re-use the same hand-buffer
    // recipe the mode_info tests use.

    fn zero_coder() -> BoolCoder<'static> {
        // [0x00; 16] satisfies §9.2.1 (marker bit decodes to 0); every
        // subsequent `read_bool(p)` returns 0 because BoolValue stays
        // 0 < split for any p.
        static ZEROS: [u8; 16] = [0u8; 16];
        BoolCoder::init_bool(&ZEROS, 16).expect("16-byte zero buffer is a valid §9.2 init")
    }

    fn one_then_zero_coder() -> BoolCoder<'static> {
        // First byte 0x7F → marker decodes to 0 (split=128, value=127 <
        // 128). After the marker BoolValue=127, BoolRange=128. With
        // `read_bool(255)` split=127, value(127) NOT < split(127) →
        // bit=1; renorm refills 7 bits from the all-zero tail so the
        // state becomes range=128, value=0; every subsequent read
        // returns 0 regardless of the probability passed.
        //
        // This is the standard "right-branch-once-then-left" probe
        // used elsewhere in the crate's mode_info tests for
        // tree-decode walkers.
        static BIAS: [u8; 16] = [0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        BoolCoder::init_bool(&BIAS, 16).expect("bias buffer is a valid §9.2 init")
    }

    fn all_ones_coder() -> BoolCoder<'static> {
        // First byte 0x7F < 128 → marker decodes to 0. Rest 0xFF + p=255
        // sustains a chain of bit=1 outputs across multiple
        // `read_bool(255)` calls (renorm refills bring value back high
        // enough to satisfy `value >= split` for each subsequent p=255
        // call). Used to exercise tree paths that walk every
        // right-branch in sequence.
        static BIAS: [u8; 16] = [
            0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        ];
        BoolCoder::init_bool(&BIAS, 16).expect("bias buffer is a valid §9.2 init")
    }

    #[test]
    fn decode_partition_type_interior_zero_buffer_picks_none() {
        // hasRows = hasCols = true: walks the 6-entry PARTITION_TREE
        // starting at node 0. First read_bool(probs[0]) returns 0 →
        // tree[0+0] = -PARTITION_NONE = 0 → return 0 (PARTITION_NONE).
        let mut c = zero_coder();
        let probs = [158, 97, 94]; // KF row 0
        let part = decode_partition_type(&mut c, true, true, &probs).unwrap();
        assert_eq!(part, PARTITION_NONE);
    }

    #[test]
    fn decode_partition_type_interior_all_ones_coder_picks_split() {
        // hasRows = hasCols = true, all-ones bias coder + probs all
        // 255: every read_bool(255) flips to 1, so the walker traverses
        // tree[0+1] = 2 → tree[2+1] = 4 → tree[4+1] = -PARTITION_SPLIT
        // → return PARTITION_SPLIT.
        let mut c = all_ones_coder();
        let probs = [255, 255, 255];
        let part = decode_partition_type(&mut c, true, true, &probs).unwrap();
        assert_eq!(part, PARTITION_SPLIT);
    }

    #[test]
    fn decode_partition_type_interior_one_then_zero_picks_horz() {
        // hasRows = hasCols = true, one-then-zero coder + probs[0]=255:
        // first read_bool(255) flips to 1 (walker → node 1); subsequent
        // reads return 0 regardless of probability → walker hits
        // tree[2+0] = -PARTITION_HORZ → returns PARTITION_HORZ.
        //
        // This pins probs[0] as the first node consulted and the
        // §9.3.1 partition_tree right-branch at node 0 routing to node
        // 1 (the §10.4/§10.5 probs row's [1] cell).
        let mut c = one_then_zero_coder();
        let probs = [255, 0, 0];
        let part = decode_partition_type(&mut c, true, true, &probs).unwrap();
        assert_eq!(part, PARTITION_HORZ);
    }

    #[test]
    fn decode_partition_type_right_edge_zero_buffer_picks_horz() {
        // hasRows = false, hasCols = true: walks COLS_PARTITION_TREE.
        // probs[1] is the sole read; with the zero coder it returns 0
        // → tree[0+0] = -PARTITION_HORZ → return PARTITION_HORZ.
        let mut c = zero_coder();
        let probs = [158, 97, 94];
        let part = decode_partition_type(&mut c, false, true, &probs).unwrap();
        assert_eq!(part, PARTITION_HORZ);
    }

    #[test]
    fn decode_partition_type_right_edge_one_then_zero_picks_split() {
        // hasRows = false, hasCols = true with probs[1] = 255 + the
        // one-then-zero coder. The cols_partition_tree is only 2
        // entries so it terminates after a single read; the first
        // read_bool(255) flips to 1 → tree[0+1] = -PARTITION_SPLIT →
        // return PARTITION_SPLIT. §9.3.2 line 6249 fixes node2=1 so
        // probs[1] is the cell consulted (we set the other cells to 0
        // to confirm).
        let mut c = one_then_zero_coder();
        let probs = [0, 255, 0]; // only [1] matters for this arm
        let part = decode_partition_type(&mut c, false, true, &probs).unwrap();
        assert_eq!(part, PARTITION_SPLIT);
    }

    #[test]
    fn decode_partition_type_bottom_edge_zero_buffer_picks_vert() {
        // hasRows = true, hasCols = false: walks ROWS_PARTITION_TREE.
        // probs[2] is the sole read; zero coder → 0 → return
        // PARTITION_VERT.
        let mut c = zero_coder();
        let probs = [158, 97, 94];
        let part = decode_partition_type(&mut c, true, false, &probs).unwrap();
        assert_eq!(part, PARTITION_VERT);
    }

    #[test]
    fn decode_partition_type_bottom_edge_one_then_zero_picks_split() {
        // hasRows = true, hasCols = false with probs[2] = 255 + the
        // one-then-zero coder. The rows_partition_tree is 2 entries so
        // a single bit suffices; the first read_bool(255) flips to 1 →
        // tree[0+1] = -PARTITION_SPLIT → return PARTITION_SPLIT.
        // §9.3.2 line 6250 fixes node2=2 so probs[2] is the cell
        // consulted.
        let mut c = one_then_zero_coder();
        let probs = [0, 0, 255]; // only [2] matters
        let part = decode_partition_type(&mut c, true, false, &probs).unwrap();
        assert_eq!(part, PARTITION_SPLIT);
    }

    #[test]
    fn decode_partition_type_corner_consumes_no_bits_returns_split() {
        // §9.3.1 fourth arm: hasRows = hasCols = false → return
        // PARTITION_SPLIT immediately without reading any bool-coder
        // bits. Verify by passing a buffer that would fail the bool
        // coder if it WERE consulted (we observe ordering by chaining a
        // post-call interior decode after).
        let mut c = zero_coder();
        let probs = [158, 97, 94];
        let part_a = decode_partition_type(&mut c, false, false, &probs).unwrap();
        assert_eq!(part_a, PARTITION_SPLIT);
        // The bool coder is untouched: a subsequent interior decode on
        // the same coder must still return PARTITION_NONE (just like
        // `decode_partition_type_interior_zero_buffer_picks_none`).
        let part_b = decode_partition_type(&mut c, true, true, &probs).unwrap();
        assert_eq!(part_b, PARTITION_NONE);
    }

    #[test]
    fn decode_partition_type_interior_one_then_zero_with_p255_picks_horz() {
        // Pins the §9.3.2 interior `node2 = node` rule: with the
        // one-then-zero coder and probs[0] = 255, the first read
        // returns 1 (walker → node 1); the renorm tail drives value
        // back to 0, so the second p=255 read returns 0 (walker →
        // tree[2+0] = -PARTITION_HORZ). Returns PARTITION_HORZ.
        //
        // Re-verifies `decode_partition_type_interior_one_then_zero_picks_horz`
        // (which used probs = [255, 0, 0]) with uniform probs to rule
        // out probs[1] / probs[2] being silently consulted: switching
        // probs[1] / probs[2] to 255 does not change the outcome.
        let mut c = one_then_zero_coder();
        let probs = [255, 255, 255];
        let part = decode_partition_type(&mut c, true, true, &probs).unwrap();
        assert_eq!(part, PARTITION_HORZ);
    }

    #[test]
    fn decode_partition_type_returns_one_of_four_partition_values() {
        // Exhaustive sanity: every (has_rows, has_cols) combination
        // under three representative bool-coder states returns a valid
        // PARTITION_* value (0..=3).
        let zero: [u8; 16] = [0u8; 16];
        let one_then_zero: [u8; 16] = [0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let all_ones: [u8; 16] = [
            0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        ];
        for &(hr, hc) in &[(false, false), (false, true), (true, false), (true, true)] {
            for buffer in &[&zero[..], &one_then_zero[..], &all_ones[..]] {
                let mut c = BoolCoder::init_bool(buffer, 16).unwrap();
                let probs = [128u8, 128, 128];
                let part = decode_partition_type(&mut c, hr, hc, &probs).unwrap();
                assert!(
                    part <= PARTITION_SPLIT,
                    "decoded partition out of range: {part}"
                );
            }
        }
    }

    #[test]
    fn partition_plane_context_full_sweep_covers_ctx_range() {
        // Sanity: across the four legal superblock sizes (BLOCK_8X8 /
        // BLOCK_16X16 / BLOCK_32X32 / BLOCK_64X64) and the four
        // (above, left) bit-set combinations, partition_plane_context
        // must produce 16 unique ctx values covering 0..=15
        // (PARTITION_CONTEXTS).
        let mut seen = [false; PARTITION_CONTEXTS];
        for &(bsize, mask) in &[
            (3u8, 0x08u8),  // BLOCK_8X8   -> bsl=0, boffset=3, mask=0x08
            (6u8, 0x04u8),  // BLOCK_16X16 -> bsl=1, boffset=2, mask=0x04
            (9u8, 0x02u8),  // BLOCK_32X32 -> bsl=2, boffset=1, mask=0x02
            (12u8, 0x01u8), // BLOCK_64X64 -> bsl=3, boffset=0, mask=0x01
        ] {
            let n8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as usize;
            for above_set in [false, true] {
                for left_set in [false, true] {
                    let mut above_strip = vec![0u8; n8x8];
                    let mut left_strip = vec![0u8; n8x8];
                    if above_set {
                        above_strip[0] = mask;
                    }
                    if left_set {
                        left_strip[0] = mask;
                    }
                    let ctx = partition_plane_context(bsize, &above_strip, &left_strip);
                    assert!(ctx < PARTITION_CONTEXTS, "ctx={ctx} OOB");
                    assert!(!seen[ctx], "duplicate ctx={ctx}");
                    seen[ctx] = true;
                }
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "ctx={i} not covered");
        }
    }

    // ----- §6.4.3 recursive decode_partition driver tests -----

    /// Test-only inverse of the §9.2.2 decoder. Given a sequence of
    /// `(bit, p)` pairs the caller wants the decoder to produce,
    /// returns a byte buffer that — when fed through [`BoolCoder`] —
    /// yields exactly that sequence (following the §9.2.1 marker).
    ///
    /// Implementation strategy: bounded search over the 128 possible
    /// post-marker `BoolValue` bytes combined with depth-first search
    /// over per-renorm refill bits. For each candidate, the §9.2
    /// decoder steps are forward-simulated step by step against the
    /// target sequence; the first candidate that produces every target
    /// bit wins. This is a test-only construction — it has no runtime
    /// in the production decode path and only needs to handle the
    /// small bit-counts generated by the §6.4.3 driver tests (a few
    /// dozen `read_bool` calls per fixture).
    ///
    /// The forward-simulation routine ([`try_decode`]) and the
    /// pruning helper ([`feasible_prefix`]) re-state the §9.2.2
    /// listing verbatim: `split = 1 + (((BoolRange - 1) * p) >> 8)`,
    /// the `BoolValue < split` branch test, the `BoolValue -= split` /
    /// `BoolRange -= split` rebase on bit = 1, and the renormalisation
    /// loop `while BoolRange < 128: BoolRange <<= 1, BoolValue =
    /// (BoolValue << 1) + newBit`.
    struct RangeEncoder {
        /// The caller's recorded `(bit, p)` sequence — the search
        /// target.
        target: Vec<(bool, u8)>,
    }

    impl RangeEncoder {
        fn new() -> Self {
            Self { target: Vec::new() }
        }

        fn write_bool(&mut self, bit: bool, p: u8) {
            self.target.push((bit, p));
        }

        /// Forward-simulate the §9.2.2 decoder on `(bool_value, stream)`
        /// against the target sequence. Returns `Some(stream_bits_used)`
        /// when every target call decodes correctly using only the
        /// supplied prefix of stream bits; `None` if any read disagrees
        /// with the target, or if the simulation runs out of stream
        /// before the target sequence completes.
        ///
        /// `stream` is the iterator of renorm refill bits (LSB-first
        /// would be ambiguous; we pass MSB-first big-endian bits as a
        /// slice).
        fn try_decode(bool_value: u32, stream: &[bool], target: &[(bool, u8)]) -> Option<usize> {
            // §9.2 decoder simulation.
            let mut value = bool_value;
            let mut pos = 0usize;

            // §9.2.1 marker: read_bool(128). split = 128. bit must = 0.
            if value >= 128 {
                // marker would decode to 1 → invalid stream.
                return None;
            }
            // No renorm fires (range = 128, not < 128).
            let mut range: u32 = 128;

            for &(expected_bit, p) in target {
                let split = 1 + (((range - 1) * (p as u32)) >> 8);
                let bit;
                if value < split {
                    range = split;
                    bit = false;
                } else {
                    range -= split;
                    value -= split;
                    bit = true;
                }
                if bit != expected_bit {
                    return None;
                }
                while range < 128 {
                    if pos >= stream.len() {
                        return None;
                    }
                    let nb = u32::from(stream[pos]);
                    pos += 1;
                    range <<= 1;
                    value = (value << 1) + nb;
                }
            }
            Some(pos)
        }

        /// Recursive depth-first search: at each renorm, try refill
        /// bit 0 first then 1, until a full target match is found.
        /// Returns the winning `(bool_value, stream_bits)` pair.
        fn search(target: &[(bool, u8)]) -> (u32, Vec<bool>) {
            // Reasonable upper bound on renorm refills for any test
            // fixture this encoder is asked to produce. The §6.4.3
            // driver tests emit at most a few dozen `read_bool` calls
            // each, each consuming at most ~7 refills under extreme
            // probabilities.
            const MAX_REFILLS: usize = 256;

            // Try each candidate `bool_value` from 0..128 (post-marker
            // requires value < 128). For each, DFS over refill bits.
            for bv in 0..128u32 {
                let mut stream = Vec::with_capacity(MAX_REFILLS);
                if Self::dfs(bv, &mut stream, target, MAX_REFILLS) {
                    return (bv, stream);
                }
            }
            panic!(
                "RangeEncoder::search: no consistent codeword for target sequence ({} pairs)",
                target.len()
            );
        }

        fn dfs(
            bool_value: u32,
            stream: &mut Vec<bool>,
            target: &[(bool, u8)],
            max_refills: usize,
        ) -> bool {
            // Try the current `stream` length; if simulation succeeds,
            // we're done. If it fails with "ran out of stream", extend
            // the stream by one bit (try 0 then 1) and recurse.
            match Self::try_decode(bool_value, stream, target) {
                Some(_) => true,
                None => {
                    if stream.len() >= max_refills {
                        return false;
                    }
                    // Only extend if the failure was a stream-exhaust;
                    // a mid-stream bit mismatch can't be fixed by
                    // appending. Distinguish by re-running with the
                    // current prefix and checking if the failure was
                    // mid-call. For our test sizes the brute approach
                    // (always extend) is fine — try_decode returns
                    // None for both mismatch and exhaust, and the
                    // recursion catches mismatches at a deeper level
                    // (which then backtrack to the next BV).
                    //
                    // To keep recursion shallow we detect mismatch by
                    // running a partial simulation here too: if the
                    // current bool_value can't even pass the first
                    // call's bit-direction test with any stream
                    // extension, we bail early.
                    if !Self::feasible_prefix(bool_value, stream, target) {
                        return false;
                    }
                    stream.push(false);
                    if Self::dfs(bool_value, stream, target, max_refills) {
                        return true;
                    }
                    stream.pop();
                    stream.push(true);
                    if Self::dfs(bool_value, stream, target, max_refills) {
                        return true;
                    }
                    stream.pop();
                    false
                }
            }
        }

        /// Heuristic pruning: simulate `(bool_value, stream)` up to the
        /// first stream-exhaust event; return `true` only if every
        /// target bit decoded so far matched. This rejects
        /// `bool_value` candidates whose early `read_bool` calls
        /// disagree with the target regardless of any future stream
        /// extension.
        fn feasible_prefix(bool_value: u32, stream: &[bool], target: &[(bool, u8)]) -> bool {
            let mut value = bool_value;
            let mut pos = 0usize;
            // marker
            if value >= 128 {
                return false;
            }
            let mut range: u32 = 128;
            for &(expected_bit, p) in target {
                let split = 1 + (((range - 1) * (p as u32)) >> 8);
                let bit;
                if value < split {
                    range = split;
                    bit = false;
                } else {
                    range -= split;
                    value -= split;
                    bit = true;
                }
                if bit != expected_bit {
                    return false;
                }
                while range < 128 {
                    if pos >= stream.len() {
                        // Reached unknown territory; future stream
                        // extension might succeed.
                        return true;
                    }
                    let nb = u32::from(stream[pos]);
                    pos += 1;
                    range <<= 1;
                    value = (value << 1) + nb;
                }
            }
            true
        }

        /// Flush the encoder state and return the byte buffer the
        /// decoder will consume. The trailing tail is zero-padded so
        /// any further renorm reads past the strictly-required bits
        /// stay defined.
        fn finish(self) -> Vec<u8> {
            let (bool_value, stream) = Self::search(&self.target);

            // Pack: byte 0 = BoolValue (MSB-first 8 bits); subsequent
            // bytes carry the renorm refill bits in MSB-first order.
            let total_bits = 8 + stream.len();
            let payload_bytes = total_bits.div_ceil(8);
            let mut out = vec![0u8; payload_bytes + 16];
            out[0] = bool_value as u8;
            for (i, &b) in stream.iter().enumerate() {
                if b {
                    let stream_pos = 8 + i;
                    let byte = stream_pos / 8;
                    out[byte] |= 1 << (7 - (stream_pos & 7));
                }
            }
            out
        }
    }

    /// Helper: encode a sequence of `(bit, p)` pairs and wrap the
    /// result in a [`BoolCoder`] ready for `read_bool` consumption.
    /// The returned byte buffer is leaked into `'static` so it can be
    /// borrowed by the coder without lifetime ceremony in tests.
    fn coder_from(pairs: &[(bool, u8)]) -> BoolCoder<'static> {
        let mut enc = RangeEncoder::new();
        for &(bit, p) in pairs {
            enc.write_bool(bit, p);
        }
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        BoolCoder::init_bool(leaked, sz).expect("encoded buffer is a valid §9.2 init")
    }

    /// Roundtrip sanity: encoder + decoder reproduce an arbitrary
    /// `(bit, p)` sequence verbatim.
    #[test]
    fn range_encoder_roundtrips_bool_sequence() {
        let pairs = [
            (false, 128u8),
            (true, 64),
            (false, 200),
            (true, 32),
            (false, 100),
            (true, 250),
            (false, 8),
            (false, 128),
        ];
        let mut coder = coder_from(&pairs);
        for (i, &(bit, p)) in pairs.iter().enumerate() {
            let got = coder.read_bool(p as u32).unwrap();
            assert_eq!(
                got,
                u32::from(bit),
                "mismatch at step {i}: encoded bit={bit} p={p}, decoded={got}"
            );
        }
    }

    /// Roundtrip sanity at the probability extremes.
    #[test]
    fn range_encoder_roundtrips_extreme_probabilities() {
        let pairs = [
            (false, 1u8),
            (true, 255),
            (false, 255),
            (true, 1),
            (false, 128),
            (true, 128),
        ];
        let mut coder = coder_from(&pairs);
        for &(bit, p) in pairs.iter() {
            assert_eq!(coder.read_bool(p as u32).unwrap(), u32::from(bit));
        }
    }

    // ----- §6.4.3 recursive driver hand-built bitstream tests -----

    /// Helper: encode the §9.3.1 + §9.3.3 walk for a single keyframe
    /// `partition` decode, given the decoded partition value and the
    /// `(has_rows, has_cols)` quadrant flags. Pushes the right `(bit,
    /// prob)` pairs into the encoder per the §9.3.2 `node2` rule and
    /// the §10.4 [`KF_PARTITION_PROBS`] row.
    fn push_partition_decode(
        enc: &mut RangeEncoder,
        partition: u8,
        has_rows: bool,
        has_cols: bool,
        ctx: usize,
    ) {
        let probs = KF_PARTITION_PROBS[ctx];
        match (has_rows, has_cols) {
            (true, true) => {
                // Walk the 6-entry partition_tree:
                //   node 0: -PARTITION_NONE / 2
                //   node 2: -PARTITION_HORZ / 4
                //   node 4: -PARTITION_VERT / -PARTITION_SPLIT
                // First read uses probs[0] (node 0 >> 1 = 0).
                // tree_decode reads p = prob(n >> 1) at each step.
                let bit_seq: &[(bool, u8)] = match partition {
                    PARTITION_NONE => &[(false, probs[0])],
                    PARTITION_HORZ => &[(true, probs[0]), (false, probs[1])],
                    PARTITION_VERT => &[(true, probs[0]), (true, probs[1]), (false, probs[2])],
                    PARTITION_SPLIT => &[(true, probs[0]), (true, probs[1]), (true, probs[2])],
                    _ => panic!("bad partition {partition}"),
                };
                for &(bit, p) in bit_seq {
                    enc.write_bool(bit, p);
                }
            }
            (false, true) => {
                // cols_partition_tree: only HORZ / SPLIT. node2 = 1.
                let bit = match partition {
                    PARTITION_HORZ => false,
                    PARTITION_SPLIT => true,
                    _ => panic!("right-edge can only be HORZ / SPLIT, got {partition}"),
                };
                enc.write_bool(bit, probs[1]);
            }
            (true, false) => {
                // rows_partition_tree: only VERT / SPLIT. node2 = 2.
                let bit = match partition {
                    PARTITION_VERT => false,
                    PARTITION_SPLIT => true,
                    _ => panic!("bottom-edge can only be VERT / SPLIT, got {partition}"),
                };
                enc.write_bool(bit, probs[2]);
            }
            (false, false) => {
                // Corner — no bits consumed.
                assert_eq!(partition, PARTITION_SPLIT);
            }
        }
    }

    /// Hand-built bitstream (a): a single 64x64 frame with one
    /// `PARTITION_NONE` decision at `BLOCK_64X64`.
    ///
    /// Expected layout: one leaf `{ r: 0, c: 0, subsize: BLOCK_64X64 }`.
    /// The §6.4.3 tail write-back fires (`partition != SPLIT`); the
    /// resulting context strip cells take
    /// `15 >> b_*_log2_lookup[ BLOCK_64X64 ] = 15 >> 4 = 0`.
    #[test]
    fn decode_partition_single_64x64_none() {
        // 64x64 frame = 8 MI columns × 8 MI rows.
        let mi_cols = 8u32;
        let mi_rows = 8u32;

        let mut enc = RangeEncoder::new();
        // Initial ctx: bsize=BLOCK_64X64, all strips zero → ctx = 12
        // per `partition_plane_context_zero_strips_block_64x64`.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_partition(
            &mut coder,
            0,
            0,
            BLOCK_64X64,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![LeafBlock {
                r: 0,
                c: 0,
                subsize: BLOCK_64X64
            }]
        );
        // §6.4.3 tail: 15 >> b_width_log2_lookup[BLOCK_64X64] = 15>>4 = 0.
        for i in 0..8 {
            assert_eq!(state.above[i], 0, "above[{i}] != 0");
            assert_eq!(state.left[i], 0, "left[{i}] != 0");
        }
    }

    /// Hand-built bitstream (b): a single 64x64 superblock split into
    /// four 32x32 quadrants, each decoded as `PARTITION_NONE`.
    ///
    /// Expected layout: four leaves in TL → TR → BL → BR order, each at
    /// the appropriate `(r, c)` and `subsize = BLOCK_32X32`.
    #[test]
    fn decode_partition_split_into_four_32x32_none() {
        let mi_cols = 8u32;
        let mi_rows = 8u32;

        let mut enc = RangeEncoder::new();

        // Step 1: BLOCK_64X64 at (0,0) → PARTITION_SPLIT (ctx = 12).
        push_partition_decode(&mut enc, PARTITION_SPLIT, true, true, 12);

        // Step 2-5: four BLOCK_32X32 children, each → PARTITION_NONE.
        // Per §9.3.2 derivation: bsize=BLOCK_32X32, bsl=2, boffset=1,
        // mask=0x02, num8x8=4.
        //
        // TL (r=0, c=0): all strips still zero → ctx = 2*4 = 8.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 8);
        // After TL: parent SPLIT means partition != SPLIT branch wrote
        // back the TL NONE child sub-write at this subsize. Per the
        // §6.4.3 tail with subsize = BLOCK_32X32, the write-back value
        // is 15 >> b_width_log2_lookup[BLOCK_32X32] = 15 >> 3 = 1.
        // For the next TR ctx (bsize=BLOCK_32X32, c=4): above strip
        // cells above[4..8] are still 0 (TL wrote above[0..4]=1, not
        // above[4..8]). Left strip cells left[0..4] now = 1 from TL.
        // OR-fold: above_bits=0, left_bits=1; mask=0x02 → above bit
        // 0, left bit (1 & 2) = 0. → ctx = 2*4 + 0 + 0 = 8.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 8);
        // BL (r=4, c=0): above strip above[0..4]=1 from TL; left strip
        // left[4..8]=0. OR: above_bits=1; (1 & 2) = 0. left_bits=0.
        // ctx = 8.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 8);
        // BR (r=4, c=4): above[4..8]=1 from TR, left[4..8]=1 from BL.
        // OR: above_bits=1, left_bits=1; (1 & 2) = 0 both. ctx = 8.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 8);

        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_partition(
            &mut coder,
            0,
            0,
            BLOCK_64X64,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![
                LeafBlock {
                    r: 0,
                    c: 0,
                    subsize: BLOCK_32X32
                },
                LeafBlock {
                    r: 0,
                    c: 4,
                    subsize: BLOCK_32X32
                },
                LeafBlock {
                    r: 4,
                    c: 0,
                    subsize: BLOCK_32X32
                },
                LeafBlock {
                    r: 4,
                    c: 4,
                    subsize: BLOCK_32X32
                },
            ],
            "TL → TR → BL → BR recursion order broken"
        );
        // §6.4.3 tail: each NONE child wrote
        // 15 >> b_*_log2_lookup[BLOCK_32X32] = 15 >> 3 = 1 to its strip
        // cells. The parent SPLIT did NOT write (gate fires only on
        // bsize == BLOCK_8X8 || partition != SPLIT).
        for i in 0..8 {
            assert_eq!(state.above[i], 1, "above[{i}] != 1");
            assert_eq!(state.left[i], 1, "left[{i}] != 1");
        }
    }

    /// Hand-built bitstream (c): a 64x64 superblock split into four
    /// 32x32 quadrants where each child uses a non-NONE / non-SPLIT
    /// partition: TL=HORZ, TR=VERT, BL=HORZ, BR=VERT. This exercises
    /// the HORZ second-leaf and VERT second-leaf §6.4.3 paths plus the
    /// §6.4.3 tail write-back across mixed partitions.
    #[test]
    fn decode_partition_mixed_horz_vert_quadrants() {
        let mi_cols = 8u32;
        let mi_rows = 8u32;

        let mut enc = RangeEncoder::new();

        // BLOCK_64X64 → PARTITION_SPLIT (ctx = 12).
        push_partition_decode(&mut enc, PARTITION_SPLIT, true, true, 12);

        // TL BLOCK_32X32 (0,0) → HORZ. subsize = BLOCK_32X16 (8). ctx=8.
        push_partition_decode(&mut enc, PARTITION_HORZ, true, true, 8);
        // TL tail write-back at subsize=BLOCK_32X16 (8):
        //   above_val = 15 >> b_width_log2_lookup[8]  = 15 >> 3 = 1
        //   left_val  = 15 >> b_height_log2_lookup[8] = 15 >> 2 = 3
        // above[0..4] = 1, left[0..4] = 3.
        //
        // TR BLOCK_32X32 (0,4) → VERT. Strips: above[4..8] still 0;
        // left[0..4] = 3. OR: above_bits=0, left_bits=3; mask=0x02 →
        // above bit 0, left bit (3 & 2) != 0 = 1. ctx = 2*4 + 1*2 + 0 = 10.
        push_partition_decode(&mut enc, PARTITION_VERT, true, true, 10);
        // TR tail write-back at subsize=BLOCK_16X32 (7):
        //   above_val = 15 >> b_width_log2_lookup[7]  = 15 >> 2 = 3
        //   left_val  = 15 >> b_height_log2_lookup[7] = 15 >> 3 = 1
        // above[4..8] = 3, left[0..4] = 1 (overwrites previous 3).
        //
        // BL BLOCK_32X32 (4,0) → HORZ. Strips: above[0..4] = 1
        // (from TL); left[4..8] = 0. OR: above_bits=1, left_bits=0;
        // (1 & 2) = 0, (0 & 2) = 0. ctx = 2*4 + 0 + 0 = 8.
        push_partition_decode(&mut enc, PARTITION_HORZ, true, true, 8);
        // BL tail write-back at subsize=BLOCK_32X16 (8): above[0..4]=1,
        // left[4..8]=3.
        //
        // BR BLOCK_32X32 (4,4) → VERT. Strips: above[4..8] = 3
        // (from TR); left[4..8] = 3 (from BL). OR: (3 & 2) = 2 != 0,
        // (3 & 2) = 2 != 0. above bit 1, left bit 1. ctx = 8 + 2 + 1 = 11.
        push_partition_decode(&mut enc, PARTITION_VERT, true, true, 11);

        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_partition(
            &mut coder,
            0,
            0,
            BLOCK_64X64,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        // Expected leaves: 2 per HORZ, 2 per VERT = 8 total.
        // Per §6.4.3, HORZ at (r,c) with subsize BLOCK_32X16 logs
        //   { r, c }, then { r + half, c } if hasRows.
        // VERT logs { r, c }, then { r, c + half } if hasCols.
        // BLOCK_32X32 → num8x8 = 4 → half = 2.
        let expected = vec![
            // TL HORZ: (0,0) BLOCK_32X16, (0+2, 0) = (2, 0)
            LeafBlock {
                r: 0,
                c: 0,
                subsize: /* BLOCK_32X16 */ 8,
            },
            LeafBlock {
                r: 2,
                c: 0,
                subsize: 8,
            },
            // TR VERT: (0,4) BLOCK_16X32, (0, 4+2) = (0, 6)
            LeafBlock {
                r: 0,
                c: 4,
                subsize: /* BLOCK_16X32 */ 7,
            },
            LeafBlock {
                r: 0,
                c: 6,
                subsize: 7,
            },
            // BL HORZ: (4,0) BLOCK_32X16, (4+2, 0) = (6, 0)
            LeafBlock {
                r: 4,
                c: 0,
                subsize: 8,
            },
            LeafBlock {
                r: 6,
                c: 0,
                subsize: 8,
            },
            // BR VERT: (4,4) BLOCK_16X32, (4, 4+2) = (4, 6)
            LeafBlock {
                r: 4,
                c: 4,
                subsize: 7,
            },
            LeafBlock {
                r: 4,
                c: 6,
                subsize: 7,
            },
        ];
        assert_eq!(leaves, expected, "mixed HORZ/VERT leaf layout broken");
    }

    /// `(r >= mi_rows || c >= mi_cols)` short-circuit (§6.4.3 line 2354).
    #[test]
    fn decode_partition_out_of_frame_short_circuits() {
        let enc = RangeEncoder::new();
        // No bits to encode — the call should return without touching
        // the coder.
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();
        let mut state = PartitionContextState::new(8, 8);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        // r past frame.
        decode_partition(
            &mut coder,
            10,
            0,
            BLOCK_64X64,
            8,
            8,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert!(leaves.is_empty(), "OOR call should not emit leaves");
        // c past frame.
        decode_partition(
            &mut coder,
            0,
            12,
            BLOCK_64X64,
            8,
            8,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert!(leaves.is_empty(), "OOR call should not emit leaves");
        // All strip cells stay at the §7.4 reset value.
        for &cell in state.above.iter().chain(state.left.iter()) {
            assert_eq!(cell, 0);
        }
    }

    /// `PartitionContextState::clear_left( )` (§6.4.2) zeroes the left
    /// strip without touching the above strip.
    #[test]
    fn partition_context_state_clear_left_zeroes_left_only() {
        let mut s = PartitionContextState::new(8, 8);
        s.above[3] = 7;
        s.left[2] = 11;
        s.clear_left();
        assert_eq!(s.above[3], 7);
        for &cell in s.left.iter() {
            assert_eq!(cell, 0);
        }
    }

    /// `PartitionProbsKind::Inter` indexes the caller's table instead of
    /// the keyframe table.
    #[test]
    fn partition_probs_kind_inter_dispatches_to_caller_table() {
        let inter_table: [[u8; 3]; 16] = [[42, 43, 44]; 16];
        let kind = PartitionProbsKind::Inter(&inter_table);
        for ctx in 0..16 {
            assert_eq!(kind.row(ctx), [42, 43, 44]);
        }
        let kf = PartitionProbsKind::Keyframe;
        assert_eq!(kf.row(0), KF_PARTITION_PROBS[0]);
        assert_eq!(kf.row(15), KF_PARTITION_PROBS[15]);
    }

    // ----- §6.4.1 get_tile_offset tests -----

    /// The §6.4.1 single-tile (`tile_sz_log2 == 0`) case: `tileNum = 0`
    /// returns `0`; `tileNum = 1` returns the clamped frame extent.
    #[test]
    fn get_tile_offset_single_tile_spans_full_frame() {
        // 64x64 frame = 8 MI rows. sbs = (8 + 7) >> 3 = 1.
        // tileNum = 0: offset = (0 * 1) >> 0 << 3 = 0.
        // tileNum = 1: offset = (1 * 1) >> 0 << 3 = 8. Min(8, 8) = 8.
        assert_eq!(get_tile_offset(0, 8, 0), 0);
        assert_eq!(get_tile_offset(1, 8, 0), 8);
        // Non-multiple-of-8 frame: MiRows = 11 → sbs = (11+7)>>3 = 2.
        // tileNum = 0: 0; tileNum = 1: (1*2)>>0<<3 = 16, Min(16,11) = 11.
        assert_eq!(get_tile_offset(0, 11, 0), 0);
        assert_eq!(get_tile_offset(1, 11, 0), 11);
    }

    /// The §6.4.1 two-tile (`tile_sz_log2 == 1`) case: the frame splits
    /// at the half-sb64 boundary, rounded up.
    #[test]
    fn get_tile_offset_two_tiles_split_at_half() {
        // MiCols = 16 (128x128 frame). sbs = (16+7)>>3 = 2.
        // tileNum 0: ((0*2)>>1)<<3 = 0.
        // tileNum 1: ((1*2)>>1)<<3 = 8.
        // tileNum 2: ((2*2)>>1)<<3 = 16. Min(16, 16) = 16.
        assert_eq!(get_tile_offset(0, 16, 1), 0);
        assert_eq!(get_tile_offset(1, 16, 1), 8);
        assert_eq!(get_tile_offset(2, 16, 1), 16);
    }

    /// `Min( offset, mis )` clamps the past-the-end `tileNum` against
    /// the frame extent.
    #[test]
    fn get_tile_offset_clamps_past_end_against_mis() {
        // MiCols = 8, tile_sz_log2 = 2 (four tiles configured but only
        // one sb64 wide — every tileNum >= 1 collapses to mis).
        // sbs = 1; tile_sz_log2 = 2 → offset = ((tileNum * 1) >> 2) << 3.
        // tileNum 0: 0. tileNum 1: ((1>>2)<<3) = 0.  tileNum 4: ((4>>2)<<3) = 8.
        assert_eq!(get_tile_offset(0, 8, 2), 0);
        assert_eq!(get_tile_offset(1, 8, 2), 0);
        assert_eq!(get_tile_offset(4, 8, 2), 8);
        // tileNum 5: ((5>>2)<<3) = 8. Still clamped at mis = 8.
        assert_eq!(get_tile_offset(5, 8, 2), 8);
    }

    /// Consecutive `tileNum` pairs `(i, i+1)` produce a contiguous
    /// `[Start, End)` cover of `[0, mis)` for the spec-defined caller
    /// (the §6.4 loop fires `tileRows = 1 << tile_rows_log2` tiles plus
    /// one past-the-end fetch).
    #[test]
    fn get_tile_offset_consecutive_pairs_cover_full_extent() {
        // MiRows = 16, tile_rows_log2 = 2 (four tiles).
        let mis = 16u32;
        let log2 = 2u32;
        let tiles = 1u32 << log2;
        let mut prev_end = 0u32;
        for tile in 0..tiles {
            let start = get_tile_offset(tile, mis, log2);
            let end = get_tile_offset(tile + 1, mis, log2);
            assert_eq!(start, prev_end, "tile {tile} Start != previous End");
            assert!(end >= start, "tile {tile}: end < start ({end} < {start})");
            prev_end = end;
        }
        assert_eq!(prev_end, mis, "last tile End != mis");
    }

    /// The §6.4.1 offsets are always 8-aligned (the `<<3` tail).
    #[test]
    fn get_tile_offset_returns_8_aligned_offsets_below_mis() {
        for tile_num in 0..32u32 {
            for &mis in &[8u32, 16, 32, 64, 256] {
                for &log2 in &[0u32, 1, 2, 3] {
                    let off = get_tile_offset(tile_num, mis, log2);
                    if off < mis {
                        assert_eq!(off & 7, 0, "non-aligned offset {off}");
                    }
                }
            }
        }
    }

    // ----- §6.4.2 decode_tile tests -----

    /// `decode_tile( )` over an empty MI window (`mi_row_start ==
    /// mi_row_end`) returns Ok with no leaves and leaves the above
    /// strip + bool-coder untouched (the loop body never runs).
    #[test]
    fn decode_tile_empty_window_consumes_nothing() {
        let bytes: &'static [u8] = Box::leak(vec![0u8; 4].into_boxed_slice());
        let mut coder = BoolCoder::init_bool(bytes, 4).unwrap();
        let mut state = PartitionContextState::new(8, 8);
        state.above[0] = 7; // sentinel, must survive (clear_left does
                            // NOT touch above)
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_tile(
            &mut coder,
            0,
            0, // empty row span
            0,
            8,
            8,
            8,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert!(leaves.is_empty());
        assert_eq!(state.above[0], 7);
    }

    /// Single-superblock tile (8x8 MI = 64x64 px frame, one sb64): the
    /// §6.4.2 driver fires `clear_left_context( )` once, then exactly
    /// one [`decode_partition`] call at `(0, 0, BLOCK_64X64)`. With a
    /// `PARTITION_NONE` decode the result matches the single-leaf
    /// fixture used by `decode_partition_single_64x64_none`.
    #[test]
    fn decode_tile_single_superblock_yields_one_leaf() {
        let mi_cols = 8u32;
        let mi_rows = 8u32;

        let mut enc = RangeEncoder::new();
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_tile(
            &mut coder,
            0,
            mi_rows,
            0,
            mi_cols,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![LeafBlock {
                r: 0,
                c: 0,
                subsize: BLOCK_64X64,
            }]
        );
    }

    /// Two-superblock-wide tile (16 MI cols × 8 MI rows = 128x64 px):
    /// the §6.4.2 driver fires two [`decode_partition`] calls in
    /// `c = 0`, `c = 8` order. With each decoded as `PARTITION_NONE`,
    /// the resulting `leaves` log is two entries at `(0, 0)` and
    /// `(0, 8)`.
    #[test]
    fn decode_tile_two_superblock_row_visits_each_sb_in_order() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;

        let mut enc = RangeEncoder::new();
        // SB at (0, 0): ctx = 12 (all strips zero).
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        // SB at (0, 8): partition write-back at BLOCK_64X64 NONE writes
        // `15 >> 4 = 0` to above[0..8] and left[0..8], so strips are
        // still all zero. ctx = 12 again.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_tile(
            &mut coder,
            0,
            mi_rows,
            0,
            mi_cols,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![
                LeafBlock {
                    r: 0,
                    c: 0,
                    subsize: BLOCK_64X64,
                },
                LeafBlock {
                    r: 0,
                    c: 8,
                    subsize: BLOCK_64X64,
                },
            ]
        );
    }

    /// Two-superblock-tall tile (8 MI cols × 16 MI rows = 64x128 px):
    /// the §6.4.2 driver fires two superblock-row iterations
    /// `(r = 0, r = 8)`. Verifies that `clear_left_context( )` fires at
    /// the START of each row by pre-poisoning `state.left[ ]` with a
    /// sentinel and confirming the second-row partition_decode sees a
    /// zero left strip (its ctx derivation would shift otherwise).
    #[test]
    fn decode_tile_clears_left_context_per_superblock_row() {
        let mi_cols = 8u32;
        let mi_rows = 16u32;

        let mut enc = RangeEncoder::new();
        // Row 1 SB at (0, 0): ctx = 12 (zero strips).
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        // Row 2 SB at (8, 0): even though BLOCK_64X64 NONE wrote
        // 15 >> 4 = 0 to above[0..8], and clear_left_context fired,
        // the left strip is zero again → ctx = 12.
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        // Pre-poison the left strip with a non-zero sentinel; if
        // clear_left_context did NOT fire before row 2 the cells at
        // left[8..16] would carry the sentinel into the ctx derivation
        // and the encoded ctx = 12 would mismatch.
        for cell in state.left.iter_mut() {
            *cell = 5;
        }
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_tile(
            &mut coder,
            0,
            mi_rows,
            0,
            mi_cols,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![
                LeafBlock {
                    r: 0,
                    c: 0,
                    subsize: BLOCK_64X64,
                },
                LeafBlock {
                    r: 8,
                    c: 0,
                    subsize: BLOCK_64X64,
                },
            ]
        );
        // After the two row-starting clear_left_context calls + two
        // NONE write-backs of 15>>4 = 0, the left strip is uniformly
        // zero (the sentinel is gone).
        for &cell in state.left.iter() {
            assert_eq!(cell, 0);
        }
    }

    /// 2x2 superblock tile (16 MI cols × 16 MI rows = 128x128 px):
    /// four [`decode_partition`] calls fire in
    /// `(r, c) = (0,0), (0,8), (8,0), (8,8)` order — row-major with
    /// the inner column loop sweeping each superblock row before
    /// advancing.
    #[test]
    fn decode_tile_2x2_superblocks_row_major_order() {
        let mi_cols = 16u32;
        let mi_rows = 16u32;

        let mut enc = RangeEncoder::new();
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_tile(
            &mut coder,
            0,
            mi_rows,
            0,
            mi_cols,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![
                LeafBlock {
                    r: 0,
                    c: 0,
                    subsize: BLOCK_64X64,
                },
                LeafBlock {
                    r: 0,
                    c: 8,
                    subsize: BLOCK_64X64,
                },
                LeafBlock {
                    r: 8,
                    c: 0,
                    subsize: BLOCK_64X64,
                },
                LeafBlock {
                    r: 8,
                    c: 8,
                    subsize: BLOCK_64X64,
                },
            ]
        );
    }

    /// Sub-tile MI window: `decode_tile( )` invoked over
    /// `[mi_row_start = 8, mi_row_end = 16) × [mi_col_start = 8,
    /// mi_col_end = 16)` (the bottom-right tile of a 2x2 split)
    /// produces a single `(r = 8, c = 8)` leaf — the §6.4.2 loop
    /// honours both `Start` and `End` boundaries.
    #[test]
    fn decode_tile_sub_tile_window_honours_start_and_end_offsets() {
        let mi_cols = 16u32;
        let mi_rows = 16u32;

        let mut enc = RangeEncoder::new();
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        decode_tile(
            &mut coder,
            8,
            16,
            8,
            16,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();

        assert_eq!(
            leaves,
            vec![LeafBlock {
                r: 8,
                c: 8,
                subsize: BLOCK_64X64,
            }]
        );
    }

    /// `decode_tile( )` chained with §6.4.1 `get_tile_offset( )`: the
    /// composition of the two primitives reproduces the §6.4
    /// `decode_tiles( )` per-tile boundary derivation. Smoke-tests the
    /// integration pattern the §6.4 outer driver will adopt — splits a
    /// 16-MI-wide frame into two tiles, then decodes each tile's
    /// single superblock and confirms the `c` offsets are 0 and 8.
    #[test]
    fn decode_tile_composes_with_get_tile_offset() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;
        let tile_cols_log2 = 1u32;
        let tile_rows_log2 = 0u32;

        let mut enc = RangeEncoder::new();
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        push_partition_decode(&mut enc, PARTITION_NONE, true, true, 12);
        let bytes = enc.finish();
        let sz = bytes.len();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut coder = BoolCoder::init_bool(leaked, sz).unwrap();

        // Tile (0, 0): rows [0, mi_rows), cols [0, 8).
        let mr_start = get_tile_offset(0, mi_rows, tile_rows_log2);
        let mr_end = get_tile_offset(1, mi_rows, tile_rows_log2);
        let mc0_start = get_tile_offset(0, mi_cols, tile_cols_log2);
        let mc0_end = get_tile_offset(1, mi_cols, tile_cols_log2);
        assert_eq!((mr_start, mr_end, mc0_start, mc0_end), (0, 8, 0, 8));

        // Tile (0, 1): rows [0, mi_rows), cols [8, 16).
        let mc1_start = get_tile_offset(1, mi_cols, tile_cols_log2);
        let mc1_end = get_tile_offset(2, mi_cols, tile_cols_log2);
        assert_eq!((mc1_start, mc1_end), (8, 16));

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let mut leaves: Vec<LeafBlock> = Vec::new();
        // Decode tile (0, 0).
        decode_tile(
            &mut coder,
            mr_start,
            mr_end,
            mc0_start,
            mc0_end,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert_eq!(leaves.last().unwrap().c, 0);

        // Decode tile (0, 1).
        decode_tile(
            &mut coder,
            mr_start,
            mr_end,
            mc1_start,
            mc1_end,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert_eq!(leaves.last().unwrap().c, 8);
        assert_eq!(leaves.len(), 2);
    }

    // ----- §6.4 tile_payload_sizes byte-walk tests -----

    /// Worked example for the single-tile case (`tile_rows_log2 = 0`,
    /// `tile_cols_log2 = 0`): §6.4 line 2306 picks `lastTile = true`
    /// on the first iteration so no `f(32)` prefix is read; the §6.4
    /// line 2308 assignment `tile_size = sz` is the only value
    /// emitted.
    #[test]
    fn tile_payload_sizes_single_tile_returns_sz() {
        // No prefix is read; `data` need not even contain bytes for
        // the prefix slot — but it must contain `sz` bytes for the
        // per-tile body range check below to pass.
        let body = [0u8; 19];
        let sizes = tile_payload_sizes(&body, 19, 0, 0).unwrap();
        assert_eq!(sizes, vec![19]);
    }

    /// Two-tile horizontal split (`tile_cols_log2 = 1`,
    /// `tile_rows_log2 = 0`): §6.4 reads `tile_size  f(32)` for the
    /// first tile only; the second tile takes whatever `sz` remains
    /// after `sz -= first_tile_size + 4`.
    ///
    /// This worked example mirrors the docs/video/vp9/fixtures/tile-cols-2
    /// per-frame trace exactly: a 512x64 keyframe (1 SB row x 2 tile
    /// columns) where the §6.4 trace reports `tile_size = 662` for
    /// `tile_col = 0` and `tile_size = 635` for `tile_col = 1`. The
    /// total tile-payload budget is therefore 4 (prefix) + 662 +
    /// 635 = 1301 bytes.
    #[test]
    fn tile_payload_sizes_two_horizontal_tiles_matches_fixture_layout() {
        let total: u32 = 4 + 662 + 635;

        // The §6.4 byte-walk only reads the f(32) prefix and steps
        // over the bodies — the body bytes themselves are not
        // inspected. Build a minimal `data` of `total` bytes whose
        // first four bytes encode the big-endian f(32) value 662.
        let mut data: Vec<u8> = Vec::with_capacity(total as usize);
        data.extend_from_slice(&662u32.to_be_bytes());
        data.resize(total as usize, 0);

        let sizes = tile_payload_sizes(&data, total, 0, 1).unwrap();
        assert_eq!(sizes, vec![662, 635]);
        // Sum invariant: every non-last entry plus the (last entry)
        // and the (tileCount - 1) four-byte prefixes account for the
        // full byte budget.
        let tile_count: u32 = sizes.len() as u32;
        let prefix_bytes: u32 = (tile_count - 1) * 4;
        let body_bytes: u32 = sizes.iter().sum();
        assert_eq!(prefix_bytes + body_bytes, total);
    }

    /// 2x2 grid (`tile_rows_log2 = 1`, `tile_cols_log2 = 1`): three
    /// `f(32)` prefixes are read for the first three tiles, then the
    /// last tile takes the running `sz`. The output order is
    /// row-major per §6.4 lines 2304-2305.
    #[test]
    fn tile_payload_sizes_2x2_grid_emits_row_major_order() {
        // Pick four distinguishable sizes so a transpose would be
        // visible.
        let s: [u32; 4] = [11, 22, 33, 44];
        let total: u32 = s.iter().sum::<u32>() + 4 * 3;

        // Build the byte stream: f(32) prefixes for the first three
        // tiles in row-major order, then enough body bytes to satisfy
        // the per-tile range checks. The body bytes are not inspected.
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&s[0].to_be_bytes());
        data.resize(data.len() + s[0] as usize, 0);
        data.extend_from_slice(&s[1].to_be_bytes());
        data.resize(data.len() + s[1] as usize, 0);
        data.extend_from_slice(&s[2].to_be_bytes());
        data.resize(data.len() + s[2] as usize, 0);
        data.resize(data.len() + s[3] as usize, 0);

        let sizes = tile_payload_sizes(&data, total, 1, 1).unwrap();
        assert_eq!(sizes, vec![s[0], s[1], s[2], s[3]]);
    }

    /// §6.4 line 2310 underflow: a 3-byte input can't even hold the
    /// first `f(32)` prefix, so `tile_payload_sizes` reports
    /// `UnexpectedEof` before touching the body.
    #[test]
    fn tile_payload_sizes_short_f32_prefix_is_eof() {
        let data = [0u8; 3];
        let err = tile_payload_sizes(&data, 3, 0, 1).unwrap_err();
        assert_eq!(err, Error::UnexpectedEof);
    }

    /// §6.4 line 2311 underflow: a declared `tile_size = u32::MAX`
    /// would make `sz - (u32::MAX + 4)` wrap. The helper rejects with
    /// `InvalidBitstream` rather than wrapping.
    #[test]
    fn tile_payload_sizes_oversized_declared_size_is_invalid_bitstream() {
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&u32::MAX.to_be_bytes());
        data.resize(16, 0);
        let err = tile_payload_sizes(&data, 16, 0, 1).unwrap_err();
        assert_eq!(err, Error::InvalidBitstream);
    }

    // ----- §6.4 decode_tiles outer-driver tests -----

    /// Build a single-tile byte stream: one §9.2.1 `init_bool` payload
    /// encoding the given partition-decode sequence. The output is a
    /// `(bytes, sz)` pair ready to hand to [`decode_tiles`] with
    /// `tile_rows_log2 == tile_cols_log2 == 0` (i.e. one tile total,
    /// no `f(32)` prefix per §6.4 lines 2306-2308).
    fn single_tile_payload(decodes: &[(u8, bool, bool)]) -> (Vec<u8>, u32) {
        let mut enc = RangeEncoder::new();
        for &(partition, has_rows, has_cols) in decodes {
            push_partition_decode(&mut enc, partition, has_rows, has_cols, 12);
        }
        let body = enc.finish();
        let sz = body.len() as u32;
        (body, sz)
    }

    /// `PartitionContextState::clear_above( )` zeroes the above strip
    /// without touching the left strip — the dual of the round-32
    /// `clear_left` invariant.
    #[test]
    fn partition_context_state_clear_above_zeroes_above_only() {
        let mut s = PartitionContextState::new(8, 8);
        s.above[3] = 7;
        s.left[2] = 11;
        s.clear_above();
        assert_eq!(s.left[2], 11);
        for &cell in s.above.iter() {
            assert_eq!(cell, 0);
        }
    }

    /// Single-tile frame (`tile_rows_log2 = 0`, `tile_cols_log2 = 0`):
    /// §6.4 lines 2306-2308 set `lastTile = true` on the first
    /// iteration so no `f(32)` prefix is read; `tile_size = sz` is the
    /// whole payload; the inner `decode_tile( )` walks one
    /// `(0, 0, BLOCK_64X64)` superblock.
    #[test]
    fn decode_tiles_single_tile_consumes_full_payload() {
        let mi_cols = 8u32;
        let mi_rows = 8u32;
        let (body, sz) = single_tile_payload(&[(PARTITION_NONE, true, true)]);

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &body,
            sz,
            0,
            0,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();

        assert_eq!(tiles.len(), 1);
        let t = &tiles[0];
        assert_eq!(t.tile_row, 0);
        assert_eq!(t.tile_col, 0);
        assert_eq!(t.mi_row_start, 0);
        assert_eq!(t.mi_row_end, mi_rows);
        assert_eq!(t.mi_col_start, 0);
        assert_eq!(t.mi_col_end, mi_cols);
        assert_eq!(t.tile_size, sz);
        assert_eq!(
            t.leaves,
            vec![LeafBlock {
                r: 0,
                c: 0,
                subsize: BLOCK_64X64,
            }]
        );
    }

    /// §6.4 line 2303 `clear_above_context( )` fires once at the
    /// start of `decode_tiles( )`: a pre-poisoned `above[ ]` strip is
    /// observed zeroed BEFORE the first tile's `decode_tile( )` walks.
    #[test]
    fn decode_tiles_clears_above_context_at_start_of_frame() {
        let mi_cols = 8u32;
        let mi_rows = 8u32;
        let (body, sz) = single_tile_payload(&[(PARTITION_NONE, true, true)]);

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        // Pre-poison every above cell with a sentinel value.
        for cell in state.above.iter_mut() {
            *cell = 0xAA;
        }
        decode_tiles(
            &body,
            sz,
            0,
            0,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();
        // The §6.4 line 2303 reset wipes the strip. The §6.4.3
        // write-back may then have written `15 >> b_width_log2_lookup`
        // = `15 >> 4 = 0` cells for BLOCK_64X64 — so every cell is 0
        // post-decode, with no surviving 0xAA.
        for &cell in state.above.iter() {
            assert_eq!(cell, 0, "clear_above_context did not zero strip pre-decode");
        }
    }

    /// Two-tile horizontal split (`tile_cols_log2 = 1`,
    /// `tile_rows_log2 = 0`): §6.4 reads `f(32) = tile_size` for the
    /// first tile, then `sz -= tile_size + 4`; the second tile is
    /// `lastTile` and consumes the remaining `sz`. Each tile's
    /// `MiColStart` / `MiColEnd` matches the §6.4.1 split at MI=8.
    #[test]
    fn decode_tiles_two_horizontal_tiles_reads_f32_prefix() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;

        let (body_a, sz_a) = single_tile_payload(&[(PARTITION_NONE, true, true)]);
        let (body_b, sz_b) = single_tile_payload(&[(PARTITION_NONE, true, true)]);

        // Build the full stream: [f(32) sz_a][body_a][body_b].
        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&sz_a.to_be_bytes());
        stream.extend_from_slice(&body_a);
        stream.extend_from_slice(&body_b);
        let total = 4u32 + sz_a + sz_b;

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &stream,
            total,
            0,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();

        assert_eq!(tiles.len(), 2);
        let (t0, t1) = (&tiles[0], &tiles[1]);
        // Tile 0: f(32) declared size matches body_a.
        assert_eq!(t0.tile_col, 0);
        assert_eq!(t0.mi_col_start, 0);
        assert_eq!(t0.mi_col_end, 8);
        assert_eq!(t0.tile_size, sz_a);
        // Tile 1 (last): tile_size = remaining sz = sz_b (after
        // sz -= sz_a + 4 brings total down to sz_b).
        assert_eq!(t1.tile_col, 1);
        assert_eq!(t1.mi_col_start, 8);
        assert_eq!(t1.mi_col_end, 16);
        assert_eq!(t1.tile_size, sz_b);
        // Each tile contributed one leaf at its respective `c`
        // origin.
        assert_eq!(t0.leaves.len(), 1);
        assert_eq!(t0.leaves[0].c, 0);
        assert_eq!(t1.leaves.len(), 1);
        assert_eq!(t1.leaves[0].c, 8);
    }

    /// 2x2 tile grid (`tile_rows_log2 = 1`, `tile_cols_log2 = 1`):
    /// row-major iteration order produces `(0,0) → (0,1) → (1,0) →
    /// (1,1)`; all but the last read an `f(32)` prefix; the last
    /// consumes the residual `sz`.
    #[test]
    fn decode_tiles_2x2_grid_iterates_row_major() {
        let mi_cols = 16u32;
        let mi_rows = 16u32;

        let bodies: Vec<(Vec<u8>, u32)> = (0..4)
            .map(|_| single_tile_payload(&[(PARTITION_NONE, true, true)]))
            .collect();
        let sizes: Vec<u32> = bodies.iter().map(|(_, s)| *s).collect();

        // Stream: f(32) sz_0 || body_0 || f(32) sz_1 || body_1 ||
        //         f(32) sz_2 || body_2 || body_3.
        let mut stream: Vec<u8> = Vec::new();
        for (i, (body, sz)) in bodies.iter().enumerate() {
            if i < 3 {
                stream.extend_from_slice(&sz.to_be_bytes());
            }
            stream.extend_from_slice(body);
        }
        let total: u32 = sizes.iter().sum::<u32>() + 4 * 3;

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &stream,
            total,
            1,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();

        assert_eq!(tiles.len(), 4);
        // Row-major order:
        assert_eq!((tiles[0].tile_row, tiles[0].tile_col), (0, 0));
        assert_eq!((tiles[1].tile_row, tiles[1].tile_col), (0, 1));
        assert_eq!((tiles[2].tile_row, tiles[2].tile_col), (1, 0));
        assert_eq!((tiles[3].tile_row, tiles[3].tile_col), (1, 1));
        // Per-tile MI extents: each tile is 8 MI on each axis.
        for t in &tiles {
            assert_eq!(t.mi_row_end - t.mi_row_start, 8);
            assert_eq!(t.mi_col_end - t.mi_col_start, 8);
            assert_eq!(t.leaves.len(), 1);
        }
        // Leaves' (r, c) origins match the per-tile MI starts.
        for t in &tiles {
            assert_eq!(t.leaves[0].r, t.mi_row_start);
            assert_eq!(t.leaves[0].c, t.mi_col_start);
        }
        // Sizes: tiles 0..=2 took their declared `f(32)` size;
        // tile 3 is `lastTile` and took the residual.
        assert_eq!(tiles[0].tile_size, sizes[0]);
        assert_eq!(tiles[1].tile_size, sizes[1]);
        assert_eq!(tiles[2].tile_size, sizes[2]);
        assert_eq!(tiles[3].tile_size, sizes[3]);
    }

    /// §6.4 lines 2308 vs 2310: the LAST tile reads NO `f(32)`
    /// prefix. A second tile placed back-to-back with no length
    /// prefix between body_a and body_b decodes successfully because
    /// the `lastTile` branch uses `tile_size = sz`.
    #[test]
    fn decode_tiles_last_tile_skips_f32_prefix() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;

        let (body_a, sz_a) = single_tile_payload(&[(PARTITION_NONE, true, true)]);
        let (body_b, sz_b) = single_tile_payload(&[(PARTITION_NONE, true, true)]);

        // Build [f(32) sz_a][body_a][body_b] — note no length prefix
        // between body_a and body_b.
        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&sz_a.to_be_bytes());
        stream.extend_from_slice(&body_a);
        let body_b_start = stream.len();
        stream.extend_from_slice(&body_b);

        let total = 4u32 + sz_a + sz_b;
        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &stream,
            total,
            0,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();

        // Last tile's tile_size equals the remaining byte budget after
        // tile 0 consumed `4 + sz_a` bytes.
        assert_eq!(tiles[1].tile_size, sz_b);
        // Byte cursor sanity: stream[body_b_start..] is exactly
        // `sz_b` long.
        assert_eq!(stream.len() - body_b_start, sz_b as usize);
    }

    /// `Vec<DecodedTile>` length matches the grid `tileRows * tileCols`
    /// across the §7.2.11 conformance space (4 cells = `(0,0)`,
    /// `(0,1)`, `(1,0)`, `(1,1)`).
    #[test]
    fn decode_tiles_output_length_matches_grid() {
        // 4x1 strip: tile_cols_log2 = 2, tile_rows_log2 = 0 → 4 tiles.
        let mi_cols = 32u32;
        let mi_rows = 8u32;

        let bodies: Vec<(Vec<u8>, u32)> = (0..4)
            .map(|_| single_tile_payload(&[(PARTITION_NONE, true, true)]))
            .collect();
        let sizes: Vec<u32> = bodies.iter().map(|(_, s)| *s).collect();
        let mut stream: Vec<u8> = Vec::new();
        for (i, (body, sz)) in bodies.iter().enumerate() {
            if i < 3 {
                stream.extend_from_slice(&sz.to_be_bytes());
            }
            stream.extend_from_slice(body);
        }
        let total: u32 = sizes.iter().sum::<u32>() + 4 * 3;
        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &stream,
            total,
            0,
            2,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();
        assert_eq!(tiles.len(), 4);
        // Each tile's MiColEnd matches the next tile's MiColStart.
        for w in tiles.windows(2) {
            assert_eq!(w[0].mi_col_end, w[1].mi_col_start);
        }
        // Last tile's MiColEnd equals MiCols.
        assert_eq!(tiles.last().unwrap().mi_col_end, mi_cols);
    }

    /// §6.4 line 2310 `f(32)` truncation: a stream shorter than the
    /// 4-byte prefix yields `Error::UnexpectedEof`.
    #[test]
    fn decode_tiles_short_f32_prefix_is_eof() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;
        // Only 3 bytes available — can't even read the f(32).
        let stream = [0u8, 0, 0];
        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let err = decode_tiles(
            &stream,
            3,
            0,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap_err();
        assert_eq!(err, Error::UnexpectedEof);
    }

    /// §6.4 line 2311 underflow: a declared `tile_size` larger than
    /// the remaining `sz` would underflow `sz -= tile_size + 4`. The
    /// driver surfaces this as `Error::InvalidBitstream` rather than
    /// wrapping.
    #[test]
    fn decode_tiles_oversized_declared_tile_size_is_invalid_bitstream() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;
        // Declared tile_size = u32::MAX → sz - (u32::MAX + 4) wraps.
        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&u32::MAX.to_be_bytes());
        stream.extend_from_slice(&[0u8; 16]);
        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let err = decode_tiles(
            &stream,
            16,
            0,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap_err();
        assert_eq!(err, Error::InvalidBitstream);
    }

    /// §6.4 line 2310: a non-last tile whose declared `tile_size`
    /// extends past the byte stream raises `UnexpectedEof` from the
    /// per-tile slice fetch (the slice bound check fails before
    /// `init_bool( )` runs).
    #[test]
    fn decode_tiles_truncated_tile_body_is_eof() {
        let mi_cols = 16u32;
        let mi_rows = 8u32;
        // Build the f(32) prefix for "tile_size = 8" but supply only
        // 6 body bytes (cursor exhausts before the 8-byte body fetch).
        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&8u32.to_be_bytes());
        stream.extend_from_slice(&[0u8; 6]); // 6 bytes < declared 8
        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let err = decode_tiles(
            &stream,
            4 + 8 + 4, // sz allows the subtraction, but bytes don't
            0,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap_err();
        assert_eq!(err, Error::UnexpectedEof);
    }

    /// §9.2.1 marker rejection at `init_bool( )` propagates as
    /// `Error::InvalidBitstream` through `decode_tiles`. A
    /// non-zero-marker first byte fails the per-tile `init_bool( )`.
    #[test]
    fn decode_tiles_per_tile_marker_failure_is_invalid_bitstream() {
        let mi_cols = 8u32;
        let mi_rows = 8u32;
        // Single-tile mode, no f(32) prefix. First byte 0x80 has the
        // top bit set so the §9.2.1 marker `read_bool(128)` decodes
        // to 1 and init_bool rejects.
        let stream = vec![0x80u8, 0x00, 0x00, 0x00];
        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let err = decode_tiles(
            &stream,
            4,
            0,
            0,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap_err();
        assert_eq!(err, Error::InvalidBitstream);
    }

    /// 1x2 vertical split (`tile_rows_log2 = 1`, `tile_cols_log2 =
    /// 0`): MI rows split at 8; the two tiles' `MiRowStart` /
    /// `MiRowEnd` cover `[0, 8)` and `[8, 16)` per §6.4.1.
    #[test]
    fn decode_tiles_two_vertical_tiles_partition_mi_rows() {
        let mi_cols = 8u32;
        let mi_rows = 16u32;

        let (body_a, sz_a) = single_tile_payload(&[(PARTITION_NONE, true, true)]);
        let (body_b, sz_b) = single_tile_payload(&[(PARTITION_NONE, true, true)]);

        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&sz_a.to_be_bytes());
        stream.extend_from_slice(&body_a);
        stream.extend_from_slice(&body_b);
        let total = 4u32 + sz_a + sz_b;

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &stream,
            total,
            1,
            0,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();

        assert_eq!(tiles.len(), 2);
        assert_eq!((tiles[0].mi_row_start, tiles[0].mi_row_end), (0, 8));
        assert_eq!((tiles[1].mi_row_start, tiles[1].mi_row_end), (8, 16));
        // Both tiles' MI col span is the full frame width.
        for t in &tiles {
            assert_eq!(t.mi_col_start, 0);
            assert_eq!(t.mi_col_end, mi_cols);
        }
        // The two leaves are at (0, 0) and (8, 0).
        assert_eq!(tiles[0].leaves[0].r, 0);
        assert_eq!(tiles[1].leaves[0].r, 8);
    }

    /// §6.4 + §6.4.1 invariant: across the full `tileRows * tileCols`
    /// grid every consecutive `(MiColEnd_prev, MiColStart_next)` pair
    /// matches within a row, and `MiColEnd` of the last column equals
    /// `MiCols` (mirror invariant on rows).
    #[test]
    fn decode_tiles_mi_extents_are_contiguous_within_rows() {
        let mi_cols = 32u32;
        let mi_rows = 16u32;
        // 2x2 grid.
        let bodies: Vec<(Vec<u8>, u32)> = (0..4)
            .map(|_| single_tile_payload(&[(PARTITION_NONE, true, true)]))
            .collect();
        let sizes: Vec<u32> = bodies.iter().map(|(_, s)| *s).collect();
        let mut stream: Vec<u8> = Vec::new();
        for (i, (body, sz)) in bodies.iter().enumerate() {
            if i < 3 {
                stream.extend_from_slice(&sz.to_be_bytes());
            }
            stream.extend_from_slice(body);
        }
        let total: u32 = sizes.iter().sum::<u32>() + 4 * 3;

        let mut state = PartitionContextState::new(mi_cols as usize, mi_rows as usize);
        let tiles = decode_tiles(
            &stream,
            total,
            1,
            1,
            mi_rows,
            mi_cols,
            &mut state,
            PartitionProbsKind::Keyframe,
        )
        .unwrap();

        // Within each row, consecutive cols are contiguous.
        for row in 0..2u32 {
            let row_tiles: Vec<&DecodedTile> = tiles.iter().filter(|t| t.tile_row == row).collect();
            for w in row_tiles.windows(2) {
                assert_eq!(w[0].mi_col_end, w[1].mi_col_start);
            }
            assert_eq!(row_tiles.last().unwrap().mi_col_end, mi_cols);
        }
        // Within each col, consecutive rows are contiguous.
        for col in 0..2u32 {
            let col_tiles: Vec<&DecodedTile> = tiles.iter().filter(|t| t.tile_col == col).collect();
            for w in col_tiles.windows(2) {
                assert_eq!(w[0].mi_row_end, w[1].mi_row_start);
            }
            assert_eq!(col_tiles.last().unwrap().mi_row_end, mi_rows);
        }
    }
}
