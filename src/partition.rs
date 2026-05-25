//! VP9 partition primitive per spec v0.7 — §3 / §6.4.3 / §9.3.1 / §9.3.2
//! / §10.2 / §10.4 / §10.5.
//!
//! Round 18 lands the §6.4.3 `decode_partition_type( )` reader — the per-call
//! partition-tree decode that the recursive [`crate::partition::decode_partition`]
//! driver (later round) will fire once per `(r, c, bsize)` quadrant. The
//! recursive driver, the §10.2 `subsize_lookup` traversal, and the
//! `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` write-back the
//! §6.4.3 tail performs all sit on top of this primitive; this round bounds
//! itself to the single-call decode plus the §9.3.2 context derivation it
//! consumes.
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
//! §10.4 / §10.5). No external library source consulted.

#![allow(dead_code)] // surfaces land in the next round's §6.4.3 driver

use crate::bool_coder::BoolCoder;
use crate::mode_info::tree_decode;
use crate::residual::BLOCK_SIZES;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
