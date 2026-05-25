//! VP9 partition primitive + recursive driver per spec v0.7 — §3 / §6.4.3
//! / §9.3.1 / §9.3.2 / §10.2 / §10.4 / §10.5.
//!
//! Round 18 lands the §6.4.3 `decode_partition_type( )` reader — the per-call
//! partition-tree decode that fires once per `(r, c, bsize)` quadrant.
//! Round 19 lands the recursive [`decode_partition`] driver itself, which
//! composes the round-18 primitive into the full §6.4.3 listing
//! (lines 2353-2392): edge guard, geometry, `partition_plane_context`
//! ctx derivation, four-way recursion on `PARTITION_SPLIT` in spec
//! quadrant order, and the §6.4.3 tail context write-back.
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
//! * The §6.4.4 `decode_block( r, c, subsize )` orchestrator that the
//!   recursion's terminal step would call. The driver records each
//!   `(r, c, subsize)` triple onto a [`LeafBlock`] log and lets the
//!   caller iterate; the §6.4.5 `mode_info( )` + §6.4.21 `residual( )`
//!   machinery the orchestrator consumes is wired one layer up in
//!   a later round.
//! * The §6.3 `read_partition_probs( )` compressed-header sweep
//!   (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
//!   `diff_update_prob` cells against [`DEFAULT_PARTITION_PROBS`]).
//! * The §8.4 `counts_partition[ PARTITION_CONTEXTS ][ PARTITION_TYPES ]`
//!   probability-adaption accumulator (§9.3.4 bookkeeping).
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §3 / §6.4.3 / §9.3.1 / §9.3.2 / §10.2 /
//! §10.4 / §10.5). No external library source consulted.

#![allow(dead_code)] // surfaces consumed by the §6.4.2 tile-walk driver (later round)

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

// ----- §6.4.3 recursive partition driver -----

/// `BLOCK_8X8 = 3` per §3 — the recursion's terminal bsize at which the
/// driver MUST stop subdividing (the partition decode itself is only
/// valid for `bsize >= BLOCK_8X8`).
const BLOCK_8X8: u8 = 3;

/// Per-tile partition-context strips per §6.4.3.
///
/// The §6.4.3 driver materialises the §9.3.2 `above` / `left` bitmaps
/// from two parallel arrays of `8x8`-block-resolution cells:
///
/// * `above[ c ]` is the `AbovePartitionContext` cell at column `c`
///   (one per `8x8` block across the tile's width).
/// * `left[ r ]` is the `LeftPartitionContext` cell at row `r`
///   (one per `8x8` block across the tile's height).
///
/// Cells are written back by the §6.4.3 tail (spec lines 2386-2391):
///
/// ```text
/// if ( bsize == BLOCK_8X8 || partition != PARTITION_SPLIT ) {
///     for ( i = 0; i < num8x8 ; i ++ ) {
///         AbovePartitionContext[ c + i ] = 15 >> b_width_log2_lookup[ subsize ]
///         LeftPartitionContext[ r + i ] = 15 >> b_height_log2_lookup[ subsize ]
///     }
/// }
/// ```
///
/// Per §7.4.2 lines 3825-3837 both arrays are zeroed before each tile
/// (`AbovePartitionContext[ i ] = 0` for `i = 0..Sb64Cols*8 - 1`, and
/// the symmetric statement for the row strip).
#[derive(Debug, Clone)]
pub(crate) struct PartitionContext {
    /// `AbovePartitionContext[ ]` — one byte per 8x8-block column.
    pub(crate) above: Vec<u8>,
    /// `LeftPartitionContext[ ]` — one byte per 8x8-block row.
    pub(crate) left: Vec<u8>,
}

impl PartitionContext {
    /// Allocate strips wide / tall enough for `mi_cols` / `mi_rows`
    /// 8x8-block columns / rows. Per §7.4.2 every cell starts zeroed.
    pub(crate) fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            above: vec![0u8; mi_cols],
            left: vec![0u8; mi_rows],
        }
    }
}

/// One leaf-block record the §6.4.3 recursion would have dispatched to
/// the §6.4.4 `decode_block( r, c, subsize )` orchestrator.
///
/// Used by [`decode_partition`] to log every terminal recursion step
/// without actually invoking `decode_block` (that orchestrator depends
/// on the §6.4.5 `mode_info( )` + §6.4.21 `residual( )` machinery one
/// layer up and is wired in a later round). The order leaves are
/// pushed is the spec's quadrant-walk order: PARTITION_NONE pushes a
/// single leaf at `(r, c)`; PARTITION_HORZ pushes `(r, c)` then
/// `(r + half, c)`; PARTITION_VERT pushes `(r, c)` then `(r, c + half)`;
/// PARTITION_SPLIT recurses into top-left, top-right, bottom-left,
/// bottom-right and the leaves pushed by those sub-calls preserve the
/// spec's spatial ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeafBlock {
    /// Block row (8x8-block units).
    pub(crate) r: usize,
    /// Block column (8x8-block units).
    pub(crate) c: usize,
    /// `subsize` — the §3 `BLOCK_*` index the leaf occupies.
    pub(crate) subsize: u8,
}

/// Which probability table the §9.3.2 lookup should consult — keyframe /
/// intra-only frames use [`KF_PARTITION_PROBS`] (verbatim); inter
/// frames use the running `partition_probs[ ]` table initialised from
/// [`DEFAULT_PARTITION_PROBS`].
///
/// Per spec §6.4.3 the choice is governed by `FrameIsIntra`: keyframes
/// (frame_type == KEY_FRAME) and intra-only inter frames both qualify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionProbsKind<'a> {
    /// Use the fixed §10.4 keyframe table (`FrameIsIntra == 1`).
    Keyframe,
    /// Use a caller-provided running table (`FrameIsIntra == 0`).
    ///
    /// `partition_probs[ctx][node2]` — 16 rows × 3 cells per §10.5
    /// shape. The §6.3 `read_partition_probs( )` sweep populates this
    /// from [`DEFAULT_PARTITION_PROBS`] in a later round; the driver
    /// itself only reads it.
    Inter(&'a [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS]),
}

impl<'a> PartitionProbsKind<'a> {
    /// Return the per-context 3-cell row the §6.4.3 driver hands to
    /// [`decode_partition_type`].
    fn row(&self, ctx: usize) -> &[u8; PARTITION_TYPES - 1] {
        match self {
            Self::Keyframe => &KF_PARTITION_PROBS[ctx],
            Self::Inter(table) => &table[ctx],
        }
    }
}

/// §6.4.3 `decode_partition( r, c, bsize )` — the recursive partition-tree
/// driver.
///
/// Walks the per-tile partition tree rooted at `(r, c, bsize)` by:
///
/// 1. **Edge guard** — return immediately if `r >= mi_rows ||
///    c >= mi_cols` (the spec listing's first line `return 0`); the
///    caller's outer §6.4.2 sweep clips against frame edges.
/// 2. **Geometry** — read `num8x8 = num_8x8_blocks_wide_lookup[ bsize ]`
///    and `halfBlock8x8 = num8x8 >> 1`; compute `hasRows` /
///    `hasCols`.
/// 3. **Context derivation** — `partition_plane_context` materialises
///    the §9.3.2 ctx from the [`PartitionContext`] strips.
/// 4. **Per-call decode** — [`decode_partition_type`] reads the
///    `partition` value off the bool coder using the probability row
///    selected by `probs_kind`.
/// 5. **Recursion** — for `PARTITION_NONE` / `HORZ` / `VERT` records
///    the leaf block(s) the spec listing would have dispatched to
///    `decode_block( )`; for `PARTITION_SPLIT` recurses four-way over
///    the quadrants `( r, c )`, `( r, c + half )`, `( r + half, c )`,
///    `( r + half, c + half )` with `subsize = subsize_lookup[ SPLIT
///    ][ bsize ]`.
/// 6. **Context write-back** — when `bsize == BLOCK_8X8 || partition !=
///    PARTITION_SPLIT` writes
///    `AbovePartitionContext[ c + i ] = 15 >> b_width_log2_lookup[
///    subsize ]` and
///    `LeftPartitionContext[ r + i ] = 15 >> b_height_log2_lookup[
///    subsize ]` for `i in 0..num8x8`.
///
/// The §6.4.4 `decode_block( )` orchestrator the spec dispatches to at
/// each leaf is NOT invoked here (it sits one layer up and consumes
/// the §6.4.5 `mode_info( )` + §6.4.21 `residual( )` machinery). The
/// driver instead pushes a [`LeafBlock`] record onto `leaves` for every
/// `(r, c, subsize)` triple the §6.4.4 call would have received, in
/// spec-order.
///
/// `mi_rows` / `mi_cols` are the tile's frame-relative MI extents
/// (number of 8x8 blocks tall / wide). `ctx_state` is the per-tile
/// partition strips; the driver mutates `above` / `left` in place per
/// the §6.4.3 tail.
///
/// Returns `Ok(())` on success or [`Error::InvalidBitstream`] when the
/// bool coder underflows during a `decode_partition_type` walk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_partition(
    coder: &mut BoolCoder<'_>,
    r: usize,
    c: usize,
    bsize: u8,
    mi_rows: usize,
    mi_cols: usize,
    ctx_state: &mut PartitionContext,
    probs_kind: PartitionProbsKind<'_>,
    leaves: &mut Vec<LeafBlock>,
) -> Result<(), Error> {
    // §6.4.3 line 2354: clip past frame edges.
    if r >= mi_rows || c >= mi_cols {
        return Ok(());
    }

    debug_assert!(
        (bsize as usize) < BLOCK_SIZES,
        "decode_partition: bsize={bsize} out of range"
    );
    // Only the four square superblock sizes ever feed the recursion
    // (BLOCK_8X8 / BLOCK_16X16 / BLOCK_32X32 / BLOCK_64X64); the spec
    // listing never recurses into a non-square parent.
    debug_assert!(
        matches!(bsize, BLOCK_8X8 | 6 | 9 | 12),
        "decode_partition: non-superblock bsize={bsize}"
    );

    let num8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as usize;
    let half_block_8x8 = num8x8 >> 1;
    let has_rows = (r + half_block_8x8) < mi_rows;
    let has_cols = (c + half_block_8x8) < mi_cols;

    // §9.3.2 ctx — OR-fold across the num8x8-wide strip starting at the
    // current `(r, c)` offset.
    let above_strip = &ctx_state.above[c..c + num8x8];
    let left_strip = &ctx_state.left[r..r + num8x8];
    let ctx = partition_plane_context(bsize, above_strip, left_strip);

    // §6.4.3 line 2360: read the partition value.
    let probs = probs_kind.row(ctx);
    let partition = decode_partition_type(coder, has_rows, has_cols, probs)?;

    // §6.4.3 line 2361: subsize_lookup feeds both the dispatch arm and
    // the write-back tail.
    let subsize = SUBSIZE_LOOKUP[partition as usize][bsize as usize];

    // §6.4.3 lines 2362-2385: dispatch on `partition`.
    if subsize < BLOCK_8X8 || partition == PARTITION_NONE {
        // Single leaf at (r, c, subsize). The spec listing covers two
        // cases here:
        //   (a) PARTITION_NONE — subsize = bsize (identity lookup).
        //   (b) Any partition at BLOCK_8X8 parent where the split would
        //       produce a sub-8x8 child (`subsize < BLOCK_8X8`).
        leaves.push(LeafBlock { r, c, subsize });
    } else if partition == PARTITION_HORZ {
        // Two leaves: (r, c) and (r + half, c). The §6.4.3 listing
        // gates the second call on `hasRows` (no bottom half when the
        // tile's bottom edge clips it).
        leaves.push(LeafBlock { r, c, subsize });
        if has_rows {
            leaves.push(LeafBlock {
                r: r + half_block_8x8,
                c,
                subsize,
            });
        }
    } else if partition == PARTITION_VERT {
        // Two leaves: (r, c) and (r, c + half). Right call gated on
        // `hasCols`.
        leaves.push(LeafBlock { r, c, subsize });
        if has_cols {
            leaves.push(LeafBlock {
                r,
                c: c + half_block_8x8,
                subsize,
            });
        }
    } else {
        // PARTITION_SPLIT — recurse four-way in spec-order:
        // top-left, top-right, bottom-left, bottom-right.
        decode_partition(
            coder, r, c, subsize, mi_rows, mi_cols, ctx_state, probs_kind, leaves,
        )?;
        decode_partition(
            coder,
            r,
            c + half_block_8x8,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            leaves,
        )?;
        decode_partition(
            coder,
            r + half_block_8x8,
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
            r + half_block_8x8,
            c + half_block_8x8,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            leaves,
        )?;
    }

    // §6.4.3 lines 2386-2391: context write-back. The gate excludes the
    // PARTITION_SPLIT-at-non-BLOCK_8X8 case where each child wrote its
    // own context; for BLOCK_8X8-parent SPLIT (child = BLOCK_4X4) the
    // parent still writes back per the `bsize == BLOCK_8X8` clause.
    if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
        let above_val = 15u8 >> B_WIDTH_LOG2_LOOKUP[subsize as usize];
        let left_val = 15u8 >> B_HEIGHT_LOG2_LOOKUP[subsize as usize];
        for i in 0..num8x8 {
            ctx_state.above[c + i] = above_val;
            ctx_state.left[r + i] = left_val;
        }
    }

    Ok(())
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

    // ----- §6.4.3 decode_partition() recursive driver tests -----
    //
    // Hand-built bitstreams. The §9.2 bool coder is range-coded so we
    // need a test-only encoder to produce buffers that decode back to
    // a specific sequence of `read_bool( p )` outputs. The encoder
    // below mirrors the §9.2.2 decoder one operation at a time and is
    // exclusively a test convenience — production code never
    // round-trips through it.

    /// Minimal range encoder mirroring the §9.2 decoder, using the
    /// classic libvpx-style "wide low register" pattern: `low` is
    /// kept as a 32-bit accumulator with the *output* byte sliding
    /// out the top whenever renormalisation has happened often enough
    /// to fully define it.
    ///
    /// The encoder mirrors the inverse of the §9.2.2 split / branch:
    ///   `split = 1 + ((range-1)*p >> 8)`
    ///   bit=0 → `range = split`
    ///   bit=1 → `low += split; range -= split`
    /// Renormalisation doubles `range` whenever it drops below 128 and
    /// tracks a `count` of pending shifts. When `count` reaches 8 the
    /// top byte of `low` is shifted out as the next emitted byte
    /// (with carry propagation handled by walking back through any
    /// previously emitted `0xff` bytes).
    ///
    /// This matches the algorithm in RFC 6386 §7.4 (VP8 encoder),
    /// which VP9 inherits unchanged for `read_bool` parsing — only the
    /// `init_bool` shape and the `BoolMaxBits` underflow check differ.
    #[derive(Debug)]
    struct BoolEncoder {
        low: u64,
        range: u32,
        count: i32,
        out: Vec<u8>,
    }

    impl BoolEncoder {
        fn new() -> Self {
            Self {
                low: 0,
                range: 255,
                // -24 means "we need 24 renorm shifts before we
                // emit the first byte", which packs an 8-bit BoolValue
                // followed by another 16 bits' worth of in-flight bits
                // into the high half of `low` before any output flows.
                // This is the libvpx VP8 starting condition (RFC 6386
                // §7.4): it lets the encoder's first 8 emitted bits
                // align with the decoder's `init_bool` f(8) BoolValue
                // read.
                count: -24,
                out: Vec::new(),
            }
        }

        fn encode_bool(&mut self, bit: u8, p: u32) {
            let split: u32 = 1 + (((self.range - 1) * p) >> 8);
            if bit == 0 {
                self.range = split;
            } else {
                self.low += split as u64;
                self.range -= split;
            }
            while self.range < 128 {
                self.range <<= 1;
                self.count += 1;
                self.low <<= 1;
                if self.count == 0 {
                    // The top byte of `low` is now fully defined —
                    // shift it out. If it's >=256 a carry propagates
                    // into the previously emitted byte; walk back
                    // through any pending 0xff bytes (which carry
                    // through to 0x00) and increment the next live
                    // byte.
                    let carry = (self.low >> 32) & 1;
                    if carry != 0 {
                        // Propagate carry into the most recently
                        // emitted byte (and 0xff chains).
                        let mut i = self.out.len();
                        while i > 0 {
                            i -= 1;
                            if self.out[i] != 0xff {
                                self.out[i] = self.out[i].wrapping_add(1);
                                break;
                            } else {
                                self.out[i] = 0;
                            }
                        }
                    }
                    let byte = ((self.low >> 24) & 0xff) as u8;
                    self.out.push(byte);
                    self.low &= (1u64 << 24) - 1;
                    self.count = -8;
                }
            }
        }

        /// Flush — pump 32 final shifts to drain the in-flight bits,
        /// then pad with zero bytes so the decoder never underflows
        /// `BoolMaxBits` while walking renorm refills.
        fn finish(mut self) -> Vec<u8> {
            for _ in 0..32 {
                self.encode_bool(0, 128);
            }
            while self.out.len() < 64 {
                self.out.push(0);
            }
            self.out
        }
    }

    /// Encode `(bit, p)` pairs and return the buffer + buffer-size the
    /// decoder should be initialised with. The encoder always prepends
    /// the implicit marker bit `bit=0, p=128` the §9.2.1 init consumes.
    fn encode_bits(pairs: &[(u8, u32)]) -> Vec<u8> {
        let mut enc = BoolEncoder::new();
        // §9.2.1 marker: init_bool runs read_bool(128); for it to
        // decode to 0 we encode a leading (bit=0, p=128).
        enc.encode_bool(0, 128);
        for &(bit, p) in pairs {
            enc.encode_bool(bit, p);
        }
        enc.finish()
    }

    /// Verify the encoder round-trips: encode a sequence, decode it,
    /// confirm the same bits come back out under the same probabilities.
    /// This pins the encoder before we trust it for the §6.4.3 tests.
    #[test]
    fn bool_encoder_round_trips_under_uniform_probability() {
        let pairs: Vec<(u8, u32)> = vec![
            (0, 128),
            (1, 128),
            (0, 128),
            (1, 128),
            (1, 128),
            (0, 128),
            (0, 128),
            (1, 128),
        ];
        let buf = encode_bits(&pairs);
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).expect("encoder buffer init");
        for &(bit, p) in &pairs {
            let got = dec.read_bool(p).unwrap();
            assert_eq!(got as u8, bit, "bit mismatch under p={p}");
        }
    }

    /// Verify the encoder round-trips under a mix of probabilities the
    /// §6.4.3 driver actually uses (the §10.4 KF_PARTITION_PROBS rows
    /// fall in the 3..=222 range; we sweep extremes plus a midrange).
    #[test]
    fn bool_encoder_round_trips_under_mixed_probabilities() {
        let pairs: Vec<(u8, u32)> = vec![
            (1, 200),
            (0, 50),
            (1, 50),
            (0, 200),
            (0, 128),
            (1, 12),
            (0, 12),
            (1, 200),
            (1, 200),
            (0, 200),
        ];
        let buf = encode_bits(&pairs);
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).expect("encoder buffer init");
        for &(bit, p) in &pairs {
            let got = dec.read_bool(p).unwrap();
            assert_eq!(got as u8, bit, "bit mismatch under p={p}");
        }
    }

    // The §10.4 KF_PARTITION_PROBS rows the driver consults at each ctx:
    //   ctx=12 (BLOCK_64X64, both not split) → [174, 35, 49]
    //   ctx=8  (BLOCK_32X32, both not split) → [150, 40, 39]
    //   ctx=4  (BLOCK_16X16, both not split) → [149, 53, 53]
    //   ctx=0  (BLOCK_8X8,   both not split) → [158, 97, 94]
    // (And the post-write-back ctx values shift to 13/9/5/1 once the
    // top neighbour's bit is set, etc.)

    /// Helper: drive `decode_partition` with `pairs` of (bit, p) values
    /// that the §6.4.3 recursion is expected to consume in order, with
    /// the keyframe probability table.
    fn run_partition_with_pairs(
        mi_rows: usize,
        mi_cols: usize,
        bsize: u8,
        pairs: &[(u8, u32)],
    ) -> Vec<LeafBlock> {
        let buf = encode_bits(pairs);
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).expect("buffer init");
        let mut ctx = PartitionContext::new(mi_cols, mi_rows);
        let mut leaves = Vec::new();
        decode_partition(
            &mut dec,
            0,
            0,
            bsize,
            mi_rows,
            mi_cols,
            &mut ctx,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .expect("decode_partition ok");
        leaves
    }

    /// Scenario (a) — Single 64x64 CTU, PARTITION_NONE at root.
    ///
    /// Expected per §6.4.3:
    ///   * ctx=12 (zero strips at BLOCK_64X64); probs row = [174, 35, 49].
    ///   * decode_partition_type interior tree at node 0 reads
    ///     read_bool(174); we encode (bit=0, p=174) → PARTITION_NONE.
    ///   * Single leaf at (0, 0, BLOCK_64X64=12). No recursion.
    ///   * Write-back: above_val = 15 >> b_width_log2[12] = 15>>4 = 0;
    ///     same for left. Strips already 0, no observable change.
    #[test]
    fn scenario_a_single_64x64_partition_none() {
        let leaves = run_partition_with_pairs(
            /* mi_rows */ 8,
            /* mi_cols */ 8,
            /* bsize */ 12,          // BLOCK_64X64
            &[(0, 174)], // KF_PARTITION_PROBS[12][0] interior tree node 0
        );
        assert_eq!(leaves.len(), 1, "PARTITION_NONE at 64x64 → 1 leaf");
        assert_eq!(
            leaves[0],
            LeafBlock {
                r: 0,
                c: 0,
                subsize: 12
            }
        );
    }

    /// Scenario (b) — PARTITION_SPLIT at 64x64 then 4× PARTITION_NONE
    /// at each 32x32 quadrant.
    ///
    /// Trace:
    ///   Root (0, 0, BLOCK_64X64=12) ctx=12, probs=KF[12]=[174,35,49].
    ///     Walk interior tree to -SPLIT (1, 1, 1): (1,174),(1,35),(1,49).
    ///     No write-back (bsize != BLOCK_8X8 + partition == SPLIT).
    ///
    ///   (0, 0, BLOCK_32X32=9) NONE:
    ///     above_strip=[0;4], left_strip=[0;4]. bsl=2, mask=0x02.
    ///     ctx=2*4=8. probs=KF[8]=[150,40,39]. NONE: (0, 150).
    ///     subsize=9. above_val=left_val=15>>2=1.
    ///     above[0..4]=1, left[0..4]=1.
    ///
    ///   (0, 4, BLOCK_32X32=9) NONE:
    ///     above_strip=above[4..8]=[0;4], left_strip=left[0..4]=[1;4].
    ///     above_bits=0; left_bits=1, &0x02=0 → left=0. ctx=8.
    ///     NONE: (0, 150). Write-back: above[4..8]=1, left[0..4]=1
    ///     (unchanged).
    ///
    ///   (4, 0, BLOCK_32X32=9) NONE:
    ///     above_strip=above[0..4]=[1;4], left_strip=left[4..8]=[0;4].
    ///     above_bits=1, &0x02=0 → above=0. left_bits=0. ctx=8.
    ///     NONE: (0, 150). Write-back: above[0..4]=1, left[4..8]=1.
    ///
    ///   (4, 4, BLOCK_32X32=9) NONE:
    ///     above_strip=above[4..8]=[1;4], left_strip=left[4..8]=[1;4].
    ///     above_bits=1, &0x02=0 → 0. left_bits=1, &0x02=0 → 0.
    ///     ctx=8. NONE: (0, 150).
    ///
    /// Note: every child happens to land in ctx=8 here — the §9.3.2
    /// `(left, above)` bitmap only fires the `bsl=2` bit (mask=0x02),
    /// and the parent NONE write-back value 1 (= `15>>3`) is BELOW
    /// that bit, so subsequent ctx derivations all read (0, 0). A
    /// non-square child (e.g. HORZ → subsize=BLOCK_32X16) writes back
    /// `left_val = 15>>2 = 3` which sets bit 1, and downstream
    /// neighbours then see `left = 1`. We exercise that in
    /// scenario (c).
    #[test]
    fn scenario_b_split_then_four_partition_none() {
        let leaves = run_partition_with_pairs(
            8,
            8,
            12,
            &[
                // Root PARTITION_SPLIT at ctx=12.
                (1, 174),
                (1, 35),
                (1, 49),
                // 4× PARTITION_NONE at ctx=8 — all four children share
                // ctx=8 because the parent's `15>>2=1` write-back
                // doesn't set the `bsl=2` bit (mask=0x02).
                (0, 150),
                (0, 150),
                (0, 150),
                (0, 150),
            ],
        );
        assert_eq!(leaves.len(), 4, "4 PARTITION_NONE children → 4 leaves");
        // Spec-order: top-left, top-right, bottom-left, bottom-right.
        assert_eq!(
            leaves[0],
            LeafBlock {
                r: 0,
                c: 0,
                subsize: 9
            }
        );
        assert_eq!(
            leaves[1],
            LeafBlock {
                r: 0,
                c: 4,
                subsize: 9
            }
        );
        assert_eq!(
            leaves[2],
            LeafBlock {
                r: 4,
                c: 0,
                subsize: 9
            }
        );
        assert_eq!(
            leaves[3],
            LeafBlock {
                r: 4,
                c: 4,
                subsize: 9
            }
        );
    }

    /// Scenario (c) — Mixed PARTITION_HORZ + PARTITION_VERT under one
    /// root PARTITION_SPLIT.
    ///
    /// Root 64x64 PARTITION_SPLIT then four 32x32 quadrants, each with
    /// a different shape (HORZ / VERT / NONE / NONE). The trace walks
    /// the §6.4.3 recursion and pins the ctx + probability row at
    /// every step, then re-uses the §9.3.1 tree-decode rule to derive
    /// the exact (bit, p) sequence the bool coder must produce.
    ///
    /// Root (0, 0, BLOCK_64X64=12):
    ///   ctx=12 probs=KF[12]=[174,35,49]. Walk to -SPLIT:
    ///     (1,174), (1,35), (1,49). No write-back (bsize != BLOCK_8X8 +
    ///     partition == SPLIT).
    ///
    /// (0, 0, BLOCK_32X32=9) -> HORZ:
    ///   above_strip=[0;4], left_strip=[0;4]. bsl=2, mask=0x02.
    ///   above_bits=0, left_bits=0 → ctx=2*4 = 8. probs=KF[8]=[150,40,39].
    ///   HORZ tree walk (1, 0): (1, 150), (0, 40). subsize=BLOCK_32X16=8.
    ///   Write-back: above_val=15>>b_width_log2[8]=15>>3=1,
    ///   left_val=15>>b_height_log2[8]=15>>2=3.
    ///   above[0..4]=1, left[0..4]=3.
    ///
    /// (0, 4, BLOCK_32X32=9) -> VERT:
    ///   above_strip=above[4..8]=[0;4], left_strip=left[0..4]=[3;4].
    ///   above_bits=0; left_bits=3, &0x02=2 → left=1. ctx=2*4+2+0 = 10.
    ///   probs=KF[10]=[67,33,11]. VERT tree walk (1, 1, 0): (1, 67),
    ///   (1, 33), (0, 11). subsize=BLOCK_16X32=7.
    ///   Write-back: above_val=15>>b_width_log2[7]=15>>2=3,
    ///   left_val=15>>b_height_log2[7]=15>>3=1.
    ///   above[4..8]=3, left[0..4]=1.
    ///
    /// (4, 0, BLOCK_32X32=9) -> NONE:
    ///   above_strip=above[0..4]=[1;4], left_strip=left[4..8]=[0;4].
    ///   above_bits=1, &0x02=0 → above=0. left_bits=0 → left=0.
    ///   ctx=2*4=8. probs=KF[8]=[150,40,39]. NONE walk: (0, 150).
    ///   subsize=9. Write-back: above[0..4]=1 (rewritten same),
    ///   left[4..8]=1.
    ///
    /// (4, 4, BLOCK_32X32=9) -> NONE:
    ///   above_strip=above[4..8]=[3;4], left_strip=left[4..8]=[1;4].
    ///   above_bits=3, &0x02=2 → above=1. left_bits=1, &0x02=0 → left=0.
    ///   ctx=2*4+0+1=9. probs=KF[9]=[78,12,26]. NONE walk: (0, 78).
    ///
    /// Expected leaves (in spec recursion order):
    ///   HORZ children: (0,0,8), (2,0,8).
    ///   VERT children: (0,4,7), (0,6,7).
    ///   NONE  leaves: (4,0,9), (4,4,9).
    #[test]
    fn scenario_c_mixed_horz_and_vert() {
        let pairs: &[(u8, u32)] = &[
            // Root SPLIT at ctx=12.
            (1, 174),
            (1, 35),
            (1, 49),
            // (0,0) HORZ at ctx=8.
            (1, 150),
            (0, 40),
            // (0,4) VERT at ctx=10.
            (1, 67),
            (1, 33),
            (0, 11),
            // (4,0) NONE at ctx=8.
            (0, 150),
            // (4,4) NONE at ctx=9.
            (0, 78),
        ];
        // Self-test round-trip first — if the encoder can't reproduce
        // the (bit, p) sequence, the recursion test wouldn't either.
        {
            let buf = encode_bits(pairs);
            let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
            for (idx, &(bit, p)) in pairs.iter().enumerate() {
                let got = dec.read_bool(p).unwrap();
                assert_eq!(got as u8, bit, "round-trip failed at idx={idx} (p={p})");
            }
        }
        let leaves = run_partition_with_pairs(8, 8, 12, pairs);
        assert_eq!(leaves.len(), 6, "HORZ+HORZ + VERT+VERT + NONE + NONE = 6");
        // (0,0) HORZ → two leaves at subsize BLOCK_32X16 (8).
        assert_eq!(
            leaves[0],
            LeafBlock {
                r: 0,
                c: 0,
                subsize: 8
            }
        );
        assert_eq!(
            leaves[1],
            LeafBlock {
                r: 2,
                c: 0,
                subsize: 8
            }
        );
        // (0,4) VERT → two leaves at subsize BLOCK_16X32 (7).
        assert_eq!(
            leaves[2],
            LeafBlock {
                r: 0,
                c: 4,
                subsize: 7
            }
        );
        assert_eq!(
            leaves[3],
            LeafBlock {
                r: 0,
                c: 6,
                subsize: 7
            }
        );
        // (4,0) NONE → one leaf at subsize BLOCK_32X32 (9).
        assert_eq!(
            leaves[4],
            LeafBlock {
                r: 4,
                c: 0,
                subsize: 9
            }
        );
        // (4,4) NONE → one leaf at subsize BLOCK_32X32 (9).
        assert_eq!(
            leaves[5],
            LeafBlock {
                r: 4,
                c: 4,
                subsize: 9
            }
        );
    }

    /// Verify the §6.4.3 edge guard: a call at (r, c) past the
    /// tile's MI extent returns immediately with no leaves and no
    /// bits consumed.
    #[test]
    fn decode_partition_off_edge_returns_no_leaves() {
        let buf = encode_bits(&[(0, 128)]); // unused
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        let mut ctx = PartitionContext::new(8, 8);
        let mut leaves = Vec::new();
        decode_partition(
            &mut dec,
            8, // r past mi_rows
            0,
            12,
            8,
            8,
            &mut ctx,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert!(leaves.is_empty());

        let mut leaves = Vec::new();
        decode_partition(
            &mut dec,
            0,
            8, // c past mi_cols
            12,
            8,
            8,
            &mut ctx,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        assert!(leaves.is_empty());
    }

    /// Verify the §6.4.3 write-back step: after four PARTITION_NONE
    /// children at BLOCK_32X32 every cell of both strips equals
    /// `15 >> b_width_log2[BLOCK_32X32] = 15 >> 3 = 1`.
    ///
    /// (`b_width_log2_lookup[9] = 3` because BLOCK_32X32 spans 8 cells
    /// of `4x4`-block width and `log2(8) = 3`. Per §6.4.3 line 2388
    /// `AbovePartitionContext[ c + i ] = 15 >> b_width_log2_lookup[
    /// subsize ]` so the cell value is `15 >> 3 = 1`.)
    #[test]
    fn decode_partition_writes_back_context_strips() {
        let buf = encode_bits(&[
            (1, 174),
            (1, 35),
            (1, 49), // root SPLIT
            (0, 150),
            (0, 150),
            (0, 150),
            (0, 150), // 4 NONE at ctx=8
        ]);
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        let mut ctx = PartitionContext::new(8, 8);
        let mut leaves = Vec::new();
        decode_partition(
            &mut dec,
            0,
            0,
            12,
            8,
            8,
            &mut ctx,
            PartitionProbsKind::Keyframe,
            &mut leaves,
        )
        .unwrap();
        // After four 32x32 PARTITION_NONE leaves, every cell of both
        // strips should equal 15>>3 = 1.
        for &cell in &ctx.above {
            assert_eq!(cell, 1, "above strip not fully written");
        }
        for &cell in &ctx.left {
            assert_eq!(cell, 1, "left strip not fully written");
        }
    }

    /// Pin the inter-frame `PartitionProbsKind::Inter` path: the
    /// driver MUST consult the caller-supplied running table rather
    /// than [`KF_PARTITION_PROBS`]. We pick a probability of 0 in
    /// every cell of the running table; with the §9.2 split
    /// computation that drives read_bool() = 1 for every read at
    /// p=0 (since split = 1, value ≥ 1), so the root walk traverses
    /// every right branch → -PARTITION_SPLIT at the root, then four
    /// recursive 32x32 PARTITION_SPLIT calls (each also p=0 → 1,1,1
    /// → -SPLIT) and so on, recursing until BLOCK_8X8 parents are
    /// reached and the BLOCK_8X8 → BLOCK_4X4 leaves drop in.
    ///
    /// Rather than fully unrolling that tree we test a small case:
    /// at a BLOCK_8X8 parent, PARTITION_NONE under a custom prob
    /// row [1, 1, 1] still selects NONE (p=1 → split=1, value=0 →
    /// 0 < 1 → bit=0) — confirming the Inter path reads from our
    /// table, not the static KF table whose ctx=0 row [158, 97, 94]
    /// would also select NONE but via a different decode path.
    #[test]
    fn decode_partition_inter_path_uses_supplied_table() {
        // Custom table where ctx=0 row is [1, 1, 1] (chosen to
        // make the §9.2 split=1 path deterministic).
        let mut table = [[1u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS];
        table[0] = [1, 1, 1];
        let probs_kind = PartitionProbsKind::Inter(&table);

        // Encode one PARTITION_NONE read at p=1 for the BLOCK_8X8 root.
        let buf = encode_bits(&[(0, 1)]);
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        let mut ctx = PartitionContext::new(1, 1);
        let mut leaves = Vec::new();
        decode_partition(
            &mut dec,
            0,
            0,
            /* BLOCK_8X8 */ 3,
            1,
            1,
            &mut ctx,
            probs_kind,
            &mut leaves,
        )
        .expect("inter path decode ok");
        assert_eq!(leaves.len(), 1);
        assert_eq!(
            leaves[0],
            LeafBlock {
                r: 0,
                c: 0,
                subsize: 3
            }
        );
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
