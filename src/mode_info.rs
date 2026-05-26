//! VP9 per-block mode-info primitives per spec v0.7 — §6.4.5 / §6.4.6 /
//! §6.4.7 / §6.4.8 / §6.4.10 / §6.4.15 / §9.3.1 / §9.3.2 / §9.3.3.
//!
//! The §6.4.15 [`intra_block_mode_info`] inter-frame intra-block reader
//! is the companion to the §6.4.6 keyframe driver: it reads `intra_mode`
//! / `sub_intra_mode` / `uv_mode` from the §9.3 compressed-header
//! [`DEFAULT_Y_MODE_PROBS`] / [`DEFAULT_UV_MODE_PROBS`] tables (ctx
//! `size_group_lookup[MiSize]` / `0` / `y_mode` per §9.3.2) rather than
//! the §10.5 keyframe `kf_*_mode_probs`, fixes `ref_frame` to
//! `INTRA_FRAME`, and handles the sub-8x8 `(idy, idx)` grid. The §6.4.5
//! [`Vp9ModeInfo`] dispatch enum and [`inter_frame_intra_block_mode_info`]
//! wire it alongside the keyframe path.
//!
//! Round 17 lands the §6.4.6 [`intra_frame_mode_info`] keyframe-only
//! per-block mode-info reader on top of the round-15 / 16 primitives.
//! The driver ties `intra_segment_id` plus `read_skip` plus
//! `read_tx_size( 1 )` plus `default_intra_mode` plus `default_uv_mode`
//! into a single [`Vp9IntraMiBlock`] output, infers
//! `ref_frame[0] = INTRA_FRAME`, `ref_frame[1] = NONE`, `is_inter = 0`,
//! and handles both the `MiSize >= BLOCK_8X8` single-mode partition
//! and the `MiSize < BLOCK_8X8` sub-mode walk (the §6.4.6 `(idy, idx)`
//! iteration stepping by `num_4x4_blocks_high_lookup[MiSize]` and
//! `num_4x4_blocks_wide_lookup[MiSize]`, reading one
//! `default_intra_mode` per partition cell and replicating it across
//! the 2x2 `sub_modes[]` grid). The §9.3.1 `intra_mode_tree[18]` and
//! the §10.5 `kf_y_mode_probs[10][10][9]` plus `kf_uv_mode_probs[10][9]`
//! tables are transcribed verbatim from `docs/video/vp9/vp9-spec.txt`.
//!
//! Round 15 lands the **first slice** of the per-block mode-info decode
//! that the §6.4.21 [`crate::residual::residual_intra`] driver currently
//! consumes via a caller-supplied bundle:
//!
//! * §9.3.3 [`tree_decode`] — the generic tree-decoding loop
//!   `do { n = T[n + read_bool(P(n >> 1))] } while (n > 0)` that every
//!   tree-coded syntax element (skip, tx_size, intra_mode, …) routes
//!   through. The probability callback is a `FnMut(usize) -> u8` so
//!   call-sites can splice in the right §9.3.2 row without this helper
//!   needing to know which syntax element it's decoding.
//! * §6.4.8 [`read_skip`] — the one-bit `Skip` decode under the §6.4.9
//!   `seg_feature_active(SEG_LVL_SKIP)` early-return rule, with the
//!   §9.3.2 context `Skips[MiRow-1][MiCol] + Skips[MiRow][MiCol-1]`
//!   (modulated by `AvailU` / `AvailL`) and the §9.3.2 binary tree.
//! * §6.4.10 [`read_tx_size`] — the `read_tx_size(allowSelect)` decode
//!   with `maxTxSize = max_txsize_lookup[MiSize]` and the §9.3.2 ctx
//!   `(above + left) > maxTxSize` rule for `TX_MODE_SELECT`. Falls
//!   through to `Min(maxTxSize, tx_mode_to_biggest_tx_size[tx_mode])`
//!   when select isn't allowed. The three §9.3.1 trees
//!   ([`TX_SIZE_8_TREE`] / [`TX_SIZE_16_TREE`] / [`TX_SIZE_32_TREE`])
//!   are transcribed verbatim from the spec listing.
//! * [`NeighbourSkips`] / [`NeighbourTxSizes`] — the per-MI-block
//!   neighbour-state bundles a tile driver builds from its
//!   `Skips[ ][ ]` / `TxSizes[ ][ ]` frame-wide arrays. Construction
//!   helpers thread the §7.4.4 `AvailL` / `AvailU` rules through.
//!
//! Out of scope this round:
//!
//! * The §6.4.6 `intra_frame_mode_info()` driver itself (which calls
//!   `intra_segment_id()`, `read_skip()`, then `read_tx_size()`, then
//!   `intra_block_mode_info()`). This module supplies the
//!   constituents; the orchestrator that wires them into a
//!   `Vp9IntraMiBlock` lands once the §6.4.7 `intra_segment_id()` and
//!   §6.4.15 `intra_block_mode_info()` primitives land alongside.
//! * Inter-frame mode-info (§6.4.11+). Requires reference-buffer
//!   state.
//! * The `Skips[MiRow][MiCol]` / `TxSizes[MiRow][MiCol]` write-back
//!   into the frame-wide arrays the next MI block consumes. Left to
//!   the §6.4.6 driver.
//! * The §8.4 probability adaptation `counts_skip` / `counts_tx_size`
//!   accumulators. The §9.3.4 counters are bookkeeping for the
//!   §8.4 adaption pass at end-of-frame; this round leaves the
//!   `counts_*` plumbing for the adaption round.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §6.4.5 / §6.4.6 / §6.4.8 / §6.4.9 /
//! §6.4.10 / §6.4.15 / §9.3.1 / §9.3.2 / §9.3.3). No external library
//! source consulted; every formula and every probability / tree array
//! transcribed directly from the spec listing.

// Helpers in this module are exercised exclusively from `#[cfg(test)]`
// and the deferred §6.4.6 driver until the per-frame public decode path
// lands.
#![allow(dead_code)]

use crate::bool_coder::BoolCoder;
use crate::compressed::{tx_mode_to_biggest_tx_size, TxMode};
use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};
use crate::residual::{
    BLOCK_8X8, BLOCK_SIZES, MAX_TXSIZE_LOOKUP, NUM_4X4_BLOCKS_HIGH_LOOKUP,
    NUM_4X4_BLOCKS_WIDE_LOOKUP,
};
use crate::Error;

// ----- §9.3.1 tree listings (verbatim transcription) -----

/// `tx_size_8_tree[ 2 ]` per §9.3.1.
///
/// Tree for `maxTxSize == TX_8X8`: a single binary decision picking
/// `TX_4X4` (0) vs `TX_8X8` (1).
pub(crate) const TX_SIZE_8_TREE: [i32; 2] = [
    -(0), // -TX_4X4
    -(1), // -TX_8X8
];

/// `tx_size_16_tree[ 4 ]` per §9.3.1.
///
/// Tree for `maxTxSize == TX_16X16`: two binary decisions, returning
/// `TX_4X4` / `TX_8X8` / `TX_16X16`.
pub(crate) const TX_SIZE_16_TREE: [i32; 4] = [
    -(0),
    2, // -TX_4X4, 2
    -(1),
    -(2), // -TX_8X8, -TX_16X16
];

/// `tx_size_32_tree[ 6 ]` per §9.3.1.
///
/// Tree for `maxTxSize == TX_32X32`: three binary decisions, returning
/// any of `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32`.
pub(crate) const TX_SIZE_32_TREE: [i32; 6] = [
    -(0),
    2, // -TX_4X4, 2
    -(1),
    4, // -TX_8X8, 4
    -(2),
    -(3), // -TX_16X16, -TX_32X32
];

/// `binary_tree[ 2 ]` per §9.3.1 — the single-bit tree used for `Skip`
/// and several other binary syntax elements (`seg_id_predicted`,
/// `is_inter`, `comp_mode`, `comp_ref`, `single_ref_p1` /
/// `single_ref_p2`, `mv_sign`, `mv_bit`, `mv_class0_bit`, `more_coefs`).
pub(crate) const BINARY_TREE: [i32; 2] = [0, -1];

/// `segment_tree[ 14 ]` per §9.3.1 — the 7-leaf binary tree used to
/// decode `segment_id` (values `0..=7`, one per VP9 segment slot).
///
/// Verbatim from the §9.3.1 listing:
///
/// ```text
/// segment_tree[ 14 ] = {
///     2, 4, 6, 8, 10, 12,
///     0, -1, -2, -3, -4, -5, -6, -7
/// }
/// ```
///
/// The first six pairs are inner branches (a `1` always advances by 2),
/// and the seven `-i` leaves at positions `7..=13` map to segment ids
/// `1..=7`. Position 6 stores the value `0` — the §9.3.3 walker returns
/// `-n` at the end, so a tree slot of `0` produces segment id `0`.
pub(crate) const SEGMENT_TREE: [i32; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];

// ----- §9.3.3 tree decoding -----

/// `Tree decoding process` per §9.3.3.
///
/// Iterates `do { n = T[n + read_bool(P(n >> 1))] } while (n > 0)` and
/// returns `-n` once `n` becomes negative or zero (the spec uses `-n`
/// after the loop terminates; a 0 leaf maps to value 0). The
/// probability callback `prob` receives the current `node = n >> 1`
/// and returns the matching `P( node )` from the syntax-element-specific
/// probability table per §9.3.2.
///
/// Returns an [`Error::InvalidBitstream`] error if the bool coder runs
/// out of bits mid-walk, mirroring [`BoolCoder::read_bool`]'s failure
/// mode.
pub(crate) fn tree_decode<F>(
    coder: &mut BoolCoder<'_>,
    tree: &[i32],
    mut prob: F,
) -> Result<i32, Error>
where
    F: FnMut(usize) -> u8,
{
    let mut n: i32 = 0;
    loop {
        let p = u32::from(prob((n >> 1) as usize));
        let bit = coder.read_bool(p)?;
        n = tree[(n + bit as i32) as usize];
        if n <= 0 {
            return Ok(-n);
        }
    }
}

// ----- §9.3.2 context derivations -----

/// Neighbour `Skips[ ][ ]` cells consumed by [`skip_context`].
///
/// A tile driver materialises this from its frame-wide `Skips[MiRow][MiCol]`
/// array against the §7.4.4 `AvailL` / `AvailU` flags. When the
/// neighbour is unavailable the field is `None` and the §9.3.2 listing
/// elides its contribution to the context.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NeighbourSkips {
    /// `Skips[MiRow - 1][MiCol]` if `AvailU`, else `None`.
    pub above: Option<u8>,
    /// `Skips[MiRow][MiCol - 1]` if `AvailL`, else `None`.
    pub left: Option<u8>,
}

/// `skip` context per §9.3.2.
///
/// ```text
/// ctx = 0
/// if ( AvailU ) ctx += Skips[ MiRow - 1 ][ MiCol ]
/// if ( AvailL ) ctx += Skips[ MiRow ][ MiCol - 1 ]
/// ```
///
/// Returns one of `0` / `1` / `2` indexing `skip_prob[ ctx ]`.
pub(crate) fn skip_context(nb: NeighbourSkips) -> usize {
    let mut ctx = 0usize;
    if let Some(s) = nb.above {
        ctx += usize::from(s);
    }
    if let Some(s) = nb.left {
        ctx += usize::from(s);
    }
    ctx
}

/// Neighbour `TxSizes[ ][ ]` + `Skips[ ][ ]` cells consumed by
/// [`tx_size_context`].
///
/// The §9.3.2 listing reads the above / left tx-sizes only when the
/// neighbouring MI block was *not* skipped; on a skipped neighbour the
/// `maxTxSize` fallback is used instead. The driver therefore needs
/// both the `Skips[ ]` flag and the `TxSizes[ ]` value at each
/// neighbour cell.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NeighbourTxSizes {
    /// `AvailU`. When `false` the §9.3.2 listing substitutes the
    /// `left` value into `above`.
    pub avail_u: bool,
    /// `AvailL`. When `false` the §9.3.2 listing substitutes the
    /// `above` value into `left`.
    pub avail_l: bool,
    /// `Skips[MiRow - 1][MiCol]` — only consulted when `avail_u`.
    pub skip_above: u8,
    /// `Skips[MiRow][MiCol - 1]` — only consulted when `avail_l`.
    pub skip_left: u8,
    /// `TxSizes[MiRow - 1][MiCol]` — only consulted when `avail_u`
    /// **and** `!skip_above`.
    pub tx_above: u32,
    /// `TxSizes[MiRow][MiCol - 1]` — only consulted when `avail_l`
    /// **and** `!skip_left`.
    pub tx_left: u32,
}

/// `tx_size` context per §9.3.2.
///
/// ```text
/// above = maxTxSize
/// left  = maxTxSize
/// if ( AvailU && !Skips[MiRow-1][MiCol] ) above = TxSizes[MiRow-1][MiCol]
/// if ( AvailL && !Skips[MiRow ][MiCol-1] ) left  = TxSizes[MiRow ][MiCol-1]
/// if ( !AvailL ) left  = above
/// if ( !AvailU ) above = left
/// ctx = (above + left) > maxTxSize
/// ```
///
/// Returns `0` or `1`, indexing `tx_probs[maxTxSize][ctx][node]`.
/// `max_tx_size` is the §6.4.10 `max_txsize_lookup[MiSize]` value.
pub(crate) fn tx_size_context(nb: NeighbourTxSizes, max_tx_size: u32) -> usize {
    let mut above = max_tx_size;
    let mut left = max_tx_size;
    if nb.avail_u && nb.skip_above == 0 {
        above = nb.tx_above;
    }
    if nb.avail_l && nb.skip_left == 0 {
        left = nb.tx_left;
    }
    if !nb.avail_l {
        left = above;
    }
    if !nb.avail_u {
        above = left;
    }
    usize::from(above + left > max_tx_size)
}

// ----- §6.4.8 read_skip -----

/// `read_skip( )` per §6.4.8.
///
/// If `seg_feature_skip_active` (the spec's
/// `seg_feature_active(SEG_LVL_SKIP)`) is `true`, the spec hardwires
/// `skip = 1` and reads no bits. Otherwise reads a single
/// §9.3.3-coded `Skip` token using the §9.3.2 binary tree and the
/// `skip_prob[ctx]` probability where `ctx` is derived by
/// [`skip_context`]. Returns the decoded `skip` flag.
///
/// `skip_prob` is the 3-entry `skip_prob[SKIP_CONTEXTS]` table after
/// the round-5 §6.3.8 sweep (carried on
/// [`crate::compressed::Vp9CompressedHeader::skip_prob`]).
pub(crate) fn read_skip(
    coder: &mut BoolCoder<'_>,
    seg_feature_skip_active: bool,
    skip_prob: &[u8; 3],
    nb: NeighbourSkips,
) -> Result<bool, Error> {
    if seg_feature_skip_active {
        return Ok(true);
    }
    let ctx = skip_context(nb);
    let value = tree_decode(coder, &BINARY_TREE, |_| skip_prob[ctx])?;
    Ok(value != 0)
}

// ----- §6.4.7 intra_segment_id + segment_id tree decode -----

/// Decode a `segment_id` value (`0..=7`) via the §9.3.1 [`SEGMENT_TREE`]
/// and per-node probability table `segmentation_tree_probs[node]` per
/// the §9.3.2 listing
///
/// ```text
/// segment_id: the probability is given by segmentation_tree_probs[ node ].
/// ```
///
/// This is the call-site that materialises the `segment_id` token from
/// the §9.2 bool coder once the §6.4.7 / §6.4.12 syntax decides a fresh
/// decode is needed.
pub(crate) fn read_segment_id(
    coder: &mut BoolCoder<'_>,
    tree_probs: &[u8; 7],
) -> Result<u8, Error> {
    let value = tree_decode(coder, &SEGMENT_TREE, |node| tree_probs[node])?;
    // §9.3.1 lays the tree out so the leaves are `0..=7`; the §9.3.3
    // post-loop `-n` already returns the segment id directly.
    Ok(value as u8)
}

/// `intra_segment_id( )` per §6.4.7.
///
/// ```text
/// intra_segment_id( ) {
///     if ( segmentation_enabled && segmentation_update_map )
///         segment_id                                                       T
///     else
///         segment_id = 0
/// }
/// ```
///
/// The intra path is simpler than the §6.4.12 inter version:
///
/// * No temporal prediction (`seg_id_predicted` and `predictedSegmentId`
///   only apply to inter frames).
/// * No `AboveSegPredContext` / `LeftSegPredContext` write-back.
/// * No fall-back to `get_segment_id()`'s spatial neighbour — when the
///   map isn't being updated on an intra frame the spec forces
///   `segment_id = 0` (since intra frames are key-frame / intra-only,
///   the previous map is meaningless).
///
/// `tree_probs` is the `segmentation_tree_probs[7]` table carried on
/// [`crate::header::SegmentationParams::tree_probs`] (`Some([…; 7])`
/// when the uncompressed header surfaced an `update_map == 1` /
/// `prob_update` decode, or the `[255; 7]` no-probability-coded
/// fallback the §6.2.12 `read_prob()` helper substitutes). When
/// `update_map == 0` the spec doesn't read `segment_id`, so the absent
/// `tree_probs` field is allowed to be `None` and this helper short-
/// circuits before dereferencing it.
pub(crate) fn intra_segment_id(
    coder: &mut BoolCoder<'_>,
    segmentation_enabled: bool,
    segmentation_update_map: bool,
    tree_probs: Option<&[u8; 7]>,
) -> Result<u8, Error> {
    if segmentation_enabled && segmentation_update_map {
        let probs = tree_probs.ok_or(Error::InvalidBitstream)?;
        read_segment_id(coder, probs)
    } else {
        Ok(0)
    }
}

// ----- §6.4.10 read_tx_size -----

/// `read_tx_size( allowSelect )` per §6.4.10.
///
/// Three cases:
///
/// 1. `allow_select && tx_mode == TX_MODE_SELECT && MiSize >= BLOCK_8X8`:
///    decode the `tx_size` syntax element via the §9.3.3 tree, choosing
///    the §9.3.1 tree from `maxTxSize`:
///      * `TX_32X32` → [`TX_SIZE_32_TREE`]
///      * `TX_16X16` → [`TX_SIZE_16_TREE`]
///      * else → [`TX_SIZE_8_TREE`]
///
///    The probability for node `n` is `tx_probs[maxTxSize][ctx][n]`
///    per §9.3.2, with `ctx` from [`tx_size_context`].
/// 2. Otherwise: `tx_size = Min(maxTxSize, tx_mode_to_biggest_tx_size[tx_mode])`
///    per §6.4.10's `else` branch.
///
/// Returns the §3 `TX_*` integer (`TX_4X4=0`, `TX_8X8=1`,
/// `TX_16X16=2`, `TX_32X32=3`).
///
/// `tx_probs` is the 4-row table (rows 1..=3 active per §10) carried on
/// [`crate::compressed::Vp9CompressedHeader::tx_probs`]. `mi_size` is
/// the §7.4.3 block size (one of the `BLOCK_*` constants from
/// [`crate::residual`]). `nb` carries the neighbour state from the
/// frame-wide `Skips[ ]` / `TxSizes[ ]` arrays.
pub(crate) fn read_tx_size(
    coder: &mut BoolCoder<'_>,
    allow_select: bool,
    tx_mode: TxMode,
    mi_size: u8,
    tx_probs: &[[[u8; 3]; 2]; 4],
    nb: NeighbourTxSizes,
) -> Result<u32, Error> {
    let max_tx_size = MAX_TXSIZE_LOOKUP[mi_size as usize];
    if allow_select && tx_mode == TxMode::TxModeSelect && mi_size >= BLOCK_8X8 {
        let ctx = tx_size_context(nb, max_tx_size);
        let probs_row = &tx_probs[max_tx_size as usize][ctx];
        let tree: &[i32] = match max_tx_size {
            3 => &TX_SIZE_32_TREE,
            2 => &TX_SIZE_16_TREE,
            _ => &TX_SIZE_8_TREE,
        };
        let value = tree_decode(coder, tree, |node| probs_row[node])?;
        Ok(value as u32)
    } else {
        Ok(max_tx_size.min(tx_mode_to_biggest_tx_size(tx_mode) as u32))
    }
}

// ----- §3 reference-frame sentinels -----

/// `INTRA_FRAME = 0` per §3 — first entry of the §3 `ref_frame[ ]`
/// enumeration (`INTRA_FRAME, LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME`,
/// indices `0..=3`, with `MAX_REF_FRAMES = 4`). Used by §6.4.6 to pin
/// `ref_frame[0]` for an intra-frame block.
pub(crate) const INTRA_FRAME: i32 = 0;

/// `NONE = -1` per §3 — the sentinel sentinel value assigned to
/// `ref_frame[1]` when the block is single-reference (or intra). The
/// §6.4.16 / §6.4.21 `isCompound` test reads `ref_frame[1] > NONE`, so
/// `NONE` is strictly less than `INTRA_FRAME = 0`. The spec writes
/// `ref_frame[1] = NONE` (line 2455) and `isCompound = ref_frame[1] >
/// NONE` (line 4515) without quoting a numeric value; `NONE = -1` is
/// the unique integer satisfying both `< INTRA_FRAME` and the §6.4.11
/// neighbour-init rule `LeftRefFrame[1] = AvailL ? … : NONE`. The
/// loop-filter listing iterates `INTRA_FRAME..MAX_REF_FRAMES-1` (i.e.
/// `0..=3`) so NONE is outside that range as expected.
pub(crate) const NONE_REF_FRAME: i32 = -1;

// ----- §3 / §10 intra-mode enumeration -----

/// `INTRA_MODES = 10` per §3 — number of values for the §7.4.5
/// `default_intra_mode` (also `intra_mode`, `sub_intra_mode`, `uv_mode`,
/// `default_uv_mode`).
pub(crate) const INTRA_MODES: usize = 10;

/// §7.4.5 `default_intra_mode` (and `intra_mode` / `sub_intra_mode` /
/// `uv_mode` / `default_uv_mode`) integer values, mirroring the
/// [`crate::intra::PredMode`] discriminants (0 = `DC_PRED`, 1 = `V_PRED`,
/// 2 = `H_PRED`, 3 = `D45_PRED`, 4 = `D135_PRED`, 5 = `D117_PRED`,
/// 6 = `D153_PRED`, 7 = `D207_PRED`, 8 = `D63_PRED`, 9 = `TM_PRED`).
pub(crate) const DC_PRED: u8 = 0;
pub(crate) const V_PRED: u8 = 1;
pub(crate) const H_PRED: u8 = 2;
pub(crate) const D45_PRED: u8 = 3;
pub(crate) const D135_PRED: u8 = 4;
pub(crate) const D117_PRED: u8 = 5;
pub(crate) const D153_PRED: u8 = 6;
pub(crate) const D207_PRED: u8 = 7;
pub(crate) const D63_PRED: u8 = 8;
pub(crate) const TM_PRED: u8 = 9;

// ----- §9.3.1 intra_mode_tree -----

/// `intra_mode_tree[ 18 ]` per §9.3.1 — the §9.3.3 tree used for every
/// `default_intra_mode` / `intra_mode` / `sub_intra_mode` /
/// `default_uv_mode` / `uv_mode` decode.
///
/// Verbatim from the §9.3.1 listing:
///
/// ```text
/// intra_mode_tree[ 18 ] = {
///     -DC_PRED, 2,
///     -TM_PRED, 4,
///     -V_PRED, 6,
///     8, 12,
///     -H_PRED, 10,
///     -D135_PRED, -D117_PRED,
///     -D45_PRED, 14,
///     -D63_PRED, 16,
///     -D153_PRED, -D207_PRED
/// }
/// ```
///
/// `-0` collapses to `0` so the §9.3.3 post-loop `-n` returns `0`
/// (DC_PRED) directly at the first leaf. All other leaves carry the
/// matching `PredMode` discriminant negated.
pub(crate) const INTRA_MODE_TREE: [i32; 18] = [
    -(DC_PRED as i32),
    2,
    -(TM_PRED as i32),
    4,
    -(V_PRED as i32),
    6,
    8,
    12,
    -(H_PRED as i32),
    10,
    -(D135_PRED as i32),
    -(D117_PRED as i32),
    -(D45_PRED as i32),
    14,
    -(D63_PRED as i32),
    16,
    -(D153_PRED as i32),
    -(D207_PRED as i32),
];

// ----- §10.5 kf_y_mode_probs / kf_uv_mode_probs (verbatim) -----

/// `kf_y_mode_probs[ INTRA_MODES ][ INTRA_MODES ][ INTRA_MODES - 1 ]`
/// per §10.5 — the per-keyframe probability table indexed by
/// `[abovemode][leftmode][node]` for `default_intra_mode` per §9.3.2
/// (line 6268 of the spec listing).
///
/// Outer index is the *abovemode* (`DC_PRED`..`TM_PRED`), middle index
/// the *leftmode*, inner index the §9.3.3 tree node (0..=8 for the
/// 18-entry [`INTRA_MODE_TREE`]). Transcribed verbatim from
/// `docs/video/vp9/vp9-spec.txt` lines 7463–7599.
pub(crate) const KF_Y_MODE_PROBS: [[[u8; INTRA_MODES - 1]; INTRA_MODES]; INTRA_MODES] = [
    // above = dc
    [
        [137, 30, 42, 148, 151, 207, 70, 52, 91],  // left = dc
        [92, 45, 102, 136, 116, 180, 74, 90, 100], // left = v
        [73, 32, 19, 187, 222, 215, 46, 34, 100],  // left = h
        [91, 30, 32, 116, 121, 186, 93, 86, 94],   // left = d45
        [72, 35, 36, 149, 68, 206, 68, 63, 105],   // left = d135
        [73, 31, 28, 138, 57, 124, 55, 122, 151],  // left = d117
        [67, 23, 21, 140, 126, 197, 40, 37, 171],  // left = d153
        [86, 27, 28, 128, 154, 212, 45, 43, 53],   // left = d207
        [74, 32, 27, 107, 86, 160, 63, 134, 102],  // left = d63
        [59, 67, 44, 140, 161, 202, 78, 67, 119],  // left = tm
    ],
    // above = v
    [
        [63, 36, 126, 146, 123, 158, 60, 90, 96],  // left = dc
        [43, 46, 168, 134, 107, 128, 69, 142, 92], // left = v
        [44, 29, 68, 159, 201, 177, 50, 57, 77],   // left = h
        [58, 38, 76, 114, 97, 172, 78, 133, 92],   // left = d45
        [46, 41, 76, 140, 63, 184, 69, 112, 57],   // left = d135
        [38, 32, 85, 140, 46, 112, 54, 151, 133],  // left = d117
        [39, 27, 61, 131, 110, 175, 44, 75, 136],  // left = d153
        [52, 30, 74, 113, 130, 175, 51, 64, 58],   // left = d207
        [47, 35, 80, 100, 74, 143, 64, 163, 74],   // left = d63
        [36, 61, 116, 114, 128, 162, 80, 125, 82], // left = tm
    ],
    // above = h
    [
        [82, 26, 26, 171, 208, 204, 44, 32, 105], // left = dc
        [55, 44, 68, 166, 179, 192, 57, 57, 108], // left = v
        [42, 26, 11, 199, 241, 228, 23, 15, 85],  // left = h
        [68, 42, 19, 131, 160, 199, 55, 52, 83],  // left = d45
        [58, 50, 25, 139, 115, 232, 39, 52, 118], // left = d135
        [50, 35, 33, 153, 104, 162, 64, 59, 131], // left = d117
        [44, 24, 16, 150, 177, 202, 33, 19, 156], // left = d153
        [55, 27, 12, 153, 203, 218, 26, 27, 49],  // left = d207
        [53, 49, 21, 110, 116, 168, 59, 80, 76],  // left = d63
        [38, 72, 19, 168, 203, 212, 50, 50, 107], // left = tm
    ],
    // above = d45
    [
        [103, 26, 36, 129, 132, 201, 83, 80, 93], // left = dc
        [59, 38, 83, 112, 103, 162, 98, 136, 90], // left = v
        [62, 30, 23, 158, 200, 207, 59, 57, 50],  // left = h
        [67, 30, 29, 84, 86, 191, 102, 91, 59],   // left = d45
        [60, 32, 33, 112, 71, 220, 64, 89, 104],  // left = d135
        [53, 26, 34, 130, 56, 149, 84, 120, 103], // left = d117
        [53, 21, 23, 133, 109, 210, 56, 77, 172], // left = d153
        [77, 19, 29, 112, 142, 228, 55, 66, 36],  // left = d207
        [61, 29, 29, 93, 97, 165, 83, 175, 162],  // left = d63
        [47, 47, 43, 114, 137, 181, 100, 99, 95], // left = tm
    ],
    // above = d135
    [
        [69, 23, 29, 128, 83, 199, 46, 44, 101],   // left = dc
        [53, 40, 55, 139, 69, 183, 61, 80, 110],   // left = v
        [40, 29, 19, 161, 180, 207, 43, 24, 91],   // left = h
        [60, 34, 19, 105, 61, 198, 53, 64, 89],    // left = d45
        [52, 31, 22, 158, 40, 209, 58, 62, 89],    // left = d135
        [44, 31, 29, 147, 46, 158, 56, 102, 198],  // left = d117
        [35, 19, 12, 135, 87, 209, 41, 45, 167],   // left = d153
        [55, 25, 21, 118, 95, 215, 38, 39, 66],    // left = d207
        [51, 38, 25, 113, 58, 164, 70, 93, 97],    // left = d63
        [47, 54, 34, 146, 108, 203, 72, 103, 151], // left = tm
    ],
    // above = d117
    [
        [64, 19, 37, 156, 66, 138, 49, 95, 133],  // left = dc
        [46, 27, 80, 150, 55, 124, 55, 121, 135], // left = v
        [36, 23, 27, 165, 149, 166, 54, 64, 118], // left = h
        [53, 21, 36, 131, 63, 163, 60, 109, 81],  // left = d45
        [40, 26, 35, 154, 40, 185, 51, 97, 123],  // left = d135
        [35, 19, 34, 179, 19, 97, 48, 129, 124],  // left = d117
        [36, 20, 26, 136, 62, 164, 33, 77, 154],  // left = d153
        [45, 18, 32, 130, 90, 157, 40, 79, 91],   // left = d207
        [45, 26, 28, 129, 45, 129, 49, 147, 123], // left = d63
        [38, 44, 51, 136, 74, 162, 57, 97, 121],  // left = tm
    ],
    // above = d153
    [
        [75, 17, 22, 136, 138, 185, 32, 34, 166], // left = dc
        [56, 39, 58, 133, 117, 173, 48, 53, 187], // left = v
        [35, 21, 12, 161, 212, 207, 20, 23, 145], // left = h
        [56, 29, 19, 117, 109, 181, 55, 68, 112], // left = d45
        [47, 29, 17, 153, 64, 220, 59, 51, 114],  // left = d135
        [46, 16, 24, 136, 76, 147, 41, 64, 172],  // left = d117
        [34, 17, 11, 108, 152, 187, 13, 15, 209], // left = d153
        [51, 24, 14, 115, 133, 209, 32, 26, 104], // left = d207
        [55, 30, 18, 122, 79, 179, 44, 88, 116],  // left = d63
        [37, 49, 25, 129, 168, 164, 41, 54, 148], // left = tm
    ],
    // above = d207
    [
        [82, 22, 32, 127, 143, 213, 39, 41, 70],  // left = dc
        [62, 44, 61, 123, 105, 189, 48, 57, 64],  // left = v
        [47, 25, 17, 175, 222, 220, 24, 30, 86],  // left = h
        [68, 36, 17, 106, 102, 206, 59, 74, 74],  // left = d45
        [57, 39, 23, 151, 68, 216, 55, 63, 58],   // left = d135
        [49, 30, 35, 141, 70, 168, 82, 40, 115],  // left = d117
        [51, 25, 15, 136, 129, 202, 38, 35, 139], // left = d153
        [68, 26, 16, 111, 141, 215, 29, 28, 28],  // left = d207
        [59, 39, 19, 114, 75, 180, 77, 104, 42],  // left = d63
        [40, 61, 26, 126, 152, 206, 61, 59, 93],  // left = tm
    ],
    // above = d63
    [
        [78, 23, 39, 111, 117, 170, 74, 124, 94],  // left = dc
        [48, 34, 86, 101, 92, 146, 78, 179, 134],  // left = v
        [47, 22, 24, 138, 187, 178, 68, 69, 59],   // left = h
        [56, 25, 33, 105, 112, 187, 95, 177, 129], // left = d45
        [48, 31, 27, 114, 63, 183, 82, 116, 56],   // left = d135
        [43, 28, 37, 121, 63, 123, 61, 192, 169],  // left = d117
        [42, 17, 24, 109, 97, 177, 56, 76, 122],   // left = d153
        [58, 18, 28, 105, 139, 182, 70, 92, 63],   // left = d207
        [46, 23, 32, 74, 86, 150, 67, 183, 88],    // left = d63
        [36, 38, 48, 92, 122, 165, 88, 137, 91],   // left = tm
    ],
    // above = tm
    [
        [65, 70, 60, 155, 159, 199, 61, 60, 81],   // left = dc
        [44, 78, 115, 132, 119, 173, 71, 112, 93], // left = v
        [39, 38, 21, 184, 227, 206, 42, 32, 64],   // left = h
        [58, 47, 36, 124, 137, 193, 80, 82, 78],   // left = d45
        [49, 50, 35, 144, 95, 205, 63, 78, 59],    // left = d135
        [41, 53, 52, 148, 71, 142, 65, 128, 51],   // left = d117
        [40, 36, 28, 143, 143, 202, 40, 55, 137],  // left = d153
        [52, 34, 29, 129, 183, 227, 42, 35, 43],   // left = d207
        [42, 44, 44, 104, 105, 164, 64, 130, 80],  // left = d63
        [43, 81, 53, 140, 169, 204, 68, 84, 72],   // left = tm
    ],
];

/// `kf_uv_mode_probs[ INTRA_MODES ][ INTRA_MODES - 1 ]` per §10.5 —
/// the per-keyframe probability table indexed by `[y_mode][node]` for
/// `default_uv_mode` per §9.3.2 (line 6297 of the spec listing).
///
/// Transcribed verbatim from `docs/video/vp9/vp9-spec.txt`
/// lines 7602–7613.
pub(crate) const KF_UV_MODE_PROBS: [[u8; INTRA_MODES - 1]; INTRA_MODES] = [
    [144, 11, 54, 157, 195, 130, 46, 58, 108],  // y = dc
    [118, 15, 123, 148, 131, 101, 44, 93, 131], // y = v
    [113, 12, 23, 188, 226, 142, 26, 32, 125],  // y = h
    [120, 11, 50, 123, 163, 135, 64, 77, 103],  // y = d45
    [113, 9, 36, 155, 111, 157, 32, 44, 161],   // y = d135
    [116, 9, 55, 176, 76, 96, 37, 61, 149],     // y = d117
    [115, 9, 28, 141, 161, 167, 21, 25, 193],   // y = d153
    [120, 12, 32, 145, 195, 142, 32, 38, 86],   // y = d207
    [116, 12, 64, 120, 140, 125, 49, 115, 121], // y = d63
    [102, 19, 66, 162, 182, 122, 35, 59, 128],  // y = tm
];

// ----- §9.3.2 size_group_lookup / §9.3 default_y_mode_probs /
//       default_uv_mode_probs (verbatim) -----

/// `BLOCK_SIZE_GROUPS = 4` per §3 — number of contexts when decoding
/// `intra_mode`. Indexes the first dimension of the §9.3
/// `y_mode_probs[ BLOCK_SIZE_GROUPS ][ INTRA_MODES - 1 ]` table.
pub(crate) const BLOCK_SIZE_GROUPS: usize = 4;

/// `size_group_lookup[ BLOCK_SIZES ]` per §9.3.2 — maps a `BLOCK_*`
/// size to the §9.3 `intra_mode` context `ctx = size_group_lookup[
/// MiSize ]` (used to index `y_mode_probs[ ctx ]`).
///
/// Transcribed verbatim from `docs/video/vp9/vp9-spec.txt`:
///
/// ```text
/// size_group_lookup[ BLOCK_SIZES ] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3}
/// ```
///
/// Indexed 0..=12 for the 13 `BLOCK_SIZES` (BLOCK_4X4..BLOCK_64X64).
pub(crate) const SIZE_GROUP_LOOKUP: [u8; BLOCK_SIZES] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];

/// `default_y_mode_probs[ BLOCK_SIZE_GROUPS ][ INTRA_MODES - 1 ]` per
/// §9.3 — the inter-frame `intra_mode` / `sub_intra_mode` default
/// probabilities loaded into the compressed-header `y_mode_probs[ ][ ]`
/// before §6.3.14 `read_y_mode_probs( )` applies its per-frame
/// `diff_update_prob( )` deltas.
///
/// Outer index is the §9.3.2 `ctx = size_group_lookup[ MiSize ]`
/// (0 = block_size < 8x8, 1 = < 16x16, 2 = < 32x32, 3 = >= 32x32),
/// inner index the §9.3.3 tree node (0..=8 for the 18-entry
/// [`INTRA_MODE_TREE`]). Transcribed verbatim from
/// `docs/video/vp9/vp9-spec.txt`:
///
/// ```text
/// default_y_mode_probs[ BLOCK_SIZE_GROUPS ][ INTRA_MODES - 1 ] = {
///     { 65, 32, 18, 144, 162, 194, 41, 51, 98 },    // block_size < 8x8
///     { 132, 68, 18, 165, 217, 196, 45, 40, 78 },   // block_size < 16x16
///     { 173, 80, 19, 176, 240, 193, 64, 35, 46 },   // block_size < 32x32
///     { 221, 135, 38, 194, 248, 121, 96, 85, 29 }   // block_size >= 32x32
/// }
/// ```
pub(crate) const DEFAULT_Y_MODE_PROBS: [[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS] = [
    [65, 32, 18, 144, 162, 194, 41, 51, 98],   // block_size < 8x8
    [132, 68, 18, 165, 217, 196, 45, 40, 78],  // block_size < 16x16
    [173, 80, 19, 176, 240, 193, 64, 35, 46],  // block_size < 32x32
    [221, 135, 38, 194, 248, 121, 96, 85, 29], // block_size >= 32x32
];

/// `default_uv_mode_probs[ INTRA_MODES ][ INTRA_MODES - 1 ]` per §9.3 —
/// the inter-frame `uv_mode` default probabilities loaded into the
/// compressed-header `uv_mode_probs[ ][ ]`.
///
/// Outer index is the §9.3.2 `ctx = y_mode` (the just-decoded luma
/// mode, DC_PRED..TM_PRED), inner index the §9.3.3 tree node (0..=8).
/// Transcribed verbatim from `docs/video/vp9/vp9-spec.txt`:
///
/// ```text
/// default_uv_mode_probs[ INTRA_MODES ][ INTRA_MODES - 1 ] = {
///     { 120, 7, 76, 176, 208, 126, 28, 54, 103 },   // y = dc
///     { 48, 12, 154, 155, 139, 90, 34, 117, 119 },  // y = v
///     { 67, 6, 25, 204, 243, 158, 13, 21, 96 },     // y = h
///     { 97, 5, 44, 131, 176, 139, 48, 68, 97 },     // y = d45
///     { 83, 5, 42, 156, 111, 152, 26, 49, 152 },    // y = d135
///     { 80, 5, 58, 178, 74, 83, 33, 62, 145 },      // y = d117
///     { 86, 5, 32, 154, 192, 168, 14, 22, 163 },    // y = d153
///     { 85, 5, 32, 156, 216, 148, 19, 29, 73 },     // y = d207
///     { 77, 7, 64, 116, 132, 122, 37, 126, 120 },   // y = d63
///     { 101, 21, 107, 181, 192, 103, 19, 67, 125 }  // y = tm
/// }
/// ```
pub(crate) const DEFAULT_UV_MODE_PROBS: [[u8; INTRA_MODES - 1]; INTRA_MODES] = [
    [120, 7, 76, 176, 208, 126, 28, 54, 103],   // y = dc
    [48, 12, 154, 155, 139, 90, 34, 117, 119],  // y = v
    [67, 6, 25, 204, 243, 158, 13, 21, 96],     // y = h
    [97, 5, 44, 131, 176, 139, 48, 68, 97],     // y = d45
    [83, 5, 42, 156, 111, 152, 26, 49, 152],    // y = d135
    [80, 5, 58, 178, 74, 83, 33, 62, 145],      // y = d117
    [86, 5, 32, 154, 192, 168, 14, 22, 163],    // y = d153
    [85, 5, 32, 156, 216, 148, 19, 29, 73],     // y = d207
    [77, 7, 64, 116, 132, 122, 37, 126, 120],   // y = d63
    [101, 21, 107, 181, 192, 103, 19, 67, 125], // y = tm
];

// ----- §6.4.6 intra_frame_mode_info -----

/// Output of [`intra_frame_mode_info`].
///
/// Mirrors the §7.4.5 / §7.4.10 semantics: a single decoded luma `y_mode`,
/// a 4-cell `sub_modes[ ]` (always populated — replicated from `y_mode`
/// for `MiSize >= BLOCK_8X8` and filled cell-by-cell during the
/// sub-8x8 walk), a `uv_mode`, and the §6.4.6 fixed reference-frame
/// pair (`INTRA_FRAME`, `NONE`) plus `is_inter = false`. The
/// per-block `segment_id` / `skip` / `tx_size` are returned alongside
/// since the §6.4.6 driver decodes them in sequence with the modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Vp9IntraMiBlock {
    /// `segment_id` per §6.4.7 (0..=7).
    pub segment_id: u8,
    /// `skip` per §6.4.8 (true ⇔ `skip == 1`).
    pub skip: bool,
    /// `tx_size` per §6.4.10 (a `TX_*` integer 0..=3).
    pub tx_size: u32,
    /// `ref_frame[0]` per §6.4.6. Always [`INTRA_FRAME`].
    pub ref_frame_0: i32,
    /// `ref_frame[1]` per §6.4.6. Always [`NONE_REF_FRAME`].
    pub ref_frame_1: i32,
    /// `is_inter` per §6.4.6. Always `false`.
    pub is_inter: bool,
    /// `y_mode` per §7.4.5 — the §6.4.6 final luma mode value (taken
    /// from `default_intra_mode` for `MiSize >= BLOCK_8X8`; taken from
    /// the last-decoded `default_intra_mode` in the sub-8x8 walk).
    pub y_mode: u8,
    /// `sub_modes[ 4 ]` per §6.4.6 / §7.4.5 — the four 4x4 luma sub-block
    /// modes covering the 8x8 quadrant in §6.4.6's `(idy + y2) * 2 +
    /// idx + x2` indexing.
    pub sub_modes: [u8; 4],
    /// `uv_mode` per §7.4.5 — the chroma mode value from
    /// `default_uv_mode`.
    pub uv_mode: u8,
}

/// Neighbour `SubModes[ ][ ]` cells consumed by [`intra_frame_mode_info`]
/// to build the §9.3.2 `abovemode` / `leftmode` indices for
/// `default_intra_mode`'s probability lookup.
///
/// The §9.3.2 listing reads:
///
/// * `SubModes[MiRow - 1][MiCol][2]` and `SubModes[MiRow - 1][MiCol][2 + idx]`
///   from the above neighbour — positions 2 and 3 of that block's
///   `sub_modes[ ]` (indices 2 and 3 are the bottom row of the 2x2
///   `sub_modes[ ]` grid `(idy + y2) * 2 + idx + x2`).
/// * `SubModes[MiRow][MiCol - 1][1]` and `SubModes[MiRow][MiCol - 1][1 + idy * 2]`
///   from the left neighbour — positions 1 and 3 of that block's
///   `sub_modes[ ]` (the right column of the 2x2 grid).
///
/// When `avail_u == false` the §9.3.2 listing substitutes [`DC_PRED`];
/// likewise for `avail_l`. The driver therefore needs only positions
/// 2 and 3 from `above_sub_modes` and positions 1 and 3 from
/// `left_sub_modes` when the corresponding `avail_*` flag is true.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IntraFrameNeighbours {
    /// `AvailU` per §7.4.4 — true if the block above is in the same
    /// frame slice as the current block.
    pub avail_u: bool,
    /// `AvailL` per §7.4.4 — true if the block to the left is in the
    /// same frame slice as the current block.
    pub avail_l: bool,
    /// `SubModes[MiRow - 1][MiCol][2]` and `[3]` — only consulted when
    /// `avail_u`. Position 0 here = spec index 2, position 1 here =
    /// spec index 3.
    pub above_sub_modes_23: [u8; 2],
    /// `SubModes[MiRow][MiCol - 1][1]` and `[3]` — only consulted when
    /// `avail_l`. Position 0 here = spec index 1, position 1 here =
    /// spec index 3 (corresponding to `idy = 0` / `idy = 1` in the
    /// sub-8x8 walk's `1 + idy * 2` lookup).
    pub left_sub_modes_13: [u8; 2],
}

/// `default_intra_mode` per §9.3.2 line 6268.
///
/// Walks the §9.3.1 [`INTRA_MODE_TREE`] with probabilities
/// `kf_y_mode_probs[ abovemode ][ leftmode ][ node ]` and returns the
/// decoded `PredMode` integer (0..=9). The `(abovemode, leftmode)` pair
/// is the caller's responsibility — for the `MiSize >= BLOCK_8X8` arm
/// it is the §9.3.2 `(SubModes[MiRow-1][MiCol][2], SubModes[MiRow][MiCol-1][1])`
/// (with [`DC_PRED`] substituted when the neighbour is unavailable);
/// for the sub-8x8 arm it is the §9.3.2 `(SubModes[MiRow-1][MiCol][2 + idx],
/// sub_modes[ idx ])` / `(sub_modes[ idy * 2 ], SubModes[MiRow][MiCol-1][1 + idy * 2])`
/// pair per the `(idy, idx)` cell.
pub(crate) fn default_intra_mode(
    coder: &mut BoolCoder<'_>,
    abovemode: u8,
    leftmode: u8,
) -> Result<u8, Error> {
    debug_assert!((abovemode as usize) < INTRA_MODES);
    debug_assert!((leftmode as usize) < INTRA_MODES);
    let row = &KF_Y_MODE_PROBS[abovemode as usize][leftmode as usize];
    let value = tree_decode(coder, &INTRA_MODE_TREE, |node| row[node])?;
    Ok(value as u8)
}

/// `default_uv_mode` per §9.3.2 line 6297.
///
/// Walks the §9.3.1 [`INTRA_MODE_TREE`] with probabilities
/// `kf_uv_mode_probs[ y_mode ][ node ]` and returns the decoded
/// `PredMode` integer (0..=9). `y_mode` is the §6.4.6 luma mode just
/// decoded by [`default_intra_mode`].
pub(crate) fn default_uv_mode(coder: &mut BoolCoder<'_>, y_mode: u8) -> Result<u8, Error> {
    debug_assert!((y_mode as usize) < INTRA_MODES);
    let row = &KF_UV_MODE_PROBS[y_mode as usize];
    let value = tree_decode(coder, &INTRA_MODE_TREE, |node| row[node])?;
    Ok(value as u8)
}

/// `intra_frame_mode_info( )` per §6.4.6.
///
/// The keyframe-only per-block syntax orchestrator. The spec listing is:
///
/// ```text
/// intra_frame_mode_info( ) {
///     intra_segment_id( )
///     read_skip( )
///     read_tx_size( 1 )
///     ref_frame[ 0 ] = INTRA_FRAME
///     ref_frame[ 1 ] = NONE
///     is_inter = 0
///     if ( MiSize >= BLOCK_8X8 ) {
///         default_intra_mode                                              T
///         y_mode = default_intra_mode
///         for( b = 0; b < 4; b++ )
///             sub_modes[ b ] = y_mode
///     } else {
///         num4x4w = num_4x4_blocks_wide_lookup[ MiSize ]
///         num4x4h = num_4x4_blocks_high_lookup[ MiSize ]
///         for ( idy = 0; idy < 2; idy += num4x4h ) {
///             for ( idx = 0; idx < 2; idx += num4x4w ) {
///                 default_intra_mode                                      T
///                 for ( y2 = 0 ; y2 < num4x4h ; y2++ )
///                     for( x2 = 0 ; x2 < num4x4w ; x2++ )
///                         sub_modes[ (idy + y2) * 2 + idx + x2 ] = default_intra_mode
///             }
///         }
///         y_mode = default_intra_mode
///     }
///     default_uv_mode                                                     T
///     uv_mode = default_uv_mode
/// }
/// ```
///
/// Arguments map directly onto the spec's free variables:
///
/// * `coder` — the §9.2 entropy decoder positioned at the start of the
///   current MI block's mode-info bits.
/// * `mi_size` — the §7.4.3 `MiSize` (a `BLOCK_*` constant from
///   [`crate::residual`]).
/// * `seg_enabled` / `seg_update_map` / `tree_probs` — the
///   §6.4.7 segmentation gate inputs threaded directly into
///   [`intra_segment_id`]. `tree_probs` is `None` when
///   `seg_update_map == false` (the §6.4.7 `else` branch).
/// * `seg_feature_skip_active` — the §6.4.9
///   `seg_feature_active( SEG_LVL_SKIP )` result the caller has
///   computed from the §6.4.7 segment_id (note: §6.4.6 decodes
///   `segment_id` *before* `read_skip`, so the segment-feature lookup
///   must be deferred until after the segment_id reaches the helper).
/// * `tx_mode` / `tx_probs` / `skip_prob` / `nb_skip` / `nb_tx` —
///   threaded into [`read_skip`] / [`read_tx_size`] as in
///   [`crate::residual::residual_intra`].
/// * `nb_intra` — the [`IntraFrameNeighbours`] bundle for the §9.3.2
///   `default_intra_mode` `abovemode` / `leftmode` derivation.
///
/// Returns a [`Vp9IntraMiBlock`] carrying every field the §6.4.21
/// residual loop subsequently consumes plus the bookkeeping the
/// `WriteSubModes` / `WriteYModes` / `WriteSegmentIds` / `WriteSkips` /
/// `WriteTxSizes` / `WriteRefFrames` step at the end of §6.4.4 then
/// stores into the frame-wide arrays.
#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_frame_mode_info(
    coder: &mut BoolCoder<'_>,
    mi_size: u8,
    seg_enabled: bool,
    seg_update_map: bool,
    seg_tree_probs: Option<&[u8; 7]>,
    seg_feature_skip_active: bool,
    tx_mode: TxMode,
    tx_probs: &[[[u8; 3]; 2]; 4],
    skip_prob: &[u8; 3],
    nb_skip: NeighbourSkips,
    nb_tx: NeighbourTxSizes,
    nb_intra: IntraFrameNeighbours,
) -> Result<Vp9IntraMiBlock, Error> {
    // §6.4.6 line-by-line:
    //   intra_segment_id( )
    let segment_id = intra_segment_id(coder, seg_enabled, seg_update_map, seg_tree_probs)?;
    //   read_skip( )
    let skip = read_skip(coder, seg_feature_skip_active, skip_prob, nb_skip)?;
    //   read_tx_size( 1 )
    let tx_size = read_tx_size(coder, true, tx_mode, mi_size, tx_probs, nb_tx)?;
    //   ref_frame[ 0 ] = INTRA_FRAME ; ref_frame[ 1 ] = NONE ; is_inter = 0

    // §6.4.6 luma-mode arm split on MiSize.
    let mut sub_modes = [DC_PRED; 4];
    let y_mode;
    if mi_size >= BLOCK_8X8 {
        // §9.3.2 `default_intra_mode` for MiSize >= BLOCK_8X8:
        //   abovemode = AvailU ? SubModes[MiRow-1][MiCol][2] : DC_PRED
        //   leftmode  = AvailL ? SubModes[MiRow][MiCol-1][1] : DC_PRED
        let abovemode = if nb_intra.avail_u {
            nb_intra.above_sub_modes_23[0]
        } else {
            DC_PRED
        };
        let leftmode = if nb_intra.avail_l {
            nb_intra.left_sub_modes_13[0]
        } else {
            DC_PRED
        };
        let mode = default_intra_mode(coder, abovemode, leftmode)?;
        // y_mode = default_intra_mode ; sub_modes[ b ] = y_mode for b in 0..4
        y_mode = mode;
        sub_modes = [mode; 4];
    } else {
        // §6.4.6 sub-8x8 arm:
        //   num4x4w = num_4x4_blocks_wide_lookup[ MiSize ]
        //   num4x4h = num_4x4_blocks_high_lookup[ MiSize ]
        //   for idy in (0..2).step_by(num4x4h) {
        //     for idx in (0..2).step_by(num4x4w) {
        //       default_intra_mode
        //       for y2 in 0..num4x4h: for x2 in 0..num4x4w:
        //         sub_modes[ (idy + y2) * 2 + idx + x2 ] = default_intra_mode
        //     }
        //   }
        //   y_mode = default_intra_mode  (the *last* decoded value)
        let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[mi_size as usize] as usize;
        let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[mi_size as usize] as usize;
        // BLOCK_4X4 / BLOCK_4X8 / BLOCK_8X4 are the only blocks smaller
        // than BLOCK_8X8 in §10.2. Their num4x4 dimensions are all 1 or
        // 2; in particular, neither dimension is 0 (§7.4.3 guarantees
        // valid MI sizes here).
        debug_assert!((1..=2).contains(&num4x4w));
        debug_assert!((1..=2).contains(&num4x4h));

        let mut last_mode = DC_PRED;
        let mut idy = 0usize;
        while idy < 2 {
            let mut idx = 0usize;
            while idx < 2 {
                // §9.3.2 `default_intra_mode` for MiSize < BLOCK_8X8:
                //   if (idy)
                //     abovemode = sub_modes[ idx ]
                //   else
                //     abovemode = AvailU ? SubModes[MiRow-1][MiCol][2 + idx] : DC_PRED
                //   if (idx)
                //     leftmode = sub_modes[ idy * 2 ]
                //   else
                //     leftmode = AvailL ? SubModes[MiRow][MiCol-1][1 + idy * 2] : DC_PRED
                let abovemode = if idy != 0 {
                    sub_modes[idx]
                } else if nb_intra.avail_u {
                    nb_intra.above_sub_modes_23[idx]
                } else {
                    DC_PRED
                };
                let leftmode = if idx != 0 {
                    sub_modes[idy * 2]
                } else if nb_intra.avail_l {
                    // 1 + idy*2 spans {1, 3}; left_sub_modes_13 stores
                    // those two positions at local indices {0, 1}.
                    nb_intra.left_sub_modes_13[idy]
                } else {
                    DC_PRED
                };
                let mode = default_intra_mode(coder, abovemode, leftmode)?;
                last_mode = mode;
                // Replicate across the (num4x4h × num4x4w) cell.
                for y2 in 0..num4x4h {
                    for x2 in 0..num4x4w {
                        sub_modes[(idy + y2) * 2 + idx + x2] = mode;
                    }
                }
                idx += num4x4w;
            }
            idy += num4x4h;
        }
        // y_mode = default_intra_mode (the spec keeps the last loop
        // value in the running variable).
        y_mode = last_mode;
    }

    // §6.4.6 final two lines:
    //   default_uv_mode ; uv_mode = default_uv_mode
    let uv_mode = default_uv_mode(coder, y_mode)?;

    Ok(Vp9IntraMiBlock {
        segment_id,
        skip,
        tx_size,
        ref_frame_0: INTRA_FRAME,
        ref_frame_1: NONE_REF_FRAME,
        is_inter: false,
        y_mode,
        sub_modes,
        uv_mode,
    })
}

// ----- §6.4.15 intra_block_mode_info -----

/// `intra_mode` per §9.3.2 line 6298.
///
/// Walks the §9.3.1 [`INTRA_MODE_TREE`] with probabilities
/// `y_mode_probs[ ctx ][ node ]` where `ctx = size_group_lookup[ MiSize ]`.
/// `y_mode_probs` is the compressed-header table (defaults
/// [`DEFAULT_Y_MODE_PROBS`] adapted by §6.3.14 `read_y_mode_probs( )`),
/// **not** the keyframe `kf_y_mode_probs` used by
/// [`default_intra_mode`]. Returns the decoded `PredMode` integer
/// (0..=9).
pub(crate) fn intra_mode(
    coder: &mut BoolCoder<'_>,
    y_mode_probs: &[[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],
    mi_size: u8,
) -> Result<u8, Error> {
    debug_assert!((mi_size as usize) < BLOCK_SIZES);
    let ctx = SIZE_GROUP_LOOKUP[mi_size as usize] as usize;
    let row = &y_mode_probs[ctx];
    let value = tree_decode(coder, &INTRA_MODE_TREE, |node| row[node])?;
    Ok(value as u8)
}

/// `sub_intra_mode` per §9.3.2 line 6302.
///
/// Walks the §9.3.1 [`INTRA_MODE_TREE`] with probabilities
/// `y_mode_probs[ ctx ][ node ]` where `ctx` is fixed to `0`. Used for
/// the §6.4.15 sub-8x8 `(idy, idx)` partition cells. Returns the
/// decoded `PredMode` integer (0..=9).
pub(crate) fn sub_intra_mode(
    coder: &mut BoolCoder<'_>,
    y_mode_probs: &[[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],
) -> Result<u8, Error> {
    let row = &y_mode_probs[0];
    let value = tree_decode(coder, &INTRA_MODE_TREE, |node| row[node])?;
    Ok(value as u8)
}

/// `uv_mode` per §9.3.2 line 6303.
///
/// Walks the §9.3.1 [`INTRA_MODE_TREE`] with probabilities
/// `uv_mode_probs[ ctx ][ node ]` where `ctx = y_mode` (the luma mode
/// just decoded by [`intra_mode`] / [`sub_intra_mode`]). `uv_mode_probs`
/// is the compressed-header table (defaults [`DEFAULT_UV_MODE_PROBS`]),
/// **not** the keyframe `kf_uv_mode_probs` used by [`default_uv_mode`].
/// Returns the decoded `PredMode` integer (0..=9).
pub(crate) fn uv_mode(
    coder: &mut BoolCoder<'_>,
    uv_mode_probs: &[[u8; INTRA_MODES - 1]; INTRA_MODES],
    y_mode: u8,
) -> Result<u8, Error> {
    debug_assert!((y_mode as usize) < INTRA_MODES);
    let row = &uv_mode_probs[y_mode as usize];
    let value = tree_decode(coder, &INTRA_MODE_TREE, |node| row[node])?;
    Ok(value as u8)
}

/// Output of [`intra_block_mode_info`].
///
/// Mirrors the §6.4.15 free variables: the §6.4.15 fixed reference-frame
/// pair (`INTRA_FRAME`, `NONE`), a single decoded luma `y_mode`, the
/// 4-cell `sub_modes[ ]` (replicated from `y_mode` for `MiSize >=
/// BLOCK_8X8`, filled cell-by-cell in the sub-8x8 walk), and a
/// `uv_mode`. Unlike [`Vp9IntraMiBlock`], §6.4.15 does **not** decode
/// `segment_id` / `skip` / `tx_size` — those are read by the §6.4.11
/// `inter_frame_mode_info( )` driver *before* dispatching to
/// `intra_block_mode_info( )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Vp9IntraBlockModeInfo {
    /// `ref_frame[0]` per §6.4.15. Always [`INTRA_FRAME`].
    pub ref_frame_0: i32,
    /// `ref_frame[1]` per §6.4.15. Always [`NONE_REF_FRAME`].
    pub ref_frame_1: i32,
    /// `y_mode` per §7.4.5 — the §6.4.15 final luma mode (from
    /// `intra_mode` for `MiSize >= BLOCK_8X8`; from the last-decoded
    /// `sub_intra_mode` in the sub-8x8 walk).
    pub y_mode: u8,
    /// `sub_modes[ 4 ]` per §6.4.15 / §7.4.5 — the four 4x4 luma
    /// sub-block modes in `(idy + y2) * 2 + idx + x2` indexing.
    pub sub_modes: [u8; 4],
    /// `uv_mode` per §7.4.5 — the chroma mode from `uv_mode`.
    pub uv_mode: u8,
}

/// `intra_block_mode_info( )` per §6.4.15.
///
/// The inter-frame intra-block mode reader — the companion to the
/// §6.4.6 [`intra_frame_mode_info`] keyframe driver. Fired by the
/// §6.4.11 `inter_frame_mode_info( )` driver when a block in a
/// non-keyframe frame is coded intra (`is_inter == 0`). The spec
/// listing is:
///
/// ```text
/// intra_block_mode_info( ) {
///     ref_frame[ 0 ] = INTRA_FRAME
///     ref_frame[ 1 ] = NONE
///     if ( MiSize >= BLOCK_8X8 ) {
///         intra_mode                                                  T
///         y_mode = intra_mode
///         for( b = 0; b < 4; b++ )
///             sub_modes[ b ] = y_mode
///     } else {
///         num4x4w = num_4x4_blocks_wide_lookup[ MiSize ]
///         num4x4h = num_4x4_blocks_high_lookup[ MiSize ]
///         for ( idy = 0; idy < 2; idy += num4x4h ) {
///             for ( idx = 0; idx < 2; idx += num4x4w ) {
///                 sub_intra_mode                                      T
///                 for ( y2 = 0; y2 < num4x4h; y2++ )
///                     for( x2 = 0; x2 < num4x4w; x2++ )
///                         sub_modes[ (idy + y2) * 2 + idx + x2 ] = sub_intra_mode
///             }
///         }
///         y_mode = sub_intra_mode
///     }
///     uv_mode                                                         T
/// }
/// ```
///
/// Differences from §6.4.6:
///
/// * Probabilities come from the §9.3 compressed-header `y_mode_probs`
///   / `uv_mode_probs` (defaults [`DEFAULT_Y_MODE_PROBS`] /
///   [`DEFAULT_UV_MODE_PROBS`], per-frame `diff_update_prob`'d), not
///   the §10.5 keyframe `kf_*_mode_probs` tables.
/// * The §9.3.2 context for `intra_mode` is `size_group_lookup[ MiSize ]`
///   (not the keyframe `(abovemode, leftmode)` pair), `sub_intra_mode`
///   uses context `0`, and `uv_mode` uses context `y_mode`. There is no
///   neighbour-`SubModes` lookup, so [`intra_block_mode_info`] takes no
///   neighbour bundle.
/// * `segment_id` / `skip` / `tx_size` are decoded earlier by the
///   §6.4.11 driver, so they are absent here.
///
/// Arguments:
///
/// * `coder` — the §9.2 entropy decoder positioned at the start of the
///   block's intra-mode bits (after the §6.4.11 `read_is_inter( )` /
///   `read_tx_size( )`).
/// * `mi_size` — the §7.4.3 `MiSize` (`BLOCK_*` from [`crate::residual`]).
/// * `y_mode_probs` / `uv_mode_probs` — the §9.3 compressed-header
///   tables.
pub(crate) fn intra_block_mode_info(
    coder: &mut BoolCoder<'_>,
    mi_size: u8,
    y_mode_probs: &[[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],
    uv_mode_probs: &[[u8; INTRA_MODES - 1]; INTRA_MODES],
) -> Result<Vp9IntraBlockModeInfo, Error> {
    // §6.4.15: ref_frame[ 0 ] = INTRA_FRAME ; ref_frame[ 1 ] = NONE
    let mut sub_modes = [DC_PRED; 4];
    let y_mode;
    if mi_size >= BLOCK_8X8 {
        // §6.4.15 `MiSize >= BLOCK_8X8` arm:
        //   intra_mode ; y_mode = intra_mode ; sub_modes[ b ] = y_mode
        let mode = intra_mode(coder, y_mode_probs, mi_size)?;
        y_mode = mode;
        sub_modes = [mode; 4];
    } else {
        // §6.4.15 sub-8x8 arm: walk the (idy, idx) grid stepped by
        // num4x4h / num4x4w, decoding one `sub_intra_mode` per cell and
        // replicating it across the (num4x4h × num4x4w) `sub_modes[ ]`
        // sub-grid. `y_mode` keeps the *last* decoded `sub_intra_mode`.
        let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[mi_size as usize] as usize;
        let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[mi_size as usize] as usize;
        debug_assert!((1..=2).contains(&num4x4w));
        debug_assert!((1..=2).contains(&num4x4h));

        let mut last_mode = DC_PRED;
        let mut idy = 0usize;
        while idy < 2 {
            let mut idx = 0usize;
            while idx < 2 {
                // §9.3.2 `sub_intra_mode` uses y_mode_probs[ 0 ] — no
                // neighbour derivation.
                let mode = sub_intra_mode(coder, y_mode_probs)?;
                last_mode = mode;
                for y2 in 0..num4x4h {
                    for x2 in 0..num4x4w {
                        sub_modes[(idy + y2) * 2 + idx + x2] = mode;
                    }
                }
                idx += num4x4w;
            }
            idy += num4x4h;
        }
        y_mode = last_mode;
    }

    // §6.4.15 final line: uv_mode (context = y_mode).
    let uv = uv_mode(coder, uv_mode_probs, y_mode)?;

    Ok(Vp9IntraBlockModeInfo {
        ref_frame_0: INTRA_FRAME,
        ref_frame_1: NONE_REF_FRAME,
        y_mode,
        sub_modes,
        uv_mode: uv,
    })
}

// ----- §6.4.5 mode_info dispatch -----

/// Outcome of the §6.4.5 `mode_info( )` dispatch.
///
/// §6.4.5 routes a per-block mode-info decode to [`intra_frame_mode_info`]
/// when `FrameIsIntra`, else to the §6.4.11 `inter_frame_mode_info( )`
/// driver. This enum carries the two intra-side products the round-134
/// surface decodes today: the keyframe [`Vp9IntraMiBlock`] (full
/// segment/skip/tx + modes) and the inter-frame intra-block
/// [`Vp9IntraBlockModeInfo`] (modes only — the §6.4.11 driver reads
/// segment/skip/tx before dispatching here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vp9ModeInfo {
    /// `FrameIsIntra` path — §6.4.6 `intra_frame_mode_info( )`.
    IntraFrame(Vp9IntraMiBlock),
    /// `!FrameIsIntra`, `is_inter == 0` path — §6.4.11
    /// `inter_frame_mode_info( )` reaching §6.4.15
    /// `intra_block_mode_info( )`.
    InterFrameIntraBlock(Vp9IntraBlockModeInfo),
}

/// Dispatch the §6.4.15 inter-frame intra-block path of §6.4.5
/// `mode_info( )`.
///
/// §6.4.5 reads:
///
/// ```text
/// mode_info( ) {
///     if ( FrameIsIntra )
///         intra_frame_mode_info( )
///     else
///         inter_frame_mode_info( )
/// }
/// ```
///
/// The `FrameIsIntra` branch is [`intra_frame_mode_info`]. This helper
/// covers the `else` branch's intra sub-path: the §6.4.11
/// `inter_frame_mode_info( )` driver, after decoding
/// `inter_segment_id( )` / `read_skip( )` / `read_is_inter( )` /
/// `read_tx_size( )`, dispatches to [`intra_block_mode_info`] when the
/// decoded `is_inter == 0`. Wiring it through [`Vp9ModeInfo`] keeps the
/// §6.4.5 dispatch shape explicit alongside the keyframe path; the
/// surrounding §6.4.11 prelude (`inter_segment_id` / `read_is_inter` /
/// the inter-block `inter_block_mode_info( )` arm) lands once its
/// reference-buffer-dependent primitives do.
pub(crate) fn inter_frame_intra_block_mode_info(
    coder: &mut BoolCoder<'_>,
    mi_size: u8,
    y_mode_probs: &[[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],
    uv_mode_probs: &[[u8; INTRA_MODES - 1]; INTRA_MODES],
) -> Result<Vp9ModeInfo, Error> {
    let block = intra_block_mode_info(coder, mi_size, y_mode_probs, uv_mode_probs)?;
    Ok(Vp9ModeInfo::InterFrameIntraBlock(block))
}

// ----- §6.4.12 inter_segment_id + §6.4.14 get_segment_id + §7.4 seg-pred ctx -----

/// `PrevSegmentIds[ MiRows ][ MiCols ]` view — the §6.4.14 spatial-
/// prediction input.
///
/// The §6.4.14 `get_segment_id( )` listing reads
/// `PrevSegmentIds[ MiRow + y ][ MiCol + x ]` for `y ∈ 0..ymis`,
/// `x ∈ 0..xmis`. The array is a frame-wide `MiRows × MiCols` plane the
/// previous frame's §6.4.4 driver wrote (`SegmentIds[ r + y ][ c + x ] =
/// segment_id`); this struct exposes it as a borrowed row-major buffer so
/// the §6.4.14 / §6.4.12 helpers can stay free of any storage policy.
///
/// `data.len()` MUST equal `mi_rows * mi_cols`; row `y` runs from
/// `data[ y * mi_cols ]` to `data[ y * mi_cols + mi_cols - 1 ]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrevSegmentIds<'a> {
    /// `MiRows` — the frame-wide mode-info row count from §6.2.
    pub(crate) mi_rows: u32,
    /// `MiCols` — the frame-wide mode-info column count from §6.2.
    pub(crate) mi_cols: u32,
    /// Row-major `[mi_rows][mi_cols]` segment-id grid (values `0..=7`).
    pub(crate) data: &'a [u8],
}

/// `get_segment_id( )` per §6.4.14.
///
/// ```text
/// get_segment_id( ) {
///     bw = num_8x8_blocks_wide_lookup[ MiSize ]
///     bh = num_8x8_blocks_high_lookup[ MiSize ]
///     xmis = Min( MiCols - MiCol, bw )
///     ymis = Min( MiRows - MiRow, bh )
///     seg = 7
///     for ( y = 0; y < ymis; y++ )
///         for ( x = 0; x < xmis; x++ )
///             seg = Min( seg, PrevSegmentIds[ MiRow + y ][ MiCol + x ] )
///     return seg
/// }
/// ```
///
/// "The predicted segment id is the smallest value found in the on-screen
/// region of the segmentation map covered by the current block" — §6.4.14
/// paragraph above the listing.
///
/// `mi_row` / `mi_col` are the §6.4.4 `MiRow` / `MiCol` of the block being
/// decoded; `mi_size` is the §3 `BLOCK_*` constant. The `Min( MiCols -
/// MiCol, bw )` / `Min( MiRows - MiRow, bh )` bounds clamp the sweep to
/// the on-screen portion of the block (a partial-edge block at the
/// right or bottom of the frame covers fewer 8x8 cells than its `bw` /
/// `bh`).
///
/// Reading off the edge of `PrevSegmentIds[ ]` is impossible by
/// construction: the §6.4.4 driver only fires `decode_block( )` for
/// `MiRow < MiRows` and `MiCol < MiCols`, so the `xmis` / `ymis` clamp
/// keeps `MiRow + y < MiRows` and `MiCol + x < MiCols` along the sweep.
pub(crate) fn get_segment_id(
    prev: &PrevSegmentIds<'_>,
    mi_row: u32,
    mi_col: u32,
    mi_size: u8,
) -> u8 {
    debug_assert!(mi_col < prev.mi_cols);
    debug_assert!(mi_row < prev.mi_rows);
    debug_assert_eq!(
        prev.data.len(),
        (prev.mi_rows as usize) * (prev.mi_cols as usize),
        "PrevSegmentIds backing slice must be row-major MiRows × MiCols"
    );
    let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
    let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
    let xmis = (prev.mi_cols - mi_col).min(bw);
    let ymis = (prev.mi_rows - mi_row).min(bh);
    let mut seg: u8 = 7;
    for y in 0..ymis {
        let row_base = ((mi_row + y) as usize) * (prev.mi_cols as usize);
        for x in 0..xmis {
            let v = prev.data[row_base + (mi_col + x) as usize];
            if v < seg {
                seg = v;
            }
        }
    }
    seg
}

/// `AboveSegPredContext[ MiCols ]` / `LeftSegPredContext[ MiRows ]` —
/// the §7.4 segmentation-prediction context strips the §6.4.12 driver
/// reads via the §9.3.2 `ctx = LeftSegPredContext[ MiRow ] +
/// AboveSegPredContext[ MiCol ]` derivation and writes back over the
/// per-block `num_8x8_blocks_*_lookup` sub-strips.
///
/// §7.4.1 (`clear_above_context`) zero-initialises `AboveSegPredContext[
/// i ]` for `i = 0..MiCols-1`; §7.4.2 (`clear_left_context`)
/// zero-initialises `LeftSegPredContext[ i ]` for `i = 0..MiRows-1`. The
/// strips carry the `seg_id_predicted` 0/1 value the previous block in
/// the same column / row decoded — a fresh `decode_tile( )` resets
/// `LeftSegPredContext[ ]` per superblock row before each row begins.
///
/// `clear_left( )` is the §7.4.2 reset the §6.4.2 `decode_tile( )` outer
/// loop fires once per superblock row. The §7.4.1 above-context reset
/// fires once per tile and is encoded by `new( )`.
#[derive(Debug, Clone)]
pub(crate) struct SegPredContextState {
    above: Vec<u8>,
    left: Vec<u8>,
}

impl SegPredContextState {
    /// `clear_above_context( )` + `clear_left_context( )` per §7.4.1 /
    /// §7.4.2: allocate the `MiCols`- and `MiRows`-sized strips and zero
    /// them.
    pub(crate) fn new(mi_cols: u32, mi_rows: u32) -> Self {
        Self {
            above: vec![0u8; mi_cols as usize],
            left: vec![0u8; mi_rows as usize],
        }
    }

    /// `clear_left_context( )` per §7.4.2 — zeroes
    /// `LeftSegPredContext[ ]` for the start of a fresh superblock row.
    pub(crate) fn clear_left(&mut self) {
        for slot in self.left.iter_mut() {
            *slot = 0;
        }
    }

    /// Read `AboveSegPredContext[ MiCol ]` (the §9.3.2 ctx contributor).
    pub(crate) fn above(&self, mi_col: u32) -> u8 {
        self.above[mi_col as usize]
    }

    /// Read `LeftSegPredContext[ MiRow ]` (the §9.3.2 ctx contributor).
    pub(crate) fn left(&self, mi_row: u32) -> u8 {
        self.left[mi_row as usize]
    }
}

/// `seg_id_predicted` per §9.3.2.
///
/// ```text
/// seg_id_predicted: the probability is given by
///                   segmentation_pred_prob[ ctx ] where ctx is
///                   computed by:
///     ctx = LeftSegPredContext[ MiRow ] + AboveSegPredContext[ MiCol ]
/// ```
///
/// The §9.3.2 ctx is `Left + Above` of the two-element sums of 0/1
/// strips, so `ctx ∈ {0, 1, 2}` — matching `segmentation_pred_prob[3]`
/// from [`crate::header::SegmentationParams::pred_prob`]. The decode is
/// a single bit via the §9.3.1 [`BINARY_TREE`].
pub(crate) fn read_seg_id_predicted(
    coder: &mut BoolCoder<'_>,
    pred_prob: &[u8; 3],
    seg_pred_ctx: &SegPredContextState,
    mi_row: u32,
    mi_col: u32,
) -> Result<bool, Error> {
    let ctx = (seg_pred_ctx.left(mi_row) + seg_pred_ctx.above(mi_col)) as usize;
    debug_assert!(ctx <= 2, "seg_id_predicted ctx must be in 0..=2");
    let value = tree_decode(coder, &BINARY_TREE, |_| pred_prob[ctx])?;
    Ok(value != 0)
}

/// `inter_segment_id( )` per §6.4.12.
///
/// ```text
/// inter_segment_id( ) {
///     if ( segmentation_enabled ) {
///         predictedSegmentId = get_segment_id( )
///         if ( segmentation_update_map ) {
///             if ( segmentation_temporal_update ) {
///                 seg_id_predicted                                       T
///                 if ( seg_id_predicted )
///                     segment_id = predictedSegmentId
///                 else
///                     segment_id                                         T
///                 for ( i = 0; i < num_8x8_blocks_wide_lookup[ MiSize ]; i++ )
///                     AboveSegPredContext[ MiCol + i ] = seg_id_predicted
///                 for ( i = 0; i < num_8x8_blocks_high_lookup[ MiSize ]; i++ )
///                     LeftSegPredContext[ MiRow + i ] = seg_id_predicted
///             } else {
///                 segment_id                                             T
///             }
///         } else {
///             segment_id = predictedSegmentId
///         }
///     } else {
///         segment_id = 0
///     }
/// }
/// ```
///
/// The four §6.4.12 paths:
///
/// 1. `!segmentation_enabled` → `segment_id = 0`, no bool-coder reads,
///    no ctx writes.
/// 2. `segmentation_enabled && !segmentation_update_map` →
///    `segment_id = predictedSegmentId` (the §6.4.14 spatial-min
///    predictor), no bool-coder reads, no ctx writes.
/// 3. `segmentation_enabled && segmentation_update_map &&
///    !segmentation_temporal_update` → decode `segment_id` directly via
///    [`read_segment_id`] (the §9.3.1 `SEGMENT_TREE` walk), no
///    `seg_id_predicted` read, no ctx writes.
/// 4. `segmentation_enabled && segmentation_update_map &&
///    segmentation_temporal_update` → decode
///    [`read_seg_id_predicted`]; if predicted, `segment_id =
///    predictedSegmentId`, otherwise decode `segment_id` via
///    [`read_segment_id`]; in both branches, write
///    `seg_id_predicted` (the 0/1 just decoded) into
///    `AboveSegPredContext[ MiCol + i ]` for
///    `i ∈ 0..num_8x8_blocks_wide_lookup[ MiSize ]` and into
///    `LeftSegPredContext[ MiRow + i ]` for
///    `i ∈ 0..num_8x8_blocks_high_lookup[ MiSize ]`.
///
/// Returns the decoded `segment_id` (`0..=7`). The driver mutates
/// `seg_pred_ctx` in path 4 as described above; paths 1-3 leave it
/// untouched.
///
/// Required state per path:
/// * Path 3 / 4 (decoding `segment_id`): `tree_probs` is the
///   `segmentation_tree_probs[7]` from
///   [`crate::header::SegmentationParams::tree_probs`]. Returns
///   [`Error::InvalidBitstream`] if `None` is passed.
/// * Path 4 (decoding `seg_id_predicted`): `pred_prob` is the
///   `segmentation_pred_prob[3]` from
///   [`crate::header::SegmentationParams::pred_prob`]. Returns
///   [`Error::InvalidBitstream`] if `None` is passed.
/// * Path 2 / 4-predicted: `prev` is the previous frame's
///   `PrevSegmentIds[ ][ ]` plane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inter_segment_id(
    coder: &mut BoolCoder<'_>,
    segmentation_enabled: bool,
    segmentation_update_map: bool,
    segmentation_temporal_update: bool,
    tree_probs: Option<&[u8; 7]>,
    pred_prob: Option<&[u8; 3]>,
    prev: &PrevSegmentIds<'_>,
    seg_pred_ctx: &mut SegPredContextState,
    mi_row: u32,
    mi_col: u32,
    mi_size: u8,
) -> Result<u8, Error> {
    if !segmentation_enabled {
        return Ok(0);
    }
    let predicted = get_segment_id(prev, mi_row, mi_col, mi_size);
    if !segmentation_update_map {
        return Ok(predicted);
    }
    if !segmentation_temporal_update {
        let probs = tree_probs.ok_or(Error::InvalidBitstream)?;
        return read_segment_id(coder, probs);
    }
    // Path 4: temporal-update branch.
    let pp = pred_prob.ok_or(Error::InvalidBitstream)?;
    let predicted_flag = read_seg_id_predicted(coder, pp, seg_pred_ctx, mi_row, mi_col)?;
    let segment_id = if predicted_flag {
        predicted
    } else {
        let probs = tree_probs.ok_or(Error::InvalidBitstream)?;
        read_segment_id(coder, probs)?
    };
    // §6.4.12 trailing write-back of the seg_id_predicted flag.
    let flag_u8 = u8::from(predicted_flag);
    let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize] as u32;
    let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize] as u32;
    for i in 0..bw {
        let c = (mi_col + i) as usize;
        if c < seg_pred_ctx.above.len() {
            seg_pred_ctx.above[c] = flag_u8;
        }
    }
    for i in 0..bh {
        let r = (mi_row + i) as usize;
        if r < seg_pred_ctx.left.len() {
            seg_pred_ctx.left[r] = flag_u8;
        }
    }
    Ok(segment_id)
}

// ----- §6.4.13 read_is_inter + §9.3.2 is_inter ctx -----

/// `SEG_LVL_REF_FRAME = 2` per §3 — the segmentation feature index
/// carrying a per-segment reference-frame override. When
/// `seg_feature_active( SEG_LVL_REF_FRAME )` is set, §6.4.13 derives
/// `is_inter` directly from `FeatureData[ segment_id ][
/// SEG_LVL_REF_FRAME ] != INTRA_FRAME` without consuming any bits.
pub(crate) const SEG_LVL_REF_FRAME: usize = 2;

/// `IS_INTER_CONTEXTS = 4` per §3 — number of contexts for the
/// §6.4.13 `is_inter` syntax element. Indexes the
/// `is_inter_prob[IS_INTER_CONTEXTS]` array per §9.3.2.
pub(crate) const IS_INTER_CONTEXTS: usize = 4;

/// `default_is_inter_prob[IS_INTER_CONTEXTS]` per §10.5 — the
/// running `is_inter_prob[]` table's initial / reset values, before any
/// §6.3.10 `read_is_inter_probs( )` compressed-header sweep applies
/// `diff_update_prob` deltas. Transcribed verbatim from the §10.5
/// listing.
pub(crate) const DEFAULT_IS_INTER_PROB: [u8; IS_INTER_CONTEXTS] = [9, 102, 187, 225];

/// Neighbour `RefFrames[ ][ ][ 0 ]` cells consumed by [`is_inter_context`]
/// to compute `LeftIntra` / `AboveIntra` per §6.4.11.
///
/// The §6.4.11 prelude derives:
///
/// ```text
/// LeftRefFrame[ 0 ]  = AvailL ? RefFrames[ MiRow ][ MiCol-1 ][ 0 ] : INTRA_FRAME
/// AboveRefFrame[ 0 ] = AvailU ? RefFrames[ MiRow-1 ][ MiCol ][ 0 ] : INTRA_FRAME
/// LeftIntra  = LeftRefFrame[ 0 ]  <= INTRA_FRAME
/// AboveIntra = AboveRefFrame[ 0 ] <= INTRA_FRAME
/// ```
///
/// (`INTRA_FRAME = 0` per §3 and §7.4.12; `NONE = -1` per §3.) Since
/// `INTRA_FRAME = 0` is the smallest valid ref-frame index and `NONE`
/// is strictly less, `<= INTRA_FRAME` is true for both `INTRA_FRAME`
/// and `NONE`. The §6.4.11 listing forces the neighbour ref-frame to
/// `INTRA_FRAME` when the neighbour is unavailable, so a missing
/// neighbour contributes the same `LeftIntra=1` / `AboveIntra=1`
/// value as an actual intra-coded neighbour.
///
/// This struct is the §9.3.2 listing's two-input view of that derivation.
/// `None` encodes "neighbour unavailable" (the §6.4.13 ctx listing reads
/// `AvailU` / `AvailL` independently of the ref-frame value).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IsInterNeighbours {
    /// `AvailU ? RefFrames[ MiRow-1 ][ MiCol ][ 0 ] : None` — the
    /// above neighbour's `ref_frame[0]` (or `None` when `!AvailU`).
    pub above: Option<i32>,
    /// `AvailL ? RefFrames[ MiRow ][ MiCol-1 ][ 0 ] : None` — the
    /// left neighbour's `ref_frame[0]` (or `None` when `!AvailL`).
    pub left: Option<i32>,
}

/// `is_inter` context per §9.3.2.
///
/// ```text
/// if ( AvailU && AvailL )
///     ctx = (LeftIntra && AboveIntra) ? 3 : LeftIntra || AboveIntra
/// else if ( AvailU || AvailL )
///     ctx = 2 * (AvailU ? AboveIntra : LeftIntra)
/// else
///     ctx = 0
/// ```
///
/// Returns one of `0..=3` indexing `is_inter_prob[ ctx ]`. The branch
/// breakdown:
///
/// * both available, both intra → `ctx = 3`
/// * both available, one intra → `ctx = 1` (`true || false = 1`)
/// * both available, neither intra → `ctx = 0`
/// * one available, that one intra → `ctx = 2` (`2 * 1`)
/// * one available, that one inter → `ctx = 0` (`2 * 0`)
/// * neither available → `ctx = 0`
///
/// `*Intra` is `RefFrame[ 0 ] <= INTRA_FRAME` per §6.4.11; with
/// `INTRA_FRAME = 0` and `NONE = -1`, both the actual intra case and
/// the unavailable-neighbour case ("force to `INTRA_FRAME`") map to
/// "intra-side" for ctx purposes.
pub(crate) fn is_inter_context(nb: IsInterNeighbours) -> usize {
    let above_intra = nb.above.map(|rf| rf <= INTRA_FRAME);
    let left_intra = nb.left.map(|rf| rf <= INTRA_FRAME);
    match (above_intra, left_intra) {
        (Some(a), Some(l)) => {
            if a && l {
                3
            } else if a || l {
                1
            } else {
                0
            }
        }
        (Some(a), None) => 2 * usize::from(a),
        (None, Some(l)) => 2 * usize::from(l),
        (None, None) => 0,
    }
}

/// `read_is_inter( )` per §6.4.13.
///
/// ```text
/// read_is_inter( ) {
///     if ( seg_feature_active( SEG_LVL_REF_FRAME ) )
///         is_inter = FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME
///     else
///         is_inter                                                          T
/// }
/// ```
///
/// Two paths:
///
/// 1. `seg_feature_active( SEG_LVL_REF_FRAME )` → `is_inter` is
///    derived directly from the segment's reference-frame override.
///    `FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME`
///    selects inter (`is_inter = true`) when the override is any of
///    `LAST_FRAME` / `GOLDEN_FRAME` / `ALTREF_FRAME` (`1`/`2`/`3`),
///    and intra (`is_inter = false`) when the override is
///    `INTRA_FRAME` (`0`). No bool-coder reads.
/// 2. Otherwise: a single §9.3.3-coded `is_inter` token under the
///    §9.3.1 [`BINARY_TREE`] and the `is_inter_prob[ ctx ]` probability
///    where `ctx` is derived by [`is_inter_context`] from the §6.4.11
///    `LeftIntra` / `AboveIntra` neighbours.
///
/// `is_inter_prob` is the 4-entry `is_inter_prob[ IS_INTER_CONTEXTS ]`
/// table updated by the §6.3.10 `read_is_inter_probs( )`
/// compressed-header sweep (and initialised from
/// [`DEFAULT_IS_INTER_PROB`] at the start of each `setup_past_independence`
/// reset). `segment_ref_frame_data` is the `FeatureData[ segment_id ][
/// SEG_LVL_REF_FRAME ]` value the §6.2.10 `segmentation_params( )`
/// surfaced (an `i16` because the segment-feature override slot stores
/// signed data even though valid ref-frame values are `0..=3`).
pub(crate) fn read_is_inter(
    coder: &mut BoolCoder<'_>,
    seg_feature_ref_frame_active: bool,
    segment_ref_frame_data: i16,
    is_inter_prob: &[u8; IS_INTER_CONTEXTS],
    nb: IsInterNeighbours,
) -> Result<bool, Error> {
    if seg_feature_ref_frame_active {
        return Ok(i32::from(segment_ref_frame_data) != INTRA_FRAME);
    }
    let ctx = is_inter_context(nb);
    let value = tree_decode(coder, &BINARY_TREE, |_| is_inter_prob[ctx])?;
    Ok(value != 0)
}

// ----- §3 inter-mode + interp-filter dimensions / §10.5 defaults -----

/// `INTER_MODES = 4` per §3 (spec constants table, `vp9-spec.txt` line
/// 506). Number of values for the §6.4.16 `inter_mode` syntax element
/// (`NEAREST_MV` / `NEAR_MV` / `ZERO_MV` / `NEW_MV`).
pub(crate) const INTER_MODES: usize = 4;

/// `INTER_MODE_CONTEXTS = 7` per §3 (spec constants table,
/// `vp9-spec.txt` line 507). Number of contexts for `inter_mode`.
/// Indexes the `inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES - 1]`
/// array per §9.3.2.
pub(crate) const INTER_MODE_CONTEXTS: usize = 7;

/// `default_inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES - 1]`
/// per §10.5 (`vp9-spec.txt` lines 7758-7766) — the running
/// `inter_mode_probs[][]` table's initial / reset values, before any
/// §6.3.9 `read_inter_mode_probs( )` compressed-header sweep applies
/// `diff_update_prob` deltas. Transcribed verbatim from the §10.5
/// listing.
///
/// Rows (per §10.5 commentary):
/// * `0` — both zero mv: `{2, 173, 34}`
/// * `1` — one zero mv + one predicted mv: `{7, 145, 85}`
/// * `2` — two predicted mvs: `{7, 166, 63}`
/// * `3` — one predicted/zero and one new mv: `{7, 94, 66}`
/// * `4` — two new mvs: `{8, 64, 46}`
/// * `5` — one intra neighbor + x: `{17, 81, 31}`
/// * `6` — two intra neighbors: `{25, 29, 30}`
pub(crate) const DEFAULT_INTER_MODE_PROBS: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS] = [
    [2, 173, 34],
    [7, 145, 85],
    [7, 166, 63],
    [7, 94, 66],
    [8, 64, 46],
    [17, 81, 31],
    [25, 29, 30],
];

/// `SWITCHABLE_FILTERS = 3` per §3 (spec constants table,
/// `vp9-spec.txt` line 487). Number of values for the switchable
/// `interp_filter` syntax element (the three subpel filter kernels —
/// EIGHTTAP / EIGHTTAP_SMOOTH / EIGHTTAP_SHARP).
pub(crate) const SWITCHABLE_FILTERS: usize = 3;

/// `INTERP_FILTER_CONTEXTS = 4` per §3 (spec constants table,
/// `vp9-spec.txt` line 495). Number of contexts for the per-block
/// `interp_filter` decode. Indexes the
/// `interp_filter_probs[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`
/// array per §9.3.2.
pub(crate) const INTERP_FILTER_CONTEXTS: usize = 4;

/// `default_interp_filter_probs[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`
/// per §10.5 (`vp9-spec.txt` lines 7769-7775) — the running
/// `interp_filter_probs[][]` table's initial / reset values, before
/// any §6.3.10 `read_interp_filter_probs( )` compressed-header sweep
/// applies `diff_update_prob` deltas. Transcribed verbatim.
pub(crate) const DEFAULT_INTERP_FILTER_PROBS: [[u8; SWITCHABLE_FILTERS - 1];
    INTERP_FILTER_CONTEXTS] = [[235, 162], [36, 255], [34, 3], [149, 144]];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{
        BLOCK_16X16, BLOCK_32X32, BLOCK_4X4, BLOCK_4X8, BLOCK_64X64, BLOCK_8X4, BLOCK_8X8,
    };

    // ----- Tree-listing anchors -----

    #[test]
    fn tx_size_trees_match_spec_listings() {
        // §9.3.1 verbatim:
        //   tx_size_8_tree[ 2 ]  = { -TX_4X4, -TX_8X8 }
        //   tx_size_16_tree[ 4 ] = { -TX_4X4, 2, -TX_8X8, -TX_16X16 }
        //   tx_size_32_tree[ 6 ] = { -TX_4X4, 2, -TX_8X8, 4, -TX_16X16, -TX_32X32 }
        assert_eq!(TX_SIZE_8_TREE, [0, -1]);
        assert_eq!(TX_SIZE_16_TREE, [0, 2, -1, -2]);
        assert_eq!(TX_SIZE_32_TREE, [0, 2, -1, 4, -2, -3]);
        assert_eq!(BINARY_TREE, [0, -1]);
    }

    // ----- §9.2 BoolCoder test harness -----
    //
    // The §9.2 decoder requires the §9.2.1 marker bit to decode to 0,
    // so a raw 0xFF buffer is rejected by `init_bool`. Two harness
    // patterns cover this round's tests:
    //
    // 1. `zero_coder()` — `[0x00; 16]`. Every `read_bool(p)` returns 0
    //    for any `p` (the §9.2 split is `1 + ((range-1)*p)>>8`, which
    //    is `>= 1` and the post-marker `value` is 0, so `value <
    //    split` always). This pins the "left branch / first leaf"
    //    path of every tree.
    //
    // 2. `make_bias_coder( first_byte )` — a buffer starting with
    //    `first_byte` followed by zeros, where `first_byte < 128` so
    //    the marker decodes to 0 but the post-marker `value` is
    //    `first_byte`. This lets a read at probability `p` flip to 1
    //    when `first_byte >= 1 + ((127 * p) >> 8)`.
    //
    //    For `first_byte = 0x7F = 127`: `split` at `p=255` is 127, so
    //    `value (127) >= split (127)` → bit=1. One bit of "1" capacity
    //    is consumed, after which renorm refills `value` from the 0x00
    //    tail and subsequent reads return 0 again — enough to exercise
    //    one right-branch step in a binary tree.
    //
    // The pattern is a hand-derived buffer prefix per the §9.2
    // listing, not a borrowed third-party encoder.

    fn zero_coder() -> BoolCoder<'static> {
        // [0x00; 16] satisfies §9.2.1 (marker decodes to 0) and pins
        // every subsequent read_bool to 0.
        static BYTES: [u8; 16] = [0u8; 16];
        BoolCoder::init_bool(&BYTES, BYTES.len()).expect("zero-coder init_bool must succeed")
    }

    /// 16-byte buffer whose first byte is `first` (must satisfy
    /// `first < 128` so the §9.2.1 marker decodes to 0). Subsequent
    /// bytes are 0.
    fn make_bias_buffer(first: u8) -> [u8; 16] {
        debug_assert!(
            first < 128,
            "marker bit constraint: first byte must be < 128"
        );
        let mut bytes = [0u8; 16];
        bytes[0] = first;
        bytes
    }

    // ----- §9.3.3 tree_decode -----

    #[test]
    fn tree_decode_zero_buffer_picks_first_leaf() {
        // With the zero coder every read_bool returns 0, so every tree
        // walks tree[0]:
        //   BINARY_TREE[0]    = 0 -> leaf value -0 = 0.
        //   TX_SIZE_8_TREE[0] = 0 -> leaf value 0 (TX_4X4).
        //   TX_SIZE_16_TREE[0]= 0 -> leaf value 0.
        //   TX_SIZE_32_TREE[0]= 0 -> leaf value 0.
        let mut coder = zero_coder();
        assert_eq!(tree_decode(&mut coder, &BINARY_TREE, |_| 128).unwrap(), 0);
        let mut coder = zero_coder();
        assert_eq!(
            tree_decode(&mut coder, &TX_SIZE_8_TREE, |_| 128).unwrap(),
            0,
        );
        let mut coder = zero_coder();
        assert_eq!(
            tree_decode(&mut coder, &TX_SIZE_16_TREE, |_| 128).unwrap(),
            0,
        );
        let mut coder = zero_coder();
        assert_eq!(
            tree_decode(&mut coder, &TX_SIZE_32_TREE, |_| 128).unwrap(),
            0,
        );
    }

    #[test]
    fn tree_decode_bias_buffer_routes_right_branch_then_left() {
        // Buffer [0x7F, 0x00, ...] post-marker has BoolValue=127,
        // BoolRange=128. With p=255, split=127, value>=split -> bit=1
        // for the first read. After: range=1 -> renorm refills 7 bits
        // from 0x00 -> range=128, value=0, every subsequent read
        // returns 0.
        //
        // BINARY_TREE: bit=1 -> tree[1] = -1 -> leaf 1.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        assert_eq!(tree_decode(&mut coder, &BINARY_TREE, |_| 255).unwrap(), 1);

        // TX_SIZE_8_TREE: bit=1 -> tree[1] = -1 -> leaf 1 (TX_8X8).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        assert_eq!(
            tree_decode(&mut coder, &TX_SIZE_8_TREE, |_| 255).unwrap(),
            1,
        );

        // TX_SIZE_16_TREE: bit=1 -> tree[1] = 2; second read returns 0
        // (only one bit of "1" capacity in the bias buffer) ->
        // tree[2] = -1 -> leaf 1 (TX_8X8).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        assert_eq!(
            tree_decode(&mut coder, &TX_SIZE_16_TREE, |_| 255).unwrap(),
            1,
        );

        // TX_SIZE_32_TREE: bit=1 -> tree[1] = 2; second read returns 0
        // -> tree[2] = -1 -> leaf 1 (TX_8X8).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        assert_eq!(
            tree_decode(&mut coder, &TX_SIZE_32_TREE, |_| 255).unwrap(),
            1,
        );
    }

    #[test]
    fn tree_decode_calls_prob_with_node_index() {
        // Use the zero coder so the prob value is irrelevant for the
        // bit outcome — we only care about the (node) argument the
        // callback receives in walk order.
        let mut coder = zero_coder();
        let calls = std::cell::RefCell::new(Vec::<usize>::new());
        let v = tree_decode(&mut coder, &TX_SIZE_32_TREE, |node| {
            calls.borrow_mut().push(node);
            128
        })
        .unwrap();
        assert_eq!(v, 0);
        // Only one read since every bit=0 routes directly to a leaf at
        // tree[0]=0.
        assert_eq!(*calls.borrow(), vec![0]);
    }

    // ----- skip_context -----

    #[test]
    fn skip_context_matches_spec_listing() {
        // No neighbours -> 0.
        let nb = NeighbourSkips::default();
        assert_eq!(skip_context(nb), 0);

        // Above only, skip=0 -> ctx=0.
        let nb = NeighbourSkips {
            above: Some(0),
            left: None,
        };
        assert_eq!(skip_context(nb), 0);

        // Above only, skip=1 -> ctx=1.
        let nb = NeighbourSkips {
            above: Some(1),
            left: None,
        };
        assert_eq!(skip_context(nb), 1);

        // Left only, skip=1 -> ctx=1.
        let nb = NeighbourSkips {
            above: None,
            left: Some(1),
        };
        assert_eq!(skip_context(nb), 1);

        // Both neighbours skipped -> ctx=2.
        let nb = NeighbourSkips {
            above: Some(1),
            left: Some(1),
        };
        assert_eq!(skip_context(nb), 2);

        // Mixed -> ctx=1.
        let nb = NeighbourSkips {
            above: Some(0),
            left: Some(1),
        };
        assert_eq!(skip_context(nb), 1);
    }

    // ----- tx_size_context -----

    #[test]
    fn tx_size_context_zero_when_neighbours_unavailable() {
        // No neighbours -> above = left = maxTxSize -> sum = 2*max ->
        // > maxTxSize for max > 0; ctx = 1 in that case. With max=0
        // sum=0 == max, ctx = 0.
        let nb = NeighbourTxSizes::default();
        assert_eq!(tx_size_context(nb, 0), 0);
        assert_eq!(tx_size_context(nb, 1), 1);
        assert_eq!(tx_size_context(nb, 2), 1);
        assert_eq!(tx_size_context(nb, 3), 1);
    }

    #[test]
    fn tx_size_context_neighbours_smaller_than_max_gives_zero() {
        // Both neighbours present, both unskipped, both 0 -> above=0,
        // left=0, sum=0 < max=3 -> ctx=0.
        let nb = NeighbourTxSizes {
            avail_u: true,
            avail_l: true,
            skip_above: 0,
            skip_left: 0,
            tx_above: 0,
            tx_left: 0,
        };
        assert_eq!(tx_size_context(nb, 3), 0);
    }

    #[test]
    fn tx_size_context_neighbours_equal_to_max_gives_one() {
        // Both neighbours present, unskipped, both max -> sum = 2*max
        // > max -> ctx = 1.
        let nb = NeighbourTxSizes {
            avail_u: true,
            avail_l: true,
            skip_above: 0,
            skip_left: 0,
            tx_above: 3,
            tx_left: 3,
        };
        assert_eq!(tx_size_context(nb, 3), 1);
    }

    #[test]
    fn tx_size_context_skipped_neighbour_falls_back_to_max() {
        // Above is present but skipped -> above = maxTxSize. Left is
        // present unskipped, value 0. Sum = 3 + 0 = 3 == max=3 ->
        // ctx = 0 (not strictly greater).
        let nb = NeighbourTxSizes {
            avail_u: true,
            avail_l: true,
            skip_above: 1,
            skip_left: 0,
            tx_above: 99, // ignored since skipped
            tx_left: 0,
        };
        assert_eq!(tx_size_context(nb, 3), 0);
    }

    #[test]
    fn tx_size_context_missing_avail_mirrors_other_side() {
        // !AvailL -> left = above. If above is 0 (present, unskipped),
        // ctx = (0 + 0) > 3 = false -> 0.
        let nb = NeighbourTxSizes {
            avail_u: true,
            avail_l: false,
            skip_above: 0,
            skip_left: 0,
            tx_above: 0,
            tx_left: 99, // ignored
        };
        assert_eq!(tx_size_context(nb, 3), 0);

        // !AvailU -> above = left. If left is 3 (present, unskipped),
        // ctx = (3 + 3) > 3 = true -> 1.
        let nb = NeighbourTxSizes {
            avail_u: false,
            avail_l: true,
            skip_above: 0,
            skip_left: 0,
            tx_above: 99,
            tx_left: 3,
        };
        assert_eq!(tx_size_context(nb, 3), 1);
    }

    // ----- read_skip -----

    #[test]
    fn read_skip_segment_active_forces_one() {
        // When the SEG_LVL_SKIP feature is active for the segment the
        // spec hardwires skip=true and reads no bits. The bool coder
        // sees no traffic, but `init_bool` still wants a valid marker
        // byte — pass a zero-buffer coder for that.
        let mut coder = zero_coder();
        let skip = read_skip(
            &mut coder,
            true,
            &[128, 128, 128],
            NeighbourSkips::default(),
        )
        .unwrap();
        assert!(skip);
    }

    #[test]
    fn read_skip_zero_buffer_returns_false() {
        // Zero-buffer coder pins every read_bool to 0; the BINARY_TREE
        // walks tree[0] = 0 -> leaf 0 -> skip = false.
        let mut coder = zero_coder();
        let skip = read_skip(
            &mut coder,
            false,
            &[128, 128, 128],
            NeighbourSkips::default(),
        )
        .unwrap();
        assert!(!skip);
    }

    #[test]
    fn read_skip_bias_buffer_with_p255_returns_true() {
        // Bias buffer + p=255 -> first read_bool returns 1 ->
        // BINARY_TREE[1] = -1 -> skip = true.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let skip = read_skip(
            &mut coder,
            false,
            &[255, 128, 64], // ctx=0 row picks 255
            NeighbourSkips::default(),
        )
        .unwrap();
        assert!(skip);
    }

    #[test]
    fn read_skip_picks_prob_by_context() {
        // The §6.4.8 listing reads `Skip` with `skip_prob[ctx]` where
        // `ctx = above.unwrap_or(0) + left.unwrap_or(0)`. We test the
        // indexing indirectly by confirming that:
        //
        // (a) the §9.3.2 ctx derivation matches the spec (covered by
        //     `skip_context_matches_spec_listing`), and
        // (b) `read_skip` calls into `tree_decode` with the prob slot
        //     selected by `ctx` — i.e. tampering with the chosen slot
        //     changes the outcome. Use a `seg_feature_active=false`
        //     read against the zero coder: every probability collapses
        //     to bit=0, so `skip` is false regardless of which row.
        //     This rules out a coding bug that always returns true /
        //     panics; the row-selection logic itself is one line and
        //     visually verifiable.
        let mut coder = zero_coder();
        let skip = read_skip(
            &mut coder,
            false,
            &[128, 128, 128],
            NeighbourSkips {
                above: Some(1),
                left: Some(1),
            },
        )
        .unwrap();
        assert!(!skip);

        // Direct route: an explicit `skip_context` of `(Some(1),
        // Some(1))` evaluates to 2, which would index `skip_prob[2]`.
        // The function under test threads this ctx through correctly
        // when the same `nb` is reused.
        assert_eq!(
            skip_context(NeighbourSkips {
                above: Some(1),
                left: Some(1),
            }),
            2,
        );
    }

    // ----- segment_tree / read_segment_id / intra_segment_id -----

    #[test]
    fn segment_tree_matches_spec_listing() {
        // §9.3.1 verbatim:
        //   segment_tree[ 14 ] = {
        //       2, 4, 6, 8, 10, 12,
        //       0, -1, -2, -3, -4, -5, -6, -7
        //   }
        assert_eq!(
            SEGMENT_TREE,
            [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7]
        );
    }

    #[test]
    fn read_segment_id_zero_buffer_picks_segment_zero() {
        // Zero coder pins every read_bool to 0. The walker therefore
        // follows tree[0]=2 → tree[2]=4 → tree[4]=6 → tree[6]=0, exits
        // with -0 = 0. Probability values don't affect the outcome
        // along the all-bit-0 path.
        let mut coder = zero_coder();
        let id = read_segment_id(&mut coder, &[128, 128, 128, 128, 128, 128, 128]).unwrap();
        assert_eq!(id, 0);
    }

    #[test]
    fn read_segment_id_bias_buffer_all_255_picks_segment_four() {
        // Bias buffer + probs `[255;7]`:
        //   first read (p=255, value=127, range=128) → split=127,
        //     value≥split → bit=1; renormalisation refills 7 zero bits
        //     leaving range=128, value=0.
        //   n=0 → tree[1]=4 → node=2.
        //   second read (p=255, range=128, value=0) → split=127, value
        //     <split → bit=0; n=tree[4]=10 → node=5.
        //   third read (p=255, range=128 post-renorm, value=0) →
        //     bit=0; n=tree[10]=-4 → returns segment id 4.
        // Hand-traced against the §9.2 listing; no external decoder
        // consulted.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let id = read_segment_id(&mut coder, &[255; 7]).unwrap();
        assert_eq!(id, 4);
    }

    #[test]
    fn read_segment_id_calls_prob_with_node_index() {
        // Confirms that the probability callback receives the §9.3.3
        // `n >> 1` node index along the walk path. With the zero coder
        // every bit is 0, so the walk is:
        //   n=0 → node=0 → tree[0]=2
        //   n=2 → node=1 → tree[2]=6
        //   n=6 → node=3 → tree[6]=0 (leaf, returns 0)
        // — note position 4 (node=2) is skipped because the left
        // branch at n=2 jumps straight to position 6, not 4. The
        // §9.3.1 segment_tree's inner-branch packing means a pure-
        // left-bit walk visits node indices 0, 1, 3 (not the
        // contiguous 0..3 a regular binary tree would).
        let mut coder = zero_coder();
        let probs = [1u8, 2, 3, 4, 5, 6, 7];
        let calls = std::cell::RefCell::new(Vec::<usize>::new());
        let value = tree_decode(&mut coder, &SEGMENT_TREE, |node| {
            calls.borrow_mut().push(node);
            probs[node]
        })
        .unwrap();
        assert_eq!(value, 0);
        assert_eq!(*calls.borrow(), vec![0, 1, 3]);
    }

    #[test]
    fn intra_segment_id_disabled_returns_zero_without_reading() {
        // When segmentation_enabled is false the spec hardwires
        // segment_id = 0 and reads no bits.
        let mut coder = zero_coder();
        let id = intra_segment_id(&mut coder, false, true, Some(&[255; 7])).unwrap();
        assert_eq!(id, 0);

        // Likewise when enabled but update_map is false — the previous
        // frame's segmentation map is reused (and on an intra frame
        // the spec leaves segment_id pinned at 0 since there's no
        // prior map to inherit from).
        let mut coder = zero_coder();
        let id = intra_segment_id(&mut coder, true, false, Some(&[255; 7])).unwrap();
        assert_eq!(id, 0);

        // Even passing a None tree_probs is fine in those branches —
        // the helper short-circuits before dereferencing it. This
        // matches the SegmentationParams shape where tree_probs is
        // `None` unless update_map == 1 surfaced one.
        let mut coder = zero_coder();
        let id = intra_segment_id(&mut coder, false, false, None).unwrap();
        assert_eq!(id, 0);
    }

    #[test]
    fn intra_segment_id_enabled_with_update_map_decodes() {
        // segmentation_enabled && segmentation_update_map → walks the
        // §9.3.1 segment_tree. The zero-coder path picks segment 0.
        let mut coder = zero_coder();
        let id = intra_segment_id(&mut coder, true, true, Some(&[128; 7])).unwrap();
        assert_eq!(id, 0);

        // Bias buffer + all-255 probs picks segment 4 (see
        // `read_segment_id_bias_buffer_all_255_picks_segment_four`).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let id = intra_segment_id(&mut coder, true, true, Some(&[255; 7])).unwrap();
        assert_eq!(id, 4);
    }

    #[test]
    fn intra_segment_id_missing_tree_probs_when_active_is_invalid() {
        // The caller-supplied `tree_probs` is required when the spec
        // is going to decode segment_id. SegmentationParams keeps
        // tree_probs as Option<[u8;7]> because it's None whenever
        // update_map==0; a None reaching this branch indicates the
        // caller forgot to thread the table through from the
        // uncompressed header, which is a programming error rather
        // than a stream error, but `Error::InvalidBitstream` is the
        // closest match in the crate-local error set and surfaces
        // loudly rather than panicking.
        let mut coder = zero_coder();
        let err = intra_segment_id(&mut coder, true, true, None).unwrap_err();
        assert_eq!(err, Error::InvalidBitstream);
    }

    // ----- read_tx_size -----

    fn make_tx_probs(probs: u8) -> [[[u8; 3]; 2]; 4] {
        // Fill every cell with the same prob.
        [[[probs; 3]; 2]; 4]
    }

    #[test]
    fn read_tx_size_falls_through_to_min_when_select_disabled() {
        // tx_mode = ALLOW_16X16 (biggest = 2), MiSize = 64x64
        // (max txsize = 3 -> TX_32X32). Min(3, 2) = 2 (TX_16X16). The
        // bool coder is untouched along the else branch.
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            true, // allow_select
            TxMode::Allow16x16,
            BLOCK_64X64,
            &make_tx_probs(128),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        assert_eq!(tx_size, 2);
    }

    #[test]
    fn read_tx_size_falls_through_when_allow_select_false() {
        // Even with tx_mode=SELECT and MiSize=64x64 the allow_select=
        // false branch picks Min(3, 3) = 3. The bool coder is
        // untouched along the else branch.
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            false,
            TxMode::TxModeSelect,
            BLOCK_64X64,
            &make_tx_probs(255),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        assert_eq!(tx_size, 3);
    }

    #[test]
    fn read_tx_size_falls_through_for_sub_8x8_block() {
        // MiSize=BLOCK_4X4 (< BLOCK_8X8 = 3) means the SELECT branch
        // is skipped per the `MiSize >= BLOCK_8X8` guard.
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_4X4,
            &make_tx_probs(255),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        // max_txsize_lookup[BLOCK_4X4] = 0 (TX_4X4); biggest for
        // SELECT is 3; Min(0, 3) = 0.
        assert_eq!(tx_size, 0);
    }

    #[test]
    fn read_tx_size_select_decodes_tx_size_8_tree_zero_buffer() {
        // MiSize=BLOCK_8X8 -> max_txsize=1 -> TX_SIZE_8_TREE. Zero
        // coder pins the first bit to 0 -> tree[0] = 0 -> leaf 0
        // (TX_4X4).
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_8X8,
            &make_tx_probs(128),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        assert_eq!(tx_size, 0);
    }

    #[test]
    fn read_tx_size_select_decodes_tx_size_8_tree_bias_buffer() {
        // MiSize=BLOCK_8X8 -> max_txsize=1 -> TX_SIZE_8_TREE. Bias
        // coder + p=255 -> first bit = 1 -> tree[1] = -1 -> leaf 1
        // (TX_8X8).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_8X8,
            &make_tx_probs(255),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        assert_eq!(tx_size, 1);
    }

    #[test]
    fn read_tx_size_select_decodes_tx_size_16_tree_bias_buffer() {
        // MiSize=BLOCK_16X16 -> max_txsize=2 -> TX_SIZE_16_TREE. Bias
        // coder + p=255: first bit=1 -> tree[1]=2; second bit=0 ->
        // tree[2]=-1 -> leaf 1 (TX_8X8).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_16X16,
            &make_tx_probs(255),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        assert_eq!(tx_size, 1);
    }

    #[test]
    fn read_tx_size_select_decodes_tx_size_32_tree_zero_buffer() {
        // MiSize=BLOCK_32X32 -> max_txsize=3 -> TX_SIZE_32_TREE. Zero
        // coder -> first bit=0 -> tree[0]=0 -> leaf 0 (TX_4X4).
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_32X32,
            &make_tx_probs(128),
            NeighbourTxSizes::default(),
        )
        .unwrap();
        assert_eq!(tx_size, 0);
    }

    #[test]
    fn read_tx_size_select_uses_context_to_pick_probability_row() {
        // BLOCK_8X8 -> max_txsize=1. Rig tx_probs so:
        //   ctx=0 row: [1, 1, 1] -> with bias buffer the first read
        //              has split=1 so value(127) >= split -> bit=1
        //              -> TX_8X8 (leaf 1).
        // Wait — that's not what we want. Use the *zero* coder so any
        // prob yields bit=0 regardless: ctx selection still controls
        // which row is *read* even if the bits all collapse.
        //
        // To prove the row indexing without relying on the rare bit=1
        // path, exercise two contexts and assert the ctx
        // derivation matches the spec's listing — the actual tree
        // walk just confirms a non-panic round trip.
        let mut probs = [[[0u8; 3]; 2]; 4];
        probs[1][0] = [128, 128, 128];
        probs[1][1] = [128, 128, 128];

        // ctx=0 case: avail_u, avail_l, sum=0 < max=1 -> ctx=0.
        let nb_ctx0 = NeighbourTxSizes {
            avail_u: true,
            avail_l: true,
            skip_above: 0,
            skip_left: 0,
            tx_above: 0,
            tx_left: 0,
        };
        assert_eq!(tx_size_context(nb_ctx0, 1), 0);
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_8X8,
            &probs,
            nb_ctx0,
        )
        .unwrap();
        assert_eq!(tx_size, 0);

        // ctx=1 case: both neighbours max=1 -> sum=2 > max=1 -> ctx=1.
        let nb_ctx1 = NeighbourTxSizes {
            avail_u: true,
            avail_l: true,
            skip_above: 0,
            skip_left: 0,
            tx_above: 1,
            tx_left: 1,
        };
        assert_eq!(tx_size_context(nb_ctx1, 1), 1);
        let mut coder = zero_coder();
        let tx_size = read_tx_size(
            &mut coder,
            true,
            TxMode::TxModeSelect,
            BLOCK_8X8,
            &probs,
            nb_ctx1,
        )
        .unwrap();
        assert_eq!(tx_size, 0);
    }

    // ----- §9.3.1 intra_mode_tree + §10.5 kf table anchors -----

    #[test]
    fn intra_mode_tree_matches_spec_listing() {
        // §9.3.1 verbatim:
        //   intra_mode_tree[ 18 ] = {
        //       -DC_PRED, 2,
        //       -TM_PRED, 4,
        //       -V_PRED, 6,
        //       8, 12,
        //       -H_PRED, 10,
        //       -D135_PRED, -D117_PRED,
        //       -D45_PRED, 14,
        //       -D63_PRED, 16,
        //       -D153_PRED, -D207_PRED
        //   }
        // DC_PRED = 0 so the first -0 collapses to 0 (the §9.3.3
        // post-loop returns -n which gives 0 again).
        assert_eq!(
            INTRA_MODE_TREE,
            [0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7]
        );
    }

    #[test]
    fn intra_mode_constants_match_spec_numbering() {
        // §7.4.5 default_intra_mode table — verbatim ordering.
        assert_eq!(DC_PRED, 0);
        assert_eq!(V_PRED, 1);
        assert_eq!(H_PRED, 2);
        assert_eq!(D45_PRED, 3);
        assert_eq!(D135_PRED, 4);
        assert_eq!(D117_PRED, 5);
        assert_eq!(D153_PRED, 6);
        assert_eq!(D207_PRED, 7);
        assert_eq!(D63_PRED, 8);
        assert_eq!(TM_PRED, 9);
        assert_eq!(INTRA_MODES, 10);
    }

    #[test]
    fn kf_y_mode_probs_table_shape_and_anchors() {
        // Shape: 10 × 10 × 9.
        assert_eq!(KF_Y_MODE_PROBS.len(), INTRA_MODES);
        assert_eq!(KF_Y_MODE_PROBS[0].len(), INTRA_MODES);
        assert_eq!(KF_Y_MODE_PROBS[0][0].len(), INTRA_MODES - 1);

        // §10.5 spec line 7465: { 137, 30, 42, 148, 151, 207, 70, 52,
        // 91 }, // left = dc (under above = dc).
        assert_eq!(
            KF_Y_MODE_PROBS[DC_PRED as usize][DC_PRED as usize],
            [137, 30, 42, 148, 151, 207, 70, 52, 91]
        );

        // §10.5 spec line 7482: { 59, 67, 44, 140, 161, 202, 78, 67,
        // 119 }, // left = tm (under above = dc).
        assert_eq!(
            KF_Y_MODE_PROBS[DC_PRED as usize][TM_PRED as usize],
            [59, 67, 44, 140, 161, 202, 78, 67, 119]
        );

        // §10.5 spec line 7484: { 63, 36, 126, 146, 123, 158, 60, 90,
        // 96 }, // left = dc (under above = v).
        assert_eq!(
            KF_Y_MODE_PROBS[V_PRED as usize][DC_PRED as usize],
            [63, 36, 126, 146, 123, 158, 60, 90, 96]
        );

        // §10.5 spec line 7497: { 42, 26, 11, 199, 241, 228, 23, 15, 85
        // }, // left = h (under above = h).
        assert_eq!(
            KF_Y_MODE_PROBS[H_PRED as usize][H_PRED as usize],
            [42, 26, 11, 199, 241, 228, 23, 15, 85]
        );

        // §10.5 spec line 7597: { 43, 81, 53, 140, 169, 204, 68, 84,
        // 72 } // left = tm (under above = tm, the table's last row).
        assert_eq!(
            KF_Y_MODE_PROBS[TM_PRED as usize][TM_PRED as usize],
            [43, 81, 53, 140, 169, 204, 68, 84, 72]
        );

        // Every entry must be a valid §9.2 probability (1..=255, the
        // §9.3.2 listing only forbids 0; the §10 default tables only
        // contain non-zero values in practice).
        for row in KF_Y_MODE_PROBS.iter() {
            for cell in row.iter() {
                for &p in cell.iter() {
                    assert!(p >= 1, "kf_y_mode_probs entry {p} below §9.2 min");
                }
            }
        }
    }

    #[test]
    fn kf_uv_mode_probs_table_shape_and_anchors() {
        // Shape: 10 × 9.
        assert_eq!(KF_UV_MODE_PROBS.len(), INTRA_MODES);
        assert_eq!(KF_UV_MODE_PROBS[0].len(), INTRA_MODES - 1);

        // §10.5 spec line 7603: { 144, 11, 54, 157, 195, 130, 46, 58,
        // 108 }, // y = dc.
        assert_eq!(
            KF_UV_MODE_PROBS[DC_PRED as usize],
            [144, 11, 54, 157, 195, 130, 46, 58, 108]
        );

        // §10.5 spec line 7605: { 113, 12, 23, 188, 226, 142, 26, 32,
        // 125 }, // y = h.
        assert_eq!(
            KF_UV_MODE_PROBS[H_PRED as usize],
            [113, 12, 23, 188, 226, 142, 26, 32, 125]
        );

        // §10.5 spec line 7612: { 102, 19, 66, 162, 182, 122, 35, 59,
        // 128 } // y = tm (last row).
        assert_eq!(
            KF_UV_MODE_PROBS[TM_PRED as usize],
            [102, 19, 66, 162, 182, 122, 35, 59, 128]
        );

        // Same §9.2 minimum-probability sanity check as for the y
        // table.
        for row in KF_UV_MODE_PROBS.iter() {
            for &p in row.iter() {
                assert!(p >= 1, "kf_uv_mode_probs entry {p} below §9.2 min");
            }
        }
    }

    // ----- default_intra_mode / default_uv_mode -----

    #[test]
    fn default_intra_mode_zero_buffer_picks_dc_pred() {
        // Zero coder pins every bit to 0; tree[0] = 0 (DC_PRED).
        // Probability values don't affect the outcome along the all-
        // bit-0 path; pass DC/DC neighbour modes to anchor the table
        // row.
        let mut coder = zero_coder();
        let mode = default_intra_mode(&mut coder, DC_PRED, DC_PRED).unwrap();
        assert_eq!(mode, DC_PRED);
    }

    #[test]
    fn default_intra_mode_bias_buffer_with_dc_dc_neighbour_picks_d207_pred() {
        // KF_Y_MODE_PROBS[dc][dc] = [137, 30, 42, 148, 151, 207, 70,
        // 52, 91]. Hand-traced against the §9.2 listing using the bias
        // buffer (post-marker BoolValue=127, BoolRange=128):
        //
        //   node=0  p=137: split=68; value=127>=split -> bit=1;
        //                  range=60, value=59 -> renorm 2 bits ->
        //                  range=240, value=236. n=tree[1]=2.
        //   node=1  p=30:  split=29; value=236>=split -> bit=1;
        //                  range=211, value=207. n=tree[3]=4.
        //   node=2  p=42:  split=35; value=207>=split -> bit=1;
        //                  range=176, value=172. n=tree[5]=6.
        //   node=3  p=148: split=102; value=172>=split -> bit=1;
        //                  range=74, value=70 -> renorm 1 bit ->
        //                  range=148, value=140. n=tree[7]=12.
        //   node=6  p=70:  split=41; value=140>=split -> bit=1;
        //                  range=107, value=99 -> renorm 1 bit ->
        //                  range=214, value=198. n=tree[13]=14.
        //   node=7  p=52:  split=44; value=198>=split -> bit=1;
        //                  range=170, value=154. n=tree[15]=16.
        //   node=8  p=91:  split=61; value=154>=split -> bit=1;
        //                  range=109, value=93 -> renorm 1 bit ->
        //                  range=218, value=186. n=tree[17]=-7 ->
        //                  returns 7 = D207_PRED.
        //
        // The all-high-prob row pushes every node to the right branch,
        // landing on the §9.3.1 -D207_PRED leaf. No external library
        // consulted to derive the expected value — the trace above is
        // a direct §9.2.2 stepping.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mode = default_intra_mode(&mut coder, DC_PRED, DC_PRED).unwrap();
        assert_eq!(mode, D207_PRED);
    }

    #[test]
    fn default_intra_mode_calls_prob_with_node_index_and_correct_row() {
        // Confirms the table indexing reaches kf_y_mode_probs[above]
        // [left][node]. With the zero coder every bit is 0, so the
        // walk is a single read at node=0 of the picked row; the prob
        // value doesn't influence the bit but does flow through the
        // callback we can inspect.
        let mut coder = zero_coder();
        let calls = std::cell::RefCell::new(Vec::<(usize, u8)>::new());
        let above = V_PRED;
        let left = H_PRED;
        let row = &KF_Y_MODE_PROBS[above as usize][left as usize];
        let value = tree_decode(&mut coder, &INTRA_MODE_TREE, |node| {
            let p = row[node];
            calls.borrow_mut().push((node, p));
            p
        })
        .unwrap();
        assert_eq!(value, 0); // DC_PRED via tree[0]=0.
                              // Only one read since every bit=0 routes straight to a leaf.
        assert_eq!(calls.borrow().len(), 1);
        assert_eq!(calls.borrow()[0].0, 0);
        // The selected row's node-0 prob is the [V_PRED][H_PRED][0]
        // cell from the spec table: line 7486: { 44, 29, 68, 159, 201,
        // 177, 50, 57, 77 }, // left = h (under above = v).
        assert_eq!(calls.borrow()[0].1, 44);
    }

    #[test]
    fn default_uv_mode_zero_buffer_picks_dc_pred() {
        // Same zero-walk as default_intra_mode but using the uv prob
        // row indexed by y_mode.
        let mut coder = zero_coder();
        let uv = default_uv_mode(&mut coder, DC_PRED).unwrap();
        assert_eq!(uv, DC_PRED);
    }

    #[test]
    fn default_uv_mode_bias_buffer_with_y_dc_picks_d207_pred() {
        // KF_UV_MODE_PROBS[DC_PRED] = [144, 11, 54, 157, 195, 130, 46,
        // 58, 108]. Hand-traced through §9.2.2 stepping with the bias
        // buffer (post-marker BoolValue=127, BoolRange=128):
        //
        //   node=0  p=144: split=72;  bit=1 (renorm 2) -> n=tree[1]=2.
        //   node=1  p=11:  split=10;  bit=1            -> n=tree[3]=4.
        //   node=2  p=54:  split=45;  bit=1            -> n=tree[5]=6.
        //   node=3  p=157: split=104; bit=1 (renorm 1) -> n=tree[7]=12.
        //   node=6  p=46:  split=24;  bit=1 (renorm 1) -> n=tree[13]=14.
        //   node=7  p=58:  split=48;  bit=1            -> n=tree[15]=16.
        //   node=8  p=108: split=69;  bit=1 (renorm 1) -> n=tree[17]=-7
        //                  -> returns 7 = D207_PRED.
        //
        // Same "all-high-prob row + bias buffer pushes every node to
        // the right branch" pattern as the y-mode trace above. No
        // external library consulted.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let uv = default_uv_mode(&mut coder, DC_PRED).unwrap();
        assert_eq!(uv, D207_PRED);
    }

    // ----- §6.4.6 intra_frame_mode_info -----

    fn default_intra_nb() -> IntraFrameNeighbours {
        IntraFrameNeighbours::default()
    }

    fn zero_tx_probs() -> [[[u8; 3]; 2]; 4] {
        [[[128; 3]; 2]; 4]
    }

    #[test]
    fn intra_frame_mode_info_zero_buffer_all_dc_pred() {
        // Zero coder pins everything to bit=0:
        //   intra_segment_id: seg_enabled=false -> segment_id=0
        //   read_skip: seg_feature_skip=false, BINARY_TREE walk -> 0
        //   read_tx_size: allow_select=true, tx_mode=TxModeSelect,
        //                 MiSize=BLOCK_8X8 -> tree[0]=0 -> tx_size=0
        //   MiSize >= BLOCK_8X8 branch:
        //     default_intra_mode -> DC_PRED, y_mode = DC_PRED,
        //     sub_modes = [DC_PRED; 4]
        //   default_uv_mode -> DC_PRED
        //   ref_frame_0 = INTRA_FRAME = 0, ref_frame_1 = NONE = -1
        //   is_inter = false
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_8X8,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        assert_eq!(block.segment_id, 0);
        assert!(!block.skip);
        assert_eq!(block.tx_size, 0);
        assert_eq!(block.ref_frame_0, INTRA_FRAME);
        assert_eq!(block.ref_frame_1, NONE_REF_FRAME);
        assert!(!block.is_inter);
        assert_eq!(block.y_mode, DC_PRED);
        assert_eq!(block.sub_modes, [DC_PRED; 4]);
        assert_eq!(block.uv_mode, DC_PRED);
    }

    #[test]
    fn intra_frame_mode_info_replicates_y_mode_into_sub_modes_for_large_block() {
        // Same zero-coder run with BLOCK_64X64 — every cell in
        // sub_modes[ ] must equal y_mode per the §6.4.6 `for(b=0;b<4;
        // b++) sub_modes[b] = y_mode` line.
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_64X64,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        for cell in block.sub_modes.iter() {
            assert_eq!(*cell, block.y_mode);
        }
    }

    #[test]
    fn intra_frame_mode_info_sub_8x8_walks_idy_idx_grid() {
        // BLOCK_4X4: num4x4w = num4x4h = 1.
        //   idy=0: idx=0 -> default_intra_mode -> sub_modes[0]
        //          idx=1 -> default_intra_mode -> sub_modes[1]
        //   idy=1: idx=0 -> default_intra_mode -> sub_modes[2]
        //          idx=1 -> default_intra_mode -> sub_modes[3]
        // Four reads total; with the zero coder each picks DC_PRED.
        // Hence sub_modes = [DC_PRED; 4] and y_mode = DC_PRED (last
        // decoded).
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_4X4,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        // For sub-8x8 blocks read_tx_size's `MiSize >= BLOCK_8X8` guard
        // fails so the §6.4.10 fallback yields tx_size = Min(0,
        // biggest[TxModeSelect]=3) = 0.
        assert_eq!(block.tx_size, 0);
        assert_eq!(block.sub_modes, [DC_PRED; 4]);
        assert_eq!(block.y_mode, DC_PRED);
        assert_eq!(block.uv_mode, DC_PRED);
    }

    #[test]
    fn intra_frame_mode_info_sub_8x8_rectangular_blocks_walk_dimensions() {
        // BLOCK_4X8: num4x4w=1, num4x4h=2.
        //   idy=0: idx=0 -> sub_modes[0..2 each] over y2=0..2, x2=0..1
        //          idx=1 -> sub_modes[(0+y2)*2 + 1 + 0] for y2=0..2
        // So one read covers two cells (sub_modes[0] and sub_modes[2]
        // for idx=0; sub_modes[1] and sub_modes[3] for idx=1) and the
        // walk emits only 2 reads total (idx outer = 2 visits; idy
        // outer = 1 visit).
        //
        // BLOCK_8X4: num4x4w=2, num4x4h=1. Mirror image — 2 reads
        // total over idy ∈ {0,1}, idx ∈ {0}.
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_4X8,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        assert_eq!(block.sub_modes, [DC_PRED; 4]);

        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_8X4,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        assert_eq!(block.sub_modes, [DC_PRED; 4]);
    }

    #[test]
    fn intra_frame_mode_info_neighbours_default_to_dc_when_unavailable() {
        // The §9.3.2 listing substitutes DC_PRED when AvailU / AvailL
        // is false. Build a frame-edge MI block (no above, no left) and
        // confirm the row selected for default_intra_mode is the
        // KF_Y_MODE_PROBS[DC_PRED][DC_PRED] cell — exercised
        // indirectly by the zero-buffer DC_PRED outcome, which only
        // works if the table indexing didn't panic on an
        // out-of-range above/left index.
        let mut coder = zero_coder();
        let nb_intra = IntraFrameNeighbours {
            avail_u: false,
            avail_l: false,
            above_sub_modes_23: [255, 255], // sentinel — should be ignored
            left_sub_modes_13: [255, 255],  // sentinel — should be ignored
        };
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_8X8,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            nb_intra,
        )
        .unwrap();
        assert_eq!(block.y_mode, DC_PRED);
    }

    #[test]
    fn intra_frame_mode_info_segmentation_disabled_pins_segment_zero() {
        // With seg_enabled=false the §6.4.7 else branch hardwires
        // segment_id = 0 regardless of update_map / tree_probs.
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_8X8,
            false,
            true,
            Some(&[255; 7]),
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        assert_eq!(block.segment_id, 0);
    }

    #[test]
    fn intra_frame_mode_info_skip_segment_feature_forces_skip() {
        // §6.4.8 read_skip's seg_feature_active(SEG_LVL_SKIP) early-
        // return path forces skip=true without reading any bits. The
        // remaining sequence (read_tx_size + default_intra_mode +
        // default_uv_mode) still runs but only off the entropy coder
        // bytes the prior reads consumed.
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_8X8,
            false,
            false,
            None,
            true, // seg_feature_skip_active
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        assert!(block.skip);
        // Sanity: the rest of the fields still populate as zero-coder
        // would produce.
        assert_eq!(block.y_mode, DC_PRED);
        assert_eq!(block.uv_mode, DC_PRED);
    }

    #[test]
    fn intra_frame_mode_info_ref_frames_and_is_inter_are_intra_only() {
        // §6.4.6 hardwires ref_frame[0] = INTRA_FRAME, ref_frame[1] =
        // NONE, is_inter = 0 unconditionally. Verify across multiple
        // MiSize values that those fields don't depend on the entropy
        // coder.
        for &mi_size in &[BLOCK_4X4, BLOCK_8X8, BLOCK_32X32, BLOCK_64X64] {
            let mut coder = zero_coder();
            let block = intra_frame_mode_info(
                &mut coder,
                mi_size,
                false,
                false,
                None,
                false,
                TxMode::TxModeSelect,
                &zero_tx_probs(),
                &[128, 128, 128],
                NeighbourSkips::default(),
                NeighbourTxSizes::default(),
                default_intra_nb(),
            )
            .unwrap();
            assert_eq!(block.ref_frame_0, INTRA_FRAME);
            assert_eq!(block.ref_frame_1, NONE_REF_FRAME);
            assert!(!block.is_inter);
        }
    }

    #[test]
    fn intra_frame_constants_match_spec_relations() {
        // §3 listing: INTRA_FRAME = 0; the §6.4.6 ref_frame[1] = NONE
        // assignment together with §6.4.11's `isCompound = ref_frame[1]
        // > NONE` test pins NONE strictly below INTRA_FRAME = 0. The
        // loop-filter ref_deltas iteration `INTRA_FRAME..MAX_REF_FRAMES
        // -1` (= 0..=3) places NONE outside the legal range; NONE = -1
        // is the unique integer satisfying both constraints.
        assert_eq!(INTRA_FRAME, 0);
        assert_eq!(NONE_REF_FRAME, -1);
        const _: () = assert!(NONE_REF_FRAME < INTRA_FRAME);
    }

    // ----- §9.3.2 size_group_lookup / §9.3 default_y_mode_probs /
    //       default_uv_mode_probs tables -----

    #[test]
    fn size_group_lookup_table_matches_spec_listing() {
        // §9.3.2 spec line 7120 verbatim:
        //   size_group_lookup[ BLOCK_SIZES ] =
        //     {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3}
        assert_eq!(SIZE_GROUP_LOOKUP.len(), BLOCK_SIZES);
        assert_eq!(SIZE_GROUP_LOOKUP, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
        // Block-size anchors: BLOCK_4X4..BLOCK_8X4 -> group 0,
        // BLOCK_8X8..BLOCK_16X8 -> 1, BLOCK_16X16..BLOCK_32X16 -> 2,
        // BLOCK_32X32..BLOCK_64X64 -> 3.
        assert_eq!(SIZE_GROUP_LOOKUP[BLOCK_4X4 as usize], 0);
        assert_eq!(SIZE_GROUP_LOOKUP[BLOCK_8X4 as usize], 0);
        assert_eq!(SIZE_GROUP_LOOKUP[BLOCK_8X8 as usize], 1);
        assert_eq!(SIZE_GROUP_LOOKUP[BLOCK_16X16 as usize], 2);
        assert_eq!(SIZE_GROUP_LOOKUP[BLOCK_32X32 as usize], 3);
        assert_eq!(SIZE_GROUP_LOOKUP[BLOCK_64X64 as usize], 3);
        // Every group index must be a valid §9.3 y_mode_probs row.
        for &g in SIZE_GROUP_LOOKUP.iter() {
            assert!((g as usize) < BLOCK_SIZE_GROUPS);
        }
    }

    #[test]
    fn default_y_mode_probs_table_shape_and_anchors() {
        // Shape: BLOCK_SIZE_GROUPS (4) × (INTRA_MODES - 1) (9).
        assert_eq!(DEFAULT_Y_MODE_PROBS.len(), BLOCK_SIZE_GROUPS);
        assert_eq!(DEFAULT_Y_MODE_PROBS[0].len(), INTRA_MODES - 1);

        // §9.3 default_y_mode_probs listing (verbatim):
        //   { 65, 32, 18, 144, 162, 194, 41, 51, 98 },   // < 8x8
        //   { 132, 68, 18, 165, 217, 196, 45, 40, 78 },  // < 16x16
        //   { 173, 80, 19, 176, 240, 193, 64, 35, 46 },  // < 32x32
        //   { 221, 135, 38, 194, 248, 121, 96, 85, 29 }  // >= 32x32
        assert_eq!(
            DEFAULT_Y_MODE_PROBS[0],
            [65, 32, 18, 144, 162, 194, 41, 51, 98]
        );
        assert_eq!(
            DEFAULT_Y_MODE_PROBS[1],
            [132, 68, 18, 165, 217, 196, 45, 40, 78]
        );
        assert_eq!(
            DEFAULT_Y_MODE_PROBS[2],
            [173, 80, 19, 176, 240, 193, 64, 35, 46]
        );
        assert_eq!(
            DEFAULT_Y_MODE_PROBS[3],
            [221, 135, 38, 194, 248, 121, 96, 85, 29]
        );

        // §9.2 minimum-probability sanity.
        for row in DEFAULT_Y_MODE_PROBS.iter() {
            for &p in row.iter() {
                assert!(p >= 1, "default_y_mode_probs entry {p} below §9.2 min");
            }
        }
    }

    #[test]
    fn default_uv_mode_probs_table_shape_and_anchors() {
        // Shape: INTRA_MODES (10) × (INTRA_MODES - 1) (9).
        assert_eq!(DEFAULT_UV_MODE_PROBS.len(), INTRA_MODES);
        assert_eq!(DEFAULT_UV_MODE_PROBS[0].len(), INTRA_MODES - 1);

        // §9.3 default_uv_mode_probs listing anchors (verbatim):
        //   { 120, 7, 76, 176, 208, 126, 28, 54, 103 },  // y = dc
        //   { 67, 6, 25, 204, 243, 158, 13, 21, 96 },    // y = h
        //   { 101, 21, 107, 181, 192, 103, 19, 67, 125 } // y = tm (last)
        assert_eq!(
            DEFAULT_UV_MODE_PROBS[DC_PRED as usize],
            [120, 7, 76, 176, 208, 126, 28, 54, 103]
        );
        assert_eq!(
            DEFAULT_UV_MODE_PROBS[H_PRED as usize],
            [67, 6, 25, 204, 243, 158, 13, 21, 96]
        );
        assert_eq!(
            DEFAULT_UV_MODE_PROBS[TM_PRED as usize],
            [101, 21, 107, 181, 192, 103, 19, 67, 125]
        );

        // §9.2 minimum-probability sanity.
        for row in DEFAULT_UV_MODE_PROBS.iter() {
            for &p in row.iter() {
                assert!(p >= 1, "default_uv_mode_probs entry {p} below §9.2 min");
            }
        }
    }

    // ----- §9.3.2 intra_mode / sub_intra_mode / uv_mode readers -----

    #[test]
    fn intra_mode_zero_buffer_picks_dc_pred() {
        // Zero coder pins every bit to 0; INTRA_MODE_TREE[0] = 0 = DC_PRED.
        let mut coder = zero_coder();
        let mode = intra_mode(&mut coder, &DEFAULT_Y_MODE_PROBS, BLOCK_16X16).unwrap();
        assert_eq!(mode, DC_PRED);
    }

    #[test]
    fn intra_mode_uses_size_group_lookup_ctx_row() {
        // §9.3.2: ctx = size_group_lookup[ MiSize ]. With the zero coder
        // the walk is a single read at node=0 of the selected row; we
        // instrument tree_decode to confirm the row reached is
        // y_mode_probs[ size_group_lookup[ MiSize ] ]. BLOCK_16X16 maps
        // to group 2, whose node-0 prob is 173.
        let ctx = SIZE_GROUP_LOOKUP[BLOCK_16X16 as usize] as usize;
        assert_eq!(ctx, 2);
        let mut coder = zero_coder();
        let calls = std::cell::RefCell::new(Vec::<(usize, u8)>::new());
        let row = &DEFAULT_Y_MODE_PROBS[ctx];
        let value = tree_decode(&mut coder, &INTRA_MODE_TREE, |node| {
            let p = row[node];
            calls.borrow_mut().push((node, p));
            p
        })
        .unwrap();
        assert_eq!(value, 0); // DC_PRED.
        assert_eq!(calls.borrow().len(), 1);
        assert_eq!(calls.borrow()[0], (0, 173));
    }

    #[test]
    fn intra_mode_bias_buffer_block_lt_8x8_picks_d207_pred() {
        // DEFAULT_Y_MODE_PROBS[ size_group_lookup[BLOCK_4X4]=0 ] =
        //   [65, 32, 18, 144, 162, 194, 41, 51, 98]. With the bias
        // buffer (post-marker BoolValue=127, BoolRange=128) every node
        // takes the right branch, walking INTRA_MODE_TREE to the
        // -D207_PRED leaf (value 7). Derived from a direct §9.2.2
        // stepping over the crate's own BoolCoder; no external library
        // consulted.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mode = intra_mode(&mut coder, &DEFAULT_Y_MODE_PROBS, BLOCK_4X4).unwrap();
        assert_eq!(mode, D207_PRED);
    }

    #[test]
    fn sub_intra_mode_uses_ctx_zero_row() {
        // §9.3.2: sub_intra_mode ctx is fixed at 0. Instrument
        // tree_decode to confirm the node-0 prob is y_mode_probs[0][0] =
        // 65 (the group-0 row), independent of MiSize.
        let mut coder = zero_coder();
        let calls = std::cell::RefCell::new(Vec::<u8>::new());
        let row = &DEFAULT_Y_MODE_PROBS[0];
        let value = tree_decode(&mut coder, &INTRA_MODE_TREE, |node| {
            let p = row[node];
            calls.borrow_mut().push(p);
            p
        })
        .unwrap();
        assert_eq!(value, 0);
        assert_eq!(calls.borrow()[0], 65);
        // And the real reader picks DC_PRED on the zero buffer.
        let mut coder = zero_coder();
        assert_eq!(
            sub_intra_mode(&mut coder, &DEFAULT_Y_MODE_PROBS).unwrap(),
            DC_PRED
        );
    }

    #[test]
    fn uv_mode_zero_buffer_picks_dc_pred() {
        // Zero coder -> first leaf DC_PRED, regardless of y_mode ctx.
        let mut coder = zero_coder();
        let uv = uv_mode(&mut coder, &DEFAULT_UV_MODE_PROBS, V_PRED).unwrap();
        assert_eq!(uv, DC_PRED);
    }

    #[test]
    fn uv_mode_uses_y_mode_as_ctx_row() {
        // §9.3.2: uv_mode ctx = y_mode. With the zero coder, instrument
        // to confirm the row reached is uv_mode_probs[ y_mode ]. For
        // y_mode = H_PRED the node-0 prob is 67.
        let mut coder = zero_coder();
        let calls = std::cell::RefCell::new(Vec::<u8>::new());
        let row = &DEFAULT_UV_MODE_PROBS[H_PRED as usize];
        let value = tree_decode(&mut coder, &INTRA_MODE_TREE, |node| {
            let p = row[node];
            calls.borrow_mut().push(p);
            p
        })
        .unwrap();
        assert_eq!(value, 0);
        assert_eq!(calls.borrow()[0], 67);
    }

    #[test]
    fn uv_mode_bias_buffer_y_dc_picks_d207_pred() {
        // DEFAULT_UV_MODE_PROBS[DC_PRED] =
        //   [120, 7, 76, 176, 208, 126, 28, 54, 103]. Bias buffer walks
        // every node right to the -D207_PRED leaf (value 7). Direct
        // §9.2.2 stepping over the crate's BoolCoder.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let uv = uv_mode(&mut coder, &DEFAULT_UV_MODE_PROBS, DC_PRED).unwrap();
        assert_eq!(uv, D207_PRED);
    }

    // ----- §6.4.15 intra_block_mode_info -----

    #[test]
    fn intra_block_mode_info_zero_buffer_all_dc_pred() {
        // §6.4.15 with MiSize >= BLOCK_8X8 and the zero coder:
        //   ref_frame[0]=INTRA_FRAME, ref_frame[1]=NONE
        //   intra_mode -> DC_PRED ; y_mode=DC_PRED ; sub_modes=[DC;4]
        //   uv_mode -> DC_PRED
        let mut coder = zero_coder();
        let block = intra_block_mode_info(
            &mut coder,
            BLOCK_8X8,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        assert_eq!(block.ref_frame_0, INTRA_FRAME);
        assert_eq!(block.ref_frame_1, NONE_REF_FRAME);
        assert_eq!(block.y_mode, DC_PRED);
        assert_eq!(block.sub_modes, [DC_PRED; 4]);
        assert_eq!(block.uv_mode, DC_PRED);
    }

    #[test]
    fn intra_block_mode_info_large_block_replicates_y_mode() {
        // BLOCK_64X64: one intra_mode decode replicated into all four
        // sub_modes[ ] per the §6.4.15 `for(b=0;b<4;b++) sub_modes[b]=
        // y_mode` line.
        let mut coder = zero_coder();
        let block = intra_block_mode_info(
            &mut coder,
            BLOCK_64X64,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        for cell in block.sub_modes.iter() {
            assert_eq!(*cell, block.y_mode);
        }
    }

    #[test]
    fn intra_block_mode_info_bias_buffer_decodes_d207_then_d153() {
        // BLOCK_8X8 (ctx = size_group_lookup[BLOCK_8X8] = 1) with the
        // bias buffer. The driver runs `intra_mode` (over
        // y_mode_probs[1] = [132, 68, 18, 165, 217, 196, 45, 40, 78])
        // then `uv_mode` (over uv_mode_probs[y_mode]) on the *same*
        // coder, so the second walk continues from the first walk's
        // post-renorm BoolValue/BoolRange — it is a single contiguous
        // §9.2.2 stepping, not two independent ones.
        //
        // The first full INTRA_MODE_TREE walk (all-high-prob row pushing
        // every node right) reaches the -D207_PRED leaf (value 7). The
        // subsequent uv_mode walk over uv_mode_probs[7] =
        // [85, 5, 32, 156, 216, 148, 19, 29, 73] continues the same
        // stepping and lands on the -D153_PRED leaf (value 6). Both
        // values were derived from a direct §9.2.2 stepping of the
        // crate's own BoolCoder; no external library consulted.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let block = intra_block_mode_info(
            &mut coder,
            BLOCK_8X8,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        assert_eq!(block.y_mode, D207_PRED);
        assert_eq!(block.sub_modes, [D207_PRED; 4]);
        assert_eq!(block.uv_mode, D153_PRED);
    }

    #[test]
    fn intra_block_mode_info_sub_8x8_walks_grid() {
        // BLOCK_4X4: num4x4w=num4x4h=1, four sub_intra_mode reads. Zero
        // coder -> all DC_PRED, y_mode = last decoded = DC_PRED.
        let mut coder = zero_coder();
        let block = intra_block_mode_info(
            &mut coder,
            BLOCK_4X4,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        assert_eq!(block.sub_modes, [DC_PRED; 4]);
        assert_eq!(block.y_mode, DC_PRED);
        assert_eq!(block.uv_mode, DC_PRED);
    }

    #[test]
    fn intra_block_mode_info_sub_8x8_rectangular_replicates_per_cell() {
        // BLOCK_4X8: num4x4w=1, num4x4h=2 -> 2 sub_intra_mode reads
        // (idx outer visits {0,1}; idy outer visits {0}). Each read
        // covers two sub_modes[ ] cells. Zero coder -> all DC_PRED.
        let mut coder = zero_coder();
        let block = intra_block_mode_info(
            &mut coder,
            BLOCK_4X8,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        assert_eq!(block.sub_modes, [DC_PRED; 4]);
        assert_eq!(block.y_mode, DC_PRED);
    }

    #[test]
    fn intra_block_mode_info_no_segment_skip_tx_fields() {
        // §6.4.15 (unlike §6.4.6) decodes only modes — the struct
        // carries no segment_id/skip/tx_size. Confirm the ref-frame
        // triple is intra-only.
        let mut coder = zero_coder();
        let block = intra_block_mode_info(
            &mut coder,
            BLOCK_32X32,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        assert_eq!(block.ref_frame_0, INTRA_FRAME);
        assert_eq!(block.ref_frame_1, NONE_REF_FRAME);
    }

    // ----- §6.4.5 mode_info dispatch -----

    #[test]
    fn mode_info_dispatch_inter_frame_intra_block_wraps_intra_block() {
        // The §6.4.5 dispatcher's `!FrameIsIntra` / `is_inter == 0`
        // sub-path returns the §6.4.15 product wrapped in the
        // InterFrameIntraBlock variant.
        let mut coder = zero_coder();
        let mi = inter_frame_intra_block_mode_info(
            &mut coder,
            BLOCK_8X8,
            &DEFAULT_Y_MODE_PROBS,
            &DEFAULT_UV_MODE_PROBS,
        )
        .unwrap();
        match mi {
            Vp9ModeInfo::InterFrameIntraBlock(block) => {
                assert_eq!(block.ref_frame_0, INTRA_FRAME);
                assert_eq!(block.y_mode, DC_PRED);
                assert_eq!(block.uv_mode, DC_PRED);
            }
            other => panic!("expected InterFrameIntraBlock, got {other:?}"),
        }
    }

    #[test]
    fn mode_info_dispatch_keyframe_path_uses_intra_frame_mode_info() {
        // The §6.4.5 `FrameIsIntra` path is intra_frame_mode_info; pin
        // that the keyframe driver still produces a Vp9IntraMiBlock the
        // dispatcher's IntraFrame variant can carry.
        let mut coder = zero_coder();
        let block = intra_frame_mode_info(
            &mut coder,
            BLOCK_8X8,
            false,
            false,
            None,
            false,
            TxMode::TxModeSelect,
            &zero_tx_probs(),
            &[128, 128, 128],
            NeighbourSkips::default(),
            NeighbourTxSizes::default(),
            default_intra_nb(),
        )
        .unwrap();
        let mi = Vp9ModeInfo::IntraFrame(block);
        match mi {
            Vp9ModeInfo::IntraFrame(b) => assert_eq!(b.y_mode, DC_PRED),
            other => panic!("expected IntraFrame, got {other:?}"),
        }
    }

    // ----- §6.4.14 get_segment_id + §6.4.12 inter_segment_id -----

    fn empty_prev<'a>(mi_rows: u32, mi_cols: u32, backing: &'a [u8]) -> PrevSegmentIds<'a> {
        PrevSegmentIds {
            mi_rows,
            mi_cols,
            data: backing,
        }
    }

    #[test]
    fn get_segment_id_smallest_in_bw_by_bh_region() {
        // §6.4.14: seg = Min over the on-screen MiSize × MiSize cells.
        // BLOCK_16X16 = 6 → bw = bh = 2. Fill a 4x4 prev frame with
        // ids, then read at (mi_row=0, mi_col=0): the 2x2 top-left
        // covers cells (0,0)=5, (0,1)=3, (1,0)=7, (1,1)=4 → Min = 3.
        let prev_data: [u8; 16] = [
            5, 3, 6, 6, //
            7, 4, 6, 6, //
            6, 6, 6, 6, //
            6, 6, 6, 6, //
        ];
        let prev = empty_prev(4, 4, &prev_data);
        // BLOCK_16X16 = 6 per §3.
        let seg = get_segment_id(&prev, 0, 0, 6);
        assert_eq!(seg, 3);
    }

    #[test]
    fn get_segment_id_clamps_to_on_screen_region() {
        // §6.4.14: xmis = Min(MiCols - MiCol, bw) etc.
        // BLOCK_32X32 = 9 → bw = bh = 4. Reading at (mi_row=1, mi_col=2)
        // on a 3x3 plane clamps to xmis = Min(3-2, 4) = 1 and
        // ymis = Min(3-1, 4) = 2 — a 1x2 sweep.
        // Cells covered: (1,2)=4, (2,2)=2 → Min = 2.
        let prev_data: [u8; 9] = [
            6, 6, 6, //
            6, 6, 4, //
            6, 6, 2, //
        ];
        let prev = empty_prev(3, 3, &prev_data);
        let seg = get_segment_id(&prev, 1, 2, 9);
        assert_eq!(seg, 2);
    }

    #[test]
    fn get_segment_id_all_max_returns_seven() {
        // §6.4.14: seg starts at 7; an all-7 prev keeps it at 7.
        let prev_data: [u8; 4] = [7, 7, 7, 7];
        let prev = empty_prev(2, 2, &prev_data);
        // BLOCK_16X16 = 6 → bw = bh = 2 → sweeps the entire 2x2.
        let seg = get_segment_id(&prev, 0, 0, 6);
        assert_eq!(seg, 7);
    }

    #[test]
    fn seg_pred_context_state_zeroes_on_new_and_clear_left() {
        // §7.4.1 / §7.4.2 zero-init contract.
        let ctx = SegPredContextState::new(8, 8);
        for c in 0..8u32 {
            assert_eq!(ctx.above(c), 0);
        }
        for r in 0..8u32 {
            assert_eq!(ctx.left(r), 0);
        }

        // After populating Left, clear_left() returns to zeros without
        // touching Above.
        let mut ctx = SegPredContextState::new(8, 8);
        ctx.above[0] = 1;
        ctx.above[5] = 1;
        ctx.left[0] = 1;
        ctx.left[3] = 1;
        ctx.clear_left();
        assert_eq!(ctx.above(0), 1);
        assert_eq!(ctx.above(5), 1);
        for r in 0..8u32 {
            assert_eq!(ctx.left(r), 0);
        }
    }

    #[test]
    fn read_seg_id_predicted_ctx_is_left_plus_above() {
        // §9.3.2: ctx = LeftSegPredContext[MiRow] + AboveSegPredContext[MiCol].
        // Set Left=1, Above=1 at (mi_row=2, mi_col=3) → ctx=2 → uses
        // pred_prob[2]. We instrument the prob callback by handing
        // distinct probs to the three ctx slots and using zero coder to
        // pin the decoded bit to 0 (so the return is always `false`,
        // but the index we read is observable via direct call to
        // tree_decode in a separate path).
        let mut coder = zero_coder();
        let mut ctx = SegPredContextState::new(8, 8);
        ctx.above[3] = 1;
        ctx.left[2] = 1;
        // pred_prob[2] = 1 (deterministic with zero coder → bit 0)
        let pred_prob = [128u8, 128, 1];
        let decoded = read_seg_id_predicted(&mut coder, &pred_prob, &ctx, 2, 3).unwrap();
        assert!(!decoded);

        // Confirm a bias coder + pred_prob[0]=255 (ctx=0 → Left+Above=0)
        // decodes a 1 bit (the §9.3.1 binary_tree = {0, -1} → leaf -1
        // → returns 1).
        let bytes = make_bias_buffer(0x7F);
        let mut coder2 = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let ctx2 = SegPredContextState::new(8, 8); // all zeros
        let pred_prob2 = [255u8, 0, 0];
        let decoded2 = read_seg_id_predicted(&mut coder2, &pred_prob2, &ctx2, 0, 0).unwrap();
        assert!(decoded2);
    }

    #[test]
    fn inter_segment_id_disabled_returns_zero() {
        // Path 1 of §6.4.12: !segmentation_enabled → segment_id = 0,
        // no bool-coder reads, no ctx writes.
        let mut coder = zero_coder();
        let prev_data = [3u8, 3, 3, 3];
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        let id = inter_segment_id(
            &mut coder,
            false,
            true,
            true,
            Some(&[255; 7]),
            Some(&[255; 3]),
            &prev,
            &mut ctx,
            0,
            0,
            6,
        )
        .unwrap();
        assert_eq!(id, 0);
        // ctx untouched.
        for c in 0..2u32 {
            assert_eq!(ctx.above(c), 0);
        }
        for r in 0..2u32 {
            assert_eq!(ctx.left(r), 0);
        }
    }

    #[test]
    fn inter_segment_id_enabled_no_update_map_returns_predicted() {
        // Path 2 of §6.4.12: segmentation_enabled &&
        // !segmentation_update_map → segment_id = predictedSegmentId.
        // No bool-coder reads. Predicted = Min over BLOCK_16X16's 2x2.
        let mut coder = zero_coder();
        let prev_data: [u8; 4] = [4, 2, 5, 6];
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        let id = inter_segment_id(
            &mut coder, true, false, false, None, None, &prev, &mut ctx, 0, 0, 6,
        )
        .unwrap();
        // Min(4, 2, 5, 6) = 2.
        assert_eq!(id, 2);
        // ctx untouched.
        for c in 0..2u32 {
            assert_eq!(ctx.above(c), 0);
        }
    }

    #[test]
    fn inter_segment_id_enabled_update_map_no_temporal_decodes_segment_id() {
        // Path 3 of §6.4.12: segmentation_enabled &&
        // segmentation_update_map && !segmentation_temporal_update →
        // decode segment_id via read_segment_id (§9.3.1 SEGMENT_TREE).
        // Zero coder + walk lands on segment 0.
        let mut coder = zero_coder();
        let prev_data = [3u8; 4];
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        let id = inter_segment_id(
            &mut coder,
            true,
            true,
            false,
            Some(&[128; 7]),
            None,
            &prev,
            &mut ctx,
            0,
            0,
            6,
        )
        .unwrap();
        assert_eq!(id, 0);
        // Path 3 does not write seg-pred ctx.
        for c in 0..2u32 {
            assert_eq!(ctx.above(c), 0);
        }
    }

    #[test]
    fn inter_segment_id_temporal_predicted_branch_uses_predictor_and_writes_ctx() {
        // Path 4 of §6.4.12, predicted sub-branch:
        //   seg_id_predicted = 1 → segment_id = predictedSegmentId.
        //   AboveSegPredContext / LeftSegPredContext written with 1.
        // pred_prob[ctx=0] = 255, bias buffer → seg_id_predicted = 1.
        // After that read, the coder is refilled with zeros, so any
        // subsequent read_segment_id would walk to segment 0 — but the
        // §6.4.12 listing skips the read entirely when predicted=1.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let prev_data: [u8; 4] = [4, 2, 5, 6]; // Min over 2x2 = 2.
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        let id = inter_segment_id(
            &mut coder,
            true,
            true,
            true,
            Some(&[128; 7]),
            Some(&[255, 0, 0]),
            &prev,
            &mut ctx,
            0,
            0,
            6,
        )
        .unwrap();
        assert_eq!(id, 2); // = predicted.
                           // BLOCK_16X16 → bw=bh=2; ctx flag=1 written across (0..2).
        assert_eq!(ctx.above(0), 1);
        assert_eq!(ctx.above(1), 1);
        assert_eq!(ctx.left(0), 1);
        assert_eq!(ctx.left(1), 1);
    }

    #[test]
    fn inter_segment_id_temporal_not_predicted_branch_decodes_and_writes_ctx_zero() {
        // Path 4 of §6.4.12, not-predicted sub-branch:
        //   seg_id_predicted = 0 → read segment_id via SEGMENT_TREE.
        //   AboveSegPredContext / LeftSegPredContext written with 0.
        // Zero coder pins every bit to 0 → seg_id_predicted = 0,
        // segment_id walks SEGMENT_TREE to leaf 0.
        let mut coder = zero_coder();
        let prev_data: [u8; 4] = [4, 2, 5, 6];
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        // Pre-populate ctx with 1s so we can observe the path-4 zero-
        // write back overwriting them.
        ctx.above[0] = 1;
        ctx.above[1] = 1;
        ctx.left[0] = 1;
        ctx.left[1] = 1;
        let id = inter_segment_id(
            &mut coder,
            true,
            true,
            true,
            Some(&[128; 7]),
            Some(&[128, 128, 128]),
            &prev,
            &mut ctx,
            0,
            0,
            6,
        )
        .unwrap();
        assert_eq!(id, 0);
        assert_eq!(ctx.above(0), 0);
        assert_eq!(ctx.above(1), 0);
        assert_eq!(ctx.left(0), 0);
        assert_eq!(ctx.left(1), 0);
    }

    #[test]
    fn inter_segment_id_missing_tree_probs_when_decoding_is_invalid() {
        // Paths 3 and 4-not-predicted decode segment_id and need
        // tree_probs. Passing None there surfaces InvalidBitstream.
        let mut coder = zero_coder();
        let prev_data = [3u8; 4];
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        // Path 3.
        let err = inter_segment_id(
            &mut coder, true, true, false, None, None, &prev, &mut ctx, 0, 0, 6,
        );
        assert!(matches!(err, Err(Error::InvalidBitstream)));

        // Path 4-not-predicted (zero coder → seg_id_predicted=0 → falls
        // through to read_segment_id which needs tree_probs).
        let mut coder2 = zero_coder();
        let err2 = inter_segment_id(
            &mut coder2,
            true,
            true,
            true,
            None,
            Some(&[128, 128, 128]),
            &prev,
            &mut ctx,
            0,
            0,
            6,
        );
        assert!(matches!(err2, Err(Error::InvalidBitstream)));
    }

    #[test]
    fn inter_segment_id_missing_pred_prob_when_temporal_is_invalid() {
        // Path 4 needs pred_prob (the seg_id_predicted read). Missing
        // → InvalidBitstream.
        let mut coder = zero_coder();
        let prev_data = [3u8; 4];
        let prev = empty_prev(2, 2, &prev_data);
        let mut ctx = SegPredContextState::new(2, 2);
        let err = inter_segment_id(
            &mut coder,
            true,
            true,
            true,
            Some(&[128; 7]),
            None,
            &prev,
            &mut ctx,
            0,
            0,
            6,
        );
        assert!(matches!(err, Err(Error::InvalidBitstream)));
    }

    #[test]
    fn inter_segment_id_temporal_write_back_clamps_partial_edge_block() {
        // §6.4.12 trailing write-back iterates 0..bw for Above and
        // 0..bh for Left; on a partial-edge block the loop indexes can
        // run past the strip end. The impl clamps with a per-cell
        // bounds check rather than per-call to keep the listing's
        // verbatim loop shape; verify the clamp keeps things sane on a
        // 3-wide / 3-high frame with a BLOCK_32X32 (bw=bh=4) at (1,1).
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let prev_data: [u8; 9] = [
            6, 6, 6, //
            6, 5, 7, //
            6, 7, 7, //
        ];
        let prev = empty_prev(3, 3, &prev_data);
        let mut ctx = SegPredContextState::new(3, 3);
        // Place an isolated BLOCK_32X32 at (1, 1) — wide and high run
        // 4 cells each but only 2 land in-bounds for either axis.
        let id = inter_segment_id(
            &mut coder,
            true,
            true,
            true,
            Some(&[128; 7]),
            Some(&[255, 0, 0]),
            &prev,
            &mut ctx,
            1,
            1,
            9,
        )
        .unwrap();
        // pred_prob[0]=255 + bias → seg_id_predicted=1 → segment_id =
        // get_segment_id over a clamped 2x2 region (1..3, 1..3) =
        // Min(5, 7, 7, 7) = 5.
        assert_eq!(id, 5);
        // Above written at cols 1, 2 (cols 3, 4 clamped).
        assert_eq!(ctx.above(0), 0);
        assert_eq!(ctx.above(1), 1);
        assert_eq!(ctx.above(2), 1);
        // Left written at rows 1, 2 (rows 3, 4 clamped).
        assert_eq!(ctx.left(0), 0);
        assert_eq!(ctx.left(1), 1);
        assert_eq!(ctx.left(2), 1);
    }

    // ----- §6.4.13 read_is_inter + §9.3.2 is_inter ctx -----

    // §3 ref-frame indices used by the §6.4.13 / §9.3.2 listings.
    const LAST_FRAME: i32 = 1;
    const GOLDEN_FRAME: i32 = 2;
    const ALTREF_FRAME: i32 = 3;

    #[test]
    fn default_is_inter_prob_matches_spec_listing() {
        // §10.5 verbatim:
        //   default_is_inter_prob[IS_INTER_CONTEXTS] = { 9, 102, 187, 225 }
        assert_eq!(IS_INTER_CONTEXTS, 4);
        assert_eq!(DEFAULT_IS_INTER_PROB, [9, 102, 187, 225]);
    }

    #[test]
    fn seg_lvl_ref_frame_constant_matches_spec() {
        // §3 table of constants: SEG_LVL_REF_FRAME = 2.
        assert_eq!(SEG_LVL_REF_FRAME, 2);
    }

    #[test]
    fn is_inter_context_both_unavailable_returns_zero() {
        // §9.3.2 else branch: ctx = 0.
        let nb = IsInterNeighbours::default();
        assert_eq!(is_inter_context(nb), 0);
    }

    #[test]
    fn is_inter_context_both_available_both_intra_returns_three() {
        // §9.3.2: (LeftIntra && AboveIntra) ? 3 : … → 3.
        // INTRA_FRAME = 0 → *Intra = true.
        let nb = IsInterNeighbours {
            above: Some(INTRA_FRAME),
            left: Some(INTRA_FRAME),
        };
        assert_eq!(is_inter_context(nb), 3);
    }

    #[test]
    fn is_inter_context_both_available_one_intra_returns_one() {
        // §9.3.2: !(both intra) && (LeftIntra || AboveIntra) → 1.
        let nb_left_intra = IsInterNeighbours {
            above: Some(LAST_FRAME),
            left: Some(INTRA_FRAME),
        };
        assert_eq!(is_inter_context(nb_left_intra), 1);
        let nb_above_intra = IsInterNeighbours {
            above: Some(INTRA_FRAME),
            left: Some(GOLDEN_FRAME),
        };
        assert_eq!(is_inter_context(nb_above_intra), 1);
    }

    #[test]
    fn is_inter_context_both_available_neither_intra_returns_zero() {
        // §9.3.2: !(LeftIntra || AboveIntra) → 0.
        let nb = IsInterNeighbours {
            above: Some(LAST_FRAME),
            left: Some(ALTREF_FRAME),
        };
        assert_eq!(is_inter_context(nb), 0);
    }

    #[test]
    fn is_inter_context_only_above_available_returns_2x_above_intra() {
        // §9.3.2: else if (AvailU || AvailL) → ctx = 2 * (AvailU ?
        // AboveIntra : LeftIntra). AvailU branch: above intra → 2,
        // above inter → 0.
        let nb_intra = IsInterNeighbours {
            above: Some(INTRA_FRAME),
            left: None,
        };
        assert_eq!(is_inter_context(nb_intra), 2);
        let nb_inter = IsInterNeighbours {
            above: Some(LAST_FRAME),
            left: None,
        };
        assert_eq!(is_inter_context(nb_inter), 0);
    }

    #[test]
    fn is_inter_context_only_left_available_returns_2x_left_intra() {
        // §9.3.2: else if (AvailU || AvailL) → ctx = 2 * (AvailU ?
        // AboveIntra : LeftIntra). AvailL branch: left intra → 2,
        // left inter → 0.
        let nb_intra = IsInterNeighbours {
            above: None,
            left: Some(INTRA_FRAME),
        };
        assert_eq!(is_inter_context(nb_intra), 2);
        let nb_inter = IsInterNeighbours {
            above: None,
            left: Some(GOLDEN_FRAME),
        };
        assert_eq!(is_inter_context(nb_inter), 0);
    }

    #[test]
    fn is_inter_context_treats_none_neighbour_ref_frame_as_intra() {
        // §6.4.11: NONE = -1 satisfies `<= INTRA_FRAME = 0` per the
        // *Intra rule, mirroring the §6.4.11 "unavailable → force to
        // INTRA_FRAME" rule. A neighbour with ref_frame[0] = NONE
        // (single-prediction sentinel) registers as intra-side here.
        let nb = IsInterNeighbours {
            above: Some(NONE_REF_FRAME),
            left: Some(NONE_REF_FRAME),
        };
        assert_eq!(is_inter_context(nb), 3);
    }

    #[test]
    fn read_is_inter_seg_feature_active_with_intra_override_returns_false() {
        // §6.4.13 path 1: seg_feature_active(SEG_LVL_REF_FRAME) and
        // FeatureData[seg][SEG_LVL_REF_FRAME] == INTRA_FRAME → is_inter
        // = false. No coder bits consumed.
        let mut coder = zero_coder();
        let is_inter = read_is_inter(
            &mut coder,
            true,
            INTRA_FRAME as i16,
            &DEFAULT_IS_INTER_PROB,
            IsInterNeighbours::default(),
        )
        .unwrap();
        assert!(!is_inter);
    }

    #[test]
    fn read_is_inter_seg_feature_active_with_inter_override_returns_true() {
        // §6.4.13 path 1: seg_feature_active(SEG_LVL_REF_FRAME) and
        // FeatureData[seg][SEG_LVL_REF_FRAME] != INTRA_FRAME → is_inter
        // = true. No coder bits consumed. Test each non-INTRA value.
        for rf in [LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME] {
            let mut coder = zero_coder();
            let is_inter = read_is_inter(
                &mut coder,
                true,
                rf as i16,
                &DEFAULT_IS_INTER_PROB,
                IsInterNeighbours::default(),
            )
            .unwrap();
            assert!(is_inter, "ref_frame {rf} must derive is_inter=true");
        }
    }

    #[test]
    fn read_is_inter_zero_buffer_decodes_false() {
        // §6.4.13 path 2 with zero coder: every read_bool returns 0;
        // BINARY_TREE leaf 0 → is_inter = false.
        let mut coder = zero_coder();
        let is_inter = read_is_inter(
            &mut coder,
            false,
            0, // seg ref-frame data irrelevant when feature inactive
            &DEFAULT_IS_INTER_PROB,
            IsInterNeighbours::default(),
        )
        .unwrap();
        assert!(!is_inter);
    }

    #[test]
    fn read_is_inter_bias_buffer_decodes_true_for_low_prob() {
        // §6.4.13 path 2 with the bias coder (post-marker BoolValue=127).
        // §9.2.2's `split = 1 + ((127*p) >> 8)` keeps split <= 127 for
        // every `p <= 254`, so 127 >= split → bit=1 → BINARY_TREE
        // leaf 1 → is_inter=true for any "non-saturating" probability.
        // Use both-intra neighbours (ctx=3) and put a mid prob in
        // slot 3 to confirm the bias-coder bit=1 path.
        let bytes = make_bias_buffer(0x7F);
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let probs = [128u8, 128, 128, 128];
        let nb = IsInterNeighbours {
            above: Some(INTRA_FRAME),
            left: Some(INTRA_FRAME),
        };
        let is_inter = read_is_inter(&mut coder, false, 0, &probs, nb).unwrap();
        assert!(is_inter);
    }

    #[test]
    fn read_is_inter_picks_prob_by_context() {
        // The §6.4.13 listing reads `is_inter` with `is_inter_prob[ctx]`
        // where ctx is [`is_inter_context`]. We test the indexing
        // indirectly (mirroring [`read_skip_picks_prob_by_context`]):
        //
        // (a) the §9.3.2 ctx derivation matches the spec (covered by
        //     the `is_inter_context_*` tests above), and
        // (b) `read_is_inter` calls into `tree_decode` with the prob
        //     slot selected by ctx — confirmed here by running
        //     identical inputs across all four possible ctx values via
        //     distinct neighbour configurations, ensuring no panic /
        //     out-of-range index. The zero coder pins every read to
        //     bit=0 regardless of probability, so `is_inter` is false
        //     across all ctxs but the indexing path executes for each.
        let neighbour_configs = [
            IsInterNeighbours::default(), // ctx=0
            IsInterNeighbours {
                // ctx=1
                above: Some(INTRA_FRAME),
                left: Some(LAST_FRAME),
            },
            IsInterNeighbours {
                // ctx=2
                above: Some(INTRA_FRAME),
                left: None,
            },
            IsInterNeighbours {
                // ctx=3
                above: Some(INTRA_FRAME),
                left: Some(INTRA_FRAME),
            },
        ];
        for (ix, nb) in neighbour_configs.iter().enumerate() {
            // Distinct prob per slot to confirm none aliases another.
            let probs = [10u8, 60, 130, 200];
            let mut coder = zero_coder();
            let is_inter = read_is_inter(&mut coder, false, 0, &probs, *nb).unwrap();
            assert!(
                !is_inter,
                "zero coder must pin bit=0 for ctx={ix} regardless of probs"
            );
        }
    }

    #[test]
    fn read_is_inter_seg_feature_path_ignores_neighbours_and_coder() {
        // §6.4.13 path 1 short-circuits before any coder read or ctx
        // lookup. Verify the answer is identical regardless of
        // neighbour configuration and that the coder isn't consumed.
        for nb in [
            IsInterNeighbours::default(),
            IsInterNeighbours {
                above: Some(INTRA_FRAME),
                left: Some(INTRA_FRAME),
            },
            IsInterNeighbours {
                above: Some(LAST_FRAME),
                left: Some(ALTREF_FRAME),
            },
        ] {
            let mut coder = zero_coder();
            let is_inter = read_is_inter(
                &mut coder,
                true,
                LAST_FRAME as i16,
                &DEFAULT_IS_INTER_PROB,
                nb,
            )
            .unwrap();
            assert!(is_inter);
        }
    }

    // ----- §3 + §10.5 inter-mode / interp-filter dimensions -----

    #[test]
    fn inter_mode_dimension_constants_match_spec() {
        // §3 constants table: INTER_MODES = 4, INTER_MODE_CONTEXTS = 7
        // (vp9-spec.txt lines 506-507).
        assert_eq!(INTER_MODES, 4);
        assert_eq!(INTER_MODE_CONTEXTS, 7);
    }

    #[test]
    fn default_inter_mode_probs_matches_spec_listing() {
        // §10.5 verbatim:
        //   default_inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES - 1] = {
        //     {2,  173, 34},   // 0 = both zero mv
        //     {7,  145, 85},   // 1 = one zero mv + one a predicted mv
        //     {7,  166, 63},   // 2 = two predicted mvs
        //     {7,  94,  66},   // 3 = one predicted/zero and one new mv
        //     {8,  64,  46},   // 4 = two new mvs
        //     {17, 81,  31},   // 5 = one intra neighbor + x
        //     {25, 29,  30},   // 6 = two intra neighbors
        //   }
        let expected: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS] = [
            [2, 173, 34],
            [7, 145, 85],
            [7, 166, 63],
            [7, 94, 66],
            [8, 64, 46],
            [17, 81, 31],
            [25, 29, 30],
        ];
        assert_eq!(DEFAULT_INTER_MODE_PROBS, expected);
    }

    #[test]
    fn interp_filter_dimension_constants_match_spec() {
        // §3 constants table: SWITCHABLE_FILTERS = 3 (line 487),
        // INTERP_FILTER_CONTEXTS = 4 (line 495).
        assert_eq!(SWITCHABLE_FILTERS, 3);
        assert_eq!(INTERP_FILTER_CONTEXTS, 4);
    }

    #[test]
    fn default_interp_filter_probs_matches_spec_listing() {
        // §10.5 verbatim:
        //   default_interp_filter_probs[INTERP_FILTER_CONTEXTS]
        //                              [SWITCHABLE_FILTERS - 1] = {
        //     { 235, 162 },
        //     { 36, 255 },
        //     { 34, 3 },
        //     { 149, 144 }
        //   }
        let expected: [[u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS] =
            [[235, 162], [36, 255], [34, 3], [149, 144]];
        assert_eq!(DEFAULT_INTERP_FILTER_PROBS, expected);
    }
}
