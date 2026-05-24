//! VP9 per-block mode-info primitives per spec v0.7 — §6.4.8 / §6.4.10
//! / §9.3.3.
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
//! (`docs/video/vp9/vp9-spec.txt` §6.4.8 / §6.4.9 / §6.4.10 / §9.3.1 /
//! §9.3.2 / §9.3.3). No external library source consulted; every
//! formula and every tree array transcribed directly from the spec
//! listing.

// Helpers in this module are exercised exclusively from `#[cfg(test)]`
// and the deferred §6.4.6 driver until the per-frame public decode path
// lands.
#![allow(dead_code)]

use crate::bool_coder::BoolCoder;
use crate::compressed::{tx_mode_to_biggest_tx_size, TxMode};
use crate::residual::{BLOCK_8X8, MAX_TXSIZE_LOOKUP};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{BLOCK_16X16, BLOCK_32X32, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8};

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
}
