//! VP9 §8.8.3 `filter_size( )` — per spec v0.7.
//!
//! This module lands the per-edge [`filter_size`] derivation as a pure
//! leaf primitive. The §8.8.2 superblock raster walker invokes it on
//! every loop-filter edge to pick the maximum filter size for the
//! `(plane, pass, x, y)` step before the §8.8.4 adaptive-strength
//! pass picks the actual strength and the §8.8.5 sample-filter pass
//! does the deblocking.
//!
//! The §8.8.3 listing (`vp9-spec.txt` §8.8.3 lines 5587-5625)
//! describes:
//!
//! 1. A `baseSize` derivation:
//!    * If `txSz == TX_4X4` and `is32Edge == 1`, `baseSize = TX_8X8`.
//!    * Otherwise `baseSize = Min(TX_16X16, txSz)`.
//! 2. A chroma-edge clip: when filtering a luma edge that the chroma
//!    plane's sub-sampling would push past the frame's right / bottom
//!    edge, `filterSize` is forced down to `TX_8X8`.
//!    * Vertical edges (`pass == 0`): if `subX == 1` (chroma is
//!      horizontally sub-sampled) and `baseSize == TX_16X16` and
//!      `(x >> 3) == MiCols - 1`, set `filterSize = TX_8X8`.
//!    * Horizontal edges (`pass == 1`): if `subY == 1` (chroma is
//!      vertically sub-sampled) and `baseSize == TX_16X16` and
//!      `(y >> 3) == MiRows - 1`, set `filterSize = TX_8X8`.
//! 3. Otherwise `filterSize = baseSize`.
//!
//! Per the §8.8.3 lead paragraph (`vp9-spec.txt` lines 5597-5599) the
//! purpose is to reduce the width of chroma filters when the filter
//! would otherwise cross the frame boundary and to clip the filter
//! size to a minimum of `TX_8X8` for boundaries on a multiple of 32
//! samples.
//!
//! ## Scope of this round
//!
//! Round 244 lands the §8.8.3 leaf only — pure-state function over
//! integers. The caller is responsible for supplying the resolved
//! `txSz` from `TxSizes[ r ][ c ]` and the `is32Edge` from
//! `(loopRow % 4 == 0)` / `(loopCol % 4 == 0)` per §8.8.2; this
//! primitive does not walk the §8.8.2 raster itself.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
//!   that calls this primitive at every edge.
//! * §8.8.4 `adaptive_filter_strength` — the per-edge `(lvl, limit,
//!   blimit, thresh)` derivation that follows §8.8.3.
//! * §8.8.5 `sample_filtering` — the actual edge-filter primitives.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.3 lines 5587-5625; §7.4.8
//! `TX_4X4 = 0` / `TX_8X8 = 1` / `TX_16X16 = 2` / `TX_32X32 = 3`
//! lines 3937-3940; §6.2 `MiCols = (FrameWidth + 7) >> 3` /
//! `MiRows = (FrameHeight + 7) >> 3` lines 1760-1761). `Min` is the
//! §5.1 primitive.

/// `TX_4X4 = 0` per §7.4.8 (`vp9-spec.txt` line 3937) — the smallest
/// VP9 transform size.
pub const TX_4X4: u8 = 0;

/// `TX_8X8 = 1` per §7.4.8 (`vp9-spec.txt` line 3938) — the §8.8.3
/// minimum filter size for boundaries on a multiple of 32 samples and
/// for sub-sampled-chroma-clipped 16x16 edges.
pub const TX_8X8: u8 = 1;

/// `TX_16X16 = 2` per §7.4.8 (`vp9-spec.txt` line 3939) — the §8.8.3
/// `Min(TX_16X16, txSz)` upper bound on `baseSize`.
pub const TX_16X16: u8 = 2;

/// `TX_32X32 = 3` per §7.4.8 (`vp9-spec.txt` line 3940) — the largest
/// VP9 transform size; never the §8.8.3 output (capped by the
/// `Min(TX_16X16, txSz)` clip).
pub const TX_32X32: u8 = 3;

/// `pass == 0` per §8.8.3 lines 5616 / 5621 — vertical edge filtering
/// pass. Sub-sampled-chroma right-edge clip applies here.
pub const PASS_VERTICAL: u8 = 0;

/// `pass == 1` per §8.8.3 lines 5616 / 5621 — horizontal edge
/// filtering pass. Sub-sampled-chroma bottom-edge clip applies here.
pub const PASS_HORIZONTAL: u8 = 1;

/// §5.1 `Min(x, y)` — the smaller of two `u8` operands.
#[inline]
fn min_u8(x: u8, y: u8) -> u8 {
    if x < y {
        x
    } else {
        y
    }
}

/// Run §8.8.3 `filter_size( )` per `vp9-spec.txt` lines 5587-5625.
///
/// Returns the §8.8.3 `filterSize` output: the maximum filter size
/// that may be used at the given edge, picked from `TX_4X4 = 0`,
/// `TX_8X8 = 1`, or `TX_16X16 = 2`. (`TX_32X32` is never returned —
/// the §8.8.3 `baseSize = Min(TX_16X16, txSz)` step clips it.)
///
/// # Inputs
///
/// * `tx_sz` — the §7.4.8 transform size at the block being filtered
///   (one of [`TX_4X4`], [`TX_8X8`], [`TX_16X16`], [`TX_32X32`]).
///   Values outside `0..=3` are treated as `TX_32X32` by the
///   `Min(TX_16X16, txSz)` clip but the caller should pass a valid
///   §7.4.8 size.
/// * `is_32_edge` — the §8.8.2 flag indicating that the current edge
///   sits on a multiple-of-32-samples boundary. When true and
///   `tx_sz == TX_4X4` the §8.8.3 spec forces `baseSize = TX_8X8` so
///   the filter is wide enough for the larger underlying super-block
///   transform.
/// * `pass` — [`PASS_VERTICAL`] (vertical edges) or
///   [`PASS_HORIZONTAL`] (horizontal edges) per §8.8.3 lines 5616 /
///   5621.
/// * `x`, `y` — the §8.8.2 luma-sample coordinates of the edge.
/// * `sub_x`, `sub_y` — the §6.2.2 chroma sub-sampling factors for the
///   plane being filtered (1 if sub-sampled, 0 otherwise).
/// * `mi_cols`, `mi_rows` — the §6.2 `MiCols` / `MiRows` frame
///   dimensions in 8x8 mode-info units. The chroma-edge clip checks
///   `(x >> 3) == mi_cols - 1` / `(y >> 3) == mi_rows - 1` for the
///   right / bottom frame edge.
///
/// # Output
///
/// One of [`TX_4X4`], [`TX_8X8`], or [`TX_16X16`]. The §8.8.5 sample-
/// filtering process branches on this output to pick the narrow /
/// wide filter kernel.
///
/// # Listing
///
/// `vp9-spec.txt` §8.8.3 lines 5587-5625:
///
/// ```text
/// 1. baseSize:
///      if (txSz == TX_4X4 && is32Edge == 1) baseSize = TX_8X8
///      else                                  baseSize = Min(TX_16X16, txSz)
/// 2. filterSize:
///      if (pass == 0 && subX == 1 && baseSize == TX_16X16 &&
///          (x >> 3) == MiCols - 1)
///          filterSize = TX_8X8
///      else if (pass == 1 && subY == 1 && baseSize == TX_16X16 &&
///               (y >> 3) == MiRows - 1)
///          filterSize = TX_8X8
///      else
///          filterSize = baseSize
/// ```
///
/// The `clippy::too_many_arguments` allow on this signature is
/// load-bearing: §8.8.3's listing (`vp9-spec.txt` lines 5587-5594)
/// declares exactly nine inputs (`txSz`, `is32Edge`, `pass`, `x`,
/// `y`, `subX`, `subY`, plus the `MiCols` / `MiRows` frame-dimension
/// reads on lines 5619 / 5624). Bundling them into a struct would
/// hide the §8.8.3 contract.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn filter_size(
    tx_sz: u8,
    is_32_edge: bool,
    pass: u8,
    x: u32,
    y: u32,
    sub_x: u8,
    sub_y: u8,
    mi_cols: u32,
    mi_rows: u32,
) -> u8 {
    // §8.8.3 lines 5609-5611: baseSize derivation.
    let base_size = if tx_sz == TX_4X4 && is_32_edge {
        // Line 5610: tx_sz == TX_4X4 && is32Edge == 1 → baseSize = TX_8X8.
        TX_8X8
    } else {
        // Line 5611: otherwise baseSize = Min(TX_16X16, txSz).
        min_u8(TX_16X16, tx_sz)
    };

    // §8.8.3 lines 5615-5625: filterSize derivation.

    // Lines 5615-5619: vertical chroma-right-edge clip.
    let on_right_edge = mi_cols > 0 && (x >> 3) == (mi_cols - 1);
    let vertical_chroma_clip =
        pass == PASS_VERTICAL && sub_x == 1 && base_size == TX_16X16 && on_right_edge;
    if vertical_chroma_clip {
        return TX_8X8;
    }

    // Lines 5620-5624: horizontal chroma-bottom-edge clip.
    let on_bottom_edge = mi_rows > 0 && (y >> 3) == (mi_rows - 1);
    let horizontal_chroma_clip =
        pass == PASS_HORIZONTAL && sub_y == 1 && base_size == TX_16X16 && on_bottom_edge;
    if horizontal_chroma_clip {
        return TX_8X8;
    }

    // Line 5625: filterSize = baseSize.
    base_size
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §8.8.3 line 5611 `baseSize = Min(TX_16X16, txSz)`: a `TX_8X8`
    /// transform on an interior edge produces `filterSize = TX_8X8`
    /// regardless of pass / subsampling.
    #[test]
    fn min_clip_keeps_smaller_tx_size() {
        // tx_sz = TX_8X8, is_32_edge = false, pass = 0 (vertical),
        // interior coords, mid-frame: baseSize = Min(TX_16X16, TX_8X8)
        // = TX_8X8. No clip applies.
        assert_eq!(
            filter_size(TX_8X8, false, PASS_VERTICAL, 32, 32, 0, 0, 16, 16),
            TX_8X8
        );
    }

    /// §8.8.3 line 5611 `baseSize = Min(TX_16X16, txSz)`: a `TX_32X32`
    /// transform clips down to `baseSize = TX_16X16` because the
    /// loop-filter never operates on wider than 16-sample kernels.
    #[test]
    fn min_clip_caps_tx_32x32_at_tx_16x16() {
        // tx_sz = TX_32X32, interior edge, no chroma sub-sampling:
        // baseSize = Min(TX_16X16, TX_32X32) = TX_16X16. No clip
        // applies because sub_x / sub_y are 0.
        assert_eq!(
            filter_size(TX_32X32, false, PASS_VERTICAL, 32, 32, 0, 0, 16, 16),
            TX_16X16
        );
    }

    /// §8.8.3 line 5611: a `TX_4X4` block on a non-`is32Edge` edge
    /// produces `baseSize = Min(TX_16X16, TX_4X4) = TX_4X4`.
    #[test]
    fn tx_4x4_on_non_32_edge_keeps_tx_4x4() {
        // tx_sz = TX_4X4, is_32_edge = false: §8.8.3 line 5610 doesn't
        // fire; line 5611 yields baseSize = TX_4X4. No clip.
        assert_eq!(
            filter_size(TX_4X4, false, PASS_VERTICAL, 16, 16, 0, 0, 16, 16),
            TX_4X4
        );
    }

    /// §8.8.3 line 5610 `txSz == TX_4X4 && is32Edge == 1 → baseSize =
    /// TX_8X8`: the §8.8.3 lead paragraph's "minimum size of TX_8X8
    /// for boundaries on a multiple of 32 samples" rule.
    #[test]
    fn tx_4x4_on_32_edge_promotes_to_tx_8x8() {
        // tx_sz = TX_4X4 BUT is_32_edge = true: line 5610 forces
        // baseSize = TX_8X8. No clip applies (interior edge, no
        // sub-sampling).
        assert_eq!(
            filter_size(TX_4X4, true, PASS_VERTICAL, 32, 32, 0, 0, 16, 16),
            TX_8X8
        );
    }

    /// §8.8.3 lines 5615-5619: vertical-pass chroma right-edge clip
    /// forces `filterSize = TX_8X8`. The §8.8.3 lead paragraph's
    /// "reduce the width of chroma filters" purpose.
    #[test]
    fn vertical_chroma_right_edge_clip_to_tx_8x8() {
        // tx_sz = TX_16X16, pass = 0, sub_x = 1, x = 16 * 8 = 128:
        // (x >> 3) = 16 = mi_cols (let mi_cols = 17) - 1. Clip fires.
        let mi_cols = 17;
        let x = (mi_cols - 1) * 8;
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, x, 32, 1, 0, mi_cols, 32),
            TX_8X8
        );
    }

    /// §8.8.3 lines 5615-5619: the vertical chroma right-edge clip
    /// only fires for `pass == 0`. The same coordinates on the
    /// horizontal pass do NOT clip the size (the horizontal-edge clip
    /// on lines 5620-5624 has its own gates).
    #[test]
    fn vertical_clip_doesnt_fire_on_horizontal_pass() {
        // Same coordinates as the previous test but pass = 1
        // (horizontal). The §8.8.3 line 5616 `pass == 0` gate is OFF.
        // baseSize = TX_16X16 stays. Horizontal clip would need
        // sub_y == 1 AND on-bottom-edge, neither true here.
        let mi_cols = 17;
        let x = (mi_cols - 1) * 8;
        assert_eq!(
            filter_size(TX_16X16, false, PASS_HORIZONTAL, x, 32, 1, 0, mi_cols, 32),
            TX_16X16
        );
    }

    /// §8.8.3 lines 5615-5619: the vertical chroma clip only fires
    /// for `sub_x == 1`. A 4:4:4 plane (no horizontal sub-sampling)
    /// keeps the `TX_16X16` filter width.
    #[test]
    fn vertical_clip_doesnt_fire_when_sub_x_zero() {
        let mi_cols = 17;
        let x = (mi_cols - 1) * 8;
        // sub_x = 0 → line 5617 condition fails → baseSize wins.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, x, 32, 0, 0, mi_cols, 32),
            TX_16X16
        );
    }

    /// §8.8.3 lines 5615-5619: the vertical chroma clip only fires
    /// when `baseSize == TX_16X16`. A `TX_8X8` baseSize on the same
    /// edge keeps `TX_8X8` (it's already the clip target).
    #[test]
    fn vertical_clip_skipped_when_base_size_already_smaller() {
        let mi_cols = 17;
        let x = (mi_cols - 1) * 8;
        // tx_sz = TX_8X8 → baseSize = TX_8X8. Clip gate's
        // "baseSize == TX_16X16" check fails.
        assert_eq!(
            filter_size(TX_8X8, false, PASS_VERTICAL, x, 32, 1, 0, mi_cols, 32),
            TX_8X8
        );
    }

    /// §8.8.3 lines 5615-5619: vertical clip's `(x >> 3) == MiCols -
    /// 1` gate only fires on the right frame edge. An interior edge
    /// (one MI to the left) doesn't clip.
    #[test]
    fn vertical_clip_skipped_on_interior_edge() {
        let mi_cols = 17;
        let x = (mi_cols - 2) * 8; // one MI inside the right edge
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, x, 32, 1, 0, mi_cols, 32),
            TX_16X16
        );
    }

    /// §8.8.3 lines 5620-5624: horizontal-pass chroma bottom-edge
    /// clip forces `filterSize = TX_8X8` mirroring the vertical case.
    #[test]
    fn horizontal_chroma_bottom_edge_clip_to_tx_8x8() {
        let mi_rows = 17;
        let y = (mi_rows - 1) * 8;
        // tx_sz = TX_16X16, pass = 1, sub_y = 1, on bottom edge.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_HORIZONTAL, 32, y, 0, 1, 32, mi_rows),
            TX_8X8
        );
    }

    /// §8.8.3 lines 5620-5624: horizontal chroma clip only fires for
    /// `pass == 1`. The same coordinates on the vertical pass keep
    /// `TX_16X16` (the vertical-pass gate's `sub_x == 1` check fails
    /// here).
    #[test]
    fn horizontal_clip_doesnt_fire_on_vertical_pass() {
        let mi_rows = 17;
        let y = (mi_rows - 1) * 8;
        // sub_x = 0 (vertical-pass gate fails), sub_y = 1 (but pass = 0).
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, 32, y, 0, 1, 32, mi_rows),
            TX_16X16
        );
    }

    /// §8.8.3 lines 5620-5624: horizontal clip only fires when
    /// `sub_y == 1`. A 4:2:2 plane (vertically full-rate) keeps
    /// `TX_16X16`.
    #[test]
    fn horizontal_clip_doesnt_fire_when_sub_y_zero() {
        let mi_rows = 17;
        let y = (mi_rows - 1) * 8;
        // sub_y = 0 → horizontal gate fails. baseSize = TX_16X16.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_HORIZONTAL, 32, y, 0, 0, 32, mi_rows),
            TX_16X16
        );
    }

    /// §8.8.3 lines 5610 + 5615: composition of the §8.8.3 step 1
    /// `is32Edge` promotion with the step 2 chroma right-edge clip.
    /// `TX_4X4` + `is32Edge = 1` → `baseSize = TX_8X8`; the chroma
    /// clip's "baseSize == TX_16X16" gate is then OFF so the result
    /// stays at `TX_8X8` (not further clipped).
    #[test]
    fn is_32_edge_promotion_doesnt_trigger_chroma_clip() {
        let mi_cols = 17;
        let x = (mi_cols - 1) * 8;
        // tx_sz = TX_4X4, is_32_edge = true → baseSize = TX_8X8.
        // Chroma right-edge gate fails (baseSize != TX_16X16).
        assert_eq!(
            filter_size(TX_4X4, true, PASS_VERTICAL, x, 32, 1, 0, mi_cols, 32),
            TX_8X8
        );
    }

    /// `mi_cols == 0` / `mi_rows == 0` edge case: the `MiCols - 1`
    /// underflow guard in the impl ensures a 0-sized frame doesn't
    /// fire the chroma clip via wrap-around.
    #[test]
    fn zero_mi_dimensions_skip_chroma_clip() {
        // Even with sub_x = 1 and tx_sz = TX_16X16 and x = 0, the
        // mi_cols = 0 guard makes on_right_edge false.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, 0, 0, 1, 1, 0, 0),
            TX_16X16
        );
    }
}
