//! VP9 §8.8.2 `superblock loop filter process` per-edge predicate
//! derivation — per spec v0.7.
//!
//! This module lands the §8.8.2 steps 1-14 as a pure leaf primitive:
//! the per-edge book-keeping that turns the superblock raster position
//! `(plane, pass, row, col, edge, i)` plus the per-MI decode state at
//! the resolved `(loopRow, loopCol)` into the
//! `(x, y, loopRow, loopCol, isBlockEdge, isTxEdge, is32Edge,
//! onScreen, applyFilter)` predicate bundle. The §8.8.2 driver then
//! feeds the `is32Edge` / `x` / `y` outputs into §8.8.3
//! [`crate::filter_size`], looks the §8.8.4 strength up at
//! `(loopRow, loopCol)`, and — when `applyFilter == 1` and `lvl > 0`
//! — runs §8.8.5 [`crate::sample_filtering`].
//!
//! The §8.8.2 listing (`vp9-spec.txt` lines 5491-5586) describes the
//! outer driver:
//!
//! 1. `subX` / `subY` from the plane (`0`/`0` for luma, else the
//!    §6.2.2 `subsampling_x` / `subsampling_y`).
//! 2. `dx` / `dy` / `sub` / `edgeLen` from `pass`:
//!    * `pass == 0` (vertical): `dx = 1`, `dy = 0`, `sub = subX`,
//!      `edgeLen = 64 >> subY`.
//!    * `pass == 1` (horizontal): `dy = 1`, `dx = 0`, `sub = subY`,
//!      `edgeLen = 64 >> subX`.
//! 3. For `edge ∈ 0..(16 >> sub) - 1` and `i ∈ 0..edgeLen - 1`, the
//!    ordered steps 1-17 of §8.8.2 (lines 5526-5586).
//!
//! Steps 1-14 (the predicate derivation lifted here) are:
//!
//! 1. `x` / `y` (luma coordinates):
//!    * `pass == 0`: `x = col*8 + edge*(4 << subX)`,
//!      `y = row*8 + (i << subY)`.
//!    * `pass == 1`: `x = col*8 + (i << subX)`,
//!      `y = row*8 + edge*(4 << subY)`.
//! 2. `loopCol = ((x >> 3) >> subX) << subX`.
//! 3. `loopRow = ((y >> 3) >> subY) << subY`.
//! 4. `MiSize = MiSizes[ loopRow ][ loopCol ]`.
//! 5. `tx_size = TxSizes[ loopRow ][ loopCol ]`.
//! 6. `txSz = (plane > 0) ? get_uv_tx_size( ) : tx_size`.
//! 7. `sbSize`: `sub == 0` → `MiSize`, else `Max(BLOCK_16X16, MiSize)`.
//! 8. `skip = Skips[ loopRow ][ loopCol ]`.
//! 9. `isIntra = RefFrames[ loopRow ][ loopCol ][ 0 ] <= INTRA_FRAME`.
//! 10. `isBlockEdge`:
//!     * `pass == 0` and `x % (8*num_8x8_blocks_wide_lookup[sbSize]) == 0`
//!       → 1.
//!     * `pass == 1` and `y % (8*num_8x8_blocks_high_lookup[sbSize]) == 0`
//!       → 1.
//!     * else 0.
//! 11. `isTxEdge`:
//!     * `pass == 1` and `subX == 1` and `MiCols` odd and `edge` odd
//!       and `(x + 8) >= MiCols*8` → 0 (chroma horizontal boundary
//!       crossing the right image edge).
//!     * else `edge % (1 << txSz) == 0` → 1.
//!     * else 0.
//! 12. `is32Edge`: `edge % 8 == 0` → 1, else 0.
//! 13. `onScreen`:
//!     * `x >= 8*MiCols` → 0.
//!     * else `y >= 8*MiRows` → 0.
//!     * else `pass == 0` and `x == 0` → 0.
//!     * else `pass == 1` and `y == 0` → 0.
//!     * else 1.
//! 14. `applyFilter`:
//!     * `onScreen == 0` → 0.
//!     * else `isBlockEdge == 1` → 1.
//!     * else `isTxEdge == 1` and `isIntra == 1` → 1.
//!     * else `isTxEdge == 1` and `skip == 0` → 1.
//!     * else 0.
//!
//! ## Scope of this round
//!
//! Round 274 lands the §8.8.2 steps 1-14 predicate leaf only — a pure
//! function over the per-edge raster position and the per-MI decode
//! state already resolved at `(loopRow, loopCol)`. Step 6's
//! `get_uv_tx_size( )` resolution is the caller's responsibility (the
//! caller passes the already-resolved `txSz`, exactly as the §8.8.3
//! [`crate::filter_size`] caller does); this primitive does not look
//! up `TxSizes[ ][ ]` itself. The `dx` / `dy` / `sub` / `edgeLen`
//! header derivation (§8.8.2 lines 5510-5519) is exposed alongside via
//! [`superblock_filter_geometry`] so the caller can drive the
//! `edge` / `i` loop bounds.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * The §8.8.2 outer `edge` / `i` loop itself plus steps 15-17 that
//!   thread §8.8.3 / §8.8.4 / §8.8.5 — those need the full per-frame
//!   `MiSizes` / `TxSizes` / `Skips` / `RefFrames` arrays and the
//!   `CurrFrame` sample plane the §6.4.4 [`crate::decode_block`]
//!   fan-out produces.
//! * `get_uv_tx_size( )` (§8.8.2 step 6) — the chroma transform-size
//!   resolution.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.2 lines 5491-5586; §10.2
//! `num_8x8_blocks_wide_lookup[ ]` / `num_8x8_blocks_high_lookup[ ]`
//! lines 7111 / 7117; §3 `BLOCK_16X16 = 6` / `INTRA_FRAME = 0`).
//! `Max` is the §5.1 primitive.

use crate::filter_size::{PASS_HORIZONTAL, PASS_VERTICAL};
use crate::mode_info::INTRA_FRAME;
use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};
use crate::residual::BLOCK_16X16;

/// §8.8.2 `dx` / `dy` / `sub` / `edgeLen` header derivation per
/// `vp9-spec.txt` lines 5510-5519.
///
/// `dx` / `dy` specify the per-sample offset across the boundary
/// being filtered. `sub` is the sub-sampling factor in the direction
/// of the filter (perpendicular to the boundary). `edgeLen` is the
/// boundary length in samples (64 for luma, fewer for sub-sampled
/// chroma). The §8.8.2 driver iterates `edge ∈ 0..(16 >> sub) - 1`
/// and `i ∈ 0..edgeLen - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperblockFilterGeometry {
    /// `dx` per §8.8.2 line 5512 / 5515 — `1` for vertical edges
    /// (`pass == 0`), `0` for horizontal edges.
    pub dx: u32,
    /// `dy` per §8.8.2 line 5512 / 5515 — `0` for vertical edges,
    /// `1` for horizontal edges.
    pub dy: u32,
    /// `sub` per §8.8.2 line 5512 / 5515 — `subX` for vertical edges,
    /// `subY` for horizontal edges. The §8.8.2 `edge` loop runs to
    /// `(16 >> sub) - 1` and step 7 reads it for `sbSize`.
    pub sub: u32,
    /// `edgeLen` per §8.8.2 line 5513 / 5516 — `64 >> subY` for
    /// vertical edges, `64 >> subX` for horizontal edges. The §8.8.2
    /// `i` loop runs to `edgeLen - 1`.
    pub edge_len: u32,
}

/// Run the §8.8.2 `dx` / `dy` / `sub` / `edgeLen` derivation per
/// `vp9-spec.txt` lines 5510-5519.
///
/// # Inputs
///
/// * `pass` — [`PASS_VERTICAL`] (`0`) or [`PASS_HORIZONTAL`] (`1`).
/// * `sub_x`, `sub_y` — the §6.2.2 chroma sub-sampling factors for
///   the plane being filtered (already resolved to `0` for luma).
#[inline]
pub fn superblock_filter_geometry(pass: u8, sub_x: u32, sub_y: u32) -> SuperblockFilterGeometry {
    if pass == PASS_VERTICAL {
        SuperblockFilterGeometry {
            dx: 1,
            dy: 0,
            sub: sub_x,
            edge_len: 64 >> sub_y,
        }
    } else {
        SuperblockFilterGeometry {
            dx: 0,
            dy: 1,
            sub: sub_y,
            edge_len: 64 >> sub_x,
        }
    }
}

/// The §8.8.2 steps 1-14 predicate bundle: the per-edge book-keeping
/// outputs the §8.8.2 driver consumes in steps 15-17.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperblockFilterEdge {
    /// `x` luma coordinate from §8.8.2 step 1 (lines 5527-5531).
    pub x: u32,
    /// `y` luma coordinate from §8.8.2 step 1 (lines 5527-5531).
    pub y: u32,
    /// `loopCol` from §8.8.2 step 2 (line 5532) — the luma column in
    /// 8x8 mode-info units the per-MI arrays are indexed by.
    pub loop_col: u32,
    /// `loopRow` from §8.8.2 step 3 (line 5533) — the luma row in
    /// 8x8 mode-info units the per-MI arrays are indexed by.
    pub loop_row: u32,
    /// `isBlockEdge` from §8.8.2 step 10 (lines 5546-5551).
    pub is_block_edge: bool,
    /// `isTxEdge` from §8.8.2 step 11 (lines 5552-5560).
    pub is_tx_edge: bool,
    /// `is32Edge` from §8.8.2 step 12 (lines 5561-5564) — the §8.8.3
    /// `is32Edge` filter-size input.
    pub is_32_edge: bool,
    /// `onScreen` from §8.8.2 step 13 (lines 5565-5572).
    pub on_screen: bool,
    /// `applyFilter` from §8.8.2 step 14 (lines 5573-5580) — gates
    /// the §8.8.2 step 17 sample-filter call (alongside `lvl > 0`).
    pub apply_filter: bool,
}

/// The per-MI decode state the §8.8.2 driver reads at the resolved
/// `(loopRow, loopCol)` in steps 4-9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperblockFilterMi {
    /// `MiSize = MiSizes[ loopRow ][ loopCol ]` per §8.8.2 step 4
    /// (line 5534) — a §3 `BLOCK_*` constant.
    pub mi_size: u8,
    /// `txSz` per §8.8.2 step 6 (line 5536) — already resolved by the
    /// caller (`(plane > 0) ? get_uv_tx_size( ) : TxSizes[ ][ ]`).
    pub tx_sz: u8,
    /// `skip = Skips[ loopRow ][ loopCol ]` per §8.8.2 step 8
    /// (line 5544).
    pub skip: bool,
    /// `RefFrames[ loopRow ][ loopCol ][ 0 ]` per §8.8.2 step 9
    /// (line 5545); `isIntra = (this <= INTRA_FRAME)`.
    pub ref_frame_0: i32,
}

/// §5.1 `Max(x, y)` for the §8.8.2 step 7 `Max(BLOCK_16X16, MiSize)`.
#[inline]
fn max_u8(x: u8, y: u8) -> u8 {
    if x > y {
        x
    } else {
        y
    }
}

/// Run §8.8.2 steps 1-14 per `vp9-spec.txt` lines 5526-5580.
///
/// Returns the [`SuperblockFilterEdge`] predicate bundle for the edge
/// at raster position `(pass, row, col, edge, i)` in the plane with
/// sub-sampling `(sub_x, sub_y)`, given the per-MI decode state `mi`
/// resolved at the `(loopRow, loopCol)` this primitive computes.
///
/// # Inputs
///
/// * `pass` — [`PASS_VERTICAL`] (`0`, vertical edges) or
///   [`PASS_HORIZONTAL`] (`1`, horizontal edges).
/// * `row`, `col` — the superblock location in 8x8 units (§8.8.2
///   inputs).
/// * `edge` — the §8.8.2 outer-loop index, `0..(16 >> sub) - 1`.
/// * `i` — the §8.8.2 inner-loop index, `0..edgeLen - 1`.
/// * `sub_x`, `sub_y` — the plane sub-sampling factors (`0` for luma).
/// * `mi_cols`, `mi_rows` — the §6.2 `MiCols` / `MiRows` frame
///   dimensions in 8x8 mode-info units.
/// * `mi` — the per-MI decode state at `(loopRow, loopCol)`; the
///   caller resolves `loopRow` / `loopCol` from a prior call or
///   re-derives them via steps 2-3 here (they are echoed in the
///   output so the caller can index its arrays).
///
/// # Note on indexing order
///
/// Steps 4-9 read the per-MI arrays at `(loopRow, loopCol)`, which
/// steps 2-3 derive from `(x, y)`. The caller therefore resolves
/// `(loopRow, loopCol)` first — either by a cheap pre-pass computing
/// just steps 1-3, or by reading the echoed `loop_row` / `loop_col`
/// from a first call and re-issuing with the looked-up `mi`. The
/// echoed coordinates are identical across both calls because the
/// derivation is a pure function of the inputs.
#[allow(clippy::too_many_arguments)]
pub fn superblock_filter_edge(
    pass: u8,
    row: u32,
    col: u32,
    edge: u32,
    i: u32,
    sub_x: u32,
    sub_y: u32,
    mi_cols: u32,
    mi_rows: u32,
    mi: &SuperblockFilterMi,
) -> SuperblockFilterEdge {
    // Step 1: x / y in luma coordinates (lines 5527-5531).
    let (x, y) = if pass == PASS_VERTICAL {
        (col * 8 + edge * (4 << sub_x), row * 8 + (i << sub_y))
    } else {
        (col * 8 + (i << sub_x), row * 8 + edge * (4 << sub_y))
    };

    // Step 2: loopCol (line 5532).
    let loop_col = ((x >> 3) >> sub_x) << sub_x;
    // Step 3: loopRow (line 5533).
    let loop_row = ((y >> 3) >> sub_y) << sub_y;

    // Step 7: sbSize (lines 5538-5541). `sub` is the §8.8.2 geometry
    // factor in the filter direction.
    let sub = if pass == PASS_VERTICAL { sub_x } else { sub_y };
    let sb_size = if sub == 0 {
        mi.mi_size
    } else {
        max_u8(BLOCK_16X16, mi.mi_size)
    };

    // Step 9: isIntra (line 5545).
    let is_intra = mi.ref_frame_0 <= INTRA_FRAME;

    // Step 10: isBlockEdge (lines 5546-5551).
    let is_block_edge = if pass == PASS_VERTICAL {
        let bw = 8 * u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[sb_size as usize]);
        x % bw == 0
    } else {
        let bh = 8 * u32::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[sb_size as usize]);
        y % bh == 0
    };

    // Step 11: isTxEdge (lines 5552-5560).
    let is_tx_edge = if pass == PASS_HORIZONTAL
        && sub_x == 1
        && (mi_cols & 1) == 1
        && (edge & 1) == 1
        && (x + 8) >= mi_cols * 8
    {
        false
    } else {
        edge % (1u32 << mi.tx_sz) == 0
    };

    // Step 12: is32Edge (lines 5561-5564).
    let is_32_edge = edge % 8 == 0;

    // Step 13: onScreen (lines 5565-5572) — true unless any of the
    // four exclusion conditions holds (off the right / bottom of the
    // visible area, or the implicit left / top frame edge).
    let off_right = x >= 8 * mi_cols;
    let off_bottom = y >= 8 * mi_rows;
    let left_frame_edge = pass == PASS_VERTICAL && x == 0;
    let top_frame_edge = pass == PASS_HORIZONTAL && y == 0;
    let on_screen = !(off_right || off_bottom || left_frame_edge || top_frame_edge);

    // Step 14: applyFilter (lines 5573-5580) — filter on-screen samples
    // that cross a prediction-block edge, or a transform-block edge of
    // an intra block, or a transform-block edge of a non-skipped block.
    let apply_filter = on_screen && (is_block_edge || (is_tx_edge && (is_intra || !mi.skip)));

    SuperblockFilterEdge {
        x,
        y,
        loop_col,
        loop_row,
        is_block_edge,
        is_tx_edge,
        is_32_edge,
        on_screen,
        apply_filter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{BLOCK_4X4, BLOCK_64X64, BLOCK_8X8};

    const TX_4X4: u8 = 0;
    const TX_8X8: u8 = 1;
    const TX_32X32: u8 = 3;

    fn mi(mi_size: u8, tx_sz: u8, skip: bool, ref_frame_0: i32) -> SuperblockFilterMi {
        SuperblockFilterMi {
            mi_size,
            tx_sz,
            skip,
            ref_frame_0,
        }
    }

    // ----- geometry (§8.8.2 lines 5510-5519) -----

    #[test]
    fn geometry_luma_vertical() {
        let g = superblock_filter_geometry(PASS_VERTICAL, 0, 0);
        assert_eq!(
            g,
            SuperblockFilterGeometry {
                dx: 1,
                dy: 0,
                sub: 0,
                edge_len: 64,
            }
        );
    }

    #[test]
    fn geometry_luma_horizontal() {
        let g = superblock_filter_geometry(PASS_HORIZONTAL, 0, 0);
        assert_eq!(
            g,
            SuperblockFilterGeometry {
                dx: 0,
                dy: 1,
                sub: 0,
                edge_len: 64,
            }
        );
    }

    #[test]
    fn geometry_chroma_420_vertical() {
        // pass 0: sub = subX = 1, edgeLen = 64 >> subY = 32.
        let g = superblock_filter_geometry(PASS_VERTICAL, 1, 1);
        assert_eq!(
            g,
            SuperblockFilterGeometry {
                dx: 1,
                dy: 0,
                sub: 1,
                edge_len: 32,
            }
        );
    }

    #[test]
    fn geometry_chroma_420_horizontal() {
        // pass 1: sub = subY = 1, edgeLen = 64 >> subX = 32.
        let g = superblock_filter_geometry(PASS_HORIZONTAL, 1, 1);
        assert_eq!(
            g,
            SuperblockFilterGeometry {
                dx: 0,
                dy: 1,
                sub: 1,
                edge_len: 32,
            }
        );
    }

    // ----- step 1 x/y coordinates -----

    #[test]
    fn xy_vertical_luma() {
        // pass 0, luma: x = col*8 + edge*4, y = row*8 + i.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            2,
            3,
            5,
            7,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
        );
        assert_eq!(e.x, 3 * 8 + 5 * 4);
        assert_eq!(e.y, 2 * 8 + 7);
    }

    #[test]
    fn xy_horizontal_luma() {
        // pass 1, luma: x = col*8 + i, y = row*8 + edge*4.
        let e = superblock_filter_edge(
            PASS_HORIZONTAL,
            2,
            3,
            5,
            7,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
        );
        assert_eq!(e.x, 3 * 8 + 7);
        assert_eq!(e.y, 2 * 8 + 5 * 4);
    }

    #[test]
    fn xy_vertical_chroma_420() {
        // pass 0, subX = subY = 1: x = col*8 + edge*(4<<1) = col*8 + edge*8,
        // y = row*8 + (i << 1).
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            1,
            1,
            2,
            3,
            1,
            1,
            64,
            64,
            &mi(BLOCK_16X16, TX_8X8, false, INTRA_FRAME),
        );
        assert_eq!(e.x, 8 + 2 * 8);
        assert_eq!(e.y, 8 + (3 << 1));
    }

    // ----- steps 2-3 loopRow / loopCol -----

    #[test]
    fn loop_coords_luma_identity() {
        // subX = subY = 0: loopCol = x >> 3, loopRow = y >> 3.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            4,
            4,
            2,
            5,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
        );
        assert_eq!(e.loop_col, e.x >> 3);
        assert_eq!(e.loop_row, e.y >> 3);
    }

    #[test]
    fn loop_coords_chroma_align_down() {
        // subX = subY = 1: loopCol = ((x>>3)>>1)<<1 — forced even.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            1,
            1,
            3,
            6,
            1,
            1,
            64,
            64,
            &mi(BLOCK_16X16, TX_8X8, false, INTRA_FRAME),
        );
        assert_eq!(e.loop_col, ((e.x >> 3) >> 1) << 1);
        assert_eq!(e.loop_row, ((e.y >> 3) >> 1) << 1);
        assert_eq!(e.loop_col & 1, 0);
        assert_eq!(e.loop_row & 1, 0);
    }

    // ----- step 10 isBlockEdge -----

    #[test]
    fn block_edge_vertical_8x8_every_8() {
        // BLOCK_8X8 → num_8x8_wide = 1 → boundary every 8 luma samples.
        // x = col*8 + edge*4. col = 0, edge = 0 → x = 0 (multiple of 8).
        let e0 = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            0,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert!(e0.is_block_edge);
        // edge = 1 → x = 4 → not a multiple of 8 → false.
        let e1 = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            1,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert!(!e1.is_block_edge);
    }

    #[test]
    fn block_edge_vertical_64x64_every_64() {
        // BLOCK_64X64 → num_8x8_wide = 8 → boundary every 64 samples.
        // edge = 2 → x = 8, not a multiple of 64 → false.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_64X64, TX_32X32, false, INTRA_FRAME),
        );
        assert!(!e.is_block_edge);
    }

    // ----- step 11 isTxEdge -----

    #[test]
    fn tx_edge_multiple_of_tx_size() {
        // TX_8X8 → 1 << txSz = 2 → isTxEdge when edge % 2 == 0.
        let even = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            4,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
        );
        assert!(even.is_tx_edge);
        let odd = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            3,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
        );
        assert!(!odd.is_tx_edge);
    }

    #[test]
    fn tx_edge_tx4x4_always() {
        // TX_4X4 → 1 << 0 = 1 → every edge is a tx edge.
        for edge in 0..8 {
            let e = superblock_filter_edge(
                PASS_VERTICAL,
                0,
                0,
                edge,
                0,
                0,
                0,
                64,
                64,
                &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
            );
            assert!(e.is_tx_edge, "edge {edge}");
        }
    }

    #[test]
    fn tx_edge_chroma_right_image_edge_suppressed() {
        // pass 1, subX = 1, MiCols odd, edge odd, (x+8) >= MiCols*8 → false.
        // MiCols = 5 (odd), so 8*MiCols = 40. Pick i so x+8 >= 40.
        // pass 1: x = col*8 + (i << subX). col = 4, i = 1 → x = 32 + 2 = 34,
        // x+8 = 42 >= 40. edge = 1 (odd). txSz arbitrary.
        let e = superblock_filter_edge(
            PASS_HORIZONTAL,
            0,
            4,
            1,
            1,
            1,
            1,
            5,
            64,
            &mi(BLOCK_16X16, TX_4X4, false, INTRA_FRAME),
        );
        assert!(!e.is_tx_edge);
        // Without the right-edge condition (even edge), TX_4X4 always
        // gives a tx edge.
        let e2 = superblock_filter_edge(
            PASS_HORIZONTAL,
            0,
            4,
            0,
            1,
            1,
            1,
            5,
            64,
            &mi(BLOCK_16X16, TX_4X4, false, INTRA_FRAME),
        );
        assert!(e2.is_tx_edge);
    }

    // ----- step 12 is32Edge -----

    #[test]
    fn is_32_edge_multiple_of_8() {
        for edge in 0..16 {
            let e = superblock_filter_edge(
                PASS_VERTICAL,
                0,
                0,
                edge,
                0,
                0,
                0,
                64,
                64,
                &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
            );
            assert_eq!(e.is_32_edge, edge % 8 == 0, "edge {edge}");
        }
    }

    // ----- step 13 onScreen -----

    #[test]
    fn on_screen_left_top_excluded() {
        // pass 0, x == 0 → off-screen (left frame edge, nothing to the left).
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            0,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert!(!e.on_screen);
        // pass 1, y == 0 → off-screen (top frame edge).
        let e2 = superblock_filter_edge(
            PASS_HORIZONTAL,
            0,
            0,
            0,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert!(!e2.on_screen);
    }

    #[test]
    fn on_screen_right_bottom_excluded() {
        // x >= 8*MiCols → off-screen. MiCols = 2 → 8*MiCols = 16.
        // pass 0: x = col*8 + edge*4. col = 2 → x = 16 >= 16.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            2,
            0,
            0,
            0,
            0,
            2,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert!(!e.on_screen);
    }

    #[test]
    fn on_screen_interior_true() {
        // Interior vertical edge: x != 0, x < 8*MiCols, y < 8*MiRows.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert_eq!(e.x, 8);
        assert!(e.on_screen);
    }

    // ----- step 14 applyFilter -----

    #[test]
    fn apply_filter_off_screen_is_false() {
        // pass 0, x == 0 → off-screen → applyFilter false regardless.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            0,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, false, INTRA_FRAME),
        );
        assert!(!e.apply_filter);
    }

    #[test]
    fn apply_filter_block_edge_true() {
        // Interior block edge → applyFilter regardless of skip/intra.
        // BLOCK_8X8, x = 8 (multiple of 8) → isBlockEdge.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_4X4, true, 1),
        );
        assert!(e.is_block_edge);
        assert!(e.apply_filter);
    }

    #[test]
    fn apply_filter_tx_edge_intra_true_even_if_skip() {
        // Not a block edge but a tx edge on an intra block → filter even
        // when skip. BLOCK_64X64 (block edge only every 64) with TX_4X4
        // (tx edge every 4), edge = 2 → x = 8: not block edge, is tx edge.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_64X64, TX_4X4, true, INTRA_FRAME),
        );
        assert!(!e.is_block_edge);
        assert!(e.is_tx_edge);
        assert!(e.apply_filter);
    }

    #[test]
    fn apply_filter_tx_edge_inter_skip_false() {
        // Tx edge, inter (ref_frame_0 > INTRA_FRAME), skip → not filtered.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_64X64, TX_4X4, true, 1),
        );
        assert!(!e.is_block_edge);
        assert!(e.is_tx_edge);
        assert!(!e.apply_filter);
    }

    #[test]
    fn apply_filter_tx_edge_inter_noskip_true() {
        // Tx edge, inter, not skip → filtered.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_64X64, TX_4X4, false, 1),
        );
        assert!(e.is_tx_edge);
        assert!(e.apply_filter);
    }

    #[test]
    fn sb_size_chroma_promotes_to_16x16() {
        // sub = 1 (chroma in filter direction): sbSize = Max(BLOCK_16X16,
        // MiSize). With MiSize = BLOCK_4X4, sbSize becomes BLOCK_16X16 →
        // num_8x8_wide[BLOCK_16X16] = 2 → block boundary every 16 samples.
        // pass 0, subX = 1: x = col*8 + edge*(4<<1) = col*8 + edge*8.
        // edge = 1 → x = 8: not a multiple of 16 → not block edge.
        let e = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            1,
            0,
            1,
            1,
            64,
            64,
            &mi(BLOCK_4X4, TX_4X4, false, INTRA_FRAME),
        );
        assert_eq!(e.x, 8);
        assert!(!e.is_block_edge);
        // edge = 2 → x = 16: multiple of 16 → block edge.
        let e2 = superblock_filter_edge(
            PASS_VERTICAL,
            0,
            0,
            2,
            0,
            1,
            1,
            64,
            64,
            &mi(BLOCK_4X4, TX_4X4, false, INTRA_FRAME),
        );
        assert_eq!(e2.x, 16);
        assert!(e2.is_block_edge);
    }
}
