//! §6.5.2 / §6.5.3 / §6.5.4 / §6.5.5 / §6.5.12 motion-vector reference
//! geometry — the `is_inside( )`, `clamp_mv_ref( )`, `clamp_mv_row( )`,
//! `clamp_mv_col( )` and `find_best_ref_mvs( )` primitives.
//!
//! These sit one layer above the §6.4.19 [`crate::mv::read_mv`] residual
//! decode: `find_best_ref_mvs( )` is what derives the `BestMv[ ref ]`
//! predictor that `read_mv( )` adds the decoded difference onto. The full
//! §6.5.1 `find_mv_refs( )` candidate scan (which populates the
//! `RefListMv[ ]` list from neighbouring mode-info and the previous
//! frame's stored vectors) needs the frame-wide `Mvs` / `RefFrames`
//! arrays and the §6.5.6-6.5.11 `add_mv_ref_list( )` / `get_block_mv( )`
//! helpers; it lands on top of this layer once those arrays are threaded.
//!
//! Everything in this module is a pure function of the per-block frame
//! geometry (`MiRow`, `MiCol`, `MiRows`, `MiCols`, `MiSize`) gathered in
//! [`MvRefGeometry`], plus the candidate list itself. There is no bool
//! coder and no probability state, so each primitive is directly testable
//! against constructed geometry in isolation from the rest of the inter
//! path.

// The §6.5.1 `find_mv_refs( )` driver that consumes these clamps lands on
// top of this leaf layer in a later round; until then the primitives are
// reachable only from the unit tests, so the crate-internal `dead_code`
// lint is silenced module-wide (mirrors `mv`'s deferred-inter residual
// primitives).
#![allow(dead_code)]

use crate::mv::use_mv_hp;
use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};

/// `MV_BORDER = 128` per §3 (`vp9-spec.txt` line 520). The clip border the
/// §6.5.3 `clamp_mv_ref( )` step passes to `clamp_mv_row/col( )`.
pub(crate) const MV_BORDER: i32 = 128;

/// `INTERP_EXTEND = 4` per §3 (`vp9-spec.txt` line 521). Sub-pel
/// interpolation reach in full-pel units.
pub(crate) const INTERP_EXTEND: i32 = 4;

/// `BORDERINPIXELS = 160` per §3 (`vp9-spec.txt` line 522). The frame
/// border in pixels used to derive the §6.5.12 clamp range.
pub(crate) const BORDERINPIXELS: i32 = 160;

/// `MI_SIZE = 8` per §3 (`vp9-spec.txt` line 464). The smallest mode-info
/// block edge in pixels; the clamp edges are `MI`-units scaled by this.
pub(crate) const MI_SIZE: i32 = 8;

/// `MAX_MV_REF_CANDIDATES = 2` per §3 (`vp9-spec.txt` line 468). The
/// number of motion vectors `find_mv_refs( )` returns and the length of
/// the `RefListMv[ ]` list `find_best_ref_mvs( )` walks.
pub(crate) const MAX_MV_REF_CANDIDATES: usize = 2;

/// `Clip3( x, y, z )` per §4.6 (`vp9-spec.txt` lines 619-624): clamps `z`
/// into the inclusive `[x, y]` range (`x` low, `y` high).
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

/// The per-block frame geometry the §6.5 clamps read. Carries the §6.4
/// `MiRow` / `MiCol` block position, the §7.2 `MiRows` / `MiCols` frame
/// dimensions (both in `MI` units) and the §6.4.4 `MiSize` block-size
/// constant. Bundling them keeps the clamp primitives pure and lets a
/// later `find_mv_refs( )` driver construct one struct per block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MvRefGeometry {
    /// §6.4 `MiRow` — the block's top edge in `MI` units.
    pub(crate) mi_row: i32,
    /// §6.4 `MiCol` — the block's left edge in `MI` units.
    pub(crate) mi_col: i32,
    /// §7.2 `MiRows` — the frame height in `MI` units.
    pub(crate) mi_rows: i32,
    /// §7.2 `MiCols` — the frame width in `MI` units.
    pub(crate) mi_cols: i32,
    /// §6.4.4 `MiSize` — the §3 `BLOCK_*` constant for this block.
    pub(crate) mi_size: usize,
    /// §6.4.2 `MiColStart` — the current tile's left edge in `MI` units.
    pub(crate) mi_col_start: i32,
    /// §6.4.2 `MiColEnd` — the current tile's right edge in `MI` units.
    pub(crate) mi_col_end: i32,
}

impl MvRefGeometry {
    /// §6.5.4 `clamp_mv_row( mvec, border )` — clamp a row (component 0)
    /// motion-vector value into the frame's vertical range.
    ///
    /// ```text
    /// clamp_mv_row( mvec, border ) {
    ///   bh = num_8x8_blocks_high_lookup[ MiSize ]
    ///   mbToTopEdge = -((MiRow * MI_SIZE) * 8)
    ///   mbToBottomEdge = ((MiRows - bh - MiRow) * MI_SIZE) * 8
    ///   return Clip3( mbToTopEdge - border, mbToBottomEdge + border, mvec )
    /// }
    /// ```
    ///
    /// The edges are in eighth-pel units (`* MI_SIZE * 8`), matching the
    /// `Mv` representation, so `border` is added directly.
    pub(crate) fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[self.mi_size] as i32;
        let mb_to_top_edge = -((self.mi_row * MI_SIZE) * 8);
        let mb_to_bottom_edge = ((self.mi_rows - bh - self.mi_row) * MI_SIZE) * 8;
        clip3(mb_to_top_edge - border, mb_to_bottom_edge + border, mvec)
    }

    /// §6.5.5 `clamp_mv_col( mvec, border )` — clamp a col (component 1)
    /// motion-vector value into the frame's horizontal range.
    ///
    /// ```text
    /// clamp_mv_col( mvec, border ) {
    ///   bw = num_8x8_blocks_wide_lookup[ MiSize ]
    ///   mbToLeftEdge = -((MiCol * MI_SIZE) * 8)
    ///   mbToRightEdge = ((MiCols - bw - MiCol) * MI_SIZE) * 8
    ///   return Clip3( mbToLeftEdge - border, mbToRightEdge + border, mvec )
    /// }
    /// ```
    pub(crate) fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[self.mi_size] as i32;
        let mb_to_left_edge = -((self.mi_col * MI_SIZE) * 8);
        let mb_to_right_edge = ((self.mi_cols - bw - self.mi_col) * MI_SIZE) * 8;
        clip3(mb_to_left_edge - border, mb_to_right_edge + border, mvec)
    }

    /// §6.5.3 `clamp_mv_ref( i )` — clamp one `RefListMv[ i ]` entry with
    /// the §3 `MV_BORDER` border.
    ///
    /// ```text
    /// clamp_mv_ref( i ) {
    ///   RefListMv[ i ][ 0 ] = clamp_mv_row( RefListMv[ i ][ 0 ], MV_BORDER )
    ///   RefListMv[ i ][ 1 ] = clamp_mv_col( RefListMv[ i ][ 1 ], MV_BORDER )
    /// }
    /// ```
    ///
    /// Returns the clamped `[row, col]` pair; the §6.5.1 caller assigns it
    /// back into the candidate list.
    pub(crate) fn clamp_mv_ref(&self, mv: [i32; 2]) -> [i32; 2] {
        [
            self.clamp_mv_row(mv[0], MV_BORDER),
            self.clamp_mv_col(mv[1], MV_BORDER),
        ]
    }

    /// §6.5.2 `is_inside( candidateR, candidateC )` — whether a candidate
    /// mode-info position is accessible for motion-vector prediction.
    ///
    /// ```text
    /// is_inside( candidateR, candidateC ) {
    ///   return (candidateR >= 0 && candidateR < MiRows
    ///             && candidateC >= MiColStart && candidateC < MiColEnd)
    /// }
    /// ```
    ///
    /// Vertical motion across the frame's top/bottom edges is allowed
    /// (bounded by the whole-frame `MiRows`); horizontal motion is bounded
    /// by the current *tile* (`MiColStart` / `MiColEnd`), since crossing a
    /// tile column edge is prohibited.
    pub(crate) fn is_inside(&self, candidate_r: i32, candidate_c: i32) -> bool {
        candidate_r >= 0
            && candidate_r < self.mi_rows
            && candidate_c >= self.mi_col_start
            && candidate_c < self.mi_col_end
    }

    /// §6.5.12 `find_best_ref_mvs( refList )` — lowering-precision rounding
    /// and final clamp of the two `RefListMv[ ]` candidates, producing the
    /// `NearestMv` / `NearMv` / `BestMv` outputs.
    ///
    /// ```text
    /// find_best_ref_mvs( refList ) {
    ///   for ( i = 0; i < MAX_MV_REF_CANDIDATES; i++ ) {
    ///     deltaRow = RefListMv[ i ][ 0 ]
    ///     deltaCol = RefListMv[ i ][ 1 ]
    ///     if ( !allow_high_precision_mv || !use_mv_hp( RefListMv[ i ] ) ) {
    ///       if ( deltaRow & 1 ) deltaRow += (deltaRow > 0 ? -1 : 1)
    ///       if ( deltaCol & 1 ) deltaCol += (deltaCol > 0 ? -1 : 1)
    ///     }
    ///     RefListMv[ i ][ 0 ] = clamp_mv_row( deltaRow,
    ///                             (BORDERINPIXELS - INTERP_EXTEND) << 3 )
    ///     RefListMv[ i ][ 1 ] = clamp_mv_col( deltaCol,
    ///                             (BORDERINPIXELS - INTERP_EXTEND) << 3 )
    ///   }
    ///   NearestMv[ refList ] = RefListMv[ 0 ]
    ///   NearMv[ refList ]    = RefListMv[ 1 ]
    ///   BestMv[ refList ]    = RefListMv[ 0 ]
    /// }
    /// ```
    ///
    /// `ref_list_mv` is the §6.5.1 `RefListMv[ ]` candidate pair (two
    /// `[row, col]` vectors). When the third fractional (eighth-pel) bit is
    /// not in use — either the frame disallows high precision or the
    /// candidate is too large per §6.5.13 `use_mv_hp( )` — each odd
    /// component is rounded toward zero to a quarter-pel grid before the
    /// final wide clamp. Returns `[NearestMv, NearMv]`; `BestMv` equals
    /// `NearestMv` so the caller need not store it separately.
    pub(crate) fn find_best_ref_mvs(
        &self,
        mut ref_list_mv: [[i32; 2]; MAX_MV_REF_CANDIDATES],
        allow_high_precision_mv: bool,
    ) -> [[i32; 2]; MAX_MV_REF_CANDIDATES] {
        // (BORDERINPIXELS - INTERP_EXTEND) << 3 — the wide eighth-pel clamp
        // border the final clamp uses (vs MV_BORDER in clamp_mv_ref).
        let border = (BORDERINPIXELS - INTERP_EXTEND) << 3;
        for cand in ref_list_mv.iter_mut() {
            let mut delta_row = cand[0];
            let mut delta_col = cand[1];
            if !allow_high_precision_mv || !use_mv_hp(*cand) {
                if delta_row & 1 != 0 {
                    delta_row += if delta_row > 0 { -1 } else { 1 };
                }
                if delta_col & 1 != 0 {
                    delta_col += if delta_col > 0 { -1 } else { 1 };
                }
            }
            cand[0] = self.clamp_mv_row(delta_row, border);
            cand[1] = self.clamp_mv_col(delta_col, border);
        }
        ref_list_mv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generously-sized frame so the clamps don't bite unless the test
    /// drives a value past the edge on purpose: a 256x256-MI frame with
    /// a single tile spanning the full width, block BLOCK_8X8 at the
    /// origin.
    fn big_frame() -> MvRefGeometry {
        MvRefGeometry {
            mi_row: 0,
            mi_col: 0,
            mi_rows: 256,
            mi_cols: 256,
            mi_size: 3, // BLOCK_8X8: bw == bh == 1.
            mi_col_start: 0,
            mi_col_end: 256,
        }
    }

    #[test]
    fn is_inside_accepts_in_range_and_rejects_edges() {
        let g = MvRefGeometry {
            mi_row: 4,
            mi_col: 4,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: 3,
            mi_col_start: 2,
            mi_col_end: 6,
        };
        // Inside the tile column band and within the frame rows.
        assert!(g.is_inside(0, 2));
        assert!(g.is_inside(7, 5));
        // Row above the frame / at-or-below MiRows is rejected.
        assert!(!g.is_inside(-1, 4));
        assert!(!g.is_inside(8, 4));
        // Column before MiColStart / at-or-after MiColEnd is rejected,
        // even though it is within the frame width.
        assert!(!g.is_inside(4, 1));
        assert!(!g.is_inside(4, 6));
    }

    #[test]
    fn clamp_mv_row_passes_small_value_through() {
        // At the origin of a big frame the top edge is 0, bottom edge is
        // large; with MV_BORDER the range easily contains a small value.
        let g = big_frame();
        assert_eq!(g.clamp_mv_row(10, MV_BORDER), 10);
        assert_eq!(g.clamp_mv_col(-7, MV_BORDER), -7);
    }

    #[test]
    fn clamp_mv_row_clips_at_top_edge() {
        // MiRow == 0 => mbToTopEdge == 0, so the low bound is -border.
        // A value below -border clips up to -border.
        let g = big_frame();
        assert_eq!(g.clamp_mv_row(-(MV_BORDER + 50), MV_BORDER), -MV_BORDER);
        assert_eq!(g.clamp_mv_col(-(MV_BORDER + 50), MV_BORDER), -MV_BORDER);
    }

    #[test]
    fn clamp_mv_row_clips_at_bottom_edge() {
        // A 1x1-MI block (BLOCK_8X8) at the very bottom-right corner of a
        // 2x2-MI frame: MiRow == MiCol == 1, bw == bh == 1, MiRows ==
        // MiCols == 2 => mbToBottomEdge/RightEdge == 0, so the high bound
        // is +border.
        let g = MvRefGeometry {
            mi_row: 1,
            mi_col: 1,
            mi_rows: 2,
            mi_cols: 2,
            mi_size: 3,
            mi_col_start: 0,
            mi_col_end: 2,
        };
        assert_eq!(g.clamp_mv_row(MV_BORDER + 50, MV_BORDER), MV_BORDER);
        assert_eq!(g.clamp_mv_col(MV_BORDER + 50, MV_BORDER), MV_BORDER);
        // mbToTopEdge == -((1 * 8) * 8) == -64, low bound == -64-128.
        assert_eq!(
            g.clamp_mv_row(-(64 + MV_BORDER + 10), MV_BORDER),
            -(64 + MV_BORDER)
        );
    }

    #[test]
    fn clamp_mv_ref_applies_row_and_col_with_mv_border() {
        let g = big_frame();
        // Row clips at the top edge (-MV_BORDER), col passes through.
        let out = g.clamp_mv_ref([-(MV_BORDER + 100), 5]);
        assert_eq!(out, [-MV_BORDER, 5]);
    }

    #[test]
    fn find_best_ref_mvs_rounds_odd_components_toward_zero_when_no_hp() {
        // No high precision: odd eighth-pel components round toward zero.
        // 5 -> 4, -5 -> -4, 3 -> 2, -1 -> 0. Values are small enough to
        // pass the wide clamp unchanged.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[5, -5], [3, -1]], false);
        assert_eq!(out[0], [4, -4]);
        assert_eq!(out[1], [2, 0]);
    }

    #[test]
    fn find_best_ref_mvs_keeps_even_components() {
        // Even components are untouched by the rounding step.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[8, -16], [0, 2]], false);
        assert_eq!(out[0], [8, -16]);
        assert_eq!(out[1], [0, 2]);
    }

    #[test]
    fn find_best_ref_mvs_high_precision_small_keeps_odd_bit() {
        // allow_high_precision_mv && use_mv_hp([1, -3]) (both Abs>>3 == 0
        // < COMPANDED_MVREF_THRESH) => no rounding, odd bit preserved.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[1, -3], [0, 0]], true);
        assert_eq!(out[0], [1, -3]);
        assert_eq!(out[1], [0, 0]);
    }

    #[test]
    fn find_best_ref_mvs_high_precision_large_still_rounds() {
        // allow_high_precision_mv but the candidate is too large for
        // use_mv_hp (Abs(72) >> 3 == 9 >= 8) => the !use_mv_hp arm fires
        // and the odd row component rounds toward zero.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[73, 0], [0, 0]], true);
        assert_eq!(out[0], [72, 0]);
    }

    #[test]
    fn find_best_ref_mvs_final_clamp_uses_wide_border() {
        // The §6.5.12 clamp border is (BORDERINPIXELS - INTERP_EXTEND) <<
        // 3 == (160 - 4) << 3 == 1248, much wider than MV_BORDER. A
        // top-edge block (MiRow == 0) clips a large negative row at
        // -1248.
        let g = big_frame();
        let border = (BORDERINPIXELS - INTERP_EXTEND) << 3;
        assert_eq!(border, 1248);
        // Even value so no rounding; below -border clips to -border.
        let out = g.find_best_ref_mvs([[-(1248 + 16), 0], [0, 0]], false);
        assert_eq!(out[0], [-1248, 0]);
    }
}
