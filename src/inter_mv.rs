//! §8.5.2.1 / §8.5.2.2 / §8.5.2.3 motion-vector selection, clamping, and
//! scaling — the three steps that turn a decoded `BlockMvs[ ]` entry into
//! the `startX` / `startY` / `stepX` / `stepY` sampling inputs the
//! §8.5.2.4 [`crate::block_inter_pred::block_inter_predict`] leaf
//! consumes.
//!
//! These sit between the per-block motion-vector decode (§6.4.16-6.4.20,
//! which fills `BlockMvs[ refList ][ blockIdx ]`) and the sub-pixel
//! convolution leaf. The §8.5.2 driver runs them in order:
//!
//! 1. §8.5.2.1 [`select_mv`] — picks (and for sub-sampled chroma blocks
//!    *averages*, via [`round_mv_comp_q2`] / [`round_mv_comp_q4`]) the
//!    motion vector for the plane/block being predicted.
//! 2. §8.5.2.2 [`clamp_mv`] — converts the selected vector to the plane's
//!    sample precision (`(2 * mv) >> s`) and clamps it so the sub-pixel
//!    read window stays within the `INTERP_EXTEND` border of the frame.
//! 3. §8.5.2.3 [`scale_mv`] — folds in the reference-frame size ratio to
//!    produce the 1/16 th-sample `startX` / `startY` start position and
//!    the `stepX` / `stepY` per-sample step.
//!
//! Motion vectors are carried as `[ row, col ]` (`mv[ 0 ]` = vertical,
//! `mv[ 1 ]` = horizontal) in eighth-pel units throughout the decoder,
//! matching the §6.4.19 `read_mv( )` convention. The clamping step
//! doubles them to the 1/16 th-sample precision the scaling step and the
//! convolution leaf use.
//!
//! Everything here is a pure function of the caller-supplied geometry, so
//! each step is directly testable in isolation from frame-wide decode
//! state — mirroring the §8.5.2.4 leaf. The §8.5.2 driver that threads
//! `BlockMvs` / `FrameStore[ ]` reference planes through these steps and
//! writes the convolved result into `CurrFrame` is a later step.
//!
//! Single source of truth: `docs/video/vp9/vp9-spec.txt` §8.5.2.1-3.

// The §8.5.2 inter prediction driver that chains these three steps into
// the §8.5.2.4 leaf (and reads the reference planes from FrameStore[ ])
// lands in a later round; until then these primitives are reachable only
// from the unit tests, so the crate-internal `dead_code` lint is silenced
// module-wide (mirrors `block_inter_pred` / `mv` / `mode_info`'s deferred
// inter primitives).
#![allow(dead_code)]

use crate::block_inter_pred::{SUBPEL_BITS, SUBPEL_MASK};
use crate::mv_ref::MI_SIZE;
use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};
use crate::residual::BLOCK_8X8;

/// `REF_SCALE_SHIFT = 14` per spec §3 (`vp9-spec.txt` line 516). Number
/// of bits of precision when scaling reference frames; `xScale` / `yScale`
/// carry the reference-to-current size ratio in `1 << REF_SCALE_SHIFT`
/// fixed-point (so `1 << REF_SCALE_SHIFT` is a same-size reference).
pub(crate) const REF_SCALE_SHIFT: u32 = 14;

/// `SUBPEL_SHIFTS = 16` per spec §3 (`vp9-spec.txt` line 518). Equals
/// `1 << SUBPEL_BITS`; the number of sub-pixel phases, used as the
/// `spelRight` / `spelBottom` adjustment in §8.5.2.2 clamping.
pub(crate) const SUBPEL_SHIFTS: i32 = 1 << SUBPEL_BITS;

/// `INTERP_EXTEND = 4` per spec §3 (`vp9-spec.txt` line 521). Value used
/// when clipping motion vectors; the half-width of the 8-tap kernel that
/// must stay inside the reference border on each side.
pub(crate) const INTERP_EXTEND: i32 = 4;

/// `round_mv_comp_q2( value )` per §8.5.2.1 (`vp9-spec.txt` lines
/// 4569-4571). Divide-by-two with round-to-nearest-away-from-zero, used
/// to average two luma motion-vector components into a chroma component
/// when one axis is sub-sampled.
#[inline]
pub(crate) fn round_mv_comp_q2(value: i32) -> i32 {
    (if value < 0 { value - 1 } else { value + 1 }) / 2
}

/// `round_mv_comp_q4( value )` per §8.5.2.1 (`vp9-spec.txt` lines
/// 4574-4576). Divide-by-four with round-to-nearest-away-from-zero, used
/// to average four luma motion-vector components into a chroma component
/// when both axes are sub-sampled.
#[inline]
pub(crate) fn round_mv_comp_q4(value: i32) -> i32 {
    (if value < 0 { value - 2 } else { value + 2 }) / 4
}

/// §8.5.2.1 motion vector selection process (`vp9-spec.txt` lines
/// 4540-4580).
///
/// Selects the motion vector to use for the block being predicted. Luma
/// (`plane == 0`) and any block `>= BLOCK_8X8` use the per-`blockIdx`
/// vector directly. A sub-8x8 chroma block can straddle several luma 4x4
/// sub-blocks once a plane is sub-sampled, so its vector is the
/// rounded average of the covered luma vectors.
///
/// Inputs (all per §8.5.2.1):
/// * `block_mvs` — the `BlockMvs[ refList ]` row of four `[ row, col ]`
///   sub-block vectors for the active reference list.
/// * `plane` — 0 = luma, 1/2 = chroma.
/// * `mi_size` — `MiSize`, the §3 `BLOCK_*` constant.
/// * `block_idx` — how much of the block is already predicted, in 4x4
///   units (the §8.5.2 `blockIdx`).
/// * `subsampling_x` / `subsampling_y` — plane sub-sampling flags.
///
/// Returns the selected `[ row, col ]` motion vector.
pub(crate) fn select_mv(
    block_mvs: &[[i32; 2]; 4],
    plane: usize,
    mi_size: u8,
    block_idx: usize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> [i32; 2] {
    // §8.5.2.1: luma, or any block >= BLOCK_8X8, takes the vector as-is.
    if plane == 0 || mi_size >= BLOCK_8X8 {
        return block_mvs[block_idx];
    }
    match (subsampling_x, subsampling_y) {
        // No sub-sampling on this chroma plane — vector as-is.
        (false, false) => block_mvs[block_idx],
        // 4:4:0 (sx=0, sy=1): average the block and the one two rows down.
        (false, true) => {
            let a = block_mvs[block_idx];
            let b = block_mvs[block_idx + 2];
            [round_mv_comp_q2(a[0] + b[0]), round_mv_comp_q2(a[1] + b[1])]
        }
        // 4:2:2 (sx=1, sy=0): average the block and the one to its right.
        (true, false) => {
            let a = block_mvs[block_idx];
            let b = block_mvs[block_idx + 1];
            [round_mv_comp_q2(a[0] + b[0]), round_mv_comp_q2(a[1] + b[1])]
        }
        // 4:2:0 (sx=1, sy=1): average all four luma sub-block vectors.
        (true, true) => [
            round_mv_comp_q4(block_mvs[0][0] + block_mvs[1][0] + block_mvs[2][0] + block_mvs[3][0]),
            round_mv_comp_q4(block_mvs[0][1] + block_mvs[1][1] + block_mvs[2][1] + block_mvs[3][1]),
        ],
    }
}

/// `Clip3( low, high, x )` per spec §3 (`vp9-spec.txt` line 624).
#[inline]
fn clip3(low: i32, high: i32, x: i32) -> i32 {
    if x < low {
        low
    } else if x > high {
        high
    } else {
        x
    }
}

/// The grid geometry the §8.5.2.2 clamping step needs to know how far the
/// block sits from each frame edge.
///
/// All fields are the §6.4 mode-info coordinates: `mi_row` / `mi_col` are
/// the block's top-left position in 8x8 mode-info units, `mi_rows` /
/// `mi_cols` the frame size in the same units, and `mi_size` the §3
/// `BLOCK_*` size constant.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockGrid {
    pub mi_row: i32,
    pub mi_col: i32,
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub mi_size: u8,
}

/// §8.5.2.2 motion vector clamping process (`vp9-spec.txt` lines
/// 4581-4617).
///
/// Converts the selected eighth-pel `[ row, col ]` vector to the plane's
/// 1/16 th-sample precision (`(2 * mv) >> s`) and clamps it so the
/// sub-pixel sample window stays within `INTERP_EXTEND` of the frame edge.
///
/// Inputs (all per §8.5.2.2):
/// * `grid` — the block's mode-info geometry.
/// * `mv` — the §8.5.2.1-selected `[ row, col ]` motion vector.
/// * `plane` — 0 = luma (no sub-sampling), 1/2 = chroma.
/// * `subsampling_x` / `subsampling_y` — chroma plane sub-sampling.
///
/// Returns the clamped `[ row, col ]` vector in 1/16 th-sample units.
pub(crate) fn clamp_mv(
    grid: &BlockGrid,
    mv: [i32; 2],
    plane: usize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> [i32; 2] {
    // §8.5.2.2: luma has no sub-sampling; chroma uses the plane's flags.
    let sx = if plane == 0 { 0 } else { subsampling_x as u32 };
    let sy = if plane == 0 { 0 } else { subsampling_y as u32 };

    let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[grid.mi_size as usize] as i32;
    let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[grid.mi_size as usize] as i32;

    // Distance from each frame edge, in 1/16 th-sample units, shifted to
    // the plane precision.
    let mb_to_top_edge = -((grid.mi_row * MI_SIZE) * 16) >> sy;
    let mb_to_bottom_edge = (((grid.mi_rows - bh - grid.mi_row) * MI_SIZE) * 16) >> sy;
    let mb_to_left_edge = -((grid.mi_col * MI_SIZE) * 16) >> sx;
    let mb_to_right_edge = (((grid.mi_cols - bw - grid.mi_col) * MI_SIZE) * 16) >> sx;

    // Sub-pixel margin the 8-tap window needs on each side.
    let spel_left = (INTERP_EXTEND + ((bw * MI_SIZE) >> sx)) << SUBPEL_BITS;
    let spel_right = spel_left - SUBPEL_SHIFTS;
    let spel_top = (INTERP_EXTEND + ((bh * MI_SIZE) >> sy)) << SUBPEL_BITS;
    let spel_bottom = spel_top - SUBPEL_SHIFTS;

    [
        clip3(
            mb_to_top_edge - spel_top,
            mb_to_bottom_edge + spel_bottom,
            (2 * mv[0]) >> sy,
        ),
        clip3(
            mb_to_left_edge - spel_left,
            mb_to_right_edge + spel_right,
            (2 * mv[1]) >> sx,
        ),
    ]
}

/// The reference-frame / current-frame geometry the §8.5.2.3 scaling step
/// needs to fold the inter-frame size ratio into the start position.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaleGeom {
    /// `RefFrameWidth[ refIdx ]` — the reference frame width in luma
    /// samples.
    pub ref_frame_width: i32,
    /// `RefFrameHeight[ refIdx ]` — the reference frame height in luma
    /// samples.
    pub ref_frame_height: i32,
    /// `FrameWidth` — the current frame width in luma samples.
    pub frame_width: i32,
    /// `FrameHeight` — the current frame height in luma samples.
    pub frame_height: i32,
    /// `subsampling_x` of the stream (used to recover the luma location
    /// for chroma planes).
    pub subsampling_x: bool,
    /// `subsampling_y` of the stream.
    pub subsampling_y: bool,
}

/// The four §8.5.2.3 outputs: the 1/16 th-sample start position and the
/// per-sample step, ready for the §8.5.2.4 leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScaledMv {
    pub start_x: i32,
    pub start_y: i32,
    pub step_x: i32,
    pub step_y: i32,
}

/// §8.5.2.3 motion vector scaling process (`vp9-spec.txt` lines
/// 4618-4681).
///
/// Computes the reference sampling location and step, compensating for any
/// difference between the reference-frame and current-frame sizes. Even
/// for chroma the fractional start position is derived from the luma block
/// location (`lumaX` / `lumaY`), per the §8.5.2.3 NOTE.
///
/// Inputs (all per §8.5.2.3):
/// * `geom` — the reference / current frame sizes and stream sub-sampling.
/// * `plane` — 0 = luma, 1/2 = chroma.
/// * `x` / `y` — the top-left location of the region in
///   `CurrFrame[ plane ]` (plane-relative samples).
/// * `clamped_mv` — the §8.5.2.2 output, in 1/16 th-sample units.
///
/// Returns the `startX` / `startY` / `stepX` / `stepY` outputs.
pub(crate) fn scale_mv(
    geom: &ScaleGeom,
    plane: usize,
    x: i32,
    y: i32,
    clamped_mv: [i32; 2],
) -> ScaledMv {
    // xScale / yScale: reference size relative to current, in
    // 1 << REF_SCALE_SHIFT fixed-point.
    let x_scale = (geom.ref_frame_width << REF_SCALE_SHIFT) / geom.frame_width;
    let y_scale = (geom.ref_frame_height << REF_SCALE_SHIFT) / geom.frame_height;

    // baseX / baseY: where a zero motion vector lands in the reference.
    let base_x = (x * x_scale) >> REF_SCALE_SHIFT;
    let base_y = (y * y_scale) >> REF_SCALE_SHIFT;

    // lumaX / lumaY: block location in luma samples (the fractional part
    // is always luma-based, even for chroma — §8.5.2.3 NOTE).
    let luma_x = if plane > 0 {
        x << (geom.subsampling_x as u32)
    } else {
        x
    };
    let luma_y = if plane > 0 {
        y << (geom.subsampling_y as u32)
    } else {
        y
    };

    let frac_x = ((16 * luma_x * x_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK;
    let frac_y = ((16 * luma_y * y_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK;

    // dX / dY: the scaled motion vector (clampedMv[1] is the col axis).
    let d_x = ((clamped_mv[1] * x_scale) >> REF_SCALE_SHIFT) + frac_x;
    let d_y = ((clamped_mv[0] * y_scale) >> REF_SCALE_SHIFT) + frac_y;

    ScaledMv {
        step_x: (16 * x_scale) >> REF_SCALE_SHIFT,
        step_y: (16 * y_scale) >> REF_SCALE_SHIFT,
        start_x: (base_x << SUBPEL_BITS) + d_x,
        start_y: (base_y << SUBPEL_BITS) + d_y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{BLOCK_16X16, BLOCK_4X4};

    // ---- §8.5.2.1 round_mv_comp helpers ---------------------------------

    #[test]
    fn round_mv_comp_q2_rounds_away_from_zero() {
        // (v + 1) / 2 for v >= 0, (v - 1) / 2 for v < 0.
        assert_eq!(round_mv_comp_q2(0), 0); // (0+1)/2 = 0
        assert_eq!(round_mv_comp_q2(1), 1); // (1+1)/2 = 1
        assert_eq!(round_mv_comp_q2(2), 1); // (2+1)/2 = 1
        assert_eq!(round_mv_comp_q2(3), 2); // (3+1)/2 = 2
        assert_eq!(round_mv_comp_q2(-1), -1); // (-1-1)/2 = -1
        assert_eq!(round_mv_comp_q2(-2), -1); // (-2-1)/2 = -1 (trunc)
        assert_eq!(round_mv_comp_q2(-3), -2); // (-3-1)/2 = -2
    }

    #[test]
    fn round_mv_comp_q4_rounds_away_from_zero() {
        assert_eq!(round_mv_comp_q4(0), 0); // (0+2)/4 = 0
        assert_eq!(round_mv_comp_q4(2), 1); // (2+2)/4 = 1
        assert_eq!(round_mv_comp_q4(5), 1); // (5+2)/4 = 1
        assert_eq!(round_mv_comp_q4(6), 2); // (6+2)/4 = 2
        assert_eq!(round_mv_comp_q4(-2), -1); // (-2-2)/4 = -1
        assert_eq!(round_mv_comp_q4(-5), -1); // (-5-2)/4 = -1 (trunc)
        assert_eq!(round_mv_comp_q4(-6), -2); // (-6-2)/4 = -2
    }

    // ---- §8.5.2.1 select_mv ---------------------------------------------

    #[test]
    fn select_mv_luma_takes_block_idx_directly() {
        let mvs = [[10, 20], [30, 40], [50, 60], [70, 80]];
        // Luma plane, sub-8x8: still the per-blockIdx vector.
        assert_eq!(select_mv(&mvs, 0, BLOCK_4X4, 2, true, true), [50, 60]);
    }

    #[test]
    fn select_mv_large_block_takes_block_idx_directly() {
        let mvs = [[10, 20], [30, 40], [50, 60], [70, 80]];
        // >= BLOCK_8X8 chroma: vector as-is regardless of sub-sampling.
        assert_eq!(select_mv(&mvs, 1, BLOCK_16X16, 0, true, true), [10, 20]);
    }

    #[test]
    fn select_mv_chroma_no_subsampling_is_direct() {
        let mvs = [[10, 20], [30, 40], [50, 60], [70, 80]];
        assert_eq!(select_mv(&mvs, 1, BLOCK_4X4, 1, false, false), [30, 40]);
    }

    #[test]
    fn select_mv_chroma_440_averages_vertical_pair() {
        // sx=0, sy=1: average blockIdx and blockIdx+2.
        let mvs = [[10, 20], [30, 40], [50, 60], [70, 80]];
        // blockIdx 0: avg([10,20],[50,60]) = [q2(60), q2(80)] = [30, 40].
        assert_eq!(select_mv(&mvs, 2, BLOCK_4X4, 0, false, true), [30, 40]);
    }

    #[test]
    fn select_mv_chroma_422_averages_horizontal_pair() {
        // sx=1, sy=0: average blockIdx and blockIdx+1.
        let mvs = [[10, 20], [30, 40], [50, 60], [70, 80]];
        // blockIdx 0: avg([10,20],[30,40]) = [q2(40), q2(60)] = [20, 30].
        assert_eq!(select_mv(&mvs, 1, BLOCK_4X4, 0, true, false), [20, 30]);
    }

    #[test]
    fn select_mv_chroma_420_averages_all_four() {
        // sx=1, sy=1: average all four sub-blocks via round_mv_comp_q4.
        let mvs = [[10, 20], [30, 40], [50, 60], [70, 80]];
        // rows sum 160 -> q4(160) = (160+2)/4 = 40; cols sum 200 -> 50.
        assert_eq!(select_mv(&mvs, 1, BLOCK_4X4, 0, true, true), [40, 50]);
    }

    // ---- §8.5.2.2 clamp_mv ----------------------------------------------

    #[test]
    fn clamp_mv_interior_block_doubles_to_sixteenth_pel() {
        // A block well inside a large frame: the clamp bounds are wide so
        // the only transform is (2 * mv) >> s. Luma, no sub-sampling.
        let grid = BlockGrid {
            mi_row: 10,
            mi_col: 10,
            mi_rows: 40,
            mi_cols: 40,
            mi_size: BLOCK_8X8,
        };
        // Small motion vector, far from any edge.
        assert_eq!(clamp_mv(&grid, [3, -5], 0, false, false), [6, -10]);
    }

    #[test]
    fn clamp_mv_clamps_against_top_left_corner() {
        // Block at the very top-left (mi_row = mi_col = 0). A large
        // negative vector is clamped to mbToTopEdge - spelTop /
        // mbToLeftEdge - spelLeft, with mbTo*Edge = 0 there.
        let grid = BlockGrid {
            mi_row: 0,
            mi_col: 0,
            mi_rows: 40,
            mi_cols: 40,
            mi_size: BLOCK_8X8,
        };
        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[BLOCK_8X8 as usize] as i32;
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[BLOCK_8X8 as usize] as i32;
        let spel_left = (INTERP_EXTEND + (bw * MI_SIZE)) << SUBPEL_BITS;
        let spel_top = (INTERP_EXTEND + (bh * MI_SIZE)) << SUBPEL_BITS;
        // mbToTopEdge / mbToLeftEdge are 0 at the corner.
        let out = clamp_mv(&grid, [-100000, -100000], 0, false, false);
        assert_eq!(out, [-spel_top, -spel_left]);
    }

    #[test]
    fn clamp_mv_chroma_subsampling_shifts_bounds_and_vector() {
        // 4:2:0 chroma: sx = sy = 1. (2 * mv) >> 1 = mv for the unclamped
        // path; verify an interior block returns the raw mv unchanged.
        let grid = BlockGrid {
            mi_row: 10,
            mi_col: 10,
            mi_rows: 40,
            mi_cols: 40,
            mi_size: BLOCK_8X8,
        };
        // (2 * 7) >> 1 = 7, (2 * -9) >> 1 = -9 — interior, no clamp.
        assert_eq!(clamp_mv(&grid, [7, -9], 1, true, true), [7, -9]);
    }

    // ---- §8.5.2.3 scale_mv ----------------------------------------------

    #[test]
    fn scale_mv_unscaled_reference_is_identity_step() {
        // Reference same size as current frame -> xScale = yScale =
        // 1 << REF_SCALE_SHIFT, so stepX = stepY = 16 and baseX/Y = x/y.
        let geom = ScaleGeom {
            ref_frame_width: 320,
            ref_frame_height: 240,
            frame_width: 320,
            frame_height: 240,
            subsampling_x: false,
            subsampling_y: false,
        };
        // Zero motion vector, luma, block at (16, 8).
        let out = scale_mv(&geom, 0, 16, 8, [0, 0]);
        assert_eq!(out.step_x, 16);
        assert_eq!(out.step_y, 16);
        // baseX = 16, baseY = 8; startX = 16 << 4, startY = 8 << 4 (frac 0).
        assert_eq!(out.start_x, 16 << SUBPEL_BITS);
        assert_eq!(out.start_y, 8 << SUBPEL_BITS);
    }

    #[test]
    fn scale_mv_unscaled_reference_carries_mv_into_start() {
        // Same-size reference: dX = clampedMv[1], dY = clampedMv[0]
        // (xScale = 1<<shift so the >> cancels; fracX/Y = 0 at zero shift).
        let geom = ScaleGeom {
            ref_frame_width: 100,
            ref_frame_height: 100,
            frame_width: 100,
            frame_height: 100,
            subsampling_x: false,
            subsampling_y: false,
        };
        // clampedMv [row=5, col=-3]; block at origin.
        let out = scale_mv(&geom, 0, 0, 0, [5, -3]);
        assert_eq!(out.start_x, -3); // baseX 0, dX = col
        assert_eq!(out.start_y, 5); // baseY 0, dY = row
    }

    #[test]
    fn scale_mv_half_size_reference_halves_step_and_base() {
        // Reference half the size of the current frame -> xScale =
        // (160 << 14) / 320 = 1 << 13, so stepX = (16 << 13) >> 14 = 8.
        let geom = ScaleGeom {
            ref_frame_width: 160,
            ref_frame_height: 120,
            frame_width: 320,
            frame_height: 240,
            subsampling_x: false,
            subsampling_y: false,
        };
        let out = scale_mv(&geom, 0, 32, 16, [0, 0]);
        assert_eq!(out.step_x, 8);
        assert_eq!(out.step_y, 8);
        // baseX = (32 * (1<<13)) >> 14 = 16; baseY = 8.
        // fracX = (16*32*(1<<13))>>14 & 15 = (16*16) & 15 = 0.
        assert_eq!(out.start_x, 16 << SUBPEL_BITS);
        assert_eq!(out.start_y, 8 << SUBPEL_BITS);
    }

    #[test]
    fn scale_mv_chroma_frac_uses_luma_location() {
        // Chroma plane with sub-sampling: lumaX = x << subsampling_x.
        // Use a half-size reference so the fractional part is non-trivial,
        // and confirm the luma-based recovery by comparing the explicit
        // formula.
        let geom = ScaleGeom {
            ref_frame_width: 160,
            ref_frame_height: 160,
            frame_width: 320,
            frame_height: 320,
            subsampling_x: true,
            subsampling_y: true,
        };
        let x = 5;
        let y = 7;
        let clamped = [11, -13];
        let out = scale_mv(&geom, 1, x, y, clamped);

        // Recompute independently using the spec formulas.
        let x_scale = (160 << REF_SCALE_SHIFT) / 320;
        let y_scale = (160 << REF_SCALE_SHIFT) / 320;
        let base_x = (x * x_scale) >> REF_SCALE_SHIFT;
        let base_y = (y * y_scale) >> REF_SCALE_SHIFT;
        let luma_x = x << 1;
        let luma_y = y << 1;
        let frac_x = ((16 * luma_x * x_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK;
        let frac_y = ((16 * luma_y * y_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK;
        let d_x = ((clamped[1] * x_scale) >> REF_SCALE_SHIFT) + frac_x;
        let d_y = ((clamped[0] * y_scale) >> REF_SCALE_SHIFT) + frac_y;
        assert_eq!(out.start_x, (base_x << SUBPEL_BITS) + d_x);
        assert_eq!(out.start_y, (base_y << SUBPEL_BITS) + d_y);
        assert_eq!(out.step_x, 8);
        assert_eq!(out.step_y, 8);
    }
}
