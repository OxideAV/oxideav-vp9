//! §8.5.2 inter prediction process — the driver that chains the
//! §8.5.2.1 motion-vector selection, §8.5.2.2 clamping, §8.5.2.3
//! scaling and §8.5.2.4 block inter prediction steps into a region
//! prediction written back into `CurrFrame`.
//!
//! This is the composition the §6.4.21 `residual( )` walk's
//! `predict_inter( plane, x, y, w, h, blockIdx )` call site invokes
//! for inter-coded blocks. The four numbered sub-steps were landed as
//! pure primitives in earlier rounds:
//!
//! 1. §8.5.2.1 [`crate::inter_mv::select_mv`] — picks (and, for
//!    sub-sampled sub-8x8 chroma, averages) the per-`blockIdx` motion
//!    vector for the active reference list.
//! 2. §8.5.2.2 [`crate::inter_mv::clamp_mv`] — converts the selected
//!    vector to plane precision and clamps it against the
//!    `INTERP_EXTEND` border.
//! 3. §8.5.2.3 [`crate::inter_mv::scale_mv`] — folds in the
//!    reference-frame size ratio to produce the 1/16 th-sample
//!    `startX` / `startY` start position and `stepX` / `stepY` step.
//! 4. §8.5.2.4 [`crate::block_inter_pred::block_inter_predict`] — the
//!    two-pass 8-tap sub-pixel convolution leaf.
//!
//! The driver per §8.5.2 (`vp9-spec.txt` lines 4505-4539):
//!
//! ```text
//! isCompound = ref_frame[ 1 ] > NONE
//! refList = 0
//! mv      = select_mv( plane, refList, blockIdx )            §8.5.2.1
//! clamped = clamp_mv( plane, mv )                            §8.5.2.2
//! (startX, startY, stepX, stepY) = scale_mv( plane, refList, x, y, clamped )  §8.5.2.3
//! preds[ 0 ] = block_inter_predict( plane, refList, … )      §8.5.2.4
//! if ( isCompound ) repeat steps 2-5 with refList = 1 -> preds[ 1 ]
//! if ( isCompound == 0 ) CurrFrame[ plane ][ y+i ][ x+j ] = preds[ 0 ][ i ][ j ]
//! else                   CurrFrame[ plane ][ y+i ][ x+j ] = Round2( preds[0][i][j] + preds[1][i][j], 1 )
//! ```
//!
//! The reference-frame contents (`ref = FrameStore[ refIdx ]`) and the
//! reference-frame geometry (`RefFrameWidth[ refIdx ]` etc.) are
//! supplied by the caller through [`RefPlanes`], built from the §8.10
//! reference-buffer state ([`crate::ref_buffer`]). `refIdx` itself is
//! resolved by the caller from `ref_frame_idx[ ref_frame[ refList ] -
//! LAST_FRAME ]` (§8.5.2.3 / §8.5.2.4 line 4635 / 4692); this driver
//! consumes the already-resolved per-`refList` reference planes.
//!
//! Everything here is a pure function of the caller-supplied geometry
//! and reference planes, so the whole §8.5.2 chain is directly testable
//! against a hand-built reference frame without threading the full
//! frame decode.
//!
//! Single source of truth: `docs/video/vp9/vp9-spec.txt` §8.5.2.

// The §6.4.21 residual( ) inter arm that invokes this driver lands on
// top in the same round; until the full inter-frame decode path is
// wired this primitive is reachable only from the unit tests, so the
// crate-internal `dead_code` lint is silenced module-wide (mirrors the
// §8.5.2.1-4 primitive modules).
#![allow(dead_code)]

use crate::block_inter_pred::block_inter_predict;
use crate::inter_mv::{clamp_mv, scale_mv, select_mv, BlockGrid, ScaleGeom};
use crate::intra::Plane;

/// `Round2( x, n )` per spec §3 (`vp9-spec.txt` line 636). The §8.5.2
/// compound average uses `Round2( p0 + p1, 1 )`.
#[inline]
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// One reference frame's plane contents + geometry for a single
/// reference list, as the §8.5.2.4 step consumes them.
///
/// `samples` is the reference plane in row-major order with stride
/// `stride`; `width` / `height` are the plane-relative extents the
/// caller stored. `ref_frame_width` / `ref_frame_height` are the
/// reference frame's *luma* dimensions (`RefFrameWidth[ refIdx ]` /
/// `RefFrameHeight[ refIdx ]`) — the §8.5.2.4 `lastX` / `lastY` and the
/// §8.5.2.3 `xScale` / `yScale` are derived from these, not from the
/// plane extents.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RefPlane<'a> {
    /// `FrameStore[ refIdx ][ plane ]` row-major samples.
    pub samples: &'a [i32],
    /// Row stride of `samples` (>= the plane width).
    pub stride: usize,
    /// `RefFrameWidth[ refIdx ]` — reference frame width in luma samples.
    pub ref_frame_width: i32,
    /// `RefFrameHeight[ refIdx ]` — reference frame height in luma
    /// samples.
    pub ref_frame_height: i32,
}

impl RefPlane<'_> {
    /// Edge-clamped reference read `ref[ row ][ col ]`. The §8.5.2.4
    /// process clamps `row` / `col` with `Clip3( 0, lastY/lastX, … )`
    /// before reaching this accessor, so the index is always in range;
    /// the `min` is a defensive guard mirroring that clamp.
    #[inline]
    fn sample(&self, row: i32, col: i32) -> i32 {
        let r = row.max(0) as usize;
        let c = col.max(0) as usize;
        self.samples[r * self.stride + c]
    }
}

/// The two reference lists' planes (`preds[ 0 ]` / `preds[ 1 ]`).
///
/// `list[ 1 ]` is `None` for single-reference blocks (`isCompound ==
/// 0`); the §8.5.2 driver only forms `preds[ 1 ]` when it is `Some`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RefPlanes<'a> {
    pub list: [Option<RefPlane<'a>>; 2],
}

/// The per-call inputs the §8.5.2 driver needs that are not the
/// reference planes or the §8.5.2.1-3 geometry: the predicted region
/// location / size and the §6.4 mode-info products.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InterPredArgs {
    /// `plane` — 0 = luma, 1/2 = chroma.
    pub plane: usize,
    /// `x` / `y` — top-left sample of the region in `CurrFrame[ plane ]`.
    pub x: i32,
    pub y: i32,
    /// `w` / `h` — region width / height in samples.
    pub w: usize,
    pub h: usize,
    /// `blockIdx` — how much of the block is already predicted, in 4x4
    /// units (the §8.5.2 / §8.5.2.1 `blockIdx`).
    pub block_idx: usize,
    /// `interp_filter` — 0..3 outer index into the §8.5.2.4 sub-pixel
    /// filter table.
    pub interp_filter: usize,
    /// `BitDepth` (8, 10 or 12) for the §8.5.2.4 `Clip1( )` range.
    pub bit_depth: u32,
    /// Whether `ref_frame[ 1 ] > NONE` — the §8.5.2 `isCompound`.
    pub is_compound: bool,
}

/// Run the §8.5.2 inter prediction process for one region of one
/// plane, writing the predicted samples into `dst`.
///
/// `dst` is `CurrFrame[ plane ]`; the predicted `w × h` region is
/// written at `dst[ y + i ][ x + j ]`. `block_mvs[ refList ]` is the
/// §6.4.19 `BlockMvs[ refList ]` row of four `[ row, col ]` eighth-pel
/// sub-block vectors. `grid` / `geom` supply the §8.5.2.2 / §8.5.2.3
/// geometry; `refs` carries the per-`refList` reference planes resolved
/// from §8.10 state.
///
/// # Panics
///
/// Debug-asserts that a compound block supplies both reference planes
/// and that a single-reference block supplies `refs.list[ 0 ]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_inter(
    dst: &mut Plane,
    args: &InterPredArgs,
    grid: &BlockGrid,
    geom: &ScaleGeom,
    block_mvs: &[[[i32; 2]; 4]; 2],
    refs: &RefPlanes<'_>,
    subsampling_x: bool,
    subsampling_y: bool,
) {
    let n_lists = if args.is_compound { 2 } else { 1 };

    // preds[ refList ] — each a flattened w×h prediction block.
    let mut preds: [Vec<i32>; 2] = [Vec::new(), Vec::new()];

    for (ref_list, pred_slot) in preds.iter_mut().enumerate().take(n_lists) {
        let ref_plane = refs.list[ref_list].expect("missing reference plane for active refList");

        // §8.5.2 step 2 — §8.5.2.1 motion vector selection.
        let mv = select_mv(
            &block_mvs[ref_list],
            args.plane,
            grid.mi_size,
            args.block_idx,
            subsampling_x,
            subsampling_y,
        );

        // §8.5.2 step 3 — §8.5.2.2 clamping (to plane precision).
        let clamped = clamp_mv(grid, mv, args.plane, subsampling_x, subsampling_y);

        // §8.5.2 step 4 — §8.5.2.3 scaling, against this refList's
        // reference-frame geometry.
        let ref_geom = ScaleGeom {
            ref_frame_width: ref_plane.ref_frame_width,
            ref_frame_height: ref_plane.ref_frame_height,
            frame_width: geom.frame_width,
            frame_height: geom.frame_height,
            subsampling_x: geom.subsampling_x,
            subsampling_y: geom.subsampling_y,
        };
        let scaled = scale_mv(&ref_geom, args.plane, args.x, args.y, clamped);

        // §8.5.2.4 lastX / lastY: bottom-right reference sample of this
        // plane. subX / subY are 0 for luma, the plane sub-sampling
        // otherwise.
        let sub_x = if args.plane == 0 {
            0u32
        } else {
            u32::from(subsampling_x)
        };
        let sub_y = if args.plane == 0 {
            0u32
        } else {
            u32::from(subsampling_y)
        };
        let last_x = ((ref_plane.ref_frame_width + sub_x as i32) >> sub_x) - 1;
        let last_y = ((ref_plane.ref_frame_height + sub_y as i32) >> sub_y) - 1;

        // §8.5.2 step 5 — §8.5.2.4 block inter prediction.
        *pred_slot = block_inter_predict(
            |row, col| ref_plane.sample(row, col),
            scaled.start_x,
            scaled.start_y,
            scaled.step_x,
            scaled.step_y,
            args.w,
            args.h,
            args.interp_filter,
            last_x,
            last_y,
            args.bit_depth,
        );
    }

    // §8.5.2 derivation of the inter predicted samples.
    //
    // The store is clipped at the allocated working extent for the same
    // reason as the §8.5.1 / §8.6.2 stores (see `predict_intra` /
    // `reconstruct_block`): a frame-edge block the §6.4.3 `hasRows` /
    // `hasCols` rules admit may overhang the MiCols*8 x MiRows*8 working
    // planes into (unobservable) superblock padding a spec CurrFrame
    // would carry. Fuzz/corpus-found on a 176x144 stream whose bottom
    // superblock row codes 64-tall inter leaves overhanging the frame.
    let (px, py) = (args.x as usize, args.y as usize);
    for i in 0..args.h {
        if py + i >= dst.height() {
            break;
        }
        for j in 0..args.w {
            if px + j >= dst.width() {
                break;
            }
            let value = if args.is_compound {
                round2(preds[0][i * args.w + j] + preds[1][i * args.w + j], 1)
            } else {
                preds[0][i * args.w + j]
            };
            dst.set(px + j, py + i, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::{BLOCK_64X64, BLOCK_8X8};

    /// Build a `w×h` reference plane with a value that stays inside the
    /// 8-bit `Clip1` range (`0..=255`) yet is unique within the small
    /// region the tests sample, so an integer-position copy is
    /// identifiable: `value = ((row & 15) << 4) | (col & 15)`.
    fn ramp_value(row: usize, col: usize) -> i32 {
        (((row & 15) << 4) | (col & 15)) as i32
    }

    fn ramp_plane(w: usize, h: usize) -> Vec<i32> {
        let mut v = vec![0i32; w * h];
        for r in 0..h {
            for c in 0..w {
                v[r * w + c] = ramp_value(r, c);
            }
        }
        v
    }

    /// A zero motion vector, same-size reference, integer block
    /// location, luma plane, phase-0 filter: the §8.5.2 chain reduces to
    /// a straight copy of the co-located reference region into
    /// `CurrFrame`.
    #[test]
    fn zero_mv_same_size_reference_is_a_plain_copy() {
        let (rw, rh) = (64usize, 64usize);
        let refp = ramp_plane(rw, rh);
        let mut dst = Plane::new(64, 64);

        let grid = BlockGrid {
            mi_row: 1,
            mi_col: 2,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: BLOCK_8X8,
        };
        let geom = ScaleGeom {
            ref_frame_width: 64,
            ref_frame_height: 64,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: false,
            subsampling_y: false,
        };
        // Predict an 8×8 region at (16, 8) with a zero MV.
        let args = InterPredArgs {
            plane: 0,
            x: 16,
            y: 8,
            w: 8,
            h: 8,
            block_idx: 0,
            interp_filter: 0,
            bit_depth: 8,
            is_compound: false,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp,
                    stride: rw,
                    ref_frame_width: 64,
                    ref_frame_height: 64,
                }),
                None,
            ],
        };
        let block_mvs = [[[0, 0]; 4]; 2];
        predict_inter(
            &mut dst, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );

        for i in 0..8 {
            for j in 0..8 {
                // Co-located reference sample at (8 + i, 16 + j).
                let expected = ramp_value(8 + i, 16 + j);
                assert_eq!(dst.get(16 + j, 8 + i), expected, "i{i} j{j}");
            }
        }
    }

    /// An integer (eighth-pel multiple-of-8) motion vector shifts the
    /// sampled reference region. MV row/col are in eighth-pel; the
    /// §8.5.2.2 clamp doubles to 1/16 th, so a value of 8 (one luma
    /// sample) lands `start += 16` = one integer sample.
    #[test]
    fn integer_mv_shifts_sampled_region() {
        let (rw, rh) = (64usize, 64usize);
        let refp = ramp_plane(rw, rh);
        let mut dst = Plane::new(64, 64);

        let grid = BlockGrid {
            mi_row: 2,
            mi_col: 2,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: BLOCK_8X8,
        };
        let geom = ScaleGeom {
            ref_frame_width: 64,
            ref_frame_height: 64,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: false,
            subsampling_y: false,
        };
        // MV = [row=8, col=16] eighth-pel = (1 sample down, 2 right).
        let block_mvs = [[[8, 16]; 4]; 2];
        let args = InterPredArgs {
            plane: 0,
            x: 16,
            y: 16,
            w: 8,
            h: 8,
            block_idx: 0,
            interp_filter: 0,
            bit_depth: 8,
            is_compound: false,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp,
                    stride: rw,
                    ref_frame_width: 64,
                    ref_frame_height: 64,
                }),
                None,
            ],
        };
        predict_inter(
            &mut dst, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );

        for i in 0..8 {
            for j in 0..8 {
                // Region origin (16, 16) shifted by (+1 row, +2 col).
                let expected = ramp_value(16 + 1 + i, 16 + 2 + j);
                assert_eq!(dst.get(16 + j, 16 + i), expected, "i{i} j{j}");
            }
        }
    }

    /// Compound prediction averages the two reference predictions with
    /// `Round2( p0 + p1, 1 )`. Two flat reference planes at distinct
    /// values produce the rounded mean everywhere.
    #[test]
    fn compound_averages_two_references() {
        let (rw, rh) = (32usize, 32usize);
        let refp0 = vec![100i32; rw * rh];
        let refp1 = vec![151i32; rw * rh];
        let mut dst = Plane::new(32, 32);

        let grid = BlockGrid {
            mi_row: 1,
            mi_col: 1,
            mi_rows: 4,
            mi_cols: 4,
            mi_size: BLOCK_8X8,
        };
        let geom = ScaleGeom {
            ref_frame_width: 32,
            ref_frame_height: 32,
            frame_width: 32,
            frame_height: 32,
            subsampling_x: false,
            subsampling_y: false,
        };
        let args = InterPredArgs {
            plane: 0,
            x: 8,
            y: 8,
            w: 8,
            h: 8,
            block_idx: 0,
            interp_filter: 0,
            bit_depth: 8,
            is_compound: true,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp0,
                    stride: rw,
                    ref_frame_width: 32,
                    ref_frame_height: 32,
                }),
                Some(RefPlane {
                    samples: &refp1,
                    stride: rw,
                    ref_frame_width: 32,
                    ref_frame_height: 32,
                }),
            ],
        };
        let block_mvs = [[[0, 0]; 4]; 2];
        predict_inter(
            &mut dst, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );

        // Round2( 100 + 151, 1 ) = (251 + 1) >> 1 = 126.
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(dst.get(8 + j, 8 + i), 126, "i{i} j{j}");
            }
        }
    }

    /// A flat reference plane passes through unchanged at any sub-pixel
    /// phase, because each §8.5.2.4 kernel sums to 128 — verifying the
    /// driver threads a half-pel MV through select/clamp/scale/predict
    /// without distortion on flat content.
    #[test]
    fn flat_reference_passes_through_at_half_pel() {
        let (rw, rh) = (64usize, 64usize);
        let refp = vec![137i32; rw * rh];
        let mut dst = Plane::new(64, 64);

        let grid = BlockGrid {
            mi_row: 0,
            mi_col: 0,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: BLOCK_64X64,
        };
        let geom = ScaleGeom {
            ref_frame_width: 64,
            ref_frame_height: 64,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: false,
            subsampling_y: false,
        };
        // MV = [4, 4] eighth-pel -> 1/16 th = 8 = half-pel phase.
        let block_mvs = [[[4, 4]; 4]; 2];
        let args = InterPredArgs {
            plane: 0,
            x: 16,
            y: 16,
            w: 16,
            h: 16,
            block_idx: 0,
            interp_filter: 2,
            bit_depth: 8,
            is_compound: false,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp,
                    stride: rw,
                    ref_frame_width: 64,
                    ref_frame_height: 64,
                }),
                None,
            ],
        };
        predict_inter(
            &mut dst, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );

        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(dst.get(16 + j, 16 + i), 137, "i{i} j{j}");
            }
        }
    }

    /// A chroma plane (`plane = 1`) with 4:2:0 sub-sampling: the
    /// §8.5.2.4 `subX`/`subY` derive `lastX`/`lastY` from the half-size
    /// chroma extent, and the §8.5.2.3 fractional part is luma-based.
    /// On a flat reference the result is the flat value regardless.
    #[test]
    fn chroma_subsampled_flat_reference_passes_through() {
        // Chroma plane is 32×32 for a 64×64 luma frame.
        let (rw, rh) = (32usize, 32usize);
        let refp = vec![88i32; rw * rh];
        let mut dst = Plane::new(32, 32);

        let grid = BlockGrid {
            mi_row: 0,
            mi_col: 0,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: BLOCK_64X64,
        };
        let geom = ScaleGeom {
            ref_frame_width: 64,
            ref_frame_height: 64,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: true,
            subsampling_y: true,
        };
        let block_mvs = [[[6, -10]; 4]; 2];
        let args = InterPredArgs {
            plane: 1,
            x: 4,
            y: 4,
            w: 8,
            h: 8,
            block_idx: 0,
            interp_filter: 1,
            bit_depth: 8,
            is_compound: false,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp,
                    stride: rw,
                    ref_frame_width: 64,
                    ref_frame_height: 64,
                }),
                None,
            ],
        };
        predict_inter(&mut dst, &args, &grid, &geom, &block_mvs, &refs, true, true);

        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(dst.get(4 + j, 4 + i), 88, "i{i} j{j}");
            }
        }
    }

    // ---- §8.5.2.3 scaled-reference path ---------------------------------

    /// Independent §8.5.2.4 two-pass 8-tap convolution, re-derived from the
    /// spec formulas (`vp9-spec.txt` lines 4709-4738) rather than calling
    /// the crate's `block_inter_predict`. Used to cross-check the
    /// scaled-reference `predict_inter` output: the test computes the
    /// `startX` / `startY` / `stepX` / `stepY` via the spec §8.5.2.3 math,
    /// runs this reference convolution, and asserts the driver produced the
    /// same samples.
    #[allow(clippy::too_many_arguments)]
    fn spec_block_predict(
        refp: &[i32],
        stride: usize,
        start_x: i32,
        start_y: i32,
        step_x: i32,
        step_y: i32,
        w: usize,
        h: usize,
        last_x: i32,
        last_y: i32,
    ) -> Vec<i32> {
        // EIGHTTAP (interp_filter = 0) kernels, summing to 128.
        const TAPS: [[i32; 8]; 16] = [
            [0, 0, 0, 128, 0, 0, 0, 0],
            [0, 1, -5, 126, 8, -3, 1, 0],
            [-1, 3, -10, 122, 18, -6, 2, 0],
            [-1, 4, -13, 118, 27, -9, 3, -1],
            [-1, 4, -16, 112, 37, -11, 4, -1],
            [-1, 5, -18, 105, 48, -14, 4, -1],
            [-1, 5, -19, 97, 58, -16, 5, -1],
            [-1, 6, -19, 88, 68, -18, 5, -1],
            [-1, 6, -19, 78, 78, -19, 6, -1],
            [-1, 5, -18, 68, 88, -19, 6, -1],
            [-1, 5, -16, 58, 97, -19, 5, -1],
            [-1, 4, -14, 48, 105, -18, 5, -1],
            [-1, 4, -11, 37, 112, -16, 4, -1],
            [-1, 3, -9, 27, 118, -13, 4, -1],
            [0, 2, -6, 18, 122, -10, 3, -1],
            [0, 1, -3, 8, 126, -5, 1, 0],
        ];
        let clip3 = |lo: i32, hi: i32, v: i32| v.clamp(lo, hi);
        let sample = |row: i32, col: i32| -> i32 {
            refp[(row.max(0) as usize) * stride + col.max(0) as usize]
        };
        let inter_h = ((((h as i32 - 1) * step_y + 15) >> 4) + 8) as usize;
        let mut intermediate = vec![0i32; inter_h * w];
        for (r, row) in intermediate.chunks_mut(w).enumerate().take(inter_h) {
            for (c, slot) in row.iter_mut().enumerate() {
                let p = start_x + step_x * c as i32;
                let taps = &TAPS[(p & 15) as usize];
                let ref_row = clip3(0, last_y, (start_y >> 4) + r as i32 - 3);
                let mut s = 0i32;
                for (t, &tap) in taps.iter().enumerate() {
                    let ref_col = clip3(0, last_x, (p >> 4) + t as i32 - 3);
                    s += tap * sample(ref_row, ref_col);
                }
                *slot = round2(s, 7).clamp(0, 255);
            }
        }
        let mut pred = vec![0i32; h * w];
        for r in 0..h {
            for c in 0..w {
                let p = (start_y & 15) + step_y * r as i32;
                let taps = &TAPS[(p & 15) as usize];
                let base_row = (p >> 4) as usize;
                let mut s = 0i32;
                for (t, &tap) in taps.iter().enumerate() {
                    s += tap * intermediate[(base_row + t) * w + c];
                }
                pred[r * w + c] = round2(s, 7).clamp(0, 255);
            }
        }
        pred
    }

    /// A half-size reference frame (the §8.5.2.3 `xScale` / `yScale` ratio
    /// is `1 << (REF_SCALE_SHIFT - 1)`, so `stepX = stepY = 8`) over a
    /// non-flat ramp: the driver's scaled output must match an independent
    /// spec-formula re-derivation of the §8.5.2.3 start/step plus the
    /// §8.5.2.4 convolution. This validates the scaled-reference path on
    /// content that actually exercises the filter taps (a flat reference
    /// would pass through unchanged regardless of scaling).
    #[test]
    fn half_size_reference_scaled_path_matches_spec_rederivation() {
        // Reference is 32×32 for a 64×64 current frame: half size.
        let (rw, rh) = (32usize, 32usize);
        let mut refp = vec![0i32; rw * rh];
        for (r, row) in refp.chunks_mut(rw).enumerate() {
            for (c, slot) in row.iter_mut().enumerate() {
                // A smooth ramp inside the 8-bit range that varies on both
                // axes, so the 8-tap filter produces non-trivial output.
                *slot = ((3 * r + 5 * c) % 200 + 20) as i32;
            }
        }

        let grid = BlockGrid {
            mi_row: 1,
            mi_col: 1,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: BLOCK_8X8,
        };
        let geom = ScaleGeom {
            ref_frame_width: 32,
            ref_frame_height: 32,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: false,
            subsampling_y: false,
        };
        let (px, py, w, h) = (16i32, 8i32, 8usize, 8usize);
        // A small high-precision MV (eighth-pel) so the §8.5.2.2 clamp
        // doubles it and the §8.5.2.3 scaling lands on a non-zero phase.
        let mv = [5i32, -7i32];
        let block_mvs = [[mv; 4]; 2];
        let args = InterPredArgs {
            plane: 0,
            x: px,
            y: py,
            w,
            h,
            block_idx: 0,
            interp_filter: 0,
            bit_depth: 8,
            is_compound: false,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp,
                    stride: rw,
                    ref_frame_width: 32,
                    ref_frame_height: 32,
                }),
                None,
            ],
        };
        let mut dst = Plane::new(64, 64);
        predict_inter(
            &mut dst, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );

        // Independently re-derive the expected samples via the spec chain.
        let clamped = clamp_mv(&grid, mv, 0, false, false);
        let scaled = scale_mv(&geom, 0, px, py, clamped);
        let last_x = 32 - 1;
        let last_y = 32 - 1;
        let expected = spec_block_predict(
            &refp,
            rw,
            scaled.start_x,
            scaled.start_y,
            scaled.step_x,
            scaled.step_y,
            w,
            h,
            last_x,
            last_y,
        );
        // Sanity: a half-size reference halves the step.
        assert_eq!(scaled.step_x, 8);
        assert_eq!(scaled.step_y, 8);
        for i in 0..h {
            for j in 0..w {
                assert_eq!(
                    dst.get(px as usize + j, py as usize + i),
                    expected[i * w + j],
                    "scaled sample mismatch at i{i} j{j}"
                );
            }
        }
    }

    /// Compound prediction over two distinct *scaled* references: the
    /// driver must scale each reference list independently (each carries
    /// its own `ref_frame_width` / `ref_frame_height`) and then average
    /// with `Round2( p0 + p1, 1 )`. Cross-checked against two independent
    /// spec re-derivations.
    #[test]
    fn compound_scaled_references_average_per_list() {
        // List 0: half-size 32×32 ramp. List 1: full-size 64×64 ramp.
        let (rw0, rh0) = (32usize, 32usize);
        let mut refp0 = vec![0i32; rw0 * rh0];
        for (r, row) in refp0.chunks_mut(rw0).enumerate() {
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = ((2 * r + 7 * c) % 180 + 30) as i32;
            }
        }
        let (rw1, rh1) = (64usize, 64usize);
        let mut refp1 = vec![0i32; rw1 * rh1];
        for (r, row) in refp1.chunks_mut(rw1).enumerate() {
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = ((5 * r + 2 * c) % 160 + 40) as i32;
            }
        }

        let grid = BlockGrid {
            mi_row: 1,
            mi_col: 2,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: BLOCK_8X8,
        };
        let geom = ScaleGeom {
            ref_frame_width: 64,
            ref_frame_height: 64,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: false,
            subsampling_y: false,
        };
        let (px, py, w, h) = (16i32, 8i32, 8usize, 8usize);
        let mv0 = [9i32, 4i32];
        let mv1 = [-6i32, 11i32];
        let block_mvs = [[mv0; 4], [mv1; 4]];
        let args = InterPredArgs {
            plane: 0,
            x: px,
            y: py,
            w,
            h,
            block_idx: 0,
            interp_filter: 0,
            bit_depth: 8,
            is_compound: true,
        };
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples: &refp0,
                    stride: rw0,
                    ref_frame_width: 32,
                    ref_frame_height: 32,
                }),
                Some(RefPlane {
                    samples: &refp1,
                    stride: rw1,
                    ref_frame_width: 64,
                    ref_frame_height: 64,
                }),
            ],
        };
        let mut dst = Plane::new(64, 64);
        predict_inter(
            &mut dst, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );

        // Re-derive each list's prediction, then average per §8.5.2.
        let geom0 = ScaleGeom {
            ref_frame_width: 32,
            ref_frame_height: 32,
            ..geom
        };
        let clamped0 = clamp_mv(&grid, mv0, 0, false, false);
        let s0 = scale_mv(&geom0, 0, px, py, clamped0);
        let pred0 = spec_block_predict(
            &refp0, rw0, s0.start_x, s0.start_y, s0.step_x, s0.step_y, w, h, 31, 31,
        );
        let clamped1 = clamp_mv(&grid, mv1, 0, false, false);
        let s1 = scale_mv(&geom, 0, px, py, clamped1);
        let pred1 = spec_block_predict(
            &refp1, rw1, s1.start_x, s1.start_y, s1.step_x, s1.step_y, w, h, 63, 63,
        );
        for i in 0..h {
            for j in 0..w {
                let expected = round2(pred0[i * w + j] + pred1[i * w + j], 1);
                assert_eq!(
                    dst.get(px as usize + j, py as usize + i),
                    expected,
                    "compound scaled mismatch at i{i} j{j}"
                );
            }
        }
    }
}
