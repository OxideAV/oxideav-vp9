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
    let (px, py) = (args.x as usize, args.y as usize);
    for i in 0..args.h {
        for j in 0..args.w {
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
}
