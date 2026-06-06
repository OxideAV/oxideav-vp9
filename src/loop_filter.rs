//! VP9 §8.8.1 `loop_filter_frame_init( )` — per spec v0.7.
//!
//! This module lands the per-frame [`loop_filter_frame_init`] book-keeping
//! primitive that builds the `LvlLookup[ MAX_SEGMENTS ][ MAX_REF_FRAMES ][
//! MAX_MODE_LF_DELTAS ]` filter-strength lookup table from the §6.2.8
//! `loop_filter_params( )` and §6.2.11 `segmentation_params( )` walker
//! outputs. The §8.8 outer driver invokes this once per frame, ahead of
//! the §8.8.2 superblock raster walk.
//!
//! The §8.8.1 listing (`vp9-spec.txt` lines 5465-5488) is reproduced
//! verbatim by this implementation:
//!
//! ```text
//! nShift = loop_filter_level >> 5
//! for ( segment_id = 0..MAX_SEGMENTS - 1 ) {
//!     1. lvlSeg = loop_filter_level
//!     2. if ( seg_feature_active( SEG_LVL_ALT_L ) ) {
//!         a. if ( segmentation_abs_or_delta_update == 1 )
//!                lvlSeg = FeatureData[ segment_id ][ SEG_LVL_ALT_L ]
//!         b. else
//!                lvlSeg = FeatureData[ segment_id ][ SEG_LVL_ALT_L ] + loop_filter_level
//!         c. lvlSeg = Clip3( 0, MAX_LOOP_FILTER, lvlSeg )
//!     }
//!     3. if ( loop_filter_delta_update == 0 )
//!            for ( ref = INTRA_FRAME..MAX_REF_FRAMES - 1 )
//!                for ( mode = 0..MAX_MODE_LF_DELTAS - 1 )
//!                    LvlLookup[ segment_id ][ ref ][ mode ] = lvlSeg
//!     4. if ( loop_filter_delta_enabled == 1 ) {
//!            intraLvl = lvlSeg + ( loop_filter_ref_deltas[ INTRA_FRAME ] << nShift )
//!            LvlLookup[ segment_id ][ INTRA_FRAME ][ 0 ] = Clip3( 0, MAX_LOOP_FILTER, intraLvl )
//!            for ( ref = LAST_FRAME; ref < MAX_REF_FRAMES; ref++ )
//!                for ( mode = 0; mode < MAX_MODE_LF_DELTAS; mode++ ) {
//!                    interLvl = lvlSeg
//!                             + ( loop_filter_ref_deltas[ ref ] << nShift )
//!                             + ( loop_filter_mode_deltas[ mode ] << nShift )
//!                    LvlLookup[ segment_id ][ ref ][ mode ] = Clip3( 0, MAX_LOOP_FILTER, interLvl )
//!                }
//!        }
//! }
//! ```
//!
//! ## Scope of this round
//!
//! Round 37 lands the §8.8.1 frame-init primitive — pure book-keeping
//! on top of the §6.2.8 [`LoopFilterParams`] and §6.2.11
//! [`SegmentationParams`] walker outputs that earlier rounds already
//! produce. The caller supplies resolved `loop_filter_ref_deltas[ 4 ]`
//! and `loop_filter_mode_deltas[ 2 ]` arrays (i.e. with the §6.2.8
//! `Option`-wrapped fields collapsed to the running decoder state per
//! the §7.2 "previous value" rule and the §7.2 `setup_past_independence`
//! initialisation).
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
//!   driving §8.8.3 / §8.8.4 / §8.8.5, which needs `MiSizes` / `TxSizes`
//!   / `Skips` / `RefFrames` arrays the §6.4.4 [`crate::decode_block`]
//!   fan-out produces.
//! * §8.8.3 `filter_size` — the `txSz` / `is32Edge` derivation.
//! * §8.8.4 `adaptive_filter_strength` — reads `LvlLookup` produced by
//!   this round; the §8.8 driver wires the two together.
//! * §8.8.5 `sample_filtering` — the actual MB-edge deblocking filter
//!   primitives (`filter4` / `filter6` / `filter8` / `filter16`).
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.1 lines 5465-5488; §3
//! `MAX_SEGMENTS = 8` line 474, `MAX_REF_FRAMES = 4` line 470,
//! `MAX_MODE_LF_DELTAS = 2` line 513, `MAX_LOOP_FILTER = 63` line 515,
//! `SEG_LVL_ALT_L = 1` line 476; §6.4.9 `seg_feature_active( )`).
//! `Clip3` is the §5.1 primitive.

use crate::dequant::seg_feature_active;
use crate::header::{LoopFilterParams, SegmentationParams, MAX_SEGMENTS};
use crate::mode_info::{ALTREF_FRAME, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME};

/// `SEG_LVL_ALT_L = 1` per §3 (`vp9-spec.txt` line 476) — the
/// segmentation-feature index that carries a per-segment loop-filter
/// override (either an absolute replacement of `loop_filter_level` or
/// a delta added to it, governed by §6.2.11
/// `segmentation_abs_or_delta_update`). §8.8.1 step 2 gates on
/// `seg_feature_active( SEG_LVL_ALT_L )` per §6.4.9 to pick up the
/// per-segment level override before broadcasting / applying deltas.
pub(crate) const SEG_LVL_ALT_L: usize = 1;

/// `MAX_MODE_LF_DELTAS = 2` per §3 (`vp9-spec.txt` line 513) — the two
/// per-mode loop-filter delta slots (`mode = 0` for `ZEROMV`-like
/// reference modes and `mode = 1` for new-mv modes per §7.2.8). Sizes
/// the inner dimension of the §8.8.1 `LvlLookup` table.
pub const MAX_MODE_LF_DELTAS: usize = 2;

/// `MAX_LOOP_FILTER = 63` per §3 (`vp9-spec.txt` line 515) — upper
/// clip of the §8.8.1 `Clip3( 0, MAX_LOOP_FILTER, … )` saturations.
pub const MAX_LOOP_FILTER: i32 = 63;

/// §8.8.1 frame-init result — the `LvlLookup[ MAX_SEGMENTS ][
/// MAX_REF_FRAMES ][ MAX_MODE_LF_DELTAS ]` table indexed by
/// `(segment_id, ref_frame, mode)` per the §8.8.1 listing line 5478 /
/// 5481 / 5486.
///
/// Cells are `u8` because the §8.8.1 `Clip3( 0, MAX_LOOP_FILTER, … )`
/// saturation bounds every output into `0..=63`. The §8.8.4
/// `adaptive_filter_strength` consumer reads this with
/// `LvlLookup[ SegmentIds[ loopRow ][ loopCol ] ][ RefFrames[ loopRow
/// ][ loopCol ][ 0 ] ][ mode_lf_lookup[ YModes[ loopRow ][ loopCol ]
/// ] ]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LvlLookup {
    /// Storage: `[segment_id][ref][mode]`, row-major per the §8.8.1
    /// listing's nested `for` iteration order.
    pub levels: [[[u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES_USIZE]; MAX_SEGMENTS],
}

/// `MAX_REF_FRAMES` (= 4) re-exposed as a `usize` to size the §8.8.1
/// `LvlLookup` middle dimension. The crate-local
/// [`crate::mode_info::MAX_REF_FRAMES`] is an `i32` to match the §3
/// constant table's contextual use as a loop bound.
const MAX_REF_FRAMES_USIZE: usize = 4;

impl LvlLookup {
    /// All-zero table.
    pub fn zeros() -> Self {
        Self {
            levels: [[[0u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES_USIZE]; MAX_SEGMENTS],
        }
    }

    /// Read `LvlLookup[ segment_id ][ ref_frame ][ mode ]`. Returns
    /// `None` for out-of-range indices.
    ///
    /// `ref_frame` accepts the §3 `INTRA_FRAME = 0` / `LAST_FRAME = 1`
    /// / `GOLDEN_FRAME = 2` / `ALTREF_FRAME = 3` integers exposed by
    /// the [`crate::mode_info`] module.
    pub fn get(&self, segment_id: usize, ref_frame: i32, mode: usize) -> Option<u8> {
        if segment_id >= MAX_SEGMENTS || mode >= MAX_MODE_LF_DELTAS {
            return None;
        }
        if ref_frame < 0 || (ref_frame as usize) >= MAX_REF_FRAMES_USIZE {
            return None;
        }
        Some(self.levels[segment_id][ref_frame as usize][mode])
    }
}

/// `Clip3( x, y, z )` from §5.1 — clamp `z` into `[x, y]`. The §8.8.1
/// listing applies it twice (line 5476 on `lvlSeg`, and lines 5481 /
/// 5486 on the per-(ref, mode) computed level).
#[inline]
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

/// Run §8.8.1 `loop_filter_frame_init( )` on the resolved
/// [`LoopFilterParams`] and [`SegmentationParams`] of one VP9 frame
/// per `vp9-spec.txt` lines 5465-5488.
///
/// Produces the §8.8.1 `LvlLookup[ MAX_SEGMENTS ][ MAX_REF_FRAMES ][
/// MAX_MODE_LF_DELTAS ]` table the §8.8 raster walk reads at every
/// `(plane, pass, row, col)` step via §8.8.4.
///
/// # Inputs
///
/// * `lf` — the [`LoopFilterParams`] produced by §6.2.8
///   `loop_filter_params( )`. The `level` / `delta_enabled` /
///   `delta_update` flags drive the §8.8.1 step gates; the `ref_deltas`
///   / `mode_deltas` `Option<i8>` slots are absent values that the
///   caller must already have resolved into the `ref_deltas` /
///   `mode_deltas` parameters per the §7.2 "use prior value" rule and
///   the §7.2 `setup_past_independence` defaults
///   (`{INTRA: 1, LAST: 0, GOLDEN: -1, ALTREF: -1}` for refs, all-zero
///   for modes).
/// * `seg` — the [`SegmentationParams`] produced by §6.2.11. The
///   `enabled` flag + `feature_enabled[ ][ SEG_LVL_ALT_L ]` cell drives
///   the §6.4.9 `seg_feature_active( SEG_LVL_ALT_L )` gate; the
///   `abs_or_delta_update` flag and `feature_data[ ][ SEG_LVL_ALT_L ]`
///   cell drive the §8.8.1 step 2.a / 2.b override.
/// * `ref_deltas` — resolved `loop_filter_ref_deltas[ MAX_REF_FRAMES ]`
///   indexed by §3 `INTRA_FRAME` / `LAST_FRAME` / `GOLDEN_FRAME` /
///   `ALTREF_FRAME`. The §7.2 `setup_past_independence` defaults are
///   `[1, 0, -1, -1]` for `[INTRA, LAST, GOLDEN, ALTREF]`.
/// * `mode_deltas` — resolved `loop_filter_mode_deltas[
///   MAX_MODE_LF_DELTAS ]`. The §7.2 `setup_past_independence` default
///   is `[0, 0]`.
///
/// The fully-resolved `ref_deltas` / `mode_deltas` inputs let this
/// primitive stand alone without a `Vp9DecoderState` carrying the
/// running deltas across frames — that state is the §7.2 inter-frame
/// orchestrator's responsibility, not §8.8.1's.
///
/// # Output
///
/// A [`LvlLookup`] table indexed by `(segment_id, ref_frame, mode)`
/// with every cell saturated into `0..=MAX_LOOP_FILTER`.
pub fn loop_filter_frame_init(
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
    ref_deltas: [i8; MAX_REF_FRAMES_USIZE],
    mode_deltas: [i8; MAX_MODE_LF_DELTAS],
) -> LvlLookup {
    let mut out = LvlLookup::zeros();

    // §8.8.1 line 5468: nShift = loop_filter_level >> 5.
    let n_shift = (lf.level as i32) >> 5;
    let loop_filter_level = lf.level as i32;

    for segment_id in 0..MAX_SEGMENTS {
        // §8.8.1 step 1: lvlSeg = loop_filter_level.
        let mut lvl_seg: i32 = loop_filter_level;

        // §8.8.1 step 2: §6.4.9 seg_feature_active( SEG_LVL_ALT_L ) gate.
        if seg_feature_active(seg, segment_id, SEG_LVL_ALT_L) {
            // §8.8.1 steps 2.a / 2.b: absolute vs. delta update.
            let feature_data = seg.feature_data[segment_id][SEG_LVL_ALT_L] as i32;
            lvl_seg = if seg.abs_or_delta_update {
                feature_data
            } else {
                feature_data + loop_filter_level
            };
            // §8.8.1 step 2.c: Clip3( 0, MAX_LOOP_FILTER, lvlSeg ).
            lvl_seg = clip3(0, MAX_LOOP_FILTER, lvl_seg);
        }

        // §8.8.1 step 3: if ( loop_filter_delta_update == 0 ) broadcast.
        if !lf.delta_update {
            for r in 0..MAX_REF_FRAMES_USIZE {
                for m in 0..MAX_MODE_LF_DELTAS {
                    out.levels[segment_id][r][m] = lvl_seg as u8;
                }
            }
        }

        // §8.8.1 step 4: if ( loop_filter_delta_enabled == 1 ) apply deltas.
        if lf.delta_enabled {
            // §8.8.1 line 5480: intraLvl = lvlSeg + ( ref_deltas[ INTRA ] << nShift ).
            let intra_lvl = lvl_seg + ((ref_deltas[INTRA_FRAME as usize] as i32) << n_shift);
            // §8.8.1 line 5481: LvlLookup[ s ][ INTRA ][ 0 ] = Clip3(...).
            out.levels[segment_id][INTRA_FRAME as usize][0] =
                clip3(0, MAX_LOOP_FILTER, intra_lvl) as u8;

            // §8.8.1 lines 5482-5487: per-(ref, mode) inter loop.
            for r in [LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME] {
                for (m, &mode_delta) in mode_deltas.iter().enumerate() {
                    // §8.8.1 line 5484-5485: interLvl = lvlSeg
                    //                                 + ( ref_deltas[ r ] << nShift )
                    //                                 + ( mode_deltas[ m ] << nShift ).
                    let inter_lvl = lvl_seg
                        + ((ref_deltas[r as usize] as i32) << n_shift)
                        + ((mode_delta as i32) << n_shift);
                    // §8.8.1 line 5486: LvlLookup[ s ][ r ][ m ] = Clip3(...).
                    out.levels[segment_id][r as usize][m] =
                        clip3(0, MAX_LOOP_FILTER, inter_lvl) as u8;
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::SEG_LVL_MAX;

    /// `setup_past_independence` defaults from §7.2 (`vp9-spec.txt`
    /// lines 3494-3499): `loop_filter_delta_enabled = 1`,
    /// `loop_filter_ref_deltas = [1, 0, -1, -1]`,
    /// `loop_filter_mode_deltas = [0, 0]`.
    const SETUP_PAST_INDEPENDENCE_REF_DELTAS: [i8; 4] = [1, 0, -1, -1];
    const SETUP_PAST_INDEPENDENCE_MODE_DELTAS: [i8; 2] = [0, 0];

    fn lf_disabled() -> LoopFilterParams {
        LoopFilterParams {
            level: 0,
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            ref_deltas: [None; 4],
            mode_deltas: [None; 2],
        }
    }

    fn seg_disabled() -> SegmentationParams {
        SegmentationParams {
            enabled: false,
            update_map: false,
            tree_probs: None,
            temporal_update: false,
            pred_prob: None,
            update_data: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }

    /// §8.8.1 base case: `loop_filter_level = 0` with everything
    /// disabled yields an all-zero `LvlLookup` (line 5468 makes
    /// `nShift = 0`; step 3 broadcasts `lvlSeg = 0` across every cell;
    /// step 4 is gated off).
    #[test]
    fn zero_level_all_disabled_yields_zero_table() {
        let lf = lf_disabled();
        let seg = seg_disabled();
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        for s in 0..MAX_SEGMENTS {
            for r in 0..MAX_REF_FRAMES_USIZE {
                for m in 0..MAX_MODE_LF_DELTAS {
                    assert_eq!(lookup.levels[s][r][m], 0, "s={s} r={r} m={m}");
                }
            }
        }
    }

    /// §8.8.1 step 3: `delta_update == 0` AND `delta_enabled == 0`
    /// broadcasts `lvlSeg = loop_filter_level` into every cell.
    #[test]
    fn delta_update_zero_broadcasts_level_into_every_cell() {
        let mut lf = lf_disabled();
        lf.level = 42;
        // delta_enabled = false, delta_update = false: only step 3 fires.
        let seg = seg_disabled();
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        for s in 0..MAX_SEGMENTS {
            for r in 0..MAX_REF_FRAMES_USIZE {
                for m in 0..MAX_MODE_LF_DELTAS {
                    assert_eq!(lookup.levels[s][r][m], 42);
                }
            }
        }
    }

    /// §8.8.1 step 4 alone: `delta_update == 1` AND `delta_enabled ==
    /// 1` skips the step-3 broadcast but step 4 still writes every
    /// `(ref, mode)` cell (line 5481 covers `INTRA / 0`; lines
    /// 5482-5487 cover `(LAST..ALTREF, 0..2)`; the `INTRA / 1` cell is
    /// NEVER written by §8.8.1 so it stays at the zero default).
    #[test]
    fn delta_update_set_with_delta_enabled_writes_all_cells_except_intra_mode_1() {
        let mut lf = lf_disabled();
        lf.level = 10;
        lf.delta_update = true;
        lf.delta_enabled = true;
        let seg = seg_disabled();
        let lookup = loop_filter_frame_init(
            &lf,
            &seg,
            SETUP_PAST_INDEPENDENCE_REF_DELTAS,
            SETUP_PAST_INDEPENDENCE_MODE_DELTAS,
        );
        // nShift = 10 >> 5 = 0; the deltas are added at scale 1.
        // INTRA / 0: lvlSeg + ref_delta[INTRA] = 10 + 1 = 11.
        assert_eq!(lookup.levels[0][INTRA_FRAME as usize][0], 11);
        // INTRA / 1: never written by §8.8.1, stays at zero default.
        assert_eq!(lookup.levels[0][INTRA_FRAME as usize][1], 0);
        // LAST / 0: lvlSeg + ref_delta[LAST] + mode_delta[0] = 10 + 0 + 0 = 10.
        assert_eq!(lookup.levels[0][LAST_FRAME as usize][0], 10);
        // GOLDEN / 1: lvlSeg + ref_delta[GOLDEN] + mode_delta[1] = 10 + -1 + 0 = 9.
        assert_eq!(lookup.levels[0][GOLDEN_FRAME as usize][1], 9);
        // ALTREF / 0: lvlSeg + ref_delta[ALTREF] + mode_delta[0] = 10 + -1 + 0 = 9.
        assert_eq!(lookup.levels[0][ALTREF_FRAME as usize][0], 9);
    }

    /// §8.8.1 line 5468 `nShift = loop_filter_level >> 5`: when
    /// `loop_filter_level >= 32` the deltas are shifted left by 1 so
    /// a `±1` delta moves the per-(ref, mode) cell by ±2 from
    /// `lvlSeg`.
    #[test]
    fn n_shift_scales_deltas_at_level_threshold_32() {
        let mut lf = lf_disabled();
        lf.level = 32;
        lf.delta_update = true;
        lf.delta_enabled = true;
        let seg = seg_disabled();
        let lookup = loop_filter_frame_init(
            &lf,
            &seg,
            SETUP_PAST_INDEPENDENCE_REF_DELTAS,
            SETUP_PAST_INDEPENDENCE_MODE_DELTAS,
        );
        // nShift = 32 >> 5 = 1; deltas scale by 2.
        // INTRA / 0: 32 + (1 << 1) = 34.
        assert_eq!(lookup.levels[0][INTRA_FRAME as usize][0], 34);
        // LAST / 0: 32 + (0 << 1) + (0 << 1) = 32.
        assert_eq!(lookup.levels[0][LAST_FRAME as usize][0], 32);
        // GOLDEN / 0: 32 + (-1 << 1) + (0 << 1) = 30.
        assert_eq!(lookup.levels[0][GOLDEN_FRAME as usize][0], 30);
        // ALTREF / 0: 32 + (-1 << 1) + (0 << 1) = 30.
        assert_eq!(lookup.levels[0][ALTREF_FRAME as usize][0], 30);
    }

    /// §8.8.1 step 4 `Clip3( 0, MAX_LOOP_FILTER, … )`: a `lvlSeg` near
    /// `MAX_LOOP_FILTER` plus a positive ref-delta saturates at 63.
    #[test]
    fn step_4_clip3_saturates_at_max_loop_filter() {
        let mut lf = lf_disabled();
        lf.level = 63;
        lf.delta_update = true;
        lf.delta_enabled = true;
        let seg = seg_disabled();
        // ref_deltas[INTRA] = 1, nShift = 63 >> 5 = 1; intraLvl = 63 + 2 = 65 → 63.
        let lookup = loop_filter_frame_init(
            &lf,
            &seg,
            SETUP_PAST_INDEPENDENCE_REF_DELTAS,
            SETUP_PAST_INDEPENDENCE_MODE_DELTAS,
        );
        assert_eq!(
            lookup.levels[0][INTRA_FRAME as usize][0],
            MAX_LOOP_FILTER as u8
        );
    }

    /// §8.8.1 step 4 `Clip3( 0, MAX_LOOP_FILTER, … )`: a `lvlSeg = 0`
    /// plus a negative ref-delta saturates at 0 (no underflow into
    /// negative `u8`).
    #[test]
    fn step_4_clip3_saturates_at_zero() {
        let mut lf = lf_disabled();
        lf.level = 0;
        lf.delta_update = true;
        lf.delta_enabled = true;
        let seg = seg_disabled();
        // ref_deltas[GOLDEN] = -1, nShift = 0; interLvl = 0 + -1 + 0 = -1 → 0.
        let lookup = loop_filter_frame_init(
            &lf,
            &seg,
            SETUP_PAST_INDEPENDENCE_REF_DELTAS,
            SETUP_PAST_INDEPENDENCE_MODE_DELTAS,
        );
        assert_eq!(lookup.levels[0][GOLDEN_FRAME as usize][0], 0);
        assert_eq!(lookup.levels[0][ALTREF_FRAME as usize][0], 0);
    }

    /// §8.8.1 step 2.a `segmentation_abs_or_delta_update == 1`: a
    /// segment with `SEG_LVL_ALT_L` enabled REPLACES `lvlSeg` with the
    /// segment's `FeatureData[ s ][ SEG_LVL_ALT_L ]` value.
    #[test]
    fn seg_alt_l_abs_replaces_lvl_seg() {
        let mut lf = lf_disabled();
        lf.level = 10;
        let mut seg = seg_disabled();
        seg.enabled = true;
        seg.abs_or_delta_update = true;
        seg.feature_enabled[2][SEG_LVL_ALT_L] = true;
        seg.feature_data[2][SEG_LVL_ALT_L] = 25;
        // Other segments: lvlSeg = 10. Segment 2: lvlSeg replaced with 25.
        // delta_update = false, delta_enabled = false → step 3 broadcast only.
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        // Segment 0 (no SEG_LVL_ALT_L): lvlSeg = 10.
        assert_eq!(lookup.levels[0][LAST_FRAME as usize][0], 10);
        // Segment 2 (SEG_LVL_ALT_L enabled, abs mode): lvlSeg = 25.
        assert_eq!(lookup.levels[2][LAST_FRAME as usize][0], 25);
        // Segment 5 (no SEG_LVL_ALT_L): lvlSeg = 10.
        assert_eq!(lookup.levels[5][LAST_FRAME as usize][0], 10);
    }

    /// §8.8.1 step 2.b `segmentation_abs_or_delta_update == 0`: a
    /// segment with `SEG_LVL_ALT_L` enabled ADDS the segment's
    /// `FeatureData[ s ][ SEG_LVL_ALT_L ]` to `loop_filter_level`.
    #[test]
    fn seg_alt_l_delta_adds_to_lvl_seg() {
        let mut lf = lf_disabled();
        lf.level = 30;
        let mut seg = seg_disabled();
        seg.enabled = true;
        seg.abs_or_delta_update = false;
        seg.feature_enabled[3][SEG_LVL_ALT_L] = true;
        seg.feature_data[3][SEG_LVL_ALT_L] = 10;
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        // Segment 3 (SEG_LVL_ALT_L enabled, delta mode): lvlSeg = 30 + 10 = 40.
        assert_eq!(lookup.levels[3][LAST_FRAME as usize][0], 40);
        // Segment 0 (no SEG_LVL_ALT_L): lvlSeg = 30.
        assert_eq!(lookup.levels[0][LAST_FRAME as usize][0], 30);
    }

    /// §8.8.1 step 2.c: the segment override is clipped INTO
    /// `[0, MAX_LOOP_FILTER]` before step 3 / step 4 see it. A
    /// `feature_data = -100` in delta mode would otherwise underflow.
    #[test]
    fn seg_alt_l_step_2c_clips_underflow_to_zero() {
        let mut lf = lf_disabled();
        lf.level = 5;
        let mut seg = seg_disabled();
        seg.enabled = true;
        seg.abs_or_delta_update = false;
        seg.feature_enabled[1][SEG_LVL_ALT_L] = true;
        seg.feature_data[1][SEG_LVL_ALT_L] = -100;
        // delta_update = false → step 3 broadcast.
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        // Segment 1: lvlSeg = Clip3(0, 63, 5 + -100) = 0.
        assert_eq!(lookup.levels[1][LAST_FRAME as usize][0], 0);
    }

    /// §8.8.1 step 2.c: same clip on the high side — `feature_data =
    /// 200` in abs mode clamps to 63.
    #[test]
    fn seg_alt_l_step_2c_clips_overflow_to_max() {
        let mut lf = lf_disabled();
        lf.level = 5;
        let mut seg = seg_disabled();
        seg.enabled = true;
        seg.abs_or_delta_update = true;
        seg.feature_enabled[4][SEG_LVL_ALT_L] = true;
        seg.feature_data[4][SEG_LVL_ALT_L] = 200;
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        // Segment 4: lvlSeg = Clip3(0, 63, 200) = 63.
        assert_eq!(
            lookup.levels[4][LAST_FRAME as usize][0],
            MAX_LOOP_FILTER as u8
        );
    }

    /// §6.4.9 `seg_feature_active( )` rule: when `segmentation_enabled
    /// == 0` the §8.8.1 step 2 gate is OFF even if
    /// `feature_enabled[ ][ SEG_LVL_ALT_L ] == 1`. The §6.4.9 listing
    /// requires both flags.
    #[test]
    fn segmentation_disabled_skips_step_2_even_when_feature_enabled_set() {
        let mut lf = lf_disabled();
        lf.level = 20;
        let mut seg = seg_disabled();
        // enabled = false BUT feature_enabled[2][SEG_LVL_ALT_L] = true.
        seg.feature_enabled[2][SEG_LVL_ALT_L] = true;
        seg.feature_data[2][SEG_LVL_ALT_L] = 50;
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        // Segment 2 sees lvlSeg = 20, NOT 50 — segmentation_enabled gates §6.4.9.
        assert_eq!(lookup.levels[2][LAST_FRAME as usize][0], 20);
    }

    /// §8.8.1 line 5481 vs. lines 5482-5487 split: `INTRA_FRAME`'s
    /// `mode` index is fixed at 0 (line 5481); the inter loop walks
    /// `LAST..ALTREF`. The `INTRA_FRAME / mode = 1` cell is therefore
    /// never written by step 4. With `delta_update == 0` step 3
    /// broadcasts `lvlSeg` into every cell first, including the
    /// `INTRA / 1` slot — i.e. the `INTRA / 1` cell retains `lvlSeg`
    /// in the `(delta_update = 0, delta_enabled = 1)` configuration.
    #[test]
    fn intra_mode_1_cell_keeps_step_3_broadcast_when_step_4_runs() {
        let mut lf = lf_disabled();
        lf.level = 20;
        lf.delta_update = false; // step 3 runs
        lf.delta_enabled = true; // step 4 also runs
        let seg = seg_disabled();
        let lookup = loop_filter_frame_init(
            &lf,
            &seg,
            SETUP_PAST_INDEPENDENCE_REF_DELTAS,
            SETUP_PAST_INDEPENDENCE_MODE_DELTAS,
        );
        // Step 3 broadcasts INTRA / 1 = 20.
        // Step 4 line 5481 only overwrites INTRA / 0.
        assert_eq!(lookup.levels[0][INTRA_FRAME as usize][1], 20);
        // INTRA / 0 overwritten: 20 + (1 << 0) = 21.
        assert_eq!(lookup.levels[0][INTRA_FRAME as usize][0], 21);
    }

    /// Public read-back surface: [`LvlLookup::get`] returns `None` for
    /// every out-of-range axis and `Some` for the in-range ones.
    #[test]
    fn lvl_lookup_get_bounds_check() {
        let lookup = LvlLookup::zeros();
        assert_eq!(lookup.get(0, INTRA_FRAME, 0), Some(0));
        assert_eq!(lookup.get(MAX_SEGMENTS - 1, ALTREF_FRAME, 1), Some(0));
        // Out of range.
        assert_eq!(lookup.get(MAX_SEGMENTS, INTRA_FRAME, 0), None);
        assert_eq!(lookup.get(0, MAX_REF_FRAMES_USIZE as i32, 0), None);
        assert_eq!(lookup.get(0, -1, 0), None);
        assert_eq!(lookup.get(0, INTRA_FRAME, MAX_MODE_LF_DELTAS), None);
    }
}
