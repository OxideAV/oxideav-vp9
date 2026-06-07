//! VP9 §8.8.4 `adaptive_filter_strength( )` — per spec v0.7.
//!
//! This module lands the per-edge [`adaptive_filter_strength`]
//! derivation as a pure leaf primitive. The §8.8.2 superblock raster
//! walker invokes it after the §8.8.3 [`crate::filter_size`] pick to
//! turn `(loopRow, loopCol)` into the four §8.8.5 strength outputs
//! `(lvl, limit, blimit, thresh)` the sample-filter pass reads.
//!
//! The §8.8.4 listing (`vp9-spec.txt` lines 5626-5661) describes:
//!
//! 1. `lvl` derivation:
//!    * `segment = SegmentIds[ loopRow ][ loopCol ]`
//!    * `ref = RefFrames[ loopRow ][ loopCol ][ 0 ]`
//!    * `mode = YModes[ loopRow ][ loopCol ]`
//!    * `modeType = (mode == NEARESTMV || mode == NEARMV || mode ==
//!      NEWMV) ? 1 : 0` (intra modes and `ZEROMV` map to 0).
//!    * `lvl = LvlLookup[ segment ][ ref ][ modeType ]`.
//! 2. `shift` derivation:
//!    * `loop_filter_sharpness > 4` → `shift = 2`.
//!    * `loop_filter_sharpness > 0` → `shift = 1`.
//!    * Otherwise → `shift = 0`.
//! 3. `limit` derivation:
//!    * `loop_filter_sharpness > 0` → `limit = Clip3( 1, 9 -
//!      loop_filter_sharpness, lvl >> shift )`.
//!    * Otherwise → `limit = Max( 1, lvl >> shift )`.
//! 4. `blimit = 2 * (lvl + 2) + limit`.
//! 5. `thresh = lvl >> 4`.
//!
//! ## Scope of this round
//!
//! Round 250 lands the §8.8.4 leaf only — pure-state function from the
//! [`crate::loop_filter::LvlLookup`] table the round-37 §8.8.1
//! [`crate::loop_filter_frame_init`] frame-init builds, plus the
//! per-MI `(segment_id, ref_frame, y_mode)` triple the §8.8.2 raster
//! walker reads from `SegmentIds[ ][ ]` / `RefFrames[ ][ ][ 0 ]` /
//! `YModes[ ][ ]`, plus the §6.2.8 `loop_filter_sharpness` header
//! field. The caller is responsible for fetching the three per-MI
//! values at `(loopRow, loopCol)` — this primitive does not walk the
//! `loopRow` / `loopCol` raster itself.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
//!   that calls this primitive at every `(loopRow, loopCol)` step.
//! * §8.8.5 `sample_filtering` — the actual edge-filter primitives
//!   that consume the `(lvl, limit, blimit, thresh)` tuple this
//!   primitive returns.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.4 lines 5626-5661; §7.4.11
//! `NEARESTMV = 10` / `NEARMV = 11` / `ZEROMV = 12` / `NEWMV = 13`
//! lines 3957-3961; §6.2.8 `loop_filter_sharpness` u3). `Clip3` and
//! `Max` are §5.1 primitives.

use crate::loop_filter::{LvlLookup, MAX_MODE_LF_DELTAS};

/// `NEARESTMV = 10` per §7.4.11 (`vp9-spec.txt` line 3958) — first of
/// the four inter `y_mode` values (`inter_mode = 0` offset by the
/// `+10` shift that places it above the ten §7.4.5 intra modes).
/// §8.8.4 step 1 maps it to `modeType = 1` (lines 5637-5638).
pub const NEARESTMV: u8 = 10;

/// `NEARMV = 11` per §7.4.11 (`vp9-spec.txt` line 3959) — second of
/// the four inter `y_mode` values. §8.8.4 step 1 maps it to
/// `modeType = 1`.
pub const NEARMV: u8 = 11;

/// `ZEROMV = 12` per §7.4.11 (`vp9-spec.txt` line 3960) — third of
/// the four inter `y_mode` values. §8.8.4 step 1 maps it to
/// `modeType = 0` (line 5638: "intra type or `ZEROMV`").
pub const ZEROMV: u8 = 12;

/// `NEWMV = 13` per §7.4.11 (`vp9-spec.txt` line 3961) — fourth of
/// the four inter `y_mode` values. §8.8.4 step 1 maps it to
/// `modeType = 1`.
pub const NEWMV: u8 = 13;

/// §8.8.4 output — the four per-edge filter-strength values the
/// §8.8.5 sample-filter pass reads at `(loopRow, loopCol)`. Each
/// field corresponds to a §8.8.4 listing line:
///
/// * `lvl` — line 5639, the [`crate::loop_filter::LvlLookup`] cell
///   indexed by `(segment, ref, modeType)`.
/// * `limit` — lines 5648-5651, the post-shift / post-sharpness
///   filter limit.
/// * `blimit` — line 5660, `2 * (lvl + 2) + limit`.
/// * `thresh` — line 5661, `lvl >> 4` — the §8.8.5.1 high-edge-
///   variance threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterStrength {
    /// `lvl` from §8.8.4 line 5639 — the
    /// [`crate::loop_filter::LvlLookup`] cell after `(segment, ref,
    /// modeType)` indexing. Bounded into `0..=63` by the §8.8.1
    /// `Clip3( 0, MAX_LOOP_FILTER, … )` saturation.
    pub lvl: u8,
    /// `limit` from §8.8.4 lines 5648-5651 — the sharpness-modulated
    /// filter limit.
    pub limit: u8,
    /// `blimit` from §8.8.4 line 5660 — `2 * (lvl + 2) + limit`.
    pub blimit: u8,
    /// `thresh` from §8.8.4 line 5661 — `lvl >> 4`, the §8.8.5.1
    /// high-edge-variance threshold.
    pub thresh: u8,
}

/// §5.1 `Clip3( x, y, z )` — clamp `z` into `[x, y]`. Used by §8.8.4
/// line 5649 to clip `lvl >> shift` into `[1, 9 -
/// loop_filter_sharpness]` when sharpness is non-zero.
#[inline]
fn clip3_u8(x: u8, y: u8, z: u8) -> u8 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

/// §8.8.4 step 1 `modeType` — return `1` if `mode` is one of
/// `NEARESTMV` / `NEARMV` / `NEWMV` (the three MV-predicting inter
/// modes), `0` otherwise (intra modes 0..9 or `ZEROMV = 12`). Lines
/// 5637-5638 of `vp9-spec.txt`.
///
/// Exposed at module scope (rather than inlined into
/// [`adaptive_filter_strength`]) because the §8.8.2 superblock raster
/// walker can use it directly to derive the second axis of the
/// §8.8.1 [`crate::loop_filter::LvlLookup`] read without re-reading
/// `YModes[ ][ ]`.
#[inline]
pub fn mode_to_mode_type(mode: u8) -> usize {
    match mode {
        NEARESTMV | NEARMV | NEWMV => 1,
        _ => 0,
    }
}

/// Run §8.8.4 `adaptive_filter_strength( )` for one `(loopRow,
/// loopCol)` luma 8x8 position per `vp9-spec.txt` lines 5626-5661.
///
/// Returns the §8.8.5 input tuple `(lvl, limit, blimit, thresh)`
/// the sample-filter pass reads at the matching edge.
///
/// # Inputs
///
/// * `lvl_lookup` — the §8.8.1 [`crate::loop_filter::LvlLookup`] the
///   round-37 [`crate::loop_filter_frame_init`] built once per frame.
/// * `segment_id` — `SegmentIds[ loopRow ][ loopCol ]` per §6.4.7 /
///   §6.4.12, in `0..MAX_SEGMENTS`.
/// * `ref_frame` — `RefFrames[ loopRow ][ loopCol ][ 0 ]` per
///   §6.4.17. The §3 reference enumeration is `INTRA_FRAME = 0`,
///   `LAST_FRAME = 1`, `GOLDEN_FRAME = 2`, `ALTREF_FRAME = 3` (so
///   `ref_frame` is always in `0..4`).
/// * `y_mode` — `YModes[ loopRow ][ loopCol ]` per §6.4.15 (intra:
///   one of the ten §7.4.5 [`crate::intra::PredMode`] discriminants
///   `0..=9`) or §7.4.11 (inter: one of the four MV modes
///   [`NEARESTMV`] / [`NEARMV`] / [`ZEROMV`] / [`NEWMV`] =
///   `10..=13`). §8.8.4 step 1 maps it to `modeType = 1` for
///   `NEARESTMV` / `NEARMV` / `NEWMV` and `modeType = 0` otherwise.
/// * `loop_filter_sharpness` — `loop_filter_sharpness` from §6.2.8
///   (a `u3`, i.e. `0..=7`). Drives the `shift` / `limit` steps.
///
/// Returns `None` for an out-of-range axis (`segment_id >=
/// MAX_SEGMENTS`, `ref_frame` outside `0..=3`).
///
/// # Listing
///
/// `vp9-spec.txt` §8.8.4 lines 5632-5661:
///
/// ```text
/// 1. segment   = SegmentIds[ loopRow ][ loopCol ]
///    ref       = RefFrames[ loopRow ][ loopCol ][ 0 ]
///    mode      = YModes[ loopRow ][ loopCol ]
///    modeType  = (mode in {NEARESTMV, NEARMV, NEWMV}) ? 1 : 0
///    lvl       = LvlLookup[ segment ][ ref ][ modeType ]
/// 2. shift:
///      if (loop_filter_sharpness > 4) shift = 2
///      else if (loop_filter_sharpness > 0) shift = 1
///      else shift = 0
/// 3. limit:
///      if (loop_filter_sharpness > 0)
///          limit = Clip3( 1, 9 - loop_filter_sharpness, lvl >> shift )
///      else
///          limit = Max( 1, lvl >> shift )
/// 4. blimit  = 2 * (lvl + 2) + limit
/// 5. thresh  = lvl >> 4
/// ```
pub fn adaptive_filter_strength(
    lvl_lookup: &LvlLookup,
    segment_id: usize,
    ref_frame: i32,
    y_mode: u8,
    loop_filter_sharpness: u8,
) -> Option<FilterStrength> {
    // §8.8.4 step 1: modeType then LvlLookup read.
    let mode_type = mode_to_mode_type(y_mode);
    debug_assert!(mode_type < MAX_MODE_LF_DELTAS);
    let lvl = lvl_lookup.get(segment_id, ref_frame, mode_type)?;

    // §8.8.4 step 2: shift = f(loop_filter_sharpness).
    let shift = if loop_filter_sharpness > 4 {
        2
    } else if loop_filter_sharpness > 0 {
        1
    } else {
        0
    };

    // §8.8.4 step 3: limit = Clip3 or Max(1, ..) depending on sharpness.
    let lvl_shifted = lvl >> shift;
    let limit = if loop_filter_sharpness > 0 {
        // §8.8.4 line 5649: Clip3( 1, 9 - loop_filter_sharpness, lvl >> shift ).
        // Per §6.2.8 loop_filter_sharpness is a u3, so loop_filter_sharpness
        // <= 7 and (9 - sharpness) >= 2 — Clip3's [low, high] interval is
        // always non-empty when sharpness > 0.
        let high = 9u8 - loop_filter_sharpness;
        clip3_u8(1, high, lvl_shifted)
    } else {
        // §8.8.4 line 5651: Max( 1, lvl >> shift ).
        lvl_shifted.max(1)
    };

    // §8.8.4 step 4: blimit = 2 * (lvl + 2) + limit. With lvl <= 63 and
    // limit <= 9 the maximum is 2 * 65 + 9 = 139 which fits a u8.
    let blimit = 2u16 * (lvl as u16 + 2) + limit as u16;
    debug_assert!(blimit <= u8::MAX as u16);
    let blimit = blimit as u8;

    // §8.8.4 step 5: thresh = lvl >> 4.
    let thresh = lvl >> 4;

    Some(FilterStrength {
        lvl,
        limit,
        blimit,
        thresh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_filter::loop_filter_frame_init;
    use crate::{LoopFilterParams, SegmentationParams, MAX_SEGMENTS, SEG_LVL_MAX};

    /// §8.8.4 step 1 mode→modeType mapping per `vp9-spec.txt` lines
    /// 5637-5638.
    #[test]
    fn mode_type_classifies_inter_mv_modes_vs_intra_and_zeromv() {
        // §8.8.4 line 5637: NEARESTMV / NEARMV / NEWMV → 1.
        assert_eq!(mode_to_mode_type(NEARESTMV), 1);
        assert_eq!(mode_to_mode_type(NEARMV), 1);
        assert_eq!(mode_to_mode_type(NEWMV), 1);
        // §8.8.4 line 5638: ZEROMV → 0.
        assert_eq!(mode_to_mode_type(ZEROMV), 0);
        // §8.8.4 line 5638: every §7.4.5 intra mode (0..=9) → 0.
        for intra in 0u8..=9 {
            assert_eq!(mode_to_mode_type(intra), 0, "intra mode {intra}");
        }
    }

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

    /// §8.8.4 base case: `lvlSeg = 16` broadcast (round-37 step 3),
    /// sharpness = 0, intra mode (modeType = 0). Verifies the four
    /// outputs against the §8.8.4 formulas verbatim.
    ///
    /// * lvl = 16 (LvlLookup[0][0][0]).
    /// * shift = 0; limit = Max(1, 16 >> 0) = 16.
    /// * blimit = 2 * (16 + 2) + 16 = 52.
    /// * thresh = 16 >> 4 = 1.
    #[test]
    fn sharpness_0_intra_mode_baseline_levels() {
        let mut lf = lf_disabled();
        lf.level = 16;
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0 /* INTRA */, NEARESTMV, 0).unwrap();
        // NEARESTMV → modeType = 1, but LvlLookup[0][0][1] also broadcasts to 16
        // (step 3 broadcast under delta_enabled = false).
        assert_eq!(out.lvl, 16);
        assert_eq!(out.limit, 16);
        assert_eq!(out.blimit, 52);
        assert_eq!(out.thresh, 1);
    }

    /// §8.8.4 step 2 threshold at `loop_filter_sharpness > 4`: shift =
    /// 2 (`vp9-spec.txt` line 5643). Verifies the boundary at
    /// `sharpness = 5`.
    ///
    /// With lvl = 32 and shift = 2: lvl >> shift = 8.
    /// Clip3(1, 9 - 5, 8) = Clip3(1, 4, 8) = 4. So limit = 4.
    #[test]
    fn sharpness_5_drives_shift_2_and_clip3() {
        let mut lf = lf_disabled();
        lf.level = 32;
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 5).unwrap();
        assert_eq!(out.lvl, 32);
        // shift = 2; lvl >> shift = 8; Clip3(1, 4, 8) = 4.
        assert_eq!(out.limit, 4);
        // blimit = 2 * (32 + 2) + 4 = 72.
        assert_eq!(out.blimit, 72);
        // thresh = 32 >> 4 = 2.
        assert_eq!(out.thresh, 2);
    }

    /// §8.8.4 step 2 threshold at `loop_filter_sharpness > 0` but
    /// `<= 4`: shift = 1 (`vp9-spec.txt` line 5644). Verifies the
    /// `sharpness = 1` boundary.
    ///
    /// With lvl = 40 and sharpness = 1: shift = 1; lvl >> shift = 20.
    /// Clip3(1, 9 - 1, 20) = Clip3(1, 8, 20) = 8.
    #[test]
    fn sharpness_1_drives_shift_1_and_clip3() {
        let mut lf = lf_disabled();
        lf.level = 40;
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 1).unwrap();
        assert_eq!(out.lvl, 40);
        // shift = 1; lvl >> shift = 20; Clip3(1, 8, 20) = 8.
        assert_eq!(out.limit, 8);
        // blimit = 2 * (40 + 2) + 8 = 92.
        assert_eq!(out.blimit, 92);
        // thresh = 40 >> 4 = 2.
        assert_eq!(out.thresh, 2);
    }

    /// §8.8.4 step 3 `Max(1, lvl >> shift)` lower clip at `lvl = 0`,
    /// sharpness = 0 (`vp9-spec.txt` line 5651). The Max ensures
    /// `limit >= 1` even when the level is zero, so a "filter off"
    /// row still has a well-formed strength tuple.
    #[test]
    fn sharpness_0_max_clip_at_lvl_zero() {
        let lf = lf_disabled();
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 0).unwrap();
        // lvl = 0; shift = 0; lvl >> shift = 0; Max(1, 0) = 1.
        assert_eq!(out.lvl, 0);
        assert_eq!(out.limit, 1);
        // blimit = 2 * (0 + 2) + 1 = 5.
        assert_eq!(out.blimit, 5);
        // thresh = 0 >> 4 = 0.
        assert_eq!(out.thresh, 0);
    }

    /// §8.8.4 step 3 `Clip3(1, 9 - sharpness, lvl >> shift)` lower
    /// clip at `lvl = 0`, sharpness > 0 (`vp9-spec.txt` lines
    /// 5649-5650). The Clip3 ensures `limit >= 1`.
    #[test]
    fn sharpness_nonzero_clip3_lower_bound_at_lvl_zero() {
        let lf = lf_disabled();
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 3).unwrap();
        // lvl = 0; shift = 1; lvl >> shift = 0; Clip3(1, 6, 0) = 1.
        assert_eq!(out.lvl, 0);
        assert_eq!(out.limit, 1);
        // blimit = 2 * (0 + 2) + 1 = 5.
        assert_eq!(out.blimit, 5);
        assert_eq!(out.thresh, 0);
    }

    /// §8.8.4 step 3 `Clip3(1, 9 - sharpness, …)` upper clip at
    /// max sharpness = 7: `9 - 7 = 2`, so `limit <= 2` regardless of
    /// `lvl >> shift`.
    #[test]
    fn sharpness_7_clip3_caps_limit_at_2() {
        let mut lf = lf_disabled();
        lf.level = 60;
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 7).unwrap();
        // lvl = 60; shift = 2; lvl >> shift = 15; Clip3(1, 2, 15) = 2.
        assert_eq!(out.lvl, 60);
        assert_eq!(out.limit, 2);
        // blimit = 2 * (60 + 2) + 2 = 126.
        assert_eq!(out.blimit, 126);
        // thresh = 60 >> 4 = 3.
        assert_eq!(out.thresh, 3);
    }

    /// §8.8.4 step 4 `blimit = 2 * (lvl + 2) + limit`: at the §8.8.1
    /// maximum `lvl = 63` and sharpness = 1, the shift is 1, so
    /// `lvl >> shift = 31`; then `limit = Clip3(1, 8, 31) = 8` and
    /// `blimit = 2 * 65 + 8 = 138`. Comfortably under `u8::MAX = 255`.
    #[test]
    fn blimit_high_water_at_max_lvl_sharpness_1() {
        let mut lf = lf_disabled();
        lf.level = 63;
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 1).unwrap();
        assert_eq!(out.lvl, 63);
        // shift = 1; lvl >> shift = 31; Clip3(1, 8, 31) = 8.
        assert_eq!(out.limit, 8);
        // blimit = 2 * (63 + 2) + 8 = 138.
        assert_eq!(out.blimit, 138);
        // thresh = 63 >> 4 = 3.
        assert_eq!(out.thresh, 3);
    }

    /// §8.8.4 step 1: a NEARESTMV / NEARMV / NEWMV block reads
    /// `LvlLookup[ s ][ ref ][ 1 ]` (modeType = 1) while a ZEROMV /
    /// intra block reads `LvlLookup[ s ][ ref ][ 0 ]`. Wire mode-
    /// deltas so the two slots have different values and verify the
    /// dispatch picks the right column.
    #[test]
    fn mode_type_dispatch_picks_correct_lvl_lookup_column() {
        let mut lf = lf_disabled();
        lf.level = 20;
        lf.delta_update = true;
        lf.delta_enabled = true;
        let seg = seg_disabled();
        // mode_deltas = [0, 4]. With nShift = 0 the mode 1 column gets
        // +4 over the mode 0 column for any (LAST/GOLDEN/ALTREF, mode).
        let lookup = loop_filter_frame_init(&lf, &seg, [0, 0, 0, 0], [0, 4]);
        // ZEROMV / LAST_FRAME: modeType = 0 → LvlLookup[0][1][0] = 20.
        let zero = adaptive_filter_strength(&lookup, 0, 1, ZEROMV, 0).unwrap();
        assert_eq!(zero.lvl, 20);
        // NEARMV / LAST_FRAME: modeType = 1 → LvlLookup[0][1][1] = 24.
        let near = adaptive_filter_strength(&lookup, 0, 1, NEARMV, 0).unwrap();
        assert_eq!(near.lvl, 24);
        // NEWMV picks the same column as NEARMV.
        let new = adaptive_filter_strength(&lookup, 0, 1, NEWMV, 0).unwrap();
        assert_eq!(new.lvl, 24);
        // NEARESTMV picks the same column.
        let nearest = adaptive_filter_strength(&lookup, 0, 1, NEARESTMV, 0).unwrap();
        assert_eq!(nearest.lvl, 24);
    }

    /// §8.8.4 step 1: a non-broadcast segment override propagates
    /// through to `lvl`. Combines §8.8.1's segment-override step with
    /// §8.8.4's lookup so the round-by-round wiring is exercised at
    /// the call site.
    #[test]
    fn segment_override_propagates_into_filter_strength() {
        let mut lf = lf_disabled();
        lf.level = 10;
        let mut seg = seg_disabled();
        seg.enabled = true;
        seg.abs_or_delta_update = true; // step 2.a: feature_data REPLACES lvlSeg.
        seg.feature_enabled[2][/* SEG_LVL_ALT_L */ 1] = true;
        seg.feature_data[2][1] = 50;
        let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0; 2]);
        // Segment 2, intra: LvlLookup[2][0][0] = 50.
        let out = adaptive_filter_strength(&lookup, 2, 0, 0, 0).unwrap();
        assert_eq!(out.lvl, 50);
        // limit = Max(1, 50 >> 0) = 50; blimit = 2 * 52 + 50 = 154;
        // thresh = 50 >> 4 = 3.
        assert_eq!(out.limit, 50);
        assert_eq!(out.blimit, 154);
        assert_eq!(out.thresh, 3);
    }

    /// Out-of-range axes return `None` instead of panicking. The
    /// §8.8.2 caller may legitimately encounter a `RefFrames[ ][ ][ 0
    /// ] = -1` neighbour cell (the `NONE = -1` sentinel from §6.4.16);
    /// returning `None` lets the raster walker skip the edge without
    /// pre-validating every input.
    #[test]
    fn out_of_range_axes_return_none() {
        let lookup = LvlLookup::zeros();
        // segment_id out of range.
        assert!(adaptive_filter_strength(&lookup, MAX_SEGMENTS, 0, 0, 0).is_none());
        // ref_frame out of range (negative — the NONE sentinel).
        assert!(adaptive_filter_strength(&lookup, 0, -1, 0, 0).is_none());
        // ref_frame too high.
        assert!(adaptive_filter_strength(&lookup, 0, 4, 0, 0).is_none());
    }
}
