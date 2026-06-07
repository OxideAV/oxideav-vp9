//! Integration tests for the public §8.8.4
//! [`adaptive_filter_strength`] API — `vp9-spec.txt` lines 5626-5661.
//!
//! Exercises the end-to-end §8.8.1 → §8.8.4 wiring from a public
//! caller's perspective: build a [`LvlLookup`] via
//! [`loop_filter_frame_init`], then read each `(loopRow, loopCol)`
//! position through [`adaptive_filter_strength`] and verify the
//! returned [`FilterStrength`] tuple against the §8.8.4 formulas.

use oxideav_vp9::{
    adaptive_filter_strength, loop_filter_frame_init, mode_to_mode_type, FilterStrength,
    LoopFilterParams, LvlLookup, SegmentationParams, MAX_LOOP_FILTER, MAX_SEGMENTS, NEARESTMV,
    NEARMV, NEWMV, SEG_LVL_MAX, ZEROMV,
};

fn lf_with(level: u8, sharpness: u8) -> LoopFilterParams {
    LoopFilterParams {
        level,
        sharpness,
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

/// §8.8.4 end-to-end: with `loop_filter_level = 25`, sharpness = 0,
/// and the §7.2 setup_past_independence defaults, every block should
/// see `(lvl=25, limit=25, blimit=2*(25+2)+25=79, thresh=25>>4=1)`.
#[test]
fn end_to_end_level_25_sharpness_0_intra_block() {
    let lf = lf_with(25, 0);
    let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [1, 0, -1, -1], [0, 0]);
    let out = adaptive_filter_strength(
        &lookup, 0, 0, /* INTRA_FRAME */
        0, /* DC_PRED */
        0,
    )
    .expect("in-range axes");
    assert_eq!(
        out,
        FilterStrength {
            lvl: 25,
            limit: 25,
            blimit: 79,
            thresh: 1,
        }
    );
}

/// §8.8.4 step 1 inter-mode dispatch: NEARESTMV uses modeType = 1 and
/// the round-37 [`loop_filter_frame_init`] populates the mode-1 slot
/// when `loop_filter_delta_enabled == 1`. With `level = 32` and the
/// §7.2 ref/mode deltas, `LAST_FRAME / mode 1 = 32 + 0 + 0 = 32`.
#[test]
fn end_to_end_nearestmv_last_frame_with_delta_enabled() {
    let mut lf = lf_with(32, 0);
    lf.delta_enabled = true;
    let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [1, 0, -1, -1], [0, 0]);
    // LAST_FRAME = 1, NEARESTMV (modeType = 1).
    let out = adaptive_filter_strength(&lookup, 0, 1, NEARESTMV, 0).expect("in-range axes");
    // lvl = 32 (LvlLookup[0][1][1]); shift = 0; limit = Max(1, 32) = 32;
    // blimit = 2 * (32 + 2) + 32 = 100; thresh = 32 >> 4 = 2.
    assert_eq!(out.lvl, 32);
    assert_eq!(out.limit, 32);
    assert_eq!(out.blimit, 100);
    assert_eq!(out.thresh, 2);
}

/// §8.8.4 step 1 modeType partition: under `delta_enabled = 1` with a
/// non-zero mode_delta[1], the §8.8.1 step-4 inner loop populates
/// mode-1 cells differently from mode-0 cells; §8.8.4 then routes
/// inter MV modes (NEARESTMV / NEARMV / NEWMV) to the mode-1 column
/// and `ZEROMV` + intra modes to the mode-0 column.
#[test]
fn end_to_end_mode_type_routing_when_mode_deltas_split_columns() {
    let mut lf = lf_with(16, 0);
    lf.delta_enabled = true;
    lf.delta_update = true;
    let seg = seg_disabled();
    // ref_deltas = [0; 4], mode_deltas = [0, 4]. With nShift = 0:
    //   LvlLookup[s][LAST][0] = 16 + 0 + 0 = 16
    //   LvlLookup[s][LAST][1] = 16 + 0 + 4 = 20
    let lookup = loop_filter_frame_init(&lf, &seg, [0; 4], [0, 4]);
    // ZEROMV → modeType 0 → lvl = 16.
    let zero = adaptive_filter_strength(&lookup, 0, 1, ZEROMV, 0).unwrap();
    assert_eq!(zero.lvl, 16);
    // NEARMV → modeType 1 → lvl = 20.
    let near = adaptive_filter_strength(&lookup, 0, 1, NEARMV, 0).unwrap();
    assert_eq!(near.lvl, 20);
    // NEWMV → modeType 1 → lvl = 20.
    let new = adaptive_filter_strength(&lookup, 0, 1, NEWMV, 0).unwrap();
    assert_eq!(new.lvl, 20);
    // Intra modes 0..=9 → modeType 0 → lvl = 16.
    for intra in 0u8..=9 {
        let out = adaptive_filter_strength(&lookup, 0, 1, intra, 0).unwrap();
        assert_eq!(out.lvl, 16, "intra mode {intra}");
    }
}

/// §8.8.4 steps 2 + 3 over the full §6.2.8 sharpness range
/// `0..=7`. For each sharpness, compute the expected (shift, limit)
/// from the §8.8.4 listing and confirm the primitive matches at
/// `lvl = 40`.
#[test]
fn full_sharpness_sweep_matches_spec_formulas() {
    let lookup = {
        let lf = lf_with(40, 0);
        loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2])
    };
    for sharpness in 0u8..=7 {
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, sharpness).unwrap();
        let shift = if sharpness > 4 {
            2
        } else if sharpness > 0 {
            1
        } else {
            0
        };
        let lvl_shifted = 40u8 >> shift;
        let expected_limit = if sharpness > 0 {
            let high = 9u8 - sharpness;
            lvl_shifted.clamp(1, high)
        } else {
            lvl_shifted.max(1)
        };
        let expected_blimit = 2u16 * (40 + 2) + expected_limit as u16;
        assert_eq!(out.lvl, 40, "sharpness {sharpness} lvl");
        assert_eq!(out.limit, expected_limit, "sharpness {sharpness} limit");
        assert_eq!(
            out.blimit, expected_blimit as u8,
            "sharpness {sharpness} blimit"
        );
        assert_eq!(out.thresh, 40 >> 4, "sharpness {sharpness} thresh");
    }
}

/// §8.8.4 step 5 `thresh = lvl >> 4` boundaries at the §8.8.1
/// `Clip3( 0, MAX_LOOP_FILTER = 63, … )` saturation high-water mark:
/// `thresh` is `0` for `lvl ∈ 0..16`, `1` for `lvl ∈ 16..32`, `2`
/// for `lvl ∈ 32..48`, and `3` for `lvl ∈ 48..=63`.
#[test]
fn thresh_partitions_lvl_into_four_bands() {
    let lookup = LvlLookup::zeros();
    // Empty lookup yields lvl = 0 — verify the lvl = 0 floor.
    let out = adaptive_filter_strength(&lookup, 0, 0, 0, 0).unwrap();
    assert_eq!(out.thresh, 0);

    // Build a per-(loopRow, loopCol) lvl from the round-37 path and
    // confirm each band.
    for &(level, expected) in &[
        (15u8, 0u8),
        (16, 1),
        (31, 1),
        (32, 2),
        (47, 2),
        (48, 3),
        (63, 3),
    ] {
        let lf = lf_with(level, 0);
        let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
        let out = adaptive_filter_strength(&lookup, 0, 0, 0, 0).unwrap();
        assert_eq!(
            out.thresh, expected,
            "lvl = {level} should produce thresh = {expected}"
        );
    }
}

/// `mode_to_mode_type` is exported as a public helper so a future
/// §8.8.2 superblock walker can derive modeType from `YModes[ ][ ]`
/// without re-implementing the §8.8.4 step-1 classification.
#[test]
fn mode_to_mode_type_public_surface_matches_spec() {
    // §7.4.11 inter modes (10..=13).
    assert_eq!(mode_to_mode_type(NEARESTMV), 1);
    assert_eq!(mode_to_mode_type(NEARMV), 1);
    assert_eq!(mode_to_mode_type(ZEROMV), 0);
    assert_eq!(mode_to_mode_type(NEWMV), 1);
    // §7.4.5 intra modes (0..=9).
    for intra in 0u8..=9 {
        assert_eq!(mode_to_mode_type(intra), 0);
    }
}

/// `MAX_LOOP_FILTER` re-export sanity — the §8.8.1 `Clip3` ceiling
/// caps `lvl` so the §8.8.4 step 5 `thresh` saturates at 3.
#[test]
fn max_loop_filter_caps_thresh_at_three() {
    let lf = lf_with(MAX_LOOP_FILTER as u8, 0);
    let lookup = loop_filter_frame_init(&lf, &seg_disabled(), [0; 4], [0; 2]);
    let out = adaptive_filter_strength(&lookup, 0, 0, 0, 0).unwrap();
    assert_eq!(out.lvl, MAX_LOOP_FILTER as u8);
    assert_eq!(out.thresh, (MAX_LOOP_FILTER as u8) >> 4);
}
