//! Integration tests for the public §8.8.1 [`loop_filter_frame_init`]
//! API surface — `vp9-spec.txt` lines 5465-5488.

use oxideav_vp9::{
    loop_filter_frame_init, LoopFilterParams, LvlLookup, SegmentationParams, MAX_LOOP_FILTER,
    MAX_MODE_LF_DELTAS, MAX_SEGMENTS, SEG_LVL_MAX,
};

/// `SEG_LVL_ALT_L = 1` per §3 (`vp9-spec.txt` line 476) — duplicated
/// here because the `pub(crate)` constant in the module isn't surfaced
/// on the crate's public API. Used to exercise the §8.8.1 step 2
/// segment-override gate from an external caller's perspective.
const SEG_LVL_ALT_L: usize = 1;

/// `setup_past_independence` defaults from §7.2 (`vp9-spec.txt` lines
/// 3494-3499): `loop_filter_ref_deltas = [1, 0, -1, -1]`,
/// `loop_filter_mode_deltas = [0, 0]`. These are the values every
/// keyframe-decoded stream sees on the very first frame; later frames
/// may override via the §6.2.8 `loop_filter_params( )` walker.
const PAST_INDEPENDENCE_REF_DELTAS: [i8; 4] = [1, 0, -1, -1];
const PAST_INDEPENDENCE_MODE_DELTAS: [i8; 2] = [0, 0];

fn lf_with(level: u8, delta_enabled: bool, delta_update: bool) -> LoopFilterParams {
    LoopFilterParams {
        level,
        sharpness: 0,
        delta_enabled,
        delta_update,
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

/// §8.8.1 step 3 broadcast path via the public API: `level = 16`,
/// `delta_enabled = 0`, `delta_update = 0`, no segmentation. Every
/// `LvlLookup[s][r][m]` cell should be 16.
#[test]
fn step_3_broadcast_lvl_seg_into_every_cell() {
    let lf = lf_with(16, false, false);
    let seg = seg_disabled();
    let lookup = loop_filter_frame_init(
        &lf,
        &seg,
        PAST_INDEPENDENCE_REF_DELTAS,
        PAST_INDEPENDENCE_MODE_DELTAS,
    );
    for s in 0..MAX_SEGMENTS {
        for r in 0i32..4 {
            for m in 0..MAX_MODE_LF_DELTAS {
                assert_eq!(
                    lookup.get(s, r, m),
                    Some(16),
                    "lookup[{s}][{r}][{m}] differs from broadcast lvlSeg = 16"
                );
            }
        }
    }
}

/// §8.8.1 line 5468: `nShift = loop_filter_level >> 5`. The three
/// boundary values (31 → 0, 32 → 1, 63 → 1) drive the public API's
/// per-(ref, mode) delta scaling. `delta_enabled = 1, delta_update =
/// 1, ref_deltas = [0, 1, 0, 0]` so only `LAST_FRAME / mode 0` lifts
/// off `lvlSeg`.
#[test]
fn n_shift_threshold_31_32_63_via_public_api() {
    let lf_31 = lf_with(31, true, true);
    let lf_32 = lf_with(32, true, true);
    let lf_63 = lf_with(63, true, true);
    let seg = seg_disabled();

    let r31 = loop_filter_frame_init(&lf_31, &seg, [0, 1, 0, 0], [0; 2]);
    let r32 = loop_filter_frame_init(&lf_32, &seg, [0, 1, 0, 0], [0; 2]);
    let r63 = loop_filter_frame_init(&lf_63, &seg, [0, 1, 0, 0], [0; 2]);

    // LAST_FRAME = 1, mode 0: lvlSeg + (1 << nShift).
    // level 31: nShift = 0; 31 + 1 = 32
    assert_eq!(r31.get(0, 1, 0), Some(32));
    // level 32: nShift = 1; 32 + (1<<1) = 34
    assert_eq!(r32.get(0, 1, 0), Some(34));
    // level 63: nShift = 1; 63 + (1<<1) = 65 → Clip3 → MAX_LOOP_FILTER = 63
    assert_eq!(r63.get(0, 1, 0), Some(MAX_LOOP_FILTER as u8));
}

/// §8.8.1 step 4 from a public caller: with `delta_update = 0` AND
/// `delta_enabled = 1`, step 3 broadcasts lvlSeg into every cell and
/// step 4 then overwrites every `(ref, mode)` cell *except*
/// `(INTRA_FRAME, 1)`. Verifies the post-condition the §8.8.4 consumer
/// will read.
#[test]
fn step_3_plus_step_4_leaves_intra_mode_1_at_broadcast() {
    let lf = lf_with(
        20, /* delta_enabled */ true, /* delta_update */ false,
    );
    let seg = seg_disabled();
    let lookup = loop_filter_frame_init(
        &lf,
        &seg,
        PAST_INDEPENDENCE_REF_DELTAS,
        PAST_INDEPENDENCE_MODE_DELTAS,
    );
    // INTRA_FRAME = 0, mode 0: step 4 line 5481 overwrites with 20 + 1 = 21.
    assert_eq!(lookup.get(0, 0, 0), Some(21));
    // INTRA_FRAME = 0, mode 1: NEVER written by step 4 (line 5481 is mode-0 only,
    //                          lines 5482-5487 skip INTRA_FRAME); keeps the step-3
    //                          broadcast value of 20.
    assert_eq!(lookup.get(0, 0, 1), Some(20));
    // LAST_FRAME = 1, mode 0: 20 + 0 + 0 = 20.
    assert_eq!(lookup.get(0, 1, 0), Some(20));
    // GOLDEN_FRAME = 2, mode 0: 20 + -1 + 0 = 19.
    assert_eq!(lookup.get(0, 2, 0), Some(19));
    // ALTREF_FRAME = 3, mode 1: 20 + -1 + 0 = 19.
    assert_eq!(lookup.get(0, 3, 1), Some(19));
}

/// §8.8.1 step 2 segment-override gate from a public caller: a
/// segment with `segmentation_enabled = 1` AND
/// `feature_enabled[s][SEG_LVL_ALT_L] = 1` has its `lvlSeg` overridden
/// from the segment's `feature_data` cell. Other segments keep
/// `loop_filter_level`.
#[test]
fn step_2_seg_alt_l_overrides_specific_segment_only() {
    let lf = lf_with(10, false, false);
    let mut seg = seg_disabled();
    seg.enabled = true;
    seg.abs_or_delta_update = true; // step 2.a: feature_data REPLACES lvlSeg.
    seg.feature_enabled[3][SEG_LVL_ALT_L] = true;
    seg.feature_data[3][SEG_LVL_ALT_L] = 40;
    let lookup = loop_filter_frame_init(
        &lf,
        &seg,
        PAST_INDEPENDENCE_REF_DELTAS,
        PAST_INDEPENDENCE_MODE_DELTAS,
    );
    // Segment 0 (no SEG_LVL_ALT_L): lvlSeg = 10 broadcast by step 3.
    assert_eq!(lookup.get(0, 1, 0), Some(10));
    // Segment 3 (SEG_LVL_ALT_L, abs mode): lvlSeg = 40 broadcast by step 3.
    assert_eq!(lookup.get(3, 1, 0), Some(40));
    // Segment 7 (no SEG_LVL_ALT_L): lvlSeg = 10.
    assert_eq!(lookup.get(7, 2, 1), Some(10));
}

/// `LvlLookup::zeros()` is the no-filtering identity — every cell is
/// 0, every read in-range returns `Some(0)`, every out-of-range read
/// returns `None`.
#[test]
fn lvl_lookup_zeros_identity() {
    let lookup = LvlLookup::zeros();
    assert_eq!(lookup.get(0, 0, 0), Some(0));
    assert_eq!(lookup.get(7, 3, 1), Some(0));
    assert_eq!(lookup.get(8, 0, 0), None);
    assert_eq!(lookup.get(0, 4, 0), None);
    assert_eq!(lookup.get(0, 0, 2), None);
    assert_eq!(lookup.get(0, -1, 0), None);
}
