//! The §8.6.1 quantizer lookups through the public API.
use oxideav_vp9::{
    MAX_SEGMENTS, QuantizationParams, SEG_LVL_MAX, SegmentationParams, ac_q, dc_q, get_ac_quant,
    get_dc_quant, get_qindex,
};

#[test]
fn table_ends() {
    assert_eq!((dc_q(8, 0), ac_q(8, 0)), (4, 4));
    assert_eq!((dc_q(8, 255), ac_q(8, 255)), (1336, 1828));
    assert_eq!((dc_q(10, 0), ac_q(10, 0)), (4, 4));
    // Clip3(0, 255, b)
    assert_eq!(dc_q(8, -9), dc_q(8, 0));
    assert_eq!(ac_q(8, 300), ac_q(8, 255));
}

#[test]
fn segment_alt_q() {
    let quant = QuantizationParams {
        base_q_idx: 100,
        delta_q_y_dc: -2,
        delta_q_uv_dc: 3,
        delta_q_uv_ac: 1,
        lossless: false,
    };
    let mut feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
    let mut feature_data = [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS];
    feature_enabled[1][0] = true;
    feature_data[1][0] = -20;
    let seg = SegmentationParams {
        enabled: true,
        update_map: false,
        tree_probs: None,
        temporal_update: false,
        pred_prob: None,
        update_data: true,
        abs_or_delta_update: false,
        feature_enabled,
        feature_data,
    };
    assert_eq!(get_qindex(&seg, &quant, 0), 100);
    assert_eq!(get_qindex(&seg, &quant, 1), 80);
    assert_eq!(get_dc_quant(0, &seg, &quant, 1, 8), dc_q(8, 78));
    assert_eq!(get_dc_quant(1, &seg, &quant, 1, 8), dc_q(8, 83));
    assert_eq!(get_ac_quant(0, &seg, &quant, 1, 8), ac_q(8, 80));
    assert_eq!(get_ac_quant(2, &seg, &quant, 1, 8), ac_q(8, 81));
}
