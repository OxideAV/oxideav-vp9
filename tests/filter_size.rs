//! Integration tests for the public §8.8.3 [`filter_size`] API surface
//! — `vp9-spec.txt` lines 5587-5625.

use oxideav_vp9::{
    filter_size, PASS_HORIZONTAL, PASS_VERTICAL, TX_16X16, TX_32X32, TX_4X4, TX_8X8,
};

/// §8.8.3 line 5611 `baseSize = Min(TX_16X16, txSz)` via the public
/// API: TX_32X32 clips to TX_16X16 on an interior edge.
#[test]
fn min_clip_tx_32x32_caps_at_tx_16x16() {
    assert_eq!(
        filter_size(TX_32X32, false, PASS_VERTICAL, 64, 64, 0, 0, 32, 32),
        TX_16X16
    );
}

/// §8.8.3 line 5610 promotion via the public API: `tx_sz = TX_4X4`
/// AND `is32Edge = 1` forces `baseSize = TX_8X8`.
#[test]
fn is_32_edge_promotion_via_public_api() {
    // tx_sz = TX_4X4 alone on an interior edge stays TX_4X4.
    assert_eq!(
        filter_size(TX_4X4, false, PASS_VERTICAL, 16, 16, 0, 0, 32, 32),
        TX_4X4
    );
    // Adding is_32_edge promotes it to TX_8X8.
    assert_eq!(
        filter_size(TX_4X4, true, PASS_VERTICAL, 32, 32, 0, 0, 32, 32),
        TX_8X8
    );
}

/// §8.8.3 lines 5615-5619 vertical chroma right-edge clip sweep via
/// the public API. Walk `mi_cols` from 1 to 8 picking the right edge
/// each time; the `TX_16X16` input always clips to `TX_8X8` on
/// `sub_x = 1`.
#[test]
fn vertical_chroma_clip_sweep_at_right_edge() {
    for mi_cols in 1u32..=8u32 {
        let x = (mi_cols - 1) * 8;
        // Sub-sampled chroma: clip fires.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, x, 32, 1, 0, mi_cols, 32),
            TX_8X8,
            "mi_cols={mi_cols}"
        );
        // Full-rate chroma: no clip.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_VERTICAL, x, 32, 0, 0, mi_cols, 32),
            TX_16X16,
            "mi_cols={mi_cols}"
        );
    }
}

/// §8.8.3 lines 5620-5624 horizontal chroma bottom-edge clip sweep
/// via the public API. Mirror of the vertical sweep, varying
/// `mi_rows` and `sub_y`.
#[test]
fn horizontal_chroma_clip_sweep_at_bottom_edge() {
    for mi_rows in 1u32..=8u32 {
        let y = (mi_rows - 1) * 8;
        // Sub-sampled chroma: clip fires.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_HORIZONTAL, 32, y, 0, 1, 32, mi_rows),
            TX_8X8,
            "mi_rows={mi_rows}"
        );
        // Full-rate chroma: no clip.
        assert_eq!(
            filter_size(TX_16X16, false, PASS_HORIZONTAL, 32, y, 0, 0, 32, mi_rows),
            TX_16X16,
            "mi_rows={mi_rows}"
        );
    }
}

/// §8.8.3 lead paragraph (`vp9-spec.txt` lines 5597-5599) purpose
/// check: a luma plane (sub_x = sub_y = 0) is never clipped by
/// §8.8.3 step 2 because both clip gates require sub_x == 1 or
/// sub_y == 1 respectively. Walk an 8x8 grid of edges; every
/// TX_16X16 input stays TX_16X16.
#[test]
fn luma_plane_never_clipped_by_chroma_step() {
    let mi_cols = 8;
    let mi_rows = 8;
    for ix in 0u32..mi_cols {
        for iy in 0u32..mi_rows {
            let x = ix * 8;
            let y = iy * 8;
            // pass 0
            assert_eq!(
                filter_size(TX_16X16, false, PASS_VERTICAL, x, y, 0, 0, mi_cols, mi_rows),
                TX_16X16,
                "luma ix={ix} iy={iy} pass=0"
            );
            // pass 1
            assert_eq!(
                filter_size(
                    TX_16X16,
                    false,
                    PASS_HORIZONTAL,
                    x,
                    y,
                    0,
                    0,
                    mi_cols,
                    mi_rows
                ),
                TX_16X16,
                "luma ix={ix} iy={iy} pass=1"
            );
        }
    }
}

/// §7.4.8 constants verbatim from `vp9-spec.txt` lines 3937-3940:
/// confirm the public values match the spec.
#[test]
fn tx_size_constants_match_spec_section_7_4_8() {
    assert_eq!(TX_4X4, 0);
    assert_eq!(TX_8X8, 1);
    assert_eq!(TX_16X16, 2);
    assert_eq!(TX_32X32, 3);
}

/// §8.8.3 lines 5616 / 5621 pass-direction constants: verify the two
/// public symbols.
#[test]
fn pass_direction_constants() {
    assert_eq!(PASS_VERTICAL, 0);
    assert_eq!(PASS_HORIZONTAL, 1);
}

/// §8.8.3 composition: the `baseSize` step's `Min(TX_16X16, txSz)`
/// kicks in BEFORE the chroma clip, so a `tx_sz = TX_32X32` input on
/// the right-edge sub-sampled-chroma vertical pass still clips to
/// `TX_8X8` (because the §8.8.3 line 5611 reduction makes
/// `baseSize = TX_16X16` which the chroma gate then accepts).
#[test]
fn tx_32x32_at_right_edge_chroma_clips_via_intermediate_tx_16x16() {
    let mi_cols = 4;
    let x = (mi_cols - 1) * 8;
    assert_eq!(
        filter_size(TX_32X32, false, PASS_VERTICAL, x, 32, 1, 0, mi_cols, 32),
        TX_8X8
    );
}
