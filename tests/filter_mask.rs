//! Integration tests for the public §8.8.5.1 [`filter_mask`] API —
//! `vp9-spec.txt` lines 5685-5792.
//!
//! Exercises the §8.8.5.1 filter mask primitive from a public
//! caller's perspective: build a 16-sample stencil
//! [`FilterMaskSamples`], read it through [`filter_mask`], and check
//! the four returned [`FilterMask`] booleans against the §8.8.5.1
//! listing verbatim.

use oxideav_vp9::{filter_mask, FilterMask, FilterMaskSamples, TX_16X16, TX_4X4, TX_8X8};

/// Build a stencil with every sample equal to `v`. With a flat
/// stencil every §8.8.5.1 abs-diff is `0`, every `> ...Bd` test is
/// false, so `hevMask = 0`, `filterMask = flatMask = flatMask2 = 1`.
fn flat(v: i32) -> FilterMaskSamples {
    FilterMaskSamples {
        p7: v,
        p6: v,
        p5: v,
        p4: v,
        p3: v,
        p2: v,
        p1: v,
        p0: v,
        q0: v,
        q1: v,
        q2: v,
        q3: v,
        q4: v,
        q5: v,
        q6: v,
        q7: v,
    }
}

/// §8.8.5.1 baseline at `TX_16X16` / `BitDepth = 8` — all four
/// outputs land at the expected default for a flat stencil.
#[test]
fn flat_stencil_tx16x16_8bit_returns_clean_mask() {
    let out = filter_mask(&flat(128), 16, 52, 1, TX_16X16, 8);
    assert_eq!(
        out,
        FilterMask {
            hev_mask: false,
            filter_mask: true,
            flat_mask: Some(true),
            flat_mask2: Some(true),
        }
    );
}

/// §8.8.5.1 lead paragraph (`vp9-spec.txt` lines 5697-5698) —
/// `flatMask` / `flatMask2` are gated by `filterSize`. At `TX_4X4`
/// neither is consulted.
#[test]
fn tx4x4_gates_flat_masks_to_none() {
    let out = filter_mask(&flat(128), 16, 52, 1, TX_4X4, 8);
    assert_eq!(out.flat_mask, None);
    assert_eq!(out.flat_mask2, None);
}

/// §8.8.5.1 lead paragraph — at `TX_8X8` `flatMask` is computed but
/// `flatMask2` is still gated off.
#[test]
fn tx8x8_gates_flat_mask2_to_none() {
    let out = filter_mask(&flat(128), 16, 52, 1, TX_8X8, 8);
    assert!(out.flat_mask.is_some());
    assert_eq!(out.flat_mask2, None);
}

/// §8.8.5.1 lines 5730-5734 — `hevMask` flips when `Abs(p1 - p0) >
/// threshBd`. At 8-bit `thresh = 4 → threshBd = 4`, so a diff of 5
/// is enough.
#[test]
fn hev_mask_triggers_on_p_side_diff() {
    let mut s = flat(128);
    s.p1 = 128 + 5;
    let out = filter_mask(&s, 16, 52, 4, TX_4X4, 8);
    assert!(out.hev_mask);
}

/// §8.8.5.1 line 5749 — the boundary term `Abs(p0 - q0)*2 + Abs(p1 -
/// q1)/2 > blimitBd` is the gate that catches large boundary
/// transitions. Set up a sample stencil that fires only on this term.
#[test]
fn boundary_term_resets_filter_mask() {
    let s = FilterMaskSamples {
        p7: 0,
        p6: 0,
        p5: 0,
        p4: 0,
        p3: 0,
        p2: 0,
        p1: 0,
        p0: 0,
        q0: 100,
        q1: 100,
        q2: 100,
        q3: 100,
        q4: 100,
        q5: 100,
        q6: 100,
        q7: 100,
    };
    // Per-pair limit large enough that |q0 - p0| = 100 cannot trip
    // the inner mask alone; blimit = 100 with |p0 - q0|*2 = 200
    // triggers the boundary term.
    let out = filter_mask(&s, 200, 100, 1, TX_4X4, 8);
    assert!(!out.filter_mask);
}

/// §8.8.5.1 lines 5759-5763 — `flatMask` polices the inner-four
/// region. With a flat boundary but a single elevated inner sample
/// the mask resets.
#[test]
fn inner_diff_resets_flat_mask() {
    let mut s = flat(128);
    // |p2 - p0| = 2 > thresholdBd = 1 (8-bit). flatMask resets.
    s.p2 = 130;
    let out = filter_mask(&s, 100, 200, 1, TX_8X8, 8);
    assert_eq!(out.flat_mask, Some(false));
}

/// §8.8.5.1 lines 5783-5790 — `flatMask2` polices the outer-four
/// ring (p4..p7 / q4..q7) relative to `p0` / `q0`. The inner mask
/// can survive while the outer mask resets.
#[test]
fn outer_ring_diff_resets_flat_mask2_only() {
    let mut s = flat(128);
    s.p7 = 130;
    let out = filter_mask(&s, 100, 200, 1, TX_16X16, 8);
    assert_eq!(out.flat_mask, Some(true));
    assert_eq!(out.flat_mask2, Some(false));
}

/// §8.8.5.1 BitDepth scaling — at 10-bit every `... << (BitDepth -
/// 8)` is a `<< 2` (4x). Build a 10-bit stencil where `thresh = 4`
/// scales to `threshBd = 16` and verify the strict `>` cutoff at
/// the boundary.
#[test]
fn bit_depth_10_scales_all_thresholds_by_4x() {
    let mut s = flat(512);
    s.p1 = 512 + 16; // equality with threshBd; stays at 0
    let out = filter_mask(&s, 16, 52, 4, TX_4X4, 10);
    assert!(!out.hev_mask);

    let mut s = flat(512);
    s.p1 = 512 + 17; // strictly greater; triggers
    let out = filter_mask(&s, 16, 52, 4, TX_4X4, 10);
    assert!(out.hev_mask);
}

/// §8.8.5.1 BitDepth scaling — at 12-bit `thresholdBd = 1 << 4 =
/// 16`, so `flatMask` only resets at diff > 16.
#[test]
fn bit_depth_12_threshold_bd_is_16() {
    let mut s = flat(2048);
    s.p2 = 2048 + 16; // 16 not > 16; flatMask stays at 1
    let out = filter_mask(&s, 100, 200, 1, TX_8X8, 12);
    assert_eq!(out.flat_mask, Some(true));

    let mut s = flat(2048);
    s.p2 = 2048 + 17;
    let out = filter_mask(&s, 100, 200, 1, TX_8X8, 12);
    assert_eq!(out.flat_mask, Some(false));
}
