//! Integration tests for the public §8.8.5 [`sample_filtering`] API —
//! `vp9-spec.txt` lines 5662-5684.
//!
//! Exercises the §8.8.5 outer driver from a public caller's
//! perspective: build a 16-sample stencil [`SampleFilterSamples`],
//! call [`sample_filtering`], and confirm the four-way dispatch
//! (`filterMask == 0` no-op / narrow / wide-`log2Size`-3 /
//! wide-`log2Size`-4 per `vp9-spec.txt` lines 5678-5684) routes to the
//! right sub-process. Cross-checks the returned [`SampleFilterOutput`]
//! against the public §8.8.5.1 [`filter_mask`], §8.8.5.2
//! [`narrow_filter`] and §8.8.5.3 [`wide_filter`] primitives.

use oxideav_vp9::{
    filter_mask, narrow_filter, sample_filtering, wide_filter, FilterMaskSamples,
    NarrowFilterSamples, SampleFilterOutput, SampleFilterSamples, WideFilterSamples, TX_16X16,
    TX_4X4, TX_8X8,
};

/// Build a flat stencil at `v`.
fn flat(v: i32) -> SampleFilterSamples {
    SampleFilterSamples {
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

/// Project a [`SampleFilterSamples`] onto the §8.8.5.1 stencil.
fn mask_samples(s: &SampleFilterSamples) -> FilterMaskSamples {
    FilterMaskSamples {
        p7: s.p7,
        p6: s.p6,
        p5: s.p5,
        p4: s.p4,
        p3: s.p3,
        p2: s.p2,
        p1: s.p1,
        p0: s.p0,
        q0: s.q0,
        q1: s.q1,
        q2: s.q2,
        q3: s.q3,
        q4: s.q4,
        q5: s.q5,
        q6: s.q6,
        q7: s.q7,
    }
}

/// Project a [`SampleFilterSamples`] onto the §8.8.5.3 stencil.
fn wide_samples(s: &SampleFilterSamples) -> WideFilterSamples {
    WideFilterSamples {
        p7: s.p7,
        p6: s.p6,
        p5: s.p5,
        p4: s.p4,
        p3: s.p3,
        p2: s.p2,
        p1: s.p1,
        p0: s.p0,
        q0: s.q0,
        q1: s.q1,
        q2: s.q2,
        q3: s.q3,
        q4: s.q4,
        q5: s.q5,
        q6: s.q6,
        q7: s.q7,
    }
}

fn assert_echo(s: &SampleFilterSamples, out: &SampleFilterOutput) {
    assert_eq!(out.p7, s.p7);
    assert_eq!(out.p6, s.p6);
    assert_eq!(out.p5, s.p5);
    assert_eq!(out.p4, s.p4);
    assert_eq!(out.p3, s.p3);
    assert_eq!(out.p2, s.p2);
    assert_eq!(out.p1, s.p1);
    assert_eq!(out.p0, s.p0);
    assert_eq!(out.q0, s.q0);
    assert_eq!(out.q1, s.q1);
    assert_eq!(out.q2, s.q2);
    assert_eq!(out.q3, s.q3);
    assert_eq!(out.q4, s.q4);
    assert_eq!(out.q5, s.q5);
    assert_eq!(out.q6, s.q6);
    assert_eq!(out.q7, s.q7);
}

/// §8.8.5 baseline — a flat boundary passes `filterMask` but every
/// filter branch is the identity on a flat region (every `filterSize`,
/// 8-bit).
#[test]
fn flat_stencil_identity_all_sizes() {
    let s = flat(128);
    for &fsize in &[TX_4X4, TX_8X8, TX_16X16] {
        assert_echo(&s, &sample_filtering(&s, 9, 80, 4, fsize, 8));
    }
}

/// §8.8.5 line 5678 — `filterMask == 0` echoes the stencil verbatim.
/// A `limit`-tripping inner jump resets `filterMask`.
#[test]
fn filter_mask_zero_is_noop() {
    let mut s = flat(128);
    s.q2 = 0;
    s.q3 = 250;
    // Confirm the mask actually resets via the public primitive.
    let m = filter_mask(&mask_samples(&s), 1, 255, 4, TX_16X16, 8);
    assert!(!m.filter_mask);
    for &fsize in &[TX_4X4, TX_8X8, TX_16X16] {
        assert_echo(&s, &sample_filtering(&s, 1, 255, 4, fsize, 8));
    }
}

/// §8.8.5 line 5679 — `filterSize == TX_4X4` routes to the §8.8.5.2
/// narrow filter; the result matches the narrow primitive on the
/// 4-sample window and leaves the rest untouched.
#[test]
fn tx4x4_routes_to_narrow_filter() {
    let s = SampleFilterSamples {
        p7: 100,
        p6: 100,
        p5: 100,
        p4: 100,
        p3: 100,
        p2: 101,
        p1: 102,
        p0: 104,
        q0: 110,
        q1: 112,
        q2: 113,
        q3: 114,
        q4: 114,
        q5: 114,
        q6: 114,
        q7: 114,
    };
    let m = filter_mask(&mask_samples(&s), 9, 80, 4, TX_4X4, 8);
    assert!(m.filter_mask);
    let nf = narrow_filter(
        &NarrowFilterSamples {
            p1: s.p1,
            p0: s.p0,
            q0: s.q0,
            q1: s.q1,
        },
        m.hev_mask,
        8,
    );

    let out = sample_filtering(&s, 9, 80, 4, TX_4X4, 8);
    assert_eq!(out.p1, nf.op1);
    assert_eq!(out.p0, nf.op0);
    assert_eq!(out.q0, nf.oq0);
    assert_eq!(out.q1, nf.oq1);
    assert_eq!(out.p2, s.p2);
    assert_eq!(out.q2, s.q2);
}

/// §8.8.5 line 5681 — `filterSize == TX_8X8` with a flat inner region
/// routes to the §8.8.5.3 wide filter at `log2Size == 3`; result
/// matches the wide primitive's inner six outputs.
#[test]
fn tx8x8_flat_routes_to_wide_log2_3() {
    let mut s = flat(100);
    s.q0 = 101;
    s.q1 = 101;
    s.q2 = 101;
    s.q3 = 101;
    s.q4 = 101;
    s.q5 = 101;
    s.q6 = 101;
    s.q7 = 101;
    let m = filter_mask(&mask_samples(&s), 9, 80, 4, TX_8X8, 8);
    assert!(m.filter_mask);
    assert_eq!(m.flat_mask, Some(true));

    let wf = wide_filter(&wide_samples(&s), 3, 8);
    let out = sample_filtering(&s, 9, 80, 4, TX_8X8, 8);
    assert_eq!(out.p2, wf.op2);
    assert_eq!(out.p0, wf.op0);
    assert_eq!(out.q0, wf.oq0);
    assert_eq!(out.q2, wf.oq2);
    // log2Size == 3 leaves p3 / q3 untouched.
    assert_eq!(out.p3, s.p3);
    assert_eq!(out.q3, s.q3);
}

/// §8.8.5 lines 5683-5684 — `filterSize == TX_16X16` with a fully flat
/// region routes to the §8.8.5.3 wide filter at `log2Size == 4`.
#[test]
fn tx16x16_fully_flat_routes_to_wide_log2_4() {
    let mut s = flat(100);
    s.q0 = 101;
    s.q1 = 101;
    s.q2 = 101;
    s.q3 = 101;
    s.q4 = 101;
    s.q5 = 101;
    s.q6 = 101;
    s.q7 = 101;
    let m = filter_mask(&mask_samples(&s), 9, 80, 4, TX_16X16, 8);
    assert!(m.filter_mask);
    assert_eq!(m.flat_mask, Some(true));
    assert_eq!(m.flat_mask2, Some(true));

    let wf = wide_filter(&wide_samples(&s), 4, 8);
    let out = sample_filtering(&s, 9, 80, 4, TX_16X16, 8);
    assert_eq!(out.p6, wf.op6);
    assert_eq!(out.p0, wf.op0);
    assert_eq!(out.q0, wf.oq0);
    assert_eq!(out.q6, wf.oq6);
    // Only p7 / q7 are echoed through.
    assert_eq!(out.p7, s.p7);
    assert_eq!(out.q7, s.q7);
}

/// §8.8.5 line 5682 — `flatMask2 == 0` at `TX_16X16` drops the
/// dispatch from `log2Size == 4` back to `log2Size == 3`.
#[test]
fn flat_mask2_zero_drops_to_wide_log2_3() {
    let mut s = flat(100);
    s.p7 = 130; // outer-ring outlier resets flatMask2.
    s.q0 = 101;
    s.q1 = 101;
    s.q2 = 101;
    s.q3 = 101;
    s.q4 = 101;
    s.q5 = 101;
    s.q6 = 101;
    s.q7 = 101;
    let m = filter_mask(&mask_samples(&s), 9, 80, 4, TX_16X16, 8);
    assert!(m.filter_mask);
    assert_eq!(m.flat_mask, Some(true));
    assert_eq!(m.flat_mask2, Some(false));

    let wf3 = wide_filter(&wide_samples(&s), 3, 8);
    let out = sample_filtering(&s, 9, 80, 4, TX_16X16, 8);
    assert_eq!(out.p2, wf3.op2);
    assert_eq!(out.q2, wf3.oq2);
    assert_eq!(out.p3, s.p3); // untouched by log2Size == 3.
    assert_eq!(out.q3, s.q3);
}

/// §8.8.5 line 5679 — `flatMask == 0` forces the narrow branch even
/// at `filterSize == TX_8X8` (a non-flat inner four samples).
#[test]
fn flat_mask_zero_forces_narrow_at_tx8x8() {
    let mut s = flat(100);
    s.p2 = 105; // p2 vs p0 diff > thresholdBd=1 → flatMask resets.
    s.p1 = 102;
    let m = filter_mask(&mask_samples(&s), 9, 80, 4, TX_8X8, 8);
    assert!(m.filter_mask);
    assert_eq!(m.flat_mask, Some(false));

    let nf = narrow_filter(
        &NarrowFilterSamples {
            p1: s.p1,
            p0: s.p0,
            q0: s.q0,
            q1: s.q1,
        },
        m.hev_mask,
        8,
    );
    let out = sample_filtering(&s, 9, 80, 4, TX_8X8, 8);
    assert_eq!(out.p0, nf.op0);
    assert_eq!(out.q0, nf.oq0);
    // Narrow window excludes p2 — it stays at the input.
    assert_eq!(out.p2, s.p2);
}

/// §8.8.5 BitDepth propagation — a flat 12-bit stencil at the midpoint
/// (2048) is the identity on every branch.
#[test]
fn flat_stencil_identity_12bit() {
    let s = flat(2048);
    for &fsize in &[TX_4X4, TX_8X8, TX_16X16] {
        assert_echo(&s, &sample_filtering(&s, 9, 80, 4, fsize, 12));
    }
}
