//! Integration tests for the public §8.8.5.2 [`narrow_filter`] API —
//! `vp9-spec.txt` lines 5795-5853.
//!
//! Exercises the §8.8.5.2 narrow filter primitive from a public
//! caller's perspective: build a 4-sample stencil
//! [`NarrowFilterSamples`], call [`narrow_filter`], and check the four
//! returned [`NarrowFilterOutput`] samples against the §8.8.5.2
//! listing verbatim.

use oxideav_vp9::{narrow_filter, NarrowFilterOutput, NarrowFilterSamples};

/// Build a flat stencil with every sample equal to `v`. With a flat
/// stencil every §8.8.5.2 working-coord difference is `0`, so
/// `filter = 0`, `filter1 = filter2 = 0`, and every output equals the
/// input.
fn flat(v: i32) -> NarrowFilterSamples {
    NarrowFilterSamples {
        p1: v,
        p0: v,
        q0: v,
        q1: v,
    }
}

/// §8.8.5.2 baseline at `BitDepth = 8` with `hev_mask == 1` — flat
/// stencil yields no change.
#[test]
fn flat_stencil_8bit_hev_no_change() {
    let out = narrow_filter(&flat(128), true, 8);
    assert_eq!(
        out,
        NarrowFilterOutput {
            op1: 128,
            op0: 128,
            oq0: 128,
            oq1: 128,
        }
    );
}

/// §8.8.5.2 baseline at `BitDepth = 8` with `hev_mask == 0` — flat
/// stencil yields no change on the smooth branch either.
#[test]
fn flat_stencil_8bit_smooth_no_change() {
    let out = narrow_filter(&flat(128), false, 8);
    assert_eq!(
        out,
        NarrowFilterOutput {
            op1: 128,
            op0: 128,
            oq0: 128,
            oq1: 128,
        }
    );
}

/// §8.8.5.2 lead paragraph (lines 5806-5811) — the `hev_mask == 1`
/// branch only mutates `op0` / `oq0`; `op1` / `oq1` carry the input
/// values through unchanged so the caller can write them back
/// unconditionally.
#[test]
fn hev_branch_preserves_outer_pair() {
    let s = NarrowFilterSamples {
        p1: 100,
        p0: 110,
        q0: 145,
        q1: 155,
    };
    let out = narrow_filter(&s, true, 8);
    assert_eq!(out.op1, 100, "hev branch passes p1 through");
    assert_eq!(out.oq1, 155, "hev branch passes q1 through");
}

/// §8.8.5.2 lines 5846-5852 — the `hev_mask == 0` branch additionally
/// mutates `op1` / `oq1` via `Round2( filter1, 1 )` half-strength
/// pass. Verify both outer samples actually move when the inner step
/// is sharp enough.
#[test]
fn smooth_branch_mutates_outer_pair() {
    // Sharp inner step (p0 = 110, q0 = 145). With hev_mask == 0,
    // `filter` starts at 0, `3 * (qs0 - ps0) = 3 * 35 = 105`,
    // clamps at 105. filter1 = 109 >> 3 = 13, filter2 = 108 >> 3 = 13.
    // Round2(13, 1) = (13 + 1) >> 1 = 7 → applied to p1/q1.
    let s = NarrowFilterSamples {
        p1: 100,
        p0: 110,
        q0: 145,
        q1: 155,
    };
    let out = narrow_filter(&s, false, 8);
    assert_ne!(out.op1, 100, "smooth branch shifts p1");
    assert_ne!(out.oq1, 155, "smooth branch shifts q1");
    // Verify the actual numbers per the listing.
    assert_eq!(out.op0, 123);
    assert_eq!(out.oq0, 132);
    assert_eq!(out.op1, 107);
    assert_eq!(out.oq1, 148);
}

/// §8.8.5.2 line 5825 — `filter4_clamp` saturates at the bit-depth
/// range. At 8-bit the working range is `[-128, 127]`. A pathological
/// step at the edges of the 8-bit dynamic range tests the saturation.
#[test]
fn pathological_step_saturates_clamp_8bit() {
    let s = NarrowFilterSamples {
        p1: 255,
        p0: 255,
        q0: 0,
        q1: 0,
    };
    let out = narrow_filter(&s, true, 8);
    // qs0 - ps0 = -255 (in working coords). 3 * (-255) = -765 →
    // clamp(-765) = -128. filter1 = clamp(-128 + 4) >> 3 =
    // -124 >> 3 = -16. filter2 = clamp(-128 + 3) >> 3 = -125 >> 3 =
    // -16.
    // oq0 = clamp(-128 - (-16)) + 128 = -112 + 128 = 16.
    // op0 = clamp(127 + (-16)) + 128 = 111 + 128 = 239.
    assert_eq!(out.op0, 239);
    assert_eq!(out.oq0, 16);
}

/// §8.8.5.2 line 5814 — `BitDepth = 10` rescales `0x80 << (BitDepth -
/// 8)` to `512`. A flat-at-512 stencil at 10-bit is the working
/// equivalent of a flat-at-128 stencil at 8-bit.
#[test]
fn flat_stencil_10bit_no_change() {
    let out = narrow_filter(&flat(512), true, 10);
    assert_eq!(
        out,
        NarrowFilterOutput {
            op1: 512,
            op0: 512,
            oq0: 512,
            oq1: 512,
        }
    );
}

/// §8.8.5.2 line 5814 — `BitDepth = 12` rescales `0x80 << (BitDepth -
/// 8)` to `2048`. A flat-at-2048 stencil at 12-bit is the working
/// equivalent of a flat-at-128 stencil at 8-bit.
#[test]
fn flat_stencil_12bit_no_change() {
    let out = narrow_filter(&flat(2048), false, 12);
    assert_eq!(
        out,
        NarrowFilterOutput {
            op1: 2048,
            op0: 2048,
            oq0: 2048,
            oq1: 2048,
        }
    );
}

/// §8.8.5.2 line 5838 — when `hev_mask == 1` and `ps1 == qs1` (i.e.
/// outer samples line up), the `filter4_clamp(ps1 - qs1)` term equals
/// `0`, so the filter's inner derivation matches the `hev_mask == 0`
/// branch up through `filter1` / `filter2`. The two branches still
/// differ on `op1` / `oq1`: the hev branch leaves them alone, the
/// smooth branch runs the half-strength `Round2` pass.
#[test]
fn matched_outer_samples_collapse_hev_filter_term() {
    let s = NarrowFilterSamples {
        p1: 128,
        p0: 120,
        q0: 136,
        q1: 128,
    };
    let out_hev = narrow_filter(&s, true, 8);
    let out_smooth = narrow_filter(&s, false, 8);
    assert_eq!(out_hev.op0, out_smooth.op0, "inner pair matches");
    assert_eq!(out_hev.oq0, out_smooth.oq0, "inner pair matches");
    // Outer pair diverges — hev passes them through, smooth runs the
    // half-strength pass.
    assert_eq!(out_hev.op1, 128);
    assert_eq!(out_hev.oq1, 128);
    assert_ne!(out_smooth.op1, out_smooth.oq1, "smooth alters outer pair");
}

/// §8.8.5.2 lines 5829-5852 — round-trip property: when the four
/// inputs differ only in the inner samples but the per-side outer
/// sample matches the inner, the filter's adjustment is symmetric
/// about the boundary midpoint. Specifically the sum `op0 + oq0 ==
/// p0 + q0` should hold modulo the +3/+4 rounding asymmetry from
/// lines 5840-5841 (off by at most 1 sample).
#[test]
fn op0_oq0_symmetric_within_one() {
    for p in [50, 100, 120, 140, 180].iter() {
        for q in [50, 100, 120, 140, 180].iter() {
            let s = NarrowFilterSamples {
                p1: *p,
                p0: *p,
                q0: *q,
                q1: *q,
            };
            let out = narrow_filter(&s, false, 8);
            let diff = (out.op0 + out.oq0) - (*p + *q);
            assert!(
                diff.abs() <= 1,
                "p={} q={} op0={} oq0={} diff={}",
                p,
                q,
                out.op0,
                out.oq0,
                diff
            );
        }
    }
}
