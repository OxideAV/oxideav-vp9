//! Integration tests for the public §8.8.5.3 [`wide_filter`] API —
//! `vp9-spec.txt` lines 5855-5888.
//!
//! Exercises the §8.8.5.3 wide filter primitive from a public caller's
//! perspective: build a 16-sample stencil [`WideFilterSamples`], call
//! [`wide_filter`] with `log2_size ∈ {3, 4}` per the §8.8.5 dispatch
//! table (`vp9-spec.txt` lines 5681-5684), and check the up-to-14
//! returned [`WideFilterOutput`] samples against the §8.8.5.3 listing
//! verbatim.

use oxideav_vp9::{wide_filter, WideFilterOutput, WideFilterSamples};

/// Build a stencil with every sample equal to `v`. The §8.8.5.3
/// kernel is a normalised low-pass so every output equals `v`.
fn flat(v: i32) -> WideFilterSamples {
    WideFilterSamples {
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

/// §8.8.5.3 unity-gain at `log2_size = 3` (8-tap). A flat stencil
/// at the 8-bit midpoint (128) yields 128 at every output position.
#[test]
fn flat_8bit_log2_3() {
    let out = wide_filter(&flat(128), 3, 8);
    assert_eq!(
        out,
        WideFilterOutput {
            op6: 128,
            op5: 128,
            op4: 128,
            op3: 128,
            op2: 128,
            op1: 128,
            op0: 128,
            oq0: 128,
            oq1: 128,
            oq2: 128,
            oq3: 128,
            oq4: 128,
            oq5: 128,
            oq6: 128,
        }
    );
}

/// §8.8.5.3 unity-gain at `log2_size = 4` (16-tap). A flat stencil
/// at the 8-bit midpoint (128) yields 128 at every output position.
#[test]
fn flat_8bit_log2_4() {
    let out = wide_filter(&flat(128), 4, 8);
    assert_eq!(
        out,
        WideFilterOutput {
            op6: 128,
            op5: 128,
            op4: 128,
            op3: 128,
            op2: 128,
            op1: 128,
            op0: 128,
            oq0: 128,
            oq1: 128,
            oq2: 128,
            oq3: 128,
            oq4: 128,
            oq5: 128,
            oq6: 128,
        }
    );
}

/// §8.8.5.3 unity-gain at `BitDepth = 10`. The §8.8.5.3 listing makes
/// no reference to `BitDepth` so the kernel behaves identically
/// across all three depths — only the working range of valid sample
/// values changes (0..=1023 for 10-bit).
#[test]
fn flat_10bit_log2_3() {
    let out = wide_filter(&flat(512), 3, 10);
    assert_eq!(out.op0, 512);
    assert_eq!(out.oq0, 512);
}

/// §8.8.5.3 unity-gain at `BitDepth = 12`. The 16-tap kernel sums
/// `2n + 2 = 16` samples each at most `4095`, so the accumulator
/// peaks at `65520` — comfortably within `i32`.
#[test]
fn flat_12bit_log2_4() {
    let out = wide_filter(&flat(4095), 4, 12);
    assert_eq!(out.op0, 4095);
    assert_eq!(out.oq0, 4095);
    assert_eq!(out.op6, 4095);
    assert_eq!(out.oq6, 4095);
}

/// §8.8.5.3 log2_3 — outer fields (`op6..op3`, `oq3..oq6`) echo the
/// corresponding input through unchanged. The 8-tap kernel only
/// touches positions `[-3, 2]` (mapping to `p2..p0` and `q0..q2`);
/// the §8.8.5 outer driver writes all 14 returned fields back so we
/// echo the input on positions the kernel doesn't compute.
#[test]
fn log2_3_outer_fields_echo() {
    let s = WideFilterSamples {
        p7: 99,
        p6: 11,
        p5: 22,
        p4: 33,
        p3: 0,
        p2: 0,
        p1: 0,
        p0: 0,
        q0: 0,
        q1: 0,
        q2: 0,
        q3: 0,
        q4: 44,
        q5: 55,
        q6: 66,
        q7: 77,
    };
    let out = wide_filter(&s, 3, 8);
    // op6..op3 echo p6..p3.
    assert_eq!(out.op6, 11);
    assert_eq!(out.op5, 22);
    assert_eq!(out.op4, 33);
    assert_eq!(out.op3, 0);
    // oq3..oq6 echo q3..q6.
    assert_eq!(out.oq3, 0);
    assert_eq!(out.oq4, 44);
    assert_eq!(out.oq5, 55);
    assert_eq!(out.oq6, 66);
    // Inner positions are filtered (all zeros → all zeros).
    assert_eq!(out.op2, 0);
    assert_eq!(out.op1, 0);
    assert_eq!(out.op0, 0);
    assert_eq!(out.oq0, 0);
    assert_eq!(out.oq1, 0);
    assert_eq!(out.oq2, 0);
}

/// §8.8.5.3 log2_3 step-response from a flat 0 (p side) to a flat
/// 100 (q side) — verify the exact values produced at the 6 mutated
/// positions, derived by hand from the §8.8.5.3 listing.
///
/// At `i = -3` (`op2`): `t = p2 + p3 + p3 + p3 + p2 + p1 + p0 + q0`
/// `= 0 + 0 + 0 + 0 + 0 + 0 + 0 + 100 = 100`.
/// `Round2(100, 3) = (100 + 4) >> 3 = 13`.
///
/// At `i = -2` (`op1`): `t = p1 + p3 + p3 + p2 + p1 + p0 + q0 + q1`
/// `= 0 + 0 + 0 + 0 + 0 + 0 + 100 + 100 = 200`.
/// `Round2(200, 3) = (200 + 4) >> 3 = 25`.
///
/// At `i = -1` (`op0`): `t = p0 + p3 + p2 + p1 + p0 + q0 + q1 + q2`
/// `= 0 + 0 + 0 + 0 + 0 + 100 + 100 + 100 = 300`.
/// `Round2(300, 3) = (300 + 4) >> 3 = 38`.
///
/// At `i = 0` (`oq0`): `t = q0 + p2 + p1 + p0 + q0 + q1 + q2 + q3`
/// `= 100 + 0 + 0 + 0 + 100 + 100 + 100 + 100 = 500`.
/// `Round2(500, 3) = (500 + 4) >> 3 = 63`.
///
/// At `i = 1` (`oq1`): `t = q1 + p1 + p0 + q0 + q1 + q2 + q3 + q3`
/// `= 100 + 0 + 0 + 100 + 100 + 100 + 100 + 100 = 600`.
/// `Round2(600, 3) = (600 + 4) >> 3 = 75`.
///
/// At `i = 2` (`oq2`): `t = q2 + p0 + q0 + q1 + q2 + q3 + q3 + q3`
/// `= 100 + 0 + 100 + 100 + 100 + 100 + 100 + 100 = 700`.
/// `Round2(700, 3) = (700 + 4) >> 3 = 88`.
#[test]
fn log2_3_step_response_hand_traced() {
    let s = WideFilterSamples {
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
    let out = wide_filter(&s, 3, 8);
    assert_eq!(out.op2, 13);
    assert_eq!(out.op1, 25);
    assert_eq!(out.op0, 38);
    assert_eq!(out.oq0, 63);
    assert_eq!(out.oq1, 75);
    assert_eq!(out.oq2, 88);
}

/// §8.8.5.3 log2_4 step-response from a flat 0 (p side) to a flat
/// 128 (q side) — verify the 16-tap kernel at the boundary.
///
/// At `i = -1` (`op0`) with `n = 7`: initial `t = sample(-1) = p0
/// = 0`. The inner loop walks `j ∈ [-7, 7]` (15 iterations); `i + j
/// ∈ [-8, 6]` falls inside the clamp window `[-8, 7]` so no
/// `Clip3` extension is triggered. The 15 sampled positions are
/// `p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6` —
/// eight zeros (p side) plus seven copies of 128 (q side, q0..q6)
/// = `7 * 128 = 896`. Plus the initial `t = p0 = 0`. Total
/// `t = 896`. Per §8.8.5.3 line 5882, `Round2(896, 4) = (896 + 8)
/// >> 4 = 904 >> 4 = 56`.
#[test]
fn log2_4_step_response_op0() {
    let s = WideFilterSamples {
        p7: 0,
        p6: 0,
        p5: 0,
        p4: 0,
        p3: 0,
        p2: 0,
        p1: 0,
        p0: 0,
        q0: 128,
        q1: 128,
        q2: 128,
        q3: 128,
        q4: 128,
        q5: 128,
        q6: 128,
        q7: 128,
    };
    let out = wide_filter(&s, 4, 8);
    assert_eq!(out.op0, 56);
}

/// §8.8.5.3 line 5879 `Clip3` edge-replication — when the kernel
/// reaches outside `[-(n+1), n]`, the outermost in-range sample is
/// duplicated. Verified by isolating `p3` at 80 in an otherwise-zero
/// 8-tap stencil: at `i = -3` (which outputs `op2`), the clamp picks
/// up p3 three additional times (for `j = -3, -2, -1` all mapping
/// to `-(n+1) = -4`).
///
/// Hand trace at `i = -3`: t = sample(-3) = p2 = 0. Inner loop j ∈
/// [-3, 3]:
///   j=-3: i+j=-6 → clamp -4 → p3 = 80
///   j=-2: -5    → clamp -4 → p3 = 80
///   j=-1: -4              → p3 = 80
///   j= 0: -3              → p2 = 0
///   j= 1: -2              → p1 = 0
///   j= 2: -1              → p0 = 0
///   j= 3:  0              → q0 = 0
/// t = 0 + 240 = 240. Round2(240, 3) = (240 + 4) >> 3 = 30.
#[test]
fn log2_3_clip3_replicates_p3() {
    let s = WideFilterSamples {
        p7: 0,
        p6: 0,
        p5: 0,
        p4: 0,
        p3: 80,
        p2: 0,
        p1: 0,
        p0: 0,
        q0: 0,
        q1: 0,
        q2: 0,
        q3: 0,
        q4: 0,
        q5: 0,
        q6: 0,
        q7: 0,
    };
    let out = wide_filter(&s, 3, 8);
    assert_eq!(out.op2, 30);
}

/// §8.8.5 dispatch — `log2_size` outside `{3, 4}` panics. The §8.8.5
/// outer driver's two-branch dispatch (lines 5682 / 5684) is the
/// only producer of `log2_size` values, so any other value is a
/// hard precondition violation.
#[test]
#[should_panic(expected = "§8.8.5.3: log2_size must be 3 or 4")]
fn log2_size_panics() {
    let _ = wide_filter(&flat(128), 7, 8);
}
