//! VP9 §8.8.5 `sample filtering process` — per spec v0.7.
//!
//! This module lands the per-edge [`sample_filtering`] outer driver as
//! a pure leaf primitive. It is the dispatcher the §8.8.2 superblock
//! raster walk invokes at every loop-filter edge (after §8.8.3
//! `filter_size` and §8.8.4 `adaptive_filter_strength` produce the
//! `filterSize` and `(limit, blimit, thresh)` inputs): it runs the
//! round-253 §8.8.5.1 [`crate::filter_mask`] step on the boundary
//! stencil, then dispatches to the round-255 §8.8.5.2
//! [`crate::narrow_filter`] or the round-259 §8.8.5.3
//! [`crate::wide_filter`] based on the four masks the first step
//! returns.
//!
//! The §8.8.5 listing (`vp9-spec.txt` lines 5662-5684) is the
//! decision step:
//!
//! ```text
//! First the filter mask process specified in section 8.8.5.1 is
//! invoked … the output is assigned to hevMask, filterMask, flatMask,
//! and flatMask2.
//!
//! Then the appropriate filter process is invoked as follows:
//! − If filterMask is equal to 0, no filter is invoked.
//! − Otherwise, if filterSize is equal to TX_4X4 or flatMask is equal
//!   to 0, the narrow filter process (8.8.5.2) is invoked with the
//!   additional input variable hevMask.
//! − Otherwise, if filterSize is equal to TX_8X8 or flatMask2 is equal
//!   to 0, the wide filter process (8.8.5.3) is invoked with log2Size
//!   set to 3.
//! − Otherwise, the wide filter process (8.8.5.3) is invoked with
//!   log2Size set to 4.
//! ```
//!
//! ## Scope of this round
//!
//! This round lands the §8.8.5 dispatch only — a pure-state function
//! over a fixed 16-sample stencil [`SampleFilterSamples`] (`p7`..`p0`
//! / `q0`..`q7`). The caller is responsible for fetching the stencil
//! from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per §8.8.5.1
//! lines 5703-5727 and writing the returned [`SampleFilterOutput`]
//! back to the matching positions — this primitive does not walk
//! `(plane, x, y, dx, dy)` itself.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.2 `superblock_loop_filter( )` — the per-superblock raster
//!   walk that assembles the stencil from `CurrFrame`, derives
//!   `(filterSize, limit, blimit, thresh)` via §8.8.3 + §8.8.4, calls
//!   this primitive for each `(plane, pass, row, col)` edge, and
//!   writes [`SampleFilterOutput`] back into `CurrFrame`.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.5 lines 5662-5684). The three
//! sub-processes (§8.8.5.1 / §8.8.5.2 / §8.8.5.3) are the leaf
//! primitives landed in earlier rounds; this round only composes
//! them per the §8.8.5 dispatch table.

use crate::filter_mask::{filter_mask, FilterMaskSamples};
use crate::filter_size::{TX_4X4, TX_8X8};
use crate::narrow_filter::{narrow_filter, NarrowFilterSamples};
use crate::wide_filter::{wide_filter, WideFilterSamples};

/// §8.8.5 input — the 16-sample stencil straddling the boundary.
///
/// Per `vp9-spec.txt` §8.8.5.1 lines 5703-5727, the §8.8.2 raster
/// walk reads these from `CurrFrame[ plane ][ y +/- dy*k ][ x +/-
/// dx*k ]`:
///
/// * `q0 = CurrFrame[ plane ][ y ][ x ]`
/// * `q[k] = CurrFrame[ plane ][ y + dy*k ][ x + dx*k ]` for `k =
///   1..=7`
/// * `p[k] = CurrFrame[ plane ][ y - dy*(k+1) ][ x - dx*(k+1) ]` for
///   `k = 0..=7` (so `p0 = CurrFrame[ plane ][ y - dy ][ x - dx ]`,
///   `p7 = CurrFrame[ plane ][ y - dy*8 ][ x - dx*8 ]`).
///
/// The outer samples (`p7..p4`, `q4..q7`) are only consulted by the
/// §8.8.5.1 `flatMask2` step and the §8.8.5.3 `log2Size == 4` kernel,
/// both of which require `filterSize == TX_16X16`. For narrower
/// `filterSize` they may carry any value and are echoed straight to
/// the output untouched.
///
/// Samples are carried as `i32` so the §8.8.5.1 abs-difference and
/// §8.8.5.2 signed working-range arithmetic don't underflow for
/// 10-bit / 12-bit pixels (whose values reach `(1 << 12) - 1 =
/// 4095`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleFilterSamples {
    /// `p7` — outermost sample on the `p` side.
    pub p7: i32,
    /// `p6`.
    pub p6: i32,
    /// `p5`.
    pub p5: i32,
    /// `p4`.
    pub p4: i32,
    /// `p3`.
    pub p3: i32,
    /// `p2`.
    pub p2: i32,
    /// `p1`.
    pub p1: i32,
    /// `p0` — boundary sample on the `p` side.
    pub p0: i32,
    /// `q0` — boundary sample on the `q` side.
    pub q0: i32,
    /// `q1`.
    pub q1: i32,
    /// `q2`.
    pub q2: i32,
    /// `q3`.
    pub q3: i32,
    /// `q4`.
    pub q4: i32,
    /// `q5`.
    pub q5: i32,
    /// `q6`.
    pub q6: i32,
    /// `q7` — outermost sample on the `q` side.
    pub q7: i32,
}

/// §8.8.5 output — the full 16-sample post-filter stencil.
///
/// Every §8.8.5 branch mutates only an inner sub-window of the
/// stencil; the untouched outer samples are echoed straight through.
/// The caller writes the whole stencil back to `CurrFrame[ plane ][ y
/// +/- dy*k ][ x +/- dx*k ]` unconditionally — positions the chosen
/// filter did not mutate carry the original value, so the write is a
/// no-op for them.
///
/// Per `vp9-spec.txt` lines 5678-5684 the mutated window is:
///
/// * `filterMask == 0` → none (the stencil is echoed verbatim).
/// * narrow filter (`filterSize == TX_4X4` or `flatMask == 0`) →
///   `p1`, `p0`, `q0`, `q1` (and `p1` / `q1` are only changed when
///   `hevMask == 0` per §8.8.5.2; the §8.8.5.2 primitive returns the
///   input there otherwise).
/// * wide filter `log2Size == 3` (`filterSize == TX_8X8` or
///   `flatMask2 == 0`) → `p2..p0`, `q0..q2`.
/// * wide filter `log2Size == 4` → `p6..p0`, `q0..q6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleFilterOutput {
    /// `p7` — never mutated by any §8.8.5 branch (position `-8` lies
    /// outside the `i ∈ [-n, n-1]` write window even for `log2Size ==
    /// 4`).
    pub p7: i32,
    /// `p6` — mutated only by the wide `log2Size == 4` branch.
    pub p6: i32,
    /// `p5` — mutated only by the wide `log2Size == 4` branch.
    pub p5: i32,
    /// `p4` — mutated only by the wide `log2Size == 4` branch.
    pub p4: i32,
    /// `p3` — mutated only by the wide `log2Size == 4` branch.
    pub p3: i32,
    /// `p2` — mutated by either wide branch.
    pub p2: i32,
    /// `p1` — mutated by either wide branch, or by the narrow branch
    /// when `hevMask == 0`.
    pub p1: i32,
    /// `p0` — mutated by every filtering branch.
    pub p0: i32,
    /// `q0` — mutated by every filtering branch.
    pub q0: i32,
    /// `q1` — mutated by either wide branch, or by the narrow branch
    /// when `hevMask == 0`.
    pub q1: i32,
    /// `q2` — mutated by either wide branch.
    pub q2: i32,
    /// `q3` — mutated only by the wide `log2Size == 4` branch.
    pub q3: i32,
    /// `q4` — mutated only by the wide `log2Size == 4` branch.
    pub q4: i32,
    /// `q5` — mutated only by the wide `log2Size == 4` branch.
    pub q5: i32,
    /// `q6` — mutated only by the wide `log2Size == 4` branch.
    pub q6: i32,
    /// `q7` — never mutated by any §8.8.5 branch (position `+7` lies
    /// outside the `i ∈ [-n, n-1]` write window even for `log2Size ==
    /// 4`).
    pub q7: i32,
}

impl SampleFilterOutput {
    /// Build the echo-through output — every position carries the
    /// original stencil sample. Used by the `filterMask == 0` arm and
    /// as the starting point each filter branch overwrites in place.
    #[inline]
    fn echo(s: &SampleFilterSamples) -> Self {
        SampleFilterOutput {
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
}

/// Run §8.8.5 `sample filtering process` for one edge per
/// `vp9-spec.txt` lines 5662-5684.
///
/// Runs the §8.8.5.1 [`filter_mask`] step on `samples`, then
/// dispatches to the §8.8.5.2 [`narrow_filter`] or the §8.8.5.3
/// [`wide_filter`] per the §8.8.5 dispatch table (lines 5678-5684).
/// Returns the full 16-sample post-filter stencil in
/// [`SampleFilterOutput`]; positions outside the chosen filter's
/// mutation window are echoed unchanged so the caller can write the
/// whole stencil back to `CurrFrame` unconditionally.
///
/// # Inputs
///
/// * `samples` — the 16-sample stencil [`SampleFilterSamples`] the
///   §8.8.2 raster walk assembled from `CurrFrame[ plane ][ y +/-
///   dy*k ][ x +/- dx*k ]` per §8.8.5.1 lines 5703-5727.
/// * `limit` / `blimit` / `thresh` — the §8.8.4 [`crate::
///   FilterStrength`] tuple, forwarded to the §8.8.5.1 mask step.
/// * `filter_size` — the §8.8.3 [`crate::filter_size`] output: one of
///   `TX_4X4` / `TX_8X8` / `TX_16X16` (the §8.8.3 step caps it below
///   `TX_32X32`). Drives both the §8.8.5.1 `flatMask` / `flatMask2`
///   gating and the §8.8.5 dispatch table.
/// * `bit_depth` — `BitDepth` per §6.2.2 (8, 10, or 12). Forwarded to
///   all three sub-processes.
///
/// # Listing
///
/// `vp9-spec.txt` §8.8.5 lines 5677-5684:
///
/// ```text
/// if filterMask == 0:        no filter
/// elif filterSize == TX_4X4 or flatMask == 0:
///                            narrow filter (8.8.5.2), input hevMask
/// elif filterSize == TX_8X8 or flatMask2 == 0:
///                            wide filter (8.8.5.3), log2Size = 3
/// else:                      wide filter (8.8.5.3), log2Size = 4
/// ```
///
/// The `flatMask` / `flatMask2` reads in the dispatch table are only
/// reached when the matching `filterSize >=` precondition (§8.8.5.1
/// lines 5697-5698) is met: the `flatMask == 0` test sits behind a
/// `filterSize == TX_4X4` short-circuit, and the `flatMask2 == 0`
/// test sits behind a `filterSize == TX_8X8` short-circuit. So the
/// §8.8.5.1 `None` returns for those masks are never dereferenced.
pub fn sample_filtering(
    samples: &SampleFilterSamples,
    limit: u8,
    blimit: u8,
    thresh: u8,
    filter_size: u8,
    bit_depth: u8,
) -> SampleFilterOutput {
    // §8.8.5 lines 5672-5674 — run the §8.8.5.1 filter mask process
    // first. The stencils share field layout, so this is a direct
    // field-for-field hand-off.
    let mask_samples = FilterMaskSamples {
        p7: samples.p7,
        p6: samples.p6,
        p5: samples.p5,
        p4: samples.p4,
        p3: samples.p3,
        p2: samples.p2,
        p1: samples.p1,
        p0: samples.p0,
        q0: samples.q0,
        q1: samples.q1,
        q2: samples.q2,
        q3: samples.q3,
        q4: samples.q4,
        q5: samples.q5,
        q6: samples.q6,
        q7: samples.q7,
    };
    let mask = filter_mask(&mask_samples, limit, blimit, thresh, filter_size, bit_depth);

    // §8.8.5 line 5678 — `if filterMask == 0, no filter is invoked`.
    if !mask.filter_mask {
        return SampleFilterOutput::echo(samples);
    }

    // §8.8.5 line 5679 — narrow filter when `filterSize == TX_4X4` OR
    // `flatMask == 0`. The `filterSize == TX_4X4` arm short-circuits
    // before `flatMask` (which is `None` exactly when `filterSize ==
    // TX_4X4`) is read, so the `unwrap_or` default is only consulted
    // on the `filterSize >= TX_8X8` paths where `flatMask` is `Some`.
    if filter_size == TX_4X4 || !mask.flat_mask.unwrap_or(true) {
        let out = narrow_filter(
            &NarrowFilterSamples {
                p1: samples.p1,
                p0: samples.p0,
                q0: samples.q0,
                q1: samples.q1,
            },
            mask.hev_mask,
            bit_depth,
        );
        let mut result = SampleFilterOutput::echo(samples);
        // §8.8.5.2 writes `op1`, `op0`, `oq0`, `oq1` (the latter two
        // unconditionally, `op1` / `oq1` equal the input when
        // `hevMask == 1` so the echo would carry the same value).
        result.p1 = out.op1;
        result.p0 = out.op0;
        result.q0 = out.oq0;
        result.q1 = out.oq1;
        return result;
    }

    // §8.8.5 lines 5681-5684 — wide filter. `log2Size = 3` when
    // `filterSize == TX_8X8` OR `flatMask2 == 0`; otherwise
    // `log2Size = 4`. The `filterSize == TX_8X8` arm short-circuits
    // before `flatMask2` (which is `None` exactly when `filterSize <
    // TX_16X16`) is read, so the `unwrap_or` default is only consulted
    // on the `filterSize == TX_16X16` path where `flatMask2` is `Some`.
    let log2_size: u32 = if filter_size == TX_8X8 || !mask.flat_mask2.unwrap_or(true) {
        3
    } else {
        4
    };

    let out = wide_filter(
        &WideFilterSamples {
            p7: samples.p7,
            p6: samples.p6,
            p5: samples.p5,
            p4: samples.p4,
            p3: samples.p3,
            p2: samples.p2,
            p1: samples.p1,
            p0: samples.p0,
            q0: samples.q0,
            q1: samples.q1,
            q2: samples.q2,
            q3: samples.q3,
            q4: samples.q4,
            q5: samples.q5,
            q6: samples.q6,
            q7: samples.q7,
        },
        log2_size,
        bit_depth,
    );

    // §8.8.5.3 returns all 14 inner fields; for `log2Size == 3` the
    // outer eight echo the input through, so writing every field is
    // safe regardless of `log2_size`. `p7` / `q7` are never mutated by
    // the wide filter (position `-8` / `+7` lies outside the write
    // window), so they keep the echoed stencil value.
    let mut result = SampleFilterOutput::echo(samples);
    result.p6 = out.op6;
    result.p5 = out.op5;
    result.p4 = out.op4;
    result.p3 = out.op3;
    result.p2 = out.op2;
    result.p1 = out.op1;
    result.p0 = out.op0;
    result.q0 = out.oq0;
    result.q1 = out.oq1;
    result.q2 = out.oq2;
    result.q3 = out.oq3;
    result.q4 = out.oq4;
    result.q5 = out.oq5;
    result.q6 = out.oq6;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_size::TX_16X16;

    /// Build a flat stencil at `v` — every sample equal. Used as a
    /// baseline: a flat boundary produces no change on any branch.
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

    /// Assert the whole stencil came through unchanged.
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

    /// §8.8.5 baseline — a flat stencil straddling the boundary passes
    /// the §8.8.5.1 `filterMask` (every nearby pair is within `limit`
    /// / `blimit`), but each filtering branch is the identity on a
    /// flat region. Verified at every `filterSize`.
    #[test]
    fn flat_stencil_unchanged_all_sizes() {
        let s = flat(128);
        for &fsize in &[TX_4X4, TX_8X8, TX_16X16] {
            let out = sample_filtering(&s, 9, 80, 4, fsize, 8);
            assert_echo(&s, &out);
        }
    }

    /// §8.8.5 line 5678 — `filterMask == 0` ⇒ no filter. A sharp jump
    /// between `p3` and `p2` trips the §8.8.5.1 `limit` test so
    /// `filterMask` is `false`; the whole stencil must echo through
    /// untouched regardless of `filterSize`.
    #[test]
    fn filter_mask_zero_echoes_stencil() {
        // p3 vs p2 differ by 200 >> limit = 1, so filterMask resets.
        let mut s = flat(128);
        s.p3 = 0;
        s.p2 = 200;
        for &fsize in &[TX_4X4, TX_8X8, TX_16X16] {
            let out = sample_filtering(&s, 1, 255, 4, fsize, 8);
            assert_echo(&s, &out);
        }
    }

    /// §8.8.5 line 5679 — narrow branch when `filterSize == TX_4X4`.
    /// A small step that passes `filterMask` but is filtered by the
    /// §8.8.5.2 narrow filter; only `p1`, `p0`, `q0`, `q1` may change.
    #[test]
    fn tx4x4_dispatches_narrow() {
        // Gentle ramp: passes limit/blimit, has a non-flat boundary.
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
        let out = sample_filtering(&s, 9, 80, 4, TX_4X4, 8);

        // Cross-check against the §8.8.5.2 primitive run directly.
        let mask = filter_mask(
            &FilterMaskSamples {
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
            },
            9,
            80,
            4,
            TX_4X4,
            8,
        );
        assert!(mask.filter_mask, "test stencil must pass filterMask");
        let nf = narrow_filter(
            &NarrowFilterSamples {
                p1: s.p1,
                p0: s.p0,
                q0: s.q0,
                q1: s.q1,
            },
            mask.hev_mask,
            8,
        );
        assert_eq!(out.p1, nf.op1);
        assert_eq!(out.p0, nf.op0);
        assert_eq!(out.q0, nf.oq0);
        assert_eq!(out.q1, nf.oq1);
        // Everything outside the 4-sample window is untouched.
        assert_eq!(out.p2, s.p2);
        assert_eq!(out.q2, s.q2);
        assert_eq!(out.p7, s.p7);
        assert_eq!(out.q7, s.q7);
    }

    /// §8.8.5 line 5679 — `flatMask == 0` forces the narrow branch
    /// even at `filterSize == TX_8X8`. A non-flat inner region
    /// (`p2` far from `p0`) trips `flatMask` while staying within
    /// `limit`, so the §8.8.5 dispatch must still pick the narrow
    /// filter (not wide).
    #[test]
    fn flat_mask_zero_forces_narrow_at_tx8x8() {
        // p2 differs from p0 by more than thresholdBd=1 (8-bit), so
        // flatMask resets; but every adjacent pair stays within
        // limit=9 so filterMask survives.
        let s = SampleFilterSamples {
            p7: 100,
            p6: 100,
            p5: 100,
            p4: 100,
            p3: 100,
            p2: 105,
            p1: 102,
            p0: 100,
            q0: 100,
            q1: 100,
            q2: 100,
            q3: 100,
            q4: 100,
            q5: 100,
            q6: 100,
            q7: 100,
        };
        let mask = filter_mask(
            &FilterMaskSamples {
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
            },
            9,
            80,
            4,
            TX_8X8,
            8,
        );
        assert!(mask.filter_mask);
        assert_eq!(mask.flat_mask, Some(false), "flatMask must reset");

        let out = sample_filtering(&s, 9, 80, 4, TX_8X8, 8);
        // Narrow path: p2 (outside the narrow 4-window) is untouched.
        assert_eq!(out.p2, s.p2);
        // Cross-check the boundary samples match the narrow primitive.
        let nf = narrow_filter(
            &NarrowFilterSamples {
                p1: s.p1,
                p0: s.p0,
                q0: s.q0,
                q1: s.q1,
            },
            mask.hev_mask,
            8,
        );
        assert_eq!(out.p0, nf.op0);
        assert_eq!(out.q0, nf.oq0);
    }

    /// §8.8.5 line 5681 — wide `log2Size == 3` when `filterSize ==
    /// TX_8X8` and the region is flat enough that `flatMask == 1`.
    /// The §8.8.5.3 8-tap kernel mutates `p2..p0`, `q0..q2`; positions
    /// `p3` / `q3` and the outer ring are untouched.
    #[test]
    fn tx8x8_flat_dispatches_wide_log2_3() {
        // Flat-enough inner region (flatMask passes) with a small step
        // at the boundary so the kernel actually moves samples.
        let s = SampleFilterSamples {
            p7: 100,
            p6: 100,
            p5: 100,
            p4: 100,
            p3: 100,
            p2: 100,
            p1: 100,
            p0: 100,
            q0: 101,
            q1: 101,
            q2: 101,
            q3: 101,
            q4: 101,
            q5: 101,
            q6: 101,
            q7: 101,
        };
        let mask = filter_mask(
            &FilterMaskSamples {
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
            },
            9,
            80,
            4,
            TX_8X8,
            8,
        );
        assert!(mask.filter_mask);
        assert_eq!(mask.flat_mask, Some(true), "region must be flat");

        let out = sample_filtering(&s, 9, 80, 4, TX_8X8, 8);

        let wf = wide_filter(
            &WideFilterSamples {
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
            },
            3,
            8,
        );
        assert_eq!(out.p2, wf.op2);
        assert_eq!(out.p1, wf.op1);
        assert_eq!(out.p0, wf.op0);
        assert_eq!(out.q0, wf.oq0);
        assert_eq!(out.q1, wf.oq1);
        assert_eq!(out.q2, wf.oq2);
        // p3 / q3 and the outer ring stay at the input (log2Size == 3
        // echoes them through).
        assert_eq!(out.p3, s.p3);
        assert_eq!(out.q3, s.q3);
        assert_eq!(out.p7, s.p7);
        assert_eq!(out.q7, s.q7);
    }

    /// §8.8.5 lines 5683-5684 — wide `log2Size == 4` when `filterSize
    /// == TX_16X16` and both `flatMask` and `flatMask2` are `1`. The
    /// §8.8.5.3 16-tap kernel mutates `p6..p0`, `q0..q6`; only `p7` /
    /// `q7` are echoed through.
    #[test]
    fn tx16x16_fully_flat_dispatches_wide_log2_4() {
        // Fully flat region with a 1-step boundary: both flat masks
        // pass and the dispatch reaches the `else` (log2Size == 4)
        // arm.
        let s = SampleFilterSamples {
            p7: 100,
            p6: 100,
            p5: 100,
            p4: 100,
            p3: 100,
            p2: 100,
            p1: 100,
            p0: 100,
            q0: 101,
            q1: 101,
            q2: 101,
            q3: 101,
            q4: 101,
            q5: 101,
            q6: 101,
            q7: 101,
        };
        let mask = filter_mask(
            &FilterMaskSamples {
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
            },
            9,
            80,
            4,
            TX_16X16,
            8,
        );
        assert!(mask.filter_mask);
        assert_eq!(mask.flat_mask, Some(true));
        assert_eq!(mask.flat_mask2, Some(true), "outer ring must be flat");

        let out = sample_filtering(&s, 9, 80, 4, TX_16X16, 8);

        let wf = wide_filter(
            &WideFilterSamples {
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
            },
            4,
            8,
        );
        assert_eq!(out.p6, wf.op6);
        assert_eq!(out.p0, wf.op0);
        assert_eq!(out.q0, wf.oq0);
        assert_eq!(out.q6, wf.oq6);
        // Only p7 / q7 are echoed through (positions -8 / +7).
        assert_eq!(out.p7, s.p7);
        assert_eq!(out.q7, s.q7);
    }

    /// §8.8.5 line 5682 — `flatMask2 == 0` at `filterSize == TX_16X16`
    /// drops the dispatch back to the wide `log2Size == 3` branch.
    /// A flat inner four samples (`flatMask == 1`) but a non-flat
    /// outer ring (`p7` far from `p0`) trips `flatMask2`, so the
    /// §8.8.5 dispatch must pick `log2Size = 3`, not `4`.
    #[test]
    fn flat_mask2_zero_drops_to_wide_log2_3() {
        let s = SampleFilterSamples {
            // outer ring far from p0 → flatMask2 resets.
            p7: 130,
            p6: 100,
            p5: 100,
            p4: 100,
            // inner four flat → flatMask passes.
            p3: 100,
            p2: 100,
            p1: 100,
            p0: 100,
            q0: 101,
            q1: 101,
            q2: 101,
            q3: 101,
            q4: 101,
            q5: 101,
            q6: 101,
            q7: 101,
        };
        let mask = filter_mask(
            &FilterMaskSamples {
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
            },
            9,
            80,
            4,
            TX_16X16,
            8,
        );
        // filterMask uses only the inner pairs (p3..q3), all flat-ish,
        // so it survives; flatMask survives (inner four flat) but
        // flatMask2 resets (p7 outlier).
        assert!(mask.filter_mask);
        assert_eq!(mask.flat_mask, Some(true));
        assert_eq!(mask.flat_mask2, Some(false));

        let out = sample_filtering(&s, 9, 80, 4, TX_16X16, 8);
        // Must equal the log2Size == 3 wide filter, NOT log2Size == 4.
        let wf3 = wide_filter(
            &WideFilterSamples {
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
            },
            3,
            8,
        );
        assert_eq!(out.p2, wf3.op2);
        assert_eq!(out.p0, wf3.op0);
        assert_eq!(out.q2, wf3.oq2);
        // log2Size == 3 leaves p3 / q3 and the outer ring untouched.
        assert_eq!(out.p3, s.p3);
        assert_eq!(out.q3, s.q3);
        assert_eq!(out.p6, s.p6);
    }

    /// §8.8.5 BitDepth propagation — the dispatch and every
    /// sub-process scale by `BitDepth`. A flat 10-bit stencil at the
    /// midpoint (512) is the identity on every branch.
    #[test]
    fn flat_stencil_unchanged_10bit() {
        let s = flat(512);
        for &fsize in &[TX_4X4, TX_8X8, TX_16X16] {
            // limit/blimit/thresh are pre-BitDepth-scale (the
            // sub-processes apply `<< (BitDepth - 8)`).
            let out = sample_filtering(&s, 9, 80, 4, fsize, 10);
            assert_echo(&s, &out);
        }
    }
}
