//! VP9 §8.8.5.1 `filter mask process` — per spec v0.7.
//!
//! This module lands the per-edge [`filter_mask`] derivation as a pure
//! leaf primitive. The §8.8.5 [`sample_filtering`]-of-edge driver
//! invokes it first — before dispatching to the §8.8.5.2 narrow filter
//! or the §8.8.5.3 wide filter — to read out the four boolean
//! conditions that decide which (if any) filter runs and how wide.
//!
//! The §8.8.5.1 listing (`vp9-spec.txt` lines 5685-5792) defines four
//! outputs from a 16-sample stencil straddling the boundary:
//!
//! * `hevMask` — high-edge-variance indicator. Computed against the
//!   four nearest samples (`p1`, `p0`, `q0`, `q1`) and the
//!   bit-depth-scaled `thresh` (lines 5730-5734).
//! * `filterMask` — top-level decision: are the seven nearest pairs
//!   within `limit` / `blimit`? (lines 5737-5750). When `0` no filter
//!   runs at this edge.
//! * `flatMask` — only used when `filterSize >= TX_8X8`. Six pair
//!   diffs over the inner four samples on each side; sets to `1` when
//!   the region is flat against the one-LSB-per-bit-depth-step
//!   `thresholdBd` (lines 5753-5774).
//! * `flatMask2` — only used when `filterSize >= TX_16X16`. Eight
//!   pair diffs over the outer four samples on each side relative to
//!   the nearest sample (lines 5777-5792).
//!
//! All comparisons use the bit-depth-scaled forms
//! `limit << (BitDepth - 8)` / `blimit << (BitDepth - 8)` / `thresh <<
//! (BitDepth - 8)` / `1 << (BitDepth - 8)`; in 8-bit operation the
//! shift is zero and the primitives degenerate to plain comparisons.
//!
//! ## Scope of this round
//!
//! Round 253 lands the §8.8.5.1 leaf only — pure-state function over
//! a fixed 16-sample stencil [`FilterMaskSamples`] (`p7`..`p0` /
//! `q0`..`q7`). The caller is responsible for fetching the stencil
//! from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per §8.8.5.1
//! lines 5703-5727 — this primitive does not walk
//! `(plane, x, y, dx, dy)` itself.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.5 `sample_filtering( )` — the per-edge outer driver that
//!   reads the stencil from `CurrFrame` and dispatches to narrow /
//!   wide filters.
//! * §8.8.5.2 `filter4` / §8.8.5.3 `filter6` / `filter8` /
//!   `filter16` — the actual sample-mutating filter primitives that
//!   read this round's `FilterMask` to decide which path runs.
//! * §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
//!   that calls §8.8.3 + §8.8.4 + §8.8.5 for each `(loopRow,
//!   loopCol)` step.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.5.1 lines 5685-5792). `Abs`
//! is the §3 absolute-value primitive.

use crate::filter_size::{TX_16X16, TX_8X8};

/// §8.8.5.1 input — the 16-sample stencil straddling the boundary.
///
/// Per `vp9-spec.txt` lines 5703-5727, the §8.8.5 outer driver reads
/// these from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]`:
///
/// * `q0 = CurrFrame[ plane ][ y ][ x ]`
/// * `q[k] = CurrFrame[ plane ][ y + dy*k ][ x + dx*k ]` for `k =
///   1..=7`
/// * `p[k] = CurrFrame[ plane ][ y - dy*k ][ x - dx*k ]` for `k =
///   1..=8` (so `p0 = CurrFrame[ plane ][ y - dy ][ x - dx ]`,
///   `p7 = CurrFrame[ plane ][ y - dy*8 ][ x - dx*8 ]`).
///
/// `q4..q7` and `p4..p7` are only consulted when `filterSize ==
/// TX_16X16` (the §8.8.5.1 `flatMask2` step at lines 5777-5792).
/// Per `vp9-spec.txt` line 5729: "Samples q4, q5, q6, q7, p4, p5, p6
/// and p7 are only used if filterSize is equal to TX_16X16."
///
/// Samples are carried as `i32` so the abs-difference subtractions
/// in the listing don't underflow for 10-bit / 12-bit pixels (whose
/// values can reach `(1 << 12) - 1 = 4095`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterMaskSamples {
    /// `p7` — the outermost sample on the `p` side (only read for
    /// `flatMask2`, i.e. `filterSize == TX_16X16`).
    pub p7: i32,
    /// `p6` — only read for `flatMask2`.
    pub p6: i32,
    /// `p5` — only read for `flatMask2`.
    pub p5: i32,
    /// `p4` — only read for `flatMask2`.
    pub p4: i32,
    /// `p3` — read by `filterMask` and `flatMask`.
    pub p3: i32,
    /// `p2` — read by `filterMask` and `flatMask`.
    pub p2: i32,
    /// `p1` — read by `hevMask`, `filterMask`, and `flatMask`.
    pub p1: i32,
    /// `p0` — the boundary sample on the `p` side; read by every
    /// output.
    pub p0: i32,
    /// `q0` — the boundary sample on the `q` side; read by every
    /// output.
    pub q0: i32,
    /// `q1` — read by `hevMask`, `filterMask`, and `flatMask`.
    pub q1: i32,
    /// `q2` — read by `filterMask` and `flatMask`.
    pub q2: i32,
    /// `q3` — read by `filterMask` and `flatMask`.
    pub q3: i32,
    /// `q4` — only read for `flatMask2`.
    pub q4: i32,
    /// `q5` — only read for `flatMask2`.
    pub q5: i32,
    /// `q6` — only read for `flatMask2`.
    pub q6: i32,
    /// `q7` — only read for `flatMask2`.
    pub q7: i32,
}

/// §8.8.5.1 output — the four boolean masks the §8.8.5 driver uses
/// to dispatch the per-edge filter.
///
/// * `hev_mask` — `hevMask` from §8.8.5.1 lines 5730-5734. When
///   `1`, the §8.8.5.2 narrow filter uses the high-edge-variance
///   branch; when `0`, it uses the smooth branch.
/// * `filter_mask` — `filterMask` from §8.8.5.1 lines 5737-5750.
///   When `0`, the §8.8.5 driver runs no filter at this edge.
/// * `flat_mask` — `flatMask` from §8.8.5.1 lines 5753-5774. Only
///   meaningful when `filterSize >= TX_8X8`; the caller passes
///   `filter_size` to [`filter_mask`] and the primitive returns
///   `None` for this field when it would be unread.
/// * `flat_mask2` — `flatMask2` from §8.8.5.1 lines 5777-5792. Only
///   meaningful when `filterSize >= TX_16X16`; same `Option`
///   convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterMask {
    /// `hevMask` from §8.8.5.1 line 5734.
    pub hev_mask: bool,
    /// `filterMask` from §8.8.5.1 line 5750.
    pub filter_mask: bool,
    /// `flatMask` from §8.8.5.1 line 5773 — `None` when `filterSize
    /// < TX_8X8` (i.e. the §8.8.5.1 lead paragraph at line 5697
    /// reads "only used if filterSize >= TX_8X8").
    pub flat_mask: Option<bool>,
    /// `flatMask2` from §8.8.5.1 line 5791 — `None` when `filterSize
    /// < TX_16X16` (line 5698 reads "only used if filterSize >=
    /// TX_16X16").
    pub flat_mask2: Option<bool>,
}

/// `Abs( a - b )` per §3 — absolute difference of two `i32` samples.
/// Per §8.8.5.1 lines 5733-5749 every difference computed by the
/// primitive flows through `Abs(.)`.
#[inline]
fn abs_diff(a: i32, b: i32) -> i32 {
    (a - b).abs()
}

/// Run §8.8.5.1 `filter mask process` for one edge per
/// `vp9-spec.txt` lines 5685-5792.
///
/// Returns the four §8.8.5 dispatch booleans `(hevMask, filterMask,
/// flatMask, flatMask2)` packaged in [`FilterMask`]. `flatMask` /
/// `flatMask2` are returned as `Some` only when the matching
/// `filterSize >=` precondition from the §8.8.5.1 lead paragraph
/// (lines 5697-5698) is met; otherwise they're `None`.
///
/// # Inputs
///
/// * `samples` — the 16-sample stencil [`FilterMaskSamples`] the
///   §8.8.5 outer driver assembled by reading `CurrFrame[ plane ][
///   y +/- dy*k ][ x +/- dx*k ]` per §8.8.5.1 lines 5703-5727.
/// * `limit` / `blimit` / `thresh` — the §8.8.4 [`crate::
///   FilterStrength`] tuple this primitive reads. All are `u8` per
///   §8.8.4 (lines 5648-5661).
/// * `filter_size` — the §8.8.3 [`crate::filter_size`] output: one
///   of `TX_4X4` / `TX_8X8` / `TX_16X16` (the §8.8.3 step caps it
///   below `TX_32X32`).
/// * `bit_depth` — `BitDepth` per §6.2.2 (8, 10, or 12). Drives the
///   four `... << (BitDepth - 8)` scalings in the §8.8.5.1 listing.
///
/// # Listing
///
/// `vp9-spec.txt` §8.8.5.1 lines 5730-5792:
///
/// ```text
/// hevMask = 0
/// threshBd = thresh << (BitDepth - 8)
/// hevMask |= (Abs( p1 - p0 ) > threshBd)
/// hevMask |= (Abs( q1 - q0 ) > threshBd)
///
/// limitBd  = limit  << (BitDepth - 8)
/// blimitBd = blimit << (BitDepth - 8)
/// mask = 0
/// mask |= (Abs( p3 - p2 ) > limitBd)
/// mask |= (Abs( p2 - p1 ) > limitBd)
/// mask |= (Abs( p1 - p0 ) > limitBd)
/// mask |= (Abs( q1 - q0 ) > limitBd)
/// mask |= (Abs( q2 - q1 ) > limitBd)
/// mask |= (Abs( q3 - q2 ) > limitBd)
/// mask |= (Abs( p0 - q0 ) * 2 + Abs( p1 - q1 ) / 2 > blimitBd)
/// filterMask = (mask == 0)
///
/// thresholdBd = 1 << (BitDepth - 8)
/// if (filterSize >= TX_8X8) {
///     mask = 0
///     mask |= (Abs( p1 - p0 ) > thresholdBd)
///     mask |= (Abs( q1 - q0 ) > thresholdBd)
///     mask |= (Abs( p2 - p0 ) > thresholdBd)
///     mask |= (Abs( q2 - q0 ) > thresholdBd)
///     mask |= (Abs( p3 - p0 ) > thresholdBd)
///     mask |= (Abs( q3 - q0 ) > thresholdBd)
///     flatMask = (mask == 0)
/// }
///
/// if (filterSize >= TX_16X16) {
///     mask = 0
///     mask |= (Abs( p7 - p0 ) > thresholdBd)
///     mask |= (Abs( q7 - q0 ) > thresholdBd)
///     mask |= (Abs( p6 - p0 ) > thresholdBd)
///     mask |= (Abs( q6 - q0 ) > thresholdBd)
///     mask |= (Abs( p5 - p0 ) > thresholdBd)
///     mask |= (Abs( q5 - q0 ) > thresholdBd)
///     mask |= (Abs( p4 - p0 ) > thresholdBd)
///     mask |= (Abs( q4 - q0 ) > thresholdBd)
///     flatMask2 = (mask == 0)
/// }
/// ```
pub fn filter_mask(
    samples: &FilterMaskSamples,
    limit: u8,
    blimit: u8,
    thresh: u8,
    filter_size: u8,
    bit_depth: u8,
) -> FilterMask {
    // §8.8.5.1: BitDepth - 8 ∈ {0, 2, 4} per §6.2.2 BitDepth ∈ {8,
    // 10, 12}.
    let shift = (bit_depth - 8) as u32;

    // §8.8.5.1 hevMask (lines 5730-5734).
    let thresh_bd = (thresh as i32) << shift;
    let hev_mask = abs_diff(samples.p1, samples.p0) > thresh_bd
        || abs_diff(samples.q1, samples.q0) > thresh_bd;

    // §8.8.5.1 filterMask (lines 5737-5750).
    let limit_bd = (limit as i32) << shift;
    let blimit_bd = (blimit as i32) << shift;
    let filter_mask = !(abs_diff(samples.p3, samples.p2) > limit_bd
        || abs_diff(samples.p2, samples.p1) > limit_bd
        || abs_diff(samples.p1, samples.p0) > limit_bd
        || abs_diff(samples.q1, samples.q0) > limit_bd
        || abs_diff(samples.q2, samples.q1) > limit_bd
        || abs_diff(samples.q3, samples.q2) > limit_bd
        || abs_diff(samples.p0, samples.q0) * 2 + abs_diff(samples.p1, samples.q1) / 2 > blimit_bd);

    // §8.8.5.1 flatMask (lines 5753-5774) — only when filterSize >=
    // TX_8X8.
    let threshold_bd = 1i32 << shift;
    let flat_mask = if filter_size >= TX_8X8 {
        Some(
            !(abs_diff(samples.p1, samples.p0) > threshold_bd
                || abs_diff(samples.q1, samples.q0) > threshold_bd
                || abs_diff(samples.p2, samples.p0) > threshold_bd
                || abs_diff(samples.q2, samples.q0) > threshold_bd
                || abs_diff(samples.p3, samples.p0) > threshold_bd
                || abs_diff(samples.q3, samples.q0) > threshold_bd),
        )
    } else {
        None
    };

    // §8.8.5.1 flatMask2 (lines 5777-5792) — only when filterSize >=
    // TX_16X16.
    let flat_mask2 = if filter_size >= TX_16X16 {
        Some(
            !(abs_diff(samples.p7, samples.p0) > threshold_bd
                || abs_diff(samples.q7, samples.q0) > threshold_bd
                || abs_diff(samples.p6, samples.p0) > threshold_bd
                || abs_diff(samples.q6, samples.q0) > threshold_bd
                || abs_diff(samples.p5, samples.p0) > threshold_bd
                || abs_diff(samples.q5, samples.q0) > threshold_bd
                || abs_diff(samples.p4, samples.p0) > threshold_bd
                || abs_diff(samples.q4, samples.q0) > threshold_bd),
        )
    } else {
        None
    };

    FilterMask {
        hev_mask,
        filter_mask,
        flat_mask,
        flat_mask2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_size::TX_4X4;

    /// Build a stencil with every sample set to `v`. A flat stencil
    /// satisfies every §8.8.5.1 abs-diff condition (all diffs are 0),
    /// so `filterMask = flatMask = flatMask2 = 1` and `hevMask = 0`.
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

    /// §8.8.5.1 baseline on a perfectly flat stencil at `BitDepth =
    /// 8`. Every `Abs(...)` is `0`, so every `> ...Bd` test is false
    /// and the masks turn out:
    ///
    /// * `hevMask = 0` (the OR of two false comparisons).
    /// * `filterMask = (mask == 0) = 1`.
    /// * `flatMask = 1`, `flatMask2 = 1` (both unmask when their
    ///   filterSize precondition is met).
    #[test]
    fn flat_stencil_passes_every_mask_8bit_tx16x16() {
        let out = filter_mask(&flat(128), 16, 52, 1, TX_16X16, 8);
        assert!(!out.hev_mask, "hevMask should be 0 for flat stencil");
        assert!(out.filter_mask, "filterMask should be 1 for flat stencil");
        assert_eq!(
            out.flat_mask,
            Some(true),
            "flatMask should be 1 for flat stencil at TX_16X16"
        );
        assert_eq!(
            out.flat_mask2,
            Some(true),
            "flatMask2 should be 1 for flat stencil at TX_16X16"
        );
    }

    /// §8.8.5.1 lead paragraph at line 5697-5698 — `flatMask` is
    /// `None` when `filterSize == TX_4X4`, `flatMask2` is `None`
    /// when `filterSize < TX_16X16`.
    #[test]
    fn filter_size_tx4x4_returns_none_for_flat_masks() {
        let out = filter_mask(&flat(128), 16, 52, 1, TX_4X4, 8);
        assert_eq!(out.flat_mask, None, "TX_4X4 → flatMask unset");
        assert_eq!(out.flat_mask2, None, "TX_4X4 → flatMask2 unset");
    }

    /// §8.8.5.1 lead paragraph at line 5697-5698 — `flatMask` is
    /// `Some` at `TX_8X8` but `flatMask2` is still `None` because
    /// `filterSize < TX_16X16`.
    #[test]
    fn filter_size_tx8x8_returns_flat_mask_only() {
        let out = filter_mask(&flat(128), 16, 52, 1, TX_8X8, 8);
        assert!(
            out.flat_mask.is_some(),
            "TX_8X8 → flatMask Some (filterSize >= TX_8X8)"
        );
        assert_eq!(
            out.flat_mask2, None,
            "TX_8X8 → flatMask2 None (filterSize < TX_16X16)"
        );
    }

    /// §8.8.5.1 line 5734 — `hevMask` flips to `1` when `Abs(p1 -
    /// p0) > threshBd` even if every other condition is fine. Build
    /// a stencil where `p1 - p0 = thresh + 1` to crest the threshold.
    #[test]
    fn hev_mask_triggers_on_p1_minus_p0_above_thresh() {
        let mut s = flat(128);
        // thresh = 4 at 8-bit → threshBd = 4. Set p1 = p0 + 5.
        s.p1 = 128 + 5;
        let out = filter_mask(&s, 16, 52, 4, TX_4X4, 8);
        assert!(out.hev_mask, "hevMask triggers when |p1 - p0| > thresh");
    }

    /// §8.8.5.1 line 5734 — `hevMask` also triggers via the `q1 -
    /// q0` term.
    #[test]
    fn hev_mask_triggers_on_q1_minus_q0_above_thresh() {
        let mut s = flat(128);
        s.q1 = 128 + 5;
        let out = filter_mask(&s, 16, 52, 4, TX_4X4, 8);
        assert!(out.hev_mask, "hevMask triggers when |q1 - q0| > thresh");
    }

    /// §8.8.5.1 line 5734 — equality should not trigger (`> threshBd`
    /// is strict).
    #[test]
    fn hev_mask_does_not_trigger_at_exact_threshold() {
        let mut s = flat(128);
        // thresh = 4 at 8-bit; |p1 - p0| = 4 should NOT trigger
        // because the test is strict `>`.
        s.p1 = 128 + 4;
        let out = filter_mask(&s, 16, 52, 4, TX_4X4, 8);
        assert!(
            !out.hev_mask,
            "hevMask uses strict `>`; equality stays at 0"
        );
    }

    /// §8.8.5.1 line 5743 — `filterMask` resets to `0` when `Abs(p3
    /// - p2) > limitBd`.
    #[test]
    fn filter_mask_resets_on_outer_p_pair() {
        let mut s = flat(128);
        // limit = 4 at 8-bit. Set p3 - p2 = 5 to crest the threshold.
        s.p3 = 128 + 5;
        let out = filter_mask(&s, 4, 52, 1, TX_4X4, 8);
        assert!(!out.filter_mask, "filterMask resets when |p3 - p2| > limit");
    }

    /// §8.8.5.1 line 5748 — `filterMask` resets on the outer `q3 -
    /// q2` pair too.
    #[test]
    fn filter_mask_resets_on_outer_q_pair() {
        let mut s = flat(128);
        s.q3 = 128 + 5;
        let out = filter_mask(&s, 4, 52, 1, TX_4X4, 8);
        assert!(!out.filter_mask, "filterMask resets when |q3 - q2| > limit");
    }

    /// §8.8.5.1 line 5749 — `filterMask` resets via the `Abs(p0 -
    /// q0) * 2 + Abs(p1 - q1) / 2 > blimitBd` boundary condition.
    /// Set `p0 = 0`, `q0 = 50`, then `Abs(p0 - q0) * 2 = 100`. With
    /// `blimit = 50` the comparison `100 > 50` triggers the reset.
    /// Keep limit = 100 so the per-pair tests don't trigger first
    /// (`Abs(p1 - p0) = 0`, `Abs(q1 - q0) = 50`, both `<= 100`).
    #[test]
    fn filter_mask_resets_on_boundary_term() {
        let s = FilterMaskSamples {
            p7: 0,
            p6: 0,
            p5: 0,
            p4: 0,
            p3: 0,
            p2: 0,
            p1: 0,
            p0: 0,
            q0: 50,
            q1: 50,
            q2: 50,
            q3: 50,
            q4: 50,
            q5: 50,
            q6: 50,
            q7: 50,
        };
        let out = filter_mask(&s, 100, 50, 1, TX_4X4, 8);
        assert!(
            !out.filter_mask,
            "filterMask resets via the |p0 - q0|*2 + |p1 - q1|/2 term"
        );
    }

    /// §8.8.5.1 line 5759 — `flatMask` resets when `Abs(p2 - p0) >
    /// thresholdBd`. At 8-bit `thresholdBd = 1`, so a diff of 2 is
    /// enough.
    #[test]
    fn flat_mask_resets_on_p2_minus_p0() {
        let mut s = flat(128);
        // p2 - p0 = 2 > thresholdBd = 1 at 8-bit.
        s.p2 = 130;
        let out = filter_mask(&s, 100, 200, 1, TX_8X8, 8);
        assert_eq!(
            out.flat_mask,
            Some(false),
            "flatMask resets when |p2 - p0| > thresholdBd"
        );
    }

    /// §8.8.5.1 line 5783 — `flatMask2` resets when `Abs(p7 - p0) >
    /// thresholdBd`. Inner samples stay flat so `flatMask` survives;
    /// the outer p7 / q7 ring is what `flatMask2` polices.
    #[test]
    fn flat_mask2_resets_on_p7_minus_p0_with_flat_mask_surviving() {
        let mut s = flat(128);
        s.p7 = 130; // 2 > 1
        let out = filter_mask(&s, 100, 200, 1, TX_16X16, 8);
        assert_eq!(
            out.flat_mask,
            Some(true),
            "flatMask survives — inner stencil still flat"
        );
        assert_eq!(
            out.flat_mask2,
            Some(false),
            "flatMask2 resets when |p7 - p0| > thresholdBd"
        );
    }

    /// §8.8.5.1 BitDepth handling — at `BitDepth = 10`, `thresh = 4`
    /// scales to `threshBd = 16`. A `p1 - p0` diff of 16 must NOT
    /// trigger (strict `>`), but `17` must.
    #[test]
    fn bit_depth_10_scales_thresh_by_4x() {
        // |p1 - p0| = 16 → not > 16 → hev_mask stays 0.
        let mut s = flat(512);
        s.p1 = 528;
        let out = filter_mask(&s, 16, 52, 4, TX_4X4, 10);
        assert!(
            !out.hev_mask,
            "10-bit threshBd = 4 << 2 = 16; equality stays 0"
        );

        // |p1 - p0| = 17 → > 16 → hev_mask flips to 1.
        let mut s = flat(512);
        s.p1 = 529;
        let out = filter_mask(&s, 16, 52, 4, TX_4X4, 10);
        assert!(out.hev_mask, "10-bit |p1 - p0| = 17 > 16 → hevMask = 1");
    }

    /// §8.8.5.1 BitDepth handling — at `BitDepth = 12`, `thresholdBd
    /// = 1 << 4 = 16`. `flatMask` only resets when a difference
    /// exceeds 16.
    #[test]
    fn bit_depth_12_scales_threshold_bd_to_16() {
        let mut s = flat(2048);
        s.p2 = 2048 + 16; // exactly 16, not > 16
        let out = filter_mask(&s, 100, 200, 1, TX_8X8, 12);
        assert_eq!(
            out.flat_mask,
            Some(true),
            "12-bit |p2 - p0| = 16 not > thresholdBd = 16; flatMask stays 1"
        );

        let mut s = flat(2048);
        s.p2 = 2048 + 17;
        let out = filter_mask(&s, 100, 200, 1, TX_8X8, 12);
        assert_eq!(
            out.flat_mask,
            Some(false),
            "12-bit |p2 - p0| = 17 > 16; flatMask resets"
        );
    }

    /// §8.8.5.1 lines 5749 — the boundary term mixes `Abs(p0 - q0)
    /// * 2` and `Abs(p1 - q1) / 2`. Verify the `/ 2` floor: with
    /// `Abs(p1 - q1) = 3`, the integer division floors to `1`. So
    /// `Abs(p0 - q0) * 2 + 1 > blimit` is the actual test.
    #[test]
    fn filter_mask_boundary_term_uses_integer_division() {
        // p0 - q0 = 5 → 5 * 2 = 10
        // p1 - q1 = 3 → 3 / 2 = 1
        // Sum = 11. With blimit = 10, 11 > 10 → reset.
        let mut s = flat(0);
        s.p0 = 0;
        s.q0 = 5;
        s.p1 = 0;
        s.q1 = 3;
        let out = filter_mask(&s, 100, 10, 1, TX_4X4, 8);
        assert!(
            !out.filter_mask,
            "boundary term 10 + 1 = 11 > blimit 10 → reset"
        );

        // Same but blimit = 11: 11 > 11 is false → survives.
        let mut s = flat(0);
        s.p0 = 0;
        s.q0 = 5;
        s.p1 = 0;
        s.q1 = 3;
        let out = filter_mask(&s, 100, 11, 1, TX_4X4, 8);
        assert!(
            out.filter_mask,
            "boundary term 11 > 11 false; filterMask survives"
        );
    }

    /// Symmetric stencil with rising slope: `p7..p0` ascending,
    /// `q0..q7` mirroring downward keeps the boundary region flat
    /// but breaks `flatMask2` because the outer p7 / q7 samples
    /// are far from p0 / q0.
    #[test]
    fn rising_slope_breaks_flat_mask2_but_not_flat_mask() {
        let s = FilterMaskSamples {
            p7: 100,
            p6: 110,
            p5: 120,
            p4: 125,
            p3: 128, // inner four match p0 exactly
            p2: 128,
            p1: 128,
            p0: 128,
            q0: 128,
            q1: 128,
            q2: 128,
            q3: 128,
            q4: 125,
            q5: 120,
            q6: 110,
            q7: 100,
        };
        let out = filter_mask(&s, 100, 200, 1, TX_16X16, 8);
        assert!(out.filter_mask, "filterMask survives flat inner region");
        assert_eq!(
            out.flat_mask,
            Some(true),
            "flatMask survives — inner four on each side match p0/q0"
        );
        assert_eq!(
            out.flat_mask2,
            Some(false),
            "flatMask2 resets — outer p7/q7 diff = 28 > thresholdBd = 1"
        );
    }
}
