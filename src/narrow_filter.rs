//! VP9 §8.8.5.2 `narrow filter process` — per spec v0.7.
//!
//! This module lands the per-edge [`narrow_filter`] sample-mutation
//! as a pure leaf primitive. The §8.8.5 outer driver invokes it after
//! the §8.8.5.1 [`crate::filter_mask`] step when `filterMask == 1` and
//! the §8.8.5.3 wide-filter preconditions (`flatMask` /
//! `flatMask2` plus `filterSize >= TX_8X8`) are not satisfied. The
//! primitive modifies the four nearest samples on each side of the
//! edge (`p1`, `p0`, `q0`, `q1`) and dispatches between two arithmetic
//! branches off the §8.8.5.1 `hevMask` output:
//!
//! * `hevMask == 1` (high edge variance) — modifies only `p0` and
//!   `q0`, using a filter constructed from all four input samples per
//!   `vp9-spec.txt` §8.8.5.2 lines 5809-5811.
//! * `hevMask == 0` (low / smooth edge) — modifies all four samples,
//!   using a filter constructed from just the two inner samples
//!   (`p0` and `q0`) plus an additional half-strength pass into `p1`
//!   and `q1` per lines 5806-5808 and 5846-5852.
//!
//! All arithmetic is performed on samples that have been offset by
//! `-0x80 << (BitDepth - 8)` so the working values land in the signed
//! range `[-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1]` per
//! lines 5814-5816. The `filter4_clamp` helper (lines 5824-5826)
//! enforces this clip with `Clip3` per §3.
//!
//! ## Scope of this round
//!
//! Round 255 lands the §8.8.5.2 leaf only — pure-state function over
//! a fixed 4-sample stencil [`NarrowFilterSamples`] (`p1`, `p0`, `q0`,
//! `q1`). The caller is responsible for:
//!
//! * Reading the stencil from `CurrFrame[ plane ][ y +/- dy*k ][ x
//!   +/- dx*k ]` per §8.8.5.2 lines 5830-5833 (the same `(dx, dy)`
//!   axis the §8.8.5.1 stencil-build uses).
//! * Checking `hevMask` against the §8.8.5.1 output before deciding
//!   the branch.
//! * Writing the four output samples back to `CurrFrame` at the
//!   matching `(y +/- dy*k, x +/- dx*k)` locations per lines 5844-5851.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.5 `sample_filtering( )` — the per-edge outer driver that
//!   reads the stencil from `CurrFrame`, runs §8.8.5.1, dispatches to
//!   §8.8.5.2 (this round) or §8.8.5.3, and writes the result back.
//! * §8.8.5.3 `wide_filter` — the `log2Size`-tap low-pass primitive
//!   the driver invokes when `flatMask` / `flatMask2` are set.
//! * §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
//!   that calls §8.8.3 + §8.8.4 + §8.8.5 for each `(loopRow,
//!   loopCol)` step.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.5.2 lines 5795-5853). `Clip3`
//! is the §3 clipping primitive; `Round2( x, n )` is §3 with
//! `(x + (1 << (n - 1))) >> n`.

/// §8.8.5.2 input — the 4-sample stencil straddling the boundary.
///
/// Per `vp9-spec.txt` lines 5830-5833, the §8.8.5 outer driver reads
/// these from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]`:
///
/// * `q0 = CurrFrame[ plane ][ y ][ x ]`
/// * `q1 = CurrFrame[ plane ][ y + dy ][ x + dx ]`
/// * `p0 = CurrFrame[ plane ][ y - dy ][ x - dx ]`
/// * `p1 = CurrFrame[ plane ][ y - dy*2 ][ x - dx*2 ]`
///
/// Samples are carried as `i32` so the `ps1 - qs1` / `qs0 - ps0`
/// subtractions in the listing don't underflow for 10-bit / 12-bit
/// pixels (whose values can reach `(1 << 12) - 1 = 4095`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NarrowFilterSamples {
    /// `p1` — second sample on the `p` side.
    pub p1: i32,
    /// `p0` — the boundary sample on the `p` side.
    pub p0: i32,
    /// `q0` — the boundary sample on the `q` side.
    pub q0: i32,
    /// `q1` — second sample on the `q` side.
    pub q1: i32,
}

/// §8.8.5.2 output — the four mutated samples.
///
/// Per `vp9-spec.txt` lines 5844-5851:
///
/// * `op0` is always written back to `CurrFrame[ plane ][ y - dy ][ x
///   - dx ]`.
/// * `oq0` is always written back to `CurrFrame[ plane ][ y ][ x ]`.
/// * `op1` / `oq1` are written back only when `hevMask == 0` (the
///   `!hevMask` block at lines 5846-5852); when `hevMask == 1` they
///   are returned equal to the input `p1` / `q1` (i.e. unchanged) so
///   the caller can write them back unconditionally without branching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NarrowFilterOutput {
    /// `op1` — replacement for the input `p1`. Unchanged when
    /// `hevMask == 1`.
    pub op1: i32,
    /// `op0` — replacement for the input `p0`. Always written.
    pub op0: i32,
    /// `oq0` — replacement for the input `q0`. Always written.
    pub oq0: i32,
    /// `oq1` — replacement for the input `q1`. Unchanged when
    /// `hevMask == 1`.
    pub oq1: i32,
}

/// §3 `Clip3( a, b, x ) = a if x < a; b if x > b; x otherwise.`
#[inline]
fn clip3(a: i32, b: i32, x: i32) -> i32 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}

/// §3 `Round2( x, n ) = (x + (1 << (n - 1))) >> n` — symmetric
/// half-up rounding for signed `i32`. Required by §8.8.5.2 line 5847.
#[inline]
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// §8.8.5.2 `filter4_clamp( value )` from `vp9-spec.txt` lines
/// 5824-5826 — clip the signed working sample into the
/// `[-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1]` range.
#[inline]
fn filter4_clamp(value: i32, bit_depth: u8) -> i32 {
    let half = 1_i32 << (bit_depth - 1);
    clip3(-half, half - 1, value)
}

/// Run §8.8.5.2 `narrow filter process` for one edge per
/// `vp9-spec.txt` lines 5795-5853.
///
/// Returns the four mutated samples packaged in [`NarrowFilterOutput`].
/// The caller writes them back to `CurrFrame` at the matching `(y +/-
/// dy*k, x +/- dx*k)` locations.
///
/// # Inputs
///
/// * `samples` — the 4-sample stencil [`NarrowFilterSamples`] the
///   §8.8.5 outer driver assembled by reading `CurrFrame[ plane ][ y
///   +/- dy*k ][ x +/- dx*k ]` per §8.8.5.2 lines 5830-5833.
/// * `hev_mask` — the §8.8.5.1 [`crate::FilterMask::hev_mask`] output
///   for this edge. Picks between the two §8.8.5.2 branches per lines
///   5806-5811.
/// * `bit_depth` — `BitDepth` per §6.2.2 (8, 10, or 12). Drives both
///   the `0x80 << (BitDepth - 8)` offset (lines 5834-5837) and the
///   `Clip3` range inside [`filter4_clamp`] (line 5825).
///
/// # Listing
///
/// `vp9-spec.txt` §8.8.5.2 lines 5829-5852:
///
/// ```text
/// q0 = CurrFrame[ plane ][ y ][ x ]
/// q1 = CurrFrame[ plane ][ y+dy ][ x+dx ]
/// p0 = CurrFrame[ plane ][ y-dy ][ x-dx ]
/// p1 = CurrFrame[ plane ][ y-dy*2 ][ x-dx*2 ]
/// ps1 = p1 - (0x80 << (BitDepth - 8))
/// ps0 = p0 - (0x80 << (BitDepth - 8))
/// qs0 = q0 - (0x80 << (BitDepth - 8))
/// qs1 = q1 - (0x80 << (BitDepth - 8))
/// filter = hevMask ? filter4_clamp( ps1 - qs1 ) : 0
/// filter = filter4_clamp( filter + 3 * (qs0 - ps0) )
/// filter1 = filter4_clamp( filter + 4 ) >> 3
/// filter2 = filter4_clamp( filter + 3 ) >> 3
/// oq0 = filter4_clamp( qs0 - filter1 ) + (0x80 << (BitDepth - 8))
/// op0 = filter4_clamp( ps0 + filter2 ) + (0x80 << (BitDepth - 8))
/// CurrFrame[ plane ][ y ][ x ] = oq0
/// CurrFrame[ plane ][ y-dy ][ x-dx ] = op0
/// if ( !hevMask ) {
///     filter = Round2( filter1, 1 )
///     oq1 = filter4_clamp( qs1 - filter ) + (0x80 << (BitDepth - 8))
///     op1 = filter4_clamp( ps1 + filter ) + (0x80 << (BitDepth - 8))
///     CurrFrame[ plane ][ y+dy ][ x+dx ] = oq1
///     CurrFrame[ plane ][ y-dy*2 ][ x-dx*2 ] = op1
/// }
/// ```
pub fn narrow_filter(
    samples: &NarrowFilterSamples,
    hev_mask: bool,
    bit_depth: u8,
) -> NarrowFilterOutput {
    // §8.8.5.2 line 5834 — `0x80 << (BitDepth - 8)` offset shifting
    // unsigned samples into the signed range. `BitDepth - 8 ∈ {0, 2,
    // 4}` per §6.2.2 `BitDepth ∈ {8, 10, 12}`.
    let shift = (bit_depth - 8) as u32;
    let offset = 0x80_i32 << shift;

    // §8.8.5.2 lines 5834-5837 — `ps1`, `ps0`, `qs0`, `qs1`.
    let ps1 = samples.p1 - offset;
    let ps0 = samples.p0 - offset;
    let qs0 = samples.q0 - offset;
    let qs1 = samples.q1 - offset;

    // §8.8.5.2 line 5838 — `filter = hevMask ? filter4_clamp( ps1 -
    // qs1 ) : 0`. Branchless via boolean discriminator preserves
    // the §3 read order: `filter4_clamp` is only evaluated on the
    // high-edge-variance path so the clamp can never narrow a value
    // we'd otherwise discard.
    let mut filter = if hev_mask {
        filter4_clamp(ps1 - qs1, bit_depth)
    } else {
        0
    };

    // §8.8.5.2 line 5839 — `filter = filter4_clamp( filter + 3 *
    // (qs0 - ps0) )`.
    filter = filter4_clamp(filter + 3 * (qs0 - ps0), bit_depth);

    // §8.8.5.2 lines 5840-5841 — `filter1 = filter4_clamp( filter +
    // 4 ) >> 3` and `filter2 = filter4_clamp( filter + 3 ) >> 3`.
    let filter1 = filter4_clamp(filter + 4, bit_depth) >> 3;
    let filter2 = filter4_clamp(filter + 3, bit_depth) >> 3;

    // §8.8.5.2 lines 5842-5843 — `oq0` and `op0` are always written.
    // The trailing `+ (0x80 << (BitDepth - 8))` undoes the §8.8.5.2
    // line 5834 offset.
    let oq0 = filter4_clamp(qs0 - filter1, bit_depth) + offset;
    let op0 = filter4_clamp(ps0 + filter2, bit_depth) + offset;

    // §8.8.5.2 lines 5846-5852 — `!hevMask` block. The smooth-edge
    // branch additionally writes `op1` and `oq1` using a half-strength
    // pass via `Round2( filter1, 1 )`. When `hevMask == 1` we return
    // the unmodified input samples per the §8.8.5.2 lead paragraph
    // (lines 5809-5811): "this process only modifies the one value on
    // each side of the specified boundary".
    let (op1, oq1) = if hev_mask {
        (samples.p1, samples.q1)
    } else {
        // §3 `Round2( filter1, 1 ) = (filter1 + 1) >> 1`.
        let filter_smooth = round2(filter1, 1);
        let oq1 = filter4_clamp(qs1 - filter_smooth, bit_depth) + offset;
        let op1 = filter4_clamp(ps1 + filter_smooth, bit_depth) + offset;
        (op1, oq1)
    };

    NarrowFilterOutput { op1, op0, oq0, oq1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §8.8.5.2 baseline — a flat 4-sample stencil at the midpoint of
    /// the 8-bit range yields no change, regardless of `hev_mask`.
    /// `qs0 - ps0 == 0` and `ps1 - qs1 == 0` so `filter == 0`,
    /// `filter1 = 4 >> 3 == 0`, `filter2 = 3 >> 3 == 0`. The trailing
    /// offset un-does the `- offset`, so each output equals the input.
    #[test]
    fn flat_stencil_high_variance_no_change() {
        let s = NarrowFilterSamples {
            p1: 128,
            p0: 128,
            q0: 128,
            q1: 128,
        };
        let out = narrow_filter(&s, true, 8);
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

    /// §8.8.5.2 baseline at `hev_mask == 0` — flat stencil also yields
    /// no change. The `Round2( 0, 1 ) = (0 + 1) >> 1 == 0` step zeroes
    /// the smooth-edge half-strength pass so `op1` / `oq1` stay flat.
    #[test]
    fn flat_stencil_low_variance_no_change() {
        let s = NarrowFilterSamples {
            p1: 128,
            p0: 128,
            q0: 128,
            q1: 128,
        };
        let out = narrow_filter(&s, false, 8);
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

    /// §8.8.5.2 lines 5806-5811 — `hev_mask == 1` only modifies `p0`
    /// and `q0`. `p1` and `q1` are returned unchanged so the caller
    /// can unconditionally write them back.
    #[test]
    fn hev_mask_set_preserves_outer_samples() {
        // Sharp step from p side (115) to q side (140). With ps1 == ps0
        // and qs0 == qs1 the `ps1 - qs1` term equals the step magnitude
        // (-25 in working coords), and the `3 * (qs0 - ps0)` term is
        // +75, so `filter = clamp(-25 + 75) = 50`, `filter1 = 54 >> 3 =
        // 6`, `filter2 = 53 >> 3 = 6`. So `oq0 = 140 - 6 = 134` and
        // `op0 = 115 + 6 = 121`. `op1`/`oq1` MUST be the input
        // 115 / 140.
        let s = NarrowFilterSamples {
            p1: 115,
            p0: 115,
            q0: 140,
            q1: 140,
        };
        let out = narrow_filter(&s, true, 8);
        assert_eq!(out.op1, 115, "hev_mask preserves p1");
        assert_eq!(out.oq1, 140, "hev_mask preserves q1");
        assert_eq!(out.op0, 121);
        assert_eq!(out.oq0, 134);
    }

    /// §8.8.5.2 lines 5846-5852 — `hev_mask == 0` modifies all four
    /// samples. The smooth-edge branch zeroes the `ps1 - qs1` step
    /// from the §8.8.5.2 line 5838 `filter` initialization and runs
    /// `Round2( filter1, 1 )` into `p1` / `q1`.
    #[test]
    fn hev_mask_clear_modifies_all_four_samples() {
        // Same step as above but with hev_mask false. ps1 - qs1 term
        // drops out (`filter` starts at 0). `3 * (qs0 - ps0) = 75`, so
        // `filter = 75`, `filter1 = 79 >> 3 = 9`, `filter2 = 78 >> 3 =
        // 9`. `oq0 = 140 - 9 = 131`, `op0 = 115 + 9 = 124`.
        // `Round2(9, 1) = (9 + 1) >> 1 = 5`, so `oq1 = 140 - 5 = 135`,
        // `op1 = 115 + 5 = 120`.
        let s = NarrowFilterSamples {
            p1: 115,
            p0: 115,
            q0: 140,
            q1: 140,
        };
        let out = narrow_filter(&s, false, 8);
        assert_eq!(out.op0, 124);
        assert_eq!(out.oq0, 131);
        assert_eq!(out.op1, 120);
        assert_eq!(out.oq1, 135);
    }

    /// §8.8.5.2 line 5838 — when `hev_mask == 1`, the `ps1 - qs1`
    /// term feeds back into the filter. With `ps1 - qs1 == 0`
    /// (i.e. matched outer samples) but a sharp inner step, the result
    /// matches the `!hev_mask` branch up through `op0` / `oq0` even
    /// though `hev_mask == 1`.
    #[test]
    fn hev_mask_set_matches_smooth_branch_when_outer_samples_match() {
        // ps1 == qs1 means `filter` starts at clamp(0) = 0 even with
        // `hev_mask == 1`. So the inner-pair derivation matches the
        // `hev_mask == 0` branch's `op0` / `oq0`. The `op1` / `oq1`
        // stay as inputs (no smooth pass on the hev branch).
        let s = NarrowFilterSamples {
            p1: 128,
            p0: 120,
            q0: 136,
            q1: 128,
        };
        let out_hev = narrow_filter(&s, true, 8);
        let out_smooth = narrow_filter(&s, false, 8);
        assert_eq!(out_hev.op0, out_smooth.op0);
        assert_eq!(out_hev.oq0, out_smooth.oq0);
        // Inner stays at the input on the hev branch, mutated on the
        // smooth branch.
        assert_eq!(out_hev.op1, 128);
        assert_eq!(out_hev.oq1, 128);
    }

    /// §8.8.5.2 line 5825 — `filter4_clamp` enforces the
    /// `[-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1]` range.
    /// At `BitDepth = 8` that's `[-128, 127]`. Verify a pathological
    /// stencil saturates instead of overflowing.
    #[test]
    fn filter4_clamp_saturates_at_8bit_range() {
        // ps0 = 255 - 128 = 127, qs0 = 0 - 128 = -128.
        // qs0 - ps0 = -255. `3 * -255 = -765`, then `clamp(-765) =
        // -128`. So `filter = -128`. `filter1 = clamp(-128 + 4) =
        // -124 >> 3 = -16` (arithmetic shift). `filter2 = clamp(-128 +
        // 3) = -125 >> 3 = -16`.
        // `oq0 = clamp(-128 - (-16)) + 128 = clamp(-112) + 128 = -112
        // + 128 = 16`.
        // `op0 = clamp(127 + (-16)) + 128 = clamp(111) + 128 = 111 +
        // 128 = 239`.
        let s = NarrowFilterSamples {
            p1: 255,
            p0: 255,
            q0: 0,
            q1: 0,
        };
        let out = narrow_filter(&s, true, 8);
        assert_eq!(out.op0, 239, "p0 lifts toward boundary midpoint");
        assert_eq!(out.oq0, 16, "q0 drops toward boundary midpoint");
        // hev_mask preserves outer samples.
        assert_eq!(out.op1, 255);
        assert_eq!(out.oq1, 0);
    }

    /// §8.8.5.2 line 5814 — `BitDepth = 10` rescales the offset to
    /// `0x80 << 2 = 512`. A flat-at-512 stencil at 10-bit is the
    /// equivalent of a flat-at-128 stencil at 8-bit: every working
    /// sample equals zero, so no change.
    #[test]
    fn bit_depth_10_offset_scales_by_4() {
        let s = NarrowFilterSamples {
            p1: 512,
            p0: 512,
            q0: 512,
            q1: 512,
        };
        let out = narrow_filter(&s, true, 10);
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

    /// §8.8.5.2 line 5825 — `BitDepth = 10` widens `filter4_clamp` to
    /// `[-512, 511]`. With a 10-bit stencil the same proportional
    /// step that saturated at 8-bit no longer saturates at 10-bit.
    #[test]
    fn bit_depth_10_widens_clamp_range() {
        // (1023 - 0) = 1023 in 10-bit. ps0 = 1023 - 512 = 511, qs0 =
        // 0 - 512 = -512. qs0 - ps0 = -1023. 3 * -1023 = -3069.
        // clamp(-3069) at 10-bit = -512 (still saturates — the
        // working range is symmetric so the clamp still triggers).
        // The point of the test is to confirm the clamp DOES kick at
        // the wider 10-bit range and yields different output than the
        // narrower 8-bit clamp.
        let s = NarrowFilterSamples {
            p1: 1023,
            p0: 1023,
            q0: 0,
            q1: 0,
        };
        let out = narrow_filter(&s, true, 10);
        // filter = clamp(-512) = -512.
        // filter1 = clamp(-512 + 4) = -508 >> 3 = -64 (arithmetic
        // shift on -508 = -63.5 floors toward -infinity → -64).
        // filter2 = clamp(-512 + 3) = -509 >> 3 = -64.
        // oq0 = clamp(-512 - (-64)) + 512 = -448 + 512 = 64.
        // op0 = clamp(511 + (-64)) + 512 = 447 + 512 = 959.
        assert_eq!(out.op0, 959);
        assert_eq!(out.oq0, 64);
        // hev_mask preserves outer.
        assert_eq!(out.op1, 1023);
        assert_eq!(out.oq1, 0);
    }

    /// §8.8.5.2 line 5825 — `BitDepth = 12` widens the clamp range
    /// further to `[-2048, 2047]`. Verify the offset rescales to
    /// `0x80 << 4 = 2048`.
    #[test]
    fn bit_depth_12_offset_scales_by_16() {
        let s = NarrowFilterSamples {
            p1: 2048,
            p0: 2048,
            q0: 2048,
            q1: 2048,
        };
        let out = narrow_filter(&s, false, 12);
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

    /// §8.8.5.2 line 5840 — `filter + 4` then `>> 3` floors the
    /// rounding. For `filter == 4`, `(4 + 4) >> 3 = 1`. For `filter
    /// == 3`, `(3 + 4) >> 3 = 0`. So a `qs0 - ps0` step of `1` (i.e.
    /// `3 * 1 = 3` going into `filter`) yields no change at all.
    #[test]
    fn one_step_with_no_outer_variance_yields_no_change() {
        // ps0 = 0, qs0 = 1 → qs0 - ps0 = 1 → filter = 3.
        // filter1 = clamp(7) >> 3 = 0. filter2 = clamp(6) >> 3 = 0.
        // oq0 = qs0 + offset = 1 + 128 = 129 (same as input).
        // op0 = ps0 + offset = 0 + 128 = 128 (same as input).
        let s = NarrowFilterSamples {
            p1: 128,
            p0: 128,
            q0: 129,
            q1: 128,
        };
        let out = narrow_filter(&s, true, 8);
        assert_eq!(out.op0, 128);
        assert_eq!(out.oq0, 129);
    }

    /// §8.8.5.2 line 5841 — `filter2` uses `+ 3` instead of `+ 4`
    /// so the `op0` side rounds with a different bias from `oq0`.
    /// Verify the asymmetry: with `filter == 4`, `filter1 = 1` but
    /// `filter2 = 0`. So `oq0` shifts by 1 toward the midline but
    /// `op0` stays put.
    #[test]
    fn filter1_filter2_asymmetric_rounding() {
        // Find a stencil that makes filter exactly 4.
        // filter starts at clamp(ps1 - qs1) with hev_mask == 1.
        // Pick ps1 - qs1 = -2 → filter = -2.
        // Add 3 * (qs0 - ps0): need 6 to reach 4. qs0 - ps0 = 2.
        // p1 = 100, p0 = 100, q0 = 102, q1 = 102.
        // ps1 - qs1 = -2, filter = -2.
        // 3 * (qs0 - ps0) = 6, filter = clamp(-2 + 6) = 4.
        // filter1 = clamp(4 + 4) >> 3 = 8 >> 3 = 1.
        // filter2 = clamp(4 + 3) >> 3 = 7 >> 3 = 0.
        // oq0 = clamp(qs0 - 1) + 128 = clamp(102 - 128 - 1) + 128
        //     = clamp(-27) + 128 = -27 + 128 = 101.
        // op0 = clamp(ps0 + 0) + 128 = clamp(100 - 128) + 128
        //     = clamp(-28) + 128 = -28 + 128 = 100.
        let s = NarrowFilterSamples {
            p1: 100,
            p0: 100,
            q0: 102,
            q1: 102,
        };
        let out = narrow_filter(&s, true, 8);
        assert_eq!(out.op0, 100, "filter2 = 0 keeps p0 at input");
        assert_eq!(out.oq0, 101, "filter1 = 1 shifts q0 by one");
    }

    /// §8.8.5.2 line 5847 — `Round2( filter1, 1 ) = (filter1 + 1)
    /// >> 1`. For `filter1 == 0`, the smooth pass is 0 (no change to
    /// `p1` / `q1`). For `filter1 == 1`, the smooth pass is
    /// `(1 + 1) >> 1 = 1`. For `filter1 == 3`, the smooth pass is
    /// `(3 + 1) >> 1 = 2`.
    #[test]
    fn smooth_edge_round2_half_strength_pass() {
        // Use a setup where filter1 = 3.
        // hev_mask == 0 means filter starts at 0.
        // 3 * (qs0 - ps0) = filter. Pick qs0 - ps0 = 7 → filter = 21.
        // filter1 = clamp(21 + 4) >> 3 = 25 >> 3 = 3.
        // filter2 = clamp(21 + 3) >> 3 = 24 >> 3 = 3.
        // oq0 = clamp(qs0 - 3) + 128.
        // op0 = clamp(ps0 + 3) + 128.
        // Round2(3, 1) = (3 + 1) >> 1 = 2.
        // oq1 = clamp(qs1 - 2) + 128.
        // op1 = clamp(ps1 + 2) + 128.
        // Use p1 = p0 = 100, q0 = q1 = 107.
        // ps0 = -28, ps1 = -28, qs0 = -21, qs1 = -21.
        // qs0 - ps0 = 7 ✓. filter = 21, filter1 = 3, filter2 = 3.
        // oq0 = clamp(-24) + 128 = -24 + 128 = 104.
        // op0 = clamp(-25) + 128 = -25 + 128 = 103.
        // oq1 = clamp(-23) + 128 = -23 + 128 = 105.
        // op1 = clamp(-26) + 128 = -26 + 128 = 102.
        let s = NarrowFilterSamples {
            p1: 100,
            p0: 100,
            q0: 107,
            q1: 107,
        };
        let out = narrow_filter(&s, false, 8);
        assert_eq!(out.op0, 103, "filter2 = 3 lifts p0 by 3");
        assert_eq!(out.oq0, 104, "filter1 = 3 drops q0 by 3");
        assert_eq!(out.op1, 102, "Round2(3, 1) = 2 lifts p1 by 2");
        assert_eq!(out.oq1, 105, "Round2(3, 1) = 2 drops q1 by 2");
    }
}
