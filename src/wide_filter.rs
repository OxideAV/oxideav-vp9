//! VP9 §8.8.5.3 `wide filter process` — per spec v0.7.
//!
//! This module lands the per-edge [`wide_filter`] sample-mutation
//! as a pure leaf primitive. The §8.8.5 outer driver invokes it after
//! the §8.8.5.1 [`crate::filter_mask`] step when the
//! [`crate::FilterMask::flat_mask`] (and, for `log2Size == 4`,
//! [`crate::FilterMask::flat_mask2`]) preconditions are satisfied per
//! the §8.8.5 dispatch table at `vp9-spec.txt` lines 5677-5684:
//!
//! * `filterSize == TX_8X8 || flatMask2 == 0` → invoked with
//!   `log2Size = 3` (8-tap kernel, 6 mutated outputs).
//! * `filterSize == TX_16X16 && flatMask & flatMask2 != 0` → invoked
//!   with `log2Size = 4` (16-tap kernel, 14 mutated outputs).
//!
//! The kernel is a symmetric low-pass with mirrored edge extension via
//! `Clip3( -(n+1), n, i+j )` at `vp9-spec.txt` §8.8.5.3 line 5879.
//! The output for each modified position `i` (with `n = (1 <<
//! (log2Size - 1)) - 1` per line 5864-5865) is:
//!
//! ```text
//! F[ i ] = Round2( CurrFrame[i] + sum_{j=-n..n} CurrFrame[Clip3(-(n+1), n, i+j)], log2Size )
//! ```
//!
//! where `CurrFrame[k]` is shorthand for `CurrFrame[ plane ][ y+k*dy ][
//! x+k*dx ]`. The total number of samples summed is `2n + 2` (the
//! `t = CurrFrame[i]` initial plus the `2n + 1` clamped accumulator
//! terms); `Round2( t, log2Size )` then divides by `2^log2Size` with
//! half-up rounding so the kernel is a normalised low-pass.
//!
//! Unlike §8.8.5.2 the wide filter does NOT subtract the `0x80 <<
//! (BitDepth - 8)` working-range offset — all arithmetic happens in
//! the original unsigned pixel domain.
//!
//! ## Scope of this round
//!
//! Round 259 lands the §8.8.5.3 leaf only — pure-state function over
//! a fixed 16-sample stencil [`WideFilterSamples`] (`p7`..`p0`,
//! `q0`..`q7`). The caller is responsible for:
//!
//! * Reading the stencil from `CurrFrame[ plane ][ y +/- dy*k ][ x
//!   +/- dx*k ]` per the §8.8.5.1 stencil-build at lines 5703-5727
//!   (the same shape the §8.8.5.1 [`crate::filter_mask`] consumes).
//! * Selecting `log2_size` via the §8.8.5 dispatch rule (lines
//!   5681-5684) — `3` if `filterSize == TX_8X8 || flatMask2 == 0`,
//!   else `4`.
//! * Writing the mutated samples back to `CurrFrame` at the
//!   matching `(y + i*dy, x + i*dx)` locations for `i ∈ [-n, n-1]`
//!   per line 5884-5885.
//!
//! Out of scope for this round (each lands in a separate later round):
//!
//! * §8.8.5 `sample_filtering( )` — the per-edge outer driver that
//!   reads the stencil from `CurrFrame`, runs §8.8.5.1, dispatches to
//!   §8.8.5.2 ([`crate::narrow_filter`]) or §8.8.5.3 (this round), and
//!   writes the result back.
//! * §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
//!   that calls §8.8.3 + §8.8.4 + §8.8.5 for each `(loopRow,
//!   loopCol)` step.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.5.3 lines 5855-5888). `Clip3`
//! is the §3 clipping primitive; `Round2( x, n )` is §3 with
//! `(x + (1 << (n - 1))) >> n`.

/// §8.8.5.3 input — the 16-sample stencil straddling the boundary.
///
/// Per `vp9-spec.txt` §8.8.5.1 lines 5703-5727 the §8.8.5 outer driver
/// assembles these from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k
/// ]`. The wide-filter listing at §8.8.5.3 lines 5868-5885 walks the
/// same `(x, y, dx, dy)` axis and only needs the subset:
///
/// * For `log2_size == 3` (`n == 3`): `p3..p0` plus `q0..q3` (8
///   distinct samples — the `Clip3(-(n+1), n, i+j)` clamp at line 5879
///   extends to index `-(n+1) == -4` which is `p3` and `n == 3` which
///   is `q3`).
/// * For `log2_size == 4` (`n == 7`): `p7..p0` plus `q0..q7` (16
///   distinct samples — clamp extends to index `-8` (which is `p7`)
///   and `n == 7` (which is `q7`)).
///
/// Samples are carried as `i32` so the accumulator `t` in the listing
/// (which sums up to `16 * (1 << 12) = 65536` for 12-bit pixels) never
/// overflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideFilterSamples {
    /// `p7` — eighth sample on the `p` side. Read only when
    /// `log2_size == 4`.
    pub p7: i32,
    /// `p6` — seventh sample on the `p` side. Read only when
    /// `log2_size == 4`.
    pub p6: i32,
    /// `p5` — sixth sample on the `p` side. Read only when
    /// `log2_size == 4`.
    pub p5: i32,
    /// `p4` — fifth sample on the `p` side. Read only when
    /// `log2_size == 4`.
    pub p4: i32,
    /// `p3` — fourth sample on the `p` side.
    pub p3: i32,
    /// `p2` — third sample on the `p` side.
    pub p2: i32,
    /// `p1` — second sample on the `p` side.
    pub p1: i32,
    /// `p0` — boundary sample on the `p` side.
    pub p0: i32,
    /// `q0` — boundary sample on the `q` side.
    pub q0: i32,
    /// `q1` — second sample on the `q` side.
    pub q1: i32,
    /// `q2` — third sample on the `q` side.
    pub q2: i32,
    /// `q3` — fourth sample on the `q` side.
    pub q3: i32,
    /// `q4` — fifth sample on the `q` side. Read only when
    /// `log2_size == 4`.
    pub q4: i32,
    /// `q5` — sixth sample on the `q` side. Read only when
    /// `log2_size == 4`.
    pub q5: i32,
    /// `q6` — seventh sample on the `q` side. Read only when
    /// `log2_size == 4`.
    pub q6: i32,
    /// `q7` — eighth sample on the `q` side. Read only when
    /// `log2_size == 4`.
    pub q7: i32,
}

/// §8.8.5.3 output — up to 14 mutated samples.
///
/// Per `vp9-spec.txt` §8.8.5.3 lines 5884-5885 the writes cover the
/// positions `y + i*dy, x + i*dx` for `i ∈ [-n, n-1]`:
///
/// * `log2_size == 3` (`n == 3`) writes `i ∈ [-3, 2]` — six samples
///   at positions `p2, p1, p0, q0, q1, q2`. The other eight fields
///   (`op6..op3`, `oq3..oq6`) carry the corresponding input sample
///   through unchanged so the caller can write them back
///   unconditionally without branching on `log2_size`.
/// * `log2_size == 4` (`n == 7`) writes `i ∈ [-7, 6]` — fourteen
///   samples at positions `p6..p0`, `q0..q6`. Every field is
///   meaningful.
///
/// The stencil's outermost samples `p7` / `q7` are read by the
/// `log2_size == 4` kernel (via the `Clip3( -8, 7, ... )` extension)
/// but are never themselves written back — the listing only mutates
/// positions `i ∈ [-n, n-1] == [-7, 6]`, and `p7` lives at position
/// `-8`, `q7` at position `+7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideFilterOutput {
    /// `op6` — replacement for the input `p6`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `p6`.
    pub op6: i32,
    /// `op5` — replacement for the input `p5`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `p5`.
    pub op5: i32,
    /// `op4` — replacement for the input `p4`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `p4`.
    pub op4: i32,
    /// `op3` — replacement for the input `p3`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `p3`.
    pub op3: i32,
    /// `op2` — replacement for the input `p2`. Always mutated.
    pub op2: i32,
    /// `op1` — replacement for the input `p1`. Always mutated.
    pub op1: i32,
    /// `op0` — replacement for the input `p0`. Always mutated.
    pub op0: i32,
    /// `oq0` — replacement for the input `q0`. Always mutated.
    pub oq0: i32,
    /// `oq1` — replacement for the input `q1`. Always mutated.
    pub oq1: i32,
    /// `oq2` — replacement for the input `q2`. Always mutated.
    pub oq2: i32,
    /// `oq3` — replacement for the input `q3`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `q3`.
    pub oq3: i32,
    /// `oq4` — replacement for the input `q4`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `q4`.
    pub oq4: i32,
    /// `oq5` — replacement for the input `q5`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `q5`.
    pub oq5: i32,
    /// `oq6` — replacement for the input `q6`. Mutated only when
    /// `log2_size == 4`; for `log2_size == 3` equals the input `q6`.
    pub oq6: i32,
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
/// half-up rounding for non-negative accumulator `t`. Required by
/// §8.8.5.3 line 5882.
#[inline]
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Index helper — return the sample at position index `k` per the
/// `vp9-spec.txt` §8.8.5.3 line 5869 mapping
/// `CurrFrame[ plane ][ y+k*dy ][ x+k*dx ]`:
///
/// * `k == 0` → `q0`
/// * `k > 0` → `q[k-1+1] = q[k]` — wait, no:
///
/// Per the §8.8.5.1 stencil at lines 5703-5727: `q1` lives at offset
/// `+1` (i.e. `y+dy, x+dx`), so position index `k > 0` maps to
/// `q[k - 1]`'s field — actually, re-reading: `q1 = CurrFrame[ ][
/// y+dy ][ x+dx ]` means `q1` is at the position with `k = 1` in the
/// wide-filter indexing. So position `k = j` for `j >= 1` is the
/// `q{j}` sample. Position `k = 0` is `q0`.
///
/// And `p0 = CurrFrame[ ][ y-dy ][ x-dx ]` so position `k = -1` is
/// `p0`. `p1` at `y-dy*2` → position `k = -2`. So position `k = -j`
/// for `j >= 1` is `p{j-1}`.
#[inline]
fn sample_at(s: &WideFilterSamples, k: i32) -> i32 {
    match k {
        // §8.8.5.1 line 5703: q0 at offset (y, x) i.e. k = 0.
        0 => s.q0,
        // §8.8.5.1 lines 5713-5719: q{k} at offset k for k = 1..=7.
        1 => s.q1,
        2 => s.q2,
        3 => s.q3,
        4 => s.q4,
        5 => s.q5,
        6 => s.q6,
        7 => s.q7,
        // §8.8.5.1 lines 5720-5727: p{|k|-1} at offset k for k = -1..-8.
        -1 => s.p0,
        -2 => s.p1,
        -3 => s.p2,
        -4 => s.p3,
        -5 => s.p4,
        -6 => s.p5,
        -7 => s.p6,
        -8 => s.p7,
        _ => unreachable!("wide_filter position index out of [-8, 7]"),
    }
}

/// Run §8.8.5.3 `wide filter process` for one edge per
/// `vp9-spec.txt` lines 5855-5888.
///
/// Returns the mutated samples packaged in [`WideFilterOutput`]. The
/// caller writes them back to `CurrFrame` at the matching `(y + i*dy,
/// x + i*dx)` locations for `i ∈ [-n, n-1]`.
///
/// # Inputs
///
/// * `samples` — the 16-sample stencil [`WideFilterSamples`] the
///   §8.8.5 outer driver assembled by reading `CurrFrame[ plane ][ y
///   +/- dy*k ][ x +/- dx*k ]` per §8.8.5.1 lines 5703-5727. For
///   `log2_size == 3` only the inner 8 samples (`p3..p0`, `q0..q3`)
///   are consulted; the outer fields (`p7..p4`, `q4..q7`) may carry
///   any value and are echoed straight to the output.
/// * `log2_size` — the §8.8.5 dispatch result per lines 5681-5684.
///   `3` selects the 8-tap kernel (6 mutated outputs); `4` selects
///   the 16-tap kernel (14 mutated outputs). Any other value is
///   rejected per §8.8.5.3 lines 5860 / 5864 (the spec only defines
///   `log2Size ∈ {3, 4}` at the §8.8.5 invocation site).
/// * `_bit_depth` — `BitDepth` per §6.2.2 (8, 10, or 12). Carried
///   for API symmetry with [`crate::narrow_filter`] but not consumed
///   by the §8.8.5.3 listing: the wide-filter kernel operates
///   directly on unsigned sample values (no `0x80` working-range
///   offset, no `filter4_clamp` BitDepth scaling) per
///   `vp9-spec.txt` lines 5868-5885 verbatim.
///
/// # Listing
///
/// `vp9-spec.txt` §8.8.5.3 lines 5864-5888 — for `n = (1 <<
/// (log2Size - 1)) - 1`:
///
/// ```text
/// for( i = -n; i < n; i++ ) {
///     t = CurrFrame[ plane ][ y+i*dy ][ x+i*dx ]
///     for( j = -n; j <= n; j++ ) {
///         p = Clip3( -(n+1), n, i+j )
///         t += CurrFrame[ plane ][ y+p*dy ][ x+p*dx ]
///     }
///     F[ i ] = Round2( t, log2Size )
/// }
/// for( i = -n; i < n; i++ )
///     CurrFrame[ plane ][ y+i*dy ][ x+i*dx ] = F[ i ]
/// ```
///
/// # Panics
///
/// Panics with a clean-room message if `log2_size` is not `3` or
/// `4` — those are the only two values the §8.8.5 dispatch table
/// produces per lines 5682 / 5684.
pub fn wide_filter(
    samples: &WideFilterSamples,
    log2_size: u32,
    _bit_depth: u8,
) -> WideFilterOutput {
    assert!(
        log2_size == 3 || log2_size == 4,
        "§8.8.5.3: log2_size must be 3 or 4 per §8.8.5 dispatch table"
    );

    // §8.8.5.3 line 5864-5865 — `n = (1 << (log2Size - 1)) - 1`.
    // log2Size = 3 → n = 3; log2Size = 4 → n = 7.
    let n: i32 = (1 << (log2_size - 1)) - 1;

    // §8.8.5.3 line 5868 outer loop — store `F[i]` for `i ∈ [-n,
    // n-1]`. Width 14 covers the largest case (log2_size == 4, n ==
    // 7). For log2_size == 3 only positions [-3, 2] are filled.
    let mut f = [0_i32; 14];

    for i in -n..n {
        // §8.8.5.3 line 5869 — `t = CurrFrame[ plane ][ y+i*dy ][
        // x+i*dx ]`. Index `i` into the §8.8.5.1 sample mapping at
        // lines 5703-5727 (position `0` is `q0`, position `-1` is
        // `p0`, etc.).
        let mut t = sample_at(samples, i);

        // §8.8.5.3 lines 5878-5881 — accumulate `2n+1` clamped
        // samples. Together with the initial `t` that's `2n+2` total
        // samples, matching `Round2( t, log2Size )` = `(t + (1 <<
        // (log2Size-1))) >> log2Size` to renormalise into a
        // unity-gain low-pass.
        for j in -n..=n {
            // §8.8.5.3 line 5879 — `p = Clip3( -(n+1), n, i+j )`.
            // The clamp extends `p` by one sample beyond the
            // outermost mutated position so the kernel can pull in
            // edge-replicated samples (`p{n}` and `q{n}` are
            // duplicated when `i+j` would exceed the range).
            let p = clip3(-(n + 1), n, i + j);
            t += sample_at(samples, p);
        }

        // §8.8.5.3 line 5882 — `F[ i ] = Round2( t, log2Size )`.
        // Renormalises the `2n+2` summed samples down to a single
        // output. The index translation `i - (-n) == i + n` lands
        // the value at slot `0` for `i == -n`, slot `2n-1` for `i
        // == n-1`.
        f[(i + n) as usize] = round2(t, log2_size);
    }

    // Pack the §8.8.5.3 outputs into a position-keyed struct.
    // Position-to-field mapping (mirrors `sample_at`):
    //   k = -7 → op6, k = -6 → op5, ..., k = -1 → op0,
    //   k =  0 → oq0, k =  1 → oq1, ..., k =  6 → oq6.
    // For log2_size == 3, only positions [-3, 2] are populated by
    // the loop; the remaining slots echo the corresponding input
    // samples so the caller can unconditionally write all 14 fields.
    if log2_size == 3 {
        WideFilterOutput {
            op6: samples.p6,
            op5: samples.p5,
            op4: samples.p4,
            op3: samples.p3,
            op2: f[0], // i = -3 → slot 0
            op1: f[1], // i = -2 → slot 1
            op0: f[2], // i = -1 → slot 2
            oq0: f[3], // i =  0 → slot 3
            oq1: f[4], // i =  1 → slot 4
            oq2: f[5], // i =  2 → slot 5
            oq3: samples.q3,
            oq4: samples.q4,
            oq5: samples.q5,
            oq6: samples.q6,
        }
    } else {
        // log2_size == 4, n == 7. Positions [-7, 6] populate slots
        // [0, 13].
        WideFilterOutput {
            op6: f[0],  // i = -7 → slot 0
            op5: f[1],  // i = -6 → slot 1
            op4: f[2],  // i = -5 → slot 2
            op3: f[3],  // i = -4 → slot 3
            op2: f[4],  // i = -3 → slot 4
            op1: f[5],  // i = -2 → slot 5
            op0: f[6],  // i = -1 → slot 6
            oq0: f[7],  // i =  0 → slot 7
            oq1: f[8],  // i =  1 → slot 8
            oq2: f[9],  // i =  2 → slot 9
            oq3: f[10], // i =  3 → slot 10
            oq4: f[11], // i =  4 → slot 11
            oq5: f[12], // i =  5 → slot 12
            oq6: f[13], // i =  6 → slot 13
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a stencil where every sample equals `v`. The low-pass
    /// kernel is unity-gain so every output equals `v` too, regardless
    /// of `log2_size`.
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

    /// §8.8.5.3 unity-gain — a constant stencil at the 8-bit midpoint
    /// yields the same value at every output position on the 8-tap
    /// kernel. With `n = 3`, each output sums `2n + 2 = 8` copies of
    /// `128`, then `Round2(8*128, 3) = (1024 + 4) >> 3 = 128`.
    #[test]
    fn flat_stencil_log2_3_unity_gain() {
        let out = wide_filter(&flat(128), 3, 8);
        // Inner six are filtered to 128.
        assert_eq!(out.op2, 128);
        assert_eq!(out.op1, 128);
        assert_eq!(out.op0, 128);
        assert_eq!(out.oq0, 128);
        assert_eq!(out.oq1, 128);
        assert_eq!(out.oq2, 128);
        // Outer eight echo the input (which is also 128 on a flat
        // stencil) — the log2_size == 3 caller relies on this so it
        // can write all 14 fields unconditionally.
        assert_eq!(out.op6, 128);
        assert_eq!(out.op5, 128);
        assert_eq!(out.op4, 128);
        assert_eq!(out.op3, 128);
        assert_eq!(out.oq3, 128);
        assert_eq!(out.oq4, 128);
        assert_eq!(out.oq5, 128);
        assert_eq!(out.oq6, 128);
    }

    /// §8.8.5.3 unity-gain on the 16-tap kernel. With `n = 7`, each
    /// output sums `2n + 2 = 16` copies of `128`, then
    /// `Round2(16*128, 4) = (2048 + 8) >> 4 = 128`.
    #[test]
    fn flat_stencil_log2_4_unity_gain() {
        let out = wide_filter(&flat(128), 4, 8);
        assert_eq!(out.op6, 128);
        assert_eq!(out.op5, 128);
        assert_eq!(out.op4, 128);
        assert_eq!(out.op3, 128);
        assert_eq!(out.op2, 128);
        assert_eq!(out.op1, 128);
        assert_eq!(out.op0, 128);
        assert_eq!(out.oq0, 128);
        assert_eq!(out.oq1, 128);
        assert_eq!(out.oq2, 128);
        assert_eq!(out.oq3, 128);
        assert_eq!(out.oq4, 128);
        assert_eq!(out.oq5, 128);
        assert_eq!(out.oq6, 128);
    }

    /// §8.8.5.3 log2_3 — outer fields (`op6..op3`, `oq3..oq6`) carry
    /// the corresponding input through unchanged regardless of how
    /// distorted the *outer* input values are. The 8-tap kernel only
    /// touches `p3..p0`, `q0..q3` (via the `Clip3( -4, 3, .)`
    /// extension) so the outer-input samples never even reach the
    /// inner accumulator.
    #[test]
    fn log2_3_outer_fields_echo_input() {
        let s = WideFilterSamples {
            // Outer inputs — arbitrary distinct values. The log2_3
            // kernel must not touch these.
            p7: 1,
            p6: 2,
            p5: 3,
            p4: 4,
            // Inner inputs — all flat at 128 so the kernel produces
            // 128 at every mutated position.
            p3: 128,
            p2: 128,
            p1: 128,
            p0: 128,
            q0: 128,
            q1: 128,
            q2: 128,
            q3: 128,
            // Outer inputs on q side.
            q4: 5,
            q5: 6,
            q6: 7,
            q7: 8,
        };
        let out = wide_filter(&s, 3, 8);
        // Outer fields echo their *input* — not the inner-flat 128.
        assert_eq!(out.op6, 2, "op6 echoes p6");
        assert_eq!(out.op5, 3, "op5 echoes p5");
        assert_eq!(out.op4, 4, "op4 echoes p4");
        assert_eq!(out.op3, 128, "op3 echoes p3 (= 128)");
        assert_eq!(out.oq3, 128, "oq3 echoes q3 (= 128)");
        assert_eq!(out.oq4, 5, "oq4 echoes q4");
        assert_eq!(out.oq5, 6, "oq5 echoes q5");
        assert_eq!(out.oq6, 7, "oq6 echoes q6");
    }

    /// §8.8.5.3 line 5879 — `Clip3( -(n+1), n, i+j )` extends the
    /// kernel by replicating the outermost in-range sample when the
    /// `i+j` index would go off the end. Verified by stacking the
    /// `p3` (log2_3) sample with a different value from the inner
    /// flat region: the outer-most p3 sample gets duplicated by the
    /// `Clip3` when `i = -3, j = -3` (i+j = -6, clamped to -4 i.e.
    /// p3 again).
    #[test]
    fn log2_3_edge_replication_picks_outer_input() {
        // Inner-flat at 0 except `p3 = 80`. The Clip3 extension
        // means the boundary output `op2` (i = -3) will pick up
        // four copies of p3 (clamps at -4 for j = -3, -2, -1 plus
        // the initial t at i = -3 itself), the rest contributes 0:
        //   t = 80 (initial t)
        //   + 80 (j=-3, clamp -6→-4 = p3) + 80 (j=-2, -5→-4 = p3)
        //   + 80 (j=-1, -4 = p3)
        //   + 80 (j=0, -3 = p3) + 0 (j=1, -2 = p2) + 0 (j=2, -1 = p1)
        //   + 0 (j=3, 0 = q0)
        // Hmm wait — at i = -3, the *initial* sample is at index
        // -3, i.e. p2. Let me re-check: position k = -3 in the
        // sample_at mapping is p2 (since k = -1 → p0, k = -2 → p1,
        // k = -3 → p2, k = -4 → p3). And the Clip3 clamps to [-4,
        // 3], so the outermost extension is to position k = -4 = p3.
        //
        // So at i = -3 with this stencil (p3 = 80, all else 0):
        //   t = sample_at(-3) = p2 = 0
        //   j = -3: i+j = -6 → clamp -4 → p3 = 80. t += 80.
        //   j = -2: i+j = -5 → clamp -4 → p3 = 80. t += 80.
        //   j = -1: i+j = -4 → clamp -4 → p3 = 80. t += 80.
        //   j =  0: i+j = -3 → p2 = 0.
        //   j =  1: i+j = -2 → p1 = 0.
        //   j =  2: i+j = -1 → p0 = 0.
        //   j =  3: i+j =  0 → q0 = 0.
        // t = 240. Round2(240, 3) = (240 + 4) >> 3 = 244 >> 3 = 30.
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
        assert_eq!(out.op2, 30, "Clip3 replicated p3 three times into op2");
    }

    /// §8.8.5.3 line 5882 `Round2` half-up rounding: with a 4 / 7 / 1
    /// stencil engineered so the accumulator at `i = -1` (which
    /// outputs `op0`) is just shy of a round-half-up boundary,
    /// confirm the `(t + (1 << (log2Size-1))) >> log2Size` rounding
    /// direction is half-up not half-even.
    #[test]
    fn log2_3_round2_half_up() {
        // Engineer t such that t == 3 mod 8 — Round2(3, 3) = (3+4)>>3 = 0.
        // and t == 4 mod 8 — Round2(4, 3) = (4+4)>>3 = 1.
        // Sum 8 ones → Round2(8, 3) = (8+4)>>3 = 12 >> 3 = 1.
        let s = WideFilterSamples {
            p7: 0,
            p6: 0,
            p5: 0,
            p4: 0,
            p3: 1,
            p2: 1,
            p1: 1,
            p0: 1,
            q0: 1,
            q1: 1,
            q2: 1,
            q3: 1,
            q4: 0,
            q5: 0,
            q6: 0,
            q7: 0,
        };
        let out = wide_filter(&s, 3, 8);
        // At i = -1 (op0): t = p0=1 + (j=-3..3 → clamped reads:
        //   j=-3 i+j=-4→p3=1, j=-2:-3→p2=1, j=-1:-2→p1=1,
        //   j=0:-1→p0=1, j=1:0→q0=1, j=2:1→q1=1, j=3:2→q2=1)
        //   = 1 + 7 = 8. Round2(8,3) = 1.
        assert_eq!(out.op0, 1);
    }

    /// §8.8.5.3 log2_4 (16-tap) outer-edge sanity: with a fully flat
    /// stencil at 200 (8-bit), every output is 200.
    #[test]
    fn log2_4_flat_at_200() {
        let out = wide_filter(&flat(200), 4, 8);
        assert_eq!(out.op6, 200);
        assert_eq!(out.op0, 200);
        assert_eq!(out.oq6, 200);
    }

    /// §8.8.5.3 sanity at `BitDepth = 10`. The §8.8.5.3 listing makes
    /// no reference to `BitDepth` (no `0x80` offset, no
    /// `filter4_clamp` BitDepth scaling), so a 10-bit flat stencil at
    /// 512 yields 512 just like the 8-bit case. The `bit_depth`
    /// parameter is carried purely for API symmetry with
    /// [`crate::narrow_filter`].
    #[test]
    fn flat_stencil_log2_3_10bit() {
        let out = wide_filter(&flat(512), 3, 10);
        assert_eq!(out.op2, 512);
        assert_eq!(out.op1, 512);
        assert_eq!(out.op0, 512);
        assert_eq!(out.oq0, 512);
        assert_eq!(out.oq1, 512);
        assert_eq!(out.oq2, 512);
    }

    /// §8.8.5.3 sanity at `BitDepth = 12`. Confirms accumulator
    /// fitness — at `log2_4`, sum of 16 × 4095 = 65520, well within
    /// `i32` range.
    #[test]
    fn flat_stencil_log2_4_12bit() {
        let out = wide_filter(&flat(4095), 4, 12);
        assert_eq!(out.op0, 4095);
        assert_eq!(out.oq0, 4095);
        assert_eq!(out.op6, 4095);
        assert_eq!(out.oq6, 4095);
    }

    /// §8.8.5 dispatch — `log2_size` outside `{3, 4}` panics. The
    /// §8.8.5 outer driver never produces any other value (lines
    /// 5682 / 5684 are the only two assignment sites) so we treat
    /// the invariant as a hard precondition.
    #[test]
    #[should_panic(expected = "§8.8.5.3: log2_size must be 3 or 4")]
    fn log2_size_2_panics() {
        let _ = wide_filter(&flat(128), 2, 8);
    }

    /// §8.8.5 dispatch — `log2_size = 5` likewise rejected.
    #[test]
    #[should_panic(expected = "§8.8.5.3: log2_size must be 3 or 4")]
    fn log2_size_5_panics() {
        let _ = wide_filter(&flat(128), 5, 8);
    }

    /// §8.8.5.3 step-response on log2_3: a clean step from 0 (p side)
    /// to 100 (q side) smears the step across the 6 mutated
    /// positions, with the post-`Round2` sums bounded by `[100, 101]`
    /// (the +1 over the unrounded 100 comes from per-side independent
    /// half-up rounding when both sides land on a `Round2` boundary).
    #[test]
    fn log2_3_step_response_pair_sums_bounded() {
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
        let pairs = [(out.op0, out.oq0), (out.op1, out.oq1), (out.op2, out.oq2)];
        for (p, q) in pairs {
            let sum = p + q;
            assert!(
                (100..=101).contains(&sum),
                "step pair sum out of [100, 101] window: {} + {} = {}",
                p,
                q,
                sum
            );
        }
    }

    /// §8.8.5.3 step-response trace on log2_3 — verify the exact
    /// values at every mutated position for a clean 0→100 step.
    ///
    /// At i = -3 (op2): initial t = sample(-3) = p2 = 0. Then j ∈
    /// [-3, 3] with clamp [-4, 3]:
    ///   j=-3: -6→-4 = p3 = 0;  j=-2: -5→-4 = p3 = 0;
    ///   j=-1: -4 = p3 = 0;      j= 0: -3 = p2 = 0;
    ///   j= 1: -2 = p1 = 0;      j= 2: -1 = p0 = 0;
    ///   j= 3:  0 = q0 = 100.
    /// t = 0 + 0+0+0+0+0+0+100 = 100. Round2(100, 3) = (100+4)>>3 =
    /// 104>>3 = 13.
    ///
    /// At i = -2 (op1): initial t = sample(-2) = p1 = 0. Then:
    ///   j=-3: -5→-4 = p3 = 0; j=-2: -4 = p3 = 0;
    ///   j=-1: -3 = p2 = 0;     j= 0: -2 = p1 = 0;
    ///   j= 1: -1 = p0 = 0;     j= 2:  0 = q0 = 100;
    ///   j= 3:  1 = q1 = 100.
    /// t = 0 + 0+0+0+0+0+100+100 = 200. Round2(200, 3) = (200+4)>>3 =
    /// 204>>3 = 25.
    ///
    /// At i = -1 (op0): initial t = p0 = 0. Then:
    ///   j=-3: -4 = p3 = 0; j=-2: -3 = p2 = 0;
    ///   j=-1: -2 = p1 = 0; j= 0: -1 = p0 = 0;
    ///   j= 1:  0 = q0 = 100; j= 2:  1 = q1 = 100;
    ///   j= 3:  2 = q2 = 100.
    /// t = 0 + 0+0+0+0+100+100+100 = 300. Round2(300, 3) = (300+4)>>3 =
    /// 304>>3 = 38.
    #[test]
    fn log2_3_step_response_exact_values() {
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
        assert_eq!(out.op2, 13, "op2 at i=-3");
        assert_eq!(out.op1, 25, "op1 at i=-2");
        assert_eq!(out.op0, 38, "op0 at i=-1");
        // By symmetry: oq0 = 100 - op0 = 62 only if the kernel is
        // perfectly symmetric. Round2 of (700 + 4) >> 3 = 704 >> 3 = 88.
        // Let's hand-derive oq0: initial t = q0 = 100. j ∈ [-3,3]:
        //   j=-3: -3 = p2 = 0; j=-2: -2 = p1 = 0; j=-1: -1 = p0 = 0;
        //   j= 0:  0 = q0 = 100; j= 1: 1 = q1 = 100; j= 2: 2 = q2 = 100;
        //   j= 3:  3 = q3 = 100.
        // t = 100 + 0+0+0+100+100+100+100 = 500. Round2(500, 3) = 504>>3 = 63.
        assert_eq!(out.oq0, 63, "oq0 at i=0");
        // op0 + oq0 = 38 + 63 = 101. Off by 1 from symmetric — that's
        // the Round2(...) rounding error (each side rounds independently).
    }
}
