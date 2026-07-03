//! Forward transforms for the VP9 **encoder** — the exact inverses of
//! the §8.7 inverse-transform processes the decoder runs.
//!
//! The VP9 specification (v0.7) defines only the *decode*-side
//! transforms; an encoder needs the corresponding forward maps. This
//! module derives them from the spec's inverse listings alone:
//!
//! * **Lossless (WHT)** — the §8.7.1.10 inverse Walsh-Hadamard butterfly
//!   at `shift == 0` is an exact integer involution: applying it twice
//!   returns the input bit-for-bit. (Proof sketch: with `s = t0 + t1`
//!   and `u = t2 - t3`, the second application computes `E = (s+u) >> 1`
//!   against the first's `e = (s-u) >> 1`; `s+u` and `s-u` share parity,
//!   so `E - e == u` exactly for every integer input, and the four
//!   outputs collapse back to `t0..t3`.) The §8.7.2 lossless decode is
//!   `R = Wcols( Wrows( Dequant >> 2 ) )` with `Dequant = 4 * Tokens`
//!   (the §8.6.1 lossless quantizer is 4, and `(4·T) >> 2 == T`
//!   exactly), so the forward map is simply the same butterfly applied
//!   in the opposite pass order: `Tokens = Wrows( Wcols( R ) )`. This
//!   makes [`forward_wht_2d`] a *perfect* inverse — any integer residual
//!   block round-trips bit-exactly through the decoder's
//!   dequant + inverse-WHT path.
//!
//! * **Lossy (DCT)** — the §8.7.1.3 inverse DCT realises the orthonormal
//!   inverse-DCT-III basis with the DC row scaled by `1/√2` and an
//!   overall gain folded into its fixed-point rotations (the in-crate
//!   §8.7 oracle tests recover the DC-only response ratio
//!   `0.70703125 ≈ 1/√2`), followed by the §8.7.2
//!   `Round2( ., Min( 6, n + 2 ) )` column rounding. [`forward_dct_2d`]
//!   evaluates the matching *forward* basis in floating point with the
//!   reciprocal gain, producing transform-domain values scaled so that
//!   the decoder's integer inverse reproduces the input residual to
//!   within the inverse network's own fixed-point tolerance. Exactness
//!   is not required on the lossy path — quantization dominates — and
//!   the encoder's reconstruction loop always replays the *decoder's*
//!   integer inverse to keep encoder / decoder state identical.
//!
//! Provenance: derived exclusively from the §8.7.1.10 / §8.7.1.3 /
//! §8.7.2 / §8.6.1 listings in `docs/video/vp9/vp9-spec.txt` and their
//! in-crate transcriptions; no external encoder was consulted.

#![allow(dead_code)]

use crate::idct::inverse_wht;

/// Forward 2D Walsh-Hadamard transform of a 4x4 residual block stored
/// row-major in `block` (in place) — the exact inverse of the §8.7.2
/// lossless inverse-transform path.
///
/// On return `block` holds the quantized `Tokens[ ]` values the §6.4.24
/// coefficient syntax codes: the decoder's §8.6.1 dequant (`* 4`), §8.7.2
/// row WHT (`shift == 2`) and column WHT (`shift == 0`) reproduce the
/// input residual bit-exactly for every integer input.
pub(crate) fn forward_wht_2d(block: &mut [i64]) {
    debug_assert_eq!(block.len(), 16, "lossless WHT operates on 4x4 blocks");
    let mut t = [0i64; 4];
    // Undo the decoder's column pass first (it ran last): the §8.7.1.10
    // butterfly at shift 0 is its own inverse.
    for j in 0..4 {
        for i in 0..4 {
            t[i] = block[i * 4 + j];
        }
        inverse_wht(&mut t, 0);
        for i in 0..4 {
            block[i * 4 + j] = t[i];
        }
    }
    // Then undo the row pass. The decoder's row pass consumed
    // `Dequant >> 2` with `Dequant == 4 * Tokens`, so the row-domain
    // values recovered here *are* the tokens.
    for row in block.chunks_exact_mut(4) {
        t.copy_from_slice(row);
        inverse_wht(&mut t, 0);
        row.copy_from_slice(&t);
    }
}

/// Forward 2D type-II DCT of an `n0 × n0` residual block (row-major in
/// `block`, `n0 = 1 << n`, `2 <= n <= 5`), scaled to the coefficient
/// domain the §8.7 integer inverse consumes.
///
/// The §8.7 inverse realises, per 1D pass, the inverse-DCT-III basis
///
/// ```text
///   x[ i ] = Σ_k ( k == 0 ? 1/√2 : 1 ) · X[ k ] · cos( (2i+1)kπ / 2N )
/// ```
///
/// with no `2/N` normalization (each pass carries an implicit gain of
/// `N/2` relative to the orthonormal transform), followed by the §8.7.2
/// `Round2( ., Min(6, n+2) )` column rounding — a division by
/// `2^Min(6, n+2)`. The matching forward map per 1D pass is
///
/// ```text
///   X[ k ] = ( k == 0 ? 1/√2 : 1 ) · (2/N) · Σ_i x[ i ] · cos( (2i+1)kπ / 2N )
/// ```
///
/// and the 2D forward applies it to rows then columns, then multiplies
/// by `2^Min(6, n+2)` to pre-compensate the decoder's final rounding.
/// The result is rounded to the nearest integer; the residual value the
/// decoder reconstructs differs from the input only by the inverse
/// network's fixed-point tolerance (a few ULPs), which the encoder's
/// reconstruction loop absorbs by replaying the decoder's own integer
/// inverse.
pub(crate) fn forward_dct_2d(block: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    debug_assert_eq!(block.len(), n0 * n0, "block must be n0*n0");
    let nf = n0 as f64;

    let mut work: Vec<f64> = block.iter().map(|&v| v as f64).collect();
    let mut t = vec![0.0f64; n0];

    // 1D forward DCT-II with the §8.7-matching per-pass scaling:
    // X[k] = (k==0 ? 1/sqrt(2) : 1) * (2/N) * sum_i x[i] cos((2i+1)k pi / 2N).
    let fwd_1d = |src: &[f64], dst: &mut [f64]| {
        for (k, slot) in dst.iter_mut().enumerate() {
            let scale = if k == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let sum: f64 = src
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    x * ((2 * i + 1) as f64 * k as f64 * std::f64::consts::PI / (2.0 * nf)).cos()
                })
                .sum();
            *slot = scale * (2.0 / nf) * sum;
        }
    };

    // Rows.
    let mut row_out = vec![0.0f64; n0];
    for r in 0..n0 {
        t.copy_from_slice(&work[r * n0..(r + 1) * n0]);
        fwd_1d(&t, &mut row_out);
        work[r * n0..(r + 1) * n0].copy_from_slice(&row_out);
    }
    // Columns.
    let mut col_in = vec![0.0f64; n0];
    let mut col_out = vec![0.0f64; n0];
    for c in 0..n0 {
        for r in 0..n0 {
            col_in[r] = work[r * n0 + c];
        }
        fwd_1d(&col_in, &mut col_out);
        for r in 0..n0 {
            work[r * n0 + c] = col_out[r];
        }
    }

    // Pre-compensate the decoder's §8.7.2 Round2( ., Min(6, n+2) ).
    let gain = f64::from(1u32 << core::cmp::min(6, n + 2));
    for (slot, &v) in block.iter_mut().zip(work.iter()) {
        *slot = (v * gain).round() as i64;
    }
}

/// Quantize a transform-domain coefficient block to the `Tokens[ ]`
/// values the §6.4.24 syntax codes — the inverse of the §8.6.2 step-1/2
/// dequant `Dequant[ i ] = Tokens[ i ] * quant / dqDenom` (with
/// `dqDenom == 1` for the sizes below TX_32X32 this encoder emits).
///
/// Rounds to nearest with ties away from zero, so the decoder's
/// dequantized coefficient differs from the input by at most `quant / 2`
/// per coefficient. `coefs[ 0 ]` uses `dc_quant`; the rest `ac_quant`.
pub(crate) fn quantize_block(coefs: &mut [i64], dc_quant: i32, ac_quant: i32) {
    for (i, c) in coefs.iter_mut().enumerate() {
        let q = i64::from(if i == 0 { dc_quant } else { ac_quant });
        let a = c.abs();
        let t = (2 * a + q) / (2 * q);
        *c = if *c < 0 { -t } else { t };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::{inverse_transform_2d, DCT_DCT};

    /// Simulate the decoder's lossless reconstruction of a token block:
    /// §8.6.1 dequant (lossless quantizer = 4 for DC and AC) then the
    /// §8.7.2 lossless inverse transform.
    fn decode_lossless(tokens: &[i64]) -> Vec<i64> {
        let mut dequant: Vec<i64> = tokens.iter().map(|&t| t * 4).collect();
        inverse_transform_2d(&mut dequant, 2, DCT_DCT, true);
        dequant
    }

    #[test]
    fn forward_wht_zero_in_zero_out() {
        let mut block = vec![0i64; 16];
        forward_wht_2d(&mut block);
        assert!(block.iter().all(|&v| v == 0));
    }

    /// Every integer residual round-trips bit-exactly through
    /// forward WHT -> (decoder) dequant + inverse WHT.
    #[test]
    fn forward_wht_roundtrips_random_residuals_exactly() {
        let mut state: u64 = 0xA3C5_9B12_44E7_D081;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..2000 {
            let residual: Vec<i64> = (0..16).map(|_| (next() % 511) - 255).collect();
            let mut tokens = residual.clone();
            forward_wht_2d(&mut tokens);
            let recon = decode_lossless(&tokens);
            assert_eq!(recon, residual, "lossless WHT round-trip not exact");
        }
    }

    /// The extreme 8-bit / 10-bit / 12-bit residual ranges round-trip
    /// exactly (all-max, all-min, checkerboard).
    #[test]
    fn forward_wht_roundtrips_extreme_residuals() {
        for &m in &[255i64, 1023, 4095] {
            let patterns: [Vec<i64>; 4] = [
                vec![m; 16],
                vec![-m; 16],
                (0..16).map(|i| if i % 2 == 0 { m } else { -m }).collect(),
                (0..16).map(|i| ((i as i64) * 2 * m / 15) - m).collect(),
            ];
            for residual in &patterns {
                let mut tokens = residual.clone();
                forward_wht_2d(&mut tokens);
                let recon = decode_lossless(&tokens);
                assert_eq!(&recon, residual, "extreme WHT round-trip (m={m})");
            }
        }
    }

    /// Coefficient magnitudes stay within the range the §6.4.26 CAT6
    /// extra-bits can code at the matching bit depth (a 4x4 WHT has a
    /// worst-case gain of 16 across both passes before the `>> 1`
    /// interior halving; measured max stays well under `1 << 14` for
    /// 8-bit residuals).
    #[test]
    fn forward_wht_coefficients_are_codeable() {
        let mut worst = 0i64;
        let mut state: u64 = 0x77E1_52C9_0A3B_66DD;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..2000 {
            let mut block: Vec<i64> = (0..16).map(|_| (next() % 511) - 255).collect();
            forward_wht_2d(&mut block);
            worst = worst.max(block.iter().map(|&v| v.abs()).max().unwrap());
        }
        // Checkerboard worst case.
        let mut cb: Vec<i64> = (0..16)
            .map(|i| if (i + i / 4) % 2 == 0 { 255 } else { -255 })
            .collect();
        forward_wht_2d(&mut cb);
        worst = worst.max(cb.iter().map(|&v| v.abs()).max().unwrap());
        assert!(
            worst < (1 << 13),
            "worst |token| = {worst} exceeds CAT6 range"
        );
    }

    /// The forward DCT inverts the decoder's integer inverse to within
    /// the fixed-point tolerance of the inverse network, for every
    /// transform size.
    #[test]
    fn forward_dct_roundtrips_within_tolerance() {
        let mut state: u64 = 0x5D0C_31F8_A9B4_7E62;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            // The inverse network's per-sample error grows with the
            // basis-sum length; a small fixed bound suffices in practice.
            let tol = 2 + n as i64;
            for _ in 0..50 {
                let residual: Vec<i64> = (0..n0 * n0).map(|_| (next() % 511) - 255).collect();
                let mut coefs = residual.clone();
                forward_dct_2d(&mut coefs, n);
                // Feed the coefficients straight to the decoder's §8.7.2
                // integer inverse (quantization is a separate layer).
                let mut dequant = coefs.clone();
                inverse_transform_2d(&mut dequant, n, DCT_DCT, false);
                for (i, (&r, &d)) in residual.iter().zip(dequant.iter()).enumerate() {
                    let err = (r - d).abs();
                    assert!(
                        err <= tol,
                        "n={n} i={i}: residual={r} decoded={d} err={err} > {tol}"
                    );
                }
            }
        }
    }

    /// Quantize + dequantize brackets every coefficient within
    /// `quant / 2`, and exact multiples pass through unchanged.
    #[test]
    fn quantize_roundtrip_error_is_bounded() {
        let dcq = 8i32;
        let acq = 12i32;
        let mut coefs: Vec<i64> = (-100..100).map(|i| i * 7 + 3).collect();
        let orig = coefs.clone();
        quantize_block(&mut coefs, dcq, acq);
        for (i, (&tok, &c)) in coefs.iter().zip(orig.iter()).enumerate() {
            let q = i64::from(if i == 0 { dcq } else { acq });
            let deq = tok * q;
            assert!(
                (deq - c).abs() <= q / 2,
                "i={i}: coef {c} -> token {tok} -> {deq}"
            );
        }
        // Exact multiples are preserved.
        let mut exact = vec![0i64, 24, -36, 12];
        quantize_block(&mut exact, 8, 12);
        assert_eq!(exact, vec![0, 2, -3, 1]);
    }

    /// DC-only forward: a flat residual concentrates (almost) all its
    /// energy in the DC coefficient.
    #[test]
    fn forward_dct_flat_block_is_dc_only() {
        for n in 2..=4u32 {
            let n0 = 1usize << n;
            let mut block = vec![100i64; n0 * n0];
            forward_dct_2d(&mut block, n);
            assert_ne!(block[0], 0, "n={n}: DC must be non-zero");
            for (i, &v) in block.iter().enumerate().skip(1) {
                assert!(v.abs() <= 1, "n={n} i={i}: AC leakage {v}");
            }
        }
    }
}
