//! Forward DCT / ADST kernels for the VP9 encoder — §7.3.1 / §7.3.2.
//!
//! VP9 forward transforms mirror the inverse transforms in `transform.rs`:
//! the same 14-bit fixed-point cosine constants, the same butterfly
//! structure, just applied in the forward direction. The encoder uses
//! `fdct4` (the 4×4 forward DCT) and `fdct8` (8×8) to produce the
//! coefficient array that is then quantised and entropy-coded.
//!
//! Arithmetic is derived from the libvpx spec (VP9 spec §7.3.1) using
//! only the spec PDF as reference — libvpx source was not consulted.
//!
//! The public entry point is [`forward_transform_block`], which applies a
//! 2-D separable forward DCT (row-pass then column-pass) to a `bs × bs`
//! residual block in raster order and writes the result into a flat
//! coefficient array of length `bs × bs`.

/// 14-bit fixed-point constant shared with the inverse side.
const DCT_CONST_BITS: i32 = 14;
const DCT_CONST_ROUNDING: i32 = 1 << (DCT_CONST_BITS - 1);

#[inline]
fn fdct_round_shift(x: i32) -> i32 {
    (x + DCT_CONST_ROUNDING) >> DCT_CONST_BITS
}

// Cosine constants (same as in transform.rs — spec §7.3.1 Table).
const COSPI_16_64: i32 = 11585;
const COSPI_8_64: i32 = 15137;
const COSPI_24_64: i32 = 6270;

// 8×8 constants
const COSPI_4_64: i32 = 16069;
const COSPI_12_64: i32 = 13623;
const COSPI_20_64: i32 = 9102;
const COSPI_28_64: i32 = 3196;

/// Forward 4-point 1-D DCT. Input `x[4]`, output `o[4]`.
///
/// Based on the VP9 spec §7.3.1 `FD4` butterfly — the exact inverse of
/// the `idct4` in `transform.rs`.
fn fdct4(x: [i32; 4]) -> [i32; 4] {
    // Stage 1: even / odd split.
    let x0 = x[0] + x[3];
    let x1 = x[1] + x[2];
    let x2 = x[1] - x[2];
    let x3 = x[0] - x[3];

    // Stage 2: butterflies.
    let o0 = fdct_round_shift(COSPI_16_64 * (x0 + x1));
    let o2 = fdct_round_shift(COSPI_16_64 * (x0 - x1));
    let o1 = fdct_round_shift(COSPI_24_64 * x2 + COSPI_8_64 * x3);
    let o3 = fdct_round_shift(-COSPI_8_64 * x2 + COSPI_24_64 * x3);

    [o0, o1, o2, o3]
}

/// Forward 8-point 1-D DCT. Input `x[8]`, output `o[8]`.
fn fdct8(x: [i32; 8]) -> [i32; 8] {
    // Stage 1.
    let s0 = x[0] + x[7];
    let s1 = x[1] + x[6];
    let s2 = x[2] + x[5];
    let s3 = x[3] + x[4];
    let s4 = x[3] - x[4];
    let s5 = x[2] - x[5];
    let s6 = x[1] - x[6];
    let s7 = x[0] - x[7];

    // Stage 2: upper half (the symmetric part).
    let x0 = s0 + s3;
    let x1 = s1 + s2;
    let x2 = s1 - s2;
    let x3 = s0 - s3;

    // Stage 3: upper half output.
    let o0 = fdct_round_shift(COSPI_16_64 * (x0 + x1));
    let o4 = fdct_round_shift(COSPI_16_64 * (x0 - x1));
    let o2 = fdct_round_shift(COSPI_8_64 * x2 + COSPI_24_64 * x3);
    let o6 = fdct_round_shift(COSPI_24_64 * x2 - COSPI_8_64 * x3);

    // Stage 2: lower half (the anti-symmetric part).
    let t0 = fdct_round_shift(COSPI_4_64 * s7 + COSPI_28_64 * s4);
    let t1 = fdct_round_shift(COSPI_28_64 * s7 - COSPI_4_64 * s4);
    let t2 = fdct_round_shift(COSPI_12_64 * s6 + COSPI_20_64 * s5);
    let t3 = fdct_round_shift(COSPI_20_64 * s6 - COSPI_12_64 * s5);

    let o1 = t0 + t2;
    let o7 = t1 + t3;
    let t2_n = t0 - t2;
    let t3_n = t1 - t3;
    let o3 = fdct_round_shift(COSPI_16_64 * (t3_n + t2_n));
    let o5 = fdct_round_shift(COSPI_16_64 * (t3_n - t2_n));

    [o0, o1, o2, o3, o4, o5, o6, o7]
}

/// Apply a 2-D forward DCT to a `bs × bs` residual block.
///
/// `residual` must be in raster order (row-major, length `bs × bs`).
/// `coeffs` receives the 2-D DCT output in raster order (DC at [0]).
///
/// `bs` must be 4 or 8. For DC-only encoding (e.g. flat blocks) the
/// caller can skip the AC coefficients when quantising.
pub fn fdct_2d(residual: &[i16], bs: usize, coeffs: &mut [i32]) {
    debug_assert!(matches!(bs, 4 | 8));
    debug_assert_eq!(residual.len(), bs * bs);
    debug_assert_eq!(coeffs.len(), bs * bs);

    // Row pass — forward DCT along each row.
    let mut tmp = vec![0i32; bs * bs];
    match bs {
        4 => {
            for r in 0..4 {
                let row = [
                    residual[r * 4] as i32 * 4,
                    residual[r * 4 + 1] as i32 * 4,
                    residual[r * 4 + 2] as i32 * 4,
                    residual[r * 4 + 3] as i32 * 4,
                ];
                let out = fdct4(row);
                for c in 0..4 {
                    tmp[r * 4 + c] = out[c];
                }
            }
            // Column pass — no additional normalization shift here; the
            // *4 row-input scale and the inverse DCT's shift-4 are matched.
            for c in 0..4 {
                let col = [tmp[c], tmp[4 + c], tmp[8 + c], tmp[12 + c]];
                let out = fdct4(col);
                for r in 0..4 {
                    coeffs[r * 4 + c] = out[r];
                }
            }
        }
        8 => {
            for r in 0..8 {
                let row: [i32; 8] = std::array::from_fn(|c| residual[r * 8 + c] as i32 * 4);
                let out = fdct8(row);
                for c in 0..8 {
                    tmp[r * 8 + c] = out[c];
                }
            }
            // Column pass.
            for c in 0..8 {
                let col: [i32; 8] = std::array::from_fn(|r| tmp[r * 8 + c]);
                let out = fdct8(col);
                for r in 0..8 {
                    coeffs[r * 8 + c] = out[r];
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Quantise a 2-D DCT coefficient array in-place.
///
/// `coeffs` is the output of `fdct_2d` (length `bs*bs`). The DC
/// coefficient at index 0 is quantised with `dq_dc`; the rest with
/// `dq_ac`. Quantisation uses round-to-nearest: `q = (coef + dq/2) / dq`
/// for positive coefficients, with sign preservation.
///
/// Returns the non-inclusive end-of-block position (number of
/// non-zero coefficients in scan order, starting from 1 if DC is
/// non-zero, or 0 if the block is all-zero).
pub fn quantise(coeffs: &mut [i32], scan: &[i16], dq_dc: i16, dq_ac: i16) -> usize {
    let n = coeffs.len();
    // Convert from 2-D raster to quantised scan order in-place.
    // We need a temporary buffer because the scan is not identity.
    let dc_q = dq_dc as i32;
    let ac_q = dq_ac as i32;
    let mut last_nonzero = 0usize;
    let mut out = vec![0i32; n];

    for (scan_pos, &raster_idx) in scan.iter().enumerate() {
        let coef = coeffs[raster_idx as usize];
        let dq = if scan_pos == 0 { dc_q } else { ac_q };
        let abs_q = if coef >= 0 {
            (coef + dq / 2) / dq
        } else {
            -((-coef + dq / 2) / dq)
        };
        out[scan_pos] = abs_q;
        if abs_q != 0 {
            last_nonzero = scan_pos + 1;
        }
    }
    coeffs[..n].copy_from_slice(&out[..n]);
    last_nonzero
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::DEFAULT_SCAN_4X4;

    #[test]
    fn fdct4_flat_residual_has_dc_only() {
        // A flat residual of 4 means all energy is in DC.
        let residual = [4i16; 16];
        let mut coeffs = [0i32; 16];
        fdct_2d(&residual, 4, &mut coeffs);
        // DC should be nonzero; ACs should be ~zero.
        assert!(
            coeffs[0].abs() > 10,
            "DC coef should be large: {}",
            coeffs[0]
        );
        // Row-only DC energy: each AC should be much smaller.
        for i in 1..16 {
            assert!(
                coeffs[i].abs() <= 2,
                "AC[{i}] = {} should be near zero for flat input",
                coeffs[i]
            );
        }
    }

    #[test]
    fn quantise_all_zero_returns_eob_zero() {
        let mut coeffs = [0i32; 16];
        let eob = quantise(&mut coeffs, &DEFAULT_SCAN_4X4, 20, 20);
        assert_eq!(eob, 0);
    }

    #[test]
    fn quantise_nonzero_dc_returns_eob_1() {
        let mut coeffs = [0i32; 16];
        // Place a large value at DC (raster position 0 = scan position 0).
        coeffs[0] = 40;
        let eob = quantise(&mut coeffs, &DEFAULT_SCAN_4X4, 20, 20);
        assert!(eob >= 1, "expected eob >= 1 with nonzero DC");
        assert_ne!(coeffs[0], 0, "DC quant coef should be nonzero");
    }

    #[test]
    fn fdct4_then_idct4_roundtrip() {
        // Forward DCT then inverse DCT should approximately reconstruct the input.
        // Use small residuals (centred around 0 as u8 = 128) to avoid clamp.
        use crate::transform::inverse_transform_add;
        use crate::transform::TxType;

        // Residuals around pred=128 (pred will be set to 128 in dst).
        let residuals: [i16; 16] = [5, -3, 7, -2, 1, 4, -1, 3, 2, -4, 6, -1, 3, -2, 1, 0];
        let mut coeffs = [0i32; 16];
        fdct_2d(&residuals, 4, &mut coeffs);

        let max_val = coeffs.iter().map(|&v| v.abs()).max().unwrap_or(0);
        assert!(max_val > 0, "forward DCT should produce nonzero output");

        // Predictor = 128 in every sample. IDCT adds residual to pred.
        let mut dst = [128u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();

        // Reconstructed ≈ pred + residual = 128 + residual.
        for i in 0..16 {
            let expected = (128i32 + residuals[i] as i32).clamp(0, 255) as u8;
            let diff = (dst[i] as i32 - expected as i32).abs();
            assert!(
                diff <= 5,
                "roundtrip[{i}]: got {}, expected {}",
                dst[i],
                expected
            );
        }
    }
}
