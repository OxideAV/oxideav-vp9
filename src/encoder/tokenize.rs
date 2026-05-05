//! VP9 coefficient token encoder — forward mirror of `detokenize.rs`.
//!
//! Encodes a quantised coefficient array in scan order into the
//! boolean-coded token stream (EOB, ZERO, ONE, TWO/THREE/FOUR/CAT1..6,
//! sign bit). The probability tables and context logic are the exact
//! inverse of `decode_coefs` in `detokenize.rs`.

use crate::detokenize::CoefProbs;
use crate::encoder::bool_encoder::BoolEncoder;
use crate::tables::{
    CAT1_PROB, CAT2_PROB, CAT3_PROB, CAT4_PROB, CAT5_PROB, CAT6_PROB, PARETO8_FULL,
};

const EOB_CONTEXT_NODE: usize = 0;
const ZERO_CONTEXT_NODE: usize = 1;
const ONE_CONTEXT_NODE: usize = 2;
const PIVOT_NODE: usize = 2;

// These must match the minimum values used by the decoder.
const CAT1_MIN_VAL: i32 = 5;
const CAT2_MIN_VAL: i32 = 7;
const CAT3_MIN_VAL: i32 = 11;
const CAT4_MIN_VAL: i32 = 19;
const CAT5_MIN_VAL: i32 = 35;
const CAT6_MIN_VAL: i32 = 67;

/// Encode one transform block's quantised coefficients (scan-order).
///
/// `coeffs` is in scan order, length = max_eob (16/64/256/1024 for
/// 4×4/8×8/16×16/32×32). `eob` is the count of non-zero positions.
/// `initial_ctx` is the nonzero-context derived from the above/left
/// NonzeroContext arrays at this block's position (0, 1, or 2),
/// matching the decoder's `token_ctx` return value.
///
/// This exactly mirrors `decode_coefs` in `detokenize.rs`.
#[allow(clippy::too_many_arguments)]
pub fn encode_coefs(
    be: &mut BoolEncoder,
    coef_probs: &CoefProbs,
    scan: &[i16],
    neighbors: &[i16],
    band_translate: &[u8],
    coeffs: &[i32],     // scan-order quantised coefficients
    eob: usize,         // number of coefficients to encode (last non-zero + 1)
    initial_ctx: usize, // nonzero context (0/1/2) from above+left flags
) {
    let max_eob = coeffs.len();
    // Track the decoder-side check_eob state: starts at true, becomes
    // false after a ZERO token, returns to true after any non-zero token.
    let mut check_eob = true;
    let mut token_cache = vec![0u8; 32 * 32];
    let mut ctx = initial_ctx.min(5);
    let mut c = 0usize;

    while c < max_eob {
        let band = band_translate[c] as usize;
        let probs = coef_probs[band][ctx];

        if check_eob {
            if c >= eob {
                // The decoder would read EOB = 0 here → break.
                be.write(0, probs[EOB_CONTEXT_NODE]);
                return;
            }
            // More coefficients follow — emit EOB = 1.
            be.write(1, probs[EOB_CONTEXT_NODE]);
        }

        let qcoef = coeffs[c];

        if qcoef == 0 {
            // ZERO token: emit 0 on ZERO_CONTEXT_NODE.
            be.write(0, probs[ZERO_CONTEXT_NODE]);
            token_cache[scan[c] as usize] = 0;
            c += 1;
            if c < max_eob {
                ctx = get_coef_context(neighbors, &token_cache, c);
            }
            // After ZERO the decoder sets check_eob = false.
            check_eob = false;
            continue;
        }

        // Non-zero: emit ZERO_CONTEXT_NODE = 1.
        be.write(1, probs[ZERO_CONTEXT_NODE]);
        // Now emit the magnitude and update token_cache.
        encode_nonzero(be, &probs, qcoef, scan, &mut token_cache, c);
        // Sign bit.
        be.write((qcoef < 0) as u32, 128);

        c += 1;
        if c < max_eob {
            ctx = get_coef_context(neighbors, &token_cache, c);
        }
        // After non-zero, decoder sets check_eob = true.
        check_eob = true;
    }
}

/// Encode the magnitude of a non-zero quantised coefficient.
///
/// Mirrors `decode_nonzero` exactly: reads `probs[ONE_CONTEXT_NODE]` to
/// choose ONE vs TWO+, then the Pareto8-extended tree for TWO..CAT6.
fn encode_nonzero(
    be: &mut BoolEncoder,
    probs: &[u8; 3],
    qcoef: i32,
    scan: &[i16],
    token_cache: &mut [u8],
    c: usize,
) {
    let abs_v = qcoef.unsigned_abs() as i32;

    if abs_v == 1 {
        // ONE token: emit 0 on ONE_CONTEXT_NODE.
        be.write(0, probs[ONE_CONTEXT_NODE]);
        token_cache[scan[c] as usize] = 1;
        return;
    }

    // TWO+: emit 1 on ONE_CONTEXT_NODE.
    be.write(1, probs[ONE_CONTEXT_NODE]);

    // Now use the Pareto8 table keyed on probs[PIVOT_NODE].
    let p_idx = (probs[PIVOT_NODE] as usize)
        .saturating_sub(1)
        .min(PARETO8_FULL.len() - 1);
    let p = &PARETO8_FULL[p_idx];

    // The decoder's tree (from decode_nonzero):
    //   p[0] == 0 → TWO / THREE / FOUR:
    //       p[1] == 0 → TWO
    //       p[1] == 1 → THREE / FOUR:
    //           p[2] == 0 → THREE
    //           p[2] == 1 → FOUR
    //   p[0] == 1 → CAT1..6:
    //       p[3] == 0 → CAT1 / CAT2:
    //           p[4] == 1 → CAT2 (7..10), extra 2 bits from CAT2_PROB
    //           p[4] == 0 → CAT1 (5..6), extra 1 bit from CAT1_PROB
    //       p[3] == 1 → CAT3..6:
    //           p[5] == 1 → CAT5 / CAT6:
    //               p[7] == 1 → CAT6 (67..2048), extra 14 bits from CAT6_PROB
    //               p[7] == 0 → CAT5 (35..66), extra 5 bits from CAT5_PROB
    //           p[5] == 0 → CAT3 / CAT4:
    //               p[6] == 1 → CAT4 (19..34), extra 4 bits from CAT4_PROB
    //               p[6] == 0 → CAT3 (11..18), extra 3 bits from CAT3_PROB

    if abs_v <= 4 {
        // TWO / THREE / FOUR branch: p[0] = 0.
        be.write(0, p[0]);
        if abs_v == 2 {
            be.write(0, p[1]);
            token_cache[scan[c] as usize] = 2;
        } else if abs_v == 3 {
            be.write(1, p[1]);
            be.write(0, p[2]);
            token_cache[scan[c] as usize] = 3;
        } else {
            // abs_v == 4
            be.write(1, p[1]);
            be.write(1, p[2]);
            token_cache[scan[c] as usize] = 3;
        }
        return;
    }

    // CAT1..6 branch: p[0] = 1.
    be.write(1, p[0]);
    token_cache[scan[c] as usize] = 5; // max token cache value for large coefs

    if abs_v < CAT3_MIN_VAL {
        // CAT1 or CAT2: p[3] = 0.
        be.write(0, p[3]);
        if abs_v < CAT2_MIN_VAL {
            // CAT1 (5..6): p[4] = 0, 1 extra bit.
            be.write(0, p[4]);
            write_extra_bits(be, abs_v - CAT1_MIN_VAL, &CAT1_PROB[..1]);
        } else {
            // CAT2 (7..10): p[4] = 1, 2 extra bits.
            be.write(1, p[4]);
            write_extra_bits(be, abs_v - CAT2_MIN_VAL, &CAT2_PROB[..2]);
        }
        return;
    }

    // CAT3..6: p[3] = 1.
    be.write(1, p[3]);

    if abs_v >= CAT5_MIN_VAL {
        // CAT5 or CAT6: p[5] = 1.
        be.write(1, p[5]);
        if abs_v >= CAT6_MIN_VAL {
            // CAT6 (67..): p[7] = 1, 14 extra bits.
            be.write(1, p[7]);
            write_extra_bits(be, abs_v - CAT6_MIN_VAL, &CAT6_PROB[..14]);
        } else {
            // CAT5 (35..66): p[7] = 0, 5 extra bits.
            be.write(0, p[7]);
            write_extra_bits(be, abs_v - CAT5_MIN_VAL, &CAT5_PROB[..5]);
        }
        return;
    }

    // CAT3 or CAT4: p[5] = 0.
    be.write(0, p[5]);
    if abs_v >= CAT4_MIN_VAL {
        // CAT4 (19..34): p[6] = 1, 4 extra bits.
        be.write(1, p[6]);
        write_extra_bits(be, abs_v - CAT4_MIN_VAL, &CAT4_PROB[..4]);
    } else {
        // CAT3 (11..18): p[6] = 0, 3 extra bits.
        be.write(0, p[6]);
        write_extra_bits(be, abs_v - CAT3_MIN_VAL, &CAT3_PROB[..3]);
    }
}

/// Write `n_bits` extra bits MSB-first using the provided probability table.
/// Mirrors `read_coeff` in detokenize.rs which does `val = (val<<1) | read(p[i])`.
fn write_extra_bits(be: &mut BoolEncoder, value: i32, probs: &[u8]) {
    let n = probs.len();
    for i in 0..n {
        let bit = ((value >> (n - 1 - i)) & 1) as u32;
        be.write(bit, probs[i]);
    }
}

/// Context lookup — exact mirror of `get_coef_context` in detokenize.rs.
fn get_coef_context(neighbors: &[i16], token_cache: &[u8], c: usize) -> usize {
    let n0 = neighbors[2 * c] as usize;
    let n1 = neighbors[2 * c + 1] as usize;
    (1 + token_cache[n0] as usize + token_cache[n1] as usize) >> 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;
    use crate::detokenize::decode_coefs;
    use crate::tables::{
        COEFBAND_TRANS_4X4, COEF_PROBS_4X4, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS,
    };

    fn default_coef_probs_4x4() -> CoefProbs {
        COEF_PROBS_4X4[0][0] // intra, Y-plane
    }

    /// Encode then decode a coefficient array, verify round-trip.
    fn roundtrip(coeffs_scan: [i32; 16], eob: usize) -> Vec<i32> {
        let probs = default_coef_probs_4x4();
        let mut be = BoolEncoder::new();
        encode_coefs(
            &mut be,
            &probs,
            &DEFAULT_SCAN_4X4,
            &DEFAULT_SCAN_4X4_NEIGHBORS,
            &COEFBAND_TRANS_4X4,
            &coeffs_scan,
            eob,
            0, // initial_ctx = 0 for unit test
        );
        let buf = be.finish();

        // Decode back — note: decode_coefs produces dequantised values.
        // To get the quantised values back we use dq=[1,1] so dequant=identity.
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let dq = [1i16, 1i16]; // identity dequant
        let mut out = [0i32; 16];
        let got_eob = decode_coefs(
            &mut bd,
            &probs,
            &dq,
            &DEFAULT_SCAN_4X4,
            &DEFAULT_SCAN_4X4_NEIGHBORS,
            &COEFBAND_TRANS_4X4,
            0, // tx_size_log2
            0, // initial_ctx
            &mut out,
        )
        .unwrap();
        assert_eq!(got_eob, eob, "eob mismatch");
        out.to_vec()
    }

    #[test]
    fn all_zero_roundtrip() {
        let coeffs = [0i32; 16];
        let decoded = roundtrip(coeffs, 0);
        for &v in &decoded {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn single_dc_roundtrip() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 3; // positive value > 1 at DC scan position
        let decoded = roundtrip(coeffs, 1);
        assert_eq!(decoded[0], 3, "DC should round-trip");
    }

    #[test]
    fn one_token_roundtrip() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let decoded = roundtrip(coeffs, 1);
        assert_eq!(decoded[0], 1);
    }

    #[test]
    fn negative_dc_roundtrip() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = -5;
        let decoded = roundtrip(coeffs, 1);
        assert_eq!(decoded[0], -5);
    }

    #[test]
    fn two_nonzero_roundtrip() {
        // In scan order: position 0 → raster index DEFAULT_SCAN_4X4[0]=0
        //                position 1 → raster index DEFAULT_SCAN_4X4[1]=4
        let mut coeffs = [0i32; 16];
        coeffs[0] = 4; // scan position 0 → raster 0 (DC)
        coeffs[1] = -2; // scan position 1 → raster 4 (first AC)
        let decoded = roundtrip(coeffs, 2);
        // Decoder writes out[scan[c]] = value, so check raster positions.
        assert_eq!(decoded[DEFAULT_SCAN_4X4[0] as usize], 4, "DC");
        assert_eq!(decoded[DEFAULT_SCAN_4X4[1] as usize], -2, "first AC");
    }

    #[test]
    fn large_cat3_roundtrip() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 15; // CAT3 range (11..18)
        let decoded = roundtrip(coeffs, 1);
        assert_eq!(decoded[0], 15);
    }

    #[test]
    fn large_cat6_roundtrip() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 100; // CAT6 range (67+)
        let decoded = roundtrip(coeffs, 1);
        assert_eq!(decoded[0], 100);
    }
}
