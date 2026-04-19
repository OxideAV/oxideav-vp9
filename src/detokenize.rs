//! VP9 coefficient / token decoder — port of libvpx
//! `vp9_detokenize.c::decode_coefs`.
//!
//! Reads up to `max_eob = 16 << (tx_size_log2 << 1)` dequantised residual
//! coefficients into `out` in the order dictated by `scan`. `neighbors`
//! must be the scan's per-position 2-entry context table — entry `2c`
//! and `2c+1` contain the two earlier-scanned positions whose decoded
//! magnitudes drive the context for position `c`.
//!
//! `coef_probs` is the 6-band × 6-context × 3-node table indexed by
//! tx-size / plane-type / ref-type. `dq[0]` is the DC dequant; `dq[1]`
//! is the AC dequant. For 32×32 blocks libvpx halves the coefficient
//! magnitude before dequant (shift right by 1) — `dq_shift` encodes
//! this.

use oxideav_core::Result;

use crate::bool_decoder::BoolDecoder;
use crate::tables::{
    CAT1_PROB, CAT2_PROB, CAT3_PROB, CAT4_PROB, CAT5_PROB, CAT6_PROB, PARETO8_FULL,
};

const EOB_CONTEXT_NODE: usize = 0;
const ZERO_CONTEXT_NODE: usize = 1;
const ONE_CONTEXT_NODE: usize = 2;
const PIVOT_NODE: usize = 2;

const CAT1_MIN_VAL: i32 = 5;
const CAT2_MIN_VAL: i32 = 7;
const CAT3_MIN_VAL: i32 = 11;
const CAT4_MIN_VAL: i32 = 19;
const CAT5_MIN_VAL: i32 = 35;
const CAT6_MIN_VAL: i32 = 67;

/// Flat coefficient-probability table accessor. Each entry is `[u8; 3]`
/// with the first 3 model probabilities (EOB, ZERO, ONE). The PIVOT
/// extends into the Pareto8 extended table at decode time.
///
/// Layout: `coef_probs[band][ctx][node]` — 6 bands × 6 contexts × 3 nodes.
pub type CoefProbs = [[[u8; 3]; 6]; 6];

/// Decode one transform block's coefficients. Returns the end-of-block
/// position (i.e. how many coefficients were decoded). Also clears the
/// output array up to `max_eob` beforehand.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs(
    bd: &mut BoolDecoder<'_>,
    coef_probs: &CoefProbs,
    dq: &[i16; 2],
    scan: &[i16],
    neighbors: &[i16],
    band_translate: &[u8],
    tx_size_log2: usize,
    initial_ctx: usize,
    out: &mut [i32],
) -> Result<usize> {
    let max_eob = 16usize << (tx_size_log2 << 1);
    debug_assert!(out.len() >= max_eob);
    for v in out.iter_mut().take(max_eob) {
        *v = 0;
    }
    let dq_shift = if tx_size_log2 == 3 { 1 } else { 0 };
    let mut dqv = dq[0];
    let mut ctx = initial_ctx;
    let mut token_cache = [0u8; 32 * 32];
    let mut c = 0usize;
    let mut band_idx = 0usize;

    loop {
        if c >= max_eob {
            break;
        }
        let band = band_translate[band_idx] as usize;
        band_idx += 1;
        let probs = coef_probs[band][ctx];
        // EOB bit.
        if bd.read(probs[EOB_CONTEXT_NODE])? == 0 {
            break;
        }
        // ZERO bit: may loop multiple times, decoding zeros and rolling
        // context forward.
        let mut probs = probs;
        loop {
            if bd.read(probs[ZERO_CONTEXT_NODE])? == 1 {
                break;
            }
            dqv = dq[1];
            token_cache[scan[c] as usize] = 0;
            c += 1;
            if c >= max_eob {
                return Ok(c);
            }
            ctx = get_coef_context(neighbors, &token_cache, c);
            let band = band_translate[band_idx] as usize;
            band_idx += 1;
            probs = coef_probs[band][ctx];
            if bd.read(probs[EOB_CONTEXT_NODE])? == 0 {
                return Ok(c);
            }
        }
        // Non-zero coefficient at `c`.
        let magnitude = decode_nonzero(bd, &probs, &mut token_cache, scan, c, dqv, dq_shift)?;
        let signed = if bd.read(128)? == 1 {
            -magnitude
        } else {
            magnitude
        };
        out[scan[c] as usize] = signed;
        c += 1;
        if c >= max_eob {
            break;
        }
        ctx = get_coef_context(neighbors, &token_cache, c);
        dqv = dq[1];
    }
    Ok(c)
}

#[inline]
fn get_coef_context(neighbors: &[i16], token_cache: &[u8], c: usize) -> usize {
    let n0 = neighbors[2 * c] as usize;
    let n1 = neighbors[2 * c + 1] as usize;
    (1 + token_cache[n0] as usize + token_cache[n1] as usize) >> 1
}

/// Decode the non-zero portion of a token (all paths after the outer
/// ZERO_CONTEXT bit returned 1). Updates `token_cache[scan[c]]` and
/// returns the absolute coefficient magnitude post-dequant / post-shift.
fn decode_nonzero(
    bd: &mut BoolDecoder<'_>,
    probs: &[u8; 3],
    token_cache: &mut [u8],
    scan: &[i16],
    c: usize,
    dqv: i16,
    dq_shift: i32,
) -> Result<i32> {
    if bd.read(probs[ONE_CONTEXT_NODE])? == 0 {
        token_cache[scan[c] as usize] = 1;
        return Ok((dqv as i32) >> dq_shift);
    }
    let p = &PARETO8_FULL[probs[PIVOT_NODE] as usize - 1];
    if bd.read(p[0])? == 0 {
        // TWO / THREE / FOUR
        if bd.read(p[1])? == 0 {
            token_cache[scan[c] as usize] = 2;
            return Ok((2 * dqv as i32) >> dq_shift);
        }
        token_cache[scan[c] as usize] = 3;
        let v = (3 + bd.read(p[2])? as i32) * dqv as i32;
        return Ok(v >> dq_shift);
    }
    if bd.read(p[3])? == 0 {
        token_cache[scan[c] as usize] = 4;
        if bd.read(p[4])? == 1 {
            let v = CAT2_MIN_VAL + read_coeff(bd, &CAT2_PROB[..2])?;
            return Ok((v * dqv as i32) >> dq_shift);
        }
        let v = CAT1_MIN_VAL + read_coeff(bd, &CAT1_PROB[..1])?;
        return Ok((v * dqv as i32) >> dq_shift);
    }
    token_cache[scan[c] as usize] = 5;
    if bd.read(p[5])? == 1 {
        if bd.read(p[7])? == 1 {
            let v = CAT6_MIN_VAL + read_coeff(bd, &CAT6_PROB[..14])?;
            return Ok((v * dqv as i32) >> dq_shift);
        }
        let v = CAT5_MIN_VAL + read_coeff(bd, &CAT5_PROB[..5])?;
        return Ok((v * dqv as i32) >> dq_shift);
    }
    if bd.read(p[6])? == 1 {
        let v = CAT4_MIN_VAL + read_coeff(bd, &CAT4_PROB[..4])?;
        return Ok((v * dqv as i32) >> dq_shift);
    }
    let v = CAT3_MIN_VAL + read_coeff(bd, &CAT3_PROB[..3])?;
    Ok((v * dqv as i32) >> dq_shift)
}

fn read_coeff(bd: &mut BoolDecoder<'_>, probs: &[u8]) -> Result<i32> {
    let mut val = 0i32;
    for &p in probs {
        val = (val << 1) | bd.read(p)? as i32;
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::COEFBAND_TRANS_4X4;

    #[test]
    fn early_eob_returns_zero() {
        // A payload whose bool decoder reads 0 for EOB_CONTEXT_NODE at the
        // first call immediately terminates with eob=0.
        let payload = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut bd = BoolDecoder::new(&payload).unwrap();
        let mut probs: CoefProbs = [[[0u8; 3]; 6]; 6];
        for band in probs.iter_mut() {
            for ctx in band.iter_mut() {
                *ctx = [1, 0, 0];
            }
        }
        let dq = [8i16, 8];
        let scan: [i16; 16] = [0, 4, 1, 5, 8, 2, 12, 9, 3, 6, 13, 10, 7, 14, 11, 15];
        let neighbors = [0i16; 34];
        let mut out = [0i32; 16];
        let eob = decode_coefs(
            &mut bd,
            &probs,
            &dq,
            &scan,
            &neighbors,
            &COEFBAND_TRANS_4X4,
            0,
            0,
            &mut out,
        )
        .unwrap();
        assert_eq!(eob, 0);
        for &v in &out {
            assert_eq!(v, 0);
        }
    }
}
