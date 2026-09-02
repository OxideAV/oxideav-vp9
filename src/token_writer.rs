//! VP9 §6.4.24 coefficient-**token writer** — the inverse of the
//! `tokens` decode primitives (`read_more_coefs` / `read_token` /
//! `read_coef` / `read_coef_token`).
//!
//! The residual encoder turns a quantized transform block into the
//! §6.4.24 token stream: per scan position a `more_coefs` bit, then a
//! `token` tree walk, then the token's extra magnitude bits and a sign.
//! This module provides the per-symbol writers that drive a
//! [`BoolEncoder`], each one the exact inverse of the matching decode
//! step so a written stream decodes back to the same coefficients.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.24 / §6.4.26 / §9.3.3; the bit
//! order mirrors the in-crate decoder exactly.

use crate::bool_encoder::BoolEncoder;
use crate::tokens::{CAT_PROBS, DCT_VAL_CATEGORY6, EXTRA_BITS, ONE_TOKEN, TOKEN_TREE, ZERO_TOKEN};
use crate::Error;

/// Write one `more_coefs` bit (§6.4.24): `B(prob)` with value 1 when more
/// coefficients follow, 0 at end-of-block.
pub(crate) fn write_more_coefs(enc: &mut BoolEncoder, more: bool, prob: u8) {
    enc.write_bool(u32::from(more), u32::from(prob));
}

/// Resolve the §9.3.3 `TOKEN_TREE` bit path that decodes to `token`.
///
/// The decoder walks `n = TOKEN_TREE[ n + bit ]` from `n = 0`, probing
/// `probs[ n >> 1 ]` at each step, until `n <= 0` yields the leaf token
/// `-n`. To *encode* we reproduce that walk: at each internal node we
/// pick the child (bit 0 or 1) whose subtree contains the target leaf.
/// Returns the `(node_index, bit)` pairs in order, where `node_index`
/// is the `probs` index (`n >> 1`) used at that step.
fn token_bit_path(token: u32) -> Result<Vec<(usize, u32)>, Error> {
    let target = -(token as i32);
    let mut path = Vec::new();
    let mut n: i32 = 0;
    // Bounded by the tree depth (max 9 internal nodes).
    for _ in 0..TOKEN_TREE.len() {
        let node = (n >> 1) as usize;
        // Try bit 0 then bit 1; pick the one whose subtree reaches the
        // target leaf.
        let mut chosen = None;
        for bit in 0u32..2 {
            let idx = (n as usize) + bit as usize;
            if subtree_contains(TOKEN_TREE[idx], target) {
                chosen = Some((bit, TOKEN_TREE[idx]));
                break;
            }
        }
        let (bit, next) = chosen.ok_or(Error::Unsupported)?;
        path.push((node, bit));
        if next <= 0 {
            if next == target {
                return Ok(path);
            }
            return Err(Error::Unsupported);
        }
        n = next;
    }
    Err(Error::Unsupported)
}

/// Does the `TOKEN_TREE` subtree rooted at `node` (a tree value: leaf
/// when `<= 0`, internal child-pair base when `> 0`) contain `target`
/// (a negated leaf token)?
fn subtree_contains(node: i32, target: i32) -> bool {
    if node <= 0 {
        return node == target;
    }
    let base = node as usize;
    subtree_contains(TOKEN_TREE[base], target) || subtree_contains(TOKEN_TREE[base + 1], target)
}

/// Write the §9.3.3 `token` tree (inverse of `read_token`). `probs` is
/// the same pre-computed 10-node probability array the decoder consumes.
pub(crate) fn write_token(
    enc: &mut BoolEncoder,
    token: u32,
    probs: &[u8; 10],
) -> Result<(), Error> {
    for (node, bit) in token_bit_path(token)? {
        enc.write_bool(bit, u32::from(probs[node]));
    }
    Ok(())
}

/// Write the §6.4.26 `read_coef` extra magnitude bits (inverse of
/// `read_coef`). `mag` is the unsigned coefficient magnitude (`>= 1`);
/// `token` must be its matching `EXTRA_BITS` category.
pub(crate) fn write_coef(
    enc: &mut BoolEncoder,
    token: u32,
    mag: u32,
    bit_depth: u32,
) -> Result<(), Error> {
    debug_assert!((ONE_TOKEN..=DCT_VAL_CATEGORY6).contains(&token));
    debug_assert!(bit_depth == 8 || bit_depth == 10 || bit_depth == 12);
    let row = &EXTRA_BITS[token as usize];
    let cat = row[0] as usize;
    let num_extra = row[1];
    let base = row[2];
    if mag < base {
        return Err(Error::Unsupported);
    }
    let mut residual = mag - base;

    // §6.4.26 high-bit loop for CAT6 at >8-bit depth: bits added at
    // shift `5 + BitDepth - e` precede the standard cat-prob bits.
    if token == DCT_VAL_CATEGORY6 && bit_depth > 8 {
        for e in 0..(bit_depth - 8) {
            let shift = 5 + bit_depth - e;
            let high_bit = (residual >> shift) & 1;
            enc.write_bool(high_bit, 255);
        }
        // Clear the high bits we just emitted so the standard loop below
        // only sees the low `num_extra` bits.
        let high_mask: u32 = {
            let mut m = 0u32;
            for e in 0..(bit_depth - 8) {
                m |= 1 << (5 + bit_depth - e);
            }
            m
        };
        residual &= !high_mask;
    }

    let cat_row = &CAT_PROBS[cat];
    for e in 0..num_extra {
        let p = cat_row[e as usize];
        let bit = (residual >> (num_extra - 1 - e)) & 1;
        enc.write_bool(bit, u32::from(p));
    }
    Ok(())
}

/// Write one §6.4.24 coefficient step (inverse of `read_coef_token`).
///
/// * If `check_eob`, write the `more_coefs` flag first; on a `None`
///   value (end-of-block) write `more_coefs == 0` and return.
/// * Otherwise write `more_coefs == 1`, then the `token` and (for
///   non-zero tokens) the magnitude bits + sign.
///
/// `value` is the signed coefficient at this scan position; `None`
/// signals end-of-block (only valid when `check_eob`).
pub(crate) fn write_coef_token(
    enc: &mut BoolEncoder,
    check_eob: bool,
    value: Option<i32>,
    more_coefs_prob: u8,
    token_probs: &[u8; 10],
    bit_depth: u32,
) -> Result<(), Error> {
    let v = match value {
        None => {
            debug_assert!(check_eob, "end-of-block only valid when check_eob");
            write_more_coefs(enc, false, more_coefs_prob);
            return Ok(());
        }
        Some(v) => v,
    };
    if check_eob {
        write_more_coefs(enc, true, more_coefs_prob);
    }
    if v == 0 {
        write_token(enc, ZERO_TOKEN, token_probs)?;
        return Ok(());
    }
    let mag = v.unsigned_abs();
    let token = token_for_magnitude(mag);
    write_token(enc, token, token_probs)?;
    write_coef(enc, token, mag, bit_depth)?;
    // sign L(1): 1 = negative.
    enc.write_literal(u32::from(v < 0), 1);
    Ok(())
}

/// Map an unsigned coefficient magnitude (`>= 1`) to its §6.4.26 token.
/// The token is the highest `EXTRA_BITS` category whose base value does
/// not exceed `mag`.
pub(crate) fn token_for_magnitude(mag: u32) -> u32 {
    debug_assert!(mag >= 1);
    // ONE/TWO/THREE/FOUR are exact; the CAT tokens cover ranges by base.
    match mag {
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5..=6 => 5,   // CAT1: base 5, 1 extra bit
        7..=10 => 6,  // CAT2: base 7, 2 extra bits
        11..=18 => 7, // CAT3: base 11, 3 extra bits
        19..=34 => 8, // CAT4: base 19, 4 extra bits
        35..=66 => 9, // CAT5: base 35, 5 extra bits
        _ => 10,      // CAT6: base 67, 14 (+highbit) extra bits
    }
}

/// Write one §6.4.24 transform block's coefficient syntax — the exact
/// inverse of [`crate::tokens::tokens`].
///
/// `coeffs` is the quantized transform block in **raster** order (length
/// `segEob = 16 << (txSz << 1)`); the writer walks the §6.4.25 `scan`
/// order, deriving the per-coefficient `ctx` from the §9.3.2 rule —
/// exactly as the decoder does — using only coefficients written at
/// strictly-earlier scan positions, so the encode/decode contexts agree
/// bit-for-bit:
///
/// * DC (`c == 0`): `above + left`, the OR-reduced non-zero context over
///   the block's `1 << txSz` 4-sample columns / rows in `nz`.
/// * AC (`c > 0`): `(1 + cache[nb0] + cache[nb1]) >> 1`, where the
///   neighbours come from [`crate::tokens::token_cache_neighbours`] and
///   `cache` holds the [`ENERGY_CLASS`] of each already-written token.
///
/// The end-of-block is the scan index one past the last non-zero
/// coefficient; the writer emits `more_coefs == 1` up to it (skipping the
/// `more_coefs` flag immediately after a `ZERO_TOKEN`, matching the
/// decoder's `check_eob` latch) and a closing `more_coefs == 0` unless
/// the block fills all `segEob` positions.
///
/// Returns the block's non-zero flag (`true` if any token — including an
/// explicit `ZERO_TOKEN` — was coded), the same value the decoder's
/// `tokens( )` returns for the §6.4.21 `nz` write-back.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn write_tokens(
    enc: &mut BoolEncoder,
    block: &crate::tokens::TokenBlockCtx,
    tx_sz: u32,
    scan: &[u16],
    coef_probs: &crate::coef_probs::CoefProbs,
    nz: &crate::tokens::NonzeroContext,
    coeffs: &[i64],
    token_cache: &mut [u8],
    counts: &mut crate::prob_adapt::FrameCounts,
) -> Result<bool, Error> {
    use crate::tokens::{
        build_token_probs, coef_band, token_cache_neighbours, ENERGY_CLASS, ZERO_TOKEN,
    };

    let seg_eob = 16usize << (tx_sz << 1);
    debug_assert_eq!(scan.len(), seg_eob);
    debug_assert_eq!(coeffs.len(), seg_eob);
    debug_assert!(token_cache.len() >= seg_eob);

    let plane_gt0 = usize::from(block.plane > 0);
    let is_inter = usize::from(block.is_inter);
    let tx_slab = &coef_probs[tx_sz as usize][plane_gt0][is_inter];

    // End-of-block: one past the last non-zero scan position. (If every
    // position is zero, `eob == 0` and only the DC `more_coefs == 0` is
    // written.)
    let mut eob = 0usize;
    for (c, &p) in scan.iter().enumerate() {
        if coeffs[p as usize] != 0 {
            eob = c + 1;
        }
    }

    let mut check_eob = true;
    for (c, &scan_pos) in scan.iter().enumerate().take(eob) {
        let pos = scan_pos as usize;
        let band = coef_band(c, tx_sz);

        // §9.3.2 ctx derivation — identical to the decoder, using only
        // already-emitted coefficients (DC reads `nz`; AC reads the
        // running `token_cache`).
        let ctx = if c == 0 {
            let numpts = 1usize << tx_sz;
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..numpts {
                if block.x4 + i < block.max_x {
                    above |= nz.above[block.x4 + i];
                }
                if block.y4 + i < block.max_y {
                    left |= nz.left[block.y4 + i];
                }
            }
            (above + left) as usize
        } else {
            let (nb0, nb1) = token_cache_neighbours(c, pos, tx_sz, block.tx_type);
            (1 + token_cache[nb0] as usize + token_cache[nb1] as usize) >> 1
        };

        let cell = &tx_slab[band][ctx];

        // §6.4.24: the `more_coefs` flag is read (and so written) only
        // when `check_eob`; after a `ZERO_TOKEN` it is suppressed. The
        // §9.3.4 `more_coefs` count fires only where the flag is
        // actually coded (the errata #249 reconstruction).
        if check_eob {
            counts.more_coefs[tx_sz as usize][plane_gt0][is_inter][band][ctx][1] += 1;
            write_more_coefs(enc, true, cell[0]);
        }

        let value = coeffs[pos];
        let token_probs = build_token_probs(cell);
        if value == 0 {
            counts.token[tx_sz as usize][plane_gt0][is_inter][band][ctx][0] += 1;
            write_token(enc, ZERO_TOKEN, &token_probs)?;
            token_cache[pos] = ENERGY_CLASS[ZERO_TOKEN as usize];
            check_eob = false;
        } else {
            let mag = value.unsigned_abs() as u32;
            let token = token_for_magnitude(mag);
            counts.token[tx_sz as usize][plane_gt0][is_inter][band][ctx]
                [(token as usize).min(2)] += 1;
            write_token(enc, token, &token_probs)?;
            write_coef(enc, token, mag, block.bit_depth)?;
            enc.write_literal(u32::from(value < 0), 1);
            token_cache[pos] = ENERGY_CLASS[token as usize];
            check_eob = true;
        }
    }

    // Closing end-of-block flag, unless the block ran to `segEob`. The
    // flag is suppressed if the last coded token was a `ZERO_TOKEN`
    // (`check_eob == false`) — the decoder would not read it there.
    if eob < seg_eob && check_eob {
        // ctx / band for the terminating position `eob`.
        let pos = scan[eob] as usize;
        let band = coef_band(eob, tx_sz);
        let ctx = if eob == 0 {
            let numpts = 1usize << tx_sz;
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..numpts {
                if block.x4 + i < block.max_x {
                    above |= nz.above[block.x4 + i];
                }
                if block.y4 + i < block.max_y {
                    left |= nz.left[block.y4 + i];
                }
            }
            (above + left) as usize
        } else {
            let (nb0, nb1) = token_cache_neighbours(eob, pos, tx_sz, block.tx_type);
            (1 + token_cache[nb0] as usize + token_cache[nb1] as usize) >> 1
        };
        counts.more_coefs[tx_sz as usize][plane_gt0][is_inter][band][ctx][0] += 1;
        write_more_coefs(enc, false, tx_slab[band][ctx][0]);
    }

    Ok(eob > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::tokens::{read_coef_token, CoefStep};

    /// A representative 10-node token-probability array (mid-range so the
    /// arithmetic actually fires); the exact values only need to match
    /// between encode and decode.
    const PROBS: [u8; 10] = [128, 100, 64, 160, 200, 90, 110, 140, 170, 50];

    /// Round-trip a coefficient sequence (each entry a signed value;
    /// terminated by an implicit end-of-block) through the token writer
    /// and back through `read_coef_token`.
    fn roundtrip_block(values: &[i32], bit_depth: u32) {
        let mut enc = BoolEncoder::new();
        // check_eob is true on the first position and after every
        // non-zero token; false immediately after a ZERO token.
        let mut check_eob = true;
        for &v in values {
            write_coef_token(&mut enc, check_eob, Some(v), 200, &PROBS, bit_depth).unwrap();
            check_eob = v != 0;
        }
        // End-of-block: only legal when check_eob.
        if check_eob {
            write_coef_token(&mut enc, true, None, 200, &PROBS, bit_depth).unwrap();
        } else {
            // A trailing ZERO with check_eob == false cannot terminate;
            // append a forced EOB by toggling: write a dummy nonzero then
            // EOB is not needed for the test corpus (all end on nonzero
            // or are length-bounded). Guard with an explicit nonzero tail.
            unreachable!("test sequences end on a nonzero coefficient");
        }
        let buf = enc.finish();

        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        let mut check = true;
        for &v in values {
            let step = read_coef_token(&mut dec, check, 200, &PROBS, bit_depth).unwrap();
            match step {
                CoefStep::Coef { value, .. } => assert_eq!(value, v, "coef value"),
                CoefStep::Eob => panic!("unexpected EOB before sequence end"),
            }
            check = v != 0;
        }
        // Final EOB.
        let last = read_coef_token(&mut dec, true, 200, &PROBS, bit_depth).unwrap();
        assert_eq!(last, CoefStep::Eob, "final EOB");
    }

    #[test]
    fn token_path_resolves_every_leaf() {
        for token in 0..=10u32 {
            let path = token_bit_path(token).expect("path exists");
            assert!(!path.is_empty(), "token {token} has a path");
        }
    }

    #[test]
    fn token_for_magnitude_category_boundaries() {
        assert_eq!(token_for_magnitude(1), 1);
        assert_eq!(token_for_magnitude(4), 4);
        assert_eq!(token_for_magnitude(5), 5);
        assert_eq!(token_for_magnitude(6), 5);
        assert_eq!(token_for_magnitude(7), 6);
        assert_eq!(token_for_magnitude(10), 6);
        assert_eq!(token_for_magnitude(11), 7);
        assert_eq!(token_for_magnitude(18), 7);
        assert_eq!(token_for_magnitude(19), 8);
        assert_eq!(token_for_magnitude(34), 8);
        assert_eq!(token_for_magnitude(35), 9);
        assert_eq!(token_for_magnitude(66), 9);
        assert_eq!(token_for_magnitude(67), 10);
        assert_eq!(token_for_magnitude(1000), 10);
    }

    #[test]
    fn roundtrip_small_coefficients() {
        roundtrip_block(&[1, -1, 2, -2, 3, -3, 4, -4], 8);
    }

    #[test]
    fn roundtrip_with_zeros() {
        roundtrip_block(&[0, 0, 1, 0, -2, 0, 0, 3], 8);
    }

    #[test]
    fn roundtrip_category_coefficients_8bit() {
        roundtrip_block(&[5, -6, 7, -10, 11, -18, 19, -34, 35, -66, 67, -200], 8);
    }

    #[test]
    fn roundtrip_cat6_highbits_10bit() {
        roundtrip_block(&[67, -500, 1023, -300], 10);
    }

    #[test]
    fn roundtrip_cat6_highbits_12bit() {
        roundtrip_block(&[67, -2000, 4095, -100], 12);
    }

    #[test]
    fn roundtrip_single_dc_coefficient() {
        roundtrip_block(&[7], 8);
    }

    #[test]
    fn roundtrip_dense_random_block() {
        // Deterministic pseudo-random signed coefficients ending on a
        // nonzero value.
        let mut vals = Vec::new();
        let mut s: u32 = 0x1234_5678;
        for _ in 0..63 {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let mag = ((s >> 16) % 70) as i32; // 0..69
            let sign = if (s >> 8) & 1 == 1 { -1 } else { 1 };
            vals.push(mag * sign);
        }
        vals.push(42); // ensure nonzero tail
        roundtrip_block(&vals, 8);
    }

    // ----- block-level write_tokens round-trips vs tokens() decode -----

    use crate::coef_probs::DEFAULT_COEF_PROBS;
    use crate::idct::{ADST_ADST, DCT_DCT};
    use crate::scan::{DEFAULT_SCAN_16X16, DEFAULT_SCAN_4X4, DEFAULT_SCAN_8X8};
    use crate::tokens::{tokens, NonzeroContext, TokenBlockCtx};

    /// Encode `coeffs` (raster order, length `segEob`) with [`write_tokens`]
    /// then decode it back with the real [`tokens`] driver and assert the
    /// recovered block + non-zero flag match. `nz` carries the (caller-set)
    /// DC non-zero context both sides share.
    fn roundtrip_tokens(
        coeffs: &[i64],
        tx_sz: u32,
        scan: &[u16],
        tx_type: u8,
        bit_depth: u32,
        nz: &NonzeroContext,
    ) {
        let seg_eob = 16usize << (tx_sz << 1);
        assert_eq!(coeffs.len(), seg_eob);

        let block = TokenBlockCtx {
            plane: 0,
            is_inter: false,
            tx_type,
            bit_depth,
            x4: 0,
            y4: 0,
            max_x: nz.above.len(),
            max_y: nz.left.len(),
        };

        let mut enc = BoolEncoder::new();
        let mut wcache = vec![0u8; seg_eob];
        let wnz = write_tokens(
            &mut enc,
            &block,
            tx_sz,
            scan,
            &DEFAULT_COEF_PROBS,
            nz,
            coeffs,
            &mut wcache,
            &mut crate::prob_adapt::FrameCounts::new_boxed(),
        )
        .expect("write tokens");
        let buf = enc.finish();

        let mut dec = BoolCoder::init_bool(&buf, buf.len()).expect("init_bool");
        let mut dcache = vec![0u8; seg_eob];
        let mut out = vec![0i64; seg_eob];
        let mut counts = crate::prob_adapt::FrameCounts::new_boxed();
        let dnz = tokens(
            &mut dec,
            &block,
            tx_sz,
            scan,
            &DEFAULT_COEF_PROBS,
            nz,
            &mut dcache,
            &mut out,
            &mut counts,
        )
        .expect("decode tokens");

        assert_eq!(wnz, dnz, "non-zero flag");
        assert_eq!(out, coeffs, "decoded block != written block");
    }

    fn fresh_nz(span: usize) -> NonzeroContext {
        NonzeroContext::new(span, span)
    }

    #[test]
    fn tokens_4x4_dc_only() {
        let mut block = vec![0i64; 16];
        block[0] = 5;
        roundtrip_tokens(&block, 0, &DEFAULT_SCAN_4X4, DCT_DCT, 8, &fresh_nz(8));
    }

    #[test]
    fn tokens_4x4_all_zero() {
        let block = vec![0i64; 16];
        roundtrip_tokens(&block, 0, &DEFAULT_SCAN_4X4, DCT_DCT, 8, &fresh_nz(8));
    }

    #[test]
    fn tokens_4x4_interior_zeros() {
        // Non-zero DC and a later AC with a run of zeros (ZERO_TOKEN) in
        // between — exercises the `check_eob` suppression after a zero.
        let mut block = vec![0i64; 16];
        block[0] = 3;
        block[1] = 0;
        block[4] = -2;
        block[5] = 0;
        block[8] = 1;
        roundtrip_tokens(&block, 0, &DEFAULT_SCAN_4X4, DCT_DCT, 8, &fresh_nz(8));
    }

    #[test]
    fn tokens_4x4_full_block_no_terminator() {
        // Every position non-zero => eob == segEob, no closing more_coefs.
        let block: Vec<i64> = (0..16).map(|i| (i as i64 % 5) + 1).collect();
        roundtrip_tokens(&block, 0, &DEFAULT_SCAN_4X4, DCT_DCT, 8, &fresh_nz(8));
    }

    #[test]
    fn tokens_4x4_with_dc_context_set() {
        // DC ctx != 0: pre-set the above/left non-zero bits the DC ctx
        // derivation OR-reduces.
        let mut nz = fresh_nz(8);
        nz.above[0] = 1;
        nz.left[0] = 1;
        let mut block = vec![0i64; 16];
        block[0] = 9;
        block[3] = -4;
        roundtrip_tokens(&block, 0, &DEFAULT_SCAN_4X4, DCT_DCT, 8, &nz);
    }

    #[test]
    fn tokens_4x4_adst_neighbour_selection() {
        // ADST_ADST uses the (above, left) neighbour pair, distinct from
        // the DCT_ADST / ADST_DCT single-edge picks.
        let mut block = vec![0i64; 16];
        block[0] = 2;
        block[1] = 3;
        block[4] = -1;
        block[5] = 7;
        roundtrip_tokens(&block, 0, &DEFAULT_SCAN_4X4, ADST_ADST, 8, &fresh_nz(8));
    }

    #[test]
    fn tokens_8x8_dense_random() {
        let mut block = vec![0i64; 64];
        let mut s: u32 = 0x9e37_79b9;
        for slot in block.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            // ~half zero, the rest small signed magnitudes.
            if (s >> 20) & 1 == 1 {
                let mag = ((s >> 8) % 12) as i64 + 1;
                *slot = if (s >> 4) & 1 == 1 { -mag } else { mag };
            }
        }
        roundtrip_tokens(&block, 1, &DEFAULT_SCAN_8X8, DCT_DCT, 8, &fresh_nz(16));
    }

    #[test]
    fn tokens_16x16_cat6_high_magnitude_10bit() {
        // A high-magnitude coefficient drives the CAT6 path; at 10-bit the
        // §6.4.26 high-bit prefix fires.
        let mut block = vec![0i64; 256];
        block[0] = 600;
        block[1] = -250;
        block[17] = 80;
        roundtrip_tokens(&block, 2, &DEFAULT_SCAN_16X16, DCT_DCT, 10, &fresh_nz(32));
    }
}
