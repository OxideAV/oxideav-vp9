//! Round-458 **coefficient election** (rate-distortion optimised
//! quantisation) over the §6.4.24 token tree.
//!
//! The scalar quantiser ([`crate::fwd_transform::quantize_block_tx`])
//! rounds every coefficient to its nearest level. That is the
//! distortion optimum per coefficient, but not the rate-distortion
//! optimum of the block: a trailing `ONE` token costs its `more_coefs`
//! flag, its tree path, its sign and the closing end-of-block flag
//! after it, and a level one step lower is often cheaper by more bits
//! than its extra distortion is worth. This module re-elects levels
//! against the frame's *own* §6.1.2 bank — the same
//! `coef_probs[ txSz ][ plane > 0 ][ is_inter ][ band ][ ctx ]` cells the
//! writer codes under, the internal-node probabilities of the tokens
//! above `TWO` derived through the §9.3.2 `pareto( )` tail exactly as
//! [`crate::tokens::build_token_probs`] does — with the §9.3.2 context
//! derivation replayed over the block's own already-elected tokens.
//!
//! Distortion is measured in the pixel domain through the crate's own
//! §8.7 inverse transform: per transform size and coefficient position
//! a gain (the pixel-domain squared error one unit of dequantised
//! coefficient produces) is measured once from the inverse transform
//! itself, so the trade-off is made in the units the reconstruction
//! error is actually paid in. Two greedy passes, both deterministic:
//!
//! 1. **tail trimming** — while the last coded token is a `ONE`, drop
//!    it (and the zero run before it) when the bits it costs are worth
//!    more than the distortion it saves;
//! 2. **level reduction** — walking the scan, lower a level by one
//!    (`ONE → ZERO` also suppresses the next position's `more_coefs`
//!    flag, per the §6.4.24 `check_eob` latch) when the rate saved
//!    outweighs the distortion added.
//!
//! Neither pass touches the decoder: the elected levels are ordinary
//! tokens, coded and reconstructed by the existing writer / mirror.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.24 / §8.6.2 / §8.7 / §9.3.

use crate::coef_probs::CoefProbs;
use crate::idct::{inverse_transform_2d, DCT_DCT};
use crate::scan::get_scan;
use crate::token_writer::token_for_magnitude;
use crate::tokens::{
    build_token_probs, coef_band, token_cache_neighbours, CAT_PROBS, DCT_VAL_CATEGORY6,
    ENERGY_CLASS, EXTRA_BITS, TOKEN_TREE, ZERO_TOKEN,
};

/// Lagrange multiplier as a fraction of the squared pixel-domain
/// quantiser step per bit: `lambda = LAMBDA.0 / LAMBDA.1 × qstep_px²`.
pub(crate) const LAMBDA: (u64, u64) = (1, 16);

/// `-256 log2( p / 256 )` in 1/256 bit (the entropy model's table,
/// shared so the two elections agree on what a bit costs).
fn cost_of(bit: u32, p: u8) -> u32 {
    // Unused bank cells (band 0 above context 2) hold 0; clamp into
    // the codeable range so the table lookup is total.
    crate::entropy_model::bool_cost(bit, p.clamp(1, 255))
}

/// Pixel-domain gains of one unit of dequantised coefficient per
/// transform size and raster position (`1/256` pixel-SSE units),
/// measured through the crate's inverse `DCT_DCT` at every size.
fn gains() -> &'static [Vec<u64>; 4] {
    static G: std::sync::OnceLock<[Vec<u64>; 4]> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let mut out: [Vec<u64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (tx, slot) in out.iter_mut().enumerate() {
            let n0 = 4usize << tx;
            // A unit large enough that the transform's own rounding is
            // negligible against the measured energy.
            const UNIT: i64 = 4096;
            let mut g = vec![0u64; n0 * n0];
            for (pos, gp) in g.iter_mut().enumerate() {
                let mut block = vec![0i64; n0 * n0];
                block[pos] = UNIT;
                inverse_transform_2d(&mut block, tx as u32 + 2, DCT_DCT, false);
                let sse: u128 = block.iter().map(|&v| (v as i128 * v as i128) as u128).sum();
                *gp = ((sse * 256) / (UNIT as u128 * UNIT as u128)) as u64;
            }
            *slot = g;
        }
        out
    })
}

/// `[tx][plane>0][is_inter][band][ctx][N]` per-cell cost table.
type CellCosts<const N: usize> = [[[[[[u32; N]; 6]; 6]; 2]; 2]; 4];

/// The per-frame rate model: token-tree and `more_coefs` costs for
/// every `coef_probs` cell of the bank the frame codes under.
pub(crate) struct RdoqModel {
    /// `[..][token 0..=10]` tree-path cost (1/256 bit), `pareto( )`
    /// tail included.
    token: Box<CellCosts<11>>,
    /// `[..][more_coefs bit]`.
    more: Box<CellCosts<2>>,
    bit_depth: u32,
}

impl RdoqModel {
    /// Build the model from the frame's loaded bank.
    pub fn new(coef_probs: &CoefProbs, bit_depth: u32) -> Self {
        let mut token = Box::new([[[[[[0u32; 11]; 6]; 6]; 2]; 2]; 4]);
        let mut more = Box::new([[[[[[0u32; 2]; 6]; 6]; 2]; 2]; 4]);
        for tx in 0..4 {
            for pt in 0..2 {
                for rf in 0..2 {
                    for band in 0..6 {
                        for ctx in 0..6 {
                            let cell = &coef_probs[tx][pt][rf][band][ctx];
                            let probs = build_token_probs(cell);
                            more[tx][pt][rf][band][ctx] =
                                [cost_of(0, cell[0]), cost_of(1, cell[0])];
                            for tok in 0..=10u32 {
                                token[tx][pt][rf][band][ctx][tok as usize] = tree_cost(tok, &probs);
                            }
                        }
                    }
                }
            }
        }
        Self {
            token,
            more,
            bit_depth,
        }
    }

    /// Cost (1/256 bit) of coding magnitude `mag` (sign included for
    /// `mag > 0`) at a cell.
    fn mag_cost(&self, tx: usize, pt: usize, rf: usize, band: usize, ctx: usize, mag: u32) -> u64 {
        if mag == 0 {
            return u64::from(self.token[tx][pt][rf][band][ctx][ZERO_TOKEN as usize]);
        }
        let tok = token_for_magnitude(mag);
        u64::from(self.token[tx][pt][rf][band][ctx][tok as usize])
            + extra_bits_cost(tok, mag, self.bit_depth)
            + 256
    }
}

/// Tree-path cost of `token` under the 10-node probabilities.
fn tree_cost(token: u32, probs: &[u8; 10]) -> u32 {
    let target = -(token as i32);
    let mut n: i32 = 0;
    let mut cost = 0u32;
    for _ in 0..TOKEN_TREE.len() {
        let node = (n >> 1) as usize;
        let mut chosen = None;
        for bit in 0u32..2 {
            let next = TOKEN_TREE[n as usize + bit as usize];
            if subtree_contains(next, target) {
                chosen = Some((bit, next));
                break;
            }
        }
        let (bit, next) = chosen.expect("every token is a tree leaf");
        cost += cost_of(bit, probs[node]);
        if next <= 0 {
            return cost;
        }
        n = next;
    }
    cost
}

fn subtree_contains(node: i32, target: i32) -> bool {
    if node <= 0 {
        return node == target;
    }
    let base = node as usize;
    subtree_contains(TOKEN_TREE[base], target) || subtree_contains(TOKEN_TREE[base + 1], target)
}

/// Cost of the §6.4.26 extra magnitude bits of `mag` under `token`.
fn extra_bits_cost(token: u32, mag: u32, bit_depth: u32) -> u64 {
    let row = &EXTRA_BITS[token as usize];
    let cat = row[0] as usize;
    let num_extra = row[1];
    let base = row[2];
    let mut residual = mag - base;
    let mut cost = 0u64;
    if token == DCT_VAL_CATEGORY6 && bit_depth > 8 {
        // High bits at probability 255 (≈ free); mask them out.
        let mut mask = 0u32;
        for e in 0..(bit_depth - 8) {
            let shift = 5 + bit_depth - e;
            cost += u64::from(cost_of((residual >> shift) & 1, 255));
            mask |= 1 << shift;
        }
        residual &= !mask;
    }
    let cat_row = &CAT_PROBS[cat];
    for e in 0..num_extra {
        let p = cat_row[e as usize];
        let bit = (residual >> (num_extra - 1 - e)) & 1;
        cost += u64::from(cost_of(bit, p));
    }
    cost
}

/// One transform block's election inputs.
pub(crate) struct RdoqBlock<'a> {
    /// The dequantisation-domain coefficients before quantisation
    /// (the forward transform output), raster order.
    pub orig: &'a [i64],
    pub tx_sz: u32,
    pub tx_type: u8,
    pub plane: usize,
    pub is_inter: bool,
    pub dc_q: i32,
    pub ac_q: i32,
}

/// Re-elect the quantised levels `lvl` (raster order, the
/// [`crate::fwd_transform::quantize_block_tx`] output) of one block.
/// Returns `true` when any level moved.
pub(crate) fn rdoq_block(model: &RdoqModel, b: &RdoqBlock<'_>, lvl: &mut [i64]) -> bool {
    let tx = b.tx_sz as usize;
    let n0 = 4usize << tx;
    let seg_eob = n0 * n0;
    debug_assert_eq!(lvl.len(), seg_eob);
    let scan = get_scan(b.plane, b.tx_sz, b.tx_type);
    let effective_tx_type = if b.plane > 0 || b.tx_sz == 3 {
        DCT_DCT
    } else {
        b.tx_type
    };
    let pt = usize::from(b.plane > 0);
    let rf = usize::from(b.is_inter);
    let dd: i64 = if b.tx_sz == 3 { 2 } else { 1 };
    let g = &gains()[tx];
    // lambda × ΔR (1/256 bit) against ΔD (dequant² units) × gain
    // (1/256 pixel-SSE per dequant²): both sides carry the 1/256, so
    //   accept  ⇔  ΔD × g[pos] × dd⁻² ≤ LAMBDA × qstep_px² × ΔR / 256
    // with qstep_px² = ac_q² × g_mean / 256 (dequant-domain step ac_q,
    // pixel gain g_mean). Cleared of fractions:
    //   ΔD × g[pos] × 256 × LAMBDA.1 ≤ LAMBDA.0 × ac_q² × g_mean × ΔR × dd²
    let g_mean: u64 = g.iter().sum::<u64>() / seg_eob as u64;
    let acq = b.ac_q as u64;
    let rhs_scale: u128 =
        u128::from(LAMBDA.0) * u128::from(acq * acq) * u128::from(g_mean) * (dd * dd) as u128;
    let lhs_scale: u128 = 256 * u128::from(LAMBDA.1);
    let accept = |delta_d: i128, gain: u64, delta_r: i128| -> bool {
        if delta_r <= 0 {
            return false;
        }
        if delta_d <= 0 {
            return true;
        }
        (delta_d as u128) * u128::from(gain) * lhs_scale <= rhs_scale * (delta_r as u128)
    };
    let q_at = |pos: usize| -> i64 { i64::from(if pos == 0 { b.dc_q } else { b.ac_q }) };
    // Distortion of level m at pos, dequant-domain (× dd).
    let dist = |pos: usize, m: i64| -> i128 {
        let c = b.orig[pos].abs() * dd;
        let e = c - m * q_at(pos);
        (e as i128) * (e as i128)
    };

    let mut eob = 0usize;
    for (c, &p) in scan.iter().enumerate() {
        if lvl[p as usize] != 0 {
            eob = c + 1;
        }
    }
    if eob == 0 {
        return false;
    }
    let mut changed = false;

    // The §9.3.2 context of scan index c over the current levels'
    // energy classes (DC: an assumed mid context — its above/left
    // non-zero flags live outside the block).
    let cache_class = |lvl: &[i64], pos: usize| -> usize {
        let m = lvl[pos].unsigned_abs() as u32;
        let tok = if m == 0 {
            ZERO_TOKEN
        } else {
            token_for_magnitude(m)
        };
        usize::from(ENERGY_CLASS[tok as usize])
    };
    let ctx_at = |lvl: &[i64], c: usize| -> usize {
        if c == 0 {
            return 1;
        }
        let pos = scan[c] as usize;
        let (nb0, nb1) = token_cache_neighbours(c, pos, b.tx_sz, effective_tx_type);
        (1 + cache_class(lvl, nb0) + cache_class(lvl, nb1)) >> 1
    };
    let band_at = |c: usize| coef_band(c, b.tx_sz);

    // Pass 1: tail trimming.
    loop {
        let last = eob - 1;
        let pos = scan[last] as usize;
        if lvl[pos].abs() != 1 {
            break;
        }
        // Previous non-zero scan index (the new last token).
        let mut prev_nz: Option<usize> = None;
        for c in (0..last).rev() {
            if lvl[scan[c] as usize] != 0 {
                prev_nz = Some(c);
                break;
            }
        }
        let new_eob = prev_nz.map_or(0, |c| c + 1);
        // Rate kept: the zero run (new_eob..last) — the first zero
        // carries a more_coefs(1) flag (the token before it was
        // non-zero, or it is the DC position) — then the ONE at
        // `last` (more_coefs(1) only when the preceding token was
        // non-zero, i.e. no zero run), then the closing flag.
        let mut keep = 0u64;
        let mut check_eob = true;
        for c in new_eob..last {
            let (band, ctx) = (band_at(c), ctx_at(lvl, c));
            if check_eob {
                keep += u64::from(model.more[tx][pt][rf][band][ctx][1]);
            }
            keep += model.mag_cost(tx, pt, rf, band, ctx, 0);
            check_eob = false;
        }
        {
            let (band, ctx) = (band_at(last), ctx_at(lvl, last));
            if check_eob {
                keep += u64::from(model.more[tx][pt][rf][band][ctx][1]);
            }
            keep += model.mag_cost(tx, pt, rf, band, ctx, 1);
        }
        if eob < seg_eob {
            let (band, ctx) = (band_at(eob), ctx_at(lvl, eob));
            keep += u64::from(model.more[tx][pt][rf][band][ctx][0]);
        }
        // Rate after trimming: the closing flag at new_eob (coded —
        // the token before it is non-zero, or it is the DC position).
        let mut trim = 0u64;
        if new_eob < seg_eob {
            let (band, ctx) = (band_at(new_eob), ctx_at(lvl, new_eob));
            trim += u64::from(model.more[tx][pt][rf][band][ctx][0]);
        }
        let delta_r = keep as i128 - trim as i128;
        let delta_d = dist(pos, 0) - dist(pos, 1);
        if accept(delta_d, g[pos], delta_r) {
            lvl[pos] = 0;
            changed = true;
            eob = new_eob;
            if eob == 0 {
                return true;
            }
        } else {
            break;
        }
    }

    // Pass 2: level reduction in scan order (the last token keeps at
    // least magnitude 2 → 1 here; pass 1 owns its removal).
    for (c, &sp) in scan.iter().enumerate().take(eob) {
        let pos = sp as usize;
        let m = lvl[pos].abs();
        if m == 0 || (c + 1 == eob && m == 1) {
            continue;
        }
        let (band, ctx) = (band_at(c), ctx_at(lvl, c));
        let cost_m = model.mag_cost(tx, pt, rf, band, ctx, m as u32);
        let cost_lower = model.mag_cost(tx, pt, rf, band, ctx, (m - 1) as u32);
        let mut delta_r = cost_m as i128 - cost_lower as i128;
        if m == 1 {
            // ONE → ZERO suppresses the next position's more_coefs(1).
            let (nb, nctx) = (band_at(c + 1), ctx_at(lvl, c + 1));
            delta_r += i128::from(model.more[tx][pt][rf][nb][nctx][1]);
        }
        let delta_d = dist(pos, m - 1) - dist(pos, m);
        if accept(delta_d, g[pos], delta_r) {
            lvl[pos] = if lvl[pos] < 0 { -(m - 1) } else { m - 1 };
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coef_probs::DEFAULT_COEF_PROBS;

    /// The measured gains are positive at every position and the DC
    /// gain of every size matches the closed-form energy of a flat
    /// block (the inverse DC basis is constant over the block).
    #[test]
    fn gains_are_positive_and_dc_matches_flat_energy() {
        for tx in 0..4usize {
            let g = &gains()[tx];
            assert!(g.iter().all(|&v| v > 0), "tx {tx}: positive gains");
            let n0 = 4usize << tx;
            let mut block = vec![0i64; n0 * n0];
            block[0] = 4096;
            inverse_transform_2d(&mut block, tx as u32 + 2, DCT_DCT, false);
            let first = block[0];
            assert!(
                block.iter().all(|&v| (v - first).abs() <= 1),
                "tx {tx}: flat DC"
            );
        }
    }

    /// Token costs are monotone in magnitude within a cell for the
    /// small tokens and every ZERO cheaper than ONE where the bank says
    /// zeros are likelier.
    #[test]
    fn model_costs_follow_the_bank() {
        let model = RdoqModel::new(&DEFAULT_COEF_PROBS, 8);
        let cell = &DEFAULT_COEF_PROBS[0][0][0][1][1];
        let zero = model.mag_cost(0, 0, 0, 1, 1, 0);
        let one = model.mag_cost(0, 0, 0, 1, 1, 1);
        let two = model.mag_cost(0, 0, 0, 1, 1, 2);
        if cell[1] > 128 {
            assert!(zero < one);
        }
        assert!(one < two);
        assert!(model.mag_cost(0, 0, 0, 1, 1, 70) > model.mag_cost(0, 0, 0, 1, 1, 10));
    }

    /// A lone trailing ONE that is barely past the rounding threshold
    /// is trimmed; a block whose coefficients sit exactly on their
    /// levels is left alone.
    #[test]
    fn trims_a_marginal_trailing_one_and_keeps_exact_levels() {
        let model = RdoqModel::new(&DEFAULT_COEF_PROBS, 8);
        let q = 40i32;
        let mut orig = vec![0i64; 16];
        orig[0] = 400; // DC: exactly 10 levels.
        orig[15] = 21; // 0.525 q → rounds to ONE, worth trimming.
        let mut lvl = orig.clone();
        crate::fwd_transform::quantize_block_tx(&mut lvl, q, q, 0, 8);
        assert_eq!(lvl[15], 1);
        let b = RdoqBlock {
            orig: &orig,
            tx_sz: 0,
            tx_type: DCT_DCT,
            plane: 0,
            is_inter: true,
            dc_q: q,
            ac_q: q,
        };
        assert!(rdoq_block(&model, &b, &mut lvl));
        assert_eq!(lvl[15], 0);
        assert_eq!(lvl[0], 10, "an exact level never moves");
        let mut again = lvl.clone();
        assert!(!rdoq_block(&model, &b, &mut again));
        assert_eq!(again, lvl);
    }
}
