//! Encoder-side **entropy model** (round 455): the writer's mirror of the
//! decoder's persistent probability state, plus the §6.3 forward-update
//! election.
//!
//! Two halves:
//!
//! * [`EntropyModel`] — the §6.1.2 / §7.2 `FrameContext[ 4 ]` banks the
//!   decoder threads across frames (`load_probs( )` / `save_probs( )`),
//!   the §6.2 `setup_past_independence( )` + `reset_frame_context`
//!   resets, and the §6.1.2 `refresh_probs( )` **backward adaptation**
//!   (§8.4.2–§8.4.4, run through the same [`crate::prob_adapt`] code the
//!   decoder uses, over the §9.3.4 counts the writers collect). A frame
//!   coded with `error_resilient_mode = 0`, `frame_parallel_decoding_mode
//!   = 0` and `refresh_frame_context = 1` leaves the bank the decoder
//!   derives — the encoder keeps the identical bank and codes the next
//!   frame against it.
//! * the **forward-update election**: given the loaded bank and a
//!   frame's symbol counts, choose per probability cell whether a §6.3.3
//!   `diff_update_prob` (or §6.3.17 `update_mv_prob`) pays for itself —
//!   the update's own cost (the `B(252)` flag plus the §6.3.4
//!   `decode_term_subexp` bits, or the `L(7)` MV probability) against the
//!   change in the frame's symbol cost under the new probability, both
//!   measured in 1/256-bit units from an integer log2 table (bit-exact
//!   across platforms, so the election is deterministic). The elected
//!   bank is what the frame's arithmetic coder runs with; the frame's
//!   *backward* adaptation still starts from the loaded bank, exactly as
//!   §6.1.2 `load_probs( )` discards the forward updates.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.1.2 (`refresh_probs`), §6.2
//! (`setup_past_independence` / `reset_frame_context`), §6.3.3–§6.3.6,
//! §6.3.17, §7.1.2, §8.4; the §9.3.4 counting table.

use crate::bool_encoder::BoolEncoder;
use crate::coef_probs::CoefProbs;
use crate::compressed::{
    inv_remap_prob, tx_mode_to_biggest_tx_size, FrameContext, ReferenceMode, TxMode,
};
use crate::header::{FrameType, Vp9FrameHeader};
use crate::mode_info::{
    BLOCK_SIZE_GROUPS, COMP_MODE_CONTEXTS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_TREE,
    INTER_MODE_CONTEXTS, INTER_MODE_TREE, INTRA_MODES, INTRA_MODE_TREE, IS_INTER_CONTEXTS,
    REF_CONTEXTS, SWITCHABLE, TX_SIZE_16_TREE, TX_SIZE_32_TREE, TX_SIZE_8_TREE,
};
use crate::mv::{MV_CLASS_TREE, MV_FR_TREE, MV_JOINT_TREE};
use crate::partition::{PARTITION_CONTEXTS, PARTITION_TREE};
use crate::prob_adapt::{adapt_coef_probs, adapt_noncoef_probs, FrameCounts};
use crate::Error;

// ----- bit-cost arithmetic -----

/// `-256 * log2( p / 256 )` for `p` in `1..=255` — the cost, in 1/256
/// bit, of coding a boolean whose probability of the coded value is
/// `p / 256`. Index 0 is unused (a probability of 0 never occurs).
///
/// Computed with integer arithmetic only (an 8-bit fractional binary
/// logarithm by repeated squaring), so every platform elects the same
/// forward updates.
fn bit_cost_table() -> &'static [u32; 256] {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (p, slot) in t.iter_mut().enumerate().skip(1) {
            // log2( p ) in 1/256: integer part + 8 fractional bits.
            let ilog = 31 - (p as u32).leading_zeros();
            // Mantissa in 16.16 fixed point, in [1, 2).
            let mut x: u64 = ((p as u64) << 16) >> ilog;
            let mut frac = 0u32;
            for _ in 0..8 {
                x = (x * x) >> 16;
                frac <<= 1;
                if x >= 2 << 16 {
                    frac |= 1;
                    x >>= 1;
                }
            }
            let log2_p = ilog * 256 + frac;
            *slot = 8 * 256 - log2_p;
        }
        t
    })
}

/// Cost (1/256 bit) of coding value `0` under probability `p`.
#[inline]
fn cost0(p: u8) -> u64 {
    u64::from(bit_cost_table()[usize::from(p)])
}

/// Cost (1/256 bit) of coding value `1` under probability `p`.
#[inline]
fn cost1(p: u8) -> u64 {
    u64::from(bit_cost_table()[256 - usize::from(p)])
}

/// Symbol cost of `c0` zeros and `c1` ones under `p`.
#[inline]
fn symbol_cost(p: u8, c0: u32, c1: u32) -> u64 {
    u64::from(c0) * cost0(p) + u64::from(c1) * cost1(p)
}

/// Number of bits the §6.3.4 `decode_term_subexp( )` cascade spends on
/// `delta` (every read is an `L(n)` literal, i.e. exactly `n` bits).
fn subexp_bits(delta: u32) -> u32 {
    if delta < 16 {
        1 + 4
    } else if delta < 32 {
        2 + 4
    } else if delta < 64 {
        3 + 5
    } else if delta < 129 {
        3 + 7
    } else {
        3 + 7 + 1
    }
}

/// The `update_prob == 0` flag cost (`B(252)` coding 0).
fn flag_off_cost() -> u64 {
    cost0(252)
}

/// The `update_prob == 1` flag cost (`B(252)` coding 1).
fn flag_on_cost() -> u64 {
    cost1(252)
}

// ----- §6.3.3–§6.3.6 / §6.3.17 writers -----

/// §6.3.4 `decode_term_subexp( )` inverse: write `v` (`0..=254`) as the
/// literal cascade the decoder reads.
pub(crate) fn write_term_subexp(enc: &mut BoolEncoder, v: u32) {
    debug_assert!(v <= 254);
    if v < 16 {
        enc.write_literal(0, 1);
        enc.write_literal(v, 4);
    } else if v < 32 {
        enc.write_literal(1, 1);
        enc.write_literal(0, 1);
        enc.write_literal(v - 16, 4);
    } else if v < 64 {
        enc.write_literal(1, 1);
        enc.write_literal(1, 1);
        enc.write_literal(0, 1);
        enc.write_literal(v - 32, 5);
    } else if v < 129 {
        enc.write_literal(1, 1);
        enc.write_literal(1, 1);
        enc.write_literal(1, 1);
        enc.write_literal(v - 64, 7);
    } else {
        // v = (w << 1) - 1 + bit with w in 65..=127.
        enc.write_literal(1, 1);
        enc.write_literal(1, 1);
        enc.write_literal(1, 1);
        let w = (v + 1) >> 1;
        let bit = (v + 1) & 1;
        enc.write_literal(w, 7);
        enc.write_literal(bit, 1);
    }
}

/// The cheapest §6.3.4 `deltaProb` that §6.3.5 `inv_remap_prob( )` maps
/// onto `new` from `old`, or `None` when no delta reaches `new` (the
/// election only ever picks reachable targets).
pub(crate) fn delta_for(old: u8, new: u8) -> Option<u32> {
    let mut best: Option<u32> = None;
    for delta in 0..=254u32 {
        if inv_remap_prob(delta, old) != new {
            continue;
        }
        match best {
            Some(b) if subexp_bits(b) <= subexp_bits(delta) => {}
            _ => best = Some(delta),
        }
    }
    best
}

/// §6.3.3 `diff_update_prob( )` inverse: code the transition `old →
/// new` (a `B(252) == 0` flag when equal). Returns
/// [`Error::Unsupported`] for an unreachable `new`.
pub(crate) fn write_diff_update_prob(enc: &mut BoolEncoder, old: u8, new: u8) -> Result<(), Error> {
    if old == new {
        enc.write_bool(0, 252);
        return Ok(());
    }
    let delta = delta_for(old, new).ok_or(Error::Unsupported)?;
    enc.write_bool(1, 252);
    write_term_subexp(enc, delta);
    Ok(())
}

/// §6.3.17 `update_mv_prob( )` inverse: a `B(252) == 0` flag when
/// `new == old`, else the flag plus the `L(7)` payload — which only
/// reaches odd probabilities (`(mv_prob << 1) | 1`); an even `new`
/// returns [`Error::Unsupported`].
pub(crate) fn write_update_mv_prob(enc: &mut BoolEncoder, old: u8, new: u8) -> Result<(), Error> {
    if old == new {
        enc.write_bool(0, 252);
        return Ok(());
    }
    if new & 1 == 0 {
        return Err(Error::Unsupported);
    }
    enc.write_bool(1, 252);
    enc.write_literal(u32::from(new >> 1), 7);
    Ok(())
}

// ----- election -----

/// Elect one `diff_update_prob` cell: the probability (== `old` for "no
/// update") minimising the update's own cost plus the cost of the
/// `c0` / `c1` symbols coded under it. Ties keep `old`.
fn elect_diff_cell(old: u8, c0: u32, c1: u32) -> u8 {
    if c0 == 0 && c1 == 0 {
        return old;
    }
    let mut best_p = old;
    let mut best_cost = flag_off_cost() + symbol_cost(old, c0, c1);
    for delta in 0..=254u32 {
        let p = inv_remap_prob(delta, old);
        if p == old {
            continue;
        }
        let cost = flag_on_cost() + u64::from(subexp_bits(delta)) * 256 + symbol_cost(p, c0, c1);
        if cost < best_cost {
            best_cost = cost;
            best_p = p;
        }
    }
    best_p
}

/// Elect one `update_mv_prob` cell over the odd probabilities the
/// `L(7)` payload reaches.
fn elect_mv_cell(old: u8, c0: u32, c1: u32) -> u8 {
    if c0 == 0 && c1 == 0 {
        return old;
    }
    let mut best_p = old;
    let mut best_cost = flag_off_cost() + symbol_cost(old, c0, c1);
    for v in 0..128u32 {
        let p = ((v << 1) | 1) as u8;
        if p == old {
            continue;
        }
        let cost = flag_on_cost() + 7 * 256 + symbol_cost(p, c0, c1);
        if cost < best_cost {
            best_cost = cost;
            best_p = p;
        }
    }
    best_p
}

/// Per-node `(node, count0, count1)` of a §9.3.1 tree given the leaf
/// counts — the same recursion §8.4.2 `merge_probs( )` folds, so a tree
/// cell's binary decision is elected on exactly the counts the
/// backward adaptation would see.
fn tree_node_counts(
    tree: &[i32],
    i: usize,
    counts: &[u32],
    out: &mut Vec<(usize, u32, u32)>,
) -> u32 {
    let s = tree[i];
    let left = if s <= 0 {
        counts[(-s) as usize]
    } else {
        tree_node_counts(tree, s as usize, counts, out)
    };
    let r = tree[i + 1];
    let right = if r <= 0 {
        counts[(-r) as usize]
    } else {
        tree_node_counts(tree, r as usize, counts, out)
    };
    out.push((i >> 1, left, right));
    left + right
}

/// Elect every node of one tree-coded probability row.
fn elect_tree(tree: &[i32], probs: &mut [u8], counts: &[u32]) {
    let mut nodes = Vec::new();
    tree_node_counts(tree, 0, counts, &mut nodes);
    for (node, c0, c1) in nodes {
        probs[node] = elect_diff_cell(probs[node], c0, c1);
    }
}

/// Elect every node of one MV tree-coded probability row (odd targets).
fn elect_mv_tree(tree: &[i32], probs: &mut [u8], counts: &[u32]) {
    let mut nodes = Vec::new();
    tree_node_counts(tree, 0, counts, &mut nodes);
    for (node, c0, c1) in nodes {
        probs[node] = elect_mv_cell(probs[node], c0, c1);
    }
}

/// Elect the §6.3.7 coefficient sweep: per active tx-size slab, the
/// per-cell elections (node 0 from `counts_more_coefs`, nodes 1–2 from
/// the §8.4.3 `small_token_tree` split of `counts_token`) are kept only
/// when the whole slab — its `update_probs = 1` bit plus every cell's
/// flag — costs less than the `update_probs = 0` bit alone.
fn elect_coef_probs(base: &CoefProbs, counts: &FrameCounts, tx_mode: TxMode) -> CoefProbs {
    let mut out = *base;
    let max_tx_size = tx_mode_to_biggest_tx_size(tx_mode);
    for t in 0..=max_tx_size {
        // Cost of the slab with no updates: one L(1) bit, symbols at
        // the base probabilities. Cost with the elected updates: one
        // L(1) bit plus every cell's flag (+ payload) and symbols.
        let mut cost_off = 256u64;
        let mut cost_on = 256u64;
        let mut any = false;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for l in 0..max_l {
                        let old = base[t][i][j][k][l];
                        let ct = counts.token[t][i][j][k][l];
                        let cm = counts.more_coefs[t][i][j][k][l];
                        let node_counts = [(cm[0], cm[1]), (ct[0], ct[1] + ct[2]), (ct[1], ct[2])];
                        let mut new = old;
                        for (n, &(c0, c1)) in node_counts.iter().enumerate() {
                            cost_off += symbol_cost(old[n], c0, c1);
                            let p = elect_diff_cell(old[n], c0, c1);
                            if p == old[n] {
                                cost_on += flag_off_cost() + symbol_cost(old[n], c0, c1);
                            } else {
                                any = true;
                                let delta = delta_for(old[n], p).expect("elected delta reachable");
                                cost_on += flag_on_cost()
                                    + u64::from(subexp_bits(delta)) * 256
                                    + symbol_cost(p, c0, c1);
                            }
                            new[n] = p;
                        }
                        out[t][i][j][k][l] = new;
                    }
                }
            }
        }
        if !any || cost_on >= cost_off {
            out[t] = base[t];
        }
    }
    out
}

/// Which §6.3 sweeps the frame's compressed header carries — the gates
/// the parser applies, so the election never touches a table the
/// header cannot code.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ForwardGates {
    /// `FrameIsIntra`: only the §6.3.1–§6.3.8 prefix is coded.
    pub frame_is_intra: bool,
    /// §6.3.1 `tx_mode` (gates §6.3.2 and the §6.3.7 slab count).
    pub tx_mode: TxMode,
    /// §6.3.12 `reference_mode` (gates the §6.3.13 sweeps).
    pub reference_mode: ReferenceMode,
    /// `interpolation_filter == SWITCHABLE` (gates §6.3.10).
    pub interp_filter_switchable: bool,
    /// `allow_high_precision_mv` (gates the §6.3.16 hp tail).
    pub allow_high_precision_mv: bool,
}

/// Elect the frame's forward updates on top of `base` from its counts.
/// The result is the coding bank: equal to `base` wherever no update
/// pays, and reachable through the §6.3 writers everywhere else.
pub(crate) fn elect_forward_updates(
    base: &FrameContext,
    counts: &FrameCounts,
    gates: ForwardGates,
) -> FrameContext {
    let mut fc = base.clone();
    let nc = &counts.noncoef;

    // §6.3.2 tx_mode_probs (TX_MODE_SELECT only).
    if matches!(gates.tx_mode, TxMode::TxModeSelect) {
        for ctx in 0..2 {
            elect_tree(
                &TX_SIZE_8_TREE,
                &mut fc.tx_probs[1][ctx][..1],
                &nc.tx_size[1][ctx],
            );
            elect_tree(
                &TX_SIZE_16_TREE,
                &mut fc.tx_probs[2][ctx][..2],
                &nc.tx_size[2][ctx],
            );
            elect_tree(
                &TX_SIZE_32_TREE,
                &mut fc.tx_probs[3][ctx][..3],
                &nc.tx_size[3][ctx],
            );
        }
    }
    // §6.3.7 coef probs.
    fc.coef_probs = elect_coef_probs(&base.coef_probs, counts, gates.tx_mode);
    // §6.3.8 skip.
    for i in 0..3 {
        fc.skip_prob[i] = elect_diff_cell(base.skip_prob[i], nc.skip[i][0], nc.skip[i][1]);
    }
    if gates.frame_is_intra {
        return fc;
    }
    // §6.3.9 inter_mode.
    for i in 0..INTER_MODE_CONTEXTS {
        elect_tree(
            &INTER_MODE_TREE,
            &mut fc.inter_mode_probs[i],
            &nc.inter_mode[i],
        );
    }
    // §6.3.10 interp_filter.
    if gates.interp_filter_switchable {
        for i in 0..INTERP_FILTER_CONTEXTS {
            elect_tree(
                &INTERP_FILTER_TREE,
                &mut fc.interp_filter_probs[i],
                &nc.interp_filter[i],
            );
        }
    }
    // §6.3.11 is_inter.
    for i in 0..IS_INTER_CONTEXTS {
        fc.is_inter_prob[i] =
            elect_diff_cell(base.is_inter_prob[i], nc.is_inter[i][0], nc.is_inter[i][1]);
    }
    // §6.3.13 reference probs, gated by reference_mode.
    if gates.reference_mode == ReferenceMode::ReferenceModeSelect {
        for i in 0..COMP_MODE_CONTEXTS {
            fc.comp_mode_prob[i] = elect_diff_cell(
                base.comp_mode_prob[i],
                nc.comp_mode[i][0],
                nc.comp_mode[i][1],
            );
        }
    }
    if gates.reference_mode != ReferenceMode::CompoundReference {
        for i in 0..REF_CONTEXTS {
            for j in 0..2 {
                fc.single_ref_prob[i][j] = elect_diff_cell(
                    base.single_ref_prob[i][j],
                    nc.single_ref[i][j][0],
                    nc.single_ref[i][j][1],
                );
            }
        }
    }
    if gates.reference_mode != ReferenceMode::SingleReference {
        for i in 0..REF_CONTEXTS {
            fc.comp_ref_prob[i] =
                elect_diff_cell(base.comp_ref_prob[i], nc.comp_ref[i][0], nc.comp_ref[i][1]);
        }
    }
    // §6.3.14 y_mode.
    for i in 0..BLOCK_SIZE_GROUPS {
        elect_tree(&INTRA_MODE_TREE, &mut fc.y_mode_probs[i], &nc.y_mode[i]);
    }
    // (uv_mode_probs: no §6.3 sweep.)
    debug_assert_eq!(fc.uv_mode_probs.len(), INTRA_MODES);
    // §6.3.15 partition.
    for i in 0..PARTITION_CONTEXTS {
        elect_tree(
            &PARTITION_TREE,
            &mut fc.partition_probs[i],
            &nc.partition[i],
        );
    }
    // §6.3.16 mv_probs.
    let mv = &mut fc.mv_probs;
    elect_mv_tree(&MV_JOINT_TREE, &mut mv.joint_probs, &nc.mv_joint);
    for i in 0..2 {
        let mc = &nc.mv_comp[i];
        mv.sign_prob[i] = elect_mv_cell(mv.sign_prob[i], mc.sign[0], mc.sign[1]);
        elect_mv_tree(&MV_CLASS_TREE, &mut mv.class_probs[i], &mc.class);
        mv.class0_bit_prob[i] =
            elect_mv_cell(mv.class0_bit_prob[i], mc.class0_bit[0], mc.class0_bit[1]);
        for (j, slot) in mv.bits_prob[i].iter_mut().enumerate() {
            *slot = elect_mv_cell(*slot, mc.bits[j][0], mc.bits[j][1]);
        }
        for j in 0..2 {
            elect_mv_tree(&MV_FR_TREE, &mut mv.class0_fr_probs[i][j], &mc.class0_fr[j]);
        }
        elect_mv_tree(&MV_FR_TREE, &mut mv.fr_probs[i], &mc.fr);
        if gates.allow_high_precision_mv {
            mv.class0_hp_prob[i] =
                elect_mv_cell(mv.class0_hp_prob[i], mc.class0_hp[0], mc.class0_hp[1]);
            mv.hp_prob[i] = elect_mv_cell(mv.hp_prob[i], mc.hp[0], mc.hp[1]);
        }
    }
    fc
}

// ----- §6.1.2 / §7.2 bank mirror -----

/// `FrameIsIntra` (§7.2): a key frame or an `intra_only` frame.
pub(crate) fn frame_is_intra(hdr: &Vp9FrameHeader) -> bool {
    hdr.frame_type == FrameType::KeyFrame || hdr.intra_only
}

/// The encoder's copy of the decoder's persistent entropy state: the
/// four §7.2 `FrameContext` banks and `LastFrameType`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EntropyModel {
    banks: [FrameContext; 4],
    last_frame_type_was_key: bool,
}

impl Default for EntropyModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyModel {
    /// Fresh state: the §10.5 default banks (what a decoder holds before
    /// its first frame), `LastFrameType` unset.
    pub fn new() -> Self {
        Self {
            banks: Default::default(),
            last_frame_type_was_key: false,
        }
    }

    /// §6.2 uncompressed-header mirror for the frame about to be coded:
    /// the `setup_past_independence( )` + `reset_frame_context` bank
    /// resets, returning the effective `frame_context_idx` the frame
    /// loads (and later saves) — forced to 0 on intra / error-resilient
    /// frames.
    pub fn begin_frame(&mut self, hdr: &Vp9FrameHeader) -> usize {
        if frame_is_intra(hdr) || hdr.error_resilient_mode {
            let reset_all = hdr.frame_type == FrameType::KeyFrame
                || hdr.error_resilient_mode
                || hdr.reset_frame_context == 3;
            if reset_all {
                self.banks = Default::default();
            } else if hdr.reset_frame_context == 2 {
                self.banks[usize::from(hdr.frame_context_idx & 3)] = FrameContext::default();
            }
            0
        } else {
            usize::from(hdr.frame_context_idx & 3)
        }
    }

    /// The §6.1.2 `load_probs( idx )` bank.
    pub fn bank(&self, idx: usize) -> &FrameContext {
        &self.banks[idx]
    }

    /// §6.1.2 `refresh_probs( )` mirror after the frame's tile data:
    /// `coding` is the bank the frame's compressed header produced (the
    /// loaded bank plus its forward updates), `counts` the frame's
    /// §9.3.4 totals. Also tracks §7.2 `LastFrameType`.
    pub fn end_frame(
        &mut self,
        hdr: &Vp9FrameHeader,
        idx: usize,
        coding: &FrameContext,
        counts: &FrameCounts,
        tx_mode: TxMode,
    ) {
        let intra = frame_is_intra(hdr);
        let non_parallel = !hdr.error_resilient_mode && !hdr.frame_parallel_decoding_mode;
        if hdr.refresh_frame_context {
            if non_parallel {
                // load_probs( idx ) — the pre-frame bank, except tx_probs
                // / skip_prob which keep the forward-updated values on an
                // intra frame (load_probs2( ) restores them on inter).
                let mut work = self.banks[idx].clone();
                if intra {
                    work.tx_probs = coding.tx_probs;
                    work.skip_prob = coding.skip_prob;
                }
                adapt_coef_probs(
                    &mut work.coef_probs,
                    &counts.token,
                    &counts.more_coefs,
                    intra,
                    self.last_frame_type_was_key,
                );
                if !intra {
                    adapt_noncoef_probs(
                        &mut work,
                        &counts.noncoef,
                        hdr.interpolation_filter == SWITCHABLE,
                        matches!(tx_mode, TxMode::TxModeSelect),
                        hdr.allow_high_precision_mv,
                    );
                }
                self.banks[idx] = work;
            } else {
                // save_probs( idx ) of the forward-updated tables.
                self.banks[idx] = coding.clone();
            }
        }
        self.last_frame_type_was_key = hdr.frame_type == FrameType::KeyFrame;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::compressed::{decode_term_subexp, read_diff_update_prob, update_mv_prob};

    #[test]
    fn bit_cost_table_is_monotone_and_anchored() {
        let t = bit_cost_table();
        // p = 128 is exactly one bit; p = 256 would be zero.
        assert_eq!(t[128], 256);
        assert_eq!(t[64], 512);
        assert_eq!(t[32], 768);
        assert_eq!(t[1], 2048);
        for p in 2..256 {
            assert!(t[p] < t[p - 1], "cost must fall as p rises ({p})");
        }
        // Within one 1/256-bit of the float value.
        for (p, &cost) in t.iter().enumerate().skip(1) {
            let f = -(p as f64 / 256.0).log2() * 256.0;
            assert!((f64::from(cost) - f).abs() <= 1.0, "p={p}: {cost} vs {f}");
        }
    }

    #[test]
    fn term_subexp_roundtrips_every_delta() {
        for v in 0..=254u32 {
            let mut enc = BoolEncoder::new();
            write_term_subexp(&mut enc, v);
            let buf = enc.finish();
            let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
            assert_eq!(decode_term_subexp(&mut dec).unwrap(), v, "delta {v}");
        }
    }

    #[test]
    fn subexp_bits_matches_the_cascade() {
        for v in 0..=254u32 {
            let mut enc = BoolEncoder::new();
            for _ in 0..64 {
                write_term_subexp(&mut enc, v);
            }
            // 64 copies: the bit count is the byte length (minus the
            // flush) to within the coder's overhead.
            let bytes = enc.finish().len() as u32;
            let expect = 64 * subexp_bits(v) / 8;
            assert!(
                bytes >= expect && bytes <= expect + 3,
                "delta {v}: {bytes} bytes vs {expect}"
            );
        }
    }

    #[test]
    fn diff_update_roundtrips_every_reachable_target() {
        let mut reachable = 0usize;
        for old in 1..=255u8 {
            for new in 1..=255u8 {
                let mut enc = BoolEncoder::new();
                match write_diff_update_prob(&mut enc, old, new) {
                    Ok(()) => {
                        reachable += 1;
                        let buf = enc.finish();
                        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
                        assert_eq!(read_diff_update_prob(&mut dec, old).unwrap(), new);
                    }
                    Err(Error::Unsupported) => assert!(delta_for(old, new).is_none()),
                    Err(e) => panic!("{e:?}"),
                }
            }
        }
        // Every 1..=254 target is reachable from every start; 255 is
        // never produced by inv_remap_prob.
        assert!(reachable >= 255 * 254, "reachable pairs: {reachable}");
    }

    #[test]
    fn update_mv_prob_roundtrips_odd_targets_and_rejects_even() {
        for old in 1..=255u8 {
            for new in (1..=255u8).step_by(2) {
                let mut enc = BoolEncoder::new();
                write_update_mv_prob(&mut enc, old, new).unwrap();
                let buf = enc.finish();
                let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
                assert_eq!(update_mv_prob(&mut dec, old).unwrap(), new);
            }
        }
        let mut enc = BoolEncoder::new();
        assert_eq!(
            write_update_mv_prob(&mut enc, 3, 4).unwrap_err(),
            Error::Unsupported
        );
    }

    #[test]
    fn election_keeps_old_without_counts_and_moves_with_skewed_counts() {
        assert_eq!(elect_diff_cell(128, 0, 0), 128);
        assert_eq!(elect_mv_cell(128, 0, 0), 128);
        // 1000 zeros, no ones: a strong update toward 255 pays.
        let p = elect_diff_cell(128, 1000, 0);
        assert!(p > 200, "elected {p}");
        // A single symbol never pays for an update (>= 6 bits of flag).
        assert_eq!(elect_diff_cell(128, 1, 0), 128);
        let m = elect_mv_cell(128, 0, 1000);
        assert!(m < 40 && m & 1 == 1, "elected {m}");
    }

    #[test]
    fn tree_node_counts_mirror_merge_probs_totals() {
        let counts = [5u32, 3, 2, 7];
        let mut nodes = Vec::new();
        let total = tree_node_counts(&INTER_MODE_TREE, 0, &counts, &mut nodes);
        assert_eq!(total, 17);
        assert_eq!(nodes.len(), 3);
        // Every node's (c0 + c1) equals its subtree total; the root sums
        // everything.
        let root = nodes.iter().find(|n| n.0 == 0).unwrap();
        assert_eq!(root.1 + root.2, 17);
    }

    #[test]
    fn model_resets_on_keyframe_and_persists_adapted_bank() {
        let mut m = EntropyModel::new();
        let mut hdr = crate::pixel_encoder::lossless_keyframe_header(64, 64);
        hdr.frame_parallel_decoding_mode = false;
        hdr.refresh_frame_context = true;
        assert_eq!(m.begin_frame(&hdr), 0);
        let mut counts = FrameCounts::new_boxed();
        counts.more_coefs[0][0][0][0][0] = [100, 0];
        let coding = FrameContext::default();
        m.end_frame(&hdr, 0, &coding, &counts, TxMode::Only4x4);
        assert_ne!(
            m.bank(0).coef_probs[0][0][0][0][0],
            FrameContext::default().coef_probs[0][0][0][0][0]
        );
        assert!(m.last_frame_type_was_key);
        // A second keyframe resets every bank.
        m.begin_frame(&hdr);
        assert_eq!(*m.bank(0), FrameContext::default());
    }
}
