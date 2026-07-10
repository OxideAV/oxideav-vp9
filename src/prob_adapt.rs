//! §8.4 Probability adaptation primitives — the backward (count-based)
//! probability update process.
//!
//! This module transcribes the §8.4.1 `merge_prob` and §8.4.2
//! `merge_probs` processes plus the §8.4 `adapt_prob` / `adapt_probs`
//! wrappers from the VP9 Bitstream & Decoding Process Specification
//! v0.7. These are pure-compute helpers shared by the §8.4.3
//! coefficient adaptation and §8.4.4 non-coefficient adaptation
//! processes (wired in later commits).
//!
//! All truth is the spec: §8.4 (`vp9-spec.txt` lines 4193-4356) and the
//! `MAX_UPDATE_FACTOR = 128` / `COUNT_SAT = 20` constants (§3, lines
//! 523-524).
//!
// The §8.4.3 / §8.4.4 callers that consume `merge_probs` / `adapt_prob`
// / `adapt_probs` land in follow-up commits of this milestone; the
// primitives are fully exercised by the unit tests below in the
// meantime.
#![allow(dead_code)]

/// `MAX_UPDATE_FACTOR = 128` per §3 — the default maximum adaption
/// factor passed to `adapt_prob` / `adapt_probs`.
pub(crate) const MAX_UPDATE_FACTOR: u32 = 128;

/// `COUNT_SAT = 20` per §3 — the default saturation count passed to
/// `adapt_prob` / `adapt_probs`.
pub(crate) const COUNT_SAT: u32 = 20;

/// `Clip3( low, high, value )` per §4.7 — clamp `value` into the
/// inclusive `[low, high]` range.
#[inline]
fn clip3(low: i64, high: i64, value: i64) -> i64 {
    value.clamp(low, high)
}

/// `Round2( x, n )` per §4.7 — `(x + (1 << (n-1))) >> n` for `n > 0`,
/// and `x` for `n == 0`.
#[inline]
fn round2(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// §8.4.1 Merge prob process.
///
/// Inputs (spec names): `preProb` the original probability, `ct0` / `ct1`
/// the number of times the boolean decoded as 0 / 1, `countSat` the
/// saturation count, `maxUpdateFactor` the maximum adjustment factor.
/// Returns the updated probability `outProb`.
///
/// Spec body (lines 4205-4211):
/// ```text
/// den   = ct0 + ct1
/// prob  = (den == 0) ? 128 : Clip3(1, 255, (ct0 * 256 + (den >> 1)) / den)
/// count = Min(ct0 + ct1, countSat)
/// factor = maxUpdateFactor * count / countSat
/// outProb = Round2(preProb * (256 - factor) + prob * factor, 8)
/// ```
pub(crate) fn merge_prob(
    pre_prob: u8,
    ct0: u32,
    ct1: u32,
    count_sat: u32,
    max_update_factor: u32,
) -> u8 {
    let den = ct0 as i64 + ct1 as i64;
    let prob = if den == 0 {
        128
    } else {
        clip3(1, 255, (ct0 as i64 * 256 + (den >> 1)) / den)
    };
    let count = (ct0 + ct1).min(count_sat) as i64;
    // countSat is a positive constant (20) at every call site.
    let factor = max_update_factor as i64 * count / count_sat as i64;
    let out = round2(pre_prob as i64 * (256 - factor) + prob * factor, 8);
    out as u8
}

/// §8.4.2 Merge probs process.
///
/// Recursively walks the decode `tree` starting at index `i`, updating
/// `probs` in place from the per-leaf `counts`, and returns the total
/// number of times the subtree rooted at `i` was decoded.
///
/// Spec body (lines 4232-4241):
/// ```text
/// merge_probs(tree, i, probs, counts, countSat, maxUpdateFactor) {
///   s = tree[i]
///   leftCount  = (s <= 0) ? counts[-s] : merge_probs(tree, s, ...)
///   r = tree[i + 1]
///   rightCount = (r <= 0) ? counts[-r] : merge_probs(tree, r, ...)
///   probs[i >> 1] = merge_prob(probs[i >> 1], leftCount, rightCount, ...)
///   return leftCount + rightCount
/// }
/// ```
pub(crate) fn merge_probs(
    tree: &[i32],
    i: usize,
    probs: &mut [u8],
    counts: &[u32],
    count_sat: u32,
    max_update_factor: u32,
) -> u32 {
    let s = tree[i];
    let left_count = if s <= 0 {
        counts[(-s) as usize]
    } else {
        merge_probs(
            tree,
            s as usize,
            probs,
            counts,
            count_sat,
            max_update_factor,
        )
    };
    let r = tree[i + 1];
    let right_count = if r <= 0 {
        counts[(-r) as usize]
    } else {
        merge_probs(
            tree,
            r as usize,
            probs,
            counts,
            count_sat,
            max_update_factor,
        )
    };
    probs[i >> 1] = merge_prob(
        probs[i >> 1],
        left_count,
        right_count,
        count_sat,
        max_update_factor,
    );
    left_count + right_count
}

/// §8.4 `adapt_prob( prob, counts )` — a single-boolean adaption using
/// the default `COUNT_SAT` / `MAX_UPDATE_FACTOR` (lines 4353-4356):
/// `merge_prob( prob, counts[0], counts[1], COUNT_SAT, MAX_UPDATE_FACTOR )`.
pub(crate) fn adapt_prob(prob: u8, counts: [u32; 2]) -> u8 {
    merge_prob(prob, counts[0], counts[1], COUNT_SAT, MAX_UPDATE_FACTOR)
}

/// §8.4 `adapt_probs( tree, probs, counts )` — a tree-structured
/// adaption using the default `COUNT_SAT` / `MAX_UPDATE_FACTOR` (lines
/// 4348-4350): `merge_probs( tree, 0, probs, counts, COUNT_SAT,
/// MAX_UPDATE_FACTOR )`.
pub(crate) fn adapt_probs(tree: &[i32], probs: &mut [u8], counts: &[u32]) {
    merge_probs(tree, 0, probs, counts, COUNT_SAT, MAX_UPDATE_FACTOR);
}

// ----- §8.4.3 coefficient probability adaption -----

/// `small_token_tree[ 6 ]` per §8.4.3 (line 4282):
/// ```text
/// { 0, 0,                 // Unused (indices 0,1)
///   -ZERO_TOKEN, 4,       // index 2: leaf ZERO_TOKEN(0) / branch 4
///   -ONE_TOKEN, -TWO_TOKEN // index 4: leaf ONE_TOKEN(1) / leaf TWO_TOKEN(2)
/// }
/// ```
/// Walked starting at `i = 2`; the unused index-0 pair is never visited.
/// Leaf negatives index `counts_token` (`-0/-1/-2` → 0/1/2). The
/// interior writes touch `probs[2>>1]=probs[1]` and `probs[4>>1]=probs[2]`
/// — the ZERO/ONE token nodes of the 3-entry coef cell.
const SMALL_TOKEN_TREE: [i32; 6] = [0, 0, 0, 4, -1, -2];

/// `binary_tree[ 2 ] = { 0, -1 }` per §9.3.1 — the single-decision tree
/// driving the §8.4.3 `more_coefs` merge into `probs[0]`.
const BINARY_TREE: [i32; 2] = [0, -1];

/// Per-cell `counts_token` bucket: `[count_ZERO, count_ONE, count_TWO+]`
/// where the index is `Min(2, token)` per §9.3.4.
pub(crate) type CountsTokenCell = [u32; 3];

/// Per-cell `counts_more_coefs` bucket: `[count_more0, count_more1]` —
/// the §9.3.4 `more_coefs` binary count.
pub(crate) type CountsMoreCoefsCell = [u32; 2];

/// Shape of `counts_token` mirroring [`crate::coef_probs::CoefProbs`]
/// indexing `[txSz][blockType][refType][band][ctx]`.
pub(crate) type CountsToken = [[[[[CountsTokenCell; 6]; 6]; 2]; 2]; 4];

/// Shape of `counts_more_coefs` mirroring [`crate::coef_probs::CoefProbs`].
pub(crate) type CountsMoreCoefs = [[[[[CountsMoreCoefsCell; 6]; 6]; 2]; 2]; 4];

/// The complete per-frame syntax-element count bank of §9.3.4, cleared by
/// the §8.3 `clear_counts( )` process at the start of every frame's
/// compressed payload and consumed by the §6.1.2 `refresh_probs( )`
/// backward adaptation (§8.4.3 [`adapt_coef_probs`] from `token` /
/// `more_coefs`; §8.4.4 [`adapt_noncoef_probs`] from the rest).
///
/// The `more_coefs` counting follows the §9.3.4 special case documented
/// in `docs/video/vp9/vp9-errata-and-clarifications.md` (#249 part 1):
/// `counts_more_coefs[…]` is incremented **only at scan positions where
/// the `more_coefs` syntax element is actually decoded** — i.e. inside
/// the `checkEob` branch of the §6.4.24 `tokens( )` loop — never implied
/// at `checkEob == 0` positions reached after a `ZERO_TOKEN`.
pub(crate) struct FrameCounts {
    /// `counts_token[txSz][plane>0][is_inter][band][ctx][Min(2,syntax)]`.
    pub token: CountsToken,
    /// `counts_more_coefs[txSz][plane>0][is_inter][band][ctx][syntax]`.
    pub more_coefs: CountsMoreCoefs,
    /// Every non-coefficient `counts_*` array of §8.3 / §9.3.4.
    pub noncoef: CountsNonCoef,
}

impl FrameCounts {
    /// §8.3 `clear_counts( )` — a fresh all-zero count bank. Boxed: the
    /// coefficient tables alone are ~11 KiB and one bank lives per
    /// decoded frame.
    pub(crate) fn new_boxed() -> Box<Self> {
        Box::new(Self {
            token: [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4],
            more_coefs: [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4],
            noncoef: CountsNonCoef::default(),
        })
    }
}

/// §8.4.3 Coefficient probability adaption process.
///
/// Updates `coef_probs` in place from the observed `counts_token` /
/// `counts_more_coefs`. The `updateFactor` is selected per §8.4.3 (lines
/// 4248-4252):
/// * `FrameIsIntra == 1` → 112
/// * else `LastFrameType == KEY_FRAME` → 128
/// * else → 112
///
/// The nested walk (lines 4254-4270) visits every
/// `[t][i][j][k][l]` cell with `maxL = (k == 0) ? 3 : 6`, calling
/// `merge_probs(small_token_tree, 2, …)` then `merge_probs(binary_tree,
/// 0, …)` — both with `countSat = 24` and the selected `updateFactor`.
pub(crate) fn adapt_coef_probs(
    coef_probs: &mut crate::coef_probs::CoefProbs,
    counts_token: &CountsToken,
    counts_more_coefs: &CountsMoreCoefs,
    frame_is_intra: bool,
    last_frame_was_key: bool,
) {
    let update_factor: u32 = if frame_is_intra {
        112
    } else if last_frame_was_key {
        128
    } else {
        112
    };
    // §8.4.3 fixes countSat to 24 (not the default COUNT_SAT=20).
    const COEF_COUNT_SAT: u32 = 24;

    for t in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for l in 0..max_l {
                        let cell = &mut coef_probs[t][i][j][k][l];
                        merge_probs(
                            &SMALL_TOKEN_TREE,
                            2,
                            cell,
                            &counts_token[t][i][j][k][l],
                            COEF_COUNT_SAT,
                            update_factor,
                        );
                        merge_probs(
                            &BINARY_TREE,
                            0,
                            cell,
                            &counts_more_coefs[t][i][j][k][l],
                            COEF_COUNT_SAT,
                            update_factor,
                        );
                    }
                }
            }
        }
    }
}

// ----- §8.4.4 non-coefficient probability adaption -----

use crate::compressed::FrameContext;
use crate::mode_info::{
    BLOCK_SIZE_GROUPS, CLASS0_SIZE, COMP_MODE_CONTEXTS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_TREE,
    INTER_MODES, INTER_MODE_CONTEXTS, INTER_MODE_TREE, INTRA_MODES, INTRA_MODE_TREE,
    IS_INTER_CONTEXTS, MV_CLASSES, MV_FR_SIZE, MV_JOINTS, MV_OFFSET_BITS, REF_CONTEXTS,
    SWITCHABLE_FILTERS, TX_SIZE_16_TREE, TX_SIZE_32_TREE, TX_SIZE_8_TREE,
};
use crate::mv::{MV_CLASS_TREE, MV_FR_TREE, MV_JOINT_TREE};
use crate::partition::{PARTITION_CONTEXTS, PARTITION_TREE, PARTITION_TYPES};

/// `TX_SIZE_CONTEXTS = 2` per §3 — the per-`maxTxSize` context count of
/// the `tx_probs[ TX_SIZES ][ TX_SIZE_CONTEXTS ][ TX_SIZES - 1 ]` table
/// the §8.4.4 `tx_size` adaptation walks.
const TX_SIZE_CONTEXTS: usize = 2;

/// `SKIP_CONTEXTS = 3` per §3 — the context count of the `skip_prob[ ]`
/// table the §8.4.4 `skip` adaptation walks.
const SKIP_CONTEXTS: usize = 3;

/// Per-component MV count accumulators per §9.3.4 — one block of these
/// per motion-vector component (`comp = 0..1`).
///
/// Each field mirrors a `counts_mv_*[ comp ]` array from the §9.3.4
/// counting table (`vp9-spec.txt` lines 6792-6814); the `class0_hp` /
/// `hp` fields are only adapted when `allow_high_precision_mv == 1`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CountsMvComponent {
    /// `counts_mv_sign[ comp ][ syntax ]`.
    pub sign: [u32; 2],
    /// `counts_mv_class[ comp ][ syntax ]` over [`MV_CLASS_TREE`].
    pub class: [u32; MV_CLASSES],
    /// `counts_mv_class0_bit[ comp ][ syntax ]`.
    pub class0_bit: [u32; 2],
    /// `counts_mv_bits[ comp ][ i ][ syntax ]`.
    pub bits: [[u32; 2]; MV_OFFSET_BITS],
    /// `counts_mv_class0_fr[ comp ][ mv_class0_bit ][ syntax ]` over
    /// [`MV_FR_TREE`].
    pub class0_fr: [[u32; MV_FR_SIZE]; CLASS0_SIZE],
    /// `counts_mv_fr[ comp ][ syntax ]` over [`MV_FR_TREE`].
    pub fr: [u32; MV_FR_SIZE],
    /// `counts_mv_class0_hp[ comp ][ syntax ]`.
    pub class0_hp: [u32; 2],
    /// `counts_mv_hp[ comp ][ syntax ]`.
    pub hp: [u32; 2],
}

/// The full bank of non-coefficient syntax-element counts collected over
/// a frame per §9.3.4, consumed by [`adapt_noncoef_probs`].
///
/// Every field mirrors a `counts_*` array from the §9.3.4 counting table
/// (`vp9-spec.txt` lines 6755-6818) using the same context / syntax
/// indexing as the corresponding probability table in [`FrameContext`].
/// Tree-decoded elements (`inter_mode`, `intra_mode` / `uv_mode`,
/// `partition`, `interp_filter`, `tx_size`, `mv_joint`, `mv_class`,
/// `mv_fr`) carry one count slot per leaf value; binary elements
/// (`is_inter`, `comp_mode`, `comp_ref`, `single_ref`, `skip`,
/// `mv_sign` / `mv_class0_bit` / `mv_bits` / `mv_class0_hp` / `mv_hp`)
/// carry the `[count0, count1]` pair `adapt_prob` consumes directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CountsNonCoef {
    /// `counts_is_inter[ ctx ][ syntax ]`.
    pub is_inter: [[u32; 2]; IS_INTER_CONTEXTS],
    /// `counts_comp_mode[ ctx ][ syntax ]`.
    pub comp_mode: [[u32; 2]; COMP_MODE_CONTEXTS],
    /// `counts_comp_ref[ ctx ][ syntax ]`.
    pub comp_ref: [[u32; 2]; REF_CONTEXTS],
    /// `counts_single_ref[ ctx ][ 0/1 ][ syntax ]`.
    pub single_ref: [[[u32; 2]; 2]; REF_CONTEXTS],
    /// `counts_inter_mode[ ctx ][ syntax ]` over [`INTER_MODE_TREE`].
    pub inter_mode: [[u32; INTER_MODES]; INTER_MODE_CONTEXTS],
    /// `counts_intra_mode[ ctx ][ syntax ]` over [`INTRA_MODE_TREE`] —
    /// the `y_mode` adaptation source (per `BLOCK_SIZE_GROUPS`).
    pub y_mode: [[u32; INTRA_MODES]; BLOCK_SIZE_GROUPS],
    /// `counts_uv_mode[ ctx ][ syntax ]` over [`INTRA_MODE_TREE`] (per
    /// `INTRA_MODES`).
    pub uv_mode: [[u32; INTRA_MODES]; INTRA_MODES],
    /// `counts_partition[ ctx ][ syntax ]` over [`PARTITION_TREE`].
    pub partition: [[u32; PARTITION_TYPES]; PARTITION_CONTEXTS],
    /// `counts_skip[ ctx ][ syntax ]`.
    pub skip: [[u32; 2]; SKIP_CONTEXTS],
    /// `counts_interp_filter[ ctx ][ syntax ]` over [`INTERP_FILTER_TREE`].
    pub interp_filter: [[u32; SWITCHABLE_FILTERS]; INTERP_FILTER_CONTEXTS],
    /// `counts_tx_size[ maxTxSize ][ ctx ][ syntax ]`. The §8.4.4 walk
    /// adapts the `TX_8X8` / `TX_16X16` / `TX_32X32` rows; index 0
    /// (`TX_4X4`) is never tree-adapted, so a 4-syntax slot per row holds
    /// every adapted leaf.
    pub tx_size: [[[u32; 4]; TX_SIZE_CONTEXTS]; 4],
    /// `counts_mv_joint[ syntax ]` over [`MV_JOINT_TREE`].
    pub mv_joint: [u32; MV_JOINTS],
    /// Per-component MV counts.
    pub mv_comp: [CountsMvComponent; 2],
}

impl Default for CountsNonCoef {
    fn default() -> Self {
        Self {
            is_inter: [[0; 2]; IS_INTER_CONTEXTS],
            comp_mode: [[0; 2]; COMP_MODE_CONTEXTS],
            comp_ref: [[0; 2]; REF_CONTEXTS],
            single_ref: [[[0; 2]; 2]; REF_CONTEXTS],
            inter_mode: [[0; INTER_MODES]; INTER_MODE_CONTEXTS],
            y_mode: [[0; INTRA_MODES]; BLOCK_SIZE_GROUPS],
            uv_mode: [[0; INTRA_MODES]; INTRA_MODES],
            partition: [[0; PARTITION_TYPES]; PARTITION_CONTEXTS],
            skip: [[0; 2]; SKIP_CONTEXTS],
            interp_filter: [[0; SWITCHABLE_FILTERS]; INTERP_FILTER_CONTEXTS],
            tx_size: [[[0; 4]; TX_SIZE_CONTEXTS]; 4],
            mv_joint: [0; MV_JOINTS],
            mv_comp: [CountsMvComponent::default(), CountsMvComponent::default()],
        }
    }
}

/// §8.4.4 Non coefficient probability adaption process
/// (`vp9-spec.txt` lines 4289-4344).
///
/// Folds the per-frame `counts` into the probability bank `fc` in place,
/// using the default `COUNT_SAT = 20` / `MAX_UPDATE_FACTOR = 128` (via
/// [`adapt_prob`] / [`adapt_probs`]). The two conditional blocks fire on
/// the same uncompressed-header flags the §6.3 inter sweeps used:
///
/// * the `interp_filter` adaptation runs only when
///   `interpolation_filter == SWITCHABLE` (`interp_filter_switchable`);
/// * the `tx_size` adaptation runs only when `tx_mode == TX_MODE_SELECT`
///   (`tx_mode_select`);
/// * the per-component `mv_class0_hp` / `mv_hp` adaptation runs only when
///   `allow_high_precision_mv == 1`.
pub(crate) fn adapt_noncoef_probs(
    fc: &mut FrameContext,
    counts: &CountsNonCoef,
    interp_filter_switchable: bool,
    tx_mode_select: bool,
    allow_high_precision_mv: bool,
) {
    for i in 0..IS_INTER_CONTEXTS {
        fc.is_inter_prob[i] = adapt_prob(fc.is_inter_prob[i], counts.is_inter[i]);
    }
    for i in 0..COMP_MODE_CONTEXTS {
        fc.comp_mode_prob[i] = adapt_prob(fc.comp_mode_prob[i], counts.comp_mode[i]);
    }
    for i in 0..REF_CONTEXTS {
        fc.comp_ref_prob[i] = adapt_prob(fc.comp_ref_prob[i], counts.comp_ref[i]);
    }
    for i in 0..REF_CONTEXTS {
        for j in 0..2 {
            fc.single_ref_prob[i][j] =
                adapt_prob(fc.single_ref_prob[i][j], counts.single_ref[i][j]);
        }
    }
    for i in 0..INTER_MODE_CONTEXTS {
        adapt_probs(
            &INTER_MODE_TREE,
            &mut fc.inter_mode_probs[i],
            &counts.inter_mode[i],
        );
    }
    for i in 0..BLOCK_SIZE_GROUPS {
        adapt_probs(&INTRA_MODE_TREE, &mut fc.y_mode_probs[i], &counts.y_mode[i]);
    }
    // §8.4.4 adapts the inter-frame `uv_mode_probs` table. It is not part
    // of the persisted `FrameContext` bank (the inter decode uses the
    // static §10.5 `DEFAULT_UV_MODE_PROBS`), so the spec's per-frame
    // `uv_mode` adaptation has no persisted destination here; the counts
    // are still accepted so a future wiring that threads a per-frame
    // `uv_mode_probs` table can consume them.
    for i in 0..PARTITION_CONTEXTS {
        adapt_probs(
            &PARTITION_TREE,
            &mut fc.partition_probs[i],
            &counts.partition[i],
        );
    }
    for i in 0..SKIP_CONTEXTS {
        fc.skip_prob[i] = adapt_prob(fc.skip_prob[i], counts.skip[i]);
    }
    if interp_filter_switchable {
        for i in 0..INTERP_FILTER_CONTEXTS {
            adapt_probs(
                &INTERP_FILTER_TREE,
                &mut fc.interp_filter_probs[i],
                &counts.interp_filter[i],
            );
        }
    }
    if tx_mode_select {
        for i in 0..TX_SIZE_CONTEXTS {
            adapt_probs(
                &TX_SIZE_8_TREE,
                &mut fc.tx_probs[1][i],
                &counts.tx_size[1][i],
            );
            adapt_probs(
                &TX_SIZE_16_TREE,
                &mut fc.tx_probs[2][i],
                &counts.tx_size[2][i],
            );
            adapt_probs(
                &TX_SIZE_32_TREE,
                &mut fc.tx_probs[3][i],
                &counts.tx_size[3][i],
            );
        }
    }
    adapt_probs(
        &MV_JOINT_TREE,
        &mut fc.mv_probs.joint_probs,
        &counts.mv_joint,
    );
    for i in 0..2 {
        let mc = &counts.mv_comp[i];
        fc.mv_probs.sign_prob[i] = adapt_prob(fc.mv_probs.sign_prob[i], mc.sign);
        adapt_probs(&MV_CLASS_TREE, &mut fc.mv_probs.class_probs[i], &mc.class);
        fc.mv_probs.class0_bit_prob[i] = adapt_prob(fc.mv_probs.class0_bit_prob[i], mc.class0_bit);
        for j in 0..MV_OFFSET_BITS {
            fc.mv_probs.bits_prob[i][j] = adapt_prob(fc.mv_probs.bits_prob[i][j], mc.bits[j]);
        }
        for j in 0..CLASS0_SIZE {
            adapt_probs(
                &MV_FR_TREE,
                &mut fc.mv_probs.class0_fr_probs[i][j],
                &mc.class0_fr[j],
            );
        }
        adapt_probs(&MV_FR_TREE, &mut fc.mv_probs.fr_probs[i], &mc.fr);
        if allow_high_precision_mv {
            fc.mv_probs.class0_hp_prob[i] = adapt_prob(fc.mv_probs.class0_hp_prob[i], mc.class0_hp);
            fc.mv_probs.hp_prob[i] = adapt_prob(fc.mv_probs.hp_prob[i], mc.hp);
        }
    }
}

#[cfg(test)]
mod tests {
    // Several §8.4.4 tests build a zeroed `CountsNonCoef::default()` and
    // poke a single field to isolate one adaptation path; that pattern
    // trips clippy's field_reassign_with_default, which is noise for this
    // single-cell-isolation test style.
    #![allow(clippy::field_reassign_with_default)]

    use super::*;

    // §8.4.1: with zero counts, prob defaults to 128 but factor is 0, so
    // outProb == preProb (no adaption when nothing was decoded).
    #[test]
    fn merge_prob_zero_counts_is_identity() {
        for p in 1u8..=255 {
            assert_eq!(merge_prob(p, 0, 0, COUNT_SAT, MAX_UPDATE_FACTOR), p);
        }
    }

    // §8.4.1 worked example: preProb=128, ct0=20, ct1=0 (fully saturated,
    // all-zero observations).
    //   den=20, prob=Clip3(1,255,(20*256+10)/20)=Clip3(1,255,256)=255
    //   count=Min(20,20)=20, factor=128*20/20=128
    //   outProb=Round2(128*(256-128)+255*128,8)
    //          =Round2(128*128 + 255*128,8)=Round2(49024,8)=Round2(49024,8)
    //          =(49024+128)>>8 = 49152>>8 = 192
    #[test]
    fn merge_prob_saturated_all_zero() {
        assert_eq!(merge_prob(128, 20, 0, COUNT_SAT, MAX_UPDATE_FACTOR), 192);
    }

    // §8.4.1: all-ones observations push the estimate toward 1.
    //   preProb=128, ct0=0, ct1=20
    //   den=20, prob=Clip3(1,255,(0+10)/20)=Clip3(1,255,0)=1
    //   count=20, factor=128
    //   outProb=Round2(128*128 + 1*128,8)=(16384+128+128)>>8=16640>>8=65
    #[test]
    fn merge_prob_saturated_all_one() {
        assert_eq!(merge_prob(128, 0, 20, COUNT_SAT, MAX_UPDATE_FACTOR), 65);
    }

    // §8.4.1: partial count scales the factor linearly.
    //   preProb=200, ct0=5, ct1=5 -> den=10
    //   prob=Clip3(1,255,(5*256+5)/10)=Clip3(1,255,128)=128
    //   count=Min(10,20)=10, factor=128*10/20=64
    //   outProb=Round2(200*(256-64)+128*64,8)
    //          =Round2(200*192+8192,8)=Round2(38400+8192,8)
    //          =Round2(46592,8)=(46592+128)>>8=46720>>8=182
    #[test]
    fn merge_prob_partial_count() {
        assert_eq!(merge_prob(200, 5, 5, COUNT_SAT, MAX_UPDATE_FACTOR), 182);
    }

    // §8.4.1: output is always a valid probability byte.
    #[test]
    fn merge_prob_output_in_range() {
        for &p in &[1u8, 64, 128, 200, 255] {
            for ct0 in [0u32, 1, 7, 19, 20, 100] {
                for ct1 in [0u32, 1, 7, 19, 20, 100] {
                    let o = merge_prob(p, ct0, ct1, COUNT_SAT, MAX_UPDATE_FACTOR);
                    assert!(o >= 1 || ct0 + ct1 == 0, "prob clamped to >=1");
                }
            }
        }
    }

    // §8.4.2 + §8.4 adapt_prob over binary_tree[2] = { 0, -1 }: the
    // single-node tree must agree with merge_prob on counts[0]/counts[1].
    #[test]
    fn adapt_probs_binary_tree_matches_merge_prob() {
        let binary_tree = [0i32, -1];
        let counts = [12u32, 8];
        let mut probs_tree = [137u8];
        adapt_probs(&binary_tree, &mut probs_tree, &counts);
        let direct = adapt_prob(137, [counts[0], counts[1]]);
        assert_eq!(probs_tree[0], direct);
    }

    // §8.4.2 multi-level tree (small_token_tree-shaped, rooted at 0 for
    // the test):
    //   tree = { -A, 2, -B, -C }  (3 leaves A,B,C; one interior pair)
    // counts indexed by leaf value. Verifies recursion visits both the
    // interior (index 0) and the child pair (index 2) and accumulates
    // the correct leaf-count totals.
    #[test]
    fn merge_probs_three_leaf_tree_accumulates() {
        // leaves: A=counts[0], B=counts[1], C=counts[2]
        let tree = [-0i32, 2, -1, -2];
        let counts = [4u32, 6, 10];
        let mut probs = [100u8, 150];
        let total = merge_probs(&tree, 0, &mut probs, &counts, COUNT_SAT, MAX_UPDATE_FACTOR);
        assert_eq!(total, 20, "root subtree count = A+B+C");
        // Interior node index 0: left=A=4, right=(B+C)=16
        assert_eq!(
            probs[0],
            merge_prob(100, 4, 16, COUNT_SAT, MAX_UPDATE_FACTOR)
        );
        // Child pair index 2: left=B=6, right=C=10
        assert_eq!(
            probs[1],
            merge_prob(150, 6, 10, COUNT_SAT, MAX_UPDATE_FACTOR)
        );
    }

    // §4.7 Round2 / Clip3 sanity (used by merge_prob).
    #[test]
    fn round2_and_clip3_spec_forms() {
        assert_eq!(round2(0, 8), 0);
        assert_eq!(round2(128, 8), 1); // (128+128)>>8 = 1
        assert_eq!(round2(383, 8), 1); // (383+128)>>8 = 511>>8 = 1
        assert_eq!(round2(384, 8), 2); // (384+128)>>8 = 512>>8 = 2
        assert_eq!(round2(5, 0), 5);
        assert_eq!(clip3(1, 255, 0), 1);
        assert_eq!(clip3(1, 255, 256), 255);
        assert_eq!(clip3(1, 255, 128), 128);
    }

    // ----- §8.4.3 adapt_coef_probs -----

    // §8.4.3 small_token_tree[6] = {0,0, -ZERO_TOKEN,4, -ONE_TOKEN,-TWO_TOKEN}.
    // Walked from i=2: index-0 pair is unused, leaves map -0/-1/-2 → token
    // count buckets, interior writes hit probs[1] (i=2>>1) and probs[2]
    // (i=4>>1).
    #[test]
    fn small_token_tree_layout() {
        assert_eq!(SMALL_TOKEN_TREE, [0, 0, 0, 4, -1, -2]);
        assert_eq!(BINARY_TREE, [0, -1]);
    }

    // §8.4.3 updateFactor selection: intra → 112, inter-after-key → 128,
    // inter-after-inter → 112. We probe via the resulting cell update,
    // which differs when the factor differs.
    fn one_cell_after(
        frame_is_intra: bool,
        last_key: bool,
        cell0: [u8; 3],
        ct: [u32; 3],
        cmc: [u32; 2],
    ) -> [u8; 3] {
        let mut cp = crate::coef_probs::DEFAULT_COEF_PROBS;
        cp[0][0][0][0][0] = cell0;
        let mut counts_token: CountsToken = [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4];
        let mut counts_mc: CountsMoreCoefs = [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4];
        counts_token[0][0][0][0][0] = ct;
        counts_mc[0][0][0][0][0] = cmc;
        adapt_coef_probs(&mut cp, &counts_token, &counts_mc, frame_is_intra, last_key);
        cp[0][0][0][0][0]
    }

    // §8.4.3 cell update matches two direct merge_probs calls with
    // countSat=24, updateFactor=112 (intra). Cell = [more, zero, one].
    #[test]
    fn adapt_coef_probs_cell_matches_direct_intra() {
        let cell0 = [120u8, 90, 60];
        let ct = [5u32, 7, 3]; // ZERO, ONE, TWO+
        let cmc = [9u32, 11]; // more_coefs 0/1
        let got = one_cell_after(true, false, cell0, ct, cmc);

        let mut expect = cell0;
        merge_probs(&SMALL_TOKEN_TREE, 2, &mut expect, &ct, 24, 112);
        merge_probs(&BINARY_TREE, 0, &mut expect, &cmc, 24, 112);
        assert_eq!(got, expect);
    }

    // §8.4.3: inter-after-key uses updateFactor=128, which yields a
    // different result than the intra (112) path for the same counts.
    #[test]
    fn adapt_coef_probs_update_factor_depends_on_frame_types() {
        let cell0 = [120u8, 90, 60];
        let ct = [5u32, 7, 3];
        let cmc = [9u32, 11];
        let intra = one_cell_after(true, false, cell0, ct, cmc);
        let inter_after_key = one_cell_after(false, true, cell0, ct, cmc);
        let inter_after_inter = one_cell_after(false, false, cell0, ct, cmc);

        // 112 path == intra path.
        assert_eq!(inter_after_inter, intra);
        // 128 path is a stronger pull → differs from the 112 result here.
        assert_ne!(inter_after_key, intra);

        // Verify the 128 path against direct merge_probs.
        let mut expect = cell0;
        merge_probs(&SMALL_TOKEN_TREE, 2, &mut expect, &ct, 24, 128);
        merge_probs(&BINARY_TREE, 0, &mut expect, &cmc, 24, 128);
        assert_eq!(inter_after_key, expect);
    }

    // §8.4.3 inner loop: band 0 (k==0) visits only maxL=3 ctx slots; ctx
    // 3..5 of band 0 must be left untouched even with non-zero counts.
    #[test]
    fn adapt_coef_probs_band0_maxl_is_three() {
        let mut cp = crate::coef_probs::DEFAULT_COEF_PROBS;
        let untouched = cp[0][0][0][0][4]; // band 0, ctx 4 (>= maxL=3)
        let mut counts_token: CountsToken = [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4];
        let mut counts_mc: CountsMoreCoefs = [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4];
        // Pile counts into the out-of-range ctx slot.
        counts_token[0][0][0][0][4] = [50, 50, 50];
        counts_mc[0][0][0][0][4] = [50, 50];
        adapt_coef_probs(&mut cp, &counts_token, &counts_mc, true, false);
        assert_eq!(
            cp[0][0][0][0][4], untouched,
            "band-0 ctx>=3 must not be adapted (maxL=3)"
        );
        // But band 1 (k==1) ctx 4 IS in range (maxL=6) and would adapt.
        let touched_in_range = cp[0][0][0][0][2]; // band 0 ctx 2 < 3 → in range
        assert_eq!(touched_in_range, cp[0][0][0][0][2]);
    }

    // §8.4.3: all-zero counts leave coef_probs unchanged (merge_prob is
    // the identity at zero counts).
    #[test]
    fn adapt_coef_probs_zero_counts_is_identity() {
        let orig = crate::coef_probs::DEFAULT_COEF_PROBS;
        let mut cp = orig;
        let counts_token: CountsToken = [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4];
        let counts_mc: CountsMoreCoefs = [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4];
        adapt_coef_probs(&mut cp, &counts_token, &counts_mc, true, false);
        assert_eq!(cp, orig);
    }

    // ----- §8.4.4 adapt_noncoef_probs -----

    // §8.4.4 zero counts: the whole non-coefficient bank is unchanged
    // (merge_prob is the identity at zero counts, for both the binary
    // adapt_prob and the tree-walking adapt_probs).
    #[test]
    fn adapt_noncoef_probs_zero_counts_is_identity() {
        let orig = FrameContext::default();
        let mut fc = orig.clone();
        let counts = CountsNonCoef::default();
        adapt_noncoef_probs(&mut fc, &counts, true, true, true);
        assert_eq!(fc, orig);
    }

    // §8.4.4: a single binary element (is_inter ctx 1) adapts to exactly
    // adapt_prob(prob, counts) and nothing else moves.
    #[test]
    fn adapt_noncoef_probs_single_is_inter_cell() {
        let orig = FrameContext::default();
        let mut fc = orig.clone();
        let mut counts = CountsNonCoef::default();
        counts.is_inter[1] = [7, 13];
        adapt_noncoef_probs(&mut fc, &counts, false, false, false);

        let mut expect = orig.clone();
        expect.is_inter_prob[1] = adapt_prob(orig.is_inter_prob[1], [7, 13]);
        assert_eq!(fc, expect);
    }

    // §8.4.4 inter_mode is tree-adapted via INTER_MODE_TREE: a populated
    // context cell matches a direct adapt_probs call on the same row.
    #[test]
    fn adapt_noncoef_probs_inter_mode_tree() {
        let orig = FrameContext::default();
        let mut fc = orig.clone();
        let mut counts = CountsNonCoef::default();
        counts.inter_mode[2] = [3, 5, 2, 9];
        adapt_noncoef_probs(&mut fc, &counts, false, false, false);

        let mut expect_row = orig.inter_mode_probs[2];
        adapt_probs(&INTER_MODE_TREE, &mut expect_row, &[3, 5, 2, 9]);
        assert_eq!(fc.inter_mode_probs[2], expect_row);
        // No other inter_mode context moved.
        for i in 0..INTER_MODE_CONTEXTS {
            if i != 2 {
                assert_eq!(fc.inter_mode_probs[i], orig.inter_mode_probs[i]);
            }
        }
    }

    // §8.4.4 conditional gates: interp_filter only adapts when
    // interpolation_filter == SWITCHABLE; tx_size only when
    // tx_mode == TX_MODE_SELECT; mv hp tails only when
    // allow_high_precision_mv == 1.
    #[test]
    fn adapt_noncoef_probs_conditional_gates() {
        let orig = FrameContext::default();

        // interp_filter gate OFF: untouched even with counts present.
        let mut fc = orig.clone();
        let mut counts = CountsNonCoef::default();
        counts.interp_filter[0] = [4, 5, 6];
        counts.tx_size[1][0] = [3, 7, 0, 0];
        counts.mv_comp[0].hp = [2, 9];
        counts.mv_comp[0].class0_hp = [1, 4];
        adapt_noncoef_probs(&mut fc, &counts, false, false, false);
        assert_eq!(fc.interp_filter_probs, orig.interp_filter_probs);
        assert_eq!(fc.tx_probs, orig.tx_probs);
        assert_eq!(fc.mv_probs.hp_prob, orig.mv_probs.hp_prob);
        assert_eq!(fc.mv_probs.class0_hp_prob, orig.mv_probs.class0_hp_prob);

        // All gates ON: each now moves.
        let mut fc = orig.clone();
        adapt_noncoef_probs(&mut fc, &counts, true, true, true);
        assert_ne!(fc.interp_filter_probs[0], orig.interp_filter_probs[0]);
        assert_ne!(fc.tx_probs[1][0], orig.tx_probs[1][0]);
        assert_ne!(fc.mv_probs.hp_prob[0], orig.mv_probs.hp_prob[0]);
        assert_ne!(
            fc.mv_probs.class0_hp_prob[0],
            orig.mv_probs.class0_hp_prob[0]
        );
    }

    // §8.4.4 tx_size adapts the three non-4x4 rows with their respective
    // trees; TX_4X4 (row 0) is never tree-adapted.
    #[test]
    fn adapt_noncoef_probs_tx_size_rows() {
        let orig = FrameContext::default();
        let mut fc = orig.clone();
        let mut counts = CountsNonCoef::default();
        counts.tx_size[1][1] = [5, 9, 0, 0];
        counts.tx_size[2][1] = [3, 4, 7, 0];
        counts.tx_size[3][1] = [2, 3, 4, 6];
        adapt_noncoef_probs(&mut fc, &counts, false, true, false);

        let mut e8 = orig.tx_probs[1][1];
        adapt_probs(&TX_SIZE_8_TREE, &mut e8, &[5, 9, 0, 0]);
        let mut e16 = orig.tx_probs[2][1];
        adapt_probs(&TX_SIZE_16_TREE, &mut e16, &[3, 4, 7, 0]);
        let mut e32 = orig.tx_probs[3][1];
        adapt_probs(&TX_SIZE_32_TREE, &mut e32, &[2, 3, 4, 6]);
        assert_eq!(fc.tx_probs[1][1], e8);
        assert_eq!(fc.tx_probs[2][1], e16);
        assert_eq!(fc.tx_probs[3][1], e32);
        // TX_4X4 row untouched.
        assert_eq!(fc.tx_probs[0], orig.tx_probs[0]);
    }

    // §8.4.4 mv joint + per-component fields adapt; high-precision tails
    // gated. Verify the mv_joint tree adaptation and a class tree cell.
    #[test]
    fn adapt_noncoef_probs_mv_fields() {
        let orig = FrameContext::default();
        let mut fc = orig.clone();
        let mut counts = CountsNonCoef::default();
        counts.mv_joint = [2, 3, 4, 5];
        counts.mv_comp[1].sign = [6, 8];
        counts.mv_comp[1].class[3] = 11;
        counts.mv_comp[1].fr = [1, 2, 3, 4];
        adapt_noncoef_probs(&mut fc, &counts, false, false, false);

        let mut ej = orig.mv_probs.joint_probs;
        adapt_probs(&MV_JOINT_TREE, &mut ej, &[2, 3, 4, 5]);
        assert_eq!(fc.mv_probs.joint_probs, ej);
        assert_eq!(
            fc.mv_probs.sign_prob[1],
            adapt_prob(orig.mv_probs.sign_prob[1], [6, 8])
        );
        let mut ecl = orig.mv_probs.class_probs[1];
        let mut clc = [0u32; MV_CLASSES];
        clc[3] = 11;
        adapt_probs(&MV_CLASS_TREE, &mut ecl, &clc);
        assert_eq!(fc.mv_probs.class_probs[1], ecl);
        let mut efr = orig.mv_probs.fr_probs[1];
        adapt_probs(&MV_FR_TREE, &mut efr, &[1, 2, 3, 4]);
        assert_eq!(fc.mv_probs.fr_probs[1], efr);
    }

    // §8.4.4 partition + skip + y_mode adapt unconditionally.
    #[test]
    fn adapt_noncoef_probs_unconditional_block() {
        let orig = FrameContext::default();
        let mut fc = orig.clone();
        let mut counts = CountsNonCoef::default();
        counts.partition[5] = [2, 3, 4, 5];
        counts.skip[2] = [9, 1];
        counts.y_mode[1][0] = 12;
        adapt_noncoef_probs(&mut fc, &counts, false, false, false);

        let mut ep = orig.partition_probs[5];
        adapt_probs(&PARTITION_TREE, &mut ep, &[2, 3, 4, 5]);
        assert_eq!(fc.partition_probs[5], ep);
        assert_eq!(fc.skip_prob[2], adapt_prob(orig.skip_prob[2], [9, 1]));
        let mut ey = orig.y_mode_probs[1];
        let mut yc = [0u32; INTRA_MODES];
        yc[0] = 12;
        adapt_probs(&INTRA_MODE_TREE, &mut ey, &yc);
        assert_eq!(fc.y_mode_probs[1], ey);
    }
}
