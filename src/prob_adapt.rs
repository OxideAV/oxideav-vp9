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

#[cfg(test)]
mod tests {
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
}
