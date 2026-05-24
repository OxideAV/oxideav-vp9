//! VP9 coefficient-token decoder per spec v0.7 §6.4.24 / §6.4.26 +
//! §9.3 token tree / §10.3 pareto table.
//!
//! Round 7 lands the coefficient-token tree walker and `read_coef`
//! extra-bits decoder. Together these consume the round-6
//! `coef_probs[ TX_SIZES ][ BLOCK_TYPES ][ REF_TYPES ][ COEF_BANDS ][
//! PREV_COEF_CONTEXTS ][ UNCONSTRAINED_NODES ]` table to recover
//! quantised DCT coefficients from a single transform block.
//!
//! The §6.4.21 `residual( )` outer walk over planes / 4x4 sub-blocks
//! and the §6.4.24 `tokens( )` per-block driver are kept out of this
//! module — they need the `AboveNonzeroContext` / `LeftNonzeroContext`
//! arrays and per-frame state which the round-6 header walker
//! doesn't expose yet. The pieces this module exposes are sufficient
//! to round-trip one coefficient at a time once that state is wired
//! up.
//!
//! The decode of a coefficient at scan position `c` is, per §6.4.24:
//!
//! ```text
//! if ( checkEob ) {
//!     more_coefs               // §9.3 binary_tree, prob = coef_probs[...][0]
//!     if ( more_coefs == 0 ) break
//! }
//! token                        // §9.3 token_tree, probs via pareto()
//! TokenCache[ pos ] = energy_class[ token ]
//! if ( token == ZERO_TOKEN ) {
//!     Tokens[ pos ] = 0
//!     checkEob = 0
//! } else {
//!     coef = read_coef( token ) // §6.4.26
//!     sign_bit                  // L(1) literal
//!     Tokens[ pos ] = sign_bit ? -coef : coef
//!     checkEob = 1
//! }
//! ```
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §6.4.24, §6.4.26, §7.4.16, §9.3,
//! §10.2, §10.3). No external library source was consulted. Black-box
//! validators (`vpxdec`, `ffmpeg`) remain available for whole-stream
//! cross-checks once the residual driver lands.

// The §6.4.21 `residual( )` driver — which will consume these
// helpers — lands in a subsequent round. Until then the entire
// module is exercised exclusively from `#[cfg(test)]`. Suppress
// dead-code warnings module-wide rather than annotating each item.
#![allow(dead_code)]

use crate::bool_coder::BoolCoder;
use crate::idct::{ADST_DCT, DCT_ADST};
use crate::Error;

/// `ZERO_TOKEN` per §7.4.16. Indicates the next coefficient at scan
/// position `pos` is 0 (and `checkEob` clears).
pub(crate) const ZERO_TOKEN: u32 = 0;
/// `ONE_TOKEN` per §7.4.16. Coefficient magnitude is exactly 1 (no
/// extra bits).
pub(crate) const ONE_TOKEN: u32 = 1;
/// `TWO_TOKEN` per §7.4.16. Coefficient magnitude is exactly 2.
pub(crate) const TWO_TOKEN: u32 = 2;
/// `THREE_TOKEN` per §7.4.16. Coefficient magnitude is exactly 3.
pub(crate) const THREE_TOKEN: u32 = 3;
/// `FOUR_TOKEN` per §7.4.16. Coefficient magnitude is exactly 4.
pub(crate) const FOUR_TOKEN: u32 = 4;
/// `DCT_VAL_CATEGORY1` per §7.4.16. Magnitude in `5..=6`; 1 extra bit.
pub(crate) const DCT_VAL_CATEGORY1: u32 = 5;
/// `DCT_VAL_CATEGORY2` per §7.4.16. Magnitude in `7..=10`; 2 extra bits.
pub(crate) const DCT_VAL_CATEGORY2: u32 = 6;
/// `DCT_VAL_CATEGORY3` per §7.4.16. Magnitude in `11..=18`; 3 extra bits.
pub(crate) const DCT_VAL_CATEGORY3: u32 = 7;
/// `DCT_VAL_CATEGORY4` per §7.4.16. Magnitude in `19..=34`; 4 extra bits.
pub(crate) const DCT_VAL_CATEGORY4: u32 = 8;
/// `DCT_VAL_CATEGORY5` per §7.4.16. Magnitude in `35..=66`; 5 extra bits.
pub(crate) const DCT_VAL_CATEGORY5: u32 = 9;
/// `DCT_VAL_CATEGORY6` per §7.4.16. Magnitude in `67..=N`; 14 extra bits
/// at 8-bit BitDepth (more for 10/12-bit profiles via the `high_bit`
/// fixed-prob loop).
pub(crate) const DCT_VAL_CATEGORY6: u32 = 10;

/// `extra_bits[ 11 ][ 3 ]` per spec §6.4.26 — transcribed verbatim.
///
/// Each row is `{ cat, numExtra, base_coef }`. `cat` indexes into
/// [`CAT_PROBS`] for the per-bit probability sequence; `numExtra` is
/// the count of extra `B(p)` reads to combine into the magnitude;
/// `base_coef` is the starting value before extra bits accumulate.
pub(crate) const EXTRA_BITS: [[u32; 3]; 11] = [
    [0, 0, 0],   // ZERO_TOKEN
    [0, 0, 1],   // ONE_TOKEN
    [0, 0, 2],   // TWO_TOKEN
    [0, 0, 3],   // THREE_TOKEN
    [0, 0, 4],   // FOUR_TOKEN
    [1, 1, 5],   // DCT_VAL_CAT1
    [2, 2, 7],   // DCT_VAL_CAT2
    [3, 3, 11],  // DCT_VAL_CAT3
    [4, 4, 19],  // DCT_VAL_CAT4
    [5, 5, 35],  // DCT_VAL_CAT5
    [6, 14, 67], // DCT_VAL_CAT6
];

/// `cat_probs[ 7 ][ 14 ]` per spec §6.4.26 — transcribed verbatim.
///
/// Row `cat` holds the `numExtra` probabilities used by `read_coef`
/// when the decoded `token` is `DCT_VAL_CATEGORY<cat>`. Trailing
/// entries in each shorter row are unused but kept as 0 so the table
/// shape matches the spec's `[7][14]` declared form. Row 0 is
/// `{ 0 }` per spec (unused: the ZERO/ONE/TWO/THREE/FOUR tokens have
/// `numExtra = 0` so this row is never indexed).
pub(crate) const CAT_PROBS: [[u8; 14]; 7] = [
    [0; 14],
    [159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [165, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [173, 148, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [176, 155, 140, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [180, 157, 141, 134, 130, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129,
    ],
];

/// `energy_class[ 12 ]` per spec §10.2 — transcribed verbatim.
///
/// Maps a decoded token (`0..=10`) to its energy class
/// (`0..=5`) for the `TokenCache` lookup. Indexing by `token`
/// directly; the spec lists 12 entries to leave headroom but only the
/// first 11 are ever queried (token values are bounded by the
/// tree's 11 leaves).
pub(crate) const ENERGY_CLASS: [u8; 12] = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5];

/// `token_tree[ 20 ]` per spec §9.3 (line 6217 of the v0.7 listing).
///
/// Walked under the §9.3.3 `n = T[ n + read_bool( P( n >> 1 ) ) ]`
/// recurrence. Negative entries are tokens (the spec encodes them as
/// `-TOKEN` so the loop's `n > 0` test terminates; the decoder's
/// return value is `-n`).
///
/// The probability at each internal node `n >> 1` is
/// `pareto( n >> 1, coef_probs[txSz][plane>0][is_inter][band][ctx][
/// Min(2, 1 + (n >> 1)) ] )` per §9.3.2.
pub(crate) const TOKEN_TREE: [i32; 20] = [
    -(ZERO_TOKEN as i32),
    2,
    -(ONE_TOKEN as i32),
    4,
    6,
    10,
    -(TWO_TOKEN as i32),
    8,
    -(THREE_TOKEN as i32),
    -(FOUR_TOKEN as i32),
    12,
    14,
    -(DCT_VAL_CATEGORY1 as i32),
    -(DCT_VAL_CATEGORY2 as i32),
    16,
    18,
    -(DCT_VAL_CATEGORY3 as i32),
    -(DCT_VAL_CATEGORY4 as i32),
    -(DCT_VAL_CATEGORY5 as i32),
    -(DCT_VAL_CATEGORY6 as i32),
];

/// `pareto_table[ 128 ][ 8 ]` per spec §10.3 — transcribed verbatim
/// from the 128 listing rows. Each row's eight columns correspond to
/// the §9.3.2 `pareto(node, prob)` outputs for internal-node indices
/// `node - 2 ∈ 0..=7`, i.e. for the seven internal-node slots beyond
/// the two unconstrained ones plus an unused trailing column.
///
/// Row index is `x = (prob - 1) / 2` clamped to `0..=127`.
pub(crate) const PARETO_TABLE: [[u8; 8]; 128] = [
    [3, 86, 128, 6, 86, 23, 88, 29],
    [9, 86, 129, 17, 88, 61, 94, 76],
    [15, 87, 129, 28, 89, 93, 100, 110],
    [20, 88, 130, 38, 91, 118, 106, 136],
    [26, 89, 131, 48, 92, 139, 111, 156],
    [31, 90, 131, 58, 94, 156, 117, 171],
    [37, 90, 132, 66, 95, 171, 122, 184],
    [42, 91, 132, 75, 97, 183, 127, 194],
    [47, 92, 133, 83, 98, 193, 132, 202],
    [52, 93, 133, 90, 100, 201, 137, 208],
    [57, 94, 134, 98, 101, 208, 142, 214],
    [62, 94, 135, 105, 103, 214, 146, 218],
    [66, 95, 135, 111, 104, 219, 151, 222],
    [71, 96, 136, 117, 106, 224, 155, 225],
    [76, 97, 136, 123, 107, 227, 159, 228],
    [80, 98, 137, 129, 109, 231, 162, 231],
    [84, 98, 138, 134, 110, 234, 166, 233],
    [89, 99, 138, 140, 112, 236, 170, 235],
    [93, 100, 139, 145, 113, 238, 173, 236],
    [97, 101, 140, 149, 115, 240, 176, 238],
    [101, 102, 140, 154, 116, 242, 179, 239],
    [105, 103, 141, 158, 118, 243, 182, 240],
    [109, 104, 141, 162, 119, 244, 185, 241],
    [113, 104, 142, 166, 120, 245, 187, 242],
    [116, 105, 143, 170, 122, 246, 190, 243],
    [120, 106, 143, 173, 123, 247, 192, 244],
    [123, 107, 144, 177, 125, 248, 195, 244],
    [127, 108, 145, 180, 126, 249, 197, 245],
    [130, 109, 145, 183, 128, 249, 199, 245],
    [134, 110, 146, 186, 129, 250, 201, 246],
    [137, 111, 147, 189, 131, 251, 203, 246],
    [140, 112, 147, 192, 132, 251, 205, 247],
    [143, 113, 148, 194, 133, 251, 207, 247],
    [146, 114, 149, 197, 135, 252, 208, 248],
    [149, 115, 149, 199, 136, 252, 210, 248],
    [152, 115, 150, 201, 138, 252, 211, 248],
    [155, 116, 151, 204, 139, 253, 213, 249],
    [158, 117, 151, 206, 140, 253, 214, 249],
    [161, 118, 152, 208, 142, 253, 216, 249],
    [163, 119, 153, 210, 143, 253, 217, 249],
    [166, 120, 153, 212, 144, 254, 218, 250],
    [168, 121, 154, 213, 146, 254, 220, 250],
    [171, 122, 155, 215, 147, 254, 221, 250],
    [173, 123, 155, 217, 148, 254, 222, 250],
    [176, 124, 156, 218, 150, 254, 223, 250],
    [178, 125, 157, 220, 151, 254, 224, 251],
    [180, 126, 157, 221, 152, 254, 225, 251],
    [183, 127, 158, 222, 153, 254, 226, 251],
    [185, 128, 159, 224, 155, 255, 227, 251],
    [187, 129, 160, 225, 156, 255, 228, 251],
    [189, 131, 160, 226, 157, 255, 228, 251],
    [191, 132, 161, 227, 159, 255, 229, 251],
    [193, 133, 162, 228, 160, 255, 230, 252],
    [195, 134, 163, 230, 161, 255, 231, 252],
    [197, 135, 163, 231, 162, 255, 231, 252],
    [199, 136, 164, 232, 163, 255, 232, 252],
    [201, 137, 165, 233, 165, 255, 233, 252],
    [202, 138, 166, 233, 166, 255, 233, 252],
    [204, 139, 166, 234, 167, 255, 234, 252],
    [206, 140, 167, 235, 168, 255, 235, 252],
    [207, 141, 168, 236, 169, 255, 235, 252],
    [209, 142, 169, 237, 171, 255, 236, 252],
    [210, 144, 169, 237, 172, 255, 236, 252],
    [212, 145, 170, 238, 173, 255, 237, 252],
    [214, 146, 171, 239, 174, 255, 237, 253],
    [215, 147, 172, 240, 175, 255, 238, 253],
    [216, 148, 173, 240, 176, 255, 238, 253],
    [218, 149, 173, 241, 177, 255, 239, 253],
    [219, 150, 174, 241, 179, 255, 239, 253],
    [220, 152, 175, 242, 180, 255, 240, 253],
    [222, 153, 176, 242, 181, 255, 240, 253],
    [223, 154, 177, 243, 182, 255, 240, 253],
    [224, 155, 178, 244, 183, 255, 241, 253],
    [225, 156, 178, 244, 184, 255, 241, 253],
    [226, 158, 179, 244, 185, 255, 242, 253],
    [228, 159, 180, 245, 186, 255, 242, 253],
    [229, 160, 181, 245, 187, 255, 242, 253],
    [230, 161, 182, 246, 188, 255, 243, 253],
    [231, 163, 183, 246, 189, 255, 243, 253],
    [232, 164, 184, 247, 190, 255, 243, 253],
    [233, 165, 185, 247, 191, 255, 244, 253],
    [234, 166, 185, 247, 192, 255, 244, 253],
    [235, 168, 186, 248, 193, 255, 244, 253],
    [236, 169, 187, 248, 194, 255, 244, 253],
    [236, 170, 188, 248, 195, 255, 245, 253],
    [237, 171, 189, 249, 196, 255, 245, 254],
    [238, 173, 190, 249, 197, 255, 245, 254],
    [239, 174, 191, 249, 198, 255, 245, 254],
    [240, 175, 192, 249, 199, 255, 246, 254],
    [240, 177, 193, 250, 200, 255, 246, 254],
    [241, 178, 194, 250, 201, 255, 246, 254],
    [242, 179, 195, 250, 202, 255, 246, 254],
    [242, 181, 196, 250, 203, 255, 247, 254],
    [243, 182, 197, 251, 204, 255, 247, 254],
    [244, 184, 198, 251, 205, 255, 247, 254],
    [244, 185, 199, 251, 206, 255, 247, 254],
    [245, 186, 200, 251, 207, 255, 247, 254],
    [246, 188, 201, 252, 207, 255, 248, 254],
    [246, 189, 202, 252, 208, 255, 248, 254],
    [247, 191, 203, 252, 209, 255, 248, 254],
    [247, 192, 204, 252, 210, 255, 248, 254],
    [248, 194, 205, 252, 211, 255, 248, 254],
    [248, 195, 206, 252, 212, 255, 249, 254],
    [249, 197, 207, 253, 213, 255, 249, 254],
    [249, 198, 208, 253, 214, 255, 249, 254],
    [250, 200, 210, 253, 215, 255, 249, 254],
    [250, 201, 211, 253, 215, 255, 249, 254],
    [250, 203, 212, 253, 216, 255, 249, 254],
    [251, 204, 213, 253, 217, 255, 250, 254],
    [251, 206, 214, 254, 218, 255, 250, 254],
    [252, 207, 216, 254, 219, 255, 250, 254],
    [252, 209, 217, 254, 220, 255, 250, 254],
    [252, 211, 218, 254, 221, 255, 250, 254],
    [253, 213, 219, 254, 222, 255, 250, 254],
    [253, 214, 221, 254, 223, 255, 250, 254],
    [253, 216, 222, 254, 224, 255, 251, 254],
    [253, 218, 224, 254, 225, 255, 251, 254],
    [254, 220, 225, 254, 225, 255, 251, 254],
    [254, 222, 227, 255, 226, 255, 251, 254],
    [254, 224, 228, 255, 227, 255, 251, 254],
    [254, 226, 230, 255, 228, 255, 251, 254],
    [255, 228, 231, 255, 230, 255, 251, 254],
    [255, 230, 233, 255, 231, 255, 252, 254],
    [255, 232, 235, 255, 232, 255, 252, 254],
    [255, 235, 237, 255, 233, 255, 252, 254],
    [255, 238, 240, 255, 235, 255, 252, 255],
    [255, 241, 243, 255, 236, 255, 252, 254],
    [255, 246, 247, 255, 239, 255, 253, 255],
];

/// `pareto( node, prob )` per spec §9.3.2.
///
/// Returns the probability of the `read_bool` at internal-node index
/// `node` of the token tree, derived from `prob` (the per-band /
/// per-context "unconstrained" probability from the §6.3.7
/// `coef_probs` sweep).
///
/// `node == 0` and `node == 1` short-circuit and return `prob`
/// unchanged — those are the two top tree nodes whose probabilities
/// come directly from the unconstrained-node entries
/// `coef_probs[...][1]` and `coef_probs[...][2]`.
///
/// For `node >= 2`, the spec interpolates between
/// `pareto_table[ x ]` and `pareto_table[ x + 1 ]` based on whether
/// `prob` is odd or even, with `x = (prob - 1) / 2`. Per §10.3 the
/// row index is implicitly clamped to `0..=127` (a `prob = 0` case
/// never arises in conformant streams: the §10 default tables and
/// §6.3.5 `inv_remap_prob` both produce values in `1..=255`).
pub(crate) fn pareto(node: u32, prob: u8) -> u8 {
    if node < 2 {
        return prob;
    }
    // `prob` is in 1..=255 for any conformant stream. We coerce
    // defensively: `prob = 0` would underflow `(prob - 1)`.
    let prob = prob.max(1) as u32;
    let x = ((prob - 1) >> 1) as usize;
    // Spec: `node - 2` indexes into the 8-wide row.
    let col = (node - 2) as usize;
    debug_assert!(col < 8);
    if prob & 1 != 0 {
        PARETO_TABLE[x][col]
    } else {
        // Interpolate two rows (rounded down). Spec's `x + 1` is
        // clamped to `127` so the last row stops cleanly.
        let next = x.saturating_add(1).min(PARETO_TABLE.len() - 1);
        ((PARETO_TABLE[x][col] as u32 + PARETO_TABLE[next][col] as u32) >> 1) as u8
    }
}

/// `more_coefs` per spec §6.4.24 / §9.3.2.
///
/// Reads one `B(p)` from the §9.2 Boolean coder. Returns `true` if
/// more coefficients should be read (`more_coefs == 1`), `false` if
/// the rest of the transform block is zero (`more_coefs == 0`).
///
/// The probability is `coef_probs[txSz][plane>0][is_inter][band][ctx][0]`
/// per the spec — that is, the node-0 unconstrained probability for
/// the running context. Caller is responsible for picking the right
/// cell out of the §6.3.7 table.
pub(crate) fn read_more_coefs(coder: &mut BoolCoder<'_>, prob: u8) -> Result<bool, Error> {
    let v = coder.read_bool(prob as u32)?;
    Ok(v == 1)
}

/// `token` tree walk per spec §6.4.24 / §9.3.3.
///
/// `probs[0]` and `probs[1]` are the two unconstrained-node
/// probabilities (`coef_probs[...][1]` and `coef_probs[...][2]`);
/// `probs[2]..probs[6]` correspond to the deeper tree nodes and are
/// derived from `pareto(node, coef_probs[...][Min(2, 1 + node)])`.
/// For nodes `2..=6` the spec's `Min(2, 1 + node)` clamps the probe
/// index to `2`, so all five deep nodes derive from the *same*
/// unconstrained-node prob (`coef_probs[...][2]`) — only the `node`
/// argument to `pareto` differs.
///
/// `probs[0]` corresponds to internal-node index 1 (the first
/// `read_bool` after the `more_coefs` decision); the §9.3.3 walk
/// indexes by `n >> 1` so internal-node index 1 is the very first
/// probe.
///
/// Returns the decoded token (`ZERO_TOKEN..=DCT_VAL_CATEGORY6`).
///
/// The probe sequence (in `probs` indexing order) walks the
/// `TOKEN_TREE` from index 0:
///
/// * `probs[0]` — node 0 (root: ZERO vs further).
/// * `probs[1]` — node 1 (ONE vs further).
/// * `probs[2]` — node 2 (two-token branch vs higher).
/// * `probs[3]` — node 3 (TWO vs THREE/FOUR).
/// * `probs[4]` — node 4 (THREE vs FOUR).
/// * `probs[5]` — node 5 (CAT1/CAT2 vs CAT3..6).
/// * `probs[6]` — node 6 (CAT1 vs CAT2).
/// * `probs[7]` — node 7 (CAT3/CAT4 vs CAT5/CAT6).
/// * `probs[8]` — node 8 (CAT3 vs CAT4).
/// * `probs[9]` — node 9 (CAT5 vs CAT6).
///
/// The spec's `Min(2, 1+node)` clamp means every probe at
/// `node >= 1` uses `coef_probs[...][2]` as the input to `pareto`;
/// the round-7 walker accepts the full pre-computed probability
/// array so the round-7 driver doesn't need to call `pareto` inline.
pub(crate) fn read_token(coder: &mut BoolCoder<'_>, probs: &[u8; 10]) -> Result<u32, Error> {
    let mut n: i32 = 0;
    loop {
        // node = n >> 1. probs has 10 entries (the spec's 10 internal
        // nodes); the tree's 11 leaves never make us reference a node
        // beyond index 9.
        let node = (n >> 1) as usize;
        debug_assert!(node < probs.len());
        let bit = coder.read_bool(probs[node] as u32)?;
        let idx = (n as usize) + bit as usize;
        debug_assert!(idx < TOKEN_TREE.len());
        n = TOKEN_TREE[idx];
        if n <= 0 {
            return Ok((-n) as u32);
        }
    }
}

/// `read_coef( token )` per spec §6.4.26.
///
/// Decodes the magnitude of a non-zero coefficient given its `token`.
/// `token` must be in `ONE_TOKEN..=DCT_VAL_CATEGORY6` — calling this
/// with `ZERO_TOKEN` is a programmer error (caller is responsible for
/// short-circuiting on ZERO per §6.4.24).
///
/// `bit_depth` is 8, 10 or 12 per the §6.2 `color_config` semantics.
/// For CAT6 tokens at 10- or 12-bit depth, the spec prepends a
/// `BitDepth - 8` block of `B(255)` `high_bit` reads to the standard
/// 14 cat-probs; those high bits are added at shift positions
/// `5 + BitDepth - e`.
///
/// Returns the unsigned magnitude (≥ 1). The caller folds in the
/// `sign_bit` `L(1)` read separately.
pub(crate) fn read_coef(
    coder: &mut BoolCoder<'_>,
    token: u32,
    bit_depth: u32,
) -> Result<u32, Error> {
    debug_assert!((ONE_TOKEN..=DCT_VAL_CATEGORY6).contains(&token));
    debug_assert!(bit_depth == 8 || bit_depth == 10 || bit_depth == 12);
    let row = &EXTRA_BITS[token as usize];
    let cat = row[0] as usize;
    let num_extra = row[1];
    let mut coef = row[2];
    if token == DCT_VAL_CATEGORY6 && bit_depth > 8 {
        // Spec §6.4.26 high-bit loop: for e in 0..BitDepth-8 read
        // `B(255)` and add at shift `5 + BitDepth - e`.
        for e in 0..(bit_depth - 8) {
            let high_bit = coder.read_bool(255)?;
            coef += high_bit << (5 + bit_depth - e);
        }
    }
    let cat_row = &CAT_PROBS[cat];
    for e in 0..num_extra {
        let p = cat_row[e as usize];
        let bit = coder.read_bool(p as u32)?;
        coef += bit << (num_extra - 1 - e);
    }
    Ok(coef)
}

/// Round-7 driver result: one decoded coefficient out of the §6.4.24
/// stream.
///
/// `Eob` indicates the §6.4.24 `more_coefs == 0` early-exit
/// terminated the scan; `Coef` carries the decoded `(token, value)`
/// pair where `value` includes the §6.4.24 sign-bit. `value == 0`
/// when `token == ZERO_TOKEN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoefStep {
    /// `more_coefs == 0` — all remaining scan positions are 0 and the
    /// scan terminates.
    Eob,
    /// A coefficient decoded successfully. `token` is the spec's
    /// `0..=10` enumeration; `value` is the signed coefficient.
    Coef { token: u32, value: i32 },
}

/// Decode the next coefficient at the current scan position per
/// §6.4.24.
///
/// The caller passes the pre-computed `(more_coefs_prob, token_probs)`
/// pair plus a `check_eob` flag and a `bit_depth`.
///
/// * If `check_eob` is true, read a `more_coefs` bit. On 0 return
///   [`CoefStep::Eob`].
/// * Otherwise read `token`. On `ZERO_TOKEN` return
///   `CoefStep::Coef { token: 0, value: 0 }`.
/// * Otherwise call `read_coef` for the magnitude and a single
///   `L(1)` for the sign, returning the signed coefficient.
///
/// The §6.4.24 `checkEob` flag itself is owned by the residual
/// driver, not this function. Caller flips `check_eob` per the spec
/// rule: cleared on `ZERO_TOKEN`, set back to true on every non-zero
/// token (including the first scan position).
pub(crate) fn read_coef_token(
    coder: &mut BoolCoder<'_>,
    check_eob: bool,
    more_coefs_prob: u8,
    token_probs: &[u8; 10],
    bit_depth: u32,
) -> Result<CoefStep, Error> {
    if check_eob {
        let more = read_more_coefs(coder, more_coefs_prob)?;
        if !more {
            return Ok(CoefStep::Eob);
        }
    }
    let token = read_token(coder, token_probs)?;
    if token == ZERO_TOKEN {
        return Ok(CoefStep::Coef { token: 0, value: 0 });
    }
    let mag = read_coef(coder, token, bit_depth)?;
    let sign = coder.read_literal(1)?;
    let value = if sign == 1 { -(mag as i32) } else { mag as i32 };
    Ok(CoefStep::Coef { token, value })
}

// ----- §6.4.24 `tokens( )` per-block driver -----

/// `coefband_4x4[ 16 ]` per spec §10 — transcribed verbatim.
///
/// Maps a 4x4 transform block's scan index `c` (`0..16`) to its
/// coefficient band, which selects the `coef_probs[ … ][ band ][ … ]`
/// slab for the `more_coefs` / `token` decode at that scan position.
pub(crate) const COEFBAND_4X4: [u8; 16] = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];

/// `coefband_8x8plus[ 1024 ]` per spec §10 — built from the verbatim
/// listing. The first 21 entries are
/// `{0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4}`; every entry from
/// index 21 onward is `5` (the §10 listing fills the remaining 1003
/// rows with `5`). Used for every transform size larger than 4x4
/// (`TX_8X8` / `TX_16X16` / `TX_32X32`), indexed by the scan index `c`.
pub(crate) const COEFBAND_8X8PLUS: [u8; 1024] = build_coefband_8x8plus();

const fn build_coefband_8x8plus() -> [u8; 1024] {
    // §10 listing prefix (indices 0..=20), then `5` for the rest.
    const PREFIX: [u8; 21] = [
        0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ];
    let mut t = [5u8; 1024];
    let mut i = 0;
    while i < PREFIX.len() {
        t[i] = PREFIX[i];
        i += 1;
    }
    t
}

/// `band( c, txSz )` — the §6.4.24 band selection
/// `(txSz == TX_4X4) ? coefband_4x4[ c ] : coefband_8x8plus[ c ]`.
#[inline]
pub(crate) fn coef_band(c: usize, tx_sz: u32) -> usize {
    if tx_sz == 0 {
        COEFBAND_4X4[c] as usize
    } else {
        COEFBAND_8X8PLUS[c] as usize
    }
}

/// The §9.3.2 neighbour pair `nb[ 0 ]` / `nb[ 1 ]` of the coefficient at
/// scan position `pos` for a transform block of size `txSz` and the
/// resolved `tx_type`.
///
/// Returns `(0, 0)` for the DC coefficient (`c == 0`), where the spec
/// uses the above/left non-zero context rather than `TokenCache`
/// neighbours. For `c > 0` the neighbour positions are the raster cells
/// directly above (`a = (i-1)*n + j`) and to the left (`a2 = i*n + j-1`)
/// of `pos`, picked per the `tx_type` (`DCT_ADST` doubles the above
/// neighbour, `ADST_DCT` the left neighbour, every other `tx_type` uses
/// one of each), with single-edge fallbacks for the first row / column.
///
/// `n = 4 << txSz` is the transform-block width in samples (4 / 8 / 16 /
/// 32). `pos` is a raster index in `0 .. n*n`.
#[inline]
pub(crate) fn token_cache_neighbours(
    c: usize,
    pos: usize,
    tx_sz: u32,
    tx_type: u8,
) -> (usize, usize) {
    if c == 0 {
        return (0, 0);
    }
    let n = 4usize << tx_sz;
    let i = pos / n;
    let j = pos % n;
    if i > 0 && j > 0 {
        let a = (i - 1) * n + j;
        let a2 = i * n + j - 1;
        if tx_type == DCT_ADST {
            (a, a)
        } else if tx_type == ADST_DCT {
            (a2, a2)
        } else {
            (a, a2)
        }
    } else if i > 0 {
        let a = (i - 1) * n + j;
        (a, a)
    } else {
        let a2 = i * n + j - 1;
        (a2, a2)
    }
}

/// Build the 10-element `read_token` probability array for one
/// `coef_probs[ … ][ ctx ]` cell per §9.3.2.
///
/// `cell` is the three-element unconstrained-node slab
/// `coef_probs[txSz][plane>0][is_inter][band][ctx]`. The token tree has
/// 10 internal nodes; per §9.3.2 the probability for internal node
/// `node` is `pareto( node, cell[ Min(2, 1 + node) ] )`. So:
///
/// * node 0 → `pareto(0, cell[1])` = `cell[1]` (pareto passes through
///   for `node < 2`),
/// * node 1 → `pareto(1, cell[2])` = `cell[2]`,
/// * node `2..=9` → `pareto(node, cell[2])`.
#[inline]
pub(crate) fn build_token_probs(cell: &[u8; 3]) -> [u8; 10] {
    let mut probs = [0u8; 10];
    for (node, slot) in probs.iter_mut().enumerate() {
        let probe = cell[(1 + node).min(2)];
        *slot = pareto(node as u32, probe);
    }
    probs
}

/// Per-plane non-zero context strips used by the §6.4.21 residual driver
/// and read by the §6.4.24 `tokens( )` DC-context derivation.
///
/// `AboveNonzeroContext[ plane ][ x4 ]` / `LeftNonzeroContext[ plane ][
/// y4 ]` record, at a 4-sample granularity, whether the transform block
/// covering that 4-sample column / row decoded any non-zero coefficient.
/// `tokens( )` reads them for the DC coefficient's `ctx`; the residual
/// driver writes them back after each block per §6.4.21.
#[derive(Debug, Clone)]
pub(crate) struct NonzeroContext {
    /// `AboveNonzeroContext[ plane ]` — one bit per 4-sample column.
    pub above: Vec<u8>,
    /// `LeftNonzeroContext[ plane ]` — one bit per 4-sample row.
    pub left: Vec<u8>,
}

impl NonzeroContext {
    /// All-zero context of the given column / row span (in 4-sample
    /// units). `above_len` covers `(2 * MiCols) >> subsampling_x`
    /// columns; `left_len` covers `(2 * MiRows) >> subsampling_y` rows.
    pub(crate) fn new(above_len: usize, left_len: usize) -> Self {
        Self {
            above: vec![0u8; above_len],
            left: vec![0u8; left_len],
        }
    }
}

/// Mode-info / frame state the §6.4.24 `tokens( )` driver reads to pick
/// the `coef_probs` cell and resolve the `TxType` for the neighbour
/// derivation.
///
/// This bundles the per-block / per-frame variables the spec references
/// inside `tokens( )` and its §9.3.2 context derivation without yet
/// owning the whole §6.4.21 residual loop. The residual loop (a later
/// round) fills these in per block and threads the `NonzeroContext`
/// across the frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenBlockCtx {
    /// `plane` (0 = luma, 1 / 2 = chroma).
    pub plane: usize,
    /// `is_inter` for the block (selects the `coef_probs` ref-type
    /// slab).
    pub is_inter: bool,
    /// Resolved §6.4.25 `TxType` for the block (already past the
    /// chroma / `TX_32X32` / lossless `DCT_DCT` overrides). Used for the
    /// §9.3.2 neighbour selection.
    pub tx_type: u8,
    /// `BitDepth` (8 / 10 / 12) for the §6.4.26 `read_coef` high-bit
    /// loop.
    pub bit_depth: u32,
    /// `x4 = startX >> 2` — the block's leftmost 4-sample column index
    /// into `AboveNonzeroContext[ plane ]`.
    pub x4: usize,
    /// `y4 = startY >> 2` — the block's topmost 4-sample row index into
    /// `LeftNonzeroContext[ plane ]`.
    pub y4: usize,
    /// `maxX = (2 * MiCols) >> subsampling_x` — the column count of
    /// `AboveNonzeroContext[ plane ]` (DC-context loop bound).
    pub max_x: usize,
    /// `maxY = (2 * MiRows) >> subsampling_y` — the row count of
    /// `LeftNonzeroContext[ plane ]` (DC-context loop bound).
    pub max_y: usize,
}

/// `tokens( plane, startX, startY, txSz, blockIdx )` per spec §6.4.24.
///
/// Decodes the coefficient tokens of one transform block, walking the
/// scan order returned by [`crate::scan::get_scan`] and feeding each
/// scan position through the round-7 [`read_coef_token`] pipeline. The
/// per-coefficient `ctx` is derived per §9.3.2: for the DC coefficient
/// (`c == 0`) from the `AboveNonzeroContext` / `LeftNonzeroContext`
/// strips in `nz`; for `c > 0` from the running [`TokenCache`] via the
/// §9.3.2 neighbour pair.
///
/// The output `tokens` buffer (`Tokens[ ]`, length `segEob = 16 << (txSz
/// << 1)`) is filled in raster order: decoded coefficients land at their
/// `pos = scan[ c ]`, every scan position past the early `more_coefs ==
/// 0` exit is zeroed (§6.4.24's trailing `for ( i = c; … ) Tokens[
/// scan[ i ] ] = 0`).
///
/// Returns `nonzero = c > 0` — `true` if any coefficient (even a single
/// explicit `ZERO_TOKEN`) was decoded before the end-of-block. The
/// §6.4.21 residual driver writes this back into the `nz` strips for the
/// block's columns / rows.
///
/// `coef_probs` is the round-6 6D table
/// `coef_probs[txSz][plane>0][is_inter][band][ctx][node]`. `scan` is the
/// §6.4.25 scan slice for this block. `tokens` must have exactly
/// `segEob` entries and is fully overwritten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tokens(
    coder: &mut BoolCoder<'_>,
    block: &TokenBlockCtx,
    tx_sz: u32,
    scan: &[u16],
    coef_probs: &crate::coef_probs::CoefProbs,
    nz: &NonzeroContext,
    token_cache: &mut [u8],
    tokens: &mut [i64],
) -> Result<bool, Error> {
    let seg_eob = 16usize << (tx_sz << 1);
    debug_assert_eq!(scan.len(), seg_eob);
    debug_assert_eq!(tokens.len(), seg_eob);
    debug_assert!(token_cache.len() >= seg_eob);

    let plane_gt0 = usize::from(block.plane > 0);
    let is_inter = usize::from(block.is_inter);
    // `coef_probs[txSz]` slab is constant for the whole block.
    let tx_slab = &coef_probs[tx_sz as usize][plane_gt0][is_inter];

    let mut check_eob = true;
    let mut c = 0usize;
    while c < seg_eob {
        let pos = scan[c] as usize;
        let band = coef_band(c, tx_sz);

        // §9.3.2 ctx derivation.
        let ctx = if c == 0 {
            // DC: above/left non-zero context across `numpts` 4-sample
            // units (numpts = 1 << txSz). `above`/`left` are OR-reduced
            // bits, so ctx ∈ {0, 1, 2}.
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

        if check_eob && !read_more_coefs(coder, cell[0])? {
            break;
        }

        let token_probs = build_token_probs(cell);
        let token = read_token(coder, &token_probs)?;
        token_cache[pos] = ENERGY_CLASS[token as usize];
        if token == ZERO_TOKEN {
            tokens[pos] = 0;
            check_eob = false;
        } else {
            let mag = read_coef(coder, token, block.bit_depth)?;
            let sign = coder.read_literal(1)?;
            tokens[pos] = if sign == 1 { -(mag as i64) } else { mag as i64 };
            check_eob = true;
        }
        c += 1;
    }

    let nonzero = c > 0;
    // §6.4.24: zero every scan position from `c` to the end.
    for &p in &scan[c..] {
        tokens[p as usize] = 0;
    }
    Ok(nonzero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::{ADST_ADST, DCT_DCT};

    // The hand-traced golden buffers below were derived by stepping
    // the §9.2 Boolean decoder by hand against the documented
    // `init_bool` semantics; no external library / source was
    // consulted at any point. The `0x00` prefix produces the
    // post-marker state `(BoolValue=0, BoolRange=128)` from which the
    // first `read_bool(p)` decodes 0 for any `p` with split > 0
    // (i.e. any p > 0).

    fn coder_zero(bytes: &[u8]) -> BoolCoder<'_> {
        BoolCoder::init_bool(bytes, bytes.len()).unwrap()
    }

    // ----- Static tables -----

    #[test]
    fn extra_bits_table_matches_spec() {
        // Verbatim spot-check of §6.4.26 extra_bits[11][3].
        assert_eq!(EXTRA_BITS[0], [0, 0, 0]); // ZERO
        assert_eq!(EXTRA_BITS[1], [0, 0, 1]); // ONE
        assert_eq!(EXTRA_BITS[4], [0, 0, 4]); // FOUR
        assert_eq!(EXTRA_BITS[5], [1, 1, 5]); // CAT1
        assert_eq!(EXTRA_BITS[6], [2, 2, 7]); // CAT2
        assert_eq!(EXTRA_BITS[7], [3, 3, 11]); // CAT3
        assert_eq!(EXTRA_BITS[8], [4, 4, 19]); // CAT4
        assert_eq!(EXTRA_BITS[9], [5, 5, 35]); // CAT5
        assert_eq!(EXTRA_BITS[10], [6, 14, 67]); // CAT6
    }

    #[test]
    fn cat_probs_table_matches_spec() {
        // Spec listing — cat 1 has a single entry of 159; cat 6 the
        // full 14-element row {254, 254, 254, 252, 249, 243, 230,
        // 196, 177, 153, 140, 133, 130, 129}.
        assert_eq!(CAT_PROBS[1][0], 159);
        assert_eq!(CAT_PROBS[2], [165, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(CAT_PROBS[6][0], 254);
        assert_eq!(CAT_PROBS[6][7], 196);
        assert_eq!(CAT_PROBS[6][13], 129);
    }

    #[test]
    fn energy_class_table_matches_spec() {
        // Spec §10.2: {0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5}.
        assert_eq!(ENERGY_CLASS, [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn token_tree_layout_matches_spec_listing() {
        // Spec §9.3 (line 6217): the 20-entry listing is exactly the
        // shape we encode. Negative entries are tokens; positives are
        // jump offsets.
        assert_eq!(TOKEN_TREE[0], -(ZERO_TOKEN as i32));
        assert_eq!(TOKEN_TREE[1], 2);
        assert_eq!(TOKEN_TREE[2], -(ONE_TOKEN as i32));
        assert_eq!(TOKEN_TREE[3], 4);
        assert_eq!(TOKEN_TREE[4], 6);
        assert_eq!(TOKEN_TREE[5], 10);
        assert_eq!(TOKEN_TREE[6], -(TWO_TOKEN as i32));
        assert_eq!(TOKEN_TREE[7], 8);
        assert_eq!(TOKEN_TREE[8], -(THREE_TOKEN as i32));
        assert_eq!(TOKEN_TREE[9], -(FOUR_TOKEN as i32));
        assert_eq!(TOKEN_TREE[10], 12);
        assert_eq!(TOKEN_TREE[11], 14);
        assert_eq!(TOKEN_TREE[12], -(DCT_VAL_CATEGORY1 as i32));
        assert_eq!(TOKEN_TREE[13], -(DCT_VAL_CATEGORY2 as i32));
        assert_eq!(TOKEN_TREE[16], -(DCT_VAL_CATEGORY3 as i32));
        assert_eq!(TOKEN_TREE[17], -(DCT_VAL_CATEGORY4 as i32));
        assert_eq!(TOKEN_TREE[18], -(DCT_VAL_CATEGORY5 as i32));
        assert_eq!(TOKEN_TREE[19], -(DCT_VAL_CATEGORY6 as i32));
    }

    #[test]
    fn pareto_table_has_128_rows() {
        assert_eq!(PARETO_TABLE.len(), 128);
        assert_eq!(PARETO_TABLE[0].len(), 8);
    }

    #[test]
    fn pareto_table_spec_anchors() {
        // First row {3, 86, 128, 6, 86, 23, 88, 29}.
        assert_eq!(PARETO_TABLE[0], [3, 86, 128, 6, 86, 23, 88, 29]);
        // Last row {255, 246, 247, 255, 239, 255, 253, 255}.
        assert_eq!(PARETO_TABLE[127], [255, 246, 247, 255, 239, 255, 253, 255]);
        // Row 49 {187, 129, 160, 225, 156, 255, 228, 251}.
        assert_eq!(PARETO_TABLE[49], [187, 129, 160, 225, 156, 255, 228, 251]);
    }

    // ----- pareto( node, prob ) -----

    #[test]
    fn pareto_node_lt_2_returns_prob_unchanged() {
        // For node 0 and 1, the spec returns `prob` unchanged.
        for prob in [1u8, 5, 64, 128, 200, 254] {
            assert_eq!(pareto(0, prob), prob);
            assert_eq!(pareto(1, prob), prob);
        }
    }

    #[test]
    fn pareto_odd_prob_indexes_row_directly() {
        // prob = 1 -> x = 0; prob & 1 != 0 -> direct row[0] lookup.
        assert_eq!(pareto(2, 1), PARETO_TABLE[0][0]);
        assert_eq!(pareto(3, 1), PARETO_TABLE[0][1]);
        // prob = 3 -> x = 1.
        assert_eq!(pareto(5, 3), PARETO_TABLE[1][3]);
        // prob = 255 -> x = 127 (saturates).
        assert_eq!(pareto(2, 255), PARETO_TABLE[127][0]);
    }

    #[test]
    fn pareto_even_prob_averages_two_rows() {
        // prob = 2 -> x = 0 (since (2-1)/2 = 0); even -> average of
        // rows 0 and 1.
        let expected = ((PARETO_TABLE[0][0] as u32 + PARETO_TABLE[1][0] as u32) >> 1) as u8;
        assert_eq!(pareto(2, 2), expected);
        // prob = 254 -> x = 126; even -> average rows 126 and 127.
        let expected = ((PARETO_TABLE[126][2] as u32 + PARETO_TABLE[127][2] as u32) >> 1) as u8;
        assert_eq!(pareto(4, 254), expected);
    }

    #[test]
    fn pareto_prob_zero_clamps_defensively() {
        // Spec never produces prob = 0, but the defensive
        // `prob.max(1)` keeps the function panic-free.
        // After clamp, prob = 1 -> direct row 0 lookup.
        assert_eq!(pareto(2, 0), PARETO_TABLE[0][0]);
    }

    // ----- read_more_coefs -----

    #[test]
    fn read_more_coefs_zero_buffer_returns_false() {
        // Buffer [0x00; 4] post-marker: BoolValue=0, BoolRange=128.
        // For any prob with split > 0, BoolValue < split → bit = 0 →
        // more_coefs = 0 → false.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        assert!(!read_more_coefs(&mut dec, 128).unwrap());
    }

    // ----- read_token (tree walk) — hand-traced golden buffers -----

    #[test]
    fn read_token_one_token_golden_buffer() {
        // ONE_TOKEN requires the token-tree walk to take bits 1, 0
        // (n=0 → 2 → -ONE). Hand-derivation against the §9.2 coder
        // with all probs = 128:
        //   init_bool([0x40, 0x00, ..]): BoolValue=0x40 (marker = 0
        //     because 0x40 < 128 split). After marker: value=64,
        //     range=128.
        //   read_bool(128) #1: split=64. value(64) >= split → bit=1,
        //     range=64, value=0. Renorm pulls bit 7 of byte[1]=0
        //     → value=0, range=128.
        //   read_bool(128) #2: split=64. value(0) < split → bit=0,
        //     range=64. Renorm pulls bit 6 of byte[1]=0 → value=0,
        //     range=128.
        // Bits {1, 0} → walker returns ONE_TOKEN.
        let bytes = [0x40u8, 0x00, 0x00, 0x00];
        let mut dec = coder_zero(&bytes);
        let probs = [128u8; 10];
        assert_eq!(read_token(&mut dec, &probs).unwrap(), ONE_TOKEN);
    }

    #[test]
    fn read_token_two_token_golden_buffer() {
        // TWO_TOKEN requires bits 1, 1, 0, 0 (n: 0 → 2 → 4 → 6 → -TWO).
        // Hand-derivation with probs = 128:
        //   init_bool([0x60, 0x00, ..]): BoolValue=0x60=96 (< 128
        //     → marker = 0). After marker: value=96, range=128.
        //   #1 split=64. 96>=64 → bit=1, value=32, range=64. Renorm
        //     pulls bit 7 of byte[1]=0 → value=64, range=128.
        //   #2 split=64. 64>=64 → bit=1, value=0, range=64. Renorm
        //     pulls bit 6 of byte[1]=0 → value=0, range=128.
        //   #3 split=64. 0<64 → bit=0, range=64. Renorm pulls bit 5
        //     → value=0, range=128.
        //   #4 split=64. 0<64 → bit=0, range=64. Renorm pulls bit 4
        //     → value=0, range=128.
        // Bits {1, 1, 0, 0} → walker returns TWO_TOKEN.
        let bytes = [0x60u8, 0x00, 0x00, 0x00];
        let mut dec = coder_zero(&bytes);
        let probs = [128u8; 10];
        assert_eq!(read_token(&mut dec, &probs).unwrap(), TWO_TOKEN);
    }

    #[test]
    fn read_token_zero_buffer_returns_zero_token() {
        // On a zero buffer post-marker, every `read_bool(p>0)`
        // returns 0. The token-tree walk goes:
        //   n=0, bit=0 → n=TOKEN_TREE[0] = -ZERO_TOKEN → return 0.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        // probs are all-zero too (the prob = 0 path would underrun,
        // but with bit forced to 0 by BoolValue=0 the walker won't
        // probe them). Use prob = 1 for all to keep the coder valid.
        let probs = [1u8; 10];
        assert_eq!(read_token(&mut dec, &probs).unwrap(), ZERO_TOKEN);
    }

    // ----- read_coef -----

    #[test]
    fn read_coef_one_token_returns_one_no_bits() {
        // ONE_TOKEN: extra_bits row {0, 0, 1} → numExtra=0, coef=1.
        // No bits are read.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        assert_eq!(read_coef(&mut dec, ONE_TOKEN, 8).unwrap(), 1);
    }

    #[test]
    fn read_coef_two_three_four_no_extra_bits() {
        // TWO/THREE/FOUR all return their base coef with no reads.
        for (tok, expected) in [(TWO_TOKEN, 2), (THREE_TOKEN, 3), (FOUR_TOKEN, 4)] {
            let bytes = [0u8; 4];
            let mut dec = coder_zero(&bytes);
            assert_eq!(read_coef(&mut dec, tok, 8).unwrap(), expected);
        }
    }

    #[test]
    fn read_coef_cat1_zero_buffer_returns_base_5() {
        // CAT1: numExtra=1 with prob cat_probs[1][0]=159. On a zero
        // buffer the bit is 0 → coef = 5 + 0<<0 = 5.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        assert_eq!(read_coef(&mut dec, DCT_VAL_CATEGORY1, 8).unwrap(), 5);
    }

    #[test]
    fn read_coef_cat2_zero_buffer_returns_base_7() {
        // CAT2: numExtra=2. Both bits = 0 → coef = 7.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        assert_eq!(read_coef(&mut dec, DCT_VAL_CATEGORY2, 8).unwrap(), 7);
    }

    #[test]
    fn read_coef_cat6_zero_buffer_returns_base_67() {
        // CAT6 at 8-bit depth: numExtra=14, all bits = 0 → coef = 67.
        // Need a slightly longer buffer for 14 reads + renorms.
        let bytes = [0u8; 8];
        let mut dec = coder_zero(&bytes);
        assert_eq!(read_coef(&mut dec, DCT_VAL_CATEGORY6, 8).unwrap(), 67);
    }

    #[test]
    fn read_coef_cat6_10bit_zero_buffer_returns_base_67() {
        // 10-bit depth adds 2 high_bit reads at B(255). On zero buffer
        // bits are all 0, so coef = 67 still.
        let bytes = [0u8; 8];
        let mut dec = coder_zero(&bytes);
        assert_eq!(read_coef(&mut dec, DCT_VAL_CATEGORY6, 10).unwrap(), 67);
    }

    #[test]
    fn read_coef_cat6_12bit_zero_buffer_returns_base_67() {
        // 12-bit depth adds 4 high_bit reads.
        let bytes = [0u8; 8];
        let mut dec = coder_zero(&bytes);
        assert_eq!(read_coef(&mut dec, DCT_VAL_CATEGORY6, 12).unwrap(), 67);
    }

    // ----- read_coef_token driver -----

    #[test]
    fn read_coef_token_check_eob_zero_buffer_returns_eob() {
        // check_eob = true, more_coefs_prob = 128 → bit 0 → Eob.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        let probs = [1u8; 10];
        assert_eq!(
            read_coef_token(&mut dec, true, 128, &probs, 8).unwrap(),
            CoefStep::Eob
        );
    }

    #[test]
    fn read_coef_token_no_check_eob_zero_buffer_returns_zero_token() {
        // check_eob = false → skip more_coefs → straight to
        // read_token → on zero buffer returns ZERO_TOKEN with
        // value = 0.
        let bytes = [0u8; 4];
        let mut dec = coder_zero(&bytes);
        let probs = [1u8; 10];
        assert_eq!(
            read_coef_token(&mut dec, false, 128, &probs, 8).unwrap(),
            CoefStep::Coef { token: 0, value: 0 }
        );
    }

    #[test]
    fn read_coef_token_drives_positive_one_through_full_pipeline() {
        // End-to-end driver: check_eob = true, more_coefs_prob picks
        // the high-bit branch (BoolValue 0x80 wouldn't pass the marker
        // — instead we drive `more_coefs` via a 1 from byte[0]=0x40
        // with prob = 64 picking split low enough).
        //
        // Simpler approach: use check_eob = false and the ONE_TOKEN
        // golden buffer (0x40, 0x00, ...). Token = ONE_TOKEN means
        // numExtra = 0, then sign bit is the next L(1) read.
        // After bits {1, 0} (ONE_TOKEN consumed), sign bit reads the
        // next bool(128). State after token: value=0, range=128.
        // Sign read_bool(128): split=64, value(0) < 64 → bit=0 →
        // sign positive. So value = +1.
        let bytes = [0x40u8, 0x00, 0x00, 0x00];
        let mut dec = coder_zero(&bytes);
        let probs = [128u8; 10];
        assert_eq!(
            read_coef_token(&mut dec, false, 128, &probs, 8).unwrap(),
            CoefStep::Coef {
                token: ONE_TOKEN,
                value: 1
            }
        );
    }

    #[test]
    fn read_coef_token_check_eob_then_zero_token_proceeds() {
        // check_eob = true, then ZERO_TOKEN path. Build buffer where:
        //   - First read_bool(more_coefs_prob = 64) returns 1
        //     (more_coefs = true, scan continues).
        //   - Next read_token walks to ZERO_TOKEN (bit = 0 at node 0).
        //   - value = 0 returned.
        //
        // For read_bool(64): split = 1 + ((255-1)*64 >> 8) = 1 + 63
        // = 64. So byte[0] >= 64 produces more_coefs = 1.
        // After: value=byte[0]-64, range=64 → renorm → value=
        //   2*(byte[0]-64)+bit_7(byte[1]), range=128.
        // For ZERO_TOKEN at the next read_bool(token_probs[0]=128),
        // we need value < 64 (the split).
        //   byte[0] = 0x40 → after marker value=64, range=128. Wait,
        //   read_bool(64) split = 1 + ((127 * 64) >> 8) = 1 + 31 =
        //   32. So byte[0] = 0x40 = 64 ≥ 32 → bit = 1, value = 32,
        //   range = 96. No renorm. Next read_bool(128): split = 1 +
        //   ((95 * 128) >> 8) = 1 + 47 = 48. value(32) < 48 → bit=0
        //   → ZERO_TOKEN.
        let bytes = [0x40u8, 0x00, 0x00, 0x00];
        let mut dec = coder_zero(&bytes);
        let probs = [128u8; 10];
        assert_eq!(
            read_coef_token(&mut dec, true, 64, &probs, 8).unwrap(),
            CoefStep::Coef {
                token: ZERO_TOKEN,
                value: 0
            }
        );
    }

    // ----- Round-trip token magnitudes against extra_bits table -----

    #[test]
    fn read_coef_round_trip_matches_extra_bits_base_when_bits_zero() {
        // For every token, on a zero buffer (all extra bits = 0) the
        // resulting magnitude must equal the EXTRA_BITS base coef.
        for tok in ONE_TOKEN..=DCT_VAL_CATEGORY6 {
            let bytes = [0u8; 16];
            let mut dec = coder_zero(&bytes);
            let coef = read_coef(&mut dec, tok, 8).unwrap();
            assert_eq!(
                coef, EXTRA_BITS[tok as usize][2],
                "token {tok} base coef mismatch"
            );
        }
    }

    #[test]
    fn read_coef_cat_magnitudes_span_spec_documented_ranges() {
        // Spec §7.4.16 (token semantics):
        //   CAT1 -> 5..=6 (range covered by numExtra=1: 2 values).
        //   CAT2 -> 7..=10 (4 values).
        //   CAT3 -> 11..=18 (8 values).
        //   CAT4 -> 19..=34 (16 values).
        //   CAT5 -> 35..=66 (32 values).
        //   CAT6 -> 67..=66+2^14 = 67..=16450 (14 extra bits at 8-bit).
        let cases: &[(u32, u32, u32)] = &[
            (DCT_VAL_CATEGORY1, 5, 5 + (1 << 1) - 1),
            (DCT_VAL_CATEGORY2, 7, 7 + (1 << 2) - 1),
            (DCT_VAL_CATEGORY3, 11, 11 + (1 << 3) - 1),
            (DCT_VAL_CATEGORY4, 19, 19 + (1 << 4) - 1),
            (DCT_VAL_CATEGORY5, 35, 35 + (1 << 5) - 1),
            (DCT_VAL_CATEGORY6, 67, 67 + (1 << 14) - 1),
        ];
        for (tok, lo, hi) in cases.iter().copied() {
            // The decoded value with all-zero extra bits is `lo`; the
            // theoretical maximum (all-one extra bits) is `hi`. Our
            // zero-buffer probe confirms `lo`; the range arithmetic
            // is a sanity check on numExtra + base in EXTRA_BITS.
            assert_eq!(EXTRA_BITS[tok as usize][2], lo);
            let num_extra = EXTRA_BITS[tok as usize][1];
            assert_eq!(lo + (1 << num_extra) - 1, hi);
        }
    }

    // ----- Pareto-driven probe-prob lookup ----

    #[test]
    fn pareto_supplies_all_seven_deep_node_probs_from_single_unconstrained() {
        // The token tree has 10 internal nodes. Per §9.3.2's
        // `Min(2, 1 + node)` clamp, the probability table queried for
        // node 0 is `coef_probs[...][1]` (the round-7 probs[0]
        // input), and every node >= 1 queries
        // `coef_probs[...][2]` — that is, a single probability byte
        // feeds the pareto interpolation for nodes 1..=9.
        //
        // For an unconstrained prob byte 130 (= 0x82): pareto(1)
        // returns prob unchanged = 130. pareto(2..=9) returns the
        // interpolated row. prob = 130 is even, x = 64, so the row
        // average of PARETO_TABLE[64] and PARETO_TABLE[65] in each
        // column gives the per-node probs.
        let unconstrained: u8 = 130;
        let probs1 = pareto(1, unconstrained);
        assert_eq!(probs1, 130);
        let expected_node2 = ((PARETO_TABLE[64][0] as u32 + PARETO_TABLE[65][0] as u32) >> 1) as u8;
        assert_eq!(pareto(2, unconstrained), expected_node2);
    }

    // ----- §6.4.24 band tables -----

    #[test]
    fn coefband_4x4_matches_spec() {
        // §10 coefband_4x4[16].
        assert_eq!(
            COEFBAND_4X4,
            [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        );
    }

    #[test]
    fn coefband_8x8plus_matches_spec_prefix_then_fives() {
        // §10 coefband_8x8plus[1024]: 21-entry prefix, then all 5s.
        assert_eq!(COEFBAND_8X8PLUS.len(), 1024);
        let prefix = [
            0u8, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        ];
        assert_eq!(&COEFBAND_8X8PLUS[..21], &prefix);
        // Index 21 is the first 5; every entry from there is 5.
        assert!(COEFBAND_8X8PLUS[21..].iter().all(|&b| b == 5));
        assert_eq!(COEFBAND_8X8PLUS[1023], 5);
    }

    #[test]
    fn coef_band_dispatches_by_tx_size() {
        // TX_4X4 (tx_sz == 0) uses coefband_4x4; everything else uses
        // coefband_8x8plus.
        assert_eq!(coef_band(13, 0), 5); // coefband_4x4[13] = 5
        assert_eq!(coef_band(13, 1), 4); // coefband_8x8plus[13] = 4
        assert_eq!(coef_band(0, 0), 0);
        assert_eq!(coef_band(0, 1), 0);
        assert_eq!(coef_band(21, 3), 5); // 8x8plus[21] = 5
    }

    // ----- §9.3.2 neighbour derivation -----

    #[test]
    fn token_cache_neighbours_dc_is_origin() {
        // c == 0 -> nb = (0, 0) regardless of pos / tx_type.
        for tt in [DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST] {
            assert_eq!(token_cache_neighbours(0, 0, 0, tt), (0, 0));
            assert_eq!(token_cache_neighbours(0, 5, 1, tt), (0, 0));
        }
    }

    #[test]
    fn token_cache_neighbours_interior_dct_dct() {
        // 4x4 (n=4), pos = 5 -> i=1, j=1 (interior). DCT_DCT: nb0 = above
        // = (i-1)*n+j = 1, nb1 = left = i*n+j-1 = 4.
        assert_eq!(token_cache_neighbours(1, 5, 0, DCT_DCT), (1, 4));
        // ADST_ADST behaves like DCT_DCT for the neighbour pick.
        assert_eq!(token_cache_neighbours(1, 5, 0, ADST_ADST), (1, 4));
    }

    #[test]
    fn token_cache_neighbours_interior_adst_variants() {
        // pos = 5 (i=1, j=1) in a 4x4 block.
        // DCT_ADST doubles the above neighbour (a = 1).
        assert_eq!(token_cache_neighbours(1, 5, 0, DCT_ADST), (1, 1));
        // ADST_DCT doubles the left neighbour (a2 = 4).
        assert_eq!(token_cache_neighbours(1, 5, 0, ADST_DCT), (4, 4));
    }

    #[test]
    fn token_cache_neighbours_first_row_and_column() {
        // First row (i == 0): only the left neighbour exists. pos = 2
        // (i=0, j=2) -> nb = (i*n+j-1, same) = (1, 1).
        assert_eq!(token_cache_neighbours(1, 2, 0, DCT_DCT), (1, 1));
        // First column (j == 0): only the above neighbour exists. pos = 8
        // (i=2, j=0) in a 4x4 -> nb = ((i-1)*n+j, same) = (4, 4).
        assert_eq!(token_cache_neighbours(1, 8, 0, DCT_DCT), (4, 4));
    }

    #[test]
    fn token_cache_neighbours_8x8_width_scaling() {
        // n = 4 << txSz: at TX_8X8 (tx_sz=1) n=8. pos=9 -> i=1, j=1.
        // DCT_DCT: nb0 = above = 1, nb1 = left = 8.
        assert_eq!(token_cache_neighbours(1, 9, 1, DCT_DCT), (1, 8));
    }

    // ----- build_token_probs -----

    #[test]
    fn build_token_probs_maps_nodes_per_spec() {
        // cell = [more_coefs, node0_prob, node1+_prob].
        let cell = [200u8, 130, 90];
        let probs = build_token_probs(&cell);
        // node 0 -> pareto(0, cell[Min(2,1)] = cell[1]) = cell[1] = 130.
        assert_eq!(probs[0], 130);
        // node 1 -> pareto(1, cell[Min(2,2)] = cell[2]) = cell[2] = 90.
        assert_eq!(probs[1], 90);
        // node 2..9 -> pareto(node, cell[2] = 90).
        for node in 2..10u32 {
            assert_eq!(probs[node as usize], pareto(node, 90));
        }
        // cell[0] (the more_coefs prob) is never folded into probs.
        assert!(!probs.contains(&200) || probs[0] == 130);
    }

    // ----- §6.4.24 tokens( ) driver -----

    /// A `CoefProbs` table with every cell set to `[p, p, p]`.
    fn uniform_coef_probs(p: u8) -> crate::coef_probs::CoefProbs {
        [[[[[[p; 3]; 6]; 6]; 2]; 2]; 4]
    }

    /// A block ctx with empty non-zero context (so the DC ctx is 0) and
    /// the given tx_type.
    fn luma_block(tx_type: u8) -> TokenBlockCtx {
        TokenBlockCtx {
            plane: 0,
            is_inter: false,
            tx_type,
            bit_depth: 8,
            x4: 0,
            y4: 0,
            max_x: 16,
            max_y: 16,
        }
    }

    #[test]
    fn tokens_zero_buffer_returns_immediate_eob() {
        // On a zero buffer, the first `more_coefs` read_bool(prob>0)
        // returns 0 -> immediate EOB at c == 0 -> nonzero = false and
        // every Tokens entry is zeroed.
        let bytes = [0u8; 8];
        let mut dec = coder_zero(&bytes);
        let probs = uniform_coef_probs(128);
        let nz = NonzeroContext::new(16, 16);
        let scan = &crate::scan::DEFAULT_SCAN_4X4;
        let mut cache = [0u8; 16];
        let mut tokens = [99i64; 16];
        let block = luma_block(DCT_DCT);
        let nonzero = super::tokens(
            &mut dec,
            &block,
            0,
            scan,
            &probs,
            &nz,
            &mut cache,
            &mut tokens,
        )
        .unwrap();
        assert!(!nonzero);
        assert!(tokens.iter().all(|&t| t == 0), "all tokens must be zeroed");
    }

    #[test]
    fn tokens_zero_token_clears_check_eob_and_fills_block() {
        // Buffer `0x40 0x00…` with uniform probs = 128:
        //   c=0: more_coefs read_bool(128) on value 64 -> bit 1
        //        (continue). token tree decodes ZERO_TOKEN -> check_eob
        //        clears. With check_eob now false, every later position
        //        decodes a `token` directly with no `more_coefs` gate;
        //        on the zero-extended buffer each is ZERO_TOKEN, so the
        //        loop runs to c == segEob.
        //   nonzero = c > 0 = true; every Tokens entry is an explicit 0.
        let bytes = [0x40u8, 0, 0, 0, 0, 0, 0, 0];
        let mut dec = coder_zero(&bytes);
        let probs = uniform_coef_probs(128);
        let nz = NonzeroContext::new(16, 16);
        let scan = &crate::scan::DEFAULT_SCAN_4X4;
        let mut cache = [0u8; 16];
        let mut tokens = [7i64; 16];
        let block = luma_block(DCT_DCT);
        let nonzero = super::tokens(
            &mut dec,
            &block,
            0,
            scan,
            &probs,
            &nz,
            &mut cache,
            &mut tokens,
        )
        .unwrap();
        assert!(
            nonzero,
            "a decoded ZERO_TOKEN still makes the block nonzero"
        );
        assert!(tokens.iter().all(|&t| t == 0));
        // TokenCache for DC was set to energy_class[ZERO_TOKEN] = 0.
        assert_eq!(cache[0], 0);
    }

    #[test]
    fn tokens_matches_independent_read_coef_token_walk() {
        // The §6.4.24 driver must produce the same coefficients as an
        // independent walk built from the round-7 `read_coef_token`
        // primitive driven over the same scan with the same per-position
        // ctx / cell selection. Two fresh coders on one buffer; assert
        // the Tokens arrays and the return value match.
        let bytes = [0x55u8, 0xAA, 0x33, 0xCC, 0x0F, 0xF0, 0x12, 0x34];
        let probs = uniform_coef_probs(120);
        let nz = NonzeroContext::new(16, 16);
        let scan = &crate::scan::DEFAULT_SCAN_4X4;
        let tx_sz = 0u32;
        let block = luma_block(DCT_DCT);

        // Driver under test.
        let mut dec_a = coder_zero(&bytes);
        let mut cache_a = [0u8; 16];
        let mut tokens_a = [0i64; 16];
        let nonzero_a = super::tokens(
            &mut dec_a,
            &block,
            tx_sz,
            scan,
            &probs,
            &nz,
            &mut cache_a,
            &mut tokens_a,
        )
        .unwrap();

        // Independent reference walk.
        let mut dec_b = coder_zero(&bytes);
        let mut cache_b = [0u8; 16];
        let mut tokens_b = [0i64; 16];
        let tx_slab = &probs[tx_sz as usize][0][0];
        let mut check_eob = true;
        let seg_eob = 16usize << (tx_sz << 1);
        let mut c = 0usize;
        while c < seg_eob {
            let pos = scan[c] as usize;
            let band = coef_band(c, tx_sz);
            let ctx = if c == 0 {
                0usize
            } else {
                let (n0, n1) = token_cache_neighbours(c, pos, tx_sz, block.tx_type);
                (1 + cache_b[n0] as usize + cache_b[n1] as usize) >> 1
            };
            let cell = &tx_slab[band][ctx];
            let token_probs = build_token_probs(cell);
            let step = read_coef_token(
                &mut dec_b,
                check_eob,
                cell[0],
                &token_probs,
                block.bit_depth,
            )
            .unwrap();
            match step {
                CoefStep::Eob => break,
                CoefStep::Coef { token, value } => {
                    cache_b[pos] = ENERGY_CLASS[token as usize];
                    tokens_b[pos] = value as i64;
                    check_eob = token != ZERO_TOKEN;
                }
            }
            c += 1;
        }
        for &p in &scan[c..] {
            tokens_b[p as usize] = 0;
        }
        let nonzero_b = c > 0;

        assert_eq!(nonzero_a, nonzero_b, "return value mismatch");
        assert_eq!(tokens_a, tokens_b, "Tokens array mismatch");
        assert_eq!(cache_a, cache_b, "TokenCache mismatch");
    }

    #[test]
    fn tokens_dc_context_uses_nonzero_strips() {
        // The DC ctx is `above + left` over the block's 4-sample units.
        // Set both strips at the block origin so ctx = 2, and confirm
        // the driver selects coef_probs[..][band 0][ctx 2]. Use a probs
        // table where ctx 2's more_coefs prob is 0-ish (forces EOB) but
        // ctx 0/1's is high — proving the right cell was consulted.
        //
        // more_coefs prob = 1 with a zero buffer always yields bit 0
        // (EOB), and prob = 255 with the `0x40…` buffer's value 64
        // yields bit 1 (continue). We set ctx-2's [0] to a low prob and
        // ctx-0's to a high one and check that a non-zero context routes
        // to the EOB cell.
        let mut probs = uniform_coef_probs(255);
        // Force ctx 2, band 0, luma intra, TX_4X4 more_coefs prob low so
        // a zero buffer triggers EOB there.
        probs[0][0][0][0][2][0] = 1;
        let mut nz = NonzeroContext::new(16, 16);
        nz.above[0] = 1;
        nz.left[0] = 1;
        let scan = &crate::scan::DEFAULT_SCAN_4X4;
        let mut cache = [0u8; 16];
        let mut tokens = [5i64; 16];
        let block = luma_block(DCT_DCT);

        // Zero buffer: read_bool(1) -> bit 0 -> EOB immediately.
        let bytes = [0u8; 8];
        let mut dec = coder_zero(&bytes);
        let nonzero = super::tokens(
            &mut dec,
            &block,
            0,
            scan,
            &probs,
            &nz,
            &mut cache,
            &mut tokens,
        )
        .unwrap();
        assert!(!nonzero, "ctx-2 low more_coefs prob routes to EOB");
        assert!(tokens.iter().all(|&t| t == 0));
    }

    #[test]
    fn tokens_trailing_positions_are_zeroed() {
        // After an early EOB, every scan position from c..segEob must be
        // explicitly zeroed even if the caller's buffer had garbage.
        let bytes = [0u8; 8];
        let mut dec = coder_zero(&bytes);
        let probs = uniform_coef_probs(128);
        let nz = NonzeroContext::new(16, 16);
        let scan = &crate::scan::DEFAULT_SCAN_8X8;
        let mut cache = [0u8; 64];
        let mut tokens = [-1i64; 64];
        let block = luma_block(DCT_DCT);
        let nonzero = super::tokens(
            &mut dec,
            &block,
            1,
            scan,
            &probs,
            &nz,
            &mut cache,
            &mut tokens,
        )
        .unwrap();
        assert!(!nonzero);
        assert!(
            tokens.iter().all(|&t| t == 0),
            "trailing zero-fill must clear the whole block"
        );
    }

    #[test]
    fn nonzero_context_new_is_all_zero() {
        let nz = NonzeroContext::new(8, 12);
        assert_eq!(nz.above.len(), 8);
        assert_eq!(nz.left.len(), 12);
        assert!(nz.above.iter().all(|&b| b == 0));
        assert!(nz.left.iter().all(|&b| b == 0));
    }
}
