//! VP9 per-frame entropy context — dynamic probability tables updated
//! by the §6.3 compressed header and §8.4 backward adaptation.
//!
//! The spec keeps a `FrameContext` snapshot per frame, seeded from either
//! the default tables (§10.5) on keyframes / intra_only / reset, or from
//! the saved context slot `frame_context_idx` on inter frames. The
//! §6.3 compressed header then applies `diff_update_prob` deltas on top,
//! and §8.4 backward adaptation folds per-frame counts back into the
//! probabilities at frame end if `refresh_frame_context` is set.
//!
//! This module carries the subset of §10.5 tables that the downstream
//! decoder actually consumes, boxed so the struct is cheap to move and
//! lives on the heap (the full coefficient table is ~1500 bytes × 3
//! nodes per context; allocating on the stack for every frame is wasteful
//! and risks stack overflow in tight recursion).

use oxideav_core::Result;

use crate::bool_decoder::BoolDecoder;
use crate::tables::{COEF_PROBS_16X16, COEF_PROBS_32X32, COEF_PROBS_4X4, COEF_PROBS_8X8};

/// Number of transform sizes — §3 (TX_SIZES).
pub const TX_SIZES: usize = 4;
/// Number of plane types — §3 (BLOCK_TYPES: Y, UV).
pub const BLOCK_TYPES: usize = 2;
/// Number of reference types — §3 (REF_TYPES: intra, inter).
pub const REF_TYPES: usize = 2;
/// Number of coefficient bands — §3 (COEF_BANDS).
pub const COEF_BANDS: usize = 6;
/// Number of coefficient contexts per band — §3 (PREV_COEF_CONTEXTS).
pub const COEF_CTX: usize = 6;
/// Number of coefficient tree probabilities — §3 (UNCONSTRAINED_NODES).
pub const COEF_NODES: usize = 3;

/// Number of skip contexts — §3 (SKIP_CONTEXTS).
pub const SKIP_CONTEXTS: usize = 3;
/// Number of tx_size contexts — §3 (TX_SIZE_CONTEXTS).
pub const TX_SIZE_CONTEXTS: usize = 2;
/// Number of inter-mode contexts — §3 (INTER_MODE_CONTEXTS).
pub const INTER_MODE_CONTEXTS: usize = 7;
/// INTER_MODES − 1 — the inter-mode decode tree has 4 leaves / 3 probs.
pub const INTER_MODE_PROBS: usize = 3;
/// Number of interpolation filter contexts — §3 (INTERP_FILTER_CONTEXTS).
pub const INTERP_FILTER_CONTEXTS: usize = 4;
/// SWITCHABLE_FILTERS − 1 — the filter decode tree has 3 leaves / 2 probs.
pub const INTERP_FILTER_PROBS: usize = 2;
/// Number of is_inter contexts — §3 (IS_INTER_CONTEXTS).
pub const IS_INTER_CONTEXTS: usize = 4;
/// Number of comp_mode contexts — §3 (COMP_MODE_CONTEXTS).
pub const COMP_MODE_CONTEXTS: usize = 5;
/// Number of single-/comp-ref contexts — §3 (REF_CONTEXTS).
pub const REF_CONTEXTS: usize = 5;

/// Coefficient probability table — `[tx_size][plane_type][ref_type][band][ctx][node]`.
///
/// Note: for `band == 0` the spec caps `ctx < 3` (maxL = 3); positions
/// beyond that are never consulted and stay zero-filled.
pub type CoefProbTable =
    [[[[[[u8; COEF_NODES]; COEF_CTX]; COEF_BANDS]; REF_TYPES]; BLOCK_TYPES]; TX_SIZES];

/// Partition probs — `[PARTITION_CONTEXTS][PARTITION_TYPES-1]`. Carried
/// here so §6.3.15 updates can apply on top of the §10.5 default.
pub const PARTITION_CONTEXTS: usize = 16;
pub const PARTITION_TYPES_M1: usize = 3;
/// Number of Y-mode block size groups — §3 (BLOCK_SIZE_GROUPS).
pub const BLOCK_SIZE_GROUPS: usize = 4;
/// INTRA_MODES − 1 — the intra-mode decode tree has 10 leaves / 9 probs.
pub const INTRA_MODES_M1: usize = 9;
/// INTRA_MODES — used for UV-mode context count.
pub const INTRA_MODES: usize = 10;

/// Default MV joint probabilities — §10.5. (`MV_JOINTS-1 = 3`.)
pub const DEFAULT_MV_JOINT_PROBS: [u8; 3] = [32, 64, 96];

/// MV_CLASSES − 1 = 10; CLASS0_SIZE = 2; MV_OFFSET_BITS = 10; MV_FR_SIZE = 4.
pub const MV_CLASSES_M1: usize = 10;
pub const CLASS0_SIZE: usize = 2;
pub const MV_OFFSET_BITS: usize = 10;
pub const MV_FR_SIZE: usize = 4;
pub const MV_FR_SIZE_M1: usize = 3;

/// Per-component MV probability sub-table — one per {horizontal, vertical}.
#[derive(Clone, Debug)]
pub struct MvComponentProbs {
    pub sign: u8,
    pub classes: [u8; MV_CLASSES_M1],
    pub class0_bit: u8,
    pub bits: [u8; MV_OFFSET_BITS],
    pub class0_fr: [[u8; MV_FR_SIZE_M1]; CLASS0_SIZE],
    pub fr: [u8; MV_FR_SIZE_M1],
    pub class0_hp: u8,
    pub hp: u8,
}

impl Default for MvComponentProbs {
    fn default() -> Self {
        // §10.5 default_mv_context values.
        Self {
            sign: 128,
            classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
            class0_bit: 216,
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fr: [[128, 128, 64], [96, 112, 64]],
            fr: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        }
    }
}

/// Full MV probabilities — §10.5 default_mv_context.
#[derive(Clone, Debug)]
pub struct MvProbs {
    pub joints: [u8; 3],
    pub comps: [MvComponentProbs; 2],
}

impl Default for MvProbs {
    fn default() -> Self {
        Self {
            joints: DEFAULT_MV_JOINT_PROBS,
            comps: [MvComponentProbs::default(), MvComponentProbs::default()],
        }
    }
}

/// The per-frame entropy state. Initialised from §10.5 defaults, then
/// overwritten by the §6.3 compressed-header updates, then finally
/// written back by §8.4 backward adaptation at end-of-frame.
#[derive(Clone, Debug)]
pub struct FrameContext {
    /// Coefficient probabilities — one big six-dim table per the spec.
    pub coef_probs: Box<CoefProbTable>,
    /// §10.5 default_skip_prob. Default: [192, 128, 64].
    pub skip_probs: [u8; SKIP_CONTEXTS],
    /// §10.5 default_tx_probs_8x8 / _16x16 / _32x32.
    pub tx_probs_8x8: [[u8; 1]; TX_SIZE_CONTEXTS],
    pub tx_probs_16x16: [[u8; 2]; TX_SIZE_CONTEXTS],
    pub tx_probs_32x32: [[u8; 3]; TX_SIZE_CONTEXTS],
    /// §10.5 default_inter_mode_probs.
    pub inter_mode_probs: [[u8; INTER_MODE_PROBS]; INTER_MODE_CONTEXTS],
    /// §10.5 default_interp_filter_probs.
    pub interp_filter_probs: [[u8; INTERP_FILTER_PROBS]; INTERP_FILTER_CONTEXTS],
    /// §10.5 default_is_inter_prob.
    pub is_inter_prob: [u8; IS_INTER_CONTEXTS],
    /// §10.5 default_comp_mode_prob.
    pub comp_mode_prob: [u8; COMP_MODE_CONTEXTS],
    /// §10.5 default_single_ref_prob (2 probs per context).
    pub single_ref_prob: [[u8; 2]; REF_CONTEXTS],
    /// §10.5 default_comp_ref_prob.
    pub comp_ref_prob: [u8; REF_CONTEXTS],
    /// §10.5 default_y_mode_probs (block-size-group conditioned).
    pub y_mode_probs: [[u8; INTRA_MODES_M1]; BLOCK_SIZE_GROUPS],
    /// §10.5 default_uv_mode_probs (Y-mode conditioned).
    pub uv_mode_probs: [[u8; INTRA_MODES_M1]; INTRA_MODES],
    /// §10.5 default_partition_probs.
    pub partition_probs: [[u8; PARTITION_TYPES_M1]; PARTITION_CONTEXTS],
    /// §10.5 default_mv_probs.
    pub mv_probs: MvProbs,
}

impl Default for FrameContext {
    fn default() -> Self {
        Self::new_default()
    }
}

impl FrameContext {
    /// Build a fresh `FrameContext` seeded with the §10.5 default tables.
    pub fn new_default() -> Self {
        let mut coef_probs: Box<CoefProbTable> = Box::new(
            [[[[[[0u8; COEF_NODES]; COEF_CTX]; COEF_BANDS]; REF_TYPES]; BLOCK_TYPES]; TX_SIZES],
        );
        fill_coef_probs(&mut coef_probs);
        Self {
            coef_probs,
            // §10.5 default_skip_prob.
            skip_probs: [192, 128, 64],
            // §10.5 default_tx_probs. TX_SIZE_CONTEXTS = 2.
            tx_probs_8x8: [[100], [66]],
            tx_probs_16x16: [[20, 152], [15, 101]],
            tx_probs_32x32: [[3, 136, 37], [5, 52, 13]],
            // §10.5 default_inter_mode_probs.
            inter_mode_probs: [
                [2, 173, 34],
                [7, 145, 85],
                [7, 166, 63],
                [7, 94, 66],
                [8, 64, 46],
                [17, 81, 31],
                [25, 29, 30],
            ],
            // §10.5 default_interp_filter_probs.
            interp_filter_probs: [[235, 162], [36, 255], [34, 3], [149, 144]],
            // §10.5 default_is_inter_prob.
            is_inter_prob: [9, 102, 187, 225],
            // §10.5 default_comp_mode_prob.
            comp_mode_prob: [239, 183, 119, 96, 41],
            // §10.5 default_single_ref_prob.
            single_ref_prob: [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]],
            // §10.5 default_comp_ref_prob.
            comp_ref_prob: [50, 126, 123, 221, 226],
            // §10.5 default_y_mode_probs (BLOCK_SIZE_GROUPS=4).
            y_mode_probs: [
                [65, 32, 18, 144, 162, 194, 41, 51, 98],
                [132, 68, 18, 165, 217, 196, 45, 40, 78],
                [173, 80, 19, 176, 240, 193, 64, 35, 46],
                [221, 135, 38, 194, 248, 121, 96, 85, 29],
            ],
            // §10.5 default_uv_mode_probs (indexed by Y-mode).
            uv_mode_probs: [
                [120, 7, 76, 176, 208, 126, 28, 221, 29],
                [48, 12, 154, 155, 139, 90, 34, 117, 119],
                [67, 6, 25, 204, 243, 158, 13, 21, 96],
                [97, 5, 44, 131, 176, 210, 49, 14, 55],
                [83, 5, 42, 156, 111, 152, 26, 49, 152],
                [80, 5, 58, 178, 74, 83, 33, 62, 145],
                [86, 5, 32, 154, 192, 168, 14, 22, 163],
                [85, 5, 32, 156, 216, 148, 19, 65, 117],
                [88, 5, 55, 147, 83, 175, 94, 63, 133],
                [109, 5, 54, 166, 105, 107, 62, 73, 122],
            ],
            // §10.5 default_partition_probs (non-key defaults).
            partition_probs: [
                [199, 122, 141],
                [147, 63, 159],
                [148, 133, 118],
                [121, 104, 114],
                [174, 73, 87],
                [92, 41, 83],
                [82, 99, 50],
                [53, 39, 39],
                [177, 58, 59],
                [68, 26, 63],
                [52, 79, 25],
                [17, 14, 12],
                [222, 34, 30],
                [72, 16, 44],
                [58, 32, 12],
                [10, 7, 6],
            ],
            mv_probs: MvProbs::default(),
        }
    }
}

/// Inverse remap table (§6.3.5). The spec's `inv_map_table[MAX_PROB]`
/// array maps the 8-bit encoded delta index back into the 1..=254
/// probability range. Values 0 and 255 are reserved (they would mean
/// "absolute zero" / "absolute max"), so the table has 254 entries.
pub const INV_MAP_TABLE: [u8; 254] = [
    7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 1,
    2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
    187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 251, 252, 253,
];

/// §6.3.4 `decode_term_subexp` — variable-length delta decode. Reads
/// between 4 and 8 bits to cover the full 0..=254 range.
fn decode_term_subexp(bd: &mut BoolDecoder<'_>) -> Result<u32> {
    let b0 = bd.read_literal(1)?;
    if b0 == 0 {
        return bd.read_literal(4);
    }
    let b1 = bd.read_literal(1)?;
    if b1 == 0 {
        return Ok(bd.read_literal(4)? + 16);
    }
    let b2 = bd.read_literal(1)?;
    if b2 == 0 {
        return Ok(bd.read_literal(5)? + 32);
    }
    let v = bd.read_literal(7)?;
    if v < 65 {
        return Ok(v + 64);
    }
    let bit = bd.read_literal(1)?;
    Ok((v << 1) - 1 + bit)
}

/// §6.3.6 `inv_recenter_nonneg(v, m)`.
fn inv_recenter_nonneg(v: u32, m: u32) -> u32 {
    if v > 2 * m {
        return v;
    }
    if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// §6.3.5 `inv_remap_prob(deltaProb, prob)`. The spec text requires
/// `deltaProb < MAX_PROB` (i.e. < 255); the `INV_MAP_TABLE` lookup
/// covers indices 0..=253. Zero-valued current probs (used for unused
/// slots in the coef table) are passed through unchanged — the zero
/// slots should never see an update bit set in practice, but being
/// defensive here avoids underflow on malformed streams.
fn inv_remap_prob(delta_prob: u32, prob: u8) -> u8 {
    if prob == 0 {
        return 0;
    }
    let idx = (delta_prob as usize).min(253);
    let v = INV_MAP_TABLE[idx] as u32;
    let m_minus = (prob as u32) - 1; // spec's `m--`.
    let out = if (m_minus << 1) <= 255 {
        1 + inv_recenter_nonneg(v, m_minus)
    } else {
        255 - inv_recenter_nonneg(v, 255 - 1 - m_minus)
    };
    out.min(255) as u8
}

/// §6.3.3 `diff_update_prob(prob)`. Reads the `B(252)` update flag; if
/// set, decodes a variable-length delta and applies the §6.3.5 remap.
pub fn diff_update_prob(bd: &mut BoolDecoder<'_>, prob: u8) -> Result<u8> {
    if bd.read(252)? == 0 {
        return Ok(prob);
    }
    let delta = decode_term_subexp(bd)?;
    Ok(inv_remap_prob(delta, prob))
}

/// §6.3.17 `update_mv_prob(prob)`. Reads the `B(252)` flag; if set,
/// decodes an `L(7)` absolute value and rebuilds the probability as
/// `(v << 1) | 1`.
pub fn update_mv_prob(bd: &mut BoolDecoder<'_>, prob: u8) -> Result<u8> {
    if bd.read(252)? == 0 {
        return Ok(prob);
    }
    let v = bd.read_literal(7)? as u8;
    Ok((v << 1) | 1)
}

fn fill_coef_probs(out: &mut CoefProbTable) {
    // tx_size = 0 (4×4)
    for p in 0..BLOCK_TYPES {
        for r in 0..REF_TYPES {
            for b in 0..COEF_BANDS {
                for c in 0..COEF_CTX {
                    for n in 0..COEF_NODES {
                        out[0][p][r][b][c][n] = COEF_PROBS_4X4[p][r][b][c][n];
                        out[1][p][r][b][c][n] = COEF_PROBS_8X8[p][r][b][c][n];
                        out[2][p][r][b][c][n] = COEF_PROBS_16X16[p][r][b][c][n];
                        out[3][p][r][b][c][n] = COEF_PROBS_32X32[p][r][b][c][n];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_context_seeds_from_spec_tables() {
        let ctx = FrameContext::new_default();
        // Spot-check: coef_probs[0][0][0][0][0] matches COEF_PROBS_4X4[0][0][0][0].
        assert_eq!(ctx.coef_probs[0][0][0][0][0], COEF_PROBS_4X4[0][0][0][0]);
        assert_eq!(ctx.coef_probs[3][1][1][5][5], COEF_PROBS_32X32[1][1][5][5]);
        assert_eq!(ctx.skip_probs, [192, 128, 64]);
        assert_eq!(ctx.mv_probs.joints, [32, 64, 96]);
    }

    #[test]
    fn context_is_small_to_move() {
        // `FrameContext` must fit on the stack (boxed coef table keeps
        // the shell bounded). Size check — exact value isn't important,
        // but it should be well under 4 KiB.
        assert!(std::mem::size_of::<FrameContext>() < 4096);
    }
}
