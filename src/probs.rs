//! VP9 default probability tables — §10.5.
//!
//! This module carries only the tables strictly needed by the code that
//! currently lives in this crate. Right now that's
//! [`KF_PARTITION_PROBS`] (keyframe partition tree probs) plus
//! [`PARTITION_PROBS`] (non-key defaults). All other §10.5 tables
//! (coefficient / skip / inter-mode / MV / Y-mode) are intentionally
//! omitted — the decoder stops before they are consumed, and carrying
//! them would bloat the crate without exercising them.
//!
//! The values below match the `kf_partition_probs` and
//! `default_partition_probs` arrays in the VP9 reference (libvpx
//! `vp9_entropymode.c`) and the specification text at §10.5.
//!
//! Access pattern (keyframe):
//!
//! ```text
//!   ctx = above_ctx | left_ctx   (0..3 derived from §6.4.2 §7.3.2)
//!   bsl = block-size log2 (0=64x64 .. 3=8x8)
//!   p   = KF_PARTITION_PROBS[ctx + bsl * 4][0..3]
//! ```
//!
//! The boolean engine reads three conditional splits to pick one of
//! `{PARTITION_NONE, HORZ, VERT, SPLIT}` (§6.4.2 Tree 10-3).

/// Keyframe partition probabilities (§10.5, `kf_partition_probs`).
///
/// Indexed by `[PARTITION_CONTEXT + 4 * BLOCK_SIZE_LOG2][3]`:
/// * PARTITION_CONTEXT: 0..=3 — derived from the left / above availability
///   and size of already-decoded neighbours (§6.4.2).
/// * BLOCK_SIZE_LOG2: 0..=3 — 0 is 64×64, 1 is 32×32, 2 is 16×16, 3 is 8×8.
///
/// The three probabilities are read in order by the boolean engine:
/// * prob 0 — P(not SPLIT).
/// * prob 1 — P(HORZ | not SPLIT).
/// * prob 2 — P(VERT | not SPLIT and not HORZ).
///
/// (Equivalently the decode tree: first bit chooses SPLIT vs non-SPLIT,
/// second chooses HORZ vs {VERT, NONE}, third chooses VERT vs NONE.)
pub const KF_PARTITION_PROBS: [[u8; 3]; 16] = [
    // 64×64 (indexed as ctx = (3 - bsl) * 4 + ...)
    [158, 97, 94],
    [93, 24, 99],
    [85, 119, 44],
    [62, 59, 67],
    // 32×32
    [149, 53, 53],
    [94, 20, 48],
    [83, 53, 24],
    [52, 18, 18],
    // 16×16
    [150, 40, 39],
    [78, 12, 26],
    [67, 33, 11],
    [24, 7, 5],
    // 8×8 — ONLY_4X4 is implicit at this level.
    [174, 35, 49],
    [68, 11, 27],
    [57, 15, 9],
    [12, 3, 3],
];

/// Non-key default partition probabilities (§10.5, `default_partition_probs`).
///
/// Same layout as [`KF_PARTITION_PROBS`]; loaded at every `KEY_FRAME` /
/// `intra_only` / `reset_frame_context >= 2` boundary.
pub const PARTITION_PROBS: [[u8; 3]; 16] = [
    // 64×64
    [199, 122, 141],
    [147, 63, 159],
    [148, 133, 118],
    [121, 104, 114],
    // 32×32
    [174, 73, 87],
    [92, 41, 83],
    [82, 99, 50],
    [53, 39, 39],
    // 16×16
    [177, 58, 59],
    [68, 26, 63],
    [52, 79, 25],
    [17, 14, 12],
    // 8×8
    [222, 34, 30],
    [72, 16, 44],
    [58, 32, 12],
    [10, 7, 6],
];

/// VP9 partition types (§6.4.2 Table 9-5). The bit-tree is read as
/// `{ SPLIT? : { HORZ? : { VERT? : NONE }}}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionType {
    None = 0,
    Horz = 1,
    Vert = 2,
    Split = 3,
}

/// Decode one partition symbol from the bool engine against the three
/// probabilities `probs` per §6.4.2 Tree 10-3. The tree-shape follows
/// the reference libvpx:
///
/// ```text
///   -PARTITION_NONE      // pk = probs[0]
///    -PARTITION_HORZ     // pk = probs[1]
///     -PARTITION_VERT    // pk = probs[2]
///     -PARTITION_SPLIT
/// ```
pub fn read_partition_from_tree(
    bd: &mut crate::bool_decoder::BoolDecoder<'_>,
    probs: [u8; 3],
) -> oxideav_core::Result<PartitionType> {
    let bit0 = bd.read(probs[0])?;
    if bit0 == 0 {
        return Ok(PartitionType::None);
    }
    let bit1 = bd.read(probs[1])?;
    if bit1 == 0 {
        return Ok(PartitionType::Horz);
    }
    let bit2 = bd.read(probs[2])?;
    if bit2 == 0 {
        Ok(PartitionType::Vert)
    } else {
        Ok(PartitionType::Split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_sizes_match_spec() {
        assert_eq!(KF_PARTITION_PROBS.len(), 16);
        assert_eq!(PARTITION_PROBS.len(), 16);
        for row in KF_PARTITION_PROBS.iter() {
            assert_eq!(row.len(), 3);
        }
    }
}
