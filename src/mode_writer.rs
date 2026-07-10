//! VP9 keyframe intra **mode-info writer** — the inverse of the §6.4.6 /
//! §6.4.7 / §6.4.8 / §6.4.10 mode-info decode primitives in `mode_info`.
//!
//! A keyframe intra block carries (in §6.4.6 order): `intra_segment_id`,
//! `read_skip`, `read_tx_size( 1 )`, then `default_intra_mode` (the luma
//! mode, replicated to the four sub-modes for `MiSize >= BLOCK_8X8`) and
//! `default_uv_mode`. This module provides the per-element writers — each
//! the exact inverse of the matching `mode_info` decoder, driven by the
//! §9.2 [`BoolEncoder`] — so a written mode-info stream decodes back to
//! the same syntax values.
//!
//! All the syntax elements here are §9.3.1 tree walks, so the core is a
//! generic [`tree_encode`] that resolves the bit path to a target leaf
//! the same way `mode_info::tree_decode` walks it.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.6-§6.4.10 / §9.3.1 / §9.3.2; the
//! bit order mirrors the in-crate decoder exactly.

use crate::bool_encoder::BoolEncoder;
use crate::compressed::{tx_mode_to_biggest_tx_size, TxMode};
use crate::mode_info::{
    BINARY_TREE, INTRA_MODE_TREE, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, SEGMENT_TREE, TX_SIZE_16_TREE,
    TX_SIZE_32_TREE, TX_SIZE_8_TREE,
};
use crate::Error;

/// Resolve the §9.3.1 tree bit path that `tree_decode` walks to reach the
/// leaf `value`. The decoder walks `n = tree[ n + bit ]` from `n = 0`,
/// probing `prob( n >> 1 )` at each step, returning `-n` once `n <= 0`.
/// We reproduce that walk, choosing at each node the child whose subtree
/// contains the target leaf. Returns the `(node_index, bit)` pairs.
fn tree_path(tree: &[i32], value: i32) -> Result<Vec<(usize, u32)>, Error> {
    let target = -value;
    let mut path = Vec::new();
    let mut n: i32 = 0;
    for _ in 0..tree.len() {
        let node = (n >> 1) as usize;
        let mut chosen = None;
        for bit in 0u32..2 {
            let idx = (n as usize) + bit as usize;
            if subtree_contains(tree, tree[idx], target) {
                chosen = Some((bit, tree[idx]));
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

/// Does the subtree rooted at `node` (leaf when `<= 0`, internal
/// child-pair base when `> 0`) contain the leaf `target`?
fn subtree_contains(tree: &[i32], node: i32, target: i32) -> bool {
    if node <= 0 {
        return node == target;
    }
    let base = node as usize;
    subtree_contains(tree, tree[base], target) || subtree_contains(tree, tree[base + 1], target)
}

/// Generic §9.3.1 tree-write: emit the bit path to `value` under `tree`,
/// coding each branch with `prob( node )`. Inverse of
/// `mode_info::tree_decode`.
pub(crate) fn tree_encode<F>(
    enc: &mut BoolEncoder,
    tree: &[i32],
    value: i32,
    mut prob: F,
) -> Result<(), Error>
where
    F: FnMut(usize) -> u8,
{
    for (node, bit) in tree_path(tree, value)? {
        enc.write_bool(bit, u32::from(prob(node)));
    }
    Ok(())
}

/// §6.4.8 `read_skip` inverse. When `seg_feature_skip_active`, the
/// decoder hardwires `skip = 1` and reads no bits, so nothing is written
/// — the caller must pass `skip == true` in that case.
pub(crate) fn write_skip(
    enc: &mut BoolEncoder,
    skip: bool,
    seg_feature_skip_active: bool,
    skip_prob: &[u8; 3],
    ctx: usize,
) -> Result<(), Error> {
    if seg_feature_skip_active {
        debug_assert!(skip, "seg_feature_active(SKIP) forces skip == 1");
        return Ok(());
    }
    tree_encode(enc, &BINARY_TREE, i32::from(skip), |_| skip_prob[ctx])
}

/// §6.4.7 `read_segment_id` inverse — the §9.3.1 7-leaf `SEGMENT_TREE`
/// under `segmentation_tree_probs[ node ]`.
pub(crate) fn write_segment_id(
    enc: &mut BoolEncoder,
    segment_id: u8,
    tree_probs: &[u8; 7],
) -> Result<(), Error> {
    tree_encode(enc, &SEGMENT_TREE, i32::from(segment_id), |node| {
        tree_probs[node]
    })
}

/// §6.4.6 `default_intra_mode` inverse — the §9.3.1 `INTRA_MODE_TREE`
/// under `kf_y_mode_probs[ abovemode ][ leftmode ]`.
pub(crate) fn write_default_intra_mode(
    enc: &mut BoolEncoder,
    y_mode: u8,
    abovemode: u8,
    leftmode: u8,
) -> Result<(), Error> {
    let row = &KF_Y_MODE_PROBS[abovemode as usize][leftmode as usize];
    tree_encode(enc, &INTRA_MODE_TREE, i32::from(y_mode), |node| row[node])
}

/// §6.4.6 `default_uv_mode` inverse — the §9.3.1 `INTRA_MODE_TREE` under
/// `kf_uv_mode_probs[ y_mode ]`.
pub(crate) fn write_default_uv_mode(
    enc: &mut BoolEncoder,
    uv_mode: u8,
    y_mode: u8,
) -> Result<(), Error> {
    let row = &KF_UV_MODE_PROBS[y_mode as usize];
    tree_encode(enc, &INTRA_MODE_TREE, i32::from(uv_mode), |node| row[node])
}

/// §6.4.10 `read_tx_size` inverse. Only emits bits when the decoder
/// would read them (`allow_select && tx_mode == TX_MODE_SELECT &&
/// MiSize >= BLOCK_8X8`); otherwise the tx size is inferred and nothing
/// is written. `max_tx_size` is `MAX_TXSIZE_LOOKUP[ MiSize ]` and `ctx`
/// is the §9.3.2 `tx_size_context`.
pub(crate) fn write_tx_size(
    enc: &mut BoolEncoder,
    tx_size: u32,
    coded: bool,
    max_tx_size: u32,
    tx_probs_row: &[u8; 3],
) -> Result<(), Error> {
    if !coded {
        return Ok(());
    }
    let tree: &[i32] = match max_tx_size {
        3 => &TX_SIZE_32_TREE,
        2 => &TX_SIZE_16_TREE,
        _ => &TX_SIZE_8_TREE,
    };
    tree_encode(enc, tree, tx_size as i32, |node| tx_probs_row[node])
}

/// Returns `true` when §6.4.10 `read_tx_size` would code a tx-size tree
/// (vs. infer it), given the frame `tx_mode`, the block `mi_size`, and
/// whether `read_tx_size` was invoked with `allow_select == 1`. Mirrors
/// the decoder's gate so the encoder and decoder agree on bit presence.
pub(crate) fn tx_size_is_coded(allow_select: bool, tx_mode: TxMode, mi_size_ge_8x8: bool) -> bool {
    allow_select && matches!(tx_mode, TxMode::TxModeSelect) && mi_size_ge_8x8
}

/// The inferred (non-coded) tx size: `min(max_tx_size,
/// tx_mode_to_biggest_tx_size(tx_mode))`, matching `read_tx_size`'s else
/// branch.
pub(crate) fn inferred_tx_size(max_tx_size: u32, tx_mode: TxMode) -> u32 {
    max_tx_size.min(tx_mode_to_biggest_tx_size(tx_mode) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::mode_info::{
        default_intra_mode, default_uv_mode, read_segment_id, read_skip, read_tx_size,
        NeighbourSkips, NeighbourTxSizes,
    };
    use crate::residual::{BLOCK_16X16, BLOCK_64X64, BLOCK_8X8};

    #[test]
    fn skip_roundtrips_both_values_all_contexts() {
        let skip_prob = [192u8, 128, 64];
        for ctx in 0..3usize {
            for skip in [false, true] {
                let mut enc = BoolEncoder::new();
                write_skip(&mut enc, skip, false, &skip_prob, ctx).unwrap();
                let buf = enc.finish();
                let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
                // The decoder derives ctx from neighbours; drive it
                // directly with a matching neighbour set.
                let nb = match ctx {
                    0 => NeighbourSkips {
                        above: None,
                        left: None,
                    },
                    1 => NeighbourSkips {
                        above: Some(1),
                        left: None,
                    },
                    _ => NeighbourSkips {
                        above: Some(1),
                        left: Some(1),
                    },
                };
                let got =
                    read_skip(&mut dec, false, &skip_prob, nb, &mut Default::default()).unwrap();
                assert_eq!(got, skip, "skip {skip} ctx {ctx}");
            }
        }
    }

    #[test]
    fn seg_skip_active_writes_nothing() {
        let skip_prob = [192u8, 128, 64];
        let mut enc = BoolEncoder::new();
        write_skip(&mut enc, true, true, &skip_prob, 0).unwrap();
        let buf = enc.finish();
        // Only the marker + flush — decoding read_skip with
        // seg_feature_active short-circuits to 1 reading nothing.
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        assert!(read_skip(
            &mut dec,
            true,
            &skip_prob,
            NeighbourSkips::default(),
            &mut Default::default()
        )
        .unwrap());
    }

    #[test]
    fn segment_id_roundtrips_all_ids() {
        let tree_probs = [200u8, 100, 128, 64, 32, 150, 90];
        for sid in 0..8u8 {
            let mut enc = BoolEncoder::new();
            write_segment_id(&mut enc, sid, &tree_probs).unwrap();
            let buf = enc.finish();
            let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
            assert_eq!(read_segment_id(&mut dec, &tree_probs).unwrap(), sid);
        }
    }

    #[test]
    fn default_intra_mode_roundtrips_all_modes() {
        // A spread of (above, left) contexts and every target mode.
        for &(am, lm) in &[(0u8, 0u8), (3, 5), (9, 9), (1, 7), (6, 2)] {
            for mode in 0..10u8 {
                let mut enc = BoolEncoder::new();
                write_default_intra_mode(&mut enc, mode, am, lm).unwrap();
                let buf = enc.finish();
                let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
                assert_eq!(
                    default_intra_mode(&mut dec, am, lm).unwrap(),
                    mode,
                    "y_mode {mode} ctx ({am},{lm})"
                );
            }
        }
    }

    #[test]
    fn default_uv_mode_roundtrips_all_modes() {
        for y_mode in [0u8, 4, 9] {
            for uv in 0..10u8 {
                let mut enc = BoolEncoder::new();
                write_default_uv_mode(&mut enc, uv, y_mode).unwrap();
                let buf = enc.finish();
                let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
                assert_eq!(default_uv_mode(&mut dec, y_mode).unwrap(), uv);
            }
        }
    }

    #[test]
    fn tx_size_roundtrips_select_path() {
        // TX_MODE_SELECT + BLOCK >= 8x8: the tree is coded. Drive every
        // tx-size for each max_tx_size tree.
        use crate::mode_info::tx_size_context;
        let tx_probs = crate::compressed::DEFAULT_TX_PROBS;
        let nb = NeighbourTxSizes {
            avail_u: false,
            avail_l: false,
            skip_above: 0,
            skip_left: 0,
            tx_above: 0,
            tx_left: 0,
        };
        for max_tx in 1u32..=3 {
            // The encoder must pick the same probability row the decoder
            // derives from the neighbour context.
            let ctx = tx_size_context(nb, max_tx);
            let row = &tx_probs[max_tx as usize][ctx];
            let mi_size = match max_tx {
                3 => BLOCK_64X64,
                2 => BLOCK_16X16,
                _ => BLOCK_8X8,
            };
            for tx in 0..=max_tx {
                let mut enc = BoolEncoder::new();
                write_tx_size(&mut enc, tx, true, max_tx, row).unwrap();
                let buf = enc.finish();
                let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
                let got = read_tx_size(
                    &mut dec,
                    true,
                    TxMode::TxModeSelect,
                    mi_size,
                    &tx_probs,
                    nb,
                    &mut Default::default(),
                )
                .unwrap();
                assert_eq!(got, tx, "tx {tx} max {max_tx}");
            }
        }
    }

    #[test]
    fn tx_size_not_coded_writes_nothing() {
        let tx_probs = crate::compressed::DEFAULT_TX_PROBS;
        let row = &tx_probs[3][0];
        // coded == false: nothing written, inferred value used.
        let mut enc = BoolEncoder::new();
        write_tx_size(&mut enc, 0, false, 3, row).unwrap();
        let buf = enc.finish();
        // Marker only; decode with the non-select path reading no bits.
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        let nb = NeighbourTxSizes {
            avail_u: false,
            avail_l: false,
            skip_above: 0,
            skip_left: 0,
            tx_above: 0,
            tx_left: 0,
        };
        // ONLY_4X4 inferred → 0.
        let got = read_tx_size(
            &mut dec,
            true,
            TxMode::Only4x4,
            BLOCK_8X8,
            &tx_probs,
            nb,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(got, 0);
    }

    #[test]
    fn coded_gate_helpers_match_decoder() {
        assert!(tx_size_is_coded(true, TxMode::TxModeSelect, true));
        assert!(!tx_size_is_coded(false, TxMode::TxModeSelect, true));
        assert!(!tx_size_is_coded(true, TxMode::Allow32x32, true));
        assert!(!tx_size_is_coded(true, TxMode::TxModeSelect, false));
        assert_eq!(inferred_tx_size(3, TxMode::Only4x4), 0);
        assert_eq!(inferred_tx_size(3, TxMode::Allow16x16), 2);
        assert_eq!(inferred_tx_size(1, TxMode::Allow32x32), 1);
    }
}
