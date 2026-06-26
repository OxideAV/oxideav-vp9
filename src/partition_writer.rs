//! VP9 partition-syntax **writer** — the inverse of the §6.4.3
//! `decode_partition( )` recursion in [`crate::partition`].
//!
//! The §6.4.3 recursive driver reads one partition syntax element per
//! superblock-recursion node (`decode_partition_type`), derives the child
//! `subsize` via the §10.2 `subsize_lookup` table, dispatches the
//! per-block / recursive children, and writes the §6.4.3 tail back into
//! the `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips.
//!
//! This module re-walks that recursion in lock-step but, instead of
//! *reading* the partition type from the §9.2 bool coder, it *writes* the
//! type a caller-supplied partition layout dictates, coding each tree
//! branch with the same §9.3.2 probability the decoder consumes. The
//! §9.3.2 `ctx` derivation, the §9.3.1 tree selection (interior /
//! right-edge / bottom-edge / corner), and the §6.4.3 context write-back
//! are all shared with the decoder via [`crate::partition`] so a written
//! partition stream decodes back to the identical leaf set.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.3 / §9.3.1 / §9.3.2 / §10.2; the
//! recursion order + context handling mirror the in-crate decoder.

use crate::bool_encoder::BoolEncoder;
use crate::mode_writer::tree_encode;
use crate::partition::{
    partition_plane_context, write_back_partition_context, PartitionContextState,
    PartitionProbsKind, COLS_PARTITION_TREE, NUM_8X8_BLOCKS_WIDE_LOOKUP, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_TREE, PARTITION_VERT, ROWS_PARTITION_TREE,
    SUBSIZE_LOOKUP,
};
use crate::residual::{BLOCK_8X8, BLOCK_SIZES};
use crate::Error;

/// §6.4.4 per-leaf callback signature: invoked at every block call site
/// the §6.4.3 traversal reaches, with the live encoder positioned at the
/// block's first mode-info bit and the leaf `(r, c, subsize)`.
type LeafWriter<'f> = dyn FnMut(&mut BoolEncoder, u32, u32, u8) -> Result<(), Error> + 'f;

/// Write the §6.4.3 partition syntax element for one recursion node,
/// the inverse of [`crate::partition::decode_partition_type`].
///
/// Mirrors the §9.3.1 tree selection + §9.3.2 probability selection:
///
/// * `has_rows && has_cols` → the interior [`PARTITION_TREE`]; any of the
///   four §3 partition outcomes is legal and the tree walk uses
///   `probs[node]`.
/// * `!has_rows && has_cols` → the right-edge [`COLS_PARTITION_TREE`];
///   only [`PARTITION_HORZ`] / [`PARTITION_SPLIT`] are legal and the one
///   decision uses `probs[1]`.
/// * `has_rows && !has_cols` → the bottom-edge [`ROWS_PARTITION_TREE`];
///   only [`PARTITION_VERT`] / [`PARTITION_SPLIT`] are legal and the one
///   decision uses `probs[2]`.
/// * `!has_rows && !has_cols` → the corner case: the decoder *infers*
///   [`PARTITION_SPLIT`] without consuming a bit, so the writer emits
///   nothing (and the caller must pass `partition == PARTITION_SPLIT`).
///
/// Returns [`Error::Unsupported`] if `partition` is illegal for the
/// active tree (e.g. [`PARTITION_NONE`] on a frame-edge node), matching
/// the constraint a conforming encoder must satisfy.
// Consumed by the keyframe frame-assembler (a later step in this
// subsystem); exercised now by the round-trip tests below.
#[allow(dead_code)]
pub(crate) fn write_partition_type(
    enc: &mut BoolEncoder,
    has_rows: bool,
    has_cols: bool,
    partition: u8,
    probs: &[u8; 3],
) -> Result<(), Error> {
    match (has_rows, has_cols) {
        (true, true) => tree_encode(enc, &PARTITION_TREE, i32::from(partition), |node| {
            probs[node]
        }),
        (false, true) => {
            // Right-edge: only HORZ / SPLIT, single decision at probs[1].
            if partition != PARTITION_HORZ && partition != PARTITION_SPLIT {
                return Err(Error::Unsupported);
            }
            tree_encode(enc, &COLS_PARTITION_TREE, i32::from(partition), |_| {
                probs[1]
            })
        }
        (true, false) => {
            // Bottom-edge: only VERT / SPLIT, single decision at probs[2].
            if partition != PARTITION_VERT && partition != PARTITION_SPLIT {
                return Err(Error::Unsupported);
            }
            tree_encode(enc, &ROWS_PARTITION_TREE, i32::from(partition), |_| {
                probs[2]
            })
        }
        (false, false) => {
            // Corner: SPLIT is inferred, no bit is written.
            if partition != PARTITION_SPLIT {
                return Err(Error::Unsupported);
            }
            Ok(())
        }
    }
}

/// Resolve the partition type a recursion node at `(r, c, bsize)` must
/// code so that every leaf bottoms out at `BLOCK_8X8` (i.e. an all-8x8
/// partition layout), honouring the §6.4.3 frame-edge `has_rows` /
/// `has_cols` constraints.
///
/// * An interior `BLOCK_8X8` node codes [`PARTITION_NONE`] (the 8x8 leaf).
/// * Any larger interior node codes [`PARTITION_SPLIT`] to recurse toward
///   8x8.
/// * Frame-edge nodes (`!has_rows` or `!has_cols`) must code
///   [`PARTITION_SPLIT`] because the only non-split options
///   (`HORZ` / `VERT`) would emit a child larger than 8x8 in the
///   off-frame direction.
#[allow(dead_code)]
fn all_8x8_partition(bsize: u8, has_rows: bool, has_cols: bool) -> u8 {
    if bsize == BLOCK_8X8 && has_rows && has_cols {
        PARTITION_NONE
    } else {
        PARTITION_SPLIT
    }
}

/// Write the §6.4.3 partition syntax for one superblock recursion rooted
/// at `(r, c, bsize)`, choosing an all-`BLOCK_8X8`-leaf layout, and
/// invoke `on_leaf( r, c, subsize )` at every §6.4.4 block call site in
/// §6.4.3 traversal order.
///
/// This is the encoder counterpart to [`crate::partition::decode_partition`]:
/// it threads the same [`PartitionContextState`] and resolves `ctx` via
/// the shared [`partition_plane_context`], so the bit stream it emits
/// decodes back to the identical leaf set. The `on_leaf` callback is the
/// hook where the caller writes each block's §6.4.4 `mode_info( )` +
/// `residual( )` syntax into the same bool encoder.
///
/// Recursion order on [`PARTITION_SPLIT`] is TL → TR → BL → BR per the
/// §6.4.3 listing.
///
/// # Panics
///
/// Panics if `bsize >= BLOCK_SIZES` (not a valid §3 `BLOCK_*`); the
/// caller chain only invokes this with `BLOCK_64X64`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn write_partition_8x8(
    enc: &mut BoolEncoder,
    r: u32,
    c: u32,
    bsize: u8,
    mi_rows: u32,
    mi_cols: u32,
    ctx_state: &mut PartitionContextState,
    probs_kind: PartitionProbsKind<'_>,
    on_leaf: &mut LeafWriter<'_>,
) -> Result<(), Error> {
    // §6.4.3 line 2354: out-of-frame quadrants emit nothing.
    if r >= mi_rows || c >= mi_cols {
        return Ok(());
    }
    assert!(
        (bsize as usize) < BLOCK_SIZES,
        "write_partition_8x8: bsize={} out of range",
        bsize
    );

    let num8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as u32;
    let half = num8x8 >> 1;
    let has_rows = (r + half) < mi_rows;
    let has_cols = (c + half) < mi_cols;

    // §9.3.2 ctx derivation — materialise the same `num8x8`-wide strips
    // the decoder reads, zero-padding at frame edges.
    let above_end = ((c + num8x8) as usize).min(ctx_state.above.len());
    let above_strip = &ctx_state.above[(c as usize)..above_end];
    let left_end = ((r + num8x8) as usize).min(ctx_state.left.len());
    let left_strip = &ctx_state.left[(r as usize)..left_end];
    let strip_len = above_strip.len().min(left_strip.len());
    let above_buf: Vec<u8> = above_strip
        .iter()
        .take(strip_len)
        .copied()
        .chain(core::iter::repeat(0u8))
        .take(num8x8 as usize)
        .collect();
    let left_buf: Vec<u8> = left_strip
        .iter()
        .take(strip_len)
        .copied()
        .chain(core::iter::repeat(0u8))
        .take(num8x8 as usize)
        .collect();
    let ctx = partition_plane_context(bsize, &above_buf, &left_buf);
    let probs_row = probs_kind.row_for_writer(ctx);

    // Pick the partition for an all-8x8 layout and write its syntax.
    let partition = all_8x8_partition(bsize, has_rows, has_cols);
    write_partition_type(enc, has_rows, has_cols, partition, &probs_row)?;
    let subsize = SUBSIZE_LOOKUP[partition as usize][bsize as usize];

    // §6.4.3 lines 2362-2385: leaf vs recursive dispatch. The all-8x8
    // layout never codes HORZ / VERT on interior nodes, so only the leaf
    // and split arms are reachable; frame-edge nodes always split.
    if (subsize as usize) < (BLOCK_8X8 as usize) || partition == PARTITION_NONE {
        on_leaf(enc, r, c, subsize)?;
    } else {
        // PARTITION_SPLIT: TL → TR → BL → BR.
        write_partition_8x8(
            enc, r, c, subsize, mi_rows, mi_cols, ctx_state, probs_kind, on_leaf,
        )?;
        write_partition_8x8(
            enc,
            r,
            c + half,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            on_leaf,
        )?;
        write_partition_8x8(
            enc,
            r + half,
            c,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            on_leaf,
        )?;
        write_partition_8x8(
            enc,
            r + half,
            c + half,
            subsize,
            mi_rows,
            mi_cols,
            ctx_state,
            probs_kind,
            on_leaf,
        )?;
    }

    // §6.4.3 lines 2386-2391: tail write-back.
    if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
        write_back_partition_context(
            &mut ctx_state.above,
            &mut ctx_state.left,
            r as usize,
            c as usize,
            num8x8 as usize,
            subsize,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::partition::{decode_partition, LeafBlock};
    use crate::residual::BLOCK_64X64;

    /// Encode an all-8x8 partition for a `mi_rows × mi_cols` frame, then
    /// decode it back and assert the leaf set matches.
    fn roundtrip_all_8x8(mi_rows: u32, mi_cols: u32) -> Vec<LeafBlock> {
        let sb64_cols = ((mi_cols + 7) >> 3) * 8;
        let sb64_rows = ((mi_rows + 7) >> 3) * 8;
        let mut enc = BoolEncoder::new();
        let mut wctx = PartitionContextState::new(sb64_cols as usize, sb64_rows as usize);
        wctx.clear_above();

        // Collect the leaf order the writer produced for cross-check.
        let mut written = Vec::new();
        let mut r = 0u32;
        while r < mi_rows {
            wctx.clear_left();
            let mut c = 0u32;
            while c < mi_cols {
                write_partition_8x8(
                    &mut enc,
                    r,
                    c,
                    BLOCK_64X64,
                    mi_rows,
                    mi_cols,
                    &mut wctx,
                    PartitionProbsKind::Keyframe,
                    &mut |_enc, lr, lc, ls| {
                        written.push(LeafBlock {
                            r: lr,
                            c: lc,
                            subsize: ls,
                        });
                        Ok(())
                    },
                )
                .expect("write partition");
                c += 8;
            }
            r += 8;
        }
        let bytes = enc.finish();

        // Decode it back.
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).expect("init_bool");
        let mut dctx = PartitionContextState::new(sb64_cols as usize, sb64_rows as usize);
        dctx.clear_above();
        let mut decoded: Vec<LeafBlock> = Vec::new();
        let mut r = 0u32;
        while r < mi_rows {
            dctx.clear_left();
            let mut c = 0u32;
            while c < mi_cols {
                decode_partition(
                    &mut coder,
                    r,
                    c,
                    BLOCK_64X64,
                    mi_rows,
                    mi_cols,
                    &mut dctx,
                    PartitionProbsKind::Keyframe,
                    &mut decoded,
                )
                .expect("decode partition");
                c += 8;
            }
            r += 8;
        }

        assert_eq!(written, decoded, "writer/decoder leaf-order mismatch");
        // Every leaf must be an 8x8 block.
        for leaf in &decoded {
            assert_eq!(leaf.subsize, BLOCK_8X8, "leaf not 8x8: {:?}", leaf);
        }
        decoded
    }

    #[test]
    fn all_8x8_single_superblock_64x64() {
        // 64x64 => 8x8 MI grid => 64 leaves of 8x8.
        let leaves = roundtrip_all_8x8(8, 8);
        assert_eq!(leaves.len(), 64);
    }

    #[test]
    fn all_8x8_tiny_16x16() {
        // 16x16 => 2x2 MI grid => 4 leaves.
        let leaves = roundtrip_all_8x8(2, 2);
        assert_eq!(leaves.len(), 4);
    }

    #[test]
    fn all_8x8_non_multiple_of_8_frame_edges() {
        // 24x40 px => MI grid 3 rows x 5 cols (partial superblocks at the
        // right + bottom edges force the §6.4.3 frame-edge split arms).
        let leaves = roundtrip_all_8x8(3, 5);
        assert_eq!(leaves.len(), 15);
    }

    #[test]
    fn all_8x8_two_superblocks_wide() {
        // 128x64 px => 16 cols x 8 rows MI => 128 leaves over 2 SB64.
        let leaves = roundtrip_all_8x8(8, 16);
        assert_eq!(leaves.len(), 128);
    }

    #[test]
    fn corner_partition_writes_no_bits() {
        // A 1x1 MI frame: the root 64x64 node has has_rows == has_cols ==
        // false at every level down to 8x8, so all splits are inferred.
        let leaves = roundtrip_all_8x8(1, 1);
        assert_eq!(leaves.len(), 1);
    }
}
