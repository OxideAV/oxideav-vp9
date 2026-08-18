//! VP9 keyframe **frame assembler** — `encode_vp9`'s top-level driver
//! that threads the encoder-bootstrap writers into a complete decodable
//! keyframe.
//!
//! The frame layout the assembler produces mirrors what
//! [`crate::decode_frame::decode_intra_frame`] consumes:
//!
//! 1. the §6.2 uncompressed header ([`crate::header_writer`]) — byte
//!    aligned, with `header_size_in_bytes` set to the compressed-header
//!    length;
//! 2. the §6.3 compressed header ([`crate::compressed_writer`]) — the
//!    intra default-probability path;
//! 3. the §6.4 tile data — §9.2 bool-coded payloads produced by walking
//!    the §6.4.3 partition recursion ([`crate::partition_writer`]) and
//!    writing each leaf block's §6.4.4 mode info + §6.4.21 residual
//!    ([`crate::block_writer`]). The legacy all-8x8 assembler emits a
//!    single tile (`tile_cols_log2 == tile_rows_log2 == 0`, so no
//!    `tile_size` prefix); the tree assemblers accept any §6.2.13
//!    tiling and mirror the §6.4 `decode_tiles( )` row-major walk (one
//!    coder bracket per tile, f(32) `tile_size` prefixes on every tile
//!    but the last, per-tile-row left-context resets).
//!
//! The §6.4.3 partition layout the assembler emits is the all-`BLOCK_8X8`
//! layout ([`crate::partition_writer::write_partition_8x8`]); every leaf
//! is an 8x8 intra block whose mode / skip / residual a caller-supplied
//! [`KeyframePlan`] dictates. This is the minimal complete keyframe: a
//! decoder reconstructs every block from §8.5.1 intra prediction plus the
//! (caller-chosen) §8.6.2 residual.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.1 / §6.2 / §6.3 / §6.4; the byte
//! layout mirrors the in-crate `decode_intra_frame` exactly.

use crate::block_writer::{write_keyframe_intra_block, BlockWriteFrameCtx, IntraBlockSpec};
use crate::bool_encoder::BoolEncoder;
use crate::coef_probs::DEFAULT_COEF_PROBS;
use crate::compressed::{TxMode, DEFAULT_SKIP_PROB, DEFAULT_TX_PROBS};
use crate::compressed_writer::write_compressed_header_intra;
use crate::decode_block::Vp9FrameState;
use crate::header::{FrameType, Vp9FrameHeader};
use crate::header_writer::write_uncompressed_header;
use crate::mode_writer::{inferred_tx_size, tx_size_is_coded};
use crate::partition::{
    PartitionContextState, PartitionProbsKind, PARTITION_NONE, PARTITION_SPLIT,
};
use crate::partition_writer::{write_partition_8x8, write_partition_tree};
use crate::residual::{BLOCK_64X64, BLOCK_8X8, MAX_TXSIZE_LOOKUP};
use crate::tokens::NonzeroContext;
use crate::Error;
use std::collections::HashMap;

/// Per-8x8-block plan: the mode / skip / segment a leaf block codes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockPlan {
    /// Luma `y_mode` (0..=9).
    pub y_mode: u8,
    /// Chroma `uv_mode` (0..=9).
    pub uv_mode: u8,
    /// `skip` flag (true ⇒ no residual coded; decoder reconstructs from
    /// prediction only).
    pub skip: bool,
    /// `segment_id` (0..=7) — coded only when the frame's segmentation is
    /// enabled with `update_map`.
    pub segment_id: u8,
}

/// The minimal keyframe the assembler can emit: an all-8x8 partition with
/// a per-block [`BlockPlan`]. The block at MI `(r, c)` is
/// `plans[(r as usize) * mi_cols + c as usize]`.
pub(crate) struct KeyframePlan {
    /// One [`BlockPlan`] per 8x8 MI cell, row-major over `MiRows × MiCols`.
    pub plans: Vec<BlockPlan>,
    /// §6.3.1 `tx_mode` (the assembler uses `Only4x4` for the all-8x8
    /// path so no `tx_size` syntax is coded).
    pub tx_mode: TxMode,
}

impl KeyframePlan {
    /// An all-skip keyframe: every 8x8 block predicts intra with `y_mode`
    /// / `uv_mode` and codes no residual.
    pub fn all_skip(mi_rows: u32, mi_cols: u32, y_mode: u8, uv_mode: u8) -> Self {
        let n = (mi_rows as usize) * (mi_cols as usize);
        Self {
            plans: vec![
                BlockPlan {
                    y_mode,
                    uv_mode,
                    skip: true,
                    segment_id: 0,
                };
                n
            ],
            tx_mode: TxMode::Only4x4,
        }
    }
}

/// Coefficient source for a non-skip frame assembly: called per coded
/// transform block with `(mi_r, mi_c, plane, start_x, start_y, block_idx)`,
/// returning the `segEob`-length quantized `Tokens` array (raster order).
pub(crate) type FrameCoefSource<'f> = dyn FnMut(u32, u32, usize, u32, u32, usize) -> Vec<i64> + 'f;

/// Per-leaf spec of a [`KeyframeTreePlan`]: the mode info one §6.4.3
/// leaf block codes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeLeafPlan {
    /// `MiSize` this leaf is expected to be coded at — must equal the
    /// `subsize` the partition tree produces at the leaf's call site
    /// (validated at assembly; the coefficient source relies on it for
    /// the §6.4.21 availability derivation).
    pub mi_size: u8,
    /// `tx_size` (`TX_*` 0..=3). Under `TxModeSelect` any value up to
    /// `MAX_TXSIZE_LOOKUP[ MiSize ]` is codeable; under the other tx
    /// modes it must equal the §6.4.10 inferred size (validated).
    pub tx_size: u32,
    /// Luma `y_mode` (0..=9).
    pub y_mode: u8,
    /// Chroma `uv_mode` (0..=9).
    pub uv_mode: u8,
    /// `skip` flag.
    pub skip: bool,
    /// `segment_id` (0..=7).
    pub segment_id: u8,
}

/// A keyframe over an **arbitrary §6.4.3 partition tree**: per-node
/// partition choices plus per-leaf mode/tx specs — the generalisation of
/// [`KeyframePlan`] past the fixed all-`BLOCK_8X8` / `TX_4X4` layout.
pub(crate) struct KeyframeTreePlan {
    /// §6.3.1 `tx_mode`.
    pub tx_mode: TxMode,
    /// Partition value per recursion node, keyed `(MiRow, MiCol, bsize)`.
    /// Missing nodes fall back to the all-8x8 layout (SPLIT above
    /// `BLOCK_8X8`, NONE at it) so a partial map stays codeable.
    pub partitions: HashMap<(u32, u32, u8), u8>,
    /// Leaf spec keyed by the leaf's top-left MI `(row, col)`. Every
    /// leaf the partition tree produces must have an entry.
    pub leaves: HashMap<(u32, u32), TreeLeafPlan>,
}

impl KeyframeTreePlan {
    /// A uniform partition layout: every leaf is `leaf_size` (square,
    /// `>= BLOCK_8X8`) at `Min( tx_size, MAX_TXSIZE_LOOKUP[ leaf ] )`
    /// under `TX_MODE_SELECT`, except where the §6.4.3 frame-edge rules
    /// force a split — edge nodes recurse toward `BLOCK_8X8` (which is
    /// always codeable as `PARTITION_NONE`), so any frame geometry gets
    /// a conforming tree. All leaves start `DC_PRED`, non-skip.
    // Uniform-layout utility for the fixed-transform-size mirror tests
    // (the production planner builds content-adaptive trees instead).
    #[allow(dead_code)]
    pub fn uniform(mi_rows: u32, mi_cols: u32, leaf_size: u8, tx_size: u32) -> Self {
        let mut plan = Self {
            tx_mode: TxMode::TxModeSelect,
            partitions: HashMap::new(),
            leaves: HashMap::new(),
        };
        for r in (0..mi_rows).step_by(8) {
            for c in (0..mi_cols).step_by(8) {
                plan.fill_uniform(r, c, BLOCK_64X64, leaf_size, tx_size, mi_rows, mi_cols);
            }
        }
        plan
    }

    #[allow(dead_code)]
    // Spec-shaped geometry fan-in (r/c/bsize + frame extents), matching
    // the §6.4.3 recursion signature style used across the crate.
    #[allow(clippy::too_many_arguments)]
    fn fill_uniform(
        &mut self,
        r: u32,
        c: u32,
        bsize: u8,
        leaf_size: u8,
        tx_size: u32,
        mi_rows: u32,
        mi_cols: u32,
    ) {
        use crate::partition::{NUM_8X8_BLOCKS_WIDE_LOOKUP, SUBSIZE_LOOKUP};
        if r >= mi_rows || c >= mi_cols {
            return;
        }
        let num8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as u32;
        let half = num8x8 >> 1;
        // NONE only for **fully-contained** blocks (stricter than the
        // §6.4.3 hasRows / hasCols admission, which lets a block's right
        // / bottom half overhang the frame): the encoder's residual is
        // computed from MI-extent target planes, so an overhanging leaf
        // would read outside them — and its overhang bits would be
        // wasted anyway. Frame-edge regions split toward BLOCK_8X8
        // (num8x8 == 1, always contained on an in-frame node).
        let contained = (r + num8x8) <= mi_rows && (c + num8x8) <= mi_cols;
        if (bsize <= leaf_size || bsize == BLOCK_8X8) && contained {
            self.partitions.insert((r, c, bsize), PARTITION_NONE);
            self.leaves.insert(
                (r, c),
                TreeLeafPlan {
                    mi_size: bsize,
                    tx_size: tx_size.min(MAX_TXSIZE_LOOKUP[bsize as usize]),
                    y_mode: 0,
                    uv_mode: 0,
                    skip: false,
                    segment_id: 0,
                },
            );
            return;
        }
        self.partitions.insert((r, c, bsize), PARTITION_SPLIT);
        let subsize = SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][bsize as usize];
        for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
            self.fill_uniform(
                r + dr,
                c + dc,
                subsize,
                leaf_size,
                tx_size,
                mi_rows,
                mi_cols,
            );
        }
    }
}

/// Assemble a complete VP9 keyframe from `hdr` + a [`KeyframeTreePlan`],
/// returning the full frame bytes — [`assemble_keyframe`] generalised
/// over the partition tree and per-leaf transform sizes.
///
/// Each leaf writes the §6.4.4 intra block at the tree's `subsize` with
/// the plan's `tx_size`: coded via the §6.4.10 tree under
/// `TxModeSelect`, or validated against the inferred
/// `Min( maxTxSize, tx_mode_to_biggest_tx_size )` otherwise (a mismatch
/// would silently desync the reconstruction, so it is rejected).
/// §6.2 `FrameIsIntra` over a *writable* header: a key frame, or a
/// hidden intra-only frame (§6.2 codes the `intra_only` flag only when
/// `show_frame == 0`). Both frame classes code the identical §6.3
/// intra compressed header and §6.4 intra body (mode_info( ) dispatches
/// on FrameIsIntra, and the §9.3.2 partition / §6.4.6 mode probabilities
/// key on FrameIsIntra, not on frame_type), so the keyframe assemblers
/// accept either.
fn header_is_intra_frame(hdr: &Vp9FrameHeader) -> bool {
    match hdr.frame_type {
        FrameType::KeyFrame => !hdr.intra_only,
        FrameType::NonKeyFrame => hdr.intra_only && !hdr.show_frame,
    }
}

// Bytes-only convenience over the `_with_state` assembler; the
// non-test encoders all thread the state, so only tests call this.
#[allow(dead_code)]
pub(crate) fn assemble_keyframe_tree(
    hdr: &Vp9FrameHeader,
    plan: &KeyframeTreePlan,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    assemble_keyframe_tree_with_state(hdr, plan, coeffs).map(|(bytes, _)| bytes)
}

/// [`assemble_keyframe_tree`] also returning the writer's final §6.4.4
/// [`Vp9FrameState`] write-back arrays — the identical per-MI state the
/// decoder holds after its own §6.4 walk of these bytes (`MiSizes` /
/// `TxSizes` / `Skips` / `YModes` / `SegmentIds` / `RefFrames`), which
/// is exactly the per-MI input the §8.8 loop-filter processes consume
/// (§8.8.2 steps 4-12, §8.8.4 step 1). The encoder's reconstruction
/// path threads it into the encode-side §8.8 filter mirror.
pub(crate) fn assemble_keyframe_tree_with_state(
    hdr: &Vp9FrameHeader,
    plan: &KeyframeTreePlan,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<(Vec<u8>, Vp9FrameState), Error> {
    if !header_is_intra_frame(hdr) {
        return Err(Error::Unsupported);
    }

    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;

    // §6.3 compressed header (default-probability path).
    let chdr_bytes = write_compressed_header_intra(plan.tx_mode, hdr.quantization.lossless)?;

    // §6.2 uncompressed header with header_size_in_bytes = compressed len.
    // (`write_tile_info` validates `tile_cols_log2` against the §6.2.14
    // min/max derivation and `tile_rows_log2 <= 2`.)
    let mut hdr2 = *hdr;
    hdr2.header_size_in_bytes = u16::try_from(chdr_bytes.len()).map_err(|_| Error::Unsupported)?;
    let uhdr_bytes = write_uncompressed_header(&hdr2)?;

    // §6.4 tile data: one bool-coded payload per tile over the plan's
    // tree, walked in the decoder's row-major tile order.
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);
    let sb64_cols = ((mi_cols + 7) >> 3) * 8;
    let sb64_rows = ((mi_rows + 7) >> 3) * 8;

    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
    ];
    let mut pctx = PartitionContextState::new(sb64_cols as usize, sb64_rows as usize);
    let mut token_cache = vec![0u8; 1024];

    let fctx = BlockWriteFrameCtx {
        mi_cols,
        mi_rows,
        subsampling_x: ssx,
        subsampling_y: ssy,
        bit_depth,
        lossless: hdr.quantization.lossless,
        tx_mode: plan.tx_mode,
        seg_enabled: hdr.segmentation.enabled,
        seg_update_map: hdr.segmentation.update_map,
    };
    let tree_probs = hdr.segmentation.tree_probs;

    // §6.4 / §7.4.1: clear_above_context( ) once per frame — the above
    // strips carry ACROSS tiles (a lower tile row reads what the tile
    // above it wrote), only the left strips reset per tile row.
    pctx.clear_above();

    let tile_cols_log2 = u32::from(hdr.tile_info.tile_cols_log2);
    let tile_rows_log2 = u32::from(hdr.tile_info.tile_rows_log2);
    let mut tiles: Vec<Vec<u8>> = Vec::new();
    for tile_row in 0..(1u32 << tile_rows_log2) {
        for tile_col in 0..(1u32 << tile_cols_log2) {
            // §6.4 decode_tiles( ): the four get_tile_offset( ) extents.
            let mi_row_start = crate::partition::get_tile_offset(tile_row, mi_rows, tile_rows_log2);
            let mi_row_end =
                crate::partition::get_tile_offset(tile_row + 1, mi_rows, tile_rows_log2);
            let mi_col_start = crate::partition::get_tile_offset(tile_col, mi_cols, tile_cols_log2);
            let mi_col_end =
                crate::partition::get_tile_offset(tile_col + 1, mi_cols, tile_cols_log2);

            // §6.4 line 2326: one fresh §9.2 coder bracket per tile.
            let mut enc = BoolEncoder::new();

            let mut r = mi_row_start;
            while r < mi_row_end {
                // §7.4.2 clear_left_context( ) per superblock row —
                // including the tile's first row, so a right-hand tile
                // never reads partition context its left neighbour wrote.
                pctx.clear_left();
                for plane_nz in nz.iter_mut() {
                    plane_nz.left.fill(0);
                }
                let mut c = mi_col_start;
                while c < mi_col_end {
                    let mut layout = |lr: u32, lc: u32, bsize: u8| -> u8 {
                        plan.partitions.get(&(lr, lc, bsize)).copied().unwrap_or(
                            if bsize == BLOCK_8X8 {
                                PARTITION_NONE
                            } else {
                                PARTITION_SPLIT
                            },
                        )
                    };
                    let mut leaf = |enc: &mut BoolEncoder,
                                    lr: u32,
                                    lc: u32,
                                    subsize: u8|
                     -> Result<(), Error> {
                        let lp = plan
                            .leaves
                            .get(&(lr, lc))
                            .copied()
                            .ok_or(Error::Unsupported)?;
                        // The plan's leaf size must match the tree's subsize
                        // (the coefficient source predicts at lp.mi_size).
                        if lp.mi_size != subsize {
                            return Err(Error::Unsupported);
                        }
                        // tx-size codeability / inference validation.
                        let max_tx = MAX_TXSIZE_LOOKUP[subsize as usize];
                        if lp.tx_size > max_tx {
                            return Err(Error::Unsupported);
                        }
                        if !tx_size_is_coded(true, plan.tx_mode, subsize >= BLOCK_8X8)
                            && lp.tx_size != inferred_tx_size(max_tx, plan.tx_mode)
                        {
                            return Err(Error::Unsupported);
                        }
                        let spec = IntraBlockSpec {
                            r: lr,
                            mi_col_start,
                            c: lc,
                            mi_size: subsize,
                            segment_id: lp.segment_id,
                            skip: lp.skip,
                            tx_size: lp.tx_size,
                            y_mode: lp.y_mode,
                            uv_mode: lp.uv_mode,
                        };
                        let mut src = |plane: usize, tx_sz: u32, sx: u32, sy: u32, b: usize| {
                            let n0 = 1usize << (tx_sz + 2);
                            let mut v = coeffs(lr, lc, plane, sx, sy, b);
                            v.resize(n0 * n0, 0);
                            v
                        };
                        write_keyframe_intra_block(
                            enc,
                            &fctx,
                            &spec,
                            &mut state,
                            &mut nz,
                            &mut token_cache,
                            tree_probs.as_ref(),
                            &DEFAULT_SKIP_PROB,
                            &DEFAULT_TX_PROBS,
                            &DEFAULT_COEF_PROBS,
                            &mut src,
                        )
                    };
                    write_partition_tree(
                        &mut enc,
                        r,
                        c,
                        BLOCK_64X64,
                        mi_rows,
                        mi_cols,
                        &mut pctx,
                        PartitionProbsKind::Keyframe,
                        &mut layout,
                        &mut leaf,
                    )?;
                    c += 8;
                }
                r += 8;
            }

            tiles.push(enc.finish());
        }
    }

    let out = concat_frame_bytes(&uhdr_bytes, &chdr_bytes, &tiles)?;
    Ok((out, state))
}

/// Concatenate uncompressed header + compressed header + the §6.4 tile
/// payloads: every tile except the last is prefixed with its f(32)
/// big-endian `tile_size` (§6.4 line 2318 — the last tile's size is
/// implied by the remaining frame bytes).
fn concat_frame_bytes(
    uhdr_bytes: &[u8],
    chdr_bytes: &[u8],
    tiles: &[Vec<u8>],
) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(
        uhdr_bytes.len() + chdr_bytes.len() + tiles.iter().map(|t| t.len() + 4).sum::<usize>(),
    );
    out.extend_from_slice(uhdr_bytes);
    out.extend_from_slice(chdr_bytes);
    for (i, tile) in tiles.iter().enumerate() {
        if i + 1 != tiles.len() {
            let sz = u32::try_from(tile.len()).map_err(|_| Error::Unsupported)?;
            out.extend_from_slice(&sz.to_be_bytes());
        }
        out.extend_from_slice(tile);
    }
    Ok(out)
}

/// Assemble a complete VP9 keyframe from `hdr` + `plan`, returning the
/// full frame bytes (uncompressed header + compressed header + single
/// tile). `coeffs` supplies, per coded transform block, the quantized
/// `Tokens` array — only called for non-skip blocks.
///
/// `hdr` must be an intra frame — a key frame, or a hidden intra-only
/// frame (see [`header_is_intra_frame`]) — with
/// `tile_cols_log2 == tile_rows_log2 == 0`; the `header_size_in_bytes`
/// field is overwritten with the actual compressed-header length.
pub(crate) fn assemble_keyframe(
    hdr: &Vp9FrameHeader,
    plan: &KeyframePlan,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    if !header_is_intra_frame(hdr) {
        return Err(Error::Unsupported);
    }
    if hdr.tile_info.tile_cols_log2 != 0 || hdr.tile_info.tile_rows_log2 != 0 {
        return Err(Error::Unsupported);
    }

    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    if plan.plans.len() != (mi_rows as usize) * (mi_cols as usize) {
        return Err(Error::Unsupported);
    }

    // §6.3 compressed header (default-probability path).
    let chdr_bytes = write_compressed_header_intra(plan.tx_mode, hdr.quantization.lossless)?;

    // §6.2 uncompressed header with header_size_in_bytes = compressed len.
    let mut hdr2 = *hdr;
    hdr2.header_size_in_bytes = u16::try_from(chdr_bytes.len()).map_err(|_| Error::Unsupported)?;
    let uhdr_bytes = write_uncompressed_header(&hdr2)?;

    // §6.4 tile data: a single bool-coded payload.
    let tile_bytes = assemble_tile(hdr, mi_rows, mi_cols, plan, coeffs)?;

    let mut out = Vec::with_capacity(uhdr_bytes.len() + chdr_bytes.len() + tile_bytes.len());
    out.extend_from_slice(&uhdr_bytes);
    out.extend_from_slice(&chdr_bytes);
    out.extend_from_slice(&tile_bytes);
    Ok(out)
}

/// Walk the §6.4.3 all-8x8 partition recursion for the single tile,
/// writing each leaf block's §6.4.4 mode info + §6.4.21 residual into a
/// fresh §9.2 bool encoder. Returns the byte-finished tile payload.
fn assemble_tile(
    hdr: &Vp9FrameHeader,
    mi_rows: u32,
    mi_cols: u32,
    plan: &KeyframePlan,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);
    let sb64_cols = ((mi_cols + 7) >> 3) * 8;
    let sb64_rows = ((mi_rows + 7) >> 3) * 8;

    let mut enc = BoolEncoder::new();
    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
    ];
    let mut pctx = PartitionContextState::new(sb64_cols as usize, sb64_rows as usize);
    let mut token_cache = vec![0u8; 1024];

    let fctx = BlockWriteFrameCtx {
        mi_cols,
        mi_rows,
        subsampling_x: ssx,
        subsampling_y: ssy,
        bit_depth,
        lossless: hdr.quantization.lossless,
        tx_mode: plan.tx_mode,
        seg_enabled: hdr.segmentation.enabled,
        seg_update_map: hdr.segmentation.update_map,
    };
    let tree_probs = hdr.segmentation.tree_probs;

    // §6.4 line 2303: clear_above_context once per frame.
    pctx.clear_above();

    let mut r = 0u32;
    while r < mi_rows {
        pctx.clear_left();
        let mut c = 0u32;
        while c < mi_cols {
            // The leaf callback writes one 8x8 intra block; `enc` arrives
            // as the partition writer's first argument.
            let mut leaf =
                |enc: &mut BoolEncoder, lr: u32, lc: u32, _ls: u8| -> Result<(), Error> {
                    let idx = (lr as usize) * (mi_cols as usize) + (lc as usize);
                    let bp = plan.plans[idx];
                    let spec = IntraBlockSpec {
                        r: lr,
                        mi_col_start: 0,
                        c: lc,
                        mi_size: BLOCK_8X8,
                        segment_id: bp.segment_id,
                        skip: bp.skip,
                        tx_size: 0,
                        y_mode: bp.y_mode,
                        uv_mode: bp.uv_mode,
                    };
                    let mut src = |plane: usize, tx_sz: u32, sx: u32, sy: u32, b: usize| {
                        let n0 = 1usize << (tx_sz + 2);
                        let mut v = coeffs(lr, lc, plane, sx, sy, b);
                        v.resize(n0 * n0, 0);
                        v
                    };
                    write_keyframe_intra_block(
                        enc,
                        &fctx,
                        &spec,
                        &mut state,
                        &mut nz,
                        &mut token_cache,
                        tree_probs.as_ref(),
                        &DEFAULT_SKIP_PROB,
                        &DEFAULT_TX_PROBS,
                        &DEFAULT_COEF_PROBS,
                        &mut src,
                    )
                };
            write_partition_8x8(
                &mut enc,
                r,
                c,
                BLOCK_64X64,
                mi_rows,
                mi_cols,
                &mut pctx,
                PartitionProbsKind::Keyframe,
                &mut leaf,
            )?;
            c += 8;
        }
        r += 8;
    }

    Ok(enc.finish())
}

/// Build a minimal valid 8-bit 4:2:0 VP9 keyframe of size `width × height`
/// — an all-`BLOCK_8X8`, all-skip, `DC_PRED` keyframe with the loop filter
/// disabled. The reconstruction is a flat DC fill (no residual), so this
/// is a *structurally* complete decodable frame rather than a
/// pixel-accurate encode of the input; the forward-transform residual path
/// (encoding the input samples) is a later milestone.
///
/// `width` / `height` must be non-zero and at most `1 << 16`.
pub(crate) fn encode_keyframe_all_skip_dc(width: u32, height: u32) -> Result<Vec<u8>, Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    use crate::header::{
        ColorConfig, ColorSpace, LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo,
    };
    let hdr = Vp9FrameHeader {
        profile: 0,
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        frame_type: FrameType::KeyFrame,
        show_frame: true,
        error_resilient_mode: false,
        intra_only: false,
        color_config: ColorConfig {
            bit_depth: 8,
            color_space: ColorSpace::Bt601,
            color_range_full: false,
            subsampling_x: true,
            subsampling_y: true,
        },
        frame_width: width,
        frame_height: height,
        render_width: width,
        render_height: height,
        reset_frame_context: 0,
        refresh_frame_flags: 0xFF,
        refresh_frame_context: true,
        frame_parallel_decoding_mode: true,
        frame_context_idx: 0,
        loop_filter: LoopFilterParams {
            level: 0,
            sharpness: 0,
            delta_enabled: true,
            delta_update: false,
            ref_deltas: [None; 4],
            mode_deltas: [None; 2],
        },
        quantization: QuantizationParams {
            base_q_idx: 64,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        },
        segmentation: SegmentationParams::default_disabled(),
        tile_info: TileInfo {
            tile_cols_log2: 0,
            tile_rows_log2: 0,
        },
        header_size_in_bytes: 0,
        uncompressed_header_size_bytes: 0,
        ref_frame_idx: None,
        ref_frame_sign_bias: [false; 3],
        allow_high_precision_mv: false,
        interpolation_filter: 0,
    };
    let mi_cols = (width + 7) >> 3;
    let mi_rows = (height + 7) >> 3;
    let plan = KeyframePlan::all_skip(mi_rows, mi_cols, 0, 0);
    let mut coeffs: Box<FrameCoefSource> = Box::new(|_r, _c, _p, _x, _y, _b| Vec::new());
    assemble_keyframe(&hdr, &plan, &mut *coeffs)
}

/// Build the §6.2 header for a minimal P-frame of `width × height`: a
/// shown, single-tile, non-error-resilient inter frame that refreshes
/// slot 1, references slot 0 (`LAST`/`GOLDEN`/`ALTREF` all = slot 0),
/// EIGHTTAP filter, no high-precision MV. Pairs with
/// [`assemble_inter_frame_all_skip_zeromv`].
pub(crate) fn inter_pframe_header(width: u32, height: u32) -> Vp9FrameHeader {
    use crate::header::{
        ColorConfig, ColorSpace, LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo,
    };
    Vp9FrameHeader {
        profile: 0,
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        frame_type: FrameType::NonKeyFrame,
        show_frame: true,
        error_resilient_mode: false,
        intra_only: false,
        color_config: ColorConfig {
            bit_depth: 8,
            color_space: ColorSpace::Bt601,
            color_range_full: false,
            subsampling_x: true,
            subsampling_y: true,
        },
        frame_width: width,
        frame_height: height,
        render_width: width,
        render_height: height,
        reset_frame_context: 0,
        refresh_frame_flags: 0x02,
        refresh_frame_context: true,
        frame_parallel_decoding_mode: true,
        frame_context_idx: 0,
        loop_filter: LoopFilterParams {
            level: 0,
            sharpness: 0,
            delta_enabled: true,
            delta_update: false,
            ref_deltas: [None; 4],
            mode_deltas: [None; 2],
        },
        quantization: QuantizationParams {
            base_q_idx: 64,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        },
        segmentation: SegmentationParams::default_disabled(),
        tile_info: TileInfo {
            tile_cols_log2: 0,
            tile_rows_log2: 0,
        },
        header_size_in_bytes: 0,
        uncompressed_header_size_bytes: 0,
        ref_frame_idx: Some([0, 0, 0]),
        ref_frame_sign_bias: [false; 3],
        allow_high_precision_mv: false,
        interpolation_filter: 0,
    }
}

/// Assemble a complete VP9 **inter** frame: an all-`BLOCK_8X8`,
/// all-skip, single-reference-`LAST`, `ZEROMV` P-frame. Every leaf block
/// copies its co-located samples from the `LAST` reference (zero motion,
/// no residual), so the frame reconstructs to a verbatim copy of that
/// reference — the minimal complete decodable inter frame.
pub(crate) fn assemble_inter_frame_all_skip_zeromv(hdr: &Vp9FrameHeader) -> Result<Vec<u8>, Error> {
    let mut no_coeffs: Box<FrameCoefSource<'_>> = Box::new(|_r, _c, _p, _x, _y, _b| Vec::new());
    assemble_inter_frame_zeromv(hdr, true, &mut *no_coeffs)
}

/// Assemble a complete VP9 **inter** frame: an all-`BLOCK_8X8`,
/// single-reference-`LAST`, `ZEROMV` P-frame whose per-block residual a
/// caller-supplied [`FrameCoefSource`] dictates.
///
/// With `skip_all == true` no residual is coded (`coeffs` never fires)
/// and the frame reconstructs to a verbatim copy of its `LAST` reference;
/// with `skip_all == false` every leaf block codes the §6.4.21 residual
/// syntax, and the decoder reconstructs `prediction + residual` — the
/// carrier for a pixel-accurate inter encode.
///
pub(crate) fn assemble_inter_frame_zeromv(
    hdr: &Vp9FrameHeader,
    skip_all: bool,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    let mut planner: Box<InterBlockPlanner<'_>> =
        Box::new(|_r, _c, _state| (crate::mode_info::ZEROMV, [0, 0], false));
    assemble_inter_frame_planned(hdr, TxMode::Only4x4, skip_all, &mut *planner, coeffs)
}

/// Per-block inter mode planner: called once per `BLOCK_8X8` leaf (in
/// §6.4.3 partition order, i.e. exactly the decode order) with the MI
/// coordinates and the **shared** [`Vp9FrameState`] as it stands *before*
/// this block is written — the same state the inter block writer derives
/// its §6.5 MV predictors and §9.3.2 contexts from — and returns the
/// block's `(y_mode, mv, skip)` triple: `ZEROMV` with `[0, 0]`, or
/// `NEWMV` with an eighth-pel `[row, col]` vector (the difference onto
/// the §6.5.12 `BestMv` is coded, so the planner must pick a difference
/// the §6.4.20 decomposition can carry under the frame's high-precision
/// gate). A `skip == true` block codes no residual — the decoder
/// reconstructs it from §8.5.2 prediction alone, so the planner elects
/// it only when the block's quantized residual is all-zero (identical
/// reconstruction, minus the per-block end-of-block bits).
pub(crate) type InterBlockPlanner<'f> =
    dyn FnMut(u32, u32, &Vp9FrameState) -> (u8, [i32; 2], bool) + 'f;

/// Assemble a complete VP9 **inter** frame: an all-`BLOCK_8X8`,
/// single-reference-`LAST` P-frame whose per-block inter mode / motion
/// vector a caller-supplied [`InterBlockPlanner`] dictates and whose
/// per-block residual comes from `coeffs` (never fired when
/// `skip_all == true`).
///
/// `hdr` must be a non-key frame (`FrameType::NonKeyFrame`, `!intra_only`,
/// shown, `tile_cols_log2 == tile_rows_log2 == 0`) carrying a valid
/// `ref_frame_idx`; the `header_size_in_bytes` field is overwritten with
/// the actual compressed-header length.
///
/// The writer models §6.5 `UsePrevFrameMvs == 0` (it holds no
/// previous-frame motion field), which matches the decoder's §7.2.6
/// derivation only when `error_resilient_mode == 1` — so any plan that
/// returns a non-`ZEROMV` block (where the §6.5 candidate scan reaches
/// the coded syntax through `BestMv`) requires an error-resilient header,
/// and a `NEWMV` plan on a non-error-resilient header is rejected.
///
/// `tx_mode` selects the frame's §6.3.1 transform mode; every block
/// codes the §6.4.10 **inferred** size
/// `Min( maxTxSize, tx_mode_to_biggest_tx_size )` (`TX_8X8` at the
/// all-`BLOCK_8X8` layout under `Allow8x8` and larger — no per-block tx
/// bits), so `TxModeSelect` is rejected (this legacy planner type
/// carries no per-block tx choice; use [`assemble_inter_frame_tree`]
/// for per-block transform-size selection). A lossless header forces
/// `Only4x4` (the §6.3.1 lossless gate never codes `tx_mode`, and the
/// WHT path is 4x4-only).
pub(crate) fn assemble_inter_frame_planned(
    hdr: &Vp9FrameHeader,
    tx_mode: TxMode,
    skip_all: bool,
    planner: &mut InterBlockPlanner<'_>,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    assemble_inter_frame_planned_with_state(hdr, tx_mode, skip_all, None, planner, coeffs)
        .map(|(bytes, _)| bytes)
}

/// [`assemble_inter_frame_planned`] with the §7.2.6 prev-motion-field
/// model and the §6.4.4 write-back state returned — the chained
/// sequence encoders thread each frame's returned state into the next
/// frame's `prev_frame_mvs` (see [`InterFrameTreePlan::prev_frame_mvs`]
/// for the model's contract). `prev_frame_mvs = None` is byte-identical
/// to [`assemble_inter_frame_planned`].
pub(crate) fn assemble_inter_frame_planned_with_state(
    hdr: &Vp9FrameHeader,
    tx_mode: TxMode,
    skip_all: bool,
    prev_frame_mvs: Option<PrevMotionField>,
    planner: &mut InterBlockPlanner<'_>,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<(Vec<u8>, Vp9FrameState), Error> {
    use crate::compressed::ReferenceMode;
    use crate::mode_info::{LAST_FRAME, NONE_REF_FRAME};

    if matches!(tx_mode, TxMode::TxModeSelect) {
        return Err(Error::Unsupported); // no per-block tx in this planner type.
    }
    let switchable = hdr.interpolation_filter == 4;
    let plan = InterFrameTreePlan {
        tx_mode,
        reference_mode: ReferenceMode::SingleReference,
        partitions: HashMap::new(), // default fallback = all-8x8 layout.
        prev_segment_ids: None,
        prev_frame_mvs_absent: false,
        prev_frame_mvs,
    };
    let mut tree_planner: Box<InterTreePlanner<'_>> =
        Box::new(|lr: u32, lc: u32, subsize: u8, state: &Vp9FrameState| {
            let (y_mode, mv, block_skip) = planner(lr, lc, state);
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: inferred_tx_size(MAX_TXSIZE_LOOKUP[subsize as usize], tx_mode),
                y_mode,
                interp_filter: if switchable {
                    0
                } else {
                    hdr.interpolation_filter
                },
                ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                mv: [mv, [0, 0]],
                skip: skip_all || block_skip,
                segment_id: 0,
                sub: None,
            }
        });
    let mut src: Box<FrameCoefSource<'_>> =
        Box::new(|lr: u32, lc: u32, p: usize, sx: u32, sy: u32, b: usize| {
            if skip_all {
                Vec::new()
            } else {
                coeffs(lr, lc, p, sx, sy, b)
            }
        });
    assemble_inter_frame_tree_with_state(hdr, &plan, &mut *tree_planner, &mut *src)
}

/// Per-leaf inter mode-info an [`InterTreePlanner`] elects: the §6.4.11 /
/// §6.4.16 syntax values the decoder will recover for one partition-tree
/// leaf.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterTreeLeaf {
    /// `MiSize` — must equal the `subsize` the partition tree produced
    /// at this leaf's call site (validated at assembly).
    pub mi_size: u8,
    /// `tx_size` (`TX_*` 0..=3). Under `TxModeSelect` any value up to
    /// `MAX_TXSIZE_LOOKUP[ MiSize ]` is codeable on a non-skip block
    /// (the §6.4.10 tree is coded); under the other tx modes — and on
    /// every **skip** block, where §6.4.10 `read_tx_size( allowSelect =
    /// !skip )` never codes bits — it must equal the inferred
    /// `Min( maxTxSize, tx_mode_to_biggest_tx_size )` (validated).
    pub tx_size: u32,
    /// Inter `y_mode` (`NEARESTMV` / `NEARMV` / `ZEROMV` / `NEWMV`).
    pub y_mode: u8,
    /// `interp_filter` — coded only when the frame filter is SWITCHABLE.
    pub interp_filter: u8,
    /// The §3 reference pair; `ref_frame[ 1 ] == NONE_REF_FRAME` ⇒
    /// single prediction, otherwise compound.
    pub ref_frame: [i32; 2],
    /// Final per-list eighth-pel motion vectors.
    pub mv: [[i32; 2]; 2],
    /// `skip` flag (true ⇒ no residual coded).
    pub skip: bool,
    /// `segment_id` (0..=7) — coded by the §6.4.7 tree when the header
    /// carries `segmentation_enabled && segmentation_update_map`; must
    /// be 0 otherwise. Blocks on a `SEG_LVL_SKIP` segment must plan
    /// `skip = true` + `ZEROMV`; blocks on a `SEG_LVL_REF_FRAME`
    /// segment must plan the override's single reference.
    pub segment_id: u8,
    /// The §6.4.16 sub-8x8 per-cell walk spec — required (`Some`) when
    /// the partition tree produces this leaf at `MiSize < BLOCK_8X8`
    /// (`y_mode` / `mv` above are then ignored), and ignored otherwise.
    /// Sub-8x8 leaves always plan `tx_size = 0` (§6.4.10 codes no bits;
    /// the inferred size is `TX_4X4`).
    pub sub: Option<crate::inter_block_writer::InterSubBlockSpec>,
}

/// Per-leaf inter planner for [`assemble_inter_frame_tree`]: called once
/// per partition-tree leaf `(mi_row, mi_col, subsize)` in §6.4.3 decode
/// order with the **shared** [`Vp9FrameState`] as it stands before the
/// block is written — the same state the inter block writer derives its
/// §6.5 MV predictors and §9.3.2 contexts from.
pub(crate) type InterTreePlanner<'f> =
    dyn FnMut(u32, u32, u8, &Vp9FrameState) -> InterTreeLeaf + 'f;

/// Frame-level plan for [`assemble_inter_frame_tree`]: the §6.3.1
/// `tx_mode`, the §6.3.12 `reference_mode`, and the §6.4.3 partition
/// layout (missing nodes fall back to the all-8x8 layout — SPLIT above
/// `BLOCK_8X8`, NONE at it — so a partial map stays codeable).
pub(crate) struct InterFrameTreePlan {
    pub tx_mode: TxMode,
    pub reference_mode: crate::compressed::ReferenceMode,
    pub partitions: HashMap<(u32, u32, u8), u8>,
    /// §6.4.14 `PrevSegmentIds[ ][ ]` — the last map-bearing frame's
    /// segment-id plane (row-major `MiRows × MiCols`). Required when the
    /// header codes `segmentation_temporal_update = 1` (the §6.4.12
    /// `seg_id_predicted` branch predicts against it); ignored — and
    /// normally `None` — otherwise.
    pub prev_segment_ids: Option<Vec<u8>>,
    /// Caller's assertion that the §7.2.6 `UsePrevFrameMvs` derivation
    /// yields 0 for this frame *without* `error_resilient_mode` — true
    /// when the previously-decoded frame is hidden (`show_frame == 0`),
    /// intra, or differently sized. Non-`ZEROMV` leaves on a
    /// non-error-resilient header are accepted under this flag or when
    /// [`Self::prev_frame_mvs`] models the field instead.
    pub prev_frame_mvs_absent: bool,
    /// §6.5.10 previous-frame motion field — the previous decoded
    /// frame's §6.4.4 `RefFrames` / `Mvs` write-back arrays (both
    /// row-major `MiRows × MiCols × 2`, exactly the
    /// [`Vp9FrameState::ref_frames`] / [`Vp9FrameState::mvs`] layout the
    /// `_with_state` assemblers return). Supplying it models the §7.2.6
    /// `UsePrevFrameMvs == 1` decode: the caller asserts the
    /// previously-decoded frame is SHOWN, same-sized, and this header is
    /// not error-resilient — the shape on which the decoder scans the
    /// prev field — and the writer feeds it to the §6.5.10 candidate
    /// scan through the same shared
    /// [`crate::inter_decode::FrameStateMvSource`] the decoder uses, so
    /// the `NearestMv` / `NearMv` / `BestMv` predictors are
    /// bit-identical. This is what lets non-error-resilient P-frame
    /// *chains* (shown predecessor — and therefore compound prediction
    /// without a hidden/intra predecessor) carry non-`ZEROMV` blocks.
    /// Mutually exclusive with `prev_frame_mvs_absent` and with an
    /// error-resilient header (both mean `UsePrevFrameMvs == 0`).
    pub prev_frame_mvs: Option<PrevMotionField>,
}

/// The previous decoded frame's motion field for
/// [`InterFrameTreePlan::prev_frame_mvs`]: the §6.4.4 `RefFrames` /
/// `Mvs` arrays snapshotted after that frame's walk (`[(row * MiCols +
/// col) * 2 + refList]` layout on both).
#[derive(Clone, Debug)]
pub(crate) struct PrevMotionField {
    /// `PrevRefFrames[ row ][ col ][ list ]`.
    pub ref_frames: Vec<i32>,
    /// `PrevMvs[ row ][ col ][ list ]`.
    pub mvs: Vec<(i16, i16)>,
}

impl PrevMotionField {
    /// Snapshot a frame's §6.4.4 write-back arrays as the next frame's
    /// prev motion field.
    pub fn from_state(state: &Vp9FrameState) -> Self {
        Self {
            ref_frames: state.ref_frames.clone(),
            mvs: state.mvs.clone(),
        }
    }

    /// The motion field after an all-intra frame (a keyframe): every MI
    /// cell holds `ref_frame = [ INTRA_FRAME, NONE ]` with zero vectors
    /// (§6.4.4 — the intra arm writes the reference pair and leaves
    /// `Mvs` at the zero initialisation). No §6.5.10 prev candidate can
    /// match against it (both passes require a `> INTRA_FRAME`
    /// reference), but the decoder still *scans* it when §7.2.6 derives
    /// `UsePrevFrameMvs = 1` over a shown keyframe predecessor, so the
    /// writer must model the same field.
    pub fn after_intra_frame(mi_rows: u32, mi_cols: u32) -> Self {
        let n = (mi_rows as usize) * (mi_cols as usize);
        let mut ref_frames = Vec::with_capacity(n * 2);
        for _ in 0..n {
            ref_frames.push(crate::mode_info::INTRA_FRAME);
            ref_frames.push(crate::mode_info::NONE_REF_FRAME);
        }
        Self {
            ref_frames,
            mvs: vec![(0, 0); n * 2],
        }
    }
}

/// Assemble a complete VP9 **inter** frame over an **arbitrary §6.4.3
/// partition tree** — [`assemble_inter_frame_planned`] generalised over
/// the partition layout (sub-8x8 leaves included: `BLOCK_4X4` /
/// `BLOCK_4X8` / `BLOCK_8X4` with per-cell inter modes and MVs through
/// [`InterTreeLeaf::sub`]), per-block transform sizes (`TxModeSelect`
/// included), reference selection (any single reference, or compound
/// when the plan's `reference_mode` + the header's sign biases allow
/// it), and per-block switchable interpolation filters.
///
/// `hdr` must be a non-key frame (`FrameType::NonKeyFrame`,
/// `!intra_only`) carrying a valid `ref_frame_idx`; the
/// `header_size_in_bytes` field is overwritten with the actual
/// compressed-header length. Any §6.2.13-codeable tiling is accepted:
/// the writer mirrors the §6.4 `decode_tiles( )` row-major walk — one
/// §9.2 coder bracket per tile, f(32) `tile_size` prefixes on every
/// tile but the last, above-context strips carrying across tiles with
/// the left strips resetting per tile row, and the §6.5 candidate scans
/// clamped to each tile's `MiColStart` / `MiColEnd` window.
///
/// The writer models the §7.2.6 `UsePrevFrameMvs` derivation both ways:
/// with [`InterFrameTreePlan::prev_frame_mvs`] supplied it scans the
/// previous frame's motion field exactly as the decoder does
/// (`UsePrevFrameMvs == 1` — shown same-sized predecessor on a
/// non-error-resilient header), and without it it models
/// `UsePrevFrameMvs == 0`, which matches the decoder only on an
/// error-resilient header or under the caller's
/// `prev_frame_mvs_absent` assertion (hidden / intra / resized
/// predecessor) — so a plan that returns a non-`ZEROMV` block (where
/// the §6.5 candidate scan reaches the coded syntax through the MV
/// predictors) on a non-error-resilient header without either is
/// rejected.
///
/// Per-leaf `tx_size` is coded through the §6.4.10 tree when
/// `read_tx_size( allowSelect = !skip )` codes it (`TxModeSelect`,
/// non-skip, `MiSize >= BLOCK_8X8`), and validated against the inferred
/// `Min( maxTxSize, tx_mode_to_biggest_tx_size )` otherwise (a mismatch
/// would silently desync the reconstruction). A lossless header forces
/// `Only4x4`.
///
/// The compound-reference configuration is the §6.3.18
/// `setup_compound_reference_mode( )` derivation over the header's sign
/// biases — the identical shared function the decoder runs — so the
/// §6.4.17 `read_ref_frames` contexts agree on both sides.
pub(crate) fn assemble_inter_frame_tree(
    hdr: &Vp9FrameHeader,
    plan: &InterFrameTreePlan,
    planner: &mut InterTreePlanner<'_>,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    assemble_inter_frame_tree_with_state(hdr, plan, planner, coeffs).map(|(bytes, _)| bytes)
}

/// [`assemble_inter_frame_tree`] also returning the writer's final
/// §6.4.4 [`Vp9FrameState`] write-back arrays — the identical per-MI
/// state the decoder holds after its own §6.4 walk of these bytes
/// (`MiSizes` / `TxSizes` / `Skips` / `YModes` / `SegmentIds` /
/// `RefFrames`), which is exactly the per-MI input the §8.8 loop-filter
/// processes consume (§8.8.2 steps 4-12, §8.8.4 step 1). The encoder's
/// reconstruction path threads it into the encode-side §8.8 filter
/// mirror.
pub(crate) fn assemble_inter_frame_tree_with_state(
    hdr: &Vp9FrameHeader,
    plan: &InterFrameTreePlan,
    planner: &mut InterTreePlanner<'_>,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<(Vec<u8>, Vp9FrameState), Error> {
    use crate::compressed::{setup_compound_reference_mode, RefFrameSignBias, ReferenceMode};
    use crate::compressed_writer::write_compressed_header_inter;
    use crate::inter_block_writer::{
        write_inter_block, InterBlockFrameCtx, InterBlockProbs, InterBlockSpec,
    };
    use crate::mode_info::ZEROMV;

    let tx_mode = plan.tx_mode;
    let reference_mode = plan.reference_mode;

    // Hidden (`show_frame == 0`) inter frames are accepted — the §6.2
    // writer codes the explicit `intra_only = 0` flag for them — and
    // serve as reference-building frames (e.g. the predecessor of a
    // compound frame, keeping §7.2.6 UsePrevFrameMvs at 0).
    if hdr.frame_type != FrameType::NonKeyFrame || hdr.intra_only {
        return Err(Error::Unsupported);
    }
    if hdr.ref_frame_idx.is_none() {
        return Err(Error::Unsupported);
    }
    if hdr.quantization.lossless && !matches!(tx_mode, TxMode::Only4x4) {
        return Err(Error::Unsupported);
    }

    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let switchable = hdr.interpolation_filter == 4;
    // sign_bias indexed by §3 ref value: [INTRA, LAST, GOLDEN, ALTREF].
    // §7.2 setup_past_independence( ): an error-resilient frame's
    // *effective* sign biases are all zero — the §6.2 bits are consumed
    // before the reset runs, so the coded values are dead. Everything
    // the writer derives from the biases (compoundReferenceAllowed,
    // the §6.3.18 compound layout, §6.5 candidate sign flips) must use
    // the effective values, exactly as the decoder does; in particular
    // a non-single `reference_mode` is uncodeable on an
    // error-resilient header (write_compressed_header_inter rejects it
    // through the §6.3.12 compoundReferenceAllowed derivation).
    let sign_bias = if hdr.error_resilient_mode {
        [false; 4]
    } else {
        [
            false,
            hdr.ref_frame_sign_bias[0],
            hdr.ref_frame_sign_bias[1],
            hdr.ref_frame_sign_bias[2],
        ]
    };

    // §6.3 compressed header (default-probability path). The write
    // validates `reference_mode` against the sign-bias-derived
    // `compoundReferenceAllowed` exactly as the §6.3.12 parser does.
    let chdr_bytes = write_compressed_header_inter(
        tx_mode,
        hdr.quantization.lossless,
        reference_mode,
        switchable,
        hdr.allow_high_precision_mv,
        &sign_bias,
    )?;

    // §6.2 uncompressed header with header_size_in_bytes = compressed len.
    let mut hdr2 = *hdr;
    hdr2.header_size_in_bytes = u16::try_from(chdr_bytes.len()).map_err(|_| Error::Unsupported)?;
    let uhdr_bytes = write_uncompressed_header(&hdr2)?;

    // §6.4 tile data.
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);
    let sb64_cols = ((mi_cols + 7) >> 3) * 8;
    let sb64_rows = ((mi_rows + 7) >> 3) * 8;

    let ctx = crate::compressed::FrameContext::default();
    // §6.3.18: the same shared derivation the decoder's compressed-header
    // parse runs on the non-SingleReference arms.
    let comp_config = setup_compound_reference_mode(&RefFrameSignBias::from_inter_biases(
        u8::from(sign_bias[1]),
        u8::from(sign_bias[2]),
        u8::from(sign_bias[3]),
    ));

    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
    ];
    let mut pctx = PartitionContextState::new(sb64_cols as usize, sb64_rows as usize);
    let mut token_cache = vec![0u8; 1024];

    // §6.4.12 temporal-update state: the §7.4 seg-pred context strips
    // (per-tile above reset here; per-superblock-row left reset below,
    // mirroring the decoder's decode_tile) and the §6.4.14
    // PrevSegmentIds plane from the plan.
    let temporal_seg =
        hdr.segmentation.enabled && hdr.segmentation.update_map && hdr.segmentation.temporal_update;
    // §7.2 setup_past_independence( ) clears PrevSegmentIds on an
    // error-resilient frame — the §6.4.14 predictor would be the
    // all-zero map, so a temporal update there is pointless and the
    // writer's caller-supplied map could silently disagree with it.
    if temporal_seg && hdr.error_resilient_mode {
        return Err(Error::Unsupported);
    }
    if temporal_seg
        && plan
            .prev_segment_ids
            .as_ref()
            .map(|m| m.len() != (mi_rows as usize) * (mi_cols as usize))
            .unwrap_or(true)
    {
        return Err(Error::Unsupported);
    }
    let mut seg_pred_ctx = crate::mode_info::SegPredContextState::new(mi_cols, mi_rows);

    // §7.2.6 UsePrevFrameMvs writer model: a supplied prev motion field
    // asserts the derivation yields 1 — which requires a
    // non-error-resilient header (§7.2.6) and contradicts the
    // prev-field-absent assertion — and must span the frame's MI grid
    // (the §6.5.10 scan indexes it at every block position).
    let use_prev_frame_mvs = plan.prev_frame_mvs.is_some();
    if let Some(pmf) = plan.prev_frame_mvs.as_ref() {
        if hdr.error_resilient_mode || plan.prev_frame_mvs_absent {
            return Err(Error::Unsupported);
        }
        let n = (mi_rows as usize) * (mi_cols as usize) * 2;
        if pmf.ref_frames.len() != n || pmf.mvs.len() != n {
            return Err(Error::Unsupported);
        }
    }

    let probs = InterBlockProbs {
        skip_prob: &ctx.skip_prob,
        tx_probs: &ctx.tx_probs,
        is_inter_prob: &ctx.is_inter_prob,
        comp_mode_prob: &ctx.comp_mode_prob,
        single_ref_prob: &ctx.single_ref_prob,
        comp_ref_prob: &ctx.comp_ref_prob,
        inter_mode_probs: &ctx.inter_mode_probs,
        interp_filter_probs: &ctx.interp_filter_probs,
        mv_probs: &ctx.mv_probs,
        coef_probs: &ctx.coef_probs,
        tree_probs: hdr.segmentation.tree_probs.as_ref(),
        pred_prob: hdr.segmentation.pred_prob.as_ref(),
    };

    // §6.4 / §7.4.1: clear_above_context( ) once per frame — the above
    // strips (partition, nonzero, seg-pred) carry ACROSS tiles; only the
    // left strips reset per tile row.
    pctx.clear_above();

    let tile_cols_log2 = u32::from(hdr.tile_info.tile_cols_log2);
    let tile_rows_log2 = u32::from(hdr.tile_info.tile_rows_log2);
    let mut tiles: Vec<Vec<u8>> = Vec::new();
    for tile_row in 0..(1u32 << tile_rows_log2) {
        for tile_col in 0..(1u32 << tile_cols_log2) {
            // §6.4 decode_tiles( ): the four get_tile_offset( ) extents.
            let mi_row_start = crate::partition::get_tile_offset(tile_row, mi_rows, tile_rows_log2);
            let mi_row_end =
                crate::partition::get_tile_offset(tile_row + 1, mi_rows, tile_rows_log2);
            let mi_col_start = crate::partition::get_tile_offset(tile_col, mi_cols, tile_cols_log2);
            let mi_col_end =
                crate::partition::get_tile_offset(tile_col + 1, mi_cols, tile_cols_log2);

            // Per-tile frame ctx: the §6.5 candidate scans clamp their
            // column window to THIS tile's extents (`MiColStart` /
            // `MiColEnd`), exactly as the decoder's per-tile
            // `BlockDecoder` does.
            let fctx = InterBlockFrameCtx {
                mi_cols,
                mi_rows,
                mi_col_end,
                subsampling_x: ssx,
                subsampling_y: ssy,
                bit_depth,
                lossless: hdr.quantization.lossless,
                tx_mode,
                seg: hdr.segmentation,
                reference_mode,
                comp_config,
                sign_bias: &sign_bias,
                interpolation_filter: hdr.interpolation_filter,
                allow_high_precision_mv: hdr.allow_high_precision_mv,
                use_prev_frame_mvs,
                prev_frame_mvs: plan.prev_frame_mvs.as_ref().map(|p| {
                    crate::inter_decode::PrevFrameMvs {
                        prev_ref_frames: &p.ref_frames,
                        prev_mvs: &p.mvs,
                    }
                }),
                prev_segment_ids: plan.prev_segment_ids.as_deref(),
            };

            // §6.4 line 2326: one fresh §9.2 coder bracket per tile.
            let mut enc = BoolEncoder::new();

            let mut r = mi_row_start;
            while r < mi_row_end {
                // §7.4.2: the left strips reset per superblock row —
                // including the tile's first row, so a right-hand tile
                // never reads context its left neighbour wrote.
                pctx.clear_left();
                seg_pred_ctx.clear_left();
                for plane_nz in nz.iter_mut() {
                    plane_nz.left.fill(0);
                }
                let mut c = mi_col_start;
                while c < mi_col_end {
                    let mut layout = |lr: u32, lc: u32, bsize: u8| -> u8 {
                        plan.partitions.get(&(lr, lc, bsize)).copied().unwrap_or(
                            if bsize == BLOCK_8X8 {
                                PARTITION_NONE
                            } else {
                                PARTITION_SPLIT
                            },
                        )
                    };
                    let mut leaf = |enc: &mut BoolEncoder,
                                    lr: u32,
                                    lc: u32,
                                    subsize: u8|
                     -> Result<(), Error> {
                        let lp = planner(lr, lc, subsize, &state);
                        if lp.mi_size != subsize {
                            return Err(Error::Unsupported);
                        }
                        // §7.2.6: a non-ZEROMV leaf (block-level, or any
                        // visited sub-8x8 cell) reaches the coded syntax
                        // through the §6.5 predictors, so the writer's
                        // UsePrevFrameMvs model must match the decoder's
                        // derivation: an error-resilient header, the
                        // caller-asserted absent prev field, or the
                        // caller-supplied prev motion field — see the
                        // function docs.
                        let any_non_zeromv = if subsize < BLOCK_8X8 {
                            match lp.sub {
                                Some(sub) => sub.modes.iter().any(|&m| m != ZEROMV),
                                None => return Err(Error::Unsupported),
                            }
                        } else {
                            lp.y_mode != ZEROMV
                        };
                        if any_non_zeromv
                            && !hdr.error_resilient_mode
                            && !plan.prev_frame_mvs_absent
                            && !use_prev_frame_mvs
                        {
                            return Err(Error::Unsupported);
                        }
                        if lp.ref_frame[1] > crate::mode_info::INTRA_FRAME
                            && reference_mode == ReferenceMode::SingleReference
                        {
                            return Err(Error::Unsupported);
                        }
                        // tx-size codeability / inference validation, per the
                        // §6.4.10 `read_tx_size( allowSelect = !skip )` gate.
                        let max_tx = MAX_TXSIZE_LOOKUP[subsize as usize];
                        if lp.tx_size > max_tx {
                            return Err(Error::Unsupported);
                        }
                        if !tx_size_is_coded(!lp.skip, tx_mode, subsize >= BLOCK_8X8)
                            && lp.tx_size != inferred_tx_size(max_tx, tx_mode)
                        {
                            return Err(Error::Unsupported);
                        }
                        let spec = InterBlockSpec {
                            r: lr,
                            mi_col_start,
                            c: lc,
                            mi_size: subsize,
                            segment_id: lp.segment_id,
                            skip: lp.skip,
                            tx_size: lp.tx_size,
                            ref_frame: lp.ref_frame,
                            y_mode: lp.y_mode,
                            interp_filter: lp.interp_filter,
                            mv: lp.mv,
                            sub: lp.sub,
                        };
                        let mut src = |p: usize, tx_sz: u32, sx: u32, sy: u32, b: usize| {
                            let n0 = 1usize << (tx_sz + 2);
                            if lp.skip {
                                vec![0i64; n0 * n0]
                            } else {
                                let mut v = coeffs(lr, lc, p, sx, sy, b);
                                v.resize(n0 * n0, 0);
                                v
                            }
                        };
                        write_inter_block(
                            enc,
                            &fctx,
                            &spec,
                            &probs,
                            &mut state,
                            Some(&mut seg_pred_ctx),
                            &mut nz,
                            &mut token_cache,
                            &mut src,
                        )
                    };
                    write_partition_tree(
                        &mut enc,
                        r,
                        c,
                        BLOCK_64X64,
                        mi_rows,
                        mi_cols,
                        &mut pctx,
                        PartitionProbsKind::Inter(&ctx.partition_probs),
                        &mut layout,
                        &mut leaf,
                    )?;
                    c += 8;
                }
                r += 8;
            }

            tiles.push(enc.finish());
        }
    }

    let out = concat_frame_bytes(&uhdr_bytes, &chdr_bytes, &tiles)?;
    Ok((out, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_frame::decode_intra_frame;
    use crate::header::{
        ColorConfig, ColorSpace, LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo,
    };

    fn keyframe_header(width: u32, height: u32) -> Vp9FrameHeader {
        Vp9FrameHeader {
            profile: 0,
            show_existing_frame: false,
            frame_to_show_map_idx: None,
            frame_type: FrameType::KeyFrame,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            color_config: ColorConfig {
                bit_depth: 8,
                color_space: ColorSpace::Bt601,
                color_range_full: false,
                subsampling_x: true,
                subsampling_y: true,
            },
            frame_width: width,
            frame_height: height,
            render_width: width,
            render_height: height,
            reset_frame_context: 0,
            refresh_frame_flags: 0xFF,
            refresh_frame_context: true,
            frame_parallel_decoding_mode: true,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                level: 0, // no deblocking — isolates reconstruction.
                sharpness: 0,
                delta_enabled: true,
                delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_q_idx: 64,
                delta_q_y_dc: 0,
                delta_q_uv_dc: 0,
                delta_q_uv_ac: 0,
                lossless: false,
            },
            segmentation: SegmentationParams::default_disabled(),
            tile_info: TileInfo {
                tile_cols_log2: 0,
                tile_rows_log2: 0,
            },
            header_size_in_bytes: 0,
            uncompressed_header_size_bytes: 0,
            ref_frame_idx: None,
            ref_frame_sign_bias: [false; 3],
            allow_high_precision_mv: false,
            interpolation_filter: 0,
        }
    }

    fn no_coeffs() -> Box<FrameCoefSource<'static>> {
        Box::new(|_r, _c, _p, _x, _y, _b| Vec::new())
    }

    /// An all-skip keyframe assembles to bytes that decode without error,
    /// and the decode is deterministic / re-decodable.
    #[test]
    fn all_skip_keyframe_decodes() {
        let hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 0, 0);
        let bytes = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        // Re-decoding the same bytes yields identical output.
        let frame2 = decode_intra_frame(&bytes).expect("decode2");
        assert_eq!(frame.y, frame2.y);
    }

    /// A non-multiple-of-64 frame (forces frame-edge partition splits)
    /// still assembles + decodes.
    #[test]
    fn all_skip_partial_superblock_decodes() {
        // 24x40 px => MI grid 3 rows x 5 cols.
        let hdr = keyframe_header(40, 24);
        let plan = KeyframePlan::all_skip(3, 5, 1, 2);
        let bytes = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.width, 40);
        assert_eq!(frame.height, 24);
    }

    /// A two-superblock-wide frame exercises the per-superblock above /
    /// left context threading across SB boundaries.
    #[test]
    fn all_skip_two_superblocks_decode() {
        let hdr = keyframe_header(128, 64);
        let plan = KeyframePlan::all_skip(8, 16, 9, 4);
        let bytes = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.width, 128);
        assert_eq!(frame.height, 64);
    }

    /// An all-skip DC_PRED keyframe reconstructs to a flat mid-grey fill:
    /// every plane is the §8.5.1 `DC_PRED` no-neighbour default
    /// `1 << (BitDepth - 1) == 128` at 8-bit (no residual, loop filter
    /// off).
    #[test]
    fn all_skip_dc_pred_is_flat_128() {
        let hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 0, 0); // DC_PRED luma + uv.
        let bytes = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert!(frame.y.iter().all(|&s| s == 128), "luma not flat 128");
        assert!(frame.u.iter().all(|&s| s == 128), "U not flat 128");
        assert!(frame.v.iter().all(|&s| s == 128), "V not flat 128");
    }

    /// A non-skip keyframe with a chosen DC coefficient: the **top-left**
    /// 4x4 luma block predicts from no neighbours (§8.5.1 `DC_PRED`
    /// no-neighbour default `1 << (BitDepth-1) == 128`), so its
    /// reconstructed samples equal `128 + r` where `r` is the
    /// independently-computed inverse transform of the dequantized DC
    /// token. This pins the residual writer's coefficients reconstructing
    /// to known samples through the *full* decode pipeline (dequant +
    /// inverse transform + reconstruct), not just the token round-trip.
    #[test]
    fn non_skip_dc_residual_reconstructs_top_left_block() {
        use crate::dequant::get_dc_quant;
        use crate::idct::{inverse_transform_2d, DCT_DCT};

        let hdr = keyframe_header(64, 64);
        let n = 8 * 8usize;
        let plan = KeyframePlan {
            plans: vec![
                BlockPlan {
                    y_mode: 0,
                    uv_mode: 0,
                    skip: false,
                    segment_id: 0,
                };
                n
            ],
            tx_mode: TxMode::Only4x4,
        };

        // Code DC token = 2 on every 4x4 luma + chroma block.
        let dc_token: i64 = 2;
        let mut coeffs: Box<FrameCoefSource> = Box::new(move |_r, _c, _p, _x, _y, _b| {
            let mut v = vec![0i64; 16];
            v[0] = dc_token;
            v
        });
        let bytes = assemble_keyframe(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");

        // Independent reconstruction of the top-left 4x4 luma block.
        let seg = SegmentationParams::default_disabled();
        let q = hdr.quantization;
        let dcq = get_dc_quant(0, &seg, &q, 0, 8);
        let mut probe = vec![0i64; 16];
        probe[0] = dc_token * dcq as i64; // dqDenom = 1 for TX_4X4.
        inverse_transform_2d(&mut probe, 2, DCT_DCT, false);
        let r = probe[0];
        let exp = (128i64 + r).clamp(0, 255) as i32;
        assert_ne!(exp, 128, "DC residual should shift the block off 128");

        // The top-left 4x4 luma samples (row-major, width 64).
        for row in 0..4usize {
            for col in 0..4usize {
                let s = frame.y[row * 64 + col] as i32;
                assert_eq!(s, exp, "top-left luma ({row},{col}) != {exp}");
            }
        }
    }

    /// A segmentation-enabled keyframe: every block codes a `segment_id`
    /// via the §6.4.7 `intra_segment_id( )` tree, and the frame decodes
    /// without error. Pins the assembler's segment-id path
    /// (block_writer's §6.4.7 write_segment_id) against the decoder.
    #[test]
    fn segmented_keyframe_decodes() {
        let mut hdr = keyframe_header(64, 64);
        let mut seg = SegmentationParams::default_disabled();
        seg.enabled = true;
        seg.update_map = true;
        seg.tree_probs = Some([128; 7]);
        seg.temporal_update = false;
        seg.pred_prob = Some([255; 3]);
        seg.update_data = true;
        seg.abs_or_delta_update = false; // delta update.
                                         // SEG_LVL_ALT_Q (feature 0) on segment 1: a -16 quantizer delta.
        seg.feature_enabled[1][0] = true;
        seg.feature_data[1][0] = -16;
        hdr.segmentation = seg;

        // Checkerboard segment ids 0 / 1 across the 8x8 grid, all skip.
        let mut plan = KeyframePlan::all_skip(8, 8, 0, 0);
        for (i, bp) in plan.plans.iter_mut().enumerate() {
            bp.segment_id = (i % 2) as u8;
        }
        let bytes = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!((frame.width, frame.height), (64, 64));
        // All-skip DC_PRED with no residual is still flat 128 regardless
        // of segment (the per-segment Q only affects coded residuals).
        assert!(frame.y.iter().all(|&s| s == 128), "luma not flat 128");
    }

    /// A 10-bit profile-2 keyframe assembles + decodes; the no-neighbour
    /// DC_PRED default is `1 << (BitDepth - 1) == 512` at 10-bit, and the
    /// packed output carries little-endian u16 pairs.
    #[test]
    fn profile2_10bit_keyframe_decodes() {
        let mut hdr = keyframe_header(64, 64);
        hdr.profile = 2;
        hdr.color_config = ColorConfig {
            bit_depth: 10,
            color_space: ColorSpace::Bt709,
            color_range_full: false,
            subsampling_x: true,
            subsampling_y: true,
        };
        let plan = KeyframePlan::all_skip(8, 8, 0, 0);
        let bytes = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.bit_depth, 10);
        assert!(
            frame.y.iter().all(|&s| s == 512),
            "10-bit luma not flat 512"
        );
    }

    /// Re-encoding the same plan is byte-stable (the assembler is a pure
    /// function of its inputs).
    #[test]
    fn assembly_is_deterministic() {
        let hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 3, 5);
        let a = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("a");
        let b = assemble_keyframe(&hdr, &plan, &mut *no_coeffs()).expect("b");
        assert_eq!(a, b, "assembly not byte-stable");
    }

    /// A non-skip keyframe carrying **AC** coefficients (not just DC) on
    /// the top-left 4x4 luma block reconstructs to the independently
    /// computed `Clip1( 128 + invtx( dequant( coeffs ) ) )` — exercising
    /// the full §6.4.25 scan + token-tree + §8.6.1/§8.7/§8.6.2 path for a
    /// non-flat residual through the assembler. The top-left block alone
    /// is checked because it is the only block predicting from a known
    /// (no-neighbour, flat 128) baseline.
    #[test]
    fn non_skip_ac_residual_reconstructs_top_left_block() {
        use crate::dequant::{get_ac_quant, get_dc_quant};
        use crate::idct::{inverse_transform_2d, DCT_DCT};

        let hdr = keyframe_header(64, 64);
        // DC_PRED everywhere so mode2txfm_map -> DCT_DCT (matches probe).
        let n = 8 * 8usize;
        let plan = KeyframePlan {
            plans: vec![
                BlockPlan {
                    y_mode: 0,
                    uv_mode: 0,
                    skip: false,
                    segment_id: 0,
                };
                n
            ],
            tx_mode: TxMode::Only4x4,
        };

        // Chosen 4x4 luma coefficients (raster order): a DC + two AC
        // terms. Only the top-left luma block (mi 0,0; plane 0; start 0,0)
        // gets this; every other coded block is all-zero (still non-skip,
        // so it codes a single more_coefs == 0 at DC).
        let tl_coeffs: [i64; 16] = {
            let mut v = [0i64; 16];
            v[0] = 3; // DC
            v[1] = -2; // AC (row 0, col 1)
            v[4] = 1; // AC (row 1, col 0)
            v
        };
        let mut coeffs: Box<FrameCoefSource> = Box::new(move |r, c, plane, sx, sy, _b| {
            let mut out = vec![0i64; 16];
            if r == 0 && c == 0 && plane == 0 && sx == 0 && sy == 0 {
                out.copy_from_slice(&tl_coeffs);
            }
            out
        });
        let bytes = assemble_keyframe(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");

        // Independent reconstruction of the top-left 4x4 luma block.
        let seg = SegmentationParams::default_disabled();
        let q = hdr.quantization;
        let dcq = get_dc_quant(0, &seg, &q, 0, 8) as i64;
        let acq = get_ac_quant(0, &seg, &q, 0, 8) as i64;
        let mut deq = vec![0i64; 16];
        for (i, &t) in tl_coeffs.iter().enumerate() {
            deq[i] = t * acq; // dqDenom = 1 for TX_4X4.
        }
        deq[0] = tl_coeffs[0] * dcq; // DC override.
        inverse_transform_2d(&mut deq, 2, DCT_DCT, false);
        let mut expected = [[0i32; 4]; 4];
        for (i, exp_row) in expected.iter_mut().enumerate() {
            for (j, slot) in exp_row.iter_mut().enumerate() {
                *slot = (128i64 + deq[i * 4 + j]).clamp(0, 255) as i32;
            }
        }
        // Residual must be non-flat (the AC terms create variation).
        let flat = expected.iter().flatten().all(|&s| s == expected[0][0]);
        assert!(!flat, "AC residual should produce a non-flat block");

        for (i, exp_row) in expected.iter().enumerate() {
            for (j, &exp) in exp_row.iter().enumerate() {
                let s = frame.y[i * 64 + j] as i32;
                assert_eq!(s, exp, "top-left luma ({i},{j})");
            }
        }
    }

    // ----- Tree-plan keyframe assembler -----

    use crate::partition::{PARTITION_HORZ, PARTITION_VERT};
    use crate::residual::{BLOCK_16X16, BLOCK_32X32};

    fn uniform_tree_plan(
        mi_rows: u32,
        mi_cols: u32,
        leaf_size: u8,
        tx_size: u32,
        skip: bool,
    ) -> KeyframeTreePlan {
        let mut plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, leaf_size, tx_size);
        for lp in plan.leaves.values_mut() {
            lp.tx_size = tx_size; // deliberately unclamped: rejection tests.
            lp.skip = skip;
        }
        plan
    }

    /// An all-skip uniform-32x32 tree keyframe under TX_MODE_SELECT
    /// decodes to the flat DC fill — the partition + tx_size syntax at
    /// >8x8 leaves threads through the whole §6.4 walk.
    #[test]
    fn tree_uniform_32x32_all_skip_decodes_flat() {
        let hdr = keyframe_header(64, 64);
        let plan = uniform_tree_plan(8, 8, BLOCK_32X32, 3, true);
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert!(frame.y.iter().all(|&s| s == 128), "luma not flat 128");
    }

    /// A single 64x64 NONE leaf at TX_32X32 with a DC token: the
    /// top-left 32x32 transform block predicts from no neighbours (flat
    /// 128) and reconstructs to `Clip1( 128 + r )` where `r` is the
    /// independently-probed §8.6.2 dequant (dqDenom == 2!) + §8.7
    /// inverse. Pins the TX_32X32 residual path through the assembler.
    #[test]
    fn tree_64x64_tx32_dc_residual_reconstructs() {
        use crate::dequant::get_dc_quant;
        use crate::idct::{inverse_transform_2d, DCT_DCT};

        let hdr = keyframe_header(64, 64);
        let mut plan = uniform_tree_plan(8, 8, BLOCK_64X64, 3, false);
        for lp in plan.leaves.values_mut() {
            lp.y_mode = 0; // DC_PRED
            lp.uv_mode = 0;
        }
        let dc_token: i64 = 5;
        let mut coeffs: Box<FrameCoefSource> = Box::new(move |_r, _c, plane, sx, sy, _b| {
            let mut v = Vec::new();
            if plane == 0 && sx == 0 && sy == 0 {
                v = vec![0i64; 32 * 32];
                v[0] = dc_token;
            }
            v
        });
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");

        // Independent probe: dqDenom = 2 at TX_32X32.
        let seg = SegmentationParams::default_disabled();
        let dcq = get_dc_quant(0, &seg, &hdr.quantization, 0, 8) as i64;
        let mut probe = vec![0i64; 32 * 32];
        probe[0] = (dc_token * dcq) / 2;
        inverse_transform_2d(&mut probe, 5, DCT_DCT, false);
        for i in 0..32usize {
            for j in 0..32usize {
                let exp = (128i64 + probe[i * 32 + j]).clamp(0, 255) as i32;
                assert_eq!(frame.y[i * 64 + j] as i32, exp, "({i},{j})");
            }
        }
    }

    /// Uniform 16x16 leaves at TX_16X16: the top-left block's DC
    /// residual reconstructs against the independent probe (dqDenom 1).
    #[test]
    fn tree_16x16_tx16_dc_residual_reconstructs() {
        use crate::dequant::get_dc_quant;
        use crate::idct::{inverse_transform_2d, DCT_DCT};

        let hdr = keyframe_header(64, 64);
        let plan = uniform_tree_plan(8, 8, BLOCK_16X16, 2, false);
        let dc_token: i64 = 4;
        let mut coeffs: Box<FrameCoefSource> = Box::new(move |r, c, plane, sx, sy, _b| {
            let mut v = Vec::new();
            if r == 0 && c == 0 && plane == 0 && sx == 0 && sy == 0 {
                v = vec![0i64; 16 * 16];
                v[0] = dc_token;
            }
            v
        });
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");

        let seg = SegmentationParams::default_disabled();
        let dcq = get_dc_quant(0, &seg, &hdr.quantization, 0, 8) as i64;
        let mut probe = vec![0i64; 16 * 16];
        probe[0] = dc_token * dcq;
        inverse_transform_2d(&mut probe, 4, DCT_DCT, false);
        for i in 0..16usize {
            for j in 0..16usize {
                let exp = (128i64 + probe[i * 16 + j]).clamp(0, 255) as i32;
                assert_eq!(frame.y[i * 64 + j] as i32, exp, "({i},{j})");
            }
        }
    }

    /// Non-square leaves: a HORZ superblock (two 64x32 leaves at
    /// TX_32X32) and a VERT one (two 32x64) both decode; the residual
    /// walk covers the rectangular num4x4w != num4x4h grids.
    #[test]
    fn tree_horz_vert_nonsquare_leaves_decode() {
        let hdr = keyframe_header(128, 64);
        let mut partitions = HashMap::new();
        let mut leaves = HashMap::new();
        partitions.insert((0, 0, BLOCK_64X64), PARTITION_HORZ);
        partitions.insert((0, 8, BLOCK_64X64), PARTITION_VERT);
        for (key, sz) in [
            ((0u32, 0u32), crate::residual::BLOCK_64X32),
            ((4, 0), crate::residual::BLOCK_64X32),
            ((0, 8), crate::residual::BLOCK_32X64),
            ((0, 12), crate::residual::BLOCK_32X64),
        ] {
            leaves.insert(
                key,
                TreeLeafPlan {
                    mi_size: sz,
                    tx_size: 3,
                    y_mode: 0,
                    uv_mode: 0,
                    skip: false,
                    segment_id: 0,
                },
            );
        }
        let plan = KeyframeTreePlan {
            tx_mode: TxMode::TxModeSelect,
            partitions,
            leaves,
        };
        let dc = 3i64;
        let mut coeffs: Box<FrameCoefSource> = Box::new(move |_r, _c, _p, _sx, _sy, _b| {
            let mut v = vec![0i64; 1];
            v[0] = dc;
            v
        });
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!((frame.width, frame.height), (128, 64));
        // The DC residual shifts every sample off the flat-128 baseline.
        assert!(frame.y.iter().all(|&s| s != 128), "residual did not code");
    }

    /// Under a non-select tx_mode the leaf's tx_size must equal the
    /// §6.4.10 inferred value: the correct value assembles + decodes,
    /// a mismatch is rejected.
    #[test]
    fn tree_inferred_tx_size_validated() {
        let hdr = keyframe_header(64, 64);
        // Allow8x8: inferred tx at a 32x32 leaf = min(TX_32X32-cap, 8x8)
        // = TX_8X8 (1).
        let mut plan = uniform_tree_plan(8, 8, BLOCK_32X32, 1, true);
        plan.tx_mode = TxMode::Allow8x8;
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *no_coeffs()).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert!(frame.y.iter().all(|&s| s == 128));

        // Wrong tx (TX_4X4) under Allow8x8 -> rejected.
        let mut bad = uniform_tree_plan(8, 8, BLOCK_32X32, 0, true);
        bad.tx_mode = TxMode::Allow8x8;
        assert_eq!(
            assemble_keyframe_tree(&hdr, &bad, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );

        // tx_size above the leaf's MAX_TXSIZE -> rejected (TX_32X32 on a
        // 16x16 leaf).
        let over = uniform_tree_plan(8, 8, BLOCK_16X16, 3, true);
        assert_eq!(
            assemble_keyframe_tree(&hdr, &over, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );
    }

    /// A leaf without a plan entry is rejected (not silently defaulted).
    #[test]
    fn tree_missing_leaf_entry_rejected() {
        let hdr = keyframe_header(64, 64);
        let mut plan = uniform_tree_plan(8, 8, BLOCK_32X32, 3, true);
        plan.leaves.remove(&(0, 4));
        assert_eq!(
            assemble_keyframe_tree(&hdr, &plan, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );
    }

    /// A mixed-depth tree (one 32x32 quadrant split to 16x16 with mixed
    /// per-leaf tx sizes, skip and non-skip) assembles and decodes.
    #[test]
    fn tree_mixed_depth_mixed_tx_decodes() {
        let hdr = keyframe_header(64, 64);
        let mut plan = uniform_tree_plan(8, 8, BLOCK_32X32, 3, false);
        // Split the TL 32x32 into 16x16 leaves with varying tx sizes.
        plan.partitions
            .insert((0, 0, BLOCK_32X32), crate::partition::PARTITION_SPLIT);
        plan.leaves.remove(&(0, 0));
        for (i, key) in [(0u32, 0u32), (0, 2), (2, 0), (2, 2)].iter().enumerate() {
            plan.partitions.insert(
                (key.0, key.1, BLOCK_16X16),
                crate::partition::PARTITION_NONE,
            );
            plan.leaves.insert(
                *key,
                TreeLeafPlan {
                    mi_size: BLOCK_16X16,
                    tx_size: (i as u32) % 3, // TX_4X4 / TX_8X8 / TX_16X16
                    y_mode: 0,
                    uv_mode: 0,
                    skip: i == 1,
                    segment_id: 0,
                },
            );
        }
        let mut coeffs: Box<FrameCoefSource> = Box::new(|_r, _c, _p, _sx, _sy, _b| {
            let mut v = vec![0i64; 1];
            v[0] = 2;
            v
        });
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!((frame.width, frame.height), (64, 64));
    }

    /// A legal §6.4.3 layout whose 32x32 NONE leaves **overhang** the
    /// frame edge (the hasRows / hasCols admission only requires the
    /// top / left half in-frame): a 56x56 frame is 7x7 MIs, so the
    /// 32x32 leaves at MI (0,4) / (4,0) / (4,4) extend past the
    /// MiCols*8 working extent. The §8.5.1 / §8.6.2 stores clip at the
    /// allocated planes (equivalent to a spec CurrFrame's superblock
    /// padding), so the stream decodes without panic — a regression
    /// test for the decoder-side out-of-bounds this layout used to hit.
    #[test]
    fn tree_overhanging_leaves_decode_without_panic() {
        let hdr = keyframe_header(56, 56);
        let mut partitions = HashMap::new();
        let mut leaves = HashMap::new();
        partitions.insert((0u32, 0u32, BLOCK_64X64), PARTITION_SPLIT);
        for key in [(0u32, 0u32), (0, 4), (4, 0), (4, 4)] {
            partitions.insert(
                (key.0, key.1, BLOCK_32X32),
                crate::partition::PARTITION_NONE,
            );
            leaves.insert(
                key,
                TreeLeafPlan {
                    mi_size: BLOCK_32X32,
                    tx_size: 3,
                    y_mode: 0,
                    uv_mode: 0,
                    skip: false,
                    segment_id: 0,
                },
            );
        }
        let plan = KeyframeTreePlan {
            tx_mode: TxMode::TxModeSelect,
            partitions,
            leaves,
        };
        let mut coeffs: Box<FrameCoefSource> = Box::new(|_r, _c, _p, _sx, _sy, _b| vec![7i64]);
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("assemble");
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!((frame.width, frame.height), (56, 56));
        // Deterministic re-decode.
        let frame2 = decode_intra_frame(&bytes).expect("decode2");
        assert_eq!(frame.y, frame2.y);
    }

    /// The tree assembler is byte-deterministic.
    #[test]
    fn tree_assembly_is_deterministic() {
        let hdr = keyframe_header(64, 64);
        let plan = uniform_tree_plan(8, 8, BLOCK_16X16, 2, true);
        let a = assemble_keyframe_tree(&hdr, &plan, &mut *no_coeffs()).expect("a");
        let b = assemble_keyframe_tree(&hdr, &plan, &mut *no_coeffs()).expect("b");
        assert_eq!(a, b);
    }

    // ----- Inter-frame assembler -----

    fn inter_header(width: u32, height: u32) -> Vp9FrameHeader {
        inter_pframe_header(width, height)
    }

    /// An all-skip ZEROMV P-frame after a flat keyframe reconstructs to a
    /// verbatim copy of the keyframe (zero motion + no residual = a copy
    /// of the LAST reference), validated end-to-end through
    /// `decode_vp9_sequence`.
    #[test]
    fn inter_all_skip_zeromv_copies_reference() {
        use crate::decode_frame::decode_vp9_sequence;

        let kf_hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 0, 0); // DC_PRED -> flat 128.
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        let p_hdr = inter_header(64, 64);
        let pf = assemble_inter_frame_all_skip_zeromv(&p_hdr).expect("p-frame");

        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode sequence");
        assert_eq!(frames.len(), 2);
        // Both frames are 64x64.
        assert_eq!((frames[0].width, frames[0].height), (64, 64));
        assert_eq!((frames[1].width, frames[1].height), (64, 64));
        // The keyframe is flat 128; the P-frame copies it.
        assert!(frames[0].y.iter().all(|&s| s == 128), "keyframe not flat");
        assert_eq!(frames[1].y, frames[0].y, "p-frame luma != reference");
        assert_eq!(frames[1].u, frames[0].u, "p-frame U != reference");
        assert_eq!(frames[1].v, frames[0].v, "p-frame V != reference");
    }

    /// A two-superblock-wide P-frame (128x64) exercises the per-superblock
    /// partition + neighbour-context threading across SB boundaries.
    #[test]
    fn inter_two_superblocks_copy_reference() {
        use crate::decode_frame::decode_vp9_sequence;

        let kf_hdr = keyframe_header(128, 64);
        let plan = KeyframePlan::all_skip(8, 16, 0, 0);
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        let p_hdr = inter_header(128, 64);
        let pf = assemble_inter_frame_all_skip_zeromv(&p_hdr).expect("p-frame");

        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode sequence");
        assert_eq!((frames[1].width, frames[1].height), (128, 64));
        assert_eq!(frames[1].y, frames[0].y, "p-frame luma != reference");
    }

    /// A non-multiple-of-64 P-frame (40x24) forces frame-edge partition
    /// splits and still copies the reference.
    #[test]
    fn inter_partial_superblock_copies_reference() {
        use crate::decode_frame::decode_vp9_sequence;

        let kf_hdr = keyframe_header(40, 24);
        let plan = KeyframePlan::all_skip(3, 5, 0, 0);
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        let p_hdr = inter_header(40, 24);
        let pf = assemble_inter_frame_all_skip_zeromv(&p_hdr).expect("p-frame");

        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode sequence");
        assert_eq!((frames[1].width, frames[1].height), (40, 24));
        assert_eq!(frames[1].y, frames[0].y, "p-frame luma != reference");
    }

    /// The inter assembler is byte-deterministic.
    #[test]
    fn inter_assembly_is_deterministic() {
        let h = inter_header(64, 64);
        let a = assemble_inter_frame_all_skip_zeromv(&h).expect("a");
        let b = assemble_inter_frame_all_skip_zeromv(&h).expect("b");
        assert_eq!(a, b, "inter assembly not byte-stable");
    }

    /// A keyframe header is rejected by the inter assembler.
    #[test]
    fn inter_assembler_rejects_keyframe() {
        let h = keyframe_header(64, 64);
        assert_eq!(
            assemble_inter_frame_all_skip_zeromv(&h).unwrap_err(),
            Error::Unsupported
        );
    }

    /// An `Allow8x8` P-frame codes every block at the §6.4.10 inferred
    /// `TX_8X8` (no per-block tx bits) and decodes end-to-end; the
    /// `TxModeSelect` and lossless-with-larger-tx gates are pinned.
    #[test]
    fn inter_tx_mode_gates_and_allow8x8_decode() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::mode_info::ZEROMV;

        let kf_hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 0, 0);
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        // A DC-only residual on every luma TX_8X8 block (the src closure
        // receives the 8x8 segEob via resize).
        let p_hdr = inter_header(64, 64);
        let mut planner: Box<InterBlockPlanner> = Box::new(|_r, _c, _s| (ZEROMV, [0, 0], false));
        let mut coeffs: Box<FrameCoefSource> =
            Box::new(
                |_r, _c, p, _sx, _sy, _b| {
                    if p == 0 {
                        vec![2i64]
                    } else {
                        Vec::new()
                    }
                },
            );
        let pf = assemble_inter_frame_planned(
            &p_hdr,
            TxMode::Allow8x8,
            false,
            &mut *planner,
            &mut *coeffs,
        )
        .expect("allow8x8 p-frame");
        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
        // The DC residual shifts the luma off the flat-128 reference.
        assert!(frames[1].y.iter().all(|&s| s != 128));

        // Gates.
        let mut p2: Box<InterBlockPlanner> = Box::new(|_r, _c, _s| (ZEROMV, [0, 0], false));
        let mut c2: Box<FrameCoefSource> = Box::new(|_r, _c, _p, _sx, _sy, _b| Vec::new());
        assert_eq!(
            assemble_inter_frame_planned(&p_hdr, TxMode::TxModeSelect, true, &mut *p2, &mut *c2)
                .unwrap_err(),
            Error::Unsupported
        );
        let mut lossless_hdr = p_hdr;
        lossless_hdr.quantization.base_q_idx = 0;
        lossless_hdr.quantization.lossless = true;
        assert_eq!(
            assemble_inter_frame_planned(&lossless_hdr, TxMode::Allow8x8, true, &mut *p2, &mut *c2)
                .unwrap_err(),
            Error::Unsupported
        );
    }

    // ----- §6.4.3 inter tree assembler -----

    use crate::compressed::ReferenceMode;
    use crate::mode_info::NONE_REF_FRAME;
    use crate::residual::{BLOCK_16X16 as B16, BLOCK_32X32 as B32};

    /// An all-skip ZEROMV leaf at `subsize` with the §6.4.10 inferred tx.
    fn skip_leaf(subsize: u8, tx_mode: TxMode) -> InterTreeLeaf {
        InterTreeLeaf {
            mi_size: subsize,
            tx_size: inferred_tx_size(MAX_TXSIZE_LOOKUP[subsize as usize], tx_mode),
            y_mode: crate::mode_info::ZEROMV,
            interp_filter: 0,
            ref_frame: [crate::mode_info::LAST_FRAME, NONE_REF_FRAME],
            mv: [[0, 0], [0, 0]],
            skip: true,
            segment_id: 0,
            sub: None,
        }
    }

    /// A mixed-leaf-size all-skip ZEROMV P-frame (64x64 NONE + HORZ +
    /// VERT + deep SPLIT superblocks) reconstructs to a verbatim copy of
    /// its reference across every partition shape, end-to-end through
    /// `decode_vp9_sequence`.
    #[test]
    fn inter_tree_mixed_leaf_sizes_copy_reference() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::partition::{PARTITION_HORZ, PARTITION_VERT};
        use crate::residual::{BLOCK_32X64, BLOCK_64X32};

        let kf_hdr = keyframe_header(128, 128);
        let plan = KeyframePlan::all_skip(16, 16, 0, 0);
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        let p_hdr = inter_header(128, 128);
        let mut partitions = HashMap::new();
        // SB (0,0): one 64x64 leaf. SB (0,8): HORZ (two 64x32).
        // SB (8,0): VERT (two 32x64). SB (8,8): default all-8x8 SPLIT.
        partitions.insert((0u32, 0u32, BLOCK_64X64), PARTITION_NONE);
        partitions.insert((0, 8, BLOCK_64X64), PARTITION_HORZ);
        partitions.insert((8, 0, BLOCK_64X64), PARTITION_VERT);
        let tree_plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions,
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let mut seen: Vec<(u32, u32, u8)> = Vec::new();
        let mut planner: Box<InterTreePlanner> = Box::new(|r, c, subsize, _s| {
            seen.push((r, c, subsize));
            skip_leaf(subsize, TxMode::Only4x4)
        });
        let pf = assemble_inter_frame_tree(&p_hdr, &tree_plan, &mut *planner, &mut *no_coeffs())
            .expect("tree p-frame");
        drop(planner);

        // The planned tree produced the expected leaf shapes.
        assert!(seen.contains(&(0, 0, BLOCK_64X64)), "64x64 NONE leaf");
        assert!(seen.contains(&(0, 8, BLOCK_64X32)), "HORZ top leaf");
        assert!(seen.contains(&(4, 8, BLOCK_64X32)), "HORZ bottom leaf");
        assert!(seen.contains(&(8, 0, BLOCK_32X64)), "VERT left leaf");
        assert!(seen.contains(&(8, 4, BLOCK_32X64)), "VERT right leaf");
        assert!(seen.contains(&(8, 8, BLOCK_8X8)), "SPLIT 8x8 leaf");

        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode sequence");
        assert_eq!(frames[1].y, frames[0].y, "p-frame luma != reference");
        assert_eq!(frames[1].u, frames[0].u, "p-frame U != reference");
        assert_eq!(frames[1].v, frames[0].v, "p-frame V != reference");
    }

    /// A `TxModeSelect` P-frame codes a different §6.4.10 tx size on each
    /// of four 32x32 leaves (TX_32X32 / TX_16X16 / TX_8X8 / TX_4X4), each
    /// with a DC-only luma residual, and decodes end-to-end — per-block
    /// **inter** transform-size selection.
    #[test]
    fn inter_tree_tx_select_codes_per_block_tx() {
        use crate::decode_frame::decode_vp9_sequence;

        let kf_hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 0, 0);
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        let p_hdr = inter_header(64, 64);
        let mut partitions = HashMap::new();
        partitions.insert((0u32, 0u32, BLOCK_64X64), PARTITION_SPLIT);
        for (r, c) in [(0u32, 0u32), (0, 4), (4, 0), (4, 4)] {
            partitions.insert((r, c, B32), PARTITION_NONE);
        }
        let tree_plan = InterFrameTreePlan {
            tx_mode: TxMode::TxModeSelect,
            reference_mode: ReferenceMode::SingleReference,
            partitions,
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        // Quadrant -> tx size: TL 32x32, TR 16x16, BL 8x8, BR 4x4.
        let mut planner: Box<InterTreePlanner> = Box::new(|r, c, subsize, _s| {
            assert_eq!(subsize, B32);
            let tx = match (r, c) {
                (0, 0) => 3,
                (0, 4) => 2,
                (4, 0) => 1,
                _ => 0,
            };
            let mut leaf = skip_leaf(subsize, TxMode::TxModeSelect);
            leaf.tx_size = tx;
            leaf.skip = false;
            leaf
        });
        // A DC token on every luma transform block, zero chroma.
        let mut coeffs: Box<FrameCoefSource> =
            Box::new(
                |_r, _c, p, _sx, _sy, _b| {
                    if p == 0 {
                        vec![40i64]
                    } else {
                        Vec::new()
                    }
                },
            );
        let pf = assemble_inter_frame_tree(&p_hdr, &tree_plan, &mut *planner, &mut *coeffs)
            .expect("tx-select p-frame");
        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
        // Every 32x32 quadrant moved off the flat-128 reference (the DC
        // residual covers each leaf at whatever tx size it coded).
        for (qr, qc) in [(0usize, 0usize), (0, 32), (32, 0), (32, 32)] {
            let touched =
                (0..32).any(|i| (0..32).any(|j| frames[1].y[(qr + i) * 64 + qc + j] != 128));
            assert!(touched, "quadrant ({qr},{qc}) untouched by its residual");
        }
        // Chroma is untouched (zero residual, ZEROMV copy).
        assert_eq!(frames[1].u, frames[0].u);
    }

    /// Sub-8x8 leaves (4x4 / 8x4 / 4x8 at the three `BLOCK_8X8`
    /// partition arms) flow through the tree assembler: an all-`ZEROMV`
    /// per-cell plan reconstructs to a verbatim copy of the reference,
    /// and a non-skip sub-8x8 leaf codes its §6.4.21 residual at the
    /// 8x8 grid — end-to-end through `decode_vp9_sequence`.
    #[test]
    fn inter_tree_sub8x8_leaves_decode_end_to_end() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::partition::{PARTITION_HORZ, PARTITION_VERT};
        use crate::residual::{BLOCK_4X4, BLOCK_4X8, BLOCK_8X4};

        let kf_hdr = keyframe_header(64, 64);
        let plan = KeyframePlan::all_skip(8, 8, 0, 0);
        let kf = assemble_keyframe(&kf_hdr, &plan, &mut *no_coeffs()).expect("keyframe");

        let p_hdr = inter_header(64, 64);
        let mut partitions = HashMap::new();
        // Default fallback splits everything to 8x8; three nodes go
        // below: SPLIT -> 4x4, HORZ -> 8x4, VERT -> 4x8.
        partitions.insert((0u32, 0u32, BLOCK_8X8), PARTITION_SPLIT);
        partitions.insert((0, 1, BLOCK_8X8), PARTITION_HORZ);
        partitions.insert((1, 0, BLOCK_8X8), PARTITION_VERT);
        let tree_plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions,
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let zero_sub = crate::inter_block_writer::InterSubBlockSpec {
            modes: [crate::mode_info::ZEROMV; 4],
            mvs: [[[0, 0]; 2]; 4],
        };
        let mut seen: Vec<(u32, u32, u8)> = Vec::new();
        let mut planner: Box<InterTreePlanner> = Box::new(|r, c, subsize, _s| {
            seen.push((r, c, subsize));
            let mut leaf = skip_leaf(subsize, TxMode::Only4x4);
            if subsize < BLOCK_8X8 {
                leaf.sub = Some(zero_sub);
                // The 4x4 leaf codes a real (non-skip) residual: a DC
                // token per luma 4x4 block of the 8x8 grid.
                if subsize == BLOCK_4X4 {
                    leaf.skip = false;
                }
            }
            leaf
        });
        let mut coeffs: Box<FrameCoefSource> =
            Box::new(
                |_r, _c, p, _sx, _sy, _b| {
                    if p == 0 {
                        vec![24i64]
                    } else {
                        Vec::new()
                    }
                },
            );
        let pf = assemble_inter_frame_tree(&p_hdr, &tree_plan, &mut *planner, &mut *coeffs)
            .expect("sub-8x8 p-frame");
        drop(planner);

        assert!(seen.contains(&(0, 0, BLOCK_4X4)), "SPLIT 4x4 leaf");
        assert!(seen.contains(&(0, 1, BLOCK_8X4)), "HORZ 8x4 leaf");
        assert!(seen.contains(&(1, 0, BLOCK_4X8)), "VERT 4x8 leaf");

        let frames = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
        // The non-skip 4x4 leaf's luma moved off the flat-128 reference.
        let touched = (0..8).any(|i| (0..8).any(|j| frames[1].y[i * 64 + j] != 128));
        assert!(touched, "4x4 leaf residual missing");
        // Everything outside the top-left 8x8 is a verbatim copy.
        let copied = (0..64usize)
            .all(|i| (0..64usize).all(|j| (i < 8 && j < 8) || frames[1].y[i * 64 + j] == 128));
        assert!(copied, "ZEROMV sub-8x8 / 8x8 leaves must copy");
        assert_eq!(frames[1].u, frames[0].u, "chroma copies");
    }

    /// The tree assembler validates leaf specs: a leaf `mi_size` that
    /// disagrees with the partition tree, an over-large `tx_size`, a
    /// non-inferred tx on a skip block under `TxModeSelect`, a compound
    /// pair under `SingleReference`, and a `NEWMV` leaf on a
    /// non-error-resilient header are all rejected.
    #[test]
    fn inter_tree_rejects_bad_leaves() {
        let p_hdr = inter_header(64, 64);
        let mk_plan = |tx_mode| InterFrameTreePlan {
            tx_mode,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };

        // Leaf size disagreeing with the tree's subsize.
        let plan = mk_plan(TxMode::Only4x4);
        let mut p: Box<InterTreePlanner> =
            Box::new(|_r, _c, _sz, _s| skip_leaf(B16, TxMode::Only4x4));
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );

        // tx_size above the leaf's maximum.
        let mut p: Box<InterTreePlanner> = Box::new(|_r, _c, sz, _s| {
            let mut l = skip_leaf(sz, TxMode::Only4x4);
            l.tx_size = 3;
            l
        });
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );

        // Skip block under TxModeSelect must carry the inferred size
        // (read_tx_size codes nothing when allowSelect == 0).
        let sel_plan = mk_plan(TxMode::TxModeSelect);
        let mut p: Box<InterTreePlanner> = Box::new(|_r, _c, sz, _s| {
            let mut l = skip_leaf(sz, TxMode::TxModeSelect);
            l.tx_size = 0; // inferred for BLOCK_8X8 under SELECT is TX_8X8.
            l
        });
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &sel_plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );

        // Compound pair under SingleReference.
        let mut p: Box<InterTreePlanner> = Box::new(|_r, _c, sz, _s| {
            let mut l = skip_leaf(sz, TxMode::Only4x4);
            l.ref_frame = [crate::mode_info::LAST_FRAME, crate::mode_info::ALTREF_FRAME];
            l
        });
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );

        // NEWMV without error_resilient_mode.
        let mut p: Box<InterTreePlanner> = Box::new(|_r, _c, sz, _s| {
            let mut l = skip_leaf(sz, TxMode::Only4x4);
            l.y_mode = crate::mode_info::NEWMV;
            l.mv = [[8, 8], [0, 0]];
            l
        });
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported
        );

        // A sub-8x8 leaf without a sub spec, and one with a NEWMV cell
        // on a non-error-resilient header.
        let mut sub_partitions = HashMap::new();
        sub_partitions.insert((0u32, 0u32, BLOCK_8X8), PARTITION_SPLIT);
        let sub_plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: sub_partitions,
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let mut p: Box<InterTreePlanner> =
            Box::new(|_r, _c, sz, _s| skip_leaf(sz, TxMode::Only4x4));
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &sub_plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported,
            "sub-8x8 leaf without a sub spec"
        );
        let mut p: Box<InterTreePlanner> = Box::new(|_r, _c, sz, _s| {
            let mut l = skip_leaf(sz, TxMode::Only4x4);
            if sz < BLOCK_8X8 {
                l.sub = Some(crate::inter_block_writer::InterSubBlockSpec {
                    modes: [
                        crate::mode_info::NEWMV,
                        crate::mode_info::ZEROMV,
                        crate::mode_info::ZEROMV,
                        crate::mode_info::ZEROMV,
                    ],
                    mvs: [[[8, 8], [0, 0]], [[0, 0]; 2], [[0, 0]; 2], [[0, 0]; 2]],
                });
            }
            l
        });
        assert_eq!(
            assemble_inter_frame_tree(&p_hdr, &sub_plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported,
            "NEWMV sub-8x8 cell without error_resilient_mode"
        );
    }

    /// NEARESTMV codes strictly fewer bits than NEWMV for the same
    /// motion vector once a neighbour established the predictor: the
    /// §6.4.18 `assign_mv` arm for predictor-referencing modes emits no
    /// §6.4.20 mv-diff syntax at all — the premise behind the encoder's
    /// NEWMV → NEARESTMV/NEARMV mode mapping.
    #[test]
    fn nearestmv_codes_fewer_bits_than_newmv_for_same_vector() {
        use crate::mode_info::{NEARESTMV, NEWMV};

        let mut hdr = inter_header(64, 64);
        hdr.error_resilient_mode = true; // non-ZEROMV leaves require it.
        hdr.allow_high_precision_mv = true;
        let plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        // Variant A: every 8x8 leaf codes NEWMV [8, 8] (the first leaf
        // establishes the predictor; the rest re-code a zero diff).
        let mk = |modes: bool| -> Vec<u8> {
            let mut first = true;
            let mut planner: Box<InterTreePlanner> = Box::new(move |_r, _c, sz, _s| {
                let mut l = skip_leaf(sz, TxMode::Only4x4);
                l.mv = [[8, 8], [0, 0]];
                if first {
                    l.y_mode = NEWMV;
                    first = false;
                } else {
                    // Variant B maps to NEARESTMV (the writer verifies
                    // the vector equals the derived §6.5.12 NearestMv,
                    // so this also pins the predictor derivation).
                    l.y_mode = if modes { NEARESTMV } else { NEWMV };
                }
                l
            });
            assemble_inter_frame_tree(&hdr, &plan, &mut *planner, &mut *no_coeffs()).expect("frame")
        };
        let all_newmv = mk(false);
        let mapped = mk(true);
        assert!(
            mapped.len() < all_newmv.len(),
            "NEARESTMV frame ({} B) must beat all-NEWMV ({} B)",
            mapped.len(),
            all_newmv.len()
        );
    }

    /// The tree assembler is byte-deterministic and byte-identical to the
    /// legacy all-8x8 assembler when given the fallback (empty) partition
    /// map — the delegation path is a strict generalisation.
    /// §7.2 setup_past_independence( ): an error-resilient frame's
    /// effective sign biases are zero, so `compoundReferenceAllowed` is
    /// 0 and a non-single `reference_mode` is uncodeable — the
    /// assembler must reject it even when the header carries asymmetric
    /// *coded* biases. Likewise a temporal seg-map update is rejected
    /// (PrevSegmentIds is cleared on error-resilient frames).
    #[test]
    fn error_resilient_rejects_compound_and_temporal_seg() {
        use crate::mode_info::{LAST_FRAME, NONE_REF_FRAME, ZEROMV};

        let mut hdr = inter_header(64, 64);
        hdr.error_resilient_mode = true;
        hdr.ref_frame_sign_bias = [false, false, true]; // dead bits under ER
        let plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::ReferenceModeSelect,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let mut p: Box<InterTreePlanner> = Box::new(|_r, _c, subsize, _s| InterTreeLeaf {
            mi_size: subsize,
            tx_size: 0,
            y_mode: ZEROMV,
            interp_filter: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            mv: [[0, 0], [0, 0]],
            skip: true,
            segment_id: 0,
            sub: None,
        });
        let mut c = no_coeffs();
        assert_eq!(
            assemble_inter_frame_tree(&hdr, &plan, &mut *p, &mut *c).unwrap_err(),
            Error::Unsupported,
            "ER + REFERENCE_MODE_SELECT must be uncodeable"
        );

        // Temporal seg-map on an error-resilient frame.
        let mut hdr2 = inter_header(64, 64);
        hdr2.error_resilient_mode = true;
        let mut seg = crate::header::SegmentationParams::default_disabled();
        seg.enabled = true;
        seg.update_map = true;
        seg.temporal_update = true;
        seg.tree_probs = Some([128; 7]);
        seg.pred_prob = Some([128; 3]);
        hdr2.segmentation = seg;
        let plan2 = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: Some(vec![0u8; 64]),
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let mut p2: Box<InterTreePlanner> = Box::new(|_r, _c, subsize, _s| InterTreeLeaf {
            mi_size: subsize,
            tx_size: 0,
            y_mode: ZEROMV,
            interp_filter: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            mv: [[0, 0], [0, 0]],
            skip: true,
            segment_id: 0,
            sub: None,
        });
        let mut c2 = no_coeffs();
        assert_eq!(
            assemble_inter_frame_tree(&hdr2, &plan2, &mut *p2, &mut *c2).unwrap_err(),
            Error::Unsupported,
            "ER + temporal seg-map must be uncodeable"
        );
    }

    #[test]
    fn inter_tree_assembly_matches_all_8x8_and_is_deterministic() {
        let h = inter_header(40, 24);
        let legacy = assemble_inter_frame_all_skip_zeromv(&h).expect("legacy");
        let plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let mut p: Box<InterTreePlanner> =
            Box::new(|_r, _c, sz, _s| skip_leaf(sz, TxMode::Only4x4));
        let a = assemble_inter_frame_tree(&h, &plan, &mut *p, &mut *no_coeffs()).expect("a");
        let b = assemble_inter_frame_tree(&h, &plan, &mut *p, &mut *no_coeffs()).expect("b");
        assert_eq!(a, b, "tree assembly not byte-stable");
        assert_eq!(a, legacy, "empty-map tree != all-8x8 layout");
    }

    /// Multi-tile KEYFRAME assembly at the staged tile-rows fixture's
    /// exact tiling (`tile_cols_log2 = 1`, `tile_rows_log2 = 2` — 4x2 =
    /// 8 tiles at 512x256): the same [`KeyframeTreePlan`] (mixed intra
    /// modes, live per-block residual) decodes through the in-crate §6.4
    /// `decode_tiles( )` walk to the **identical reconstruction** as the
    /// `tile_rows_log2 = 0` assembly of the same plan at the same column
    /// split. Tile ROWS never change the §8 reconstruction — §6.4.4
    /// `AvailU` is the frame-wide `MiRow > 0`, so prediction reads
    /// across a tile-row edge — while the entropy layout changes
    /// completely (8 fresh §9.2 brackets, per-tile-row left resets), so
    /// row-split sample-identity pins the writer's whole tile-row walk.
    /// Tile COLUMNS are *not* reconstruction-invariant (`AvailL` is
    /// `MiCol > MiColStart`: the left neighbour vanishes at a column
    /// boundary), which is why the baseline carries the same column
    /// split — and why a genuinely different single-tile reconstruction
    /// is additionally asserted, pinning that the writer models the
    /// clamped `AvailL` exactly like the decoder rather than predicting
    /// across the column edge.
    #[test]
    fn keyframe_tile_row_split_reconstructs_identically() {
        use crate::partition::tile_payload_sizes;

        let (w, h) = (512u32, 256u32);
        let (mi_rows, mi_cols) = (h / 8, w / 8);
        let mut plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, BLOCK_16X16, 1);
        for (&(lr, lc), leaf) in plan.leaves.iter_mut() {
            // Mixed intra modes: the §6.4.6 default_intra_mode ctx is
            // (abovemode, leftmode) — at a tile-row edge the above MI is
            // still read from the frame-wide state written by the tile
            // above, so varied modes exercise that threading.
            leaf.y_mode = ((lr / 2 + lc / 2) % 10) as u8;
            leaf.uv_mode = ((lr / 2) % 10) as u8;
            leaf.skip = false;
        }
        // A varying DC token per luma transform block: live §6.4.21
        // residual in every tile (AboveNonzeroContext carries across
        // tile rows; LeftNonzeroContext resets per tile row).
        let mk_coeffs = || -> Box<FrameCoefSource<'static>> {
            Box::new(|r, c, p, sx, sy, _b| {
                if p == 0 {
                    vec![1 + ((r + c + sx + sy) % 3) as i64]
                } else {
                    Vec::new()
                }
            })
        };

        // Baseline: two tile columns, ONE tile row.
        let mut hdr_cols = keyframe_header(w, h);
        hdr_cols.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 0,
        };
        let cols_only = assemble_keyframe_tree(&hdr_cols, &plan, &mut *mk_coeffs()).expect("2-col");

        // The fixture tiling: two tile columns x FOUR tile rows.
        let mut hdr_tiled = keyframe_header(w, h);
        hdr_tiled.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 2,
        };
        let tiled = assemble_keyframe_tree(&hdr_tiled, &plan, &mut *mk_coeffs()).expect("tiled");

        // Structural pin: the coded header carries the tiling, and the
        // §6.4.1 tile-size walk consumes the tile data exactly (7 f(32)
        // prefixes + 8 payloads, no residue) — the same walk the staged
        // fixture's notes validate.
        let phdr = crate::header::parse_uncompressed_header(&tiled).expect("tiled header");
        assert_eq!(phdr.tile_info.tile_cols_log2, 1);
        assert_eq!(phdr.tile_info.tile_rows_log2, 2);
        let body_start =
            phdr.uncompressed_header_size_bytes + usize::from(phdr.header_size_in_bytes);
        let body = &tiled[body_start..];
        let sizes = tile_payload_sizes(body, body.len() as u32, 2, 1).expect("tile sizes");
        assert_eq!(sizes.len(), 8, "4x2 = 8 tile payloads");
        let accounted: usize = sizes.iter().map(|&s| s as usize).sum::<usize>() + 7 * 4;
        assert_eq!(accounted, body.len(), "tile walk must consume exactly");

        let f_cols = decode_intra_frame(&cols_only).expect("decode 2-col");
        let f_tiled = decode_intra_frame(&tiled).expect("decode tiled");
        assert_eq!(f_tiled.y, f_cols.y, "luma differs across a row split");
        assert_eq!(f_tiled.u, f_cols.u, "U differs across a row split");
        assert_eq!(f_tiled.v, f_cols.v, "V differs across a row split");

        // And the column split is genuinely reconstruction-changing:
        // the single-tile assembly of the same plan predicts across
        // MI col 32 (AvailL present), the tiled ones must not.
        let hdr_single = keyframe_header(w, h);
        let single = assemble_keyframe_tree(&hdr_single, &plan, &mut *mk_coeffs()).expect("single");
        let f_single = decode_intra_frame(&single).expect("decode single");
        assert_ne!(
            f_single.y, f_cols.y,
            "a tile-column split must change intra AvailL at the boundary"
        );
    }

    /// Degenerate tile rows: at 64x192 (`Sb64Rows = 3`),
    /// `tile_rows_log2 = 2` makes the §6.4.1 `get_tile_offset( )` ladder
    /// produce an EMPTY first tile row (`MiRowStart == MiRowEnd == 0`).
    /// The §6.4 walk still brackets it with its own §9.2 coder (init /
    /// exit around zero superblocks), so the writer must emit a valid
    /// empty-tile payload — and the reconstruction must equal the
    /// single-tile assembly's.
    #[test]
    fn keyframe_degenerate_empty_tile_row_round_trips() {
        let (w, h) = (64u32, 192u32);
        let (mi_rows, mi_cols) = (h / 8, w / 8);
        let plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, BLOCK_32X32, 2);

        let hdr_single = keyframe_header(w, h);
        let single =
            assemble_keyframe_tree(&hdr_single, &plan, &mut *no_coeffs()).expect("single-tile");

        let mut hdr_tiled = keyframe_header(w, h);
        hdr_tiled.tile_info = TileInfo {
            tile_cols_log2: 0,
            tile_rows_log2: 2,
        };
        let tiled = assemble_keyframe_tree(&hdr_tiled, &plan, &mut *no_coeffs()).expect("tiled");

        let f_single = decode_intra_frame(&single).expect("decode single");
        let f_tiled = decode_intra_frame(&tiled).expect("decode tiled");
        assert_eq!(f_tiled.y, f_single.y, "luma differs across tilings");
        assert_eq!(f_tiled.u, f_single.u, "U differs across tilings");
        assert_eq!(f_tiled.v, f_single.v, "V differs across tilings");
    }

    /// Multi-tile INTER assembly at the staged fixture's tiling
    /// (`tile_cols_log2 = 1`, `tile_rows_log2 = 2`, 512x256): a P-frame
    /// mixing `NEWMV` leaves (the §6.5 candidate scans run under each
    /// tile's clamped `MiColStart` / `MiColEnd` window), non-skip
    /// residual, and `ZEROMV` copies reconstructs **identically** to the
    /// single-tile assembly of the same plan through
    /// `decode_vp9_sequence` — and the writer's returned per-MI state
    /// matches across tilings (the §6.4.4 write-back is
    /// tiling-invariant).
    #[test]
    fn inter_tiled_2col_4row_reconstructs_identically_to_single_tile() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::mode_info::{NEWMV, ZEROMV};

        let (w, h) = (512u32, 256u32);
        let (mi_rows, mi_cols) = (h / 8, w / 8);

        // A non-flat keyframe (varying DC per 16x16 leaf) so motion is
        // observable.
        let mut kf_plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, BLOCK_16X16, 1);
        for leaf in kf_plan.leaves.values_mut() {
            leaf.skip = false;
        }
        let mut kf_coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, _sx, _sy, _b| {
            if p == 0 {
                vec![((r * 7 + c * 3) % 11) as i64]
            } else {
                Vec::new()
            }
        });
        let kf_hdr = keyframe_header(w, h);
        let kf = assemble_keyframe_tree(&kf_hdr, &kf_plan, &mut *kf_coeffs).expect("keyframe");

        // Error-resilient P-frame (§7.2.6 UsePrevFrameMvs == 0 on both
        // sides) with NEWMV leaves along the tile-boundary rows/cols.
        let mut p_hdr = inter_header(w, h);
        p_hdr.error_resilient_mode = true;
        let mk_planner = || -> Box<InterTreePlanner<'static>> {
            Box::new(|r, c, subsize, _s| {
                let mut leaf = skip_leaf(subsize, TxMode::Only4x4);
                // Tile-row boundaries sit at MI rows 8/16/24; the tile-col
                // boundary at MI col 32. Put NEWMV blocks around them.
                if (r % 8 == 0 && c % 16 == 0) || ((28..36).contains(&c) && r % 4 == 0) {
                    leaf.y_mode = NEWMV;
                    leaf.mv = [[16, -8], [0, 0]];
                    leaf.skip = false;
                } else {
                    leaf.y_mode = ZEROMV;
                }
                leaf
            })
        };
        let mk_coeffs = || -> Box<FrameCoefSource<'static>> {
            Box::new(|r, c, p, _sx, _sy, _b| {
                if p == 0 {
                    vec![((r + 2 * c) % 5) as i64]
                } else {
                    Vec::new()
                }
            })
        };
        let plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };

        let (pf_single, st_single) = assemble_inter_frame_tree_with_state(
            &p_hdr,
            &plan,
            &mut *mk_planner(),
            &mut *mk_coeffs(),
        )
        .expect("single-tile p-frame");

        let mut p_hdr_tiled = p_hdr;
        p_hdr_tiled.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 2,
        };
        let (pf_tiled, st_tiled) = assemble_inter_frame_tree_with_state(
            &p_hdr_tiled,
            &plan,
            &mut *mk_planner(),
            &mut *mk_coeffs(),
        )
        .expect("tiled p-frame");

        assert_eq!(
            st_tiled.mi_sizes, st_single.mi_sizes,
            "per-MI write-back must be tiling-invariant"
        );
        assert_eq!(st_tiled.skips, st_single.skips);

        let f_single = decode_vp9_sequence(&[&kf, &pf_single]).expect("decode single");
        let f_tiled = decode_vp9_sequence(&[&kf, &pf_tiled]).expect("decode tiled");
        // The NEWMV blocks moved real content (not a pure copy).
        assert_ne!(f_single[1].y, f_single[0].y, "motion must be observable");
        assert_eq!(
            f_tiled[1].y, f_single[1].y,
            "P-frame luma differs across tilings"
        );
        assert_eq!(f_tiled[1].u, f_single[1].u);
        assert_eq!(f_tiled[1].v, f_single[1].v);
    }

    /// Tile-parallel §6.4 decode — KEYFRAME: a live-residual 2col x
    /// 4row 512x256 keyframe (the staged fixture's tiling) decodes
    /// **byte-identically** under every thread budget, and the §9.3.4
    /// count bank — the §6.1.2 `refresh_probs( )` input — is equal
    /// cell-for-cell, pinning the per-worker count merge, not just the
    /// sample merge. Budgets cover the uneven-chunk case (3 workers
    /// over 2 columns clamps to 2) and over-provisioning (8 threads,
    /// 2 columns).
    #[test]
    fn tile_parallel_keyframe_decode_matches_serial_samples_and_counts() {
        use crate::decode_frame::decode_intra_products_for_test;
        use oxideav_core::ExecutionContext;

        let (w, h) = (512u32, 256u32);
        let (mi_rows, mi_cols) = (h / 8, w / 8);
        let mut plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, BLOCK_16X16, 1);
        for (&(lr, lc), leaf) in plan.leaves.iter_mut() {
            leaf.y_mode = ((lr / 2 + lc / 2) % 10) as u8;
            leaf.uv_mode = ((lr / 2) % 10) as u8;
            leaf.skip = false;
        }
        let mut coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, sx, sy, _b| {
            if p == 0 {
                vec![1 + ((r + c + sx + sy) % 3) as i64]
            } else {
                Vec::new()
            }
        });
        let mut hdr = keyframe_header(w, h);
        hdr.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 2,
        };
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("tiled keyframe");

        let (serial, serial_counts) =
            decode_intra_products_for_test(&bytes, &ExecutionContext::serial()).expect("serial");
        for threads in [2usize, 3, 8] {
            let (par, par_counts) =
                decode_intra_products_for_test(&bytes, &ExecutionContext::with_threads(threads))
                    .unwrap_or_else(|e| panic!("parallel decode ({threads} threads): {e}"));
            assert_eq!(par.y, serial.y, "{threads} threads: luma differs");
            assert_eq!(par.u, serial.u, "{threads} threads: U differs");
            assert_eq!(par.v, serial.v, "{threads} threads: V differs");
            assert!(
                *par_counts == *serial_counts,
                "{threads} threads: §9.3.4 count bank differs from the serial walk"
            );
        }
    }

    /// Tile-parallel §6.4 decode — INTER: the 2col x 4row P-frame
    /// stream (NEWMV leaves straddling the tile boundaries, live
    /// residual) decodes **byte-identically** through
    /// [`crate::decode_frame::decode_vp9_sequence_with`] at every
    /// thread budget, packed output compared frame-for-frame against
    /// the serial [`crate::decode_frame::decode_vp9_sequence`].
    #[test]
    fn tile_parallel_inter_sequence_matches_serial() {
        use crate::decode_frame::{decode_vp9_sequence, decode_vp9_sequence_with};
        use crate::mode_info::{NEWMV, ZEROMV};
        use oxideav_core::ExecutionContext;

        let (w, h) = (512u32, 256u32);
        let (mi_rows, mi_cols) = (h / 8, w / 8);

        let mut kf_plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, BLOCK_16X16, 1);
        for leaf in kf_plan.leaves.values_mut() {
            leaf.skip = false;
        }
        let mut kf_coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, _sx, _sy, _b| {
            if p == 0 {
                vec![((r * 7 + c * 3) % 11) as i64]
            } else {
                Vec::new()
            }
        });
        let mut kf_hdr = keyframe_header(w, h);
        kf_hdr.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 2,
        };
        let kf = assemble_keyframe_tree(&kf_hdr, &kf_plan, &mut *kf_coeffs).expect("keyframe");

        let mut p_hdr = inter_header(w, h);
        p_hdr.error_resilient_mode = true;
        p_hdr.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 2,
        };
        let mut planner: Box<InterTreePlanner> = Box::new(|r, c, subsize, _s| {
            let mut leaf = skip_leaf(subsize, TxMode::Only4x4);
            if (r % 8 == 0 && c % 16 == 0) || ((28..36).contains(&c) && r % 4 == 0) {
                leaf.y_mode = NEWMV;
                leaf.mv = [[16, -8], [0, 0]];
                leaf.skip = false;
            } else {
                leaf.y_mode = ZEROMV;
            }
            leaf
        });
        let mut p_coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, _sx, _sy, _b| {
            if p == 0 {
                vec![((r + 2 * c) % 5) as i64]
            } else {
                Vec::new()
            }
        });
        let plan = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let pf = assemble_inter_frame_tree(&p_hdr, &plan, &mut *planner, &mut *p_coeffs)
            .expect("tiled p-frame");

        let serial = decode_vp9_sequence(&[&kf, &pf]).expect("serial decode");
        assert_ne!(serial[1].y, serial[0].y, "motion must be observable");
        for threads in [2usize, 3, 8] {
            let par =
                decode_vp9_sequence_with(&[&kf, &pf], &ExecutionContext::with_threads(threads))
                    .unwrap_or_else(|e| panic!("parallel decode ({threads} threads): {e}"));
            assert_eq!(par.len(), serial.len());
            for (i, (pf, sf)) in par.iter().zip(&serial).enumerate() {
                assert_eq!(
                    pf.to_planar_bytes(),
                    sf.to_planar_bytes(),
                    "{threads} threads: frame {i} differs from the serial decode"
                );
            }
        }
    }

    /// Tile-parallel §6.4 decode — UNEVEN tile columns: 576x64
    /// (`Sb64Cols = 9`, past the §6.2.14 `MIN_TILE_WIDTH_B64 = 4`
    /// bound for `tile_cols_log2 = 1`) splits into 32 + 40 MI columns
    /// per the §6.4.1 `get_tile_offset( )` ladder, which must decode
    /// byte-identically (samples and counts) under a heavily
    /// over-provisioned budget (`effective_workers` clamps 16 threads
    /// to the 2 available columns).
    #[test]
    fn tile_parallel_minimal_two_column_frame_matches_serial() {
        use crate::decode_frame::decode_intra_products_for_test;
        use oxideav_core::ExecutionContext;

        let (w, h) = (576u32, 64u32);
        let (mi_rows, mi_cols) = (h / 8, w / 8);
        let mut plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, BLOCK_16X16, 1);
        for (&(lr, lc), leaf) in plan.leaves.iter_mut() {
            leaf.y_mode = ((lr + lc) % 10) as u8;
            leaf.skip = false;
        }
        let mut coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, _sx, _sy, _b| {
            if p == 0 {
                vec![((r * 3 + c) % 7) as i64]
            } else {
                Vec::new()
            }
        });
        let mut hdr = keyframe_header(w, h);
        hdr.tile_info = TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 0,
        };
        let bytes = assemble_keyframe_tree(&hdr, &plan, &mut *coeffs).expect("2-col keyframe");

        let (serial, serial_counts) =
            decode_intra_products_for_test(&bytes, &ExecutionContext::serial()).expect("serial");
        let (par, par_counts) =
            decode_intra_products_for_test(&bytes, &ExecutionContext::with_threads(16))
                .expect("parallel");
        assert_eq!(par.y, serial.y);
        assert_eq!(par.u, serial.u);
        assert_eq!(par.v, serial.v);
        assert!(*par_counts == *serial_counts, "count bank differs");
    }

    /// [`PrevMotionField::after_intra_frame`] equals the actual §6.4.4
    /// write-back state a keyframe leaves (`ref_frame = [ INTRA, NONE ]`
    /// with zero vectors on every MI cell) — pinned against the keyframe
    /// assembler's own returned state so the chained sequence encoders'
    /// first P-frame models the identical prev field the decoder scans.
    #[test]
    fn prev_motion_field_after_intra_matches_keyframe_state() {
        let hdr = keyframe_header(64, 64);
        let mut plan = KeyframeTreePlan::uniform(8, 8, BLOCK_16X16, 1);
        for leaf in plan.leaves.values_mut() {
            leaf.skip = false;
        }
        let mut coeffs: Box<FrameCoefSource> = Box::new(
            |_r, _c, p, _sx, _sy, _b| {
                if p == 0 {
                    vec![3]
                } else {
                    Vec::new()
                }
            },
        );
        let (_bytes, st) =
            assemble_keyframe_tree_with_state(&hdr, &plan, &mut *coeffs).expect("kf");
        let f = PrevMotionField::after_intra_frame(8, 8);
        assert_eq!(f.ref_frames, st.ref_frames, "reference pairs");
        assert_eq!(f.mvs, st.mvs, "vectors");
    }

    /// §7.2.6 `UsePrevFrameMvs == 1` writer model, end-to-end: a
    /// non-error-resilient SHOWN P-frame chain (keyframe → P1 → P2, no
    /// hidden/intra predecessor) codes real motion by supplying each
    /// frame's plan with the previous frame's §6.4.4 motion field. Three
    /// pins:
    ///
    /// 1. P2 codes `NEARMV` with a vector whose ONLY §6.5.10 source is
    ///    the previous-frame pass — every spatial neighbour is `ZEROMV`,
    ///    so `NearMv` equals the prev-frame candidate exactly when the
    ///    writer scans the supplied field.
    /// 2. The same `NEARMV` plan WITHOUT the prev field is uncodeable
    ///    (`Error::Unsupported` from the mv-equals-predictor check) —
    ///    the differential proof that the field genuinely feeds the
    ///    predictors rather than being carried inertly.
    /// 3. The chain's decoded output is sample-identical to an
    ///    error-resilient twin coding the identical final vectors as
    ///    `NEWMV` (`UsePrevFrameMvs == 0` on both sides): the §8
    ///    reconstruction depends only on the final vectors, while the
    ///    entropy paths differ completely — and the prev chain decodes
    ///    through the decoder's REAL §7.2.6 derivation (shown
    ///    same-sized predecessor, non-ER header ⇒ it scans the prev
    ///    field), so any writer/decoder prev-model mismatch desyncs.
    #[test]
    fn inter_tree_prev_frame_mvs_chain_matches_er_twin() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::mode_info::{NEARMV, NEWMV, ZEROMV};

        let (w, h) = (64u32, 64u32);
        // Non-flat keyframe: per-16x16 varying DC so displaced content
        // is visibly different.
        let mut kf_plan = KeyframeTreePlan::uniform(8, 8, BLOCK_16X16, 1);
        for leaf in kf_plan.leaves.values_mut() {
            leaf.skip = false;
        }
        let mut kf_coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, _sx, _sy, _b| {
            if p == 0 {
                vec![((r * 5 + c * 3) % 9) as i64 + 1]
            } else {
                Vec::new()
            }
        });
        let kf_hdr = keyframe_header(w, h);
        let (kf, kf_state) =
            assemble_keyframe_tree_with_state(&kf_hdr, &kf_plan, &mut *kf_coeffs).expect("kf");

        let snapshot = |st: &Vp9FrameState| PrevMotionField {
            ref_frames: st.ref_frames.clone(),
            mvs: st.mvs.clone(),
        };

        // 8 px down, 8 px left: crosses the 16x16 content bands.
        let mv = [[64, -64], [0, 0]];
        let mk_planner = move |target_mode: u8| -> Box<InterTreePlanner<'static>> {
            Box::new(move |r, c, subsize, _s| {
                let mut leaf = skip_leaf(subsize, TxMode::Only4x4);
                if (r, c) == (4, 4) {
                    leaf.y_mode = target_mode;
                    leaf.mv = mv;
                } else {
                    leaf.y_mode = ZEROMV;
                }
                leaf
            })
        };
        let mk_plan = |prev: Option<PrevMotionField>| InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: prev,
        };

        let mut p_hdr = inter_header(w, h);
        p_hdr.refresh_frame_flags = 0x01;
        p_hdr.ref_frame_idx = Some([0, 0, 0]);
        assert!(!p_hdr.error_resilient_mode);

        // P1: NEWMV over the keyframe. The decoder derives
        // UsePrevFrameMvs = 1 here too (the keyframe is shown and
        // same-sized), so the writer must model the scan over the
        // keyframe's state — an all-INTRA field with no matching
        // candidates, but scanned on both sides.
        let plan1 = mk_plan(Some(snapshot(&kf_state)));
        let (p1, p1_state) = assemble_inter_frame_tree_with_state(
            &p_hdr,
            &plan1,
            &mut *mk_planner(NEWMV),
            &mut *no_coeffs(),
        )
        .expect("P1");

        // P2: NEARMV — the (64,-64) candidate only exists through the
        // prev-frame pass over P1's motion field.
        let plan2 = mk_plan(Some(snapshot(&p1_state)));
        let p2 =
            assemble_inter_frame_tree(&p_hdr, &plan2, &mut *mk_planner(NEARMV), &mut *no_coeffs())
                .expect("P2 NEARMV via prev-frame candidate");

        // Differential pin: without the prev field the NEARMV predictor
        // is ZeroMv, so the identical plan is rejected.
        let mut plan2_no_prev = mk_plan(None);
        plan2_no_prev.prev_frame_mvs_absent = true;
        assert_eq!(
            assemble_inter_frame_tree(
                &p_hdr,
                &plan2_no_prev,
                &mut *mk_planner(NEARMV),
                &mut *no_coeffs(),
            )
            .unwrap_err(),
            Error::Unsupported,
            "the NEARMV vector must be reachable ONLY through the prev field"
        );

        // The error-resilient twin: identical final vectors as NEWMV.
        let mut er_hdr = p_hdr;
        er_hdr.error_resilient_mode = true;
        let er_plan = mk_plan(None);
        let p1_er = assemble_inter_frame_tree(
            &er_hdr,
            &er_plan,
            &mut *mk_planner(NEWMV),
            &mut *no_coeffs(),
        )
        .expect("P1 er");
        let p2_er = assemble_inter_frame_tree(
            &er_hdr,
            &er_plan,
            &mut *mk_planner(NEWMV),
            &mut *no_coeffs(),
        )
        .expect("P2 er");

        let fa = decode_vp9_sequence(&[&kf, &p1, &p2]).expect("decode prev-mv chain");
        let fb = decode_vp9_sequence(&[&kf, &p1_er, &p2_er]).expect("decode er twin");
        assert_ne!(fa[1].y, fa[0].y, "P1 motion must be visible");
        for i in 1..3 {
            assert_eq!(fa[i].y, fb[i].y, "frame {i}: luma differs vs er twin");
            assert_eq!(fa[i].u, fb[i].u, "frame {i}: U differs vs er twin");
            assert_eq!(fa[i].v, fb[i].v, "frame {i}: V differs vs er twin");
        }
        // And the streams themselves differ (different entropy paths).
        assert_ne!(p2, p2_er, "prev-modeled stream must differ from the twin");
    }

    /// The prev-motion-field plan is validated up front: supplying it on
    /// an error-resilient header (§7.2.6 derives 0 there), together with
    /// the `prev_frame_mvs_absent` assertion (a contradiction), or with
    /// arrays that don't span the MI grid, are all rejected.
    #[test]
    fn inter_tree_prev_frame_mvs_rejections() {
        let (w, h) = (64u32, 64u32);
        let mk_plan = |prev: Option<PrevMotionField>| InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: prev,
        };
        let n = 8usize * 8 * 2;
        let good = PrevMotionField {
            ref_frames: vec![crate::mode_info::LAST_FRAME; n],
            mvs: vec![(0, 0); n],
        };
        let mut p: Box<InterTreePlanner> =
            Box::new(|_r, _c, sz, _s| skip_leaf(sz, TxMode::Only4x4));

        // Error-resilient header + prev field.
        let mut er_hdr = inter_header(w, h);
        er_hdr.error_resilient_mode = true;
        assert_eq!(
            assemble_inter_frame_tree(
                &er_hdr,
                &mk_plan(Some(good.clone())),
                &mut *p,
                &mut *no_coeffs()
            )
            .unwrap_err(),
            Error::Unsupported,
            "ER header derives UsePrevFrameMvs = 0"
        );

        // Contradiction: absent assertion + supplied field.
        let hdr = inter_header(w, h);
        let mut plan = mk_plan(Some(good.clone()));
        plan.prev_frame_mvs_absent = true;
        assert_eq!(
            assemble_inter_frame_tree(&hdr, &plan, &mut *p, &mut *no_coeffs()).unwrap_err(),
            Error::Unsupported,
            "absent + supplied is a contradiction"
        );

        // Short arrays.
        let bad = PrevMotionField {
            ref_frames: vec![crate::mode_info::LAST_FRAME; n - 2],
            mvs: vec![(0, 0); n],
        };
        assert_eq!(
            assemble_inter_frame_tree(&hdr, &mk_plan(Some(bad)), &mut *p, &mut *no_coeffs())
                .unwrap_err(),
            Error::Unsupported,
            "prev arrays must span the MI grid"
        );
    }

    /// Compound prediction on a SHOWN non-error-resilient chain — the
    /// header shape that was uncodeable before the prev-motion-field
    /// model (compound needs non-ER sign biases, and a non-ER frame
    /// after a shown same-sized predecessor decodes with
    /// `UsePrevFrameMvs == 1`): keyframe (parked in slot 1 as ALTREF) →
    /// shown P1 (slot 0) → P2 coding `[ LAST, ALTREF ]` compound
    /// `ZEROMV` skip everywhere plus one compound `NEWMV` block, with
    /// P1's motion field supplied as the prev model. Every `ZEROMV`
    /// compound sample must equal the §8.5.2
    /// `Round2( LAST + ALTREF, 1 )` average of the two decoded
    /// references — computed independently from the decoded frames.
    #[test]
    fn inter_tree_compound_on_shown_chain_via_prev_mvs() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::mode_info::{ALTREF_FRAME, LAST_FRAME, NEWMV, ZEROMV};

        let (w, h) = (64u32, 64u32);
        let mut kf_plan = KeyframeTreePlan::uniform(8, 8, BLOCK_16X16, 1);
        for leaf in kf_plan.leaves.values_mut() {
            leaf.skip = false;
        }
        let mut kf_coeffs: Box<FrameCoefSource> = Box::new(|r, c, p, _sx, _sy, _b| {
            if p == 0 {
                vec![((r * 3 + c * 7) % 8) as i64 + 1]
            } else {
                Vec::new()
            }
        });
        let kf_hdr = keyframe_header(w, h);
        let (kf, kf_state) =
            assemble_keyframe_tree_with_state(&kf_hdr, &kf_plan, &mut *kf_coeffs).expect("kf");

        let snapshot = |st: &Vp9FrameState| PrevMotionField {
            ref_frames: st.ref_frames.clone(),
            mvs: st.mvs.clone(),
        };

        // P1: shown, non-ER, refreshes slot 0 only (slot 1 keeps the
        // keyframe), real motion everywhere below MI row 4.
        let mut p1_hdr = inter_header(w, h);
        p1_hdr.refresh_frame_flags = 0x01;
        p1_hdr.ref_frame_idx = Some([0, 0, 0]);
        let plan1 = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::SingleReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: Some(snapshot(&kf_state)),
        };
        let mut planner1: Box<InterTreePlanner> = Box::new(|r, _c, sz, _s| {
            let mut leaf = skip_leaf(sz, TxMode::Only4x4);
            if r >= 4 {
                leaf.y_mode = NEWMV;
                leaf.mv = [[-32, 32], [0, 0]];
            } else {
                leaf.y_mode = ZEROMV;
            }
            leaf
        });
        let (p1, p1_state) = assemble_inter_frame_tree_with_state(
            &p1_hdr,
            &plan1,
            &mut *planner1,
            &mut *no_coeffs(),
        )
        .expect("P1");

        // P2: LAST = slot 0 (P1), ALTREF = slot 1 (keyframe), asymmetric
        // sign bias (compoundReferenceAllowed), non-ER, shown — prev
        // model = P1's motion field.
        let mut p2_hdr = inter_header(w, h);
        p2_hdr.refresh_frame_flags = 0x04;
        p2_hdr.ref_frame_idx = Some([0, 1, 1]);
        p2_hdr.ref_frame_sign_bias = [false, false, true];
        let plan2 = InterFrameTreePlan {
            tx_mode: TxMode::Only4x4,
            reference_mode: ReferenceMode::CompoundReference,
            partitions: HashMap::new(),
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: Some(snapshot(&p1_state)),
        };
        let mut planner2: Box<InterTreePlanner> = Box::new(|r, c, sz, _s| {
            let mut leaf = skip_leaf(sz, TxMode::Only4x4);
            leaf.ref_frame = [LAST_FRAME, ALTREF_FRAME];
            if (r, c) == (4, 4) {
                leaf.y_mode = NEWMV;
                leaf.mv = [[16, 8], [16, 8]];
            } else {
                leaf.y_mode = ZEROMV;
            }
            leaf
        });
        let p2 = assemble_inter_frame_tree(&p2_hdr, &plan2, &mut *planner2, &mut *no_coeffs())
            .expect("P2 compound on a shown chain");

        let frames = decode_vp9_sequence(&[&kf, &p1, &p2]).expect("decode");
        assert_eq!(frames.len(), 3);
        let (f0, f1, f2) = (&frames[0], &frames[1], &frames[2]);
        assert_ne!(f1.y, f0.y, "P1 motion must be visible");

        // Outside the NEWMV block (luma px 32..40 x 32..40), every
        // sample is the compound ZEROMV average of the two references.
        let avg = |a: u16, b: u16| -> u16 { (a + b + 1) >> 1 };
        for row in 0..64usize {
            for col in 0..64usize {
                if (32..40).contains(&row) && (32..40).contains(&col) {
                    continue;
                }
                let want = avg(f1.y[row * 64 + col], f0.y[row * 64 + col]);
                assert_eq!(
                    f2.y[row * 64 + col],
                    want,
                    "luma ({row},{col}) != Round2(LAST + ALTREF, 1)"
                );
            }
        }
        for i in 0..f2.u.len() {
            let (urow, ucol) = (i / 32, i % 32);
            if (16..20).contains(&urow) && (16..20).contains(&ucol) {
                continue;
            }
            assert_eq!(f2.u[i], avg(f1.u[i], f0.u[i]), "U ({urow},{ucol})");
            assert_eq!(f2.v[i], avg(f1.v[i], f0.v[i]), "V ({urow},{ucol})");
        }
    }
}
