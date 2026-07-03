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
//! 3. the §6.4 tile data — a single tile (`tile_cols_log2 ==
//!    tile_rows_log2 == 0`, so no `tile_size` prefix), a §9.2 bool-coded
//!    payload produced by walking the §6.4.3 partition recursion
//!    ([`crate::partition_writer`]) and writing each leaf block's §6.4.4
//!    mode info + §6.4.21 residual ([`crate::block_writer`]).
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
/// leaf block codes. The leaf's `MiSize` comes from the partition tree
/// itself (the `subsize` at the leaf call site).
// Consumed by the tree-plan lossy encoder (the next step in this
// subsystem); exercised now by the assembler round-trip tests below.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeLeafPlan {
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
#[allow(dead_code)]
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

/// Assemble a complete VP9 keyframe from `hdr` + a [`KeyframeTreePlan`],
/// returning the full frame bytes — [`assemble_keyframe`] generalised
/// over the partition tree and per-leaf transform sizes.
///
/// Each leaf writes the §6.4.4 intra block at the tree's `subsize` with
/// the plan's `tx_size`: coded via the §6.4.10 tree under
/// `TxModeSelect`, or validated against the inferred
/// `Min( maxTxSize, tx_mode_to_biggest_tx_size )` otherwise (a mismatch
/// would silently desync the reconstruction, so it is rejected).
#[allow(dead_code)]
pub(crate) fn assemble_keyframe_tree(
    hdr: &Vp9FrameHeader,
    plan: &KeyframeTreePlan,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    if hdr.frame_type != FrameType::KeyFrame || hdr.intra_only {
        return Err(Error::Unsupported);
    }
    if hdr.tile_info.tile_cols_log2 != 0 || hdr.tile_info.tile_rows_log2 != 0 {
        return Err(Error::Unsupported);
    }

    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;

    // §6.3 compressed header (default-probability path).
    let chdr_bytes = write_compressed_header_intra(plan.tx_mode, hdr.quantization.lossless)?;

    // §6.2 uncompressed header with header_size_in_bytes = compressed len.
    let mut hdr2 = *hdr;
    hdr2.header_size_in_bytes = u16::try_from(chdr_bytes.len()).map_err(|_| Error::Unsupported)?;
    let uhdr_bytes = write_uncompressed_header(&hdr2)?;

    // §6.4 tile data: a single bool-coded payload over the plan's tree.
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

    pctx.clear_above();

    let mut r = 0u32;
    while r < mi_rows {
        pctx.clear_left();
        let mut c = 0u32;
        while c < mi_cols {
            let mut layout = |lr: u32, lc: u32, bsize: u8| -> u8 {
                plan.partitions
                    .get(&(lr, lc, bsize))
                    .copied()
                    .unwrap_or(if bsize == BLOCK_8X8 {
                        PARTITION_NONE
                    } else {
                        PARTITION_SPLIT
                    })
            };
            let mut leaf =
                |enc: &mut BoolEncoder, lr: u32, lc: u32, subsize: u8| -> Result<(), Error> {
                    let lp = plan
                        .leaves
                        .get(&(lr, lc))
                        .copied()
                        .ok_or(Error::Unsupported)?;
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
                        mi_col_start: 0,
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

    let tile_bytes = enc.finish();

    let mut out = Vec::with_capacity(uhdr_bytes.len() + chdr_bytes.len() + tile_bytes.len());
    out.extend_from_slice(&uhdr_bytes);
    out.extend_from_slice(&chdr_bytes);
    out.extend_from_slice(&tile_bytes);
    Ok(out)
}

/// Assemble a complete VP9 keyframe from `hdr` + `plan`, returning the
/// full frame bytes (uncompressed header + compressed header + single
/// tile). `coeffs` supplies, per coded transform block, the quantized
/// `Tokens` array — only called for non-skip blocks.
///
/// `hdr` must be a key frame (`FrameType::KeyFrame`, `!intra_only`,
/// `tile_cols_log2 == tile_rows_log2 == 0`); the `header_size_in_bytes`
/// field is overwritten with the actual compressed-header length.
pub(crate) fn assemble_keyframe(
    hdr: &Vp9FrameHeader,
    plan: &KeyframePlan,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    if hdr.frame_type != FrameType::KeyFrame || hdr.intra_only {
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
        Box::new(|_r, _c, _state| (crate::mode_info::ZEROMV, [0, 0]));
    assemble_inter_frame_planned(hdr, skip_all, &mut *planner, coeffs)
}

/// Per-block inter mode planner: called once per `BLOCK_8X8` leaf (in
/// §6.4.3 partition order, i.e. exactly the decode order) with the MI
/// coordinates and the **shared** [`Vp9FrameState`] as it stands *before*
/// this block is written — the same state the inter block writer derives
/// its §6.5 MV predictors and §9.3.2 contexts from — and returns the
/// block's `(y_mode, mv)` pair: `ZEROMV` with `[0, 0]`, or `NEWMV` with
/// an eighth-pel `[row, col]` vector (the difference onto the §6.5.12
/// `BestMv` is coded, so the planner must pick a difference the §6.4.20
/// decomposition can carry under the frame's high-precision gate).
pub(crate) type InterBlockPlanner<'f> = dyn FnMut(u32, u32, &Vp9FrameState) -> (u8, [i32; 2]) + 'f;

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
pub(crate) fn assemble_inter_frame_planned(
    hdr: &Vp9FrameHeader,
    skip_all: bool,
    planner: &mut InterBlockPlanner<'_>,
    coeffs: &mut FrameCoefSource<'_>,
) -> Result<Vec<u8>, Error> {
    use crate::compressed::ReferenceMode;
    use crate::compressed_writer::write_compressed_header_inter;
    use crate::inter_block_writer::{
        write_inter_block, InterBlockFrameCtx, InterBlockProbs, InterBlockSpec,
    };
    use crate::mode_info::{LAST_FRAME, NONE_REF_FRAME, ZEROMV};

    if hdr.frame_type != FrameType::NonKeyFrame || hdr.intra_only || !hdr.show_frame {
        return Err(Error::Unsupported);
    }
    if hdr.tile_info.tile_cols_log2 != 0 || hdr.tile_info.tile_rows_log2 != 0 {
        return Err(Error::Unsupported);
    }
    if hdr.ref_frame_idx.is_none() {
        return Err(Error::Unsupported);
    }

    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let tx_mode = TxMode::Only4x4;
    let reference_mode = ReferenceMode::SingleReference;
    let switchable = hdr.interpolation_filter == 4;
    // sign_bias indexed by §3 ref value: [INTRA, LAST, GOLDEN, ALTREF].
    let sign_bias = [
        false,
        hdr.ref_frame_sign_bias[0],
        hdr.ref_frame_sign_bias[1],
        hdr.ref_frame_sign_bias[2],
    ];

    // §6.3 compressed header (default-probability path).
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
    let comp_config = crate::compressed::CompoundReferenceConfig {
        fixed_ref: crate::mode_info::ALTREF_FRAME,
        var_ref: [LAST_FRAME, crate::mode_info::GOLDEN_FRAME],
    };

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

    let fctx = InterBlockFrameCtx {
        mi_cols,
        mi_rows,
        mi_col_end: mi_cols,
        subsampling_x: ssx,
        subsampling_y: ssy,
        bit_depth,
        lossless: hdr.quantization.lossless,
        tx_mode,
        seg_enabled: false,
        seg_update_map: false,
        reference_mode,
        comp_config,
        sign_bias: &sign_bias,
        interpolation_filter: hdr.interpolation_filter,
        allow_high_precision_mv: hdr.allow_high_precision_mv,
        use_prev_frame_mvs: false,
    };
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
        tree_probs: None,
    };

    pctx.clear_above();

    let mut r = 0u32;
    while r < mi_rows {
        pctx.clear_left();
        let mut c = 0u32;
        while c < mi_cols {
            let mut leaf =
                |enc: &mut BoolEncoder, lr: u32, lc: u32, _ls: u8| -> Result<(), Error> {
                    let (y_mode, mv) = planner(lr, lc, &state);
                    if y_mode != ZEROMV && !hdr.error_resilient_mode {
                        // §7.2.6: the decoder would run the §6.5 scan with
                        // UsePrevFrameMvs == 1, which this writer does not
                        // model — see the function docs.
                        return Err(Error::Unsupported);
                    }
                    let spec = InterBlockSpec {
                        r: lr,
                        mi_col_start: 0,
                        c: lc,
                        mi_size: BLOCK_8X8,
                        segment_id: 0,
                        skip: skip_all,
                        tx_size: 0,
                        ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                        y_mode,
                        interp_filter: if switchable {
                            0
                        } else {
                            hdr.interpolation_filter
                        },
                        mv: [mv, [0, 0]],
                    };
                    let mut src = |p: usize, tx_sz: u32, sx: u32, sy: u32, b: usize| {
                        let n0 = 1usize << (tx_sz + 2);
                        if skip_all {
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
                        &mut nz,
                        &mut token_cache,
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
                PartitionProbsKind::Inter(&ctx.partition_probs),
                &mut leaf,
            )?;
            c += 8;
        }
        r += 8;
    }

    let tile_bytes = enc.finish();

    let mut out = Vec::with_capacity(uhdr_bytes.len() + chdr_bytes.len() + tile_bytes.len());
    out.extend_from_slice(&uhdr_bytes);
    out.extend_from_slice(&chdr_bytes);
    out.extend_from_slice(&tile_bytes);
    Ok(out)
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
        let mut partitions = HashMap::new();
        let mut leaves = HashMap::new();
        // Walk every recursion node: split above leaf_size, NONE at it.
        for r in (0..mi_rows).step_by(8) {
            for c in (0..mi_cols).step_by(8) {
                fill_uniform(&mut partitions, &mut leaves, r, c, BLOCK_64X64, leaf_size);
            }
        }
        for lp in leaves.values_mut() {
            lp.tx_size = tx_size;
            lp.skip = skip;
        }
        KeyframeTreePlan {
            tx_mode: TxMode::TxModeSelect,
            partitions,
            leaves,
        }
    }

    fn fill_uniform(
        partitions: &mut HashMap<(u32, u32, u8), u8>,
        leaves: &mut HashMap<(u32, u32), TreeLeafPlan>,
        r: u32,
        c: u32,
        bsize: u8,
        leaf_size: u8,
    ) {
        use crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP;
        if bsize == leaf_size {
            partitions.insert((r, c, bsize), crate::partition::PARTITION_NONE);
            leaves.insert(
                (r, c),
                TreeLeafPlan {
                    tx_size: 0,
                    y_mode: 0,
                    uv_mode: 0,
                    skip: false,
                    segment_id: 0,
                },
            );
            return;
        }
        partitions.insert((r, c, bsize), crate::partition::PARTITION_SPLIT);
        let half = (NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] >> 1) as u32;
        let subsize = crate::partition::SUBSIZE_LOOKUP[crate::partition::PARTITION_SPLIT as usize]
            [bsize as usize];
        for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
            fill_uniform(partitions, leaves, r + dr, c + dc, subsize, leaf_size);
        }
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
        for key in [(0u32, 0u32), (4, 0), (0, 8), (0, 12)] {
            leaves.insert(
                key,
                TreeLeafPlan {
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
}
