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
use crate::partition::{PartitionContextState, PartitionProbsKind};
use crate::partition_writer::write_partition_8x8;
use crate::residual::{BLOCK_64X64, BLOCK_8X8};
use crate::tokens::NonzeroContext;
use crate::Error;

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
}
