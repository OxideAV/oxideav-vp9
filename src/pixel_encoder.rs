//! Pixel-accurate VP9 **keyframe encoder** — encodes arbitrary input
//! samples into a keyframe that the §8 decode process reconstructs
//! **byte-exact** (lossless mode).
//!
//! The encoder is built as a mirror of the decoder's own reconstruction
//! loop. For every coded transform block — visited in exactly the order
//! the §6.4.21 `residual( )` walk decodes them, via the
//! [`crate::frame_writer`] coefficient callback — it:
//!
//! 1. runs the decoder's §8.5.1 [`predict_intra`] over the encoder's
//!    reconstruction planes (initialised to zero exactly like the
//!    decoder's `CurrFrame`, and updated only by coded blocks), with the
//!    identical `AvailL || x > 0` / `AvailU || y > 0` /
//!    `x + step < num4x4w` availability derivation the decoder applies;
//! 2. forms the residual `target − prediction` for the block;
//! 3. forward-transforms it with the exact
//!    [`crate::fwd_transform::forward_wht_2d`] (lossless mode codes every
//!    block as a 4x4 WHT, quantizer 4, which round-trips bit-exactly);
//! 4. replays the decoder's §8.6.2 [`reconstruct_block`] into the
//!    reconstruction planes so the *next* block's prediction sees exactly
//!    the state the decoder will see.
//!
//! Because every state transition is the decoder's own, the emitted
//! bitstream reconstructs to the target planes with **zero** error —
//! validated end-to-end through [`crate::decode_frame::decode_intra_frame`]
//! in the test suite (and in `tests/encode_lossless.rs`).
//!
//! The target planes cover the MI-padded working extents
//! (`MiCols * 8 × MiRows * 8` luma); samples outside the visible
//! `width × height` crop are edge-replicated from the source so
//! frame-edge blocks (which the decoder reconstructs at the padded
//! extents and crops on output) carry deterministic content.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.21 / §8.5.1 / §8.6 / §8.7; the
//! walk order, availability rules and reconstruction state mirror the
//! in-crate decoder exactly.

use crate::dequant::{get_ac_quant, get_dc_quant};
use crate::frame_writer::{
    assemble_inter_frame_zeromv, assemble_keyframe, BlockPlan, FrameCoefSource, KeyframePlan,
};
use crate::fwd_transform::forward_wht_2d;
use crate::header::{
    ColorConfig, ColorSpace, FrameType, LoopFilterParams, QuantizationParams, SegmentationParams,
    TileInfo, Vp9FrameHeader,
};
use crate::idct::DCT_DCT;
use crate::inter_mv::{BlockGrid, ScaleGeom};
use crate::inter_pred::{predict_inter, InterPredArgs, RefPlane, RefPlanes};
use crate::intra::{predict_intra, Plane, PredMode};
use crate::reconstruct::reconstruct_block;
use crate::residual::{get_plane_block_size, BLOCK_8X8, NUM_4X4_BLOCKS_WIDE_LOOKUP};
use crate::Error;

/// Build a padded target [`Plane`] from a row-major 8-bit source
/// rectangle: the visible `vis_w × vis_h` samples are copied through and
/// the right / bottom padding (up to `pad_w × pad_h`) is edge-replicated,
/// matching the §8.10 crop convention (padding never reaches the output
/// but participates in prediction near the frame edge).
pub(crate) fn padded_plane_from_bytes(
    data: &[u8],
    vis_w: usize,
    vis_h: usize,
    pad_w: usize,
    pad_h: usize,
) -> Plane {
    debug_assert!(data.len() >= vis_w * vis_h);
    debug_assert!(pad_w >= vis_w && pad_h >= vis_h && vis_w > 0 && vis_h > 0);
    let mut plane = Plane::new(pad_w, pad_h);
    for y in 0..pad_h {
        let sy = y.min(vis_h - 1);
        for x in 0..pad_w {
            let sx = x.min(vis_w - 1);
            plane.set(x, y, i32::from(data[sy * vis_w + sx]));
        }
    }
    plane
}

/// Build a padded target [`Plane`] from a row-major `u16` source
/// rectangle (10/12-bit content), edge-replicating the right / bottom
/// padding exactly like [`padded_plane_from_bytes`].
pub(crate) fn padded_plane_from_u16(
    data: &[u16],
    vis_w: usize,
    vis_h: usize,
    pad_w: usize,
    pad_h: usize,
) -> Plane {
    debug_assert!(data.len() >= vis_w * vis_h);
    debug_assert!(pad_w >= vis_w && pad_h >= vis_h && vis_w > 0 && vis_h > 0);
    let mut plane = Plane::new(pad_w, pad_h);
    for y in 0..pad_h {
        let sy = y.min(vis_h - 1);
        for x in 0..pad_w {
            let sx = x.min(vis_w - 1);
            plane.set(x, y, i32::from(data[sy * vis_w + sx]));
        }
    }
    plane
}

/// The §6.2 header for a lossless keyframe: profile 0, 8-bit 4:2:0,
/// `base_q_idx == 0` with zero deltas (the §6.2.9 `Lossless` derivation),
/// loop filter off (lossless reconstruction must not be filtered),
/// single tile.
pub(crate) fn lossless_keyframe_header(width: u32, height: u32) -> Vp9FrameHeader {
    lossless_keyframe_header_ex(width, height, 0, 8, true, true)
}

/// [`lossless_keyframe_header`] generalised over the §6.2 profile /
/// bit-depth / chroma-subsampling triple:
///
/// * profile 0 — 8-bit 4:2:0;
/// * profile 1 — 8-bit 4:4:4 (`subsampling_x == subsampling_y == false`);
/// * profile 2 — 10/12-bit 4:2:0;
/// * profile 3 — 10/12-bit 4:4:4.
pub(crate) fn lossless_keyframe_header_ex(
    width: u32,
    height: u32,
    profile: u8,
    bit_depth: u8,
    ssx: bool,
    ssy: bool,
) -> Vp9FrameHeader {
    Vp9FrameHeader {
        profile,
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        frame_type: FrameType::KeyFrame,
        show_frame: true,
        error_resilient_mode: false,
        intra_only: false,
        color_config: ColorConfig {
            bit_depth,
            color_space: ColorSpace::Bt601,
            color_range_full: false,
            subsampling_x: ssx,
            subsampling_y: ssy,
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
            base_q_idx: 0,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: true,
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

/// Encoder-side reconstruction state: the three working planes plus the
/// frame geometry needed to mirror the decoder's per-block prediction.
pub(crate) struct ReconState {
    /// Reconstruction planes (luma, U, V) at the MI-padded extents,
    /// zero-initialised exactly like the decoder's `CurrFrame`.
    pub planes: [Plane; 3],
    /// `MiCols` / `MiRows`.
    pub mi_cols: u32,
    /// See `mi_cols`.
    pub mi_rows: u32,
    /// Chroma subsampling.
    pub subsampling_x: bool,
    /// See `subsampling_x`.
    pub subsampling_y: bool,
    /// `BitDepth`.
    pub bit_depth: u32,
}

impl ReconState {
    /// Fresh state for a `mi_cols × mi_rows` frame.
    pub fn new(mi_cols: u32, mi_rows: u32, ssx: bool, ssy: bool, bit_depth: u32) -> Self {
        let y_w = (mi_cols * 8) as usize;
        let y_h = (mi_rows * 8) as usize;
        let uv_w = y_w >> usize::from(ssx);
        let uv_h = y_h >> usize::from(ssy);
        Self {
            planes: [
                Plane::new(y_w, y_h),
                Plane::new(uv_w, uv_h),
                Plane::new(uv_w, uv_h),
            ],
            mi_cols,
            mi_rows,
            subsampling_x: ssx,
            subsampling_y: ssy,
            bit_depth,
        }
    }

    /// Predict one intra transform block on the reconstruction planes,
    /// exactly as the decoder's §6.4.21 walk does for a `MiSize >=
    /// BLOCK_8X8` block at MI `(mi_r, mi_c)` — deriving `have_left` /
    /// `have_above` / `not_on_right` from the block's position inside the
    /// MI block and the frame-level `AvailL` / `AvailU` (single tile:
    /// `AvailU = MiRow > 0`, `AvailL = MiCol > 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn predict_block(
        &mut self,
        mi_r: u32,
        mi_c: u32,
        mi_size: u8,
        plane: usize,
        tx_sz: u32,
        start_x: u32,
        start_y: u32,
        mode: PredMode,
    ) {
        let sub_x = plane > 0 && self.subsampling_x;
        let sub_y = plane > 0 && self.subsampling_y;
        let base_x = (mi_c * 8) >> u32::from(sub_x);
        let base_y = (mi_r * 8) >> u32::from(sub_y);
        let step = 1u32 << tx_sz;
        let x = (start_x - base_x) / 4;
        let y = (start_y - base_y) / 4;
        let bsize = mi_size.max(BLOCK_8X8);
        let plane_sz = get_plane_block_size(bsize, plane, self.subsampling_x, self.subsampling_y);
        let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
        let maxx = (self.mi_cols * 8) >> u32::from(sub_x);
        let maxy = (self.mi_rows * 8) >> u32::from(sub_y);

        predict_intra(
            &mut self.planes[plane],
            start_x as usize,
            start_y as usize,
            mi_c > 0 || x > 0,
            mi_r > 0 || y > 0,
            x + step < num4x4w,
            tx_sz,
            mode,
            (maxx - 1) as usize,
            (maxy - 1) as usize,
            self.bit_depth,
        );
    }
}

/// Encode a lossless keyframe whose reconstruction equals `targets`
/// (three MI-padded planes) byte-exactly.
///
/// `hdr` must be a lossless keyframe header (see
/// [`lossless_keyframe_header`]); `targets` must be sized to the
/// MI-padded extents (`MiCols * 8 × MiRows * 8` luma, subsampled
/// chroma) with every sample in `[0, (1 << BitDepth) - 1]`.
pub(crate) fn encode_keyframe_lossless(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
) -> Result<Vec<u8>, Error> {
    if hdr.frame_type != FrameType::KeyFrame || !hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);

    // All-8x8, all-non-skip, DC_PRED plan: the residual carries the
    // content; DC_PRED keeps the §6.4.25 TxType selection on the
    // lossless WHT path for every block.
    let n = (mi_rows as usize) * (mi_cols as usize);
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
        tx_mode: crate::compressed::TxMode::Only4x4,
    };

    let mut recon = ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth);
    let seg = hdr.segmentation;
    let quant = hdr.quantization;
    let bd8 = hdr.color_config.bit_depth;

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        move |mi_r: u32, mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            // 1. Predict, exactly as the decoder will.
            recon.predict_block(mi_r, mi_c, BLOCK_8X8, plane, 0, sx, sy, PredMode::DcPred);

            // 2. Residual = target - prediction over the 4x4 block.
            let mut block = vec![0i64; 16];
            for i in 0..4usize {
                for j in 0..4usize {
                    let t = targets[plane].get(sx as usize + j, sy as usize + i);
                    let p = recon.planes[plane].get(sx as usize + j, sy as usize + i);
                    block[i * 4 + j] = i64::from(t) - i64::from(p);
                }
            }

            // 3. Exact forward WHT -> quantized tokens.
            forward_wht_2d(&mut block);

            // 4. Replay the decoder's reconstruction so the next block's
            //    prediction sees the decoder's state. Lossless quantizers
            //    are 4 / 4 (§8.6.1 at qindex 0); reconstruction lands on
            //    the target exactly.
            let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
            let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
            reconstruct_block(
                &mut recon.planes[plane],
                sx as usize,
                sy as usize,
                0,
                &block,
                dc_q,
                ac_q,
                DCT_DCT,
                true,
                recon.bit_depth,
            );
            debug_assert_eq!(
                recon.planes[plane].get(sx as usize, sy as usize),
                targets[plane].get(sx as usize, sy as usize),
                "lossless reconstruction diverged from target"
            );

            block
        },
    );

    assemble_keyframe(hdr, &plan, &mut *coeffs)
}

/// Encode an 8-bit 4:2:0 planar frame (`Y` then `U` then `V`, the
/// [`crate::decode_vp9`] output layout) into a lossless VP9 keyframe.
///
/// The returned frame decodes **byte-exact** back to `pixels` through
/// [`crate::decode_vp9`].
pub(crate) fn encode_keyframe_lossless_420(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    if pixels.len() < w * h + 2 * cw * ch {
        return Err(Error::Unsupported);
    }

    let hdr = lossless_keyframe_header(width, height);
    let mi_cols = ((width + 7) >> 3) as usize;
    let mi_rows = ((height + 7) >> 3) as usize;
    let y_w = mi_cols * 8;
    let y_h = mi_rows * 8;
    let uv_w = y_w >> 1;
    let uv_h = y_h >> 1;

    let y_plane = padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h);
    let u_plane = padded_plane_from_bytes(&pixels[w * h..w * h + cw * ch], cw, ch, uv_w, uv_h);
    let v_plane = padded_plane_from_bytes(&pixels[w * h + cw * ch..], cw, ch, uv_w, uv_h);

    encode_keyframe_lossless(&hdr, &[y_plane, u_plane, v_plane])
}

/// Encode an 8-bit **4:4:4** planar frame (`Y` then `U` then `V`, each
/// `width × height`) into a lossless profile-1 VP9 keyframe that decodes
/// byte-exact back to `pixels`.
pub(crate) fn encode_keyframe_lossless_444(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    let w = width as usize;
    let h = height as usize;
    if pixels.len() < 3 * w * h {
        return Err(Error::Unsupported);
    }

    let hdr = lossless_keyframe_header_ex(width, height, 1, 8, false, false);
    let mi_cols = ((width + 7) >> 3) as usize;
    let mi_rows = ((height + 7) >> 3) as usize;
    let y_w = mi_cols * 8;
    let y_h = mi_rows * 8;

    let y_plane = padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h);
    let u_plane = padded_plane_from_bytes(&pixels[w * h..2 * w * h], w, h, y_w, y_h);
    let v_plane = padded_plane_from_bytes(&pixels[2 * w * h..], w, h, y_w, y_h);

    encode_keyframe_lossless(&hdr, &[y_plane, u_plane, v_plane])
}

/// Encode a 10/12-bit planar frame (native `u16` samples, `Y` then `U`
/// then `V`) into a lossless high-bit-depth VP9 keyframe that decodes
/// sample-exact back to `samples`.
///
/// `subsample == true` selects 4:2:0 (profile 2, chroma planes
/// `ceil(w/2) × ceil(h/2)`); `false` selects 4:4:4 (profile 3, chroma
/// planes `width × height`). `bit_depth` must be 10 or 12 and every
/// sample must fit in `[0, (1 << bit_depth) - 1]`.
pub(crate) fn encode_keyframe_lossless_hbd(
    samples: &[u16],
    width: u32,
    height: u32,
    bit_depth: u8,
    subsample: bool,
) -> Result<Vec<u8>, Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    if bit_depth != 10 && bit_depth != 12 {
        return Err(Error::Unsupported);
    }
    let w = width as usize;
    let h = height as usize;
    let (cw, ch) = if subsample {
        (width.div_ceil(2) as usize, height.div_ceil(2) as usize)
    } else {
        (w, h)
    };
    if samples.len() < w * h + 2 * cw * ch {
        return Err(Error::Unsupported);
    }
    let max = (1u16 << bit_depth) - 1;
    if samples.iter().any(|&s| s > max) {
        return Err(Error::Unsupported);
    }

    let profile = if subsample { 2 } else { 3 };
    let hdr = lossless_keyframe_header_ex(width, height, profile, bit_depth, subsample, subsample);
    let mi_cols = ((width + 7) >> 3) as usize;
    let mi_rows = ((height + 7) >> 3) as usize;
    let y_w = mi_cols * 8;
    let y_h = mi_rows * 8;
    let (uv_w, uv_h) = if subsample {
        (y_w >> 1, y_h >> 1)
    } else {
        (y_w, y_h)
    };

    let y_plane = padded_plane_from_u16(&samples[..w * h], w, h, y_w, y_h);
    let u_plane = padded_plane_from_u16(&samples[w * h..w * h + cw * ch], cw, ch, uv_w, uv_h);
    let v_plane = padded_plane_from_u16(&samples[w * h + cw * ch..], cw, ch, uv_w, uv_h);

    encode_keyframe_lossless(&hdr, &[y_plane, u_plane, v_plane])
}

// ----- Lossy keyframe encoding -----

/// Encode a **lossy** keyframe at `base_q_idx` whose reconstruction is
/// exactly the decoder's: per coded 4x4 block the encoder predicts with
/// the decoder's §8.5.1 process over its reconstruction planes,
/// forward-DCT-transforms the `target − prediction` residual, quantizes
/// it with the §8.6.1 quantizers, then replays the decoder's §8.6.2
/// dequant + integer inverse transform + `Clip1` reconstruction — so the
/// encoder's in-loop reference state and the decoder's output are
/// bit-identical, and only the (bounded) quantization error separates
/// the reconstruction from the source.
///
/// Returns the coded frame plus the encoder's reconstruction state (the
/// decoder's exact output at the MI-padded extents) so callers and tests
/// can pin the mirror.
pub(crate) fn encode_keyframe_lossy(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
) -> Result<(Vec<u8>, ReconState), Error> {
    if hdr.frame_type != FrameType::KeyFrame || hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);

    let n = (mi_rows as usize) * (mi_cols as usize);
    let plan = KeyframePlan {
        plans: vec![
            BlockPlan {
                y_mode: 0, // DC_PRED -> §6.4.25 DCT_DCT on the luma path.
                uv_mode: 0,
                skip: false,
                segment_id: 0,
            };
            n
        ],
        tx_mode: crate::compressed::TxMode::Only4x4,
    };

    let mut recon = ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth);
    let seg = hdr.segmentation;
    let quant = hdr.quantization;
    let bd8 = hdr.color_config.bit_depth;

    let bytes = {
        let recon_ref = &mut recon;
        let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
            move |mi_r: u32, mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
                recon_ref.predict_block(mi_r, mi_c, BLOCK_8X8, plane, 0, sx, sy, PredMode::DcPred);

                let mut block = vec![0i64; 16];
                for i in 0..4usize {
                    for j in 0..4usize {
                        let t = targets[plane].get(sx as usize + j, sy as usize + i);
                        let p = recon_ref.planes[plane].get(sx as usize + j, sy as usize + i);
                        block[i * 4 + j] = i64::from(t) - i64::from(p);
                    }
                }

                let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
                let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
                crate::fwd_transform::forward_dct_2d(&mut block, 2);
                crate::fwd_transform::quantize_block(&mut block, dc_q, ac_q);

                // Replay the decoder's reconstruction (integer inverse,
                // not the float forward) to keep encoder state exact.
                reconstruct_block(
                    &mut recon_ref.planes[plane],
                    sx as usize,
                    sy as usize,
                    0,
                    &block,
                    dc_q,
                    ac_q,
                    DCT_DCT,
                    false,
                    bit_depth,
                );

                block
            },
        );
        assemble_keyframe(hdr, &plan, &mut *coeffs)?
    };

    Ok((bytes, recon))
}

/// Encode an 8-bit 4:2:0 planar frame into a **lossy** VP9 keyframe at
/// quantizer index `base_q_idx` (`1..=255`; use [`crate::encode_vp9`]
/// for lossless). The decoder's output equals the encoder's in-loop
/// reconstruction bit-for-bit; distortion against the source is bounded
/// by the §8.6.1 quantizer step.
pub(crate) fn encode_keyframe_lossy_420(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    if base_q_idx == 0 {
        return Err(Error::Unsupported); // qindex 0 is the lossless path.
    }
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    if pixels.len() < w * h + 2 * cw * ch {
        return Err(Error::Unsupported);
    }

    let mut hdr = lossless_keyframe_header(width, height);
    hdr.quantization = QuantizationParams {
        base_q_idx,
        delta_q_y_dc: 0,
        delta_q_uv_dc: 0,
        delta_q_uv_ac: 0,
        lossless: false,
    };

    let mi_cols = ((width + 7) >> 3) as usize;
    let mi_rows = ((height + 7) >> 3) as usize;
    let y_w = mi_cols * 8;
    let y_h = mi_rows * 8;
    let uv_w = y_w >> 1;
    let uv_h = y_h >> 1;

    let y_plane = padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h);
    let u_plane = padded_plane_from_bytes(&pixels[w * h..w * h + cw * ch], cw, ch, uv_w, uv_h);
    let v_plane = padded_plane_from_bytes(&pixels[w * h + cw * ch..], cw, ch, uv_w, uv_h);

    encode_keyframe_lossy(&hdr, &[y_plane, u_plane, v_plane]).map(|(bytes, _)| bytes)
}

// ----- Lossless inter (P-frame) encoding -----

/// The §6.2 header for a lossless ZEROMV P-frame: profile 0, 8-bit
/// 4:2:0, `LAST` / `GOLDEN` / `ALTREF` all resolving to slot 0,
/// `refresh_frame_flags == 0x01` so each frame becomes the next frame's
/// `LAST` reference, EIGHTTAP filter, loop filter off, lossless
/// quantization.
pub(crate) fn lossless_pframe_header(width: u32, height: u32) -> Vp9FrameHeader {
    let mut hdr = lossless_keyframe_header(width, height);
    hdr.frame_type = FrameType::NonKeyFrame;
    hdr.refresh_frame_flags = 0x01;
    hdr.ref_frame_idx = Some([0, 0, 0]);
    hdr
}

/// Run the §8.5.2 inter prediction for every all-`BLOCK_8X8` `ZEROMV`
/// MI block of a P-frame, writing the predicted planes (at the MI-padded
/// working extents) — exactly what the decoder's §6.4.21 inter arm
/// produces before the token loop.
///
/// `reference` carries the previous frame's **visible-extent** planes
/// (the §8.10 `FrameStore` crop): `(samples, stride)` per plane, with
/// `vis_w × vis_h` the reference's luma dimensions.
// Spec-shaped geometry fan-in, matching the style of the §8.5.2 driver.
#[allow(clippy::too_many_arguments)]
fn predict_frame_zeromv(
    reference: &[(&[i32], usize); 3],
    vis_w: u32,
    vis_h: u32,
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) -> [Plane; 3] {
    let y_w = (mi_cols * 8) as usize;
    let y_h = (mi_rows * 8) as usize;
    let uv_w = y_w >> usize::from(ssx);
    let uv_h = y_h >> usize::from(ssy);
    let mut pred = [
        Plane::new(y_w, y_h),
        Plane::new(uv_w, uv_h),
        Plane::new(uv_w, uv_h),
    ];

    let zero_mvs = [[[0i32; 2]; 4]; 2];
    for r in 0..mi_rows {
        for c in 0..mi_cols {
            for (plane, pred_plane) in pred.iter_mut().enumerate() {
                let sub_x = plane > 0 && ssx;
                let sub_y = plane > 0 && ssy;
                let base_x = (c * 8) >> u32::from(sub_x);
                let base_y = (r * 8) >> u32::from(sub_y);
                let region = 8usize >> usize::from(sub_x);
                let region_h = 8usize >> usize::from(sub_y);

                let (samples, stride) = reference[plane];
                let refs = RefPlanes {
                    list: [
                        Some(RefPlane {
                            samples,
                            stride,
                            ref_frame_width: vis_w as i32,
                            ref_frame_height: vis_h as i32,
                        }),
                        None,
                    ],
                };
                let grid = BlockGrid {
                    mi_row: r as i32,
                    mi_col: c as i32,
                    mi_rows: mi_rows as i32,
                    mi_cols: mi_cols as i32,
                    mi_size: BLOCK_8X8,
                };
                let geom = ScaleGeom {
                    ref_frame_width: vis_w as i32,
                    ref_frame_height: vis_h as i32,
                    frame_width: vis_w as i32,
                    frame_height: vis_h as i32,
                    subsampling_x: ssx,
                    subsampling_y: ssy,
                };
                let args = InterPredArgs {
                    plane,
                    x: base_x as i32,
                    y: base_y as i32,
                    w: region,
                    h: region_h,
                    block_idx: 0,
                    interp_filter: 0, // EIGHTTAP; ZEROMV full-pel is a copy.
                    bit_depth,
                    is_compound: false,
                };
                predict_inter(pred_plane, &args, &grid, &geom, &zero_mvs, &refs, ssx, ssy);
            }
        }
    }
    pred
}

/// Encode one lossless `ZEROMV` P-frame whose reconstruction equals
/// `targets` (MI-padded planes) exactly, referencing `reference` (the
/// previous frame's visible-extent planes).
///
/// Per coded transform block the §6.4.21 walk supplies the coefficient
/// callback; the residual is `target − prediction` with the prediction
/// computed by the decoder's own §8.5.2 process, forward-WHT-transformed
/// exactly. With zero motion the reconstruction is `Clip1( prediction +
/// residual ) == target` sample-for-sample, so the frame chain stays
/// bit-exact and the next frame may reference `targets`' visible crop.
pub(crate) fn encode_pframe_lossless_zeromv(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
) -> Result<Vec<u8>, Error> {
    if hdr.frame_type != FrameType::NonKeyFrame || !hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);

    let pred = predict_frame_zeromv(
        reference, ref_w, ref_h, mi_cols, mi_rows, ssx, ssy, bit_depth,
    );

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        move |_mi_r: u32, _mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            let mut block = vec![0i64; 16];
            for i in 0..4usize {
                for j in 0..4usize {
                    let t = targets[plane].get(sx as usize + j, sy as usize + i);
                    let p = pred[plane].get(sx as usize + j, sy as usize + i);
                    block[i * 4 + j] = i64::from(t) - i64::from(p);
                }
            }
            forward_wht_2d(&mut block);
            block
        },
    );

    assemble_inter_frame_zeromv(hdr, false, &mut *coeffs)
}

/// Encode a sequence of 8-bit 4:2:0 planar frames (each `Y` then `U`
/// then `V`, the [`crate::decode_vp9`] layout) into a lossless VP9
/// stream: a keyframe followed by `ZEROMV` P-frames, each coding the
/// exact `frame − prediction` residual.
///
/// Every returned coded frame decodes **byte-exact** back to its input
/// through [`crate::decode_frame::decode_vp9_sequence`].
pub(crate) fn encode_sequence_lossless_420(
    frames: &[&[u8]],
    width: u32,
    height: u32,
) -> Result<Vec<Vec<u8>>, Error> {
    if frames.is_empty() {
        return Err(Error::Unsupported);
    }
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let need = w * h + 2 * cw * ch;
    if frames.iter().any(|f| f.len() < need) {
        return Err(Error::Unsupported);
    }

    let mi_cols = ((width + 7) >> 3) as usize;
    let mi_rows = ((height + 7) >> 3) as usize;
    let y_w = mi_cols * 8;
    let y_h = mi_rows * 8;
    let uv_w = y_w >> 1;
    let uv_h = y_h >> 1;

    let padded_targets = |pixels: &[u8]| -> [Plane; 3] {
        [
            padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h),
            padded_plane_from_bytes(&pixels[w * h..w * h + cw * ch], cw, ch, uv_w, uv_h),
            padded_plane_from_bytes(&pixels[w * h + cw * ch..], cw, ch, uv_w, uv_h),
        ]
    };
    // Visible-extent reference planes (the §8.10 FrameStore crop) for
    // the next frame: the previous frame's source samples, since a
    // lossless frame reconstructs to its target exactly.
    let visible_ref = |pixels: &[u8]| -> [Vec<i32>; 3] {
        [
            pixels[..w * h].iter().map(|&s| i32::from(s)).collect(),
            pixels[w * h..w * h + cw * ch]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            pixels[w * h + cw * ch..w * h + 2 * cw * ch]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
        ]
    };

    let mut out = Vec::with_capacity(frames.len());
    out.push(encode_keyframe_lossless_420(frames[0], width, height)?);

    for i in 1..frames.len() {
        let targets = padded_targets(frames[i]);
        let prev = visible_ref(frames[i - 1]);
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), w),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        let hdr = lossless_pframe_header(width, height);
        out.push(encode_pframe_lossless_zeromv(
            &hdr, &targets, &reference, width, height,
        )?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_frame::decode_intra_frame;

    /// Deterministic pseudo-random planar 4:2:0 frame.
    fn noise_frame(width: u32, height: u32, seed: u64) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        (0..w * h + 2 * cw * ch).map(|_| next()).collect()
    }

    fn assert_lossless_roundtrip(width: u32, height: u32, pixels: &[u8]) {
        let stream = encode_keyframe_lossless_420(pixels, width, height).expect("encode");
        let frame = decode_intra_frame(&stream).expect("decode");
        assert_eq!((frame.width, frame.height), (width, height));
        let out = frame.to_planar_bytes();
        assert_eq!(out.len(), pixels.len(), "planar size mismatch");
        assert_eq!(out, pixels, "lossless round-trip not byte-exact");
    }

    /// Pseudo-random noise — the hardest content — round-trips
    /// byte-exact through the full decoder.
    #[test]
    fn noise_64x64_roundtrips_byte_exact() {
        let px = noise_frame(64, 64, 0xBEEF_CAFE_1234_5678);
        assert_lossless_roundtrip(64, 64, &px);
    }

    /// A non-multiple-of-8 frame (frame-edge partition splits + padded
    /// target replication) round-trips byte-exact.
    #[test]
    fn noise_40x24_roundtrips_byte_exact() {
        let px = noise_frame(40, 24, 0x1122_3344_5566_7788);
        assert_lossless_roundtrip(40, 24, &px);
    }

    /// A multi-superblock frame (2x2 SB grid, neighbour threading across
    /// SB boundaries) round-trips byte-exact.
    #[test]
    fn noise_128x96_roundtrips_byte_exact() {
        let px = noise_frame(128, 96, 0x0F0F_F0F0_A5A5_5A5A);
        assert_lossless_roundtrip(128, 96, &px);
    }

    /// Extreme-value content (0 / 255 checkerboard) hits the residual
    /// range extremes and the CAT6 token path.
    #[test]
    fn checkerboard_extremes_roundtrip_byte_exact() {
        let (w, h) = (48u32, 32u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
        for i in 0..h as usize {
            for j in 0..w as usize {
                px.push(if (i + j) % 2 == 0 { 0 } else { 255 });
            }
        }
        for i in 0..ch {
            for j in 0..cw {
                px.push(if (i + j) % 2 == 0 { 255 } else { 0 });
            }
        }
        for i in 0..ch {
            for j in 0..cw {
                px.push(if (i * j) % 3 == 0 { 7 } else { 250 });
            }
        }
        assert_lossless_roundtrip(w, h, &px);
    }

    /// A smooth gradient (small residuals after DC prediction) also
    /// round-trips — the opposite coding regime from noise.
    #[test]
    fn gradient_roundtrips_byte_exact() {
        let (w, h) = (64u32, 48u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let mut px = Vec::new();
        for i in 0..h as usize {
            for j in 0..w as usize {
                px.push(((i * 2 + j * 3) % 256) as u8);
            }
        }
        for i in 0..ch {
            for j in 0..cw {
                px.push(((128 + i + j) % 256) as u8);
            }
        }
        for i in 0..ch {
            for j in 0..cw {
                px.push(((64 + i * 2 + j) % 256) as u8);
            }
        }
        assert_lossless_roundtrip(w, h, &px);
    }

    /// Degenerate geometries (1x1, 3x5, 8x1) round-trip.
    #[test]
    fn degenerate_geometries_roundtrip_byte_exact() {
        for &(w, h) in &[(1u32, 1u32), (3, 5), (8, 1), (1, 16)] {
            let px = noise_frame(w, h, u64::from(w) * 977 + u64::from(h));
            assert_lossless_roundtrip(w, h, &px);
        }
    }

    /// The encoder is byte-deterministic.
    #[test]
    fn lossless_encode_is_deterministic() {
        let px = noise_frame(32, 32, 42);
        let a = encode_keyframe_lossless_420(&px, 32, 32).expect("a");
        let b = encode_keyframe_lossless_420(&px, 32, 32).expect("b");
        assert_eq!(a, b);
    }

    /// Short input / degenerate dimensions are rejected.
    #[test]
    fn lossless_encode_rejects_bad_inputs() {
        assert_eq!(
            encode_keyframe_lossless_420(&[0u8; 8], 64, 64).unwrap_err(),
            Error::Unsupported
        );
        assert_eq!(
            encode_keyframe_lossless_420(&[0u8; 8], 0, 4).unwrap_err(),
            Error::Unsupported
        );
    }

    /// The lossy encoder's in-loop reconstruction equals the decoder's
    /// output **bit-for-bit** — the strong decoder-mirror pin: whatever
    /// quantization discards, encoder and decoder agree exactly on what
    /// was kept.
    #[test]
    fn lossy_decode_equals_encoder_recon_exactly() {
        let (w, h) = (48u32, 40u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        let px: Vec<u8> = (0..n).map(|i| ((i * 73 + 19) % 256) as u8).collect();

        let mut hdr = lossless_keyframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: 80,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let y_w = (((w + 7) >> 3) * 8) as usize;
        let y_h = (((h + 7) >> 3) * 8) as usize;
        let targets = [
            padded_plane_from_bytes(&px[..(w * h) as usize], w as usize, h as usize, y_w, y_h),
            padded_plane_from_bytes(
                &px[(w * h) as usize..(w * h) as usize + cw * ch],
                cw,
                ch,
                y_w >> 1,
                y_h >> 1,
            ),
            padded_plane_from_bytes(
                &px[(w * h) as usize + cw * ch..],
                cw,
                ch,
                y_w >> 1,
                y_h >> 1,
            ),
        ];
        let (bytes, recon) = encode_keyframe_lossy(&hdr, &targets).expect("encode");
        let frame = decode_intra_frame(&bytes).expect("decode");

        // Visible-region comparison, all three planes.
        for row in 0..h as usize {
            for col in 0..w as usize {
                assert_eq!(
                    i32::from(frame.y[row * w as usize + col]),
                    recon.planes[0].get(col, row),
                    "luma ({col},{row})"
                );
            }
        }
        for row in 0..ch {
            for col in 0..cw {
                assert_eq!(
                    i32::from(frame.u[row * cw + col]),
                    recon.planes[1].get(col, row),
                    "U ({col},{row})"
                );
                assert_eq!(
                    i32::from(frame.v[row * cw + col]),
                    recon.planes[2].get(col, row),
                    "V ({col},{row})"
                );
            }
        }
    }

    /// At `base_q_idx == 1` (quantizer 8/8) the lossy path is
    /// near-lossless: every reconstructed sample is within a small bound
    /// of the source.
    #[test]
    fn lossy_q1_is_near_lossless() {
        let (w, h) = (32u32, 32u32);
        let px = noise_frame(w, h, 0xC0FF_EE00_1234_5678);
        let bytes = encode_keyframe_lossy_420(&px, w, h, 1).expect("encode");
        let frame = decode_intra_frame(&bytes).expect("decode");
        let out = frame.to_planar_bytes();
        let max_err = out
            .iter()
            .zip(&px)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap();
        assert!(max_err <= 12, "q=1 max sample error {max_err} > 12");
    }

    /// Distortion shrinks and the stream grows as `base_q_idx`
    /// decreases; the decode also stays deterministic.
    #[test]
    fn lossy_distortion_and_size_scale_with_q() {
        let (w, h) = (64u32, 48u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        // Textured but not pure noise (some spatial correlation).
        let px: Vec<u8> = (0..n)
            .map(|i| {
                let x = i % w as usize;
                let y = i / w as usize;
                (((x * 3 + y * 5) % 97) + ((x * y) % 41) + 60) as u8
            })
            .collect();

        let mse = |q: u8| -> (f64, usize) {
            let bytes = encode_keyframe_lossy_420(&px, w, h, q).expect("encode");
            let size = bytes.len();
            let out = decode_intra_frame(&bytes)
                .expect("decode")
                .to_planar_bytes();
            let sum: f64 = out
                .iter()
                .zip(&px)
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum();
            (sum / out.len() as f64, size)
        };

        let (mse_lo_q, size_lo_q) = mse(40);
        let (mse_hi_q, size_hi_q) = mse(200);
        assert!(
            mse_lo_q <= mse_hi_q,
            "MSE must not grow as q drops: q40 {mse_lo_q} vs q200 {mse_hi_q}"
        );
        assert!(
            size_hi_q <= size_lo_q,
            "stream must not grow as q rises: q200 {size_hi_q} B vs q40 {size_lo_q} B"
        );
        assert!(mse_lo_q < 100.0, "q40 MSE {mse_lo_q} unexpectedly large");
    }

    /// The lossy entry rejects the lossless qindex and bad geometry.
    #[test]
    fn lossy_encode_rejects_bad_inputs() {
        let px = noise_frame(16, 16, 3);
        assert_eq!(
            encode_keyframe_lossy_420(&px, 16, 16, 0).unwrap_err(),
            Error::Unsupported
        );
        assert_eq!(
            encode_keyframe_lossy_420(&px[..4], 16, 16, 50).unwrap_err(),
            Error::Unsupported
        );
        assert_eq!(
            encode_keyframe_lossy_420(&px, 0, 16, 50).unwrap_err(),
            Error::Unsupported
        );
    }

    /// A three-frame moving-content sequence (keyframe + two ZEROMV
    /// P-frames with real coded residuals) reconstructs **byte-exact**
    /// frame-for-frame through `decode_vp9_sequence` — the lossless
    /// inter path end-to-end, §8.5.2 motion compensation and §8.10
    /// reference threading included.
    #[test]
    fn moving_sequence_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (48u32, 32u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        // A diagonal gradient translating by (t, t) per frame plus a
        // moving bright square: every frame differs from its predecessor.
        let make_frame = |t: usize| -> Vec<u8> {
            let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
            for i in 0..h as usize {
                for j in 0..w as usize {
                    let mut v = ((i + j + 5 * t) % 256) as u8;
                    let (sq_x, sq_y) = (4 + 6 * t, 8 + 2 * t);
                    if (sq_y..sq_y + 8).contains(&i) && (sq_x..sq_x + 8).contains(&j) {
                        v = 240;
                    }
                    px.push(v);
                }
            }
            for i in 0..ch {
                for j in 0..cw {
                    px.push(((100 + i * 2 + j + 3 * t) % 256) as u8);
                }
            }
            for i in 0..ch {
                for j in 0..cw {
                    push_v(&mut px, i, j, t);
                }
            }
            px
        };
        fn push_v(px: &mut Vec<u8>, i: usize, j: usize, t: usize) {
            px.push(((200 + i + j * 2 + 7 * t) % 256) as u8);
        }

        let inputs: Vec<Vec<u8>> = (0..3).map(make_frame).collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        let coded = encode_sequence_lossless_420(&refs, w, h).expect("encode sequence");
        assert_eq!(coded.len(), 3);

        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode sequence");
        assert_eq!(decoded.len(), 3);
        for (i, (frame, input)) in decoded.iter().zip(&inputs).enumerate() {
            assert_eq!((frame.width, frame.height), (w, h), "frame {i} geometry");
            assert_eq!(&frame.to_planar_bytes(), input, "frame {i} not byte-exact");
        }
    }

    /// A mostly-static sequence codes its P-frames far smaller than the
    /// keyframe: unchanged blocks carry all-zero residual syntax.
    #[test]
    fn static_sequence_pframes_are_smaller_than_keyframe() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 64u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        // Textured base frame; frame 2 changes only one 8x8 region.
        let base: Vec<u8> = (0..n).map(|i| ((i * 61 + 17) % 256) as u8).collect();
        let mut moved = base.clone();
        for i in 16..24usize {
            for j in 32..40usize {
                moved[i * w as usize + j] = 255 - moved[i * w as usize + j];
            }
        }

        let refs: [&[u8]; 2] = [&base, &moved];
        let coded = encode_sequence_lossless_420(&refs, w, h).expect("encode");
        assert!(
            coded[1].len() < coded[0].len() / 4,
            "static P-frame ({} B) should be far smaller than keyframe ({} B)",
            coded[1].len(),
            coded[0].len()
        );

        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode");
        assert_eq!(decoded[0].to_planar_bytes(), base);
        assert_eq!(decoded[1].to_planar_bytes(), moved);
    }

    /// Non-multiple-of-8 geometry through the inter path (padded-region
    /// prediction clamps against the visible reference extents).
    #[test]
    fn sequence_partial_superblock_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (36u32, 20u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        let f0: Vec<u8> = (0..n).map(|i| ((i * 41 + 3) % 256) as u8).collect();
        let f1: Vec<u8> = (0..n).map(|i| ((i * 97 + 29) % 256) as u8).collect();
        let refs: [&[u8]; 2] = [&f0, &f1];
        let coded = encode_sequence_lossless_420(&refs, w, h).expect("encode");
        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode");
        assert_eq!(decoded[0].to_planar_bytes(), f0, "frame 0");
        assert_eq!(decoded[1].to_planar_bytes(), f1, "frame 1");
    }

    /// A longer chain (keyframe + 4 P-frames) keeps the reference
    /// threading exact across every hop.
    #[test]
    fn five_frame_chain_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (32u32, 24u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        let inputs: Vec<Vec<u8>> = (0..5u64)
            .map(|t| {
                (0..n)
                    .map(|i| (((i as u64) * 13 + t * 37 + 5) % 256) as u8)
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        let coded = encode_sequence_lossless_420(&refs, w, h).expect("encode");
        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode");
        assert_eq!(decoded.len(), 5);
        for (i, (frame, input)) in decoded.iter().zip(&inputs).enumerate() {
            assert_eq!(&frame.to_planar_bytes(), input, "frame {i}");
        }
    }

    /// Sequence rejections: empty input, short frame buffer.
    #[test]
    fn sequence_encode_rejects_bad_inputs() {
        assert_eq!(
            encode_sequence_lossless_420(&[], 16, 16).unwrap_err(),
            Error::Unsupported
        );
        let ok = vec![0u8; 16 * 16 + 2 * 64];
        let short = vec![0u8; 10];
        assert_eq!(
            encode_sequence_lossless_420(&[&ok, &short], 16, 16).unwrap_err(),
            Error::Unsupported
        );
    }

    /// An 8-bit 4:4:4 (profile 1) frame round-trips byte-exact — the
    /// chroma planes carry full-resolution content.
    #[test]
    fn noise_444_roundtrips_sample_exact() {
        let (w, h) = (40u32, 32u32);
        let n = (w * h) as usize;
        let mut state: u64 = 0x1357_9BDF_2468_ACE0;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        let px: Vec<u8> = (0..3 * n).map(|_| next()).collect();
        let stream = encode_keyframe_lossless_444(&px, w, h).expect("encode");
        let frame = decode_intra_frame(&stream).expect("decode");
        assert_eq!((frame.width, frame.height), (w, h));
        assert!(!frame.subsampling_x && !frame.subsampling_y, "not 4:4:4");
        let y_ok = frame
            .y
            .iter()
            .zip(&px[..n])
            .all(|(&d, &s)| d == u16::from(s));
        let u_ok = frame
            .u
            .iter()
            .zip(&px[n..2 * n])
            .all(|(&d, &s)| d == u16::from(s));
        let v_ok = frame
            .v
            .iter()
            .zip(&px[2 * n..])
            .all(|(&d, &s)| d == u16::from(s));
        assert!(y_ok && u_ok && v_ok, "4:4:4 round-trip not sample-exact");
    }

    /// 10-bit and 12-bit 4:2:0 (profile 2) frames round-trip
    /// sample-exact — the residual range exceeds 8-bit and the token
    /// writer's high-bit CAT6 path carries it.
    #[test]
    fn noise_hbd_420_roundtrips_sample_exact() {
        for &bd in &[10u8, 12u8] {
            let (w, h) = (24u32, 16u32);
            let cw = w.div_ceil(2) as usize;
            let ch = h.div_ceil(2) as usize;
            let n = (w * h) as usize + 2 * cw * ch;
            let max = (1u32 << bd) - 1;
            let mut state: u64 = 0xFEED_D0D0_0BAD_F00D ^ u64::from(bd);
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as u32 % (max + 1)) as u16
            };
            let samples: Vec<u16> = (0..n).map(|_| next()).collect();
            let stream =
                encode_keyframe_lossless_hbd(&samples, w, h, bd, true).expect("encode hbd");
            let frame = decode_intra_frame(&stream).expect("decode hbd");
            assert_eq!(frame.bit_depth, bd);
            let wh = (w * h) as usize;
            assert_eq!(frame.y, samples[..wh], "{bd}-bit luma");
            assert_eq!(frame.u, samples[wh..wh + cw * ch], "{bd}-bit U");
            assert_eq!(frame.v, samples[wh + cw * ch..], "{bd}-bit V");
        }
    }

    /// 10-bit 4:4:4 (profile 3) round-trips sample-exact.
    #[test]
    fn noise_hbd_444_roundtrips_sample_exact() {
        let (w, h) = (16u32, 24u32);
        let n = 3 * (w * h) as usize;
        let samples: Vec<u16> = (0..n).map(|i| ((i * 619 + 41) % 1024) as u16).collect();
        let stream = encode_keyframe_lossless_hbd(&samples, w, h, 10, false).expect("encode");
        let frame = decode_intra_frame(&stream).expect("decode");
        assert_eq!(frame.bit_depth, 10);
        assert!(!frame.subsampling_x && !frame.subsampling_y);
        let wh = (w * h) as usize;
        assert_eq!(frame.y, samples[..wh]);
        assert_eq!(frame.u, samples[wh..2 * wh]);
        assert_eq!(frame.v, samples[2 * wh..]);
    }

    /// HBD rejections: bad bit depth, out-of-range samples, short input.
    #[test]
    fn hbd_encode_rejects_bad_inputs() {
        let samples = vec![0u16; 6 * 16];
        assert_eq!(
            encode_keyframe_lossless_hbd(&samples, 8, 8, 9, true).unwrap_err(),
            Error::Unsupported
        );
        let mut over = vec![0u16; 8 * 8 + 2 * 16];
        over[0] = 1 << 10; // exceeds 10-bit range
        assert_eq!(
            encode_keyframe_lossless_hbd(&over, 8, 8, 10, true).unwrap_err(),
            Error::Unsupported
        );
        assert_eq!(
            encode_keyframe_lossless_hbd(&samples[..4], 8, 8, 10, true).unwrap_err(),
            Error::Unsupported
        );
    }

    /// Padded-plane construction edge-replicates.
    #[test]
    fn padded_plane_edge_replicates() {
        let data = [1u8, 2, 3, 4, 5, 6]; // 3x2
        let p = padded_plane_from_bytes(&data, 3, 2, 5, 4);
        assert_eq!(p.get(0, 0), 1);
        assert_eq!(p.get(2, 1), 6);
        assert_eq!(p.get(4, 0), 3); // right padding replicates col 2
        assert_eq!(p.get(1, 3), 5); // bottom padding replicates row 1
        assert_eq!(p.get(4, 3), 6); // corner
    }
}
