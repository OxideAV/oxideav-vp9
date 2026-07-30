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
use crate::frame_writer::{assemble_keyframe, BlockPlan, FrameCoefSource, KeyframePlan};
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
use crate::residual::{
    get_plane_block_size, BLOCK_8X8, NUM_4X4_BLOCKS_HIGH_LOOKUP, NUM_4X4_BLOCKS_WIDE_LOOKUP,
};
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
    // Key frames and hidden intra-only frames code the identical §6.3 /
    // §6.4 intra syntax (the assembler validates the header class); both
    // are accepted here.
    let intra = hdr.frame_type == FrameType::KeyFrame
        || (hdr.frame_type == FrameType::NonKeyFrame && hdr.intra_only);
    if !intra || !hdr.quantization.lossless {
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

/// Choose the per-MI-block intra modes for a lossy keyframe by trial
/// prediction against the target planes.
///
/// For every `y_mode` / `uv_mode` candidate (all ten §7.4.5 modes) the
/// block's 4x4 sub-blocks are predicted with the decoder's §8.5.1
/// process over a scratch copy of the target plane (so neighbour samples
/// approximate a high-quality reconstruction), the summed absolute
/// difference against the target is accumulated, and the
/// lowest-SAD mode wins. This is an encoder-side heuristic only — the
/// coded mode reaches the decoder through the §6.4.6 syntax and any
/// choice is decodable; a better choice just shrinks the residual.
// Retained as the fixed-layout baseline the tree-encoder tests compare
// against (the public path now plans adaptively).
#[allow(dead_code)]
fn select_keyframe_modes(
    targets: &[Plane; 3],
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) -> Vec<BlockPlan> {
    let mut scratch = [targets[0].clone(), targets[1].clone(), targets[2].clone()];
    let mut plans = Vec::with_capacity((mi_rows as usize) * (mi_cols as usize));

    // SAD of predicting one 4x4 block with `mode`, then restore the
    // scratch region from the target.
    let mut trial = |plane: usize,
                     mode: PredMode,
                     sx: usize,
                     sy: usize,
                     have_left: bool,
                     have_above: bool,
                     not_on_right: bool,
                     max_x: usize,
                     max_y: usize|
     -> u64 {
        predict_intra(
            &mut scratch[plane],
            sx,
            sy,
            have_left,
            have_above,
            not_on_right,
            0,
            mode,
            max_x,
            max_y,
            bit_depth,
        );
        let mut sad = 0u64;
        for i in 0..4usize {
            for j in 0..4usize {
                let d = scratch[plane].get(sx + j, sy + i) - targets[plane].get(sx + j, sy + i);
                sad += d.unsigned_abs() as u64;
                // Restore so later trials/neighbours see the target.
                scratch[plane].set(sx + j, sy + i, targets[plane].get(sx + j, sy + i));
            }
        }
        sad
    };

    let modes: Vec<PredMode> = (0..10u8).map(|m| PredMode::from_raw(m).unwrap()).collect();

    for r in 0..mi_rows {
        for c in 0..mi_cols {
            // Luma: four 4x4 sub-blocks of the 8x8 MI block.
            let maxx = (mi_cols * 8) as usize - 1;
            let maxy = (mi_rows * 8) as usize - 1;
            let mut best_y = (u64::MAX, 0u8);
            for (m_raw, &mode) in modes.iter().enumerate() {
                let mut sad = 0u64;
                for y in 0..2u32 {
                    for x in 0..2u32 {
                        sad += trial(
                            0,
                            mode,
                            (c * 8 + 4 * x) as usize,
                            (r * 8 + 4 * y) as usize,
                            c > 0 || x > 0,
                            r > 0 || y > 0,
                            x == 0,
                            maxx,
                            maxy,
                        );
                    }
                }
                if sad < best_y.0 {
                    best_y = (sad, m_raw as u8);
                }
            }

            // Chroma: the (subsampled) plane blocks of the MI block,
            // U and V SAD summed per candidate.
            let plane_sz = get_plane_block_size(BLOCK_8X8, 1, ssx, ssy);
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
            let num4x4h = crate::residual::NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];
            let mut best_uv = (u64::MAX, 0u8);
            for (m_raw, &mode) in modes.iter().enumerate() {
                let mut sad = 0u64;
                for plane in 1..3usize {
                    let sub_x = ssx;
                    let sub_y = ssy;
                    let base_x = ((c * 8) >> u32::from(sub_x)) as usize;
                    let base_y = ((r * 8) >> u32::from(sub_y)) as usize;
                    let maxx_c = ((mi_cols * 8) >> u32::from(sub_x)) as usize - 1;
                    let maxy_c = ((mi_rows * 8) >> u32::from(sub_y)) as usize - 1;
                    for y in 0..num4x4h {
                        for x in 0..num4x4w {
                            sad += trial(
                                plane,
                                mode,
                                base_x + 4 * x as usize,
                                base_y + 4 * y as usize,
                                c > 0 || x > 0,
                                r > 0 || y > 0,
                                x + 1 < num4x4w,
                                maxx_c,
                                maxy_c,
                            );
                        }
                    }
                }
                if sad < best_uv.0 {
                    best_uv = (sad, m_raw as u8);
                }
            }

            plans.push(BlockPlan {
                y_mode: best_y.1,
                uv_mode: best_uv.1,
                skip: false,
                segment_id: 0,
            });
        }
    }
    plans
}

/// Encode a **lossy** keyframe at `base_q_idx` whose reconstruction is
/// exactly the decoder's: per coded 4x4 block the encoder predicts with
/// the decoder's §8.5.1 process over its reconstruction planes (with the
/// per-block intra mode chosen by [`select_keyframe_modes`], or `DC_PRED`
/// everywhere when `select_modes == false`), forward-transforms the
/// `target − prediction` residual with the §6.4.25 `TxType` the decoder
/// will derive for that mode, quantizes it with the §8.6.1 quantizers,
/// then replays the decoder's §8.6.2 reconstruction (dequant, integer
/// inverse transform, `Clip1`) — so the encoder's in-loop reference state
/// and the decoder's output are bit-identical, and only the (bounded)
/// quantization error separates the reconstruction from the source.
///
/// Returns the coded frame plus the encoder's reconstruction state (the
/// decoder's exact output at the MI-padded extents) so callers and tests
/// can pin the mirror.
// Retained as the fixed all-BLOCK_8X8 / TX_4X4 baseline engine the
// tree-encoder tests compare compression against.
#[allow(dead_code)]
pub(crate) fn encode_keyframe_lossy(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    select_modes: bool,
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
    let plans = if select_modes {
        select_keyframe_modes(targets, mi_cols, mi_rows, ssx, ssy, bit_depth)
    } else {
        vec![
            BlockPlan {
                y_mode: 0, // DC_PRED -> §6.4.25 DCT_DCT on the luma path.
                uv_mode: 0,
                skip: false,
                segment_id: 0,
            };
            n
        ]
    };
    let plan = KeyframePlan {
        plans: plans.clone(),
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
                let bp = plans[(mi_r as usize) * (mi_cols as usize) + mi_c as usize];
                let mode_raw = if plane == 0 { bp.y_mode } else { bp.uv_mode };
                let mode = PredMode::from_raw(mode_raw).expect("plan mode in range");
                // §6.4.25 TxType exactly as the decoder derives it for a
                // non-lossless 4x4 intra block: chroma forces DCT_DCT,
                // luma follows mode2txfm_map[ y_mode ].
                let tx_type = if plane > 0 {
                    DCT_DCT
                } else {
                    crate::reconstruct::tx_type_for_intra(mode)
                };

                recon_ref.predict_block(mi_r, mi_c, BLOCK_8X8, plane, 0, sx, sy, mode);

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
                crate::fwd_transform::forward_transform_2d(&mut block, 2, tx_type);
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
                    tx_type,
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
///
/// The partition / transform layout is content-adaptive
/// ([`plan_keyframe_tree`]): smooth regions code large blocks up to
/// `BLOCK_64X64` at `TX_32X32`, detailed regions split toward
/// `BLOCK_8X8`, and every leaf picks its intra modes by trial
/// prediction.
pub(crate) fn encode_keyframe_lossy_420(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    // Level-0 encode, then the §8.8 filter-params election (round
    // 420): a standalone keyframe's decoded output is its filtered
    // reconstruction, so the election lifts display quality at zero
    // rate cost exactly as in the sequence encoders. (The `_with_recon`
    // variants stay level-0: they are the sequence/fixture primitives
    // whose callers run their own election.)
    let (bytes0, recon0, state0) =
        encode_keyframe_lossy_420_with_recon_state(pixels, width, height, base_q_idx)?;
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let y_w = w.div_ceil(8) * 8;
    let y_h = h.div_ceil(8) * 8;
    let targets = [
        padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h),
        padded_plane_from_bytes(&pixels[w * h..w * h + cw * ch], cw, ch, y_w >> 1, y_h >> 1),
        padded_plane_from_bytes(&pixels[w * h + cw * ch..], cw, ch, y_w >> 1, y_h >> 1),
    ];
    let hdr = lossy_keyframe_header_420(width, height, base_q_idx);
    let (bytes, _recon) =
        finish_frame_with_filter(&hdr, bytes0, recon0, state0, &targets, w, h, |hdr2| {
            let plan = plan_keyframe_tree(
                &targets,
                (height + 7) >> 3,
                (width + 7) >> 3,
                true,
                true,
                8,
                base_q_idx,
            );
            encode_keyframe_lossy_tree_with_state(hdr2, &targets, &plan)
        })?;
    Ok(bytes)
}

/// The §6.2 header for a **lossy** 8-bit 4:2:0 keyframe at `base_q_idx`
/// — [`lossless_keyframe_header`] with the quantizer swapped in
/// (`loop_filter.level` stays 0; the sequence encoders overwrite it
/// with the elected level before assembly).
pub(crate) fn lossy_keyframe_header_420(width: u32, height: u32, base_q_idx: u8) -> Vp9FrameHeader {
    let mut hdr = lossless_keyframe_header(width, height);
    hdr.quantization = QuantizationParams {
        base_q_idx,
        delta_q_y_dc: 0,
        delta_q_uv_dc: 0,
        delta_q_uv_ac: 0,
        lossless: false,
    };
    hdr
}

/// [`encode_keyframe_lossy_420`] at level 0, also returning the
/// encoder's in-loop reconstruction (== the decoder's exact output)
/// for reference threading.
// Bytes+recon convenience over the `_state` variant; the non-test
// encoders all thread the state, so only tests call this.
#[allow(dead_code)]
pub(crate) fn encode_keyframe_lossy_420_with_recon(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<(Vec<u8>, ReconState), Error> {
    encode_keyframe_lossy_420_with_recon_state(pixels, width, height, base_q_idx)
        .map(|(b, r, _)| (b, r))
}

/// [`encode_keyframe_lossy_420_with_recon`] also returning the writer's
/// final §6.4.4 [`crate::decode_block::Vp9FrameState`] per-MI arrays —
/// the input the encode-side §8.8 loop-filter mirror consumes.
pub(crate) fn encode_keyframe_lossy_420_with_recon_state(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<(Vec<u8>, ReconState, crate::decode_block::Vp9FrameState), Error> {
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

    let hdr = lossy_keyframe_header_420(width, height, base_q_idx);

    let mi_cols = ((width + 7) >> 3) as usize;
    let mi_rows = ((height + 7) >> 3) as usize;
    let y_w = mi_cols * 8;
    let y_h = mi_rows * 8;
    let uv_w = y_w >> 1;
    let uv_h = y_h >> 1;

    let y_plane = padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h);
    let u_plane = padded_plane_from_bytes(&pixels[w * h..w * h + cw * ch], cw, ch, uv_w, uv_h);
    let v_plane = padded_plane_from_bytes(&pixels[w * h + cw * ch..], cw, ch, uv_w, uv_h);
    let targets = [y_plane, u_plane, v_plane];

    let plan = plan_keyframe_tree(
        &targets,
        mi_rows as u32,
        mi_cols as u32,
        true,
        true,
        8,
        base_q_idx,
    );
    encode_keyframe_lossy_tree_with_state(&hdr, &targets, &plan)
}

/// Plan a content-adaptive partition + transform-size tree for a lossy
/// keyframe — the superblock-tree decision layer feeding
/// [`encode_keyframe_lossy_tree`].
///
/// This is an **encoder-side heuristic only** (any conforming tree is
/// decodable; a better tree just codes fewer bits): starting at each
/// `BLOCK_64X64` root, a node is split when the luma content is
/// *prediction-inhomogeneous* at the quantizer's scale — the maximum
/// deviation of the four quadrant means from the node mean exceeds the
/// §8.6.1 AC quantizer step (structure a single leaf prediction cannot
/// track but the quantizer would preserve) — or when the block is not
/// fully contained in the MI grid (frame edge). Recursion stops at
/// `BLOCK_8X8`. Each leaf codes the largest §6.4.10-codeable transform
/// (`MAX_TXSIZE_LOOKUP[ MiSize ]`) and picks its `y_mode` / `uv_mode`
/// over all ten §7.4.5 intra modes by trial §8.5.1 prediction SAD at
/// the leaf's transform-block granularity (over a scratch copy of the
/// targets, so neighbour samples approximate a high-quality
/// reconstruction — the same approximation the 4x4 selector uses).
pub(crate) fn plan_keyframe_tree(
    targets: &[Plane; 3],
    mi_rows: u32,
    mi_cols: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    base_q_idx: u8,
) -> crate::frame_writer::KeyframeTreePlan {
    use crate::frame_writer::KeyframeTreePlan;
    use crate::residual::BLOCK_64X64;

    // The split threshold: one AC quantizer step at segment 0. Content
    // whose quadrant means differ by less than a quantizer step gains
    // nothing from a finer prediction grid.
    let quant = QuantizationParams {
        base_q_idx,
        delta_q_y_dc: 0,
        delta_q_uv_dc: 0,
        delta_q_uv_ac: 0,
        lossless: false,
    };
    let seg = SegmentationParams::default_disabled();
    let threshold = i64::from(get_ac_quant(0, &seg, &quant, 0, bit_depth as u8));

    let mut plan = KeyframeTreePlan {
        tx_mode: crate::compressed::TxMode::TxModeSelect,
        partitions: std::collections::HashMap::new(),
        leaves: std::collections::HashMap::new(),
    };
    let mut scratch = [targets[0].clone(), targets[1].clone(), targets[2].clone()];
    let ctx = PlannerCtx {
        targets,
        mi_rows,
        mi_cols,
        ssx,
        ssy,
        bit_depth,
        threshold,
    };

    for r in (0..mi_rows).step_by(8) {
        for c in (0..mi_cols).step_by(8) {
            ctx.walk(&mut plan, &mut scratch, r, c, BLOCK_64X64);
        }
    }
    plan
}

/// Frame-level inputs of the [`plan_keyframe_tree`] recursion.
struct PlannerCtx<'t> {
    targets: &'t [Plane; 3],
    mi_rows: u32,
    mi_cols: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    /// The quantizer-scaled split threshold (one §8.6.1 AC step).
    threshold: i64,
}

impl PlannerCtx<'_> {
    /// Luma mean over an MI-aligned region (in 8px MI units).
    fn mean_of(&self, r: u32, c: u32, mi_w: u32, mi_h: u32) -> i64 {
        let x0 = (c * 8) as usize;
        let y0 = (r * 8) as usize;
        let w = (mi_w * 8) as usize;
        let h = (mi_h * 8) as usize;
        let mut sum = 0i64;
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                sum += i64::from(self.targets[0].get(x, y));
            }
        }
        sum / (w as i64 * h as i64)
    }

    /// The recursive split-or-leaf decision (see [`plan_keyframe_tree`]).
    fn walk(
        &self,
        plan: &mut crate::frame_writer::KeyframeTreePlan,
        scratch: &mut [Plane; 3],
        r: u32,
        c: u32,
        bsize: u8,
    ) {
        use crate::frame_writer::TreeLeafPlan;
        use crate::partition::{
            NUM_8X8_BLOCKS_WIDE_LOOKUP, PARTITION_NONE, PARTITION_SPLIT, SUBSIZE_LOOKUP,
        };
        use crate::residual::{BLOCK_8X8 as B8, MAX_TXSIZE_LOOKUP};
        if r >= self.mi_rows || c >= self.mi_cols {
            return;
        }
        let num8x8 = NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize] as u32;
        let half = num8x8 >> 1;
        let contained = (r + num8x8) <= self.mi_rows && (c + num8x8) <= self.mi_cols;

        let mut split = !contained;
        if contained && bsize != B8 {
            // Quadrant-mean inhomogeneity at the quantizer scale.
            let m = self.mean_of(r, c, num8x8, num8x8);
            for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                let qm = self.mean_of(r + dr, c + dc, half, half);
                if (qm - m).abs() > self.threshold {
                    split = true;
                    break;
                }
            }
        }

        if !split {
            let tx_size = MAX_TXSIZE_LOOKUP[bsize as usize];
            let (y_mode, uv_mode) = select_leaf_modes(
                scratch,
                self.targets,
                r,
                c,
                bsize,
                tx_size,
                self.mi_rows,
                self.mi_cols,
                self.ssx,
                self.ssy,
                self.bit_depth,
            );
            plan.partitions.insert((r, c, bsize), PARTITION_NONE);
            plan.leaves.insert(
                (r, c),
                TreeLeafPlan {
                    mi_size: bsize,
                    tx_size,
                    y_mode,
                    uv_mode,
                    skip: false,
                    segment_id: 0,
                },
            );
            return;
        }

        plan.partitions.insert((r, c, bsize), PARTITION_SPLIT);
        let subsize = SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][bsize as usize];
        for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
            self.walk(plan, scratch, r + dr, c + dc, subsize);
        }
    }
}

/// Pick the `(y_mode, uv_mode)` for one tree leaf by trial §8.5.1
/// prediction SAD over all ten §7.4.5 intra modes at the leaf's
/// transform-block granularity — [`select_keyframe_modes`] generalised
/// past the 8x8/4x4 layout. `scratch` is a mutable copy of `targets`
/// whose leaf region each trial predicts into and restores.
#[allow(clippy::too_many_arguments)]
fn select_leaf_modes(
    scratch: &mut [Plane; 3],
    targets: &[Plane; 3],
    r: u32,
    c: u32,
    mi_size: u8,
    tx_size: u32,
    mi_rows: u32,
    mi_cols: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) -> (u8, u8) {
    use crate::residual::{get_uv_tx_size, NUM_4X4_BLOCKS_HIGH_LOOKUP, NUM_4X4_BLOCKS_WIDE_LOOKUP};

    // Trial-predict one transform block on `scratch`, SAD it against the
    // target, restore the region.
    let mut trial = |plane: usize,
                     mode: PredMode,
                     sx: usize,
                     sy: usize,
                     tx_sz: u32,
                     have_left: bool,
                     have_above: bool,
                     not_on_right: bool,
                     max_x: usize,
                     max_y: usize|
     -> u64 {
        predict_intra(
            &mut scratch[plane],
            sx,
            sy,
            have_left,
            have_above,
            not_on_right,
            tx_sz,
            mode,
            max_x,
            max_y,
            bit_depth,
        );
        let n0 = 4usize << tx_sz;
        let mut sad = 0u64;
        for i in 0..n0 {
            if sy + i > max_y {
                break;
            }
            for j in 0..n0 {
                if sx + j > max_x {
                    break;
                }
                let d = scratch[plane].get(sx + j, sy + i) - targets[plane].get(sx + j, sy + i);
                sad += d.unsigned_abs() as u64;
                scratch[plane].set(sx + j, sy + i, targets[plane].get(sx + j, sy + i));
            }
        }
        sad
    };

    let modes: Vec<PredMode> = (0..10u8).map(|m| PredMode::from_raw(m).unwrap()).collect();
    let bsize = mi_size.max(BLOCK_8X8);

    // Luma sweep at tx_size granularity.
    let plane_sz0 = get_plane_block_size(bsize, 0, ssx, ssy);
    let n4w0 = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz0 as usize];
    let n4h0 = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz0 as usize];
    let step0 = 1u32 << tx_size;
    let maxx0 = (mi_cols * 8) as usize - 1;
    let maxy0 = (mi_rows * 8) as usize - 1;
    let mut best_y = (u64::MAX, 0u8);
    for (m_raw, &mode) in modes.iter().enumerate() {
        let mut sad = 0u64;
        let mut y = 0u32;
        while y < n4h0 {
            let mut x = 0u32;
            while x < n4w0 {
                let sx = (c * 8 + 4 * x) as usize;
                let sy = (r * 8 + 4 * y) as usize;
                if sx <= maxx0 && sy <= maxy0 {
                    sad += trial(
                        0,
                        mode,
                        sx,
                        sy,
                        tx_size,
                        c > 0 || x > 0,
                        r > 0 || y > 0,
                        x + step0 < n4w0,
                        maxx0,
                        maxy0,
                    );
                }
                x += step0;
            }
            y += step0;
        }
        if sad < best_y.0 {
            best_y = (sad, m_raw as u8);
        }
    }

    // Chroma sweep at the §6.4.22 UV tx size, U + V summed.
    let uv_tx = get_uv_tx_size(tx_size, mi_size, ssx, ssy);
    let plane_sz1 = get_plane_block_size(bsize, 1, ssx, ssy);
    let n4w1 = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz1 as usize];
    let n4h1 = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz1 as usize];
    let step1 = 1u32 << uv_tx;
    let maxx1 = ((mi_cols * 8) >> u32::from(ssx)) as usize - 1;
    let maxy1 = ((mi_rows * 8) >> u32::from(ssy)) as usize - 1;
    let mut best_uv = (u64::MAX, 0u8);
    for (m_raw, &mode) in modes.iter().enumerate() {
        let mut sad = 0u64;
        for plane in 1..3usize {
            let base_x = ((c * 8) >> u32::from(ssx)) as usize;
            let base_y = ((r * 8) >> u32::from(ssy)) as usize;
            let mut y = 0u32;
            while y < n4h1 {
                let mut x = 0u32;
                while x < n4w1 {
                    let sx = base_x + (4 * x) as usize;
                    let sy = base_y + (4 * y) as usize;
                    if sx <= maxx1 && sy <= maxy1 {
                        sad += trial(
                            plane,
                            mode,
                            sx,
                            sy,
                            uv_tx,
                            c > 0 || x > 0,
                            r > 0 || y > 0,
                            x + step1 < n4w1,
                            maxx1,
                            maxy1,
                        );
                    }
                    x += step1;
                }
                y += step1;
            }
        }
        if sad < best_uv.0 {
            best_uv = (sad, m_raw as u8);
        }
    }

    (best_y.1, best_uv.1)
}

/// Encode a **lossy** keyframe over an arbitrary [`KeyframeTreePlan`] —
/// the decoder-mirror loop of [`encode_keyframe_lossy`] generalised to
/// every partition / transform size the tree elects.
///
/// Per coded transform block (any `tx_size` 0..=3, at the §6.4.21 walk
/// order of the leaf's `MiSize`) the encoder:
///
/// 1. predicts with the decoder's §8.5.1 process at the block's actual
///    transform size over its reconstruction planes,
/// 2. forward-transforms the `target − prediction` residual with the
///    §6.4.25 `TxType` the decoder will derive (chroma / `TX_32X32`
///    force `DCT_DCT`; luma follows `mode2txfm_map[ y_mode ]`, with the
///    forward ADST8 / ADST16 bases where the mode selects them),
/// 3. quantizes with the §8.6.1 quantizers under the §8.6.2 `dqDenom`
///    (2 at `TX_32X32`), and
/// 4. replays the decoder's §8.6.2 integer reconstruction — so the
///    decoder's output equals the encoder's in-loop state bit-for-bit
///    for **any** plan.
///
/// Plan leaves must be non-skip (a skip leaf reconstructs from
/// prediction the mirror never replays; the planner codes all-zero
/// blocks instead, which cost only the per-block `more_coefs` bits).
// Bytes+recon convenience over the `_with_state` encoder; the non-test
// encoders all thread the state, so only tests call this.
#[allow(dead_code)]
pub(crate) fn encode_keyframe_lossy_tree(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    plan: &crate::frame_writer::KeyframeTreePlan,
) -> Result<(Vec<u8>, ReconState), Error> {
    encode_keyframe_lossy_tree_with_state(hdr, targets, plan).map(|(b, r, _)| (b, r))
}

/// [`encode_keyframe_lossy_tree`] also returning the writer's final
/// §6.4.4 [`crate::decode_block::Vp9FrameState`] per-MI arrays — the
/// input the encode-side §8.8 loop-filter mirror consumes.
pub(crate) fn encode_keyframe_lossy_tree_with_state(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    plan: &crate::frame_writer::KeyframeTreePlan,
) -> Result<(Vec<u8>, ReconState, crate::decode_block::Vp9FrameState), Error> {
    if hdr.frame_type != FrameType::KeyFrame || hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    if plan.leaves.values().any(|lp| lp.skip) {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);

    let mut recon = ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth);
    let seg = hdr.segmentation;
    let quant = hdr.quantization;
    let bd8 = hdr.color_config.bit_depth;

    let (bytes, state) = {
        let recon_ref = &mut recon;
        let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
            move |mi_r: u32, mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
                let lp = plan
                    .leaves
                    .get(&(mi_r, mi_c))
                    .copied()
                    .expect("assembler validated the leaf exists");
                // Per-plane tx size exactly as §6.4.21 derives it.
                let tx_sz = if plane == 0 {
                    lp.tx_size
                } else {
                    crate::residual::get_uv_tx_size(lp.tx_size, lp.mi_size, ssx, ssy)
                };
                let mode_raw = if plane == 0 { lp.y_mode } else { lp.uv_mode };
                let mode = PredMode::from_raw(mode_raw).expect("plan mode in range");
                // §6.4.25 TxType for a non-lossless intra block.
                let tx_type = if plane > 0 || tx_sz == 3 {
                    DCT_DCT
                } else {
                    crate::reconstruct::tx_type_for_intra(mode)
                };

                recon_ref.predict_block(mi_r, mi_c, lp.mi_size, plane, tx_sz, sx, sy, mode);

                let n0 = 4usize << tx_sz;
                let mut block = vec![0i64; n0 * n0];
                for i in 0..n0 {
                    for j in 0..n0 {
                        let t = targets[plane].get(sx as usize + j, sy as usize + i);
                        let p = recon_ref.planes[plane].get(sx as usize + j, sy as usize + i);
                        block[i * n0 + j] = i64::from(t) - i64::from(p);
                    }
                }

                let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
                let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
                crate::fwd_transform::forward_transform_2d(&mut block, tx_sz + 2, tx_type);
                crate::fwd_transform::quantize_block_tx(&mut block, dc_q, ac_q, tx_sz, bit_depth);

                // Replay the decoder's §8.6.2 reconstruction (incl. the
                // dqDenom division) so encoder state stays exact.
                reconstruct_block(
                    &mut recon_ref.planes[plane],
                    sx as usize,
                    sy as usize,
                    tx_sz,
                    &block,
                    dc_q,
                    ac_q,
                    tx_type,
                    false,
                    bit_depth,
                );

                block
            },
        );
        crate::frame_writer::assemble_keyframe_tree_with_state(hdr, plan, &mut *coeffs)?
    };

    Ok((bytes, recon, state))
}

// ----- Lossless inter (P-frame) encoding -----

/// The §6.2 header for a lossless P-frame: profile 0, 8-bit 4:2:0,
/// `LAST` / `GOLDEN` / `ALTREF` all resolving to slot 0,
/// `refresh_frame_flags == 0x01` so each frame becomes the next frame's
/// `LAST` reference, EIGHTTAP filter, loop filter off, lossless
/// quantization.
///
/// The frame is **error-resilient**: §7.2.6 then pins
/// `UsePrevFrameMvs == 0` on the decode side, matching the inter block
/// writer's model exactly, so the §6.5 MV candidate scan (which reaches
/// the coded syntax through `BestMv` and the `inter_mode` probability
/// context) is bit-identical between encoder and decoder for every
/// mode including `NEWMV`. `allow_high_precision_mv` is enabled so
/// eighth-pel MV differences are codeable whenever the §6.5.13
/// `use_mv_hp` gate allows.
pub(crate) fn lossless_pframe_header(width: u32, height: u32) -> Vp9FrameHeader {
    let mut hdr = lossless_keyframe_header(width, height);
    hdr.frame_type = FrameType::NonKeyFrame;
    hdr.error_resilient_mode = true;
    hdr.refresh_frame_context = false;
    hdr.frame_parallel_decoding_mode = true;
    hdr.frame_context_idx = 0;
    hdr.refresh_frame_flags = 0x01;
    hdr.ref_frame_idx = Some([0, 0, 0]);
    hdr.allow_high_precision_mv = true;
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
/// Run the §8.5.2 inter prediction for one `BLOCK_8X8` MI block at
/// `(r, c)` with the (eighth-pel) motion vector `mv`, writing all three
/// planes' predicted regions — exactly what the decoder's §6.4.21 inter
/// arm produces before the token loop.
///
/// `reference` carries the previous frame's **visible-extent** planes
/// (the §8.10 `FrameStore` crop): `(samples, stride)` per plane, with
/// `vis_w × vis_h` the reference's luma dimensions.
// Spec-shaped geometry fan-in, matching the style of the §8.5.2 driver.
#[allow(clippy::too_many_arguments)]
fn predict_mi_block(
    pred: &mut [Plane; 3],
    reference: &[(&[i32], usize); 3],
    vis_w: u32,
    vis_h: u32,
    r: u32,
    c: u32,
    mv: [i32; 2],
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) {
    predict_inter_leaf(
        pred, reference, vis_w, vis_h, r, c, BLOCK_8X8, mv, mi_cols, mi_rows, ssx, ssy, bit_depth,
    );
}

/// Run the §8.5.2 inter prediction for one `MiSize >= BLOCK_8X8` leaf at
/// `(r, c)` with the (eighth-pel) motion vector `mv`, writing all three
/// planes' predicted regions — [`predict_mi_block`] generalised over the
/// leaf's `MiSize`, exactly what the decoder's §6.4.21 inter arm produces
/// before the token loop.
// Spec-shaped geometry fan-in, matching the style of the §8.5.2 driver.
#[allow(clippy::too_many_arguments)]
fn predict_inter_leaf(
    pred: &mut [Plane; 3],
    reference: &[(&[i32], usize); 3],
    vis_w: u32,
    vis_h: u32,
    r: u32,
    c: u32,
    mi_size: u8,
    mv: [i32; 2],
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) {
    predict_inter_leaf2(
        pred,
        reference,
        None,
        vis_w,
        vis_h,
        r,
        c,
        mi_size,
        [mv, [0, 0]],
        mi_cols,
        mi_rows,
        ssx,
        ssy,
        bit_depth,
    );
}

/// [`predict_inter_leaf`] generalised over **compound** prediction: with
/// `second == Some( planes )` the §8.5.2 driver forms both lists'
/// predictions (list 0 from `reference` at `mv[ 0 ]`, list 1 from
/// `second` at `mv[ 1 ]`) and writes the `Round2( p0 + p1, 1 )` average
/// — the exact decoder compound path.
// Spec-shaped geometry fan-in, matching the style of the §8.5.2 driver.
#[allow(clippy::too_many_arguments)]
fn predict_inter_leaf2(
    pred: &mut [Plane; 3],
    reference: &[(&[i32], usize); 3],
    second: Option<&[(&[i32], usize); 3]>,
    vis_w: u32,
    vis_h: u32,
    r: u32,
    c: u32,
    mi_size: u8,
    mv: [[i32; 2]; 2],
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) {
    use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};
    let num8x8w = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize]);
    let num8x8h = u32::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize]);
    let block_mvs = [[mv[0]; 4], [mv[1]; 4]];
    for (plane, pred_plane) in pred.iter_mut().enumerate() {
        let sub_x = plane > 0 && ssx;
        let sub_y = plane > 0 && ssy;
        let base_x = (c * 8) >> u32::from(sub_x);
        let base_y = (r * 8) >> u32::from(sub_y);
        let region = (num8x8w * 8) as usize >> usize::from(sub_x);
        let region_h = (num8x8h * 8) as usize >> usize::from(sub_y);

        let (samples, stride) = reference[plane];
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples,
                    stride,
                    ref_frame_width: vis_w as i32,
                    ref_frame_height: vis_h as i32,
                }),
                second.map(|sp| {
                    let (s2, stride2) = sp[plane];
                    RefPlane {
                        samples: s2,
                        stride: stride2,
                        ref_frame_width: vis_w as i32,
                        ref_frame_height: vis_h as i32,
                    }
                }),
            ],
        };
        let grid = BlockGrid {
            mi_row: r as i32,
            mi_col: c as i32,
            mi_rows: mi_rows as i32,
            mi_cols: mi_cols as i32,
            mi_size,
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
            interp_filter: 0, // EIGHTTAP.
            bit_depth,
            is_compound: second.is_some(),
        };
        predict_inter(pred_plane, &args, &grid, &geom, &block_mvs, &refs, ssx, ssy);
    }
}

/// Run the §8.5.2 inter prediction for one **sub-8x8** leaf at `(r, c)`
/// with the §6.4.16 `BlockMvs[ refList ][ 4 ]` per-cell vectors,
/// predicting each 4x4 sub-block of every plane exactly as the decoder's
/// §6.4.21 inter arm does for `MiSize < BLOCK_8X8`: the per-plane grid is
/// the 8x8 MI cell's (`get_plane_block_size( BLOCK_8X8, plane )`) 4x4
/// walk, each step a 4x4 `predict_inter` region at `blockIdx = y *
/// num4x4w + x` — the shared §8.5.2 driver resolves the per-`blockIdx`
/// luma vector and the §8.5.2.1 averaged chroma vectors internally.
// Spec-shaped geometry fan-in, matching the style of the §8.5.2 driver.
#[allow(clippy::too_many_arguments)]
fn predict_inter_leaf_sub8x8(
    pred: &mut [Plane; 3],
    reference: &[(&[i32], usize); 3],
    second: Option<&[(&[i32], usize); 3]>,
    vis_w: u32,
    vis_h: u32,
    r: u32,
    c: u32,
    mi_size: u8,
    block_mvs: &[[[i32; 2]; 4]; 2],
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
) {
    debug_assert!(mi_size < BLOCK_8X8);
    for (plane, pred_plane) in pred.iter_mut().enumerate() {
        let sub_x = plane > 0 && ssx;
        let sub_y = plane > 0 && ssy;
        // §6.4.21: bsize = Max( MiSize, BLOCK_8X8 ) selects the plane
        // grid; the sub-8x8 arm predicts per 4x4 step.
        let plane_sz = get_plane_block_size(BLOCK_8X8, plane, ssx, ssy);
        let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
        let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];
        let base_x = ((c * 8) >> u32::from(sub_x)) as i32;
        let base_y = ((r * 8) >> u32::from(sub_y)) as i32;

        let (samples, stride) = reference[plane];
        let refs = RefPlanes {
            list: [
                Some(RefPlane {
                    samples,
                    stride,
                    ref_frame_width: vis_w as i32,
                    ref_frame_height: vis_h as i32,
                }),
                second.map(|sp| {
                    let (s2, stride2) = sp[plane];
                    RefPlane {
                        samples: s2,
                        stride: stride2,
                        ref_frame_width: vis_w as i32,
                        ref_frame_height: vis_h as i32,
                    }
                }),
            ],
        };
        let grid = BlockGrid {
            mi_row: r as i32,
            mi_col: c as i32,
            mi_rows: mi_rows as i32,
            mi_cols: mi_cols as i32,
            mi_size,
        };
        let geom = ScaleGeom {
            ref_frame_width: vis_w as i32,
            ref_frame_height: vis_h as i32,
            frame_width: vis_w as i32,
            frame_height: vis_h as i32,
            subsampling_x: ssx,
            subsampling_y: ssy,
        };
        for y in 0..num4x4h {
            for x in 0..num4x4w {
                let args = InterPredArgs {
                    plane,
                    x: base_x + 4 * x as i32,
                    y: base_y + 4 * y as i32,
                    w: 4,
                    h: 4,
                    block_idx: (y * num4x4w + x) as usize,
                    interp_filter: 0, // EIGHTTAP.
                    bit_depth,
                    is_compound: second.is_some(),
                };
                predict_inter(pred_plane, &args, &grid, &geom, block_mvs, &refs, ssx, ssy);
            }
        }
    }
}

/// The §6.4.16 `BlockMvs[ refList ][ 4 ]` replication for a sub-8x8
/// leaf's per-cell plan: each visited `(idy, idx)` cell's vector is
/// copied across its `(num4x4h × num4x4w)` span, exactly as the decoder's
/// per-sub-block walk does.
fn sub8x8_block_mvs(
    mi_size: u8,
    sub: &crate::inter_block_writer::InterSubBlockSpec,
) -> [[[i32; 2]; 4]; 2] {
    let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[mi_size as usize] as usize;
    let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[mi_size as usize] as usize;
    let mut block_mvs = [[[0i32; 2]; 4]; 2];
    let mut idy = 0usize;
    while idy < 2 {
        let mut idx = 0usize;
        while idx < 2 {
            let cell = idy * 2 + idx;
            for y2 in 0..num4x4h {
                for x2 in 0..num4x4w {
                    let b = (idy + y2) * 2 + idx + x2;
                    block_mvs[0][b] = sub.mvs[cell][0];
                    block_mvs[1][b] = sub.mvs[cell][1];
                }
            }
            idx += num4x4w;
        }
        idy += num4x4h;
    }
    block_mvs
}

/// Encode one **lossless** P-frame over a caller-chosen §6.4.3 partition
/// layout — [`encode_pframe_lossless_motion`] generalised past the
/// all-`BLOCK_8X8` grid: the layout may split `BLOCK_8X8` nodes into the
/// sub-8x8 shapes (4x4 / 4x8 / 8x4 leaves carrying per-cell inter modes
/// and motion vectors through
/// [`crate::frame_writer::InterTreeLeaf::sub`]).
///
/// Each non-skip leaf is §8.5.2-predicted with exactly the vectors it
/// codes (the per-`blockIdx` sub-8x8 walk included), and the §8.7.2 WHT
/// residual makes `Clip1( prediction + residual ) == target` hold
/// sample-for-sample, so the frame chain stays bit-exact. Leaves planned
/// `skip = true` reconstruct from prediction alone — the caller elects
/// skip only when the prediction is exact (e.g. `ZEROMV` copy blocks).
///
/// `leaf_plan` supplies the per-leaf mode info in §6.4.3 decode order
/// against the shared [`Vp9FrameState`]; any non-`ZEROMV` plan (block
/// level or sub-8x8 cell) requires an error-resilient header, per
/// [`crate::frame_writer::assemble_inter_frame_tree`]'s §7.2.6 model.
// Drives the sub-8x8 corpus-stream builder (`build_sub8x8_inter_stream`)
// and its round-trip tests; the production sequence encoders adopt
// sub-8x8 layouts once their partition search elects them.
#[allow(dead_code)]
// Spec-shaped fan-in (header + targets + per-list references + layout),
// matching the other pixel-encoder drivers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_pframe_lossless_layout(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    second: Option<&[(&[i32], usize); 3]>,
    ref_w: u32,
    ref_h: u32,
    partitions: std::collections::HashMap<(u32, u32, u8), u8>,
    leaf_plan: &mut dyn FnMut(
        u32,
        u32,
        u8,
        &crate::decode_block::Vp9FrameState,
    ) -> crate::frame_writer::InterTreeLeaf,
) -> Result<Vec<u8>, Error> {
    use crate::frame_writer::{InterFrameTreePlan, InterTreeLeaf, InterTreePlanner};
    use crate::mode_info::{ALTREF_FRAME, INTRA_FRAME, LAST_FRAME, NONE_REF_FRAME};
    use std::cell::RefCell;

    if hdr.frame_type != FrameType::NonKeyFrame || !hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);

    let y_w = (mi_cols * 8) as usize;
    let y_h = (mi_rows * 8) as usize;
    let uv_w = y_w >> usize::from(ssx);
    let uv_h = y_h >> usize::from(ssy);
    let pred = RefCell::new([
        Plane::new(y_w, y_h),
        Plane::new(uv_w, uv_h),
        Plane::new(uv_w, uv_h),
    ]);

    let mut planner: Box<InterTreePlanner<'_>> =
        Box::new(|r: u32, c: u32, subsize: u8, state| -> InterTreeLeaf {
            let leaf = leaf_plan(r, c, subsize, state);
            // The prediction below maps `reference` to LAST and (when
            // planned compound) `second` to ALTREF — the §6.3.18
            // [ CompVarRef[ 0 ], CompFixedRef ] pair under the
            // asymmetric-ALTREF sign biases. Any other reference plan
            // would silently predict from the wrong planes.
            let is_comp = leaf.ref_frame[1] > INTRA_FRAME;
            debug_assert_eq!(
                leaf.ref_frame[0], LAST_FRAME,
                "layout encoder predicts LAST"
            );
            debug_assert!(
                leaf.ref_frame[1] == NONE_REF_FRAME
                    || (leaf.ref_frame[1] == ALTREF_FRAME && second.is_some()),
                "compound leaves pair [ LAST, ALTREF ] and need `second`"
            );
            let second_for_leaf = if is_comp { second } else { None };
            // Predict with exactly what will be coded, so the residual
            // callback below sees the decoder's prediction. Skip leaves
            // never request coefficients, so their prediction is only
            // needed when it IS the reconstruction — which the caller
            // guarantees by planning skip solely for exact-copy blocks;
            // predicting them anyway keeps the planes decoder-true.
            let mut pred = pred.borrow_mut();
            if subsize < BLOCK_8X8 {
                if let Some(sub) = leaf.sub.as_ref() {
                    let block_mvs = sub8x8_block_mvs(subsize, sub);
                    predict_inter_leaf_sub8x8(
                        &mut pred,
                        reference,
                        second_for_leaf,
                        ref_w,
                        ref_h,
                        r,
                        c,
                        subsize,
                        &block_mvs,
                        mi_cols,
                        mi_rows,
                        ssx,
                        ssy,
                        bit_depth,
                    );
                }
            } else {
                predict_inter_leaf2(
                    &mut pred,
                    reference,
                    second_for_leaf,
                    ref_w,
                    ref_h,
                    r,
                    c,
                    subsize,
                    leaf.mv,
                    mi_cols,
                    mi_rows,
                    ssx,
                    ssy,
                    bit_depth,
                );
            }
            leaf
        });

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        |_mi_r: u32, _mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            let pred = pred.borrow();
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

    // Compound leaves need the §6.3.12 reference-mode syntax; a frame
    // without a second reference stays on the single-reference arm.
    let reference_mode = if second.is_some() {
        crate::compressed::ReferenceMode::ReferenceModeSelect
    } else {
        crate::compressed::ReferenceMode::SingleReference
    };
    // A non-error-resilient header is this encoder's assertion that the
    // §7.2.6 UsePrevFrameMvs derivation yields 0 for the frame (hidden
    // / intra / differently-sized predecessor) — see the function docs.
    let plan = InterFrameTreePlan {
        tx_mode: crate::compressed::TxMode::Only4x4,
        reference_mode,
        partitions,
        prev_segment_ids: None,
        prev_frame_mvs_absent: !hdr.error_resilient_mode,
        prev_frame_mvs: None,
    };
    crate::frame_writer::assemble_inter_frame_tree(hdr, &plan, &mut *planner, &mut *coeffs)
}

/// Full-search one 8x8 luma block over integer motion vectors in
/// `[-range, range]²`, returning `((dy, dx), best_sad, zero_sad)`.
///
/// The reference read is edge-clamped to the visible extents, matching
/// the §8.5.2.4 `Clip3( 0, lastX/lastY, . )` sampling for full-pel
/// vectors, so the SAD equals the true prediction error.
#[allow(clippy::too_many_arguments)]
fn search_block_mv(
    target: &Plane,
    ref_samples: &[i32],
    ref_stride: usize,
    vis_w: i32,
    vis_h: i32,
    bx: i32,
    by: i32,
    range: i32,
) -> ((i32, i32), u64, u64) {
    search_block_mv_wh(
        target,
        ref_samples,
        ref_stride,
        vis_w,
        vis_h,
        bx,
        by,
        8,
        8,
        range,
    )
}

/// Full-search one `bw × bh` luma block over integer motion vectors in
/// `[-range, range]²` — [`search_block_mv`] generalised over the block
/// extents. Returns `((dy, dx), best_sad, zero_sad)`.
#[allow(clippy::too_many_arguments)]
fn search_block_mv_wh(
    target: &Plane,
    ref_samples: &[i32],
    ref_stride: usize,
    vis_w: i32,
    vis_h: i32,
    bx: i32,
    by: i32,
    bw: i32,
    bh: i32,
    range: i32,
) -> ((i32, i32), u64, u64) {
    let sad_at = |dy: i32, dx: i32| -> u64 {
        let mut sad = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                let ry = (by + i + dy).clamp(0, vis_h - 1) as usize;
                let rx = (bx + j + dx).clamp(0, vis_w - 1) as usize;
                let t = target.get((bx + j) as usize, (by + i) as usize);
                let p = ref_samples[ry * ref_stride + rx];
                sad += (t - p).unsigned_abs() as u64;
            }
        }
        sad
    };
    let zero_sad = sad_at(0, 0);
    let mut best = ((0i32, 0i32), zero_sad);
    for dy in -range..=range {
        for dx in -range..=range {
            if dy == 0 && dx == 0 {
                continue;
            }
            let sad = sad_at(dy, dx);
            if sad < best.1 {
                best = ((dy, dx), sad);
            }
        }
    }
    (best.0, best.1, zero_sad)
}

/// Sub-pel refinement of one leaf's motion vector around an integer
/// (or already-refined) starting point, scoring candidates with the
/// **decoder-mirror §8.5.2 interpolation**: each candidate's luma
/// prediction is produced by the same `predict_inter` chain (§8.5.2.1
/// select / §8.5.2.2 clamp / §8.5.2.3 scale / §8.5.2.4 two-pass 8-tap
/// convolution over the `subpel_filters` kernels) the coded block will
/// reconstruct with, so the SAD is the *true* prediction error at that
/// vector — not a bilinear approximation.
///
/// The walk is a coarse-to-fine neighbourhood descent in eighth-pel
/// units: half-pel (±4), then quarter-pel (±2), then eighth-pel (±1)
/// **only when the §6.5.13 `use_mv_hp` gate allows the hp bit**
/// (`start_mv` must already be §6.4.20-codeable against the predictor;
/// even steps preserve the difference parity, so every candidate stays
/// codeable, and the ±1 step is reachable only when odd differences
/// are).
///
/// Returns the best vector and its luma SAD over the leaf's region
/// (clipped to the MI-padded plane extents).
#[allow(clippy::too_many_arguments)]
fn refine_leaf_mv_subpel(
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    vis_w: u32,
    vis_h: u32,
    r: u32,
    c: u32,
    mi_size: u8,
    start_mv: [i32; 2],
    use_hp: bool,
    mi_cols: u32,
    mi_rows: u32,
    bit_depth: u32,
    scratch: &mut Plane,
) -> ([i32; 2], u64) {
    use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};

    let num8x8w = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize]);
    let num8x8h = u32::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[mi_size as usize]);
    let base_x = (c * 8) as usize;
    let base_y = (r * 8) as usize;
    let maxx = (mi_cols * 8) as usize;
    let maxy = (mi_rows * 8) as usize;
    let region_w = (num8x8w * 8) as usize;
    let region_h = (num8x8h * 8) as usize;

    let (samples, stride) = reference[0];
    let mut sad_of = |mv: [i32; 2]| -> u64 {
        let block_mvs = [[mv; 4], [[0i32; 2]; 4]];
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
            mi_size,
        };
        let geom = ScaleGeom {
            ref_frame_width: vis_w as i32,
            ref_frame_height: vis_h as i32,
            frame_width: vis_w as i32,
            frame_height: vis_h as i32,
            subsampling_x: false,
            subsampling_y: false,
        };
        let args = InterPredArgs {
            plane: 0,
            x: base_x as i32,
            y: base_y as i32,
            w: region_w,
            h: region_h,
            block_idx: 0,
            interp_filter: 0, // EIGHTTAP — what the leaf will code.
            bit_depth,
            is_compound: false,
        };
        predict_inter(
            scratch, &args, &grid, &geom, &block_mvs, &refs, false, false,
        );
        let mut sad = 0u64;
        for i in 0..region_h {
            for j in 0..region_w {
                let (x, y) = (base_x + j, base_y + i);
                if x < maxx && y < maxy {
                    let d = targets[0].get(x, y) - scratch.get(x, y);
                    sad += d.unsigned_abs() as u64;
                }
            }
        }
        sad
    };

    let mut best = (start_mv, sad_of(start_mv));
    // Coarse-to-fine: half-pel, quarter-pel, then eighth-pel under hp.
    for &step in &[4i32, 2, 1] {
        if step == 1 && !use_hp {
            break; // odd differences are not §6.4.20-codeable.
        }
        // Descend at this granularity until no neighbour improves
        // (bounded: each move strictly lowers the SAD).
        let mut moved = true;
        let mut rounds = 0;
        while moved && rounds < 4 {
            moved = false;
            rounds += 1;
            let centre = best.0;
            for dy in [-step, 0, step] {
                for dx in [-step, 0, step] {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let cand = [centre[0] + dy, centre[1] + dx];
                    let sad = sad_of(cand);
                    if sad < best.1 {
                        best = (cand, sad);
                        moved = true;
                    }
                }
            }
        }
    }
    best
}

/// Encode one lossless P-frame whose reconstruction equals `targets`
/// (MI-padded planes) exactly, referencing `reference` (the previous
/// frame's visible-extent planes), with per-block integer motion search
/// over `[-search_range, search_range]²` luma pixels (`0` disables the
/// search: every block codes `ZEROMV`).
///
/// The planner derives each block's §6.5.12 `BestMv` with the **shared**
/// `find_mv_refs` / `find_best_ref_mvs` over the same `Vp9FrameState`
/// the inter block writer reads (so the predictors are bit-identical),
/// elects `NEWMV` when the searched vector beats `ZEROMV` by a margin,
/// and snaps the MV difference to the §6.4.20-codeable grid when the
/// §6.5.13 `use_mv_hp( BestMv )` gate disables the eighth-pel bit (the
/// no-hp decode fixes `hp == 1`, so only even-magnitude differences are
/// codeable; a snapped vector merely changes the prediction, which the
/// exact WHT residual absorbs). Each block is §8.5.2-predicted with the
/// vector actually coded, so `Clip1( prediction + residual ) == target`
/// holds sample-for-sample and the frame chain stays bit-exact.
///
/// With `subpel == true` the integer winner is additionally refined at
/// half- / quarter- / (under the hp gate) eighth-pel precision against
/// the decoder-mirror §8.5.2 interpolation ([`refine_leaf_mv_subpel`]),
/// and blocks whose integer winner is `(0, 0)` but whose `ZEROMV` error
/// is non-trivial are probed for pure sub-pel motion too.
///
/// `search_range > 0` requires an error-resilient header (the §7.2.6
/// `UsePrevFrameMvs == 0` model — see
/// [`crate::frame_writer::assemble_inter_frame_planned`]) — or the
/// chained variant below, which models `UsePrevFrameMvs == 1` instead.
pub(crate) fn encode_pframe_lossless_motion(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
    search_range: i32,
    subpel: bool,
) -> Result<Vec<u8>, Error> {
    encode_pframe_lossless_motion_prev(
        hdr,
        targets,
        reference,
        ref_w,
        ref_h,
        search_range,
        subpel,
        None,
    )
    .map(|(bytes, _)| bytes)
}

/// [`encode_pframe_lossless_motion`] with the §7.2.6 prev-motion-field
/// model: `prev` (the previous frame's §6.4.4 motion field) is fed to
/// BOTH the planner's §6.5 predictor derivation and the block writer's
/// scan, so a non-error-resilient SHOWN chain codes `NEWMV` /
/// `NEARESTMV` / `NEARMV` with predictors bit-identical to the
/// decoder's `UsePrevFrameMvs == 1` derivation — and a vector the
/// previous frame already coded at the same position maps to
/// `NEARESTMV` / `NEARMV` (no §6.4.20 mv-diff bits) through the prev
/// candidate. Returns the frame's write-back state for the next
/// frame's `prev`. `prev = None` on an error-resilient header is the
/// classic model, byte-identical to
/// [`encode_pframe_lossless_motion`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_pframe_lossless_motion_prev(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
    search_range: i32,
    subpel: bool,
    prev: Option<&crate::frame_writer::PrevMotionField>,
) -> Result<(Vec<u8>, crate::decode_block::Vp9FrameState), Error> {
    use crate::inter_decode::{FrameStateMvSource, PrevFrameMvs};
    use crate::mode_info::{LAST_FRAME, NEARESTMV, NEARMV, NEWMV, ZEROMV};
    use crate::mv::use_mv_hp;
    use crate::mv_ref::MvRefGeometry;
    use std::cell::RefCell;

    if hdr.frame_type != FrameType::NonKeyFrame || !hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);
    // §7.2 setup_past_independence( ): error-resilient frames have
    // all-zero *effective* sign biases (see assemble_inter_frame_tree).
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

    let y_w = (mi_cols * 8) as usize;
    let y_h = (mi_rows * 8) as usize;
    let uv_w = y_w >> usize::from(ssx);
    let uv_h = y_h >> usize::from(ssy);
    let pred = RefCell::new([
        Plane::new(y_w, y_h),
        Plane::new(uv_w, uv_h),
        Plane::new(uv_w, uv_h),
    ]);
    let scratch = RefCell::new(Plane::new(y_w, y_h));

    // Prefer NEWMV only for a clear win: the mode + MV syntax costs bits
    // that a marginal SAD gain does not repay.
    const NEWMV_SAD_MARGIN: u64 = 64;

    let mut planner: Box<crate::frame_writer::InterBlockPlanner<'_>> =
        Box::new(|r, c, state| -> (u8, [i32; 2], bool) {
            let mut choice: (u8, [i32; 2]) = (ZEROMV, [0, 0]);
            if search_range > 0 {
                let ((dy, dx), best_sad, zero_sad) = search_block_mv(
                    &targets[0],
                    reference[0].0,
                    reference[0].1,
                    ref_w as i32,
                    ref_h as i32,
                    (c * 8) as i32,
                    (r * 8) as i32,
                    search_range,
                );
                // Worth deriving predictors: an integer winner beating
                // ZEROMV by the margin, or (sub-pel mode) a non-trivial
                // ZEROMV error that pure sub-pel motion might explain.
                let int_winner = (dy, dx) != (0, 0) && best_sad + NEWMV_SAD_MARGIN < zero_sad;
                if int_winner || (subpel && zero_sad > NEWMV_SAD_MARGIN) {
                    // §6.5 predictors over the shared state — identical
                    // to the derivation the inter block writer performs.
                    let geom = MvRefGeometry {
                        mi_row: r as i32,
                        mi_col: c as i32,
                        mi_rows: mi_rows as i32,
                        mi_cols: mi_cols as i32,
                        mi_size: BLOCK_8X8 as usize,
                        mi_col_start: 0,
                        mi_col_end: mi_cols as i32,
                    };
                    let src = FrameStateMvSource::new(
                        state,
                        prev.map(|p| PrevFrameMvs {
                            prev_ref_frames: &p.ref_frames,
                            prev_mvs: &p.mvs,
                        }),
                    );
                    let mv_refs =
                        geom.find_mv_refs(&src, LAST_FRAME, -1, &sign_bias, prev.is_some());
                    let preds =
                        geom.find_best_ref_mvs(mv_refs.ref_list_mv, hdr.allow_high_precision_mv);
                    let best = preds[0];

                    // §6.4.16 mode mapping: a vector equal to a §6.5
                    // predictor codes NEARESTMV / NEARMV (no §6.4.20
                    // mv-diff bits) instead of NEWMV.
                    let mode_for = |mv: [i32; 2]| -> u8 {
                        if mv == preds[0] {
                            NEARESTMV
                        } else if mv == preds[1] {
                            NEARMV
                        } else {
                            NEWMV
                        }
                    };

                    let start = if int_winner { [8 * dy, 8 * dx] } else { [0, 0] };
                    let mut mv = start;
                    let use_hp = hdr.allow_high_precision_mv && use_mv_hp(best);
                    for (comp, m) in mv.iter_mut().enumerate() {
                        let d = *m - best[comp];
                        if d != 0 && !use_hp && (d & 1) != 0 {
                            // Only even-magnitude differences are codeable
                            // without the hp bit; nudge by one eighth-pel.
                            *m -= 1;
                        }
                    }
                    if subpel {
                        let (refined, refined_sad) = refine_leaf_mv_subpel(
                            targets,
                            reference,
                            ref_w,
                            ref_h,
                            r,
                            c,
                            BLOCK_8X8,
                            mv,
                            use_hp,
                            mi_cols,
                            mi_rows,
                            bit_depth,
                            &mut scratch.borrow_mut(),
                        );
                        if refined != [0, 0] && refined_sad + NEWMV_SAD_MARGIN < zero_sad {
                            choice = (mode_for(refined), refined);
                        }
                    } else if int_winner {
                        choice = (mode_for(mv), mv);
                    }
                }
            }
            // Predict this block with the vector that will be coded, so
            // the residual callbacks below see the decoder's prediction.
            let mut pred = pred.borrow_mut();
            predict_mi_block(
                &mut pred, reference, ref_w, ref_h, r, c, choice.1, mi_cols, mi_rows, ssx, ssy,
                bit_depth,
            );
            // Elect `skip` when the prediction is *exact* over every
            // visible sample of the MI block (all three planes): the
            // WHT of an all-zero residual is all-zero tokens, so a skip
            // block reconstructs identically while saving the per-block
            // end-of-block bits.
            let mut exact = true;
            'outer: for plane in 0..3usize {
                let sub_x = plane > 0 && ssx;
                let sub_y = plane > 0 && ssy;
                let base_x = ((c * 8) >> u32::from(sub_x)) as usize;
                let base_y = ((r * 8) >> u32::from(sub_y)) as usize;
                let maxx = ((mi_cols * 8) >> u32::from(sub_x)) as usize;
                let maxy = ((mi_rows * 8) >> u32::from(sub_y)) as usize;
                let region_w = 8usize >> usize::from(sub_x);
                let region_h = 8usize >> usize::from(sub_y);
                for i in 0..region_h {
                    for j in 0..region_w {
                        let (x, y) = (base_x + j, base_y + i);
                        if x < maxx && y < maxy && targets[plane].get(x, y) != pred[plane].get(x, y)
                        {
                            exact = false;
                            break 'outer;
                        }
                    }
                }
            }
            (choice.0, choice.1, exact)
        });

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        |_mi_r: u32, _mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            let pred = pred.borrow();
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

    crate::frame_writer::assemble_inter_frame_planned_with_state(
        hdr,
        crate::compressed::TxMode::Only4x4,
        false,
        prev.cloned(),
        &mut *planner,
        &mut *coeffs,
    )
}

/// Integer motion-search window (±luma pixels) for the P-frame
/// sequence encoders.
pub(crate) const PFRAME_SEARCH_RANGE: i32 = 8;

/// Encode one **lossy** P-frame at the header's `base_q_idx`: per-block
/// `ZEROMV` / `NEWMV` motion (integer full search against the reference —
/// the previous frame's *reconstruction*, not its source), quantized
/// forward-DCT residual, and the decoder's §8.6.2 reconstruction replayed
/// in place — so the returned [`ReconState`] equals the decoder's output
/// bit-for-bit and its visible crop is the next frame's reference.
///
/// The frame codes `tx_mode = Allow8x8`: every `BLOCK_8X8` block's luma
/// residual is one §6.4.10-**inferred** `TX_8X8` DCT (no per-block tx
/// bits; chroma stays at its §6.4.22 subsampled size), replacing the
/// four 4x4 blocks per MI the `Only4x4` layout coded.
///
/// Requires an error-resilient non-key lossy header (see
/// [`crate::frame_writer::assemble_inter_frame_planned`]).
///
/// Retained as the fixed-layout (`Allow8x8`, all-`BLOCK_8X8`) baseline
/// the tree encoder's rate tests compare against; the production
/// sequence encoders drive [`encode_pframe_lossy_tree_motion`].
#[allow(dead_code)]
pub(crate) fn encode_pframe_lossy_motion(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
    search_range: i32,
) -> Result<(Vec<u8>, ReconState), Error> {
    use crate::inter_decode::FrameStateMvSource;
    use crate::mode_info::{LAST_FRAME, NEWMV, ZEROMV};
    use crate::mv::use_mv_hp;
    use crate::mv_ref::MvRefGeometry;
    use std::cell::RefCell;

    if hdr.frame_type != FrameType::NonKeyFrame || hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);
    // §7.2 setup_past_independence( ): error-resilient frames have
    // all-zero *effective* sign biases (see assemble_inter_frame_tree).
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
    let seg = hdr.segmentation;
    let quant = hdr.quantization;
    let bd8 = hdr.color_config.bit_depth;

    // Work planes: the planner writes each block's §8.5.2 prediction,
    // then (for non-skip blocks) replays the decoder's §8.6.2
    // reconstruction in place with the pre-computed tokens.
    let work = RefCell::new(ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth));
    // Per-transform-block quantized tokens, computed by the planner (it
    // needs them all before the block's syntax is written to elect
    // `skip`) and served back by the coefficient callback, keyed by
    // `(plane, start_x, start_y)`.
    let token_cache: RefCell<std::collections::HashMap<(usize, u32, u32), Vec<i64>>> =
        RefCell::new(std::collections::HashMap::new());

    // §6.4.10 inferred tx under Allow8x8 at BLOCK_8X8: TX_8X8 luma; the
    // chroma size follows §6.4.22.
    let tx_mode = crate::compressed::TxMode::Allow8x8;
    let luma_tx = crate::mode_writer::inferred_tx_size(
        crate::residual::MAX_TXSIZE_LOOKUP[BLOCK_8X8 as usize],
        tx_mode,
    );

    const NEWMV_SAD_MARGIN: u64 = 64;

    let mut planner: Box<crate::frame_writer::InterBlockPlanner<'_>> =
        Box::new(|r, c, state| -> (u8, [i32; 2], bool) {
            let mut choice: (u8, [i32; 2]) = (ZEROMV, [0, 0]);
            if search_range > 0 {
                let ((dy, dx), best_sad, zero_sad) = search_block_mv(
                    &targets[0],
                    reference[0].0,
                    reference[0].1,
                    ref_w as i32,
                    ref_h as i32,
                    (c * 8) as i32,
                    (r * 8) as i32,
                    search_range,
                );
                if (dy, dx) != (0, 0) && best_sad + NEWMV_SAD_MARGIN < zero_sad {
                    let geom = MvRefGeometry {
                        mi_row: r as i32,
                        mi_col: c as i32,
                        mi_rows: mi_rows as i32,
                        mi_cols: mi_cols as i32,
                        mi_size: BLOCK_8X8 as usize,
                        mi_col_start: 0,
                        mi_col_end: mi_cols as i32,
                    };
                    let src = FrameStateMvSource::new(state, None);
                    let mv_refs = geom.find_mv_refs(&src, LAST_FRAME, -1, &sign_bias, false);
                    let best =
                        geom.find_best_ref_mvs(mv_refs.ref_list_mv, hdr.allow_high_precision_mv)[0];

                    let mut mv = [8 * dy, 8 * dx];
                    let use_hp = hdr.allow_high_precision_mv && use_mv_hp(best);
                    for (comp, m) in mv.iter_mut().enumerate() {
                        let d = *m - best[comp];
                        if d != 0 && !use_hp && (d & 1) != 0 {
                            *m -= 1;
                        }
                    }
                    choice = (NEWMV, mv);
                }
            }
            let mut work = work.borrow_mut();
            predict_mi_block(
                &mut work.planes,
                reference,
                ref_w,
                ref_h,
                r,
                c,
                choice.1,
                mi_cols,
                mi_rows,
                ssx,
                ssy,
                bit_depth,
            );

            // Quantize every coded transform block of this MI now (the
            // same §6.4.21 grid the residual writer will walk), elect
            // `skip` when the whole block is zero, and for a non-skip
            // block replay the decoder's reconstruction immediately.
            let mut cache = token_cache.borrow_mut();
            let mut all_zero = true;
            let mut blocks: Vec<(usize, u32, u32, u32, Vec<i64>)> = Vec::new();
            // The §6.4.21 loop is keyed by `plane` and indexes the work /
            // target planes and subsampling by that same index, mirroring
            // the spec listing directly.
            #[allow(clippy::needless_range_loop)]
            for plane in 0..3usize {
                let tx_sz = if plane == 0 {
                    luma_tx
                } else {
                    crate::residual::get_uv_tx_size(luma_tx, BLOCK_8X8, ssx, ssy)
                };
                let sub_x = plane > 0 && ssx;
                let sub_y = plane > 0 && ssy;
                let base_x = (c * 8) >> u32::from(sub_x);
                let base_y = (r * 8) >> u32::from(sub_y);
                let maxx = (mi_cols * 8) >> u32::from(sub_x);
                let maxy = (mi_rows * 8) >> u32::from(sub_y);
                let n0 = 4u32 << tx_sz;
                let region_w = 8u32 >> u32::from(sub_x);
                let region_h = 8u32 >> u32::from(sub_y);
                let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
                let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
                let mut sy = base_y;
                while sy < base_y + region_h {
                    let mut sx = base_x;
                    while sx < base_x + region_w {
                        if sx < maxx && sy < maxy {
                            let n0u = n0 as usize;
                            let mut block = vec![0i64; n0u * n0u];
                            for i in 0..n0u {
                                for j in 0..n0u {
                                    let t = targets[plane].get(sx as usize + j, sy as usize + i);
                                    let p =
                                        work.planes[plane].get(sx as usize + j, sy as usize + i);
                                    block[i * n0u + j] = i64::from(t) - i64::from(p);
                                }
                            }
                            // §6.4.25: inter blocks are DCT_DCT at every size.
                            crate::fwd_transform::forward_dct_2d(&mut block, tx_sz + 2);
                            crate::fwd_transform::quantize_block_tx(
                                &mut block, dc_q, ac_q, tx_sz, bit_depth,
                            );
                            all_zero &= block.iter().all(|&v| v == 0);
                            blocks.push((plane, sx, sy, tx_sz, block));
                        }
                        sx += n0;
                    }
                    sy += n0;
                }
            }
            let skip = all_zero;
            for (plane, sx, sy, tx_sz, block) in blocks {
                if !skip {
                    let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
                    let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
                    reconstruct_block(
                        &mut work.planes[plane],
                        sx as usize,
                        sy as usize,
                        tx_sz,
                        &block,
                        dc_q,
                        ac_q,
                        DCT_DCT,
                        false,
                        bit_depth,
                    );
                    cache.insert((plane, sx, sy), block);
                }
                // Skip blocks reconstruct from prediction alone — the
                // work planes already hold it.
            }

            (choice.0, choice.1, skip)
        });

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        |_mi_r: u32, _mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            token_cache
                .borrow_mut()
                .remove(&(plane, sx, sy))
                .expect("planner pre-computed this block's tokens")
        },
    );

    let bytes = crate::frame_writer::assemble_inter_frame_planned(
        hdr,
        tx_mode,
        false,
        &mut *planner,
        &mut *coeffs,
    )?;
    drop(planner);
    drop(coeffs);
    Ok((bytes, work.into_inner()))
}

/// The per-8x8-cell margin a searched `NEWMV` must beat `ZEROMV` by
/// before it is elected: the mode + MV syntax costs bits that a marginal
/// SAD gain does not repay.
const NEWMV_SAD_MARGIN_PER_MI: u64 = 64;

/// §6.4.3 partition value per recursion node, keyed `(MiRow, MiCol,
/// bsize)` (the [`crate::frame_writer::InterFrameTreePlan`] map).
type InterPartitionMap = std::collections::HashMap<(u32, u32, u8), u8>;

/// Per-leaf integer motion-vector hints keyed by the leaf's top-left MI.
type InterMvHints = std::collections::HashMap<(u32, u32), (i32, i32)>;

/// A below-8x8 split the planner elected for one 8x8 MI cell: the
/// §6.4.3 partition value at the `BLOCK_8X8` node (`PARTITION_HORZ` /
/// `PARTITION_VERT` / `PARTITION_SPLIT`) plus the per-cell integer
/// motion vectors in §6.4.16 `idy * 2 + idx` layout (only the cells the
/// sub-8x8 walk visits are read).
#[derive(Debug, Clone, Copy)]
struct Sub8x8Hint {
    partition: u8,
    cell_mvs: [(i32, i32); 4],
}

/// Elected sub-8x8 splits keyed by the 8x8 cell's `(MiRow, MiCol)`.
type InterSub8x8Hints = std::collections::HashMap<(u32, u32), Sub8x8Hint>;

/// Margin a 4x4 quadrant's searched vector must beat its `ZEROMV` SAD
/// by before the sub-8x8 probe adopts it (the quadrant-level analogue
/// of [`NEWMV_SAD_MARGIN_PER_MI`], scaled to a quarter of the area).
const SUB8X8_QUAD_MARGIN: u64 = 24;

/// Per-coded-vector syntax margin a below-8x8 decomposition must beat
/// the cell's best single-vector SAD by before it is elected: each
/// visited cell codes its own `inter_mode` token and (for `NEWMV`)
/// §6.4.20 mv-diff bits, so a split must pay for that rate in
/// prediction error.
const SUB8X8_SPLIT_MARGIN_PER_MV: u64 = 48;

/// Plan the §6.4.3 partition tree of a lossy P-frame from its integer
/// motion field — the inter counterpart of [`plan_keyframe_tree`].
///
/// First every 8x8 MI cell full-searches an integer motion vector
/// against the reference (electing `(0, 0)` unless the winner beats
/// `ZEROMV` by [`NEWMV_SAD_MARGIN_PER_MI`]); then, with `sub8x8`
/// enabled, each cell probes its four 4x4 quadrants independently — a
/// cell whose quadrants elect **divergent** vectors that beat the best
/// single-vector SAD by [`SUB8X8_SPLIT_MARGIN_PER_MV`] per coded vector
/// becomes a below-8x8 leaf (`PARTITION_HORZ` → 8x4 when only the top /
/// bottom halves differ, `PARTITION_VERT` → 4x8 when only the left /
/// right halves differ, `PARTITION_SPLIT` → 4x4 otherwise) with
/// per-cell MV hints; then the superblock tree merges bottom-up: a node
/// becomes one leaf when it is fully contained in the frame's MI
/// extents **and** every 8x8 cell under it elected the same vector with
/// no sub-8x8 split (uniform motion — one coded MV serves the whole
/// leaf, and a larger transform can span the coherent residual).
/// Anything else splits toward `BLOCK_8X8`.
///
/// Returns the partition map (feeding
/// [`crate::frame_writer::InterFrameTreePlan`]), the per-leaf
/// integer-MV hints keyed by the leaf's top-left MI, and the elected
/// sub-8x8 splits.
#[allow(clippy::too_many_arguments)]
fn plan_inter_partitions(
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
    mi_cols: u32,
    mi_rows: u32,
    search_range: i32,
    sub8x8: bool,
) -> (InterPartitionMap, InterMvHints, InterSub8x8Hints) {
    use crate::partition::{
        NUM_8X8_BLOCKS_WIDE_LOOKUP, PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT,
        PARTITION_VERT, SUBSIZE_LOOKUP,
    };
    use crate::residual::BLOCK_64X64;

    // The per-8x8-cell integer motion field + each cell's effective
    // (margin-adjusted) single-vector SAD.
    let mut field = vec![(0i32, 0i32); (mi_rows * mi_cols) as usize];
    let mut cell_sad = vec![0u64; (mi_rows * mi_cols) as usize];
    if search_range > 0 {
        for r in 0..mi_rows {
            for c in 0..mi_cols {
                let ((dy, dx), best_sad, zero_sad) = search_block_mv(
                    &targets[0],
                    reference[0].0,
                    reference[0].1,
                    ref_w as i32,
                    ref_h as i32,
                    (c * 8) as i32,
                    (r * 8) as i32,
                    search_range,
                );
                let cell = (r * mi_cols + c) as usize;
                if (dy, dx) != (0, 0) && best_sad + NEWMV_SAD_MARGIN_PER_MI < zero_sad {
                    field[cell] = (dy, dx);
                    cell_sad[cell] = best_sad + NEWMV_SAD_MARGIN_PER_MI;
                } else {
                    cell_sad[cell] = zero_sad;
                }
            }
        }
    }

    // Sub-8x8 probe: per-quadrant 4x4 searches, shape selection from
    // which quadrant pairs agree, and the per-coded-vector election
    // margin against the cell's best single-vector SAD.
    let mut sub_hints: InterSub8x8Hints = std::collections::HashMap::new();
    if sub8x8 && search_range > 0 {
        for r in 0..mi_rows {
            for c in 0..mi_cols {
                let cell = (r * mi_cols + c) as usize;
                // Quadrants in §6.4.16 idy * 2 + idx layout.
                let mut qmv = [(0i32, 0i32); 4];
                let mut qsad = [0u64; 4];
                for (q, (dy2, dx2)) in [(0i32, 0i32), (0, 1), (1, 0), (1, 1)]
                    .into_iter()
                    .enumerate()
                {
                    let ((dy, dx), best_sad, zero_sad) = search_block_mv_wh(
                        &targets[0],
                        reference[0].0,
                        reference[0].1,
                        ref_w as i32,
                        ref_h as i32,
                        (c * 8) as i32 + 4 * dx2,
                        (r * 8) as i32 + 4 * dy2,
                        4,
                        4,
                        search_range,
                    );
                    if (dy, dx) != (0, 0) && best_sad + SUB8X8_QUAD_MARGIN < zero_sad {
                        qmv[q] = (dy, dx);
                        qsad[q] = best_sad;
                    } else {
                        qsad[q] = zero_sad;
                    }
                }
                if qmv.iter().all(|&m| m == qmv[0]) {
                    continue; // uniform quadrant motion — no split.
                }
                // Shape: HORZ when rows agree, VERT when columns agree,
                // SPLIT otherwise; coded-vector count per §6.4.16 (the
                // visited cells).
                let (partition, n_mvs) = if qmv[0] == qmv[1] && qmv[2] == qmv[3] {
                    (PARTITION_HORZ, 2u64)
                } else if qmv[0] == qmv[2] && qmv[1] == qmv[3] {
                    (PARTITION_VERT, 2u64)
                } else {
                    (PARTITION_SPLIT, 4u64)
                };
                let sub_sad: u64 = qsad.iter().sum();
                // Election needs BOTH the absolute per-coded-vector
                // margin and a large (>= 2x) relative improvement: on
                // noise-like content the minimum-of-search quadrant
                // SADs sit only slightly below the single-vector SAD
                // (a statistical artefact, not motion), and a split
                // elected there would also forgo the >= 8x8 leaf's
                // compound / multi-reference candidates.
                if sub_sad + SUB8X8_SPLIT_MARGIN_PER_MV * n_mvs < cell_sad[cell]
                    && 2 * sub_sad < cell_sad[cell]
                {
                    sub_hints.insert(
                        (r, c),
                        Sub8x8Hint {
                            partition,
                            cell_mvs: qmv,
                        },
                    );
                }
            }
        }
    }

    let mut partitions = std::collections::HashMap::new();
    let mut hints = std::collections::HashMap::new();

    // Recursive merge: leaf on contained + uniform motion (a cell with
    // an elected sub-8x8 split never merges upward).
    // Spec-shaped geometry fan-in, matching the §6.4.3 recursion style.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        partitions: &mut InterPartitionMap,
        hints: &mut InterMvHints,
        sub_hints: &InterSub8x8Hints,
        field: &[(i32, i32)],
        mi_rows: u32,
        mi_cols: u32,
        r: u32,
        c: u32,
        bsize: u8,
    ) {
        use crate::residual::BLOCK_8X8 as B8;
        if r >= mi_rows || c >= mi_cols {
            return;
        }
        let num8x8 = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[bsize as usize]);
        let half = num8x8 >> 1;
        let contained = (r + num8x8) <= mi_rows && (c + num8x8) <= mi_cols;

        let mut merge = contained && bsize != B8;
        if merge {
            let mv0 = field[(r * mi_cols + c) as usize];
            'scan: for i in 0..num8x8 {
                for j in 0..num8x8 {
                    if field[((r + i) * mi_cols + (c + j)) as usize] != mv0
                        || sub_hints.contains_key(&(r + i, c + j))
                    {
                        merge = false;
                        break 'scan;
                    }
                }
            }
        }

        if bsize == B8 {
            if let Some(hint) = sub_hints.get(&(r, c)) {
                partitions.insert((r, c, bsize), hint.partition);
            } else {
                partitions.insert((r, c, bsize), PARTITION_NONE);
                hints.insert((r, c), field[(r * mi_cols + c) as usize]);
            }
            return;
        }
        if merge {
            partitions.insert((r, c, bsize), PARTITION_NONE);
            hints.insert((r, c), field[(r * mi_cols + c) as usize]);
            return;
        }
        partitions.insert((r, c, bsize), PARTITION_SPLIT);
        let subsize = SUBSIZE_LOOKUP[PARTITION_SPLIT as usize][bsize as usize];
        for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
            walk(
                partitions,
                hints,
                sub_hints,
                field,
                mi_rows,
                mi_cols,
                r + dr,
                c + dc,
                subsize,
            );
        }
    }

    for r in (0..mi_rows).step_by(8) {
        for c in (0..mi_cols).step_by(8) {
            walk(
                &mut partitions,
                &mut hints,
                &sub_hints,
                &field,
                mi_rows,
                mi_cols,
                r,
                c,
                BLOCK_64X64,
            );
        }
    }
    (partitions, hints, sub_hints)
}

/// One quantized transform block of an inter leaf: `(plane, start_x,
/// start_y, tx_size, tokens)` in §6.4.21 walk order.
type LeafTokenBlock = (usize, u32, u32, u32, Vec<i64>);

/// Elect the **per-block inter transform size** of one non-skip leaf:
/// trial forward-DCT + §8.6 quantization at every §6.4.10-codeable luma
/// size (`TX_4X4` up to `MAX_TXSIZE_LOOKUP[ MiSize ]`, chroma following
/// §6.4.22), costing each candidate by its total nonzero-token count
/// plus one per coded transform block (the per-block `more_coefs` /
/// EOB overhead proxy). Ties prefer the larger transform. Inter blocks
/// are `DCT_DCT` at every size (§6.4.25).
///
/// `work` must already hold the leaf's §8.5.2 prediction. Returns the
/// winning `(tx_size, blocks, all_zero)`; the caller replays the
/// decoder's §8.6.2 reconstruction with `blocks` (or elects `skip` when
/// `all_zero`).
#[allow(clippy::too_many_arguments)]
fn select_inter_leaf_tx(
    targets: &[Plane; 3],
    work: &[Plane; 3],
    r: u32,
    c: u32,
    mi_size: u8,
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    seg: &SegmentationParams,
    quant: &QuantizationParams,
) -> (u32, Vec<LeafTokenBlock>, bool) {
    use crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP;
    use crate::residual::MAX_TXSIZE_LOOKUP;

    let bd8 = bit_depth as u8;
    let num8x8 = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize]);
    let max_tx = MAX_TXSIZE_LOOKUP[mi_size as usize];

    let mut best: Option<(u64, u32, Vec<LeafTokenBlock>, bool)> = None;
    // Larger candidates first so a cost tie keeps the larger transform
    // (fewer coded blocks, fewer per-block overheads).
    for cand in (0..=max_tx).rev() {
        let mut cost = 0u64;
        let mut blocks: Vec<LeafTokenBlock> = Vec::new();
        let mut all_zero = true;
        // The §6.4.21 loop is keyed by `plane` and indexes the planes /
        // subsampling by that same index, mirroring the spec listing.
        #[allow(clippy::needless_range_loop)]
        for plane in 0..3usize {
            let tx_sz = if plane == 0 {
                cand
            } else {
                crate::residual::get_uv_tx_size(cand, mi_size, ssx, ssy)
            };
            let sub_x = plane > 0 && ssx;
            let sub_y = plane > 0 && ssy;
            let base_x = (c * 8) >> u32::from(sub_x);
            let base_y = (r * 8) >> u32::from(sub_y);
            let maxx = (mi_cols * 8) >> u32::from(sub_x);
            let maxy = (mi_rows * 8) >> u32::from(sub_y);
            let n0 = 4u32 << tx_sz;
            let region_w = (num8x8 * 8) >> u32::from(sub_x);
            let region_h = (num8x8 * 8) >> u32::from(sub_y);
            let dc_q = get_dc_quant(plane, seg, quant, 0, bd8);
            let ac_q = get_ac_quant(plane, seg, quant, 0, bd8);
            let mut sy = base_y;
            while sy < base_y + region_h {
                let mut sx = base_x;
                while sx < base_x + region_w {
                    if sx < maxx && sy < maxy {
                        let n0u = n0 as usize;
                        let mut block = vec![0i64; n0u * n0u];
                        for i in 0..n0u {
                            for j in 0..n0u {
                                let t = targets[plane].get(sx as usize + j, sy as usize + i);
                                let p = work[plane].get(sx as usize + j, sy as usize + i);
                                block[i * n0u + j] = i64::from(t) - i64::from(p);
                            }
                        }
                        // §6.4.25: inter blocks are DCT_DCT at every size.
                        crate::fwd_transform::forward_dct_2d(&mut block, tx_sz + 2);
                        crate::fwd_transform::quantize_block_tx(
                            &mut block, dc_q, ac_q, tx_sz, bit_depth,
                        );
                        let nonzero = block.iter().filter(|&&v| v != 0).count() as u64;
                        all_zero &= nonzero == 0;
                        cost += nonzero + 1;
                        blocks.push((plane, sx, sy, tx_sz, block));
                    }
                    sx += n0;
                }
                sy += n0;
            }
        }
        let better = match &best {
            None => true,
            Some((bc, ..)) => cost < *bc,
        };
        if better {
            best = Some((cost, cand, blocks, all_zero));
        }
    }
    let (_, tx, blocks, all_zero) = best.expect("at least one tx candidate");
    (tx, blocks, all_zero)
}

/// Sum of squared `target − work` errors over one leaf's region, all
/// three planes, clipped to the MI-padded plane extents.
#[allow(clippy::too_many_arguments)]
fn leaf_sse(
    targets: &[Plane; 3],
    work: &[Plane; 3],
    r: u32,
    c: u32,
    mi_size: u8,
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
) -> u64 {
    use crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP;
    let num8x8 = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize]);
    let mut sse = 0u64;
    for plane in 0..3usize {
        let sub_x = plane > 0 && ssx;
        let sub_y = plane > 0 && ssy;
        let base_x = ((c * 8) >> u32::from(sub_x)) as usize;
        let base_y = ((r * 8) >> u32::from(sub_y)) as usize;
        let maxx = ((mi_cols * 8) >> u32::from(sub_x)) as usize;
        let maxy = ((mi_rows * 8) >> u32::from(sub_y)) as usize;
        let region_w = ((num8x8 * 8) >> u32::from(sub_x)) as usize;
        let region_h = ((num8x8 * 8) >> u32::from(sub_y)) as usize;
        for i in 0..region_h {
            for j in 0..region_w {
                let (x, y) = (base_x + j, base_y + i);
                if x < maxx && y < maxy {
                    let d = i64::from(targets[plane].get(x, y)) - i64::from(work[plane].get(x, y));
                    sse += (d * d) as u64;
                }
            }
        }
    }
    sse
}

/// Luma-only SAD of `target − work` over one leaf's region, clipped to
/// the MI-padded plane extents (the compound-candidate score).
fn leaf_luma_sad(
    targets: &[Plane; 3],
    work: &[Plane; 3],
    r: u32,
    c: u32,
    mi_size: u8,
    mi_cols: u32,
    mi_rows: u32,
) -> u64 {
    use crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP;
    let num8x8 = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize]);
    let base_x = (c * 8) as usize;
    let base_y = (r * 8) as usize;
    let maxx = (mi_cols * 8) as usize;
    let maxy = (mi_rows * 8) as usize;
    let region = (num8x8 * 8) as usize;
    let mut sad = 0u64;
    for i in 0..region {
        for j in 0..region {
            let (x, y) = (base_x + j, base_y + i);
            if x < maxx && y < maxy {
                let d = targets[0].get(x, y) - work[0].get(x, y);
                sad += d.unsigned_abs() as u64;
            }
        }
    }
    sad
}

/// Whether the transform block starting at plane coordinates
/// `(sx, sy)` of `plane` lies inside the leaf at MI `(r, c)` of size
/// `mi_size` (used to drop a reverted leaf's cached tokens).
// Spec-shaped geometry fan-in, matching the crate's §6.4.21 helpers.
#[allow(clippy::too_many_arguments)]
fn leaf_contains(
    r: u32,
    c: u32,
    mi_size: u8,
    ssx: bool,
    ssy: bool,
    plane: usize,
    sx: u32,
    sy: u32,
) -> bool {
    use crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP;
    let num8x8 = u32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[mi_size as usize]);
    let sub_x = plane > 0 && ssx;
    let sub_y = plane > 0 && ssy;
    let base_x = (c * 8) >> u32::from(sub_x);
    let base_y = (r * 8) >> u32::from(sub_y);
    let region_w = (num8x8 * 8) >> u32::from(sub_x);
    let region_h = (num8x8 * 8) >> u32::from(sub_y);
    sx >= base_x && sx < base_x + region_w && sy >= base_y && sy < base_y + region_h
}

/// Plan + reconstruct one **elected sub-8x8 leaf** of the lossy tree
/// encoder: build the §6.4.16 per-cell spec from the planner's
/// [`Sub8x8Hint`] (per-cell `NEWMV` for searched vectors — snapped
/// §6.4.20-codeable against the leaf's §6.5.12 `BestMv` under the
/// §6.5.13 `use_mv_hp` gate, exactly as the writer will verify —
/// `ZEROMV` otherwise), predict via the decoder-mirror per-`blockIdx`
/// §8.5.2 walk, elect skip at the forced `TX_4X4` (§6.4.10 codes no tx
/// bits below 8x8), replay the §8.6.2 reconstruction for coded blocks,
/// and apply the same strict-SSE-improvement skip guard as `>= 8x8`
/// leaves. Single-reference `LAST` (the planner's motion field).
#[allow(clippy::too_many_arguments)]
fn plan_sub8x8_leaf(
    hint: &Sub8x8Hint,
    allow_high_precision_mv: bool,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
    r: u32,
    c: u32,
    subsize: u8,
    mi_cols: u32,
    mi_rows: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    seg: &SegmentationParams,
    quant: &QuantizationParams,
    sign_bias: &[bool; 4],
    state: &crate::decode_block::Vp9FrameState,
    work: &mut ReconState,
    token_cache: &mut std::collections::HashMap<(usize, u32, u32), Vec<i64>>,
) -> crate::frame_writer::InterTreeLeaf {
    use crate::frame_writer::InterTreeLeaf;
    use crate::inter_block_writer::InterSubBlockSpec;
    use crate::inter_decode::FrameStateMvSource;
    use crate::mode_info::{LAST_FRAME, NEWMV, NONE_REF_FRAME, ZEROMV};
    use crate::mv::use_mv_hp;
    use crate::mv_ref::MvRefGeometry;
    use crate::residual::{BLOCK_4X4, BLOCK_4X8, BLOCK_8X4};

    // The §6.4.16 walk's visited cells for this shape.
    let cells: &[usize] = match subsize {
        BLOCK_4X4 => &[0, 1, 2, 3],
        BLOCK_4X8 => &[0, 1],
        BLOCK_8X4 => &[0, 2],
        _ => unreachable!("sub-8x8 leaf with MiSize >= BLOCK_8X8"),
    };

    // §6.5.12 BestMv at the leaf's own geometry (the writer derives the
    // identical value from the shared state).
    let geom = MvRefGeometry {
        mi_row: r as i32,
        mi_col: c as i32,
        mi_rows: mi_rows as i32,
        mi_cols: mi_cols as i32,
        mi_size: subsize as usize,
        mi_col_start: 0,
        mi_col_end: mi_cols as i32,
    };
    let src = FrameStateMvSource::new(state, None);
    let mv_refs = geom.find_mv_refs(&src, LAST_FRAME, -1, sign_bias, false);
    let best = geom.find_best_ref_mvs(mv_refs.ref_list_mv, allow_high_precision_mv)[0];
    let use_hp = allow_high_precision_mv && use_mv_hp(best);

    let mut modes = [ZEROMV; 4];
    let mut mvs = [[[0i32; 2]; 2]; 4];
    for &cell in cells {
        let (dy, dx) = hint.cell_mvs[cell];
        if (dy, dx) != (0, 0) {
            let mut m = [8 * dy, 8 * dx];
            for (comp, mm) in m.iter_mut().enumerate() {
                let d = *mm - best[comp];
                if d != 0 && !use_hp && (d & 1) != 0 {
                    // Only even-magnitude differences are codeable
                    // without the hp bit; nudge by one eighth-pel.
                    *mm -= 1;
                }
            }
            modes[cell] = NEWMV;
            mvs[cell][0] = m;
        }
    }
    let sub = InterSubBlockSpec { modes, mvs };

    // Decoder-mirror per-blockIdx §8.5.2 prediction.
    let block_mvs = sub8x8_block_mvs(subsize, &sub);
    predict_inter_leaf_sub8x8(
        &mut work.planes,
        reference,
        None,
        ref_w,
        ref_h,
        r,
        c,
        subsize,
        &block_mvs,
        mi_cols,
        mi_rows,
        ssx,
        ssy,
        bit_depth,
    );

    // Trial quantization at the forced TX_4X4 (the only §6.4.10
    // candidate below 8x8), then the skip / strict-SSE-guard election.
    let (tx, blocks, all_zero) = select_inter_leaf_tx(
        targets,
        &work.planes,
        r,
        c,
        subsize,
        mi_cols,
        mi_rows,
        ssx,
        ssy,
        bit_depth,
        seg,
        quant,
    );
    debug_assert_eq!(tx, 0, "sub-8x8 leaves are TX_4X4 only");
    let mut skip = all_zero;
    if !skip {
        let sse_skip = leaf_sse(
            targets,
            &work.planes,
            r,
            c,
            subsize,
            mi_cols,
            mi_rows,
            ssx,
            ssy,
        );
        for (plane, sx, sy, tx_sz, block) in blocks {
            let dc_q = get_dc_quant(plane, seg, quant, 0, bit_depth as u8);
            let ac_q = get_ac_quant(plane, seg, quant, 0, bit_depth as u8);
            reconstruct_block(
                &mut work.planes[plane],
                sx as usize,
                sy as usize,
                tx_sz,
                &block,
                dc_q,
                ac_q,
                DCT_DCT,
                false,
                bit_depth,
            );
            token_cache.insert((plane, sx, sy), block);
        }
        let sse_coded = leaf_sse(
            targets,
            &work.planes,
            r,
            c,
            subsize,
            mi_cols,
            mi_rows,
            ssx,
            ssy,
        );
        if sse_coded >= sse_skip {
            token_cache
                .retain(|&(p, sx, sy), _| !leaf_contains(r, c, subsize, ssx, ssy, p, sx, sy));
            predict_inter_leaf_sub8x8(
                &mut work.planes,
                reference,
                None,
                ref_w,
                ref_h,
                r,
                c,
                subsize,
                &block_mvs,
                mi_cols,
                mi_rows,
                ssx,
                ssy,
                bit_depth,
            );
            skip = true;
        }
    }
    InterTreeLeaf {
        mi_size: subsize,
        tx_size: 0,
        y_mode: ZEROMV, // ignored for sub-8x8 (per-cell modes apply)
        interp_filter: 0,
        ref_frame: [LAST_FRAME, NONE_REF_FRAME],
        mv: [[0, 0], [0, 0]],
        skip,
        segment_id: 0,
        sub: Some(sub),
    }
}

/// Encode one **lossy** P-frame over a content-adaptive §6.4.3 partition
/// tree — [`encode_pframe_lossy_motion`] generalised past the fixed
/// all-`BLOCK_8X8` / inferred-`TX_8X8` layout:
///
/// * the partition tree merges uniform-motion regions into leaves up to
///   `BLOCK_64X64` ([`plan_inter_partitions`]);
/// * each leaf codes `ZEROMV` or `NEWMV` from the integer full search
///   (the §6.5.12 `BestMv` is derived with the shared `find_mv_refs` /
///   `find_best_ref_mvs` over the same `Vp9FrameState` the writer codes
///   against, with the §6.4.20 codeability snap under the §6.5.13
///   `use_mv_hp` gate);
/// * the frame codes `tx_mode = TX_MODE_SELECT` and every non-skip leaf
///   elects its **own** §6.4.10 transform size by trial quantization
///   ([`select_inter_leaf_tx`]);
/// * per-leaf `skip` election: a leaf whose quantized residual is
///   all-zero codes no residual (and carries the §6.4.10 inferred tx
///   size, since `read_tx_size( allowSelect = !skip )` codes nothing);
/// * non-skip leaves replay the decoder's §8.6.2 reconstruction in
///   place, so the returned [`ReconState`] equals the decoder's output
///   bit-for-bit and its visible crop is the next frame's reference.
///
/// With `subpel == true` each leaf's motion vector is additionally
/// refined at half- / quarter- / (under the hp gate) eighth-pel
/// precision against the decoder-mirror §8.5.2 interpolation
/// ([`refine_leaf_mv_subpel`]), including pure-sub-pel probes on leaves
/// whose integer winner is `(0, 0)`.
///
/// With `golden == Some( planes )` the encoder is **multi-reference**:
/// every leaf evaluates both `LAST` (`reference`) and `GOLDEN`
/// (`golden`) — ZEROMV error plus a searched/refined NEWMV per
/// reference — and codes the §6.4.17 `ref_frame` of the winner. The
/// header's `ref_frame_idx` must map `GOLDEN` to a slot holding the
/// `golden` planes (the sequence encoders park the keyframe there as a
/// long-term reference).
///
/// With `sub8x8 == true` the planner additionally probes each 8x8
/// cell's four 4x4 quadrants and elects a below-8x8 leaf (4x4 / 4x8 /
/// 8x4 per the agreeing quadrant pairs) where divergent per-quadrant
/// motion beats the best single vector by the per-coded-vector margin
/// ([`plan_inter_partitions`]); elected leaves code the §6.4.16
/// per-`(idy, idx)` walk (per-cell `NEWMV` / `ZEROMV` over `LAST`,
/// snapped §6.4.20-codeable against the leaf's §6.5.12 `BestMv`),
/// predict via the decoder-mirror per-`blockIdx` §8.5.2 walk, and run
/// the same skip / SSE-guard election as `>= 8x8` leaves at the forced
/// `TX_4X4`.
///
/// Requires an error-resilient non-key lossy header (see
/// [`crate::frame_writer::assemble_inter_frame_tree`]).
// Spec-shaped fan-in (header + targets + per-reference planes + search
// options), matching the crate's encoder-driver style. Bytes+recon
// convenience over the `_with_state` encoder; the non-test encoders
// all thread the state, so only tests (and the fixture builders in
// them) call this.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn encode_pframe_lossy_tree_motion(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    golden: Option<&[(&[i32], usize); 3]>,
    ref_w: u32,
    ref_h: u32,
    search_range: i32,
    subpel: bool,
    sub8x8: bool,
) -> Result<(Vec<u8>, ReconState), Error> {
    encode_pframe_lossy_tree_motion_with_state(
        hdr,
        targets,
        reference,
        golden,
        ref_w,
        ref_h,
        search_range,
        subpel,
        sub8x8,
    )
    .map(|(b, r, _)| (b, r))
}

/// [`encode_pframe_lossy_tree_motion`] also returning the writer's final
/// §6.4.4 [`crate::decode_block::Vp9FrameState`] per-MI arrays — the
/// input the encode-side §8.8 loop-filter mirror consumes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_pframe_lossy_tree_motion_with_state(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    golden: Option<&[(&[i32], usize); 3]>,
    ref_w: u32,
    ref_h: u32,
    search_range: i32,
    subpel: bool,
    sub8x8: bool,
) -> Result<(Vec<u8>, ReconState, crate::decode_block::Vp9FrameState), Error> {
    use crate::frame_writer::{InterFrameTreePlan, InterTreeLeaf, InterTreePlanner};
    use crate::inter_decode::FrameStateMvSource;
    use crate::mode_info::{LAST_FRAME, NEARESTMV, NEARMV, NEWMV, NONE_REF_FRAME, ZEROMV};
    use crate::mv::use_mv_hp;
    use crate::mv_ref::MvRefGeometry;
    use crate::residual::MAX_TXSIZE_LOOKUP;
    use std::cell::RefCell;

    if hdr.frame_type != FrameType::NonKeyFrame || hdr.quantization.lossless {
        return Err(Error::Unsupported);
    }
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);
    // §7.2 setup_past_independence( ): error-resilient frames have
    // all-zero *effective* sign biases (see assemble_inter_frame_tree).
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
    let seg = hdr.segmentation;
    let quant = hdr.quantization;

    let (partitions, _hints, sub_hints) = plan_inter_partitions(
        targets,
        reference,
        ref_w,
        ref_h,
        mi_cols,
        mi_rows,
        search_range,
        sub8x8,
    );

    let work = RefCell::new(ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth));
    let token_cache: RefCell<std::collections::HashMap<(usize, u32, u32), Vec<i64>>> =
        RefCell::new(std::collections::HashMap::new());
    let scratch = RefCell::new(Plane::new((mi_cols * 8) as usize, (mi_rows * 8) as usize));
    let scratch3 = RefCell::new(ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth).planes);

    // Compound prediction needs a second reference *and* the §6.3.12
    // sign-bias asymmetry (compoundReferenceAllowed). The candidate
    // pair coded is [ LAST, ALTREF ], which is the §6.3.18
    // fixed/variable layout exactly when LAST and GOLDEN share a bias
    // that ALTREF does not.
    let compound_ok =
        golden.is_some() && sign_bias[1] == sign_bias[2] && sign_bias[3] != sign_bias[1];
    let reference_mode = if compound_ok {
        crate::compressed::ReferenceMode::ReferenceModeSelect
    } else {
        crate::compressed::ReferenceMode::SingleReference
    };

    let tx_mode = crate::compressed::TxMode::TxModeSelect;
    const NEWMV_SAD_MARGIN: u64 = 64;

    let mut planner: Box<InterTreePlanner<'_>> =
        Box::new(|r: u32, c: u32, subsize: u8, state| -> InterTreeLeaf {
            // Elected sub-8x8 leaves: the §6.4.16 per-cell walk over the
            // planner's quadrant hints (single-reference LAST).
            if subsize < crate::residual::BLOCK_8X8 {
                let hint = *sub_hints
                    .get(&(r, c))
                    .expect("sub-8x8 leaf without an elected hint");
                return plan_sub8x8_leaf(
                    &hint,
                    hdr.allow_high_precision_mv,
                    targets,
                    reference,
                    ref_w,
                    ref_h,
                    r,
                    c,
                    subsize,
                    mi_cols,
                    mi_rows,
                    ssx,
                    ssy,
                    bit_depth,
                    &seg,
                    &quant,
                    &sign_bias,
                    state,
                    &mut work.borrow_mut(),
                    &mut token_cache.borrow_mut(),
                );
            }
            let max_tx = MAX_TXSIZE_LOOKUP[subsize as usize];
            let num8x8 = u32::from(crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP[subsize as usize]);

            // Evaluate one reference frame: leaf-level ZEROMV error plus
            // (when profitable) a snapped / sub-pel-refined NEWMV
            // candidate. The §6.5 predictors come from the shared state
            // with the candidate's own `ref_frame`, exactly as the
            // writer will derive them.
            let eval_ref =
                |ref_frame: i32, planes: &[(&[i32], usize); 3]| -> (u64, Option<([i32; 2], u64)>) {
                    let ((dy, dx), int_sad, zero_sad) = search_block_mv_wh(
                        &targets[0],
                        planes[0].0,
                        planes[0].1,
                        ref_w as i32,
                        ref_h as i32,
                        (c * 8) as i32,
                        (r * 8) as i32,
                        (num8x8 * 8) as i32,
                        (num8x8 * 8) as i32,
                        search_range,
                    );
                    let int_winner = (dy, dx) != (0, 0) && int_sad + NEWMV_SAD_MARGIN < zero_sad;
                    let probe = subpel && search_range > 0 && zero_sad > NEWMV_SAD_MARGIN;
                    if !int_winner && !probe {
                        return (zero_sad, None);
                    }
                    let geom = MvRefGeometry {
                        mi_row: r as i32,
                        mi_col: c as i32,
                        mi_rows: mi_rows as i32,
                        mi_cols: mi_cols as i32,
                        mi_size: subsize as usize,
                        mi_col_start: 0,
                        mi_col_end: mi_cols as i32,
                    };
                    let src = FrameStateMvSource::new(state, None);
                    let mv_refs = geom.find_mv_refs(&src, ref_frame, -1, &sign_bias, false);
                    let best =
                        geom.find_best_ref_mvs(mv_refs.ref_list_mv, hdr.allow_high_precision_mv)[0];

                    let mut mv = if int_winner { [8 * dy, 8 * dx] } else { [0, 0] };
                    let use_hp = hdr.allow_high_precision_mv && use_mv_hp(best);
                    for (comp, m) in mv.iter_mut().enumerate() {
                        let d = *m - best[comp];
                        if d != 0 && !use_hp && (d & 1) != 0 {
                            // Only even-magnitude differences are codeable
                            // without the hp bit; nudge by one eighth-pel.
                            *m -= 1;
                        }
                    }
                    if subpel {
                        let (refined, refined_sad) = refine_leaf_mv_subpel(
                            targets,
                            planes,
                            ref_w,
                            ref_h,
                            r,
                            c,
                            subsize,
                            mv,
                            use_hp,
                            mi_cols,
                            mi_rows,
                            bit_depth,
                            &mut scratch.borrow_mut(),
                        );
                        if refined != [0, 0] {
                            (zero_sad, Some((refined, refined_sad)))
                        } else {
                            (zero_sad, None)
                        }
                    } else if int_winner {
                        (zero_sad, Some((mv, int_sad)))
                    } else {
                        (zero_sad, None)
                    }
                };

            // Candidate sweep over the available references (and, when
            // the sign biases admit it, the compound pair): score =
            // SAD + the mode's syntax margin; ties keep the earlier
            // (cheaper-to-code) candidate.
            let mut choice: ([i32; 2], u8, [[i32; 2]; 2]) =
                ([LAST_FRAME, NONE_REF_FRAME], ZEROMV, [[0, 0], [0, 0]]);
            {
                let (l_zero, l_new) = eval_ref(LAST_FRAME, reference);
                let mut best_score = l_zero;
                if let Some((mv, sad)) = l_new {
                    if search_range > 0 && sad + NEWMV_SAD_MARGIN < best_score {
                        best_score = sad + NEWMV_SAD_MARGIN;
                        choice = ([LAST_FRAME, NONE_REF_FRAME], NEWMV, [mv, [0, 0]]);
                    }
                }
                let mut g_new_mv: Option<[i32; 2]> = None;
                if let Some(gplanes) = golden {
                    let (g_zero, g_new) = eval_ref(crate::mode_info::GOLDEN_FRAME, gplanes);
                    if g_zero < best_score {
                        best_score = g_zero;
                        choice = (
                            [crate::mode_info::GOLDEN_FRAME, NONE_REF_FRAME],
                            ZEROMV,
                            [[0, 0], [0, 0]],
                        );
                    }
                    if let Some((mv, sad)) = g_new {
                        g_new_mv = Some(mv);
                        if search_range > 0 && sad + NEWMV_SAD_MARGIN < best_score {
                            best_score = sad + NEWMV_SAD_MARGIN;
                            choice = (
                                [crate::mode_info::GOLDEN_FRAME, NONE_REF_FRAME],
                                NEWMV,
                                [mv, [0, 0]],
                            );
                        }
                    }
                }
                // Compound (LAST + ALTREF) candidates — the §8.5.2
                // Round2( p0 + p1, 1 ) average of both references
                // (ALTREF resolves to the same slot as GOLDEN in the
                // sequence layout). ZEROMV averages the co-located
                // blocks; NEWMV pairs the per-reference winners, each
                // re-snapped against its own list's §6.5.12 BestMv.
                if compound_ok {
                    let gplanes = golden.expect("compound requires golden planes");
                    let comp_pair = [LAST_FRAME, crate::mode_info::ALTREF_FRAME];
                    let comp_sad = |mvs: [[i32; 2]; 2]| -> u64 {
                        let mut sc = scratch3.borrow_mut();
                        predict_inter_leaf2(
                            &mut sc,
                            reference,
                            Some(gplanes),
                            ref_w,
                            ref_h,
                            r,
                            c,
                            subsize,
                            mvs,
                            mi_cols,
                            mi_rows,
                            ssx,
                            ssy,
                            bit_depth,
                        );
                        leaf_luma_sad(targets, &sc, r, c, subsize, mi_cols, mi_rows)
                    };
                    let zero_pair = [[0, 0], [0, 0]];
                    let cz = comp_sad(zero_pair);
                    // The comp-mode + second-ref syntax is a few bits;
                    // a token margin keeps ties on the single path.
                    if cz + 8 < best_score {
                        best_score = cz + 8;
                        choice = (comp_pair, ZEROMV, zero_pair);
                    }
                    let l_mv = l_new.map(|(mv, _)| mv).unwrap_or([0, 0]);
                    let a_mv = g_new_mv.unwrap_or([0, 0]);
                    if search_range > 0 && (l_mv != [0, 0] || a_mv != [0, 0]) {
                        // Re-snap the ALTREF-list vector against the
                        // ALTREF predictors (its parity gate may differ
                        // from GOLDEN's).
                        let geom = MvRefGeometry {
                            mi_row: r as i32,
                            mi_col: c as i32,
                            mi_rows: mi_rows as i32,
                            mi_cols: mi_cols as i32,
                            mi_size: subsize as usize,
                            mi_col_start: 0,
                            mi_col_end: mi_cols as i32,
                        };
                        let src = FrameStateMvSource::new(state, None);
                        let mv_refs = geom.find_mv_refs(
                            &src,
                            crate::mode_info::ALTREF_FRAME,
                            -1,
                            &sign_bias,
                            false,
                        );
                        let best_a = geom
                            .find_best_ref_mvs(mv_refs.ref_list_mv, hdr.allow_high_precision_mv)[0];
                        let use_hp_a = hdr.allow_high_precision_mv && use_mv_hp(best_a);
                        let mut a_mv = a_mv;
                        for (comp, m) in a_mv.iter_mut().enumerate() {
                            let d = *m - best_a[comp];
                            if d != 0 && !use_hp_a && (d & 1) != 0 {
                                *m -= 1;
                            }
                        }
                        let pair = [l_mv, a_mv];
                        let cn = comp_sad(pair);
                        if cn + 2 * NEWMV_SAD_MARGIN < best_score {
                            choice = (comp_pair, NEWMV, pair);
                        }
                    }
                }
            }
            let is_compound = choice.0[1] > crate::mode_info::INTRA_FRAME;
            // §6.4.16 mode mapping: a NEWMV whose vector(s) equal a §6.5
            // predictor codes the predictor-referencing mode instead —
            // NEARESTMV / NEARMV carry **no** §6.4.20 mv-diff bits (the
            // decoder recovers the MV from the same shared `find_mv_refs`
            // / `find_best_ref_mvs` scan the writer verifies against).
            if choice.1 == NEWMV {
                let geom = MvRefGeometry {
                    mi_row: r as i32,
                    mi_col: c as i32,
                    mi_rows: mi_rows as i32,
                    mi_cols: mi_cols as i32,
                    mi_size: subsize as usize,
                    mi_col_start: 0,
                    mi_col_end: mi_cols as i32,
                };
                let src = FrameStateMvSource::new(state, None);
                let lists = 1 + usize::from(is_compound);
                let mut nearest_all = true;
                let mut near_all = true;
                for j in 0..lists {
                    let mv_refs = geom.find_mv_refs(&src, choice.0[j], -1, &sign_bias, false);
                    let best =
                        geom.find_best_ref_mvs(mv_refs.ref_list_mv, hdr.allow_high_precision_mv);
                    nearest_all &= choice.2[j] == best[0];
                    near_all &= choice.2[j] == best[1];
                }
                if nearest_all {
                    choice.1 = NEARESTMV;
                } else if near_all {
                    choice.1 = NEARMV;
                }
            }
            let ref_planes: &[(&[i32], usize); 3] = if choice.0[0] == LAST_FRAME {
                reference
            } else {
                golden.expect("GOLDEN elected only when present")
            };
            let second_planes: Option<&[(&[i32], usize); 3]> = if is_compound {
                Some(golden.expect("compound requires golden planes"))
            } else {
                None
            };

            let mut work = work.borrow_mut();
            predict_inter_leaf2(
                &mut work.planes,
                ref_planes,
                second_planes,
                ref_w,
                ref_h,
                r,
                c,
                subsize,
                choice.2,
                mi_cols,
                mi_rows,
                ssx,
                ssy,
                bit_depth,
            );

            // Per-leaf transform-size election + skip election, then the
            // decoder-mirror reconstruction replay for non-skip leaves.
            let (tx, blocks, all_zero) = select_inter_leaf_tx(
                targets,
                &work.planes,
                r,
                c,
                subsize,
                mi_cols,
                mi_rows,
                ssx,
                ssy,
                bit_depth,
                &seg,
                &quant,
            );
            let mut skip = all_zero;
            let bd8 = hdr.color_config.bit_depth;
            if !skip {
                // Prediction-vs-target SSE before coding: the distortion
                // of electing `skip`.
                let sse_skip = leaf_sse(
                    targets,
                    &work.planes,
                    r,
                    c,
                    subsize,
                    mi_cols,
                    mi_rows,
                    ssx,
                    ssy,
                );
                {
                    let mut cache = token_cache.borrow_mut();
                    for (plane, sx, sy, tx_sz, block) in blocks {
                        let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
                        let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
                        reconstruct_block(
                            &mut work.planes[plane],
                            sx as usize,
                            sy as usize,
                            tx_sz,
                            &block,
                            dc_q,
                            ac_q,
                            DCT_DCT,
                            false,
                            bit_depth,
                        );
                        cache.insert((plane, sx, sy), block);
                    }
                }
                // Skip-if-no-gain guard: coding the residual must
                // *strictly* reduce the leaf's SSE, else the bytes are
                // pure waste — and, chain-wise, a block whose coded
                // reconstruction is no closer than its prediction would
                // re-code equivalent noise every frame (the quantized
                // fixed point is never reached). Electing skip instead
                // makes the static-content chain monotone: a leaf's
                // reconstruction only ever changes when it strictly
                // improves.
                let sse_coded = leaf_sse(
                    targets,
                    &work.planes,
                    r,
                    c,
                    subsize,
                    mi_cols,
                    mi_rows,
                    ssx,
                    ssy,
                );
                if sse_coded >= sse_skip {
                    // Revert: restore the pure §8.5.2 prediction (the
                    // reconstruction overwrote it) and drop the tokens.
                    token_cache.borrow_mut().retain(|&(p, sx, sy), _| {
                        !leaf_contains(r, c, subsize, ssx, ssy, p, sx, sy)
                    });
                    predict_inter_leaf2(
                        &mut work.planes,
                        ref_planes,
                        second_planes,
                        ref_w,
                        ref_h,
                        r,
                        c,
                        subsize,
                        choice.2,
                        mi_cols,
                        mi_rows,
                        ssx,
                        ssy,
                        bit_depth,
                    );
                    skip = true;
                }
            }
            // Skip blocks reconstruct from prediction alone — the work
            // planes already hold it. §6.4.10: a skip block never codes
            // tx bits, so it carries the inferred size (== max_tx under
            // TX_MODE_SELECT).
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: if skip { max_tx } else { tx },
                y_mode: choice.1,
                interp_filter: 0,
                ref_frame: choice.0,
                mv: choice.2,
                skip,
                segment_id: 0,
                sub: None,
            }
        });

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        |_mi_r: u32, _mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            token_cache
                .borrow_mut()
                .remove(&(plane, sx, sy))
                .expect("planner pre-computed this block's tokens")
        },
    );

    // A non-error-resilient header is this encoder's assertion that the
    // §7.2.6 UsePrevFrameMvs derivation yields 0 for the frame (hidden
    // / intra / differently-sized predecessor) — see the function docs.
    let plan = InterFrameTreePlan {
        tx_mode,
        reference_mode,
        partitions,
        prev_segment_ids: None,
        prev_frame_mvs_absent: !hdr.error_resilient_mode,
        prev_frame_mvs: None,
    };
    let (bytes, state) = crate::frame_writer::assemble_inter_frame_tree_with_state(
        hdr,
        &plan,
        &mut *planner,
        &mut *coeffs,
    )?;
    drop(planner);
    drop(coeffs);
    Ok((bytes, work.into_inner(), state))
}

/// Close out one lossy frame with the §8.8 encode-side loop filter:
/// elect the frame `loop_filter_level` from the level-0 encode products
/// ([`crate::recon_filter::elect_filter_level`] — the level never
/// changes the reconstruction, only the header field and the §8.8
/// post-pass), re-assemble the frame with the elected level when it is
/// non-zero (byte-deterministic: only the §6.2.8 fixed-width level
/// field changes, so the stream length is invariant), and apply the
/// §8.8 chain to the reconstruction exactly as every conforming
/// decoder will before §8.10 stores it — the filtered planes are what
/// the next frame must reference.
///
/// `hdr` is the level-0 header the `bytes0` encode used; `re_encode`
/// re-runs the identical (deterministic) assembly under the substituted
/// header.
// Spec-shaped fan-in (header + the three level-0 encode products +
// source/extent scoring inputs), matching the crate's encoder-driver
// style.
#[allow(clippy::too_many_arguments)]
fn finish_frame_with_filter(
    hdr: &Vp9FrameHeader,
    bytes0: Vec<u8>,
    recon0: ReconState,
    state0: crate::decode_block::Vp9FrameState,
    targets: &[Plane; 3],
    vis_w: usize,
    vis_h: usize,
    re_encode: impl FnOnce(
        &Vp9FrameHeader,
    )
        -> Result<(Vec<u8>, ReconState, crate::decode_block::Vp9FrameState), Error>,
) -> Result<(Vec<u8>, ReconState), Error> {
    use crate::recon_filter::{elect_filter_params, filter_reconstruction};

    let (level, sharpness) = elect_filter_params(&recon0, &state0, hdr, targets, vis_w, vis_h);
    if level == 0 {
        // §8.1 step 2: level 0 codes no filtering — the level-0 encode
        // is already final and the reconstruction stays raw.
        return Ok((bytes0, recon0));
    }
    let mut hdr2 = *hdr;
    hdr2.loop_filter.level = level;
    hdr2.loop_filter.sharpness = sharpness;
    let (bytes, mut recon, state) = re_encode(&hdr2)?;
    debug_assert_eq!(
        bytes.len(),
        bytes0.len(),
        "§6.2.8 filter_level / sharpness are fixed-width; re-assembly must not change the length"
    );
    filter_reconstruction(&mut recon, &state, &hdr2);
    Ok((bytes, recon))
}

/// Encode a sequence of 8-bit 4:2:0 planar frames into a **lossy** VP9
/// stream at quantizer index `base_q_idx` (`1..=255`): a lossy keyframe
/// followed by lossy P-frames with per-block `ZEROMV` / `NEWMV` motion,
/// each referencing the previous frame's in-loop **reconstruction** (the
/// decoder's exact output), so encoder and decoder never drift.
///
/// Every frame runs the encode-side §8.8 loop filter with a per-frame
/// **elected** `loop_filter_level` (see [`finish_frame_with_filter`]):
/// the reference chain threads the *filtered* reconstructions, exactly
/// mirroring the §8.10 post-filter frame store every conforming decoder
/// keeps.
pub(crate) fn encode_sequence_lossy_420(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    if frames.is_empty() || base_q_idx == 0 {
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
    // §8.10 FrameStore crop of a reconstruction: the visible extents.
    let visible_crop = |recon: &ReconState| -> [Vec<i32>; 3] {
        let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(vw * vh);
            for y in 0..vh {
                for x in 0..vw {
                    out.push(p.get(x, y));
                }
            }
            out
        };
        [
            crop(&recon.planes[0], w, h),
            crop(&recon.planes[1], cw, ch),
            crop(&recon.planes[2], cw, ch),
        ]
    };

    // Lossy keyframe over the content-adaptive partition/tx tree, with
    // the elected §8.8 filter level and the filtered reconstruction.
    let kf_targets = padded_targets(frames[0]);
    let kf_hdr = lossy_keyframe_header_420(width, height, base_q_idx);
    let kf_plan = plan_keyframe_tree(
        &kf_targets,
        mi_rows as u32,
        mi_cols as u32,
        true,
        true,
        8,
        base_q_idx,
    );
    let (kf0, kf_recon0, kf_state0) =
        encode_keyframe_lossy_tree_with_state(&kf_hdr, &kf_targets, &kf_plan)?;
    let (kf_bytes, kf_recon) = finish_frame_with_filter(
        &kf_hdr,
        kf0,
        kf_recon0,
        kf_state0,
        &kf_targets,
        w,
        h,
        |hdr2| encode_keyframe_lossy_tree_with_state(hdr2, &kf_targets, &kf_plan),
    )?;

    let mut out = Vec::with_capacity(frames.len());
    out.push(kf_bytes);

    // Long-term GOLDEN reference: the keyframe's reconstruction stays
    // parked in §8.10 slot 1 (the keyframe refreshes every slot; the
    // P-frames refresh only slot 0), so `ref_frame_idx = [0, 1, 1]`
    // resolves LAST to the previous frame and GOLDEN to the keyframe.
    // Post-filter, per §8.10: the store happens after §8.1 step 2.
    let golden = visible_crop(&kf_recon);
    let golden_ref: [(&[i32], usize); 3] = [
        (golden[0].as_slice(), w),
        (golden[1].as_slice(), cw),
        (golden[2].as_slice(), cw),
    ];
    let mut prev_recon = kf_recon;

    for &frame in frames.iter().skip(1) {
        let targets = padded_targets(frame);
        let prev = visible_crop(&prev_recon);
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), w),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        let mut hdr = lossless_pframe_header(width, height);
        hdr.ref_frame_idx = Some([0, 1, 1]);
        // ALTREF (also slot 1) carries the opposite sign bias so the
        // §6.3.12 compoundReferenceAllowed derivation admits the
        // [ LAST, ALTREF ] compound pair.
        hdr.ref_frame_sign_bias = [false, false, true];
        hdr.quantization = QuantizationParams {
            base_q_idx,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (p0, recon0, state0) = encode_pframe_lossy_tree_motion_with_state(
            &hdr,
            &targets,
            &reference,
            Some(&golden_ref),
            width,
            height,
            PFRAME_SEARCH_RANGE,
            true,
            true,
        )?;
        let (bytes, recon) =
            finish_frame_with_filter(&hdr, p0, recon0, state0, &targets, w, h, |hdr2| {
                encode_pframe_lossy_tree_motion_with_state(
                    hdr2,
                    &targets,
                    &reference,
                    Some(&golden_ref),
                    width,
                    height,
                    PFRAME_SEARCH_RANGE,
                    true,
                    true,
                )
            })?;
        out.push(bytes);
        prev_recon = recon;
    }
    Ok(out)
}

/// Bisect `base_q_idx` for the **lowest** quantizer whose coded frame
/// fits `target_bytes` — the per-frame rate-control primitive.
///
/// `encode( q )` must be a pure function of `q` (byte-deterministic,
/// which every encoder in this module is). Coded size is monotone
/// non-increasing in `q` for all practical content, so a binary search
/// over `1..=255` finds the best-quality fitting quantizer in at most 8
/// probes; each accepted probe re-uses its actual encode (no re-run).
/// If even `q == 255` overflows the budget the `q == 255` encode is
/// returned **best-effort** (the caller keeps a decodable stream rather
/// than an error — a budget below the syntax floor is unrepresentable).
fn bisect_q<T>(
    mut encode: impl FnMut(u8) -> Result<(Vec<u8>, T), Error>,
    target_bytes: usize,
) -> Result<(Vec<u8>, T, u8), Error> {
    let mut lo = 1u16;
    let mut hi = 255u16;
    let mut best: Option<(Vec<u8>, T, u8)> = None;
    while lo <= hi {
        let mid = ((lo + hi) / 2) as u8;
        let (bytes, aux) = encode(mid)?;
        if bytes.len() <= target_bytes {
            best = Some((bytes, aux, mid));
            if mid == 1 {
                break;
            }
            hi = u16::from(mid) - 1;
        } else {
            lo = u16::from(mid) + 1;
        }
    }
    match best {
        Some(b) => Ok(b),
        None => {
            let (bytes, aux) = encode(255)?;
            Ok((bytes, aux, 255))
        }
    }
}

/// Rate-controlled lossy sequence encoder: every frame is coded at the
/// **lowest** `base_q_idx` whose size fits `target_bytes_per_frame`
/// (per-frame [`bisect_q`]; best-effort `q == 255` when the budget is
/// below the syntax floor). The keyframe runs the content-adaptive
/// partition/tx planner per probe; P-frames motion-search against the
/// chosen previous reconstruction, so the decoder mirror stays exact at
/// whatever quantizer each frame lands on.
pub(crate) fn encode_sequence_lossy_rc_420(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    target_bytes_per_frame: usize,
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

    let padded_targets = |pixels: &[u8]| -> [Plane; 3] {
        let y_w = (((width + 7) >> 3) * 8) as usize;
        let y_h = (((height + 7) >> 3) * 8) as usize;
        [
            padded_plane_from_bytes(&pixels[..w * h], w, h, y_w, y_h),
            padded_plane_from_bytes(&pixels[w * h..w * h + cw * ch], cw, ch, y_w >> 1, y_h >> 1),
            padded_plane_from_bytes(&pixels[w * h + cw * ch..], cw, ch, y_w >> 1, y_h >> 1),
        ]
    };
    let visible_crop = |recon: &ReconState| -> [Vec<i32>; 3] {
        let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(vw * vh);
            for y in 0..vh {
                for x in 0..vw {
                    out.push(p.get(x, y));
                }
            }
            out
        };
        [
            crop(&recon.planes[0], w, h),
            crop(&recon.planes[1], cw, ch),
            crop(&recon.planes[2], cw, ch),
        ]
    };

    // Keyframe: bisect q over the planner-driven encoder, then elect
    // the §8.8 filter level at the chosen q (the level is a fixed-width
    // §6.2.8 field, so the election never disturbs the fitted size).
    let (kf0, (kf_recon0, kf_state0, kf_q), _) = bisect_q(
        |q| {
            encode_keyframe_lossy_420_with_recon_state(frames[0], width, height, q)
                .map(|(b, r, s)| (b, (r, s, q)))
        },
        target_bytes_per_frame,
    )?;
    let kf_targets = padded_targets(frames[0]);
    let kf_hdr = lossy_keyframe_header_420(width, height, kf_q);
    let (kf_bytes, kf_recon) = finish_frame_with_filter(
        &kf_hdr,
        kf0,
        kf_recon0,
        kf_state0,
        &kf_targets,
        w,
        h,
        |hdr2| {
            let mi_cols = (width + 7) >> 3;
            let mi_rows = (height + 7) >> 3;
            let plan = plan_keyframe_tree(&kf_targets, mi_rows, mi_cols, true, true, 8, kf_q);
            encode_keyframe_lossy_tree_with_state(hdr2, &kf_targets, &plan)
        },
    )?;

    let mut out = Vec::with_capacity(frames.len());
    out.push(kf_bytes);

    // Long-term GOLDEN reference (see `encode_sequence_lossy_420`) —
    // the keyframe's *filtered* reconstruction, per the §8.10
    // post-filter store.
    let golden = visible_crop(&kf_recon);
    let golden_ref: [(&[i32], usize); 3] = [
        (golden[0].as_slice(), w),
        (golden[1].as_slice(), cw),
        (golden[2].as_slice(), cw),
    ];
    let mut prev_recon = kf_recon;

    for &frame in frames.iter().skip(1) {
        let targets = padded_targets(frame);
        let prev = visible_crop(&prev_recon);
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), w),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        let pframe_hdr = |q: u8| -> Vp9FrameHeader {
            let mut hdr = lossless_pframe_header(width, height);
            hdr.ref_frame_idx = Some([0, 1, 1]);
            hdr.ref_frame_sign_bias = [false, false, true];
            hdr.quantization = QuantizationParams {
                base_q_idx: q,
                delta_q_y_dc: 0,
                delta_q_uv_dc: 0,
                delta_q_uv_ac: 0,
                lossless: false,
            };
            hdr
        };
        let encode_at = |hdr: &Vp9FrameHeader| {
            encode_pframe_lossy_tree_motion_with_state(
                hdr,
                &targets,
                &reference,
                Some(&golden_ref),
                width,
                height,
                PFRAME_SEARCH_RANGE,
                true,
                true,
            )
        };
        let (p0, (recon0, state0, p_q), _) = bisect_q(
            |q| encode_at(&pframe_hdr(q)).map(|(b, r, s)| (b, (r, s, q))),
            target_bytes_per_frame,
        )?;
        let (bytes, recon) = finish_frame_with_filter(
            &pframe_hdr(p_q),
            p0,
            recon0,
            state0,
            &targets,
            w,
            h,
            |hdr2| encode_at(hdr2),
        )?;
        out.push(bytes);
        prev_recon = recon;
    }
    Ok(out)
}

/// Encode a sequence of 8-bit 4:2:0 planar frames (each `Y` then `U`
/// then `V`, the [`crate::decode_vp9`] layout) into a lossless VP9
/// stream: a keyframe followed by P-frames, each coding the exact
/// `frame − prediction` residual with per-block `ZEROMV` / `NEWMV`
/// motion (integer full search over ±[`PFRAME_SEARCH_RANGE`] pixels).
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
        out.push(encode_pframe_lossless_motion(
            &hdr,
            &targets,
            &reference,
            width,
            height,
            PFRAME_SEARCH_RANGE,
            true,
        )?);
    }
    Ok(out)
}

/// [`encode_sequence_lossless_420`] on **non-error-resilient** framing
/// with the §7.2.6 `UsePrevFrameMvs == 1` chain model: every P-frame is
/// shown and non-error-resilient, so the decoder scans the previous
/// frame's motion field — and the encoder supplies each frame's plan
/// with exactly that field (the keyframe's all-intra field first, then
/// each P-frame's §6.4.4 write-back), so the §6.5 predictors are
/// bit-identical on both sides. A vector the previous frame already
/// coded at the same position reaches the §6.5.10 prev candidate list
/// and maps to `NEARESTMV` / `NEARMV` (no §6.4.20 mv-diff bits), so
/// motion that persists across frames — where spatial neighbours
/// predict it wrongly — codes fewer bytes than the error-resilient
/// chain (rate A/B pinned by test).
pub(crate) fn encode_sequence_lossless_chained_420(
    frames: &[&[u8]],
    width: u32,
    height: u32,
) -> Result<Vec<Vec<u8>>, Error> {
    use crate::frame_writer::PrevMotionField;

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

    // The keyframe leaves an all-intra §6.4.4 motion field.
    let mut prev_field = PrevMotionField::after_intra_frame(mi_rows as u32, mi_cols as u32);

    for i in 1..frames.len() {
        let targets = padded_targets(frames[i]);
        let prev = visible_ref(frames[i - 1]);
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), w),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        // Shown, non-error-resilient: §7.2.6 derives UsePrevFrameMvs = 1
        // on the decode side for every P-frame of this chain.
        let mut hdr = lossless_pframe_header(width, height);
        hdr.error_resilient_mode = false;
        let (bytes, state) = encode_pframe_lossless_motion_prev(
            &hdr,
            &targets,
            &reference,
            width,
            height,
            PFRAME_SEARCH_RANGE,
            true,
            Some(&prev_field),
        )?;
        out.push(bytes);
        prev_field = PrevMotionField::from_state(&state);
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
        let (bytes, recon) = encode_keyframe_lossy(&hdr, &targets, true).expect("encode");
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

    // ----- tree-plan lossy encoder (large transforms) -----

    fn padded_targets_420(px: &[u8], w: u32, h: u32) -> [Plane; 3] {
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let y_w = (((w + 7) >> 3) * 8) as usize;
        let y_h = (((h + 7) >> 3) * 8) as usize;
        [
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
        ]
    }

    fn lossy_header(w: u32, h: u32, q: u8) -> Vp9FrameHeader {
        let mut hdr = lossless_keyframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: q,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        hdr
    }

    fn assert_tree_mirror_exact(w: u32, h: u32, q: u8, leaf_size: u8, tx_size: u32) -> usize {
        use crate::frame_writer::KeyframeTreePlan;
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        let px: Vec<u8> = (0..n).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        let hdr = lossy_header(w, h, q);
        let targets = padded_targets_420(&px, w, h);
        let mi_cols = (w + 7) >> 3;
        let mi_rows = (h + 7) >> 3;
        let plan = KeyframeTreePlan::uniform(mi_rows, mi_cols, leaf_size, tx_size);
        let (bytes, recon) = encode_keyframe_lossy_tree(&hdr, &targets, &plan).expect("encode");
        let frame = decode_intra_frame(&bytes).expect("decode");
        for row in 0..h as usize {
            for col in 0..w as usize {
                assert_eq!(
                    i32::from(frame.y[row * w as usize + col]),
                    recon.planes[0].get(col, row),
                    "luma ({col},{row}) leaf={leaf_size} tx={tx_size}"
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
        bytes.len()
    }

    /// TX_32X32 / TX_16X16 / TX_8X8 tree keyframes on noise: the decoder
    /// output equals the encoder's in-loop reconstruction bit-for-bit at
    /// every transform size (the first >4x4 transform *content* the
    /// encoder emits, incl. the §8.6.2 dqDenom == 2 path).
    #[test]
    fn lossy_tree_decode_equals_encoder_recon_all_tx_sizes() {
        use crate::residual::{BLOCK_16X16, BLOCK_32X32, BLOCK_64X64};
        assert_tree_mirror_exact(64, 64, 80, BLOCK_64X64, 3);
        assert_tree_mirror_exact(64, 64, 80, BLOCK_32X32, 2);
        assert_tree_mirror_exact(64, 64, 80, BLOCK_16X16, 1);
    }

    /// Non-multiple-of-64 geometry: the uniform plan splits at frame
    /// edges (mixed leaf sizes) and the mirror stays exact.
    #[test]
    fn lossy_tree_partial_superblock_mirror_exact() {
        use crate::residual::BLOCK_32X32;
        assert_tree_mirror_exact(80, 48, 64, BLOCK_32X32, 3);
        assert_tree_mirror_exact(40, 24, 120, BLOCK_32X32, 2);
    }

    /// On smooth content a large-transform tree codes fewer bytes than
    /// the all-4x4 encoder at the same quantizer — the point of >4x4
    /// transform support.
    #[test]
    fn lossy_tree_large_tx_smaller_on_smooth_content() {
        use crate::frame_writer::KeyframeTreePlan;
        use crate::residual::BLOCK_64X64;
        let (w, h) = (64u32, 64u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        // Smooth diagonal gradient.
        let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
        for y in 0..h as usize {
            for x in 0..w as usize {
                px.push(((x + y) * 2) as u8);
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                px.push((128 + x + y) as u8);
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                px.push((64 + 2 * x + y) as u8);
            }
        }
        let q = 60u8;
        let hdr = lossy_header(w, h, q);
        let targets = padded_targets_420(&px, w, h);
        let plan = KeyframeTreePlan::uniform(8, 8, BLOCK_64X64, 3);
        let (tree_bytes, _) = encode_keyframe_lossy_tree(&hdr, &targets, &plan).expect("tree");
        // Compare against the fixed all-BLOCK_8X8 / TX_4X4 engine (the
        // public path now plans adaptively, so invoke it directly).
        let (small_bytes, _) = encode_keyframe_lossy(&hdr, &targets, true).expect("4x4");
        assert!(
            tree_bytes.len() < small_bytes.len(),
            "TX_32X32 tree ({}) not smaller than all-4x4 ({}) on smooth content",
            tree_bytes.len(),
            small_bytes.len()
        );
        // And it still decodes to bounded distortion vs the source.
        let frame = decode_intra_frame(&tree_bytes).expect("decode");
        let mut mse = 0f64;
        for row in 0..h as usize {
            for col in 0..w as usize {
                let d = f64::from(frame.y[row * w as usize + col])
                    - f64::from(px[row * w as usize + col]);
                mse += d * d;
            }
        }
        mse /= f64::from(w * h);
        assert!(mse < 100.0, "TX_32X32 gradient MSE {mse} too high");
    }

    /// Directional-mode leaves at TX_8X8 / TX_16X16 exercise the forward
    /// ADST8 / ADST16 bases through the full encode → decode mirror.
    #[test]
    fn lossy_tree_adst_modes_mirror_exact() {
        use crate::frame_writer::KeyframeTreePlan;
        use crate::residual::BLOCK_16X16;
        let (w, h) = (64u32, 64u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let n = (w * h) as usize + 2 * cw * ch;
        let px: Vec<u8> = (0..n).map(|i| ((i * 31 + 7) % 256) as u8).collect();
        let hdr = lossy_header(w, h, 100);
        let targets = padded_targets_420(&px, w, h);
        let mut plan = KeyframeTreePlan::uniform(8, 8, BLOCK_16X16, 2);
        // V_PRED -> ADST_DCT, H_PRED -> DCT_ADST, TM_PRED -> ADST_ADST
        // (§6.4.25 mode2txfm_map), cycled across the leaves; chroma keeps
        // DC (forced DCT_DCT on chroma regardless).
        let modes = [1u8, 2, 9, 5];
        for (i, lp) in plan.leaves.values_mut().enumerate() {
            lp.y_mode = modes[i % modes.len()];
        }
        let (bytes, recon) = encode_keyframe_lossy_tree(&hdr, &targets, &plan).expect("encode");
        let frame = decode_intra_frame(&bytes).expect("decode");
        for row in 0..h as usize {
            for col in 0..w as usize {
                assert_eq!(
                    i32::from(frame.y[row * w as usize + col]),
                    recon.planes[0].get(col, row),
                    "luma ({col},{row})"
                );
            }
        }
    }

    /// The planner elects one 64x64 leaf at TX_32X32 per superblock on
    /// flat content, and splits toward 8x8 on high-contrast structure.
    #[test]
    fn planner_adapts_partition_to_content() {
        let (w, h) = (64u32, 64u32);
        // Flat frame.
        let flat = vec![90u8; (w * h) as usize + 2 * 32 * 32];
        let t_flat = padded_targets_420(&flat, w, h);
        let plan_flat = plan_keyframe_tree(&t_flat, 8, 8, true, true, 8, 60);
        assert_eq!(plan_flat.leaves.len(), 1, "flat content: one 64x64 leaf");
        let lp = plan_flat.leaves[&(0, 0)];
        assert_eq!(lp.mi_size, crate::residual::BLOCK_64X64);
        assert_eq!(lp.tx_size, 3);

        // Quadrant-contrast frame: four flat 32x32 luma quadrants at
        // very different levels force a split at the 64x64 root, then
        // each 32x32 quadrant is homogeneous.
        let mut px = vec![0u8; (w * h) as usize + 2 * 32 * 32];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let q = (usize::from(y >= 32) << 1) | usize::from(x >= 32);
                px[y * w as usize + x] = [30u8, 100, 170, 240][q];
            }
        }
        for s in px[(w * h) as usize..].iter_mut() {
            *s = 128;
        }
        let t = padded_targets_420(&px, w, h);
        let plan = plan_keyframe_tree(&t, 8, 8, true, true, 8, 60);
        assert_eq!(plan.leaves.len(), 4, "quadrant content: four 32x32 leaves");
        assert!(plan
            .leaves
            .values()
            .all(|l| l.mi_size == crate::residual::BLOCK_32X32));
    }

    /// The planner's split threshold scales with the quantizer: content
    /// that splits at a fine quantizer stays whole at a coarse one.
    #[test]
    fn planner_threshold_scales_with_q() {
        let (w, h) = (64u32, 64u32);
        // Mild quadrant contrast (±18 around 128) — above the fine-q
        // AC step, far below the coarse-q one.
        let mut px = vec![128u8; (w * h) as usize + 2 * 32 * 32];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let q = (usize::from(y >= 32) << 1) | usize::from(x >= 32);
                px[y * w as usize + x] = [110u8, 122, 134, 146][q];
            }
        }
        let t = padded_targets_420(&px, w, h);
        let plan_fine = plan_keyframe_tree(&t, 8, 8, true, true, 8, 5);
        let plan_coarse = plan_keyframe_tree(&t, 8, 8, true, true, 8, 220);
        assert!(
            plan_fine.leaves.len() > plan_coarse.leaves.len(),
            "fine q leaves {} <= coarse q leaves {}",
            plan_fine.leaves.len(),
            plan_coarse.leaves.len()
        );
        assert_eq!(plan_coarse.leaves.len(), 1);
    }

    /// The public lossy path (now planner-driven) round-trips: the
    /// coded frame decodes, distortion is bounded, and on mixed
    /// content it codes fewer bytes than the fixed all-4x4 engine.
    #[test]
    fn public_lossy_adaptive_beats_fixed_4x4_on_mixed_content() {
        let (w, h) = (128u32, 64u32);
        let cw = 64usize;
        let ch = 32usize;
        // Left superblock: smooth gradient; right superblock: noise.
        let mut px = vec![0u8; (w * h) as usize + 2 * cw * ch];
        for y in 0..h as usize {
            for x in 0..w as usize {
                px[y * w as usize + x] = if x < 64 {
                    ((x + y) / 2) as u8
                } else {
                    ((x * 37 + y * 91 + 13) % 256) as u8
                };
            }
        }
        for s in px[(w * h) as usize..].iter_mut() {
            *s = 128;
        }
        let q = 80u8;
        let adaptive = encode_keyframe_lossy_420(&px, w, h, q).expect("adaptive");
        let hdr = lossy_header(w, h, q);
        let targets = padded_targets_420(&px, w, h);
        let (fixed, _) = encode_keyframe_lossy(&hdr, &targets, true).expect("fixed 4x4");
        assert!(
            adaptive.len() < fixed.len(),
            "adaptive ({}) not smaller than fixed 4x4 ({})",
            adaptive.len(),
            fixed.len()
        );
        // Bounded distortion on the smooth half.
        let frame = decode_intra_frame(&adaptive).expect("decode");
        let mut mse = 0f64;
        for y in 0..h as usize {
            for x in 0..64usize {
                let d = f64::from(frame.y[y * w as usize + x]) - f64::from(px[y * w as usize + x]);
                mse += d * d;
            }
        }
        mse /= f64::from(64 * h);
        assert!(mse < 200.0, "smooth-half MSE {mse} too high at q=80");
    }

    /// Per-block skip election, pinned directly: a lossy P-frame whose
    /// reference **equals** its target (exact prediction at `ZEROMV`)
    /// quantizes every residual to zero, elects skip on every block,
    /// and codes a tiny all-syntax frame whose reconstruction equals
    /// the reference. On a static *sequence* (where each P-frame
    /// legitimately refines the previous frame's quantization error
    /// toward the source) the P-frames still shrink an order of
    /// magnitude below the keyframe and decode through the chain
    /// mirror.
    #[test]
    fn lossy_pframe_skip_election_on_static_content() {
        use crate::decode_frame::decode_vp9_sequence;
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let px: Vec<u8> = (0..n).map(|i| ((i * 73 + 19) % 256) as u8).collect();

        // Direct pin: reference == target => all-skip.
        let targets = padded_targets_420(&px, w, h);
        let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(vw * vh);
            for y in 0..vh {
                for x in 0..vw {
                    out.push(p.get(x, y));
                }
            }
            out
        };
        let (cw, ch) = (32usize, 32usize);
        let ref_planes = [
            crop(&targets[0], w as usize, h as usize),
            crop(&targets[1], cw, ch),
            crop(&targets[2], cw, ch),
        ];
        let reference: [(&[i32], usize); 3] = [
            (ref_planes[0].as_slice(), w as usize),
            (ref_planes[1].as_slice(), cw),
            (ref_planes[2].as_slice(), cw),
        ];
        let mut hdr = lossless_pframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: 60,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (bytes, recon) =
            encode_pframe_lossy_motion(&hdr, &targets, &reference, w, h, PFRAME_SEARCH_RANGE)
                .expect("p-frame");
        assert!(
            bytes.len() < 100,
            "exact-prediction P-frame ({} B) should be all-skip-small",
            bytes.len()
        );
        // Skip reconstruction == prediction == reference.
        for y in 0..h as usize {
            for x in 0..w as usize {
                assert_eq!(recon.planes[0].get(x, y), targets[0].get(x, y), "({x},{y})");
            }
        }

        // Sequence-level: static P-frames stay an order of magnitude
        // below the keyframe and the chain decodes.
        let frames: Vec<&[u8]> = vec![&px, &px, &px];
        let coded = encode_sequence_lossy_420(&frames, w, h, 60).expect("encode");
        assert!(coded[1].len() * 8 < coded[0].len(), "P1 << keyframe");
        assert!(coded[2].len() <= coded[1].len(), "P2 <= P1 (converging)");
        let refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        decode_vp9_sequence(&refs).expect("decode");
    }

    /// The lossless P-frame elects skip on exactly-predicted blocks: a
    /// static lossless pair codes a near-empty P-frame and stays
    /// byte-exact.
    #[test]
    fn lossless_pframe_skip_election_stays_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let px: Vec<u8> = (0..n).map(|i| ((i * 31 + 7) % 256) as u8).collect();
        let frames: Vec<&[u8]> = vec![&px, &px];
        let coded = encode_sequence_lossless_420(&frames, w, h).expect("encode");
        assert!(
            coded[1].len() < 100,
            "static lossless P-frame ({} B) should be all-skip-small",
            coded[1].len()
        );
        let refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let dec = decode_vp9_sequence(&refs).expect("decode");
        for (i, f) in dec.iter().enumerate() {
            let y_w = w as usize;
            for row in 0..h as usize {
                for col in 0..y_w {
                    assert_eq!(
                        f.y[row * y_w + col],
                        u16::from(px[row * y_w + col]),
                        "frame {i} luma ({col},{row})"
                    );
                }
            }
        }
    }

    // ----- rate control -----

    /// Every frame of a rate-controlled sequence fits the byte budget
    /// (when the budget is above the syntax floor), and the stream
    /// decodes end-to-end.
    #[test]
    fn rc_sequence_respects_frame_budget() {
        use crate::decode_frame::decode_vp9_sequence;
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let frames: Vec<Vec<u8>> = (0..3u64)
            .map(|t| {
                (0..n)
                    .map(|i| ((i as u64 * 73 + 19 + 5 * t) % 256) as u8)
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();

        let budget = 2000usize;
        let coded = encode_sequence_lossy_rc_420(&refs, w, h, budget).expect("rc encode");
        assert_eq!(coded.len(), 3);
        for (i, f) in coded.iter().enumerate() {
            assert!(
                f.len() <= budget,
                "frame {i} size {} exceeds budget {budget}",
                f.len()
            );
        }
        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode");
        assert_eq!(decoded.len(), 3);
    }

    /// A larger budget buys lower distortion (the bisection lands on a
    /// finer quantizer).
    #[test]
    fn rc_quality_improves_with_budget() {
        use crate::decode_frame::decode_vp9_sequence;
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let px: Vec<u8> = (0..n).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        let refs: Vec<&[u8]> = vec![px.as_slice()];

        let mse_at = |budget: usize| -> f64 {
            let coded = encode_sequence_lossy_rc_420(&refs, w, h, budget).expect("rc");
            assert!(coded[0].len() <= budget, "budget {budget} not met");
            let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
            let dec = decode_vp9_sequence(&coded_refs).expect("decode");
            let mut mse = 0f64;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let d =
                        f64::from(dec[0].y[y * w as usize + x]) - f64::from(px[y * w as usize + x]);
                    mse += d * d;
                }
            }
            mse / f64::from(w * h)
        };
        let coarse = mse_at(600);
        let fine = mse_at(8000);
        assert!(
            fine <= coarse,
            "MSE at 8000 B ({fine}) worse than at 600 B ({coarse})"
        );
    }

    /// Deterministic banded-motion content: each 8-px luma band
    /// translates rigidly over an unbounded texture with a per-band
    /// horizontal velocity cycling through `+2 / -2 / +4 / -4` px per
    /// frame, so every 8x8 block has an exact integer-motion match in
    /// the previous frame (its whole band translates, and the texture
    /// continues past the frame edge) while vertically adjacent blocks
    /// carry DIFFERENT vectors — the §6.5 spatial predictors are wrong
    /// at every band boundary, and the §6.5.10 previous-frame candidate
    /// (same position, same persistent motion) is exact everywhere.
    /// Chroma is flat.
    fn banded_motion_frames(w: u32, h: u32, n: usize) -> Vec<Vec<u8>> {
        let wu = w as usize;
        let hu = h as usize;
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let tex = |x: i64, y: i64| -> u8 { ((x * 7 + y * 13).rem_euclid(61) * 4 + 8) as u8 };
        const VELS: [i64; 4] = [2, -2, 4, -4];
        (0..n)
            .map(|i| {
                let mut f = vec![128u8; wu * hu + 2 * cw * ch];
                for y in 0..hu {
                    let vel = VELS[(y / 8) % VELS.len()];
                    for x in 0..wu {
                        f[y * wu + x] = tex(x as i64 - vel * i as i64, y as i64);
                    }
                }
                f
            })
            .collect()
    }

    /// The chained (non-error-resilient, §7.2.6 `UsePrevFrameMvs == 1`)
    /// lossless sequence encoder keeps the byte-exact guarantee
    /// end-to-end: every decoded frame equals its input. The decode side
    /// derives `UsePrevFrameMvs = 1` for every P-frame here (shown
    /// same-sized predecessors, non-ER headers), so byte-exactness
    /// proves the encoder's prev-field model matches the decoder's scan
    /// — any mismatch desyncs the predictors and corrupts the NEWMV /
    /// NEARESTMV blocks.
    #[test]
    fn lossless_chained_sequence_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        let (w, h) = (64u32, 64u32);
        let frames = banded_motion_frames(w, h, 5);
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let coded = encode_sequence_lossless_chained_420(&refs, w, h).expect("chained encode");
        assert_eq!(coded.len(), 5);
        // Every P-frame header is shown and non-error-resilient — the
        // §7.2.6 shape whose decode scans the prev motion field.
        for p in coded.iter().skip(1) {
            let ref_dims = vec![(w, h); 8];
            let hdr = crate::header::parse_uncompressed_header_with_refs(
                p,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: crate::header::ColorConfig {
                        bit_depth: 8,
                        color_space: crate::header::ColorSpace::Bt601,
                        color_range_full: false,
                        subsampling_x: true,
                        subsampling_y: true,
                    },
                }),
            )
            .expect("P header");
            assert!(hdr.show_frame && !hdr.error_resilient_mode);
        }
        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let dec = decode_vp9_sequence(&coded_refs).expect("decode");
        for (i, d) in dec.iter().enumerate() {
            assert_eq!(
                d.to_planar_bytes(),
                frames[i],
                "frame {i} not byte-exact through the chained encode"
            );
        }
    }

    /// Rate A/B: on banded persistent motion the chained encoder
    /// codes strictly fewer total bytes than the error-resilient chain —
    /// the prev-frame candidate turns `NEWMV` blocks whose spatial
    /// predictors point the wrong way into `NEARESTMV` / `NEARMV`
    /// (no §6.4.20 mv-diff bits). Both chains decode byte-exact, so the
    /// delta is pure entropy-path savings.
    #[test]
    fn lossless_chained_sequence_beats_error_resilient_rate() {
        let (w, h) = (64u32, 128u32);
        let frames = banded_motion_frames(w, h, 6);
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();

        let chained = encode_sequence_lossless_chained_420(&refs, w, h).expect("chained");
        let er = encode_sequence_lossless_420(&refs, w, h).expect("er");
        let chained2 = encode_sequence_lossless_chained_420(&refs, w, h).expect("chained again");
        assert_eq!(chained, chained2, "chained encode must be deterministic");

        // Keyframes are identical; compare the P-frame totals.
        assert_eq!(chained[0], er[0], "keyframes must match");
        let chained_p: usize = chained.iter().skip(1).map(|f| f.len()).sum();
        let er_p: usize = er.iter().skip(1).map(|f| f.len()).sum();
        assert!(
            chained_p < er_p,
            "chained P-frames ({chained_p} B) must beat the error-resilient \
             chain ({er_p} B) on persistent banded motion"
        );
    }

    /// A budget below the syntax floor returns a best-effort q=255
    /// stream (still decodable) instead of failing.
    #[test]
    fn rc_best_effort_below_syntax_floor() {
        use crate::decode_frame::decode_vp9_sequence;
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let px: Vec<u8> = (0..n).map(|i| ((i * 31 + 3) % 256) as u8).collect();
        let refs: Vec<&[u8]> = vec![px.as_slice()];
        let coded = encode_sequence_lossy_rc_420(&refs, w, h, 4).expect("best effort");
        assert!(coded[0].len() > 4, "a 4-byte frame is not representable");
        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        decode_vp9_sequence(&coded_refs).expect("decode");
    }

    /// Rate-control input validation matches the fixed-q sequence path.
    #[test]
    fn rc_rejects_bad_inputs() {
        assert_eq!(
            encode_sequence_lossy_rc_420(&[], 64, 64, 1000).unwrap_err(),
            Error::Unsupported
        );
        let short = vec![0u8; 10];
        assert_eq!(
            encode_sequence_lossy_rc_420(&[short.as_slice()], 64, 64, 1000).unwrap_err(),
            Error::Unsupported
        );
        let px = vec![0u8; 64 * 64 + 2 * 32 * 32];
        assert_eq!(
            encode_sequence_lossy_rc_420(&[px.as_slice()], 0, 64, 1000).unwrap_err(),
            Error::Unsupported
        );
    }

    /// Skip leaves are rejected (the mirror never replays a skip block's
    /// prediction).
    #[test]
    fn lossy_tree_rejects_skip_leaves() {
        use crate::frame_writer::KeyframeTreePlan;
        use crate::residual::BLOCK_32X32;
        let hdr = lossy_header(64, 64, 80);
        let px = vec![128u8; 64 * 64 + 2 * 32 * 32];
        let targets = padded_targets_420(&px, 64, 64);
        let mut plan = KeyframeTreePlan::uniform(8, 8, BLOCK_32X32, 3);
        for lp in plan.leaves.values_mut() {
            lp.skip = true;
        }
        match encode_keyframe_lossy_tree(&hdr, &targets, &plan) {
            Err(e) => assert_eq!(e, Error::Unsupported),
            Ok(_) => panic!("skip leaves must be rejected"),
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

    /// On strongly directional content (vertical stripes) the §7.4.5
    /// mode selection beats forced `DC_PRED` on **both** axes: smaller
    /// stream and lower distortion at the same quantizer. Also pins that
    /// the selection actually picks a non-DC luma mode somewhere.
    #[test]
    fn mode_selection_beats_dc_on_directional_content() {
        let (w, h) = (64u32, 48u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        // Vertical stripes: every column constant, strong V_PRED fit.
        let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
        for _i in 0..h as usize {
            for j in 0..w as usize {
                px.push((40 + (j % 8) * 25) as u8);
            }
        }
        for _i in 0..ch {
            for j in 0..cw {
                px.push((60 + (j % 4) * 40) as u8);
            }
        }
        for _i in 0..ch {
            for j in 0..cw {
                px.push((200 - (j % 4) * 30) as u8);
            }
        }

        let mut hdr = lossless_keyframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: 60,
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

        let (sel_bytes, _) = encode_keyframe_lossy(&hdr, &targets, true).expect("select");
        let (dc_bytes, _) = encode_keyframe_lossy(&hdr, &targets, false).expect("dc");

        let mse = |bytes: &[u8]| -> f64 {
            let out = decode_intra_frame(bytes).expect("decode").to_planar_bytes();
            out.iter()
                .zip(&px)
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum::<f64>()
                / out.len() as f64
        };
        let sel_mse = mse(&sel_bytes);
        let dc_mse = mse(&dc_bytes);
        assert!(
            sel_bytes.len() < dc_bytes.len(),
            "mode selection ({} B) must beat DC-only ({} B) on stripes",
            sel_bytes.len(),
            dc_bytes.len()
        );
        // Rate/distortion: near-perfect directional prediction quantizes
        // its (tiny) residual to zero instead of coding corrections, so
        // the MSE may sit slightly above the DC path's — but both must
        // stay in the same near-transparent regime.
        assert!(
            sel_mse <= dc_mse + 2.0,
            "mode selection MSE {sel_mse} strayed from DC-only {dc_mse}"
        );

        // The selection itself picks a non-DC luma mode on stripes.
        let plans =
            super::select_keyframe_modes(&targets, (w + 7) >> 3, (h + 7) >> 3, true, true, 8);
        assert!(
            plans.iter().any(|p| p.y_mode != 0),
            "stripes should elect a directional luma mode"
        );
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

    /// A lossless P-frame over a mixed layout with **sub-8x8 leaves**
    /// (4x4 / 8x4 / 4x8) carrying distinct per-cell NEWMV / NEARESTMV /
    /// ZEROMV vectors — noise-content keyframe and target, so every
    /// sub-block's residual is live — reconstructs **byte-exact**
    /// through `decode_vp9_sequence` (per-`blockIdx` §8.5.2 prediction,
    /// averaged chroma vectors, and the §8.7.2 WHT residual included).
    #[test]
    fn lossless_layout_sub8x8_leaves_roundtrip_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::frame_writer::InterTreeLeaf;
        use crate::inter_block_writer::InterSubBlockSpec;
        use crate::mode_info::{LAST_FRAME, NEARESTMV, NEWMV, NONE_REF_FRAME, ZEROMV};
        use crate::partition::{PARTITION_HORZ, PARTITION_SPLIT, PARTITION_VERT};
        use crate::residual::{BLOCK_4X4, BLOCK_4X8, BLOCK_8X4};

        let (w, h) = (64u32, 64u32);
        let (wu, hu, cw, ch) = (64usize, 64usize, 32usize, 32usize);
        let f0 = noise_frame(w, h, 0x5EED_0001);
        let f1 = noise_frame(w, h, 0x5EED_0002);

        let kf = encode_keyframe_lossless_420(&f0, w, h).expect("keyframe");

        let targets = [
            padded_plane_from_bytes(&f1[..wu * hu], wu, hu, wu, hu),
            padded_plane_from_bytes(&f1[wu * hu..wu * hu + cw * ch], cw, ch, cw, ch),
            padded_plane_from_bytes(&f1[wu * hu + cw * ch..], cw, ch, cw, ch),
        ];
        let prev: [Vec<i32>; 3] = [
            f0[..wu * hu].iter().map(|&s| i32::from(s)).collect(),
            f0[wu * hu..wu * hu + cw * ch]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            f0[wu * hu + cw * ch..]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), wu),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];

        let hdr = lossless_pframe_header(w, h);
        let mut partitions = std::collections::HashMap::new();
        partitions.insert((0u32, 0u32, BLOCK_8X8), PARTITION_SPLIT);
        partitions.insert((3, 4, BLOCK_8X8), PARTITION_HORZ);
        partitions.insert((6, 2, BLOCK_8X8), PARTITION_VERT);

        // All components even => codeable under either §6.5.13 hp gate.
        let a = [16, 8];
        let b = [-16, 24];
        let mut leaf_plan = |_r: u32,
                             _c: u32,
                             subsize: u8,
                             _s: &crate::decode_block::Vp9FrameState|
         -> InterTreeLeaf {
            let sub = match subsize {
                // 4x4: NEWMV a, NEARESTMV (block 1 seeds BlockMvs[0] =
                // a), NEWMV b, ZEROMV.
                BLOCK_4X4 => Some(InterSubBlockSpec {
                    modes: [NEWMV, NEARESTMV, NEWMV, ZEROMV],
                    mvs: [[a, [0, 0]], [a, [0, 0]], [b, [0, 0]], [[0, 0], [0, 0]]],
                }),
                // 8x4: cells {0, 2} visited.
                BLOCK_8X4 => Some(InterSubBlockSpec {
                    modes: [NEWMV, ZEROMV, NEWMV, ZEROMV],
                    mvs: [
                        [[8, -8], [0, 0]],
                        [[0, 0]; 2],
                        [[4, 6], [0, 0]],
                        [[0, 0]; 2],
                    ],
                }),
                // 4x8: cells {0, 1} visited.
                BLOCK_4X8 => Some(InterSubBlockSpec {
                    modes: [ZEROMV, NEWMV, ZEROMV, ZEROMV],
                    mvs: [[[0, 0]; 2], [[-8, -16], [0, 0]], [[0, 0]; 2], [[0, 0]; 2]],
                }),
                _ => None,
            };
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: 0,
                y_mode: ZEROMV,
                interp_filter: 0,
                ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                mv: [[0, 0], [0, 0]],
                skip: false,
                segment_id: 0,
                sub,
            }
        };
        let pf = encode_pframe_lossless_layout(
            &hdr,
            &targets,
            &reference,
            None,
            w,
            h,
            partitions,
            &mut leaf_plan,
        )
        .expect("sub-8x8 layout p-frame");

        let decoded = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
        assert_eq!(decoded[0].to_planar_bytes(), f0, "keyframe");
        assert_eq!(decoded[1].to_planar_bytes(), f1, "sub-8x8 P-frame");
    }

    /// **Compound sub-8x8** end-to-end: a P-frame pairing [ LAST,
    /// ALTREF ] on sub-8x8 cells (per-cell two-list NEWMV / ZEROMV /
    /// NEARESTMV) and on a full 8x8 leaf, predicting LAST from the
    /// keyframe and ALTREF from a hidden intra-only frame in another
    /// slot — the §8.5.2 `Round2( p0 + p1, 1 )` compound average runs
    /// per 4x4 `blockIdx` — reconstructs **byte-exact** through
    /// `decode_vp9_sequence`.
    #[test]
    fn lossless_layout_compound_sub8x8_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::frame_writer::InterTreeLeaf;
        use crate::header::FrameType;
        use crate::inter_block_writer::InterSubBlockSpec;
        use crate::mode_info::{
            ALTREF_FRAME, LAST_FRAME, NEARESTMV, NEWMV, NONE_REF_FRAME, ZEROMV,
        };
        use crate::partition::{PARTITION_HORZ, PARTITION_SPLIT};
        use crate::residual::{BLOCK_4X4, BLOCK_8X4};

        let (w, h) = (64u32, 64u32);
        let (wu, hu, cw, ch) = (64usize, 64usize, 32usize, 32usize);
        let f_last = noise_frame(w, h, 0xC0FF_0001); // keyframe / LAST
        let f_alt = noise_frame(w, h, 0xC0FF_0002); // intra-only / ALTREF
        let f_tgt = noise_frame(w, h, 0xC0FF_0003); // P-frame target

        let kf = encode_keyframe_lossless_420(&f_last, w, h).expect("keyframe");

        // Hidden intra-only frame carrying the ALTREF content -> slot 1.
        let mut hdr1 = lossless_keyframe_header(w, h);
        hdr1.frame_type = FrameType::NonKeyFrame;
        hdr1.intra_only = true;
        hdr1.show_frame = false;
        hdr1.reset_frame_context = 3;
        hdr1.refresh_frame_context = false;
        hdr1.refresh_frame_flags = 0x02;
        let alt_planes = [
            padded_plane_from_bytes(&f_alt[..wu * hu], wu, hu, wu, hu),
            padded_plane_from_bytes(&f_alt[wu * hu..wu * hu + cw * ch], cw, ch, cw, ch),
            padded_plane_from_bytes(&f_alt[wu * hu + cw * ch..], cw, ch, cw, ch),
        ];
        let f1 = encode_keyframe_lossless(&hdr1, &alt_planes).expect("intra-only frame");

        let targets = [
            padded_plane_from_bytes(&f_tgt[..wu * hu], wu, hu, wu, hu),
            padded_plane_from_bytes(&f_tgt[wu * hu..wu * hu + cw * ch], cw, ch, cw, ch),
            padded_plane_from_bytes(&f_tgt[wu * hu + cw * ch..], cw, ch, cw, ch),
        ];
        let as_i32 = |px: &[u8]| -> [Vec<i32>; 3] {
            [
                px[..wu * hu].iter().map(|&s| i32::from(s)).collect(),
                px[wu * hu..wu * hu + cw * ch]
                    .iter()
                    .map(|&s| i32::from(s))
                    .collect(),
                px[wu * hu + cw * ch..]
                    .iter()
                    .map(|&s| i32::from(s))
                    .collect(),
            ]
        };
        let last_i = as_i32(&f_last);
        let alt_i = as_i32(&f_alt);
        let reference: [(&[i32], usize); 3] = [
            (last_i[0].as_slice(), wu),
            (last_i[1].as_slice(), cw),
            (last_i[2].as_slice(), cw),
        ];
        let second: [(&[i32], usize); 3] = [
            (alt_i[0].as_slice(), wu),
            (alt_i[1].as_slice(), cw),
            (alt_i[2].as_slice(), cw),
        ];

        // Compound needs the §6.3.12 asymmetric sign bias (ALTREF only)
        // — and per §7.2 setup_past_independence the frame must NOT be
        // error-resilient (error-resilient frames zero the effective
        // sign biases, making compound uncodeable). The §7.2.6
        // UsePrevFrameMvs derivation still yields 0 because the
        // previously-decoded frame (the intra-only ALTREF carrier) is
        // hidden.
        let mut hdr2 = lossless_pframe_header(w, h);
        hdr2.error_resilient_mode = false;
        hdr2.ref_frame_idx = Some([0, 0, 1]);
        hdr2.ref_frame_sign_bias = [false, false, true];
        hdr2.refresh_frame_flags = 0x04;

        let mut partitions = std::collections::HashMap::new();
        partitions.insert((0u32, 0u32, BLOCK_8X8), PARTITION_SPLIT);
        partitions.insert((4, 4, BLOCK_8X8), PARTITION_HORZ);

        let comp = [LAST_FRAME, ALTREF_FRAME];
        let mut leaf_plan = |r: u32,
                             c: u32,
                             subsize: u8,
                             _s: &crate::decode_block::Vp9FrameState|
         -> InterTreeLeaf {
            let (ref_frame, sub, mv) = match (r, c, subsize) {
                // Compound 4x4: two-list NEWMV cells, a ZEROMV cell, and
                // a compound NEARESTMV (block 3 seeds each list's
                // NearestMv from that list's BlockMvs[ 2 ]).
                (0, 0, BLOCK_4X4) => (
                    comp,
                    Some(InterSubBlockSpec {
                        modes: [NEWMV, ZEROMV, NEWMV, NEARESTMV],
                        mvs: [
                            [[16, 8], [-8, 16]],
                            [[0, 0], [0, 0]],
                            [[-8, 24], [8, -8]],
                            [[-8, 24], [8, -8]],
                        ],
                    }),
                    [[0, 0], [0, 0]],
                ),
                // Compound 8x4 (cells {0, 2}).
                (4, 4, BLOCK_8X4) => (
                    comp,
                    Some(InterSubBlockSpec {
                        modes: [NEWMV, ZEROMV, ZEROMV, ZEROMV],
                        mvs: [[[8, -8], [16, 8]], [[0, 0]; 2], [[0, 0]; 2], [[0, 0]; 2]],
                    }),
                    [[0, 0], [0, 0]],
                ),
                // A full compound 8x8 leaf with distinct per-list MVs.
                (2, 2, _) => (comp, None, [[16, 0], [0, 16]]),
                _ => ([LAST_FRAME, NONE_REF_FRAME], None, [[0, 0], [0, 0]]),
            };
            let y_mode = if mv != [[0, 0], [0, 0]] {
                NEWMV
            } else {
                ZEROMV
            };
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: 0,
                y_mode,
                interp_filter: 0,
                ref_frame,
                mv,
                skip: false,
                segment_id: 0,
                sub,
            }
        };
        let pf = encode_pframe_lossless_layout(
            &hdr2,
            &targets,
            &reference,
            Some(&second),
            w,
            h,
            partitions,
            &mut leaf_plan,
        )
        .expect("compound sub-8x8 p-frame");

        let decoded = decode_vp9_sequence(&[&kf, &f1, &pf]).expect("decode");
        assert_eq!(decoded.len(), 2, "intra-only frame is hidden");
        assert_eq!(decoded[0].to_planar_bytes(), f_last, "keyframe");
        assert_eq!(decoded[1].to_planar_bytes(), f_tgt, "compound P-frame");
    }

    /// **4:4:4 sub-8x8** end-to-end: at full-resolution chroma the
    /// §8.5.2 chroma prediction reads each 4x4 cell's own `blockIdx`
    /// vector (no sub-8x8 MV averaging) — a distinct arm from the
    /// 4:2:0 test above. Byte-exact through `decode_vp9_sequence`.
    #[test]
    fn lossless_layout_sub8x8_yuv444_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::frame_writer::InterTreeLeaf;
        use crate::inter_block_writer::InterSubBlockSpec;
        use crate::mode_info::{LAST_FRAME, NEWMV, NONE_REF_FRAME, ZEROMV};
        use crate::partition::{PARTITION_SPLIT, PARTITION_VERT};
        use crate::residual::{BLOCK_4X4, BLOCK_4X8};

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        // 4:4:4 planar: three full-resolution planes.
        let mk = |seed: u64| -> Vec<u8> {
            let mut state = seed;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            };
            (0..3 * n).map(|_| next()).collect()
        };
        let f0 = mk(0x4441);
        let f1 = mk(0x4442);

        let kf = encode_keyframe_lossless_444(&f0, w, h).expect("444 keyframe");

        let mut hdr = lossless_pframe_header(w, h);
        hdr.profile = 1;
        hdr.color_config.subsampling_x = false;
        hdr.color_config.subsampling_y = false;

        let targets = [
            padded_plane_from_bytes(&f1[..n], 64, 64, 64, 64),
            padded_plane_from_bytes(&f1[n..2 * n], 64, 64, 64, 64),
            padded_plane_from_bytes(&f1[2 * n..], 64, 64, 64, 64),
        ];
        let prev: [Vec<i32>; 3] = [
            f0[..n].iter().map(|&s| i32::from(s)).collect(),
            f0[n..2 * n].iter().map(|&s| i32::from(s)).collect(),
            f0[2 * n..].iter().map(|&s| i32::from(s)).collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), 64),
            (prev[1].as_slice(), 64),
            (prev[2].as_slice(), 64),
        ];

        let mut partitions = std::collections::HashMap::new();
        partitions.insert((1u32, 1u32, BLOCK_8X8), PARTITION_SPLIT);
        partitions.insert((6, 3, BLOCK_8X8), PARTITION_VERT);

        let mut leaf_plan = |_r: u32,
                             _c: u32,
                             subsize: u8,
                             _s: &crate::decode_block::Vp9FrameState|
         -> InterTreeLeaf {
            let sub = match subsize {
                BLOCK_4X4 => Some(InterSubBlockSpec {
                    modes: [NEWMV; 4],
                    mvs: [
                        [[16, 8], [0, 0]],
                        [[-8, 16], [0, 0]],
                        [[8, -16], [0, 0]],
                        [[24, 0], [0, 0]],
                    ],
                }),
                BLOCK_4X8 => Some(InterSubBlockSpec {
                    modes: [NEWMV, ZEROMV, ZEROMV, ZEROMV],
                    mvs: [[[-16, -8], [0, 0]], [[0, 0]; 2], [[0, 0]; 2], [[0, 0]; 2]],
                }),
                _ => None,
            };
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: 0,
                y_mode: ZEROMV,
                interp_filter: 0,
                ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                mv: [[0, 0], [0, 0]],
                skip: false,
                segment_id: 0,
                sub,
            }
        };
        let pf = encode_pframe_lossless_layout(
            &hdr,
            &targets,
            &reference,
            None,
            w,
            h,
            partitions,
            &mut leaf_plan,
        )
        .expect("444 sub-8x8 p-frame");

        let decoded = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
        assert_eq!(decoded[0].to_planar_bytes(), f0, "444 keyframe");
        assert_eq!(decoded[1].to_planar_bytes(), f1, "444 sub-8x8 P-frame");
    }

    /// **4:2:2 and 4:4:0 sub-8x8** end-to-end: half-resolution chroma
    /// on one axis makes the §8.5.2 sub-8x8 chroma prediction average
    /// exactly *two* luma cell vectors per chroma 4x4 (the third
    /// averaging arm, between 4:2:0's four-cell average and 4:4:4's
    /// none). Byte-exact through `decode_vp9_sequence` for both
    /// orientations.
    #[test]
    fn lossless_layout_sub8x8_yuv422_and_yuv440_roundtrip_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::frame_writer::InterTreeLeaf;
        use crate::inter_block_writer::InterSubBlockSpec;
        use crate::mode_info::{LAST_FRAME, NEWMV, NONE_REF_FRAME, ZEROMV};
        use crate::partition::{PARTITION_HORZ, PARTITION_SPLIT};
        use crate::residual::{BLOCK_4X4, BLOCK_8X4};

        for (ssx, ssy) in [(true, false), (false, true)] {
            let (w, h) = (64u32, 64u32);
            let n = (w * h) as usize;
            let cw = 64usize >> usize::from(ssx);
            let ch = 64usize >> usize::from(ssy);
            let cn = cw * ch;
            let mk = |seed: u64| -> Vec<u8> {
                let mut state = seed;
                let mut next = move || {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 33) as u8
                };
                (0..n + 2 * cn).map(|_| next()).collect()
            };
            let f0 = mk(0x2242 + u64::from(ssx));
            let f1 = mk(0x2243 + u64::from(ssy));

            let planes_of = |px: &[u8]| -> [Plane; 3] {
                [
                    padded_plane_from_bytes(&px[..n], 64, 64, 64, 64),
                    padded_plane_from_bytes(&px[n..n + cn], cw, ch, cw, ch),
                    padded_plane_from_bytes(&px[n + cn..], cw, ch, cw, ch),
                ]
            };
            let hdr_kf = lossless_keyframe_header_ex(w, h, 1, 8, ssx, ssy);
            let kf = encode_keyframe_lossless(&hdr_kf, &planes_of(&f0)).expect("keyframe");

            let mut hdr = lossless_pframe_header(w, h);
            hdr.profile = 1;
            hdr.color_config.subsampling_x = ssx;
            hdr.color_config.subsampling_y = ssy;

            let targets = planes_of(&f1);
            let prev: [Vec<i32>; 3] = [
                f0[..n].iter().map(|&s| i32::from(s)).collect(),
                f0[n..n + cn].iter().map(|&s| i32::from(s)).collect(),
                f0[n + cn..].iter().map(|&s| i32::from(s)).collect(),
            ];
            let reference: [(&[i32], usize); 3] = [
                (prev[0].as_slice(), 64),
                (prev[1].as_slice(), cw),
                (prev[2].as_slice(), cw),
            ];

            let mut partitions = std::collections::HashMap::new();
            partitions.insert((2u32, 5u32, BLOCK_8X8), PARTITION_SPLIT);
            partitions.insert((5, 1, BLOCK_8X8), PARTITION_HORZ);

            let mut leaf_plan = |_r: u32,
                                 _c: u32,
                                 subsize: u8,
                                 _s: &crate::decode_block::Vp9FrameState|
             -> InterTreeLeaf {
                let sub = match subsize {
                    // Four distinct vectors => the chroma pair averages
                    // differ per cell on both axes.
                    BLOCK_4X4 => Some(InterSubBlockSpec {
                        modes: [NEWMV; 4],
                        mvs: [
                            [[16, 8], [0, 0]],
                            [[-8, 16], [0, 0]],
                            [[8, -16], [0, 0]],
                            [[24, 0], [0, 0]],
                        ],
                    }),
                    BLOCK_8X4 => Some(InterSubBlockSpec {
                        modes: [NEWMV, ZEROMV, NEWMV, ZEROMV],
                        mvs: [
                            [[8, -8], [0, 0]],
                            [[0, 0]; 2],
                            [[-16, 8], [0, 0]],
                            [[0, 0]; 2],
                        ],
                    }),
                    _ => None,
                };
                InterTreeLeaf {
                    mi_size: subsize,
                    tx_size: 0,
                    y_mode: ZEROMV,
                    interp_filter: 0,
                    ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                    mv: [[0, 0], [0, 0]],
                    skip: false,
                    segment_id: 0,
                    sub,
                }
            };
            let pf = encode_pframe_lossless_layout(
                &hdr,
                &targets,
                &reference,
                None,
                w,
                h,
                partitions,
                &mut leaf_plan,
            )
            .expect("chroma-geometry sub-8x8 p-frame");

            let decoded = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
            assert_eq!(decoded[0].to_planar_bytes(), f0, "keyframe (ssx={ssx})");
            assert_eq!(
                decoded[1].to_planar_bytes(),
                f1,
                "sub-8x8 P-frame (ssx={ssx}, ssy={ssy})"
            );
        }
    }

    /// **10-bit sub-8x8** end-to-end (profile 2, 4:2:0): the sub-8x8
    /// per-`blockIdx` §8.5.2 prediction runs the bit-depth-scaled
    /// convolution clamps and the §8.7.2 WHT carries HBD-range
    /// residual. Byte-exact through `decode_vp9_sequence`.
    #[test]
    fn lossless_layout_sub8x8_10bit_roundtrips_byte_exact() {
        use crate::decode_frame::decode_vp9_sequence;
        use crate::frame_writer::InterTreeLeaf;
        use crate::inter_block_writer::InterSubBlockSpec;
        use crate::mode_info::{LAST_FRAME, NEWMV, NONE_REF_FRAME, ZEROMV};
        use crate::partition::PARTITION_SPLIT;
        use crate::residual::BLOCK_4X4;

        let (w, h) = (64u32, 64u32);
        let (n, cn) = (64usize * 64, 32usize * 32);
        let mk = |seed: u64| -> Vec<u16> {
            let mut state = seed;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) & 0x3FF) as u16
            };
            (0..n + 2 * cn).map(|_| next()).collect()
        };
        let f0 = mk(0x10B1);
        let f1 = mk(0x10B2);

        let kf = encode_keyframe_lossless_hbd(&f0, w, h, 10, true).expect("10-bit keyframe");

        let mut hdr = lossless_pframe_header(w, h);
        hdr.profile = 2;
        hdr.color_config.bit_depth = 10;

        let targets = [
            padded_plane_from_u16(&f1[..n], 64, 64, 64, 64),
            padded_plane_from_u16(&f1[n..n + cn], 32, 32, 32, 32),
            padded_plane_from_u16(&f1[n + cn..], 32, 32, 32, 32),
        ];
        let prev: [Vec<i32>; 3] = [
            f0[..n].iter().map(|&s| i32::from(s)).collect(),
            f0[n..n + cn].iter().map(|&s| i32::from(s)).collect(),
            f0[n + cn..].iter().map(|&s| i32::from(s)).collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), 64),
            (prev[1].as_slice(), 32),
            (prev[2].as_slice(), 32),
        ];

        let mut partitions = std::collections::HashMap::new();
        partitions.insert((4u32, 4u32, BLOCK_8X8), PARTITION_SPLIT);

        let mut leaf_plan = |_r: u32,
                             _c: u32,
                             subsize: u8,
                             _s: &crate::decode_block::Vp9FrameState|
         -> InterTreeLeaf {
            let sub = if subsize == BLOCK_4X4 {
                Some(InterSubBlockSpec {
                    modes: [NEWMV; 4],
                    mvs: [
                        [[16, 8], [0, 0]],
                        [[-8, 16], [0, 0]],
                        [[8, -16], [0, 0]],
                        [[24, 0], [0, 0]],
                    ],
                })
            } else {
                None
            };
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: 0,
                y_mode: ZEROMV,
                interp_filter: 0,
                ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                mv: [[0, 0], [0, 0]],
                skip: false,
                segment_id: 0,
                sub,
            }
        };
        let pf = encode_pframe_lossless_layout(
            &hdr,
            &targets,
            &reference,
            None,
            w,
            h,
            partitions,
            &mut leaf_plan,
        )
        .expect("10-bit sub-8x8 p-frame");

        let decoded = decode_vp9_sequence(&[&kf, &pf]).expect("decode");
        let planar = |px: &[u16]| -> Vec<u8> { px.iter().flat_map(|&s| s.to_le_bytes()).collect() };
        assert_eq!(decoded[0].to_planar_bytes(), planar(&f0), "10-bit keyframe");
        assert_eq!(
            decoded[1].to_planar_bytes(),
            planar(&f1),
            "10-bit sub-8x8 P-frame"
        );
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

    /// The lossy **sequence** encoder's per-frame in-loop reconstruction
    /// equals the decoder's `decode_vp9_sequence` output bit-for-bit —
    /// the chain-level decoder-mirror pin: prediction references, motion
    /// compensation, quantized residuals and reconstructions all agree
    /// across every hop.
    #[test]
    fn lossy_sequence_decode_matches_encoder_recon_chain() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (48u32, 32u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let pattern = |x: i64, y: i64| -> u8 { (((x * 5 + y * 11) % 53) * 4 + (x * y) % 23) as u8 };
        let inputs: Vec<Vec<u8>> = (0..4i64)
            .map(|t| {
                let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
                for i in 0..h as i64 {
                    for j in 0..w as i64 {
                        px.push(pattern(j + 2 * t, i + t));
                    }
                }
                for i in 0..ch as i64 {
                    for j in 0..cw as i64 {
                        px.push(pattern(j + t + 30, i));
                    }
                }
                for i in 0..ch as i64 {
                    for j in 0..cw as i64 {
                        px.push(pattern(j, i + t + 30));
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();

        let coded = encode_sequence_lossy_420(&refs, w, h, 70).expect("encode");
        assert_eq!(coded.len(), 4);
        let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode");

        // Distortion stays bounded across the chain (no drift): each
        // frame's MSE against its source is in the quantizer regime.
        for (i, (frame, input)) in decoded.iter().zip(&inputs).enumerate() {
            let out = frame.to_planar_bytes();
            let mse: f64 = out
                .iter()
                .zip(input)
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum::<f64>()
                / out.len() as f64;
            assert!(mse < 400.0, "frame {i}: MSE {mse} out of the q=70 regime");
        }

        // Bit-exact mirror: re-encode and compare the final frame's
        // ReconState against the decode (the sequence API discards the
        // intermediate states, so rebuild the last hop explicitly).
        let w_us = w as usize;
        let y_w = (((w + 7) >> 3) * 8) as usize;
        let y_h = (((h + 7) >> 3) * 8) as usize;
        let last = inputs.last().unwrap();
        let targets = [
            padded_plane_from_bytes(&last[..w_us * h as usize], w_us, h as usize, y_w, y_h),
            padded_plane_from_bytes(
                &last[w_us * h as usize..w_us * h as usize + cw * ch],
                cw,
                ch,
                y_w >> 1,
                y_h >> 1,
            ),
            padded_plane_from_bytes(
                &last[w_us * h as usize + cw * ch..],
                cw,
                ch,
                y_w >> 1,
                y_h >> 1,
            ),
        ];
        // Reference = decoder's frame 2 output (== encoder recon);
        // GOLDEN = the keyframe's output, exactly as the sequence
        // encoder parks it in slot 1.
        let prev_frame = &decoded[2];
        let prev: [Vec<i32>; 3] = [
            prev_frame.y.iter().map(|&s| i32::from(s)).collect(),
            prev_frame.u.iter().map(|&s| i32::from(s)).collect(),
            prev_frame.v.iter().map(|&s| i32::from(s)).collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), w_us),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        let kf_frame = &decoded[0];
        let gold: [Vec<i32>; 3] = [
            kf_frame.y.iter().map(|&s| i32::from(s)).collect(),
            kf_frame.u.iter().map(|&s| i32::from(s)).collect(),
            kf_frame.v.iter().map(|&s| i32::from(s)).collect(),
        ];
        let golden_ref: [(&[i32], usize); 3] = [
            (gold[0].as_slice(), w_us),
            (gold[1].as_slice(), cw),
            (gold[2].as_slice(), cw),
        ];
        let mut hdr = lossless_pframe_header(w, h);
        hdr.ref_frame_idx = Some([0, 1, 1]);
        hdr.ref_frame_sign_bias = [false, false, true];
        hdr.quantization = QuantizationParams {
            base_q_idx: 70,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let encode_at = |hdr: &Vp9FrameHeader| {
            encode_pframe_lossy_tree_motion_with_state(
                hdr,
                &targets,
                &reference,
                Some(&golden_ref),
                w,
                h,
                PFRAME_SEARCH_RANGE,
                true,
                true, // the sequence encoder runs with sub-8x8 election on
            )
        };
        let (p0, recon0, state0) = encode_at(&hdr).expect("re-encode last hop");
        // Replay the sequence encoder's per-frame filter-level election
        // + §8.8 recon filtering — the last hop's exact close-out.
        let (bytes, recon) = finish_frame_with_filter(
            &hdr,
            p0,
            recon0,
            state0,
            &targets,
            w_us,
            h as usize,
            |hdr2| encode_at(hdr2),
        )
        .expect("elect + filter last hop");
        assert_eq!(bytes, coded[3], "re-encoded last frame must be identical");
        // The decoder's output IS the filtered reconstruction — the
        // §8.8 encode-side mirror holds sample-exactly.
        let last_decoded = &decoded[3];
        for row in 0..h as usize {
            for col in 0..w as usize {
                assert_eq!(
                    i32::from(last_decoded.y[row * w_us + col]),
                    recon.planes[0].get(col, row),
                    "luma mirror ({col},{row})"
                );
            }
        }
    }

    /// A/B pin for the encode-side §8.8 loop filter: on gently-graded
    /// content at a coarse quantizer, the sequence encoder's per-frame
    /// filter-level election (a) codes a non-zero `loop_filter_level`
    /// on the keyframe, (b) leaves the keyframe's coded size exactly
    /// unchanged (the §6.2.8 level field is fixed-width — filtering is
    /// rate-free on the frame that elects it), and (c) lands the whole
    /// decoded GOP strictly closer to the source than the identical
    /// chain with filtering forced off.
    #[test]
    fn filter_election_improves_gop_quality_at_equal_keyframe_rate() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 48u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let q = 140u8;
        // Slow diagonal ramp translating between frames: coarse
        // quantization leaves small block-edge steps (inside the
        // §8.8.5.1 filterMask thresholds) that the filter smooths back
        // toward the source.
        let frame_at = |t: i64| -> Vec<u8> {
            let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    px.push((100 + (x * 3 + y * 2 + 5 * t) / 4 % 48) as u8);
                }
            }
            for y in 0..ch as i64 {
                for x in 0..cw as i64 {
                    px.push((90 + (x + y * 3 + 2 * t) / 3 % 30) as u8);
                }
            }
            for y in 0..ch as i64 {
                for x in 0..cw as i64 {
                    px.push((130 + (x * 2 + y + 3 * t) / 5 % 26) as u8);
                }
            }
            px
        };
        let inputs: Vec<Vec<u8>> = (0..3).map(frame_at).collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();

        // A: the sequence encoder with election (the shipping path).
        let elected = encode_sequence_lossy_420(&refs, w, h, q).expect("elected encode");

        // (a) The keyframe header carries a non-zero elected level.
        let kf_hdr = crate::header::parse_uncompressed_header(&elected[0]).expect("kf header");
        assert!(
            kf_hdr.loop_filter.level > 0,
            "coarse-q graded content should elect keyframe filtering"
        );

        // B: the identical chain with filtering forced off — the
        // pre-round-420 encoder, replayed via the same primitives.
        let padded = |px: &[u8]| -> [Plane; 3] {
            let y_w = (((w + 7) >> 3) * 8) as usize;
            let y_h = (((h + 7) >> 3) * 8) as usize;
            let wu = w as usize;
            let hu = h as usize;
            [
                padded_plane_from_bytes(&px[..wu * hu], wu, hu, y_w, y_h),
                padded_plane_from_bytes(&px[wu * hu..wu * hu + cw * ch], cw, ch, y_w / 2, y_h / 2),
                padded_plane_from_bytes(&px[wu * hu + cw * ch..], cw, ch, y_w / 2, y_h / 2),
            ]
        };
        let crop3 = |r: &ReconState| -> [Vec<i32>; 3] {
            let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
                let mut out = Vec::with_capacity(vw * vh);
                for y in 0..vh {
                    for x in 0..vw {
                        out.push(p.get(x, y));
                    }
                }
                out
            };
            [
                crop(&r.planes[0], w as usize, h as usize),
                crop(&r.planes[1], cw, ch),
                crop(&r.planes[2], cw, ch),
            ]
        };
        let (kf0, kf_recon0) =
            encode_keyframe_lossy_420_with_recon(&inputs[0], w, h, q).expect("kf level 0");
        // (b) Same coded size: the elected keyframe differs only in the
        // fixed-width §6.2.8 level field.
        assert_eq!(
            elected[0].len(),
            kf0.len(),
            "keyframe rate must be unchanged"
        );
        assert_ne!(elected[0], kf0, "elected keyframe must differ (level bits)");

        let mut unfiltered = vec![kf0];
        let golden = crop3(&kf_recon0);
        let golden_ref: [(&[i32], usize); 3] = [
            (golden[0].as_slice(), w as usize),
            (golden[1].as_slice(), cw),
            (golden[2].as_slice(), cw),
        ];
        let mut prev_recon = kf_recon0;
        for px in inputs.iter().skip(1) {
            let targets = padded(px);
            let prev = crop3(&prev_recon);
            let reference: [(&[i32], usize); 3] = [
                (prev[0].as_slice(), w as usize),
                (prev[1].as_slice(), cw),
                (prev[2].as_slice(), cw),
            ];
            let mut hdr = lossless_pframe_header(w, h);
            hdr.ref_frame_idx = Some([0, 1, 1]);
            hdr.ref_frame_sign_bias = [false, false, true];
            hdr.quantization = QuantizationParams {
                base_q_idx: q,
                delta_q_y_dc: 0,
                delta_q_uv_dc: 0,
                delta_q_uv_ac: 0,
                lossless: false,
            };
            let (bytes, recon) = encode_pframe_lossy_tree_motion(
                &hdr,
                &targets,
                &reference,
                Some(&golden_ref),
                w,
                h,
                PFRAME_SEARCH_RANGE,
                true,
                true,
            )
            .expect("unfiltered p-frame");
            unfiltered.push(bytes);
            prev_recon = recon;
        }

        // (c) Whole-GOP SSE vs source: elected strictly better. (The
        // measured run: keyframe elects level 49; GOP SSE 78850 vs
        // 85340 — a 7.6% distortion cut — at an identical 443-byte
        // total rate on both chains.)
        let sse_of = |coded: &[Vec<u8>]| -> u64 {
            let refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
            let decoded = decode_vp9_sequence(&refs).expect("decode");
            decoded
                .iter()
                .zip(&inputs)
                .map(|(f, src)| {
                    f.to_planar_bytes()
                        .iter()
                        .zip(src.iter())
                        .map(|(&a, &b)| {
                            let d = i64::from(a) - i64::from(b);
                            (d * d) as u64
                        })
                        .sum::<u64>()
                })
                .sum()
        };
        let sse_elected = sse_of(&elected);
        let sse_unfiltered = sse_of(&unfiltered);
        assert!(
            sse_elected < sse_unfiltered,
            "elected filtering must strictly improve the GOP \
             ({sse_elected} vs {sse_unfiltered})"
        );
    }

    /// The lossy sequence rejections mirror the lossless ones plus the
    /// lossless-qindex guard.
    #[test]
    fn lossy_sequence_rejects_bad_inputs() {
        let ok = vec![0u8; 16 * 16 + 2 * 64];
        assert_eq!(
            encode_sequence_lossy_420(&[], 16, 16, 50).unwrap_err(),
            Error::Unsupported
        );
        assert_eq!(
            encode_sequence_lossy_420(&[&ok], 16, 16, 0).unwrap_err(),
            Error::Unsupported
        );
        let short = vec![0u8; 4];
        assert_eq!(
            encode_sequence_lossy_420(&[&ok, &short], 16, 16, 50).unwrap_err(),
            Error::Unsupported
        );
    }

    /// On translating content the `NEWMV` motion search codes a
    /// substantially smaller P-frame than forced `ZEROMV` — and both
    /// stay byte-exact through the full decoder (which also pins the
    /// §6.5 predictor derivation and the §6.4.20 hp-gate snapping
    /// against the decode side, since any mismatch desyncs the stream).
    #[test]
    fn motion_search_beats_zeromv_on_translating_content() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 48u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        // Textured pattern translated by (dy, dx) = (3, 5) px between
        // frames (sampled from a shared infinite pattern so the motion
        // is real, not a wrap-around).
        let pattern = |x: i64, y: i64| -> u8 { (((x * 7 + y * 13) % 61) * 4 + (x + y) % 17) as u8 };
        let frame_at = |ox: i64, oy: i64| -> Vec<u8> {
            let mut px = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
            for i in 0..h as i64 {
                for j in 0..w as i64 {
                    px.push(pattern(j + ox, i + oy));
                }
            }
            for i in 0..ch as i64 {
                for j in 0..cw as i64 {
                    px.push(pattern(j + ox / 2 + 40, i + oy / 2));
                }
            }
            for i in 0..ch as i64 {
                for j in 0..cw as i64 {
                    px.push(pattern(j + ox / 2, i + oy / 2 + 40));
                }
            }
            px
        };
        let f0 = frame_at(0, 0);
        let f1 = frame_at(5, 3);

        // Shared setup for both P-frame encodes.
        let wl = w as usize;
        let y_w = (((w + 7) >> 3) * 8) as usize;
        let y_h = (((h + 7) >> 3) * 8) as usize;
        let targets = [
            padded_plane_from_bytes(&f1[..wl * h as usize], wl, h as usize, y_w, y_h),
            padded_plane_from_bytes(
                &f1[wl * h as usize..wl * h as usize + cw * ch],
                cw,
                ch,
                y_w >> 1,
                y_h >> 1,
            ),
            padded_plane_from_bytes(&f1[wl * h as usize + cw * ch..], cw, ch, y_w >> 1, y_h >> 1),
        ];
        let prev: [Vec<i32>; 3] = [
            f0[..wl * h as usize]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            f0[wl * h as usize..wl * h as usize + cw * ch]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            f0[wl * h as usize + cw * ch..]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), wl),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        let hdr = lossless_pframe_header(w, h);

        let with_motion = encode_pframe_lossless_motion(&hdr, &targets, &reference, w, h, 8, true)
            .expect("motion");
        let zero_only = encode_pframe_lossless_motion(&hdr, &targets, &reference, w, h, 0, true)
            .expect("zeromv");
        assert!(
            with_motion.len() * 2 < zero_only.len(),
            "motion search ({} B) should code far fewer bits than ZEROMV ({} B)",
            with_motion.len(),
            zero_only.len()
        );

        // Both must still be byte-exact through the full decoder.
        let kf = encode_keyframe_lossless_420(&f0, w, h).expect("keyframe");
        for pf in [&with_motion, &zero_only] {
            let decoded = decode_vp9_sequence(&[&kf, pf]).expect("decode");
            assert_eq!(decoded[1].to_planar_bytes(), f1, "P-frame not byte-exact");
        }
    }

    /// Build a 4:2:0 frame pair `(f0, f1)` where `f1`'s luma is `f0`'s
    /// shifted by exactly **half a pixel** horizontally (each sample the
    /// rounded average of two neighbours of a smooth ramp — no integer
    /// vector can explain it), chroma flat.
    fn halfpel_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
        let (cw, ch) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        let ramp = |x: i64, y: i64| -> i64 { (4 * x).min(180) + y + 10 };
        let mut f0 = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
        let mut f1 = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
        for y in 0..i64::from(h) {
            for x in 0..i64::from(w) {
                f0.push(ramp(x, y) as u8);
                // ref(x + 0.5): the average of ref(x) and ref(x + 1).
                f1.push(((ramp(x, y) + ramp(x + 1, y) + 1) / 2) as u8);
            }
        }
        for _ in 0..2 * cw * ch {
            f0.push(128);
            f1.push(128);
        }
        (f0, f1)
    }

    /// Sub-pel refinement pays on half-pel motion: the lossless P-frame
    /// with quarter/eighth-pel search codes fewer bytes than the
    /// full-pel-only search on content translated by exactly half a
    /// pixel, and both stay byte-exact through the decoder.
    #[test]
    fn subpel_search_beats_fullpel_on_halfpel_motion() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 48u32);
        let (f0, f1) = halfpel_pair(w, h);

        let y_w = (((w + 7) >> 3) * 8) as usize;
        let y_h = (((h + 7) >> 3) * 8) as usize;
        let (cw, ch) = (32usize, 24usize);
        let wl = w as usize;
        let targets = [
            padded_plane_from_bytes(&f1[..wl * h as usize], wl, h as usize, y_w, y_h),
            padded_plane_from_bytes(
                &f1[wl * h as usize..wl * h as usize + cw * ch],
                cw,
                ch,
                y_w >> 1,
                y_h >> 1,
            ),
            padded_plane_from_bytes(&f1[wl * h as usize + cw * ch..], cw, ch, y_w >> 1, y_h >> 1),
        ];
        let prev: [Vec<i32>; 3] = [
            f0[..wl * h as usize]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            f0[wl * h as usize..wl * h as usize + cw * ch]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            f0[wl * h as usize + cw * ch..]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), wl),
            (prev[1].as_slice(), cw),
            (prev[2].as_slice(), cw),
        ];
        let hdr = lossless_pframe_header(w, h);

        let subpel = encode_pframe_lossless_motion(&hdr, &targets, &reference, w, h, 8, true)
            .expect("subpel");
        let fullpel = encode_pframe_lossless_motion(&hdr, &targets, &reference, w, h, 8, false)
            .expect("fullpel");
        assert!(
            subpel.len() < fullpel.len(),
            "sub-pel search ({} B) should beat full-pel-only ({} B) on half-pel motion",
            subpel.len(),
            fullpel.len()
        );

        // Both remain byte-exact through the full decoder.
        let kf = encode_keyframe_lossless_420(&f0, w, h).expect("keyframe");
        for pf in [&subpel, &fullpel] {
            let decoded = decode_vp9_sequence(&[&kf, pf]).expect("decode");
            assert_eq!(decoded[1].to_planar_bytes(), f1, "P-frame not byte-exact");
        }
    }

    /// The lossy tree encoder's sub-pel refinement also pays on half-pel
    /// motion, and the sub-pel frame still mirrors the decoder exactly.
    #[test]
    fn lossy_tree_subpel_beats_fullpel_on_halfpel_motion() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 64u32);
        let (f0, f1) = halfpel_pair(w, h);
        // Low quantizer: the half-pel residual (±2 on the ramp) must
        // survive quantization, or both variants skip everything.
        let q = 4u8;
        let (kf, kf_recon) = encode_keyframe_lossy_420_with_recon(&f0, w, h, q).expect("kf");
        let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(vw * vh);
            for y in 0..vh {
                for x in 0..vw {
                    out.push(p.get(x, y));
                }
            }
            out
        };
        let prev = [
            crop(&kf_recon.planes[0], 64, 64),
            crop(&kf_recon.planes[1], 32, 32),
            crop(&kf_recon.planes[2], 32, 32),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), 64),
            (prev[1].as_slice(), 32),
            (prev[2].as_slice(), 32),
        ];
        let targets = padded_targets_420(&f1, w, h);
        let mut hdr = lossless_pframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: q,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };

        let (subpel, subpel_recon) =
            encode_pframe_lossy_tree_motion(&hdr, &targets, &reference, None, w, h, 8, true, false)
                .expect("subpel");
        let (fullpel, _) = encode_pframe_lossy_tree_motion(
            &hdr, &targets, &reference, None, w, h, 8, false, false,
        )
        .expect("fullpel");
        assert!(
            subpel.len() < fullpel.len(),
            "sub-pel tree ({} B) should beat full-pel tree ({} B) on half-pel motion",
            subpel.len(),
            fullpel.len()
        );

        // Decoder mirror for the sub-pel frame.
        let decoded = decode_vp9_sequence(&[&kf, &subpel]).expect("decode");
        let d = &decoded[1];
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    i32::from(d.y[y * 64 + x]),
                    subpel_recon.planes[0].get(x, y),
                    "luma mirror ({x},{y})"
                );
            }
        }
    }

    /// Multi-reference election pays when content returns to the
    /// keyframe: on an A → B → A sequence the third frame codes far
    /// fewer bytes with the keyframe parked as `GOLDEN` (leaves elect
    /// `GOLDEN` + `ZEROMV` + skip) than with `LAST` alone, and the
    /// multi-ref frame decodes to exactly the encoder's reconstruction
    /// through the §8.10 slot threading.
    #[test]
    fn multiref_golden_pays_on_content_returning_to_keyframe() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let pattern = |x: i64, y: i64, ph: i64| -> u8 {
            ((((x + ph) * 7 + y * 13) % 61) * 4 + (x + y) % 17) as u8
        };
        let frame_with_phase = |ph: i64| -> Vec<u8> {
            let mut px = Vec::with_capacity(n);
            for i in 0..h as i64 {
                for j in 0..w as i64 {
                    px.push(pattern(j, i, ph));
                }
            }
            px.extend(std::iter::repeat_n(128u8, 2 * 32 * 32));
            px
        };
        let fa = frame_with_phase(0);
        // Unrelated content for B: LCG noise (nothing motion search or
        // the texture period can explain from A).
        let fb: Vec<u8> = {
            let mut v = Vec::with_capacity(n);
            let mut s: u64 = 0x1234_5678_9abc_def0;
            for i in 0..n {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push(if i < (w * h) as usize {
                    (s >> 33) as u8
                } else {
                    128
                });
            }
            v
        };
        let q = 60u8;

        let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(vw * vh);
            for y in 0..vh {
                for x in 0..vw {
                    out.push(p.get(x, y));
                }
            }
            out
        };
        fn as_ref_planes(v: &[Vec<i32>; 3]) -> [(&[i32], usize); 3] {
            [
                (v[0].as_slice(), 64),
                (v[1].as_slice(), 32),
                (v[2].as_slice(), 32),
            ]
        }

        // Keyframe A, then P1 = B (referencing A).
        let (kf, kf_recon) = encode_keyframe_lossy_420_with_recon(&fa, w, h, q).expect("kf");
        let gold = [
            crop(&kf_recon.planes[0], 64, 64),
            crop(&kf_recon.planes[1], 32, 32),
            crop(&kf_recon.planes[2], 32, 32),
        ];
        let mut hdr = lossless_pframe_header(w, h);
        hdr.ref_frame_idx = Some([0, 1, 1]);
        hdr.quantization = QuantizationParams {
            base_q_idx: q,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (p1, p1_recon) = encode_pframe_lossy_tree_motion(
            &hdr,
            &padded_targets_420(&fb, w, h),
            &as_ref_planes(&gold),
            None,
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("p1");
        let prev = [
            crop(&p1_recon.planes[0], 64, 64),
            crop(&p1_recon.planes[1], 32, 32),
            crop(&p1_recon.planes[2], 32, 32),
        ];

        // P2 = A again: with GOLDEN available it should mostly skip.
        let targets2 = padded_targets_420(&fa, w, h);
        let (p2_multi, p2_recon) = encode_pframe_lossy_tree_motion(
            &hdr,
            &targets2,
            &as_ref_planes(&prev),
            Some(&as_ref_planes(&gold)),
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("p2 multi");
        let (p2_single, _) = encode_pframe_lossy_tree_motion(
            &hdr,
            &targets2,
            &as_ref_planes(&prev),
            None,
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("p2 single");
        assert!(
            p2_multi.len() * 4 < p2_single.len(),
            "GOLDEN election ({} B) should be far below LAST-only ({} B) on A-B-A content",
            p2_multi.len(),
            p2_single.len()
        );

        // Decoder mirror through the §8.10 slots: keyframe fills every
        // slot, P-frames refresh only slot 0, so GOLDEN (slot 1) is the
        // keyframe when P2 decodes.
        let decoded = decode_vp9_sequence(&[&kf, &p1, &p2_multi]).expect("decode");
        let d = &decoded[2];
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    i32::from(d.y[y * 64 + x]),
                    p2_recon.planes[0].get(x, y),
                    "luma mirror ({x},{y})"
                );
            }
        }
        for y in 0..32usize {
            for x in 0..32usize {
                assert_eq!(i32::from(d.u[y * 32 + x]), p2_recon.planes[1].get(x, y));
                assert_eq!(i32::from(d.v[y * 32 + x]), p2_recon.planes[2].get(x, y));
            }
        }
    }

    /// Compound prediction pays on a cross-fade: a frame that is the
    /// pixel average of the keyframe (A) and the previous frame (B)
    /// codes far fewer bytes when the [ LAST, ALTREF ] compound average
    /// is available (sign-bias asymmetry admits it) than with single
    /// references only — and the compound frame decodes to exactly the
    /// encoder's reconstruction.
    #[test]
    fn compound_prediction_pays_on_crossfade() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let noise = |seed: u64| -> Vec<u8> {
            let mut v = Vec::with_capacity(n);
            let mut s = seed;
            for i in 0..n {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push(if i < (w * h) as usize {
                    (s >> 33) as u8
                } else {
                    128
                });
            }
            v
        };
        let fa = noise(0x1111_2222_3333_4444);
        let fb = noise(0x9999_8888_7777_6666);
        // The cross-fade midpoint.
        let fmid: Vec<u8> = fa
            .iter()
            .zip(&fb)
            .map(|(&a, &b)| (u16::from(a) + u16::from(b)).div_ceil(2) as u8)
            .collect();
        let q = 60u8;

        fn as_ref_planes(v: &[Vec<i32>; 3]) -> [(&[i32], usize); 3] {
            [
                (v[0].as_slice(), 64),
                (v[1].as_slice(), 32),
                (v[2].as_slice(), 32),
            ]
        }
        let crop3 = |r: &ReconState| -> [Vec<i32>; 3] {
            let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
                let mut out = Vec::with_capacity(vw * vh);
                for y in 0..vh {
                    for x in 0..vw {
                        out.push(p.get(x, y));
                    }
                }
                out
            };
            [
                crop(&r.planes[0], 64, 64),
                crop(&r.planes[1], 32, 32),
                crop(&r.planes[2], 32, 32),
            ]
        };

        let (kf, kf_recon) = encode_keyframe_lossy_420_with_recon(&fa, w, h, q).expect("kf");
        let gold = crop3(&kf_recon);

        // P1 codes the second noise pattern — HIDDEN, so the compound
        // frame's §7.2.6 UsePrevFrameMvs derivation yields 0 (a
        // non-error-resilient frame is required for compound: per §7.2
        // setup_past_independence, error-resilient frames zero the
        // effective sign biases and compoundReferenceAllowed with them).
        let mut hdr1 = lossless_pframe_header(w, h);
        hdr1.show_frame = false;
        hdr1.ref_frame_idx = Some([0, 1, 1]);
        hdr1.quantization = QuantizationParams {
            base_q_idx: q,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (p1, p1_recon) = encode_pframe_lossy_tree_motion(
            &hdr1,
            &padded_targets_420(&fb, w, h),
            &as_ref_planes(&gold),
            None,
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("p1");
        let prev = crop3(&p1_recon);

        // Compound-capable header: sign-bias asymmetry on ALTREF +
        // error_resilient_mode == 0 (hidden predecessor ⇒ no prev MVs).
        let mut hdr = hdr1;
        hdr.show_frame = true;
        hdr.error_resilient_mode = false;
        hdr.ref_frame_sign_bias = [false, false, true];
        let targets2 = padded_targets_420(&fmid, w, h);
        let (p2_comp, p2_recon) = encode_pframe_lossy_tree_motion(
            &hdr,
            &targets2,
            &as_ref_planes(&prev),
            Some(&as_ref_planes(&gold)),
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("p2 compound");
        // Baseline: same references, but symmetric sign biases forbid
        // compound (single-reference election only).
        let mut hdr_single = hdr;
        hdr_single.ref_frame_sign_bias = [false, false, false];
        let (p2_single, _) = encode_pframe_lossy_tree_motion(
            &hdr_single,
            &targets2,
            &as_ref_planes(&prev),
            Some(&as_ref_planes(&gold)),
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("p2 single");
        assert!(
            p2_comp.len() * 2 < p2_single.len(),
            "compound average ({} B) should be far below single-ref ({} B) on a cross-fade",
            p2_comp.len(),
            p2_single.len()
        );

        // Decoder mirror for the compound frame (P1 is hidden).
        let decoded = decode_vp9_sequence(&[&kf, &p1, &p2_comp]).expect("decode");
        let d = &decoded[1];
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    i32::from(d.y[y * 64 + x]),
                    p2_recon.planes[0].get(x, y),
                    "luma mirror ({x},{y})"
                );
            }
        }
        for y in 0..32usize {
            for x in 0..32usize {
                assert_eq!(i32::from(d.u[y * 32 + x]), p2_recon.planes[1].get(x, y));
                assert_eq!(i32::from(d.v[y * 32 + x]), p2_recon.planes[2].get(x, y));
            }
        }
    }

    /// Shared texture for the inter-tree planner tests.
    fn tree_pattern(x: i64, y: i64) -> i32 {
        ((((x * 7 + y * 13) % 61) * 4 + (x + y) % 17) & 0xff) as i32
    }

    /// The inter partition planner merges uniform-motion superblocks into
    /// one `BLOCK_64X64` leaf (with the searched vector as its hint) and
    /// splits superblocks whose 8x8 cells elected different vectors.
    #[test]
    fn inter_partition_planner_merges_uniform_motion_and_splits_mixed() {
        use crate::partition::{PARTITION_NONE, PARTITION_SPLIT};
        use crate::residual::{BLOCK_32X32, BLOCK_64X64};

        let (mi_cols, mi_rows) = (8u32, 8u32); // one 64x64 superblock
        let (w, h) = (64usize, 64usize);
        let ref_y: Vec<i32> = (0..h as i64)
            .flat_map(|y| (0..w as i64).map(move |x| tree_pattern(x, y)))
            .collect();
        let flat_uv: Vec<i32> = vec![128; (w / 2) * (h / 2)];
        let reference: [(&[i32], usize); 3] = [
            (ref_y.as_slice(), w),
            (flat_uv.as_slice(), w / 2),
            (flat_uv.as_slice(), w / 2),
        ];
        let mk_targets = |luma: &dyn Fn(i64, i64) -> i32| -> [Plane; 3] {
            let mut t = [
                Plane::new(w, h),
                Plane::new(w / 2, h / 2),
                Plane::new(w / 2, h / 2),
            ];
            for y in 0..h {
                for x in 0..w {
                    t[0].set(x, y, luma(x as i64, y as i64));
                }
            }
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    t[1].set(x, y, 128);
                    t[2].set(x, y, 128);
                }
            }
            t
        };

        // Uniform global (dy, dx) = (2, 3) motion: cur(x, y) =
        // ref(x+3, y+2), with the source coordinates clamped to the
        // frame exactly like the §8.5.2.4 edge-clamped reference read —
        // so the vector is a perfect match on edge cells too.
        let uniform = mk_targets(&|x, y| tree_pattern((x + 3).min(63), (y + 2).min(63)));
        let (parts, hints, _sub) =
            plan_inter_partitions(&uniform, &reference, 64, 64, mi_cols, mi_rows, 8, false);
        assert_eq!(
            parts.get(&(0, 0, BLOCK_64X64)),
            Some(&PARTITION_NONE),
            "uniform motion must merge to one 64x64 leaf"
        );
        assert_eq!(
            hints.get(&(0, 0)),
            Some(&(2, 3)),
            "merged leaf carries the MV"
        );

        // Mixed motion: the top-left 32x32 quadrant moves by (2, 3), the
        // rest is static. The root splits; each 32x32 quadrant (uniform
        // within itself) merges.
        let mixed = mk_targets(&|x, y| {
            if x < 32 && y < 32 {
                tree_pattern(x + 3, y + 2)
            } else {
                tree_pattern(x, y)
            }
        });
        let (parts, hints, _sub) =
            plan_inter_partitions(&mixed, &reference, 64, 64, mi_cols, mi_rows, 8, false);
        assert_eq!(
            parts.get(&(0, 0, BLOCK_64X64)),
            Some(&PARTITION_SPLIT),
            "mixed motion must split the superblock"
        );
        assert_eq!(parts.get(&(0, 0, BLOCK_32X32)), Some(&PARTITION_NONE));
        assert_eq!(parts.get(&(0, 4, BLOCK_32X32)), Some(&PARTITION_NONE));
        assert_eq!(hints.get(&(0, 0)), Some(&(2, 3)), "moving quadrant MV");
        assert_eq!(hints.get(&(0, 4)), Some(&(0, 0)), "static quadrant MV");
    }

    /// Alias-free texture for the sub-8x8 election tests: a spatial
    /// hash, so no two distinct small shifts of the plane agree on any
    /// 4x4 block (the structured `tree_pattern` aliases at 4x4 scale).
    fn sub8x8_texture(x: i64, y: i64) -> i32 {
        let v = (x as u64)
            .wrapping_add((y as u64).wrapping_mul(131))
            .wrapping_add(7);
        let h = v
            .wrapping_mul(v)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(v.wrapping_mul(97));
        ((h >> 24) & 0xff) as i32
    }

    /// The luma displacement field of the sub-8x8 election tests: three
    /// 8x8 cells whose 4x4 quadrants move divergently — MI (2,2) left /
    /// right halves at `(0, ±4)` (VERT), MI (3,6) top / bottom halves
    /// at `(±4, 0)` (HORZ), MI (5,5) all four quadrants distinct
    /// (SPLIT) — everything else static.
    fn sub8x8_disp(x: i64, y: i64) -> (i64, i64) {
        let (cr, cc) = (y / 8, x / 8); // MI cell
        let (qr, qc) = ((y % 8) / 4, (x % 8) / 4); // quadrant within it
        match (cr, cc) {
            (2, 2) => (0, if qc == 0 { 4 } else { -4 }),
            (3, 6) => (if qr == 0 { 4 } else { -4 }, 0),
            (5, 5) => (if qr == 0 { 4 } else { -4 }, if qc == 0 { 4 } else { -4 }),
            _ => (0, 0),
        }
    }

    /// With `sub8x8` enabled the planner elects below-8x8 leaves on
    /// cells whose quadrants move divergently — the shape follows which
    /// quadrant pairs agree (VERT / HORZ / SPLIT) and the hints carry
    /// the per-cell vectors — while static cells and the surrounding
    /// tree are untouched; with the probe disabled the same content
    /// plans no sub-8x8 nodes.
    #[test]
    fn planner_elects_sub8x8_on_divergent_quadrant_motion() {
        use crate::partition::{PARTITION_HORZ, PARTITION_SPLIT, PARTITION_VERT};
        use crate::residual::BLOCK_8X8;

        let (mi_cols, mi_rows) = (8u32, 8u32);
        let (w, h) = (64usize, 64usize);
        let ref_y: Vec<i32> = (0..h as i64)
            .flat_map(|y| (0..w as i64).map(move |x| sub8x8_texture(x, y)))
            .collect();
        let flat_uv: Vec<i32> = vec![128; (w / 2) * (h / 2)];
        let reference: [(&[i32], usize); 3] = [
            (ref_y.as_slice(), w),
            (flat_uv.as_slice(), w / 2),
            (flat_uv.as_slice(), w / 2),
        ];
        let mut targets = [
            Plane::new(w, h),
            Plane::new(w / 2, h / 2),
            Plane::new(w / 2, h / 2),
        ];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (dy, dx) = sub8x8_disp(x, y);
                targets[0].set(x as usize, y as usize, sub8x8_texture(x + dx, y + dy));
            }
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                targets[1].set(x, y, 128);
                targets[2].set(x, y, 128);
            }
        }

        let (parts, _hints, sub) =
            plan_inter_partitions(&targets, &reference, 64, 64, mi_cols, mi_rows, 8, true);
        assert_eq!(sub.len(), 3, "exactly the three divergent cells split");

        let v = sub.get(&(2, 2)).expect("VERT cell");
        assert_eq!(v.partition, PARTITION_VERT);
        assert_eq!(v.cell_mvs[0], (0, 4));
        assert_eq!(v.cell_mvs[1], (0, -4));
        assert_eq!(parts.get(&(2, 2, BLOCK_8X8)), Some(&PARTITION_VERT));

        let hz = sub.get(&(3, 6)).expect("HORZ cell");
        assert_eq!(hz.partition, PARTITION_HORZ);
        assert_eq!(hz.cell_mvs[0], (4, 0));
        assert_eq!(hz.cell_mvs[2], (-4, 0));
        assert_eq!(parts.get(&(3, 6, BLOCK_8X8)), Some(&PARTITION_HORZ));

        let sp = sub.get(&(5, 5)).expect("SPLIT cell");
        assert_eq!(sp.partition, PARTITION_SPLIT);
        assert_eq!(
            sp.cell_mvs,
            [(4, 4), (4, -4), (-4, 4), (-4, -4)],
            "per-quadrant vectors in §6.4.16 cell layout"
        );
        assert_eq!(parts.get(&(5, 5, BLOCK_8X8)), Some(&PARTITION_SPLIT));

        // Disabled probe: no sub-8x8 nodes on the same content.
        let (_p2, _h2, sub_off) =
            plan_inter_partitions(&targets, &reference, 64, 64, mi_cols, mi_rows, 8, false);
        assert!(sub_off.is_empty(), "probe off ⇒ no sub-8x8 election");
    }

    /// Sub-8x8 election pays on divergent sub-block motion — **rate and
    /// quality together**: against the same lossless keyframe reference,
    /// the sub-8x8-enabled encode of a frame whose 8x8 cells contain
    /// opposing quadrant motion is strictly smaller *and* strictly
    /// closer to the target than the 8x8-limited encode (the elected
    /// leaves predict exactly and skip; the 8x8 encoder must code a
    /// large mismatch residual). The coded stream decodes byte-exact
    /// against the encoder's reconstruction (decoder mirror).
    #[test]
    fn sub8x8_election_beats_8x8_on_rate_and_quality() {
        let (w, h) = (64u32, 64u32);
        // Keyframe content: the tree_pattern texture (lossless keyframe
        // ⇒ the decoder's reference IS this pattern).
        let mut kf_px = vec![0u8; (64 * 64 + 2 * 32 * 32) as usize];
        for y in 0..64i64 {
            for x in 0..64i64 {
                kf_px[(y * 64 + x) as usize] = sub8x8_texture(x, y) as u8;
            }
        }
        for px in kf_px.iter_mut().skip(64 * 64) {
            *px = 128;
        }
        let kf = crate::encode_vp9(&kf_px, w, h).expect("lossless keyframe");

        // P-frame target: the displacement field of the planner test.
        let mut targets = [Plane::new(64, 64), Plane::new(32, 32), Plane::new(32, 32)];
        for y in 0..64i64 {
            for x in 0..64i64 {
                let (dy, dx) = sub8x8_disp(x, y);
                targets[0].set(x as usize, y as usize, sub8x8_texture(x + dx, y + dy));
            }
        }
        for y in 0..32 {
            for x in 0..32 {
                targets[1].set(x, y, 128);
                targets[2].set(x, y, 128);
            }
        }
        let ref_y: Vec<i32> = (0..64i64)
            .flat_map(|y| (0..64i64).map(move |x| sub8x8_texture(x, y)))
            .collect();
        let flat_uv: Vec<i32> = vec![128; 32 * 32];
        let reference: [(&[i32], usize); 3] = [
            (ref_y.as_slice(), 64),
            (flat_uv.as_slice(), 32),
            (flat_uv.as_slice(), 32),
        ];

        let mut hdr = lossless_pframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: 80,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let encode = |sub8x8: bool| {
            encode_pframe_lossy_tree_motion(
                &hdr,
                &targets,
                &reference,
                None,
                w,
                h,
                PFRAME_SEARCH_RANGE,
                false,
                sub8x8,
            )
            .expect("lossy tree p-frame")
        };
        let (bytes_on, recon_on) = encode(true);
        let (bytes_off, recon_off) = encode(false);

        let sse = |recon: &ReconState| -> u64 {
            leaf_sse(
                &targets,
                &recon.planes,
                0,
                0,
                crate::residual::BLOCK_64X64,
                8,
                8,
                true,
                true,
            )
        };
        let (sse_on, sse_off) = (sse(&recon_on), sse(&recon_off));
        eprintln!(
            "sub-8x8 A/B: on = {} B / SSE {sse_on}, off = {} B / SSE {sse_off}",
            bytes_on.len(),
            bytes_off.len()
        );
        assert!(
            bytes_on.len() < bytes_off.len(),
            "sub-8x8 rate: {} B (on) vs {} B (off)",
            bytes_on.len(),
            bytes_off.len()
        );
        assert!(
            sse_on < sse_off,
            "sub-8x8 quality: SSE {sse_on} (on) vs {sse_off} (off)"
        );

        // Decoder mirror for the sub-8x8-elected frame.
        let decoded = crate::decode_vp9_sequence(&[&kf, &bytes_on]).expect("decode");
        let d = &decoded[1];
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    i32::from(d.y[y * 64 + x]),
                    recon_on.planes[0].get(x, y),
                    "luma mirror ({x},{y})"
                );
            }
        }
        for y in 0..32usize {
            for x in 0..32usize {
                assert_eq!(i32::from(d.u[y * 32 + x]), recon_on.planes[1].get(x, y));
                assert_eq!(i32::from(d.v[y * 32 + x]), recon_on.planes[2].get(x, y));
            }
        }
    }

    /// The per-leaf inter transform-size election adapts to the residual:
    /// a smooth low-frequency residual over a 64x64 leaf elects
    /// `TX_32X32` (fewest coded blocks, energy compacts into few
    /// coefficients), while a residual whose energy is dense
    /// high-frequency noise confined to one 8x8 block elects a small
    /// transform (isolating the busy block keeps every other block
    /// all-zero at one end-of-block bit each).
    #[test]
    fn inter_leaf_tx_election_adapts_to_residual_shape() {
        use crate::residual::BLOCK_64X64;

        let (mi_cols, mi_rows) = (8u32, 8u32);
        let (w, h) = (64usize, 64usize);
        // Prediction (work planes) = flat 128.
        let mut work = [
            Plane::new(w, h),
            Plane::new(w / 2, h / 2),
            Plane::new(w / 2, h / 2),
        ];
        for p in &mut work {
            let (pw, ph) = (p.width(), p.height());
            for y in 0..ph {
                for x in 0..pw {
                    p.set(x, y, 128);
                }
            }
        }
        let seg = SegmentationParams::default_disabled();
        let quant = QuantizationParams {
            base_q_idx: 60,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };

        // Smooth gradient residual: target = 128 + (x + y) / 2.
        let mut smooth = [work[0].clone(), work[1].clone(), work[2].clone()];
        for y in 0..h {
            for x in 0..w {
                smooth[0].set(x, y, 128 + ((x + y) / 2) as i32);
            }
        }
        let (tx, _, all_zero) = select_inter_leaf_tx(
            &smooth,
            &work,
            0,
            0,
            BLOCK_64X64,
            mi_cols,
            mi_rows,
            true,
            true,
            8,
            &seg,
            &quant,
        );
        assert!(!all_zero, "gradient residual must quantize to tokens");
        assert_eq!(tx, 3, "smooth residual should elect TX_32X32");

        // Dense ±100 checkerboard noise confined to the 8x8 block at
        // (16, 16); zero residual elsewhere.
        let mut noisy = [work[0].clone(), work[1].clone(), work[2].clone()];
        for y in 16..24 {
            for x in 16..24 {
                let s = if (x + y) % 2 == 0 { 228 } else { 28 };
                noisy[0].set(x, y, s);
            }
        }
        let (tx, _, all_zero) = select_inter_leaf_tx(
            &noisy,
            &work,
            0,
            0,
            BLOCK_64X64,
            mi_cols,
            mi_rows,
            true,
            true,
            8,
            &seg,
            &quant,
        );
        assert!(!all_zero);
        assert!(
            tx <= 1,
            "localised high-frequency residual should elect a small transform (got TX id {tx})"
        );
    }

    /// On uniform-motion content the tree P-frame encoder (one 64x64
    /// NEWMV leaf, per-leaf tx selection) codes fewer bytes than the
    /// fixed all-`BLOCK_8X8` / inferred-`TX_8X8` baseline at the same
    /// quantizer, and the coded frame still decodes to exactly the
    /// encoder's reconstruction.
    #[test]
    fn lossy_tree_pframe_beats_fixed_8x8_on_uniform_motion() {
        use crate::decode_frame::decode_vp9_sequence;

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize + 2 * 32 * 32;
        let pattern = |x: i64, y: i64| -> u8 { (((x * 7 + y * 13) % 61) * 4 + (x + y) % 17) as u8 };
        let frame_at = |ox: i64, oy: i64| -> Vec<u8> {
            let mut px = Vec::with_capacity(n);
            for i in 0..h as i64 {
                for j in 0..w as i64 {
                    px.push(pattern(j + ox, i + oy));
                }
            }
            for _ in 0..2 {
                for i in 0..32i64 {
                    for j in 0..32i64 {
                        px.push(pattern(j + ox / 2, i + oy / 2));
                    }
                }
            }
            px
        };
        let f0 = frame_at(0, 0);
        let f1 = frame_at(5, 3);

        let q = 60u8;
        let (kf, kf_recon) = encode_keyframe_lossy_420_with_recon(&f0, w, h, q).expect("kf");
        let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(vw * vh);
            for y in 0..vh {
                for x in 0..vw {
                    out.push(p.get(x, y));
                }
            }
            out
        };
        let prev = [
            crop(&kf_recon.planes[0], 64, 64),
            crop(&kf_recon.planes[1], 32, 32),
            crop(&kf_recon.planes[2], 32, 32),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), 64),
            (prev[1].as_slice(), 32),
            (prev[2].as_slice(), 32),
        ];
        let targets = padded_targets_420(&f1, w, h);
        let mut hdr = lossless_pframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: q,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };

        let (tree, tree_recon) = encode_pframe_lossy_tree_motion(
            &hdr,
            &targets,
            &reference,
            None,
            w,
            h,
            PFRAME_SEARCH_RANGE,
            true,
            false,
        )
        .expect("tree p-frame");
        let (fixed, _) =
            encode_pframe_lossy_motion(&hdr, &targets, &reference, w, h, PFRAME_SEARCH_RANGE)
                .expect("fixed p-frame");
        assert!(
            tree.len() < fixed.len(),
            "adaptive tree ({} B) should beat the fixed 8x8 layout ({} B)",
            tree.len(),
            fixed.len()
        );

        // Decoder mirror: the coded tree frame reconstructs to exactly
        // the encoder's in-loop state on all three planes.
        let decoded = decode_vp9_sequence(&[&kf, &tree]).expect("decode");
        let d = &decoded[1];
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    i32::from(d.y[y * 64 + x]),
                    tree_recon.planes[0].get(x, y),
                    "luma mirror ({x},{y})"
                );
            }
        }
        for y in 0..32usize {
            for x in 0..32usize {
                assert_eq!(i32::from(d.u[y * 32 + x]), tree_recon.planes[1].get(x, y));
                assert_eq!(i32::from(d.v[y * 32 + x]), tree_recon.planes[2].get(x, y));
            }
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
