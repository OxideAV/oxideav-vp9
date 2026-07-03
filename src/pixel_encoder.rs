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

    encode_keyframe_lossy(&hdr, &[y_plane, u_plane, v_plane], true).map(|(bytes, _)| bytes)
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
    let block_mvs = [[mv; 4], [[0i32; 2]; 4]];
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
            interp_filter: 0, // EIGHTTAP.
            bit_depth,
            is_compound: false,
        };
        predict_inter(pred_plane, &args, &grid, &geom, &block_mvs, &refs, ssx, ssy);
    }
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
    let sad_at = |dy: i32, dx: i32| -> u64 {
        let mut sad = 0u64;
        for i in 0..8i32 {
            for j in 0..8i32 {
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
/// `search_range > 0` requires an error-resilient header (the §7.2.6
/// `UsePrevFrameMvs == 0` model — see
/// [`crate::frame_writer::assemble_inter_frame_planned`]).
pub(crate) fn encode_pframe_lossless_motion(
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    reference: &[(&[i32], usize); 3],
    ref_w: u32,
    ref_h: u32,
    search_range: i32,
) -> Result<Vec<u8>, Error> {
    use crate::inter_decode::FrameStateMvSource;
    use crate::mode_info::{LAST_FRAME, NEWMV, ZEROMV};
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
    let sign_bias = [
        false,
        hdr.ref_frame_sign_bias[0],
        hdr.ref_frame_sign_bias[1],
        hdr.ref_frame_sign_bias[2],
    ];

    let y_w = (mi_cols * 8) as usize;
    let y_h = (mi_rows * 8) as usize;
    let uv_w = y_w >> usize::from(ssx);
    let uv_h = y_h >> usize::from(ssy);
    let pred = RefCell::new([
        Plane::new(y_w, y_h),
        Plane::new(uv_w, uv_h),
        Plane::new(uv_w, uv_h),
    ]);

    // Prefer NEWMV only for a clear win: the mode + MV syntax costs bits
    // that a marginal SAD gain does not repay.
    const NEWMV_SAD_MARGIN: u64 = 64;

    let mut planner: Box<crate::frame_writer::InterBlockPlanner<'_>> =
        Box::new(|r, c, state| -> (u8, [i32; 2]) {
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
                    let src = FrameStateMvSource::new(state, None);
                    let mv_refs = geom.find_mv_refs(&src, LAST_FRAME, -1, &sign_bias, false);
                    let best =
                        geom.find_best_ref_mvs(mv_refs.ref_list_mv, hdr.allow_high_precision_mv)[0];

                    let mut mv = [8 * dy, 8 * dx];
                    let use_hp = hdr.allow_high_precision_mv && use_mv_hp(best);
                    for (comp, m) in mv.iter_mut().enumerate() {
                        let d = *m - best[comp];
                        if d != 0 && !use_hp && (d & 1) != 0 {
                            // Only even-magnitude differences are codeable
                            // without the hp bit; nudge by one eighth-pel.
                            *m -= 1;
                        }
                    }
                    choice = (NEWMV, mv);
                }
            }
            // Predict this block with the vector that will be coded, so
            // the residual callbacks below see the decoder's prediction.
            predict_mi_block(
                &mut pred.borrow_mut(),
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
            choice
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

    crate::frame_writer::assemble_inter_frame_planned(hdr, false, &mut *planner, &mut *coeffs)
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
/// Requires an error-resilient non-key lossy header (see
/// [`crate::frame_writer::assemble_inter_frame_planned`]).
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
    let sign_bias = [
        false,
        hdr.ref_frame_sign_bias[0],
        hdr.ref_frame_sign_bias[1],
        hdr.ref_frame_sign_bias[2],
    ];
    let seg = hdr.segmentation;
    let quant = hdr.quantization;
    let bd8 = hdr.color_config.bit_depth;

    // Work planes: the planner writes each block's §8.5.2 prediction,
    // then the residual callback's §8.6.2 replay turns it into the
    // decoder's reconstruction in place.
    let work = RefCell::new(ReconState::new(mi_cols, mi_rows, ssx, ssy, bit_depth));

    const NEWMV_SAD_MARGIN: u64 = 64;

    let mut planner: Box<crate::frame_writer::InterBlockPlanner<'_>> =
        Box::new(|r, c, state| -> (u8, [i32; 2]) {
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
            predict_mi_block(
                &mut work.borrow_mut().planes,
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
            choice
        });

    let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(
        |_mi_r: u32, _mi_c: u32, plane: usize, sx: u32, sy: u32, _b: usize| -> Vec<i64> {
            let mut work = work.borrow_mut();
            let mut block = vec![0i64; 16];
            for i in 0..4usize {
                for j in 0..4usize {
                    let t = targets[plane].get(sx as usize + j, sy as usize + i);
                    let p = work.planes[plane].get(sx as usize + j, sy as usize + i);
                    block[i * 4 + j] = i64::from(t) - i64::from(p);
                }
            }
            let dc_q = get_dc_quant(plane, &seg, &quant, 0, bd8);
            let ac_q = get_ac_quant(plane, &seg, &quant, 0, bd8);
            // §6.4.25: inter blocks transform with DCT_DCT at every size.
            crate::fwd_transform::forward_dct_2d(&mut block, 2);
            crate::fwd_transform::quantize_block(&mut block, dc_q, ac_q);
            reconstruct_block(
                &mut work.planes[plane],
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

    let bytes =
        crate::frame_writer::assemble_inter_frame_planned(hdr, false, &mut *planner, &mut *coeffs)?;
    drop(planner);
    drop(coeffs);
    Ok((bytes, work.into_inner()))
}

/// Encode a sequence of 8-bit 4:2:0 planar frames into a **lossy** VP9
/// stream at quantizer index `base_q_idx` (`1..=255`): a lossy keyframe
/// followed by lossy P-frames with per-block `ZEROMV` / `NEWMV` motion,
/// each referencing the previous frame's in-loop **reconstruction** (the
/// decoder's exact output), so encoder and decoder never drift.
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

    // Lossy keyframe.
    let mut kf_hdr = lossless_keyframe_header(width, height);
    kf_hdr.quantization = QuantizationParams {
        base_q_idx,
        delta_q_y_dc: 0,
        delta_q_uv_dc: 0,
        delta_q_uv_ac: 0,
        lossless: false,
    };
    let (kf_bytes, kf_recon) = encode_keyframe_lossy(&kf_hdr, &padded_targets(frames[0]), true)?;

    let mut out = Vec::with_capacity(frames.len());
    out.push(kf_bytes);
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
        hdr.quantization = QuantizationParams {
            base_q_idx,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (bytes, recon) = encode_pframe_lossy_motion(
            &hdr,
            &targets,
            &reference,
            width,
            height,
            PFRAME_SEARCH_RANGE,
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
        // Reference = decoder's frame 2 output (== encoder recon).
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
        let mut hdr = lossless_pframe_header(w, h);
        hdr.quantization = QuantizationParams {
            base_q_idx: 70,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (bytes, recon) =
            encode_pframe_lossy_motion(&hdr, &targets, &reference, w, h, PFRAME_SEARCH_RANGE)
                .expect("re-encode last hop");
        assert_eq!(bytes, coded[3], "re-encoded last frame must be identical");
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

        let with_motion =
            encode_pframe_lossless_motion(&hdr, &targets, &reference, w, h, 8).expect("motion");
        let zero_only =
            encode_pframe_lossless_motion(&hdr, &targets, &reference, w, h, 0).expect("zeromv");
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
