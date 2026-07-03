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

/// The §6.2 header for a lossless keyframe: profile 0, 8-bit 4:2:0,
/// `base_q_idx == 0` with zero deltas (the §6.2.9 `Lossless` derivation),
/// loop filter off (lossless reconstruction must not be filtered),
/// single tile.
pub(crate) fn lossless_keyframe_header(width: u32, height: u32) -> Vp9FrameHeader {
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
