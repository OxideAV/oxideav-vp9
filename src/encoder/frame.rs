//! Top-level keyframe assembly — stitches the §6.2 uncompressed
//! header, §6.3 compressed header, and tile payload into a single
//! VP9 access-unit byte buffer.
//!
//! This is what callers hand to an IVF / WebM muxer or to the VP9
//! decoder's `send_packet`.

use crate::compressed_header::TxMode;
use crate::encoder::compressed_header::emit_compressed_header;
use crate::encoder::inter::build_pframe;
use crate::encoder::params::{EncoderParams, ReferenceFrame, ReferenceSet, YuvFrame};
use crate::encoder::tile::emit_keyframe_tile;
use crate::encoder::tile_pixel::build_pixel_keyframe;
use crate::encoder::uncompressed_header::emit_uncompressed_header;

/// Build one complete VP9 keyframe from an [`EncoderParams`] alone.
/// Emits a DC_PRED / skip=1 / ONLY_4X4 keyframe that reconstructs to
/// midgrey (sample 128 in every plane). Useful for ingest / muxer
/// plumbing tests where the pixel content is irrelevant. For
/// correctness on arbitrary input use [`encode_keyframe_yuv`].
pub fn encode_keyframe(p: &EncoderParams) -> Vec<u8> {
    let tx_mode = TxMode::Only4x4;
    let ch = emit_compressed_header(tx_mode, false);
    let tile = emit_keyframe_tile(p, tx_mode);
    let uh = emit_uncompressed_header(p, ch.len() as u16);
    let mut out = Vec::with_capacity(uh.len() + ch.len() + tile.len());
    out.extend_from_slice(&uh);
    out.extend_from_slice(&ch);
    out.extend_from_slice(&tile);
    out
}

/// Encode a keyframe carrying source pixel content.
///
/// Round 2: uses forward 4×4 DCT + quantisation + VP9 token coding
/// to encode the residual between the DC_PRED predictor and the source
/// pixels. The resulting PSNR is well above 35 dB on smooth fixtures
/// at the default `base_q_idx = 64`, and gracefully degrades on higher-
/// frequency content as expected for a fixed-QP encoder.
///
/// The bitstream is self-consistent: it decodes through this crate's
/// own `Vp9Decoder` and through ffmpeg without errors.
pub fn encode_keyframe_yuv(p: &EncoderParams, src: &YuvFrame<'_>) -> Vec<u8> {
    build_pixel_keyframe(p, src.y, src.y_stride, src.u, src.v, src.uv_stride)
}

/// Encode a P-frame against a reconstructed LAST_FRAME reference.
///
/// Single-reference convenience wrapper. For multi-reference (LAST +
/// GOLDEN per-CU RDO) use [`encode_pframe_yuv_multi_ref`].
///
/// Returns a full VP9 access-unit byte buffer that can be packetised
/// into IVF / WebM or fed to `Vp9Decoder::send_packet`. The caller is
/// responsible for keeping the reference frame in sync — for a typical
/// key + P chain that's the reconstruction of the preceding keyframe
/// (which the caller obtains by decoding their own
/// `encode_keyframe_yuv` output).
pub fn encode_pframe_yuv(p: &EncoderParams, src: &YuvFrame<'_>, refr: &ReferenceFrame) -> Vec<u8> {
    let refs = ReferenceSet {
        last: refr,
        golden: None,
    };
    build_pframe(p, src, &refs)
}

/// Encode a P-frame with multi-reference RDO between LAST_FRAME and
/// GOLDEN_FRAME (r-multiref round). For every CU the encoder runs ME
/// against both reference planes and picks the lower-SAD reference
/// (with a small bit-rate penalty for GOLDEN's extra `single_ref_p2`
/// bit). Per-block `ref_frame[0]` is signalled via the §6.4.5
/// single-ref tree.
///
/// DPB convention (mirrored in the uncompressed header):
/// * `refs.last`   → DPB slot 0. The encoder asks the decoder to
///   refresh slot 0 after this P-frame (`refresh_frame_flags = 0x01`),
///   so the next P-frame's `last` will be this P-frame's reconstruction.
/// * `refs.golden` → DPB slot 1. The keyframe writes all 8 slots
///   (`refresh_frame_flags = 0xFF`), so slot 1 stays the keyframe as
///   long as P-frames refresh only slot 0.
///
/// When `refs.golden` is `None` this collapses to single-LAST exactly
/// like [`encode_pframe_yuv`].
pub fn encode_pframe_yuv_multi_ref(
    p: &EncoderParams,
    src: &YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
) -> Vec<u8> {
    build_pframe(p, src, refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::make_decoder;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

    #[test]
    fn encode_keyframe_parses_back() {
        let p = EncoderParams::keyframe(64, 64);
        let frame = encode_keyframe(&p);
        let codec_id = CodecId::new(crate::CODEC_ID_STR);
        let params = CodecParameters::video(codec_id);
        let mut d = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 30), frame);
        d.send_packet(&pkt).unwrap();
        let f = d.receive_frame().unwrap();
        match f {
            Frame::Video(v) => {
                assert_eq!(v.planes[0].stride, 64);
                assert_eq!(v.planes[0].data.len(), 64 * 64);
            }
            _ => panic!("expected Video frame"),
        }
    }
}
