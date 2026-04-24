//! Top-level keyframe assembly — stitches the §6.2 uncompressed
//! header, §6.3 compressed header, and tile payload into a single
//! VP9 access-unit byte buffer.
//!
//! This is what callers hand to an IVF / WebM muxer or to the VP9
//! decoder's `send_packet`.

use crate::compressed_header::TxMode;
use crate::encoder::compressed_header::emit_compressed_header;
use crate::encoder::params::EncoderParams;
use crate::encoder::tile::emit_keyframe_tile;
use crate::encoder::uncompressed_header::emit_uncompressed_header;

/// Build one complete VP9 keyframe from an [`EncoderParams`] plus the
/// (currently unused — MVP reconstructs to constant midgrey) source
/// YUV planes. Returns the packet-sized byte buffer.
///
/// The MVP encoder ignores the pixel content and emits a DC_PRED /
/// skip=1 / ONLY_4X4 keyframe that reconstructs to midgrey. Forward
/// transforms + tokenisation (steps 5–7 in the crate README) follow
/// in later commits.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::make_decoder;
    use oxideav_codec::Decoder;
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
                assert_eq!(v.width, 64);
                assert_eq!(v.height, 64);
            }
            _ => panic!("expected Video frame"),
        }
    }
}
