//! VP9 uncompressed header emitter — §6.2.
//!
//! Produces the bytes a VP9 decoder (including this crate's own
//! [`crate::headers::parse_uncompressed_header`]) expects before the
//! compressed header. Scope: keyframe, profile 0, 4:2:0 8-bit, single
//! tile, no segmentation, no loop-filter delta, fixed `base_q_idx`.
//!
//! The caller must already know the compressed-header length: the
//! 16-bit `header_size` is part of the uncompressed-header bitstream
//! and lives at an arbitrary bit offset, making post-hoc patching
//! awkward. Emitting in a single pass keeps the code straightforward.

use crate::encoder::bitwriter::BitWriter;
use crate::encoder::params::EncoderParams;

/// VP9 sync code (§6.2 `uncompressed_header`).
const VP9_SYNC_CODE: u32 = 0x49_8342;

/// Emit the keyframe uncompressed header. `compressed_header_size` is
/// the byte length of the §6.3 compressed header that will follow.
pub fn emit_uncompressed_header(p: &EncoderParams, compressed_header_size: u16) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // §6.2: frame_marker (2) = 2.
    bw.write(2, 2);
    // profile_low (1) / profile_high (1) — profile 0 = 00.
    bw.write(0, 1);
    bw.write(0, 1);
    // show_existing_frame = 0.
    bw.bit(false);
    // frame_type: 0 = KEY_FRAME.
    bw.bit(false);
    // show_frame = 1.
    bw.bit(true);
    // error_resilient_mode = 0.
    bw.bit(false);

    // Sync code (24 bits) — constant per spec.
    bw.write(VP9_SYNC_CODE, 24);

    // §6.2.1 color_config — profile 0 — fields:
    //   color_space (3): pick BT601 = 1.
    //   color_range (1): 0 (studio range).
    bw.write(1, 3);
    bw.bit(false);

    // §6.2.2 frame_size:
    //   width_minus_1 (16), height_minus_1 (16).
    bw.write(p.width - 1, 16);
    bw.write(p.height - 1, 16);

    // render_and_frame_size_different = 0.
    bw.bit(false);

    // §6.2 trailing bits before loop_filter (keyframe path):
    //   refresh_frame_context (1)
    //   frame_parallel_decoding_mode (1)
    //   frame_context_idx (2)
    bw.bit(true);
    bw.bit(false);
    bw.write(0, 2);

    // §6.2.3 loop_filter_params — level=0 (disables deblocking).
    bw.write(p.loop_filter_level as u32, 6);
    // sharpness (3).
    bw.write(0, 3);
    // mode_ref_delta_enabled = 0.
    bw.bit(false);

    // §6.2.4 quantization_params.
    bw.write(p.base_q_idx as u32, 8);
    // delta_q_y_dc / delta_q_uv_dc / delta_q_uv_ac — all absent (0 bit each).
    bw.bit(false);
    bw.bit(false);
    bw.bit(false);

    // §6.2.5 segmentation_params — disabled.
    bw.bit(false);

    // §6.2.6 tile_info.
    write_tile_info(&mut bw, p.width);

    // §6.2 trailing header_size (16 bits).
    bw.write(compressed_header_size as u32, 16);

    // Byte-align — parser reads `uncompressed_header_size` as
    // `byte_aligned_position()`.
    bw.finish()
}

fn write_tile_info(bw: &mut BitWriter, width: u32) {
    // §6.2.6 tile_info. We pick `log2_tile_cols = min_log2` (single tile
    // when possible). The decoder reads up to `max_log2 - min_log2`
    // increment bits, stopping on the first 0. So if `min_log2 <
    // max_log2` we emit a single 0 bit; otherwise emit nothing.
    let sb_cols = width.max(1).div_ceil(64);
    let mut min_log2 = 0u32;
    while (64u32 << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let max_log2 = {
        let mut m = 1u32;
        while ((sb_cols + (1 << m) - 1) >> m) >= 4 {
            m += 1;
        }
        m.saturating_sub(1)
    };
    if min_log2 < max_log2 {
        bw.bit(false);
    }
    // log2_tile_rows: 1 bit = 0 (no rows).
    bw.bit(false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::{parse_uncompressed_header, FrameType};

    #[test]
    fn emit_64x64_parses_back() {
        let p = EncoderParams::keyframe(64, 64);
        let mut bytes = emit_uncompressed_header(&p, 0);
        // Append dummy bytes for the (absent) compressed header.
        bytes.extend_from_slice(&[0u8; 4]);
        let h = parse_uncompressed_header(&bytes, None).expect("parse");
        assert_eq!(h.profile, 0);
        assert_eq!(h.frame_type, FrameType::Key);
        assert!(h.show_frame);
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 64);
        assert_eq!(h.color_config.bit_depth, 8);
        assert!(h.color_config.subsampling_x);
        assert!(h.color_config.subsampling_y);
        assert_eq!(h.quantization.base_q_idx, p.base_q_idx);
        assert!(!h.segmentation.enabled);
        assert_eq!(h.tile_info.log2_tile_cols, 0);
        assert_eq!(h.tile_info.log2_tile_rows, 0);
        assert_eq!(h.header_size, 0);
    }

    #[test]
    fn emit_128x96_parses_back() {
        let p = EncoderParams::keyframe(128, 96);
        let bytes = emit_uncompressed_header(&p, 0);
        let mut data = bytes.clone();
        data.extend_from_slice(&[0u8; 4]);
        let h = parse_uncompressed_header(&data, None).expect("parse");
        assert_eq!(h.width, 128);
        assert_eq!(h.height, 96);
    }

    #[test]
    fn emit_carries_header_size() {
        let p = EncoderParams::keyframe(64, 64);
        let mut bytes = emit_uncompressed_header(&p, 1234);
        bytes.extend_from_slice(&[0u8; 4]);
        let h = parse_uncompressed_header(&bytes, None).expect("parse");
        assert_eq!(h.header_size, 1234);
    }

    #[test]
    fn emit_256x256_single_tile() {
        let p = EncoderParams::keyframe(256, 256);
        let mut bytes = emit_uncompressed_header(&p, 0);
        bytes.extend_from_slice(&[0u8; 4]);
        let h = parse_uncompressed_header(&bytes, None).expect("parse");
        assert_eq!(h.width, 256);
        assert_eq!(h.height, 256);
        assert_eq!(h.tile_info.log2_tile_cols, 0);
    }
}
