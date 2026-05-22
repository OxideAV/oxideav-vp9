//! Integration tests for the VP9 §6.3 compressed-header walker
//! (round 3). Each test:
//!
//!  1. Builds a `uncompressed_header()` buffer with a known
//!     `header_size_in_bytes`, slicing off the byte-aligned tail.
//!  2. Appends a hand-computed §9.2 Boolean-coder buffer producing
//!     the desired `tx_mode` value.
//!  3. Calls `parse_uncompressed_header` to recover
//!     `uncompressed_header_size_bytes`, then
//!     `parse_compressed_header` against the next
//!     `header_size_in_bytes` of the buffer.
//!
//! No external library was consulted — the §9.2 byte vectors here
//! are the same golden buffers verified in
//! `src/compressed.rs`'s unit tests.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`).

use oxideav_vp9::{parse_compressed_header, parse_uncompressed_header, FrameType, TxMode};

/// Minimal MSB-first bit builder mirroring §9.1 read order.
struct BitBuilder {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitBuilder {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn push_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_index = self.bit_pos >> 3;
            if byte_index >= self.bytes.len() {
                self.bytes.push(0);
            }
            let bit_in_byte = 7 - (self.bit_pos & 7);
            self.bytes[byte_index] |= bit << bit_in_byte;
            self.bit_pos += 1;
        }
    }
    fn align_to_byte(&mut self) {
        while self.bit_pos & 7 != 0 {
            self.push_bits(0, 1);
        }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Builds a 64x64 key-frame uncompressed header with `header_size`
/// stamped into the f(16) `header_size_in_bytes` slot and all
/// other fields minimal (loop_filter / quant / segmentation off,
/// tile_info minimum log2's). Returns the byte vector ready to be
/// concatenated with `header_size` bytes of compressed-header
/// payload.
fn build_uncompressed_64x64_key(header_size: u16, lossless: bool) -> Vec<u8> {
    let mut b = BitBuilder::new();
    // frame_marker = 2
    b.push_bits(2, 2);
    // Profile 0
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit

    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame = 1
    b.push_bits(0, 1); // error_resilient_mode = 0

    // frame_sync_code 0x49 0x83 0x42
    b.push_bits(0x49, 8);
    b.push_bits(0x83, 8);
    b.push_bits(0x42, 8);
    // color_config (profile 0): color_space + color_range.
    b.push_bits(1, 3); // CS_BT_601
    b.push_bits(0, 1); // studio swing

    // frame_size 64x64 → minus_1 = 63 each.
    b.push_bits(63, 16);
    b.push_bits(63, 16);
    // render_and_frame_size_different = 0
    b.push_bits(0, 1);

    // Tail for error_resilient_mode == 0 + key frame:
    b.push_bits(0, 1); // refresh_frame_context
    b.push_bits(0, 1); // frame_parallel_decoding_mode
    b.push_bits(0, 2); // frame_context_idx

    // loop_filter_params: level/sharpness/delta_enabled = 0.
    b.push_bits(0, 6);
    b.push_bits(0, 3);
    b.push_bits(0, 1);

    // quantization_params:
    if lossless {
        b.push_bits(0, 8); // base_q_idx = 0
        b.push_bits(0, 1); // delta_coded y_dc = 0
        b.push_bits(0, 1); // delta_coded uv_dc = 0
        b.push_bits(0, 1); // delta_coded uv_ac = 0
    } else {
        b.push_bits(42, 8); // base_q_idx nonzero -> Lossless = false
        b.push_bits(0, 1);
        b.push_bits(0, 1);
        b.push_bits(0, 1);
    }

    // segmentation_params: enabled = 0.
    b.push_bits(0, 1);

    // tile_info: 64x64 → Sb64Cols = 1, min_log2 = 0, max_log2 = 0.
    // No increment loop bits required (min == max).
    // tile_rows_log2 first bit = 0.
    b.push_bits(0, 1);

    // header_size_in_bytes
    b.push_bits(header_size as u32, 16);

    // trailing_bits zero pad to byte boundary.
    b.align_to_byte();
    b.finish()
}

#[test]
fn end_to_end_tx_mode_only_4x4_lossless() {
    // §9.2 buffer (any valid marker buffer works on the lossless path).
    let payload = vec![0x00u8, 0x00, 0x00, 0x00];
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, true);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).expect("uncompressed header walks");
    assert_eq!(h.frame_type, FrameType::KeyFrame);
    assert_eq!(h.frame_width, 64);
    assert_eq!(h.frame_height, 64);
    assert_eq!(h.header_size_in_bytes, header_size);
    assert_eq!(h.uncompressed_header_size_bytes, uncompressed_end);
    assert!(h.quantization.lossless);

    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless)
        .expect("compressed header walks");
    assert_eq!(c.tx_mode, TxMode::Only4x4);
}

#[test]
fn end_to_end_tx_mode_select() {
    // Non-lossless frame; golden buffer 0x70 → TX_MODE_SELECT.
    let payload = vec![0x70u8, 0x00, 0x00, 0x00];
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, false);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).expect("uncompressed header walks");
    assert!(!h.quantization.lossless);
    assert_eq!(h.header_size_in_bytes, header_size);

    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless)
        .expect("compressed header walks");
    assert_eq!(c.tx_mode, TxMode::TxModeSelect);
}

#[test]
fn end_to_end_tx_mode_allow_16x16() {
    let payload = vec![0x40u8, 0x00, 0x00, 0x00];
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, false);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).expect("uncompressed header walks");
    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless).unwrap();
    assert_eq!(c.tx_mode, TxMode::Allow16x16);
}

#[test]
fn empty_compressed_payload_rejected() {
    let empty: [u8; 0] = [];
    assert!(parse_compressed_header(&empty, false).is_err());
}

#[test]
fn end_to_end_tx_mode_select_runs_tx_mode_probs_and_skip_prob() {
    // TX_MODE_SELECT path with a zero-filled tail. Round 5 fires the
    // §6.3.2 tx_mode_probs sweep then the §6.3.8 read_skip_prob
    // sweep; on a zero buffer every B(252) update_prob decodes to 0
    // so the post-sweep tables equal their §10 defaults.
    let mut payload = vec![0u8; 16];
    payload[0] = 0x70; // L(2)=3, L(1)=1 → TX_MODE_SELECT.
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, false);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).expect("uncompressed header walks");
    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless)
        .expect("compressed header walks");
    assert_eq!(c.tx_mode, TxMode::TxModeSelect);
    // The §10 default tables survive the zero-buffer sweep verbatim.
    assert_eq!(c.tx_probs[1], [[100, 0, 0], [66, 0, 0]]);
    assert_eq!(c.tx_probs[2], [[20, 152, 0], [15, 101, 0]]);
    assert_eq!(c.tx_probs[3], [[3, 136, 37], [5, 52, 13]]);
    assert_eq!(c.skip_prob, [192, 128, 64]);
}

#[test]
fn end_to_end_non_select_tx_mode_skips_tx_mode_probs_sweep() {
    // ALLOW_16X16 (0x40 prefix): the §6.3.2 tx_mode_probs sweep is
    // gated on TX_MODE_SELECT and must NOT fire. §6.3.8
    // read_skip_prob still runs.
    let payload = vec![0x40u8, 0x00, 0x00, 0x00];
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, false);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).unwrap();
    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless).unwrap();
    assert_eq!(c.tx_mode, TxMode::Allow16x16);
    // §10 defaults preserved since tx_mode_probs() was not invoked.
    assert_eq!(c.tx_probs[1], [[100, 0, 0], [66, 0, 0]]);
    assert_eq!(c.skip_prob, [192, 128, 64]);
}

#[test]
fn end_to_end_tx_mode_select_runs_coef_probs_sweep() {
    // Round-6 end-to-end: TX_MODE_SELECT path drives
    // read_tx_mode → tx_mode_probs (§6.3.2) → read_coef_probs (§6.3.7)
    // → read_skip_prob (§6.3.8). On a zero-filled payload every outer
    // L(1) update_probs decodes to 0 across all four tx-size slabs, so
    // the §10 default coef-probability anchors survive verbatim.
    let mut payload = vec![0u8; 16];
    payload[0] = 0x70; // L(2)=3, L(1)=1 → TX_MODE_SELECT.
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, false);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).expect("uncompressed header walks");
    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless)
        .expect("compressed header walks");
    assert_eq!(c.tx_mode, TxMode::TxModeSelect);
    // Pick a representative anchor from the §10 default_coef_probs
    // listing: TX_4X4 / block-type 0 / Intra / Coeff Band 0 / context
    // 0 → { 195, 29, 183 }.
    assert_eq!(c.coef_probs[0][0][0][0][0], [195, 29, 183]);
    // Another anchor: TX_32X32 / block-type 1 / Inter / Coeff Band 5
    // / context 5 → { 1, 16, 6 } (the trailing entry of the §10
    // listing).
    assert_eq!(c.coef_probs[3][1][1][5][5], [1, 16, 6]);
}

#[test]
fn end_to_end_only_4x4_visits_only_first_tx_size_coef_slab() {
    // ONLY_4X4: §6.3.7 outer loop visits tx-size 0 only.
    // §6.3.2 is skipped (TX_MODE_SELECT gate); §6.3.7 reads ONE outer
    // L(1) flag and (on a zero buffer) makes no inner updates.
    let payload = vec![0u8; 8];
    let header_size = payload.len() as u16;
    let mut buffer = build_uncompressed_64x64_key(header_size, false);
    let uncompressed_end = buffer.len();
    buffer.extend_from_slice(&payload);

    let h = parse_uncompressed_header(&buffer).unwrap();
    let payload_slice =
        &buffer[uncompressed_end..uncompressed_end + h.header_size_in_bytes as usize];
    let c = parse_compressed_header(payload_slice, h.quantization.lossless).unwrap();
    assert_eq!(c.tx_mode, TxMode::Only4x4);
    // tx-size 0 anchor preserved.
    assert_eq!(c.coef_probs[0][0][0][0][0], [195, 29, 183]);
    // tx-size 3 anchor preserved (it wasn't touched by the outer
    // loop — maxTxSize was 0).
    assert_eq!(c.coef_probs[3][1][1][5][5], [1, 16, 6]);
    assert_eq!(c.skip_prob, [192, 128, 64]);
}
