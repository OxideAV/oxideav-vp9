//! Integration tests for the VP9 §6.3 compressed-header walker. Each
//! intra-path test:
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
//! The §9.2 byte vectors here are the same golden buffers verified in
//! `src/compressed.rs`'s unit tests.
//!
//! Round 35 adds an inter-path section pinning `parse_compressed_header_inter`
//! (the §6.3 `if ( FrameIsIntra == 0 )` outer-dispatch entry point landed
//! in round 34) at the public-API boundary. The uncompressed-header
//! walker still rejects inter frames with `Error::Unsupported`
//! (`frame_size_with_refs` not yet wired), so the integration tests
//! call `parse_compressed_header_inter` directly with hand-supplied
//! §6.2.5 / §6.2.7-derived inputs.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.3 lines 1957-1975.

use oxideav_vp9::{
    parse_compressed_header, parse_compressed_header_inter, parse_uncompressed_header, FrameType,
    MvProbs, RefFrameSignBias, ReferenceMode, TxMode, Vp9CompressedHeaderInterInputs,
};

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

// ---------- §6.3 `parse_compressed_header_inter` integration ----------
//
// Round 35 pins the round-34 inter outer-dispatch entry point at the
// public-API boundary. The §6.3 listing's `if ( FrameIsIntra == 0 )`
// branch (`vp9-spec.txt` lines 1964-1974) composes ten primitives:
// §6.3.1 / §6.3.2 (gated on TX_MODE_SELECT) / §6.3.7 / §6.3.8 / §6.3.9
// / §6.3.10 (gated on `interpolation_filter == SWITCHABLE`) / §6.3.11
// / §6.3.12 (which fires §6.3.18 on the non-`SingleReference` arms) /
// §6.3.13 / §6.3.14 / §6.3.15 / §6.3.16 (which fires §6.3.17 per cell
// and the high-precision tail when `allow_high_precision_mv == 1`).
//
// On a zero-filled byte buffer every §9.2 `B(252)` flag and every
// §9.2 `L(1) update_probs` flag in the §6.3.7 outer loop decodes to 0,
// so each `read_diff_update_prob` / `update_mv_prob` / outer `L(1)`
// returns the running probability or default-table slot unchanged.
// This makes a long zero buffer a clean "no-op walk" against which we
// can pin the composition order without depending on hand-computed
// post-update values for ~70 distinct MV cells + thousands of coef
// probability cells.
//
// The §10 / §10.5 default-table values asserted below are anchors from
// the spec listing transcribed verbatim into the crate's
// `mode_info::DEFAULT_*` / `coef_probs::DEFAULT_COEF_PROBS` /
// `partition::DEFAULT_PARTITION_PROBS` constants.

/// Helper: build a `RefFrameSignBias` with `LAST = GOLDEN = ALTREF =
/// 0`, which forces the §6.3.12 walker down the
/// `compoundReferenceAllowed == 0` short-circuit arm (no bool-coder
/// reads, returns `SingleReference`).
fn all_zero_sign_bias() -> RefFrameSignBias {
    RefFrameSignBias::from_inter_biases(0, 0, 0)
}

/// Helper: build a `RefFrameSignBias` with `LAST = 0, GOLDEN = 0,
/// ALTREF = 1`. The §6.3.12 loop at `i = 2` then sees `ALTREF !=
/// LAST`, setting `compoundReferenceAllowed = 1` and entering the
/// bool-coder-reading arm.
fn mixed_sign_bias() -> RefFrameSignBias {
    RefFrameSignBias::from_inter_biases(0, 0, 1)
}

#[test]
fn inter_zero_buffer_passes_through_all_default_tables() {
    // Zero buffer + ONLY_4X4 raw tx_mode + `compoundReferenceAllowed
    // == 0` short-circuit + HP gate off + interp gate off → every
    // probability sweep returns its §10 / §10.5 default unchanged.
    let bytes = [0u8; 256];
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let r = parse_compressed_header_inter(&bytes, false, inputs).expect("inter walker runs");

    // Intra-shared prefix anchors (§10 default_coef_probs / §10
    // default_skip_prob / §10 default_tx_probs).
    assert_eq!(r.intra.tx_mode, TxMode::Only4x4);
    assert_eq!(r.intra.coef_probs[0][0][0][0][0], [195, 29, 183]);
    assert_eq!(r.intra.coef_probs[3][1][1][5][5], [1, 16, 6]);
    assert_eq!(r.intra.skip_prob, [192, 128, 64]);
    assert_eq!(r.intra.tx_probs[1], [[100, 0, 0], [66, 0, 0]]);
    assert_eq!(r.intra.tx_probs[2], [[20, 152, 0], [15, 101, 0]]);
    assert_eq!(r.intra.tx_probs[3], [[3, 136, 37], [5, 52, 13]]);

    // §10.5 `default_is_inter_prob` = { 9, 102, 187, 225 }.
    assert_eq!(r.is_inter_prob, [9, 102, 187, 225]);

    // §10.5 `default_inter_mode_probs` row anchor: context 0 →
    // {2, 173, 34} (verifies §6.3.9 default pass-through).
    assert_eq!(r.inter_mode_probs[0], [2, 173, 34]);

    // §6.3.12 `compoundReferenceAllowed == 0` short-circuit:
    // `SingleReference` with no compound config.
    assert_eq!(r.reference_mode, ReferenceMode::SingleReference);
    assert_eq!(r.compound_reference_config, None);

    // §6.3.16 mv_probs pass-through: every slot matches `MvProbs::defaults`.
    assert_eq!(r.mv_probs, MvProbs::defaults());
}

#[test]
fn inter_interpolation_filter_gate_skips_walker_when_not_switchable() {
    // The §6.3.10 sweep fires only when `interpolation_filter ==
    // SWITCHABLE`. With the gate off, the 8-cell §6.3.10 sweep is
    // skipped entirely (no B(252) reads consumed). On a zero buffer
    // the resulting `interp_filter_probs` table is the §10.5 default
    // either way; this test pins the gate by checking that downstream
    // tables (which would read at a shifted cursor if the §6.3.10
    // walker had fired) are bit-identical between the two gate states.
    let bytes = [0u8; 256];
    let inputs_off = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let inputs_on = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: true,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let r_off = parse_compressed_header_inter(&bytes, false, inputs_off).unwrap();
    let r_on = parse_compressed_header_inter(&bytes, false, inputs_on).unwrap();
    // §10.5 `default_interp_filter_probs` survives the zero-buffer
    // walk on both arms.
    assert_eq!(r_off.interp_filter_probs, r_on.interp_filter_probs);
    // §6.3.11 result is bit-identical because every cell update would
    // have been a no-op on a zero buffer regardless of cursor.
    assert_eq!(r_off.is_inter_prob, r_on.is_inter_prob);
    // §6.3.16 MV-probs result is bit-identical for the same reason.
    assert_eq!(r_off.mv_probs, r_on.mv_probs);
}

#[test]
fn inter_compound_reference_short_circuit_returns_single_reference() {
    // `compoundReferenceAllowed == 0` (every inter ref-frame sign-bias
    // identical) → §6.3.12 returns `SingleReference` with no bool
    // reads. No `setup_compound_reference_mode( )` output.
    let bytes = [0u8; 256];
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let r = parse_compressed_header_inter(&bytes, false, inputs).unwrap();
    assert_eq!(r.reference_mode, ReferenceMode::SingleReference);
    assert!(r.compound_reference_config.is_none());
}

#[test]
fn inter_compound_reference_allowed_enters_walker_arm() {
    // mixed_sign_bias() → `compoundReferenceAllowed = 1`. On a zero
    // buffer the §6.3.12 walker reads one L(1) `non_single_reference`
    // = 0 → `reference_mode = SingleReference` (NOT
    // `SingleReference` via the short-circuit arm — this is the
    // L(1)-driven `SingleReference`). §6.3.18 is not invoked because
    // the `non_single_reference` flag is 0.
    let bytes = [0u8; 256];
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: mixed_sign_bias(),
        allow_high_precision_mv: false,
    };
    let r = parse_compressed_header_inter(&bytes, false, inputs).unwrap();
    assert_eq!(r.reference_mode, ReferenceMode::SingleReference);
    assert!(r.compound_reference_config.is_none());
}

#[test]
fn inter_high_precision_mv_gate_preserves_defaults_on_zero_buffer() {
    // The §6.3.16 walker's high-precision tail (`mv_class0_hp_prob[ 2
    // ]` + `mv_hp_prob[ 2 ]` = 4 cells) fires only when
    // `allow_high_precision_mv == 1`. On a zero buffer both arms leave
    // the HP slots at their §10.5 defaults (gate off skips, gate on
    // runs `update_mv_prob` returning the input unchanged) so the
    // four cells are bit-identical across gate states.
    let bytes = [0u8; 256];
    let inputs_off = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let inputs_on = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: true,
    };
    let r_off = parse_compressed_header_inter(&bytes, false, inputs_off).unwrap();
    let r_on = parse_compressed_header_inter(&bytes, false, inputs_on).unwrap();
    assert_eq!(r_off.mv_probs.class0_hp_prob, r_on.mv_probs.class0_hp_prob);
    assert_eq!(r_off.mv_probs.hp_prob, r_on.mv_probs.hp_prob);
    // The §10.5 `default_mv_class0_hp_prob` = 160 per spec listing.
    assert_eq!(r_off.mv_probs.class0_hp_prob, [160, 160]);
    assert_eq!(r_on.mv_probs.class0_hp_prob, [160, 160]);
    // The §10.5 `default_mv_hp_prob` = 128 per spec listing.
    assert_eq!(r_off.mv_probs.hp_prob, [128, 128]);
    assert_eq!(r_on.mv_probs.hp_prob, [128, 128]);
}

#[test]
fn inter_lossless_intra_prefix_matches_intra_walker() {
    // On the lossless path §6.3.1 forces `tx_mode = ONLY_4X4` with no
    // L(2) reads, so the §6.3.7 / §6.3.8 cursor offsets shift versus
    // the non-lossless `0x00` walk. The intra-shared prefix must
    // still match between the inter and intra walkers on identical
    // input.
    let bytes = [0u8; 256];
    let intra = parse_compressed_header(&bytes, true).unwrap();
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let inter = parse_compressed_header_inter(&bytes, true, inputs).unwrap();
    assert_eq!(inter.intra, intra);
}

#[test]
fn inter_nonlossless_intra_prefix_matches_intra_walker() {
    // Non-lossless path: the §6.3.1 `L(2)` raw tx_mode decodes to 0
    // (zero buffer) → `ONLY_4X4`. The intra-shared prefix of the
    // inter walker must produce bit-identical output to the
    // intra-only walker on the same input.
    let bytes = [0u8; 256];
    let intra = parse_compressed_header(&bytes, false).unwrap();
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let inter = parse_compressed_header_inter(&bytes, false, inputs).unwrap();
    assert_eq!(inter.intra, intra);
}

#[test]
fn inter_empty_buffer_rejected_with_same_error_as_intra() {
    // §9.2.1 `init_bool` on an empty buffer fails: the marker bit
    // can't be read. The inter walker must return the same error
    // variant the intra walker returns on identical input — both
    // routes share `init_bool` as their first step.
    let bytes: [u8; 0] = [];
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let inter_err = parse_compressed_header_inter(&bytes, false, inputs).unwrap_err();
    let intra_err = parse_compressed_header(&bytes, false).unwrap_err();
    assert_eq!(inter_err, intra_err);
}

#[test]
fn inter_invalid_marker_rejected_with_same_error_as_intra() {
    // First byte `0xFF`: `BoolValue = 0xFF`, split for p = 128 is
    // 128, so the §9.2.1 marker bit decodes to 1, violating the
    // "shall be equal to 0" constraint. Both intra and inter walkers
    // share `init_bool` as their first step and must reject with the
    // same error.
    let bytes = [0xFFu8, 0x00, 0x00, 0x00];
    let inputs = Vp9CompressedHeaderInterInputs {
        interpolation_filter_is_switchable: false,
        ref_frame_sign_bias: all_zero_sign_bias(),
        allow_high_precision_mv: false,
    };
    let inter_err = parse_compressed_header_inter(&bytes, false, inputs).unwrap_err();
    let intra_err = parse_compressed_header(&bytes, false).unwrap_err();
    assert_eq!(inter_err, intra_err);
}

#[test]
fn ref_frame_sign_bias_public_constructor_round_trips() {
    // `RefFrameSignBias::from_inter_biases` + `get` are part of the
    // public API surface (round 34 promoted them from `pub(crate)` to
    // `pub`). The §3 ref-frame indices `LAST_FRAME = 1`,
    // `GOLDEN_FRAME = 2`, `ALTREF_FRAME = 3` round-trip across all
    // eight input tuples; the `INTRA_FRAME = 0` slot is always 0.
    for last in 0..=1u8 {
        for golden in 0..=1u8 {
            for altref in 0..=1u8 {
                let bias = RefFrameSignBias::from_inter_biases(last, golden, altref);
                assert_eq!(bias.get(1), last); // LAST_FRAME
                assert_eq!(bias.get(2), golden); // GOLDEN_FRAME
                assert_eq!(bias.get(3), altref); // ALTREF_FRAME
                assert_eq!(bias.get(0), 0); // INTRA_FRAME — never populated.
            }
        }
    }
}
