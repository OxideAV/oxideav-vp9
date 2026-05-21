//! Synthetic-buffer tests for the VP9 uncompressed-header walker.
//! Each test builds a byte stream MSB-first per spec §9.1 and checks
//! the walker's struct against the expected field values.
//!
//! No external fixtures are involved — every input is constructed bit
//! by bit from the §6.2 syntax tree.

use oxideav_vp9::{parse_uncompressed_header, ColorSpace, Error, FrameType};

/// Minimal MSB-first bit builder for assembling test buffers. Pushes
/// the lowest `n` bits of `value` MSB-first into an internal byte
/// vector.
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
        assert!(n <= 32);
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

    /// Push an `s(n)` value per spec §4.9.2: magnitude bits then sign.
    fn push_signed(&mut self, value: i32, n: u32) {
        let magnitude = value.unsigned_abs();
        let sign: u32 = if value < 0 { 1 } else { 0 };
        self.push_bits(magnitude, n);
        self.push_bits(sign, 1);
    }

    /// Pad with zero bits up to the next byte boundary (§6.1.1).
    fn align_to_byte(&mut self) {
        while self.bit_pos & 7 != 0 {
            self.push_bits(0, 1);
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

fn push_frame_sync(b: &mut BitBuilder) {
    // §7.2.1: 0x49, 0x83, 0x42.
    b.push_bits(0x49, 8);
    b.push_bits(0x83, 8);
    b.push_bits(0x42, 8);
}

/// Push the minimal "everything disabled" tail of `uncompressed_header()`
/// for a non-error-resilient frame whose intra/error-resilient state
/// already forced `frame_context_idx` to 0:
///
/// `refresh_frame_context = 0`, `frame_parallel_decoding_mode = 0`,
/// `frame_context_idx = 0`, then minimum-bit loop_filter /
/// quantization / segmentation / tile_info, then `header_size_in_bytes`,
/// then `trailing_bits()` zero pad.
///
/// `frame_width` controls the tile_info loop bounds via §6.2.6.
fn push_minimal_tail_error_resilient(b: &mut BitBuilder, frame_width: u32) {
    // refresh_frame_context = 0 / frame_parallel_decoding_mode = 1 are
    // FORCED by error_resilient_mode == 1 (no bits read).
    // frame_context_idx still gets read (then reset to 0 by the spec).
    b.push_bits(0, 2);
    push_disabled_lfqs_tile(b, frame_width);
}

/// Same as above but for the error_resilient_mode == 0 branch — the
/// two bits for refresh_frame_context + frame_parallel_decoding_mode
/// ARE read.
fn push_minimal_tail_normal(b: &mut BitBuilder, frame_width: u32) {
    b.push_bits(0, 1); // refresh_frame_context
    b.push_bits(0, 1); // frame_parallel_decoding_mode
    b.push_bits(0, 2); // frame_context_idx
    push_disabled_lfqs_tile(b, frame_width);
}

/// loop_filter (all zero + delta_enabled = 0) + quantization (base = 0,
/// all delta_coded = 0) + segmentation (off) + tile_info (default
/// log2's, both rows 0) + header_size_in_bytes = 0 + trailing-bits
/// zero pad. The `frame_width` argument advances the tile_info loop
/// to its expected min/max so the loop emits the right number of
/// `increment_tile_cols_log2` bits (we pick the minimum).
fn push_disabled_lfqs_tile(b: &mut BitBuilder, frame_width: u32) {
    // loop_filter_params: level=0 / sharpness=0 / delta_enabled=0.
    b.push_bits(0, 6); // level
    b.push_bits(0, 3); // sharpness
    b.push_bits(0, 1); // delta_enabled

    // quantization_params: base_q_idx=0, three read_delta_q with
    // delta_coded=0 each.
    b.push_bits(0, 8); // base_q_idx
    b.push_bits(0, 1); // delta_coded for y_dc
    b.push_bits(0, 1); // delta_coded for uv_dc
    b.push_bits(0, 1); // delta_coded for uv_ac

    // segmentation_params: segmentation_enabled=0 -> single bit.
    b.push_bits(0, 1);

    // tile_info: emit one "break" (increment=0) for every level from
    // min_log2 up to max_log2 - exclusive (the loop pre-checks
    // `< max_log2`). Then 1 bit for tile_rows_log2.
    let mi_cols = (frame_width + 7) >> 3;
    let sb64_cols = (mi_cols + 7) >> 3;
    let min_log2 = calc_min_log2_tile_cols(sb64_cols);
    let max_log2 = calc_max_log2_tile_cols(sb64_cols);
    // We pick the minimum tile_cols_log2: emit a single
    // increment_tile_cols_log2 = 0 to break the loop (only required
    // when min_log2 < max_log2).
    if min_log2 < max_log2 {
        b.push_bits(0, 1);
    }
    b.push_bits(0, 1); // tile_rows_log2 first bit = 0

    // header_size_in_bytes = 0.
    b.push_bits(0, 16);

    // trailing_bits zero pad.
    b.align_to_byte();
}

fn calc_min_log2_tile_cols(sb64_cols: u32) -> u8 {
    let mut min_log2 = 0u8;
    while (64u32 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    min_log2
}

fn calc_max_log2_tile_cols(sb64_cols: u32) -> u8 {
    let mut max_log2 = 1u8;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 - 1
}

#[test]
fn profile0_keyframe_yuv420_studio_swing() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(0, 1); // error_resilient_mode = 0
    push_frame_sync(&mut b);
    // color_config (profile == 0): color_space, color_range
    b.push_bits(2, 3); // color_space = CS_BT_709
    b.push_bits(0, 1); // color_range = studio swing

    // frame_size: 1280x720 -> minus_1 = 1279/719.
    b.push_bits(1279, 16);
    b.push_bits(719, 16);
    // render_size: render_and_frame_size_different = 0
    b.push_bits(0, 1);

    // Tail (error_resilient_mode == 0 branch).
    push_minimal_tail_normal(&mut b, 1280);

    let h = parse_uncompressed_header(&b.finish()).expect("walker should accept this buffer");
    assert_eq!(h.profile, 0);
    assert!(!h.show_existing_frame);
    assert_eq!(h.frame_type, FrameType::KeyFrame);
    assert!(h.show_frame);
    assert!(!h.error_resilient_mode);
    assert!(!h.intra_only);
    assert_eq!(h.color_config.bit_depth, 8);
    assert_eq!(h.color_config.color_space, ColorSpace::Bt709);
    assert!(!h.color_config.color_range_full);
    assert!(h.color_config.subsampling_x);
    assert!(h.color_config.subsampling_y);
    assert_eq!(h.frame_width, 1280);
    assert_eq!(h.frame_height, 720);
    assert_eq!(h.render_width, 1280);
    assert_eq!(h.render_height, 720);
    // Round-2 fields.
    assert_eq!(h.refresh_frame_flags, 0xFF); // key frame
    assert_eq!(h.reset_frame_context, 0);
    assert!(!h.refresh_frame_context); // we wrote 0
    assert!(!h.frame_parallel_decoding_mode); // we wrote 0
                                              // FrameIsIntra=1 forces frame_context_idx to 0.
    assert_eq!(h.frame_context_idx, 0);
    assert_eq!(h.loop_filter.level, 0);
    assert!(!h.loop_filter.delta_enabled);
    assert_eq!(h.quantization.base_q_idx, 0);
    assert!(h.quantization.lossless);
    assert!(!h.segmentation.enabled);
    assert_eq!(h.header_size_in_bytes, 0);
    // The header is byte-aligned by trailing_bits and the size is
    // reported.
    assert!(h.uncompressed_header_size_bytes > 0);
}

#[test]
fn profile2_keyframe_10bit_with_render_override_and_quant_deltas() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(1, 1); // profile_high_bit -> Profile 2
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(1, 1); // error_resilient_mode = 1
    push_frame_sync(&mut b);
    // color_config (profile >= 2): ten_or_twelve_bit, color_space,
    // color_range. subsampling defaults to 4:2:0 (profile 2).
    b.push_bits(0, 1); // ten_or_twelve_bit = 0 -> 10-bit
    b.push_bits(5, 3); // color_space = CS_BT_2020
    b.push_bits(1, 1); // color_range = full swing

    // frame_size: 3840x2160 -> minus_1 = 3839/2159.
    b.push_bits(3839, 16);
    b.push_bits(2159, 16);
    // render_size override to 1920x1080.
    b.push_bits(1, 1);
    b.push_bits(1919, 16);
    b.push_bits(1079, 16);

    // Tail — error-resilient branch (no refresh_frame_context /
    // frame_parallel_decoding_mode bits).
    // frame_context_idx (then reset to 0).
    b.push_bits(0, 2);
    // loop_filter with delta_enabled but no update.
    b.push_bits(20, 6); // level
    b.push_bits(2, 3); // sharpness
    b.push_bits(1, 1); // delta_enabled
    b.push_bits(0, 1); // delta_update = 0
                       // quantization with nonzero base_q_idx and a Y DC delta.
    b.push_bits(64, 8); // base_q_idx
    b.push_bits(1, 1); // delta_coded for y_dc
    b.push_signed(-3, 4); // delta_q_y_dc = -3
    b.push_bits(0, 1); // delta_coded for uv_dc = 0
    b.push_bits(0, 1); // delta_coded for uv_ac = 0
                       // segmentation_enabled = 0.
    b.push_bits(0, 1);
    // tile_info for 3840-wide frame: max_log2 = 3, min_log2 = 0.
    // Emit "increment, increment, increment" then bit for rows.
    // We'll increment up to max_log2 by sending three 1-bits then a
    // 0-bit IS NOT actually required because the loop terminates at
    // tile_cols_log2 == max_log2; just keep min == 0.
    // To keep this terse choose min_log2 (write three 0-bits to break
    // immediately at the first iteration).
    b.push_bits(0, 1); // increment = 0 -> break at tile_cols_log2 = 0
                       // tile_rows_log2: first bit = 1, then increment bit.
    b.push_bits(1, 1);
    b.push_bits(1, 1); // -> tile_rows_log2 = 2
                       // header_size_in_bytes = 17 (arbitrary nonzero).
    b.push_bits(17, 16);
    b.align_to_byte();

    let h = parse_uncompressed_header(&b.finish()).expect("profile 2 keyframe");
    assert_eq!(h.profile, 2);
    assert!(h.error_resilient_mode);
    assert_eq!(h.color_config.bit_depth, 10);
    assert_eq!(h.color_config.color_space, ColorSpace::Bt2020);
    assert!(h.color_config.color_range_full);
    assert_eq!(h.frame_width, 3840);
    assert_eq!(h.frame_height, 2160);
    assert_eq!(h.render_width, 1920);
    assert_eq!(h.render_height, 1080);

    assert_eq!(h.loop_filter.level, 20);
    assert_eq!(h.loop_filter.sharpness, 2);
    assert!(h.loop_filter.delta_enabled);
    assert!(!h.loop_filter.delta_update);

    assert_eq!(h.quantization.base_q_idx, 64);
    assert_eq!(h.quantization.delta_q_y_dc, -3);
    assert_eq!(h.quantization.delta_q_uv_dc, 0);
    assert_eq!(h.quantization.delta_q_uv_ac, 0);
    assert!(!h.quantization.lossless);

    assert_eq!(h.tile_info.tile_cols_log2, 0);
    assert_eq!(h.tile_info.tile_rows_log2, 2);

    assert_eq!(h.header_size_in_bytes, 17);
    assert!(!h.refresh_frame_context);
    assert!(h.frame_parallel_decoding_mode);
    assert_eq!(h.frame_context_idx, 0);
}

#[test]
fn profile3_keyframe_rgb_full_swing_444() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(1, 1); // profile_low_bit
    b.push_bits(1, 1); // profile_high_bit -> Profile 3
    b.push_bits(0, 1); // reserved_zero (profile 3)
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(0, 1); // error_resilient_mode
    push_frame_sync(&mut b);
    // color_config: profile 3
    b.push_bits(1, 1); // ten_or_twelve_bit = 1 -> 12-bit
    b.push_bits(7, 3); // color_space = CS_RGB

    // CS_RGB on profile 3: only reserved_zero follows.
    b.push_bits(0, 1); // reserved_zero
    b.push_bits(63, 16); // frame_width_minus_1
    b.push_bits(63, 16); // frame_height_minus_1
    b.push_bits(0, 1); // render_and_frame_size_different = 0

    push_minimal_tail_normal(&mut b, 64);

    let h = parse_uncompressed_header(&b.finish()).expect("profile 3 RGB keyframe");
    assert_eq!(h.profile, 3);
    assert_eq!(h.color_config.bit_depth, 12);
    assert_eq!(h.color_config.color_space, ColorSpace::Rgb);
    assert!(h.color_config.color_range_full);
    assert!(!h.color_config.subsampling_x);
    assert!(!h.color_config.subsampling_y);
    assert_eq!(h.frame_width, 64);
    assert_eq!(h.frame_height, 64);
    assert_eq!(h.refresh_frame_flags, 0xFF);
}

#[test]
fn show_existing_frame_returns_early() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(1, 1); // show_existing_frame
    b.push_bits(5, 3); // frame_to_show_map_idx = 5

    let h = parse_uncompressed_header(&b.finish()).expect("show_existing_frame path");
    assert_eq!(h.profile, 0);
    assert!(h.show_existing_frame);
    assert_eq!(h.frame_to_show_map_idx, Some(5));
    // Spec §6.2 forces header_size_in_bytes = 0 and
    // refresh_frame_flags = 0 on this path.
    assert_eq!(h.header_size_in_bytes, 0);
    assert_eq!(h.refresh_frame_flags, 0);
}

#[test]
fn intra_only_inter_frame_profile0_uses_default_color_config() {
    // frame_type = NON_KEY_FRAME, show_frame = 0, intra_only = 1.
    // error_resilient_mode = 1 so reset_frame_context is skipped.
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(1, 1); // frame_type = NON_KEY_FRAME
    b.push_bits(0, 1); // show_frame = 0 -> intra_only flag follows
    b.push_bits(1, 1); // error_resilient_mode = 1
    b.push_bits(1, 1); // intra_only = 1

    // Profile 0 intra-only skips color_config(); only sync +
    // refresh_flags + frame_size + render_size remain.
    push_frame_sync(&mut b);
    b.push_bits(0x77, 8); // refresh_frame_flags
    b.push_bits(319, 16); // frame_width_minus_1 -> 320
    b.push_bits(239, 16); // frame_height_minus_1 -> 240
    b.push_bits(0, 1); // render_and_frame_size_different

    push_minimal_tail_error_resilient(&mut b, 320);

    let h =
        parse_uncompressed_header(&b.finish()).expect("intra-only profile 0 inter-frame header");
    assert_eq!(h.profile, 0);
    assert_eq!(h.frame_type, FrameType::NonKeyFrame);
    assert!(h.intra_only);
    assert!(!h.show_frame);
    assert!(h.error_resilient_mode);
    assert_eq!(h.color_config.bit_depth, 8);
    assert_eq!(h.color_config.color_space, ColorSpace::Bt601);
    assert_eq!(h.frame_width, 320);
    assert_eq!(h.frame_height, 240);
    assert_eq!(h.refresh_frame_flags, 0x77);
    // Error resilient + intra-only -> frame_context_idx reset to 0,
    // refresh_frame_context = 0, frame_parallel_decoding_mode = 1.
    assert_eq!(h.frame_context_idx, 0);
    assert!(!h.refresh_frame_context);
    assert!(h.frame_parallel_decoding_mode);
}

#[test]
fn loop_filter_with_full_delta_update() {
    // Build a key-frame whose tail exercises every loop_filter_params
    // sub-branch: delta_enabled=1, delta_update=1, with mixed
    // update_ref_delta and update_mode_delta flags.
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(1, 1); // error_resilient_mode = 1 -> skip rfc/fpdm
    push_frame_sync(&mut b);
    b.push_bits(1, 3); // color_space = CS_BT_601
    b.push_bits(0, 1); // color_range = studio
    b.push_bits(255, 16); // 256 width
    b.push_bits(255, 16); // 256 height
    b.push_bits(0, 1); // render_and_frame_size_different
                       // tail (error-resilient): frame_context_idx = 0
    b.push_bits(0, 2);
    // loop_filter_params:
    b.push_bits(35, 6); // level
    b.push_bits(5, 3); // sharpness
    b.push_bits(1, 1); // delta_enabled
    b.push_bits(1, 1); // delta_update
                       // ref_deltas: update [Y, N, Y, N], values [+10, _, -7, _].
    b.push_bits(1, 1); // update_ref_delta[0]
    b.push_signed(10, 6);
    b.push_bits(0, 1); // update_ref_delta[1]
    b.push_bits(1, 1); // update_ref_delta[2]
    b.push_signed(-7, 6);
    b.push_bits(0, 1); // update_ref_delta[3]
                       // mode_deltas: update [N, Y], value -1.
    b.push_bits(0, 1); // update_mode_delta[0]
    b.push_bits(1, 1); // update_mode_delta[1]
    b.push_signed(-1, 6);
    // quantization: base 90, no deltas.
    b.push_bits(90, 8);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    // segmentation off.
    b.push_bits(0, 1);
    // tile_info: 256-wide -> Sb64Cols = 4, min_log2 = 0, max_log2 = 0
    // (4 >> 1 = 2 < 4 so loop runs once, returning maxLog2 - 1 = 0).
    b.push_bits(0, 1); // tile_rows_log2 first bit
                       // header_size = 0.
    b.push_bits(0, 16);
    b.align_to_byte();

    let h = parse_uncompressed_header(&b.finish()).expect("loop filter delta update");
    assert!(h.loop_filter.delta_enabled);
    assert!(h.loop_filter.delta_update);
    assert_eq!(h.loop_filter.ref_deltas[0], Some(10));
    assert_eq!(h.loop_filter.ref_deltas[1], None);
    assert_eq!(h.loop_filter.ref_deltas[2], Some(-7));
    assert_eq!(h.loop_filter.ref_deltas[3], None);
    assert_eq!(h.loop_filter.mode_deltas[0], None);
    assert_eq!(h.loop_filter.mode_deltas[1], Some(-1));
    assert_eq!(h.loop_filter.level, 35);
    assert_eq!(h.loop_filter.sharpness, 5);
    assert_eq!(h.quantization.base_q_idx, 90);
    assert!(!h.quantization.lossless);
    assert_eq!(h.tile_info.tile_cols_log2, 0);
    assert_eq!(h.tile_info.tile_rows_log2, 0);
}

#[test]
fn segmentation_enabled_with_update_map_and_data() {
    // Key-frame with segmentation enabled, map update with temporal
    // update, and data update for a couple of features.
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(1, 1); // error_resilient_mode = 1
    push_frame_sync(&mut b);
    b.push_bits(1, 3); // color_space = CS_BT_601
    b.push_bits(0, 1); // color_range
    b.push_bits(255, 16); // 256 width
    b.push_bits(255, 16); // 256 height
    b.push_bits(0, 1); // render_and_frame_size_different
    b.push_bits(0, 2); // frame_context_idx
                       // loop_filter (disabled).
    b.push_bits(0, 6);
    b.push_bits(0, 3);
    b.push_bits(0, 1);
    // quantization (base 0, all delta_coded = 0) -> lossless.
    b.push_bits(0, 8);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);

    // segmentation_enabled = 1.
    b.push_bits(1, 1);
    // update_map = 1.
    b.push_bits(1, 1);
    // 7 tree probs. Mix prob_coded values:
    // Slot 0: prob_coded=1, prob=128.
    b.push_bits(1, 1);
    b.push_bits(128, 8);
    // Slot 1..6: prob_coded=0 -> implicit 255.
    for _ in 1..7 {
        b.push_bits(0, 1);
    }
    // temporal_update = 1.
    b.push_bits(1, 1);
    // 3 pred probs, each prob_coded=1 with values 10, 20, 30.
    b.push_bits(1, 1);
    b.push_bits(10, 8);
    b.push_bits(1, 1);
    b.push_bits(20, 8);
    b.push_bits(1, 1);
    b.push_bits(30, 8);

    // update_data = 1.
    b.push_bits(1, 1);
    // abs_or_delta_update = 1.
    b.push_bits(1, 1);
    // For each of MAX_SEGMENTS=8 segments, SEG_LVL_MAX=4 features.
    // We enable: segment 0 / feature 0 (8-bit signed) = +50,
    //           segment 1 / feature 1 (6-bit signed) = -5,
    //           segment 2 / feature 2 (2-bit unsigned) = 3,
    //           segment 3 / feature 3 (flag-only)     = "on".
    // Everything else: feature_enabled = 0.
    for seg in 0..8 {
        for feat in 0..4 {
            let enable_combo = (seg == 0 && feat == 0)
                || (seg == 1 && feat == 1)
                || (seg == 2 && feat == 2)
                || (seg == 3 && feat == 3);
            if enable_combo {
                b.push_bits(1, 1); // feature_enabled
                if feat == 0 {
                    b.push_bits(50, 8);
                    b.push_bits(0, 1); // sign +
                } else if feat == 1 {
                    b.push_bits(5, 6);
                    b.push_bits(1, 1); // sign -
                } else if feat == 2 {
                    b.push_bits(3, 2);
                    // unsigned, no sign bit.
                }
                // feat==3: 0 bits, unsigned, no sign — just the
                // feature_enabled flag we already pushed.
            } else {
                b.push_bits(0, 1); // feature_enabled = 0
            }
        }
    }

    // tile_info: 256-wide.
    b.push_bits(0, 1); // tile_rows_log2 first bit
                       // header_size = 5.
    b.push_bits(5, 16);
    b.align_to_byte();

    let h = parse_uncompressed_header(&b.finish()).expect("segmentation enabled");
    assert!(h.segmentation.enabled);
    assert!(h.segmentation.update_map);
    let tree = h.segmentation.tree_probs.expect("tree_probs present");
    assert_eq!(tree[0], 128);
    for &p in &tree[1..7] {
        assert_eq!(p, 255);
    }
    assert!(h.segmentation.temporal_update);
    let pred = h.segmentation.pred_prob.expect("pred_prob present");
    assert_eq!(pred, [10, 20, 30]);
    assert!(h.segmentation.update_data);
    assert!(h.segmentation.abs_or_delta_update);
    assert!(h.segmentation.feature_enabled[0][0]);
    assert_eq!(h.segmentation.feature_data[0][0], 50);
    assert!(h.segmentation.feature_enabled[1][1]);
    assert_eq!(h.segmentation.feature_data[1][1], -5);
    assert!(h.segmentation.feature_enabled[2][2]);
    assert_eq!(h.segmentation.feature_data[2][2], 3);
    assert!(h.segmentation.feature_enabled[3][3]);
    assert_eq!(h.segmentation.feature_data[3][3], 0); // skip flag, value irrelevant
                                                      // Everything else stays disabled.
    assert!(!h.segmentation.feature_enabled[4][0]);
    assert!(!h.segmentation.feature_enabled[0][1]);

    assert_eq!(h.header_size_in_bytes, 5);
}

#[test]
fn tile_info_picks_up_increments_for_4k_frame() {
    // 4K key-frame with min_log2 = 0 and max_log2 = 3. Push two
    // increment=1 then increment=0 to land tile_cols_log2 = 2.
    let mut b = BitBuilder::new();
    b.push_bits(2, 2);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(1, 1);
    b.push_bits(1, 1); // error_resilient_mode = 1
    push_frame_sync(&mut b);
    b.push_bits(1, 3); // color_space = CS_BT_601
    b.push_bits(0, 1); // color_range
    b.push_bits(3839, 16); // 3840 width
    b.push_bits(2159, 16); // 2160 height
    b.push_bits(0, 1); // render_and_frame_size_different
                       // tail
    b.push_bits(0, 2); // frame_context_idx
                       // loop_filter disabled
    b.push_bits(0, 6);
    b.push_bits(0, 3);
    b.push_bits(0, 1);
    // quant lossless
    b.push_bits(0, 8);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    // segmentation off
    b.push_bits(0, 1);
    // tile_info: emit increment=1, increment=1, increment=0 to land
    // at tile_cols_log2 = 2.
    b.push_bits(1, 1);
    b.push_bits(1, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1); // tile_rows_log2 first bit = 0
                       // header_size = 0
    b.push_bits(0, 16);
    b.align_to_byte();

    let h = parse_uncompressed_header(&b.finish()).expect("tile_info increment walk");
    assert_eq!(h.tile_info.tile_cols_log2, 2);
    assert_eq!(h.tile_info.tile_rows_log2, 0);
}

#[test]
fn nonzero_trailing_bit_is_rejected() {
    // Build a valid key frame whose trailing-bit pad slot we then
    // flip from 0 to 1 to confirm the §7.1.1 check fires.
    let mut b = BitBuilder::new();
    b.push_bits(2, 2);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(1, 1);
    b.push_bits(1, 1); // error_resilient_mode = 1
    push_frame_sync(&mut b);
    b.push_bits(1, 3);
    b.push_bits(0, 1);
    b.push_bits(255, 16); // 256
    b.push_bits(255, 16); // 256
    b.push_bits(0, 1);
    b.push_bits(0, 2);
    b.push_bits(0, 6);
    b.push_bits(0, 3);
    b.push_bits(0, 1);
    b.push_bits(0, 8);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1);
    b.push_bits(0, 1); // tile rows first bit
    b.push_bits(0, 16); // header_size
                        // Don't align — set a stray 1 in the pad region.
                        // Currently bit_pos isn't byte aligned. Push a 1 at the next
                        // position to violate trailing_bits.
    let pos = b.bit_pos;
    if pos & 7 != 0 {
        b.push_bits(1, 1);
        b.align_to_byte();
    } else {
        // If already aligned, append an extra byte with a high bit
        // and shift bit_pos manually — but parse won't read into
        // them. Skip the case.
        b.push_bits(0, 1);
        b.align_to_byte();
    }

    let err = parse_uncompressed_header(&b.finish()).unwrap_err();
    assert_eq!(err, Error::InvalidBitstream);
}

#[test]
fn bad_frame_marker_is_rejected() {
    let mut b = BitBuilder::new();
    b.push_bits(1, 2); // frame_marker = 1 (must be 2)
    b.push_bits(0, 6); // padding
    assert_eq!(
        parse_uncompressed_header(&b.finish()).unwrap_err(),
        Error::InvalidBitstream
    );
}

#[test]
fn bad_frame_sync_code_is_rejected() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(0, 1); // error_resilient_mode

    // Wrong sync bytes.
    b.push_bits(0x49, 8);
    b.push_bits(0x83, 8);
    b.push_bits(0x43, 8); // should be 0x42
    b.push_bits(0, 24); // pad rest

    assert_eq!(
        parse_uncompressed_header(&b.finish()).unwrap_err(),
        Error::InvalidBitstream
    );
}

#[test]
fn truncated_buffer_returns_unexpected_eof() {
    // Stop mid-header: just frame_marker + half the profile bits.
    let mut b = BitBuilder::new();
    b.push_bits(2, 2);
    b.push_bits(0, 1);
    // Don't even push the high bit — finish with a 3-bit-only byte.
    let mut data = b.finish();
    // Trim away the trailing zero-padding so the reader runs out
    // before reaching show_existing_frame.
    data.truncate(0);
    assert_eq!(
        parse_uncompressed_header(&data).unwrap_err(),
        Error::UnexpectedEof
    );
}
