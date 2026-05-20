//! Synthetic-buffer tests for the round-1 VP9 uncompressed-header
//! walker. Each test builds a byte stream MSB-first per spec §9.1 and
//! checks the walker's struct against the expected field values.
//!
//! No external fixtures are involved — every input is constructed bit
//! by bit from the §6.2 syntax.

use oxideav_vp9::{parse_uncompressed_header, ColorSpace, Error, FrameType};

/// Minimal MSB-first bit builder for assembling test buffers. Pushes
/// the lowest `n` bits of `value` MSB-first into an internal byte
/// vector. The padding bits at the end of the final partial byte are
/// always zero, which matches what a real VP9 encoder produces for the
/// `trailing_bits()` zeros (spec §6.1.1) and never breaks the walker.
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

#[test]
fn profile0_keyframe_yuv420_studio_swing() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(0, 1); // profile_high_bit -> Profile 0
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(0, 1); // error_resilient_mode
    push_frame_sync(&mut b);
    // color_config (profile == 0): color_space, color_range
    b.push_bits(2, 3); // color_space = CS_BT_709
    b.push_bits(0, 1); // color_range = studio swing

    // frame_size: 1280x720 -> minus_1 = 1279/719.
    b.push_bits(1279, 16);
    b.push_bits(719, 16);
    // render_size: render_and_frame_size_different = 0
    b.push_bits(0, 1);

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
}

#[test]
fn profile2_keyframe_10bit_with_render_override() {
    let mut b = BitBuilder::new();
    b.push_bits(2, 2); // frame_marker
    b.push_bits(0, 1); // profile_low_bit
    b.push_bits(1, 1); // profile_high_bit -> Profile 2
    b.push_bits(0, 1); // show_existing_frame
    b.push_bits(0, 1); // frame_type = KEY_FRAME
    b.push_bits(1, 1); // show_frame
    b.push_bits(1, 1); // error_resilient_mode
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

    let h = parse_uncompressed_header(&b.finish()).expect("profile 2 keyframe");
    assert_eq!(h.profile, 2);
    assert!(h.error_resilient_mode);
    assert_eq!(h.color_config.bit_depth, 10);
    assert_eq!(h.color_config.color_space, ColorSpace::Bt2020);
    assert!(h.color_config.color_range_full);
    assert!(h.color_config.subsampling_x);
    assert!(h.color_config.subsampling_y);
    assert_eq!(h.frame_width, 3840);
    assert_eq!(h.frame_height, 2160);
    assert_eq!(h.render_width, 1920);
    assert_eq!(h.render_height, 1080);
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

    let h = parse_uncompressed_header(&b.finish()).expect("profile 3 RGB keyframe");
    assert_eq!(h.profile, 3);
    assert_eq!(h.color_config.bit_depth, 12);
    assert_eq!(h.color_config.color_space, ColorSpace::Rgb);
    assert!(h.color_config.color_range_full);
    assert!(!h.color_config.subsampling_x);
    assert!(!h.color_config.subsampling_y);
    assert_eq!(h.frame_width, 64);
    assert_eq!(h.frame_height, 64);
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
    b.push_bits(0xFF, 8); // refresh_frame_flags
    b.push_bits(319, 16); // frame_width_minus_1 -> 320
    b.push_bits(239, 16); // frame_height_minus_1 -> 240
    b.push_bits(0, 1); // render_and_frame_size_different

    let h =
        parse_uncompressed_header(&b.finish()).expect("intra-only profile 0 inter-frame header");
    assert_eq!(h.profile, 0);
    assert_eq!(h.frame_type, FrameType::NonKeyFrame);
    assert!(h.intra_only);
    assert!(!h.show_frame);
    assert!(h.error_resilient_mode);
    // Spec §6.2 default for intra_only / Profile 0: BT.601, 4:2:0, 8-bit.
    assert_eq!(h.color_config.bit_depth, 8);
    assert_eq!(h.color_config.color_space, ColorSpace::Bt601);
    assert_eq!(h.frame_width, 320);
    assert_eq!(h.frame_height, 240);
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
