//! Integration tests for the VP9 keyframe encoder entry `encode_vp9`.
//!
//! `encode_vp9` is a **lossless** encoder: the emitted keyframe decodes
//! byte-exact back to the input planar frame. These tests exercise the
//! public encode → decode round-trip across a sweep of frame geometries
//! (including 1x1, partial-superblock, multi-superblock and non-square
//! shapes) and content regimes (flat, noise, gradient), asserting the
//! bit-for-bit `decode_vp9( encode_vp9( pixels ) ) == pixels` contract.

use oxideav_vp9::{decode_intra_frame, decode_vp9, encode_vp9};

/// Build a flat 8-bit 4:2:0 planar pixel buffer for `width × height`.
fn flat_pixels(width: u32, height: u32, fill: u8) -> Vec<u8> {
    vec![fill; planar_len(width, height)]
}

fn planar_len(width: u32, height: u32) -> usize {
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    (width as usize) * (height as usize) + 2 * cw * ch
}

/// Deterministic pseudo-random planar 4:2:0 frame.
fn noise_pixels(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    (0..planar_len(width, height)).map(|_| next()).collect()
}

/// `encode_vp9` round-trips **byte-exact** through the decoder for a
/// representative geometry sweep, on pseudo-random (worst-case) content.
#[test]
fn encode_decode_geometry_sweep_is_byte_exact() {
    let sizes = [
        (1u32, 1u32),
        (8, 8),
        (16, 16),
        (64, 64),
        (65, 65),   // 1 px past a superblock — partial SB on both axes.
        (128, 64),  // two SB wide.
        (64, 128),  // two SB tall.
        (40, 24),   // non-multiple-of-8 both axes.
        (256, 144), // multi-SB, non-square.
        (1, 64),    // degenerate thin column.
        (64, 1),    // degenerate thin row.
    ];
    for &(w, h) in &sizes {
        let pixels = noise_pixels(w, h, u64::from(w) * 131_071 + u64::from(h));
        let stream = encode_vp9(&pixels, w, h).unwrap_or_else(|e| panic!("encode {w}x{h}: {e:?}"));
        let frame = decode_intra_frame(&stream).unwrap_or_else(|e| panic!("decode {w}x{h}: {e:?}"));
        assert_eq!((frame.width, frame.height), (w, h), "geometry {w}x{h}");

        let bytes = decode_vp9(&stream).unwrap_or_else(|e| panic!("decode_vp9 {w}x{h}: {e:?}"));
        assert_eq!(bytes, pixels, "lossless round-trip {w}x{h}");
    }
}

/// A flat input reconstructs to exactly that flat value (not the
/// prediction default): the residual path carries the content.
#[test]
fn encoded_keyframe_reproduces_flat_fill() {
    let pixels = flat_pixels(64, 64, 200);
    let stream = encode_vp9(&pixels, 64, 64).expect("encode");
    let frame = decode_intra_frame(&stream).expect("decode");
    assert!(frame.y.iter().all(|&s| s == 200), "luma not flat 200");
    assert_eq!(decode_vp9(&stream).expect("planar"), pixels);
}

/// `encode_vp9` rejects degenerate / out-of-range inputs without
/// panicking.
#[test]
fn encode_rejects_bad_inputs() {
    // Zero dimension.
    assert!(encode_vp9(&flat_pixels(1, 1, 0), 0, 16).is_err());
    assert!(encode_vp9(&flat_pixels(1, 1, 0), 16, 0).is_err());
    // Too-short pixel buffer.
    assert!(encode_vp9(&[0u8; 3], 64, 64).is_err());
}

/// Encoding is a pure function of its inputs: the same pixels always
/// produce identical bytes, and different pixels produce a different
/// stream that still round-trips.
#[test]
fn encode_is_deterministic_and_content_sensitive() {
    let px1 = noise_pixels(96, 80, 7);
    let px2 = noise_pixels(96, 80, 8);
    let a = encode_vp9(&px1, 96, 80).expect("a");
    let b = encode_vp9(&px1, 96, 80).expect("b");
    let c = encode_vp9(&px2, 96, 80).expect("c");
    assert_eq!(a, b, "same input must encode identically");
    assert_ne!(a, c, "different content must yield a different stream");
    assert_eq!(decode_vp9(&c).expect("decode c"), px2);
}
