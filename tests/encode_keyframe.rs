//! Integration tests for the VP9 keyframe encoder entry `encode_vp9`.
//!
//! These exercise the public encode → decode round-trip across a sweep of
//! frame geometries (including 1x1, partial-superblock, multi-superblock,
//! and non-square shapes). The encoder currently emits an all-skip
//! DC_PRED keyframe, so every decode must succeed and report the
//! requested geometry; the robustness sweep additionally confirms the
//! encoder never panics on edge sizes.

use oxideav_vp9::{decode_intra_frame, decode_vp9, encode_vp9};

/// Build a flat 8-bit 4:2:0 planar pixel buffer for `width × height`.
fn flat_pixels(width: u32, height: u32, fill: u8) -> Vec<u8> {
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let len = (width as usize) * (height as usize) + 2 * cw * ch;
    vec![fill; len]
}

/// `encode_vp9` produces a stream `decode_intra_frame` accepts, reporting
/// the requested geometry, for a representative geometry sweep.
#[test]
fn encode_decode_geometry_sweep() {
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
        let pixels = flat_pixels(w, h, 128);
        let stream = encode_vp9(&pixels, w, h).unwrap_or_else(|e| panic!("encode {w}x{h}: {e:?}"));
        let frame = decode_intra_frame(&stream).unwrap_or_else(|e| panic!("decode {w}x{h}: {e:?}"));
        assert_eq!((frame.width, frame.height), (w, h), "geometry {w}x{h}");

        // The packed planar output is the full 4:2:0 size.
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let bytes = decode_vp9(&stream).unwrap_or_else(|e| panic!("decode_vp9 {w}x{h}: {e:?}"));
        assert_eq!(
            bytes.len(),
            (w * h) as usize + 2 * cw * ch,
            "packed size {w}x{h}"
        );
    }
}

/// The all-skip DC_PRED keyframe reconstructs to a flat 128 mid-grey
/// fill on the luma plane (no residual, loop filter off).
#[test]
fn encoded_keyframe_is_flat_grey() {
    let pixels = flat_pixels(64, 64, 200); // input fill is ignored (all-skip).
    let stream = encode_vp9(&pixels, 64, 64).expect("encode");
    let frame = decode_intra_frame(&stream).expect("decode");
    assert!(frame.y.iter().all(|&s| s == 128), "luma not flat 128");
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

/// Encoding is a pure function of (width, height) for the all-skip path:
/// the same geometry always produces identical bytes.
#[test]
fn encode_is_deterministic() {
    let a = encode_vp9(&flat_pixels(96, 80, 10), 96, 80).expect("a");
    let b = encode_vp9(&flat_pixels(96, 80, 99), 96, 80).expect("b");
    assert_eq!(a, b, "same geometry must encode identically (all-skip)");
}
