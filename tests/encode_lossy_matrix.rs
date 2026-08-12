//! Public-surface tests for the round-441 lossy format matrix:
//! [`encode_vp9_lossy_444`] / [`encode_vp9_lossy_422`] (profile 1) and
//! [`encode_vp9_lossy_hbd`] / [`encode_vp9_lossy_hbd_422`] (profiles
//! 2/3), each decoded back through the in-crate decoder.
//!
//! The deep decoder-mirror pins live in the crate's unit tests
//! (`pixel_encoder::lossy_matrix_tests`); these integration tests pin
//! the *public* contract: the coded stream decodes at the requested
//! §7.2.2 format, distortion is bounded by the §8.6.1 quantizer step,
//! the encode is byte-deterministic, and bad arguments are rejected.

use oxideav_vp9::{
    decode_intra_frame, encode_vp9_lossy_422, encode_vp9_lossy_444, encode_vp9_lossy_hbd,
    encode_vp9_lossy_hbd_422, Error,
};

/// Deterministic 8-bit planar content at the given chroma dims.
fn planar_u8(w: usize, h: usize, cw: usize, ch: usize) -> Vec<u8> {
    let f = |x: usize, y: usize, s: usize| ((x * 7 + y * 13 + s) % 251) as u8;
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            px.push(f(x, y, 0));
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            px.push(f(x, y, 40));
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            px.push(f(x, y, 90));
        }
    }
    px
}

/// Deterministic HBD planar content in the `bit_depth` range.
fn planar_u16(w: usize, h: usize, cw: usize, ch: usize, bit_depth: u32) -> Vec<u16> {
    let max = (1u32 << bit_depth) - 1;
    let f = |x: usize, y: usize, s: usize| ((x * 29 + y * 53 + s * 17) as u32 % (max + 1)) as u16;
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            px.push(f(x, y, 0));
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            px.push(f(x, y, 3));
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            px.push(f(x, y, 7));
        }
    }
    px
}

/// MSE of the decoded frame against the planar source.
fn mse_u16(dec: &[u16], src: &[u16]) -> f64 {
    let sse: f64 = dec
        .iter()
        .zip(src)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    sse / dec.len() as f64
}

#[test]
fn lossy_444_encodes_and_decodes() {
    let (w, h) = (52usize, 36usize);
    let px = planar_u8(w, h, w, h);
    let bytes = encode_vp9_lossy_444(&px, w as u32, h as u32, 100).expect("encode");
    assert_eq!(
        bytes,
        encode_vp9_lossy_444(&px, w as u32, h as u32, 100).expect("encode 2"),
        "byte-determinism"
    );
    let frame = decode_intra_frame(&bytes).expect("decode");
    assert_eq!((frame.width, frame.height), (w as u32, h as u32));
    assert_eq!(frame.bit_depth, 8);
    assert!(!frame.subsampling_x && !frame.subsampling_y, "4:4:4");
    let src: Vec<u16> = px.iter().map(|&b| u16::from(b)).collect();
    let dec: Vec<u16> = frame
        .y
        .iter()
        .chain(&frame.u)
        .chain(&frame.v)
        .copied()
        .collect();
    assert!(mse_u16(&dec, &src) < 300.0, "q=100 distortion regime");
}

#[test]
fn lossy_422_encodes_and_decodes() {
    let (w, h) = (52usize, 36usize);
    let (cw, ch) = (w.div_ceil(2), h);
    let px = planar_u8(w, h, cw, ch);
    let bytes = encode_vp9_lossy_422(&px, w as u32, h as u32, 100).expect("encode");
    let frame = decode_intra_frame(&bytes).expect("decode");
    assert_eq!((frame.width, frame.height), (w as u32, h as u32));
    assert!(frame.subsampling_x && !frame.subsampling_y, "4:2:2");
    assert_eq!(frame.u.len(), cw * ch, "§8.10 chroma extent");
    let src: Vec<u16> = px.iter().map(|&b| u16::from(b)).collect();
    let dec: Vec<u16> = frame
        .y
        .iter()
        .chain(&frame.u)
        .chain(&frame.v)
        .copied()
        .collect();
    assert!(mse_u16(&dec, &src) < 300.0, "q=100 distortion regime");
}

#[test]
fn lossy_hbd_420_and_444_encode_and_decode() {
    for (bit_depth, subsample) in [(10u8, true), (12, true), (10, false), (12, false)] {
        let (w, h) = (44usize, 28usize);
        let (cw, ch) = if subsample {
            (w.div_ceil(2), h.div_ceil(2))
        } else {
            (w, h)
        };
        let px = planar_u16(w, h, cw, ch, u32::from(bit_depth));
        let bytes = encode_vp9_lossy_hbd(&px, w as u32, h as u32, bit_depth, subsample, 110)
            .unwrap_or_else(|e| panic!("encode bd={bit_depth} ss={subsample}: {e}"));
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.bit_depth, bit_depth);
        assert_eq!(frame.subsampling_x, subsample);
        assert_eq!(frame.subsampling_y, subsample);
        assert_eq!(frame.u.len(), cw * ch);
        // The §8.6.1 quantizer step scales with depth; a generous
        // depth-scaled regime bound catches drift-class bugs.
        let scale = f64::from(1u32 << (bit_depth - 8));
        let dec: Vec<u16> = frame
            .y
            .iter()
            .chain(&frame.u)
            .chain(&frame.v)
            .copied()
            .collect();
        let mse = mse_u16(&dec, &px);
        assert!(
            mse < 2000.0 * scale * scale,
            "bd={bit_depth} ss={subsample}: MSE {mse}"
        );
    }
}

#[test]
fn lossy_hbd_422_encodes_and_decodes() {
    for bit_depth in [10u8, 12] {
        let (w, h) = (44usize, 28usize);
        let (cw, ch) = (w.div_ceil(2), h);
        let px = planar_u16(w, h, cw, ch, u32::from(bit_depth));
        let bytes = encode_vp9_lossy_hbd_422(&px, w as u32, h as u32, bit_depth, 110)
            .unwrap_or_else(|e| panic!("encode bd={bit_depth}: {e}"));
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.bit_depth, bit_depth);
        assert!(frame.subsampling_x && !frame.subsampling_y, "4:2:2");
        assert_eq!(frame.u.len(), cw * ch);
        let scale = f64::from(1u32 << (bit_depth - 8));
        let dec: Vec<u16> = frame
            .y
            .iter()
            .chain(&frame.u)
            .chain(&frame.v)
            .copied()
            .collect();
        let mse = mse_u16(&dec, &px);
        assert!(mse < 2000.0 * scale * scale, "bd={bit_depth}: MSE {mse}");
    }
}

/// Env-gated dump for **black-box validation**: writes each matrix
/// format's coded keyframe as `input.ivf` plus this crate's decode as
/// `crate-decode.yuv` (the [`oxideav_vp9::decode_vp9`] packed layout:
/// one byte per 8-bit sample, little-endian `u16` pairs at 10/12-bit)
/// under `$OXIDEAV_VP9_MATRIX_DUMP_DIR/<name>/`, so an external
/// reference decoder can be run as an opaque process against the same
/// bytes and byte-compared. No-op unless the env var is set.
#[test]
fn dump_matrix_streams_for_black_box_validation() {
    let Some(dir) = std::env::var_os("OXIDEAV_VP9_MATRIX_DUMP_DIR") else {
        return;
    };

    let ivf_wrap = |frames: &[Vec<u8>], w: u16, h: u16| -> Vec<u8> {
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&w.to_le_bytes());
        ivf.extend_from_slice(&h.to_le_bytes());
        ivf.extend_from_slice(&25u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        for (i, f) in frames.iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        ivf
    };

    let (w, h) = (64usize, 48usize);
    let q = 110u8;
    // (name, encoded bytes) per matrix format — sizes chosen MI-exact
    // so external raw-frame comparisons need no crop handling beyond
    // the coded dimensions.
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "lossy-420-8bit",
            oxideav_vp9::encode_vp9_lossy(
                &planar_u8(w, h, w.div_ceil(2), h.div_ceil(2)),
                w as u32,
                h as u32,
                q,
            )
            .expect("420"),
        ),
        (
            "lossy-444-8bit",
            encode_vp9_lossy_444(&planar_u8(w, h, w, h), w as u32, h as u32, q).expect("444"),
        ),
        (
            "lossy-422-8bit",
            encode_vp9_lossy_422(&planar_u8(w, h, w.div_ceil(2), h), w as u32, h as u32, q)
                .expect("422"),
        ),
        (
            "lossy-420-10bit",
            encode_vp9_lossy_hbd(
                &planar_u16(w, h, w.div_ceil(2), h.div_ceil(2), 10),
                w as u32,
                h as u32,
                10,
                true,
                q,
            )
            .expect("420-10"),
        ),
        (
            "lossy-420-12bit",
            encode_vp9_lossy_hbd(
                &planar_u16(w, h, w.div_ceil(2), h.div_ceil(2), 12),
                w as u32,
                h as u32,
                12,
                true,
                q,
            )
            .expect("420-12"),
        ),
        (
            "lossy-444-10bit",
            encode_vp9_lossy_hbd(
                &planar_u16(w, h, w, h, 10),
                w as u32,
                h as u32,
                10,
                false,
                q,
            )
            .expect("444-10"),
        ),
        (
            "lossy-444-12bit",
            encode_vp9_lossy_hbd(
                &planar_u16(w, h, w, h, 12),
                w as u32,
                h as u32,
                12,
                false,
                q,
            )
            .expect("444-12"),
        ),
        (
            "lossy-422-10bit",
            encode_vp9_lossy_hbd_422(
                &planar_u16(w, h, w.div_ceil(2), h, 10),
                w as u32,
                h as u32,
                10,
                q,
            )
            .expect("422-10"),
        ),
        (
            "lossy-422-12bit",
            encode_vp9_lossy_hbd_422(
                &planar_u16(w, h, w.div_ceil(2), h, 12),
                w as u32,
                h as u32,
                12,
                q,
            )
            .expect("422-12"),
        ),
    ];

    for (name, bytes) in cases {
        let sub = std::path::Path::new(&dir).join(name);
        std::fs::create_dir_all(&sub).expect("create dump dir");
        let ivf = ivf_wrap(std::slice::from_ref(&bytes), w as u16, h as u16);
        std::fs::write(sub.join("input.ivf"), ivf).expect("write ivf");
        let decoded = oxideav_vp9::decode_vp9(&bytes).expect("crate decode");
        std::fs::write(sub.join("crate-decode.yuv"), decoded).expect("write yuv");
    }
}

#[test]
fn lossy_matrix_public_rejections() {
    let px8 = vec![0u8; 32 * 32 * 3];
    let px16 = vec![0u16; 32 * 32 * 3];
    assert_eq!(
        encode_vp9_lossy_444(&px8, 32, 32, 0).unwrap_err(),
        Error::Unsupported,
        "q == 0 is the lossless path"
    );
    assert_eq!(
        encode_vp9_lossy_422(&px8[..10], 32, 32, 100).unwrap_err(),
        Error::Unsupported,
        "short buffer"
    );
    assert_eq!(
        encode_vp9_lossy_hbd(&px16, 32, 32, 9, true, 100).unwrap_err(),
        Error::Unsupported,
        "bit_depth 9 is not a §7.2.2 depth"
    );
    let mut hot = vec![0u16; 32 * 32 * 3];
    hot[0] = 1 << 12;
    assert_eq!(
        encode_vp9_lossy_hbd_422(&hot, 32, 32, 12, 100).unwrap_err(),
        Error::Unsupported,
        "sample exceeds the 12-bit range"
    );
}
