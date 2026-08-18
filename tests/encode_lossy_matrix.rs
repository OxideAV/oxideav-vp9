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
    decode_intra_frame, encode_vp9_lossy_422, encode_vp9_lossy_440, encode_vp9_lossy_444,
    encode_vp9_lossy_hbd, encode_vp9_lossy_hbd_422, encode_vp9_lossy_hbd_440, Error,
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
fn lossy_440_encodes_and_decodes() {
    let (w, h) = (52usize, 36usize);
    let (cw, ch) = (w, h.div_ceil(2));
    let px = planar_u8(w, h, cw, ch);
    let bytes = encode_vp9_lossy_440(&px, w as u32, h as u32, 100).expect("encode");
    assert_eq!(
        bytes,
        encode_vp9_lossy_440(&px, w as u32, h as u32, 100).expect("encode 2"),
        "byte-determinism"
    );
    let frame = decode_intra_frame(&bytes).expect("decode");
    assert_eq!((frame.width, frame.height), (w as u32, h as u32));
    assert!(!frame.subsampling_x && frame.subsampling_y, "4:4:0");
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

#[test]
fn lossy_hbd_440_encodes_and_decodes() {
    for bit_depth in [10u8, 12] {
        let (w, h) = (44usize, 28usize);
        let (cw, ch) = (w, h.div_ceil(2));
        let px = planar_u16(w, h, cw, ch, u32::from(bit_depth));
        let bytes = encode_vp9_lossy_hbd_440(&px, w as u32, h as u32, bit_depth, 110)
            .unwrap_or_else(|e| panic!("encode bd={bit_depth}: {e}"));
        let frame = decode_intra_frame(&bytes).expect("decode");
        assert_eq!(frame.bit_depth, bit_depth);
        assert!(!frame.subsampling_x && frame.subsampling_y, "4:4:0");
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

/// 8-bit planar GOP frame `k`: the [`planar_u8`] texture translating
/// by (2, 1) px per frame — P-frames earn genuine `NEWMV` leaves.
fn planar_u8_frame(w: usize, h: usize, cw: usize, ch: usize, k: usize) -> Vec<u8> {
    let f = |x: usize, y: usize, s: usize| (((x + 2 * k) * 7 + (y + k) * 13 + s) % 251) as u8;
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

/// HBD planar GOP frame `k` (translating like [`planar_u8_frame`]).
fn planar_u16_frame(
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    bit_depth: u32,
    k: usize,
) -> Vec<u16> {
    let max = (1u32 << bit_depth) - 1;
    let f = |x: usize, y: usize, s: usize| {
        (((x + 2 * k) * 29 + (y + k) * 53 + s * 17) as u32 % (max + 1)) as u16
    };
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

/// The lossy **sequence** encoders across the format matrix: a 4-frame
/// translating GOP per format decodes through
/// [`oxideav_vp9::decode_vp9_sequence`] with every frame inside the
/// quantizer's distortion regime (reference-chain drift would blow the
/// later frames' MSE), at the coded format, byte-deterministically.
#[test]
fn lossy_sequence_matrix_decodes_without_drift() {
    let (w, h) = (48usize, 32usize);
    let q = 110u8;
    // (name, bit_depth, ssx, ssy, encode closure)
    type SeqEncode = Box<dyn Fn(&[Vec<u16>]) -> Vec<Vec<u8>>>;
    type SeqEncodeU8<'e> = &'e dyn Fn(&[&[u8]]) -> Result<Vec<Vec<u8>>, Error>;
    let seq_u8 = |frames: &[Vec<u16>], enc: SeqEncodeU8<'_>| -> Vec<Vec<u8>> {
        let frames8: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| f.iter().map(|&s| s as u8).collect())
            .collect();
        let refs: Vec<&[u8]> = frames8.iter().map(|f| f.as_slice()).collect();
        enc(&refs).expect("encode")
    };
    let cases: Vec<(&str, u8, bool, bool, SeqEncode)> = vec![
        (
            "seq-444-8bit",
            8,
            false,
            false,
            Box::new(move |frames| {
                seq_u8(frames, &|refs| {
                    oxideav_vp9::encode_vp9_lossy_sequence_444(refs, w as u32, h as u32, q)
                })
            }),
        ),
        (
            "seq-422-8bit",
            8,
            true,
            false,
            Box::new(move |frames| {
                seq_u8(frames, &|refs| {
                    oxideav_vp9::encode_vp9_lossy_sequence_422(refs, w as u32, h as u32, q)
                })
            }),
        ),
        (
            "seq-440-8bit",
            8,
            false,
            true,
            Box::new(move |frames| {
                seq_u8(frames, &|refs| {
                    oxideav_vp9::encode_vp9_lossy_sequence_440(refs, w as u32, h as u32, q)
                })
            }),
        ),
        (
            "seq-420-10bit",
            10,
            true,
            true,
            Box::new(move |frames| {
                let refs: Vec<&[u16]> = frames.iter().map(|f| f.as_slice()).collect();
                oxideav_vp9::encode_vp9_lossy_sequence_hbd(&refs, w as u32, h as u32, 10, true, q)
                    .expect("encode")
            }),
        ),
        (
            "seq-444-12bit",
            12,
            false,
            false,
            Box::new(move |frames| {
                let refs: Vec<&[u16]> = frames.iter().map(|f| f.as_slice()).collect();
                oxideav_vp9::encode_vp9_lossy_sequence_hbd(&refs, w as u32, h as u32, 12, false, q)
                    .expect("encode")
            }),
        ),
        (
            "seq-422-10bit",
            10,
            true,
            false,
            Box::new(move |frames| {
                let refs: Vec<&[u16]> = frames.iter().map(|f| f.as_slice()).collect();
                oxideav_vp9::encode_vp9_lossy_sequence_hbd_422(&refs, w as u32, h as u32, 10, q)
                    .expect("encode")
            }),
        ),
        (
            "seq-440-12bit",
            12,
            false,
            true,
            Box::new(move |frames| {
                let refs: Vec<&[u16]> = frames.iter().map(|f| f.as_slice()).collect();
                oxideav_vp9::encode_vp9_lossy_sequence_hbd_440(&refs, w as u32, h as u32, 12, q)
                    .expect("encode")
            }),
        ),
    ];

    for (name, bd, ssx, ssy, encode) in cases {
        let cw = if ssx { w.div_ceil(2) } else { w };
        let ch = if ssy { h.div_ceil(2) } else { h };
        let frames: Vec<Vec<u16>> = (0..4)
            .map(|k| planar_u16_frame(w, h, cw, ch, u32::from(bd), k))
            .collect();
        let coded = encode(&frames);
        assert_eq!(coded, encode(&frames), "{name}: byte-determinism");
        assert_eq!(coded.len(), 4, "{name}");

        let refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
        let decoded = oxideav_vp9::decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4, "{name}");
        let scale = f64::from(1u32 << (bd - 8));
        for (i, (frame, src)) in decoded.iter().zip(&frames).enumerate() {
            assert_eq!(
                (frame.bit_depth, frame.subsampling_x, frame.subsampling_y),
                (bd, ssx, ssy),
                "{name} frame {i}"
            );
            let dec: Vec<u16> = frame
                .y
                .iter()
                .chain(&frame.u)
                .chain(&frame.v)
                .copied()
                .collect();
            let mse = mse_u16(&dec, src);
            assert!(
                mse < 2000.0 * scale * scale,
                "{name} frame {i}: MSE {mse} — reference-chain drift?"
            );
        }
    }
}

/// Env-gated dump for **black-box validation**: writes each matrix
/// format's coded keyframe as `input.ivf` plus this crate's decode as
/// `crate-decode.yuv` (the [`oxideav_vp9::decode_vp9`] packed layout:
/// one byte per 8-bit sample, little-endian `u16` pairs at 10/12-bit)
/// under `$OXIDEAV_VP9_MATRIX_DUMP_DIR/<name>/`, so an external
/// reference decoder can be run as an opaque process against the same
/// bytes and byte-compared. Multi-frame GOP dumps per sequence entry
/// point ride along (concatenated per-frame decodes). No-op unless the
/// env var is set.
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
    // Half-flat content: the flat half's leaves quantize to zero and
    // take the round-441 keyframe **skip election** — this stream's
    // black-box decode pins the elected skip syntax externally.
    let half_flat: Vec<u8> = {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h {
            for x in 0..w {
                px.push(if x < w / 2 {
                    128
                } else {
                    ((x * 31 + y * 17) % 223) as u8
                });
            }
        }
        px.resize(w * h + 2 * cw * ch, 128);
        px
    };
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
            "lossy-420-8bit-skipkf",
            oxideav_vp9::encode_vp9_lossy(&half_flat, w as u32, h as u32, 120).expect("skipkf"),
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
            "lossy-440-8bit",
            encode_vp9_lossy_440(&planar_u8(w, h, w, h.div_ceil(2)), w as u32, h as u32, q)
                .expect("440"),
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
        (
            "lossy-440-12bit",
            encode_vp9_lossy_hbd_440(
                &planar_u16(w, h, w, h.div_ceil(2), 12),
                w as u32,
                h as u32,
                12,
                q,
            )
            .expect("440-12"),
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

    // Multi-frame GOPs through the new sequence entry points (chain
    // framing: prev-MV modeling + compound election live at every
    // format).
    let n = 4usize;
    let gop_u8 = |ssx: bool, ssy: bool| -> Vec<Vec<u8>> {
        let cw = if ssx { w.div_ceil(2) } else { w };
        let ch = if ssy { h.div_ceil(2) } else { h };
        (0..n).map(|k| planar_u8_frame(w, h, cw, ch, k)).collect()
    };
    let gop_u16 = |bd: u32, ssx: bool, ssy: bool| -> Vec<Vec<u16>> {
        let cw = if ssx { w.div_ceil(2) } else { w };
        let ch = if ssy { h.div_ceil(2) } else { h };
        (0..n)
            .map(|k| planar_u16_frame(w, h, cw, ch, bd, k))
            .collect()
    };
    let gop_cases: Vec<(&str, Vec<Vec<u8>>)> = {
        let f444 = gop_u8(false, false);
        let f422 = gop_u8(true, false);
        let f420_10 = gop_u16(10, true, true);
        let f444_12 = gop_u16(12, false, false);
        let f422_10 = gop_u16(10, true, false);
        fn refs8(v: &[Vec<u8>]) -> Vec<&[u8]> {
            v.iter().map(|f| f.as_slice()).collect()
        }
        fn refs16(v: &[Vec<u16>]) -> Vec<&[u16]> {
            v.iter().map(|f| f.as_slice()).collect()
        }
        vec![
            (
                "gop-444-8bit",
                oxideav_vp9::encode_vp9_lossy_sequence_444(&refs8(&f444), w as u32, h as u32, q)
                    .expect("gop 444"),
            ),
            (
                "gop-422-8bit",
                oxideav_vp9::encode_vp9_lossy_sequence_422(&refs8(&f422), w as u32, h as u32, q)
                    .expect("gop 422"),
            ),
            (
                "gop-420-10bit",
                oxideav_vp9::encode_vp9_lossy_sequence_hbd(
                    &refs16(&f420_10),
                    w as u32,
                    h as u32,
                    10,
                    true,
                    q,
                )
                .expect("gop 420-10"),
            ),
            (
                "gop-444-12bit",
                oxideav_vp9::encode_vp9_lossy_sequence_hbd(
                    &refs16(&f444_12),
                    w as u32,
                    h as u32,
                    12,
                    false,
                    q,
                )
                .expect("gop 444-12"),
            ),
            (
                "gop-422-10bit",
                oxideav_vp9::encode_vp9_lossy_sequence_hbd_422(
                    &refs16(&f422_10),
                    w as u32,
                    h as u32,
                    10,
                    q,
                )
                .expect("gop 422-10"),
            ),
            // Mixed static/moving content at a coarse quantizer: the
            // chained encoder's §6.2.8 loop-filter **delta election**
            // codes a mode-delta update mid-GOP and a later frame
            // relies on the §7.2.8 persisted value — the black-box
            // decode pins both against the reference decoder.
            ("gop-420-8bit-lfdeltas", {
                let tex = |x: i64, y: i64| -> u8 { (((x * 7 + y * 13) % 61) * 4 % 251) as u8 };
                let mixed: Vec<Vec<u8>> = (0..4i64)
                    .map(|k| {
                        let cw = w.div_ceil(2);
                        let ch = h.div_ceil(2);
                        let mut px = vec![128u8; w * h + 2 * cw * ch];
                        for y in 0..h as i64 {
                            for x in 0..w as i64 {
                                px[(y * w as i64 + x) as usize] = if x < (w as i64) / 2 {
                                    tex(x, y)
                                } else {
                                    tex(x + 3 * k, y + 2 * k)
                                };
                            }
                        }
                        px
                    })
                    .collect();
                let refs: Vec<&[u8]> = mixed.iter().map(|f| f.as_slice()).collect();
                oxideav_vp9::encode_vp9_lossy_sequence_chained(&refs, w as u32, h as u32, 170)
                    .expect("gop lfdeltas")
            }),
            // Round-445 default-path streams: the promoted chained
            // defaults themselves (4:2:0 lossy, lossless with the
            // skip-elected keyframe) and the rate-controlled chain
            // (bisected per-frame q + budget-guarded delta election).
            ("gop-420-8bit-default", {
                let f420 = gop_u8(true, true);
                oxideav_vp9::encode_vp9_lossy_sequence(&refs8(&f420), w as u32, h as u32, q)
                    .expect("gop 420 default")
            }),
            ("gop-lossless-chained", {
                let flat_patch: Vec<Vec<u8>> = (0..4usize)
                    .map(|k| {
                        let cw = w.div_ceil(2);
                        let ch = h.div_ceil(2);
                        let mut px = vec![100u8; w * h + 2 * cw * ch];
                        for y in 0..16usize {
                            for x in 0..16usize {
                                px[(y + 12) * w + x + 8 + 3 * k] =
                                    ((x * 31 + y * 17 + 7) % 251) as u8;
                            }
                        }
                        px
                    })
                    .collect();
                let refs: Vec<&[u8]> = flat_patch.iter().map(|f| f.as_slice()).collect();
                oxideav_vp9::encode_vp9_lossless_sequence(&refs, w as u32, h as u32)
                    .expect("gop lossless chained")
            }),
            ("gop-420-8bit-rc", {
                let f420 = gop_u8(true, true);
                oxideav_vp9::encode_vp9_lossy_sequence_rc(&refs8(&f420), w as u32, h as u32, 1200)
                    .expect("gop rc")
            }),
            // Round-448 public 4:4:0 entries: the last §7.2.2 geometry
            // without a dedicated public encoder (ssx = 0, ssy = 1 —
            // the §8.5.2.1 row-only chroma MV rounding), at profile 1
            // (8-bit) and the profile-3 12-bit corner.
            ("gop-440-8bit", {
                let f440 = gop_u8(false, true);
                oxideav_vp9::encode_vp9_lossy_sequence_440(&refs8(&f440), w as u32, h as u32, q)
                    .expect("gop 440")
            }),
            ("gop-440-12bit", {
                let f440_12 = gop_u16(12, false, true);
                oxideav_vp9::encode_vp9_lossy_sequence_hbd_440(
                    &refs16(&f440_12),
                    w as u32,
                    h as u32,
                    12,
                    q,
                )
                .expect("gop 440-12")
            }),
        ]
    };
    for (name, frames) in gop_cases {
        let sub = std::path::Path::new(&dir).join(name);
        std::fs::create_dir_all(&sub).expect("create dump dir");
        std::fs::write(sub.join("input.ivf"), ivf_wrap(&frames, w as u16, h as u16))
            .expect("write ivf");
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = oxideav_vp9::decode_vp9_sequence(&refs).expect("crate decode");
        let mut yuv = Vec::new();
        for f in &decoded {
            yuv.extend_from_slice(&f.to_planar_bytes());
        }
        std::fs::write(sub.join("crate-decode.yuv"), yuv).expect("write yuv");
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
    assert_eq!(
        encode_vp9_lossy_440(&px8, 32, 32, 0).unwrap_err(),
        Error::Unsupported,
        "q == 0 is the lossless path"
    );
    assert_eq!(
        encode_vp9_lossy_440(&px8[..10], 32, 32, 100).unwrap_err(),
        Error::Unsupported,
        "short buffer"
    );
    assert_eq!(
        encode_vp9_lossy_hbd_440(&px16, 32, 32, 9, 100).unwrap_err(),
        Error::Unsupported,
        "bit_depth 9 is not a §7.2.2 depth"
    );
    assert_eq!(
        encode_vp9_lossy_hbd_440(&hot, 32, 32, 12, 100).unwrap_err(),
        Error::Unsupported,
        "sample exceeds the 12-bit range"
    );
}
