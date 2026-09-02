//! Public-surface tests for the round-452 **GOP-structure** encoder
//! entries — the alt-ref pyramid ([`encode_vp9_lossy_sequence_altref`])
//! and its siblings — decoded back through the in-crate decoder.
//!
//! The deep decoder-mirror pins (every shown frame equal to the
//! encoder's reconstruction, sample-for-sample, hidden alt-refs
//! surfacing at their `show_existing_frame` position) live in the
//! crate's unit tests; these integration tests pin the *public*
//! contract: packet structure, shown-frame count, bounded distortion,
//! byte-determinism, and rejections.

use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence, encode_vp9_lossy_sequence_altref,
    encode_vp9_lossy_sequence_resized, encode_vp9_lossy_sequence_resized_with,
    encode_vp9_lossy_sequence_with, Error, Vp9GopConfig, Vp9Segmentation,
};

/// Source frame `k` of a translating 4:2:0 scene (a ramp plus a sharp
/// textured patch moving 2 px/frame).
fn scene_frame(w: usize, h: usize, k: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            let mut v = ((x + 2 * k) * 3 + y * 2) % 200 + 20;
            let px_x = x as i64 - 2 * k as i64;
            if (8..24).contains(&px_x) && (8..24).contains(&y) {
                v = (px_x as usize * 37 + y * 53) % 255;
            }
            px.push(v as u8);
        }
    }
    for plane in 0..2usize {
        for y in 0..ch {
            for x in 0..cw {
                px.push(((x + k) * 5 + y * 3 + plane * 90) as u8);
            }
        }
    }
    px
}

fn refs(v: &[Vec<u8>]) -> Vec<&[u8]> {
    v.iter().map(|f| f.as_slice()).collect()
}

fn psnr_luma(dec: &[u16], src: &[u8], n: usize) -> f64 {
    let sse: f64 = dec[..n]
        .iter()
        .zip(src)
        .map(|(&d, &s)| {
            let e = f64::from(d) - f64::from(s);
            e * e
        })
        .sum();
    let mse = sse / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Packet structure: keyframe, then per full group `interval + 1`
/// packets (hidden ARF + `interval - 1` shown P-frames + one
/// `show_existing_frame` byte), a trailing partial group likewise, a
/// lone trailing frame as a plain P-frame; the decoder returns exactly
/// the input frame count in display order at bounded distortion.
#[test]
fn altref_pyramid_structure_and_decode() {
    let (w, h) = (64u32, 48u32);
    let n = 8usize;
    let src: Vec<Vec<u8>> = (0..n).map(|k| scene_frame(64, 48, k)).collect();
    let packets = encode_vp9_lossy_sequence_altref(&refs(&src), w, h, 100, 3).expect("encode");
    // groups: [1,2,3] [4,5,6] [7] -> 1 + 4 + 4 + 1.
    assert_eq!(packets.len(), 10);
    assert_eq!(
        packets[4].len(),
        1,
        "show_existing_frame packet is one byte"
    );
    assert_eq!(packets[8].len(), 1);
    // §6.2 bit layout: frame_marker(2) = 2, profile_low_bit = 0,
    // profile_high_bit = 0, show_existing_frame = 1, then the 3-bit
    // frame_to_show_map_idx — the first group's ARF lives in slot 2.
    assert_eq!(packets[4][0], 0b1000_1010);
    // Second group's ARF took the freed slot 0.
    assert_eq!(packets[8][0], 0b1000_1000);

    let decoded = decode_vp9_sequence(&refs(&packets)).expect("decode");
    assert_eq!(decoded.len(), n);
    for (k, f) in decoded.iter().enumerate() {
        assert_eq!((f.width, f.height), (w, h));
        let p = psnr_luma(&f.y, &src[k], (w * h) as usize);
        assert!(p > 30.0, "frame {k} luma PSNR {p:.2} dB too low");
    }
    // Byte-determinism.
    assert_eq!(
        packets,
        encode_vp9_lossy_sequence_altref(&refs(&src), w, h, 100, 3).unwrap()
    );
}

/// Interval 1 degenerates to plain shown P-frames: one packet per
/// frame, no one-byte packets, every frame decodable.
#[test]
fn altref_pyramid_interval_one_is_plain_chain() {
    let src: Vec<Vec<u8>> = (0..4).map(|k| scene_frame(32, 32, k)).collect();
    let packets = encode_vp9_lossy_sequence_altref(&refs(&src), 32, 32, 120, 1).expect("encode");
    assert_eq!(packets.len(), 4);
    assert!(packets.iter().all(|p| p.len() > 1));
    assert_eq!(decode_vp9_sequence(&refs(&packets)).unwrap().len(), 4);
}

/// On a cross-fade (frame 2 is the exact midpoint of frames 1 and 3)
/// the pyramid's `[ LAST, ALTREF ]` compound over the true future
/// alt-ref codes the midpoint frame far below the two-slot chain,
/// whose ALTREF aliases the keyframe.
#[test]
fn altref_pyramid_compound_over_future_altref_wins_on_cross_fade() {
    let (w, h) = (64usize, 64usize);
    let cw = 32usize;
    let mut a = vec![0u8; w * h + 2 * cw * cw];
    let mut b = vec![0u8; w * h + 2 * cw * cw];
    let mut s = 0x1234_5678u32;
    for i in 0..a.len() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        a[i] = (s >> 24) as u8;
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        b[i] = (s >> 24) as u8;
    }
    let mid: Vec<u8> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (u16::from(x) + u16::from(y)).div_ceil(2) as u8)
        .collect();
    let flat = vec![128u8; a.len()];
    let src = vec![flat, a, mid, b];
    let pyr = encode_vp9_lossy_sequence_altref(&refs(&src), 64, 64, 60, 3).expect("pyr");
    let chain = encode_vp9_lossy_sequence(&refs(&src), 64, 64, 60).expect("chain");
    // pyramid packets: kf, ARF(b), P(a), P(mid), SE — the midpoint is
    // packets[3]; the chain's midpoint is chain[2].
    assert_eq!(pyr.len(), 5);
    assert!(
        pyr[3].len() * 2 < chain[2].len(),
        "pyramid midpoint {} B should be well under the chain's {} B",
        pyr[3].len(),
        chain[2].len()
    );
    assert_eq!(decode_vp9_sequence(&refs(&pyr)).unwrap().len(), 4);
}

#[test]
fn altref_pyramid_rejections() {
    let f = scene_frame(16, 16, 0);
    assert_eq!(
        encode_vp9_lossy_sequence_altref(&[], 16, 16, 100, 2).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_altref(&[&f], 16, 16, 0, 2).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_altref(&[&f], 16, 16, 100, 0).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_altref(&[&f[..10]], 16, 16, 100, 2).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_altref(&[&f], 0, 16, 100, 2).unwrap_err(),
        Error::Unsupported
    );
}

/// The [`Vp9GopConfig`] entry: default config is a plain chain (one
/// packet per frame), the alt-ref entry is a thin wrapper (byte-identical
/// packets), and every [`Vp9Segmentation`] mode yields a decodable GOP
/// with exactly the input frame count at bounded distortion.
#[test]
fn gop_config_entry_contract() {
    let (w, h) = (64u32, 48u32);
    let src: Vec<Vec<u8>> = (0..5).map(|k| scene_frame(64, 48, k)).collect();

    let base = encode_vp9_lossy_sequence_with(&refs(&src), w, h, &Vp9GopConfig::new(110))
        .expect("default config");
    assert_eq!(base.len(), 5);

    let mut alt = Vp9GopConfig::new(110);
    alt.altref_interval = 2;
    let via_cfg = encode_vp9_lossy_sequence_with(&refs(&src), w, h, &alt).expect("cfg altref");
    let via_entry =
        encode_vp9_lossy_sequence_altref(&refs(&src), w, h, 110, 2).expect("entry altref");
    assert_eq!(via_cfg, via_entry, "the altref entry is a thin wrapper");

    for seg in [
        Vp9Segmentation::AdaptiveQuant,
        Vp9Segmentation::StaticSkip,
        Vp9Segmentation::Full,
    ] {
        let mut cfg = Vp9GopConfig::new(110);
        cfg.altref_interval = 3;
        cfg.segmentation = seg;
        let packets = encode_vp9_lossy_sequence_with(&refs(&src), w, h, &cfg).expect("seg cfg");
        let decoded = decode_vp9_sequence(&refs(&packets)).expect("decode");
        assert_eq!(decoded.len(), 5, "{seg:?}");
        for (k, f) in decoded.iter().enumerate() {
            let p = psnr_luma(&f.y, &src[k], (w * h) as usize);
            assert!(p > 28.0, "{seg:?} frame {k} luma PSNR {p:.2} dB");
        }
        // Byte-determinism.
        assert_eq!(
            packets,
            encode_vp9_lossy_sequence_with(&refs(&src), w, h, &cfg).unwrap()
        );
    }

    // Intra-only alt-refs: same packet structure, the hidden frames are
    // §6.2 intra-only (profile-0 header, no reference), still exactly
    // the input frame count of shown frames.
    let mut io = Vp9GopConfig::new(110);
    io.altref_interval = 3;
    io.intra_only_altref = true;
    let packets = encode_vp9_lossy_sequence_with(&refs(&src), w, h, &io).expect("intra-only cfg");
    // 5 frames: kf + group [1,2,3] (ARF, P, P, SE) + the lone frame 4.
    assert_eq!(packets.len(), 1 + 4 + 1);
    let decoded = decode_vp9_sequence(&refs(&packets)).expect("decode");
    assert_eq!(decoded.len(), 5);
    // §6.2 bit layout of the hidden intra-only header: frame_marker 10,
    // profile bits 00, show_existing 0, frame_type 1 (non-key),
    // show_frame 0, error_resilient 0, intra_only 1 -> 1000_0100 1...
    assert_eq!(packets[1][0], 0b1000_0100);
    assert_eq!(packets[1][1] >> 7, 1);

    let mut bad = Vp9GopConfig::new(0);
    bad.segmentation = Vp9Segmentation::Full;
    assert_eq!(
        encode_vp9_lossy_sequence_with(&refs(&src), w, h, &bad).unwrap_err(),
        Error::Unsupported
    );
}

/// Tile axes on [`Vp9GopConfig`]: a 2-tile-column config on a
/// wide-enough frame yields a decodable multi-tile GOP; configs outside
/// the §6.2.13 / §6.2.14 writable range are rejected.
#[test]
fn gop_config_tiles_contract() {
    // 2 columns need sb64_cols >= 8 (512 px).
    let (w, h) = (512u32, 32u32);
    let src: Vec<Vec<u8>> = (0..2)
        .map(|k| scene_frame(w as usize, h as usize, k))
        .collect();
    let mut cfg = Vp9GopConfig::new(160);
    cfg.tile_cols_log2 = 1;
    cfg.tile_rows_log2 = 0;
    let packets = encode_vp9_lossy_sequence_with(&refs(&src), w, h, &cfg).expect("tiled encode");
    let decoded = decode_vp9_sequence(&refs(&packets)).expect("decode");
    assert_eq!(decoded.len(), 2);
    for (k, f) in decoded.iter().enumerate() {
        let p = psnr_luma(&f.y, &src[k], (w * h) as usize);
        assert!(p > 26.0, "frame {k} luma PSNR {p:.2} dB");
    }

    // A 64-px-wide frame cannot code 2 tile columns (max_log2 = 0).
    let narrow: Vec<Vec<u8>> = (0..2).map(|k| scene_frame(64, 48, k)).collect();
    let mut bad = Vp9GopConfig::new(160);
    bad.tile_cols_log2 = 1;
    assert_eq!(
        encode_vp9_lossy_sequence_with(&refs(&narrow), 64, 48, &bad).unwrap_err(),
        Error::Unsupported
    );
    // tile_rows_log2 > 2 is uncodeable (§6.2.13).
    let mut bad2 = Vp9GopConfig::new(160);
    bad2.tile_rows_log2 = 3;
    assert_eq!(
        encode_vp9_lossy_sequence_with(&refs(&narrow), 64, 48, &bad2).unwrap_err(),
        Error::Unsupported
    );
}

/// The resized-sequence entry: mid-stream coded-size changes decode at
/// their declared sizes; out-of-§5-ratio and malformed inputs reject.
#[test]
fn resized_sequence_contract() {
    let sizes: [(u32, u32); 3] = [(96, 64), (48, 32), (96, 64)];
    let src: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(k, &(w, h))| scene_frame(w as usize, h as usize, k))
        .collect();
    let packets =
        encode_vp9_lossy_sequence_resized(&refs(&src), &sizes, 120).expect("resized encode");
    let decoded = decode_vp9_sequence(&refs(&packets)).expect("decode");
    assert_eq!(decoded.len(), 3);
    for (k, (f, &(w, h))) in decoded.iter().zip(sizes.iter()).enumerate() {
        assert_eq!((f.width, f.height), (w, h), "frame {k}");
    }
    assert_eq!(
        packets,
        encode_vp9_lossy_sequence_resized(&refs(&src), &sizes, 120).unwrap()
    );

    // > 2x downscale between consecutive frames is outside the §5
    // scaling bounds.
    let big = scene_frame(128, 128, 0);
    let tiny = scene_frame(32, 32, 1);
    assert_eq!(
        encode_vp9_lossy_sequence_resized(&[&big, &tiny], &[(128, 128), (32, 32)], 120)
            .unwrap_err(),
        Error::Unsupported
    );
    // Mismatched lengths / empty input / lossless qindex reject.
    assert_eq!(
        encode_vp9_lossy_sequence_resized(&[&big], &[(128, 128), (64, 64)], 120).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_resized(&[], &[], 120).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_resized(&[&big], &[(128, 128)], 0).unwrap_err(),
        Error::Unsupported
    );
}

// ----- round-452 fixture packages: builders + staging + identity -----

/// Wrap coded packets in a minimal IVF container (the corpus layout).
fn ivf_wrap(packets: &[Vec<u8>], w: u16, h: u16) -> Vec<u8> {
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&w.to_le_bytes());
    ivf.extend_from_slice(&h.to_le_bytes());
    ivf.extend_from_slice(&25u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (i, f) in packets.iter().enumerate() {
        ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes());
        ivf.extend_from_slice(f);
    }
    ivf
}

/// The four round-452 fixture streams, built deterministically through
/// the PUBLIC entries: `(name, ivf bytes, expected.yuv = the crate's
/// own decode, locally verified byte-exact against the black-box
/// reference decoder at build time — hidden packets carry IVF
/// timestamps, so the reference decode runs with timestamp passthrough,
/// and the resized stream additionally with output auto-scaling
/// disabled)`.
fn build_r452_fixture_streams() -> Vec<(&'static str, Vec<u8>, Vec<u8>)> {
    // The packages pin the ROUND-452 framing: the default probability
    // banks with `frame_parallel_decoding_mode = 1`. Round 455 moved the
    // public default to the adaptive entropy model, so every builder
    // opts out explicitly and the staged bytes stay reproducible.
    let r452 = |q: u8| {
        let mut cfg = Vp9GopConfig::new(q);
        cfg.entropy_adaptation = false;
        cfg
    };
    let yuv_of = |packets: &[Vec<u8>]| -> Vec<u8> {
        let decoded = decode_vp9_sequence(&refs(packets)).expect("fixture decodes");
        let mut yuv = Vec::new();
        for f in &decoded {
            yuv.extend_from_slice(&f.to_planar_bytes());
        }
        yuv
    };
    let mut out = Vec::new();

    // 1. altref-pyramid-gop — the corpus's first ENCODER-MINTED hidden
    // alt-ref + show_existing_frame stream (three-slot election).
    {
        let src: Vec<Vec<u8>> = (0..8).map(|k| scene_frame(64, 48, k)).collect();
        let mut cfg = r452(100);
        cfg.altref_interval = 3;
        let packets = encode_vp9_lossy_sequence_with(&refs(&src), 64, 48, &cfg).expect("pyramid");
        let yuv = yuv_of(&packets);
        out.push(("altref-pyramid-gop", ivf_wrap(&packets, 64, 48), yuv));
    }
    // 2. seg-emitted-full-gop — all four SEG_LVL_* features emitted by
    // the encoder (fitted tree/pred probs, temporal updates, persistent
    // table) on the pyramid framing.
    {
        let src: Vec<Vec<u8>> = (0..6).map(|k| half_static_scene(64, 48, k)).collect();
        let mut cfg = r452(100);
        cfg.altref_interval = 3;
        cfg.segmentation = Vp9Segmentation::Full;
        let packets = encode_vp9_lossy_sequence_with(&refs(&src), 64, 48, &cfg).expect("seg");
        let yuv = yuv_of(&packets);
        out.push(("seg-emitted-full-gop", ivf_wrap(&packets, 64, 48), yuv));
    }
    // 3. tiles-2col-encoded-gop — the first self-encoded
    // multi-tile-column stream (the §9.2.4 tile-parallel decoder's
    // consumer-side twin).
    {
        let src: Vec<Vec<u8>> = (0..3).map(|k| scene_frame(512, 32, k)).collect();
        let mut cfg = r452(140);
        cfg.altref_interval = 2;
        cfg.tile_cols_log2 = 1;
        let packets = encode_vp9_lossy_sequence_with(&refs(&src), 512, 32, &cfg).expect("tiles");
        let yuv = yuv_of(&packets);
        out.push(("tiles-2col-encoded-gop", ivf_wrap(&packets, 512, 32), yuv));
    }
    // 4. intra-only-emitted-gop — the first ENCODER-MINTED hidden
    // intra-only frames (mid-GOP refresh points shown through
    // show_existing_frame), with the full segmentation emission.
    {
        let src: Vec<Vec<u8>> = (0..7).map(|k| half_static_scene(64, 48, k)).collect();
        let mut cfg = r452(100);
        cfg.altref_interval = 3;
        cfg.segmentation = Vp9Segmentation::Full;
        cfg.intra_only_altref = true;
        let packets =
            encode_vp9_lossy_sequence_with(&refs(&src), 64, 48, &cfg).expect("intra-only");
        let yuv = yuv_of(&packets);
        out.push(("intra-only-emitted-gop", ivf_wrap(&packets, 64, 48), yuv));
    }
    // 5. resized-encoded-gop — the first PIXEL-ACCURATE self-encoded
    // §8.5.2.3 stream (the r409 scaled-reference fixture is all-skip).
    {
        let sizes: [(u32, u32); 4] = [(128, 96), (64, 48), (96, 64), (128, 96)];
        let src: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(k, &(w, h))| scene_frame(w as usize, h as usize, k))
            .collect();
        let packets = encode_vp9_lossy_sequence_resized_with(&refs(&src), &sizes, &r452(110))
            .expect("resized");
        let yuv = yuv_of(&packets);
        out.push(("resized-encoded-gop", ivf_wrap(&packets, 128, 96), yuv));
    }
    out
}

/// [`scene_frame`] with a genuinely static (luma AND chroma) left half
/// — the static-skip segment's habitat (mirrors the unit-test scene).
fn half_static_scene(w: usize, h: usize, k: usize) -> Vec<u8> {
    let mut px = scene_frame(w, h, k);
    for y in 0..h {
        for x in 0..w / 2 {
            px[y * w + x] = ((x >> 1) + (y >> 1) + 60) as u8;
        }
    }
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    for plane in 0..2usize {
        let base = w * h + plane * cw * ch;
        for y in 0..ch {
            for x in 0..cw / 2 {
                px[base + y * cw + x] = (x + y + 90 + plane * 40) as u8;
            }
        }
    }
    px
}

/// The builders are deterministic (staging relies on it) and every
/// stream decodes through the in-crate decoder.
#[test]
fn r452_fixture_builders_are_deterministic() {
    let a = build_r452_fixture_streams();
    let b = build_r452_fixture_streams();
    assert_eq!(a.len(), 5);
    for ((n1, ivf1, yuv1), (n2, ivf2, yuv2)) in a.iter().zip(b.iter()) {
        assert_eq!(n1, n2);
        assert_eq!(ivf1, ivf2, "{n1} ivf not deterministic");
        assert_eq!(yuv1, yuv2, "{n1} yuv not deterministic");
        assert!(!yuv1.is_empty());
    }
}

/// Fixture-staging generator (round 452): under `OXIDEAV_VP9_STAGE_DIR`
/// emits each package as `<name>/input.ivf` + `<name>/expected.yuv`
/// (the crate's own decode — locally verified byte-exact against the
/// black-box reference decoder). No-op unless the env var is set.
#[test]
fn stage_round_452_fixtures_when_requested() {
    let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
        return;
    };
    for (name, ivf, yuv) in build_r452_fixture_streams() {
        let sub = std::path::Path::new(&dir).join(name);
        std::fs::create_dir_all(&sub).expect("create stage dir");
        std::fs::write(sub.join("input.ivf"), &ivf).expect("write input.ivf");
        std::fs::write(sub.join("expected.yuv"), &yuv).expect("write expected.yuv");
    }
}

/// Docs-gated identity: a staged round-452 package must be
/// byte-identical to the builder's output (the fixture IS this crate's
/// writer output). No-op while the corpus lacks the package.
#[test]
fn staged_r452_fixtures_match_builders() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    for (name, ivf, yuv) in build_r452_fixture_streams() {
        let sub = root.join(name);
        if !sub.join("input.ivf").is_file() {
            eprintln!("{name}: not staged yet; docs-gated");
            continue;
        }
        let staged = std::fs::read(sub.join("input.ivf")).expect("staged ivf");
        assert_eq!(staged, ivf, "{name}: staged bytes != builder output");
        let expected = std::fs::read(sub.join("expected.yuv")).expect("staged yuv");
        assert_eq!(expected, yuv, "{name}: staged expected.yuv != crate decode");
    }
}

/// Regression for the `encode_gop_structures` fuzz crash
/// (73f3c7e7…): fuzz bytes `fc fc fc 3a 0a f6 fc 26` derive a 33x20
/// frame on the resized arm, whose half-size twin the harness took as
/// `floor(33 / 2) = 16` — a 2x-PLUS downscale the §5 scaling bounds
/// forbid, so the encoder's `Unsupported` was the correct answer and
/// the harness input was invalid. Pins both halves: the floor twin
/// rejects, the `ceil` twin (the corrected derivation) encodes and
/// decodes at its declared sizes in both directions.
#[test]
fn resized_odd_geometry_half_twin_regression() {
    let (w, h) = (33u32, 20u32);
    let f0 = scene_frame(33, 20, 0);
    let floor_twin = scene_frame(16, 10, 1);
    assert_eq!(
        encode_vp9_lossy_sequence_resized(&[&f0, &floor_twin], &[(w, h), (16, 10)], 253)
            .unwrap_err(),
        Error::Unsupported,
        "2 * 16 < 33 is outside the §5 2x downscale bound"
    );
    let sizes = [(33u32, 20u32), (17, 10), (33, 20), (17, 10)];
    let src: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(k, &(fw, fh))| scene_frame(fw as usize, fh as usize, k))
        .collect();
    let packets = encode_vp9_lossy_sequence_resized(&refs(&src), &sizes, 253).expect("ceil twin");
    let decoded = decode_vp9_sequence(&refs(&packets)).expect("decode");
    assert_eq!(decoded.len(), 4);
    for (d, &(fw, fh)) in decoded.iter().zip(&sizes) {
        assert_eq!((d.width, d.height), (fw, fh));
    }
}
