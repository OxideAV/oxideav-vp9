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

    let mut bad = Vp9GopConfig::new(0);
    bad.segmentation = Vp9Segmentation::Full;
    assert_eq!(
        encode_vp9_lossy_sequence_with(&refs(&src), w, h, &bad).unwrap_err(),
        Error::Unsupported
    );
}
