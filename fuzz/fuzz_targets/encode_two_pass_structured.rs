//! Oracle-carrying fuzz over the round-458 **structured two-pass**
//! write paths: `encode_vp9_lossy_sequence_rc_two_pass_with` (and its
//! 4:4:4 / 4:2:2 / 4:4:0 / 10-bit 4:2:0 wrappers) over fuzz-derived
//! `Vp9GopConfig` axes — alt-ref interval, §6.2.11 segmentation mode,
//! tile rows, intra-only alt-refs, scene-cut keyframe placement,
//! adaptive group length, coefficient election — with fuzz-derived
//! geometry, budget, VBV and content. Every trial quantisation runs
//! the round-458 coefficient election and the sub-8x8 / scaled-leaf
//! switchable-filter election lives on the same paths.
//!
//! The oracle is stream-the-encoder-emitted-MUST-decode, structurally
//! strengthened: the decoder returns exactly the source frame count at
//! the coded size and declared format; one report entry per packet;
//! the first packet is a keyframe; every display frame is coded by
//! exactly one non-`show_existing_frame` packet; every decoded packet
//! lands within its budget unless it sits at the `q == 255` syntax
//! floor; `show_existing_frame` packets are one byte (two on profile
//! 3). A desynchronised entropy mirror, a mis-planned packet sequence
//! or a mis-elected token corrupts the §9.2 bool decode of the next
//! frame and surfaces as a decode error by construction.
//!
//! Geometry stays small (<= 40 px per axis) so the two full passes —
//! the second bisecting every packet — stay cheap.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence_rc_two_pass_with,
    encode_vp9_lossy_sequence_rc_two_pass_with_422,
    encode_vp9_lossy_sequence_rc_two_pass_with_440,
    encode_vp9_lossy_sequence_rc_two_pass_with_444,
    encode_vp9_lossy_sequence_rc_two_pass_with_hbd, Vp9GopConfig, Vp9Segmentation,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let w = 8 + (u32::from(data[0]) | (u32::from(data[1]) << 8)) % 33;
    let h = 8 + (u32::from(data[2]) | (u32::from(data[3]) << 8)) % 33;
    let n_frames = 2 + (data[4] as usize) % 5;
    let q = 1 + data[5] % 255;
    let axes = data[6];
    let plan = data[7];
    let budget_byte = data[8];
    let fmt = data[9] % 5; // 420 / 444 / 422 / 440 / hbd10-420
    let shift = 1 + (data[10] as usize) % 97;
    let seed = data[11];
    let content = &data[12..];

    let mut cfg = Vp9GopConfig::new(q);
    cfg.altref_interval = 1 + u32::from(axes & 3);
    cfg.segmentation = match (axes >> 2) & 3 {
        0 => Vp9Segmentation::Off,
        1 => Vp9Segmentation::AdaptiveQuant,
        2 => Vp9Segmentation::StaticSkip,
        _ => Vp9Segmentation::Full,
    };
    cfg.tile_rows_log2 = (axes >> 4) & 1;
    cfg.intra_only_altref = axes & 0x20 != 0;
    cfg.entropy_adaptation = axes & 0x40 != 0 || plan & 0x80 != 0;
    cfg.switchable_interp_filter = axes & 0x80 != 0;
    cfg.scene_cut_keyframes = plan & 1 != 0;
    cfg.adaptive_group_length = plan & 2 != 0;
    cfg.coefficient_rdo = plan & 4 != 0;

    let (ssx, ssy, bit_depth) = match fmt {
        0 => (true, true, 8u32),
        1 => (false, false, 8),
        2 => (true, false, 8),
        3 => (false, true, 8),
        _ => (true, true, 10),
    };
    let cw = if ssx { w.div_ceil(2) } else { w } as usize;
    let ch = if ssy { h.div_ceil(2) } else { h } as usize;
    let (wu, hu) = (w as usize, h as usize);
    let sample = |x: usize, y: usize, k: usize, plane: usize| -> u8 {
        let i = (y * wu + x + k * shift + plane * 11 + usize::from(seed)) % content.len().max(1);
        let base = content.get(i).copied().unwrap_or(0);
        if plane == 0 {
            // A scene cut halfway through when the seed says so.
            let cut = seed & 1 != 0 && k >= n_frames / 2;
            base.wrapping_add(((x + 2 * k) * 3 + y * 5 + usize::from(cut) * 97) as u8)
        } else {
            base.wrapping_add((x + y + plane * 40) as u8)
        }
    };
    let frame8 = |k: usize| -> Vec<u8> {
        let mut px = Vec::with_capacity(wu * hu + 2 * cw * ch);
        for y in 0..hu {
            for x in 0..wu {
                px.push(sample(x, y, k, 0));
            }
        }
        for plane in 1..3usize {
            for y in 0..ch {
                for x in 0..cw {
                    px.push(sample(x, y, k, plane));
                }
            }
        }
        px
    };
    let raw = wu * hu + 2 * cw * ch;
    let target = 1 + (raw * (1 + usize::from(budget_byte & 15))) / 48;
    let vbv = if budget_byte & 16 == 0 {
        0
    } else {
        target * (1 + usize::from(budget_byte >> 5))
    };

    let (packets, report) = if bit_depth == 8 {
        let frames: Vec<Vec<u8>> = (0..n_frames).map(frame8).collect();
        let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
        match fmt {
            0 => encode_vp9_lossy_sequence_rc_two_pass_with(&refs, w, h, &cfg, target, vbv),
            1 => encode_vp9_lossy_sequence_rc_two_pass_with_444(&refs, w, h, &cfg, target, vbv),
            2 => encode_vp9_lossy_sequence_rc_two_pass_with_422(&refs, w, h, &cfg, target, vbv),
            _ => encode_vp9_lossy_sequence_rc_two_pass_with_440(&refs, w, h, &cfg, target, vbv),
        }
    } else {
        let frames: Vec<Vec<u16>> = (0..n_frames)
            .map(|k| frame8(k).into_iter().map(|v| u16::from(v) << 2).collect())
            .collect();
        let refs: Vec<&[u16]> = frames.iter().map(Vec::as_slice).collect();
        encode_vp9_lossy_sequence_rc_two_pass_with_hbd(&refs, w, h, 10, true, &cfg, target, vbv)
    }
    .expect("structured two-pass encodes");

    assert_eq!(report.len(), packets.len(), "one report entry per packet");
    assert!(report[0].keyframe && report[0].frame == 0, "first packet is the keyframe");
    let mut coded = vec![0u8; n_frames];
    for (p, r) in packets.iter().zip(&report) {
        assert_eq!(p.len(), r.coded_bytes);
        assert!(r.frame < n_frames);
        if r.show_existing {
            assert!(r.coded_bytes <= 2, "show_existing_frame packet size");
            assert_eq!((r.budget, r.base_q_idx), (0, 0));
        } else {
            coded[r.frame] += 1;
            assert!(r.coded_bytes <= r.budget || r.base_q_idx == 255, "budget respected");
        }
    }
    assert!(coded.iter().all(|&c| c == 1), "every display frame coded once: {coded:?}");
    let prefs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    let decoded = decode_vp9_sequence(&prefs).expect("structured two-pass stream must decode");
    assert_eq!(decoded.len(), n_frames, "shown frame count");
    for f in &decoded {
        assert_eq!((f.width, f.height), (w, h), "coded size");
        assert_eq!(
            (f.subsampling_x, f.subsampling_y, u32::from(f.bit_depth)),
            (ssx, ssy, bit_depth),
            "declared format"
        );
    }
});
