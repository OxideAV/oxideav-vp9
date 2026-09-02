//! Oracle-carrying fuzz over the round-455 **adaptive chain** write
//! paths: the default lossy chain (`encode_vp9_lossy_sequence` —
//! §8.4 backward adaptation mirrored encoder-side, cost-elected §6.3
//! forward updates, per-leaf switchable interpolation filter) and the
//! two-pass rate-controlled chain (`encode_vp9_lossy_sequence_rc_two_pass`
//! — first-pass statistics, VBV-modeled allocation, per-frame
//! bisection), over fuzz-derived geometry, quantizer / budget, and
//! translating content.
//!
//! The oracle is stream-the-encoder-emitted-MUST-decode, structurally
//! strengthened: the decoder returns exactly the source frame count at
//! the coded size, and on the two-pass arm every frame lands within
//! its reported budget unless it sits at the `q == 255` syntax floor.
//! A desynchronised entropy mirror (writer counts / banks diverging
//! from the decoder's) corrupts the §9.2 bool decode of the next frame
//! and surfaces as a decode error by construction.
//!
//! Geometry stays small (<= 48 px per axis) so the per-frame motion
//! search stays cheap.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence, encode_vp9_lossy_sequence_rc_two_pass,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let w = 8 + (u32::from(data[0]) | (u32::from(data[1]) << 8)) % 41;
    let h = 8 + (u32::from(data[2]) | (u32::from(data[3]) << 8)) % 41;
    let n_frames = 2 + (data[4] as usize) % 3;
    let q = 1 + data[5] % 255;
    let arm = data[6];
    let shift = 1 + (data[7] as usize) % 97;
    let content = &data[8..];

    let cw = w.div_ceil(2) as usize;
    let ch = h.div_ceil(2) as usize;
    let frame = |k: usize| -> Vec<u8> {
        let (wu, hu) = (w as usize, h as usize);
        let mut px = Vec::with_capacity(wu * hu + 2 * cw * ch);
        for y in 0..hu {
            for x in 0..wu {
                let i = (y * wu + x + k * shift) % content.len().max(1);
                let base = content.get(i).copied().unwrap_or(0);
                px.push(base.wrapping_add(((x + 2 * k) * 3 + y * 5) as u8));
            }
        }
        for plane in 0..2usize {
            for y in 0..ch {
                for x in 0..cw {
                    let i = (y * cw + x + plane * 7 + k) % content.len().max(1);
                    px.push(content.get(i).copied().unwrap_or(0).wrapping_add((x + y) as u8));
                }
            }
        }
        px
    };
    let frames: Vec<Vec<u8>> = (0..n_frames).map(frame).collect();
    let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();

    if arm & 1 == 0 {
        let packets = encode_vp9_lossy_sequence(&refs, w, h, q).expect("adaptive chain encodes");
        let prefs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        let decoded = decode_vp9_sequence(&prefs).expect("adaptive chain stream must decode");
        assert_eq!(decoded.len(), n_frames, "shown frame count");
        for f in &decoded {
            assert_eq!((f.width, f.height), (w, h), "coded size");
        }
    } else {
        // Budget: a fuzz-derived fraction of the frame's raw size, never
        // zero (the syntax floor is a best-effort q == 255).
        let raw = (w as usize) * (h as usize) + 2 * cw * ch;
        let target = 1 + (raw * (1 + usize::from(arm >> 1))) / 64;
        let vbv = if arm & 2 == 0 { 0 } else { target * 3 };
        let (packets, report) = encode_vp9_lossy_sequence_rc_two_pass(&refs, w, h, target, vbv)
            .expect("two-pass chain encodes");
        assert_eq!(report.len(), n_frames);
        for (p, r) in packets.iter().zip(&report) {
            assert_eq!(p.len(), r.coded_bytes);
            assert!(r.coded_bytes <= r.budget || r.base_q_idx == 255, "budget respected");
        }
        let prefs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        let decoded = decode_vp9_sequence(&prefs).expect("two-pass stream must decode");
        assert_eq!(decoded.len(), n_frames, "shown frame count");
    }
});
