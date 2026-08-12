//! Round-trip fuzz for the round-441 **lossy format matrix** keyframe
//! entries — `encode_vp9_lossy_444` / `encode_vp9_lossy_422` (profile
//! 1) and `encode_vp9_lossy_hbd` / `encode_vp9_lossy_hbd_422`
//! (profiles 2/3) — the paths that generalise the lossy pipeline over
//! the §7.2.2 `(BitDepth, subsampling_x, subsampling_y)` triple.
//!
//! The fuzzer derives a format selector, frame `(width, height)`
//! (bounded to 1..=64 so the §8.8 election probes stay cheap), and
//! `base_q_idx` from the leading bytes, then fills the planar buffer
//! by cycling the remaining fuzz bytes (masked into the bit-depth
//! range for the high-bit-depth formats).
//!
//! Oracle-carrying: a stream the encoder *did* emit MUST decode —
//! `decode_vp9` runs this crate's full §8.1 chain (elected §8.8
//! filter and skip-elected leaves included) on the self-encoded
//! frame. Any `Err` there means the encoder wrote a stream its own
//! decoder rejects, which is always a bug.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{
    decode_vp9, encode_vp9_lossy, encode_vp9_lossy_422, encode_vp9_lossy_444, encode_vp9_lossy_hbd,
    encode_vp9_lossy_hbd_422,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    // (bit_depth, ssx, ssy) across every format the public lossy
    // entry points reach (4:4:0 has no dedicated public entry).
    let formats: [(u8, bool, bool); 7] = [
        (8, true, true),
        (8, false, false),
        (8, true, false),
        (10, true, true),
        (12, true, true),
        (10, false, false),
        (12, true, false),
    ];
    let (bd, ssx, ssy) = formats[usize::from(data[0]) % formats.len()];
    let w = 1 + (u32::from(data[1]) | (u32::from(data[2]) << 8)) % 64;
    let h = 1 + (u32::from(data[3]) | (u32::from(data[4]) << 8)) % 64;
    let q = data[5]; // 0 is the rejected lossless qindex — also fuzzed.

    let cw = if ssx { w.div_ceil(2) } else { w } as usize;
    let ch = if ssy { h.div_ceil(2) } else { h } as usize;
    let need = (w as usize) * (h as usize) + 2 * cw * ch;
    let content = &data[6..];

    let stream = if bd == 8 {
        let pixels: Vec<u8> = if content.is_empty() {
            vec![0u8; need]
        } else {
            content.iter().copied().cycle().take(need).collect()
        };
        match (ssx, ssy) {
            (true, true) => encode_vp9_lossy(&pixels, w, h, q),
            (false, false) => encode_vp9_lossy_444(&pixels, w, h, q),
            (true, false) => encode_vp9_lossy_422(&pixels, w, h, q),
            (false, true) => unreachable!("not in the format table"),
        }
    } else {
        let mask = (1u16 << bd) - 1;
        let samples: Vec<u16> = if content.is_empty() {
            vec![0u16; need]
        } else {
            content
                .chunks(2)
                .map(|c| {
                    let lo = u16::from(c[0]);
                    let hi = u16::from(*c.get(1).unwrap_or(&0));
                    (lo | (hi << 8)) & mask
                })
                .cycle()
                .take(need)
                .collect()
        };
        match (ssx, ssy) {
            (true, true) | (false, false) => {
                encode_vp9_lossy_hbd(&samples, w, h, bd, ssx && ssy, q)
            }
            (true, false) => encode_vp9_lossy_hbd_422(&samples, w, h, bd, q),
            (false, true) => unreachable!("not in the format table"),
        }
    };

    if let Ok(bytes) = stream {
        decode_vp9(&bytes).expect("self-encoded lossy matrix keyframe must decode");
    }
});
