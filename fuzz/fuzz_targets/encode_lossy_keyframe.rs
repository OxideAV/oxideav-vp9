//! Round-trip fuzz for the **lossy** keyframe encoder entry
//! `encode_vp9_lossy` — the path that runs the encode-side §8.8 loop
//! filter with the two-stage `(loop_filter_level, loop_filter_sharpness)`
//! election on every successful encode.
//!
//! The fuzzer derives a frame `(width, height)` and `base_q_idx` from
//! the leading bytes (dimensions bounded to 1..=96 so the 64 + 7
//! full-frame election probes stay cheap), fills the 4:2:0 pixel
//! buffer by cycling the remaining fuzz bytes (arbitrary content
//! drives arbitrary partition/tx/mode plans, reconstructions, and
//! elected filter parameters), then round-trips.
//!
//! Unlike the pure panic-surface targets, this one carries an oracle:
//! a stream the encoder *did* emit MUST decode — `decode_vp9` running
//! this crate's full §8.1 chain, §8.8 filter included, on the
//! self-encoded frame. Any `Err` there means the encoder wrote a
//! stream its own decoder rejects, which is always a bug.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{decode_vp9, encode_vp9_lossy};

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    // Bound dimensions so the election's 71 full-frame §8.8 probes and
    // the planner stay fast and memory-light.
    let w = 1 + (u32::from(data[0]) | (u32::from(data[1]) << 8)) % 96;
    let h = 1 + (u32::from(data[2]) | (u32::from(data[3]) << 8)) % 96;
    let q = data[4]; // 0 is the rejected lossless qindex — also fuzzed.

    let cw = w.div_ceil(2) as usize;
    let ch = h.div_ceil(2) as usize;
    let need = (w as usize) * (h as usize) + 2 * cw * ch;
    let content = &data[5..];
    let pixels: Vec<u8> = if content.is_empty() {
        vec![0u8; need]
    } else {
        content.iter().copied().cycle().take(need).collect()
    };

    if let Ok(stream) = encode_vp9_lossy(&pixels, w, h, q) {
        decode_vp9(&stream).expect("self-encoded lossy keyframe must decode");
    }
});
