//! Panic-surface fuzz for the keyframe **encoder** entry `encode_vp9` and
//! its decode round-trip.
//!
//! The fuzzer derives a frame `(width, height)` from the first four bytes
//! (bounded to a small range so the harness stays fast and memory-light),
//! builds a flat 4:2:0 pixel buffer of the matching length, then runs
//! `encode_vp9` and — on success — feeds the produced stream straight back
//! through `decode_vp9`. A conforming encoder must never panic, and every
//! frame it emits must decode without panicking; an `Err` from either side
//! is a correct answer for a rejected geometry. Both return values are
//! intentionally discarded — the contract is "no panic / no overflow / no
//! OOM", not an output oracle.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{decode_vp9, encode_vp9};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    // Bound dimensions to 1..=256 so the buffer + bool encoder stay small.
    let w = 1 + (u32::from(data[0]) | (u32::from(data[1]) << 8)) % 256;
    let h = 1 + (u32::from(data[2]) | (u32::from(data[3]) << 8)) % 256;

    let cw = w.div_ceil(2) as usize;
    let ch = h.div_ceil(2) as usize;
    let need = (w as usize) * (h as usize) + 2 * cw * ch;
    // Reuse the fuzz bytes (cycled) as the flat pixel fill — content is
    // ignored by the all-skip encoder, but this keeps allocation honest.
    let fill = data.get(4).copied().unwrap_or(0);
    let pixels = vec![fill; need];

    if let Ok(stream) = encode_vp9(&pixels, w, h) {
        let _ = decode_vp9(&stream);
    }
});
