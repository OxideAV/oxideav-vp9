//! Panic-surface fuzz for the deepest public decode entries — the
//! whole §6.1 → §6.2 → §6.3 → §6.4 → §8.5/§8.6/§8.7 → §8.8 → §8.10
//! pipeline reached through [`decode_vp9`], plus the Annex B
//! [`split_superframe`] index walker and the multi-frame
//! [`decode_vp9_sequence`] driver that threads the §8.10 reference
//! buffers / §6.1.2 entropy contexts / §6.5 motion field across frames.
//!
//! The `frame_header` / `compressed_header` targets stop at the header
//! walkers; this target carries arbitrary bytes all the way through the
//! tile + partition + per-block walk, the inverse transforms, intra /
//! inter prediction, and the loop filter, so a panic anywhere in the
//! reconstruction path (an out-of-bounds plane write, an arithmetic
//! overflow in a transform butterfly, a bad superblock geometry) is
//! surfaced rather than reached only by a full corpus replay.
//!
//! No oracle — arbitrary bytes in, "no panic / no overflow / no OOM"
//! out. `Err` (or an empty split) is a correct answer for garbage
//! input; every return value is intentionally discarded.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{decode_vp9, decode_vp9_sequence, split_superframe};

fuzz_target!(|data: &[u8]| {
    // 1. Single-frame decode of the chunk verbatim (the common
    //    "one VP9 frame per packet" carriage).
    let _ = decode_vp9(data);

    // 2. Annex B split, then both per-sub-frame single decode and the
    //    multi-frame sequence driver over the enclosed slices. The
    //    split never allocates beyond `data`'s own bytes (it returns
    //    borrows), so this stays within the same memory envelope.
    let subs = split_superframe(data);
    for sub in &subs {
        let _ = decode_vp9(sub);
    }
    let _ = decode_vp9_sequence(&subs);
});
