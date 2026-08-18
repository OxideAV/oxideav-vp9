//! Differential fuzz for the round-448 **tile-parallel** decode path:
//! arbitrary bytes are decoded twice — once through the serial
//! [`decode_vp9_sequence`] walk and once through
//! [`decode_vp9_sequence_with`] under a 4-thread
//! [`ExecutionContext`] — and the two outcomes MUST agree exactly.
//!
//! The oracle is total, malformed input included:
//!
//! * both `Ok` — every shown frame's packed planar bytes must be
//!   identical (the §6.4 tile columns are independent, so the
//!   per-column merge must reproduce the serial raster bit-for-bit);
//! * both `Err` — the error values must be equal (tiles are
//!   independent, so the set of failing tiles is schedule-invariant
//!   and the parallel path surfaces the lowest raster-order tile
//!   error — exactly the one the serial walk hits first);
//! * a split outcome (`Ok` vs `Err`) is a bug by construction.
//!
//! Single-tile-column inputs exercise the `effective_workers` clamp
//! (the plain serial path); inputs whose §6.2.13 headers claim
//! `tile_cols_log2 > 0` drive the real fan-out, worker chunking, and
//! the column-range merges.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::ExecutionContext;
use oxideav_vp9::{decode_vp9_sequence, decode_vp9_sequence_with, split_superframe};

fuzz_target!(|data: &[u8]| {
    let subs = split_superframe(data);
    let serial = decode_vp9_sequence(&subs);
    let parallel = decode_vp9_sequence_with(&subs, &ExecutionContext::with_threads(4));
    match (serial, parallel) {
        (Ok(s), Ok(p)) => {
            assert_eq!(
                s.len(),
                p.len(),
                "tile-parallel decode diverged from serial: shown-frame count"
            );
            for (i, (a, b)) in s.iter().zip(&p).enumerate() {
                assert!(
                    a.to_planar_bytes() == b.to_planar_bytes(),
                    "tile-parallel decode diverged from serial at frame {i}"
                );
            }
        }
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "tile-parallel decode surfaced a different error");
        }
        (s, p) => panic!(
            "tile-parallel decode outcome diverged: serial ok={} parallel ok={}",
            s.is_ok(),
            p.is_ok()
        ),
    }
});
