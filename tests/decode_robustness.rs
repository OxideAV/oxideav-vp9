//! Panic-surface regression pins for the public decode entries.
//!
//! The scheduled cargo-fuzz `decode_frame` target exercises the same
//! surface continuously, but these tests lock the garbage-in contract
//! (an error or empty result, never a panic / overflow / out-of-bounds)
//! into the standard CI run so a regression in the deep decode path (a
//! transform butterfly overflow, a superblock-geometry out-of-bounds
//! plane write, a bad Annex-B index length) is caught on every push
//! rather than only when the fuzz budget happens to rediscover it.
//!
//! Inputs are derived from the staged corpus where present (truncations
//! and single-byte flips reach deep into real bitstream structure) and
//! from a deterministic pseudo-random generator (so the run is
//! reproducible and bisectable). No oracle: a clean decode, an error
//! return, and an empty split are all acceptable — only an unwind is a
//! failure.

use oxideav_vp9::{decode_vp9, decode_vp9_sequence, split_superframe};
use std::path::Path;

/// All multi-frame fixtures whose leading IVF frame is a single VP9
/// chunk we can slice out without an IVF demuxer dependency.
const FIXTURES: &[&str] = &[
    "profile-0-yuv420-8bit",
    "segments-aq-mode",
    "superframe-2",
    "i-frame-then-p-frame-64x64",
    "frame-parallel-mode",
    "show-existing-frame",
    "tile-cols-2",
    "profile-2-yuv420-10bit",
    "profile-3-yuv444-12bit",
];

/// Read the first VP9 chunk out of a fixture's `input.ivf`, or `None`
/// when the docs corpus is not checked out (standalone CI).
fn first_chunk(fixture: &str) -> Option<Vec<u8>> {
    let base = Path::new("../../docs/video/vp9/fixtures").join(fixture);
    if !base.is_dir() {
        return None;
    }
    let ivf = std::fs::read(base.join("input.ivf")).ok()?;
    if ivf.len() < 44 {
        return None;
    }
    let size = u32::from_le_bytes([ivf[32], ivf[33], ivf[34], ivf[35]]) as usize;
    if 44 + size > ivf.len() {
        return None;
    }
    Some(ivf[44..44 + size].to_vec())
}

/// Run every public decode entry over `bytes` inside a catch-unwind and
/// report (do not abort on) any panic, so a single test surfaces all of
/// them.
fn exercise(bytes: &[u8], tag: &str, panics: &mut Vec<String>) {
    let b = bytes.to_vec();
    if std::panic::catch_unwind(move || {
        let _ = decode_vp9(&b);
    })
    .is_err()
    {
        panics.push(format!("decode_vp9 {tag}"));
    }
    let b = bytes.to_vec();
    if std::panic::catch_unwind(move || {
        let subs = split_superframe(&b);
        for sub in &subs {
            let _ = decode_vp9(sub);
        }
        let _ = decode_vp9_sequence(&subs);
    })
    .is_err()
    {
        panics.push(format!("split+sequence {tag}"));
    }
}

#[test]
fn truncation_and_corruption_never_panic() {
    // Silence the default panic hook so the catch-unwind probes don't
    // spew backtraces for the expected internal panics they catch.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut panics = Vec::new();

    let mut any_fixture = false;
    for fx in FIXTURES {
        let Some(frame) = first_chunk(fx) else {
            continue;
        };
        any_fixture = true;

        // Truncation lengths on a coarse stride (still hits every
        // header / tile / superblock boundary class) plus a dense sweep
        // of the first 64 bytes where the header guards live.
        for n in (0..frame.len().min(64)).chain((0..=frame.len()).step_by(17)) {
            exercise(&frame[..n], &format!("{fx} trunc {n}"), &mut panics);
        }

        // Single-byte flips at a spread of offsets, three bit patterns
        // each (LSB, MSB, all bits) so both the header fields and the
        // arithmetic-coded payload get perturbed.
        let mut off = 0;
        while off < frame.len() {
            for bit in [0x01u8, 0x80, 0xFF] {
                let mut f = frame.clone();
                f[off] ^= bit;
                exercise(&f, &format!("{fx} flip {off}/{bit:#x}"), &mut panics);
            }
            off += 29;
        }
    }

    std::panic::set_hook(prev);

    if !any_fixture {
        eprintln!("docs corpus not present; docs-gated robustness probe skipped");
        return;
    }
    assert!(panics.is_empty(), "decode path panicked on: {panics:?}");
}

#[test]
fn pseudo_random_input_never_panics() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut panics = Vec::new();

    // Deterministic linear-congruential generator for reproducibility.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    // Optionally graft a real chunk prefix so the random tail reaches
    // past the header into the arithmetic-coded body.
    let seed = first_chunk("superframe-2");

    for iter in 0..2000u32 {
        let len = (next() % 512) as usize;
        let mut buf: Vec<u8> = Vec::with_capacity(len + 32);
        if let Some(seed) = &seed {
            if next() & 1 == 0 {
                let take = (next() as usize) % (seed.len() + 1);
                buf.extend_from_slice(&seed[..take]);
            }
        }
        while buf.len() < len {
            buf.push((next() & 0xFF) as u8);
        }
        // Sometimes append a plausible Annex-B superframe-index trailer
        // (marker `0b110` in the top three bits, then the size bytes,
        // then the marker again) to drive the split walker.
        if next() & 3 == 0 {
            let cnt = (next() % 8) as u8 & 0x07;
            let bps = (next() % 4) as u8 & 0x03;
            let marker = 0xC0u8 | (bps << 3) | cnt;
            buf.push(marker);
            for _ in 0..((bps as usize + 1) * (cnt as usize + 1)) {
                buf.push((next() & 0xFF) as u8);
            }
            buf.push(marker);
        }
        exercise(&buf, &format!("rand {iter}"), &mut panics);
    }

    std::panic::set_hook(prev);
    assert!(panics.is_empty(), "decode path panicked on: {panics:?}");
}

/// A handful of hand-built degenerate inputs that target specific
/// early-return guards, pinned so a future refactor can't reintroduce
/// an unguarded slice / divide.
#[test]
fn degenerate_inputs_are_rejected_cleanly() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let cases: &[&[u8]] = &[
        &[],
        &[0x00],
        &[0xFF; 1],
        &[0x82, 0x49, 0x83, 0x42],
        &[0xC0],             // bare superframe marker, no payload
        &[0xC0, 0x00, 0xC0], // marker / size / marker, zero-size frame
        &[0xFF; 64],
        &[0x00; 64],
    ];
    let mut panics = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        exercise(c, &format!("degenerate[{i}]"), &mut panics);
    }

    std::panic::set_hook(prev);
    assert!(panics.is_empty(), "degenerate input panicked: {panics:?}");
}

/// Fuzz-found OOM regression (scheduled-fuzz run 28736285692, target
/// `decode_frame`): a syntactically valid §6.2 header claiming a huge
/// `frame_size` drove a multi-GiB `CurrFrame` plane allocation before
/// any tile data was validated. The decoder now bounds the MI-padded
/// luma picture size by the largest `Max luma picture size` any VP9
/// level defines (35 651 584, levels 6.x) and rejects such frames as
/// `Unsupported` *before* allocating. Pinned with the original crash
/// input plus a hand-built maximal-dimensions header prefix.
#[test]
fn oversized_frame_dimensions_are_rejected_before_allocation() {
    // The minimised fuzz artifact (41 bytes) — decode must return an
    // error without attempting the frame-geometry-sized allocations.
    let crash: &[u8] = &[
        0x83, 0x49, 0x83, 0x42, 0x83, 0xff, 0xff, 0x7a, 0xbb, 0x27, 0xff, 0x29, 0xff, 0x36, 0x24,
        0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0x00, 0x00, 0x01, 0x00, 0x20, 0x00, 0x00, 0x1e, 0xff, 0x37,
        0x9c, 0xef, 0xff, 0xdf, 0xb2, 0x4b, 0x02, 0x49, 0x83, 0x42, 0x83,
    ];
    assert!(oxideav_vp9::decode_vp9(crash).is_err());
    let subs = oxideav_vp9::split_superframe(crash);
    for sub in &subs {
        assert!(oxideav_vp9::decode_vp9(sub).is_err());
    }
    assert!(oxideav_vp9::decode_vp9_sequence(&subs).is_err());
}
