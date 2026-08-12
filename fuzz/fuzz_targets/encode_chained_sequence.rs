//! Round-trip fuzz for the **chained** lossless sequence encoder
//! `encode_vp9_lossless_sequence_chained` — the path that codes
//! non-error-resilient SHOWN P-frame chains under the §7.2.6
//! `UsePrevFrameMvs == 1` writer model (every P-frame's §6.5.10
//! candidate scan reads the previous frame's §6.4.4 motion field on
//! BOTH sides).
//!
//! The fuzzer derives a frame geometry (bounded to 1..=48 so the
//! per-frame motion search stays cheap), a frame count (2..=4), and
//! per-frame pixel buffers by cycling the remaining fuzz bytes from a
//! per-frame offset (so consecutive frames are shifted copies of the
//! same byte stream — arbitrary "motion" for the search to chase).
//!
//! The oracle is the full lossless guarantee, which is strictly
//! stronger than "must decode": every decoded frame MUST equal its
//! input byte-exact. Any divergence means the encoder's prev-field
//! model disagreed with the decoder's §7.2.6 scan (a predictor desync
//! corrupts the entropy stream), which is always a bug.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{decode_vp9_sequence, encode_vp9_lossless_sequence_chained};

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    // Bound dimensions so the ±8 px full search over 2..=4 frames stays
    // fast and memory-light.
    let w = 1 + (u32::from(data[0]) | (u32::from(data[1]) << 8)) % 48;
    let h = 1 + (u32::from(data[2]) | (u32::from(data[3]) << 8)) % 48;
    let n_frames = 2 + (data[4] as usize) % 3;
    let shift = 1 + (data[5] as usize) % 97;

    let cw = w.div_ceil(2) as usize;
    let ch = h.div_ceil(2) as usize;
    let need = (w as usize) * (h as usize) + 2 * cw * ch;
    let content = &data[6..];

    let frames: Vec<Vec<u8>> = (0..n_frames)
        .map(|i| {
            if content.is_empty() {
                vec![0u8; need]
            } else {
                content
                    .iter()
                    .copied()
                    .cycle()
                    .skip((i * shift) % content.len())
                    .take(need)
                    .collect()
            }
        })
        .collect();
    let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();

    let coded =
        encode_vp9_lossless_sequence_chained(&refs, w, h).expect("bounded valid input must encode");
    let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
    let decoded =
        decode_vp9_sequence(&coded_refs).expect("self-encoded chained sequence must decode");
    assert_eq!(decoded.len(), frames.len());
    for (i, d) in decoded.iter().enumerate() {
        assert_eq!(
            d.to_planar_bytes(),
            frames[i],
            "chained lossless frame {i} not byte-exact"
        );
    }
});
