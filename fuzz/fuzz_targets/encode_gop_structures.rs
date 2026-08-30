//! Oracle-carrying fuzz over the round-452 **structured GOP** write
//! paths: fuzz-derived `Vp9GopConfig` axes (alt-ref pyramid interval,
//! §6.2.11 segmentation mode, tile rows) over fuzz-derived translating
//! content, plus the §8.5.2.3 **resized-sequence** arm (mid-stream
//! coded-size changes within the §5 ratio bounds).
//!
//! The oracle is stream-the-encoder-emitted-MUST-decode, strengthened
//! structurally: the decoder must return exactly the source frame
//! count of shown frames (hidden alt-refs and `show_existing_frame`
//! packets resolve to the display order), at the declared per-frame
//! sizes on the resized arm. Any decode error or count/size mismatch
//! is an encoder-side §7.2.6 / §6.2.11 / §6.4.14 modeling bug by
//! construction.
//!
//! Geometry stays small (<= 48 px per axis; tile *columns* need >= 512
//! px frames, so the config axis fuzzed here is tile rows — the column
//! path is pinned by the in-crate 512-px tests) so the per-frame
//! motion search stays cheap.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence_resized, encode_vp9_lossy_sequence_with,
    Vp9GopConfig, Vp9Segmentation,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let w = 8 + (u32::from(data[0]) | (u32::from(data[1]) << 8)) % 41;
    let h = 8 + (u32::from(data[2]) | (u32::from(data[3]) << 8)) % 41;
    let n_frames = 2 + (data[4] as usize) % 3;
    let q = 1 + data[5] % 255;
    let sel = data[6];
    let shift = 1 + (data[7] as usize) % 97;
    let content = &data[8..];

    let frame = |fw: u32, fh: u32, k: usize| -> Vec<u8> {
        let cw = fw.div_ceil(2) as usize;
        let ch = fh.div_ceil(2) as usize;
        let need = (fw as usize) * (fh as usize) + 2 * cw * ch;
        if content.is_empty() {
            vec![128u8; need]
        } else {
            content
                .iter()
                .copied()
                .cycle()
                .skip((k * shift) % content.len())
                .take(need)
                .collect()
        }
    };

    if sel & 0x80 != 0 {
        // Resized arm: alternate between (w, h) and its half-size twin.
        // §5 bounds consecutive coded sizes to a 2x downscale, so the
        // twin is ceil(w / 2): floor(w / 2) on an odd w is a 2x-PLUS
        // downscale the encoder correctly rejects (CI crash
        // 73f3c7e7: w = 33 -> 16). The derivation MUST stay inside
        // the contract so `expect` only fires on a real defect.
        let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
        let sizes: Vec<(u32, u32)> = (0..n_frames)
            .map(|k| if k % 2 == 0 { (w, h) } else { (w2, h2) })
            .collect();
        let frames: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(k, &(fw, fh))| frame(fw, fh, k))
            .collect();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let packets = encode_vp9_lossy_sequence_resized(&refs, &sizes, q)
            .expect("bounded resized input must encode");
        let prefs: Vec<&[u8]> = packets.iter().map(|p| p.as_slice()).collect();
        let decoded = decode_vp9_sequence(&prefs).expect("self-encoded resized GOP must decode");
        assert_eq!(decoded.len(), n_frames);
        for (d, &(fw, fh)) in decoded.iter().zip(&sizes) {
            assert_eq!((d.width, d.height), (fw, fh), "declared size");
        }
        return;
    }

    let mut cfg = Vp9GopConfig::new(q);
    cfg.altref_interval = 1 + u32::from(sel & 3);
    cfg.segmentation = match (sel >> 2) & 3 {
        0 => Vp9Segmentation::Off,
        1 => Vp9Segmentation::AdaptiveQuant,
        2 => Vp9Segmentation::StaticSkip,
        _ => Vp9Segmentation::Full,
    };
    // Tile rows need sb64_rows >= 2 to matter; still legal at any size
    // (tile_rows_log2 <= 2), so fuzz the axis unconditionally.
    cfg.tile_rows_log2 = (sel >> 4) & 1;

    let frames: Vec<Vec<u8>> = (0..n_frames).map(|k| frame(w, h, k)).collect();
    let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
    let packets = encode_vp9_lossy_sequence_with(&refs, w, h, &cfg)
        .expect("bounded structured-GOP input must encode");
    let prefs: Vec<&[u8]> = packets.iter().map(|p| p.as_slice()).collect();
    let decoded = decode_vp9_sequence(&prefs).expect("self-encoded structured GOP must decode");
    assert_eq!(
        decoded.len(),
        n_frames,
        "shown-frame count (hidden alt-refs + show_existing must net out)"
    );
    for d in &decoded {
        assert_eq!((d.width, d.height), (w, h));
    }
});
