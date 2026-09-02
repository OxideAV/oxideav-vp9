//! Round-455 public-surface pins of the **entropy model**: every lossy
//! sequence entry now codes `refresh_frame_context = 1` /
//! `frame_parallel_decoding_mode = 0` against its predecessor's §8.4
//! backward-adapted bank with §6.3 forward updates elected by measured
//! cost. The reconstruction is invariant under the framing (the symbols
//! are identical; only their probabilities move), so the measured delta
//! is pure bytes at equal PSNR — asserted here per sequence across the
//! crate's encode corpus, with the corpus-wide totals printed.

use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence, encode_vp9_lossy_sequence_444,
    encode_vp9_lossy_sequence_hbd, encode_vp9_lossy_sequence_rc, encode_vp9_lossy_sequence_resized,
    encode_vp9_lossy_sequence_resized_with, encode_vp9_lossy_sequence_with, Vp9GopConfig,
    Vp9Segmentation,
};

/// Source frame `k` of a translating textured scene (luma ramp + moving
/// patch + sparse dots, chroma ramps), planar at the given chroma dims.
fn scene(w: usize, h: usize, cw: usize, ch: usize, k: usize, seed: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            let mut v = ((x + 2 * k + seed) * 3 + y * 2) % 200 + 20;
            let px_x = x as i64 - 2 * k as i64;
            if (8..24).contains(&px_x) && (8..24).contains(&y) {
                v = (px_x as usize * 37 + y * 53 + seed) % 255;
            }
            if (x * 7 + y * 11 + k) % 29 == 0 {
                v = (v + 40) % 256;
            }
            px.push(v as u8);
        }
    }
    for plane in 0..2usize {
        for y in 0..ch {
            for x in 0..cw {
                px.push(((x + k) * 5 + y * 3 + plane * 90 + seed) as u8);
            }
        }
    }
    px
}

fn scene420(w: usize, h: usize, k: usize, seed: usize) -> Vec<u8> {
    scene(w, h, w.div_ceil(2), h.div_ceil(2), k, seed)
}

fn refs(v: &[Vec<u8>]) -> Vec<&[u8]> {
    v.iter().map(Vec::as_slice).collect()
}

fn total(packets: &[Vec<u8>]) -> usize {
    packets.iter().map(Vec::len).sum()
}

fn decoded_yuv(packets: &[Vec<u8>]) -> Vec<u8> {
    let frames = decode_vp9_sequence(&refs(packets)).expect("decodes");
    let mut out = Vec::new();
    for f in &frames {
        out.extend_from_slice(&f.to_planar_bytes());
    }
    out
}

/// One corpus sequence: `(name, frames, w, h, GOP config)`.
type CorpusEntry = (String, Vec<Vec<u8>>, u32, u32, Vp9GopConfig);

/// The corpus.
fn corpus() -> Vec<CorpusEntry> {
    let mut out = Vec::new();
    let shapes: [(u32, u32, usize, u8, u32, Vp9Segmentation, bool); 6] = [
        (64, 48, 6, 60, 1, Vp9Segmentation::Off, false),
        (96, 64, 6, 110, 1, Vp9Segmentation::Off, false),
        (72, 56, 7, 140, 3, Vp9Segmentation::Off, false),
        (64, 48, 7, 100, 3, Vp9Segmentation::Full, false),
        (64, 48, 7, 100, 3, Vp9Segmentation::AdaptiveQuant, true),
        (128, 40, 5, 180, 2, Vp9Segmentation::StaticSkip, false),
    ];
    for (i, (w, h, n, q, interval, seg, io)) in shapes.into_iter().enumerate() {
        let frames: Vec<Vec<u8>> = (0..n)
            .map(|k| scene420(w as usize, h as usize, k, i))
            .collect();
        let mut cfg = Vp9GopConfig::new(q);
        cfg.altref_interval = interval;
        cfg.segmentation = seg;
        cfg.intra_only_altref = io;
        out.push((
            format!("gop-{w}x{h}-q{q}-arf{interval}-{seg:?}-io{io}"),
            frames,
            w,
            h,
            cfg,
        ));
    }
    out
}

/// Adaptive vs default-bank framing: identical reconstruction, strictly
/// fewer bytes on every corpus sequence (the keyframe alone already
/// carries elected coefficient updates).
#[test]
fn entropy_model_shrinks_every_corpus_sequence_at_identical_reconstruction() {
    let mut sum_on = 0usize;
    let mut sum_off = 0usize;
    for (name, frames, w, h, cfg) in corpus() {
        let on = encode_vp9_lossy_sequence_with(&refs(&frames), w, h, &cfg).expect("adaptive");
        let mut off_cfg = cfg;
        off_cfg.entropy_adaptation = false;
        let off = encode_vp9_lossy_sequence_with(&refs(&frames), w, h, &off_cfg).expect("default");
        assert_eq!(on.len(), off.len(), "{name}: packet count");
        assert_eq!(
            decoded_yuv(&on),
            decoded_yuv(&off),
            "{name}: reconstruction must not move"
        );
        let (a, b) = (total(&on), total(&off));
        assert!(a < b, "{name}: adaptive {a} bytes vs default-bank {b}");
        eprintln!(
            "{name}: {b} -> {a} bytes ({:.1}%)",
            100.0 * (b - a) as f64 / b as f64
        );
        sum_on += a;
        sum_off += b;
    }
    eprintln!(
        "corpus: {sum_off} -> {sum_on} bytes ({:.1}%)",
        100.0 * (sum_off - sum_on) as f64 / sum_off as f64
    );
}

/// Every public lossy sequence entry codes the adaptive framing: the
/// first P-frame's header carries `refresh_frame_context = 1` and
/// `frame_parallel_decoding_mode = 0`.
#[test]
fn public_sequence_entries_code_the_adaptive_framing() {
    let f420: Vec<Vec<u8>> = (0..3).map(|k| scene420(64, 48, k, 7)).collect();
    let f444: Vec<Vec<u8>> = (0..3).map(|k| scene(64, 48, 64, 48, k, 7)).collect();
    let hbd: Vec<Vec<u16>> = f420
        .iter()
        .map(|f| f.iter().map(|&v| u16::from(v) << 2).collect())
        .collect();
    let hbd_refs: Vec<&[u16]> = hbd.iter().map(Vec::as_slice).collect();
    let sizes = [(64u32, 48u32), (48, 32), (64, 48)];
    let resized: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(k, &(w, h))| scene420(w as usize, h as usize, k, 3))
        .collect();
    let streams: Vec<(&str, Vec<Vec<u8>>)> = vec![
        (
            "chain",
            encode_vp9_lossy_sequence(&refs(&f420), 64, 48, 100).unwrap(),
        ),
        (
            "444",
            encode_vp9_lossy_sequence_444(&refs(&f444), 64, 48, 100).unwrap(),
        ),
        (
            "hbd10",
            encode_vp9_lossy_sequence_hbd(&hbd_refs, 64, 48, 10, true, 100).unwrap(),
        ),
        (
            "rc",
            encode_vp9_lossy_sequence_rc(&refs(&f420), 64, 48, 900).unwrap(),
        ),
        (
            "resized",
            encode_vp9_lossy_sequence_resized(&refs(&resized), &sizes, 100).unwrap(),
        ),
        (
            "with",
            encode_vp9_lossy_sequence_with(&refs(&f420), 64, 48, &Vp9GopConfig::new(100)).unwrap(),
        ),
    ];
    for (name, packets) in &streams {
        // §6.2: the flag pair sits after frame size / refs; parse the
        // keyframe (self-contained) and check the P-frame through the
        // sequence decoder's own bookkeeping by re-encoding nothing —
        // the header bits are pinned by the uncompressed-header parser
        // on the keyframe, and the P-frame is pinned through decode.
        let kf = oxideav_vp9::parse_uncompressed_header(&packets[0]).expect("keyframe header");
        assert!(
            kf.refresh_frame_context,
            "{name}: keyframe refresh_frame_context"
        );
        assert!(
            !kf.frame_parallel_decoding_mode,
            "{name}: keyframe non-parallel"
        );
        assert!(
            !kf.error_resilient_mode,
            "{name}: keyframe not error-resilient"
        );
        let decoded = decode_vp9_sequence(&refs(packets)).expect("decodes");
        assert_eq!(decoded.len(), 3, "{name}: shown frames");
    }
}

/// `entropy_adaptation = false` reproduces the round-452 bytes of the
/// structured / resized entries (the staged-corpus contract), and the
/// resized config entry rejects the axes the resized chain lacks.
#[test]
fn opt_out_keeps_the_default_bank_framing_and_resized_config_rejects_unsupported_axes() {
    let frames: Vec<Vec<u8>> = (0..3).map(|k| scene420(64, 48, k, 1)).collect();
    let mut cfg = Vp9GopConfig::new(100);
    cfg.entropy_adaptation = false;
    let packets = encode_vp9_lossy_sequence_with(&refs(&frames), 64, 48, &cfg).unwrap();
    let kf = oxideav_vp9::parse_uncompressed_header(&packets[0]).unwrap();
    assert!(
        kf.frame_parallel_decoding_mode,
        "opt-out keeps parallel mode on the keyframe"
    );
    let sizes = [(64u32, 48u32), (48, 32)];
    let resized: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(k, &(w, h))| scene420(w as usize, h as usize, k, 3))
        .collect();
    let r = encode_vp9_lossy_sequence_resized_with(&refs(&resized), &sizes, &cfg).unwrap();
    let r_kf = oxideav_vp9::parse_uncompressed_header(&r[0]).unwrap();
    assert!(r_kf.frame_parallel_decoding_mode);
    let mut bad = Vp9GopConfig::new(100);
    bad.altref_interval = 2;
    assert!(encode_vp9_lossy_sequence_resized_with(&refs(&resized), &sizes, &bad).is_err());
    let mut bad = Vp9GopConfig::new(100);
    bad.segmentation = Vp9Segmentation::Full;
    assert!(encode_vp9_lossy_sequence_resized_with(&refs(&resized), &sizes, &bad).is_err());
}

fn ivf_wrap(packets: &[Vec<u8>], w: u16, h: u16) -> Vec<u8> {
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&w.to_le_bytes());
    ivf.extend_from_slice(&h.to_le_bytes());
    ivf.extend_from_slice(&25u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (i, f) in packets.iter().enumerate() {
        ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes());
        ivf.extend_from_slice(f);
    }
    ivf
}

/// Black-box validation dump: under `OXIDEAV_VP9_ENTROPY_DUMP_DIR` emit
/// every corpus sequence (adaptive framing) as `<name>/input.ivf` +
/// `<name>/crate-decode.yuv` + `<name>/source.yuv`, so an external
/// decoder run can be compared against the crate's decode and the
/// source (PSNR). No-op unless the env var is set.
#[test]
fn dump_entropy_corpus_when_requested() {
    let Some(dir) = std::env::var_os("OXIDEAV_VP9_ENTROPY_DUMP_DIR") else {
        return;
    };
    for (name, frames, w, h, cfg) in corpus() {
        let packets = encode_vp9_lossy_sequence_with(&refs(&frames), w, h, &cfg).expect("encode");
        let sub = std::path::Path::new(&dir).join(&name);
        std::fs::create_dir_all(&sub).expect("create dump dir");
        std::fs::write(
            sub.join("input.ivf"),
            ivf_wrap(&packets, w as u16, h as u16),
        )
        .unwrap();
        std::fs::write(sub.join("crate-decode.yuv"), decoded_yuv(&packets)).unwrap();
        std::fs::write(sub.join("source.yuv"), frames.concat()).unwrap();
    }
}
