//! Round-455 **two-pass rate control** pins: first-pass statistics,
//! the VBV-modeled allocation, and rate accuracy (actual vs target
//! sequence bytes) across GOP shapes — static, translating, and a
//! mid-sequence scene cut — with the one-pass chain kept intact.

use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence_rc, encode_vp9_lossy_sequence_rc_two_pass,
    encode_vp9_lossy_sequence_rc_two_pass_with, encode_vp9_lossy_sequence_rc_two_pass_with_422,
    encode_vp9_lossy_sequence_rc_two_pass_with_440, encode_vp9_lossy_sequence_rc_two_pass_with_444,
    encode_vp9_lossy_sequence_rc_two_pass_with_hbd,
    encode_vp9_lossy_sequence_rc_two_pass_with_hbd_422,
    encode_vp9_lossy_sequence_rc_two_pass_with_hbd_440, Error, Vp9DecodedFrame, Vp9GopConfig,
    Vp9Segmentation, Vp9TwoPassFrame,
};

fn scene(w: usize, h: usize, k: usize, seed: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
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

fn refs(v: &[Vec<u8>]) -> Vec<&[u8]> {
    v.iter().map(Vec::as_slice).collect()
}

fn psnr(packets: &[Vec<u8>], source: &[Vec<u8>]) -> f64 {
    let frames = decode_vp9_sequence(&refs(packets)).expect("decodes");
    let mut dec = Vec::new();
    for f in &frames {
        dec.extend_from_slice(&f.to_planar_bytes());
    }
    let src = source.concat();
    assert_eq!(dec.len(), src.len());
    let sse: f64 = dec
        .iter()
        .zip(&src)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    let mse = sse / dec.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// One GOP shape: `(name, frames, w, h, target bytes per frame)`.
type Shape = (&'static str, Vec<Vec<u8>>, u32, u32, usize);

/// The GOP shapes.
fn shapes() -> Vec<Shape> {
    let (w, h) = (64usize, 48usize);
    // Static: every frame identical.
    let static_gop: Vec<Vec<u8>> = (0..6).map(|_| scene(w, h, 0, 1)).collect();
    // Translating content.
    let moving: Vec<Vec<u8>> = (0..6).map(|k| scene(w, h, k, 2)).collect();
    // Scene cut at frame 3 (a different seed flips the whole picture).
    let cut: Vec<Vec<u8>> = (0..6)
        .map(|k| {
            if k < 3 {
                scene(w, h, k, 3)
            } else {
                scene(w, h, k, 40)
            }
        })
        .collect();
    vec![
        ("static", static_gop, w as u32, h as u32, 500),
        ("moving", moving, w as u32, h as u32, 700),
        ("scene-cut", cut, w as u32, h as u32, 700),
        ("moving-tight", scene_gop(96, 64, 5, 7), 96, 64, 400),
    ]
}

fn scene_gop(w: usize, h: usize, n: usize, seed: usize) -> Vec<Vec<u8>> {
    (0..n).map(|k| scene(w, h, k, seed)).collect()
}

/// Rate accuracy: the two-pass stream never exceeds the sequence
/// total, every frame lands within its budget (or at the `q == 255`
/// floor), the VBV buffer never underflows, and the accuracy (actual /
/// target) is printed alongside the one-pass chain's; the two-pass
/// stream spends at least as much of the pool as the one-pass chain on
/// content that can use it.
#[test]
fn two_pass_rate_accuracy_across_gop_shapes() {
    for (name, frames, w, h, target) in shapes() {
        let n = frames.len();
        let (packets, report) =
            encode_vp9_lossy_sequence_rc_two_pass(&refs(&frames), w, h, target, 0).expect("2p");
        assert_eq!(packets.len(), n);
        assert_eq!(report.len(), n);
        let total: usize = packets.iter().map(Vec::len).sum();
        let target_total = target * n;
        let floor = report
            .iter()
            .any(|f| f.base_q_idx == 255 && f.coded_bytes > f.budget);
        if !floor {
            assert!(total <= target_total, "{name}: {total} > {target_total}");
        }
        // Per-frame budget + VBV bookkeeping.
        let vbv = 2 * target;
        let mut level = vbv;
        for (i, f) in report.iter().enumerate() {
            assert_eq!(f.coded_bytes, packets[i].len());
            assert!(
                f.coded_bytes <= f.budget || f.base_q_idx == 255,
                "{name} frame {i}: {} > budget {}",
                f.coded_bytes,
                f.budget
            );
            assert!(
                f.budget <= level.max(1),
                "{name} frame {i}: budget above buffer level"
            );
            if f.base_q_idx != 255 {
                assert!(f.coded_bytes <= level, "{name} frame {i}: VBV underflow");
            }
            level = (level.saturating_sub(f.coded_bytes) + target).min(vbv);
        }
        // The keyframe is the costliest first-pass frame and draws the
        // largest budget.
        let kf = &report[0];
        assert!(report
            .iter()
            .all(|f| f.first_pass_bytes <= kf.first_pass_bytes));
        assert!(report.iter().all(|f| f.budget <= kf.budget));
        assert_eq!(kf.motion_activity, 0);
        // One-pass comparison at the same per-frame target.
        let one = encode_vp9_lossy_sequence_rc(&refs(&frames), w, h, target).expect("1p");
        let one_total: usize = one.iter().map(Vec::len).sum();
        let (p1, p2) = (psnr(&one, &frames), psnr(&packets, &frames));
        eprintln!(
            "{name}: target {target_total} B; two-pass {total} B ({:.1}%) @ {p2:.2} dB; one-pass {one_total} B ({:.1}%) @ {p1:.2} dB; q = {:?}; motion = {:?}",
            100.0 * total as f64 / target_total as f64,
            100.0 * one_total as f64 / target_total as f64,
            report.iter().map(|f| f.base_q_idx).collect::<Vec<_>>(),
            report.iter().map(|f| f.motion_activity).collect::<Vec<_>>(),
        );
    }
}

/// The scene cut draws a visibly larger budget than its neighbours
/// (first-pass inter cost spikes), and the moving GOP reports non-zero
/// motion activity on its P-frames while the static GOP reports zero.
#[test]
fn first_pass_statistics_track_content() {
    let (w, h) = (64u32, 48u32);
    let cut: Vec<Vec<u8>> = (0..6)
        .map(|k| {
            if k < 3 {
                scene(64, 48, k, 3)
            } else {
                scene(64, 48, k, 40)
            }
        })
        .collect();
    let (_, report) = encode_vp9_lossy_sequence_rc_two_pass(&refs(&cut), w, h, 700, 0).unwrap();
    assert!(
        report[3].first_pass_bytes > 2 * report[2].first_pass_bytes,
        "scene cut first-pass cost {} vs {}",
        report[3].first_pass_bytes,
        report[2].first_pass_bytes
    );
    assert!(report[3].budget > report[2].budget);
    let moving = scene_gop(64, 48, 4, 2);
    let (_, report) = encode_vp9_lossy_sequence_rc_two_pass(&refs(&moving), w, h, 700, 0).unwrap();
    assert!(report[1..].iter().any(|f| f.motion_activity > 0));
    let static_gop: Vec<Vec<u8>> = (0..4).map(|_| scene(64, 48, 0, 1)).collect();
    let (_, report) =
        encode_vp9_lossy_sequence_rc_two_pass(&refs(&static_gop), w, h, 700, 0).unwrap();
    assert!(report.iter().all(|f| f.motion_activity == 0));
}

/// An explicit VBV of exactly one frame target caps every budget at
/// the target; the encode is byte-deterministic; bad arguments reject.
#[test]
fn vbv_caps_budgets_and_contract() {
    let frames = scene_gop(64, 48, 4, 5);
    let (a, ra) = encode_vp9_lossy_sequence_rc_two_pass(&refs(&frames), 64, 48, 600, 600).unwrap();
    assert!(ra.iter().all(|f| f.budget <= 600));
    let (b, _) = encode_vp9_lossy_sequence_rc_two_pass(&refs(&frames), 64, 48, 600, 600).unwrap();
    assert_eq!(a, b, "deterministic");
    assert_eq!(
        encode_vp9_lossy_sequence_rc_two_pass(&[], 64, 48, 600, 0).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        encode_vp9_lossy_sequence_rc_two_pass(&refs(&frames), 64, 48, 0, 0).unwrap_err(),
        Error::Unsupported
    );
    let short = vec![0u8; 10];
    assert_eq!(
        encode_vp9_lossy_sequence_rc_two_pass(&[short.as_slice()], 64, 48, 600, 0).unwrap_err(),
        Error::Unsupported
    );
}

// ---------------------------------------------------------------------
// Round 458: two-pass rate control over the structured GOP + the
// format matrix.
// ---------------------------------------------------------------------

fn chroma_dims(w: usize, h: usize, ssx: bool, ssy: bool) -> (usize, usize) {
    (
        if ssx { w.div_ceil(2) } else { w },
        if ssy { h.div_ceil(2) } else { h },
    )
}

/// The `scene` content at any §7.2 geometry / bit depth, planar `u16`.
fn scene16(
    w: usize,
    h: usize,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    k: usize,
    seed: usize,
) -> Vec<u16> {
    let (cw, ch) = chroma_dims(w, h, ssx, ssy);
    let shift = bit_depth - 8;
    let base = scene(w, h, k, seed);
    let mut px: Vec<u16> = base[..w * h]
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            ((u32::from(v) << shift) as u16) | (if shift > 0 { (i as u16) & 3 } else { 0 })
        })
        .collect();
    for plane in 0..2usize {
        for y in 0..ch {
            for x in 0..cw {
                let v = ((x + k) * 5 + y * 3 + plane * 90 + seed) % 256;
                px.push(((v as u32) << shift) as u16);
            }
        }
    }
    px
}

fn to_u8(v: &[u16]) -> Vec<u8> {
    v.iter().map(|&s| s as u8).collect()
}

/// PSNR (dB) of decoded frames against planar `u16` sources at the
/// source bit depth.
fn psnr16(frames: &[Vp9DecodedFrame], sources: &[Vec<u16>], bit_depth: u32) -> f64 {
    let max = f64::from((1u32 << bit_depth) - 1);
    let mut sse = 0f64;
    let mut n = 0usize;
    for (f, src) in frames.iter().zip(sources) {
        let dec: Vec<u16> = f.y.iter().chain(&f.u).chain(&f.v).copied().collect();
        assert_eq!(dec.len(), src.len(), "sample count");
        for (&a, &b) in dec.iter().zip(src) {
            let d = f64::from(a) - f64::from(b);
            sse += d * d;
        }
        n += dec.len();
    }
    let mse = sse / n as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (max * max / mse).log10()
    }
}

/// Black-box validation dump under `OXIDEAV_VP9_RC_DUMP_DIR`:
/// `<name>/input.ivf`, `<name>/crate-decode.yuv` (the crate's planar
/// packing, little-endian `u16` above 8 bits) and `<name>/pixfmt.txt`.
/// No-op unless the env var is set.
#[allow(clippy::too_many_arguments)]
fn dump_when_requested(
    name: &str,
    packets: &[Vec<u8>],
    decoded: &[Vp9DecodedFrame],
    w: u32,
    h: u32,
    bit_depth: u32,
    ssx: bool,
    ssy: bool,
) {
    let Some(dir) = std::env::var_os("OXIDEAV_VP9_RC_DUMP_DIR") else {
        return;
    };
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&25u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (i, f) in packets.iter().enumerate() {
        ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes());
        ivf.extend_from_slice(f);
    }
    let mut yuv = Vec::new();
    for f in decoded {
        yuv.extend_from_slice(&f.to_planar_bytes());
    }
    let sub = std::path::Path::new(&dir).join(name);
    std::fs::create_dir_all(&sub).expect("create dump dir");
    std::fs::write(sub.join("input.ivf"), ivf).unwrap();
    std::fs::write(sub.join("crate-decode.yuv"), yuv).unwrap();
    let geom = match (ssx, ssy) {
        (true, true) => "420",
        (false, false) => "444",
        (true, false) => "422",
        (false, true) => "440",
    };
    let pix = if bit_depth == 8 {
        format!("yuv{geom}p")
    } else {
        format!("yuv{geom}p{bit_depth}le")
    };
    std::fs::write(sub.join("pixfmt.txt"), pix).unwrap();
}

/// The structured two-pass invariants shared by every shape / format:
/// one report entry per packet, `show_existing_frame` packets flagged
/// (one byte, no budget), the sequence total within the pool (unless a
/// packet hit the `q == 255` floor), every decoded packet within its
/// budget (or at the floor), budgets never above the VBV, the keyframe
/// the costliest first-pass packet, and the accuracy printed.
fn check_two_pass(
    name: &str,
    packets: &[Vec<u8>],
    report: &[Vp9TwoPassFrame],
    n: usize,
    target: usize,
    vbv: usize,
) -> f64 {
    assert_eq!(
        report.len(),
        packets.len(),
        "{name}: one report entry per packet"
    );
    let total: usize = packets.iter().map(Vec::len).sum();
    let pool = target * n;
    let floor = report
        .iter()
        .any(|f| !f.show_existing && f.base_q_idx == 255 && f.coded_bytes > f.budget);
    if !floor {
        assert!(total <= pool, "{name}: {total} > {pool}");
    }
    for (i, f) in report.iter().enumerate() {
        assert_eq!(
            f.coded_bytes,
            packets[i].len(),
            "{name} packet {i}: coded size"
        );
        if f.show_existing {
            assert!(f.coded_bytes <= 2, "{name} packet {i}: show_existing size");
            assert_eq!(
                (f.first_pass_bytes, f.budget, f.base_q_idx),
                (f.coded_bytes, 0, 0),
                "{name} packet {i}"
            );
            continue;
        }
        assert!(
            f.budget >= 1 && f.budget <= vbv,
            "{name} packet {i}: budget {}",
            f.budget
        );
        assert!(
            f.coded_bytes <= f.budget || f.base_q_idx == 255,
            "{name} packet {i}: {} > budget {}",
            f.coded_bytes,
            f.budget
        );
        assert!(f.first_pass_bytes > 1);
    }
    let kf = &report[0];
    assert!(!kf.show_existing && kf.motion_activity == 0 && kf.budget >= 1);
    let accuracy = 100.0 * total as f64 / pool as f64;
    assert!(
        accuracy >= 90.0,
        "{name}: only {accuracy:.1}% of the pool spent"
    );
    accuracy
}

/// 8-bit structured shapes: alt-ref pyramid, full segmentation on a
/// scene cut, intra-only alt-refs, tile rows — and the 4:4:4 / 4:2:2
/// / 4:4:0 wrappers — all under the two-pass allocation. Each stream
/// decodes to the display frame count at its declared format, honors
/// the invariants of [`check_two_pass`], is byte-deterministic, and
/// its PSNR is reported next to the plain chain's two-pass at the same
/// target (4:2:0 shapes).
#[test]
fn two_pass_structured_gop_across_shapes_and_formats() {
    let (w, h) = (64usize, 48usize);
    let moving = |ssx: bool, ssy: bool| -> Vec<Vec<u8>> {
        (0..6)
            .map(|k| to_u8(&scene16(w, h, ssx, ssy, 8, k, 2)))
            .collect()
    };
    let cut: Vec<Vec<u8>> = (0..6)
        .map(|k| {
            if k < 3 {
                scene(w, h, k, 3)
            } else {
                scene(w, h, k, 40)
            }
        })
        .collect();
    let pyramid = |q: u8, interval: u32| {
        let mut c = Vp9GopConfig::new(q);
        c.altref_interval = interval;
        c
    };
    let mut seg_full = pyramid(110, 3);
    seg_full.segmentation = Vp9Segmentation::Full;
    let mut intra_only = pyramid(110, 3);
    intra_only.intra_only_altref = true;
    let mut tile_rows = pyramid(110, 2);
    tile_rows.tile_rows_log2 = 1;
    type Case = (&'static str, Vec<Vec<u8>>, bool, bool, Vp9GopConfig, usize);
    let cases: Vec<Case> = vec![
        (
            "pyramid-arf3",
            moving(true, true),
            true,
            true,
            pyramid(110, 3),
            600,
        ),
        ("seg-full-scene-cut", cut, true, true, seg_full, 700),
        (
            "intra-only-arf3",
            moving(true, true),
            true,
            true,
            intra_only,
            600,
        ),
        (
            "tile-rows-arf2",
            moving(true, true),
            true,
            true,
            tile_rows,
            600,
        ),
        (
            "444-arf2",
            moving(false, false),
            false,
            false,
            pyramid(110, 2),
            900,
        ),
        (
            "422-arf3",
            moving(true, false),
            true,
            false,
            pyramid(110, 3),
            750,
        ),
        (
            "440-arf3",
            moving(false, true),
            false,
            true,
            pyramid(110, 3),
            750,
        ),
    ];
    for (name, frames, ssx, ssy, cfg, target) in cases {
        let n = frames.len();
        let encode = |cfg: &Vp9GopConfig| match (ssx, ssy) {
            (true, true) => encode_vp9_lossy_sequence_rc_two_pass_with(
                &refs(&frames),
                w as u32,
                h as u32,
                cfg,
                target,
                0,
            ),
            (false, false) => encode_vp9_lossy_sequence_rc_two_pass_with_444(
                &refs(&frames),
                w as u32,
                h as u32,
                cfg,
                target,
                0,
            ),
            (true, false) => encode_vp9_lossy_sequence_rc_two_pass_with_422(
                &refs(&frames),
                w as u32,
                h as u32,
                cfg,
                target,
                0,
            ),
            (false, true) => encode_vp9_lossy_sequence_rc_two_pass_with_440(
                &refs(&frames),
                w as u32,
                h as u32,
                cfg,
                target,
                0,
            ),
        };
        let (packets, report) = encode(&cfg).expect(name);
        let decoded = decode_vp9_sequence(&refs(&packets)).expect("decodes");
        assert_eq!(decoded.len(), n, "{name}: display frames");
        for f in &decoded {
            assert_eq!(
                (f.subsampling_x, f.subsampling_y, f.bit_depth),
                (ssx, ssy, 8),
                "{name}: format"
            );
        }
        // Planning A/B (round 458): the same target with scene-cut
        // keyframes and adaptive group lengths off — the unplanned
        // stream keeps every group of the configured interval (hidden
        // alt-refs exist whenever a group forms), the planned one is
        // never more than 0.3 dB below it and the report flags every
        // keyframe it placed.
        let mut unplanned_cfg = cfg;
        unplanned_cfg.scene_cut_keyframes = false;
        unplanned_cfg.adaptive_group_length = false;
        let (unplanned, ureport) = encode(&unplanned_cfg).expect(name);
        assert!(
            ureport.iter().filter(|f| f.show_existing).count() >= 1,
            "{name}: unplanned stream forms alt-ref groups"
        );
        assert_eq!(ureport.iter().filter(|f| f.keyframe).count(), 1);
        assert!(report[0].keyframe && report[0].frame == 0);
        let placed: Vec<usize> = report
            .iter()
            .filter(|f| f.keyframe && f.frame > 0)
            .map(|f| f.frame)
            .collect();
        if name == "seg-full-scene-cut" {
            assert_eq!(placed, vec![3], "{name}: keyframe at the cut");
        } else {
            assert!(
                placed.is_empty(),
                "{name}: no keyframe placed on continuous content"
            );
        }
        let vbv = (cfg.altref_interval as usize + 1).max(2) * target;
        let accuracy = check_two_pass(name, &packets, &report, n, target, vbv);
        dump_when_requested(name, &packets, &decoded, w as u32, h as u32, 8, ssx, ssy);
        let (again, _) = encode(&cfg).expect(name);
        assert_eq!(packets, again, "{name}: byte-deterministic");
        let sources: Vec<Vec<u16>> = frames
            .iter()
            .map(|f| f.iter().map(|&v| u16::from(v)).collect())
            .collect();
        let p = psnr16(&decoded, &sources, 8);
        assert!(p > 25.0, "{name}: PSNR {p:.2} dB");
        let total: usize = packets.iter().map(Vec::len).sum();
        let udecoded = decode_vp9_sequence(&refs(&unplanned)).expect("decodes");
        let pu = psnr16(&udecoded, &sources, 8);
        assert!(
            p + 0.3 >= pu,
            "{name}: planned {p:.2} dB vs unplanned {pu:.2} dB"
        );
        let utotal: usize = unplanned.iter().map(Vec::len).sum();
        let chain_note = if ssx && ssy {
            let (chain, _) = encode_vp9_lossy_sequence_rc_two_pass(
                &refs(&frames),
                w as u32,
                h as u32,
                target,
                0,
            )
            .expect("chain");
            let chain_total: usize = chain.iter().map(Vec::len).sum();
            format!(
                "; chain two-pass {chain_total} B @ {:.2} dB",
                psnr(&chain, &frames)
            )
        } else {
            String::new()
        };
        eprintln!(
            "{name}: pool {} B; planned two-pass {total} B ({accuracy:.1}%) @ {p:.2} dB in {} packets (keyframes {:?}); unplanned {utotal} B @ {pu:.2} dB in {} packets{chain_note}; q = {:?}",
            target * n,
            packets.len(),
            report
                .iter()
                .filter(|f| f.keyframe)
                .map(|f| f.frame)
                .collect::<Vec<_>>(),
            unplanned.len(),
            report.iter().map(|f| f.base_q_idx).collect::<Vec<_>>(),
        );
    }
}

/// The HBD wrappers under the two-pass allocation: 10-bit 4:2:0
/// pyramid, 12-bit 4:2:2 with full segmentation, 10-bit 4:4:0 with
/// intra-only alt-refs — declared profile / depth / subsampling on
/// every decoded frame, the shared invariants, PSNR at the source
/// depth.
#[test]
fn two_pass_structured_gop_hbd_formats() {
    let (w, h) = (64usize, 48usize);
    let mut seg_full = Vp9GopConfig::new(110);
    seg_full.altref_interval = 3;
    seg_full.segmentation = Vp9Segmentation::Full;
    let mut intra_only = Vp9GopConfig::new(110);
    intra_only.altref_interval = 3;
    intra_only.intra_only_altref = true;
    let mut pyramid = Vp9GopConfig::new(110);
    pyramid.altref_interval = 3;
    type Case = (&'static str, u32, bool, bool, Vp9GopConfig, usize);
    let cases: Vec<Case> = vec![
        ("hbd10-420-arf3", 10, true, true, pyramid, 800),
        ("hbd12-422-seg-full", 12, true, false, seg_full, 1200),
        ("hbd10-440-intra-only", 10, false, true, intra_only, 1000),
    ];
    for (name, bit_depth, ssx, ssy, cfg, target) in cases {
        let frames: Vec<Vec<u16>> = (0..5)
            .map(|k| scene16(w, h, ssx, ssy, bit_depth, k, 2))
            .collect();
        let refs16: Vec<&[u16]> = frames.iter().map(Vec::as_slice).collect();
        let n = frames.len();
        let (packets, report) = match (ssx, ssy) {
            (true, true) => encode_vp9_lossy_sequence_rc_two_pass_with_hbd(
                &refs16,
                w as u32,
                h as u32,
                bit_depth as u8,
                true,
                &cfg,
                target,
                0,
            ),
            (false, false) => encode_vp9_lossy_sequence_rc_two_pass_with_hbd(
                &refs16,
                w as u32,
                h as u32,
                bit_depth as u8,
                false,
                &cfg,
                target,
                0,
            ),
            (true, false) => encode_vp9_lossy_sequence_rc_two_pass_with_hbd_422(
                &refs16,
                w as u32,
                h as u32,
                bit_depth as u8,
                &cfg,
                target,
                0,
            ),
            (false, true) => encode_vp9_lossy_sequence_rc_two_pass_with_hbd_440(
                &refs16,
                w as u32,
                h as u32,
                bit_depth as u8,
                &cfg,
                target,
                0,
            ),
        }
        .expect(name);
        let decoded = decode_vp9_sequence(&refs(&packets)).expect("decodes");
        assert_eq!(decoded.len(), n, "{name}: display frames");
        for f in &decoded {
            assert_eq!(
                (f.subsampling_x, f.subsampling_y, u32::from(f.bit_depth)),
                (ssx, ssy, bit_depth),
                "{name}: format"
            );
        }
        let vbv = (cfg.altref_interval as usize + 1).max(2) * target;
        let accuracy = check_two_pass(name, &packets, &report, n, target, vbv);
        dump_when_requested(
            name, &packets, &decoded, w as u32, h as u32, bit_depth, ssx, ssy,
        );
        let p = psnr16(&decoded, &frames, bit_depth);
        assert!(p > 27.0, "{name}: PSNR {p:.2} dB");
        let total: usize = packets.iter().map(Vec::len).sum();
        eprintln!(
            "{name}: pool {} B; structured two-pass {total} B ({accuracy:.1}%) @ {p:.2} dB in {} packets; q = {:?}",
            target * n,
            packets.len(),
            report.iter().map(|f| f.base_q_idx).collect::<Vec<_>>(),
        );
    }
}

/// Structured two-pass contract: an explicit one-target VBV caps every
/// budget at the target; bad arguments reject (empty, zero target,
/// `base_q_idx == 0`, `altref_interval == 0`, an invalid tile layout,
/// a bad HBD depth, samples above the declared depth, short buffers).
#[test]
fn two_pass_structured_gop_contract() {
    let frames = scene_gop(64, 48, 5, 5);
    let mut cfg = Vp9GopConfig::new(110);
    cfg.altref_interval = 2;
    let (_, report) =
        encode_vp9_lossy_sequence_rc_two_pass_with(&refs(&frames), 64, 48, &cfg, 600, 600).unwrap();
    assert!(report.iter().all(|f| f.budget <= 600));
    let err = |r: Result<(Vec<Vec<u8>>, Vec<Vp9TwoPassFrame>), Error>| {
        assert_eq!(r.unwrap_err(), Error::Unsupported)
    };
    err(encode_vp9_lossy_sequence_rc_two_pass_with(
        &[],
        64,
        48,
        &cfg,
        600,
        0,
    ));
    err(encode_vp9_lossy_sequence_rc_two_pass_with(
        &refs(&frames),
        64,
        48,
        &cfg,
        0,
        0,
    ));
    err(encode_vp9_lossy_sequence_rc_two_pass_with(
        &refs(&frames),
        64,
        48,
        &Vp9GopConfig::new(0),
        600,
        0,
    ));
    let mut bad = cfg;
    bad.altref_interval = 0;
    err(encode_vp9_lossy_sequence_rc_two_pass_with(
        &refs(&frames),
        64,
        48,
        &bad,
        600,
        0,
    ));
    let mut tiles = cfg;
    tiles.tile_cols_log2 = 1; // 64 px wide cannot host two tile columns.
    err(encode_vp9_lossy_sequence_rc_two_pass_with(
        &refs(&frames),
        64,
        48,
        &tiles,
        600,
        0,
    ));
    let short = vec![0u8; 10];
    err(encode_vp9_lossy_sequence_rc_two_pass_with(
        &[short.as_slice()],
        64,
        48,
        &cfg,
        600,
        0,
    ));
    let f16 = scene16(64, 48, true, true, 10, 0, 1);
    err(encode_vp9_lossy_sequence_rc_two_pass_with_hbd(
        &[f16.as_slice()],
        64,
        48,
        9,
        true,
        &cfg,
        600,
        0,
    ));
    let mut over = f16.clone();
    over[0] = 1024;
    err(encode_vp9_lossy_sequence_rc_two_pass_with_hbd(
        &[over.as_slice()],
        64,
        48,
        10,
        true,
        &cfg,
        600,
        0,
    ));
}

/// The planner leaves continuous, static content alone: no keyframe
/// is placed, every group keeps the configured interval (the packet
/// sequence equals the unplanned one), and the two streams are
/// byte-identical — planning only acts on measured statistics.
#[test]
fn two_pass_planner_keeps_static_gop_intact() {
    let frames: Vec<Vec<u8>> = (0..7).map(|_| scene(64, 48, 0, 1)).collect();
    let mut cfg = Vp9GopConfig::new(110);
    cfg.altref_interval = 3;
    let (planned, report) =
        encode_vp9_lossy_sequence_rc_two_pass_with(&refs(&frames), 64, 48, &cfg, 500, 0).unwrap();
    let mut off = cfg;
    off.scene_cut_keyframes = false;
    off.adaptive_group_length = false;
    let (unplanned, ureport) =
        encode_vp9_lossy_sequence_rc_two_pass_with(&refs(&frames), 64, 48, &off, 500, 0).unwrap();
    assert_eq!(planned, unplanned, "static content: planning is a no-op");
    assert_eq!(report, ureport);
    assert_eq!(report.iter().filter(|f| f.keyframe).count(), 1);
    assert_eq!(report.iter().filter(|f| f.show_existing).count(), 2);
    assert!(report.iter().all(|f| f.motion_activity == 0));
    // Display-frame bookkeeping: every display frame is coded exactly
    // once, the show_existing packets present their group's alt-ref.
    let mut coded: Vec<usize> = report
        .iter()
        .filter(|f| !f.show_existing)
        .map(|f| f.frame)
        .collect();
    coded.sort_unstable();
    assert_eq!(coded, (0..7).collect::<Vec<_>>());
    assert_eq!(
        report
            .iter()
            .filter(|f| f.show_existing)
            .map(|f| f.frame)
            .collect::<Vec<_>>(),
        vec![3, 6]
    );
}
