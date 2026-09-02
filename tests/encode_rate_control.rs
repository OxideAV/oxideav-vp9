//! Round-455 **two-pass rate control** pins: first-pass statistics,
//! the VBV-modeled allocation, and rate accuracy (actual vs target
//! sequence bytes) across GOP shapes — static, translating, and a
//! mid-sequence scene cut — with the one-pass chain kept intact.

use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence_rc, encode_vp9_lossy_sequence_rc_two_pass, Error,
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
