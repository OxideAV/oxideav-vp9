//! Integration tests against the docs/video/vp9/ fixture corpus.
//!
//! Each fixture under `../../docs/video/vp9/fixtures/<name>/` ships an
//! `input.ivf` (raw VP9 frames in IVF, including any libvpx-emitted
//! Annex B superframes), an `expected.yuv` byte-for-byte ground truth,
//! a `notes.md` describing the bitstream feature focus, and a
//! `trace.txt` capturing the per-frame `VP9_TRACE` events emitted by an
//! instrumented FFmpeg `vp9.c` decoder. The corpus and trace
//! vocabulary are documented in
//! `docs/video/vp9/vp9-fixtures-and-traces.md`.
//!
//! This driver decodes every fixture through the in-tree
//! [`Vp9Decoder`] and reports the per-fixture pixel-match rate against
//! the expected YUV.
//!
//! Acceptance:
//! * `Tier::BitExact` — must round-trip exactly. Failure = CI red.
//! * `Tier::ReportOnly` — divergence is logged but the test does NOT
//!   fail. Use this for fixtures that exercise codec features the
//!   in-tree decoder is still bringing up (HBD, 4:4:4, segmentation,
//!   superframes / show_existing_frame, multi-tile, …). Promote to
//!   `BitExact` once the underlying gap is closed.
//! * `Tier::Ignored` — disabled with `#[ignore]`; reserved for fixtures
//!   that need infrastructure not currently available (the present
//!   driver has none in this state).
//!
//! All fixtures start at `ReportOnly`. The driver still prints
//! per-fixture and per-frame stats via `eprintln!` regardless of tier
//! so that the matrix stays visible in `cargo test` output even before
//! anything is promoted.
//!
//! The trace.txt files are NOT consumed by this driver — they are an
//! aid for the human implementer when localising divergences via the
//! `VP9_TRACE` event vocabulary
//! (`docs/video/vp9/vp9-fixtures-and-traces.md`). Each per-fixture
//! `evaluate()` call references the trace path in the eprintln! header
//! so a failing run prints a clickable pointer.
//!
//! Spec references throughout follow the **VP9 Bitstream & Decoding
//! Process Specification, version 0.7** (the spec PDF). Per the
//! workspace policy, NO external decoder source (libvpx, libavcodec,
//! …) was consulted while writing this driver — fixtures are data,
//! traces are behavioural diff targets, the spec PDF is the authority.
//!
//! ## Reference shape vs. decoder output
//!
//! libvpx emits the reference `expected.yuv` in the bitstream's native
//! shape: 4:2:0 / 4:4:4 / GBR planar at 8/10/12 bits, little-endian
//! u16 containers for HBD. The in-tree decoder currently always
//! surfaces `Yuv420P` 8-bit `VideoFrame`s regardless of profile (see
//! `pixel_format_from_color_config`), so for non-yuv420p-8bit
//! fixtures the per-plane sizes will not match the reference. We
//! still load and size-check those fixtures; the per-frame diff is
//! recorded as a `plane size mismatch` error and the fixture stays
//! ReportOnly. Promote when the decoder grows the wider pixel-format
//! support.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_vp9::{decoder::Vp9Decoder, CODEC_ID_STR};

const IVF_HEADER_LEN: usize = 32;
const IVF_FRAME_HEADER_LEN: usize = 12;

/// Locate `docs/video/vp9/fixtures/<name>/`. Tests run with CWD set to
/// the crate root, so we walk two levels up to reach the workspace
/// root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/vp9/fixtures").join(name)
}

/// Iterate every IVF "frame" inside an IVF byte slice. Each entry is
/// the raw VP9 access-unit bytes; libvpx may pack several VP9 frames
/// into one IVF entry as an Annex-B superframe — those are split
/// internally by `Vp9Decoder` (see `split_superframe`).
fn ivf_frames(ivf: &[u8]) -> Vec<Vec<u8>> {
    assert!(ivf.len() >= IVF_HEADER_LEN, "IVF too short");
    assert_eq!(&ivf[0..4], b"DKIF");
    let header_len = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut off = header_len;
    let mut out = Vec::new();
    while off + IVF_FRAME_HEADER_LEN <= ivf.len() {
        let size =
            u32::from_le_bytes([ivf[off], ivf[off + 1], ivf[off + 2], ivf[off + 3]]) as usize;
        off += IVF_FRAME_HEADER_LEN;
        if off + size > ivf.len() {
            break;
        }
        out.push(ivf[off..off + size].to_vec());
        off += size;
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum PixFmt {
    /// 4:2:0, 8-bit. Luma full-res, chroma half-w * half-h. 8-bit per
    /// sample. The native shape of `Vp9Decoder`'s output today.
    Yuv420P8,
    /// 4:4:4, 8-bit. All planes full-res. (Profile 1.)
    Yuv444P8,
    /// 4:2:0, 10-bit little-endian. Chroma half-w/half-h, u16-LE
    /// containers carrying 10-bit samples. (Profile 2.)
    Yuv420P10Le,
    /// 4:4:4, 10-bit little-endian. (Profile 3.)
    Yuv444P10Le,
    /// 4:4:4, 12-bit little-endian. (Profile 3.)
    Yuv444P12Le,
    /// GBR planar 10-bit little-endian. RGB carried in profile 3 with
    /// `color_space=RGB`. Plane order in the reference is G/B/R; for
    /// pixel-match purposes we treat them like three full-res planes.
    Gbrp10Le,
}

impl PixFmt {
    fn bit_depth(&self) -> u32 {
        match self {
            PixFmt::Yuv420P8 | PixFmt::Yuv444P8 => 8,
            PixFmt::Yuv420P10Le | PixFmt::Yuv444P10Le | PixFmt::Gbrp10Le => 10,
            PixFmt::Yuv444P12Le => 12,
        }
    }

    /// Bytes per source sample in the reference `expected.yuv`. 1 for
    /// 8-bit, 2 for 10/12-bit.
    fn ref_bytes_per_sample(&self) -> usize {
        if self.bit_depth() == 8 {
            1
        } else {
            2
        }
    }

    /// Per-plane (width, height) for plane index `p`, in samples (NOT
    /// bytes — multiply by `ref_bytes_per_sample()` to get the byte
    /// span in the reference buffer).
    fn plane_dims(&self, width: usize, height: usize, p: usize) -> (usize, usize) {
        match (self, p) {
            (_, 0) => (width, height),
            (PixFmt::Yuv420P8, _) | (PixFmt::Yuv420P10Le, _) => {
                (width.div_ceil(2), height.div_ceil(2))
            }
            (PixFmt::Yuv444P8, _)
            | (PixFmt::Yuv444P10Le, _)
            | (PixFmt::Yuv444P12Le, _)
            | (PixFmt::Gbrp10Le, _) => (width, height),
        }
    }

    /// Per-frame size in bytes of the reference `expected.yuv`.
    fn frame_bytes(&self, width: usize, height: usize) -> usize {
        let bps = self.ref_bytes_per_sample();
        let mut total = 0usize;
        for p in 0..3 {
            let (w, h) = self.plane_dims(width, height, p);
            total += w * h * bps;
        }
        total
    }
}

/// Per-frame decode result against the per-frame slice of
/// `expected.yuv`. Counters are in samples (post-narrowing for HBD),
/// not bytes.
#[derive(Default)]
struct FrameDiff {
    y_total: usize,
    y_exact: usize,
    y_max: i32,
    uv_total: usize,
    uv_exact: usize,
    uv_max: i32,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let exact = self.y_exact + self.uv_exact;
        let total = self.y_total + self.uv_total;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }

    fn merge(&mut self, other: &FrameDiff) {
        self.y_total += other.y_total;
        self.y_exact += other.y_exact;
        self.y_max = self.y_max.max(other.y_max);
        self.uv_total += other.uv_total;
        self.uv_exact += other.uv_exact;
        self.uv_max = self.uv_max.max(other.uv_max);
    }

    /// Per-plane PSNR (dB) over the merged Y vs. UV samples — clamped
    /// at 99.99 dB when there's zero error to keep the column width
    /// finite.
    fn psnr_y(&self) -> f64 {
        psnr_from_max_scaled(self.y_max as f64, self.y_total)
    }
    fn psnr_uv(&self) -> f64 {
        psnr_from_max_scaled(self.uv_max as f64, self.uv_total)
    }
}

/// Cheap MAX-error-based pseudo-PSNR. We don't store SSE per plane to
/// keep the buffer footprint zero on a 6 MB corpus; the headline
/// "match-pct" is the primary metric. PSNR is logged as an additional
/// human-readable signal of how far off any divergent samples are.
fn psnr_from_max_scaled(max: f64, n: usize) -> f64 {
    if n == 0 || max == 0.0 {
        99.99
    } else {
        20.0 * (255.0_f64 / max).log10()
    }
}

/// Compare a single plane of our (u8) output against the reference.
/// For 8-bit reference data we compare byte-for-byte. For HBD
/// reference data we read u16-LE, narrow to u8 by `>> shift`, and
/// then compare — match-pct is therefore measured in 8-bit space; HBD
/// round-tripping is always lossy by 2-4 bits and would be invisible
/// if we compared in u16 space without a narrowing.
fn diff_plane(our: &[u8], refp: &[u8], bit_depth: u32) -> (usize, usize, i32) {
    let mut ex = 0usize;
    let mut max = 0i32;
    if bit_depth == 8 {
        let n = our.len().min(refp.len());
        for i in 0..n {
            let d = (our[i] as i32 - refp[i] as i32).abs();
            if d == 0 {
                ex += 1;
            }
            if d > max {
                max = d;
            }
        }
        (n, ex, max)
    } else {
        let shift = bit_depth - 8;
        let n_samples = (refp.len() / 2).min(our.len());
        for i in 0..n_samples {
            let lo = refp[i * 2];
            let hi = refp[i * 2 + 1];
            let r16 = u16::from_le_bytes([lo, hi]);
            let r8 = (r16 >> shift).min(255) as i32;
            let o8 = our[i] as i32;
            let d = (o8 - r8).abs();
            if d == 0 {
                ex += 1;
            }
            if d > max {
                max = d;
            }
        }
        (n_samples, ex, max)
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // BitExact unused at first commit; promote fixtures over time
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence.
    BitExact,
    /// Decode is permitted to diverge from the reference; the
    /// per-fixture stats are logged but the test does not fail.
    /// Promote to `BitExact` once the underlying decoder gap is
    /// closed.
    ReportOnly,
}

struct CorpusCase {
    name: &'static str,
    width: usize,
    height: usize,
    n_frames: usize,
    pix_fmt: PixFmt,
    tier: Tier,
}

/// Aggregate report of one fixture's decode pass.
struct DecodeReport {
    per_frame: Vec<Result<FrameDiff, String>>,
    visible_produced: usize,
    /// First non-NeedMore error from `send_packet` / `receive_frame`
    /// (recorded for the report banner; does NOT stop iteration).
    fatal: Option<String>,
}

fn decode_fixture(case: &CorpusCase) -> Option<DecodeReport> {
    let dir = fixture_dir(case.name);
    let ivf_path = dir.join("input.ivf");
    let yuv_path = dir.join("expected.yuv");
    let trace_path = dir.join("trace.txt");
    let ivf = match fs::read(&ivf_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {}: missing {} ({e}). docs/video/vp9/ corpus lives in the \
                 workspace umbrella repo; the standalone crate checkout has no \
                 fixtures.",
                case.name,
                ivf_path.display()
            );
            return None;
        }
    };
    let yuv_ref = match fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, yuv_path.display());
            return None;
        }
    };
    eprintln!(
        "fixture {}: ivf={} bytes, expected.yuv={} bytes, trace={}",
        case.name,
        ivf.len(),
        yuv_ref.len(),
        trace_path.display()
    );

    let frames = ivf_frames(&ivf);
    assert!(
        !frames.is_empty(),
        "fixture {} has no IVF frames",
        case.name
    );

    let frame_size = case.pix_fmt.frame_bytes(case.width, case.height);
    assert_eq!(
        yuv_ref.len(),
        case.n_frames * frame_size,
        "fixture {} expected.yuv size mismatch (have {} bytes, expected {} = {} frames * {})",
        case.name,
        yuv_ref.len(),
        case.n_frames * frame_size,
        case.n_frames,
        frame_size
    );

    let mut dec = Vp9Decoder::new(CodecId::new(CODEC_ID_STR));
    let mut visible_idx = 0usize;
    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(case.n_frames);
    let mut fatal: Option<String> = None;

    for (pkt_idx, frame_bytes) in frames.iter().enumerate() {
        let mut pkt = Packet::new(0, TimeBase::new(1, 1000), frame_bytes.clone());
        pkt.pts = Some(pkt_idx as i64);
        if let Err(e) = dec.send_packet(&pkt) {
            let msg = format!("packet {pkt_idx}: send_packet: {e:?}");
            per_frame.push(Err(msg.clone()));
            if fatal.is_none() {
                fatal = Some(msg);
            }
            continue;
        }
        // Drain any visible frames produced by this packet (a single
        // IVF entry can yield 0..N visible VP9 sub-frames after
        // superframe split; hidden alt-refs produce 0).
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    if visible_idx >= case.n_frames {
                        // Decoder produced more visible frames than
                        // the reference. Record but do not compare.
                        visible_idx += 1;
                        continue;
                    }
                    let ref_off = visible_idx * frame_size;
                    let ref_slice = &yuv_ref[ref_off..ref_off + frame_size];
                    let bps = case.pix_fmt.ref_bytes_per_sample();
                    let mut diff = FrameDiff::default();
                    let mut size_mismatch: Option<String> = None;
                    let mut ref_off_within = 0usize;
                    for p in 0..3 {
                        let (pw, ph) = case.pix_fmt.plane_dims(case.width, case.height, p);
                        let plane_bytes = pw * ph * bps;
                        let ref_plane = &ref_slice[ref_off_within..ref_off_within + plane_bytes];
                        ref_off_within += plane_bytes;
                        let our_plane = match vf.planes.get(p) {
                            Some(pl) => pl.data.as_slice(),
                            None => {
                                size_mismatch = Some(format!(
                                    "visible {visible_idx}: decoder produced {} planes, \
                                     reference expects 3",
                                    vf.planes.len()
                                ));
                                break;
                            }
                        };
                        let expected_our_len = pw * ph;
                        if our_plane.len() != expected_our_len {
                            // The decoder emits the bitstream's
                            // native plane sizes for 4:2:0 8-bit but
                            // narrows everything else to 4:2:0 today.
                            // Record a per-frame "plane size mismatch"
                            // error so the gap is visible without
                            // tripping the BitExact assertion.
                            size_mismatch = Some(format!(
                                "visible {visible_idx} plane {p}: our len {}, \
                                 expected {} samples ({pw}x{ph}); reference \
                                 was {plane_bytes} bytes ({} bpp)",
                                our_plane.len(),
                                expected_our_len,
                                case.pix_fmt.bit_depth()
                            ));
                            break;
                        }
                        let (n, ex, mx) =
                            diff_plane(our_plane, ref_plane, case.pix_fmt.bit_depth());
                        if p == 0 {
                            diff.y_total += n;
                            diff.y_exact += ex;
                            diff.y_max = diff.y_max.max(mx);
                        } else {
                            diff.uv_total += n;
                            diff.uv_exact += ex;
                            diff.uv_max = diff.uv_max.max(mx);
                        }
                    }
                    if let Some(msg) = size_mismatch {
                        per_frame.push(Err(msg));
                    } else {
                        per_frame.push(Ok(diff));
                    }
                    visible_idx += 1;
                }
                Ok(_) => continue,
                Err(Error::NeedMore) => break,
                Err(e) => {
                    let msg = format!("visible {visible_idx}: receive_frame: {e:?}");
                    per_frame.push(Err(msg.clone()));
                    if fatal.is_none() {
                        fatal = Some(msg);
                    }
                    break;
                }
            }
        }
    }

    Some(DecodeReport {
        per_frame,
        visible_produced: visible_idx,
        fatal,
    })
}

/// Pretty-print + tier-aware assertion. Per-frame stats always go to
/// stderr; the BitExact tier upgrades any divergence into a panic.
fn evaluate(case: &CorpusCase) {
    let report = match decode_fixture(case) {
        Some(r) => r,
        None => return, // missing fixture — already logged
    };

    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, r) in report.per_frame.iter().enumerate() {
        match r {
            Ok(d) => {
                eprintln!(
                    "  frame {i}: Y {}/{} exact (max diff {}, ~PSNR {:.2} dB), \
                     UV {}/{} exact (max diff {}, ~PSNR {:.2} dB), pct={:.2}%",
                    d.y_exact,
                    d.y_total,
                    d.y_max,
                    d.psnr_y(),
                    d.uv_exact,
                    d.uv_total,
                    d.uv_max,
                    d.psnr_uv(),
                    d.pct()
                );
                agg.merge(d);
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }

    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: aggregate {}/{} exact ({pct:.2}%), Y max diff {} (~PSNR {:.2} dB), \
         UV max diff {} (~PSNR {:.2} dB), visible_produced={}/{}{}",
        case.tier,
        case.name,
        agg.y_exact + agg.uv_exact,
        agg.y_total + agg.uv_total,
        agg.y_max,
        agg.psnr_y(),
        agg.uv_max,
        agg.psnr_uv(),
        report.visible_produced,
        case.n_frames,
        match &report.fatal {
            Some(f) => format!(", first_fatal=\"{f}\""),
            None => String::new(),
        }
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y_exact + agg.uv_exact,
                agg.y_total + agg.uv_total,
                "{}: not bit-exact (Y max diff {}, UV max diff {}; {:.4}% match)",
                case.name,
                agg.y_max,
                agg.uv_max,
                pct
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln! output above is the report.
            // TODO(vp9-corpus): tighten to BitExact once the
            // underlying decoder gap for this fixture is closed.
            let _ = pct;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// All fixtures start as ReportOnly. As the in-tree VP9 decoder closes
// the relevant gap, individual cases should be promoted to BitExact.
//
// Trace files (referenced in `evaluate()` via the eprintln! header)
// live alongside each fixture and capture FRAME / LOOPFILTER / TILING
// / SEGMENT / SEGMENT_FEAT / REFUPDATE / TILE / SB events emitted by
// the instrumented FFmpeg `vp9.c` decoder on the bitstream — useful
// for diffing against our own decoder's behaviour. See
// `docs/video/vp9/vp9-fixtures-and-traces.md` for the event
// vocabulary.

/// Smallest possible VP9 bitstream: 16x16 keyframe in profile 0,
/// `loop_filter_level=0`, single 64x64 SB partitioned down to 8x8.
/// Trace: docs/video/vp9/fixtures/tiny-i-only-16x16/trace.txt
#[test]
fn corpus_tiny_i_only_16x16() {
    evaluate(&CorpusCase {
        name: "tiny-i-only-16x16",
        width: 16,
        height: 16,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Profile 0 (4:2:0, 8-bit) common-path fixture — 4 frames at 128x128
/// (2x2 SB grid) with 1 KEY + 3 INTER. Trips through the typical
/// real-world decoder path.
/// Trace: docs/video/vp9/fixtures/profile-0-yuv420-8bit/trace.txt
#[test]
fn corpus_profile_0_yuv420_8bit() {
    evaluate(&CorpusCase {
        name: "profile-0-yuv420-8bit",
        width: 128,
        height: 128,
        n_frames: 4,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Profile 1 (4:4:4, 8-bit). Per VP9 spec v0.7 §6.2.2 / §6.2.5 the
/// keyframe header carries the colorspace block and a reserved-zero
/// bit. U/V planes are full-res (64x64 instead of 32x32).
/// Trace: docs/video/vp9/fixtures/profile-1-yuv444-8bit/trace.txt
#[test]
fn corpus_profile_1_yuv444_8bit() {
    evaluate(&CorpusCase {
        name: "profile-1-yuv444-8bit",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv444P8,
        tier: Tier::ReportOnly,
    });
}

/// Profile 2 (4:2:0, 10-bit). The decoder's `bpp_index=1` selects the
/// 10-bit DC/AC qlookup. Reference is u16-LE, narrowed to u8 by `>> 2`
/// for comparison (HBD round-trip is lossy in this driver — we only
/// score the 8-bit-narrowed match).
/// Trace: docs/video/vp9/fixtures/profile-2-yuv420-10bit/trace.txt
#[test]
fn corpus_profile_2_yuv420_10bit() {
    evaluate(&CorpusCase {
        name: "profile-2-yuv420-10bit",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv420P10Le,
        tier: Tier::ReportOnly,
    });
}

/// Profile 3 (4:4:4, 10-bit). The highest-profile fixture short of
/// 12-bit; exercises both the 4:4:4 chroma path and the 10-bit dequant
/// LUT in the same bitstream.
/// Trace: docs/video/vp9/fixtures/profile-3-yuv444-10bit/trace.txt
#[test]
fn corpus_profile_3_yuv444_10bit() {
    evaluate(&CorpusCase {
        name: "profile-3-yuv444-10bit",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv444P10Le,
        tier: Tier::ReportOnly,
    });
}

/// Profile 3 (4:4:4, 12-bit). The 12-bit corner of the bit-depth
/// matrix; `bpp_index=2` selects the 12-bit qlookup. Decoders that
/// implement only 8/10-bit qlookups will diverge in every coefficient.
/// Trace: docs/video/vp9/fixtures/profile-3-yuv444-12bit/trace.txt
#[test]
fn corpus_profile_3_yuv444_12bit() {
    evaluate(&CorpusCase {
        name: "profile-3-yuv444-12bit",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv444P12Le,
        tier: Tier::ReportOnly,
    });
}

/// Profile 3 with `color_space=RGB`. Plane order in the reference is
/// G/B/R 10-bit LE. We compare in 8-bit narrowed space; the layout
/// difference is invisible to the per-plane diff (G vs Y, B vs U,
/// R vs V) — but the channels are different enough that any match
/// here would be coincidence, which is the correct signal.
/// Trace: docs/video/vp9/fixtures/bit-depth-10-rgb/trace.txt
#[test]
fn corpus_bit_depth_10_rgb() {
    evaluate(&CorpusCase {
        name: "bit-depth-10-rgb",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Gbrp10Le,
        tier: Tier::ReportOnly,
    });
}

/// VP9 lossless mode (§6.2.7 derived `lossless=1`): `base_q_idx==0`
/// and all qdeltas zero. Decoder must force 4x4 transforms and use
/// the WHT-only inverse path on the Y plane.
/// Trace: docs/video/vp9/fixtures/lossless-i-only/trace.txt
#[test]
fn corpus_lossless_i_only() {
    evaluate(&CorpusCase {
        name: "lossless-i-only",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Low-extreme quantizer (`yac_qi=4`). Stresses the smallest non-zero
/// entries of the 8-bit DC/AC qlookup — an off-by-one diverges in
/// every reconstructed sample.
/// Trace: docs/video/vp9/fixtures/q-low/trace.txt
#[test]
fn corpus_q_low() {
    evaluate(&CorpusCase {
        name: "q-low",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// High-extreme quantizer (`yac_qi=240`, the libvpx ceiling). Stresses
/// the largest entries of the qlookup tables; complementary to
/// `q-low`.
/// Trace: docs/video/vp9/fixtures/q-high/trace.txt
#[test]
fn corpus_q_high() {
    evaluate(&CorpusCase {
        name: "q-high",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Smallest useful inter-frame test: 1 KEY + 1 P at 64x64.
/// `refresh_mask=0x01` (only LAST_FRAME refreshed),
/// `highprec_mvs=1` and FILTER_8TAP_SMOOTH on the P-frame.
/// Trace: docs/video/vp9/fixtures/i-frame-then-p-frame-64x64/trace.txt
#[test]
fn corpus_i_frame_then_p_frame_64x64() {
    evaluate(&CorpusCase {
        name: "i-frame-then-p-frame-64x64",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Two tile columns (`log2_tile_cols=1`) at 512x64. Two `TILE`
/// boundaries with independent VPx range-coder contexts; tests that
/// the in-tree multi-tile splitter resets state at each boundary.
/// Trace: docs/video/vp9/fixtures/tile-cols-2/trace.txt
#[test]
fn corpus_tile_cols_2() {
    evaluate(&CorpusCase {
        name: "tile-cols-2",
        width: 512,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// `segmentation_enabled=1` with libvpx's variance-AQ heuristic
/// (`-aq-mode 1`). Per-segment `q_enabled` + `q_val` deltas; the
/// decoder must build the per-MB segment map and apply `SEG_LVL_ALT_Q`
/// through the qindex pipeline.
/// Trace: docs/video/vp9/fixtures/segments-aq-mode/trace.txt
#[test]
fn corpus_segments_aq_mode() {
    evaluate(&CorpusCase {
        name: "segments-aq-mode",
        width: 128,
        height: 128,
        n_frames: 4,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// `error_resilient=1` + `parallel_mode=1` on every frame —
/// `refresh_ctx=0` so the decoder MUST NOT call entropy adaptation
/// between frames in this mode (VP9 spec v0.7 §6.2).
/// Trace: docs/video/vp9/fixtures/frame-parallel-mode/trace.txt
#[test]
fn corpus_frame_parallel_mode() {
    evaluate(&CorpusCase {
        name: "frame-parallel-mode",
        width: 64,
        height: 64,
        n_frames: 4,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Annex B superframes carrying hidden alt-refs (`show_frame=0` ARFs
/// concatenated with the next visible frame). Decoder must split each
/// IVF entry into N VP9 sub-frames via the trailing marker byte and
/// process the hidden ARF before the visible frame in the same
/// superframe. 16 visible frames / 18 FRAME events in the trace.
/// Trace: docs/video/vp9/fixtures/superframe-2/trace.txt
#[test]
fn corpus_superframe_2() {
    evaluate(&CorpusCase {
        name: "superframe-2",
        width: 64,
        height: 64,
        n_frames: 16,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// `show_existing_frame=1` repeats — first 4 bits `0010` followed by
/// the 1-bit show_existing flag and a 3-bit `frame_to_show_map_idx`.
/// On these frames the decoder copies a previously-decoded reference
/// to output and skips header / tile / loopfilter for that frame
/// entirely. 24 visible output frames.
/// Trace: docs/video/vp9/fixtures/show-existing-frame/trace.txt
#[test]
fn corpus_show_existing_frame() {
    evaluate(&CorpusCase {
        name: "show-existing-frame",
        width: 64,
        height: 64,
        n_frames: 24,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}
