//! `Decoder::output_params` / `output_params_generation` /
//! `output_pixel_format` — the stream-level description of the frames
//! the registry decoder emits (frames themselves carry no size or
//! format).
//!
//! Pins:
//!
//! * the answer is there **before the first frame** (from the first
//!   keyframe header submitted) and, after every `receive_frame`,
//!   describes exactly that frame — checked against the emitted plane
//!   shapes and against black-box `ffprobe -show_streams` values
//!   (`tests/fixtures/output_params/ffprobe.txt`, regenerated with
//!   `generate.sh` there) across 8/10/12-bit, 4:2:0/4:2:2/4:4:0/4:4:4,
//!   RGB, odd dimensions and signalled colour spaces;
//! * the generation bumps exactly at the first returned frame of a new
//!   shape — a keyframe at a new size/format, a §8.5.2.3
//!   reference-scaled inter frame, the first frame after a `reset`
//!   into a differently-shaped segment — and never otherwise;
//! * the per-frame (answer, generation) sequence is identical at 1 and
//!   8 threads.

use oxideav_core::registry::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, ColorRange, ColorSignal, Error as CoreError, ExecutionContext, Frame,
    MatrixCoefficients, OutputParamsWatch, Packet, PixelFormat, TimeBase, VideoFrame,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/output_params");

/// The frame payloads of an IVF file.
fn ivf_frames(name: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(format!("{DIR}/{name}.ivf")).expect("fixture");
    assert_eq!(&data[..4], b"DKIF", "{name}: IVF signature");
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut at = header_len;
    let mut out = Vec::new();
    while at + 12 <= data.len() {
        let size = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        at += 12;
        out.push(data[at..at + size].to_vec());
        at += size;
    }
    out
}

/// One `ffprobe.txt` row.
struct Probe {
    width: u32,
    height: u32,
    format: PixelFormat,
    range: ColorRange,
    matrix: MatrixCoefficients,
}

fn probes() -> Vec<(String, Probe)> {
    let text = std::fs::read_to_string(format!("{DIR}/ffprobe.txt")).expect("ffprobe.txt");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(f.len(), 6, "ffprobe row {l:?}");
            let format = match f[3] {
                "yuv420p" => PixelFormat::Yuv420P,
                "yuv422p" => PixelFormat::Yuv422P,
                "yuv440p" => PixelFormat::Yuv440P,
                "yuv444p" => PixelFormat::Yuv444P,
                "yuv420p10le" => PixelFormat::Yuv420P10Le,
                "yuv422p10le" => PixelFormat::Yuv422P10Le,
                "yuv444p10le" => PixelFormat::Yuv444P10Le,
                "yuv420p12le" => PixelFormat::Yuv420P12Le,
                "yuv444p12le" => PixelFormat::Yuv444P12Le,
                "gbrp" => PixelFormat::Gbrp8,
                "gbrp10le" => PixelFormat::Gbrp10Le,
                other => panic!("unmapped ffprobe pix_fmt {other}"),
            };
            let range = match f[4] {
                "tv" => ColorRange::Limited,
                "pc" => ColorRange::Full,
                other => panic!("unmapped ffprobe color_range {other}"),
            };
            let matrix = match f[5] {
                "unknown" => MatrixCoefficients::UNSPECIFIED,
                "gbr" => MatrixCoefficients::IDENTITY,
                "bt709" => MatrixCoefficients::BT709,
                "bt2020nc" => MatrixCoefficients::BT2020_NCL,
                other => panic!("unmapped ffprobe color_space {other}"),
            };
            (
                f[0].to_string(),
                Probe {
                    width: f[1].parse().unwrap(),
                    height: f[2].parse().unwrap(),
                    format,
                    range,
                    matrix,
                },
            )
        })
        .collect()
}

fn probe(name: &str) -> Probe {
    probes()
        .into_iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("{name} missing from ffprobe.txt"))
        .1
}

fn assert_matches_probe(p: &CodecParameters, want: &Probe, what: &str) {
    assert_eq!(p.width, Some(want.width), "{what}: width");
    assert_eq!(p.height, Some(want.height), "{what}: height");
    assert_eq!(p.pixel_format, Some(want.format), "{what}: pixel_format");
    assert_eq!(p.color_signal.range, want.range, "{what}: color range");
    assert_eq!(p.color_signal.matrix, want.matrix, "{what}: matrix");
    assert_eq!(p.frame_rate, None, "{what}: VP9 codes no frame rate");
}

/// The emitted planes are the truth: the description must predict
/// every plane's extent and storage exactly.
fn assert_frame_matches(v: &VideoFrame, p: &CodecParameters, what: &str) {
    let (w, h) = (p.width.unwrap(), p.height.unwrap());
    let fmt = p.pixel_format.unwrap();
    let bps = match fmt {
        PixelFormat::Yuv420P
        | PixelFormat::Yuv422P
        | PixelFormat::Yuv440P
        | PixelFormat::Yuv444P
        | PixelFormat::Gbrp8 => 1,
        _ => 2,
    };
    let planes = v.image_planes();
    assert_eq!(planes.len(), 3, "{what}: plane count");
    for (i, plane) in planes.iter().enumerate() {
        let (pw, ph) = fmt.plane_dimensions(i, w, h).expect("plane dims");
        assert_eq!(plane.stride, pw as usize * bps, "{what}: plane {i} stride");
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "{what}: plane {i} rows"
        );
    }
}

fn decoder(threads: usize) -> oxideav_vp9::Vp9Decoder {
    let mut dec = oxideav_vp9::Vp9Decoder::new();
    if threads > 1 {
        dec.set_execution_context(&ExecutionContext::with_threads(threads));
    }
    dec
}

/// Per returned frame: (output_params, generation), after checking the
/// answer against the frame's planes.
type Trace = Vec<(CodecParameters, u64)>;

fn drain(dec: &mut oxideav_vp9::Vp9Decoder, trace: &mut Trace, what: &str) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                let p = dec.output_params().expect("answer after a frame");
                assert_frame_matches(&v, &p, what);
                assert_eq!(dec.output_pixel_format(), p.pixel_format, "{what}: narrow");
                assert_eq!(dec.pixel_format(), p.pixel_format, "{what}: inherent");
                trace.push((p, dec.output_params_generation()));
            }
            Ok(_) => panic!("{what}: non-video frame"),
            Err(CoreError::NeedMore) | Err(CoreError::Eof) => return,
            Err(e) => panic!("{what}: {e}"),
        }
    }
}

/// The comparable part of a trace (every field the decoder answers).
type Key = (
    Option<u32>,
    Option<u32>,
    Option<PixelFormat>,
    ColorSignal,
    u64,
);

fn keys(t: &Trace) -> Vec<Key> {
    t.iter()
        .map(|(p, g)| (p.width, p.height, p.pixel_format, p.color_signal, *g))
        .collect()
}

fn decode_trace(packets: &[Vec<u8>], threads: usize, what: &str) -> Trace {
    let mut dec = decoder(threads);
    let mut trace = Vec::new();
    for (i, data) in packets.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::MILLIS, data.clone()).with_pts(i as i64))
            .unwrap_or_else(|e| panic!("{what}: packet {i}: {e}"));
        drain(&mut dec, &mut trace, what);
    }
    dec.flush().unwrap();
    drain(&mut dec, &mut trace, what);
    trace
}

#[test]
fn output_params_match_ffprobe_before_and_after_the_first_frame() {
    let all = probes();
    assert!(all.len() >= 14, "fixture set");
    for (name, want) in &all {
        let packets = ivf_frames(name);
        for threads in [1usize, 8] {
            let what = format!("{name} @{threads}t");
            let mut dec = decoder(threads);
            assert!(dec.output_params().is_none(), "{what}: nothing known yet");
            assert_eq!(dec.output_pixel_format(), None, "{what}");
            dec.send_packet(&Packet::new(0, TimeBase::MILLIS, packets[0].clone()))
                .unwrap();
            // Known from the keyframe header, before any frame is
            // returned (at 8 threads the frame may still be decoding).
            let early = dec.output_params().expect("answer before the first frame");
            assert_matches_probe(&early, want, &format!("{what} (before frame 0)"));
            assert_eq!(dec.output_params_generation(), 0);

            let mut stream = CodecParameters::video(CodecId::new("vp9"));
            let watch = OutputParamsWatch::new();
            assert!(watch.prime(&dec, &mut stream), "{what}: prime fills");
            assert_eq!(stream.width, Some(want.width));

            let mut trace = Vec::new();
            drain(&mut dec, &mut trace, &what);
            for data in &packets[1..] {
                dec.send_packet(&Packet::new(0, TimeBase::MILLIS, data.clone()))
                    .unwrap();
                drain(&mut dec, &mut trace, &what);
            }
            dec.flush().unwrap();
            drain(&mut dec, &mut trace, &what);
            assert_eq!(trace.len(), 2, "{what}: shown frames");
            for (i, (p, generation)) in trace.iter().enumerate() {
                assert_matches_probe(p, want, &format!("{what} frame {i}"));
                assert_eq!(*generation, 0, "{what}: a fixed-shape stream never bumps");
            }
        }
    }
}

/// Concatenated segments: each segment's keyframe changes the shape;
/// the generation moves exactly at that segment's first returned frame.
#[test]
fn generation_bumps_at_the_first_frame_of_each_new_segment() {
    let segments = [
        "p0-420-8",
        "seg-32x24-444-10",
        "p1-rgb-8",
        "p1-rgb-8",
        "odd-35x19-420",
    ];
    let mut packets = Vec::new();
    let mut expect: Vec<(Probe, u64)> = Vec::new();
    let mut generation = 0u64;
    let mut prev: Option<&str> = None;
    for name in segments {
        let frames = ivf_frames(name);
        if prev.is_some_and(|p| p != name) {
            generation += 1;
        }
        for _ in &frames {
            expect.push((probe(name), generation));
        }
        packets.extend(frames);
        prev = Some(name);
    }
    let serial = decode_trace(&packets, 1, "concat @1t");
    assert_eq!(serial.len(), expect.len());
    for (i, ((p, g), (want, wg))) in serial.iter().zip(&expect).enumerate() {
        assert_matches_probe(p, want, &format!("concat frame {i}"));
        assert_eq!(g, wg, "concat frame {i}: generation");
    }
    let threaded = decode_trace(&packets, 8, "concat @8t");
    assert_eq!(keys(&serial), keys(&threaded), "1 vs 8 threads");

    // The consumer helper re-reads exactly at the bumps.
    let mut dec = decoder(8);
    let mut stream = CodecParameters::video(CodecId::new("vp9"));
    let mut watch = OutputParamsWatch::new();
    let mut changed_at = Vec::new();
    let mut n = 0usize;
    let mut pull = |dec: &mut oxideav_vp9::Vp9Decoder, stream: &mut CodecParameters| {
        while let Ok(_f) = dec.receive_frame() {
            if watch.observe(&*dec, stream) {
                changed_at.push(n);
            }
            n += 1;
        }
    };
    for data in &packets {
        dec.send_packet(&Packet::new(0, TimeBase::MILLIS, data.clone()))
            .unwrap();
        pull(&mut dec, &mut stream);
    }
    dec.flush().unwrap();
    pull(&mut dec, &mut stream);
    assert_eq!(changed_at, vec![0, 2, 4, 8]);
    assert_eq!(stream.width, Some(35));
}

/// Reference scaling: every inter frame of the in-crate resized chain
/// is coded at its own §6.2.5 size and predicted through the §8.5.2.3
/// scaled sampler — the generation moves at each size change.
#[test]
fn generation_bumps_on_reference_scaled_inter_frames() {
    let sizes = [
        (64u32, 48u32),
        (64, 48),
        (32, 24),
        (48, 32),
        (48, 32),
        (64, 48),
    ];
    let frames: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(k, &(w, h))| {
            let (w, h) = (w as usize, h as usize);
            let n = w * h + 2 * (w / 2) * (h / 2);
            (0..n).map(|i| ((i * 7 + k * 13) % 251) as u8).collect()
        })
        .collect();
    let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
    let packets = oxideav_vp9::encode_vp9_lossy_sequence_resized(&refs, &sizes, 110).unwrap();
    let serial = decode_trace(&packets, 1, "resized @1t");
    let got: Vec<(u32, u32, u64)> = serial
        .iter()
        .map(|(p, g)| (p.width.unwrap(), p.height.unwrap(), *g))
        .collect();
    assert_eq!(
        got,
        vec![
            (64, 48, 0),
            (64, 48, 0),
            (32, 24, 1),
            (48, 32, 2),
            (48, 32, 2),
            (64, 48, 3)
        ]
    );
    for (p, _) in &serial {
        assert_eq!(p.pixel_format, Some(PixelFormat::Yuv420P));
    }
    assert_eq!(
        keys(&serial),
        keys(&decode_trace(&packets, 8, "resized @8t")),
        "1 vs 8 threads"
    );
}

/// `reset` (a seek): the answer keeps describing the last returned
/// frame until the first frame after the seek, which carries the bump
/// when its shape differs — and no bump when it does not.
#[test]
fn reset_follows_the_frame_boundary_rule() {
    for threads in [1usize, 8] {
        let a = ivf_frames("p0-420-8");
        let b = ivf_frames("seg-32x24-444-10");
        let mut dec = decoder(threads);
        let mut trace = Vec::new();
        for data in &a {
            dec.send_packet(&Packet::new(0, TimeBase::MILLIS, data.clone()))
                .unwrap();
        }
        dec.flush().unwrap();
        drain(&mut dec, &mut trace, "seg a");
        assert_eq!(trace.len(), 2);

        dec.reset().unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::MILLIS, b[0].clone()))
            .unwrap();
        let p = dec.output_params().unwrap();
        assert_eq!((p.width, dec.output_params_generation()), (Some(48), 0));
        dec.flush().unwrap();
        drain(&mut dec, &mut trace, "seg b");
        let last = trace.last().unwrap();
        assert_eq!((last.0.width, last.1), (Some(32), 1), "@{threads}t");

        // Seek back into the same shape: no bump.
        dec.reset().unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::MILLIS, b[0].clone()))
            .unwrap();
        dec.flush().unwrap();
        drain(&mut dec, &mut trace, "seg b again");
        assert_eq!(trace.last().unwrap().1, 1, "@{threads}t");
    }
}

/// Over the staged clean-room corpus (docs-gated): every returned
/// frame's planes match the answer, the generation moves exactly when
/// consecutive frames differ in shape (hidden alt-refs, intra-only
/// frames, §8.9 re-displays and reference-scaled frames included), the
/// trace is thread-count independent, and a few streams pin the
/// black-box ffprobe answer (`render_size( )` is NOT the emitted size;
/// `CS_BT_601` is ffprobe's `bt470bg`).
#[test]
fn docs_corpus_traces_are_consistent_and_thread_independent() {
    let root = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/video/vp9/fixtures"
    ));
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let pinned: [(&str, u32, u32, PixelFormat, ColorRange, MatrixCoefficients); 5] = [
        (
            "render-size-128x72",
            64,
            64,
            PixelFormat::Yuv420P,
            ColorRange::Limited,
            MatrixCoefficients::BT470_SYSTEM_BG,
        ),
        (
            "scaled-reference",
            128,
            128,
            PixelFormat::Yuv420P,
            ColorRange::Limited,
            MatrixCoefficients::BT470_SYSTEM_BG,
        ),
        (
            "intra-only",
            64,
            64,
            PixelFormat::Yuv420P,
            ColorRange::Limited,
            MatrixCoefficients::BT470_SYSTEM_BG,
        ),
        (
            "color-bt709-full-range",
            96,
            64,
            PixelFormat::Yuv420P,
            ColorRange::Full,
            MatrixCoefficients::BT709,
        ),
        (
            "bit-depth-10-rgb",
            64,
            64,
            PixelFormat::Gbrp10Le,
            ColorRange::Full,
            MatrixCoefficients::IDENTITY,
        ),
    ];
    let mut swept = 0usize;
    let mut bumped = 0usize;
    for entry in std::fs::read_dir(root).expect("fixtures dir") {
        let dir = entry.expect("entry").path();
        let ivf = dir.join("input.ivf");
        if !ivf.is_file() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let data = std::fs::read(&ivf).unwrap();
        let mut at = u16::from_le_bytes([data[6], data[7]]) as usize;
        let mut packets = Vec::new();
        while at + 12 <= data.len() {
            let size = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
            at += 12;
            packets.push(data[at..at + size].to_vec());
            at += size;
        }
        let serial = decode_trace(&packets, 1, &name);
        assert!(!serial.is_empty(), "{name}: frames");
        for w in serial.windows(2) {
            let same = (
                w[0].0.width,
                w[0].0.height,
                w[0].0.pixel_format,
                w[0].0.color_signal,
            ) == (
                w[1].0.width,
                w[1].0.height,
                w[1].0.pixel_format,
                w[1].0.color_signal,
            );
            let step = w[1].1.wrapping_sub(w[0].1);
            assert_eq!(step, u64::from(!same), "{name}: bump iff the shape changed");
            bumped += usize::from(!same);
        }
        assert_eq!(serial[0].1, 0, "{name}: first frame is not a change");
        assert_eq!(
            keys(&serial),
            keys(&decode_trace(&packets, 8, &name)),
            "{name}: 1 vs 8 threads"
        );
        if let Some(&(_, w, h, fmt, range, matrix)) = pinned.iter().find(|p| p.0 == name) {
            let p = &serial[0].0;
            assert_eq!(
                (p.width, p.height),
                (Some(w), Some(h)),
                "{name}: ffprobe size"
            );
            assert_eq!(p.pixel_format, Some(fmt), "{name}: ffprobe pix_fmt");
            assert_eq!(p.color_signal.range, range, "{name}: ffprobe range");
            assert_eq!(p.color_signal.matrix, matrix, "{name}: ffprobe space");
        }
        swept += 1;
    }
    assert!(swept >= 40, "corpus sweep ({swept})");
    assert!(
        bumped >= 4,
        "the corpus carries mid-stream size changes ({bumped})"
    );
}
