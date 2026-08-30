//! Registry integration tests: the framework-facing `Vp9Decoder` /
//! `Vp9Encoder` pair, the `make_decoder` / `make_encoder` factories,
//! and `register( )`.
//!
//! The load-bearing pins:
//!
//! * the registry **Encoder rides the §7.2.6 chain-framed default GOP
//!   path** — its packet bytes are byte-identical to the matching
//!   public batch entry (`encode_vp9_lossy_sequence` /
//!   `encode_vp9_lossless_sequence` / the format-matrix variants);
//! * the registry **Decoder is the incremental `Vp9SequenceDecoder`**
//!   — packet-by-packet decode (Annex B split included) yields exactly
//!   the `decode_vp9_sequence` outputs, corpus-validated.

use oxideav_core::registry::{Decoder, Encoder};
use oxideav_core::{
    CodecCapabilities, CodecId, CodecParameters, Error as CoreError, Frame, Packet, PixelFormat,
    RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

/// Deterministic translating planar content at any geometry (matches
/// the crate's matrix-GOP test content model).
fn planar_frame_u8(w: usize, h: usize, cw: usize, ch: usize, k: usize) -> Vec<u8> {
    let f = |x: usize, y: usize, s: usize| (((x + 2 * k) * 7 + (y + k) * 13 + s) % 251) as u8;
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            px.push(f(x, y, 0));
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            px.push(f(x, y, 40));
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            px.push(f(x, y, 90));
        }
    }
    px
}

/// Wrap a packed planar buffer into a framework video frame (tight
/// strides), with a caller-chosen pts.
fn video_frame(
    planar: &[u8],
    dims: [(usize, usize); 3],
    bytes_per_sample: usize,
    pts: i64,
) -> Frame {
    let mut planes = Vec::with_capacity(3);
    let mut at = 0usize;
    for (w, h) in dims {
        let len = w * h * bytes_per_sample;
        planes.push(VideoPlane {
            stride: w * bytes_per_sample,
            data: planar[at..at + len].to_vec(),
        });
        at += len;
    }
    Frame::Video(VideoFrame {
        pts: Some(pts),
        planes,
    })
}

fn vp9_params(w: u32, h: u32, fmt: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("vp9"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(fmt);
    p
}

/// Drive `frames` through a registry encoder, asserting one packet per
/// frame, and return the coded packets.
fn encode_all(enc: &mut dyn Encoder, frames: &[Frame]) -> Vec<Packet> {
    let mut packets = Vec::new();
    for f in frames {
        enc.send_frame(f).expect("send_frame");
        packets.push(enc.receive_packet().expect("one packet per frame"));
    }
    assert!(
        matches!(enc.receive_packet(), Err(CoreError::NeedMore)),
        "no extra packets before flush"
    );
    enc.flush().expect("flush");
    assert!(
        matches!(enc.receive_packet(), Err(CoreError::Eof)),
        "Eof after flush"
    );
    packets
}

/// Drive coded packets through a registry decoder and return the
/// decoded frames.
fn decode_all(dec: &mut dyn Decoder, packets: &[Packet]) -> Vec<VideoFrame> {
    let mut out = Vec::new();
    for p in packets {
        dec.send_packet(p).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => out.push(v),
                Ok(_) => panic!("expected video frames"),
                Err(CoreError::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e:?}"),
            }
        }
    }
    dec.flush().expect("flush");
    assert!(
        matches!(dec.receive_frame(), Err(CoreError::Eof)),
        "Eof after flush"
    );
    out
}

/// `register( )` installs a decode+encode video registration under the
/// "vp9" id with the §7.2 format matrix and the wire-tag claims, and
/// both registry factories construct.
#[test]
fn register_installs_decoder_and_encoder() {
    let mut ctx = RuntimeContext::new();
    oxideav_vp9::register(&mut ctx);

    let id = CodecId::new("vp9");
    assert!(ctx.codecs.has_decoder(&id));
    assert!(ctx.codecs.has_encoder(&id));

    let impls = ctx.codecs.implementations(&id);
    assert_eq!(impls.len(), 1);
    let caps: &CodecCapabilities = &impls[0].caps;
    assert!(caps.decode && caps.encode && caps.lossy && caps.lossless);
    assert_eq!(
        caps.accepted_pixel_formats.len(),
        12,
        "the full §7.2.2 matrix: 4 geometries x 3 depths"
    );
    for fmt in [
        PixelFormat::Yuv440P,
        PixelFormat::Yuv440P10Le,
        PixelFormat::Yuv440P12Le,
    ] {
        assert!(
            caps.accepted_pixel_formats.contains(&fmt),
            "{fmt:?} is a registered VP9 format"
        );
    }

    // The encoder options schema is discoverable through the registry.
    let schema = ctx
        .codecs
        .encoder_options_schema(&id)
        .expect("encoder options schema");
    assert!(schema.iter().any(|f| f.name == "q"));
    assert!(schema.iter().any(|f| f.name == "lossless"));

    // Both factories construct through the registry lookup.
    let params = vp9_params(64, 48, PixelFormat::Yuv420P);
    ctx.codecs.first_decoder(&params).expect("decoder factory");
    ctx.codecs.first_encoder(&params).expect("encoder factory");

    // The direct factories (the dual-API convention) match.
    oxideav_vp9::make_decoder(&params).expect("direct decoder factory");
    oxideav_vp9::make_encoder(&params).expect("direct encoder factory");
}

/// The registry Encoder's packet bytes ARE the chain-framed default
/// batch entry's bytes (8-bit 4:2:0 lossy), the first packet is
/// flagged as the keyframe, pts passes through, and the registry
/// Decoder round-trips the stream to the `decode_vp9_sequence` output.
#[test]
fn registry_encoder_rides_the_chained_default_420() {
    let (w, h, cw, ch) = (64usize, 48usize, 32usize, 24usize);
    let inputs: Vec<Vec<u8>> = (0..4).map(|k| planar_frame_u8(w, h, cw, ch, k)).collect();
    let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
    let expected = oxideav_vp9::encode_vp9_lossy_sequence(&refs, 64, 48, 110).expect("batch");

    let mut params = vp9_params(64, 48, PixelFormat::Yuv420P);
    params.options.insert("q", "110");
    let mut enc = oxideav_vp9::make_encoder(&params).expect("encoder");
    let frames: Vec<Frame> = inputs
        .iter()
        .enumerate()
        .map(|(i, f)| video_frame(f, [(w, h), (cw, ch), (cw, ch)], 1, 33 * i as i64))
        .collect();
    let packets = encode_all(enc.as_mut(), &frames);

    assert_eq!(packets.len(), expected.len());
    for (i, (p, e)) in packets.iter().zip(&expected).enumerate() {
        assert_eq!(&p.data, e, "packet {i}: registry bytes != batch bytes");
        assert_eq!(p.flags.keyframe, i == 0, "packet {i} keyframe flag");
        assert_eq!(p.pts, Some(33 * i as i64), "packet {i} pts passthrough");
    }

    // Round-trip through the registry decoder == the batch decode.
    let batch_refs: Vec<&[u8]> = expected.iter().map(|f| f.as_slice()).collect();
    let batch_decoded = oxideav_vp9::decode_vp9_sequence(&batch_refs).expect("batch decode");
    let mut dec = oxideav_vp9::make_decoder(&params).expect("decoder");
    let decoded = decode_all(dec.as_mut(), &packets);
    assert_eq!(decoded.len(), batch_decoded.len());
    for (i, (v, b)) in decoded.iter().zip(&batch_decoded).enumerate() {
        assert_eq!(v.pts, Some(33 * i as i64), "frame {i} pts");
        assert_eq!(v.planes.len(), 3);
        let expected_planes = [
            (b.y.as_slice(), w),
            (b.u.as_slice(), cw),
            (b.v.as_slice(), cw),
        ];
        for (pi, (plane, (samples, pw))) in v.planes.iter().zip(expected_planes).enumerate() {
            assert_eq!(plane.stride, pw, "frame {i} plane {pi} stride");
            let bytes: Vec<u8> = samples.iter().map(|&s| s as u8).collect();
            assert_eq!(plane.data, bytes, "frame {i} plane {pi} samples");
        }
    }
}

/// `lossless=true` rides the chain-framed lossless default: packet
/// bytes equal `encode_vp9_lossless_sequence`, and the decoded output
/// equals the source byte-exact.
#[test]
fn registry_encoder_lossless_matches_default_and_roundtrips() {
    let (w, h, cw, ch) = (48usize, 32usize, 24usize, 16usize);
    let inputs: Vec<Vec<u8>> = (0..3).map(|k| planar_frame_u8(w, h, cw, ch, k)).collect();
    let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
    let expected = oxideav_vp9::encode_vp9_lossless_sequence(&refs, 48, 32).expect("batch");

    let mut params = vp9_params(48, 32, PixelFormat::Yuv420P);
    params.options.insert("lossless", "true");
    let mut enc = oxideav_vp9::make_encoder(&params).expect("encoder");
    let frames: Vec<Frame> = inputs
        .iter()
        .enumerate()
        .map(|(i, f)| video_frame(f, [(w, h), (cw, ch), (cw, ch)], 1, i as i64))
        .collect();
    let packets = encode_all(enc.as_mut(), &frames);
    assert_eq!(packets.len(), expected.len());
    for (i, (p, e)) in packets.iter().zip(&expected).enumerate() {
        assert_eq!(&p.data, e, "packet {i}: registry bytes != batch bytes");
    }

    let mut dec = oxideav_vp9::make_decoder(&params).expect("decoder");
    let decoded = decode_all(dec.as_mut(), &packets);
    assert_eq!(decoded.len(), inputs.len());
    for (i, (v, src)) in decoded.iter().zip(&inputs).enumerate() {
        let packed: Vec<u8> = v.planes.iter().flat_map(|p| p.data.clone()).collect();
        assert_eq!(&packed, src, "frame {i}: lossless round-trip byte-exact");
    }
}

/// The format matrix maps through the registry: a 10-bit 4:2:0 GOP
/// (Yuv420P10Le — §8.10 little-endian pairs) and an 8-bit 4:4:4 GOP
/// each produce the matching batch entry's exact bytes.
#[test]
fn registry_encoder_covers_the_format_matrix() {
    // 10-bit 4:2:0 (profile 2).
    {
        let (w, h, cw, ch) = (32usize, 24usize, 16usize, 12usize);
        let inputs: Vec<Vec<u16>> = (0..3)
            .map(|k| {
                planar_frame_u8(w, h, cw, ch, k)
                    .into_iter()
                    .map(|b| u16::from(b) << 2)
                    .collect()
            })
            .collect();
        let refs: Vec<&[u16]> = inputs.iter().map(|f| f.as_slice()).collect();
        let expected =
            oxideav_vp9::encode_vp9_lossy_sequence_hbd(&refs, 32, 24, 10, true, 120).expect("hbd");

        let mut params = vp9_params(32, 24, PixelFormat::Yuv420P10Le);
        params.options.insert("q", "120");
        let mut enc = oxideav_vp9::make_encoder(&params).expect("encoder");
        let frames: Vec<Frame> = inputs
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let bytes: Vec<u8> = f.iter().flat_map(|s| s.to_le_bytes()).collect();
                video_frame(&bytes, [(w, h), (cw, ch), (cw, ch)], 2, i as i64)
            })
            .collect();
        let packets = encode_all(enc.as_mut(), &frames);
        assert_eq!(packets.len(), expected.len());
        for (i, (p, e)) in packets.iter().zip(&expected).enumerate() {
            assert_eq!(&p.data, e, "hbd packet {i}");
        }

        // Decode side: 10-bit planes come back as LE pairs.
        let mut dec = oxideav_vp9::make_decoder(&params).expect("decoder");
        let decoded = decode_all(dec.as_mut(), &packets);
        assert_eq!(decoded.len(), inputs.len());
        assert_eq!(decoded[0].planes[0].stride, w * 2, "16-bit luma stride");
    }

    // 8-bit 4:4:4 (profile 1).
    {
        let (w, h) = (32usize, 24usize);
        let inputs: Vec<Vec<u8>> = (0..3).map(|k| planar_frame_u8(w, h, w, h, k)).collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        let expected = oxideav_vp9::encode_vp9_lossy_sequence_444(&refs, 32, 24, 120).expect("444");

        let mut params = vp9_params(32, 24, PixelFormat::Yuv444P);
        params.options.insert("q", "120");
        let mut enc = oxideav_vp9::make_encoder(&params).expect("encoder");
        let frames: Vec<Frame> = inputs
            .iter()
            .enumerate()
            .map(|(i, f)| video_frame(f, [(w, h), (w, h), (w, h)], 1, i as i64))
            .collect();
        let packets = encode_all(enc.as_mut(), &frames);
        assert_eq!(packets.len(), expected.len());
        for (i, (p, e)) in packets.iter().zip(&expected).enumerate() {
            assert_eq!(&p.data, e, "444 packet {i}");
        }
    }
}

/// Strided input (rows padded beyond the visible width) de-strides to
/// the same bytes as tight input — the plane walk honours `stride`.
#[test]
fn registry_encoder_destrides_padded_planes() {
    let (w, h, cw, ch) = (24usize, 16usize, 12usize, 8usize);
    let planar = planar_frame_u8(w, h, cw, ch, 1);
    let tight = video_frame(&planar, [(w, h), (cw, ch), (cw, ch)], 1, 0);

    // Re-lay each plane with 5 bytes of per-row padding garbage.
    let pad = 5usize;
    let mut planes = Vec::new();
    let mut at = 0usize;
    for (pw, ph) in [(w, h), (cw, ch), (cw, ch)] {
        let mut data = Vec::with_capacity((pw + pad) * ph);
        for row in 0..ph {
            data.extend_from_slice(&planar[at + row * pw..at + (row + 1) * pw]);
            data.extend(std::iter::repeat_n(0xEE, pad));
        }
        at += pw * ph;
        planes.push(VideoPlane {
            stride: pw + pad,
            data,
        });
    }
    let strided = Frame::Video(VideoFrame {
        pts: Some(0),
        planes,
    });

    let mut params = vp9_params(24, 16, PixelFormat::Yuv420P);
    params.options.insert("q", "140");
    let mut enc_a = oxideav_vp9::make_encoder(&params).expect("encoder a");
    let mut enc_b = oxideav_vp9::make_encoder(&params).expect("encoder b");
    enc_a.send_frame(&tight).expect("tight");
    enc_b.send_frame(&strided).expect("strided");
    assert_eq!(
        enc_a.receive_packet().expect("a").data,
        enc_b.receive_packet().expect("b").data,
        "stride padding must not leak into the encode"
    );
}

/// Construction rejections: q out of range, lossless at a non-4:2:0
/// format, an unmappable pixel format, an unknown option key, and a
/// missing geometry.
#[test]
fn registry_encoder_rejects_bad_parameters() {
    let base = |fmt| vp9_params(32, 24, fmt);

    let mut p = base(PixelFormat::Yuv420P);
    p.options.insert("q", "0");
    assert!(oxideav_vp9::make_encoder(&p).is_err(), "q = 0");

    let mut p = base(PixelFormat::Yuv420P);
    p.options.insert("q", "256");
    assert!(oxideav_vp9::make_encoder(&p).is_err(), "q = 256");

    let mut p = base(PixelFormat::Yuv444P);
    p.options.insert("lossless", "true");
    assert!(
        oxideav_vp9::make_encoder(&p).is_err(),
        "lossless GOP path is 4:2:0 only"
    );

    let p = base(PixelFormat::Rgb24);
    assert!(oxideav_vp9::make_encoder(&p).is_err(), "unmappable format");

    let mut p = base(PixelFormat::Yuv420P);
    p.options.insert("nonsense", "1");
    assert!(oxideav_vp9::make_encoder(&p).is_err(), "unknown option key");

    let mut p = base(PixelFormat::Yuv420P);
    p.width = None;
    assert!(oxideav_vp9::make_encoder(&p).is_err(), "missing width");
}

/// `reset( )` wipes the cross-frame state: after a reset the decoder
/// accepts a fresh keyframe-led stream (and refuses to resume
/// mid-stream state it no longer has).
#[test]
fn registry_decoder_reset_starts_a_fresh_stream() {
    let (w, h, cw, ch) = (32usize, 24usize, 16usize, 12usize);
    let inputs: Vec<Vec<u8>> = (0..3).map(|k| planar_frame_u8(w, h, cw, ch, k)).collect();
    let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
    let coded = oxideav_vp9::encode_vp9_lossy_sequence(&refs, 32, 24, 120).expect("batch");
    let packets: Vec<Packet> = coded
        .iter()
        .map(|f| Packet::new(0, TimeBase::MILLIS, f.clone()))
        .collect();

    let params = vp9_params(32, 24, PixelFormat::Yuv420P);
    let mut dec = oxideav_vp9::make_decoder(&params).expect("decoder");
    dec.send_packet(&packets[0]).expect("keyframe");
    dec.send_packet(&packets[1]).expect("p-frame");
    dec.reset().expect("reset");
    // After reset the P-frame's reference state is gone…
    assert!(
        dec.send_packet(&packets[2]).is_err(),
        "a mid-stream P-frame must not decode against wiped state"
    );
    // …but a fresh reset + keyframe-led stream decodes fully.
    dec.reset().expect("reset again");
    let mut n = 0;
    for p in &packets {
        dec.send_packet(p).expect("fresh stream");
        while let Ok(f) = dec.receive_frame() {
            let _ = f;
            n += 1;
        }
    }
    assert_eq!(n, packets.len(), "all frames decode after reset");
}

/// Corpus sweep (docs-gated): every staged multi-frame fixture decodes
/// packet-by-packet through the registry Decoder to exactly the
/// `decode_vp9_sequence` outputs — IVF chunks (superframes included)
/// map to packets, so the §B.2 in-decoder split and the §8.9
/// `show_existing_frame` path are corpus-validated at the registry
/// surface. Every §7.2.2 triple in the corpus — the 4:4:0 fixtures
/// included — decodes with the framework label the decoder reports
/// matching the stream's `(BitDepth, ssx, ssy)` triple, and every
/// plane's extent equal to that label's own `plane_dimensions( )`.
#[test]
fn registry_decoder_matches_batch_decode_on_the_corpus() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let mut swept = 0usize;
    let mut saw_440 = 0usize;
    for entry in std::fs::read_dir(root).expect("read fixtures dir") {
        let dir = entry.expect("dir entry").path();
        let ivf_path = dir.join("input.ivf");
        if !ivf_path.is_file() {
            continue;
        }
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        let ivf = std::fs::read(&ivf_path).expect("read input.ivf");
        // Minimal IVF walk: 32-byte header, then per-chunk 12-byte
        // headers (LE size + timestamp).
        assert!(ivf.len() >= 32 && &ivf[..4] == b"DKIF", "{name}: IVF");
        let mut chunks: Vec<&[u8]> = Vec::new();
        let mut at = 32usize;
        while at + 12 <= ivf.len() {
            let size =
                u32::from_le_bytes([ivf[at], ivf[at + 1], ivf[at + 2], ivf[at + 3]]) as usize;
            at += 12;
            if at + size > ivf.len() {
                break;
            }
            chunks.push(&ivf[at..at + size]);
            at += size;
        }

        // Ground truth: the batch decode over the split frames.
        let mut split: Vec<&[u8]> = Vec::new();
        for c in &chunks {
            split.extend(oxideav_vp9::split_superframe(c));
        }
        let expected = oxideav_vp9::decode_vp9_sequence(&split).expect("batch decode");

        let mut dec = oxideav_vp9::Vp9Decoder::new();
        let packets: Vec<Packet> = chunks
            .iter()
            .map(|c| Packet::new(0, TimeBase::MILLIS, c.to_vec()))
            .collect();
        let decoded = decode_all(&mut dec, &packets);
        assert_eq!(decoded.len(), expected.len(), "{name}: shown-frame count");
        let last = expected.last().expect("at least one shown frame");
        let label = oxideav_vp9::pixel_format_for_triple(
            last.bit_depth,
            last.subsampling_x,
            last.subsampling_y,
        )
        .expect("every §7.2.2 triple has a framework label");
        assert_eq!(dec.pixel_format(), Some(label), "{name}: decoder label");
        if !last.subsampling_x && last.subsampling_y {
            assert!(
                matches!(
                    label,
                    PixelFormat::Yuv440P | PixelFormat::Yuv440P10Le | PixelFormat::Yuv440P12Le
                ),
                "{name}: 4:4:0 streams carry the Yuv440P family, got {label:?}"
            );
            saw_440 += 1;
        }
        for (i, (v, b)) in decoded.iter().zip(&expected).enumerate() {
            let fmt =
                oxideav_vp9::pixel_format_for_triple(b.bit_depth, b.subsampling_x, b.subsampling_y)
                    .expect("label");
            let bps = if b.bit_depth == 8 { 1 } else { 2 };
            let planes_expected = [b.y.as_slice(), b.u.as_slice(), b.v.as_slice()];
            assert_eq!(v.image_planes().len(), 3, "{name} frame {i}: plane count");
            for (pi, (plane, samples)) in v.planes.iter().zip(planes_expected).enumerate() {
                let (pw, ph) = fmt
                    .plane_dimensions(pi, b.width, b.height)
                    .expect("plane dims");
                let expected_bytes: Vec<u8> = if b.bit_depth == 8 {
                    samples.iter().map(|&s| s as u8).collect()
                } else {
                    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
                };
                assert_eq!(
                    plane.stride,
                    pw as usize * bps,
                    "{name} frame {i} plane {pi}: stride is the label's plane width"
                );
                assert_eq!(
                    plane.data.len(),
                    (pw * ph) as usize * bps,
                    "{name} frame {i} plane {pi}: extent is the label's plane geometry"
                );
                assert_eq!(
                    plane.data, expected_bytes,
                    "{name}: frame {i} plane {pi} samples"
                );
            }
        }
        swept += 1;
    }
    assert!(swept >= 40, "corpus sweep covered {swept} fixtures");
    assert!(
        saw_440 >= 1,
        "the staged corpus carries at least one 4:4:0 stream (profile-1-yuv440-8bit-inter)"
    );
}

/// 4:4:0 at every §7.2.2 depth through the registry pair: the Encoder
/// accepts `Yuv440P` / `Yuv440P10Le` / `Yuv440P12Le` input (profile 1
/// at 8-bit, profile 3 at 10/12-bit) with packet bytes identical to
/// the public `encode_vp9_lossy_sequence_440` /
/// `encode_vp9_lossy_sequence_hbd_440` entries, and the Decoder hands
/// the frames back labelled with the same format, chroma planes at
/// the framework's full-width / half-height 4:4:0 geometry (odd
/// height: `ceil( h / 2 )` rows), sample-exact against
/// `decode_vp9_sequence`.
#[test]
fn registry_pair_carries_440_at_every_depth() {
    let (w, h) = (24usize, 19usize);
    let cases = [
        (PixelFormat::Yuv440P, 8u8),
        (PixelFormat::Yuv440P10Le, 10u8),
        (PixelFormat::Yuv440P12Le, 12u8),
    ];
    for (fmt, depth) in cases {
        assert_eq!(fmt.chroma_subsampling(), Some((0, 1)), "{fmt:?} is 4:4:0");
        let (cw, ch) = fmt
            .plane_dimensions(1, w as u32, h as u32)
            .map(|(a, b)| (a as usize, b as usize))
            .expect("chroma dims");
        assert_eq!((cw, ch), (w, h.div_ceil(2)), "{fmt:?} chroma geometry");
        assert_eq!(
            oxideav_vp9::pixel_format_for_triple(depth, false, true),
            Some(fmt)
        );

        let inputs8: Vec<Vec<u8>> = (0..3).map(|k| planar_frame_u8(w, h, cw, ch, k)).collect();
        let (expected, frames): (Vec<Vec<u8>>, Vec<Frame>) = if depth == 8 {
            let refs: Vec<&[u8]> = inputs8.iter().map(|f| f.as_slice()).collect();
            let expected =
                oxideav_vp9::encode_vp9_lossy_sequence_440(&refs, 24, 19, 120).expect("440 batch");
            let frames = inputs8
                .iter()
                .enumerate()
                .map(|(i, f)| video_frame(f, [(w, h), (cw, ch), (cw, ch)], 1, i as i64))
                .collect();
            (expected, frames)
        } else {
            let inputs: Vec<Vec<u16>> = inputs8
                .iter()
                .map(|f| f.iter().map(|&b| u16::from(b) << (depth - 8)).collect())
                .collect();
            let refs: Vec<&[u16]> = inputs.iter().map(|f| f.as_slice()).collect();
            let expected =
                oxideav_vp9::encode_vp9_lossy_sequence_hbd_440(&refs, 24, 19, depth, 120)
                    .expect("hbd 440 batch");
            let frames = inputs
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let bytes: Vec<u8> = f.iter().flat_map(|s| s.to_le_bytes()).collect();
                    video_frame(&bytes, [(w, h), (cw, ch), (cw, ch)], 2, i as i64)
                })
                .collect();
            (expected, frames)
        };

        let mut params = vp9_params(24, 19, fmt);
        params.options.insert("q", "120");
        let mut enc = oxideav_vp9::make_encoder(&params).expect("440 encoder");
        assert_eq!(enc.output_params().pixel_format, Some(fmt));
        let packets = encode_all(enc.as_mut(), &frames);
        assert_eq!(packets.len(), expected.len(), "{fmt:?}: packet count");
        for (i, (p, e)) in packets.iter().zip(&expected).enumerate() {
            assert_eq!(&p.data, e, "{fmt:?}: packet {i} bytes");
        }

        // Header pins: the stream actually signals ssx = 0, ssy = 1.
        let hdr = oxideav_vp9::parse_uncompressed_header(&packets[0].data).expect("header");
        assert!(
            !hdr.color_config.subsampling_x && hdr.color_config.subsampling_y,
            "{fmt:?}: 4:4:0 header"
        );
        assert_eq!(hdr.color_config.bit_depth, depth);

        let split: Vec<&[u8]> = packets.iter().map(|p| p.data.as_slice()).collect();
        let batch = oxideav_vp9::decode_vp9_sequence(&split).expect("batch decode");
        let mut dec = oxideav_vp9::Vp9Decoder::new();
        let decoded = decode_all(&mut dec, &packets);
        assert_eq!(dec.pixel_format(), Some(fmt), "{fmt:?}: decoder label");
        assert_eq!(decoded.len(), batch.len());
        let bps = if depth == 8 { 1 } else { 2 };
        for (i, (v, b)) in decoded.iter().zip(&batch).enumerate() {
            assert!(!b.subsampling_x && b.subsampling_y);
            assert_eq!(v.planes[0].stride, w * bps);
            assert_eq!(v.planes[1].stride, cw * bps);
            assert_eq!(v.planes[1].data.len(), cw * ch * bps, "{fmt:?} frame {i} U");
            assert_eq!(v.planes[2].data.len(), cw * ch * bps, "{fmt:?} frame {i} V");
            let pack = |s: &[u16]| -> Vec<u8> {
                if depth == 8 {
                    s.iter().map(|&x| x as u8).collect()
                } else {
                    s.iter().flat_map(|x| x.to_le_bytes()).collect()
                }
            };
            assert_eq!(v.planes[0].data, pack(&b.y), "{fmt:?} frame {i} Y");
            assert_eq!(v.planes[1].data, pack(&b.u), "{fmt:?} frame {i} U");
            assert_eq!(v.planes[2].data, pack(&b.v), "{fmt:?} frame {i} V");
        }
    }
}

/// Round-448 tile-parallel decode at the registry surface: a
/// [`oxideav_vp9::Vp9Decoder`] granted a multi-thread budget through
/// the framework `set_execution_context` hook decodes the staged
/// multi-tile streams **frame-identically** to a default (serial)
/// registry decoder — and the budget survives `reset( )` (it is stream
/// configuration, not stream state). Docs-gated on the staged corpus.
#[test]
fn registry_decoder_execution_context_is_output_invariant() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let params = vp9_params(0, 0, PixelFormat::Yuv420P);
    for name in [
        "tile-cols-2",
        "tiles-2col-inter",
        "tiles-4col-inter",
        "tiles-2col-4row-inter",
    ] {
        let ivf_path = root.join(name).join("input.ivf");
        if !ivf_path.is_file() {
            eprintln!("{name}: not staged; skipping");
            continue;
        }
        let ivf = std::fs::read(&ivf_path).expect("read input.ivf");
        assert!(ivf.len() >= 32 && &ivf[..4] == b"DKIF", "{name}: IVF");
        let mut packets: Vec<Packet> = Vec::new();
        let mut at = 32usize;
        while at + 12 <= ivf.len() {
            let size =
                u32::from_le_bytes([ivf[at], ivf[at + 1], ivf[at + 2], ivf[at + 3]]) as usize;
            at += 12;
            if at + size > ivf.len() {
                break;
            }
            packets.push(Packet::new(
                0,
                TimeBase::MILLIS,
                ivf[at..at + size].to_vec(),
            ));
            at += size;
        }

        let mut serial_dec = oxideav_vp9::make_decoder(&params).expect("serial decoder");
        let serial = decode_all(serial_dec.as_mut(), &packets);

        let mut par_dec = oxideav_vp9::Vp9Decoder::new();
        par_dec.set_execution_context(&oxideav_core::ExecutionContext::with_threads(4));
        let parallel = decode_all(&mut par_dec, &packets);
        assert_eq!(parallel.len(), serial.len(), "{name}: frame count");
        for (i, (p, s)) in parallel.iter().zip(&serial).enumerate() {
            for (pi, (pp, sp)) in p.planes.iter().zip(&s.planes).enumerate() {
                assert_eq!(
                    pp.data, sp.data,
                    "{name}: frame {i} plane {pi} differs under the thread budget"
                );
            }
        }

        // The budget survives reset( ): the re-decoded stream still
        // matches the serial output.
        par_dec.reset().expect("reset");
        let after_reset = decode_all(&mut par_dec, &packets);
        assert_eq!(after_reset.len(), serial.len(), "{name}: post-reset count");
        for (i, (p, s)) in after_reset.iter().zip(&serial).enumerate() {
            for (pp, sp) in p.planes.iter().zip(&s.planes) {
                assert_eq!(pp.data, sp.data, "{name}: post-reset frame {i} differs");
            }
        }
    }
}
