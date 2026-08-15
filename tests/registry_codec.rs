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
    assert_eq!(caps.accepted_pixel_formats.len(), 9, "the §7.2 matrix");

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
/// surface. Fixtures whose §7.2.2 triple has no framework pixel-format
/// label (4:4:0) are asserted to fail with `Unsupported` instead.
#[test]
fn registry_decoder_matches_batch_decode_on_the_corpus() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let mut swept = 0usize;
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

        let params = vp9_params(0, 0, PixelFormat::Yuv420P);
        let mut dec = oxideav_vp9::make_decoder(&params).expect("decoder");
        let is_440 = expected
            .first()
            .map(|f| !f.subsampling_x && f.subsampling_y)
            .unwrap_or(false);
        if is_440 {
            let err = chunks.iter().find_map(|c| {
                dec.send_packet(&Packet::new(0, TimeBase::MILLIS, c.to_vec()))
                    .err()
            });
            assert!(
                matches!(err, Some(CoreError::Unsupported(_))),
                "{name}: 4:4:0 must surface Unsupported"
            );
            continue;
        }
        let packets: Vec<Packet> = chunks
            .iter()
            .map(|c| Packet::new(0, TimeBase::MILLIS, c.to_vec()))
            .collect();
        let decoded = decode_all(dec.as_mut(), &packets);
        assert_eq!(decoded.len(), expected.len(), "{name}: shown-frame count");
        for (i, (v, b)) in decoded.iter().zip(&expected).enumerate() {
            let (cw, _) = if b.subsampling_x {
                ((b.width as usize).div_ceil(2), 0)
            } else {
                (b.width as usize, 0)
            };
            let dims = [
                (b.y.as_slice(), b.width as usize),
                (b.u.as_slice(), cw),
                (b.v.as_slice(), cw),
            ];
            for (pi, (plane, (samples, pw))) in v.planes.iter().zip(dims).enumerate() {
                let expected_bytes: Vec<u8> = if b.bit_depth == 8 {
                    samples.iter().map(|&s| s as u8).collect()
                } else {
                    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
                };
                let width_bytes = pw * if b.bit_depth == 8 { 1 } else { 2 };
                assert_eq!(plane.stride, width_bytes, "{name} frame {i} plane {pi}");
                assert_eq!(
                    plane.data, expected_bytes,
                    "{name}: frame {i} plane {pi} samples"
                );
            }
        }
        swept += 1;
    }
    assert!(swept >= 40, "corpus sweep covered {swept} fixtures");
}
