//! End-to-end decode of the bundled 128×128 keyframe intra-only VP9
//! fixture (`tests/fixtures/vp9-intra.ivf`).
//!
//! The fixture was generated with:
//!
//! ```text
//! ffmpeg -f lavfi -i "testsrc=size=128x128:rate=24:duration=0.042" \
//!        -c:v libvpx-vp9 -g 1 -keyint_min 1 -deadline realtime -f ivf \
//!        tests/fixtures/vp9-intra.ivf
//! ```
//!
//! Success criteria:
//! * The first IVF frame decodes via the high-level `Vp9Decoder` API.
//! * The produced `VideoFrame` reports 128×128, `Yuv420P`.
//! * Luma mean is in `32..=224` (avoids all-black / all-white artefacts).
//! * Distinct luma sample count is `> 20` (i.e. not a flat-fill fallback).

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

const FIXTURE: &str = "tests/fixtures/vp9-intra.ivf";

#[test]
fn decode_first_keyframe_produces_plausible_luma() {
    if !Path::new(FIXTURE).exists() {
        panic!(
            "fixture {FIXTURE} missing — regenerate with the ffmpeg \
             invocation in the module docs"
        );
    }
    let data = std::fs::read(FIXTURE).expect("read fixture");
    let (hdr, _) = ivf::parse_header(&data).expect("IVF header");
    assert_eq!(&hdr.fourcc, b"VP90");
    assert_eq!(hdr.width, 128);
    assert_eq!(hdr.height, 128);

    let frame_payload = {
        let mut it = ivf::iter_frames(&data).expect("iter frames");
        it.next()
            .expect("at least one frame")
            .expect("frame ok")
            .payload
            .to_vec()
    };

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 24), frame_payload);
    dec.send_packet(&pkt).expect("send_packet");
    let f = dec.receive_frame().expect("receive_frame");
    let v = match f {
        Frame::Video(v) => v,
        other => panic!("expected Video, got {other:?}"),
    };
    assert_eq!(v.planes.len(), 3);

    let y_plane = &v.planes[0];
    assert_eq!(y_plane.stride, 128);
    assert_eq!(y_plane.data.len(), y_plane.stride * 128);

    let sum: u64 = y_plane.data.iter().map(|&b| b as u64).sum();
    let mean = (sum / y_plane.data.len() as u64) as u8;
    let mut distinct = std::collections::HashSet::new();
    for &s in &y_plane.data {
        distinct.insert(s);
    }
    eprintln!(
        "vp9-intra fixture: luma mean = {mean}, distinct luma values = {}",
        distinct.len()
    );
    assert!(
        (32..=224).contains(&mean),
        "luma mean {mean} out of plausible range (expected 32..=224)"
    );
    assert!(
        distinct.len() > 20,
        "distinct luma samples = {} (expected > 20)",
        distinct.len()
    );
}
