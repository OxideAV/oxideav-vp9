//! End-to-end decode of a 2-frame VP9 clip (1 keyframe + 1 P frame).
//!
//! Fixture `tests/fixtures/vp9-p.ivf` was generated with:
//!
//! ```text
//! ffmpeg -y -f lavfi -i "testsrc=size=128x128:rate=24:duration=0.083" \
//!        -vf format=yuv420p -c:v libvpx-vp9 -g 2 -keyint_min 2 \
//!        -deadline realtime -f ivf tests/fixtures/vp9-p.ivf
//! ```
//!
//! Success criteria:
//! * Both frames decode without surfacing `Error::Unsupported`.
//! * Frame 1 (key) comes out 128×128 with `Yuv420P`.
//! * Frame 2 (P) comes out 128×128 with `Yuv420P` — proving the inter
//!   path runs end-to-end (MV decode + sub-pel MC + residual add +
//!   DPB refresh).
//! * The two frames' luma planes are **not byte-identical**: we require
//!   at least one differing luma sample between them. This guards
//!   against a regression to the "inter = copy frame 1 verbatim"
//!   shortcut.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

const FIXTURE: &str = "tests/fixtures/vp9-p.ivf";

#[test]
fn decode_keyframe_then_p_frame() {
    if !Path::new(FIXTURE).exists() {
        panic!("fixture {FIXTURE} missing — regenerate via module docs");
    }
    let data = std::fs::read(FIXTURE).expect("read fixture");
    let (hdr, _) = ivf::parse_header(&data).expect("IVF header");
    assert_eq!(&hdr.fourcc, b"VP90");
    assert_eq!(hdr.width, 128);
    assert_eq!(hdr.height, 128);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");

    let mut iter = ivf::iter_frames(&data).expect("iter frames");
    // Frame 1 (key).
    let f1 = iter.next().expect("frame 1").expect("frame 1 ok");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 24), f1.payload.to_vec()))
        .expect("send frame 1");
    let frame1 = match dec.receive_frame().expect("receive frame 1") {
        Frame::Video(v) => v,
        other => panic!("frame 1: expected Video, got {other:?}"),
    };
    assert_eq!(frame1.width, 128);
    assert_eq!(frame1.height, 128);
    assert_eq!(frame1.planes.len(), 3);

    // Frame 2 (P).
    let f2 = iter.next().expect("frame 2").expect("frame 2 ok");
    dec.send_packet(&Packet::new(1, TimeBase::new(1, 24), f2.payload.to_vec()))
        .expect("send frame 2 — must not be Unsupported");
    let frame2 = match dec.receive_frame().expect("receive frame 2") {
        Frame::Video(v) => v,
        other => panic!("frame 2: expected Video, got {other:?}"),
    };
    assert_eq!(frame2.width, 128);
    assert_eq!(frame2.height, 128);
    assert_eq!(frame2.planes.len(), 3);

    // Pixel-level difference: require >= 1 luma sample to differ. A
    // lenient bar — the P frame might even coincide bit-exact with
    // frame 1 if the encoder emitted 0 MV and 0 residual, but for
    // real libvpx output against our default-probability context-0
    // decoder the two will diverge.
    let y1 = &frame1.planes[0].data;
    let y2 = &frame2.planes[0].data;
    assert_eq!(y1.len(), y2.len());
    let diffs = y1.iter().zip(y2.iter()).filter(|(a, b)| a != b).count();
    eprintln!("vp9-p fixture: luma samples differing between f1 and f2 = {diffs}");
    assert!(
        diffs > 0,
        "frame 2 identical to frame 1 — inter decode produced no change"
    );
}
