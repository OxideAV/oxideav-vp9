//! End-to-end decode of a multi-tile VP9 clip.
//!
//! Fixture `tests/fixtures/vp9-multi-tile.ivf` was generated with:
//!
//! ```text
//! ffmpeg -f lavfi -i testsrc=size=640x360:rate=25 \
//!        -c:v libvpx-vp9 -tile-columns 2 -t 0.5 \
//!        -f ivf tests/fixtures/vp9-multi-tile.ivf
//! ```
//!
//! Success criteria:
//! * Keyframe decodes without surfacing `Error::Unsupported`.
//! * `log2_tile_cols >= 1` — confirming the fixture actually exercises
//!   the multi-tile path.
//! * The decoded luma plane is not constant — the decoder produced
//!   per-tile pixels rather than all zeros.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, parse_uncompressed_header, CODEC_ID_STR};

const FIXTURE: &str = "tests/fixtures/vp9-multi-tile.ivf";

#[test]
fn decode_multi_tile_keyframe() {
    if !Path::new(FIXTURE).exists() {
        panic!("fixture {FIXTURE} missing — regenerate via module docs");
    }
    let data = std::fs::read(FIXTURE).expect("read fixture");
    let (hdr, _) = ivf::parse_header(&data).expect("IVF header");
    assert_eq!(&hdr.fourcc, b"VP90");
    assert_eq!(hdr.width, 640);
    assert_eq!(hdr.height, 360);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");

    let mut iter = ivf::iter_frames(&data).expect("iter frames");
    let f1 = iter.next().expect("frame 1").expect("frame 1 ok");

    // Peek at the parsed header to confirm multi-tile.
    let parsed = parse_uncompressed_header(f1.payload, None).expect("parse header");
    assert!(
        parsed.tile_info.log2_tile_cols >= 1,
        "fixture should have log2_tile_cols >= 1, got {}",
        parsed.tile_info.log2_tile_cols
    );
    eprintln!(
        "vp9 multi-tile fixture: log2_tile_cols={} log2_tile_rows={}",
        parsed.tile_info.log2_tile_cols, parsed.tile_info.log2_tile_rows
    );

    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), f1.payload.to_vec()))
        .expect("send multi-tile frame — must not be Unsupported");
    let frame = match dec.receive_frame().expect("receive") {
        Frame::Video(v) => v,
        other => panic!("expected Video, got {other:?}"),
    };
    assert_eq!(frame.width, 640);
    assert_eq!(frame.height, 360);
    let y = &frame.planes[0].data;
    let mn = *y.iter().min().unwrap();
    let mx = *y.iter().max().unwrap();
    assert!(
        mx > mn,
        "decoded luma is flat (min={mn}, max={mx}) — tile walk likely failed"
    );
}
