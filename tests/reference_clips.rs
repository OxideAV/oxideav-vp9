//! Integration tests against ffmpeg-generated VP9 reference clips.
//!
//! Fixtures (skipped if missing — CI without ffmpeg still passes):
//!   /tmp/vp9.ivf  (IVF container, 64x64, 24fps, ~3 frames)
//!   /tmp/vp9.mp4  (MP4 / vp09 sample entry, same content)
//!
//! Generate them with:
//!   ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=0.1" \
//!          -c:v libvpx-vp9 -keyint_min 1 -g 1 -f ivf /tmp/vp9.ivf
//!   ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=0.1" \
//!          -c:v libvpx-vp9 -keyint_min 1 -g 1 /tmp/vp9.mp4

use std::path::Path;

use oxideav_vp9::ivf;
use oxideav_vp9::{parse_uncompressed_header, FrameType};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

/// Walk an IVF stream and pull out the first encoded frame's payload.
fn first_ivf_frame(buf: &[u8]) -> &[u8] {
    // IVF header is 32 bytes:
    //   0..4   "DKIF"
    //   4..6   version
    //   6..8   header length
    //   8..12  fourcc
    //   12..14 width
    //   14..16 height
    //   16..20 framerate numerator
    //   20..24 framerate denominator
    //   24..28 frame count
    //   28..32 reserved
    // Each frame: 4 bytes size LE + 8 bytes pts LE + payload.
    assert!(&buf[..4] == b"DKIF", "bad IVF magic");
    let frame_size = u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]) as usize;
    &buf[44..44 + frame_size]
}

#[test]
fn parse_ivf_first_frame_header() {
    let Some(data) = read_fixture("/tmp/vp9.ivf") else {
        return;
    };
    let frame = first_ivf_frame(&data);
    let h = parse_uncompressed_header(frame, None).expect("parse uncompressed header");
    assert_eq!(h.frame_type, FrameType::Key);
    assert_eq!(h.width, 64);
    assert_eq!(h.height, 64);
    assert_eq!(h.color_config.bit_depth, 8);
    assert!(h.show_frame);
}

#[test]
fn ivf_iterator_walks_every_frame() {
    let Some(data) = read_fixture("/tmp/vp9.ivf") else {
        return;
    };
    let (hdr, _) = ivf::parse_header(&data).expect("parse IVF header");
    assert_eq!(&hdr.fourcc, b"VP90");
    assert_eq!(hdr.width, 64);
    assert_eq!(hdr.height, 64);
    let mut n = 0;
    for f in ivf::iter_frames(&data).unwrap() {
        let f = f.expect("frame");
        assert!(!f.payload.is_empty());
        n += 1;
    }
    assert!(n >= 1, "expected at least one frame in fixture");
}

#[test]
fn ivf_first_frame_keyframe_header_via_iter() {
    // The iterator-based path must agree with the manual offset-based one.
    let Some(data) = read_fixture("/tmp/vp9.ivf") else {
        return;
    };
    let mut it = ivf::iter_frames(&data).unwrap();
    let f = it.next().unwrap().unwrap();
    let h = parse_uncompressed_header(f.payload, None).expect("parse");
    assert_eq!(h.frame_type, FrameType::Key);
    assert_eq!(h.width, 64);
    assert_eq!(h.height, 64);
}

#[test]
fn keyframe_partition_walk_completes() {
    // End-to-end: IVF demux → uncompressed header → compressed header →
    // tile walk. The tile walker exercises the §6.4.2 partition
    // quadtree against real VP9 default probabilities (§10.5) and the
    // bool decoder (§9.2). Now that block decode is wired the walk
    // completes without surfacing Unsupported.
    use oxideav_vp9::compressed_header::parse_compressed_header;
    use oxideav_vp9::tile::decode_tiles;

    let Some(data) = read_fixture("/tmp/vp9.ivf") else {
        return;
    };
    let frame = ivf::iter_frames(&data).unwrap().next().unwrap().unwrap();
    let h = parse_uncompressed_header(frame.payload, None).expect("uncompressed");
    assert_eq!(h.frame_type, FrameType::Key);
    let cmp_start = h.uncompressed_header_size;
    let cmp_end = cmp_start + h.header_size as usize;
    assert!(cmp_end <= frame.payload.len(), "compressed header fits");
    let ch =
        parse_compressed_header(&frame.payload[cmp_start..cmp_end], &h).expect("compressed header");
    let tile_payload = &frame.payload[cmp_end..];
    decode_tiles(tile_payload, &h, &ch).expect("tile decode");
}
