//! End-to-end VP9 segmentation (§6.4.7 / §6.4.12 / §6.4.14) conformance.
//!
//! The fixture `tests/fixtures/vp9-segmentation.ivf` is a 128×128
//! 2-keyframe clip encoded with `libvpx-vp9 -aq-mode 1` — that triggers
//! adaptive-quantization segmentation:
//!
//! ```text
//! ffmpeg -y -f lavfi -i "testsrc=size=128x128:rate=24:duration=0.042" \
//!        -vf format=yuv420p \
//!        -c:v libvpx-vp9 -g 1 -keyint_min 1 -aq-mode 1 \
//!        -b:v 80k -deadline realtime -cpu-used 8 -f ivf \
//!        tests/fixtures/vp9-segmentation.ivf
//! ```
//!
//! What the fixture exercises in our crate, per the spec §-numbers cited
//! in `src/segmentation.rs`:
//!
//! * §6.2.11 `segmentation_params` — the parser must keep
//!   `segmentation_tree_probs[0..7]` and honour the per-segment
//!   `FeatureEnabled / FeatureData`.
//! * §6.4.7 `intra_segment_id()` — the keyframe path must tree-decode
//!   `segment_id` from the stream because `update_map == 1`.
//! * §6.4.9 `seg_feature_active()` + §8.6.1 `SEG_LVL_ALT_Q` — blocks
//!   stamped with segments 0/1/2/4 take the segment-specific qindex.
//! * §6.4.14 `get_segment_id()` + §8.1 step 3 — `SegmentIds` persists
//!   into the DPB so that future inter frames can look it up as
//!   `PrevSegmentIds` (here we only test intra, but the save path is
//!   still walked).
//!
//! Skip gracefully when the fixture or ffmpeg are missing — developers
//! without those tools can still run the crate's unit tests.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, parse_uncompressed_header, CODEC_ID_STR, SEG_LVL_ALT_Q};

const FIXTURE: &str = "tests/fixtures/vp9-segmentation.ivf";

#[test]
fn segmentation_clip_parses_and_decodes() {
    if !Path::new(FIXTURE).exists() {
        eprintln!("fixture {FIXTURE} missing — skipping");
        return;
    }
    let data = std::fs::read(FIXTURE).expect("read fixture");
    let (hdr, _) = ivf::parse_header(&data).expect("IVF header");
    assert_eq!(&hdr.fourcc, b"VP90");
    assert_eq!(hdr.width, 128);
    assert_eq!(hdr.height, 128);

    // --- Parse the first frame's uncompressed header and assert that
    //     the spec's segmentation fields survived §6.2.11 parsing. ---
    let mut iter = ivf::iter_frames(&data).expect("iter frames");
    let frame0 = iter.next().expect("have frame 0").expect("frame ok");
    let uh = parse_uncompressed_header(frame0.payload, None).expect("uncompressed header");
    assert!(uh.segmentation.enabled, "fixture must enable segmentation");
    assert!(
        uh.segmentation.update_map,
        "fixture keyframe must carry segmentation_update_map = 1"
    );
    // The tree_probs field must have at least one entry that isn't the
    // read_prob "no update" default (255). This asserts that our
    // §6.2.11 / §6.2.12 read_prob() path actually consumed an f(8) byte.
    let tree_non_default = uh.segmentation.tree_probs.iter().any(|&p| p != 255);
    assert!(
        tree_non_default,
        "segmentation_tree_probs should carry at least one coded byte (got {:?})",
        uh.segmentation.tree_probs
    );
    // At least one segment has SEG_LVL_ALT_Q enabled with a non-zero
    // delta — the encoder uses this to express per-segment QP.
    let any_alt_q = (0..8).any(|seg| {
        uh.segmentation.feature_enabled[seg][SEG_LVL_ALT_Q]
            && uh.segmentation.feature_data[seg][SEG_LVL_ALT_Q] != 0
    });
    assert!(
        any_alt_q,
        "expected at least one segment with a non-zero SEG_LVL_ALT_Q delta"
    );

    // --- Decode both frames end-to-end — this is the part that would
    //     desync before the §6.4.7 intra_segment_id implementation: the
    //     encoder writes tree-coded segment_id bits per block and the
    //     decoder must consume them in lockstep, otherwise subsequent
    //     bool reads (skip / tx_size / intra_mode) drift. ---
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    // Re-iterate from the start so we feed both frames including frame 0.
    let mut iter = ivf::iter_frames(&data).expect("iter frames");
    let mut decoded = 0usize;
    for (i, frame) in iter.by_ref().enumerate() {
        let frame = frame.expect("frame ok");
        let pkt = Packet::new(i as u32, TimeBase::new(1, 24), frame.payload.to_vec());
        dec.send_packet(&pkt)
            .unwrap_or_else(|e| panic!("send_packet frame {i}: {e:?}"));
        let f = dec
            .receive_frame()
            .unwrap_or_else(|e| panic!("receive_frame frame {i}: {e:?}"));
        let v = match f {
            Frame::Video(v) => v,
            other => panic!("frame {i}: expected Video, got {other:?}"),
        };
        assert_eq!(v.planes.len(), 3);
        assert_eq!(v.planes[0].stride, 128);
        assert_eq!(v.planes[0].data.len(), 128 * 128);
        // Luma must not be all the same value — catches the common
        // "segmentation misparse truncated coefficient decode" failure.
        let y = &v.planes[0].data;
        let mn = *y.iter().min().unwrap();
        let mx = *y.iter().max().unwrap();
        assert!(
            mx > mn,
            "frame {i}: luma plane is constant after decode — likely segmentation desync"
        );
        decoded += 1;
    }
    assert!(
        decoded >= 2,
        "fixture should yield at least 2 frames, got {decoded}"
    );
}
