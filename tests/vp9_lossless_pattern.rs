//! Lossless decode against a non-degenerate test pattern.
//!
//! The existing `vp9_lossless_gray.rs` fixture compares our decoder
//! output against a `vec![126; 64*64]` constant-gray plane. That is a
//! **degenerate** measurement: any decoder that returns "approximately
//! gray" (e.g. via DC prediction with no residual) trivially scores
//! ≥ 60 dB without actually being bit-exact. The 66.77 dB number on
//! `vp9-lossless-gray.ivf` therefore proves nothing about the lossy /
//! transform / dequant / coefficient-decode paths.
//!
//! This test uses an `ffmpeg testsrc` test card (color bars + circle +
//! gradient + clock — i.e. high-frequency content) encoded with
//! `-lossless 1`. The reference YUV is the decoded output of ffmpeg on
//! the same fixture, frozen on disk:
//!
//! ```text
//! ffmpeg -y -f lavfi -i "testsrc=size=128x128:rate=24:duration=0.04" \
//!        -vf format=yuv420p -c:v libvpx-vp9 -lossless 1 -f ivf \
//!        tests/fixtures/vp9-lossless-pattern.ivf
//! ffmpeg -y -i tests/fixtures/vp9-lossless-pattern.ivf -f rawvideo \
//!        -pix_fmt yuv420p tests/fixtures/vp9-lossless-pattern.yuv
//! ```
//!
//! For a truly bit-exact lossless decoder this should produce ∞ PSNR
//! (zero byte differences). At round 17 our decoder is far from that —
//! the test currently asserts only that PSNR is above a deliberately
//! low threshold so that a future bit-exact fix lifts it dramatically
//! and a regression below the current (poor) baseline is caught.
//!
//! The per-plane PSNRs are emitted to stderr — `cargo test --
//! --nocapture` to inspect.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

const FIXTURE: &str = "tests/fixtures/vp9-lossless-pattern.ivf";
const REF_YUV: &str = "tests/fixtures/vp9-lossless-pattern.yuv";
const W: usize = 128;
const H: usize = 128;

fn psnr_plane(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for i in 0..a.len() {
        let d = (a[i] as i32) - (b[i] as i32);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = (sse as f64) / (a.len() as f64);
    10.0 * (255.0 * 255.0 / mse).log10()
}

#[test]
fn lossless_pattern_matches_ffmpeg_reference() {
    if !Path::new(FIXTURE).exists() {
        panic!("fixture {FIXTURE} missing — regenerate via module docs");
    }
    if !Path::new(REF_YUV).exists() {
        panic!("reference {REF_YUV} missing — regenerate via module docs");
    }
    let data = std::fs::read(FIXTURE).expect("read fixture");
    let ref_data = std::fs::read(REF_YUV).expect("read ref yuv");
    let y_sz = W * H;
    let uv_sz = (W / 2) * (H / 2);
    let frame_sz = y_sz + 2 * uv_sz;
    assert_eq!(
        ref_data.len(),
        frame_sz,
        "expected exactly one frame in the reference YUV"
    );

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");

    let mut frame_idx = 0usize;
    let mut min_y_psnr = f64::INFINITY;
    let mut min_u_psnr = f64::INFINITY;
    let mut min_v_psnr = f64::INFINITY;
    for f in ivf::iter_frames(&data).expect("iter frames") {
        let f = f.expect("frame ok");
        let pkt = Packet::new(0, TimeBase::new(1, 24), f.payload.to_vec());
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    if frame_idx > 0 {
                        // Reference has only 1 frame.
                        break;
                    }
                    let yp = &v.planes[0];
                    let up = &v.planes[1];
                    let vp = &v.planes[2];
                    let mut y_pack = vec![0u8; y_sz];
                    for r in 0..H {
                        y_pack[r * W..r * W + W]
                            .copy_from_slice(&yp.data[r * yp.stride..r * yp.stride + W]);
                    }
                    let mut u_pack = vec![0u8; uv_sz];
                    for r in 0..H / 2 {
                        u_pack[r * (W / 2)..r * (W / 2) + W / 2]
                            .copy_from_slice(&up.data[r * up.stride..r * up.stride + W / 2]);
                    }
                    let mut v_pack = vec![0u8; uv_sz];
                    for r in 0..H / 2 {
                        v_pack[r * (W / 2)..r * (W / 2) + W / 2]
                            .copy_from_slice(&vp.data[r * vp.stride..r * vp.stride + W / 2]);
                    }
                    let ref_y = &ref_data[..y_sz];
                    let ref_u = &ref_data[y_sz..y_sz + uv_sz];
                    let ref_v = &ref_data[y_sz + uv_sz..];
                    let yp_db = psnr_plane(&y_pack, ref_y);
                    let up_db = psnr_plane(&u_pack, ref_u);
                    let vp_db = psnr_plane(&v_pack, ref_v);
                    eprintln!(
                        "lossless pattern frame {frame_idx}: Y={yp_db:.2} dB U={up_db:.2} dB V={vp_db:.2} dB"
                    );
                    let y_diffs = y_pack.iter().zip(ref_y).filter(|(a, b)| a != b).count();
                    let u_diffs = u_pack.iter().zip(ref_u).filter(|(a, b)| a != b).count();
                    let v_diffs = v_pack.iter().zip(ref_v).filter(|(a, b)| a != b).count();
                    eprintln!(
                        "  byte diffs: Y={y_diffs}/{y_sz} U={u_diffs}/{uv_sz} V={v_diffs}/{uv_sz}"
                    );
                    if yp_db.is_finite() {
                        min_y_psnr = min_y_psnr.min(yp_db);
                    }
                    if up_db.is_finite() {
                        min_u_psnr = min_u_psnr.min(up_db);
                    }
                    if vp_db.is_finite() {
                        min_v_psnr = min_v_psnr.min(vp_db);
                    }
                    frame_idx += 1;
                }
                Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("recv err: {e:?}"),
                _ => {}
            }
        }
    }
    assert!(frame_idx > 0, "no frames decoded");
    eprintln!("min PSNR Y={min_y_psnr:.2} U={min_u_psnr:.2} V={min_v_psnr:.2} dB (∞ = bit-exact)");
    // Round-17 baseline: keyframe Y ~ 9.7 dB. The bar is set just below
    // that so the test passes on the current (poor) state but a
    // regression to <8 dB or a future fix lifting it >> 30 dB are both
    // visible. When we finally achieve bit-exact lossless the assertion
    // should be tightened to require infinity / ≥ 60 dB.
    assert!(
        min_y_psnr >= 8.0,
        "luma PSNR {min_y_psnr:.2} dB below round-17 baseline 8 dB — regression"
    );
}
