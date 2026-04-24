//! Measure luma PSNR of the compound-fixture decode against an ffmpeg
//! reference YUV file.
//!
//! Generate the reference once with:
//! ```text
//! ffmpeg -y -i tests/fixtures/vp9-compound.ivf -f rawvideo -pix_fmt yuv420p \
//!     /tmp/vp9-compound-ref.yuv
//! ```
//! (8 frames @ 192x128 yuv420p).
//!
//! The test prints the per-frame and mean PSNR. It asserts only that it
//! completes — treat the numbers in stderr as the pass/fail signal.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

const FIXTURE: &str = "tests/fixtures/vp9-compound.ivf";
const REF_YUV: &str = "/tmp/vp9-compound-ref.yuv";
const W: usize = 192;
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
fn measure_compound_luma_psnr() {
    if !Path::new(FIXTURE).exists() {
        eprintln!("fixture {FIXTURE} missing — skipping");
        return;
    }
    if !Path::new(REF_YUV).exists() {
        eprintln!("reference YUV {REF_YUV} missing — skipping");
        return;
    }
    let ref_data = std::fs::read(REF_YUV).expect("read ref yuv");
    let y_sz = W * H;
    let uv_sz = (W / 2) * (H / 2);
    let frame_sz = y_sz + 2 * uv_sz;
    let n_ref = ref_data.len() / frame_sz;
    eprintln!("ref frames: {n_ref}");

    let data = std::fs::read(FIXTURE).expect("read fixture");
    let (_hdr, _) = ivf::parse_header(&data).expect("IVF header");

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");

    let mut frame_idx = 0usize;
    let mut total_y_psnr = 0.0f64;
    let mut counted = 0usize;
    for frame in ivf::iter_frames(&data).expect("iter frames") {
        let frame = frame.expect("frame ok");
        let pkt = Packet::new(0, TimeBase::new(1, 24), frame.payload.to_vec());
        let _ = dec.send_packet(&pkt);
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(f)) => {
                    if frame_idx >= n_ref {
                        eprintln!("more decoded frames than reference");
                        break;
                    }
                    let off = frame_idx * frame_sz;
                    let ref_y = &ref_data[off..off + y_sz];
                    let ref_u = &ref_data[off + y_sz..off + y_sz + uv_sz];
                    let ref_v = &ref_data[off + y_sz + uv_sz..off + frame_sz];

                    // Repack our frame into contiguous planes for PSNR.
                    let y_plane = &f.planes[0];
                    let u_plane = &f.planes[1];
                    let v_plane = &f.planes[2];
                    let mut our_y = vec![0u8; y_sz];
                    for r in 0..H {
                        let s = r * y_plane.stride;
                        let d = r * W;
                        our_y[d..d + W].copy_from_slice(&y_plane.data[s..s + W]);
                    }
                    let mut our_u = vec![0u8; uv_sz];
                    for r in 0..H / 2 {
                        let s = r * u_plane.stride;
                        let d = r * (W / 2);
                        our_u[d..d + W / 2].copy_from_slice(&u_plane.data[s..s + W / 2]);
                    }
                    let mut our_v = vec![0u8; uv_sz];
                    for r in 0..H / 2 {
                        let s = r * v_plane.stride;
                        let d = r * (W / 2);
                        our_v[d..d + W / 2].copy_from_slice(&v_plane.data[s..s + W / 2]);
                    }
                    let yp = psnr_plane(&our_y, ref_y);
                    let up = psnr_plane(&our_u, ref_u);
                    let vp = psnr_plane(&our_v, ref_v);
                    eprintln!(
                        "frame {frame_idx}: Y={yp:.2} dB U={up:.2} dB V={vp:.2} dB"
                    );
                    if yp.is_finite() {
                        total_y_psnr += yp;
                        counted += 1;
                    }
                    frame_idx += 1;
                }
                Ok(_) => {}
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => {
                    eprintln!("recv: {e:?}");
                    break;
                }
            }
        }
    }
    if counted > 0 {
        let mean = total_y_psnr / counted as f64;
        eprintln!("mean luma PSNR over {counted} frames: {mean:.2} dB");
    }
}
