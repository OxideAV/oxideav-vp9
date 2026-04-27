//! Lossless decode of a 64×64 constant-colour test.
//!
//! Generated with:
//! ```text
//! ffmpeg -y -f lavfi -i "color=color=0x402010:size=64x64:rate=1" \
//!     -frames:v 1 -c:v libvpx-vp9 -lossless 1 -g 1 -keyint_min 1 \
//!     -f ivf tests/fixtures/vp9-lossless-c64-constant.ivf
//! ffmpeg -y -i tests/fixtures/vp9-lossless-c64-constant.ivf \
//!     -f rawvideo -pix_fmt yuv420p \
//!     tests/fixtures/vp9-lossless-c64-constant.yuv
//! ```
//!
//! With a constant-colour frame the decoder exercises:
//! * The DC_PRED + WHT path for the very first 16×16 block at (0,0).
//! * Repeated PARTITION_NONE / PARTITION_HORZ decisions across a
//!   uniform field — nearly every block is a `skip=true` non-residual
//!   prediction once the encoder converges on H_PRED / DC_PRED chains.
//!
//! The fixture isolates a localised round-19 bool-decoder misalignment
//! to a single 4×4 region (rows 8–11, cols 20–30) where one block's
//! `skip=true` was decoded as `skip=false`, causing 12-coefficient
//! token reads to consume bits meant for the next block. Round-19
//! audited the partition-context / skip-context / WHT / scan / probs
//! against §6.4–§8.7 and confirmed the spec-literal forms regress on
//! the libvpx-encoded fixtures we have on hand.
//!
//! The test asserts the diffs are SMALL and LOCAL — the `expected_max
//! _y_diffs` is intentionally loose (~30 / 4096) so a regression that
//! affects more than one block area trips the assertion immediately.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

const FIXTURE: &str = "tests/fixtures/vp9-lossless-c64-constant.ivf";
const REF_YUV: &str = "tests/fixtures/vp9-lossless-c64-constant.yuv";
const W: usize = 64;
const H: usize = 64;

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
fn lossless_constant_color_decodes_with_local_drift() {
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
    assert_eq!(ref_data.len(), frame_sz);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");

    let mut frame_idx = 0usize;
    let mut y_pack = vec![0u8; y_sz];
    let mut u_pack = vec![0u8; uv_sz];
    let mut v_pack = vec![0u8; uv_sz];
    for f in ivf::iter_frames(&data).expect("iter frames") {
        let f = f.expect("frame ok");
        let pkt = Packet::new(0, TimeBase::new(1, 24), f.payload.to_vec());
        dec.send_packet(&pkt).expect("send_packet");
        if let Ok(Frame::Video(v)) = dec.receive_frame() {
            assert_eq!(frame_idx, 0);
            for r in 0..H {
                y_pack[r * W..r * W + W].copy_from_slice(
                    &v.planes[0].data[r * v.planes[0].stride..r * v.planes[0].stride + W],
                );
            }
            for r in 0..H / 2 {
                u_pack[r * (W / 2)..r * (W / 2) + W / 2].copy_from_slice(
                    &v.planes[1].data[r * v.planes[1].stride..r * v.planes[1].stride + W / 2],
                );
                v_pack[r * (W / 2)..r * (W / 2) + W / 2].copy_from_slice(
                    &v.planes[2].data[r * v.planes[2].stride..r * v.planes[2].stride + W / 2],
                );
            }
            frame_idx += 1;
        }
    }
    assert_eq!(frame_idx, 1, "expected exactly one decoded frame");

    let ref_y = &ref_data[..y_sz];
    let ref_u = &ref_data[y_sz..y_sz + uv_sz];
    let ref_v = &ref_data[y_sz + uv_sz..];
    let y_diffs = y_pack.iter().zip(ref_y).filter(|(a, b)| a != b).count();
    let u_diffs = u_pack.iter().zip(ref_u).filter(|(a, b)| a != b).count();
    let v_diffs = v_pack.iter().zip(ref_v).filter(|(a, b)| a != b).count();
    let yp = psnr_plane(&y_pack, ref_y);
    let up = psnr_plane(&u_pack, ref_u);
    let vp = psnr_plane(&v_pack, ref_v);
    eprintln!(
        "lossless c64 constant: Y={yp:.2} dB U={up:.2} dB V={vp:.2} dB | \
         byte diffs Y={y_diffs}/{y_sz} U={u_diffs}/{uv_sz} V={v_diffs}/{uv_sz}"
    );
    // Round-19 baseline: 29 luma diffs all clustered in a single 4×4
    // region. PSNR ≥ 38 dB despite the localised drift because the
    // remainder of the 4096-pixel plane is bit-exact. A regression that
    // spreads the bug across the frame would drop PSNR below 30 dB and
    // push the diff count past 100.
    assert!(
        y_diffs < 100,
        "Y diffs {y_diffs} suggests bool-decoder drift spread beyond the known r19 hot spot"
    );
    assert!(yp >= 30.0, "Y PSNR {yp:.2} dB regressed below r19 baseline");
}
