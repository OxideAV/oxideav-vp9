//! Round-2 encoder PSNR gate: 256×256 smooth gradient, ffmpeg cross-decode.
//!
//! The pixel encoder (forward 4×4 DCT + quantise + VP9 token coding) must
//! achieve PSNR_Y ≥ 35 dB on a 256×256 smooth-gradient fixture at
//! base_q_idx=64. The fixture is a linear gradient:
//!   Y[r][c] = clamp(128 + (r - 128 + c - 128) / 4, 0, 255)
//! with uniform U=V=128 chroma.
//!
//! Cross-validation: the encoded IVF is decoded through ffmpeg and the
//! resulting raw YUV is compared against the source.

use std::process::Command;

use oxideav_vp9::encoder::{encode_keyframe_yuv, EncoderParams, YuvFrame};

fn smooth_256x256() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 256 * 256];
    for r in 0..256usize {
        for c in 0..256usize {
            let v = (128i32 + ((r as i32 - 128) + (c as i32 - 128)) / 4).clamp(0, 255);
            y[r * 256 + c] = v as u8;
        }
    }
    let u = vec![128u8; 128 * 128];
    let v = vec![128u8; 128 * 128];
    (y, u, v)
}

fn build_ivf(width: u16, height: u16, frame: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"DKIF");
    v.extend_from_slice(&0u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"VP90");
    v.extend_from_slice(&width.to_le_bytes());
    v.extend_from_slice(&height.to_le_bytes());
    v.extend_from_slice(&30u32.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes());
    v.extend_from_slice(frame);
    v
}

fn psnr(reference: &[u8], coded: &[u8]) -> f64 {
    assert_eq!(reference.len(), coded.len());
    let n = reference.len() as f64;
    let mse: f64 = reference
        .iter()
        .zip(coded.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / n;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// Round-2 encoder PSNR gate: 256×256 smooth gradient through ffmpeg cross-decode.
/// PSNR_Y ≥ 35 dB hard gate.
#[test]
fn encoder_256x256_smooth_psnr_via_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping — ffmpeg not available");
        return;
    }

    let (y, u, v) = smooth_256x256();
    let src = YuvFrame {
        y: &y,
        y_stride: 256,
        u: &u,
        v: &v,
        uv_stride: 128,
        width: 256,
        height: 256,
    };
    let p = EncoderParams::keyframe(256, 256);
    let frame = encode_keyframe_yuv(&p, &src);
    let ivf = build_ivf(256, 256, &frame);

    let dir = std::env::temp_dir();
    let in_path = dir.join("oxideav_vp9_psnr256_in.ivf");
    let out_path = dir.join("oxideav_vp9_psnr256_out.yuv");
    std::fs::write(&in_path, &ivf).unwrap();

    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "ivf", "-i"])
        .arg(&in_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out_path)
        .output()
        .expect("run ffmpeg");

    let _ = std::fs::remove_file(&in_path);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_file(&out_path);
        panic!("ffmpeg failed: {stderr}");
    }

    let decoded = std::fs::read(&out_path).unwrap_or_default();
    let _ = std::fs::remove_file(&out_path);

    let expected_size = 256 * 256 + 2 * 128 * 128; // 98304 bytes
    assert_eq!(
        decoded.len(),
        expected_size,
        "unexpected decoded frame size"
    );

    let y_psnr = psnr(&y, &decoded[..256 * 256]);
    let u_psnr = psnr(&u, &decoded[256 * 256..256 * 256 + 128 * 128]);
    let v_psnr = psnr(&v, &decoded[256 * 256 + 128 * 128..]);
    eprintln!("256×256 smooth PSNR Y={y_psnr:.2} dB  U={u_psnr:.2} dB  V={v_psnr:.2} dB");
    assert!(
        y_psnr >= 35.0,
        "PSNR_Y {y_psnr:.2} dB < 35 dB target (U={u_psnr:.2}, V={v_psnr:.2})"
    );
}

/// Round-40 mode-RDO benefit gate: a 256×256 horizontal-stripe pattern
/// (every row is constant, rows differ) is the canonical V_PRED shape —
/// V_PRED copies the above row down, so its residual is the row-to-row
/// luminance step (small) versus DC's full-row magnitude (large). The
/// mode-RDO picker should select V_PRED at most non-top-row blocks and
/// pull PSNR_Y substantially above the all-DC baseline.
fn horizontal_stripes_256x256() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 256 * 256];
    for r in 0..256usize {
        // Smooth row gradient. Each row is constant across columns.
        let v = (16 + (r as i32 * 224) / 256) as u8;
        for c in 0..256usize {
            y[r * 256 + c] = v;
        }
    }
    let u = vec![128u8; 128 * 128];
    let v = vec![128u8; 128 * 128];
    (y, u, v)
}

#[test]
fn encoder_256x256_horizontal_stripes_self_roundtrip() {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
    use oxideav_vp9::{make_decoder, CODEC_ID_STR};

    let (y, u, v) = horizontal_stripes_256x256();
    let src = YuvFrame {
        y: &y,
        y_stride: 256,
        u: &u,
        v: &v,
        uv_stride: 128,
        width: 256,
        height: 256,
    };
    let p = EncoderParams::keyframe(256, 256);
    let frame_bytes = encode_keyframe_yuv(&p, &src);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();
    let pkt = Packet::new(0, TimeBase::new(1, 30), frame_bytes);
    dec.send_packet(&pkt).unwrap();
    let f = dec.receive_frame().unwrap();
    let vf = match f {
        Frame::Video(v) => v,
        other => panic!("expected Video, got {other:?}"),
    };
    let decoded_y = &vf.planes[0].data;
    let y_psnr = psnr(&y, decoded_y);
    eprintln!("256×256 horizontal-stripes PSNR_Y = {y_psnr:.2} dB");
    // Mode-RDO should beat all-DC easily on a row-constant signal where
    // V_PRED tracks the single inter-row step exactly.
    assert!(
        y_psnr >= 40.0,
        "stripes PSNR_Y {y_psnr:.2} dB < 40 dB target (mode-RDO regression?)"
    );
}

/// Self-roundtrip via our own decoder — confirms the pixel-encoded
/// stream decodes back to a VideoFrame without errors.
#[test]
fn encoder_256x256_self_roundtrip() {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
    use oxideav_vp9::{make_decoder, CODEC_ID_STR};

    let (y, u, v) = smooth_256x256();
    let src = YuvFrame {
        y: &y,
        y_stride: 256,
        u: &u,
        v: &v,
        uv_stride: 128,
        width: 256,
        height: 256,
    };
    let p = EncoderParams::keyframe(256, 256);
    let frame_bytes = encode_keyframe_yuv(&p, &src);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();
    let pkt = Packet::new(0, TimeBase::new(1, 30), frame_bytes);
    dec.send_packet(&pkt).unwrap();
    let f = dec.receive_frame().unwrap();
    let vf = match f {
        Frame::Video(v) => v,
        other => panic!("expected Video, got {other:?}"),
    };
    assert_eq!(vf.planes[0].stride, 256);
    assert_eq!(vf.planes[0].data.len(), 256 * 256);

    // Compute PSNR against the source.
    let decoded_y = &vf.planes[0].data;
    let y_psnr = psnr(&y, decoded_y);
    eprintln!("256×256 self-roundtrip PSNR_Y = {y_psnr:.2} dB");
    assert!(
        y_psnr >= 35.0,
        "self-roundtrip PSNR_Y {y_psnr:.2} dB < 35 dB target"
    );
}
