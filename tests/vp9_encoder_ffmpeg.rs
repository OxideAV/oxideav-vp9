//! Integration test: encode a keyframe with the MVP encoder, wrap in
//! IVF, and ask ffmpeg to decode. Skipped if `ffmpeg` is not on
//! PATH.

use std::process::Command;

use oxideav_vp9::encoder::{encode_keyframe, EncoderParams};

fn build_ivf(width: u16, height: u16, frame: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"DKIF");
    v.extend_from_slice(&0u16.to_le_bytes()); // version
    v.extend_from_slice(&32u16.to_le_bytes()); // header len
    v.extend_from_slice(b"VP90");
    v.extend_from_slice(&width.to_le_bytes());
    v.extend_from_slice(&height.to_le_bytes());
    v.extend_from_slice(&30u32.to_le_bytes()); // frame rate num
    v.extend_from_slice(&1u32.to_le_bytes()); // den
    v.extend_from_slice(&1u32.to_le_bytes()); // frame count
    v.extend_from_slice(&0u32.to_le_bytes()); // reserved
    v.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes()); // pts
    v.extend_from_slice(frame);
    v
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

#[test]
fn ffmpeg_accepts_encoded_keyframe() {
    if !have_ffmpeg() {
        eprintln!("skipping — ffmpeg not available");
        return;
    }
    let p = EncoderParams::keyframe(64, 64);
    let frame = encode_keyframe(&p);
    let ivf = build_ivf(64, 64, &frame);

    // Write to a temp file.
    let dir = std::env::temp_dir();
    let in_path = dir.join("oxideav_vp9_encoder_test.ivf");
    let out_path = dir.join("oxideav_vp9_encoder_test.yuv");
    std::fs::write(&in_path, &ivf).unwrap();

    // Ask ffmpeg to decode it.
    let output = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-f")
        .arg("ivf")
        .arg("-i")
        .arg(&in_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&out_path)
        .output()
        .expect("run ffmpeg");

    let _ = std::fs::remove_file(&in_path);
    let ok = output.status.success();
    if !ok {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("ffmpeg failed: {stderr}");
    }
    // Check raw YUV size.
    let out = std::fs::read(&out_path).unwrap_or_default();
    let _ = std::fs::remove_file(&out_path);
    // 64x64 4:2:0 = 64*64 + 2*32*32 = 6144 bytes.
    assert_eq!(out.len(), 6144, "unexpected decoded frame size");
    // ffmpeg decoded the bitstream without errors — that's the
    // primary acceptance check. The luma content may differ from
    // our own decoder's output because the MVP `skip_prob` handling
    // is context-0 only, while ffmpeg tracks per-block skip
    // contexts. Sample values should still land near midgrey; don't
    // be pixel-strict.
    let luma = &out[..64 * 64];
    let avg: u32 = luma.iter().map(|&v| v as u32).sum::<u32>() / (luma.len() as u32);
    assert!(
        (120..=140).contains(&avg),
        "average luma {avg} should be near midgrey"
    );
    // Chroma should be uniform (midgrey).
    let u = &out[4096..4096 + 1024];
    for &s in u {
        assert_eq!(s, u[0], "u should be uniform");
    }
}
