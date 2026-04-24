//! Dump a single-keyframe IVF produced by the MVP encoder to stdout.
//!
//! Usage: `cargo run --example write_ivf > out.ivf`
use oxideav_vp9::encoder::{encode_keyframe, EncoderParams};

fn main() {
    let width: u16 = 64;
    let height: u16 = 64;
    let p = EncoderParams::keyframe(width as u32, height as u32);
    let frame = encode_keyframe(&p);

    let mut out = Vec::new();
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP90");
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&frame);

    use std::io::Write;
    std::io::stdout().write_all(&out).unwrap();
    eprintln!(
        "wrote {} bytes total ({} IVF + {} VP9)",
        out.len(),
        32 + 12,
        frame.len()
    );
    eprintln!(
        "first 32 bytes of VP9 frame: {:02x?}",
        &frame[..frame.len().min(32)]
    );
}
