//! Decode a single-frame IVF and dump the luma plane statistics.
//!
//! Usage: `cargo run --example decode_dump -- /tmp/vp9_noisy.ivf`
//!
//! Prints reference-YUV statistics (when an .yuv path is supplied as a
//! second arg) for direct comparison with our decode.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

fn stats(d: &[u8]) -> (f64, f64, u8, u8) {
    let n = d.len() as f64;
    let s: u64 = d.iter().map(|&b| b as u64).sum();
    let mean = s as f64 / n;
    let ss: f64 = d.iter().map(|&b| (b as f64 - mean).powi(2)).sum();
    let std = (ss / n).sqrt();
    let lo = *d.iter().min().unwrap_or(&0);
    let hi = *d.iter().max().unwrap_or(&0);
    (mean, std, lo, hi)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ivf_path = std::env::args().nth(1).expect("ivf path");
    let buf = std::fs::read(&ivf_path)?;
    let (hdr, _) = ivf::parse_header(&buf)?;
    eprintln!("IVF: {}x{}", hdr.width, hdr.height);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params)?;

    let mut frame_idx = 0usize;
    for f in ivf::iter_frames(&buf)? {
        let f = f?;
        let pkt = Packet::new(0, TimeBase::new(1, 24), f.payload.to_vec());
        let _ = dec.send_packet(&pkt);
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    let y = &v.planes[0].data;
                    let (m, s, lo, hi) = stats(y);
                    eprintln!("frame {frame_idx}: Y mean={m:.1} std={s:.2} min={lo} max={hi}");
                    // Dump a small slice of the top-left pixels
                    let st = v.planes[0].stride;
                    eprintln!("  top-left 8x8 luma:");
                    for r in 0..8.min(v.height as usize) {
                        eprint!("    ");
                        for c in 0..8.min(v.width as usize) {
                            eprint!("{:3} ", y[r * st + c]);
                        }
                        eprintln!();
                    }
                    frame_idx += 1;
                }
                Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
                Err(e) => {
                    eprintln!("recv err: {e:?}");
                    break;
                }
                _ => {}
            }
        }
    }

    if let Some(yuv_path) = std::env::args().nth(2) {
        let yuv = std::fs::read(&yuv_path)?;
        let y_sz = (hdr.width as usize) * (hdr.height as usize);
        eprintln!("\n=== reference YUV ===");
        let n_frames = yuv.len() / (y_sz * 3 / 2);
        eprintln!("frames: {n_frames}");
        for i in 0..n_frames {
            let off = i * (y_sz * 3 / 2);
            let y = &yuv[off..off + y_sz];
            let (m, s, lo, hi) = stats(y);
            eprintln!("ref frame {i}: Y mean={m:.1} std={s:.2} min={lo} max={hi}");
            eprintln!("  top-left 8x8 luma:");
            for r in 0..8 {
                eprint!("    ");
                for c in 0..8 {
                    eprint!("{:3} ", y[r * (hdr.width as usize) + c]);
                }
                eprintln!();
            }
        }
    }

    Ok(())
}
