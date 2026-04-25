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
    let mut psnr_sum = 0.0f64;
    let mut psnr_n = 0usize;
    let ref_yuv_data = std::env::args().nth(2).and_then(|p| std::fs::read(p).ok());
    let w = hdr.width as usize;
    let h_ = hdr.height as usize;
    let y_sz = w * h_;
    let uv_sz = (w / 2) * (h_ / 2);
    let fs = y_sz + 2 * uv_sz;
    for f in ivf::iter_frames(&buf)? {
        let f = f?;
        let pkt = Packet::new(0, TimeBase::new(1, 24), f.payload.to_vec());
        let _ = dec.send_packet(&pkt);
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    let y = &v.planes[0].data;
                    let (m, s, lo, hi) = stats(y);
                    if let Some(rd) = &ref_yuv_data {
                        let off = frame_idx * fs;
                        if off + y_sz <= rd.len() {
                            let st = v.planes[0].stride;
                            let mut sse: u64 = 0;
                            for r in 0..h_ {
                                for c in 0..w {
                                    let a = y[r * st + c] as i32;
                                    let b = rd[off + r * w + c] as i32;
                                    sse += ((a - b) * (a - b)) as u64;
                                }
                            }
                            let psnr = if sse == 0 {
                                f64::INFINITY
                            } else {
                                let mse = sse as f64 / y_sz as f64;
                                10.0 * (255.0 * 255.0 / mse).log10()
                            };
                            eprintln!(
                                "frame {frame_idx}: Y mean={m:.1} std={s:.2} min={lo} max={hi} PSNR={psnr:.2}"
                            );
                            if psnr.is_finite() {
                                psnr_sum += psnr;
                                psnr_n += 1;
                            }
                        } else {
                            eprintln!(
                                "frame {frame_idx}: Y mean={m:.1} std={s:.2} min={lo} max={hi}"
                            );
                        }
                    } else {
                        eprintln!("frame {frame_idx}: Y mean={m:.1} std={s:.2} min={lo} max={hi}");
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

    if psnr_n > 0 {
        eprintln!(
            "mean luma PSNR over {psnr_n} frames: {:.2} dB",
            psnr_sum / psnr_n as f64
        );
    }

    Ok(())
}
