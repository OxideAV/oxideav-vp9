//! Decode an IVF and dump the raw YUV (yuv420p) to a file path.
//!
//! Usage: `cargo run --example dump_yuv -- input.ivf out.yuv`

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ivf_path = std::env::args().nth(1).expect("ivf path");
    let out_path = std::env::args().nth(2).expect("out yuv path");
    let buf = std::fs::read(&ivf_path)?;
    let (hdr, _) = ivf::parse_header(&buf)?;
    let w = hdr.width as usize;
    let h = hdr.height as usize;

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params)?;

    let mut out = std::fs::File::create(&out_path)?;
    let mut frame_idx = 0usize;
    for f in ivf::iter_frames(&buf)? {
        let f = f?;
        let pkt = Packet::new(0, TimeBase::new(1, 24), f.payload.to_vec());
        let _ = dec.send_packet(&pkt);
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    // Y plane
                    let y = &v.planes[0];
                    for r in 0..h {
                        out.write_all(&y.data[r * y.stride..r * y.stride + w])?;
                    }
                    // U plane
                    let u = &v.planes[1];
                    for r in 0..h / 2 {
                        out.write_all(&u.data[r * u.stride..r * u.stride + w / 2])?;
                    }
                    // V plane
                    let vv = &v.planes[2];
                    for r in 0..h / 2 {
                        out.write_all(&vv.data[r * vv.stride..r * vv.stride + w / 2])?;
                    }
                    frame_idx += 1;
                    eprintln!("wrote frame {frame_idx}");
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
    eprintln!("dumped {frame_idx} frames to {out_path}");
    Ok(())
}
