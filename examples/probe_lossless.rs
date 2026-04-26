//! Probe whether a VP9 bitstream is being parsed as lossless.

use oxideav_vp9::ivf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("ivf path");
    let buf = std::fs::read(&path)?;
    let (hdr, _) = ivf::parse_header(&buf)?;
    eprintln!("ivf {}x{}", hdr.width, hdr.height);
    for (i, f) in ivf::iter_frames(&buf)?.enumerate() {
        let f = f?;
        let h = oxideav_vp9::headers::parse_uncompressed_header(f.payload, None)?;
        eprintln!(
            "frame {i}: type={:?} intra_only={} q={} lossless={} (uh_size={}, hdr_size={})",
            h.frame_type,
            h.intra_only,
            h.quantization.base_q_idx,
            h.quantization.lossless,
            h.uncompressed_header_size,
            h.header_size,
        );
        if i > 5 {
            break;
        }
    }
    Ok(())
}
