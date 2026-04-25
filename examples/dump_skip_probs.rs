//! Inspect per-frame skip_probs after the compressed header is decoded.
//!
//! Usage:
//!   cargo run --example dump_skip_probs -- <ivf-path>
//!
//! This was added during the Round-15 §6.4.7 / §7.4.6 skip-context
//! investigation. It parses each frame's uncompressed + compressed
//! headers (the same bit-stream prefix the tile decoder consumes) and
//! prints the `skip_probs[0..3]` array along with the frame type,
//! reference mode, tx mode, and qindex. Useful for confirming whether
//! a real-world bitstream has updated `skip_probs` from the §10.5
//! defaults `[192, 128, 64]` before falling into the per-block tile
//! decode (which is where the spec-form `skip_probs[ctx]` divergence
//! shows up).
//!
//! Inter frames whose dimensions come from `frame_size_with_refs` will
//! report `0x0` because this example doesn't carry a DPB; that's
//! cosmetic — the compressed header probabilities are still correctly
//! decoded. Superframes are split via `decoder::split_superframe`.

use oxideav_vp9::{
    ivf, parse_compressed_header, parse_uncompressed_header, ColorConfig, FrameType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_skip_probs <ivf>");
    let buf = std::fs::read(&path)?;
    let (hdr, _) = ivf::parse_header(&buf)?;
    eprintln!("IVF: {}x{}", hdr.width, hdr.height);

    let mut frame_idx = 0usize;
    let mut last_color: Option<ColorConfig> = None;
    for ivf_f in ivf::iter_frames(&buf)? {
        let ivf_f = ivf_f?;
        for sub in oxideav_vp9::decoder::split_superframe(ivf_f.payload) {
            let f_payload = sub;
            let h = parse_uncompressed_header(f_payload, last_color)?;
            if h.frame_type == FrameType::Key || h.intra_only {
                last_color = Some(h.color_config);
            }
            let cmp_start = h.uncompressed_header_size;
            let cmp_end = cmp_start.saturating_add(h.header_size as usize);
            if cmp_end > f_payload.len() {
                eprintln!(
                    "frame {frame_idx}: compressed header truncated payload={} uhsz={} hsz={} (show_existing={})",
                    f_payload.len(),
                    cmp_start,
                    h.header_size,
                    h.show_existing_frame
                );
                frame_idx += 1;
                continue;
            }
            if h.show_existing_frame {
                eprintln!("frame {frame_idx}: show_existing");
                frame_idx += 1;
                continue;
            }
            let ch = parse_compressed_header(&f_payload[cmp_start..cmp_end], &h)?;
            let kind = match h.frame_type {
                FrameType::Key => "K",
                FrameType::NonKey => {
                    if h.intra_only {
                        "I"
                    } else {
                        "P"
                    }
                }
            };
            eprintln!(
                "frame {frame_idx} {kind} {}x{} show={} sub={}: skip_probs={:?} ref_mode={:?} tx_mode={:?} qp={} lossless={}",
                h.width,
                h.height,
                h.show_frame,
                f_payload.len(),
                ch.ctx.skip_probs,
                ch.reference_mode,
                ch.tx_mode,
                h.quantization.base_q_idx,
                h.quantization.lossless,
            );
            frame_idx += 1;
        }
    }
    Ok(())
}
