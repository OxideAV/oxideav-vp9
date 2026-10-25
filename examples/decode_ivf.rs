//! Decode an IVF-wrapped VP9 stream through the framework [`Vp9Decoder`]
//! at a chosen thread budget, report the wall time, and optionally write
//! the shown frames as raw planar samples (8-bit streams: one byte per
//! sample; deeper streams: little-endian `u16`).
//!
//! ```text
//! cargo run --release --example decode_ivf -- input.ivf [threads|auto] [out.yuv]
//! ```
//!
//! `threads` = `0` / omitted keeps the serial default; `auto` grants
//! [`ExecutionContext::auto`].

use std::io::Write;
use std::time::Instant;

use oxideav_core::{Decoder, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_vp9::Vp9Decoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: decode_ivf input.ivf [threads|auto] [out.yuv]");
    let data = std::fs::read(path).expect("read input");
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;

    let mut packets = Vec::new();
    let mut pos = hdr_len;
    while pos + 12 <= data.len() {
        let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let pts = i64::from_le_bytes(data[pos + 4..pos + 12].try_into().unwrap());
        pos += 12;
        let end = (pos + size).min(data.len());
        packets.push((pts, data[pos..end].to_vec()));
        pos = end;
    }

    let exec = match args.get(2).map(String::as_str) {
        None | Some("0") => ExecutionContext::serial(),
        Some("auto") => ExecutionContext::auto(),
        Some(n) => ExecutionContext::with_threads(n.parse().expect("thread count")),
    };
    let mut out = args
        .get(3)
        .map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("create output")));

    let mut dec = Vp9Decoder::new();
    dec.set_execution_context(&exec);
    let mut frames = 0usize;
    let mut decode_time = std::time::Duration::ZERO;
    let start = Instant::now();
    let mut drain = |dec: &mut Vp9Decoder, decode_time: &mut std::time::Duration| loop {
        let t = Instant::now();
        let got = dec.receive_frame();
        *decode_time += t.elapsed();
        let Ok(Frame::Video(vf)) = got else { break };
        frames += 1;
        if let Some(w) = out.as_mut() {
            for plane in &vf.planes {
                w.write_all(&plane.data).expect("write");
            }
        }
    };
    for (pts, payload) in &packets {
        let pkt = Packet::new(0, TimeBase::new(1, 1000), payload.clone()).with_pts(*pts);
        let t = Instant::now();
        dec.send_packet(&pkt).expect("decode");
        decode_time += t.elapsed();
        drain(&mut dec, &mut decode_time);
    }
    dec.flush().expect("flush");
    drain(&mut dec, &mut decode_time);
    let wall = start.elapsed();
    eprintln!(
        "{frames} frames, decode {:.3} s ({:.1} fps), wall {:.3} s",
        decode_time.as_secs_f64(),
        frames as f64 / decode_time.as_secs_f64(),
        wall.as_secs_f64()
    );
}
