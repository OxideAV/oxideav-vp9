//! Differential fuzz for the **frame-threaded** framework decoder: the
//! same packets go through a serial [`Vp9Decoder`] and one holding a
//! multi-thread [`ExecutionContext`] (several frames in flight, each
//! frame's entropy decode / reconstruction / loop filter running as
//! pool jobs that wait on the rows of the frames they reference), and
//! the two MUST produce the same output.
//!
//! Input layout: byte 0 picks the thread budget (`2 + b % 8`). The rest
//! is an IVF file when it starts with `DKIF` (one packet per IVF frame,
//! the IVF timestamp as `pts`), otherwise a list of `u16`
//! little-endian length-prefixed packets (a short tail is one last
//! packet).
//!
//! The oracle is total, malformed input included:
//!
//! * the shown frames — count, order, `pts`, plane strides and every
//!   sample byte — are identical;
//! * the `Frame::StreamParams` events land before the same frames;
//! * an error is reported by one decode exactly when the other reports
//!   one (a threaded decode reports a frame's failure from the call
//!   that completes the frame, then replays the frames submitted behind
//!   it on the restored state and reports only that first failure — so
//!   the calls and the count differ, the frames do not).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{Decoder, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_vp9::Vp9Decoder;

/// What one decode of the packet list produced.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    /// Per shown frame: `pts`, whether a stream-params event preceded
    /// it, and per plane `(stride, bytes)`.
    frames: Vec<(Option<i64>, bool, Vec<(usize, Vec<u8>)>)>,
    errors: usize,
}

fn packets(data: &[u8]) -> Vec<(i64, &[u8])> {
    let mut out = Vec::new();
    if data.len() >= 32 && &data[..4] == b"DKIF" {
        let mut pos = (u16::from_le_bytes([data[6], data[7]]) as usize).max(32);
        while pos + 12 <= data.len() && out.len() < 64 {
            let size = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            let pts = i64::from(data[pos + 4]);
            pos += 12;
            let end = pos.saturating_add(size).min(data.len());
            out.push((pts, &data[pos..end]));
            pos = end;
        }
    } else {
        let mut rest = data;
        while rest.len() >= 2 && out.len() < 64 {
            let len = u16::from_le_bytes([rest[0], rest[1]]) as usize;
            if len > rest.len() - 2 {
                break;
            }
            out.push((out.len() as i64, &rest[2..2 + len]));
            rest = &rest[2 + len..];
        }
        if !rest.is_empty() {
            out.push((out.len() as i64, rest));
        }
    }
    out
}

fn decode(packets: &[(i64, &[u8])], exec: &ExecutionContext) -> Outcome {
    let mut dec = Vp9Decoder::new();
    dec.set_execution_context(exec);
    let mut out = Outcome {
        frames: Vec::new(),
        errors: 0,
    };
    let mut event = false;
    let mut drain = |dec: &mut Vp9Decoder, out: &mut Outcome| {
        // Errors while draining are counted; a bounded number of
        // attempts keeps a persistently failing queue from spinning.
        let mut budget = 4 * packets.len() + 16;
        while budget > 0 {
            budget -= 1;
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    let planes = vf.planes.into_iter().map(|p| (p.stride, p.data)).collect();
                    out.frames
                        .push((vf.pts, std::mem::take(&mut event), planes));
                }
                Ok(f) if f.is_stream_params() => event = true,
                Ok(_) => {}
                // "Needs more input" / end of stream ends the drain; a
                // decode error is counted and the drain goes on.
                Err(e) if e.is_need_more() || e.is_eof() => break,
                Err(_) => out.errors += 1,
            }
        }
    };
    for &(pts, payload) in packets {
        let pkt = Packet::new(0, TimeBase::new(1, 1000), payload.to_vec()).with_pts(pts);
        if dec.send_packet(&pkt).is_err() {
            out.errors += 1;
        }
        drain(&mut dec, &mut out);
    }
    if dec.flush().is_err() {
        out.errors += 1;
    }
    drain(&mut dec, &mut out);
    out
}

fuzz_target!(|data: &[u8]| {
    let Some((&pick, rest)) = data.split_first() else {
        return;
    };
    let threads = 2 + usize::from(pick % 8);
    let packets = packets(rest);
    let serial = decode(&packets, &ExecutionContext::serial());
    let threaded = decode(&packets, &ExecutionContext::with_threads(threads));
    assert_eq!(
        serial.frames.len(),
        threaded.frames.len(),
        "{threads} threads: shown-frame count diverged from the serial decode"
    );
    assert_eq!(
        serial.errors > 0,
        threaded.errors > 0,
        "{threads} threads: one decode reported an error, the other none ({} vs {})",
        serial.errors,
        threaded.errors
    );
    for (i, (a, b)) in serial.frames.iter().zip(&threaded.frames).enumerate() {
        assert!(
            a == b,
            "{threads} threads: frame {i} diverged from the serial decode"
        );
    }
});
