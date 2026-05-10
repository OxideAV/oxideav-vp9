#![no_main]

//! Fuzz: feed arbitrary bytes through `Vp9Decoder` and assert it
//! returns a `Result` rather than panicking.
//!
//! Per `oxideav_core::Decoder`, both `send_packet` and `receive_frame`
//! return `Result<...>`; any panic is a bug we'd like to find. This
//! harness intentionally has no oracle — it only exercises the panic
//! surface.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Decoder, Packet, TimeBase};
use oxideav_vp9::decoder::{make_decoder, Vp9Decoder};

fuzz_target!(|data: &[u8]| {
    // Cap at 1 MiB of raw input — typical VP9 fuzz finds are ~hundred
    // bytes, and longer inputs just exercise the same paths slowly.
    if data.len() > 1 << 20 {
        return;
    }
    // Note: oxideav-vp9 enforces `MAX_FRAME_PIXELS` (8192×8192) inside
    // `Vp9Decoder::ingest_one` — a fuzz mutation that declares a
    // 65535×65535 frame surfaces as `Error::InvalidData` rather than
    // tipping the decoder into a multi-GiB allocation. No harness
    // pre-filter is needed.

    // Path A: factory + trait-object — exercises `make_decoder`.
    {
        let params = CodecParameters::video(CodecId::new("vp9"));
        if let Ok(mut dec) = make_decoder(&params) {
            let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
            // Either Ok or Err is fine — we just want no panic.
            let _ = dec.send_packet(&pkt);
            // Drain any frames that came out (loop-bounded so a buggy
            // decoder reporting infinite frames can't hang the fuzzer).
            for _ in 0..32 {
                match dec.receive_frame() {
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
        }
    }

    // Path B: direct constructor — exercises `Vp9Decoder::new` /
    // ingest paths the trait-object route may not hit.
    {
        let mut dec = Vp9Decoder::new(CodecId::new("vp9"));
        let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
        let _ = dec.send_packet(&pkt);
        for _ in 0..32 {
            match dec.receive_frame() {
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        let _ = dec.flush();
        for _ in 0..32 {
            match dec.receive_frame() {
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        let _ = dec.reset();
    }
});
