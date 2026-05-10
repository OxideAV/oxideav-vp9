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
use oxideav_vp9::headers::parse_uncompressed_header;

/// Maximum total pixel count per fuzz frame (256 KP). The decoder
/// will allocate buffers proportional to the declared dimensions
/// before validating the bitstream further; an unbounded fuzz input
/// can trivially declare a 65535×65535 keyframe and exhaust memory.
/// That's a real DoS-class bug in the decoder; this cap keeps the
/// fuzz iter density high while the decoder grows its own allocation
/// guards. 256 KP = a 512×512 frame — well above the 64×64 size we
/// use for the roundtrip target's source frames.
const MAX_PIXELS: u32 = 1 << 18;

fuzz_target!(|data: &[u8]| {
    // Cap at 1 MiB of raw input — typical VP9 fuzz finds are ~hundred
    // bytes, and longer inputs just exercise the same paths slowly.
    if data.len() > 1 << 20 {
        return;
    }

    // Sniff the uncompressed header. If we can parse it, skip inputs
    // whose declared frame size would force a multi-MB allocation —
    // see MAX_PIXELS rationale above. If the header is unparsable
    // the decoder will reject early without large allocs.
    if let Ok(h) = parse_uncompressed_header(data, None) {
        let pixels = h.width.saturating_mul(h.height);
        if pixels > MAX_PIXELS {
            return;
        }
    }

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
