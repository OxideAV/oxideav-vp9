//! Panic-surface fuzz for the §6.3 compressed-header walkers — the
//! deepest decode entries currently exposed over the §9.2 Boolean
//! (range) decoder. The first input byte is a control byte steering
//! the caller-supplied flags the walkers take from the uncompressed
//! header; the remaining bytes are fed to the coder verbatim:
//!
//!   bit 0 — `Lossless` (§6.2.9 derivation)
//!   bit 1 — intra prefix only vs. full `FrameIsIntra == 0` tail
//!   bit 2 — `interpolation_filter == SWITCHABLE` (§6.2.7)
//!   bit 3 — `allow_high_precision_mv` (§6.2.7)
//!   bits 4..6 — `ref_frame_sign_bias` for LAST / GOLDEN / ALTREF
//!
//! No oracle — `Err` is a correct answer for garbage input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{
    parse_compressed_header, parse_compressed_header_inter, RefFrameSignBias,
    Vp9CompressedHeaderInterInputs,
};

fuzz_target!(|data: &[u8]| {
    let Some((&ctl, payload)) = data.split_first() else {
        return;
    };
    let lossless = ctl & 0x01 != 0;
    if ctl & 0x02 == 0 {
        let _ = parse_compressed_header(payload, lossless);
    } else {
        let inputs = Vp9CompressedHeaderInterInputs {
            interpolation_filter_is_switchable: ctl & 0x04 != 0,
            allow_high_precision_mv: ctl & 0x08 != 0,
            ref_frame_sign_bias: RefFrameSignBias::from_inter_biases(
                (ctl >> 4) & 1,
                (ctl >> 5) & 1,
                (ctl >> 6) & 1,
            ),
        };
        let _ = parse_compressed_header_inter(payload, lossless, inputs);
    }
});
