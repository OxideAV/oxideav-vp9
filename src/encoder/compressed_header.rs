//! VP9 compressed header emitter — §6.3.
//!
//! Emits the boolean-coded payload that sits between the uncompressed
//! header and the first tile. Scope: keyframe, default probabilities,
//! fixed `tx_mode`, `SINGLE_REFERENCE`.
//!
//! Keyframes skip several sub-procedures (inter-mode probs, MV probs,
//! frame_reference_mode) per §6.3. For a non-lossless keyframe we
//! emit:
//!
//! * `tx_mode` — 2 bits, extended to 3 if value reaches 3. We fix
//!   `tx_mode = ONLY_4X4` (0) for the MVP so no `tx_size` symbol is
//!   consumed per block.
//! * coef-probs / skip-probs / delta probabilities — all skipped by
//!   writing the "no update" flag bits where applicable. For the MVP
//!   we encode the bare minimum the decoder tolerates.
//!
//! Note: the decoder's `parse_compressed_header` only consumes
//! `tx_mode` + `reference_mode` today. Additional probability updates
//! are emitted but ignored downstream — they are present so ffmpeg
//! and other VP9 decoders (which DO consume them) receive a well-
//! formed compressed header.

use crate::compressed_header::TxMode;
use crate::encoder::bool_encoder::BoolEncoder;

/// Emit the keyframe compressed header. Returns the encoded bytes.
///
/// `tx_mode`: selects per-block transform size ceiling. For the MVP we
/// pass `TxMode::Only4x4` (0) so block decode uses 4×4 tx universally.
///
/// `lossless` controls whether `tx_mode` is written at all — in
/// lossless mode the decoder skips §6.3.1 and forces `tx_mode=0`.
pub fn emit_compressed_header(tx_mode: TxMode, lossless: bool) -> Vec<u8> {
    let mut be = BoolEncoder::new();
    if !lossless {
        write_tx_mode(&mut be, tx_mode);
    }
    // §6.3.3 read_coef_probs:
    //   for tx_size = 0..=max_tx:
    //     update_probs = f(1)
    //     if update_probs: nested update bits per band/ctx/node
    // Writing `update_probs=0` skips the nested loop entirely.
    let max_tx = match tx_mode {
        TxMode::Only4x4 => 0,
        TxMode::Allow8x8 => 1,
        TxMode::Allow16x16 => 2,
        TxMode::Allow32x32 | TxMode::Select => 3,
    };
    for _tx in 0..=max_tx {
        // update_probs — emit 0 with prob 252 (default).
        be.write(0, 252);
    }

    // §6.3.4 read_skip_prob — 3 probs, each guarded by `update_prob`
    // bit (prob 252). All zero = keep defaults.
    for _ in 0..3 {
        be.write(0, 252);
    }

    // §6.3.5+ (inter_mode_probs, interp_filter_probs, is_inter_probs,
    // frame_reference_mode, y_mode_probs, partition_probs, mv_probs)
    // are only read for non-key / non-intra-only frames per §6.3. For
    // keyframes the compressed header ends right after skip_prob.

    be.finish()
}

fn write_tx_mode(be: &mut BoolEncoder, tx_mode: TxMode) {
    // §6.3.1 read_tx_mode. Two literal bits for values 0..=2; value 3
    // reads a third bit; we select from {0, 1, 2, 3, 4} where 4 means
    // TX_MODE_SELECT (emitted as 3 + 1).
    let v = tx_mode as u32;
    if v <= 2 {
        be.write_literal(v, 2);
    } else {
        // v in {3, 4}: first emit 3 as 2 bits, then (v-3) as 1 bit.
        be.write_literal(3, 2);
        be.write_literal(v - 3, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed_header::{parse_compressed_header, ReferenceMode};
    use crate::encoder::params::EncoderParams;
    use crate::encoder::uncompressed_header::emit_uncompressed_header;
    use crate::headers::parse_uncompressed_header;

    #[test]
    fn emit_then_parse_via_decoder() {
        let p = EncoderParams::keyframe(64, 64);
        let ch_bytes = emit_compressed_header(TxMode::Only4x4, false);
        let uh = emit_uncompressed_header(&p, ch_bytes.len() as u16);
        let mut full = uh.clone();
        full.extend_from_slice(&ch_bytes);
        // Append some tile bytes so downstream parsing doesn't EOF.
        full.extend_from_slice(&[0u8; 16]);
        // Parse uncompressed header first.
        let h = parse_uncompressed_header(&full, None).unwrap();
        assert_eq!(h.header_size as usize, ch_bytes.len());
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start + h.header_size as usize;
        // Parse compressed header via the decoder.
        let ch = parse_compressed_header(&full[cmp_start..cmp_end], &h).unwrap();
        assert_eq!(ch.tx_mode, Some(TxMode::Only4x4));
        assert_eq!(ch.reference_mode, Some(ReferenceMode::SingleReference));
    }

    #[test]
    fn emit_allow_8x8_roundtrip() {
        let p = EncoderParams::keyframe(64, 64);
        let ch_bytes = emit_compressed_header(TxMode::Allow8x8, false);
        let uh = emit_uncompressed_header(&p, ch_bytes.len() as u16);
        let mut full = uh.clone();
        full.extend_from_slice(&ch_bytes);
        full.extend_from_slice(&[0u8; 16]);
        let h = parse_uncompressed_header(&full, None).unwrap();
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start + h.header_size as usize;
        let ch = parse_compressed_header(&full[cmp_start..cmp_end], &h).unwrap();
        assert_eq!(ch.tx_mode, Some(TxMode::Allow8x8));
    }

    #[test]
    fn emit_allow_32x32_roundtrip() {
        let p = EncoderParams::keyframe(64, 64);
        let ch_bytes = emit_compressed_header(TxMode::Allow32x32, false);
        let uh = emit_uncompressed_header(&p, ch_bytes.len() as u16);
        let mut full = uh.clone();
        full.extend_from_slice(&ch_bytes);
        full.extend_from_slice(&[0u8; 32]);
        let h = parse_uncompressed_header(&full, None).unwrap();
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start + h.header_size as usize;
        let ch = parse_compressed_header(&full[cmp_start..cmp_end], &h).unwrap();
        assert_eq!(ch.tx_mode, Some(TxMode::Allow32x32));
    }
}
