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
use crate::frame_ctx::{
    COMP_MODE_CONTEXTS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_PROBS, INTER_MODE_CONTEXTS,
    INTER_MODE_PROBS, IS_INTER_CONTEXTS, REF_CONTEXTS,
};

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

/// Emit the inter-frame compressed header (§6.3, non-key path). All
/// §10.5 probability tables are kept at their defaults — we emit an
/// "update_prob = 0" guard bit in front of every diff_update_prob site
/// so the decoder skips the per-prob delta read. This is the
/// inverse of `parse_compressed_header` for the non-key branch
/// (frame_type=NonKey, intra_only=false): see §6.3.9 .. §6.3.16.
///
/// `interpolation_filter < 4` means the frame-level filter is fixed
/// (no switchable per-block reads), so we skip §6.3.10.
///
/// `allow_high_precision_mv` controls whether the §6.3.16 `class0_hp`
/// and `hp` mv probability deltas are present.
///
/// `compound_allowed` reports the §6.3.12 `compoundReferenceAllowed`
/// computation result (`false` when all three ref_frame_sign_bias
/// slots agree — the round-49 default with a single LAST_FRAME ref):
/// when false the §6.3.12 frame_reference_mode bit is NOT read by the
/// decoder, so we skip it on the emit side as well.
pub fn emit_compressed_header_p(
    tx_mode: TxMode,
    lossless: bool,
    interpolation_filter: u8,
    allow_high_precision_mv: bool,
    compound_allowed: bool,
) -> Vec<u8> {
    let mut be = BoolEncoder::new();
    if !lossless {
        write_tx_mode(&mut be, tx_mode);
    }
    // §6.3.7 coef_probs — `update_probs` flags per tx_size, all 0.
    let max_tx = match tx_mode {
        TxMode::Only4x4 => 0,
        TxMode::Allow8x8 => 1,
        TxMode::Allow16x16 => 2,
        TxMode::Allow32x32 | TxMode::Select => 3,
    };
    for _tx in 0..=max_tx {
        // `read_literal(1)` reads ONE raw bit (prob 128). The
        // §6.3.7 outer `update_probs` is a literal bit, NOT a
        // prob-252 gated diff bit, so we must emit prob 128 here.
        be.write_literal(0, 1);
    }

    // §6.3.8 read_skip_prob — diff_update_prob over 3 contexts.
    for _ in 0..3 {
        be.write(0, 252);
    }

    // §6.3.9 read_inter_mode_probs.
    for _ in 0..INTER_MODE_CONTEXTS {
        for _ in 0..INTER_MODE_PROBS {
            be.write(0, 252);
        }
    }

    // §6.3.10 read_interp_filter_probs — only when SWITCHABLE (== 4).
    if interpolation_filter == 4 {
        for _ in 0..INTERP_FILTER_CONTEXTS {
            for _ in 0..INTERP_FILTER_PROBS {
                be.write(0, 252);
            }
        }
    }

    // §6.3.11 read_is_inter_probs.
    for _ in 0..IS_INTER_CONTEXTS {
        be.write(0, 252);
    }

    // §6.3.12 frame_reference_mode — present only when at least two
    // of `ref_frame_sign_bias[LAST/GOLDEN/ALTREF]` differ. With a
    // single LAST_FRAME ref and uniform sign-bias the decoder
    // short-circuits to `SingleReference` without reading any bit
    // (see `read_reference_mode`). So we emit zero bits here.
    let reference_mode_is_select = false;
    if compound_allowed {
        // emit `non_single = 0` (→ SingleReference). 1 literal bit.
        be.write_literal(0, 1);
    }

    // §6.3.13 frame_reference_mode_probs — branches:
    //   - reference_mode == SELECT → comp_mode_prob deltas.
    //   - reference_mode != CompoundReference → single_ref_prob deltas (×2 per ctx).
    //   - reference_mode != SingleReference → comp_ref_prob deltas.
    if reference_mode_is_select {
        for _ in 0..COMP_MODE_CONTEXTS {
            be.write(0, 252);
        }
    }
    // Always emit single_ref deltas (SingleReference != CompoundReference).
    for _ in 0..REF_CONTEXTS {
        for _ in 0..2 {
            be.write(0, 252);
        }
    }
    // comp_ref skipped (SingleReference branch elides it).

    // §6.3.14 read_y_mode_probs (BLOCK_SIZE_GROUPS=4, INTRA_MODES-1=9).
    for _ in 0..4 {
        for _ in 0..9 {
            be.write(0, 252);
        }
    }

    // §6.3.15 read_partition_probs (PARTITION_CONTEXTS=16, PARTITION_TYPES-1=3).
    for _ in 0..16 {
        for _ in 0..3 {
            be.write(0, 252);
        }
    }

    // §6.3.16 mv_probs.
    // joints: 3 deltas (MV_JOINTS-1).
    for _ in 0..3 {
        be.write(0, 252);
    }
    // For each of 2 components: sign(1) + classes(10) + class0_bit(1) + bits(10).
    for _ in 0..2 {
        // sign
        be.write(0, 252);
        // classes
        for _ in 0..10 {
            be.write(0, 252);
        }
        // class0_bit
        be.write(0, 252);
        // bits (10 per component)
        for _ in 0..10 {
            be.write(0, 252);
        }
    }
    // For each of 2 components: class0_fr [2][3] + fr [3].
    for _ in 0..2 {
        for _ in 0..2 {
            for _ in 0..3 {
                be.write(0, 252);
            }
        }
        for _ in 0..3 {
            be.write(0, 252);
        }
    }
    if allow_high_precision_mv {
        for _ in 0..2 {
            // class0_hp
            be.write(0, 252);
            // hp
            be.write(0, 252);
        }
    }

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
