//! VP9 §6.3 compressed-header **writer** for the intra (key / intra-only)
//! frame path — the inverse of `compressed::parse_compressed_header`.
//!
//! The encoder bootstrap emits the §6.3 compressed header with the
//! **probability tables left at their §10.5 defaults** (no forward
//! updates): every `diff_update_prob` writes its `update_prob == 0`
//! flag, and every §6.3.7 per-tx-size `update_probs` flag is 0. That is
//! the simplest conformant compressed header — the decoder reconstructs
//! the default banks, which is exactly what a from-defaults encoder uses
//! to drive the residual arithmetic coder.
//!
//! The walk mirrors `parse_compressed_header_intra_prefix`:
//! §6.3.1 `read_tx_mode` → conditional §6.3.2 `tx_mode_probs` →
//! §6.3.7 `read_coef_probs` → §6.3.8 `read_skip_prob`. The inter-only
//! §6.3.9..§6.3.16 tail is not emitted (intra frames don't carry it).
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.3; the field order mirrors the
//! in-crate parser exactly.

use crate::bool_encoder::BoolEncoder;
use crate::compressed::{tx_mode_to_biggest_tx_size, ReferenceMode, TxMode};
use crate::mode_info::{
    COMP_MODE_CONTEXTS, INTERP_FILTER_CONTEXTS, INTER_MODES, INTER_MODE_CONTEXTS,
    IS_INTER_CONTEXTS, REF_CONTEXTS, SWITCHABLE_FILTERS,
};
use crate::Error;

/// `diff_update_prob`-no-update probability (§6.3.3): the `update_prob`
/// flag is coded with `B(252)`.
const DIFF_UPDATE_PROB: u32 = 252;

/// `TX_SIZE_CONTEXTS` from §3 — the per-tx-size context count swept by
/// §6.3.2 `tx_mode_probs`.
const TX_SIZE_CONTEXTS: usize = 2;

/// Write the §6.3.1 `tx_mode` syntax — inverse of `read_tx_mode`. On a
/// lossless frame the decoder hardwires `ONLY_4X4` and reads no bits, so
/// nothing is written.
fn write_tx_mode(enc: &mut BoolEncoder, tx_mode: TxMode, lossless: bool) {
    if lossless {
        debug_assert!(matches!(tx_mode, TxMode::Only4x4));
        return;
    }
    let value: u32 = match tx_mode {
        TxMode::Only4x4 => 0,
        TxMode::Allow8x8 => 1,
        TxMode::Allow16x16 => 2,
        TxMode::Allow32x32 => 3,
        TxMode::TxModeSelect => 4,
    };
    if value < 3 {
        enc.write_literal(value, 2);
    } else {
        // raw = 3, then a 1-bit select: ALLOW_32X32 (0) / TX_MODE_SELECT (1).
        enc.write_literal(3, 2);
        enc.write_literal(value - 3, 1);
    }
}

/// Write the §6.3.2 `tx_mode_probs` sweep with every cell left
/// unchanged (each `read_diff_update_prob` consumes a `B(252) == 0`).
/// Only emitted when `tx_mode == TX_MODE_SELECT`.
fn write_tx_mode_probs_no_update(enc: &mut BoolEncoder) {
    // tx_probs_8x8: i < TX_SIZE_CONTEXTS, j < 1.
    // tx_probs_16x16: i < TX_SIZE_CONTEXTS, j < 2.
    // tx_probs_32x32: i < TX_SIZE_CONTEXTS, j < 3.
    for cols in [1usize, 2, 3] {
        for _ in 0..TX_SIZE_CONTEXTS {
            for _ in 0..cols {
                enc.write_bool(0, DIFF_UPDATE_PROB);
            }
        }
    }
}

/// Write the §6.3.7 `read_coef_probs` sweep with no updates: one
/// `L(1) update_probs == 0` per active tx-size slab.
fn write_coef_probs_no_update(enc: &mut BoolEncoder, tx_mode: TxMode) {
    let max_tx_size = tx_mode_to_biggest_tx_size(tx_mode);
    for _ in 0..=max_tx_size {
        enc.write_literal(0, 1);
    }
}

/// Write the §6.3.8 `read_skip_prob` sweep with no updates (three
/// `B(252) == 0` flags).
fn write_skip_prob_no_update(enc: &mut BoolEncoder) {
    for _ in 0..3 {
        enc.write_bool(0, DIFF_UPDATE_PROB);
    }
}

/// Emit a complete §6.3 intra compressed header that leaves all
/// probability tables at their §10.5 defaults, for the given `tx_mode`
/// and `lossless` flag. Returns the coded byte buffer (the
/// `header_size_in_bytes` payload).
pub(crate) fn write_compressed_header_intra(
    tx_mode: TxMode,
    lossless: bool,
) -> Result<Vec<u8>, Error> {
    if lossless && !matches!(tx_mode, TxMode::Only4x4) {
        return Err(Error::Unsupported);
    }
    let mut enc = BoolEncoder::new();
    write_tx_mode(&mut enc, tx_mode, lossless);
    if matches!(tx_mode, TxMode::TxModeSelect) {
        write_tx_mode_probs_no_update(&mut enc);
    }
    write_coef_probs_no_update(&mut enc, tx_mode);
    write_skip_prob_no_update(&mut enc);
    Ok(enc.finish())
}

/// Emit `n` `diff_update_prob` no-update flags (each a `B(252) == 0`).
fn write_diff_no_update(enc: &mut BoolEncoder, n: usize) {
    for _ in 0..n {
        enc.write_bool(0, DIFF_UPDATE_PROB);
    }
}

/// §6.3.12 `frame_reference_mode( )` inverse. `compound_reference_allowed`
/// is derived from the §6.2.5 `ref_frame_sign_bias` exactly as the parser
/// does; when it is false only `SingleReference` is codable (no bits).
/// Otherwise a `non_single_reference L(1)` is coded, and when set a
/// `reference_select L(1)` distinguishes `CompoundReference` (0) from
/// `ReferenceModeSelect` (1).
fn write_frame_reference_mode(
    enc: &mut BoolEncoder,
    reference_mode: ReferenceMode,
    sign_bias: &[bool; 4],
) -> Result<(), Error> {
    // compoundReferenceAllowed: set when any of GOLDEN / ALTREF sign bias
    // differs from LAST (§3 ref indices 1 / 2 / 3).
    let last_bias = sign_bias[1];
    let compound_allowed = sign_bias[2] != last_bias || sign_bias[3] != last_bias;
    if !compound_allowed {
        if reference_mode != ReferenceMode::SingleReference {
            return Err(Error::Unsupported);
        }
        return Ok(());
    }
    match reference_mode {
        ReferenceMode::SingleReference => enc.write_literal(0, 1),
        ReferenceMode::CompoundReference => {
            enc.write_literal(1, 1);
            enc.write_literal(0, 1);
        }
        ReferenceMode::ReferenceModeSelect => {
            enc.write_literal(1, 1);
            enc.write_literal(1, 1);
        }
    }
    Ok(())
}

/// §6.3.13 `frame_reference_mode_probs( )` no-update sweep — the three
/// nested `diff_update_prob` sweeps gated by `reference_mode` exactly as
/// the parser fires them.
fn write_frame_reference_mode_probs_no_update(
    enc: &mut BoolEncoder,
    reference_mode: ReferenceMode,
) {
    if reference_mode == ReferenceMode::ReferenceModeSelect {
        write_diff_no_update(enc, COMP_MODE_CONTEXTS);
    }
    if reference_mode != ReferenceMode::CompoundReference {
        write_diff_no_update(enc, REF_CONTEXTS * 2);
    }
    if reference_mode != ReferenceMode::SingleReference {
        write_diff_no_update(enc, REF_CONTEXTS);
    }
}

/// §6.3.16 `mv_probs( )` no-update sweep — every `update_mv_prob` cell is
/// a `B(252) == 0`. The cell layout mirrors the parser's four phases:
/// joints (3) + per-component bulk (2 × 22) + per-component fractional
/// (2 × 9) + the conditional high-precision tail (2 × 2 when
/// `allow_high_precision_mv`).
fn write_mv_probs_no_update(enc: &mut BoolEncoder, allow_high_precision_mv: bool) {
    use crate::mode_info::{CLASS0_SIZE, MV_CLASSES, MV_FR_SIZE, MV_JOINTS, MV_OFFSET_BITS};
    // Phase 1: joint probs.
    write_diff_no_update(enc, MV_JOINTS - 1);
    // Phase 2: per-component bulk — sign + class + class0_bit + bits.
    let bulk = 1 + (MV_CLASSES - 1) + 1 + MV_OFFSET_BITS;
    write_diff_no_update(enc, 2 * bulk);
    // Phase 3: per-component fractional — class0_fr + fr.
    let frac = CLASS0_SIZE * (MV_FR_SIZE - 1) + (MV_FR_SIZE - 1);
    write_diff_no_update(enc, 2 * frac);
    // Phase 4 (conditional): class0_hp + hp.
    if allow_high_precision_mv {
        write_diff_no_update(enc, 2 * 2);
    }
}

/// Emit a complete §6.3 **inter** compressed header that leaves all
/// probability tables at their §10.5 defaults (no forward updates), for
/// the given `tx_mode` / `lossless` / `reference_mode` /
/// `interpolation_filter` / `allow_high_precision_mv` and the §6.2.5
/// `ref_frame_sign_bias`.
///
/// The walk mirrors `parse_compressed_header_inter_with_ctx`: the
/// intra-shared prefix (§6.3.1 / §6.3.2 / §6.3.7 / §6.3.8), then the
/// inter tail — §6.3.9 `read_inter_mode_probs`, the §6.3.10
/// `read_interp_filter_probs` (only when the frame filter is SWITCHABLE),
/// §6.3.11 `read_is_inter_probs`, §6.3.12 `frame_reference_mode`, §6.3.13
/// `frame_reference_mode_probs`, §6.3.14 `read_y_mode_probs`, §6.3.15
/// `read_partition_probs`, and §6.3.16 `mv_probs` — every
/// `diff_update_prob` / `update_mv_prob` coded as a no-update flag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compressed_header_inter(
    tx_mode: TxMode,
    lossless: bool,
    reference_mode: ReferenceMode,
    interpolation_filter_is_switchable: bool,
    allow_high_precision_mv: bool,
    sign_bias: &[bool; 4],
) -> Result<Vec<u8>, Error> {
    if lossless && !matches!(tx_mode, TxMode::Only4x4) {
        return Err(Error::Unsupported);
    }
    let mut enc = BoolEncoder::new();

    // §6.3 intra-shared prefix.
    write_tx_mode(&mut enc, tx_mode, lossless);
    if matches!(tx_mode, TxMode::TxModeSelect) {
        write_tx_mode_probs_no_update(&mut enc);
    }
    write_coef_probs_no_update(&mut enc, tx_mode);
    write_skip_prob_no_update(&mut enc);

    // §6.3.9 read_inter_mode_probs.
    write_diff_no_update(&mut enc, INTER_MODE_CONTEXTS * (INTER_MODES - 1));
    // §6.3.10 read_interp_filter_probs (switchable only).
    if interpolation_filter_is_switchable {
        write_diff_no_update(&mut enc, INTERP_FILTER_CONTEXTS * (SWITCHABLE_FILTERS - 1));
    }
    // §6.3.11 read_is_inter_probs.
    write_diff_no_update(&mut enc, IS_INTER_CONTEXTS);
    // §6.3.12 frame_reference_mode.
    write_frame_reference_mode(&mut enc, reference_mode, sign_bias)?;
    // §6.3.13 frame_reference_mode_probs.
    write_frame_reference_mode_probs_no_update(&mut enc, reference_mode);
    // §6.3.14 read_y_mode_probs.
    write_diff_no_update(
        &mut enc,
        crate::mode_info::BLOCK_SIZE_GROUPS * (crate::mode_info::INTRA_MODES - 1),
    );
    // §6.3.15 read_partition_probs.
    write_diff_no_update(
        &mut enc,
        crate::partition::PARTITION_CONTEXTS * (crate::partition::PARTITION_TYPES - 1),
    );
    // §6.3.16 mv_probs.
    write_mv_probs_no_update(&mut enc, allow_high_precision_mv);

    Ok(enc.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed::{parse_compressed_header, DEFAULT_SKIP_PROB, DEFAULT_TX_PROBS};

    fn assert_intra_header_roundtrips(tx_mode: TxMode, lossless: bool) {
        let bytes = write_compressed_header_intra(tx_mode, lossless).expect("write chdr");
        let parsed = parse_compressed_header(&bytes, lossless).expect("parse chdr");
        assert_eq!(parsed.tx_mode, tx_mode, "tx_mode round-trip");
        // No updates → tables stay at defaults.
        assert_eq!(parsed.tx_probs, DEFAULT_TX_PROBS, "tx_probs unchanged");
        assert_eq!(parsed.skip_prob, DEFAULT_SKIP_PROB, "skip_prob unchanged");
        // coef_probs: a no-update sweep leaves the whole bank at default.
        let defaults = crate::coef_probs::DEFAULT_COEF_PROBS;
        assert_eq!(parsed.coef_probs, defaults, "coef_probs unchanged");
    }

    #[test]
    fn only_4x4_intra_header_roundtrips() {
        assert_intra_header_roundtrips(TxMode::Only4x4, false);
    }

    #[test]
    fn lossless_forces_only_4x4_and_roundtrips() {
        assert_intra_header_roundtrips(TxMode::Only4x4, true);
    }

    #[test]
    fn allow_8x8_intra_header_roundtrips() {
        assert_intra_header_roundtrips(TxMode::Allow8x8, false);
    }

    #[test]
    fn allow_16x16_intra_header_roundtrips() {
        assert_intra_header_roundtrips(TxMode::Allow16x16, false);
    }

    #[test]
    fn allow_32x32_intra_header_roundtrips() {
        assert_intra_header_roundtrips(TxMode::Allow32x32, false);
    }

    #[test]
    fn tx_mode_select_intra_header_roundtrips() {
        assert_intra_header_roundtrips(TxMode::TxModeSelect, false);
    }

    #[test]
    fn lossless_rejects_non_only_4x4() {
        assert_eq!(
            write_compressed_header_intra(TxMode::Allow8x8, true).unwrap_err(),
            Error::Unsupported
        );
    }

    #[test]
    fn header_is_nonempty_and_byte_aligned() {
        let bytes = write_compressed_header_intra(TxMode::Only4x4, false).unwrap();
        assert!(!bytes.is_empty());
        // Final byte must not be a superframe marker (BoolEncoder::finish
        // guarantees this).
        assert_ne!(bytes.last().unwrap() & 0xe0, 0xc0);
    }

    // ----- §6.3 inter compressed header -----

    use crate::compressed::{
        parse_compressed_header_inter, FrameContext, RefFrameSignBias,
        Vp9CompressedHeaderInterInputs,
    };

    /// Round-trip the written inter header through `parse_compressed_header_inter`
    /// and assert the parsed reference_mode matches and every probability
    /// bank stays at its §10.5 default.
    fn assert_inter_header_roundtrips(
        tx_mode: TxMode,
        lossless: bool,
        reference_mode: ReferenceMode,
        switchable: bool,
        allow_hp: bool,
        sign_bias: [bool; 4],
    ) {
        let bytes = write_compressed_header_inter(
            tx_mode,
            lossless,
            reference_mode,
            switchable,
            allow_hp,
            &sign_bias,
        )
        .expect("write inter chdr");
        let inputs = Vp9CompressedHeaderInterInputs {
            interpolation_filter_is_switchable: switchable,
            ref_frame_sign_bias: RefFrameSignBias::from_inter_biases(
                sign_bias[1] as u8,
                sign_bias[2] as u8,
                sign_bias[3] as u8,
            ),
            allow_high_precision_mv: allow_hp,
        };
        let parsed =
            parse_compressed_header_inter(&bytes, lossless, inputs).expect("parse inter chdr");
        let def = FrameContext::default();
        assert_eq!(parsed.reference_mode, reference_mode, "reference_mode");
        assert_eq!(parsed.intra.tx_mode, tx_mode, "tx_mode");
        assert_eq!(parsed.inter_mode_probs, def.inter_mode_probs, "inter_mode");
        assert_eq!(parsed.is_inter_prob, def.is_inter_prob, "is_inter");
        assert_eq!(parsed.comp_mode_prob, def.comp_mode_prob, "comp_mode");
        assert_eq!(parsed.single_ref_prob, def.single_ref_prob, "single_ref");
        assert_eq!(parsed.comp_ref_prob, def.comp_ref_prob, "comp_ref");
        assert_eq!(parsed.y_mode_probs, def.y_mode_probs, "y_mode");
        assert_eq!(parsed.partition_probs, def.partition_probs, "partition");
        assert_eq!(parsed.mv_probs, def.mv_probs, "mv_probs");
        assert_eq!(
            parsed.interp_filter_probs, def.interp_filter_probs,
            "interp_filter"
        );
    }

    #[test]
    fn inter_single_reference_header_roundtrips() {
        // Uniform sign bias ⇒ compound not allowed ⇒ SingleReference, no
        // reference-mode bits.
        assert_inter_header_roundtrips(
            TxMode::Only4x4,
            false,
            ReferenceMode::SingleReference,
            false,
            false,
            [false; 4],
        );
    }

    #[test]
    fn inter_compound_reference_header_roundtrips() {
        // ALTREF sign bias differs from LAST ⇒ compound allowed.
        assert_inter_header_roundtrips(
            TxMode::Only4x4,
            false,
            ReferenceMode::CompoundReference,
            false,
            false,
            [false, false, false, true],
        );
    }

    #[test]
    fn inter_reference_mode_select_header_roundtrips() {
        assert_inter_header_roundtrips(
            TxMode::TxModeSelect,
            false,
            ReferenceMode::ReferenceModeSelect,
            false,
            false,
            [false, false, true, false],
        );
    }

    #[test]
    fn inter_switchable_filter_header_roundtrips() {
        assert_inter_header_roundtrips(
            TxMode::Only4x4,
            false,
            ReferenceMode::SingleReference,
            true,
            false,
            [false; 4],
        );
    }

    #[test]
    fn inter_high_precision_mv_header_roundtrips() {
        assert_inter_header_roundtrips(
            TxMode::Only4x4,
            false,
            ReferenceMode::SingleReference,
            false,
            true,
            [false; 4],
        );
    }

    #[test]
    fn inter_all_tx_modes_roundtrip() {
        for tx in [
            TxMode::Only4x4,
            TxMode::Allow8x8,
            TxMode::Allow16x16,
            TxMode::Allow32x32,
            TxMode::TxModeSelect,
        ] {
            assert_inter_header_roundtrips(
                tx,
                false,
                ReferenceMode::SingleReference,
                false,
                false,
                [false; 4],
            );
        }
    }

    #[test]
    fn inter_compound_when_not_allowed_is_rejected() {
        // Uniform sign bias ⇒ compound not allowed ⇒ requesting compound
        // is rejected by the writer.
        let r = write_compressed_header_inter(
            TxMode::Only4x4,
            false,
            ReferenceMode::CompoundReference,
            false,
            false,
            &[false; 4],
        );
        assert_eq!(r.unwrap_err(), Error::Unsupported);
    }
}
