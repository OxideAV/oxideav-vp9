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
use crate::Error;

use crate::compressed::FrameContext;
use crate::entropy_model::{write_diff_update_prob, write_update_mv_prob};

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

/// Write the §6.3.2 `tx_mode_probs` sweep as the `base → coding`
/// transitions (a `B(252) == 0` per unchanged cell). Only emitted when
/// `tx_mode == TX_MODE_SELECT`.
fn write_tx_mode_probs(
    enc: &mut BoolEncoder,
    base: &[[[u8; 3]; 2]; 4],
    coding: &[[[u8; 3]; 2]; 4],
) -> Result<(), Error> {
    // tx_probs_8x8: i < TX_SIZE_CONTEXTS, j < 1.
    // tx_probs_16x16: i < TX_SIZE_CONTEXTS, j < 2.
    // tx_probs_32x32: i < TX_SIZE_CONTEXTS, j < 3.
    for (size, cols) in [(1usize, 1usize), (2, 2), (3, 3)] {
        for i in 0..TX_SIZE_CONTEXTS {
            for j in 0..cols {
                write_diff_update_prob(enc, base[size][i][j], coding[size][i][j])?;
            }
        }
    }
    Ok(())
}

/// Write the §6.3.7 `read_coef_probs` sweep: per active tx-size slab an
/// `L(1) update_probs` flag — 1 (followed by every cell's
/// `diff_update_prob`) when any cell of the slab moves, else 0.
fn write_coef_probs(
    enc: &mut BoolEncoder,
    tx_mode: TxMode,
    base: &crate::coef_probs::CoefProbs,
    coding: &crate::coef_probs::CoefProbs,
) -> Result<(), Error> {
    let max_tx_size = tx_mode_to_biggest_tx_size(tx_mode);
    for t in 0..=max_tx_size {
        if base[t] == coding[t] {
            enc.write_literal(0, 1);
            continue;
        }
        enc.write_literal(1, 1);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for l in 0..max_l {
                        for m in 0..3 {
                            write_diff_update_prob(
                                enc,
                                base[t][i][j][k][l][m],
                                coding[t][i][j][k][l][m],
                            )?;
                        }
                    }
                }
            }
        }
    }
    // Slabs beyond maxTxSize are not coded: the coding bank must not
    // differ from the loaded bank there.
    for t in max_tx_size + 1..4 {
        if base[t] != coding[t] {
            return Err(Error::Unsupported);
        }
    }
    Ok(())
}

/// Write one flat `diff_update_prob` sweep over paired cells.
fn write_diff_sweep(enc: &mut BoolEncoder, base: &[u8], coding: &[u8]) -> Result<(), Error> {
    debug_assert_eq!(base.len(), coding.len());
    for (&b, &c) in base.iter().zip(coding) {
        write_diff_update_prob(enc, b, c)?;
    }
    Ok(())
}

/// Emit a complete §6.3 intra compressed header that leaves all
/// probability tables at their §10.5 defaults, for the given `tx_mode`
/// and `lossless` flag. Returns the coded byte buffer (the
/// `header_size_in_bytes` payload).
pub(crate) fn write_compressed_header_intra(
    tx_mode: TxMode,
    lossless: bool,
) -> Result<Vec<u8>, Error> {
    let defaults = FrameContext::default();
    write_compressed_header_intra_ctx(tx_mode, lossless, &defaults, &defaults)
}

/// Emit a complete §6.3 intra compressed header coding the forward
/// updates `base → coding`: `base` is the §6.1.2 `load_probs( )` bank
/// the decoder holds, `coding` the bank the frame's tile data is coded
/// against (equal to `base` wherever no update is elected). Every
/// table outside the §6.3.1–§6.3.8 sweeps must agree between the two
/// (an intra header cannot code them).
pub(crate) fn write_compressed_header_intra_ctx(
    tx_mode: TxMode,
    lossless: bool,
    base: &FrameContext,
    coding: &FrameContext,
) -> Result<Vec<u8>, Error> {
    if lossless && !matches!(tx_mode, TxMode::Only4x4) {
        return Err(Error::Unsupported);
    }
    let mut enc = BoolEncoder::new();
    write_intra_prefix(&mut enc, tx_mode, lossless, base, coding)?;
    Ok(enc.finish())
}

/// The §6.3.1 / §6.3.2 / §6.3.7 / §6.3.8 prefix shared by both frame
/// kinds.
fn write_intra_prefix(
    enc: &mut BoolEncoder,
    tx_mode: TxMode,
    lossless: bool,
    base: &FrameContext,
    coding: &FrameContext,
) -> Result<(), Error> {
    write_tx_mode(enc, tx_mode, lossless);
    if matches!(tx_mode, TxMode::TxModeSelect) {
        write_tx_mode_probs(enc, &base.tx_probs, &coding.tx_probs)?;
    } else if base.tx_probs != coding.tx_probs {
        return Err(Error::Unsupported);
    }
    write_coef_probs(enc, tx_mode, &base.coef_probs, &coding.coef_probs)?;
    write_diff_sweep(enc, &base.skip_prob, &coding.skip_prob)?;
    Ok(())
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

/// §6.3.13 `frame_reference_mode_probs( )` — the three nested
/// `diff_update_prob` sweeps gated by `reference_mode` exactly as the
/// parser fires them; an ungated table must agree between the banks.
fn write_frame_reference_mode_probs(
    enc: &mut BoolEncoder,
    reference_mode: ReferenceMode,
    base: &FrameContext,
    coding: &FrameContext,
) -> Result<(), Error> {
    if reference_mode == ReferenceMode::ReferenceModeSelect {
        write_diff_sweep(enc, &base.comp_mode_prob, &coding.comp_mode_prob)?;
    } else if base.comp_mode_prob != coding.comp_mode_prob {
        return Err(Error::Unsupported);
    }
    if reference_mode != ReferenceMode::CompoundReference {
        write_diff_sweep(
            enc,
            base.single_ref_prob.as_flattened(),
            coding.single_ref_prob.as_flattened(),
        )?;
    } else if base.single_ref_prob != coding.single_ref_prob {
        return Err(Error::Unsupported);
    }
    if reference_mode != ReferenceMode::SingleReference {
        write_diff_sweep(enc, &base.comp_ref_prob, &coding.comp_ref_prob)?;
    } else if base.comp_ref_prob != coding.comp_ref_prob {
        return Err(Error::Unsupported);
    }
    Ok(())
}

/// §6.3.16 `mv_probs( )` sweep — every cell an `update_mv_prob`
/// transition, in the parser's four phases: joints (3) + per-component
/// bulk (2 × 22) + per-component fractional (2 × 9) + the conditional
/// high-precision tail (2 × 2 when `allow_high_precision_mv`).
fn write_mv_probs(
    enc: &mut BoolEncoder,
    allow_high_precision_mv: bool,
    base: &crate::compressed::MvProbs,
    coding: &crate::compressed::MvProbs,
) -> Result<(), Error> {
    for (&b, &c) in base.joint_probs.iter().zip(&coding.joint_probs) {
        write_update_mv_prob(enc, b, c)?;
    }
    for i in 0..2 {
        write_update_mv_prob(enc, base.sign_prob[i], coding.sign_prob[i])?;
        for (&b, &c) in base.class_probs[i].iter().zip(&coding.class_probs[i]) {
            write_update_mv_prob(enc, b, c)?;
        }
        write_update_mv_prob(enc, base.class0_bit_prob[i], coding.class0_bit_prob[i])?;
        for (&b, &c) in base.bits_prob[i].iter().zip(&coding.bits_prob[i]) {
            write_update_mv_prob(enc, b, c)?;
        }
    }
    for i in 0..2 {
        for j in 0..crate::mode_info::CLASS0_SIZE {
            for (&b, &c) in base.class0_fr_probs[i][j]
                .iter()
                .zip(&coding.class0_fr_probs[i][j])
            {
                write_update_mv_prob(enc, b, c)?;
            }
        }
        for (&b, &c) in base.fr_probs[i].iter().zip(&coding.fr_probs[i]) {
            write_update_mv_prob(enc, b, c)?;
        }
    }
    if allow_high_precision_mv {
        for i in 0..2 {
            write_update_mv_prob(enc, base.class0_hp_prob[i], coding.class0_hp_prob[i])?;
            write_update_mv_prob(enc, base.hp_prob[i], coding.hp_prob[i])?;
        }
    } else if base.class0_hp_prob != coding.class0_hp_prob || base.hp_prob != coding.hp_prob {
        return Err(Error::Unsupported);
    }
    Ok(())
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
    let defaults = FrameContext::default();
    write_compressed_header_inter_ctx(
        tx_mode,
        lossless,
        reference_mode,
        interpolation_filter_is_switchable,
        allow_high_precision_mv,
        sign_bias,
        &defaults,
        &defaults,
    )
}

/// [`write_compressed_header_inter`] coding the forward updates `base →
/// coding` (see [`write_compressed_header_intra_ctx`]): every §6.3.9–
/// §6.3.16 cell the header's gates expose is written as its transition;
/// a gated-off table (or `uv_mode_probs`, which has no sweep) must agree
/// between the banks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compressed_header_inter_ctx(
    tx_mode: TxMode,
    lossless: bool,
    reference_mode: ReferenceMode,
    interpolation_filter_is_switchable: bool,
    allow_high_precision_mv: bool,
    sign_bias: &[bool; 4],
    base: &FrameContext,
    coding: &FrameContext,
) -> Result<Vec<u8>, Error> {
    if lossless && !matches!(tx_mode, TxMode::Only4x4) {
        return Err(Error::Unsupported);
    }
    if base.uv_mode_probs != coding.uv_mode_probs {
        return Err(Error::Unsupported);
    }
    let mut enc = BoolEncoder::new();

    // §6.3 intra-shared prefix.
    write_intra_prefix(&mut enc, tx_mode, lossless, base, coding)?;

    // §6.3.9 read_inter_mode_probs.
    write_diff_sweep(
        &mut enc,
        base.inter_mode_probs.as_flattened(),
        coding.inter_mode_probs.as_flattened(),
    )?;
    // §6.3.10 read_interp_filter_probs (switchable only).
    if interpolation_filter_is_switchable {
        write_diff_sweep(
            &mut enc,
            base.interp_filter_probs.as_flattened(),
            coding.interp_filter_probs.as_flattened(),
        )?;
    } else if base.interp_filter_probs != coding.interp_filter_probs {
        return Err(Error::Unsupported);
    }
    // §6.3.11 read_is_inter_probs.
    write_diff_sweep(&mut enc, &base.is_inter_prob, &coding.is_inter_prob)?;
    // §6.3.12 frame_reference_mode.
    write_frame_reference_mode(&mut enc, reference_mode, sign_bias)?;
    // §6.3.13 frame_reference_mode_probs.
    write_frame_reference_mode_probs(&mut enc, reference_mode, base, coding)?;
    // §6.3.14 read_y_mode_probs.
    write_diff_sweep(
        &mut enc,
        base.y_mode_probs.as_flattened(),
        coding.y_mode_probs.as_flattened(),
    )?;
    // §6.3.15 read_partition_probs.
    write_diff_sweep(
        &mut enc,
        base.partition_probs.as_flattened(),
        coding.partition_probs.as_flattened(),
    )?;
    // §6.3.16 mv_probs.
    write_mv_probs(
        &mut enc,
        allow_high_precision_mv,
        &base.mv_probs,
        &coding.mv_probs,
    )?;

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

    // ----- round-455 forward updates -----

    use crate::compressed::parse_compressed_header_with_ctx;

    /// A deterministic pseudo-random walk over a bank: every cell the
    /// §6.3 sweeps can reach is moved to a reachable target (odd for the
    /// MV tables) with probability ~1/3.
    fn perturbed(base: &FrameContext, seed: u32) -> FrameContext {
        let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        let mut moved = |p: &mut u8| {
            if next() % 3 == 0 {
                let target = (next() % 254 + 1) as u8;
                if crate::entropy_model::delta_for(*p, target).is_some() {
                    *p = target;
                }
            }
        };
        let mut c = base.clone();
        for size in 1..4 {
            for ctx in 0..2 {
                for j in 0..size {
                    moved(&mut c.tx_probs[size][ctx][j]);
                }
            }
        }
        for slab in c.coef_probs.iter_mut() {
            for i in slab.iter_mut() {
                for j in i.iter_mut() {
                    for (k, band) in j.iter_mut().enumerate() {
                        let max_l = if k == 0 { 3 } else { 6 };
                        for cell in band.iter_mut().take(max_l) {
                            for p in cell.iter_mut() {
                                moved(p);
                            }
                        }
                    }
                }
            }
        }
        for p in c.skip_prob.iter_mut() {
            moved(p);
        }
        for row in c.inter_mode_probs.iter_mut() {
            for p in row.iter_mut() {
                moved(p);
            }
        }
        for row in c.interp_filter_probs.iter_mut() {
            for p in row.iter_mut() {
                moved(p);
            }
        }
        for p in c.is_inter_prob.iter_mut() {
            moved(p);
        }
        for p in c.comp_mode_prob.iter_mut() {
            moved(p);
        }
        for row in c.single_ref_prob.iter_mut() {
            for p in row.iter_mut() {
                moved(p);
            }
        }
        for p in c.comp_ref_prob.iter_mut() {
            moved(p);
        }
        for row in c.y_mode_probs.iter_mut() {
            for p in row.iter_mut() {
                moved(p);
            }
        }
        for row in c.partition_probs.iter_mut() {
            for p in row.iter_mut() {
                moved(p);
            }
        }
        let mut mv_moved = |p: &mut u8| {
            if next() % 3 == 0 {
                *p = ((next() % 128) << 1) as u8 | 1;
            }
        };
        let mv = &mut c.mv_probs;
        for p in mv.joint_probs.iter_mut() {
            mv_moved(p);
        }
        for i in 0..2 {
            mv_moved(&mut mv.sign_prob[i]);
            for p in mv.class_probs[i].iter_mut() {
                mv_moved(p);
            }
            mv_moved(&mut mv.class0_bit_prob[i]);
            for p in mv.bits_prob[i].iter_mut() {
                mv_moved(p);
            }
            for j in 0..2 {
                for p in mv.class0_fr_probs[i][j].iter_mut() {
                    mv_moved(p);
                }
            }
            for p in mv.fr_probs[i].iter_mut() {
                mv_moved(p);
            }
            mv_moved(&mut mv.class0_hp_prob[i]);
            mv_moved(&mut mv.hp_prob[i]);
        }
        c
    }

    #[test]
    fn intra_forward_updates_roundtrip_through_the_parser() {
        for seed in 0..4u32 {
            let base = perturbed(&FrameContext::default(), 100 + seed);
            let coding = perturbed(&base, seed);
            for tx_mode in [TxMode::Only4x4, TxMode::Allow16x16, TxMode::TxModeSelect] {
                // Tables the intra header cannot code must agree; the
                // writer rejects a bank that differs there.
                let mut c = coding.clone();
                if !matches!(tx_mode, TxMode::TxModeSelect) {
                    c.tx_probs = base.tx_probs;
                }
                let max = crate::compressed::tx_mode_to_biggest_tx_size(tx_mode);
                for t in max + 1..4 {
                    c.coef_probs[t] = base.coef_probs[t];
                }
                let bytes = write_compressed_header_intra_ctx(tx_mode, false, &base, &c)
                    .expect("intra ctx header");
                let parsed = parse_compressed_header_with_ctx(&bytes, false, &base).expect("parse");
                assert_eq!(parsed.tx_mode, tx_mode);
                assert_eq!(
                    parsed.tx_probs, c.tx_probs,
                    "seed {seed} {tx_mode:?}: tx_probs"
                );
                assert_eq!(
                    parsed.coef_probs, c.coef_probs,
                    "seed {seed} {tx_mode:?}: coef"
                );
                assert_eq!(
                    parsed.skip_prob, c.skip_prob,
                    "seed {seed} {tx_mode:?}: skip"
                );
            }
        }
    }

    #[test]
    fn inter_forward_updates_roundtrip_through_the_parser() {
        let sign_bias = [false, false, false, true];
        for seed in 0..4u32 {
            let base = perturbed(&FrameContext::default(), 200 + seed);
            let coding = perturbed(&base, 50 + seed);
            for (reference_mode, switchable, hp) in [
                (ReferenceMode::SingleReference, false, false),
                (ReferenceMode::CompoundReference, true, false),
                (ReferenceMode::ReferenceModeSelect, true, true),
            ] {
                let mut c = coding.clone();
                // Gated-off tables must agree with the base.
                if reference_mode != ReferenceMode::ReferenceModeSelect {
                    c.comp_mode_prob = base.comp_mode_prob;
                }
                if reference_mode == ReferenceMode::CompoundReference {
                    c.single_ref_prob = base.single_ref_prob;
                }
                if reference_mode == ReferenceMode::SingleReference {
                    c.comp_ref_prob = base.comp_ref_prob;
                }
                if !switchable {
                    c.interp_filter_probs = base.interp_filter_probs;
                }
                if !hp {
                    c.mv_probs.class0_hp_prob = base.mv_probs.class0_hp_prob;
                    c.mv_probs.hp_prob = base.mv_probs.hp_prob;
                }
                let bytes = write_compressed_header_inter_ctx(
                    TxMode::TxModeSelect,
                    false,
                    reference_mode,
                    switchable,
                    hp,
                    &sign_bias,
                    &base,
                    &c,
                )
                .expect("inter ctx header");
                let inputs = Vp9CompressedHeaderInterInputs {
                    interpolation_filter_is_switchable: switchable,
                    ref_frame_sign_bias: RefFrameSignBias::from_inter_biases(0, 0, 1),
                    allow_high_precision_mv: hp,
                };
                let parsed = crate::compressed::parse_compressed_header_inter_with_ctx(
                    &bytes, false, inputs, &base,
                )
                .expect("parse");
                let mut got = base.clone();
                got.apply_inter(&parsed);
                assert!(got == c, "seed {seed} {reference_mode:?}: bank mismatch");
                assert_eq!(parsed.reference_mode, reference_mode);
            }
        }
    }

    #[test]
    fn ungated_table_differences_are_rejected() {
        let base = FrameContext::default();
        let mut c = base.clone();
        c.uv_mode_probs[0][0] ^= 1;
        assert_eq!(
            write_compressed_header_inter_ctx(
                TxMode::Only4x4,
                false,
                ReferenceMode::SingleReference,
                false,
                false,
                &[false; 4],
                &base,
                &c,
            )
            .unwrap_err(),
            Error::Unsupported
        );
        let mut c = base.clone();
        c.tx_probs[1][0][0] = 90;
        assert_eq!(
            write_compressed_header_intra_ctx(TxMode::Allow8x8, false, &base, &c).unwrap_err(),
            Error::Unsupported
        );
    }

    #[test]
    fn default_banks_reproduce_the_no_update_headers() {
        let d = FrameContext::default();
        for tx in [TxMode::Only4x4, TxMode::TxModeSelect] {
            assert_eq!(
                write_compressed_header_intra_ctx(tx, false, &d, &d).unwrap(),
                write_compressed_header_intra(tx, false).unwrap()
            );
        }
    }
}
