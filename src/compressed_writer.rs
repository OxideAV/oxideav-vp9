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
use crate::compressed::{tx_mode_to_biggest_tx_size, TxMode};
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
}
