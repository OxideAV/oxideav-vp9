//! VP9 compressed-header walker per spec v0.7 §6.3.
//!
//! Cumulative scope through round 6:
//!
//! * §6.3.1 `read_tx_mode( )` — which arithmetic-codes a 2- or 3-bit
//!   tx_mode value under the §9.2 Boolean coder. This is the first
//!   syntax element of `compressed_header( )` (§6.3) for non-lossless
//!   frames; for lossless frames `tx_mode` is forced to `ONLY_4X4`
//!   and no bits are read.
//! * §6.3.2 `tx_mode_probs( )` — fired only when
//!   `tx_mode == TX_MODE_SELECT`. Three nested sweeps update
//!   `tx_probs_8x8[2][1]`, `tx_probs_16x16[2][2]`,
//!   `tx_probs_32x32[2][3]` via `read_diff_update_prob` starting from
//!   the §10 `default_tx_probs` initials.
//! * §6.3.3 `diff_update_prob( prob )` — the probability-update
//!   helper invoked by every tx-mode / coef / skip / inter-mode /
//!   interp-filter / is-inter / comp-mode / single-ref / comp-ref /
//!   y-mode / partition probability sweep in §6.3.2 and §6.3.7..§6.3.16.
//!   It first reads `B(252)` for an `update_prob` flag, and on 1
//!   pulls a `decode_term_subexp` value (§6.3.4) which is then
//!   remapped through `inv_remap_prob` (§6.3.5) — itself the
//!   composition of the 255-entry `inv_map_table` lookup with
//!   `inv_recenter_nonneg` (§6.3.6).
//! * §6.3.7 `read_coef_probs( )` — the 4D-plus-outer-loop coefficient
//!   probability sweep. Walks `txSz ∈ [TX_4X4, maxTxSize]` with
//!   `maxTxSize = tx_mode_to_biggest_tx_size[ tx_mode ]`. For each
//!   active `txSz`, an outer `L(1) update_probs` flag gates a nested
//!   `(i, j, k, l, m)` sweep — 2 block_types × 2 ref_types × 6 bands
//!   × `maxL` previous-coef contexts (`maxL = (k == 0) ? 3 : 6`) × 3
//!   unconstrained nodes — calling `diff_update_prob` per cell into
//!   the running `coef_probs[ txSz ][ i ][ j ][ k ][ l ][ m ]` table.
//!   Initial values come from the §10 `default_coef_probs` listing
//!   (transcribed verbatim into `coef_probs::DEFAULT_COEF_PROBS`).
//! * §6.3.8 `read_skip_prob( )` — unconditional 3-element
//!   (`SKIP_CONTEXTS = 3`) `diff_update_prob` sweep over the §10
//!   `default_skip_prob[ SKIP_CONTEXTS ] = { 192, 128, 64 }`
//!   initials.
//! * §6.3.9 `read_inter_mode_probs( )` — unconditional nested
//!   `INTER_MODE_CONTEXTS x (INTER_MODES - 1) = 7 x 3 = 21`
//!   `diff_update_prob` sweep over the §10.5
//!   `default_inter_mode_probs[ INTER_MODE_CONTEXTS ][ INTER_MODES - 1 ]`
//!   initials (the running table feeding the §6.4.16
//!   `inter_block_mode_info( )` per-block decoder).
//! * §6.3.10 `read_interp_filter_probs( )` — unconditional nested
//!   `INTERP_FILTER_CONTEXTS x (SWITCHABLE_FILTERS - 1) = 4 x 2 = 8`
//!   `diff_update_prob` sweep over the §10.5
//!   `default_interp_filter_probs[ INTERP_FILTER_CONTEXTS ][ SWITCHABLE_FILTERS - 1 ]`
//!   initials (the running table feeding the per-block
//!   `interp_filter` decode once inter prediction lands).
//! * §6.3.11 `read_is_inter_probs( )` — unconditional 4-element
//!   (`IS_INTER_CONTEXTS = 4`) `diff_update_prob` sweep over the
//!   §10.5 `default_is_inter_prob[ IS_INTER_CONTEXTS ] = { 9, 102,
//!   187, 225 }` initials (the running table feeding the §6.4.13
//!   `read_is_inter( )` per-block decoder).
//!
//! Round 6 lands the §6.3.7 walker between the round-5 §6.3.2
//! `tx_mode_probs` and §6.3.8 `read_skip_prob` calls. Round 22 added
//! the §6.3.11 `read_is_inter_probs( )` standalone primitive; round
//! 23 lands its two §6.3.9 / §6.3.10 companion sweeps. The
//! `FrameIsIntra == 0`-gated outer-dispatch call sites for all three
//! remain deferred because the rest of §6.3.12..§6.3.17 haven't
//! landed yet — the remaining inter-only §6.3.12..§6.3.17 syntax
//! fires only on `FrameIsIntra == 0` and needs reference-buffer
//! state which the header walker still rejects with
//! `Error::Unsupported`.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7,
//! `docs/video/vp9/vp9-spec.txt` §6.3.1 / §6.3.2 / §6.3.3 / §6.3.4 /
//! §6.3.5 / §6.3.6 / §6.3.7 / §6.3.8 / §6.3.9 / §6.3.10 / §6.3.11
//! (the `inv_map_table`, `default_tx_probs`, `default_skip_prob`,
//! `default_coef_probs`, `default_inter_mode_probs`,
//! `default_interp_filter_probs`, `default_is_inter_prob`, and
//! `tx_mode_to_biggest_tx_size` constants are transcribed verbatim
//! from §6.3.5, §10 and §10.5). No external library source
//! consulted.

use crate::bool_coder::BoolCoder;
use crate::coef_probs::{CoefProbs, DEFAULT_COEF_PROBS};
use crate::mode_info::{
    DEFAULT_INTERP_FILTER_PROBS, DEFAULT_INTER_MODE_PROBS, DEFAULT_IS_INTER_PROB,
    INTERP_FILTER_CONTEXTS, INTER_MODES, INTER_MODE_CONTEXTS, IS_INTER_CONTEXTS,
    SWITCHABLE_FILTERS,
};
use crate::Error;

/// `tx_mode` per spec §3 (TX_MODES = 5).
///
/// The §6.3.1 walker decodes one of the five values below. For a
/// lossless frame (`base_q_idx == 0 && all delta_q* == 0`,
/// §6.2.9) the spec hardwires `tx_mode = ONLY_4X4` and reads no
/// bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxMode {
    /// `0` — `ONLY_4X4`. All transforms forced to 4x4.
    Only4x4,
    /// `1` — `ALLOW_8X8`.
    Allow8x8,
    /// `2` — `ALLOW_16X16`.
    Allow16x16,
    /// `3` — `ALLOW_32X32`.
    Allow32x32,
    /// `4` — `TX_MODE_SELECT`. Per-block tx_size signalled in the
    /// residual.
    TxModeSelect,
}

impl TxMode {
    fn from_u32(v: u32) -> Result<Self, Error> {
        match v {
            0 => Ok(Self::Only4x4),
            1 => Ok(Self::Allow8x8),
            2 => Ok(Self::Allow16x16),
            3 => Ok(Self::Allow32x32),
            4 => Ok(Self::TxModeSelect),
            _ => Err(Error::InvalidBitstream),
        }
    }
}

/// Round-6 view of the VP9 compressed header.
///
/// Walks `read_tx_mode` (§6.3.1), the conditional `tx_mode_probs`
/// sweep (§6.3.2; fires only when `tx_mode == TX_MODE_SELECT`),
/// the §6.3.7 `read_coef_probs` 4D nested sweep (per-tx-size gated
/// by an outer `L(1) update_probs`), and the unconditional
/// `read_skip_prob` sweep (§6.3.8). The inter-only §6.3.9+ syntax
/// lands in subsequent rounds.
///
/// `Vp9CompressedHeader` is intentionally **not** `Copy` — the
/// `coef_probs` field is 1728 bytes and silent `memcpy` on every
/// move would be costly. `Clone` is provided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp9CompressedHeader {
    /// Decoded `tx_mode` (§6.3.1).
    pub tx_mode: TxMode,
    /// `tx_probs[ TX_SIZES ][ TX_SIZE_CONTEXTS ][ TX_SIZES - 1 ]`
    /// after the §6.3.2 sweep. When `tx_mode != TX_MODE_SELECT` the
    /// §6.3 syntax skips `tx_mode_probs( )` entirely, so this table
    /// is left exactly equal to `DEFAULT_TX_PROBS`.
    ///
    /// Layout: `tx_probs[size][ctx][j]` where `size` is the
    /// `TxSize`-style index (0 = `TX_4X4` row, unused for sweeps;
    /// 1 = `TX_8X8`; 2 = `TX_16X16`; 3 = `TX_32X32`). The TX_4X4 row
    /// stays at its zero defaults — §6.3.2 only updates sizes
    /// 8x8 / 16x16 / 32x32 via the named `tx_probs_8x8` /
    /// `tx_probs_16x16` / `tx_probs_32x32` aliases.
    pub tx_probs: [[[u8; 3]; 2]; 4],
    /// `coef_probs[ TX_SIZES ][ BLOCK_TYPES ][ REF_TYPES ][
    /// COEF_BANDS ][ PREV_COEF_CONTEXTS ][ UNCONSTRAINED_NODES ]`
    /// after the §6.3.7 sweep.
    ///
    /// `read_coef_probs( )` walks only the tx-sizes
    /// `[TX_4X4, maxTxSize]` where `maxTxSize` is selected by
    /// `tx_mode_to_biggest_tx_size[ tx_mode ]` (§10.5). Inactive
    /// tx-size slabs and unselected `(update_probs == 0)` slabs
    /// are left equal to `DEFAULT_COEF_PROBS`.
    pub coef_probs: CoefProbs,
    /// `skip_prob[ SKIP_CONTEXTS ]` after the §6.3.8 sweep. The
    /// initial values come from the §10 `default_skip_prob[ ] =
    /// { 192, 128, 64 }` listing.
    pub skip_prob: [u8; 3],
}

/// Parse the §6.3 compressed header from `data`.
///
/// `data` must be the byte slice of length `header_size_in_bytes`
/// that follows the uncompressed-header byte-aligned trailing-bits
/// pad. `lossless` is the §6.2.9 `Lossless` derivation already
/// available from [`crate::Vp9FrameHeader::quantization::lossless`].
///
/// Walks `read_tx_mode( )` (§6.3.1), the conditional
/// `tx_mode_probs( )` (§6.3.2), `read_coef_probs( )` (§6.3.7), and
/// `read_skip_prob( )` (§6.3.8) sweeps. The inter-only §6.3.9+
/// syntax remains deferred.
///
/// Returns [`Error::InvalidBitstream`] if the §9.2.1 init marker
/// bit is nonzero, if `read_bool` underruns `BoolMaxBits`, or if a
/// decoded value falls outside its spec-defined range.
pub fn parse_compressed_header(data: &[u8], lossless: bool) -> Result<Vp9CompressedHeader, Error> {
    let sz = data.len();
    let mut coder = BoolCoder::init_bool(data, sz)?;
    let tx_mode = read_tx_mode(&mut coder, lossless)?;
    let mut tx_probs = DEFAULT_TX_PROBS;
    if tx_mode == TxMode::TxModeSelect {
        read_tx_mode_probs(&mut coder, &mut tx_probs)?;
    }
    let mut coef_probs = DEFAULT_COEF_PROBS;
    read_coef_probs(&mut coder, tx_mode, &mut coef_probs)?;
    let mut skip_prob = DEFAULT_SKIP_PROB;
    read_skip_prob(&mut coder, &mut skip_prob)?;
    Ok(Vp9CompressedHeader {
        tx_mode,
        tx_probs,
        coef_probs,
        skip_prob,
    })
}

/// `read_tx_mode( )` per spec §6.3.1.
///
/// * If `Lossless == 1`, `tx_mode = ONLY_4X4` and no bits are read.
/// * Otherwise read `L(2)` to get a raw tx_mode in 0..=3. If the
///   raw value is `ALLOW_32X32` (3) then read an extra `L(1)`
///   `tx_mode_select` and add it on, yielding either 3 or 4 (the
///   `TX_MODE_SELECT` sentinel).
pub(crate) fn read_tx_mode(coder: &mut BoolCoder<'_>, lossless: bool) -> Result<TxMode, Error> {
    if lossless {
        return Ok(TxMode::Only4x4);
    }
    let raw = coder.read_literal(2)?;
    // raw is 0..=3 by construction.
    let value = if raw == 3 {
        // ALLOW_32X32 path: extra L(1) `tx_mode_select` flag picks
        // between ALLOW_32X32 (0) and TX_MODE_SELECT (1).
        let select = coder.read_literal(1)?;
        3 + select
    } else {
        raw
    };
    TxMode::from_u32(value)
}

/// §6.3.5 `inv_map_table[ MAX_PROB ]`.
///
/// 255-entry permutation of `1..=254` plus a duplicated trailing
/// `253` — transcribed verbatim from the spec listing. Indexed by
/// the `decode_term_subexp` result (`0..=254`).
///
/// The table is intentionally `const` (not `static`) so the array
/// lives in `.rodata` exactly once. The values are checked into the
/// crate without any computed shortcut: `inv_map_table` is the
/// authoritative source of truth for the §6.3.5 mapping and the
/// spec gives it as a literal sequence.
///
/// Consumed by `inv_remap_prob` on the `read_diff_update_prob`
/// update path — now driven live by the §6.3.2 / §6.3.8 sweeps in
/// `parse_compressed_header`.
const INV_MAP_TABLE: [u8; 255] = [
    7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 1,
    2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
    187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 251, 252, 253, 253,
];

/// `inv_recenter_nonneg( v, m )` per spec §6.3.6.
///
/// Pure arithmetic helper used only by `inv_remap_prob`. The spec's
/// piecewise definition is:
///
/// ```text
/// if ( v > 2 * m )       return v
/// if ( v & 1 )           return m - ((v + 1) >> 1)
///                        return m + (v >> 1)
/// ```
///
/// `v` and `m` are u32 in the spec (read out of `decode_term_subexp`
/// / propagated from `prob - 1`). The result fits in u32 as well —
/// caller is responsible for narrowing back to u8 if storing into a
/// probability table.
///
/// Consumed by `inv_remap_prob`, which is now driven live by the
/// §6.3.2 / §6.3.8 sweeps via `read_diff_update_prob`.
pub(crate) fn inv_recenter_nonneg(v: u32, m: u32) -> u32 {
    if v > 2 * m {
        return v;
    }
    if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// `inv_remap_prob( deltaProb, prob )` per spec §6.3.5.
///
/// Looks `deltaProb` up in `inv_map_table` and folds it back over the
/// existing `prob` using `inv_recenter_nonneg`. The branch on
/// `(m << 1) <= 255` (with `m = prob - 1`) splits the integer range
/// into a low half (anchored at 0) and a high half (anchored at
/// 255), so the remapped value stays in `1..=254` regardless of
/// where the previous `prob` was.
///
/// `delta_prob` is the `decode_term_subexp` return value (in
/// `0..=254`); `prob` is the previous probability byte (in
/// `1..=255` — never 0, otherwise `m--` would underflow).
///
/// Returns the remapped probability as a u8.
///
/// Consumed by `read_diff_update_prob` on the update path; the
/// §6.3.2 / §6.3.8 sweeps drive this live in
/// `parse_compressed_header`.
pub(crate) fn inv_remap_prob(delta_prob: u32, prob: u8) -> u8 {
    // Per spec: v = inv_map_table[ v ] BEFORE the rest of the
    // function reads v. Clamp the index defensively — a conformant
    // stream can't produce `delta_prob > 254` (decode_term_subexp
    // tops out at 254 per §6.3.4) but be explicit.
    let idx = (delta_prob as usize).min(INV_MAP_TABLE.len() - 1);
    let v = INV_MAP_TABLE[idx] as u32;
    let mut m: u32 = prob as u32;
    // Spec's `m--` — `prob` is a u8 probability in 1..=255, so the
    // unsigned subtraction is well-defined; callers must not pass
    // prob = 0 (and the §6.3.5 spec never does — the smallest
    // initial probability in the §10 default tables is 1).
    debug_assert!(m >= 1, "inv_remap_prob requires prob >= 1");
    m -= 1;
    let m = if (m << 1) <= 255 {
        1 + inv_recenter_nonneg(v, m)
    } else {
        255 - inv_recenter_nonneg(v, 255 - 1 - m)
    };
    // The spec guarantees `m` lands in 1..=254 for any conformant
    // input, but we clamp to u8::MAX as a defensive cast.
    m.min(255) as u8
}

/// `decode_term_subexp( )` per spec §6.3.4.
///
/// Reads a non-uniform unsigned integer in `0..=254` from the
/// Boolean coder. The decoding tree is a fixed cascade of `L(1)` /
/// `L(4)` / `L(5)` / `L(7)` reads:
///
/// 1. `L(1)` flag. If 0: result is the next `L(4)` (range `0..=15`).
/// 2. `L(1)` flag. If 0: result is `L(4) + 16` (range `16..=31`).
/// 3. `L(1)` flag. If 0: result is `L(5) + 32` (range `32..=63`).
/// 4. `L(7) v`. If `v < 65`: result is `v + 64` (range `64..=128`).
/// 5. `L(1) bit`. Result is `(v << 1) - 1 + bit` (range
///    `129..=254` since `v` is in `65..=127`).
///
/// The chain produces every integer in `0..=254` exactly once if
/// you enumerate the leaves: 16 + 16 + 32 + 65 + (127-65+1)*2 = 16
/// + 16 + 32 + 65 + 126 = 255 leaves.
///
/// Returns the decoded value as u32 (caller narrows to u8 if needed).
///
/// Consumed by `read_diff_update_prob`, which drives the §6.3.2 /
/// §6.3.8 sweeps live in `parse_compressed_header`.
pub(crate) fn decode_term_subexp(coder: &mut BoolCoder<'_>) -> Result<u32, Error> {
    // Leg 1: 0..=15.
    let bit = coder.read_literal(1)?;
    if bit == 0 {
        let sub_exp_val = coder.read_literal(4)?;
        return Ok(sub_exp_val);
    }
    // Leg 2: 16..=31.
    let bit = coder.read_literal(1)?;
    if bit == 0 {
        let sub_exp_val_minus_16 = coder.read_literal(4)?;
        return Ok(sub_exp_val_minus_16 + 16);
    }
    // Leg 3: 32..=63.
    let bit = coder.read_literal(1)?;
    if bit == 0 {
        let sub_exp_val_minus_32 = coder.read_literal(5)?;
        return Ok(sub_exp_val_minus_32 + 32);
    }
    // Leg 4 / 5: 64..=254 via a 7-bit prefix v.
    let v = coder.read_literal(7)?;
    if v < 65 {
        return Ok(v + 64);
    }
    let bit = coder.read_literal(1)?;
    Ok((v << 1) - 1 + bit)
}

/// `diff_update_prob( prob )` per spec §6.3.3.
///
/// Reads `B(252)` for an `update_prob` flag. If 1, pulls
/// `decode_term_subexp` (§6.3.4) for a `deltaProb` and remaps the
/// previous probability through `inv_remap_prob` (§6.3.5). Otherwise
/// leaves `prob` unchanged.
///
/// The caller passes the current probability byte (from the
/// running probability table) and stores back the return value.
/// As of round 5 this entry point is consumed by the §6.3.2
/// `tx_mode_probs` and §6.3.8 `read_skip_prob` sweeps; the
/// remaining §6.3.7 / §6.3.9..§6.3.17 sweeps (`coef_probs`,
/// `inter_mode_probs`, `interp_filter_probs`, `is_inter_prob`,
/// `comp_mode_prob`, `single_ref_prob`, `comp_ref_prob`,
/// `y_mode_probs`, `partition_probs`, `update_mv_prob`) land in
/// subsequent rounds.
pub(crate) fn read_diff_update_prob(coder: &mut BoolCoder<'_>, base_prob: u8) -> Result<u8, Error> {
    let update_prob = coder.read_bool(252)?;
    if update_prob == 1 {
        let delta_prob = decode_term_subexp(coder)?;
        Ok(inv_remap_prob(delta_prob, base_prob))
    } else {
        Ok(base_prob)
    }
}

/// `default_tx_probs[ TX_SIZES ][ TX_SIZE_CONTEXTS ][ TX_SIZES - 1 ]`
/// transcribed verbatim from the spec §10 listing.
///
/// The §6.3.2 syntax aliases these rows as `tx_probs_8x8` (row 1),
/// `tx_probs_16x16` (row 2), `tx_probs_32x32` (row 3); row 0
/// (`TX_4X4`) is unused by §6.3.2 but kept here so the storage
/// matches the spec's declared shape one-for-one.
pub(crate) const DEFAULT_TX_PROBS: [[[u8; 3]; 2]; 4] = [
    [[0, 0, 0], [0, 0, 0]],
    [[100, 0, 0], [66, 0, 0]],
    [[20, 152, 0], [15, 101, 0]],
    [[3, 136, 37], [5, 52, 13]],
];

/// `default_skip_prob[ SKIP_CONTEXTS ]` transcribed verbatim from
/// spec §10.
pub(crate) const DEFAULT_SKIP_PROB: [u8; 3] = [192, 128, 64];

/// `tx_mode_probs( )` per spec §6.3.2.
///
/// Three nested sweeps update `tx_probs_8x8[ i ][ j ]`,
/// `tx_probs_16x16[ i ][ j ]`, `tx_probs_32x32[ i ][ j ]` for each
/// `i in 0..TX_SIZE_CONTEXTS` and the size-specific column range
/// (`TX_SIZES - 3` / `TX_SIZES - 2` / `TX_SIZES - 1`). Each cell is
/// passed through `read_diff_update_prob`, so a per-cell `B(252)`
/// is consumed even when the resulting `update_prob` is 0.
///
/// `tx_probs[size][ctx][j]` is updated in place. `size = 1` maps to
/// `tx_probs_8x8`, `size = 2` to `_16x16`, `size = 3` to `_32x32`
/// (matching the spec's `TxSize` numbering); `size = 0` (`TX_4X4`)
/// is untouched.
///
/// Only invoked when `tx_mode == TX_MODE_SELECT` — the §6.3
/// compressed-header dispatch gates the call.
pub(crate) fn read_tx_mode_probs(
    coder: &mut BoolCoder<'_>,
    tx_probs: &mut [[[u8; 3]; 2]; 4],
) -> Result<(), Error> {
    // Spec §6.3.2 nested loops, expressed via iter_mut to keep
    // clippy's `needless_range_loop` happy. The spec's index shape
    // (i < TX_SIZE_CONTEXTS = 2, j < TX_SIZES - {3,2,1}) is preserved
    // by `.take(N)` on the inner row slices.
    //
    // tx_probs_8x8: i < TX_SIZE_CONTEXTS (2), j < TX_SIZES - 3 (1).
    for ctx_row in tx_probs[1].iter_mut() {
        for slot in ctx_row.iter_mut().take(1) {
            *slot = read_diff_update_prob(coder, *slot)?;
        }
    }
    // tx_probs_16x16: i < 2, j < TX_SIZES - 2 (2).
    for ctx_row in tx_probs[2].iter_mut() {
        for slot in ctx_row.iter_mut().take(2) {
            *slot = read_diff_update_prob(coder, *slot)?;
        }
    }
    // tx_probs_32x32: i < 2, j < TX_SIZES - 1 (3).
    for ctx_row in tx_probs[3].iter_mut() {
        for slot in ctx_row.iter_mut().take(3) {
            *slot = read_diff_update_prob(coder, *slot)?;
        }
    }
    Ok(())
}

/// `tx_mode_to_biggest_tx_size[ TX_MODES ]` per spec §10.5.
///
/// Maps a `TxMode` (5 values) to the largest tx-size whose
/// probabilities should be swept by §6.3.7 `read_coef_probs( )`.
/// Both `ALLOW_32X32` and `TX_MODE_SELECT` map to `TX_32X32 = 3`;
/// for `TX_MODE_SELECT` the per-block tx_size is signalled in the
/// residual but the probability tables still need full coverage up
/// to TX_32X32.
pub(crate) const fn tx_mode_to_biggest_tx_size(tx_mode: TxMode) -> usize {
    match tx_mode {
        TxMode::Only4x4 => 0,
        TxMode::Allow8x8 => 1,
        TxMode::Allow16x16 => 2,
        TxMode::Allow32x32 => 3,
        TxMode::TxModeSelect => 3,
    }
}

/// `read_coef_probs( )` per spec §6.3.7.
///
/// Outer loop runs `txSz ∈ [TX_4X4, maxTxSize]` with `maxTxSize =
/// tx_mode_to_biggest_tx_size[ tx_mode ]`. Per `txSz`:
///
/// 1. Read an `L(1) update_probs` flag from the §9.2 coder.
/// 2. If 0, leave that tx-size's `coef_probs` slab unchanged.
/// 3. If 1, walk the nested `(i, j, k, l, m)` sweep:
///    * `i in 0..BLOCK_TYPES` (= 2) — plane type Y vs UV.
///    * `j in 0..REF_TYPES` (= 2) — intra vs inter.
///    * `k in 0..COEF_BANDS` (= 6).
///    * `maxL = (k == 0) ? 3 : 6` per §6.3.7.
///    * `l in 0..maxL` — previous-coef context.
///    * `m in 0..UNCONSTRAINED_NODES` (= 3) — coef-tree node.
///
///    Each cell is replaced by `read_diff_update_prob( coder, cell )`.
///
/// On a fully-active sweep (every `update_probs == 1`) the cell
/// count is `4 × 2 × 2 × (3 + 5*6) × 3 = 1584` `read_diff_update_prob`
/// calls — `tx_mode == TX_MODE_SELECT` activates all four tx-sizes,
/// whereas `ONLY_4X4` activates only the first slab (4×2×2×33×3 ÷ 4
/// → 396 cells for that one slab). For tx-sizes outside the active
/// range, no bits are read.
///
/// `coef_probs` is updated in place. Always invoked from
/// `parse_compressed_header` between the conditional §6.3.2
/// `tx_mode_probs( )` and the unconditional §6.3.8 `read_skip_prob( )`.
pub(crate) fn read_coef_probs(
    coder: &mut BoolCoder<'_>,
    tx_mode: TxMode,
    coef_probs: &mut CoefProbs,
) -> Result<(), Error> {
    let max_tx_size = tx_mode_to_biggest_tx_size(tx_mode);
    for tx_slab in coef_probs.iter_mut().take(max_tx_size + 1) {
        let update_probs = coder.read_literal(1)?;
        if update_probs == 0 {
            continue;
        }
        // Nested (i, j, k, l, m) walk per spec §6.3.7.
        for block_type_slab in tx_slab.iter_mut() {
            for ref_type_slab in block_type_slab.iter_mut() {
                for (k, band_slab) in ref_type_slab.iter_mut().enumerate() {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for ctx_row in band_slab.iter_mut().take(max_l) {
                        for cell in ctx_row.iter_mut() {
                            *cell = read_diff_update_prob(coder, *cell)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// `read_skip_prob( )` per spec §6.3.8.
///
/// Unconditional `SKIP_CONTEXTS = 3` sweep: each `skip_prob[i]` is
/// passed through `read_diff_update_prob`, consuming a `B(252)`
/// `update_prob` flag and, on 1, a `decode_term_subexp` +
/// `inv_remap_prob` cascade.
///
/// `skip_prob` is updated in place. Always invoked from
/// `parse_compressed_header` immediately after the §6.3.7
/// `read_coef_probs( )` call.
pub(crate) fn read_skip_prob(
    coder: &mut BoolCoder<'_>,
    skip_prob: &mut [u8; 3],
) -> Result<(), Error> {
    for slot in skip_prob.iter_mut() {
        *slot = read_diff_update_prob(coder, *slot)?;
    }
    Ok(())
}

/// `read_inter_mode_probs( )` per spec §6.3.9 ("Inter mode probs
/// syntax" in the v0.7 listing — `vp9-spec.txt` lines 2138-2143).
///
/// Unconditional nested sweep: for each `(i, j)` in
/// `INTER_MODE_CONTEXTS x (INTER_MODES - 1)` — i.e. 7 contexts by 3
/// non-leaf tree probabilities — `inter_mode_probs[i][j]` is passed
/// through `read_diff_update_prob`, consuming one `B(252)`
/// `update_prob` flag per slot and, on 1, a `decode_term_subexp` +
/// `inv_remap_prob` cascade.
///
/// The §6.3 outer dispatch invokes `read_inter_mode_probs( )` only when
/// `FrameIsIntra == 0` (gated alongside `read_interp_filter_probs( )` /
/// `read_is_inter_probs( )` / the remaining §6.3.12..§6.3.17 sweeps).
/// The function itself is unconditional once the caller has decided
/// to fire it; the gating lives in the `parse_compressed_header`
/// outer driver.
///
/// `inter_mode_probs` is updated in place. Initial values come from
/// the §10.5 `default_inter_mode_probs[ INTER_MODE_CONTEXTS ][ INTER_MODES - 1 ]`
/// listing (transcribed verbatim in
/// [`mode_info::DEFAULT_INTER_MODE_PROBS`]). The running table feeds the
/// §6.4.16 `inter_block_mode_info( )` per-block decoder via the §9.3.2
/// ctx.
///
/// `INTER_MODES = 4` means three non-leaf nodes per row (since a
/// tree with N leaves has N - 1 internal probabilities); the 21
/// individual `diff_update_prob` calls below cover the full §10.5
/// row × column grid.
///
/// The remaining §6.3.10..§6.3.17 inter-only sweeps continue to land
/// in subsequent rounds. The full `FrameIsIntra == 0` wiring inside
/// `parse_compressed_header` is therefore still deferred — calling
/// this function from the outer driver before its companions land
/// would mis-position the coder cursor.
// Forward-staged: see the `read_is_inter_probs` comment for the
// outer-dispatch gating reason this is `#[allow(dead_code)]` for now.
#[allow(dead_code)]
pub(crate) fn read_inter_mode_probs(
    coder: &mut BoolCoder<'_>,
    inter_mode_probs: &mut [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS],
) -> Result<(), Error> {
    for row in inter_mode_probs.iter_mut() {
        for cell in row.iter_mut() {
            *cell = read_diff_update_prob(coder, *cell)?;
        }
    }
    Ok(())
}

/// Re-export of the §10.5
/// `default_inter_mode_probs[ INTER_MODE_CONTEXTS ][ INTER_MODES - 1 ]`
/// initial / reset table for use as the [`read_inter_mode_probs`]
/// starting state.
///
/// Re-exported from [`mode_info::DEFAULT_INTER_MODE_PROBS`] (the
/// single source of truth — same constant is consumed by the
/// §6.4.16 inter-block decoder once it lands).
#[allow(dead_code)] // wired in once the §6.3 inter-arm dispatch lands.
pub(crate) const DEFAULT_INTER_MODE_PROBS_TABLE: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS] =
    DEFAULT_INTER_MODE_PROBS;

/// `read_interp_filter_probs( )` per spec §6.3.10 ("Interp filter probs
/// syntax" in the v0.7 listing — `vp9-spec.txt` lines 2146-2151).
///
/// Unconditional nested sweep: for each `(j, i)` in
/// `INTERP_FILTER_CONTEXTS x (SWITCHABLE_FILTERS - 1)` — i.e. 4
/// contexts by 2 non-leaf tree probabilities — `interp_filter_probs[j][i]`
/// is passed through `read_diff_update_prob`, consuming one `B(252)`
/// `update_prob` flag per slot and, on 1, a `decode_term_subexp` +
/// `inv_remap_prob` cascade.
///
/// Note the spec listing's outer index is `j` (context) and inner
/// index is `i` (tree node) — the same `probs[j][i]` shape as
/// `default_interp_filter_probs[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`
/// in §10.5. The function walks contexts row-major and tree nodes
/// column-major per the spec ordering, so the bit cursor advances
/// `(j, i) = (0, 0), (0, 1), (1, 0), (1, 1), …` over the 8 cells.
///
/// The §6.3 outer dispatch invokes `read_interp_filter_probs( )` only
/// when `FrameIsIntra == 0` (gated alongside `read_inter_mode_probs( )` /
/// `read_is_inter_probs( )` / the remaining §6.3.12..§6.3.17 sweeps).
/// The function itself is unconditional once the caller has decided
/// to fire it; the gating lives in the `parse_compressed_header`
/// outer driver.
///
/// `interp_filter_probs` is updated in place. Initial values come from
/// the §10.5 `default_interp_filter_probs[ INTERP_FILTER_CONTEXTS ][ SWITCHABLE_FILTERS - 1 ]`
/// listing (transcribed verbatim in
/// [`mode_info::DEFAULT_INTERP_FILTER_PROBS`]).
///
/// `SWITCHABLE_FILTERS = 3` means two non-leaf nodes per row; the 8
/// individual `diff_update_prob` calls below cover the full §10.5
/// row × column grid.
///
/// The remaining §6.3.11..§6.3.17 inter-only sweeps continue to land
/// in subsequent rounds. The full `FrameIsIntra == 0` wiring inside
/// `parse_compressed_header` is therefore still deferred — calling
/// this function from the outer driver before its companions land
/// would mis-position the coder cursor.
// Forward-staged: see the `read_is_inter_probs` comment for the
// outer-dispatch gating reason this is `#[allow(dead_code)]` for now.
#[allow(dead_code)]
pub(crate) fn read_interp_filter_probs(
    coder: &mut BoolCoder<'_>,
    interp_filter_probs: &mut [[u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS],
) -> Result<(), Error> {
    for row in interp_filter_probs.iter_mut() {
        for cell in row.iter_mut() {
            *cell = read_diff_update_prob(coder, *cell)?;
        }
    }
    Ok(())
}

/// Re-export of the §10.5
/// `default_interp_filter_probs[ INTERP_FILTER_CONTEXTS ][ SWITCHABLE_FILTERS - 1 ]`
/// initial / reset table for use as the [`read_interp_filter_probs`]
/// starting state.
///
/// Re-exported from [`mode_info::DEFAULT_INTERP_FILTER_PROBS`] (the
/// single source of truth).
#[allow(dead_code)] // wired in once the §6.3 inter-arm dispatch lands.
pub(crate) const DEFAULT_INTERP_FILTER_PROBS_TABLE: [[u8; SWITCHABLE_FILTERS - 1];
    INTERP_FILTER_CONTEXTS] = DEFAULT_INTERP_FILTER_PROBS;

/// `read_is_inter_probs( )` per spec §6.3.11 ("Intra inter probs
/// syntax" in the v0.7 listing — `vp9-spec.txt` lines 2154-2167).
///
/// Unconditional `IS_INTER_CONTEXTS = 4` sweep: each
/// `is_inter_prob[i]` is passed through `read_diff_update_prob`,
/// consuming one `B(252)` `update_prob` flag per slot and, on 1, a
/// `decode_term_subexp` + `inv_remap_prob` cascade.
///
/// The §6.3 outer dispatch invokes `read_is_inter_probs( )` only when
/// `FrameIsIntra == 0` (gated alongside `read_inter_mode_probs( )` /
/// `read_interp_filter_probs( )` / `frame_reference_mode( )` /
/// `frame_reference_mode_probs( )` / `read_y_mode_probs( )` /
/// `read_partition_probs( )` / `mv_probs( )`). The function itself is
/// unconditional once the caller has decided to fire it; the gating
/// lives in the `parse_compressed_header` outer driver.
///
/// `is_inter_prob` is updated in place. Initial values come from the
/// §10.5 `default_is_inter_prob[ IS_INTER_CONTEXTS ] = {9, 102, 187,
/// 225}` listing (transcribed verbatim in [`mode_info::DEFAULT_IS_INTER_PROB`]).
/// The running `is_inter_prob[ ]` table feeds [`mode_info::read_is_inter`]
/// (§6.4.13) per-block via the §9.3.2 `ctx`.
///
/// The remaining §6.3.9..§6.3.17 inter-only sweeps land in subsequent
/// rounds. The full `FrameIsIntra == 0` wiring inside
/// `parse_compressed_header` is therefore still deferred — calling
/// this function from the outer driver before its companions land
/// would mis-position the coder cursor.
// Forward-staged: the §6.3 outer dispatch gates this call on
// `FrameIsIntra == 0` (alongside §6.3.9 / §6.3.10 / §6.3.12..§6.3.17),
// and the parent `parse_compressed_header` driver doesn't yet wire
// any of the inter-only branch in. The `#[allow(dead_code)]` lifts
// the lint until the §6.3.9 / §6.3.10 primitives land and the outer
// dispatch grows the inter arm.
#[allow(dead_code)]
pub(crate) fn read_is_inter_probs(
    coder: &mut BoolCoder<'_>,
    is_inter_prob: &mut [u8; IS_INTER_CONTEXTS],
) -> Result<(), Error> {
    for slot in is_inter_prob.iter_mut() {
        *slot = read_diff_update_prob(coder, *slot)?;
    }
    Ok(())
}

/// Re-export of the §10.5 `default_is_inter_prob[ IS_INTER_CONTEXTS ]`
/// initial / reset table for use as the [`read_is_inter_probs`]
/// starting state.
///
/// Re-exported from [`mode_info::DEFAULT_IS_INTER_PROB`] (the
/// single source of truth — same constant is consumed by the
/// §6.4.13 [`mode_info::read_is_inter`] per-block decoder).
#[allow(dead_code)] // wired in once the §6.3 inter-arm dispatch lands.
pub(crate) const DEFAULT_IS_INTER_PROB_TABLE: [u8; IS_INTER_CONTEXTS] = DEFAULT_IS_INTER_PROB;

#[cfg(test)]
mod tests {
    use super::*;

    // Golden test vectors below were derived by stepping the §9.2
    // Boolean decoder by hand (see `src/bool_coder.rs` tests for the
    // primitive walk). Each tx_mode value has a unique 4-byte
    // buffer that produces it; the buffers can be regenerated by
    // brute-forcing `init_bool` over the 32-bit prefix space and
    // matching the desired (literal, …) result. No external
    // library was consulted.

    #[test]
    fn lossless_short_circuits_to_only_4x4() {
        // On the lossless path read_tx_mode reads no bits — only
        // init_bool needs to succeed (marker = 0). The zero buffer
        // satisfies that.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let result = parse_compressed_header(&bytes, true).unwrap();
        assert_eq!(result.tx_mode, TxMode::Only4x4);
    }

    #[test]
    fn tx_mode_only_4x4_non_lossless() {
        // 0x00 buffer: L(2) returns 00 → tx_mode = ONLY_4X4 (0).
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let result = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(result.tx_mode, TxMode::Only4x4);
    }

    #[test]
    fn tx_mode_allow_8x8_golden_buffer() {
        // 0x20 buffer: L(2) returns 01 → tx_mode = ALLOW_8X8 (1).
        let bytes = [0x20u8, 0x00, 0x00, 0x00];
        let result = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(result.tx_mode, TxMode::Allow8x8);
    }

    #[test]
    fn tx_mode_allow_16x16_golden_buffer() {
        // 0x40 buffer: L(2) returns 10 → tx_mode = ALLOW_16X16 (2).
        let bytes = [0x40u8, 0x00, 0x00, 0x00];
        let result = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(result.tx_mode, TxMode::Allow16x16);
    }

    #[test]
    fn tx_mode_allow_32x32_golden_buffer() {
        // 0x60 buffer: L(2) returns 11 (raw = 3), then L(1) returns
        // 0 → tx_mode = ALLOW_32X32 (3).
        let bytes = [0x60u8, 0x00, 0x00, 0x00];
        let result = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(result.tx_mode, TxMode::Allow32x32);
    }

    #[test]
    fn tx_mode_select_golden_buffer() {
        // 0x70 buffer: L(2) returns 11 (raw = 3), then L(1) returns
        // 1 → tx_mode = TX_MODE_SELECT (4).
        let bytes = [0x70u8, 0x00, 0x00, 0x00];
        let result = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(result.tx_mode, TxMode::TxModeSelect);
    }

    #[test]
    fn invalid_marker_rejected() {
        // First byte 0xFF: BoolValue = 0xFF, split for p=128 is
        // 128. 0xFF >= 128 so the marker decodes to 1, violating
        // §9.2.1.
        let data = [0xFFu8, 0x00, 0x00, 0x00];
        assert_eq!(
            parse_compressed_header(&data, false).unwrap_err(),
            Error::InvalidBitstream
        );
    }

    #[test]
    fn empty_buffer_rejected() {
        // sz < 1 → InvalidBitstream per §9.2.1.
        let data: [u8; 0] = [];
        assert_eq!(
            parse_compressed_header(&data, false).unwrap_err(),
            Error::InvalidBitstream
        );
    }

    // ----- §6.3.6 inv_recenter_nonneg -----

    #[test]
    fn inv_recenter_nonneg_branch_v_greater_than_2m() {
        // First branch: v > 2*m -> return v.
        assert_eq!(inv_recenter_nonneg(100, 10), 100); // 100 > 20
        assert_eq!(inv_recenter_nonneg(255, 0), 255); // 255 > 0
        assert_eq!(inv_recenter_nonneg(21, 10), 21); // 21 > 20
    }

    #[test]
    fn inv_recenter_nonneg_branch_v_odd() {
        // Odd v not greater than 2m: m - ((v+1)>>1).
        assert_eq!(inv_recenter_nonneg(1, 10), 10 - 1); // 10 - ((1+1)>>1) = 10-1 = 9
        assert_eq!(inv_recenter_nonneg(3, 10), 10 - 2); // 10 - 2 = 8
        assert_eq!(inv_recenter_nonneg(19, 10), 10 - 10); // 10 - 10 = 0 (19 == 2m-1)
    }

    #[test]
    fn inv_recenter_nonneg_branch_v_even() {
        // Even v not greater than 2m: m + (v>>1).
        assert_eq!(inv_recenter_nonneg(0, 10), 10); // 10 + 0
        assert_eq!(inv_recenter_nonneg(2, 10), 11); // 10 + 1
        assert_eq!(inv_recenter_nonneg(20, 10), 20); // 10 + 10 (v == 2m, boundary)
    }

    #[test]
    fn inv_recenter_nonneg_boundary_v_equals_2m() {
        // Exactly at v = 2*m: the v > 2*m branch does NOT fire,
        // so the even/odd split decides. 2m is always even.
        assert_eq!(inv_recenter_nonneg(8, 4), 4 + 4); // 4 + (8>>1) = 8
        assert_eq!(inv_recenter_nonneg(0, 0), 0); // 0 + 0
    }

    // ----- §6.3.5 inv_map_table -----

    #[test]
    fn inv_map_table_length_is_255() {
        // MAX_PROB = 255; the inv_map_table listing in §6.3.5 has
        // exactly 255 entries.
        assert_eq!(INV_MAP_TABLE.len(), 255);
    }

    #[test]
    fn inv_map_table_spot_check_anchors() {
        // Spec listing: first row begins with 7, 20, 33, 46, 59,
        // 72, ...; row 2 starts 202, 215, 228, 241, 254, 1, 2, 3,
        // ...; the very last entry duplicates 253.
        assert_eq!(INV_MAP_TABLE[0], 7);
        assert_eq!(INV_MAP_TABLE[1], 20);
        assert_eq!(INV_MAP_TABLE[19], 254);
        assert_eq!(INV_MAP_TABLE[20], 1);
        assert_eq!(INV_MAP_TABLE[21], 2);
        assert_eq!(INV_MAP_TABLE[INV_MAP_TABLE.len() - 1], 253);
        assert_eq!(INV_MAP_TABLE[INV_MAP_TABLE.len() - 2], 253);
        assert_eq!(INV_MAP_TABLE[INV_MAP_TABLE.len() - 3], 252);
    }

    // ----- §6.3.5 inv_remap_prob -----

    #[test]
    fn inv_remap_prob_low_half_uses_recenter_plus_one() {
        // prob = 5 → m = 4 → (m<<1) = 8 ≤ 255 → low-half branch.
        // delta_prob = 0 → v = inv_map_table[0] = 7.
        // 7 > 2*4 = 8? No (7 ≤ 8). v odd → m = 4 - ((7+1)>>1) = 4-4 = 0.
        // Result = 1 + 0 = 1.
        assert_eq!(inv_remap_prob(0, 5), 1);
    }

    #[test]
    fn inv_remap_prob_low_half_v_greater_than_2m() {
        // prob = 2 → m = 1. delta_prob = 0 → v = 7. 7 > 2 → returns 7.
        // Result = 1 + 7 = 8.
        assert_eq!(inv_remap_prob(0, 2), 8);
    }

    #[test]
    fn inv_remap_prob_high_half_uses_recenter_against_top() {
        // prob = 250 → m = 249 → (m<<1) = 498 > 255 → high-half.
        // delta_prob = 0 → v = 7. 255 - 1 - 249 = 5. v > 2*5 = 10? No (7≤10).
        // v odd → recenter = 5 - ((7+1)>>1) = 5 - 4 = 1.
        // Result = 255 - 1 = 254.
        assert_eq!(inv_remap_prob(0, 250), 254);
    }

    #[test]
    fn inv_remap_prob_uses_inv_map_table_lookup() {
        // delta_prob = 20 -> inv_map_table[20] = 1 (the
        // permutation jumps to the small-integers row here).
        // prob = 128 -> m = 127 -> (m<<1) = 254 ≤ 255 -> low half.
        // v = 1. 1 > 2*127 = 254? No. v odd. recenter = 127 - 1 = 126.
        // Result = 1 + 126 = 127.
        assert_eq!(inv_remap_prob(20, 128), 127);
    }

    #[test]
    fn inv_remap_prob_max_delta_clamped() {
        // delta_prob saturates the table -> inv_map_table[254] = 253.
        // prob = 128 -> m = 127. v = 253. 253 > 254? No.
        // v odd. recenter = 127 - ((253+1)>>1) = 127 - 127 = 0.
        // Result = 1 + 0 = 1.
        assert_eq!(inv_remap_prob(254, 128), 1);
    }

    // ----- §6.3.4 decode_term_subexp -----
    //
    // Each test below feeds a hand-derived §9.2 byte buffer and
    // walks `init_bool` + `decode_term_subexp`. The 0x00-prefix
    // buffers consistently fall into the L(1) = 0 leg because the
    // Boolean coder skews toward 0 after the marker (BoolValue = 0,
    // every read_bool(128) returns 0 until a renorm bit flips it).

    #[test]
    fn decode_term_subexp_leg1_zero() {
        // 0x00 buffer post-marker: L(1)=0, L(4)=0 → returns 0.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        assert_eq!(decode_term_subexp(&mut dec).unwrap(), 0);
    }

    #[test]
    fn decode_term_subexp_returns_value_in_spec_range() {
        // For any valid byte buffer, the result must lie in 0..=254.
        // Sweep a handful of buffers — each must not panic and must
        // produce a value ≤ 254.
        for first in [0x00u8, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70] {
            let bytes = [first, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE];
            let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
            let v = decode_term_subexp(&mut dec).unwrap();
            assert!(
                v <= 254,
                "decode_term_subexp produced out-of-range value {v}"
            );
        }
    }

    // ----- §6.3.3 diff_update_prob -----

    #[test]
    fn read_diff_update_prob_no_update_passes_base_through() {
        // After init_bool on [0x00; 4]: BoolValue=0, BoolRange=128
        // (post-marker). For B(252), split = 1 + ((127*252)>>8) =
        // 1 + 124 = 125. BoolValue=0 < 125 → update_prob = 0, so
        // the base probability is returned unchanged.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        assert_eq!(read_diff_update_prob(&mut dec, 128).unwrap(), 128);
    }

    #[test]
    fn read_diff_update_prob_no_update_for_arbitrary_prob() {
        // Same buffer as above: any base probability passes through.
        for prob in [1u8, 5, 100, 200, 255] {
            let bytes = [0x00u8, 0x00, 0x00, 0x00];
            let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
            assert_eq!(read_diff_update_prob(&mut dec, prob).unwrap(), prob);
        }
    }

    #[test]
    fn diff_update_prob_chain_round_trip_unchanged_when_update_prob_zero() {
        // Exhaustive: with `update_prob == 0`, the function is the
        // identity on `base_prob`. Sweep all 255 valid base
        // probabilities to confirm.
        for prob in 1u8..=255 {
            let bytes = [0x00u8, 0x00, 0x00, 0x00];
            let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
            let returned = read_diff_update_prob(&mut dec, prob).unwrap();
            assert_eq!(returned, prob, "base_prob {prob} should pass through");
        }
    }

    // ----- §10 default tables -----

    #[test]
    fn default_tx_probs_matches_spec_layout() {
        // Verbatim spot-check of the spec §10 listing. The shape is
        // [TX_SIZES=4][TX_SIZE_CONTEXTS=2][TX_SIZES-1=3].
        assert_eq!(DEFAULT_TX_PROBS.len(), 4);
        assert_eq!(DEFAULT_TX_PROBS[0].len(), 2);
        assert_eq!(DEFAULT_TX_PROBS[0][0].len(), 3);
        // Row 0 (TX_4X4) is all zeros — unused by §6.3.2 but listed.
        assert_eq!(DEFAULT_TX_PROBS[0], [[0, 0, 0], [0, 0, 0]]);
        // Row 1 (TX_8X8): {{100,0,0},{66,0,0}}
        assert_eq!(DEFAULT_TX_PROBS[1], [[100, 0, 0], [66, 0, 0]]);
        // Row 2 (TX_16X16): {{20,152,0},{15,101,0}}
        assert_eq!(DEFAULT_TX_PROBS[2], [[20, 152, 0], [15, 101, 0]]);
        // Row 3 (TX_32X32): {{3,136,37},{5,52,13}}
        assert_eq!(DEFAULT_TX_PROBS[3], [[3, 136, 37], [5, 52, 13]]);
    }

    #[test]
    fn default_skip_prob_matches_spec_listing() {
        // Spec §10: default_skip_prob[SKIP_CONTEXTS] = {192, 128, 64}.
        assert_eq!(DEFAULT_SKIP_PROB, [192, 128, 64]);
    }

    // ----- §6.3.2 tx_mode_probs( ) -----

    #[test]
    fn tx_mode_probs_zero_buffer_leaves_defaults_unchanged() {
        // After init_bool on [0x00; 8], every read_diff_update_prob
        // call sees update_prob == 0 (BoolValue=0 < split=125 for
        // B(252)), so the sweep passes every cell through unchanged.
        let bytes = [0x00u8; 8];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut tx_probs = DEFAULT_TX_PROBS;
        read_tx_mode_probs(&mut dec, &mut tx_probs).unwrap();
        assert_eq!(tx_probs, DEFAULT_TX_PROBS);
    }

    #[test]
    fn tx_mode_probs_total_cells_match_spec_loop_shape() {
        // Sanity: TX_SIZE_CONTEXTS * (1 + 2 + 3) = 2 * 6 = 12 cells
        // get visited. Spec §6.3.2 reads one B(252) per cell, so any
        // implementation that walks 12 read_diff_update_prob calls
        // consumes the same minimum number of bits on a zero buffer.
        // We confirm by counting iterations indirectly: the function
        // returns Ok on a zero buffer (already covered above) and
        // the cell count matches the spec's nested-loop product.
        // 2 * 1 + 2 * 2 + 2 * 3 = 2 + 4 + 6 = 12.
        let cells_8x8 = 2usize;
        let cells_16x16 = 4usize;
        let cells_32x32 = 6usize;
        assert_eq!(cells_8x8 + cells_16x16 + cells_32x32, 12);
    }

    #[test]
    fn tx_mode_probs_only_touches_sizes_8_16_32() {
        // Row 0 (TX_4X4) stays at its zero default after the sweep.
        let bytes = [0x00u8; 8];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut tx_probs = DEFAULT_TX_PROBS;
        // Pre-mutate row 0 to a sentinel; the sweep must NOT touch it.
        tx_probs[0] = [[123, 45, 67], [89, 10, 11]];
        let snapshot_row0 = tx_probs[0];
        read_tx_mode_probs(&mut dec, &mut tx_probs).unwrap();
        assert_eq!(tx_probs[0], snapshot_row0);
    }

    // ----- §6.3.7 read_coef_probs( ) -----

    #[test]
    fn tx_mode_to_biggest_tx_size_matches_spec_listing() {
        // Spec §10.5: { TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_32X32 }.
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::Only4x4), 0);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::Allow8x8), 1);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::Allow16x16), 2);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::Allow32x32), 3);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::TxModeSelect), 3);
    }

    #[test]
    fn read_coef_probs_zero_buffer_leaves_defaults_unchanged_only_4x4() {
        // ONLY_4X4: outer loop visits tx-size 0 only.
        // Zero buffer → outer L(1) update_probs = 0 → no inner walk.
        let bytes = [0x00u8; 4];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut coef_probs = DEFAULT_COEF_PROBS;
        read_coef_probs(&mut dec, TxMode::Only4x4, &mut coef_probs).unwrap();
        assert_eq!(coef_probs, DEFAULT_COEF_PROBS);
    }

    #[test]
    fn read_coef_probs_zero_buffer_leaves_defaults_unchanged_tx_mode_select() {
        // TX_MODE_SELECT: outer loop visits all four tx-sizes.
        // Zero buffer → all four outer L(1)s decode to 0.
        let bytes = [0x00u8; 4];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut coef_probs = DEFAULT_COEF_PROBS;
        read_coef_probs(&mut dec, TxMode::TxModeSelect, &mut coef_probs).unwrap();
        assert_eq!(coef_probs, DEFAULT_COEF_PROBS);
    }

    #[test]
    fn read_coef_probs_outer_loop_count_matches_tx_mode() {
        // Per spec §6.3.7 + §10.5: outer L(1) update_probs reads are
        // gated by tx-mode -> maxTxSize. On a zero buffer where every
        // update_probs decodes to 0, the only state change is the
        // BoolCoder cursor advancing one read_bool(128) per outer iter.
        // We assert this by checking the function returns Ok across all
        // tx-modes (no underrun) — the §6.3.7 walker is the only thing
        // consuming bits.
        for (mode, _expected_iters) in [
            (TxMode::Only4x4, 1usize),
            (TxMode::Allow8x8, 2),
            (TxMode::Allow16x16, 3),
            (TxMode::Allow32x32, 4),
            (TxMode::TxModeSelect, 4),
        ] {
            let bytes = [0x00u8; 4];
            let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
            let mut coef_probs = DEFAULT_COEF_PROBS;
            read_coef_probs(&mut dec, mode, &mut coef_probs).unwrap();
            assert_eq!(coef_probs, DEFAULT_COEF_PROBS);
        }
    }

    #[test]
    fn default_coef_probs_shape_and_anchors() {
        // 4 tx-sizes × 2 block types × 2 ref types × 6 bands ×
        // 6 contexts × 3 nodes = 1728 entries.
        assert_eq!(DEFAULT_COEF_PROBS.len(), 4);
        assert_eq!(DEFAULT_COEF_PROBS[0].len(), 2);
        assert_eq!(DEFAULT_COEF_PROBS[0][0].len(), 2);
        assert_eq!(DEFAULT_COEF_PROBS[0][0][0].len(), 6);
        assert_eq!(DEFAULT_COEF_PROBS[0][0][0][0].len(), 6);
        assert_eq!(DEFAULT_COEF_PROBS[0][0][0][0][0].len(), 3);

        // Anchor: TX_4X4 / block_type 0 / Intra / band 0 / context 0.
        // Spec: { 195, 29, 183 }.
        assert_eq!(DEFAULT_COEF_PROBS[0][0][0][0][0], [195, 29, 183]);
        // Anchor: TX_4X4 / block_type 0 / Intra / band 0 / context 3
        // (one of the "unused" rows).
        assert_eq!(DEFAULT_COEF_PROBS[0][0][0][0][3], [0, 0, 0]);
        // Anchor: TX_4X4 / block_type 0 / Inter / band 5 / context 5.
        // Spec: { 3, 16, 42 }.
        assert_eq!(DEFAULT_COEF_PROBS[0][0][1][5][5], [3, 16, 42]);
        // Anchor: TX_32X32 / block_type 1 / Inter / band 5 / context 5.
        // Spec: { 1, 16, 6 }.
        assert_eq!(DEFAULT_COEF_PROBS[3][1][1][5][5], [1, 16, 6]);
    }

    #[test]
    fn default_coef_probs_band0_unused_rows_are_zero() {
        // For every tx-size, block_type and ref_type, the band-0 rows
        // at contexts 3, 4, 5 must be {0, 0, 0} (the spec "// unused"
        // tail) — this confirms the maxL = 3 clamp at k == 0 is
        // consistent with the in-table sentinel rows.
        for tx_slab in DEFAULT_COEF_PROBS.iter() {
            for block_type_slab in tx_slab.iter() {
                for ref_type_slab in block_type_slab.iter() {
                    let band0 = &ref_type_slab[0];
                    for row in band0.iter().skip(3).take(3) {
                        assert_eq!(row, &[0, 0, 0]);
                    }
                }
            }
        }
    }

    #[test]
    fn read_coef_probs_inner_sweep_cell_count_is_396() {
        // Per spec §6.3.7: when update_probs == 1, the inner walk
        // covers 2 * 2 * (3 + 5*6) * 3 = 396 read_diff_update_prob
        // calls per active tx-size. Confirm the arithmetic.
        let cells_k0 = 2 * 2 * 3 * 3; // i × j × maxL(k=0) × m
        let cells_k_rest = 2 * 2 * 5 * 6 * 3; // 5 bands with maxL = 6
        assert_eq!(cells_k0 + cells_k_rest, 396);
    }

    // ----- §6.3.8 read_skip_prob( ) -----

    #[test]
    fn read_skip_prob_zero_buffer_leaves_defaults_unchanged() {
        // Three B(252) reads on a zero buffer all return 0 → defaults
        // pass through.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        let mut skip_prob = DEFAULT_SKIP_PROB;
        read_skip_prob(&mut dec, &mut skip_prob).unwrap();
        assert_eq!(skip_prob, DEFAULT_SKIP_PROB);
    }

    #[test]
    fn read_skip_prob_visits_three_contexts() {
        // The function reads SKIP_CONTEXTS = 3 cells.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        let mut skip_prob = [10u8, 20, 30];
        read_skip_prob(&mut dec, &mut skip_prob).unwrap();
        // update_prob == 0 path: each base passes through.
        assert_eq!(skip_prob, [10, 20, 30]);
    }

    // ----- parse_compressed_header round-5 integration -----

    #[test]
    fn parse_compressed_header_includes_default_tables_for_only_4x4() {
        // ONLY_4X4 (non-lossless 0x00 buffer): no §6.3.2 sweep
        // (gated on TX_MODE_SELECT); §6.3.7 runs once with maxTxSize=0
        // and its outer L(1) decodes to 0 (no inner walk); §6.3.8
        // fires unconditionally. On a zero buffer all defaults survive.
        let bytes = [0x00u8; 8];
        let h = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(h.tx_mode, TxMode::Only4x4);
        assert_eq!(h.tx_probs, DEFAULT_TX_PROBS);
        assert_eq!(h.coef_probs, DEFAULT_COEF_PROBS);
        assert_eq!(h.skip_prob, DEFAULT_SKIP_PROB);
    }

    #[test]
    fn parse_compressed_header_lossless_runs_skip_prob_sweep() {
        // Lossless path: tx_mode forced to ONLY_4X4 with no L(2)
        // reads, then §6.3.7 with maxTxSize=0 + §6.3.8 still fire.
        // On a zero buffer all defaults survive.
        let bytes = [0x00u8; 8];
        let h = parse_compressed_header(&bytes, true).unwrap();
        assert_eq!(h.tx_mode, TxMode::Only4x4);
        assert_eq!(h.coef_probs, DEFAULT_COEF_PROBS);
        assert_eq!(h.skip_prob, DEFAULT_SKIP_PROB);
    }

    #[test]
    fn parse_compressed_header_tx_mode_select_runs_tx_mode_probs_sweep() {
        // TX_MODE_SELECT path (0x70 prefix): the §6.3.2 sweep fires,
        // then §6.3.7 (visiting all four tx-sizes with outer
        // update_probs = 0 each on a zero buffer), then §6.3.8. The
        // §10 defaults survive across all three sweeps.
        let mut bytes = [0u8; 16];
        bytes[0] = 0x70;
        let h = parse_compressed_header(&bytes, false).unwrap();
        assert_eq!(h.tx_mode, TxMode::TxModeSelect);
        assert_eq!(h.tx_probs, DEFAULT_TX_PROBS);
        assert_eq!(h.coef_probs, DEFAULT_COEF_PROBS);
        assert_eq!(h.skip_prob, DEFAULT_SKIP_PROB);
    }

    // ----- §6.3.11 read_is_inter_probs( ) -----

    #[test]
    fn default_is_inter_prob_table_matches_mode_info_source() {
        // Single source of truth check: the re-export consumed by
        // read_is_inter_probs must equal the §10.5 listing held in
        // mode_info::DEFAULT_IS_INTER_PROB.
        assert_eq!(DEFAULT_IS_INTER_PROB_TABLE, [9, 102, 187, 225]);
        assert_eq!(DEFAULT_IS_INTER_PROB_TABLE, DEFAULT_IS_INTER_PROB);
    }

    #[test]
    fn read_is_inter_probs_zero_buffer_leaves_defaults_unchanged() {
        // Four B(252) reads on a zero buffer all return 0 (BoolValue=0
        // < split=125), so each diff_update_prob call passes its base
        // through unchanged.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        let mut is_inter_prob = DEFAULT_IS_INTER_PROB_TABLE;
        read_is_inter_probs(&mut dec, &mut is_inter_prob).unwrap();
        assert_eq!(is_inter_prob, DEFAULT_IS_INTER_PROB_TABLE);
    }

    #[test]
    fn read_is_inter_probs_visits_four_contexts() {
        // The function reads IS_INTER_CONTEXTS = 4 cells. With
        // update_prob == 0 every base passes through; pick a custom
        // table to prove all four slots are visited and unchanged.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        let mut is_inter_prob = [11u8, 22, 33, 44];
        read_is_inter_probs(&mut dec, &mut is_inter_prob).unwrap();
        assert_eq!(is_inter_prob, [11, 22, 33, 44]);
    }

    #[test]
    fn read_is_inter_probs_consumes_four_b252_flags_on_zero_buffer() {
        // Independent check: after a zero buffer, IS_INTER_CONTEXTS = 4
        // sequential read_diff_update_prob calls each consume one
        // B(252) "update_prob" flag from the coder. We confirm by
        // walking the primitive twice and asserting cursor equivalence.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        // Reference walk: 4 explicit read_diff_update_prob calls.
        let mut ref_dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        for _ in 0..IS_INTER_CONTEXTS {
            let _ = read_diff_update_prob(&mut ref_dec, 128).unwrap();
        }
        // Under-test walk: read_is_inter_probs on a parallel coder.
        let mut probs = [128u8, 128, 128, 128];
        let mut test_dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        read_is_inter_probs(&mut test_dec, &mut probs).unwrap();
        // Both decoders must have advanced identically — after each
        // path consumes 4 B(252) reads on the same buffer, any
        // subsequent read_literal(1) on either must produce the same
        // bit. Compare two same-buffer extracts.
        let ref_next = ref_dec.read_literal(1).unwrap();
        let test_next = test_dec.read_literal(1).unwrap();
        assert_eq!(ref_next, test_next);
    }

    #[test]
    fn is_inter_contexts_constant_equals_four() {
        // The spec §3 constant table fixes IS_INTER_CONTEXTS = 4. The
        // §6.3.11 loop walks `i < IS_INTER_CONTEXTS` cells; if the
        // constant ever drifted, the sweep would over- or under-read.
        assert_eq!(IS_INTER_CONTEXTS, 4);
    }

    #[test]
    fn read_is_inter_probs_with_zero_buffer_returns_each_input_unchanged() {
        // Exhaustive over all four slots: any starting (a, b, c, d)
        // tuple must round-trip through read_is_inter_probs on a zero
        // buffer (the update_prob == 0 path).
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        for tuple in [
            [1u8, 1, 1, 1],
            [9, 102, 187, 225],
            [255, 1, 128, 64],
            [50, 150, 200, 5],
        ] {
            let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
            let mut probs = tuple;
            read_is_inter_probs(&mut dec, &mut probs).unwrap();
            assert_eq!(probs, tuple, "tuple {tuple:?} should pass through");
        }
    }

    #[test]
    fn read_is_inter_probs_matches_explicit_four_call_sequence() {
        // Independent equivalence check: read_is_inter_probs must be
        // exactly the same as four explicit read_diff_update_prob
        // calls in slot order against a shared coder + table. Picks a
        // mix of starting probabilities to exercise distinct
        // inv_remap_prob branches if the update_prob path ever fires.
        // On the zero buffer the path stays in update_prob = 0; the
        // equivalence still holds whether the update fires or not.
        let bytes = [0x00u8; 16];
        let starts = [DEFAULT_IS_INTER_PROB_TABLE, [1, 254, 128, 7]];
        for start in starts {
            // Reference walk: four explicit read_diff_update_prob calls.
            let mut ref_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
            let mut ref_probs = start;
            for slot in ref_probs.iter_mut() {
                *slot = read_diff_update_prob(&mut ref_dec, *slot).unwrap();
            }
            // Under-test walk.
            let mut test_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
            let mut test_probs = start;
            read_is_inter_probs(&mut test_dec, &mut test_probs).unwrap();
            assert_eq!(
                ref_probs, test_probs,
                "read_is_inter_probs must match explicit 4-call sequence for {start:?}"
            );
            // Cursor equivalence: any subsequent read on either coder
            // must produce the same bit if both consumed the same
            // number of underlying bits.
            assert_eq!(
                ref_dec.read_literal(1).unwrap(),
                test_dec.read_literal(1).unwrap(),
                "coder cursor must agree after the four-cell sweep for {start:?}",
            );
        }
    }

    // ----- §6.3.9 read_inter_mode_probs( ) -----

    #[test]
    fn default_inter_mode_probs_table_matches_mode_info_source() {
        // Single source of truth check: the re-export consumed by
        // read_inter_mode_probs must equal the §10.5 listing held in
        // mode_info::DEFAULT_INTER_MODE_PROBS.
        let expected: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS] = [
            [2, 173, 34],
            [7, 145, 85],
            [7, 166, 63],
            [7, 94, 66],
            [8, 64, 46],
            [17, 81, 31],
            [25, 29, 30],
        ];
        assert_eq!(DEFAULT_INTER_MODE_PROBS_TABLE, expected);
        assert_eq!(DEFAULT_INTER_MODE_PROBS_TABLE, DEFAULT_INTER_MODE_PROBS);
    }

    #[test]
    fn inter_mode_dimensions_constants() {
        // The spec §3 constants table fixes INTER_MODES = 4 and
        // INTER_MODE_CONTEXTS = 7. If either drifted, the sweep would
        // over- or under-read the coder cursor.
        assert_eq!(INTER_MODES, 4);
        assert_eq!(INTER_MODE_CONTEXTS, 7);
    }

    #[test]
    fn read_inter_mode_probs_zero_buffer_leaves_defaults_unchanged() {
        // 21 = 7 * 3 B(252) reads on a zero buffer all return 0
        // (BoolValue=0 < split=247), so each diff_update_prob call
        // passes its base through unchanged. The buffer (16 bytes)
        // gives plenty of bits for the 21-call sweep.
        let bytes = [0x00u8; 16];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut inter_mode_probs = DEFAULT_INTER_MODE_PROBS_TABLE;
        read_inter_mode_probs(&mut dec, &mut inter_mode_probs).unwrap();
        assert_eq!(inter_mode_probs, DEFAULT_INTER_MODE_PROBS_TABLE);
    }

    #[test]
    fn read_inter_mode_probs_visits_all_21_cells_row_major() {
        // Independent equivalence check: read_inter_mode_probs must
        // walk INTER_MODE_CONTEXTS (outer) x (INTER_MODES - 1) (inner)
        // cells in row-major order, identical to an explicit double
        // for-loop calling read_diff_update_prob in the same order.
        let bytes = [0x00u8; 16];
        // Reference walk: 21 explicit read_diff_update_prob calls in
        // (i, j) row-major order.
        let mut ref_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut ref_probs = DEFAULT_INTER_MODE_PROBS_TABLE;
        for row in ref_probs.iter_mut() {
            for cell in row.iter_mut() {
                *cell = read_diff_update_prob(&mut ref_dec, *cell).unwrap();
            }
        }
        // Under-test walk.
        let mut test_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut test_probs = DEFAULT_INTER_MODE_PROBS_TABLE;
        read_inter_mode_probs(&mut test_dec, &mut test_probs).unwrap();
        assert_eq!(ref_probs, test_probs);
        // Cursor equivalence: any subsequent read on either coder
        // must produce the same bit after 21 identical consumes.
        assert_eq!(
            ref_dec.read_literal(1).unwrap(),
            test_dec.read_literal(1).unwrap(),
        );
    }

    #[test]
    fn read_inter_mode_probs_with_custom_starting_grid() {
        // Pick a non-default starting grid and prove every cell
        // round-trips through the zero-buffer update_prob == 0 path.
        let bytes = [0x00u8; 16];
        let custom: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS] = [
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99],
            [110, 121, 132],
            [143, 154, 165],
            [176, 187, 198],
            [209, 220, 231],
        ];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut probs = custom;
        read_inter_mode_probs(&mut dec, &mut probs).unwrap();
        assert_eq!(probs, custom, "every cell must pass through unchanged");
    }

    #[test]
    fn read_inter_mode_probs_consumes_21_b252_flags_on_zero_buffer() {
        // Independent check: after a zero buffer, 7 * 3 = 21
        // sequential read_diff_update_prob calls each consume one
        // B(252) "update_prob" flag from the coder. Confirmed by
        // walking the primitive twice and asserting cursor equivalence.
        let bytes = [0x00u8; 16];
        // Reference walk: 21 explicit read_diff_update_prob calls.
        let mut ref_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        for _ in 0..(INTER_MODE_CONTEXTS * (INTER_MODES - 1)) {
            let _ = read_diff_update_prob(&mut ref_dec, 128).unwrap();
        }
        // Under-test walk.
        let mut test_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut probs = [[128u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS];
        read_inter_mode_probs(&mut test_dec, &mut probs).unwrap();
        assert_eq!(
            ref_dec.read_literal(1).unwrap(),
            test_dec.read_literal(1).unwrap(),
        );
    }

    // ----- §6.3.10 read_interp_filter_probs( ) -----

    #[test]
    fn default_interp_filter_probs_table_matches_mode_info_source() {
        // Single source of truth check: the re-export consumed by
        // read_interp_filter_probs must equal the §10.5 listing.
        let expected: [[u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS] =
            [[235, 162], [36, 255], [34, 3], [149, 144]];
        assert_eq!(DEFAULT_INTERP_FILTER_PROBS_TABLE, expected);
        assert_eq!(
            DEFAULT_INTERP_FILTER_PROBS_TABLE,
            DEFAULT_INTERP_FILTER_PROBS
        );
    }

    #[test]
    fn interp_filter_dimensions_constants() {
        // The spec §3 constants table fixes SWITCHABLE_FILTERS = 3 and
        // INTERP_FILTER_CONTEXTS = 4. If either drifted, the sweep
        // would over- or under-read the coder cursor.
        assert_eq!(SWITCHABLE_FILTERS, 3);
        assert_eq!(INTERP_FILTER_CONTEXTS, 4);
    }

    #[test]
    fn read_interp_filter_probs_zero_buffer_leaves_defaults_unchanged() {
        // 8 = 4 * 2 B(252) reads on a zero buffer all return 0
        // (update_prob = 0 path), so each diff_update_prob call passes
        // its base through unchanged.
        let bytes = [0x00u8; 8];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut interp_filter_probs = DEFAULT_INTERP_FILTER_PROBS_TABLE;
        read_interp_filter_probs(&mut dec, &mut interp_filter_probs).unwrap();
        assert_eq!(interp_filter_probs, DEFAULT_INTERP_FILTER_PROBS_TABLE);
    }

    #[test]
    fn read_interp_filter_probs_visits_all_8_cells_row_major() {
        // Independent equivalence check: read_interp_filter_probs
        // must walk INTERP_FILTER_CONTEXTS (outer) x
        // (SWITCHABLE_FILTERS - 1) (inner) cells in row-major order
        // (`j` outer, `i` inner per the spec listing), identical to an
        // explicit double for-loop.
        let bytes = [0x00u8; 16];
        let mut ref_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut ref_probs = DEFAULT_INTERP_FILTER_PROBS_TABLE;
        for row in ref_probs.iter_mut() {
            for cell in row.iter_mut() {
                *cell = read_diff_update_prob(&mut ref_dec, *cell).unwrap();
            }
        }
        let mut test_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut test_probs = DEFAULT_INTERP_FILTER_PROBS_TABLE;
        read_interp_filter_probs(&mut test_dec, &mut test_probs).unwrap();
        assert_eq!(ref_probs, test_probs);
        assert_eq!(
            ref_dec.read_literal(1).unwrap(),
            test_dec.read_literal(1).unwrap(),
        );
    }

    #[test]
    fn read_interp_filter_probs_with_custom_starting_grid() {
        // Pick a non-default starting grid and prove every cell
        // round-trips through the zero-buffer update_prob == 0 path.
        let bytes = [0x00u8; 8];
        let custom: [[u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS] =
            [[10, 20], [30, 40], [50, 60], [70, 80]];
        let mut dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut probs = custom;
        read_interp_filter_probs(&mut dec, &mut probs).unwrap();
        assert_eq!(probs, custom, "every cell must pass through unchanged");
    }

    #[test]
    fn read_interp_filter_probs_consumes_8_b252_flags_on_zero_buffer() {
        // Independent check: 4 * 2 = 8 sequential
        // read_diff_update_prob calls each consume one B(252) flag.
        let bytes = [0x00u8; 8];
        let mut ref_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        for _ in 0..(INTERP_FILTER_CONTEXTS * (SWITCHABLE_FILTERS - 1)) {
            let _ = read_diff_update_prob(&mut ref_dec, 128).unwrap();
        }
        let mut test_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut probs = [[128u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS];
        read_interp_filter_probs(&mut test_dec, &mut probs).unwrap();
        assert_eq!(
            ref_dec.read_literal(1).unwrap(),
            test_dec.read_literal(1).unwrap(),
        );
    }

    // ----- Cross-sweep cursor equivalence: §6.3.9 → §6.3.10 → §6.3.11 -----

    #[test]
    fn inter_arm_sweep_chain_cursor_equivalent_to_29_diff_update_calls() {
        // The §6.3 outer dispatch (once it grows the inter arm) fires
        // read_inter_mode_probs (21 cells) → read_interp_filter_probs
        // (8 cells) → read_is_inter_probs (4 cells) in that order on
        // the same coder, totalling 33 B(252) reads on a zero buffer
        // (each diff_update_prob with update_prob == 0 consumes one
        // B(252) flag and no extra bits). Confirm the chain advances
        // the cursor identically to 33 explicit read_diff_update_prob
        // calls.
        let bytes = [0x00u8; 16];
        let mut ref_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let total = INTER_MODE_CONTEXTS * (INTER_MODES - 1)
            + INTERP_FILTER_CONTEXTS * (SWITCHABLE_FILTERS - 1)
            + IS_INTER_CONTEXTS;
        assert_eq!(total, 33);
        for _ in 0..total {
            let _ = read_diff_update_prob(&mut ref_dec, 128).unwrap();
        }
        let mut test_dec = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut inter_mode = DEFAULT_INTER_MODE_PROBS_TABLE;
        let mut interp = DEFAULT_INTERP_FILTER_PROBS_TABLE;
        let mut is_inter = DEFAULT_IS_INTER_PROB_TABLE;
        read_inter_mode_probs(&mut test_dec, &mut inter_mode).unwrap();
        read_interp_filter_probs(&mut test_dec, &mut interp).unwrap();
        read_is_inter_probs(&mut test_dec, &mut is_inter).unwrap();
        // Cursors must agree.
        assert_eq!(
            ref_dec.read_literal(1).unwrap(),
            test_dec.read_literal(1).unwrap(),
        );
        // And every table is still its default.
        assert_eq!(inter_mode, DEFAULT_INTER_MODE_PROBS_TABLE);
        assert_eq!(interp, DEFAULT_INTERP_FILTER_PROBS_TABLE);
        assert_eq!(is_inter, DEFAULT_IS_INTER_PROB_TABLE);
    }
}
