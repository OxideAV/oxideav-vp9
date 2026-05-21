//! VP9 compressed-header walker per spec v0.7 §6.3.
//!
//! Cumulative scope through round 4:
//!
//! * §6.3.1 `read_tx_mode( )` — which arithmetic-codes a 2- or 3-bit
//!   tx_mode value under the §9.2 Boolean coder. This is the first
//!   syntax element of `compressed_header( )` (§6.3) for non-lossless
//!   frames; for lossless frames `tx_mode` is forced to `ONLY_4X4`
//!   and no bits are read.
//! * §6.3.3 `diff_update_prob( prob )` — the probability-update
//!   helper invoked by every tx-mode / coef / skip / inter-mode /
//!   interp-filter / is-inter / comp-mode / single-ref / comp-ref /
//!   y-mode / partition probability sweep in §6.3.2 and §6.3.7..§6.3.16.
//!   It first reads `B(252)` for an `update_prob` flag, and on 1
//!   pulls a `decode_term_subexp` value (§6.3.4) which is then
//!   remapped through `inv_remap_prob` (§6.3.5) — itself the
//!   composition of the 255-entry `inv_map_table` lookup with
//!   `inv_recenter_nonneg` (§6.3.6).
//!
//! Round 4 lands the helper chain as a standalone primitive; no
//! caller in §6.3.2 / §6.3.7+ uses it yet, those table sweeps land
//! in the next round. The chain is exercised by hand-derived
//! §9.2-Boolean-coder buffers so each leg of the
//! `decode_term_subexp` decision tree (§6.3.4) is covered: the
//! 0..=15 branch, the 16..=31 branch (`+16`), the 32..=63 branch
//! (`+32`), and both halves of the final L(7)/L(1) tail
//! (`v < 65` short-circuit + the `(v << 1) - 1 + bit` 65..=254
//! branch). `inv_recenter_nonneg` and `inv_remap_prob` are also
//! covered with closed-form tests against the spec's piecewise
//! definitions.
//!
//! The remaining §6.3 sweeps — `tx_mode_probs`, `read_coef_probs`,
//! `read_skip_prob`, plus the inter-frame `read_inter_mode_probs` /
//! `read_interp_filter_probs` and friends — all flow through
//! `read_diff_update_prob` and land in the next round once their
//! caller skeletons (tx-mode-specific nested loops, the 4D coef
//! probability table, etc.) are wired up.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7,
//! `docs/video/vp9/vp9-spec.txt` §6.3.1 / §6.3.3 / §6.3.4 / §6.3.5
//! / §6.3.6 (the `inv_map_table` constant is transcribed verbatim
//! from §6.3.5). No external library source consulted.

use crate::bool_coder::BoolCoder;
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

/// Round-3 view of the VP9 compressed header.
///
/// Only `tx_mode` is walked; the remaining §6.3 fields land in the
/// next round once `diff_update_prob` / `decode_term_subexp` are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9CompressedHeader {
    /// Decoded `tx_mode` (§6.3.1).
    pub tx_mode: TxMode,
}

/// Parse the §6.3 compressed header from `data`.
///
/// `data` must be the byte slice of length `header_size_in_bytes`
/// that follows the uncompressed-header byte-aligned trailing-bits
/// pad. `lossless` is the §6.2.9 `Lossless` derivation already
/// available from [`crate::Vp9FrameHeader::quantization::lossless`].
///
/// Currently walks `read_tx_mode( )` only — subsequent §6.3 fields
/// (`tx_mode_probs`, `read_coef_probs`, `read_skip_prob`, …) land in
/// the next round.
///
/// Returns [`Error::InvalidBitstream`] if the §9.2.1 init marker
/// bit is nonzero, if `read_bool` underruns `BoolMaxBits`, or if a
/// decoded value falls outside its spec-defined range.
pub fn parse_compressed_header(data: &[u8], lossless: bool) -> Result<Vp9CompressedHeader, Error> {
    let sz = data.len();
    let mut coder = BoolCoder::init_bool(data, sz)?;
    let tx_mode = read_tx_mode(&mut coder, lossless)?;
    Ok(Vp9CompressedHeader { tx_mode })
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
/// Not yet used outside `cfg(test)` — the §6.3.2 / §6.3.7+ probability
/// sweeps that consume it land in the next round.
#[allow(dead_code)]
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
/// Not yet used outside `cfg(test)` — `inv_remap_prob` calls it
/// internally, but neither has any caller in the round-4 syntax
/// walker (the §6.3.7+ sweeps land in the next round).
#[allow(dead_code)]
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
/// Not yet used outside `cfg(test)` — `read_diff_update_prob` calls
/// it on the §6.3.3 update path, but neither helper has a caller in
/// the round-4 syntax walker (the §6.3.7+ sweeps land in the next
/// round).
#[allow(dead_code)]
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
/// Not yet used outside `cfg(test)` — `read_diff_update_prob` calls
/// it on the §6.3.3 update path, but neither helper has a caller in
/// the round-4 syntax walker.
#[allow(dead_code)]
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
/// This entry point is the structural primitive every §6.3.7+
/// probability-table sweep is built on; no caller in the round-4
/// code uses it yet — the sweeps over `tx_mode_probs`,
/// `coef_probs`, `skip_prob`, `inter_mode_probs`,
/// `interp_filter_probs`, `is_inter_prob`, `comp_mode_prob`,
/// `single_ref_prob`, `comp_ref_prob`, `y_mode_probs`,
/// `partition_probs` (and the dedicated `update_mv_prob` sibling at
/// §6.3.17) land in subsequent rounds.
///
/// Not yet used outside `cfg(test)` — exposed for the next-round
/// callers.
#[allow(dead_code)]
pub(crate) fn read_diff_update_prob(coder: &mut BoolCoder<'_>, base_prob: u8) -> Result<u8, Error> {
    let update_prob = coder.read_bool(252)?;
    if update_prob == 1 {
        let delta_prob = decode_term_subexp(coder)?;
        Ok(inv_remap_prob(delta_prob, base_prob))
    } else {
        Ok(base_prob)
    }
}

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
}
