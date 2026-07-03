//! VP9 reconstruct process per spec v0.7 §8.6.2.
//!
//! Round 11 lands the §8.6.2 reconstruct driver — the conceptual point
//! (`reconstruct( plane, startX, startY, txSz )` in the §6.4.21 residual
//! syntax) where the rounds 7-10 pieces are tied together:
//!
//! * the round-7 coefficient-token decode supplies the per-block
//!   `Tokens[ ]` array (signed quantized coefficients in raster order);
//! * the round-8 §8.6.1 dequantizers ([`crate::dequant::get_dc_quant`] /
//!   [`crate::dequant::get_ac_quant`]) scale them into the `Dequant`
//!   buffer;
//! * the round-9 §8.7.2 inverse transform ([`crate::idct::
//!   inverse_transform_2d`]) turns the `Dequant` buffer into a spatial
//!   residual;
//! * the round-10 §8.5.1 intra prediction ([`crate::intra::
//!   predict_intra`]) has already written the prediction into
//!   `CurrFrame[ plane ]`;
//! * step 4 adds the residual to the prediction and clips with `Clip1`.
//!
//! The pieces this module exposes:
//!
//! * [`tx_type_for_intra`] — the §6.4.25 `mode2txfm_map[ y_mode ]`
//!   lookup that selects the `TxType` (`DCT_DCT` / `ADST_DCT` /
//!   `DCT_ADST` / `ADST_ADST`) for a luma intra block from its
//!   prediction mode. (The §6.4.25 plane / `TX_32X32` / 4x4-lossless
//!   overrides to `DCT_DCT` are applied by the residual driver and
//!   surfaced here through the `lossless` / plane arguments of
//!   [`reconstruct_block`].)
//! * [`reconstruct_block`] — the §8.6.2 process proper: given the
//!   already-predicted plane, the `Tokens` array, the plane-specific
//!   DC/AC quantizers, the `TxType`, and the `Lossless` flag, it builds
//!   `Dequant`, inverse-transforms it, and adds the residual back into
//!   `CurrFrame[ plane ]` with `Clip1`.
//! * [`reconstruct_intra_block`] — the end-to-end driver for one luma
//!   intra transform block: it predicts via [`crate::intra::
//!   predict_intra`], derives the `TxType` via [`tx_type_for_intra`],
//!   then runs [`reconstruct_block`].
//!
//! The §6.4.21 residual loop that supplies the real availability flags,
//! per-block `Tokens` arrays and segment/quantizer state across a whole
//! frame lands in a later round; this module is the pure §8.6.2 layer it
//! will call.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.6.2, plus §6.4.25 `mode2txfm_map`,
//! §6.2.9 `Lossless`, and §3 `Clip1`). Every formula is transcribed
//! directly from the §8.6.2 listing.

// The §6.4.21 residual driver that calls this with real per-block state
// lands in a subsequent round; until then the module is exercised
// exclusively from `#[cfg(test)]`.
#![allow(dead_code)]

use crate::idct::{inverse_transform_2d, ADST_ADST, ADST_DCT, DCT_ADST, DCT_DCT};
use crate::intra::{predict_intra, Plane, PredMode};

/// `mode2txfm_map[ MB_MODE_COUNT ]` from §10.5 — the intra prefix of the
/// table, mapping each of the 10 §7.4.5 intra prediction modes
/// (`DC_PRED` = 0 .. `TM_PRED` = 9) to its `TxType`. Transcribed
/// verbatim from the §10.5 listing; the four inter-mode entries
/// (`NEARESTMV`/`NEARMV`/`ZEROMV`/`NEWMV`, all `DCT_DCT`) are omitted
/// because [`tx_type_for_intra`] only ever indexes the intra prefix.
const MODE2TXFM_MAP_INTRA: [u8; 10] = [
    DCT_DCT,   // DC_PRED
    ADST_DCT,  // V_PRED
    DCT_ADST,  // H_PRED
    DCT_DCT,   // D45_PRED
    ADST_ADST, // D135_PRED
    ADST_DCT,  // D117_PRED
    DCT_ADST,  // D153_PRED
    DCT_ADST,  // D207_PRED
    ADST_DCT,  // D63_PRED
    ADST_ADST, // TM_PRED
];

/// `Clip1( x ) = Clip3( 0, (1 << BitDepth) - 1, x )` per §3.
#[inline]
fn clip1(x: i64, bit_depth: u32) -> i64 {
    let max = (1i64 << bit_depth) - 1;
    x.clamp(0, max)
}

/// The §6.4.25 `mode2txfm_map[ y_mode ]` lookup for a luma intra block.
///
/// Returns the `TxType` (one of [`DCT_DCT`] / [`ADST_DCT`] /
/// [`DCT_ADST`] / [`ADST_ADST`]) selected for the inverse transform of a
/// luma intra block predicted with `mode`.
///
/// Per §6.4.25 the resulting `TxType` is overridden to `DCT_DCT` when
/// the plane is chroma, when `txSz == TX_32X32`, or for a `TX_4X4` block
/// in a lossless frame; those overrides are applied by the caller (and
/// surfaced through the arguments of [`reconstruct_block`]), so this
/// helper only models the per-mode mapping itself.
pub(crate) fn tx_type_for_intra(mode: PredMode) -> u8 {
    MODE2TXFM_MAP_INTRA[mode as usize]
}

/// The §8.6.2 reconstruct process.
///
/// Inputs mirror the spec one-for-one:
///
/// * `plane_buf` — `CurrFrame[ plane ]`, already holding the §8.5.1
///   prediction in the transform-block region; the residual is added
///   back into it.
/// * `x` / `y` — top-left sample of the transform block in the plane.
/// * `tx_sz` — `txSz` (0 = TX_4X4 .. 3 = TX_32X32); `n = 2 + txSz`,
///   `n0 = 1 << n`.
/// * `tokens` — the §6.4.24 `Tokens[ ]` array for this block: the signed
///   quantized coefficients in raster order, `n0 * n0` entries.
/// * `dc_quant` / `ac_quant` — the §8.6.1 `get_dc_quant( plane )` /
///   `get_ac_quant( plane )` results for this plane and segment.
/// * `tx_type` — the §6.4.25 `TxType` for this block (ignored when
///   `lossless` is set).
/// * `lossless` — the §6.2.9 `Lossless` flag (selects the WHT path).
/// * `bit_depth` — `BitDepth` (8, 10 or 12) for the `Clip1` of step 4.
///
/// On return the transform-block region of `plane_buf` holds the
/// reconstructed samples.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_block(
    plane_buf: &mut Plane,
    x: usize,
    y: usize,
    tx_sz: u32,
    tokens: &[i64],
    dc_quant: i32,
    ac_quant: i32,
    tx_type: u8,
    lossless: bool,
    bit_depth: u32,
) {
    // §8.6.2: dqDenom = 2 if txSz == TX_32X32, else 1.
    let dq_denom: i64 = if tx_sz == 3 { 2 } else { 1 };
    // n = 2 + txSz, n0 = 1 << n.
    let n = 2 + tx_sz;
    let n0 = 1usize << n;

    debug_assert_eq!(
        tokens.len(),
        n0 * n0,
        "Tokens length must be n0*n0 for the transform size"
    );

    // Step 1: Dequant[i][j] = (Tokens[i*n0+j] * ac_quant) / dqDenom.
    // Integer division truncates toward zero (§4.1); Rust's `/` on i64
    // matches.
    let mut dequant = vec![0i64; n0 * n0];
    let ac = ac_quant as i64;
    for (slot, &tok) in dequant.iter_mut().zip(tokens.iter()) {
        *slot = (tok * ac) / dq_denom;
    }
    // Step 2: Dequant[0][0] = (Tokens[0] * dc_quant) / dqDenom.
    dequant[0] = (tokens[0] * dc_quant as i64) / dq_denom;

    // Step 3: 2D inverse transform writes the residual back into Dequant.
    inverse_transform_2d(&mut dequant, n, tx_type, lossless);

    // Step 4: CurrFrame[plane][y+i][x+j] =
    //   Clip1( CurrFrame[plane][y+i][x+j] + Dequant[i][j] ).
    //
    // Clipped to the allocated working extent for the same reason as
    // the §8.5.1 store (see `predict_intra`): a frame-edge block the
    // §6.4.3 `hasRows` / `hasCols` rules admit may extend past the
    // MiCols*8 x MiRows*8 planes into (unobservable) superblock
    // padding a spec CurrFrame would carry.
    for i in 0..n0 {
        if y + i >= plane_buf.height() {
            break;
        }
        for j in 0..n0 {
            if x + j >= plane_buf.width() {
                break;
            }
            let pred = plane_buf.get(x + j, y + i) as i64;
            let recon = clip1(pred + dequant[i * n0 + j], bit_depth);
            plane_buf.set(x + j, y + i, recon as i32);
        }
    }
}

/// End-to-end reconstruct of one luma intra transform block.
///
/// Ties §8.5.1 intra prediction (round 10) to the §8.6.2 reconstruct
/// process (this round): it predicts the block with [`predict_intra`],
/// derives the `TxType` via [`tx_type_for_intra`] (forced to `DCT_DCT`
/// when `lossless` is set or `txSz == TX_32X32`, per §6.4.25), then runs
/// [`reconstruct_block`].
///
/// Arguments fold the [`predict_intra`] inputs (`have_left` /
/// `have_above` / `not_on_right` / `max_x` / `max_y`) together with the
/// §8.6.2 inputs (`tokens` / `dc_quant` / `ac_quant` / `lossless`). This
/// is the single-block shape the (deferred) §6.4.21 residual loop will
/// drive once it threads the real availability and quantizer state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_intra_block(
    plane_buf: &mut Plane,
    x: usize,
    y: usize,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    tx_sz: u32,
    mode: PredMode,
    max_x: usize,
    max_y: usize,
    tokens: &[i64],
    dc_quant: i32,
    ac_quant: i32,
    lossless: bool,
    bit_depth: u32,
) {
    predict_intra(
        plane_buf,
        x,
        y,
        have_left,
        have_above,
        not_on_right,
        tx_sz,
        mode,
        max_x,
        max_y,
        bit_depth,
    );

    // §6.4.25: a TX_32X32 block (or any lossless block) forces DCT_DCT.
    // Lossless also routes inverse_transform_2d down the WHT path, where
    // tx_type is ignored, but we normalise it for clarity.
    let tx_type = if lossless || tx_sz == 3 {
        DCT_DCT
    } else {
        tx_type_for_intra(mode)
    };

    reconstruct_block(
        plane_buf, x, y, tx_sz, tokens, dc_quant, ac_quant, tx_type, lossless, bit_depth,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const BD: u32 = 8;

    /// `mode2txfm_map` intra prefix matches the §10.5 listing exactly.
    #[test]
    fn tx_type_for_intra_matches_mode2txfm_map() {
        assert_eq!(tx_type_for_intra(PredMode::DcPred), DCT_DCT);
        assert_eq!(tx_type_for_intra(PredMode::VPred), ADST_DCT);
        assert_eq!(tx_type_for_intra(PredMode::HPred), DCT_ADST);
        assert_eq!(tx_type_for_intra(PredMode::D45Pred), DCT_DCT);
        assert_eq!(tx_type_for_intra(PredMode::D135Pred), ADST_ADST);
        assert_eq!(tx_type_for_intra(PredMode::D117Pred), ADST_DCT);
        assert_eq!(tx_type_for_intra(PredMode::D153Pred), DCT_ADST);
        assert_eq!(tx_type_for_intra(PredMode::D207Pred), DCT_ADST);
        assert_eq!(tx_type_for_intra(PredMode::D63Pred), ADST_DCT);
        assert_eq!(tx_type_for_intra(PredMode::TmPred), ADST_ADST);
    }

    /// clip1 clamps to the bit-depth range.
    #[test]
    fn clip1_clamps_to_bit_depth_range() {
        assert_eq!(clip1(-5, 8), 0);
        assert_eq!(clip1(300, 8), 255);
        assert_eq!(clip1(128, 8), 128);
        assert_eq!(clip1(2000, 10), 1023);
    }

    /// An all-zero `Tokens` block leaves the prediction untouched: the
    /// residual is zero everywhere, so step 4 adds nothing.
    #[test]
    fn all_zero_tokens_leaves_prediction_unchanged() {
        // Pre-load a 4x4 region with a known prediction at (1,1).
        let mut p = Plane::new(8, 8);
        for i in 0..4 {
            for j in 0..4 {
                p.set(1 + j, 1 + i, 50 + (i * 4 + j) as i32);
            }
        }
        let tokens = [0i64; 16];
        reconstruct_block(&mut p, 1, 1, 0, &tokens, 100, 80, DCT_DCT, false, BD);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 50 + (i * 4 + j) as i32, "i={i} j={j}");
            }
        }
    }

    /// A DC-only `Tokens` block round-trips to prediction + a flat
    /// residual: every reconstructed sample equals the (constant)
    /// prediction plus the same residual delta.
    #[test]
    fn dc_only_tokens_adds_flat_residual_to_prediction() {
        // Flat prediction of 40 over a 4x4 block at (1,1).
        let mut p = Plane::new(8, 8);
        for i in 0..4 {
            for j in 0..4 {
                p.set(1 + j, 1 + i, 40);
            }
        }
        // DC-only Tokens: Tokens[0] = 3, rest 0. Dequant[0][0] =
        // 3 * dc_quant / 1. Pick dc_quant so the residual stays in range.
        let mut tokens = [0i64; 16];
        tokens[0] = 3;
        let dc_quant: i32 = 20; // Dequant[0][0] = 60.
        let ac_quant: i32 = 16;

        // Compute the expected flat residual independently: a DC-only
        // DCT_DCT block produces a constant residual r; reconstruct it
        // via the same idct path on a private buffer.
        let mut probe = vec![0i64; 16];
        // dqDenom is 1 for tx_sz=2, so the division is a no-op here.
        probe[0] = 3 * dc_quant as i64;
        inverse_transform_2d(&mut probe, 2, DCT_DCT, false);
        let r = probe[0];
        // Flat residual: all entries equal.
        assert!(probe.iter().all(|&v| v == r));

        reconstruct_block(
            &mut p, 1, 1, 0, &tokens, dc_quant, ac_quant, DCT_DCT, false, BD,
        );
        let expected = (40i64 + r).clamp(0, 255) as i32;
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), expected, "i={i} j={j}");
            }
        }
    }

    /// Step 4 clips: a large positive residual on top of a high
    /// prediction saturates to the bit-depth maximum (255 at 8-bit).
    #[test]
    fn step4_clips_to_bit_depth_max() {
        let mut p = Plane::new(8, 8);
        for i in 0..4 {
            for j in 0..4 {
                p.set(1 + j, 1 + i, 250);
            }
        }
        let mut tokens = [0i64; 16];
        tokens[0] = 50; // big DC -> big positive residual.
        reconstruct_block(&mut p, 1, 1, 0, &tokens, 200, 16, DCT_DCT, false, BD);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 255, "i={i} j={j}");
            }
        }
    }

    /// Step 4 clips below zero: a large negative residual on a low
    /// prediction saturates to 0.
    #[test]
    fn step4_clips_to_zero() {
        let mut p = Plane::new(8, 8);
        for i in 0..4 {
            for j in 0..4 {
                p.set(1 + j, 1 + i, 5);
            }
        }
        let mut tokens = [0i64; 16];
        tokens[0] = -50; // big negative DC.
        reconstruct_block(&mut p, 1, 1, 0, &tokens, 200, 16, DCT_DCT, false, BD);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 0, "i={i} j={j}");
            }
        }
    }

    /// TX_32X32 uses dqDenom = 2 inside `reconstruct_block`: the DC token
    /// is dequantized as `(Tokens[0] * dc_quant) / 2` before the inverse
    /// transform. Drive the real 32x32 path on a zero prediction and
    /// compare against an independent probe that applies dqDenom = 2.
    #[test]
    fn tx32x32_dequant_uses_dqdenom_2() {
        let dc_quant: i32 = 5;
        let mut tokens = vec![0i64; 32 * 32];
        tokens[0] = 7; // 7 * 5 = 35; /2 -> 17 (truncated toward zero).

        // Real driver: 32x32 (tx_sz=3) over a zero plane region at (0,0).
        let mut p = Plane::new(48, 48);
        reconstruct_block(&mut p, 0, 0, 3, &tokens, dc_quant, 4, DCT_DCT, false, BD);

        // Independent probe: dqDenom = 2 for TX_32X32.
        let mut probe = vec![0i64; 32 * 32];
        probe[0] = (7 * dc_quant as i64) / 2;
        assert_eq!(probe[0], 17);
        inverse_transform_2d(&mut probe, 5, DCT_DCT, false);

        for i in 0..32 {
            for j in 0..32 {
                let expected = probe[i * 32 + j].clamp(0, 255) as i32;
                assert_eq!(p.get(j, i), expected, "i={i} j={j}");
            }
        }
    }

    /// `reconstruct_intra_block` ties predict_intra + reconstruct: with
    /// DC_PRED both-neighbours and zero Tokens, the block equals the
    /// pure DC prediction.
    #[test]
    fn intra_block_dc_pred_zero_tokens_equals_dc_prediction() {
        let mut p = Plane::new(8, 8);
        // aboveRow = [4,4,4,4], leftCol = [8,8,8,8] -> DC avg = 6.
        for j in 0..4 {
            p.set(1 + j, 0, 4);
        }
        for i in 0..4 {
            p.set(0, 1 + i, 8);
        }
        let tokens = [0i64; 16];
        reconstruct_intra_block(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::DcPred,
            7,
            7,
            &tokens,
            100,
            80,
            false,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 6, "i={i} j={j}");
            }
        }
    }

    /// `reconstruct_intra_block` with a known DC residual reconstructs
    /// the predicted DC plus the flat residual.
    #[test]
    fn intra_block_dc_pred_with_dc_residual_reconstructs_expected() {
        let mut p = Plane::new(8, 8);
        // DC prediction: aboveRow = leftCol = 100 -> avg = 100.
        for j in 0..4 {
            p.set(1 + j, 0, 100);
        }
        for i in 0..4 {
            p.set(0, 1 + i, 100);
        }
        let mut tokens = [0i64; 16];
        tokens[0] = 2;
        let dc_quant: i32 = 16;
        let ac_quant: i32 = 16;

        // Independent residual probe (DC_PRED -> mode2txfm_map -> DCT_DCT).
        let mut probe = vec![0i64; 16];
        // dqDenom is 1 for tx_sz=2, so the division is a no-op here.
        probe[0] = 2 * dc_quant as i64;
        inverse_transform_2d(&mut probe, 2, DCT_DCT, false);
        let r = probe[0];

        reconstruct_intra_block(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::DcPred,
            7,
            7,
            &tokens,
            dc_quant,
            ac_quant,
            false,
            BD,
        );
        let expected = (100i64 + r).clamp(0, 255) as i32;
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), expected, "i={i} j={j}");
            }
        }
    }

    /// Lossless TX_4X4: the WHT path runs and `tx_type` is ignored. A
    /// DC-only Tokens block (dc_quant=ac_quant=1 in lossless frames)
    /// spreads to a flat residual added onto a flat prediction.
    #[test]
    fn lossless_intra_block_uses_wht_path() {
        // Set the above row + left column to 30 so TM_PRED reads a known
        // neighbourhood; predict_intra overwrites the block region.
        let mut p = Plane::new(8, 8);
        for j in 0..4 {
            p.set(1 + j, 0, 30);
        }
        for i in 0..4 {
            p.set(0, 1 + i, 30);
        }
        let mut tokens = [0i64; 16];
        tokens[0] = 64; // DC coefficient.

        // Independent WHT residual probe (lossless ignores tx_type).
        let mut probe = vec![0i64; 16];
        probe[0] = 64; // dc_quant = 1 in a lossless frame.
        inverse_transform_2d(&mut probe, 2, DCT_DCT, true);
        // Lossless DC-only must be flat.
        let r = probe[0];
        assert!(probe.iter().all(|&v| v == r));

        reconstruct_intra_block(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::TmPred, // any mode; lossless forces DCT_DCT + WHT.
            7,
            7,
            &tokens,
            1,
            1,
            true,
            BD,
        );
        // TM_PRED with flat neighbours (30) and aboveRow[-1]=0 (corner
        // unset) is not flat, so verify against an independent predict.
        let mut q = Plane::new(8, 8);
        for j in 0..4 {
            q.set(1 + j, 0, 30);
        }
        for i in 0..4 {
            q.set(0, 1 + i, 30);
        }
        predict_intra(
            &mut q,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::TmPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                let pred = q.get(1 + j, 1 + i) as i64;
                let expected = (pred + r).clamp(0, 255) as i32;
                assert_eq!(p.get(1 + j, 1 + i), expected, "i={i} j={j}");
            }
        }
    }
}
