//! VP9 §6.4.18 / §6.4.19 / §6.4.20 motion-vector **writers** — the inverse
//! of the `assign_mv( )` / `read_mv( )` / `read_mv_component( )` decode
//! primitives in [`crate::mv`].
//!
//! The decoder, when `y_mode == NEWMV`, reads a §6.4.19 `read_mv( ref )`
//! for each in-use reference list: a §6.4.20 `mv_joint` selector and then,
//! per the joint value, zero / one / both signed magnitude components via
//! `read_mv_component( comp )`, the result added onto the §6.5.3
//! `BestMv[ ref ]` predictor. For the other three inter modes (`NEARESTMV`
//! / `NEARMV` / `ZEROMV`) no motion-vector bits are coded at all — the MV
//! is the predictor (or zero) directly.
//!
//! These writers re-walk that exact decision tree but *encode* the
//! difference the caller supplies: given the final `Mv[ ref ]` and the
//! `BestMv[ ref ]` predictor, the writer computes `diffMv = Mv - BestMv`,
//! derives the §6.4.20 `mv_joint` from which components are non-zero, and
//! writes each component's sign / class / magnitude bits the §6.4.20
//! listing reads. Because the §6.5.13 `use_mv_hp( )` predicate is a pure
//! function of `BestMv` (shared with [`crate::mv::use_mv_hp`]), and the
//! magnitude decomposition is the exact inverse of the §6.4.20 `mag`
//! reconstruction, a written MV stream decodes back bit-for-bit.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.18 / §6.4.19 / §6.4.20 / §6.5.13 /
//! §9.3.1 / §9.3.3; the bit order mirrors [`crate::mv`] exactly.

// Driven by the inter-block writer (a later step in this subsystem);
// exercised now by the round-trip tests below.
#![allow(dead_code)]

use crate::adaptive_filter_strength::{NEARESTMV, NEARMV, NEWMV};
use crate::bool_encoder::BoolEncoder;
use crate::compressed::MvProbs;
use crate::mode_writer::tree_encode;
use crate::mv::{
    use_mv_hp, MvPredictors, MV_CLASS_0, MV_CLASS_TREE, MV_FR_TREE, MV_JOINT_HNZVNZ,
    MV_JOINT_HNZVZ, MV_JOINT_HZVNZ, MV_JOINT_TREE, MV_JOINT_ZERO,
};
use crate::Error;

/// `CLASS0_SIZE == 1 << 1` per §3 — the class-0 integer-part shift, mirror
/// of the constant `read_mv_component` decomposes against.
const CLASS0_SIZE_SHIFT: u32 = 1;

/// §6.4.20 `read_mv_component( comp )` inverse — encode one signed
/// motion-vector difference component (eighth-pel units) so the decoder
/// recovers exactly `value`.
///
/// Decomposes `value` into the §6.4.20 fields (`mv_sign`, `mv_class`, and
/// either the class-0 `bit`/`fr`/`hp` triple or the higher-class
/// `mv_bit` offset loop + `fr`/`hp`), writing each with the same
/// probability the matching decode read consumes. `use_hp` is the final
/// `UseHp` (`allow_high_precision_mv && use_mv_hp( BestMv )`); when it is
/// `false` the `hp` bit is absent and the decoder fixes it to `1`, so the
/// caller's `value` must have been formed with the corresponding hp == 1
/// low bit (the [`write_mv`] driver guarantees this by reconstructing the
/// decoder-visible difference before splitting it into components).
pub(crate) fn write_mv_component(
    enc: &mut BoolEncoder,
    probs: &MvProbs,
    comp: usize,
    value: i32,
    use_hp: bool,
) -> Result<(), Error> {
    // value == 0 is never a coded component: the §6.4.20 mag is always
    // >= 1, so a zero component means the joint excluded it.
    if value == 0 {
        return Err(Error::Unsupported);
    }
    let mv_sign = u32::from(value < 0);
    enc.write_bool(mv_sign, u32::from(probs.sign_prob[comp]));

    let mag = value.unsigned_abs();

    // Recover mv_class from the magnitude. Class 0 covers mag 1..=16
    // (mag = ((bit<<3)|(fr<<1)|hp) + 1, with bit in 0..=1, fr in 0..=3,
    // hp in 0..=1, so the bracket is 0..=15 and mag 1..=16). Higher
    // classes: mag = (CLASS0_SIZE << (class+2)) + ((d<<3)|(fr<<1)|hp) + 1,
    // where d has `class` bits, so the class-`k` band is
    // [base_k + 1, 2·base_k] with base_k = CLASS0_SIZE << (k+2).
    let mv_class = mv_class_of(mag);
    let class_probs = &probs.class_probs[comp];
    tree_encode(enc, &MV_CLASS_TREE, mv_class as i32, |node| {
        class_probs[node]
    })?;

    if mv_class == MV_CLASS_0 as u32 {
        let bracket = mag - 1; // 0..=15
        let mv_class0_bit = (bracket >> 3) & 1;
        let mv_class0_fr = (bracket >> 1) & 3;
        let mv_class0_hp = bracket & 1;
        enc.write_bool(mv_class0_bit, u32::from(probs.class0_bit_prob[comp]));
        let fr_probs = &probs.class0_fr_probs[comp][mv_class0_bit as usize];
        tree_encode(enc, &MV_FR_TREE, mv_class0_fr as i32, |node| fr_probs[node])?;
        write_hp(enc, probs.class0_hp_prob[comp], mv_class0_hp, use_hp);
    } else {
        let base = 1u32 << (CLASS0_SIZE_SHIFT + mv_class + 2);
        let bracket = mag - base - 1; // (d<<3)|(fr<<1)|hp
        let d = bracket >> 3;
        let mv_fr = (bracket >> 1) & 3;
        let mv_hp = bracket & 1;
        // d carries `mv_class` low bits, written LSB-first against
        // bits_prob[comp][i] exactly as the decode loop reads them.
        for i in 0..mv_class as usize {
            let mv_bit = (d >> i) & 1;
            enc.write_bool(mv_bit, u32::from(probs.bits_prob[comp][i]));
        }
        let fr_probs = &probs.fr_probs[comp];
        tree_encode(enc, &MV_FR_TREE, mv_fr as i32, |node| fr_probs[node])?;
        write_hp(enc, probs.hp_prob[comp], mv_hp, use_hp);
    }
    Ok(())
}

/// The §6.4.20 `mv_class` of a magnitude `mag >= 1`. Class 0 is mag
/// 1..=16. For class `k > 0` the decode forms
/// `mag = base_k + ((d << 3) | (fr << 1) | hp) + 1` with
/// `base_k = CLASS0_SIZE << (k + 2) = 1 << (k + 3)` and `d` carrying `k`
/// bits, so the `(d<<3)|…+1` addend ranges `1 ..= 8·2^k = base_k`. The
/// class-`k` band is therefore `[base_k + 1, 2·base_k]`.
fn mv_class_of(mag: u32) -> u32 {
    if mag <= 16 {
        return MV_CLASS_0 as u32;
    }
    for k in 1u32..=10 {
        let base = 1u32 << (CLASS0_SIZE_SHIFT + k + 2);
        if mag <= 2 * base {
            return k;
        }
    }
    10
}

/// §9.3.3 `mv_class0_hp` / `mv_hp` write. When `use_hp == 1` the bit is
/// present and written against `prob`; when `use_hp == 0` the bit is
/// absent (the decoder fixes it to `1`), so nothing is written. The
/// caller has already guaranteed `bit == 1` in the no-hp case.
fn write_hp(enc: &mut BoolEncoder, prob: u8, bit: u32, use_hp: bool) {
    if use_hp {
        enc.write_bool(bit, u32::from(prob));
    }
}

/// §6.4.19 `read_mv( ref )` inverse — encode the full motion-vector
/// difference `mv - best_mv` so the decoder recovers `mv`.
///
/// Mirrors the decode: `UseHp = allow_hp && use_mv_hp( best_mv )`, the
/// §6.4.20 `mv_joint` selecting which of the two components are non-zero,
/// then a `write_mv_component( )` per non-zero component. `mv` and
/// `best_mv` are `[row, col]` in eighth-pel units.
///
/// The caller must supply a `mv` reachable from `best_mv` under the
/// active `UseHp`: when `UseHp == 0` each non-zero difference component's
/// low (hp) bit is fixed to `1` by the decoder, so a component difference
/// must be odd-bracketed accordingly. A later rate-distortion search
/// layer rounds chosen vectors onto the reachable lattice; the round-trip
/// tests below use only reachable values.
pub(crate) fn write_mv(
    enc: &mut BoolEncoder,
    probs: &MvProbs,
    mv: [i32; 2],
    best_mv: [i32; 2],
    allow_hp: bool,
) -> Result<(), Error> {
    let use_hp = allow_hp && use_mv_hp(best_mv);
    let diff = [mv[0] - best_mv[0], mv[1] - best_mv[1]];

    let joint = mv_joint_of(diff);
    tree_encode(enc, &MV_JOINT_TREE, joint, |node| probs.joint_probs[node])?;

    if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
        write_mv_component(enc, probs, 0, diff[0], use_hp)?;
    }
    if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
        write_mv_component(enc, probs, 1, diff[1], use_hp)?;
    }
    Ok(())
}

/// The §6.4.20 `mv_joint` selector for a difference `[row, col]`.
fn mv_joint_of(diff: [i32; 2]) -> i32 {
    match (diff[0] != 0, diff[1] != 0) {
        (false, false) => MV_JOINT_ZERO,
        (false, true) => MV_JOINT_HNZVZ,
        (true, false) => MV_JOINT_HZVNZ,
        (true, true) => MV_JOINT_HNZVNZ,
    }
}

/// §6.4.18 `assign_mv( isCompound )` inverse — encode the per-list motion
/// vectors the caller chose, given `y_mode` and the predictors.
///
/// For `NEWMV` each in-use list writes a §6.4.19 `read_mv( i )`
/// difference onto `preds[ i ].best`; the other three modes write no MV
/// bits (the decoder picks `NearestMv` / `NearMv` / `ZeroMv`), so the
/// caller's `mv` must equal the corresponding predictor for those modes —
/// the writer verifies this (a mismatch is a caller bug, returned as
/// `Error::Unsupported`).
pub(crate) fn write_assign_mv(
    enc: &mut BoolEncoder,
    probs: &MvProbs,
    y_mode: u8,
    is_compound: bool,
    preds: &[MvPredictors; 2],
    mv: &[[i32; 2]; 2],
    allow_hp: bool,
) -> Result<(), Error> {
    let count = if is_compound { 2 } else { 1 };
    for i in 0..count {
        match y_mode {
            NEWMV => write_mv(enc, probs, mv[i], preds[i].best, allow_hp)?,
            NEARESTMV => {
                if mv[i] != preds[i].nearest {
                    return Err(Error::Unsupported);
                }
            }
            NEARMV => {
                if mv[i] != preds[i].near {
                    return Err(Error::Unsupported);
                }
            }
            // ZEROMV.
            _ => {
                if mv[i] != [0, 0] {
                    return Err(Error::Unsupported);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::mv::{assign_mv, read_mv, read_mv_component};

    fn enc_to_dec(enc: BoolEncoder) -> BoolCoder<'static> {
        let buf = enc.finish();
        // Leak the buffer so the coder can borrow it for 'static in tests.
        let leaked: &'static [u8] = Box::leak(buf.into_boxed_slice());
        BoolCoder::init_bool(leaked, leaked.len()).unwrap()
    }

    #[test]
    fn write_mv_component_roundtrips_class0_all_brackets() {
        let probs = MvProbs::defaults();
        for comp in 0..2usize {
            for use_hp in [false, true] {
                // Class-0 magnitudes 1..=16, both signs. With use_hp ==
                // false, only even brackets (hp == 1) are reachable; pick
                // magnitudes whose low bit is 1 (mag even => bracket odd
                // => hp == 1).
                let mags: &[i32] = if use_hp {
                    &[1, 2, 8, 15, 16, -1, -9, -16]
                } else {
                    // mag-1 must be odd => mag even.
                    &[2, 4, 16, -2, -10, -16]
                };
                for &v in mags {
                    let mut enc = BoolEncoder::new();
                    write_mv_component(&mut enc, &probs, comp, v, use_hp).unwrap();
                    let mut dec = enc_to_dec(enc);
                    let got =
                        read_mv_component(&mut dec, &probs, comp, use_hp, &mut Default::default())
                            .unwrap();
                    assert_eq!(got, v, "comp {comp} hp {use_hp} v {v}");
                }
            }
        }
    }

    #[test]
    fn write_mv_component_roundtrips_higher_classes() {
        let probs = MvProbs::defaults();
        for comp in 0..2usize {
            // Magnitudes spanning several classes (use_hp = true so any
            // bracket is reachable).
            for &v in &[17, 33, 64, 100, 255, 600, -17, -64, -255, -600] {
                let mut enc = BoolEncoder::new();
                write_mv_component(&mut enc, &probs, comp, v, true).unwrap();
                let mut dec = enc_to_dec(enc);
                let got = read_mv_component(&mut dec, &probs, comp, true, &mut Default::default())
                    .unwrap();
                assert_eq!(got, v, "comp {comp} v {v}");
            }
        }
    }

    #[test]
    fn write_mv_roundtrips_all_joints() {
        let probs = MvProbs::defaults();
        let best = [4, -6];
        // Differences exercising each joint (hp on so any value works).
        let diffs = [[0, 0], [0, 5], [-3, 0], [7, -9]];
        for d in diffs {
            let mv = [best[0] + d[0], best[1] + d[1]];
            let mut enc = BoolEncoder::new();
            write_mv(&mut enc, &probs, mv, best, true).unwrap();
            let mut dec = enc_to_dec(enc);
            let got = read_mv(&mut dec, &probs, best, true, &mut Default::default()).unwrap();
            assert_eq!(got, mv, "diff {d:?}");
        }
    }

    #[test]
    fn write_mv_low_precision_predictor_suppresses_hp() {
        // A large best_mv makes use_mv_hp false even with allow_hp; the
        // hp bit is then absent and the decoded low bit is forced to 1.
        let probs = MvProbs::defaults();
        let best = [200, 0]; // abs>>3 == 25 >= 8 -> use_mv_hp false.
        assert!(!use_mv_hp(best));
        // A reachable diff: row component bracket must have hp == 1.
        // diff[0] = 2 => mag 2 => bracket 1 => hp == 1: reachable.
        let mv = [best[0] + 2, best[1]];
        let mut enc = BoolEncoder::new();
        write_mv(&mut enc, &probs, mv, best, true).unwrap();
        let mut dec = enc_to_dec(enc);
        let got = read_mv(&mut dec, &probs, best, true, &mut Default::default()).unwrap();
        assert_eq!(got, mv);
    }

    #[test]
    fn write_assign_mv_newmv_roundtrips() {
        let probs = MvProbs::defaults();
        let preds = [
            MvPredictors {
                nearest: [1, 2],
                near: [3, 4],
                best: [5, 6],
            },
            MvPredictors {
                nearest: [-1, -2],
                near: [-3, -4],
                best: [-5, -6],
            },
        ];
        let mv = [[5 + 8, 6 - 4], [-5 + 2, -6 + 10]];
        let mut enc = BoolEncoder::new();
        write_assign_mv(&mut enc, &probs, NEWMV, true, &preds, &mv, true).unwrap();
        let mut dec = enc_to_dec(enc);
        let got = assign_mv(
            &mut dec,
            &probs,
            NEWMV,
            true,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(got, mv);
    }

    #[test]
    fn write_assign_mv_nonnew_modes_write_nothing() {
        let probs = MvProbs::defaults();
        let preds = [
            MvPredictors {
                nearest: [1, 2],
                near: [3, 4],
                best: [5, 6],
            },
            MvPredictors::default(),
        ];
        // NEARESTMV single: mv must equal nearest, nothing coded.
        let mv = [[1, 2], [0, 0]];
        let mut enc = BoolEncoder::new();
        write_assign_mv(&mut enc, &probs, NEARESTMV, false, &preds, &mv, true).unwrap();
        let mut dec = enc_to_dec(enc);
        let got = assign_mv(
            &mut dec,
            &probs,
            NEARESTMV,
            false,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(got, mv);
    }

    #[test]
    fn write_assign_mv_mode_predictor_mismatch_rejected() {
        let probs = MvProbs::defaults();
        let preds = [
            MvPredictors {
                nearest: [1, 2],
                near: [3, 4],
                best: [5, 6],
            },
            MvPredictors::default(),
        ];
        // NEARMV but mv != near -> rejected.
        let mv = [[9, 9], [0, 0]];
        let mut enc = BoolEncoder::new();
        let r = write_assign_mv(&mut enc, &probs, NEARMV, false, &preds, &mv, true);
        assert_eq!(r.unwrap_err(), Error::Unsupported);
    }

    #[test]
    fn mv_class_of_band_boundaries() {
        assert_eq!(mv_class_of(1), 0);
        assert_eq!(mv_class_of(16), 0);
        assert_eq!(mv_class_of(17), 1);
        assert_eq!(mv_class_of(32), 1);
        assert_eq!(mv_class_of(33), 2);
    }
}
