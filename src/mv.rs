//! VP9 motion-vector representation + decode (§6.4.19 / §10.5).
//!
//! Motion vectors in VP9 are 1/8-pel for luma. The §6.4.19 decode tree
//! produces a (row, col) displacement in 1/8-pel units relative to the
//! current block's top-left in the reference frame. Sub-pel position
//! is extracted as the low 3 bits of each component during
//! interpolation (see `inter::mc_block`).
//!
//! The decoder models a component as:
//!
//! 1. `mv_joint` — which components are non-zero. 2 bits tree-coded.
//! 2. Per non-zero component:
//!    * `sign` — 1 bit.
//!    * `class` — 0..10 tree-coded. Class 0 covers the very-near
//!      range [1, 2] (post-sign), classes 1..=10 cover [3, 4094]
//!      with a variable bit payload.
//!    * `class0`-bit / `class0`-fr / `class0`-hp *or* `bits` / `fr` /
//!      `hp` depending on which class was read.
//!
//! The reference-probability tables (§10.5) are baked into this module
//! so callers only need a `BoolDecoder` cursor. High-precision MV
//! (`allow_high_precision_mv`) toggles whether the `hp` bit is read;
//! low-precision streams skip it and the component is 1/4-pel only.

use oxideav_core::Result;

use crate::bool_decoder::BoolDecoder;

/// Motion vector in 1/8-pel luma units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    /// Row component (vertical); positive means further down.
    pub row: i16,
    /// Column component (horizontal); positive means further right.
    pub col: i16,
}

impl Mv {
    pub const ZERO: Mv = Mv { row: 0, col: 0 };

    pub fn new(row: i16, col: i16) -> Self {
        Self { row, col }
    }
}

/// `mv_joint` — which MV components are non-zero (§6.4.19 Tree 10-34).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MvJoint {
    Zero = 0,
    Hnzvz = 1, // horizontal non-zero, vertical zero
    Hzvnz = 2, // horizontal zero, vertical non-zero
    Hnzvnz = 3,
}

impl MvJoint {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Zero,
            1 => Self::Hnzvz,
            2 => Self::Hzvnz,
            _ => Self::Hnzvnz,
        }
    }

    pub fn has_col(&self) -> bool {
        matches!(self, Self::Hnzvz | Self::Hnzvnz)
    }
    pub fn has_row(&self) -> bool {
        matches!(self, Self::Hzvnz | Self::Hnzvnz)
    }
}

/// Default MV-joint probabilities (§10.5 `default_mv_joint_probs`).
/// Tree order: `[not_zero, hnzvz_vs_rest_given_not_zero,
/// split_hzvnz_hnzvnz]`.
pub const MV_JOINT_PROBS: [u8; 3] = [32, 64, 96];

/// Default per-component MV probabilities (§10.5 `default_mv_context`).
/// We only keep the subset we actually read:
///
/// * `sign` — 1 bit, prob.
/// * `class` — 10-symbol tree; the 10 internal probs below.
/// * `class0_bit`, `class0_fr`, `class0_hp` — in-class-0 refinements.
/// * `bits` — 1 bit per integer bit for classes 1..10.
/// * `fr` — 2 bits for the 1/4-pel fractional part for classes 1..10.
/// * `hp` — 1 bit for the 1/8-pel fractional part.
#[derive(Clone, Copy, Debug)]
pub struct MvComponentProbs {
    pub sign: u8,
    /// 10-node class tree (first split selects class 0 vs non-zero class).
    pub classes: [u8; 10],
    pub class0_bit: u8,
    /// `class0_fr`: 3 probabilities. First bit picks fr>=2; second
    /// conditional on that; third is used when first bit was 0.
    pub class0_fr: [u8; 3],
    pub class0_hp: u8,
    /// `bits` for classes 1..=10: up to 10 integer-magnitude bits.
    pub bits: [u8; 10],
    /// `fr` for classes >= 1: same layout as `class0_fr`.
    pub fr: [u8; 3],
    pub hp: u8,
}

/// libvpx default MV component context ("default_mv_context" in
/// `vp9_entropymv.c`). Same values for both horizontal and vertical.
pub const DEFAULT_MV_COMP_PROBS: MvComponentProbs = MvComponentProbs {
    sign: 128,
    classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
    class0_bit: 216,
    class0_fr: [128, 128, 64],
    class0_hp: 160,
    bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
    fr: [64, 96, 64],
    hp: 128,
};

/// Decode one `mv_joint` symbol against `MV_JOINT_PROBS`.
pub fn read_mv_joint(bd: &mut BoolDecoder<'_>, p: [u8; 3]) -> Result<MvJoint> {
    // VP9 tree-shape: first bit picks ZERO vs non-zero joint; second
    // splits HZVNZ/HNZVNZ vs HNZVZ etc.
    // Tree (libvpx `vp9_mv_joint_tree`):
    //   -MV_JOINT_ZERO, 2
    //    -MV_JOINT_HNZVZ, 4
    //     -MV_JOINT_HZVNZ, -MV_JOINT_HNZVNZ
    if bd.read(p[0])? == 0 {
        return Ok(MvJoint::Zero);
    }
    if bd.read(p[1])? == 0 {
        return Ok(MvJoint::Hnzvz);
    }
    if bd.read(p[2])? == 0 {
        Ok(MvJoint::Hzvnz)
    } else {
        Ok(MvJoint::Hnzvnz)
    }
}

/// Decode one MV component (row or col) value in 1/8-pel units.
/// `allow_high_precision_mv` gates the `hp` bit (§6.4.19 end).
pub fn read_mv_component(
    bd: &mut BoolDecoder<'_>,
    p: &MvComponentProbs,
    allow_high_precision_mv: bool,
) -> Result<i16> {
    let sign = bd.read(p.sign)? != 0;
    let class = read_mv_class(bd, &p.classes)?;
    let mag = if class == 0 {
        // Class 0: value is in [1, 2], refined with class0_fr + class0_hp.
        let d = bd.read(p.class0_bit)? as i32; // 0 or 1
        let fr = read_mv_fr(bd, &p.class0_fr)?;
        let hp = if allow_high_precision_mv {
            bd.read(p.class0_hp)? as i32
        } else {
            1
        };
        // Value in 1/8-pel: (d*2 + 1 (implicit class-0 constant) + fr + hp/8)
        // libvpx `decode_mv_component`:
        //   mag = ((d << 3) | (fr << 1) | hp) + 1;
        ((d << 3) | (fr << 1) | hp) + 1
    } else {
        // Classes 1..=10: value >= 3. `class_base[c]` = MV_CLASS0_SIZE
        // << (c + 2) = 2 << (c + 2) = 8 << c. Read `c` integer bits,
        // then fr, then hp.
        let mut d = 0i32;
        for i in 0..class {
            d |= (bd.read(p.bits[i])? as i32) << i;
        }
        let fr = read_mv_fr(bd, &p.fr)?;
        let hp = if allow_high_precision_mv {
            bd.read(p.hp)? as i32
        } else {
            1
        };
        // mag = (((1 << class) << 3) + (d << 3) + (fr << 1) + hp + 1)
        //   equivalently (1 << (class + 3)) + (d << 3) + (fr << 1) + hp + 1.
        (1i32 << (class + 3)) + (d << 3) + (fr << 1) + hp + 1
    };
    let signed = if sign { -mag } else { mag };
    Ok(signed.clamp(-32768, 32767) as i16)
}

/// Decode the `mv_class` tree (§6.4.19 Tree 10-35 / libvpx
/// `vp9_mv_class_tree`).
///
/// Layout of the tree: `{0: 2, {1: 4, {2, 3}}, {6, {{4, 5}, {8, {9, 10}}}}}`.
/// Returns 0..=10.
fn read_mv_class(bd: &mut BoolDecoder<'_>, p: &[u8; 10]) -> Result<usize> {
    // Tree (libvpx `vp9_mv_class_tree`):
    //   -CLASS0,       2
    //    -CLASS1,      4
    //     -CLASS2,     6
    //      -CLASS3,    8
    //       -CLASS4,   10
    //        -CLASS5,  12
    //         -CLASS6, 14
    //          -CLASS7, 16
    //           -CLASS8, 18
    //            -CLASS9, -CLASS10
    if bd.read(p[0])? == 0 {
        return Ok(0);
    }
    if bd.read(p[1])? == 0 {
        return Ok(1);
    }
    if bd.read(p[2])? == 0 {
        return Ok(2);
    }
    if bd.read(p[3])? == 0 {
        return Ok(3);
    }
    if bd.read(p[4])? == 0 {
        return Ok(4);
    }
    if bd.read(p[5])? == 0 {
        return Ok(5);
    }
    if bd.read(p[6])? == 0 {
        return Ok(6);
    }
    if bd.read(p[7])? == 0 {
        return Ok(7);
    }
    if bd.read(p[8])? == 0 {
        return Ok(8);
    }
    if bd.read(p[9])? == 0 {
        Ok(9)
    } else {
        Ok(10)
    }
}

/// Decode the 2-bit `fr` (fractional) subtree against 3 probabilities.
/// Returns 0..=3.
fn read_mv_fr(bd: &mut BoolDecoder<'_>, p: &[u8; 3]) -> Result<i32> {
    // Tree (libvpx `vp9_mv_fp_tree`):
    //   -0, 2
    //    -1, 4
    //     -2, -3
    let b0 = bd.read(p[0])?;
    if b0 == 0 {
        return Ok(0);
    }
    let b1 = bd.read(p[1])?;
    if b1 == 0 {
        return Ok(1);
    }
    let b2 = bd.read(p[2])?;
    Ok(if b2 == 0 { 2 } else { 3 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the joint tree with a bool decoder that returns 0 for
    /// every read — expected outcome is `Zero`.
    #[test]
    fn joint_zero_from_flat_zero_stream() {
        let buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let j = read_mv_joint(&mut bd, MV_JOINT_PROBS).unwrap();
        // With p[0]=32 and a very low bool-value, boolean() returns 0
        // (since value < split is more likely when value is tiny).
        assert_eq!(j, MvJoint::Zero);
    }

    #[test]
    fn component_decodes_small_magnitude() {
        // Flat-zero stream: expect sign=0, class=0, and a class0 body
        // that produces a small-magnitude MV. The exact value depends
        // on the probability tree but must be in [-16, 16].
        let buf = [0x00u8; 32];
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let v = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
        assert!(v.abs() < 16, "expected small magnitude, got {v}");
    }

    #[test]
    fn mv_joint_has_helpers() {
        assert!(MvJoint::Hnzvz.has_col());
        assert!(!MvJoint::Hnzvz.has_row());
        assert!(MvJoint::Hzvnz.has_row());
        assert!(!MvJoint::Hzvnz.has_col());
        assert!(MvJoint::Hnzvnz.has_row() && MvJoint::Hnzvnz.has_col());
        assert!(!MvJoint::Zero.has_row() && !MvJoint::Zero.has_col());
    }
}
