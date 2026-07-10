//! §6.4.19 / §6.4.20 motion-vector residual syntax — the `read_mv( )` /
//! `read_mv_component( )` leaf primitives plus the §6.5.13
//! `use_mv_hp( )` predicate they depend on.
//!
//! These three functions are the bottom of the inter-block motion-vector
//! decode. The §6.4.18 `assign_mv( )` driver calls `read_mv( ref )` once
//! per active reference list entry when `y_mode == NEWMV`; `read_mv( )`
//! in turn decodes the §6.4.20 `mv_joint` selector and (per the joint
//! value) zero, one, or both `read_mv_component( comp )` magnitudes, then
//! adds the resulting difference to the §6.5.3 `BestMv[ ref ]` predictor.
//!
//! Everything here is a pure function of the shared §9.2 bool coder and
//! the §6.3.16 [`MvProbs`] bundle the compressed header parsed — there is
//! no frame-wide state to thread, which makes the primitives directly
//! testable against a constructed bool-coder buffer in isolation from the
//! rest of the inter path. The motion-vector *prediction* that produces
//! `BestMv` (§6.5 `find_mv_refs( )` / `find_best_ref_mvs( )`) and the
//! §8.5.2 inter prediction process that consumes the decoded `Mv` are
//! separate steps that land on top of this layer.
//!
//! Tree shapes per §9.3.3 listing (`vp9-spec.txt` lines 6183-6211):
//!
//! ```text
//! mv_joint_tree[ 6 ] = { -MV_JOINT_ZERO, 2, -MV_JOINT_HNZVZ, 4,
//!                        -MV_JOINT_HZVNZ, -MV_JOINT_HNZVNZ }
//! mv_class_tree[ 20 ] = { -MV_CLASS_0, 2, -MV_CLASS_1, 4, 6, 8,
//!                         -MV_CLASS_2, -MV_CLASS_3, 10, 12,
//!                         -MV_CLASS_4, -MV_CLASS_5, -MV_CLASS_6, 14,
//!                         16, 18, -MV_CLASS_7, -MV_CLASS_8,
//!                         -MV_CLASS_9, -MV_CLASS_10 }
//! mv_fr_tree[ 6 ] = { -0, 2, -1, 4, -2, -3 }
//! ```
//!
//! `mv_class0_hp` / `mv_hp` use [`BINARY_TREE`] when `UseHp == 1`; when
//! `UseHp == 0` the bit is not present in the bitstream and the decoded
//! value is fixed to `1` (§9.3.3 listing, `vp9-spec.txt` lines 6214-6216).

use crate::adaptive_filter_strength::{NEARESTMV, NEARMV, NEWMV};
use crate::bool_coder::BoolCoder;
use crate::compressed::MvProbs;
use crate::mode_info::{tree_decode, BINARY_TREE};
use crate::prob_adapt::CountsNonCoef;
use crate::Error;

/// `MV_JOINT_ZERO = 0` per §7.4.13 (`vp9-spec.txt` line 4014). Neither
/// motion-vector difference component is non-zero.
pub(crate) const MV_JOINT_ZERO: i32 = 0;
/// `MV_JOINT_HNZVZ = 1` per §7.4.13 (`vp9-spec.txt` line 4015). The col
/// (component 1) difference is non-zero; the row (component 0) is zero.
pub(crate) const MV_JOINT_HNZVZ: i32 = 1;
/// `MV_JOINT_HZVNZ = 2` per §7.4.13 (`vp9-spec.txt` line 4016). The row
/// (component 0) difference is non-zero; the col (component 1) is zero.
pub(crate) const MV_JOINT_HZVNZ: i32 = 2;
/// `MV_JOINT_HNZVNZ = 3` per §7.4.13 (`vp9-spec.txt` line 4017). Both
/// difference components are non-zero.
pub(crate) const MV_JOINT_HNZVNZ: i32 = 3;

/// `MV_CLASS_0 = 0` per §7.4.14 (`vp9-spec.txt` line 4030). The smallest
/// magnitude class; decoded via the `mv_class0_*` syntax elements rather
/// than the `mv_bit` offset loop.
pub(crate) const MV_CLASS_0: i32 = 0;

/// `CLASS0_SIZE = 2` per §3 (`vp9-spec.txt` line 510). Number of
/// `mv_class0_bit` values — the integer part of a class-0 difference.
const CLASS0_SIZE_SHIFT: u32 = 1; // CLASS0_SIZE == 1 << 1.

/// `mv_joint_tree[ 6 ]` per §9.3.3 (`vp9-spec.txt` lines 6184-6188).
pub(crate) const MV_JOINT_TREE: [i32; 6] = [
    -MV_JOINT_ZERO,
    2,
    -MV_JOINT_HNZVZ,
    4,
    -MV_JOINT_HZVNZ,
    -MV_JOINT_HNZVNZ,
];

/// `mv_class_tree[ 20 ]` per §9.3.3 (`vp9-spec.txt` lines 6192-6203). The
/// leaves are the 11 `MV_CLASS_*` values (`0..=10`); a negated leaf marks
/// a terminal node in the §9.3.3 `do { } while` walk.
pub(crate) const MV_CLASS_TREE: [i32; 20] = [
    0, 2, // -MV_CLASS_0, 2
    -1, 4, // -MV_CLASS_1, 4
    6, 8, //
    -2, -3, // -MV_CLASS_2, -MV_CLASS_3
    10, 12, //
    -4, -5, // -MV_CLASS_4, -MV_CLASS_5
    -6, 14, // -MV_CLASS_6, 14
    16, 18, //
    -7, -8, // -MV_CLASS_7, -MV_CLASS_8
    -9, -10, // -MV_CLASS_9, -MV_CLASS_10
];

/// `mv_fr_tree[ 6 ]` per §9.3.3 (`vp9-spec.txt` lines 6207-6211). Shared
/// by the `mv_class0_fr` and `mv_fr` syntax elements; decodes one of the
/// four fractional-pel values `0..=3`.
pub(crate) const MV_FR_TREE: [i32; 6] = [0, 2, -1, 4, -2, -3];

/// `COMPANDED_MVREF_THRESH = 8` per §3 (`vp9-spec.txt` line 514).
/// Threshold (in full-pel units, i.e. after a `>> 3` from the eighth-pel
/// `Mv` representation) past which a motion vector is considered large
/// enough that high-precision signalling is suppressed.
const COMPANDED_MVREF_THRESH: i32 = 8;

/// §6.5.13 `use_mv_hp( deltaMv )` — the high-precision predicate.
///
/// ```text
/// use_mv_hp( deltaMv ) {
///   return ( (Abs( deltaMv[ 0 ] ) >> 3) < COMPANDED_MVREF_THRESH &&
///            (Abs( deltaMv[ 1 ] ) >> 3) < COMPANDED_MVREF_THRESH )
/// }
/// ```
///
/// `deltaMv` here is the §6.5.3 `BestMv[ ref ]` predictor in eighth-pel
/// units. Returns `true` when both components are small enough for the
/// third fractional bit (`mv_hp` / `mv_class0_hp`) to be signalled; the
/// caller still gates the final `UseHp` on `allow_high_precision_mv`.
pub(crate) fn use_mv_hp(delta_mv: [i32; 2]) -> bool {
    (delta_mv[0].abs() >> 3) < COMPANDED_MVREF_THRESH
        && (delta_mv[1].abs() >> 3) < COMPANDED_MVREF_THRESH
}

/// §6.4.20 `read_mv_component( comp )` — decode one signed motion-vector
/// difference component in eighth-pel units.
///
/// ```text
/// read_mv_component( comp ) {
///   mv_sign
///   mv_class
///   if ( mv_class == MV_CLASS_0 ) {
///     mv_class0_bit
///     mv_class0_fr
///     mv_class0_hp
///     mag = ( ( mv_class0_bit << 3 ) | ( mv_class0_fr << 1 ) |
///             mv_class0_hp ) + 1
///   } else {
///     d = 0
///     for ( i = 0; i < mv_class; i++ ) { mv_bit; d |= mv_bit << i }
///     mag = CLASS0_SIZE << ( mv_class + 2 )
///     mv_fr
///     mv_hp
///     mag += ( ( d << 3 ) | ( mv_fr << 1 ) | mv_hp ) + 1
///   }
///   return mv_sign ? -mag : mag
/// }
/// ```
///
/// `comp` selects the per-component probability rows (0 = row, 1 = col)
/// per the §9.3.2 listing (`vp9-spec.txt` lines 6587-6637). `use_hp` is
/// the final `UseHp` flag the caller derived from
/// `allow_high_precision_mv && use_mv_hp( BestMv[ ref ] )`; when it is
/// `false` the `mv_class0_hp` / `mv_hp` bit is absent and reads as `1`.
pub(crate) fn read_mv_component(
    coder: &mut BoolCoder<'_>,
    probs: &MvProbs,
    comp: usize,
    use_hp: bool,
    counts: &mut CountsNonCoef,
) -> Result<i32, Error> {
    let mc = &mut counts.mv_comp[comp];

    // mv_sign: mv_sign_prob[ comp ].
    let mv_sign = coder.read_bool(u32::from(probs.sign_prob[comp]))?;
    mc.sign[mv_sign as usize] += 1;

    // mv_class: mv_class_probs[ comp ] over mv_class_tree.
    let class_probs = &probs.class_probs[comp];
    let mv_class = tree_decode(coder, &MV_CLASS_TREE, |node| class_probs[node])?;
    mc.class[mv_class as usize] += 1;

    let mag = if mv_class == MV_CLASS_0 {
        // mv_class0_bit: mv_class0_bit_prob[ comp ].
        let mv_class0_bit = coder.read_bool(u32::from(probs.class0_bit_prob[comp]))?;
        mc.class0_bit[mv_class0_bit as usize] += 1;
        // mv_class0_fr: mv_class0_fr_probs[ comp ][ mv_class0_bit ][ node ].
        let fr_probs = &probs.class0_fr_probs[comp][mv_class0_bit as usize];
        let mv_class0_fr = tree_decode(coder, &MV_FR_TREE, |node| fr_probs[node])?;
        mc.class0_fr[mv_class0_bit as usize][mv_class0_fr as usize] += 1;
        // mv_class0_hp: mv_class0_hp_prob[ comp ]; fixed to 1 when !UseHp.
        let mv_class0_hp = read_hp(coder, probs.class0_hp_prob[comp], use_hp)?;
        // §9.3 step 2 runs even on the §9.3.1 integer-return arm
        // (UseHp == 0 → value fixed to 1 with no bit read), so the
        // count is unconditional.
        mc.class0_hp[mv_class0_hp as usize] += 1;
        ((mv_class0_bit << 3) | ((mv_class0_fr as u32) << 1) | mv_class0_hp) as i32 + 1
    } else {
        // d = sum over i in [0, mv_class) of (mv_bit << i).
        // mv_bit: mv_bits_prob[ comp ][ i ].
        let mut d: u32 = 0;
        for i in 0..mv_class as usize {
            let mv_bit = coder.read_bool(u32::from(probs.bits_prob[comp][i]))?;
            mc.bits[i][mv_bit as usize] += 1;
            d |= mv_bit << i;
        }
        // mag = CLASS0_SIZE << ( mv_class + 2 ).
        let mut mag = 1i32 << (CLASS0_SIZE_SHIFT as i32 + mv_class + 2);
        // mv_fr: mv_fr_probs[ comp ][ node ].
        let fr_probs = &probs.fr_probs[comp];
        let mv_fr = tree_decode(coder, &MV_FR_TREE, |node| fr_probs[node])?;
        mc.fr[mv_fr as usize] += 1;
        // mv_hp: mv_hp_prob[ comp ]; fixed to 1 when !UseHp. Counted
        // unconditionally (§9.3 step 2 on the integer-return arm).
        let mv_hp = read_hp(coder, probs.hp_prob[comp], use_hp)?;
        mc.hp[mv_hp as usize] += 1;
        mag += ((d << 3) | ((mv_fr as u32) << 1) | mv_hp) as i32 + 1;
        mag
    };

    Ok(if mv_sign != 0 { -mag } else { mag })
}

/// §9.3.3 `mv_class0_hp` / `mv_hp` decode. When `use_hp == 1` the bit is
/// read against `binary_tree` with probability `prob`; otherwise the bit
/// is absent and the value is fixed to `1` (`vp9-spec.txt` lines
/// 6214-6216).
fn read_hp(coder: &mut BoolCoder<'_>, prob: u8, use_hp: bool) -> Result<u32, Error> {
    if use_hp {
        // binary_tree[ 2 ] = { 0, -1 } -> read_bool( prob ) directly.
        Ok(tree_decode(coder, &BINARY_TREE, |_| prob)? as u32)
    } else {
        Ok(1)
    }
}

/// §6.4.19 `read_mv( ref )` — decode a full motion-vector difference and
/// add it to the §6.5.3 `BestMv[ ref ]` predictor.
///
/// ```text
/// read_mv( ref ) {
///   UseHp = allow_high_precision_mv && use_mv_hp( BestMv[ ref ] )
///   diffMv = ZeroMv
///   mv_joint
///   if ( mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ )
///     diffMv[ 0 ] = read_mv_component( 0 )
///   if ( mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ )
///     diffMv[ 1 ] = read_mv_component( 1 )
///   Mv[ ref ][ 0 ] = BestMv[ ref ][ 0 ] + diffMv[ 0 ]
///   Mv[ ref ][ 1 ] = BestMv[ ref ][ 1 ] + diffMv[ 1 ]
/// }
/// ```
///
/// `best_mv` is the §6.5.3 `BestMv[ ref ]` predictor; `allow_hp` is the
/// frame-level §7.2 `allow_high_precision_mv` flag. The two are combined
/// here into the local `UseHp` and threaded into each component read.
/// Returns the final `Mv[ ref ]` `[row, col]` in eighth-pel units.
pub(crate) fn read_mv(
    coder: &mut BoolCoder<'_>,
    probs: &MvProbs,
    best_mv: [i32; 2],
    allow_hp: bool,
    counts: &mut CountsNonCoef,
) -> Result<[i32; 2], Error> {
    let use_hp = allow_hp && use_mv_hp(best_mv);

    // mv_joint: mv_joint_probs[ node ].
    let joint = tree_decode(coder, &MV_JOINT_TREE, |node| probs.joint_probs[node])?;
    counts.mv_joint[joint as usize] += 1;

    let mut diff = [0i32; 2];
    if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
        diff[0] = read_mv_component(coder, probs, 0, use_hp, counts)?;
    }
    if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
        diff[1] = read_mv_component(coder, probs, 1, use_hp, counts)?;
    }

    Ok([best_mv[0] + diff[0], best_mv[1] + diff[1]])
}

/// The per-reference-list motion-vector predictors `assign_mv( )` reads.
///
/// One [`MvPredictors`] holds the §6.5.12 / §6.5.14 outputs for a single
/// reference list slot `i`: `nearest = NearestMv[ i ]`, `near =
/// NearMv[ i ]`, and `best = BestMv[ i ]`. For a `MiSize >= BLOCK_8X8`
/// block these come from §6.5.12 `find_best_ref_mvs( i )`; for a sub-8x8
/// block the `NEARESTMV` / `NEARMV` pair is replaced by §6.5.14
/// `append_sub8x8_mvs( )` per sub-block while `best` still carries the
/// `find_best_ref_mvs( )` value the `NEWMV` `read_mv( )` predicts onto.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MvPredictors {
    /// `NearestMv[ i ]` — the §6.4.18 `NEARESTMV` result for list `i`.
    pub nearest: [i32; 2],
    /// `NearMv[ i ]` — the §6.4.18 `NEARMV` result for list `i`.
    pub near: [i32; 2],
    /// `BestMv[ i ]` — the §6.5.3 predictor a `NEWMV` `read_mv( i )`
    /// difference is added onto.
    pub best: [i32; 2],
}

/// §6.4.18 `assign_mv( isCompound )` — resolve the final per-list motion
/// vectors `Mv[ 0 ]` / `Mv[ 1 ]` from the just-decoded `y_mode`.
///
/// ```text
/// assign_mv( isCompound ) {
///   Mv[ 1 ] = ZeroMv
///   for ( i = 0; i < 1 + isCompound; i++ ) {
///     if ( y_mode == NEWMV )
///       read_mv( i )
///     else if ( y_mode == NEARESTMV )
///       Mv[ i ] = NearestMv[ i ]
///     else if ( y_mode == NEARMV )
///       Mv[ i ] = NearMv[ i ]
///     else
///       Mv[ i ] = ZeroMv
///   }
/// }
/// ```
///
/// `y_mode` is one of the four §7.4.11 inter modes (`NEARESTMV = 10`,
/// `NEARMV = 11`, `ZEROMV = 12`, `NEWMV = 13`). `is_compound` is true
/// when `ref_frame[ 1 ] > INTRA_FRAME`, in which case both list slots are
/// resolved; otherwise only slot 0 is read and slot 1 stays the
/// pre-initialised §3 `ZeroMv`. `preds[ i ]` supplies the `NearestMv` /
/// `NearMv` / `BestMv` predictors for list `i`; only the `NEWMV` arm
/// touches the shared bool coder, decoding one §6.4.19 `read_mv( i )`
/// difference onto `BestMv[ i ]`. The returned `[[i32; 2]; 2]` is
/// `[Mv[ 0 ], Mv[ 1 ]]` in eighth-pel units.
pub(crate) fn assign_mv(
    coder: &mut BoolCoder<'_>,
    probs: &MvProbs,
    y_mode: u8,
    is_compound: bool,
    preds: &[MvPredictors; 2],
    allow_hp: bool,
    counts: &mut CountsNonCoef,
) -> Result<[[i32; 2]; 2], Error> {
    // Mv[ 1 ] = ZeroMv (set before the loop so a non-compound block
    // leaves slot 1 zero regardless of y_mode).
    let mut mv = [[0i32; 2], [0i32; 2]];

    let count = if is_compound { 2 } else { 1 };
    for i in 0..count {
        mv[i] = match y_mode {
            NEWMV => read_mv(coder, probs, preds[i].best, allow_hp, counts)?,
            NEARESTMV => preds[i].nearest,
            NEARMV => preds[i].near,
            // ZEROMV (and the §6.4.16 SEG_LVL_SKIP forced-ZEROMV case).
            _ => [0, 0],
        };
    }

    Ok(mv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed::MvProbs;

    /// A bool coder whose decoded bits are all 0 when the probability is
    /// at least 1: `init_bool` over an all-zero buffer.
    fn zero_coder(buf: &[u8]) -> BoolCoder<'_> {
        BoolCoder::init_bool(buf, buf.len()).unwrap()
    }

    #[test]
    fn use_mv_hp_small_components_true() {
        // Abs >> 3 of {0, 7} == {0, 0} < 8 -> true.
        assert!(use_mv_hp([0, 7]));
        assert!(use_mv_hp([-7, 7]));
    }

    #[test]
    fn use_mv_hp_large_component_false() {
        // Abs(64) >> 3 == 8, not < 8 -> false.
        assert!(!use_mv_hp([64, 0]));
        assert!(!use_mv_hp([0, -64]));
    }

    #[test]
    fn use_mv_hp_boundary_just_below_threshold() {
        // Abs(63) >> 3 == 7 < 8 -> true; Abs(64) >> 3 == 8 -> false.
        assert!(use_mv_hp([63, 63]));
        assert!(!use_mv_hp([63, 64]));
    }

    #[test]
    fn read_mv_zero_coder_joint_zero_returns_best_mv() {
        // Every read_bool returns 0 -> mv_joint walks to leaf
        // -MV_JOINT_ZERO, so no component is read and Mv == BestMv.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let probs = MvProbs::defaults();
        let mv = read_mv(&mut coder, &probs, [5, -9], true, &mut Default::default()).unwrap();
        assert_eq!(mv, [5, -9]);
    }

    #[test]
    fn read_mv_component_zero_coder_class0_min_magnitude() {
        // Zero coder: mv_sign=0, mv_class walks to MV_CLASS_0,
        // mv_class0_bit=0, mv_class0_fr -> leaf 0, mv_class0_hp:
        // binary_tree read_bool -> 0. mag = ((0<<3)|(0<<1)|0)+1 = 1.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let probs = MvProbs::defaults();
        let v = read_mv_component(&mut coder, &probs, 0, true, &mut Default::default()).unwrap();
        assert_eq!(v, 1);
    }

    #[test]
    fn read_mv_component_no_hp_still_class0_min_magnitude() {
        // With use_hp == false the mv_class0_hp bit is absent and reads
        // as 1: mag = ((0<<3)|(0<<1)|1)+1 = 2.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let probs = MvProbs::defaults();
        let v = read_mv_component(&mut coder, &probs, 0, false, &mut Default::default()).unwrap();
        assert_eq!(v, 2);
    }

    // ----- §9.3.4 mv counting -----

    #[test]
    fn read_mv_component_counts_every_decoded_element() {
        use crate::mode_info::MV_OFFSET_BITS;
        use crate::prob_adapt::CountsMvComponent;
        // Zero coder, use_hp = true, comp = 1: mv_sign=0, mv_class=0,
        // mv_class0_bit=0, mv_class0_fr=0 (under bit 0), mv_class0_hp=0.
        // Each decoded element charges exactly one §9.3.4 count in the
        // matching per-component array.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let probs = MvProbs::defaults();
        let mut counts = CountsNonCoef::default();
        read_mv_component(&mut coder, &probs, 1, true, &mut counts).unwrap();
        let mc = &counts.mv_comp[1];
        assert_eq!(mc.sign, [1, 0]);
        assert_eq!(mc.class[0], 1);
        assert_eq!(mc.class.iter().sum::<u32>(), 1);
        assert_eq!(mc.class0_bit, [1, 0]);
        assert_eq!(mc.class0_fr[0][0], 1, "fr counted under class0_bit row");
        assert_eq!(mc.class0_fr[1], [0; 4]);
        assert_eq!(mc.class0_hp, [1, 0]);
        // The other component and the non-class0 arrays stay untouched.
        assert_eq!(counts.mv_comp[0], CountsMvComponent::default());
        assert_eq!(mc.fr, [0; 4]);
        assert_eq!(mc.hp, [0, 0]);
        assert_eq!(mc.bits, [[0; 2]; MV_OFFSET_BITS]);
    }

    #[test]
    fn read_mv_component_counts_phantom_hp_when_use_hp_off() {
        // §9.3 step 2 anchored: with UseHp == 0 the §9.3.1 tree selection
        // for mv_class0_hp returns the integer 1 WITHOUT reading a bit,
        // and the counting process still runs — counts_mv_class0_hp[comp]
        // [1] += 1. A decoder that skips the count on the absent-bit arm
        // drifts in §8.4.4 adaptation whenever allow_high_precision_mv
        // is on but a large predictor suppresses the hp bit.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let probs = MvProbs::defaults();
        let mut counts = CountsNonCoef::default();
        read_mv_component(&mut coder, &probs, 0, false, &mut counts).unwrap();
        assert_eq!(
            counts.mv_comp[0].class0_hp,
            [0, 1],
            "absent hp bit still counts its fixed value 1"
        );
    }

    #[test]
    fn read_mv_counts_joint_only_on_zero_joint() {
        use crate::prob_adapt::CountsMvComponent;
        // MV_JOINT_ZERO: one counts_mv_joint event, no component counts.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let probs = MvProbs::defaults();
        let mut counts = CountsNonCoef::default();
        read_mv(&mut coder, &probs, [5, -9], true, &mut counts).unwrap();
        assert_eq!(counts.mv_joint, [1, 0, 0, 0]);
        assert_eq!(counts.mv_comp[0], CountsMvComponent::default());
        assert_eq!(counts.mv_comp[1], CountsMvComponent::default());
    }

    #[test]
    fn read_mv_joint_zero_ignores_allow_hp_flag() {
        // With the zero coder mv_joint walks to MV_JOINT_ZERO regardless
        // of UseHp, so neither component is read and Mv == BestMv whether
        // allow_high_precision_mv is set or not. This pins that the
        // allow_hp gate only influences the per-component hp bit, never
        // the joint selector or the no-difference fast path.
        let buf = [0u8; 16];
        let probs = MvProbs::defaults();
        let mut c0 = zero_coder(&buf);
        let mut c1 = zero_coder(&buf);
        assert_eq!(
            read_mv(&mut c0, &probs, [3, 4], false, &mut Default::default()).unwrap(),
            [3, 4]
        );
        assert_eq!(
            read_mv(&mut c1, &probs, [3, 4], true, &mut Default::default()).unwrap(),
            [3, 4]
        );
    }

    #[test]
    fn mv_class_tree_leaves_cover_zero_through_ten() {
        // The 11 leaves are exactly -0..=-10 (i.e. classes 0..=10).
        let mut leaves: Vec<i32> = MV_CLASS_TREE
            .iter()
            .filter(|&&n| n <= 0)
            .map(|&n| -n)
            .collect();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..=10).collect::<Vec<_>>());
    }

    #[test]
    fn mv_joint_tree_leaves_cover_zero_through_three() {
        let mut leaves: Vec<i32> = MV_JOINT_TREE
            .iter()
            .copied()
            .filter(|&n| n <= 0)
            .map(|n| -n)
            .collect();
        leaves.sort_unstable();
        assert_eq!(leaves, vec![0, 1, 2, 3]);
    }

    // ----- §6.4.18 assign_mv( ) -----

    const ZEROMV: u8 = 12;

    /// Two distinct predictor triples so a wrong-arm pick is visible.
    fn sample_preds() -> [MvPredictors; 2] {
        [
            MvPredictors {
                nearest: [3, -4],
                near: [5, 6],
                best: [7, 8],
            },
            MvPredictors {
                nearest: [-10, 11],
                near: [12, -13],
                best: [14, 15],
            },
        ]
    }

    #[test]
    fn assign_mv_zeromv_both_lists_zero_no_coder_read() {
        // ZEROMV reads nothing; slot 1 stays its pre-loop ZeroMv.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            ZEROMV,
            true,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(mv, [[0, 0], [0, 0]]);
    }

    #[test]
    fn assign_mv_nearestmv_single_picks_nearest_slot0_only() {
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            NEARESTMV,
            false,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        // Non-compound: slot 0 = NearestMv[0]; slot 1 stays ZeroMv.
        assert_eq!(mv, [[3, -4], [0, 0]]);
    }

    #[test]
    fn assign_mv_nearmv_compound_picks_near_both_lists() {
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            NEARMV,
            true,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(mv, [[5, 6], [12, -13]]);
    }

    #[test]
    fn assign_mv_nearestmv_compound_picks_nearest_both_lists() {
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            NEARESTMV,
            true,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(mv, [[3, -4], [-10, 11]]);
    }

    #[test]
    fn assign_mv_newmv_reads_difference_onto_best() {
        // An all-zero buffer makes mv_joint walk to MV_JOINT_ZERO, so
        // diffMv == ZeroMv and Mv[ i ] == BestMv[ i ] unchanged.
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            NEWMV,
            false,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        // Slot 0 = BestMv[0] + 0; slot 1 untouched ZeroMv.
        assert_eq!(mv, [[7, 8], [0, 0]]);
    }

    #[test]
    fn assign_mv_newmv_compound_reads_each_best() {
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            NEWMV,
            true,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(mv, [[7, 8], [14, 15]]);
    }

    #[test]
    fn assign_mv_noncompound_never_touches_slot1_predictors() {
        // Even with junk NearMv in slot 1, a non-compound NEARMV leaves
        // slot 1 at ZeroMv (the loop runs i = 0 only).
        let buf = [0u8; 16];
        let mut coder = zero_coder(&buf);
        let preds = sample_preds();
        let mv = assign_mv(
            &mut coder,
            &MvProbs::defaults(),
            NEARMV,
            false,
            &preds,
            true,
            &mut Default::default(),
        )
        .unwrap();
        assert_eq!(mv[1], [0, 0]);
        assert_eq!(mv[0], [5, 6]);
    }
}
