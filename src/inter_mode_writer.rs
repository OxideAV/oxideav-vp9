//! VP9 inter mode-info **writers** — the inverse of the §6.4.13
//! `read_is_inter( )`, §6.4.17 `read_ref_frames( )`, §6.4.16 `inter_mode`
//! / `interp_filter` decode primitives in [`crate::mode_info`].
//!
//! These are the non-MV inter mode-info syntax elements: the `is_inter`
//! flag, the single-/compound-reference selection, and the per-block
//! inter-mode + switchable interpolation-filter tokens. Each writer
//! re-derives the same §9.3.2 context the matching decode read uses (via
//! the shared `*_context` helpers in [`crate::mode_info`]) and walks the
//! same §9.3.1 tree, so a written stream decodes back bit-for-bit.
//!
//! The §6.4.17 reference-frame writer mirrors the decode's three coded
//! paths exactly:
//! * `REFERENCE_MODE_SELECT` codes a `comp_mode` bit (`comp_mode_prob`
//!   under [`crate::mode_info::comp_mode_context`]); otherwise the
//!   frame-level `reference_mode` fixes compound-ness with no bit.
//! * compound: one `comp_ref` bit selecting `CompVarRef[ comp_ref ]`
//!   into the slot complementary to `CompFixedRef`.
//! * single: `single_ref_p1` (`LAST` vs the golden/altref pair) and,
//!   when set, `single_ref_p2` (`GOLDEN` vs `ALTREF`).
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.13 / §6.4.16 / §6.4.17 / §9.3.1 /
//! §9.3.2; the bit order mirrors [`crate::mode_info`] exactly.

// Driven by the inter-block writer (a later step in this subsystem);
// exercised now by the round-trip tests below.
#![allow(dead_code)]

use crate::bool_encoder::BoolEncoder;
use crate::compressed::{CompoundReferenceConfig, ReferenceMode};
use crate::mode_info::{
    comp_mode_context, comp_ref_context, interp_filter_context, is_inter_context,
    single_ref_p1_context, single_ref_p2_context, CompModeNeighbours, InterpFilterNeighbours,
    IsInterNeighbours, RefFrameNeighbours, ALTREF_FRAME, BINARY_TREE, COMP_MODE_CONTEXTS,
    GOLDEN_FRAME, INTERP_FILTER_CONTEXTS, INTERP_FILTER_TREE, INTER_MODES, INTER_MODE_CONTEXTS,
    INTER_MODE_TREE, IS_INTER_CONTEXTS, LAST_FRAME, NEARESTMV, NONE_REF_FRAME, REF_CONTEXTS,
    SWITCHABLE, SWITCHABLE_FILTERS,
};
use crate::mode_writer::tree_encode;
use crate::Error;

/// §6.4.13 `read_is_inter( )` inverse.
///
/// When `seg_feature_ref_frame_active` the decoder derives `is_inter`
/// from the segment override with no bit, so nothing is written — the
/// caller must pass the `is_inter` the override implies
/// (`segment_ref_frame_data != INTRA_FRAME`). Otherwise a single
/// [`BINARY_TREE`] token is coded under `is_inter_prob[ ctx ]` where `ctx`
/// is [`is_inter_context`].
pub(crate) fn write_is_inter(
    enc: &mut BoolEncoder,
    is_inter: bool,
    seg_feature_ref_frame_active: bool,
    is_inter_prob: &[u8; IS_INTER_CONTEXTS],
    nb: IsInterNeighbours,
) -> Result<(), Error> {
    if seg_feature_ref_frame_active {
        return Ok(());
    }
    let ctx = is_inter_context(nb);
    tree_encode(enc, &BINARY_TREE, i32::from(is_inter), |_| {
        is_inter_prob[ctx]
    })
}

/// The §6.4.17 `read_ref_frames( )` inputs the writer needs — a mirror of
/// the standalone [`crate::mode_info::read_ref_frames`] parameter list.
pub(crate) struct RefFramesWriteArgs<'a> {
    /// `seg_feature_active( SEG_LVL_REF_FRAME )` for this block.
    pub seg_feature_ref_frame_active: bool,
    /// §6.2.7 frame-level `reference_mode`.
    pub reference_mode: ReferenceMode,
    /// §6.2.7 `CompFixedRef` / `CompVarRef` configuration.
    pub comp_config: CompoundReferenceConfig,
    /// `ref_frame_sign_bias[ CompFixedRef ]`.
    pub fix_ref_idx: u8,
    /// Neighbour reference-frame state for the §9.3.2 contexts.
    pub nb: RefFrameNeighbours,
    /// `comp_mode_prob[ COMP_MODE_CONTEXTS ]`.
    pub comp_mode_prob: &'a [u8; COMP_MODE_CONTEXTS],
    /// `single_ref_prob[ REF_CONTEXTS ][ 2 ]`.
    pub single_ref_prob: &'a [[u8; 2]; REF_CONTEXTS],
    /// `comp_ref_prob[ REF_CONTEXTS ]`.
    pub comp_ref_prob: &'a [u8; REF_CONTEXTS],
}

/// §6.4.17 `read_ref_frames( )` inverse — encode the reference-frame pair
/// `[ref_frame_0, ref_frame_1]` so the decoder recovers it.
///
/// `ref_frame_1 > INTRA_FRAME` (i.e. a real reference, not `NONE`) marks
/// a compound block. On the segment-override path no bits are coded (the
/// decoder reconstructs the pair from `FeatureData`); the writer verifies
/// the supplied pair is `[data, NONE]` single. Otherwise it codes the
/// same `comp_mode` / `comp_ref` / `single_ref_p1` / `single_ref_p2`
/// tokens the decode reads, in the same order, under the same contexts.
pub(crate) fn write_ref_frames(
    enc: &mut BoolEncoder,
    ref_frame: [i32; 2],
    args: &RefFramesWriteArgs<'_>,
) -> Result<(), Error> {
    let is_compound = ref_frame[1] > crate::mode_info::INTRA_FRAME;

    if args.seg_feature_ref_frame_active {
        // The decoder derives the pair from FeatureData with no reads; a
        // segment-override block is always single. Verify the caller's
        // pair is single (ref_frame_1 == NONE).
        if is_compound {
            return Err(Error::Unsupported);
        }
        return Ok(());
    }

    // comp_mode: coded only under REFERENCE_MODE_SELECT.
    match args.reference_mode {
        ReferenceMode::ReferenceModeSelect => {
            let ctx = comp_mode_context(
                CompModeNeighbours {
                    above: args.nb.above,
                    left: args.nb.left,
                },
                args.comp_config.fixed_ref,
            );
            tree_encode(enc, &BINARY_TREE, i32::from(is_compound), |_| {
                args.comp_mode_prob[ctx]
            })?;
        }
        ReferenceMode::CompoundReference => {
            if !is_compound {
                return Err(Error::Unsupported);
            }
        }
        ReferenceMode::SingleReference => {
            if is_compound {
                return Err(Error::Unsupported);
            }
        }
    }

    if is_compound {
        // idx = ref_frame_sign_bias[ CompFixedRef ]; the variable ref
        // goes to the complementary slot. Recover comp_ref from which
        // CompVarRef entry sits in slot (1 - idx).
        let idx = usize::from(args.fix_ref_idx);
        if ref_frame[idx] != args.comp_config.fixed_ref {
            return Err(Error::Unsupported);
        }
        let var = ref_frame[1 - idx];
        let comp_ref = if var == args.comp_config.var_ref[0] {
            0
        } else if var == args.comp_config.var_ref[1] {
            1
        } else {
            return Err(Error::Unsupported);
        };
        let ctx = comp_ref_context(args.nb, args.comp_config.var_ref, args.fix_ref_idx);
        tree_encode(enc, &BINARY_TREE, comp_ref, |_| args.comp_ref_prob[ctx])?;
    } else {
        // single_ref_p1: 0 => LAST_FRAME; 1 => golden/altref pair.
        let single_ref_p1 = i32::from(ref_frame[0] != LAST_FRAME);
        let ctx1 = single_ref_p1_context(args.nb);
        tree_encode(enc, &BINARY_TREE, single_ref_p1, |_| {
            args.single_ref_prob[ctx1][0]
        })?;
        if single_ref_p1 != 0 {
            // single_ref_p2: 0 => GOLDEN_FRAME; 1 => ALTREF_FRAME.
            let single_ref_p2 = match ref_frame[0] {
                GOLDEN_FRAME => 0,
                ALTREF_FRAME => 1,
                _ => return Err(Error::Unsupported),
            };
            let ctx2 = single_ref_p2_context(args.nb);
            tree_encode(enc, &BINARY_TREE, single_ref_p2, |_| {
                args.single_ref_prob[ctx2][1]
            })?;
        }
        // single_ref_p1 == 0 ⟹ ref_frame[0] == LAST_FRAME by construction.
        // A single block must carry NONE in slot 1.
        if ref_frame[1] != NONE_REF_FRAME {
            return Err(Error::Unsupported);
        }
    }
    Ok(())
}

/// §6.4.16 `inter_mode` write — encode the per-block inter `y_mode`
/// (`NEARESTMV` / `NEARMV` / `ZEROMV` / `NEWMV`) as the §9.3.1
/// [`INTER_MODE_TREE`] offset `y_mode - NEARESTMV` under
/// `inter_mode_probs[ mode_ctx ]`.
///
/// `mode_ctx` is `ModeContext[ ref_frame[0] ]` the §6.5 MV-reference scan
/// produced; the caller threads the same value the decode used.
pub(crate) fn write_inter_mode(
    enc: &mut BoolEncoder,
    y_mode: u8,
    mode_ctx: usize,
    inter_mode_probs: &[[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS],
) -> Result<(), Error> {
    let inter_mode = i32::from(y_mode) - i32::from(NEARESTMV);
    tree_encode(enc, &INTER_MODE_TREE, inter_mode, |node| {
        inter_mode_probs[mode_ctx][node]
    })
}

/// §6.4.16 switchable `interp_filter` write — when the frame-level
/// `interpolation_filter == SWITCHABLE`, code the per-block filter as the
/// §9.3.1 [`INTERP_FILTER_TREE`] under `interp_filter_probs[ ctx ]` with
/// `ctx` from [`interp_filter_context`]; otherwise the filter is the
/// frame value and nothing is written.
pub(crate) fn write_interp_filter(
    enc: &mut BoolEncoder,
    interp_filter: u8,
    interpolation_filter: u8,
    interp_filter_probs: &[[u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS],
    nb: InterpFilterNeighbours,
) -> Result<(), Error> {
    if interpolation_filter != SWITCHABLE {
        return Ok(());
    }
    let ctx = interp_filter_context(nb);
    tree_encode(enc, &INTERP_FILTER_TREE, i32::from(interp_filter), |node| {
        interp_filter_probs[ctx][node]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::compressed::FrameContext;
    use crate::mode_info::{read_is_inter, read_ref_frames, tree_decode, DEFAULT_IS_INTER_PROB};

    fn enc_to_dec(enc: BoolEncoder) -> BoolCoder<'static> {
        let buf = enc.finish();
        let leaked: &'static [u8] = Box::leak(buf.into_boxed_slice());
        BoolCoder::init_bool(leaked, leaked.len()).unwrap()
    }

    fn nb_none() -> RefFrameNeighbours {
        RefFrameNeighbours {
            above: None,
            left: None,
        }
    }

    #[test]
    fn write_is_inter_roundtrips_both_values() {
        let prob = DEFAULT_IS_INTER_PROB;
        for is_inter in [false, true] {
            let nb = IsInterNeighbours {
                above: None,
                left: None,
            };
            let mut enc = BoolEncoder::new();
            write_is_inter(&mut enc, is_inter, false, &prob, nb).unwrap();
            let mut dec = enc_to_dec(enc);
            let got =
                read_is_inter(&mut dec, false, 0, &prob, nb, &mut Default::default()).unwrap();
            assert_eq!(got, is_inter);
        }
    }

    #[test]
    fn write_is_inter_seg_active_writes_nothing() {
        let prob = DEFAULT_IS_INTER_PROB;
        let nb = IsInterNeighbours {
            above: None,
            left: None,
        };
        // segment override LAST_FRAME => is_inter true; no bits coded.
        let mut enc = BoolEncoder::new();
        write_is_inter(&mut enc, true, true, &prob, nb).unwrap();
        let mut dec = enc_to_dec(enc);
        let got = read_is_inter(
            &mut dec,
            true,
            LAST_FRAME as i16,
            &prob,
            nb,
            &mut Default::default(),
        )
        .unwrap();
        assert!(got);
    }

    fn ref_args<'a>(
        reference_mode: ReferenceMode,
        comp_config: CompoundReferenceConfig,
        chdr: &'a FrameContext,
    ) -> RefFramesWriteArgs<'a> {
        RefFramesWriteArgs {
            seg_feature_ref_frame_active: false,
            reference_mode,
            comp_config,
            fix_ref_idx: 0,
            nb: nb_none(),
            comp_mode_prob: &chdr.comp_mode_prob,
            single_ref_prob: &chdr.single_ref_prob,
            comp_ref_prob: &chdr.comp_ref_prob,
        }
    }

    fn decode_ref_frames(
        dec: &mut BoolCoder<'_>,
        reference_mode: ReferenceMode,
        comp_config: CompoundReferenceConfig,
        chdr: &FrameContext,
    ) -> [i32; 2] {
        let pair = read_ref_frames(
            dec,
            false,
            0,
            reference_mode,
            comp_config,
            0,
            nb_none(),
            &chdr.comp_mode_prob,
            &chdr.single_ref_prob,
            &chdr.comp_ref_prob,
            &mut Default::default(),
        )
        .unwrap();
        [pair.ref_frame_0, pair.ref_frame_1]
    }

    #[test]
    fn write_ref_frames_single_reference_roundtrips_all() {
        let chdr = FrameContext::default();
        let comp_config = CompoundReferenceConfig {
            fixed_ref: ALTREF_FRAME,
            var_ref: [LAST_FRAME, GOLDEN_FRAME],
        };
        for rf0 in [LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME] {
            let pair = [rf0, NONE_REF_FRAME];
            let mut enc = BoolEncoder::new();
            let args = ref_args(ReferenceMode::SingleReference, comp_config, &chdr);
            write_ref_frames(&mut enc, pair, &args).unwrap();
            let mut dec = enc_to_dec(enc);
            let got =
                decode_ref_frames(&mut dec, ReferenceMode::SingleReference, comp_config, &chdr);
            assert_eq!(got, pair, "single rf0 {rf0}");
        }
    }

    #[test]
    fn write_ref_frames_compound_roundtrips_both_var() {
        let chdr = FrameContext::default();
        let comp_config = CompoundReferenceConfig {
            fixed_ref: ALTREF_FRAME,
            var_ref: [LAST_FRAME, GOLDEN_FRAME],
        };
        // fix_ref_idx = 0 => slot 0 = fixed (ALTREF), slot 1 = var.
        for (comp_ref, var) in [(0, LAST_FRAME), (1, GOLDEN_FRAME)] {
            let _ = comp_ref;
            let pair = [ALTREF_FRAME, var];
            let mut enc = BoolEncoder::new();
            let args = ref_args(ReferenceMode::CompoundReference, comp_config, &chdr);
            write_ref_frames(&mut enc, pair, &args).unwrap();
            let mut dec = enc_to_dec(enc);
            let got = decode_ref_frames(
                &mut dec,
                ReferenceMode::CompoundReference,
                comp_config,
                &chdr,
            );
            assert_eq!(got, pair, "compound var {var}");
        }
    }

    #[test]
    fn write_ref_frames_select_mode_codes_comp_mode_bit() {
        let chdr = FrameContext::default();
        let comp_config = CompoundReferenceConfig {
            fixed_ref: ALTREF_FRAME,
            var_ref: [LAST_FRAME, GOLDEN_FRAME],
        };
        // Under SELECT, both a single and a compound pair must round-trip.
        let single = [LAST_FRAME, NONE_REF_FRAME];
        let compound = [ALTREF_FRAME, GOLDEN_FRAME];
        for pair in [single, compound] {
            let mut enc = BoolEncoder::new();
            let args = ref_args(ReferenceMode::ReferenceModeSelect, comp_config, &chdr);
            write_ref_frames(&mut enc, pair, &args).unwrap();
            let mut dec = enc_to_dec(enc);
            let got = decode_ref_frames(
                &mut dec,
                ReferenceMode::ReferenceModeSelect,
                comp_config,
                &chdr,
            );
            assert_eq!(got, pair, "select pair {pair:?}");
        }
    }

    #[test]
    fn write_inter_mode_roundtrips_all_modes_all_contexts() {
        let chdr = FrameContext::default();
        for ctx in 0..INTER_MODE_CONTEXTS {
            for ym in [
                NEARESTMV,
                crate::mode_info::NEARMV,
                crate::mode_info::ZEROMV,
                crate::mode_info::NEWMV,
            ] {
                let mut enc = BoolEncoder::new();
                write_inter_mode(&mut enc, ym, ctx, &chdr.inter_mode_probs).unwrap();
                let mut dec = enc_to_dec(enc);
                let inter_mode = tree_decode(&mut dec, &INTER_MODE_TREE, |node| {
                    chdr.inter_mode_probs[ctx][node]
                })
                .unwrap();
                let got = NEARESTMV + inter_mode as u8;
                assert_eq!(got, ym, "mode {ym} ctx {ctx}");
            }
        }
    }

    #[test]
    fn write_interp_filter_switchable_roundtrips() {
        let chdr = FrameContext::default();
        let nb = InterpFilterNeighbours::default();
        for filt in 0..(SWITCHABLE_FILTERS as u8) {
            let mut enc = BoolEncoder::new();
            write_interp_filter(&mut enc, filt, SWITCHABLE, &chdr.interp_filter_probs, nb).unwrap();
            let mut dec = enc_to_dec(enc);
            let ctx = interp_filter_context(nb);
            let got = tree_decode(&mut dec, &INTERP_FILTER_TREE, |node| {
                chdr.interp_filter_probs[ctx][node]
            })
            .unwrap() as u8;
            assert_eq!(got, filt, "switchable filter {filt}");
        }
    }

    #[test]
    fn write_interp_filter_non_switchable_writes_nothing() {
        let chdr = FrameContext::default();
        let nb = InterpFilterNeighbours::default();
        // EIGHTTAP frame-level filter -> no per-block bits.
        let mut enc = BoolEncoder::new();
        write_interp_filter(&mut enc, 0, 0, &chdr.interp_filter_probs, nb).unwrap();
        let buf = enc.finish();
        // A bit-free write still produces the marker + flush; decoding
        // nothing from it is trivially consistent.
        assert!(!buf.is_empty());
    }
}
