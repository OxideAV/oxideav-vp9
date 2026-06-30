//! VP9 §6.4.11 / §6.4.16 inter **block writer** — the inverse of
//! `BlockDecoder::decode_block_inter( )` for `MiSize >= BLOCK_8X8`.
//!
//! The decoder's §6.4.11 `inter_frame_mode_info( )` runs, in order:
//!
//! 1. §6.4.12 `inter_segment_id( )`;
//! 2. §6.4.8 `read_skip( )`;
//! 3. §6.4.13 `read_is_inter( )`;
//! 4. §6.4.10 `read_tx_size( !skip || !is_inter )`;
//! 5. §6.4.16 `inter_block_mode_info( )` (the `is_inter` arm) —
//!    §6.4.17 `read_ref_frames( )`, the §6.5 MV-reference scan
//!    (`find_mv_refs` / `find_best_ref_mvs`) deriving the predictors +
//!    `ModeContext`, the `inter_mode` token, the switchable
//!    `interp_filter` token, and §6.4.18 `assign_mv( )`;
//! 6. §6.4.21 `residual( )` (inter arm);
//! 7. the §6.4.4 fan-out into the frame-wide arrays.
//!
//! This module re-walks that exact sequence but *writes* each syntax
//! element a caller-supplied [`InterBlockSpec`] dictates. The §6.5
//! MV-reference scan is the **shared decode code** (`geom.find_mv_refs` /
//! `find_best_ref_mvs`) reading the same [`Vp9FrameState`], so the
//! predictors / `ModeContext` the writer derives are bit-identical to the
//! decoder's, and a written block decodes back to the same mode info +
//! motion vectors + coefficients.
//!
//! Scope: `MiSize >= BLOCK_8X8` inter blocks with segmentation either
//! disabled or a non-temporal `update_map` (the per-block `segment_id`
//! coded by the §6.4.7 tree). The sub-8x8 per-(idy, idx) MV walk
//! (`MiSize < BLOCK_8X8`) and the §6.4.12 temporal-predicted segment-id
//! branch are later milestones, mirroring how the keyframe block writer
//! deferred sub-8x8 intra.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.11 / §6.4.13 / §6.4.16 / §6.4.17 /
//! §6.4.18 / §6.4.21 / §6.5; the field order mirrors the in-crate decoder
//! exactly.

// Driven by the inter-frame assembler (a later step in this subsystem);
// exercised now by the round-trip tests below.
#![allow(dead_code)]

use crate::bool_encoder::BoolEncoder;
use crate::compressed::{CompoundReferenceConfig, MvProbs, ReferenceMode, TxMode};
use crate::decode_block::{decode_block_apply, DecodedBlockResult, Vp9FrameState};
use crate::inter_decode::FrameStateMvSource;
use crate::inter_mode_writer::{
    write_inter_mode, write_interp_filter, write_is_inter, write_ref_frames, RefFramesWriteArgs,
};
use crate::mode_info::{
    tx_size_context, InterpFilterNeighbours, IsInterNeighbours, NeighbourSkips, NeighbourTxSizes,
    RefFrameNeighbours, INTERP_FILTER_CONTEXTS, INTER_MODES, INTER_MODE_CONTEXTS, INTRA_FRAME,
    IS_INTER_CONTEXTS, NONE_REF_FRAME, SWITCHABLE, SWITCHABLE_FILTERS,
};
use crate::mode_writer::{tx_size_is_coded, write_skip, write_tx_size};
use crate::mv::MvPredictors;
use crate::mv_ref::MvRefGeometry;
use crate::mv_writer::write_assign_mv;
use crate::residual::{BLOCK_8X8, MAX_TXSIZE_LOOKUP};
use crate::residual_writer::{write_residual_inter, CoefSource, ResidualWriteCtx};
use crate::Error;

/// Per-block inter spec the caller supplies. The mode-info fields are the
/// values the decoder will recover; the residual comes through the
/// `coef_src` callback supplied to [`write_inter_block`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterBlockSpec {
    /// `(MiRow, MiCol)` top-left MI coordinate.
    pub r: u32,
    /// `MiCol` start of the current tile.
    pub mi_col_start: u32,
    /// `MiCol` (block column).
    pub c: u32,
    /// `MiSize` — must be `>= BLOCK_8X8`.
    pub mi_size: u8,
    /// `segment_id` (0..=7).
    pub segment_id: u8,
    /// `skip` flag (true ⇒ no residual coded).
    pub skip: bool,
    /// `tx_size` (`TX_*` 0..=3).
    pub tx_size: u32,
    /// The §3 reference pair `[ref_frame_0, ref_frame_1]`. `ref_frame_1 ==
    /// NONE` ⇒ single prediction; otherwise compound.
    pub ref_frame: [i32; 2],
    /// Inter `y_mode` (`NEARESTMV` / `NEARMV` / `ZEROMV` / `NEWMV`).
    pub y_mode: u8,
    /// `interp_filter` (`EIGHTTAP` / `_SMOOTH` / `_SHARP`) — coded only
    /// when the frame filter is `SWITCHABLE`.
    pub interp_filter: u8,
    /// The final per-list motion vectors `[Mv[0], Mv[1]]` (eighth-pel).
    /// For non-`NEWMV` modes these must equal the corresponding predictor
    /// (the writer verifies it); for `NEWMV` the difference onto `BestMv`
    /// is coded.
    pub mv: [[i32; 2]; 2],
}

/// Frame-level parameters the inter block writer threads (the subset of
/// the §6.2 / §6.3 header the §6.4.11 / §6.4.16 walk depends on).
#[derive(Clone, Copy)]
pub(crate) struct InterBlockFrameCtx<'a> {
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub mi_col_end: u32,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub bit_depth: u32,
    pub lossless: bool,
    pub tx_mode: TxMode,
    pub seg_enabled: bool,
    pub seg_update_map: bool,
    /// §6.2.7 frame-level `reference_mode`.
    pub reference_mode: ReferenceMode,
    /// §6.2.7 compound-reference config.
    pub comp_config: CompoundReferenceConfig,
    /// `ref_frame_sign_bias[ 4 ]`.
    pub sign_bias: &'a [bool; 4],
    /// §6.2.7 frame-level `interpolation_filter` (`SWITCHABLE` ⇒ per-block).
    pub interpolation_filter: u8,
    /// §6.2 `allow_high_precision_mv`.
    pub allow_high_precision_mv: bool,
    /// §6.5 `UsePrevFrameMvs` (false ⇒ no previous-frame motion field).
    pub use_prev_frame_mvs: bool,
}

/// The §6.3 probability banks the inter block writer codes against.
#[derive(Clone, Copy)]
pub(crate) struct InterBlockProbs<'a> {
    pub skip_prob: &'a [u8; 3],
    pub tx_probs: &'a [[[u8; 3]; 2]; 4],
    pub is_inter_prob: &'a [u8; IS_INTER_CONTEXTS],
    pub comp_mode_prob: &'a [u8; crate::mode_info::COMP_MODE_CONTEXTS],
    pub single_ref_prob: &'a [[u8; 2]; crate::mode_info::REF_CONTEXTS],
    pub comp_ref_prob: &'a [u8; crate::mode_info::REF_CONTEXTS],
    pub inter_mode_probs: &'a [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS],
    pub interp_filter_probs: &'a [[u8; SWITCHABLE_FILTERS - 1]; INTERP_FILTER_CONTEXTS],
    pub mv_probs: &'a MvProbs,
    pub coef_probs: &'a crate::coef_probs::CoefProbs,
    /// `segmentation_tree_probs` (only consulted when seg + update_map).
    pub tree_probs: Option<&'a [u8; 7]>,
}

/// Write one `MiSize >= BLOCK_8X8` inter MI block — the inverse of
/// `decode_block_inter( )`.
///
/// Derives the §9.3.2 neighbour contexts from `state`, writes the §6.4.11
/// prelude (segment id / skip / is_inter / tx_size), the §6.4.16 inter
/// mode info (ref frames / inter_mode / interp_filter / MVs), then the
/// §6.4.21 inter residual via [`write_residual_inter`], and finally fans
/// the per-MI values into the frame-wide arrays via [`decode_block_apply`]
/// — exactly as the decoder does — so the next block reads the right
/// neighbour context.
#[allow(clippy::too_many_arguments)]
// The `j in 0..2` reference-list loop indexes `spec.ref_frame` / `preds`
// in lock-step with the §6.4.16 listing, matching the decoder's form.
#[allow(clippy::needless_range_loop)]
pub(crate) fn write_inter_block(
    enc: &mut BoolEncoder,
    fctx: &InterBlockFrameCtx<'_>,
    spec: &InterBlockSpec,
    probs: &InterBlockProbs<'_>,
    state: &mut Vp9FrameState,
    nz: &mut [crate::tokens::NonzeroContext; 3],
    token_cache: &mut [u8],
    coef_src: &mut CoefSource<'_>,
) -> Result<(), Error> {
    if spec.mi_size < BLOCK_8X8 {
        // Sub-8x8 per-(idy, idx) MV walk is a later milestone.
        return Err(Error::Unsupported);
    }
    let (r, c) = (spec.r, spec.c);
    let avail_u = r > 0;
    let avail_l = c > spec.mi_col_start;

    // §6.4.11 prelude neighbour bundles from the frame-wide arrays.
    let above_rf = |rl: usize| -> Option<i32> {
        if avail_u {
            state.get_ref_frame(r - 1, c, rl)
        } else {
            None
        }
    };
    let left_rf = |rl: usize| -> Option<i32> {
        if avail_l {
            state.get_ref_frame(r, c - 1, rl)
        } else {
            None
        }
    };
    let above_pair = above_rf(0).map(|rf0| (rf0, above_rf(1).unwrap_or(NONE_REF_FRAME)));
    let left_pair = left_rf(0).map(|rf0| (rf0, left_rf(1).unwrap_or(NONE_REF_FRAME)));

    let nb_skip = NeighbourSkips {
        above: if avail_u {
            state.get_skip(r - 1, c)
        } else {
            None
        },
        left: if avail_l {
            state.get_skip(r, c - 1)
        } else {
            None
        },
    };
    let nb_tx = NeighbourTxSizes {
        avail_u,
        avail_l,
        skip_above: nb_skip.above.unwrap_or(0),
        skip_left: nb_skip.left.unwrap_or(0),
        tx_above: if avail_u {
            state.get_tx_size(r - 1, c).unwrap_or(0) as u32
        } else {
            0
        },
        tx_left: if avail_l {
            state.get_tx_size(r, c - 1).unwrap_or(0) as u32
        } else {
            0
        },
    };
    let is_inter_nb = IsInterNeighbours {
        above: above_rf(0),
        left: left_rf(0),
    };
    let interp_nb = InterpFilterNeighbours {
        above: above_rf(0).and_then(|rf0| {
            if rf0 > INTRA_FRAME {
                state.get_interp_filter(r - 1, c)
            } else {
                None
            }
        }),
        left: left_rf(0).and_then(|rf0| {
            if rf0 > INTRA_FRAME {
                state.get_interp_filter(r, c - 1)
            } else {
                None
            }
        }),
    };
    let ref_nb = RefFrameNeighbours {
        above: above_pair,
        left: left_pair,
    };

    // §6.4.12 inter_segment_id( ): the non-temporal path. Coded by the
    // §6.4.7 tree only when seg enabled + update_map; otherwise no bits.
    if fctx.seg_enabled && fctx.seg_update_map {
        let tp = probs.tree_probs.ok_or(Error::InvalidBitstream)?;
        crate::mode_writer::write_segment_id(enc, spec.segment_id, tp)?;
    } else if spec.segment_id != 0 {
        // A reused / disabled map can only carry segment 0 from the writer.
        return Err(Error::Unsupported);
    }

    // §6.4.8 read_skip( ). seg_feature_active(SKIP) is not modelled here
    // (the writer targets non-forced skips).
    let skip_ctx = usize::from(nb_skip.above.unwrap_or(0)) + usize::from(nb_skip.left.unwrap_or(0));
    write_skip(enc, spec.skip, false, probs.skip_prob, skip_ctx)?;

    // §6.4.13 read_is_inter( ): always inter here.
    write_is_inter(enc, true, false, probs.is_inter_prob, is_inter_nb)?;

    // §6.4.10 read_tx_size( !skip || !is_inter ): allow_select = !skip
    // (is_inter is true). Coded only under TX_MODE_SELECT for >= 8x8.
    let allow_select = !spec.skip;
    let max_tx_size = MAX_TXSIZE_LOOKUP[spec.mi_size as usize];
    let coded = tx_size_is_coded(allow_select, fctx.tx_mode, spec.mi_size >= BLOCK_8X8);
    if coded {
        let ctx = tx_size_context(nb_tx, max_tx_size);
        let row = &probs.tx_probs[max_tx_size as usize][ctx];
        write_tx_size(enc, spec.tx_size, true, max_tx_size, row)?;
    }

    // §6.4.17 read_ref_frames( ).
    let ref_args = RefFramesWriteArgs {
        seg_feature_ref_frame_active: false,
        reference_mode: fctx.reference_mode,
        comp_config: fctx.comp_config,
        fix_ref_idx: fctx.sign_bias[fctx.comp_config.fixed_ref as usize] as u8,
        nb: ref_nb,
        comp_mode_prob: probs.comp_mode_prob,
        single_ref_prob: probs.single_ref_prob,
        comp_ref_prob: probs.comp_ref_prob,
    };
    write_ref_frames(enc, spec.ref_frame, &ref_args)?;
    let is_compound = spec.ref_frame[1] > INTRA_FRAME;

    // §6.5 MV-reference scan — shared decode code, reading `state`.
    let geom = MvRefGeometry {
        mi_row: r as i32,
        mi_col: c as i32,
        mi_rows: fctx.mi_rows as i32,
        mi_cols: fctx.mi_cols as i32,
        mi_size: spec.mi_size as usize,
        mi_col_start: spec.mi_col_start as i32,
        mi_col_end: fctx.mi_col_end as i32,
    };
    let src = FrameStateMvSource::new(state, None);
    let mut preds = [MvPredictors::default(); 2];
    let mut mode_context = 0u8;
    for j in 0..2 {
        if spec.ref_frame[j] > INTRA_FRAME {
            let mv_refs = geom.find_mv_refs(
                &src,
                spec.ref_frame[j],
                -1,
                fctx.sign_bias,
                fctx.use_prev_frame_mvs,
            );
            if j == 0 {
                mode_context = mv_refs.mode_context;
            }
            let best = geom.find_best_ref_mvs(mv_refs.ref_list_mv, fctx.allow_high_precision_mv);
            preds[j] = MvPredictors {
                nearest: best[0],
                near: best[1],
                best: best[0],
            };
        }
    }
    let mode_ctx = usize::from(mode_context);

    // §6.4.16 inter_mode token (>= BLOCK_8X8 arm).
    write_inter_mode(enc, spec.y_mode, mode_ctx, probs.inter_mode_probs)?;

    // §6.4.16 switchable interp_filter.
    write_interp_filter(
        enc,
        spec.interp_filter,
        fctx.interpolation_filter,
        probs.interp_filter_probs,
        interp_nb,
    )?;

    // §6.4.18 assign_mv( ).
    write_assign_mv(
        enc,
        probs.mv_probs,
        spec.y_mode,
        is_compound,
        &preds,
        &spec.mv,
        fctx.allow_high_precision_mv,
    )?;

    // §6.4.21 residual( ) — inter arm.
    let rctx = ResidualWriteCtx {
        mi_cols: fctx.mi_cols,
        mi_rows: fctx.mi_rows,
        subsampling_x: fctx.subsampling_x,
        subsampling_y: fctx.subsampling_y,
        bit_depth: fctx.bit_depth,
        lossless: fctx.lossless,
    };
    let eob_total = write_residual_inter(
        enc,
        &rctx,
        (r, c),
        spec.mi_size,
        spec.tx_size,
        spec.skip,
        probs.coef_probs,
        nz,
        token_cache,
        coef_src,
    )?;

    // §6.4.4 fan-out. The interp_filter the decoder stores is the
    // resolved per-block value (frame value when not switchable).
    let stored_interp = if fctx.interpolation_filter == SWITCHABLE {
        spec.interp_filter
    } else {
        fctx.interpolation_filter
    };
    let block_mvs = {
        let mut bm = [[[0i32; 2]; 4]; 2];
        let ref_count = 1 + usize::from(is_compound);
        for (rl, slot) in bm.iter_mut().enumerate().take(ref_count) {
            for cell in slot.iter_mut() {
                *cell = spec.mv[rl];
            }
        }
        bm
    };
    let block_mvs_i16: [[(i16, i16); 4]; 2] = {
        let mut out = [[(0i16, 0i16); 4]; 2];
        for (rl, row) in block_mvs.iter().enumerate() {
            for (b, mv) in row.iter().enumerate() {
                out[rl][b] = (mv[0] as i16, mv[1] as i16);
            }
        }
        out
    };
    let result = DecodedBlockResult {
        skip: spec.skip,
        tx_size: spec.tx_size as u8,
        y_mode: spec.y_mode,
        segment_id: spec.segment_id,
        ref_frame: spec.ref_frame,
        is_inter: true,
        eob_total,
        interp_filter: stored_interp,
        block_mvs: block_mvs_i16,
        sub_modes: [crate::mode_info::DC_PRED; 4],
    };
    decode_block_apply(state, r, c, spec.mi_size, &result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed::FrameContext;
    use crate::mode_info::{ALTREF_FRAME, GOLDEN_FRAME, LAST_FRAME, NEARESTMV, NEWMV, ZEROMV};
    use crate::residual::{BLOCK_16X16, BLOCK_64X64};
    use crate::tokens::NonzeroContext;

    fn fresh_nz() -> [NonzeroContext; 3] {
        [
            NonzeroContext::new(64, 64),
            NonzeroContext::new(64, 64),
            NonzeroContext::new(64, 64),
        ]
    }

    fn frame_ctx<'a>(
        sign_bias: &'a [bool; 4],
        tx_mode: TxMode,
        reference_mode: ReferenceMode,
        interpolation_filter: u8,
        allow_hp: bool,
    ) -> InterBlockFrameCtx<'a> {
        InterBlockFrameCtx {
            mi_cols: 8,
            mi_rows: 8,
            mi_col_end: 8,
            subsampling_x: true,
            subsampling_y: true,
            bit_depth: 8,
            lossless: false,
            tx_mode,
            seg_enabled: false,
            seg_update_map: false,
            reference_mode,
            comp_config: CompoundReferenceConfig {
                fixed_ref: ALTREF_FRAME,
                var_ref: [LAST_FRAME, GOLDEN_FRAME],
            },
            sign_bias,
            interpolation_filter,
            allow_high_precision_mv: allow_hp,
            use_prev_frame_mvs: false,
        }
    }

    fn probs_view(fc: &FrameContext) -> InterBlockProbs<'_> {
        InterBlockProbs {
            skip_prob: &fc.skip_prob,
            tx_probs: &fc.tx_probs,
            is_inter_prob: &fc.is_inter_prob,
            comp_mode_prob: &fc.comp_mode_prob,
            single_ref_prob: &fc.single_ref_prob,
            comp_ref_prob: &fc.comp_ref_prob,
            inter_mode_probs: &fc.inter_mode_probs,
            interp_filter_probs: &fc.interp_filter_probs,
            mv_probs: &fc.mv_probs,
            coef_probs: &fc.coef_probs,
            tree_probs: None,
        }
    }

    fn no_coeffs() -> Box<CoefSource<'static>> {
        Box::new(|_p, tx_sz, _x, _y, _b| {
            let n0 = 1usize << (tx_sz + 2);
            vec![0i64; n0 * n0]
        })
    }

    /// Write a list of inter blocks, then decode the mode info back
    /// through the real decoder's inter block walk and confirm every
    /// recovered block matches the spec.
    fn roundtrip(
        fc: &InterBlockFrameCtx<'_>,
        ctx: &FrameContext,
        specs: &[InterBlockSpec],
    ) -> Vec<crate::decode_frame::RecoveredInterBlock> {
        let probs = probs_view(ctx);
        let mut state = Vp9FrameState::new(fc.mi_rows, fc.mi_cols);
        let mut nz = fresh_nz();
        let mut cache = vec![0u8; 4096];
        let mut enc = BoolEncoder::new();
        for spec in specs {
            let mut src = no_coeffs();
            write_inter_block(
                &mut enc, fc, spec, &probs, &mut state, &mut nz, &mut cache, &mut src,
            )
            .unwrap();
        }
        let bytes = enc.finish();
        crate::decode_frame::decode_inter_blocks_for_test(
            &bytes,
            fc.mi_rows,
            fc.mi_cols,
            ctx,
            fc.tx_mode,
            fc.reference_mode,
            fc.comp_config,
            fc.sign_bias,
            fc.interpolation_filter,
            fc.allow_high_precision_mv,
            specs.iter().map(|s| (s.r, s.c, s.mi_size)).collect(),
        )
        .unwrap()
    }

    fn base_spec(r: u32, c: u32, mi_size: u8) -> InterBlockSpec {
        InterBlockSpec {
            r,
            mi_col_start: 0,
            c,
            mi_size,
            segment_id: 0,
            skip: true,
            tx_size: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            y_mode: ZEROMV,
            interp_filter: 0,
            mv: [[0, 0], [0, 0]],
        }
    }

    #[test]
    fn single_zeromv_block_roundtrips() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::SingleReference,
            0,
            false,
        );
        let ctx = FrameContext::default();
        let got = roundtrip(&fc, &ctx, &[base_spec(0, 0, BLOCK_64X64)]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].y_mode, ZEROMV);
        assert_eq!(got[0].ref_frame, [LAST_FRAME, NONE_REF_FRAME]);
        assert_eq!(got[0].mv[0], [0, 0]);
        assert!(got[0].skip);
        assert_eq!(got[0].segment_id, 0);
    }

    #[test]
    fn newmv_block_roundtrips() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::SingleReference,
            0,
            true,
        );
        let ctx = FrameContext::default();
        // ZEROMV neighbourhood ⇒ BestMv = [0,0]; NEWMV codes a difference.
        let mut s = base_spec(0, 0, BLOCK_64X64);
        s.y_mode = NEWMV;
        s.mv = [[10, -6], [0, 0]];
        let got = roundtrip(&fc, &ctx, &[s]);
        assert_eq!(got[0].y_mode, NEWMV);
        assert_eq!(got[0].mv[0], [10, -6]);
    }

    #[test]
    fn switchable_interp_filter_roundtrips() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::SingleReference,
            SWITCHABLE,
            false,
        );
        let ctx = FrameContext::default();
        let mut s = base_spec(0, 0, BLOCK_64X64);
        s.interp_filter = 2; // EIGHTTAP_SHARP
        let got = roundtrip(&fc, &ctx, &[s]);
        assert_eq!(got[0].interp_filter, 2);
    }

    #[test]
    fn compound_reference_block_roundtrips() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::CompoundReference,
            0,
            false,
        );
        let ctx = FrameContext::default();
        let mut s = base_spec(0, 0, BLOCK_64X64);
        s.ref_frame = [ALTREF_FRAME, GOLDEN_FRAME];
        let got = roundtrip(&fc, &ctx, &[s]);
        assert_eq!(got[0].ref_frame, [ALTREF_FRAME, GOLDEN_FRAME]);
    }

    #[test]
    fn two_blocks_neighbour_context_roundtrips() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::SingleReference,
            0,
            false,
        );
        let ctx = FrameContext::default();
        // Two 8x8 blocks: the second reads the first's neighbour context.
        let mut a = base_spec(0, 0, BLOCK_8X8);
        a.ref_frame = [LAST_FRAME, NONE_REF_FRAME];
        let mut b = base_spec(0, 1, BLOCK_8X8);
        b.ref_frame = [GOLDEN_FRAME, NONE_REF_FRAME];
        b.y_mode = NEARESTMV;
        let got = roundtrip(&fc, &ctx, &[a, b]);
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].ref_frame, [GOLDEN_FRAME, NONE_REF_FRAME]);
    }

    #[test]
    fn non_skip_residual_block_roundtrips() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::SingleReference,
            0,
            false,
        );
        let ctx = FrameContext::default();
        let mut s = base_spec(0, 0, BLOCK_8X8);
        s.skip = false;
        let probs = probs_view(&ctx);
        let mut state = Vp9FrameState::new(fc.mi_rows, fc.mi_cols);
        let mut nz = fresh_nz();
        let mut cache = vec![0u8; 4096];
        let mut enc = BoolEncoder::new();
        // A DC token on every coded 4x4 block.
        let mut src: Box<CoefSource> = Box::new(|_p, tx_sz, _x, _y, _b| {
            let n0 = 1usize << (tx_sz + 2);
            let mut v = vec![0i64; n0 * n0];
            v[0] = 2;
            v
        });
        write_inter_block(
            &mut enc, &fc, &s, &probs, &mut state, &mut nz, &mut cache, &mut src,
        )
        .unwrap();
        let bytes = enc.finish();
        let got = crate::decode_frame::decode_inter_blocks_for_test(
            &bytes,
            fc.mi_rows,
            fc.mi_cols,
            &ctx,
            fc.tx_mode,
            fc.reference_mode,
            fc.comp_config,
            fc.sign_bias,
            fc.interpolation_filter,
            fc.allow_high_precision_mv,
            vec![(0, 0, BLOCK_8X8)],
        )
        .unwrap();
        assert_eq!(got[0].y_mode, ZEROMV);
    }

    #[test]
    fn tx_mode_select_codes_tx_size() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::TxModeSelect,
            ReferenceMode::SingleReference,
            0,
            false,
        );
        let ctx = FrameContext::default();
        let mut s = base_spec(0, 0, BLOCK_16X16);
        s.skip = false;
        s.tx_size = 1; // TX_8X8
        let got = roundtrip(&fc, &ctx, &[s]);
        assert_eq!(got[0].tx_size, 1);
    }

    #[test]
    fn rejects_sub_8x8() {
        let sb = [false; 4];
        let fc = frame_ctx(
            &sb,
            TxMode::Only4x4,
            ReferenceMode::SingleReference,
            0,
            false,
        );
        let ctx = FrameContext::default();
        let probs = probs_view(&ctx);
        let mut state = Vp9FrameState::new(fc.mi_rows, fc.mi_cols);
        let mut nz = fresh_nz();
        let mut cache = vec![0u8; 4096];
        let mut enc = BoolEncoder::new();
        let s = base_spec(0, 0, crate::residual::BLOCK_4X4);
        let mut src = no_coeffs();
        let r = write_inter_block(
            &mut enc, &fc, &s, &probs, &mut state, &mut nz, &mut cache, &mut src,
        );
        assert_eq!(r.unwrap_err(), Error::Unsupported);
    }
}
