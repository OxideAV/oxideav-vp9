//! VP9 §6.4.4 keyframe intra **block writer** — the inverse of
//! `BlockDecoder::decode_block_intra( )`.
//!
//! The decoder's §6.4.4 keyframe block decode runs, in order:
//!
//! 1. the §9.3.2 neighbour bundles (`NeighbourSkips` / `NeighbourTxSizes`
//!    / `IntraFrameNeighbours`) read from the frame-wide `Skips[ ]` /
//!    `TxSizes[ ]` / `SubModes[ ]` arrays the previous blocks' §6.4.4
//!    fan-outs populated;
//! 2. §6.4.7 `intra_segment_id( )`;
//! 3. §6.4.6 `intra_frame_mode_info( )` — `read_skip` / `read_tx_size(1)`
//!    / `default_intra_mode` / `default_uv_mode`;
//! 4. §6.4.21 `residual( )`;
//! 5. the §6.4.4 fan-out of the per-MI values into the frame-wide arrays.
//!
//! This module re-walks that sequence but *writes* each syntax element a
//! caller-supplied [`IntraBlockSpec`] dictates, deriving every §9.3.2
//! context from the same shared [`Vp9FrameState`] the decoder reads — so
//! a written block stream decodes back to the identical mode info +
//! coefficients. The neighbour derivation, the mode-info writers
//! ([`crate::mode_writer`]), the residual writer
//! ([`crate::residual_writer`]) and the §6.4.4 fan-out
//! ([`crate::decode_block::decode_block_apply`]) are all shared with the
//! decoder.
//!
//! Scope: `MiSize >= BLOCK_8X8` keyframe intra blocks (the common
//! keyframe path: a single luma mode replicated to the four sub-modes,
//! plus the chroma mode). The sub-8x8 per-4x4 mode walk
//! (`MiSize < BLOCK_8X8`) is a later milestone.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.4 / §6.4.6 / §6.4.7 / §6.4.8 /
//! §6.4.10 / §6.4.21; the field order mirrors the in-crate decoder
//! exactly.

// Driven by the top-level keyframe assembler (a later step in this
// subsystem); exercised now by the round-trip tests below.
#![allow(dead_code)]

use crate::bool_encoder::BoolEncoder;
use crate::compressed::TxMode;
use crate::decode_block::{decode_block_apply, DecodedBlockResult, Vp9FrameState};
use crate::mode_info::{
    tx_size_context, IntraFrameNeighbours, NeighbourSkips, NeighbourTxSizes, Vp9IntraMiBlock,
    DC_PRED, SIZE_GROUP_LOOKUP,
};
use crate::mode_writer::{
    tx_size_is_coded, write_default_intra_mode, write_default_uv_mode, write_segment_id,
    write_skip, write_tx_size,
};
use crate::residual::{BLOCK_8X8, MAX_TXSIZE_LOOKUP};
use crate::residual_writer::{write_residual_intra, CoefSource, ResidualWriteCtx};
use crate::Error;

const INTRA_FRAME: i32 = 0;
const NONE_REF_FRAME: i32 = -1;

/// Per-block keyframe intra spec the caller supplies to the writer. The
/// modes / tx-size / skip are the values the decoder will recover; the
/// residual coefficients come through the `coef_src` callback.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IntraBlockSpec {
    /// `(MiRow, MiCol)` top-left MI coordinate.
    pub r: u32,
    /// `MiCol` start of the current tile (for the §6.4.4 `AvailL`
    /// derivation `c > MiColStart`).
    pub mi_col_start: u32,
    /// `MiCol` (block column).
    pub c: u32,
    /// `MiSize` — the §3 `BLOCK_*` constant; must be `>= BLOCK_8X8`.
    pub mi_size: u8,
    /// `segment_id` (0..=7).
    pub segment_id: u8,
    /// `skip` flag.
    pub skip: bool,
    /// `tx_size` (`TX_*` 0..=3) — must be representable for the active
    /// `tx_mode` / `MiSize` (the writer verifies the coded/inferred gate).
    pub tx_size: u32,
    /// Luma `y_mode` (0..=9), replicated to the four sub-modes.
    pub y_mode: u8,
    /// Chroma `uv_mode` (0..=9).
    pub uv_mode: u8,
}

/// Frame-level parameters the block writer threads (the subset of the
/// §6.2 / §6.3 header the §6.4.6 / §6.4.21 walk depends on).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockWriteFrameCtx {
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub bit_depth: u32,
    pub lossless: bool,
    /// §6.3.1 `tx_mode` (gates whether `tx_size` is coded).
    pub tx_mode: TxMode,
    /// §6.2.11 segmentation enabled.
    pub seg_enabled: bool,
    /// §6.2.11 `segmentation_update_map`.
    pub seg_update_map: bool,
}

/// Write one keyframe intra MI block (`MiSize >= BLOCK_8X8`) — the
/// inverse of `decode_block_intra( )`.
///
/// Derives the §9.3.2 neighbour contexts from `state`, writes the
/// §6.4.7 segment id, the §6.4.6 mode info (skip / tx_size /
/// default_intra_mode / default_uv_mode), then the §6.4.21 residual via
/// [`write_residual_intra`], and finally fans the per-MI values into the
/// frame-wide arrays via [`decode_block_apply`] — exactly as the decoder
/// does — so the next block reads the right neighbour context.
///
/// `tree_probs` is `segmentation_tree_probs` (only consulted when
/// `seg_enabled && seg_update_map`). `skip_prob` / `tx_probs` /
/// `coef_probs` are the §6.3 default banks the from-defaults encoder
/// uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_keyframe_intra_block(
    enc: &mut BoolEncoder,
    fctx: &BlockWriteFrameCtx,
    spec: &IntraBlockSpec,
    state: &mut Vp9FrameState,
    nz: &mut [crate::tokens::NonzeroContext; 3],
    token_cache: &mut [u8],
    tree_probs: Option<&[u8; 7]>,
    skip_prob: &[u8; 3],
    tx_probs: &[[[u8; 3]; 2]; 4],
    coef_probs: &crate::coef_probs::CoefProbs,
    coef_src: &mut CoefSource<'_>,
) -> Result<(), Error> {
    if spec.mi_size < BLOCK_8X8 {
        // Sub-8x8 per-4x4 mode walk is a later milestone.
        return Err(Error::Unsupported);
    }
    let (r, c) = (spec.r, spec.c);

    // §6.4.4 lines 2400-2401: AvailU = r > 0, AvailL = c > MiColStart.
    let avail_u = r > 0;
    let avail_l = c > spec.mi_col_start;

    // §9.3.2 neighbour bundles from the frame-wide arrays.
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
    let nb_intra = IntraFrameNeighbours {
        avail_u,
        avail_l,
        above_sub_modes_23: if avail_u {
            [
                state.get_sub_mode(r - 1, c, 2).unwrap_or(DC_PRED),
                state.get_sub_mode(r - 1, c, 3).unwrap_or(DC_PRED),
            ]
        } else {
            [DC_PRED; 2]
        },
        left_sub_modes_13: if avail_l {
            [
                state.get_sub_mode(r, c - 1, 1).unwrap_or(DC_PRED),
                state.get_sub_mode(r, c - 1, 3).unwrap_or(DC_PRED),
            ]
        } else {
            [DC_PRED; 2]
        },
    };

    // §6.4.7 intra_segment_id( ): only coded when seg enabled + update_map.
    if fctx.seg_enabled && fctx.seg_update_map {
        let tp = tree_probs.ok_or(Error::InvalidBitstream)?;
        write_segment_id(enc, spec.segment_id, tp)?;
    }

    // §6.4.6 read_skip( ). seg_feature_skip_active is not handled in this
    // milestone (keyframe corpus paths don't force SKIP); pass false.
    let skip_ctx = skip_context(&nb_skip);
    write_skip(enc, spec.skip, false, skip_prob, skip_ctx)?;

    // §6.4.10 read_tx_size( 1 ): coded only under TX_MODE_SELECT for
    // MiSize >= BLOCK_8X8.
    let max_tx_size = MAX_TXSIZE_LOOKUP[spec.mi_size as usize];
    let coded = tx_size_is_coded(true, fctx.tx_mode, spec.mi_size >= BLOCK_8X8);
    if coded {
        let ctx = tx_size_context(nb_tx, max_tx_size);
        let row = &tx_probs[max_tx_size as usize][ctx];
        write_tx_size(enc, spec.tx_size, true, max_tx_size, row)?;
    }

    // §6.4.6 default_intra_mode (MiSize >= BLOCK_8X8 arm).
    let abovemode = if avail_u {
        nb_intra.above_sub_modes_23[0]
    } else {
        DC_PRED
    };
    let leftmode = if avail_l {
        nb_intra.left_sub_modes_13[0]
    } else {
        DC_PRED
    };
    write_default_intra_mode(enc, spec.y_mode, abovemode, leftmode)?;

    // §6.4.6 default_uv_mode.
    write_default_uv_mode(enc, spec.uv_mode, spec.y_mode)?;

    // §6.4.21 residual( ).
    let mi = Vp9IntraMiBlock {
        segment_id: spec.segment_id,
        skip: spec.skip,
        tx_size: spec.tx_size,
        ref_frame_0: INTRA_FRAME,
        ref_frame_1: NONE_REF_FRAME,
        is_inter: false,
        y_mode: spec.y_mode,
        sub_modes: [spec.y_mode; 4],
        uv_mode: spec.uv_mode,
    };
    let rctx = ResidualWriteCtx {
        mi_cols: fctx.mi_cols,
        mi_rows: fctx.mi_rows,
        subsampling_x: fctx.subsampling_x,
        subsampling_y: fctx.subsampling_y,
        bit_depth: fctx.bit_depth,
        lossless: fctx.lossless,
    };
    let eob_total = write_residual_intra(
        enc,
        &rctx,
        (r, c),
        spec.mi_size,
        &mi,
        coef_probs,
        nz,
        token_cache,
        coef_src,
    )?;

    // §6.4.4 fan-out. (Intra: the EobTotal skip-rewrite is a no-op.)
    let result = DecodedBlockResult {
        skip: spec.skip,
        tx_size: spec.tx_size as u8,
        y_mode: spec.y_mode,
        segment_id: spec.segment_id,
        ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
        is_inter: false,
        eob_total,
        interp_filter: 0,
        block_mvs: [[(0, 0); 4]; 2],
        sub_modes: [spec.y_mode; 4],
    };
    decode_block_apply(state, r, c, spec.mi_size, &result);
    Ok(())
}

/// §9.3.2 skip context: `(above_skip) + (left_skip)` over the available
/// neighbours (the same derivation `read_skip` performs from
/// `NeighbourSkips`).
fn skip_context(nb: &NeighbourSkips) -> usize {
    usize::from(nb.above.unwrap_or(0)) + usize::from(nb.left.unwrap_or(0))
}

/// `size_group_lookup[ MiSize ]` — re-exported for the assembler's
/// convenience (not used directly here, but keeps the §9.3.2 lookup in
/// one place for the keyframe-mode arm).
pub(crate) fn size_group(mi_size: u8) -> u8 {
    SIZE_GROUP_LOOKUP[mi_size as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::coef_probs::DEFAULT_COEF_PROBS;
    use crate::compressed::{DEFAULT_SKIP_PROB, DEFAULT_TX_PROBS};
    use crate::decode_frame::decode_keyframe_blocks_for_test;
    use crate::intra::Plane;
    use crate::tokens::NonzeroContext;

    fn fctx(tx_mode: TxMode) -> BlockWriteFrameCtx {
        BlockWriteFrameCtx {
            mi_cols: 8,
            mi_rows: 8,
            subsampling_x: true,
            subsampling_y: true,
            bit_depth: 8,
            lossless: false,
            tx_mode,
            seg_enabled: false,
            seg_update_map: false,
        }
    }

    fn fresh_nz() -> [NonzeroContext; 3] {
        [
            NonzeroContext::new(16, 16),
            NonzeroContext::new(16, 16),
            NonzeroContext::new(16, 16),
        ]
    }

    /// Write a grid of keyframe intra blocks, then decode the mode info +
    /// residual back and confirm every recovered block matches the spec.
    fn roundtrip_blocks(fc: BlockWriteFrameCtx, specs: &[IntraBlockSpec]) {
        let mut state = Vp9FrameState::new(fc.mi_rows, fc.mi_cols);
        let mut nz = fresh_nz();
        let mut cache = vec![0u8; 1024];
        let mut enc = BoolEncoder::new();

        for spec in specs {
            let mut src: Box<CoefSource> = Box::new(|_p, tx_sz, _x, _y, _b| {
                let n0 = 1usize << (tx_sz + 2);
                let mut v = vec![0i64; n0 * n0];
                v[0] = 3; // a small DC so the residual path is exercised.
                v
            });
            write_keyframe_intra_block(
                &mut enc,
                &fc,
                spec,
                &mut state,
                &mut nz,
                &mut cache,
                None,
                &DEFAULT_SKIP_PROB,
                &DEFAULT_TX_PROBS,
                &DEFAULT_COEF_PROBS,
                &mut src,
            )
            .unwrap();
        }
        let bytes = enc.finish();

        // Decode it back through the real decoder's block walk.
        let mut coder = BoolCoder::init_bool(&bytes, bytes.len()).unwrap();
        let mut y = Plane::new((fc.mi_cols * 8) as usize, (fc.mi_rows * 8) as usize);
        let mut u = Plane::new((fc.mi_cols * 4) as usize, (fc.mi_rows * 4) as usize);
        let mut v = Plane::new((fc.mi_cols * 4) as usize, (fc.mi_rows * 4) as usize);
        let recovered = decode_keyframe_blocks_for_test(
            &mut coder,
            fc.mi_rows,
            fc.mi_cols,
            fc.tx_mode,
            specs.iter().map(|s| (s.r, s.c, s.mi_size)).collect(),
            &mut y,
            &mut u,
            &mut v,
        )
        .unwrap();

        assert_eq!(recovered.len(), specs.len(), "block count");
        for (got, want) in recovered.iter().zip(specs.iter()) {
            assert_eq!(got.skip, want.skip, "skip @ ({},{})", want.r, want.c);
            assert_eq!(
                got.tx_size as u32, want.tx_size,
                "tx_size @ ({},{})",
                want.r, want.c
            );
            assert_eq!(got.y_mode, want.y_mode, "y_mode @ ({},{})", want.r, want.c);
            // uv_mode is not retained in the §6.4.4 frame-wide arrays, so
            // it is validated end-to-end at frame assembly (the chroma
            // reconstruction depends on it); here we confirm the
            // mode-info bit positions stayed in sync (a wrong uv_mode bit
            // count would desync the subsequent block's read).
            assert_eq!(
                got.segment_id, want.segment_id,
                "segment_id @ ({},{})",
                want.r, want.c
            );
        }
    }

    fn spec(r: u32, c: u32, mi_size: u8, tx: u32, y: u8, uv: u8) -> IntraBlockSpec {
        IntraBlockSpec {
            r,
            mi_col_start: 0,
            c,
            mi_size,
            segment_id: 0,
            skip: false,
            tx_size: tx,
            y_mode: y,
            uv_mode: uv,
        }
    }

    #[test]
    fn single_8x8_block_only4x4_roundtrips() {
        roundtrip_blocks(fctx(TxMode::Only4x4), &[spec(0, 0, BLOCK_8X8, 0, 0, 0)]);
    }

    #[test]
    fn two_8x8_blocks_neighbour_context_roundtrips() {
        // Two horizontally-adjacent 8x8 blocks: the second reads the
        // first's skip / tx / sub-mode neighbour context.
        roundtrip_blocks(
            fctx(TxMode::Only4x4),
            &[
                spec(0, 0, BLOCK_8X8, 0, 9, 4),
                spec(0, 1, BLOCK_8X8, 0, 3, 1),
            ],
        );
    }

    #[test]
    fn block_skip_true_roundtrips() {
        let mut s = spec(0, 0, BLOCK_8X8, 0, 0, 0);
        s.skip = true;
        roundtrip_blocks(fctx(TxMode::Only4x4), &[s]);
    }

    #[test]
    fn tx_mode_select_codes_tx_size_roundtrips() {
        // BLOCK_16X16 under TX_MODE_SELECT: tx_size is coded.
        roundtrip_blocks(
            fctx(TxMode::TxModeSelect),
            &[spec(0, 0, crate::residual::BLOCK_16X16, 1, 0, 0)],
        );
    }

    #[test]
    fn rejects_sub_8x8() {
        let fc = fctx(TxMode::Only4x4);
        let mut state = Vp9FrameState::new(fc.mi_rows, fc.mi_cols);
        let mut nz = fresh_nz();
        let mut cache = vec![0u8; 1024];
        let mut enc = BoolEncoder::new();
        let mut src: Box<CoefSource> = Box::new(|_p, tx_sz, _x, _y, _b| {
            let n0 = 1usize << (tx_sz + 2);
            vec![0i64; n0 * n0]
        });
        let s = spec(0, 0, crate::residual::BLOCK_4X4, 0, 0, 0);
        let r = write_keyframe_intra_block(
            &mut enc,
            &fc,
            &s,
            &mut state,
            &mut nz,
            &mut cache,
            None,
            &DEFAULT_SKIP_PROB,
            &DEFAULT_TX_PROBS,
            &DEFAULT_COEF_PROBS,
            &mut src,
        );
        assert_eq!(r.unwrap_err(), Error::Unsupported);
    }
}
