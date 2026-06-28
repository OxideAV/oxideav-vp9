//! VP9 §6.4.21 residual **encode driver** — the inverse of the
//! `BlockDecoder::residual( )` per-block walk.
//!
//! The decoder's §6.4.21 residual loop walks, per plane, the
//! `num4x4w × num4x4h` 4x4 transform-block grid with `step = 1 << txSz`,
//! decodes each block's §6.4.24 coefficient tokens from the shared §9.2
//! coder, reconstructs the block, and writes the per-block non-zero flag
//! back into the `AboveNonzeroContext` / `LeftNonzeroContext` strips.
//!
//! This module re-walks that same grid but, instead of *reading* tokens
//! from the §9.2 coder, it *writes* the coefficient syntax a caller
//! supplies (the quantized `Tokens` array per transform block, in raster
//! order) into a [`BoolEncoder`] via [`crate::token_writer::write_tokens`].
//! The per-plane tx-size selection (§6.4.22 `get_uv_tx_size`), the
//! §6.4.25 `TxType` selection (with the chroma / `TX_32X32` / 4x4-lossless
//! `DCT_DCT` overrides), the §6.4.25 `get_scan` scan order, the per-block
//! `TokenBlockCtx` derivation, and the §6.4.21 trailing non-zero
//! write-back are all shared with the decoder so a written residual
//! stream decodes back to the identical coefficients.
//!
//! Scope: the **intra** path (`is_inter == 0`), matching the
//! `decode_block_intra` residual the encoder bootstrap targets. The
//! caller supplies coefficients; quantization / forward-transform choices
//! live one layer up (the residual is simply whatever the caller wants
//! the decoder to reconstruct).
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.4.21 / §6.4.22 / §6.4.25 / §9.3.2;
//! the grid walk + context handling mirror the in-crate decoder exactly.

// The block / frame writer that drives this lands incrementally in this
// subsystem; until the top-level `encode_vp9` assembly is wired the
// module is exercised from the round-trip tests below.
#![allow(dead_code)]

use crate::bool_encoder::BoolEncoder;
use crate::idct::DCT_DCT;
use crate::reconstruct::tx_type_for_intra;
use crate::residual::{
    get_plane_block_size, get_uv_tx_size, BLOCK_8X8, BLOCK_INVALID, NUM_4X4_BLOCKS_HIGH_LOOKUP,
    NUM_4X4_BLOCKS_WIDE_LOOKUP,
};
use crate::scan::get_scan;
use crate::tokens::{NonzeroContext, TokenBlockCtx};
use crate::Error;
use crate::{intra::PredMode, mode_info::Vp9IntraMiBlock};

/// Per-frame geometry the residual writer needs, mirroring the
/// `BlockDecoder` fields the decoder's `residual( )` reads.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidualWriteCtx {
    /// `MiCols` — frame width in 8x8 MI units.
    pub mi_cols: u32,
    /// `MiRows` — frame height in 8x8 MI units.
    pub mi_rows: u32,
    /// `subsampling_x` from §6.2.2.
    pub subsampling_x: bool,
    /// `subsampling_y` from §6.2.2.
    pub subsampling_y: bool,
    /// `BitDepth` (8 / 10 / 12).
    pub bit_depth: u32,
    /// §6.2.9 `Lossless` flag (selects the 4x4 WHT `DCT_DCT` override).
    pub lossless: bool,
}

/// Coefficient source for the residual writer: invoked once per
/// *coded* transform block (the on-visible blocks of a non-skip MI
/// block) with the block's `(plane, tx_sz, start_x, start_y, block_idx)`
/// coordinates, returning the `segEob`-length quantized `Tokens` array
/// (raster order) the decoder should reconstruct. Returning an all-zero
/// array codes an explicit all-zero block (a `more_coefs == 0` at DC).
pub(crate) type CoefSource<'f> = dyn FnMut(usize, u32, u32, u32, usize) -> Vec<i64> + 'f;

/// The §6.4.25 `TxType` for an intra transform block, applying the
/// chroma / `TX_32X32` / 4x4-lossless `DCT_DCT` overrides exactly as the
/// decoder's `residual( )` does.
fn intra_tx_type(plane: usize, tx_sz: u32, lossless: bool, mode: PredMode) -> u8 {
    if plane > 0 || tx_sz == 3 {
        DCT_DCT
    } else if tx_sz == 0 {
        if lossless {
            DCT_DCT
        } else {
            tx_type_for_intra(mode)
        }
    } else {
        tx_type_for_intra(mode)
    }
}

/// Write the §6.4.21 residual syntax for one intra MI block — the inverse
/// of `BlockDecoder::residual( )`.
///
/// Walks the same per-plane / per-4x4 grid the decoder does, deriving the
/// per-block tx-size, `TxType`, scan order and `TokenBlockCtx` identically,
/// then for each on-visible coded block writes its coefficient tokens via
/// [`crate::token_writer::write_tokens`] and threads the §6.4.21 trailing
/// non-zero write-back into `nz`.
///
/// `coords` is `(MiRow, MiCol)`; `mi_size` is the §7.4.3 block size; `mi`
/// carries the decoded mode info (skip / tx_size / modes / segment_id).
/// When `mi.skip` is set, no tokens are written (the decoder reconstructs
/// from prediction only) but the `nz` write-back of zeros still runs for
/// every grid step, matching the decoder.
///
/// `coef_src` supplies the quantized coefficients per coded block; it is
/// only called for non-skip MI blocks at on-visible grid positions.
///
/// Returns `EobTotal` — the §6.4.24 accumulator (sum of the per-block
/// non-zero flags), the same value the decoder's `residual( )` returns.
#[allow(clippy::too_many_arguments)]
// The §6.4.21 loop is keyed by `plane` and indexes the three `nz` strips,
// the subsampling flags, and the per-plane mode by that same index, so the
// explicit-index form mirrors the spec listing directly.
#[allow(clippy::needless_range_loop)]
pub(crate) fn write_residual_intra(
    enc: &mut BoolEncoder,
    ctx: &ResidualWriteCtx,
    coords: (u32, u32),
    mi_size: u8,
    mi: &Vp9IntraMiBlock,
    coef_probs: &crate::coef_probs::CoefProbs,
    nz: &mut [NonzeroContext; 3],
    token_cache: &mut [u8],
    coef_src: &mut CoefSource<'_>,
) -> Result<u32, Error> {
    let (r, c) = coords;
    // §6.4.21: bsize = MiSize < BLOCK_8X8 ? BLOCK_8X8 : MiSize.
    let bsize = mi_size.max(BLOCK_8X8);
    let mut eob_total = 0u32;

    for plane in 0..3usize {
        // §6.4.21: txSz = (plane > 0) ? get_uv_tx_size( ) : tx_size.
        let tx_sz = if plane == 0 {
            mi.tx_size
        } else {
            get_uv_tx_size(mi.tx_size, mi_size, ctx.subsampling_x, ctx.subsampling_y)
        };
        let step = 1u32 << tx_sz;

        let plane_sz = get_plane_block_size(bsize, plane, ctx.subsampling_x, ctx.subsampling_y);
        if plane_sz == BLOCK_INVALID {
            return Err(Error::InvalidBitstream);
        }
        let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
        let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];

        let sub_x = plane > 0 && ctx.subsampling_x;
        let sub_y = plane > 0 && ctx.subsampling_y;
        let base_x = (c * 8) >> u32::from(sub_x);
        let base_y = (r * 8) >> u32::from(sub_y);
        let maxx = (ctx.mi_cols * 8) >> u32::from(sub_x);
        let maxy = (ctx.mi_rows * 8) >> u32::from(sub_y);

        let mut block_idx = 0usize;
        let mut y = 0u32;
        while y < num4x4h {
            let mut x = 0u32;
            while x < num4x4w {
                let start_x = base_x + 4 * x;
                let start_y = base_y + 4 * y;
                let mut nonzero = false;

                if start_x < maxx && start_y < maxy && !mi.skip {
                    // §8.5.1 mode selection mirrors the decoder.
                    let mode_raw = if plane > 0 {
                        mi.uv_mode
                    } else if mi_size >= BLOCK_8X8 {
                        mi.y_mode
                    } else {
                        mi.sub_modes[block_idx & 3]
                    };
                    let mode = PredMode::from_raw(mode_raw).ok_or(Error::InvalidBitstream)?;

                    let tx_type = intra_tx_type(plane, tx_sz, ctx.lossless, mode);
                    let scan = get_scan(plane, tx_sz, tx_type);
                    let n0 = 1usize << (tx_sz + 2);
                    let seg_eob = n0 * n0;

                    let tbc = TokenBlockCtx {
                        plane,
                        is_inter: false,
                        tx_type,
                        bit_depth: ctx.bit_depth,
                        x4: (start_x >> 2) as usize,
                        y4: (start_y >> 2) as usize,
                        max_x: ((2 * ctx.mi_cols) >> u32::from(sub_x)) as usize,
                        max_y: ((2 * ctx.mi_rows) >> u32::from(sub_y)) as usize,
                    };

                    let coeffs = coef_src(plane, tx_sz, start_x, start_y, block_idx);
                    debug_assert_eq!(coeffs.len(), seg_eob, "coef source length must be segEob");

                    token_cache[..seg_eob].fill(0);
                    nonzero = crate::token_writer::write_tokens(
                        enc,
                        &tbc,
                        tx_sz,
                        scan,
                        coef_probs,
                        &nz[plane],
                        &coeffs,
                        &mut token_cache[..seg_eob],
                    )?;
                    eob_total += u32::from(nonzero);
                }

                // §6.4.21 trailing write-back — fires for every (x, y)
                // step including off-visible / skip blocks (nonzero = 0).
                let x4 = (start_x >> 2) as usize;
                let y4 = (start_y >> 2) as usize;
                for i in 0..step as usize {
                    if x4 + i < nz[plane].above.len() {
                        nz[plane].above[x4 + i] = u8::from(nonzero);
                    }
                    if y4 + i < nz[plane].left.len() {
                        nz[plane].left[y4 + i] = u8::from(nonzero);
                    }
                }

                block_idx += 1;
                x += step;
            }
            y += step;
        }
    }

    Ok(eob_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coef_probs::DEFAULT_COEF_PROBS;
    use crate::mode_info::Vp9IntraMiBlock;

    const INTRA_FRAME: i32 = 0;
    const NONE_REF_FRAME: i32 = -1;

    fn mi_block(tx_size: u32, skip: bool, y_mode: u8, uv_mode: u8) -> Vp9IntraMiBlock {
        Vp9IntraMiBlock {
            segment_id: 0,
            skip,
            tx_size,
            ref_frame_0: INTRA_FRAME,
            ref_frame_1: NONE_REF_FRAME,
            is_inter: false,
            y_mode,
            sub_modes: [y_mode; 4],
            uv_mode,
        }
    }

    fn fresh_nz(mi_cols: u32, mi_rows: u32) -> [NonzeroContext; 3] {
        [
            NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
            NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
            NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        ]
    }

    use crate::bool_coder::BoolCoder;
    use crate::scan::get_scan;
    use crate::tokens::tokens;

    /// Re-decode a residual the writer produced by mirroring the
    /// decoder's §6.4.21 grid walk, recovering each transform block's
    /// `Tokens` array, and return them keyed by `(plane, start_x,
    /// start_y)`. This is the exact inverse walk the decoder runs in
    /// `BlockDecoder::residual( )`.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::needless_range_loop)]
    fn decode_residual_intra(
        bytes: &[u8],
        ctx: &ResidualWriteCtx,
        coords: (u32, u32),
        mi_size: u8,
        mi: &Vp9IntraMiBlock,
        coef_probs: &crate::coef_probs::CoefProbs,
    ) -> Vec<((usize, u32, u32), Vec<i64>)> {
        let (r, c) = coords;
        let bsize = mi_size.max(BLOCK_8X8);
        let mut nz = fresh_nz(ctx.mi_cols, ctx.mi_rows);
        let mut coder = BoolCoder::init_bool(bytes, bytes.len()).unwrap();
        let mut out = Vec::new();
        let mut cache = vec![0u8; 1024];

        for plane in 0..3usize {
            let tx_sz = if plane == 0 {
                mi.tx_size
            } else {
                get_uv_tx_size(mi.tx_size, mi_size, ctx.subsampling_x, ctx.subsampling_y)
            };
            let step = 1u32 << tx_sz;
            let plane_sz = get_plane_block_size(bsize, plane, ctx.subsampling_x, ctx.subsampling_y);
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];
            let sub_x = plane > 0 && ctx.subsampling_x;
            let sub_y = plane > 0 && ctx.subsampling_y;
            let base_x = (c * 8) >> u32::from(sub_x);
            let base_y = (r * 8) >> u32::from(sub_y);
            let maxx = (ctx.mi_cols * 8) >> u32::from(sub_x);
            let maxy = (ctx.mi_rows * 8) >> u32::from(sub_y);

            let mut block_idx = 0usize;
            let mut y = 0u32;
            while y < num4x4h {
                let mut x = 0u32;
                while x < num4x4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;
                    if start_x < maxx && start_y < maxy && !mi.skip {
                        let mode_raw = if plane > 0 {
                            mi.uv_mode
                        } else if mi_size >= BLOCK_8X8 {
                            mi.y_mode
                        } else {
                            mi.sub_modes[block_idx & 3]
                        };
                        let mode = PredMode::from_raw(mode_raw).unwrap();
                        let tx_type = intra_tx_type(plane, tx_sz, ctx.lossless, mode);
                        let scan = get_scan(plane, tx_sz, tx_type);
                        let n0 = 1usize << (tx_sz + 2);
                        let seg_eob = n0 * n0;
                        let tbc = TokenBlockCtx {
                            plane,
                            is_inter: false,
                            tx_type,
                            bit_depth: ctx.bit_depth,
                            x4: (start_x >> 2) as usize,
                            y4: (start_y >> 2) as usize,
                            max_x: ((2 * ctx.mi_cols) >> u32::from(sub_x)) as usize,
                            max_y: ((2 * ctx.mi_rows) >> u32::from(sub_y)) as usize,
                        };
                        cache[..seg_eob].fill(0);
                        let mut buf = vec![0i64; seg_eob];
                        nonzero = tokens(
                            &mut coder,
                            &tbc,
                            tx_sz,
                            scan,
                            coef_probs,
                            &nz[plane],
                            &mut cache[..seg_eob],
                            &mut buf,
                        )
                        .unwrap();
                        out.push(((plane, start_x, start_y), buf));
                    }
                    let x4 = (start_x >> 2) as usize;
                    let y4 = (start_y >> 2) as usize;
                    for i in 0..step as usize {
                        if x4 + i < nz[plane].above.len() {
                            nz[plane].above[x4 + i] = u8::from(nonzero);
                        }
                        if y4 + i < nz[plane].left.len() {
                            nz[plane].left[y4 + i] = u8::from(nonzero);
                        }
                    }
                    block_idx += 1;
                    x += step;
                }
                y += step;
            }
        }
        out
    }

    /// Encode a non-skip residual with caller-chosen coefficients per
    /// block, then decode it back and assert every block's `Tokens`
    /// array matches what the source supplied.
    fn roundtrip_residual(
        ctx: &ResidualWriteCtx,
        coords: (u32, u32),
        mi_size: u8,
        mi: &Vp9IntraMiBlock,
        chooser: impl Fn(usize, u32, u32, u32, usize) -> Vec<i64> + Copy,
    ) {
        let mut nz = fresh_nz(ctx.mi_cols, ctx.mi_rows);
        let mut cache = vec![0u8; 1024];
        let mut enc = BoolEncoder::new();
        // Record what we wrote per (plane, start_x, start_y).
        let mut written: Vec<((usize, u32, u32), Vec<i64>)> = Vec::new();
        {
            let mut src: Box<CoefSource> = Box::new(|p, t, sx, sy, b| {
                let v = chooser(p, t, sx, sy, b);
                written.push(((p, sx, sy), v.clone()));
                v
            });
            write_residual_intra(
                &mut enc,
                ctx,
                coords,
                mi_size,
                mi,
                &DEFAULT_COEF_PROBS,
                &mut nz,
                &mut cache,
                &mut src,
            )
            .unwrap();
        }
        let bytes = enc.finish();
        let decoded = decode_residual_intra(&bytes, ctx, coords, mi_size, mi, &DEFAULT_COEF_PROBS);
        assert_eq!(decoded.len(), written.len(), "block count mismatch");
        for (w, d) in written.iter().zip(decoded.iter()) {
            assert_eq!(w.0, d.0, "block coords mismatch");
            assert_eq!(w.1, d.1, "block {:?} tokens mismatch", w.0);
        }
    }

    fn ctx_420() -> ResidualWriteCtx {
        ResidualWriteCtx {
            mi_cols: 8,
            mi_rows: 8,
            subsampling_x: true,
            subsampling_y: true,
            bit_depth: 8,
            lossless: false,
        }
    }

    #[test]
    fn roundtrip_8x8_dc_only_luma() {
        let mi = mi_block(1, false, 0, 0);
        roundtrip_residual(
            &ctx_420(),
            (0, 0),
            BLOCK_8X8,
            &mi,
            |plane, tx_sz, _x, _y, _b| {
                let n0 = 1usize << (tx_sz + 2);
                let mut v = vec![0i64; n0 * n0];
                // DC-only, different per plane to confirm independence.
                v[0] = if plane == 0 { 5 } else { -2 };
                v
            },
        );
    }

    #[test]
    fn roundtrip_8x8_dense_4x4_grid() {
        // tx_size = TX_4X4 over a BLOCK_8X8 MI: four luma 4x4 blocks +
        // one chroma 4x4 per plane — exercises the nz write-back chain
        // across multiple blocks within the MI.
        let mi = mi_block(0, false, 9, 4);
        roundtrip_residual(
            &ctx_420(),
            (0, 0),
            BLOCK_8X8,
            &mi,
            |plane, tx_sz, sx, sy, _b| {
                let n0 = 1usize << (tx_sz + 2);
                let mut v = vec![0i64; n0 * n0];
                let seed = (plane as i64 + 1) * 3 + (sx as i64) - (sy as i64);
                v[0] = seed % 7;
                if n0 * n0 > 5 {
                    v[1] = -(seed % 3);
                    v[5] = (seed % 4) - 1;
                }
                v
            },
        );
    }

    #[test]
    fn roundtrip_16x16_high_magnitude() {
        let mi = mi_block(2, false, 0, 0);
        roundtrip_residual(
            &ctx_420(),
            (0, 0),
            crate::residual::BLOCK_16X16,
            &mi,
            |plane, tx_sz, _x, _y, _b| {
                let n0 = 1usize << (tx_sz + 2);
                let mut v = vec![0i64; n0 * n0];
                v[0] = if plane == 0 { 120 } else { 30 };
                v[1] = -45;
                v
            },
        );
    }

    /// A skip MI block writes no token bits and leaves `nz` cleared.
    #[test]
    fn skip_block_writes_nothing_and_clears_nz() {
        let ctx = ResidualWriteCtx {
            mi_cols: 8,
            mi_rows: 8,
            subsampling_x: true,
            subsampling_y: true,
            bit_depth: 8,
            lossless: false,
        };
        let mi = mi_block(0, true, 0, 0);
        let mut nz = fresh_nz(8, 8);
        // Pre-dirty an nz cell to confirm the write-back clears it.
        nz[0].above[0] = 1;
        let mut cache = vec![0u8; 1024];
        let mut enc = BoolEncoder::new();
        let mut src: Box<CoefSource> =
            Box::new(|_p, _t, _x, _y, _b| panic!("skip block must not request coeffs"));
        let eob = write_residual_intra(
            &mut enc,
            &ctx,
            (0, 0),
            BLOCK_8X8,
            &mi,
            &DEFAULT_COEF_PROBS,
            &mut nz,
            &mut cache,
            &mut src,
        )
        .unwrap();
        assert_eq!(eob, 0, "skip block has no EOB total");
        assert_eq!(nz[0].above[0], 0, "skip clears the nz write-back");
        let bytes = enc.finish();
        // Finish always emits at least the flush bytes; the point is no
        // token syntax was attempted (the panic source proves it).
        assert!(!bytes.is_empty());
    }
}
