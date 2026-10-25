//! Top-level intra-frame decode wiring — §6.4 `decode_tiles( )` driving
//! the §6.4.3 partition walk, the §6.4.4 `decode_block( )` composition
//! (§6.4.6 `intra_frame_mode_info( )` → §6.4.21 `residual( )` →
//! §6.4.4 fan-out) and the §8.8 frame-level loop filter, per the VP9
//! Bitstream & Decoding Process Specification v0.7.
//!
//! This module composes the primitives earlier rounds landed into a
//! whole-keyframe decode:
//!
//! 1. §6.2 [`parse_uncompressed_header`] — frame geometry, color
//!    config, quantizers, loop-filter / segmentation / tile params.
//! 2. §6.3 [`parse_compressed_header`] — `tx_mode`, `tx_probs`,
//!    `coef_probs`, `skip_prob` (the intra prefix; intra frames never
//!    reach the `FrameIsIntra == 0` tail).
//! 3. §6.4 tile walk — `tile_payload_sizes( )` + `get_tile_offset( )`
//!    per tile, `init_bool( ) / exit_bool( )` bracketing each tile's
//!    §9.2 coder, `clear_above_context( )` once per frame and
//!    `clear_left_context( )` per superblock row (§7.4.1 / §7.4.2),
//!    then `decode_partition( r, c, BLOCK_64X64 )` per superblock.
//! 4. §6.4.4 `decode_block( )` at every partition leaf, with the
//!    block syntax interleaved in the same bool coder as the
//!    partition syntax: §6.4.6 mode info (segment id, skip, tx size,
//!    intra modes), the §6.4.21 residual walk (per-plane §8.5.1
//!    `predict_intra` → §6.4.24 `tokens( )` → §8.6.2 `reconstruct`),
//!    and the fan-out into the frame-wide `Skips` / `TxSizes` /
//!    `MiSizes` / `YModes` / `SegmentIds` / `SubModes` arrays.
//! 5. §8.8 loop filter — `loop_filter_frame_init( )` from the §6.2.8
//!    header state, then the frame-level [`frame_loop_filter`] raster
//!    over the reconstructed planes.
//! 6. §8.10 output — the decoded planes cropped to `FrameWidth` x
//!    `FrameHeight` (luma) and the subsampled chroma extents.
//!
//! Keyframes and intra-only frames decode end-to-end; inter frames
//! (which need reference-buffer state and the §8.5.2 inter prediction
//! process) return [`Error::Unsupported`].
//!
//! ## Provenance
//!
//! Clean-room, single source of truth: `docs/video/vp9/vp9-spec.txt`
//! (§6.2, §6.3, §6.4-§6.4.26, §7.2.6, §7.4.1-§7.4.4, §8.5.1, §8.6,
//! §8.7, §8.8, §8.10).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use oxideav_core::ExecutionContext;

use crate::Error;
use crate::bool_coder::BoolCoder;
use crate::compressed::{
    CompoundReferenceConfig, FrameContext, RefFrameSignBias, Vp9CompressedHeader,
    Vp9CompressedHeaderInter, Vp9CompressedHeaderInterInputs, parse_compressed_header,
    parse_compressed_header_inter_with_ctx, parse_compressed_header_with_ctx,
};
use crate::decode_block::{DecodedBlockResult, Vp9FrameState, decode_block_apply};
use crate::dequant::{get_ac_quant, get_dc_quant, seg_feature_active};
use crate::frame_buf::{Band, FrameBuf, HALO, MiRow, PlaneGeom, band_rows};
use crate::header::{
    ColorConfig, FrameType, MAX_SEGMENTS, QuantizationParams, RefFrameState, SEG_LVL_MAX,
    SegmentationParams, Vp9FrameHeader, parse_uncompressed_header,
    parse_uncompressed_header_with_refs,
};
use crate::inter_decode::{FrameStateMvSource, PrevFrameMvs, build_ref_planes};
use crate::inter_mv::{BlockGrid, ScaleGeom};
use crate::inter_pred::InterPredArgs;
use crate::intra::{Plane, PredMode, predict_intra};
use crate::loop_filter::loop_filter_frame_init;
use crate::mode_info::{
    ALTREF_FRAME, DC_PRED, GOLDEN_FRAME, INTRA_FRAME, InterFrameModeArgs, InterRefFrameArgs,
    InterSegmentIdArgs, InterpFilterNeighbours, IntraFrameNeighbours, IsInterNeighbours,
    LAST_FRAME, NONE_REF_FRAME, NeighbourSkips, NeighbourTxSizes, PrevSegmentIds,
    RefFrameNeighbours, SEG_LVL_SKIP, SegFeatureTables, SegPredContextState, Vp9InterFrameBlock,
    Vp9IntraMiBlock, inter_frame_mode_info, intra_frame_mode_info, intra_segment_id,
};
use crate::mv_ref::MvRefGeometry;
use crate::partition::{
    LeafSink, PartitionContextState, PartitionProbsKind, decode_partition, get_tile_offset,
    tile_payload_sizes,
};
use crate::reconstruct::{reconstruct_block, tx_type_for_intra};
use crate::ref_buffer::{REFS_PER_FRAME, RefBuffers};
use crate::residual::{
    BLOCK_8X8, BLOCK_64X64, BLOCK_INVALID, NUM_4X4_BLOCKS_HIGH_LOOKUP, NUM_4X4_BLOCKS_WIDE_LOOKUP,
    get_plane_block_size, get_uv_tx_size,
};
use crate::scan::get_scan;
use crate::superblock_loop_filter::{
    LfRowCells, PlaneWindow, SuperblockFilterFrame, superblock_loop_filter_cells,
};
use crate::tokens::{NonzeroContext, TokenBlockCtx, tokens_eob};
use crate::worker_pool::{Batch, Job, WorkerPool};

/// One decoded VP9 frame, cropped to the §8.10 output extents.
///
/// Planes are row-major, one `u16` per sample (the native range for
/// every `BitDepth`; 8-bit samples occupy the low byte). Luma spans
/// `width x height`; each chroma plane spans
/// `((width + subsampling_x) >> subsampling_x) x
/// ((height + subsampling_y) >> subsampling_y)` per §8.10.
#[derive(Debug, Clone)]
pub struct Vp9DecodedFrame {
    /// `FrameWidth` per §7.2.5.
    pub width: u32,
    /// `FrameHeight` per §7.2.5.
    pub height: u32,
    /// `BitDepth` per §7.2.2 (8, 10 or 12).
    pub bit_depth: u8,
    /// §7.2.2 `subsampling_x`.
    pub subsampling_x: bool,
    /// §7.2.2 `subsampling_y`.
    pub subsampling_y: bool,
    /// Decoded luma plane (`width * height` samples).
    pub y: Vec<u16>,
    /// Decoded U chroma plane (§8.10 subsampled extent).
    pub u: Vec<u16>,
    /// Decoded V chroma plane (§8.10 subsampled extent).
    pub v: Vec<u16>,
}

use crate::loop_filter::{DEFAULT_LOOP_FILTER_MODE_DELTAS, DEFAULT_LOOP_FILTER_REF_DELTAS};

/// Per-frame state threaded through the §6.4.4 `decode_block( )` call
/// sites by the §6.4.3 partition walk.
///
/// Owns mutable views of everything one block decode touches: the
/// `CurrFrame` planes, the §6.4.4 frame-wide arrays, the
/// `AboveNonzeroContext` / `LeftNonzeroContext` strips, plus scratch
/// buffers for the §6.4.24 `Tokens[ ]` / `TokenCache[ ]` arrays. The
/// per-tile `MiColStart` feeds the §6.4.4 `AvailL = c > MiColStart`
/// derivation.
struct BlockDecoder<'a> {
    seg: &'a SegmentationParams,
    quant: &'a QuantizationParams,
    chdr: &'a Vp9CompressedHeader,
    mi_rows: u32,
    mi_cols: u32,
    mi_col_start: u32,
    mi_col_end: u32,
    subsampling_x: bool,
    subsampling_y: bool,
    bit_depth: u32,
    lossless: bool,
    state: &'a mut Vp9FrameState,
    /// The pixel work (predictions, reconstructions): done now or
    /// logged for replay.
    sink: Sink<'a>,
    nz: &'a mut [NonzeroContext; 3],
    token_cache: &'a mut [u8; 1024],
    tok_buf: &'a mut [i64; 1024],
    /// §9.3.4 per-frame syntax-element count bank (cleared per frame by
    /// §8.3 `clear_counts( )`; consumed by §6.1.2 `refresh_probs( )`).
    counts: &'a mut crate::prob_adapt::FrameCounts,
    /// `Some(_)` on inter (non-intra-only) frames; `None` on keyframes /
    /// intra-only frames. Carries the §6.3 inter compressed-header tail,
    /// the §8.10 reference buffers, the §6.2 inter header flags, the
    /// §6.4.11 segmentation-prediction context, and the previous frame's
    /// motion field. The §6.4.11 inter mode-info decode + §8.5.2
    /// `predict_inter( )` residual arm only run when this is `Some(_)`.
    inter: Option<InterCtx<'a>>,
}

/// The inter-frame decode context the §6.4.11 `inter_frame_mode_info( )`
/// driver and the §8.5.2 `predict_inter( )` residual arm need on a
/// non-intra-only frame.
struct InterCtx<'a> {
    /// The §6.3 inter compressed-header tail (probability tables +
    /// `reference_mode` + compound config + `mv_probs`).
    chdr: &'a Vp9CompressedHeaderInter,
    /// §8.10 reference-frame slots resolved through `ref_frame_idx`.
    refs: &'a RefBuffers,
    /// §6.2 `ref_frame_idx[ 3 ]` slot map.
    ref_frame_idx: [u8; REFS_PER_FRAME],
    /// §6.2.5 `ref_frame_sign_bias[ 4 ]` (indexed by §3 ref-frame value).
    sign_bias: [bool; 4],
    /// §6.2.7 `interpolation_filter` frame-level value (`SWITCHABLE = 4`
    /// ⇒ per-block tree read).
    interpolation_filter: u8,
    /// §6.2 `allow_high_precision_mv`.
    allow_high_precision_mv: bool,
    /// §7.2.6 `UsePrevFrameMvs`.
    use_prev_frame_mvs: bool,
    /// §6.3.18 compound-reference config (`SingleReference` frames carry
    /// a placeholder; `read_ref_frames( )` only reads it on the compound
    /// arms).
    comp_config: CompoundReferenceConfig,
    /// §8.10 `FrameWidth` / `FrameHeight` of the frame being decoded (for
    /// the §8.5.2.3 scale geometry).
    frame_width: u32,
    frame_height: u32,
    /// Previous decoded frame's `RefFrames[ ][ ][ 0 ]` (`[mi*2+list]`),
    /// read by the §6.5.10 candidate scan when `use_prev_frame_mvs`.
    prev_ref_frames: &'a [i32],
    /// Previous decoded frame's `Mvs[ ][ ][ ]` (`[mi*2+list]`).
    prev_mvs: &'a [(i16, i16)],
    /// Previous decoded frame's `SegmentId[ ][ ]` plane (§6.4.14).
    prev_segment_ids: &'a [u8],
    /// §6.4.11 segmentation-prediction context strips.
    seg_pred_ctx: &'a mut SegPredContextState,
}

impl LeafSink for BlockDecoder<'_> {
    fn leaf(
        &mut self,
        coder: &mut BoolCoder<'_>,
        r: u32,
        c: u32,
        subsize: u8,
    ) -> Result<(), Error> {
        self.decode_block(coder, r, c, subsize)
    }

    fn record_partition(&mut self, ctx: usize, partition: u8) {
        // §9.3.4: counts_partition[ ctx ][ syntax ] += 1.
        self.counts.noncoef.partition[ctx][partition as usize] += 1;
    }
}

impl BlockDecoder<'_> {
    /// §6.4.4 `decode_block( r, c, subsize )` — dispatches the §6.4.5
    /// intra-frame (keyframe) arm or the §6.4.11 inter-frame arm.
    fn decode_block(
        &mut self,
        coder: &mut BoolCoder<'_>,
        r: u32,
        c: u32,
        subsize: u8,
    ) -> Result<(), Error> {
        if self.inter.is_some() {
            self.decode_block_inter(coder, r, c, subsize)
        } else {
            self.decode_block_intra(coder, r, c, subsize)
        }
    }

    /// §6.4.4 `decode_block( r, c, subsize )` — the keyframe / intra-only
    /// arm (§6.4.6 `intra_frame_mode_info( )`).
    fn decode_block_intra(
        &mut self,
        coder: &mut BoolCoder<'_>,
        r: u32,
        c: u32,
        subsize: u8,
    ) -> Result<(), Error> {
        // §6.4.4 lines 2400-2401: AvailU = r > 0, AvailL = c > MiColStart.
        let avail_u = r > 0;
        let avail_l = c > self.mi_col_start;

        // §9.3.2 neighbour bundles from the frame-wide arrays the
        // previous blocks' §6.4.4 fan-outs populated.
        let nb_skip = NeighbourSkips {
            above: if avail_u {
                self.state.get_skip(r - 1, c)
            } else {
                None
            },
            left: if avail_l {
                self.state.get_skip(r, c - 1)
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
                self.state.get_tx_size(r - 1, c).unwrap_or(0) as u32
            } else {
                0
            },
            tx_left: if avail_l {
                self.state.get_tx_size(r, c - 1).unwrap_or(0) as u32
            } else {
                0
            },
        };
        let nb_intra = IntraFrameNeighbours {
            avail_u,
            avail_l,
            above_sub_modes_23: if avail_u {
                [
                    self.state.get_sub_mode(r - 1, c, 2).unwrap_or(DC_PRED),
                    self.state.get_sub_mode(r - 1, c, 3).unwrap_or(DC_PRED),
                ]
            } else {
                [DC_PRED; 2]
            },
            left_sub_modes_13: if avail_l {
                [
                    self.state.get_sub_mode(r, c - 1, 1).unwrap_or(DC_PRED),
                    self.state.get_sub_mode(r, c - 1, 3).unwrap_or(DC_PRED),
                ]
            } else {
                [DC_PRED; 2]
            },
        };

        // §6.4.6 first line — §6.4.7 intra_segment_id( ) — hoisted out
        // of `intra_frame_mode_info` so the §6.4.9
        // `seg_feature_active( SEG_LVL_SKIP )` gate the subsequent
        // §6.4.8 read_skip( ) needs can be derived from the decoded
        // segment id. The inner call is then handed a disabled
        // segmentation triple, making its own §6.4.7 step a bit-free
        // `segment_id = 0` that we overwrite. Bit order is identical
        // to the spec listing.
        let segment_id = intra_segment_id(
            coder,
            self.seg.enabled,
            self.seg.update_map,
            self.seg.tree_probs.as_ref(),
        )?;
        let seg_skip_active = seg_feature_active(self.seg, segment_id as usize, SEG_LVL_SKIP);

        let mut mi = intra_frame_mode_info(
            coder,
            subsize,
            false,
            false,
            None,
            seg_skip_active,
            self.chdr.tx_mode,
            &self.chdr.tx_probs,
            &self.chdr.skip_prob,
            nb_skip,
            nb_tx,
            nb_intra,
            &mut self.counts.noncoef,
        )?;
        mi.segment_id = segment_id;

        // §6.4.4 lines 2403-2404: EobTotal = 0; residual( ).
        let eob_total = self.residual(coder, r, c, subsize, &mi, avail_u, avail_l)?;

        // §6.4.4 lines 2405-2436: the skip rewrite (a no-op on intra
        // blocks) plus the frame-wide fan-out.
        let result = DecodedBlockResult {
            skip: mi.skip,
            tx_size: mi.tx_size as u8,
            y_mode: mi.y_mode,
            segment_id,
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            is_inter: false,
            eob_total,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: mi.sub_modes,
        };
        decode_block_apply(self.state, r, c, subsize, &result);
        Ok(())
    }

    /// §6.4.21 `residual( )` — the intra arm, decoding tokens inline
    /// from the shared §9.2 coder and reconstructing each transform
    /// block in place. Returns the §6.4.24 `EobTotal` accumulator.
    #[allow(clippy::too_many_arguments)]
    fn residual(
        &mut self,
        coder: &mut BoolCoder<'_>,
        r: u32,
        c: u32,
        mi_size: u8,
        mi: &Vp9IntraMiBlock,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<u32, Error> {
        // §6.4.21: bsize = MiSize < BLOCK_8X8 ? BLOCK_8X8 : MiSize.
        let bsize = mi_size.max(BLOCK_8X8);
        let mut eob_total = 0u32;

        for plane in 0..3usize {
            // §6.4.21: txSz = (plane > 0) ? get_uv_tx_size( ) : tx_size.
            let tx_sz = if plane == 0 {
                mi.tx_size
            } else {
                get_uv_tx_size(mi.tx_size, mi_size, self.subsampling_x, self.subsampling_y)
            };
            let step = 1u32 << tx_sz;

            let plane_sz =
                get_plane_block_size(bsize, plane, self.subsampling_x, self.subsampling_y);
            if plane_sz == BLOCK_INVALID {
                // §7.4.3 conformance forbids this combination.
                return Err(Error::InvalidBitstream);
            }
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];

            let sub_x = plane > 0 && self.subsampling_x;
            let sub_y = plane > 0 && self.subsampling_y;
            let base_x = (c * 8) >> u32::from(sub_x);
            let base_y = (r * 8) >> u32::from(sub_y);
            let maxx = (self.mi_cols * 8) >> u32::from(sub_x);
            let maxy = (self.mi_rows * 8) >> u32::from(sub_y);

            let dc_quant = get_dc_quant(
                plane,
                self.seg,
                self.quant,
                mi.segment_id as usize,
                self.bit_depth as u8,
            );
            let ac_quant = get_ac_quant(
                plane,
                self.seg,
                self.quant,
                mi.segment_id as usize,
                self.bit_depth as u8,
            );

            let mut block_idx = 0usize;
            let mut y = 0u32;
            while y < num4x4h {
                let mut x = 0u32;
                while x < num4x4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;

                    if start_x < maxx && start_y < maxy {
                        // §8.5.1 mode selection: uv_mode for chroma,
                        // y_mode for MiSize >= BLOCK_8X8 luma,
                        // sub_modes[ blockIdx ] for sub-8x8 luma.
                        let mode_raw = if plane > 0 {
                            mi.uv_mode
                        } else if mi_size >= BLOCK_8X8 {
                            mi.y_mode
                        } else {
                            mi.sub_modes[block_idx & 3]
                        };
                        let mode = PredMode::from_raw(mode_raw).ok_or(Error::InvalidBitstream)?;

                        // §6.4.21: predict_intra( plane, startX, startY,
                        //   AvailL || x > 0, AvailU || y > 0,
                        //   x + step < num4x4w, txSz, blockIdx ).
                        self.px_intra(IntraOp {
                            plane: plane as u8,
                            x: start_x as u16,
                            y: start_y as u16,
                            have_left: avail_l || x > 0,
                            have_above: avail_u || y > 0,
                            not_on_right: x + step < num4x4w,
                            tx_sz: tx_sz as u8,
                            mode,
                            max_x: (maxx - 1) as u16,
                            max_y: (maxy - 1) as u16,
                        });

                        if !mi.skip {
                            // §6.4.25 TxType selection (intra blocks:
                            // is_inter == 0).
                            let tx_type = if plane > 0 || tx_sz == 3 {
                                crate::idct::DCT_DCT
                            } else if tx_sz == 0 {
                                if self.lossless {
                                    crate::idct::DCT_DCT
                                } else {
                                    tx_type_for_intra(mode)
                                }
                            } else {
                                // txSz >= TX_8X8 implies MiSize >=
                                // BLOCK_8X8, so mode == y_mode here.
                                tx_type_for_intra(mode)
                            };
                            let scan = get_scan(plane, tx_sz, tx_type);
                            let n0 = 1usize << (tx_sz + 2);
                            let seg_eob = n0 * n0;

                            let tbc = TokenBlockCtx {
                                plane,
                                is_inter: false,
                                tx_type,
                                bit_depth: self.bit_depth,
                                x4: (start_x >> 2) as usize,
                                y4: (start_y >> 2) as usize,
                                max_x: ((2 * self.mi_cols) >> u32::from(sub_x)) as usize,
                                max_y: ((2 * self.mi_rows) >> u32::from(sub_y)) as usize,
                            };
                            // No TokenCache[ ] reset: every context
                            // neighbour precedes its position in the scan
                            // (pinned by a tokens.rs unit test), so only
                            // this block's own entries are read.
                            let eob = tokens_eob(
                                coder,
                                &tbc,
                                tx_sz,
                                scan,
                                &self.chdr.coef_probs,
                                &self.nz[plane],
                                &mut self.token_cache[..],
                                &mut self.tok_buf[..seg_eob],
                                self.counts,
                            )?;
                            nonzero = eob > 0;
                            // §6.4.24: EobTotal += nonzero.
                            eob_total += u32::from(nonzero);

                            // §8.6.2 reconstruct( plane, startX, startY,
                            // txSz ) — a no-op on an all-zero residual
                            // (the transform of zeros is zero).
                            if nonzero {
                                self.px_recon(
                                    ReconOp {
                                        plane: plane as u8,
                                        x: start_x as u16,
                                        y: start_y as u16,
                                        tx_sz: tx_sz as u8,
                                        tx_type,
                                        dc_quant,
                                        ac_quant,
                                        coefs: (0, 0),
                                    },
                                    scan,
                                    eob,
                                );
                            }
                        }
                    }

                    // §6.4.21 trailing write-back — fires for every
                    // (x, y) step including off-visible blocks, with
                    // nonzero = 0 for those.
                    let x4 = (start_x >> 2) as usize;
                    let y4 = (start_y >> 2) as usize;
                    for i in 0..step as usize {
                        if x4 + i < self.nz[plane].above.len() {
                            self.nz[plane].above[x4 + i] = u8::from(nonzero);
                        }
                        if y4 + i < self.nz[plane].left.len() {
                            self.nz[plane].left[y4 + i] = u8::from(nonzero);
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

    /// §6.4.4 `decode_block( r, c, subsize )` — the §6.4.11 inter-frame
    /// arm. Decodes the per-block §6.4.11 `inter_frame_mode_info( )`
    /// (which dispatches §6.4.16 inter / §6.4.15 intra), runs the
    /// §6.4.21 residual (with the §8.5.2 `predict_inter( )` arm for inter
    /// blocks), and fans the products into the frame-wide arrays.
    fn decode_block_inter(
        &mut self,
        coder: &mut BoolCoder<'_>,
        r: u32,
        c: u32,
        subsize: u8,
    ) -> Result<(), Error> {
        let avail_u = r > 0;
        let avail_l = c > self.mi_col_start;

        // §6.4.11 prelude neighbour bundles — derived from the frame-wide
        // arrays earlier blocks' §6.4.4 fan-outs populated.
        let above_rf = |rl: usize| -> Option<i32> {
            if avail_u {
                self.state.get_ref_frame(r - 1, c, rl)
            } else {
                None
            }
        };
        let left_rf = |rl: usize| -> Option<i32> {
            if avail_l {
                self.state.get_ref_frame(r, c - 1, rl)
            } else {
                None
            }
        };
        let above_pair = above_rf(0).map(|rf0| (rf0, above_rf(1).unwrap_or(NONE_REF_FRAME)));
        let left_pair = left_rf(0).map(|rf0| (rf0, left_rf(1).unwrap_or(NONE_REF_FRAME)));

        let nb_skip = NeighbourSkips {
            above: if avail_u {
                self.state.get_skip(r - 1, c)
            } else {
                None
            },
            left: if avail_l {
                self.state.get_skip(r, c - 1)
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
                self.state.get_tx_size(r - 1, c).unwrap_or(0) as u32
            } else {
                0
            },
            tx_left: if avail_l {
                self.state.get_tx_size(r, c - 1).unwrap_or(0) as u32
            } else {
                0
            },
        };
        let is_inter_nb = IsInterNeighbours {
            above: above_rf(0),
            left: left_rf(0),
        };
        // §9.3.2 interp_filter neighbours: the InterpFilters cell is only
        // meaningful when that neighbour is inter (RefFrame[0] > INTRA).
        let interp_nb = InterpFilterNeighbours {
            above: above_rf(0).and_then(|rf0| {
                if rf0 > INTRA_FRAME {
                    self.state.get_interp_filter(r - 1, c)
                } else {
                    None
                }
            }),
            left: left_rf(0).and_then(|rf0| {
                if rf0 > INTRA_FRAME {
                    self.state.get_interp_filter(r, c - 1)
                } else {
                    None
                }
            }),
        };
        let ref_nb = RefFrameNeighbours {
            above: above_pair,
            left: left_pair,
        };

        // Snapshot the inter-context scalars the decode reads (avoid
        // holding a borrow of `self.inter` across the `&mut self`
        // mode-info call which also touches `self.state` / planes).
        let inter = self.inter.as_ref().expect("inter ctx on inter frame");
        let interpolation_filter = inter.interpolation_filter;
        let allow_hp_mv = inter.allow_high_precision_mv;
        let use_prev = inter.use_prev_frame_mvs;
        let comp_config = inter.comp_config;
        let reference_mode = inter.chdr.reference_mode;
        let sign_bias = inter.sign_bias;
        let fix_ref_idx = sign_bias[comp_config.fixed_ref as usize] as u8;

        let geom = MvRefGeometry {
            mi_row: r as i32,
            mi_col: c as i32,
            mi_rows: self.mi_rows as i32,
            mi_cols: self.mi_cols as i32,
            mi_size: subsize as usize,
            mi_col_start: self.mi_col_start as i32,
            mi_col_end: self.mi_col_end as i32,
        };

        let prev = if use_prev {
            Some(PrevFrameMvs {
                prev_ref_frames: inter.prev_ref_frames,
                prev_mvs: inter.prev_mvs,
            })
        } else {
            None
        };
        let src = FrameStateMvSource::new(self.state, prev);

        let prev_seg = PrevSegmentIds {
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            data: inter.prev_segment_ids,
        };
        let seg = SegFeatureTables {
            enabled: self.seg.enabled,
            feature_enabled: &self.seg.feature_enabled,
            feature_data: &self.seg.feature_data,
        };
        let chdr_inter = inter.chdr;

        let args = InterFrameModeArgs {
            geom: &geom,
            src: &src,
            seg,
            seg_id: InterSegmentIdArgs {
                update_map: self.seg.update_map,
                temporal_update: self.seg.temporal_update,
                tree_probs: self.seg.tree_probs.as_ref(),
                pred_prob: self.seg.pred_prob.as_ref(),
                prev: prev_seg,
            },
            skip_prob: &chdr_inter.intra.skip_prob,
            skip_nb: nb_skip,
            is_inter_prob: &chdr_inter.is_inter_prob,
            is_inter_nb,
            tx_mode: chdr_inter.intra.tx_mode,
            tx_probs: &chdr_inter.intra.tx_probs,
            tx_nb: nb_tx,
            ref_frame: InterRefFrameArgs {
                // The §6.4.11 driver re-resolves the SEG_LVL_REF_FRAME
                // override against the just-decoded segment_id; these two
                // fields are placeholders it overwrites.
                seg_feature_ref_frame_active: false,
                segment_ref_frame_data: 0,
                reference_mode,
                comp_config,
                fix_ref_idx,
                nb: ref_nb,
                comp_mode_prob: &chdr_inter.comp_mode_prob,
                single_ref_prob: &chdr_inter.single_ref_prob,
                comp_ref_prob: &chdr_inter.comp_ref_prob,
            },
            mv_probs: &chdr_inter.mv_probs,
            inter_mode_probs: &chdr_inter.inter_mode_probs,
            interp_filter_probs: &chdr_inter.interp_filter_probs,
            interp_nb,
            interpolation_filter,
            allow_high_precision_mv: allow_hp_mv,
            use_prev_frame_mvs: use_prev,
            sign_bias: &sign_bias,
            y_mode_probs: &chdr_inter.y_mode_probs,
            uv_mode_probs: &chdr_inter.uv_mode_probs,
        };

        let seg_pred_ctx = &mut self.inter.as_mut().unwrap().seg_pred_ctx;
        let mi = inter_frame_mode_info(coder, args, seg_pred_ctx, r, c, &mut self.counts.noncoef)?;

        // Map the §6.4.11 products into the §6.4.4 fan-out + residual.
        let (ref_frame, is_inter, y_mode, interp_filter, block_mvs, sub_modes, uv_mode) =
            match mi.block {
                Vp9InterFrameBlock::Inter(b) => (
                    [b.ref_frame_0, b.ref_frame_1],
                    true,
                    b.y_mode,
                    b.interp_filter,
                    b.block_mvs,
                    [DC_PRED; 4],
                    DC_PRED,
                ),
                Vp9InterFrameBlock::Intra(b) => (
                    [b.ref_frame_0, b.ref_frame_1],
                    false,
                    b.y_mode,
                    0u8,
                    [[[0i32; 2]; 4]; 2],
                    b.sub_modes,
                    b.uv_mode,
                ),
            };

        // §6.4.21 residual + (inter) §8.5.2 predict_inter.
        let tx_size = mi.tx_size as u8;
        let eob_total = self.residual_inter(
            coder,
            r,
            c,
            subsize,
            tx_size,
            mi.segment_id,
            mi.skip,
            is_inter,
            y_mode,
            uv_mode,
            &sub_modes,
            ref_frame,
            interp_filter,
            &block_mvs,
            avail_u,
            avail_l,
        )?;

        // §6.4.4 line 2405-2407 skip rewrite + fan-out.
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
            skip: mi.skip,
            tx_size: mi.tx_size as u8,
            y_mode,
            segment_id: mi.segment_id,
            ref_frame,
            is_inter,
            eob_total,
            interp_filter,
            block_mvs: block_mvs_i16,
            sub_modes,
        };
        decode_block_apply(self.state, r, c, subsize, &result);
        Ok(())
    }

    /// §6.4.21 `residual( )` — the inter-capable arm. Runs §8.5.2
    /// `predict_inter( )` for inter blocks (before the token loop, per
    /// the §6.4.21 listing) and §8.5.1 `predict_intra( )` for intra
    /// blocks, then the shared §6.4.24 token + §8.6.2 reconstruct loop.
    /// Returns the §6.4.24 `EobTotal` accumulator.
    #[allow(clippy::too_many_arguments)]
    fn residual_inter(
        &mut self,
        coder: &mut BoolCoder<'_>,
        r: u32,
        c: u32,
        mi_size: u8,
        tx_size: u8,
        segment_id: u8,
        skip: bool,
        is_inter: bool,
        y_mode: u8,
        uv_mode: u8,
        sub_modes: &[u8; 4],
        ref_frame: [i32; 2],
        interp_filter: u8,
        block_mvs: &[[[i32; 2]; 4]; 2],
        avail_u: bool,
        avail_l: bool,
    ) -> Result<u32, Error> {
        let bsize = mi_size.max(BLOCK_8X8);
        let mut eob_total = 0u32;
        let segment_id = segment_id as usize;
        let is_compound = ref_frame[1] > INTRA_FRAME;
        let tx_size = tx_size as u32;

        for plane in 0..3usize {
            let tx_sz = if plane == 0 {
                tx_size
            } else {
                get_uv_tx_size(tx_size, mi_size, self.subsampling_x, self.subsampling_y)
            };
            let step = 1u32 << tx_sz;

            let plane_sz =
                get_plane_block_size(bsize, plane, self.subsampling_x, self.subsampling_y);
            if plane_sz == BLOCK_INVALID {
                return Err(Error::InvalidBitstream);
            }
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];

            let sub_x = plane > 0 && self.subsampling_x;
            let sub_y = plane > 0 && self.subsampling_y;
            let base_x = (c * 8) >> u32::from(sub_x);
            let base_y = (r * 8) >> u32::from(sub_y);
            let maxx = (self.mi_cols * 8) >> u32::from(sub_x);
            let maxy = (self.mi_rows * 8) >> u32::from(sub_y);

            // §6.4.21: for inter blocks, predict the whole plane region
            // BEFORE the token/reconstruct loop. Sub-8x8 blocks predict
            // each 4x4 sub-block; >= BLOCK_8X8 predicts the whole plane.
            if is_inter {
                if mi_size < BLOCK_8X8 {
                    for y in 0..num4x4h {
                        for x in 0..num4x4w {
                            self.predict_inter_region(
                                plane,
                                (base_x + 4 * x) as i32,
                                (base_y + 4 * y) as i32,
                                4,
                                4,
                                (y * num4x4w + x) as usize,
                                r,
                                c,
                                mi_size,
                                ref_frame,
                                interp_filter,
                                is_compound,
                                block_mvs,
                            );
                        }
                    }
                } else {
                    self.predict_inter_region(
                        plane,
                        base_x as i32,
                        base_y as i32,
                        (num4x4w * 4) as usize,
                        (num4x4h * 4) as usize,
                        0,
                        r,
                        c,
                        mi_size,
                        ref_frame,
                        interp_filter,
                        is_compound,
                        block_mvs,
                    );
                }
            }

            let dc_quant = get_dc_quant(
                plane,
                self.seg,
                self.quant,
                segment_id,
                self.bit_depth as u8,
            );
            let ac_quant = get_ac_quant(
                plane,
                self.seg,
                self.quant,
                segment_id,
                self.bit_depth as u8,
            );

            let mut block_idx = 0usize;
            let mut y = 0u32;
            while y < num4x4h {
                let mut x = 0u32;
                while x < num4x4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;

                    if start_x < maxx && start_y < maxy {
                        if !is_inter {
                            let mode_raw = if plane > 0 {
                                uv_mode
                            } else if mi_size >= BLOCK_8X8 {
                                y_mode
                            } else {
                                sub_modes[block_idx & 3]
                            };
                            let mode =
                                PredMode::from_raw(mode_raw).ok_or(Error::InvalidBitstream)?;
                            self.px_intra(IntraOp {
                                plane: plane as u8,
                                x: start_x as u16,
                                y: start_y as u16,
                                have_left: avail_l || x > 0,
                                have_above: avail_u || y > 0,
                                not_on_right: x + step < num4x4w,
                                tx_sz: tx_sz as u8,
                                mode,
                                max_x: (maxx - 1) as u16,
                                max_y: (maxy - 1) as u16,
                            });
                        }

                        if !skip {
                            // §6.4.25 TxType: inter blocks always DCT_DCT;
                            // chroma / TX_32X32 / lossless force DCT_DCT;
                            // other intra luma blocks follow the §6.4.25 map.
                            let tx_type = if is_inter || plane > 0 || tx_sz == 3 || self.lossless {
                                crate::idct::DCT_DCT
                            } else {
                                let mode_raw = if mi_size >= BLOCK_8X8 {
                                    y_mode
                                } else {
                                    sub_modes[block_idx & 3]
                                };
                                let mode =
                                    PredMode::from_raw(mode_raw).ok_or(Error::InvalidBitstream)?;
                                tx_type_for_intra(mode)
                            };
                            let scan = get_scan(plane, tx_sz, tx_type);
                            let n0 = 1usize << (tx_sz + 2);
                            let seg_eob = n0 * n0;

                            let tbc = TokenBlockCtx {
                                plane,
                                is_inter,
                                tx_type,
                                bit_depth: self.bit_depth,
                                x4: (start_x >> 2) as usize,
                                y4: (start_y >> 2) as usize,
                                max_x: ((2 * self.mi_cols) >> u32::from(sub_x)) as usize,
                                max_y: ((2 * self.mi_rows) >> u32::from(sub_y)) as usize,
                            };
                            // No TokenCache[ ] reset: every context
                            // neighbour precedes its position in the scan
                            // (pinned by a tokens.rs unit test), so only
                            // this block's own entries are read.
                            let eob = tokens_eob(
                                coder,
                                &tbc,
                                tx_sz,
                                scan,
                                &self.chdr.coef_probs,
                                &self.nz[plane],
                                &mut self.token_cache[..],
                                &mut self.tok_buf[..seg_eob],
                                self.counts,
                            )?;
                            nonzero = eob > 0;
                            eob_total += u32::from(nonzero);

                            // An all-zero residual (no token read) leaves
                            // the prediction unchanged: the §8.7 transform
                            // of zeros is zero and Clip1 of an in-range
                            // sample is the sample.
                            if nonzero {
                                self.px_recon(
                                    ReconOp {
                                        plane: plane as u8,
                                        x: start_x as u16,
                                        y: start_y as u16,
                                        tx_sz: tx_sz as u8,
                                        tx_type,
                                        dc_quant,
                                        ac_quant,
                                        coefs: (0, 0),
                                    },
                                    scan,
                                    eob,
                                );
                            }
                        }
                    }

                    let x4 = (start_x >> 2) as usize;
                    let y4 = (start_y >> 2) as usize;
                    for i in 0..step as usize {
                        if x4 + i < self.nz[plane].above.len() {
                            self.nz[plane].above[x4 + i] = u8::from(nonzero);
                        }
                        if y4 + i < self.nz[plane].left.len() {
                            self.nz[plane].left[y4 + i] = u8::from(nonzero);
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

    /// §8.5.2 `predict_inter( plane, x, y, w, h, blockIdx )` for one
    /// region — run now or logged (see [`Sink`]).
    #[allow(clippy::too_many_arguments)]
    fn predict_inter_region(
        &mut self,
        plane: usize,
        x: i32,
        y: i32,
        w: usize,
        h: usize,
        block_idx: usize,
        r: u32,
        c: u32,
        mi_size: u8,
        ref_frame: [i32; 2],
        interp_filter: u8,
        is_compound: bool,
        block_mvs: &[[[i32; 2]; 4]; 2],
    ) {
        let op = InterOp {
            plane: plane as u8,
            x: x as u16,
            y: y as u16,
            w: w as u8,
            h: h as u8,
            block_idx: block_idx as u8,
            r,
            c,
            mi_size,
            ref_frame: [ref_frame[0] as i8, ref_frame[1] as i8],
            interp_filter,
            is_compound,
            block_mvs: *block_mvs,
        };
        match &mut self.sink {
            Sink::Log(log) => {
                log.ops.push(PixelOp::Inter(log.inters.len() as u32));
                log.inters.push(op);
            }
            Sink::Now { y, u, v, pred } => {
                let inter = self.inter.as_ref().expect("inter ctx");
                let ctx = PixelCtx {
                    refs: Some(inter.refs),
                    ref_frame_idx: inter.ref_frame_idx,
                    frame_width: inter.frame_width,
                    frame_height: inter.frame_height,
                    mi_rows: self.mi_rows,
                    mi_cols: self.mi_cols,
                    subsampling_x: self.subsampling_x,
                    subsampling_y: self.subsampling_y,
                    bit_depth: self.bit_depth,
                    lossless: self.lossless,
                };
                let dst: &mut Plane = match plane {
                    0 => y,
                    1 => u,
                    _ => v,
                };
                ctx.inter(dst, pred, &op);
            }
        }
    }

    /// §8.5.1 `predict_intra( )` for one transform block — run now or
    /// logged.
    fn px_intra(&mut self, op: IntraOp) {
        match &mut self.sink {
            Sink::Log(log) => log.ops.push(PixelOp::Intra(op)),
            Sink::Now { y, u, v, .. } => {
                let dst: &mut Plane = match op.plane {
                    0 => y,
                    1 => u,
                    _ => v,
                };
                intra_op(dst, &op, self.bit_depth);
            }
        }
    }

    /// §8.6.2 `reconstruct( )` of one transform block whose `Tokens[ ]`
    /// (the first `eob` positions of `scan`) are in `tok_buf` — run now
    /// or logged with its coefficients; either way `tok_buf` is left
    /// all-zero.
    fn px_recon(&mut self, mut op: ReconOp, scan: &[u16], eob: usize) {
        let n0 = 1usize << (op.tx_sz + 2);
        match &mut self.sink {
            Sink::Log(log) => {
                let start = log.coefs.len();
                for &p in &scan[..eob] {
                    let v = self.tok_buf[p as usize];
                    if v != 0 {
                        log.coefs.push((p, v as i32));
                        self.tok_buf[p as usize] = 0;
                    }
                }
                op.coefs = (start as u32, (log.coefs.len() - start) as u32);
                log.ops.push(PixelOp::Recon(op));
            }
            Sink::Now { y, u, v, .. } => {
                let dst: &mut Plane = match op.plane {
                    0 => y,
                    1 => u,
                    _ => v,
                };
                reconstruct_block(
                    dst,
                    usize::from(op.x),
                    usize::from(op.y),
                    u32::from(op.tx_sz),
                    &self.tok_buf[..n0 * n0],
                    op.dc_quant,
                    op.ac_quant,
                    op.tx_type,
                    self.lossless,
                    self.bit_depth,
                );
                // Keep Tokens[ ] all-zero between blocks.
                for &p in &scan[..eob] {
                    self.tok_buf[p as usize] = 0;
                }
            }
        }
    }
}

/// Where a [`BlockDecoder`]'s pixel work goes: straight into the working
/// planes, or into a [`ReconLog`] replayed later (in the same order, by
/// [`replay`]) — which lets a tile's entropy decode run ahead of its
/// reconstruction. Either way the same §8.5 / §8.6 operations run in the
/// same order on the same inputs.
enum Sink<'a> {
    /// Straight into the planes (the block-level test helpers).
    #[cfg_attr(not(test), allow(dead_code))]
    Now {
        y: &'a mut Plane,
        u: &'a mut Plane,
        v: &'a mut Plane,
        pred: &'a mut crate::inter_pred::PredScratch,
    },
    Log(&'a mut ReconLog),
}

/// §8.5.1 intra prediction of one transform block.
#[derive(Clone, Copy, Debug)]
struct IntraOp {
    plane: u8,
    x: u16,
    y: u16,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    tx_sz: u8,
    mode: PredMode,
    max_x: u16,
    max_y: u16,
}

/// §8.6.2 reconstruction of one transform block with non-zero tokens
/// (`coefs`: its `(position, token)` pairs in the log's arena).
#[derive(Clone, Copy, Debug)]
struct ReconOp {
    plane: u8,
    x: u16,
    y: u16,
    tx_sz: u8,
    tx_type: u8,
    dc_quant: i32,
    ac_quant: i32,
    coefs: (u32, u32),
}

/// §8.5.2 inter prediction of one region.
#[derive(Clone, Copy, Debug)]
struct InterOp {
    plane: u8,
    x: u16,
    y: u16,
    w: u8,
    h: u8,
    block_idx: u8,
    r: u32,
    c: u32,
    mi_size: u8,
    ref_frame: [i8; 2],
    interp_filter: u8,
    is_compound: bool,
    block_mvs: [[[i32; 2]; 4]; 2],
}

#[derive(Debug)]
enum PixelOp {
    Intra(IntraOp),
    Recon(ReconOp),
    /// Index into [`ReconLog::inters`].
    Inter(u32),
}

/// One superblock row's logged pixel work.
#[derive(Debug, Default)]
pub(crate) struct ReconLog {
    ops: Vec<PixelOp>,
    coefs: Vec<(u16, i32)>,
    /// The (large) inter operations, out of line.
    inters: Vec<InterOp>,
}

impl ReconLog {
    fn clear(&mut self) {
        self.ops.clear();
        self.coefs.clear();
        self.inters.clear();
    }
}

/// The frame-level inputs the pixel operations read.
struct PixelCtx<'a> {
    refs: Option<&'a RefBuffers>,
    ref_frame_idx: [u8; REFS_PER_FRAME],
    frame_width: u32,
    frame_height: u32,
    mi_rows: u32,
    mi_cols: u32,
    subsampling_x: bool,
    subsampling_y: bool,
    bit_depth: u32,
    lossless: bool,
}

fn intra_op(dst: &mut Plane, op: &IntraOp, bit_depth: u32) {
    predict_intra(
        dst,
        usize::from(op.x),
        usize::from(op.y),
        op.have_left,
        op.have_above,
        op.not_on_right,
        u32::from(op.tx_sz),
        op.mode,
        usize::from(op.max_x),
        usize::from(op.max_y),
        bit_depth,
    );
}

impl PixelCtx<'_> {
    /// §8.5.2 `predict_inter( )` of `op` into `dst`.
    fn inter(&self, dst: &mut Plane, pred: &mut crate::inter_pred::PredScratch, op: &InterOp) {
        let plane = usize::from(op.plane);
        let ref_frame = [i32::from(op.ref_frame[0]), i32::from(op.ref_frame[1])];
        let refs = build_ref_planes(
            self.refs.expect("inter frame references"),
            ref_frame,
            &self.ref_frame_idx,
            plane,
        );
        let grid = BlockGrid {
            mi_row: op.r as i32,
            mi_col: op.c as i32,
            mi_rows: self.mi_rows as i32,
            mi_cols: self.mi_cols as i32,
            mi_size: op.mi_size,
        };
        let geom = ScaleGeom {
            // ref_frame_width/height are per-list and overridden inside
            // the driver from each RefPlane; these frame-level dims feed
            // the §8.5.2.3 scale ratio numerator.
            ref_frame_width: self.frame_width as i32,
            ref_frame_height: self.frame_height as i32,
            frame_width: self.frame_width as i32,
            frame_height: self.frame_height as i32,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
        };
        let args = InterPredArgs {
            plane,
            x: i32::from(op.x),
            y: i32::from(op.y),
            w: usize::from(op.w),
            h: usize::from(op.h),
            block_idx: usize::from(op.block_idx),
            interp_filter: usize::from(op.interp_filter),
            bit_depth: self.bit_depth,
            is_compound: op.is_compound,
        };
        crate::inter_pred::predict_inter_with(
            dst,
            &args,
            &grid,
            &geom,
            &op.block_mvs,
            &refs,
            self.subsampling_x,
            self.subsampling_y,
            pred,
        );
    }
}

/// Replay a logged superblock row's pixel work into the tile's planes
/// (`tok_buf` all-zero on entry and exit).
fn replay(
    log: &ReconLog,
    ctx: &PixelCtx<'_>,
    planes: [&mut Plane; 3],
    pred: &mut crate::inter_pred::PredScratch,
    tok_buf: &mut [i64; 1024],
) {
    let [y, u, v] = planes;
    for op in &log.ops {
        match op {
            PixelOp::Intra(op) => {
                let dst: &mut Plane = match op.plane {
                    0 => &mut *y,
                    1 => &mut *u,
                    _ => &mut *v,
                };
                intra_op(dst, op, ctx.bit_depth);
            }
            PixelOp::Inter(idx) => {
                let op = &log.inters[*idx as usize];
                let dst: &mut Plane = match op.plane {
                    0 => &mut *y,
                    1 => &mut *u,
                    _ => &mut *v,
                };
                ctx.inter(dst, pred, op);
            }
            PixelOp::Recon(op) => {
                let dst: &mut Plane = match op.plane {
                    0 => &mut *y,
                    1 => &mut *u,
                    _ => &mut *v,
                };
                let n0 = 1usize << (op.tx_sz + 2);
                let (start, len) = (op.coefs.0 as usize, op.coefs.1 as usize);
                let coefs = &log.coefs[start..start + len];
                for &(p, val) in coefs {
                    tok_buf[p as usize] = i64::from(val);
                }
                reconstruct_block(
                    dst,
                    usize::from(op.x),
                    usize::from(op.y),
                    u32::from(op.tx_sz),
                    &tok_buf[..n0 * n0],
                    op.dc_quant,
                    op.ac_quant,
                    op.tx_type,
                    ctx.lossless,
                    ctx.bit_depth,
                );
                for &(p, _) in coefs {
                    tok_buf[p as usize] = 0;
                }
            }
        }
    }
}

/// One frame's reconstruction products: the decoded (loop-filtered)
/// frame and the §6.4.4 state the next frame's §6.5 prev-MV scan reads.
struct FrameDecodeProducts {
    /// The MI-aligned decoded planes + geometry (shared by the §8.10
    /// reference slots and the output).
    frame: Arc<FrameBuf>,
    /// The §9.3.4 syntax-element count bank accumulated over the frame,
    /// consumed by the §6.1.2 `refresh_probs( )` backward adaptation.
    #[cfg_attr(not(test), allow(dead_code))]
    counts: Box<crate::prob_adapt::FrameCounts>,
}

/// The §6.3 compressed-header product threaded into [`launch_single_frame`]:
/// either the intra-only header or the full inter header.
enum ChdrKind {
    Intra(Box<Vp9CompressedHeader>),
    Inter(Box<Vp9CompressedHeaderInter>),
}

/// The read-only per-frame inputs every §6.4.2 `decode_tile( )` call
/// shares — the parsed headers and the §8.10 reference buffers (the
/// previous frame's §6.5 / §6.4.14 snapshots arrive per superblock row,
/// see `PrevRows`). Nothing in here is written during the §6.4 tile
/// walk.
struct TileDecodeShared<'a> {
    hdr: &'a Vp9FrameHeader,
    chdr: &'a Vp9CompressedHeader,
    chdr_inter: Option<&'a Vp9CompressedHeaderInter>,
    refs: Option<&'a RefBuffers>,
    mi_rows: u32,
    mi_cols: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    lossless: bool,
    use_prev_frame_mvs: bool,
}

/// One `(tileRow, tileCol)` cell of the §6.4 tile grid: its
/// `get_tile_offset( )` MI window and its §9.2 coder byte slice (the
/// pure byte-arithmetic prepass over the `tile_payload_sizes( )`
/// output).
struct TileTask<'a> {
    mi_row_start: u32,
    mi_row_end: u32,
    mi_col_start: u32,
    mi_col_end: u32,
    slice: &'a [u8],
}

/// The mutable per-frame decode state one §6.4.2 `decode_tile( )` call
/// writes — the `CurrFrame` planes, the §6.4.4 frame-wide arrays, the
/// §7.4.1 context strips, the §9.3.4 count bank, and the §6.4.24
/// scratch buffers. The serial path reborrows the frame-level state per
/// tile; the tile-parallel path hands each worker its own private
/// instance (a tile column never reads another column's slice of any of
/// these — `AvailL = MiCol > MiColStart`, §8.5.1 edge reads bounded by
/// the block's own extent, §6.5 candidate scans clamped to the tile
/// window — so a private zero-initialised copy evolves exactly like the
/// shared serial state over the worker's columns).
struct TileWorkState<'a> {
    state: &'a mut Vp9FrameState,
    /// The log the tile's pixel work goes to (see [`Sink`]).
    log: &'a mut ReconLog,
    nz: &'a mut [NonzeroContext; 3],
    pctx: &'a mut PartitionContextState,
    seg_pred_ctx: &'a mut SegPredContextState,
    counts: &'a mut crate::prob_adapt::FrameCounts,
    token_cache: &'a mut [u8; 1024],
    tok_buf: &'a mut [i64; 1024],
}

/// The previous frame's §6.5 motion field and the §6.4.14
/// `PrevSegmentIds[ ][ ]` map as one tile column reads them: frame-wide
/// arrays (the layout the §6.5 / §6.4.14 readers index) filled one
/// superblock row at a time, right before that row decodes, from the
/// source frames' frozen mode-info rows — so a frame can start while
/// the frame before it is still decoding. Every read of a superblock
/// row's blocks stays inside that row (§6.5 reads the co-located
/// `PrevMvs[ MiRow ][ MiCol ]`; §6.4.14 the block's own extent).
pub(crate) struct PrevRows {
    ref_frames: Vec<i32>,
    mvs: Vec<(i16, i16)>,
    seg: Vec<u8>,
}

impl PrevRows {
    fn new() -> Self {
        Self {
            ref_frames: Vec::new(),
            mvs: Vec::new(),
            seg: Vec::new(),
        }
    }

    /// Size the arrays for a `mi_rows x mi_cols` frame (contents are
    /// rewritten row by row before use).
    fn fit(&mut self, mi_rows: u32, mi_cols: u32) {
        let n = (mi_rows * mi_cols) as usize;
        self.ref_frames.resize(2 * n, 0);
        self.mvs.resize(2 * n, (0, 0));
        self.seg.resize(n, 0);
    }

    /// Fill superblock row `sb` (MI rows `8 * sb ..`) from `mv_src` /
    /// `seg_src` (zero when absent), waiting for their rows to freeze.
    fn import(
        &mut self,
        sb: usize,
        mv_src: Option<&FrameBuf>,
        seg_src: Option<&FrameBuf>,
        mi_rows: u32,
        mi_cols: u32,
    ) -> Result<(), Error> {
        let cols = mi_cols as usize;
        let r0 = sb * 8;
        let r1 = (r0 + 8).min(mi_rows as usize);
        let n = (r1 - r0) * cols;
        let at = r0 * cols;
        match mv_src.map(|f| f.mi_row(sb)) {
            Some(None) => return Err(Error::InvalidBitstream),
            Some(Some(row)) => {
                for i in 0..n {
                    self.ref_frames[2 * (at + i)] = row.ref0[i];
                    self.ref_frames[2 * (at + i) + 1] = row.ref1[i];
                    self.mvs[2 * (at + i)] = row.mv0[i];
                    self.mvs[2 * (at + i) + 1] = row.mv1[i];
                }
            }
            None => {
                self.ref_frames[2 * at..2 * (at + n)].fill(INTRA_FRAME);
                self.mvs[2 * at..2 * (at + n)].fill((0, 0));
            }
        }
        match seg_src.map(|f| f.mi_row(sb)) {
            Some(None) => return Err(Error::InvalidBitstream),
            Some(Some(row)) => self.seg[at..at + n].copy_from_slice(&row.seg_ids[..n]),
            None => self.seg[at..at + n].fill(0),
        }
        Ok(())
    }
}

/// §6.4.2 `decode_tile( )` with its §6.4 line-2326/2328 coder bracket:
/// `init_bool( tile_size )`, the superblock raster with the §7.4.2
/// per-superblock-row left resets, `exit_bool( )`. Every tile — serial
/// or concurrent — runs through this one body.
///
/// `before_row( prev, r )` runs before each superblock row (MI row `r`)
/// and fills that row's previous-frame snapshots into `prev` (see
/// `PrevRows`); `on_row( r, y, u, v, state )` runs after it, with the
/// tile's planes and state.
#[allow(clippy::type_complexity)]
fn decode_one_tile(
    sh: &TileDecodeShared<'_>,
    task: &TileTask<'_>,
    ws: TileWorkState<'_>,
    prev: &mut PrevRows,
    before_row: &dyn Fn(&mut PrevRows, u32) -> Result<(), Error>,
    on_row: &mut dyn FnMut(u32, &mut ReconLog, &Vp9FrameState),
) -> Result<(), Error> {
    // §6.4 line 2326: init_bool( tile_size ).
    let mut coder = BoolCoder::init_bool(task.slice, task.slice.len())?;

    // §6.4.3 partition probs: keyframe fixed table or the inter
    // running `partition_probs[ ]`.
    let part_kind = match sh.chdr_inter {
        Some(ci) => PartitionProbsKind::Inter(&ci.partition_probs),
        None => PartitionProbsKind::Keyframe,
    };
    let TileWorkState {
        state,
        log,
        nz,
        pctx,
        seg_pred_ctx,
        counts,
        token_cache,
        tok_buf,
    } = ws;

    // §6.4.2 decode_tile( ): superblock raster with the
    // §7.4.2 clear_left_context( ) reset per superblock row
    // (LeftNonzeroContext + LeftPartitionContext).
    let mut r = task.mi_row_start;
    while r < task.mi_row_end {
        before_row(prev, r)?;
        let prev = &*prev;
        // The per-row inter context (re-borrowing the per-worker
        // `seg_pred_ctx`). On keyframes / intra-only frames this is
        // `None` and the §6.4.6 intra arm runs.
        let inter_ctx = match (sh.chdr_inter, sh.refs) {
            (Some(ci), Some(rb)) => {
                let comp_config = ci
                    .compound_reference_config
                    .unwrap_or(CompoundReferenceConfig {
                        fixed_ref: LAST_FRAME,
                        var_ref: [GOLDEN_FRAME, ALTREF_FRAME],
                    });
                // §6.2.5 ref_frame_sign_bias[ 4 ] indexed by §3 value.
                let mut sb = [false; 4];
                let hdr_sb = sh.hdr.ref_frame_sign_bias;
                sb[LAST_FRAME as usize] = hdr_sb[0];
                sb[GOLDEN_FRAME as usize] = hdr_sb[1];
                sb[ALTREF_FRAME as usize] = hdr_sb[2];
                Some(InterCtx {
                    chdr: ci,
                    refs: rb,
                    ref_frame_idx: sh.hdr.ref_frame_idx.unwrap_or([0, 1, 2]),
                    sign_bias: sb,
                    interpolation_filter: sh.hdr.interpolation_filter,
                    allow_high_precision_mv: sh.hdr.allow_high_precision_mv,
                    use_prev_frame_mvs: sh.use_prev_frame_mvs,
                    comp_config,
                    frame_width: sh.hdr.frame_width,
                    frame_height: sh.hdr.frame_height,
                    prev_ref_frames: &prev.ref_frames,
                    prev_mvs: &prev.mvs,
                    prev_segment_ids: &prev.seg,
                    seg_pred_ctx: &mut *seg_pred_ctx,
                })
            }
            _ => None,
        };

        {
            let mut block_decoder = BlockDecoder {
                seg: &sh.hdr.segmentation,
                quant: &sh.hdr.quantization,
                chdr: sh.chdr,
                mi_rows: sh.mi_rows,
                mi_cols: sh.mi_cols,
                mi_col_start: task.mi_col_start,
                mi_col_end: task.mi_col_end,
                subsampling_x: sh.ssx,
                subsampling_y: sh.ssy,
                bit_depth: sh.bit_depth,
                lossless: sh.lossless,
                state: &mut *state,
                sink: Sink::Log(&mut *log),
                nz: &mut *nz,
                token_cache: &mut *token_cache,
                tok_buf: &mut *tok_buf,
                counts: &mut *counts,
                inter: inter_ctx,
            };

            pctx.clear_left();
            if let Some(ic) = block_decoder.inter.as_mut() {
                ic.seg_pred_ctx.clear_left();
            }
            for plane_nz in block_decoder.nz.iter_mut() {
                plane_nz.left.fill(0);
            }
            let mut c = task.mi_col_start;
            while c < task.mi_col_end {
                decode_partition(
                    &mut coder,
                    r,
                    c,
                    BLOCK_64X64,
                    sh.mi_rows,
                    sh.mi_cols,
                    pctx,
                    part_kind,
                    &mut block_decoder,
                )?;
                c += 8;
            }
        }
        on_row(r, log, state);
        r += 8;
    }

    // §6.4 line 2328: exit_bool( ).
    coder.exit_bool()?;
    Ok(())
}

// ───────────────────────── frame pipeline ─────────────────────────
//
// A frame decodes as a batch of jobs on the decoder's persistent
// [`WorkerPool`] (or on the caller, in the same order, without helper
// threads):
//
// 1. one job per §6.4 tile *column* (every tile row of it, in row
//    order), decoding into the worker's private working planes and
//    §6.4.4 state — tile columns are independent by construction
//    (§6.4.4 `AvailL = MiCol > MiColStart`, SB64-aligned tile edges,
//    §8.5.1 edge reads bounded by the block, §6.5 candidate scans
//    clamped to the tile window), so each column evolves exactly as in
//    the serial raster. After each superblock row the job copies its
//    reconstructed rows into that row's band and its mode info into the
//    row's mode-info record, then publishes the row; the last column to
//    publish a row freezes its mode info into the frame;
// 2. one job per superblock row for the §8.8 loop filter, which starts
//    as soon as every tile column has published the row and runs two
//    superblocks behind the row above (see [`loop_filter_row_job`] for
//    why that wavefront reproduces the raster), freezing each band once
//    nothing can change it any more. Intra prediction reads the tile's
//    private (unfiltered) planes, never the bands, so filtering overlaps
//    decoding.
//
// Consecutive frames overlap too: a frame's batch may be submitted
// while earlier ones still run. Its tile rows read the reference frames
// and the previous frame's motion field / segment map through the
// frozen rows of those frames, blocking only on rows not frozen yet
// (`FrameBuf::row`, `PrevRows::import`); every such wait is on a job
// queued earlier, which the pool's FIFO order keeps deadlock-free.
//
// The bytes do not depend on the schedule: every band and every slot is
// written by exactly one job, merges are by disjoint column ranges, and
// the §9.3.4 counts are summed.

/// One tile's MI window and byte range inside the frame payload.
#[derive(Clone, Copy, Debug)]
struct TileSpan {
    mi_row_start: u32,
    mi_row_end: u32,
    mi_col_start: u32,
    mi_col_end: u32,
    start: usize,
    len: usize,
}

/// The owned per-frame inputs the jobs share.
struct FrameJob {
    data: Arc<[u8]>,
    hdr: Vp9FrameHeader,
    chdr: Vp9CompressedHeader,
    chdr_inter: Option<Vp9CompressedHeaderInter>,
    refs: Option<RefBuffers>,
    /// The §6.5 previous-frame motion field source (`None` = zeros).
    prev_mvs: Option<Arc<FrameBuf>>,
    /// The §6.4.14 `PrevSegmentIds[ ][ ]` source (`None` = zeros).
    prev_seg: Option<Arc<FrameBuf>>,
    use_prev_frame_mvs: bool,
    tiles: Vec<TileSpan>,
    tile_cols: usize,
    mi_rows: u32,
    mi_cols: u32,
    sb64_rows: u32,
    sb64_cols: u32,
    ssx: bool,
    ssy: bool,
    bit_depth: u32,
    lossless: bool,
    /// Per plane: (width = stride, height, band shift, log2 SB width).
    geom: [(usize, usize, u32, u32); 3],
    /// `Some` when the §8.8 loop filter runs (`loop_filter_level != 0`).
    lf: Option<LfParams>,
}

impl FrameJob {
    fn shared(&self) -> TileDecodeShared<'_> {
        TileDecodeShared {
            hdr: &self.hdr,
            chdr: &self.chdr,
            chdr_inter: self.chdr_inter.as_ref(),
            refs: self.refs.as_ref(),
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            ssx: self.ssx,
            ssy: self.ssy,
            bit_depth: self.bit_depth,
            lossless: self.lossless,
            use_prev_frame_mvs: self.use_prev_frame_mvs,
        }
    }

    fn task(&self, i: usize) -> TileTask<'_> {
        let t = &self.tiles[i];
        TileTask {
            mi_row_start: t.mi_row_start,
            mi_row_end: t.mi_row_end,
            mi_col_start: t.mi_col_start,
            mi_col_end: t.mi_col_end,
            slice: &self.data[t.start..t.start + t.len],
        }
    }
}

/// What the §8.8 loop-filter row jobs need to know about the frame.
struct LfParams {
    /// Per plane: (width = stride, height, band shift, log2 SB width).
    geom: [(usize, usize, u32, u32); 3],
    sb_rows: usize,
    sb_cols: usize,
    mi_rows: u32,
    mi_cols: u32,
    ssx: bool,
    ssy: bool,
    sharpness: u8,
    bit_depth: u8,
    lvl_lookup: crate::loop_filter::LvlLookup,
    /// Publications (tile columns) a superblock row needs before it is
    /// filtered.
    ready_at: usize,
}

struct Progress {
    /// Tile columns that have published each superblock row's pixels /
    /// mode info.
    tiles_done: Vec<usize>,
    mi_done: Vec<usize>,
    /// Halo slots each loop-filter row has published.
    lf_published: Vec<usize>,
    abort: bool,
}

/// A finished tile column.
struct TileOut {
    counts: Box<crate::prob_adapt::FrameCounts>,
}

/// A tile column's outcome: its counts, or its first error with the
/// tile index.
type TileResult = Result<TileOut, (usize, Error)>;

/// Recycled band storage for the next frame (one entry per band).
pub(crate) type Bands = Vec<Option<Band>>;

/// The shared mutable state of one frame's batch.
struct FramePipe {
    /// The frame being built: rows are frozen into it as they finish.
    frame: Arc<FrameBuf>,
    /// Per superblock row: the three planes' working bands (`HALO`
    /// scratch rows then the band's own rows), until frozen.
    bands: Vec<Mutex<Band>>,
    /// Per superblock row: the mode info the tile columns publish,
    /// until frozen.
    mi_build: Vec<Mutex<MiRow>>,
    /// `slots[ r * Sb64Cols + c ]`: the bottom `HALO` rows of superblock
    /// `(r, c)` per plane, as loop-filter row `r` leaves them.
    slots: Vec<OnceLock<[Vec<u16>; 3]>>,
    progress: Mutex<Progress>,
    /// `changed[ r ]`: signalled when loop-filter row `r`'s inputs move
    /// (a tile column published row `r`, or row `r - 1` published a
    /// slot), so each waiter wakes only for its own row.
    changed: Vec<Condvar>,
    /// Per tile column: its products, or its first error (with the tile
    /// index, for the deterministic raster-order error surface).
    tile_out: Mutex<Vec<Option<TileResult>>>,
    /// Signalled when a tile column finishes.
    tiles_finished: Condvar,
    /// Per tile column: logged rows from its entropy decode to its
    /// reconstruction.
    row_queues: Vec<RowQueue>,
    /// The next superblock row a loop-filter job claims.
    next_lf_row: AtomicUsize,
}

impl FramePipe {
    fn progress(&self) -> std::sync::MutexGuard<'_, Progress> {
        self.progress.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Block (on row `r`'s condvar) until `ready` holds; `false` when
    /// the batch is aborting.
    fn wait(&self, r: usize, ready: impl Fn(&Progress) -> bool) -> bool {
        let mut p = self.progress();
        loop {
            if p.abort {
                return false;
            }
            if ready(&p) {
                return true;
            }
            p = self.changed[r].wait(p).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn abort(&self) {
        self.progress().abort = true;
        for cv in &self.changed {
            cv.notify_all();
        }
        self.frame.fail();
    }

    /// Wake row `r`'s waiter (if that row exists).
    fn notify(&self, r: usize) {
        if let Some(cv) = self.changed.get(r) {
            cv.notify_all();
        }
    }

    /// Block until every tile column has finished; their outcomes.
    fn wait_tiles(&self) -> std::sync::MutexGuard<'_, Vec<Option<TileResult>>> {
        let mut t = lock(&self.tile_out);
        while t.iter().any(Option::is_none) {
            t = self
                .tiles_finished
                .wait(t)
                .unwrap_or_else(|e| e.into_inner());
        }
        t
    }
}

/// Releases a pipeline's waiters if the job holding it panics (the
/// pool re-raises the panic on the caller).
struct AbortOnPanic<'a>(&'a FramePipe);

impl Drop for AbortOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.abort();
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A tile column's reusable entropy-decode storage (kept by the
/// sequence decoder across frames). The token buffer is all-zero between
/// blocks, so a scratch that saw a decode error is dropped, not
/// returned.
pub(crate) struct ParseScratch {
    token_cache: Box<[u8; 1024]>,
    tok_buf: Box<[i64; 1024]>,
    /// The §6.4.4 state allocation (zeroed before use).
    state: Option<Vp9FrameState>,
    prev: PrevRows,
}

/// A tile column's reusable reconstruction storage: the working planes,
/// the prediction scratch and the (all-zero between blocks) dequant
/// buffer.
pub(crate) struct ReconScratch {
    y: Plane,
    u: Plane,
    v: Plane,
    tok_buf: Box<[i64; 1024]>,
    pred: Box<crate::inter_pred::PredScratch>,
}

/// The decoder's pools of tile-column storage.
#[derive(Default)]
pub(crate) struct Scratches {
    parse: Vec<ParseScratch>,
    recon: Vec<ReconScratch>,
    logs: Vec<ReconLog>,
}

/// The decoder's tile-column storage, shared with the jobs.
pub(crate) type ScratchPool = Arc<Mutex<Scratches>>;

fn take_parse_scratch(pool: &ScratchPool) -> ParseScratch {
    lock(pool).parse.pop().unwrap_or_else(|| ParseScratch {
        token_cache: Box::new([0; 1024]),
        tok_buf: Box::new([0; 1024]),
        state: None,
        prev: PrevRows::new(),
    })
}

fn take_recon_scratch(pool: &ScratchPool, job: &FrameJob) -> ReconScratch {
    let (yw, yh, ..) = job.geom[0];
    let (cw, chh, ..) = job.geom[1];
    let mut list = lock(pool);
    while let Some(s) = list.recon.pop() {
        if (s.y.width(), s.y.height(), s.u.width(), s.u.height()) == (yw, yh, cw, chh) {
            return s;
        }
    }
    ReconScratch {
        y: Plane::new(yw, yh),
        u: Plane::new(cw, chh),
        v: Plane::new(cw, chh),
        tok_buf: Box::new([0; 1024]),
        pred: crate::inter_pred::PredScratch::new_boxed(),
    }
}

/// A logged superblock row in flight: `(mi_row, MiColStart, MiColEnd,
/// log)`.
type LoggedRow = (u32, u32, u32, ReconLog);

/// One tile column's logged superblock rows on their way from the
/// column's entropy-decode job to its reconstruction job, in order, plus
/// "no more rows".
#[derive(Default)]
struct RowQueue {
    rows: Mutex<(VecDeque<LoggedRow>, bool)>,
    ready: Condvar,
}

impl RowQueue {
    fn push(&self, item: LoggedRow) {
        lock(&self.rows).0.push_back(item);
        self.ready.notify_all();
    }

    fn close(&self) {
        lock(&self.rows).1 = true;
        self.ready.notify_all();
    }

    /// The next row, or `None` once the queue is closed and drained.
    fn pop(&self) -> Option<LoggedRow> {
        let mut q = lock(&self.rows);
        loop {
            if let Some(item) = q.0.pop_front() {
                return Some(item);
            }
            if q.1 {
                return None;
            }
            q = self.ready.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Closes a row queue when its producer ends (normally or by panic).
struct CloseOnDrop<'a>(&'a RowQueue);

impl Drop for CloseOnDrop<'_> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Copy MI row range `[ r0, r1 )`, columns `[ c0, c1 )` of `src` (`m`
/// entries per MI, entry `k` taken) into a row-local array starting at
/// MI row `base`.
#[allow(clippy::too_many_arguments)]
fn copy_mi_rows<T: Copy, U>(
    dst: &mut [U],
    src: &[T],
    (r0, r1, c0, c1, base, cols): (usize, usize, usize, usize, usize, usize),
    f: impl Fn(T) -> U,
    m: usize,
    k: usize,
) {
    for r in r0..r1 {
        let d = &mut dst[(r - base) * cols + c0..(r - base) * cols + c1];
        let s = &src[(r * cols + c0) * m + k..(r * cols + c1) * m];
        for (o, v) in d.iter_mut().zip(s.iter().step_by(m)) {
            *o = f(*v);
        }
    }
}

/// The entropy-decode job of tile column `col`: decode every tile of
/// the column (partition, mode info, motion vectors, tokens), logging
/// each superblock row's pixel work for the column's reconstruction job
/// and publishing the row's mode info.
fn tile_parse_job(job: &FrameJob, pipe: &FramePipe, scratch_pool: &ScratchPool, col: usize) {
    let _guard = AbortOnPanic(pipe);
    let _close = CloseOnDrop(&pipe.row_queues[col]);
    let sh = job.shared();
    let mut scratch = take_parse_scratch(scratch_pool);
    let (mi_rows, mi_cols) = (job.mi_rows, job.mi_cols);
    let mut state = match scratch.state.take() {
        Some(mut st) if (st.mi_rows, st.mi_cols) == (mi_rows, mi_cols) => {
            st.clear();
            st
        }
        _ => Vp9FrameState::new(mi_rows, mi_cols),
    };
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(job.ssx)) as usize,
            ((2 * mi_rows) >> u32::from(job.ssy)) as usize,
        ),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(job.ssx)) as usize,
            ((2 * mi_rows) >> u32::from(job.ssy)) as usize,
        ),
    ];
    let mut pctx =
        PartitionContextState::new((job.sb64_cols * 8) as usize, (job.sb64_rows * 8) as usize);
    // §7.4.1 clear_above_context( ): fresh strips are zero; the explicit
    // reset documents the once-per-frame spec step.
    pctx.clear_above();
    let mut seg_pred_ctx = SegPredContextState::new(mi_cols, mi_rows);
    let mut counts = crate::prob_adapt::FrameCounts::new_boxed();
    scratch.prev.fit(mi_rows, mi_cols);
    let new_log = || lock(scratch_pool).logs.pop().unwrap_or_default();
    let mut log = new_log();

    let mut failure = None;
    for idx in (col..job.tiles.len()).step_by(job.tile_cols) {
        let task = job.task(idx);
        let (c0, c1) = (task.mi_col_start, task.mi_col_end);
        let ParseScratch {
            token_cache,
            tok_buf,
            prev,
            ..
        } = &mut scratch;
        let before = |prev: &mut PrevRows, r: u32| -> Result<(), Error> {
            prev.import(
                (r / 8) as usize,
                job.prev_mvs.as_deref(),
                job.prev_seg.as_deref(),
                mi_rows,
                mi_cols,
            )
        };
        let mut publish = |r: u32, l: &mut ReconLog, st: &Vp9FrameState| {
            publish_tile_mi(job, pipe, r, c0, c1, st);
            let row = std::mem::replace(l, new_log());
            pipe.row_queues[col].push((r, c0, c1, row));
        };
        let res = decode_one_tile(
            &sh,
            &task,
            TileWorkState {
                state: &mut state,
                log: &mut log,
                nz: &mut nz,
                pctx: &mut pctx,
                seg_pred_ctx: &mut seg_pred_ctx,
                counts: &mut counts,
                token_cache,
                tok_buf,
            },
            prev,
            &before,
            &mut publish,
        );
        if let Err(e) = res {
            failure = Some((idx, e));
            break;
        }
    }
    let out = match failure {
        Some(err) => {
            pipe.abort();
            Err(err)
        }
        None => {
            scratch.state = Some(state);
            let mut pools = lock(scratch_pool);
            pools.parse.push(scratch);
            log.clear();
            pools.logs.push(log);
            Ok(TileOut { counts })
        }
    };
    lock(&pipe.tile_out)[col] = Some(out);
    pipe.tiles_finished.notify_all();
}

/// The reconstruction job of tile column `col`: replay each logged
/// superblock row into the column's working planes and publish it.
fn tile_recon_job(job: &FrameJob, pipe: &FramePipe, scratch_pool: &ScratchPool, col: usize) {
    let _guard = AbortOnPanic(pipe);
    let mut scratch = take_recon_scratch(scratch_pool, job);
    let ctx = PixelCtx {
        refs: job.refs.as_ref(),
        ref_frame_idx: job.hdr.ref_frame_idx.unwrap_or([0, 1, 2]),
        frame_width: job.hdr.frame_width,
        frame_height: job.hdr.frame_height,
        mi_rows: job.mi_rows,
        mi_cols: job.mi_cols,
        subsampling_x: job.ssx,
        subsampling_y: job.ssy,
        bit_depth: job.bit_depth,
        lossless: job.lossless,
    };
    while let Some((r, c0, c1, mut log)) = pipe.row_queues[col].pop() {
        let ReconScratch {
            y,
            u,
            v,
            tok_buf,
            pred,
        } = &mut scratch;
        replay(&log, &ctx, [&mut *y, &mut *u, &mut *v], pred, tok_buf);
        publish_tile_pixels(job, pipe, r, c0, c1, [y, u, v]);
        log.clear();
        lock(scratch_pool).logs.push(log);
    }
    lock(scratch_pool).recon.push(scratch);
}

/// Copy superblock row `r`'s reconstruction (MI columns `[ c0, c1 )`)
/// into the row's band, then publish it (without a loop filter the last
/// column to publish freezes the band).
fn publish_tile_pixels(
    job: &FrameJob,
    pipe: &FramePipe,
    r: u32,
    c0: u32,
    c1: u32,
    planes: [&Plane; 3],
) {
    let sb = (r / 8) as usize;
    {
        let mut band = lock(&pipe.bands[sb]);
        for (p, plane) in planes.iter().enumerate() {
            let (w, h, shift, _) = job.geom[p];
            let sx = if p == 0 { 0 } else { u32::from(job.ssx) };
            let y0 = sb << shift;
            let rows = band_rows(h, shift, sb);
            let x0 = ((c0 * 8) >> sx) as usize;
            let x1 = (((c1 * 8) >> sx) as usize).min(w);
            let src = plane.samples();
            for k in 0..rows {
                let at = (y0 + k) * w;
                band[p][(HALO + k) * w + x0..(HALO + k) * w + x1]
                    .copy_from_slice(&src[at + x0..at + x1]);
            }
        }
    }
    let complete = {
        let mut p = pipe.progress();
        p.tiles_done[sb] += 1;
        p.tiles_done[sb] == job.tile_cols
    };
    if complete && job.lf.is_none() {
        let band = std::mem::take(&mut *lock(&pipe.bands[sb]));
        pipe.frame.freeze_band(sb, band);
    }
    pipe.notify(sb);
}

/// Copy superblock row `r`'s mode info (MI columns `[ c0, c1 )`) into the
/// row's record; the last column to publish freezes it.
fn publish_tile_mi(job: &FrameJob, pipe: &FramePipe, r: u32, c0: u32, c1: u32, st: &Vp9FrameState) {
    let sb = (r / 8) as usize;
    {
        let mut mi = lock(&pipe.mi_build[sb]);
        let cols = job.mi_cols as usize;
        let r0 = r as usize;
        let r1 = (r as usize + 8).min(job.mi_rows as usize);
        let g = (r0, r1, c0 as usize, c1 as usize, r0, cols);
        let MiRow {
            mi_sizes,
            tx_sizes,
            skips,
            ref0,
            ref1,
            mv0,
            mv1,
            y_modes,
            seg_ids,
        } = &mut *mi;
        copy_mi_rows(mi_sizes, &st.mi_sizes, g, |v| v, 1, 0);
        copy_mi_rows(tx_sizes, &st.tx_sizes, g, |v| v, 1, 0);
        copy_mi_rows(skips, &st.skips, g, |v| v != 0, 1, 0);
        copy_mi_rows(ref0, &st.ref_frames, g, |v| v, 2, 0);
        copy_mi_rows(ref1, &st.ref_frames, g, |v| v, 2, 1);
        copy_mi_rows(mv0, &st.mvs, g, |v| v, 2, 0);
        copy_mi_rows(mv1, &st.mvs, g, |v| v, 2, 1);
        copy_mi_rows(y_modes, &st.y_modes, g, |v| v, 1, 0);
        copy_mi_rows(seg_ids, &st.segment_ids, g, |v| v, 1, 0);
    }
    let complete = {
        let mut p = pipe.progress();
        p.mi_done[sb] += 1;
        p.mi_done[sb] == job.tile_cols
    };
    if complete {
        let row = std::mem::replace(&mut *lock(&pipe.mi_build[sb]), MiRow::new(0));
        pipe.frame.freeze_mi(sb, row);
    }
}

/// The loop-filter job body for superblock row `r`.
///
/// # Why the rows can overlap
///
/// The §8.8.2 call for superblock `(row, col)` reads and writes only
/// the samples of its own superblock plus the [`HALO`] rows above it
/// (its top edge, horizontal pass) and the seven columns left of it
/// (its left edge, vertical pass). In the raster the samples it needs
/// from the superblock row above are final once that row has finished
/// superblock `col + 1` — the last call of the row above that writes
/// them (the left edge of `col + 1` reaches seven columns into `col`).
/// No call of the row above after that point touches them, and no call
/// of this row touches anything the row above still has to read. So
/// each superblock row runs in a wavefront two superblocks behind the
/// row above and every call sees exactly the samples the serial raster
/// shows it.
///
/// # Data flow
///
/// The row owns its band (locked for the whole job). When it finishes
/// superblock `col`, it publishes the bottom [`HALO`] rows of superblock
/// `col - 1` (final as far as it is concerned) into a per-superblock
/// slot; the row below copies the slot into its band's halo rows before
/// filtering superblock `col - 1`, and once done writes the (now final)
/// halo rows back into the band above, which it then freezes (the last
/// row freezes its own band too).
fn loop_filter_row_job(lfp: &LfParams, pipe: &FramePipe, r: usize) {
    let _guard = AbortOnPanic(pipe);
    let (sb_rows, sb_cols) = (lfp.sb_rows, lfp.sb_cols);
    if !pipe.wait(r, |p| p.tiles_done[r] >= lfp.ready_at) {
        return;
    }
    let Some(lf) = pipe.frame.mi_row(r) else {
        return;
    };
    let mut band = lock(&pipe.bands[r]);
    let frame = SuperblockFilterFrame {
        mi_sizes: &lf.mi_sizes,
        tx_sizes: &lf.tx_sizes,
        skips: &lf.skips,
        ref_frames_0: &lf.ref0,
        y_modes: &lf.y_modes,
        segment_ids: &lf.seg_ids,
        mi_cols: lfp.mi_cols,
        mi_rows: lfp.mi_rows,
        subsampling_x: u8::from(lfp.ssx),
        subsampling_y: u8::from(lfp.ssy),
        loop_filter_sharpness: lfp.sharpness,
        bit_depth: lfp.bit_depth,
        lvl_lookup: &lfp.lvl_lookup,
    };
    let cells = LfRowCells::new(&frame, (r * 8) as u32);
    let col_range = |p: usize, c: usize| -> (usize, usize) {
        let (w, _, _, sbw) = lfp.geom[p];
        ((c << sbw).min(w), ((c + 1) << sbw).min(w))
    };
    // The bottom HALO own rows of this row's band at superblock `c`.
    let publish_slot = |band: &Band, c: usize| {
        let cells: [Vec<u16>; 3] = std::array::from_fn(|p| {
            let (w, h, shift, _) = lfp.geom[p];
            let own = band_rows(h, shift, r);
            let (x0, x1) = col_range(p, c);
            let mut cell = Vec::with_capacity(HALO * (x1 - x0));
            for k in own..own + HALO {
                cell.extend_from_slice(&band[p][k * w + x0..k * w + x1]);
            }
            cell
        });
        let _ = pipe.slots[r * sb_cols + c].set(cells);
    };
    for c in 0..sb_cols {
        if r > 0 {
            if !pipe.wait(r, |p| p.lf_published[r - 1] > c) {
                return;
            }
            let cells = pipe.slots[(r - 1) * sb_cols + c]
                .get()
                .expect("slot published before progress");
            for p in 0..3 {
                let (w, ..) = lfp.geom[p];
                let (x0, x1) = col_range(p, c);
                if x1 == x0 {
                    continue;
                }
                for (k, src) in cells[p].chunks_exact(x1 - x0).enumerate() {
                    band[p][k * w + x0..k * w + x1].copy_from_slice(src);
                }
            }
        }
        for p in 0..3 {
            let (w, h, shift, _) = lfp.geom[p];
            let own = band_rows(h, shift, r);
            let y0 = r << shift;
            let mut window = if r == 0 {
                PlaneWindow {
                    data: &mut band[p][HALO * w..],
                    stride: w,
                    width: w,
                    height: own,
                    y_origin: 0,
                }
            } else {
                PlaneWindow {
                    data: &mut band[p][..],
                    stride: w,
                    width: w,
                    height: HALO + own,
                    y_origin: y0 - HALO,
                }
            };
            for pass in 0..2u8 {
                superblock_loop_filter_cells(
                    &mut window,
                    &cells,
                    p as u8,
                    pass,
                    (r * 8) as u32,
                    (c * 8) as u32,
                );
            }
        }
        if c > 0 && r + 1 < sb_rows {
            publish_slot(&band, c - 1);
            pipe.progress().lf_published[r] = c;
            pipe.notify(r + 1);
        }
    }
    if r + 1 < sb_rows {
        publish_slot(&band, sb_cols - 1);
    }
    pipe.progress().lf_published[r] = sb_cols;
    pipe.notify(r + 1);
    // The halo rows are final: they are the tail of the band above
    // (whose top froze when its own row finished).
    if r > 0 {
        let tail: Band = std::array::from_fn(|p| {
            let (w, ..) = lfp.geom[p];
            debug_assert_eq!(pipe.frame.tail_rows(p, r - 1), HALO);
            band[p][..HALO * w].to_vec()
        });
        pipe.frame.freeze_tail(r - 1, tail);
    }
    // This band is final above its tail; the next row's filter only
    // changes the tail (through its own halo copy).
    let own = std::mem::take(&mut *band);
    if r + 1 == sb_rows {
        pipe.frame.freeze_band(r, own);
    } else {
        pipe.frame.freeze_top(r, own);
    }
}

/// The banded loop-filter row jobs over a whole [`crate::CurrFrame`]
/// on a `workers`-thread pool (test oracle hook: the bytes must equal the
/// serial [`crate::frame_loop_filter`] raster at every worker count).
#[cfg(test)]
pub(crate) fn frame_loop_filter_banded(
    curr: &mut crate::CurrFrame,
    frame: &SuperblockFilterFrame,
    workers: usize,
) {
    let (mi_rows, mi_cols) = (frame.mi_rows, frame.mi_cols);
    let (sb_rows, sb_cols) = (mi_rows.div_ceil(8) as usize, mi_cols.div_ceil(8) as usize);
    let (sxu, syu) = (
        u32::from(frame.subsampling_x),
        u32::from(frame.subsampling_y),
    );
    let geom: [(usize, usize, u32, u32); 3] = std::array::from_fn(|p| {
        let v = &curr.planes[p];
        if p == 0 {
            (v.width, v.height, 6, 6)
        } else {
            (v.width, v.height, 6 - syu, 6 - sxu)
        }
    });
    let lfp = LfParams {
        geom,
        sb_rows,
        sb_cols,
        mi_rows,
        mi_cols,
        ssx: sxu == 1,
        ssy: syu == 1,
        sharpness: frame.loop_filter_sharpness,
        bit_depth: frame.bit_depth,
        lvl_lookup: *frame.lvl_lookup,
        ready_at: 0,
    };
    let fgeom: [PlaneGeom; 3] = std::array::from_fn(|p| PlaneGeom {
        stride: geom[p].0,
        height: geom[p].1,
        band_shift: geom[p].2,
    });
    let target = Arc::new(FrameBuf::new(
        fgeom,
        curr.planes[0].width as u32,
        curr.planes[0].height as u32,
        sxu == 1,
        syu == 1,
        frame.bit_depth,
        mi_rows,
    ));
    for r in 0..sb_rows {
        let mut row = MiRow::new(8 * mi_cols as usize);
        let cols = mi_cols as usize;
        let r0 = r * 8;
        let r1 = (r0 + 8).min(mi_rows as usize);
        for (i, src) in (r0 * cols..r1 * cols).enumerate() {
            row.mi_sizes[i] = frame.mi_sizes[src];
            row.tx_sizes[i] = frame.tx_sizes[src];
            row.skips[i] = frame.skips[src];
            row.ref0[i] = frame.ref_frames_0[src];
            row.y_modes[i] = frame.y_modes[src];
            row.seg_ids[i] = frame.segment_ids[src];
        }
        target.freeze_mi(r, row);
    }
    let pipe = Arc::new(FramePipe {
        frame: Arc::clone(&target),
        bands: (0..sb_rows)
            .map(|b| {
                Mutex::new(std::array::from_fn(|p| {
                    let (w, h, shift, _) = geom[p];
                    let rows = band_rows(h, shift, b);
                    let v = &curr.planes[p];
                    let mut buf = vec![0u16; (HALO + rows) * w];
                    for k in 0..rows {
                        let y = (b << shift) + k;
                        for (d, &s) in buf[(HALO + k) * w..(HALO + k + 1) * w]
                            .iter_mut()
                            .zip(&v.data[y * v.stride..y * v.stride + w])
                        {
                            *d = s as u16;
                        }
                    }
                    buf
                }))
            })
            .collect(),
        mi_build: Vec::new(),
        slots: (0..sb_rows * sb_cols).map(|_| OnceLock::new()).collect(),
        progress: Mutex::new(Progress {
            tiles_done: vec![0; sb_rows],
            mi_done: vec![0; sb_rows],
            lf_published: vec![0; sb_rows],
            abort: false,
        }),
        changed: (0..sb_rows).map(|_| Condvar::new()).collect(),
        tile_out: Mutex::new(Vec::new()),
        tiles_finished: Condvar::new(),
        next_lf_row: AtomicUsize::new(0),
        row_queues: Vec::new(),
    });
    let lfp = Arc::new(lfp);
    let jobs: Vec<Job> = (0..sb_rows)
        .map(|r| {
            let (lfp, pipe) = (Arc::clone(&lfp), Arc::clone(&pipe));
            Box::new(move || loop_filter_row_job(&lfp, &pipe, r)) as Job
        })
        .collect();
    WorkerPool::new(workers.saturating_sub(1)).run(jobs);
    for (p, &(w, h, ..)) in geom.iter().enumerate() {
        let v = &mut curr.planes[p];
        for y in 0..h {
            for (d, &s) in v.data[y * v.stride..y * v.stride + w]
                .iter_mut()
                .zip(&target.row(p, y)[..w])
            {
                *d = i32::from(s);
            }
        }
    }
}

/// Loop-filter jobs per frame (each filters every `LF_JOBS`-th
/// superblock row).
const LF_JOBS: usize = 3;

/// A frame whose batch has been submitted.
pub(crate) struct InFlight {
    frame: Arc<FrameBuf>,
    pipe: Arc<FramePipe>,
    batch: Option<Batch>,
    /// The merged §9.3.4 counts (or the first tile error), once known.
    counts: Option<Result<Box<crate::prob_adapt::FrameCounts>, Error>>,
}

impl InFlight {
    /// The frame being decoded.
    pub(crate) fn frame(&self) -> &Arc<FrameBuf> {
        &self.frame
    }

    /// Block until every tile column has finished; the merged §9.3.4
    /// counts, or the lowest raster-order tile error (the one the serial
    /// §6.4 walk would have hit first).
    fn counts(&mut self) -> Result<&crate::prob_adapt::FrameCounts, Error> {
        if self.counts.is_none() {
            let mut outs = self.pipe.wait_tiles();
            let mut first_err: Option<(usize, Error)> = None;
            let mut merged: Option<Box<crate::prob_adapt::FrameCounts>> = None;
            for out in outs.iter_mut() {
                match out.take().expect("every tile column ran") {
                    Ok(o) => match merged.as_mut() {
                        Some(m) => m.merge_from(&o.counts),
                        None => merged = Some(o.counts),
                    },
                    Err((idx, e)) => {
                        if first_err.as_ref().is_none_or(|(i, _)| idx < *i) {
                            first_err = Some((idx, e));
                        }
                    }
                }
            }
            drop(outs);
            self.counts = Some(match first_err {
                Some((_, e)) => Err(e),
                None => Ok(merged.unwrap_or_else(crate::prob_adapt::FrameCounts::new_boxed)),
            });
        }
        match self.counts.as_ref().expect("computed") {
            Ok(c) => Ok(c),
            Err(e) => Err(*e),
        }
    }

    /// The counts after a successful [`Self::counts`].
    fn take_counts(&mut self) -> Box<crate::prob_adapt::FrameCounts> {
        match self.counts.take() {
            Some(Ok(c)) => c,
            _ => crate::prob_adapt::FrameCounts::new_boxed(),
        }
    }
}

/// Submit one frame's batch (tile columns, then loop-filter rows) and
/// return without waiting. `reuse` lends a recycled frame's bands.
fn launch_frame(
    job: FrameJob,
    pool: Option<&WorkerPool>,
    scratch_pool: &ScratchPool,
    reuse: Option<Bands>,
) -> InFlight {
    let sb_rows = job.sb64_rows as usize;
    let sb_cols = job.sb64_cols as usize;
    let lf_on = job.lf.is_some();
    let fgeom: [PlaneGeom; 3] = std::array::from_fn(|p| PlaneGeom {
        stride: job.geom[p].0,
        height: job.geom[p].1,
        band_shift: job.geom[p].2,
    });
    let frame = Arc::new(
        FrameBuf::new(
            fgeom,
            job.hdr.frame_width,
            job.hdr.frame_height,
            job.ssx,
            job.ssy,
            job.hdr.color_config.bit_depth,
            job.mi_rows,
        )
        .with_color(
            job.hdr.color_config.color_space,
            job.hdr.color_config.color_range_full,
        ),
    );
    // Band storage: a recycled frame's when the geometry matches (every
    // own row is overwritten by the tile columns and every halo row the
    // loop filter reads is first filled from a slot, so stale contents
    // are never observed), else fresh.
    let mut reuse = reuse.filter(|r| {
        r.len() == sb_rows
            && r.iter().enumerate().all(|(b, band)| {
                band.as_ref()
                    .is_some_and(|band| (0..3).all(|p| band[p].len() == fgeom[p].band_len(b)))
            })
    });
    let bands: Vec<Mutex<Band>> = (0..sb_rows)
        .map(|b| {
            let band = match reuse.as_mut() {
                Some(r) => r[b].take().expect("checked band"),
                None => std::array::from_fn(|p| vec![0u16; fgeom[p].band_len(b)]),
            };
            Mutex::new(band)
        })
        .collect();
    let pipe = Arc::new(FramePipe {
        frame: Arc::clone(&frame),
        bands,
        mi_build: (0..sb_rows)
            .map(|_| Mutex::new(MiRow::new(8 * job.mi_cols as usize)))
            .collect(),
        slots: if lf_on {
            (0..sb_rows * sb_cols).map(|_| OnceLock::new()).collect()
        } else {
            Vec::new()
        },
        progress: Mutex::new(Progress {
            tiles_done: vec![0; sb_rows],
            mi_done: vec![0; sb_rows],
            lf_published: vec![0; sb_rows],
            abort: false,
        }),
        changed: (0..sb_rows).map(|_| Condvar::new()).collect(),
        tile_out: Mutex::new((0..job.tile_cols).map(|_| None).collect()),
        tiles_finished: Condvar::new(),
        next_lf_row: AtomicUsize::new(0),
        row_queues: (0..job.tile_cols).map(|_| RowQueue::default()).collect(),
    });
    let job = Arc::new(job);

    let mut jobs: Vec<Job> = Vec::new();
    // Every column's entropy decode, then every column's reconstruction
    // (each waits only on its own column's decode, queued earlier).
    for col in 0..job.tile_cols {
        let (job, pipe, scratch) = (
            Arc::clone(&job),
            Arc::clone(&pipe),
            Arc::clone(scratch_pool),
        );
        jobs.push(Box::new(move || tile_parse_job(&job, &pipe, &scratch, col)));
    }
    for col in 0..job.tile_cols {
        let (job, pipe, scratch) = (
            Arc::clone(&job),
            Arc::clone(&pipe),
            Arc::clone(scratch_pool),
        );
        jobs.push(Box::new(move || tile_recon_job(&job, &pipe, &scratch, col)));
    }
    if lf_on {
        // A few loop-filter jobs that claim superblock rows in order: a
        // waiting job holds its thread, so one job per row would park
        // most of the pool on rows not decoded yet (and keep the next
        // frame's tile columns queued behind them). Claiming in order
        // keeps the FIFO argument: a row only ever waits on the tile
        // columns (queued earlier) and on the row above, which a running
        // job claimed first.
        for _ in 0..LF_JOBS.min(sb_rows) {
            let (job, pipe) = (Arc::clone(&job), Arc::clone(&pipe));
            jobs.push(Box::new(move || {
                let lfp = job.lf.as_ref().expect("loop filter on");
                loop {
                    let r = pipe.next_lf_row.fetch_add(1, Ordering::Relaxed);
                    if r >= lfp.sb_rows {
                        break;
                    }
                    loop_filter_row_job(lfp, &pipe, r);
                }
            }));
        }
    }
    let batch = match pool {
        Some(pool) => Some(pool.submit(jobs)),
        None => {
            jobs.into_iter().for_each(|j| j());
            None
        }
    };
    InFlight {
        frame,
        pipe,
        batch,
        counts: None,
    }
}

/// Wait for a launched frame and collect its products.
fn finish_frame(
    pool: Option<&WorkerPool>,
    mut inflight: InFlight,
) -> Result<FrameDecodeProducts, Error> {
    if let (Some(pool), Some(batch)) = (pool, inflight.batch.as_ref()) {
        pool.wait(batch);
    }
    inflight.counts()?;
    if !inflight.frame.wait_complete() {
        return Err(Error::InvalidBitstream);
    }
    let counts = inflight.take_counts();
    Ok(FrameDecodeProducts {
        frame: inflight.frame,
        counts,
    })
}

/// Decode one intra (key or intra-only) VP9 frame to planar samples.
///
/// `data` is a single frame's byte payload (uncompressed header +
/// compressed header + tile data) — e.g. one IVF frame body. Inter
/// frames need reference-buffer state; use [`decode_vp9_sequence`] to
/// decode a multi-frame stream that includes P-frames.
/// `show_existing_frame` short frames return [`Error::Unsupported`].
pub fn decode_intra_frame(data: &[u8]) -> Result<Vp9DecodedFrame, Error> {
    // §6.2 uncompressed header.
    let hdr: Vp9FrameHeader = parse_uncompressed_header(data)?;
    if hdr.show_existing_frame {
        return Err(Error::Unsupported);
    }
    let frame_is_intra = matches!(hdr.frame_type, FrameType::KeyFrame) || hdr.intra_only;
    if !frame_is_intra {
        return Err(Error::Unsupported);
    }

    // §6.3 compressed header: the `header_size_in_bytes` slice right
    // after the byte-aligned uncompressed header.
    let lossless = hdr.quantization.lossless;
    let ch_start = hdr.uncompressed_header_size_bytes;
    let ch_size = hdr.header_size_in_bytes as usize;
    if ch_size == 0 {
        // §7.2: a non-show-existing frame carries a compressed header.
        return Err(Error::InvalidBitstream);
    }
    let ch_end = ch_start
        .checked_add(ch_size)
        .ok_or(Error::InvalidBitstream)?;
    if data.len() < ch_end {
        return Err(Error::UnexpectedEof);
    }
    let chdr = ChdrKind::Intra(Box::new(parse_compressed_header(
        &data[ch_start..ch_end],
        lossless,
    )?));
    let payload: Arc<[u8]> = Arc::from(data);
    let scratch = ScratchPool::default();
    let inflight = launch_single_frame(
        &payload,
        &hdr,
        ch_end,
        chdr,
        None,
        (None, None, false),
        None,
        &scratch,
        None,
    )?;
    Ok(finish_frame(None, inflight)?.frame.to_decoded_frame())
}

/// The default (serial) execution context for the entry points that
/// take no caller-supplied one.
const EXEC_SERIAL: ExecutionContext = ExecutionContext::serial();

/// Test-only companion of [`decode_intra_frame`] that decodes under a
/// caller-supplied [`ExecutionContext`] and also returns the §9.3.4
/// count bank — the direct pin that the tile-parallel merge reproduces
/// the serial walk's counts (the §6.1.2 `refresh_probs( )` input)
/// exactly, not just the output samples.
#[cfg(test)]
pub(crate) fn decode_intra_products_for_test(
    data: &[u8],
    exec: &ExecutionContext,
) -> Result<(Vp9DecodedFrame, Box<crate::prob_adapt::FrameCounts>), Error> {
    let hdr: Vp9FrameHeader = parse_uncompressed_header(data)?;
    if hdr.show_existing_frame {
        return Err(Error::Unsupported);
    }
    let frame_is_intra = matches!(hdr.frame_type, FrameType::KeyFrame) || hdr.intra_only;
    if !frame_is_intra {
        return Err(Error::Unsupported);
    }
    let lossless = hdr.quantization.lossless;
    let ch_start = hdr.uncompressed_header_size_bytes;
    let ch_size = hdr.header_size_in_bytes as usize;
    if ch_size == 0 {
        return Err(Error::InvalidBitstream);
    }
    let ch_end = ch_start
        .checked_add(ch_size)
        .ok_or(Error::InvalidBitstream)?;
    if data.len() < ch_end {
        return Err(Error::UnexpectedEof);
    }
    let chdr = ChdrKind::Intra(Box::new(parse_compressed_header(
        &data[ch_start..ch_end],
        lossless,
    )?));
    let payload: Arc<[u8]> = Arc::from(data);
    let scratch = ScratchPool::default();
    let pool = (exec.threads > 1).then(|| WorkerPool::new(exec.threads - 1));
    let inflight = launch_single_frame(
        &payload,
        &hdr,
        ch_end,
        chdr,
        None,
        (None, None, false),
        pool.as_ref(),
        &scratch,
        None,
    )?;
    let products = finish_frame(pool.as_ref(), inflight)?;
    Ok((products.frame.to_decoded_frame(), products.counts))
}

/// Decoder resource bound on the MI-padded luma picture size, in
/// samples: the largest `Max luma picture size` any VP9 level defines
/// (35 651 584, levels 6 / 6.1 / 6.2 — see the level table staged at
/// `docs/video/vp9/webmproject-vp9-levels-testing.html`). A syntactically
/// valid §6.2 header can claim up to 65536 × 65536 (17 GiB of working
/// planes per frame); no conformant stream exceeds this bound, so frames
/// past it are rejected as [`Error::Unsupported`] before any
/// frame-geometry-sized allocation happens.
const MAX_LUMA_PICTURE_SIZE: u64 = 35_651_584;

/// Decode one frame (intra or inter) given the parsed header, the
/// compressed header, the optional §8.10 reference buffers, and the
/// optional previous-frame mode-info sources. Returns the in-flight
/// frame (see [`finish_frame`]).
///
/// The frame runs as one batch of tile-column and loop-filter-row jobs
/// (see the frame-pipeline section) on `pool` — inline, in the same
/// order, when there is none. The output is byte-identical for every
/// pool size.
#[allow(clippy::too_many_arguments)]
fn launch_single_frame(
    data: &Arc<[u8]>,
    hdr: &Vp9FrameHeader,
    ch_end: usize,
    chdr_kind: ChdrKind,
    refs: Option<&RefBuffers>,
    (prev_mvs, prev_seg, prev_shown): (Option<Arc<FrameBuf>>, Option<Arc<FrameBuf>>, bool),
    pool: Option<&WorkerPool>,
    scratch: &ScratchPool,
    reuse: Option<Bands>,
) -> Result<InFlight, Error> {
    let lossless = hdr.quantization.lossless;

    // §7.2.6: MiCols / MiRows; §7.2 Sb64 extents for the §7.4.1 strip
    // spans (AbovePartitionContext is read beyond MiCols).
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
    // Resource guard: bound the working-plane allocation by the largest
    // luma picture size any VP9 level admits (see
    // [`MAX_LUMA_PICTURE_SIZE`]) before allocating anything sized by the
    // header's claimed geometry.
    if u64::from(mi_cols * 8) * u64::from(mi_rows * 8) > MAX_LUMA_PICTURE_SIZE {
        return Err(Error::Unsupported);
    }
    let sb64_cols = (mi_cols + 7) >> 3;
    let sb64_rows = (mi_rows + 7) >> 3;
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let bit_depth = u32::from(hdr.color_config.bit_depth);

    // CurrFrame plane extents at the §6.4.21 / §8.5.1 working size
    // (MiCols * 8 wide — prediction and reconstruction may touch the
    // full MI-aligned area; §8.10 crops on output).
    let y_w = (mi_cols * 8) as usize;
    let y_h = (mi_rows * 8) as usize;
    let uv_w = ((mi_cols * 8) >> u32::from(ssx)) as usize;
    let uv_h = ((mi_rows * 8) >> u32::from(ssy)) as usize;
    let (sxu, syu) = (u32::from(ssx), u32::from(ssy));
    let geom = [
        (y_w, y_h, 6, 6),
        (uv_w, uv_h, 6 - syu, 6 - sxu),
        (uv_w, uv_h, 6 - syu, 6 - sxu),
    ];

    let (chdr, chdr_inter) = match chdr_kind {
        ChdrKind::Intra(c) => (*c, None),
        ChdrKind::Inter(c) => (c.intra.clone(), Some(*c)),
    };
    let is_inter_frame = chdr_inter.is_some();

    // §7.2.6 UsePrevFrameMvs: requires a same-dimension prev frame that
    // was shown, error_resilient_mode == 0, and FrameIsIntra == 0. (A
    // size-mismatched prev never reaches here — the caller passes no
    // source — and the zero snapshots stand in, as §6.4.14 clears
    // PrevSegmentIds.)
    let use_prev_frame_mvs = is_inter_frame && prev_shown && !hdr.error_resilient_mode;

    // §6.4 decode_tiles( sz ): per-tile byte budget walk, then the
    // per-tile bool-coder bracket around §6.4.2 decode_tile( ).
    let tile_data = &data[ch_end..];
    let sz = tile_data.len() as u32;
    let tile_cols_log2 = hdr.tile_info.tile_cols_log2;
    let tile_rows_log2 = hdr.tile_info.tile_rows_log2;
    let sizes = tile_payload_sizes(tile_data, sz, tile_rows_log2, tile_cols_log2)?;

    let tile_cols = 1u32 << tile_cols_log2;
    let tile_rows = 1u32 << tile_rows_log2;

    // Pre-slice every tile payload — the pure §6.4 byte arithmetic
    // (`tile_payload_sizes( )` has already validated every f(32)
    // prefix, the running `sz` budget, and every body range).
    let mut tiles: Vec<TileSpan> = Vec::with_capacity(sizes.len());
    {
        let mut byte_cursor = ch_end;
        let mut size_idx = 0usize;
        for tile_row in 0..tile_rows {
            for tile_col in 0..tile_cols {
                let last_tile = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
                if !last_tile {
                    byte_cursor += 4; // the f(32) tile_size prefix
                }
                let tile_size = sizes[size_idx] as usize;
                size_idx += 1;
                tiles.push(TileSpan {
                    mi_row_start: get_tile_offset(tile_row, mi_rows, u32::from(tile_rows_log2)),
                    mi_row_end: get_tile_offset(tile_row + 1, mi_rows, u32::from(tile_rows_log2)),
                    mi_col_start: get_tile_offset(tile_col, mi_cols, u32::from(tile_cols_log2)),
                    mi_col_end: get_tile_offset(tile_col + 1, mi_cols, u32::from(tile_cols_log2)),
                    start: byte_cursor,
                    len: tile_size,
                });
                byte_cursor += tile_size;
            }
        }
    }

    // §8.8 loop filter setup. The §8.8.1 init consumes the §6.2.8 deltas
    // resolved against the §7.2 setup_past_independence defaults (intra
    // frames reset them). §8.1 step 2: the §8.8 process runs only "if
    // loop_filter_level is not equal to 0" — the gate is on the frame
    // header's level itself, not on the LvlLookup entries a delta or a
    // SEG_LVL_ALT_L override may lift above zero.
    //
    // §8.8 input is CurrFrame at the MI-aligned working extents: the
    // §8.8.2 step-13 onScreen predicate tests `x >= 8 * MiCols` /
    // `y >= 8 * MiRows`, NOT the visible FrameWidth / FrameHeight, and
    // the §8.8.5.1 stencil reads the real reconstructed samples beyond
    // the visible edge (§8.10 crops after filtering).
    let lf = (hdr.loop_filter.level != 0).then(|| {
        let (ref_deltas, mode_deltas) = crate::loop_filter::resolve_lf_deltas(&hdr.loop_filter);
        LfParams {
            geom,
            sb_rows: sb64_rows as usize,
            sb_cols: sb64_cols as usize,
            mi_rows,
            mi_cols,
            ssx,
            ssy,
            sharpness: hdr.loop_filter.sharpness,
            bit_depth: hdr.color_config.bit_depth,
            lvl_lookup: loop_filter_frame_init(
                &hdr.loop_filter,
                &hdr.segmentation,
                ref_deltas,
                mode_deltas,
            ),
            ready_at: tile_cols as usize,
        }
    });

    let job = FrameJob {
        data: Arc::clone(data),
        hdr: *hdr,
        chdr,
        chdr_inter,
        refs: refs.cloned(),
        prev_mvs,
        prev_seg,
        use_prev_frame_mvs,
        tiles,
        tile_cols: tile_cols as usize,
        mi_rows,
        mi_cols,
        sb64_rows,
        sb64_cols,
        ssx,
        ssy,
        bit_depth,
        lossless,
        geom,
        lf,
    };
    Ok(launch_frame(job, pool, scratch, reuse))
}

/// Decode a multi-frame VP9 stream (keyframe followed by inter / P-frames)
/// from a slice of per-frame payloads, threading the §8.10 reference
/// buffers + §6.5 previous-frame motion field across frames.
///
/// `frames` is a list of frame payloads in decode order (each: §6.2
/// uncompressed header + §6.3 compressed header + §6.4 tile data — e.g.
/// the body of one IVF packet). The first frame must be a key frame.
/// Returns one [`Vp9DecodedFrame`] per *shown* frame (frames with
/// `show_frame == 0` are decoded as references but not emitted; a
/// `show_existing_frame` packet re-emits a stored frame).
///
/// Inter frames reconstruct end-to-end via the §8.5.2 inter prediction
/// process; single + compound prediction, segmentation, and the §8.10
/// reference-buffer update are all wired.
pub fn decode_vp9_sequence(frames: &[&[u8]]) -> Result<Vec<Vp9DecodedFrame>, Error> {
    decode_vp9_sequence_with(frames, &EXEC_SERIAL)
}

/// [`decode_vp9_sequence`] under a caller-supplied
/// [`ExecutionContext`] — the threading authority for the §6.4 tile
/// walk of every frame.
///
/// Multi-tile-column frames decode their tile columns on up to
/// `exec.effective_workers( tileCols )` workers (the §9.2.4 multi-coder
/// path: one §9.2 boolean decoder live per concurrent tile), and the
/// output is **byte-identical** to [`decode_vp9_sequence`] — tile
/// columns are independent by construction (§6.4.4
/// `AvailL = MiCol > MiColStart`, SB64-aligned tile boundaries, §6.5
/// candidate scans clamped to the tile window), so the merge of the
/// per-column decodes reproduces the serial raster exactly. Frames with
/// a single tile column (and every frame under
/// [`ExecutionContext::serial`], the [`decode_vp9_sequence`] default)
/// take the plain serial path.
pub fn decode_vp9_sequence_with(
    frames: &[&[u8]],
    exec: &ExecutionContext,
) -> Result<Vec<Vp9DecodedFrame>, Error> {
    let mut decoder = Vp9SequenceDecoder::new();
    decoder.set_execution_context(exec);
    let mut out = Vec::new();
    for &frame in frames {
        if let Some(f) = decoder.push_frame(frame)? {
            out.push(f);
        }
    }
    Ok(out)
}

/// Incremental multi-frame VP9 stream decoder — the stateful form of
/// [`decode_vp9_sequence`], threading the identical cross-frame state
/// one coded frame at a time.
///
/// Feed coded frames in decode order via [`Self::push_frame`] (the
/// first must be a key frame). The struct owns every piece of §-defined
/// persistent decode state:
///
/// * the §8.10 reference buffers + `FrameStore[ ]` display copies
///   (`refresh_frame_flags` updates, §8.9 `show_existing_frame`);
/// * the §6.5 previous-frame motion field feeding the §7.2.6
///   `UsePrevFrameMvs` derivation;
/// * the §6.1.2 / §7.2 `FrameContext[ 4 ]` entropy probability banks
///   (`load_probs( )` / `save_probs( )`, including the full §6.1.2
///   `refresh_probs( )` backward adaptation on non-frame-parallel
///   streams);
/// * the §7.2.10 persistent segmentation feature table, the §7.2.8
///   persistent loop-filter deltas, and the §6.4.14
///   `PrevSegmentIds[ ][ ]` map.
///
/// A caller feeding the same frames in the same order gets exactly the
/// [`decode_vp9_sequence`] outputs (that function is a thin loop over
/// this struct). Superframes (Annex B) must be split *before* pushing
/// — see [`crate::split_superframe`].
pub struct Vp9SequenceDecoder {
    // Every piece of §-defined persistent decode state (see below).
    st: SeqState,
    // The threading authority for the frame batches (serial until told
    // otherwise, per the framework contract).
    exec: ExecutionContext,
    // The last decoded frame's §9.3.4 count bank — the encoder-side
    // decoder-mirror tests compare the writer's counts against it.
    last_counts: Option<Box<crate::prob_adapt::FrameCounts>>,
    // The persistent worker pool the frame batches run on (created on
    // the first frame that can use more than one thread, sized by the
    // ExecutionContext budget and the frame's useful parallelism).
    pool: Option<WorkerPool>,
    // Tile-column working storage reused across frames.
    scratch: ScratchPool,
    // Every frame this decoder produced that may still be referenced;
    // one nobody else holds any more donates its band storage to the
    // next frame.
    recycle: Vec<Arc<FrameBuf>>,
    // Frames submitted and not yet completed, oldest first.
    inflight: VecDeque<Pending>,
    // Completed shown frames (with the caller's tag), in output order.
    ready: VecDeque<(Arc<FrameBuf>, i64)>,
}

/// The §-defined persistent decode state a frame's header processing
/// reads and updates.
#[derive(Clone)]
struct SeqState {
    bufs: RefBuffers,
    // The most-recently-decoded color config + frame dims (for the §6.2
    // inter header's inherited color config + §8.10 ref dims).
    last_color: Option<ColorConfig>,
    // The previous decoded frame (its §6.4.4 mode info rows) + its
    // size + show_frame, for the §6.5 prev-MV scan + §7.2.6
    // UsePrevFrameMvs derivation.
    prev_state: Option<(Arc<FrameBuf>, u32, u32, bool)>,
    // §6.1.2 / §7.2 `FrameContext[ 4 ]` — the persistent entropy
    // probability banks `load_probs( )` / `save_probs( )` thread across
    // frames. Initialised to the §10.5 defaults; a key / intra-only /
    // error-resilient frame's §7.2 `setup_past_independence( )` resets
    // the bank(s) again before the frame folds its own compressed-header
    // forward updates on top.
    frame_contexts: [FrameContext; 4],
    // §7.2 `LastFrameType` — the previous (non-show-existing) frame's
    // `frame_type`, feeding the §8.4.3 `updateFactor` selection. The
    // uncompressed-header listing sets `LastFrameType = frame_type`
    // BEFORE reading the new `frame_type`, and a `show_existing_frame`
    // packet returns earlier, leaving it untouched.
    last_frame_type_was_key: bool,
    // §7.2.10 / §7.2 persistent segmentation feature table: per §7.2.10
    // "segmentation_update_data equal to 0 indicates that the
    // segmentation parameters should keep their existing values", so
    // `FeatureEnabled` / `FeatureData` / `abs_or_delta_update` carry
    // across frames until a `segmentation_update_data == 1` frame
    // rewrites them or §7.2 `setup_past_independence( )` (key /
    // intra-only / error-resilient frame) clears them to zero. The
    // per-frame header parse alone cannot see this state, so it is
    // threaded here and substituted into the parsed header below.
    persist_feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    persist_feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    persist_abs_or_delta: bool,
    // §7.2.8 persistent loop-filter deltas: `loop_filter_delta_update ==
    // 0` (or an individual update flag of 0) keeps the existing
    // `loop_filter_ref_deltas` / `loop_filter_mode_deltas`; §7.2
    // `setup_past_independence( )` resets them to the defaults.
    persist_ref_deltas: [i8; 4],
    persist_mode_deltas: [i8; 2],
    // §6.4.14 `PrevSegmentIds[ ][ ]` — the persistent segmentation map a
    // §6.4.12 `temporal_update` / `update_map == 0` frame predicts from:
    // the last frame with `segmentation_update_map == 1` (its mode-info
    // rows), with its size. §8.1 step 3 only refreshes it after such a
    // frame; a frame that reuses the map leaves it pointing at the *last
    // map-bearing* frame, NOT its own (possibly Min-collapsed)
    // `SegmentIds`. Tracking it separately from the per-frame
    // `prev_state` is what keeps a run of `update_map == 0` inter frames
    // reading the original map.
    prev_segment_ids_map: Option<(Arc<FrameBuf>, u32, u32)>,
}

/// An intra frame's forward-updated `(tx_probs, skip_prob)` — the
/// tables its §6.1.2 `refresh_probs( )` keeps.
type IntraForward = ([[[u8; 3]; 2]; 4], [u8; 3]);

/// The §6.1.2 `refresh_probs( )` backward adaptation a submitted frame
/// owes the bank it decoded from, applied once its counts exist.
struct PendingAdapt {
    idx: usize,
    base: FrameContext,
    /// `Some( (tx_probs, skip_prob) )` for an intra frame (the forward
    /// updates `load_probs( )` keeps), `None` for an inter frame.
    intra_fwd: Option<IntraForward>,
    last_frame_type_was_key: bool,
    interp_switchable: bool,
    tx_mode_select: bool,
    allow_high_precision_mv: bool,
}

/// A submitted coded frame.
struct Pending {
    /// The coded frame (replayed if an earlier frame fails).
    payload: Arc<[u8]>,
    /// The caller's tag for the output (the packet `pts`).
    tag: i64,
    kind: PendingKind,
}

enum PendingKind {
    /// A §8.9 `show_existing_frame` re-display of a stored frame.
    ShowExisting(Arc<FrameBuf>),
    Decode {
        inflight: InFlight,
        shown: bool,
        /// The decoder state as it was before this frame's decode
        /// updates (restored if the decode fails).
        snapshot: Box<SeqState>,
        adapt: Option<Box<PendingAdapt>>,
    },
}

impl std::fmt::Debug for Vp9SequenceDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The banks/buffers are megabyte-scale working state; summarise.
        f.debug_struct("Vp9SequenceDecoder")
            .field("last_color", &self.st.last_color)
            .field("has_prev_frame", &self.st.prev_state.is_some())
            .field("persist_ref_deltas", &self.st.persist_ref_deltas)
            .field("persist_mode_deltas", &self.st.persist_mode_deltas)
            .field("in_flight", &self.inflight.len())
            .finish_non_exhaustive()
    }
}

impl Default for Vp9SequenceDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Vp9SequenceDecoder {
    fn drop(&mut self) {
        // Let every submitted batch finish before the pool goes away.
        self.drain();
    }
}

impl Vp9SequenceDecoder {
    /// Fresh decoder state: empty §8.10 buffers, default §10.5
    /// probability banks, default §7.2.8 deltas, no previous frame.
    pub fn new() -> Self {
        Self {
            st: SeqState {
                bufs: RefBuffers::new(),
                last_color: None,
                prev_state: None,
                frame_contexts: Default::default(),
                last_frame_type_was_key: false,
                persist_feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
                persist_feature_data: [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS],
                persist_abs_or_delta: false,
                persist_ref_deltas: DEFAULT_LOOP_FILTER_REF_DELTAS,
                persist_mode_deltas: DEFAULT_LOOP_FILTER_MODE_DELTAS,
                prev_segment_ids_map: None,
            },
            exec: ExecutionContext::serial(),
            last_counts: None,
            pool: None,
            scratch: ScratchPool::default(),
            recycle: Vec::new(),
            inflight: VecDeque::new(),
            ready: VecDeque::new(),
        }
    }

    /// The §7.2 `FrameContext[ idx ]` bank as the decoder holds it after
    /// the last pushed frame (encoder-mirror tests).
    #[cfg(test)]
    pub(crate) fn frame_context(&self, idx: usize) -> &FrameContext {
        &self.st.frame_contexts[idx & 3]
    }

    /// The last decoded frame's §9.3.4 counts (`None` before the first
    /// decoded frame and after a `show_existing_frame` packet).
    #[cfg(test)]
    pub(crate) fn last_frame_counts(&self) -> Option<&crate::prob_adapt::FrameCounts> {
        self.last_counts.as_deref()
    }

    /// Set the threading authority for every subsequently pushed frame
    /// (the framework `set_execution_context` contract: serial until
    /// told otherwise). The frame batches run on a persistent pool of
    /// `exec.effective_workers( useful )` threads, where `useful` is a
    /// frame's tile columns plus superblock rows. Decoded output is
    /// byte-identical at every budget — see [`decode_vp9_sequence_with`].
    pub fn set_execution_context(&mut self, exec: &ExecutionContext) {
        self.drain();
        self.exec = exec.clone();
        // Rebuilt at the new budget on the next frame.
        self.pool = None;
    }

    /// Decode one coded frame (one §6.2 uncompressed header + §6.3
    /// compressed header + §6.4 tile data — e.g. the body of one IVF
    /// packet, after any Annex B superframe split), threading every
    /// piece of persistent state exactly as [`decode_vp9_sequence`]
    /// does.
    ///
    /// Returns `Ok( Some( frame ) )` when the pushed frame is *shown*
    /// (`show_frame == 1`, or a §8.9 `show_existing_frame` re-display)
    /// and `Ok( None )` for a hidden reference-only frame.
    pub fn push_frame(&mut self, frame: &[u8]) -> Result<Option<Vp9DecodedFrame>, Error> {
        Ok(self.push_frame_buf(frame)?.map(|f| f.to_decoded_frame()))
    }

    /// [`Self::push_frame`] returning the shown frame's decoded storage.
    pub(crate) fn push_frame_buf(&mut self, frame: &[u8]) -> Result<Option<Arc<FrameBuf>>, Error> {
        self.drain();
        self.ready.clear();
        self.submit(frame, 0, 0)?;
        Ok(self.ready.pop_front().map(|(f, _)| f))
    }

    /// Submit one coded frame for decoding, completing the oldest
    /// submitted frames until at most `max_in_flight` remain. Completed
    /// shown frames queue (tagged with `tag`) for [`Self::pop_ready`].
    /// Header errors return immediately; a decode error returns from the
    /// call that completes the failing frame — the decoder state is then
    /// exactly as if that frame had been pushed alone and failed, and the
    /// frames submitted after it are replayed on top of it.
    pub(crate) fn submit(
        &mut self,
        frame: &[u8],
        tag: i64,
        max_in_flight: usize,
    ) -> Result<(), Error> {
        self.last_counts = None;
        // The bank this frame loads from may still owe the previous
        // frame's backward adaptation.
        self.resolve_adaptation()?;
        let payload: Arc<[u8]> = Arc::from(frame);
        let pending = self.start_frame(payload, tag)?;
        self.inflight.push_back(pending);
        while self.inflight.len() > max_in_flight {
            self.complete_front()?;
        }
        Ok(())
    }

    /// Complete every submitted frame (errors are dropped: the caller
    /// asked for a full stop).
    pub(crate) fn drain(&mut self) {
        while !self.inflight.is_empty() {
            let _ = self.complete_front();
        }
    }

    /// Complete the oldest submitted frame, if any.
    pub(crate) fn finish_one(&mut self) -> Option<Result<(), Error>> {
        if self.inflight.is_empty() {
            None
        } else {
            Some(self.complete_front())
        }
    }

    /// The oldest completed shown frame and its tag.
    pub(crate) fn pop_ready(&mut self) -> Option<(Arc<FrameBuf>, i64)> {
        self.ready.pop_front()
    }

    /// Apply the latest submitted frame's owed backward adaptation (its
    /// counts complete first); a failure there completes the queue up to
    /// the failing frame.
    fn resolve_adaptation(&mut self) -> Result<(), Error> {
        let Some(Pending {
            kind: PendingKind::Decode {
                inflight, adapt, ..
            },
            ..
        }) = self.inflight.back_mut()
        else {
            return Ok(());
        };
        if adapt.is_none() {
            return Ok(());
        }
        match inflight.counts() {
            Ok(counts) => {
                let a = adapt.take().expect("checked");
                apply_adaptation(&mut self.st.frame_contexts, a, counts);
                Ok(())
            }
            Err(_) => {
                // Complete in order: the first failing frame reports.
                while !self.inflight.is_empty() {
                    self.complete_front()?;
                }
                Ok(())
            }
        }
    }

    /// Complete the front frame: wait for it, queue its output, and on
    /// failure restore the state it started from and replay the frames
    /// queued after it.
    fn complete_front(&mut self) -> Result<(), Error> {
        let Some(front) = self.inflight.pop_front() else {
            return Ok(());
        };
        match front.kind {
            PendingKind::ShowExisting(f) => {
                self.ready.push_back((f, front.tag));
                Ok(())
            }
            PendingKind::Decode {
                mut inflight,
                shown,
                snapshot,
                adapt,
            } => {
                if let (Some(pool), Some(batch)) = (self.pool.as_ref(), inflight.batch.as_ref()) {
                    pool.wait(batch);
                }
                let result = match inflight.counts().map(|_| ()) {
                    Ok(()) if inflight.frame.wait_complete() => Ok(()),
                    Ok(()) => Err(Error::InvalidBitstream),
                    Err(e) => Err(e),
                };
                match result {
                    Ok(()) => {
                        let counts = inflight.take_counts();
                        if let Some(a) = adapt {
                            apply_adaptation(&mut self.st.frame_contexts, a, &counts);
                        }
                        self.last_counts = Some(counts);
                        if shown {
                            self.ready
                                .push_back((Arc::clone(&inflight.frame), front.tag));
                        }
                        Ok(())
                    }
                    Err(e) => {
                        // Wait out everything after it, then rebuild the
                        // state the failing frame left (as a lone push
                        // would) and replay the later frames on it.
                        let later: Vec<Pending> = self.inflight.drain(..).collect();
                        for p in &later {
                            if let PendingKind::Decode { inflight, .. } = &p.kind
                                && let (Some(pool), Some(batch)) =
                                    (self.pool.as_ref(), inflight.batch.as_ref())
                            {
                                pool.wait(batch);
                            }
                        }
                        self.st = *snapshot;
                        self.last_counts = None;
                        for p in later {
                            // Errors of replayed frames are dropped: the
                            // first failure is the one reported.
                            if let Ok(pending) = self.start_frame(Arc::clone(&p.payload), p.tag) {
                                self.inflight.push_back(pending);
                                let _ = self.complete_front();
                            }
                        }
                        Err(e)
                    }
                }
            }
        }
    }

    /// Process one coded frame's headers, update the persistent state,
    /// and launch its decode.
    fn start_frame(&mut self, payload: Arc<[u8]>, tag: i64) -> Result<Pending, Error> {
        let frame: &[u8] = &payload;
        let Self {
            st,
            exec,
            pool,
            scratch,
            recycle,
            ..
        } = self;
        // Peek the uncompressed header (with inherited ref state so inter
        // frames parse).
        let ref_dims: Vec<(u32, u32)> = (0..8)
            .map(|i| (st.bufs.ref_frame_width(i), st.bufs.ref_frame_height(i)))
            .collect();
        let ref_state = st.last_color.map(|cc| RefFrameState {
            ref_dims: &ref_dims,
            color_config: cc,
        });
        let mut hdr = parse_uncompressed_header_with_refs(frame, ref_state)?;

        if hdr.show_existing_frame {
            // §8.9: output `FrameStore[ frame_to_show_map_idx ]` verbatim.
            // Per §8.10 step 1 every decoded frame — hidden alt-ref
            // frames included — writes the slots `refresh_frame_flags`
            // names, and the slot keeps that frame's own visible planes,
            // dimensions, subsampling and bit depth, so the displayable
            // frame is rebuilt from it.
            let idx = hdr.frame_to_show_map_idx.ok_or(Error::InvalidBitstream)? as usize;
            if idx >= crate::ref_buffer::NUM_REF_FRAMES || !st.bufs.is_initialised(idx) {
                return Err(Error::InvalidBitstream);
            }
            let f = Arc::clone(st.bufs.slot(idx).frame.as_ref().expect("initialised slot"));
            // §7.2.6 NOTE: compute_image_size is not invoked for a
            // show_existing_frame packet, so it leaves `prev_state` (the
            // §7.2.6 "previous invocation" record) untouched.
            return Ok(Pending {
                payload,
                tag,
                kind: PendingKind::ShowExisting(f),
            });
        }

        let frame_is_intra = matches!(hdr.frame_type, FrameType::KeyFrame) || hdr.intra_only;

        // §7.2 setup_past_independence( ): a key / intra-only /
        // error-resilient frame clears the persistent segmentation
        // feature table and resets the loop-filter deltas to the
        // defaults BEFORE its own loop_filter_params( ) /
        // segmentation_params( ) apply on top.
        if frame_is_intra || hdr.error_resilient_mode {
            st.persist_feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
            st.persist_feature_data = [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS];
            st.persist_abs_or_delta = false;
            st.persist_ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
            st.persist_mode_deltas = DEFAULT_LOOP_FILTER_MODE_DELTAS;
            // §7.2: `ref_frame_sign_bias[ i ] = 0 for i = 0..3`. The
            // §6.2 ref-frames loop consumes the sign-bias *bits* before
            // setup_past_independence( ) runs (the reset sits after the
            // ref-frames section in the §6.2 listing), so on an
            // error-resilient inter frame the coded values are dead and
            // the effective biases are all zero — which in turn makes
            // §6.3.12 `compoundReferenceAllowed` 0: an error-resilient
            // frame can never code compound prediction, and the §6.5
            // scan never sign-flips candidates. (Found by black-box
            // reference-decoder divergence on a self-encoded
            // error-resilient compound stream, round 418.)
            hdr.ref_frame_sign_bias = [false; 3];
        }

        // §7.2.8: a `loop_filter_delta_update == 1` frame folds the
        // per-delta updates it carries into the persistent deltas (an
        // individual absent update keeps the prior value); every frame
        // then decodes with the fully-resolved persistent values.
        for (slot, delta) in st
            .persist_ref_deltas
            .iter_mut()
            .zip(hdr.loop_filter.ref_deltas)
        {
            if let Some(d) = delta {
                *slot = d;
            }
        }
        for (slot, delta) in st
            .persist_mode_deltas
            .iter_mut()
            .zip(hdr.loop_filter.mode_deltas)
        {
            if let Some(d) = delta {
                *slot = d;
            }
        }
        hdr.loop_filter.ref_deltas = st.persist_ref_deltas.map(Some);
        hdr.loop_filter.mode_deltas = st.persist_mode_deltas.map(Some);

        // §7.2.10: `segmentation_update_data == 0` keeps the existing
        // feature table (even across `segmentation_enabled == 0`
        // frames); `segmentation_update_data == 1` rewrites it.
        if hdr.segmentation.update_data {
            st.persist_feature_enabled = hdr.segmentation.feature_enabled;
            st.persist_feature_data = hdr.segmentation.feature_data;
            st.persist_abs_or_delta = hdr.segmentation.abs_or_delta_update;
        } else {
            hdr.segmentation.feature_enabled = st.persist_feature_enabled;
            hdr.segmentation.feature_data = st.persist_feature_data;
            hdr.segmentation.abs_or_delta_update = st.persist_abs_or_delta;
        }

        let lossless = hdr.quantization.lossless;
        let ch_start = hdr.uncompressed_header_size_bytes;
        let ch_size = hdr.header_size_in_bytes as usize;
        if ch_size == 0 {
            return Err(Error::InvalidBitstream);
        }
        let ch_end = ch_start
            .checked_add(ch_size)
            .ok_or(Error::InvalidBitstream)?;
        if frame.len() < ch_end {
            return Err(Error::UnexpectedEof);
        }

        // §6.5 / §6.4.14 previous-frame snapshot (same-dimension only).
        // §6.5 MV/ref prediction reads the immediately-preceding decoded
        // frame (and only when it was shown — §7.2.6 (c)); §6.4.14
        // segment prediction reads the separately-tracked
        // `prev_segment_ids_map` (last `update_map == 1` frame), which a
        // hidden predecessor does NOT invalidate (§8.1 step 3 has no
        // show_frame condition). §7.2 setup_past_independence( ) on an
        // error-resilient frame clears `PrevSegmentIds[ ][ ]` to zero
        // (falling through to the previous frame's ids would resurrect
        // the cleared state). Without a same-size previous frame both
        // snapshots are zero (§7.2.6 UsePrevFrameMvs = 0, §6.4.14
        // PrevSegmentIds cleared).
        let prev = st
            .prev_state
            .as_ref()
            .filter(|(_, pw, ph, _)| *pw == hdr.frame_width && *ph == hdr.frame_height);
        let prev_shown = prev.is_some_and(|(_, _, _, shown)| *shown);
        let prev_mvs = prev.map(|(f, ..)| Arc::clone(f));
        let prev_seg = prev.and_then(|(f, ..)| {
            if hdr.error_resilient_mode {
                None
            } else {
                Some(
                    st.prev_segment_ids_map
                        .as_ref()
                        .filter(|(_, mw, mh)| *mw == hdr.frame_width && *mh == hdr.frame_height)
                        .map_or_else(|| Arc::clone(f), |(m, ..)| Arc::clone(m)),
                )
            }
        });

        // The frame batch's pool: sized by the budget and the frame's
        // useful parallelism (tile columns + loop-filter rows), built
        // once and kept for the decoder's lifetime.
        let useful = {
            let mi_rows = (hdr.frame_height + 7) >> 3;
            (1usize << hdr.tile_info.tile_cols_log2) + mi_rows.div_ceil(8) as usize
        };
        let helpers = exec.effective_workers(useful) - 1;
        if helpers > 0 && pool.as_ref().is_none_or(|p| p.helpers() < helpers) {
            *pool = Some(WorkerPool::new(helpers));
        }
        // A frame no slot / caller holds any more lends its storage.
        let reuse: Option<Bands> = recycle
            .iter()
            .position(|f| Arc::strong_count(f) == 1)
            .and_then(|i| Arc::try_unwrap(recycle.swap_remove(i)).ok())
            .map(FrameBuf::into_bands);

        // §6.2 setup_past_independence + reset_frame_context: a key /
        // intra-only / error-resilient frame decodes free of prior-frame
        // entropy state. §7.2 resets the working probability tables to the
        // §10.5 defaults, then §6.2 saves those defaults into the frame
        // context(s) named by `reset_frame_context` (all four on a
        // keyframe / error-resilient frame), and finally forces
        // `frame_context_idx = 0`.
        let frame_context_idx = if frame_is_intra || hdr.error_resilient_mode {
            let reset_all = matches!(hdr.frame_type, FrameType::KeyFrame)
                || hdr.error_resilient_mode
                || hdr.reset_frame_context == 3;
            if reset_all {
                st.frame_contexts = Default::default();
            } else if hdr.reset_frame_context == 2 {
                st.frame_contexts[hdr.frame_context_idx as usize] = FrameContext::default();
            }
            0usize
        } else {
            hdr.frame_context_idx as usize
        };

        // §6.1.2 load_probs( frame_context_idx ): the compressed-header
        // forward updates fold onto the loaded bank, not the static
        // defaults.
        let base_ctx = st.frame_contexts[frame_context_idx].clone();

        // §6.1.2 refresh_probs( ) gate: with error_resilient_mode == 0
        // and frame_parallel_decoding_mode == 0, the §8.4 backward
        // adaptation runs after decode_tiles( ) — load_probs( idx )
        // reloads the PRE-frame bank (discarding this frame's forward
        // updates on every table except tx_probs / skip_prob, which
        // load_probs( ) does not cover), the counts adapt it, and
        // save_probs( idx ) persists the result (`PendingAdapt`, applied
        // once the counts exist). In parallel / error-resilient mode
        // save_probs( ) persists the forward-updated working tables
        // instead.
        let non_parallel = !hdr.error_resilient_mode && !hdr.frame_parallel_decoding_mode;
        let owes_adaptation = hdr.refresh_frame_context && non_parallel;

        let (chdr, adapt) = if frame_is_intra {
            let chdr_parsed =
                parse_compressed_header_with_ctx(&frame[ch_start..ch_end], lossless, &base_ctx)?;
            // §6.1.2 save_probs( frame_context_idx ) — parallel-mode
            // path: no backward adaptation, persist the forward-updated
            // tables directly.
            if hdr.refresh_frame_context && !non_parallel {
                st.frame_contexts[frame_context_idx].apply_intra(&chdr_parsed);
            }
            // The forward-updated tx_probs / skip_prob survive an intra
            // frame's refresh_probs( ) (load_probs( ) skips them and
            // FrameIsIntra bypasses load_probs2( )).
            let adapt = owes_adaptation.then(|| {
                Box::new(PendingAdapt {
                    idx: frame_context_idx,
                    base: base_ctx.clone(),
                    intra_fwd: Some((chdr_parsed.tx_probs, chdr_parsed.skip_prob)),
                    last_frame_type_was_key: st.last_frame_type_was_key,
                    interp_switchable: false,
                    tx_mode_select: false,
                    allow_high_precision_mv: false,
                })
            });
            (ChdrKind::Intra(Box::new(chdr_parsed)), adapt)
        } else {
            let inputs = Vp9CompressedHeaderInterInputs {
                interpolation_filter_is_switchable: hdr.interpolation_filter == SWITCHABLE_FILTER,
                ref_frame_sign_bias: RefFrameSignBias::from_inter_biases(
                    hdr.ref_frame_sign_bias[0] as u8,
                    hdr.ref_frame_sign_bias[1] as u8,
                    hdr.ref_frame_sign_bias[2] as u8,
                ),
                allow_high_precision_mv: hdr.allow_high_precision_mv,
            };
            let chdr_parsed = parse_compressed_header_inter_with_ctx(
                &frame[ch_start..ch_end],
                lossless,
                inputs,
                &base_ctx,
            )?;
            let tx_mode_select =
                chdr_parsed.intra.tx_mode == crate::compressed::TxMode::TxModeSelect;
            if hdr.refresh_frame_context && !non_parallel {
                st.frame_contexts[frame_context_idx].apply_inter(&chdr_parsed);
            }
            let adapt = owes_adaptation.then(|| {
                Box::new(PendingAdapt {
                    idx: frame_context_idx,
                    base: base_ctx.clone(),
                    intra_fwd: None,
                    last_frame_type_was_key: st.last_frame_type_was_key,
                    interp_switchable: hdr.interpolation_filter == SWITCHABLE_FILTER,
                    tx_mode_select,
                    allow_high_precision_mv: hdr.allow_high_precision_mv,
                })
            });
            (ChdrKind::Inter(Box::new(chdr_parsed)), adapt)
        };

        // The state a failing decode leaves (as before this frame's
        // decode updates).
        let snapshot = Box::new(st.clone());
        let inflight = launch_single_frame(
            &payload,
            &hdr,
            ch_end,
            chdr,
            Some(&st.bufs),
            (prev_mvs, prev_seg, prev_shown),
            pool.as_ref(),
            scratch,
            reuse,
        )?;
        let decoded = Arc::clone(inflight.frame());

        // §7.2: LastFrameType tracks the just-decoded frame's type for
        // the next frame's §8.4.3 updateFactor selection.
        st.last_frame_type_was_key = matches!(hdr.frame_type, FrameType::KeyFrame);

        // §8.10 reference frame update: the slots `refresh_frame_flags`
        // names now hold the (loop-filtered) decoded frame — later
        // frames read its rows as they are finished.
        st.bufs.update(hdr.refresh_frame_flags, &decoded);
        // Keep the frames still referenced plus a couple of free ones
        // (their storage serves the next frames).
        let mut free = 0;
        recycle.retain(|f| {
            let keep = Arc::strong_count(f) > 1 || free < 2;
            free += usize::from(Arc::strong_count(f) == 1);
            keep
        });
        recycle.push(Arc::clone(&decoded));

        // Carry inherited color config (§6.2: an intra frame establishes
        // it; an inter frame before any intra frame is malformed but we
        // tolerate it by adopting its own config) + the prev-frame motion
        // field.
        if frame_is_intra || st.last_color.is_none() {
            st.last_color = Some(hdr.color_config);
        }
        // §8.1 step 3: refresh PrevSegmentIds only after a frame that
        // built a fresh map (`segmentation_enabled && update_map`). A
        // frame reusing the map leaves the persistent copy untouched so
        // the *next* frame still predicts from the original.
        if hdr.segmentation.enabled && hdr.segmentation.update_map {
            st.prev_segment_ids_map =
                Some((Arc::clone(&decoded), hdr.frame_width, hdr.frame_height));
        }
        st.prev_state = Some((decoded, hdr.frame_width, hdr.frame_height, hdr.show_frame));

        // §8.9: a hidden (`show_frame == 0`) frame is decoded as a
        // reference but not emitted.
        Ok(Pending {
            payload,
            tag,
            kind: PendingKind::Decode {
                inflight,
                shown: hdr.show_frame,
                snapshot,
                adapt,
            },
        })
    }
}

/// §6.1.2 `refresh_probs( )`: adapt the pre-frame bank with the frame's
/// §9.3.4 counts and `save_probs( idx )` the result.
fn apply_adaptation(
    banks: &mut [FrameContext; 4],
    a: Box<PendingAdapt>,
    counts: &crate::prob_adapt::FrameCounts,
) {
    let PendingAdapt {
        idx,
        base: mut work,
        intra_fwd,
        last_frame_type_was_key,
        interp_switchable,
        tx_mode_select,
        allow_high_precision_mv,
    } = *a;
    match intra_fwd {
        Some((tx_probs, skip_prob)) => {
            // FrameIsIntra == 1 arm: load_probs( idx ) ;
            // adapt_coef_probs( ) ; save_probs( idx ).
            work.tx_probs = tx_probs;
            work.skip_prob = skip_prob;
            crate::prob_adapt::adapt_coef_probs(
                &mut work.coef_probs,
                &counts.token,
                &counts.more_coefs,
                true,
                last_frame_type_was_key,
            );
        }
        None => {
            // FrameIsIntra == 0 arm: load_probs( idx ) ;
            // adapt_coef_probs( ) ; load_probs2( idx ) ;
            // adapt_noncoef_probs( ) ; save_probs( idx ). load_probs( ) +
            // load_probs2( ) together restore the whole pre-frame bank,
            // so the frame's forward updates never persist on this path.
            crate::prob_adapt::adapt_coef_probs(
                &mut work.coef_probs,
                &counts.token,
                &counts.more_coefs,
                false,
                last_frame_type_was_key,
            );
            crate::prob_adapt::adapt_noncoef_probs(
                &mut work,
                &counts.noncoef,
                interp_switchable,
                tx_mode_select,
                allow_high_precision_mv,
            );
        }
    }
    banks[idx] = work;
}

/// `SWITCHABLE = 4` per §7.4.10 — the frame-level interpolation-filter
/// sentinel that gates the §6.4.16 per-block `interp_filter` tree read
/// and the §6.3.10 `read_interp_filter_probs( )` sweep.
const SWITCHABLE_FILTER: u8 = 4;

/// Test-only helper: decode a fixed list of keyframe intra MI blocks from
/// `coder`, mirroring `decode_block_intra( )` for each, and return the
/// per-block recovered [`DecodedBlockResult`]-equivalent values read back
/// from the §6.4.4 frame-wide arrays.
///
/// Used by the encoder block-writer round-trip tests to confirm a written
/// block stream decodes back to the spec'd mode info + residual through
/// the real decoder path (default §6.3 probability banks, single 4:2:0
/// tile, segmentation disabled).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_keyframe_blocks_for_test(
    coder: &mut BoolCoder<'_>,
    mi_rows: u32,
    mi_cols: u32,
    tx_mode: crate::compressed::TxMode,
    blocks: Vec<(u32, u32, u8)>,
    y: &mut Plane,
    u: &mut Plane,
    v: &mut Plane,
) -> Result<Vec<DecodedBlockResult>, Error> {
    use crate::coef_probs::DEFAULT_COEF_PROBS;
    use crate::compressed::{DEFAULT_SKIP_PROB, DEFAULT_TX_PROBS};

    let chdr = Vp9CompressedHeader {
        tx_mode,
        tx_probs: DEFAULT_TX_PROBS,
        coef_probs: DEFAULT_COEF_PROBS,
        skip_prob: DEFAULT_SKIP_PROB,
    };
    let seg = SegmentationParams::default_disabled();
    let quant = QuantizationParams {
        base_q_idx: 64,
        delta_q_y_dc: 0,
        delta_q_uv_dc: 0,
        delta_q_uv_ac: 0,
        lossless: false,
    };
    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
    ];
    let mut token_cache = [0u8; 1024];
    let mut tok_buf = [0i64; 1024];
    let mut pred_scratch = crate::inter_pred::PredScratch::new_boxed();
    let mut counts = crate::prob_adapt::FrameCounts::new_boxed();

    let mut results = Vec::with_capacity(blocks.len());
    {
        let mut bd = BlockDecoder {
            seg: &seg,
            quant: &quant,
            chdr: &chdr,
            mi_rows,
            mi_cols,
            mi_col_start: 0,
            mi_col_end: mi_cols,
            subsampling_x: true,
            subsampling_y: true,
            bit_depth: 8,
            lossless: false,
            state: &mut state,
            sink: Sink::Now {
                y,
                u,
                v,
                pred: &mut pred_scratch,
            },
            nz: &mut nz,
            token_cache: &mut token_cache,
            tok_buf: &mut tok_buf,
            counts: &mut counts,
            inter: None,
        };
        for &(r, c, subsize) in &blocks {
            bd.decode_block_intra(coder, r, c, subsize)?;
        }
    }

    for &(r, c, _subsize) in &blocks {
        results.push(DecodedBlockResult {
            skip: state.get_skip(r, c).unwrap_or(0) != 0,
            tx_size: state.get_tx_size(r, c).unwrap_or(0),
            y_mode: state.get_y_mode(r, c).unwrap_or(0),
            segment_id: state.get_segment_id(r, c).unwrap_or(0),
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            is_inter: false,
            eob_total: 0,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [
                state.get_sub_mode(r, c, 0).unwrap_or(0),
                state.get_sub_mode(r, c, 1).unwrap_or(0),
                state.get_sub_mode(r, c, 2).unwrap_or(0),
                state.get_sub_mode(r, c, 3).unwrap_or(0),
            ],
        });
    }
    Ok(results)
}

/// The mode-info products a written inter block decodes back to — used by
/// the `inter_block_writer` round-trip tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecoveredInterBlock {
    pub skip: bool,
    pub tx_size: u32,
    pub y_mode: u8,
    pub segment_id: u8,
    pub ref_frame: [i32; 2],
    pub interp_filter: u8,
    pub mv: [[i32; 2]; 2],
    /// The full §6.4.16 `BlockMvs[ refList ][ 4 ]` array — per-cell for a
    /// sub-8x8 block, replicated for `MiSize >= BLOCK_8X8`.
    pub block_mvs: [[[i32; 2]; 4]; 2],
}

/// Decode a fixed list of **inter** blocks (any `MiSize`, sub-8x8
/// included) at the mode-info + residual-token level (no §8.5.2 motion
/// compensation), for the `inter_block_writer` round-trip tests.
///
/// Mirrors `decode_block_inter`'s §6.4.11 argument assembly exactly — the
/// same neighbour bundles, the same `InterFrameModeArgs`, the shared §6.5
/// `find_mv_refs` / `find_best_ref_mvs` (run inside `inter_frame_mode_info`)
/// — then advances the coder over the §6.4.21 residual by running the real
/// `tokens( )` decoder per coded transform block (DCT_DCT, `is_inter`),
/// and fans the products into the frame state via `decode_block_apply` so
/// the next block reads the right neighbour context. The pixel
/// reconstruction the full decoder runs produces no bitstream syntax, so
/// it is omitted; the recovered mode info is bit-exact with the writer's
/// input when the round-trip holds.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn decode_inter_blocks_for_test(
    data: &[u8],
    mi_rows: u32,
    mi_cols: u32,
    ctx: &crate::compressed::FrameContext,
    tx_mode: crate::compressed::TxMode,
    reference_mode: crate::compressed::ReferenceMode,
    comp_config: CompoundReferenceConfig,
    sign_bias: &[bool; 4],
    interpolation_filter: u8,
    allow_high_precision_mv: bool,
    seg: &SegmentationParams,
    prev_segment_ids: Option<&[u8]>,
    blocks: Vec<(u32, u32, u8)>,
) -> Result<Vec<RecoveredInterBlock>, Error> {
    use crate::residual::{
        BLOCK_8X8, BLOCK_INVALID, NUM_4X4_BLOCKS_HIGH_LOOKUP, NUM_4X4_BLOCKS_WIDE_LOOKUP,
        get_plane_block_size, get_uv_tx_size,
    };
    use crate::tokens::TokenBlockCtx;

    let chdr_inter = build_inter_chdr_for_test(ctx, tx_mode, reference_mode, comp_config);
    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
    ];
    let mut seg_pred_ctx = SegPredContextState::new(mi_cols, mi_rows);
    let prev_seg_ids = match prev_segment_ids {
        Some(map) => {
            assert_eq!(map.len(), (mi_rows * mi_cols) as usize);
            map.to_vec()
        }
        None => vec![0u8; (mi_rows * mi_cols) as usize],
    };

    let mut coder = BoolCoder::init_bool(data, data.len())?;
    let mut counts = crate::prob_adapt::FrameCounts::new_boxed();
    let mut results = Vec::with_capacity(blocks.len());

    for &(r, c, subsize) in &blocks {
        let avail_u = r > 0;
        let avail_l = c > 0;
        let above_rf = |rl: usize| {
            if avail_u {
                state.get_ref_frame(r - 1, c, rl)
            } else {
                None
            }
        };
        let left_rf = |rl: usize| {
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

        let geom = MvRefGeometry {
            mi_row: r as i32,
            mi_col: c as i32,
            mi_rows: mi_rows as i32,
            mi_cols: mi_cols as i32,
            mi_size: subsize as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as i32,
        };
        let src = FrameStateMvSource::new(&state, None);
        let fix_ref_idx = sign_bias[comp_config.fixed_ref as usize] as u8;
        let prev_seg = PrevSegmentIds {
            mi_rows,
            mi_cols,
            data: &prev_seg_ids,
        };

        let args = InterFrameModeArgs {
            geom: &geom,
            src: &src,
            seg: SegFeatureTables {
                enabled: seg.enabled,
                feature_enabled: &seg.feature_enabled,
                feature_data: &seg.feature_data,
            },
            seg_id: InterSegmentIdArgs {
                update_map: seg.update_map,
                temporal_update: seg.temporal_update,
                tree_probs: seg.tree_probs.as_ref(),
                pred_prob: seg.pred_prob.as_ref(),
                prev: prev_seg,
            },
            skip_prob: &chdr_inter.intra.skip_prob,
            skip_nb: nb_skip,
            is_inter_prob: &chdr_inter.is_inter_prob,
            is_inter_nb,
            tx_mode: chdr_inter.intra.tx_mode,
            tx_probs: &chdr_inter.intra.tx_probs,
            tx_nb: nb_tx,
            ref_frame: InterRefFrameArgs {
                seg_feature_ref_frame_active: false,
                segment_ref_frame_data: 0,
                reference_mode,
                comp_config,
                fix_ref_idx,
                nb: ref_nb,
                comp_mode_prob: &chdr_inter.comp_mode_prob,
                single_ref_prob: &chdr_inter.single_ref_prob,
                comp_ref_prob: &chdr_inter.comp_ref_prob,
            },
            mv_probs: &chdr_inter.mv_probs,
            inter_mode_probs: &chdr_inter.inter_mode_probs,
            interp_filter_probs: &chdr_inter.interp_filter_probs,
            interp_nb,
            interpolation_filter,
            allow_high_precision_mv,
            use_prev_frame_mvs: false,
            sign_bias,
            y_mode_probs: &chdr_inter.y_mode_probs,
            uv_mode_probs: &chdr_inter.uv_mode_probs,
        };

        let mi = inter_frame_mode_info(
            &mut coder,
            args,
            &mut seg_pred_ctx,
            r,
            c,
            &mut counts.noncoef,
        )?;
        let (ref_frame, y_mode, interp_filter, block_mvs) = match mi.block {
            Vp9InterFrameBlock::Inter(b) => (
                [b.ref_frame_0, b.ref_frame_1],
                b.y_mode,
                b.interp_filter,
                b.block_mvs,
            ),
            Vp9InterFrameBlock::Intra(_) => return Err(Error::Unsupported),
        };
        let is_inter = mi.is_inter;
        let is_compound = ref_frame[1] > INTRA_FRAME;

        // Advance the coder over the §6.4.21 residual tokens (inter arm,
        // DCT_DCT). No reconstruction — just the bit consumption + nz
        // write-back, matching the encoder's write_residual_inter.
        let mut eob_total = 0u32;
        let bsize = subsize.max(BLOCK_8X8);
        let mut token_cache = [0u8; 1024];
        let mut tok_buf = [0i64; 1024];
        for plane in 0..3usize {
            let tx_sz = if plane == 0 {
                mi.tx_size
            } else {
                get_uv_tx_size(mi.tx_size, subsize, true, true)
            };
            let step = 1u32 << tx_sz;
            let plane_sz = get_plane_block_size(bsize, plane, true, true);
            if plane_sz == BLOCK_INVALID {
                return Err(Error::InvalidBitstream);
            }
            let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
            let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];
            let sub = plane > 0;
            let base_x = (c * 8) >> u32::from(sub);
            let base_y = (r * 8) >> u32::from(sub);
            let maxx = (mi_cols * 8) >> u32::from(sub);
            let maxy = (mi_rows * 8) >> u32::from(sub);
            let mut y = 0u32;
            while y < num4x4h {
                let mut x = 0u32;
                while x < num4x4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;
                    if start_x < maxx && start_y < maxy && !mi.skip {
                        let scan = crate::scan::get_scan(plane, tx_sz, crate::idct::DCT_DCT);
                        let n0 = 1usize << (tx_sz + 2);
                        let seg_eob = n0 * n0;
                        let tbc = TokenBlockCtx {
                            plane,
                            is_inter: true,
                            tx_type: crate::idct::DCT_DCT,
                            bit_depth: 8,
                            x4: (start_x >> 2) as usize,
                            y4: (start_y >> 2) as usize,
                            max_x: ((2 * mi_cols) >> u32::from(sub)) as usize,
                            max_y: ((2 * mi_rows) >> u32::from(sub)) as usize,
                        };
                        token_cache[..seg_eob].fill(0);
                        nonzero = crate::tokens::tokens(
                            &mut coder,
                            &tbc,
                            tx_sz,
                            scan,
                            &chdr_inter.intra.coef_probs,
                            &nz[plane],
                            &mut token_cache[..seg_eob],
                            &mut tok_buf[..seg_eob],
                            &mut counts,
                        )?;
                        eob_total += u32::from(nonzero);
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
                    x += step;
                }
                y += step;
            }
        }

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
            skip: mi.skip,
            tx_size: mi.tx_size as u8,
            y_mode,
            segment_id: mi.segment_id,
            ref_frame,
            is_inter,
            eob_total,
            interp_filter,
            block_mvs: block_mvs_i16,
            sub_modes: [DC_PRED; 4],
        };
        decode_block_apply(&mut state, r, c, subsize, &result);

        results.push(RecoveredInterBlock {
            skip: mi.skip,
            tx_size: mi.tx_size,
            y_mode,
            segment_id: mi.segment_id,
            ref_frame,
            interp_filter,
            mv: [
                [block_mvs[0][3][0], block_mvs[0][3][1]],
                [block_mvs[1][3][0], block_mvs[1][3][1]],
            ],
            block_mvs,
        });
        let _ = is_compound;
    }
    Ok(results)
}

/// Build a minimal `Vp9CompressedHeaderInter` from a `FrameContext`'s
/// default probability banks, for the inter-block round-trip test helper.
#[cfg(test)]
fn build_inter_chdr_for_test(
    ctx: &crate::compressed::FrameContext,
    tx_mode: crate::compressed::TxMode,
    reference_mode: crate::compressed::ReferenceMode,
    comp_config: CompoundReferenceConfig,
) -> Vp9CompressedHeaderInter {
    let compound_reference_config = match reference_mode {
        crate::compressed::ReferenceMode::SingleReference => None,
        _ => Some(comp_config),
    };
    Vp9CompressedHeaderInter {
        intra: Vp9CompressedHeader {
            tx_mode,
            tx_probs: ctx.tx_probs,
            coef_probs: ctx.coef_probs,
            skip_prob: ctx.skip_prob,
        },
        inter_mode_probs: ctx.inter_mode_probs,
        interp_filter_probs: ctx.interp_filter_probs,
        is_inter_prob: ctx.is_inter_prob,
        reference_mode,
        compound_reference_config,
        comp_mode_prob: ctx.comp_mode_prob,
        single_ref_prob: ctx.single_ref_prob,
        comp_ref_prob: ctx.comp_ref_prob,
        y_mode_probs: ctx.y_mode_probs,
        uv_mode_probs: ctx.uv_mode_probs,
        partition_probs: ctx.partition_probs,
        mv_probs: ctx.mv_probs.clone(),
    }
}

impl Vp9DecodedFrame {
    /// Pack the frame as planar bytes (Y then U then V): one byte per
    /// sample for `BitDepth == 8`, little-endian `u16` pairs for 10 /
    /// 12-bit content.
    pub fn to_planar_bytes(&self) -> Vec<u8> {
        let total = self.y.len() + self.u.len() + self.v.len();
        if self.bit_depth == 8 {
            let mut out = Vec::with_capacity(total);
            for plane in [&self.y, &self.u, &self.v] {
                out.extend(plane.iter().map(|&s| s as u8));
            }
            out
        } else {
            let mut out = Vec::with_capacity(total * 2);
            for plane in [&self.y, &self.u, &self.v] {
                for &s in plane.iter() {
                    out.extend_from_slice(&s.to_le_bytes());
                }
            }
            out
        }
    }
}
