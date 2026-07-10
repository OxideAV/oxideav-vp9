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

use crate::bool_coder::BoolCoder;
use crate::compressed::{
    parse_compressed_header, parse_compressed_header_inter_with_ctx,
    parse_compressed_header_with_ctx, CompoundReferenceConfig, FrameContext, RefFrameSignBias,
    Vp9CompressedHeader, Vp9CompressedHeaderInter, Vp9CompressedHeaderInterInputs,
};
use crate::decode_block::{decode_block_apply, DecodedBlockResult, Vp9FrameState};
use crate::dequant::{get_ac_quant, get_dc_quant, seg_feature_active};
use crate::frame_loop_filter::{frame_loop_filter, CurrFrame};
use crate::header::{
    parse_uncompressed_header, parse_uncompressed_header_with_refs, ColorConfig, FrameType,
    QuantizationParams, RefFrameState, SegmentationParams, Vp9FrameHeader, MAX_SEGMENTS,
    SEG_LVL_MAX,
};
use crate::inter_decode::{build_ref_planes, FrameStateMvSource, PrevFrameMvs};
use crate::inter_mv::{BlockGrid, ScaleGeom};
use crate::inter_pred::{predict_inter, InterPredArgs};
use crate::intra::{predict_intra, Plane, PredMode};
use crate::loop_filter::loop_filter_frame_init;
use crate::mode_info::{
    inter_frame_mode_info, intra_frame_mode_info, intra_segment_id, InterFrameModeArgs,
    InterRefFrameArgs, InterSegmentIdArgs, InterpFilterNeighbours, IntraFrameNeighbours,
    IsInterNeighbours, NeighbourSkips, NeighbourTxSizes, PrevSegmentIds, RefFrameNeighbours,
    SegFeatureTables, SegPredContextState, Vp9InterFrameBlock, Vp9IntraMiBlock, ALTREF_FRAME,
    DC_PRED, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, NONE_REF_FRAME, SEG_LVL_SKIP,
};
use crate::mv_ref::MvRefGeometry;
use crate::partition::{
    decode_partition, get_tile_offset, tile_payload_sizes, LeafSink, PartitionContextState,
    PartitionProbsKind,
};
use crate::reconstruct::{reconstruct_block, tx_type_for_intra};
use crate::ref_buffer::{CurrFramePlanes, RefBuffers, REFS_PER_FRAME};
use crate::residual::{
    get_plane_block_size, get_uv_tx_size, BLOCK_64X64, BLOCK_8X8, BLOCK_INVALID,
    NUM_4X4_BLOCKS_HIGH_LOOKUP, NUM_4X4_BLOCKS_WIDE_LOOKUP,
};
use crate::scan::get_scan;
use crate::superblock_loop_filter::{SuperblockFilterFrame, SuperblockFilterPlane};
use crate::tokens::{tokens, NonzeroContext, TokenBlockCtx};
use crate::Error;

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

/// §7.2 `setup_past_independence( )` default `loop_filter_ref_deltas`
/// (`vp9-spec.txt` §7.2: 1 for INTRA_FRAME, 0 for LAST_FRAME, -1 for
/// GOLDEN_FRAME and ALTREF_FRAME).
const DEFAULT_LOOP_FILTER_REF_DELTAS: [i8; 4] = [1, 0, -1, -1];

/// §7.2 `setup_past_independence( )` default `loop_filter_mode_deltas`
/// (both zero).
const DEFAULT_LOOP_FILTER_MODE_DELTAS: [i8; 2] = [0, 0];

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
    y: &'a mut Plane,
    u: &'a mut Plane,
    v: &'a mut Plane,
    nz: &'a mut [NonzeroContext; 3],
    token_cache: &'a mut [u8; 1024],
    tok_buf: &'a mut [i64; 1024],
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

                        let plane_buf: &mut Plane = match plane {
                            0 => self.y,
                            1 => self.u,
                            _ => self.v,
                        };
                        // §6.4.21: predict_intra( plane, startX, startY,
                        //   AvailL || x > 0, AvailU || y > 0,
                        //   x + step < num4x4w, txSz, blockIdx ).
                        predict_intra(
                            plane_buf,
                            start_x as usize,
                            start_y as usize,
                            avail_l || x > 0,
                            avail_u || y > 0,
                            x + step < num4x4w,
                            tx_sz,
                            mode,
                            (maxx - 1) as usize,
                            (maxy - 1) as usize,
                            self.bit_depth,
                        );

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
                            self.token_cache[..seg_eob].fill(0);
                            nonzero = tokens(
                                coder,
                                &tbc,
                                tx_sz,
                                scan,
                                &self.chdr.coef_probs,
                                &self.nz[plane],
                                &mut self.token_cache[..],
                                &mut self.tok_buf[..seg_eob],
                            )?;
                            // §6.4.24: EobTotal += nonzero.
                            eob_total += u32::from(nonzero);

                            // §8.6.2 reconstruct( plane, startX, startY,
                            // txSz ).
                            reconstruct_block(
                                plane_buf,
                                start_x as usize,
                                start_y as usize,
                                tx_sz,
                                &self.tok_buf[..seg_eob],
                                dc_quant,
                                ac_quant,
                                tx_type,
                                self.lossless,
                                self.bit_depth,
                            );
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
            uv_mode_probs: &crate::mode_info::DEFAULT_UV_MODE_PROBS,
        };

        let seg_pred_ctx = &mut self.inter.as_mut().unwrap().seg_pred_ctx;
        let mi = inter_frame_mode_info(coder, args, seg_pred_ctx, r, c)?;

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
                        let plane_buf: &mut Plane = match plane {
                            0 => self.y,
                            1 => self.u,
                            _ => self.v,
                        };

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
                            predict_intra(
                                plane_buf,
                                start_x as usize,
                                start_y as usize,
                                avail_l || x > 0,
                                avail_u || y > 0,
                                x + step < num4x4w,
                                tx_sz,
                                mode,
                                (maxx - 1) as usize,
                                (maxy - 1) as usize,
                                self.bit_depth,
                            );
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
                            self.token_cache[..seg_eob].fill(0);
                            nonzero = tokens(
                                coder,
                                &tbc,
                                tx_sz,
                                scan,
                                &self.chdr.coef_probs,
                                &self.nz[plane],
                                &mut self.token_cache[..],
                                &mut self.tok_buf[..seg_eob],
                            )?;
                            eob_total += u32::from(nonzero);

                            reconstruct_block(
                                plane_buf,
                                start_x as usize,
                                start_y as usize,
                                tx_sz,
                                &self.tok_buf[..seg_eob],
                                dc_quant,
                                ac_quant,
                                tx_type,
                                self.lossless,
                                self.bit_depth,
                            );
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
    /// region — resolves the §8.10 reference planes and runs the
    /// [`predict_inter`] driver, writing into the working plane.
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
        let inter = self.inter.as_ref().expect("inter ctx");
        let refs = build_ref_planes(inter.refs, ref_frame, &inter.ref_frame_idx, plane);
        let grid = BlockGrid {
            mi_row: r as i32,
            mi_col: c as i32,
            mi_rows: self.mi_rows as i32,
            mi_cols: self.mi_cols as i32,
            mi_size,
        };
        let geom = ScaleGeom {
            // ref_frame_width/height are per-list and overridden inside
            // the driver from each RefPlane; these frame-level dims feed
            // the §8.5.2.3 scale ratio numerator.
            ref_frame_width: inter.frame_width as i32,
            ref_frame_height: inter.frame_height as i32,
            frame_width: inter.frame_width as i32,
            frame_height: inter.frame_height as i32,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
        };
        let args = InterPredArgs {
            plane,
            x,
            y,
            w,
            h,
            block_idx,
            interp_filter: interp_filter as usize,
            bit_depth: self.bit_depth,
            is_compound,
        };
        let dst: &mut Plane = match plane {
            0 => self.y,
            1 => self.u,
            _ => self.v,
        };
        predict_inter(
            dst,
            &args,
            &grid,
            &geom,
            block_mvs,
            &refs,
            self.subsampling_x,
            self.subsampling_y,
        );
    }
}

/// One frame's reconstruction products: the cropped output plus the
/// MI-aligned working planes and §6.4.4 state the §8.10 reference update
/// + the next frame's §6.5 prev-MV scan read.
struct FrameDecodeProducts {
    out: Vp9DecodedFrame,
    /// MI-aligned working luma plane (stride `mi_cols * 8`).
    plane_y: Plane,
    plane_u: Plane,
    plane_v: Plane,
    /// `MiCols * 8` luma stride.
    y_w: usize,
    uv_w: usize,
    /// The §6.4.4 frame-wide state (carries `Mvs` / `RefFrames` /
    /// `SegmentIds` for the next frame's §6.5 / §6.4.14 reads).
    state: Vp9FrameState,
}

/// The §6.3 compressed-header product threaded into [`decode_single_frame`]:
/// either the intra-only header or the full inter header.
enum ChdrKind {
    Intra(Box<Vp9CompressedHeader>),
    Inter(Box<Vp9CompressedHeaderInter>),
}

impl ChdrKind {
    fn intra(&self) -> &Vp9CompressedHeader {
        match self {
            ChdrKind::Intra(c) => c,
            ChdrKind::Inter(c) => &c.intra,
        }
    }
}

/// The previous-frame state an inter frame's §6.5 / §6.4.14 reads need.
struct PrevFrameState<'a> {
    /// `RefFrames[ ][ ][ ]` stored `[mi * 2 + list]` (both ref lists).
    ref_frames: &'a [i32],
    /// `Mvs[ ][ ][ ]` stored `[mi * 2 + list]`.
    mvs: &'a [(i16, i16)],
    /// `SegmentId[ ][ ]` stored `[mi]`.
    segment_ids: &'a [u8],
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
    let products = decode_single_frame(data, &hdr, ch_end, chdr, None, None)?;
    Ok(products.out)
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
/// optional previous-frame motion field. Returns the reconstruction
/// products (cropped output + working planes + §6.4.4 state).
fn decode_single_frame(
    data: &[u8],
    hdr: &Vp9FrameHeader,
    ch_end: usize,
    chdr_kind: ChdrKind,
    refs: Option<&RefBuffers>,
    prev: Option<PrevFrameState<'_>>,
) -> Result<FrameDecodeProducts, Error> {
    let lossless = hdr.quantization.lossless;
    let chdr = chdr_kind.intra().clone();

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

    // CurrFrame plane buffers at the §6.4.21 / §8.5.1 working extents
    // (MiCols * 8 wide — prediction and reconstruction may touch the
    // full MI-aligned area; §8.10 crops on output).
    let y_w = (mi_cols * 8) as usize;
    let y_h = (mi_rows * 8) as usize;
    let uv_w = ((mi_cols * 8) >> u32::from(ssx)) as usize;
    let uv_h = ((mi_rows * 8) >> u32::from(ssy)) as usize;
    let mut plane_y = Plane::new(y_w, y_h);
    let mut plane_u = Plane::new(uv_w, uv_h);
    let mut plane_v = Plane::new(uv_w, uv_h);

    // §6.4.4 frame-wide arrays + §9.3.2 context strips. The §7.4.1
    // NOTE sizes AboveNonzeroContext to MiCols * 2 (reads are bounded
    // by maxX = (2 * MiCols) >> subsampling_x); the partition strips
    // span Sb64Cols * 8 / Sb64Rows * 8 because §6.4.3 reads beyond
    // MiCols / MiRows.
    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
        NonzeroContext::new(
            ((2 * mi_cols) >> u32::from(ssx)) as usize,
            ((2 * mi_rows) >> u32::from(ssy)) as usize,
        ),
    ];
    let mut pctx = PartitionContextState::new((sb64_cols * 8) as usize, (sb64_rows * 8) as usize);
    let mut token_cache = [0u8; 1024];
    let mut tok_buf = [0i64; 1024];

    // Inter-frame scaffolding: the §6.4.11 segmentation-prediction
    // context, the §6.5 prev-MV / §6.4.14 prev-segment snapshots, and the
    // §6.3 inter-only compressed-header tail.
    let chdr_inter_ref: Option<&Vp9CompressedHeaderInter> = match &chdr_kind {
        ChdrKind::Inter(c) => Some(c),
        ChdrKind::Intra(_) => None,
    };
    let is_inter_frame = chdr_inter_ref.is_some();
    let mut seg_pred_ctx = SegPredContextState::new(mi_cols, mi_rows);
    let zero_prev_ref = vec![INTRA_FRAME; (mi_rows * mi_cols * 2) as usize];
    let zero_prev_mvs = vec![(0i16, 0i16); (mi_rows * mi_cols * 2) as usize];
    let zero_prev_seg = vec![0u8; (mi_rows * mi_cols) as usize];
    let (prev_ref_frames, prev_mvs, prev_segment_ids): (&[i32], &[(i16, i16)], &[u8]) = match &prev
    {
        Some(p)
            if p.ref_frames.len() == zero_prev_ref.len()
                && p.mvs.len() == zero_prev_mvs.len()
                && p.segment_ids.len() == zero_prev_seg.len() =>
        {
            (p.ref_frames, p.mvs, p.segment_ids)
        }
        // Dimension change or no prev frame: §7.2.6 UsePrevFrameMvs is 0
        // and §6.4.14 PrevSegmentIds is cleared to zero.
        _ => (&zero_prev_ref, &zero_prev_mvs, &zero_prev_seg),
    };

    // §7.2.6 UsePrevFrameMvs: requires a same-dimension prev frame that
    // was shown, error_resilient_mode == 0, and FrameIsIntra == 0.
    let use_prev_frame_mvs = is_inter_frame
        && prev.is_some()
        && prev_mvs.len() == zero_prev_mvs.len()
        && !hdr.error_resilient_mode;

    // §6.4 decode_tiles( sz ): per-tile byte budget walk, then the
    // per-tile bool-coder bracket around §6.4.2 decode_tile( ).
    let tile_data = &data[ch_end..];
    let sz = tile_data.len() as u32;
    let tile_cols_log2 = hdr.tile_info.tile_cols_log2;
    let tile_rows_log2 = hdr.tile_info.tile_rows_log2;
    let sizes = tile_payload_sizes(tile_data, sz, tile_rows_log2, tile_cols_log2)?;

    // §6.4 line 2303 / §7.4.1: clear_above_context( ) — once per
    // frame. The freshly allocated strips are already zero; the
    // explicit reset documents the spec step.
    pctx.clear_above();

    let tile_cols = 1u32 << tile_cols_log2;
    let tile_rows = 1u32 << tile_rows_log2;
    let mut byte_cursor = 0usize;
    let mut size_idx = 0usize;
    for tile_row in 0..tile_rows {
        for tile_col in 0..tile_cols {
            let last_tile = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
            if !last_tile {
                byte_cursor += 4; // the f(32) tile_size prefix
            }
            let tile_size = sizes[size_idx] as usize;
            size_idx += 1;

            let mi_row_start = get_tile_offset(tile_row, mi_rows, u32::from(tile_rows_log2));
            let mi_row_end = get_tile_offset(tile_row + 1, mi_rows, u32::from(tile_rows_log2));
            let mi_col_start = get_tile_offset(tile_col, mi_cols, u32::from(tile_cols_log2));
            let mi_col_end = get_tile_offset(tile_col + 1, mi_cols, u32::from(tile_cols_log2));

            // §6.4 line 2326: init_bool( tile_size ).
            let tile_slice = &tile_data[byte_cursor..byte_cursor + tile_size];
            let mut coder = BoolCoder::init_bool(tile_slice, tile_size)?;

            // Build the per-tile inter context (re-borrowing the shared
            // per-frame `seg_pred_ctx`). On keyframes / intra-only frames
            // this is `None` and the §6.4.6 intra arm runs.
            let inter_ctx = match (chdr_inter_ref, refs) {
                (Some(ci), Some(rb)) => {
                    let comp_config =
                        ci.compound_reference_config
                            .unwrap_or(CompoundReferenceConfig {
                                fixed_ref: LAST_FRAME,
                                var_ref: [GOLDEN_FRAME, ALTREF_FRAME],
                            });
                    // §6.2.5 ref_frame_sign_bias[ 4 ] indexed by §3 value.
                    let mut sb = [false; 4];
                    let hdr_sb = hdr.ref_frame_sign_bias;
                    sb[LAST_FRAME as usize] = hdr_sb[0];
                    sb[GOLDEN_FRAME as usize] = hdr_sb[1];
                    sb[ALTREF_FRAME as usize] = hdr_sb[2];
                    Some(InterCtx {
                        chdr: ci,
                        refs: rb,
                        ref_frame_idx: hdr.ref_frame_idx.unwrap_or([0, 1, 2]),
                        sign_bias: sb,
                        interpolation_filter: hdr.interpolation_filter,
                        allow_high_precision_mv: hdr.allow_high_precision_mv,
                        use_prev_frame_mvs,
                        comp_config,
                        frame_width: hdr.frame_width,
                        frame_height: hdr.frame_height,
                        prev_ref_frames,
                        prev_mvs,
                        prev_segment_ids,
                        seg_pred_ctx: &mut seg_pred_ctx,
                    })
                }
                _ => None,
            };

            let mut block_decoder = BlockDecoder {
                seg: &hdr.segmentation,
                quant: &hdr.quantization,
                chdr: &chdr,
                mi_rows,
                mi_cols,
                mi_col_start,
                mi_col_end,
                subsampling_x: ssx,
                subsampling_y: ssy,
                bit_depth,
                lossless,
                state: &mut state,
                y: &mut plane_y,
                u: &mut plane_u,
                v: &mut plane_v,
                nz: &mut nz,
                token_cache: &mut token_cache,
                tok_buf: &mut tok_buf,
                inter: inter_ctx,
            };

            // §6.4.3 partition probs: keyframe fixed table or the inter
            // running `partition_probs[ ]`.
            let part_kind = match chdr_inter_ref {
                Some(ci) => PartitionProbsKind::Inter(&ci.partition_probs),
                None => PartitionProbsKind::Keyframe,
            };

            // §6.4.2 decode_tile( ): superblock raster with the
            // §7.4.2 clear_left_context( ) reset per superblock row
            // (LeftNonzeroContext + LeftPartitionContext).
            let mut r = mi_row_start;
            while r < mi_row_end {
                pctx.clear_left();
                if let Some(ic) = block_decoder.inter.as_mut() {
                    ic.seg_pred_ctx.clear_left();
                }
                for plane_nz in block_decoder.nz.iter_mut() {
                    plane_nz.left.fill(0);
                }
                let mut c = mi_col_start;
                while c < mi_col_end {
                    decode_partition(
                        &mut coder,
                        r,
                        c,
                        BLOCK_64X64,
                        mi_rows,
                        mi_cols,
                        &mut pctx,
                        part_kind,
                        &mut block_decoder,
                    )?;
                    c += 8;
                }
                r += 8;
            }

            // §6.4 line 2328: exit_bool( ).
            coder.exit_bool()?;
            byte_cursor += tile_size;
        }
    }

    // §8.8 loop filter over the reconstructed frame. The §8.8.1 init
    // consumes the §6.2.8 deltas resolved against the §7.2
    // setup_past_independence defaults (intra frames reset them).
    let mut ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
    for (slot, delta) in ref_deltas.iter_mut().zip(hdr.loop_filter.ref_deltas) {
        if let Some(d) = delta {
            *slot = d;
        }
    }
    let mut mode_deltas = DEFAULT_LOOP_FILTER_MODE_DELTAS;
    for (slot, delta) in mode_deltas.iter_mut().zip(hdr.loop_filter.mode_deltas) {
        if let Some(d) = delta {
            *slot = d;
        }
    }
    let lvl_lookup =
        loop_filter_frame_init(&hdr.loop_filter, &hdr.segmentation, ref_deltas, mode_deltas);

    let skips_bool: Vec<bool> = state.skips.iter().map(|&s| s != 0).collect();
    let ref_frames_0: Vec<i32> = state.ref_frames.chunks_exact(2).map(|p| p[0]).collect();
    let filter_frame = SuperblockFilterFrame {
        mi_sizes: &state.mi_sizes,
        tx_sizes: &state.tx_sizes,
        skips: &skips_bool,
        ref_frames_0: &ref_frames_0,
        y_modes: &state.y_modes,
        segment_ids: &state.segment_ids,
        mi_cols,
        mi_rows,
        subsampling_x: u8::from(ssx),
        subsampling_y: u8::from(ssy),
        loop_filter_sharpness: hdr.loop_filter.sharpness,
        bit_depth: hdr.color_config.bit_depth,
        lvl_lookup: &lvl_lookup,
    };

    // §8.8 input is CurrFrame at the FrameWidth x FrameHeight extent;
    // the working planes are MI-aligned, so the filter views carry the
    // working stride with the §8.10 visible extents.
    let crop_w = hdr.frame_width as usize;
    let crop_h = hdr.frame_height as usize;
    let uv_crop_w = ((hdr.frame_width + u32::from(ssx)) >> u32::from(ssx)) as usize;
    let uv_crop_h = ((hdr.frame_height + u32::from(ssy)) >> u32::from(ssy)) as usize;
    // §8.1 step 2: the §8.8 loop filter process runs only "if
    // loop_filter_level is not equal to 0". The gate is on the frame
    // header's `loop_filter_level` itself — with level 0 no edge is
    // filtered even where the §8.8.1 deltas (e.g. the +1 INTRA
    // ref-delta) or a SEG_LVL_ALT_L override would lift a LvlLookup
    // entry above zero.
    if hdr.loop_filter.level != 0 {
        let mut curr = CurrFrame {
            planes: [
                SuperblockFilterPlane {
                    data: plane_y.samples_mut(),
                    stride: y_w,
                    width: crop_w,
                    height: crop_h,
                },
                SuperblockFilterPlane {
                    data: plane_u.samples_mut(),
                    stride: uv_w,
                    width: uv_crop_w,
                    height: uv_crop_h,
                },
                SuperblockFilterPlane {
                    data: plane_v.samples_mut(),
                    stride: uv_w,
                    width: uv_crop_w,
                    height: uv_crop_h,
                },
            ],
        };
        frame_loop_filter(&mut curr, &filter_frame);
    }

    // §8.10 output: crop the working planes to the visible extents.
    let crop = |plane: &Plane, w: usize, h: usize| -> Vec<u16> {
        let mut out = Vec::with_capacity(w * h);
        for yy in 0..h {
            for xx in 0..w {
                out.push(plane.get(xx, yy) as u16);
            }
        }
        out
    };

    let out = Vp9DecodedFrame {
        width: hdr.frame_width,
        height: hdr.frame_height,
        bit_depth: hdr.color_config.bit_depth,
        subsampling_x: ssx,
        subsampling_y: ssy,
        y: crop(&plane_y, crop_w, crop_h),
        u: crop(&plane_u, uv_crop_w, uv_crop_h),
        v: crop(&plane_v, uv_crop_w, uv_crop_h),
    };

    Ok(FrameDecodeProducts {
        out,
        plane_y,
        plane_u,
        plane_v,
        y_w,
        uv_w,
        state,
    })
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
    let mut bufs = RefBuffers::new();
    let mut out = Vec::new();
    // The most-recently-decoded color config + frame dims (for the §6.2
    // inter header's inherited color config + §8.10 ref dims).
    let mut last_color: Option<ColorConfig> = None;
    // The previous decoded frame's §6.4.4 state + show/size for the §6.5
    // prev-MV scan + §7.2.6 UsePrevFrameMvs derivation.
    let mut prev_state: Option<(Vp9FrameState, u32, u32, bool)> = None;
    // §8.10 `FrameStore[ ]`: the eight reference slots as displayable
    // frames, for a later §8.9 `show_existing_frame` output. Per §8.10
    // step 1, EVERY decoded frame writes the slots `refresh_frame_flags`
    // names — including hidden (`show_frame == 0`) alt-ref frames — and a
    // `show_existing_frame` packet re-displays `FrameStore[ idx ]` with
    // that stored frame's own §8.9 dimensions / bit depth. `RefBuffers`
    // already tracks the sample planes; this parallel array keeps the
    // crop + packing metadata needed to re-emit the frame verbatim.
    let mut frame_store: [Option<Vp9DecodedFrame>; 8] = Default::default();

    // §6.1.2 / §7.2 `FrameContext[ 4 ]` — the persistent entropy
    // probability banks `load_probs( )` / `save_probs( )` thread across
    // frames. Initialised to the §10.5 defaults; a key / intra-only /
    // error-resilient frame's §7.2 `setup_past_independence( )` resets
    // the bank(s) again before the frame folds its own compressed-header
    // forward updates on top.
    let mut frame_contexts: [FrameContext; 4] = Default::default();

    // §7.2.10 / §7.2 persistent segmentation feature table: per §7.2.10
    // "segmentation_update_data equal to 0 indicates that the
    // segmentation parameters should keep their existing values", so
    // `FeatureEnabled` / `FeatureData` / `abs_or_delta_update` carry
    // across frames until a `segmentation_update_data == 1` frame
    // rewrites them or §7.2 `setup_past_independence( )` (key /
    // intra-only / error-resilient frame) clears them to zero. The
    // per-frame header parse alone cannot see this state, so it is
    // threaded here and substituted into the parsed header below.
    let mut persist_feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
    let mut persist_feature_data = [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS];
    let mut persist_abs_or_delta = false;

    // §7.2.8 persistent loop-filter deltas: `loop_filter_delta_update ==
    // 0` (or an individual update flag of 0) keeps the existing
    // `loop_filter_ref_deltas` / `loop_filter_mode_deltas`; §7.2
    // `setup_past_independence( )` resets them to the defaults.
    let mut persist_ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
    let mut persist_mode_deltas = DEFAULT_LOOP_FILTER_MODE_DELTAS;

    // §6.4.14 `PrevSegmentIds[ ][ ]` — the persistent segmentation map a
    // §6.4.12 `temporal_update` / `update_map == 0` frame predicts from.
    // §8.1 step 3 only refreshes it after a frame with
    // `segmentation_update_map == 1`; a frame that reuses the map
    // (`update_map == 0`) leaves it pointing at the *last map-bearing*
    // frame, NOT its own (possibly Min-collapsed) `SegmentIds`. Tracking
    // it separately from the per-frame `prev_state` snapshot is what keeps
    // a run of `update_map == 0` inter frames reading the original map.
    let mut prev_segment_ids_map: Option<(Vec<u8>, u32, u32)> = None;

    for &frame in frames {
        // Peek the uncompressed header (with inherited ref state so inter
        // frames parse).
        let ref_dims: Vec<(u32, u32)> = (0..8)
            .map(|i| (bufs.ref_frame_width(i), bufs.ref_frame_height(i)))
            .collect();
        let ref_state = last_color.map(|cc| RefFrameState {
            ref_dims: &ref_dims,
            color_config: cc,
        });
        let mut hdr = parse_uncompressed_header_with_refs(frame, ref_state)?;

        if hdr.show_existing_frame {
            // §8.9: output `FrameStore[ frame_to_show_map_idx ]` verbatim.
            // The slot may have been written by a hidden alt-ref frame, so
            // it is sourced from `frame_store` (updated for every decoded
            // frame) rather than only the shown ones.
            let idx = hdr.frame_to_show_map_idx.ok_or(Error::InvalidBitstream)? as usize;
            let f = frame_store
                .get(idx)
                .and_then(|s| s.clone())
                .ok_or(Error::InvalidBitstream)?;
            out.push(f);
            // §7.2.6 NOTE: compute_image_size is not invoked for a
            // show_existing_frame packet, so it leaves `prev_state` (the
            // §7.2.6 "previous invocation" record) untouched.
            continue;
        }

        let frame_is_intra = matches!(hdr.frame_type, FrameType::KeyFrame) || hdr.intra_only;

        // §7.2 setup_past_independence( ): a key / intra-only /
        // error-resilient frame clears the persistent segmentation
        // feature table and resets the loop-filter deltas to the
        // defaults BEFORE its own loop_filter_params( ) /
        // segmentation_params( ) apply on top.
        if frame_is_intra || hdr.error_resilient_mode {
            persist_feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
            persist_feature_data = [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS];
            persist_abs_or_delta = false;
            persist_ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
            persist_mode_deltas = DEFAULT_LOOP_FILTER_MODE_DELTAS;
        }

        // §7.2.8: a `loop_filter_delta_update == 1` frame folds the
        // per-delta updates it carries into the persistent deltas (an
        // individual absent update keeps the prior value); every frame
        // then decodes with the fully-resolved persistent values.
        for (slot, delta) in persist_ref_deltas
            .iter_mut()
            .zip(hdr.loop_filter.ref_deltas)
        {
            if let Some(d) = delta {
                *slot = d;
            }
        }
        for (slot, delta) in persist_mode_deltas
            .iter_mut()
            .zip(hdr.loop_filter.mode_deltas)
        {
            if let Some(d) = delta {
                *slot = d;
            }
        }
        hdr.loop_filter.ref_deltas = persist_ref_deltas.map(Some);
        hdr.loop_filter.mode_deltas = persist_mode_deltas.map(Some);

        // §7.2.10: `segmentation_update_data == 0` keeps the existing
        // feature table (even across `segmentation_enabled == 0`
        // frames); `segmentation_update_data == 1` rewrites it.
        if hdr.segmentation.update_data {
            persist_feature_enabled = hdr.segmentation.feature_enabled;
            persist_feature_data = hdr.segmentation.feature_data;
            persist_abs_or_delta = hdr.segmentation.abs_or_delta_update;
        } else {
            hdr.segmentation.feature_enabled = persist_feature_enabled;
            hdr.segmentation.feature_data = persist_feature_data;
            hdr.segmentation.abs_or_delta_update = persist_abs_or_delta;
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

        // §6.4.14 PrevSegmentIds: the last map-bearing frame's
        // `SegmentIds`, if it matches the current dimensions. A run of
        // `update_map == 0` frames keeps predicting from this same map.
        let prev_seg_slice: Option<&[u8]> =
            prev_segment_ids_map.as_ref().and_then(|(map, mw, mh)| {
                if *mw == hdr.frame_width && *mh == hdr.frame_height {
                    Some(map.as_slice())
                } else {
                    None
                }
            });

        // §6.5 / §6.4.14 previous-frame snapshot (same-dimension only).
        // §6.5 MV/ref prediction reads the immediately-preceding decoded
        // frame; §6.4.14 segment prediction reads the separately-tracked
        // `prev_segment_ids_map` (last `update_map == 1` frame).
        let prev = prev_state.as_ref().and_then(|(ps, pw, ph, shown)| {
            if *pw == hdr.frame_width && *ph == hdr.frame_height && *shown {
                Some(PrevFrameState {
                    ref_frames: &ps.ref_frames,
                    mvs: &ps.mvs,
                    segment_ids: prev_seg_slice.unwrap_or(&ps.segment_ids),
                })
            } else {
                None
            }
        });

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
                frame_contexts = Default::default();
            } else if hdr.reset_frame_context == 2 {
                frame_contexts[hdr.frame_context_idx as usize] = FrameContext::default();
            }
            0usize
        } else {
            hdr.frame_context_idx as usize
        };

        // §6.1.2 load_probs( frame_context_idx ): the compressed-header
        // forward updates fold onto the loaded bank, not the static
        // defaults.
        let base_ctx = frame_contexts[frame_context_idx].clone();

        let products = if frame_is_intra {
            let chdr_parsed =
                parse_compressed_header_with_ctx(&frame[ch_start..ch_end], lossless, &base_ctx)?;
            // §6.1.2 save_probs( frame_context_idx ) (parallel-mode path:
            // no backward adaptation, just persist the forward-updated
            // tables).
            if hdr.refresh_frame_context {
                frame_contexts[frame_context_idx].apply_intra(&chdr_parsed);
            }
            let chdr = ChdrKind::Intra(Box::new(chdr_parsed));
            decode_single_frame(frame, &hdr, ch_end, chdr, Some(&bufs), prev)?
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
            if hdr.refresh_frame_context {
                frame_contexts[frame_context_idx].apply_inter(&chdr_parsed);
            }
            let chdr = ChdrKind::Inter(Box::new(chdr_parsed));
            decode_single_frame(frame, &hdr, ch_end, chdr, Some(&bufs), prev)?
        };

        // §8.10 reference frame update: store the (loop-filtered) working
        // planes into the slots `refresh_frame_flags` names.
        let curr = CurrFramePlanes {
            y: products.plane_y.samples(),
            y_stride: products.y_w,
            u: products.plane_u.samples(),
            v: products.plane_v.samples(),
            uv_stride: products.uv_w,
            frame_width: hdr.frame_width,
            frame_height: hdr.frame_height,
            subsampling_x: hdr.color_config.subsampling_x,
            subsampling_y: hdr.color_config.subsampling_y,
            bit_depth: hdr.color_config.bit_depth,
        };
        bufs.update(hdr.refresh_frame_flags, &curr);

        // Carry inherited color config (§6.2: an intra frame establishes
        // it; an inter frame before any intra frame is malformed but we
        // tolerate it by adopting its own config) + the prev-frame motion
        // field.
        if frame_is_intra || last_color.is_none() {
            last_color = Some(hdr.color_config);
        }
        // §8.1 step 3: refresh PrevSegmentIds only after a frame that
        // built a fresh map (`segmentation_enabled && update_map`). A
        // frame reusing the map leaves the persistent copy untouched so
        // the *next* frame still predicts from the original.
        if hdr.segmentation.enabled && hdr.segmentation.update_map {
            prev_segment_ids_map = Some((
                products.state.segment_ids.clone(),
                hdr.frame_width,
                hdr.frame_height,
            ));
        }

        let shown = hdr.show_frame;
        prev_state = Some((products.state, hdr.frame_width, hdr.frame_height, shown));

        // §8.10 step 1: store the decoded frame into the `FrameStore[ ]`
        // slots `refresh_frame_flags` names. This happens for every decoded
        // frame, including hidden (`show_frame == 0`) alt-ref frames, so a
        // later `show_existing_frame` can re-display whatever a hidden ARF
        // wrote into a slot.
        for (i, slot) in frame_store.iter_mut().enumerate() {
            if (hdr.refresh_frame_flags >> i) & 1 == 1 {
                *slot = Some(products.out.clone());
            }
        }

        // §8.9: a hidden (`show_frame == 0`) frame is decoded as a reference
        // but not emitted; only shown frames are output here.
        if shown {
            out.push(products.out);
        }
    }

    Ok(out)
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
            y,
            u,
            v,
            nz: &mut nz,
            token_cache: &mut token_cache,
            tok_buf: &mut tok_buf,
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
}

/// Decode a fixed list of `MiSize >= BLOCK_8X8` **inter** blocks at the
/// mode-info + residual-token level (no §8.5.2 motion compensation), for
/// the `inter_block_writer` round-trip tests.
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
    blocks: Vec<(u32, u32, u8)>,
) -> Result<Vec<RecoveredInterBlock>, Error> {
    use crate::residual::{
        get_plane_block_size, get_uv_tx_size, BLOCK_8X8, BLOCK_INVALID, NUM_4X4_BLOCKS_HIGH_LOOKUP,
        NUM_4X4_BLOCKS_WIDE_LOOKUP,
    };
    use crate::tokens::{tokens, TokenBlockCtx};

    let chdr_inter = build_inter_chdr_for_test(ctx, tx_mode, reference_mode, comp_config);
    let seg = SegmentationParams::default_disabled();
    let mut state = Vp9FrameState::new(mi_rows, mi_cols);
    let mut nz = [
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
        NonzeroContext::new((2 * mi_cols) as usize, (2 * mi_rows) as usize),
    ];
    let mut seg_pred_ctx = SegPredContextState::new(mi_cols, mi_rows);
    let prev_seg_ids = vec![0u8; (mi_rows * mi_cols) as usize];

    let mut coder = BoolCoder::init_bool(data, data.len())?;
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
                update_map: false,
                temporal_update: false,
                tree_probs: None,
                pred_prob: None,
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
            uv_mode_probs: &crate::mode_info::DEFAULT_UV_MODE_PROBS,
        };

        let mi = inter_frame_mode_info(&mut coder, args, &mut seg_pred_ctx, r, c)?;
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
                        nonzero = tokens(
                            &mut coder,
                            &tbc,
                            tx_sz,
                            scan,
                            &chdr_inter.intra.coef_probs,
                            &nz[plane],
                            &mut token_cache[..seg_eob],
                            &mut tok_buf[..seg_eob],
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
