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
use crate::compressed::{parse_compressed_header, Vp9CompressedHeader};
use crate::decode_block::{decode_block_apply, DecodedBlockResult, Vp9FrameState};
use crate::dequant::{get_ac_quant, get_dc_quant, seg_feature_active};
use crate::frame_loop_filter::{frame_loop_filter, CurrFrame};
use crate::header::{
    parse_uncompressed_header, FrameType, QuantizationParams, SegmentationParams, Vp9FrameHeader,
};
use crate::intra::{predict_intra, Plane, PredMode};
use crate::loop_filter::loop_filter_frame_init;
use crate::mode_info::{
    intra_frame_mode_info, intra_segment_id, IntraFrameNeighbours, NeighbourSkips,
    NeighbourTxSizes, Vp9IntraMiBlock, DC_PRED, INTRA_FRAME, NONE_REF_FRAME, SEG_LVL_SKIP,
};
use crate::partition::{
    decode_partition, get_tile_offset, tile_payload_sizes, LeafSink, PartitionContextState,
    PartitionProbsKind,
};
use crate::reconstruct::{reconstruct_block, tx_type_for_intra};
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
    /// §6.4.4 `decode_block( r, c, subsize )` — the intra arm.
    fn decode_block(
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
}

/// Decode one intra (key or intra-only) VP9 frame to planar samples.
///
/// `data` is a single frame's byte payload (uncompressed header +
/// compressed header + tile data) — e.g. one IVF frame body. Inter
/// frames and `show_existing_frame` short frames return
/// [`Error::Unsupported`] (they need reference-buffer state).
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
    let chdr = parse_compressed_header(&data[ch_start..ch_end], lossless)?;

    // §7.2.6: MiCols / MiRows; §7.2 Sb64 extents for the §7.4.1 strip
    // spans (AbovePartitionContext is read beyond MiCols).
    let mi_cols = (hdr.frame_width + 7) >> 3;
    let mi_rows = (hdr.frame_height + 7) >> 3;
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

            let mut block_decoder = BlockDecoder {
                seg: &hdr.segmentation,
                quant: &hdr.quantization,
                chdr: &chdr,
                mi_rows,
                mi_cols,
                mi_col_start,
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
            };

            // §6.4.2 decode_tile( ): superblock raster with the
            // §7.4.2 clear_left_context( ) reset per superblock row
            // (LeftNonzeroContext + LeftPartitionContext).
            let mut r = mi_row_start;
            while r < mi_row_end {
                pctx.clear_left();
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
                        PartitionProbsKind::Keyframe,
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
    {
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

    Ok(Vp9DecodedFrame {
        width: hdr.frame_width,
        height: hdr.frame_height,
        bit_depth: hdr.color_config.bit_depth,
        subsampling_x: ssx,
        subsampling_y: ssy,
        y: crop(&plane_y, crop_w, crop_h),
        u: crop(&plane_u, uv_crop_w, uv_crop_h),
        v: crop(&plane_v, uv_crop_w, uv_crop_h),
    })
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
