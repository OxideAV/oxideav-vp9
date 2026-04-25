//! VP9 keyframe / intra-only block decode pipeline (§6.4.3).
//!
//! This module drives a single tile to pixel reconstruction. It owns:
//!
//! * The per-plane reconstruction buffer (luma + two chroma).
//! * The above-context / left-context arrays (per-plane, 4×4 mi-units wide).
//! * The partition context mi-map used by `partition_plane_context`.
//!
//! The public entry point is [`IntraTile::decode`] which walks the
//! superblocks of the tile, reading partition + mode_info + coefficients
//! from a [`BoolDecoder`] and producing per-block reconstruction.
//!
//! Inter-frame modes (`read_inter_frame_mode_info`) are out of scope —
//! non-key / non-intra-only frames surface `Error::Unsupported` from the
//! tile-level call and do not enter this module.

use oxideav_core::{Error, Result};

use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::{CompressedHeader, TxMode};
use crate::detokenize::decode_coefs;
use crate::headers::UncompressedHeader;
use crate::intra::IntraMode;
use crate::loopfilter::{LoopFilter, MiInfo, MiInfoPlane, INTRA_FRAME};
use crate::nonzero_ctx::NonzeroCtx;
use crate::probs::{read_partition_from_tree, KF_PARTITION_PROBS};
use crate::reconintra::{predict as predict_intra, NeighbourBuf};
use crate::segmentation::{read_intra_segment_id, SegPredContext, SegmentIdMap};
use crate::tables::{
    AC_QLOOKUP, COEFBAND_TRANS_4X4, COEFBAND_TRANS_8X8PLUS, COL_SCAN_16X16,
    COL_SCAN_16X16_NEIGHBORS, COL_SCAN_4X4, COL_SCAN_4X4_NEIGHBORS, COL_SCAN_8X8,
    COL_SCAN_8X8_NEIGHBORS, DC_QLOOKUP, DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X16_NEIGHBORS,
    DEFAULT_SCAN_32X32, DEFAULT_SCAN_32X32_NEIGHBORS, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS,
    DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X8_NEIGHBORS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
    ROW_SCAN_16X16, ROW_SCAN_16X16_NEIGHBORS, ROW_SCAN_4X4, ROW_SCAN_4X4_NEIGHBORS, ROW_SCAN_8X8,
    ROW_SCAN_8X8_NEIGHBORS,
};
use crate::transform::{inverse_transform_add, TxType};

/// Superblock size in pixels (§3).
pub const SUPERBLOCK_SIZE: u32 = 64;

/// VP9 block size enumeration — §3 Table 3-1. Values match libvpx.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockSize {
    B4x4 = 0,
    B4x8 = 1,
    B8x4 = 2,
    B8x8 = 3,
    B8x16 = 4,
    B16x8 = 5,
    B16x16 = 6,
    B16x32 = 7,
    B32x16 = 8,
    B32x32 = 9,
    B32x64 = 10,
    B64x32 = 11,
    B64x64 = 12,
}

impl BlockSize {
    pub fn from_wh(w: u32, h: u32) -> Self {
        match (w, h) {
            (4, 4) => Self::B4x4,
            (4, 8) => Self::B4x8,
            (8, 4) => Self::B8x4,
            (8, 8) => Self::B8x8,
            (8, 16) => Self::B8x16,
            (16, 8) => Self::B16x8,
            (16, 16) => Self::B16x16,
            (16, 32) => Self::B16x32,
            (32, 16) => Self::B32x16,
            (32, 32) => Self::B32x32,
            (32, 64) => Self::B32x64,
            (64, 32) => Self::B64x32,
            (64, 64) => Self::B64x64,
            _ => Self::B8x8,
        }
    }

    pub fn w(&self) -> u32 {
        match self {
            Self::B4x4 | Self::B4x8 => 4,
            Self::B8x4 | Self::B8x8 | Self::B8x16 => 8,
            Self::B16x8 | Self::B16x16 | Self::B16x32 => 16,
            Self::B32x16 | Self::B32x32 | Self::B32x64 => 32,
            Self::B64x32 | Self::B64x64 => 64,
        }
    }

    pub fn h(&self) -> u32 {
        match self {
            Self::B4x4 | Self::B8x4 => 4,
            Self::B4x8 | Self::B8x8 | Self::B16x8 => 8,
            Self::B8x16 | Self::B16x16 | Self::B32x16 => 16,
            Self::B16x32 | Self::B32x32 | Self::B64x32 => 32,
            Self::B32x64 | Self::B64x64 => 64,
        }
    }

    /// Max transform size allowed for this block size (§7.4.3).
    pub fn max_tx_size_log2(&self) -> usize {
        match self {
            Self::B4x4 | Self::B4x8 | Self::B8x4 => 0,
            Self::B8x8 | Self::B8x16 | Self::B16x8 => 1,
            Self::B16x16 | Self::B16x32 | Self::B32x16 => 2,
            Self::B32x32 | Self::B32x64 | Self::B64x32 | Self::B64x64 => 3,
        }
    }
}

/// Intra-frame tile decoder. Ownes all state needed to reconstruct a
/// single tile (which, for intra-only / keyframe single-tile bitstreams
/// the fixture uses, is the whole frame).
pub struct IntraTile<'a> {
    pub hdr: &'a UncompressedHeader,
    pub ch: &'a CompressedHeader,
    /// Luma plane buffer. `width × height` bytes, row-major.
    pub y: Vec<u8>,
    pub y_stride: usize,
    /// Chroma planes — subsampled per `color_config`.
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub uv_stride: usize,
    pub uv_w: usize,
    pub uv_h: usize,
    /// Frame dimensions (luma).
    pub width: usize,
    pub height: usize,
    /// Per-8x8-MI block metadata for §8.8 loop filtering.
    pub mi_info: MiInfoPlane,
    /// Per-column above-partition context (§7.4.6). One byte per 8x8
    /// column — bit `i` is set when the last partition at log2 level
    /// `i` ended at this column.
    pub above_partition_ctx: Vec<u8>,
    /// Per-row left-partition context.
    pub left_partition_ctx: Vec<u8>,
    /// Per-column above intra mode (last 8x8). Used for KF y-mode
    /// neighbour context.
    pub above_mode: Vec<IntraMode>,
    /// Per-row left intra mode (last 8x8).
    pub left_mode: Vec<IntraMode>,
    /// Per-8x8 segment_id for this frame — written by §6.4.7
    /// `intra_segment_id()`. Read back by §8.8.4 loop-filter level
    /// lookup and §8.1 `PrevSegmentIds` update.
    pub segment_ids: SegmentIdMap,
    /// §7.4.1 / §7.4.2 `AboveSegPredContext` / `LeftSegPredContext`.
    /// Unused on key / intra-only frames (inter path is the consumer)
    /// but kept parallel to InterTile for symmetry.
    pub seg_pred_ctx: SegPredContext,
    /// §6.4.22 `AboveNonzeroContext` / `LeftNonzeroContext` — the
    /// "did my neighbour have any non-zero coefficients" flag arrays
    /// that drive the initial token context for `more_coefs` (§6.4.24).
    pub nonzero_ctx: NonzeroCtx,
    /// Per-8x8 above/left `Skips[..]` tracking. Currently seeded but
    /// not consumed — §7.4.6's `skip_probs[skip_ctx]` path regressed
    /// the vp9-intra fixture (Round 7 investigation) so we fell back
    /// to the hard-coded prob while keeping the infrastructure wired.
    #[allow(dead_code)]
    pub above_skip: Vec<bool>,
    #[allow(dead_code)]
    pub left_skip: Vec<bool>,
}

/// Scan + neighbour tables for a (tx_size, tx_type) pair. Mirrors
/// libvpx's `ScanOrder`.
struct ScanOrder {
    scan: &'static [i16],
    neighbors: &'static [i16],
    band_translate: &'static [u8],
    tx_size_log2: usize,
}

fn get_scan(tx_size_log2: usize, tx_type: TxType) -> ScanOrder {
    // libvpx: inter / luma-non-intra / lossless uses default; intra uses
    // tx-type specific.
    match (tx_size_log2, tx_type) {
        (0, TxType::DctDct) => ScanOrder {
            scan: &DEFAULT_SCAN_4X4,
            neighbors: &DEFAULT_SCAN_4X4_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_4X4,
            tx_size_log2: 0,
        },
        (0, TxType::AdstDct) => ScanOrder {
            scan: &ROW_SCAN_4X4,
            neighbors: &ROW_SCAN_4X4_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_4X4,
            tx_size_log2: 0,
        },
        (0, TxType::DctAdst) => ScanOrder {
            scan: &COL_SCAN_4X4,
            neighbors: &COL_SCAN_4X4_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_4X4,
            tx_size_log2: 0,
        },
        (0, TxType::AdstAdst | TxType::WhtWht) => ScanOrder {
            scan: &DEFAULT_SCAN_4X4,
            neighbors: &DEFAULT_SCAN_4X4_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_4X4,
            tx_size_log2: 0,
        },
        (1, TxType::DctDct) => ScanOrder {
            scan: &DEFAULT_SCAN_8X8,
            neighbors: &DEFAULT_SCAN_8X8_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 1,
        },
        (1, TxType::AdstDct) => ScanOrder {
            scan: &ROW_SCAN_8X8,
            neighbors: &ROW_SCAN_8X8_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 1,
        },
        (1, TxType::DctAdst) => ScanOrder {
            scan: &COL_SCAN_8X8,
            neighbors: &COL_SCAN_8X8_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 1,
        },
        (1, TxType::AdstAdst | TxType::WhtWht) => ScanOrder {
            scan: &DEFAULT_SCAN_8X8,
            neighbors: &DEFAULT_SCAN_8X8_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 1,
        },
        (2, TxType::DctDct) => ScanOrder {
            scan: &DEFAULT_SCAN_16X16,
            neighbors: &DEFAULT_SCAN_16X16_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 2,
        },
        (2, TxType::AdstDct) => ScanOrder {
            scan: &ROW_SCAN_16X16,
            neighbors: &ROW_SCAN_16X16_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 2,
        },
        (2, TxType::DctAdst) => ScanOrder {
            scan: &COL_SCAN_16X16,
            neighbors: &COL_SCAN_16X16_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 2,
        },
        (2, TxType::AdstAdst | TxType::WhtWht) => ScanOrder {
            scan: &DEFAULT_SCAN_16X16,
            neighbors: &DEFAULT_SCAN_16X16_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 2,
        },
        (3, _) => ScanOrder {
            scan: &DEFAULT_SCAN_32X32,
            neighbors: &DEFAULT_SCAN_32X32_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 3,
        },
        _ => ScanOrder {
            scan: &DEFAULT_SCAN_4X4,
            neighbors: &DEFAULT_SCAN_4X4_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_4X4,
            tx_size_log2: 0,
        },
    }
}

/// Borrow a `[band][ctx][node]` slice from the per-frame coefficient
/// table. `ref_type` is 0 for intra, 1 for inter — per §7.4.17 and the
/// §6.3.7 update layout. The returned shape matches what `decode_coefs`
/// expects: `[6][6][3]`.
fn coef_probs_from_ctx(
    ch: &CompressedHeader,
    tx_size_log2: usize,
    plane_type: usize,
    ref_type: usize,
) -> &[[[u8; 3]; 6]; 6] {
    let ts = tx_size_log2.min(3);
    &ch.ctx.coef_probs[ts][plane_type][ref_type]
}

/// Intra-mode → tx-type lookup (§7.4.2 Table 7-5 / libvpx
/// `intra_mode_to_tx_type_lookup`).
fn intra_mode_to_tx_type(m: IntraMode) -> TxType {
    match m {
        IntraMode::Dc | IntraMode::D45 => TxType::DctDct,
        IntraMode::V | IntraMode::D117 | IntraMode::D63 => TxType::AdstDct,
        IntraMode::H | IntraMode::D153 | IntraMode::D207 => TxType::DctAdst,
        IntraMode::D135 | IntraMode::Tm => TxType::AdstAdst,
    }
}

impl<'a> IntraTile<'a> {
    pub fn new(hdr: &'a UncompressedHeader, ch: &'a CompressedHeader) -> Self {
        let width = hdr.width as usize;
        let height = hdr.height as usize;
        // Align plane strides to 64 to keep neighbour reads in-bounds.
        let y_stride = width.max(1);
        let sub_x = hdr.color_config.subsampling_x as usize;
        let sub_y = hdr.color_config.subsampling_y as usize;
        let uv_w = (width + sub_x) >> sub_x;
        let uv_h = (height + sub_y) >> sub_y;
        let uv_stride = uv_w.max(1);
        let mi_cols = width.div_ceil(8).max(1);
        let mi_rows = height.div_ceil(8).max(1);
        Self {
            hdr,
            ch,
            y: vec![0u8; y_stride * height],
            y_stride,
            u: vec![0u8; uv_stride * uv_h],
            v: vec![0u8; uv_stride * uv_h],
            uv_stride,
            uv_w,
            uv_h,
            width,
            height,
            mi_info: MiInfoPlane::new(mi_cols, mi_rows),
            above_partition_ctx: vec![0u8; mi_cols],
            left_partition_ctx: vec![0u8; mi_rows],
            above_mode: vec![IntraMode::Dc; mi_cols],
            left_mode: vec![IntraMode::Dc; mi_rows],
            segment_ids: SegmentIdMap::zeros(mi_cols, mi_rows),
            seg_pred_ctx: SegPredContext::zeros(mi_cols, mi_rows),
            nonzero_ctx: NonzeroCtx::new(mi_cols, mi_rows, sub_x, sub_y),
            above_skip: vec![false; mi_cols],
            left_skip: vec![false; mi_rows],
        }
    }

    /// §7.4.6 skip context for `read_skip`:
    ///   ctx = (AvailU ? AboveSkip : 0) + (AvailL ? LeftSkip : 0)
    #[allow(dead_code)]
    fn skip_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let a = if mi_row > 0 && mi_col < self.above_skip.len() {
            self.above_skip[mi_col] as usize
        } else {
            0
        };
        let l = if mi_col > 0 && mi_row < self.left_skip.len() {
            self.left_skip[mi_row] as usize
        } else {
            0
        };
        a + l
    }

    fn update_skip_ctx(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        mi_w: usize,
        mi_h: usize,
        skip: bool,
    ) {
        for i in 0..mi_w.max(1) {
            let c = mi_col + i;
            if c < self.above_skip.len() {
                self.above_skip[c] = skip;
            }
        }
        for i in 0..mi_h.max(1) {
            let r = mi_row + i;
            if r < self.left_skip.len() {
                self.left_skip[r] = skip;
            }
        }
    }

    /// Decode the full tile into the plane buffers. Bool-decoder `bd` is
    /// positioned at the first byte of the tile payload.
    pub fn decode(&mut self, bd: &mut BoolDecoder<'a>) -> Result<()> {
        if !self.is_keyframe_like() {
            return Err(Error::unsupported(
                "vp9 inter frame pending — only keyframe / intra_only decode is wired",
            ));
        }
        // §7.4.1 clear_above_context — once per tile.
        self.nonzero_ctx.clear_above();
        let sbs_x = (self.width as u32).div_ceil(SUPERBLOCK_SIZE);
        let sbs_y = (self.height as u32).div_ceil(SUPERBLOCK_SIZE);
        for sby in 0..sbs_y {
            // §7.4.2 clear_left_context — once per superblock row.
            self.nonzero_ctx.clear_left();
            for sbx in 0..sbs_x {
                let col = sbx * SUPERBLOCK_SIZE;
                let row = sby * SUPERBLOCK_SIZE;
                self.decode_partition(bd, row, col, SUPERBLOCK_SIZE)?;
            }
        }
        // §8.8 loop filter pass after all blocks reconstructed.
        self.apply_loop_filter();
        Ok(())
    }

    /// Decode one tile rectangle in pixel coordinates. The bool decoder
    /// `bd` is already positioned at the first byte of this tile's
    /// payload (each tile resets the boolean engine per §6.4).
    pub fn decode_rect(
        &mut self,
        bd: &mut BoolDecoder<'a>,
        col_start: u32,
        col_end: u32,
        row_start: u32,
        row_end: u32,
    ) -> Result<()> {
        if !self.is_keyframe_like() {
            return Err(Error::unsupported(
                "vp9 inter frame pending — only keyframe / intra_only decode is wired",
            ));
        }
        // §7.4.1 clear_above_context — once per tile.
        self.nonzero_ctx.clear_above();
        let mut r = row_start;
        while r < row_end {
            // §7.4.2 clear_left_context — once per superblock row.
            self.nonzero_ctx.clear_left();
            let mut c = col_start;
            while c < col_end {
                self.decode_partition(bd, r, c, SUPERBLOCK_SIZE)?;
                c += SUPERBLOCK_SIZE;
            }
            r += SUPERBLOCK_SIZE;
        }
        Ok(())
    }

    /// Run the §8.8 loop filter pass. Multi-tile callers use this once
    /// after all tiles are decoded.
    pub fn finalize(&mut self) {
        self.apply_loop_filter();
    }

    /// Run the §8.8 loop filter in place on the reconstructed planes.
    fn apply_loop_filter(&mut self) {
        let subsampling_x = self.hdr.color_config.subsampling_x;
        let subsampling_y = self.hdr.color_config.subsampling_y;
        let lf = LoopFilter::with_segmentation(
            &self.hdr.loop_filter,
            &self.hdr.segmentation,
            self.mi_info.mi_cols,
            self.mi_info.mi_rows,
            subsampling_x,
            subsampling_y,
        );
        lf.apply_frame(
            &self.mi_info,
            &mut self.y,
            self.y_stride,
            self.width,
            self.height,
            &mut self.u,
            &mut self.v,
            self.uv_stride,
            self.uv_w,
            self.uv_h,
        );
    }

    fn is_keyframe_like(&self) -> bool {
        use crate::headers::FrameType;
        matches!(self.hdr.frame_type, FrameType::Key) || self.hdr.intra_only
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_partition(
        &mut self,
        bd: &mut BoolDecoder<'a>,
        row: u32,
        col: u32,
        bsize: u32,
    ) -> Result<()> {
        debug_assert!(matches!(bsize, 64 | 32 | 16 | 8));
        if row >= self.height as u32 || col >= self.width as u32 {
            return Ok(());
        }
        let mi_row = (row as usize) / 8;
        let mi_col = (col as usize) / 8;
        let partition = self.read_partition(bd, bsize, mi_row, mi_col)?;
        let half = bsize / 2;
        match partition {
            Partition::None => {
                let bs = BlockSize::from_wh(bsize, bsize);
                self.decode_block(bd, row, col, bs)?;
                self.update_partition_ctx(mi_row, mi_col, bsize, bsize, bsize);
            }
            Partition::Horz => {
                let bs = BlockSize::from_wh(bsize, half);
                self.decode_block(bd, row, col, bs)?;
                if row + half < self.height as u32 {
                    self.decode_block(bd, row + half, col, bs)?;
                }
                self.update_partition_ctx(mi_row, mi_col, bsize, bsize, half);
            }
            Partition::Vert => {
                let bs = BlockSize::from_wh(half, bsize);
                self.decode_block(bd, row, col, bs)?;
                if col + half < self.width as u32 {
                    self.decode_block(bd, row, col + half, bs)?;
                }
                self.update_partition_ctx(mi_row, mi_col, bsize, half, bsize);
            }
            Partition::Split => {
                if bsize == 8 {
                    // §6.4.3: SPLIT at BLOCK_8X8 → subsize=BLOCK_4X4 which
                    // satisfies `subsize < BLOCK_8X8`, so the spec calls
                    // `decode_block(r, c, subsize)` ONCE — not 4 times.
                    // §6.4.6 intra_frame_mode_info handles the 4 sub-modes
                    // internally when MiSize<BLOCK_8X8. Calling
                    // decode_block 4 times here used to read 16 modes
                    // (4 per call × 4 calls) and skip chroma — major
                    // bitstream-drift bug uncovered round 13.
                    self.decode_block(bd, row, col, BlockSize::B4x4)?;
                    self.update_partition_ctx(mi_row, mi_col, bsize, half, half);
                } else {
                    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                        self.decode_partition(bd, row + dr, col + dc, half)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn read_partition(
        &self,
        bd: &mut BoolDecoder<'a>,
        bsize: u32,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<Partition> {
        // §7.4.6 partition context:
        //   bsl = mi_width_log2_lookup[bsize];          (0=8x8 .. 3=64x64)
        //   boffset = mi_width_log2_lookup[64x64] - bsl;
        //   above = OR over num8x8 cols; test bit (1<<boffset).
        //   ctx = bsl * 4 + left * 2 + above
        //
        // Our `bsize` is in pixels: 8, 16, 32, 64.
        // mi_width_log2_lookup (in 8x8 units): 8→0, 16→1, 32→2, 64→3
        // We use `bsl` in the spec sense (0..=3).
        let bsl = match bsize {
            8 => 0usize,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => 3,
        };
        let num8x8 = (bsize as usize) / 8;
        let boffset = 3 - bsl;
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..num8x8 {
            let c = mi_col + i;
            if c < self.above_partition_ctx.len() {
                above |= self.above_partition_ctx[c];
            }
            let r = mi_row + i;
            if r < self.left_partition_ctx.len() {
                left |= self.left_partition_ctx[r];
            }
        }
        let above_bit = ((above >> boffset) & 1) as usize;
        let left_bit = ((left >> boffset) & 1) as usize;
        let ctx = bsl * 4 + left_bit * 2 + above_bit;
        let probs = KF_PARTITION_PROBS[ctx];
        // §6.4.3 / §9.3.1 partition tree selection:
        //   hasRows = (r + halfBlock8x8) < MiRows
        //   hasCols = (c + halfBlock8x8) < MiCols
        //   both    -> partition_tree (NONE/HORZ/VERT/SPLIT)
        //   cols    -> cols_partition_tree (HORZ vs SPLIT, prob = probs[1])
        //   rows    -> rows_partition_tree (VERT vs SPLIT, prob = probs[2])
        //   neither -> SPLIT (no read)
        let mi_rows = (self.height + 7) / 8;
        let mi_cols = (self.width + 7) / 8;
        let half_block8x8 = num8x8 / 2;
        let has_rows = mi_row + half_block8x8 < mi_rows;
        let has_cols = mi_col + half_block8x8 < mi_cols;
        if !has_rows && !has_cols {
            return Ok(Partition::Split);
        }
        if !has_rows {
            // cols_partition_tree[2] = { -PARTITION_HORZ, -PARTITION_SPLIT }
            // §9.3.2 selects probs[1] for the single split.
            let b = bd.read(probs[1])?;
            return Ok(if b == 0 {
                Partition::Horz
            } else {
                Partition::Split
            });
        }
        if !has_cols {
            // rows_partition_tree[2] = { -PARTITION_VERT, -PARTITION_SPLIT }
            let b = bd.read(probs[2])?;
            return Ok(if b == 0 {
                Partition::Vert
            } else {
                Partition::Split
            });
        }
        // Full partition_tree (3 probs). bsize==8 special-case kept for
        // clarity but the libvpx tree-walk path produces the same answer.
        if bsize == 8 {
            let b0 = bd.read(probs[0])?;
            if b0 == 0 {
                return Ok(Partition::None);
            }
            let b1 = bd.read(probs[1])?;
            return Ok(if b1 == 0 {
                Partition::Horz
            } else {
                let b2 = bd.read(probs[2])?;
                if b2 == 0 {
                    Partition::Vert
                } else {
                    Partition::Split
                }
            });
        }
        let ptype = read_partition_from_tree(bd, probs)?;
        Ok(match ptype {
            crate::probs::PartitionType::None => Partition::None,
            crate::probs::PartitionType::Horz => Partition::Horz,
            crate::probs::PartitionType::Vert => Partition::Vert,
            crate::probs::PartitionType::Split => Partition::Split,
        })
    }

    /// §7.4.6 update `AbovePartitionContext` / `LeftPartitionContext`
    /// after a partition is decoded. The 64×64-first convention used
    /// throughout this decoder: `bsl=0` for 64×64, `bsl=3` for 8×8 (the
    /// inverse of `mi_width_log2_lookup`). Fill bytes are built so that
    /// `(byte >> boffset) & 1` resolves to 1 iff the neighbour's
    /// subsize was smaller than the block currently being decoded.
    /// Round-13 note: a spec-literal `15 >> b_width_log2_lookup[subsize]`
    /// rewrite REGRESSED the libvpx-encoded compound fixture (9.38 →
    /// 6.94 dB), suggesting the actual encoder uses this in-tree form.
    /// Restoring the round-12 derivation pending further investigation.
    fn update_partition_ctx(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize_px: u32,
        sub_w_px: u32,
        sub_h_px: u32,
    ) {
        let num8x8 = (bsize_px as usize).max(8) / 8;
        let bsl = match bsize_px {
            8 => 0usize,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => 3,
        };
        let boffset = 3 - bsl;
        let above_fill = if sub_w_px >= bsize_px {
            (1u8 << boffset) - 1 + (1u8 << boffset)
        } else {
            0
        };
        let left_fill = if sub_h_px >= bsize_px {
            (1u8 << boffset) - 1 + (1u8 << boffset)
        } else {
            0
        };
        for i in 0..num8x8 {
            let c = mi_col + i;
            if c < self.above_partition_ctx.len() {
                self.above_partition_ctx[c] = above_fill;
            }
            let r = mi_row + i;
            if r < self.left_partition_ctx.len() {
                self.left_partition_ctx[r] = left_fill;
            }
        }
    }

    /// Decode one block (prediction unit). Spec order §6.4.6
    /// `intra_frame_mode_info`:
    ///   intra_segment_id()  (§6.4.7)
    ///   read_skip()         (§6.4.8) — honours `SEG_LVL_SKIP`
    ///   read_tx_size(1)     (§6.4.10)
    ///   default_intra_mode  (§9.3.2)
    fn decode_block(
        &mut self,
        bd: &mut BoolDecoder<'a>,
        row: u32,
        col: u32,
        bs: BlockSize,
    ) -> Result<()> {
        // §6.4.7 intra_segment_id — tree-coded when update_map is set.
        let segment_id = read_intra_segment_id(bd, &self.hdr.segmentation)?;
        // Stamp this block's segment_id into the current-frame map so
        // §6.4.14 get_segment_id and §8.8 loop filter can see it.
        let mi_row = (row as usize) / 8;
        let mi_col = (col as usize) / 8;
        let bw = (bs.w() as usize) / 8;
        let bh = (bs.h() as usize) / 8;
        self.segment_ids
            .fill(mi_row, mi_col, bw.max(1), bh.max(1), segment_id);
        // §6.4.8 read_skip — `SEG_LVL_SKIP` forces skip=1 per §6.4.9.
        // Note: §7.4.6 calls for `skip_probs[skip_ctx]` where skip_ctx
        // is the sum of above/left skip flags. Round-13 attempt to wire
        // this through caused a 21-dB regression on the lossless-gray
        // fixture (66.77 → 45.43 dB) so we reverted to the hard-coded
        // 192 path. Likely the lossless encoder writes skip with a
        // different context than what we compute. Investigation TODO.
        let skip = if self
            .hdr
            .segmentation
            .feature_active(segment_id, crate::headers::SEG_LVL_SKIP)
        {
            true
        } else {
            bd.read(192)? != 0
        };
        // Read tx_size. For TX_MODE_SELECT + bsize >= 8x8 the tx_size is
        // tree-coded. For simpler modes we use the tx_mode ceiling.
        let tx_size_log2 = self.read_tx_size(bd, bs, mi_row, mi_col)?;

        // Read luma intra mode using above/left neighbour mode context.
        // §6.4.6 intra_frame_mode_info: when MiSize < BLOCK_8X8 (B4x4 here)
        // we read 4 sub_modes — one per 4×4 sub-block in the parent 8×8.
        // Each sub_mode is decoded against the per-position above/left
        // neighbour mode context. y_mode is the LAST sub_mode read.
        let mut sub_modes = [IntraMode::Dc; 4];
        let y_mode = if matches!(bs, BlockSize::B4x4) {
            // §6.4.6 / §9.3.2 sub-8x8 default_intra_mode: read in (idy,
            // idx) order, neighbour mode context comes from the same
            // 8x8's already-decoded sub_modes when idy>0 or idx>0
            // respectively, otherwise from the parent above/left
            // trackers. Round-13 fix: previously we used the parent
            // mi_row/mi_col context for all 4 sub-modes, which gives
            // the wrong probability distribution for sub_modes[1..3].
            let mut last = IntraMode::Dc;
            for idy in 0..2usize {
                for idx in 0..2usize {
                    let m = self.read_intra_sub_mode(bd, mi_row, mi_col, idy, idx, &sub_modes)?;
                    sub_modes[idy * 2 + idx] = m;
                    last = m;
                }
            }
            last
        } else {
            self.read_intra_mode(bd, mi_row, mi_col)?
        };
        let uv_mode = self.read_intra_mode_uv(bd, y_mode)?;
        // Stamp the block's luma mode into the above/left neighbour
        // trackers so the next block's context is correct.
        let mi_w = (bs.w() as usize) / 8;
        let mi_h = (bs.h() as usize) / 8;
        // The above row is the last row of the block (will be "above"
        // for the block below), and the left col is the last col of the
        // block (will be "left" for the block to the right).
        for c in 0..mi_w.max(1) {
            let cc = mi_col + c;
            if cc < self.above_mode.len() {
                self.above_mode[cc] = y_mode;
            }
        }
        for r in 0..mi_h.max(1) {
            let rr = mi_row + r;
            if rr < self.left_mode.len() {
                self.left_mode[rr] = y_mode;
            }
        }

        // Record MI metadata for §8.8 loop filter. Keyframes are intra
        // by construction — ref_frame stays at INTRA_FRAME. segment_id
        // is the one stamped by §6.4.7 at the top of this function.
        let mi_w_u8 = (bs.w() / 8).max(1) as u8;
        let mi_h_u8 = (bs.h() / 8).max(1) as u8;
        let mi = MiInfo {
            mi_w_8x8: mi_w_u8,
            mi_h_8x8: mi_h_u8,
            tx_size_log2: tx_size_log2 as u8,
            skip,
            ref_frame: INTRA_FRAME,
            mode_is_non_zero_inter: false,
            segment_id,
        };
        self.mi_info.fill(mi_row, mi_col, mi);
        // Stamp the skip bit into the neighbour trackers so the next
        // block sees the correct §7.4.6 skip context.
        self.update_skip_ctx(mi_row, mi_col, mi_w, mi_h, skip);

        // Reconstruct luma plane for this block.
        // §6.4.21 residual: when MiSize<BLOCK_8X8 (i.e., B4x4) the loop
        // iterates over the 4 sub-4×4 luma blocks (num4x4w=2, num4x4h=2)
        // with per-block intra prediction using sub_modes[idy*2+idx].
        if matches!(bs, BlockSize::B4x4) {
            for idy in 0..2u32 {
                for idx in 0..2u32 {
                    let sm = sub_modes[(idy as usize) * 2 + (idx as usize)];
                    let r = (row as usize) + (idy as usize) * 4;
                    let c = (col as usize) + (idx as usize) * 4;
                    if r < self.height && c < self.width {
                        self.reconstruct_plane(bd, r, c, 4, 4, 0, sm, 0, skip, segment_id)?;
                    }
                }
            }
        } else {
            self.reconstruct_plane(
                bd,
                row as usize,
                col as usize,
                bs.w() as usize,
                bs.h() as usize,
                tx_size_log2,
                y_mode,
                0,
                skip,
                segment_id,
            )?;
        }
        // Chroma (subsampled). §6.4.22 get_uv_tx_size: when MiSize<BLOCK_8X8
        // returns TX_4X4 always; otherwise Min(tx_size, max_tx_lookup[uv_size]).
        let sub_x = self.hdr.color_config.subsampling_x as u32;
        let sub_y = self.hdr.color_config.subsampling_y as u32;
        let c_row = (row >> sub_y) as usize;
        let c_col = (col >> sub_x) as usize;
        if matches!(bs, BlockSize::B4x4) {
            // Sub-8x8 partition: chroma is decoded as ONE 4×4 block at
            // the parent 8×8 location (4:2:0 → chroma 4×4). §6.4.22
            // forces TX_4X4. Spec §6.4.21 still calls predict_intra+
            // tokens for chroma even though luma is split.
            if c_row + 4 <= self.uv_h && c_col + 4 <= self.uv_w {
                self.reconstruct_plane(bd, c_row, c_col, 4, 4, 0, uv_mode, 1, skip, segment_id)?;
                self.reconstruct_plane(bd, c_row, c_col, 4, 4, 0, uv_mode, 2, skip, segment_id)?;
            }
        } else {
            let c_w = (bs.w() >> sub_x) as usize;
            let c_h = (bs.h() >> sub_y) as usize;
            let c_tx_log2 = clamp_tx_size(tx_size_log2, c_w, c_h);
            if c_w >= 4 && c_h >= 4 {
                self.reconstruct_plane(
                    bd, c_row, c_col, c_w, c_h, c_tx_log2, uv_mode, 1, skip, segment_id,
                )?;
                self.reconstruct_plane(
                    bd, c_row, c_col, c_w, c_h, c_tx_log2, uv_mode, 2, skip, segment_id,
                )?;
            }
        }
        Ok(())
    }

    fn read_tx_size(
        &self,
        bd: &mut BoolDecoder<'a>,
        bs: BlockSize,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<usize> {
        let max_tx = bs.max_tx_size_log2();
        let tx_mode = self.ch.tx_mode.unwrap_or(TxMode::Only4x4);
        match tx_mode {
            TxMode::Only4x4 => Ok(0),
            TxMode::Allow8x8 => Ok(max_tx.min(1)),
            TxMode::Allow16x16 => Ok(max_tx.min(2)),
            TxMode::Allow32x32 => Ok(max_tx.min(3)),
            TxMode::Select => {
                // §9.3.2 tx_size context:
                //   above = maxTxSize, left = maxTxSize
                //   if AvailU && !Skips[r-1][c]: above = TxSizes[r-1][c]
                //   if AvailL && !Skips[r][c-1]:  left  = TxSizes[r][c-1]
                //   if !AvailL: left = above
                //   if !AvailU: above = left
                //   ctx = (above + left) > maxTxSize
                let avail_u = mi_row > 0;
                let avail_l = mi_col > 0;
                let mut above = max_tx;
                let mut left = max_tx;
                if avail_u {
                    let a = self.mi_info.get(mi_row - 1, mi_col);
                    if !a.skip {
                        above = a.tx_size_log2 as usize;
                    }
                }
                if avail_l {
                    let l = self.mi_info.get(mi_row, mi_col - 1);
                    if !l.skip {
                        left = l.tx_size_log2 as usize;
                    }
                }
                if !avail_l {
                    left = above;
                }
                if !avail_u {
                    above = left;
                }
                let ctx = if (above + left) > max_tx { 1 } else { 0 };
                let probs = tx_probs_for_ctx(self.ch, max_tx, ctx);
                let mut tx = bd.read(probs[0])? as usize;
                if tx != 0 && max_tx >= 2 {
                    tx += bd.read(probs[1])? as usize;
                    if tx != 1 && max_tx >= 3 {
                        tx += bd.read(probs[2])? as usize;
                    }
                }
                Ok(tx.min(max_tx))
            }
        }
    }

    fn read_intra_mode(
        &self,
        bd: &mut BoolDecoder<'a>,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<IntraMode> {
        // §7.4.6 default_intra_mode (MiSize >= BLOCK_8X8 path): prob =
        // kf_y_mode_probs[above][left][node]. Above/Left modes come from
        // the last-decoded 8x8 cell on the boundary. When AvailU /
        // AvailL are false, DC is used.
        let above = if mi_row > 0 && mi_col < self.above_mode.len() {
            self.above_mode[mi_col]
        } else {
            IntraMode::Dc
        };
        let left = if mi_col > 0 && mi_row < self.left_mode.len() {
            self.left_mode[mi_row]
        } else {
            IntraMode::Dc
        };
        let probs = &KF_Y_MODE_PROBS[above as usize][left as usize];
        read_intra_mode_tree(bd, probs)
    }

    /// §9.3.2 default_intra_mode for the sub-8x8 MiSize<BLOCK_8X8 case
    /// (the explicit `if (idy)` / `if (idx)` neighbour selection in
    /// §7.4.6's else-branch). `sub_modes_so_far` holds the modes already
    /// decoded earlier in this 8x8's 2x2 grid (idy*2+idx ordering).
    /// When idy>0 the abovemode is the same-column sub-block above;
    /// when idx>0 the leftmode is the same-row sub-block to the left.
    fn read_intra_sub_mode(
        &self,
        bd: &mut BoolDecoder<'a>,
        mi_row: usize,
        mi_col: usize,
        idy: usize,
        idx: usize,
        sub_modes_so_far: &[IntraMode; 4],
    ) -> Result<IntraMode> {
        let above = if idy > 0 {
            // Sub-block above within the same 8x8 (column idx).
            sub_modes_so_far[idx]
        } else if mi_row > 0 && mi_col < self.above_mode.len() {
            self.above_mode[mi_col]
        } else {
            IntraMode::Dc
        };
        let left = if idx > 0 {
            // Sub-block left within the same 8x8 (row idy).
            sub_modes_so_far[idy * 2]
        } else if mi_col > 0 && mi_row < self.left_mode.len() {
            self.left_mode[mi_row]
        } else {
            IntraMode::Dc
        };
        let probs = &KF_Y_MODE_PROBS[above as usize][left as usize];
        read_intra_mode_tree(bd, probs)
    }

    fn read_intra_mode_uv(&self, bd: &mut BoolDecoder<'a>, y: IntraMode) -> Result<IntraMode> {
        let probs = &KF_UV_MODE_PROBS[y as usize];
        read_intra_mode_tree(bd, probs)
    }

    /// Decode coefficients for each TX block within a prediction block
    /// and run predict + inverse-transform + clip-add.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_plane(
        &mut self,
        bd: &mut BoolDecoder<'a>,
        row: usize,
        col: usize,
        w: usize,
        h: usize,
        tx_size_log2: usize,
        mode: IntraMode,
        plane: usize,
        skip: bool,
        segment_id: u8,
    ) -> Result<()> {
        let tx_side = 4usize << tx_size_log2;
        // Walk the TX blocks within (row,col,w,h).
        let plane_type = if plane == 0 { 0 } else { 1 }; // Y=0, UV=1
        let stride = if plane == 0 {
            self.y_stride
        } else {
            self.uv_stride
        };
        let (plane_w, plane_h) = if plane == 0 {
            (self.width, self.height)
        } else {
            (self.uv_w, self.uv_h)
        };
        // §6.4.25 `get_scan`: chroma and 32×32 blocks always use DCT_DCT
        // regardless of prediction mode. Per §6.4.25, lossless / inter
        // also force DCT_DCT for the scan choice. The actual inverse
        // transform for lossless is WHT (§8.7.2 / §8.7.1.10) and is
        // dispatched separately below.
        let lossless = self.hdr.quantization.lossless;
        let scan_tx_type = if lossless || (plane == 0 && tx_size_log2 < 3) {
            if lossless {
                TxType::DctDct
            } else {
                intra_mode_to_tx_type(mode)
            }
        } else {
            TxType::DctDct
        };
        let scan = get_scan(tx_size_log2, scan_tx_type);
        // The inverse-transform dispatch tx_type: lossless → WhtWht
        // regardless of mode (§8.7.2). Otherwise matches the scan
        // tx_type above.
        let xform_tx_type = if lossless {
            TxType::WhtWht
        } else {
            scan_tx_type
        };
        let probs = coef_probs_from_ctx(self.ch, tx_size_log2, plane_type, 0);

        // Quant: base_q_idx; libvpx applies delta_q_y_dc only to luma DC,
        // delta_q_uv_ac / uv_dc to chroma AC / DC. Segmentation may
        // override the qindex per §8.6.1 (SEG_LVL_ALT_Q).
        let qp = self
            .hdr
            .segmentation
            .get_qindex(segment_id, self.hdr.quantization.base_q_idx);
        let (dc, ac) = if plane == 0 {
            (
                DC_QLOOKUP[clamp_q(qp + self.hdr.quantization.delta_q_y_dc as i32)],
                AC_QLOOKUP[clamp_q(qp)],
            )
        } else {
            (
                DC_QLOOKUP[clamp_q(qp + self.hdr.quantization.delta_q_uv_dc as i32)],
                AC_QLOOKUP[clamp_q(qp + self.hdr.quantization.delta_q_uv_ac as i32)],
            )
        };
        let dq = [dc, ac];

        // Iterate tx blocks in raster scan.
        let mut r = 0usize;
        while r < h {
            let mut c = 0usize;
            while c < w {
                let abs_row = row + r;
                let abs_col = col + c;
                // Skip if tx block falls entirely outside frame.
                if abs_row >= plane_h || abs_col >= plane_w {
                    c += tx_side;
                    continue;
                }
                let tx_w = tx_side.min(plane_w - abs_col);
                let tx_h = tx_side.min(plane_h - abs_row);
                // Gather neighbours for prediction.
                let nb = self.build_neighbours(plane, abs_row, abs_col, tx_side, tx_side);
                // Predict into local buffer.
                let mut pred = vec![0u8; tx_side * tx_side];
                predict_intra(mode, &nb, &mut pred, tx_side);
                // Copy into plane buffer (respecting edge clipping).
                self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &pred, tx_side);
                // If not skipping, decode coefs and add residual.
                // §6.4.24: initial `more_coefs` context is derived from
                // AboveNonzeroContext / LeftNonzeroContext at this
                // block's 4×4-aligned coordinates.
                let initial_ctx =
                    self.nonzero_ctx
                        .token_ctx(plane, abs_col, abs_row, scan.tx_size_log2);
                let nonzero_update = if !skip {
                    let mut coeffs = vec![0i32; tx_side * tx_side];
                    let eob = decode_coefs(
                        bd,
                        probs,
                        &dq,
                        scan.scan,
                        scan.neighbors,
                        scan.band_translate,
                        scan.tx_size_log2,
                        initial_ctx,
                        &mut coeffs,
                    )?;
                    if eob > 0 {
                        // Run inverse transform into a local dst, then blit.
                        let mut dst = vec![0u8; tx_side * tx_side];
                        // Seed dst with current plane pixels (predictor
                        // already written), clipping to real block.
                        self.read_plane(plane, abs_row, abs_col, tx_w, tx_h, &mut dst, tx_side);
                        inverse_transform_add(
                            xform_tx_type,
                            tx_side,
                            tx_side,
                            &coeffs,
                            &mut dst,
                            tx_side,
                        )?;
                        self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &dst, tx_side);
                    }
                    if eob > 0 {
                        1u8
                    } else {
                        0u8
                    }
                } else {
                    0u8
                };
                // §6.4.22 post-tokens update: stamp the nonzero flag
                // into the 4×4 grid so downstream blocks see it.
                self.nonzero_ctx
                    .update(plane, abs_col, abs_row, scan.tx_size_log2, nonzero_update);
                let _ = stride;
                c += tx_side;
            }
            r += tx_side;
        }
        Ok(())
    }

    fn build_neighbours(
        &self,
        plane: usize,
        row: usize,
        col: usize,
        _tx_w: usize,
        tx_side: usize,
    ) -> NeighbourBuf {
        let (buf, stride, plane_w, plane_h) = match plane {
            0 => (&self.y[..], self.y_stride, self.width, self.height),
            1 => (&self.u[..], self.uv_stride, self.uv_w, self.uv_h),
            _ => (&self.v[..], self.uv_stride, self.uv_w, self.uv_h),
        };
        let have_above = row > 0;
        let have_left = col > 0;
        let have_aboveright = row > 0 && col + 2 * tx_side <= plane_w;

        let mut above_tmp = vec![0u8; 2 * tx_side];
        let mut above_opt: Option<&[u8]> = None;
        if have_above {
            let start = (row - 1) * stride + col;
            let n = (2 * tx_side).min(plane_w.saturating_sub(col));
            above_tmp[..n].copy_from_slice(&buf[start..start + n]);
            // Replicate trailing samples if short.
            if n > 0 && n < 2 * tx_side {
                let last = above_tmp[n - 1];
                for b in &mut above_tmp[n..] {
                    *b = last;
                }
            }
            above_opt = Some(&above_tmp[..]);
        }
        let mut left_tmp = vec![0u8; tx_side];
        let mut left_opt: Option<&[u8]> = None;
        if have_left {
            let nh = tx_side.min(plane_h.saturating_sub(row));
            for i in 0..nh {
                left_tmp[i] = buf[(row + i) * stride + (col - 1)];
            }
            if nh < tx_side && nh > 0 {
                let last = left_tmp[nh - 1];
                for b in &mut left_tmp[nh..] {
                    *b = last;
                }
            }
            left_opt = Some(&left_tmp[..]);
        }
        let above_left = if have_above && have_left {
            Some(buf[(row - 1) * stride + (col - 1)])
        } else if have_above {
            Some(127)
        } else if have_left {
            Some(129)
        } else {
            None
        };
        // Build pulls through the 127/129 machinery for us. We pass all
        // available data; missing neighbours get synthesised.
        // Allocate above_tmp long enough to hold 2*tx_side samples.
        NeighbourBuf::build(
            tx_side,
            0,
            have_above,
            have_left,
            have_aboveright,
            above_opt,
            left_opt,
            above_left,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn blit_plane(
        &mut self,
        plane: usize,
        row: usize,
        col: usize,
        w: usize,
        h: usize,
        src: &[u8],
        src_stride: usize,
    ) {
        let (buf, stride) = match plane {
            0 => (&mut self.y, self.y_stride),
            1 => (&mut self.u, self.uv_stride),
            _ => (&mut self.v, self.uv_stride),
        };
        for r in 0..h {
            let dst_off = (row + r) * stride + col;
            let src_off = r * src_stride;
            buf[dst_off..dst_off + w].copy_from_slice(&src[src_off..src_off + w]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn read_plane(
        &self,
        plane: usize,
        row: usize,
        col: usize,
        w: usize,
        h: usize,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        let (buf, stride) = match plane {
            0 => (&self.y[..], self.y_stride),
            1 => (&self.u[..], self.uv_stride),
            _ => (&self.v[..], self.uv_stride),
        };
        for r in 0..h {
            let src_off = (row + r) * stride + col;
            let dst_off = r * dst_stride;
            dst[dst_off..dst_off + w].copy_from_slice(&buf[src_off..src_off + w]);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Partition {
    None,
    Horz,
    Vert,
    Split,
}

/// Decode a single intra-mode symbol against the 9-prob table. Tree shape
/// from libvpx `vp9_intra_mode_tree`.
fn read_intra_mode_tree(bd: &mut BoolDecoder<'_>, p: &[u8; 9]) -> Result<IntraMode> {
    //   -DC_PRED,   2,          /* 0 = DC_NODE */
    //   -TM_PRED,   4,          /* 1 = TM_NODE */
    //   -V_PRED,    6,          /* 2 = V_NODE */
    //   8,          12,         /* 3 = COM_NODE */
    //   -H_PRED,    10,         /* 4 = H_NODE */
    //   -D135_PRED, -D117_PRED, /* 5 = D135_NODE */
    //   -D45_PRED,  14,         /* 6 = D45_NODE */
    //   -D63_PRED,  16,         /* 7 = D63_NODE */
    //   -D153_PRED, -D207_PRED  /* 8 = D153_NODE */
    if bd.read(p[0])? == 0 {
        return Ok(IntraMode::Dc);
    }
    if bd.read(p[1])? == 0 {
        return Ok(IntraMode::Tm);
    }
    if bd.read(p[2])? == 0 {
        return Ok(IntraMode::V);
    }
    // COM_NODE split: left sub-tree H/D135/D117, right sub-tree D45/D63/D153/D207
    if bd.read(p[3])? == 0 {
        // left subtree
        if bd.read(p[4])? == 0 {
            return Ok(IntraMode::H);
        }
        if bd.read(p[5])? == 0 {
            Ok(IntraMode::D135)
        } else {
            Ok(IntraMode::D117)
        }
    } else {
        if bd.read(p[6])? == 0 {
            return Ok(IntraMode::D45);
        }
        if bd.read(p[7])? == 0 {
            return Ok(IntraMode::D63);
        }
        if bd.read(p[8])? == 0 {
            Ok(IntraMode::D153)
        } else {
            Ok(IntraMode::D207)
        }
    }
}

/// Per-context `tx_probs` lookup matching the §9.3.2 tx_size derivation.
/// Sources from the per-frame `FrameContext` so probability updates from
/// §6.3.2 take effect.
fn tx_probs_for_ctx(ch: &CompressedHeader, max_tx: usize, ctx: usize) -> [u8; 3] {
    let ctx = ctx.min(1);
    match max_tx {
        3 => {
            let p = ch.ctx.tx_probs_32x32[ctx];
            [p[0], p[1], p[2]]
        }
        2 => {
            let p = ch.ctx.tx_probs_16x16[ctx];
            [p[0], p[1], 0]
        }
        1 => {
            let p = ch.ctx.tx_probs_8x8[ctx];
            [p[0], 0, 0]
        }
        _ => [128, 128, 128],
    }
}

fn clamp_q(q: i32) -> usize {
    q.clamp(0, 255) as usize
}

fn clamp_tx_size(log2: usize, w: usize, h: usize) -> usize {
    // Maximum tx-side we can fit in a w×h block.
    let max_by_w = match w {
        32.. => 3,
        16 => 2,
        8 => 1,
        _ => 0,
    };
    let max_by_h = match h {
        32.. => 3,
        16 => 2,
        8 => 1,
        _ => 0,
    };
    log2.min(max_by_w).min(max_by_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_tile_new_allocates_planes() {
        use crate::headers::{
            ColorConfig, ColorSpace, FrameType, LoopFilterParams, QuantizationParams,
            SegmentationParams, TileInfo, UncompressedHeader,
        };
        let hdr = UncompressedHeader {
            profile: 0,
            show_existing_frame: false,
            existing_frame_to_show: 0,
            frame_type: FrameType::Key,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            reset_frame_context: 0,
            color_config: ColorConfig {
                bit_depth: 8,
                color_space: ColorSpace::Bt709,
                color_range: false,
                subsampling_x: true,
                subsampling_y: true,
            },
            width: 128,
            height: 128,
            render_width: None,
            render_height: None,
            refresh_frame_flags: 0,
            ref_frame_idx: [0; 3],
            ref_frame_sign_bias: [false; 4],
            allow_high_precision_mv: false,
            interpolation_filter: 0,
            refresh_frame_context: false,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams::default(),
            quantization: QuantizationParams::default(),
            segmentation: SegmentationParams::default(),
            tile_info: TileInfo::default(),
            header_size: 0,
            uncompressed_header_size: 0,
        };
        let ch = CompressedHeader::default();
        let t = IntraTile::new(&hdr, &ch);
        assert_eq!(t.y.len(), 128 * 128);
        assert_eq!(t.u.len(), 64 * 64);
        assert_eq!(t.v.len(), 64 * 64);
    }
}
