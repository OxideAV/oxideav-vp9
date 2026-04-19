//! VP9 inter-frame tile decoder (§6.4 + §8.5 / §8.6).
//!
//! The symmetric sibling of [`crate::block::IntraTile`], but for non-key
//! / non-intra-only frames. Walks the same superblock / partition
//! quadtree (§6.4.2) and, for each leaf, either
//!
//! * runs the intra path (same primitives as intra-only frames — §8.5.1
//!   prediction + §8.7.1 inverse transform + clip-add), or
//! * runs the **inter** path: select reference frame (§6.2), decode the
//!   motion vector (§6.4.19), interpolate a reference block with the
//!   §8.5.1 8-tap sub-pel filter, optionally add a residual decoded from
//!   §6.4.23 tokens.
//!
//! Simplifications used in this first inter revision — documented in the
//! README's "Deferred" section:
//!
//! * **Compound prediction (§6.4.20)** — forced off; `reference_mode`
//!   must be SINGLE for each block.
//! * **Scaled references (§8.5.4)** — assumed off; reference dimensions
//!   must equal the current frame.
//! * **Neighbour-aware probability contexts** — context 0 everywhere
//!   (same compromise as the keyframe path).
//! * **Segmentation deltas** — never applied.

use oxideav_core::{Error, Result};

use crate::block::{BlockSize, SUPERBLOCK_SIZE};
use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::{CompressedHeader, TxMode};
use crate::detokenize::decode_coefs;
use crate::dpb::RefFrame;
use crate::headers::UncompressedHeader;
use crate::intra::IntraMode;
use crate::mcfilter::{mc_block, InterpFilter, RefSampler};
use crate::mv::{read_mv_component, read_mv_joint, DEFAULT_MV_COMP_PROBS, MV_JOINT_PROBS};
use crate::probs::{read_partition_from_tree, PARTITION_PROBS};
use crate::reconintra::{predict as predict_intra, NeighbourBuf};
use crate::tables::{
    AC_QLOOKUP, COEFBAND_TRANS_4X4, COEFBAND_TRANS_8X8PLUS, COEF_PROBS_16X16, COEF_PROBS_32X32,
    COEF_PROBS_4X4, COEF_PROBS_8X8, DC_QLOOKUP, DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X16_NEIGHBORS,
    DEFAULT_SCAN_32X32, DEFAULT_SCAN_32X32_NEIGHBORS, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS,
    DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X8_NEIGHBORS,
};
use crate::transform::{inverse_transform_add, TxType};

/// Default-probability approximation for the `is_inter` flag on
/// non-key frames. libvpx's per-context table lives in
/// `vp9_entropymode.c`; for a single-context fallback we use the
/// average of the four contexts (≈ 195).
const DEFAULT_IS_INTER_PROB: u8 = 195;

/// Default skip probability (§10.5 `default_skip_probs`). Again we
/// collapse the three contexts to their average.
const DEFAULT_SKIP_PROB: u8 = 192;

/// Inter mode tree defaults (§10.5 `default_inter_mode_probs`, ctx 0).
/// Layout: `[p_non_zeromv, p_not_nearestmv, p_not_nearmv]`.
const DEFAULT_INTER_MODE_PROBS: [u8; 3] = [2, 173, 34];

/// `comp_mode`-ignored single-reference selection prob (§10.5 ctx 0).
const DEFAULT_SINGLE_REF_PROB: [u8; 2] = [33, 16];

/// Default interp-filter selection probabilities — used when
/// `interpolation_filter == SWITCHABLE` (§7.3.7). ctx 0, two conditional
/// splits: first EIGHTTAP vs rest, then EIGHTTAP_SMOOTH vs EIGHTTAP_SHARP.
const DEFAULT_INTERP_PROBS: [u8; 2] = [235, 162];

/// Inter-frame intra-mode probs (§10.5 `default_if_y_mode_probs`,
/// ctx 0). Decoded with the same 9-prob tree as KF intra modes.
const DEFAULT_IF_Y_MODE_PROBS: [u8; 9] = [65, 32, 18, 144, 162, 194, 41, 51, 98];

/// Inter-frame chroma-intra-mode probs — used when `is_inter = false`.
/// Indexed by luma mode; we always use the 9-prob sub-row for luma=DC.
const DEFAULT_IF_UV_MODE_PROBS: [u8; 9] = [120, 7, 76, 176, 208, 126, 28, 221, 29];

/// Inter-only mode enumeration (§6.4.16 Table 9-31).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterMode {
    Nearestmv,
    Nearmv,
    Zeromv,
    Newmv,
}

/// Decode one inter-mode symbol against the 3-probability tree.
/// libvpx `vp9_inter_mode_tree`:
///   -ZEROMV, 2
///    -NEARESTMV, 4
///     -NEARMV, -NEWMV
/// (i.e. bit0 picks ZEROMV vs rest; bit1 picks NEARESTMV vs rest;
///  bit2 picks NEARMV vs NEWMV)
fn read_inter_mode(bd: &mut BoolDecoder<'_>, probs: [u8; 3]) -> Result<InterMode> {
    if bd.read(probs[0])? == 0 {
        return Ok(InterMode::Zeromv);
    }
    if bd.read(probs[1])? == 0 {
        return Ok(InterMode::Nearestmv);
    }
    if bd.read(probs[2])? == 0 {
        Ok(InterMode::Nearmv)
    } else {
        Ok(InterMode::Newmv)
    }
}

/// Decode the switchable-per-block interpolation-filter selection.
fn read_switchable_filter(bd: &mut BoolDecoder<'_>, probs: [u8; 2]) -> Result<InterpFilter> {
    // Tree shape in libvpx (`vp9_switchable_interp_tree`):
    //   -EIGHTTAP,        2
    //    -EIGHTTAP_SMOOTH, -EIGHTTAP_SHARP
    if bd.read(probs[0])? == 0 {
        return Ok(InterpFilter::EightTap);
    }
    if bd.read(probs[1])? == 0 {
        Ok(InterpFilter::EightTapSmooth)
    } else {
        Ok(InterpFilter::EightTapSharp)
    }
}

/// Wrap a `RefFrame` plane so `mc_block` can read it with edge clamp.
struct LumaSampler<'a>(&'a RefFrame);
impl RefSampler for LumaSampler<'_> {
    fn sample(&self, row: isize, col: isize) -> u8 {
        self.0.sample_y(row, col)
    }
}

struct ChromaSampler<'a> {
    frame: &'a RefFrame,
    plane: u8, // 1 or 2
}
impl RefSampler for ChromaSampler<'_> {
    fn sample(&self, row: isize, col: isize) -> u8 {
        self.frame.sample_uv(self.plane, row, col)
    }
}

/// Inter-frame tile decoder. Shares buffer layout with `IntraTile`.
pub struct InterTile<'a> {
    pub hdr: &'a UncompressedHeader,
    pub ch: &'a CompressedHeader,
    pub refs: [Option<&'a RefFrame>; 3],
    pub y: Vec<u8>,
    pub y_stride: usize,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub uv_stride: usize,
    pub uv_w: usize,
    pub uv_h: usize,
    pub width: usize,
    pub height: usize,
    /// Effective interpolation filter picked per header. When
    /// `interpolation_filter == SWITCHABLE` each block reads its own.
    pub default_filter: Option<InterpFilter>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Partition {
    None,
    Horz,
    Vert,
    Split,
}

impl<'a> InterTile<'a> {
    pub fn new(
        hdr: &'a UncompressedHeader,
        ch: &'a CompressedHeader,
        width: usize,
        height: usize,
        refs: [Option<&'a RefFrame>; 3],
    ) -> Self {
        let y_stride = width.max(1);
        let sub_x = hdr.color_config.subsampling_x as usize;
        let sub_y = hdr.color_config.subsampling_y as usize;
        let uv_w = (width + sub_x) >> sub_x;
        let uv_h = (height + sub_y) >> sub_y;
        let uv_stride = uv_w.max(1);
        let default_filter = if hdr.interpolation_filter < 4 {
            Some(InterpFilter::from_u8(hdr.interpolation_filter))
        } else {
            None
        };
        Self {
            hdr,
            ch,
            refs,
            y: vec![0u8; y_stride * height],
            y_stride,
            u: vec![0u8; uv_stride * uv_h],
            v: vec![0u8; uv_stride * uv_h],
            uv_stride,
            uv_w,
            uv_h,
            width,
            height,
            default_filter,
        }
    }

    /// Decode the tile's superblocks in raster order. The bool decoder
    /// is positioned at the first byte of the tile payload.
    pub fn decode(&mut self, bd: &mut BoolDecoder<'_>) -> Result<()> {
        let sbs_x = (self.width as u32).div_ceil(SUPERBLOCK_SIZE);
        let sbs_y = (self.height as u32).div_ceil(SUPERBLOCK_SIZE);
        for sby in 0..sbs_y {
            for sbx in 0..sbs_x {
                let col = sbx * SUPERBLOCK_SIZE;
                let row = sby * SUPERBLOCK_SIZE;
                self.decode_partition(bd, row, col, SUPERBLOCK_SIZE)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_partition(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bsize: u32,
    ) -> Result<()> {
        debug_assert!(matches!(bsize, 64 | 32 | 16 | 8));
        if row >= self.height as u32 || col >= self.width as u32 {
            return Ok(());
        }
        let partition = self.read_partition(bd, bsize)?;
        let half = bsize / 2;
        match partition {
            Partition::None => {
                let bs = BlockSize::from_wh(bsize, bsize);
                self.decode_block(bd, row, col, bs)?;
            }
            Partition::Horz => {
                let bs = BlockSize::from_wh(bsize, half);
                self.decode_block(bd, row, col, bs)?;
                if row + half < self.height as u32 {
                    self.decode_block(bd, row + half, col, bs)?;
                }
            }
            Partition::Vert => {
                let bs = BlockSize::from_wh(half, bsize);
                self.decode_block(bd, row, col, bs)?;
                if col + half < self.width as u32 {
                    self.decode_block(bd, row, col + half, bs)?;
                }
            }
            Partition::Split => {
                if bsize == 8 {
                    for (dr, dc) in [(0, 0), (0, 4), (4, 0), (4, 4)] {
                        let r = row + dr;
                        let c = col + dc;
                        if r < self.height as u32 && c < self.width as u32 {
                            self.decode_block(bd, r, c, BlockSize::B4x4)?;
                        }
                    }
                } else {
                    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                        self.decode_partition(bd, row + dr, col + dc, half)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn read_partition(&self, bd: &mut BoolDecoder<'_>, bsize: u32) -> Result<Partition> {
        let bsl = match bsize {
            64 => 0usize,
            32 => 1,
            16 => 2,
            8 => 3,
            _ => 0,
        };
        let probs = PARTITION_PROBS[bsl * 4];
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

    fn decode_block(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
    ) -> Result<()> {
        // §6.4.3 `decode_block`:
        //   1. read_is_inter (prob from ctx; default fallback).
        //   2. if is_inter: read_inter_frame_mode_info else read_intra_frame_mode_info.
        //   3. read_skip (§6.4.8).
        //   4. read tx_size (§6.4.10).
        //   5. read residual tokens per plane.
        //   6. reconstruct.
        let skip = bd.read(DEFAULT_SKIP_PROB)? != 0;
        let is_inter = bd.read(DEFAULT_IS_INTER_PROB)? != 0;
        let tx_size_log2 = self.read_tx_size(bd, bs)?;
        if is_inter {
            self.decode_inter_block(bd, row, col, bs, tx_size_log2, skip)
        } else {
            self.decode_intra_block(bd, row, col, bs, tx_size_log2, skip)
        }
    }

    fn decode_inter_block(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
        tx_size_log2: usize,
        skip: bool,
    ) -> Result<()> {
        // Single-reference only. Read the 1-bit reference selector
        // (slot 0 vs rest — §6.4.15). For our purposes (default probs
        // at ctx 0) we just read two bits to pick LAST / GOLDEN / ALTREF.
        let first = bd.read(DEFAULT_SINGLE_REF_PROB[0])?;
        let ref_slot = if first == 0 {
            0 // LAST
        } else {
            let second = bd.read(DEFAULT_SINGLE_REF_PROB[1])?;
            if second == 0 {
                1 // GOLDEN
            } else {
                2 // ALTREF
            }
        };
        let inter_mode = read_inter_mode(bd, DEFAULT_INTER_MODE_PROBS)?;

        // Interpolation filter (optionally switchable).
        let filter = if let Some(f) = self.default_filter {
            f
        } else {
            read_switchable_filter(bd, DEFAULT_INTERP_PROBS)?
        };

        // Motion vector. For ZEROMV the spec forces (0, 0) without any
        // reads. NEARESTMV / NEARMV look up neighbour predictors (we
        // currently don't build the candidate list → they degrade to
        // (0, 0)). NEWMV reads a full §6.4.19 MV delta on top of the
        // (0, 0) predictor.
        let mv = if matches!(inter_mode, InterMode::Newmv) {
            let joint = read_mv_joint(bd, MV_JOINT_PROBS)?;
            let mv_row = if joint.has_row() {
                read_mv_component(bd, &DEFAULT_MV_COMP_PROBS, self.hdr.allow_high_precision_mv)?
            } else {
                0
            };
            let mv_col = if joint.has_col() {
                read_mv_component(bd, &DEFAULT_MV_COMP_PROBS, self.hdr.allow_high_precision_mv)?
            } else {
                0
            };
            (mv_row, mv_col)
        } else {
            (0i16, 0i16)
        };

        // Apply motion compensation if the selected reference exists —
        // otherwise fall back to a flat mid-grey block (degraded but
        // avoids aborting the whole frame).
        let rf = self.refs.get(ref_slot).copied().flatten();
        if let Some(rf) = rf {
            self.mc_luma_block(rf, row as usize, col as usize, bs, mv.0, mv.1, filter);
            let sub_x = self.hdr.color_config.subsampling_x as i32;
            let sub_y = self.hdr.color_config.subsampling_y as i32;
            let c_row = (row as usize) >> sub_y;
            let c_col = (col as usize) >> sub_x;
            let c_w = bs.w() as usize >> sub_x;
            let c_h = bs.h() as usize >> sub_y;
            if c_w >= 4 && c_h >= 4 {
                self.mc_chroma_block(rf, c_row, c_col, c_w, c_h, mv.0, mv.1, filter, 1);
                self.mc_chroma_block(rf, c_row, c_col, c_w, c_h, mv.0, mv.1, filter, 2);
            }
        } else {
            // No reference — fill with 128. Surfaces as grey but decode
            // stays sound.
            self.fill_block(
                row as usize,
                col as usize,
                bs.w() as usize,
                bs.h() as usize,
                0,
                128,
            );
            let sub_x = self.hdr.color_config.subsampling_x as i32;
            let sub_y = self.hdr.color_config.subsampling_y as i32;
            let c_row = (row as usize) >> sub_y;
            let c_col = (col as usize) >> sub_x;
            let c_w = bs.w() as usize >> sub_x;
            let c_h = bs.h() as usize >> sub_y;
            if c_w >= 4 && c_h >= 4 {
                self.fill_block(c_row, c_col, c_w, c_h, 1, 128);
                self.fill_block(c_row, c_col, c_w, c_h, 2, 128);
            }
        }

        if !skip {
            // Residual: tx-blocks over the prediction unit with
            // tx_type = DCT_DCT for inter blocks (§7.4.3).
            self.add_residual(bd, row, col, bs, tx_size_log2, TxType::DctDct)?;
        }
        Ok(())
    }

    fn decode_intra_block(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
        tx_size_log2: usize,
        skip: bool,
    ) -> Result<()> {
        let y_mode = read_intra_mode_tree(bd, &DEFAULT_IF_Y_MODE_PROBS)?;
        let uv_mode = read_intra_mode_tree(bd, &DEFAULT_IF_UV_MODE_PROBS)?;
        // Luma predict + residual.
        self.recon_intra_plane(
            bd,
            row as usize,
            col as usize,
            bs.w() as usize,
            bs.h() as usize,
            tx_size_log2,
            y_mode,
            0,
            skip,
        )?;
        let sub_x = self.hdr.color_config.subsampling_x as u32;
        let sub_y = self.hdr.color_config.subsampling_y as u32;
        let c_row = (row >> sub_y) as usize;
        let c_col = (col >> sub_x) as usize;
        let c_w = (bs.w() >> sub_x) as usize;
        let c_h = (bs.h() >> sub_y) as usize;
        let c_tx = clamp_tx_size(tx_size_log2, c_w, c_h);
        if c_w >= 4 && c_h >= 4 {
            self.recon_intra_plane(bd, c_row, c_col, c_w, c_h, c_tx, uv_mode, 1, skip)?;
            self.recon_intra_plane(bd, c_row, c_col, c_w, c_h, c_tx, uv_mode, 2, skip)?;
        }
        Ok(())
    }

    fn read_tx_size(&self, bd: &mut BoolDecoder<'_>, bs: BlockSize) -> Result<usize> {
        let max_tx = bs.max_tx_size_log2();
        let tx_mode = self.ch.tx_mode.unwrap_or(TxMode::Only4x4);
        match tx_mode {
            TxMode::Only4x4 => Ok(0),
            TxMode::Allow8x8 => Ok(max_tx.min(1)),
            TxMode::Allow16x16 => Ok(max_tx.min(2)),
            TxMode::Allow32x32 => Ok(max_tx.min(3)),
            TxMode::Select => {
                let tx_probs = tx_probs_for(max_tx);
                let mut tx = bd.read(tx_probs[0])? as usize;
                if tx != 0 && max_tx >= 2 {
                    tx += bd.read(tx_probs[1])? as usize;
                    if tx != 1 && max_tx >= 3 {
                        tx += bd.read(tx_probs[2])? as usize;
                    }
                }
                Ok(tx.min(max_tx))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mc_luma_block(
        &mut self,
        rf: &RefFrame,
        row: usize,
        col: usize,
        bs: BlockSize,
        mv_row: i16,
        mv_col: i16,
        filter: InterpFilter,
    ) {
        let w = bs.w() as usize;
        let h = bs.h() as usize;
        let eff_w = w.min(self.width.saturating_sub(col));
        let eff_h = h.min(self.height.saturating_sub(row));
        if eff_w == 0 || eff_h == 0 {
            return;
        }
        // MV in 1/8-pel; integer = mv >> 3, sub-phase = (mv & 7) * 2
        // (mapping to 0..16 sub-pel table). libvpx encodes sub_row with
        // `SUBPEL_BITS = 4` on the interpolator side.
        let int_row = row as isize + (mv_row as isize >> 3);
        let int_col = col as isize + (mv_col as isize >> 3);
        let sub_row = ((mv_row & 7) as u32) << 1;
        let sub_col = ((mv_col & 7) as u32) << 1;
        let mut dst = vec![0u8; eff_w * eff_h];
        let sampler = LumaSampler(rf);
        mc_block(
            &sampler, filter, &mut dst, eff_w, eff_w, eff_h, int_row, int_col, sub_row, sub_col,
        );
        self.blit_plane(0, row, col, eff_w, eff_h, &dst, eff_w);
    }

    #[allow(clippy::too_many_arguments)]
    fn mc_chroma_block(
        &mut self,
        rf: &RefFrame,
        row: usize,
        col: usize,
        w: usize,
        h: usize,
        mv_row_luma: i16,
        mv_col_luma: i16,
        filter: InterpFilter,
        plane: u8,
    ) {
        let eff_w = w.min(self.uv_w.saturating_sub(col));
        let eff_h = h.min(self.uv_h.saturating_sub(row));
        if eff_w == 0 || eff_h == 0 {
            return;
        }
        // Convert luma 1/8-pel MV → chroma 1/16-pel for 4:2:0 (spec
        // §8.5.1): chroma_mv = (luma_mv * 2) >> subsampling.
        let sub_x = self.hdr.color_config.subsampling_x as u32;
        let sub_y = self.hdr.color_config.subsampling_y as u32;
        // For subsampling=1 this is effectively mv (already 1/8-pel luma
        // → 1/16-pel chroma because the chroma grid is half the luma
        // resolution). For subsampling=0 (4:4:4) we'd scale by 2.
        let mv_row = mv_row_luma as i32;
        let mv_col = mv_col_luma as i32;
        let int_row = row as isize + ((mv_row >> (3 + sub_y as i32)) as isize);
        let int_col = col as isize + ((mv_col >> (3 + sub_x as i32)) as isize);
        let sub_row_phase =
            ((mv_row & ((1 << (3 + sub_y as i32)) - 1)) as u32) << (1 - sub_y.min(1));
        let sub_col_phase =
            ((mv_col & ((1 << (3 + sub_x as i32)) - 1)) as u32) << (1 - sub_x.min(1));
        let sub_row = sub_row_phase & 15;
        let sub_col = sub_col_phase & 15;
        let mut dst = vec![0u8; eff_w * eff_h];
        let sampler = ChromaSampler { frame: rf, plane };
        mc_block(
            &sampler, filter, &mut dst, eff_w, eff_w, eff_h, int_row, int_col, sub_row, sub_col,
        );
        self.blit_plane(plane as usize, row, col, eff_w, eff_h, &dst, eff_w);
    }

    fn fill_block(&mut self, row: usize, col: usize, w: usize, h: usize, plane: usize, v: u8) {
        let (buf, stride, plane_w, plane_h) = match plane {
            0 => (&mut self.y, self.y_stride, self.width, self.height),
            1 => (&mut self.u, self.uv_stride, self.uv_w, self.uv_h),
            _ => (&mut self.v, self.uv_stride, self.uv_w, self.uv_h),
        };
        let eff_w = w.min(plane_w.saturating_sub(col));
        let eff_h = h.min(plane_h.saturating_sub(row));
        for r in 0..eff_h {
            let base = (row + r) * stride + col;
            for c in 0..eff_w {
                buf[base + c] = v;
            }
        }
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

    #[allow(clippy::too_many_arguments)]
    fn add_residual(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
        tx_size_log2: usize,
        tx_type: TxType,
    ) -> Result<()> {
        let w = bs.w() as usize;
        let h = bs.h() as usize;
        // Luma.
        self.decode_plane_residual(
            bd,
            row as usize,
            col as usize,
            w,
            h,
            tx_size_log2,
            tx_type,
            0,
        )?;
        let sub_x = self.hdr.color_config.subsampling_x as u32;
        let sub_y = self.hdr.color_config.subsampling_y as u32;
        let c_row = (row >> sub_y) as usize;
        let c_col = (col >> sub_x) as usize;
        let c_w = (bs.w() >> sub_x) as usize;
        let c_h = (bs.h() >> sub_y) as usize;
        let c_tx = clamp_tx_size(tx_size_log2, c_w, c_h);
        if c_w >= 4 && c_h >= 4 {
            self.decode_plane_residual(bd, c_row, c_col, c_w, c_h, c_tx, TxType::DctDct, 1)?;
            self.decode_plane_residual(bd, c_row, c_col, c_w, c_h, c_tx, TxType::DctDct, 2)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_plane_residual(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: usize,
        col: usize,
        w: usize,
        h: usize,
        tx_size_log2: usize,
        tx_type: TxType,
        plane: usize,
    ) -> Result<()> {
        let tx_side = 4usize << tx_size_log2;
        let plane_type = if plane == 0 { 0 } else { 1 };
        let (plane_w, plane_h) = if plane == 0 {
            (self.width, self.height)
        } else {
            (self.uv_w, self.uv_h)
        };
        let scan = get_scan(tx_size_log2);
        let probs = coef_probs_for(tx_size_log2, plane_type);

        let qp = self.hdr.quantization.base_q_idx as usize;
        let (dc, ac) = if plane == 0 {
            (
                DC_QLOOKUP[clamp_q(qp as i32 + self.hdr.quantization.delta_q_y_dc as i32)],
                AC_QLOOKUP[clamp_q(qp as i32)],
            )
        } else {
            (
                DC_QLOOKUP[clamp_q(qp as i32 + self.hdr.quantization.delta_q_uv_dc as i32)],
                AC_QLOOKUP[clamp_q(qp as i32 + self.hdr.quantization.delta_q_uv_ac as i32)],
            )
        };
        let dq = [dc, ac];

        let mut r = 0usize;
        while r < h {
            let mut c = 0usize;
            while c < w {
                let abs_row = row + r;
                let abs_col = col + c;
                if abs_row >= plane_h || abs_col >= plane_w {
                    c += tx_side;
                    continue;
                }
                let tx_w = tx_side.min(plane_w - abs_col);
                let tx_h = tx_side.min(plane_h - abs_row);
                let mut coeffs = vec![0i32; tx_side * tx_side];
                let eob = decode_coefs(
                    bd,
                    probs,
                    &dq,
                    scan.scan,
                    scan.neighbors,
                    scan.band_translate,
                    scan.tx_size_log2,
                    0,
                    &mut coeffs,
                )?;
                if eob > 0 {
                    let mut dst = vec![0u8; tx_side * tx_side];
                    self.read_plane(plane, abs_row, abs_col, tx_w, tx_h, &mut dst, tx_side);
                    inverse_transform_add(tx_type, tx_side, tx_side, &coeffs, &mut dst, tx_side)?;
                    self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &dst, tx_side);
                }
                c += tx_side;
            }
            r += tx_side;
        }
        Ok(())
    }

    /// Mirror of `IntraTile::reconstruct_plane` for the intra-in-inter
    /// path. Keep them parallel so changes stay in sync.
    #[allow(clippy::too_many_arguments)]
    fn recon_intra_plane(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: usize,
        col: usize,
        w: usize,
        h: usize,
        tx_size_log2: usize,
        mode: IntraMode,
        plane: usize,
        skip: bool,
    ) -> Result<()> {
        let tx_side = 4usize << tx_size_log2;
        let plane_type = if plane == 0 { 0 } else { 1 };
        let (plane_w, plane_h) = if plane == 0 {
            (self.width, self.height)
        } else {
            (self.uv_w, self.uv_h)
        };
        let scan = get_scan(tx_size_log2);
        let probs = coef_probs_for(tx_size_log2, plane_type);

        let qp = self.hdr.quantization.base_q_idx as usize;
        let (dc, ac) = if plane == 0 {
            (
                DC_QLOOKUP[clamp_q(qp as i32 + self.hdr.quantization.delta_q_y_dc as i32)],
                AC_QLOOKUP[clamp_q(qp as i32)],
            )
        } else {
            (
                DC_QLOOKUP[clamp_q(qp as i32 + self.hdr.quantization.delta_q_uv_dc as i32)],
                AC_QLOOKUP[clamp_q(qp as i32 + self.hdr.quantization.delta_q_uv_ac as i32)],
            )
        };
        let dq = [dc, ac];

        let mut r = 0usize;
        while r < h {
            let mut c = 0usize;
            while c < w {
                let abs_row = row + r;
                let abs_col = col + c;
                if abs_row >= plane_h || abs_col >= plane_w {
                    c += tx_side;
                    continue;
                }
                let tx_w = tx_side.min(plane_w - abs_col);
                let tx_h = tx_side.min(plane_h - abs_row);
                let nb = self.build_neighbours(plane, abs_row, abs_col, tx_side);
                let mut pred = vec![0u8; tx_side * tx_side];
                predict_intra(mode, &nb, &mut pred, tx_side);
                self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &pred, tx_side);
                if !skip {
                    let mut coeffs = vec![0i32; tx_side * tx_side];
                    let eob = decode_coefs(
                        bd,
                        probs,
                        &dq,
                        scan.scan,
                        scan.neighbors,
                        scan.band_translate,
                        scan.tx_size_log2,
                        0,
                        &mut coeffs,
                    )?;
                    if eob > 0 {
                        let mut dst = vec![0u8; tx_side * tx_side];
                        self.read_plane(plane, abs_row, abs_col, tx_w, tx_h, &mut dst, tx_side);
                        inverse_transform_add(
                            TxType::DctDct,
                            tx_side,
                            tx_side,
                            &coeffs,
                            &mut dst,
                            tx_side,
                        )?;
                        self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &dst, tx_side);
                    }
                }
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
}

/// Scan-order lookup (simplified — inter blocks always use
/// `DEFAULT_SCAN` because `tx_type` is always `DCT_DCT`).
struct ScanOrder {
    scan: &'static [i16],
    neighbors: &'static [i16],
    band_translate: &'static [u8],
    tx_size_log2: usize,
}

fn get_scan(tx_size_log2: usize) -> ScanOrder {
    match tx_size_log2 {
        0 => ScanOrder {
            scan: &DEFAULT_SCAN_4X4,
            neighbors: &DEFAULT_SCAN_4X4_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_4X4,
            tx_size_log2: 0,
        },
        1 => ScanOrder {
            scan: &DEFAULT_SCAN_8X8,
            neighbors: &DEFAULT_SCAN_8X8_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 1,
        },
        2 => ScanOrder {
            scan: &DEFAULT_SCAN_16X16,
            neighbors: &DEFAULT_SCAN_16X16_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 2,
        },
        _ => ScanOrder {
            scan: &DEFAULT_SCAN_32X32,
            neighbors: &DEFAULT_SCAN_32X32_NEIGHBORS,
            band_translate: &COEFBAND_TRANS_8X8PLUS,
            tx_size_log2: 3,
        },
    }
}

fn coef_probs_for(tx_size_log2: usize, plane_type: usize) -> &'static [[[u8; 3]; 6]; 6] {
    match tx_size_log2 {
        0 => &COEF_PROBS_4X4[plane_type][0],
        1 => &COEF_PROBS_8X8[plane_type][0],
        2 => &COEF_PROBS_16X16[plane_type][0],
        3 => &COEF_PROBS_32X32[plane_type][0],
        _ => &COEF_PROBS_4X4[0][0],
    }
}

fn read_intra_mode_tree(bd: &mut BoolDecoder<'_>, p: &[u8; 9]) -> Result<IntraMode> {
    if bd.read(p[0])? == 0 {
        return Ok(IntraMode::Dc);
    }
    if bd.read(p[1])? == 0 {
        return Ok(IntraMode::Tm);
    }
    if bd.read(p[2])? == 0 {
        return Ok(IntraMode::V);
    }
    if bd.read(p[3])? == 0 {
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

fn tx_probs_for(max_tx: usize) -> [u8; 3] {
    match max_tx {
        3 => [3, 136, 37],
        2 => [20, 152, 0],
        1 => [100, 0, 0],
        _ => [128, 128, 128],
    }
}

fn clamp_q(q: i32) -> usize {
    q.clamp(0, 255) as usize
}

fn clamp_tx_size(log2: usize, w: usize, h: usize) -> usize {
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

/// Remove unused warning avoidance — `Error` used via `Result`.
#[allow(dead_code)]
fn _touch_error() -> Error {
    Error::unsupported("vp9 inter: sentinel")
}

#[cfg(not(any()))]
// Give the type a public re-export helper so downstream code can inspect.
impl InterMode {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nearestmv => "NEARESTMV",
            Self::Nearmv => "NEARMV",
            Self::Zeromv => "ZEROMV",
            Self::Newmv => "NEWMV",
        }
    }
}
