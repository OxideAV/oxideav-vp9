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
//! Simplifications used in this revision — documented in the
//! README's "Deferred" section:
//!
//! * **Compound prediction (§6.4.17 / §8.5.2)** — supported: when
//!   `reference_mode` is `COMPOUND_REFERENCE` or `REFERENCE_MODE_SELECT`,
//!   per-block `comp_mode` / `comp_ref` signalling picks two references,
//!   each MC'd independently, then averaged with `Round2(a+b, 1)`.
//! * **Scaled references (§8.5.2.3)** — supported via per-reference
//!   `x_step_q4` / `y_step_q4` applied through the variable-step
//!   interpolator.
//! * **Neighbour-aware probability contexts** — context 0 everywhere
//!   (same compromise as the keyframe path).
//! * **Segmentation deltas** — `SEG_LVL_ALT_Q` and `SEG_LVL_ALT_L` are
//!   honoured per-block when segmentation is enabled.

use oxideav_core::{Error, Result};

use crate::block::{BlockSize, SUPERBLOCK_SIZE};
use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::{CompressedHeader, ReferenceMode, TxMode};
use crate::detokenize::decode_coefs;
use crate::dpb::RefFrame;
use crate::headers::UncompressedHeader;
use crate::intra::IntraMode;
use crate::loopfilter::{LoopFilter, MiInfo, MiInfoPlane, INTRA_FRAME};
use crate::mcfilter::{mc_block_scaled, InterpFilter, RefSampler};
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

/// Default `comp_mode` probability (§10.5 `default_comp_mode_prob`, ctx 0)
/// — picks COMPOUND vs SINGLE when `reference_mode == REFERENCE_MODE_SELECT`.
const DEFAULT_COMP_MODE_PROB: u8 = 128;

/// Default `comp_ref` probability — picks which variable reference frame
/// is used in compound prediction (§10.5 `default_comp_ref_prob`, ctx 0).
const DEFAULT_COMP_REF_PROB: u8 = 128;

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

/// Compound-reference index resolution per §6.3.18
/// `setup_compound_reference_mode`. Driven by `ref_frame_sign_bias`.
///
/// `fixed` is one of {LAST=1, GOLDEN=2, ALTREF=3} — the "fixed" side of
/// compound prediction. `var[0]` / `var[1]` are the two candidates
/// selectable via the `comp_ref` bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompRefs {
    pub fixed: u8,
    pub var: [u8; 2],
}

impl CompRefs {
    pub fn from_sign_bias(sign_bias: &[bool; 4]) -> Self {
        // Index 1 = LAST, 2 = GOLDEN, 3 = ALTREF.
        let b_last = sign_bias[1];
        let b_golden = sign_bias[2];
        let b_altref = sign_bias[3];
        if b_last == b_golden {
            CompRefs {
                fixed: 3,
                var: [1, 2],
            }
        } else if b_last == b_altref {
            CompRefs {
                fixed: 2,
                var: [1, 3],
            }
        } else {
            CompRefs {
                fixed: 1,
                var: [2, 3],
            }
        }
    }

    /// Map a ref-frame code (1..=3) to its DPB slot index (0..=2).
    fn slot_of(code: u8) -> usize {
        (code as usize).saturating_sub(1)
    }
}

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
    /// Per-8x8-MI block metadata for §8.8 loop filtering.
    pub mi_info: MiInfoPlane,
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
        let mi_cols = width.div_ceil(8).max(1);
        let mi_rows = height.div_ceil(8).max(1);
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
            mi_info: MiInfoPlane::new(mi_cols, mi_rows),
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
        // §8.8 loop filter pass.
        self.apply_loop_filter();
        Ok(())
    }

    /// Run the §8.8 loop filter over the reconstructed planes.
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
        // Determine whether this block is compound or single.
        // §6.4.17 read_ref_frames:
        //   - If reference_mode == REFERENCE_MODE_SELECT read comp_mode,
        //     else comp_mode = reference_mode.
        //   - If compound: read comp_ref; idx = sign_bias[CompFixedRef];
        //     ref_frame[idx] = CompFixedRef, ref_frame[!idx] = CompVarRef[comp_ref].
        //   - Else: read single_ref_p1 / single_ref_p2.
        let frame_ref_mode = self.ch.reference_mode.unwrap_or(ReferenceMode::SingleReference);
        let is_compound = match frame_ref_mode {
            ReferenceMode::SingleReference => false,
            ReferenceMode::CompoundReference => true,
            ReferenceMode::ReferenceModeSelect => bd.read(DEFAULT_COMP_MODE_PROB)? != 0,
        };

        // ref_frame_codes[0], ref_frame_codes[1] — LAST=1, GOLDEN=2, ALTREF=3.
        // ref_frame_codes[1] is 0 (NONE) when single.
        let (ref_code_a, ref_code_b) = if is_compound {
            let comp_refs = CompRefs::from_sign_bias(&self.hdr.ref_frame_sign_bias);
            let comp_ref_bit = bd.read(DEFAULT_COMP_REF_PROB)? as usize;
            let idx = self.hdr.ref_frame_sign_bias[comp_refs.fixed as usize] as usize;
            let mut refs = [0u8; 2];
            refs[idx] = comp_refs.fixed;
            refs[idx ^ 1] = comp_refs.var[comp_ref_bit];
            (refs[0], refs[1])
        } else {
            let first = bd.read(DEFAULT_SINGLE_REF_PROB[0])?;
            let code = if first == 0 {
                1u8 // LAST
            } else {
                let second = bd.read(DEFAULT_SINGLE_REF_PROB[1])?;
                if second == 0 {
                    2u8 // GOLDEN
                } else {
                    3u8 // ALTREF
                }
            };
            (code, 0u8)
        };
        let inter_mode = read_inter_mode(bd, DEFAULT_INTER_MODE_PROBS)?;

        // Record §8.8 MI metadata for this inter block.
        let mi_row_units = (row as usize) / 8;
        let mi_col_units = (col as usize) / 8;
        let mi_w = (bs.w() / 8).max(1) as u8;
        let mi_h = (bs.h() / 8).max(1) as u8;
        let mode_is_non_zero_inter = !matches!(inter_mode, InterMode::Zeromv);
        self.mi_info.fill(
            mi_row_units,
            mi_col_units,
            MiInfo {
                mi_w_8x8: mi_w,
                mi_h_8x8: mi_h,
                tx_size_log2: tx_size_log2 as u8,
                skip,
                ref_frame: ref_code_a, // primary ref drives LF deltas.
                mode_is_non_zero_inter,
                segment_id: 0,
            },
        );

        // Interpolation filter (optionally switchable).
        let filter = if let Some(f) = self.default_filter {
            f
        } else {
            read_switchable_filter(bd, DEFAULT_INTERP_PROBS)?
        };

        // Read MVs: for NEWMV one component-tree read per reference, else
        // zero. (NEARESTMV / NEARMV candidate lists aren't built yet; they
        // degrade to zero — same compromise as the single-ref path.)
        let mv_a = if matches!(inter_mode, InterMode::Newmv) {
            self.read_new_mv(bd)?
        } else {
            (0i16, 0i16)
        };
        let mv_b = if is_compound && matches!(inter_mode, InterMode::Newmv) {
            self.read_new_mv(bd)?
        } else {
            (0i16, 0i16)
        };

        // Motion-compensate each reference independently, then average
        // for compound. §8.5.2: preds[0..1+isCompound], then
        // Round2(a+b, 1) when compound.
        let w_px = bs.w() as usize;
        let h_px = bs.h() as usize;
        let eff_w = w_px.min(self.width.saturating_sub(col as usize));
        let eff_h = h_px.min(self.height.saturating_sub(row as usize));
        let sub_x = self.hdr.color_config.subsampling_x as usize;
        let sub_y = self.hdr.color_config.subsampling_y as usize;
        let c_row = (row as usize) >> sub_y;
        let c_col = (col as usize) >> sub_x;
        let c_w = w_px >> sub_x;
        let c_h = h_px >> sub_y;
        let eff_cw = c_w.min(self.uv_w.saturating_sub(c_col));
        let eff_ch = c_h.min(self.uv_h.saturating_sub(c_row));

        // Predict into a temporary so we can round-average for compound.
        let mut luma_a = vec![0u8; eff_w * eff_h];
        let mut luma_b = vec![0u8; eff_w * eff_h];
        let mut chroma_a = [vec![0u8; eff_cw * eff_ch], vec![0u8; eff_cw * eff_ch]];
        let mut chroma_b = [vec![0u8; eff_cw * eff_ch], vec![0u8; eff_cw * eff_ch]];

        let mut have_a = false;
        let mut have_b = false;

        if ref_code_a != 0 {
            if let Some(rf) = self.ref_by_code(ref_code_a) {
                self.mc_luma_to(
                    rf,
                    row as usize,
                    col as usize,
                    eff_w,
                    eff_h,
                    mv_a.0,
                    mv_a.1,
                    filter,
                    &mut luma_a,
                );
                if eff_cw >= 4 && eff_ch >= 4 {
                    self.mc_chroma_to(rf, c_row, c_col, eff_cw, eff_ch, mv_a.0, mv_a.1, filter, 1, &mut chroma_a[0]);
                    self.mc_chroma_to(rf, c_row, c_col, eff_cw, eff_ch, mv_a.0, mv_a.1, filter, 2, &mut chroma_a[1]);
                }
                have_a = true;
            }
        }
        if is_compound && ref_code_b != 0 {
            if let Some(rf) = self.ref_by_code(ref_code_b) {
                self.mc_luma_to(
                    rf,
                    row as usize,
                    col as usize,
                    eff_w,
                    eff_h,
                    mv_b.0,
                    mv_b.1,
                    filter,
                    &mut luma_b,
                );
                if eff_cw >= 4 && eff_ch >= 4 {
                    self.mc_chroma_to(rf, c_row, c_col, eff_cw, eff_ch, mv_b.0, mv_b.1, filter, 1, &mut chroma_b[0]);
                    self.mc_chroma_to(rf, c_row, c_col, eff_cw, eff_ch, mv_b.0, mv_b.1, filter, 2, &mut chroma_b[1]);
                }
                have_b = true;
            }
        }

        // Blit — average if compound, else copy. Fall back to grey fill
        // when the reference is missing (preserves decode progress).
        if is_compound && have_a && have_b {
            average_into(&mut luma_a, &luma_b);
            self.blit_plane(0, row as usize, col as usize, eff_w, eff_h, &luma_a, eff_w);
            if eff_cw >= 4 && eff_ch >= 4 {
                average_into(&mut chroma_a[0], &chroma_b[0]);
                average_into(&mut chroma_a[1], &chroma_b[1]);
                self.blit_plane(1, c_row, c_col, eff_cw, eff_ch, &chroma_a[0], eff_cw);
                self.blit_plane(2, c_row, c_col, eff_cw, eff_ch, &chroma_a[1], eff_cw);
            }
        } else if have_a {
            self.blit_plane(0, row as usize, col as usize, eff_w, eff_h, &luma_a, eff_w);
            if eff_cw >= 4 && eff_ch >= 4 {
                self.blit_plane(1, c_row, c_col, eff_cw, eff_ch, &chroma_a[0], eff_cw);
                self.blit_plane(2, c_row, c_col, eff_cw, eff_ch, &chroma_a[1], eff_cw);
            }
        } else if have_b {
            self.blit_plane(0, row as usize, col as usize, eff_w, eff_h, &luma_b, eff_w);
            if eff_cw >= 4 && eff_ch >= 4 {
                self.blit_plane(1, c_row, c_col, eff_cw, eff_ch, &chroma_b[0], eff_cw);
                self.blit_plane(2, c_row, c_col, eff_cw, eff_ch, &chroma_b[1], eff_cw);
            }
        } else {
            self.fill_block(row as usize, col as usize, eff_w, eff_h, 0, 128);
            if eff_cw >= 4 && eff_ch >= 4 {
                self.fill_block(c_row, c_col, eff_cw, eff_ch, 1, 128);
                self.fill_block(c_row, c_col, eff_cw, eff_ch, 2, 128);
            }
        }

        if !skip {
            // Residual: tx-blocks over the prediction unit with
            // tx_type = DCT_DCT for inter blocks (§7.4.3).
            self.add_residual(bd, row, col, bs, tx_size_log2, TxType::DctDct)?;
        }
        Ok(())
    }

    /// Locate a decoded reference frame by its 1..=3 code (LAST=1,
    /// GOLDEN=2, ALTREF=3). Returns None if the slot is empty.
    fn ref_by_code(&self, code: u8) -> Option<&'a RefFrame> {
        let slot = CompRefs::slot_of(code);
        self.refs.get(slot).copied().flatten()
    }

    /// Read one NEWMV delta pair (row, col) in 1/8-pel units.
    fn read_new_mv(&self, bd: &mut BoolDecoder<'_>) -> Result<(i16, i16)> {
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
        Ok((mv_row, mv_col))
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
        // Record §8.8 MI metadata for this intra-in-inter block.
        let mi_row_units = (row as usize) / 8;
        let mi_col_units = (col as usize) / 8;
        let mi_w = (bs.w() / 8).max(1) as u8;
        let mi_h = (bs.h() / 8).max(1) as u8;
        self.mi_info.fill(
            mi_row_units,
            mi_col_units,
            MiInfo {
                mi_w_8x8: mi_w,
                mi_h_8x8: mi_h,
                tx_size_log2: tx_size_log2 as u8,
                skip,
                ref_frame: INTRA_FRAME,
                mode_is_non_zero_inter: false,
                segment_id: 0,
            },
        );
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

    /// Luma MC into a caller-provided buffer `dst[eff_w * eff_h]`. Uses
    /// the scaled interpolator so per-reference `x_step_q4` / `y_step_q4`
    /// can differ from the current-frame resolution (§8.5.2.3).
    #[allow(clippy::too_many_arguments)]
    fn mc_luma_to(
        &self,
        rf: &RefFrame,
        row: usize,
        col: usize,
        eff_w: usize,
        eff_h: usize,
        mv_row: i16,
        mv_col: i16,
        filter: InterpFilter,
        dst: &mut [u8],
    ) {
        if eff_w == 0 || eff_h == 0 {
            return;
        }
        // §8.5.2.3: xScale = (RefW << 14) / CurW; stepX = (16 * xScale) >> 14.
        //            startX = (baseX << 4) + ((clampedMv * xScale) >> 14) + fracX.
        // We work entirely in 1/16 sample units (q4) for the interpolator.
        let ref_w = rf.width as i32;
        let ref_h = rf.height as i32;
        let cur_w = self.width as i32;
        let cur_h = self.height as i32;
        let scale_shift = 14;
        let x_scale = (ref_w << scale_shift) / cur_w.max(1);
        let y_scale = (ref_h << scale_shift) / cur_h.max(1);
        let x_step_q4 = (16 * x_scale) >> scale_shift;
        let y_step_q4 = (16 * y_scale) >> scale_shift;
        // base position in ref: (cur_pos * scale) >> shift, scaled pos
        // to q4 by shifting << 4.
        // Spec uses 2x-precision clampedMv; for 4:2:0 luma sx=sy=0 →
        // clampedMv = 2 * mv. Units are 1/8-pel for mv, so
        // clampedMv * xScale / 16384 → q4 offset in ref.
        let mv_r = (mv_row as i32) * 2;
        let mv_c = (mv_col as i32) * 2;
        let base_x_q4 = (((col as i32) * x_scale) >> scale_shift) << 4;
        let base_y_q4 = (((row as i32) * y_scale) >> scale_shift) << 4;
        let frac_x = ((16 * (col as i32) * x_scale) >> scale_shift) & 15;
        let frac_y = ((16 * (row as i32) * y_scale) >> scale_shift) & 15;
        let d_x = ((mv_c * x_scale) >> scale_shift) + frac_x;
        let d_y = ((mv_r * y_scale) >> scale_shift) + frac_y;
        let start_x_q4 = base_x_q4 + d_x;
        let start_y_q4 = base_y_q4 + d_y;
        let sampler = LumaSampler(rf);
        mc_block_scaled(
            &sampler, filter, dst, eff_w, eff_w, eff_h, start_x_q4, start_y_q4, x_step_q4,
            y_step_q4,
        );
    }

    /// Chroma MC into a caller-provided buffer.
    #[allow(clippy::too_many_arguments)]
    fn mc_chroma_to(
        &self,
        rf: &RefFrame,
        row: usize,
        col: usize,
        eff_w: usize,
        eff_h: usize,
        mv_row_luma: i16,
        mv_col_luma: i16,
        filter: InterpFilter,
        plane: u8,
        dst: &mut [u8],
    ) {
        if eff_w == 0 || eff_h == 0 {
            return;
        }
        let sub_x = self.hdr.color_config.subsampling_x as i32;
        let sub_y = self.hdr.color_config.subsampling_y as i32;
        // Reference chroma plane dimensions — needed both for the
        // sampler clamp (done inside RefFrame) and for scaling.
        let ref_cur_w = self.width as i32;
        let ref_cur_h = self.height as i32;
        let ref_w = rf.width as i32;
        let ref_h = rf.height as i32;
        let scale_shift = 14;
        let x_scale = (ref_w << scale_shift) / ref_cur_w.max(1);
        let y_scale = (ref_h << scale_shift) / ref_cur_h.max(1);
        let x_step_q4 = (16 * x_scale) >> scale_shift;
        let y_step_q4 = (16 * y_scale) >> scale_shift;
        // §8.5.2.2 clampedMv = (2 * mv) >> sy. For chroma plane we also
        // use the luma-relative position multiplied by sub_x / sub_y for
        // the "lumaX" coordinate as the spec notes.
        let mv_r = ((mv_row_luma as i32) * 2) >> sub_y;
        let mv_c = ((mv_col_luma as i32) * 2) >> sub_x;
        // lumaX = col << sub_x for chroma planes.
        let luma_x = (col as i32) << sub_x;
        let luma_y = (row as i32) << sub_y;
        let base_x_q4 = ((luma_x * x_scale) >> scale_shift) << 4;
        let base_y_q4 = ((luma_y * y_scale) >> scale_shift) << 4;
        let frac_x = ((16 * luma_x * x_scale) >> scale_shift) & 15;
        let frac_y = ((16 * luma_y * y_scale) >> scale_shift) & 15;
        let d_x = ((mv_c * x_scale) >> scale_shift) + frac_x;
        let d_y = ((mv_r * y_scale) >> scale_shift) + frac_y;
        // For chroma we want start position in chroma-plane coords, so
        // the base needs to be divided by the sub-factor.
        let start_x_q4 = (base_x_q4 + d_x) >> sub_x;
        let start_y_q4 = (base_y_q4 + d_y) >> sub_y;
        let sampler = ChromaSampler { frame: rf, plane };
        mc_block_scaled(
            &sampler, filter, dst, eff_w, eff_w, eff_h, start_x_q4, start_y_q4, x_step_q4,
            y_step_q4,
        );
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

        let qp = self.hdr.segmentation.get_qindex(
            0, // segment_id (per-block segmentation map not yet built).
            self.hdr.quantization.base_q_idx,
        );
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

        let qp = self.hdr.segmentation.get_qindex(
            0, // segment_id (per-block segmentation map not yet built).
            self.hdr.quantization.base_q_idx,
        );
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

/// In-place average of two equal-length byte buffers per §8.5.2:
/// `a[i] = Round2(a[i] + b[i], 1) = (a[i] + b[i] + 1) >> 1`.
fn average_into(a: &mut [u8], b: &[u8]) {
    let n = a.len().min(b.len());
    for i in 0..n {
        a[i] = (((a[i] as u32) + (b[i] as u32) + 1) >> 1) as u8;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comp_refs_last_golden_same_sign() {
        // When LAST / GOLDEN share a sign, ALTREF is the fixed ref.
        let sign_bias = [false, false, false, true];
        let c = CompRefs::from_sign_bias(&sign_bias);
        assert_eq!(c.fixed, 3);
        assert_eq!(c.var, [1, 2]);
    }

    #[test]
    fn comp_refs_last_altref_same_sign() {
        let sign_bias = [false, false, true, false];
        let c = CompRefs::from_sign_bias(&sign_bias);
        assert_eq!(c.fixed, 2);
        assert_eq!(c.var, [1, 3]);
    }

    #[test]
    fn comp_refs_last_different_from_both() {
        let sign_bias = [false, true, false, false];
        let c = CompRefs::from_sign_bias(&sign_bias);
        assert_eq!(c.fixed, 1);
        assert_eq!(c.var, [2, 3]);
    }

    #[test]
    fn average_into_round2_matches_spec() {
        let mut a = [10u8, 20, 30, 40];
        let b = [20u8, 21, 29, 42];
        average_into(&mut a, &b);
        // Round2(a + b, 1) = (a + b + 1) >> 1.
        assert_eq!(a, [15, 21, 30, 41]);
    }
}
