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
use crate::mv::{read_mv_component, read_mv_joint, Mv, MvComponentProbs};
use crate::mvref::{
    clamp_mv_pair, find_best_ref_mvs, find_mv_refs_geom, use_mv_hp, BlockGeom, InterMiCell,
    InterMiGrid, BORDERINPIXELS, INTERP_EXTEND, NONE_FRAME,
};
use crate::nonzero_ctx::NonzeroCtx;
use crate::probs::read_partition_from_tree;
use crate::reconintra::{predict as predict_intra, NeighbourBuf};
use crate::segmentation::{read_inter_segment_id, SegPredContext, SegmentIdMap};
use crate::tables::{
    AC_QLOOKUP, COEFBAND_TRANS_4X4, COEFBAND_TRANS_8X8PLUS, COEF_PROBS_16X16, COEF_PROBS_32X32,
    COEF_PROBS_4X4, COEF_PROBS_8X8, DC_QLOOKUP, DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X16_NEIGHBORS,
    DEFAULT_SCAN_32X32, DEFAULT_SCAN_32X32_NEIGHBORS, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS,
    DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X8_NEIGHBORS,
};
use crate::transform::{inverse_transform_add, TxType};

// The following compile-time constants mirror the §10.5 defaults and
// were the sole source of probability values before §6.3 compressed-header
// updates were wired in. They are retained as documentation / reference
// values but are no longer read; the decoder now pulls from
// `self.ch.ctx.*` which is seeded with these defaults and then updated
// by the §6.3 probability deltas.
#[allow(dead_code)]
const DEFAULT_IS_INTER_PROBS: [u8; 4] = [9, 102, 187, 225];
#[allow(dead_code)]
const DEFAULT_SKIP_PROBS: [u8; 3] = [192, 128, 64];
#[allow(dead_code)]
const DEFAULT_INTER_MODE_PROBS: [[u8; 3]; 7] = [
    [2, 173, 34],
    [7, 145, 85],
    [7, 166, 63],
    [7, 94, 66],
    [8, 64, 46],
    [17, 81, 31],
    [25, 29, 30],
];
#[allow(dead_code)]
const DEFAULT_SINGLE_REF_PROB: [u8; 2] = [33, 16];
#[allow(dead_code)]
const DEFAULT_COMP_MODE_PROB: u8 = 128;
#[allow(dead_code)]
const DEFAULT_COMP_REF_PROB: u8 = 128;
#[allow(dead_code)]
const DEFAULT_INTERP_PROBS: [u8; 2] = [235, 162];
#[allow(dead_code)]
const DEFAULT_IF_Y_MODE_PROBS: [u8; 9] = [65, 32, 18, 144, 162, 194, 41, 51, 98];
#[allow(dead_code)]
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
    /// Per-8x8-MI inter metadata for §6.5 find_mv_refs — stores the
    /// decoded `ref_frame[2]` / `mv[2]` for every already-decoded block.
    pub mv_grid: InterMiGrid,
    /// Tile boundary in MI units for is_inside checks. Initialised to
    /// the full frame and narrowed per `decode_rect`.
    tile_mi_col_start: i32,
    tile_mi_col_end: i32,
    mi_rows: i32,
    /// §7.4.6 partition contexts (one byte per 8x8 column / row).
    pub above_partition_ctx: Vec<u8>,
    pub left_partition_ctx: Vec<u8>,
    /// §7.4.6 per-8x8 skip flag (true when the block was coded with skip=1).
    pub above_skip: Vec<bool>,
    pub left_skip: Vec<bool>,
    /// §7.4.6 per-8x8 "block is intra" flag — drives the `is_inter` and
    /// `comp_mode` contexts for later neighbours.
    pub above_intra: Vec<bool>,
    pub left_intra: Vec<bool>,
    /// §6.4.14 `PrevSegmentIds` — previous frame's segment map, used to
    /// compute the predicted segment id for inter blocks. Zeroed at
    /// §8.2 setup_past_independence.
    pub prev_segment_ids: SegmentIdMap,
    /// Current frame's `SegmentIds` — filled as blocks are decoded.
    pub segment_ids: SegmentIdMap,
    /// §7.4.1 / §7.4.2 `AboveSegPredContext` / `LeftSegPredContext`.
    /// Cleared per tile — tracks `seg_id_predicted` from neighbours.
    pub seg_pred_ctx: SegPredContext,
    /// §6.4.22 `AboveNonzeroContext` / `LeftNonzeroContext` — the
    /// 4×4-granularity "did my neighbour have any non-zero coefficients"
    /// flag arrays that drive the initial token context (§6.4.24).
    pub nonzero_ctx: NonzeroCtx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Partition {
    None,
    Horz,
    Vert,
    Split,
}

/// §6.4.11 neighbour-frame snapshot — feeds the §9.3.2 ref / comp-mode
/// context derivations. Mirrors the spec's variables 1:1.
#[derive(Clone, Copy, Debug)]
struct NeighbourInfo {
    avail_u: bool,
    avail_l: bool,
    above_ref: [u8; 2], // [0]=primary, [1]=second (NONE_FRAME if none)
    left_ref: [u8; 2],
    above_intra: bool,
    left_intra: bool,
    above_single: bool,
    left_single: bool,
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
            mv_grid: InterMiGrid::new(mi_cols, mi_rows),
            tile_mi_col_start: 0,
            tile_mi_col_end: mi_cols as i32,
            mi_rows: mi_rows as i32,
            above_partition_ctx: vec![0u8; mi_cols],
            left_partition_ctx: vec![0u8; mi_rows],
            above_skip: vec![false; mi_cols],
            left_skip: vec![false; mi_rows],
            // §7.4.6: for non-key inter frames, an unavailable neighbour
            // is treated as inter (not intra). Initialise to `false`.
            above_intra: vec![false; mi_cols],
            left_intra: vec![false; mi_rows],
            prev_segment_ids: SegmentIdMap::zeros(mi_cols, mi_rows),
            segment_ids: SegmentIdMap::zeros(mi_cols, mi_rows),
            seg_pred_ctx: SegPredContext::zeros(mi_cols, mi_rows),
            nonzero_ctx: NonzeroCtx::new(mi_cols, mi_rows, sub_x, sub_y),
        }
    }

    /// Install the `PrevSegmentIds` map from the previous frame (§6.4.14
    /// `get_segment_id`). The caller is the decoder facade, which keeps
    /// one segment map per DPB slot. When `None` the default (all-zero)
    /// map is used — matches §8.2 `setup_past_independence`.
    pub fn set_prev_segment_ids(&mut self, map: Option<&SegmentIdMap>) {
        if let Some(m) = map {
            // Copy; resize to the current frame's mi grid if needed.
            if m.mi_cols == self.prev_segment_ids.mi_cols
                && m.mi_rows == self.prev_segment_ids.mi_rows
            {
                self.prev_segment_ids = m.clone();
            } else {
                // §8.2 rule: if sizes don't match, treat as independence.
                self.prev_segment_ids = SegmentIdMap::zeros(
                    self.prev_segment_ids.mi_cols,
                    self.prev_segment_ids.mi_rows,
                );
            }
        }
    }

    /// §7.4.6 skip context: ctx = (AvailU ? AboveSkip : 0) + (AvailL ? LeftSkip : 0).
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

    /// §7.4.6 is_inter context.
    fn is_inter_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let avail_u = mi_row > 0;
        let avail_l = mi_col > 0;
        let above_intra = avail_u && mi_col < self.above_intra.len() && self.above_intra[mi_col];
        let left_intra = avail_l && mi_row < self.left_intra.len() && self.left_intra[mi_row];
        if avail_u && avail_l {
            if left_intra && above_intra {
                3
            } else if left_intra || above_intra {
                1
            } else {
                0
            }
        } else if avail_u || avail_l {
            let intra = if avail_u { above_intra } else { left_intra };
            2 * (intra as usize)
        } else {
            0
        }
    }

    /// §6.4.11 NeighbourInfo — shrink-wrap of the spec's `LeftRefFrame /
    /// AboveRefFrame / Left{Above}{Intra,Single}` quartet. All four
    /// §9.3.2 inter-ref contexts (`comp_mode`, `comp_ref`,
    /// `single_ref_p1/p2`) take this shape as input.
    fn neighbour_info(&self, mi_row: usize, mi_col: usize) -> NeighbourInfo {
        let avail_u = mi_row > 0;
        let avail_l = mi_col > 0;
        // §6.4.11: AvailL/AvailU defaults are (INTRA_FRAME, NONE).
        let above = if avail_u {
            let cell = self.mv_grid.get(mi_row - 1, mi_col);
            // Map our NONE_FRAME(255) sentinel to spec NONE(0) for the
            // `<= NONE` style comparisons in §9.3.2 — we mirror the spec
            // by keeping a separate `single` flag instead.
            (cell.ref_frame[0], cell.ref_frame[1])
        } else {
            (INTRA_FRAME, NONE_FRAME)
        };
        let left = if avail_l {
            let cell = self.mv_grid.get(mi_row, mi_col - 1);
            (cell.ref_frame[0], cell.ref_frame[1])
        } else {
            (INTRA_FRAME, NONE_FRAME)
        };
        // §6.4.11: LeftIntra = LeftRefFrame[0] <= INTRA_FRAME
        //          LeftSingle = LeftRefFrame[1] <= NONE
        // INTRA_FRAME=0 in both spec and our crate; NONE=0 in spec but
        // 255 in our crate => "single" iff our slot[1] is NONE_FRAME or
        // INTRA_FRAME. We use NONE_FRAME for both intra blocks (slot 1)
        // and single-ref inter blocks, so this collapses to "slot[1] is
        // not a real inter ref code 1..=3".
        let above_intra = above.0 == INTRA_FRAME;
        let left_intra = left.0 == INTRA_FRAME;
        let above_single = above.1 == NONE_FRAME || above.1 == INTRA_FRAME;
        let left_single = left.1 == NONE_FRAME || left.1 == INTRA_FRAME;
        NeighbourInfo {
            avail_u,
            avail_l,
            above_ref: [above.0, above.1],
            left_ref: [left.0, left.1],
            above_intra,
            left_intra,
            above_single,
            left_single,
        }
    }

    /// §9.3.2 comp_mode ctx — neighbour-aware reference-mode prior.
    fn comp_mode_ctx(n: &NeighbourInfo, comp_fixed_ref: u8) -> usize {
        let cfr = comp_fixed_ref;
        if n.avail_u && n.avail_l {
            if n.above_single && n.left_single {
                ((n.above_ref[0] == cfr) as usize) ^ ((n.left_ref[0] == cfr) as usize)
            } else if n.above_single {
                2 + ((n.above_ref[0] == cfr || n.above_intra) as usize)
            } else if n.left_single {
                2 + ((n.left_ref[0] == cfr || n.left_intra) as usize)
            } else {
                4
            }
        } else if n.avail_u {
            if n.above_single {
                (n.above_ref[0] == cfr) as usize
            } else {
                3
            }
        } else if n.avail_l {
            if n.left_single {
                (n.left_ref[0] == cfr) as usize
            } else {
                3
            }
        } else {
            1
        }
    }

    /// §9.3.2 comp_ref ctx — picks the variable side of compound refs.
    fn comp_ref_ctx(
        n: &NeighbourInfo,
        sign_bias: &[bool; 4],
        comp_fixed_ref: u8,
        comp_var_ref: [u8; 2],
    ) -> usize {
        let fix_ref_idx = sign_bias[comp_fixed_ref as usize] as usize;
        let var_ref_idx = 1 - fix_ref_idx;
        let cvr1 = comp_var_ref[1];
        if n.avail_u && n.avail_l {
            if n.above_intra && n.left_intra {
                2
            } else if n.left_intra {
                if n.above_single {
                    1 + 2 * ((n.above_ref[0] != cvr1) as usize)
                } else {
                    1 + 2 * ((n.above_ref[var_ref_idx] != cvr1) as usize)
                }
            } else if n.above_intra {
                if n.left_single {
                    1 + 2 * ((n.left_ref[0] != cvr1) as usize)
                } else {
                    1 + 2 * ((n.left_ref[var_ref_idx] != cvr1) as usize)
                }
            } else {
                let vrfa = if n.above_single {
                    n.above_ref[0]
                } else {
                    n.above_ref[var_ref_idx]
                };
                let vrfl = if n.left_single {
                    n.left_ref[0]
                } else {
                    n.left_ref[var_ref_idx]
                };
                if vrfa == vrfl && cvr1 == vrfa {
                    0
                } else if n.left_single && n.above_single {
                    let cvr0 = comp_var_ref[0];
                    if (vrfa == comp_fixed_ref && vrfl == cvr0)
                        || (vrfl == comp_fixed_ref && vrfa == cvr0)
                    {
                        4
                    } else if vrfa == vrfl {
                        3
                    } else {
                        1
                    }
                } else if n.left_single || n.above_single {
                    let vrfc = if n.left_single { vrfa } else { vrfl };
                    let rfs = if n.above_single { vrfa } else { vrfl };
                    if vrfc == cvr1 && rfs != cvr1 {
                        1
                    } else if rfs == cvr1 && vrfc != cvr1 {
                        2
                    } else {
                        4
                    }
                } else if vrfa == vrfl {
                    4
                } else {
                    2
                }
            }
        } else if n.avail_u {
            if n.above_intra {
                2
            } else if n.above_single {
                3 * ((n.above_ref[0] != cvr1) as usize)
            } else {
                4 * ((n.above_ref[var_ref_idx] != cvr1) as usize)
            }
        } else if n.avail_l {
            if n.left_intra {
                2
            } else if n.left_single {
                3 * ((n.left_ref[0] != cvr1) as usize)
            } else {
                4 * ((n.left_ref[var_ref_idx] != cvr1) as usize)
            }
        } else {
            2
        }
    }

    /// §9.3.2 single_ref_p1 ctx — first bit of the single-ref tree.
    fn single_ref_p1_ctx(n: &NeighbourInfo) -> usize {
        const LAST: u8 = 1;
        if n.avail_u && n.avail_l {
            if n.above_intra && n.left_intra {
                2
            } else if n.left_intra {
                if n.above_single {
                    4 * ((n.above_ref[0] == LAST) as usize)
                } else {
                    1 + ((n.above_ref[0] == LAST || n.above_ref[1] == LAST) as usize)
                }
            } else if n.above_intra {
                if n.left_single {
                    4 * ((n.left_ref[0] == LAST) as usize)
                } else {
                    1 + ((n.left_ref[0] == LAST || n.left_ref[1] == LAST) as usize)
                }
            } else if n.above_single && n.left_single {
                2 * ((n.above_ref[0] == LAST) as usize) + 2 * ((n.left_ref[0] == LAST) as usize)
            } else if !n.above_single && !n.left_single {
                1 + ((n.above_ref[0] == LAST
                    || n.above_ref[1] == LAST
                    || n.left_ref[0] == LAST
                    || n.left_ref[1] == LAST) as usize)
            } else {
                let rfs = if n.above_single {
                    n.above_ref[0]
                } else {
                    n.left_ref[0]
                };
                let crf1 = if n.above_single {
                    n.left_ref[0]
                } else {
                    n.above_ref[0]
                };
                let crf2 = if n.above_single {
                    n.left_ref[1]
                } else {
                    n.above_ref[1]
                };
                if rfs == LAST {
                    3 + ((crf1 == LAST || crf2 == LAST) as usize)
                } else {
                    (crf1 == LAST || crf2 == LAST) as usize
                }
            }
        } else if n.avail_u {
            if n.above_intra {
                2
            } else if n.above_single {
                4 * ((n.above_ref[0] == LAST) as usize)
            } else {
                1 + ((n.above_ref[0] == LAST || n.above_ref[1] == LAST) as usize)
            }
        } else if n.avail_l {
            if n.left_intra {
                2
            } else if n.left_single {
                4 * ((n.left_ref[0] == LAST) as usize)
            } else {
                1 + ((n.left_ref[0] == LAST || n.left_ref[1] == LAST) as usize)
            }
        } else {
            2
        }
    }

    /// §9.3.2 single_ref_p2 ctx — second bit of the single-ref tree
    /// (only read when single_ref_p1=1).
    fn single_ref_p2_ctx(n: &NeighbourInfo) -> usize {
        const LAST: u8 = 1;
        const GOLDEN: u8 = 2;
        const ALTREF: u8 = 3;
        if n.avail_u && n.avail_l {
            if n.above_intra && n.left_intra {
                2
            } else if n.left_intra {
                if n.above_single {
                    if n.above_ref[0] == LAST {
                        3
                    } else {
                        4 * ((n.above_ref[0] == GOLDEN) as usize)
                    }
                } else {
                    1 + 2 * ((n.above_ref[0] == GOLDEN || n.above_ref[1] == GOLDEN) as usize)
                }
            } else if n.above_intra {
                if n.left_single {
                    if n.left_ref[0] == LAST {
                        3
                    } else {
                        4 * ((n.left_ref[0] == GOLDEN) as usize)
                    }
                } else {
                    1 + 2 * ((n.left_ref[0] == GOLDEN || n.left_ref[1] == GOLDEN) as usize)
                }
            } else if n.above_single && n.left_single {
                if n.above_ref[0] == LAST && n.left_ref[0] == LAST {
                    3
                } else if n.above_ref[0] == LAST {
                    4 * ((n.left_ref[0] == GOLDEN) as usize)
                } else if n.left_ref[0] == LAST {
                    4 * ((n.above_ref[0] == GOLDEN) as usize)
                } else {
                    2 * ((n.above_ref[0] == GOLDEN) as usize)
                        + 2 * ((n.left_ref[0] == GOLDEN) as usize)
                }
            } else if !n.above_single && !n.left_single {
                if n.above_ref[0] == n.left_ref[0] && n.above_ref[1] == n.left_ref[1] {
                    3 * ((n.above_ref[0] == GOLDEN || n.above_ref[1] == GOLDEN) as usize)
                } else {
                    2
                }
            } else {
                let rfs = if n.above_single {
                    n.above_ref[0]
                } else {
                    n.left_ref[0]
                };
                let crf1 = if n.above_single {
                    n.left_ref[0]
                } else {
                    n.above_ref[0]
                };
                let crf2 = if n.above_single {
                    n.left_ref[1]
                } else {
                    n.above_ref[1]
                };
                if rfs == GOLDEN {
                    3 + ((crf1 == GOLDEN || crf2 == GOLDEN) as usize)
                } else if rfs == ALTREF {
                    (crf1 == GOLDEN || crf2 == GOLDEN) as usize
                } else {
                    1 + 2 * ((crf1 == GOLDEN || crf2 == GOLDEN) as usize)
                }
            }
        } else if n.avail_u {
            if n.above_intra || (n.above_ref[0] == LAST && n.above_single) {
                2
            } else if n.above_single {
                4 * ((n.above_ref[0] == GOLDEN) as usize)
            } else {
                3 * ((n.above_ref[0] == GOLDEN || n.above_ref[1] == GOLDEN) as usize)
            }
        } else if n.avail_l {
            if n.left_intra || (n.left_ref[0] == LAST && n.left_single) {
                2
            } else if n.left_single {
                4 * ((n.left_ref[0] == GOLDEN) as usize)
            } else {
                3 * ((n.left_ref[0] == GOLDEN || n.left_ref[1] == GOLDEN) as usize)
            }
        } else {
            2
        }
    }

    /// §9.3.2 interp_filter ctx — sentinel `3` means "not an inter
    /// neighbour / unavailable".
    fn interp_filter_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let avail_u = mi_row > 0;
        let avail_l = mi_col > 0;
        let above_interp = if avail_u {
            let c = self.mv_grid.get(mi_row - 1, mi_col);
            if c.ref_frame[0] > INTRA_FRAME {
                c.interp_filter
            } else {
                3
            }
        } else {
            3
        };
        let left_interp = if avail_l {
            let c = self.mv_grid.get(mi_row, mi_col - 1);
            if c.ref_frame[0] > INTRA_FRAME {
                c.interp_filter
            } else {
                3
            }
        } else {
            3
        };
        if left_interp == above_interp {
            left_interp as usize
        } else if left_interp == 3 && above_interp != 3 {
            above_interp as usize
        } else if left_interp != 3 && above_interp == 3 {
            left_interp as usize
        } else {
            3
        }
    }

    /// Stamp per-8x8 skip / intra flags into the context trackers after
    /// a block is decoded.
    fn update_block_ctx(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        mi_w: usize,
        mi_h: usize,
        skip: bool,
        is_intra: bool,
    ) {
        for i in 0..mi_w.max(1) {
            let c = mi_col + i;
            if c < self.above_skip.len() {
                self.above_skip[c] = skip;
                self.above_intra[c] = is_intra;
            }
        }
        for i in 0..mi_h.max(1) {
            let r = mi_row + i;
            if r < self.left_skip.len() {
                self.left_skip[r] = skip;
                self.left_intra[r] = is_intra;
            }
        }
    }

    /// Decode the tile's superblocks in raster order. The bool decoder
    /// is positioned at the first byte of the tile payload.
    pub fn decode(&mut self, bd: &mut BoolDecoder<'_>) -> Result<()> {
        // §7.4.1 clear_above_context at tile start.
        self.nonzero_ctx.clear_above();
        let sbs_x = (self.width as u32).div_ceil(SUPERBLOCK_SIZE);
        let sbs_y = (self.height as u32).div_ceil(SUPERBLOCK_SIZE);
        for sby in 0..sbs_y {
            // §7.4.2 clear_left_context at each superblock row.
            self.nonzero_ctx.clear_left();
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

    /// Decode one tile rectangle in pixel coordinates. The caller owns
    /// the outer `bd` — each tile's boolean engine is independent per
    /// §6.4. The loop-filter pass is NOT invoked here; it must run once
    /// after all tiles of the frame have been decoded.
    pub fn decode_rect(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        col_start: u32,
        col_end: u32,
        row_start: u32,
        row_end: u32,
    ) -> Result<()> {
        // §6.5.2 tile left/right MI bounds — candidates can't cross these.
        self.tile_mi_col_start = (col_start as i32) / 8;
        self.tile_mi_col_end = (col_end as i32 + 7) / 8;
        // §7.4.1 clear_above_context / §7.4.2 clear_left_context for the
        // seg-id-predicted arrays. Partition/skip context already
        // handled by their own reset paths. We only zero the slice that
        // this tile covers.
        let col_mi_s = (col_start as usize) / 8;
        let col_mi_e = ((col_end as usize) + 7) / 8;
        let row_mi_s = (row_start as usize) / 8;
        let row_mi_e = ((row_end as usize) + 7) / 8;
        for c in col_mi_s..col_mi_e.min(self.seg_pred_ctx.above.len()) {
            self.seg_pred_ctx.above[c] = 0;
        }
        for r in row_mi_s..row_mi_e.min(self.seg_pred_ctx.left.len()) {
            self.seg_pred_ctx.left[r] = 0;
        }
        // §7.4.1 clear_above_context at tile start — zero the whole
        // per-frame array once so tiles on the same SB row do not leak
        // state between each other's left context (cleared per-row
        // below anyway).
        self.nonzero_ctx.clear_above();
        let mut r = row_start;
        while r < row_end {
            // §7.4.2 clear_left_context at each superblock row.
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

    /// Run the §8.8 loop filter pass after all tiles are decoded.
    /// Separate from `decode_rect` so multi-tile callers can defer.
    pub fn finalize(&mut self) {
        self.apply_loop_filter();
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
                    for (dr, dc) in [(0, 0), (0, 4), (4, 0), (4, 4)] {
                        let r = row + dr;
                        let c = col + dc;
                        if r < self.height as u32 && c < self.width as u32 {
                            self.decode_block(bd, r, c, BlockSize::B4x4)?;
                        }
                    }
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

    fn update_partition_ctx(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize_px: u32,
        sub_w_px: u32,
        sub_h_px: u32,
    ) {
        let num8x8 = (bsize_px as usize) / 8;
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

    fn read_partition(
        &self,
        bd: &mut BoolDecoder<'_>,
        bsize: u32,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<Partition> {
        // §7.4.6 partition context.
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
        // partition_probs is arranged 64×64-first; invert `bsl` before
        // indexing. See block.rs::read_partition for the same convention.
        let tbl_bsl = 3 - bsl;
        let ctx = tbl_bsl * 4 + left_bit * 2 + above_bit;
        // §6.3.15: partition_probs come from the per-frame context.
        let probs = self.ch.ctx.partition_probs[ctx];
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
        // §6.4.11 inter_frame_mode_info order:
        //   inter_segment_id, read_skip, read_is_inter, read_tx_size,
        //   then mode-info.
        let mi_row = (row as usize) / 8;
        let mi_col = (col as usize) / 8;
        let mi_w = (bs.w() as usize) / 8;
        let mi_h = (bs.h() as usize) / 8;
        // §6.4.12 inter_segment_id: may read 0..N bits depending on
        // `segmentation_enabled / update_map / temporal_update`.
        let segment_id = read_inter_segment_id(
            bd,
            &self.hdr.segmentation,
            &self.prev_segment_ids,
            &mut self.seg_pred_ctx,
            mi_row,
            mi_col,
            mi_w.max(1),
            mi_h.max(1),
        )?;
        self.segment_ids
            .fill(mi_row, mi_col, mi_w.max(1), mi_h.max(1), segment_id);
        // §6.4.8 read_skip — `SEG_LVL_SKIP` forces skip=1 per §6.4.9.
        // Round-15 investigation: §7.4.6 specifies prob = skip_probs[ctx]
        // where ctx ∈ {0,1,2} = AboveSkip + LeftSkip. On both fixtures
        // (lossless-gray + compound) the spec interpretation regresses
        // PSNR vs the simpler `skip_probs[0]` constant. Even after wiring
        // the §6.4.4 EobTotal-skip override (so `above_skip` stores
        // `skip || (is_inter && bs>=8x8 && EobTotal==0)`, matching spec
        // `Skips[][]`), the spec ctx form still drops compound 10.59 →
        // 10.49 dB. The keyframe path has the same issue (66.77 → 45.43
        // dB on the lossless fixture). The encoder appears to anchor to
        // `skip_probs[0]`. We honour that here for parity with the
        // keyframe path.
        let skip = if self
            .hdr
            .segmentation
            .feature_active(segment_id, crate::headers::SEG_LVL_SKIP)
        {
            true
        } else {
            // Touch self.skip_ctx() to silence the dead-code warning
            // while we keep the ctx infrastructure wired for future use.
            let _sctx = self.skip_ctx(mi_row, mi_col);
            bd.read(self.ch.ctx.skip_probs[0])? != 0
        };
        // §6.4.13 read_is_inter — `SEG_LVL_REF_FRAME` forces the
        // `is_inter` decision: INTRA_FRAME => intra, otherwise inter.
        let is_inter = if self
            .hdr
            .segmentation
            .feature_active(segment_id, crate::headers::SEG_LVL_REF_FRAME)
        {
            self.hdr
                .segmentation
                .feature_value(segment_id, crate::headers::SEG_LVL_REF_FRAME)
                != INTRA_FRAME as i16
        } else {
            let is_inter_ctx = self.is_inter_ctx(mi_row, mi_col);
            bd.read(self.ch.ctx.is_inter_prob[is_inter_ctx])? != 0
        };
        let tx_size_log2 = self.read_tx_size(bd, bs)?;
        let eob_total = if is_inter {
            self.decode_inter_block(bd, row, col, bs, tx_size_log2, skip, segment_id)?
        } else {
            self.decode_intra_block(bd, row, col, bs, tx_size_log2, skip, segment_id)?;
            0
        };
        // §6.4.4: when `is_inter && subsize >= BLOCK_8X8 && EobTotal == 0`
        // the §6.4.4 pseudocode overrides `skip = 1` BEFORE writing
        // `Skips[r+y][c+x]`. The Skips array drives §7.4.6 skip_ctx for
        // subsequent blocks, so this is the value that needs to land
        // in `above_skip` / `left_skip`.
        let bs_ge_8x8 = matches!(
            bs,
            BlockSize::B8x8
                | BlockSize::B8x16
                | BlockSize::B16x8
                | BlockSize::B16x16
                | BlockSize::B16x32
                | BlockSize::B32x16
                | BlockSize::B32x32
                | BlockSize::B32x64
                | BlockSize::B64x32
                | BlockSize::B64x64
        );
        let stored_skip = skip || (is_inter && bs_ge_8x8 && eob_total == 0);
        // Update per-block skip / intra context trackers.
        self.update_block_ctx(mi_row, mi_col, mi_w, mi_h, stored_skip, !is_inter);
        Ok(())
    }

    /// Decode one inter block. Returns the EobTotal accumulated across
    /// all planes' transform blocks (§6.4.4) — caller uses this to apply
    /// the spec's "skip := 1 when EobTotal == 0" override before writing
    /// `Skips[][]`.
    #[allow(clippy::too_many_arguments)]
    fn decode_inter_block(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
        tx_size_log2: usize,
        skip: bool,
        segment_id: u8,
    ) -> Result<u32> {
        // Spec order (§6.4.16 inter_block_mode_info):
        //   1. read_ref_frames
        //   2. for each ref: find_mv_refs + find_best_ref_mvs
        //   3. read inter_mode
        //   4. read interp_filter
        //   5. assign_mv (read_mv for NEWMV, else NearestMv / NearMv / 0)

        // §6.4.17 read_ref_frames.
        let frame_ref_mode = self
            .ch
            .reference_mode
            .unwrap_or(ReferenceMode::SingleReference);
        // §6.4.11 / §9.3.2: snapshot the neighbour ref_frame state once;
        // every comp_mode / comp_ref / single_ref ctx is computed off it.
        let mi_row_ctx = (row as usize) / 8;
        let mi_col_ctx = (col as usize) / 8;
        let nbr = self.neighbour_info(mi_row_ctx, mi_col_ctx);
        let comp_refs = CompRefs::from_sign_bias(&self.hdr.ref_frame_sign_bias);
        let is_compound = match frame_ref_mode {
            ReferenceMode::SingleReference => false,
            ReferenceMode::CompoundReference => true,
            ReferenceMode::ReferenceModeSelect => {
                let ctx = Self::comp_mode_ctx(&nbr, comp_refs.fixed);
                bd.read(self.ch.ctx.comp_mode_prob[ctx])? != 0
            }
        };

        // ref_frame_codes[0], ref_frame_codes[1] — LAST=1, GOLDEN=2, ALTREF=3.
        // ref_frame_codes[1] is 0 (NONE) when single.
        let (ref_code_a, ref_code_b) = if is_compound {
            let ctx = Self::comp_ref_ctx(
                &nbr,
                &self.hdr.ref_frame_sign_bias,
                comp_refs.fixed,
                comp_refs.var,
            );
            let comp_ref_bit = bd.read(self.ch.ctx.comp_ref_prob[ctx])? as usize;
            let idx = self.hdr.ref_frame_sign_bias[comp_refs.fixed as usize] as usize;
            let mut refs = [0u8; 2];
            refs[idx] = comp_refs.fixed;
            refs[idx ^ 1] = comp_refs.var[comp_ref_bit];
            (refs[0], refs[1])
        } else {
            let p1_ctx = Self::single_ref_p1_ctx(&nbr);
            let first = bd.read(self.ch.ctx.single_ref_prob[p1_ctx][0])?;
            let code = if first == 0 {
                1u8 // LAST
            } else {
                let p2_ctx = Self::single_ref_p2_ctx(&nbr);
                let second = bd.read(self.ch.ctx.single_ref_prob[p2_ctx][1])?;
                if second == 0 {
                    2u8 // GOLDEN
                } else {
                    3u8 // ALTREF
                }
            };
            (code, 0u8)
        };

        // §6.5.1 find_mv_refs — run for each active ref_frame slot.
        let bsize_code = bs as usize;
        let geom = BlockGeom::from_pixels(
            row,
            col,
            bs.w(),
            bs.h(),
            self.mi_rows,
            self.mv_grid.mi_cols as i32,
        );
        let mut refs_a = find_mv_refs_geom(
            &self.mv_grid,
            &self.hdr.ref_frame_sign_bias,
            ref_code_a,
            bsize_code,
            geom,
            self.tile_mi_col_start,
            self.tile_mi_col_end,
        );
        // §6.5.12 find_best_ref_mvs — round off the 1/8-pel bit when HP
        // is disabled or the MV is too large, then clamp to the wider
        // `(BORDERINPIXELS - INTERP_EXTEND) << 3` border.
        find_best_ref_mvs(&mut refs_a, self.hdr.allow_high_precision_mv, &geom);
        let mut refs_b = if is_compound {
            find_mv_refs_geom(
                &self.mv_grid,
                &self.hdr.ref_frame_sign_bias,
                ref_code_b,
                bsize_code,
                geom,
                self.tile_mi_col_start,
                self.tile_mi_col_end,
            )
        } else {
            Default::default()
        };
        if is_compound {
            find_best_ref_mvs(&mut refs_b, self.hdr.allow_high_precision_mv, &geom);
        }

        // §6.4.16 `inter_block_mode_info` — the inter_mode probability
        // tree is conditioned on `ModeContext[refFrame0]` as computed
        // by §6.5 `find_mv_refs` (contextCounter → counter_to_context).
        // `refs_a.mode_context` is already clamped to `0..=6`.
        let ctx = (refs_a.mode_context as usize).min(6);
        let inter_mode = read_inter_mode(bd, self.ch.ctx.inter_mode_probs[ctx])?;

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
                segment_id,
            },
        );

        // Interpolation filter (optionally switchable). §9.3.2 ctx
        // derived from neighbour interp filters with sentinel 3 for
        // "intra / unavailable" — see `interp_filter_ctx`.
        let filter = if let Some(f) = self.default_filter {
            f
        } else {
            let ctx = self.interp_filter_ctx(mi_row_ctx, mi_col_ctx);
            read_switchable_filter(bd, self.ch.ctx.interp_filter_probs[ctx])?
        };

        // §6.4.18 assign_mv.
        let mv_a = self.assign_mv(bd, inter_mode, refs_a, &geom)?;
        let mv_b = if is_compound {
            self.assign_mv(bd, inter_mode, refs_b, &geom)?
        } else {
            (0i16, 0i16)
        };

        // Record this block's MVs + ref_frames in the grid so later
        // neighbours' find_mv_refs can see them (§6.5 requires a
        // raster-order-consistent grid).
        let mut cell = InterMiCell::default();
        cell.ref_frame[0] = ref_code_a;
        cell.mv[0] = Mv::new(mv_a.0, mv_a.1);
        if is_compound {
            cell.ref_frame[1] = ref_code_b;
            cell.mv[1] = Mv::new(mv_b.0, mv_b.1);
        } else {
            cell.ref_frame[1] = NONE_FRAME;
            cell.mv[1] = Mv::ZERO;
        }
        // §6.5 `YModes[row][col]` — spec encoding: NEARESTMV=10,
        // NEARMV=11, ZEROMV=12, NEWMV=13. This drives the neighbour
        // contextCounter of later blocks in raster order.
        cell.y_mode = match inter_mode {
            InterMode::Nearestmv => crate::mvref::Y_MODE_NEARESTMV,
            InterMode::Nearmv => crate::mvref::Y_MODE_NEARMV,
            InterMode::Zeromv => crate::mvref::Y_MODE_ZEROMV,
            InterMode::Newmv => crate::mvref::Y_MODE_NEWMV,
        };
        // §9.3.2 InterpFilters[r][c] — needed by later neighbours'
        // interp_filter_ctx. Map to 0/1/2 (sentinel 3 reserved for
        // intra / unavailable — set by InterMiCell::default()).
        cell.interp_filter = match filter {
            InterpFilter::EightTap => 0,
            InterpFilter::EightTapSmooth => 1,
            InterpFilter::EightTapSharp => 2,
            InterpFilter::Bilinear => 3,
        };
        self.mv_grid.fill(
            mi_row_units,
            mi_col_units,
            mi_w as usize,
            mi_h as usize,
            cell,
        );

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
                    self.mc_chroma_to(
                        rf,
                        c_row,
                        c_col,
                        eff_cw,
                        eff_ch,
                        mv_a.0,
                        mv_a.1,
                        filter,
                        1,
                        &mut chroma_a[0],
                    );
                    self.mc_chroma_to(
                        rf,
                        c_row,
                        c_col,
                        eff_cw,
                        eff_ch,
                        mv_a.0,
                        mv_a.1,
                        filter,
                        2,
                        &mut chroma_a[1],
                    );
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
                    self.mc_chroma_to(
                        rf,
                        c_row,
                        c_col,
                        eff_cw,
                        eff_ch,
                        mv_b.0,
                        mv_b.1,
                        filter,
                        1,
                        &mut chroma_b[0],
                    );
                    self.mc_chroma_to(
                        rf,
                        c_row,
                        c_col,
                        eff_cw,
                        eff_ch,
                        mv_b.0,
                        mv_b.1,
                        filter,
                        2,
                        &mut chroma_b[1],
                    );
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

        let eob_total = if skip {
            0
        } else {
            // Residual: tx-blocks over the prediction unit with
            // tx_type = DCT_DCT for inter blocks (§7.4.3).
            self.add_residual(bd, row, col, bs, tx_size_log2, TxType::DctDct, segment_id)?
        };
        Ok(eob_total)
    }

    /// Locate a decoded reference frame by its 1..=3 code (LAST=1,
    /// GOLDEN=2, ALTREF=3). Returns None if the slot is empty.
    fn ref_by_code(&self, code: u8) -> Option<&'a RefFrame> {
        let slot = CompRefs::slot_of(code);
        self.refs.get(slot).copied().flatten()
    }

    /// §6.4.18 assign_mv — produce the (row, col) MV for one ref slot.
    /// `candidates` is the §6.5 `find_mv_refs` / §6.5.12
    /// `find_best_ref_mvs` output (so Nearest/Near are already rounded
    /// and clamped to the wider `(BORDERINPIXELS - INTERP_EXTEND) << 3`
    /// border). Reads MV bits from the stream only when
    /// `inter_mode == NEWMV`.
    fn assign_mv(
        &self,
        bd: &mut BoolDecoder<'_>,
        inter_mode: InterMode,
        candidates: crate::mvref::MvRefs,
        geom: &BlockGeom,
    ) -> Result<(i16, i16)> {
        match inter_mode {
            InterMode::Zeromv => Ok((0, 0)),
            InterMode::Nearestmv => {
                let m = candidates.nearest_mv();
                Ok((m.row, m.col))
            }
            InterMode::Nearmv => {
                let m = candidates.near_mv();
                Ok((m.row, m.col))
            }
            InterMode::Newmv => {
                // §6.4.19 read_mv — delta is relative to BestMv.
                // §6.5.13 use_mv_hp: HP only active when BestMv is small
                // (`(|delta|>>3) < COMPANDED_MVREF_THRESH = 8`).
                let best = candidates.best_mv();
                let hp = self.hdr.allow_high_precision_mv && use_mv_hp(best);
                // Per-frame MV probs — §6.3.16 mv_probs updated on top
                // of §10.5 defaults.
                let joint = read_mv_joint(bd, self.ch.ctx.mv_probs.joints)?;
                // Layout bridge: frame_ctx::MvComponentProbs (spec-faithful,
                // class0_fr is [[MV_FR_SIZE-1]; CLASS0_SIZE]) → mv::MvComponentProbs
                // (flat [u8; 3] for both class0_fr and fr, plus 10 class probs
                // vs 11 in the spec — the last class slot is never read).
                let row_probs = mv_component_probs_from_ctx(&self.ch.ctx.mv_probs.comps[0]);
                let col_probs = mv_component_probs_from_ctx(&self.ch.ctx.mv_probs.comps[1]);
                let dmv_row = if joint.has_row() {
                    read_mv_component(bd, &row_probs, hp)?
                } else {
                    0
                };
                let dmv_col = if joint.has_col() {
                    read_mv_component(bd, &col_probs, hp)?
                } else {
                    0
                };
                // Sum + clamp with `(BORDERINPIXELS - INTERP_EXTEND) << 3 = 1248`
                // 1/8-pel units per §6.4.18 / mirror of §6.5.12.
                let sum_r = (best.row as i32 + dmv_row as i32).clamp(-32768, 32767);
                let sum_c = (best.col as i32 + dmv_col as i32).clamp(-32768, 32767);
                let clamped = clamp_mv_pair(
                    Mv::new(sum_r as i16, sum_c as i16),
                    (BORDERINPIXELS - INTERP_EXTEND) << 3,
                    geom,
                );
                Ok((clamped.row, clamped.col))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_intra_block(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
        tx_size_log2: usize,
        skip: bool,
        segment_id: u8,
    ) -> Result<()> {
        // §9.3.2: `y_mode_probs[BlockSizeGroup][INTRA_MODES-1]` where
        // BlockSizeGroup is derived from MiSize (0: ≤8×8, 1: ≤16×16,
        // 2: ≤32×32, 3: 64×64). For intra-in-inter we map it off the
        // picked BlockSize.
        let bsg = block_size_group(bs);
        let y_probs = self.ch.ctx.y_mode_probs[bsg];
        let y_mode = read_intra_mode_tree(bd, &y_probs)?;
        // UV-mode conditioned on the just-chosen Y mode (INTRA_MODES=10).
        let uv_probs = self.ch.ctx.uv_mode_probs[y_mode as usize];
        let uv_mode = read_intra_mode_tree(bd, &uv_probs)?;
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
                segment_id,
            },
        );
        // Populate the MV-ref grid too so downstream inter blocks see
        // this cell as an intra neighbour (§6.5 contextCounter += 9).
        // `ref_frame[0] = INTRA_FRAME`, MVs zero, y_mode carries the
        // chosen intra mode (0..=9).
        let mut intra_cell = InterMiCell::default();
        intra_cell.ref_frame[0] = INTRA_FRAME;
        intra_cell.ref_frame[1] = NONE_FRAME;
        intra_cell.mv[0] = Mv::ZERO;
        intra_cell.mv[1] = Mv::ZERO;
        intra_cell.y_mode = y_mode as u8; // DC_PRED..TM_PRED = 0..9
        self.mv_grid.fill(
            mi_row_units,
            mi_col_units,
            mi_w as usize,
            mi_h as usize,
            intra_cell,
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
            segment_id,
        )?;
        let sub_x = self.hdr.color_config.subsampling_x as u32;
        let sub_y = self.hdr.color_config.subsampling_y as u32;
        let c_row = (row >> sub_y) as usize;
        let c_col = (col >> sub_x) as usize;
        let c_w = (bs.w() >> sub_x) as usize;
        let c_h = (bs.h() >> sub_y) as usize;
        let c_tx = clamp_tx_size(tx_size_log2, c_w, c_h);
        if c_w >= 4 && c_h >= 4 {
            self.recon_intra_plane(
                bd, c_row, c_col, c_w, c_h, c_tx, uv_mode, 1, skip, segment_id,
            )?;
            self.recon_intra_plane(
                bd, c_row, c_col, c_w, c_h, c_tx, uv_mode, 2, skip, segment_id,
            )?;
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

    /// Decode the residual for one inter block, returning the
    /// `EobTotal` accumulated across all planes (§6.4.4). The caller
    /// uses this to apply the §6.4.4 "skip = 1 when EobTotal == 0" rule
    /// before writing `Skips[][]`.
    #[allow(clippy::too_many_arguments)]
    fn add_residual(
        &mut self,
        bd: &mut BoolDecoder<'_>,
        row: u32,
        col: u32,
        bs: BlockSize,
        tx_size_log2: usize,
        tx_type: TxType,
        segment_id: u8,
    ) -> Result<u32> {
        let w = bs.w() as usize;
        let h = bs.h() as usize;
        // Luma.
        let mut eob_total = self.decode_plane_residual(
            bd,
            row as usize,
            col as usize,
            w,
            h,
            tx_size_log2,
            tx_type,
            0,
            segment_id,
        )?;
        let sub_x = self.hdr.color_config.subsampling_x as u32;
        let sub_y = self.hdr.color_config.subsampling_y as u32;
        let c_row = (row >> sub_y) as usize;
        let c_col = (col >> sub_x) as usize;
        let c_w = (bs.w() >> sub_x) as usize;
        let c_h = (bs.h() >> sub_y) as usize;
        let c_tx = clamp_tx_size(tx_size_log2, c_w, c_h);
        if c_w >= 4 && c_h >= 4 {
            eob_total += self.decode_plane_residual(
                bd,
                c_row,
                c_col,
                c_w,
                c_h,
                c_tx,
                TxType::DctDct,
                1,
                segment_id,
            )?;
            eob_total += self.decode_plane_residual(
                bd,
                c_row,
                c_col,
                c_w,
                c_h,
                c_tx,
                TxType::DctDct,
                2,
                segment_id,
            )?;
        }
        Ok(eob_total)
    }

    /// Returns the accumulated `EobTotal` (sum of EOBs across all
    /// transform blocks in this plane). Used by `add_residual` to
    /// implement the §6.4.4 skip override.
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
        segment_id: u8,
    ) -> Result<u32> {
        let tx_side = 4usize << tx_size_log2;
        let plane_type = if plane == 0 { 0 } else { 1 };
        let (plane_w, plane_h) = if plane == 0 {
            (self.width, self.height)
        } else {
            (self.uv_w, self.uv_h)
        };
        let scan = get_scan(tx_size_log2);
        // Inter block: ref_type = 1.
        let probs = coef_probs_from_ctx(self.ch, tx_size_log2, plane_type, 1);

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

        let mut r = 0usize;
        let mut eob_total: u32 = 0;
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
                // §6.4.24 initial token context from AboveNonzeroContext /
                // LeftNonzeroContext (§6.4.22).
                let initial_ctx =
                    self.nonzero_ctx
                        .token_ctx(plane, abs_col, abs_row, scan.tx_size_log2);
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
                    let mut dst = vec![0u8; tx_side * tx_side];
                    self.read_plane(plane, abs_row, abs_col, tx_w, tx_h, &mut dst, tx_side);
                    inverse_transform_add(tx_type, tx_side, tx_side, &coeffs, &mut dst, tx_side)?;
                    self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &dst, tx_side);
                }
                eob_total += eob as u32;
                // §6.4.22 post-tokens nonzero update.
                self.nonzero_ctx.update(
                    plane,
                    abs_col,
                    abs_row,
                    scan.tx_size_log2,
                    if eob > 0 { 1 } else { 0 },
                );
                c += tx_side;
            }
            r += tx_side;
        }
        Ok(eob_total)
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
        segment_id: u8,
    ) -> Result<()> {
        let tx_side = 4usize << tx_size_log2;
        let plane_type = if plane == 0 { 0 } else { 1 };
        let (plane_w, plane_h) = if plane == 0 {
            (self.width, self.height)
        } else {
            (self.uv_w, self.uv_h)
        };
        let scan = get_scan(tx_size_log2);
        // Intra block inside an inter frame: ref_type = 0.
        let probs = coef_probs_from_ctx(self.ch, tx_size_log2, plane_type, 0);

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
                let not_on_right = (c + tx_side) < w;
                let nb = self.build_neighbours(
                    plane,
                    abs_row,
                    abs_col,
                    tx_side,
                    tx_size_log2,
                    not_on_right,
                );
                let mut pred = vec![0u8; tx_side * tx_side];
                predict_intra(mode, &nb, &mut pred, tx_side);
                self.blit_plane(plane, abs_row, abs_col, tx_w, tx_h, &pred, tx_side);
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
                    if eob > 0 {
                        1u8
                    } else {
                        0u8
                    }
                } else {
                    0u8
                };
                // §6.4.22 post-tokens update.
                self.nonzero_ctx
                    .update(plane, abs_col, abs_row, scan.tx_size_log2, nonzero_update);
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
        tx_size_log2: usize,
        not_on_right: bool,
    ) -> NeighbourBuf {
        let (buf, stride, plane_w, plane_h) = match plane {
            0 => (&self.y[..], self.y_stride, self.width, self.height),
            1 => (&self.u[..], self.uv_stride, self.uv_w, self.uv_h),
            _ => (&self.v[..], self.uv_stride, self.uv_w, self.uv_h),
        };
        let have_above = row > 0;
        let have_left = col > 0;
        // §8.5.1: aboveRow extension (positions size..2*size-1) only
        // when `haveAbove && notOnRight && txSz == TX_4X4`.
        let have_aboveright =
            row > 0 && tx_size_log2 == 0 && not_on_right && col + 2 * tx_side <= plane_w;
        let mut above_tmp = vec![0u8; 2 * tx_side];
        let mut above_opt: Option<&[u8]> = None;
        if have_above {
            let start = (row - 1) * stride + col;
            let n_target = if have_aboveright {
                2 * tx_side
            } else {
                tx_side
            };
            let n = n_target.min(plane_w.saturating_sub(col));
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

#[allow(dead_code)]
fn coef_probs_for(tx_size_log2: usize, plane_type: usize) -> &'static [[[u8; 3]; 6]; 6] {
    match tx_size_log2 {
        0 => &COEF_PROBS_4X4[plane_type][0],
        1 => &COEF_PROBS_8X8[plane_type][0],
        2 => &COEF_PROBS_16X16[plane_type][0],
        3 => &COEF_PROBS_32X32[plane_type][0],
        _ => &COEF_PROBS_4X4[0][0],
    }
}

/// Borrow the per-frame coefficient probabilities for a given
/// tx_size / plane_type / ref_type. Mirrors `block::coef_probs_from_ctx`
/// but lives here so inter blocks don't depend on `block`.
fn coef_probs_from_ctx(
    ch: &CompressedHeader,
    tx_size_log2: usize,
    plane_type: usize,
    ref_type: usize,
) -> &[[[u8; 3]; 6]; 6] {
    let ts = tx_size_log2.min(3);
    &ch.ctx.coef_probs[ts][plane_type][ref_type]
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

/// Convert the spec-faithful `frame_ctx::MvComponentProbs` into the
/// (simplified) `mv::MvComponentProbs` the existing MV decoder consumes.
///
/// The existing decoder uses a single 3-prob `class0_fr` row instead of
/// the spec's `[CLASS0_SIZE=2][MV_FR_SIZE-1=3]` table — we map the
/// `d=0` row through. `classes` has 10 entries in both layouts
/// (`MV_CLASSES - 1 = 10`). `bits` is 10 entries in both
/// (`MV_OFFSET_BITS = 10`).
fn mv_component_probs_from_ctx(src: &crate::frame_ctx::MvComponentProbs) -> MvComponentProbs {
    MvComponentProbs {
        sign: src.sign,
        classes: src.classes,
        class0_bit: src.class0_bit,
        class0_fr: src.class0_fr[0],
        class0_hp: src.class0_hp,
        bits: src.bits,
        fr: src.fr,
        hp: src.hp,
    }
}

/// §9.3.2 `BlockSizeGroup` lookup — used to index `y_mode_probs`.
/// Groups 0..=3 cover: {8x8, 16x8, 8x16}, {16x16, 32x16, 16x32},
/// {32x32, 64x32, 32x64}, {64x64}.
fn block_size_group(bs: BlockSize) -> usize {
    match bs {
        BlockSize::B4x4
        | BlockSize::B4x8
        | BlockSize::B8x4
        | BlockSize::B8x8
        | BlockSize::B8x16
        | BlockSize::B16x8 => 0,
        BlockSize::B16x16 | BlockSize::B16x32 | BlockSize::B32x16 => 1,
        BlockSize::B32x32 | BlockSize::B32x64 | BlockSize::B64x32 => 2,
        BlockSize::B64x64 => 3,
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

    fn ni(avail_u: bool, avail_l: bool, above_ref: [u8; 2], left_ref: [u8; 2]) -> NeighbourInfo {
        let above_intra = above_ref[0] == INTRA_FRAME;
        let left_intra = left_ref[0] == INTRA_FRAME;
        let above_single = above_ref[1] == NONE_FRAME || above_ref[1] == INTRA_FRAME;
        let left_single = left_ref[1] == NONE_FRAME || left_ref[1] == INTRA_FRAME;
        NeighbourInfo {
            avail_u,
            avail_l,
            above_ref,
            left_ref,
            above_intra,
            left_intra,
            above_single,
            left_single,
        }
    }

    #[test]
    fn comp_mode_ctx_no_neighbours_is_one() {
        // §9.3.2 comp_mode: !AvailU && !AvailL → ctx = 1.
        let n = ni(false, false, [0, 255], [0, 255]);
        assert_eq!(InterTile::comp_mode_ctx(&n, 3), 1);
    }

    #[test]
    fn comp_mode_ctx_both_single_xor() {
        // Both single, neither is CompFixedRef → 0 ^ 0 = 0.
        let n = ni(true, true, [1, 255], [2, 255]);
        assert_eq!(InterTile::comp_mode_ctx(&n, 3), 0);
        // Both single, both are CompFixedRef → 1 ^ 1 = 0.
        let n = ni(true, true, [3, 255], [3, 255]);
        assert_eq!(InterTile::comp_mode_ctx(&n, 3), 0);
        // One match, one not → 1 ^ 0 = 1.
        let n = ni(true, true, [3, 255], [1, 255]);
        assert_eq!(InterTile::comp_mode_ctx(&n, 3), 1);
    }

    #[test]
    fn comp_mode_ctx_both_compound_is_four() {
        // Both compound (slot[1] non-NONE inter ref) → ctx = 4.
        let n = ni(true, true, [1, 3], [2, 3]);
        assert_eq!(InterTile::comp_mode_ctx(&n, 3), 4);
    }

    #[test]
    fn single_ref_p1_ctx_no_neighbours_is_two() {
        let n = ni(false, false, [0, 255], [0, 255]);
        assert_eq!(InterTile::single_ref_p1_ctx(&n), 2);
    }

    #[test]
    fn single_ref_p1_ctx_both_intra_is_two() {
        let n = ni(true, true, [0, 255], [0, 255]);
        assert_eq!(InterTile::single_ref_p1_ctx(&n), 2);
    }

    #[test]
    fn single_ref_p1_ctx_both_single_last() {
        // Both single, both LAST → 2*1 + 2*1 = 4.
        let n = ni(true, true, [1, 255], [1, 255]);
        assert_eq!(InterTile::single_ref_p1_ctx(&n), 4);
    }

    #[test]
    fn single_ref_p2_ctx_no_neighbours_is_two() {
        let n = ni(false, false, [0, 255], [0, 255]);
        assert_eq!(InterTile::single_ref_p2_ctx(&n), 2);
    }

    #[test]
    fn single_ref_p2_ctx_both_single_last_is_three() {
        // Both single, both LAST → ctx = 3.
        let n = ni(true, true, [1, 255], [1, 255]);
        assert_eq!(InterTile::single_ref_p2_ctx(&n), 3);
    }

    #[test]
    fn comp_ref_ctx_no_neighbours_is_two() {
        let n = ni(false, false, [0, 255], [0, 255]);
        assert_eq!(InterTile::comp_ref_ctx(&n, &[false; 4], 3, [1, 2]), 2);
    }

    #[test]
    fn comp_ref_ctx_both_intra_is_two() {
        let n = ni(true, true, [0, 255], [0, 255]);
        assert_eq!(InterTile::comp_ref_ctx(&n, &[false; 4], 3, [1, 2]), 2);
    }
}
