//! VP9 P-frame (inter) tile encoder — round 49 + r-next sub-pel ME + r-next quadtree partitions
//! + r-next-8x8 16×16 HORZ/VERT shapes + 8×8 leaf.
//!
//! Emits a non-keyframe VP9 tile payload that the in-tree decoder
//! reconstructs into motion-compensated pixel output. Scope:
//!
//! * Single-reference inter (LAST_FRAME = slot 0) — no compound, no
//!   GOLDEN, no ALTREF (round 49 deferral).
//! * Quadtree partitions per §6.4.16 / §6.5: each 64×64 SB may emit
//!   `PARTITION_NONE` (one 64×64 block) or `PARTITION_SPLIT` into
//!   four 32×32 sub-blocks. Each 32×32 may further emit
//!   `PARTITION_NONE` or `PARTITION_SPLIT` into four 16×16 sub-blocks.
//!   At 16×16 we evaluate all of `{NONE, HORZ (16×8 + 16×8), VERT
//!   (8×16 + 8×16), SPLIT (4 × 8×8)}` and pick by RDO. 8×8 emits
//!   `PARTITION_NONE` as a leaf — for inter, 8×8 is the minimum
//!   block size; sub-8×8 (B4x4 / B4x8 / B8x4) is intra-only territory
//!   per §6.5.18 and not emitted by this encoder. The split decision is
//!   RDO-shaped: the encoder runs ME at the parent block size,
//!   computes the SAD at the picked MV, then runs ME at each
//!   candidate sub-block, sums their SADs, adds a per-sub-block
//!   bit-rate penalty, and picks the lower-cost shape. For edge SBs
//!   we recurse the keyframe-style partition splits so
//!   non-multiple-of-64 frames still produce a valid tile.
//! * Per-SB three-stage motion estimation against the reconstructed
//!   LAST_FRAME plane:
//!     1. Integer-pel ±16 px full-search, SAD cost.
//!     2. 8-neighbour HALF-PEL refinement around the integer-pel
//!        best — VP9 §6.3 8-tap EightTap luma filter
//!        (`mcfilter::FILTER_EIGHTTAP`, §8.5.4.2) interpolates the
//!        reference block at each candidate; SAD picks the lowest.
//!     3. 8-neighbour QUARTER-PEL refinement around the half-pel best.
//!
//!   1/8-pel (`allow_high_precision_mv`) is deferred — r-next stops at
//!   1/4-pel which is the standard VP9 baseline precision.
//! * Two inter modes: `ZEROMV` (best MV = (0,0)) or `NEWMV` (any
//!   other sub-pel MV). `NEARESTMV` / `NEARMV` not emitted —
//!   round 49 doesn't track BestMv per spec, so emitting those would
//!   risk mismatch with the decoder's `find_best_ref_mvs` result.
//! * `skip = 1` everywhere — no residual encoding. PSNR comes
//!   entirely from MC quality. Translation fixtures with sub-pel
//!   alignment reconstruct via the 8-tap filter modulo §8.8
//!   loop-filter smoothing at SB boundaries.
//! * `tx_mode = ONLY_4X4` — `read_tx_size` returns 0 bits regardless.
//! * `interpolation_filter = 0` (EightTap) frame-level fixed — no
//!   per-block switchable-filter bits.
//! * `allow_high_precision_mv = false` — `hp` bit elided.
//!
//! Block emit order per §6.4.11 / §6.4.13 / §6.4.16:
//!   1. partition bit(s)
//!   2. `inter_segment_id` — 0 bits (segmentation disabled)
//!   3. `skip` — 1 bit against `skip_probs[skip_ctx]`
//!   4. `is_inter` — 1 bit against `is_inter_prob[ctx]` (always 1)
//!   5. `read_tx_size` — 0 bits in ONLY_4X4
//!   6. `comp_mode` — 0 bits (frame_reference_mode = SingleReference)
//!   7. `single_ref_prob[p1][0]` — 1 bit (0 → LAST_FRAME)
//!   8. interp_filter — 0 bits (non-switchable)
//!   9. `inter_mode_probs[ctx]` — 1 bit (ZEROMV) or 3 bits (NEWMV)
//!  10. for NEWMV: MV joint + per-component class/bits/fr.
//!
//! For `skip = 1` the decoder short-circuits the residual loop —
//! we emit no coefficients.

use crate::compressed_header::TxMode;
use crate::encoder::bool_encoder::BoolEncoder;
use crate::encoder::params::{EncoderParams, ReferenceFrame};
use crate::frame_ctx::FrameContext;
use crate::mcfilter::{mc_block, InterpFilter, RefSampler};
use crate::mv::{DEFAULT_MV_COMP_PROBS, MV_JOINT_PROBS};
use crate::mvref::{
    find_best_ref_mvs, find_mv_refs_geom, BlockGeom, InterMiCell, InterMiGrid, Y_MODE_NEWMV,
    Y_MODE_ZEROMV,
};
use crate::probs::PARTITION_PROBS;

/// §10.5 default skip_probs[3] — also the encoder's emit-time source.
const SKIP_PROBS: [u8; 3] = [192, 128, 64];

/// §10.5 default is_inter_prob[4].
const IS_INTER_PROB: [u8; 4] = [9, 102, 187, 225];

/// §10.5 default single_ref_prob[5][2] — second column unused (only
/// LAST is emitted).
const SINGLE_REF_PROB: [[u8; 2]; 5] = [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]];

/// §10.5 default inter_mode_probs[7][3].
const INTER_MODE_PROBS: [[u8; 3]; 7] = [
    [2, 173, 34],
    [7, 145, 85],
    [7, 166, 63],
    [7, 94, 66],
    [8, 64, 46],
    [17, 81, 31],
    [25, 29, 30],
];

/// Block-matching ME search radius (integer-pel).
const ME_SEARCH_RADIUS: i32 = 16;

/// SAD threshold below which a non-zero MV win is considered
/// "significant" — used by the encoder to decide ZEROMV vs NEWMV. If
/// the SAD with the best MV is more than this much smaller than the
/// SAD at MV=(0,0), pick NEWMV; otherwise ZEROMV. Saves the MV bits
/// when the source is approximately stationary.
const ME_NEWMV_GATE_SAD: u32 = 64;

/// Approximate bit-rate cost of one PARTITION_SPLIT decision plus the
/// extra per-sub-block symbols (skip / is_inter / ref / inter_mode /
/// MV joint + components — ~16 bits per sub-block on average vs the
/// ~10 bits a single block costs). The SPLIT path emits 3 extra
/// partition bits (the four bits to read SPLIT vs NONE) plus 4×
/// per-block overhead vs the parent's 1× overhead. SAD units are 8-bit
/// luma absolute differences; with eff_w*eff_h samples per block at
/// 8-bit precision, the SAD ranges over 0..=255*samples. A per-bit
/// cost of `λ * samples` where `λ ≈ 0.5` lets the cost trade real SAD
/// gain against the partition + sub-block bit budget. Keep low so
/// splits are picked when sub-block ME genuinely reduces SAD; raise
/// to discourage gratuitous splits on smooth content.
const SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS: u32 = 20;

/// Emit a complete P-frame tile payload using single-reference LAST.
/// Returns the raw bool-coded tile bytes; the caller assembles the
/// frame by prepending uncompressed + compressed headers.
pub fn emit_pframe_tile(
    p: &EncoderParams,
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
) -> Vec<u8> {
    let width = p.width as usize;
    let height = p.height as usize;
    let mut be = BoolEncoder::new();

    let mi_cols = width.div_ceil(8);
    let mi_rows = height.div_ceil(8);
    let mut ctx = InterCtx {
        partition_above: vec![0u8; mi_cols],
        partition_left: vec![0u8; mi_rows],
        skip_above: vec![false; mi_cols],
        skip_left: vec![false; mi_rows],
        intra_above: vec![false; mi_cols],
        intra_left: vec![false; mi_rows],
        mv_grid: InterMiGrid::new(mi_cols, mi_rows),
        mi_cols,
        mi_rows,
        debug_force_16x16_only: p.debug_force_16x16_only,
    };

    let sb_cols = p.width.div_ceil(64);
    let sb_rows = p.height.div_ceil(64);
    for sby in 0..sb_rows {
        for sbx in 0..sb_cols {
            let col = sbx * 64;
            let row = sby * 64;
            emit_inter_partition(
                &mut be, &mut ctx, row, col, 64, p.width, p.height, src, refr,
            );
        }
    }

    be.finish()
}

/// Partition / skip / intra context state — mirror of the decoder's
/// `InterTile` trackers.
struct InterCtx {
    partition_above: Vec<u8>,
    partition_left: Vec<u8>,
    skip_above: Vec<bool>,
    skip_left: Vec<bool>,
    intra_above: Vec<bool>,
    intra_left: Vec<bool>,
    mv_grid: InterMiGrid,
    mi_cols: usize,
    mi_rows: usize,
    /// Mirror of `EncoderParams::debug_force_16x16_only` — when true,
    /// the 16×16 partition picker short-circuits to PARTITION_NONE so
    /// regression tests can compare full-shape RDO vs 16×16-NONE-only
    /// reconstruction on the same wire/loop-filter path.
    debug_force_16x16_only: bool,
}

impl InterCtx {
    fn partition_probs(&self, bsize: u32, mi_row: usize, mi_col: usize) -> [u8; 3] {
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
            if c < self.partition_above.len() {
                above |= self.partition_above[c];
            }
            let r = mi_row + i;
            if r < self.partition_left.len() {
                left |= self.partition_left[r];
            }
        }
        let above_bit = ((above >> boffset) & 1) as usize;
        let left_bit = ((left >> boffset) & 1) as usize;
        // Non-key partition_probs are indexed by `tbl_bsl = 3 - bsl`
        // (see InterTile::read_partition).
        let tbl_bsl = 3 - bsl;
        let ctx_idx = tbl_bsl * 4 + left_bit * 2 + above_bit;
        PARTITION_PROBS[ctx_idx]
    }

    fn skip_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let a = if mi_row > 0 && mi_col < self.skip_above.len() {
            self.skip_above[mi_col] as usize
        } else {
            0
        };
        let l = if mi_col > 0 && mi_row < self.skip_left.len() {
            self.skip_left[mi_row] as usize
        } else {
            0
        };
        (a + l).min(2)
    }

    fn is_inter_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let avail_u = mi_row > 0;
        let avail_l = mi_col > 0;
        let above_intra = avail_u && mi_col < self.intra_above.len() && self.intra_above[mi_col];
        let left_intra = avail_l && mi_row < self.intra_left.len() && self.intra_left[mi_row];
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

    fn update_partition(
        &mut self,
        bsize_px: u32,
        sub_w_px: u32,
        sub_h_px: u32,
        mi_row: usize,
        mi_col: usize,
    ) {
        let num8x8 = (bsize_px as usize) / 8;
        let b_w_log2 = match sub_w_px {
            4 => 0u8,
            8 => 1,
            16 => 2,
            32 => 3,
            64 => 4,
            _ => 4,
        };
        let b_h_log2 = match sub_h_px {
            4 => 0u8,
            8 => 1,
            16 => 2,
            32 => 3,
            64 => 4,
            _ => 4,
        };
        let above_fill = 15u8 >> b_w_log2;
        let left_fill = 15u8 >> b_h_log2;
        for i in 0..num8x8 {
            let c = mi_col + i;
            if c < self.partition_above.len() {
                self.partition_above[c] = above_fill;
            }
            let r = mi_row + i;
            if r < self.partition_left.len() {
                self.partition_left[r] = left_fill;
            }
        }
    }

    fn stamp_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        mi_w: usize,
        mi_h: usize,
        skip: bool,
        mv: (i16, i16),
        is_zeromv: bool,
    ) {
        for i in 0..mi_w.max(1) {
            let c = mi_col + i;
            if c < self.skip_above.len() {
                self.skip_above[c] = skip;
                self.intra_above[c] = false;
            }
        }
        for i in 0..mi_h.max(1) {
            let r = mi_row + i;
            if r < self.skip_left.len() {
                self.skip_left[r] = skip;
                self.intra_left[r] = false;
            }
        }
        // mv_grid fill — record cell with ref_frame=LAST=1, MV.
        let mut cell = InterMiCell::default();
        cell.ref_frame[0] = 1; // LAST_FRAME
        cell.ref_frame[1] = 255; // NONE
        cell.mv[0] = crate::mv::Mv::new(mv.0, mv.1);
        cell.sub_mvs[0] = [crate::mv::Mv::new(mv.0, mv.1); 4];
        cell.y_mode = if is_zeromv {
            Y_MODE_ZEROMV
        } else {
            Y_MODE_NEWMV
        };
        cell.interp_filter = 0; // EightTap
        self.mv_grid
            .fill(mi_row, mi_col, mi_w.max(1), mi_h.max(1), cell);
    }
}

/// Recursive partition emitter — mirrors `encoder/tile.rs::emit_partition`
/// for the keyframe path but with r-next RDO between PARTITION_NONE /
/// PARTITION_HORZ / PARTITION_VERT / PARTITION_SPLIT for interior blocks.
///
/// Decision shape at an interior bsize:
///   * bsize = 64 / 32: NONE vs SPLIT RDO. (HORZ / VERT would still be
///     valid per spec but the rectangular shapes at these sizes carry
///     little marginal SAD gain on smooth + naturally-textured content
///     vs the simpler SPLIT path that recurses one level deeper. Saved
///     for a later round.)
///   * bsize = 16: full {NONE, HORZ (16×8 + 16×8), VERT (8×16 + 8×16),
///     SPLIT (4 × 8×8)} RDO. Wire bits per §6.4.2 partition tree:
///     NONE  → "0", HORZ  → "1, 0", VERT  → "1, 1, 0", SPLIT → "1, 1, 1".
///   * bsize = 8: always emit PARTITION_NONE (one 8×8 block). For inter,
///     8×8 is the spec minimum; sub-8×8 (B4x4 / B4x8 / B8x4) lives behind
///     the §6.5.18 subsize syntax which this encoder doesn't yet emit.
#[allow(clippy::too_many_arguments)]
fn emit_inter_partition(
    be: &mut BoolEncoder,
    ctx: &mut InterCtx,
    row: u32,
    col: u32,
    bsize: u32,
    frame_w: u32,
    frame_h: u32,
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
) {
    if row >= frame_h || col >= frame_w {
        return;
    }
    let on_right = col + bsize > frame_w;
    let on_bottom = row + bsize > frame_h;
    let mi_row = (row as usize) / 8;
    let mi_col = (col as usize) / 8;
    let probs = ctx.partition_probs(bsize, mi_row, mi_col);
    let half = bsize / 2;

    if on_right && on_bottom {
        if bsize == 8 {
            // 8×8 at corner — emit as NONE.
            emit_inter_block(be, ctx, row, col, bsize, bsize, src, refr);
            ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        // SPLIT (forced, no bit read).
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refr,
        );
        return;
    }
    if on_right {
        // Only VERT or SPLIT readable per spec (single bit at probs[2]).
        // We pick SPLIT.
        be.write(1, probs[2]);
        if bsize == 8 {
            emit_inter_block(be, ctx, row, col, bsize, bsize, src, refr);
            ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refr,
        );
        return;
    }
    if on_bottom {
        be.write(1, probs[1]);
        if bsize == 8 {
            emit_inter_block(be, ctx, row, col, bsize, bsize, src, refr);
            ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refr,
        );
        return;
    }
    // Interior block.
    if bsize == 8 {
        // §6.4.2 still has a partition bit at bsize=8 (NONE vs HORZ vs
        // VERT vs SPLIT). For inter we always emit NONE — no sub-8×8
        // splits are produced.
        be.write(0, probs[0]);
        emit_inter_block(be, ctx, row, col, bsize, bsize, src, refr);
        ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
        return;
    }
    if bsize == 16 {
        let pick = if ctx.debug_force_16x16_only {
            Partition16::None
        } else {
            pick_partition_16(row, col, src, refr)
        };
        match pick {
            Partition16::None => {
                be.write(0, probs[0]);
                emit_inter_block(be, ctx, row, col, bsize, bsize, src, refr);
                ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            }
            Partition16::Horz => {
                // PARTITION_HORZ — wire "1, 0".
                be.write(1, probs[0]);
                be.write(0, probs[1]);
                emit_inter_block(be, ctx, row, col, bsize, half, src, refr);
                if row + half < frame_h {
                    emit_inter_block(be, ctx, row + half, col, bsize, half, src, refr);
                }
                ctx.update_partition(bsize, bsize, half, mi_row, mi_col);
            }
            Partition16::Vert => {
                // PARTITION_VERT — wire "1, 1, 0".
                be.write(1, probs[0]);
                be.write(1, probs[1]);
                be.write(0, probs[2]);
                emit_inter_block(be, ctx, row, col, half, bsize, src, refr);
                if col + half < frame_w {
                    emit_inter_block(be, ctx, row, col + half, half, bsize, src, refr);
                }
                ctx.update_partition(bsize, half, bsize, mi_row, mi_col);
            }
            Partition16::Split => {
                // PARTITION_SPLIT — wire "1, 1, 1", recurse to 8×8.
                be.write(1, probs[0]);
                be.write(1, probs[1]);
                be.write(1, probs[2]);
                emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refr);
                emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refr);
                emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refr);
                emit_inter_partition(
                    be,
                    ctx,
                    row + half,
                    col + half,
                    half,
                    frame_w,
                    frame_h,
                    src,
                    refr,
                );
            }
        }
        return;
    }
    // bsize ∈ {64, 32} — NONE vs SPLIT RDO.
    if should_split(row, col, bsize, src, refr) {
        // PARTITION_SPLIT — wire bits "1, 1, 1" per §6.4.2 partition
        // tree (`read_partition_from_tree`):
        //   bit0 = 1 (not NONE), bit1 = 1 (not HORZ), bit2 = 1 (SPLIT).
        be.write(1, probs[0]);
        be.write(1, probs[1]);
        be.write(1, probs[2]);
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refr);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refr);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refr,
        );
        return;
    }
    // Interior PARTITION_NONE — emit `bit=0` against probs[0].
    be.write(0, probs[0]);
    emit_inter_block(be, ctx, row, col, bsize, bsize, src, refr);
    ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
}

/// Outcome of the 16×16 partition RDO: one of the four §6.4.2 shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Partition16 {
    None,
    Horz,
    Vert,
    Split,
}

/// RDO across {NONE, HORZ, VERT, SPLIT} for a fully-interior 16×16
/// block. Runs `me_search` at each candidate shape (parent 16×16, the
/// two 16×8 halves, the two 8×16 halves, the four 8×8 sub-blocks),
/// sums SADs, adds a fixed bit-rate penalty proportional to the number
/// of additional inter blocks vs PARTITION_NONE, and picks the lowest
/// total.
///
/// Penalty unit `SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS`:
/// * NONE  — 0 extra inter blocks → 0 penalty.
/// * HORZ  — 1 extra inter block (2 blocks vs 1) → 1× penalty.
/// * VERT  — 1 extra inter block → 1× penalty.
/// * SPLIT — 3 extra inter blocks + partition + per-8×8 partition bits
///   → 3× penalty.
fn pick_partition_16(
    row: u32,
    col: u32,
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
) -> Partition16 {
    let bsize = 16u32;
    let half = bsize / 2;
    let parent_eff_w = (bsize as usize).min((src.width as usize).saturating_sub(col as usize));
    let parent_eff_h = (bsize as usize).min((src.height as usize).saturating_sub(row as usize));
    if parent_eff_w == 0 || parent_eff_h == 0 {
        return Partition16::None;
    }
    let parent = me_search(
        src,
        refr,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        parent_eff_h as i32,
    );
    let cost_none = parent.best_sad as u64;

    // HORZ — 16×8 top + 16×8 bottom.
    let top = me_search(
        src,
        refr,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        (half as i32).min(parent_eff_h as i32),
    );
    let bot_eff_h = (parent_eff_h as i32) - (half as i32);
    let bot_sad = if bot_eff_h > 0 {
        me_search(
            src,
            refr,
            (row + half) as i32,
            col as i32,
            parent_eff_w as i32,
            bot_eff_h,
        )
        .best_sad as u64
    } else {
        0
    };
    let cost_horz = top.best_sad as u64 + bot_sad + SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64;

    // VERT — 8×16 left + 8×16 right.
    let left = me_search(
        src,
        refr,
        row as i32,
        col as i32,
        (half as i32).min(parent_eff_w as i32),
        parent_eff_h as i32,
    );
    let right_eff_w = (parent_eff_w as i32) - (half as i32);
    let right_sad = if right_eff_w > 0 {
        me_search(
            src,
            refr,
            row as i32,
            (col + half) as i32,
            right_eff_w,
            parent_eff_h as i32,
        )
        .best_sad as u64
    } else {
        0
    };
    let cost_vert = left.best_sad as u64 + right_sad + SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64;

    // SPLIT — four 8×8 children.
    let mut child_sad_sum: u64 = 0;
    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
        let child_row = row + dr;
        let child_col = col + dc;
        let eff_w = (half as usize).min((src.width as usize).saturating_sub(child_col as usize));
        let eff_h = (half as usize).min((src.height as usize).saturating_sub(child_row as usize));
        if eff_w == 0 || eff_h == 0 {
            continue;
        }
        let child = me_search(
            src,
            refr,
            child_row as i32,
            child_col as i32,
            eff_w as i32,
            eff_h as i32,
        );
        child_sad_sum += child.best_sad as u64;
    }
    let cost_split = child_sad_sum + (SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64) * 3;

    // Tie-break: NONE wins ties (cheapest wire cost when SADs match).
    let mut best = Partition16::None;
    let mut best_cost = cost_none;
    if cost_horz < best_cost {
        best = Partition16::Horz;
        best_cost = cost_horz;
    }
    if cost_vert < best_cost {
        best = Partition16::Vert;
        best_cost = cost_vert;
    }
    if cost_split < best_cost {
        best = Partition16::Split;
    }
    best
}

/// RDO: should we PARTITION_SPLIT this interior bsize ∈ {64, 32} block?
///
/// Runs ME at the parent bsize and at each of the 4 half-sized children,
/// summing SADs. SPLIT wins if the sum of child SADs plus a fixed
/// `SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS * 3` bit-rate penalty
/// (3 = the 3 extra sub-blocks that wouldn't have been emitted as
/// PARTITION_NONE) is strictly less than the parent SAD.
fn should_split(
    row: u32,
    col: u32,
    bsize: u32,
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
) -> bool {
    let parent_eff_w = (bsize as usize).min((src.width as usize).saturating_sub(col as usize));
    let parent_eff_h = (bsize as usize).min((src.height as usize).saturating_sub(row as usize));
    if parent_eff_w == 0 || parent_eff_h == 0 {
        return false;
    }
    let parent = me_search(
        src,
        refr,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        parent_eff_h as i32,
    );
    let half = bsize / 2;
    // Sum child SADs. Skip out-of-frame children (they'd have been
    // handled by the on_right / on_bottom edge branch in the caller —
    // we never reach here unless the block is fully interior).
    let mut child_sad_sum: u64 = 0;
    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
        let child_row = row + dr;
        let child_col = col + dc;
        let eff_w = (half as usize).min((src.width as usize).saturating_sub(child_col as usize));
        let eff_h = (half as usize).min((src.height as usize).saturating_sub(child_row as usize));
        if eff_w == 0 || eff_h == 0 {
            continue;
        }
        let child = me_search(
            src,
            refr,
            child_row as i32,
            child_col as i32,
            eff_w as i32,
            eff_h as i32,
        );
        child_sad_sum += child.best_sad as u64;
    }
    // 3 extra sub-blocks beyond what PARTITION_NONE would emit.
    let split_penalty = (SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64) * 3;
    child_sad_sum + split_penalty < parent.best_sad as u64
}

/// Result of a single block-level motion estimation pass.
struct MeResult {
    /// Best MV in 1/8-pel units (`row`, `col`), aligned to 1/4-pel (the
    /// last bit is always 0 because `allow_high_precision_mv = false`).
    best_mv_1_8: (i32, i32),
    /// SAD at `best_mv_1_8` (sub-pel SAD via the EightTap luma filter
    /// when the MV has any sub-pel phase, otherwise integer-pel SAD).
    best_sad: u32,
    /// SAD at MV = (0, 0) — caller uses this to pick ZEROMV vs NEWMV.
    sad_zero: u32,
}

/// Three-stage motion search: integer-pel full search → half-pel
/// 8-neighbour refinement → quarter-pel 8-neighbour refinement.
/// Mirrors the body of `emit_inter_block` so the partition RDO and the
/// actual emit see identical SADs.
fn me_search(
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
    row: i32,
    col: i32,
    eff_w: i32,
    eff_h: i32,
) -> MeResult {
    let (int_mv_row, int_mv_col, _int_best_sad, sad_zero) =
        block_match_integer(src, refr, row, col, eff_w, eff_h);
    let mut best_mv_1_8 = (int_mv_row * 8, int_mv_col * 8);
    let mut best_sad = compute_sad_subpel(
        src,
        refr,
        row,
        col,
        eff_w,
        eff_h,
        best_mv_1_8.0,
        best_mv_1_8.1,
    );
    refine_subpel_8nb(
        src,
        refr,
        row,
        col,
        eff_w,
        eff_h,
        4,
        &mut best_mv_1_8,
        &mut best_sad,
    );
    refine_subpel_8nb(
        src,
        refr,
        row,
        col,
        eff_w,
        eff_h,
        2,
        &mut best_mv_1_8,
        &mut best_sad,
    );
    MeResult {
        best_mv_1_8,
        best_sad,
        sad_zero,
    }
}

/// Emit one inter block at (row, col, bsize_w_px × bsize_h_px). Performs
/// integer-pel ME against the LAST_FRAME reference, picks ZEROMV /
/// NEWMV, and writes the symbol sequence per §6.4.11 / §6.4.16.
///
/// `bsize_w_px` and `bsize_h_px` need not be equal — HORZ at 16×16 emits
/// two 16×8 blocks, VERT at 16×16 emits two 8×16 blocks. The decoder
/// consumes the same `decode_block(row, col, BlockSize::from_wh(w, h))`
/// for every rectangular shape we emit here.
#[allow(clippy::too_many_arguments)]
fn emit_inter_block(
    be: &mut BoolEncoder,
    ctx: &mut InterCtx,
    row: u32,
    col: u32,
    bsize_w_px: u32,
    bsize_h_px: u32,
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
) {
    let mi_row = (row as usize) / 8;
    let mi_col = (col as usize) / 8;
    let mi_w = ((bsize_w_px as usize) / 8).max(1);
    let mi_h = ((bsize_h_px as usize) / 8).max(1);

    // §6.4.8 read_skip — emit skip=1 unconditionally (round 49 simplification).
    let sctx = ctx.skip_ctx(mi_row, mi_col);
    be.write(1, SKIP_PROBS[sctx]);

    // §6.4.13 read_is_inter — emit 1 (inter).
    let ictx = ctx.is_inter_ctx(mi_row, mi_col);
    be.write(1, IS_INTER_PROB[ictx]);

    // §6.3.1 tx_mode=ONLY_4X4 → no tx_size bits.

    // §6.4.17 read_ref_frames — single ref (compoundReferenceAllowed=false
    // when sign_bias is uniform → no comp_mode bit). Emit p1=0 → LAST_FRAME.
    // The §9.3.2 single_ref_p1 context for a frame with all-LAST neighbours
    // collapses to a fixed value; we compute it via `single_ref_p1_ctx` over
    // the current neighbour info.
    let p1_ctx = single_ref_p1_ctx(ctx, mi_row, mi_col);
    be.write(0, SINGLE_REF_PROB[p1_ctx][0]);

    // Interpolation filter is non-switchable (frame-level EightTap), no bits.

    // §6.4.16 ME + inter_mode tree. Three-stage refinement on the luma plane.
    // Body lives in `me_search` so the partition-RDO and the emit path
    // see bit-identical SAD numbers.
    let eff_w = (bsize_w_px as usize).min((src.width as usize).saturating_sub(col as usize));
    let eff_h = (bsize_h_px as usize).min((src.height as usize).saturating_sub(row as usize));
    let me = me_search(
        src,
        refr,
        row as i32,
        col as i32,
        eff_w as i32,
        eff_h as i32,
    );
    let best_mv_1_8 = me.best_mv_1_8;
    let best_sad = me.best_sad;
    let sad_zero = me.sad_zero;

    let is_zeromv = best_sad + ME_NEWMV_GATE_SAD >= sad_zero;
    let (mv_row_1_8, mv_col_1_8) = if is_zeromv {
        (0i16, 0i16)
    } else {
        let r = best_mv_1_8.0.clamp(-32768, 32767) as i16;
        let c = best_mv_1_8.1.clamp(-32768, 32767) as i16;
        // Round to the smallest representable unit. Without
        // allow_high_precision_mv the encoder must emit MVs whose
        // 1/8-pel components are even (i.e. quarter-pel quantum). The
        // refinement loop already constrains to even values; assert
        // here.
        debug_assert_eq!(r & 1, 0, "low-precision MV row must be 1/4-pel-aligned");
        debug_assert_eq!(c & 1, 0, "low-precision MV col must be 1/4-pel-aligned");
        (r, c)
    };

    // §6.5.1 find_mv_refs + §6.5.12 find_best_ref_mvs — exactly what the
    // decoder will compute, so our NEWMV delta lines up.
    let mi_cols_i32 = ctx.mi_cols as i32;
    let mi_rows_i32 = ctx.mi_rows as i32;
    let geom = BlockGeom::from_pixels(row, col, bsize_w_px, bsize_h_px, mi_rows_i32, mi_cols_i32);
    let sign_bias: [bool; 4] = [false; 4];
    let bsize_code = block_size_code_for(bsize_w_px, bsize_h_px) as usize;
    let mut refs_a = find_mv_refs_geom(
        &ctx.mv_grid,
        &sign_bias,
        1,
        bsize_code,
        geom,
        0,
        mi_cols_i32,
    );
    let allow_hp = false;
    find_best_ref_mvs(&mut refs_a, allow_hp, &geom);
    let mode_ctx = (refs_a.mode_context as usize).min(6);

    // Emit inter_mode tree symbol.
    let probs = INTER_MODE_PROBS[mode_ctx];
    if is_zeromv {
        be.write(0, probs[0]); // ZEROMV
    } else {
        // NEWMV: emit bits "1, 1, 1".
        be.write(1, probs[0]);
        be.write(1, probs[1]);
        be.write(1, probs[2]);
        // Compute delta = mv - best_mv.
        let best = refs_a.best_mv();
        let dmv_r = (mv_row_1_8 as i32) - (best.row as i32);
        let dmv_c = (mv_col_1_8 as i32) - (best.col as i32);
        emit_mv(be, dmv_r, dmv_c, allow_hp);
    }

    // Skip=1 ⇒ no residual.

    // Update context trackers + mv_grid.
    ctx.stamp_block(
        mi_row,
        mi_col,
        mi_w,
        mi_h,
        true,
        (mv_row_1_8, mv_col_1_8),
        is_zeromv,
    );
}

/// §9.3.2 `single_ref_p1_ctx` — neighbour-aware context for the
/// first ref-frame bit. Round-49 simplification: we only ever emit
/// LAST as the primary, never write intra blocks inside a P-frame,
/// so neighbour state collapses. The full §9.3.2 derivation is
/// duplicated here so we agree with `InterTile::single_ref_p1_ctx`
/// on the wire.
fn single_ref_p1_ctx(ctx: &InterCtx, mi_row: usize, mi_col: usize) -> usize {
    let avail_u = mi_row > 0;
    let avail_l = mi_col > 0;
    let above_intra = avail_u && mi_col < ctx.intra_above.len() && ctx.intra_above[mi_col];
    let left_intra = avail_l && mi_row < ctx.intra_left.len() && ctx.intra_left[mi_row];
    let above_ref = if avail_u && !above_intra { 1u8 } else { 0u8 };
    let left_ref = if avail_l && !left_intra { 1u8 } else { 0u8 };
    // §9.3.2: ctx derived from above/left ref-frame & intra flags. With
    // all neighbours either unavailable or LAST inter blocks, the
    // resulting ctx is one of {0, 2, 3} — match `InterTile::single_ref_p1_ctx`
    // for the all-LAST case.
    if avail_u && avail_l {
        if above_intra && left_intra {
            2
        } else if above_intra || left_intra {
            // intra one side, LAST inter the other → ctx=1 if LAST, ctx=3 otherwise.
            if above_ref == 1 || left_ref == 1 {
                1
            } else {
                3
            }
        } else {
            // Both inter neighbours. Both LAST → ctx 0.
            // (Spec derivation: 2 * (above==LAST != left==LAST) +
            //  (above==LAST && left==LAST ? 0 : 1) — collapses to 0 in our case.)
            0
        }
    } else if avail_u || avail_l {
        let intra = if avail_u { above_intra } else { left_intra };
        let ref_is_last = if avail_u {
            above_ref == 1
        } else {
            left_ref == 1
        };
        if intra {
            2
        } else if ref_is_last {
            0
        } else {
            2
        }
    } else {
        2
    }
}

/// §3 Table 3-1 block_size_code lookup: returns the spec block-size
/// code for a rectangular `w × h` block. The inter encoder ships
/// 64/32/16/8 squares plus 16×8 + 8×16 (r-next-8x8 round). Sub-8×8
/// shapes are intra-only and not emitted.
fn block_size_code_for(w: u32, h: u32) -> u8 {
    // Spec BlockSize enum: B4x4=0 ... B64x64=12.
    match (w, h) {
        (8, 8) => 3,
        (8, 16) => 4,
        (16, 8) => 5,
        (16, 16) => 6,
        (16, 32) => 7,
        (32, 16) => 8,
        (32, 32) => 9,
        (32, 64) => 10,
        (64, 32) => 11,
        (64, 64) => 12,
        // Defaults for unexpected shapes — fall back to 64×64 (encoder
        // never reaches this path on validated inputs).
        _ => 12,
    }
}

/// §6.4.19 MV emission. `dmv_row` / `dmv_col` are deltas in 1/8-pel
/// units. Writes the joint + per-component class/bits/fr (and `hp`
/// when allowed).
fn emit_mv(be: &mut BoolEncoder, dmv_row: i32, dmv_col: i32, allow_hp: bool) {
    let joint = match (dmv_row != 0, dmv_col != 0) {
        (false, false) => 0u32,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    };
    // joint tree: 3 probs — `[not_zero, hzvz_vs_rest, hzvnz_vs_hnzvnz]`.
    match joint {
        0 => be.write(0, MV_JOINT_PROBS[0]),
        1 => {
            be.write(1, MV_JOINT_PROBS[0]);
            be.write(0, MV_JOINT_PROBS[1]);
        }
        2 => {
            be.write(1, MV_JOINT_PROBS[0]);
            be.write(1, MV_JOINT_PROBS[1]);
            be.write(0, MV_JOINT_PROBS[2]);
        }
        _ => {
            be.write(1, MV_JOINT_PROBS[0]);
            be.write(1, MV_JOINT_PROBS[1]);
            be.write(1, MV_JOINT_PROBS[2]);
        }
    }
    if dmv_row != 0 {
        emit_mv_component(be, dmv_row, allow_hp);
    }
    if dmv_col != 0 {
        emit_mv_component(be, dmv_col, allow_hp);
    }
}

/// Emit one MV component (signed 1/8-pel magnitude). Inverse of
/// `crate::mv::read_mv_component`.
fn emit_mv_component(be: &mut BoolEncoder, value: i32, allow_hp: bool) {
    let p = &DEFAULT_MV_COMP_PROBS;
    let sign = value < 0;
    let mag = value.unsigned_abs() as i32;
    // Sign bit.
    be.write(sign as u32, p.sign);
    // Determine class — inverse of `read_mv_class`.
    //   class 0: mag in [1, 16]
    //   class c (1..=10): mag in [(1<<c+3)+1, (1<<c+4)]
    let class = mv_class_of(mag);
    emit_mv_class_tree(be, &p.classes, class);

    if class == 0 {
        // mag - 1 = (d << 3) | (fr << 1) | hp  ∈  0..=15
        let body = (mag - 1) as u32;
        let d = (body >> 3) & 1;
        let fr = (body >> 1) & 3;
        let hp = body & 1;
        be.write(d, p.class0_bit);
        emit_mv_fr(be, &p.class0_fr, fr);
        if allow_hp {
            be.write(hp, p.class0_hp);
        }
    } else {
        // mag = (1 << (class+3)) + (d << 3) + (fr << 1) + hp + 1
        let base = 1i32 << (class + 3);
        let body = (mag - base - 1) as u32;
        let d = (body >> 3) & ((1u32 << class) - 1);
        let fr = (body >> 1) & 3;
        let hp = body & 1;
        for i in 0..class {
            let bit = (d >> i) & 1;
            be.write(bit, p.bits[i]);
        }
        emit_mv_fr(be, &p.fr, fr);
        if allow_hp {
            be.write(hp, p.hp);
        }
    }
}

/// Walk the `read_mv_class` decision tree to emit `class` ∈ 0..=10.
fn emit_mv_class_tree(be: &mut BoolEncoder, probs: &[u8; 10], class: usize) {
    debug_assert!(class <= 10);
    // libvpx vp9_mv_class_tree: linear cascade through probs[0..=9].
    for i in 0..=9.min(class) {
        if i == class {
            // For the leaf at `class == i`, emit 0 to terminate.
            // EXCEPT when `class == 10` we reach the bottom (probs[9]
            // and need to emit 1 to pick 10 vs 9).
            be.write(0, probs[i]);
            return;
        }
        be.write(1, probs[i]);
    }
    // If we fell through (class == 10), no extra symbol — the last
    // loop iteration emitted `1` against probs[9].
}

/// §6.4.19 fr tree (3 probs).
fn emit_mv_fr(be: &mut BoolEncoder, probs: &[u8; 3], fr: u32) {
    match fr {
        0 => be.write(0, probs[0]),
        1 => {
            be.write(1, probs[0]);
            be.write(0, probs[1]);
        }
        2 => {
            be.write(1, probs[0]);
            be.write(1, probs[1]);
            be.write(0, probs[2]);
        }
        _ => {
            be.write(1, probs[0]);
            be.write(1, probs[1]);
            be.write(1, probs[2]);
        }
    }
}

/// Compute the §6.4.19 mv_class for a magnitude. Mirrors the decoder
/// reconstruction in `read_mv_component`.
fn mv_class_of(mag: i32) -> usize {
    debug_assert!(mag > 0);
    // class 0: mag in 1..=16
    if mag <= 16 {
        return 0;
    }
    // class c >= 1: mag in (1<<c+3)+1 ..= (1<<c+4)
    for c in 1..=10 {
        if mag <= (1i32 << (c + 4)) {
            return c;
        }
    }
    10
}

/// Integer-pel block matching against the reference luma plane.
/// Returns `(best_mv_row, best_mv_col, best_sad, sad_at_zero)` —
/// caller compares `best_sad` against `sad_at_zero` to decide
/// ZEROMV vs NEWMV.
///
/// `bs_w` / `bs_h` are the EFFECTIVE width/height of the block (clamped
/// to frame edge for non-multiple-of-64 cases). MVs are searched in a
/// ±`ME_SEARCH_RADIUS` square window; out-of-bounds reference samples
/// are edge-clamped (matching the decoder's RefFrame::sample_y).
fn block_match_integer(
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
    row: i32,
    col: i32,
    bs_w: i32,
    bs_h: i32,
) -> (i32, i32, u32, u32) {
    let mut best_sad = u32::MAX;
    let mut best_mv = (0i32, 0i32);
    let mut sad_at_zero = u32::MAX;

    for dr in -ME_SEARCH_RADIUS..=ME_SEARCH_RADIUS {
        for dc in -ME_SEARCH_RADIUS..=ME_SEARCH_RADIUS {
            let sad = compute_sad(src, refr, row, col, bs_w, bs_h, dr, dc);
            if dr == 0 && dc == 0 {
                sad_at_zero = sad;
            }
            if sad < best_sad {
                best_sad = sad;
                best_mv = (dr, dc);
            }
        }
    }

    (best_mv.0, best_mv.1, best_sad, sad_at_zero)
}

/// SAD between a `bs_w × bs_h` source patch at (row, col) and the
/// reference patch at (row+dr, col+dc) — with edge-clamp on the
/// reference and source.
#[allow(clippy::too_many_arguments)]
fn compute_sad(
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
    row: i32,
    col: i32,
    bs_w: i32,
    bs_h: i32,
    dr: i32,
    dc: i32,
) -> u32 {
    let mut sad: u32 = 0;
    let src_w = src.width as i32;
    let src_h = src.height as i32;
    let ref_w = refr.width as i32;
    let ref_h = refr.height as i32;
    for r in 0..bs_h {
        for c in 0..bs_w {
            let sr = (row + r).clamp(0, src_h - 1) as usize;
            let sc = (col + c).clamp(0, src_w - 1) as usize;
            let s = src.y[sr * src.y_stride + sc] as i32;
            let rr = (row + r + dr).clamp(0, ref_h - 1) as usize;
            let rc = (col + c + dc).clamp(0, ref_w - 1) as usize;
            let p = refr.y[rr * refr.y_stride + rc] as i32;
            sad += (s - p).unsigned_abs();
        }
    }
    sad
}

/// Edge-clamped reference-luma sampler for `mc_block`. The encoder
/// keeps the reference plane in `ReferenceFrame::y` with `y_stride`;
/// out-of-bounds reads replicate the nearest sample (§8.5.4).
struct EncRefLumaSampler<'a> {
    refr: &'a ReferenceFrame,
}

impl<'a> RefSampler for EncRefLumaSampler<'a> {
    fn sample(&self, row: isize, col: isize) -> u8 {
        let r = row.clamp(0, self.refr.height as isize - 1) as usize;
        let c = col.clamp(0, self.refr.width as isize - 1) as usize;
        self.refr.y[r * self.refr.y_stride + c]
    }
}

/// SAD for a sub-pel MV `(mv_row_1_8, mv_col_1_8)` in 1/8-pel units.
///
/// Decomposes the 1/8-pel MV into:
///   * integer part = `mv >> 3` (sign-aware floored)
///   * sub-pel phase = `mv & 7` (0..=7), mapped onto the 16-phase
///     EightTap filter table as `phase * 2` (the existing 16-phase
///     bank covers 1/16-pel chroma; for 1/8-pel luma we use only the
///     even phases, matching §6.3 + §8.5.4.2 + the inter decoder).
///
/// Fast path: integer-pel positions (`sub == 0`) short-circuit to the
/// plain `compute_sad` for cheaper inner-loop cost.
#[allow(clippy::too_many_arguments)]
fn compute_sad_subpel(
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
    row: i32,
    col: i32,
    bs_w: i32,
    bs_h: i32,
    mv_row_1_8: i32,
    mv_col_1_8: i32,
) -> u32 {
    // Floor-divide-by-8 gives the integer-pel offset; the residual is
    // the sub-pel phase. We use `div_euclid` / `rem_euclid` so negative
    // MVs split into the expected (integer, non-negative phase) pair.
    let int_r = mv_row_1_8.div_euclid(8);
    let int_c = mv_col_1_8.div_euclid(8);
    let sub_r = mv_row_1_8.rem_euclid(8) as u32;
    let sub_c = mv_col_1_8.rem_euclid(8) as u32;

    if sub_r == 0 && sub_c == 0 {
        return compute_sad(src, refr, row, col, bs_w, bs_h, int_r, int_c);
    }

    // 8-tap EightTap luma filter. The 16-phase bank's even phases map
    // 1:1 onto VP9's 1/8-pel luma offsets (§6.3 sub_pel_filters_8).
    let bw = bs_w as usize;
    let bh = bs_h as usize;
    let mut interp = vec![0u8; bw * bh];
    let sampler = EncRefLumaSampler { refr };
    mc_block(
        &sampler,
        InterpFilter::EightTap,
        &mut interp,
        bw,
        bw,
        bh,
        (row + int_r) as isize,
        (col + int_c) as isize,
        sub_r * 2,
        sub_c * 2,
    );

    let mut sad: u32 = 0;
    let src_w = src.width as i32;
    let src_h = src.height as i32;
    for r in 0..bh {
        for c in 0..bw {
            let sr = (row + r as i32).clamp(0, src_h - 1) as usize;
            let sc = (col + c as i32).clamp(0, src_w - 1) as usize;
            let s = src.y[sr * src.y_stride + sc] as i32;
            let p = interp[r * bw + c] as i32;
            sad += (s - p).unsigned_abs();
        }
    }
    sad
}

/// 8-neighbour sub-pel refinement around `*best_mv_1_8` at `step`
/// granularity (in 1/8-pel units): `step == 4` ⇒ half-pel,
/// `step == 2` ⇒ quarter-pel. On entry `*best_sad` holds the SAD at
/// the current best; on exit, both are updated if any neighbour beats
/// the centre.
///
/// Repeats up to a small bounded number of iterations so the local
/// minimum can drift one step in either axis. VP9 conformant clients
/// don't care about how the encoder picks its MV — only that the
/// emitted MV decodes correctly — so an iterated diamond is fine.
#[allow(clippy::too_many_arguments)]
fn refine_subpel_8nb(
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
    row: i32,
    col: i32,
    bs_w: i32,
    bs_h: i32,
    step: i32,
    best_mv_1_8: &mut (i32, i32),
    best_sad: &mut u32,
) {
    // Two passes keep cost bounded while still letting the search drift
    // a step in either axis. Practical local minima land within 1 pass.
    for _iter in 0..2 {
        let mut improved = false;
        for dr in [-step, 0, step] {
            for dc in [-step, 0, step] {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let cand = (best_mv_1_8.0 + dr, best_mv_1_8.1 + dc);
                let sad = compute_sad_subpel(src, refr, row, col, bs_w, bs_h, cand.0, cand.1);
                if sad < *best_sad {
                    *best_sad = sad;
                    *best_mv_1_8 = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// Build a complete P-frame: uncompressed + compressed + tile.
pub fn build_pframe(
    p: &EncoderParams,
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
) -> Vec<u8> {
    use crate::encoder::compressed_header::emit_compressed_header_p;
    use crate::encoder::uncompressed_header::emit_uncompressed_header_p;

    // Round 49 wiring choices:
    let interpolation_filter = 0u8; // EightTap
    let allow_hp = false;
    let compound_allowed = false; // all sign_bias slots are 0 → uniform → no compound.

    let ch = emit_compressed_header_p(
        TxMode::Only4x4,
        false,
        interpolation_filter,
        allow_hp,
        compound_allowed,
    );
    let tile = emit_pframe_tile(p, src, refr);
    let uh = emit_uncompressed_header_p(
        p,
        ch.len() as u16,
        0x01, // refresh slot 0 (the LAST_FRAME slot).
        0,    // ref_frame_idx[LAST] = slot 0.
        interpolation_filter,
        allow_hp,
    );
    let mut out = Vec::with_capacity(uh.len() + ch.len() + tile.len());
    out.extend_from_slice(&uh);
    out.extend_from_slice(&ch);
    out.extend_from_slice(&tile);
    // Silence "unused" of FrameContext import for now — kept for future expansion.
    let _ = FrameContext::new_default();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv_class_of_matches_inverse() {
        // mag = 1 → class 0.
        assert_eq!(mv_class_of(1), 0);
        assert_eq!(mv_class_of(16), 0);
        assert_eq!(mv_class_of(17), 1);
        assert_eq!(mv_class_of(32), 1);
        assert_eq!(mv_class_of(33), 2);
        assert_eq!(mv_class_of(64), 2);
        assert_eq!(mv_class_of(65), 3);
    }

    #[test]
    fn emit_mv_component_roundtrips_small_positive() {
        // Encode value=8 (1 full pel = mag 8 → class 0), decode back.
        use crate::bool_decoder::BoolDecoder;
        use crate::mv::{read_mv_component, DEFAULT_MV_COMP_PROBS};
        let mut be = BoolEncoder::new();
        emit_mv_component(&mut be, 8, false);
        let buf = be.finish();
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let v = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
        assert_eq!(v, 8, "MV component round-trip failed");
    }

    #[test]
    fn emit_mv_component_roundtrips_4pel() {
        // 4 pels = 32 in 1/8-pel units → class 1.
        use crate::bool_decoder::BoolDecoder;
        use crate::mv::{read_mv_component, DEFAULT_MV_COMP_PROBS};
        let mut be = BoolEncoder::new();
        emit_mv_component(&mut be, 32, false);
        let buf = be.finish();
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let v = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
        assert_eq!(v, 32);
    }

    #[test]
    fn emit_mv_component_roundtrips_negative() {
        use crate::bool_decoder::BoolDecoder;
        use crate::mv::{read_mv_component, DEFAULT_MV_COMP_PROBS};
        let mut be = BoolEncoder::new();
        emit_mv_component(&mut be, -32, false);
        let buf = be.finish();
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let v = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
        assert_eq!(v, -32);
    }

    #[test]
    fn emit_mv_joint_roundtrip_hnzvnz() {
        use crate::bool_decoder::BoolDecoder;
        use crate::mv::{read_mv_joint, MV_JOINT_PROBS};
        let mut be = BoolEncoder::new();
        // joint=3 → HNZVNZ.
        be.write(1, MV_JOINT_PROBS[0]);
        be.write(1, MV_JOINT_PROBS[1]);
        be.write(1, MV_JOINT_PROBS[2]);
        let buf = be.finish();
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let j = read_mv_joint(&mut bd, MV_JOINT_PROBS).unwrap();
        assert_eq!(j as u32, 3);
    }
}
