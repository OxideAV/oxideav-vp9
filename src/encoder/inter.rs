//! VP9 P-frame (inter) tile encoder — round 49 + r-next sub-pel ME + r-next
//! quadtree partitions + r-next-8x8 16×16 HORZ/VERT shapes + 8×8 leaf +
//! r-next-sub8 8×8 four-way RDO (HORZ B8x4 / VERT B4x8 / SPLIT B4x4 sub-8×8
//! emission).
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
//!   (8×16 + 8×16), SPLIT (4 × 8×8)}` and pick by RDO. At 8×8 we evaluate
//!   `{NONE (B8x8), HORZ (B8x4 — 2 sub-blocks), VERT (B4x8 — 2 sub-
//!   blocks), SPLIT (B4x4 — 4 sub-blocks)}` and pick by RDO; sub-8×8
//!   shapes engage the §6.4.16 (idy, idx) sub-block walk where the
//!   cell-level header (skip / is_inter / ref / interp filter) emits
//!   ONCE and only inter_mode + (NEWMV) MV-delta repeat per 4×4-aligned
//!   sub-block. The split decision is
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
//!     4. (r-next-hp, optional) 8-neighbour 1/8-PEL refinement around
//!        the quarter-pel best — only runs when the per-frame
//!        `EncoderParams::allow_high_precision_mv` flag is `true`. The
//!        emitted MV components keep all 3 fractional bits in that
//!        case, and the §6.4.19 `hp` bit is written in the bool stream.
//!        When the flag is `false` (the round-49 default), MV
//!        components stay 1/4-pel-aligned and the `hp` bit is elided.
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
//! * `allow_high_precision_mv` — gated per-frame via `EncoderParams`.
//!   Defaults to `false` (round-49 behaviour: `hp` bit elided,
//!   1/4-pel MV quantum). When `true` (r-next-hp), the encoder runs the
//!   extra 1/8-pel ME refinement stage and emits the `hp` bit.
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
use crate::encoder::params::{EncoderParams, ReferenceFrame, ReferenceSet};
use crate::frame_ctx::FrameContext;
use crate::mcfilter::{mc_block, InterpFilter, RefSampler};
use crate::mv::{DEFAULT_MV_COMP_PROBS, MV_JOINT_PROBS};
use crate::mvref::{
    find_best_ref_mvs, find_mv_refs_geom, BlockGeom, InterMiCell, InterMiGrid, INTRA_FRAME,
    NONE_FRAME, Y_MODE_NEWMV, Y_MODE_ZEROMV,
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

/// Emit a complete P-frame tile payload. Single-LAST when
/// `refs.golden.is_none()`; LAST + GOLDEN per-CU RDO otherwise.
/// Returns the raw bool-coded tile bytes; the caller assembles the
/// frame by prepending uncompressed + compressed headers.
pub fn emit_pframe_tile(
    p: &EncoderParams,
    src: &crate::encoder::params::YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
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
        debug_force_8x8_none_only: p.debug_force_8x8_none_only,
        allow_high_precision_mv: p.allow_high_precision_mv,
    };

    let sb_cols = p.width.div_ceil(64);
    let sb_rows = p.height.div_ceil(64);
    for sby in 0..sb_rows {
        for sbx in 0..sb_cols {
            let col = sbx * 64;
            let row = sby * 64;
            emit_inter_partition(
                &mut be, &mut ctx, row, col, 64, p.width, p.height, src, refs,
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
    /// Mirror of `EncoderParams::debug_force_8x8_none_only` — when true,
    /// the 8×8 partition picker short-circuits to PARTITION_NONE so
    /// regression tests can compare full-shape sub-8×8 RDO vs the
    /// 8×8-NONE-only baseline.
    debug_force_8x8_none_only: bool,
    /// Mirror of `EncoderParams::allow_high_precision_mv`. Gates the
    /// 1/8-pel ME refinement stage and the `hp` bit in `emit_mv*`.
    /// When `false`, MV components must be 1/4-pel aligned (even 1/8-pel
    /// units) and the `hp` bit is elided per §6.4.19.
    allow_high_precision_mv: bool,
}

/// Per-CU reference-frame code (§4.8 `LAST_FRAME = 1`, `GOLDEN_FRAME = 2`).
/// Only the two single-ref slots the r-multiref round emits; ALTREF
/// (`3`) is reserved for a future round and is never picked here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefPick {
    Last,
    Golden,
}

impl RefPick {
    fn ref_code(self) -> u8 {
        match self {
            RefPick::Last => 1,
            RefPick::Golden => 2,
        }
    }
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

    #[allow(clippy::too_many_arguments)]
    fn stamp_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        mi_w: usize,
        mi_h: usize,
        skip: bool,
        mv: (i16, i16),
        is_zeromv: bool,
        ref_pick: RefPick,
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
        // mv_grid fill — record cell with ref_frame = LAST(1) or GOLDEN(2).
        let mut cell = InterMiCell::default();
        cell.ref_frame[0] = ref_pick.ref_code();
        cell.ref_frame[1] = NONE_FRAME;
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

    /// Variant of `stamp_block` for sub-8×8 cells (B8x4 / B4x8 / B4x4).
    /// `block_mvs` is `SubMvs[refList=0][b]` — the per-4×4 sub-block MVs
    /// in spec block-index order (`b = idy*2 + idx`). The cell anchor
    /// `mv[0]` comes from the LAST sub-block's MV (§6.4.4 line 2420 —
    /// `mv` is `BlockMvs[0][3]`), driving `y_mode` for the §6.5
    /// contextCounter.
    #[allow(clippy::too_many_arguments)]
    fn stamp_block_sub8x8(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        mi_w: usize,
        mi_h: usize,
        skip: bool,
        mv: (i16, i16),
        is_zeromv: bool,
        block_mvs: &[crate::mv::Mv; 4],
        ref_pick: RefPick,
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
        let mut cell = InterMiCell::default();
        cell.ref_frame[0] = ref_pick.ref_code();
        cell.ref_frame[1] = NONE_FRAME;
        cell.mv[0] = crate::mv::Mv::new(mv.0, mv.1);
        cell.sub_mvs[0] = *block_mvs;
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
///   * bsize = 8: full {NONE (B8x8), HORZ (B8x4 — 2 sub-blocks), VERT
///     (B4x8 — 2 sub-blocks), SPLIT (B4x4 — 4 sub-blocks)} RDO. For
///     sub-8×8 shapes the encoder emits the cell-level mode-info ONCE
///     and engages the §6.4.16 (idy, idx) per-4×4 inter_mode + MV walk.
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
    refs: &ReferenceSet<'_>,
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
            emit_inter_block(be, ctx, row, col, bsize, bsize, src, refs);
            ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        // SPLIT (forced, no bit read).
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refs,
        );
        return;
    }
    if on_right {
        // Only VERT or SPLIT readable per spec (single bit at probs[2]).
        // We pick SPLIT.
        be.write(1, probs[2]);
        if bsize == 8 {
            emit_inter_block(be, ctx, row, col, bsize, bsize, src, refs);
            ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refs,
        );
        return;
    }
    if on_bottom {
        be.write(1, probs[1]);
        if bsize == 8 {
            emit_inter_block(be, ctx, row, col, bsize, bsize, src, refs);
            ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refs,
        );
        return;
    }
    // Interior block.
    if bsize == 8 {
        // §6.4.2 partition tree at bsize=8 picks one of {NONE, HORZ
        // (B8x4), VERT (B4x8), SPLIT (B4x4)}. For sub-8×8 shapes the
        // decoder's `decode_block` calls happen ONCE per cell with the
        // rectangular BlockSize; the spec's §6.4.16 (idy, idx) sub-block
        // walk inside the cell handles per-4×4 inter_mode + MV reads.
        // The encoder mirrors that: ONE cell-level header (skip /
        // is_inter / ref / interp filter) followed by N sub-block
        // (inter_mode + optional MV-delta) bursts where N ∈ {1, 2, 2, 4}
        // for {NONE, HORZ, VERT, SPLIT}.
        let pick = if ctx.debug_force_8x8_none_only {
            Partition8::None
        } else {
            pick_partition_8(row, col, src, refs, ctx.allow_high_precision_mv)
        };
        match pick {
            Partition8::None => {
                be.write(0, probs[0]);
                emit_inter_block(be, ctx, row, col, bsize, bsize, src, refs);
                ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            }
            Partition8::Horz => {
                // Wire "1, 0".
                be.write(1, probs[0]);
                be.write(0, probs[1]);
                emit_inter_block_sub8x8(be, ctx, row, col, 8, 4, src, refs);
                ctx.update_partition(bsize, bsize, 4, mi_row, mi_col);
            }
            Partition8::Vert => {
                // Wire "1, 1, 0".
                be.write(1, probs[0]);
                be.write(1, probs[1]);
                be.write(0, probs[2]);
                emit_inter_block_sub8x8(be, ctx, row, col, 4, 8, src, refs);
                ctx.update_partition(bsize, 4, bsize, mi_row, mi_col);
            }
            Partition8::Split => {
                // Wire "1, 1, 1". For SPLIT @ bsize=8 the spec calls
                // `decode_block(B4x4)` ONCE — the (idy, idx) loop walks
                // 4 sub-blocks inside that single decode_block call.
                // So the encoder emits ONE cell with 4 sub-block bursts,
                // NOT four separate partition recursions.
                be.write(1, probs[0]);
                be.write(1, probs[1]);
                be.write(1, probs[2]);
                emit_inter_block_sub8x8(be, ctx, row, col, 4, 4, src, refs);
                ctx.update_partition(bsize, 4, 4, mi_row, mi_col);
            }
        }
        return;
    }
    if bsize == 16 {
        let pick = if ctx.debug_force_16x16_only {
            Partition16::None
        } else {
            pick_partition_16(row, col, src, refs, ctx.allow_high_precision_mv)
        };
        match pick {
            Partition16::None => {
                be.write(0, probs[0]);
                emit_inter_block(be, ctx, row, col, bsize, bsize, src, refs);
                ctx.update_partition(bsize, bsize, bsize, mi_row, mi_col);
            }
            Partition16::Horz => {
                // PARTITION_HORZ — wire "1, 0".
                be.write(1, probs[0]);
                be.write(0, probs[1]);
                emit_inter_block(be, ctx, row, col, bsize, half, src, refs);
                if row + half < frame_h {
                    emit_inter_block(be, ctx, row + half, col, bsize, half, src, refs);
                }
                ctx.update_partition(bsize, bsize, half, mi_row, mi_col);
            }
            Partition16::Vert => {
                // PARTITION_VERT — wire "1, 1, 0".
                be.write(1, probs[0]);
                be.write(1, probs[1]);
                be.write(0, probs[2]);
                emit_inter_block(be, ctx, row, col, half, bsize, src, refs);
                if col + half < frame_w {
                    emit_inter_block(be, ctx, row, col + half, half, bsize, src, refs);
                }
                ctx.update_partition(bsize, half, bsize, mi_row, mi_col);
            }
            Partition16::Split => {
                // PARTITION_SPLIT — wire "1, 1, 1", recurse to 8×8.
                be.write(1, probs[0]);
                be.write(1, probs[1]);
                be.write(1, probs[2]);
                emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refs);
                emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refs);
                emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refs);
                emit_inter_partition(
                    be,
                    ctx,
                    row + half,
                    col + half,
                    half,
                    frame_w,
                    frame_h,
                    src,
                    refs,
                );
            }
        }
        return;
    }
    // bsize ∈ {64, 32} — NONE vs SPLIT RDO.
    if should_split(row, col, bsize, src, refs, ctx.allow_high_precision_mv) {
        // PARTITION_SPLIT — wire bits "1, 1, 1" per §6.4.2 partition
        // tree (`read_partition_from_tree`):
        //   bit0 = 1 (not NONE), bit1 = 1 (not HORZ), bit2 = 1 (SPLIT).
        be.write(1, probs[0]);
        be.write(1, probs[1]);
        be.write(1, probs[2]);
        emit_inter_partition(be, ctx, row, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row, col + half, half, frame_w, frame_h, src, refs);
        emit_inter_partition(be, ctx, row + half, col, half, frame_w, frame_h, src, refs);
        emit_inter_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            src,
            refs,
        );
        return;
    }
    // Interior PARTITION_NONE — emit `bit=0` against probs[0].
    be.write(0, probs[0]);
    emit_inter_block(be, ctx, row, col, bsize, bsize, src, refs);
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
    refs: &ReferenceSet<'_>,
    allow_hp: bool,
) -> Partition16 {
    let bsize = 16u32;
    let half = bsize / 2;
    let parent_eff_w = (bsize as usize).min((src.width as usize).saturating_sub(col as usize));
    let parent_eff_h = (bsize as usize).min((src.height as usize).saturating_sub(row as usize));
    if parent_eff_w == 0 || parent_eff_h == 0 {
        return Partition16::None;
    }
    let parent_sad = best_ref_sad(
        src,
        refs,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        parent_eff_h as i32,
        allow_hp,
    );
    let cost_none = parent_sad as u64;

    // HORZ — 16×8 top + 16×8 bottom.
    let top_sad = best_ref_sad(
        src,
        refs,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        (half as i32).min(parent_eff_h as i32),
        allow_hp,
    );
    let bot_eff_h = (parent_eff_h as i32) - (half as i32);
    let bot_sad = if bot_eff_h > 0 {
        best_ref_sad(
            src,
            refs,
            (row + half) as i32,
            col as i32,
            parent_eff_w as i32,
            bot_eff_h,
            allow_hp,
        ) as u64
    } else {
        0
    };
    let cost_horz = top_sad as u64 + bot_sad + SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64;

    // VERT — 8×16 left + 8×16 right.
    let left_sad = best_ref_sad(
        src,
        refs,
        row as i32,
        col as i32,
        (half as i32).min(parent_eff_w as i32),
        parent_eff_h as i32,
        allow_hp,
    );
    let right_eff_w = (parent_eff_w as i32) - (half as i32);
    let right_sad = if right_eff_w > 0 {
        best_ref_sad(
            src,
            refs,
            row as i32,
            (col + half) as i32,
            right_eff_w,
            parent_eff_h as i32,
            allow_hp,
        ) as u64
    } else {
        0
    };
    let cost_vert = left_sad as u64 + right_sad + SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64;

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
        let child_sad = best_ref_sad(
            src,
            refs,
            child_row as i32,
            child_col as i32,
            eff_w as i32,
            eff_h as i32,
            allow_hp,
        );
        child_sad_sum += child_sad as u64;
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

/// Outcome of the 8×8 partition RDO: one of the four §6.4.2 shapes
/// available at the smallest spec partition level. NONE keeps the cell
/// at B8x8 (one block, one MV); HORZ / VERT / SPLIT enter the §6.4.16
/// sub-8×8 (idy, idx) walk with shape B8x4 / B4x8 / B4x4 respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Partition8 {
    None,
    Horz,
    Vert,
    Split,
}

/// RDO across {NONE, HORZ (2 × 8×4), VERT (2 × 4×8), SPLIT (4 × 4×4)}
/// for an interior 8×8 block. Mirrors `pick_partition_16` but operates
/// at the smallest §6.4.2 partition level. The picked shape determines
/// how many independent sub-block MVs the cell can encode (1 / 2 / 2 / 4).
///
/// Per `pick_partition_16`'s rate-penalty table — at bsize=8 the cost
/// of the PARTITION bits + per-sub-block (inter_mode + MV) bursts:
/// * NONE  — 0 extra sub-block bursts → 0 penalty.
/// * HORZ  — 1 extra sub-block burst → 1× penalty.
/// * VERT  — 1 extra sub-block burst → 1× penalty.
/// * SPLIT — 3 extra sub-block bursts → 3× penalty.
fn pick_partition_8(
    row: u32,
    col: u32,
    src: &crate::encoder::params::YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
    allow_hp: bool,
) -> Partition8 {
    let bsize = 8u32;
    let half = bsize / 2;
    let parent_eff_w = (bsize as usize).min((src.width as usize).saturating_sub(col as usize));
    let parent_eff_h = (bsize as usize).min((src.height as usize).saturating_sub(row as usize));
    if parent_eff_w == 0 || parent_eff_h == 0 {
        return Partition8::None;
    }
    let parent_sad = best_ref_sad(
        src,
        refs,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        parent_eff_h as i32,
        allow_hp,
    );
    let cost_none = parent_sad as u64;

    // HORZ — 8×4 top + 8×4 bottom.
    let top_eff_h = (half as i32).min(parent_eff_h as i32);
    let cost_horz = if top_eff_h > 0 {
        let top_sad = best_ref_sad(
            src,
            refs,
            row as i32,
            col as i32,
            parent_eff_w as i32,
            top_eff_h,
            allow_hp,
        );
        let bot_eff_h = (parent_eff_h as i32) - (half as i32);
        let bot_sad = if bot_eff_h > 0 {
            best_ref_sad(
                src,
                refs,
                (row + half) as i32,
                col as i32,
                parent_eff_w as i32,
                bot_eff_h,
                allow_hp,
            ) as u64
        } else {
            0
        };
        top_sad as u64 + bot_sad + SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64
    } else {
        u64::MAX
    };

    // VERT — 4×8 left + 4×8 right.
    let left_eff_w = (half as i32).min(parent_eff_w as i32);
    let cost_vert = if left_eff_w > 0 {
        let left_sad = best_ref_sad(
            src,
            refs,
            row as i32,
            col as i32,
            left_eff_w,
            parent_eff_h as i32,
            allow_hp,
        );
        let right_eff_w = (parent_eff_w as i32) - (half as i32);
        let right_sad = if right_eff_w > 0 {
            best_ref_sad(
                src,
                refs,
                row as i32,
                (col + half) as i32,
                right_eff_w,
                parent_eff_h as i32,
                allow_hp,
            ) as u64
        } else {
            0
        };
        left_sad as u64 + right_sad + SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64
    } else {
        u64::MAX
    };

    // SPLIT — four 4×4 children.
    let mut child_sad_sum: u64 = 0;
    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
        let child_row = row + dr;
        let child_col = col + dc;
        let eff_w = (half as usize).min((src.width as usize).saturating_sub(child_col as usize));
        let eff_h = (half as usize).min((src.height as usize).saturating_sub(child_row as usize));
        if eff_w == 0 || eff_h == 0 {
            continue;
        }
        let child_sad = best_ref_sad(
            src,
            refs,
            child_row as i32,
            child_col as i32,
            eff_w as i32,
            eff_h as i32,
            allow_hp,
        );
        child_sad_sum += child_sad as u64;
    }
    let cost_split = child_sad_sum + (SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64) * 3;

    let mut best = Partition8::None;
    let mut best_cost = cost_none;
    if cost_horz < best_cost {
        best = Partition8::Horz;
        best_cost = cost_horz;
    }
    if cost_vert < best_cost {
        best = Partition8::Vert;
        best_cost = cost_vert;
    }
    if cost_split < best_cost {
        best = Partition8::Split;
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
    refs: &ReferenceSet<'_>,
    allow_hp: bool,
) -> bool {
    let parent_eff_w = (bsize as usize).min((src.width as usize).saturating_sub(col as usize));
    let parent_eff_h = (bsize as usize).min((src.height as usize).saturating_sub(row as usize));
    if parent_eff_w == 0 || parent_eff_h == 0 {
        return false;
    }
    let parent_sad = best_ref_sad(
        src,
        refs,
        row as i32,
        col as i32,
        parent_eff_w as i32,
        parent_eff_h as i32,
        allow_hp,
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
        let child_sad = best_ref_sad(
            src,
            refs,
            child_row as i32,
            child_col as i32,
            eff_w as i32,
            eff_h as i32,
            allow_hp,
        );
        child_sad_sum += child_sad as u64;
    }
    // 3 extra sub-blocks beyond what PARTITION_NONE would emit.
    let split_penalty = (SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS as u64) * 3;
    child_sad_sum + split_penalty < parent_sad as u64
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

/// Bit-rate proxy for the extra symbols GOLDEN_FRAME costs vs
/// LAST_FRAME at a single block. Per §6.4.5: LAST emits 1 bit
/// (`single_ref_p1 = 0`); GOLDEN emits 2 bits (`single_ref_p1 = 1`
/// then `single_ref_p2 = 0`). The marginal cost is ~1 entropy-coded
/// bit; we charge a constant SAD penalty proxy so RDO only flips to
/// GOLDEN when the SAD win is meaningful. Same units as
/// `SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS`.
const GOLDEN_REF_RATE_PENALTY_SAD: u32 = 32;

/// Pick the lower-SAD reference for one block. When `refs.golden` is
/// `None` the picker always returns `(LAST, last_me)`. Otherwise it
/// runs `me_search` against both refs and compares
/// `last.best_sad` vs `golden.best_sad + GOLDEN_REF_RATE_PENALTY_SAD`,
/// returning whichever is cheaper.
fn pick_ref_and_me(
    src: &crate::encoder::params::YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
    row: i32,
    col: i32,
    eff_w: i32,
    eff_h: i32,
    allow_hp: bool,
) -> (RefPick, MeResult) {
    let last_me = me_search(src, refs.last, row, col, eff_w, eff_h, allow_hp);
    let Some(golden) = refs.golden else {
        return (RefPick::Last, last_me);
    };
    let golden_me = me_search(src, golden, row, col, eff_w, eff_h, allow_hp);
    if (golden_me.best_sad as u64) + (GOLDEN_REF_RATE_PENALTY_SAD as u64)
        < (last_me.best_sad as u64)
    {
        (RefPick::Golden, golden_me)
    } else {
        (RefPick::Last, last_me)
    }
}

/// Best SAD across the available refs (LAST + optional GOLDEN) for
/// partition-RDO use. Mirrors `pick_ref_and_me` but discards the
/// `MeResult` payload since RDO only needs the SAD.
fn best_ref_sad(
    src: &crate::encoder::params::YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
    row: i32,
    col: i32,
    eff_w: i32,
    eff_h: i32,
    allow_hp: bool,
) -> u32 {
    let last_sad = me_search(src, refs.last, row, col, eff_w, eff_h, allow_hp).best_sad;
    let Some(golden) = refs.golden else {
        return last_sad;
    };
    let golden_sad = me_search(src, golden, row, col, eff_w, eff_h, allow_hp).best_sad;
    // The +GOLDEN_REF_RATE_PENALTY_SAD penalty IS what
    // `pick_ref_and_me` uses to choose; mirror it here so the
    // partition picker's SAD numbers match the actual emit-path SAD.
    if (golden_sad as u64) + (GOLDEN_REF_RATE_PENALTY_SAD as u64) < (last_sad as u64) {
        golden_sad + GOLDEN_REF_RATE_PENALTY_SAD
    } else {
        last_sad
    }
}

/// Three-stage motion search: integer-pel full search → half-pel
/// 8-neighbour refinement → quarter-pel 8-neighbour refinement.
/// With `allow_high_precision_mv = true` a fourth 1/8-pel 8-neighbour
/// refinement stage runs after quarter-pel.
/// Mirrors the body of `emit_inter_block` so the partition RDO and the
/// actual emit see identical SADs.
fn me_search(
    src: &crate::encoder::params::YuvFrame<'_>,
    refr: &ReferenceFrame,
    row: i32,
    col: i32,
    eff_w: i32,
    eff_h: i32,
    allow_hp: bool,
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
    if allow_hp {
        // r-next-hp 1/8-pel refinement — `step == 1` in 1/8-pel units.
        // Only ever runs when `allow_high_precision_mv = true` so the
        // emitted MV stays 1/4-pel-aligned in the default-precision path.
        refine_subpel_8nb(
            src,
            refr,
            row,
            col,
            eff_w,
            eff_h,
            1,
            &mut best_mv_1_8,
            &mut best_sad,
        );
    }
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
    refs: &ReferenceSet<'_>,
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

    // §6.4.16 ME + per-CU ref-frame RDO. Body lives in `pick_ref_and_me`
    // so the partition-RDO (via `best_ref_sad`) and the emit path see
    // bit-identical SAD numbers + the same ref pick.
    let eff_w = (bsize_w_px as usize).min((src.width as usize).saturating_sub(col as usize));
    let eff_h = (bsize_h_px as usize).min((src.height as usize).saturating_sub(row as usize));
    let allow_hp = ctx.allow_high_precision_mv;
    let (ref_pick, me) = pick_ref_and_me(
        src,
        refs,
        row as i32,
        col as i32,
        eff_w as i32,
        eff_h as i32,
        allow_hp,
    );
    let best_mv_1_8 = me.best_mv_1_8;
    let best_sad = me.best_sad;
    let sad_zero = me.sad_zero;

    // §6.4.17 read_ref_frames — single ref (compoundReferenceAllowed=false
    // when sign_bias is uniform → no comp_mode bit). Emit LAST (one bit)
    // or GOLDEN (two bits: p1=1 then p2=0) per §6.4.5.
    emit_single_ref_bits(be, ctx, mi_row, mi_col, ref_pick);

    // Interpolation filter is non-switchable (frame-level EightTap), no bits.

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
        // here. With HP enabled all 1/8-pel values are legal.
        if !allow_hp {
            debug_assert_eq!(r & 1, 0, "low-precision MV row must be 1/4-pel-aligned");
            debug_assert_eq!(c & 1, 0, "low-precision MV col must be 1/4-pel-aligned");
        }
        (r, c)
    };

    // §6.5.1 find_mv_refs + §6.5.12 find_best_ref_mvs — exactly what the
    // decoder will compute, so our NEWMV delta lines up. Note the
    // `ref_code` passed in here must match `ref_pick` so the neighbour
    // scan picks MVs whose ref_frame matches.
    let mi_cols_i32 = ctx.mi_cols as i32;
    let mi_rows_i32 = ctx.mi_rows as i32;
    let geom = BlockGeom::from_pixels(row, col, bsize_w_px, bsize_h_px, mi_rows_i32, mi_cols_i32);
    let sign_bias: [bool; 4] = [false; 4];
    let bsize_code = block_size_code_for(bsize_w_px, bsize_h_px) as usize;
    let mut refs_a = find_mv_refs_geom(
        &ctx.mv_grid,
        &sign_bias,
        ref_pick.ref_code(),
        bsize_code,
        geom,
        0,
        mi_cols_i32,
    );
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
        ref_pick,
    );
}

/// Emit one sub-8×8 inter cell at (row, col) with shape `sub_w_px ×
/// sub_h_px` ∈ {8×4, 4×8, 4×4}. The cell occupies one 8×8 MI slot;
/// per §6.4.16 the (idy, idx) sub-block walk runs `inter_mode +
/// assign_mv` once per 4×4-aligned sub-block:
///   * 8×4 → (num4x4w=2, num4x4h=1) → 2 sub-block bursts (idy=0 then idy=1)
///   * 4×8 → (num4x4w=1, num4x4h=2) → 2 sub-block bursts (idx=0 then idx=1)
///   * 4×4 → (num4x4w=1, num4x4h=1) → 4 sub-block bursts (idy×idx)
///
/// The cell-level mode-info (skip / is_inter / ref_frame / interp_filter)
/// is emitted ONCE up front; only `inter_mode + (NEWMV) MV-delta` are
/// per-sub-block. The cell-level `BestMv` (resolved against the
/// `BlockGeom` for the rectangular shape) anchors every sub-block's
/// NEWMV delta — this matches the decoder's `assign_mv` which pins
/// NEWMV's `BestMv` to the cell-level `RefListMv[0]` per
/// `sub8x8_refined_refs`.
///
/// Per-sub-block ME runs an integer + half + quarter-pel search
/// against the reference plane on the sub-block footprint, identical
/// to `emit_inter_block`'s body but applied to each (idy, idx) cell.
#[allow(clippy::too_many_arguments)]
fn emit_inter_block_sub8x8(
    be: &mut BoolEncoder,
    ctx: &mut InterCtx,
    row: u32,
    col: u32,
    sub_w_px: u32,
    sub_h_px: u32,
    src: &crate::encoder::params::YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
) {
    debug_assert!(matches!((sub_w_px, sub_h_px), (8, 4) | (4, 8) | (4, 4)));
    let mi_row = (row as usize) / 8;
    let mi_col = (col as usize) / 8;
    // The cell footprint is always one 8×8 MI cell (1, 1) — sub-8×8
    // shapes don't span multiple MI rows/cols.
    let cell_mi_w = 1usize;
    let cell_mi_h = 1usize;

    // Cell-level reference pick — the §6.4.5 single-ref bits emit
    // ONCE for the whole 8×8 cell, so all (idy, idx) sub-blocks share
    // the same reference. We pick the ref that wins for the parent
    // 8×8 footprint, then run per-sub-block ME against that ref only.
    let cell_eff_w = 8usize.min((src.width as usize).saturating_sub(col as usize));
    let cell_eff_h = 8usize.min((src.height as usize).saturating_sub(row as usize));
    let allow_hp = ctx.allow_high_precision_mv;
    let (ref_pick, _) = pick_ref_and_me(
        src,
        refs,
        row as i32,
        col as i32,
        cell_eff_w as i32,
        cell_eff_h as i32,
        allow_hp,
    );
    let me_refr: &ReferenceFrame = match ref_pick {
        RefPick::Last => refs.last,
        RefPick::Golden => refs.golden.expect("Golden pick requires golden ref"),
    };

    // Cell-level header: skip / is_inter / ref / (interp implicit).
    let sctx = ctx.skip_ctx(mi_row, mi_col);
    be.write(1, SKIP_PROBS[sctx]);
    let ictx = ctx.is_inter_ctx(mi_row, mi_col);
    be.write(1, IS_INTER_PROB[ictx]);
    // tx_mode = ONLY_4X4 → no tx_size bits.
    emit_single_ref_bits(be, ctx, mi_row, mi_col, ref_pick);
    // Interpolation filter is non-switchable (frame-level EightTap), no bits.

    // Cell-level find_mv_refs / find_best_ref_mvs against the rectangular
    // BlockGeom — `BestMv` is then pinned for every sub-block's NEWMV
    // delta (matches decoder `sub8x8_refined_refs`).
    let mi_cols_i32 = ctx.mi_cols as i32;
    let mi_rows_i32 = ctx.mi_rows as i32;
    let geom = BlockGeom::from_pixels(row, col, sub_w_px, sub_h_px, mi_rows_i32, mi_cols_i32);
    let sign_bias: [bool; 4] = [false; 4];
    let bsize_code = block_size_code_for(sub_w_px, sub_h_px) as usize;
    let mut refs_a = find_mv_refs_geom(
        &ctx.mv_grid,
        &sign_bias,
        ref_pick.ref_code(),
        bsize_code,
        geom,
        0,
        mi_cols_i32,
    );
    find_best_ref_mvs(&mut refs_a, allow_hp, &geom);
    let mode_ctx = (refs_a.mode_context as usize).min(6);
    let inter_mode_probs = INTER_MODE_PROBS[mode_ctx];
    let best_anchor_mv = refs_a.best_mv();

    // (num4x4w, num4x4h) per §6.4.16 — step through the (idy, idx) loop.
    let (num4x4w, num4x4h) = match (sub_w_px, sub_h_px) {
        (8, 4) => (2usize, 1usize),
        (4, 8) => (1usize, 2usize),
        (4, 4) => (1usize, 1usize),
        _ => (1usize, 1usize),
    };

    // Per-sub-block MV record for the mv_grid update at the end —
    // §6.4.4 SubMvs[r][c][refList][b], indexed by spec block index
    // `b = idy*2 + idx` in 4×4 raster order. Slot `3` doubles as the
    // cell-level anchor for ≥8×8 neighbours per §6.4.16 line 2700.
    let mut block_mvs: [crate::mv::Mv; 4] = [crate::mv::Mv::ZERO; 4];
    let mut last_mv_row_1_8 = 0i16;
    let mut last_mv_col_1_8 = 0i16;
    let mut last_is_zeromv = true;

    let mut idy = 0usize;
    while idy < 2 {
        let mut idx = 0usize;
        while idx < 2 {
            // Sub-block pixel coords + footprint.
            let sub_row_px = (row as usize) + idy * 4;
            let sub_col_px = (col as usize) + idx * 4;
            let sub_w = num4x4w * 4;
            let sub_h = num4x4h * 4;
            let eff_w = sub_w.min((src.width as usize).saturating_sub(sub_col_px));
            let eff_h = sub_h.min((src.height as usize).saturating_sub(sub_row_px));

            // Per-sub-block ME against the cell's chosen reference luma plane.
            let me = if eff_w > 0 && eff_h > 0 {
                me_search(
                    src,
                    me_refr,
                    sub_row_px as i32,
                    sub_col_px as i32,
                    eff_w as i32,
                    eff_h as i32,
                    allow_hp,
                )
            } else {
                MeResult {
                    best_mv_1_8: (0, 0),
                    best_sad: 0,
                    sad_zero: 0,
                }
            };
            let is_zeromv = me.best_sad + ME_NEWMV_GATE_SAD >= me.sad_zero;
            let (mv_row_1_8, mv_col_1_8) = if is_zeromv {
                (0i16, 0i16)
            } else {
                let r = me.best_mv_1_8.0.clamp(-32768, 32767) as i16;
                let c = me.best_mv_1_8.1.clamp(-32768, 32767) as i16;
                if !allow_hp {
                    debug_assert_eq!(r & 1, 0, "low-precision MV row must be 1/4-pel-aligned");
                    debug_assert_eq!(c & 1, 0, "low-precision MV col must be 1/4-pel-aligned");
                }
                (r, c)
            };

            // Emit inter_mode tree symbol.
            if is_zeromv {
                be.write(0, inter_mode_probs[0]); // ZEROMV
            } else {
                // NEWMV: bits "1, 1, 1".
                be.write(1, inter_mode_probs[0]);
                be.write(1, inter_mode_probs[1]);
                be.write(1, inter_mode_probs[2]);
                let dmv_r = (mv_row_1_8 as i32) - (best_anchor_mv.row as i32);
                let dmv_c = (mv_col_1_8 as i32) - (best_anchor_mv.col as i32);
                emit_mv(be, dmv_r, dmv_c, allow_hp);
            }

            // Record this sub-block's MV in `block_mvs` — every 4×4 cell
            // in the sub-block footprint shares this MV (per spec
            // BlockMvs[refList][(idy+y2)*2+idx+x2]).
            for y2 in 0..num4x4h {
                for x2 in 0..num4x4w {
                    let bi = (idy + y2) * 2 + (idx + x2);
                    block_mvs[bi] = crate::mv::Mv::new(mv_row_1_8, mv_col_1_8);
                }
            }
            last_mv_row_1_8 = mv_row_1_8;
            last_mv_col_1_8 = mv_col_1_8;
            last_is_zeromv = is_zeromv;

            idx += num4x4w;
        }
        idy += num4x4h;
    }

    // Stamp the 8×8 cell with the LAST sub-block's MV at the cell anchor
    // (`mv[0]` = `BlockMvs[0][3]` per §6.4.4 line 2420) and the per-4×4
    // sub_mvs from the (idy, idx) walk so future neighbours see the right
    // §6.5.11 SubMvs lookup.
    ctx.stamp_block_sub8x8(
        mi_row,
        mi_col,
        cell_mi_w,
        cell_mi_h,
        true,
        (last_mv_row_1_8, last_mv_col_1_8),
        last_is_zeromv,
        &block_mvs,
        ref_pick,
    );
}

/// Encoder-side neighbour snapshot — the subset of `crate::inter::NeighbourInfo`
/// that the four §9.3.2 single-ref contexts depend on. Computed once
/// per emit; the encoder mirrors the decoder's `neighbour_info` body
/// using its own `InterCtx::mv_grid` + `intra_above` / `intra_left`
/// tracker.
///
/// The r-multiref round only emits single-LAST or single-GOLDEN blocks
/// (no compound, no ALTREF), so `above_single` / `left_single` are
/// always true for non-intra neighbours — but we expose the same shape
/// as the decoder so future compound emission slots in cleanly.
struct EncNeighbourInfo {
    avail_u: bool,
    avail_l: bool,
    above_ref: [u8; 2], // [0]=primary, [1]=NONE_FRAME when single
    left_ref: [u8; 2],
    above_intra: bool,
    left_intra: bool,
    above_single: bool,
    left_single: bool,
}

fn enc_neighbour_info(ctx: &InterCtx, mi_row: usize, mi_col: usize) -> EncNeighbourInfo {
    let avail_u = mi_row > 0;
    let avail_l = mi_col > 0;
    let (above_r0, above_r1) = if avail_u {
        let cell = ctx.mv_grid.get(mi_row - 1, mi_col);
        (cell.ref_frame[0], cell.ref_frame[1])
    } else {
        (INTRA_FRAME, NONE_FRAME)
    };
    let (left_r0, left_r1) = if avail_l {
        let cell = ctx.mv_grid.get(mi_row, mi_col - 1);
        (cell.ref_frame[0], cell.ref_frame[1])
    } else {
        (INTRA_FRAME, NONE_FRAME)
    };
    let above_intra = above_r0 == INTRA_FRAME;
    let left_intra = left_r0 == INTRA_FRAME;
    let above_single = above_r1 == NONE_FRAME || above_r1 == INTRA_FRAME;
    let left_single = left_r1 == NONE_FRAME || left_r1 == INTRA_FRAME;
    // Defensive: `intra_above` / `intra_left` ought to agree with the
    // grid; honour either if it claims intra.
    let above_intra =
        above_intra || (avail_u && mi_col < ctx.intra_above.len() && ctx.intra_above[mi_col]);
    let left_intra =
        left_intra || (avail_l && mi_row < ctx.intra_left.len() && ctx.intra_left[mi_row]);
    EncNeighbourInfo {
        avail_u,
        avail_l,
        above_ref: [above_r0, above_r1],
        left_ref: [left_r0, left_r1],
        above_intra,
        left_intra,
        above_single,
        left_single,
    }
}

/// §9.3.2 `single_ref_p1` ctx — first bit of the single-ref tree
/// (0 = LAST, 1 = {GOLDEN, ALTREF}). Mirrors the decoder's
/// `InterTile::single_ref_p1_ctx` VERBATIM so the encoder picks the
/// same SINGLE_REF_PROB[ctx][0] as the decoder will.
fn single_ref_p1_ctx_from_nbr(n: &EncNeighbourInfo) -> usize {
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

/// §9.3.2 `single_ref_p2` ctx — second bit of the single-ref tree
/// (only read when `single_ref_p1 = 1`; selects GOLDEN vs ALTREF).
/// Mirrors `InterTile::single_ref_p2_ctx` VERBATIM.
fn single_ref_p2_ctx_from_nbr(n: &EncNeighbourInfo) -> usize {
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
                2 * ((n.above_ref[0] == GOLDEN) as usize) + 2 * ((n.left_ref[0] == GOLDEN) as usize)
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

/// Emit the per-block reference-frame symbols for a single-ref pick
/// (LAST or GOLDEN). The full §6.4.5 single-ref tree:
///   * `single_ref_p1` bit: 0 → LAST_FRAME, 1 → {GOLDEN, ALTREF}.
///   * if `single_ref_p1 = 1`: `single_ref_p2` bit: 0 → GOLDEN, 1 → ALTREF.
///
/// The r-multiref round only emits LAST / GOLDEN (no ALTREF picks).
fn emit_single_ref_bits(
    be: &mut BoolEncoder,
    ctx: &InterCtx,
    mi_row: usize,
    mi_col: usize,
    pick: RefPick,
) {
    let nbr = enc_neighbour_info(ctx, mi_row, mi_col);
    let p1_ctx = single_ref_p1_ctx_from_nbr(&nbr);
    match pick {
        RefPick::Last => {
            be.write(0, SINGLE_REF_PROB[p1_ctx][0]);
        }
        RefPick::Golden => {
            be.write(1, SINGLE_REF_PROB[p1_ctx][0]);
            let p2_ctx = single_ref_p2_ctx_from_nbr(&nbr);
            be.write(0, SINGLE_REF_PROB[p2_ctx][1]);
        }
    }
}

/// §3 Table 3-1 block_size_code lookup: returns the spec block-size
/// code for a rectangular `w × h` block. Covers all 13 spec block sizes
/// (B4x4=0 .. B64x64=12); sub-8×8 entries (B4x4 / B4x8 / B8x4) are
/// emitted by the r-next-sub8 round when the 8×8 picker selects HORZ /
/// VERT / SPLIT.
fn block_size_code_for(w: u32, h: u32) -> u8 {
    // Spec BlockSize enum: B4x4=0 ... B64x64=12.
    match (w, h) {
        (4, 4) => 0,
        (4, 8) => 1,
        (8, 4) => 2,
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
///
/// `refs.last` is always populated; `refs.golden` is optional. When
/// present, the encoder runs per-CU RDO between LAST and GOLDEN and
/// emits §6.4.5 single-ref bits accordingly. DPB layout:
///   * `ref_frame_idx[LAST]   = 0` — slot 0, refreshed each P-frame.
///   * `ref_frame_idx[GOLDEN] = 1` — slot 1, holds the keyframe across
///     subsequent P-frames (refresh_frame_flags = 0x01 only refreshes
///     slot 0, so slot 1 stays the keyframe).
///   * `ref_frame_idx[ALTREF] = 0` — unused but must point to a valid
///     populated slot (the keyframe fills all 8 slots, so 0 is fine).
pub fn build_pframe(
    p: &EncoderParams,
    src: &crate::encoder::params::YuvFrame<'_>,
    refs: &ReferenceSet<'_>,
) -> Vec<u8> {
    use crate::encoder::compressed_header::emit_compressed_header_p;
    use crate::encoder::uncompressed_header::emit_uncompressed_header_p;

    // Wiring choices (carry-over from round 49 + r-multiref deltas; r-next-hp
    // adds `allow_high_precision_mv` as a per-frame encoder param).
    let interpolation_filter = 0u8; // EightTap
    let allow_hp = p.allow_high_precision_mv;
    let compound_allowed = false; // all sign_bias slots are 0 → uniform → no compound.

    let ch = emit_compressed_header_p(
        TxMode::Only4x4,
        false,
        interpolation_filter,
        allow_hp,
        compound_allowed,
    );
    let tile = emit_pframe_tile(p, src, refs);
    // r-multiref DPB layout: LAST → slot 0, GOLDEN → slot 1, ALTREF
    // unused (point at 0 — a valid populated slot since the keyframe
    // filled all 8). When `golden` is absent we keep `[0, 0, 0]` which
    // matches the round-49 single-LAST behaviour exactly.
    let ref_frame_idx = if refs.golden.is_some() {
        [0u8, 1, 0]
    } else {
        [0u8, 0, 0]
    };
    let uh = emit_uncompressed_header_p(
        p,
        ch.len() as u16,
        0x01, // refresh slot 0 (the LAST_FRAME slot).
        ref_frame_idx,
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

    /// Check that a 4-sub-block NEWMV burst (matching what
    /// `emit_inter_block_sub8x8` emits for B4x4) round-trips through
    /// the bool-stream.
    #[test]
    fn b4x4_4newmv_burst_roundtrips() {
        use crate::bool_decoder::BoolDecoder;
        use crate::mv::{read_mv_component, read_mv_joint, DEFAULT_MV_COMP_PROBS, MV_JOINT_PROBS};
        let mut be = BoolEncoder::new();
        // 4 NEWMV sub-blocks each with delta (16, 16). Anchor=(0,0).
        // mode_ctx=2, INTER_MODE_PROBS[2] = [7, 166, 63].
        let inter_probs = INTER_MODE_PROBS[2];
        for _ in 0..4 {
            // NEWMV bits "1, 1, 1".
            be.write(1, inter_probs[0]);
            be.write(1, inter_probs[1]);
            be.write(1, inter_probs[2]);
            emit_mv(&mut be, 16, 16, false);
        }
        let buf = be.finish();
        let mut bd = BoolDecoder::new(&buf).unwrap();
        for blk in 0..4 {
            // read_inter_mode bool tree.
            assert_eq!(bd.read(inter_probs[0]).unwrap(), 1, "blk {blk} mode bit0");
            assert_eq!(bd.read(inter_probs[1]).unwrap(), 1, "blk {blk} mode bit1");
            assert_eq!(bd.read(inter_probs[2]).unwrap(), 1, "blk {blk} mode bit2");
            // assign_mv NEWMV: read joint + 2 components.
            let j = read_mv_joint(&mut bd, MV_JOINT_PROBS).unwrap();
            assert_eq!(j as u32, 3, "blk {blk} joint");
            let dr = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
            let dc = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
            assert_eq!((dr, dc), (16, 16), "blk {blk} delta");
        }
    }

    #[test]
    fn emit_mv_delta_32_32_class1_roundtrips() {
        use crate::bool_decoder::BoolDecoder;
        use crate::mv::{read_mv_component, read_mv_joint, DEFAULT_MV_COMP_PROBS, MV_JOINT_PROBS};
        // This is the exact symbol pattern emit_inter_block_sub8x8 produces
        // for a NEWMV at row=8 col=0 block_idx=0 with delta=(32, 32):
        // emit_mv writes joint=3, then row component for +32, then col +32.
        let mut be = BoolEncoder::new();
        emit_mv(&mut be, 32, 32, false);
        let buf = be.finish();
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let j = read_mv_joint(&mut bd, MV_JOINT_PROBS).unwrap();
        assert_eq!(j as u32, 3, "joint");
        let dr = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
        let dc = read_mv_component(&mut bd, &DEFAULT_MV_COMP_PROBS, false).unwrap();
        assert_eq!((dr, dc), (32, 32), "delta round-trip");
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
