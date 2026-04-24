//! VP9 motion-vector candidate list — §6.5 `find_mv_refs` /
//! `find_best_ref_mvs`.
//!
//! Bit-accurate translation of the spec's pseudo-code (§6.5.1–§6.5.13).
//!
//! The algorithm:
//!
//! 1. **First two neighbours** (indices 0..2 in `mv_ref_blocks[MiSize]`):
//!    if the cell has a slot `j` whose `ref_frame` matches our target,
//!    add `Mvs[j]` to the list.
//! 2. **Remaining six neighbours** (indices 2..8): call
//!    `if_same_ref_frame_add_mv` — first matching-ref slot wins, add once.
//! 3. If the list is short **and at least one in-tile neighbour exists**
//!    (`differentRefFound == 1`), re-scan all 8 neighbours calling
//!    `if_diff_ref_frame_add_mv` — scale different-ref MVs per §6.5.9
//!    (negate when sign-bias flips) and add up to 2.
//! 4. Finally `clamp_mv_ref(i)` — clip every `RefListMv[i]` row/col to
//!    the frame edges widened by `MV_BORDER = 128` 1/8-pel units (16
//!    whole pixels).
//!
//! `find_best_ref_mvs` (§6.5.12) is applied right before MV assignment:
//! if high-precision is off or not usable (§6.5.13
//! `(|delta|>>3) < COMPANDED_MVREF_THRESH = 8`), round away the 1/8-pel
//! bit, then clamp to the wider `(BORDERINPIXELS - INTERP_EXTEND) << 3
//! = (160 - 4) << 3 = 1248` 1/8-pel border. `NearestMv = list[0]`,
//! `NearMv = list[1]`, `BestMv = list[0]`.
//!
//! Simplifications still in effect:
//! * `UsePrevFrameMvs = false` — no temporal MV cache. The spec's
//!   `if_same_ref_frame_add_mv(MiRow, MiCol, refFrame, 1)` /
//!   `if_diff_ref_frame_add_mv(MiRow, MiCol, refFrame, 1)` paths are
//!   dropped. This is a real bit-accuracy gap on frames where the
//!   previous frame's MI grid was preserved; measurable as a small PSNR
//!   drop in long GOPs.
//! * `block` parameter is always `-1` — sub-8x8 partition MV search
//!   uses the whole-block MV. §6.5.14 `append_sub8x8_mvs` is not yet
//!   wired; callers only invoke `find_mv_refs` with whole blocks.

use crate::mv::Mv;

/// §4.8 constants
pub const MAX_MV_REF_CANDIDATES: usize = 2;
pub const MVREF_NEIGHBOURS: usize = 8;

/// §4.8 clipping constants.
pub const MV_BORDER: i32 = 128; // 16 whole-pel in 1/8-pel units
pub const BORDERINPIXELS: i32 = 160;
pub const INTERP_EXTEND: i32 = 4;
pub const COMPANDED_MVREF_THRESH: i32 = 8;
pub const MI_SIZE: i32 = 8;

/// §4.8 `INTRA_FRAME = 0`, `LAST_FRAME = 1`, `GOLDEN_FRAME = 2`,
/// `ALTREF_FRAME = 3`.
pub const INTRA_FRAME: u8 = 0;

/// Sentinel meaning "no second reference". Stored in the grid for
/// single-reference inter blocks and for intra blocks.
pub const NONE_FRAME: u8 = 255;

/// Per-8x8 metadata for §6.5 find_mv_refs.
#[derive(Clone, Copy, Debug)]
pub struct InterMiCell {
    /// First (and possibly only) reference-frame code. `INTRA_FRAME`
    /// means the block was decoded intra.
    pub ref_frame: [u8; 2],
    /// MV for each ref_frame slot.
    pub mv: [Mv; 2],
}

impl Default for InterMiCell {
    fn default() -> Self {
        Self {
            ref_frame: [INTRA_FRAME, NONE_FRAME],
            mv: [Mv::ZERO, Mv::ZERO],
        }
    }
}

/// 8x8 mi-grid of inter metadata for the current frame. Written by
/// `InterTile::decode_inter_block`, read by `find_mv_refs`.
#[derive(Clone, Debug, Default)]
pub struct InterMiGrid {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub cells: Vec<InterMiCell>,
}

impl InterMiGrid {
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            mi_cols,
            mi_rows,
            cells: vec![InterMiCell::default(); mi_cols.max(1) * mi_rows.max(1)],
        }
    }

    pub fn get(&self, mi_row: usize, mi_col: usize) -> InterMiCell {
        if self.mi_cols == 0 || self.mi_rows == 0 {
            return InterMiCell::default();
        }
        let r = mi_row.min(self.mi_rows - 1);
        let c = mi_col.min(self.mi_cols - 1);
        self.cells[r * self.mi_cols + c]
    }

    /// Stamp `info` into every 8x8 MI cell covered by a `w × h` (8x8
    /// units) prediction block rooted at `(mi_row, mi_col)`.
    pub fn fill(&mut self, mi_row: usize, mi_col: usize, w_8x8: usize, h_8x8: usize, info: InterMiCell) {
        let w = w_8x8.max(1);
        let h = h_8x8.max(1);
        for dy in 0..h {
            let r = mi_row + dy;
            if r >= self.mi_rows {
                break;
            }
            for dx in 0..w {
                let c = mi_col + dx;
                if c >= self.mi_cols {
                    break;
                }
                self.cells[r * self.mi_cols + c] = info;
            }
        }
    }
}

/// §6.5 `mv_ref_blocks[BLOCK_SIZES][MVREF_NEIGHBOURS][2]` — candidate
/// neighbour offsets in MI units relative to (`MiRow`, `MiCol`).
/// Indexed by VP9 `BlockSize` code (0..=12).
pub const MV_REF_BLOCKS: [[(i32, i32); MVREF_NEIGHBOURS]; 13] = [
    // 0 BLOCK_4X4
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    // 1 BLOCK_4X8
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    // 2 BLOCK_8X4
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    // 3 BLOCK_8X8
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    // 4 BLOCK_8X16
    [(0, -1), (-1, 0), (1, -1), (-1, -1), (0, -2), (-2, 0), (-2, -1), (-1, -2)],
    // 5 BLOCK_16X8
    [(-1, 0), (0, -1), (-1, 1), (-1, -1), (-2, 0), (0, -2), (-1, -2), (-2, -1)],
    // 6 BLOCK_16X16
    [(-1, 0), (0, -1), (-1, 1), (1, -1), (-1, -1), (-3, 0), (0, -3), (-3, -3)],
    // 7 BLOCK_16X32
    [(0, -1), (-1, 0), (2, -1), (-1, -1), (-1, 1), (0, -3), (-3, 0), (-3, -3)],
    // 8 BLOCK_32X16
    [(-1, 0), (0, -1), (-1, 2), (-1, -1), (1, -1), (-3, 0), (0, -3), (-3, -3)],
    // 9 BLOCK_32X32
    [(-1, 1), (1, -1), (-1, 2), (2, -1), (-1, -1), (-3, 0), (0, -3), (-3, -3)],
    // 10 BLOCK_32X64
    [(0, -1), (-1, 0), (4, -1), (-1, 2), (-1, -1), (0, -3), (-3, 0), (2, -1)],
    // 11 BLOCK_64X32
    [(-1, 0), (0, -1), (-1, 4), (2, -1), (-1, -1), (-3, 0), (0, -3), (-1, 2)],
    // 12 BLOCK_64X64
    [(-1, 3), (3, -1), (-1, 4), (4, -1), (-1, -1), (-1, 0), (0, -1), (-1, 6)],
];

/// §6.5 result of `find_mv_refs` — the two-slot candidate list.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvRefs {
    pub list: [Mv; MAX_MV_REF_CANDIDATES],
    pub count: u8,
}

impl MvRefs {
    /// Raw list accessors — used by §6.5.12 `find_best_ref_mvs` to
    /// produce `NearestMv` / `NearMv` / `BestMv`.
    pub fn best_mv(&self) -> Mv {
        self.list[0]
    }
    pub fn nearest_mv(&self) -> Mv {
        self.list[0]
    }
    pub fn near_mv(&self) -> Mv {
        self.list[1]
    }
}

/// Sign-bias table (indexed by ref_frame code; [0] = INTRA unused).
pub type RefSignBias = [bool; 4];

/// Geometry of the current block in 8x8 MI units — needed by §6.5.3
/// `clamp_mv_ref` and §6.5.12 `find_best_ref_mvs`. All values are MI
/// coordinates or counts.
#[derive(Clone, Copy, Debug)]
pub struct BlockGeom {
    pub mi_row: i32,
    pub mi_col: i32,
    /// `num_8x8_blocks_wide_lookup[MiSize]`
    pub bw_8x8: i32,
    /// `num_8x8_blocks_high_lookup[MiSize]`
    pub bh_8x8: i32,
    /// Frame height in MI units.
    pub mi_rows: i32,
    /// Frame width in MI units.
    pub mi_cols: i32,
}

impl BlockGeom {
    /// Helper for callers that have pixel `(row, col)`, `(w, h)` and the
    /// frame dimensions in pixels. Converts to MI units per §4.6 (MI =
    /// 8 luma pels).
    pub fn from_pixels(
        row_px: u32,
        col_px: u32,
        w_px: u32,
        h_px: u32,
        mi_rows: i32,
        mi_cols: i32,
    ) -> Self {
        Self {
            mi_row: (row_px as i32) / MI_SIZE,
            mi_col: (col_px as i32) / MI_SIZE,
            bw_8x8: ((w_px as i32) / MI_SIZE).max(1),
            bh_8x8: ((h_px as i32) / MI_SIZE).max(1),
            mi_rows,
            mi_cols,
        }
    }
}

/// §6.5.1 `find_mv_refs` — gather up to 2 candidate MVs for `ref_frame`.
///
/// `block_size_code` is the VP9 Table 3-1 code (0..=12) of the current
/// prediction block. `(mi_row, mi_col)` is its top-left in 8x8 MI units.
/// Tile bounds are `mi_col_start..mi_col_end` (inclusive/exclusive).
///
/// After the three neighbour scans the function applies
/// §6.5.3 `clamp_mv_ref` to each candidate, so the returned MVs are
/// guaranteed to lie within `[mbToTopEdge - MV_BORDER,
/// mbToBottomEdge + MV_BORDER]` × `[mbToLeftEdge - MV_BORDER,
/// mbToRightEdge + MV_BORDER]` 1/8-pel units.
#[allow(clippy::too_many_arguments)]
pub fn find_mv_refs(
    grid: &InterMiGrid,
    sign_bias: &RefSignBias,
    ref_frame: u8,
    block_size_code: usize,
    mi_row: i32,
    mi_col: i32,
    mi_col_start: i32,
    mi_col_end: i32,
    mi_rows: i32,
) -> MvRefs {
    // Default geometry — use block-size code to look up bw/bh.
    let geom = BlockGeom {
        mi_row,
        mi_col,
        bw_8x8: block_size_w_8x8(block_size_code),
        bh_8x8: block_size_h_8x8(block_size_code),
        mi_rows,
        mi_cols: mi_col_end.max(1), // fallback; see `find_mv_refs_geom` for the accurate call.
    };
    find_mv_refs_geom(
        grid,
        sign_bias,
        ref_frame,
        block_size_code,
        geom,
        mi_col_start,
        mi_col_end,
    )
}

/// Same as `find_mv_refs` but takes a fully populated [`BlockGeom`].
/// Preferred entrypoint — supplies accurate `mi_cols` for the
/// `clamp_mv_col` right-edge computation.
pub fn find_mv_refs_geom(
    grid: &InterMiGrid,
    sign_bias: &RefSignBias,
    ref_frame: u8,
    block_size_code: usize,
    geom: BlockGeom,
    mi_col_start: i32,
    mi_col_end: i32,
) -> MvRefs {
    let bsize = block_size_code.min(12);
    let searches = &MV_REF_BLOCKS[bsize];
    let mut out = MvRefs::default();
    let mut different_ref_found = false;

    // Step 1 (§6.5.1 lines `for i < 2`): First 2 neighbours — any slot j
    // whose ref_frame matches is taken; sets `differentRefFound` either
    // way (the spec unconditionally flips the flag once any in-tile
    // neighbour is seen, regardless of match).
    for i in 0..2 {
        let (dr, dc) = searches[i];
        let (r, c) = (geom.mi_row + dr, geom.mi_col + dc);
        if !is_inside(r, c, mi_col_start, mi_col_end, geom.mi_rows) {
            continue;
        }
        different_ref_found = true;
        let cell = grid.get(r as usize, c as usize);
        for j in 0..2 {
            if cell.ref_frame[j] == ref_frame {
                add_mv_ref_list(&mut out, cell.mv[j]);
                break;
            }
        }
    }

    // Step 2 (§6.5.1 lines `for 2 <= i < MVREF_NEIGHBOURS`):
    // Call `if_same_ref_frame_add_mv` on each in-tile neighbour.
    for i in 2..MVREF_NEIGHBOURS {
        let (dr, dc) = searches[i];
        let (r, c) = (geom.mi_row + dr, geom.mi_col + dc);
        if !is_inside(r, c, mi_col_start, mi_col_end, geom.mi_rows) {
            continue;
        }
        different_ref_found = true;
        let cell = grid.get(r as usize, c as usize);
        if_same_ref_frame_add_mv(&mut out, &cell, ref_frame);
    }

    // Step 3 (`if UsePrevFrameMvs`): temporal candidate — skipped
    // (not implemented; `UsePrevFrameMvs` always false).

    // Step 4 (§6.5.1 `if differentRefFound`): scan every neighbour once
    // more for different-ref candidates, scaling with `scale_mv`.
    if different_ref_found {
        for i in 0..MVREF_NEIGHBOURS {
            let (dr, dc) = searches[i];
            let (r, c) = (geom.mi_row + dr, geom.mi_col + dc);
            if !is_inside(r, c, mi_col_start, mi_col_end, geom.mi_rows) {
                continue;
            }
            let cell = grid.get(r as usize, c as usize);
            if_diff_ref_frame_add_mv(&mut out, &cell, ref_frame, sign_bias);
            if out.count as usize >= MAX_MV_REF_CANDIDATES {
                break;
            }
        }
    }

    // Step 5 (§6.5.3): clamp each candidate to the frame edges widened
    // by `MV_BORDER`.
    for i in 0..(out.count as usize) {
        out.list[i] = clamp_mv_ref(out.list[i], &geom);
    }
    out
}

/// §6.5.2 `is_inside` — accessible-for-MV-prediction predicate.
/// Moving across top / bottom frame edges is allowed (returns false when
/// off-frame vertically); moving across left / right **tile** edges is
/// prohibited.
fn is_inside(r: i32, c: i32, mi_col_start: i32, mi_col_end: i32, mi_rows: i32) -> bool {
    r >= 0 && r < mi_rows && c >= mi_col_start && c < mi_col_end
}

/// §6.5.6 `add_mv_ref_list` — append unless full or duplicate of the
/// first already-added candidate.
fn add_mv_ref_list(out: &mut MvRefs, mv: Mv) {
    if out.count as usize >= MAX_MV_REF_CANDIDATES {
        return;
    }
    if out.count > 0 && out.list[0] == mv {
        return;
    }
    out.list[out.count as usize] = mv;
    out.count += 1;
}

/// §6.5.7 `if_same_ref_frame_add_mv` — for each slot j, if the cell's
/// ref_frame[j] matches, add its mv and return (only the first slot
/// contributes per call).
fn if_same_ref_frame_add_mv(out: &mut MvRefs, cell: &InterMiCell, ref_frame: u8) {
    for j in 0..2 {
        if cell.ref_frame[j] == ref_frame {
            add_mv_ref_list(out, cell.mv[j]);
            return;
        }
    }
}

/// §6.5.8 `if_diff_ref_frame_add_mv` — pull in up to 2 scaled
/// different-ref MVs from a neighbour cell, honouring the spec's
/// `mvsSame` dedup rule (skip slot 1 when both MVs are identical).
fn if_diff_ref_frame_add_mv(
    out: &mut MvRefs,
    cell: &InterMiCell,
    ref_frame: u8,
    sign_bias: &RefSignBias,
) {
    let mvs_same = cell.mv[0] == cell.mv[1];
    let cand0 = cell.ref_frame[0];
    if cand0 > INTRA_FRAME && cand0 != NONE_FRAME && cand0 != ref_frame {
        let m = scale_mv(cell.mv[0], cand0, ref_frame, sign_bias);
        add_mv_ref_list(out, m);
    }
    let cand1 = cell.ref_frame[1];
    if cand1 > INTRA_FRAME && cand1 != NONE_FRAME && cand1 != ref_frame && !mvs_same {
        let m = scale_mv(cell.mv[1], cand1, ref_frame, sign_bias);
        add_mv_ref_list(out, m);
    }
}

/// §6.5.9 `scale_mv` — flip MV sign when `ref_frame_sign_bias[candFrame]
/// != ref_frame_sign_bias[refFrame]`.
fn scale_mv(mv: Mv, cand_ref: u8, ref_frame: u8, sign_bias: &RefSignBias) -> Mv {
    let ci = (cand_ref as usize).min(3);
    let ri = (ref_frame as usize).min(3);
    if sign_bias[ci] != sign_bias[ri] {
        // Spec: multiply both components by -1.
        Mv::new(neg_i16(mv.row), neg_i16(mv.col))
    } else {
        mv
    }
}

/// Two's-complement-safe negation that saturates at i16 boundaries.
#[inline]
fn neg_i16(v: i16) -> i16 {
    (-(v as i32)).clamp(-32768, 32767) as i16
}

/// §6.5.4 `clamp_mv_row` — clip a row MV to `[mbToTopEdge - border,
/// mbToBottomEdge + border]` where `border` is in 1/8-pel units and
/// the edges are computed from `(mi_row, bh_8x8, mi_rows)`.
fn clamp_mv_row(row_q3: i16, border: i32, geom: &BlockGeom) -> i16 {
    let mb_to_top = -((geom.mi_row * MI_SIZE) * 8);
    let mb_to_bottom = ((geom.mi_rows - geom.bh_8x8 - geom.mi_row) * MI_SIZE) * 8;
    let v = (row_q3 as i32).clamp(mb_to_top - border, mb_to_bottom + border);
    v.clamp(-32768, 32767) as i16
}

/// §6.5.5 `clamp_mv_col` — column counterpart of `clamp_mv_row`.
fn clamp_mv_col(col_q3: i16, border: i32, geom: &BlockGeom) -> i16 {
    let mb_to_left = -((geom.mi_col * MI_SIZE) * 8);
    let mb_to_right = ((geom.mi_cols - geom.bw_8x8 - geom.mi_col) * MI_SIZE) * 8;
    let v = (col_q3 as i32).clamp(mb_to_left - border, mb_to_right + border);
    v.clamp(-32768, 32767) as i16
}

/// §6.5.3 `clamp_mv_ref` — clamp a single MV with the smaller
/// `MV_BORDER = 128` used inside `find_mv_refs`.
fn clamp_mv_ref(mv: Mv, geom: &BlockGeom) -> Mv {
    Mv::new(
        clamp_mv_row(mv.row, MV_BORDER, geom),
        clamp_mv_col(mv.col, MV_BORDER, geom),
    )
}

/// Public helper — `{clamp_mv_row, clamp_mv_col}` applied together with
/// an explicit border. Used by §6.4.18 NEWMV to clip `BestMv + delta`
/// before motion compensation.
pub fn clamp_mv_pair(mv: Mv, border: i32, geom: &BlockGeom) -> Mv {
    Mv::new(
        clamp_mv_row(mv.row, border, geom),
        clamp_mv_col(mv.col, border, geom),
    )
}

/// §6.5.13 `use_mv_hp` — the `allow_high_precision_mv` flag only kicks
/// in when `BestMv`'s magnitude (in whole pixels) is below
/// `COMPANDED_MVREF_THRESH = 8`, i.e. `|delta| >> 3 < 8` per component.
pub fn use_mv_hp(delta: Mv) -> bool {
    (delta.row.unsigned_abs() as i32) >> 3 < COMPANDED_MVREF_THRESH
        && (delta.col.unsigned_abs() as i32) >> 3 < COMPANDED_MVREF_THRESH
}

/// §6.5.12 `find_best_ref_mvs` — optionally round off the 1/8-pel bit
/// and re-clamp with the wider `(BORDERINPIXELS - INTERP_EXTEND) << 3
/// = 1248` border. Mutates `refs` in place; returns the list so
/// `(NearestMv, NearMv, BestMv)` can be extracted via accessors.
pub fn find_best_ref_mvs(
    refs: &mut MvRefs,
    allow_high_precision_mv: bool,
    geom: &BlockGeom,
) {
    let wider_border = (BORDERINPIXELS - INTERP_EXTEND) << 3;
    for i in 0..(refs.count as usize) {
        let mut row = refs.list[i].row;
        let mut col = refs.list[i].col;
        let hp_here = allow_high_precision_mv && use_mv_hp(refs.list[i]);
        if !hp_here {
            row = round_hp_off(row);
            col = round_hp_off(col);
        }
        let clamped = Mv::new(
            clamp_mv_row(row, wider_border, geom),
            clamp_mv_col(col, wider_border, geom),
        );
        refs.list[i] = clamped;
    }
}

/// §6.5.12 body: when HP is not usable, force the low bit to 0 by
/// rounding "away from zero" by one: `if (v & 1) v += (v > 0 ? -1 : 1)`.
fn round_hp_off(v: i16) -> i16 {
    if v & 1 != 0 {
        if v > 0 {
            v - 1
        } else {
            v + 1
        }
    } else {
        v
    }
}

/// Back-compat alias — old name for the component rounding. Kept so
/// existing call sites don't break.
pub fn round_to_quarter_pel(mv: Mv) -> Mv {
    Mv::new(round_hp_off(mv.row), round_hp_off(mv.col))
}

/// `num_8x8_blocks_wide_lookup[BlockSize]` — matches
/// `crate::block::BlockSize::w() / 8` rounded up to 1 (so 4xN blocks
/// occupy one 8x8 MI column).
fn block_size_w_8x8(code: usize) -> i32 {
    match code {
        0..=4 => 1,    // 4x4, 4x8, 8x4, 8x8, 8x16
        5..=7 => 2,    // 16x8, 16x16, 16x32
        8..=10 => 4,   // 32x16, 32x32, 32x64
        11 | 12 => 8,  // 64x32, 64x64
        _ => 1,
    }
}

/// `num_8x8_blocks_high_lookup[BlockSize]`.
fn block_size_h_8x8(code: usize) -> i32 {
    match code {
        0 | 2 => 1,    // 4x4, 8x4
        1 | 3 | 5 => 1, // 4x8, 8x8, 16x8
        4 | 6 | 8 => 2, // 8x16, 16x16, 32x16
        7 | 9 | 11 => 4, // 16x32, 32x32, 64x32
        10 | 12 => 8,  // 32x64, 64x64
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(mi_row: i32, mi_col: i32, bw: i32, bh: i32, rows: i32, cols: i32) -> BlockGeom {
        BlockGeom {
            mi_row,
            mi_col,
            bw_8x8: bw,
            bh_8x8: bh,
            mi_rows: rows,
            mi_cols: cols,
        }
    }

    #[test]
    fn empty_grid_returns_zero_candidates() {
        let grid = InterMiGrid::new(4, 4);
        let sb: RefSignBias = [false; 4];
        let r = find_mv_refs(&grid, &sb, 1, 6, 0, 0, 0, 4, 4);
        assert_eq!(r.count, 0);
        assert_eq!(r.list[0], Mv::ZERO);
    }

    #[test]
    fn neighbour_with_matching_ref_is_picked_and_clamped() {
        let mut grid = InterMiGrid::new(4, 4);
        let mv = Mv::new(16, 24);
        grid.fill(
            1,
            1,
            1,
            1,
            InterMiCell {
                ref_frame: [1, NONE_FRAME],
                mv: [mv, Mv::ZERO],
            },
        );
        let sb: RefSignBias = [false; 4];
        // 16x16 block at (mi_row=2, mi_col=1) — (-1,0) neighbour is (1,1).
        let r = find_mv_refs(&grid, &sb, 1, 6, 2, 1, 0, 4, 4);
        assert_eq!(r.count, 1);
        // Small MV — clamping doesn't move it.
        assert_eq!(r.list[0], mv);
    }

    #[test]
    fn dedup_same_mv_is_not_added_twice() {
        let mut grid = InterMiGrid::new(4, 4);
        let mv = Mv::new(8, 8);
        grid.fill(1, 1, 1, 1, InterMiCell { ref_frame: [1, NONE_FRAME], mv: [mv, Mv::ZERO] });
        grid.fill(2, 1, 1, 1, InterMiCell { ref_frame: [1, NONE_FRAME], mv: [mv, Mv::ZERO] });
        let sb: RefSignBias = [false; 4];
        let r = find_mv_refs(&grid, &sb, 1, 6, 2, 2, 0, 4, 4);
        assert_eq!(r.count, 1, "duplicate MVs should collapse");
        assert_eq!(r.list[0], mv);
    }

    #[test]
    fn different_ref_mv_is_scaled_when_bias_flips() {
        let mut grid = InterMiGrid::new(4, 4);
        grid.fill(
            1,
            1,
            1,
            1,
            InterMiCell {
                ref_frame: [2, NONE_FRAME],
                mv: [Mv::new(20, -10), Mv::ZERO],
            },
        );
        let mut sb: RefSignBias = [false; 4];
        sb[2] = true;
        let r = find_mv_refs(&grid, &sb, 1, 6, 2, 1, 0, 4, 4);
        assert_eq!(r.count, 1);
        assert_eq!(r.list[0], Mv::new(-20, 10));
    }

    #[test]
    fn clamp_mv_ref_limits_row_to_frame_border() {
        // Block at (mi_row=0, mi_col=0) in an 8x8 MI frame — top edge
        // is at row 0, so mbToTopEdge = 0 and the row must be >=
        // -MV_BORDER = -128.
        let g = geom(0, 0, 2, 2, 8, 8);
        assert_eq!(clamp_mv_row(-1000, MV_BORDER, &g), -128);
        // Bottom edge: mbToBottomEdge = ((8 - 2 - 0) * 8) * 8 = 384.
        assert_eq!(clamp_mv_row(10_000, MV_BORDER, &g), 384 + 128);
    }

    #[test]
    fn clamp_mv_col_limits_col_to_frame_border() {
        let g = geom(0, 0, 2, 2, 8, 8);
        assert_eq!(clamp_mv_col(-1000, MV_BORDER, &g), -128);
        assert_eq!(clamp_mv_col(10_000, MV_BORDER, &g), 384 + 128);
    }

    #[test]
    fn find_mv_refs_clamps_at_frame_origin() {
        // Very large MV stored in grid. After find_mv_refs it must be
        // clamped to `mbToBottomEdge + MV_BORDER` since the block sits
        // at (0, 0).
        let mut grid = InterMiGrid::new(4, 4);
        grid.fill(
            1,
            1,
            1,
            1,
            InterMiCell {
                ref_frame: [1, NONE_FRAME],
                mv: [Mv::new(10_000, 10_000), Mv::ZERO],
            },
        );
        let sb: RefSignBias = [false; 4];
        // 16x16 at (mi_row=2, mi_col=2), mi_rows=4, mi_cols=4 passed
        // via geom version.
        let r = find_mv_refs_geom(
            &grid,
            &sb,
            1,
            6,
            geom(2, 2, 2, 2, 4, 4),
            0,
            4,
        );
        assert_eq!(r.count, 1);
        // mbToBottomEdge = ((4-2-2)*8)*8 = 0, so max = MV_BORDER = 128.
        assert_eq!(r.list[0].row, 128);
        assert_eq!(r.list[0].col, 128);
    }

    #[test]
    fn find_best_ref_mvs_rounds_when_hp_disabled() {
        let mut refs = MvRefs { list: [Mv::new(17, -21), Mv::ZERO], count: 1 };
        let g = geom(4, 4, 2, 2, 16, 16);
        find_best_ref_mvs(&mut refs, false, &g);
        // 17 is odd → (17 > 0 ? -1 : 1) → 16. -21 is odd → -20.
        assert_eq!(refs.list[0], Mv::new(16, -20));
    }

    #[test]
    fn use_mv_hp_only_when_all_components_small() {
        assert!(use_mv_hp(Mv::new(0, 0)));
        // (abs >> 3) < 8 — magnitude < 64 in 1/8-pel units.
        assert!(use_mv_hp(Mv::new(63, 63)));
        assert!(!use_mv_hp(Mv::new(64, 0)));
        assert!(!use_mv_hp(Mv::new(-64, 10)));
        assert!(!use_mv_hp(Mv::new(10, 64)));
    }

    #[test]
    fn find_best_ref_mvs_keeps_hp_bit_when_allowed_and_small() {
        let mut refs = MvRefs { list: [Mv::new(5, 7), Mv::ZERO], count: 1 };
        let g = geom(4, 4, 2, 2, 16, 16);
        find_best_ref_mvs(&mut refs, true, &g);
        // HP applicable → no rounding.
        assert_eq!(refs.list[0], Mv::new(5, 7));
    }

    #[test]
    fn find_best_ref_mvs_rounds_large_mv_even_with_hp_flag() {
        let mut refs = MvRefs { list: [Mv::new(513, 3), Mv::ZERO], count: 1 };
        let g = geom(4, 4, 2, 2, 16, 16);
        find_best_ref_mvs(&mut refs, true, &g);
        // |513|>>3 = 64 not < 8, so HP disabled — 513 rounds to 512.
        assert_eq!(refs.list[0].row, 512);
        // Col 3 is odd: rounded to 2.
        assert_eq!(refs.list[0].col, 2);
    }
}
