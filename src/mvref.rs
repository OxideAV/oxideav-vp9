//! VP9 motion-vector candidate list — §6.5 `find_mv_refs`.
//!
//! Builds the two-slot `RefListMv` for a given block by searching up to
//! eight neighbour positions (`MVREF_NEIGHBOURS = 8`) as defined by
//! `mv_ref_blocks[MiSize]`. The first two neighbours that share the
//! reference frame contribute their MVs directly; subsequent same-ref
//! neighbours are added (deduped); finally, if the list is still short,
//! different-ref MVs are scaled via `scale_mv` per §6.5.9 and added.
//!
//! Output: `NearestMv = RefListMv[0]`, `NearMv = RefListMv[1]`,
//! `BestMv = RefListMv[0]` (per §6.5.12 `find_best_ref_mvs`).
//!
//! This module reads an [`InterMiGrid`] that records per-8x8 MI cell the
//! two reference-frame codes and their MVs for every decoded inter
//! block. Intra cells report `INTRA_FRAME` / zero MV.
//!
//! Simplifications kept from the previous scaffold:
//! * `UsePrevFrameMvs` is always `false` — we do not cache last frame's
//!   MV grid. Temporal candidates are therefore skipped.
//! * Block parameter `block` (for sub-8x8 partitions) is always `-1` in
//!   callers — we only support whole-block MV prediction.

use crate::mv::Mv;

/// Max candidates returned by find_mv_refs (§4.8).
pub const MAX_MV_REF_CANDIDATES: usize = 2;

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
pub const MV_REF_BLOCKS: [[(i32, i32); 8]; 13] = [
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
    /// §6.5.12 `find_best_ref_mvs` effectively: NearestMv = list[0],
    /// NearMv = list[1], BestMv = list[0]. Round to 1/8 precision when
    /// `allow_high_precision_mv` is disabled.
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

/// §6.5.1 `find_mv_refs` — gather up to 2 candidate MVs for `ref_frame`.
///
/// `block_size_code` is the VP9 Table 3-1 code (0..=12) of the current
/// prediction block. `(mi_row, mi_col)` is its top-left in 8x8 MI units.
/// Tile bounds are `mi_col_start..mi_col_end` (inclusive exclusive).
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
    let bsize = block_size_code.min(12);
    let searches = &MV_REF_BLOCKS[bsize];
    let mut out = MvRefs::default();

    // First 2 neighbours: exact same ref_frame slot match → add.
    for i in 0..2 {
        let dr = searches[i].0;
        let dc = searches[i].1;
        let (r, c) = (mi_row + dr, mi_col + dc);
        if !is_inside(r, c, mi_row, mi_col, mi_col_start, mi_col_end, mi_rows) {
            continue;
        }
        let cell = grid.get(r as usize, c as usize);
        for j in 0..2 {
            if cell.ref_frame[j] == ref_frame {
                add_mv_ref_list(&mut out, cell.mv[j]);
                break;
            }
        }
    }

    // Neighbours 2..8: any slot matching ref_frame.
    for i in 2..8 {
        let dr = searches[i].0;
        let dc = searches[i].1;
        let (r, c) = (mi_row + dr, mi_col + dc);
        if !is_inside(r, c, mi_row, mi_col, mi_col_start, mi_col_end, mi_rows) {
            continue;
        }
        let cell = grid.get(r as usize, c as usize);
        if_same_ref_frame_add_mv(&mut out, &cell, ref_frame);
    }

    // If the list is short, scan again for different-ref candidates.
    if out.count < MAX_MV_REF_CANDIDATES as u8 {
        for i in 0..8 {
            let dr = searches[i].0;
            let dc = searches[i].1;
            let (r, c) = (mi_row + dr, mi_col + dc);
            if !is_inside(r, c, mi_row, mi_col, mi_col_start, mi_col_end, mi_rows) {
                continue;
            }
            let cell = grid.get(r as usize, c as usize);
            if_diff_ref_frame_add_mv(&mut out, &cell, ref_frame, sign_bias);
        }
    }
    out
}

/// §6.5.2 is_inside: a candidate is valid if it's within the frame and
/// doesn't cross left / right tile edges.
fn is_inside(
    r: i32,
    c: i32,
    _self_r: i32,
    _self_c: i32,
    mi_col_start: i32,
    mi_col_end: i32,
    mi_rows: i32,
) -> bool {
    r >= 0 && r < mi_rows && c >= mi_col_start && c < mi_col_end
}

/// §6.5.6 add_mv_ref_list — append `mv` to the list unless full or dup.
fn add_mv_ref_list(out: &mut MvRefs, mv: Mv) {
    if out.count >= MAX_MV_REF_CANDIDATES as u8 {
        return;
    }
    if out.count > 0 && out.list[0] == mv {
        return;
    }
    out.list[out.count as usize] = mv;
    out.count += 1;
}

/// §6.5.7 if_same_ref_frame_add_mv — for each ref slot j, if the cell's
/// ref_frame[j] matches, add its mv.
fn if_same_ref_frame_add_mv(out: &mut MvRefs, cell: &InterMiCell, ref_frame: u8) {
    for j in 0..2 {
        if cell.ref_frame[j] == ref_frame {
            add_mv_ref_list(out, cell.mv[j]);
            return;
        }
    }
}

/// §6.5.8 if_diff_ref_frame_add_mv — pull in up to 2 different-ref MVs,
/// scaled per §6.5.9.
fn if_diff_ref_frame_add_mv(
    out: &mut MvRefs,
    cell: &InterMiCell,
    ref_frame: u8,
    sign_bias: &RefSignBias,
) {
    let mut mvs = [Mv::ZERO, Mv::ZERO];
    let mut frames = [INTRA_FRAME, NONE_FRAME];
    for j in 0..2 {
        mvs[j] = cell.mv[j];
        frames[j] = cell.ref_frame[j];
    }
    let mvs_same = mvs[0] == mvs[1];
    for j in 0..2 {
        let cand = frames[j];
        if cand > INTRA_FRAME && cand != NONE_FRAME && cand != ref_frame {
            let m = scale_mv(mvs[j], cand, ref_frame, sign_bias);
            add_mv_ref_list(out, m);
            if j == 0 && mvs_same {
                // Don't duplicate when both slots would scale to the same MV.
                break;
            }
        }
    }
}

/// §6.5.9 scale_mv — flip MV sign when the cand ref and current ref
/// have opposite `ref_frame_sign_bias`.
fn scale_mv(mv: Mv, cand_ref: u8, ref_frame: u8, sign_bias: &RefSignBias) -> Mv {
    let ci = (cand_ref as usize).min(3);
    let ri = (ref_frame as usize).min(3);
    if sign_bias[ci] != sign_bias[ri] {
        Mv::new(-mv.row, -mv.col)
    } else {
        mv
    }
}

/// §6.5.12 `find_best_ref_mvs` rounding: if HP is off, snap the 1/8-pel
/// components to 1/4-pel (round away from zero).
pub fn round_to_quarter_pel(mv: Mv) -> Mv {
    fn r(v: i16) -> i16 {
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
    Mv::new(r(mv.row), r(mv.col))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid_returns_zero_candidates() {
        let grid = InterMiGrid::new(4, 4);
        let sb: RefSignBias = [false; 4];
        let r = find_mv_refs(&grid, &sb, 1, 6, 0, 0, 0, 4, 4);
        assert_eq!(r.count, 0);
        assert_eq!(r.list[0], Mv::ZERO);
    }

    #[test]
    fn neighbour_with_matching_ref_is_picked() {
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
        // Block at (2,1) of size 16x16 — offsets include (-1, 0) → (1,1).
        let r = find_mv_refs(&grid, &sb, 1, 6, 2, 1, 0, 4, 4);
        assert_eq!(r.count, 1);
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
        // Neighbour references frame 2 (GOLDEN) with MV (20, -10).
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
        // Request ref_frame = 1 (LAST). Opposite sign_bias → negate MV.
        let mut sb: RefSignBias = [false; 4];
        sb[2] = true;
        let r = find_mv_refs(&grid, &sb, 1, 6, 2, 1, 0, 4, 4);
        assert_eq!(r.count, 1);
        assert_eq!(r.list[0], Mv::new(-20, 10));
    }
}
