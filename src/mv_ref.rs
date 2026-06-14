//! §6.5.1 / §6.5.2 / §6.5.3 / §6.5.4 / §6.5.5 / §6.5.6-6.5.12
//! motion-vector reference derivation — the `find_mv_refs( )` candidate
//! scan, its `is_inside( )`, `clamp_mv_ref( )`, `clamp_mv_row( )`,
//! `clamp_mv_col( )`, `add_mv_ref_list( )`, `if_same_ref_frame_add_mv( )`,
//! `if_diff_ref_frame_add_mv( )`, `scale_mv( )`, `get_block_mv( )`,
//! `get_sub_block_mv( )` helpers and the `find_best_ref_mvs( )` finaliser.
//!
//! These sit one layer above the §6.4.19 [`crate::mv::read_mv`] residual
//! decode: `find_mv_refs( )` (§6.5.1) walks the `mv_ref_blocks[ MiSize ]`
//! neighbour positions (and, when `UsePrevFrameMvs` is set, the previous
//! frame's stored vector) to populate the two `RefListMv[ ]` predictors and
//! the `ModeContext[ refFrame ]` value; `find_best_ref_mvs( )` (§6.5.12)
//! then rounds and finally clamps those candidates into the `BestMv[ ref ]`
//! predictor that `read_mv( )` adds the decoded difference onto.
//!
//! The geometry-only clamps ([`is_inside`](MvRefGeometry::is_inside) and
//! the `clamp_*` family) are pure functions of the per-block frame
//! geometry gathered in [`MvRefGeometry`]. The §6.5.1 scan additionally
//! reads neighbour mode-info through the [`MvCandidateSource`] trait, which
//! the §6.4.4 `decode_block` fan-out's `Mvs` / `RefFrames` / `SubMvs` /
//! `YModes` arrays will back once the inter decode driver threads them.
//! There is no bool coder and no probability state here, so every
//! primitive is directly testable against constructed geometry and a
//! synthetic candidate frame in isolation from the rest of the inter path.

// The §6.4.16 `inter_block_mode_info( )` driver that consumes
// [`MvRefGeometry::find_mv_refs`] / [`MvRefGeometry::find_best_ref_mvs`]
// lands on top of this layer in a later round; until then the primitives
// are reachable only from the unit tests, so the crate-internal
// `dead_code` lint is silenced module-wide (mirrors `mv`'s deferred-inter
// residual primitives).
#![allow(dead_code)]

use crate::mv::use_mv_hp;
use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};

/// `MV_BORDER = 128` per §3 (`vp9-spec.txt` line 520). The clip border the
/// §6.5.3 `clamp_mv_ref( )` step passes to `clamp_mv_row/col( )`.
pub(crate) const MV_BORDER: i32 = 128;

/// `INTERP_EXTEND = 4` per §3 (`vp9-spec.txt` line 521). Sub-pel
/// interpolation reach in full-pel units.
pub(crate) const INTERP_EXTEND: i32 = 4;

/// `BORDERINPIXELS = 160` per §3 (`vp9-spec.txt` line 522). The frame
/// border in pixels used to derive the §6.5.12 clamp range.
pub(crate) const BORDERINPIXELS: i32 = 160;

/// `MI_SIZE = 8` per §3 (`vp9-spec.txt` line 464). The smallest mode-info
/// block edge in pixels; the clamp edges are `MI`-units scaled by this.
pub(crate) const MI_SIZE: i32 = 8;

/// `MAX_MV_REF_CANDIDATES = 2` per §3 (`vp9-spec.txt` line 468). The
/// number of motion vectors `find_mv_refs( )` returns and the length of
/// the `RefListMv[ ]` list `find_best_ref_mvs( )` walks.
pub(crate) const MAX_MV_REF_CANDIDATES: usize = 2;

/// `MVREF_NEIGHBOURS = 8` per §3 (`vp9-spec.txt` line 459). The number of
/// candidate positions §6.5.1 `find_mv_refs( )` searches via the
/// `mv_ref_blocks[ ]` table.
pub(crate) const MVREF_NEIGHBOURS: usize = 8;

/// `INTRA_FRAME = 0` per §3 — the §6.5.8 `if_diff_ref_frame_add_mv( )`
/// gate `CandidateFrame[ j ] > INTRA_FRAME` excludes intra candidates.
const INTRA_FRAME: i32 = 0;

/// §6.5.1 `ModeContext[ ]` values — the `counter_to_context[ ]` codomain
/// per §3 (`vp9-spec.txt` lines 525-533). These index the §9.3.2
/// `inter_mode` probability table the §6.4.16 driver reads.
const BOTH_ZERO: u8 = 0;
const ZERO_PLUS_PREDICTED: u8 = 1;
const BOTH_PREDICTED: u8 = 2;
const NEW_PLUS_NON_INTRA: u8 = 3;
const BOTH_NEW: u8 = 4;
const INTRA_PLUS_NON_INTRA: u8 = 5;
const BOTH_INTRA: u8 = 6;
/// `INVALID_CASE = 9` per §3 (`vp9-spec.txt` line 533) — a sentinel for a
/// `counter_to_context[ ]` slot that the §6.5.1 `contextCounter`
/// accumulation can never reach.
const INVALID_CASE: u8 = 9;

/// §6.5.1 `mv_ref_blocks[ BLOCK_SIZES ][ MVREF_NEIGHBOURS ][ 2 ]`
/// (`vp9-spec.txt` lines 3117-3131) — the per-`MiSize` candidate
/// `(deltaRow, deltaCol)` search offsets (in `MI` units) relative to
/// `(MiRow, MiCol)`.
const MV_REF_BLOCKS: [[[i32; 2]; MVREF_NEIGHBOURS]; 13] = [
    // BLOCK_4X4 / 4X8 / 8X4 / 8X8 all share the first pattern.
    [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ],
    // BLOCK_8X16
    [
        [0, -1],
        [-1, 0],
        [1, -1],
        [-1, -1],
        [0, -2],
        [-2, 0],
        [-2, -1],
        [-1, -2],
    ],
    // BLOCK_16X8
    [
        [-1, 0],
        [0, -1],
        [-1, 1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-1, -2],
        [-2, -1],
    ],
    // BLOCK_16X16
    [
        [-1, 0],
        [0, -1],
        [-1, 1],
        [1, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    // BLOCK_16X32
    [
        [0, -1],
        [-1, 0],
        [2, -1],
        [-1, -1],
        [-1, 1],
        [0, -3],
        [-3, 0],
        [-3, -3],
    ],
    // BLOCK_32X16
    [
        [-1, 0],
        [0, -1],
        [-1, 2],
        [-1, -1],
        [1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    // BLOCK_32X32
    [
        [-1, 1],
        [1, -1],
        [-1, 2],
        [2, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    // BLOCK_32X64
    [
        [0, -1],
        [-1, 0],
        [4, -1],
        [-1, 2],
        [-1, -1],
        [0, -3],
        [-3, 0],
        [2, -1],
    ],
    // BLOCK_64X32
    [
        [-1, 0],
        [0, -1],
        [-1, 4],
        [2, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-1, 2],
    ],
    // BLOCK_64X64
    [
        [-1, 3],
        [3, -1],
        [-1, 4],
        [4, -1],
        [-1, -1],
        [-1, 0],
        [0, -1],
        [-1, 6],
    ],
];

/// §6.5.1 `mode_2_counter[ MB_MODE_COUNT ]` (`vp9-spec.txt` lines
/// 3135-3146) — maps a candidate `YModes[ ][ ]` value to the increment
/// added to `contextCounter`. Intra modes (0..=9) contribute 9; the four
/// inter modes `NEARESTMV` (10) / `NEARMV` (11) / `ZEROMV` (12) /
/// `NEWMV` (13) contribute 0 / 0 / 3 / 1.
const MODE_2_COUNTER: [u8; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];

/// §6.5.1 `counter_to_context[ 19 ]` (`vp9-spec.txt` lines 3149-3170) —
/// maps the accumulated `contextCounter` (at most `2 * 9 = 18`) to the
/// final `ModeContext[ refFrame ]` value.
const COUNTER_TO_CONTEXT: [u8; 19] = [
    BOTH_PREDICTED,
    NEW_PLUS_NON_INTRA,
    BOTH_NEW,
    ZERO_PLUS_PREDICTED,
    NEW_PLUS_NON_INTRA,
    INVALID_CASE,
    BOTH_ZERO,
    INVALID_CASE,
    INVALID_CASE,
    INTRA_PLUS_NON_INTRA,
    INTRA_PLUS_NON_INTRA,
    INVALID_CASE,
    INTRA_PLUS_NON_INTRA,
    INVALID_CASE,
    INVALID_CASE,
    INVALID_CASE,
    INVALID_CASE,
    INVALID_CASE,
    BOTH_INTRA,
];

/// §6.5.11 `idx_n_column_to_subblock[ 4 ][ 2 ]` (`vp9-spec.txt` lines
/// 3293-3298) — selects which of a candidate's four sub-block motion
/// vectors `get_sub_block_mv( )` reads, indexed by `(block, deltaCol ==
/// 0)`.
const IDX_N_COLUMN_TO_SUBBLOCK: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

/// `ZeroMv` per §3 — the `[row, col] == [0, 0]` motion vector the §6.5.1
/// `RefListMv[ ]` entries default to.
const ZERO_MV: [i32; 2] = [0, 0];

/// `Clip3( x, y, z )` per §4.6 (`vp9-spec.txt` lines 619-624): clamps `z`
/// into the inclusive `[x, y]` range (`x` low, `y` high).
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

/// The per-block frame geometry the §6.5 clamps read. Carries the §6.4
/// `MiRow` / `MiCol` block position, the §7.2 `MiRows` / `MiCols` frame
/// dimensions (both in `MI` units) and the §6.4.4 `MiSize` block-size
/// constant. Bundling them keeps the clamp primitives pure and lets a
/// later `find_mv_refs( )` driver construct one struct per block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MvRefGeometry {
    /// §6.4 `MiRow` — the block's top edge in `MI` units.
    pub(crate) mi_row: i32,
    /// §6.4 `MiCol` — the block's left edge in `MI` units.
    pub(crate) mi_col: i32,
    /// §7.2 `MiRows` — the frame height in `MI` units.
    pub(crate) mi_rows: i32,
    /// §7.2 `MiCols` — the frame width in `MI` units.
    pub(crate) mi_cols: i32,
    /// §6.4.4 `MiSize` — the §3 `BLOCK_*` constant for this block.
    pub(crate) mi_size: usize,
    /// §6.4.2 `MiColStart` — the current tile's left edge in `MI` units.
    pub(crate) mi_col_start: i32,
    /// §6.4.2 `MiColEnd` — the current tile's right edge in `MI` units.
    pub(crate) mi_col_end: i32,
}

impl MvRefGeometry {
    /// §6.5.4 `clamp_mv_row( mvec, border )` — clamp a row (component 0)
    /// motion-vector value into the frame's vertical range.
    ///
    /// ```text
    /// clamp_mv_row( mvec, border ) {
    ///   bh = num_8x8_blocks_high_lookup[ MiSize ]
    ///   mbToTopEdge = -((MiRow * MI_SIZE) * 8)
    ///   mbToBottomEdge = ((MiRows - bh - MiRow) * MI_SIZE) * 8
    ///   return Clip3( mbToTopEdge - border, mbToBottomEdge + border, mvec )
    /// }
    /// ```
    ///
    /// The edges are in eighth-pel units (`* MI_SIZE * 8`), matching the
    /// `Mv` representation, so `border` is added directly.
    pub(crate) fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[self.mi_size] as i32;
        let mb_to_top_edge = -((self.mi_row * MI_SIZE) * 8);
        let mb_to_bottom_edge = ((self.mi_rows - bh - self.mi_row) * MI_SIZE) * 8;
        clip3(mb_to_top_edge - border, mb_to_bottom_edge + border, mvec)
    }

    /// §6.5.5 `clamp_mv_col( mvec, border )` — clamp a col (component 1)
    /// motion-vector value into the frame's horizontal range.
    ///
    /// ```text
    /// clamp_mv_col( mvec, border ) {
    ///   bw = num_8x8_blocks_wide_lookup[ MiSize ]
    ///   mbToLeftEdge = -((MiCol * MI_SIZE) * 8)
    ///   mbToRightEdge = ((MiCols - bw - MiCol) * MI_SIZE) * 8
    ///   return Clip3( mbToLeftEdge - border, mbToRightEdge + border, mvec )
    /// }
    /// ```
    pub(crate) fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[self.mi_size] as i32;
        let mb_to_left_edge = -((self.mi_col * MI_SIZE) * 8);
        let mb_to_right_edge = ((self.mi_cols - bw - self.mi_col) * MI_SIZE) * 8;
        clip3(mb_to_left_edge - border, mb_to_right_edge + border, mvec)
    }

    /// §6.5.3 `clamp_mv_ref( i )` — clamp one `RefListMv[ i ]` entry with
    /// the §3 `MV_BORDER` border.
    ///
    /// ```text
    /// clamp_mv_ref( i ) {
    ///   RefListMv[ i ][ 0 ] = clamp_mv_row( RefListMv[ i ][ 0 ], MV_BORDER )
    ///   RefListMv[ i ][ 1 ] = clamp_mv_col( RefListMv[ i ][ 1 ], MV_BORDER )
    /// }
    /// ```
    ///
    /// Returns the clamped `[row, col]` pair; the §6.5.1 caller assigns it
    /// back into the candidate list.
    pub(crate) fn clamp_mv_ref(&self, mv: [i32; 2]) -> [i32; 2] {
        [
            self.clamp_mv_row(mv[0], MV_BORDER),
            self.clamp_mv_col(mv[1], MV_BORDER),
        ]
    }

    /// §6.5.2 `is_inside( candidateR, candidateC )` — whether a candidate
    /// mode-info position is accessible for motion-vector prediction.
    ///
    /// ```text
    /// is_inside( candidateR, candidateC ) {
    ///   return (candidateR >= 0 && candidateR < MiRows
    ///             && candidateC >= MiColStart && candidateC < MiColEnd)
    /// }
    /// ```
    ///
    /// Vertical motion across the frame's top/bottom edges is allowed
    /// (bounded by the whole-frame `MiRows`); horizontal motion is bounded
    /// by the current *tile* (`MiColStart` / `MiColEnd`), since crossing a
    /// tile column edge is prohibited.
    pub(crate) fn is_inside(&self, candidate_r: i32, candidate_c: i32) -> bool {
        candidate_r >= 0
            && candidate_r < self.mi_rows
            && candidate_c >= self.mi_col_start
            && candidate_c < self.mi_col_end
    }

    /// §6.5.12 `find_best_ref_mvs( refList )` — lowering-precision rounding
    /// and final clamp of the two `RefListMv[ ]` candidates, producing the
    /// `NearestMv` / `NearMv` / `BestMv` outputs.
    ///
    /// ```text
    /// find_best_ref_mvs( refList ) {
    ///   for ( i = 0; i < MAX_MV_REF_CANDIDATES; i++ ) {
    ///     deltaRow = RefListMv[ i ][ 0 ]
    ///     deltaCol = RefListMv[ i ][ 1 ]
    ///     if ( !allow_high_precision_mv || !use_mv_hp( RefListMv[ i ] ) ) {
    ///       if ( deltaRow & 1 ) deltaRow += (deltaRow > 0 ? -1 : 1)
    ///       if ( deltaCol & 1 ) deltaCol += (deltaCol > 0 ? -1 : 1)
    ///     }
    ///     RefListMv[ i ][ 0 ] = clamp_mv_row( deltaRow,
    ///                             (BORDERINPIXELS - INTERP_EXTEND) << 3 )
    ///     RefListMv[ i ][ 1 ] = clamp_mv_col( deltaCol,
    ///                             (BORDERINPIXELS - INTERP_EXTEND) << 3 )
    ///   }
    ///   NearestMv[ refList ] = RefListMv[ 0 ]
    ///   NearMv[ refList ]    = RefListMv[ 1 ]
    ///   BestMv[ refList ]    = RefListMv[ 0 ]
    /// }
    /// ```
    ///
    /// `ref_list_mv` is the §6.5.1 `RefListMv[ ]` candidate pair (two
    /// `[row, col]` vectors). When the third fractional (eighth-pel) bit is
    /// not in use — either the frame disallows high precision or the
    /// candidate is too large per §6.5.13 `use_mv_hp( )` — each odd
    /// component is rounded toward zero to a quarter-pel grid before the
    /// final wide clamp. Returns `[NearestMv, NearMv]`; `BestMv` equals
    /// `NearestMv` so the caller need not store it separately.
    pub(crate) fn find_best_ref_mvs(
        &self,
        mut ref_list_mv: [[i32; 2]; MAX_MV_REF_CANDIDATES],
        allow_high_precision_mv: bool,
    ) -> [[i32; 2]; MAX_MV_REF_CANDIDATES] {
        // (BORDERINPIXELS - INTERP_EXTEND) << 3 — the wide eighth-pel clamp
        // border the final clamp uses (vs MV_BORDER in clamp_mv_ref).
        let border = (BORDERINPIXELS - INTERP_EXTEND) << 3;
        for cand in ref_list_mv.iter_mut() {
            let mut delta_row = cand[0];
            let mut delta_col = cand[1];
            if !allow_high_precision_mv || !use_mv_hp(*cand) {
                if delta_row & 1 != 0 {
                    delta_row += if delta_row > 0 { -1 } else { 1 };
                }
                if delta_col & 1 != 0 {
                    delta_col += if delta_col > 0 { -1 } else { 1 };
                }
            }
            cand[0] = self.clamp_mv_row(delta_row, border);
            cand[1] = self.clamp_mv_col(delta_col, border);
        }
        ref_list_mv
    }
}

/// The per-MI decode state §6.5.1 `find_mv_refs( )` reads from neighbour
/// positions, plus the previous frame's stored vectors. The §6.4.4
/// `decode_block` fan-out produces the `Mvs` / `RefFrames` / `SubMvs` /
/// `YModes` arrays that back the current-frame accessors; `PrevMvs` /
/// `PrevRefFrames` come from the previous decoded frame when
/// `UsePrevFrameMvs` is set per §6.2.5.
///
/// The trait keeps [`find_mv_refs`] a pure function of geometry plus the
/// candidate data: every per-position read goes through it, so a test can
/// back it with a small synthetic frame and the driver needs no global
/// frame buffer. The §6.5.2 `is_inside( )` gate guarantees the
/// `(candidate_r, candidate_c)` passed to the current-frame accessors lies
/// inside `[0, MiRows) x [MiColStart, MiColEnd)`; the previous-frame
/// accessors are only ever read at `(MiRow, MiCol)`.
pub(crate) trait MvCandidateSource {
    /// §6.5.1 `YModes[ candidateR ][ candidateC ]` — the candidate's
    /// `y_mode` (an intra mode 0..=9 or one of the four inter modes
    /// 10..=13), used to accumulate `contextCounter` via `mode_2_counter`.
    fn y_mode(&self, r: i32, c: i32) -> u8;

    /// §6.5.10 current-frame `RefFrames[ candidateR ][ candidateC ][
    /// refList ]` — the candidate's reference-frame index for list
    /// `ref_list`.
    fn ref_frame(&self, r: i32, c: i32, ref_list: usize) -> i32;

    /// §6.5.10 current-frame `Mvs[ candidateR ][ candidateC ][ refList ]`
    /// — the candidate's `[row, col]` motion vector for list `ref_list`.
    fn mv(&self, r: i32, c: i32, ref_list: usize) -> [i32; 2];

    /// §6.5.11 `SubMvs[ candidateR ][ candidateC ][ refList ][ idx ]` —
    /// the candidate's sub-block motion vector. For blocks larger than
    /// 8x8 every `idx` holds the same vector as [`MvCandidateSource::mv`].
    fn sub_mv(&self, r: i32, c: i32, ref_list: usize, idx: usize) -> [i32; 2];

    /// §6.5.10 previous-frame `PrevRefFrames[ candidateR ][ candidateC ][
    /// refList ]`. Only read at `(MiRow, MiCol)` and only when
    /// `UsePrevFrameMvs` is set.
    fn prev_ref_frame(&self, r: i32, c: i32, ref_list: usize) -> i32;

    /// §6.5.10 previous-frame `PrevMvs[ candidateR ][ candidateC ][
    /// refList ]`. Only read at `(MiRow, MiCol)` and only when
    /// `UsePrevFrameMvs` is set.
    fn prev_mv(&self, r: i32, c: i32, ref_list: usize) -> [i32; 2];
}

/// The output of §6.5.1 `find_mv_refs( )`: the two clamped candidate
/// motion vectors `RefListMv[ 0 ]` / `RefListMv[ 1 ]` and the
/// `ModeContext[ refFrame ]` value the §6.4.16 driver consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MvRefs {
    /// §6.5.1 `RefListMv[ ]` — the two candidate predictors, each already
    /// passed through §6.5.3 `clamp_mv_ref( )`.
    pub(crate) ref_list_mv: [[i32; 2]; MAX_MV_REF_CANDIDATES],
    /// §6.5.1 `ModeContext[ refFrame ]` — the `counter_to_context[ ]`
    /// value derived from the neighbour `YModes`.
    pub(crate) mode_context: u8,
}

/// Mutable book-keeping for the §6.5.1 candidate scan: the running
/// `RefListMv[ ]` / `RefMvCount` plus the §6.5.7/.8/.10 `CandidateMv[ ]` /
/// `CandidateFrame[ ]` scratch the per-position helpers write into. Bundled
/// so the §6.5.6 `add_mv_ref_list( )` dedup can see the partially-built
/// list while a helper is mid-scan.
struct MvRefState {
    ref_list_mv: [[i32; 2]; MAX_MV_REF_CANDIDATES],
    ref_mv_count: usize,
    candidate_mv: [[i32; 2]; 2],
    candidate_frame: [i32; 2],
}

impl MvRefState {
    fn new() -> Self {
        MvRefState {
            ref_list_mv: [ZERO_MV; MAX_MV_REF_CANDIDATES],
            ref_mv_count: 0,
            candidate_mv: [ZERO_MV; 2],
            candidate_frame: [INTRA_FRAME; 2],
        }
    }

    /// §6.5.6 `add_mv_ref_list( refList )` (`vp9-spec.txt` lines
    /// 3216-3225) — append `CandidateMv[ refList ]` to `RefListMv[ ]`
    /// unless the list is full or the vector duplicates `RefListMv[ 0 ]`.
    fn add_mv_ref_list(&mut self, ref_list: usize) {
        if self.ref_mv_count >= MAX_MV_REF_CANDIDATES {
            return;
        }
        if self.ref_mv_count > 0 && self.candidate_mv[ref_list] == self.ref_list_mv[0] {
            return;
        }
        self.ref_list_mv[self.ref_mv_count] = self.candidate_mv[ref_list];
        self.ref_mv_count += 1;
    }
}

impl MvRefGeometry {
    /// §6.5.10 `get_block_mv( candidateR, candidateC, refList, usePrev )`
    /// (`vp9-spec.txt` lines 3274-3282) — load `CandidateMv[ refList ]` /
    /// `CandidateFrame[ refList ]` from the current or previous frame.
    fn get_block_mv<S: MvCandidateSource>(
        state: &mut MvRefState,
        src: &S,
        r: i32,
        c: i32,
        ref_list: usize,
        use_prev: bool,
    ) {
        if use_prev {
            state.candidate_mv[ref_list] = src.prev_mv(r, c, ref_list);
            state.candidate_frame[ref_list] = src.prev_ref_frame(r, c, ref_list);
        } else {
            state.candidate_mv[ref_list] = src.mv(r, c, ref_list);
            state.candidate_frame[ref_list] = src.ref_frame(r, c, ref_list);
        }
    }

    /// §6.5.11 `get_sub_block_mv( candidateR, candidateC, refList,
    /// deltaCol, block )` (`vp9-spec.txt` lines 3286-3289) — load
    /// `CandidateMv[ refList ]` from one of the candidate's four sub-block
    /// vectors. `block < 0` selects sub-block 3 (the bottom-right vector,
    /// used for non-sub-8x8 callers).
    fn get_sub_block_mv<S: MvCandidateSource>(
        state: &mut MvRefState,
        src: &S,
        r: i32,
        c: i32,
        ref_list: usize,
        delta_col: i32,
        block: i32,
    ) {
        let idx = if block >= 0 {
            IDX_N_COLUMN_TO_SUBBLOCK[block as usize][usize::from(delta_col == 0)]
        } else {
            3
        };
        state.candidate_mv[ref_list] = src.sub_mv(r, c, ref_list, idx);
    }

    /// §6.5.9 `scale_mv( refList, refFrame )` (`vp9-spec.txt` lines
    /// 3265-3270) — negate `CandidateMv[ refList ]` when the candidate's
    /// reference frame and `refFrame` disagree on sign bias. `sign_bias`
    /// is the §6.2.5 `ref_frame_sign_bias[ ]` array indexed by ref frame.
    fn scale_mv(state: &mut MvRefState, ref_list: usize, ref_frame: i32, sign_bias: &[bool; 4]) {
        let cand_frame = state.candidate_frame[ref_list];
        if sign_bias[cand_frame as usize] != sign_bias[ref_frame as usize] {
            state.candidate_mv[ref_list][0] = -state.candidate_mv[ref_list][0];
            state.candidate_mv[ref_list][1] = -state.candidate_mv[ref_list][1];
        }
    }

    /// §6.5.7 `if_same_ref_frame_add_mv( candidateR, candidateC, refFrame,
    /// usePrev )` (`vp9-spec.txt` lines 3229-3237) — add the first of the
    /// candidate's two motion vectors whose reference frame matches
    /// `refFrame`.
    fn if_same_ref_frame_add_mv<S: MvCandidateSource>(
        state: &mut MvRefState,
        src: &S,
        r: i32,
        c: i32,
        ref_frame: i32,
        use_prev: bool,
    ) {
        for j in 0..2 {
            Self::get_block_mv(state, src, r, c, j, use_prev);
            if state.candidate_frame[j] == ref_frame {
                state.add_mv_ref_list(j);
                return;
            }
        }
    }

    /// §6.5.8 `if_diff_ref_frame_add_mv( candidateR, candidateC, refFrame,
    /// usePrev )` (`vp9-spec.txt` lines 3241-3261) — add the candidate's
    /// inter motion vectors whose reference frame differs from `refFrame`,
    /// sign-scaled via §6.5.9, skipping the second if it duplicates the
    /// first.
    fn if_diff_ref_frame_add_mv<S: MvCandidateSource>(
        state: &mut MvRefState,
        src: &S,
        r: i32,
        c: i32,
        ref_frame: i32,
        use_prev: bool,
        sign_bias: &[bool; 4],
    ) {
        for j in 0..2 {
            Self::get_block_mv(state, src, r, c, j, use_prev);
        }
        let mvs_same = state.candidate_mv[0] == state.candidate_mv[1];
        if state.candidate_frame[0] > INTRA_FRAME && state.candidate_frame[0] != ref_frame {
            Self::scale_mv(state, 0, ref_frame, sign_bias);
            state.add_mv_ref_list(0);
        }
        if state.candidate_frame[1] > INTRA_FRAME
            && state.candidate_frame[1] != ref_frame
            && !mvs_same
        {
            Self::scale_mv(state, 1, ref_frame, sign_bias);
            state.add_mv_ref_list(1);
        }
    }

    /// §6.5.1 `find_mv_refs( refFrame, block )` (`vp9-spec.txt` lines
    /// 3056-3113) — scan the §6.5.1 `mv_ref_blocks[ MiSize ]` neighbour
    /// positions (and, when `use_prev_frame_mvs` is set, the previous
    /// frame's vector at `(MiRow, MiCol)`) to build the two `RefListMv[ ]`
    /// predictors and the `ModeContext[ refFrame ]` value.
    ///
    /// The scan runs in three passes per the listing: positions 0..2 read
    /// the candidate's sub-block vector via §6.5.11 when the reference
    /// frame matches; positions 2..`MVREF_NEIGHBOURS` use §6.5.7; then,
    /// if any in-frame neighbour was found, a §6.5.8 different-reference
    /// pass over all positions fills any remaining slot. `block` is the
    /// §6.4.25 sub-8x8 block index (`-1` for whole-block callers); for
    /// `block >= 0` only the §6.5.11 `idx_n_column_to_subblock[ ]` read
    /// differs.
    ///
    /// `ref_frame` is the §3 reference index being predicted; `sign_bias`
    /// is the §6.2.5 `ref_frame_sign_bias[ ]` array; `use_prev_frame_mvs`
    /// is the §6.2.5 `UsePrevFrameMvs` flag;
    /// `allow_high_precision_mv` is unused here (it feeds §6.5.12, a
    /// separate step) and so is not a parameter.
    pub(crate) fn find_mv_refs<S: MvCandidateSource>(
        &self,
        src: &S,
        ref_frame: i32,
        block: i32,
        sign_bias: &[bool; 4],
        use_prev_frame_mvs: bool,
    ) -> MvRefs {
        let mut state = MvRefState::new();
        let mut different_ref_found = false;
        let mut context_counter: usize = 0;
        let search = &MV_REF_BLOCKS[self.mi_size];

        // Pass 1 — the two closest neighbours (positions 0..2): read the
        // sub-block vector when the reference frame matches.
        for entry in search.iter().take(2) {
            let candidate_r = self.mi_row + entry[0];
            let candidate_c = self.mi_col + entry[1];
            if self.is_inside(candidate_r, candidate_c) {
                different_ref_found = true;
                context_counter +=
                    MODE_2_COUNTER[src.y_mode(candidate_r, candidate_c) as usize] as usize;
                for j in 0..2 {
                    if src.ref_frame(candidate_r, candidate_c, j) == ref_frame {
                        Self::get_sub_block_mv(
                            &mut state,
                            src,
                            candidate_r,
                            candidate_c,
                            j,
                            entry[1],
                            block,
                        );
                        state.add_mv_ref_list(j);
                        break;
                    }
                }
            }
        }

        // Pass 2 — the remaining neighbours (positions 2..MVREF_NEIGHBOURS):
        // the whole-block same-reference helper.
        for entry in search.iter().take(MVREF_NEIGHBOURS).skip(2) {
            let candidate_r = self.mi_row + entry[0];
            let candidate_c = self.mi_col + entry[1];
            if self.is_inside(candidate_r, candidate_c) {
                different_ref_found = true;
                Self::if_same_ref_frame_add_mv(
                    &mut state,
                    src,
                    candidate_r,
                    candidate_c,
                    ref_frame,
                    false,
                );
            }
        }

        // Previous-frame same-reference candidate at the current position.
        if use_prev_frame_mvs {
            Self::if_same_ref_frame_add_mv(
                &mut state,
                src,
                self.mi_row,
                self.mi_col,
                ref_frame,
                true,
            );
        }

        // Pass 3 — different-reference fill, only if an in-frame neighbour
        // was seen during passes 1/2.
        if different_ref_found {
            for entry in search.iter().take(MVREF_NEIGHBOURS) {
                let candidate_r = self.mi_row + entry[0];
                let candidate_c = self.mi_col + entry[1];
                if self.is_inside(candidate_r, candidate_c) {
                    Self::if_diff_ref_frame_add_mv(
                        &mut state,
                        src,
                        candidate_r,
                        candidate_c,
                        ref_frame,
                        false,
                        sign_bias,
                    );
                }
            }
        }

        // Previous-frame different-reference candidate at the current
        // position.
        if use_prev_frame_mvs {
            Self::if_diff_ref_frame_add_mv(
                &mut state,
                src,
                self.mi_row,
                self.mi_col,
                ref_frame,
                true,
                sign_bias,
            );
        }

        let mode_context = COUNTER_TO_CONTEXT[context_counter];

        // §6.5.3 clamp_mv_ref( i ) for both candidates.
        for i in 0..MAX_MV_REF_CANDIDATES {
            state.ref_list_mv[i] = self.clamp_mv_ref(state.ref_list_mv[i]);
        }

        MvRefs {
            ref_list_mv: state.ref_list_mv,
            mode_context,
        }
    }

    /// §6.5.14 `append_sub8x8_mvs( block, refList )` (`vp9-spec.txt` lines
    /// 3339-3357) — derive the `NearestMv[ refList ]` / `NearMv[ refList ]`
    /// predictors for one sub-block of a sub-8x8 inter block.
    ///
    /// ```text
    /// append_sub8x8_mvs( block, refList ) {
    ///   find_mv_refs( ref_frame[ refList ], block )
    ///   dst = 0
    ///   if ( block == 0 ) {
    ///     for ( i = 0; i < 2; i++ ) sub8x8Mvs[ dst++ ] = RefListMv[ i ]
    ///   } else if ( block <= 2 ) {
    ///     sub8x8Mvs[ dst++ ] = BlockMvs[ refList ][ 0 ]
    ///   } else {
    ///     sub8x8Mvs[ dst++ ] = BlockMvs[ refList ][ 2 ]
    ///     for ( idx = 1; idx >= 0 && dst < 2; idx-- )
    ///       if ( BlockMvs[ refList ][ idx ] != sub8x8Mvs[ 0 ] )
    ///         sub8x8Mvs[ dst++ ] = BlockMvs[ refList ][ idx ]
    ///   }
    ///   for ( n = 0; n < 2 && dst < 2; n++ )
    ///     if ( RefListMv[ n ] != sub8x8Mvs[ 0 ] )
    ///       sub8x8Mvs[ dst++ ] = RefListMv[ n ]
    ///   if ( dst < 2 ) sub8x8Mvs[ dst++ ] = ZeroMv
    ///   NearestMv[ refList ] = sub8x8Mvs[ 0 ]
    ///   NearMv[ refList ]    = sub8x8Mvs[ 1 ]
    /// }
    /// ```
    ///
    /// `block` is the §6.4.16 `idy * 2 + idx` sub-block index in `0..=3`;
    /// `ref_frame` is `ref_frame[ refList ]` for the current block; `block_mvs`
    /// is `BlockMvs[ refList ]` — the four sub-block motion vectors decoded so
    /// far for this 8x8 block (lower-indexed sub-blocks of the same 8x8 are
    /// already filled by §6.4.16's per-sub-block loop). The remaining
    /// arguments are forwarded to §6.5.1 [`MvRefGeometry::find_mv_refs`]: the
    /// neighbour-candidate `src`, the §6.2.5 `ref_frame_sign_bias[ ]` array,
    /// and the §6.2.5 `UsePrevFrameMvs` flag. The §6.5.14 `refList` parameter
    /// is not taken explicitly: the spec uses it only to index `BlockMvs[ ]`,
    /// `NearestMv[ ]` and `NearMv[ ]`, all of which the caller already
    /// resolves — `block_mvs` is the `BlockMvs[ refList ]` row and the return
    /// value is the per-`refList` `[NearestMv, NearMv]` pair.
    ///
    /// Returns `[NearestMv, NearMv]` for the caller's `refList`. The two
    /// `RefListMv[ ]` candidates come from `find_mv_refs( )` (already
    /// §6.5.3-clamped); the `BlockMvs` entries are inserted verbatim per the
    /// listing (no further clamp), and a §3 `ZeroMv` backfills any slot the
    /// dedup left empty.
    pub(crate) fn append_sub8x8_mvs<S: MvCandidateSource>(
        &self,
        src: &S,
        block: usize,
        ref_frame: i32,
        block_mvs: &[[i32; 2]; 4],
        sign_bias: &[bool; 4],
        use_prev_frame_mvs: bool,
    ) -> [[i32; 2]; MAX_MV_REF_CANDIDATES] {
        let ref_list_mv = self
            .find_mv_refs(src, ref_frame, block as i32, sign_bias, use_prev_frame_mvs)
            .ref_list_mv;

        let mut sub8x8_mvs = [ZERO_MV; MAX_MV_REF_CANDIDATES];
        let mut dst = 0usize;

        if block == 0 {
            // The candidate predictors fill both slots directly.
            for cand in ref_list_mv.iter().take(MAX_MV_REF_CANDIDATES) {
                sub8x8_mvs[dst] = *cand;
                dst += 1;
            }
        } else if block <= 2 {
            // Top-right / bottom-left sub-blocks seed from sub-block 0.
            sub8x8_mvs[dst] = block_mvs[0];
            dst += 1;
        } else {
            // Bottom-right sub-block seeds from sub-block 2, then walks
            // sub-blocks 1 then 0, adding any that differ from slot 0.
            sub8x8_mvs[dst] = block_mvs[2];
            dst += 1;
            let mut idx = 1i32;
            while idx >= 0 && dst < MAX_MV_REF_CANDIDATES {
                if block_mvs[idx as usize] != sub8x8_mvs[0] {
                    sub8x8_mvs[dst] = block_mvs[idx as usize];
                    dst += 1;
                }
                idx -= 1;
            }
        }

        // Fill any remaining slot from the find_mv_refs candidates, skipping
        // a duplicate of slot 0.
        let mut n = 0usize;
        while n < MAX_MV_REF_CANDIDATES && dst < MAX_MV_REF_CANDIDATES {
            if ref_list_mv[n] != sub8x8_mvs[0] {
                sub8x8_mvs[dst] = ref_list_mv[n];
                dst += 1;
            }
            n += 1;
        }

        // ZeroMv backfill for a still-empty second slot.
        if dst < MAX_MV_REF_CANDIDATES {
            sub8x8_mvs[dst] = ZERO_MV;
        }

        sub8x8_mvs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generously-sized frame so the clamps don't bite unless the test
    /// drives a value past the edge on purpose: a 256x256-MI frame with
    /// a single tile spanning the full width, block BLOCK_8X8 at the
    /// origin.
    fn big_frame() -> MvRefGeometry {
        MvRefGeometry {
            mi_row: 0,
            mi_col: 0,
            mi_rows: 256,
            mi_cols: 256,
            mi_size: 3, // BLOCK_8X8: bw == bh == 1.
            mi_col_start: 0,
            mi_col_end: 256,
        }
    }

    #[test]
    fn is_inside_accepts_in_range_and_rejects_edges() {
        let g = MvRefGeometry {
            mi_row: 4,
            mi_col: 4,
            mi_rows: 8,
            mi_cols: 8,
            mi_size: 3,
            mi_col_start: 2,
            mi_col_end: 6,
        };
        // Inside the tile column band and within the frame rows.
        assert!(g.is_inside(0, 2));
        assert!(g.is_inside(7, 5));
        // Row above the frame / at-or-below MiRows is rejected.
        assert!(!g.is_inside(-1, 4));
        assert!(!g.is_inside(8, 4));
        // Column before MiColStart / at-or-after MiColEnd is rejected,
        // even though it is within the frame width.
        assert!(!g.is_inside(4, 1));
        assert!(!g.is_inside(4, 6));
    }

    #[test]
    fn clamp_mv_row_passes_small_value_through() {
        // At the origin of a big frame the top edge is 0, bottom edge is
        // large; with MV_BORDER the range easily contains a small value.
        let g = big_frame();
        assert_eq!(g.clamp_mv_row(10, MV_BORDER), 10);
        assert_eq!(g.clamp_mv_col(-7, MV_BORDER), -7);
    }

    #[test]
    fn clamp_mv_row_clips_at_top_edge() {
        // MiRow == 0 => mbToTopEdge == 0, so the low bound is -border.
        // A value below -border clips up to -border.
        let g = big_frame();
        assert_eq!(g.clamp_mv_row(-(MV_BORDER + 50), MV_BORDER), -MV_BORDER);
        assert_eq!(g.clamp_mv_col(-(MV_BORDER + 50), MV_BORDER), -MV_BORDER);
    }

    #[test]
    fn clamp_mv_row_clips_at_bottom_edge() {
        // A 1x1-MI block (BLOCK_8X8) at the very bottom-right corner of a
        // 2x2-MI frame: MiRow == MiCol == 1, bw == bh == 1, MiRows ==
        // MiCols == 2 => mbToBottomEdge/RightEdge == 0, so the high bound
        // is +border.
        let g = MvRefGeometry {
            mi_row: 1,
            mi_col: 1,
            mi_rows: 2,
            mi_cols: 2,
            mi_size: 3,
            mi_col_start: 0,
            mi_col_end: 2,
        };
        assert_eq!(g.clamp_mv_row(MV_BORDER + 50, MV_BORDER), MV_BORDER);
        assert_eq!(g.clamp_mv_col(MV_BORDER + 50, MV_BORDER), MV_BORDER);
        // mbToTopEdge == -((1 * 8) * 8) == -64, low bound == -64-128.
        assert_eq!(
            g.clamp_mv_row(-(64 + MV_BORDER + 10), MV_BORDER),
            -(64 + MV_BORDER)
        );
    }

    #[test]
    fn clamp_mv_ref_applies_row_and_col_with_mv_border() {
        let g = big_frame();
        // Row clips at the top edge (-MV_BORDER), col passes through.
        let out = g.clamp_mv_ref([-(MV_BORDER + 100), 5]);
        assert_eq!(out, [-MV_BORDER, 5]);
    }

    #[test]
    fn find_best_ref_mvs_rounds_odd_components_toward_zero_when_no_hp() {
        // No high precision: odd eighth-pel components round toward zero.
        // 5 -> 4, -5 -> -4, 3 -> 2, -1 -> 0. Values are small enough to
        // pass the wide clamp unchanged.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[5, -5], [3, -1]], false);
        assert_eq!(out[0], [4, -4]);
        assert_eq!(out[1], [2, 0]);
    }

    #[test]
    fn find_best_ref_mvs_keeps_even_components() {
        // Even components are untouched by the rounding step.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[8, -16], [0, 2]], false);
        assert_eq!(out[0], [8, -16]);
        assert_eq!(out[1], [0, 2]);
    }

    #[test]
    fn find_best_ref_mvs_high_precision_small_keeps_odd_bit() {
        // allow_high_precision_mv && use_mv_hp([1, -3]) (both Abs>>3 == 0
        // < COMPANDED_MVREF_THRESH) => no rounding, odd bit preserved.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[1, -3], [0, 0]], true);
        assert_eq!(out[0], [1, -3]);
        assert_eq!(out[1], [0, 0]);
    }

    #[test]
    fn find_best_ref_mvs_high_precision_large_still_rounds() {
        // allow_high_precision_mv but the candidate is too large for
        // use_mv_hp (Abs(72) >> 3 == 9 >= 8) => the !use_mv_hp arm fires
        // and the odd row component rounds toward zero.
        let g = big_frame();
        let out = g.find_best_ref_mvs([[73, 0], [0, 0]], true);
        assert_eq!(out[0], [72, 0]);
    }

    #[test]
    fn find_best_ref_mvs_final_clamp_uses_wide_border() {
        // The §6.5.12 clamp border is (BORDERINPIXELS - INTERP_EXTEND) <<
        // 3 == (160 - 4) << 3 == 1248, much wider than MV_BORDER. A
        // top-edge block (MiRow == 0) clips a large negative row at
        // -1248.
        let g = big_frame();
        let border = (BORDERINPIXELS - INTERP_EXTEND) << 3;
        assert_eq!(border, 1248);
        // Even value so no rounding; below -border clips to -border.
        let out = g.find_best_ref_mvs([[-(1248 + 16), 0], [0, 0]], false);
        assert_eq!(out[0], [-1248, 0]);
    }

    // ----- §6.5.1 find_mv_refs( ) -----

    const INTRA_MODE: u8 = 0; // a §6.5.1 intra YModes value (0..=9).
    const NEARESTMV: u8 = 10;
    const NEWMV: u8 = 13;

    /// A per-MI table backing [`MvCandidateSource`] for a small synthetic
    /// frame. Each MI position holds two `(ref_frame, mv)` list entries, a
    /// `y_mode`, and four sub-block vectors per list; an unset position
    /// reads back as intra (`ref_frame = INTRA_FRAME`, `mv = ZeroMv`).
    #[derive(Clone)]
    struct TestSource {
        mi_cols: i32,
        // Keyed by (r * mi_cols + c).
        cells: std::collections::HashMap<i32, Cell>,
        prev: std::collections::HashMap<i32, Cell>,
    }

    #[derive(Clone, Copy)]
    struct Cell {
        y_mode: u8,
        ref_frame: [i32; 2],
        mv: [[i32; 2]; 2],
        sub_mv: [[[i32; 2]; 4]; 2],
    }

    impl Cell {
        fn inter(ref_frame: i32, mv: [i32; 2], y_mode: u8) -> Self {
            Cell {
                y_mode,
                ref_frame: [ref_frame, NONE_INTRA],
                mv: [mv, ZERO_MV],
                // All four sub-blocks share `mv` for a non-sub-8x8 block.
                sub_mv: [[mv; 4], [ZERO_MV; 4]],
            }
        }
    }

    /// `NONE = -1` for the second list slot of a single-reference block.
    const NONE_INTRA: i32 = -1;

    impl TestSource {
        fn new(mi_cols: i32) -> Self {
            TestSource {
                mi_cols,
                cells: std::collections::HashMap::new(),
                prev: std::collections::HashMap::new(),
            }
        }
        fn set(&mut self, r: i32, c: i32, cell: Cell) {
            self.cells.insert(r * self.mi_cols + c, cell);
        }
        fn set_prev(&mut self, r: i32, c: i32, cell: Cell) {
            self.prev.insert(r * self.mi_cols + c, cell);
        }
        fn get(&self, r: i32, c: i32) -> Cell {
            self.cells
                .get(&(r * self.mi_cols + c))
                .copied()
                .unwrap_or(Cell {
                    y_mode: INTRA_MODE,
                    ref_frame: [INTRA_FRAME, NONE_INTRA],
                    mv: [ZERO_MV; 2],
                    sub_mv: [[ZERO_MV; 4]; 2],
                })
        }
        fn get_prev(&self, r: i32, c: i32) -> Cell {
            self.prev
                .get(&(r * self.mi_cols + c))
                .copied()
                .unwrap_or(Cell {
                    y_mode: INTRA_MODE,
                    ref_frame: [INTRA_FRAME, NONE_INTRA],
                    mv: [ZERO_MV; 2],
                    sub_mv: [[ZERO_MV; 4]; 2],
                })
        }
    }

    impl MvCandidateSource for TestSource {
        fn y_mode(&self, r: i32, c: i32) -> u8 {
            self.get(r, c).y_mode
        }
        fn ref_frame(&self, r: i32, c: i32, ref_list: usize) -> i32 {
            self.get(r, c).ref_frame[ref_list]
        }
        fn mv(&self, r: i32, c: i32, ref_list: usize) -> [i32; 2] {
            self.get(r, c).mv[ref_list]
        }
        fn sub_mv(&self, r: i32, c: i32, ref_list: usize, idx: usize) -> [i32; 2] {
            self.get(r, c).sub_mv[ref_list][idx]
        }
        fn prev_ref_frame(&self, r: i32, c: i32, ref_list: usize) -> i32 {
            self.get_prev(r, c).ref_frame[ref_list]
        }
        fn prev_mv(&self, r: i32, c: i32, ref_list: usize) -> [i32; 2] {
            self.get_prev(r, c).mv[ref_list]
        }
    }

    /// Geometry for an interior BLOCK_8X8 block away from every frame edge
    /// so the §6.5 clamps pass small vectors through unchanged.
    fn interior_geom() -> MvRefGeometry {
        MvRefGeometry {
            mi_row: 8,
            mi_col: 8,
            mi_rows: 64,
            mi_cols: 64,
            mi_size: 3, // BLOCK_8X8.
            mi_col_start: 0,
            mi_col_end: 64,
        }
    }

    const NO_BIAS: [bool; 4] = [false; 4];

    #[test]
    fn find_mv_refs_empty_neighbourhood_yields_zero_and_both_intra() {
        // Every neighbour is intra (default), so no candidate is added and
        // contextCounter accumulates 9 per in-frame neighbour. For an
        // interior BLOCK_8X8 all 8 search positions are in-frame; but only
        // the first two contribute to contextCounter (passes via
        // mode_2_counter run only in pass 1), so counter == 18 ->
        // counter_to_context[18] == BOTH_INTRA.
        let g = interior_geom();
        let src = TestSource::new(64);
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv, [ZERO_MV, ZERO_MV]);
        assert_eq!(out.mode_context, BOTH_INTRA);
    }

    const LAST_FRAME: i32 = 1;
    const GOLDEN_FRAME: i32 = 2;

    #[test]
    fn find_mv_refs_pass1_above_neighbour_matches_ref() {
        // mv_ref_blocks[BLOCK_8X8][0] == {-1, 0}: the directly-above MI.
        // An inter NEARESTMV candidate referencing LAST_FRAME with mv
        // [4, -8] is read via the sub-block path (idx for deltaCol == 0 is
        // column 1 of idx_n_column_to_subblock -> sub_mv index 2, which on
        // this whole-block candidate equals mv).
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [4, -8], NEARESTMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv[0], [4, -8]);
        assert_eq!(out.ref_list_mv[1], ZERO_MV);
    }

    #[test]
    fn find_mv_refs_dedup_skips_identical_second_candidate() {
        // Two neighbours with the same LAST_FRAME mv: the second is
        // dropped by §6.5.6 add_mv_ref_list dedup, leaving slot 1 zero.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        // Position 0 ({-1,0}) and position 1 ({0,-1}) both [6, 6].
        src.set(7, 8, Cell::inter(LAST_FRAME, [6, 6], NEARESTMV));
        src.set(8, 7, Cell::inter(LAST_FRAME, [6, 6], NEARESTMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv[0], [6, 6]);
        assert_eq!(out.ref_list_mv[1], ZERO_MV);
    }

    #[test]
    fn find_mv_refs_two_distinct_candidates_fill_both_slots() {
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [2, 2], NEARESTMV));
        src.set(8, 7, Cell::inter(LAST_FRAME, [10, -4], NEARESTMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv[0], [2, 2]);
        assert_eq!(out.ref_list_mv[1], [10, -4]);
    }

    #[test]
    fn find_mv_refs_diff_ref_pass_fills_remaining_slot() {
        // One same-ref candidate fills slot 0 in pass 1/2; a different-ref
        // candidate (GOLDEN_FRAME) fills slot 1 in pass 3.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [3, 3], NEARESTMV)); // same ref
        src.set(8, 7, Cell::inter(GOLDEN_FRAME, [9, -9], NEARESTMV)); // diff ref
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv[0], [3, 3]);
        assert_eq!(out.ref_list_mv[1], [9, -9]);
    }

    #[test]
    fn find_mv_refs_scale_mv_negates_on_sign_bias_mismatch() {
        // A different-ref candidate whose ref frame has opposite sign bias
        // gets its mv negated by §6.5.9 scale_mv before being added.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        // No same-ref candidate, so slot 0 comes from the diff-ref pass.
        // Need differentRefFound set: any in-frame inter neighbour does it.
        src.set(7, 8, Cell::inter(GOLDEN_FRAME, [5, -7], NEARESTMV));
        // sign_bias[GOLDEN_FRAME=2] = true, sign_bias[LAST_FRAME=1] = false
        let mut bias = NO_BIAS;
        bias[GOLDEN_FRAME as usize] = true;
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &bias, false);
        assert_eq!(out.ref_list_mv[0], [-5, 7]);
    }

    #[test]
    fn find_mv_refs_uses_prev_frame_mvs_when_enabled() {
        // No current-frame neighbour matches; the previous frame's vector
        // at (MiRow, MiCol) supplies the candidate via the usePrev path.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set_prev(8, 8, Cell::inter(LAST_FRAME, [12, 0], NEARESTMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, true);
        assert_eq!(out.ref_list_mv[0], [12, 0]);
        // Disabling the flag drops the candidate.
        let out2 = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out2.ref_list_mv[0], ZERO_MV);
    }

    #[test]
    fn find_mv_refs_context_counter_newmv_neighbours() {
        // Two NEWMV neighbours in pass-1 positions: mode_2_counter[NEWMV]
        // == 1 each, contextCounter == 2 -> counter_to_context[2] ==
        // BOTH_NEW.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [1, 1], NEWMV));
        src.set(8, 7, Cell::inter(LAST_FRAME, [2, 2], NEWMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.mode_context, BOTH_NEW);
    }

    #[test]
    fn find_mv_refs_is_inside_excludes_out_of_frame_neighbours() {
        // A BLOCK_8X8 at the top-left corner: the {-1,*} / {*,-1} search
        // positions fall outside the frame and tile, so no candidate is
        // added and contextCounter stays 0 -> counter_to_context[0] ==
        // BOTH_PREDICTED (the all-zero-counter context).
        let g = MvRefGeometry {
            mi_row: 0,
            mi_col: 0,
            mi_rows: 64,
            mi_cols: 64,
            mi_size: 3,
            mi_col_start: 0,
            mi_col_end: 64,
        };
        let mut src = TestSource::new(64);
        // Place an inter block above-left; it is out of frame so ignored.
        src.set(-1, 0, Cell::inter(LAST_FRAME, [4, 4], NEARESTMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv, [ZERO_MV, ZERO_MV]);
        assert_eq!(out.mode_context, BOTH_PREDICTED);
    }

    #[test]
    fn find_mv_refs_clamps_large_candidate() {
        // A top-edge block with a large negative-row candidate clamps to
        // the §6.5.3 MV_BORDER range.
        let g = MvRefGeometry {
            mi_row: 0,
            mi_col: 8,
            mi_rows: 64,
            mi_cols: 64,
            mi_size: 3,
            mi_col_start: 0,
            mi_col_end: 64,
        };
        let mut src = TestSource::new(64);
        // Left neighbour {0,-1} is in-frame at (0, 7).
        src.set(0, 7, Cell::inter(LAST_FRAME, [-100000, 0], NEARESTMV));
        let out = g.find_mv_refs(&src, LAST_FRAME, -1, &NO_BIAS, false);
        // mbToTopEdge == 0 at MiRow 0, so low bound == -MV_BORDER.
        assert_eq!(out.ref_list_mv[0][0], -MV_BORDER);
    }

    #[test]
    fn find_mv_refs_sub_block_index_for_sub8x8() {
        // block >= 0 selects via idx_n_column_to_subblock. For block == 0
        // and deltaCol != 0 (left neighbour {0,-1} has deltaCol -1 != 0)
        // the index is column 0 -> idx_n_column_to_subblock[0][0] == 1.
        // Give the candidate distinct sub-block vectors to prove the right
        // index is read.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        let mut cell = Cell::inter(LAST_FRAME, [0, 0], NEARESTMV);
        cell.sub_mv[0] = [[1, 1], [22, 22], [3, 3], [4, 4]];
        // Left neighbour {0,-1}: deltaCol == -1 -> index column 0 -> sub 1.
        src.set(8, 7, cell);
        // Disable position-0 above neighbour by leaving it intra.
        let out = g.find_mv_refs(&src, LAST_FRAME, 0, &NO_BIAS, false);
        assert_eq!(out.ref_list_mv[0], [22, 22]); // sub_mv index 1.
    }

    // --- §6.5.14 append_sub8x8_mvs --------------------------------------

    /// The four `BlockMvs[ refList ]` sub-block vectors used by the
    /// §6.5.14 tests for non-zero `block`.
    const NO_BLOCK_MVS: [[i32; 2]; 4] = [ZERO_MV; 4];

    #[test]
    fn append_sub8x8_block0_takes_both_ref_list_candidates() {
        // block == 0: sub8x8Mvs is exactly RefListMv[0], RefListMv[1].
        // Two distinct neighbours fill both find_mv_refs slots.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [2, 2], NEARESTMV));
        src.set(8, 7, Cell::inter(LAST_FRAME, [10, -4], NEARESTMV));
        let out = g.append_sub8x8_mvs(&src, 0, LAST_FRAME, &NO_BLOCK_MVS, &NO_BIAS, false);
        assert_eq!(out, [[2, 2], [10, -4]]);
    }

    #[test]
    fn append_sub8x8_block0_empty_neighbourhood_is_zero() {
        // block == 0 with no candidates: both slots ZeroMv (RefListMv is
        // all-zero, the dedup keeps slot 1 at zero too).
        let g = interior_geom();
        let src = TestSource::new(64);
        let out = g.append_sub8x8_mvs(&src, 0, LAST_FRAME, &NO_BLOCK_MVS, &NO_BIAS, false);
        assert_eq!(out, [ZERO_MV, ZERO_MV]);
    }

    #[test]
    fn append_sub8x8_block1_seeds_from_block_mv0_then_ref_list() {
        // block == 1 (<= 2): slot 0 = BlockMvs[refList][0]; slot 1 filled
        // from the first RefListMv that differs from slot 0.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [10, -4], NEARESTMV));
        let block_mvs = [[5, 5], [0, 0], [0, 0], [0, 0]];
        let out = g.append_sub8x8_mvs(&src, 1, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[5, 5], [10, -4]]);
    }

    #[test]
    fn append_sub8x8_block2_ref_list_dedup_drops_match() {
        // block == 2 (<= 2): slot 0 = BlockMvs[0]; a RefListMv equal to
        // slot 0 is skipped, the next distinct one fills slot 1.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        // RefListMv[0] duplicates BlockMvs[0]; RefListMv[1] is distinct.
        src.set(7, 8, Cell::inter(LAST_FRAME, [5, 5], NEARESTMV));
        src.set(8, 7, Cell::inter(LAST_FRAME, [9, 9], NEARESTMV));
        let block_mvs = [[5, 5], [0, 0], [0, 0], [0, 0]];
        let out = g.append_sub8x8_mvs(&src, 2, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[5, 5], [9, 9]]);
    }

    #[test]
    fn append_sub8x8_block1_zero_backfill_when_no_distinct_candidate() {
        // block == 1: slot 0 from BlockMvs[0]; no neighbour candidate and
        // the all-zero RefListMv duplicates nothing distinct, so the
        // ZeroMv backfill fills slot 1.
        let g = interior_geom();
        let src = TestSource::new(64);
        let block_mvs = [[7, -7], [0, 0], [0, 0], [0, 0]];
        let out = g.append_sub8x8_mvs(&src, 1, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[7, -7], ZERO_MV]);
    }

    #[test]
    fn append_sub8x8_block3_seeds_from_block_mv2_then_walks_1_0() {
        // block == 3 (else arm): slot 0 = BlockMvs[2]; then idx 1, 0 are
        // walked, each added if it differs from slot 0. BlockMvs[1] differs
        // and fills slot 1; the idx==0 step never runs (dst already 2).
        let g = interior_geom();
        let src = TestSource::new(64);
        let block_mvs = [[1, 1], [8, 8], [3, 3], [0, 0]];
        let out = g.append_sub8x8_mvs(&src, 3, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[3, 3], [8, 8]]); // slot0 = BlockMvs[2], slot1 = BlockMvs[1].
    }

    #[test]
    fn append_sub8x8_block3_skips_block_mv_matching_slot0() {
        // block == 3: BlockMvs[1] duplicates BlockMvs[2] (slot 0) so it is
        // skipped; BlockMvs[0] differs and fills slot 1.
        let g = interior_geom();
        let src = TestSource::new(64);
        let block_mvs = [[4, 4], [3, 3], [3, 3], [0, 0]];
        let out = g.append_sub8x8_mvs(&src, 3, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[3, 3], [4, 4]]); // slot0 = BlockMvs[2]; idx1 dup-skipped; idx0 fills.
    }

    #[test]
    fn append_sub8x8_block3_all_block_mvs_equal_then_ref_list_fills() {
        // block == 3: all BlockMvs equal slot 0, so the idx loop adds
        // nothing; a distinct RefListMv candidate fills slot 1 instead.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        src.set(7, 8, Cell::inter(LAST_FRAME, [12, -6], NEARESTMV));
        let block_mvs = [[2, 2], [2, 2], [2, 2], [2, 2]];
        let out = g.append_sub8x8_mvs(&src, 3, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[2, 2], [12, -6]]);
    }

    #[test]
    fn append_sub8x8_block3_zero_backfill_when_nothing_distinct() {
        // block == 3: all BlockMvs equal slot 0, no neighbour candidate,
        // RefListMv all zero — ZeroMv backfill fills slot 1.
        let g = interior_geom();
        let src = TestSource::new(64);
        let block_mvs = [[9, 9], [9, 9], [9, 9], [9, 9]];
        let out = g.append_sub8x8_mvs(&src, 3, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        assert_eq!(out, [[9, 9], ZERO_MV]);
    }

    #[test]
    fn append_sub8x8_block_passes_index_to_find_mv_refs() {
        // For block != 0 the find_mv_refs call still uses `block` as the
        // sub-block index. With block == 1 and a left neighbour carrying
        // distinct sub-block vectors, RefListMv[0] reads sub_mv index 1
        // (idx_n_column_to_subblock[1][0] for deltaCol != 0), proving the
        // block index is threaded through.
        let g = interior_geom();
        let mut src = TestSource::new(64);
        let mut cell = Cell::inter(LAST_FRAME, [0, 0], NEARESTMV);
        cell.sub_mv[0] = [[1, 1], [22, 22], [3, 3], [44, 44]];
        src.set(8, 7, cell); // left neighbour {0,-1}, deltaCol -1.
                             // block == 1: idx_n_column_to_subblock[1][deltaCol==0? :] -> [1,3];
                             // deltaCol != 0 -> column 0 -> sub index 1 -> [22,22].
        let block_mvs = [[100, 100], [0, 0], [0, 0], [0, 0]];
        let out = g.append_sub8x8_mvs(&src, 1, LAST_FRAME, &block_mvs, &NO_BIAS, false);
        // slot 0 from BlockMvs[0]; slot 1 from RefListMv[0] == [22,22].
        assert_eq!(out, [[100, 100], [22, 22]]);
    }
}
