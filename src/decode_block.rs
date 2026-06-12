//! VP9 §6.4.4 `decode_block( r, c, subsize )` driver — per spec v0.7.
//!
//! This module lands the §6.4.4 listing as a pure book-keeping primitive
//! that fans the per-MI outputs of `mode_info( )` and `residual( )` into
//! the frame-wide `Skips[ ]` / `TxSizes[ ]` / `MiSizes[ ]` / `YModes[ ]`
//! / `SegmentIds[ ]` / `RefFrames[ ]` / `InterpFilters[ ]` / `Mvs[ ]` /
//! `SubMvs[ ]` / `SubModes[ ]` 2D arrays at every `(r + y, c + x)` cell
//! for `y ∈ 0..num_8x8_blocks_high_lookup[ subsize ]`,
//! `x ∈ 0..num_8x8_blocks_wide_lookup[ subsize ]`.
//!
//! The §6.4.4 listing (`vp9-spec.txt` lines 2395-2437) is reproduced
//! verbatim by this implementation:
//!
//! ```text
//! decode_block( r, c, subsize ) {
//!     MiRow = r
//!     MiCol = c
//!     MiSize = subsize
//!     AvailU = r > 0
//!     AvailL = c > MiColStart
//!     mode_info( )
//!     EobTotal = 0
//!     residual( )
//!     if ( is_inter && subsize >= BLOCK_8X8 && EobTotal == 0 ) {
//!         skip = 1
//!     }
//!     for ( y = 0; y < num_8x8_blocks_high_lookup[ subsize ]; y++ )
//!         for ( x = 0; x < num_8x8_blocks_wide_lookup[ subsize ]; x++ ) {
//!             Skips[ r + y ][ c + x ] = skip
//!             TxSizes[ r + y ][ c + x ] = tx_size
//!             MiSizes[ r + y ][ c + x ] = MiSize
//!             YModes[ r + y ][ c + x ] = y_mode
//!             SegmentIds[ r + y ][ c + x ] = segment_id
//!             for( refList = 0; refList < 2; refList++ )
//!                 RefFrames[ r + y ][ c + x ][ refList ] = ref_frame[ refList ]
//!             if ( is_inter ) {
//!                 InterpFilters[ r + y ][ c + x ] = interp_filter
//!                 for( refList = 0; refList < 2; refList++ ) {
//!                     Mvs[ r + y ][ c + x ][ refList ] = BlockMvs[ refList ][ 3 ]
//!                     for( b = 0; b < 4; b++ )
//!                         SubMvs[ r + y ][ c + x ][ refList ][ b ] = BlockMvs[ refList ][ b ]
//!                 }
//!             } else {
//!                 for( b = 0; b < 4; b++ )
//!                     SubModes[ r + y ][ c + x ][ b ] = sub_modes[ b ]
//!             }
//!         }
//! }
//! ```
//!
//! ## Scope of this round
//!
//! The §6.4.4 listing has two halves:
//!
//! 1. The **header** (lines 2397-2407) — assignments of `MiRow` / `MiCol`
//!    / `MiSize` / `AvailU` / `AvailL`, plus the `mode_info( )` and
//!    `residual( )` invocations and the `is_inter && subsize >= BLOCK_8X8
//!    && EobTotal == 0 ⇒ skip = 1` rule. The `mode_info( )` and
//!    `residual( )` primitives themselves were landed in earlier rounds
//!    (§6.4.5 / §6.4.6 / §6.4.15 / §6.4.21).
//! 2. The **fan-out** (lines 2408-2436) — the `num_8x8_blocks_*_lookup`
//!    write-back loop that propagates the per-MI decoded values into the
//!    frame-wide arrays.
//!
//! This round wires the §6.4.4 driver as a pure-state primitive: the
//! caller supplies the per-MI `DecodedBlockResult` produced by
//! `mode_info( ) + residual( )`, and [`decode_block_apply`] runs the
//! header's `skip` rewrite plus the fan-out loop against a
//! [`Vp9FrameState`] that owns the §6.4.4 frame-wide arrays.
//!
//! The round-284 `decode_frame` module wires `decode_block_apply`
//! into the §6.4.3 partition walk: its `BlockDecoder` leaf sink runs
//! the §6.4.6 mode-info + §6.4.21 residual decode inline and applies
//! this driver's fan-out at every leaf.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §6.4.4 lines 2395-2437; §3 `BLOCK_*`
//! line 462; §10.2 `num_8x8_blocks_wide_lookup[ ]` /
//! `num_8x8_blocks_high_lookup[ ]` lines 7111 / 7117). The §3
//! `NONE = -1` sentinel for `ref_frame[ 1 ]` on intra blocks is the
//! [`crate::mode_info::NONE_REF_FRAME`] constant.

#![allow(dead_code)] // surfaces are pub(crate); the inter-only fields
                     // (Mvs / SubMvs / InterpFilters accessors) have
                     // no production reader until the inter-frame
                     // rounds land — the intra path is wired via the
                     // round-284 decode_frame module.

use crate::mode_info::NONE_REF_FRAME;
use crate::partition::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};
use crate::residual::BLOCK_8X8;

/// Per-block decoded values produced by §6.4.5 `mode_info( )` plus
/// §6.4.21 `residual( )`.
///
/// This struct bundles every field the §6.4.4 listing (lines 2410-2426)
/// reads after the `mode_info( )` / `residual( )` calls return:
///
/// * `skip` — the §6.4.8 `read_skip( )` flag (which §6.4.4 line 2405-2407
///   may rewrite to `1` per the `is_inter && subsize >= BLOCK_8X8 &&
///   EobTotal == 0` rule).
/// * `tx_size` — the §6.4.10 / §7.4.3 transform-block size (one of
///   `TX_4X4 = 0` / `TX_8X8 = 1` / `TX_16X16 = 2` / `TX_32X32 = 3`).
/// * `y_mode` — the §6.4.5 / §6.4.6 / §6.4.15 final luma intra-mode
///   value (one of the §3 `INTRA_MODES` slots `DC_PRED..TM_PRED`); on
///   inter blocks the §7.4.5 listings treat `y_mode` as an `inter_mode`
///   value (`NEARESTMV..NEWMV`) but the §6.4.4 write-back lands the
///   raw decoded value regardless.
/// * `segment_id` — §6.4.7 / §6.4.12 segment id (0..MAX_SEGMENTS-1).
/// * `ref_frame` — §3 `ref_frame[ 2 ]` per-block reference pair.
///   `ref_frame[ 0 ] = INTRA_FRAME = 0` and `ref_frame[ 1 ] = NONE = -1`
///   on intra blocks (§6.4.6 lines 2469-2470); on inter blocks
///   `ref_frame[ 0 ]` is `LAST_FRAME` / `GOLDEN_FRAME` / `ALTREF_FRAME`
///   and `ref_frame[ 1 ]` is one of those values OR `NONE` for
///   single-reference blocks (§6.4.11).
/// * `is_inter` — §6.4.13 `read_is_inter( )` boolean; `0` on intra
///   frames or intra blocks of inter frames, `1` on inter blocks.
/// * `eob_total` — §6.4.21 `residual( )` accumulator of end-of-block
///   counts across all transform blocks of all planes. Drives the
///   §6.4.4 `skip = 1` rewrite when zero.
/// * `interp_filter` — §6.4.16 / §7.4.10 interpolation-filter selection
///   (one of `EIGHTTAP_SMOOTH = 0` / `EIGHTTAP = 1` /
///   `EIGHTTAP_SHARP = 2` / `BILINEAR = 3`; only read when `is_inter`).
/// * `block_mvs` — §6.4.19 `BlockMvs[ 2 ][ 4 ]` per-sub-block motion
///   vectors. The §6.4.4 fan-out writes the last entry
///   `BlockMvs[ refList ][ 3 ]` into `Mvs[ ][ ][ refList ]` and the
///   full 4-entry slice into `SubMvs[ ][ ][ refList ][ 0..4 ]`. Only
///   read when `is_inter`.
/// * `sub_modes` — §6.4.6 / §6.4.15 `sub_modes[ 4 ]` per-4x4-luma
///   sub-block intra mode (only read when `!is_inter`; the §6.4.6
///   listing replicates `y_mode` into all four entries for
///   `MiSize >= BLOCK_8X8`, and walks the §6.4.6 `(idy, idx)` grid for
///   `MiSize < BLOCK_8X8`).
///
/// The `Mv` and motion-vector representation reuses a plain
/// `(i16, i16)` (row, col) pair so this module does not need to know
/// about §6.5 `mv_joint_decode` internals — the §6.4.4 fan-out is just
/// a memcpy of the per-`refList` array slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedBlockResult {
    /// `skip` per §6.4.4 line 2410.
    pub skip: bool,
    /// `tx_size` per §6.4.4 line 2411.
    pub tx_size: u8,
    /// `y_mode` per §6.4.4 line 2413.
    pub y_mode: u8,
    /// `segment_id` per §6.4.4 line 2414.
    pub segment_id: u8,
    /// `ref_frame[ 2 ]` per §6.4.4 line 2416.
    pub ref_frame: [i32; 2],
    /// `is_inter` per §6.4.4 line 2405 / 2417.
    pub is_inter: bool,
    /// `EobTotal` accumulator per §6.4.4 lines 2403-2405. Drives the
    /// `skip = 1` rewrite when zero on `is_inter` blocks of size
    /// `>= BLOCK_8X8`.
    pub eob_total: u32,
    /// `interp_filter` per §6.4.4 line 2418 (read only when `is_inter`).
    pub interp_filter: u8,
    /// `BlockMvs[ 2 ][ 4 ]` per §6.4.4 lines 2420-2422 (read only when
    /// `is_inter`). Outer index is `refList ∈ {0, 1}`, inner index is
    /// the per-sub-block index `b ∈ {0, 1, 2, 3}`. The §6.4.4 listing
    /// writes `Mvs[ r + y ][ c + x ][ refList ] = BlockMvs[ refList ][ 3 ]`
    /// (the last entry) and the full slice into `SubMvs[ ][ ][ refList ]`.
    pub block_mvs: [[(i16, i16); 4]; 2],
    /// `sub_modes[ 4 ]` per §6.4.4 line 2426 (read only when `!is_inter`).
    pub sub_modes: [u8; 4],
}

impl Default for DecodedBlockResult {
    /// All-zero default with `ref_frame = [INTRA_FRAME = 0, NONE = -1]`
    /// per §6.4.6 intra-block init lines 2469-2470. This matches the
    /// state the §6.4.4 listing reads when called on a §6.4.6
    /// `intra_frame_mode_info( )`-decoded intra block.
    fn default() -> Self {
        Self {
            skip: false,
            tx_size: 0,
            y_mode: 0,
            segment_id: 0,
            ref_frame: [0, NONE_REF_FRAME],
            is_inter: false,
            eob_total: 0,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [0; 4],
        }
    }
}

/// Frame-wide §6.4.4 write-back arrays.
///
/// Each array is sized `MiRows × MiCols` (in row-major order) and
/// indexed by `[row * mi_cols + col]`. The §6.4.4 listing writes one
/// `[r + y][c + x]` cell per `(y, x)` step of the
/// `num_8x8_blocks_*_lookup[ subsize ]` fan-out loop.
///
/// The motion-vector slots `mvs` / `sub_mvs` carry one `(i16, i16)`
/// per `refList ∈ {0, 1}` per MI; intra blocks leave them zero (the
/// §6.4.4 listing only writes them on `is_inter`).
///
/// `sub_modes` carries four `u8` per MI; inter blocks leave them zero
/// (the §6.4.4 listing only writes them on `!is_inter`).
///
/// `ref_frames` carries two `i32` per MI; both branches write it (the
/// fan-out's outer `refList` loop on lines 2415-2416 runs
/// unconditionally).
#[derive(Debug, Clone)]
pub(crate) struct Vp9FrameState {
    /// Number of 8x8 MI columns per row — `MiCols` per §6.2.6 line 1769.
    pub mi_cols: u32,
    /// Number of 8x8 MI rows — `MiRows` per §6.2.6 line 1768.
    pub mi_rows: u32,
    /// `Skips[ MiRows ][ MiCols ]` per §6.4.4 line 2410.
    pub skips: Vec<u8>,
    /// `TxSizes[ MiRows ][ MiCols ]` per §6.4.4 line 2411.
    pub tx_sizes: Vec<u8>,
    /// `MiSizes[ MiRows ][ MiCols ]` per §6.4.4 line 2412.
    pub mi_sizes: Vec<u8>,
    /// `YModes[ MiRows ][ MiCols ]` per §6.4.4 line 2413.
    pub y_modes: Vec<u8>,
    /// `SegmentIds[ MiRows ][ MiCols ]` per §6.4.4 line 2414.
    pub segment_ids: Vec<u8>,
    /// `RefFrames[ MiRows ][ MiCols ][ 2 ]` per §6.4.4 line 2416.
    /// Stored as `[row * mi_cols * 2 + col * 2 + refList]`.
    pub ref_frames: Vec<i32>,
    /// `InterpFilters[ MiRows ][ MiCols ]` per §6.4.4 line 2418 (only
    /// written on `is_inter`).
    pub interp_filters: Vec<u8>,
    /// `Mvs[ MiRows ][ MiCols ][ 2 ]` — last-of-four per `refList`
    /// — per §6.4.4 line 2420. Stored as
    /// `[row * mi_cols * 2 + col * 2 + refList]`.
    pub mvs: Vec<(i16, i16)>,
    /// `SubMvs[ MiRows ][ MiCols ][ 2 ][ 4 ]` per §6.4.4 line 2422.
    /// Stored as `[((row * mi_cols + col) * 2 + refList) * 4 + b]`.
    pub sub_mvs: Vec<(i16, i16)>,
    /// `SubModes[ MiRows ][ MiCols ][ 4 ]` per §6.4.4 line 2426 (only
    /// written on `!is_inter`). Stored as
    /// `[(row * mi_cols + col) * 4 + b]`.
    pub sub_modes: Vec<u8>,
}

impl Vp9FrameState {
    /// Allocate the frame-wide §6.4.4 arrays for an `MiRows × MiCols`
    /// frame. All cells start at zero; `ref_frames` cells start at
    /// `INTRA_FRAME = 0` (single zero is a valid intra `ref_frame[ 0 ]`
    /// and the §6.4.4 fan-out overwrites every cell that is decoded).
    pub fn new(mi_rows: u32, mi_cols: u32) -> Self {
        let n = (mi_rows as usize) * (mi_cols as usize);
        Self {
            mi_cols,
            mi_rows,
            skips: vec![0u8; n],
            tx_sizes: vec![0u8; n],
            mi_sizes: vec![0u8; n],
            y_modes: vec![0u8; n],
            segment_ids: vec![0u8; n],
            ref_frames: vec![0i32; n * 2],
            interp_filters: vec![0u8; n],
            mvs: vec![(0i16, 0i16); n * 2],
            sub_mvs: vec![(0i16, 0i16); n * 2 * 4],
            sub_modes: vec![0u8; n * 4],
        }
    }

    /// Row-major MI index for `(row, col)`. Returns `None` if the
    /// coordinate is outside `MiRows × MiCols`.
    fn mi_index(&self, row: u32, col: u32) -> Option<usize> {
        if row < self.mi_rows && col < self.mi_cols {
            Some((row as usize) * (self.mi_cols as usize) + (col as usize))
        } else {
            None
        }
    }

    /// Read `Skips[ row ][ col ]`. Returns `None` for out-of-frame.
    pub fn get_skip(&self, row: u32, col: u32) -> Option<u8> {
        self.mi_index(row, col).map(|i| self.skips[i])
    }

    /// Read `TxSizes[ row ][ col ]`. Returns `None` for out-of-frame.
    pub fn get_tx_size(&self, row: u32, col: u32) -> Option<u8> {
        self.mi_index(row, col).map(|i| self.tx_sizes[i])
    }

    /// Read `MiSizes[ row ][ col ]`. Returns `None` for out-of-frame.
    pub fn get_mi_size(&self, row: u32, col: u32) -> Option<u8> {
        self.mi_index(row, col).map(|i| self.mi_sizes[i])
    }

    /// Read `YModes[ row ][ col ]`. Returns `None` for out-of-frame.
    pub fn get_y_mode(&self, row: u32, col: u32) -> Option<u8> {
        self.mi_index(row, col).map(|i| self.y_modes[i])
    }

    /// Read `SegmentIds[ row ][ col ]`. Returns `None` for out-of-frame.
    pub fn get_segment_id(&self, row: u32, col: u32) -> Option<u8> {
        self.mi_index(row, col).map(|i| self.segment_ids[i])
    }

    /// Read `RefFrames[ row ][ col ][ ref_list ]`. Returns `None` for
    /// out-of-frame or `ref_list > 1`.
    pub fn get_ref_frame(&self, row: u32, col: u32, ref_list: usize) -> Option<i32> {
        if ref_list > 1 {
            return None;
        }
        self.mi_index(row, col)
            .map(|i| self.ref_frames[i * 2 + ref_list])
    }

    /// Read `InterpFilters[ row ][ col ]`. Returns `None` for
    /// out-of-frame.
    pub fn get_interp_filter(&self, row: u32, col: u32) -> Option<u8> {
        self.mi_index(row, col).map(|i| self.interp_filters[i])
    }

    /// Read `Mvs[ row ][ col ][ ref_list ]`. Returns `None` for
    /// out-of-frame or `ref_list > 1`.
    pub fn get_mv(&self, row: u32, col: u32, ref_list: usize) -> Option<(i16, i16)> {
        if ref_list > 1 {
            return None;
        }
        self.mi_index(row, col).map(|i| self.mvs[i * 2 + ref_list])
    }

    /// Read `SubMvs[ row ][ col ][ ref_list ][ b ]`. Returns `None`
    /// for out-of-frame, `ref_list > 1`, or `b > 3`.
    pub fn get_sub_mv(&self, row: u32, col: u32, ref_list: usize, b: usize) -> Option<(i16, i16)> {
        if ref_list > 1 || b > 3 {
            return None;
        }
        self.mi_index(row, col)
            .map(|i| self.sub_mvs[(i * 2 + ref_list) * 4 + b])
    }

    /// Read `SubModes[ row ][ col ][ b ]`. Returns `None` for
    /// out-of-frame or `b > 3`.
    pub fn get_sub_mode(&self, row: u32, col: u32, b: usize) -> Option<u8> {
        if b > 3 {
            return None;
        }
        self.mi_index(row, col).map(|i| self.sub_modes[i * 4 + b])
    }
}

/// Apply the §6.4.4 `decode_block( r, c, subsize )` driver per
/// `vp9-spec.txt` lines 2395-2437.
///
/// `state` carries the §6.4.4 frame-wide arrays
/// (`Skips` / `TxSizes` / `MiSizes` / `YModes` / `SegmentIds` /
/// `RefFrames` / `InterpFilters` / `Mvs` / `SubMvs` / `SubModes`).
/// `result` carries the per-MI decoded values produced upstream by
/// `mode_info( )` and `residual( )`.
///
/// Performs in order:
///
/// 1. The §6.4.4 line 2405-2407 `skip` rewrite: if `is_inter && subsize
///    >= BLOCK_8X8 && EobTotal == 0` set `skip = 1`.
/// 2. The §6.4.4 lines 2408-2436 fan-out: for `y ∈ 0..bh`, `x ∈ 0..bw`
///    with `bh = num_8x8_blocks_high_lookup[ subsize ]` and
///    `bw = num_8x8_blocks_wide_lookup[ subsize ]`, write the per-MI
///    cells `Skips[ r + y ][ c + x ]` … `SubMvs[ r + y ][ c + x ][ ][ ]`
///    / `SubModes[ r + y ][ c + x ][ ]`.
///
/// Returns the effective `skip` value (after the §6.4.4 line 2405-2407
/// rewrite), which the caller may want for downstream §8.4
/// probability-adaption counts.
///
/// `(r, c)` is the MI-coordinate of the block's top-left corner;
/// `subsize` is the §3 `BLOCK_*` constant (`0..=12`); out-of-frame
/// cells (`r + y >= MiRows` or `c + x >= MiCols`) are silently
/// skipped per the §7.4.3 conformance assumption that a valid
/// bitstream never decodes an out-of-frame block.
///
/// # Panics
///
/// Panics if `subsize >= BLOCK_SIZES = 13` (i.e. `subsize` is not a
/// valid §3 `BLOCK_*` constant).
pub(crate) fn decode_block_apply(
    state: &mut Vp9FrameState,
    r: u32,
    c: u32,
    subsize: u8,
    result: &DecodedBlockResult,
) -> bool {
    // §6.4.4 lines 2405-2407: skip rewrite.
    let skip = if result.is_inter && subsize >= BLOCK_8X8 && result.eob_total == 0 {
        true
    } else {
        result.skip
    };

    // §6.4.4 lines 2408-2409: fan-out loop bounds.
    let bh = NUM_8X8_BLOCKS_HIGH_LOOKUP[subsize as usize] as u32;
    let bw = NUM_8X8_BLOCKS_WIDE_LOOKUP[subsize as usize] as u32;

    for y in 0..bh {
        for x in 0..bw {
            let row = r + y;
            let col = c + x;
            // Silent out-of-frame skip: §7.4.3 conformance assumes a
            // valid bitstream never decodes an MI cell outside
            // MiRows × MiCols, but a defensive bounds-check keeps the
            // decoder useful on partial / truncated streams.
            let idx = match state.mi_index(row, col) {
                Some(i) => i,
                None => continue,
            };

            // §6.4.4 lines 2410-2414: scalar writes.
            state.skips[idx] = skip as u8;
            state.tx_sizes[idx] = result.tx_size;
            state.mi_sizes[idx] = subsize;
            state.y_modes[idx] = result.y_mode;
            state.segment_ids[idx] = result.segment_id;

            // §6.4.4 lines 2415-2416: RefFrames[ ][ ][ refList ] —
            // both branches write this.
            state.ref_frames[idx * 2] = result.ref_frame[0];
            state.ref_frames[idx * 2 + 1] = result.ref_frame[1];

            if result.is_inter {
                // §6.4.4 lines 2417-2423: inter branch.
                state.interp_filters[idx] = result.interp_filter;
                for ref_list in 0..2 {
                    // Mvs[ ][ ][ refList ] = BlockMvs[ refList ][ 3 ]
                    state.mvs[idx * 2 + ref_list] = result.block_mvs[ref_list][3];
                    // SubMvs[ ][ ][ refList ][ b ] = BlockMvs[ refList ][ b ]
                    for b in 0..4 {
                        state.sub_mvs[(idx * 2 + ref_list) * 4 + b] = result.block_mvs[ref_list][b];
                    }
                }
            } else {
                // §6.4.4 lines 2424-2426: intra branch.
                for b in 0..4 {
                    state.sub_modes[idx * 4 + b] = result.sub_modes[b];
                }
            }
        }
    }

    skip
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode_info::{ALTREF_FRAME, DC_PRED, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, V_PRED};
    use crate::residual::{BLOCK_16X16, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8};

    /// §6.4.4 line 2410: a default-initialised result on an intra block
    /// writes zero into `Skips[ r ][ c ]`. The base case.
    #[test]
    fn intra_default_writes_zero_skip_into_top_left_cell() {
        let mut state = Vp9FrameState::new(8, 8);
        let result = DecodedBlockResult::default();
        let skip = decode_block_apply(&mut state, 0, 0, BLOCK_8X8, &result);
        assert!(!skip);
        assert_eq!(state.get_skip(0, 0), Some(0));
        // Intra `ref_frame[0] = INTRA_FRAME = 0`, `ref_frame[1] = NONE = -1`
        // per Default::default().
        assert_eq!(state.get_ref_frame(0, 0, 0), Some(INTRA_FRAME));
        assert_eq!(state.get_ref_frame(0, 0, 1), Some(NONE_REF_FRAME));
    }

    /// §6.4.4 lines 2410-2414 / 2425-2426: an intra block of size
    /// `BLOCK_8X8` writes exactly one MI cell (1×1 fan-out) with the
    /// scalar values plus the four-entry `sub_modes[ ]`.
    #[test]
    fn intra_block_8x8_writes_one_cell_with_sub_modes() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: true,
            tx_size: 1,
            y_mode: V_PRED,
            segment_id: 2,
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            is_inter: false,
            eob_total: 7,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [V_PRED, DC_PRED, V_PRED, DC_PRED],
        };
        decode_block_apply(&mut state, 1, 2, BLOCK_8X8, &result);
        // The targeted cell.
        assert_eq!(state.get_skip(1, 2), Some(1));
        assert_eq!(state.get_tx_size(1, 2), Some(1));
        assert_eq!(state.get_mi_size(1, 2), Some(BLOCK_8X8));
        assert_eq!(state.get_y_mode(1, 2), Some(V_PRED));
        assert_eq!(state.get_segment_id(1, 2), Some(2));
        assert_eq!(state.get_sub_mode(1, 2, 0), Some(V_PRED));
        assert_eq!(state.get_sub_mode(1, 2, 1), Some(DC_PRED));
        assert_eq!(state.get_sub_mode(1, 2, 2), Some(V_PRED));
        assert_eq!(state.get_sub_mode(1, 2, 3), Some(DC_PRED));
        // Untouched neighbour: §6.4.4 only writes `(r..r+bh, c..c+bw)`.
        assert_eq!(state.get_y_mode(0, 0), Some(0));
        assert_eq!(state.get_y_mode(1, 1), Some(0));
        assert_eq!(state.get_y_mode(2, 2), Some(0));
        // §6.4.4 line 2426 only writes `SubModes[ ][ ]` on the intra
        // branch — `interp_filter` and motion vectors stay zero.
        assert_eq!(state.get_interp_filter(1, 2), Some(0));
        assert_eq!(state.get_mv(1, 2, 0), Some((0, 0)));
        assert_eq!(state.get_mv(1, 2, 1), Some((0, 0)));
    }

    /// §6.4.4 line 2412: a 16x16 intra block fans the same `MiSize`
    /// into all 2x2 = 4 MI cells, leaving the surrounding cells
    /// untouched.
    #[test]
    fn intra_block_16x16_writes_2x2_fanout() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 2,
            y_mode: DC_PRED,
            segment_id: 1,
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            is_inter: false,
            eob_total: 3,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [DC_PRED; 4],
        };
        decode_block_apply(&mut state, 1, 1, BLOCK_16X16, &result);
        // The 4 targeted cells (1,1)..(2,2).
        for r in 1..=2 {
            for c in 1..=2 {
                assert_eq!(state.get_mi_size(r, c), Some(BLOCK_16X16));
                assert_eq!(state.get_y_mode(r, c), Some(DC_PRED));
                assert_eq!(state.get_segment_id(r, c), Some(1));
                assert_eq!(state.get_tx_size(r, c), Some(2));
            }
        }
        // Untouched neighbours.
        assert_eq!(state.get_mi_size(0, 0), Some(0));
        assert_eq!(state.get_mi_size(3, 3), Some(0));
        assert_eq!(state.get_mi_size(0, 1), Some(0));
        assert_eq!(state.get_mi_size(1, 0), Some(0));
    }

    /// §6.4.4 lines 2408-2409: a 64x64 intra block fans across the
    /// `num_8x8_blocks_*_lookup[ BLOCK_64X64 ] = 8` × `8 = 64`-cell
    /// grid — i.e. all 64 cells of an 8x8 MI frame.
    #[test]
    fn intra_block_64x64_writes_full_8x8_mi_frame() {
        let mut state = Vp9FrameState::new(8, 8);
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 3,
            y_mode: V_PRED,
            segment_id: 0,
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            is_inter: false,
            eob_total: 10,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [V_PRED; 4],
        };
        decode_block_apply(&mut state, 0, 0, BLOCK_64X64, &result);
        let mut covered = 0usize;
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(state.get_mi_size(r, c), Some(BLOCK_64X64));
                assert_eq!(state.get_y_mode(r, c), Some(V_PRED));
                covered += 1;
            }
        }
        assert_eq!(covered, 64);
    }

    /// §6.4.4 line 2405-2407: the `is_inter && subsize >= BLOCK_8X8 &&
    /// EobTotal == 0` rule rewrites `skip` to `1` even when the input
    /// arrived as `skip = false`.
    #[test]
    fn inter_skip_rewrite_fires_when_eob_zero_and_size_8x8_or_larger() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: false, // arrived from §6.4.8 read_skip( )
            tx_size: 1,
            y_mode: 0,
            segment_id: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            is_inter: true,
            eob_total: 0, // no nonzero coefficients
            interp_filter: 1,
            block_mvs: [[(1, 2); 4], [(0, 0); 4]],
            sub_modes: [0; 4],
        };
        let skip = decode_block_apply(&mut state, 0, 0, BLOCK_8X8, &result);
        assert!(skip);
        assert_eq!(state.get_skip(0, 0), Some(1));
    }

    /// §6.4.4 line 2405-2407: the rule does NOT fire when
    /// `subsize < BLOCK_8X8` (sub-8x8 blocks always retain decoded
    /// `skip`).
    #[test]
    fn inter_skip_rewrite_does_not_fire_for_sub_8x8() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 0,
            y_mode: 0,
            segment_id: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            is_inter: true,
            eob_total: 0,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [0; 4],
        };
        let skip = decode_block_apply(&mut state, 0, 0, BLOCK_4X4, &result);
        assert!(!skip);
        assert_eq!(state.get_skip(0, 0), Some(0));
    }

    /// §6.4.4 line 2405-2407: the rule does NOT fire when
    /// `EobTotal > 0` (nonzero coefficients block the rewrite).
    #[test]
    fn inter_skip_rewrite_does_not_fire_when_eob_positive() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 1,
            y_mode: 0,
            segment_id: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            is_inter: true,
            eob_total: 1, // at least one nonzero coefficient
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [0; 4],
        };
        let skip = decode_block_apply(&mut state, 0, 0, BLOCK_8X8, &result);
        assert!(!skip);
        assert_eq!(state.get_skip(0, 0), Some(0));
    }

    /// §6.4.4 line 2405-2407: the rule does NOT fire on intra blocks
    /// regardless of `EobTotal` or size.
    #[test]
    fn intra_skip_rewrite_does_not_fire() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 1,
            y_mode: V_PRED,
            segment_id: 0,
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            is_inter: false,
            eob_total: 0,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [V_PRED; 4],
        };
        let skip = decode_block_apply(&mut state, 0, 0, BLOCK_64X64, &result);
        assert!(!skip);
    }

    /// §6.4.4 lines 2417-2423: inter branch writes `InterpFilters[ ]`,
    /// `Mvs[ ][ ][ refList ]` and `SubMvs[ ][ ][ refList ][ b ]`.
    /// The `Mvs[ ][ ][ refList ]` cell carries `BlockMvs[ refList ][ 3 ]`
    /// (the last sub-block index per the spec listing).
    #[test]
    fn inter_branch_writes_mvs_interp_filter_and_sub_mvs() {
        let mut state = Vp9FrameState::new(4, 4);
        let block_mvs = [
            [(10, 20), (30, 40), (50, 60), (70, 80)],
            [(-10, -20), (-30, -40), (-50, -60), (-70, -80)],
        ];
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 2,
            y_mode: 0,
            segment_id: 0,
            ref_frame: [LAST_FRAME, ALTREF_FRAME],
            is_inter: true,
            eob_total: 5,
            interp_filter: 2,
            block_mvs,
            sub_modes: [0; 4],
        };
        decode_block_apply(&mut state, 1, 1, BLOCK_16X16, &result);
        for r in 1..=2 {
            for c in 1..=2 {
                assert_eq!(state.get_interp_filter(r, c), Some(2));
                // `Mvs[ r ][ c ][ 0 ] = BlockMvs[ 0 ][ 3 ] = (70, 80)`.
                assert_eq!(state.get_mv(r, c, 0), Some((70, 80)));
                // `Mvs[ r ][ c ][ 1 ] = BlockMvs[ 1 ][ 3 ] = (-70, -80)`.
                assert_eq!(state.get_mv(r, c, 1), Some((-70, -80)));
                // `SubMvs[ r ][ c ][ 0 ][ 0..3 ] = BlockMvs[ 0 ][ 0..3 ]`.
                assert_eq!(state.get_sub_mv(r, c, 0, 0), Some((10, 20)));
                assert_eq!(state.get_sub_mv(r, c, 0, 1), Some((30, 40)));
                assert_eq!(state.get_sub_mv(r, c, 0, 2), Some((50, 60)));
                assert_eq!(state.get_sub_mv(r, c, 0, 3), Some((70, 80)));
                assert_eq!(state.get_sub_mv(r, c, 1, 0), Some((-10, -20)));
                assert_eq!(state.get_sub_mv(r, c, 1, 1), Some((-30, -40)));
                assert_eq!(state.get_sub_mv(r, c, 1, 2), Some((-50, -60)));
                assert_eq!(state.get_sub_mv(r, c, 1, 3), Some((-70, -80)));
                // `SubModes[ ][ ]` not written on inter branch.
                assert_eq!(state.get_sub_mode(r, c, 0), Some(0));
            }
        }
    }

    /// §6.4.4 line 2416: both `ref_frame[ 0 ]` and `ref_frame[ 1 ]` are
    /// written even on intra blocks (the §6.4.4 outer `refList` loop
    /// runs unconditionally; only the `is_inter ⇒ Mvs/SubMvs` branch
    /// gates).
    #[test]
    fn ref_frames_written_on_both_branches() {
        let mut state = Vp9FrameState::new(4, 4);
        // Intra block: ref_frame[0] = INTRA_FRAME = 0, [1] = NONE = -1.
        let intra = DecodedBlockResult {
            ref_frame: [INTRA_FRAME, NONE_REF_FRAME],
            ..DecodedBlockResult::default()
        };
        decode_block_apply(&mut state, 0, 0, BLOCK_8X8, &intra);
        assert_eq!(state.get_ref_frame(0, 0, 0), Some(INTRA_FRAME));
        assert_eq!(state.get_ref_frame(0, 0, 1), Some(NONE_REF_FRAME));
        // Inter block: ref_frame[0] = LAST, [1] = GOLDEN.
        let inter = DecodedBlockResult {
            is_inter: true,
            ref_frame: [LAST_FRAME, GOLDEN_FRAME],
            ..DecodedBlockResult::default()
        };
        decode_block_apply(&mut state, 1, 1, BLOCK_8X8, &inter);
        assert_eq!(state.get_ref_frame(1, 1, 0), Some(LAST_FRAME));
        assert_eq!(state.get_ref_frame(1, 1, 1), Some(GOLDEN_FRAME));
    }

    /// §6.4.4 lines 2408-2436 plus §7.4.3 conformance: out-of-frame
    /// cells (`r + y >= MiRows` or `c + x >= MiCols`) are silently
    /// skipped. Tests a 32x32 block (4x4 fan-out) starting at (6, 6)
    /// in an 8x8 MI frame — only the (6, 6), (6, 7), (7, 6), (7, 7)
    /// cells are written; (6, 8)+ and (8+, 6)+ are not even attempted.
    #[test]
    fn fanout_clips_at_frame_edge() {
        let mut state = Vp9FrameState::new(8, 8);
        let result = DecodedBlockResult {
            tx_size: 3,
            y_mode: V_PRED,
            ..DecodedBlockResult::default()
        };
        // BLOCK_32X32 = 9 (§3 / §10.2 ordering): num_8x8 = 4x4.
        let block_32x32: u8 = 9;
        decode_block_apply(&mut state, 6, 6, block_32x32, &result);
        // In-frame cells written.
        assert_eq!(state.get_mi_size(6, 6), Some(block_32x32));
        assert_eq!(state.get_mi_size(6, 7), Some(block_32x32));
        assert_eq!(state.get_mi_size(7, 6), Some(block_32x32));
        assert_eq!(state.get_mi_size(7, 7), Some(block_32x32));
        // Out-of-frame cells: get_* returns None.
        assert_eq!(state.get_mi_size(6, 8), None);
        assert_eq!(state.get_mi_size(8, 6), None);
        // The in-bounds (7, 7) was the only edge cell.
        for r in 0..6u32 {
            for c in 0..8u32 {
                assert_eq!(state.get_mi_size(r, c), Some(0));
            }
        }
    }

    /// §6.4.4 line 2410: the §6.4.4 `skip` write-back uses the
    /// *rewritten* `skip` value, so the inter-skip rule propagates
    /// through the fan-out — every cell of the 16x16 block sees
    /// `Skips[ ][ ] = 1`.
    #[test]
    fn inter_skip_rewrite_propagates_into_all_cells_of_fanout() {
        let mut state = Vp9FrameState::new(4, 4);
        let result = DecodedBlockResult {
            skip: false,
            tx_size: 1,
            y_mode: 0,
            segment_id: 0,
            ref_frame: [LAST_FRAME, NONE_REF_FRAME],
            is_inter: true,
            eob_total: 0,
            interp_filter: 0,
            block_mvs: [[(0, 0); 4]; 2],
            sub_modes: [0; 4],
        };
        decode_block_apply(&mut state, 1, 1, BLOCK_16X16, &result);
        for r in 1..=2 {
            for c in 1..=2 {
                assert_eq!(state.get_skip(r, c), Some(1));
            }
        }
    }

    /// §10.2 `num_8x8_blocks_*_lookup[ ]` pinning — sanity check that
    /// the table values [`crate::partition::NUM_8X8_BLOCKS_WIDE_LOOKUP`]
    /// / [`crate::partition::NUM_8X8_BLOCKS_HIGH_LOOKUP`] match the
    /// §10.2 spec listing for the §6.4.4 fan-out. Acts as a
    /// belt-and-braces guard against a future edit to the lookup tables
    /// breaking the §6.4.4 driver silently.
    #[test]
    fn lookup_table_pinning() {
        // §10.2 line 7111: { 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8 }.
        assert_eq!(NUM_8X8_BLOCKS_WIDE_LOOKUP[0], 1); // BLOCK_4X4
        assert_eq!(NUM_8X8_BLOCKS_WIDE_LOOKUP[3], 1); // BLOCK_8X8
        assert_eq!(NUM_8X8_BLOCKS_WIDE_LOOKUP[6], 2); // BLOCK_16X16
        assert_eq!(NUM_8X8_BLOCKS_WIDE_LOOKUP[9], 4); // BLOCK_32X32
        assert_eq!(NUM_8X8_BLOCKS_WIDE_LOOKUP[12], 8); // BLOCK_64X64

        // §10.2 line 7117: { 1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8 }.
        assert_eq!(NUM_8X8_BLOCKS_HIGH_LOOKUP[0], 1); // BLOCK_4X4
        assert_eq!(NUM_8X8_BLOCKS_HIGH_LOOKUP[3], 1); // BLOCK_8X8
        assert_eq!(NUM_8X8_BLOCKS_HIGH_LOOKUP[6], 2); // BLOCK_16X16
        assert_eq!(NUM_8X8_BLOCKS_HIGH_LOOKUP[9], 4); // BLOCK_32X32
        assert_eq!(NUM_8X8_BLOCKS_HIGH_LOOKUP[12], 8); // BLOCK_64X64
    }
}
