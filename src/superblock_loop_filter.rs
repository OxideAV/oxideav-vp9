//! VP9 §8.8.2 `superblock loop filter process` — the full per-plane,
//! per-pass driver, per spec v0.7.
//!
//! Round 278 composes the previously-landed loop-filter primitives
//! into the complete §8.8.2 process (`vp9-spec.txt` lines 5491-5586):
//!
//! * the §8.8.2 `dx` / `dy` / `sub` / `edgeLen` header — round-274
//!   [`crate::superblock_filter_geometry`] (lines 5510-5519);
//! * steps 1-14 (per-edge predicates) — round-274
//!   [`crate::superblock_filter_edge`] (lines 5526-5578);
//! * step 15 (`filterSize`) — round-244 §8.8.3 [`crate::filter_size`]
//!   (lines 5579-5581);
//! * step 16 (`lvl` / `limit` / `blimit` / `thresh`) — round-250
//!   §8.8.4 [`crate::adaptive_filter_strength`] (lines 5582-5583);
//! * step 17 (the conditional §8.8.5 sample-filter call) — round-267
//!   [`crate::sample_filtering`] (lines 5584-5586), including the
//!   16-sample stencil gather / scatter against `CurrFrame[ plane ]`
//!   per the §8.8.5.1 sample addressing (lines 5703-5727).
//!
//! ## Step-6 `txSz` resolution
//!
//! For chroma planes the per-edge `txSz` is resolved through §6.4.22
//! `get_uv_tx_size( )` (`vp9-spec.txt` lines 2871-2876) using the
//! `MiSize` / `tx_size` read at `(loopRow, loopCol)` — the same
//! crate-local helper the §6.4 residual path uses.
//!
//! ## Off-screen mode-info reads
//!
//! §8.8.2 steps 4-9 nominally read the per-MI arrays at every
//! `(loopRow, loopCol)`, including positions beyond the `MiCols` /
//! `MiRows` frame extent (the `edge` / `i` raster covers the whole
//! 64x64 superblock even when the frame ends mid-superblock). For
//! those positions step 13 forces `onScreen = 0`, so step 14 forces
//! `applyFilter = 0` and step 17 never runs — the read values are
//! dead. This driver therefore short-circuits the right / bottom
//! off-screen positions *before* the per-MI reads, which keeps every
//! array access inside the `MiRows x MiCols` bounds (when
//! `onScreen == 1`, `loopCol <= x >> 3 < MiCols` and
//! `loopRow <= y >> 3 < MiRows` by the step-2/3 align-down).
//!
//! ## Out-of-range stencil samples
//!
//! The §8.8.5.1 stencil reads `p7..p0` / `q0..q7` unconditionally,
//! but its NOTE (below line 5727) limits the outer ring (`p4..p7`,
//! `q4..q7`) to `filterSize == TX_16X16` — and a `TX_16X16` filter
//! only fires on edges whose plane coordinate is 16-aligned, where
//! the whole stencil is in-bounds. Reads outside the plane are
//! clamped to the plane edge; they can only feed echoed (unmutated)
//! outputs. Write-back skips positions whose true coordinate lies
//! outside the plane, so the clamped reads never alias a mutated
//! sample.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.2 lines 5491-5586; §8.8.5.1
//! stencil addressing lines 5703-5727; §6.4.22 `get_uv_tx_size( )`
//! lines 2871-2876).

use crate::adaptive_filter_strength::adaptive_filter_strength;
use crate::filter_size::{filter_size, PASS_VERTICAL};
use crate::loop_filter::LvlLookup;
use crate::residual::get_uv_tx_size;
use crate::sample_filtering::{sample_filtering, SampleFilterOutput, SampleFilterSamples};
use crate::superblock_filter::{
    superblock_filter_edge, superblock_filter_geometry, SuperblockFilterMi,
};

/// A mutable view of one `CurrFrame[ plane ]` sample plane for the
/// §8.8.2 driver.
///
/// `data` is row-major with `stride` samples per row;
/// `data[ y * stride + x ]` is `CurrFrame[ plane ][ y ][ x ]` for
/// `x < width`, `y < height`. Samples are `i32` to match the §8.8.5
/// [`SampleFilterSamples`] working type at every BitDepth.
#[derive(Debug)]
pub struct SuperblockFilterPlane<'a> {
    /// The plane samples, row-major. Must satisfy
    /// `data.len() >= stride * height`.
    pub data: &'a mut [i32],
    /// Samples per row. Must satisfy `stride >= width`.
    pub stride: usize,
    /// Plane width in samples (already subsampling-adjusted: the luma
    /// plane of a `MiCols`-wide frame is `8 * MiCols` samples wide,
    /// the 4:2:0 chroma planes `(8 * MiCols) >> 1`).
    pub width: usize,
    /// Plane height in samples (subsampling-adjusted like `width`).
    pub height: usize,
}

impl SuperblockFilterPlane<'_> {
    /// Read the sample at `(x, y)` with the coordinates clamped into
    /// the plane (edge replication). Only ever feeds stencil
    /// positions whose output is echoed unmutated — see the module
    /// docs.
    #[inline]
    fn get_clamped(&self, x: i64, y: i64) -> i32 {
        let cx = x.clamp(0, self.width as i64 - 1) as usize;
        let cy = y.clamp(0, self.height as i64 - 1) as usize;
        self.data[cy * self.stride + cx]
    }

    /// Write `value` at `(x, y)` if the true coordinate lies inside
    /// the plane; out-of-plane positions are dropped (they were read
    /// clamped and echoed — writing them anywhere would corrupt).
    #[inline]
    fn set_in_bounds(&mut self, x: i64, y: i64, value: i32) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.data[y as usize * self.stride + x as usize] = value;
        }
    }
}

/// The per-frame decode state the §8.8.2 driver reads.
///
/// The six per-MI arrays are row-major `MiRows x MiCols` (index
/// `loopRow * mi_cols + loopCol`) and carry the §6.4.4
/// `decode_block( )` fan-out state the §8.8.2 steps 4-9 and the
/// §8.8.4 step-16 lookup read:
///
/// * `mi_sizes` — `MiSizes[ ][ ]` (§8.8.2 step 4), §3 `BLOCK_*`
///   constants `0..=12`;
/// * `tx_sizes` — `TxSizes[ ][ ]` (§8.8.2 step 5), `TX_4X4..TX_32X32`;
/// * `skips` — `Skips[ ][ ]` (§8.8.2 step 8);
/// * `ref_frames_0` — `RefFrames[ ][ ][ 0 ]` (§8.8.2 step 9 + §8.8.4),
///   `INTRA_FRAME = 0 ..= ALTREF_FRAME = 3`;
/// * `y_modes` — `YModes[ ][ ]` (§8.8.4 `modeType` derivation);
/// * `segment_ids` — `SegmentIds[ ][ ]` (§8.8.4 `segment`), `0..=7`.
///
/// The frame-level scalars complete the §8.8.2 / §8.8.4 environment:
/// `mi_cols` / `mi_rows` (§6.2 frame size in 8x8 mode-info units),
/// `subsampling_x` / `subsampling_y` (§6.2.2 color config),
/// `loop_filter_sharpness` (§6.2.8), `bit_depth` (§6.2.2: 8, 10 or
/// 12), and the §8.8.1 [`LvlLookup`] produced by
/// [`crate::loop_filter_frame_init`].
#[derive(Debug)]
pub struct SuperblockFilterFrame<'a> {
    /// `MiSizes[ ][ ]` per §8.8.2 step 4.
    pub mi_sizes: &'a [u8],
    /// `TxSizes[ ][ ]` per §8.8.2 step 5.
    pub tx_sizes: &'a [u8],
    /// `Skips[ ][ ]` per §8.8.2 step 8.
    pub skips: &'a [bool],
    /// `RefFrames[ ][ ][ 0 ]` per §8.8.2 step 9 and §8.8.4.
    pub ref_frames_0: &'a [i32],
    /// `YModes[ ][ ]` per §8.8.4.
    pub y_modes: &'a [u8],
    /// `SegmentIds[ ][ ]` per §8.8.4.
    pub segment_ids: &'a [u8],
    /// `MiCols` per §6.2 — frame width in 8x8 mode-info units.
    pub mi_cols: u32,
    /// `MiRows` per §6.2 — frame height in 8x8 mode-info units.
    pub mi_rows: u32,
    /// §6.2.2 `subsampling_x` (`0` or `1`).
    pub subsampling_x: u8,
    /// §6.2.2 `subsampling_y` (`0` or `1`).
    pub subsampling_y: u8,
    /// §6.2.8 `loop_filter_sharpness` (`0..=7`).
    pub loop_filter_sharpness: u8,
    /// §6.2.2 `BitDepth` (8, 10 or 12).
    pub bit_depth: u8,
    /// The §8.8.1 `LvlLookup[ ][ ][ ]` filter-strength table.
    pub lvl_lookup: &'a LvlLookup,
}

/// Gather the §8.8.5.1 16-sample stencil at plane position `(x, y)`
/// along the `(dx, dy)` filter direction per `vp9-spec.txt` lines
/// 5703-5727 (`q_k` at `+k`, `p_k` at `-(k + 1)`).
fn gather_stencil(
    plane: &SuperblockFilterPlane,
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
) -> SampleFilterSamples {
    let s = |k: i64| plane.get_clamped(x + dx * k, y + dy * k);
    SampleFilterSamples {
        p7: s(-8),
        p6: s(-7),
        p5: s(-6),
        p4: s(-5),
        p3: s(-4),
        p2: s(-3),
        p1: s(-2),
        p0: s(-1),
        q0: s(0),
        q1: s(1),
        q2: s(2),
        q3: s(3),
        q4: s(4),
        q5: s(5),
        q6: s(6),
        q7: s(7),
    }
}

/// Scatter the §8.8.5 post-filter stencil back to `CurrFrame[ plane ]`
/// at the true (unclamped) sample positions; positions outside the
/// plane are dropped (their values are clamped-read echoes).
fn scatter_stencil(
    plane: &mut SuperblockFilterPlane,
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
    out: &SampleFilterOutput,
) {
    let positions: [(i64, i32); 16] = [
        (-8, out.p7),
        (-7, out.p6),
        (-6, out.p5),
        (-5, out.p4),
        (-4, out.p3),
        (-3, out.p2),
        (-2, out.p1),
        (-1, out.p0),
        (0, out.q0),
        (1, out.q1),
        (2, out.q2),
        (3, out.q3),
        (4, out.q4),
        (5, out.q5),
        (6, out.q6),
        (7, out.q7),
    ];
    for (k, v) in positions {
        plane.set_in_bounds(x + dx * k, y + dy * k, v);
    }
}

/// Run the §8.8.2 `superblock loop filter process` per `vp9-spec.txt`
/// lines 5491-5586, modifying `plane_buf` in place.
///
/// # Inputs (mirroring the §8.8.2 listing)
///
/// * `plane_buf` — the `CurrFrame[ plane ]` sample plane being
///   filtered.
/// * `frame` — the per-frame decode state (see
///   [`SuperblockFilterFrame`]).
/// * `plane` — `0` (Y), `1` (U) or `2` (V).
/// * `pass` — [`crate::PASS_VERTICAL`] (`0`, vertical block
///   boundaries) or [`crate::PASS_HORIZONTAL`] (`1`, horizontal block
///   boundaries).
/// * `row`, `col` — the superblock location in units of 8x8 blocks.
///
/// # Ordering
///
/// Edges are processed in the §8.8.2 raster order (`edge` outer, `i`
/// inner, both increasing) with in-place write-back, so a later edge
/// reads any samples an earlier edge of the same call already
/// filtered — exactly the spec's ordered-steps semantics.
///
/// # Panics
///
/// * If the plane view is inconsistent (`stride < width` or
///   `data.len() < stride * height`) or any per-MI array is shorter
///   than `mi_rows * mi_cols`.
/// * If an on-screen `(loopRow, loopCol)` carries per-MI state the
///   §8.8.4 lookup rejects (`segment_id >= 8` or `ref_frame` outside
///   `0..=3`) — invalid decode state, not a bitstream condition.
pub fn superblock_loop_filter(
    plane_buf: &mut SuperblockFilterPlane,
    frame: &SuperblockFilterFrame,
    plane: u8,
    pass: u8,
    row: u32,
    col: u32,
) {
    assert!(
        plane_buf.stride >= plane_buf.width
            && plane_buf.data.len() >= plane_buf.stride * plane_buf.height,
        "§8.8.2: inconsistent CurrFrame plane view"
    );
    let mi_cells = (frame.mi_rows as usize) * (frame.mi_cols as usize);
    assert!(
        frame.mi_sizes.len() >= mi_cells
            && frame.tx_sizes.len() >= mi_cells
            && frame.skips.len() >= mi_cells
            && frame.ref_frames_0.len() >= mi_cells
            && frame.y_modes.len() >= mi_cells
            && frame.segment_ids.len() >= mi_cells,
        "§8.8.2: per-MI arrays shorter than MiRows * MiCols"
    );

    // §8.8.2 lines 5505-5508: subX / subY from the plane.
    let (sub_x, sub_y) = if plane == 0 {
        (0u32, 0u32)
    } else {
        (
            u32::from(frame.subsampling_x),
            u32::from(frame.subsampling_y),
        )
    };

    // §8.8.2 lines 5510-5519: dx / dy / sub / edgeLen.
    let geom = superblock_filter_geometry(pass, sub_x, sub_y);

    // §8.8.2 lines 5524-5525: edge ∈ 0..(16 >> sub) - 1, i ∈ 0..edgeLen - 1.
    for edge in 0..(16u32 >> geom.sub) {
        for i in 0..geom.edge_len {
            // Step 1 (lines 5527-5531): x / y in luma coordinates.
            let (x, y) = if pass == PASS_VERTICAL {
                (col * 8 + edge * (4 << sub_x), row * 8 + (i << sub_y))
            } else {
                (col * 8 + (i << sub_x), row * 8 + edge * (4 << sub_y))
            };

            // Step 13 right / bottom exclusions, hoisted ahead of the
            // steps 4-9 per-MI reads: off-screen positions force
            // applyFilter = 0 (step 14), making the reads dead — and
            // hoisting keeps (loopRow, loopCol) inside MiRows x MiCols.
            if x >= 8 * frame.mi_cols || y >= 8 * frame.mi_rows {
                continue;
            }

            // Steps 2-3 (lines 5532-5533): loopCol / loopRow.
            let loop_col = ((x >> 3) >> sub_x) << sub_x;
            let loop_row = ((y >> 3) >> sub_y) << sub_y;
            let idx = loop_row as usize * frame.mi_cols as usize + loop_col as usize;

            // Steps 4-5 (lines 5534-5535): MiSize / tx_size.
            let mi_size = frame.mi_sizes[idx];
            let tx_size = frame.tx_sizes[idx];

            // Step 6 (line 5537): txSz = (plane > 0) ? get_uv_tx_size( )
            // : tx_size, with §6.4.22 reading the MiSize / tx_size at
            // (loopRow, loopCol) and the frame subsampling.
            let tx_sz = if plane > 0 {
                get_uv_tx_size(
                    u32::from(tx_size),
                    mi_size,
                    frame.subsampling_x == 1,
                    frame.subsampling_y == 1,
                ) as u8
            } else {
                tx_size
            };

            // Steps 7-14 (lines 5538-5578): the round-274 predicate
            // bundle (isBlockEdge / isTxEdge / is32Edge / onScreen /
            // applyFilter).
            let mi = SuperblockFilterMi {
                mi_size,
                tx_sz,
                skip: frame.skips[idx],
                ref_frame_0: frame.ref_frames_0[idx],
            };
            let e = superblock_filter_edge(
                pass,
                row,
                col,
                edge,
                i,
                sub_x,
                sub_y,
                frame.mi_cols,
                frame.mi_rows,
                &mi,
            );
            if !e.apply_filter {
                // Step 17's applyFilter == 1 gate: steps 15-16 are
                // pure derivations with no other consumer, so the
                // whole tail is skipped.
                continue;
            }

            // Step 15 (lines 5579-5581): filterSize via §8.8.3.
            let f_size = filter_size(
                tx_sz,
                e.is_32_edge,
                pass,
                e.x,
                e.y,
                sub_x as u8,
                sub_y as u8,
                frame.mi_cols,
                frame.mi_rows,
            );

            // Step 16 (lines 5582-5583): lvl / limit / blimit / thresh
            // via §8.8.4 at (loopRow, loopCol).
            let strength = adaptive_filter_strength(
                frame.lvl_lookup,
                frame.segment_ids[idx] as usize,
                frame.ref_frames_0[idx],
                frame.y_modes[idx],
                frame.loop_filter_sharpness,
            )
            .expect("§8.8.4: segment_id must be < 8 and ref_frame in 0..=3");

            // Step 17 (lines 5584-5586): if applyFilter == 1 and
            // lvl > 0, run §8.8.5 at (x >> subX, y >> subY) along
            // (dx, dy).
            if strength.lvl > 0 {
                let px = i64::from(e.x >> sub_x);
                let py = i64::from(e.y >> sub_y);
                let (dx, dy) = (i64::from(geom.dx), i64::from(geom.dy));
                let stencil = gather_stencil(plane_buf, px, py, dx, dy);
                let out = sample_filtering(
                    &stencil,
                    strength.limit,
                    strength.blimit,
                    strength.thresh,
                    f_size,
                    frame.bit_depth,
                );
                scatter_stencil(plane_buf, px, py, dx, dy, &out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_size::{PASS_HORIZONTAL, TX_4X4};
    use crate::loop_filter::{loop_filter_frame_init, SEG_LVL_ALT_L};
    use crate::residual::{BLOCK_64X64, BLOCK_8X8};
    use crate::{LoopFilterParams, SegmentationParams, MAX_SEGMENTS, SEG_LVL_MAX};

    fn lf(level: u8) -> LoopFilterParams {
        LoopFilterParams {
            level,
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            ref_deltas: [None; 4],
            mode_deltas: [None; 2],
        }
    }

    fn seg_disabled() -> SegmentationParams {
        SegmentationParams {
            enabled: false,
            update_map: false,
            tree_probs: None,
            temporal_update: false,
            pred_prob: None,
            update_data: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }

    /// Owns the per-MI arrays + LvlLookup so a test can borrow a
    /// [`SuperblockFilterFrame`] view.
    struct Fixture {
        mi_sizes: Vec<u8>,
        tx_sizes: Vec<u8>,
        skips: Vec<bool>,
        refs: Vec<i32>,
        y_modes: Vec<u8>,
        seg_ids: Vec<u8>,
        lookup: LvlLookup,
        mi_cols: u32,
        mi_rows: u32,
        sub_x: u8,
        sub_y: u8,
        bit_depth: u8,
    }

    impl Fixture {
        /// A uniform MiRows x MiCols grid with a step-3 broadcast
        /// LvlLookup at `level` (sharpness 0, no deltas, no
        /// segmentation).
        fn uniform(
            mi_cols: u32,
            mi_rows: u32,
            mi_size: u8,
            tx_size: u8,
            skip: bool,
            ref0: i32,
            level: u8,
        ) -> Fixture {
            let n = (mi_cols * mi_rows) as usize;
            Fixture {
                mi_sizes: vec![mi_size; n],
                tx_sizes: vec![tx_size; n],
                skips: vec![skip; n],
                refs: vec![ref0; n],
                y_modes: vec![0; n], // DC_PRED — modeType 0
                seg_ids: vec![0; n],
                lookup: loop_filter_frame_init(&lf(level), &seg_disabled(), [0; 4], [0; 2]),
                mi_cols,
                mi_rows,
                sub_x: 0,
                sub_y: 0,
                bit_depth: 8,
            }
        }

        fn frame(&self) -> SuperblockFilterFrame<'_> {
            SuperblockFilterFrame {
                mi_sizes: &self.mi_sizes,
                tx_sizes: &self.tx_sizes,
                skips: &self.skips,
                ref_frames_0: &self.refs,
                y_modes: &self.y_modes,
                segment_ids: &self.seg_ids,
                mi_cols: self.mi_cols,
                mi_rows: self.mi_rows,
                subsampling_x: self.sub_x,
                subsampling_y: self.sub_y,
                loop_filter_sharpness: 0,
                bit_depth: self.bit_depth,
                lvl_lookup: &self.lookup,
            }
        }
    }

    /// Owns a plane sample buffer (stride == width).
    struct PlaneBuf {
        data: Vec<i32>,
        w: usize,
        h: usize,
    }

    impl PlaneBuf {
        fn filled(w: usize, h: usize, v: i32) -> PlaneBuf {
            PlaneBuf {
                data: vec![v; w * h],
                w,
                h,
            }
        }

        /// Vertical step: columns `< at` carry `lo`, the rest `hi`.
        fn vstep(w: usize, h: usize, at: usize, lo: i32, hi: i32) -> PlaneBuf {
            let mut p = PlaneBuf::filled(w, h, lo);
            for y in 0..h {
                for x in at..w {
                    p.data[y * w + x] = hi;
                }
            }
            p
        }

        /// Horizontal step: rows `< at` carry `lo`, the rest `hi`.
        fn hstep(w: usize, h: usize, at: usize, lo: i32, hi: i32) -> PlaneBuf {
            let mut p = PlaneBuf::filled(w, h, lo);
            for y in at..h {
                for x in 0..w {
                    p.data[y * w + x] = hi;
                }
            }
            p
        }

        fn view(&mut self) -> SuperblockFilterPlane<'_> {
            SuperblockFilterPlane {
                data: &mut self.data,
                stride: self.w,
                width: self.w,
                height: self.h,
            }
        }

        fn get(&self, x: usize, y: usize) -> i32 {
            self.data[y * self.w + x]
        }
    }

    /// The §8.8.5.1 two-level stencil: every `p` sample `lo`, every
    /// `q` sample `hi`.
    fn two_level_stencil(lo: i32, hi: i32) -> SampleFilterSamples {
        SampleFilterSamples {
            p7: lo,
            p6: lo,
            p5: lo,
            p4: lo,
            p3: lo,
            p2: lo,
            p1: lo,
            p0: lo,
            q0: hi,
            q1: hi,
            q2: hi,
            q3: hi,
            q4: hi,
            q5: hi,
            q6: hi,
            q7: hi,
        }
    }

    /// §8.8.5 identity on flat regions: a constant plane survives both
    /// passes untouched (every filter branch is the identity on a flat
    /// stencil).
    #[test]
    fn flat_plane_is_identity_both_passes() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        let mut pb = PlaneBuf::filled(64, 64, 128);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);
        assert!(pb.data.iter().all(|&v| v == 128));
    }

    /// §8.8.2 step 17 `lvl > 0` gate: a zero LvlLookup leaves even a
    /// sharp step untouched.
    #[test]
    fn lvl_zero_is_no_op() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 0);
        let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
        let reference = pb.data.clone();
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);
        assert_eq!(pb.data, reference);
    }

    /// §8.8.2 steps 15-17 vertical thread: a step at `x = 8` on a
    /// BLOCK_8X8 / TX_4X4 grid is filtered exactly like a direct
    /// §8.8.5 call on the two-level stencil, and only the narrow
    /// 4-sample window moves.
    #[test]
    fn vertical_step_matches_direct_sample_filtering() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);

        // Step 16 equivalent: §8.8.4 at any (loopRow, loopCol) of the
        // uniform grid.
        let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
        // Step 17 equivalent: §8.8.5 with filterSize TX_4X4 (edge 2 is
        // not a multiple of 8, so §8.8.3 keeps Min(TX_16X16, TX_4X4)).
        let out = sample_filtering(
            &two_level_stencil(60, 68),
            s.limit,
            s.blimit,
            s.thresh,
            TX_4X4,
            8,
        );
        // The step is inside the filter limits, so the edge moved.
        assert_ne!(out.q0, 68);
        for y in 0..64 {
            for x in 0..64usize {
                let expect = match x {
                    6 => out.p1,
                    7 => out.p0,
                    8 => out.q0,
                    9 => out.q1,
                    _ if x < 8 => 60,
                    _ => 68,
                };
                assert_eq!(pb.get(x, y), expect, "({x}, {y})");
            }
        }
    }

    /// Mirror of the vertical test for `pass == 1`: a step at `y = 8`
    /// moves rows 6..=9.
    #[test]
    fn horizontal_step_matches_direct_sample_filtering() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        let mut pb = PlaneBuf::hstep(64, 64, 8, 60, 68);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);

        let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
        let out = sample_filtering(
            &two_level_stencil(60, 68),
            s.limit,
            s.blimit,
            s.thresh,
            TX_4X4,
            8,
        );
        for y in 0..64usize {
            for x in 0..64 {
                let expect = match y {
                    6 => out.p1,
                    7 => out.p0,
                    8 => out.q0,
                    9 => out.q1,
                    _ if y < 8 => 60,
                    _ => 68,
                };
                assert_eq!(pb.get(x, y), expect, "({x}, {y})");
            }
        }
    }

    /// §8.8.2 step 14 inter-skip gate threaded through the driver: a
    /// pure tx edge (BLOCK_64X64 so no interior block edges) on a
    /// skipped inter block is not filtered; the same edge filters once
    /// `skip == 0`.
    #[test]
    fn inter_skip_tx_edge_gates_filtering() {
        // skip == 1, inter (ref 1 = LAST): untouched.
        let fx = Fixture::uniform(8, 8, BLOCK_64X64, TX_4X4, true, 1, 25);
        let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
        let reference = pb.data.clone();
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
        assert_eq!(pb.data, reference);

        // skip == 0: the tx edge at x = 8 filters.
        let fx2 = Fixture::uniform(8, 8, BLOCK_64X64, TX_4X4, false, 1, 25);
        let mut pb2 = PlaneBuf::vstep(64, 64, 8, 60, 68);
        superblock_loop_filter(&mut pb2.view(), &fx2.frame(), 0, PASS_VERTICAL, 0, 0);
        assert_ne!(pb2.get(8, 0), 68);
    }

    /// §8.8.2 step 14 block-edge arm: a BLOCK_8X8 boundary filters
    /// even when the block is skipped inter (isBlockEdge wins over the
    /// skip gate).
    #[test]
    fn block_edge_filters_even_when_skipped_inter() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, true, 1, 25);
        let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
        assert_ne!(pb.get(8, 0), 68);
    }

    /// §8.8.2 step 13 left / top frame-edge exclusion: the `x == 0`
    /// vertical edge and the `y == 0` horizontal edge never write
    /// (a discontinuity hugging the frame edge would trip the §8.8.5.2
    /// hev branch if the excluded edge ran against clamped reads).
    #[test]
    fn frame_edge_columns_and_rows_untouched() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        // Column 0 carries 60, everything else 68: only the excluded
        // x == 0 edge sees a step, so nothing changes.
        let mut pb = PlaneBuf::vstep(64, 64, 1, 60, 68);
        let reference = pb.data.clone();
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
        assert_eq!(pb.data, reference);

        // Row 0 carries 60: only the excluded y == 0 edge sees a step.
        let mut pb2 = PlaneBuf::hstep(64, 64, 1, 60, 68);
        let reference2 = pb2.data.clone();
        superblock_loop_filter(&mut pb2.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);
        assert_eq!(pb2.data, reference2);
    }

    /// 4:2:0 chroma plane: the luma `x = 8` edge lands at chroma
    /// `x' = 4`; the step-6 `get_uv_tx_size` resolution keeps TX_4X4,
    /// the narrow window (chroma columns 2..=5) matches a direct
    /// §8.8.5 call, and the clamped outer-ring reads (`p4..p7` at
    /// chroma columns -1..-4) leave columns 0..=1 untouched.
    #[test]
    fn chroma_420_vertical_matches_direct() {
        let mut fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        fx.sub_x = 1;
        fx.sub_y = 1;
        let mut pb = PlaneBuf::vstep(32, 32, 4, 60, 68);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 1, PASS_VERTICAL, 0, 0);

        let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
        let out = sample_filtering(
            &two_level_stencil(60, 68),
            s.limit,
            s.blimit,
            s.thresh,
            TX_4X4,
            8,
        );
        assert_ne!(out.q0, 68);
        for y in 0..32 {
            for x in 0..32usize {
                let expect = match x {
                    2 => out.p1,
                    3 => out.p0,
                    4 => out.q0,
                    5 => out.q1,
                    _ if x < 4 => 60,
                    _ => 68,
                };
                assert_eq!(pb.get(x, y), expect, "({x}, {y})");
            }
        }
    }

    /// A frame that ends mid-superblock (MiCols = MiRows = 6, 48x48
    /// luma): the off-screen short-circuit skips the per-MI reads for
    /// `x >= 48` / `y >= 48` without panicking, and an interior step
    /// still filters.
    #[test]
    fn partial_superblock_off_screen_edges_skipped() {
        let fx = Fixture::uniform(6, 6, BLOCK_8X8, TX_4X4, false, 0, 25);
        let mut pb = PlaneBuf::vstep(48, 48, 40, 60, 68);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);

        let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
        let out = sample_filtering(
            &two_level_stencil(60, 68),
            s.limit,
            s.blimit,
            s.thresh,
            TX_4X4,
            8,
        );
        assert_eq!(pb.get(40, 0), out.q0);
        assert_eq!(pb.get(39, 0), out.p0);
        // The horizontal pass saw no step (rows are uniform after the
        // vertical pass mutated whole columns), so row values agree
        // down each column.
        for x in 0..48usize {
            for y in 1..48 {
                assert_eq!(pb.get(x, y), pb.get(x, 0), "({x}, {y})");
            }
        }
    }

    /// 10-bit path: the step survives the same §8.8.5 thread with
    /// BitDepth-scaled limits.
    #[test]
    fn bit_depth_10_step_matches_direct() {
        let mut fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        fx.bit_depth = 10;
        let mut pb = PlaneBuf::vstep(64, 64, 8, 480, 544);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);

        let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
        let out = sample_filtering(
            &two_level_stencil(480, 544),
            s.limit,
            s.blimit,
            s.thresh,
            TX_4X4,
            10,
        );
        assert_ne!(out.q0, 544);
        for y in 0..64 {
            assert_eq!(pb.get(7, y), out.p0, "row {y}");
            assert_eq!(pb.get(8, y), out.q0, "row {y}");
        }
    }

    /// §8.8.2 step 16 reads the strength at (loopRow, loopCol): a
    /// segment-1 SEG_LVL_ALT_L absolute override of 0 turns off the
    /// edge whose MI sits in segment 1 (loopCol 1 → the x = 8 edge)
    /// while the segment-0 edge at x = 16 still filters.
    #[test]
    fn per_segment_lvl_partitions_edges() {
        let mut fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        // Segment 1: SEG_LVL_ALT_L absolute 0 → lvl 0.
        let mut seg = seg_disabled();
        seg.enabled = true;
        seg.abs_or_delta_update = true;
        seg.feature_enabled[1][SEG_LVL_ALT_L] = true;
        seg.feature_data[1][SEG_LVL_ALT_L] = 0;
        fx.lookup = loop_filter_frame_init(&lf(25), &seg, [0; 4], [0; 2]);
        // The x = 8 edge reads the MI at loopCol = 1: put it in
        // segment 1 on every row.
        for r in 0..8usize {
            fx.seg_ids[r * 8 + 1] = 1;
        }

        // Steps at x = 8 (60 → 68, segment 1) and x = 16 (68 → 76,
        // segment 0).
        let mut pb = PlaneBuf::filled(64, 64, 60);
        for y in 0..64 {
            for x in 8..16 {
                pb.data[y * 64 + x] = 68;
            }
            for x in 16..64 {
                pb.data[y * 64 + x] = 76;
            }
        }
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);

        // x = 8 (segment 1, lvl 0): untouched.
        assert_eq!(pb.get(7, 0), 60);
        assert_eq!(pb.get(8, 0), 68);
        // x = 16 (segment 0, lvl 25): matches the direct call.
        let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
        let out = sample_filtering(
            &two_level_stencil(68, 76),
            s.limit,
            s.blimit,
            s.thresh,
            TX_4X4,
            8,
        );
        assert_ne!(out.q0, 76);
        for y in 0..64 {
            assert_eq!(pb.get(14, y), out.p1, "row {y}");
            assert_eq!(pb.get(15, y), out.p0, "row {y}");
            assert_eq!(pb.get(16, y), out.q0, "row {y}");
            assert_eq!(pb.get(17, y), out.q1, "row {y}");
        }
    }

    /// Per-MI arrays shorter than MiRows * MiCols are rejected up
    /// front.
    #[test]
    #[should_panic(expected = "per-MI arrays")]
    fn short_mi_arrays_rejected() {
        let mut fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        fx.mi_sizes.truncate(63);
        let mut pb = PlaneBuf::filled(64, 64, 128);
        superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
    }

    /// An inconsistent plane view (stride < width) is rejected up
    /// front.
    #[test]
    #[should_panic(expected = "plane view")]
    fn inconsistent_plane_view_rejected() {
        let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
        let mut data = vec![128; 64 * 64];
        let mut view = SuperblockFilterPlane {
            data: &mut data,
            stride: 32,
            width: 64,
            height: 64,
        };
        superblock_loop_filter(&mut view, &fx.frame(), 0, PASS_VERTICAL, 0, 0);
    }
}
