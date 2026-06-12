//! VP9 §8.8 `loop filter process` — the frame-level driver, per spec
//! v0.7.
//!
//! Round 281 lands the outermost layer of the §8.8 loop-filter arc:
//! the raster walk over every superblock of the frame
//! (`vp9-spec.txt` lines 5442-5455) that invokes the round-278
//! §8.8.2 [`crate::superblock_loop_filter`] driver once per
//! `(row, col, plane, pass)` combination, plus the 3-plane
//! [`CurrFrame`] container the process reads and modifies:
//!
//! ```text
//! for ( row = 0; row < MiRows; row += 8 )
//!     for ( col = 0; col < MiCols; col += 8 )
//!         for ( plane = 0; plane < 3; plane++ )
//!             for ( pass = 0; pass < 2; pass++ )
//!                 The superblock loop filter process specified in
//!                 8.8.2 is invoked with the variables plane, pass,
//!                 row, and col as inputs.
//! ```
//!
//! ## The §8.8.1 first step
//!
//! §8.8 line 5441 invokes the loop filter frame init process
//! (§8.8.1) before the raster. That step is the round-37
//! [`crate::loop_filter_frame_init`] primitive; its [`crate::LvlLookup`]
//! output reaches this driver through the
//! [`SuperblockFilterFrame::lvl_lookup`] field, so callers run it
//! once per frame and thread the table in — keeping this driver free
//! of the §6.2.8 / §6.2.11 header state §8.8.1 consumes.
//!
//! ## Ordering
//!
//! The §8.8 NOTE below the listing (lines 5458-5460) requires the
//! edge-processing order given by the raster to be respected because
//! many samples are filtered more than once. [`frame_loop_filter`]
//! therefore iterates exactly in the listing's nesting order — `row`
//! outermost, then `col`, `plane`, `pass` — and every §8.8.2 call
//! mutates the plane in place before the next call reads it.
//!
//! ## Plane extents
//!
//! `CurrFrame[ 0 ]` spans `FrameWidth x FrameHeight` samples and
//! `CurrFrame[ 1..2 ]` span
//! `((FrameWidth + subsampling_x) >> subsampling_x) x
//! ((FrameHeight + subsampling_y) >> subsampling_y)` — the extents
//! the §8.10 reference-frame-update copy enumerates (`vp9-spec.txt`
//! lines 5944-5948). [`frame_loop_filter`] checks the three plane
//! views against each other (and the luma extent against the §7.2.6
//! `MiCols = (FrameWidth + 7) >> 3` / `MiRows = (FrameHeight + 7) >>
//! 3` relations, lines 1760-1761) up front, then relies on the
//! §8.8.2 driver's off-screen short-circuit and clamped-read /
//! dropped-write stencil handling for the partial-superblock right /
//! bottom borders.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8 lines 5436-5463; §8.10 plane
//! extents lines 5944-5948; §7.2.6 `compute_image_size( )` lines
//! 1760-1761).

use crate::superblock_loop_filter::{
    superblock_loop_filter, SuperblockFilterFrame, SuperblockFilterPlane,
};

/// The 3-plane `CurrFrame` array of reconstructed samples — the §8.8
/// input/output (`vp9-spec.txt` lines 5437-5438).
///
/// `planes[ 0 ]` is the Y plane (`FrameWidth x FrameHeight`);
/// `planes[ 1 ]` / `planes[ 2 ]` are the U / V planes at the §8.10
/// subsampled extents (lines 5944-5948). Each plane is a
/// [`SuperblockFilterPlane`] mutable `data / stride / width / height`
/// view (`i32` samples, the §8.8.5 working type at every BitDepth).
#[derive(Debug)]
pub struct CurrFrame<'a> {
    /// `CurrFrame[ plane ]` for `plane = 0..2` (Y, U, V).
    pub planes: [SuperblockFilterPlane<'a>; 3],
}

/// Run the §8.8 `loop filter process` over a whole frame per
/// `vp9-spec.txt` lines 5436-5455, modifying `curr` in place.
///
/// Walks the §8.8 raster — `row` over `0, 8, .. < MiRows`, `col` over
/// `0, 8, .. < MiCols`, `plane` over `0..2`, `pass` over `0..1` —
/// invoking the §8.8.2 [`superblock_loop_filter`] process at each
/// step (lines 5451-5455), in exactly that nesting order per the §8.8
/// ordering NOTE (lines 5458-5460).
///
/// The §8.8 first step — the §8.8.1 frame init (line 5441) — is the
/// caller's [`crate::loop_filter_frame_init`] invocation; its
/// [`crate::LvlLookup`] output arrives via `frame.lvl_lookup`.
///
/// # Inputs
///
/// * `curr` — the [`CurrFrame`] 3-plane array of reconstructed
///   samples (§8.8 input, lines 5437-5438).
/// * `frame` — the per-frame decode state the §8.8.2 / §8.8.4 steps
///   read (see [`SuperblockFilterFrame`]), including the §8.8.1
///   `LvlLookup`.
///
/// # Panics
///
/// * If the luma extent disagrees with the §7.2.6 mode-info grid
///   (`MiCols != (width + 7) >> 3` or `MiRows != (height + 7) >> 3`,
///   lines 1760-1761).
/// * If a chroma extent disagrees with the §8.10 subsampled extents
///   (lines 5944-5948): `width != (luma_width + subsampling_x) >>
///   subsampling_x` or the `height` analogue.
/// * Whenever the inner §8.8.2 driver panics (inconsistent plane
///   view, short per-MI arrays, or invalid per-MI decode state).
pub fn frame_loop_filter(curr: &mut CurrFrame, frame: &SuperblockFilterFrame) {
    // §7.2.6 lines 1760-1761: MiCols = (FrameWidth + 7) >> 3,
    // MiRows = (FrameHeight + 7) >> 3 — tie the luma plane extent to
    // the MiRows x MiCols grid the raster (and the per-MI arrays) use.
    let y = &curr.planes[0];
    assert!(
        (y.width + 7) >> 3 == frame.mi_cols as usize
            && (y.height + 7) >> 3 == frame.mi_rows as usize,
        "§8.8: luma plane extent inconsistent with MiCols / MiRows (§7.2.6)"
    );
    // §8.10 lines 5944-5948: chroma planes span
    // ((FrameWidth + subsampling_x) >> subsampling_x) x
    // ((FrameHeight + subsampling_y) >> subsampling_y).
    let uv_w = (y.width + frame.subsampling_x as usize) >> frame.subsampling_x;
    let uv_h = (y.height + frame.subsampling_y as usize) >> frame.subsampling_y;
    for plane in 1..3 {
        assert!(
            curr.planes[plane].width == uv_w && curr.planes[plane].height == uv_h,
            "§8.8: chroma plane extent inconsistent with subsampling (§8.10)"
        );
    }

    // §8.8 lines 5451-5455: the four-deep superblock raster.
    for row in (0..frame.mi_rows).step_by(8) {
        for col in (0..frame.mi_cols).step_by(8) {
            for plane in 0..3u8 {
                for pass in 0..2u8 {
                    superblock_loop_filter(
                        &mut curr.planes[plane as usize],
                        frame,
                        plane,
                        pass,
                        row,
                        col,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_size::TX_4X4;
    use crate::loop_filter::{loop_filter_frame_init, LvlLookup};
    use crate::residual::BLOCK_8X8;
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

    /// Owns the per-MI arrays + `LvlLookup` backing a
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
        /// A uniform MiRows x MiCols grid (BLOCK_8X8 / TX_4X4 /
        /// unskipped intra) with a step-3 broadcast LvlLookup at
        /// `level` — every internal 8x8 boundary is a filtering block
        /// edge.
        fn uniform(mi_cols: u32, mi_rows: u32, sub_x: u8, sub_y: u8, level: u8) -> Fixture {
            let n = (mi_cols * mi_rows) as usize;
            Fixture {
                mi_sizes: vec![BLOCK_8X8; n],
                tx_sizes: vec![TX_4X4; n],
                skips: vec![false; n],
                refs: vec![0; n],    // INTRA_FRAME
                y_modes: vec![0; n], // DC_PRED — §8.8.4 modeType 0
                seg_ids: vec![0; n],
                lookup: loop_filter_frame_init(&lf(level), &seg_disabled(), [0; 4], [0; 2]),
                mi_cols,
                mi_rows,
                sub_x,
                sub_y,
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

    /// Owns one plane sample buffer (stride == width).
    #[derive(Clone)]
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

        /// Deterministic pseudo-random fill in `0..=max` — makes
        /// nearly every loop-filter edge order-sensitive.
        fn noise(w: usize, h: usize, seed: &mut u32, max: i32) -> PlaneBuf {
            let mut p = PlaneBuf::filled(w, h, 0);
            for v in p.data.iter_mut() {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = ((*seed >> 24) as i32) % (max + 1);
            }
            p
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

    /// Build the three plane buffers for a `fix`-shaped frame.
    fn noise_planes(fix: &Fixture, seed: u32, max: i32) -> [PlaneBuf; 3] {
        let w = fix.mi_cols as usize * 8;
        let h = fix.mi_rows as usize * 8;
        let uv_w = (w + fix.sub_x as usize) >> fix.sub_x;
        let uv_h = (h + fix.sub_y as usize) >> fix.sub_y;
        let mut s = seed;
        [
            PlaneBuf::noise(w, h, &mut s, max),
            PlaneBuf::noise(uv_w, uv_h, &mut s, max),
            PlaneBuf::noise(uv_w, uv_h, &mut s, max),
        ]
    }

    /// The §8.8 raster transcribed directly from the listing (lines
    /// 5451-5455) over individual §8.8.2 calls — the reference the
    /// frame driver must match sample-exactly.
    fn reference_raster(planes: &mut [PlaneBuf; 3], fix: &Fixture) {
        let frame = fix.frame();
        let mut row = 0;
        while row < fix.mi_rows {
            let mut col = 0;
            while col < fix.mi_cols {
                for plane in 0..3u8 {
                    for pass in 0..2u8 {
                        superblock_loop_filter(
                            &mut planes[plane as usize].view(),
                            &frame,
                            plane,
                            pass,
                            row,
                            col,
                        );
                    }
                }
                col += 8;
            }
            row += 8;
        }
    }

    fn run_frame(planes: &mut [PlaneBuf; 3], fix: &Fixture) {
        let [y, u, v] = planes;
        let mut curr = CurrFrame {
            planes: [y.view(), u.view(), v.view()],
        };
        frame_loop_filter(&mut curr, &fix.frame());
    }

    #[test]
    fn flat_frame_is_identity_on_all_planes() {
        // 2x2 superblocks, 4:2:0 — every §8.8.5 branch is the
        // identity on flat content.
        let fix = Fixture::uniform(16, 16, 1, 1, 32);
        let mut planes = [
            PlaneBuf::filled(128, 128, 80),
            PlaneBuf::filled(64, 64, 90),
            PlaneBuf::filled(64, 64, 100),
        ];
        let before = planes.clone();
        run_frame(&mut planes, &fix);
        for (p, b) in planes.iter().zip(before.iter()) {
            assert_eq!(p.data, b.data);
        }
    }

    #[test]
    fn level_zero_leaves_a_sharp_step_untouched() {
        // §8.8.2 step 17's lvl > 0 gate, threaded through the whole
        // frame raster (the 60 -> 68 step is small enough to pass the
        // §8.8.5.1 filterMask thresholds at any non-zero level).
        let fix = Fixture::uniform(16, 16, 1, 1, 0);
        let mut planes = [
            PlaneBuf::vstep(128, 128, 64, 60, 68),
            PlaneBuf::filled(64, 64, 90),
            PlaneBuf::filled(64, 64, 100),
        ];
        let before = planes.clone();
        run_frame(&mut planes, &fix);
        for (p, b) in planes.iter().zip(before.iter()) {
            assert_eq!(p.data, b.data);
        }
    }

    #[test]
    fn matches_the_transcribed_raster_on_a_multi_superblock_noise_frame() {
        // 2x2 superblock grid, 4:2:0, order-sensitive content: the
        // driver must reproduce the listing's nesting order
        // sample-exactly.
        let fix = Fixture::uniform(16, 16, 1, 1, 40);
        let mut got = noise_planes(&fix, 0x1234_5678, 255);
        let mut want = got.clone();
        run_frame(&mut got, &fix);
        reference_raster(&mut want, &fix);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.data, w.data);
        }
        // Sanity: the filter actually moved samples.
        let flat = noise_planes(&fix, 0x1234_5678, 255);
        assert_ne!(got[0].data, flat[0].data);
    }

    #[test]
    fn matches_the_transcribed_raster_on_a_partial_superblock_frame() {
        // MiCols = MiRows = 12 — the second superblock row / column
        // covers only 4 of 8 mode-info units (96x96 luma).
        let fix = Fixture::uniform(12, 12, 1, 1, 40);
        let mut got = noise_planes(&fix, 0x0bad_f00d, 255);
        let mut want = got.clone();
        run_frame(&mut got, &fix);
        reference_raster(&mut want, &fix);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.data, w.data);
        }
    }

    #[test]
    fn matches_the_transcribed_raster_on_non_mi_aligned_extents() {
        // FrameWidth = 52, FrameHeight = 36 — MiCols = 7, MiRows = 5
        // per §7.2.6, chroma 26x18 per §8.10; the right / bottom MI
        // columns hang past the plane.
        let fix = Fixture::uniform(7, 5, 1, 1, 40);
        let mk = |seed: u32| {
            let mut s = seed;
            [
                PlaneBuf::noise(52, 36, &mut s, 255),
                PlaneBuf::noise(26, 18, &mut s, 255),
                PlaneBuf::noise(26, 18, &mut s, 255),
            ]
        };
        let mut got = mk(0xdead_beef);
        let mut want = got.clone();
        run_frame(&mut got, &fix);
        reference_raster(&mut want, &fix);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.data, w.data);
        }
    }

    #[test]
    fn filters_the_cross_superblock_vertical_boundary() {
        // A luma step exactly at x = 64 — the boundary between the
        // col = 0 and col = 8 superblocks — is the col = 8 call's
        // edge 0; the frame walk must reach it. The 60 -> 68 step is
        // small enough to pass the §8.8.5.1 filterMask thresholds.
        let fix = Fixture::uniform(16, 16, 1, 1, 32);
        let mut planes = [
            PlaneBuf::vstep(128, 128, 64, 60, 68),
            PlaneBuf::filled(64, 64, 90),
            PlaneBuf::filled(64, 64, 100),
        ];
        run_frame(&mut planes, &fix);
        // The boundary-adjacent samples moved toward each other...
        assert!(planes[0].get(63, 32) > 60);
        assert!(planes[0].get(64, 32) < 68);
        // ...while columns far from any value discontinuity are
        // untouched (flat content is a filter fixed point).
        assert_eq!(planes[0].get(0, 32), 60);
        assert_eq!(planes[0].get(127, 32), 68);
        // The flat chroma planes stay flat.
        assert!(planes[1].data.iter().all(|&v| v == 90));
        assert!(planes[2].data.iter().all(|&v| v == 100));
    }

    #[test]
    fn routes_chroma_planes_through_the_subsampled_raster() {
        // A U-plane step at chroma x = 32 (luma 64) filters; the
        // flat Y / V planes are untouched.
        let fix = Fixture::uniform(16, 16, 1, 1, 32);
        let mut planes = [
            PlaneBuf::filled(128, 128, 80),
            PlaneBuf::vstep(64, 64, 32, 60, 68),
            PlaneBuf::filled(64, 64, 100),
        ];
        run_frame(&mut planes, &fix);
        assert!(planes[1].get(31, 16) > 60);
        assert!(planes[1].get(32, 16) < 68);
        assert!(planes[0].data.iter().all(|&v| v == 80));
        assert!(planes[2].data.iter().all(|&v| v == 100));
    }

    #[test]
    fn matches_the_transcribed_raster_at_bit_depth_10() {
        let mut fix = Fixture::uniform(16, 16, 1, 1, 40);
        fix.bit_depth = 10;
        let mut got = noise_planes(&fix, 0x5eed_cafe, 1023);
        let mut want = got.clone();
        run_frame(&mut got, &fix);
        reference_raster(&mut want, &fix);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.data, w.data);
        }
    }

    #[test]
    #[should_panic(expected = "luma plane extent")]
    fn panics_when_luma_extent_disagrees_with_the_mi_grid() {
        let fix = Fixture::uniform(16, 16, 1, 1, 32);
        let mut planes = [
            PlaneBuf::filled(64, 128, 80), // (64 + 7) >> 3 = 8 != 16
            PlaneBuf::filled(64, 64, 90),
            PlaneBuf::filled(64, 64, 100),
        ];
        run_frame(&mut planes, &fix);
    }

    #[test]
    #[should_panic(expected = "chroma plane extent")]
    fn panics_when_a_chroma_extent_disagrees_with_the_subsampling() {
        let fix = Fixture::uniform(16, 16, 1, 1, 32);
        let mut planes = [
            PlaneBuf::filled(128, 128, 80),
            PlaneBuf::filled(128, 128, 90), // not subsampled
            PlaneBuf::filled(64, 64, 100),
        ];
        run_frame(&mut planes, &fix);
    }
}
