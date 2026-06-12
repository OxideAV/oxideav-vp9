//! Integration tests for the §8.8 `loop filter process` frame-level
//! driver — [`frame_loop_filter`] — at the public-API boundary, per
//! the VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8 lines 5436-5463).

use oxideav_vp9::{
    frame_loop_filter, loop_filter_frame_init, superblock_loop_filter, CurrFrame, LoopFilterParams,
    LvlLookup, SegmentationParams, SuperblockFilterFrame, SuperblockFilterPlane, MAX_SEGMENTS,
    SEG_LVL_MAX, TX_4X4,
};

/// `BLOCK_8X8 = 3` per §3 — the block-size enumeration index the
/// `MiSizes[ ][ ]` array carries.
const BLOCK_8X8: u8 = 3;

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
    /// A uniform MiRows x MiCols grid (BLOCK_8X8 / TX_4X4 / unskipped
    /// intra) with a step-3 broadcast LvlLookup at `level`.
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

    /// Deterministic pseudo-random fill in `0..=max`.
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

/// The §8.8 raster transcribed directly from the listing (lines
/// 5451-5455) over individual §8.8.2 calls.
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
fn flat_frame_is_identity() {
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
fn level_zero_is_identity_on_a_filterable_step() {
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
fn matches_the_transcribed_raster_on_a_noise_frame() {
    // 2x2 superblock grid, 4:2:0, order-sensitive content.
    let fix = Fixture::uniform(16, 16, 1, 1, 40);
    let mk = |seed: u32| {
        let mut s = seed;
        [
            PlaneBuf::noise(128, 128, &mut s, 255),
            PlaneBuf::noise(64, 64, &mut s, 255),
            PlaneBuf::noise(64, 64, &mut s, 255),
        ]
    };
    let mut got = mk(0x1234_5678);
    let mut want = got.clone();
    run_frame(&mut got, &fix);
    reference_raster(&mut want, &fix);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.data, w.data);
    }
    // Sanity: the filter actually moved samples.
    assert_ne!(got[0].data, mk(0x1234_5678)[0].data);
}

#[test]
fn matches_the_transcribed_raster_on_non_mi_aligned_extents() {
    // FrameWidth = 52, FrameHeight = 36 — MiCols = 7, MiRows = 5 per
    // §7.2.6, chroma 26x18 per §8.10.
    let fix = Fixture::uniform(7, 5, 1, 1, 40);
    let mut s = 0xdead_beefu32;
    let mut got = [
        PlaneBuf::noise(52, 36, &mut s, 255),
        PlaneBuf::noise(26, 18, &mut s, 255),
        PlaneBuf::noise(26, 18, &mut s, 255),
    ];
    let mut want = got.clone();
    run_frame(&mut got, &fix);
    reference_raster(&mut want, &fix);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.data, w.data);
    }
}

#[test]
fn filters_the_cross_superblock_boundary_and_routes_chroma() {
    // A luma step at x = 64 (the col = 0 / col = 8 superblock
    // boundary) and a U step at chroma x = 16 both filter; the flat
    // V plane is untouched.
    let fix = Fixture::uniform(16, 16, 1, 1, 32);
    let mut planes = [
        PlaneBuf::vstep(128, 128, 64, 60, 68),
        PlaneBuf::vstep(64, 64, 16, 60, 68),
        PlaneBuf::filled(64, 64, 100),
    ];
    run_frame(&mut planes, &fix);
    assert!(planes[0].get(63, 32) > 60);
    assert!(planes[0].get(64, 32) < 68);
    assert!(planes[1].get(15, 16) > 60);
    assert!(planes[1].get(16, 16) < 68);
    assert!(planes[2].data.iter().all(|&v| v == 100));
}

#[test]
#[should_panic(expected = "chroma plane extent")]
fn panics_on_inconsistent_chroma_extents() {
    let fix = Fixture::uniform(16, 16, 1, 1, 32);
    let mut planes = [
        PlaneBuf::filled(128, 128, 80),
        PlaneBuf::filled(128, 128, 90), // not subsampled
        PlaneBuf::filled(64, 64, 100),
    ];
    run_frame(&mut planes, &fix);
}
