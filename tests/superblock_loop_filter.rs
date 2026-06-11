//! Integration tests for the §8.8.2 `superblock loop filter process`
//! driver — [`superblock_loop_filter`] — at the public-API boundary,
//! per the VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.8.2 lines 5491-5586).

use oxideav_vp9::{
    adaptive_filter_strength, loop_filter_frame_init, sample_filtering, superblock_loop_filter,
    LoopFilterParams, LvlLookup, SampleFilterSamples, SegmentationParams, SuperblockFilterFrame,
    SuperblockFilterPlane, MAX_SEGMENTS, PASS_HORIZONTAL, PASS_VERTICAL, SEG_LVL_MAX, TX_4X4,
};

/// `BLOCK_8X8 = 3` per §3 — the block-size enumeration index the
/// `MiSizes[ ][ ]` array carries.
const BLOCK_8X8: u8 = 3;
/// `BLOCK_64X64 = 12` per §3.
const BLOCK_64X64: u8 = 12;

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
            y_modes: vec![0; n], // DC_PRED — §8.8.4 modeType 0
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

/// Every `p` sample `lo`, every `q` sample `hi` — the §8.8.5.1
/// stencil the driver gathers at a clean two-level step.
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

/// §8.8.5 identity on flat content: both passes leave a constant
/// plane untouched.
#[test]
fn flat_plane_survives_both_passes() {
    let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
    let mut pb = PlaneBuf::filled(64, 64, 128);
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);
    assert!(pb.data.iter().all(|&v| v == 128));
}

/// §8.8.2 step 17 `lvl > 0` gate: `loop_filter_level == 0` makes the
/// whole driver a no-op even across a sharp step.
#[test]
fn zero_level_is_a_no_op() {
    let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 0);
    let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
    let reference = pb.data.clone();
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);
    assert_eq!(pb.data, reference);
}

/// §8.8.2 steps 15-17 vertical thread cross-checked against the
/// §8.8.4 + §8.8.5 primitives invoked directly on the same stencil.
#[test]
fn vertical_step_matches_direct_primitives() {
    let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
    let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);

    let s = adaptive_filter_strength(&fx.lookup, 0, 0, 0, 0).unwrap();
    let out = sample_filtering(
        &two_level_stencil(60, 68),
        s.limit,
        s.blimit,
        s.thresh,
        TX_4X4,
        8,
    );
    assert_ne!(out.q0, 68, "the step must be inside the filter limits");
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

/// Mirror of the vertical cross-check for `pass == 1`.
#[test]
fn horizontal_step_matches_direct_primitives() {
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

/// §8.8.2 step 14 gating threaded end-to-end: a pure tx edge on a
/// skipped inter block does not filter; clearing `skip` (or making
/// the edge a block edge via BLOCK_8X8) does.
#[test]
fn skip_and_block_edge_gating() {
    // BLOCK_64X64 ⇒ x = 8 is a tx edge only. Skipped inter: no-op.
    let fx = Fixture::uniform(8, 8, BLOCK_64X64, TX_4X4, true, 1, 25);
    let mut pb = PlaneBuf::vstep(64, 64, 8, 60, 68);
    let reference = pb.data.clone();
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
    assert_eq!(pb.data, reference);

    // Same geometry, skip == 0: the tx edge filters.
    let fx2 = Fixture::uniform(8, 8, BLOCK_64X64, TX_4X4, false, 1, 25);
    let mut pb2 = PlaneBuf::vstep(64, 64, 8, 60, 68);
    superblock_loop_filter(&mut pb2.view(), &fx2.frame(), 0, PASS_VERTICAL, 0, 0);
    assert_ne!(pb2.get(8, 0), 68);

    // BLOCK_8X8 makes x = 8 a block edge: filters even when skipped.
    let fx3 = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, true, 1, 25);
    let mut pb3 = PlaneBuf::vstep(64, 64, 8, 60, 68);
    superblock_loop_filter(&mut pb3.view(), &fx3.frame(), 0, PASS_VERTICAL, 0, 0);
    assert_ne!(pb3.get(8, 0), 68);
}

/// §8.8.2 step 13 left / top frame-edge exclusion: a discontinuity
/// hugging `x == 0` / `y == 0` is never touched (the excluded edge
/// would trip the §8.8.5.2 hev branch if it ran).
#[test]
fn frame_edges_excluded() {
    let fx = Fixture::uniform(8, 8, BLOCK_8X8, TX_4X4, false, 0, 25);
    let mut pb = PlaneBuf::vstep(64, 64, 1, 60, 68);
    let reference = pb.data.clone();
    superblock_loop_filter(&mut pb.view(), &fx.frame(), 0, PASS_VERTICAL, 0, 0);
    assert_eq!(pb.data, reference);

    let mut pb2 = PlaneBuf::hstep(64, 64, 1, 60, 68);
    let reference2 = pb2.data.clone();
    superblock_loop_filter(&mut pb2.view(), &fx.frame(), 0, PASS_HORIZONTAL, 0, 0);
    assert_eq!(pb2.data, reference2);
}

/// 4:2:0 chroma plane: the luma `x = 8` boundary lands at chroma
/// column 4 and the narrow window (chroma columns 2..=5) matches the
/// direct §8.8.5 call; the clamped outer-ring reads leave columns
/// 0..=1 untouched.
#[test]
fn chroma_420_vertical_matches_direct_primitives() {
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

/// A frame ending mid-superblock (MiCols = MiRows = 6): the §8.8.2
/// raster still covers the 64x64 superblock but every `x >= 48` /
/// `y >= 48` position is off-screen — no out-of-bounds access, and
/// the interior step at `x = 40` still filters.
#[test]
fn partial_superblock_is_safe() {
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
    assert_eq!(pb.get(39, 0), out.p0);
    assert_eq!(pb.get(40, 0), out.q0);
}
