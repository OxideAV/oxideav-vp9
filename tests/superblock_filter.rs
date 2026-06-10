//! Integration tests for the public §8.8.2 superblock-loop-filter
//! per-edge predicate API — `vp9-spec.txt` lines 5491-5586.
//!
//! Exercises the §8.8.2 steps 1-14 predicate derivation
//! ([`superblock_filter_edge`]) and the §8.8.2 `dx` / `dy` / `sub` /
//! `edgeLen` geometry header ([`superblock_filter_geometry`]) from a
//! public caller's perspective: drive the `edge` / `i` raster loop
//! with the geometry, resolve per-MI state at the returned
//! `(loop_row, loop_col)`, and confirm the `applyFilter` / `is32Edge`
//! gating the §8.8.2 steps 15-17 hand-off matches the listing.

use oxideav_vp9::{
    superblock_filter_edge, superblock_filter_geometry, SuperblockFilterGeometry,
    SuperblockFilterMi,
};

const TX_4X4: u8 = 0;
const TX_8X8: u8 = 1;
const TX_32X32: u8 = 3;

// §3 BLOCK_* constants used as MiSize inputs.
const BLOCK_8X8: u8 = 3;
const BLOCK_16X16: u8 = 6;
const BLOCK_64X64: u8 = 12;
const INTRA_FRAME: i32 = 0;

fn mi(mi_size: u8, tx_sz: u8, skip: bool, ref_frame_0: i32) -> SuperblockFilterMi {
    SuperblockFilterMi {
        mi_size,
        tx_sz,
        skip,
        ref_frame_0,
    }
}

#[test]
fn geometry_drives_loop_bounds() {
    // Luma: 16 vertical edges, 64-sample boundary.
    let g = superblock_filter_geometry(0, 0, 0);
    assert_eq!(
        g,
        SuperblockFilterGeometry {
            dx: 1,
            dy: 0,
            sub: 0,
            edge_len: 64,
        }
    );
    assert_eq!(16u32 >> g.sub, 16); // edge count
                                    // 4:2:0 chroma vertical: 8 edges, 32-sample boundary.
    let gc = superblock_filter_geometry(0, 1, 1);
    assert_eq!(16u32 >> gc.sub, 8);
    assert_eq!(gc.edge_len, 32);
}

#[test]
fn first_luma_superblock_row_left_edge_not_filtered() {
    // The leftmost vertical edge of the frame (col = 0, edge = 0,
    // x = 0) is off-screen per step 13, so applyFilter is false.
    let g = superblock_filter_geometry(0, 0, 0);
    for i in 0..g.edge_len {
        let e = superblock_filter_edge(
            0,
            0,
            0,
            0,
            i,
            0,
            0,
            64,
            64,
            &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
        );
        assert_eq!(e.x, 0);
        assert!(!e.on_screen);
        assert!(!e.apply_filter);
    }
}

#[test]
fn interior_block_edges_filter_regardless_of_skip() {
    // Walk the first superblock's vertical edges for an 8x8-partitioned
    // intra block. Every edge on a multiple-of-8 x (except x == 0) is a
    // block edge and must filter even when the block is skipped.
    let g = superblock_filter_geometry(0, 0, 0);
    let block = mi(BLOCK_8X8, TX_4X4, true, INTRA_FRAME);
    for edge in 1..(16u32 >> g.sub) {
        let e = superblock_filter_edge(0, 0, 0, edge, 0, 0, 0, 64, 64, &block);
        if e.x % 8 == 0 {
            assert!(e.is_block_edge, "edge {edge} x {}", e.x);
            assert!(e.apply_filter);
        }
    }
}

#[test]
fn is_32_edge_feeds_filter_size_input() {
    // is32Edge is the §8.8.3 input; true exactly on edge multiples of 8.
    for edge in 0..16 {
        let e = superblock_filter_edge(
            0,
            0,
            1,
            edge,
            0,
            0,
            0,
            64,
            64,
            &mi(BLOCK_64X64, TX_32X32, false, INTRA_FRAME),
        );
        assert_eq!(e.is_32_edge, edge % 8 == 0);
    }
}

#[test]
fn tx_edge_intra_vs_inter_skip_gating() {
    // BLOCK_64X64 (block boundary only every 64) with TX_4X4 (tx edge
    // every 4): a non-block tx edge filters for intra (always) and for
    // non-skip inter, but not for skipped inter.
    let intra = mi(BLOCK_64X64, TX_4X4, true, INTRA_FRAME);
    let inter_skip = mi(BLOCK_64X64, TX_4X4, true, 1);
    let inter_noskip = mi(BLOCK_64X64, TX_4X4, false, 1);

    // edge = 2 → x = 8: tx edge, not a 64-block edge.
    let e_intra = superblock_filter_edge(0, 0, 0, 2, 0, 0, 0, 64, 64, &intra);
    let e_is = superblock_filter_edge(0, 0, 0, 2, 0, 0, 0, 64, 64, &inter_skip);
    let e_ins = superblock_filter_edge(0, 0, 0, 2, 0, 0, 0, 64, 64, &inter_noskip);

    assert!(e_intra.is_tx_edge && !e_intra.is_block_edge);
    assert!(e_intra.apply_filter);
    assert!(!e_is.apply_filter);
    assert!(e_ins.apply_filter);
}

#[test]
fn chroma_loop_coords_are_even_aligned() {
    // 4:2:0 chroma: loopRow / loopCol are aligned down to even MI units.
    let g = superblock_filter_geometry(0, 1, 1);
    for edge in 0..(16u32 >> g.sub) {
        for i in [0u32, 5, 17, 31] {
            let e = superblock_filter_edge(
                0,
                2,
                2,
                edge,
                i,
                1,
                1,
                64,
                64,
                &mi(BLOCK_16X16, TX_8X8, false, INTRA_FRAME),
            );
            assert_eq!(e.loop_col & 1, 0);
            assert_eq!(e.loop_row & 1, 0);
        }
    }
}

#[test]
fn horizontal_pass_top_edge_off_screen() {
    // pass 1, row 0, edge 0 → y = 0, top of frame → off-screen.
    let e = superblock_filter_edge(
        1,
        0,
        3,
        0,
        4,
        0,
        0,
        64,
        64,
        &mi(BLOCK_8X8, TX_8X8, false, INTRA_FRAME),
    );
    assert_eq!(e.y, 0);
    assert!(!e.on_screen);
    assert!(!e.apply_filter);
}

#[test]
fn chroma_right_image_edge_tx_edge_suppressed() {
    // pass 1, subX = 1, MiCols = 5 (odd), edge odd, x+8 >= 40 → tx edge
    // suppressed (chroma horizontal boundary off the right image edge).
    let e = superblock_filter_edge(
        1,
        0,
        4,
        1,
        1,
        1,
        1,
        5,
        64,
        &mi(BLOCK_16X16, TX_4X4, false, INTRA_FRAME),
    );
    assert!(!e.is_tx_edge);
}
