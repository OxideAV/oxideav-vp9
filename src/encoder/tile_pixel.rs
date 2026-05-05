//! VP9 keyframe tile encoder — pixel-encoding path (round 2).
//!
//! Encodes source YUV pixels into a valid VP9 keyframe tile payload.
//! Strategy per 4×4 block:
//! 1. DC_PRED intra prediction from already-encoded neighbours.
//! 2. Compute residual = source − prediction.
//! 3. Forward 4×4 DCT → quantise → token-encode residual.
//! 4. Emit skip=0 if any nonzero coefficients, skip=1 otherwise.
//!
//! The compressed header is fixed (tx_mode=ONLY_4X4, no prob updates),
//! so the decoder consumes the §10.5 default coefficient probabilities.
//!
//! The partition tree is the same as the MVP (PARTITION_NONE at every
//! 64×64 superblock, split at edges). The intra mode is DC_PRED for
//! every block, using the above-row / left-column chain seeded from
//! the reconstructed buffer.

use crate::compressed_header::TxMode;
use crate::encoder::bool_encoder::BoolEncoder;
use crate::encoder::fwdtransform::{fdct_2d, quantise};
use crate::encoder::params::EncoderParams;
use crate::encoder::tokenize::encode_coefs;
use crate::probs::KF_PARTITION_PROBS;
use crate::tables::{
    AC_QLOOKUP, COEFBAND_TRANS_4X4, COEF_PROBS_4X4, DC_QLOOKUP, DEFAULT_SCAN_4X4,
    DEFAULT_SCAN_4X4_NEIGHBORS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
};

const MODE_DC: usize = 0;
const SKIP_PROBS: [u8; 3] = [192, 128, 64];

/// Emit a pixel-encoding keyframe tile for one 4:2:0 8-bit frame.
///
/// `y_src` / `u_src` / `v_src` are the source planes in row-major order.
/// `y_stride` is the luma stride; `uv_stride` the chroma stride.
/// `width` / `height` are the frame dimensions.
pub fn emit_pixel_tile(
    p: &EncoderParams,
    y_src: &[u8],
    y_stride: usize,
    u_src: &[u8],
    v_src: &[u8],
    uv_stride: usize,
) -> Vec<u8> {
    let width = p.width as usize;
    let height = p.height as usize;
    let base_q_idx = p.base_q_idx as usize;
    let dq_dc = DC_QLOOKUP[base_q_idx];
    let dq_ac = AC_QLOOKUP[base_q_idx];

    // Reconstruction buffers (for prediction chaining).
    let recon_w = width.next_multiple_of(4);
    let recon_h = height.next_multiple_of(4);
    let mut recon_y = vec![128u8; recon_w * recon_h];
    let uv_w = (width + 1) / 2;
    let uv_h = (height + 1) / 2;
    let recon_uv_w = uv_w.next_multiple_of(4);
    let recon_uv_h = uv_h.next_multiple_of(4);
    let mut recon_u = vec![128u8; recon_uv_w * recon_uv_h];
    let mut recon_v = vec![128u8; recon_uv_w * recon_uv_h];

    // NonzeroContext for coefficient context derivation.
    let x4_cols_y = recon_w / 4;
    let x4_cols_uv = recon_uv_w / 4;
    let mut above_nz_y = vec![0u8; x4_cols_y];
    let mut above_nz_u = vec![0u8; x4_cols_uv];
    let mut above_nz_v = vec![0u8; x4_cols_uv];
    let mut left_nz_y = vec![0u8; recon_h / 4];
    let mut left_nz_u = vec![0u8; recon_uv_h / 4];
    let mut left_nz_v = vec![0u8; recon_uv_h / 4];

    let mut be = BoolEncoder::new();

    // Partition context state (same as MVP tile encoder).
    let mi_cols = width.div_ceil(8);
    let mi_rows = height.div_ceil(8);
    let mut part_above = vec![0u8; mi_cols];
    let mut part_left = vec![0u8; mi_rows];
    let mut skip_above = vec![false; mi_cols];
    let mut skip_left = vec![false; mi_rows];

    let sb_cols = p.width.div_ceil(64) as usize;
    let sb_rows = p.height.div_ceil(64) as usize;

    for sby in 0..sb_rows {
        // Clear left nonzero context at each superblock row.
        left_nz_y.iter_mut().for_each(|v| *v = 0);
        left_nz_u.iter_mut().for_each(|v| *v = 0);
        left_nz_v.iter_mut().for_each(|v| *v = 0);

        for sbx in 0..sb_cols {
            let col = sbx * 64;
            let row = sby * 64;
            emit_sb(
                &mut be,
                &mut part_above,
                &mut part_left,
                &mut skip_above,
                &mut skip_left,
                row as u32,
                col as u32,
                64,
                p.width,
                p.height,
                // Pixel encoding state.
                y_src,
                y_stride,
                u_src,
                v_src,
                uv_stride,
                width,
                height,
                uv_w,
                uv_h,
                dq_dc,
                dq_ac,
                &mut recon_y,
                recon_w,
                &mut recon_u,
                &mut recon_v,
                recon_uv_w,
                &mut above_nz_y,
                &mut above_nz_u,
                &mut above_nz_v,
                &mut left_nz_y,
                &mut left_nz_u,
                &mut left_nz_v,
            );
        }
    }

    be.finish()
}

#[allow(clippy::too_many_arguments)]
fn emit_sb(
    be: &mut BoolEncoder,
    part_above: &mut Vec<u8>,
    part_left: &mut Vec<u8>,
    skip_above: &mut Vec<bool>,
    skip_left: &mut Vec<bool>,
    row: u32,
    col: u32,
    bsize: u32,
    frame_w: u32,
    frame_h: u32,
    // Pixel state.
    y_src: &[u8],
    y_stride: usize,
    u_src: &[u8],
    v_src: &[u8],
    uv_stride: usize,
    width: usize,
    height: usize,
    uv_w: usize,
    uv_h: usize,
    dq_dc: i16,
    dq_ac: i16,
    recon_y: &mut Vec<u8>,
    recon_yw: usize,
    recon_u: &mut Vec<u8>,
    recon_v: &mut Vec<u8>,
    recon_uvw: usize,
    above_nz_y: &mut Vec<u8>,
    above_nz_u: &mut Vec<u8>,
    above_nz_v: &mut Vec<u8>,
    left_nz_y: &mut Vec<u8>,
    left_nz_u: &mut Vec<u8>,
    left_nz_v: &mut Vec<u8>,
) {
    if row >= frame_h || col >= frame_w {
        return;
    }
    let on_right = col + bsize > frame_w;
    let on_bottom = row + bsize > frame_h;
    let mi_row = (row as usize) / 8;
    let mi_col = (col as usize) / 8;
    let probs = lookup_partition(part_above, part_left, bsize, mi_row, mi_col);
    let half = bsize / 2;

    // Edge / interior partition logic (same as MVP tile.rs).
    if on_right && on_bottom {
        if bsize == 8 {
            // Leaf block at corner.
            emit_block_at(
                be, skip_above, skip_left, mi_row, mi_col, bsize, y_src, y_stride, u_src, v_src,
                uv_stride, width, height, uv_w, uv_h, dq_dc, dq_ac, recon_y, recon_yw, recon_u,
                recon_v, recon_uvw, above_nz_y, above_nz_u, above_nz_v, left_nz_y, left_nz_u,
                left_nz_v,
            );
            update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_sb(
            be, part_above, part_left, skip_above, skip_left, row, col, half, frame_w, frame_h,
            y_src, y_stride, u_src, v_src, uv_stride, width, height, uv_w, uv_h, dq_dc, dq_ac,
            recon_y, recon_yw, recon_u, recon_v, recon_uvw, above_nz_y, above_nz_u, above_nz_v,
            left_nz_y, left_nz_u, left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row,
            col + half,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row + half,
            col,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        return;
    }
    if on_right {
        be.write(1, probs[2]);
        if bsize == 8 {
            emit_block_at(
                be, skip_above, skip_left, mi_row, mi_col, bsize, y_src, y_stride, u_src, v_src,
                uv_stride, width, height, uv_w, uv_h, dq_dc, dq_ac, recon_y, recon_yw, recon_u,
                recon_v, recon_uvw, above_nz_y, above_nz_u, above_nz_v, left_nz_y, left_nz_u,
                left_nz_v,
            );
            update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_sb(
            be, part_above, part_left, skip_above, skip_left, row, col, half, frame_w, frame_h,
            y_src, y_stride, u_src, v_src, uv_stride, width, height, uv_w, uv_h, dq_dc, dq_ac,
            recon_y, recon_yw, recon_u, recon_v, recon_uvw, above_nz_y, above_nz_u, above_nz_v,
            left_nz_y, left_nz_u, left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row,
            col + half,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row + half,
            col,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        return;
    }
    if on_bottom {
        be.write(1, probs[1]);
        if bsize == 8 {
            emit_block_at(
                be, skip_above, skip_left, mi_row, mi_col, bsize, y_src, y_stride, u_src, v_src,
                uv_stride, width, height, uv_w, uv_h, dq_dc, dq_ac, recon_y, recon_yw, recon_u,
                recon_v, recon_uvw, above_nz_y, above_nz_u, above_nz_v, left_nz_y, left_nz_u,
                left_nz_v,
            );
            update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_sb(
            be, part_above, part_left, skip_above, skip_left, row, col, half, frame_w, frame_h,
            y_src, y_stride, u_src, v_src, uv_stride, width, height, uv_w, uv_h, dq_dc, dq_ac,
            recon_y, recon_yw, recon_u, recon_v, recon_uvw, above_nz_y, above_nz_u, above_nz_v,
            left_nz_y, left_nz_u, left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row,
            col + half,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row + half,
            col,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            y_src,
            y_stride,
            u_src,
            v_src,
            uv_stride,
            width,
            height,
            uv_w,
            uv_h,
            dq_dc,
            dq_ac,
            recon_y,
            recon_yw,
            recon_u,
            recon_v,
            recon_uvw,
            above_nz_y,
            above_nz_u,
            above_nz_v,
            left_nz_y,
            left_nz_u,
            left_nz_v,
        );
        return;
    }
    // Interior PARTITION_NONE.
    be.write(0, probs[0]);
    emit_block_at(
        be, skip_above, skip_left, mi_row, mi_col, bsize, y_src, y_stride, u_src, v_src, uv_stride,
        width, height, uv_w, uv_h, dq_dc, dq_ac, recon_y, recon_yw, recon_u, recon_v, recon_uvw,
        above_nz_y, above_nz_u, above_nz_v, left_nz_y, left_nz_u, left_nz_v,
    );
    update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
}

fn lookup_partition(
    above: &[u8],
    left: &[u8],
    bsize: u32,
    mi_row: usize,
    mi_col: usize,
) -> [u8; 3] {
    let bsl = match bsize {
        8 => 0usize,
        16 => 1,
        32 => 2,
        64 => 3,
        _ => 3,
    };
    let num8x8 = (bsize as usize) / 8;
    let boffset = 3 - bsl;
    let mut a = 0u8;
    let mut l = 0u8;
    for i in 0..num8x8 {
        let c = mi_col + i;
        if c < above.len() {
            a |= above[c];
        }
        let r = mi_row + i;
        if r < left.len() {
            l |= left[r];
        }
    }
    let ai = ((a >> boffset) & 1) as usize;
    let li = ((l >> boffset) & 1) as usize;
    let ctx = bsl * 4 + li * 2 + ai;
    KF_PARTITION_PROBS[ctx]
}

fn update_partition_ctx(
    above: &mut Vec<u8>,
    left: &mut Vec<u8>,
    bsize_px: u32,
    sub_w: u32,
    sub_h: u32,
    mi_row: usize,
    mi_col: usize,
) {
    let num8x8 = (bsize_px as usize) / 8;
    let bwlog = match sub_w {
        4 => 0u8,
        8 => 1,
        16 => 2,
        32 => 3,
        64 => 4,
        _ => 4,
    };
    let bhlog = match sub_h {
        4 => 0u8,
        8 => 1,
        16 => 2,
        32 => 3,
        64 => 4,
        _ => 4,
    };
    let af = 15u8 >> bwlog;
    let lf = 15u8 >> bhlog;
    for i in 0..num8x8 {
        let c = mi_col + i;
        if c < above.len() {
            above[c] = af;
        }
        let r = mi_row + i;
        if r < left.len() {
            left[r] = lf;
        }
    }
}

/// Emit skip + modes + coefficients for one block of size `bsize_px × bsize_px`.
///
/// For ONLY_4X4, the prediction block matches the block size, but the
/// TX unit is always 4×4. For bsize=64 that's 16×16=256 4×4 TX units;
/// for bsize=8 that's 4 4×4 TX units.
#[allow(clippy::too_many_arguments)]
fn emit_block_at(
    be: &mut BoolEncoder,
    skip_above: &mut Vec<bool>,
    skip_left: &mut Vec<bool>,
    mi_row: usize,
    mi_col: usize,
    bsize_px: u32,
    y_src: &[u8],
    y_stride: usize,
    u_src: &[u8],
    v_src: &[u8],
    uv_stride: usize,
    width: usize,
    height: usize,
    uv_w: usize,
    uv_h: usize,
    dq_dc: i16,
    dq_ac: i16,
    recon_y: &mut Vec<u8>,
    recon_yw: usize,
    recon_u: &mut Vec<u8>,
    recon_v: &mut Vec<u8>,
    recon_uvw: usize,
    above_nz_y: &mut Vec<u8>,
    above_nz_u: &mut Vec<u8>,
    above_nz_v: &mut Vec<u8>,
    left_nz_y: &mut Vec<u8>,
    left_nz_u: &mut Vec<u8>,
    left_nz_v: &mut Vec<u8>,
) {
    let bs = bsize_px as usize;
    let px_col = mi_col * 8;
    let px_row = mi_row * 8;

    // Compute skip context and whether this block is skip (all-zero residual).
    // First, compute all residuals and check if any are nonzero.
    // For simplicity, compute residuals for all 4×4 sub-blocks and check eob.

    // Luma 4×4 sub-blocks.
    let n4 = bs / 4; // number of 4×4 blocks per dimension
    let mut all_skip = true;
    let mut luma_coefs: Vec<Vec<i32>> = Vec::new(); // scan-order quantised for each 4×4
    let mut luma_eobs: Vec<usize> = Vec::new();
    let mut luma_ictx: Vec<usize> = Vec::new(); // initial_ctx per block

    for ty in 0..n4 {
        for tx in 0..n4 {
            let bx = px_col + tx * 4;
            let by = px_row + ty * 4;
            let (coefs_scan, eob, ictx) = encode_4x4_block(
                y_src, y_stride, width, height, bx, by, dq_dc, dq_ac, recon_y, recon_yw,
                above_nz_y, left_nz_y,
            );
            if eob > 0 {
                all_skip = false;
            }
            luma_coefs.push(coefs_scan);
            luma_eobs.push(eob);
            luma_ictx.push(ictx);
        }
    }

    // Chroma sub-blocks (one 4×4 per 8×8 luma, i.e. n4/2 per dimension for 4:2:0).
    let n4_uv = (bs / 2) / 4; // chroma 4×4 per dimension
    let mut u_coefs: Vec<Vec<i32>> = Vec::new();
    let mut u_eobs: Vec<usize> = Vec::new();
    let mut u_ictx: Vec<usize> = Vec::new();
    let mut v_coefs: Vec<Vec<i32>> = Vec::new();
    let mut v_eobs: Vec<usize> = Vec::new();
    let mut v_ictx: Vec<usize> = Vec::new();

    for ty in 0..n4_uv.max(1) {
        for tx in 0..n4_uv.max(1) {
            let bx = px_col / 2 + tx * 4;
            let by = px_row / 2 + ty * 4;
            let (cu, eu, icu) = encode_4x4_block(
                u_src, uv_stride, uv_w, uv_h, bx, by, dq_dc, dq_ac, recon_u, recon_uvw, above_nz_u,
                left_nz_u,
            );
            let (cv, ev, icv) = encode_4x4_block(
                v_src, uv_stride, uv_w, uv_h, bx, by, dq_dc, dq_ac, recon_v, recon_uvw, above_nz_v,
                left_nz_v,
            );
            if eu > 0 || ev > 0 {
                all_skip = false;
            }
            u_coefs.push(cu);
            u_eobs.push(eu);
            u_ictx.push(icu);
            v_coefs.push(cv);
            v_eobs.push(ev);
            v_ictx.push(icv);
        }
    }

    // Emit skip.
    let sctx = {
        let a = if px_col > 0 && mi_col < skip_above.len() {
            skip_above[mi_col] as usize
        } else {
            0
        };
        let l = if px_row > 0 && mi_row < skip_left.len() {
            skip_left[mi_row] as usize
        } else {
            0
        };
        (a + l).min(2)
    };
    be.write(all_skip as u32, SKIP_PROBS[sctx]);

    // Update skip context.
    let mi_w = bs / 8;
    let mi_h = bs / 8;
    for i in 0..mi_w.max(1) {
        let c = mi_col + i;
        if c < skip_above.len() {
            skip_above[c] = all_skip;
        }
    }
    for i in 0..mi_h.max(1) {
        let r = mi_row + i;
        if r < skip_left.len() {
            skip_left[r] = all_skip;
        }
    }

    // Emit luma mode (DC_PRED).
    let p = &KF_Y_MODE_PROBS[MODE_DC][MODE_DC];
    emit_intra_mode_tree(be, p, MODE_DC);

    // Emit UV mode (DC_PRED).
    let puv = &KF_UV_MODE_PROBS[MODE_DC];
    emit_intra_mode_tree(be, puv, MODE_DC);

    if all_skip {
        return;
    }

    // Emit luma coefficients.
    let coef_probs = &COEF_PROBS_4X4[0][0]; // intra, Y-plane
    for ((coefs, eob), ictx) in luma_coefs
        .iter()
        .zip(luma_eobs.iter())
        .zip(luma_ictx.iter())
    {
        encode_coefs(
            be,
            coef_probs,
            &DEFAULT_SCAN_4X4,
            &DEFAULT_SCAN_4X4_NEIGHBORS,
            &COEFBAND_TRANS_4X4,
            coefs,
            *eob,
            *ictx,
        );
    }

    // Emit chroma coefficients.
    // plane_type=1 for UV (not Y=0), ref_type=0 for intra.
    let coef_probs_uv = &COEF_PROBS_4X4[1][0]; // UV-plane, intra
    for ((coefs, eob), ictx) in u_coefs.iter().zip(u_eobs.iter()).zip(u_ictx.iter()) {
        encode_coefs(
            be,
            coef_probs_uv,
            &DEFAULT_SCAN_4X4,
            &DEFAULT_SCAN_4X4_NEIGHBORS,
            &COEFBAND_TRANS_4X4,
            coefs,
            *eob,
            *ictx,
        );
    }
    for ((coefs, eob), ictx) in v_coefs.iter().zip(v_eobs.iter()).zip(v_ictx.iter()) {
        encode_coefs(
            be,
            coef_probs_uv,
            &DEFAULT_SCAN_4X4,
            &DEFAULT_SCAN_4X4_NEIGHBORS,
            &COEFBAND_TRANS_4X4,
            coefs,
            *eob,
            *ictx,
        );
    }
}

/// Encode one 4×4 block of a plane.
///
/// Performs DC_PRED from `recon` buffer, computes residual, forward DCT,
/// quantises, and reconstructs in `recon`. Returns scan-order quantised
/// coefficients, eob count, and the initial_ctx for coefficient entropy
/// coding (derived from above/left NonzeroContext BEFORE this block updates them).
fn encode_4x4_block(
    src: &[u8],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    bx: usize,
    by: usize,
    dq_dc: i16,
    dq_ac: i16,
    recon: &mut Vec<u8>,
    recon_w: usize,
    above_nz: &mut Vec<u8>,
    left_nz: &mut Vec<u8>,
) -> (Vec<i32>, usize, usize) {
    // Compute initial_ctx from above+left BEFORE this block's update.
    // Mirrors NonzeroCtx::token_ctx: ctx = above_nz[x4] + left_nz[y4].
    let x4 = bx / 4;
    let y4 = by / 4;
    let above = if x4 < above_nz.len() {
        above_nz[x4] as usize
    } else {
        0
    };
    let left = if y4 < left_nz.len() {
        left_nz[y4] as usize
    } else {
        0
    };
    let initial_ctx = (above + left).min(2);

    // DC prediction from recon buffer.
    let pred = dc_pred_4x4(recon, recon_w, bx, by);

    // Gather source 4×4 block.
    let mut residual = [0i16; 16];
    for r in 0..4 {
        for c in 0..4 {
            let px = (bx + c).min(src_w - 1);
            let py = (by + r).min(src_h - 1);
            let s = src[py * src_stride + px] as i16;
            residual[r * 4 + c] = s - pred as i16;
        }
    }

    // Forward DCT.
    let mut coeffs_raster = [0i32; 16];
    fdct_2d(&residual, 4, &mut coeffs_raster);

    // Quantise (in-place, scan order).
    let mut coeffs_scan = vec![0i32; 16];
    coeffs_scan.copy_from_slice(&coeffs_raster);
    let eob = quantise(&mut coeffs_scan, &DEFAULT_SCAN_4X4, dq_dc, dq_ac);

    // Reconstruct into recon buffer for next-block prediction chaining.
    // Dequantise and inverse DCT.
    let mut dequant = [0i32; 16];
    for (i, &scan_idx) in DEFAULT_SCAN_4X4.iter().enumerate() {
        let q = coeffs_scan[i];
        let dq = if i == 0 { dq_dc as i32 } else { dq_ac as i32 };
        dequant[scan_idx as usize] = q * dq;
    }
    use crate::transform::{inverse_transform_add, TxType};
    let mut recon_u8 = [pred; 16];
    inverse_transform_add(TxType::DctDct, 4, 4, &dequant, &mut recon_u8, 4).ok();
    // Copy back into recon buffer.
    for r in 0..4 {
        for c in 0..4 {
            let rx = bx + c;
            let ry = by + r;
            if rx < recon_w && ry < (recon.len() / recon_w) {
                recon[ry * recon_w + rx] = recon_u8[r * 4 + c];
            }
        }
    }

    // Update NonzeroContext AFTER reconstruction so downstream blocks see it.
    let nz = if eob > 0 { 1u8 } else { 0u8 };
    if x4 < above_nz.len() {
        above_nz[x4] = nz;
    }
    if y4 < left_nz.len() {
        left_nz[y4] = nz;
    }

    (coeffs_scan, eob, initial_ctx)
}

/// DC prediction for a 4×4 block from the recon buffer.
/// Returns the prediction sample value (uniform for DC_PRED).
fn dc_pred_4x4(recon: &[u8], recon_w: usize, bx: usize, by: usize) -> u8 {
    let have_above = by > 0;
    let have_left = bx > 0;
    match (have_above, have_left) {
        (true, true) => {
            let sa: u32 = (0..4)
                .map(|c| recon[(by - 1) * recon_w + bx + c] as u32)
                .sum();
            let sl: u32 = (0..4)
                .map(|r| recon[(by + r) * recon_w + bx - 1] as u32)
                .sum();
            ((sa + sl + 4) / 8) as u8
        }
        (true, false) => {
            let sa: u32 = (0..4)
                .map(|c| recon[(by - 1) * recon_w + bx + c] as u32)
                .sum();
            ((sa + 2) / 4) as u8
        }
        (false, true) => {
            let sl: u32 = (0..4)
                .map(|r| recon[(by + r) * recon_w + bx - 1] as u32)
                .sum();
            ((sl + 2) / 4) as u8
        }
        (false, false) => 128,
    }
}

/// Emit an intra mode symbol `m` against a 9-prob tree.
/// Mirrors `encoder/tile.rs::emit_intra_mode_tree`.
fn emit_intra_mode_tree(be: &mut BoolEncoder, p: &[u8; 9], m: usize) {
    match m {
        0 => be.write(0, p[0]),
        9 => {
            be.write(1, p[0]);
            be.write(0, p[1]);
        }
        1 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(0, p[2]);
        }
        2 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(0, p[3]);
            be.write(0, p[4]);
        }
        4 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(0, p[3]);
            be.write(1, p[4]);
            be.write(0, p[5]);
        }
        5 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(0, p[3]);
            be.write(1, p[4]);
            be.write(1, p[5]);
        }
        3 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(0, p[6]);
        }
        8 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(1, p[6]);
            be.write(0, p[7]);
        }
        6 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(1, p[6]);
            be.write(1, p[7]);
            be.write(0, p[8]);
        }
        7 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(1, p[6]);
            be.write(1, p[7]);
            be.write(1, p[8]);
        }
        _ => unreachable!("invalid intra mode {m}"),
    }
}

/// Build a complete VP9 keyframe encoding source pixels.
/// Returns the final byte buffer (uh + ch + tile).
pub fn build_pixel_keyframe(
    p: &EncoderParams,
    y: &[u8],
    y_stride: usize,
    u: &[u8],
    v: &[u8],
    uv_stride: usize,
) -> Vec<u8> {
    use crate::encoder::compressed_header::emit_compressed_header;
    use crate::encoder::uncompressed_header::emit_uncompressed_header;

    let ch = emit_compressed_header(TxMode::Only4x4, false);
    let tile = emit_pixel_tile(p, y, y_stride, u, v, uv_stride);
    let uh = emit_uncompressed_header(p, ch.len() as u16);
    let mut out = Vec::with_capacity(uh.len() + ch.len() + tile.len());
    out.extend_from_slice(&uh);
    out.extend_from_slice(&ch);
    out.extend_from_slice(&tile);
    out
}
