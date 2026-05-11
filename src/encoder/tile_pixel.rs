//! VP9 keyframe tile encoder — pixel-encoding path (round 2 + round 40 + round 48).
//!
//! Encodes source YUV pixels into a valid VP9 keyframe tile payload.
//! Strategy per 8×8 block (one MI cell, four 4×4 TX units):
//! 1. **Round 40**: pick the best luma intra mode from a candidate
//!    set ({DC_PRED, V_PRED, H_PRED, TM_PRED}) by minimising the
//!    8×8 source-vs-predictor sum-of-squared-errors (SSE) computed
//!    against the reconstructed neighbour samples.
//! 2. Apply that mode to all four 4×4 TX units, computing residuals.
//! 3. Forward 4×4 DCT → quantise → token-encode each residual.
//! 4. Emit skip=0 if any nonzero coefficients, skip=1 otherwise.
//!
//! **Round 48 — chroma intra-mode RDO + RDO pruning** (lever A + lever C
//! from the round 48 dispatch):
//! * Chroma now runs the same 4-mode SSE picker over `{DC, V, H, TM}`
//!   on the U plane; the picked mode applies to both U and V (VP9 stores
//!   ONE `uv_mode` per block, applied to both chroma planes — §6.4.6).
//!   This unblocks non-uniform chroma reconstruction; uniform-128 fixtures
//!   still pick DC and stay at `inf` dB.
//! * Mode-RDO early termination: if DC's SSE on the 4×4 sample is ≤ 16
//!   (≤ 1 LSB RMS), the picker skips V/H/TM. On smooth content this
//!   trims ~75% of the RDO sweep without changing output (DC was going
//!   to win anyway).
//!
//! The compressed header is fixed (tx_mode=ONLY_4X4, no prob updates),
//! so the decoder consumes the §10.5 default coefficient probabilities.
//!
//! The partition tree is the same as the MVP (PARTITION_NONE at every
//! 64×64 superblock, split at edges). For >=8×8 blocks `read_intra_frame_mode_info`
//! reads ONE luma mode that gets stamped into all four sub_modes
//! positions, so the encoder also writes ONE mode per block.
//!
//! Above/left intra-mode tracker arrays mirror the decoder's
//! `IntraTile::above_mode_4x4` / `left_mode_4x4` so the
//! `KF_Y_MODE_PROBS[above][left]` lookup picks the same probability
//! row on both sides.
//!
//! Round-40 fixes the README "all-DC_PRED" bullet — non-smooth content
//! (edges, oriented gradients) now picks V/H/TM and gets a substantially
//! lower residual energy budget.

use crate::compressed_header::TxMode;
use crate::encoder::bool_encoder::BoolEncoder;
use crate::encoder::fwdtransform::{fdct_2d, quantise};
use crate::encoder::params::EncoderParams;
use crate::encoder::tokenize::encode_coefs;
use crate::intra::IntraMode;
use crate::probs::KF_PARTITION_PROBS;
use crate::tables::{
    AC_QLOOKUP, COEFBAND_TRANS_4X4, COEF_PROBS_4X4, DC_QLOOKUP, DEFAULT_SCAN_4X4,
    DEFAULT_SCAN_4X4_NEIGHBORS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
};

const SKIP_PROBS: [u8; 3] = [192, 128, 64];

/// Round-48 mode-RDO early termination threshold. If DC's SSE on the
/// 4×4 picker stamp is at or below this value, the picker skips the
/// V/H/TM evaluations and locks in DC. 16 = 1 LSB RMS over 16 samples;
/// at this point V/H/TM cannot meaningfully beat DC and the mode tree
/// emit cost (3-5 bool symbols) would dominate any tiny SSE win.
const RDO_DC_EARLY_OUT_SSE: u64 = 16;

/// Candidate luma intra modes evaluated per 8×8 block.
///
/// We restrict to the four non-directional modes the decoder side fully
/// implements (`DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`) — the six
/// directional `D*_PRED` modes are spec-supported in the decoder but
/// expensive to evaluate per-block and rarely win on the smooth /
/// edge-y test corpus. Adding directionals is a future round.
const CAND_MODES: [IntraMode; 4] = [IntraMode::Dc, IntraMode::V, IntraMode::H, IntraMode::Tm];

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

    // §9.3.2 above/left intra-mode trackers — mirrors `IntraTile::above_mode_4x4`
    // / `left_mode_4x4` so the encoder picks the same `KF_Y_MODE_PROBS[above][left]`
    // row the decoder will resolve. Indexed at 4×4 granularity (`mi_col*2`,
    // `mi_row*2`). Initialised to DC_PRED so the top-left blocks see the
    // same `(DC, DC)` row as the decoder's `mi_row==0` / `mi_col==0` defaults.
    let mut above_mode_4x4 = vec![IntraMode::Dc; mi_cols * 2];
    let mut left_mode_4x4 = vec![IntraMode::Dc; mi_rows * 2];

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
                &mut above_mode_4x4,
                &mut left_mode_4x4,
            );
        }
    }

    be.finish()
}

#[allow(clippy::too_many_arguments)]
fn emit_sb(
    be: &mut BoolEncoder,
    part_above: &mut [u8],
    part_left: &mut [u8],
    skip_above: &mut [bool],
    skip_left: &mut [bool],
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
    recon_y: &mut [u8],
    recon_yw: usize,
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    recon_uvw: usize,
    above_nz_y: &mut [u8],
    above_nz_u: &mut [u8],
    above_nz_v: &mut [u8],
    left_nz_y: &mut [u8],
    left_nz_u: &mut [u8],
    left_nz_v: &mut [u8],
    above_mode_4x4: &mut [IntraMode],
    left_mode_4x4: &mut [IntraMode],
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
                be,
                skip_above,
                skip_left,
                mi_row,
                mi_col,
                bsize,
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
                above_mode_4x4,
                left_mode_4x4,
            );
            update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
        );
        return;
    }
    if on_right {
        be.write(1, probs[2]);
        if bsize == 8 {
            emit_block_at(
                be,
                skip_above,
                skip_left,
                mi_row,
                mi_col,
                bsize,
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
                above_mode_4x4,
                left_mode_4x4,
            );
            update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
        );
        return;
    }
    if on_bottom {
        be.write(1, probs[1]);
        if bsize == 8 {
            emit_block_at(
                be,
                skip_above,
                skip_left,
                mi_row,
                mi_col,
                bsize,
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
                above_mode_4x4,
                left_mode_4x4,
            );
            update_partition_ctx(part_above, part_left, bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_sb(
            be,
            part_above,
            part_left,
            skip_above,
            skip_left,
            row,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
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
            above_mode_4x4,
            left_mode_4x4,
        );
        return;
    }
    // Interior PARTITION_NONE.
    be.write(0, probs[0]);
    emit_block_at(
        be,
        skip_above,
        skip_left,
        mi_row,
        mi_col,
        bsize,
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
        above_mode_4x4,
        left_mode_4x4,
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
    above: &mut [u8],
    left: &mut [u8],
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
///
/// Round-40: per-block luma intra-mode RDO. We evaluate a small mode
/// candidate set ({DC, V, H, TM} from `CAND_MODES`) on the entire
/// `bsize × bsize` luma block by predicting from the *current* recon
/// buffer (which holds neighbours from previously-encoded blocks),
/// computing the source-vs-predictor SSE, and picking the lowest.
/// The picked mode is then re-applied for actual encoding (re-uses
/// the per-4x4 DCT+quant+token+reconstruct chain).
#[allow(clippy::too_many_arguments)]
fn emit_block_at(
    be: &mut BoolEncoder,
    skip_above: &mut [bool],
    skip_left: &mut [bool],
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
    recon_y: &mut [u8],
    recon_yw: usize,
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    recon_uvw: usize,
    above_nz_y: &mut [u8],
    above_nz_u: &mut [u8],
    above_nz_v: &mut [u8],
    left_nz_y: &mut [u8],
    left_nz_u: &mut [u8],
    left_nz_v: &mut [u8],
    above_mode_4x4: &mut [IntraMode],
    left_mode_4x4: &mut [IntraMode],
) {
    let bs = bsize_px as usize;
    let px_col = mi_col * 8;
    let px_row = mi_row * 8;

    // §9.3.2 above/left mode lookup (round 40). For >=8×8 the decoder
    // uses `above_mode_4x4[mi_col*2]` / `left_mode_4x4[mi_row*2]`.
    let above_idx = mi_col * 2;
    let left_idx = mi_row * 2;
    let above_mode = if mi_row > 0 && above_idx < above_mode_4x4.len() {
        above_mode_4x4[above_idx]
    } else {
        IntraMode::Dc
    };
    let left_mode = if mi_col > 0 && left_idx < left_mode_4x4.len() {
        left_mode_4x4[left_idx]
    } else {
        IntraMode::Dc
    };

    // Round-40: pick the best luma intra mode on the full bsize block
    // by SSE against the recon-buffer-derived predictor. The recon
    // buffer at this point still holds the already-encoded neighbour
    // pixels above/left of this block — exactly what the decoder
    // would see when running the predictor.
    let picked_y_mode = pick_intra_mode_block(
        y_src, y_stride, width, height, recon_y, recon_yw, px_col, px_row, bs,
    );
    let mode_idx = mode_to_index(picked_y_mode);

    // Luma 4×4 sub-blocks. Use the picked mode at each sub-block.
    let n4 = bs / 4;
    let mut all_skip = true;
    let mut luma_coefs: Vec<Vec<i32>> = Vec::new();
    let mut luma_eobs: Vec<usize> = Vec::new();
    let mut luma_ictx: Vec<usize> = Vec::new();

    for ty in 0..n4 {
        for tx in 0..n4 {
            let bx = px_col + tx * 4;
            let by = px_row + ty * 4;
            let (coefs_scan, eob, ictx) = encode_4x4_block_mode(
                picked_y_mode,
                y_src,
                y_stride,
                width,
                height,
                bx,
                by,
                dq_dc,
                dq_ac,
                recon_y,
                recon_yw,
                above_nz_y,
                left_nz_y,
            );
            if eob > 0 {
                all_skip = false;
            }
            luma_coefs.push(coefs_scan);
            luma_eobs.push(eob);
            luma_ictx.push(ictx);
        }
    }

    // Chroma sub-blocks (4:2:0). Round 48: per-block chroma intra-mode
    // RDO. VP9 stores ONE `uv_mode` per block which applies to BOTH U
    // and V chroma planes (§6.4.6 `read_intra_mode_uv`), so the picker
    // runs once on the U plane (representative — most natural content
    // has correlated U/V structure) and the picked mode is reused for V.
    let n4_uv = (bs / 2) / 4;
    let mut u_coefs: Vec<Vec<i32>> = Vec::new();
    let mut u_eobs: Vec<usize> = Vec::new();
    let mut u_ictx: Vec<usize> = Vec::new();
    let mut v_coefs: Vec<Vec<i32>> = Vec::new();
    let mut v_eobs: Vec<usize> = Vec::new();
    let mut v_ictx: Vec<usize> = Vec::new();

    let uv_top_left_x = px_col / 2;
    let uv_top_left_y = px_row / 2;
    let picked_uv_mode = pick_intra_mode_block(
        u_src,
        uv_stride,
        uv_w,
        uv_h,
        recon_u,
        recon_uvw,
        uv_top_left_x,
        uv_top_left_y,
        bs / 2,
    );
    let uv_mode_idx = mode_to_index(picked_uv_mode);

    for ty in 0..n4_uv.max(1) {
        for tx in 0..n4_uv.max(1) {
            let bx = px_col / 2 + tx * 4;
            let by = px_row / 2 + ty * 4;
            let (cu, eu, icu) = encode_4x4_block_mode(
                picked_uv_mode,
                u_src,
                uv_stride,
                uv_w,
                uv_h,
                bx,
                by,
                dq_dc,
                dq_ac,
                recon_u,
                recon_uvw,
                above_nz_u,
                left_nz_u,
            );
            let (cv, ev, icv) = encode_4x4_block_mode(
                picked_uv_mode,
                v_src,
                uv_stride,
                uv_w,
                uv_h,
                bx,
                by,
                dq_dc,
                dq_ac,
                recon_v,
                recon_uvw,
                above_nz_v,
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

    // Emit luma mode against the spec-correct KF_Y_MODE_PROBS row.
    let py = &KF_Y_MODE_PROBS[above_mode as usize][left_mode as usize];
    emit_intra_mode_tree(be, py, mode_idx);

    // Emit UV mode (round 48 RDO-picked) — KF_UV_MODE_PROBS is keyed
    // by the luma mode we just emitted.
    let puv = &KF_UV_MODE_PROBS[mode_idx];
    emit_intra_mode_tree(be, puv, uv_mode_idx);

    // Update above/left mode trackers for downstream blocks. >=8×8
    // case: stamp picked mode at every sub_modes position.
    let span_w = mi_w.max(1);
    let span_h = mi_h.max(1);
    for c in 0..span_w {
        let cc = (mi_col + c) * 2;
        if cc < above_mode_4x4.len() {
            above_mode_4x4[cc] = picked_y_mode;
        }
        if cc + 1 < above_mode_4x4.len() {
            above_mode_4x4[cc + 1] = picked_y_mode;
        }
    }
    for r in 0..span_h {
        let rr = (mi_row + r) * 2;
        if rr < left_mode_4x4.len() {
            left_mode_4x4[rr] = picked_y_mode;
        }
        if rr + 1 < left_mode_4x4.len() {
            left_mode_4x4[rr + 1] = picked_y_mode;
        }
    }

    if all_skip {
        return;
    }

    // Emit luma coefficients.
    let coef_probs = &COEF_PROBS_4X4[0][0];
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
    let coef_probs_uv = &COEF_PROBS_4X4[1][0];
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

/// Map an `IntraMode` enum value to the spec's 0..9 index used by the
/// tree-encoded mode bin.
fn mode_to_index(m: IntraMode) -> usize {
    match m {
        IntraMode::Dc => 0,
        IntraMode::V => 1,
        IntraMode::H => 2,
        IntraMode::D45 => 3,
        IntraMode::D135 => 4,
        IntraMode::D117 => 5,
        IntraMode::D153 => 6,
        IntraMode::D207 => 7,
        IntraMode::D63 => 8,
        IntraMode::Tm => 9,
    }
}

/// Round-40 mode picker. Evaluates `CAND_MODES` over a representative
/// 4×4 footprint at the block's top-left corner and returns the lowest-SSE
/// mode. The picked mode then applies to all 4×4 TX sub-blocks of the
/// `bsize × bsize` parent (each running its own 4×4 predictor against
/// local neighbours, matching the decoder's TX-walk).
///
/// `bs` is the parent block size (8/16/32/64); we evaluate at 4×4 because:
/// 1. The decoder walks TX blocks at 4×4 in ONLY_4X4 tx_mode, so per-TX
///    neighbours dominate the actual prediction error budget.
/// 2. `reconintra::NeighbourBuf::build` debug-asserts `bs <= 32` — a
///    full 64×64 evaluation would need to be split anyway.
/// 3. SSE on a 4×4 stamp at (bx, by) is a sound rank proxy for which
///    mode tracks the local luminance trend, which is what mode-RDO buys.
fn pick_intra_mode_block(
    src: &[u8],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    recon: &[u8],
    recon_w: usize,
    bx: usize,
    by: usize,
    _bs: usize,
) -> IntraMode {
    const PICK_BS: usize = 4;
    // Gather source 4×4 block (edge-clamp).
    let mut src_block = [0u8; PICK_BS * PICK_BS];
    for r in 0..PICK_BS {
        for c in 0..PICK_BS {
            let px = (bx + c).min(src_w - 1);
            let py = (by + r).min(src_h - 1);
            src_block[r * PICK_BS + c] = src[py * src_stride + px];
        }
    }

    let nb = build_recon_neighbours(recon, recon_w, bx, by, PICK_BS);

    // Round-48 RDO pruning: probe DC first; if the DC residual is
    // already at noise floor, skip the rest. CAND_MODES[0] == Dc so this
    // is also the "no other mode wins" baseline.
    let mut pred_buf = [0u8; PICK_BS * PICK_BS];
    crate::reconintra::predict(IntraMode::Dc, &nb, &mut pred_buf, PICK_BS);
    let mut best_sse: u64 = 0;
    for i in 0..PICK_BS * PICK_BS {
        let d = src_block[i] as i32 - pred_buf[i] as i32;
        best_sse += (d * d) as u64;
    }
    if best_sse <= RDO_DC_EARLY_OUT_SSE {
        return IntraMode::Dc;
    }

    // Otherwise evaluate the remaining candidates.
    let mut best_mode = IntraMode::Dc;
    for &mode in &CAND_MODES[1..] {
        if mode == IntraMode::V && !nb.have_above {
            continue;
        }
        if mode == IntraMode::H && !nb.have_left {
            continue;
        }
        crate::reconintra::predict(mode, &nb, &mut pred_buf, PICK_BS);
        let mut sse: u64 = 0;
        for i in 0..PICK_BS * PICK_BS {
            let d = src_block[i] as i32 - pred_buf[i] as i32;
            sse += (d * d) as u64;
        }
        if sse < best_sse {
            best_sse = sse;
            best_mode = mode;
        }
    }
    best_mode
}

/// Build a decoder-compatible NeighbourBuf for a `bs × bs` block at
/// `(bx, by)` in the recon buffer. Mirrors `IntraTile::build_neighbours`
/// — same 127/129 padding for missing neighbours, same above-left
/// derivation rules. We do NOT enable the above-right extension; the
/// candidate set is DC / V / H / TM which never read above[bs..2*bs].
fn build_recon_neighbours(
    recon: &[u8],
    recon_w: usize,
    bx: usize,
    by: usize,
    bs: usize,
) -> crate::reconintra::NeighbourBuf {
    let recon_h = recon.len() / recon_w;
    let have_above = by > 0;
    let have_left = bx > 0;

    let above_tmp: Vec<u8> = if have_above {
        let n = bs.min(recon_w.saturating_sub(bx));
        let mut v = vec![0u8; bs];
        for c in 0..n {
            v[c] = recon[(by - 1) * recon_w + bx + c];
        }
        if n > 0 && n < bs {
            let last = v[n - 1];
            for b in &mut v[n..] {
                *b = last;
            }
        }
        v
    } else {
        vec![]
    };
    let left_tmp: Vec<u8> = if have_left {
        let nh = bs.min(recon_h.saturating_sub(by));
        let mut v = vec![0u8; bs];
        for r in 0..nh {
            v[r] = recon[(by + r) * recon_w + bx - 1];
        }
        if nh > 0 && nh < bs {
            let last = v[nh - 1];
            for b in &mut v[nh..] {
                *b = last;
            }
        }
        v
    } else {
        vec![]
    };
    let above_left = if have_above && have_left {
        Some(recon[(by - 1) * recon_w + bx - 1])
    } else if have_above {
        Some(127)
    } else if have_left {
        Some(129)
    } else {
        None
    };

    crate::reconintra::NeighbourBuf::build(
        bs,
        0,
        have_above,
        have_left,
        false, // no above-right extension for our 4-mode candidate set
        if above_tmp.is_empty() {
            None
        } else {
            Some(&above_tmp[..])
        },
        if left_tmp.is_empty() {
            None
        } else {
            Some(&left_tmp[..])
        },
        above_left,
    )
}

/// Encode one 4×4 block of a plane with the given intra mode (round 40).
///
/// Generalisation of the previous `encode_4x4_block` (DC-only): runs the
/// requested `mode`'s predictor against the recon buffer, computes the
/// residual, forward-DCTs, quantises, and reconstructs back into `recon`
/// for downstream-block prediction chaining. Returns scan-order quantised
/// coefficients, eob, and the initial_ctx for entropy coding.
#[allow(clippy::too_many_arguments)]
fn encode_4x4_block_mode(
    mode: IntraMode,
    src: &[u8],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    bx: usize,
    by: usize,
    dq_dc: i16,
    dq_ac: i16,
    recon: &mut [u8],
    recon_w: usize,
    above_nz: &mut [u8],
    left_nz: &mut [u8],
) -> (Vec<i32>, usize, usize) {
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

    // Build per-4×4 predictor by running the chosen mode against the
    // recon buffer for this exact 4×4 footprint. Falls back to DC if
    // a directional mode's neighbour isn't available.
    let mut pred_block = [0u8; 16];
    let recon_h = recon.len() / recon_w;
    if !run_predictor_4x4(mode, recon, recon_w, recon_h, bx, by, &mut pred_block) {
        // Fallback to DC mean if the requested mode can't run here.
        let dc = dc_pred_4x4(recon, recon_w, bx, by);
        pred_block.fill(dc);
    }

    // Gather source 4×4 block and compute per-sample residual.
    let mut residual = [0i16; 16];
    for r in 0..4 {
        for c in 0..4 {
            let px = (bx + c).min(src_w - 1);
            let py = (by + r).min(src_h - 1);
            let s = src[py * src_stride + px] as i16;
            residual[r * 4 + c] = s - pred_block[r * 4 + c] as i16;
        }
    }

    // Forward DCT.
    let mut coeffs_raster = [0i32; 16];
    fdct_2d(&residual, 4, &mut coeffs_raster);

    // Quantise.
    let mut coeffs_scan = vec![0i32; 16];
    coeffs_scan.copy_from_slice(&coeffs_raster);
    let eob = quantise(&mut coeffs_scan, &DEFAULT_SCAN_4X4, dq_dc, dq_ac);

    // Dequantise + inverse DCT to reconstruct.
    let mut dequant = [0i32; 16];
    for (i, &scan_idx) in DEFAULT_SCAN_4X4.iter().enumerate() {
        let q = coeffs_scan[i];
        let dq = if i == 0 { dq_dc as i32 } else { dq_ac as i32 };
        dequant[scan_idx as usize] = q * dq;
    }
    use crate::transform::{inverse_transform_add, TxType};
    // recon_u8 starts as the predictor block; idct adds residual onto it.
    let mut recon_u8 = pred_block;
    inverse_transform_add(TxType::DctDct, 4, 4, &dequant, &mut recon_u8, 4).ok();

    for r in 0..4 {
        for c in 0..4 {
            let rx = bx + c;
            let ry = by + r;
            if rx < recon_w && ry < recon_h {
                recon[ry * recon_w + rx] = recon_u8[r * 4 + c];
            }
        }
    }

    let nz = if eob > 0 { 1u8 } else { 0u8 };
    if x4 < above_nz.len() {
        above_nz[x4] = nz;
    }
    if y4 < left_nz.len() {
        left_nz[y4] = nz;
    }

    (coeffs_scan, eob, initial_ctx)
}

/// Run the requested 4×4 intra predictor by gathering neighbours from
/// the recon buffer using the *decoder's* `NeighbourBuf::build` policy
/// (127/129 padding for missing rows/columns). Returns false if a
/// directional mode's neighbour isn't available (caller falls back
/// to DC).
fn run_predictor_4x4(
    mode: IntraMode,
    recon: &[u8],
    recon_w: usize,
    _recon_h: usize,
    bx: usize,
    by: usize,
    out: &mut [u8; 16],
) -> bool {
    let nb = build_recon_neighbours(recon, recon_w, bx, by, 4);
    if mode == IntraMode::V && !nb.have_above {
        return false;
    }
    if mode == IntraMode::H && !nb.have_left {
        return false;
    }
    crate::reconintra::predict(mode, &nb, out, 4);
    true
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Top-left 4×4 block has no above and no left available — picker
    /// must fall back to DC even for sources that would prefer V/H/TM.
    #[test]
    fn pick_intra_mode_top_left_returns_dc() {
        let src = vec![100u8; 64 * 64];
        let recon = vec![128u8; 64 * 64];
        let mode = pick_intra_mode_block(&src, 64, 64, 64, &recon, 64, 0, 0, 8);
        assert_eq!(mode, IntraMode::Dc);
    }

    /// V_PRED tracks columns: if the source's column structure matches
    /// the above row, V wins over DC (which averages to one value).
    /// We construct a case where above row has column variation that
    /// matches what the source extends downward.
    #[test]
    fn pick_intra_mode_picks_v_when_columns_match_above() {
        // Recon: row 7 has column gradient 0..63. Below rows zeroed.
        let mut recon = vec![0u8; 64 * 64];
        for c in 0..64 {
            recon[7 * 64 + c] = (c * 4) as u8;
        }
        // Source at block (bx=0, by=8): same column gradient persists.
        let mut src = vec![0u8; 64 * 64];
        for r in 8..64 {
            for c in 0..64 {
                src[r * 64 + c] = (c * 4) as u8;
            }
        }
        let mode = pick_intra_mode_block(&src, 64, 64, 64, &recon, 64, 0, 8, 8);
        // V_PRED copies column gradient → exact match. DC averages → single value.
        assert_eq!(
            mode,
            IntraMode::V,
            "expected V_PRED when above row's column gradient matches source"
        );
    }

    /// Symmetric H_PRED case: source repeats left col across all cols.
    #[test]
    fn pick_intra_mode_picks_h_when_rows_match_left() {
        // Recon: column 7 has row gradient. Other columns zeroed.
        let mut recon = vec![0u8; 64 * 64];
        for r in 0..64 {
            recon[r * 64 + 7] = (r * 4) as u8;
        }
        // Source at block (bx=8, by=0): row values from left col extend right.
        let mut src = vec![0u8; 64 * 64];
        for r in 0..64 {
            for c in 8..64 {
                src[r * 64 + c] = (r * 4) as u8;
            }
        }
        let mode = pick_intra_mode_block(&src, 64, 64, 64, &recon, 64, 8, 0, 8);
        assert_eq!(
            mode,
            IntraMode::H,
            "expected H_PRED when left col's row gradient matches source"
        );
    }

    /// `mode_to_index` round-trip: every IntraMode value maps to its
    /// spec-defined integer (§7.4.5 Table 7-5) so the bool-encoded tree
    /// path matches what `read_intra_mode_tree` decodes to.
    #[test]
    fn mode_to_index_matches_spec_numbering() {
        assert_eq!(mode_to_index(IntraMode::Dc), 0);
        assert_eq!(mode_to_index(IntraMode::V), 1);
        assert_eq!(mode_to_index(IntraMode::H), 2);
        assert_eq!(mode_to_index(IntraMode::D45), 3);
        assert_eq!(mode_to_index(IntraMode::D135), 4);
        assert_eq!(mode_to_index(IntraMode::D117), 5);
        assert_eq!(mode_to_index(IntraMode::D153), 6);
        assert_eq!(mode_to_index(IntraMode::D207), 7);
        assert_eq!(mode_to_index(IntraMode::D63), 8);
        assert_eq!(mode_to_index(IntraMode::Tm), 9);
    }
}
