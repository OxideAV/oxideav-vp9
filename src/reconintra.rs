//! VP9 intra prediction, full port of libvpx `vp9_reconintra.c` + the
//! relevant `vpx_dsp/intrapred.c` predictors. Covers all 10 intra modes
//! at all four transform sizes (4×4, 8×8, 16×16, 32×32) for 8-bit luma
//! and chroma samples.
//!
//! Reference: VP9 spec §8.5.1, libvpx
//! `libvpx/vpx_dsp/intrapred.c` (commit `1.15.2`).
//!
//! The predictors consume a 2·N + 1 sample "above" row (with index `-1`
//! providing the above-left corner, indices `0..N-1` the direct above row,
//! and `N..2N-1` the above-right extension used by D45 / D63) plus a
//! length-N "left" column. The caller is responsible for materialising
//! those neighbour buffers per libvpx's `build_intra_predictors` rules:
//!
//! * NEED_LEFT  — dc_pred, h_pred, tm_pred, d135, d117, d153, d207.
//! * NEED_ABOVE — dc_pred, v_pred, tm_pred, d135, d117, d153.
//! * NEED_ABOVERIGHT — d45, d63.
//!
//! Missing neighbours are replaced per libvpx:
//! * above-row absent: fill with 127.
//! * left-col absent:  fill with 129.
//! * above-left corner: 127 when above-row also absent, 129 when left
//!   absent but above present, else real sample.
//!
//! This module intentionally mirrors libvpx's flat-array layout so a
//! reviewer can diff the two side-by-side.

use crate::intra::IntraMode;

/// Buffer layout: `buf[0]` = above-left corner, `buf[1..=2*bs]` = above row
/// with above-right extension. Total length `2*bs + 1` = up to 129.
pub const ABOVE_BUF_CAP: usize = 2 * 32 + 1;
/// Left column has exactly `bs` entries (up to 32).
pub const LEFT_BUF_CAP: usize = 32;

/// A block-size-agnostic neighbourhood. `above` holds the above-left
/// corner at index 0, the above row at indices `1..=bs`, and the
/// above-right extension at `bs+1..=2*bs`. `left` holds the left column
/// (length `bs`). `have_above` / `have_left` track whether those
/// neighbours are real pixels or synthesised padding (127 / 129).
#[derive(Clone)]
pub struct NeighbourBuf {
    pub above: [u8; ABOVE_BUF_CAP],
    pub left: [u8; LEFT_BUF_CAP],
    pub bs: usize,
    pub have_above: bool,
    pub have_left: bool,
}

impl NeighbourBuf {
    /// Build a neighbour buffer from above / left availability plus the
    /// decoded reconstruction buffer. `ref_slice` points at the top-left
    /// pixel of the block; the row above it lives at `ref_slice - stride`
    /// and the left column at `ref_slice[r * stride - 1]`.
    ///
    /// Matches libvpx's `build_intra_predictors` — 127 / 129 fill for
    /// unavailable neighbours, above-right extension for D45 / D63.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        bs: usize,
        tx_size_log2: usize,
        have_above: bool,
        have_left: bool,
        have_aboveright: bool,
        above_row: Option<&[u8]>,
        left_col: Option<&[u8]>,
        above_left: Option<u8>,
    ) -> Self {
        debug_assert!(bs <= 32);
        let _ = tx_size_log2;
        let mut buf = Self {
            above: [127u8; ABOVE_BUF_CAP],
            left: [129u8; LEFT_BUF_CAP],
            bs,
            have_above,
            have_left,
        };
        // Left column.
        if have_left {
            if let Some(src) = left_col {
                let n = src.len().min(bs);
                buf.left[..n].copy_from_slice(&src[..n]);
                // Replicate last sample for bottom extension.
                if n < bs {
                    let last = if n > 0 { src[n - 1] } else { 129 };
                    for b in &mut buf.left[n..bs] {
                        *b = last;
                    }
                }
            }
        }
        // Above row + corner.
        buf.above[0] = if have_above && have_left {
            above_left.unwrap_or(127)
        } else if have_above {
            129
        } else {
            // No above row: fall back to 127 whether or not left is present.
            127
        };
        if have_above {
            if let Some(src) = above_row {
                let n = src.len().min(bs);
                buf.above[1..=n].copy_from_slice(&src[..n]);
                if n < bs {
                    let last = if n > 0 { src[n - 1] } else { 127 };
                    for b in &mut buf.above[n + 1..=bs] {
                        *b = last;
                    }
                }
            }
            // Above-right extension (for D45 / D63).
            if have_aboveright {
                if let Some(src) = above_row {
                    // above_row might provide extended samples past bs if the
                    // caller knew the right neighbour existed.
                    if src.len() >= 2 * bs {
                        buf.above[bs + 1..=2 * bs].copy_from_slice(&src[bs..2 * bs]);
                        return buf;
                    }
                }
            }
            // Above-right unavailable or not provided: replicate last sample.
            let last = buf.above[bs];
            for b in &mut buf.above[bs + 1..=2 * bs] {
                *b = last;
            }
        } else {
            // No above samples: fill with 127.
            for b in &mut buf.above[1..=2 * bs] {
                *b = 127;
            }
        }
        buf
    }

    /// Slice view of the above samples used by the predictor closures
    /// (index 0 is the above-left corner; index 1 is above[0] in libvpx's
    /// C code, which uses negative offsets `above[-1]`).
    fn above_with_corner(&self) -> &[u8] {
        &self.above
    }

    fn left_slice(&self) -> &[u8] {
        &self.left[..self.bs]
    }
}

/// Predict a `bs × bs` block using `mode`. Writes `bs` rows of `bs` bytes
/// into `dst` with stride `dst_stride`. `bs` must be a power of two in
/// `{4, 8, 16, 32}`.
pub fn predict(mode: IntraMode, nb: &NeighbourBuf, dst: &mut [u8], dst_stride: usize) {
    let bs = nb.bs;
    debug_assert!(matches!(bs, 4 | 8 | 16 | 32));
    let above_buf = nb.above_with_corner();
    // In libvpx the above pointer points at above[0], so above[-1] is
    // our buffer index 0 (the corner). Build an alias for clarity.
    let above = &above_buf[1..]; // length = 2*bs
    let above_left = above_buf[0];
    let left = nb.left_slice();
    match mode {
        IntraMode::Dc => dc_pred(
            nb.have_above,
            nb.have_left,
            bs,
            above,
            left,
            dst,
            dst_stride,
        ),
        IntraMode::V => v_pred(bs, above, dst, dst_stride),
        IntraMode::H => h_pred(bs, left, dst, dst_stride),
        IntraMode::D45 => d45_pred(bs, above, dst, dst_stride),
        IntraMode::D135 => d135_pred(bs, above, left, above_left, dst, dst_stride),
        IntraMode::D117 => d117_pred(bs, above, left, above_left, dst, dst_stride),
        IntraMode::D153 => d153_pred(bs, above, left, above_left, dst, dst_stride),
        IntraMode::D207 => d207_pred(bs, left, dst, dst_stride),
        IntraMode::D63 => d63_pred(bs, above, dst, dst_stride),
        IntraMode::Tm => tm_pred(bs, above, left, above_left, dst, dst_stride),
    }
}

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u32 + b as u32 + 1) >> 1) as u8
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u32 + 2 * b as u32 + c as u32 + 2) >> 2) as u8
}

fn fill_row(dst: &mut [u8], stride: usize, r: usize, v: u8, bs: usize) {
    let base = r * stride;
    for b in &mut dst[base..base + bs] {
        *b = v;
    }
}

fn dc_pred(
    up: bool,
    lf: bool,
    bs: usize,
    above: &[u8],
    left: &[u8],
    dst: &mut [u8],
    stride: usize,
) {
    let dc = match (up, lf) {
        (false, false) => 128u8,
        (true, false) => {
            let s: u32 = above[..bs].iter().map(|&v| v as u32).sum();
            ((s + (bs as u32) / 2) / (bs as u32)) as u8
        }
        (false, true) => {
            let s: u32 = left[..bs].iter().map(|&v| v as u32).sum();
            ((s + (bs as u32) / 2) / (bs as u32)) as u8
        }
        (true, true) => {
            let sa: u32 = above[..bs].iter().map(|&v| v as u32).sum();
            let sl: u32 = left[..bs].iter().map(|&v| v as u32).sum();
            let total = sa + sl;
            let denom = 2 * (bs as u32);
            ((total + denom / 2) / denom) as u8
        }
    };
    for r in 0..bs {
        fill_row(dst, stride, r, dc, bs);
    }
}

fn v_pred(bs: usize, above: &[u8], dst: &mut [u8], stride: usize) {
    for r in 0..bs {
        let base = r * stride;
        dst[base..base + bs].copy_from_slice(&above[..bs]);
    }
}

fn h_pred(bs: usize, left: &[u8], dst: &mut [u8], stride: usize) {
    for (r, &v) in left.iter().take(bs).enumerate() {
        fill_row(dst, stride, r, v, bs);
    }
}

fn tm_pred(bs: usize, above: &[u8], left: &[u8], above_left: u8, dst: &mut [u8], stride: usize) {
    let al = above_left as i32;
    for (r, &lv) in left.iter().take(bs).enumerate() {
        let base = r * stride;
        let lr = lv as i32;
        for c in 0..bs {
            let p = lr + above[c] as i32 - al;
            dst[base + c] = p.clamp(0, 255) as u8;
        }
    }
}

fn d45_pred(bs: usize, above: &[u8], dst: &mut [u8], stride: usize) {
    // §8.5.1 D45_PRED:
    //   pred[i][j] = (i+j+2 < 2*size)
    //                  ? Round2(above[i+j] + 2*above[i+j+1] + above[i+j+2], 2)
    //                  : above[2*size - 1]
    //
    // Round-14 fix: the boundary value is `above[2*size-1]` per spec,
    // not `above[size-1]`. This only matters when the §8.5.1 above-row
    // extension is enabled (txSz==0 && notOnRight): then the extension
    // holds real pixels and the boundary differs. The first-row
    // last-pixel was previously short-circuited to the wrong constant;
    // now it falls through the unified loop.
    let above_right = above[2 * bs - 1];
    for r in 0..bs {
        let base = r * stride;
        for c in 0..bs {
            let idx = r + c;
            if idx + 2 < 2 * bs {
                dst[base + c] = avg3(above[idx], above[idx + 1], above[idx + 2]);
            } else {
                dst[base + c] = above_right;
            }
        }
    }
}

fn d63_pred(bs: usize, above: &[u8], dst: &mut [u8], stride: usize) {
    // libvpx:
    //   dst[0][c]       = AVG2(above[c], above[c+1])
    //   dst[1][c]       = AVG3(above[c], above[c+1], above[c+2])
    //   even rows 2k    = memcpy(dst[2k-2] + k, bs - k); pad right with above[bs-1]
    //   odd  rows 2k+1  = memcpy(dst[2k-1] + k, bs - k); pad right with above[bs-1]
    // Equivalent, cleaner: dst[r][c] uses above starting at c + r/2.
    let last = above[bs - 1];
    for r in 0..bs {
        let base = r * stride;
        let shift = r >> 1;
        let is_avg2 = (r & 1) == 0;
        for c in 0..bs {
            // Indices used for the AVG kernel.
            let i = c + shift;
            if is_avg2 {
                // AVG2(above[i], above[i+1])
                if i + 1 < 2 * bs {
                    dst[base + c] = avg2(above[i], above[i + 1]);
                } else {
                    dst[base + c] = last;
                }
            } else if i + 2 < 2 * bs {
                dst[base + c] = avg3(above[i], above[i + 1], above[i + 2]);
            } else {
                dst[base + c] = last;
            }
        }
    }
}

fn d207_pred(bs: usize, left: &[u8], dst: &mut [u8], stride: usize) {
    // libvpx: vertical-mirror-like pattern from the left column.
    //   first column:
    //     dst[r][0] = AVG2(left[r], left[r+1]) for r < bs-1;
    //     dst[bs-1][0] = left[bs-1]
    //   second column:
    //     dst[r][1] = AVG3(left[r], left[r+1], left[r+2]) for r < bs-2
    //     dst[bs-2][1] = AVG3(left[bs-2], left[bs-1], left[bs-1])
    //     dst[bs-1][1] = left[bs-1]
    //   remainder:
    //     dst[bs-1][c] = left[bs-1] for all c >= 0
    //     dst[r][c] = dst[r+1][c-2] for r in bs-2..=0, c in 0..bs-2

    // First column
    for r in 0..(bs - 1) {
        dst[r * stride] = avg2(left[r], left[r + 1]);
    }
    dst[(bs - 1) * stride] = left[bs - 1];
    // Second column
    for r in 0..(bs - 2) {
        dst[r * stride + 1] = avg3(left[r], left[r + 1], left[r + 2]);
    }
    dst[(bs - 2) * stride + 1] = avg3(left[bs - 2], left[bs - 1], left[bs - 1]);
    dst[(bs - 1) * stride + 1] = left[bs - 1];
    // Last row (cols 2..bs-1) filled with left[bs-1]
    for c in 2..bs {
        dst[(bs - 1) * stride + c] = left[bs - 1];
    }
    // Rest of block: rows bs-2 down to 0, columns 2..bs-1.
    // Increment loop from bottom up so we can read row+1.
    for r in (0..=bs - 2).rev() {
        for c in 2..bs {
            dst[r * stride + c] = dst[(r + 1) * stride + c - 2];
        }
    }
}

fn d117_pred(bs: usize, above: &[u8], left: &[u8], al: u8, dst: &mut [u8], stride: usize) {
    // Row 0: AVG2(above[c-1], above[c]), c=0 uses above[-1]=al.
    {
        let base = 0;
        dst[base] = avg2(al, above[0]);
        for c in 1..bs {
            dst[base + c] = avg2(above[c - 1], above[c]);
        }
    }
    // Row 1:
    {
        let base = stride;
        dst[base] = avg3(left[0], al, above[0]);
        // c=1 uses above[-1], above[0], above[1]
        dst[base + 1] = avg3(al, above[0], above[1]);
        for c in 2..bs {
            dst[base + c] = avg3(above[c - 2], above[c - 1], above[c]);
        }
    }
    // Column 0 for rows 2..bs: AVG3(...) with left
    {
        dst[2 * stride] = avg3(al, left[0], left[1]);
        for r in 3..bs {
            dst[r * stride] = avg3(left[r - 3], left[r - 2], left[r - 1]);
        }
    }
    // Rest of block: dst[r][c] = dst[r-2][c-1]
    for r in 2..bs {
        for c in 1..bs {
            dst[r * stride + c] = dst[(r - 2) * stride + c - 1];
        }
    }
}

fn d135_pred(bs: usize, above: &[u8], left: &[u8], al: u8, dst: &mut [u8], stride: usize) {
    // libvpx builds a border array of length 2*bs-1 containing the outer
    // border from bottom-left to top-right, with 3-tap averaging.
    let mut border = [0u8; 64]; // covers up to bs=32 -> 63 entries
                                // Border from bottom-left upward.
    for i in 0..(bs - 2) {
        border[i] = avg3(left[bs - 3 - i], left[bs - 2 - i], left[bs - 1 - i]);
    }
    // border[bs-2] = AVG3(al, left[0], left[1])
    border[bs - 2] = avg3(al, left[0], left[1]);
    // border[bs-1] = AVG3(left[0], al, above[0])
    border[bs - 1] = avg3(left[0], al, above[0]);
    // border[bs]   = AVG3(al, above[0], above[1])
    border[bs] = avg3(al, above[0], above[1]);
    // remaining top border ascending.
    for i in 0..(bs - 2) {
        border[bs + 1 + i] = avg3(above[i], above[i + 1], above[i + 2]);
    }
    // Fill dst from border.
    for r in 0..bs {
        let base = r * stride;
        for c in 0..bs {
            dst[base + c] = border[bs - 1 - r + c];
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn d153_pred(bs: usize, above: &[u8], left: &[u8], al: u8, dst: &mut [u8], stride: usize) {
    // Column 0
    dst[0] = avg2(al, left[0]);
    for r in 1..bs {
        dst[r * stride] = avg2(left[r - 1], left[r]);
    }
    // Column 1
    dst[1] = avg3(left[0], al, above[0]);
    dst[stride + 1] = avg3(al, left[0], left[1]);
    for r in 2..bs {
        dst[r * stride + 1] = avg3(left[r - 2], left[r - 1], left[r]);
    }
    // Row 0 columns 2..bs — 3-tap horizontal average.
    for c in 2..bs {
        let j = c - 2;
        let a_m1 = if j == 0 { al } else { above[j - 1] };
        let a_0 = above[j];
        let a_p1 = above[j + 1];
        dst[c] = avg3(a_m1, a_0, a_p1);
    }
    // Rest of block: for r in 1..bs, c in 2..bs: dst[r][c] = dst[r-1][c-2]
    for r in 1..bs {
        for c in 2..bs {
            dst[r * stride + c] = dst[(r - 1) * stride + c - 2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_nb(bs: usize) -> NeighbourBuf {
        let above_row = vec![100u8; 2 * bs];
        let left_col = vec![120u8; bs];
        NeighbourBuf::build(
            bs,
            0,
            true,
            true,
            true,
            Some(&above_row),
            Some(&left_col),
            Some(110),
        )
    }

    #[test]
    fn dc_pred_4x4_constant_mean() {
        let nb = mk_nb(4);
        let mut dst = [0u8; 16];
        predict(IntraMode::Dc, &nb, &mut dst, 4);
        for &v in &dst {
            assert_eq!(v, 110);
        }
    }

    #[test]
    fn dc_pred_no_neighbours_gives_128() {
        let nb = NeighbourBuf::build(4, 0, false, false, false, None, None, None);
        let mut dst = [0u8; 16];
        predict(IntraMode::Dc, &nb, &mut dst, 4);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn v_pred_copies_row() {
        let nb = mk_nb(8);
        let mut dst = [0u8; 64];
        predict(IntraMode::V, &nb, &mut dst, 8);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 8 + c], 100);
            }
        }
    }

    #[test]
    fn h_pred_copies_col() {
        let nb = mk_nb(8);
        let mut dst = [0u8; 64];
        predict(IntraMode::H, &nb, &mut dst, 8);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 8 + c], 120);
            }
        }
    }

    #[test]
    fn tm_pred_4x4_matches_formula() {
        let nb = mk_nb(4);
        let mut dst = [0u8; 16];
        predict(IntraMode::Tm, &nb, &mut dst, 4);
        // (120 + 100 - 110) = 110 for all samples given uniform neighbours.
        for &v in &dst {
            assert_eq!(v, 110);
        }
    }

    #[test]
    fn d45_pred_terminates_with_above_right() {
        let nb = mk_nb(4);
        let mut dst = [0u8; 16];
        predict(IntraMode::D45, &nb, &mut dst, 4);
        // Uniform neighbours -> uniform prediction.
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn all_modes_produce_output_at_32x32() {
        let nb = mk_nb(32);
        for m in [
            IntraMode::Dc,
            IntraMode::V,
            IntraMode::H,
            IntraMode::D45,
            IntraMode::D135,
            IntraMode::D117,
            IntraMode::D153,
            IntraMode::D207,
            IntraMode::D63,
            IntraMode::Tm,
        ] {
            let mut dst = vec![0u8; 32 * 32];
            predict(m, &nb, &mut dst, 32);
            // At uniform input all outputs should be 100 or 120 (nearly).
            let mn = *dst.iter().min().unwrap();
            let mx = *dst.iter().max().unwrap();
            assert!(mn >= 80 && mx <= 140, "mode {m:?}: min={mn} max={mx}");
        }
    }
}
