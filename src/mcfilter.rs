//! VP9 sub-pel interpolation filter (§8.5.1 "Sub-sample interpolation
//! process" / libvpx `vp9_filter.c`).
//!
//! Inter prediction reads a block from a reference frame at a fractional
//! offset. The reference block is sampled through a separable 8-tap FIR
//! filter (or 4-tap for bilinear mode). Taps are fixed-point
//! `SUBPEL_BITS = 7` and per-phase rows live in this module's tables.
//!
//! Four filter flavours (§7.3.7 / libvpx `FILTERS`):
//!
//! * `EIGHTTAP` (0) — default 8-tap, regular smoothness.
//! * `EIGHTTAP_SMOOTH` (1) — wider, smoother 8-tap.
//! * `EIGHTTAP_SHARP` (2) — sharper 8-tap.
//! * `BILINEAR` (3) — 4-tap linear (effectively 8-tap with zero
//!   outside the middle 4 positions).
//!
//! `SUBPEL_SHIFTS = 16`: each pixel is sampled at one of 16 sub-pel
//! phases, numbered 0..15. Phase 0 is integer alignment and re-uses
//! the reference pixel directly (VP9 spec explicitly short-circuits
//! this).

/// Output of the 8-tap convolution shift — VP9 normalises by
/// `1 << SUBPEL_BITS` (i.e. 128).
pub const FILTER_BITS: i32 = 7;
/// Number of sub-pel phases per integer unit.
pub const SUBPEL_SHIFTS: usize = 16;
/// Filter length.
pub const SUBPEL_TAPS: usize = 8;

/// Filter kind — §7.3.7. The `SWITCHABLE` encoding (4) is resolved per
/// block to one of the concrete 3 before calling the interpolator; we
/// never store SWITCHABLE here.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpFilter {
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
}

impl InterpFilter {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::EightTap,
            1 => Self::EightTapSmooth,
            2 => Self::EightTapSharp,
            _ => Self::Bilinear,
        }
    }
}

/// libvpx `sub_pel_filters_8` — the default EIGHTTAP filter.
pub const FILTER_EIGHTTAP: [[i32; SUBPEL_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-1, 5, -18, 68, 88, -19, 6, -1],
    [-1, 5, -16, 58, 97, -19, 5, -1],
    [-1, 4, -14, 48, 105, -18, 5, -1],
    [-1, 4, -11, 37, 112, -16, 4, -1],
    [-1, 3, -9, 27, 118, -13, 4, -1],
    [0, 2, -6, 18, 122, -10, 3, -1],
    [0, 1, -3, 8, 126, -5, 1, 0],
];

/// libvpx `sub_pel_filters_8lp` — EIGHTTAP_SMOOTH (low-pass).
pub const FILTER_EIGHTTAP_SMOOTH: [[i32; SUBPEL_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 26, 63, 43, 4, -4, 0],
    [-2, -3, 24, 62, 46, 5, -4, 0],
    [-2, -3, 21, 60, 49, 7, -4, 0],
    [-1, -4, 18, 59, 51, 9, -4, 0],
    [-1, -4, 16, 57, 53, 12, -4, -1],
    [-1, -4, 14, 55, 55, 14, -4, -1],
    [-1, -4, 12, 53, 57, 16, -4, -1],
    [0, -4, 9, 51, 59, 18, -4, -1],
    [0, -4, 7, 49, 60, 21, -3, -2],
    [0, -4, 5, 46, 62, 24, -3, -2],
    [0, -4, 4, 43, 63, 26, -2, -2],
    [0, -3, 2, 41, 63, 29, -2, -2],
    [0, -3, 1, 38, 64, 32, -1, -3],
];

/// libvpx `sub_pel_filters_8s` — EIGHTTAP_SHARP.
pub const FILTER_EIGHTTAP_SHARP: [[i32; SUBPEL_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-1, 3, -7, 127, 8, -3, 1, 0],
    [-2, 5, -13, 125, 17, -6, 3, -1],
    [-3, 7, -17, 121, 27, -10, 5, -2],
    [-4, 9, -20, 115, 37, -13, 6, -2],
    [-4, 10, -23, 108, 48, -16, 8, -3],
    [-4, 10, -24, 100, 59, -19, 9, -3],
    [-4, 11, -24, 90, 70, -21, 10, -4],
    [-4, 11, -23, 80, 80, -23, 11, -4],
    [-4, 10, -21, 70, 90, -24, 11, -4],
    [-3, 9, -19, 59, 100, -24, 10, -4],
    [-3, 8, -16, 48, 108, -23, 10, -4],
    [-2, 6, -13, 37, 115, -20, 9, -4],
    [-2, 5, -10, 27, 121, -17, 7, -3],
    [-1, 3, -6, 17, 125, -13, 5, -2],
    [0, 1, -3, 8, 127, -7, 3, -1],
];

/// libvpx `sub_pel_filters_4` — BILINEAR (4-tap, encoded as 8-tap).
pub const FILTER_BILINEAR: [[i32; SUBPEL_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

/// Look up the full sub-pel filter bank for a given kind.
pub fn filter_table(kind: InterpFilter) -> &'static [[i32; SUBPEL_TAPS]; SUBPEL_SHIFTS] {
    match kind {
        InterpFilter::EightTap => &FILTER_EIGHTTAP,
        InterpFilter::EightTapSmooth => &FILTER_EIGHTTAP_SMOOTH,
        InterpFilter::EightTapSharp => &FILTER_EIGHTTAP_SHARP,
        InterpFilter::Bilinear => &FILTER_BILINEAR,
    }
}

/// Sampler closure: given (row, col) in integer pixel coordinates, return
/// the sample with edge-clamped access.
pub trait RefSampler {
    fn sample(&self, row: isize, col: isize) -> u8;
}

/// Produce one `w × h` interpolated block starting at `(row, col)` in the
/// reference frame, offset by `(mv_row, mv_col)` in 1/8-pel units for luma
/// and 1/16-pel for chroma (the caller scales by subsampling).
///
/// The MV is expected in the same pel-unit as `subpel_bits_y` / `_x`: 3 bits
/// for 1/8-pel at luma, 4 bits for 1/16-pel at chroma.
#[allow(clippy::too_many_arguments)]
pub fn mc_block<S: RefSampler>(
    src: &S,
    filter: InterpFilter,
    dst: &mut [u8],
    dst_stride: usize,
    dst_w: usize,
    dst_h: usize,
    int_row: isize,
    int_col: isize,
    sub_row: u32, // 0..16
    sub_col: u32, // 0..16
) {
    debug_assert!(sub_row < SUBPEL_SHIFTS as u32);
    debug_assert!(sub_col < SUBPEL_SHIFTS as u32);

    // Fast path: integer alignment.
    if sub_row == 0 && sub_col == 0 {
        for r in 0..dst_h {
            for c in 0..dst_w {
                dst[r * dst_stride + c] = src.sample(int_row + r as isize, int_col + c as isize);
            }
        }
        return;
    }

    let tbl = filter_table(filter);
    let frow = &tbl[sub_row as usize];
    let fcol = &tbl[sub_col as usize];

    // Separable filter: first horizontal pass into a 16-bit buffer that
    // keeps the intermediate before the second shift, then vertical pass.
    // Buffer height = dst_h + 7 (extra rows for the vertical convolution).
    let inter_h = dst_h + SUBPEL_TAPS - 1;
    let mut inter = vec![0i32; inter_h * dst_w];
    for r in 0..inter_h {
        let src_row = int_row + r as isize - 3; // taps center at index 3
        for c in 0..dst_w {
            let base_col = int_col + c as isize;
            let mut acc = 0i32;
            for k in 0..SUBPEL_TAPS {
                let s = src.sample(src_row, base_col + k as isize - 3) as i32;
                acc += s * fcol[k];
            }
            // libvpx rounds/shifts by `FILTER_BITS` with `ROUND_POWER_OF_TWO`.
            let v = (acc + (1 << (FILTER_BITS - 1))) >> FILTER_BITS;
            inter[r * dst_w + c] = v.clamp(0, 255);
        }
    }

    // Vertical pass.
    for r in 0..dst_h {
        for c in 0..dst_w {
            let mut acc = 0i32;
            for k in 0..SUBPEL_TAPS {
                acc += inter[(r + k) * dst_w + c] * frow[k];
            }
            let v = (acc + (1 << (FILTER_BITS - 1))) >> FILTER_BITS;
            dst[r * dst_stride + c] = v.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flat(u8);
    impl RefSampler for Flat {
        fn sample(&self, _row: isize, _col: isize) -> u8 {
            self.0
        }
    }

    #[test]
    fn integer_sub_pel_copies_reference() {
        let src = Flat(99);
        let mut dst = vec![0u8; 16 * 16];
        mc_block(&src, InterpFilter::EightTap, &mut dst, 16, 16, 16, 0, 0, 0, 0);
        for &v in &dst {
            assert_eq!(v, 99);
        }
    }

    #[test]
    fn filter_preserves_constant_field() {
        // A flat field should stay flat through any sub-pel phase: the
        // filter taps sum to 128 for every phase.
        for phase in 0..SUBPEL_SHIFTS as u32 {
            let src = Flat(123);
            let mut dst = vec![0u8; 16 * 16];
            mc_block(
                &src,
                InterpFilter::EightTap,
                &mut dst,
                16,
                16,
                16,
                0,
                0,
                phase,
                0,
            );
            for &v in &dst {
                assert_eq!(v, 123, "phase {phase} broke flat field");
            }
        }
    }

    #[test]
    fn filter_table_all_sum_to_128() {
        for kind in [
            InterpFilter::EightTap,
            InterpFilter::EightTapSmooth,
            InterpFilter::EightTapSharp,
            InterpFilter::Bilinear,
        ] {
            let t = filter_table(kind);
            for (i, row) in t.iter().enumerate() {
                let s: i32 = row.iter().sum();
                assert_eq!(s, 128, "{kind:?} phase {i} tap-sum = {s}");
            }
        }
    }
}
