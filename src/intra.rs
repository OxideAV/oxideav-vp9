//! VP9 intra prediction process per spec v0.7 §8.5.1.
//!
//! Round 10 lands the §8.5.1 intra prediction process: given the
//! already-reconstructed neighbour samples (`aboveRow` / `leftCol`)
//! and a prediction `mode`, it fills a `size`-by-`size` transform
//! block (`size = 1 << (txSz + 2)`) of the current plane. This is the
//! prediction half of the §8.6.2 reconstruct loop (the residual half
//! is the round-7 token decode → round-8 dequant → round-9 inverse
//! transform chain); the still-deferred §8.6.2 driver adds the two
//! together and clips.
//!
//! The pieces this module exposes:
//!
//! * [`PredMode`] — the 10 §7.4.5 intra prediction modes
//!   (`DC_PRED` = 0 .. `TM_PRED` = 9).
//! * [`Plane`] — a minimal row-major plane buffer abstraction
//!   standing in for `CurrFrame[ plane ]`. The §8.5.1 process reads
//!   the already-reconstructed neighbour samples from it and writes
//!   the prediction back into it.
//! * [`predict_intra`] — the §8.5.1 process. It derives `aboveRow` /
//!   `leftCol` from the plane per the `haveAbove` / `haveLeft` /
//!   `notOnRight` availability flags and the `(maxX, maxY)` plane
//!   clamps, builds the `pred` array for the selected `mode`, and
//!   stores it back into the plane.
//!
//! The §8.6.2 reconstruct driver that calls this with the right
//! availability flags (derived from tile / frame edges) and then adds
//! the inverse-transformed residual lands in a later round; this
//! module is the pure prediction layer it will call.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.5.1, plus §3 `Round2` / `Clip1`
//! and §7.4.5 mode numbering). Every formula is transcribed directly
//! from the §8.5.1 listing.

// The §8.6.2 reconstruct process that drives this with real
// availability flags lands in a subsequent round; until then the
// module is exercised exclusively from `#[cfg(test)]`.
#![allow(dead_code)]

/// The 10 intra prediction modes from §7.4.5 (`default_intra_mode`).
///
/// The discriminants match the spec's numbering exactly so the
/// (deferred) mode-info decode can cast a decoded `L(n)` value
/// straight to a `PredMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
// The spec names every mode with a `_PRED` suffix (`DC_PRED`,
// `V_PRED`, …); the shared `Pred` postfix is intrinsic to that naming
// and is kept for one-to-one fidelity with §7.4.5.
#[allow(clippy::enum_variant_names)]
pub enum PredMode {
    /// DC prediction (average of available neighbours).
    DcPred = 0,
    /// Vertical prediction (copy `aboveRow` down every row).
    VPred = 1,
    /// Horizontal prediction (copy `leftCol` across every column).
    HPred = 2,
    /// 45-degree diagonal-down-left prediction.
    D45Pred = 3,
    /// 135-degree diagonal-down-right prediction.
    D135Pred = 4,
    /// 117-degree prediction.
    D117Pred = 5,
    /// 153-degree prediction.
    D153Pred = 6,
    /// 207-degree prediction.
    D207Pred = 7,
    /// 63-degree prediction.
    D63Pred = 8,
    /// "TrueMotion" prediction (`aboveRow + leftCol - aboveRow[-1]`).
    TmPred = 9,
}

impl PredMode {
    /// Maps a raw mode value (0..=9) to a [`PredMode`], per the §7.4.5
    /// `default_intra_mode` table. Returns `None` for out-of-range
    /// inputs.
    pub fn from_raw(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::DcPred,
            1 => Self::VPred,
            2 => Self::HPred,
            3 => Self::D45Pred,
            4 => Self::D135Pred,
            5 => Self::D117Pred,
            6 => Self::D153Pred,
            7 => Self::D207Pred,
            8 => Self::D63Pred,
            9 => Self::TmPred,
            _ => return None,
        })
    }
}

/// A minimal row-major plane buffer standing in for `CurrFrame[ plane ]`.
///
/// Samples are stored as `i32` so the high-bit-depth (10/12-bit) paths
/// fit without separate buffer types; values are conceptually in
/// `0..(1 << BitDepth)`. The §8.5.1 process reads neighbour samples
/// from rows / columns adjacent to the transform block and writes the
/// prediction back into the block region.
#[derive(Debug, Clone)]
pub struct Plane {
    width: usize,
    height: usize,
    samples: Vec<i32>,
}

impl Plane {
    /// Builds a `width`-by-`height` plane initialised to `0`.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            samples: vec![0; width * height],
        }
    }

    /// Plane width in samples.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Plane height in samples.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Reads the sample at row `y`, column `x`.
    ///
    /// Panics if `(x, y)` is outside the plane; the §8.5.1 process
    /// always clamps neighbour coordinates with `Min(maxX, .)` /
    /// `Min(maxY, .)` before reading, so a panic indicates a caller
    /// supplied an out-of-range `(maxX, maxY)` relative to the buffer.
    pub fn get(&self, x: usize, y: usize) -> i32 {
        self.samples[y * self.width + x]
    }

    /// Writes `value` at row `y`, column `x`.
    pub fn set(&mut self, x: usize, y: usize, value: i32) {
        self.samples[y * self.width + x] = value;
    }

    /// Row-major sample slice (stride = `width`). Used by the §8.8
    /// loop-filter driver, which operates on the same `CurrFrame`
    /// storage through its own plane view.
    pub(crate) fn samples_mut(&mut self) -> &mut [i32] {
        &mut self.samples
    }

    /// Row-major immutable sample slice (stride = `width`). Used by the
    /// §8.10 reference-buffer update, which copies the reconstructed +
    /// loop-filtered working plane into a `FrameStore[ ]` slot.
    pub(crate) fn samples(&self) -> &[i32] {
        &self.samples
    }
}

/// `Round2( x, n ) = ( x + (1 << (n - 1)) ) >> n` per §3.
///
/// All §8.5.1 averaging steps use `n >= 1`, so the `1 << (n - 1)`
/// rounding bias is always well defined.
#[inline]
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// `Clip1( x ) = Clip3( 0, (1 << BitDepth) - 1, x )` per §3.
#[inline]
fn clip1(x: i32, bit_depth: u32) -> i32 {
    let max = (1 << bit_depth) - 1;
    x.clamp(0, max)
}

/// The §8.5.1 intra prediction process.
///
/// Inputs mirror the spec one-for-one:
///
/// * `plane_buf` — `CurrFrame[ plane ]`, read for neighbour samples
///   and written with the prediction.
/// * `x` / `y` — top-left sample of the transform block in the plane.
/// * `have_left` / `have_above` — `haveLeft` / `haveAbove` (1 if valid
///   samples exist to the left / above).
/// * `not_on_right` — `notOnRight` (1 if the transform block is not on
///   the right edge of the prediction block).
/// * `tx_sz` — `txSz` (0 = TX_4X4 .. 3 = TX_32X32); `log2Size = txSz +
///   2`, `size = 1 << log2Size`.
/// * `mode` — the prediction mode.
/// * `max_x` / `max_y` — `maxX` / `maxY`, the inclusive plane-edge
///   clamps for this plane (already subsampling-adjusted by the
///   caller).
/// * `bit_depth` — `BitDepth` (8, 10 or 12).
///
/// On return `plane_buf[ y + i ][ x + j ]` holds `pred[ i ][ j ]` for
/// `i, j` in `0..size`.
#[allow(clippy::too_many_arguments)]
pub fn predict_intra(
    plane_buf: &mut Plane,
    x: usize,
    y: usize,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    tx_sz: u32,
    mode: PredMode,
    max_x: usize,
    max_y: usize,
    bit_depth: u32,
) {
    let log2_size = tx_sz + 2;
    let size = 1usize << log2_size;

    // §8.5.1 base value for "no neighbour" fills.
    let base = 1i32 << (bit_depth - 1);

    // aboveRow is indexed -1 .. 2*size-1. Store it in a Vec offset by
    // one so logical index -1 maps to slot 0.
    //
    //   above_buf[ k + 1 ] == aboveRow[ k ]   for k in -1 .. 2*size-1
    let mut above_buf = vec![0i32; 2 * size + 1];
    // aboveRow[ i ] for i = 0 .. size-1
    for i in 0..size {
        let v = if !have_above {
            base - 1
        } else {
            let col = (x + i).min(max_x);
            plane_buf.get(col, y - 1)
        };
        above_buf[i + 1] = v;
    }
    // aboveRow[ i ] for i = size .. 2*size-1
    for i in size..2 * size {
        let v = if have_above && not_on_right && tx_sz == 0 {
            let col = (x + i).min(max_x);
            plane_buf.get(col, y - 1)
        } else {
            // aboveRow[ size - 1 ]
            above_buf[size]
        };
        above_buf[i + 1] = v;
    }
    // aboveRow[ -1 ]
    above_buf[0] = if have_above && have_left {
        // x-1 cannot underflow: haveLeft == 1 implies x >= 1.
        let col = (x - 1).min(max_x);
        plane_buf.get(col, y - 1)
    } else if have_above {
        base + 1
    } else {
        base - 1
    };

    // leftCol[ i ] for i = 0 .. size-1
    let mut left = vec![0i32; size];
    for (i, slot) in left.iter_mut().enumerate() {
        *slot = if have_left {
            let row = (y + i).min(max_y);
            plane_buf.get(x - 1, row)
        } else {
            base + 1
        };
    }

    // Closure shorthands mapping the spec's logical indices.
    //   ar(k) == aboveRow[ k ]  (k may be -1)
    let ar = |k: isize| above_buf[(k + 1) as usize];
    let lc = |i: usize| left[i];

    // pred[ i ][ j ], row-major in a flat Vec of length size*size.
    let mut pred = vec![0i32; size * size];
    let idx = |i: usize, j: usize| i * size + j;

    match mode {
        PredMode::VPred => {
            for i in 0..size {
                for j in 0..size {
                    pred[idx(i, j)] = ar(j as isize);
                }
            }
        }
        PredMode::HPred => {
            for i in 0..size {
                for j in 0..size {
                    pred[idx(i, j)] = lc(i);
                }
            }
        }
        PredMode::D207Pred => {
            // 1. pred[size-1][j] = leftCol[size-1]
            for j in 0..size {
                pred[idx(size - 1, j)] = lc(size - 1);
            }
            // 2. pred[i][0] = Round2(leftCol[i] + leftCol[i+1], 1), i=0..size-2
            for i in 0..size - 1 {
                pred[idx(i, 0)] = round2(lc(i) + lc(i + 1), 1);
            }
            // 3. pred[i][1] = Round2(leftCol[i] + 2*leftCol[i+1] + leftCol[i+2], 2),
            //    i=0..size-3
            if size >= 3 {
                for i in 0..size - 2 {
                    pred[idx(i, 1)] = round2(lc(i) + 2 * lc(i + 1) + lc(i + 2), 2);
                }
            }
            // 4. pred[size-2][1] = Round2(leftCol[size-2] + 3*leftCol[size-1], 2)
            if size >= 2 {
                pred[idx(size - 2, 1)] = round2(lc(size - 2) + 3 * lc(size - 1), 2);
            }
            // 5. pred[i][j] = pred[i+1][j-2], i=(size-2)..0 (reverse), j=2..size-1
            for j in 2..size {
                for i in (0..=size - 2).rev() {
                    pred[idx(i, j)] = pred[idx(i + 1, j - 2)];
                }
            }
        }
        PredMode::D45Pred => {
            let last = ar((2 * size - 1) as isize);
            for i in 0..size {
                for j in 0..size {
                    pred[idx(i, j)] = if i + j + 2 < size * 2 {
                        round2(
                            ar((i + j) as isize)
                                + ar((i + j + 1) as isize) * 2
                                + ar((i + j + 2) as isize),
                            2,
                        )
                    } else {
                        last
                    };
                }
            }
        }
        PredMode::D63Pred => {
            for i in 0..size {
                for j in 0..size {
                    let h = i / 2;
                    pred[idx(i, j)] = if i & 1 == 1 {
                        round2(
                            ar((h + j) as isize)
                                + ar((h + j + 1) as isize) * 2
                                + ar((h + j + 2) as isize),
                            2,
                        )
                    } else {
                        round2(ar((h + j) as isize) + ar((h + j + 1) as isize), 1)
                    };
                }
            }
        }
        PredMode::D117Pred => {
            // 1. pred[0][j] = Round2(aboveRow[j-1] + aboveRow[j], 1), j=0..size-1
            for j in 0..size {
                pred[idx(0, j)] = round2(ar(j as isize - 1) + ar(j as isize), 1);
            }
            // 2. pred[1][0] = Round2(leftCol[0] + 2*aboveRow[-1] + aboveRow[0], 2)
            pred[idx(1, 0)] = round2(lc(0) + 2 * ar(-1) + ar(0), 2);
            // 3. pred[1][j] = Round2(aboveRow[j-2] + 2*aboveRow[j-1] + aboveRow[j], 2),
            //    j=1..size-1
            for j in 1..size {
                pred[idx(1, j)] = round2(
                    ar(j as isize - 2) + 2 * ar(j as isize - 1) + ar(j as isize),
                    2,
                );
            }
            // 4. pred[2][0] = Round2(aboveRow[-1] + 2*leftCol[0] + leftCol[1], 2)
            if size >= 2 {
                pred[idx(2, 0)] = round2(ar(-1) + 2 * lc(0) + lc(1), 2);
            }
            // 5. pred[i][0] = Round2(leftCol[i-3] + 2*leftCol[i-2] + leftCol[i-1], 2),
            //    i=3..size-1
            for i in 3..size {
                pred[idx(i, 0)] = round2(lc(i - 3) + 2 * lc(i - 2) + lc(i - 1), 2);
            }
            // 6. pred[i][j] = pred[i-2][j-1], i=2..size-1, j=1..size-1
            for i in 2..size {
                for j in 1..size {
                    pred[idx(i, j)] = pred[idx(i - 2, j - 1)];
                }
            }
        }
        PredMode::D135Pred => {
            // 1. pred[0][0] = Round2(leftCol[0] + 2*aboveRow[-1] + aboveRow[0], 2)
            pred[idx(0, 0)] = round2(lc(0) + 2 * ar(-1) + ar(0), 2);
            // 2. pred[0][j] = Round2(aboveRow[j-2] + 2*aboveRow[j-1] + aboveRow[j], 2),
            //    j=1..size-1
            for j in 1..size {
                pred[idx(0, j)] = round2(
                    ar(j as isize - 2) + 2 * ar(j as isize - 1) + ar(j as isize),
                    2,
                );
            }
            // 3. pred[1][0] = Round2(aboveRow[-1] + 2*leftCol[0] + leftCol[1], 2)
            if size >= 2 {
                pred[idx(1, 0)] = round2(ar(-1) + 2 * lc(0) + lc(1), 2);
            }
            // 4. pred[i][0] = Round2(leftCol[i-2] + 2*leftCol[i-1] + leftCol[i], 2),
            //    i=2..size-1
            for i in 2..size {
                pred[idx(i, 0)] = round2(lc(i - 2) + 2 * lc(i - 1) + lc(i), 2);
            }
            // 5. pred[i][j] = pred[i-1][j-1], i=1..size-1, j=1..size-1
            for i in 1..size {
                for j in 1..size {
                    pred[idx(i, j)] = pred[idx(i - 1, j - 1)];
                }
            }
        }
        PredMode::D153Pred => {
            // 1. pred[0][0] = Round2(leftCol[0] + aboveRow[-1], 1)
            pred[idx(0, 0)] = round2(lc(0) + ar(-1), 1);
            // 2. pred[i][0] = Round2(leftCol[i-1] + leftCol[i], 1), i=1..size-1
            for i in 1..size {
                pred[idx(i, 0)] = round2(lc(i - 1) + lc(i), 1);
            }
            // 3. pred[0][1] = Round2(leftCol[0] + 2*aboveRow[-1] + aboveRow[0], 2)
            if size >= 2 {
                pred[idx(0, 1)] = round2(lc(0) + 2 * ar(-1) + ar(0), 2);
            }
            // 4. pred[1][1] = Round2(aboveRow[-1] + 2*leftCol[0] + leftCol[1], 2)
            if size >= 2 {
                pred[idx(1, 1)] = round2(ar(-1) + 2 * lc(0) + lc(1), 2);
            }
            // 5. pred[i][1] = Round2(leftCol[i-2] + 2*leftCol[i-1] + leftCol[i], 2),
            //    i=2..size-1
            for i in 2..size {
                pred[idx(i, 1)] = round2(lc(i - 2) + 2 * lc(i - 1) + lc(i), 2);
            }
            // 6. pred[0][j] = Round2(aboveRow[j-3] + 2*aboveRow[j-2] + aboveRow[j-1], 2),
            //    j=2..size-1
            for j in 2..size {
                pred[idx(0, j)] = round2(
                    ar(j as isize - 3) + 2 * ar(j as isize - 2) + ar(j as isize - 1),
                    2,
                );
            }
            // 7. pred[i][j] = pred[i-1][j-2], i=1..size-1, j=2..size-1
            for i in 1..size {
                for j in 2..size {
                    pred[idx(i, j)] = pred[idx(i - 1, j - 2)];
                }
            }
        }
        PredMode::TmPred => {
            for i in 0..size {
                for j in 0..size {
                    pred[idx(i, j)] = clip1(ar(j as isize) + lc(i) - ar(-1), bit_depth);
                }
            }
        }
        PredMode::DcPred => {
            let fill = if have_left && have_above {
                let mut sum = 0i32;
                for k in 0..size {
                    sum += lc(k);
                    sum += ar(k as isize);
                }
                (sum + size as i32) >> (log2_size + 1)
            } else if have_left {
                let mut sum = 0i32;
                for k in 0..size {
                    sum += lc(k);
                }
                (sum + (1 << (log2_size - 1))) >> log2_size
            } else if have_above {
                let mut sum = 0i32;
                for k in 0..size {
                    sum += ar(k as isize);
                }
                (sum + (1 << (log2_size - 1))) >> log2_size
            } else {
                1 << (bit_depth - 1)
            };
            for slot in pred.iter_mut() {
                *slot = fill;
            }
        }
    }

    // Store pred back into CurrFrame.
    for i in 0..size {
        for j in 0..size {
            plane_buf.set(x + j, y + i, pred[idx(i, j)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BD: u32 = 8;

    /// A plane large enough that the block + one row/col of neighbours
    /// fits, with `max_x` / `max_y` set to the inclusive edges.
    fn plane(w: usize, h: usize) -> Plane {
        Plane::new(w, h)
    }

    #[test]
    fn pred_mode_from_raw_round_trips_spec_numbering() {
        assert_eq!(PredMode::from_raw(0), Some(PredMode::DcPred));
        assert_eq!(PredMode::from_raw(1), Some(PredMode::VPred));
        assert_eq!(PredMode::from_raw(2), Some(PredMode::HPred));
        assert_eq!(PredMode::from_raw(3), Some(PredMode::D45Pred));
        assert_eq!(PredMode::from_raw(4), Some(PredMode::D135Pred));
        assert_eq!(PredMode::from_raw(5), Some(PredMode::D117Pred));
        assert_eq!(PredMode::from_raw(6), Some(PredMode::D153Pred));
        assert_eq!(PredMode::from_raw(7), Some(PredMode::D207Pred));
        assert_eq!(PredMode::from_raw(8), Some(PredMode::D63Pred));
        assert_eq!(PredMode::from_raw(9), Some(PredMode::TmPred));
        assert_eq!(PredMode::from_raw(10), None);
        assert_eq!(PredMode::DcPred as u8, 0);
        assert_eq!(PredMode::TmPred as u8, 9);
    }

    #[test]
    fn round2_matches_spec_definition() {
        // Round2(x, n) = (x + (1 << (n-1))) >> n.
        assert_eq!(round2(0, 1), 0);
        assert_eq!(round2(1, 1), 1); // (1 + 1) >> 1
        assert_eq!(round2(2, 1), 1); // (2 + 1) >> 1
        assert_eq!(round2(3, 1), 2); // (3 + 1) >> 1
        assert_eq!(round2(5, 2), 1); // (5 + 2) >> 2
        assert_eq!(round2(6, 2), 2); // (6 + 2) >> 2
    }

    #[test]
    fn clip1_clamps_to_bit_depth_range() {
        assert_eq!(clip1(-5, 8), 0);
        assert_eq!(clip1(300, 8), 255);
        assert_eq!(clip1(128, 8), 128);
        assert_eq!(clip1(300, 10), 300);
        assert_eq!(clip1(2000, 10), 1023);
    }

    /// V_PRED: every row equals aboveRow[0..size-1].
    #[test]
    fn v_pred_copies_above_row_down() {
        // 4x4 block at (1,1); row above (y=0) carries a ramp 10..13 in
        // columns 1..4. haveLeft is irrelevant to V_PRED.
        let mut p = plane(8, 8);
        for j in 0..4 {
            p.set(1 + j, 0, 10 + j as i32);
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::VPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 10 + j as i32, "i={i} j={j}");
            }
        }
    }

    /// H_PRED: every column equals leftCol[0..size-1].
    #[test]
    fn h_pred_copies_left_col_across() {
        let mut p = plane(8, 8);
        // left column (x=0) ramp 20..23 in rows 1..4.
        for i in 0..4 {
            p.set(0, 1 + i, 20 + i as i32);
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::HPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 20 + i as i32, "i={i} j={j}");
            }
        }
    }

    /// DC_PRED with both neighbours: avg = (sum + size) >> (log2Size+1).
    #[test]
    fn dc_pred_both_averages_above_and_left() {
        let mut p = plane(8, 8);
        // aboveRow = [4,4,4,4], leftCol = [8,8,8,8] -> sum = 48, size=4.
        // avg = (48 + 4) >> 3 = 52 >> 3 = 6.
        for j in 0..4 {
            p.set(1 + j, 0, 4);
        }
        for i in 0..4 {
            p.set(0, 1 + i, 8);
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::DcPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 6, "i={i} j={j}");
            }
        }
    }

    /// DC_PRED with only the left neighbour:
    /// leftAvg = (sum + (1 << (log2Size-1))) >> log2Size.
    #[test]
    fn dc_pred_left_only() {
        let mut p = plane(8, 8);
        // leftCol = [10,10,10,10] -> sum=40, log2Size=2.
        // leftAvg = (40 + 2) >> 2 = 42 >> 2 = 10.
        for i in 0..4 {
            p.set(0, 1 + i, 10);
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            false,
            false,
            0,
            PredMode::DcPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 10, "i={i} j={j}");
            }
        }
    }

    /// DC_PRED with only the above neighbour mirrors the left-only path.
    #[test]
    fn dc_pred_above_only() {
        let mut p = plane(8, 8);
        // aboveRow = [12,12,12,12] -> aboveAvg = (48 + 2) >> 2 = 12.
        for j in 0..4 {
            p.set(1 + j, 0, 12);
        }
        predict_intra(
            &mut p,
            1,
            1,
            false,
            true,
            false,
            0,
            PredMode::DcPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 12, "i={i} j={j}");
            }
        }
    }

    /// DC_PRED with no neighbours fills 1 << (BitDepth - 1).
    #[test]
    fn dc_pred_no_neighbours_fills_midpoint() {
        let mut p = plane(8, 8);
        predict_intra(
            &mut p,
            1,
            1,
            false,
            false,
            false,
            0,
            PredMode::DcPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 128, "i={i} j={j}");
            }
        }
    }

    /// TM_PRED: pred[i][j] = Clip1(aboveRow[j] + leftCol[i] - aboveRow[-1]).
    #[test]
    fn tm_pred_true_motion_formula() {
        let mut p = plane(8, 8);
        // aboveRow[-1] is CurrFrame[plane][y-1][x-1] = (0,0).
        p.set(0, 0, 100); // aboveRow[-1]
        for j in 0..4 {
            p.set(1 + j, 0, 100 + j as i32); // aboveRow[j]
        }
        for i in 0..4 {
            p.set(0, 1 + i, 100 + 10 * i as i32); // leftCol[i]
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::TmPred,
            7,
            7,
            BD,
        );
        for i in 0..4 {
            for j in 0..4 {
                // (100+j) + (100+10i) - 100 = 100 + j + 10i, clipped to [0,255].
                let expected = (100 + j as i32 + 10 * i as i32).clamp(0, 255);
                assert_eq!(p.get(1 + j, 1 + i), expected, "i={i} j={j}");
            }
        }
    }

    /// TM_PRED clips below 0 and above the bit-depth max.
    #[test]
    fn tm_pred_clips_out_of_range() {
        let mut p = plane(8, 8);
        // aboveRow[-1] high, aboveRow / leftCol low -> negative -> clip to 0.
        p.set(0, 0, 250); // aboveRow[-1]
        for j in 0..4 {
            p.set(1 + j, 0, 10);
        }
        for i in 0..4 {
            p.set(0, 1 + i, 10);
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            true,
            false,
            0,
            PredMode::TmPred,
            7,
            7,
            BD,
        );
        // 10 + 10 - 250 = -230 -> 0.
        assert_eq!(p.get(1, 1), 0);
    }

    /// Constant neighbours -> constant prediction for every directional
    /// mode. With aboveRow == leftCol == aboveRow[-1] == C, every
    /// weighted Round2 average collapses to C, so all modes must fill C.
    #[test]
    fn directional_modes_constant_neighbours_give_constant() {
        let modes = [
            PredMode::D45Pred,
            PredMode::D63Pred,
            PredMode::D117Pred,
            PredMode::D135Pred,
            PredMode::D153Pred,
            PredMode::D207Pred,
        ];
        const C: i32 = 64;
        for &mode in &modes {
            for tx in 0..=3u32 {
                let size = 1usize << (tx + 2);
                // Big enough plane: block at (1,1), neighbours need
                // aboveRow up to 2*size, so width >= 1 + 2*size + 1.
                let w = 2 + 2 * size + 2;
                let h = 2 + size + 2;
                let mut p = plane(w, h);
                // Fill the whole above row and left column (and the
                // corner) with C so every neighbour read is C.
                for x in 0..w {
                    p.set(x, 0, C);
                }
                for yy in 0..h {
                    p.set(0, yy, C);
                }
                let max_x = w - 1;
                let max_y = h - 1;
                predict_intra(&mut p, 1, 1, true, true, true, tx, mode, max_x, max_y, BD);
                for i in 0..size {
                    for j in 0..size {
                        assert_eq!(p.get(1 + j, 1 + i), C, "mode={mode:?} tx={tx} i={i} j={j}");
                    }
                }
            }
        }
    }

    /// D207_PRED bottom row is filled with leftCol[size-1] (step 1).
    #[test]
    fn d207_bottom_row_is_left_last() {
        let mut p = plane(8, 8);
        // leftCol ramp 0,10,20,30; aboveRow all 0.
        for i in 0..4 {
            p.set(0, 1 + i, 10 * i as i32);
        }
        predict_intra(
            &mut p,
            1,
            1,
            true,
            false,
            false,
            0,
            PredMode::D207Pred,
            7,
            7,
            BD,
        );
        // pred[size-1][j] = leftCol[size-1] = 30 for all j.
        for j in 0..4 {
            assert_eq!(p.get(1 + j, 1 + 3), 30, "j={j}");
        }
        // pred[0][0] = Round2(leftCol[0] + leftCol[1], 1) = Round2(0+10,1) = 5.
        assert_eq!(p.get(1, 1), 5);
        // pred[1][0] = Round2(leftCol[1] + leftCol[2], 1) = Round2(10+20,1) = 15.
        assert_eq!(p.get(1, 2), 15);
    }

    /// V_PRED extends aboveRow[size-1] for the upper-right region when
    /// notOnRight is 0 (the right-extension condition fails). With a
    /// step ramp in the above row, columns beyond size-1 are not read
    /// for V_PRED at all (j only spans 0..size), so this test instead
    /// checks the aboveRow[size..] extension via D45 at the block's
    /// bottom-right corner.
    #[test]
    fn d45_bottom_right_uses_extended_above() {
        // For 4x4 with txSz=0 but notOnRight=0, aboveRow[size..2size-1]
        // all equal aboveRow[size-1]. pred[size-1][size-1] uses
        // aboveRow[2*size-1] which is the extension value.
        let mut p = plane(16, 8);
        // aboveRow[0..size-1] = 0..3 (columns 1..4 of row 0). Columns
        // beyond stay 0 but are not consulted because notOnRight=0.
        for j in 0..4 {
            p.set(1 + j, 0, j as i32);
        }
        predict_intra(
            &mut p,
            1,
            1,
            false,
            true,
            false,
            0,
            PredMode::D45Pred,
            15,
            7,
            BD,
        );
        // pred[3][3]: i+j+2 = 8 = size*2, so not < size*2 -> aboveRow[2*size-1].
        // Since notOnRight=0, aboveRow[2*size-1] = aboveRow[size-1] = 3.
        assert_eq!(p.get(1 + 3, 1 + 3), 3);
    }

    /// max_x / max_y clamp neighbour reads at the plane edge: a block
    /// flush against the right edge re-reads the last valid column.
    #[test]
    fn neighbour_reads_clamp_at_max_x() {
        // 4x4 block whose right side is at the plane edge. aboveRow[i]
        // for x+i beyond max_x must clamp to CurrFrame[plane][y-1][max_x].
        let mut p = plane(6, 8);
        // Plane is 6 wide (cols 0..5). Block at x=2 spans cols 2..5,
        // but neighbour reads of aboveRow go up to x+size-1 = 5 here
        // (in range). Force a clamp by setting max_x smaller than the
        // buffer width to emulate a frame edge inside a padded buffer.
        for x in 0..6 {
            p.set(x, 0, x as i32);
        }
        let max_x = 3; // emulate frame edge at column 3.
        predict_intra(
            &mut p,
            2,
            1,
            false,
            true,
            false,
            0,
            PredMode::VPred,
            max_x,
            7,
            BD,
        );
        // aboveRow[0] = row0[min(3, 2)] = 2.
        assert_eq!(p.get(2, 1), 2);
        // aboveRow[1] = row0[min(3, 3)] = 3.
        assert_eq!(p.get(3, 1), 3);
        // aboveRow[2] = row0[min(3, 4)] = row0[3] = 3 (clamped).
        assert_eq!(p.get(4, 1), 3);
        // aboveRow[3] = row0[min(3, 5)] = row0[3] = 3 (clamped).
        assert_eq!(p.get(5, 1), 3);
    }

    /// notOnRight + txSz==0 + haveAbove enables the real upper-right
    /// extension of aboveRow (vs. the aboveRow[size-1] hold).
    #[test]
    fn upper_right_extension_enabled_only_for_tx0_not_on_right() {
        // 4x4 (txSz=0) D45 with notOnRight=1: aboveRow[size..2size-1]
        // read real samples; pred[0][size-1] uses aboveRow[size-1],
        // aboveRow[size], aboveRow[size+1].
        let mut p = plane(16, 8);
        // Above row columns 1..8 carry 0,1,2,3,4,5,6,7.
        for j in 0..8 {
            p.set(1 + j, 0, j as i32);
        }
        predict_intra(
            &mut p,
            1,
            1,
            false,
            true,
            true,
            0,
            PredMode::D45Pred,
            15,
            7,
            BD,
        );
        // pred[0][3]: i+j+2 = 5 < 8, so Round2(ar[3] + 2*ar[4] + ar[5], 2)
        //  = Round2(3 + 8 + 5, 2) = Round2(16, 2) = 4.
        assert_eq!(p.get(1 + 3, 1), 4);
    }

    /// Every §8.5.1 predictor — *all ten* modes, not just the six
    /// directionals already pinned — must reproduce a uniform value when
    /// the entire above row, left column, and corner are that same value:
    /// each mode is an affine combination of edge samples whose
    /// coefficients sum to one (the `Round2` averages, the `V`/`H`/`D*`
    /// copies, the DC mean, and the `TM = left + above - corner` formula
    /// which collapses to the corner). Pins flat-preservation across every
    /// mode and transform size in one sweep.
    #[test]
    fn every_mode_preserves_a_flat_neighbourhood() {
        let modes = [
            PredMode::DcPred,
            PredMode::VPred,
            PredMode::HPred,
            PredMode::D45Pred,
            PredMode::D135Pred,
            PredMode::D117Pred,
            PredMode::D153Pred,
            PredMode::D207Pred,
            PredMode::D63Pred,
            PredMode::TmPred,
        ];
        const C: i32 = 100;
        for &mode in &modes {
            for tx in 0..=3u32 {
                let size = 1usize << (tx + 2);
                let w = 2 + 2 * size + 2;
                let h = 2 + size + 2;
                let mut p = plane(w, h);
                for x in 0..w {
                    p.set(x, 0, C);
                }
                for yy in 0..h {
                    p.set(0, yy, C);
                }
                let (max_x, max_y) = (w - 1, h - 1);
                predict_intra(&mut p, 1, 1, true, true, true, tx, mode, max_x, max_y, BD);
                for i in 0..size {
                    for j in 0..size {
                        assert_eq!(p.get(1 + j, 1 + i), C, "mode={mode:?} tx={tx} i={i} j={j}");
                    }
                }
            }
        }
    }

    /// Every §8.5.1 predictor output sample is `Clip1`-bounded into
    /// `[0, (1 << BitDepth) - 1]` regardless of the neighbour values. Only
    /// `TM_PRED` (`left + above - corner`) and the DC-with-no-neighbour
    /// fills can leave the source range, but the spec clips them; the
    /// directional / DC-average modes are convex combinations and stay in
    /// range by construction. This drives every mode with deterministic
    /// pseudo-random in-range neighbours (and the saturating extremes that
    /// would make `TM` overshoot) and asserts no out-of-range write — a
    /// missing `Clip1` would surface here rather than as a corrupt
    /// reconstructed sample downstream.
    #[test]
    fn every_mode_output_is_clip1_bounded() {
        let modes = [
            PredMode::DcPred,
            PredMode::VPred,
            PredMode::HPred,
            PredMode::D45Pred,
            PredMode::D135Pred,
            PredMode::D117Pred,
            PredMode::D153Pred,
            PredMode::D207Pred,
            PredMode::D63Pred,
            PredMode::TmPred,
        ];
        let max_val = (1i32 << BD) - 1;
        let mut state: u64 = 0xBADC_0FFE_E0DD_F00D;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i32
        };
        for &mode in &modes {
            for tx in 0..=3u32 {
                let size = 1usize << (tx + 2);
                let w = 2 + 2 * size + 2;
                let h = 2 + size + 2;
                for trial in 0..40 {
                    let mut p = plane(w, h);
                    // Bias the neighbours toward the range extremes on
                    // some trials so TM_PRED's left+above-corner overshoots
                    // both ends and the clip is actually exercised.
                    let fill = |n: &mut dyn FnMut() -> i32, trial: usize| -> i32 {
                        match trial % 3 {
                            0 => n() % (max_val + 1),
                            1 => max_val,
                            _ => 0,
                        }
                    };
                    for x in 0..w {
                        p.set(x, 0, fill(&mut next, trial));
                    }
                    for yy in 0..h {
                        p.set(0, yy, fill(&mut next, trial + 1));
                    }
                    let (max_x, max_y) = (w - 1, h - 1);
                    predict_intra(&mut p, 1, 1, true, true, true, tx, mode, max_x, max_y, BD);
                    for i in 0..size {
                        for j in 0..size {
                            let v = p.get(1 + j, 1 + i);
                            assert!(
                                (0..=max_val).contains(&v),
                                "mode={mode:?} tx={tx} trial={trial} i={i} j={j}: {v} out of [0,{max_val}]"
                            );
                        }
                    }
                }
            }
        }
    }
}
