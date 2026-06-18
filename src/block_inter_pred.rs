//! §8.5.2.4 block inter prediction process — the sub-pixel interpolation
//! engine that turns a scaled reference-frame sampling location into a
//! `w × h` block of predicted samples.
//!
//! This is the leaf of the §8.5.2 inter prediction process. The earlier
//! steps — §8.5.2.1 motion-vector selection, §8.5.2.2 clamping, and
//! §8.5.2.3 scaling — combine the decoded `BlockMvs[ ]` with the
//! reference-frame geometry to produce the inputs this process consumes:
//! `startX` / `startY` (the top-left reference sampling location, in
//! units of 1/16 th of a sample) and `xStep` / `yStep` (the per-sample
//! step, also in 1/16 ths, at most 80 due to the §8.5.2.3 scaling
//! restriction). Those three steps are separate primitives that land on
//! top of this one; this module is the convolution kernel they feed.
//!
//! The interpolation is two one-dimensional 8-tap convolutions per
//! §8.5.2.4 (`vp9-spec.txt` lines 4711-4738): first a horizontal pass
//! builds an `intermediateHeight × w` array from the reference plane,
//! then a vertical pass filters that array into the final `h × w`
//! prediction. The fractional part of the sampling position selects one
//! of the 16 phase rows of [`SUBPEL_FILTERS`]; phase 0 is a straight
//! sample copy (`{ 0, 0, 0, 128, 0, 0, 0, 0 }`). The reference reads are
//! edge-clamped with `Clip3( 0, lastX/lastY, … )` so a motion vector
//! that points off the reference plane replicates the border sample.
//!
//! Everything here is a pure function of the caller-supplied reference
//! plane and the scalar inputs, so it is directly testable against a
//! hand-built reference plane without threading any frame-wide decode
//! state. The §8.5.2 driver that selects, clamps, and scales the motion
//! vector — and the reference-buffer state it reads `FrameStore[ ]` from
//! — is a later step.
//!
//! Single source of truth: `docs/video/vp9/vp9-spec.txt` §8.5.2.4.

// The §8.5.2 inter prediction driver that invokes this leaf (selecting,
// clamping, and scaling the motion vector, then writing the result into
// CurrFrame) lands in a later round; until then this primitive is
// reachable only from the unit tests, so the crate-internal `dead_code`
// lint is silenced module-wide (mirrors `mv` / `mode_info`'s deferred
// inter primitives).
#![allow(dead_code)]

/// `SUBPEL_BITS = 4` per spec §3 (`vp9-spec.txt` line 517). Number of
/// bits of precision when performing inter prediction; the sampling
/// position is carried in 1/16 th-of-a-sample units, so the integer part
/// is `p >> SUBPEL_BITS` and the fractional phase is `p & SUBPEL_MASK`.
pub(crate) const SUBPEL_BITS: u32 = 4;

/// `SUBPEL_MASK = 15` per spec §3 (`vp9-spec.txt` line 519). Equals
/// `(1 << SUBPEL_BITS) - 1`; masks the sampling position down to the
/// 16-entry sub-pixel phase index into [`SUBPEL_FILTERS`].
pub(crate) const SUBPEL_MASK: i32 = 15;

/// `subpel_filters[ 4 ][ 16 ][ 8 ]` per §8.5.2.4 (`vp9-spec.txt` lines
/// 4742-4831). Outer index is `interp_filter`
/// (0 = `EIGHTTAP`, 1 = `EIGHTTAP_SMOOTH`, 2 = `EIGHTTAP_SHARP`,
/// 3 = `BILINEAR`); middle index is the 16-phase sub-pixel position
/// (`p & 15`); inner index is the 8-tap kernel. Each kernel sums to 128
/// (`1 << 7`), matching the `Round2( s, 7 )` normalisation. Transcribed
/// verbatim from the §8.5.2.4 listing.
pub(crate) const SUBPEL_FILTERS: [[[i32; 8]; 16]; 4] = [
    // interp_filter = 0 — EIGHTTAP.
    [
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
    ],
    // interp_filter = 1 — EIGHTTAP_SMOOTH.
    [
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
    ],
    // interp_filter = 2 — EIGHTTAP_SHARP.
    [
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
    ],
    // interp_filter = 3 — BILINEAR.
    [
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
    ],
];

/// `Clip3( low, high, x )` per spec §3 (`vp9-spec.txt` line 624).
#[inline]
fn clip3(low: i32, high: i32, x: i32) -> i32 {
    if x < low {
        low
    } else if x > high {
        high
    } else {
        x
    }
}

/// `Round2( x, n ) = ( x + (1 << (n − 1)) ) >> n` per spec §3
/// (`vp9-spec.txt` line 636). The §8.5.2.4 convolutions normalise each
/// 8-tap sum with `Round2( s, 7 )`.
#[inline]
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// `Clip1( x ) = Clip3( 0, (1 << BitDepth) - 1, x )` per spec §3
/// (`vp9-spec.txt` line 626). Both §8.5.2.4 passes clip their rounded
/// sum to the sample range so the intermediate stays representable.
#[inline]
fn clip1(x: i32, bit_depth: u32) -> i32 {
    clip3(0, (1 << bit_depth) - 1, x)
}

/// §8.5.2.4 block inter prediction process (`vp9-spec.txt` lines
/// 4682-4738).
///
/// Runs the two-pass 8-tap sub-pixel convolution that turns a scaled
/// reference sampling location into a `w × h` prediction block. The
/// caller supplies the reference plane through `ref_sample`, a closure
/// returning `ref[ plane ][ row ][ col ]` for already edge-clamped
/// integer coordinates `0 ≤ row ≤ last_y`, `0 ≤ col ≤ last_x`.
///
/// Inputs (all per §8.5.2.4):
/// * `x` / `y` — block location in units of 1/16 th of a sample
///   (`startX` / `startY` from §8.5.2.3).
/// * `x_step` / `y_step` — per-sample step in 1/16 ths (`xStep` /
///   `yStep`; at most 80).
/// * `w` / `h` — prediction block width / height in samples.
/// * `interp_filter` — 0..3, the outer [`SUBPEL_FILTERS`] index.
/// * `last_x` / `last_y` — coordinates of the bottom-right reference
///   sample (`( ( RefFrameWidth + subX ) >> subX ) - 1` etc.).
/// * `bit_depth` — `BitDepth` (8, 10, or 12) for the `Clip1( )` range.
///
/// Returns the `pred` array flattened row-major (`pred[ r ][ c ]` at
/// index `r * w + c`).
///
/// # Panics
///
/// Debug-asserts that `interp_filter < 4` and that `w` and `h` are
/// non-zero; release builds index the filter table directly.
// The §8.5.2.4 process is parameterised on the full set of scaled
// sampling inputs the §8.5.2.1-3 steps produce; the positional list
// mirrors the spec rather than bundling into a struct.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_inter_predict<F>(
    ref_sample: F,
    x: i32,
    y: i32,
    x_step: i32,
    y_step: i32,
    w: usize,
    h: usize,
    interp_filter: usize,
    last_x: i32,
    last_y: i32,
    bit_depth: u32,
) -> Vec<i32>
where
    F: Fn(i32, i32) -> i32,
{
    debug_assert!(interp_filter < 4, "interp_filter must be 0..3");
    debug_assert!(w > 0 && h > 0, "block dimensions must be non-zero");

    let filter = &SUBPEL_FILTERS[interp_filter];

    // §8.5.2.4 line 4709: intermediateHeight = (((h - 1) * yStep + 15) >> 4) + 8.
    let intermediate_height = ((((h as i32 - 1) * y_step + 15) >> 4) + 8) as usize;

    // First pass — §8.5.2.4 lines 4716-4726. Horizontal 8-tap filter
    // builds `intermediate[ intermediateHeight ][ w ]` from the
    // reference plane, edge-clamping both axes.
    let mut intermediate = vec![0i32; intermediate_height * w];
    for r in 0..intermediate_height {
        for c in 0..w {
            // p = x + xStep * c.
            let p = x + x_step * c as i32;
            let phase = (p & SUBPEL_MASK) as usize;
            let taps = &filter[phase];
            // ref row is fixed within this pass: Clip3( 0, lastY, (y >> 4) + r - 3 ).
            let ref_row = clip3(0, last_y, (y >> 4) + r as i32 - 3);
            let mut s = 0i32;
            for (t, &tap) in taps.iter().enumerate() {
                // ref col: Clip3( 0, lastX, (p >> 4) + t - 3 ).
                let ref_col = clip3(0, last_x, (p >> 4) + t as i32 - 3);
                s += tap * ref_sample(ref_row, ref_col);
            }
            intermediate[r * w + c] = clip1(round2(s, 7), bit_depth);
        }
    }

    // Second pass — §8.5.2.4 lines 4729-4738. Vertical 8-tap filter
    // turns the intermediate array into the final `pred[ h ][ w ]`.
    let mut pred = vec![0i32; h * w];
    for r in 0..h {
        for c in 0..w {
            // p = (y & 15) + yStep * r.
            let p = (y & SUBPEL_MASK) + y_step * r as i32;
            let phase = (p & SUBPEL_MASK) as usize;
            let taps = &filter[phase];
            let base_row = (p >> 4) as usize;
            let mut s = 0i32;
            for (t, &tap) in taps.iter().enumerate() {
                s += tap * intermediate[(base_row + t) * w + c];
            }
            pred[r * w + c] = clip1(round2(s, 7), bit_depth);
        }
    }

    pred
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every §8.5.2.4 sub-pixel kernel sums to `1 << 7 = 128`, the
    /// normalisation that `Round2( s, 7 )` divides out.
    #[test]
    fn every_subpel_kernel_sums_to_128() {
        for (fi, filter) in SUBPEL_FILTERS.iter().enumerate() {
            for (phase, taps) in filter.iter().enumerate() {
                let sum: i32 = taps.iter().sum();
                assert_eq!(sum, 128, "filter {fi} phase {phase} sum {sum}");
            }
        }
    }

    /// Phase 0 of every filter is the identity copy kernel
    /// `{ 0, 0, 0, 128, 0, 0, 0, 0 }` — the tap on index 3.
    #[test]
    fn phase_zero_is_identity_copy() {
        for filter in &SUBPEL_FILTERS {
            assert_eq!(filter[0], [0, 0, 0, 128, 0, 0, 0, 0]);
        }
    }

    /// With both motion-vector fractional parts zero (`x`/`y` multiples
    /// of 16) and `xStep = yStep = 16`, the filtering reduces to a plain
    /// sample copy: `pred[ r ][ c ] = ref[ (y>>4)+r ][ (x>>4)+c ]`.
    #[test]
    fn integer_position_is_a_plain_copy() {
        // 8×8 ramp reference plane, value = row * 16 + col (all in the
        // 8-bit Clip1 range so the copy is bit-exact).
        let last_x = 7;
        let last_y = 7;
        let refp = |row: i32, col: i32| row * 16 + col;

        // startX = 2 samples, startY = 1 sample, both integer (×16).
        let (sx, sy) = (2 * 16, 16);
        let (w, h) = (4usize, 3usize);
        let pred = block_inter_predict(refp, sx, sy, 16, 16, w, h, 0, last_x, last_y, 8);

        for r in 0..h {
            for c in 0..w {
                let expected = refp(1 + r as i32, 2 + c as i32);
                assert_eq!(pred[r * w + c], expected, "r{r} c{c}");
            }
        }
    }

    /// A flat reference plane (every sample equal) must round-trip to the
    /// same flat value regardless of sub-pixel phase, because each kernel
    /// sums to 128 and `Round2( 128 * v, 7 ) = v`.
    #[test]
    fn flat_plane_passes_through_at_every_phase() {
        let refp = |_r: i32, _c: i32| 137;
        // Half-pel in both axes (frac 8), eighttap-sharp filter.
        let pred = block_inter_predict(refp, 8, 8, 16, 16, 4, 4, 2, 31, 31, 8);
        for &v in &pred {
            assert_eq!(v, 137);
        }
    }

    /// Edge clamping: a reference location pushed past the bottom-right
    /// corner replicates the corner sample. A flat-but-clamped read still
    /// yields the corner value everywhere.
    #[test]
    fn off_plane_reads_clamp_to_border() {
        // 2×2 reference; sample (1,1) = 200, others smaller.
        let refp = |r: i32, c: i32| match (r, c) {
            (1, 1) => 200,
            (0, 0) => 10,
            (0, 1) => 20,
            (1, 0) => 30,
            _ => panic!("out of range read ({r},{c}) — clamping failed"),
        };
        // start far past the corner so every clamped read lands on (1,1).
        let pred = block_inter_predict(refp, 100 * 16, 100 * 16, 16, 16, 2, 2, 0, 1, 1, 8);
        for &v in &pred {
            assert_eq!(v, 200);
        }
    }

    /// Half-pel horizontal interpolation on a two-value step edge with
    /// the EIGHTTAP filter reproduces the hand-computed convolution.
    #[test]
    fn half_pel_horizontal_matches_hand_convolution() {
        // 1-row-relevant reference: col < 4 -> 0, col >= 4 -> 64.
        // Use a tall flat-in-row plane so the vertical pass is identity.
        let refp = |_r: i32, c: i32| if c >= 4 { 64 } else { 0 };
        let last_x = 15;
        let last_y = 15;

        // Predict a single sample at integer y (phase 0 vertical), x at
        // half-pel phase 8, sample column 0 -> p = startX.
        // startX = 4 samples << 4 = 64, plus half-pel phase 8 -> 64+8=72.
        let start_x = 4 * 16 + 8;
        let start_y = 4 * 16; // integer row, vertical pass is a copy.
        let pred = block_inter_predict(refp, start_x, start_y, 16, 16, 1, 1, 0, last_x, last_y, 8);

        // Hand-compute the horizontal half-pel (phase 8) tap over the
        // reference row at p>>4 = 4, taps centred at col index (4 + t - 3).
        let taps = SUBPEL_FILTERS[0][8];
        let mut s = 0i32;
        for (t, &tap) in taps.iter().enumerate() {
            let col = clip3(0, last_x, 4 + t as i32 - 3);
            s += tap * (if col >= 4 { 64 } else { 0 });
        }
        let expected = clip1(round2(s, 7), 8);
        assert_eq!(pred[0], expected);
    }

    /// 10-bit `Clip1` range: a flat 1000-valued plane passes through and
    /// stays within `0..=1023`, confirming `bit_depth` threads to the
    /// clip range.
    #[test]
    fn ten_bit_clip_range_threads_through() {
        let refp = |_r: i32, _c: i32| 1000;
        let pred = block_inter_predict(refp, 8, 8, 16, 16, 2, 2, 1, 31, 31, 10);
        for &v in &pred {
            assert_eq!(v, 1000);
            assert!(v <= 1023);
        }
    }

    /// `intermediateHeight` follows the §8.5.2.4 formula. With a scaled
    /// `yStep` larger than 16 the intermediate array grows; the function
    /// must not panic indexing it. Predict an 8-tall block at yStep=80.
    #[test]
    fn scaled_ystep_grows_intermediate_without_panic() {
        let refp = |r: i32, c: i32| (r + c) & 0xff;
        // h = 8, yStep = 80 -> intermediateHeight = (((7*80+15)>>4)+8) = 43.
        let pred = block_inter_predict(refp, 0, 0, 16, 80, 4, 8, 0, 63, 63, 8);
        assert_eq!(pred.len(), 4 * 8);
    }
}
