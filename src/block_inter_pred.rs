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
use crate::frame_buf::Px;

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
/// Row access to a reference plane: `row( y )` is the plane's row `y`
/// (at least `lastX + 1` samples).
pub(crate) trait SampleRows {
    /// The stored sample type.
    type Px: crate::frame_buf::Px;

    fn row(&self, y: usize) -> &[Self::Px];

    /// Rows `y0..y1` as one flat slice (`(data, stride)`, row `y` at
    /// `data[ (y - y0) * stride.. ]`) when they are stored contiguously.
    fn window(&self, y0: usize, y1: usize) -> Option<(&[Self::Px], usize)> {
        let _ = (y0, y1);
        None
    }
}

/// A flat row-major plane of `stride` samples per row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlatRows<'a> {
    pub samples: &'a [i32],
    pub stride: usize,
}

impl SampleRows for FlatRows<'_> {
    type Px = i32;

    #[inline]
    fn row(&self, y: usize) -> &[i32] {
        let s = &self.samples[y * self.stride..];
        &s[..self.stride.min(s.len())]
    }

    #[inline]
    fn window(&self, y0: usize, y1: usize) -> Option<(&[i32], usize)> {
        (y1 * self.stride <= self.samples.len()).then(|| {
            (
                &self.samples[y0 * self.stride..y1 * self.stride],
                self.stride,
            )
        })
    }
}

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
// The generic reference form: the §8.5.2 driver runs the plane form
// below (round 458); this one stays as the oracle the equivalence
// test and the cross-check tests read from.
#[cfg_attr(not(test), allow(dead_code))]
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

/// [`block_inter_predict`] over a row-major sample plane (round 458):
/// the same §8.5.2.4 arithmetic — the horizontal pass reads the
/// reference through `Clip3( 0, lastX, · )` / `Clip3( 0, lastY, · )`
/// exactly as the generic form does — but a column whose eight-tap
/// window lies entirely inside `[ 0, lastX ]` reads its taps straight
/// from the clipped row's slice (no per-tap clip, no per-sample call),
/// which is every interior column. Bit-identical to the generic form
/// (the tap products are accumulated in the same order).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_inter_predict_plane<R: SampleRows>(
    rows_of: &R,
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
) -> Vec<i32> {
    debug_assert!(interp_filter < 4, "interp_filter must be 0..3");
    debug_assert!(w > 0 && h > 0, "block dimensions must be non-zero");

    let filter = &SUBPEL_FILTERS[interp_filter];
    let intermediate_height = ((((h as i32 - 1) * y_step + 15) >> 4) + 8) as usize;

    // Phase 0 is the `{ 0, 0, 0, 128, 0, 0, 0, 0 }` kernel: the filtered
    // value is the centre sample itself (`Clip1(Round2(128 * v, 7)) == v`
    // for a sample of the plane). Without scaling, a whole-sample vertical
    // position reads only intermediate rows 3 ..= h + 2.
    let rows = if y_step == 16 && (y & SUBPEL_MASK) == 0 {
        3..h + 3
    } else {
        0..intermediate_height
    };
    let mut intermediate = vec![0i32; intermediate_height * w];
    for r in rows {
        let ref_row = clip3(0, last_y, (y >> 4) + r as i32 - 3) as usize;
        let row = &rows_of.row(ref_row)[..last_x as usize + 1];
        let px = |c: usize| crate::frame_buf::Px::get(row[c]);
        let out = &mut intermediate[r * w..(r + 1) * w];
        for (c, slot) in out.iter_mut().enumerate() {
            let p = x + x_step * c as i32;
            let phase = (p & SUBPEL_MASK) as usize;
            if phase == 0 {
                *slot = px(clip3(0, last_x, p >> 4) as usize);
                continue;
            }
            let taps = &filter[phase];
            let base = (p >> 4) - 3;
            let mut s = 0i32;
            if base >= 0 && base + 7 <= last_x {
                let win = &row[base as usize..base as usize + 8];
                for (&tap, &v) in taps.iter().zip(win) {
                    s += tap * crate::frame_buf::Px::get(v);
                }
            } else {
                for (t, &tap) in taps.iter().enumerate() {
                    let ref_col = clip3(0, last_x, base + t as i32);
                    s += tap * px(ref_col as usize);
                }
            }
            *slot = clip1(round2(s, 7), bit_depth);
        }
    }

    let mut pred = vec![0i32; h * w];
    for r in 0..h {
        let p = (y & SUBPEL_MASK) + y_step * r as i32;
        let phase = (p & SUBPEL_MASK) as usize;
        let taps = &filter[phase];
        let base_row = (p >> 4) as usize;
        if phase == 0 {
            let src = (base_row + 3) * w;
            pred[r * w..(r + 1) * w].copy_from_slice(&intermediate[src..src + w]);
            continue;
        }
        for c in 0..w {
            let mut s = 0i32;
            for (t, &tap) in taps.iter().enumerate() {
                s += tap * intermediate[(base_row + t) * w + c];
            }
            pred[r * w + c] = clip1(round2(s, 7), bit_depth);
        }
    }
    pred
}

/// [`block_inter_predict_plane`] writing its `w x h` prediction into
/// `out` (row `r` at `out[ r * out_stride.. ]`).
///
/// The common case — an unscaled reference (`xStep == yStep == 16`)
/// whose whole 8-tap footprint lies inside the plane — runs a direct
/// separable filter over row slices with the phase fixed per block
/// (an unscaled position advances by whole samples, so `p & 15` is the
/// same for every column / row); its clamps are no-ops there, so the
/// bytes are those of the §8.5.2.3 process. Everything else (scaled
/// references, footprints crossing the plane edge) takes the general
/// path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_inter_predict_into<R: SampleRows>(
    rows_of: &R,
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
    out: &mut [i32],
    out_stride: usize,
    scratch: &mut [i32; 64 * 71],
) {
    let (ix, iy) = (x >> 4, y >> 4);
    let unscaled = x_step == 16 && y_step == 16 && w <= 64 && h <= 64;
    let inside = ix - 3 >= 0
        && ix + w as i32 + 4 <= last_x + 1
        && iy - 3 >= 0
        && iy + h as i32 + 4 <= last_y + 1;
    if unscaled && !inside {
        // The footprint crosses the plane edge: gather it with the
        // §8.5.2.4 Clip3( 0, lastX / lastY, · ) coordinate clamps applied
        // (edge replication — exactly the samples the per-tap clamped
        // reads return), then filter it like an interior block.
        let (fw, fh) = (w + 7, h + 7);
        let mut emu = [0i32; 71 * 71];
        for r in 0..fh {
            let ry = (iy + r as i32 - 3).clamp(0, last_y) as usize;
            let row = rows_of.row(ry);
            for (c, d) in emu[r * fw..(r + 1) * fw].iter_mut().enumerate() {
                *d = row[(ix + c as i32 - 3).clamp(0, last_x) as usize].get();
            }
        }
        let win = Window {
            data: &emu[..fw * fh],
            stride: fw,
            y0: 0,
        };
        let args = (
            3,
            3,
            (x & SUBPEL_MASK) as usize,
            (y & SUBPEL_MASK) as usize,
            w,
            h,
            interp_filter,
            bit_depth,
        );
        unscaled_kernel(&win, args, out, out_stride, scratch);
        return;
    }
    if !unscaled {
        let pred = block_inter_predict_plane(
            rows_of,
            x,
            y,
            x_step,
            y_step,
            w,
            h,
            interp_filter,
            last_x,
            last_y,
            bit_depth,
        );
        for (r, src) in pred.chunks_exact(w).enumerate() {
            out[r * out_stride..r * out_stride + w].copy_from_slice(src);
        }
        return;
    }
    let (ix, iy) = (ix as usize, iy as usize);
    let fx = (x & SUBPEL_MASK) as usize;
    let fy = (y & SUBPEL_MASK) as usize;
    // The rows the footprint reads, as one flat window when the source
    // stores them contiguously (the common case).
    let (y0, y1) = if fy == 0 {
        (iy, iy + h)
    } else {
        (iy - 3, iy + h + 4)
    };
    let args = (ix, iy, fx, fy, w, h, interp_filter, bit_depth);
    match rows_of.window(y0, y1) {
        Some((data, stride)) => {
            let win = Window { data, stride, y0 };
            unscaled_kernel(&win, args, out, out_stride, scratch);
        }
        None => unscaled_kernel(rows_of, args, out, out_stride, scratch),
    }
}

/// Rows `y0..` of a plane stored contiguously (`stride` apart).
struct Window<'a, P> {
    data: &'a [P],
    stride: usize,
    y0: usize,
}

impl<P: crate::frame_buf::Px> SampleRows for Window<'_, P> {
    type Px = P;

    #[inline]
    fn row(&self, y: usize) -> &[P] {
        let at = (y - self.y0) * self.stride;
        &self.data[at..at + self.stride]
    }
}

/// The unscaled, in-plane §8.5.2.4 filter of
/// [`block_inter_predict_into`].
#[inline(always)]
fn unscaled_kernel<R: SampleRows>(
    rows_of: &R,
    (ix, iy, fx, fy, w, h, interp_filter, bit_depth): (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        u32,
    ),
    out: &mut [i32],
    out_stride: usize,
    scratch: &mut [i32; 64 * 71],
) {
    let max = (1i32 << bit_depth) - 1;
    let filter = &SUBPEL_FILTERS[interp_filter];
    let dst_row = |out: &mut [i32], r: usize| -> std::ops::Range<usize> {
        let _ = out;
        r * out_stride..r * out_stride + w
    };

    if fx == 0 && fy == 0 {
        for r in 0..h {
            let d = dst_row(out, r);
            for (o, &v) in out[d].iter_mut().zip(&rows_of.row(iy + r)[ix..ix + w]) {
                *o = v.get();
            }
        }
        return;
    }

    let hfilt = |row: &[R::Px], dst: &mut [i32]| {
        let taps = &filter[fx];
        for (c, d) in dst.iter_mut().enumerate() {
            let win = &row[c..c + 8];
            let mut s = 0i32;
            for k in 0..8 {
                s += taps[k] * win[k].get();
            }
            *d = ((s + 64) >> 7).clamp(0, max);
        }
    };

    if fy == 0 {
        // Horizontal only: the rows the vertical pass would copy.
        for r in 0..h {
            let d = dst_row(out, r);
            hfilt(&rows_of.row(iy + r)[ix - 3..ix + w + 4], &mut out[d]);
        }
        return;
    }

    // Intermediate rows iy - 3 .. iy + h + 4 (h + 7 rows).
    let rows = h + 7;
    let inter = &mut scratch[..rows * w];
    if fx == 0 {
        for (r, dst) in inter.chunks_exact_mut(w).enumerate() {
            for (o, &v) in dst.iter_mut().zip(&rows_of.row(iy + r - 3)[ix..ix + w]) {
                *o = v.get();
            }
        }
    } else {
        for (r, dst) in inter.chunks_exact_mut(w).enumerate() {
            hfilt(&rows_of.row(iy + r - 3)[ix - 3..ix + w + 4], dst);
        }
    }
    let taps = &filter[fy];
    for r in 0..h {
        let src = &inter[r * w..(r + 8) * w];
        let d = dst_row(out, r);
        for (c, o) in out[d].iter_mut().enumerate() {
            let mut s = 0i32;
            for k in 0..8 {
                s += taps[k] * src[k * w + c];
            }
            *o = ((s + 64) >> 7).clamp(0, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plane form equals the generic form sample for sample across
    /// every kernel, scaled and unscaled steps, and blocks straddling
    /// every plane edge (the clipped-window fallback) — a hash-texture
    /// reference, every phase exercised.
    #[test]
    fn plane_form_matches_generic_form_everywhere() {
        let (pw, ph) = (40usize, 28usize);
        let samples: Vec<i32> = (0..pw * ph)
            .map(|i| ((i as u64).wrapping_mul(2_654_435_761) >> 20) as i32 & 0xff)
            .collect();
        let (last_x, last_y) = (pw as i32 - 1, ph as i32 - 1);
        for filt in 0..4usize {
            for (xs, ys) in [(16, 16), (32, 16), (16, 24), (20, 20)] {
                for (w, h) in [(4usize, 4usize), (8, 4), (16, 16)] {
                    for x in [-70, -13, 0, 5, 51, 77, 200, 333, 500, 640] {
                        for y in [-40, -9, 0, 7, 51, 98, 150, 300, 430] {
                            let a = block_inter_predict(
                                |r, c| samples[r as usize * pw + c as usize],
                                x,
                                y,
                                xs,
                                ys,
                                w,
                                h,
                                filt,
                                last_x,
                                last_y,
                                8,
                            );
                            let rows = FlatRows {
                                samples: &samples,
                                stride: pw,
                            };
                            let b = block_inter_predict_plane(
                                &rows, x, y, xs, ys, w, h, filt, last_x, last_y, 8,
                            );
                            assert_eq!(a, b, "filter {filt} step ({xs},{ys}) {w}x{h} at ({x},{y})");
                            let mut c = vec![0i32; w * h];
                            let mut scratch = Box::new([0i32; 64 * 71]);
                            block_inter_predict_into(
                                &rows,
                                x,
                                y,
                                xs,
                                ys,
                                w,
                                h,
                                filt,
                                last_x,
                                last_y,
                                8,
                                &mut c,
                                w,
                                &mut scratch,
                            );
                            assert_eq!(a, c, "into: filter {filt} ({xs},{ys}) {w}x{h} ({x},{y})");
                        }
                    }
                }
            }
        }
    }

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
