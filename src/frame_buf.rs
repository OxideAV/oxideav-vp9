//! Superblock-row banded frame storage with per-row completion.
//!
//! A decoded frame's planes are stored as one buffer per superblock row
//! (a *band*: `64 >> subsampling_y` rows of the MI-aligned plane, all
//! three planes together), each preceded by [`HALO`] scratch rows. While
//! a frame decodes, its bands live in the frame pipeline, each owned by
//! exactly one job at a time — tile columns copy their reconstructed
//! rows in, the §8.8 loop filter of that superblock row filters it in
//! place (its top-edge stencils reach into the halo rows, which carry the
//! bottom rows of the band above and are written back there once final).
//! When a band can no longer change it is *frozen* into the shared
//! [`FrameBuf`], in raster order; so is each superblock row's mode info.
//! A later frame that references this one (§8.5.2 inter prediction, the
//! §6.5 previous-frame motion vectors, the §6.4.14 previous segment ids)
//! reads frozen rows without locking and blocks only on rows not yet
//! frozen — which is what lets consecutive frames decode concurrently.

use std::sync::{Condvar, Mutex, OnceLock};

/// A stored sample: `i32` working planes, `u16` decoded frames (every
/// §7.2.2 bit depth fits).
pub(crate) trait Px: Copy + Default + Send + Sync + 'static {
    /// The sample as the §8 processes' integer.
    fn get(self) -> i32;
    /// Store a sample (`v` is in `0..1 << BitDepth`).
    fn put(v: i32) -> Self;
}

impl Px for i32 {
    #[inline(always)]
    fn get(self) -> i32 {
        self
    }
    #[inline(always)]
    fn put(v: i32) -> Self {
        v
    }
}

impl Px for u16 {
    #[inline(always)]
    fn get(self) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn put(v: i32) -> Self {
        v as u16
    }
}

/// Scratch rows above each band: the reach of the widest §8.8.5
/// stencil above a superblock row's top edge.
pub(crate) const HALO: usize = 8;

/// Rows `[ b << band_shift, min( height, (b + 1) << band_shift ) )`.
pub(crate) fn band_rows(height: usize, band_shift: u32, b: usize) -> usize {
    let start = b << band_shift;
    height.min(start + (1 << band_shift)) - start
}

/// One plane's layout: `height` rows of `stride` samples (the
/// MI-aligned extent) in bands of `1 << band_shift` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneGeom {
    pub(crate) stride: usize,
    pub(crate) height: usize,
    pub(crate) band_shift: u32,
}

impl PlaneGeom {
    /// Samples in band `b`'s buffer (halo + own rows).
    pub(crate) fn band_len(&self, b: usize) -> usize {
        (HALO + band_rows(self.height, self.band_shift, b)) * self.stride
    }
}

/// The three planes of one superblock row (halo + own rows each).
pub(crate) type Band = [Vec<u16>; 3];

/// One superblock row's mode info (`8 * MiCols` entries per array, MI
/// row-major from the row's first MI row): what the §8.8 loop filter
/// and the next frame's §6.5 / §6.4.14 reads need.
#[derive(Debug, Clone)]
pub(crate) struct MiRow {
    pub(crate) mi_sizes: Vec<u8>,
    pub(crate) tx_sizes: Vec<u8>,
    pub(crate) skips: Vec<bool>,
    pub(crate) ref0: Vec<i32>,
    pub(crate) ref1: Vec<i32>,
    pub(crate) mv0: Vec<(i16, i16)>,
    pub(crate) mv1: Vec<(i16, i16)>,
    pub(crate) y_modes: Vec<u8>,
    pub(crate) seg_ids: Vec<u8>,
}

impl MiRow {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            mi_sizes: vec![0; n],
            tx_sizes: vec![0; n],
            skips: vec![false; n],
            ref0: vec![0; n],
            ref1: vec![0; n],
            mv0: vec![(0, 0); n],
            mv1: vec![(0, 0); n],
            y_modes: vec![0; n],
            seg_ids: vec![0; n],
        }
    }
}

#[derive(Debug, Default)]
struct FrameSync {
    /// Leading bands / mode-info rows frozen so far.
    bands_final: usize,
    mi_final: usize,
    /// The frame's decode failed: its unfrozen rows never arrive.
    failed: bool,
}

/// A decoded (or decoding) frame: geometry, colour format, and the rows
/// frozen so far.
#[derive(Debug)]
pub(crate) struct FrameBuf {
    pub(crate) geom: [PlaneGeom; 3],
    /// §7.2.5 `FrameWidth` / `FrameHeight` (the visible extent).
    pub(crate) frame_width: u32,
    pub(crate) frame_height: u32,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) bit_depth: u8,
    bands: Vec<OnceLock<Band>>,
    mi: Vec<OnceLock<MiRow>>,
    sync: Mutex<FrameSync>,
    changed: Condvar,
    /// What a row of a failed frame reads as.
    zero_row: Vec<u16>,
}

impl FrameBuf {
    /// A frame with nothing frozen yet.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        geom: [PlaneGeom; 3],
        frame_width: u32,
        frame_height: u32,
        subsampling_x: bool,
        subsampling_y: bool,
        bit_depth: u8,
        mi_rows: u32,
    ) -> Self {
        let n_bands = geom[0].height.div_ceil(1 << geom[0].band_shift);
        let n_mi = mi_rows.div_ceil(8) as usize;
        Self {
            geom,
            frame_width,
            frame_height,
            subsampling_x,
            subsampling_y,
            bit_depth,
            bands: (0..n_bands).map(|_| OnceLock::new()).collect(),
            mi: (0..n_mi).map(|_| OnceLock::new()).collect(),
            sync: Mutex::new(FrameSync::default()),
            changed: Condvar::new(),
            zero_row: vec![0; geom[0].stride.max(geom[1].stride)],
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FrameSync> {
        self.sync.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Freeze band `b` (its contents are final).
    pub(crate) fn freeze_band(&self, b: usize, band: Band) {
        let _ = self.bands[b].set(band);
        let mut s = self.lock();
        while s.bands_final < self.bands.len() && self.bands[s.bands_final].get().is_some() {
            s.bands_final += 1;
        }
        self.changed.notify_all();
    }

    /// Freeze superblock row `r`'s mode info.
    pub(crate) fn freeze_mi(&self, r: usize, row: MiRow) {
        let _ = self.mi[r].set(row);
        let mut s = self.lock();
        while s.mi_final < self.mi.len() && self.mi[s.mi_final].get().is_some() {
            s.mi_final += 1;
        }
        self.changed.notify_all();
    }

    /// Mark the decode failed (releases every waiter).
    pub(crate) fn fail(&self) {
        self.lock().failed = true;
        self.changed.notify_all();
    }

    /// Block until band `b` is frozen; `false` if the frame failed.
    pub(crate) fn wait_band(&self, b: usize) -> bool {
        if self.bands[b].get().is_some() {
            return true;
        }
        let mut s = self.lock();
        while s.bands_final <= b && !s.failed {
            s = self.changed.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        self.bands[b].get().is_some()
    }

    /// Superblock row `r`'s mode info once frozen; `None` if the frame
    /// failed before it was.
    pub(crate) fn mi_row(&self, r: usize) -> Option<&MiRow> {
        if let Some(m) = self.mi[r].get() {
            return Some(m);
        }
        let mut s = self.lock();
        while s.mi_final <= r && !s.failed {
            s = self.changed.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        drop(s);
        self.mi[r].get()
    }

    /// Row `y` (`< height`) of plane `p`, `stride` samples, blocking
    /// until its band is frozen (zeros if the frame failed).
    #[inline]
    pub(crate) fn row(&self, p: usize, y: usize) -> &[u16] {
        let g = &self.geom[p];
        let b = y >> g.band_shift;
        let band = match self.bands[b].get() {
            Some(band) => band,
            None => {
                if !self.wait_band(b) {
                    return &self.zero_row[..g.stride];
                }
                self.bands[b].get().expect("frozen band")
            }
        };
        let at = (HALO + (y & ((1 << g.band_shift) - 1))) * g.stride;
        &band[p][at..at + g.stride]
    }

    /// Rows `y0..y1` of plane `p` as one flat slice when they share a
    /// band (blocking until it is frozen); `None` across a band edge or
    /// on a failed frame.
    #[inline]
    pub(crate) fn window(&self, p: usize, y0: usize, y1: usize) -> Option<(&[u16], usize)> {
        let g = &self.geom[p];
        let b = y0 >> g.band_shift;
        if (y1 - 1) >> g.band_shift != b {
            return None;
        }
        let band = match self.bands[b].get() {
            Some(band) => band,
            None => {
                if !self.wait_band(b) {
                    return None;
                }
                self.bands[b].get()?
            }
        };
        let mask = (1 << g.band_shift) - 1;
        let at = (HALO + (y0 & mask)) * g.stride;
        let end = (HALO + ((y1 - 1) & mask) + 1) * g.stride;
        Some((&band[p][at..end], g.stride))
    }

    /// Wait for every band; `false` if the frame failed.
    pub(crate) fn wait_complete(&self) -> bool {
        self.bands.is_empty() || self.wait_band(self.bands.len() - 1)
    }

    /// Take the band storage back (recycling a frame nobody references).
    pub(crate) fn into_bands(self) -> Vec<Option<Band>> {
        self.bands.into_iter().map(OnceLock::into_inner).collect()
    }

    /// Visible chroma extent per §8.10.
    pub(crate) fn chroma_dims(&self) -> (usize, usize) {
        (
            ((self.frame_width + u32::from(self.subsampling_x)) >> u32::from(self.subsampling_x))
                as usize,
            ((self.frame_height + u32::from(self.subsampling_y)) >> u32::from(self.subsampling_y))
                as usize,
        )
    }

    /// The `w x h` top-left crop of plane `p` as `u16` samples.
    pub(crate) fn crop_u16(&self, p: usize, w: usize, h: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            out.extend_from_slice(&self.row(p, y)[..w]);
        }
        out
    }

    /// The §8.10 output crop (waits for the frame).
    pub(crate) fn to_decoded_frame(&self) -> crate::Vp9DecodedFrame {
        let (cw, ch) = self.chroma_dims();
        let (w, h) = (self.frame_width as usize, self.frame_height as usize);
        crate::Vp9DecodedFrame {
            width: self.frame_width,
            height: self.frame_height,
            bit_depth: self.bit_depth,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            y: self.crop_u16(0, w, h),
            u: self.crop_u16(1, cw, ch),
            v: self.crop_u16(2, cw, ch),
        }
    }

    /// A complete frame built from flat planes (`stride`-wide rows,
    /// `planes[ p ].len() / stride` rows each) — tests and helpers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_flat(
        planes: [(&[i32], usize); 3],
        frame_width: u32,
        frame_height: u32,
        subsampling_x: bool,
        subsampling_y: bool,
        bit_depth: u8,
    ) -> Self {
        let ssy = u32::from(subsampling_y);
        let geom: [PlaneGeom; 3] = std::array::from_fn(|p| {
            let (s, stride) = planes[p];
            PlaneGeom {
                stride,
                height: s.len() / stride,
                band_shift: if p == 0 { 6 } else { 6 - ssy },
            }
        });
        let mi_rows = frame_height.div_ceil(8);
        let mi_cols = frame_width.div_ceil(8);
        let f = Self::new(
            geom,
            frame_width,
            frame_height,
            subsampling_x,
            subsampling_y,
            bit_depth,
            mi_rows,
        );
        for b in 0..f.bands.len() {
            let band: Band = std::array::from_fn(|p| {
                let g = geom[p];
                let mut buf = vec![0; g.band_len(b)];
                let rows = band_rows(g.height, g.band_shift, b);
                for k in 0..rows {
                    let y = (b << g.band_shift) + k;
                    let src = &planes[p].0[y * g.stride..(y + 1) * g.stride];
                    for (d, &v) in buf[(HALO + k) * g.stride..(HALO + k + 1) * g.stride]
                        .iter_mut()
                        .zip(src)
                    {
                        *d = v as u16;
                    }
                }
                buf
            });
            f.freeze_band(b, band);
        }
        for r in 0..f.mi.len() {
            f.freeze_mi(r, MiRow::new(8 * mi_cols as usize));
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_address_across_bands() {
        let (stride, height) = (5usize, 70usize);
        let flat: Vec<i32> = (0..(stride * height) as i32).collect();
        let empty = vec![0i32; stride * 35];
        let f = FrameBuf::from_flat(
            [(&flat, stride), (&empty, stride), (&empty, stride)],
            5,
            70,
            true,
            true,
            8,
        );
        assert_eq!(f.bands.len(), 2);
        for y in 0..height {
            let want: Vec<u16> = flat[y * stride..(y + 1) * stride]
                .iter()
                .map(|&v| v as u16)
                .collect();
            assert_eq!(f.row(0, y), &want[..]);
        }
        assert_eq!(band_rows(height, 5, 2), 6);
        assert_eq!(f.crop_u16(0, 3, 2), vec![0, 1, 2, 5, 6, 7]);
        assert!(f.wait_complete());
    }

    /// A reader blocked on an unfrozen band wakes when it is frozen, and
    /// a failed frame releases its readers with zero rows.
    #[test]
    fn readers_wait_for_frozen_rows() {
        let geom = [PlaneGeom {
            stride: 4,
            height: 128,
            band_shift: 6,
        }; 3];
        let f = std::sync::Arc::new(FrameBuf::new(geom, 4, 128, false, false, 8, 16));
        let reader = {
            let f = std::sync::Arc::clone(&f);
            std::thread::spawn(move || f.row(0, 70)[0])
        };
        let band = |v: u16| -> Band { std::array::from_fn(|_| vec![v; geom[0].band_len(0)]) };
        f.freeze_band(1, band(7));
        f.freeze_band(0, band(3));
        assert_eq!(reader.join().unwrap(), 7);

        let g = std::sync::Arc::new(FrameBuf::new(geom, 4, 128, false, false, 8, 16));
        let reader = {
            let g = std::sync::Arc::clone(&g);
            std::thread::spawn(move || (g.row(0, 100)[0], g.mi_row(1).is_none()))
        };
        g.fail();
        assert_eq!(reader.join().unwrap(), (0, true));
    }
}
