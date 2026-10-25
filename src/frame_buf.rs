//! Superblock-row banded frame storage.
//!
//! A decoded frame's planes are stored as one buffer per superblock row
//! (a *band*: `64 >> subsampling_y` rows of the MI-aligned plane), each
//! preceded by [`HALO`] scratch rows. The decode pipeline hands each
//! band to exactly one job at a time — tile columns copy their
//! reconstructed rows in, the §8.8 loop filter of that superblock row
//! filters it in place (its top-edge stencils reach into the halo rows,
//! which carry the bottom rows of the band above and are written back
//! there once final) — so no two threads ever share a buffer. Once
//! decoded, the frame is immutable and shared (behind an `Arc`) by the
//! §8.10 reference slots and the output; [`BandPlane::row`] addresses it
//! by absolute row.

/// Scratch rows above each band: the reach of the widest §8.8.5
/// stencil above a superblock row's top edge.
pub(crate) const HALO: usize = 8;

/// One plane of a [`FrameBuf`]: `height` rows of `stride` samples
/// (the MI-aligned extent), split into bands of `1 << band_shift` rows.
#[derive(Debug, Clone)]
pub(crate) struct BandPlane {
    pub(crate) stride: usize,
    /// Rows in the plane (the last band may be short).
    #[allow(dead_code)]
    pub(crate) height: usize,
    pub(crate) band_shift: u32,
    /// `bands[ b ]`: `HALO` scratch rows, then the band's own rows.
    pub(crate) bands: Vec<Vec<i32>>,
}

impl BandPlane {
    /// Rows `[ b << band_shift, min( height, (b + 1) << band_shift ) )`.
    pub(crate) fn band_rows(height: usize, band_shift: u32, b: usize) -> usize {
        let start = b << band_shift;
        height.min(start + (1 << band_shift)) - start
    }

    /// An all-zero plane.
    pub(crate) fn new(stride: usize, height: usize, band_shift: u32) -> Self {
        let n = height.div_ceil(1 << band_shift);
        Self {
            stride,
            height,
            band_shift,
            bands: (0..n)
                .map(|b| vec![0; (HALO + Self::band_rows(height, band_shift, b)) * stride])
                .collect(),
        }
    }

    /// Row `y` (`< height`), `stride` samples.
    #[inline]
    pub(crate) fn row(&self, y: usize) -> &[i32] {
        let b = y >> self.band_shift;
        let at = (HALO + (y & ((1 << self.band_shift) - 1))) * self.stride;
        &self.bands[b][at..at + self.stride]
    }

    /// The `w x h` top-left crop as `u16` samples, row-major.
    pub(crate) fn crop_u16(&self, w: usize, h: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            out.extend(self.row(y)[..w].iter().map(|&v| v as u16));
        }
        out
    }

    /// Copy a flat `stride`-wide plane in (test / encoder-mirror use).
    pub(crate) fn from_flat(
        samples: &[i32],
        stride: usize,
        height: usize,
        band_shift: u32,
    ) -> Self {
        let mut p = Self::new(stride, height, band_shift);
        for (y, src) in samples.chunks_exact(stride).take(height).enumerate() {
            let b = y >> band_shift;
            let at = (HALO + (y & ((1 << band_shift) - 1))) * stride;
            p.bands[b][at..at + stride].copy_from_slice(src);
        }
        p
    }
}

/// A decoded frame: the three MI-aligned planes (post loop filter) and
/// the §7.2 frame geometry / colour format they carry.
#[derive(Debug, Clone)]
pub(crate) struct FrameBuf {
    pub(crate) planes: [BandPlane; 3],
    /// §7.2.5 `FrameWidth` / `FrameHeight` (the visible extent).
    pub(crate) frame_width: u32,
    pub(crate) frame_height: u32,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) bit_depth: u8,
}

impl FrameBuf {
    /// Visible chroma extent per §8.10.
    pub(crate) fn chroma_dims(&self) -> (usize, usize) {
        (
            ((self.frame_width + u32::from(self.subsampling_x)) >> u32::from(self.subsampling_x))
                as usize,
            ((self.frame_height + u32::from(self.subsampling_y)) >> u32::from(self.subsampling_y))
                as usize,
        )
    }

    /// The §8.10 output crop.
    pub(crate) fn to_decoded_frame(&self) -> crate::Vp9DecodedFrame {
        let (cw, ch) = self.chroma_dims();
        let (w, h) = (self.frame_width as usize, self.frame_height as usize);
        crate::Vp9DecodedFrame {
            width: self.frame_width,
            height: self.frame_height,
            bit_depth: self.bit_depth,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            y: self.planes[0].crop_u16(w, h),
            u: self.planes[1].crop_u16(cw, ch),
            v: self.planes[2].crop_u16(cw, ch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_address_across_bands() {
        let (stride, height) = (5usize, 70usize);
        let flat: Vec<i32> = (0..(stride * height) as i32).collect();
        let p = BandPlane::from_flat(&flat, stride, height, 5);
        assert_eq!(p.bands.len(), 3);
        for y in 0..height {
            assert_eq!(p.row(y), &flat[y * stride..(y + 1) * stride]);
        }
        assert_eq!(BandPlane::band_rows(height, 5, 2), 6);
        assert_eq!(p.crop_u16(3, 2), vec![0, 1, 2, 5, 6, 7]);
    }
}
