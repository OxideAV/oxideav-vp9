//! §8.10 reference frame-buffer state — the `FrameStore[ ]` slots and
//! the per-slot `RefFrameWidth` / `RefFrameHeight` / `RefSubsamplingX` /
//! `RefSubsamplingY` / `RefBitDepth` geometry a VP9 decoder threads
//! between frames so inter frames can sample their references.
//!
//! VP9 keeps `NUM_REF_FRAMES = 8` reference slots. Each decoded frame
//! may refresh any subset of those slots (the §6.2 `refresh_frame_flags`
//! bit-mask); an inter frame names which slots its `LAST` / `GOLDEN` /
//! `ALTREF` reference lists draw from via the §6.2 `ref_frame_idx[ ]`
//! (3 entries). The §8.5.2.3 / §8.5.2.4 prediction steps resolve a
//! per-block `ref_frame[ refList ]` to a slot index with
//! `ref_frame_idx[ ref_frame[ refList ] - LAST_FRAME ]`, then read
//! `FrameStore[ refIdx ]` and `RefFrameWidth[ refIdx ]` /
//! `RefFrameHeight[ refIdx ]`.
//!
//! This module owns:
//!
//! * [`RefBuffers`] — the eight `FrameStore` slots plus their geometry.
//! * [`RefBuffers::update`] — the §8.10 reference frame update process:
//!   for each slot whose `refresh_frame_flags` bit is set, copy the
//!   current frame's planes + record `FrameWidth` / `FrameHeight` /
//!   sub-sampling / bit-depth into that slot.
//! * [`resolve_ref_idx`] — the §8.5.2.3 / §8.5.2.4
//!   `ref_frame_idx[ ref_frame[ refList ] - LAST_FRAME ]` slot
//!   resolution.
//!
//! The §6.2 `ref_frame_sign_bias[ ]` array is frame-local header state
//! (re-read each inter frame), not slot state, so it lives on the frame
//! header rather than here; [`resolve_ref_idx`] is the only piece of the
//! §8.5.2 refIdx machinery that belongs to the persistent buffer set.
//!
//! Single source of truth: `docs/video/vp9/vp9-spec.txt` §8.10 (lines
//! 5932-5953), §6.2.5 (`frame_size_with_refs`), §8.5.2.3 line 4635.

// The persistent buffer set is consumed by the inter-frame sequence
// decoder landing on top; until that path is wired the production
// readers of several accessors do not exist yet, so the crate-internal
// `dead_code` lint is silenced module-wide.
#![allow(dead_code)]

use crate::mode_info::LAST_FRAME;

/// `NUM_REF_FRAMES = 8` per §3 (`vp9-spec.txt` line 472). The number of
/// reference-frame slots a VP9 decoder maintains; `refresh_frame_flags`
/// is an 8-bit mask over these slots.
pub(crate) const NUM_REF_FRAMES: usize = 8;

/// `REFS_PER_FRAME = 3` per §3 (`vp9-spec.txt` line 474). The number of
/// reference frames an inter frame may use (the `LAST` / `GOLDEN` /
/// `ALTREF` lists), sizing `ref_frame_idx[ ]`.
pub(crate) const REFS_PER_FRAME: usize = 3;

/// One reference-frame slot's stored planes + geometry.
///
/// `planes[ 0 ]` is luma (`width × height`), `planes[ 1 ]` / `planes[ 2 ]`
/// are the sub-sampled chroma planes. The geometry mirrors the §8.10
/// per-slot records (`RefFrameWidth` / `RefFrameHeight` /
/// `RefSubsamplingX` / `RefSubsamplingY` / `RefBitDepth`). A freshly
/// allocated slot is empty (`width == 0`); a valid stream never samples
/// an uninitialised slot (§7.4 conformance: `ref_frame_idx` must name a
/// slot a prior frame refreshed).
#[derive(Clone, Debug, Default)]
pub(crate) struct RefSlot {
    /// `RefFrameWidth[ i ]` — slot frame width in luma samples.
    pub frame_width: u32,
    /// `RefFrameHeight[ i ]` — slot frame height in luma samples.
    pub frame_height: u32,
    /// `RefSubsamplingX[ i ]`.
    pub subsampling_x: bool,
    /// `RefSubsamplingY[ i ]`.
    pub subsampling_y: bool,
    /// `RefBitDepth[ i ]`.
    pub bit_depth: u8,
    /// `FrameStore[ i ][ 0 ]` luma plane, row-major, stride = luma
    /// width.
    pub y: Vec<i32>,
    /// `FrameStore[ i ][ 1 ]` U chroma plane, row-major, stride = chroma
    /// width.
    pub u: Vec<i32>,
    /// `FrameStore[ i ][ 2 ]` V chroma plane, row-major.
    pub v: Vec<i32>,
}

impl RefSlot {
    /// Chroma plane width (`(FrameWidth + subsampling_x) >>
    /// subsampling_x`) per §8.10.
    pub fn chroma_width(&self) -> usize {
        ((self.frame_width + u32::from(self.subsampling_x)) >> u32::from(self.subsampling_x))
            as usize
    }

    /// Chroma plane height (`(FrameHeight + subsampling_y) >>
    /// subsampling_y`) per §8.10.
    pub fn chroma_height(&self) -> usize {
        ((self.frame_height + u32::from(self.subsampling_y)) >> u32::from(self.subsampling_y))
            as usize
    }
}

/// The eight §8.10 `FrameStore[ ]` reference slots.
#[derive(Clone, Debug, Default)]
pub(crate) struct RefBuffers {
    slots: [RefSlot; NUM_REF_FRAMES],
}

/// The current-frame planes the §8.10 update copies into the refreshed
/// slots. Each plane is row-major; `y_stride` / `uv_stride` are the
/// working strides (the working planes may be MI-aligned wider than the
/// visible extent, so the update copies the visible `FrameWidth` /
/// chroma extents out of the working buffer).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CurrFramePlanes<'a> {
    pub y: &'a [i32],
    pub y_stride: usize,
    pub u: &'a [i32],
    pub v: &'a [i32],
    pub uv_stride: usize,
    pub frame_width: u32,
    pub frame_height: u32,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub bit_depth: u8,
}

impl RefBuffers {
    /// Allocate an empty reference-buffer set (all eight slots empty).
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow slot `idx` (`0..NUM_REF_FRAMES`).
    pub fn slot(&self, idx: usize) -> &RefSlot {
        &self.slots[idx]
    }

    /// `RefFrameWidth[ idx ]` per §8.10 / §6.2.5.
    pub fn ref_frame_width(&self, idx: usize) -> u32 {
        self.slots[idx].frame_width
    }

    /// `RefFrameHeight[ idx ]` per §8.10 / §6.2.5.
    pub fn ref_frame_height(&self, idx: usize) -> u32 {
        self.slots[idx].frame_height
    }

    /// `true` if slot `idx` has been refreshed by a prior frame (i.e.
    /// is samplable). §7.4 conformance requires every `ref_frame_idx`
    /// to name such a slot.
    pub fn is_initialised(&self, idx: usize) -> bool {
        self.slots[idx].frame_width != 0
    }

    /// §8.10 reference frame update process (`vp9-spec.txt` lines
    /// 5938-5949).
    ///
    /// For each slot `i` whose `refresh_frame_flags` bit is set, copy the
    /// current frame's visible planes into `FrameStore[ i ]` and record
    /// `RefFrameWidth[ i ]` / `RefFrameHeight[ i ]` / the sub-sampling /
    /// bit-depth. Slots whose bit is clear are left untouched (they keep
    /// the frame a previous update stored).
    pub fn update(&mut self, refresh_frame_flags: u8, curr: &CurrFramePlanes<'_>) {
        let cw = ((curr.frame_width + u32::from(curr.subsampling_x))
            >> u32::from(curr.subsampling_x)) as usize;
        let ch = ((curr.frame_height + u32::from(curr.subsampling_y))
            >> u32::from(curr.subsampling_y)) as usize;
        let lw = curr.frame_width as usize;
        let lh = curr.frame_height as usize;

        // Crop the working planes to the visible extents once; the
        // refreshed slots all receive the same cropped copy.
        let crop = |src: &[i32], stride: usize, w: usize, h: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(w * h);
            for row in 0..h {
                let base = row * stride;
                out.extend_from_slice(&src[base..base + w]);
            }
            out
        };

        for i in 0..NUM_REF_FRAMES {
            if (refresh_frame_flags >> i) & 1 == 0 {
                continue;
            }
            let slot = &mut self.slots[i];
            slot.frame_width = curr.frame_width;
            slot.frame_height = curr.frame_height;
            slot.subsampling_x = curr.subsampling_x;
            slot.subsampling_y = curr.subsampling_y;
            slot.bit_depth = curr.bit_depth;
            slot.y = crop(curr.y, curr.y_stride, lw, lh);
            slot.u = crop(curr.u, curr.uv_stride, cw, ch);
            slot.v = crop(curr.v, curr.uv_stride, cw, ch);
        }
    }
}

/// §8.5.2.3 / §8.5.2.4 `refIdx = ref_frame_idx[ ref_frame[ refList ] -
/// LAST_FRAME ]` (`vp9-spec.txt` lines 4635 / 4692).
///
/// `ref_frame` is the per-block `ref_frame[ refList ]` value (one of
/// `LAST_FRAME = 1` / `GOLDEN_FRAME = 2` / `ALTREF_FRAME = 3`);
/// `ref_frame_idx` is the frame's three-entry slot map from §6.2.
/// Returns the `FrameStore` slot index. Panics on an out-of-range
/// `ref_frame` (the caller only invokes this for inter `refList`s whose
/// `ref_frame` is `>= LAST_FRAME`).
#[inline]
pub(crate) fn resolve_ref_idx(ref_frame: i32, ref_frame_idx: &[u8; REFS_PER_FRAME]) -> usize {
    let i = (ref_frame - LAST_FRAME) as usize;
    ref_frame_idx[i] as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode_info::{ALTREF_FRAME, GOLDEN_FRAME};

    fn make_curr(w: u32, h: u32, fill_y: i32, fill_c: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let cw = ((w + 1) >> 1) as usize;
        let ch = ((h + 1) >> 1) as usize;
        (
            vec![fill_y; (w * h) as usize],
            vec![fill_c; cw * ch],
            vec![fill_c + 1; cw * ch],
        )
    }

    /// §8.10: a fresh buffer set has every slot empty / uninitialised.
    #[test]
    fn fresh_buffers_are_uninitialised() {
        let bufs = RefBuffers::new();
        for i in 0..NUM_REF_FRAMES {
            assert!(!bufs.is_initialised(i));
            assert_eq!(bufs.ref_frame_width(i), 0);
        }
    }

    /// §8.10 step 1: a keyframe's `refresh_frame_flags = 0xFF` writes
    /// the current frame into every slot with the correct geometry.
    #[test]
    fn keyframe_refresh_fills_every_slot() {
        let mut bufs = RefBuffers::new();
        let (y, u, v) = make_curr(64, 64, 42, 7);
        let curr = CurrFramePlanes {
            y: &y,
            y_stride: 64,
            u: &u,
            v: &v,
            uv_stride: 32,
            frame_width: 64,
            frame_height: 64,
            subsampling_x: true,
            subsampling_y: true,
            bit_depth: 8,
        };
        bufs.update(0xFF, &curr);
        for i in 0..NUM_REF_FRAMES {
            assert!(bufs.is_initialised(i));
            assert_eq!(bufs.ref_frame_width(i), 64);
            assert_eq!(bufs.ref_frame_height(i), 64);
            let slot = bufs.slot(i);
            assert_eq!(slot.y.len(), 64 * 64);
            assert_eq!(slot.y[0], 42);
            assert_eq!(slot.u[0], 7);
            assert_eq!(slot.v[0], 8);
            assert_eq!(slot.chroma_width(), 32);
            assert_eq!(slot.chroma_height(), 32);
        }
    }

    /// §8.10 step 1: a `refresh_frame_flags = 0x01` (LAST only) inter
    /// frame refreshes slot 0 and leaves the others as the keyframe
    /// stored them.
    #[test]
    fn partial_refresh_only_touches_masked_slots() {
        let mut bufs = RefBuffers::new();
        // Keyframe fills all slots with luma 10.
        let (y0, u0, v0) = make_curr(64, 64, 10, 1);
        bufs.update(
            0xFF,
            &CurrFramePlanes {
                y: &y0,
                y_stride: 64,
                u: &u0,
                v: &v0,
                uv_stride: 32,
                frame_width: 64,
                frame_height: 64,
                subsampling_x: true,
                subsampling_y: true,
                bit_depth: 8,
            },
        );
        // P-frame refreshes only slot 0 with luma 99.
        let (y1, u1, v1) = make_curr(64, 64, 99, 2);
        bufs.update(
            0x01,
            &CurrFramePlanes {
                y: &y1,
                y_stride: 64,
                u: &u1,
                v: &v1,
                uv_stride: 32,
                frame_width: 64,
                frame_height: 64,
                subsampling_x: true,
                subsampling_y: true,
                bit_depth: 8,
            },
        );
        assert_eq!(bufs.slot(0).y[0], 99); // refreshed
        for i in 1..NUM_REF_FRAMES {
            assert_eq!(bufs.slot(i).y[0], 10, "slot {i} should be untouched");
        }
    }

    /// §8.10: the update crops an MI-aligned working plane (stride wider
    /// than FrameWidth) down to the visible extent.
    #[test]
    fn update_crops_mi_aligned_working_plane() {
        let mut bufs = RefBuffers::new();
        // Working luma is 16-wide stride but the frame is only 10 wide.
        let stride = 16usize;
        let lw = 10u32;
        let lh = 4u32;
        let mut y = vec![0i32; stride * lh as usize];
        for row in 0..lh as usize {
            for col in 0..stride {
                y[row * stride + col] = (row * 100 + col) as i32;
            }
        }
        let cw = ((lw + 1) >> 1) as usize;
        let ch = ((lh + 1) >> 1) as usize;
        let u = vec![5i32; 8 * ch]; // chroma stride 8
        let v = vec![6i32; 8 * ch];
        bufs.update(
            0x02,
            &CurrFramePlanes {
                y: &y,
                y_stride: stride,
                u: &u,
                v: &v,
                uv_stride: 8,
                frame_width: lw,
                frame_height: lh,
                subsampling_x: true,
                subsampling_y: true,
                bit_depth: 8,
            },
        );
        let slot = bufs.slot(1);
        assert_eq!(slot.y.len(), (lw * lh) as usize);
        // Row 0 of the cropped plane is cols 0..9 of the working plane.
        for col in 0..lw as usize {
            assert_eq!(slot.y[col], col as i32);
        }
        // Row 1, col 0 is working (1,0) = 100, not the stride-tail of row 0.
        assert_eq!(slot.y[lw as usize], 100);
        assert_eq!(slot.u.len(), cw * ch);
    }

    /// §8.5.2.3 / §8.5.2.4: `resolve_ref_idx` maps a per-block
    /// `ref_frame[ refList ]` through `ref_frame_idx[ ]`.
    #[test]
    fn resolve_ref_idx_maps_through_ref_frame_idx() {
        // ref_frame_idx maps LAST->slot 0, GOLDEN->slot 1, ALTREF->slot 2.
        let map = [0u8, 1, 2];
        assert_eq!(resolve_ref_idx(LAST_FRAME, &map), 0);
        assert_eq!(resolve_ref_idx(GOLDEN_FRAME, &map), 1);
        assert_eq!(resolve_ref_idx(ALTREF_FRAME, &map), 2);
        // A different slot map (LAST->3, GOLDEN->5, ALTREF->1).
        let map2 = [3u8, 5, 1];
        assert_eq!(resolve_ref_idx(LAST_FRAME, &map2), 3);
        assert_eq!(resolve_ref_idx(GOLDEN_FRAME, &map2), 5);
        assert_eq!(resolve_ref_idx(ALTREF_FRAME, &map2), 1);
    }
}
