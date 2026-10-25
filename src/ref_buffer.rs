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
//!   for each slot whose `refresh_frame_flags` bit is set, store the
//!   current frame (its planes + `FrameWidth` / `FrameHeight` /
//!   sub-sampling / bit-depth) into that slot. A decoded frame is
//!   immutable, so the slots share it (`Arc`) instead of copying it.
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

use std::sync::Arc;

use crate::frame_buf::FrameBuf;
use crate::mode_info::LAST_FRAME;

/// `NUM_REF_FRAMES` from §3: the number of `FrameStore` slots.
pub(crate) const NUM_REF_FRAMES: usize = 8;

/// `REFS_PER_FRAME` from §3: the number of reference lists an inter
/// frame names (`LAST`, `GOLDEN`, `ALTREF`).
pub(crate) const REFS_PER_FRAME: usize = 3;

/// One §8.10 `FrameStore[ i ]` slot: the stored frame (planes + the
/// `RefFrameWidth` / `RefFrameHeight` / `RefSubsamplingX` /
/// `RefSubsamplingY` / `RefBitDepth` geometry), `None` until a frame
/// refreshes it.
#[derive(Clone, Debug, Default)]
pub(crate) struct RefSlot {
    pub frame: Option<Arc<FrameBuf>>,
}

impl RefSlot {
    /// `RefFrameWidth[ i ]` (0 for an empty slot).
    pub fn frame_width(&self) -> u32 {
        self.frame.as_ref().map_or(0, |f| f.frame_width)
    }

    /// `RefFrameHeight[ i ]` (0 for an empty slot).
    pub fn frame_height(&self) -> u32 {
        self.frame.as_ref().map_or(0, |f| f.frame_height)
    }

    /// Sample `(x, y)` of plane `p` (tests).
    pub fn sample(&self, p: usize, x: usize, y: usize) -> i32 {
        i32::from(self.frame.as_ref().expect("initialised slot").row(p, y)[x])
    }

    /// The stored frame cropped to its §8.10 visible extents.
    pub fn to_decoded_frame(&self) -> crate::Vp9DecodedFrame {
        self.frame
            .as_ref()
            .expect("initialised slot")
            .to_decoded_frame()
    }

    /// Visible chroma width per §8.10.
    pub fn chroma_width(&self) -> usize {
        self.frame.as_ref().map_or(0, |f| f.chroma_dims().0)
    }

    /// Visible chroma height per §8.10.
    pub fn chroma_height(&self) -> usize {
        self.frame.as_ref().map_or(0, |f| f.chroma_dims().1)
    }
}

/// The eight §8.10 `FrameStore[ ]` slots.
#[derive(Clone, Debug, Default)]
pub(crate) struct RefBuffers {
    slots: [RefSlot; NUM_REF_FRAMES],
}

/// A flat view of a just-decoded frame's working planes (tests and
/// encoder-mirror helpers; the decoder stores its [`FrameBuf`]
/// directly through [`RefBuffers::update`]).
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn slot(&self, idx: usize) -> &RefSlot {
        &self.slots[idx]
    }

    pub fn ref_frame_width(&self, idx: usize) -> u32 {
        self.slots[idx].frame_width()
    }

    pub fn ref_frame_height(&self, idx: usize) -> u32 {
        self.slots[idx].frame_height()
    }

    /// Whether slot `idx` has ever been written.
    pub fn is_initialised(&self, idx: usize) -> bool {
        self.slots[idx].frame.is_some()
    }

    /// §8.10 reference frame update: every slot whose
    /// `refresh_frame_flags` bit is set now holds `frame`.
    pub fn update(&mut self, refresh_frame_flags: u8, frame: &Arc<FrameBuf>) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if (refresh_frame_flags >> i) & 1 != 0 {
                slot.frame = Some(Arc::clone(frame));
            }
        }
    }

    /// [`Self::update`] from flat planes (copied into a [`FrameBuf`]).
    pub fn update_planes(&mut self, refresh_frame_flags: u8, curr: &CurrFramePlanes<'_>) {
        let frame = Arc::new(FrameBuf::from_flat(
            [
                (curr.y, curr.y_stride),
                (curr.u, curr.uv_stride),
                (curr.v, curr.uv_stride),
            ],
            curr.frame_width,
            curr.frame_height,
            curr.subsampling_x,
            curr.subsampling_y,
            curr.bit_depth,
        ));
        self.update(refresh_frame_flags, &frame);
    }
}

/// §8.5.2.3 / §8.5.2.4 slot resolution: `refIdx = ref_frame_idx[
/// ref_frame[ refList ] - LAST_FRAME ]`.
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
        bufs.update_planes(0xFF, &curr);
        for i in 0..NUM_REF_FRAMES {
            assert!(bufs.is_initialised(i));
            assert_eq!(bufs.ref_frame_width(i), 64);
            assert_eq!(bufs.ref_frame_height(i), 64);
            let slot = bufs.slot(i);
            assert_eq!(slot.sample(0, 0, 0), 42);
            assert_eq!(slot.sample(0, 63, 63), 42);
            assert_eq!(slot.sample(1, 0, 0), 7);
            assert_eq!(slot.sample(2, 0, 0), 8);
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
        bufs.update_planes(
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
        bufs.update_planes(
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
        assert_eq!(bufs.slot(0).sample(0, 0, 0), 99); // refreshed
        for i in 1..NUM_REF_FRAMES {
            assert_eq!(
                bufs.slot(i).sample(0, 0, 0),
                10,
                "slot {i} should be untouched"
            );
        }
    }

    /// §8.10: the stored frame keeps the MI-aligned working plane and
    /// its visible extent; the output crop drops the stride tail.
    #[test]
    fn update_keeps_mi_aligned_plane_and_visible_crop() {
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
        bufs.update_planes(
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
        assert_eq!(slot.sample(0, 15, 1), 115);
        let out = slot.to_decoded_frame();
        assert_eq!(out.y.len(), (lw * lh) as usize);
        // Row 0 of the cropped plane is cols 0..9 of the working plane.
        for col in 0..lw as usize {
            assert_eq!(out.y[col], col as u16);
        }
        // Row 1, col 0 is working (1,0) = 100, not the stride-tail of row 0.
        assert_eq!(out.y[lw as usize], 100);
        assert_eq!(out.u.len(), cw * ch);
        assert_eq!(slot.chroma_width(), cw);
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
