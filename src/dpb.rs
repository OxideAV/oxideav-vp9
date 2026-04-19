//! VP9 reference-frame buffer ("DPB") — §6.2.
//!
//! VP9 keeps 8 reference-frame slots. Every decoded frame owns a
//! `refresh_frame_flags` bitmap indicating which of the 8 slots it
//! writes into. Later inter frames select up to 3 of these 8 slots as
//! their `LAST_FRAME`, `GOLDEN_FRAME` and `ALTREF_FRAME` via
//! `ref_frame_idx[0..3]` (§6.2).
//!
//! This module stores only the bits we need to predict from: the three
//! reconstructed planes, the plane dimensions, and the chroma
//! subsampling the frame was encoded with. Decoded probability state
//! (§10.5) is not yet persisted across frames — the crate still reads
//! default probability tables each frame.

/// One decoded reference frame. Carries reconstructed Y / U / V planes
/// plus the geometry needed to interpolate them at sub-pel precision.
#[derive(Clone, Debug)]
pub struct RefFrame {
    /// Luma plane, row-major, stride = `y_stride`.
    pub y: Vec<u8>,
    pub y_stride: usize,
    /// Chroma planes (interleaved as two buffers), stride = `uv_stride`.
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub uv_stride: usize,
    /// Luma plane dimensions.
    pub width: usize,
    pub height: usize,
    /// Chroma plane dimensions (after subsampling).
    pub uv_width: usize,
    pub uv_height: usize,
    /// Subsampling: 1 if subsampled, 0 otherwise.
    pub subsampling_x: u8,
    pub subsampling_y: u8,
}

impl RefFrame {
    /// Read a luma sample with edge-clamped access (§8.5.4).
    pub fn sample_y(&self, row: isize, col: isize) -> u8 {
        let r = row.clamp(0, self.height as isize - 1) as usize;
        let c = col.clamp(0, self.width as isize - 1) as usize;
        self.y[r * self.y_stride + c]
    }

    /// Read a chroma sample from plane `plane` (1 = U, 2 = V) with
    /// edge-clamped access.
    pub fn sample_uv(&self, plane: u8, row: isize, col: isize) -> u8 {
        let r = row.clamp(0, self.uv_height as isize - 1) as usize;
        let c = col.clamp(0, self.uv_width as isize - 1) as usize;
        let buf = if plane == 1 { &self.u } else { &self.v };
        buf[r * self.uv_stride + c]
    }
}

/// 8-slot decoded-picture buffer. Index 0..=7 matches the VP9
/// `FrameSlots` state (§6.2).
#[derive(Clone, Debug, Default)]
pub struct Dpb {
    slots: [Option<RefFrame>; 8],
}

impl Dpb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `frame` into every slot whose bit is set in
    /// `refresh_frame_flags` (§6.2 last paragraph).
    pub fn refresh(&mut self, refresh_frame_flags: u8, frame: &RefFrame) {
        for i in 0..8 {
            if (refresh_frame_flags >> i) & 1 == 1 {
                self.slots[i] = Some(frame.clone());
            }
        }
    }

    /// Fetch a slot by index. Returns `None` when the slot was never
    /// written (the decoder must surface this as a malformed stream).
    pub fn get(&self, idx: u8) -> Option<&RefFrame> {
        self.slots.get(idx as usize).and_then(|s| s.as_ref())
    }

    /// Whether the given slot currently holds a frame.
    pub fn has(&self, idx: u8) -> bool {
        self.get(idx).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_frame() -> RefFrame {
        RefFrame {
            y: vec![42; 64 * 64],
            y_stride: 64,
            u: vec![77; 32 * 32],
            v: vec![88; 32 * 32],
            uv_stride: 32,
            width: 64,
            height: 64,
            uv_width: 32,
            uv_height: 32,
            subsampling_x: 1,
            subsampling_y: 1,
        }
    }

    #[test]
    fn refresh_writes_selected_slots() {
        let mut dpb = Dpb::new();
        let f = synth_frame();
        dpb.refresh(0b0000_0101, &f);
        assert!(dpb.has(0));
        assert!(!dpb.has(1));
        assert!(dpb.has(2));
        assert!(!dpb.has(3));
        assert!(!dpb.has(7));
    }

    #[test]
    fn edge_clamped_sample_access() {
        let f = synth_frame();
        // In-bounds reads hit the stored value.
        assert_eq!(f.sample_y(10, 10), 42);
        // Out-of-bounds reads clamp.
        assert_eq!(f.sample_y(-5, -5), 42);
        assert_eq!(f.sample_y(1000, 1000), 42);
        assert_eq!(f.sample_uv(1, 100, 100), 77);
        assert_eq!(f.sample_uv(2, -1, -1), 88);
    }
}
