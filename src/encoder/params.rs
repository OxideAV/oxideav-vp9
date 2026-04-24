//! Encoder input parameters.
//!
//! Minimal parameter set needed by the keyframe-only MVP encoder. The
//! defaults match the scope in the crate README: profile 0, 4:2:0 8-bit,
//! single tile, loop filter off, fixed quantisation.

/// Encoder parameters for a single keyframe.
#[derive(Clone, Copy, Debug)]
pub struct EncoderParams {
    /// Frame width in pixels. Must be >= 1 and <= 65536.
    pub width: u32,
    /// Frame height in pixels. Must be >= 1 and <= 65536.
    pub height: u32,
    /// `base_q_idx` per §6.2.4. 0 = lossless mode; typical visually-lossy
    /// values sit in 60..=120.
    pub base_q_idx: u8,
    /// Loop filter level (0..=63) per §6.2.3. MVP uses 0 (disabled).
    pub loop_filter_level: u8,
}

impl EncoderParams {
    /// Sensible defaults for a keyframe of `width × height`.
    pub fn keyframe(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            base_q_idx: 64,
            loop_filter_level: 0,
        }
    }
}

/// Source planes for a 4:2:0 8-bit YUV keyframe. All planes are
/// row-major; strides may exceed the plane width. The MVP encoder
/// ignores this payload (it emits a constant-midgrey keyframe) but
/// the type is here so callers can hand us their source now and the
/// future forward-transform path can pick it up without an API break.
#[derive(Clone, Debug)]
pub struct YuvFrame<'a> {
    pub y: &'a [u8],
    pub y_stride: usize,
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub uv_stride: usize,
    pub width: u32,
    pub height: u32,
}
