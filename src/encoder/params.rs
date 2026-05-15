//! Encoder input parameters.
//!
//! Minimal parameter set needed by the keyframe-only MVP encoder. The
//! defaults match the scope in the crate README: profile 0, 4:2:0 8-bit,
//! single tile, fixed quantisation, QP-derived loop filter level.

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
    /// Loop filter level (0..=63) per §6.2.3. As of round 40 the default
    /// constructor derives this from `base_q_idx` via a libvpx-shape
    /// monotonic heuristic; setting it manually overrides the default.
    pub loop_filter_level: u8,
    /// Debug-only: force the inter encoder's 16×16 partition picker to
    /// always emit PARTITION_NONE (no HORZ / VERT / SPLIT at bsize=16).
    /// Used by regression tests to compare the r-next-8x8 encoder against
    /// the pre-this-round 16×16-NONE-only baseline on a SHARED loop
    /// filter / header / ME path. NOT a stable public API — gated to
    /// the regression-test target. Off-by-default; production encoder
    /// keeps the full {NONE, HORZ, VERT, SPLIT} RDO at bsize=16.
    pub debug_force_16x16_only: bool,
    /// Debug-only: force the inter encoder's 8×8 partition picker to
    /// always emit PARTITION_NONE (no HORZ B8x4 / VERT B4x8 / SPLIT B4x4
    /// sub-8×8 emission at bsize=8). Used by the r-next-sub8 regression
    /// test to compare full-shape RDO at 8×8 against the previous
    /// 8×8-NONE-only baseline on a SHARED loop filter / header / outer-
    /// partition path. NOT a stable public API. Off-by-default.
    pub debug_force_8x8_none_only: bool,
}

impl EncoderParams {
    /// Sensible defaults for a keyframe of `width × height`.
    ///
    /// Round 40: `loop_filter_level` is now derived from `base_q_idx`
    /// via [`Self::default_filter_level`] so encoded streams pick up
    /// the §8.8 deblocking pass at non-zero QP. `base_q_idx == 0`
    /// (lossless) keeps `loop_filter_level == 0` (filter disabled).
    pub fn keyframe(width: u32, height: u32) -> Self {
        let base_q_idx = 64;
        Self {
            width,
            height,
            base_q_idx,
            loop_filter_level: Self::default_filter_level(base_q_idx),
            debug_force_16x16_only: false,
            debug_force_8x8_none_only: false,
        }
    }

    /// Default loop-filter level for a given quantiser. Mirrors
    /// libvpx's "no fractional QP" path:
    ///   * Lossless (`q == 0`)            → level 0 (filter disabled).
    ///   * Visually-lossy (`q ∈ [1, 255]`) → level ≈ q * 0.45 + 1, clipped
    ///     to the §6.2.3 [0, 63] range.
    ///
    /// The slope is the libvpx default for INTER / KF intra frames at
    /// 8-bit. Higher Q implies coarser quantisation, which leaves bigger
    /// block edges to deblock.
    pub const fn default_filter_level(base_q_idx: u8) -> u8 {
        if base_q_idx == 0 {
            return 0;
        }
        // q * 0.45 + 1, integer-arithmetic. Scaled by 100: q*45 + 100.
        let lvl = (base_q_idx as u16 * 45 + 100) / 100;
        let clamped = if lvl > 63 { 63 } else { lvl };
        clamped as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_filter_level_lossless_returns_zero() {
        assert_eq!(EncoderParams::default_filter_level(0), 0);
    }

    #[test]
    fn default_filter_level_at_q64_matches_low_visual_quality() {
        // q=64 -> 64*45/100 + 1 = 28 + 1 = 29.8 → 29 (integer arith with rounding).
        let lvl = EncoderParams::default_filter_level(64);
        assert!((28..=30).contains(&lvl), "q=64 → lvl={lvl}");
    }

    #[test]
    fn default_filter_level_clamps_to_63_at_max_q() {
        let lvl = EncoderParams::default_filter_level(255);
        assert_eq!(lvl, 63);
    }

    #[test]
    fn keyframe_default_picks_nonzero_filter_at_q64() {
        let p = EncoderParams::keyframe(64, 64);
        assert!(
            p.loop_filter_level > 0,
            "round 40: keyframe defaults must enable deblocking at q=64"
        );
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

/// A reconstructed reference frame's YUV planes — produced by decoding
/// the previous I-frame (or P-frame) bitstream and fed to the P-frame
/// encoder as the LAST_FRAME reference for motion estimation /
/// compensation. Round 49 ships single-reference inter only, so
/// callers populate exactly one slot.
///
/// The encoder reads from `y`/`u`/`v` at decoder-shape stride; ME
/// runs against the luma plane in integer-pel units.
#[derive(Clone, Debug)]
pub struct ReferenceFrame {
    /// Reconstructed luma plane (row-major, length = `y_stride * height`).
    pub y: Vec<u8>,
    /// Luma stride in bytes.
    pub y_stride: usize,
    /// Reconstructed U plane (row-major).
    pub u: Vec<u8>,
    /// Reconstructed V plane.
    pub v: Vec<u8>,
    /// Chroma stride in bytes (shared between U and V).
    pub uv_stride: usize,
    /// Frame width (luma sample count per row).
    pub width: u32,
    /// Frame height (luma row count).
    pub height: u32,
}
