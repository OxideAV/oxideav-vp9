#![allow(clippy::derivable_impls)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::field_reassign_with_default)]
//! Pure-Rust VP9 video decoder — keyframes + inter + compound + multi-tile.
//!
//! The decoder runs §6.2 uncompressed header + §6.3 compressed header +
//! §6.4 tile / superblock / partition walk + §6.4.23 coefficient decode +
//! §8.5.1 intra prediction (all 10 modes) + §8.7.1 inverse transforms
//! (4/8/16/32, DCT + ADST combos, 4×4 WHT) + clip-add reconstruction,
//! producing a `Yuv420P` `VideoFrame` from any 8-bit 4:2:0 VP9 keyframe.
//!
//! On top of that it keeps the 8-slot reference buffer (§6.2) and runs
//! inter prediction (§8.5 / §8.6) with full §6.4.19 MV decode + §8.5.1
//! 8-tap sub-pel interpolation. Single-reference (LAST / GOLDEN /
//! ALTREF) and compound-reference (§6.4.17 / §8.5.2) modes both decode
//! into `VideoFrame`s, with Round2(a + b, 1) blending for compound.
//!
//! Scaled-reference motion compensation (§8.5.2.3) is supported: a
//! variable-step interpolator applies per-reference `x_step_q4` /
//! `y_step_q4` computed from `RefFrameWidth / FrameWidth`.
//!
//! Multi-tile frames (§6.4) split the tile payload at 4-byte
//! big-endian length prefixes, reset the boolean engine per tile, and
//! run the §8.8 loop filter once after all tiles are decoded.
//!
//! Segmentation deltas §8.6.1 `SEG_LVL_ALT_Q` and §8.8.1 `SEG_LVL_ALT_L`
//! apply through `SegmentationParams::get_qindex` and
//! `LvlLookup::build_with_segmentation`. Per-block segment_id decode
//! (§6.4.7 intra / §6.4.12 inter / §6.4.14 predicted) is wired through
//! the `segmentation` module, with `PrevSegmentIds` carried across
//! frames alongside the DPB and `AboveSegPredContext / LeftSegPredContext`
//! cleared per tile (§7.4.1 / §7.4.2).
//!
//! Deferred: higher bit depths (10/12-bit), neighbour-aware probability
//! contexts.
//!
//! Reference: VP9 Bitstream & Decoding Process Specification, version 0.7
//! (2017): <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>.

pub mod bitreader;
pub mod block;
pub mod bool_decoder;
pub mod compressed_header;
pub mod decoder;
pub mod detokenize;
pub mod dpb;
pub mod encoder;
pub mod frame_ctx;
pub mod headers;
pub mod inter;
pub mod intra;
pub mod ivf;
pub mod loopfilter;
pub mod mcfilter;
pub mod mv;
pub mod mvref;
pub mod nonzero_ctx;
pub mod probs;
pub mod reconintra;
pub mod segmentation;
pub mod tables;
pub mod tile;
pub mod transform;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry};

pub const CODEC_ID_STR: &str = "vp9";

/// Register the VP9 decoder with the codec registry. The implementation
/// reports `intra_only=false` (VP9 has inter prediction) and `lossy=true`.
/// The factory returns a decoder which will currently fail with
/// `Error::Unsupported` at frame-pull time — but parses headers and
/// populates parameters successfully.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("vp9_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    // AVI FourCC claims — `VP90` canonical, `VP9 ` trailing-space variant.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([CodecTag::fourcc(b"VP90"), CodecTag::fourcc(b"VP9 ")]),
    );
}

pub use compressed_header::{parse_compressed_header, CompressedHeader, ReferenceMode, TxMode};
pub use decoder::{
    codec_parameters_from_header, frame_rate_from_container, make_decoder,
    pixel_format_from_color_config, Vp9Decoder,
};
pub use headers::{
    parse_uncompressed_header, ColorConfig, ColorSpace, FrameType, LoopFilterParams,
    QuantizationParams, RefFrame, SegmentationParams, TileInfo, UncompressedHeader, SEG_LVL_ALT_L,
    SEG_LVL_ALT_Q, SEG_LVL_REF_FRAME, SEG_LVL_SKIP,
};
