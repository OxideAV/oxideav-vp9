//! Pure-Rust VP9 video decoder — keyframe + single-reference inter.
//!
//! The decoder runs §6.2 uncompressed header + §6.3 compressed header +
//! §6.4 tile / superblock / partition walk + §6.4.23 coefficient decode +
//! §8.5.1 intra prediction (all 10 modes) + §8.7.1 inverse transforms
//! (4/8/16/32, DCT + ADST combos, 4×4 WHT) + clip-add reconstruction,
//! producing a `Yuv420P` `VideoFrame` from any 8-bit 4:2:0 VP9 keyframe.
//!
//! On top of that it keeps the 8-slot reference buffer (§6.2) and runs
//! single-reference inter prediction (§8.5 / §8.6) with full §6.4.19 MV
//! decode + §8.5.1 8-tap sub-pel interpolation, so single-reference P
//! frames (LAST / GOLDEN / ALTREF, NEARESTMV / NEARMV / ZEROMV / NEWMV)
//! decode into `VideoFrame`s.
//!
//! Deferred: compound prediction (§6.4.20), scaled references (§8.5.4),
//! multi-tile frames, segmentation-driven deltas, higher bit depths.
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
pub mod headers;
pub mod inter;
pub mod intra;
pub mod ivf;
pub mod loopfilter;
pub mod mcfilter;
pub mod mv;
pub mod probs;
pub mod reconintra;
pub mod tables;
pub mod tile;
pub mod transform;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

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
    QuantizationParams, RefFrame, SegmentationParams, TileInfo, UncompressedHeader,
};
