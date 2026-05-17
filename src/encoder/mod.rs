//! VP9 encoder — keyframe + P-frame scaffold.
//!
//! Reference: VP9 Bitstream & Decoding Process Specification v0.7.
//!
//! Keyframe path (rounds 1..48): assembles a valid VP9 keyframe step by
//! step — §6.2 uncompressed header, §6.3 compressed header, §6.4 tile /
//! partition walk, §6.4.3 per-block prediction/transform/quantise/
//! tokenise. Per-block luma and chroma intra-mode RDO (round 48) plus
//! mode-RDO early termination.
//!
//! P-frame path (round 49 + r-next + r-multiref): single- or multi-
//! reference inter encode. LAST_FRAME is mandatory; GOLDEN_FRAME is
//! optional and engaged per-CU via SAD-driven RDO when present.
//! Quadtree partitions descend to 8×8 with four-way RDO at every
//! interior level (sub-8×8 §6.4.16 (idy, idx) walk for B8x4 / B4x8 /
//! B4x4). Three-stage ME (integer + half-pel + quarter-pel) per the
//! §6.3 8-tap EightTap luma filter. `skip = 1` everywhere — PSNR
//! comes entirely from MC quality. See `inter.rs` for details.

pub mod bitwriter;
pub mod bool_encoder;
pub mod compressed_header;
pub mod frame;
pub mod fwdtransform;
pub mod inter;
pub mod params;
pub mod tile;
pub mod tile_pixel;
pub mod tokenize;
pub mod uncompressed_header;

pub use bitwriter::BitWriter;
pub use bool_encoder::BoolEncoder;
pub use compressed_header::{emit_compressed_header, emit_compressed_header_p};
pub use frame::{
    encode_keyframe, encode_keyframe_yuv, encode_pframe_yuv, encode_pframe_yuv_multi_ref,
};
pub use inter::{build_pframe, emit_pframe_tile};
pub use params::{EncoderParams, ReferenceFrame, ReferenceSet, YuvFrame};
pub use tile::emit_keyframe_tile;
pub use uncompressed_header::{emit_uncompressed_header, emit_uncompressed_header_p};
