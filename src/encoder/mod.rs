//! VP9 encoder — keyframe + round-49 P-frame (single-reference) scaffold.
//!
//! Reference: VP9 Bitstream & Decoding Process Specification v0.7.
//!
//! Keyframe path (rounds 1..48): assembles a valid VP9 keyframe step by
//! step — §6.2 uncompressed header, §6.3 compressed header, §6.4 tile /
//! partition walk, §6.4.3 per-block prediction/transform/quantise/
//! tokenise. Per-block luma and chroma intra-mode RDO (round 48) plus
//! mode-RDO early termination.
//!
//! P-frame path (round 49): single-reference (LAST_FRAME) inter encode
//! with 64×64 PARTITION_NONE blocks, integer-pel block-matching ME
//! (±16 px full search, SAD cost), ZEROMV / NEWMV inter modes,
//! `skip = 1` everywhere (no residual). See `inter.rs` for details.

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
pub use frame::{encode_keyframe, encode_keyframe_yuv, encode_pframe_yuv};
pub use inter::{build_pframe, emit_pframe_tile};
pub use params::{EncoderParams, ReferenceFrame, YuvFrame};
pub use tile::emit_keyframe_tile;
pub use uncompressed_header::{emit_uncompressed_header, emit_uncompressed_header_p};
