//! VP9 encoder — keyframe-only MVP.
//!
//! Reference: VP9 Bitstream & Decoding Process Specification v0.7.
//!
//! The encoder builds a valid VP9 keyframe step by step: §6.2
//! uncompressed header, §6.3 compressed header, §6.4 tile / partition
//! walk, §6.4.3 per-block prediction/transform/quantise/tokenise. Its
//! output can be round-tripped through this crate's decoder as the
//! primary correctness gauge.
//!
//! Status: header emission only. The quantiser, transforms, token coder,
//! and forward boolean coder follow in separate incremental commits.

pub mod bitwriter;
pub mod bool_encoder;
pub mod compressed_header;
pub mod frame;
pub mod params;
pub mod tile;
pub mod uncompressed_header;

pub use bitwriter::BitWriter;
pub use bool_encoder::BoolEncoder;
pub use compressed_header::emit_compressed_header;
pub use frame::encode_keyframe;
pub use params::EncoderParams;
pub use tile::emit_keyframe_tile;
pub use uncompressed_header::emit_uncompressed_header;
