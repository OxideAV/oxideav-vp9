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
pub mod params;
pub mod uncompressed_header;

pub use bitwriter::BitWriter;
pub use params::EncoderParams;
pub use uncompressed_header::emit_uncompressed_header;
