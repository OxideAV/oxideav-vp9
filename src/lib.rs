//! # oxideav-vp9
//!
//! Pure-Rust, clean-room VP9 codec crate. Round-1 cut covers the
//! uncompressed-header structural walker per VP9 Bitstream & Decoding
//! Process Specification v0.7 §6.2 / §7.2 — enough to extract profile,
//! frame type, color config, frame size and render size from a VP9
//! frame byte stream. Entropy decoding, intra prediction, inter
//! prediction, transforms and loop filtering are all out of scope and
//! land in later rounds.
//!
//! ## Round-1 scope (per `docs/IMPLEMENTOR_ROUND.md` dispatch)
//!
//! * MSB-first `f(n)` bit reader (spec §9.1).
//! * Uncompressed header walker up through `render_size()`:
//!   `frame_marker`, `Profile`, `show_existing_frame`,
//!   `frame_type`, `show_frame`, `error_resilient_mode`,
//!   `frame_sync_code`, `color_config`, `frame_size`, `render_size`.
//! * Both key-frame and intra-only inter-frame paths are walked.
//! * Inter-frame (non-intra-only) headers — `frame_size_with_refs`,
//!   motion-vector / interpolation-filter flags — return
//!   [`Error::Unsupported`] for now; they need reference-buffer
//!   state.
//!
//! Everything past `render_size()` in §6.2 (loop_filter_params,
//! quantization_params, segmentation_params, tile_info,
//! `header_size_in_bytes`), the trailing-zero alignment in §6.1.1,
//! and the compressed header are out of round-1 scope and will land
//! in subsequent rounds.
//!
//! ## Provenance
//!
//! Clean-room, single source of truth: `docs/video/vp9/vp9-spec.txt`
//! (the v0.7 specification snapshot). No external library source was
//! consulted.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

mod bitreader;
mod header;

pub use header::{parse_uncompressed_header, ColorConfig, ColorSpace, FrameType, Vp9FrameHeader};

/// Crate-local error type.
///
/// `decode_vp9` / `encode_vp9` still return [`Error::NotImplemented`]
/// — the round-1 cut only lands the uncompressed-header walker
/// (see [`parse_uncompressed_header`]). Future rounds will wire the
/// full decode/encode pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The decoder or encoder pipeline is not wired up yet.
    NotImplemented,
    /// The reader ran out of bits while walking the header.
    UnexpectedEof,
    /// A "shall be equal to" constraint from spec §7.2 was violated
    /// (e.g. `frame_marker != 2`, `reserved_zero != 0`, an illegal
    /// `color_space` / profile combination, or a bad
    /// `frame_sync_code`).
    InvalidBitstream,
    /// The input header is valid but uses a syntax path that has not
    /// yet been implemented in the round-1 cut (currently:
    /// inter-frame headers that need `frame_size_with_refs` and
    /// reference-buffer state).
    Unsupported,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "oxideav-vp9: decode/encode pipeline not wired up yet (round-1 header walker only)"
            ),
            Self::UnexpectedEof => {
                write!(f, "oxideav-vp9: ran out of bits while parsing header")
            }
            Self::InvalidBitstream => {
                write!(f, "oxideav-vp9: VP9 bitstream constraint violated")
            }
            Self::Unsupported => write!(
                f,
                "oxideav-vp9: header path not yet implemented in this crate version"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Decode a VP9 elementary stream.
///
/// Returns [`Error::NotImplemented`] — the full decode pipeline lands
/// in a later round. For header introspection, use
/// [`parse_uncompressed_header`] directly.
pub fn decode_vp9(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Encode YUV data into a VP9 elementary stream.
///
/// Returns [`Error::NotImplemented`] — the encoder lands in a later
/// round.
pub fn encode_vp9(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — round 1 has nothing to register into the
/// runtime context until the decode / encode paths land.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("vp9", register);
