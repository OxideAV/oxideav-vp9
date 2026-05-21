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
//! ## Cumulative scope (rounds 1 + 2 + 3)
//!
//! * MSB-first `f(n)` bit reader plus `s(n)` signed-integer reader
//!   (spec §9.1 + §4.9.2).
//! * Uncompressed header walker through the end of
//!   `uncompressed_header()` and the §6.1.1 `trailing_bits()`
//!   zero-fill alignment.
//!   * Round 1 covered: `frame_marker`, `Profile`,
//!     `show_existing_frame`, `frame_type`, `show_frame`,
//!     `error_resilient_mode`, `frame_sync_code`, `color_config`,
//!     `frame_size`, `render_size`.
//!   * Round 2 added: `refresh_frame_context`,
//!     `frame_parallel_decoding_mode`, `frame_context_idx`,
//!     `loop_filter_params()` (§6.2.8), `quantization_params()`
//!     (§6.2.9), `segmentation_params()` (§6.2.11), `tile_info()`
//!     (§6.2.13), `header_size_in_bytes` (f(16)), and the §6.1.1
//!     `trailing_bits()` zero-fill conformance check.
//! * Round 3 added: the §9.2 Boolean (range) decoder primitives —
//!   `init_bool( sz )` / `read_bool( p )` / `read_literal( n )` /
//!   `exit_bool( )` — plus the §6.3.1 `read_tx_mode( )` walk
//!   exposed via [`parse_compressed_header`]. The §6.3.2+ syntax
//!   (`tx_mode_probs`, `read_coef_probs`, `read_skip_prob`, …) all
//!   depend on the §6.3.3 `diff_update_prob` chain and have been
//!   deferred to the next round.
//! * Both key-frame and intra-only inter-frame paths are walked.
//! * Inter-frame (non-intra-only) headers — `frame_size_with_refs`,
//!   motion-vector / interpolation-filter flags — return
//!   [`Error::Unsupported`] for now; they need reference-buffer
//!   state.
//!
//! The remaining §6.3 fields and the entropy / transform / loop
//! filter pipelines remain out of scope and land in subsequent
//! rounds.
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
mod bool_coder;
mod compressed;
mod header;

pub use compressed::{parse_compressed_header, TxMode, Vp9CompressedHeader};
pub use header::{
    parse_uncompressed_header, ColorConfig, ColorSpace, FrameType, LoopFilterParams,
    QuantizationParams, SegmentationParams, TileInfo, Vp9FrameHeader, MAX_SEGMENTS,
    SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_SIGNED, SEG_LVL_MAX,
};

/// Crate-local error type.
///
/// `decode_vp9` / `encode_vp9` still return [`Error::NotImplemented`]
/// — the current cut only lands the uncompressed-header walker
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
    /// yet been implemented (currently: inter-frame headers that need
    /// `frame_size_with_refs` and reference-buffer state).
    Unsupported,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "oxideav-vp9: decode/encode pipeline not wired up yet (uncompressed-header walker only)"
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
