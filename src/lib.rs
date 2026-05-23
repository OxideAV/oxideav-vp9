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
//! ## Cumulative scope (rounds 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)
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
//!   exposed via [`parse_compressed_header`].
//! * Round 4 added: the §6.3.3 `diff_update_prob` chain —
//!   `read_diff_update_prob` (§6.3.3) + `decode_term_subexp`
//!   (§6.3.4) + `inv_remap_prob` (§6.3.5) +
//!   `inv_recenter_nonneg` (§6.3.6) + the 255-entry
//!   `INV_MAP_TABLE` constant — as a `pub(crate)` primitive.
//! * Round 5 wired the round-4 chain into the §6.3.2
//!   `tx_mode_probs( )` (conditional on `tx_mode == TX_MODE_SELECT`)
//!   and §6.3.8 `read_skip_prob( )` sweeps. `Vp9CompressedHeader`
//!   now exposes the post-sweep `tx_probs` and `skip_prob` tables;
//!   defaults come from the §10 `default_tx_probs` and
//!   `default_skip_prob` listings (transcribed verbatim into
//!   `DEFAULT_TX_PROBS` / `DEFAULT_SKIP_PROB`).
//! * Round 6 lands the §6.3.7 `read_coef_probs( )` 6D
//!   coefficient-probability sweep between the round-5 §6.3.2 and
//!   §6.3.8 calls. The walker visits `txSz ∈ [TX_4X4, maxTxSize]`
//!   with `maxTxSize = tx_mode_to_biggest_tx_size[ tx_mode ]`, an
//!   outer `L(1) update_probs` per active slab, and a nested
//!   `BLOCK_TYPES × REF_TYPES × COEF_BANDS × maxL × UNCONSTRAINED_NODES`
//!   `read_diff_update_prob` walk. `Vp9CompressedHeader` exposes
//!   the post-sweep `coef_probs: CoefProbs` table (1728 entries);
//!   defaults come from the §10 `default_coef_probs` listing
//!   transcribed verbatim into `DEFAULT_COEF_PROBS`. The inter-only
//!   §6.3.9+ syntax remains deferred.
//! * Round 7 lands the §6.4.24 / §6.4.26 coefficient-token decoder
//!   (a crate-local module `tokens`) — the §9.3.3 `token_tree`
//!   walker, the §9.3.2 `pareto( node, prob )` helper backed by the
//!   §10.3 128-entry pareto table, the `more_coefs` `B(p)` reader,
//!   the §6.4.26 `read_coef` extra-bits + 8/10/12-bit `high_bit`
//!   decode, and a `read_coef_token` driver returning `CoefStep::
//!   Eob | Coef { token, value }`. The §11 `extra_bits[11][3]`
//!   table, the `cat_probs[7][14]` table, and the §10.2
//!   `energy_class[12]` table are all transcribed verbatim. The
//!   §6.4.21 `residual( )` plane / sub-block driver and the
//!   `AboveNonzeroContext` / `LeftNonzeroContext` state still need
//!   a later round; the round-7 surface is internal-only.
//! * Round 8 lands the §8.6.1 dequantization functions (a crate-local
//!   module `dequant`) — `dc_q` / `ac_q` indexing the verbatim
//!   `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
//!   `(BitDepth-8)>>1` row and a `Clip3(0,255,b)` column,
//!   `seg_feature_active` (§6.4.9), `get_qindex` (with the
//!   `SEG_LVL_ALT_Q` absolute/delta segment-feature path), and
//!   `get_dc_quant` / `get_ac_quant` threading the plane-specific
//!   `delta_q_*` header deltas through. The §8.6.2 reconstruct
//!   driver that scales the round-7 `Tokens` by these quantizers
//!   lands in a later round; the round-8 surface is internal-only.
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
mod coef_probs;
mod compressed;
mod dequant;
mod header;
mod tokens;

pub use coef_probs::CoefProbs;
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
