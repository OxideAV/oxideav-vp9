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
//! ## Cumulative scope (rounds 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18)
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
//! * Round 9 lands the §8.7 inverse transform process (a crate-local
//!   module `idct`) — the §8.7.1.1 butterfly primitives (`B` / `H` /
//!   `SB` / `SH`, the `16+32*k` two-multiply fast path, `cos64` /
//!   `sin64` backed by the verbatim 33-entry `cos64_lookup` table,
//!   `brev`), the recursive §8.7.1.2/§8.7.1.3 inverse DCT for
//!   `2 <= n <= 5`, the §8.7.1.4 .. §8.7.1.9 inverse ADST
//!   (ADST4/8/16) for `2 <= n <= 4`, the §8.7.1.10 inverse
//!   Walsh-Hadamard transform, and the §8.7.2 2D driver
//!   `inverse_transform_2d` (per-`TxType` row/column transforms, the
//!   lossless WHT path, and the `Round2(T[i], Min(6, n+2))` column
//!   rounding). The §8.6.2 reconstruct driver that builds the
//!   `Dequant` input and adds the residual to the prediction lands
//!   in a later round; the round-9 surface is internal-only.
//! * Round 10 lands the §8.5.1 intra prediction process (a crate-local
//!   module `intra`) — the `PredMode` enum (the 10 §7.4.5 modes,
//!   `DC_PRED` = 0 .. `TM_PRED` = 9), a `Plane` row-major buffer
//!   standing in for `CurrFrame[ plane ]`, and `predict_intra` which
//!   builds the `aboveRow` / `leftCol` neighbour arrays (with the
//!   `haveAbove` / `haveLeft` / `notOnRight` availability rules and
//!   the `Min(maxX, .)` / `Min(maxY, .)` plane-edge clamps), forms the
//!   `pred` block for each mode (`V`/`H`/`DC` with all four neighbour
//!   cases, `D45`/`D63`/`D117`/`D135`/`D153`/`D207` directional, and
//!   `TM` with `Clip1`), and stores it back. The §8.6.2 reconstruct
//!   driver that supplies the real availability flags (from tile /
//!   frame edges) and adds the round-9 inverse-transformed residual
//!   lands in a later round; the round-10 surface is internal-only.
//! * Round 11 lands the §8.6.2 reconstruct driver (a crate-local module
//!   `reconstruct`) that finally ties the rounds 7-10 pieces together:
//!   `tx_type_for_intra` (the §6.4.25 `mode2txfm_map[ y_mode ]` lookup
//!   selecting `TxType` from the intra `PredMode`), `reconstruct_block`
//!   (the §8.6.2 process — `dqDenom = 2 if txSz == TX_32X32 else 1`,
//!   step 1/2 `Dequant[i][j] = (Tokens[i*n0+j] * ac_quant) / dqDenom`
//!   with the `Dequant[0][0]` DC override, step 3 the §8.7.2 inverse
//!   transform, step 4 `Clip1( CurrFrame + Dequant )`), and
//!   `reconstruct_intra_block` (the end-to-end one-block driver that
//!   predicts via §8.5.1 `predict_intra`, derives the `TxType` with the
//!   §6.4.25 `TX_32X32` / lossless `DCT_DCT` overrides, then runs
//!   `reconstruct_block`). The §6.4.21 residual loop that supplies the
//!   real per-block `Tokens` arrays, availability flags and
//!   segment/quantizer state across a whole frame lands in a later
//!   round; the round-11 surface is internal-only.
//! * Round 12 lands the §6.4.25 `get_scan( )` scan-order selection (a
//!   crate-local module `scan`) — the first step of the §6.4.24
//!   `tokens( )` per-block driver. The §10.1 scan tables
//!   (`default_scan_4x4` .. `default_scan_32x32`, plus the `row_scan`
//!   / `col_scan` variants for 4x4 / 8x8 / 16x16) are transcribed
//!   verbatim, and `get_scan( plane, txSz, txType )` selects between
//!   them — `ADST_DCT` → `row_scan`, `DCT_ADST` → `col_scan`, else
//!   `default` — applying the §6.4.25 chroma / `TX_32X32`
//!   force-to-`DCT_DCT` first half. The §6.4.24 `tokens( )` loop that
//!   walks `pos = scan[ c ]` and the §6.4.21 residual driver above it
//!   land in a later round; the round-12 surface is internal-only.
//! * Round 13 lands the §6.4.24 `tokens( )` per-block coefficient
//!   driver (extending the crate-local `tokens` module) — the §10
//!   `coefband_4x4` / `coefband_8x8plus` band tables and the
//!   `coef_band` dispatch, the §9.3.2 `token_cache_neighbours`
//!   derivation (`nb[ 0 ]` / `nb[ 1 ]`), the `build_token_probs`
//!   §9.3.2 10-node probability array, the `NonzeroContext` /
//!   `TokenBlockCtx` state bundles, and `tokens( )` itself: the
//!   `checkEob`-gated walk over `pos = scan[ c ]` that derives the
//!   per-coefficient `ctx` (DC from the non-zero strips, `c > 0` from
//!   `TokenCache`), picks the
//!   `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell, runs the
//!   round-7 `more_coefs` / `token` / `read_coef` / `sign_bit` decode,
//!   writes `TokenCache[ pos ] = energy_class[ token ]`, applies the
//!   `ZERO_TOKEN`-clears-`checkEob` rule, zero-fills the trailing scan
//!   positions, and returns `nonzero = c > 0`. The §6.4.21 residual
//!   loop that threads `NonzeroContext` across the frame and feeds the
//!   round-11 reconstruct driver lands in a later round; the round-13
//!   surface is internal-only.
//! * Round 14 lands the §6.4.21 `residual( )` intra driver (a
//!   crate-local module `residual`) — the per-plane, per-4x4-sub-block
//!   walk that owns the `AboveNonzeroContext` / `LeftNonzeroContext`
//!   write-back across a whole MI block, drives the round-13 §6.4.24
//!   `tokens( )` per-block decode, and feeds the round-11 §8.6.2
//!   `reconstruct_block` with real per-block `Tokens` arrays,
//!   availability flags and plane/quantizer state. Surfaces the §10.2
//!   `num_4x4_blocks_wide_lookup` / `_high_lookup`, §6.4.10
//!   `max_txsize_lookup`, §6.4.23 `ss_size_lookup` tables, the
//!   `get_plane_block_size` / `get_uv_tx_size` helpers (§6.4.23 /
//!   §6.4.22), the `ResidualBlockCtx` / `AvailFlags` / `PlaneBuffers`
//!   state bundles, and `residual_intra( planes, nz, block, avail,
//!   token_source )` itself: per plane, computes the §6.4.21
//!   `bsize = max(MiSize, BLOCK_8X8)`, the per-plane `num4x4w` /
//!   `num4x4h` dimensions and chroma `txSz`, then steps the `(y, x)`
//!   4x4 grid by `step = 1 << txSz` calling `predict_intra` /
//!   (`!skip`) `tokens` + `reconstruct_block` per in-bounds block, and
//!   writing the `nonzero` flag back into the `AboveNonzeroContext` /
//!   `LeftNonzeroContext` strips. The `is_inter` branch (which calls
//!   `predict_inter( )`) is deferred until reference-buffer state
//!   lands; the per-block mode-info decode (`y_mode` / `sub_modes` /
//!   `tx_size` / `skip` / `segment_id` from §6.4) that the residual
//!   loop reads is also a later-round increment. The round-14 surface
//!   is internal-only.
//! * Round 15 lands the first slice of the §6.4 per-block mode-info
//!   decode the round-14 [`residual_intra`] driver currently consumes
//!   via a caller-supplied bundle: the §9.3.3 generic `tree_decode( )`
//!   helper, the §6.4.8 `read_skip( )` syntax element (with the §6.4.9
//!   `seg_feature_active( SEG_LVL_SKIP )` early-return and the §9.3.2
//!   ctx `Skips[ MiRow-1 ][ MiCol ] + Skips[ MiRow ][ MiCol-1 ]`), and
//!   the §6.4.10 `read_tx_size( allowSelect )` syntax element (the
//!   `TX_MODE_SELECT && MiSize >= BLOCK_8X8` `tx_size` decode using
//!   the §9.3.1 `tx_size_8_tree` / `tx_size_16_tree` / `tx_size_32_tree`
//!   transcribed verbatim, indexed by `max_txsize_lookup[ MiSize ]`,
//!   plus the §9.3.2 ctx `( above + left ) > maxTxSize` from
//!   `TxSizes[ ][ ]` / `Skips[ ][ ]` neighbour cells, and the
//!   `Min( maxTxSize, tx_mode_to_biggest_tx_size[ tx_mode ] )`
//!   fallback). `NeighbourSkips` / `NeighbourTxSizes` bundles thread
//!   the §7.4.4 `AvailL` / `AvailU` rules through. The §6.4.6
//!   `intra_frame_mode_info( )` orchestrator that wires `read_skip` +
//!   `read_tx_size` + the deferred §6.4.7 `intra_segment_id` +
//!   §6.4.15 `intra_block_mode_info` into a single `Vp9IntraMiBlock`
//!   lands in a later round; the round-15 surface is internal-only.
//! * Round 16 extends the round-15 `mode_info` module with the next
//!   slice of the §6.4.6 orchestrator's primitives: the §9.3.1
//!   `segment_tree[14]` transcribed verbatim (the 7-leaf binary tree
//!   used by every `segment_id` decode site), `read_segment_id( coder,
//!   tree_probs )` running the §9.3.3 walk over it with
//!   `segmentation_tree_probs[node]` per the §9.3.2 listing, and
//!   `intra_segment_id( coder, segmentation_enabled,
//!   segmentation_update_map, tree_probs )` (§6.4.7) gating the decode
//!   on `segmentation_enabled && segmentation_update_map` and falling
//!   through to `segment_id = 0` otherwise (intra has no
//!   `segmentation_temporal_update` / `seg_id_predicted` machinery —
//!   that's inter-only and lands when §6.4.12 does). The §6.4.15
//!   `intra_block_mode_info` and §6.4.6 `intra_frame_mode_info()`
//!   orchestrator are deferred to the next round; the round-16
//!   surface is internal-only.
//! * Round 17 lands the §6.4.6 `intra_frame_mode_info()` keyframe-only
//!   per-block driver on top of the rounds 15 / 16 primitives. The
//!   driver wires `intra_segment_id` plus `read_skip` plus
//!   `read_tx_size( 1 )` plus `default_intra_mode` plus
//!   `default_uv_mode` into a single `Vp9IntraMiBlock` output
//!   (`segment_id`, `skip`, `tx_size`, `y_mode`, `sub_modes[4]`,
//!   `uv_mode`), plus the §6.4.6 fixed `ref_frame[0] = INTRA_FRAME =
//!   0`, `ref_frame[1] = NONE = -1`, `is_inter = false` triple. The
//!   `MiSize >= BLOCK_8X8` arm decodes one `default_intra_mode` and
//!   replicates it into all four `sub_modes[ ]` cells; the
//!   `MiSize < BLOCK_8X8` arm walks the §6.4.6 `(idy, idx)` grid
//!   stepped by `num_4x4_blocks_high_lookup` and
//!   `num_4x4_blocks_wide_lookup` (covering BLOCK_4X4 / BLOCK_4X8 /
//!   BLOCK_8X4 with 4, 2, and 2 `default_intra_mode` decodes
//!   respectively), with each cell receiving its own decoded mode
//!   replicated across the (num4x4h × num4x4w) `sub_modes[ ]`
//!   sub-grid, and `y_mode` set to the *last* decoded
//!   `default_intra_mode`. `default_intra_mode` uses the §9.3.2
//!   `kf_y_mode_probs[abovemode][leftmode][node]` row (with the
//!   §9.3.2 above/left derivation handling both the
//!   `MiSize >= BLOCK_8X8` `(SubModes[MiRow-1][MiCol][2],
//!   SubModes[MiRow][MiCol-1][1])` path and the sub-8x8
//!   `(SubModes[MiRow-1][MiCol][2 + idx], sub_modes[ idx ])` plus
//!   `(sub_modes[ idy * 2 ], SubModes[MiRow][MiCol-1][1 + idy * 2])`
//!   paths, with `DC_PRED` substituted when `AvailU` or `AvailL` is
//!   false). `default_uv_mode` uses `kf_uv_mode_probs[y_mode][node]`.
//!   The §9.3.1 `intra_mode_tree[18]` and the §10.5
//!   `kf_y_mode_probs[10][10][9]` plus `kf_uv_mode_probs[10][9]`
//!   tables are transcribed verbatim from `docs/video/vp9/vp9-spec.txt`.
//!   The §6.4.15 `intra_block_mode_info` (used on intra blocks within
//!   inter frames; uses `y_mode_probs[size_group_lookup[MiSize]]`,
//!   `y_mode_probs[0]`, `uv_mode_probs[y_mode]` from the compressed
//!   header rather than the keyframe `kf_*_mode_probs` tables) lands
//!   alongside the §6.4.11 inter-frame driver in a later round; the
//!   round-17 surface is internal-only.
//! * Round 18 lands the §6.4.3 `decode_partition_type( )` partition reader
//!   (a crate-local module `partition`) — the per-call decoder that fires
//!   once per `(r, c, bsize)` quadrant inside the recursive §6.4.3
//!   `decode_partition( )` driver. Covers the §9.3.1 tree-selection
//!   (`partition_tree[6]` interior, `cols_partition_tree[2]` right-edge,
//!   `rows_partition_tree[2]` bottom-edge, plus the four-corner case that
//!   returns `PARTITION_SPLIT` without consuming any bits), the §9.3.2
//!   probability-selection rule (`node2 = node` for interior, fixed at 1
//!   for right-edge, fixed at 2 for bottom-edge), the §9.3.2 `ctx = bsl *
//!   4 + left * 2 + above` derivation via `partition_plane_context`
//!   (`bsl = mi_width_log2_lookup[bsize]`, `boffset = 3 - bsl`, OR-fold
//!   of `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips
//!   across `num8x8 = num_8x8_blocks_wide_lookup[bsize]` cells), the §3
//!   `PARTITION_NONE`=0 / `PARTITION_HORZ`=1 / `PARTITION_VERT`=2 /
//!   `PARTITION_SPLIT`=3 constants and `PARTITION_TYPES=4` /
//!   `PARTITION_CONTEXTS=16` dimensions, the §10.2
//!   `b_width_log2_lookup` / `b_height_log2_lookup` /
//!   `mi_width_log2_lookup` / `num_8x8_blocks_wide_lookup` lookups
//!   transcribed verbatim, the §10.2 `subsize_lookup[4][13]`
//!   PARTITION→child block-size table transcribed verbatim (with
//!   `BLOCK_INVALID = 14` for illegal combinations), and the §10.4
//!   `kf_partition_probs[16][3]` keyframe fixed-probability table plus
//!   the §10.5 `default_partition_probs[16][3]` inter-frame initial
//!   probability table both transcribed verbatim. The recursive §6.4.3
//!   driver itself (the `decode_partition( r, c, bsize )` function that
//!   splits on the decoded `partition`, threads `subsize_lookup[
//!   partition ][ bsize ]` into four recursive calls when
//!   `PARTITION_SPLIT`, and writes back the
//!   `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips with
//!   `15 >> b_*_log2_lookup[ subsize ]`) and the §6.3
//!   `read_partition_probs( )` compressed-header sweep both land in a
//!   later round; the round-18 surface is internal-only.
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
mod idct;
mod intra;
mod mode_info;
mod partition;
mod reconstruct;
mod residual;
mod scan;
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
