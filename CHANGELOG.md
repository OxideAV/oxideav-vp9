# Changelog

All notable changes to `oxideav-vp9` are recorded here.

## [Unreleased]

### Added

* **Round 2: full §6.2 uncompressed-header walk.** Extends the round-1
  walker through the end of `uncompressed_header()` and the §6.1.1
  `trailing_bits()` zero-fill alignment:
  * `s(n)` signed-integer reader per spec §4.9.2 plus
    `BitReader::trailing_bits()` zero-pad consumer with §7.1.1
    zero-bit conformance check (`src/bitreader.rs`).
  * `read_loop_filter_params()` (§6.2.8) with full `delta_enabled` /
    `delta_update` / per-ref / per-mode `s(6)` delta walk.
  * `read_quantization_params()` (§6.2.9) with `read_delta_q()`
    (§6.2.10) for `delta_q_y_dc` / `_uv_dc` / `_uv_ac` and the
    `Lossless` derivation.
  * `read_segmentation_params()` (§6.2.11) with `read_prob()`
    (§6.2.12), the 7 `tree_probs`, the 3 `pred_prob` (with
    `temporal_update` switching between f(0)-implicit-255 and
    `read_prob()`), and the per-segment / per-feature
    `feature_enabled` / `feature_value` / optional `feature_sign`
    loop driven by `segmentation_feature_bits` /
    `segmentation_feature_signed`.
  * `read_tile_info()` (§6.2.13) with `Sb64Cols` computed from
    `FrameWidth` via §6.2.6 (`MiCols = (W+7)>>3`,
    `Sb64Cols = (MiCols+7)>>3`) and `calc_min_log2_tile_cols` /
    `calc_max_log2_tile_cols` per §6.2.14
    (`MIN_TILE_WIDTH_B64 = 4`, `MAX_TILE_WIDTH_B64 = 64`). The
    §7.2.11 `tile_cols_log2 <= 6` conformance constraint is checked.
  * f(16) `header_size_in_bytes`, then `trailing_bits()` consumed so
    `uncompressed_header_size_bytes` exposes the byte-aligned offset
    at which the §6.3 compressed header starts.
  * `refresh_frame_context`, `frame_parallel_decoding_mode`,
    `frame_context_idx` (with the `FrameIsIntra ||
    error_resilient_mode` reset to 0 per §6.2 `setup_past_independence`),
    `reset_frame_context`, `refresh_frame_flags` (0xFF for key
    frames per spec).
  * New public types: `LoopFilterParams`, `QuantizationParams`,
    `SegmentationParams`, `TileInfo`, plus constants `MAX_SEGMENTS`,
    `SEG_LVL_MAX`, `SEGMENTATION_FEATURE_BITS`,
    `SEGMENTATION_FEATURE_SIGNED`. `Vp9FrameHeader` extended with all
    the new fields plus `uncompressed_header_size_bytes`.
  * 6 additional bit-reader tests (`s(n)` round-trip,
    `trailing_bits` accept/reject/no-op) + 2 `tile_info`
    arithmetic tests + 4 additional integration tests
    (loop_filter delta-update, segmentation full update,
    `tile_info` 4K increment walk, nonzero trailing bit rejection).
    Existing integration tests extended with the new tail.

* **Round 1: uncompressed-header walker.** A clean-room implementation
  of the structural portion of VP9 spec v0.7 §6.2 / §7.2:
  * MSB-first `f(n)` bit reader (`src/bitreader.rs`) per spec §9.1.
  * `parse_uncompressed_header()` walking `frame_marker`, `Profile`
    (including the `profile == 3` reserved bit), `show_existing_frame`
    early-return with `frame_to_show_map_idx`, `frame_type`,
    `show_frame`, `error_resilient_mode`, `frame_sync_code`,
    `color_config()` (with full §7.2.2 constraint checks including
    CS_RGB-on-profile-0/2 rejection and `reserved_zero` enforcement),
    `frame_size()` and `render_size()`.
  * Public `Vp9FrameHeader`, `ColorConfig`, `ColorSpace`, `FrameType`
    types in the crate root.
  * `Error::UnexpectedEof`, `Error::InvalidBitstream`,
    `Error::Unsupported` variants in addition to the existing
    `NotImplemented`.
  * 3 internal bit-reader tests plus 8 integration tests
    (`tests/uncompressed_header.rs`) covering all four profiles,
    studio/full-swing color ranges, render-size overrides, the
    `show_existing_frame` early return, the intra-only inter-frame
    branch, and three failure paths (bad `frame_marker`, bad
    `frame_sync_code`, truncated buffer).

  `decode_vp9()` / `encode_vp9()` continue to return
  `Error::NotImplemented`; their full pipelines land in later rounds.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule (no external library source as
  reference, not even as a sanity check). Per the workspace's
  Implementer-Round procedure, such audit failures are unrecoverable
  via incremental cleanup and require an orphan rebuild.

  Every public API path returned `Error::NotImplemented`. A
  clean-room re-implementation against the VP9 Bitstream & Decoding
  Process Specification (v0.7) has now begun (see "Added" above).

  No `old` branch is retained; long-standing audit failures forfeit
  the archive per workspace policy.
