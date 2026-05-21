# oxideav-vp9

Pure-Rust VP9 codec — clean-room re-implementation against the VP9
Bitstream & Decoding Process Specification v0.7.

## Status — 2026-05-21

**Round 2: full §6.2 uncompressed-header walk.** Building on round 1,
`parse_uncompressed_header(&[u8]) -> Result<Vp9FrameHeader, Error>` now
walks the entire `uncompressed_header()` syntax tree plus the §6.1.1
`trailing_bits()` zero-fill alignment. Per-field coverage:

* Round 1: `frame_marker`, `Profile` (with the `profile == 3`
  `reserved_zero` bit), `show_existing_frame` early-return path
  (returns `frame_to_show_map_idx`), `frame_type`, `show_frame`,
  `error_resilient_mode`, `frame_sync_code` (`0x49 / 0x83 / 0x42`),
  `color_config()` (bit depth 8/10/12, all 8 `color_space` values,
  `color_range`, `subsampling_x` / `subsampling_y` with the §7.2.2
  reserved-zero and CS_RGB-on-profile-0/2 constraint checks), and
  `frame_size` / `render_size` (with the
  `render_and_frame_size_different == 1` override).
* Round 2: post-`render_size` syntax — `refresh_frame_context`,
  `frame_parallel_decoding_mode`, `frame_context_idx`, the
  `setup_past_independence` `frame_context_idx = 0` reset for intra /
  error-resilient frames, `loop_filter_params()` (§6.2.8) including
  full delta-update walk with `s(6)` ref/mode deltas,
  `quantization_params()` (§6.2.9) with `read_delta_q()` (§6.2.10),
  `segmentation_params()` (§6.2.11) with `read_prob()` (§6.2.12) +
  per-segment / per-feature data using the `segmentation_feature_bits`
  / `segmentation_feature_signed` tables, `tile_info()` (§6.2.13)
  driven by `Sb64Cols` computed from `FrameWidth` per §6.2.6 /
  §6.2.14, the f(16) `header_size_in_bytes`, and the §6.1.1
  `trailing_bits()` zero-pad alignment with §7.1.1 zero-bit
  conformance check.

`decode_vp9()` / `encode_vp9()` still return `Error::NotImplemented`;
the compressed header (§6.3) plus entropy decoder, intra/inter
prediction, transforms and loop filter land in later rounds. Inter
(non-intra-only) headers — which require `frame_size_with_refs` and
reference-buffer state — return `Error::Unsupported` from the header
walker for now.

The `Vp9FrameHeader` struct now also exposes
`uncompressed_header_size_bytes` (the byte-aligned offset at which the
§6.3 compressed header starts), giving callers everything needed to
slice the compressed-header payload of `header_size_in_bytes` from the
remainder of the frame.

## Test surface

* `cargo test`: 9 internal tests (`src/bitreader.rs` — MSB-first
  `f(n)`, `s(n)` round-trip, `trailing_bits` accept/reject/no-op;
  `src/header.rs` — `Sb64Cols` / `calc_*_log2_tile_cols` arithmetic
  for small and 4K frames) + 12 integration tests
  (`tests/uncompressed_header.rs`).
* Integration coverage spans the four profiles, studio/full-swing
  color ranges, render-size overrides, the `show_existing_frame` early
  return, the intra-only inter-frame branch (with the spec's BT.601 /
  4:2:0 / 8-bit defaults for Profile 0), full `loop_filter_params`
  delta update with mixed `update_ref_delta` / `update_mode_delta`
  flags and signed `s(6)` deltas, `quantization_params` with a
  nonzero `base_q_idx` and signed `delta_q_y_dc`, full segmentation
  with `update_map` + `temporal_update` + `update_data` driving the
  per-segment / per-feature inner loop including the 0-magnitude-bit
  skip feature, `tile_info` increment-walk for a 4K-wide frame, plus
  three failure paths (bad `frame_marker`, bad `frame_sync_code`,
  truncated buffer) and the §7.1.1 nonzero trailing-bit rejection.
* No external fixtures are involved yet; each test constructs its
  input bit-by-bit and round-trips against the expected
  `Vp9FrameHeader` fields.

## Provenance

Single source of truth: VP9 Bitstream & Decoding Process Specification
v0.7 (`docs/video/vp9/vp9-spec.txt`). No external library source —
`libvpx`, `libaom`, FFmpeg's `libavcodec/vp9*`, `dav1d`, `libgav1`,
prior 0.0.x releases of this crate — has been consulted, quoted, or
cross-checked. Black-box `ffmpeg` binary invocations remain
permissible as opaque validators but are not yet wired into the test
harness.

## Roadmap

Future rounds, roughly in order:

1. Boolean (range) decoder per §9.2 and compressed-header walk
   (§6.3 / §7.3).
2. Inter (non-intra-only) header path — `frame_size_with_refs`,
   `allow_high_precision_mv`, `read_interpolation_filter` — once
   reference-buffer state is in place.
3. Intra prediction (§8.5) over a single tile.
4. Inverse transforms + dequant.
5. Inter prediction, loop filter, multi-tile, then encoder paths.

## License

MIT. See `LICENSE`.
