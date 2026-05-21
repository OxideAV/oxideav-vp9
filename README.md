# oxideav-vp9

Pure-Rust VP9 codec — clean-room re-implementation against the VP9
Bitstream & Decoding Process Specification v0.7.

## Status — 2026-05-22

**Round 5: §6.3.2 `tx_mode_probs` + §6.3.8 `read_skip_prob` sweeps
wired into the compressed-header walker.** Round 5 consumes the
round-4 `read_diff_update_prob` primitive in two table sweeps:

* `tx_mode_probs( )` (§6.3.2) — gated on `tx_mode == TX_MODE_SELECT`,
  walks `tx_probs_8x8[2][1]`, `tx_probs_16x16[2][2]`,
  `tx_probs_32x32[2][3]` (12 cells total) via
  `read_diff_update_prob` starting from the §10 `default_tx_probs`
  initials transcribed verbatim into `DEFAULT_TX_PROBS`.
* `read_skip_prob( )` (§6.3.8) — unconditional 3-element
  (`SKIP_CONTEXTS = 3`) sweep over the §10 `default_skip_prob = {
  192, 128, 64 }` initials transcribed verbatim into
  `DEFAULT_SKIP_PROB`.

`Vp9CompressedHeader` now exposes the post-sweep `tx_probs` /
`skip_prob` tables. When `tx_mode != TX_MODE_SELECT` the §6.3
syntax skips `tx_mode_probs( )` entirely, so the field is left
equal to `DEFAULT_TX_PROBS`. `parse_compressed_header` runs both
sweeps in spec order: `read_tx_mode` → optional `read_tx_mode_probs`
→ `read_skip_prob`.

The §6.3.7 `read_coef_probs` 4D nested sweep (gated by an outer
`L(1) update_probs` flag per tx-size) plus all inter-only §6.3.9+
syntax remain deferred — they fire only on
`tx_mode == TX_MODE_SELECT` (for coef) or `FrameIsIntra == 0` (for
the inter sweeps).

**Round 4: §6.3.3 `diff_update_prob` chain (`decode_term_subexp` +
`inv_remap_prob` + `inv_recenter_nonneg` + 255-entry
`inv_map_table`).** Round 4 lands the helper chain every §6.3.7+
probability sweep is built on:

* `read_diff_update_prob( coder, base_prob )` (§6.3.3) — reads the
  `B(252)` `update_prob` flag and, on 1, pulls a
  `decode_term_subexp` value (§6.3.4) then remaps the previous
  probability through `inv_remap_prob` (§6.3.5). On 0, passes the
  base probability straight through.
* `decode_term_subexp` (§6.3.4) — the 5-leg cascade
  (`L(1) → L(4)` / `L(1) → L(4)+16` / `L(1) → L(5)+32` /
  `L(7) → +64` / `L(7), L(1) → (v<<1)-1+bit`) yielding a value in
  `0..=254`.
* `inv_remap_prob` (§6.3.5) — the low-half / high-half piecewise
  remap, with the 255-entry `INV_MAP_TABLE` constant transcribed
  verbatim from the §6.3.5 listing.
* `inv_recenter_nonneg` (§6.3.6) — pure arithmetic helper.

The chain is structural — no caller in §6.3.2 / §6.3.7+ uses it yet.
That's deferred to the next round so this drop lands the primitive
in isolation. All four helpers are `pub(crate)` with explicit
`#[allow(dead_code)]` until the next round wires them into the
table sweeps.

**Round 3: §9.2 Boolean decoder + §6.3.1 `read_tx_mode` walk.** Building
on round 2's full §6.2 uncompressed-header walker, round 3 lands the
arithmetic-decode primitive — `init_bool( sz )` / `read_bool( p )` /
`read_literal( n )` / `exit_bool( )` per spec §9.2.1–§9.2.4 — plus the
first §6.3 compressed-header field (`read_tx_mode( )`, §6.3.1) exposed
via `parse_compressed_header(payload, lossless) -> Vp9CompressedHeader`.

Decoded `tx_mode` covers all five §3 values (`ONLY_4X4`, `ALLOW_8X8`,
`ALLOW_16X16`, `ALLOW_32X32`, `TX_MODE_SELECT`), including the lossless
short-circuit to `ONLY_4X4`. The §9.2.1 marker-bit zero-conformance
check is enforced (`InvalidBitstream` on a nonzero marker), and the
§9.2.2 `BoolMaxBits` underflow path is surfaced rather than silently
papered over.

The remaining §6.3.7+ syntax — `tx_mode_probs`, `read_coef_probs`,
`read_skip_prob`, `read_inter_mode_probs`,
`read_interp_filter_probs`, … — all funnel through
`read_diff_update_prob` (landed this round) and the table sweeps
themselves land in the next round. `read_inter_mode_probs` /
`read_interp_filter_probs` only fire on inter-frame headers, which
already return `Error::Unsupported` until reference-buffer state
lands.

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

* `cargo test`: 52 unit tests + 18 integration tests (6 in
  `tests/compressed_header.rs` plus 12 in
  `tests/uncompressed_header.rs`).
* Round-5 additions: 10 unit tests covering `DEFAULT_TX_PROBS`
  shape + value spot-check against the §10 listing, `DEFAULT_SKIP_PROB
  = [192, 128, 64]`, `read_tx_mode_probs` on a zero buffer
  (defaults pass through; row 0 (`TX_4X4`) is never touched), the
  12-cell loop-count shape, `read_skip_prob` on a zero buffer plus
  a 3-context-visit sanity check, and three `parse_compressed_header`
  integration tests confirming the §10 defaults survive the
  zero-buffer sweep for the `ONLY_4X4`, lossless, and `TX_MODE_SELECT`
  paths. 2 new end-to-end integration tests
  (`tests/compressed_header.rs`): one driving the
  `TX_MODE_SELECT` → `tx_mode_probs` → `read_skip_prob` chain from
  a 64×64 uncompressed-header splice, one confirming `ALLOW_16X16`
  skips the §6.3.2 sweep.
* Round-4 additions: 16 unit tests covering `inv_recenter_nonneg`
  (all three piecewise branches plus the `v == 2*m` boundary),
  `INV_MAP_TABLE` length + spot-checks against the spec listing
  anchors (`[0]=7`, `[19]=254`, `[20]=1`, `[254]=253` with the
  duplicated trailing 253), `inv_remap_prob` covering both
  low-half and high-half `(m << 1) ≤ 255` branches plus the
  `v > 2m` short-circuit, `decode_term_subexp` against
  hand-derived §9.2 buffers (leg-1 zero plus a sweep that confirms
  the result stays in `0..=254` for every first-byte family), and
  `read_diff_update_prob` confirming the `update_prob == 0`
  passthrough against the full 1..=255 base-probability sweep.
* Round-3 additions: 9 `bool_coder` unit tests covering
  `init_bool` (zero-size / short-slice / nonzero-marker rejection +
  the zero-buffer accept path), `read_bool` against hand-traced
  golden buffers (mixed probabilities, extreme p=255 run), `read_literal`,
  and `exit_bool` accept/reject paths; plus 7 `compressed` unit tests
  for `parse_compressed_header` covering each `TxMode` value
  (lossless short-circuit, `ONLY_4X4`/`ALLOW_8X8`/`ALLOW_16X16`/
  `ALLOW_32X32`/`TX_MODE_SELECT` golden buffers) and the
  marker / empty-buffer rejection paths; and 4 end-to-end
  integration tests (`tests/compressed_header.rs`) that build a
  64x64 key-frame uncompressed header, splice a §9.2 payload past
  the byte-aligned `trailing_bits` pad, and verify the walker
  picks up the right `tx_mode` from the spliced payload.
* Uncompressed-header coverage (rounds 1 + 2, unchanged) spans the
  four profiles, studio/full-swing color ranges, render-size
  overrides, the `show_existing_frame` early return, the intra-only
  inter-frame branch (with the spec's BT.601 / 4:2:0 / 8-bit defaults
  for Profile 0), full `loop_filter_params` delta update with mixed
  `update_ref_delta` / `update_mode_delta` flags and signed `s(6)`
  deltas, `quantization_params` with a nonzero `base_q_idx` and
  signed `delta_q_y_dc`, full segmentation with `update_map` +
  `temporal_update` + `update_data` driving the per-segment /
  per-feature inner loop including the 0-magnitude-bit skip feature,
  `tile_info` increment-walk for a 4K-wide frame, plus three failure
  paths (bad `frame_marker`, bad `frame_sync_code`, truncated buffer)
  and the §7.1.1 nonzero trailing-bit rejection.
* No external fixtures are involved yet; each test constructs its
  input bit-by-bit (and §9.2 golden buffers are hand-derived by
  stepping the decoder, not borrowed from any third-party VP9
  implementation).

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

1. §6.3.7 `read_coef_probs` (4D nested sweep gated by an outer
   `L(1) update_probs` flag per tx-size). The sweep walks `txSz ∈
   [TX_4X4, maxTxSize]` with `maxTxSize = tx_mode_to_biggest_tx_size[
   tx_mode ]`, so it produces visible output for every non-trivial
   `tx_mode` and consumes the round-4 `read_diff_update_prob`. Needs
   the §10 `default_coef_probs[ TX_SIZES ][ BLOCK_TYPES ][ REF_TYPES
   ][ COEF_BANDS ][ PREV_COEF_CONTEXTS ][ UNCONSTRAINED_NODES ]`
   table transcribed into a const lookup first.
2. Inter (non-intra-only) header path — `frame_size_with_refs`,
   `allow_high_precision_mv`, `read_interpolation_filter` — plus
   the inter-only §6.3.9–§6.3.16 syntax (`read_inter_mode_probs`,
   `read_interp_filter_probs`, `read_is_inter_probs`,
   `frame_reference_mode`, `mv_probs`) once reference-buffer state
   is in place.
3. Intra prediction (§8.5) over a single tile.
4. Inverse transforms + dequant.
5. Inter prediction, loop filter, multi-tile, then encoder paths.

## License

MIT. See `LICENSE`.
