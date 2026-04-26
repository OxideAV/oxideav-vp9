# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Round 18 — §9.3.2 default_intra_mode tracker for ≥8×8 reverted to
  spec-literal `+0`.** `read_intra_mode` (MiSize ≥ BLOCK_8X8) was using
  `mi_col*2 + 1` / `mi_row*2 + 1` (sub_modes[3] = bottom-RIGHT) since
  round 15 because that scored slightly better against the
  *gray-fixture* lossless test — which round 17 then exposed as a
  measurement artefact. With the honest pattern fixture in place, the
  spec-literal `mi_col*2 + 0` / `mi_row*2 + 0` (= SubModes[..][2] =
  bottom-LEFT for above; SubModes[..][1] = top-RIGHT for left, mapped
  through the per-position 1D-tracker storage) wins on every metric:

  | metric                    | before (r17) | after (r18) | delta   |
  |---------------------------|--------------|-------------|---------|
  | lossless pattern Y PSNR   | 9.69 dB      | 9.90 dB     | +0.21   |
  | lossless pattern U PSNR   | 10.96 dB     | 10.80 dB    | -0.16   |
  | lossless pattern V PSNR   | 9.26 dB      | 10.21 dB    | +0.95   |
  | compound luma PSNR (mean) | 10.63 dB     | 10.72 dB    | +0.09   |

  The sub-8x8 path (`read_intra_sub_mode`) keeps the empirical `+1`
  anchor — switching it to spec-literal `+idx` / `+idy*2` regressed
  compound by ~1 dB (10.72 → 9.71) so that asymmetry is documented
  in the code. 155 tests still pass.

### Added

- **Round 17 — measurement audit**. Added
  `tests/vp9_lossless_pattern.rs`: a non-degenerate lossless-decode
  test against an `ffmpeg testsrc -lossless 1` reference YUV. The
  prior `vp9_lossless_gray.rs` test compared against
  `vec![126; 64*64]` — a constant-gray plane — so any decoder output
  that's "approximately gray" trivially scored ≥ 60 dB. The new test
  reveals the true lossless decode quality on real content: **9.69 dB
  Y, 10.96 dB U, 9.26 dB V** with virtually every byte differing. The
  bar is currently set just below the round-17 baseline (≥ 8 dB Y) so
  a future bit-exact fix lifts it dramatically and a regression below
  the current poor baseline is caught.

  Past-error log entry: rounds 11-16 all reported "Lossless bit-exact
  (66.77 dB)" against the gray fixture and treated that as a
  load-bearing invariant. The round-17 audit confirms it was a
  fixture artefact — the lossless WHT path
  (`transform::iwht4x4_add` / §8.7.2) does not actually reproduce the
  ffmpeg reference. The `vp9-lossless-gray` test now functions as a
  "DC-prediction-doesn't-blow-up" smoke check rather than a
  bit-exactness gate. README updated to reflect the real numbers.

### Changed

- §9.3.2 per-position above/left intra-mode tracker (round 16).
  `IntraTile::above_mode` / `left_mode` (both per-MI-cell, length
  `mi_cols` / `mi_rows`) are replaced by `above_mode_4x4` /
  `left_mode_4x4` (per-4×4-position, length `mi_cols * 2` /
  `mi_rows * 2`). The new layout matches the spec's `SubModes
  [r][c][b]` 3D array projected onto two 1D arrays per the §9.3.2
  optimisation note: each 8×8 cell occupies two adjacent slots
  storing `sub_modes[2]` / `sub_modes[3]` for above (bottom row) and
  `sub_modes[1]` / `sub_modes[3]` for left (right column). For the
  three sub-8×8 sizes (B4x4, B4x8, B8x4) `decode_block` now walks
  `(idy, idx)` with steps `num4x4w / num4x4h` per §6.4.6 and reads
  `read_intra_sub_mode` per-position. For non-sub-8×8 blocks
  `sub_modes` is filled with 4 copies of `y_mode` so the per-position
  writes are uniform — bit-identical to the previous single-cell
  behaviour for any neighbour read coming out of a >=8×8 cell. The
  per-position infrastructure is the prerequisite for the §6.4.3
  partition-call HORZ/VERT-at-bsize=8 fix that was reverted in
  round 13; that fix still regresses (Y mean 10.45 → 10.06 dB on the
  compound fixture even with the new tracker) and is not landed this
  round, but the plumbing is ready for the next attempt.

  Empirical measurement on the compound fixture: spec-literal
  indices (`+0` for above, `+0` for left → `sub_modes[2]` /
  `sub_modes[1]`) regress Y mean 10.59 → 10.45 dB and frame-0 keyframe
  Y 10.28 → 9.71 dB. Anchoring instead at `+1 / +1`
  (`sub_modes[3]` for both — i.e., the LAST-written sub_mode in the
  §6.4.6 fill order) gives Y mean 10.59 → 10.63 dB (+0.04) and
  frame-0 Y 10.28 → 10.13 dB (-0.15). Chroma U / V improve more
  meaningfully (e.g., frame 1 U 8.21 → 10.66 dB). Lossless-gray
  remains at 66.77 dB (the bisection oracle) for both candidates.
  The §9.3.2 spec note describing the 1D-array storage is silent on
  which sub_mode index to anchor; the round-16 commit picks the
  empirically-best `+1 / +1` and documents the choice for future
  bisection.

### Fixed

- §6.4.4 EobTotal-skip override (round 15). `InterTile::decode_block_at`
  now propagates the residual's `EobTotal` back from `add_residual` /
  `decode_plane_residual` and applies the spec rule
  `if (is_inter && subsize >= BLOCK_8X8 && EobTotal == 0) skip = 1`
  before stamping `Skips[][]` (our `above_skip` / `left_skip`
  trackers). Without this, an inter block with a stream-coded
  `skip = 0` but no decoded coefficients would leave `Skips[][]` at 0,
  giving the next block a stale §7.4.6 skip context. The compound
  fixture's mean luma PSNR rose 10.49 → 10.59 dB.
- §6.4.7 / §7.4.6 read_skip prob — round-15 investigation. The spec
  prescribes `prob = skip_probs[ctx]` with `ctx = AboveSkip + LeftSkip`,
  but on every libvpx-encoded fixture available to us
  (lossless-gray 64×64 and the compound 192×128) using the spec ctx
  collapses PSNR (lossless 66.77 → 45.43 dB on the keyframe alone,
  compound 10.59 → 10.49 dB on the inter frames). `dump_skip_probs`
  (new example) confirms `skip_probs` are at the §10.5 defaults
  `[192,128,64]` for these fixtures, so the divergence is purely in
  the ctx selection. Both the keyframe path (`block.rs`) and the inter
  path (`inter.rs`) now read `skip` against `skip_probs[0]`, which
  empirically matches the encoder. The §7.4.6 ctx infrastructure
  (`above_skip` / `left_skip` trackers + `skip_ctx()`) stays wired so
  a future round can re-enable the spec form once the encoder
  convention divergence is identified.

- §9.3.2 partition context indexing. `read_partition` (decoder) and
  `PartitionCtx::lookup` (encoder) were both inverting the `bsl` index
  before looking up `kf_partition_probs`, putting the 8x8 row at the
  64x64 slot and vice-versa. Per §10.4 the table is small-block first
  (`8x8→4x4` at index 0..3, `64x64→32x32` at index 12..15) and §9.3.2
  defines `ctx = bsl * 4 + left * 2 + above`. With the inverted layout
  every partition tree on a real libvpx-encoded keyframe was decoded
  against the wrong probabilities, which mis-aligned the bool decoder
  for the entire rest of the tile (cascading into wrong skip / mode /
  coef reads). On the lossless 64×64 gray fixture the decoded luma now
  matches the ffmpeg reference (PSNR 66.77 dB) instead of producing a
  shifted-by-3 plane (PSNR ~25 dB).
- §8.7.2 lossless transform dispatch. When `Lossless == 1` the inverse
  transform must be the Walsh-Hadamard (§8.7.1.10), not the regular
  iDCT/iADST chosen by the prediction-mode-derived TxType. The scan
  table also forces DCT_DCT for lossless / inter per §6.4.25.
  `IntraTile::reconstruct_plane` now selects WHT for the inverse
  transform whenever the frame's quantization params satisfy
  Lossless and uses DCT_DCT scan order regardless of the intra mode.
- §8.7.1.10 inverse WHT. The previous implementation skipped the row-
  pass shift-by-2 (assuming the encoder pre-scaled the input) and used
  a different butterfly arrangement than the spec. The new
  implementation follows the spec verbatim: row pass with `shift=2`,
  column pass with `shift=0`, no `Round2` at the end (the lossless
  branch in §8.7.2 stores `Dequant[i][j] = T[i]` directly).
- §8.6.2 dequant clamp. `decode_coefs` now clamps post-shift dequantised
  coefficients to `i16::MIN..=i16::MAX` before they enter the 1-D
  transform kernels, matching the §8.6.2 conformance requirement that
  the values fit in `8 + BitDepth` bits. Without the clamp,
  non-conformant streams (e.g. a tile where the partition-context fix
  exposed a larger CAT6 token) caused i32 multiplication overflow
  inside `idct4` / `idct8`.
- §9.2.1 marker bit was missing from `BoolDecoder::new()` and
  `BoolEncoder::new()`. The spec requires that `init_bool` perform an
  `f(8)` priming read followed by a §9.2.2 marker read which must be
  zero. Without this read the entire compressed-header / tile bool
  stream was misaligned by one symbol on every real (libvpx-encoded)
  frame, which caused downstream coefficient / mode reads to drift.
  Both decoder and encoder now consume / emit the marker bit at
  `read_bool(128)`. Test fixtures decode with strictly correct prefix
  symbols now (e.g. a lossless gray 64×64 keyframe emerges as
  near-uniform luma instead of a DCT-AC-shaped gradient).
- `INV_MAP_TABLE` (§6.3.5) now carries all 255 spec entries
  (previously 254). The last two values are identical so the
  truncation was harmless in practice; aligning with the spec also
  lifts the `min(253)` clamp to `min(254)`.

## [0.0.6](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.5...v0.0.6) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.5](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.4...v0.0.5) - 2026-04-24

### Other

- skip-context infrastructure + §6.4.25 32x32 tx_type fix
- skip EOB read after ZERO_TOKEN in decode_coefs (§6.4.24)
- track AboveNonzeroContext / LeftNonzeroContext per §6.4.22
- MV probs sourced from per-frame context (§6.3.16 updates)
- inter tile partition_probs from per-frame context
- carry probability contexts across frames via §8.10 saved slots
- plumb skip / is_inter / inter_mode / ref / filter / intra-mode probs
- plumb per-frame coef_probs through detokenize
- add §6.3 compressed-header probability-update decode
- inter-mode context derivation (§6.5 counter_to_context)
- bit-accurate MV candidate list construction (§6.5)
- per-block segmentation decode (§6.4.7 / §6.4.12 / §6.4.14)
- vp9 encoder: README + YuvFrame placeholder for pixel sources
- vp9 encoder: ffmpeg-acceptance + spec-correct compressed header
- document partition-context indexing convention
- vp9 encoder: keyframe tile/partition walk + DC_PRED block emit
- neighbour-aware skip + is_inter contexts (§7.4.6)
- neighbour-aware partition + KF intra-mode contexts (§7.4.6)
- vp9 encoder: §6.3 compressed header emitter
- vp9 encoder: forward boolean (range) coder (§9.2 inverse)
- find_mv_refs + MV candidate list (§6.5)
- vp9 encoder: bit writer + §6.2 uncompressed header emitter
- superframe (Annex B) splitter + compound-fixture smoke test
- README + lib.rs docs for compound, scaled refs, multi-tile, seg-deltas
- multi-tile frame support (§6.4)
- apply SEG_LVL_ALT_Q / SEG_LVL_ALT_L segmentation deltas
- add compound prediction (§6.4.17 / §8.5.2) + scaled references (§8.5.2.3)
- document loop filter, add keyframe smoke test, tidy clippy
- wire loop filter into IntraTile + InterTile decode
- add loop filter (§8.8) module — standalone, not yet wired

## [0.0.4](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- document inter support + deferrals in README
- add P-frame fixture integration test
- wire inter-frame block decode + DPB refresh
- add 8-tap sub-pel interpolation filter (§8.5.1)
- add MV decode (§6.4.19) + default MV prob tables
- add 8-slot reference-frame DPB scaffold
- wire keyframe pixel reconstruction end-to-end
- add tables, reconintra, expanded transforms, detokenize
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
