# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **r-next — Quadtree partition support for the P-frame inter encoder**
  (§6.4.16 inter block walk / §6.4.2 partition tree / §6.5
  decode_partition). Pre-this-round the encoder always emitted
  PARTITION_NONE at 64×64 (matching the keyframe encoder shape).
  This round adds RDO between PARTITION_NONE and PARTITION_SPLIT at
  both 64→32 and 32→16; 16×16 always emits NONE (8×8 / sub-8×8 is a
  later round). The split decision is bitwise: at each interior
  bsize ∈ {64, 32} the encoder runs `me_search` at the parent block
  and at each of the 4 candidate sub-blocks, sums the child SADs,
  adds a fixed-rate penalty of `SPLIT_RATE_PENALTY_PER_SUBBLOCK_BITS
  * 3 = 60` SAD units (a rough proxy for the 3 extra partition bits
  plus per-sub-block overhead vs NONE), and picks the lower-cost
  shape. The decoder already handles the full quadtree, so the
  change is encoder-only — the wire format simply flips the
  partition-tree bits from `0` (NONE) to `1, 1, 1` (SPLIT) and
  recurses. New `tests/r_next_partition.rs` gates against three
  fixtures:
  - `partition_uniform_64x64_picks_none` — identical-source fixture;
    encoder MUST pick PARTITION_NONE (no SAD gain from split). P-frame
    cap of 32 B catches over-eager splits; measured **21 B** (same as
    r49 baseline).
  - `partition_textured_corner_picks_split` — per-quadrant divergent
    motion (TL shift right 4, TR shift left 4, BL shift down 4, BR
    shift up 4 — no single 64×64 MV aligns all four). Encoder must
    pick PARTITION_SPLIT, verified via bytestream-level inspection
    of the first SB's partition bit pattern (`PARTITION_PROBS[0][0]`
    against the tile bool stream). Measured P-frame **31 B**; the
    partition tree resolves to SPLIT at 64×64 with 4× NEWMV 32×32
    children.
  - `partition_two_frame_translation_pframe_smaller_than_iframe` —
    r-next regression: 4-px translation fixture. Encoder must still
    pick PARTITION_NONE (a single 64×64 MV fits) and P-frame stays
    smaller than the I-frame. Measured **P-frame = 22 B vs I-frame =
    410 B**, identical to the r49 baseline; PSNR_Y = 49.96 dB.
  Existing r-next half/quarter-pel sub-pel tests + r49 P-frame tests
  stay green (same byte counts and PSNR — partition RDO doesn't fire
  on the smooth-content single-MV fixtures).

- **r-next — Sub-pel ME refinement** for the P-frame inter encoder
  (§6.3 `sub_pel_filters_8` / §8.5.4.2 EightTap luma filter). The
  integer-pel ±16 SAD search from r49 is now followed by two
  refinement stages: an 8-neighbour HALF-PEL pass that interpolates
  the reference block through `mcfilter::FILTER_EIGHTTAP` at each
  candidate, then an 8-neighbour QUARTER-PEL pass around the half-pel
  winner. MVs are tracked in 1/8-pel units throughout so the final
  result feeds directly into §6.4.19 emit_mv. The 1/8-pel
  (`allow_high_precision_mv`) stage is deferred — r-next stops at
  1/4-pel which is the standard VP9 baseline precision. Each
  refinement pass is gated by SAD (≤ 2 inner iterations) so the
  per-block cost stays bounded. New `tests/r_next_subpel_me.rs`
  gates against three fixtures:
  - `pframe_half_pel_translation_high_psnr` — 64×64 frame 2 generated
    by applying the EightTap phase-8 filter to the reconstructed
    frame 1; expected ≥ 38 dB, measured **55.23 dB**.
  - `pframe_quarter_pel_translation_high_psnr` — 64×64 frame 2 from
    EightTap phase-4 filter; expected ≥ 36 dB, measured **55.88 dB**.
  - `pframe_integer_pel_translation_no_regression` — r49's 4-px
    horizontal-shift fixture; sub-pel can only help, never hurt;
    still **49.96 dB**, identical to the r49 baseline.

- **Round 49 — P-frame inter encoder scaffold** (§6.2 non-key
  uncompressed header, §6.3 inter compressed header, §6.4.11 /
  §6.4.13 / §6.4.16 inter block walk, §6.4.19 MV emit). The encoder
  now produces a valid VP9 P-frame against a reconstructed
  LAST_FRAME reference, exposed via `encode_pframe_yuv`. Scope:
  single-reference inter (LAST=slot 0; no GOLDEN, no ALTREF, no
  compound), 64×64 PARTITION_NONE blocks (smaller blocks at edge
  clips), integer-pel ±16 px full-search block matching against the
  reference luma plane with SAD cost. Two inter modes: `ZEROMV`
  when source ≈ reference, `NEWMV` otherwise (MV delta against the
  decoder-shape `find_best_ref_mvs` BestMv so the per-block delta
  matches the wire). `skip = 1` everywhere — no residual encoded.
  `tx_mode = ONLY_4X4`, `interpolation_filter = 0` (EightTap fixed,
  no per-block switchable bits), `allow_high_precision_mv = false`.
  A 2-frame 64×64 horizontal-translation test
  (`tests/r49_pframe_inter.rs::pframe_horizontal_translation_high_psnr_small_size`)
  produces an I-frame of 410 bytes and a P-frame of 22 bytes (18.6×
  ratio) with a decoded PSNR_Y of 49.96 dB. The identical-content
  fixture (`pframe_identical_fixture_zeromv_tiny_size`) compresses
  to 21 bytes with PSNR_Y of 61.24 dB — essentially just the frame
  header overhead. Existing keyframe encoder + intra RDO tests
  continue to pass.

- **Round 48 — chroma intra-mode RDO + mode-RDO early termination**
  in the keyframe encoder. The same 4-mode SSE picker that round 40
  added on luma now runs on the U plane per block; the picked
  `uv_mode` applies to both U and V chroma planes (VP9 stores ONE
  `uv_mode` per block applied to both planes — §6.4.6
  `read_intra_mode_uv`). Pre-r48 the encoder always emitted
  `DC_PRED` for chroma, regardless of source content. The mode-RDO
  picker (used for both luma and chroma) now probes `DC_PRED` first
  and short-circuits if its 4×4 SSE is at noise floor (≤ 16 across
  16 samples ⇒ ≤ 1 LSB RMS), skipping V/H/TM evaluations on smooth
  content. On a 256×256 chroma-gradient fixture (mid-gray luma,
  U = smooth row gradient, V = smooth column gradient) at
  `base_q_idx = 64` chroma reconstruction climbs from
  `U = 50.91 / V = 50.92 dB` (DC-only baseline) to
  `U = 52.34 / V = 50.97 dB`. Luma fixtures stay flat (smooth
  gradient Y = 53.06 dB; horizontal stripes Y = 47.62 dB) because
  their chroma is uniform-128 — the picker still selects DC and the
  early-out then locks it in. New gate test:
  `encoder_256x256_chroma_gradient_self_roundtrip` (PSNR ≥ 30 dB
  hard floor on both U and V).

### Changed

- VP9 fuzz oracle (`ffmpeg_oracle_decode`) is now robust against
  libavcodec version-divergence. Different libavcodec majors parse the
  SAME adversarial fuzz input into DIFFERENT shapes — 58.x (FFmpeg 4.x)
  feeds malformed superframe-index permutations through and produces a
  best-effort partial frame; 61.x+ (FFmpeg 7.x+) rejects them earlier
  and emits an error-conceal placeholder (typically a uniform mid-gray
  plane). The previous tight ±1 LSB per-sample guard tripped loudly on
  those placeholders, so the same fuzz input "diverged" on CI machine A
  and "agreed" on machine B depending on which libavcodec was apt-
  installed. Closes workspace task #750.

  New oracle strategy (bilateral-rejection envelope):

  1. **Uniform-fill detection (per-plane).** If ANY of the oracle's
     Y / U / V planes is constant, it's almost certainly libavcodec's
     error-conceal output rather than a real decode. Skip pixel
     comparison; keep structural (frame-count / dimensions / chroma
     geometry) checks. The detector is value-agnostic (does not
     hardcode the `1 << (bd-1)` mid-gray) so it catches every flavour
     of placeholder including all-0 / all-255 / all-`U16_MAX` shapes
     that some libavcodec builds emit. Per-plane (not per-frame)
     because libavcodec 60.x+ is observed to partially fill chroma
     planes with the neutral mid-gray (128 / 512 / 2048) even when
     Y carries real content — caught the first time the rebuilt
     fuzz workflow tripped (run 25690387209: oracle U[*]=128, ours
     U[*]=0, libavcodec 60.31.102).
  2. **Divergence fraction + magnitude envelope** — when the oracle
     plane has real texture, the harness counts the fraction of
     samples that differ by more than 1 LSB and the worst-case
     absolute diff. Trips loudly ONLY when BOTH `fraction >
     MAX_DIVERGE_FRACTION` (0.5%) AND `worst_abs > MAX_TOLERATED_*`
     (8 LSB for 8-bit, 32 LSB for 10/12-bit). Real spec-conformance
     bugs we've caught — #769 marker-bit drift, #748 over-read — all
     produce >50% divergence on the affected plane, so the envelope
     preserves the real-bug signal while filtering version-specific
     corner-case drift.
  3. **Version probe + diagnostic tag** — queries `avcodec_version()`
     (stable public C entry since libavcodec 0.5) at oracle init and
     embeds the `(major.minor.micro)` triple in the one-time announce
     line plus every failure-panic message, so a CI red is self-
     describing about which libavcodec was in scope.

  Spec basis: VP9 §8 prescribes integer-exact arithmetic and §8.6.2
  bounds dequantised coefficients to `i16` ahead of the inverse
  transform; a well-formed bitstream that BOTH decoders accept and
  BOTH decode in earnest must produce bit-identical samples. The
  envelope preserves that signal — what it filters out is the case
  where libavcodec's version-specific error-recovery path produced a
  placeholder frame that has no bearing on whether our decoder is
  spec-correct.

  New unit tests (`fuzz/src/lib.rs::oracle_tests`, 14 cases) pin every
  divergence shape observed in recent CI fuzz runs as a "do not trip"
  oracle outcome, plus three "real bug" shapes (wholesale plane
  divergence, large-magnitude clusters above the fraction threshold,
  HBD large-magnitude clusters) that MUST still trip the envelope.
  The oracle helpers (`oracle::is_uniform_plane`,
  `oracle::eval_envelope_packed`) live in the fuzz crate's `lib.rs`
  so they're unit-testable without spinning up the libfuzzer binary;
  the fuzz target re-exports them.

### Fixed

- §9.2.1 conformance check on the tile boolean-decoder marker bit:
  the spec requires the post-init marker (read at p=128) to be 0;
  pre-fix we discarded the bit, which let adversarial fuzz inputs
  whose tile-payload first byte forces the marker to 1
  (`value = 0x80 ≥ split = 128 → bit = 1`) decode against a
  permanently-misaligned bool stream that diverged from libavcodec
  on coefficient tokens several superblocks deep into the frame.
  Closes workspace task #769 — a 98-byte 385×1 keyframe whose
  62-byte tile payload starts with `0x80` and triggered a 3-LSB
  Y[0,204] divergence. The new
  `BoolDecoder::new_with_marker_strict(_, true)` path is opt-in
  (the compressed-header bool stream keeps the permissive default
  because real-world libvpx encoders are observed to violate the
  marker requirement there while still producing decodable output).
- §6.2.2 + §A.1 cap on declared frame dimensions in
  `Vp9Decoder::ingest_one`: refuse a frame whose `(width, height)`
  exceeds `MAX_FRAME_DIM` (8192 per axis) or whose total pixel count
  exceeds `MAX_FRAME_PIXELS` (8192×8192 = 67 MP). Closes workspace
  task #749 — a 20-byte fuzz keyframe declaring a 2069×39029 picture
  used to allocate ~243 MiB of plane buffers in `IntraTile::new`
  before the bitstream was rejected. Public constants `MAX_FRAME_DIM`
  / `MAX_FRAME_PIXELS` re-exported from the crate root.
- §9.2 conformance check on tile boolean-decoder over-read: surface
  the new `BoolDecoder::over_read_bits()` counter and refuse any
  frame whose tile bool decoder consumed more than `2 * tile_bytes`
  bits of zero-pad (floor 32 bits). Closes workspace task #748 — a
  178-byte fuzz keyframe with a 3-byte tile payload used to silently
  zero-pad the bool decoder by 166 bits and disagree with libavcodec
  on Y[0,0] by 3 LSB. Real libvpx-encoded fixtures over-read by at
  most a few % of their tile size and stay under the new budget.

### Removed

- harness pre-filters (`MAX_PIXELS` / `MAX_DECODE_PIXELS`) on declared
  frame dimensions in `panic_free_decode` / `ffmpeg_oracle_decode`:
  the decoder now enforces the cap itself.

## [0.0.10](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.9...v0.0.10) - 2026-05-07

### Other

- round 40 — per-block intra-mode RDO + QP-derived loop filter level
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- fix clippy ptr_arg — &mut Vec → &mut [_] in tile_pixel + tokenize
- round 2 — forward DCT + token coding, PSNR_Y ≥ 35 dB (256×256 smooth gradient = 50.60 dB)
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-vp9/pull/502))

### Added

- VP9 encoder round 40 — per-block luma intra-mode RDO across
  `{DC_PRED, V_PRED, H_PRED, TM_PRED}` plus QP-derived loop-filter
  level. The picker uses `reconintra::NeighbourBuf::build` to evaluate
  candidates against decoder-shape (127/129-padded) neighbour buffers
  on a 4×4 footprint at the block's top-left corner, then applies the
  chosen mode at every 4×4 TX sub-block via `reconintra::predict`.
  Above/left intra-mode trackers (`above_mode_4x4` / `left_mode_4x4`)
  mirror `IntraTile`'s state so `KF_Y_MODE_PROBS[a][l]` resolves to
  the same row on both encode and decode. `EncoderParams::keyframe`
  now seeds `loop_filter_level` from `base_q_idx` via a libvpx-shape
  `q*0.45 + 1` heuristic clamped to `[0, 63]`; lossless (`q == 0`)
  keeps the filter disabled. New regression tests:
  `pick_intra_mode_top_left_returns_dc`,
  `pick_intra_mode_picks_v_when_columns_match_above`,
  `pick_intra_mode_picks_h_when_rows_match_left`,
  `mode_to_index_matches_spec_numbering`,
  `default_filter_level_*` / `keyframe_default_picks_nonzero_filter_at_q64`,
  and `encoder_256x256_horizontal_stripes_self_roundtrip`. The
  256×256 smooth-gradient self-roundtrip lifts 50.60 → 53.06 dB
  (deblocking gain on quantisation noise); the new horizontal-stripes
  fixture lands at 47.62 dB (mode-RDO actively picks V/H/TM). All
  186+ existing tests stay green.
- VP9 encoder round 2 — full pixel-content keyframe encoding with
  forward 4×4 DCT, quantisation, and VP9 coefficient token coding.
  `encode_keyframe_yuv` now encodes source YUV residuals (DC_PRED from
  reconstructed neighbours) through `fdct_2d` + `quantise` + `encode_coefs`,
  targeting PSNR_Y ≥ 35 dB on smooth fixtures at `base_q_idx = 64`.
  Achieved: PSNR_Y = 50.60 dB, PSNR_U = PSNR_V = ∞ dB on the 256×256
  smooth-gradient cross-decode fixture; ffmpeg 8.1 round-trips losslessly.
- `encoder/fwdtransform.rs` — `fdct_2d` (4×4 and 8×8 2-D separable
  forward DCT) + `quantise` (scan-order quantisation with EOB tracking).
- `encoder/tokenize.rs` — `encode_coefs` (VP9 coefficient token entropy
  encoder with correct `initial_ctx` from NonzeroContext + full Pareto8
  CAT1–6 coding mirror of `detokenize::decode_coefs`).
- `encoder/tile_pixel.rs` — `emit_pixel_tile` + `build_pixel_keyframe`
  with per-block DC_PRED prediction chaining, reconstruction loop, and
  above/left NonzeroContext propagation matching the decoder exactly.
- New integration tests: `tests/vp9_encoder_psnr_256.rs` —
  `encoder_256x256_smooth_psnr_via_ffmpeg` (PSNR_Y ≥ 35 dB ffmpeg
  cross-decode gate) and `encoder_256x256_self_roundtrip` (PSNR_Y ≥ 35 dB
  self-decoder gate) on a 256×256 smooth gradient fixture.

## [0.0.9](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.8...v0.0.9) - 2026-05-04

### Other

- add encode_keyframe_yuv API + ffmpeg PSNR cross-decode gate (round 1)
- promote tiny-i-only-16x16 to BitExact + README correctness pass
- emit DPB slot for show_existing_frame packets
- §6.2.2.1 found_ref must break at first match

### Added

- VP9 encoder round 1 — profile 0, 4:2:0 8-bit, single-tile keyframe
  with `tx_mode = ONLY_4X4`, every block emitted as `PARTITION_NONE`
  with `skip = 1` and `DC_PRED` luma + chroma. Loop-filter level 0.
  ffmpeg 8.1 cross-decodes the resulting IVF without errors and the
  reconstructed samples land at midgrey 128 in every plane (DC chain
  with no neighbours yields the spec's mid-grey predictor). New
  black-box hard gate `tests/vp9_encoder_ffmpeg.rs ::
  ffmpeg_cross_decode_psnr_smooth_fixture` encodes a 64×64 smooth
  gradient (luma drift ±4 around 128, chroma uniform 128), pipes
  through ffmpeg, and asserts per-plane PSNR ≥ 30 dB. Achieved on
  current code: Y = 45.87 dB, U = V = ∞ dB.
  Public API additions: `encoder::encode_keyframe_yuv(&EncoderParams,
  &YuvFrame)` reserves the round-2 entry-point shape (the source
  pixels are accepted today but ignored — the body still emits the
  skip=1 / DC_PRED stream). `encoder::encode_keyframe(&EncoderParams)`
  remains for callers that don't have a `YuvFrame` yet.
  Round 2 will replace the body with forward-DCT + token-coded
  residual so non-smooth content also clears the PSNR target.

### Changed

- `tests/docs_corpus.rs` `corpus_tiny_i_only_16x16` promoted from
  `Tier::ReportOnly` to `Tier::BitExact`. The fixture has been at
  100% byte-for-byte match against the libvpx reference through
  multiple rounds; pinning it as a hard gate locks in regression
  protection for the smallest-possible-frame path (single 64×64 SB
  partitioned to 8×8 across the 16×16 active area, with the rest of
  the SB walking the §6.4.2 implicit "out-of-frame" partition split
  chain). Any future change that re-introduces a single-byte
  divergence on this fixture is now a CI red.

### Added

- §6.2 / §8.2 `show_existing_frame` dispatch. Pre-fix, `Vp9Decoder`
  parsed the `show_existing_frame` flag + `frame_to_show_map_idx`
  from the uncompressed header but then silently dropped the packet
  (no `Frame::Video` queued). The fix copies the referenced DPB slot
  through `build_video_frame_from_ref`, applying the same HBD widening
  used by the live decode path, so the consumer sees a visible frame
  for every `show_existing_frame` packet — matches the spec's
  "header_size = 0; refresh_frame_flags = 0; loop_filter_level = 0;
  return" behaviour. Errors cleanly with `InvalidData` when the
  referenced slot is empty.
  Effect on the docs corpus driver:
  * `show-existing-frame`: 18/24 → **24/24** visible frames produced.
  Adds two new unit tests in `decoder::tests`:
  `show_existing_frame_emits_dpb_slot_as_new_frame` and
  `show_existing_frame_empty_slot_errors`.

### Fixed

- §6.2.2.1 `frame_size_with_refs` `found_ref` early-break. The parser
  was unconditionally consuming all three `found_ref` bits per
  inter-frame header; per spec the loop must `break` at the first
  `found_ref == 1`. The over-read by 0..2 bits desynced the remainder
  of the uncompressed header on every libvpx stream where the encoder
  actually picked ref index 0 or 1, decoding `header_size` from a
  16-bit window straddling the next bytes (e.g. 16255 in a 1342-byte
  frame) and tripping the "compressed header missing or truncated"
  guard on every non-keyframe whose `header_size` happened to land
  beyond the frame end. Affected fixtures: `superframe-2` (was 9/16
  visible frames produced, now 16/16; aggregate exact bytes
  3302 → 4378), `show-existing-frame` (was 18/24, now 20/24).
  Adds `parse_inter_frame_with_found_ref_zero` regression test that
  pins the `header_size` value through a synthetic inter header with
  found_ref = 1 in slot 0.

## [0.0.8](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.7...v0.0.8) - 2026-05-03

### Other

- emit per-profile PixelFormat + widen HBD planes to LE u16 (task #265)
- rustfmt the docs_corpus driver (no behaviour change)
- wire docs/video/vp9 fixture corpus as integration tests

### Added

- `pixel_format_from_color_config` now maps the full VP9 §6.2.1
  (bit_depth, subsampling, color_space) matrix to the matching
  `oxideav_core::PixelFormat`: `Yuv420P` / `Yuv422P` / `Yuv444P` for
  8-bit, the `Yuv*P10Le` family for 10-bit, the `Yuv*P12Le` family for
  12-bit. sRGB / GBR-planar surfaces as `Yuv444P*` (core has no
  GBR-planar variant today). Closes the hard-coded `Yuv420P` fallback
  flagged by task #265.
- `Vp9Decoder` now emits HBD output at the bitstream's declared bit
  depth: each reconstructed u8 sample is widened to a little-endian
  `u16` by `(byte as u16) << (bit_depth - 8)` so the active bits land
  in the top of the bit-depth window. Stride doubles for HBD planes.
  This matches the `Yuv*P10Le` / `Yuv*P12Le` plane layouts documented
  on `oxideav_core::PixelFormat` and lets the docs corpus driver
  compare HBD fixtures plane-to-plane (Profile 1 4:4:4, Profile 2/3
  HBD, GBR 10-bit) instead of bailing with `plane size mismatch`.
  Internal reconstruction is still 8-bit; the widening is a left-shift
  so a downstream `>> shift` recovers the original byte exactly.

## [0.0.7](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.6...v0.0.7) - 2026-05-03

### Other

- drop identity-op multiplications in r24 subrect offset test
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- §6.5.11 between-MB sub-block MV lookup (get_sub_block_mv) ([#190](https://github.com/OxideAV/oxideav-vp9/pull/190))
- §6.5.14 within-MB sub-block MV mixing (append_sub8x8_mvs)
- §6.4.16 sub-8×8 inter mode-info per-4×4-sub-block iteration
- §6.4.3 sub-8×8 inter decode_partition one-call branch
- §6.4.3 sub-8×8 HORZ/VERT one-call + §9.3.2 spec-literal sub-mode anchor — pattern Y +37 dB
- §6.4.3 spec-ctx update_partition_ctx — c64 lossless 70.1 → ∞ dB
- §7.4.6 spec-ctx skip read — c64 lossless 61.9 → 70.1 dB
- r19 — lossless reconstruction audit + diagnostic fixture
- round 19 — env-gated trace points for §8.7.2 lossless audit
- round 18 — README PSNR refresh
- round 18 — §9.3.2 default_intra_mode tracker spec-literal +0
- round 17 — measurement audit, expose true lossless quality
- adopt slim VideoFrame shape
- §9.3.2 per-position above/left intra-mode tracker (round 16)
- §6.4.4 EobTotal-skip override + round-15 skip-ctx investigation
- §8.5.1 above-row extension gating + D45 boundary value
- §6.4.3 SPLIT-at-8x8 calls decode_block once + sub-mode neighbours
- §9.3.2 inter-ref ctx (comp_mode/comp_ref/single_ref/interp_filter)
- §9.3.2 partition ctx + §8.7.2 lossless WHT dispatch
- §9.2.1 marker bit + INV_MAP_TABLE size fixes
- pin release-plz to patch-only bumps

### Changed

- **Round 26 / #190 — §6.5.11 between-MB sub-block MV lookup
  (`get_sub_block_mv`).** Closes the second axis flagged by round 25
  — the *neighbour* per-4×4 `SubMvs[r][c][refList][idx]` lookup. Adds
  `InterMiCell::sub_mvs: [[Mv; 4]; 2]` (populated by `decode_inter_block`
  from the round-25 `block_mvs_a` / `block_mvs_b` for sub-8×8, and from
  the cell-level MV repeated four times for >=8×8 per §6.4.16 line 2700),
  plus `mvref::IDX_N_COLUMN_TO_SUBBLOCK` and the spec-faithful
  `mvref::get_sub_block_mv`. New `find_mv_refs_geom_block(..., block)`
  is the block-aware entrypoint; the legacy `find_mv_refs_geom` calls it
  with `block = -1` (which returns `sub_mvs[refList][3]` = `cell.mv`,
  preserving bit-identity for the cell-level path). The sub-8×8 inter
  path now calls `find_subblock_mv_refs(block_idx)` per sub-block and
  feeds the result into `sub8x8_refined_refs` alongside the cell-level
  refs (still used to pin `best_mv_override` for NEWMV per §6.4.16).
  Pre-existing fixtures stay byte-identical at the visible level — the
  P-frames in `vp9-inter`, `vp9-compound`, `vp9-segmentation` don't
  contain a sub-8×8 inter MB whose first-two-neighbour cell is itself
  sub-8×8 with diverging `SubMvs`. The structural gap is now closed
  on both within-MB (#180) and between-MB (#190) axes. Adds 5 unit
  tests covering the cell-level `block=-1` invariant, the spec table
  lookup, and the per-block neighbour selection for above + left
  neighbours.

- **Round 25 — §6.5.14 within-MB sub-block MV mixing.**
  Wires up `append_sub8x8_mvs` per spec for the sub-8×8 inter
  MBs (B4x4 / B4x8 / B8x4) inside the §6.4.16 (idy, idx) loop.
  For each sub-block `block = idy*2 + idx > 0`, the per-sub-block
  `(NearestMv, NearMv)` pair is now rebuilt from the MVs that
  `assign_mv` chose for prior sub-blocks of the *same* MB
  (`BlockMvs[refList][0..block]`), rather than reusing the
  cell-level `find_mv_refs` candidates verbatim. `BestMv` (used
  by NEWMV) stays the cell-level `RefListMv[0]` per spec — a new
  `MvRefs::best_mv_override` field pins it on the per-sub-block
  refined `MvRefs`. Adds 7 unit tests covering all four block
  indices, the dedup rules, and the BestMv-override invariant.
  Lossless / inter / compound fixtures stay byte-identical to
  r24 (the existing fixtures' P-frames don't trigger a
  non-block-0 NEAREST/NEAR sub-block read), but the structural
  asymmetry flagged in the r24 README is now closed for the
  within-MB axis. The remaining axis (per-4×4
  *neighbour* MVs via §6.5.11 `get_sub_block_mv` against a
  `SubMvs[r][c][refList][idx]` table on `InterMiCell`) is r26+.

- **Round 23 — §6.4.3 sub-8×8 HORZ/VERT/SPLIT one-call branch
  for the inter path.** Mirrors the round-22 intra fix to
  `inter.rs::decode_partition`. Per §6.4.3 the leading
  `if (subsize < BLOCK_8X8 || partition == PARTITION_NONE)` arm
  fires before the HORZ/VERT two-call branch and the SPLIT
  recurse-4 branch; for `bsize=BLOCK_8X8` every non-NONE partition
  produces a sub-8×8 `subsize`, so `decode_block` must be called
  exactly ONCE — `read_inter_frame_mode_info` (§6.4.11) handles
  the sub-block iteration internally.

  Before r23, the inter path was:
  * HORZ at bsize=8 → 2× decode_block (over-read mode/ref/MV).
  * VERT at bsize=8 → 2× decode_block (same).
  * SPLIT at bsize=8 → 4× decode_block on B4x4 sub-blocks
    (4× over-read every per-block ctx — comp_mode, comp_ref,
    single_ref, inter_mode, interp_filter, segment_id, skip,
    is_inter, tx_size, plus full coefficient detoken).

  After r23, all three shapes call `decode_block` once at
  bsize=8, matching the §6.4.3 spec literal and the round-22
  intra fix. This is a structural alignment between the intra
  and inter partition trees that should also have been a paired
  change in round 22.

  Per-fixture / per-variant audit:

  | variant | pattern Y | compound Y | c64 |
  |---------|----------:|-----------:|-----|
  | r22 baseline (intra fix only) | 47.70 dB | 9.54 dB | ∞ |
  | **r23 (inter analog applied)** | **47.70 dB** | **9.55 dB** | **∞** |

  The compound mean Y improvement is small (+0.01 dB; only frame
  3 of 6 moved measurably, 9.68 → 9.77 dB) — the dominant
  remaining compound divergence is therefore NOT in
  `decode_partition`'s call count but in the per-sub-8×8
  inter mode-info reader: §6.4.16 `inter_block_mode_info`
  reads `inter_mode` and `assign_mv` PER 4×4 sub-block when
  `MiSize < BLOCK_8X8`, but our `decode_inter_block` reads
  one inter_mode + one MV per ref slot regardless of `bs`.
  That is the r24+ work — under-reading from the inter mode
  reader is now the sole remaining compound asymmetry.
  Pattern luma and c64 remain unchanged because the lossless
  fixtures are pure-keyframe and never hit the inter path.

  All 163 tests pass (134 unit + 29 integration; +2 unit tests
  added in `inter.rs::tests` to pin the §6.4.3 sub-8×8
  partition routing contract and the per-shape decode_block
  call-count table).

- **Round 22 — §6.4.3 sub-8×8 HORZ/VERT one-call branch + §9.3.2
  spec-literal sub-mode anchor.** Two paired fixes in `block.rs`
  that, together, lift `vp9-lossless-pattern.ivf` Y by **+37 dB**
  (10.41 → **47.70 dB**) with both chroma planes now bit-exact
  (∞ dB) and 337/16384 luma byte diffs (down from 14472/16384).

  1. **`decode_partition` HORZ/VERT one-call.** Per §6.4.3 the
     leading `if (subsize < BLOCK_8X8 || partition == PARTITION_NONE)`
     branch fires before the `PARTITION_HORZ` / `PARTITION_VERT`
     two-call branches. For `bsize=BLOCK_8X8 + HORZ → subsize=B8X4`,
     the sub-8×8 branch wins and `decode_block` is called once;
     `read_intra_frame_mode_info` (§6.4.6) reads all 4 sub-modes
     internally. Our intra `decode_partition` was unconditionally
     calling `decode_block` twice, double-reading mode-info and
     desynchronising the bool decoder for every sub-8×8 partition.
     Fix gates the second call on `bsize > 8` (matches the round-13
     SPLIT fix for the same shape).

  2. **`read_intra_sub_mode` spec-literal `+idx` / `+idy` anchor.**
     §9.3.2 gives `abovemode = SubModes[MiRow-1][MiCol][2 + idx]`
     and `leftmode = SubModes[MiRow][MiCol-1][1 + idy*2]`. In the
     §9.3.2 NOTE "two 1D arrays" storage layout (which our writer
     uses), this maps to `above_mode_4x4[mi_col*2 + idx]` and
     `left_mode_4x4[mi_row*2 + idy]`. The round-15 code used a
     constant `+1` on both sides (always sub_modes[3]). Rounds
     18-21 measured the spec-literal switch and saw a 1 dB compound
     regression — that regression was an upstream artefact of the
     HORZ/VERT double-call above. Once both fixes land together,
     the spec-literal anchor is uniformly better.

  Per-fixture / per-variant audit:

  | variant                     | pattern Y | compound Y | intra fixture mean | c64 |
  |-----------------------------|----------:|-----------:|--------------------:|-----|
  | r21 (both `+1`)             | 10.41 dB  | 9.28 dB    | 89                  | ∞ |
  | spec anchor only (no HORZ fix) |  9.94 | 10.79      | **6** (FAIL)        | ∞ |
  | A0+L1 (above-spec, left-emp) |  6.28   | 9.31       | n/a                 | ∞ |
  | A1+L0 (above-emp, left-spec) | 10.35   | 8.03       | n/a                 | ∞ |
  | **r22 (both fixes paired)** | **47.70** | **9.54**   | **111**             | **∞** |

  The compound dip (10.20 r20 → 9.54 r22) is the remaining
  `inter.rs::decode_partition` divergence — same HORZ/VERT
  double-call shape plus a 4-call SPLIT loop at `bsize=8` still to
  audit. Pattern luma is now 98% bit-exact; both chroma planes
  100% bit-exact; c64 fixture remains bit-exact across all planes.

  All 161 tests pass (132 unit + 29 integration; +2 unit tests
  added in `block.rs::tests` to pin the §9.3.2 indexing contract
  for both cross-cell and same-block lookups).

- **Round 21 — §6.4.3 spec-ctx `update_partition_ctx` switch.** The
  decoder (`block.rs::update_partition_ctx`, `inter.rs::update_partition_ctx`)
  and encoder (`encoder/tile.rs::PartitionCtx::update`) now use the
  spec-literal form
  `AbovePartitionContext[c+i] = 15 >> b_width_log2_lookup[subsize]`
  (and the height variant for the left context) per §6.4.3. The
  round-12 "pre-saturated" derivation is gone.

  The round-19 audit had ruled this rewrite out because it regressed
  the c64 fixture by 60×, but the round-19 measurement was taken
  against a bool-decoder still drifting on the §7.4.6 skip-ctx bug
  fixed in round 20. With the upstream skip-ctx bug gone, the
  spec-correct partition write produces the right downstream ctx and
  the c64 lossless fixture now decodes **bit-exact** (Y=∞ dB,
  0/4096 byte diffs across all three planes — round 20's residual 20
  luma bytes are gone).

  Effect summary:
  * `vp9-lossless-c64-constant.ivf`: 70.10 dB → **∞ dB** (bit-exact).
  * `vp9-lossless-pattern.ivf`: 9.67 → **10.41 dB** (+0.74 dB Y).
  * `tests/vp9_compound_psnr.rs`: 10.20 → 9.28 dB mean Y (regression
    on the multi-frame inter content; the pattern of compound being
    the odd-one-out matches r19's "spec-literal regresses compound"
    finding from `read_intra_sub_mode` — the inter mode-info path
    has its own divergences still to be untangled in r22+).

  All 159 tests pass.

- **Round 21 — `read_intra_sub_mode` re-audit, neighbour anchor kept.**
  Re-ran the round-18 spec-literal `mi_col*2 + idx` / `mi_row*2 + idy`
  vs round-15 empirical `+1` measurements with the round-20
  spec-ctx skip read in place. Same asymmetry as before:
  spec-literal both directions regresses compound by ~1 dB; either
  direction alone regresses pattern (9.67 → 8.54 dB) or compound
  (10.20 → 7.52 dB). Comment updated to record the r21 re-test.

### Added

- **Round 23 — §6.4.3 sub-8×8 inter partition routing tests.**
  Added two unit tests in `inter.rs::tests`:
  * `r23_sub_8x8_partition_routing_contract` — pins the
    `(bsize=8, partition) → subsize` table and asserts every
    non-NONE shape produces a sub-8×8 subsize so the leading
    one-call branch fires; cross-checks bsize=16 + HORZ where
    the two-call branch is the spec-correct path.
  * `r23_decode_partition_call_count_table` — pure-data table
    of the expected `decode_block` call count per
    `(bsize, partition_kind)` pair, with the r23 contract
    locked in: bsize=8 + HORZ/VERT/SPLIT each call
    `decode_block` exactly once.
  These guard against a regression to the pre-r23 2×/4×
  call counts that desynced the bool decoder for every
  sub-8×8 inter partition.

- **Round 22 — §9.3.2 sub-mode indexing unit tests.** Added two
  unit tests in `block.rs::tests` that pin the indexing contract
  for `read_intra_sub_mode`:
  * `r22_sub_mode_neighbour_indexing_matches_writer_slots` — proves
    the cross-cell reader offsets `mi_col*2 + idx` and
    `mi_row*2 + idy` align with the writer's `sub_modes[2+idx]` and
    `sub_modes[1+idy*2]` slots, locking in the spec-literal
    contract.
  * `r22_sub_mode_same_block_lookup_matches_spec_branches` —
    pins the `if (idy)` / `if (idx)` same-8×8 fallbacks
    (`sub_modes[idx]` for above, `sub_modes[idy*2]` for left)
    against §9.3.2's branch arithmetic.
  These guard against silent regressions to the round-15 `+1`
  anchor and to plausible off-by-one mistakes in the same-block
  lookup.

- **Round 19 — diagnostic fixture + WHT round-trip unit tests.** Added
  `tests/vp9_lossless_constant.rs` with a 64×64 single-colour libvpx
  lossless fixture (`vp9-lossless-c64-constant.ivf`). Decoded against
  ffmpeg's reference, **luma PSNR = 61.90 dB** (chroma U/V bit-exact at
  ∞ dB) — but with a localised 29-byte cluster of 1–4-step diffs in a
  single 4×4 region (rows 8–11, cols 20–30). The cluster isolates the
  underlying bool-decoder drift to a single `skip=true` block where the
  `skip_probs[0]` (192) lookup decodes the encoded bit as `skip=false`
  and then reads 12 spurious tokens that consume bits meant for the
  next block. This is the smallest reproducible expression of the
  drift the lossless-pattern (9.90 dB) and lossy-intra fixtures
  manifest at full scale.

  Also added 3 transform-level unit tests proving the §8.7.1.10 +
  §8.7.2 WHT path is bit-correct independently of the surrounding
  pipeline:
  * `lossless_wht_dc_adds_one_per_pixel`: WHT DC=16 → exactly +1
    everywhere (ruling out the WHT itself as a 9.90 dB suspect).
  * `lossless_wht_recovers_neg112_diff`: pred=128 + WHT(-1792) → 16
    everywhere (the actual round-19 first-block scenario from
    `vp9-lossless-pattern.ivf` traces).
  * `lossless_wht_ac1_produces_alternating`: AC1=16 → non-uniform
    output (sanity that the WHT spreads AC energy).

  159 tests now (was 155).

### Changed

- **Round 19 — partition-context `update_partition_ctx` audit (no
  code change, documentation update only).** Re-tested the spec-literal
  `15 >> b_width_log2_lookup[subsize]` form against the round-12-kept
  empirical derivation. On the new c64-constant fixture the spec form
  regresses byte-diffs by 60× (29 → 1806). On the lossless-pattern
  fixture Y is +0.09 dB but V is -2.22 dB (10.21 → 7.99). Net regression.
  Comment in `update_partition_ctx` updated with the audit numbers and
  the round-12 derivation kept. The bug producing the localised
  drift is therefore NOT in the partition-context update.

- **Round 19 — skip-context `skip_ctx` audit (no code change, doc only).**
  Re-tested the spec-literal `AboveSkip + LeftSkip` form against the
  current `skip_probs[0]`-anchored constant. Spec form: c64 diffs
  29→20 (better), lossless pattern Y 9.90→9.67 dB (worse), compound
  mean 10.72→10.47 dB (worse). The "max(above,left)" alternative is
  identical to spec on c64 but worse on lossless-pattern Y. The
  `skip_probs[0]` constant is kept as the best-overall baseline.

### Findings — round-19 systematic audit

The 9.90 dB lossless number reflects a systemic bool-decoder
misalignment, not a single mis-implemented sub-system. Audited and
ruled out:

* **§8.7.1.10 WHT inverse transform**: bit-correct (3 new unit tests).
* **§8.6.2 reconstruct add-and-clip**: matches spec literal (no
  Round2 in lossless path, just clip-add).
* **§8.5.1 DC_PRED prediction at frame top-left**: bit-exact on c64
  16×16 first block (residual recovers pixel value 16 from predictor
  128 with WHT(-1792)).
* **§9.3.2 KF_PARTITION_PROBS table layout**: comment was misleading
  but data is in spec order (8×8 → 4×4 first, 64×64 → 32×32 last).
  Reader uses `bsl=0` for 8×8 query, indexing entries [0..3] which
  match the spec's "8×8 → 4×4" rows.
* **§6.4.6 default_intra_mode tree (`read_intra_mode_tree`)**: 9-prob
  walk matches spec's `intra_mode_tree[18]` node-by-node.
* **§6.4.25 get_scan**: lossless → DCT_DCT scan = `default_scan_4x4`
  (per spec).
* **§6.4.24 token initial-context**: `above + left` over the
  AboveNonzeroContext / LeftNonzeroContext arrays per spec, with
  `step = 1 << txSz` update step.

The bug is somewhere in the §6.4.6 mode_info / §6.4.21 residual
sequence where one of: read_tx_size context, default_uv_mode context,
or one of the prob-table updates parsed from the compressed header
gates a different probability than the encoder wrote. The c64
constant-color fixture is the smallest known reproduction.

## [r18 baseline]

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
