# Changelog

All notable changes to `oxideav-vp9` are recorded here.

## [Unreleased]

### Added

* **Round 10: §8.5.1 intra prediction process (crate-local `intra`
  module).**
  * `PredMode` — the 10 §7.4.5 intra prediction modes with
    discriminants matching the spec numbering exactly (`DC_PRED` = 0,
    `V_PRED` = 1, `H_PRED` = 2, `D45_PRED` = 3, `D135_PRED` = 4,
    `D117_PRED` = 5, `D153_PRED` = 6, `D207_PRED` = 7, `D63_PRED` = 8,
    `TM_PRED` = 9) plus `from_raw` for the (deferred) mode-info decode.
  * `Plane` — a minimal row-major `i32` plane buffer standing in for
    `CurrFrame[ plane ]`; the §8.5.1 process reads neighbour samples
    from it and writes the prediction back.
  * `predict_intra( plane, x, y, have_left, have_above, not_on_right,
    tx_sz, mode, max_x, max_y, bit_depth )` (§8.5.1) — builds
    `aboveRow[-1 .. 2*size-1]` and `leftCol[0 .. size-1]` per the
    `haveAbove` / `haveLeft` / `notOnRight` availability rules (the
    upper-right extension fires only for `txSz == 0`; missing
    neighbours fill `(1<<(BitDepth-1)) ± 1`), forms the `pred` block
    for the selected mode — `V`/`H` copies; the four `DC` neighbour
    cases (`avg` = `(sum + size) >> (log2Size+1)`, `leftAvg` /
    `aboveAvg` = `(sum + (1<<(log2Size-1))) >> log2Size`, and the
    `1<<(BitDepth-1)` no-neighbour fill); the `D45`/`D63`/`D117`/
    `D135`/`D153`/`D207` directional `Round2` recurrences (including
    the reverse-order `D207` step 5); and `TM` with `Clip1` — then
    stores it back. Neighbour reads clamp with `Min(maxX, .)` /
    `Min(maxY, .)`. Crate-local `round2` (§3) and `clip1` (§3)
    helpers.
  * 16 unit tests: `PredMode::from_raw` round-tripping the §7.4.5
    numbering (and the discriminant values), `round2` / `clip1`
    against their §3 definitions, `V_PRED` / `H_PRED` copy semantics,
    all four `DC_PRED` neighbour cases, the `TM_PRED` formula plus
    out-of-range clipping, a constant-neighbour invariant collapsing
    every directional mode to the neighbour constant across all four
    transform sizes, the `D207_PRED` bottom-row / step-2 formulas, the
    `notOnRight`-gated upper-right `aboveRow` extension (via
    `D45_PRED`, enabled only for `txSz == 0`), and the `Min(maxX, .)`
    plane-edge clamping of neighbour reads. No external library /
    source was consulted; every formula is transcribed directly from
    the spec §8.5.1 listing. The §8.6.2 reconstruct driver that
    supplies the real availability flags and adds the round-9
    inverse-transformed residual to this prediction remains deferred
    to a future round; the round-10 surface is internal-only.

* **Round 9: §8.7 inverse transform process (crate-local `idct`
  module).**
  * The §8.7.1.1 butterfly primitives — `B` (butterfly rotation,
    including the `16 + 32*k` two-multiply fast path), `H` (Hadamard
    rotation), `SB` (butterfly into the high-precision `S` array) and
    `SH` (Hadamard rotation + `Round2(·,14)` out of `S`) — plus the
    `cos64` / `sin64` angle functions backed by the verbatim 33-entry
    `COS64_LOOKUP` quarter-wave table and the `brev` bit-reversal
    helper. All fixed-point intermediates are evaluated in `i64`
    (the spec notes `S` needs `24 + BitDepth` bits).
  * `inverse_dct( t, n )` (§8.7.1.2 + §8.7.1.3) — the inverse-DCT
    array permutation (`T[i] = copyT[ brev(n, i) ]`) followed by the
    recursive inverse DCT process for `2 <= n <= 5`.
  * `inverse_adst( t, n )` (§8.7.1.4 .. §8.7.1.9) — the ADST
    input/output permutations and the ADST4 / ADST8 / ADST16
    processes (the `SINPI_1_9 .. SINPI_4_9` constants transcribed
    verbatim) dispatched by `n` for `2 <= n <= 4`.
  * `inverse_wht( t, shift )` (§8.7.1.10) — the in-place inverse
    Walsh-Hadamard transform with the `shift` pre-scaling argument.
  * `inverse_transform_2d( dequant, n, tx_type, lossless )` (§8.7.2)
    — the 2D driver: per-`TxType` row transform then column
    transform over a `(1<<n)` by `(1<<n)` `Dequant` block, the
    lossless WHT path (`shift = 2` rows / `0` columns), and the
    `Round2( T[i], Min(6, n+2) )` column rounding. `TxType` constants
    `DCT_DCT` / `ADST_DCT` / `DCT_ADST` / `ADST_ADST` are defined per
    §3.
  * 20 unit tests: `cos64` quarter-wave symmetry + periodicity,
    `sin64` shift, `brev`, the Hadamard sum/difference + flip
    semantics, the `16+32*k` butterfly fast-path equivalence, the
    DC-only "flat output" property of the 4/8/16/32-point inverse
    DCT, zero-in/zero-out for all ADST sizes, the §8.7.1.5 output
    permutation indices, and the 2D driver's zero-in/zero-out (all
    four `TxType`s, `n = 2..5`) + DC-only flat-block property (lossy
    and lossless paths). No external library / source was consulted;
    the `cos64_lookup` table and `SINPI_*_9` constants are
    transcribed directly from the spec §8.7.1 listings. The §8.6.2
    reconstruct driver that builds the `Dequant` input (round-7 token
    magnitudes scaled by the round-8 quantizers) and adds the
    residual to the prediction remains deferred to a future round.

* **Round 8: §8.6.1 dequantization functions (crate-local `dequant`
  module).**
  * `dc_q( bit_depth, b )` / `ac_q( bit_depth, b )` (§8.6.1) — index
    the `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
    `(BitDepth - 8) >> 1` row and the `Clip3(0, 255, b)` column.
    Both 256-entry tables are transcribed verbatim from the §8.6.1
    listing into `DC_QLOOKUP` / `AC_QLOOKUP`.
  * `seg_feature_active( seg, segment_id, feature )` (§6.4.9) —
    `segmentation_enabled && FeatureEnabled[ segment_id ][ feature ]`.
  * `get_qindex( seg, quant, segment_id )` (§8.6.1) — applies the
    `SEG_LVL_ALT_Q` segment feature (absolute update replaces
    `base_q_idx`, delta update offsets it, then `Clip3(0, 255, .)`)
    or returns `base_q_idx` when the feature is inactive.
  * `get_dc_quant( plane, .. )` / `get_ac_quant( plane, .. )`
    (§8.6.1) — combine `get_qindex()` with the plane-specific header
    delta (`delta_q_y_dc` luma DC, `delta_q_uv_dc` chroma DC,
    `delta_q_uv_ac` chroma AC; luma AC has none) and dispatch to
    `dc_q` / `ac_q`.
  * `SEG_LVL_ALT_Q` constant (§3 table of constants) plus a private
    `clip3` helper (§5.1).
  * 13 unit tests including table-shape / spec-anchor checks, the
    `clip3` clamp branches, `dc_q` / `ac_q` index clipping + row
    selection, all three `get_qindex` paths, plane-delta selection
    for `get_dc_quant` / `get_ac_quant`, and the high-bit-depth
    divergence of the same qindex across the three table rows. The
    §8.6.2 reconstruct driver that consumes these helpers (scaling
    the round-7 `Tokens` array, with the `dqDenom = 2` halving for
    `TX_32X32`) remains deferred to a future round. No external
    library / source was consulted; the lookup tables are
    transcribed directly from the spec §8.6.1 listing.

* **Round 7: §6.4.24 / §6.4.26 coefficient-token decoder (crate-local
  `tokens` module).**
  * `read_token( coder, &probs )` (§6.4.24, §9.3.3) — walks the
    20-entry `token_tree` returning one of `ZERO_TOKEN` ..
    `DCT_VAL_CATEGORY6`. `probs[0..=9]` are the 10 internal-node
    `read_bool` probabilities pre-derived from
    `coef_probs[...][1]` / `coef_probs[...][2]` via `pareto`.
  * `pareto( node, prob )` (§9.3.2) — short-circuits to `prob` for
    `node < 2`; otherwise looks up
    `PARETO_TABLE[ (prob - 1) / 2 ][ node - 2 ]` (odd `prob`) or
    interpolates two adjacent rows (even `prob`). The full 128×8
    pareto table is transcribed verbatim from spec §10.3.
  * `read_more_coefs( coder, prob )` (§6.4.24, §9.3.2) — single
    `B(p)` returning `true` (continue scan) / `false` (EOB).
  * `read_coef( coder, token, bit_depth )` (§6.4.26) — extra-bits
    decoder. For tokens `ONE`..`FOUR` no bits are read and the base
    coef is returned. For `DCT_VAL_CATEGORY1..6` the `numExtra`
    `B(p)` reads against `cat_probs[cat]` build the magnitude in
    `EXTRA_BITS[token][2] + (extra_bits_value)`. For `CAT6` at
    `bit_depth ∈ {10, 12}` an additional `BitDepth - 8` `B(255)`
    `high_bit` reads prepend MSBs at shift `5 + BitDepth - e`.
  * `read_coef_token( coder, check_eob, more_coefs_prob,
    &token_probs, bit_depth )` — driver returning
    `CoefStep::Eob | Coef { token, value }`. Folds `read_more_coefs`
    + `read_token` + `read_coef` + `L(1) sign_bit` together. The
    `checkEob` flag itself stays in the caller's residual loop per
    §6.4.24.
  * `EXTRA_BITS[11][3]`, `CAT_PROBS[7][14]`, `ENERGY_CLASS[12]`,
    `TOKEN_TREE[20]`, `PARETO_TABLE[128][8]` constants — all
    transcribed verbatim from spec §6.4.26 / §10.2 / §10.3 / §9.3.
  * 28 unit tests including hand-traced golden buffers for
    `ONE_TOKEN` (`0x40 0x00 …`) and `TWO_TOKEN` (`0x60 0x00 …`)
    derived by stepping the §9.2 decoder by hand. No external
    library / source was consulted; the §6.4.21 `residual( )` driver
    that will consume these helpers remains deferred to a future
    round.

* **Round 6: §6.3.7 `read_coef_probs` 6D coefficient-probability
  sweep wired into `parse_compressed_header`.**
  * `read_coef_probs(&mut coder, tx_mode, &mut coef_probs)` (§6.3.7)
    — walks `txSz ∈ [TX_4X4, maxTxSize]` with `maxTxSize =
    tx_mode_to_biggest_tx_size[ tx_mode ]` (§10.5). Per active
    tx-size, reads an outer `L(1) update_probs` flag and, on 1,
    drives a nested `(i, j, k, l, m)` sweep over `BLOCK_TYPES=2 ×
    REF_TYPES=2 × COEF_BANDS=6 × maxL(k) × UNCONSTRAINED_NODES=3`
    cells, with `maxL = (k == 0) ? 3 : 6` per §6.3.7 (band 0 has
    only 3 valid previous-coef contexts). Each cell becomes
    `read_diff_update_prob( coder, cell )`. Fully-active inner
    walk is 396 cells per tx-size, 1584 for a TX_MODE_SELECT
    frame.
  * `tx_mode_to_biggest_tx_size( tx_mode )` const-fn (§10.5) —
    maps `TxMode` to the spec's biggest-tx-size index, with the
    `ALLOW_32X32` and `TX_MODE_SELECT` rows both mapping to
    `TX_32X32 = 3`.
  * `coef_probs::DEFAULT_COEF_PROBS: CoefProbs` constant in the
    new `src/coef_probs.rs` module — the §10
    `default_coef_probs[TX_SIZES=4][BLOCK_TYPES=2][REF_TYPES=2][
    COEF_BANDS=6][PREV_COEF_CONTEXTS=6][UNCONSTRAINED_NODES=3]`
    table (1728 u8 entries) transcribed verbatim from the spec
    listing. Band-0 trailing `{0, 0, 0} // unused` rows preserved
    as in-table sentinels matching the `maxL = 3` clamp.
  * `CoefProbs` public type alias re-exported from the crate root
    (`pub use coef_probs::CoefProbs;`), naming the 6D array shape.
  * `Vp9CompressedHeader` extended with `pub coef_probs:
    CoefProbs`. `parse_compressed_header` now runs the sweeps in
    spec order: `read_tx_mode` → optional `read_tx_mode_probs` →
    `read_coef_probs` → `read_skip_prob`.
  * `Vp9CompressedHeader` no longer derives `Copy` — the 1728-byte
    `coef_probs` field makes silent copies costly. `Clone` is
    retained.
  * 7 new unit tests: `tx_mode_to_biggest_tx_size` against the
    §10.5 listing, `read_coef_probs` zero-buffer passthrough for
    `ONLY_4X4` (1-slab) and `TX_MODE_SELECT` (4-slab) modes, an
    across-mode sweep, `DEFAULT_COEF_PROBS` shape +
    spec-listing anchors (TX_4X4 / block-type 0 / Intra / band 0
    / ctx 0 = {195, 29, 183}; TX_32X32 / block-type 1 / Inter /
    band 5 / ctx 5 = {1, 16, 6}), the band-0 unused-row sentinel
    invariant, and the inner-sweep `2 × 2 × (3 + 5 × 6) × 3 =
    396` cell-count check.
  * 2 new end-to-end integration tests
    (`tests/compressed_header.rs`):
    `end_to_end_tx_mode_select_runs_coef_probs_sweep` driving the
    full TX_MODE_SELECT → tx_mode_probs → coef_probs → skip_prob
    chain through `parse_uncompressed_header` +
    `parse_compressed_header` and verifying two §10 default
    anchors survive verbatim; and
    `end_to_end_only_4x4_visits_only_first_tx_size_coef_slab`
    confirming the outer-loop tx-size clipping for the ONLY_4X4
    path.

* **Round 5: §6.3.2 `tx_mode_probs` + §6.3.8 `read_skip_prob` sweeps
  wired into `parse_compressed_header`.**
  * `read_tx_mode_probs(&mut coder, &mut tx_probs)` (§6.3.2) —
    three nested sweeps (`TX_SIZE_CONTEXTS * (1+2+3) = 12` cells)
    walking `tx_probs_8x8`, `tx_probs_16x16`, `tx_probs_32x32` via
    `read_diff_update_prob`. Gated on
    `tx_mode == TX_MODE_SELECT` per the §6.3 compressed-header
    syntax dispatch.
  * `read_skip_prob(&mut coder, &mut skip_prob)` (§6.3.8) —
    unconditional `SKIP_CONTEXTS = 3` sweep via
    `read_diff_update_prob`.
  * `DEFAULT_TX_PROBS: [[[u8; 3]; 2]; 4]` and
    `DEFAULT_SKIP_PROB: [u8; 3] = [192, 128, 64]` constants
    transcribed verbatim from the §10 default-tables listing.
  * `Vp9CompressedHeader` extended with `tx_probs` and `skip_prob`
    fields exposing the post-sweep tables to callers.
    `parse_compressed_header` runs the sweeps in spec order:
    `read_tx_mode` → optional `read_tx_mode_probs` →
    `read_skip_prob`.
  * 10 new unit tests covering `DEFAULT_TX_PROBS` shape + value
    spot-check, `DEFAULT_SKIP_PROB` value, `read_tx_mode_probs`
    zero-buffer passthrough, the 12-cell sweep count, the row-0
    (`TX_4X4`) non-modification invariant, `read_skip_prob`
    zero-buffer passthrough across `SKIP_CONTEXTS = 3`, and three
    `parse_compressed_header` integration scenarios (ONLY_4X4
    non-lossless, lossless, and TX_MODE_SELECT). 2 new end-to-end
    integration tests in `tests/compressed_header.rs` driving the
    full uncompressed-header splice plus the new sweeps.
  * `read_diff_update_prob`, `decode_term_subexp`,
    `inv_remap_prob`, `inv_recenter_nonneg`, and `INV_MAP_TABLE`
    drop their round-4 `#[allow(dead_code)]` markers — they are
    now driven live by the §6.3.2 / §6.3.8 sweeps.

* **Round 4: §6.3.3 `diff_update_prob` chain (`decode_term_subexp` +
  `inv_remap_prob` + `inv_recenter_nonneg` + 255-entry
  `inv_map_table`).**
  * `read_diff_update_prob( coder, base_prob )` (§6.3.3) — reads the
    `B(252)` `update_prob` flag, on 1 pulls a `decode_term_subexp`
    value and remaps the previous probability through
    `inv_remap_prob`. On 0, passes the base probability straight
    through.
  * `decode_term_subexp( )` (§6.3.4) — the 5-leg
    `L(1) → L(4)` / `L(1) → L(4)+16` / `L(1) → L(5)+32` /
    `L(7) → +64` / `L(7), L(1) → (v<<1)-1+bit` cascade producing a
    value in `0..=254`.
  * `inv_remap_prob( delta_prob, prob )` (§6.3.5) — low-half /
    high-half piecewise remap calling `inv_recenter_nonneg`.
  * `inv_recenter_nonneg( v, m )` (§6.3.6) — pure arithmetic
    helper covering the `v > 2*m` short-circuit plus the
    odd / even split.
  * `INV_MAP_TABLE: [u8; 255]` — the §6.3.5 listing transcribed
    verbatim (a permutation of 1..=254 with a duplicated trailing
    253).
  * 16 new unit tests covering `inv_recenter_nonneg` (all three
    piecewise branches plus the `v == 2*m` boundary),
    `INV_MAP_TABLE` length + spot-checks, `inv_remap_prob` against
    both low-half / high-half branches and the `v > 2m`
    short-circuit, `decode_term_subexp` against hand-derived §9.2
    buffers (leg-1 zero plus a sweep that confirms the result
    stays in `0..=254`), and `read_diff_update_prob` confirming
    the `update_prob == 0` passthrough across the full
    1..=255 base-probability sweep.
  * The chain is structural; no caller in §6.3.2 / §6.3.7+ uses it
    yet, so each helper carries `#[allow(dead_code)]` until the
    next round wires them into the table sweeps.

* **Round 3: §9.2 Boolean decoder primitives + §6.3.1 `read_tx_mode`
  walk.**
  * New `src/bool_coder.rs` module implementing the four §9.2
    primitives: `init_bool( sz )` (§9.2.1) with the marker-bit
    zero-conformance check, `read_bool( p )` (§9.2.2) with the
    `split = 1 + (((BoolRange-1) * p) >> 8)` narrow and the
    `range < 128` renormalisation refill, `read_literal( n )`
    (§9.2.4) folded over `read_bool(128)`, and `exit_bool( )`
    (§9.2.3) consuming the remaining `BoolMaxBits` with a
    zero-pad conformance check. `BoolMaxBits` underflow raises
    `InvalidBitstream` instead of silently injecting 0.
  * New `src/compressed.rs` module with `Vp9CompressedHeader`,
    `TxMode` (5-variant enum mapping §3 `TX_MODES`), and the
    `parse_compressed_header(payload, lossless)` entry point.
    `read_tx_mode( )` (§6.3.1) short-circuits to `ONLY_4X4` when
    `Lossless == 1` (§6.2.9), otherwise reads `L(2)` and (for the
    `ALLOW_32X32` raw value) an extra `L(1)` `tx_mode_select` to
    distinguish `ALLOW_32X32` from `TX_MODE_SELECT`.
  * 9 `bool_coder` unit tests + 7 `compressed` unit tests + 4
    end-to-end integration tests (`tests/compressed_header.rs`)
    splicing a hand-derived §9.2 payload past
    `Vp9FrameHeader::uncompressed_header_size_bytes`. Every byte
    vector used was derived by stepping the §9.2 decoder, not
    borrowed from any third-party VP9 implementation.
  * The §6.3.2+ syntax (`tx_mode_probs`, `read_coef_probs`,
    `read_skip_prob`, `read_inter_mode_probs`,
    `read_interp_filter_probs`, …) all flow through the §6.3.3
    `diff_update_prob` chain (`decode_term_subexp` (§6.3.4) +
    `inv_remap_prob` (§6.3.5) + `inv_recenter_nonneg` (§6.3.6) +
    the 255-entry `inv_map_table` constant) and have been
    deferred to the next round so this drop lands a verifiable
    Boolean-coder primitive plus the §6.3.1 walk in isolation.

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
