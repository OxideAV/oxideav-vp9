# Changelog

All notable changes to `oxideav-vp9` are recorded here.

## [Unreleased]

### Added

* **Round 14: §6.4.21 `residual( )` intra driver (crate-local `residual`
  module).** The §6.4.21 outer loop for the intra path — the per-plane,
  per-4x4-sub-block walk that owns the `AboveNonzeroContext` /
  `LeftNonzeroContext` write-back across a whole MI block, drives the
  round-13 §6.4.24 `tokens( )` per-block decode, and feeds the round-11
  §8.6.2 `reconstruct_block` with real per-block `Tokens` arrays,
  availability flags and plane/quantizer state.
  * §10.2 `num_4x4_blocks_wide_lookup[13]` / `num_4x4_blocks_high_lookup[13]`,
    §6.4.10 `max_txsize_lookup[13]`, and §6.4.23 `ss_size_lookup[13][2][2]`
    tables transcribed verbatim, alongside the `BLOCK_4X4 .. BLOCK_64X64`
    / `BLOCK_INVALID` `subsize` constants from §3.
  * `get_plane_block_size( subsize, plane, subsampling_x, subsampling_y )`
    (§6.4.23) and `get_uv_tx_size( tx_size, mi_size, subsampling_x,
    subsampling_y )` (§6.4.22) — the chroma-plane block-size /
    transform-size derivations that key the per-plane loop.
  * `ResidualBlockCtx` — the per-MI-block / per-frame bundle (`MiCol` /
    `MiRow` / `MiCols` / `MiRows`, `MiSize`, `tx_size`, `subsampling_x` /
    `y`, `skip`, `Lossless`, `BitDepth`, the per-block `PredMode` for
    luma and chroma, and the per-plane DC/AC quantizers from round 8);
    plus `AvailFlags` for §7.4.4 `AvailL` / `AvailU` and a
    `PlaneBuffers` wrapper for the three `CurrFrame[ plane ]` planes.
  * `residual_intra( planes, nz, block, avail, token_source )` — the
    §6.4.21 driver proper: per plane, computes `bsize = MiSize <
    BLOCK_8X8 ? BLOCK_8X8 : MiSize`, the per-plane `planeSz` +
    `num4x4w` / `num4x4h` dimensions and chroma `txSz`, then walks the
    `(y, x)` 4x4 grid stepping by `step = 1 << txSz`. For each in-bounds
    transform block (`startX < maxx && startY < maxy`) it calls the
    round-10 `predict_intra` with the resolved `have_left` /
    `have_above` / `not_on_right` flags, pulls `Tokens[ ]` from a
    per-block `TokenSource` callback (when `!skip`), derives the §6.4.25
    `TxType` (chroma / `TX_32X32` / lossless force `DCT_DCT`; luma intra
    uses round-11 `tx_type_for_intra`), runs the round-11
    `reconstruct_block`, and writes
    `AboveNonzeroContext[ plane ][ x4 + i ] = LeftNonzeroContext[
    plane ][ y4 + i ] = nonzero` for `i ∈ 0..step` per the §6.4.21
    trailing write-back.
  * 12 unit tests: the §10.2 / §6.4.10 / §6.4.23 table-anchor checks
    (luma-identity invariant + 4:2:0 / asymmetric subsampling
    anchors), the §6.4.22 `get_uv_tx_size` chroma cap and the sub-8x8
    short-circuit, the `skip = true` path leaving every strip cell at 0
    (no token decode), the full `skip = false` walk firing
    `token_source` exactly 16 luma + 4 U + 4 V times for a `BLOCK_16X16`
    MI block at `tx_size = TX_4X4` (each call recorded with `(plane,
    block_idx, tx_sz)`), the `nonzero = true` strip write-back over
    `step = 1 << tx_sz` 4-sample units for `tx_size = TX_8X8`, the
    out-of-bounds block skip with intact zero context, a DC-only luma
    block at MI (1,1) lockstep against an independent `predict_intra` +
    `reconstruct_block` probe, and the `bsize = max(MiSize, BLOCK_8X8)`
    widening for a `BLOCK_4X4` MI block. No external library / source
    was consulted; every formula and table is transcribed directly from
    the §6.4.21 / §6.4.22 / §6.4.23 / §10.2 listings.
  * The `is_inter` branch of §6.4.21 (which calls `predict_inter( )`
    before the per-block loop) is deferred until the §8.5.2 inter
    prediction process and reference-buffer state land; the per-block
    mode-info decode (`y_mode` / `sub_modes` / `tx_size` / `skip` /
    `segment_id` from §6.4) that the residual loop reads is also a
    later-round increment. The round-14 surface is internal-only.

* **Round 13: §6.4.24 `tokens( )` per-block coefficient driver
  (crate-local `tokens` module).** Walks the round-12 §6.4.25 scan
  order and feeds each scan position through the round-7
  `read_coef_token` pipeline, recovering one transform block's
  quantised coefficients into a `Tokens[ ]` array.
  * The §10 band tables — `coefband_4x4[ 16 ]` transcribed verbatim and
    `coefband_8x8plus[ 1024 ]` built from the verbatim 21-entry prefix
    plus the all-`5` tail — selected by `coef_band( c, txSz )` per the
    §6.4.24 `(txSz == TX_4X4) ? coefband_4x4 : coefband_8x8plus` rule.
  * `token_cache_neighbours( c, pos, txSz, txType )` — the §9.3.2
    neighbour pair (`nb[ 0 ]` / `nb[ 1 ]`): `(0, 0)` for the DC
    coefficient, and for `c > 0` the above (`(i-1)*n + j`) / left
    (`i*n + j-1`) raster cells with the `DCT_ADST` (double above) /
    `ADST_DCT` (double left) / first-row / first-column variants
    (`n = 4 << txSz`).
  * `build_token_probs( cell )` — the §9.3.2 10-node probability array:
    node 0 → `cell[1]`, node 1 → `cell[2]`, node `2..=9` →
    `pareto( node, cell[2] )`.
  * `NonzeroContext` (the per-plane `AboveNonzeroContext` /
    `LeftNonzeroContext` 4-sample strips) and `TokenBlockCtx` (the
    per-block / per-frame state `tokens( )` reads — `plane`,
    `is_inter`, resolved `TxType`, `BitDepth`, `x4` / `y4`, `maxX` /
    `maxY`).
  * `tokens( coder, block, txSz, scan, coef_probs, nz, token_cache,
    tokens )` — the §6.4.24 driver: `segEob = 16 << (txSz << 1)`, the
    `checkEob` gating, the §9.3.2 per-coefficient `ctx` (DC from the
    non-zero strips, `c > 0` from `TokenCache`), the
    `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell pick, the
    `more_coefs` / `token` / `read_coef` / `sign_bit` decode, the
    `TokenCache[ pos ] = energy_class[ token ]` write, the
    `ZERO_TOKEN`-clears-`checkEob` rule, the trailing `Tokens[ scan[ i
    ] ] = 0` zero-fill, and the `nonzero = c > 0` return.
  * 15 unit tests: the band tables vs the §10 listing + the
    `coef_band` dispatch, the §9.3.2 neighbour derivation (DC, interior
    DCT_DCT / ADST variants, first row / column, 8x8 width scaling),
    `build_token_probs` node mapping, and the `tokens( )` driver
    (zero-buffer immediate EOB + all-zero fill, the
    `ZERO_TOKEN`-clears-`checkEob` block fill, a lockstep match against
    an independent `read_coef_token` walk, the DC non-zero-context cell
    routing, the trailing zero-fill, and the `NonzeroContext::new`
    all-zero invariant). No external library source was consulted; the
    band tables and every formula are transcribed directly from the
    §6.4.24 / §9.3.2 / §10 listings. The §6.4.21 residual loop that
    threads `NonzeroContext` across the frame and feeds the round-11
    reconstruct driver lands in a later round; the round-13 surface is
    internal-only.

* **Round 12: §6.4.25 `get_scan` scan-order selection (crate-local
  `scan` module).** The first step of the §6.4.24 `tokens( )` per-block
  driver — it picks the scan order (the sequence of raster positions
  `pos = scan[ c ]` the coefficient loop visits) for a transform block.
  * The §10.1 scan tables transcribed verbatim: `default_scan_4x4` /
    `col_scan_4x4` / `row_scan_4x4` (16 entries), the 8x8 trio (64),
    the 16x16 trio (256), and `default_scan_32x32` (1024). Element
    type `u16` so the 32x32 table's `0..=1023` range fits.
  * `get_scan( plane, txSz, txType )` — the §6.4.25 selection:
    `ADST_DCT` → `row_scan`, `DCT_ADST` → `col_scan`, else
    (`DCT_DCT` / `ADST_ADST`) → `default`, applying the §6.4.25 first
    half (a chroma plane or a `TX_32X32` block forces `TxType =
    DCT_DCT`). The mode-info-dependent `mode2txfm_map[ y_mode ]`
    `TxType` derivation already lives in
    [`reconstruct::tx_type_for_intra`]; the per-block mode-info state
    is owned by the (deferred) §6.4.21 residual driver.
  * `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` `txSz` index
    constants (§3).
  * 10 unit tests: every table's spec length, the permutation
    invariant (each raster position appears exactly once), the
    DC-first invariant, §10.1 listing anchors (first four + last
    entry of each table), the §6.4.25 `txType` → table selection for
    4x4 / 8x8 / 16x16, the `TX_32X32`-always-default and
    chroma-forces-default first-half overrides, and the
    `16 << (txSz << 1)` `segEob` length match. No external library
    source was consulted; the tables are transcribed directly from the
    §10.1 listing.

* **Round 11: §8.6.2 reconstruct driver (crate-local `reconstruct`
  module).** Ties the rounds 7-10 pieces together at the conceptual
  `reconstruct( plane, startX, startY, txSz )` call site of the
  §6.4.21 residual syntax.
  * `tx_type_for_intra( mode )` — the §6.4.25 `mode2txfm_map[ y_mode ]`
    lookup selecting the `TxType` (`DCT_DCT` / `ADST_DCT` / `DCT_ADST`
    / `ADST_ADST`) for a luma intra block from its `PredMode`. The
    10-entry intra prefix of `mode2txfm_map` (§10.5) is transcribed
    verbatim.
  * `reconstruct_block( plane_buf, x, y, tx_sz, tokens, dc_quant,
    ac_quant, tx_type, lossless, bit_depth )` (§8.6.2) — sets
    `dqDenom = 2` for `txSz == TX_32X32` else `1`, `n = 2 + txSz`,
    `n0 = 1 << n`; step 1 `Dequant[i][j] = (Tokens[i*n0+j] *
    get_ac_quant) / dqDenom`; step 2 the `Dequant[0][0] = (Tokens[0] *
    get_dc_quant) / dqDenom` DC override; step 3 the §8.7.2
    `inverse_transform_2d`; step 4 `CurrFrame[plane][y+i][x+j] =
    Clip1( CurrFrame[...] + Dequant[i][j] )`. Integer division
    truncates toward zero per §4.1 (Rust's `i64 /` matches).
  * `reconstruct_intra_block( … )` — the end-to-end one-block driver:
    predicts via §8.5.1 `predict_intra` (round 10), derives the
    `TxType` with the §6.4.25 `TX_32X32` / lossless `DCT_DCT`
    overrides, then runs `reconstruct_block`. The shape the deferred
    §6.4.21 residual loop will call once it threads real availability
    and quantizer state.
  * Crate-local `clip1` (§3) helper operating in `i64` so the
    high-precision residual sum does not overflow before clamping.
  * 10 unit tests: the `mode2txfm_map` intra prefix vs the §10.5
    listing, `clip1` range clamping, an all-zero `Tokens` block
    leaving the prediction unchanged, a DC-only `Tokens` block adding
    a flat residual to a flat prediction, step-4 clipping at both the
    bit-depth max and zero, the `TX_32X32` `dqDenom = 2` halving
    through the real driver, `reconstruct_intra_block` with DC_PRED +
    zero tokens equalling the pure DC prediction, DC_PRED + a known DC
    residual reconstructing the expected pixels, and the lossless WHT
    path driven via `reconstruct_intra_block`.

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
