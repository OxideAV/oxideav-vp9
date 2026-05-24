# oxideav-vp9

Pure-Rust VP9 codec — clean-room re-implementation against the VP9
Bitstream & Decoding Process Specification v0.7.

## Status — 2026-05-24

**Round 13: §6.4.24 `tokens( )` per-block coefficient driver.** Round 13
adds the §6.4.24 `tokens( )` driver to the `tokens` module — the per-block
loop that walks the round-12 §6.4.25 scan order (`pos = scan[ c ]`) and
feeds each scan position through the round-7 `read_coef_token` pipeline,
recovering one transform block's quantised coefficients into a `Tokens[ ]`
array:

* The §10 band tables — `coefband_4x4[ 16 ]` transcribed verbatim and
  `coefband_8x8plus[ 1024 ]` built from the verbatim 21-entry prefix plus
  the all-`5` tail — picked by `coef_band( c, txSz )` per the §6.4.24
  `(txSz == TX_4X4) ? coefband_4x4 : coefband_8x8plus` rule.
* `token_cache_neighbours( c, pos, txSz, txType )` — the §9.3.2 neighbour
  pair (`nb[ 0 ]` / `nb[ 1 ]`): `(0, 0)` for the DC coefficient, and for
  `c > 0` the above (`(i-1)*n + j`) / left (`i*n + j-1`) raster cells with
  the `DCT_ADST` (double above) / `ADST_DCT` (double left) / first-row /
  first-column variants (`n = 4 << txSz`).
* `build_token_probs( cell )` — the §9.3.2 10-node probability array
  (node 0 → `cell[1]`, node 1 → `cell[2]`, node `2..=9` →
  `pareto( node, cell[2] )`).
* `NonzeroContext` (the per-plane `AboveNonzeroContext` /
  `LeftNonzeroContext` 4-sample strips) and `TokenBlockCtx` (the per-block
  / per-frame state the driver reads).
* `tokens( coder, block, txSz, scan, coef_probs, nz, token_cache, tokens )`
  — the §6.4.24 driver proper: `segEob = 16 << (txSz << 1)`, the `checkEob`
  gating, the §9.3.2 per-coefficient `ctx` (DC from the non-zero strips,
  `c > 0` from `TokenCache`), the
  `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell pick, the
  `more_coefs` / `token` / `read_coef` / `sign_bit` decode, the
  `TokenCache[ pos ] = energy_class[ token ]` write, the
  `ZERO_TOKEN`-clears-`checkEob` rule, the trailing `Tokens[ scan[ i ] ] =
  0` zero-fill, and the `nonzero = c > 0` return.

The §6.4.21 `residual( )` plane / sub-block driver — which owns the
`AboveNonzeroContext` / `LeftNonzeroContext` write-back across a whole
frame, the per-block mode-info decode, and the wiring into the round-11
§8.6.2 `reconstruct_block` — lands in a later round. The round-13 surface
is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their result
types exclusively.

**Round 12: §6.4.25 `get_scan` scan-order selection.** Round 12 adds a
`scan` module (crate-internal, `pub(crate)`) implementing the §6.4.25
`get_scan( )` process — the first step of the §6.4.24 `tokens( )`
per-block driver, which selects the scan order (the sequence of raster
positions `pos = scan[ c ]` the coefficient loop walks) for a transform
block:

* The §10.1 scan tables transcribed verbatim — `default_scan_4x4` /
  `col_scan_4x4` / `row_scan_4x4` (16), the 8x8 trio (64), the 16x16
  trio (256), and `default_scan_32x32` (1024). The element type is
  `u16` so the 32x32 table's `0..=1023` raster positions fit.
* `get_scan( plane, tx_sz, tx_type )` (§6.4.25) — selects between the
  tables by the resolved `TxType`: `ADST_DCT` → `row_scan`, `DCT_ADST`
  → `col_scan`, else (`DCT_DCT` / `ADST_ADST`) → `default`. The §6.4.25
  first half (a chroma plane `plane > 0` or a `TX_32X32` block forces
  `TxType = DCT_DCT`) is applied here, so a caller passing the luma
  `TxType` for every plane still selects the right scan. The
  mode-info-dependent `mode2txfm_map[ y_mode ]` `TxType` derivation
  already lives in `reconstruct::tx_type_for_intra` (round 11); the
  per-block mode-info state (`y_mode`, `sub_modes`, `Lossless`,
  `is_inter`) is owned by the deferred §6.4.21 residual driver.
* `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` `txSz` index constants
  (§3).

The §6.4.24 `tokens( )` loop that walks `pos = scan[ c ]`, derives the
per-coefficient `ctx` from `AboveNonzeroContext` / `LeftNonzeroContext`
and `TokenCache`, and feeds the round-7 token decode lands in a later
round. The round-12 surface is internal-only; the public API still
exposes `parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 11: §8.6.2 reconstruct driver.** Round 11 adds a `reconstruct`
module (crate-internal, `pub(crate)`) implementing the §8.6.2
reconstruct process — the conceptual `reconstruct( plane, startX,
startY, txSz )` call site of the §6.4.21 residual syntax — that finally
ties the rounds 7-10 pieces into `reconstruct = predict + residual`:

* `tx_type_for_intra( mode )` (§6.4.25) — the `mode2txfm_map[ y_mode ]`
  lookup selecting the `TxType` (`DCT_DCT` / `ADST_DCT` / `DCT_ADST` /
  `ADST_ADST`) for a luma intra block from its `PredMode`. The 10-entry
  intra prefix of `mode2txfm_map` (§10.5) is transcribed verbatim (the
  four inter-mode entries, all `DCT_DCT`, are omitted as the helper
  only indexes the intra prefix).
* `reconstruct_block( plane_buf, x, y, tx_sz, tokens, dc_quant,
  ac_quant, tx_type, lossless, bit_depth )` (§8.6.2) — sets `dqDenom =
  2` for `txSz == TX_32X32` else `1`, `n = 2 + txSz`, `n0 = 1 << n`;
  step 1 `Dequant[i][j] = (Tokens[i*n0+j] * get_ac_quant) / dqDenom`,
  step 2 the `Dequant[0][0] = (Tokens[0] * get_dc_quant) / dqDenom` DC
  override, step 3 the round-9 §8.7.2 `inverse_transform_2d`, step 4
  `CurrFrame[ plane ][ y+i ][ x+j ] = Clip1( CurrFrame[..] +
  Dequant[i][j] )`. Integer division truncates toward zero per §4.1.
* `reconstruct_intra_block( .. )` — the end-to-end one-block driver:
  predicts via the round-10 §8.5.1 `predict_intra`, derives the
  `TxType` with the §6.4.25 `TX_32X32` / lossless `DCT_DCT` overrides,
  then runs `reconstruct_block`. This is the single-block shape the
  deferred §6.4.21 residual loop will drive once it threads real
  availability and quantizer state.

The §6.4.21 residual loop — which supplies the real per-block `Tokens`
arrays, availability flags and segment/quantizer state across a whole
frame, and wires this driver into a public decode path — lands in a
later round. The round-11 surface is internal-only; the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

**Round 10: §8.5.1 intra prediction process.** Round 10 adds an
`intra` module (crate-internal, `pub(crate)`) implementing the §8.5.1
intra prediction the §8.6.2 reconstruct process invokes for intra
blocks (the prediction half of `reconstruct = predict + residual`):

* `PredMode` — the 10 §7.4.5 intra prediction modes, with
  discriminants matching the spec numbering exactly (`DC_PRED` = 0,
  `V_PRED` = 1, `H_PRED` = 2, `D45_PRED` = 3, `D135_PRED` = 4,
  `D117_PRED` = 5, `D153_PRED` = 6, `D207_PRED` = 7, `D63_PRED` = 8,
  `TM_PRED` = 9), plus `from_raw` for the (deferred) mode-info decode.
* `Plane` — a minimal row-major `i32` plane buffer standing in for
  `CurrFrame[ plane ]`, read for neighbour samples and written with
  the prediction.
* `predict_intra( .. )` (§8.5.1) — builds the `aboveRow[-1 .. 2*size-1]`
  and `leftCol[0 .. size-1]` neighbour arrays per the `haveAbove` /
  `haveLeft` / `notOnRight` availability rules (including the
  upper-right extension that fires only for `txSz == 0` and the
  `(1<<(BitDepth-1)) ± 1` no-neighbour fills), then forms the `pred`
  block for the selected mode: `V`/`H` copies, the four `DC` neighbour
  cases (`avg` / `leftAvg` / `aboveAvg` / midpoint), the
  `D45`/`D63`/`D117`/`D135`/`D153`/`D207` directional `Round2`
  recurrences, and `TM` with `Clip1`. Neighbour reads clamp with
  `Min(maxX, .)` / `Min(maxY, .)`; the result is stored back into the
  plane.

The §8.6.2 reconstruct driver — which supplies the real `haveAbove` /
`haveLeft` / `notOnRight` flags (from tile / frame edges) and adds the
round-9 inverse-transformed residual to this prediction — lands in a
later round. The round-10 surface is internal-only; the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

**Round 9: §8.7 inverse transform process.** Round 9 adds an `idct`
module (crate-internal, `pub(crate)`) implementing the §8.7 inverse
transform stage the §8.6.2 reconstruct process invokes after the
round-8 dequantization step:

* The §8.7.1.1 butterfly primitives — `B` (butterfly rotation, with
  the `16 + 32*k` two-multiply fast path), `H` (Hadamard rotation),
  `SB` (butterfly into the high-precision `S` array) and `SH`
  (Hadamard rotation + `Round2(·, 14)` out of `S`) — plus `cos64` /
  `sin64` backed by the verbatim 33-entry `cos64_lookup` quarter-wave
  table and the `brev` bit-reversal helper. Fixed-point intermediates
  are evaluated in `i64` (the spec notes `S` needs `24 + BitDepth`
  bits of precision).
* `inverse_dct( t, n )` (§8.7.1.2 + §8.7.1.3) — the inverse-DCT array
  bit-reversal permutation followed by the recursive inverse DCT
  process for `2 <= n <= 5` (4/8/16/32-point).
* `inverse_adst( t, n )` (§8.7.1.4 .. §8.7.1.9) — the ADST
  input/output permutations and the ADST4 / ADST8 / ADST16 processes
  (with the `SINPI_1_9 .. SINPI_4_9` constants transcribed verbatim)
  dispatched by `n` for `2 <= n <= 4`.
* `inverse_wht( t, shift )` (§8.7.1.10) — the in-place inverse
  Walsh-Hadamard transform.
* `inverse_transform_2d( dequant, n, tx_type, lossless )` (§8.7.2) —
  the 2D driver applying the per-`TxType` row transform then column
  transform over a `(1<<n)` by `(1<<n)` block, the lossless WHT path
  (`shift = 2` rows / `0` columns), and the `Round2( T[i], Min(6,
  n+2) )` column rounding. `TxType` constants `DCT_DCT` / `ADST_DCT`
  / `DCT_ADST` / `ADST_ADST` follow §3.

The §8.6.2 reconstruct driver — which builds the `Dequant` input
(round-7 token magnitudes scaled by the round-8 quantizers, with the
`dqDenom = 2` halving for `TX_32X32`), calls this transform layer and
adds the residual to the prediction — lands in a later round. The
round-9 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 8: §8.6.1 dequantization functions.** Round 8 adds a
`dequant` module (crate-internal, `pub(crate)`) implementing the
quantizer-value derivation the §8.6.2 reconstruct process consumes
between the round-7 coefficient-token decode and the §8.7 inverse
transform:

* `dc_q( bit_depth, b )` / `ac_q( bit_depth, b )` (§8.6.1) — index
  the `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
  `(BitDepth - 8) >> 1` row (0 / 1 / 2 for 8- / 10- / 12-bit) and the
  `Clip3(0, 255, b)` column. Both 256-entry tables are transcribed
  verbatim from the §8.6.1 listing into `DC_QLOOKUP` / `AC_QLOOKUP`.
* `seg_feature_active( seg, segment_id, feature )` (§6.4.9) —
  `segmentation_enabled && FeatureEnabled[ segment_id ][ feature ]`.
* `get_qindex( seg, quant, segment_id )` (§8.6.1) — the per-block
  quantizer index. When `seg_feature_active( SEG_LVL_ALT_Q )`, the
  segment's `FeatureData` either replaces `base_q_idx` (absolute
  update) or offsets it (delta update), then `Clip3(0, 255, .)`;
  otherwise `base_q_idx` is returned directly.
* `get_dc_quant( plane, .. )` / `get_ac_quant( plane, .. )`
  (§8.6.1) — combine `get_qindex()` with the plane-appropriate
  header delta (`delta_q_y_dc` luma DC, `delta_q_uv_dc` chroma DC,
  `delta_q_uv_ac` chroma AC; luma AC has no delta in VP9) and
  dispatch to `dc_q` / `ac_q`.

The §8.6.2 reconstruct driver — which scales the round-7 `Tokens`
array by these quantizers (with the `dqDenom = 2` halving for
`TX_32X32`), runs the §8.7 inverse transform and adds the residual
to the prediction — lands in a later round. The round-8 surface is
internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 7: §6.4.24 / §6.4.26 coefficient-token decoder.** Round 7
adds a `tokens` module (crate-internal, `pub(crate)`) implementing
the pieces needed to recover one quantised DCT coefficient at a time
from the §9.2 Boolean coder, given a pre-selected
`coef_probs[ txSz ][ plane>0 ][ is_inter ][ band ][ ctx ][..3 ]` cell:

* The §9.3 `token_tree[20]` walker — `read_token( coder, &probs )` —
  returning one of the 11 spec tokens (`ZERO_TOKEN` through
  `DCT_VAL_CATEGORY6`).
* The §9.3.2 `pareto( node, prob )` helper backed by the §10.3
  128×8 `pareto_table` (transcribed verbatim).
* The §6.4.24 `more_coefs` `B(p)` reader.
* The §6.4.26 `read_coef( coder, token, bit_depth )` extra-bits
  decoder consuming the `extra_bits[11][3]` and `cat_probs[7][14]`
  tables (transcribed verbatim) plus the CAT6 `high_bit` `B(255)`
  loop that prepends `BitDepth - 8` MSBs for 10- and 12-bit profiles
  at shift positions `5 + BitDepth - e`.
* A `read_coef_token` driver that wires `more_coefs` + `read_token` +
  `read_coef` + `L(1) sign_bit` together and returns
  `CoefStep::{ Eob, Coef { token, value } }`.
* The §10.2 `energy_class[12]` table (for `TokenCache` once the
  residual driver lands).

The hand-traced golden buffers in the test suite cover every token
slot's "all-zero extra bits" base value, both legs of the
`more_coefs` decision, and two non-trivial tree-walk paths
(`ONE_TOKEN` from buffer `0x40 0x00 …`, `TWO_TOKEN` from buffer
`0x60 0x00 …`).

The §6.4.21 `residual( )` plane/sub-block driver still lands in a
later round — it owns the `AboveNonzeroContext` / `LeftNonzeroContext`
arrays and the per-block `ctx` derivation that picks the right
`coef_probs` cell. The round-7 surface is internal-only; the public
API still exposes `parse_uncompressed_header` and
`parse_compressed_header` exclusively.

**Round 6: §6.3.7 `read_coef_probs` 6D coefficient-probability sweep
wired in.** Round 6 inserts the §6.3.7 walker between the round-5
§6.3.2 `tx_mode_probs` and §6.3.8 `read_skip_prob` calls. The
walker:

* Visits `txSz ∈ [TX_4X4, maxTxSize]` where `maxTxSize =
  tx_mode_to_biggest_tx_size[ tx_mode ]` per §10.5 (1, 2, 3, 4 or 4
  active tx-size slabs for `ONLY_4X4` / `ALLOW_8X8` / `ALLOW_16X16`
  / `ALLOW_32X32` / `TX_MODE_SELECT` respectively).
* For each active slab, reads an outer `L(1) update_probs` flag.
* If 1, walks the nested `(i, j, k, l, m)` sweep — `BLOCK_TYPES=2 ×
  REF_TYPES=2 × COEF_BANDS=6 × maxL(k) × UNCONSTRAINED_NODES=3`
  cells where `maxL = (k == 0) ? 3 : 6` (band 0 has only 3 valid
  previous-coef contexts; the §10 listing trails it with `{0, 0, 0}
  // unused` rows).
* Each cell becomes `read_diff_update_prob( coder, cell )` against
  the running `coef_probs[ txSz ][ i ][ j ][ k ][ l ][ m ]` table.

When `update_probs == 1` the inner sweep traverses `2 × 2 × (3 + 5
× 6) × 3 = 396` cells per active tx-size, totalling up to `4 × 396
= 1584` `diff_update_prob` calls for a fully-active
`TX_MODE_SELECT` frame.

Initial probabilities come from the §10 `default_coef_probs[
TX_SIZES ][ BLOCK_TYPES ][ REF_TYPES ][ COEF_BANDS ][
PREV_COEF_CONTEXTS ][ UNCONSTRAINED_NODES ]` listing (1728 u8
entries) transcribed verbatim into a new `DEFAULT_COEF_PROBS`
constant in `src/coef_probs.rs`. The public type alias `CoefProbs`
names the 6D shape so callers of [`parse_compressed_header`] can
match on the new `Vp9CompressedHeader::coef_probs` field.

The §6.3.7 walker is **structural-parse only**: it advances the
§9.2 Boolean coder by exactly the bits the spec dictates and
updates the running `coef_probs` table, but the actual entropy
decode of residual coefficients (§6.4) still lands in a later
round.

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

The inter-only §6.3.9+ syntax (`read_inter_mode_probs`,
`read_interp_filter_probs`, `read_is_inter_probs`,
`frame_reference_mode`, `mv_probs`) remains deferred — those fire
only on `FrameIsIntra == 0` and need reference-buffer state which
the uncompressed-header walker still rejects with
`Error::Unsupported`.

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

* `cargo test`: 171 unit tests + 20 integration tests (8 in
  `tests/compressed_header.rs` plus 12 in
  `tests/uncompressed_header.rs`).
* Round-13 additions: 15 unit tests covering `coefband_4x4` /
  `coefband_8x8plus` against the §10 listing (21-entry prefix + all-`5`
  tail) and the `coef_band` tx-size dispatch, the §9.3.2
  `token_cache_neighbours` derivation (DC origin, interior DCT_DCT,
  the `DCT_ADST` / `ADST_DCT` double-neighbour variants, first-row /
  first-column fallbacks, and the `n = 4 << txSz` 8x8 width scaling),
  `build_token_probs` node mapping (node 0 → `cell[1]`, node 1 →
  `cell[2]`, node `2..=9` → `pareto`), and the `tokens( )` driver
  itself: a zero-buffer immediate EOB (returns `nonzero = false`, all
  `Tokens` zeroed), the `ZERO_TOKEN`-clears-`checkEob` block fill
  (`nonzero = true`, explicit zeros), a lockstep equality against an
  independent `read_coef_token` walk over the same scan / buffer
  (`Tokens` + `TokenCache` + return value), the DC non-zero-context
  cell routing (a non-zero strip selects the `ctx = 2` cell), the
  trailing `c..segEob` zero-fill on an 8x8 block, and the
  `NonzeroContext::new` all-zero invariant.
* Round-12 additions: 10 unit tests covering every §10.1 scan table's
  spec length, the permutation invariant (each raster position appears
  exactly once — the property the §6.4.24 loop relies on to zero
  untouched coefficients), the DC-first invariant, §10.1 listing
  anchors (first four + last entry of each table), the §6.4.25
  `txType` → table selection for 4x4 / 8x8 / 16x16, the
  `TX_32X32`-always-default and chroma-forces-`DCT_DCT` first-half
  overrides, and the `16 << (txSz << 1)` `segEob` scan-length match.
* Round-11 additions: 10 unit tests covering the `mode2txfm_map` intra
  prefix vs the §10.5 listing, the local `clip1` range clamping, an
  all-zero `Tokens` block leaving the prediction unchanged, a DC-only
  `Tokens` block adding a flat residual to a flat prediction, step-4
  `Clip1` saturation at both the bit-depth max and zero, the
  `TX_32X32` `dqDenom = 2` halving exercised through the real
  `reconstruct_block`, and three `reconstruct_intra_block` end-to-end
  cases (DC_PRED + zero tokens equalling the pure DC prediction,
  DC_PRED + a known DC residual reconstructing the expected pixels,
  and the lossless WHT path).
* Round-10 additions: 16 unit tests covering `PredMode::from_raw`
  round-tripping the §7.4.5 numbering (and rejecting 10+), the local
  `round2` / `clip1` helpers against their §3 definitions, `V_PRED`
  copying `aboveRow` down and `H_PRED` copying `leftCol` across, all
  four `DC_PRED` neighbour cases (both / left-only / above-only /
  none), `TM_PRED`'s `Clip1(aboveRow[j] + leftCol[i] - aboveRow[-1])`
  formula (including out-of-range clipping), a constant-neighbour
  invariant that collapses every directional mode to the neighbour
  constant across all four transform sizes, the `D207_PRED` bottom-row
  / step-2 formula, the `notOnRight`-gated upper-right extension of
  `aboveRow` (enabled only for `txSz == 0`) via `D45_PRED`, and the
  `Min(maxX, .)` plane-edge clamping of neighbour reads.
* Round-8 additions: 13 unit tests covering the `DC_QLOOKUP` /
  `AC_QLOOKUP` shape (256 entries per row) and §8.6.1 listing
  anchors (first/last entry of every row plus two interior anchors),
  `qlookup_row` bit-depth mapping (8→0 / 10→1 / 12→2), `clip3`
  both-end clamping, `dc_q` / `ac_q` index clipping + bit-depth row
  selection, `get_qindex` for the segmentation-off / delta-update /
  absolute-update paths (with the `Clip3` clamps), `seg_feature_active`
  needing both `segmentation_enabled` and the per-segment bit,
  `get_dc_quant` / `get_ac_quant` plane-delta selection (luma AC has
  no delta), a segment-overridden qindex threaded through both, and
  the high-bit-depth divergence of the same qindex across the three
  table rows.
* Round-6 additions: 7 unit tests covering
  `tx_mode_to_biggest_tx_size` against the §10.5 listing,
  `read_coef_probs` zero-buffer passthrough for both `ONLY_4X4`
  (1-slab) and `TX_MODE_SELECT` (4-slab) paths plus an across-mode
  sweep, `DEFAULT_COEF_PROBS` shape (4 × 2 × 2 × 6 × 6 × 3 = 1728
  entries) + four hand-picked anchor values from the §10 listing,
  the band-0 unused-row sentinel invariant (every `(txSz, i, j)`
  triple's band 0 has `{0, 0, 0}` at contexts 3..6), and the inner
  sweep cell-count check `2 × 2 × (3 + 5 × 6) × 3 = 396`. 2 new
  end-to-end integration tests (`tests/compressed_header.rs`)
  splicing a TX_MODE_SELECT frame and an ONLY_4X4 frame through
  the full pipeline and verifying §10 default anchors survive the
  zero-buffer §6.3.7 sweep verbatim.
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

1. The §6.4.21 `residual()` plane / sub-block driver — the outer walk
   over planes and 4x4 sub-blocks that owns the `AboveNonzeroContext` /
   `LeftNonzeroContext` write-back across a whole frame, drives the
   round-13 §6.4.24 `tokens()` per-block decode, and feeds the round-11
   §8.6.2 `reconstruct_block` / `reconstruct_intra_block` with real
   per-block `Tokens` arrays, availability flags, and segment/quantizer
   state — followed by the per-block mode-info decode (`y_mode` /
   `sub_modes` / `tx_size` / `skip` / `segment_id`) that the residual
   driver and `predict_intra` consume, then a public single-frame intra
   decode path. (The §6.4.24 `tokens()` per-block driver landed in
   round 13; the §8.6.2 reconstruct driver in round 11; the §6.4.25
   scan-order selection in round 12.)
2. Inter (non-intra-only) header path — `frame_size_with_refs`,
   `allow_high_precision_mv`, `read_interpolation_filter` — plus
   the inter-only §6.3.9–§6.3.16 syntax (`read_inter_mode_probs`,
   `read_interp_filter_probs`, `read_is_inter_probs`,
   `frame_reference_mode`, `mv_probs`) once reference-buffer state
   is in place.
3. Per-tile partition-tree walk (§6.4) including the §6.4.21
   `residual()` driver that finally consumes the round-7 tokens and
   the round-6 `coef_probs` tables, plus the per-block mode-info
   decode that feeds `predict_intra`.
4. Inter prediction (§8.5.2), loop filter (§8.8), multi-tile, then
   encoder paths.

## License

MIT. See `LICENSE`.
