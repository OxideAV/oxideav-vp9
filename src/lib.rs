//! # oxideav-vp9
//!
//! Pure-Rust, clean-room VP9 codec crate, implemented against the VP9
//! Bitstream & Decoding Process Specification v0.7. [`decode_vp9`] /
//! [`decode_intra_frame`] decode intra (key / intra-only) frames
//! end-to-end — §6.2 / §6.3 headers, the §6.4 tile + partition +
//! block walk, §8.5.1 intra prediction, §8.6 dequantization, §8.7
//! inverse transforms and the §8.8 loop filter. [`decode_vp9_sequence`]
//! decodes a multi-frame stream (keyframe + inter / P-frames) end-to-end,
//! adding the §6.4.11 inter mode-info decode, the §6.5 motion-vector
//! reference search, the §8.5.2 inter prediction process, and the §8.10
//! reference-buffer update threaded across frames.
//!
//! ## Cumulative scope (rounds 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20 + 21 + 22 + 37 + 244 + 250 + 253 + 255 + 259 + … + 284)
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
//! * Round 19 lands the §6.4.3 recursive `decode_partition()` driver
//!   itself, composing the round-18 [`partition::decode_partition_type`]
//!   primitive with the §10.2 `subsize_lookup` traversal and the §6.4.3
//!   tail write-back into the `AbovePartitionContext[ ]` /
//!   `LeftPartitionContext[ ]` strips. The driver walks the §6.4.3
//!   listing line-for-line: the `(r >= MiRows || c >= MiCols)` quadrant
//!   short-circuit, the `num8x8` / `halfBlock8x8` / `hasRows` /
//!   `hasCols` derivation, the partition decode via the round-18
//!   primitive, the four-way dispatch on the `PARTITION_*` value (HORZ
//!   second-leaf gated by `hasRows`; VERT second-leaf gated by
//!   `hasCols`; SPLIT recursing in spec order TL → TR → BL → BR), and
//!   the §6.4.3 tail write-back gated by `bsize == BLOCK_8X8 ||
//!   partition != PARTITION_SPLIT`. `PartitionContextState` exposes the
//!   `Sb64Cols * 8` / `Sb64Rows * 8` strips with the §7.4 zero-reset
//!   and the §6.4.2 `clear_left( )` helper; `PartitionProbsKind` selects
//!   between the keyframe [`partition::KF_PARTITION_PROBS`] direct path
//!   and an inter-frame running `partition_probs[ ]` table per the
//!   §9.3.2 listing. Leaf blocks (the §6.4.4 `decode_block( )` call
//!   sites — `mode_info` / `residual` not yet wired) are accumulated
//!   into a caller-supplied `Vec<LeafBlock>` log in §6.4.3 traversal
//!   order; the recursion is validated against three hand-built
//!   bitstreams (single-leaf 64x64 NONE, four-leaf SPLIT-into-32x32-NONE,
//!   mixed HORZ + VERT quadrants), produced by a test-only minimal
//!   range encoder mirroring the §9.2.2 decode steps inverse-by-inverse.
//!   The §6.3 `read_partition_probs( )` compressed-header sweep, the
//!   §6.4.4 `decode_block( )` mode-info + residual wiring, and the
//!   §6.4.2 `decode_tile( )` outer loop all land in later rounds; the
//!   round-19 surface is internal-only.
//! * Round 22 lands the §6.3.11 `read_is_inter_probs( )`
//!   compressed-header sweep — the unconditional 4-element
//!   (`IS_INTER_CONTEXTS = 4`) `diff_update_prob` walk over the §10.5
//!   `default_is_inter_prob[ IS_INTER_CONTEXTS ] = { 9, 102, 187, 225 }`
//!   initials. Returns the post-sweep `is_inter_prob[ ]` table that
//!   the round-21 §6.4.13 `read_is_inter( )` per-block decoder
//!   consumes via the §9.3.2 ctx. The `FrameIsIntra == 0`-gated outer
//!   dispatch in `parse_compressed_header` still skips the call —
//!   §6.3.9 / §6.3.10 / §6.3.12..§6.3.17 must land first so the coder
//!   cursor lines up across the whole inter branch. Single source of
//!   truth: the §10.5 default table is re-exported from
//!   [`mode_info::DEFAULT_IS_INTER_PROB`] (round 21) into
//!   `compressed::DEFAULT_IS_INTER_PROB_TABLE`. The round-22 surface
//!   is internal-only.
//! * Round 34 wires the §6.3 `if ( FrameIsIntra == 0 )` outer
//!   dispatch into a new public [`parse_compressed_header_inter`]
//!   entry point that composes the round-22..30 primitives in spec
//!   order:
//!   §6.3.9 [`compressed::read_inter_mode_probs`] →
//!   §6.3.10 [`compressed::read_interp_filter_probs`] gated on
//!   `interpolation_filter == SWITCHABLE` →
//!   §6.3.11 [`compressed::read_is_inter_probs`] →
//!   §6.3.12 [`compressed::frame_reference_mode`] (which also fires
//!   §6.3.18 [`compressed::setup_compound_reference_mode`] on the
//!   non-`SingleReference` arms) →
//!   §6.3.13 [`compressed::read_frame_reference_mode_probs`] →
//!   §6.3.14 [`compressed::read_y_mode_probs`] →
//!   §6.3.15 [`compressed::read_partition_probs`] →
//!   §6.3.16 [`compressed::mv_probs`] (which fires
//!   §6.3.17 [`compressed::update_mv_prob`] per cell and walks the
//!   high-precision tail when `allow_high_precision_mv == 1`).
//!   The intra prefix (§6.3.1 / §6.3.2 / §6.3.7 / §6.3.8) is
//!   factored out into a shared crate-local helper so both
//!   [`parse_compressed_header`] (intra-only / keyframe) and the new
//!   inter entry point reuse it bit-identically. New public types:
//!   [`Vp9CompressedHeaderInterInputs`] (the three §6.2-derived flags
//!   the inter tail needs), [`Vp9CompressedHeaderInter`] (the
//!   intra-shared prefix plus 9 inter-only probability tables +
//!   `reference_mode` decision + optional
//!   [`CompoundReferenceConfig`]), [`RefFrameSignBias`],
//!   [`ReferenceMode`], [`CompoundReferenceConfig`], [`MvProbs`]
//!   (all promoted from `pub(crate)` to `pub` since they surface in
//!   the result). Wiring the new entry point into the public
//!   [`decode_vp9`] still depends on the uncompressed-header walker
//!   accepting inter frames (currently still
//!   [`Error::Unsupported`]).
//! * Round 37 lands the §8.8.1 `loop_filter_frame_init( )` book-keeping
//!   primitive that builds the `LvlLookup[ MAX_SEGMENTS ][
//!   MAX_REF_FRAMES ][ MAX_MODE_LF_DELTAS ]` filter-strength table the
//!   §8.8.4 adaptive-strength consumer reads at every superblock
//!   raster step. Covers the four §8.8.1 steps verbatim: step 1
//!   `lvlSeg = loop_filter_level`, step 2 `seg_feature_active(
//!   SEG_LVL_ALT_L )` segment override (abs vs delta + step-2.c
//!   `Clip3` saturation), step 3 `delta_update == 0` per-segment
//!   broadcast, step 4 `delta_enabled == 1` per-(ref, mode) delta
//!   walk (with the line 5481 `INTRA / mode 0` write + lines 5482-5487
//!   `LAST..ALTREF / 0..MAX_MODE_LF_DELTAS - 1` cover). The
//!   `nShift = loop_filter_level >> 5` scaling (line 5468) and the
//!   final `Clip3( 0, MAX_LOOP_FILTER, … )` saturations (line 5481 /
//!   5486) are applied verbatim. The caller supplies resolved
//!   `loop_filter_ref_deltas[ 4 ]` / `loop_filter_mode_deltas[ 2 ]`
//!   arrays per §7.2's "previous value" rule (the §7.2
//!   `setup_past_independence` defaults are `[1, 0, -1, -1]` for
//!   refs and `[0, 0]` for modes). Public types:
//!   [`loop_filter_frame_init`] + [`LvlLookup`] +
//!   [`MAX_MODE_LF_DELTAS`] + [`MAX_LOOP_FILTER`]. The §8.8.2
//!   superblock raster walk, §8.8.3 `filter_size`, §8.8.4
//!   `adaptive_filter_strength`, and §8.8.5 sample-filtering
//!   primitives are deferred.
//! * Round 244 lifts §8.8.3 `filter_size( )` to a public leaf
//!   primitive — [`filter_size`] — covering the `baseSize` derivation
//!   (`txSz == TX_4X4 && is32Edge == 1 → TX_8X8`; otherwise
//!   `Min(TX_16X16, txSz)`) plus the vertical chroma-right-edge clip
//!   (`pass == 0 && sub_x == 1 && baseSize == TX_16X16 && (x >> 3)
//!   == MiCols - 1 → TX_8X8`) and horizontal chroma-bottom-edge clip
//!   (`pass == 1 && sub_y == 1 && baseSize == TX_16X16 && (y >> 3)
//!   == MiRows - 1 → TX_8X8`) per `vp9-spec.txt` §8.8.3 lines
//!   5587-5625. Public surface: [`filter_size`] + the four §7.4.8
//!   transform-size constants [`TX_4X4`], [`TX_8X8`], [`TX_16X16`],
//!   [`TX_32X32`] (verbatim from §7.4.8 lines 3937-3940) + the two
//!   pass-direction constants [`PASS_VERTICAL`] and [`PASS_HORIZONTAL`].
//! * Round 250 lifts §8.8.4 `adaptive_filter_strength( )` to a public
//!   leaf primitive — [`adaptive_filter_strength`] — covering the
//!   four §8.8.4 steps verbatim. Step 1 reads `LvlLookup[ segment ][
//!   ref ][ modeType ]` from the round-37 §8.8.1
//!   [`loop_filter_frame_init`] output, deriving `modeType = 1` for
//!   `NEARESTMV` / `NEARMV` / `NEWMV` per `vp9-spec.txt` lines
//!   5637-5638 and `modeType = 0` for intra modes and `ZEROMV`. Step
//!   2 picks `shift ∈ {0, 1, 2}` from `loop_filter_sharpness`. Step
//!   3 yields `limit` via the sharpness-gated `Clip3( 1, 9 -
//!   loop_filter_sharpness, lvl >> shift )` (sharpness > 0) or
//!   `Max( 1, lvl >> shift )` (sharpness = 0). Step 4 sets `blimit =
//!   2 * (lvl + 2) + limit`. Step 5 sets `thresh = lvl >> 4`. Public
//!   surface: [`adaptive_filter_strength`] + [`FilterStrength`] +
//!   [`mode_to_mode_type`] + the four §7.4.11 inter-mode constants
//!   [`NEARESTMV`], [`NEARMV`], [`ZEROMV`], [`NEWMV`] (verbatim
//!   from `vp9-spec.txt` §7.4.11 lines 3957-3961). The §8.8.2
//!   superblock raster walk that calls this primitive at every
//!   `(loopRow, loopCol)` step and the §8.8.5 sample-filter pass
//!   that consumes its output both remain deferred.
//! * Round 253 lifts §8.8.5.1 `filter mask process` to a public leaf
//!   primitive — [`filter_mask`] — covering the four mask outputs
//!   verbatim. The primitive accepts a 16-sample stencil
//!   [`FilterMaskSamples`] (`p7`..`p0`, `q0`..`q7`) and the §8.8.4
//!   [`FilterStrength`] tuple, plus `filterSize` and `BitDepth`. It
//!   returns [`FilterMask`] with `hev_mask` from `vp9-spec.txt` lines
//!   5730-5734, `filter_mask` from lines 5737-5750 (the seven inner
//!   abs-diff thresholds plus the `Abs(p0 - q0)*2 + Abs(p1 - q1)/2`
//!   boundary term), `flat_mask` from lines 5753-5774 (six diffs
//!   relative to `p0` / `q0`, gated by `filterSize >= TX_8X8`), and
//!   `flat_mask2` from lines 5777-5792 (eight outer diffs relative
//!   to `p0` / `q0`, gated by `filterSize >= TX_16X16`). The
//!   bit-depth scalings `threshBd = thresh << (BitDepth - 8)`,
//!   `limitBd`, `blimitBd`, and `thresholdBd = 1 << (BitDepth - 8)`
//!   are all honoured verbatim. The §8.8.5 outer driver that builds
//!   the stencil from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k
//!   ]` and the §8.8.5.2 / §8.8.5.3 sample-filter primitives that
//!   read the mask remain deferred.
//! * Round 255 lifts §8.8.5.2 `narrow filter process` to a public
//!   leaf primitive — [`narrow_filter`] — covering the per-edge
//!   sample mutation `vp9-spec.txt` §8.8.5.2 lines 5795-5853
//!   verbatim. The primitive accepts a 4-sample stencil
//!   [`NarrowFilterSamples`] (`p1`, `p0`, `q0`, `q1`), the
//!   §8.8.5.1 `hev_mask` boolean, and `BitDepth`. It returns
//!   [`NarrowFilterOutput`] (`op1`, `op0`, `oq0`, `oq1`). Both the
//!   `hev_mask == 1` "two-sample" branch (lines 5809-5811, modifies
//!   only `op0` / `oq0`) and the `hev_mask == 0` "four-sample"
//!   branch (lines 5806-5808 + 5846-5852, modifies all four with
//!   the `Round2( filter1, 1 )` half-strength pass into `op1` /
//!   `oq1`) are wired. The internal `filter4_clamp` helper (line
//!   5825) clips into the bit-depth-scaled signed range
//!   `[-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1]`; the
//!   `0x80 << (BitDepth - 8)` working-range offset (lines 5834-5837)
//!   is applied and undone verbatim. The §8.8.5 outer driver that
//!   reads the stencil from `CurrFrame[ plane ][ y +/- dy*k ][ x
//!   +/- dx*k ]` and writes the four outputs back and the §8.8.2
//!   superblock raster walk remain deferred.
//! * Round 259 lifts §8.8.5.3 `wide filter process` to a public
//!   leaf primitive — [`wide_filter`] — covering the per-edge
//!   sample mutation `vp9-spec.txt` §8.8.5.3 lines 5855-5888
//!   verbatim. The primitive accepts a 16-sample stencil
//!   [`WideFilterSamples`] (`p7`..`p0`, `q0`..`q7`), a `log2_size`
//!   ∈ `{3, 4}` per the §8.8.5 dispatch table (lines 5681-5684),
//!   and `BitDepth` (carried for API symmetry — the §8.8.5.3
//!   listing makes no reference to `BitDepth`). Returns
//!   [`WideFilterOutput`] with up to 14 mutated samples (`op6`..
//!   `op0`, `oq0`..`oq6`). The §8.8.5.3 kernel
//!   `F[ i ] = Round2( CurrFrame[i] + sum_{j=-n..n} CurrFrame[Clip3(-(n+1), n, i+j)], log2Size )`
//!   is implemented verbatim with `n = (1 << (log2Size - 1)) - 1`
//!   (lines 5864-5865) and the `Clip3` edge-replication extension
//!   (line 5879). For `log2_size == 3` only the inner six
//!   positions (`p2..p0`, `q0..q2`) are mutated; the outer eight
//!   output fields echo the corresponding input so the caller can
//!   write all 14 fields unconditionally. The §8.8.5 outer driver
//!   (which builds the stencil, picks `log2_size`, and writes the
//!   outputs back) and the §8.8.2 superblock raster walk both remain
//!   deferred.
//! * Round 284 composes everything above into the public
//!   [`decode_vp9`] / [`decode_intra_frame`] entry points (module
//!   `decode_frame`): the §6.4 `decode_tiles( )` walk (per-tile
//!   §9.2 coder bracket, §7.4.1 / §7.4.2 context resets), the
//!   §6.4.3 partition recursion firing a `LeafSink` at every §6.4.4
//!   `decode_block( )` site, the §6.4.6 mode-info → §6.4.21
//!   residual → §6.4.24 token → §8.6.2 reconstruct chain per block,
//!   the §6.4.4 frame-wide fan-out, the §8.8 loop filter over the
//!   reconstructed planes, and the §8.10 output crop into
//!   [`Vp9DecodedFrame`]. Byte-exact against the staged corpus
//!   fixtures (4:2:0 / 4:4:4, 8 / 10 / 12-bit, lossless WHT,
//!   multi-tile, segmentation AQ).
//! * Both key-frame and intra-only inter-frame paths are walked.
//! * Inter-frame (non-intra-only) headers — `frame_size_with_refs`,
//!   motion-vector / interpolation-filter flags — return
//!   [`Error::Unsupported`] for now; they need reference-buffer
//!   state.
//!
//! Inter-frame decode (reference buffers, §8.5.2 inter prediction,
//! §8.4 probability adaptation) lands in subsequent rounds.
//!
//! ## Provenance
//!
//! Clean-room, single source of truth: `docs/video/vp9/vp9-spec.txt`
//! (the v0.7 specification snapshot).

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

mod adaptive_filter_strength;
mod bitreader;
mod block_inter_pred;
mod block_writer;
mod bool_coder;
mod bool_encoder;
mod coef_probs;
mod compressed;
// The §6.3 compressed-header writer is built ahead of the keyframe
// encoder that consumes it; the allowance is removed when that lands.
#[allow(dead_code)]
mod compressed_writer;
mod decode_block;
mod decode_frame;
mod dequant;
mod filter_mask;
mod filter_size;
mod frame_loop_filter;
mod frame_writer;
mod fwd_transform;
mod header;
// The §6.2 uncompressed-header writer is built ahead of the keyframe
// encoder that consumes it; the allowance is removed when that lands.
#[allow(dead_code)]
mod header_writer;
mod idct;
mod inter_decode;
mod inter_mv;
mod inter_pred;
mod intra;
mod loop_filter;
mod mode_info;
// The keyframe intra mode-info writer is built ahead of the tile-data
// encoder that consumes it; the allowance is removed when that lands.
mod inter_block_writer;
mod inter_mode_writer;
#[allow(dead_code)]
mod mode_writer;
mod mv;
mod mv_ref;
mod mv_writer;
mod narrow_filter;
mod partition;
mod partition_writer;
mod pixel_encoder;
mod prob_adapt;
mod reconstruct;
mod ref_buffer;
mod residual;
mod residual_writer;
mod sample_filtering;
mod scan;
mod superblock_filter;
mod superblock_loop_filter;
mod superframe;
// The §6.4.24 token writer is built ahead of the residual encoder that
// consumes it; the allowance is removed when that lands.
#[allow(dead_code)]
mod token_writer;
mod tokens;
mod wide_filter;

pub use adaptive_filter_strength::{
    adaptive_filter_strength, mode_to_mode_type, FilterStrength, NEARESTMV, NEARMV, NEWMV, ZEROMV,
};
pub use coef_probs::CoefProbs;
pub use compressed::{
    parse_compressed_header, parse_compressed_header_inter, CompoundReferenceConfig, MvProbs,
    RefFrameSignBias, ReferenceMode, TxMode, Vp9CompressedHeader, Vp9CompressedHeaderInter,
    Vp9CompressedHeaderInterInputs,
};
pub use decode_frame::{decode_intra_frame, decode_vp9_sequence, Vp9DecodedFrame};
pub use filter_mask::{filter_mask, FilterMask, FilterMaskSamples};
pub use filter_size::{
    filter_size, PASS_HORIZONTAL, PASS_VERTICAL, TX_16X16, TX_32X32, TX_4X4, TX_8X8,
};
pub use frame_loop_filter::{frame_loop_filter, CurrFrame};
pub use header::{
    parse_uncompressed_header, parse_uncompressed_header_with_refs, ColorConfig, ColorSpace,
    FrameType, LoopFilterParams, QuantizationParams, RefFrameState, SegmentationParams, TileInfo,
    Vp9FrameHeader, MAX_SEGMENTS, SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_SIGNED,
    SEG_LVL_MAX,
};
pub use loop_filter::{loop_filter_frame_init, LvlLookup, MAX_LOOP_FILTER, MAX_MODE_LF_DELTAS};
pub use narrow_filter::{narrow_filter, NarrowFilterOutput, NarrowFilterSamples};
pub use partition::tile_payload_sizes;
pub use sample_filtering::{sample_filtering, SampleFilterOutput, SampleFilterSamples};
pub use superblock_filter::{
    superblock_filter_edge, superblock_filter_geometry, SuperblockFilterEdge,
    SuperblockFilterGeometry, SuperblockFilterMi,
};
pub use superblock_loop_filter::{
    superblock_loop_filter, SuperblockFilterFrame, SuperblockFilterPlane,
};
pub use superframe::split_superframe;
pub use wide_filter::{wide_filter, WideFilterOutput, WideFilterSamples};

/// Crate-local error type.
///
/// [`decode_vp9`] / [`decode_intra_frame`] decode a single intra (key /
/// intra-only) frame; an inter frame fed to them returns
/// [`Error::Unsupported`] because it needs reference-buffer state — use
/// [`decode_vp9_sequence`] to decode a multi-frame stream (keyframe +
/// P-frames) end-to-end. [`encode_vp9`] still returns
/// [`Error::NotImplemented`].
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

/// Decode one VP9 intra frame to planar YUV bytes.
///
/// `bytes` is a single frame's payload (e.g. one IVF frame body):
/// §6.2 uncompressed header + §6.3 compressed header + §6.4 tile
/// data. Key frames and intra-only frames decode end-to-end (mode
/// info → intra prediction → coefficient decode → dequant → inverse
/// transform → reconstruction → §8.8 loop filter); a standalone inter
/// frame returns [`Error::Unsupported`] (it needs reference-buffer
/// state — use [`decode_vp9_sequence`] for a keyframe + P-frame stream).
///
/// The output is planar Y then U then V at the §8.10 cropped extents
/// — one byte per sample for 8-bit content, little-endian `u16` pairs
/// for 10 / 12-bit. For plane geometry and native `u16` samples use
/// [`decode_intra_frame`] directly.
pub fn decode_vp9(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Ok(decode_intra_frame(bytes)?.to_planar_bytes())
}

/// Encode a single VP9 keyframe of size `width × height` — **lossless**:
/// the returned frame decodes byte-exact back to `pixels`.
///
/// `pixels` is an 8-bit 4:2:0 planar frame (`Y` then `U` then `V`, each
/// chroma plane `ceil(w/2) × ceil(h/2)` — the [`decode_vp9`] output
/// layout). The encoder codes a lossless (`base_q_idx == 0`) keyframe:
/// per transform block it predicts with the decoder's own §8.5.1 intra
/// process over the shared reconstruction state, forward-WHT-transforms
/// the residual (the exact inverse of the §8.7.2 lossless inverse
/// transform), and replays the decoder's §8.6.2 reconstruction — so
/// `decode_vp9( encode_vp9( pixels ) ) == pixels` holds bit-for-bit.
///
/// Returns [`Error::Unsupported`] for `width` / `height` outside
/// `1..=65536`, or when `pixels` is too short for a `width × height`
/// 4:2:0 frame.
pub fn encode_vp9(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossless_420(pixels, width, height)
}

/// Encode an 8-bit **4:4:4** planar frame (`Y` then `U` then `V`, each
/// plane `width × height`) into a lossless profile-1 VP9 keyframe.
///
/// Like [`encode_vp9`], the output decodes byte-exact back to `pixels`
/// (chroma at full resolution). Returns [`Error::Unsupported`] for
/// degenerate dimensions or a too-short buffer.
pub fn encode_vp9_lossless_444(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossless_444(pixels, width, height)
}

/// Encode a 10/12-bit planar frame (native `u16` samples, `Y` then `U`
/// then `V`) into a lossless high-bit-depth VP9 keyframe.
///
/// `subsample == true` selects 4:2:0 (profile 2, chroma planes
/// `ceil(w/2) × ceil(h/2)`); `false` selects 4:4:4 (profile 3, chroma at
/// full resolution). The output decodes **sample-exact** back to
/// `samples` (compare against [`Vp9DecodedFrame`]'s native `u16`
/// planes). Returns [`Error::Unsupported`] when `bit_depth` is not 10 or
/// 12, any sample exceeds the bit-depth range, the buffer is too short,
/// or the dimensions are degenerate.
pub fn encode_vp9_lossless_hbd(
    samples: &[u16],
    width: u32,
    height: u32,
    bit_depth: u8,
    subsample: bool,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossless_hbd(samples, width, height, bit_depth, subsample)
}

/// Encode a **sequence** of 8-bit 4:2:0 planar frames into a lossless
/// VP9 stream: a keyframe followed by `ZEROMV` inter (P-)frames, each
/// coding the exact `frame − prediction` residual against the previous
/// frame (single `LAST` reference, §8.10 slot 0 refreshed per frame).
///
/// Every frame of `decode_vp9_sequence( encode_vp9_lossless_sequence(
/// frames ) )` equals its input **byte-exact**. Each element of `frames`
/// is one planar frame in the [`decode_vp9`] layout (`Y` then `U` then
/// `V`, chroma `ceil(w/2) × ceil(h/2)`).
///
/// Returns [`Error::Unsupported`] for an empty sequence, degenerate
/// dimensions, or any too-short frame buffer.
pub fn encode_vp9_lossless_sequence(
    frames: &[&[u8]],
    width: u32,
    height: u32,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossless_420(frames, width, height)
}

/// Encode a minimal VP9 **inter (P-frame) sequence** of `width × height`:
/// a keyframe followed by `num_pframes` all-skip, single-reference-`LAST`,
/// `ZEROMV` P-frames.
///
/// Each P-frame copies its co-located samples from the `LAST` reference
/// (zero motion, no residual), so the whole sequence reconstructs to the
/// keyframe's flat-DC fill — a *structurally* complete decodable inter
/// sequence rather than a pixel-accurate encode. Returns the per-frame
/// coded byte buffers in decode order (the keyframe first), each a
/// complete VP9 frame that [`decode_vp9_sequence`] threads through the
/// §8.10 reference buffers.
///
/// This exercises the §6.2 inter uncompressed header, the §6.3 inter
/// compressed header, and the §6.4.11 / §6.4.16 inter block writer
/// end-to-end; the P-frame reconstruction is validated byte-exact against
/// the keyframe through the in-crate decoder.
///
/// Returns [`Error::Unsupported`] for `width` / `height` outside
/// `1..=65536`.
pub fn encode_vp9_pframe_sequence(
    width: u32,
    height: u32,
    num_pframes: usize,
) -> Result<Vec<Vec<u8>>, Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    let keyframe = frame_writer::encode_keyframe_all_skip_dc(width, height)?;
    let mut frames = Vec::with_capacity(1 + num_pframes);
    frames.push(keyframe);
    for _ in 0..num_pframes {
        let hdr = frame_writer::inter_pframe_header(width, height);
        frames.push(frame_writer::assemble_inter_frame_all_skip_zeromv(&hdr)?);
    }
    Ok(frames)
}

/// No-op codec registration — round 1 has nothing to register into the
/// runtime context until the decode / encode paths land.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("vp9", register);

#[cfg(test)]
mod encode_roundtrip_tests {
    use super::*;

    /// `encode_vp9` produces a stream that `decode_vp9` reconstructs
    /// **byte-exact** back to the input (lossless contract).
    #[test]
    fn encode_then_decode_64x64() {
        let w = 64u32;
        let h = 64u32;
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let pixels: Vec<u8> = (0..(w * h) as usize + 2 * cw * ch)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        let stream = encode_vp9(&pixels, w, h).expect("encode");
        let frame = decode_intra_frame(&stream).expect("decode");
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        let bytes = decode_vp9(&stream).expect("decode_vp9");
        assert_eq!(bytes, pixels, "lossless round-trip not byte-exact");
    }

    /// A non-square, non-multiple-of-8 frame round-trips byte-exact.
    #[test]
    fn encode_then_decode_40x24() {
        let (w, h) = (40u32, 24u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let pixels: Vec<u8> = (0..(w * h) as usize + 2 * cw * ch)
            .map(|i| ((i * 89 + 3) % 256) as u8)
            .collect();
        let stream = encode_vp9(&pixels, w, h).expect("encode");
        let frame = decode_intra_frame(&stream).expect("decode");
        assert_eq!((frame.width, frame.height), (w, h));
        assert_eq!(decode_vp9(&stream).expect("planar"), pixels);
    }

    /// Too-short input is rejected.
    #[test]
    fn encode_rejects_short_input() {
        assert_eq!(
            encode_vp9(&[0u8; 10], 64, 64).unwrap_err(),
            Error::Unsupported
        );
    }

    /// `encode_vp9_pframe_sequence` produces a keyframe + P-frame stream
    /// that `decode_vp9_sequence` reconstructs byte-exact: every P-frame
    /// copies the keyframe's flat-DC reference.
    #[test]
    fn encode_pframe_sequence_decodes_byte_exact() {
        let frames = encode_vp9_pframe_sequence(64, 64, 3).expect("encode seq");
        assert_eq!(frames.len(), 4); // keyframe + 3 P-frames.
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode seq");
        assert_eq!(decoded.len(), 4);
        let kf = &decoded[0];
        assert!(kf.y.iter().all(|&s| s == 128), "keyframe not flat 128");
        for (i, f) in decoded.iter().enumerate().skip(1) {
            assert_eq!(f.y, kf.y, "p-frame {i} luma != reference");
            assert_eq!(f.u, kf.u, "p-frame {i} U != reference");
            assert_eq!(f.v, kf.v, "p-frame {i} V != reference");
        }
    }

    /// A zero-pframe request yields just the keyframe.
    #[test]
    fn encode_pframe_sequence_zero_pframes_is_keyframe_only() {
        let frames = encode_vp9_pframe_sequence(64, 64, 0).expect("encode seq");
        assert_eq!(frames.len(), 1);
        let decoded = decode_vp9_sequence(&[&frames[0]]).expect("decode");
        assert_eq!((decoded[0].width, decoded[0].height), (64, 64));
    }

    /// Degenerate geometry is rejected.
    #[test]
    fn encode_pframe_sequence_rejects_zero_dim() {
        assert_eq!(
            encode_vp9_pframe_sequence(0, 64, 1).unwrap_err(),
            Error::Unsupported
        );
    }
}
