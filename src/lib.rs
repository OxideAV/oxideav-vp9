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
//!   primitive — `filter_size` — covering the `baseSize` derivation
//!   (`txSz == TX_4X4 && is32Edge == 1 → TX_8X8`; otherwise
//!   `Min(TX_16X16, txSz)`) plus the vertical chroma-right-edge clip
//!   (`pass == 0 && sub_x == 1 && baseSize == TX_16X16 && (x >> 3)
//!   == MiCols - 1 → TX_8X8`) and horizontal chroma-bottom-edge clip
//!   (`pass == 1 && sub_y == 1 && baseSize == TX_16X16 && (y >> 3)
//!   == MiRows - 1 → TX_8X8`) per `vp9-spec.txt` §8.8.3 lines
//!   5587-5625. Public surface: `filter_size` + the four §7.4.8
//!   transform-size constants [`TX_4X4`], [`TX_8X8`], [`TX_16X16`],
//!   [`TX_32X32`] (verbatim from §7.4.8 lines 3937-3940) + the two
//!   pass-direction constants [`PASS_VERTICAL`] and [`PASS_HORIZONTAL`].
//! * Round 250 lifts §8.8.4 `adaptive_filter_strength( )` to a public
//!   leaf primitive — `adaptive_filter_strength` — covering the
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
//!   surface: `adaptive_filter_strength` + [`FilterStrength`] +
//!   [`mode_to_mode_type`] + the four §7.4.11 inter-mode constants
//!   [`NEARESTMV`], [`NEARMV`], [`ZEROMV`], [`NEWMV`] (verbatim
//!   from `vp9-spec.txt` §7.4.11 lines 3957-3961). The §8.8.2
//!   superblock raster walk that calls this primitive at every
//!   `(loopRow, loopCol)` step and the §8.8.5 sample-filter pass
//!   that consumes its output both remain deferred.
//! * Round 253 lifts §8.8.5.1 `filter mask process` to a public leaf
//!   primitive — `filter_mask` — covering the four mask outputs
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
//!   leaf primitive — `narrow_filter` — covering the per-edge
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
//!   leaf primitive — `wide_filter` — covering the per-edge
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
mod codec;
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
mod recon_filter;
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

// Stable entry points: single-frame + multi-frame decode and the
// decoded-frame type in their visible signatures.
pub use codec::{
    make_decoder, make_encoder, pixel_format_for_triple, Vp9Decoder, Vp9Encoder, Vp9EncoderOptions,
};

pub use decode_frame::{
    decode_intra_frame, decode_vp9_sequence, decode_vp9_sequence_with, Vp9DecodedFrame,
    Vp9SequenceDecoder,
};
// Stable container utility: the Annex B superframe split that a caller
// runs before feeding frames to `decode_vp9_sequence`.
pub use superframe::split_superframe;

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use adaptive_filter_strength::{
    adaptive_filter_strength, mode_to_mode_type, FilterStrength, NEARESTMV, NEARMV, NEWMV, ZEROMV,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use coef_probs::CoefProbs;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use compressed::{
    parse_compressed_header, parse_compressed_header_inter, CompoundReferenceConfig, MvProbs,
    RefFrameSignBias, ReferenceMode, TxMode, Vp9CompressedHeader, Vp9CompressedHeaderInter,
    Vp9CompressedHeaderInterInputs,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use filter_mask::{filter_mask, FilterMask, FilterMaskSamples};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use filter_size::{
    filter_size, PASS_HORIZONTAL, PASS_VERTICAL, TX_16X16, TX_32X32, TX_4X4, TX_8X8,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use frame_loop_filter::{frame_loop_filter, CurrFrame};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use header::{
    parse_uncompressed_header, parse_uncompressed_header_with_refs, ColorConfig, ColorSpace,
    FrameType, LoopFilterParams, QuantizationParams, RefFrameState, SegmentationParams, TileInfo,
    Vp9FrameHeader, MAX_SEGMENTS, SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_SIGNED,
    SEG_LVL_MAX,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use loop_filter::{loop_filter_frame_init, LvlLookup, MAX_LOOP_FILTER, MAX_MODE_LF_DELTAS};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use narrow_filter::{narrow_filter, NarrowFilterOutput, NarrowFilterSamples};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use partition::tile_payload_sizes;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use sample_filtering::{sample_filtering, SampleFilterOutput, SampleFilterSamples};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use superblock_filter::{
    superblock_filter_edge, superblock_filter_geometry, SuperblockFilterEdge,
    SuperblockFilterGeometry, SuperblockFilterMi,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use superblock_loop_filter::{
    superblock_loop_filter, SuperblockFilterFrame, SuperblockFilterPlane,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
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
/// VP9 stream: a keyframe followed by inter (P-)frames, each coding
/// the exact `frame − prediction` residual against the previous frame
/// with `ZEROMV` / `NEWMV` integer motion search (single `LAST`
/// reference, §8.10 slot 0 refreshed per frame).
///
/// **Chain framing (default since round 445):** every P-frame is shown
/// and non-error-resilient, so the §7.2.6 `UsePrevFrameMvs` derivation
/// is 1 on the decode side — and the encoder models it, threading each
/// frame's §6.4.4 motion field into the next frame's §6.5.10 candidate
/// scan. Motion that persists at the same position across frames
/// reaches the previous-frame candidate and codes `NEARESTMV` /
/// `NEARMV` (no §6.4.20 mv-diff bits) even where the spatial
/// neighbours predict it wrongly, so temporally coherent motion codes
/// fewer bytes than the error-resilient framing. Callers that need
/// per-frame decode independence (§6.2 `error_resilient_mode = 1`)
/// opt out via [`encode_vp9_lossless_sequence_error_resilient`].
///
/// **Byte stability:** the coded bytes of this entry changed in round
/// 445 when the chain framing became the default; the pre-445 bytes
/// remain available, frozen, through the explicit
/// [`encode_vp9_lossless_sequence_error_resilient`] opt-out (the
/// staged self-encoded corpus fixtures pin that path's exact output).
/// Within one crate version every entry point remains
/// byte-deterministic: identical inputs give identical bytes.
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
    pixel_encoder::encode_sequence_lossless_chained_420(frames, width, height)
}

/// [`encode_vp9_lossless_sequence`] on the **classic error-resilient
/// framing** — the explicit opt-out from the round-445
/// chained-as-default promotion: every P-frame codes §6.2
/// `error_resilient_mode = 1`, so the §7.2.6 `UsePrevFrameMvs`
/// derivation is 0 on the decode side (no previous-frame motion
/// candidates; each frame's entropy state is independent per §7.2
/// `setup_past_independence( )`), and any single P-frame can be
/// decoded after a reference loss without entropy-state drift.
///
/// This path's coded bytes are **frozen**: they are the exact bytes
/// [`encode_vp9_lossless_sequence`] produced before round 445, and the
/// staged self-encoded corpus fixtures (e.g. `odd-dims-59x37`) pin
/// them byte-for-byte.
///
/// The lossless guarantee is identical to the default entry: every
/// frame of the decoded sequence equals its input **byte-exact**.
pub fn encode_vp9_lossless_sequence_error_resilient(
    frames: &[&[u8]],
    width: u32,
    height: u32,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossless_420(frames, width, height)
}

/// Historical alias of [`encode_vp9_lossless_sequence`]: since round
/// 445 the §7.2.6 chain framing IS the default sequence path, so the
/// two entries are the same encoder (kept so pre-445 callers of the
/// explicit `_chained` name keep their exact behavior and bytes).
pub fn encode_vp9_lossless_sequence_chained(
    frames: &[&[u8]],
    width: u32,
    height: u32,
) -> Result<Vec<Vec<u8>>, Error> {
    encode_vp9_lossless_sequence(frames, width, height)
}

/// Encode an 8-bit 4:2:0 planar frame into a **lossy** VP9 keyframe at
/// quantizer index `base_q_idx` (`1..=255` — smaller is higher quality;
/// use [`encode_vp9`] for lossless).
///
/// The encoder quantizes the forward-DCT residual with the §8.6.1
/// quantizers and replays the decoder's own §8.6.2 reconstruction as its
/// in-loop reference, so the decoder's output equals the encoder's
/// reconstruction bit-for-bit; distortion against the source is bounded
/// by the quantizer step, shrinking (and the stream growing) as
/// `base_q_idx` decreases.
///
/// Returns [`Error::Unsupported`] for `base_q_idx == 0` (the lossless
/// path), degenerate dimensions, or a too-short buffer.
pub fn encode_vp9_lossy(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_420(pixels, width, height, base_q_idx)
}

/// Encode an 8-bit **4:4:4** planar frame (`Y` then `U` then `V`, each
/// plane `width × height`) into a **lossy** profile-1 VP9 keyframe at
/// quantizer index `base_q_idx` (`1..=255`; use
/// [`encode_vp9_lossless_444`] for lossless).
///
/// Same decoder-mirror guarantee as [`encode_vp9_lossy`]: the decoded
/// output equals the encoder's in-loop reconstruction bit-for-bit
/// (elected §8.8 loop filter included), with distortion against the
/// source bounded by the §8.6.1 quantizer step. Returns
/// [`Error::Unsupported`] for `base_q_idx == 0`, degenerate dimensions,
/// or a too-short buffer.
pub fn encode_vp9_lossy_444(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_u8(pixels, width, height, base_q_idx, false, false)
}

/// Encode an 8-bit **4:2:2** planar frame (`Y` then `U` then `V`,
/// chroma planes `ceil(w/2) × height`) into a **lossy** profile-1 VP9
/// keyframe at quantizer index `base_q_idx` (`1..=255`).
///
/// 4:2:2 is the `subsampling_x = 1, subsampling_y = 0` §7.2.2 geometry
/// (profile 1 codes both subsampling flags in its §6.2.2
/// `color_config( )`). Same decoder-mirror guarantee and error cases
/// as [`encode_vp9_lossy_444`].
pub fn encode_vp9_lossy_422(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_u8(pixels, width, height, base_q_idx, true, false)
}

/// Encode a 10/12-bit planar frame (native `u16` samples, `Y` then `U`
/// then `V`) into a **lossy** high-bit-depth VP9 keyframe at quantizer
/// index `base_q_idx` (`1..=255`; use [`encode_vp9_lossless_hbd`] for
/// lossless).
///
/// `subsample == true` selects 4:2:0 (profile 2, chroma planes
/// `ceil(w/2) × ceil(h/2)`); `false` selects 4:4:4 (profile 3, chroma
/// at full resolution) — mirroring [`encode_vp9_lossless_hbd`]'s
/// layout. The decoded output (compare [`Vp9DecodedFrame`]'s native
/// `u16` planes) equals the encoder's in-loop reconstruction
/// bit-for-bit, elected §8.8 loop filter included. Returns
/// [`Error::Unsupported`] when `bit_depth` is not 10 or 12, any sample
/// exceeds the bit-depth range, `base_q_idx == 0`, the buffer is too
/// short, or the dimensions are degenerate.
pub fn encode_vp9_lossy_hbd(
    samples: &[u16],
    width: u32,
    height: u32,
    bit_depth: u8,
    subsample: bool,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_u16(
        samples, width, height, bit_depth, base_q_idx, subsample, subsample,
    )
}

/// Encode a 10/12-bit **4:2:2** planar frame (native `u16` samples,
/// chroma planes `ceil(w/2) × height`) into a **lossy** profile-3 VP9
/// keyframe at quantizer index `base_q_idx` (`1..=255`).
///
/// The `subsampling_x = 1, subsampling_y = 0` high-bit-depth geometry
/// [`encode_vp9_lossy_hbd`]'s two-way `subsample` flag cannot express.
/// Same guarantees and error cases as [`encode_vp9_lossy_hbd`].
pub fn encode_vp9_lossy_hbd_422(
    samples: &[u16],
    width: u32,
    height: u32,
    bit_depth: u8,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_u16(
        samples, width, height, bit_depth, base_q_idx, true, false,
    )
}

/// Encode an 8-bit **4:4:0** planar frame (`Y` then `U` then `V`,
/// chroma planes `width × ceil(h/2)`) into a **lossy** profile-1 VP9
/// keyframe at quantizer index `base_q_idx` (`1..=255`).
///
/// 4:4:0 is the `subsampling_x = 0, subsampling_y = 1` §7.2.2 geometry
/// — the vertical-only mirror of [`encode_vp9_lossy_422`] (profile 1
/// codes both subsampling flags in its §6.2.2 `color_config( )`); the
/// §8.5.2.1 chroma MV derivation rounds only the row component
/// (`round_mv_comp_q2` on axis 0). Same decoder-mirror guarantee and
/// error cases as [`encode_vp9_lossy_444`].
pub fn encode_vp9_lossy_440(
    pixels: &[u8],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_u8(pixels, width, height, base_q_idx, false, true)
}

/// Encode a 10/12-bit **4:4:0** planar frame (native `u16` samples,
/// chroma planes `width × ceil(h/2)`) into a **lossy** profile-3 VP9
/// keyframe at quantizer index `base_q_idx` (`1..=255`).
///
/// The `subsampling_x = 0, subsampling_y = 1` high-bit-depth geometry
/// [`encode_vp9_lossy_hbd`]'s two-way `subsample` flag cannot express
/// (the vertical mirror of [`encode_vp9_lossy_hbd_422`]). Same
/// guarantees and error cases as [`encode_vp9_lossy_hbd`].
pub fn encode_vp9_lossy_hbd_440(
    samples: &[u16],
    width: u32,
    height: u32,
    bit_depth: u8,
    base_q_idx: u8,
) -> Result<Vec<u8>, Error> {
    pixel_encoder::encode_keyframe_lossy_u16(
        samples, width, height, bit_depth, base_q_idx, false, true,
    )
}

/// Encode a **sequence** of 8-bit 4:2:0 planar frames into a lossy VP9
/// stream at quantizer index `base_q_idx` (`1..=255`): a lossy keyframe
/// followed by lossy inter (P-)frames with per-block `ZEROMV` / `NEWMV`
/// integer motion search, each quantizing the forward-DCT residual
/// against the previous frame's in-loop **reconstruction** (the
/// decoder's exact output, replayed by the encoder), so encoder and
/// decoder never drift across the chain.
///
/// **Chain framing (default since round 445):** every P-frame is shown
/// and non-error-resilient, so the §7.2.6 `UsePrevFrameMvs` derivation
/// is 1 on the decode side — the encoder models it by threading each
/// frame's §6.4.4 motion field into the next frame's §6.5 predictor
/// derivations and block writer. Consequences over the classic
/// error-resilient framing
/// ([`encode_vp9_lossy_sequence_error_resilient`]):
///
/// * temporally persistent motion that the spatial neighbours
///   mispredict reaches the §6.5.10 previous-frame candidate and codes
///   `NEARESTMV` / `NEARMV` instead of paying §6.4.20 mv-diff bits;
/// * a non-error-resilient frame keeps its **coded** sign biases (§7.2
///   `setup_past_independence` only zeroes them on error-resilient
///   frames), so the `[ LAST, ALTREF ]` compound election is live
///   inside the ordinary shown GOP — cross-fade content codes compound
///   averages with no hidden-predecessor construction;
/// * the round-441 elections run: the keyframe's §6.4.2 **skip
///   election** (all-zero-residual leaves code `skip = 1` — identical
///   reconstruction, strictly fewer bytes) and the §6.2.8 **loop-filter
///   delta election** on every P-frame, threading the §7.2.8 persistent
///   delta baseline exactly as the decoder folds it.
///
/// **Byte stability:** the coded bytes of this entry changed in round
/// 445 when the chain framing became the default; the pre-445 bytes
/// remain available, frozen, through the explicit
/// [`encode_vp9_lossy_sequence_error_resilient`] opt-out (the staged
/// `lossy-filtered-gop` corpus fixture pins that path's exact output).
/// Within one crate version every entry point remains
/// byte-deterministic.
///
/// The decoded output of every frame equals the encoder's in-loop
/// reconstruction bit-for-bit; distortion against the source is bounded
/// by the quantizer step. Returns [`Error::Unsupported`] for an empty
/// sequence, `base_q_idx == 0` (use [`encode_vp9_lossless_sequence`]),
/// degenerate dimensions, or any too-short frame buffer.
pub fn encode_vp9_lossy_sequence(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_chained_420(frames, width, height, base_q_idx)
}

/// [`encode_vp9_lossy_sequence`] on the **classic error-resilient
/// framing** — the explicit opt-out from the round-445
/// chained-as-default promotion: every P-frame codes §6.2
/// `error_resilient_mode = 1`, so §7.2.6 derives `UsePrevFrameMvs == 0`
/// and §7.2 `setup_past_independence( )` zeroes the effective sign
/// biases (no compound prediction) and resets the entropy state per
/// frame — any single P-frame stays decodable after a reference loss.
/// The round-441 keyframe skip and §6.2.8 delta elections do NOT run
/// here: this path's coded bytes are **frozen** as the exact pre-445
/// output of `encode_vp9_lossy_sequence`, pinned byte-for-byte by the
/// staged `lossy-filtered-gop` corpus fixture.
///
/// The decoder-mirror guarantee is identical to the default entry: the
/// decoded output equals the encoder's in-loop reconstruction
/// bit-for-bit.
pub fn encode_vp9_lossy_sequence_error_resilient(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_420(frames, width, height, base_q_idx)
}

/// Historical alias of [`encode_vp9_lossy_sequence`]: since round 445
/// the §7.2.6 chain framing IS the default lossy sequence path, so the
/// two entries are the same encoder (kept so pre-445 callers of the
/// explicit `_chained` name keep their exact behavior and bytes).
pub fn encode_vp9_lossy_sequence_chained(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    encode_vp9_lossy_sequence(frames, width, height, base_q_idx)
}

/// Encode a **sequence** of 8-bit **4:4:4** planar frames (each plane
/// `width × height`) into a lossy profile-1 VP9 stream at quantizer
/// index `base_q_idx` (`1..=255`).
///
/// Runs on the §7.2.6 **chain framing** (shown non-error-resilient
/// P-frames — the [`encode_vp9_lossy_sequence_chained`] model), with
/// the same motion search / multi-reference / compound election /
/// per-frame §8.8 filter election pipeline as the 4:2:0 encoders and
/// the identical decoder-mirror guarantee: every decoded frame equals
/// the encoder's in-loop reconstruction bit-for-bit.
pub fn encode_vp9_lossy_sequence_444(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_u8(frames, width, height, base_q_idx, false, false, true)
}

/// [`encode_vp9_lossy_sequence_444`] at **4:2:2** (chroma planes
/// `ceil(w/2) × height`, the `subsampling_x = 1, subsampling_y = 0`
/// profile-1 geometry). Chain framing; same guarantees.
pub fn encode_vp9_lossy_sequence_422(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_u8(frames, width, height, base_q_idx, true, false, true)
}

/// [`encode_vp9_lossy_sequence_444`] at **4:4:0** (chroma planes
/// `width × ceil(h/2)`, the `subsampling_x = 0, subsampling_y = 1`
/// profile-1 geometry — decode-side, the §8.5.2.1 chroma MV derivation
/// rounds only the row component). Chain framing; same guarantees.
pub fn encode_vp9_lossy_sequence_440(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_u8(frames, width, height, base_q_idx, false, true, true)
}

/// Encode a **sequence** of 10/12-bit planar frames (native `u16`
/// samples) into a lossy high-bit-depth VP9 stream at quantizer index
/// `base_q_idx` (`1..=255`) — profile 2 when `subsample == true`
/// (4:2:0), profile 3 when `false` (4:4:4), mirroring
/// [`encode_vp9_lossless_hbd`]'s layout.
///
/// Chain framing ([`encode_vp9_lossy_sequence_chained`] model) with
/// the full lossy pipeline — motion search, LAST/GOLDEN + compound
/// election, per-frame §8.8 filter election — and the decoder-mirror
/// guarantee against [`Vp9DecodedFrame`]'s native `u16` planes.
/// Returns [`Error::Unsupported`] when `bit_depth` is not 10 or 12,
/// any sample exceeds the bit-depth range, `base_q_idx == 0`, the
/// sequence is empty, or any buffer is too short.
pub fn encode_vp9_lossy_sequence_hbd(
    frames: &[&[u16]],
    width: u32,
    height: u32,
    bit_depth: u8,
    subsample: bool,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_u16(
        frames, width, height, bit_depth, base_q_idx, subsample, subsample, true,
    )
}

/// [`encode_vp9_lossy_sequence_hbd`] at **4:2:2** (profile 3, chroma
/// planes `ceil(w/2) × height`) — the high-bit-depth geometry the
/// two-way `subsample` flag cannot express. Chain framing; same
/// guarantees.
pub fn encode_vp9_lossy_sequence_hbd_422(
    frames: &[&[u16]],
    width: u32,
    height: u32,
    bit_depth: u8,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_u16(
        frames, width, height, bit_depth, base_q_idx, true, false, true,
    )
}

/// [`encode_vp9_lossy_sequence_hbd`] at **4:4:0** (profile 3, chroma
/// planes `width × ceil(h/2)` — the vertical mirror of
/// [`encode_vp9_lossy_sequence_hbd_422`]). Chain framing; same
/// guarantees.
pub fn encode_vp9_lossy_sequence_hbd_440(
    frames: &[&[u16]],
    width: u32,
    height: u32,
    bit_depth: u8,
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_u16(
        frames, width, height, bit_depth, base_q_idx, false, true, true,
    )
}

/// **Rate-controlled** lossy sequence encode: like
/// [`encode_vp9_lossy_sequence`] but instead of a fixed quantizer, every
/// frame is coded at the *lowest* `base_q_idx` (best quality) whose
/// coded size fits `target_bytes_per_frame` — a per-frame binary search
/// over the quantizer range (at most 8 trial encodes per frame; every
/// encoder in the chain is byte-deterministic, so the search is exact).
///
/// Rides the **§7.2.6 chain framing** like the default sequence entry
/// (round 445): shown non-error-resilient P-frames with prev-frame-MV
/// modeling and live compound election, the round-441 keyframe skip
/// election (a strict rate win, so the fitted quantizer can only
/// improve), and the §6.2.8 loop-filter delta election **under the
/// byte budget** — a moved delta slot costs coded §6.2.8 update bits,
/// so whenever the update would overflow `target_bytes_per_frame` the
/// frame falls back to the update-free encode (identical length to the
/// fitted trial; the §7.2.8 persistent baseline stays unmoved on both
/// sides) and the budget guarantee is preserved.
///
/// When even the coarsest quantizer (`base_q_idx == 255`) overflows the
/// budget, that frame is returned **best-effort** at `q == 255` rather
/// than failing: a budget below the frame's syntax floor is not
/// representable, and the stream stays decodable. The decoder-mirror
/// guarantee is unchanged — each frame's decoded output equals the
/// encoder's in-loop reconstruction bit-for-bit at whatever quantizer
/// it landed on, and P-frames reference the *chosen* previous
/// reconstruction.
///
/// Returns [`Error::Unsupported`] for an empty sequence, degenerate
/// dimensions, or any too-short frame buffer.
pub fn encode_vp9_lossy_sequence_rc(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    target_bytes_per_frame: usize,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_rc_420(frames, width, height, target_bytes_per_frame)
}

/// Encode an 8-bit 4:2:0 sequence as a lossy **alt-ref pyramid** GOP
/// (round 452): a keyframe, then groups of `altref_interval` display
/// frames each coded as a **hidden alt-ref** (`show_frame = 0`, the
/// group's last source frame, refreshed into a free §8.10 slot) followed
/// by the group's earlier frames as shown P-frames over the three-slot
/// reference set `ref_frame_idx = [ LAST, GOLDEN, ALTREF ]` with
/// `ref_frame_sign_bias = [ 0, 0, 1 ]` — every leaf elects among single
/// `LAST` / `GOLDEN` / `ALTREF` prediction and the `[ LAST, ALTREF ]`
/// §6.3.18 compound pair (`reference_select`) — and closed by a
/// `show_existing_frame` packet that displays the alt-ref at its
/// display position without re-coding it. The alt-ref slot then
/// becomes `LAST` and the freed slot hosts the next group's alt-ref;
/// the keyframe stays parked as the long-term `GOLDEN`.
///
/// The returned packets are in **decode order** — one entry per §6.1
/// frame, the hidden alt-refs and the one-byte `show_existing_frame`
/// packets included — so `out.len() > frames.len()` whenever a group
/// forms; [`decode_vp9_sequence`] returns exactly `frames.len()` shown
/// frames in display order. Every shown frame equals the encoder's
/// in-loop reconstruction bit-for-bit (the hidden alt-ref's
/// reconstruction is what the `show_existing_frame` packet displays).
///
/// The §7.2.6 `UsePrevFrameMvs` model follows the spec's
/// `compute_image_size( )` bookkeeping exactly: the frame decoded right
/// after a hidden alt-ref sees `show_frame == 0` at the previous
/// invocation and codes with the prev field absent; a
/// `show_existing_frame` packet invokes neither `compute_image_size( )`
/// nor the §6.1.2 `PrevMvs` save, so the following frame still models
/// the last *decoded* frame's motion field.
///
/// `altref_interval == 1` degenerates to plain shown P-frames (no
/// hidden frames; byte-identical GOP structure to a two-slot chain but
/// NOT byte-identical to [`encode_vp9_lossy_sequence`], which threads
/// its own slot layout). Returns [`Error::Unsupported`] for an empty
/// sequence, `base_q_idx == 0`, `altref_interval == 0`, degenerate
/// dimensions, or any too-short frame buffer.
pub fn encode_vp9_lossy_sequence_altref(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    base_q_idx: u8,
    altref_interval: u32,
) -> Result<Vec<Vec<u8>>, Error> {
    let mut cfg = Vp9GopConfig::new(base_q_idx);
    cfg.altref_interval = altref_interval;
    encode_vp9_lossy_sequence_with(frames, width, height, &cfg)
}

/// §6.2.11 segmentation feature emission selectable through
/// [`Vp9GopConfig::segmentation`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Vp9Segmentation {
    /// Single-segment frames (`segmentation_enabled = 0`).
    #[default]
    Off,
    /// **Adaptive quantization**: four activity-class segments (luma
    /// mean-absolute-deviation buckets per leaf) carrying
    /// `SEG_LVL_ALT_Q` deltas `[-16, -6, +4, +12]` around `base_q_idx`
    /// (flat content quantizes finer, busy content coarser) plus a
    /// per-frame elected `SEG_LVL_ALT_L` per class.
    AdaptiveQuant,
    /// **Static-content skip**: leaves whose co-located `LAST`
    /// prediction already matches the source code on a segment carrying
    /// `SEG_LVL_SKIP` + `SEG_LVL_REF_FRAME = LAST_FRAME`, so §6.4.8 /
    /// §6.4.13 / §6.4.16 / §6.4.17 read no skip / is_inter / ref_frame /
    /// inter_mode syntax and no residual for them.
    StaticSkip,
    /// Both of the above (all four `SEG_LVL_*` features).
    Full,
}

/// GOP-structure configuration of [`encode_vp9_lossy_sequence_with`]
/// (round 452). Construct with [`Vp9GopConfig::new`] and set the public
/// fields; the struct is `#[non_exhaustive]` so later rounds can add
/// axes without a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Vp9GopConfig {
    /// §7.2.9 `base_q_idx` for every frame (`1..=255`).
    pub base_q_idx: u8,
    /// Display frames per alt-ref group (`1` = no hidden alt-refs; see
    /// [`encode_vp9_lossy_sequence_altref`]). `0` is rejected.
    pub altref_interval: u32,
    /// §6.2.11 feature emission.
    pub segmentation: Vp9Segmentation,
    /// §6.2.13 `tile_cols_log2`: the frame is coded as `1 <<
    /// tile_cols_log2` independently decodable tile columns (validated
    /// against the §6.2.14 min/max derivation — two columns need a
    /// frame at least 8 SB64s ≈ 512 px wide), feeding the decoder's
    /// §9.2.4 tile-parallel path.
    pub tile_cols_log2: u8,
    /// §6.2.13 `tile_rows_log2` (`0..=2`).
    pub tile_rows_log2: u8,
    /// Code each alt-ref group's hidden frame as a §6.2 **intra-only**
    /// frame instead of a P-frame: a mid-GOP refresh point (the
    /// prediction chain is cut without a keyframe in the display
    /// stream; the decoder's §7.2 `setup_past_independence( )` runs on
    /// it and the frame surfaces later through its
    /// `show_existing_frame` packet). Requires `altref_interval >= 2`
    /// to have any effect.
    pub intra_only_altref: bool,
}

impl Vp9GopConfig {
    /// A plain two-slot-equivalent chain at `base_q_idx`: no alt-refs,
    /// no segmentation.
    pub fn new(base_q_idx: u8) -> Self {
        Self {
            base_q_idx,
            altref_interval: 1,
            segmentation: Vp9Segmentation::Off,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            intra_only_altref: false,
        }
    }
}

/// Encode an 8-bit 4:2:0 sequence as a lossy GOP under a
/// [`Vp9GopConfig`]: the alt-ref pyramid of
/// [`encode_vp9_lossy_sequence_altref`] and the §6.2.11 segmentation
/// features of [`Vp9Segmentation`], on the same three-slot,
/// chain-framed, filter-elected pipeline.
///
/// With segmentation on, every decoded frame codes
/// `segmentation_update_map = 1` with per-frame fitted
/// `segmentation_tree_probs`, `segmentation_temporal_update = 1` on
/// every frame that has a §6.4.14 `PrevSegmentIds` predictor (the
/// keyframe's §7.2 `setup_past_independence( )` clears it; a
/// `show_existing_frame` packet leaves it untouched per §8.1 step 3),
/// and `segmentation_update_data = 1` with the delta-mode feature table.
/// Note that §6.2.9 derives `Lossless` from the frame's `base_q_idx`
/// alone, so a per-segment lossless mode does not exist in VP9 — a
/// `SEG_LVL_ALT_Q` segment reaching `qindex 0` still dequantizes
/// through §8.6.2 at `dc_q( 0 )` / `ac_q( 0 )`.
///
/// Packets are returned in decode order (see
/// [`encode_vp9_lossy_sequence_altref`] for the hidden-frame /
/// `show_existing_frame` packet layout); every shown frame equals the
/// encoder's in-loop reconstruction bit-for-bit. Returns
/// [`Error::Unsupported`] for an empty sequence, `base_q_idx == 0`,
/// `altref_interval == 0`, degenerate dimensions, or any too-short
/// frame buffer.
pub fn encode_vp9_lossy_sequence_with(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    cfg: &Vp9GopConfig,
) -> Result<Vec<Vec<u8>>, Error> {
    let fmt = pixel_encoder::LossyFormat::new(8, true, true)?;
    let structure = pixel_encoder::GopStructure {
        altref_interval: cfg.altref_interval as usize,
        segmentation: match cfg.segmentation {
            Vp9Segmentation::Off => pixel_encoder::SegMode::Off,
            Vp9Segmentation::AdaptiveQuant => pixel_encoder::SegMode::AdaptiveQuant,
            Vp9Segmentation::StaticSkip => pixel_encoder::SegMode::StaticSkip,
            Vp9Segmentation::Full => pixel_encoder::SegMode::Full,
        },
        tile_cols_log2: cfg.tile_cols_log2,
        tile_rows_log2: cfg.tile_rows_log2,
        intra_only_altref: cfg.intra_only_altref,
    };
    pixel_encoder::encode_sequence_lossy_structured_u8(
        frames,
        width,
        height,
        cfg.base_q_idx,
        fmt,
        structure,
    )
}

/// Encode an 8-bit 4:2:0 sequence whose frames **change coded size
/// mid-stream** (round 452): a keyframe at `sizes[ 0 ]`, then one shown
/// P-frame per later frame at its own §6.2.5 explicit size, predicted
/// from the previous frame's §8.10 store through the **§8.5.2.3 scaled
/// motion compensation** whenever consecutive sizes differ (`xScale =
/// (RefFrameWidth << 14) / FrameWidth`; equal sizes reduce to the
/// unscaled sampler bit-for-bit). Each `frames[ i ]` is a planar
/// 4:2:0 buffer at `sizes[ i ]`.
///
/// Per leaf the encoder elects ZEROMV vs. a log-diamond NEWMV descent,
/// every candidate scored by the decoder's own §8.5.2.3 prediction, so
/// the reconstruction mirror is exact across the size change; §7.2.6
/// derives `UsePrevFrameMvs = 0` across a resize (condition (b)) and
/// the encoder models exactly that. The §5 scaling bounds apply
/// between consecutive sizes: at most 2x downscale and 16x upscale per
/// axis — outside them (or for an empty sequence, mismatched
/// `frames` / `sizes` lengths, `base_q_idx == 0`, degenerate
/// dimensions, or a short buffer) returns [`Error::Unsupported`].
pub fn encode_vp9_lossy_sequence_resized(
    frames: &[&[u8]],
    sizes: &[(u32, u32)],
    base_q_idx: u8,
) -> Result<Vec<Vec<u8>>, Error> {
    pixel_encoder::encode_sequence_lossy_resized_u8(frames, sizes, base_q_idx)
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

/// Codec registration: install the [`Vp9Decoder`] / [`Vp9Encoder`]
/// factories, capability description, §7.2 format-matrix pixel
/// formats, and container tag claims into the runtime context
/// ([`make_decoder`] / [`make_encoder`] are the matching direct
/// factories).
pub fn register(ctx: &mut RuntimeContext) {
    codec::register(ctx);
}

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

    // ----- §8.5.2.3 scaled-reference sequence (round 409) -----

    /// Build the round-409 **scaled-reference** stream: a 128x128
    /// lossless keyframe followed by inter frames whose coded size
    /// differs from their live reference's, forcing the §8.5.2.3 scaled
    /// motion-compensation sampler (`stepX` / `stepY != 16`) on every
    /// inter block:
    ///
    /// * F1 — 64x64 all-skip `ZEROMV` over the 128x128 keyframe in slot
    ///   0: both dimensions at the `2 * FrameWidth >= RefFrameWidth`
    ///   conformance **extreme** (ref = exactly 2x), `xScale = yScale =
    ///   32768`, `stepX = stepY = 32`. Refreshes slot 1.
    /// * F2 — 128x128 over slot 1 (F1's 64x64 reconstruction): 1/2x
    ///   reference (upscale), `xScale = 8192`, `stepX = 8` — half-pel
    ///   interpolation phases from pure scaling.
    /// * F3 — 96x96 over slot 0 (128x128): the fractional 4/3 ratio,
    ///   `xScale = (128 << 14) / 96 = 21845` — non-power-of-two
    ///   `fracX` / `fracY` arithmetic (§8.5.2.3 lines 4667-4668).
    /// * F4 — 64x64 **error-resilient** `NEWMV [8, 16]` (eighth-pel;
    ///   1 px down, 2 px right in current-frame units) over slot 0
    ///   (2x ref): the §8.5.2.1-§8.5.2.3 motion-vector scaling path
    ///   with a non-zero vector on a scaled reference.
    ///
    /// Every inter block is skip (no residual), so each frame's
    /// reconstruction IS the scaled §8.5.2 prediction — the sampler's
    /// output is observable directly.
    fn build_scaled_reference_stream() -> Vec<Vec<u8>> {
        use crate::compressed::TxMode;
        use crate::frame_writer::{
            assemble_inter_frame_all_skip_zeromv, assemble_inter_frame_planned,
            inter_pframe_header, FrameCoefSource, InterBlockPlanner,
        };

        // Keyframe: a deterministic high-detail pattern (diagonal
        // gradients + per-plane offsets) so scaling artifacts are
        // position-sensitive.
        let (w, h) = (128u32, 128u32);
        let (cw, ch) = (64usize, 64usize);
        let mut pixels = Vec::with_capacity((w * h) as usize + 2 * cw * ch);
        for y in 0..h as usize {
            for x in 0..w as usize {
                pixels.push(((x * 2 + y * 3) ^ (x >> 2)) as u8);
            }
        }
        for plane in 0..2usize {
            for y in 0..ch {
                for x in 0..cw {
                    pixels.push(((x * 5 + y * 7 + plane * 64) % 251) as u8);
                }
            }
        }
        let kf = encode_vp9(&pixels, w, h).expect("keyframe");

        // F1: 64x64 all-skip ZEROMV over slot 0 (128x128 keyframe).
        // inter_pframe_header refs slot 0 and refreshes slot 1.
        let hdr1 = inter_pframe_header(64, 64);
        let f1 = assemble_inter_frame_all_skip_zeromv(&hdr1).expect("f1");

        // F2: 128x128 over slot 1 (the 64x64 F1 recon) — upscale.
        let mut hdr2 = inter_pframe_header(128, 128);
        hdr2.ref_frame_idx = Some([1, 1, 1]);
        hdr2.refresh_frame_flags = 0x04;
        let f2 = assemble_inter_frame_all_skip_zeromv(&hdr2).expect("f2");

        // F3: 96x96 over slot 0 (128x128) — fractional 4/3 downscale.
        let mut hdr3 = inter_pframe_header(96, 96);
        hdr3.refresh_frame_flags = 0x08;
        let f3 = assemble_inter_frame_all_skip_zeromv(&hdr3).expect("f3");

        // F4: 64x64 error-resilient NEWMV [8, 16] over slot 0 (2x ref).
        // Error-resilient framing pins §7.2.6 UsePrevFrameMvs == 0 on
        // both sides (the writer holds no previous-frame motion field);
        // the MV difference stays even-magnitude for the no-hp §6.4.20
        // decomposition.
        let mut hdr4 = inter_pframe_header(64, 64);
        hdr4.error_resilient_mode = true;
        hdr4.refresh_frame_context = false;
        hdr4.frame_parallel_decoding_mode = true;
        hdr4.refresh_frame_flags = 0x10;
        let mut planner: Box<InterBlockPlanner<'_>> =
            Box::new(|_r, _c, _state| (crate::mode_info::NEWMV, [8, 16], true));
        let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(|_r, _c, _p, _x, _y, _b| Vec::new());
        let f4 =
            assemble_inter_frame_planned(&hdr4, TxMode::Only4x4, true, &mut *planner, &mut *coeffs)
                .expect("f4");

        vec![kf, f1, f2, f3, f4]
    }

    /// The scaled-reference stream decodes end-to-end, every frame at
    /// its declared size, and the exact-2x downscale frame satisfies the
    /// §8.5.2.3 **phase-0 identity**: with `xScale = 32768` and a zero
    /// MV, every sample position lands on an integer reference sample
    /// (`fracX = fracY = 0`), where the 8-tap `subpel_filters` row is
    /// the identity tap — so F1 must equal the keyframe reconstruction
    /// decimated 2:1 in both dimensions. This is an independent
    /// closed-form derivation of the sampler's output, not a replay of
    /// the implementation.
    #[test]
    fn scaled_reference_sequence_decodes_with_2x_phase0_identity() {
        let frames = build_scaled_reference_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("scaled-ref sequence");
        assert_eq!(out.len(), 5);
        let dims: Vec<(u32, u32)> = out.iter().map(|f| (f.width, f.height)).collect();
        assert_eq!(
            dims,
            vec![(128, 128), (64, 64), (128, 128), (96, 96), (64, 64)]
        );

        // Phase-0 identity at exact 2x (F1 vs the keyframe recon).
        let kf = &out[0];
        let f1 = &out[1];
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    f1.y[y * 64 + x],
                    kf.y[(2 * y) * 128 + 2 * x],
                    "luma ({x},{y}): 2x scaled ZEROMV must decimate the reference"
                );
            }
        }
        for y in 0..32usize {
            for x in 0..32usize {
                assert_eq!(f1.u[y * 32 + x], kf.u[(2 * y) * 64 + 2 * x], "U ({x},{y})");
                assert_eq!(f1.v[y * 32 + x], kf.v[(2 * y) * 64 + 2 * x], "V ({x},{y})");
            }
        }

        // F4: the scaled NEWMV [8, 16] (1 px down, 2 px right in
        // current-frame eighth-pel units) over the 2x reference —
        // §8.5.2.3 scales the vector into reference units (2 px down,
        // 4 px right), keeping every phase at 0: F4[y][x] must equal
        // K[2y + 2][2x + 4] where in range.
        let f4 = &out[4];
        for y in 0..48usize {
            for x in 0..48usize {
                assert_eq!(
                    f4.y[y * 64 + x],
                    kf.y[(2 * y + 2) * 128 + 2 * x + 4],
                    "luma ({x},{y}): scaled NEWMV phase-0 identity"
                );
            }
        }

        // Byte-determinism (fixture staging relies on it).
        let again = build_scaled_reference_stream();
        assert_eq!(frames, again);
    }

    /// Fixture-staging generator (round 409): when `OXIDEAV_VP9_STAGE_DIR`
    /// is set, writes the scaled-reference stream as `input.ivf` into that
    /// directory, for staging under `docs/video/vp9/fixtures/
    /// scaled-reference/` alongside a black-box reference decode
    /// (`expected.yuv`). A no-op otherwise — the stream itself is
    /// byte-deterministic and pinned by
    /// [`scaled_reference_sequence_decodes_with_2x_phase0_identity`].
    #[test]
    fn stage_scaled_reference_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_scaled_reference_stream();
        // IVF: 32-byte file header + 12-byte per-frame headers. The
        // file-header dimensions carry the first frame's size; each VP9
        // frame self-describes its own coded size.
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes()); // version
        ivf.extend_from_slice(&32u16.to_le_bytes()); // header length
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&128u16.to_le_bytes()); // width
        ivf.extend_from_slice(&128u16.to_le_bytes()); // height
        ivf.extend_from_slice(&25u32.to_le_bytes()); // timebase denominator
        ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase numerator
        ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes()); // unused
        for (i, f) in frames.iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        let path = std::path::Path::new(&dir).join("input.ivf");
        std::fs::write(&path, &ivf).expect("write input.ivf");
    }

    // ----- §6.2 intra-only frames (round 412) -----

    /// The deterministic 64x64 4:2:0 test pattern for the intra-only
    /// stream (`sel` picks between two distinct patterns).
    fn intra_only_pattern(sel: u8) -> Vec<u8> {
        let (w, h, cw, ch) = (64usize, 64usize, 32usize, 32usize);
        let mut pixels = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h {
            for x in 0..w {
                pixels.push(match sel {
                    0 => ((x * 3 + y * 5) ^ (y >> 1)) as u8,
                    _ => ((x * 7) ^ (y * 2) ^ 0xA5) as u8,
                });
            }
        }
        for plane in 0..2usize {
            for y in 0..ch {
                for x in 0..cw {
                    pixels.push(match sel {
                        0 => ((x * 2 + y * 9 + plane * 80) % 253) as u8,
                        _ => ((x * 11 + y * 4 + plane * 40) % 249) as u8,
                    });
                }
            }
        }
        pixels
    }

    /// Build the round-412 **intra-only** stream (profile 0, 8-bit
    /// 4:2:0, 64x64):
    ///
    /// * F0 — shown lossless keyframe (pattern A), refreshes all slots.
    /// * F1 — **hidden intra-only frame** (pattern B): `show_frame = 0`,
    ///   `intra_only = 1`, `reset_frame_context = 3` (§6.2 resets all
    ///   four §6.1.2 banks to the §10.5 defaults, so the
    ///   default-probability writers stay bit-aligned with the
    ///   decoder), `refresh_frame_flags = 0x02` (slot 1 only).
    /// * F2 — `show_existing_frame` displaying slot 1 (the intra-only
    ///   reconstruction).
    /// * F3 — shown all-skip `ZEROMV` P-frame whose `LAST` reference is
    ///   slot 1: its reconstruction is a verbatim copy of the
    ///   intra-only frame (§8.5.2 prediction from an
    ///   intra-only-refreshed §8.10 slot).
    fn build_intra_only_stream() -> Vec<Vec<u8>> {
        use crate::frame_writer::{assemble_inter_frame_all_skip_zeromv, inter_pframe_header};
        use crate::header_writer::write_uncompressed_header;
        use crate::pixel_encoder::{
            encode_keyframe_lossless, lossless_keyframe_header, padded_plane_from_bytes,
        };

        // F0: shown keyframe, pattern A.
        let kf = encode_vp9(&intra_only_pattern(0), 64, 64).expect("keyframe");

        // F1: hidden intra-only frame, pattern B, lossless, slot 1.
        let mut hdr1 = lossless_keyframe_header(64, 64);
        hdr1.frame_type = FrameType::NonKeyFrame;
        hdr1.intra_only = true;
        hdr1.show_frame = false;
        hdr1.reset_frame_context = 3;
        hdr1.refresh_frame_context = false;
        hdr1.refresh_frame_flags = 0x02;
        let pat_b = intra_only_pattern(1);
        let y = padded_plane_from_bytes(&pat_b[..64 * 64], 64, 64, 64, 64);
        let u = padded_plane_from_bytes(&pat_b[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32);
        let v = padded_plane_from_bytes(&pat_b[64 * 64 + 32 * 32..], 32, 32, 32, 32);
        let f1 = encode_keyframe_lossless(&hdr1, &[y, u, v]).expect("intra-only frame");

        // F2: show_existing_frame → slot 1.
        let mut hdr2 = lossless_keyframe_header(64, 64);
        hdr2.show_existing_frame = true;
        hdr2.frame_to_show_map_idx = Some(1);
        let f2 = write_uncompressed_header(&hdr2).expect("show-existing packet");

        // F3: shown all-skip ZEROMV P-frame over slot 1.
        let mut hdr3 = inter_pframe_header(64, 64);
        hdr3.ref_frame_idx = Some([1, 1, 1]);
        hdr3.refresh_frame_flags = 0x04;
        let f3 = assemble_inter_frame_all_skip_zeromv(&hdr3).expect("p-frame");

        vec![kf, f1, f2, f3]
    }

    /// The intra-only stream decodes end-to-end: the hidden intra-only
    /// frame never appears in the output, the `show_existing_frame`
    /// packet re-displays its slot, and the P-frame referencing that
    /// slot reconstructs a verbatim copy — all byte-exact against the
    /// lossless targets (an independent oracle: the encoder's input
    /// pattern, not its reconstruction).
    #[test]
    fn intra_only_sequence_decodes_byte_exact_against_targets() {
        let frames = build_intra_only_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("intra-only sequence");
        // 4 coded packets, 3 shown frames (F1 is hidden).
        assert_eq!(out.len(), 3);

        let pat_a = intra_only_pattern(0);
        let pat_b = intra_only_pattern(1);
        assert_eq!(out[0].to_planar_bytes(), pat_a, "keyframe != pattern A");
        assert_eq!(
            out[1].to_planar_bytes(),
            pat_b,
            "show-existing of the intra-only slot != pattern B"
        );
        assert_eq!(
            out[2].to_planar_bytes(),
            pat_b,
            "ZEROMV P-frame over the intra-only slot != pattern B"
        );

        // The F1 header really is a hidden intra-only frame with the
        // bank-resetting reset_frame_context = 3.
        let hdr = crate::header::parse_uncompressed_header(&frames[1]).expect("F1 header");
        assert_eq!(hdr.frame_type, FrameType::NonKeyFrame);
        assert!(hdr.intra_only && !hdr.show_frame);
        assert_eq!(hdr.reset_frame_context, 3);
        assert_eq!(hdr.refresh_frame_flags, 0x02);

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_intra_only_stream());
    }

    /// Fixture-staging generator (round 412): when `OXIDEAV_VP9_STAGE_DIR`
    /// is set, writes the intra-only stream as `intra-only/input.ivf`
    /// under that directory, for staging under
    /// `docs/video/vp9/fixtures/intra-only/` alongside a black-box
    /// reference decode (`expected.yuv`). A no-op otherwise.
    #[test]
    fn stage_intra_only_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_intra_only_stream();
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&64u16.to_le_bytes());
        ivf.extend_from_slice(&64u16.to_le_bytes());
        ivf.extend_from_slice(&25u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        for (i, f) in frames.iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        let subdir = std::path::Path::new(&dir).join("intra-only");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    // ----- truly-odd luma dimensions (round 412) -----

    /// Build the round-412 **odd-dimensions** stream: 59x37 8-bit 4:2:0
    /// (chroma 30x19 — both luma dimensions odd, so the §8.10 output
    /// crops mid-MI on both axes and chroma rounds up per
    /// `(w + 1) >> 1`), a lossless keyframe + 3 lossless P-frames of
    /// diagonally-translating deterministic content (real `NEWMV`
    /// motion at the odd frame edges). The common black-box encoder
    /// pipelines round input dimensions to even, so odd-luma streams
    /// are only mintable by the in-crate writers.
    fn build_odd_dims_stream() -> Vec<Vec<u8>> {
        let (w, h) = (59u32, 37u32);
        let (cw, ch) = (30usize, 19usize);
        let n = (w * h) as usize + 2 * cw * ch;
        let frame_at = |shift: usize| -> Vec<u8> {
            let mut pixels = Vec::with_capacity(n);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let sx = x + 2 * shift;
                    let sy = y + shift;
                    pixels.push(((sx * 5 + sy * 3) ^ (sy & 7)) as u8);
                }
            }
            for plane in 0..2usize {
                for y in 0..ch {
                    for x in 0..cw {
                        pixels.push(((x + shift) * 4 + (y + shift) * 6 + plane * 96) as u8);
                    }
                }
            }
            pixels
        };
        let content: Vec<Vec<u8>> = (0..4).map(frame_at).collect();
        let refs: Vec<&[u8]> = content.iter().map(|f| f.as_slice()).collect();
        // The staged fixture pins the classic error-resilient framing's
        // exact bytes (this WAS the default path when the fixture was
        // staged in round 412; round 445 promoted the chain framing to
        // the default, so the builder names the frozen opt-out).
        encode_vp9_lossless_sequence_error_resilient(&refs, w, h)
            .expect("odd-dims lossless sequence")
    }

    /// The odd-dimensions stream decodes byte-exact back to its source
    /// content (lossless), at the exact odd geometry, and is
    /// byte-deterministic.
    #[test]
    fn odd_dims_59x37_sequence_decodes_byte_exact() {
        let frames = build_odd_dims_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("odd-dims sequence");
        assert_eq!(out.len(), 4);
        for (k, f) in out.iter().enumerate() {
            assert_eq!((f.width, f.height), (59, 37), "frame {k} geometry");
            assert_eq!(f.u.len(), 30 * 19, "frame {k} chroma extent");
        }
        assert_eq!(frames, build_odd_dims_stream());
    }

    /// Fixture-staging generator (round 412): stages the odd-dimensions
    /// stream as `odd-dims-59x37/input.ivf` under `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_odd_dims_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_odd_dims_stream();
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&59u16.to_le_bytes());
        ivf.extend_from_slice(&37u16.to_le_bytes());
        ivf.extend_from_slice(&25u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        for (i, f) in frames.iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        let subdir = std::path::Path::new(&dir).join("odd-dims-59x37");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }
    // ----- per-segment feature streams (round 412) -----

    /// Build the round-412 **segment-features** stream (profile 0, 8-bit
    /// 4:2:0, 64x64):
    ///
    /// * F0 — shown lossless keyframe (pattern A), fills all slots.
    /// * F1 — hidden intra-only frame (pattern B) → slot 1.
    /// * F2 — shown all-skip `ZEROMV` P-frame, `LAST` = slot 0 (A) /
    ///   `GOLDEN` = slot 1 (B), with a live segmentation map (four
    ///   segments striped by superblock row) carrying the three
    ///   corpus-untested features: segment 1 = **`SEG_LVL_SKIP`**
    ///   (§6.4.8 forced skip, §6.4.16 forced ZEROMV — no skip /
    ///   inter_mode bits), segment 2 = **`SEG_LVL_REF_FRAME` = GOLDEN**
    ///   (§6.4.13/§6.4.17 derive is_inter + the reference pair — those
    ///   rows reconstruct from pattern B), segment 3 =
    ///   **`SEG_LVL_ALT_L` = +32** (§8.8.1 per-segment loop-filter
    ///   strength). The frame codes `loop_filter_level = 16` with
    ///   **`loop_filter_sharpness = 3`** (both corpus-firsts for a
    ///   self-encoded stream), so the §8.8 filter runs over the coded
    ///   block edges with per-segment strengths. Refreshes slot 2.
    /// * F3 — shown all-skip `ZEROMV` P-frame over slot 2 with
    ///   `loop_filter_level = 0` and segmentation disabled: a verbatim
    ///   copy of F2's (post-filter) reconstruction.
    fn build_seg_features_stream() -> Vec<Vec<u8>> {
        use crate::frame_writer::{
            assemble_inter_frame_all_skip_zeromv, assemble_inter_frame_tree, inter_pframe_header,
            FrameCoefSource, InterFrameTreePlan, InterTreeLeaf, InterTreePlanner,
        };
        use crate::header::SegmentationParams;
        use crate::mode_info::{
            GOLDEN_FRAME, LAST_FRAME, NONE_REF_FRAME, SEG_LVL_REF_FRAME, SEG_LVL_SKIP, ZEROMV,
        };
        use crate::pixel_encoder::{
            encode_keyframe_lossless, lossless_keyframe_header, padded_plane_from_bytes,
        };

        // F0: shown keyframe, pattern A.
        let kf = encode_vp9(&intra_only_pattern(0), 64, 64).expect("keyframe");

        // F1: hidden intra-only frame, pattern B, lossless, slot 1.
        let mut hdr1 = lossless_keyframe_header(64, 64);
        hdr1.frame_type = FrameType::NonKeyFrame;
        hdr1.intra_only = true;
        hdr1.show_frame = false;
        hdr1.reset_frame_context = 3;
        hdr1.refresh_frame_context = false;
        hdr1.refresh_frame_flags = 0x02;
        let pat_b = intra_only_pattern(1);
        let y = padded_plane_from_bytes(&pat_b[..64 * 64], 64, 64, 64, 64);
        let u = padded_plane_from_bytes(&pat_b[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32);
        let v = padded_plane_from_bytes(&pat_b[64 * 64 + 32 * 32..], 32, 32, 32, 32);
        let f1 = encode_keyframe_lossless(&hdr1, &[y, u, v]).expect("intra-only frame");

        // F2: the segment-features frame.
        let mut hdr2 = inter_pframe_header(64, 64);
        hdr2.ref_frame_idx = Some([0, 1, 1]);
        hdr2.refresh_frame_flags = 0x04;
        hdr2.loop_filter.level = 16;
        hdr2.loop_filter.sharpness = 3;
        let mut seg = SegmentationParams::default_disabled();
        seg.enabled = true;
        seg.update_map = true;
        seg.tree_probs = Some([128; 7]);
        seg.update_data = true;
        seg.abs_or_delta_update = false;
        seg.feature_enabled[1][SEG_LVL_SKIP] = true;
        seg.feature_enabled[2][SEG_LVL_REF_FRAME] = true;
        seg.feature_data[2][SEG_LVL_REF_FRAME] = GOLDEN_FRAME as i16;
        seg.feature_enabled[3][crate::loop_filter::SEG_LVL_ALT_L] = true;
        seg.feature_data[3][crate::loop_filter::SEG_LVL_ALT_L] = 32;
        hdr2.segmentation = seg;

        let plan = InterFrameTreePlan {
            tx_mode: crate::compressed::TxMode::Only4x4,
            reference_mode: crate::compressed::ReferenceMode::SingleReference,
            partitions: std::collections::HashMap::new(), // all-8x8 leaves
            prev_segment_ids: None,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        // Segments striped by MI row pairs: rows 0-1 seg 0 (plain skip),
        // rows 2-3 seg 1 (SKIP-forced), rows 4-5 seg 2 (GOLDEN
        // override → pattern B), rows 6-7 seg 3 (ALT_L).
        let mut planner: Box<InterTreePlanner<'_>> =
            Box::new(|lr: u32, _lc: u32, subsize: u8, _state| {
                let segment_id = (lr / 2).min(3) as u8;
                let ref0 = if segment_id == 2 {
                    GOLDEN_FRAME
                } else {
                    LAST_FRAME
                };
                InterTreeLeaf {
                    mi_size: subsize,
                    tx_size: 0, // inferred under Only4x4
                    y_mode: ZEROMV,
                    interp_filter: 0,
                    ref_frame: [ref0, NONE_REF_FRAME],
                    mv: [[0, 0], [0, 0]],
                    skip: true,
                    segment_id,
                    sub: None,
                }
            });
        let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(|_r, _c, _p, _x, _y, _b| Vec::new());
        let f2 = assemble_inter_frame_tree(&hdr2, &plan, &mut planner, &mut coeffs).expect("f2");

        // F3: verbatim copy of F2's post-filter reconstruction.
        let mut hdr3 = inter_pframe_header(64, 64);
        hdr3.ref_frame_idx = Some([2, 2, 2]);
        hdr3.refresh_frame_flags = 0x08;
        let f3 = assemble_inter_frame_all_skip_zeromv(&hdr3).expect("f3");

        vec![kf, f1, f2, f3]
    }

    /// The segment-features stream decodes end-to-end with the feature
    /// semantics observable in the output: the `SEG_LVL_REF_FRAME`
    /// stripe reconstructs from the GOLDEN (intra-only) pattern while
    /// the other stripes keep the keyframe pattern (checked on
    /// filter-untouched interior samples), and the `loop_filter_level=0`
    /// copy frame equals the (filtered) feature frame byte-for-byte —
    /// pinning that the §8.10 reference stores are post-§8.8 samples.
    #[test]
    fn seg_features_sequence_decodes_with_observable_feature_semantics() {
        let frames = build_seg_features_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("seg-features sequence");
        assert_eq!(out.len(), 3); // F1 is hidden.

        let pat_a = intra_only_pattern(0);
        let pat_b = intra_only_pattern(1);
        assert_eq!(out[0].to_planar_bytes(), pat_a, "keyframe != pattern A");

        // Interior luma samples ≥ 3 samples away from every 8-aligned
        // edge line are untouched by the §8.8 filter (the strongest
        // filter this stream admits is the 8-tap, modifying p2..q2):
        // sample (4, 12) sits in the seg-0 stripe (pattern A), (36, 12)
        // in the SEG_LVL_REF_FRAME stripe (GOLDEN → pattern B),
        // (52, 12) in the ALT_L stripe (still LAST → pattern A).
        let f2 = &out[1];
        assert_eq!(
            f2.y[4 * 64 + 12],
            u16::from(pat_a[4 * 64 + 12]),
            "seg-0 stripe reconstructs from LAST (pattern A)"
        );
        assert_eq!(
            f2.y[36 * 64 + 12],
            u16::from(pat_b[36 * 64 + 12]),
            "SEG_LVL_REF_FRAME stripe reconstructs from GOLDEN (pattern B)"
        );
        assert_eq!(
            f2.y[52 * 64 + 12],
            u16::from(pat_a[52 * 64 + 12]),
            "ALT_L stripe still reconstructs from LAST (pattern A)"
        );
        // The filter did run: the A/B stripe boundary at row 32 must
        // show filtered samples (the prediction is a hard edge there).
        let row31: Vec<u16> = (0..64).map(|x| f2.y[31 * 64 + x]).collect();
        let unfiltered31: Vec<u16> = (0..64).map(|x| u16::from(pat_a[31 * 64 + x])).collect();
        assert_ne!(
            row31, unfiltered31,
            "\u{a7}8.8 filtering visible at the stripe boundary"
        );

        // F3 (level-0, no residual) is a verbatim copy of F2's stored
        // (post-filter) reconstruction.
        assert_eq!(
            out[2].to_planar_bytes(),
            out[1].to_planar_bytes(),
            "copy frame != stored post-filter reconstruction"
        );

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_seg_features_stream());
    }

    /// Fixture-staging generator (round 412): stages the segment-features
    /// stream as `seg-features-skip-ref-altl/input.ivf` under
    /// `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_seg_features_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_seg_features_stream();
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&64u16.to_le_bytes());
        ivf.extend_from_slice(&64u16.to_le_bytes());
        ivf.extend_from_slice(&25u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        for (i, f) in frames.iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        let subdir = std::path::Path::new(&dir).join("seg-features-skip-ref-altl");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    // ----- sub-8x8 inter blocks (round 415) -----

    /// The `BLOCK_8X8` nodes the round-415 stream splits below 8x8, as
    /// `(MiRow, MiCol)` pairs (two SPLIT → 4x4, two HORZ → 8x4, two
    /// VERT → 4x8; see [`build_sub8x8_inter_stream`]).
    const SUB8X8_NODES: [(u32, u32); 6] = [(0, 0), (0, 7), (2, 6), (3, 3), (5, 5), (7, 0)];

    /// The F1 target of the sub-8x8 stream: pattern A everywhere except
    /// the six sub-8x8 MI cells, which carry pattern-B content (so their
    /// §8.7.2 WHT residual is live while every other block is an exact
    /// ZEROMV copy).
    fn sub8x8_composite_pattern() -> Vec<u8> {
        let mut px = intra_only_pattern(0);
        let pat_b = intra_only_pattern(1);
        for &(r, c) in SUB8X8_NODES.iter() {
            let (r, c) = (r as usize, c as usize);
            for y in r * 8..r * 8 + 8 {
                for x in c * 8..c * 8 + 8 {
                    px[y * 64 + x] = pat_b[y * 64 + x];
                }
            }
            for plane in 0..2usize {
                let off = 64 * 64 + plane * 32 * 32;
                for y in r * 4..r * 4 + 4 {
                    for x in c * 4..c * 4 + 4 {
                        px[off + y * 32 + x] = pat_b[off + y * 32 + x];
                    }
                }
            }
        }
        px
    }

    /// Build the round-415 **sub-8x8 inter** stream (profile 0, 8-bit
    /// 4:2:0, 64x64):
    ///
    /// * F0 — shown lossless keyframe (pattern A), fills all slots.
    /// * F1 — shown lossless P-frame whose §6.4.3 layout splits six
    ///   `BLOCK_8X8` nodes below 8x8 — two SPLIT (4x4 leaves), two HORZ
    ///   (8x4), two VERT (4x8) — carrying per-cell `NEWMV` /
    ///   `NEARESTMV` / `NEARMV` / `ZEROMV` modes with distinct motion
    ///   vectors (integer and quarter-pel) through the §6.4.16
    ///   per-`(idy, idx)` walk and its §6.5.14 `append_sub8x8_mvs( )`
    ///   predictor rewrites. The sub-8x8 regions reconstruct pattern-B
    ///   content via live §8.7.2 WHT inter residual at the 8x8 grid;
    ///   every other 8x8 block is an exact-copy `ZEROMV` skip.
    ///   Error-resilient (the §7.2.6 `UsePrevFrameMvs == 0` model NEWMV
    ///   requires), refreshes slot 0.
    /// * F2 — shown all-skip `ZEROMV` copy frame over slot 0 (pins the
    ///   §8.10 store of F1's reconstruction).
    ///
    /// No black-box encoder wrapper exposes per-sub-block motion
    /// planning, so this axis is only mintable by the in-crate writers.
    fn build_sub8x8_inter_stream() -> Vec<Vec<u8>> {
        use crate::frame_writer::{assemble_inter_frame_all_skip_zeromv, InterTreeLeaf};
        use crate::inter_block_writer::InterSubBlockSpec;
        use crate::mode_info::{LAST_FRAME, NEARESTMV, NEARMV, NEWMV, NONE_REF_FRAME, ZEROMV};
        use crate::partition::{PARTITION_HORZ, PARTITION_SPLIT, PARTITION_VERT};
        use crate::pixel_encoder::{
            encode_pframe_lossless_layout, lossless_pframe_header, padded_plane_from_bytes,
        };
        use crate::residual::BLOCK_8X8;

        // F0: shown keyframe, pattern A.
        let pat_a = intra_only_pattern(0);
        let kf = encode_vp9(&pat_a, 64, 64).expect("keyframe");

        // F1: the sub-8x8 P-frame toward the composite target.
        let composite = sub8x8_composite_pattern();
        let targets = [
            padded_plane_from_bytes(&composite[..64 * 64], 64, 64, 64, 64),
            padded_plane_from_bytes(&composite[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32),
            padded_plane_from_bytes(&composite[64 * 64 + 32 * 32..], 32, 32, 32, 32),
        ];
        let prev: [Vec<i32>; 3] = [
            pat_a[..64 * 64].iter().map(|&s| i32::from(s)).collect(),
            pat_a[64 * 64..64 * 64 + 32 * 32]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
            pat_a[64 * 64 + 32 * 32..]
                .iter()
                .map(|&s| i32::from(s))
                .collect(),
        ];
        let reference: [(&[i32], usize); 3] = [
            (prev[0].as_slice(), 64),
            (prev[1].as_slice(), 32),
            (prev[2].as_slice(), 32),
        ];

        let hdr1 = lossless_pframe_header(64, 64);
        let mut partitions = std::collections::HashMap::new();
        partitions.insert((0u32, 0u32, BLOCK_8X8), PARTITION_SPLIT);
        partitions.insert((0, 7, BLOCK_8X8), PARTITION_HORZ);
        partitions.insert((2, 6, BLOCK_8X8), PARTITION_VERT);
        partitions.insert((3, 3, BLOCK_8X8), PARTITION_VERT);
        partitions.insert((5, 5, BLOCK_8X8), PARTITION_HORZ);
        partitions.insert((7, 0, BLOCK_8X8), PARTITION_SPLIT);

        // Every MV component is even, so each difference stays §6.4.20
        // codeable under either §6.5.13 hp-gate outcome.
        let zero2 = [[0, 0], [0, 0]];
        let sv = |mv: [i32; 2]| -> [[i32; 2]; 2] { [mv, [0, 0]] };
        let sub_for = move |r: u32, c: u32| -> Option<InterSubBlockSpec> {
            match (r, c) {
                // 4x4: NEWMV a, NEARESTMV (block 1 seeds BlockMvs[0] =
                // a), NEWMV b, ZEROMV.
                (0, 0) => Some(InterSubBlockSpec {
                    modes: [NEWMV, NEARESTMV, NEWMV, ZEROMV],
                    mvs: [sv([16, 8]), sv([16, 8]), sv([-16, 24]), zero2],
                }),
                // 8x4 (cells {0, 2}): integer + quarter-pel NEWMV.
                (0, 7) => Some(InterSubBlockSpec {
                    modes: [NEWMV, ZEROMV, NEWMV, ZEROMV],
                    mvs: [sv([8, -8]), zero2, sv([4, 6]), zero2],
                }),
                // 4x8 (cells {0, 1}): ZEROMV + NEWMV.
                (2, 6) => Some(InterSubBlockSpec {
                    modes: [ZEROMV, NEWMV, ZEROMV, ZEROMV],
                    mvs: [zero2, sv([-8, -16]), zero2, zero2],
                }),
                // 4x8: NEWMV + NEARESTMV (block 1 seeds BlockMvs[0]).
                (3, 3) => Some(InterSubBlockSpec {
                    modes: [NEWMV, NEARESTMV, ZEROMV, ZEROMV],
                    mvs: [sv([24, 0]), sv([24, 0]), zero2, zero2],
                }),
                // 8x4: NEWMV + NEARESTMV (block 2 seeds BlockMvs[0]).
                (5, 5) => Some(InterSubBlockSpec {
                    modes: [NEWMV, ZEROMV, NEARESTMV, ZEROMV],
                    mvs: [sv([0, 32]), zero2, sv([0, 32]), zero2],
                }),
                // 4x4 with a NEARMV cell: block 3 seeds BlockMvs[2] = b2
                // then walks BlockMvs[1] = a2 (differs) => NearMv = a2.
                (7, 0) => Some(InterSubBlockSpec {
                    modes: [NEWMV, NEARESTMV, NEWMV, NEARMV],
                    mvs: [sv([8, 16]), sv([8, 16]), sv([-8, 8]), sv([8, 16])],
                }),
                _ => None,
            }
        };
        let mut leaf_plan = move |r: u32,
                                  c: u32,
                                  subsize: u8,
                                  _s: &crate::decode_block::Vp9FrameState|
              -> InterTreeLeaf {
            let sub = if subsize < BLOCK_8X8 {
                sub_for(r, c)
            } else {
                None
            };
            let skip = sub.is_none(); // plain 8x8 blocks are exact copies.
            InterTreeLeaf {
                mi_size: subsize,
                tx_size: 0, // inferred under Only4x4
                y_mode: ZEROMV,
                interp_filter: 0,
                ref_frame: [LAST_FRAME, NONE_REF_FRAME],
                mv: [[0, 0], [0, 0]],
                skip,
                segment_id: 0,
                sub,
            }
        };
        let f1 = encode_pframe_lossless_layout(
            &hdr1,
            &targets,
            &reference,
            None,
            64,
            64,
            partitions,
            &mut leaf_plan,
        )
        .expect("sub-8x8 p-frame");

        // F2: verbatim copy of F1's reconstruction (slot 0).
        let hdr2 = lossless_pframe_header(64, 64);
        let f2 = assemble_inter_frame_all_skip_zeromv(&hdr2).expect("copy p-frame");

        vec![kf, f1, f2]
    }

    /// The sub-8x8 stream decodes byte-exact against its lossless
    /// targets (pattern A, then the composite carrying pattern-B content
    /// exactly in the six sub-8x8 MI cells, then the copy frame), and is
    /// byte-deterministic.
    #[test]
    fn sub8x8_inter_sequence_decodes_byte_exact() {
        let frames = build_sub8x8_inter_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("sub-8x8 sequence");
        assert_eq!(out.len(), 3);

        let pat_a = intra_only_pattern(0);
        let composite = sub8x8_composite_pattern();
        assert_eq!(out[0].to_planar_bytes(), pat_a, "keyframe != pattern A");
        assert_eq!(
            out[1].to_planar_bytes(),
            composite,
            "sub-8x8 P-frame != composite target"
        );
        assert_eq!(
            out[2].to_planar_bytes(),
            composite,
            "copy frame != stored reconstruction"
        );

        // The composite really differs from pattern A inside every
        // sub-8x8 cell (the assertion above would otherwise pass with
        // dead sub-8x8 residual).
        for &(r, c) in SUB8X8_NODES.iter() {
            let (r, c) = (r as usize, c as usize);
            let differs = (0..8).any(|y| {
                (0..8).any(|x| {
                    composite[(r * 8 + y) * 64 + c * 8 + x] != pat_a[(r * 8 + y) * 64 + c * 8 + x]
                })
            });
            assert!(differs, "node ({r}, {c}) carries no live content");
        }

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_sub8x8_inter_stream());
    }

    /// Fixture-staging generator (round 415): stages the sub-8x8 stream
    /// as `sub8x8-inter-mvs/input.ivf` under `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_sub8x8_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_sub8x8_inter_stream();
        let ivf = ivf_wrap_64x64(&frames);
        let subdir = std::path::Path::new(&dir).join("sub8x8-inter-mvs");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    /// The staged corpus fixture is byte-identical to the builder's
    /// output — the fixture IS this crate's writer output (docs-gated).
    #[test]
    fn staged_sub8x8_fixture_matches_builder() {
        let path = std::path::Path::new("../../docs/video/vp9/fixtures/sub8x8-inter-mvs/input.ivf");
        if !path.is_file() {
            eprintln!("docs corpus not present; docs-gated");
            return;
        }
        let staged = std::fs::read(path).expect("staged input.ivf");
        assert_eq!(
            staged,
            ivf_wrap_64x64(&build_sub8x8_inter_stream()),
            "staged fixture bytes != builder output"
        );
    }

    // ----- render_and_frame_size_different (round 415) -----

    /// Build the round-415 **render-size** stream (profile 0, 8-bit
    /// 4:2:0, coded 64x64, render 128x72): every coded frame writes the
    /// §6.2.3 `render_and_frame_size_different = 1` arm — explicit
    /// `render_width_minus_1` / `render_height_minus_1` — across all
    /// three `render_size( )` call sites:
    ///
    /// * F0 — shown lossless keyframe (pattern A; the §6.2 key-frame
    ///   arm), fills all slots.
    /// * F1 — hidden lossless intra-only frame (pattern B; the §6.2
    ///   intra-only arm) → slot 1.
    /// * F2 — shown all-skip `ZEROMV` P-frame over slot 1 (the §6.2.5
    ///   `frame_size_with_refs( )` explicit-size arm) → slot 2.
    /// * F3 — shown all-skip `ZEROMV` copy P-frame over slot 2.
    ///
    /// The black-box encoder wrappers only signal display geometry
    /// through container-level aspect metadata (the bitstream field
    /// stays 0), so this header axis is only mintable by the in-crate
    /// writers.
    fn build_render_size_stream() -> Vec<Vec<u8>> {
        use crate::frame_writer::{assemble_inter_frame_all_skip_zeromv, inter_pframe_header};
        use crate::pixel_encoder::{
            encode_keyframe_lossless, lossless_keyframe_header, padded_plane_from_bytes,
        };

        const RENDER: (u32, u32) = (128, 72);

        // F0: shown keyframe, pattern A, render override.
        let mut hdr0 = lossless_keyframe_header(64, 64);
        hdr0.render_width = RENDER.0;
        hdr0.render_height = RENDER.1;
        let pat_a = intra_only_pattern(0);
        let planes_a = [
            padded_plane_from_bytes(&pat_a[..64 * 64], 64, 64, 64, 64),
            padded_plane_from_bytes(&pat_a[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32),
            padded_plane_from_bytes(&pat_a[64 * 64 + 32 * 32..], 32, 32, 32, 32),
        ];
        let f0 = encode_keyframe_lossless(&hdr0, &planes_a).expect("keyframe");

        // F1: hidden intra-only frame, pattern B, slot 1, render override.
        let mut hdr1 = lossless_keyframe_header(64, 64);
        hdr1.frame_type = FrameType::NonKeyFrame;
        hdr1.intra_only = true;
        hdr1.show_frame = false;
        hdr1.reset_frame_context = 3;
        hdr1.refresh_frame_context = false;
        hdr1.refresh_frame_flags = 0x02;
        hdr1.render_width = RENDER.0;
        hdr1.render_height = RENDER.1;
        let pat_b = intra_only_pattern(1);
        let planes_b = [
            padded_plane_from_bytes(&pat_b[..64 * 64], 64, 64, 64, 64),
            padded_plane_from_bytes(&pat_b[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32),
            padded_plane_from_bytes(&pat_b[64 * 64 + 32 * 32..], 32, 32, 32, 32),
        ];
        let f1 = encode_keyframe_lossless(&hdr1, &planes_b).expect("intra-only frame");

        // F2: shown all-skip ZEROMV P-frame over slot 1, render override.
        let mut hdr2 = inter_pframe_header(64, 64);
        hdr2.ref_frame_idx = Some([1, 1, 1]);
        hdr2.refresh_frame_flags = 0x04;
        hdr2.render_width = RENDER.0;
        hdr2.render_height = RENDER.1;
        let f2 = assemble_inter_frame_all_skip_zeromv(&hdr2).expect("p-frame");

        // F3: copy P-frame over slot 2, render override.
        let mut hdr3 = inter_pframe_header(64, 64);
        hdr3.ref_frame_idx = Some([2, 2, 2]);
        hdr3.refresh_frame_flags = 0x08;
        hdr3.render_width = RENDER.0;
        hdr3.render_height = RENDER.1;
        let f3 = assemble_inter_frame_all_skip_zeromv(&hdr3).expect("copy p-frame");

        vec![f0, f1, f2, f3]
    }

    /// The render-size stream decodes byte-exact against its lossless
    /// targets, every coded frame's parsed header carries the 128x72
    /// render override (the §6.2.3 different-size arm across the
    /// key-frame / intra-only / inter call sites), and the stream is
    /// byte-deterministic.
    #[test]
    fn render_size_sequence_decodes_byte_exact_with_render_override() {
        let frames = build_render_size_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("render-size sequence");
        assert_eq!(out.len(), 3); // F1 is hidden.

        let pat_a = intra_only_pattern(0);
        let pat_b = intra_only_pattern(1);
        assert_eq!(out[0].to_planar_bytes(), pat_a, "keyframe != pattern A");
        assert_eq!(out[1].to_planar_bytes(), pat_b, "P over intra-only slot");
        assert_eq!(out[2].to_planar_bytes(), pat_b, "copy frame");

        // Every coded frame parses back with the render override; the
        // inter headers need reference-size state for the §6.2.5
        // frame_size_with_refs( ) arm.
        let ref_dims = vec![(64u32, 64u32); 8];
        let cc0 = crate::header::parse_uncompressed_header(&frames[0])
            .expect("F0")
            .color_config;
        for (i, f) in frames.iter().enumerate() {
            let hdr = if i < 2 {
                crate::header::parse_uncompressed_header(f).expect("intra header")
            } else {
                crate::header::parse_uncompressed_header_with_refs(
                    f,
                    Some(crate::header::RefFrameState {
                        ref_dims: &ref_dims,
                        color_config: cc0,
                    }),
                )
                .expect("inter header")
            };
            assert_eq!((hdr.frame_width, hdr.frame_height), (64, 64), "frame {i}");
            assert_eq!(
                (hdr.render_width, hdr.render_height),
                (128, 72),
                "frame {i}: render_and_frame_size_different arm"
            );
        }

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_render_size_stream());
    }

    /// Fixture-staging generator (round 415): stages the render-size
    /// stream as `render-size-128x72/input.ivf` under
    /// `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_render_size_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_render_size_stream();
        let ivf = ivf_wrap_64x64(&frames);
        let subdir = std::path::Path::new(&dir).join("render-size-128x72");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    /// The staged corpus fixture is byte-identical to the builder's
    /// output — the fixture IS this crate's writer output (docs-gated).
    #[test]
    fn staged_render_size_fixture_matches_builder() {
        let path =
            std::path::Path::new("../../docs/video/vp9/fixtures/render-size-128x72/input.ivf");
        if !path.is_file() {
            eprintln!("docs corpus not present; docs-gated");
            return;
        }
        let staged = std::fs::read(path).expect("staged input.ivf");
        assert_eq!(
            staged,
            ivf_wrap_64x64(&build_render_size_stream()),
            "staged fixture bytes != builder output"
        );
    }

    // ----- §6.4.12 temporal-predicted segment ids (round 418) -----

    /// The round-418 temporal-seg-map segmentation parameters: a live
    /// two-segment map whose segment 1 carries `SEG_LVL_REF_FRAME =
    /// GOLDEN` (so the map is *visible* — segment-1 blocks reconstruct
    /// from the GOLDEN slot's pattern-B content), coded either
    /// non-temporally (the §6.4.7 tree) or temporally (the §6.4.12
    /// `seg_id_predicted` branch under `segmentation_pred_prob`).
    fn temporal_seg_params(temporal: bool) -> crate::header::SegmentationParams {
        use crate::mode_info::SEG_LVL_REF_FRAME;
        let mut seg = crate::header::SegmentationParams::default_disabled();
        seg.enabled = true;
        seg.update_map = true;
        seg.temporal_update = temporal;
        seg.tree_probs = Some([128; 7]);
        if temporal {
            seg.pred_prob = Some([180, 128, 220]);
        }
        seg.update_data = true;
        seg.abs_or_delta_update = false;
        seg.feature_enabled[1][SEG_LVL_REF_FRAME] = true;
        seg.feature_data[1][SEG_LVL_REF_FRAME] = crate::mode_info::GOLDEN_FRAME as i16;
        seg
    }

    /// The round-418 segment map: MI rows `t..t+3` are segment 1 (the
    /// GOLDEN-override band), everything else segment 0 (64x64 → an
    /// 8x8 MI grid).
    fn band_map(t: u32) -> Vec<u8> {
        let mut map = vec![0u8; 64];
        for r in t..(t + 3).min(8) {
            for c in 0..8u32 {
                map[(r * 8 + c) as usize] = 1;
            }
        }
        map
    }

    /// The visible reconstruction the band map produces: pattern-B
    /// content on the band's rows (luma rows `8t..8(t+3)`, chroma rows
    /// `4t..4(t+3)`), pattern A elsewhere.
    fn band_composite(t: usize) -> Vec<u8> {
        let mut px = intra_only_pattern(0);
        let pat_b = intra_only_pattern(1);
        for y in 8 * t..8 * (t + 3) {
            for x in 0..64usize {
                px[y * 64 + x] = pat_b[y * 64 + x];
            }
        }
        for plane in 0..2usize {
            let off = 64 * 64 + plane * 32 * 32;
            for y in 4 * t..4 * (t + 3) {
                for x in 0..32usize {
                    px[off + y * 32 + x] = pat_b[off + y * 32 + x];
                }
            }
        }
        px
    }

    /// Assemble one all-skip `ZEROMV` P-frame carrying the band map
    /// (per-MI-row segment ids from `map`; segment-1 blocks take the
    /// §6.4.13/§6.4.17 GOLDEN override, segment-0 blocks LAST), with
    /// the map coded temporally against `prev` when supplied.
    fn temporal_segmap_pframe(
        hdr: &crate::header::Vp9FrameHeader,
        map: &[u8],
        prev: Option<Vec<u8>>,
    ) -> Vec<u8> {
        use crate::frame_writer::{
            assemble_inter_frame_tree, FrameCoefSource, InterFrameTreePlan, InterTreeLeaf,
            InterTreePlanner,
        };
        use crate::mode_info::{GOLDEN_FRAME, LAST_FRAME, NONE_REF_FRAME, ZEROMV};

        let plan = InterFrameTreePlan {
            tx_mode: crate::compressed::TxMode::Only4x4,
            reference_mode: crate::compressed::ReferenceMode::SingleReference,
            partitions: std::collections::HashMap::new(), // all-8x8 leaves
            prev_segment_ids: prev,
            prev_frame_mvs_absent: false,
            prev_frame_mvs: None,
        };
        let mut planner: Box<InterTreePlanner<'_>> =
            Box::new(|lr: u32, lc: u32, subsize: u8, _state| {
                let segment_id = map[(lr * 8 + lc) as usize];
                let ref0 = if segment_id == 1 {
                    GOLDEN_FRAME
                } else {
                    LAST_FRAME
                };
                InterTreeLeaf {
                    mi_size: subsize,
                    tx_size: 0, // inferred under Only4x4
                    y_mode: ZEROMV,
                    interp_filter: 0,
                    ref_frame: [ref0, NONE_REF_FRAME],
                    mv: [[0, 0], [0, 0]],
                    skip: true,
                    segment_id,
                    sub: None,
                }
            });
        let mut coeffs: Box<FrameCoefSource<'_>> = Box::new(|_r, _c, _p, _x, _y, _b| Vec::new());
        assemble_inter_frame_tree(hdr, &plan, &mut planner, &mut coeffs).expect("segmap p-frame")
    }

    /// Build the round-418 **temporal-predicted segment-map** stream
    /// (profile 0, 8-bit 4:2:0, 64x64):
    ///
    /// * F0 — shown lossless keyframe (pattern A), fills all slots.
    /// * F1 — hidden lossless intra-only frame (pattern B) → slot 1.
    /// * F2 — shown all-skip P-frame establishing the map
    ///   **non-temporally** (§6.4.7 tree): segment-1 band on MI rows
    ///   0..3 with `SEG_LVL_REF_FRAME = GOLDEN` (those rows reconstruct
    ///   pattern B), refreshes nothing — its coded `SegmentIds` become
    ///   the §6.4.14 `PrevSegmentIds`.
    /// * F3 — shown all-skip P-frame, **`segmentation_temporal_update =
    ///   1`**: the band shifts to rows 1..4, so the §6.4.12 walk codes
    ///   `seg_id_predicted = 1` on the unchanged rows and the §6.4.7
    ///   tree escape on rows 0 and 3, threading the ctx strips across
    ///   the frame.
    /// * F4 — shown all-skip P-frame, temporal again over F3's map:
    ///   band rows 2..5 (escapes on rows 1 and 4) → slot 2.
    /// * F5 — shown all-skip `ZEROMV` copy frame over slot 2 with
    ///   segmentation disabled: a verbatim copy of F4's reconstruction.
    ///
    /// The black-box encoder wrappers only emit temporal seg-map
    /// updates through their AQ heuristics (uncontrollable placement);
    /// the *writer-side* §6.4.12 branch with hand-planned predicted /
    /// escape blocks is only mintable by the in-crate writers.
    fn build_temporal_segmap_stream() -> Vec<Vec<u8>> {
        use crate::frame_writer::{assemble_inter_frame_all_skip_zeromv, inter_pframe_header};
        use crate::pixel_encoder::{
            encode_keyframe_lossless, lossless_keyframe_header, padded_plane_from_bytes,
        };

        // F0: shown keyframe, pattern A.
        let kf = encode_vp9(&intra_only_pattern(0), 64, 64).expect("keyframe");

        // F1: hidden intra-only frame, pattern B, lossless, slot 1.
        let mut hdr1 = lossless_keyframe_header(64, 64);
        hdr1.frame_type = FrameType::NonKeyFrame;
        hdr1.intra_only = true;
        hdr1.show_frame = false;
        hdr1.reset_frame_context = 3;
        hdr1.refresh_frame_context = false;
        hdr1.refresh_frame_flags = 0x02;
        let pat_b = intra_only_pattern(1);
        let y = padded_plane_from_bytes(&pat_b[..64 * 64], 64, 64, 64, 64);
        let u = padded_plane_from_bytes(&pat_b[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32);
        let v = padded_plane_from_bytes(&pat_b[64 * 64 + 32 * 32..], 32, 32, 32, 32);
        let f1 = encode_keyframe_lossless(&hdr1, &[y, u, v]).expect("intra-only frame");

        // F2: non-temporal map establishment (band rows 0..3).
        let mut hdr2 = inter_pframe_header(64, 64);
        hdr2.ref_frame_idx = Some([0, 1, 1]);
        hdr2.refresh_frame_flags = 0;
        hdr2.segmentation = temporal_seg_params(false);
        let f2 = temporal_segmap_pframe(&hdr2, &band_map(0), None);

        // F3: temporal update, band rows 1..4, predicted vs F2's map.
        let mut hdr3 = inter_pframe_header(64, 64);
        hdr3.ref_frame_idx = Some([0, 1, 1]);
        hdr3.refresh_frame_flags = 0;
        hdr3.segmentation = temporal_seg_params(true);
        let f3 = temporal_segmap_pframe(&hdr3, &band_map(1), Some(band_map(0)));

        // F4: temporal update, band rows 2..5, predicted vs F3's map.
        let mut hdr4 = inter_pframe_header(64, 64);
        hdr4.ref_frame_idx = Some([0, 1, 1]);
        hdr4.refresh_frame_flags = 0x04;
        hdr4.segmentation = temporal_seg_params(true);
        let f4 = temporal_segmap_pframe(&hdr4, &band_map(2), Some(band_map(1)));

        // F5: verbatim copy of F4's reconstruction (slot 2), seg off.
        let mut hdr5 = inter_pframe_header(64, 64);
        hdr5.ref_frame_idx = Some([2, 2, 2]);
        hdr5.refresh_frame_flags = 0x08;
        let f5 = assemble_inter_frame_all_skip_zeromv(&hdr5).expect("copy p-frame");

        vec![kf, f1, f2, f3, f4, f5]
    }

    /// The temporal-seg-map stream decodes end-to-end with the map
    /// semantics observable in the output: the GOLDEN-override band
    /// (pattern B) sits at rows 0..3 / 1..4 / 2..5 across the three
    /// map-bearing frames — the two temporal frames recover their maps
    /// through the §6.4.12 `seg_id_predicted` walk over §6.4.14
    /// `PrevSegmentIds` — and the copy frame equals the second temporal
    /// frame byte-for-byte. Headers pin the feature class, and the
    /// stream is byte-deterministic.
    #[test]
    fn temporal_segmap_sequence_decodes_with_observable_map_semantics() {
        let frames = build_temporal_segmap_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("temporal-segmap sequence");
        assert_eq!(out.len(), 5); // F1 is hidden.

        assert_eq!(
            out[0].to_planar_bytes(),
            intra_only_pattern(0),
            "keyframe != pattern A"
        );
        assert_eq!(out[1].to_planar_bytes(), band_composite(0), "F2 band 0..3");
        assert_eq!(
            out[2].to_planar_bytes(),
            band_composite(1),
            "F3 (temporal) band 1..4"
        );
        assert_eq!(
            out[3].to_planar_bytes(),
            band_composite(2),
            "F4 (temporal) band 2..5"
        );
        assert_eq!(
            out[4].to_planar_bytes(),
            out[3].to_planar_bytes(),
            "copy frame != stored reconstruction"
        );

        // Header pins: F3/F4 code segmentation_temporal_update = 1 with
        // the chosen pred_prob; F2 codes the non-temporal map.
        let ref_dims = vec![(64u32, 64u32); 8];
        let cc0 = crate::header::parse_uncompressed_header(&frames[0])
            .expect("F0")
            .color_config;
        let parse_inter = |f: &[u8]| {
            crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: cc0,
                }),
            )
            .expect("inter header")
        };
        let h2 = parse_inter(&frames[2]);
        assert!(h2.segmentation.enabled && h2.segmentation.update_map);
        assert!(!h2.segmentation.temporal_update);
        for f in [&frames[3], &frames[4]] {
            let h = parse_inter(f);
            assert!(h.segmentation.enabled && h.segmentation.update_map);
            assert!(h.segmentation.temporal_update, "temporal seg-map frame");
            assert_eq!(h.segmentation.pred_prob, Some([180, 128, 220]));
        }

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_temporal_segmap_stream());
    }

    /// Fixture-staging generator (round 418): stages the temporal
    /// seg-map stream as `temporal-seg-predicted/input.ivf` under
    /// `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_temporal_segmap_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_temporal_segmap_stream();
        let ivf = ivf_wrap_64x64(&frames);
        let subdir = std::path::Path::new(&dir).join("temporal-seg-predicted");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    /// The staged corpus fixture is byte-identical to the builder's
    /// output — the fixture IS this crate's writer output (docs-gated).
    #[test]
    fn staged_temporal_segmap_fixture_matches_builder() {
        let path =
            std::path::Path::new("../../docs/video/vp9/fixtures/temporal-seg-predicted/input.ivf");
        if !path.is_file() {
            eprintln!("docs corpus not present; docs-gated");
            return;
        }
        let staged = std::fs::read(path).expect("staged input.ivf");
        assert_eq!(
            staged,
            ivf_wrap_64x64(&build_temporal_segmap_stream()),
            "staged fixture bytes != builder output"
        );
    }

    // ----- lossy sub-8x8 election (round 418) -----

    /// Alias-free texture for the sub-8x8-election stream: a spatial
    /// hash, so no two distinct small shifts of the plane agree on any
    /// 4x4 block (the search must have a unique zero-SAD winner).
    fn lossy_sub8x8_texture(x: i64, y: i64) -> i32 {
        let v = (x as u64)
            .wrapping_add((y as u64).wrapping_mul(131))
            .wrapping_add(7);
        let h = v
            .wrapping_mul(v)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(v.wrapping_mul(97));
        ((h >> 24) & 0xff) as i32
    }

    /// The luma displacement field of the sub-8x8-election stream:
    /// three 8x8 cells whose 4x4 quadrants move divergently — MI (2,2)
    /// left/right halves at `(0, ±4)` (VERT), MI (3,6) top/bottom
    /// halves at `(±4, 0)` (HORZ), MI (5,5) all four quadrants distinct
    /// (SPLIT) — everything else static.
    fn lossy_sub8x8_disp(x: i64, y: i64) -> (i64, i64) {
        let (cr, cc) = (y / 8, x / 8);
        let (qr, qc) = ((y % 8) / 4, (x % 8) / 4);
        match (cr, cc) {
            (2, 2) => (0, if qc == 0 { 4 } else { -4 }),
            (3, 6) => (if qr == 0 { 4 } else { -4 }, 0),
            (5, 5) => (if qr == 0 { 4 } else { -4 }, if qc == 0 { 4 } else { -4 }),
            _ => (0, 0),
        }
    }

    /// Build the round-418 **lossy sub-8x8-elected** stream (profile 0,
    /// 8-bit 4:2:0, 64x64):
    ///
    /// * F0 — shown **lossless** keyframe of the hash texture (the
    ///   decoder's reference IS the texture), fills all slots.
    /// * F1 — shown **lossy** P-frame (`base_q_idx = 80`) whose §6.4.3
    ///   layout is elected by the content-adaptive partition search:
    ///   three 8x8 cells carry opposing 4x4-quadrant motion, so the
    ///   sub-8x8 probe elects one `PARTITION_VERT` (4x8), one
    ///   `PARTITION_HORZ` (8x4) and one `PARTITION_SPLIT` (4x4) leaf
    ///   with per-cell `NEWMV` vectors — exact predictions, so the
    ///   leaves skip — while the static remainder merges upward into
    ///   large `ZEROMV` leaves. Error-resilient, refreshes slot 0.
    /// * F2 — shown all-skip `ZEROMV` copy frame over slot 0 (pins the
    ///   §8.10 store of F1's reconstruction).
    ///
    /// Unlike `sub8x8-inter-mvs` (hand-planned layout), this stream's
    /// sub-8x8 leaves are **search-elected** — the encoder's own RD
    /// probe places them.
    fn build_lossy_sub8x8_stream() -> Vec<Vec<u8>> {
        use crate::frame_writer::assemble_inter_frame_all_skip_zeromv;
        use crate::header::QuantizationParams;
        use crate::intra::Plane;
        use crate::pixel_encoder::{
            encode_pframe_lossy_tree_motion, lossless_pframe_header, PFRAME_SEARCH_RANGE,
        };

        // F0: lossless keyframe of the texture.
        let mut kf_px = vec![0u8; 64 * 64 + 2 * 32 * 32];
        for y in 0..64i64 {
            for x in 0..64i64 {
                kf_px[(y * 64 + x) as usize] = lossy_sub8x8_texture(x, y) as u8;
            }
        }
        for px in kf_px.iter_mut().skip(64 * 64) {
            *px = 128;
        }
        let kf = encode_vp9(&kf_px, 64, 64).expect("lossless keyframe");

        // F1: the search-elected sub-8x8 lossy P-frame.
        let mut targets = [Plane::new(64, 64), Plane::new(32, 32), Plane::new(32, 32)];
        for y in 0..64i64 {
            for x in 0..64i64 {
                let (dy, dx) = lossy_sub8x8_disp(x, y);
                targets[0].set(x as usize, y as usize, lossy_sub8x8_texture(x + dx, y + dy));
            }
        }
        for y in 0..32 {
            for x in 0..32 {
                targets[1].set(x, y, 128);
                targets[2].set(x, y, 128);
            }
        }
        let ref_y: Vec<i32> = (0..64i64)
            .flat_map(|y| (0..64i64).map(move |x| lossy_sub8x8_texture(x, y)))
            .collect();
        let flat_uv: Vec<i32> = vec![128; 32 * 32];
        let reference: [(&[i32], usize); 3] = [
            (ref_y.as_slice(), 64),
            (flat_uv.as_slice(), 32),
            (flat_uv.as_slice(), 32),
        ];
        let mut hdr = lossless_pframe_header(64, 64);
        hdr.quantization = QuantizationParams {
            base_q_idx: 80,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (f1, _recon) = encode_pframe_lossy_tree_motion(
            &hdr,
            &targets,
            &reference,
            None,
            64,
            64,
            PFRAME_SEARCH_RANGE,
            false,
            true,
        )
        .expect("sub-8x8-elected p-frame");

        // F2: verbatim copy of F1's reconstruction (slot 0).
        let mut hdr2 = lossless_pframe_header(64, 64);
        hdr2.refresh_frame_flags = 0x02;
        let f2 = assemble_inter_frame_all_skip_zeromv(&hdr2).expect("copy p-frame");

        vec![kf, f1, f2]
    }

    /// The lossy sub-8x8-elected stream decodes byte-exact against the
    /// encoder's reconstruction chain: the keyframe recovers the exact
    /// texture (lossless), the search-elected sub-8x8 P-frame recovers
    /// the displaced target **exactly** (the elected per-cell vectors
    /// predict perfectly, so every sub-8x8 leaf skips), and the copy
    /// frame equals it. Byte-deterministic.
    #[test]
    fn lossy_sub8x8_sequence_decodes_byte_exact() {
        let frames = build_lossy_sub8x8_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("lossy sub-8x8 sequence");
        assert_eq!(out.len(), 3);

        // F0: the exact texture.
        for y in 0..64i64 {
            for x in 0..64i64 {
                assert_eq!(
                    i64::from(out[0].y[(y * 64 + x) as usize]),
                    i64::from(lossy_sub8x8_texture(x, y) as u8),
                    "keyframe texture ({x},{y})"
                );
            }
        }
        // F1: the displaced target, exactly (all elected leaves predict
        // perfectly and skip; the static remainder is a ZEROMV copy).
        for y in 0..64i64 {
            for x in 0..64i64 {
                let (dy, dx) = lossy_sub8x8_disp(x, y);
                assert_eq!(
                    i64::from(out[1].y[(y * 64 + x) as usize]),
                    i64::from(lossy_sub8x8_texture(x + dx, y + dy) as u8),
                    "sub-8x8 frame ({x},{y})"
                );
            }
        }
        // F2: verbatim copy.
        assert_eq!(out[2].to_planar_bytes(), out[1].to_planar_bytes());

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_lossy_sub8x8_stream());
    }

    /// Fixture-staging generator (round 418): stages the lossy
    /// sub-8x8-elected stream as `lossy-sub8x8-elected/input.ivf` under
    /// `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_lossy_sub8x8_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_lossy_sub8x8_stream();
        let ivf = ivf_wrap_64x64(&frames);
        let subdir = std::path::Path::new(&dir).join("lossy-sub8x8-elected");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    /// The staged corpus fixture is byte-identical to the builder's
    /// output — the fixture IS this crate's writer output (docs-gated).
    #[test]
    fn staged_lossy_sub8x8_fixture_matches_builder() {
        let path =
            std::path::Path::new("../../docs/video/vp9/fixtures/lossy-sub8x8-elected/input.ivf");
        if !path.is_file() {
            eprintln!("docs corpus not present; docs-gated");
            return;
        }
        let staged = std::fs::read(path).expect("staged input.ivf");
        assert_eq!(
            staged,
            ivf_wrap_64x64(&build_lossy_sub8x8_stream()),
            "staged fixture bytes != builder output"
        );
    }

    // ----- lossy compound election on a cross-fade (round 418) -----

    /// Deterministic iid-noise planar frame (luma noise, flat chroma)
    /// for the compound-election stream.
    fn compound_noise_frame(seed: u64) -> Vec<u8> {
        let n = 64 * 64 + 2 * 32 * 32;
        let mut v = Vec::with_capacity(n);
        let mut s = seed;
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push(if i < 64 * 64 { (s >> 33) as u8 } else { 128 });
        }
        v
    }

    /// Build the round-418 **lossy compound-elected** stream (profile
    /// 0, 8-bit 4:2:0, 64x64) — the encoder's own reference election
    /// choosing the §8.5.2 `Round2( p0 + p1, 1 )` compound average on a
    /// cross-fade:
    ///
    /// * F0 — shown lossy keyframe (noise pattern A), fills all slots.
    /// * F1 — **hidden** lossy P-frame of noise pattern B over `LAST` =
    ///   A (single-reference, error-resilient) → slot 0. Hidden so the
    ///   compound frame's §7.2.6 `UsePrevFrameMvs` derivation yields 0.
    /// * F2 — `show_existing_frame` displaying slot 0 (pattern B).
    /// * F3 — shown **non-error-resilient** lossy P-frame of the A/B
    ///   cross-fade midpoint with `LAST` = F1's reconstruction and
    ///   `GOLDEN`/`ALTREF` = the keyframe (slot 1) under ALTREF
    ///   sign-bias asymmetry (`compoundReferenceAllowed`): the search
    ///   elects `[ LAST, ALTREF ]` compound leaves (`reference_mode =
    ///   SELECT`) because the two-reference average predicts the
    ///   midpoint — the frame codes a fraction of F1's bytes. Per §7.2
    ///   `setup_past_independence( )` an error-resilient frame zeroes
    ///   its effective sign biases, so compound REQUIRES the
    ///   non-error-resilient header (the round-418 spec fix).
    ///
    /// The corpus's existing compound streams are black-box encodes;
    /// this one pins the in-crate **election** path end-to-end.
    fn build_lossy_compound_stream() -> Vec<Vec<u8>> {
        use crate::header::QuantizationParams;
        use crate::header_writer::write_uncompressed_header;
        use crate::intra::Plane;
        use crate::pixel_encoder::{
            encode_keyframe_lossy_420_with_recon, encode_pframe_lossy_tree_motion,
            lossless_pframe_header, padded_plane_from_bytes, ReconState, PFRAME_SEARCH_RANGE,
        };

        let fa = compound_noise_frame(0x1111_2222_3333_4444);
        let fb = compound_noise_frame(0x9999_8888_7777_6666);
        let fmid: Vec<u8> = fa
            .iter()
            .zip(&fb)
            .map(|(&a, &b)| (u16::from(a) + u16::from(b)).div_ceil(2) as u8)
            .collect();
        let q = 60u8;

        let targets = |px: &[u8]| -> [Plane; 3] {
            [
                padded_plane_from_bytes(&px[..64 * 64], 64, 64, 64, 64),
                padded_plane_from_bytes(&px[64 * 64..64 * 64 + 32 * 32], 32, 32, 32, 32),
                padded_plane_from_bytes(&px[64 * 64 + 32 * 32..], 32, 32, 32, 32),
            ]
        };
        let crop3 = |r: &ReconState| -> [Vec<i32>; 3] {
            let crop = |p: &Plane, vw: usize, vh: usize| -> Vec<i32> {
                let mut out = Vec::with_capacity(vw * vh);
                for y in 0..vh {
                    for x in 0..vw {
                        out.push(p.get(x, y));
                    }
                }
                out
            };
            [
                crop(&r.planes[0], 64, 64),
                crop(&r.planes[1], 32, 32),
                crop(&r.planes[2], 32, 32),
            ]
        };
        fn as_ref_planes(v: &[Vec<i32>; 3]) -> [(&[i32], usize); 3] {
            [
                (v[0].as_slice(), 64),
                (v[1].as_slice(), 32),
                (v[2].as_slice(), 32),
            ]
        }

        let (kf, kf_recon) = encode_keyframe_lossy_420_with_recon(&fa, 64, 64, q).expect("kf");
        let gold = crop3(&kf_recon);

        // F1: hidden single-reference P of pattern B (error-resilient).
        let mut hdr1 = lossless_pframe_header(64, 64);
        hdr1.show_frame = false;
        hdr1.ref_frame_idx = Some([0, 1, 1]);
        hdr1.quantization = QuantizationParams {
            base_q_idx: q,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: false,
        };
        let (f1, p1_recon) = encode_pframe_lossy_tree_motion(
            &hdr1,
            &targets(&fb),
            &as_ref_planes(&gold),
            None,
            64,
            64,
            PFRAME_SEARCH_RANGE,
            true,
            true,
        )
        .expect("p1");
        let prev = crop3(&p1_recon);

        // F2: show_existing_frame → slot 0 (displays pattern B).
        let mut hdr_show = lossless_pframe_header(64, 64);
        hdr_show.show_existing_frame = true;
        hdr_show.frame_to_show_map_idx = Some(0);
        let f2 = write_uncompressed_header(&hdr_show).expect("show-existing packet");

        // F3: the non-error-resilient compound cross-fade frame.
        let mut hdr3 = hdr1;
        hdr3.show_frame = true;
        hdr3.error_resilient_mode = false;
        hdr3.ref_frame_sign_bias = [false, false, true];
        hdr3.refresh_frame_flags = 0;
        let gold_refs = as_ref_planes(&gold);
        let (f3, _p3_recon) = encode_pframe_lossy_tree_motion(
            &hdr3,
            &targets(&fmid),
            &as_ref_planes(&prev),
            Some(&gold_refs),
            64,
            64,
            PFRAME_SEARCH_RANGE,
            true,
            true,
        )
        .expect("p3 compound");

        vec![kf, f1, f2, f3]
    }

    /// The compound-elected stream decodes end-to-end with the election
    /// visible in the rate: the cross-fade frame (whose content is the
    /// average of its two references — exactly what the elected §8.5.2
    /// compound average predicts) codes less than a third of the
    /// single-reference noise frame's bytes. Byte-deterministic.
    #[test]
    fn lossy_compound_sequence_decodes_with_compound_rate_win() {
        let frames = build_lossy_compound_stream();
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let out = decode_vp9_sequence(&refs).expect("compound sequence");
        assert_eq!(out.len(), 3); // F1 is hidden; F2 re-displays it.

        assert!(
            frames[3].len() * 3 < frames[1].len(),
            "compound cross-fade frame ({} B) should be far below the single-ref noise frame ({} B)",
            frames[3].len(),
            frames[1].len()
        );

        // Byte-determinism (fixture staging relies on it).
        assert_eq!(frames, build_lossy_compound_stream());
    }

    /// Fixture-staging generator (round 418): stages the compound
    /// stream as `lossy-compound-elected/input.ivf` under
    /// `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_lossy_compound_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_lossy_compound_stream();
        let ivf = ivf_wrap_64x64(&frames);
        let subdir = std::path::Path::new(&dir).join("lossy-compound-elected");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    /// The staged corpus fixture is byte-identical to the builder's
    /// output — the fixture IS this crate's writer output (docs-gated).
    #[test]
    fn staged_lossy_compound_fixture_matches_builder() {
        let path =
            std::path::Path::new("../../docs/video/vp9/fixtures/lossy-compound-elected/input.ivf");
        if !path.is_file() {
            eprintln!("docs corpus not present; docs-gated");
            return;
        }
        let staged = std::fs::read(path).expect("staged input.ivf");
        assert_eq!(
            staged,
            ivf_wrap_64x64(&build_lossy_compound_stream()),
            "staged fixture bytes != builder output"
        );
    }

    // ----- lossy filtered GOP: encode-side §8.8 election (round 420) -----

    /// Source frame `t` of the filtered-GOP stream: slow diagonal ramps
    /// translating over time (luma period ~64px, low-amplitude chroma).
    /// Coarse quantization leaves small block-edge steps — inside the
    /// §8.8.5.1 filterMask thresholds — so the per-frame filter-level
    /// election picks non-zero levels throughout.
    fn filtered_gop_frame(t: i64) -> Vec<u8> {
        let (w, h, c) = (64i64, 64i64, 32i64);
        let mut px = Vec::with_capacity((w * h + 2 * c * c) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push((100 + (x * 3 + y * 2 + 5 * t) / 4 % 48) as u8);
            }
        }
        for y in 0..c {
            for x in 0..c {
                px.push((90 + (x + y * 3 + 2 * t) / 3 % 30) as u8);
            }
        }
        for y in 0..c {
            for x in 0..c {
                px.push((130 + (x * 2 + y + 3 * t) / 5 % 26) as u8);
            }
        }
        px
    }

    /// Build the round-420 **lossy filtered GOP** stream (profile 0,
    /// 8-bit 4:2:0, 64x64, `base_q_idx = 140`) through the public
    /// [`encode_vp9_lossy_sequence_error_resilient`] API (the default
    /// sequence path at staging time): a lossy keyframe + three lossy
    /// P-frames over gently-graded translating content, every frame
    /// closing out through the encode-side §8.8 loop-filter chain with
    /// a per-frame **elected** non-zero `loop_filter_level` — the
    /// reference chain threads the *filtered* reconstructions (the
    /// §8.10 post-filter store every conforming decoder keeps).
    ///
    /// The corpus's existing non-zero-filter-level streams are
    /// black-box encodes; this is the first stream whose filter levels
    /// are elected (and whose reference frames are filtered) by this
    /// crate's own encoder.
    fn build_lossy_filtered_gop_stream() -> Vec<Vec<u8>> {
        let inputs: Vec<Vec<u8>> = (0..4).map(filtered_gop_frame).collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        // The staged fixture pins the classic error-resilient framing's
        // exact bytes (the default path at staging time, round 420;
        // round 445 promoted the chain framing to the default, so the
        // builder names the frozen opt-out).
        encode_vp9_lossy_sequence_error_resilient(&refs, 64, 64, 140).expect("filtered GOP encode")
    }

    /// The filtered GOP decodes end-to-end with every header carrying a
    /// non-zero elected `loop_filter_level`, distortion bounded by the
    /// quantizer regime, and byte-determinism (fixture staging relies
    /// on it).
    #[test]
    fn lossy_filtered_gop_decodes_with_nonzero_elected_levels() {
        let frames = build_lossy_filtered_gop_stream();
        assert_eq!(frames.len(), 4);

        // Every frame's §6.2.8 header codes an elected non-zero level.
        let kf_hdr = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert!(kf_hdr.loop_filter.level > 0, "keyframe level");
        let ref_dims = vec![(64u32, 64u32); 8];
        for (i, f) in frames.iter().enumerate().skip(1) {
            let hdr = crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: kf_hdr.color_config,
                }),
            )
            .expect("p header");
            assert!(hdr.loop_filter.level > 0, "frame {i} level");
        }

        // Decoded output stays in the q=140 distortion regime against
        // the source (no drift across the filtered reference chain).
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for (i, (frame, src)) in decoded
            .iter()
            .zip((0..4).map(filtered_gop_frame))
            .enumerate()
        {
            let out = frame.to_planar_bytes();
            let mse: f64 = out
                .iter()
                .zip(&src)
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum::<f64>()
                / out.len() as f64;
            assert!(mse < 400.0, "frame {i}: MSE {mse} out of the q=140 regime");
        }

        // Byte-determinism.
        assert_eq!(frames, build_lossy_filtered_gop_stream());
    }

    /// Fixture-staging generator (round 420): stages the filtered GOP
    /// as `lossy-filtered-gop/input.ivf` under `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_lossy_filtered_gop_fixture_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        let frames = build_lossy_filtered_gop_stream();
        let ivf = ivf_wrap_64x64(&frames);
        let subdir = std::path::Path::new(&dir).join("lossy-filtered-gop");
        std::fs::create_dir_all(&subdir).expect("create stage dir");
        std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
    }

    /// The staged corpus fixture is byte-identical to the builder's
    /// output — the fixture IS this crate's writer output (docs-gated).
    #[test]
    fn staged_lossy_filtered_gop_fixture_matches_builder() {
        let path =
            std::path::Path::new("../../docs/video/vp9/fixtures/lossy-filtered-gop/input.ivf");
        if !path.is_file() {
            eprintln!("docs corpus not present; docs-gated");
            return;
        }
        let staged = std::fs::read(path).expect("staged input.ivf");
        assert_eq!(
            staged,
            ivf_wrap_64x64(&build_lossy_filtered_gop_stream()),
            "staged fixture bytes != builder output"
        );
    }

    /// Wrap coded frames in a minimal 64x64 IVF container (the layout
    /// every 64x64 staging generator in this module emits).
    fn ivf_wrap_64x64(frames: &[Vec<u8>]) -> Vec<u8> {
        ivf_wrap_dims(frames, 64, 64)
    }

    /// [`ivf_wrap_64x64`] at arbitrary display dimensions (the
    /// round-441 fixtures are 64x48).
    fn ivf_wrap_dims(frames: &[Vec<u8>], w: u16, h: u16) -> Vec<u8> {
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&w.to_le_bytes());
        ivf.extend_from_slice(&h.to_le_bytes());
        ivf.extend_from_slice(&25u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        for (i, f) in frames.iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        ivf
    }

    // ----- round-441 fixtures: lossy format matrix + LF-delta election -----

    /// Translating 8-bit texture frame `k` at any chroma geometry —
    /// shared content of the round-441 format-matrix fixtures (motion
    /// (2, 1) px/frame, so P-frames elect genuine `NEWMV` leaves).
    fn matrix_gop_frame_u8(w: usize, h: usize, cw: usize, ch: usize, k: usize) -> Vec<u8> {
        let f = |x: usize, y: usize, s: usize| (((x + 2 * k) * 7 + (y + k) * 13 + s) % 251) as u8;
        let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h {
            for x in 0..w {
                px.push(f(x, y, 0));
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                px.push(f(x, y, 40));
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                px.push(f(x, y, 90));
            }
        }
        px
    }

    /// Build the round-441 **lossy 4:4:4 GOP** (profile 1, 8-bit,
    /// 64x48, `base_q_idx = 110`) through the public
    /// [`encode_vp9_lossy_sequence_444`] API — the corpus's first
    /// **self-encoded non-4:2:0 lossy** stream, on the §7.2.6 chain
    /// framing (shown non-error-resilient P-frames, prev-MV modeling,
    /// per-frame §8.8 election, keyframe skip election).
    fn build_lossy_444_gop_stream() -> Vec<Vec<u8>> {
        let inputs: Vec<Vec<u8>> = (0..4)
            .map(|k| matrix_gop_frame_u8(64, 48, 64, 48, k))
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_444(&refs, 64, 48, 110).expect("444 GOP encode")
    }

    /// Build the round-441 **lossy 10-bit GOP** (profile 2, 4:2:0,
    /// 64x48, `base_q_idx = 110`) through the public
    /// [`encode_vp9_lossy_sequence_hbd`] API — the corpus's first
    /// **self-encoded high-bit-depth lossy** stream (the existing HBD
    /// streams are black-box encodes or lossless self-encodes).
    fn build_lossy_hbd10_gop_stream() -> Vec<Vec<u8>> {
        let (w, h, cw, ch) = (64usize, 48usize, 32usize, 24usize);
        let inputs: Vec<Vec<u16>> = (0..4usize)
            .map(|k| {
                let f = |x: usize, y: usize, s: usize| {
                    (((x + 2 * k) * 29 + (y + k) * 53 + s * 17) % 1024) as u16
                };
                let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
                for y in 0..h {
                    for x in 0..w {
                        px.push(f(x, y, 0));
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        px.push(f(x, y, 3));
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        px.push(f(x, y, 7));
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u16]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_hbd(&refs, 64, 48, 10, true, 110).expect("hbd10 GOP encode")
    }

    /// Build the round-441 **loop-filter-delta GOP** (profile 0, 8-bit
    /// 4:2:0, 64x48, `base_q_idx = 170`) through the public
    /// [`encode_vp9_lossy_sequence_chained`] API over mixed
    /// static/moving content (left half static texture — skip/`ZEROMV`
    /// blocks; right half translating — `NEWMV` blocks): the §6.2.8
    /// **delta election** codes a `loop_filter_mode_deltas` update on a
    /// mid-GOP frame and a later frame filters with the §7.2.8
    /// **persisted** value while coding no update — the corpus's first
    /// stream carrying a `loop_filter_delta_update = 1` frame.
    fn build_lossy_lf_deltas_gop_stream() -> Vec<Vec<u8>> {
        let (w, h) = (64i64, 48i64);
        let tex = |x: i64, y: i64| -> u8 { (((x * 7 + y * 13) % 61) * 4 % 251) as u8 };
        let inputs: Vec<Vec<u8>> = (0..4i64)
            .map(|k| {
                let cw = (w as usize).div_ceil(2);
                let ch = (h as usize).div_ceil(2);
                let mut px = vec![128u8; (w * h) as usize + 2 * cw * ch];
                for y in 0..h {
                    for x in 0..w {
                        px[(y * w + x) as usize] = if x < w / 2 {
                            tex(x, y)
                        } else {
                            tex(x + 3 * k, y + 2 * k)
                        };
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_chained(&refs, 64, 48, 170).expect("lf-deltas GOP encode")
    }

    /// The 4:4:4 GOP decodes end-to-end at profile 1 with full-res
    /// chroma on every frame; byte-deterministic.
    #[test]
    fn lossy_444_gop_decodes_at_profile_1() {
        let frames = build_lossy_444_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert_eq!(h0.profile, 1);
        assert!(!h0.color_config.subsampling_x && !h0.color_config.subsampling_y);
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for f in &decoded {
            assert_eq!((f.bit_depth, f.u.len()), (8, 64 * 48));
        }
        assert_eq!(frames, build_lossy_444_gop_stream());
    }

    /// The 10-bit GOP decodes end-to-end at profile 2 / 10-bit on
    /// every frame; byte-deterministic.
    #[test]
    fn lossy_hbd10_gop_decodes_at_profile_2() {
        let frames = build_lossy_hbd10_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert_eq!(h0.profile, 2);
        assert_eq!(h0.color_config.bit_depth, 10);
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for f in &decoded {
            assert_eq!((f.bit_depth, f.u.len()), (10, 32 * 24));
        }
        assert_eq!(frames, build_lossy_hbd10_gop_stream());
    }

    /// The LF-delta GOP carries the round-441 election shape: at least
    /// one P-frame codes `loop_filter_delta_update = 1` with a moved
    /// slot, and a LATER P-frame codes level > 0 with NO update — its
    /// filter runs on the §7.2.8 persisted values, so the byte-exact
    /// corpus sweep pins the persistence model against the reference
    /// decoder. Byte-deterministic.
    #[test]
    fn lossy_lf_deltas_gop_codes_update_then_persists() {
        let frames = build_lossy_lf_deltas_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        let ref_dims = vec![(64u32, 48u32); 8];
        let mut update_at: Option<usize> = None;
        let mut persist_after = false;
        for (i, f) in frames.iter().enumerate().skip(1) {
            let h = crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: h0.color_config,
                }),
            )
            .expect("p header");
            assert!(!h.error_resilient_mode, "frame {i}: chain framing");
            if h.loop_filter.delta_update {
                assert!(
                    h.loop_filter.ref_deltas.iter().any(Option::is_some)
                        || h.loop_filter.mode_deltas.iter().any(Option::is_some),
                    "frame {i}: an update must move a slot"
                );
                update_at.get_or_insert(i);
            } else if update_at.is_some() && h.loop_filter.level > 0 {
                persist_after = true;
            }
        }
        assert!(update_at.is_some(), "the election must code an update");
        assert!(
            persist_after,
            "a post-update frame must filter on the persisted values"
        );

        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        assert_eq!(decode_vp9_sequence(&refs).expect("decode").len(), 4);
        assert_eq!(frames, build_lossy_lf_deltas_gop_stream());
    }

    /// Fixture-staging generator (round 441): stages the three
    /// format-matrix / delta-election GOPs under
    /// `OXIDEAV_VP9_STAGE_DIR`.
    #[test]
    fn stage_round_441_fixtures_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        for (name, frames) in [
            ("lossy-444-gop", build_lossy_444_gop_stream()),
            ("lossy-hbd10-gop", build_lossy_hbd10_gop_stream()),
            ("lossy-lf-deltas-gop", build_lossy_lf_deltas_gop_stream()),
        ] {
            let ivf = ivf_wrap_dims(&frames, 64, 48);
            let subdir = std::path::Path::new(&dir).join(name);
            std::fs::create_dir_all(&subdir).expect("create stage dir");
            std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
        }
    }

    /// The staged round-441 fixtures are byte-identical to the
    /// builders' output — each fixture IS this crate's writer output
    /// (docs-gated, per-fixture presence-gated).
    #[test]
    fn staged_round_441_fixtures_match_builders() {
        for (name, frames) in [
            ("lossy-444-gop", build_lossy_444_gop_stream()),
            ("lossy-hbd10-gop", build_lossy_hbd10_gop_stream()),
            ("lossy-lf-deltas-gop", build_lossy_lf_deltas_gop_stream()),
        ] {
            let path = std::path::Path::new("../../docs/video/vp9/fixtures")
                .join(name)
                .join("input.ivf");
            if !path.is_file() {
                eprintln!("{name}: not yet staged; docs-gated");
                continue;
            }
            let staged = std::fs::read(&path).expect("staged input.ivf");
            assert_eq!(
                staged,
                ivf_wrap_dims(&frames, 64, 48),
                "{name}: staged fixture bytes != builder output"
            );
        }
    }

    // ----- round-445 fixtures: chained-default stream classes -----

    /// Build the round-445 **chain-framed lossless GOP** (profile 0,
    /// 8-bit 4:2:0, 64x48, 4 frames) through the public — now
    /// chained-by-default — [`encode_vp9_lossless_sequence`] API: a
    /// skip-electing lossless keyframe (flat background = exact §8.5.1
    /// DC prediction ⇒ `skip = 1` MIs) followed by shown
    /// non-error-resilient lossless P-frames (§7.2.6
    /// `UsePrevFrameMvs == 1` with §8.7.2 WHT residuals and prev-MV
    /// modeling — a moving textured patch codes real motion). The
    /// corpus's first **chain-framed lossless** stream and its first
    /// skip-elected lossless keyframe (`lossless-inter` is
    /// error-resilient framing; `odd-dims-59x37` is the classic
    /// self-encoded chain).
    fn build_lossless_chained_gop_stream() -> Vec<Vec<u8>> {
        let (w, h) = (64usize, 48usize);
        let n = w * h + 2 * 32 * 24;
        let inputs: Vec<Vec<u8>> = (0..4usize)
            .map(|k| {
                let mut px = vec![100u8; n];
                for y in 0..16usize {
                    for x in 0..16usize {
                        px[(y + 12) * w + x + 8 + 3 * k] = ((x * 31 + y * 17 + 7) % 251) as u8;
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossless_sequence(&refs, 64, 48).expect("lossless chained GOP encode")
    }

    /// The lossless chained GOP's source frames (for byte-exact
    /// round-trip assertions).
    fn lossless_chained_gop_sources() -> Vec<Vec<u8>> {
        let (w, h) = (64usize, 48usize);
        let n = w * h + 2 * 32 * 24;
        (0..4usize)
            .map(|k| {
                let mut px = vec![100u8; n];
                for y in 0..16usize {
                    for x in 0..16usize {
                        px[(y + 12) * w + x + 8 + 3 * k] = ((x * 31 + y * 17 + 7) % 251) as u8;
                    }
                }
                px
            })
            .collect()
    }

    /// Build the round-445 **12-bit 4:2:2 GOP** (profile 3, 64x48,
    /// `base_q_idx = 110`) through the public
    /// [`encode_vp9_lossy_sequence_hbd_422`] API — the §7.2 matrix's
    /// deepest corner (12-bit × the `ssx = 1, ssy = 0` geometry) as a
    /// self-encoded chain-framed stream: CAT6 18-bit tokens through
    /// the full lossy chain (motion search, compound-capable framing,
    /// §8.8 + §6.2.8 elections). The staged
    /// `profile-3-yuv422-12bit-inter` covers this format as a
    /// black-box encode; this is the first **self-encoded** one.
    fn build_lossy_hbd12_422_gop_stream() -> Vec<Vec<u8>> {
        let (w, h, cw, ch) = (64usize, 48usize, 32usize, 48usize);
        let inputs: Vec<Vec<u16>> = (0..4usize)
            .map(|k| {
                let f = |x: usize, y: usize, s: usize| {
                    (((x + 2 * k) * 61 + (y + k) * 113 + s * 29) % 4096) as u16
                };
                let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
                for y in 0..h {
                    for x in 0..w {
                        px.push(f(x, y, 0));
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        px.push(f(x, y, 5));
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        px.push(f(x, y, 11));
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u16]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_hbd_422(&refs, 64, 48, 12, 110).expect("hbd12-422 GOP encode")
    }

    /// Build the round-445 **rate-controlled GOP** (profile 0, 8-bit
    /// 4:2:0, 64x48, `target_bytes_per_frame = 1000`) through the
    /// public [`encode_vp9_lossy_sequence_rc`] API — since round 445
    /// the RC chain rides the §7.2.6 chain framing with the keyframe
    /// skip election inside the quantizer bisection and the §6.2.8
    /// delta election under the byte budget. Mixed static/moving
    /// halves (the r441 delta-election probe shape) so the per-class
    /// delta axes genuinely diverge. The corpus's first
    /// rate-controlled stream.
    fn build_lossy_rc_gop_stream() -> Vec<Vec<u8>> {
        let (w, h) = (64i64, 48i64);
        let tex = |x: i64, y: i64| -> u8 { (((x * 7 + y * 13) % 61) * 4 % 251) as u8 };
        let inputs: Vec<Vec<u8>> = (0..4i64)
            .map(|k| {
                let cw = (w as usize).div_ceil(2);
                let ch = (h as usize).div_ceil(2);
                let mut px = vec![128u8; (w * h) as usize + 2 * cw * ch];
                for y in 0..h {
                    for x in 0..w {
                        px[(y * w + x) as usize] = if x < w / 2 {
                            tex(x, y)
                        } else {
                            tex(x + 3 * k, y + 2 * k)
                        };
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_rc(&refs, 64, 48, 1000).expect("rc GOP encode")
    }

    /// The lossless chained GOP pins the r445 default-path shape:
    /// shown non-error-resilient P-frames, a skip-electing keyframe
    /// (strictly smaller than the classic writer's on this content),
    /// byte-exact lossless round-trip, and byte-determinism.
    #[test]
    fn lossless_chained_gop_fixture_shape() {
        let frames = build_lossless_chained_gop_stream();
        assert_eq!(frames.len(), 4);

        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert_eq!(h0.frame_type, FrameType::KeyFrame);
        assert!(h0.quantization.lossless, "lossless keyframe");
        let ref_dims = vec![(64u32, 48u32); 8];
        for (i, f) in frames.iter().enumerate().skip(1) {
            let hdr = crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: h0.color_config,
                }),
            )
            .expect("p header");
            assert!(
                hdr.show_frame && !hdr.error_resilient_mode,
                "frame {i}: chain framing"
            );
            assert!(hdr.quantization.lossless, "frame {i}: lossless P-frame");
        }

        // The keyframe skip election bites on the flat background.
        let sources = lossless_chained_gop_sources();
        let refs: Vec<&[u8]> = sources.iter().map(|f| f.as_slice()).collect();
        let classic = encode_vp9_lossless_sequence_error_resilient(&refs, 64, 48).expect("classic");
        assert!(
            frames[0].len() < classic[0].len(),
            "skip-elected keyframe must be strictly smaller"
        );

        // Lossless byte-exact round-trip.
        let coded_refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&coded_refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for (frame, src) in decoded.iter().zip(&sources) {
            assert_eq!(&frame.to_planar_bytes(), src, "lossless round-trip");
        }

        assert_eq!(frames, build_lossless_chained_gop_stream());
    }

    /// The 12-bit 4:2:2 GOP decodes end-to-end at profile 3 with the
    /// `ssx = 1, ssy = 0` geometry on every frame; chain-framed;
    /// byte-deterministic.
    #[test]
    fn lossy_hbd12_422_gop_fixture_shape() {
        let frames = build_lossy_hbd12_422_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert_eq!(h0.profile, 3);
        assert_eq!(h0.color_config.bit_depth, 12);
        assert!(h0.color_config.subsampling_x && !h0.color_config.subsampling_y);
        let ref_dims = vec![(64u32, 48u32); 8];
        for (i, f) in frames.iter().enumerate().skip(1) {
            let hdr = crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: h0.color_config,
                }),
            )
            .expect("p header");
            assert!(
                hdr.show_frame && !hdr.error_resilient_mode,
                "frame {i}: chain framing"
            );
        }
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for f in &decoded {
            assert_eq!((f.bit_depth, f.u.len()), (12, 32 * 48));
        }
        assert_eq!(frames, build_lossy_hbd12_422_gop_stream());
    }

    /// The RC GOP holds its per-frame byte budget, rides the chain
    /// framing, decodes end-to-end, and is byte-deterministic.
    #[test]
    fn lossy_rc_gop_fixture_shape() {
        let frames = build_lossy_rc_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        let ref_dims = vec![(64u32, 48u32); 8];
        for (i, f) in frames.iter().enumerate() {
            assert!(f.len() <= 1000, "frame {i}: budget overflow ({})", f.len());
            if i > 0 {
                let hdr = crate::header::parse_uncompressed_header_with_refs(
                    f,
                    Some(crate::header::RefFrameState {
                        ref_dims: &ref_dims,
                        color_config: h0.color_config,
                    }),
                )
                .expect("p header");
                assert!(
                    hdr.show_frame && !hdr.error_resilient_mode,
                    "frame {i}: chain framing"
                );
            }
        }
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        assert_eq!(decode_vp9_sequence(&refs).expect("decode").len(), 4);
        assert_eq!(frames, build_lossy_rc_gop_stream());
    }

    /// Fixture-staging generator (round 445): stages the three
    /// chained-default stream classes under `OXIDEAV_VP9_STAGE_DIR`.
    /// Alongside each `input.ivf` it writes `crate-decode.yuv` — the
    /// crate's own [`decode_vp9_sequence`] output in the corpus
    /// `expected.yuv` packing (shown frames concatenated, planar,
    /// little-endian pairs at 10/12-bit) — so the black-box
    /// verification step is a byte compare between a reference decode
    /// of `input.ivf` and this file.
    #[test]
    fn stage_round_445_fixtures_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        for (name, frames) in [
            ("lossless-chained-gop", build_lossless_chained_gop_stream()),
            ("lossy-hbd12-422-gop", build_lossy_hbd12_422_gop_stream()),
            ("lossy-rc-gop", build_lossy_rc_gop_stream()),
        ] {
            let ivf = ivf_wrap_dims(&frames, 64, 48);
            let subdir = std::path::Path::new(&dir).join(name);
            std::fs::create_dir_all(&subdir).expect("create stage dir");
            std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
            let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let decoded = decode_vp9_sequence(&refs).expect("crate decode");
            let mut yuv = Vec::new();
            for f in &decoded {
                yuv.extend_from_slice(&f.to_planar_bytes());
            }
            std::fs::write(subdir.join("crate-decode.yuv"), &yuv).expect("write crate-decode.yuv");
        }
    }

    /// The staged round-445 fixtures are byte-identical to the
    /// builders' output (docs-gated, per-fixture presence-gated).
    #[test]
    fn staged_round_445_fixtures_match_builders() {
        for (name, frames) in [
            ("lossless-chained-gop", build_lossless_chained_gop_stream()),
            ("lossy-hbd12-422-gop", build_lossy_hbd12_422_gop_stream()),
            ("lossy-rc-gop", build_lossy_rc_gop_stream()),
        ] {
            let path = std::path::Path::new("../../docs/video/vp9/fixtures")
                .join(name)
                .join("input.ivf");
            if !path.is_file() {
                eprintln!("{name}: not yet staged; docs-gated");
                continue;
            }
            let staged = std::fs::read(&path).expect("staged input.ivf");
            assert_eq!(
                staged,
                ivf_wrap_dims(&frames, 64, 48),
                "{name}: staged fixture bytes != builder output"
            );
        }
    }

    // ----- round-448 fixtures: the 4:4:0 public-entry stream classes -----

    /// Build the round-448 **4:4:0 GOP** (profile 1, 8-bit, 64x48,
    /// `base_q_idx = 110`) through the new public
    /// [`encode_vp9_lossy_sequence_440`] API — the corpus's first
    /// **self-encoded** 4:4:0 stream (`ssx = 0, ssy = 1`: the §8.5.2.1
    /// row-only chroma MV rounding, chroma planes `w × ceil(h/2)`),
    /// chain-framed with the full lossy pipeline. The staged
    /// `profile-1-yuv440-8bit-inter` covers this geometry as a
    /// black-box encode; this one pins the *writer* side.
    fn build_lossy_440_gop_stream() -> Vec<Vec<u8>> {
        let inputs: Vec<Vec<u8>> = (0..4usize)
            .map(|k| matrix_gop_frame_u8(64, 48, 64, 24, k))
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_440(&refs, 64, 48, 110).expect("440 GOP encode")
    }

    /// Build the round-448 **12-bit 4:4:0 GOP** (profile 3, 64x48,
    /// `base_q_idx = 110`) through the new public
    /// [`encode_vp9_lossy_sequence_hbd_440`] API — the only §7.2
    /// format-matrix corner with **no corpus fixture at all** (the
    /// 8-bit 4:4:0 geometry has `profile-1-yuv440-8bit-inter`; the
    /// high-bit-depth 4:4:0 rows have nothing): CAT6 tokens over the
    /// row-only chroma MV rounding, chain-framed.
    fn build_lossy_hbd12_440_gop_stream() -> Vec<Vec<u8>> {
        let (w, h, cw, ch) = (64usize, 48usize, 64usize, 24usize);
        let inputs: Vec<Vec<u16>> = (0..4usize)
            .map(|k| {
                let f = |x: usize, y: usize, s: usize| {
                    (((x + 2 * k) * 61 + (y + k) * 113 + s * 29) % 4096) as u16
                };
                let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
                for y in 0..h {
                    for x in 0..w {
                        px.push(f(x, y, 0));
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        px.push(f(x, y, 5));
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        px.push(f(x, y, 11));
                    }
                }
                px
            })
            .collect();
        let refs: Vec<&[u16]> = inputs.iter().map(|f| f.as_slice()).collect();
        encode_vp9_lossy_sequence_hbd_440(&refs, 64, 48, 12, 110).expect("hbd12-440 GOP encode")
    }

    /// The 4:4:0 GOP decodes end-to-end at profile 1 with the
    /// `ssx = 0, ssy = 1` geometry on every frame; chain-framed;
    /// byte-deterministic.
    #[test]
    fn lossy_440_gop_fixture_shape() {
        let frames = build_lossy_440_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert_eq!(h0.profile, 1);
        assert_eq!(h0.color_config.bit_depth, 8);
        assert!(!h0.color_config.subsampling_x && h0.color_config.subsampling_y);
        let ref_dims = vec![(64u32, 48u32); 8];
        for (i, f) in frames.iter().enumerate().skip(1) {
            let hdr = crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: h0.color_config,
                }),
            )
            .expect("p header");
            assert!(
                hdr.show_frame && !hdr.error_resilient_mode,
                "frame {i}: chain framing"
            );
        }
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for f in &decoded {
            assert!(!f.subsampling_x && f.subsampling_y, "4:4:0 output");
            assert_eq!(f.u.len(), 64 * 24, "§8.10 chroma extent");
        }
        assert_eq!(frames, build_lossy_440_gop_stream());
    }

    /// The 12-bit 4:4:0 GOP decodes end-to-end at profile 3 with the
    /// `ssx = 0, ssy = 1` geometry on every frame; chain-framed;
    /// byte-deterministic.
    #[test]
    fn lossy_hbd12_440_gop_fixture_shape() {
        let frames = build_lossy_hbd12_440_gop_stream();
        assert_eq!(frames.len(), 4);
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        assert_eq!(h0.profile, 3);
        assert_eq!(h0.color_config.bit_depth, 12);
        assert!(!h0.color_config.subsampling_x && h0.color_config.subsampling_y);
        let ref_dims = vec![(64u32, 48u32); 8];
        for (i, f) in frames.iter().enumerate().skip(1) {
            let hdr = crate::header::parse_uncompressed_header_with_refs(
                f,
                Some(crate::header::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: h0.color_config,
                }),
            )
            .expect("p header");
            assert!(
                hdr.show_frame && !hdr.error_resilient_mode,
                "frame {i}: chain framing"
            );
        }
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decode");
        assert_eq!(decoded.len(), 4);
        for f in &decoded {
            assert_eq!((f.bit_depth, f.u.len()), (12, 64 * 24));
        }
        assert_eq!(frames, build_lossy_hbd12_440_gop_stream());
    }

    /// Fixture-staging generator (round 448): stages the two 4:4:0
    /// public-entry stream classes under `OXIDEAV_VP9_STAGE_DIR`.
    /// Alongside each `input.ivf` it writes `crate-decode.yuv` — the
    /// crate's own [`decode_vp9_sequence`] output in the corpus
    /// `expected.yuv` packing — so the black-box verification step is
    /// a byte compare between a reference decode of `input.ivf` and
    /// this file.
    #[test]
    fn stage_round_448_fixtures_when_requested() {
        let Some(dir) = std::env::var_os("OXIDEAV_VP9_STAGE_DIR") else {
            return;
        };
        for (name, frames) in [
            ("lossy-440-gop", build_lossy_440_gop_stream()),
            ("lossy-hbd12-440-gop", build_lossy_hbd12_440_gop_stream()),
        ] {
            let ivf = ivf_wrap_dims(&frames, 64, 48);
            let subdir = std::path::Path::new(&dir).join(name);
            std::fs::create_dir_all(&subdir).expect("create stage dir");
            std::fs::write(subdir.join("input.ivf"), &ivf).expect("write input.ivf");
            let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let decoded = decode_vp9_sequence(&refs).expect("crate decode");
            let mut yuv = Vec::new();
            for f in &decoded {
                yuv.extend_from_slice(&f.to_planar_bytes());
            }
            std::fs::write(subdir.join("crate-decode.yuv"), &yuv).expect("write crate-decode.yuv");
        }
    }

    /// The staged round-448 fixtures are byte-identical to the
    /// builders' output (docs-gated, per-fixture presence-gated).
    #[test]
    fn staged_round_448_fixtures_match_builders() {
        for (name, frames) in [
            ("lossy-440-gop", build_lossy_440_gop_stream()),
            ("lossy-hbd12-440-gop", build_lossy_hbd12_440_gop_stream()),
        ] {
            let path = std::path::Path::new("../../docs/video/vp9/fixtures")
                .join(name)
                .join("input.ivf");
            if !path.is_file() {
                eprintln!("{name}: not yet staged; docs-gated");
                continue;
            }
            let staged = std::fs::read(&path).expect("staged input.ivf");
            assert_eq!(
                staged,
                ivf_wrap_dims(&frames, 64, 48),
                "{name}: staged fixture bytes != builder output"
            );
        }
    }

    // ----- round 445: chained framing IS the default sequence path -----

    /// Parse the P-frame headers of a coded 64x48 4:2:0 sequence and
    /// return each frame's §6.2 `error_resilient_mode` flag.
    fn pframe_er_flags(frames: &[Vec<u8>]) -> Vec<bool> {
        let h0 = crate::header::parse_uncompressed_header(&frames[0]).expect("kf header");
        let ref_dims = vec![(64u32, 48u32); 8];
        frames
            .iter()
            .skip(1)
            .map(|f| {
                crate::header::parse_uncompressed_header_with_refs(
                    f,
                    Some(crate::header::RefFrameState {
                        ref_dims: &ref_dims,
                        color_config: h0.color_config,
                    }),
                )
                .expect("p header")
                .error_resilient_mode
            })
            .collect()
    }

    /// Round-445 default-promotion pin (lossless): the default sequence
    /// entry codes shown NON-error-resilient P-frames (§7.2.6
    /// `UsePrevFrameMvs == 1` chain framing), the `_chained` alias is
    /// byte-identical to it, the `_error_resilient` opt-out codes the
    /// classic §6.2 `error_resilient_mode = 1` framing, and both decode
    /// byte-exact back to the source.
    #[test]
    fn default_lossless_sequence_rides_chain_framing() {
        let inputs: Vec<Vec<u8>> = (0..3)
            .map(|k| matrix_gop_frame_u8(64, 48, 32, 24, k))
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();

        let default = encode_vp9_lossless_sequence(&refs, 64, 48).expect("default");
        let alias = encode_vp9_lossless_sequence_chained(&refs, 64, 48).expect("alias");
        let er = encode_vp9_lossless_sequence_error_resilient(&refs, 64, 48).expect("opt-out");

        assert_eq!(default, alias, "the _chained alias IS the default path");
        assert!(
            pframe_er_flags(&default).iter().all(|&f| !f),
            "default P-frames must be non-error-resilient (chain framing)"
        );
        assert!(
            pframe_er_flags(&er).iter().all(|&f| f),
            "opt-out P-frames must code error_resilient_mode = 1"
        );

        // Both framings keep the lossless byte-exact guarantee.
        for coded in [&default, &er] {
            let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
            let decoded = decode_vp9_sequence(&coded_refs).expect("decode");
            assert_eq!(decoded.len(), inputs.len());
            for (frame, src) in decoded.iter().zip(&inputs) {
                assert_eq!(&frame.to_planar_bytes(), src, "lossless round-trip");
            }
        }
    }

    /// Round-445 default-promotion pin (lossy): chain framing on the
    /// default entry, byte-identity of the `_chained` alias, classic
    /// framing on the `_error_resilient` opt-out, and the
    /// decoder-mirror end-to-end decode on both.
    #[test]
    fn default_lossy_sequence_rides_chain_framing() {
        let inputs: Vec<Vec<u8>> = (0..3)
            .map(|k| matrix_gop_frame_u8(64, 48, 32, 24, k))
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|f| f.as_slice()).collect();

        let default = encode_vp9_lossy_sequence(&refs, 64, 48, 120).expect("default");
        let alias = encode_vp9_lossy_sequence_chained(&refs, 64, 48, 120).expect("alias");
        let er = encode_vp9_lossy_sequence_error_resilient(&refs, 64, 48, 120).expect("opt-out");

        assert_eq!(default, alias, "the _chained alias IS the default path");
        assert!(
            pframe_er_flags(&default).iter().all(|&f| !f),
            "default P-frames must be non-error-resilient (chain framing)"
        );
        assert!(
            pframe_er_flags(&er).iter().all(|&f| f),
            "opt-out P-frames must code error_resilient_mode = 1"
        );

        for coded in [&default, &er] {
            let coded_refs: Vec<&[u8]> = coded.iter().map(|f| f.as_slice()).collect();
            assert_eq!(
                decode_vp9_sequence(&coded_refs).expect("decode").len(),
                inputs.len()
            );
        }
    }
}
