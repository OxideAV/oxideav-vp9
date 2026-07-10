# oxideav-vp9

[![CI](https://github.com/OxideAV/oxideav-vp9/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-vp9/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-vp9.svg)](https://crates.io/crates/oxideav-vp9) [![docs.rs](https://docs.rs/oxideav-vp9/badge.svg)](https://docs.rs/oxideav-vp9) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust, clean-room VP9 codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework,
implemented against the VP9 Bitstream & Decoding Process Specification
v0.7.

## Status

Intra **and inter (P-frame)** decode work end-to-end, and the encoder
is **pixel-accurate**: `encode_vp9` emits a lossless keyframe that
decodes byte-exact back to its input, `encode_vp9_lossless_sequence`
does the same for whole videos (motion-compensated P-frames with
sub-pel search), and `encode_vp9_lossy` / `encode_vp9_lossy_sequence`
provide quantized encoding over content-adaptive partition/transform
trees on **both keyframes and P-frames** (`BLOCK_8X8`..`BLOCK_64X64`,
`TX_4X4`..`TX_32X32`, per-block inter transform-size selection,
quarter/eighth-pel motion search, multi-reference LAST/GOLDEN election
and `[ LAST, ALTREF ]` compound prediction) whose output equals the
encoder's in-loop reconstruction bit-for-bit, and
`encode_vp9_lossy_sequence_rc` adds per-frame byte-budget rate control
(see the "Encoder" section). [`decode_vp9`]
and [`decode_intra_frame`] decode a complete VP9 keyframe to packed
planar samples, byte-exact against a 13-fixture staged corpus covering
4:2:0 and 4:4:4 chroma, 8/10/12-bit depth, RGB, multiple tile columns,
segmentation AQ, lossless, and both quantizer extremes.
[`decode_vp9_sequence`] decodes a multi-frame stream (keyframe followed
by P-frames), threading the §8.10 reference buffers, the §6.5 previous-frame
motion field, the §6.1.2 / §7.2 `FrameContext[ 4 ]` entropy probability
banks (`load_probs( )` / `save_probs( )`), and the §6.4.14 PrevSegmentIds
map across frames. Five corpus fixtures reconstruct byte-exact against
their `expected.yuv`: `i-frame-then-p-frame-64x64` (keyframe +
single-reference LAST P-frame, high-precision MVs, 8-tap-smooth filter),
`frame-parallel-mode` (keyframe + three single-reference P-frames at 64x64,
`error_resilient=1` so no entropy threading), `profile-0-yuv420-8bit` (the
common path: keyframe + three P-frames at **128x128** — a 2x2 superblock
grid — with `tx_mode_select`, deep partitions to 4x4, sub-8x8 inter blocks,
and `refresh_frame_context=1` so the per-frame `load_probs( ) / save_probs(
)` entropy threading is exercised), the 24-frame `show-existing-frame`
`auto-alt-ref=2` stream (hidden ARFs + `show_existing_frame` re-displays),
and `segments-aq-mode` (per-segment AQ: its P-frames carry
`segmentation_update_data=0`, pinning the §7.2.10 rule that the keyframe's
per-segment `SEG_LVL_ALT_Q` feature table persists across frames instead
of resetting to zero). `superframe-2` (hidden-ARF superframes) still
diverges by ±1 in a handful of samples from the keyframe onward (a
`TX_MODE_SELECT` small-transform / ADST reconstruction precision
discrepancy at very low quantizer) — a reconstruction, not
entropy-adaptation, discrepancy (the corpus is entirely
`frame_parallel_decoding_mode=1`, so §8.4 adaptation never runs); see the
"Not yet supported" section for the localised next-round target.

[`split_superframe`] implements the Annex B superframe split: it parses
the §B.2.1 superframe index and returns the enclosed coded-frame slices in
decode order, with the §B.4 single-frame fallback when no valid index is
present. This is VP9-intrinsic framing (Annex B of the bitstream spec) —
the split must precede the §6.2 per-frame header walk for any chunk that
might carry hidden alt-ref frames.

The decode path composes:

* §6.2 / §6.3 uncompressed- and compressed-header walkers.
* §9.1 MSB-first `f(n)` / `s(n)` bit reader and the §9.2 Boolean
  (range) decoder.
* §6.4 tile + partition + per-block walk (`decode_tiles` /
  `decode_partition` / `decode_block`), including the §6.4.6 intra
  mode-info decode and §6.4.21 / §6.4.24 residual + token decode.
* §8.5.1 intra prediction, §8.6 dequantization (per-segment
  quantizers, lossless WHT path), and §8.7 inverse transforms.
* The complete §8.8 loop filter — from the §8.8.1 frame init through
  the §8.8.2 superblock driver, §8.8.3 / §8.8.4 strength derivation,
  and the §8.8.5 sample-filter primitives, down to the §8.8
  frame-level raster ([`frame_loop_filter`]).
* §8.10 output: [`Vp9DecodedFrame`] planar `u16` samples with
  `to_planar_bytes` packing (8-bit bytes; little-endian pairs for
  10/12-bit).

The inter-frame mode-info decode is assembled through the §6.4.11
`inter_frame_mode_info` driver, which runs the full per-block prelude in
one pass — §6.4.12 `inter_segment_id`, §6.4.8 `read_skip`, §6.4.13
`read_is_inter`, and §6.4.10 `read_tx_size( !skip || !is_inter )`,
resolving the §6.4.9 `seg_feature_active` predicates (`SEG_LVL_SKIP` /
`SEG_LVL_REF_FRAME`) against the just-decoded `segment_id` — then
dispatches the §6.4.5 arm. The inter arm is the §6.4.16
`inter_block_mode_info` driver, which ties the motion-vector primitives
into one per-block pass: §6.4.17 `read_ref_frames` resolves
`ref_frame[ 0 ]` / `ref_frame[ 1 ]` (single, compound, segment-override)
atop the §9.3.2 `comp_mode` / `comp_ref` / `single_ref_p1` /
`single_ref_p2` contexts; the §6.5 reference geometry (`find_mv_refs`,
`find_best_ref_mvs`, `append_sub8x8_mvs`, `clamp_mv_*`) supplies the
`NearestMv` / `NearMv` / `BestMv` predictors; the §9.3.1
`inter_mode_tree` / `interp_filter_tree` read the per-block `inter_mode`
(via `ModeContext[ ref_frame[ 0 ] ]`) and switchable `interp_filter`;
and §6.4.18 `assign_mv` plus §6.4.19 / §6.4.20 `read_mv` /
`read_mv_component` fill `BlockMvs[ refList ][ block ]` for both the
`MiSize >= BLOCK_8X8` single-mode and sub-8x8 `(idy, idx)` partition
walks. The intra arm is the §6.4.15 `intra_block_mode_info` reader.

The §8.5.2 inter prediction (motion-compensation) process is now
complete: the §8.5.2.4 block inter prediction leaf (two-pass 8-tap
sub-pixel convolution over the `subpel_filters[4][16][8]` kernels), the
three preceding steps that feed it (§8.5.2.1 `select_mv` with the
`round_mv_comp_q2` / `round_mv_comp_q4` chroma averaging, §8.5.2.2
`clamp_mv`, §8.5.2.3 `scale_mv`), and the §8.5.2 driver
(`predict_inter`) that chains them per plane / reference-list, samples
the reference planes, and writes the single or compound-averaged
(`Round2( p0 + p1, 1 )`) result into `CurrFrame`. The §8.10 reference
frame-buffer state (`RefBuffers` — the eight `FrameStore[ ]` slots, the
`refresh_frame_flags` update, and the §8.5.2.3 `ref_frame_idx[ ]` slot
resolution) and the §6.2 inter (non-intra-only) uncompressed-header
parse (`frame_size_with_refs`, `ref_frame_idx` / `ref_frame_sign_bias`,
`allow_high_precision_mv`, §6.2.7 `read_interpolation_filter`) are also
landed — the latter pinned byte-exact against the
`i-frame-then-p-frame-64x64` corpus P-frame. The `FrameStateMvSource` /
`build_ref_planes` adapters bridge the frame-wide §6.4.4 arrays +
`RefBuffers` to the §6.5 MV-reference scan and the §8.5.2 driver.

The §6.4.4 `decode_block` inter arm is now wired into the partition
walk: the §6.4.11 `inter_frame_mode_info` decode (segment id / skip /
is_inter / tx size prelude → §6.4.16 `inter_block_mode_info` or §6.4.15
`intra_block_mode_info`), the §6.4.21 `residual( )` inter arm that runs
§8.5.2 `predict_inter( )` per plane (single + compound) before the token
loop, and the §6.4.4 fan-out of the per-block `RefFrames` / `Mvs` /
`SubMvs` / `InterpFilters` into the frame-wide arrays. The multi-frame
sequence driver ([`decode_vp9_sequence`]) threads the §8.10 `RefBuffers`
update, the inherited color config, the §6.5 previous-frame motion field,
the §7.2.6 `UsePrevFrameMvs` derivation, and the §6.4.14 previous-segment
map between frames.

### Encoder

The encoder is **pixel-accurate**, built as a mirror of the decoder's
own reconstruction loop: per coded transform block (visited in exactly
the §6.4.21 `residual( )` decode order) it predicts with the decoder's
§8.5.1 intra / §8.5.2 inter process over encoder-held reconstruction
planes, forward-transforms the `target − prediction` residual, and
replays the decoder's §8.6.2 reconstruction so the next block's
prediction sees the decoder's exact state. Public entry points:

* [`encode_vp9`] — **lossless keyframe** (8-bit 4:2:0, profile 0):
  `decode_vp9( encode_vp9( pixels ) ) == pixels` bit-for-bit. The
  forward WHT is a *perfect* integer inverse of the §8.7.2 lossless
  path (the §8.7.1.10 butterfly at `shift == 0` is proven an exact
  involution), so any input round-trips exactly — validated across
  noise / gradients / 0-255 extremes and an 11-geometry sweep (1x1
  through 256x144). [`encode_vp9_lossless_444`] (profile 1) and
  [`encode_vp9_lossless_hbd`] (profiles 2/3, 10/12-bit native `u16`)
  extend the same guarantee to all four VP9 profiles.
* [`encode_vp9_lossless_sequence`] — **lossless video**: a keyframe
  plus P-frames coding the exact `frame − prediction` residual with
  per-block `ZEROMV` / `NEWMV` integer motion search (±8 px full
  search; the §6.5.12 `BestMv` is derived with the *shared*
  `find_mv_refs` / `find_best_ref_mvs` over the same `Vp9FrameState`
  the writer codes against, and the MV difference is snapped to the
  §6.4.20-codeable grid when the §6.5.13 `use_mv_hp` gate disables the
  eighth-pel bit). Every frame decodes byte-exact through
  [`decode_vp9_sequence`]; on translating content the motion search
  codes less than half the forced-`ZEROMV` bytes. P-frames use
  error-resilient framing so the §7.2.6 `UsePrevFrameMvs == 0`
  derivation is pinned identically on both sides.
* [`encode_vp9_lossy`] / [`encode_vp9_lossy_sequence`] — **lossy**
  encoding at a caller-chosen `base_q_idx` (1..=255) with a
  **content-adaptive partition + transform-size planner**
  (`plan_keyframe_tree`): the keyframe's superblock tree splits where
  the four quadrant means deviate from the node mean by more than one
  §8.6.1 AC quantizer step (flat content codes one `BLOCK_64X64` /
  `TX_32X32` leaf per superblock; detail splits toward `BLOCK_8X8`),
  every leaf codes the largest §6.4.10-codeable transform under
  `TX_MODE_SELECT`, and per-leaf `y_mode` / `uv_mode` come from
  trial-prediction SAD over all ten §7.4.5 modes at the leaf's
  transform-block granularity (the §6.4.25 `TxType` — including the
  ADST8 / ADST16 rows — follows the coded mode). Sequence P-frames are
  motion-compensated against the previous frame's in-loop
  *reconstruction* over their own content-adaptive partition tree
  (`plan_inter_partitions` merges uniform-motion regions into leaves up
  to `BLOCK_64X64`) under `TX_MODE_SELECT`, with **per-block inter
  transform-size selection** (`select_inter_leaf_tx`: trial
  quantization at every §6.4.10-codeable size, minimum-token cost,
  ties to the larger transform), **sub-pel motion search**
  (`refine_leaf_mv_subpel`: half- / quarter- / eighth-pel descent
  scored with the decoder's own §8.5.2 8-tap interpolation, the
  eighth-pel step gated by §6.5.13 `use_mv_hp`), **multi-reference
  election** (the keyframe's reconstruction parked as a long-term
  `GOLDEN` in §8.10 slot 1, `ref_frame_idx = [0, 1, 1]`; each leaf
  codes the better of `LAST` / `GOLDEN`), **compound prediction**
  (the `[ LAST, ALTREF ]` §8.5.2 `Round2( p0 + p1, 1 )` average,
  admitted through the §6.3.12 sign-bias asymmetry and coded under
  `ReferenceModeSelect` — the cross-fade predictor), and **per-leaf
  skip election with a skip-if-no-gain guard** (a leaf codes its
  residual only when that strictly reduces its SSE, so static content
  converges to all-skip instead of re-coding quantization noise every
  frame).
  The decoder's output equals the encoder's reconstruction
  bit-for-bit at every partition / transform size (pinned
  sample-for-sample on all three planes across `TX_8X8` / `TX_16X16` /
  `TX_32X32` uniform trees, ADST-mode mixes, partial superblocks, and
  chain-level across a 4-frame sequence), so encoder and decoder never
  drift; only the bounded quantization error separates the result
  from the source. On mixed content the adaptive tree codes fewer
  bytes than the fixed all-4x4 layout at the same quantizer (pinned).
* [`encode_vp9_lossy_sequence_rc`] — **rate control**: every frame is
  coded at the lowest `base_q_idx` whose size fits a caller-chosen
  per-frame byte budget, via an exact per-frame binary search over the
  quantizer range (≤ 8 byte-deterministic trial encodes per frame;
  best-effort `q == 255` when the budget is below the frame's syntax
  floor). Budget compliance, monotone quality-vs-budget, and
  end-to-end decodability are pinned.

The encode path composes the bitstream-writer
primitives — each derived as the exact inverse of the matching decode step
and validated by round-tripping back through the in-crate decoder (no
external encoder consulted):

* the §9.2 Boolean (range) **encoder** (`bool_encoder`) — `write_bool` /
  `write_literal` / `finish`, the arithmetic-coder inverse of the §9.2
  decoder, with `0xff`-run carry propagation and §9.2.3 superframe-marker
  avoidance;
* the §6.2 uncompressed-header **writer** (`header_writer`) — the key-frame
  branch + `show_existing_frame` sentinel across all four profiles, **plus
  the inter (non-intra-only, shown) branch** (`ref_frame_idx` /
  `ref_frame_sign_bias` / explicit `frame_size` / `allow_high_precision_mv`
  / §6.2.7 `interpolation_filter`);
* the §6.3 compressed-header **writer** (`compressed_writer`) — the
  default-probability path (no forward updates) for every `tx_mode`, **both
  the intra prefix and the §6.3.9-§6.3.16 inter tail** (`inter_mode` /
  `interp_filter` / `is_inter` / `frame_reference_mode` / reference-mode /
  `y_mode` / `partition` / `mv` probability sweeps);
* the §6.4.13 / §6.4.16 / §6.4.17 / §6.4.18-§6.4.20 **inter mode-info + MV
  writers** (`inter_mode_writer` / `mv_writer`) — the inverse of
  `read_is_inter` / `read_ref_frames` / the `inter_mode` + `interp_filter`
  tokens and `assign_mv` / `read_mv` / `read_mv_component`, with the
  §6.4.20 magnitude decomposition and §6.5.13 `use_mv_hp` gate;
* the §6.4.11 / §6.4.16 **inter block writer** (`inter_block_writer`) — the
  inverse of `decode_block_inter( )` for `MiSize >= BLOCK_8X8`, reusing the
  **shared decode** §6.5 `find_mv_refs` / `find_best_ref_mvs` over the same
  `Vp9FrameState` so the MV predictors and `ModeContext` are bit-identical;
* the §6.4.24 coefficient-**token writer** (`token_writer`) — `more_coefs`
  / `token` tree / `read_coef` magnitude (incl. the CAT6 high-bit prefix at
  10/12-bit) / sign, plus the block-level `write_tokens` driver that walks
  the §6.4.25 scan deriving the §9.3.2 per-coefficient context exactly as
  the decoder does;
* the §6.4.6 keyframe intra **mode-info writer** (`mode_writer`) — a
  generic §9.3.1 `tree_encode` plus `skip` / `segment_id` /
  `default_intra_mode` / `default_uv_mode` / `tx_size`;
* the §6.4.3 **partition-tree writer** (`partition_writer`) — any
  caller-chosen `NONE` / `HORZ` / `VERT` / `SPLIT` layout
  (`write_partition_tree`, mirroring `decode_partition( )`
  arm-for-arm incl. the frame-edge conditional second leaf and the
  uncodeable-edge-pick rejection) plus the fixed all-8x8 recursion,
  with the §9.3.2 neighbour-context bookkeeping shared with the
  decoder so a written partition stream decodes to the identical leaf
  set;
* the §6.4.21 **residual encode driver** (`residual_writer`) — the inverse
  of `BlockDecoder::residual( )`, walking the same per-plane / per-4x4 grid,
  per-block tx-size (§6.4.22) / `TxType` (§6.4.25) / scan selection, and
  the §6.4.21 `AboveNonzeroContext` / `LeftNonzeroContext` write-back;
* the §6.4.4 keyframe intra **block writer** (`block_writer`) — the inverse
  of `decode_block_intra( )`, deriving every §9.3.2 neighbour context from
  the shared `Vp9FrameState`, writing the §6.4.7/§6.4.6/§6.4.21 syntax, and
  fanning the per-MI values into the frame-wide arrays via
  `decode_block_apply`;
* the **forward transforms** (`fwd_transform`) — derived exclusively
  from the spec's §8.7 *inverse* listings: the exact lossless forward
  WHT (bit-exact round-trips pinned over 2000 random + extreme-range
  vectors), the forward DCT-II / ADST4 bases matched to the §8.7.1.3 /
  §8.7.1.6 integer inverses, and the forward **ADST8 / ADST16** bases
  obtained by measuring the §8.7.1.7 / §8.7.1.8 integer networks'
  response matrices (scaled impulses through the in-crate
  transcription) and Gauss-Jordan-inverting them — every §6.4.25
  `TxType` at every transform size round-trips the decoder's integer
  inverse within a small fixed-point tolerance. `quantize_block_tx`
  applies the §8.6.2 `dqDenom` (2 at `TX_32X32`; round-to-nearest,
  error ≤ `quant / 2`) and clamps tokens into the §6.4.26-codeable
  CAT6 range per bit depth;
* the top-level **frame assemblers** (`frame_writer`) — thread the
  uncompressed + compressed headers and the single-tile §6.4 partition /
  block walk into a complete frame, with `header_size_in_bytes` set to
  the compressed-header length. The **tree-plan keyframe assembler**
  (`assemble_keyframe_tree`) codes an arbitrary partition tree with
  per-leaf `tx_size` (coded through the §6.4.10 tree under
  `TX_MODE_SELECT`, or validated against the inferred size otherwise),
  incl. non-square `HORZ` / `VERT` leaves and frame-edge-overhanging
  blocks the §6.4.3 `hasRows` / `hasCols` rules admit. The **inter tree
  assembler** (`assemble_inter_frame_tree`) walks an arbitrary §6.4.3
  partition tree with a per-leaf **planner** callback that receives the
  shared `Vp9FrameState` in decode order and dictates each leaf's full
  §6.4.11/§6.4.16 mode info — `tx_size` (coded through the §6.4.10
  tree exactly when `read_tx_size( allowSelect = !skip )` codes it,
  `TX_MODE_SELECT` included; validated against the inferred size
  otherwise), the §6.4.17 `ref_frame` pair (any single reference, or
  compound when the plan's `reference_mode` and the §6.3.18 sign-bias
  derivation admit it), mode / MVs / switchable filter / skip. The
  legacy all-`BLOCK_8X8` `assemble_inter_frame_planned` delegates
  through it (pinned byte-identical); the all-skip `ZEROMV`
  specialisation reconstructs to a verbatim copy of its `LAST`
  reference (validated byte-exact through `decode_vp9_sequence`
  across 64x64, 128x64, 128x128-mixed-tree and 40x24 geometries).

The `pixel_encoder` layer drives those writers with real content (see
the entry points above): reconstruction-mirrored prediction at every
partition / transform size, exact or quantized residuals, the
superblock-tree partition planner, per-leaf intra mode selection,
integer motion search with per-block skip election, and per-frame
quantizer rate control. Everything is byte-deterministic, and every
encode test validates end-to-end through the in-crate decoder.

### Not yet supported

* Compound + scaled-reference inter prediction are wired and unit-tested
  against an independent spec re-derivation of the §8.5.2.3 scaling +
  §8.5.2.4 convolution (half-size reference, and compound over two
  distinct per-list-scaled references), but not yet *corpus*-validated
  end-to-end (the corpus inter fixtures are single-reference, same-size
  LAST); broader inter fixtures land in later rounds.
* One corpus fixture still diverges from `expected.yuv`:
  `superframe-2` (hidden-ARF superframes, 64x64, `yac_qi=4`) diverges
  by ~13-23 bytes **per frame, starting at the keyframe** (frame 0,
  23/6144 bytes). The keyframe runs with loop-filter `level=0`, so the
  error is in the §8.6.2 reconstruction itself, not §8.8 deblocking. It
  is `TX_MODE_SELECT`-specific: the otherwise-identical `q-low` fixture
  (also 64x64, `yac_qi=4`, lossy) forces `tx_mode=ALLOW_32X32` and is
  byte-exact, while `superframe-2`'s keyframe uses `TX_MODE_SELECT` and
  mixes the smaller (4x4/8x8/16x16, ADST-eligible) transforms. The
  errors are all ±1 — a rounding/precision discrepancy in the small-tx
  or ADST reconstruct path at very low quantizer. The keyframe profile
  (23 differing bytes, every `|delta| == 1`) is pinned as an upper
  bound by `superframe2_keyframe_divergence_profile_is_bounded`, so any
  future fix or regression is caught. (The former second divergence —
  `segments-aq-mode` frames 2-3 — was the §7.2.10 segmentation feature
  persistence bug, fixed in round 406; that fixture is now byte-exact.)
* §8.4 probability adaptation / §6.1.2 `refresh_probs( )` — the complete
  §8.4 backward-adaptation transform set is implemented and unit-tested in
  the `prob_adapt` module: the §8.4.1/§8.4.2 `merge_prob` / `merge_probs`
  primitives, the §8.4.3 `adapt_coef_probs` coefficient-adaption transform
  (with `CountsToken` / `CountsMoreCoefs` accumulators), and the §8.4.4
  `adapt_noncoef_probs` non-coefficient-adaption transform (with the
  `CountsNonCoef` / `CountsMvComponent` accumulators mirroring the §9.3.4
  counting table, and the three §8.4.4 conditional gates on
  `SWITCHABLE` interp-filter / `TX_MODE_SELECT` / `allow_high_precision_mv`).
  None are yet wired into the decode loop's `refresh_probs( )`.
  Note that **every fixture in the staged corpus carries
  `frame_parallel_decoding_mode = 1`**, for which §6.1.2 `refresh_probs( )`
  skips the entire adaptation branch (`adapt_coef_probs` /
  `adapt_noncoef_probs`) and only runs `save_probs`; the current
  forward-updated-bank `save_probs` path is therefore already correct for
  the corpus, and the two divergences above are *not* attributable to
  missing backward adaptation. Wiring §8.4 fully (for non-parallel
  streams) is additionally blocked on a v0.7 docs gap: the §9.3.4
  "special case (for more_coefs)" the spec references is absent from the
  PDF (page 126 ends at the `more_coefs` counting-table row; the promised
  end-of-section paragraph is blank). Contexts are reset to §10 defaults
  per frame meanwhile, which is exact for the all-parallel-mode corpus.
* §9.2.4 multi-coder tile parallelism (tiles decode sequentially).
* Encoder depth beyond the planner-driven baseline. Keyframes **and
  P-frames** plan partitions `BLOCK_8X8`..`BLOCK_64X64` with
  transforms `TX_4X4`..`TX_32X32` (P-frames select per-block inter tx
  sizes, search motion to quarter/eighth-pel, and elect
  LAST/GOLDEN/compound references). `NEARESTMV` / `NEARMV`
  mode-mapping (a searched vector equal to a §6.5 predictor still
  codes `NEWMV`'s costlier syntax), encode-side loop filtering (frames
  are coded with `filter_level == 0`), keyframe skip election, and the
  sub-8x8 per-(idy, idx) MV walk / §6.4.12 temporal-predicted
  segment-id branch are later milestones. Lossy encoding is 8-bit
  4:2:0 (the lossless path covers all four profiles). The inter
  *writers* already carry compound references and
  `MiSize >= BLOCK_8X8`; the planner simply does not elect them yet.

## Testing

The crate carries 985+ lib unit tests plus integration suites in
`tests/` (including the keyframe **and inter** encoder writers, each
round-tripped back through the in-crate decoder; `encode_keyframe`
exercising the public `encode_vp9` → decode **byte-exact lossless**
round-trip across a geometry sweep; the lossless / lossy sequence
encoders reconstructed through `decode_vp9_sequence` with chain-level
decoder-mirror pins; and the motion-search / mode-selection rate
assertions). Tests construct their inputs bit-by-bit; §9.2 golden buffers
are hand-derived by stepping the decoder, not borrowed from any
third-party VP9 implementation. Several precision-critical primitives
also carry *independent* oracles that share no code with the
implementation they check: the §8.7 inverse DCT / ADST4 against
from-spec closed-form transform bases (and the §8.7.1.10 WHT against its
orthogonal matrix + involution property), the §9.3.2 coefficient-context
neighbour derivation against a strict scan-order causality invariant, and
every §8.5.1 intra mode against flat-preservation + `Clip1`-bound
properties. A `cargo-fuzz` harness lives in `fuzz/` with panic-surface
targets over the header parsers, the Boolean-decoder walkers, the whole
`decode_frame` pipeline (single-frame, Annex B split, and the multi-frame
sequence driver), and the `encode_keyframe` encode → decode round-trip;
the `decode_robustness` integration suite pins the same
garbage-in-no-panic contract in standard CI, including a fuzz-found
OOM regression (headers claiming huge frame geometries are rejected
against the largest per-level `Max luma picture size` — 35 651 584
samples — before any frame-sized allocation).

## Provenance

Single source of truth: the VP9 Bitstream & Decoding Process
Specification v0.7 (`docs/video/vp9/vp9-spec.txt`) plus any clean-room
trace material staged under `docs/video/vp9/`. The workspace
clean-room rule applies in full: only material in this project's
`docs/` tree informs the code. Black-box binary invocations remain
permissible as opaque validators but are not derived from.

## License

MIT. See `LICENSE`.
