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
trees on **both keyframes and P-frames**. **Round 445 promotes the
§7.2.6 chain framing to the DEFAULT public GOP path**: both sequence
encoders emit shown non-error-resilient P-frames (prev-frame-MV
modeling, live `[ LAST, ALTREF ]` compound election, the r441 keyframe
skip + §6.2.8 loop-filter delta elections), with the classic
error-resilient framing kept as the explicit
`encode_vp9_lossless_sequence_error_resilient` /
`encode_vp9_lossy_sequence_error_resilient` opt-outs whose bytes are
**frozen** (the staged self-encoded fixtures pin them), and — also
round 445 — the crate registers real framework codec factories:
[`Vp9Encoder`] rides the chained default GOP path byte-identically to
the batch entries, [`Vp9Decoder`] streams packets through the new
incremental [`Vp9SequenceDecoder`] (per-packet Annex B split), and
`register( )` / [`make_decoder`] / [`make_encoder`] expose both halves
of the dual-API convention. The lossy planner covers
(`BLOCK_8X8`..`BLOCK_64X64`,
`TX_4X4`..`TX_32X32`, per-block inter transform-size selection,
quarter/eighth-pel motion search, multi-reference LAST/GOLDEN election,
`[ LAST, ALTREF ]` compound prediction, and — round 420 — the
**encode-side §8.8 loop filter** with per-frame elected
`loop_filter_level` / `loop_filter_sharpness`, the reference chain
threading the *filtered* reconstructions per the §8.10 post-filter
store) whose output equals the
encoder's in-loop reconstruction bit-for-bit, and
`encode_vp9_lossy_sequence_rc` adds per-frame byte-budget rate control
(see the "Encoder" section). **Round 441 drives the lossy pipeline
across the full §7.2 format matrix** — `encode_vp9_lossy_444` /
`encode_vp9_lossy_422` / `encode_vp9_lossy_hbd` /
`encode_vp9_lossy_hbd_422` keyframes plus the matching
`encode_vp9_lossy_sequence_*` chain-framed GOPs (profiles 1/2/3,
10/12-bit native `u16`, 4:4:4 / 4:2:2), every stream black-box
validated byte-exact — and adds **keyframe skip election** (§6.4.2:
all-zero-residual leaves code `skip = 1`, a strict rate win at
bit-exact identical reconstruction) and **loop-filter delta election**
(the six §6.2.8 / §8.8.1 `ref_deltas` / `mode_deltas` axes elected on
chain-framed P-frames on strict SSE wins, with the §7.2.8 persistent
baseline threaded writer-side exactly as the decoder folds it). [`decode_vp9`]
and [`decode_intra_frame`] decode a complete VP9 keyframe to packed
planar samples, byte-exact against a 13-fixture staged corpus covering
4:2:0 and 4:4:4 chroma, 8/10/12-bit depth, RGB, multiple tile columns,
segmentation AQ, lossless, and both quantizer extremes.
[`decode_vp9_sequence`] decodes a multi-frame stream (keyframe followed
by P-frames), threading the §8.10 reference buffers, the §6.5 previous-frame
motion field, the §6.1.2 / §7.2 `FrameContext[ 4 ]` entropy probability
banks (`load_probs( )` / `save_probs( )` — **including the full §6.1.2
`refresh_probs( )` backward adaptation on non-frame-parallel streams**,
see below), and the §6.4.14 PrevSegmentIds
map across frames. **All 43 staged corpus fixtures reconstruct fully
byte-exact against their `expected.yuv` through `decode_vp9_sequence`**
(pinned by `full_corpus_sequences_byte_exact`), including the
profile-1/2/3 P-frames (§8.5.2 motion compensation at 4:4:4 chroma and
10/12-bit depth), 181 real compound blocks across the two
`auto-alt-ref` streams, the round-406 extensions —
`profile-1-yuv422-8bit-inter` (4:2:2 chroma inter),
`intra-blocks-in-inter` (a mid-GOP scene cut coding 62 intra blocks
inside P-frames), `qcif-inter-gop` (176x144: partial superblocks with
frame-edge inter leaves) — the five round-409 extensions:
`backward-adaptation` (the corpus's first
`frame_parallel_decoding_mode=0` stream: every frame's entropy decode
runs on §8.4-adapted probabilities), `scaled-reference` (mid-GOP coded
size changes 128→64→128→96→64: §8.5.2.3 reference scaling at the 2x
conformance extreme, a 1/2x upscale, the fractional 4/3 ratio, and a
scaled `NEWMV`), `lossless-inter` (§8.7.2 WHT on inter residuals),
`tiles-2col-inter` (two tile columns on P-frames), and
`hbd-backward-adaptation` (profile-2 10-bit + backward adaptation) —
and the **twelve round-412 extensions** driving whole-corpus decode
generality: `profile-1-yuv440-8bit-inter` (**4:4:0** — `ssx=0, ssy=1`,
the §8.5.2 y-only chroma MV rounding, previously untested),
`profile-2-yuv420-12bit-inter` + `profile-3-yuv422-12bit-inter`
(**12-bit inter**, closing the bit-depth × chroma-geometry matrix),
`cyclic-refresh-aq` (**`segmentation_temporal_update=1`** P-frames —
the §6.4.12 `seg_id_predicted` branch corpus-validated — plus
inter-frame `SEG_LVL_ALT_Q`), `tiles-4col-inter` (`tile_cols_log2=2`),
`color-bt709-full-range` + `rgb-8bit-inter` (color-space / full-range
signalling incl. `CS_RGB` inter), `partial-mi-58x36-yuv420` /
`partial-mi-58x36-yuv444` (non-multiple-of-8 luma both axes, odd
29-wide chroma; the 444 stream is the regression stream for the
round-412 §8.8.2 fix — the loop filter's step-13 `onScreen` predicate
extends to the MI grid, not the visible crop, so edges at the visible
boundary filter with real reconstructed overhang samples),
`odd-dims-59x37` (truly odd luma, self-encoded — encoder pipelines
round to even), `intra-only` (a hidden **intra-only frame** with
`reset_frame_context=3`, re-displayed via `show_existing_frame` and
referenced by a P-frame — self-encoded through the round-412 §6.2
intra-only header-writer branch), and `seg-features-skip-ref-altl`
(**`SEG_LVL_SKIP` + `SEG_LVL_REF_FRAME` + `SEG_LVL_ALT_L`** segments
with the corpus's first non-zero `loop_filter_sharpness` — self-encoded
through the round-412 segment-feature-aware inter block writer; its
level-0 copy frame pins that §8.10 reference stores are post-§8.8
samples) — and the **two round-415 extensions**: `sub8x8-inter-mvs`
(the corpus's first *planned* per-sub-block motion — six sub-8x8 nodes
at 4x4 / 8x4 / 4x8 with NEWMV / NEARESTMV / NEARMV / ZEROMV cells,
integer + quarter-pel vectors and live sub-8x8 WHT residual,
self-encoded through the round-415 sub-8x8 inter writer) and
`render-size-128x72` (`render_and_frame_size_different = 1` on every
frame across all three §6.2.3 `render_size( )` call sites,
self-encoded) — and the **three round-418 extensions**:
`temporal-seg-predicted` (the corpus's first *writer-side* §6.4.12
temporal seg-map stream — hand-planned `seg_id_predicted` / tree-escape
placement over a visible `SEG_LVL_REF_FRAME = GOLDEN` band shifting
across two `segmentation_temporal_update = 1` P-frames),
`lossy-sub8x8-elected` (the encoder's own lossy partition search
electing 4x8 / 8x4 / 4x4 leaves with per-cell `NEWMV` — search-elected,
where `sub8x8-inter-mvs` is hand-planned), and `lossy-compound-elected`
(the encoder's reference election choosing `[ LAST, ALTREF ]`
§8.5.2 compound averages on a cross-fade at a 26x rate win — on a
**non-error-resilient** frame with a hidden predecessor, per the
round-418 §7.2 `setup_past_independence` finding: error-resilient
frames zero their effective `ref_frame_sign_bias`, so compound is
uncodeable there).
The **round-420 extension** `lossy-filtered-gop` is now staged — the
corpus's first stream whose non-zero `loop_filter_level`s are *elected
by this crate's own encoder* (per-frame levels 19/27/19/24 through the
public lossy sequence API; the staged package is verified byte-exact
through a black-box reference decode against the crate's own output) —
and the **round-434 extension** `tiles-2col-4row-inter` closes the
corpus's last fixture-less axis: 512x256 with `tile_rows_log2 = 2`
**and** `tile_cols_log2 = 1` on both the keyframe and the P-frame
(`Sb64Cols = 8`, `Sb64Rows = 4`, so the §6.4.1 row-major tile walk runs
4x2 = 8 genuine tile payloads per frame — eight §9.2 coder brackets,
per-tile context resets interacting with the row traversal, and intra +
§8.5.2 inter prediction across tile-row boundaries), byte-exact with no
code change — the §6.4 tile-row walk was already correct.
The three **round-441 packages** are staged and byte-exact in the
sweep: `lossy-444-gop` (the first self-encoded non-4:2:0 lossy
stream), `lossy-hbd10-gop` (the first self-encoded high-bit-depth
lossy stream), and `lossy-lf-deltas-gop` (the first stream with a
`loop_filter_delta_update = 1` frame — a mid-GOP mode-delta update
plus a later frame filtering on the §7.2.8 *persisted* value, so the
byte-exact sweep pins the persistence fold).
Three **round-445 packages** are built, black-box verified byte-exact,
and wired (presence-gated) but not yet staged — the chained-default
stream classes: `lossless-chained-gop` (the corpus's first
chain-framed **lossless** stream — §7.2.6 `UsePrevFrameMvs == 1` over
§8.7.2 WHT residuals — and its first skip-elected lossless keyframe),
`lossy-hbd12-422-gop` (the first self-encoded stream at the §7.2
matrix's deepest corner: profile 3, 12-bit, 4:2:2), and `lossy-rc-gop`
(the corpus's first **rate-controlled** stream: per-frame bisected
`base_q_idx` on the chain framing with the budget-guarded §6.2.8
delta election).
Highlights: `i-frame-then-p-frame-64x64`
(keyframe + single-reference LAST P-frame, high-precision MVs,
8-tap-smooth filter), `frame-parallel-mode` (keyframe + three
single-reference P-frames at 64x64, `error_resilient=1` so no entropy
threading), `profile-0-yuv420-8bit` (the common path: keyframe + three
P-frames at **128x128** — a 2x2 superblock grid — with `tx_mode_select`,
deep partitions to 4x4, sub-8x8 inter blocks, and
`refresh_frame_context=1` so the per-frame `load_probs( ) / save_probs( )`
entropy threading is exercised), the 24-frame `show-existing-frame`
`auto-alt-ref=2` stream (hidden ARFs + `show_existing_frame` re-displays),
`segments-aq-mode` (per-segment AQ: its P-frames carry
`segmentation_update_data=0`, pinning the §7.2.10 rule that the keyframe's
per-segment `SEG_LVL_ALT_Q` feature table persists across frames instead
of resetting to zero), and `superframe-2` (16 shown frames with hidden
`show_frame=0` alt-refs in Annex B superframes, `loop_filter_level=0` with
live deltas on every frame — pinning the §8.1 step-2 rule that the whole
§8.8 loop filter is gated on `loop_filter_level != 0`).

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
  codes less than half the forced-`ZEROMV` bytes. **Since round 445
  this entry rides the §7.2.6 chain framing by default** (the round-434
  chained model): every P-frame is shown and non-error-resilient, the
  decoder's §7.2.6 derivation is 1, and the encoder threads each
  frame's §6.4.4 motion field into the next frame's §6.5.10 candidate
  scan — temporally persistent motion that the spatial neighbours
  mispredict codes `NEARESTMV` / `NEARMV` instead of `NEWMV`, a strict
  deterministic rate win pinned on a banded-motion probe — and the
  keyframe elects §6.4.2 skip (all-zero-WHT MIs — exact DC prediction —
  a strict rate win at a still byte-exact reconstruction).
  [`encode_vp9_lossless_sequence_error_resilient`] is the explicit
  opt-out: classic §6.2 `error_resilient_mode = 1` framing (the
  §7.2.6 `UsePrevFrameMvs == 0` model, per-frame entropy independence)
  with **frozen** pre-445 bytes, pinned by the staged self-encoded
  fixtures; [`encode_vp9_lossless_sequence_chained`] remains as a
  byte-identical alias of the default.
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
  `ReferenceModeSelect` — the cross-fade predictor), **per-leaf
  skip election with a skip-if-no-gain guard** (a leaf codes its
  residual only when that strictly reduces its SSE, so static content
  converges to all-skip instead of re-coding quantization noise every
  frame), and the **encode-side §8.8 loop filter with per-frame
  parameter election** (round 420): every frame — keyframe and
  standalone `encode_vp9_lossy` keyframes included — closes out
  through the identical §8.8 chain the decoder runs (the §8.8.1 init
  over the §7.2-resolved deltas, then the superblock raster, keyed by
  the writer's own §6.4.4 per-MI arrays), with `loop_filter_level`
  elected by a full 0..=63 SSE-vs-source sweep and
  `loop_filter_sharpness` by a second-stage 0..=7 sweep at the winning
  level — both §6.2.8 fields are fixed-width, so the election is
  rate-free — and each P-frame references the previous frame's
  *filtered* reconstruction, mirroring the §8.10 post-filter
  `FrameStore[ ]`. A/B pin: keyframe elects level 49 on graded
  content at `q = 140` and the decoded GOP's SSE drops 7.6% at an
  identical 443-byte total rate vs filtering forced off.
  The standalone lossy keyframe APIs (every format) also run the
  round-441 **keyframe skip election**: leaves whose entire quantized
  residual is zero code `skip = 1` — §6.4.2 drops the residual syntax
  while a non-skip all-zero block still pays the §6.4.24 EOB token
  per coded block, and the keyframe's §6.4.6 `read_tx_size( 1 )`
  keeps coding the planned transform — a strict rate win at bit-exact
  identical reconstruction (a flat 256x192 keyframe drops 59 → 36
  bytes, 704/768 MIs elected).
  [`encode_vp9_lossy_444`] / [`encode_vp9_lossy_422`] (8-bit profile
  1) and [`encode_vp9_lossy_hbd`] / [`encode_vp9_lossy_hbd_422`]
  (10/12-bit profiles 2/3, native `u16`) extend the same keyframe
  pipeline — planner, decoder-mirror, both elections — to the full
  §7.2 format matrix (round 441), pinned by a 9-format decoder-mirror
  sweep (4:4:0 included) and black-box byte-exact validation of all
  nine matrix keyframe streams.
  The decoder's output equals the encoder's reconstruction
  bit-for-bit at every partition / transform size (pinned
  sample-for-sample on all three planes across `TX_8X8` / `TX_16X16` /
  `TX_32X32` uniform trees, ADST-mode mixes, partial superblocks, and
  chain-level across a 4-frame sequence), so encoder and decoder never
  drift; only the bounded quantization error separates the result
  from the source. On mixed content the adaptive tree codes fewer
  bytes than the fixed all-4x4 layout at the same quantizer (pinned).
* Since round 445 [`encode_vp9_lossy_sequence`] itself rides the
  **non-error-resilient chain framing** (the round-434 chained model):
  shown P-frames decode with §7.2.6 `UsePrevFrameMvs == 1`, the
  encoder threads each frame's §6.4.4 motion field into every §6.5
  predictor derivation of the next frame's search and writer, and —
  because a non-error-resilient frame keeps its *coded* sign biases —
  the `[ LAST, ALTREF ]` compound election is **live inside the
  ordinary shown GOP**: a cross-fade midpoint frame elects compound
  leaves with no hidden-predecessor construction (the r418 restriction
  dissolves), pinned on the writer's own per-MI state.
  [`encode_vp9_lossy_sequence_error_resilient`] is the explicit
  opt-out (classic framing, **frozen** pre-445 bytes — the staged
  `lossy-filtered-gop` fixture pins them; no compound, no chained
  elections); [`encode_vp9_lossy_sequence_chained`] remains as a
  byte-identical alias of the default.
  Round 441 adds two chain-framing elections: the **keyframe skip
  election** (above) and the **§6.2.8 loop-filter delta election** —
  every P-frame runs bounded coordinate descent over the six §8.8.1
  delta axes (`loop_filter_ref_deltas[ 4 ]` per reference frame,
  `loop_filter_mode_deltas[ 2 ]` per mode class) after the
  `(level, sharpness)` election, so mixed content reaches per-class
  strengths the single frame level cannot express (a static-vs-moving
  probe elects LAST +16 / ZEROMV-class −16); a moved slot costs a
  coded §6.2.8 update (1 + 7 bits), so the election only moves on a
  strict SSE win, codes exactly the slots that moved off the §7.2.8
  persistent baseline, and threads that baseline across the chain
  exactly as the decoder's persistent fold does — pinned by a
  decoder-mirror at coded deltas *plus* an update-free successor
  frame, and black-box byte-exact validation of a delta-electing GOP.
  [`encode_vp9_lossy_sequence_444`] / [`encode_vp9_lossy_sequence_422`]
  / [`encode_vp9_lossy_sequence_hbd`] /
  [`encode_vp9_lossy_sequence_hbd_422`] (round 441) run this whole
  chain-framed pipeline at every §7.2 format, each GOP black-box
  validated byte-exact.
* [`encode_vp9_lossy_sequence_rc`] — **rate control**: every frame is
  coded at the lowest `base_q_idx` whose size fits a caller-chosen
  per-frame byte budget, via an exact per-frame binary search over the
  quantizer range (≤ 8 byte-deterministic trial encodes per frame;
  best-effort `q == 255` when the budget is below the frame's syntax
  floor). Since round 445 the RC chain rides the §7.2.6 chain framing
  with the keyframe skip election inside the bisection (a strict rate
  win, so the fitted quantizer can only improve) and the §6.2.8
  loop-filter delta election **under the byte budget**: a moved slot
  costs coded §6.2.8 update bits, so an over-budget update falls back
  to the update-free frame — same length as the fitted trial, filtered
  on the §7.2.8 *persisted* baseline exactly as the decoder folds it,
  persistent state unmoved on both sides. Budget compliance, monotone
  quality-vs-budget, chain-framing headers, and end-to-end
  decodability are pinned.

### Framework registry (round 445)

`register( )` installs a real decode+encode registration under the
`"vp9"` codec id (capabilities: lossy + lossless, the nine §7.2
format-matrix pixel formats; wire-tag claims `VP90` / `VP09` /
`V_VP9`; a typed `q` / `lossless` encoder-options schema), and
[`make_decoder`] / [`make_encoder`] expose the direct-factory half of
the dual-API convention:

* [`Vp9Encoder`] rides the **chained default GOP path** one frame per
  `send_frame` (the stateful push forms of the sequence engines):
  packet bytes are pinned **byte-identical** to the matching batch
  entry at 8-bit 4:2:0 / 4:4:4, 10/12-bit, and under `lossless=true`
  (8-bit 4:2:0), with keyframe packet flags and pts passthrough.
* [`Vp9Decoder`] streams packets through [`Vp9SequenceDecoder`] — the
  incremental, stateful form of [`decode_vp9_sequence`] (which is now
  a thin loop over it): the §8.10 reference buffers + `FrameStore[ ]`,
  the §6.5 previous-frame motion field, the §6.1.2 / §7.2
  `FrameContext[ 4 ]` banks with §8.4 backward adaptation, and the
  §7.2.8 / §7.2.10 persistent header state all thread across packets,
  with the §B.2 Annex B split applied per packet — a whole-corpus
  packet-by-packet sweep decodes equal to the batch API (superframes,
  hidden alt-refs, `show_existing_frame` included; the 4:4:0 stream
  surfaces `Unsupported` at the frame-conversion boundary since the
  framework has no 4:4:0 pixel-format label).

The encode path composes the bitstream-writer
primitives — each derived as the exact inverse of the matching decode step
and validated by round-tripping back through the in-crate decoder (no
external encoder consulted):

* the §9.2 Boolean (range) **encoder** (`bool_encoder`) — `write_bool` /
  `write_literal` / `finish`, the arithmetic-coder inverse of the §9.2
  decoder, with `0xff`-run carry propagation and §9.2.3 superframe-marker
  avoidance;
* the §6.2 uncompressed-header **writer** (`header_writer`) — the key-frame
  branch + `show_existing_frame` sentinel across all four profiles, **the
  inter branch — shown or hidden** (`ref_frame_idx` /
  `ref_frame_sign_bias` / explicit `frame_size` / `allow_high_precision_mv`
  / §6.2.7 `interpolation_filter`; a hidden inter frame codes the explicit
  `intra_only = 0` flag), **and the intra-only branch** (`show_frame == 0`,
  `intra_only = 1`: `reset_frame_context`, mid-stream `frame_sync_code`,
  the `Profile > 0` `color_config( )`, `refresh_frame_flags`, explicit
  sizes — the keyframe assemblers accept intra-only headers since §6.4.5
  `mode_info( )` and the §9.3.2 probability selection key on
  `FrameIsIntra`, so both intra classes code the identical body);
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
  uncompressed + compressed headers and the §6.4 partition / block walk
  into a complete frame, with `header_size_in_bytes` set to the
  compressed-header length. The tree assemblers accept **any §6.2.13
  tiling** (round 434): they mirror the §6.4 `decode_tiles( )`
  row-major walk — one §9.2 coder bracket per tile, f(32) `tile_size`
  prefixes on every tile but the last, above-context strips carrying
  across tiles with per-tile-row left resets, the §6.5 candidate scans
  clamped to each tile's `MiColStart` / `MiColEnd` window, and the
  §6.4.4 `AvailL = MiCol > MiColStart` intra clamp at column edges —
  pinned by reconstruction-identity across row splits at the staged
  fixture's 2col x 4row tiling (tile rows never change the §8
  reconstruction), a degenerate empty-tile-row geometry, an
  all-inter tiled P-frame equal to its single-tile assembly, and a
  pinned reconstruction *difference* across a column split (the
  writer must lose the left neighbour exactly where the decoder
  does). The **tree-plan keyframe assembler**
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

### §8.4 backward adaptation (round 409)

The full §6.1.2 `refresh_probs( )` process is wired: on streams with
`error_resilient_mode == 0 && frame_parallel_decoding_mode == 0`, every
decoded frame accumulates the complete §9.3.4 syntax-element count bank
(`FrameCounts`: `counts_token` / `counts_more_coefs` plus every
non-coefficient array down to the mv family) and folds it back into the
`FrameContext` bank — `load_probs( )` per its §7.1.2 definition (all
tables *except* `tx_probs` / `skip_prob`), §8.4.3 `adapt_coef_probs( )`
with the `LastFrameType`-driven updateFactor, `load_probs2( )` + §8.4.4
`adapt_noncoef_probs( )` on inter frames, `save_probs( )`. The
`more_coefs` counting implements the §9.3.4 special case **absent from
the v0.7 spec text**, reconstructed in
`docs/video/vp9/vp9-errata-and-clarifications.md` (#249 part 1): count
only where the element is decoded (the `checkEob` branch), never
implied after a `ZERO_TOKEN`. Two further §9.3 step-2 subtleties are
implemented and pinned: the corner-inferred `PARTITION_SPLIT` and the
absent `mv_hp` / `mv_class0_hp` bit are both *counted* even though no
bit is read (§9.3.1 integer-return arms). Corpus proof: the
`backward-adaptation` and `hbd-backward-adaptation` fixtures decode
byte-exact — with adaptation disabled they fail outright, so the pin
covers the whole counting + adaptation chain. The inter-frame
`uv_mode_probs` table joined the persisted bank (§8.4.4 adapts it; §6.3
carries no forward update for it).

### Not yet supported

* ~~Scaled-reference inter prediction~~ — **corpus-validated as of
  round 409**: the `scaled-reference` fixture (self-encoded per the
  #249-part-2 tooling-gap workaround — no black-box encoder CLI mints
  mid-stream coded-size changes) decodes byte-exact against a black-box
  reference decode, covering the 2x conformance extreme, a 1/2x
  upscale, the fractional 4/3 ratio, and a scaled `NEWMV`; two
  closed-form §8.5.2.3 phase-0 identities are pinned in-crate.
* ~~Tile **rows** (`tile_rows_log2 >= 1`)~~ — **corpus-validated as of
  round 434**: the staged `tiles-2col-4row-inter` fixture (docs ask
  #270) decodes byte-exact with no code change. The long-standing
  "custom encoder tooling needed" conclusion was wrong: the black-box
  encoder clamps tile rows to 0 whenever it runs more than one thread,
  and a single-threaded encode emits them (see the fixture's
  `notes.md`), so the gap was a threading interaction, not a missing
  feature. The three other round-412-verified wrapper gaps were closed
  by self-encoding: non-zero `loop_filter_sharpness` and the
  per-segment `SEG_LVL_SKIP` / `SEG_LVL_REF_FRAME` / `SEG_LVL_ALT_L`
  features (the wrapper only ever emits `SEG_LVL_ALT_Q`) are
  corpus-validated via the `seg-features-skip-ref-altl` fixture
  (round 412), and `render_and_frame_size_different = 1` (SAR
  signalling doesn't mint the bitstream field) is corpus-validated via
  the round-415 `render-size-128x72` fixture across all three §6.2.3
  `render_size( )` call sites — the encoder-tooling-gap ledger is now
  fully closed.
* §9.2.4 multi-coder tile parallelism (tiles decode sequentially).
* Encoder depth beyond the planner-driven baseline. Keyframes **and
  P-frames** plan partitions `BLOCK_4X4`..`BLOCK_64X64` with
  transforms `TX_4X4`..`TX_32X32` (P-frames select per-block inter tx
  sizes, search motion to quarter/eighth-pel, elect LAST/GOLDEN and —
  on non-error-resilient frames — `[ LAST, ALTREF ]` compound
  references, and probe each 8x8 cell's 4x4 quadrants to elect
  below-8x8 leaves where divergent motion wins — round 418). The
  §6.4.12 temporal-predicted segment-id **writer** branch landed in
  round 418, closing the last §6.4.11 writer arm; **encode-side §8.8
  loop filtering with per-frame `level` / `sharpness` election landed
  in round 420** (the sequence encoders' reference chains thread
  filtered reconstructions); **previous-frame-MV modeling in the
  writer landed in round 434** — `InterFrameTreePlan::prev_frame_mvs`
  supplies the previous frame's §6.4.4 motion field and the writer
  scans it through the decoder's own shared §6.5.10 path, so
  non-error-resilient SHOWN P-frame chains (and therefore compound
  without a hidden/intra predecessor) are codeable, pinned by a
  NEARMV-only-reachable-through-the-prev-field differential, an
  error-resilient-twin sample-identity, and a compound-average check
  on a shown chain — with **chained variants of both sequence
  encoders** ([`encode_vp9_lossless_sequence_chained`] /
  [`encode_vp9_lossy_sequence_chained`]) shipping on that model.
  **Keyframe skip election landed in round 441** (standalone keyframes
  at every format + the chain-framed sequences; the classic sequence
  paths keep their staged-fixture-pinned bytes), as did the **§6.2.8
  loop-filter delta election** on chain-framed P-frames (per-segment
  `SEG_LVL_ALT_L` election stays out of scope while the lossy encoders
  code single-segment frames). **The chained framing became the
  default sequence path in round 445** — no fixture restaging was
  needed: the staged self-encoded packages pin the classic encoders'
  exact bytes through the explicit `_error_resilient` opt-out entries
  (whose output is frozen as the pre-445 default bytes), the r441
  elections + a new lossless keyframe skip election ride the default,
  and the framework registry Encoder rides the same path.
  **Lossy encoding covers the full §7.2 format matrix as of round
  441** — 8-bit 4:2:0/4:4:4/4:2:2 (profiles 0/1) and 10/12-bit
  4:2:0/4:4:4/4:2:2 (profiles 2/3) on both standalone keyframes and
  chain-framed sequences, every stream black-box validated; 4:4:0 has
  no dedicated public entry, but the internals are
  subsampling-generic and the 9-format decoder-mirror sweep pins it.
  The inter *writers* carry compound references and
  **every** `MiSize` — the round-415 sub-8x8 per-(idy, idx) MV walk
  included, driven by `encode_pframe_lossless_layout` over arbitrary
  §6.4.3 layouts; the partition search simply does not *elect*
  compound or sub-8x8 shapes yet.

## Testing

The crate carries 1250+ tests (lib unit tests plus integration suites
in `tests/`, including the keyframe **and inter** encoder writers, each
round-tripped back through the in-crate decoder; `encode_keyframe`
exercising the public `encode_vp9` → decode **byte-exact lossless**
round-trip across a geometry sweep; the lossless / lossy sequence
encoders reconstructed through `decode_vp9_sequence` with chain-level
decoder-mirror pins; the `registry_codec` suite pinning the framework
Encoder byte-identical to the batch entries and the framework Decoder
equal to the batch decode across the whole staged corpus; and the
motion-search / mode-selection rate assertions). Tests construct their inputs bit-by-bit; §9.2 golden buffers
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
sequence driver), the `encode_keyframe` encode → decode round-trip,
the `encode_lossy_keyframe` **oracle-carrying** round-trip (a stream the
lossy encoder emitted — elected §8.8 filter included — MUST decode), and
the `encode_chained_sequence` round-trip carrying the **full lossless
oracle** over the §7.2.6 chain model (every decoded frame must equal
its input byte-exact — a 24-case deterministic smoke of the same body
also runs in standard CI), and the `encode_lossy_matrix`
oracle-carrying round-trip over all seven public-entry §7.2 formats
(fuzz-derived format / geometry / quantizer / bit-depth-masked
content through the matching matrix keyframe entry; a 28-case
deterministic smoke also runs in standard CI);
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
