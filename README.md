# oxideav-vp9

Pure-Rust, clean-room VP9 codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework,
implemented against the VP9 Bitstream & Decoding Process Specification
v0.7.

## Status

Intra **and inter (P-frame)** decode work end-to-end, and a **keyframe
encoder** assembles a complete decoder-reconstructible frame (see the
"Encoder" section). [`decode_vp9`]
and [`decode_intra_frame`] decode a complete VP9 keyframe to packed
planar samples, byte-exact against a 13-fixture staged corpus covering
4:2:0 and 4:4:4 chroma, 8/10/12-bit depth, RGB, multiple tile columns,
segmentation AQ, lossless, and both quantizer extremes.
[`decode_vp9_sequence`] decodes a multi-frame stream (keyframe followed
by P-frames), threading the §8.10 reference buffers, the §6.5 previous-frame
motion field, the §6.1.2 / §7.2 `FrameContext[ 4 ]` entropy probability
banks (`load_probs( )` / `save_probs( )`), and the §6.4.14 PrevSegmentIds
map across frames. Four corpus fixtures reconstruct byte-exact against
their `expected.yuv`: `i-frame-then-p-frame-64x64` (keyframe +
single-reference LAST P-frame, high-precision MVs, 8-tap-smooth filter),
`frame-parallel-mode` (keyframe + three single-reference P-frames at 64x64,
`error_resilient=1` so no entropy threading), `profile-0-yuv420-8bit` (the
common path: keyframe + three P-frames at **128x128** — a 2x2 superblock
grid — with `tx_mode_select`, deep partitions to 4x4, sub-8x8 inter blocks,
and `refresh_frame_context=1` so the per-frame `load_probs( ) / save_probs(
)` entropy threading is exercised), and the 24-frame `show-existing-frame`
`auto-alt-ref=2` stream (hidden ARFs + `show_existing_frame` re-displays).
The `segments-aq-mode` (per-segment AQ) and `superframe-2` (hidden-ARF
superframes) fixtures decode end-to-end but still diverge from their
`expected.yuv`: `segments-aq-mode` is exact through frame 1 and diverges
from frame 2 (where the P-frame switches to `tx_mode=ALLOW_32X32`);
`superframe-2` diverges by ±1 in a handful of samples from the keyframe
onward (a `TX_MODE_SELECT` small-transform / ADST reconstruction
precision discrepancy at very low quantizer). Both are reconstruction —
not entropy-adaptation — discrepancies (the corpus is entirely
`frame_parallel_decoding_mode=1`, so §8.4 adaptation never runs); see the
"Not yet supported" section for the localised next-round targets.

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

[`encode_vp9`] assembles a **complete, decoder-reconstructible VP9
keyframe** end-to-end. The encode path composes the bitstream-writer
primitives — each derived as the exact inverse of the matching decode step
and validated by round-tripping back through the in-crate decoder (no
external encoder consulted):

* the §9.2 Boolean (range) **encoder** (`bool_encoder`) — `write_bool` /
  `write_literal` / `finish`, the arithmetic-coder inverse of the §9.2
  decoder, with `0xff`-run carry propagation and §9.2.3 superframe-marker
  avoidance;
* the §6.2 uncompressed-header **writer** (`header_writer`) — the key-frame
  branch + `show_existing_frame` sentinel across all four profiles;
* the §6.3 compressed-header **writer** (`compressed_writer`) — the intra,
  default-probability path (no forward updates) for every `tx_mode`;
* the §6.4.24 coefficient-**token writer** (`token_writer`) — `more_coefs`
  / `token` tree / `read_coef` magnitude (incl. the CAT6 high-bit prefix at
  10/12-bit) / sign, plus the block-level `write_tokens` driver that walks
  the §6.4.25 scan deriving the §9.3.2 per-coefficient context exactly as
  the decoder does;
* the §6.4.6 keyframe intra **mode-info writer** (`mode_writer`) — a
  generic §9.3.1 `tree_encode` plus `skip` / `segment_id` /
  `default_intra_mode` / `default_uv_mode` / `tx_size`;
* the §6.4.3 **partition-tree writer** (`partition_writer`) — the all-8x8
  partition recursion with its §9.3.2 neighbour-context bookkeeping (shared
  with the decoder so a written partition stream decodes to the identical
  leaf set);
* the §6.4.21 **residual encode driver** (`residual_writer`) — the inverse
  of `BlockDecoder::residual( )`, walking the same per-plane / per-4x4 grid,
  per-block tx-size (§6.4.22) / `TxType` (§6.4.25) / scan selection, and
  the §6.4.21 `AboveNonzeroContext` / `LeftNonzeroContext` write-back;
* the §6.4.4 keyframe intra **block writer** (`block_writer`) — the inverse
  of `decode_block_intra( )`, deriving every §9.3.2 neighbour context from
  the shared `Vp9FrameState`, writing the §6.4.7/§6.4.6/§6.4.21 syntax, and
  fanning the per-MI values into the frame-wide arrays via
  `decode_block_apply`;
* the top-level **frame assembler** (`frame_writer`) — threads the
  uncompressed + compressed headers and the single-tile §6.4 partition /
  block walk into a complete frame, with `header_size_in_bytes` set to the
  compressed-header length.

The residual writer's coefficients are validated to reconstruct to known
samples through the *full* decode pipeline (§8.6.1 dequant + §8.7 inverse
transform + §8.6.2 reconstruct): a DC-only block coded on the top-left 4x4
luma block (which predicts from no neighbours) reconstructs to `128 + r`
where `r` is the independently-computed inverse transform of the
dequantized DC. The assembler covers 8/10-bit (profile 0 / 2), 4:2:0,
segmentation (per-block `segment_id` via the §6.4.7 tree), partial- and
multi-superblock geometries (1x1 through 256x144 including degenerate
strips), and is byte-deterministic. The emitted frame is an all-skip
`DC_PRED` keyframe — structurally complete but a flat DC reconstruction
rather than a pixel-accurate encode of the input (see "Not yet supported").

### Not yet supported

* Compound + scaled-reference inter prediction are wired and unit-tested
  against an independent spec re-derivation of the §8.5.2.3 scaling +
  §8.5.2.4 convolution (half-size reference, and compound over two
  distinct per-list-scaled references), but not yet *corpus*-validated
  end-to-end (the corpus inter fixtures are single-reference, same-size
  LAST); broader inter fixtures land in later rounds.
* Two corpus fixtures still diverge from `expected.yuv`, and the
  per-frame byte diff localises the remaining work precisely:
  * `superframe-2` (hidden-ARF superframes, 64x64, `yac_qi=4`) diverges
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
    bound by `superframe2_keyframe_divergence_profile_is_bounded`, and
    the `segments-aq-mode` frame-2/3 profile by
    `segments_aq_divergence_profile_is_bounded`, so any future fix or
    regression is caught.
  * `segments-aq-mode` (per-segment AQ, 128x128) is byte-exact for
    frames 0-1 (keyframe + an `ALLOW_8X8`, `PARTITION_NONE` P-frame) and
    diverges only from frame 2 onward, where the P-frame switches to
    `tx_mode=ALLOW_32X32` (trace `tx_mode=3`) **and** all four
    superblocks switch from `PARTITION_NONE` (`bp=0`) to
    `PARTITION_SPLIT` (`bp=3`). A per-8x8-cell error heatmap of frame 2
    (524 differing bytes, `maxdelta=26`, ~equal counts across Y/U/V =
    172/169/183) localises the errors to a scattered handful of 8x8
    cells across *three* of the four superblocks — not frame-wide, and
    not a single block — so it is a per-block configuration that only
    these blocks hit, not a global segment-quantizer mismatch. Two
    distinct delta signatures are visible: (a) a smooth 16-wide block
    with a uniform `+1` gradient error (a prediction off-by-one,
    intra-in-inter or a sub-pel MV phase), and (b) a high-detail block
    with mixed `-10..+8` deltas (a wrong-coefficient residual). The
    ±tens magnitude rules out the keyframe's ±1 rounding cause; the
    near-equal Y/U/V error counts point at a motion/segment-shared
    per-block decode rather than a luma-only transform issue.
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
* Pixel-accurate encoding (forward transform + quantization + intra-mode
  / partition search). `encode_vp9` now produces a **complete,
  decoder-reconstructible keyframe** (see the "Encoder" section), but the
  emitted frame is an all-`BLOCK_8X8`, all-skip, `DC_PRED` keyframe (a
  flat DC reconstruction) rather than a rate-distortion-optimised encode
  of the input samples. Choosing the residual coefficients / intra modes /
  partition layout that reconstruct an arbitrary input frame is the next
  encoder milestone; the residual *machinery* to carry such coefficients
  is already landed and validated end-to-end (a chosen DC coefficient
  reconstructs to known samples through the full decode pipeline).

## Testing

The crate carries 860+ lib unit tests plus integration suites in
`tests/` (including the encoder writers, each round-tripped back through
the in-crate decoder, and `encode_keyframe` exercising the public
`encode_vp9` → decode round-trip across a geometry sweep). Tests construct their inputs bit-by-bit; §9.2 golden buffers
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
garbage-in-no-panic contract in standard CI.

## Provenance

Single source of truth: the VP9 Bitstream & Decoding Process
Specification v0.7 (`docs/video/vp9/vp9-spec.txt`) plus any clean-room
trace material staged under `docs/video/vp9/`. The workspace
clean-room rule applies in full: only material in this project's
`docs/` tree informs the code. Black-box binary invocations remain
permissible as opaque validators but are not derived from.

## License

MIT. See `LICENSE`.
