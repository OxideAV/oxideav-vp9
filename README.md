# oxideav-vp9

Pure-Rust, clean-room VP9 codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework,
implemented against the VP9 Bitstream & Decoding Process Specification
v0.7.

## Status

Intra **and inter (P-frame)** decode work end-to-end. [`decode_vp9`]
and [`decode_intra_frame`] decode a complete VP9 keyframe to packed
planar samples, byte-exact against a 13-fixture staged corpus covering
4:2:0 and 4:4:4 chroma, 8/10/12-bit depth, RGB, multiple tile columns,
segmentation AQ, lossless, and both quantizer extremes.
[`decode_vp9_sequence`] decodes a multi-frame stream (keyframe followed
by P-frames), threading the §8.10 reference buffers + §6.5 previous-frame
motion field across frames. Two corpus fixtures reconstruct byte-exact
against their `expected.yuv`: the `i-frame-then-p-frame-64x64` fixture
(keyframe + single-reference LAST P-frame, high-precision MVs, 8-tap-smooth
filter) and the `frame-parallel-mode` fixture (keyframe + **three**
consecutive single-reference P-frames at 64x64, `error_resilient=1` /
`parallel_mode=1` / `refresh_ctx=0`, so no inter-frame entropy adaptation —
the §8.10 reference threading is validated across a longer P-frame run).

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

### Not yet supported

* Compound + scaled-reference inter prediction are wired but not yet
  fixture-validated (the validated inter fixtures are single-reference,
  same-size LAST); broader inter fixtures land in later rounds.
* Hidden alt-ref superframes (`superframe-2`) and the
  `show-existing-frame` corpus decode their keyframe + early frames but
  hit a divergence on a later hidden-ARF P-frame; the Annex B split is
  in place but the full hidden-ARF inter path is not yet byte-exact.
* §8.4 probability adaptation / §6.1.2 frame-context refresh
  (contexts are reset per frame; running adaptation is deferred).
* §9.2.4 multi-coder tile parallelism (tiles decode sequentially).
* Encoder paths.

## Testing

The crate carries 738 lib unit tests plus integration suites in
`tests/`. Tests construct their inputs bit-by-bit; §9.2 golden buffers
are hand-derived by stepping the decoder, not borrowed from any
third-party VP9 implementation. A `cargo-fuzz` harness lives in
`fuzz/` with panic-surface targets over the header parsers and the
Boolean-decoder walkers.

## Provenance

Single source of truth: the VP9 Bitstream & Decoding Process
Specification v0.7 (`docs/video/vp9/vp9-spec.txt`) plus any clean-room
trace material staged under `docs/video/vp9/`. The workspace
clean-room rule applies in full: only material in this project's
`docs/` tree informs the code. Black-box binary invocations remain
permissible as opaque validators but are not derived from.

## License

MIT. See `LICENSE`.
