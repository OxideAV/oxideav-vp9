# oxideav-vp9

Pure-Rust, clean-room VP9 codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework,
implemented against the VP9 Bitstream & Decoding Process Specification
v0.7.

## Status

Intra (key / intra-only) frame decode works end-to-end. [`decode_vp9`]
and [`decode_intra_frame`] decode a complete VP9 keyframe to packed
planar samples, byte-exact against a 13-fixture staged corpus covering
4:2:0 and 4:4:4 chroma, 8/10/12-bit depth, RGB, multiple tile columns,
segmentation AQ, lossless, and both quantizer extremes.

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

### Not yet supported

* Inter frames end-to-end: the §6.4.11 `inter_frame_mode_info` driver
  is present and tested, but it is not yet wired into the §6.4.4
  `decode_block` fan-out against the frame-wide per-MI arrays
  (`RefFrames` / `YModes` / `InterpFilters` / `Mvs`), the multi-frame
  reference-buffer state the §6.5 MV-reference scan reads from
  `RefFrames` of decoded references, or §8.5.2 inter prediction (motion
  compensation). Those are the remaining steps for an end-to-end
  inter-frame decode.
* `show_existing_frame` (returns `Error::Unsupported`).
* §8.4 probability adaptation / §6.1.2 frame-context refresh
  (single-frame decode does not persist contexts).
* §9.2.4 multi-coder tile parallelism (tiles decode sequentially).
* Encoder paths.

## Testing

The crate carries 696 lib unit tests plus integration suites in
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
