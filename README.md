# oxideav-vp9

Pure-Rust VP9 codec — clean-room re-implementation against the VP9
Bitstream & Decoding Process Specification v0.7.

## Status — 2026-05-20

**Round 1: uncompressed-header walker.** The crate exposes
`parse_uncompressed_header(&[u8]) -> Result<Vp9FrameHeader, Error>`,
which walks the structural part of VP9 spec §6.2 / §7.2:

* `frame_marker`, `Profile` (with the `profile == 3` reserved bit).
* `show_existing_frame` early-return path (returns
  `frame_to_show_map_idx`).
* `frame_type` (`KEY_FRAME` / `NON_KEY_FRAME`), `show_frame`,
  `error_resilient_mode`.
* `frame_sync_code` (`0x49 / 0x83 / 0x42`) on both key-frame and
  intra-only inter-frame paths.
* `color_config()` — `BitDepth` (8 / 10 / 12), `color_space`,
  `color_range`, `subsampling_x` / `subsampling_y` plus the §7.2.2
  `reserved_zero` and CS_RGB-on-profile-0/2 constraint checks.
* `frame_size` and `render_size`, including the
  `render_and_frame_size_different == 1` override.

`decode_vp9()` / `encode_vp9()` still return `Error::NotImplemented`;
the entropy decoder, intra/inter prediction, transforms and loop
filter land in later rounds. Inter (non-intra-only) headers — which
require `frame_size_with_refs` and reference-buffer state — return
`Error::Unsupported` from the header walker for now.

## Test surface

* `cargo test`: 3 internal MSB-first bit-reader tests
  (`src/bitreader.rs`) + 8 integration tests against synthetic byte
  buffers built per §6.2 syntax in `tests/uncompressed_header.rs`.
* No external fixtures are involved yet; each test constructs its
  input bit-by-bit and round-trips against the expected
  `Vp9FrameHeader` fields.

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

1. Walk the post-`render_size` section of §6.2: `loop_filter_params`,
   `quantization_params`, `segmentation_params`, `tile_info`,
   `header_size_in_bytes`, plus the `trailing_bits()` zero-fill from
   §6.1.1.
2. Boolean (range) decoder per §9.2 and compressed-header walk.
3. Intra prediction (§8.5) over a single tile.
4. Inverse transforms + dequant.
5. Inter prediction, loop filter, multi-tile, then encoder paths.

## License

MIT. See `LICENSE`.
