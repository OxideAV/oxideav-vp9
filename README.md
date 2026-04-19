# oxideav-vp9

Pure-Rust **VP9** video header parser and partition-tree walker.
Decodes the full uncompressed header (§6.2), a useful subset of the
compressed header (§6.3), walks the tile / superblock partition
quadtree (§6.4.2) against the default keyframe probabilities (§10.5),
and ships DC / V / H / TM intra predictors plus 4×4 and 8×8 inverse
DCT-DCT as standalone primitives. Also provides a tiny IVF container
demuxer. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## What this crate actually does today

Pixel reconstruction is **not implemented**. The crate is a precise
front end: it parses the bitstream metadata, exercises the bool (range)
decoder on real compressed-header bytes, runs the partition tree down
to 8×8 blocks, and surfaces an `Error::Unsupported` with a pointer to
the exact VP9 clause where block-level decode (§6.4.3) would begin.

Concretely, given an IVF or MP4 clip carrying VP9:

- `oxideav_vp9::ivf::iter_frames` hands you the per-frame payloads.
- `oxideav_vp9::parse_uncompressed_header` returns the frame type,
  dimensions, color config, quantization parameters, segmentation,
  tile layout, and compressed-header size.
- `oxideav_vp9::parse_compressed_header` consumes `tx_mode` and
  `reference_mode` from the bool-coded compressed header.
- `oxideav_vp9::tile::decode_tiles` (or `TileDecoder::walk_partitions`)
  walks the superblock / partition quadtree and returns either a
  `PartitionPlan` of leaves or an `Error::Unsupported` with
  `§6.4.3 decode_block not implemented` as the message.
- `oxideav_vp9::Vp9Decoder` (registered as codec id `"vp9"`) parses
  headers on `send_packet` and returns the same `Unsupported` error
  on `receive_frame`, so higher-level code (containers, the codec
  registry, CLI listing) sees a clean VP9 stream with correct
  `CodecParameters` (width / height / pixel format).

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-vp9 = "0.0"
```

## Quick use (header inspection)

```rust
use oxideav_vp9::{ivf, parse_uncompressed_header, FrameType};

let data = std::fs::read("sample.ivf")?;
for frame in ivf::iter_frames(&data)? {
    let frame = frame?;
    let h = parse_uncompressed_header(frame.payload, None)?;
    println!(
        "{:?} frame, {}x{}, {} bpp",
        h.frame_type,
        h.width,
        h.height,
        h.color_config.bit_depth,
    );
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Registering the decoder

```rust
use oxideav_codec::CodecRegistry;
let mut reg = CodecRegistry::new();
oxideav_vp9::register(&mut reg);
// reg.make_decoder(&params) now returns a Vp9Decoder that parses
// headers successfully; receive_frame() fails with Error::Unsupported
// pointing at §6.4.3.
```

## What's still missing

The decoder is header-and-scaffolding only. To go from "partition plan"
to "decoded YUV" the following work is pending and not provided here:

- §6.4.3 `decode_block`: per-block mode / reference / motion-vector
  syntax (intra is simpler than inter but both need the same token
  machinery).
- §10.5 `kf_intra_mode_probs`, coefficient tree probabilities,
  `skip_prob`, inter-mode probs, MV probs. Default probability
  tables for inter-frame coding aren't carried in this crate.
- §8.6 coefficient token decode and dequantisation (AC/DC tables per
  `base_q_idx` and per-bit-depth).
- §8.5.1 directional intra predictors (`D45`, `D135`, `D117`, `D153`,
  `D207`, `D63`) and the per-edge smoothing they use.
- §8.7.1 inverse transforms beyond 4×4 / 8×8 DCT-DCT: ADST-DCT,
  DCT-ADST, ADST-ADST at all sizes; 16×16 and 32×32 DCTs; the
  lossless 4×4 Walsh-Hadamard.
- §8.8 in-loop deblock filter.
- §8.6 inter prediction: sub-pel interpolation (EIGHTTAP family),
  compound reference blending, reference-frame scaling for
  `frame_size_with_refs`.
- Multi-tile byte-prefix parsing: `decode_tiles` currently walks only
  the first tile in a multi-tile frame. The tile-size prefix syntax
  (§6.4) is simple but brings no test coverage until block decode
  lands.
- Higher bit depths (profiles 1/2/3). Only 8-bit 4:2:0 Profile 0 is
  recognised; higher-depth frames parse their headers but
  `pixel_format_from_color_config` always reports `Yuv420P`.
- Partition context derivation across a frame. The quadtree walker
  currently uses context 0 everywhere (§6.4.2's
  `partition_plane_context` requires tracking decoded block sizes
  along the above / left edges of each SB).

If you need a working VP9 decoder today, link against libvpx via a
non-oxideav crate or use `oxideav-av1` (AV1 is further along in this
workspace). This crate's goal is to be honest and precise — it
reports exactly where in the spec it gives up so future contributions
can fill in the gap incrementally.

## Codec / container IDs

- Codec: `"vp9"`; maps from MP4's `vp09` sample entry via
  `oxideav_mp4::codec_id::from_sample_entry`.
- IVF demux: `oxideav_vp9::ivf::iter_frames` (not yet a full
  `Container` impl — a thin helper for tests and standalone use).

## License

MIT — see [LICENSE](LICENSE).
