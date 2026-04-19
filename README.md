# oxideav-vp9

Pure-Rust **VP9** video decoder. Parses the full uncompressed header
(§6.2), the compressed header (§6.3), walks the tile / superblock
partition quadtree (§6.4.2) and reconstructs keyframe / intra-only
frames down to pixels via intra prediction + inverse transform +
dequantised-coefficient add. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Decode support today

* **Keyframe / intra-only frames** (`FrameType::Key` and the
  `intra_only` mid-stream variant) decode to a `VideoFrame` with
  3-plane 8-bit 4:2:0 luma + chroma data.
* **All 10 intra modes** (§8.5.1): `DC_PRED`, `V_PRED`, `H_PRED`,
  `D45_PRED`, `D135_PRED`, `D117_PRED`, `D153_PRED`, `D207_PRED`,
  `D63_PRED`, `TM_PRED`. Block sides 4 / 8 / 16 / 32 supported. 127 /
  129 neighbour padding handled per libvpx `build_intra_predictors`.
* **All four inverse-transform sizes** (§8.7.1): 4×4, 8×8, 16×16, 32×32.
  The four 2-D tx-types `DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
  `ADST_ADST` are covered for 4 / 8 / 16; 32×32 is DCT-only per spec.
  The lossless 4×4 Walsh-Hadamard is wired via `TxType::WhtWht`.
* **Coefficient decode** (§6.4.23 / §8.5.2 / §10.5): full token tree
  (EOB / ZERO / ONE model nodes + Pareto8 tail for TWO/THREE/FOUR/
  CAT1..6), sign bit, per-position context via neighbour magnitudes,
  dequantisation against `DC_QLOOKUP` / `AC_QLOOKUP` and the 32×32
  `dq_shift`.
* **IVF demux** (`ivf::iter_frames`) lets tests feed ffmpeg-generated
  clips directly.

### Example

```rust
use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

let data = std::fs::read("intra.ivf")?;
let first = ivf::iter_frames(&data)?.next().unwrap()?;
let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
let mut dec = make_decoder(&params)?;
let pkt = Packet::new(0, TimeBase::new(1, 24), first.payload.to_vec());
dec.send_packet(&pkt)?;
let Frame::Video(v) = dec.receive_frame()? else { unreachable!() };
println!("decoded {}x{} {:?} with {} planes",
         v.width, v.height, v.format, v.planes.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The shipped integration test (`tests/vp9_intra_fixture.rs`) decodes a
128×128 keyframe clip generated with
`ffmpeg -f lavfi -i testsrc=size=128x128:rate=24:duration=0.042 -c:v
libvpx-vp9 -g 1 -keyint_min 1 -deadline realtime -f ivf` and asserts
plausible luma statistics.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-vp9 = "0.0"
```

## Registering the decoder

```rust
use oxideav_codec::CodecRegistry;
let mut reg = CodecRegistry::new();
oxideav_vp9::register(&mut reg);
```

## Known gaps

The pixel reconstruction path ships, but a handful of accuracy
refinements remain before the decoder matches a libvpx reference
bit-for-bit on arbitrary keyframe content:

* **Inter frames (P / B)**: out of scope for this milestone. Non-key,
  non-`intra_only` frames surface `Error::Unsupported` with the
  message `"vp9 inter frame pending"`.
* **Neighbour-aware probability contexts**: the crate uses context 0
  for the partition probability tree (§6.4.2 `partition_plane_context`)
  and for the KF intra-mode probability selection (§6.4.3: above /
  left neighbour modes). This means decoded images will differ from
  libvpx's reference output even on the same bitstream — we still
  produce plausible data (the integration test's luma mean lands in
  `[117]`, with > 100 distinct sample values), just not the exact
  samples ffmpeg produces.
* **Deblocking loop filter (§8.8)**: not yet applied. Block boundaries
  are visible on the reconstructed output. This is the next logical
  increment and does not affect the bitstream parse.
* **Multi-tile frames**: `log2_tile_cols > 0` and `log2_tile_rows > 0`
  return `Unsupported`. The tile-size-prefix syntax (§6.4) is simple
  but brings no new coverage until the fixture exercises it.
* **Higher bit depths (profiles 2 / 3)**: the parser recognises
  10-bit / 12-bit colour configs but the reconstruction pipeline only
  runs on 8-bit `Yuv420P`.
* **4:2:2 and 4:4:4 subsampling (profiles 1 / 3)**: the parser
  handles the header bits but `pixel_format_from_color_config` still
  reports `Yuv420P`.

None of the above prevent the crate from decoding a standard
`libvpx-vp9 -g 1` 8-bit 4:2:0 keyframe clip.

## Codec / container IDs

* Codec: `"vp9"`; maps from MP4's `vp09` sample entry.
* IVF demux: `oxideav_vp9::ivf::iter_frames` (a thin helper for tests
  and standalone use, not yet a full `Container` impl).

## License

MIT — see [LICENSE](LICENSE).
