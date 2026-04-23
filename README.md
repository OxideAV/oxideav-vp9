# oxideav-vp9

Pure-Rust **VP9** video decoder. Parses the full uncompressed header
(§6.2), the compressed header (§6.3), walks the tile / superblock
partition quadtree (§6.4.2) and reconstructs keyframe / intra-only
**and** inter frames (single and compound reference, scaled
references) across one or more tiles, down to pixels via intra
prediction, 8-tap sub-pel motion compensation, inverse transform and
dequantised-coefficient add. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Decode support today

* **Keyframe / intra-only frames** (`FrameType::Key` and the
  `intra_only` mid-stream variant) decode to a `VideoFrame` with
  3-plane 8-bit 4:2:0 luma + chroma data.
* **Inter (P) frames — single and compound reference**. The decoder
  keeps the VP9 8-slot DPB (§6.2, `refresh_frame_flags` semantics),
  reads every block's `is_inter` / `ref_frame` / `inter_mode` / MV and
  runs §8.5.2 motion compensation. For compound blocks (§6.4.17 with
  `reference_mode` ∈ {COMPOUND, SELECT}) both references are MC'd
  independently, then blended per §8.5.2 with `Round2(a + b, 1)`.
* **Scaled references (§8.5.2.3)**. When a reference frame's size
  differs from the current frame, the MV and sub-pel interpolator pick
  up `x_step_q4 = (16 * (RefW << 14) / CurW) >> 14` (same for y).
* **Multi-tile frames (§6.4)**. Tile-column and tile-row partitioning
  is supported. The tile payload is split at 4-byte big-endian length
  prefixes (last tile consumes the remainder), the boolean engine is
  reset per tile, and §6.4.1 `get_tile_offset` yields each tile's
  pixel bounds. The §8.8 loop filter runs once after all tiles are
  decoded.
* **Segmentation deltas (§8.6.1 / §8.8.1)**: `SEG_LVL_ALT_Q` overrides
  the block quantiser; `SEG_LVL_ALT_L` overrides the per-segment
  loop-filter level. Both respect `abs_delta` vs delta mode. The
  per-block segmentation-map read is still scaffold — every block
  currently reports `segment_id = 0`.
* **All 10 intra modes** (§8.5.1): `DC_PRED`, `V_PRED`, `H_PRED`,
  `D45_PRED`, `D135_PRED`, `D117_PRED`, `D153_PRED`, `D207_PRED`,
  `D63_PRED`, `TM_PRED`. Block sides 4 / 8 / 16 / 32 supported. 127 /
  129 neighbour padding handled per libvpx `build_intra_predictors`.
* **All four inverse-transform sizes** (§8.7.1): 4×4, 8×8, 16×16, 32×32.
  The four 2-D tx-types `DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
  `ADST_ADST` are covered for 4 / 8 / 16; 32×32 is DCT-only per spec.
  The lossless 4×4 Walsh-Hadamard is wired via `TxType::WhtWht`.
* **Sub-pel interpolation** (§8.5.1): full 16-phase 8-tap filter
  banks for `EIGHTTAP`, `EIGHTTAP_SMOOTH`, `EIGHTTAP_SHARP`, plus
  `BILINEAR`. Per-frame filter or per-block `SWITCHABLE` both
  supported. 1/8-pel luma, 1/16-pel chroma.
* **MV decode** (§6.4.19): full `mv_joint` + per-component
  `sign / class / class0_* / bits / fr / hp` tree with
  `allow_high_precision_mv` gating the extra bit. `NEARESTMV` /
  `NEARMV` / `ZEROMV` / `NEWMV` modes supported (see "Deferred"
  below for the MV-candidate caveat).
* **Coefficient decode** (§6.4.23 / §8.5.2 / §10.5): full token tree
  (EOB / ZERO / ONE model nodes + Pareto8 tail for TWO/THREE/FOUR/
  CAT1..6), sign bit, per-position context via neighbour magnitudes,
  dequantisation against `DC_QLOOKUP` / `AC_QLOOKUP` and the 32×32
  `dq_shift`.
* **Loop filter / deblocking** (§8.8): after reconstruction the tile
  walker applies the §8.8 deblocking pass — §8.8.1 `LvlLookup` with
  ref / mode deltas, §8.8.2 raster walk in 8×8-MI units over both
  passes and all three planes, §8.8.3 filter-size clamp, §8.8.4
  adaptive filter strength (limit / blimit / thresh) and §8.8.5
  sample filter (narrow `filter4`, wide `filter8` / `filter16` with
  `flat_mask` / `flat_mask2`). Segmentation deltas and multi-tile
  coordination are still pending. 8-bit 4:2:0 only.
* **IVF demux** (`ivf::iter_frames`) lets tests feed ffmpeg-generated
  clips directly.

### Example

```rust
use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{ivf, make_decoder, CODEC_ID_STR};

let data = std::fs::read("clip.ivf")?;
let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
let mut dec = make_decoder(&params)?;
for (pts, frame) in ivf::iter_frames(&data)?.enumerate() {
    let frame = frame?;
    let pkt = Packet::new(pts as i64, TimeBase::new(1, 24), frame.payload.to_vec());
    dec.send_packet(&pkt)?;
    let Frame::Video(v) = dec.receive_frame()? else { unreachable!() };
    println!("frame {pts}: {}x{} {:?}", v.width, v.height, v.format);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The shipped integration tests live at `tests/vp9_intra_fixture.rs`
(128×128 keyframe, luma-statistics gate) and
`tests/vp9_inter_fixture.rs` (1 key + 1 P frame, asserts the P frame
decodes into `Frame::Video` and differs from the keyframe at pixel
level).

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

Both the intra and inter paths ship, but a handful of accuracy
refinements remain before the decoder matches a libvpx reference
bit-for-bit:

* **MV-candidate list (`find_mv_refs` / `find_best_ref_mvs`)**:
  `NEARESTMV` and `NEARMV` currently resolve to a zero predictor
  instead of the full neighbour-derived candidate list (§6.4.17).
  `ZEROMV` and `NEWMV` are correct; `NEARESTMV`/`NEARMV` degrade to
  the ZEROMV spatial result.
* **Per-block segmentation map**: `SEG_LVL_ALT_Q` / `SEG_LVL_ALT_L`
  deltas are applied, but the segmentation-map tree decode + temporal
  predicted-segment lookup is not wired — every block falls through
  to segment 0. `SEG_LVL_REF_FRAME` / `SEG_LVL_SKIP` are not applied.
* **Neighbour-aware probability contexts**: the crate uses context 0
  for the partition tree (§6.4.2 `partition_plane_context`), for the
  KF intra-mode probability selection, for `skip_prob`, `is_inter`,
  `comp_mode` and `comp_ref`. Output diverges from the libvpx
  reference but stays plausible.
* **Higher bit depths (profiles 2 / 3)**: the parser recognises
  10-bit / 12-bit colour configs but the reconstruction pipeline only
  runs on 8-bit `Yuv420P`.
* **4:2:2 and 4:4:4 subsampling (profiles 1 / 3)**: the parser
  handles the header bits but the output format stays `Yuv420P`.
* **`B` / reordered frames**: VP9 can emit no-show altref frames
  (`show_frame=0`). These flow through the DPB correctly but are not
  emitted as `Frame::Video` until a later `show_existing_frame`
  references them.

None of the above prevent the crate from decoding a standard
`libvpx-vp9 -g N` 8-bit 4:2:0 IPPP stream into frames the caller can
render.

## Codec / container IDs

* Codec: `"vp9"`; maps from MP4's `vp09` sample entry.
* IVF demux: `oxideav_vp9::ivf::iter_frames` (a thin helper for tests
  and standalone use, not yet a full `Container` impl).

## License

MIT — see [LICENSE](LICENSE).
