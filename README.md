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

## Round-22 §6.4.3 HORZ/VERT sub-8×8 fix + §9.3.2 spec-literal sub-mode anchor

`tests/vp9_lossless_pattern.rs` compares against an `ffmpeg testsrc
-lossless 1` reference. Headline numbers per round:

| round | lossless pattern Y | lossless c64 Y | lossless gray Y | compound Y mean |
|-------|-------------------:|---------------:|----------------:|----------------:|
| r17   | 9.69 dB            | n/a            | 66.77 dB        | 10.63 dB        |
| r18   | 9.90 dB            | n/a            | 66.77 dB        | 10.72 dB        |
| r19   | 9.90 dB            | 61.90 dB       | 66.77 dB        | 10.72 dB        |
| r20   | 9.67 dB            | 70.10 dB       | 45.43 dB        | 10.20 dB        |
| r21   | 10.41 dB           | ∞ (bit-exact) | 45.43 dB        | 9.28 dB         |
| r22   | **47.70 dB**       | **∞ (bit-exact)** | 45.43 dB     | **9.54 dB**     |

Round 22 lands two paired §6.4.3 / §9.3.2 fixes that, together,
unlock a **+37 dB jump** on the lossless-pattern Y plane and lift
both chroma planes to **bit-exact**.

### `decode_partition` HORZ/VERT one-call branch (`block.rs`)

`§6.4.3 decode_partition`:

```text
if ( subsize < BLOCK_8X8 || partition == PARTITION_NONE )
    decode_block( r, c, subsize )
else if ( partition == PARTITION_HORZ ) {
    decode_block( r, c, subsize )
    if ( hasRows ) decode_block( r + halfBlock8x8, c, subsize )
} ...
```

The leading branch fires **first** when `subsize < BLOCK_8X8`. For
`bsize=BLOCK_8X8 + PARTITION_HORZ → subsize=BLOCK_8X4`, the
sub-8×8 branch wins and `decode_block` is called **once** —
`§6.4.6 read_intra_frame_mode_info` then reads the 4 sub-modes
internally. The same applies to `BLOCK_8X8 + PARTITION_VERT →
BLOCK_4X8`.

Our intra `decode_partition` was unconditionally calling
`decode_block` twice for HORZ/VERT, double-reading mode-info and
desynchronising the bool decoder for every sub-8×8 partition. The
fix gates the second call on `bsize > 8`. Same shape as the
round-13 SPLIT fix.

### `read_intra_sub_mode` spec-literal neighbour anchor (`block.rs`)

§9.3.2 default_intra_mode for `MiSize < BLOCK_8X8`:

```text
if (idy) abovemode = sub_modes[idx]
else     abovemode = AvailU ? SubModes[MiRow-1][MiCol][2 + idx] : DC
if (idx) leftmode  = sub_modes[idy*2]
else     leftmode  = AvailL ? SubModes[MiRow][MiCol-1][1 + idy*2] : DC
```

In the §9.3.2 NOTE "two 1D arrays" layout (which our decoder uses):

* `above_mode_4x4[mi_col*2 + 0] = sub_modes[2]` (writer for cell
  above, position `(idy=1, idx=0)`)
* `above_mode_4x4[mi_col*2 + 1] = sub_modes[3]` (`(idy=1, idx=1)`)
* `left_mode_4x4 [mi_row*2 + 0] = sub_modes[1]` (`(idy=0, idx=1)`)
* `left_mode_4x4 [mi_row*2 + 1] = sub_modes[3]` (`(idy=1, idx=1)`)

The reader index that picks the spec-literal slot is therefore
`above_mode_4x4[mi_col*2 + idx]` and
`left_mode_4x4 [mi_row*2 + idy]`.

The round-15 code used a constant `+1` on both sides (always
sub_modes[3], the bottom-right). Rounds 18-21 measured the
spec-literal switch and saw a 1 dB compound regression — but that
regression was an artefact of the upstream HORZ/VERT double-call
(see above). Once both fixes land together, the spec-literal
anchor is uniformly better.

### Per-fixture / per-variant audit (round 22)

Pre-fix baseline = r21 anchor (both `+1`):

| variant                     | pattern Y | compound Y | intra fixture mean | c64 |
|-----------------------------|----------:|-----------:|--------------------:|-----|
| r21 (both `+1`)             | 10.41 dB  | 9.28 dB    | 89                  | ∞ |
| spec-A0+L0 only             |  9.94 dB  | 10.79 dB   | **6** (FAIL)        | ∞ |
| HORZ/VERT fix only          | (not measured separately — pairs with anchor) ||||
| **r22 (both fixes paired)** | **47.70 dB** | **9.54 dB** | **111**          | **∞** |

The "spec-A0+L0 only" row is the lesson: the two §6.4.3 / §9.3.2
divergences were masking each other. Spec-correct on one but not
the other produces a desync (`vp9_intra_fixture` luma mean
collapses from 89 to 6). Spec-correct on both lifts pattern Y from
9.94 → 47.70 dB and intra fixture mean from 89 → 111.

The 0.74 dB compound regression vs. r21 (10.20 → 9.54) is the
remaining `inter.rs` divergence — the inter `decode_partition` has
the same HORZ/VERT double-call shape and a 4-call SPLIT loop at
`bsize=8` that still need a parallel audit. Handed to r23+.

`vp9-lossless-pattern.ivf` is now within **337 of 16384 luma
bytes** of bit-exact (98% of luma matches; both chroma planes
match 100%).

## Round-20 §7.4.6 spec-ctx skip read

Round 20 fixed the §6.4.8 / §7.4.6 skip-context wiring. Since round
13 the decoder had been reading `skip` against the constant
`skip_probs[0]=192`, ignoring the spec ctx
`(AvailU ? AboveSkip : 0) + (AvailL ? LeftSkip : 0)`. The
`vp9-lossless-c64-constant.ivf` fixture pinpointed the bug: a 16×8
H_PRED block at (mi_row=1, mi_col=2) where the libvpx encoder had
emitted `skip=true` against `skip_probs[1]=128` (left=0,above=1) but
our decoder kept reading against `skip_probs[0]=192`, decoding
skip=false and consuming 12 spurious 4×4 token reads which then
ran the bool decoder past EOF and corrupted the rest of the frame.

The fix is one line in `block.rs::decode_block` (intra path) and a
matching line in `inter.rs` — both now pass `skip_ctx(mi_row,
mi_col)` into `skip_probs[]`. The encoder side in
`encoder/tile.rs::emit_block` was updated to use the same per-context
prob so the self-roundtrip stays bit-exact.

Effect:
* `vp9-lossless-c64-constant.ivf` lifts **61.9 → 70.1 dB**, byte
  diffs collapse from 29 → 20 (all chroma is now bit-exact).
* `vp9-lossless-gray.ivf` drops 66.77 → 45.43 dB. The gray fixture
  is a degenerate `vec![126; 64*64]` reference: a single
  bool-decoder mis-skip used to score 66.77 dB by chance because
  every block's DC prediction matched 126; the spec-correct skip
  read now exposes the underlying compressed-header / coef-prob
  drift that the bool-decoder happy-accident was masking. Test
  threshold is 30 dB; we still pass.
* `vp9-lossless-pattern.ivf` Y drops 9.90 → 9.67 dB (still > 8 dB
  threshold). The pattern fixture has high-frequency content that
  exercises every sub-8x8 mode-info path; the round-20 fix is one
  of several stacked spec divergences and lifting pattern Y to
  ≥30 dB is r21+.
* compound mean luma 10.49 → 10.20 dB (test prints only, no
  threshold; the inter path also moved to spec ctx for parity).

Audited but ruled in/out:
* WHT round-trip (§8.7.1.10): bit-correct on DC=16, DC=-1792 (the
  exact first-block scenario), and a non-DC AC1 sanity case.
* DC_PRED at frame top-left: bit-exact reconstruction of pixel value
  16 from predictor 128 with WHT(-1792) on the first 16×16 block of
  every fixture (lossless-pattern, c64, gray).
* (Note: the r19 audit ruled `update_partition_ctx` spec-literal
  out — that ruling was overturned in r21 once the r20 skip-ctx
  fix removed the upstream bool-decoder drift that was tainting
  the r19 measurement.)

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

## Encoder (experimental — keyframe MVP)

The `encoder` module produces a valid VP9 keyframe bitstream accepted
by ffmpeg / libvpx and round-trippable through this crate's decoder.

Scope today:
* Profile 0, 4:2:0 8-bit, single tile, loop-filter disabled.
* Emits the full §6.2 uncompressed header, §6.3 compressed header
  (tx_mode, coef_probs-update flags, skip_prob-update flags) and the
  tile / partition / block symbols per §6.4.
* Per-block strategy: `PARTITION_NONE` at 64×64 (SPLIT at edges for
  non-multiple-of-64 frames), `skip=1` (no coefficient residual),
  `DC_PRED` luma + chroma intra modes.
* Partition-context and intra-mode-context trackers mirror the
  decoder's §7.4.6 state exactly so second-and-later blocks resolve
  to the correct probability row.
* `BoolEncoder` is the inverse of `BoolDecoder` — standard binary
  range coder with pending-byte carry propagation. Roundtrip-tested
  against the decoder with mixed / skewed / equal / carry-forcing
  probs and 2048-symbol PRNG streams.

Self-roundtrip: the produced bitstream decodes through
`Vp9Decoder::send_packet` → `receive_frame` to a uniform 128 (midgrey)
4:2:0 frame.

ffmpeg-acceptance: `tests/vp9_encoder_ffmpeg.rs` decodes the frame
with the system `ffmpeg` binary and confirms zero decode errors.

Deferred to follow-up work:
* Forward 4×4/8×8 DCT + ADST (inverse of the decoder's §8.7.1).
* Forward quantisation at a fixed QP.
* Token encoding against the §10.5 coef tree probabilities.
* Per-block skip-context / is_inter / reference-mode / intra-mode
  decision making based on RD / SAD of the source YUV.

Entry points: `encoder::encode_keyframe(&EncoderParams)` yields the
final byte buffer ready for IVF-muxing or `Vp9Decoder::send_packet`.

## Codec / container IDs

* Codec: `"vp9"`; maps from MP4's `vp09` sample entry.
* IVF demux: `oxideav_vp9::ivf::iter_frames` (a thin helper for tests
  and standalone use, not yet a full `Container` impl).

## License

MIT — see [LICENSE](LICENSE).
